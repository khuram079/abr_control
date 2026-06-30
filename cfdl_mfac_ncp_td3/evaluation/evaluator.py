"""Closed-loop rollout harness and controller evaluation.

Runs any controller exposing ``control(eta, nu, eta_d, eta_d_dot, dt)`` (the
benchmark controllers and the :class:`HybridController` all comply) around the
full nonlinear REMUS plant, with optional ocean current and actuator faults,
logging the trajectory and returning performance metrics.
"""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig, default_config
from ..dynamics import REMUS6DOF, REMUSParams, OceanCurrent, ThrusterModel
from ..dynamics.remus6dof import rotation_matrix
from ..trajectories import Trajectory, make_trajectory
from ..benchmark.base import pose_error
from . import metrics as M


def rollout(controller, trajectory: Trajectory | str = "sinusoidal",
            config: ExperimentConfig | None = None,
            current=False, fault=None, seed: int = 0,
            params: REMUSParams | None = None) -> dict:
    """Simulate one closed-loop episode and return a log + metrics.

    Parameters
    ----------
    controller:
        Object with ``.control(...)`` and (optionally) ``.reset()``.
    trajectory:
        A :class:`Trajectory` or registry name.
    current:
        If true, enable a mean ocean current disturbance.
    fault:
        Optional ``(channel, effectiveness)`` actuator fault tuple.
    """

    cfg = config or default_config()
    dt = cfg.sim.dt
    params = params or REMUSParams()
    veh = REMUS6DOF(params, cfg.sim)
    veh.reset()
    rng = np.random.default_rng(seed)
    cur = OceanCurrent(cfg.current, rng=rng) if current else None
    if cur is not None:
        cur.reset(mean_velocity=cfg.current.mean_velocity)
    thr = ThrusterModel(cfg.thruster)
    thr.reset()
    if fault is not None:
        thr.set_fault(int(fault[0]), float(fault[1]))

    traj = trajectory if isinstance(trajectory, Trajectory) else make_trajectory(trajectory)
    if hasattr(controller, "reset"):
        controller.reset()

    n = int(min(cfg.sim.horizon, traj.duration) / dt)
    log = {k: [] for k in ("t", "eta", "eta_d", "nu", "tau", "error", "alpha")}
    for k in range(n):
        t = k * dt
        eta_d, eta_d_dot = traj.reference(t)
        tau_cmd = controller.control(veh.eta, veh.nu, eta_d, eta_d_dot, dt)
        tau = thr.step(tau_cmd, dt)
        nu_c = None
        if cur is not None:
            R = rotation_matrix(*veh.eta[3:])
            nu_c = cur.body_velocity(R.T, dt)
        veh.step(tau, nu_c=nu_c)

        err = pose_error(eta_d, veh.eta)
        log["t"].append(t)
        log["eta"].append(veh.eta.copy())
        log["eta_d"].append(eta_d.copy())
        log["nu"].append(veh.nu.copy())
        log["tau"].append(tau.copy())
        log["error"].append(err)
        log["alpha"].append(getattr(controller, "last_info", {}).get("alpha", 0.0))
        if not np.all(np.isfinite(veh.state)):
            break

    for key in log:
        log[key] = np.asarray(log[key])
    metrics = M.summarize(log["error"], log["tau"], dt)
    metrics["diverged"] = bool(not np.all(np.isfinite(log["eta"])))
    return {"log": log, "metrics": metrics, "name": getattr(controller, "name", "controller")}


def evaluate_controllers(controllers: dict, **kwargs) -> dict:
    """Evaluate a ``{name: controller}`` dict on the same task."""

    return {name: rollout(ctrl, **kwargs) for name, ctrl in controllers.items()}
