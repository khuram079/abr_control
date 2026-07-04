"""Single-vehicle robustness simulator (mirrors the formation study).

One REMUS vehicle tracks a reference trajectory under the same randomized
conditions used for the formation campaign -- random initial position, ocean
current, per-step sensor noise and an actuator fault injected at a random time
-- with threshold-triggered intelligent recovery (an outer-gain boost for
controllers that advertise ``supports_recovery``).  Returns the data needed for
the five single-vehicle indicators.
"""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig, default_config
from ..dynamics import REMUS6DOF, REMUSParams, OceanCurrent, ThrusterModel
from ..dynamics.remus6dof import rotation_matrix
from ..trajectories import make_trajectory
from ..benchmark.base import pose_error

SV_INDICATORS = ("tracking_rmse", "attitude_rmse", "recovery_time",
                 "control_energy", "max_deviation")


def sv_indicators(res: dict) -> dict:
    te = np.asarray(res["track_err"])
    ae = np.asarray(res["att_err"])
    return {
        "tracking_rmse": float(np.sqrt(np.mean(te ** 2))),
        "attitude_rmse": float(np.sqrt(np.mean(ae ** 2))),
        "recovery_time": float(res["recovery_time"]),
        "control_energy": float(res["energy"]),
        "max_deviation": float(np.max(te)),
    }


class SingleVehicleSimulator:
    """Simulate one randomized single-vehicle tracking episode."""

    def __init__(self, config: ExperimentConfig | None = None, trajectory: str = "sinusoidal",
                 recovery_threshold: float = 1.5, recovery_exit_ratio: float = 0.5,
                 recovery_boost: float = 2.0, prediction_gain: float = 2.5,
                 fault_prob: float = 0.6, fault_time_frac: float = 0.4):
        self.cfg = config or default_config()
        self.trajectory_name = trajectory
        self.dt = self.cfg.sim.dt
        self.rec_thr = recovery_threshold
        self.rec_exit = recovery_exit_ratio
        self.rec_boost = recovery_boost
        self.pred_gain = prediction_gain
        self.fault_prob = fault_prob
        self.fault_time_frac = fault_time_frac

    def simulate(self, controller_factory, seed: int = 0, randomize: bool = True,
                 fault: bool = True, recovery_threshold: float | None = None,
                 prediction_gain: float | None = None) -> dict:
        rng = np.random.default_rng(seed)
        rec_thr = self.rec_thr if recovery_threshold is None else recovery_threshold
        pred_gain = self.pred_gain if prediction_gain is None else prediction_gain

        p = REMUSParams()
        if randomize:
            p = REMUSParams(mass=30.48 * (1.0 + rng.uniform(-0.2, 0.2)))
        veh = REMUS6DOF(p, self.cfg.sim)
        traj = make_trajectory(self.trajectory_name)
        eta_d0, _ = traj.reference(0.0)
        eta0 = eta_d0.copy()
        if randomize:
            eta0[:3] += rng.uniform(-1.5, 1.5, size=3)
            eta0[5] += rng.uniform(-0.3, 0.3)
        veh.reset(eta=eta0)

        ctrl = controller_factory()
        if hasattr(ctrl, "cfdl_feedforward"):
            ctrl.cfdl_feedforward = pred_gain
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        base_k1 = np.array(ctrl.k1).copy() if hasattr(ctrl, "k1") else None
        support = bool(getattr(ctrl, "supports_recovery", False))

        thr = ThrusterModel(self.cfg.thruster)
        cur = OceanCurrent(self.cfg.current, rng=rng)
        cur.reset(mean_velocity=rng.uniform(-0.4, 0.4, size=3) if randomize else (0.0, 0.0, 0.0))
        meas_noise = 0.02 if randomize else 0.0

        n_steps = int(min(self.cfg.sim.horizon, traj.duration) / self.dt)
        fault_step, fault_ch, fault_eff = -1, -1, 1.0
        if fault and rng.random() < self.fault_prob:
            fault_step = int(self.fault_time_frac * n_steps)
            fault_ch = int(rng.integers(6))
            fault_eff = float(rng.uniform(0.2, 0.5))

        track_err = np.zeros(n_steps)
        att_err = np.zeros(n_steps)
        energy = 0.0
        recovery_time = 0.0
        in_recovery = False

        for k in range(n_steps):
            t = k * self.dt
            eta_d, eta_d_dot = traj.reference(t)
            if k == fault_step:
                thr.set_fault(fault_ch, fault_eff)

            e = pose_error(eta_d, veh.eta)
            enorm = float(np.linalg.norm(e[:3]))
            track_err[k] = enorm
            att_err[k] = float(np.linalg.norm(e[3:]))

            # recovery-time metric (uniform for all controllers)
            if not in_recovery and enorm > rec_thr:
                in_recovery = True
            elif in_recovery and enorm < rec_thr * self.rec_exit:
                in_recovery = False
            if in_recovery:
                recovery_time += self.dt
            # recovery action (only for controllers that support it)
            if support and base_k1 is not None:
                ctrl.k1 = base_k1 * (self.rec_boost if in_recovery else 1.0)

            eta_meas = veh.eta + rng.normal(0, meas_noise, 6)
            nu_meas = veh.nu + rng.normal(0, meas_noise, 6)
            tau = ctrl.control(eta_meas, nu_meas, eta_d, eta_d_dot, self.dt)
            tau_app = thr.step(tau, self.dt)
            energy += float(tau_app @ tau_app) * self.dt
            R = rotation_matrix(*veh.eta[3:])
            nu_c = cur.body_velocity(R.T, self.dt)
            veh.step(tau_app, nu_c=nu_c)
            if not np.all(np.isfinite(veh.state)):
                track_err[k:] = enorm
                break

        return {"track_err": track_err, "att_err": att_err, "energy": energy,
                "recovery_time": recovery_time, "n_steps": n_steps, "dt": self.dt,
                "fault": (fault_ch, fault_eff, fault_step)}
