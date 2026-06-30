"""Gymnasium environment for 6-DOF AUV trajectory tracking.

The environment wraps the REMUS plant, ocean current and actuator model into a
standard RL interface:

* **observation** ``[pose_error(6), nu(6), eta_d_dot(6)]`` (18-D),
* **action** normalised wrench in ``[-1, 1]^6`` scaled by the actuator limits,
* **reward** ``-(w_e ||pose_error||^2 + w_u ||a||^2)`` plus an alive bonus, with
  early termination on divergence.

It supports **domain randomization** (mass, current, sensor noise),
**curriculum learning** (difficulty in ``[0, 1]`` scales disturbances and
trajectory aggressiveness) and **actuator-fault injection**, all of which are
essential for training a robust policy and for the Monte-Carlo evaluation.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _BASE = object

from ..config import ExperimentConfig, default_config
from ..dynamics import REMUS6DOF, REMUSParams, OceanCurrent, ThrusterModel
from ..dynamics.remus6dof import rotation_matrix
from ..trajectories import make_trajectory
from ..benchmark.base import pose_error


class AUVEnv(_BASE):
    """6-DOF AUV tracking environment with randomization and curriculum."""

    metadata = {"render_modes": []}

    def __init__(self, config: ExperimentConfig | None = None,
                 trajectory: str = "sinusoidal", difficulty: float = 0.5,
                 randomize: bool = True, fault_prob: float = 0.0, seed: int = 0):
        self.cfg = config or default_config()
        self.trajectory_name = trajectory
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.randomize = randomize
        self.fault_prob = float(fault_prob)
        self.rng = np.random.default_rng(seed)

        self.dt = self.cfg.sim.dt
        self.tau_max = np.asarray(self.cfg.thruster.tau_max, dtype=float)
        self.w_e, self.w_u = 1.0, 0.01

        self.obs_dim = 18
        self.act_dim = 6
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)

        self._build()
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        self.params = REMUSParams()
        self.vehicle = REMUS6DOF(self.params, self.cfg.sim)
        self.current = OceanCurrent(self.cfg.current, rng=self.rng)
        self.thruster = ThrusterModel(self.cfg.thruster)

    def set_difficulty(self, d: float) -> None:
        """Curriculum hook: set task difficulty in [0, 1]."""

        self.difficulty = float(np.clip(d, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.current.rng = self.rng

        d = self.difficulty
        # Domain randomization scaled by curriculum difficulty.
        if self.randomize:
            mass_scale = 1.0 + self.rng.uniform(-0.15, 0.15) * d
            self.params = REMUSParams(mass=30.48 * mass_scale)
            self.vehicle = REMUS6DOF(self.params, self.cfg.sim)
            mean_cur = self.rng.uniform(-0.4, 0.4, size=3) * d
            self.current.reset(mean_velocity=mean_cur)
            self.meas_noise = 0.01 * d
        else:
            self.vehicle = REMUS6DOF(self.params, self.cfg.sim)
            self.current.reset()
            self.meas_noise = 0.0

        self.thruster.reset()
        # Actuator-fault injection.
        if self.rng.random() < self.fault_prob:
            ch = int(self.rng.integers(6))
            self.thruster.set_fault(ch, self.rng.uniform(0.2, 0.7))

        self.vehicle.reset()
        self.traj = make_trajectory(self.trajectory_name)
        self.t = 0.0
        self.max_t = min(self.cfg.sim.horizon, self.traj.duration)
        self.steps = 0
        obs = self._observe()
        return obs, {}

    # ------------------------------------------------------------------ #
    def _observe(self) -> np.ndarray:
        eta_d, eta_d_dot = self.traj.reference(self.t)
        eta_meas = self.vehicle.eta + self.rng.normal(0, self.meas_noise, 6)
        nu_meas = self.vehicle.nu + self.rng.normal(0, self.meas_noise, 6)
        err = pose_error(eta_d, eta_meas)
        return np.concatenate([err, nu_meas, eta_d_dot]).astype(np.float32)

    def _current_body(self) -> np.ndarray:
        R = rotation_matrix(*self.vehicle.eta[3:])
        return self.current.body_velocity(R.T, self.dt)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float).reshape(6), -1.0, 1.0)
        tau_cmd = action * self.tau_max
        tau = self.thruster.step(tau_cmd, self.dt)
        nu_c = self._current_body()
        self.vehicle.step(tau, nu_c=nu_c)
        self.t += self.dt
        self.steps += 1

        eta_d, _ = self.traj.reference(self.t)
        err = pose_error(eta_d, self.vehicle.eta)
        err_norm = float(np.linalg.norm(err))
        reward = -(self.w_e * err_norm ** 2 + self.w_u * float(action @ action)) + 0.5

        terminated = err_norm > 25.0 or not np.all(np.isfinite(self.vehicle.state))
        truncated = self.t >= self.max_t
        obs = self._observe()
        info = {"error_norm": err_norm, "tau": tau, "eta_d": eta_d}
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ #
    @property
    def eta(self) -> np.ndarray:
        return self.vehicle.eta

    @property
    def nu(self) -> np.ndarray:
        return self.vehicle.nu
