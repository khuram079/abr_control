"""Per-DOF PID benchmark controller (NED pose tracking)."""

from __future__ import annotations

import numpy as np

from ..dynamics.remus6dof import jacobian
from .base import BaseController, pose_error


class PIDController(BaseController):
    """Decoupled PID on the 6-DOF pose error with anti-windup clamping."""

    name = "PID"

    def __init__(self, kp=None, ki=None, kd=None, tau_max=None, i_limit: float = 5.0):
        self.kp = np.array(kp if kp is not None else [40, 40, 40, 20, 30, 30], dtype=float)
        self.ki = np.array(ki if ki is not None else [2, 2, 2, 1, 1, 1], dtype=float)
        self.kd = np.array(kd if kd is not None else [60, 60, 60, 20, 30, 30], dtype=float)
        self.tau_max = None if tau_max is None else np.asarray(tau_max, dtype=float)
        self.i_limit = float(i_limit)
        self.reset()

    def reset(self) -> None:
        self._integral = np.zeros(6)

    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float = 0.05) -> np.ndarray:
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, dtype=float)

        e = pose_error(eta_d, eta)
        e_dot = eta_d_dot - jacobian(eta) @ nu  # error rate in the NED frame

        self._integral = np.clip(self._integral + e * dt, -self.i_limit, self.i_limit)
        tau = self.kp * e + self.ki * self._integral + self.kd * e_dot
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau
