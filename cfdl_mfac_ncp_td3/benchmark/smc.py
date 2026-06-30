"""Sliding-mode benchmark controller with a boundary layer.

A sliding surface ``s = e_dot + Lambda e`` is defined on the NED pose error.
A robust reaching law (proportional term + saturated switching term) produces
a generalised force in the NED frame, mapped to the body-frame wrench through
``tau = J(eta)^T f_eta``.  The ``tanh`` boundary layer replaces the
discontinuous ``sign`` to suppress chattering.
"""

from __future__ import annotations

import numpy as np

from ..dynamics.remus6dof import jacobian
from .base import BaseController, pose_error


class SMCController(BaseController):
    """Boundary-layer sliding-mode pose-tracking controller."""

    name = "SMC"

    def __init__(self, lam=None, kd=None, ks=None, phi: float = 0.1, tau_max=None):
        self.lam = np.array(lam if lam is not None else [1.5] * 6, dtype=float)
        self.kd = np.array(kd if kd is not None else [50, 50, 50, 20, 30, 30], dtype=float)
        self.ks = np.array(ks if ks is not None else [20, 20, 20, 8, 12, 12], dtype=float)
        self.phi = float(phi)
        self.tau_max = None if tau_max is None else np.asarray(tau_max, dtype=float)

    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float = 0.05) -> np.ndarray:
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, dtype=float)

        J = jacobian(eta)
        e = pose_error(eta_d, eta)
        e_dot = eta_d_dot - J @ nu
        s = e_dot + self.lam * e

        # Reaching law in NED space: PD-on-surface + saturated switching.
        f_eta = self.kd * s + self.ks * np.tanh(s / self.phi)
        tau = J.T @ f_eta
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau
