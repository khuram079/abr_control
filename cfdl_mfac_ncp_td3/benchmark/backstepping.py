"""Model-based backstepping benchmark controller.

A standard two-step backstepping design for the cascade
``eta_dot = J(eta) nu`` , ``M nu_dot = tau - C nu - D nu - g``:

* Step 1 - virtual velocity ``nu_d = J^{-1}(eta_d_dot + K1 z1)`` for the
  kinematic error ``z1 = eta_d - eta``.
* Step 2 - wrench ``tau = (C+D)nu + g + M(nu_d_dot + K2 z2) + J^T z1`` for the
  velocity error ``z2 = nu_d - nu``.

This yields ``V_dot = -z1^T K1 z1 - z2^T K2 z2 <= 0`` (the cross terms cancel),
so the closed loop is Lyapunov-stable by construction.  ``nu_d_dot`` is
obtained by finite differencing the virtual control.
"""

from __future__ import annotations

import numpy as np

from ..dynamics.hydrodynamics import REMUSParams
from ..dynamics.remus6dof import jacobian
from .base import BaseController, pose_error


def _safe_jacobian_inv(J: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(J)
    except np.linalg.LinAlgError:  # pragma: no cover - near gimbal lock
        return np.linalg.pinv(J)


class BacksteppingController(BaseController):
    """Lyapunov-based backstepping pose-tracking controller."""

    name = "Backstepping"

    def __init__(self, params: REMUSParams | None = None, k1=None, k2=None, tau_max=None):
        self.p = params or REMUSParams()
        self.M = self.p.mass_matrix()
        self.k1 = np.array(k1 if k1 is not None else [2.0] * 6, dtype=float)
        self.k2 = np.array(k2 if k2 is not None else [4.0] * 6, dtype=float)
        self.tau_max = None if tau_max is None else np.asarray(tau_max, dtype=float)
        self.reset()

    def reset(self) -> None:
        self._nu_d_prev = None

    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float = 0.05) -> np.ndarray:
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, dtype=float)

        J = jacobian(eta)
        Jinv = _safe_jacobian_inv(J)

        z1 = pose_error(eta_d, eta)
        nu_d = Jinv @ (eta_d_dot + self.k1 * z1)
        if self._nu_d_prev is None:
            nu_d_dot = np.zeros(6)
        else:
            nu_d_dot = (nu_d - self._nu_d_prev) / dt
        self._nu_d_prev = nu_d

        z2 = nu_d - nu
        C = self.p.coriolis(nu, self.M)
        D = self.p.damping(nu)
        g = self.p.restoring(eta)
        n = C @ nu + D @ nu + g
        tau = n + self.M @ (nu_d_dot + self.k2 * z2) + J.T @ z1
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau
