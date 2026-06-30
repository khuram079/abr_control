"""Extended Kalman Filter state observer for the 6-DOF AUV.

The observer fuses the (known) nonlinear plant model with noisy measurements
to produce a denoised estimate of the full 12-dimensional state
``x = [eta; nu]``.  It supports partial measurements through a measurement
matrix ``H`` (e.g. pose-only sensing, with body velocities reconstructed by
the filter).  The prediction Jacobian is obtained numerically, which keeps the
observer agnostic to the specific dynamics implementation.
"""

from __future__ import annotations

import numpy as np

from ..config import ObserverConfig, SimConfig
from ..dynamics.remus6dof import REMUS6DOF


def _wrap_angle_indices(vec: np.ndarray, idx) -> np.ndarray:
    out = vec.copy()
    out[idx] = (out[idx] + np.pi) % (2 * np.pi) - np.pi
    return out


class StateObserver:
    """EKF over ``x = [eta; nu]`` using the REMUS model for prediction."""

    ANGLE_IDX = (3, 4, 5)

    def __init__(
        self,
        model: REMUS6DOF | None = None,
        config: ObserverConfig | None = None,
        H: np.ndarray | None = None,
        dt: float | None = None,
    ):
        self.model = model or REMUS6DOF()
        self.cfg = config or ObserverConfig()
        self.dt = dt if dt is not None else self.model.sim.dt
        self.H = np.eye(12) if H is None else np.asarray(H, dtype=float)
        self.m = self.H.shape[0]
        self.Q = self.cfg.process_noise * np.eye(12)
        self.R = self.cfg.meas_noise * np.eye(self.m)
        self.reset()

    def reset(self, x0: np.ndarray | None = None) -> None:
        self.x = np.zeros(12) if x0 is None else np.asarray(x0, dtype=float).reshape(12)
        self.P = np.eye(12)

    # ------------------------------------------------------------------ #
    def _f_disc(self, x: np.ndarray, tau: np.ndarray, nu_c) -> np.ndarray:
        """One-step RK4 propagation of the model (pure function of ``x``)."""

        dt = self.dt

        def f(s):
            return self.model.deriv(s, tau, nu_c=nu_c)

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _jacobian(self, x: np.ndarray, tau: np.ndarray, nu_c, eps: float = 1e-6):
        n = x.size
        F = np.zeros((n, n))
        f0 = self._f_disc(x, tau, nu_c)
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps
            F[:, j] = (self._f_disc(x + dx, tau, nu_c) - f0) / eps
        return F, f0

    # ------------------------------------------------------------------ #
    def predict(self, tau: np.ndarray, nu_c=None) -> np.ndarray:
        F, f0 = self._jacobian(self.x, np.asarray(tau, dtype=float).reshape(6), nu_c)
        self.x = _wrap_angle_indices(f0, list(self.ANGLE_IDX))
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, y_meas: np.ndarray) -> np.ndarray:
        """Kalman measurement update; returns the corrected state estimate."""

        y_meas = np.asarray(y_meas, dtype=float).reshape(self.m)
        innovation = y_meas - self.H @ self.x
        # Wrap any measured angle innovations into (-pi, pi].
        for k, idx in enumerate(self.ANGLE_IDX):
            rows = np.where(np.isclose(self.H[:, idx], 1.0))[0]
            for r in rows:
                innovation[r] = (innovation[r] + np.pi) % (2 * np.pi) - np.pi
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = _wrap_angle_indices(self.x + K @ innovation, list(self.ANGLE_IDX))
        self.P = (np.eye(12) - K @ self.H) @ self.P
        return self.x

    def step(self, tau: np.ndarray, y_meas: np.ndarray, nu_c=None) -> np.ndarray:
        """Convenience predict+update cycle."""

        self.predict(tau, nu_c=nu_c)
        return self.update(y_meas)

    @property
    def eta(self) -> np.ndarray:
        return self.x[:6]

    @property
    def nu(self) -> np.ndarray:
        return self.x[6:]
