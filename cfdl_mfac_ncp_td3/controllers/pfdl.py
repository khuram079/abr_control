"""Partial-Form Dynamic Linearization Model-Free Adaptive Controller (PFDL-MFAC).

Where CFDL (compact form) linearizes the plant with a *single* pseudo-gradient
on the current control increment,

    Delta y(k+1) = phi(k) Delta u(k),

PFDL (partial form) uses a **sliding window of the L most recent control
increments** and a pseudo-gradient *vector*:

    Delta y(k+1) = phi(k)^T Delta U_L(k),
    Delta U_L(k) = [Delta u(k), Delta u(k-1), ..., Delta u(k-L+1)]^T,
    phi(k)       = [phi_1(k), ..., phi_L(k)]^T.

Minimising ``J = (y*(k+1) - y(k+1))^2 + lambda Delta u(k)^2`` under this model
gives the control law

    Delta u(k) = rho phi_1(k) [ (y* - y(k)) - S ] / (lambda + phi_1(k)^2),
    S          = sum_{i=2..L} phi_i(k) Delta u(k-i+1),

i.e. the new increment is computed *after subtracting the predicted effect of
the recent increments still propagating through the plant* (the term ``S``).
This "memory" curbs the over-commanding that makes the compact-form integrator
limit-cycle on the surge velocity loop (see results/ENERGY_ANALYSIS.md); for
``L = 1`` the law reduces exactly to CFDL-MFAC.

This implementation is SISO (the AUV hybrid uses a per-DOF bank), matching the
``CFDLMFAC(1, 1, ...)`` public interface (``control(y, y_ref)`` / ``reset`` /
``gain``) so it is a drop-in alternative.

Reference: Z. Hou and S. Jin, *Model Free Adaptive Control: Theory and
Applications*, CRC Press, 2013 (Chapter on Partial-Form Dynamic Linearization).
"""

from __future__ import annotations

import numpy as np

from ..config import MFACConfig


class PFDLMFAC:
    """SISO partial-form MFAC controller (drop-in for ``CFDLMFAC(1, 1, ...)``)."""

    def __init__(self, n_out: int = 1, n_in: int = 1,
                 config: MFACConfig | None = None,
                 u_bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 L: int | None = None):
        if int(n_out) != 1 or int(n_in) != 1:
            raise ValueError("PFDLMFAC is SISO (n_out == n_in == 1)")
        self.cfg = config or MFACConfig()
        self.L = int(L if L is not None else self.cfg.pfdl_L)
        self.u_bounds = u_bounds
        self.reset()

    # ------------------------------------------------------------------ #
    def _phi_init_vec(self) -> np.ndarray:
        phi = np.zeros(self.L)
        phi[0] = self.cfg.phi_init  # leading (compact-form) gain
        return phi

    def reset(self) -> None:
        self._phi = self._phi_init_vec()
        self._phi0 = self._phi_init_vec()
        self._du_hist = np.zeros(self.L)  # [du(k-1), du(k-2), ..., du(k-L)]
        self._u = 0.0
        self._y_prev = None
        self._initialised = False

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    @property
    def gain(self) -> float:
        """Leading pseudo-gradient phi_1 (the CFDL-equivalent d(dy)/d(du))."""

        return float(self._phi[0])

    # ------------------------------------------------------------------ #
    def _update_estimator(self, dy: float) -> None:
        dU = self._du_hist  # Delta U_L(k-1)
        denom = self.cfg.mu + float(dU @ dU)
        innovation = dy - float(self._phi @ dU)
        self._phi = self._phi + self.cfg.eta * innovation * dU / denom
        # Reset conditions (PFDL analogue of the CFDL projection resets).
        eps = self.cfg.epsilon
        if (np.linalg.norm(self._phi) <= eps or np.linalg.norm(dU) <= eps
                or np.sign(self._phi[0]) != np.sign(self._phi0[0])):
            self._phi = self._phi0.copy()

    def control(self, y, y_ref) -> np.ndarray:
        """Compute the scalar control ``u`` for measurement ``y`` / reference ``y_ref``."""

        y = float(np.asarray(y).reshape(-1)[0])
        y_ref = float(np.asarray(y_ref).reshape(-1)[0])

        # 1) Estimator update using Delta y(k) and the past-increment window.
        if self._initialised:
            self._update_estimator(y - self._y_prev)
        self._y_prev = y
        self._initialised = True

        phi = self._phi
        phi1 = phi[0]
        # 2) S = sum_{i=2..L} phi_i Delta u(k-i+1); du_hist[j] = Delta u(k-1-j).
        S = float(phi[1:] @ self._du_hist[: self.L - 1]) if self.L > 1 else 0.0
        err = y_ref - y
        denom = self.cfg.lam + phi1 * phi1
        du = self.cfg.rho * phi1 * (err - S) / denom

        # 3) Increment clip + commit + absolute saturation.
        du = float(np.clip(du, -self.cfg.u_limit, self.cfg.u_limit))
        u = self._u + du
        if self.u_bounds is not None:
            low, high = self.u_bounds
            u = float(np.clip(u, float(np.asarray(low).reshape(-1)[0]),
                              float(np.asarray(high).reshape(-1)[0])))
        du_actual = u - self._u
        self._u = u

        # 4) Shift the increment window (most-recent first).
        if self.L > 1:
            self._du_hist = np.concatenate([[du_actual], self._du_hist[:-1]])
        else:
            self._du_hist = np.array([du_actual])
        return np.array([u])
