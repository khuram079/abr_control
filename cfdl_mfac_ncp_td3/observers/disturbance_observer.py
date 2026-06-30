"""Nonlinear Disturbance Observer (NDOB) for the AUV velocity dynamics.

For the body-frame dynamics ``M nu_dot = tau - C(nu)nu - D(nu)nu - g(eta) + d``
the lumped disturbance ``d`` (unmodelled hydrodynamics, currents, actuator
faults projected onto the wrench, ...) is estimated *without differentiating
the velocity* using Chen's nonlinear disturbance observer:

    d_hat = z + L M nu
    z_dot = -L z - L ( L M nu + tau - C(nu)nu - D(nu)nu - g(eta) )

which gives the linear error dynamics ``d_tilde_dot = -L d_tilde`` (for slowly
varying ``d``), so the estimate converges with time-constant ``1/L``.

Reference: W.-H. Chen et al., "A nonlinear disturbance observer for robotic
manipulators", IEEE T-IE, 2000.
"""

from __future__ import annotations

import numpy as np

from ..config import ObserverConfig
from ..dynamics.hydrodynamics import REMUSParams


class DisturbanceObserver:
    """Chen-type nonlinear disturbance observer (6-DOF wrench estimate)."""

    def __init__(
        self,
        params: REMUSParams | None = None,
        config: ObserverConfig | None = None,
        gain: float | None = None,
    ):
        self.p = params or REMUSParams()
        self.cfg = config or ObserverConfig()
        self.M = self.p.mass_matrix()
        L = self.cfg.ndob_gain if gain is None else gain
        self.L = L * np.eye(6)  # observer gain (scalar * I)
        self.reset()

    def reset(self) -> None:
        self._z = np.zeros(6)
        self.d_hat = np.zeros(6)

    def update(
        self, eta: np.ndarray, nu: np.ndarray, tau: np.ndarray, dt: float
    ) -> np.ndarray:
        """Advance the observer one step and return the disturbance estimate."""

        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        tau = np.asarray(tau, dtype=float).reshape(6)

        C = self.p.coriolis(nu, self.M)
        D = self.p.damping(nu)
        g = self.p.restoring(eta)
        # Known part of M nu_dot (everything except the disturbance d).
        known = tau - C @ nu - D @ nu - g

        p_nu = self.L @ (self.M @ nu)
        z_dot = -self.L @ self._z - self.L @ (p_nu + known)
        self._z = self._z + dt * z_dot
        self.d_hat = self._z + p_nu
        return self.d_hat
