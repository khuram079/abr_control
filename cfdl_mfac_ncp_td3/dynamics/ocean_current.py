"""Ocean-current disturbance model.

The current is modelled as a (slowly varying) mean flow expressed in the NED
earth frame plus a first-order Gauss-Markov turbulence component.  When the
vehicle dynamics are written in terms of the *relative* velocity
``nu_r = nu - nu_c`` this captures the dominant effect of a quasi-irrotational
current on the hydrodynamic forces.
"""

from __future__ import annotations

import numpy as np

from ..config import CurrentConfig


class OceanCurrent:
    """Mean + Gauss-Markov turbulent ocean current.

    Parameters
    ----------
    config:
        :class:`~cfdl_mfac_ncp_td3.config.CurrentConfig` instance.
    rng:
        Optional NumPy random generator for reproducible turbulence.
    """

    def __init__(self, config: CurrentConfig | None = None, rng=None):
        self.cfg = config or CurrentConfig()
        self.rng = rng or np.random.default_rng()
        self._mean = np.asarray(self.cfg.mean_velocity, dtype=float)
        self._turb = np.zeros(3)

    def reset(self, mean_velocity=None) -> None:
        """Reset the turbulence state and optionally override the mean flow."""

        if mean_velocity is not None:
            self._mean = np.asarray(mean_velocity, dtype=float)
        self._turb = np.zeros(3)

    def velocity_ned(self, dt: float) -> np.ndarray:
        """Advance the turbulence one step and return the NED current [m/s]."""

        if not self.cfg.enabled:
            return np.zeros(3)
        # First-order Gauss-Markov update: tau dx = -x dt + sigma dW.
        tau = max(self.cfg.correlation_time, 1e-6)
        beta = dt / tau
        sigma = self.cfg.turbulence_intensity
        noise = self.rng.normal(0.0, 1.0, size=3)
        self._turb += -beta * self._turb + sigma * np.sqrt(2.0 * beta) * noise
        return self._mean + self._turb

    def body_velocity(self, R_body_from_ned: np.ndarray, dt: float) -> np.ndarray:
        """Return the 6-DOF body-frame current ``nu_c`` (rotational part = 0).

        Parameters
        ----------
        R_body_from_ned:
            3x3 rotation matrix mapping NED vectors into the body frame
            (i.e. the transpose of the body->NED rotation ``R(eta)``).
        """

        v_ned = self.velocity_ned(dt)
        v_body = R_body_from_ned @ v_ned
        nu_c = np.zeros(6)
        nu_c[:3] = v_body
        return nu_c
