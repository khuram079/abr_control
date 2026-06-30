"""Confidence estimator for model-free adaptive control.

The estimator condenses the current operating condition into a scalar
*confidence* in the CFDL-MFAC controller.  Confidence is high when the
tracking error, the observed disturbance and the actuator-fault severity are
all small, and degrades smoothly as any of them grows::

    c_mfac = exp( -( w_e * e_n + w_d * d_n + w_f * f_n ) ) in (0, 1],

with normalised, saturated signals ``e_n, d_n, f_n``.  The complementary
quantity ``alpha = 1 - c_mfac`` is the authority handed to the (robust) learned
policy.  A first-order low-pass filter prevents authority chattering.
"""

from __future__ import annotations

import numpy as np

from ..config import SupervisorConfig


class ConfidenceEstimator:
    """Maps error / disturbance / fault signals to MFAC confidence and RL authority."""

    def __init__(self, config: SupervisorConfig | None = None,
                 error_scale: float = 1.0, dist_scale: float = 10.0):
        self.cfg = config or SupervisorConfig()
        self.error_scale = float(error_scale)
        self.dist_scale = float(dist_scale)
        self.reset()

    def reset(self) -> None:
        self._alpha = self.cfg.blend_floor
        self.confidence = 1.0

    def update(self, error: np.ndarray, d_hat: np.ndarray | None = None,
               theta: np.ndarray | None = None) -> float:
        """Return the (smoothed) RL authority ``alpha in [floor, ceiling]``."""

        error = np.asarray(error, dtype=float).ravel()
        e_n = np.tanh(np.linalg.norm(error) / max(self.error_scale, 1e-9))

        d_n = 0.0
        if d_hat is not None:
            d_n = np.tanh(np.linalg.norm(d_hat) / max(self.dist_scale, 1e-9))

        f_n = 0.0
        if theta is not None:
            theta = np.asarray(theta, dtype=float).ravel()
            # Fault severity = worst-case effectiveness loss in [0, 1].
            f_n = float(np.clip(1.0 - np.min(theta), 0.0, 1.0))

        cost = self.cfg.w_error * e_n + self.cfg.w_disturbance * d_n + self.cfg.w_fault * f_n
        self.confidence = float(np.exp(-cost))
        alpha_raw = np.clip(1.0 - self.confidence, self.cfg.blend_floor, self.cfg.blend_ceiling)

        # First-order low-pass on the authority signal.
        s = self.cfg.smoothing
        self._alpha = (1.0 - s) * self._alpha + s * alpha_raw
        return self._alpha

    @property
    def authority(self) -> float:
        return self._alpha
