"""Adaptive actuator-fault (effectiveness) observer.

A partial actuator fault is modelled multiplicatively: the realised wrench is
``tau_real = Theta tau_cmd`` with per-channel effectiveness
``Theta = diag(theta_i), theta_i in (0, 1]`` (``theta_i = 1`` is healthy,
``theta_i = 0`` is a total loss).  The realised wrench is reconstructed from
the commanded wrench plus the disturbance-observer estimate that is aligned
with the input channels (``tau_real ~= tau_cmd + d_hat``), and each
effectiveness factor is estimated by a normalised recursive least-squares
update with a forgetting factor::

    theta_i <- clip( theta_i + gamma * tau_cmd_i (tau_real_i - theta_i tau_cmd_i)
                     / (lambda_f + tau_cmd_i^2), 0, 1 ).

Channels with negligible command are frozen (no information), which prevents
estimator wind-off when the actuator is idle.
"""

from __future__ import annotations

import numpy as np

from ..config import ObserverConfig


class FaultObserver:
    """Recursive estimator of per-channel actuator effectiveness."""

    def __init__(self, n: int = 6, config: ObserverConfig | None = None):
        self.n = int(n)
        self.cfg = config or ObserverConfig()
        self.reset()

    def reset(self) -> None:
        self.theta = np.ones(self.n)

    def update(
        self, tau_cmd: np.ndarray, tau_real: np.ndarray, excitation_floor: float = 1e-2
    ) -> np.ndarray:
        """Update and return the effectiveness estimate ``theta in (0, 1]``.

        Parameters
        ----------
        tau_cmd:
            Commanded generalised force.
        tau_real:
            Reconstructed realised force (e.g. ``tau_cmd + d_hat`` projected
            onto the input channels).
        excitation_floor:
            Channels with ``|tau_cmd| < floor`` carry no fault information and
            are left unchanged.
        """

        tau_cmd = np.asarray(tau_cmd, dtype=float).reshape(self.n)
        tau_real = np.asarray(tau_real, dtype=float).reshape(self.n)
        gamma = self.cfg.fault_adapt_gain
        lam = 1.0 - self.cfg.fault_forgetting + 1e-6

        for i in range(self.n):
            if abs(tau_cmd[i]) < excitation_floor:
                continue
            pred = self.theta[i] * tau_cmd[i]
            denom = lam + tau_cmd[i] ** 2
            self.theta[i] += gamma * tau_cmd[i] * (tau_real[i] - pred) / denom
        self.theta = np.clip(self.theta, 0.0, 1.0)
        return self.theta

    @property
    def healthy(self) -> np.ndarray:
        """Boolean mask of channels considered healthy (effectiveness > 0.8)."""

        return self.theta > 0.8
