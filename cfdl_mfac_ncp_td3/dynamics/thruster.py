"""Actuator model and control allocation.

A REMUS-class AUV is actuated by a single stern propeller (surge thrust) plus
control surfaces (rudder + stern planes) that, at trim speed, produce sway,
heave, pitch and yaw forces/moments.  Rather than model each surface
explicitly, we expose a *generalised-force* interface: the controller commands
a 6-DOF wrench ``tau_cmd`` and the :class:`ThrusterModel` applies

* first-order actuator lag,
* rate limiting, and
* per-channel saturation,

returning the realised wrench ``tau`` that drives the rigid-body dynamics.
:func:`AdaptiveAllocator` additionally rescales an infeasible command so that
the most-saturated channel rides its limit while preserving direction, which
the supervisor uses for graceful degradation under thruster faults.
"""

from __future__ import annotations

import numpy as np

from ..config import ThrusterConfig


class ThrusterModel:
    """First-order actuator dynamics with rate limits and saturation."""

    def __init__(self, config: ThrusterConfig | None = None):
        self.cfg = config or ThrusterConfig()
        self.tau_max = np.asarray(self.cfg.tau_max, dtype=float)
        self._tau = np.zeros(6)
        # Per-channel effectiveness in [0, 1]; 1.0 == healthy actuator.
        self.effectiveness = np.ones(6)

    def reset(self) -> None:
        self._tau = np.zeros(6)
        self.effectiveness = np.ones(6)

    def set_fault(self, channel: int, effectiveness: float) -> None:
        """Inject a partial actuator fault on ``channel`` (0..5)."""

        self.effectiveness[channel] = float(np.clip(effectiveness, 0.0, 1.0))

    def saturate(self, tau: np.ndarray) -> np.ndarray:
        return np.clip(tau, -self.tau_max, self.tau_max)

    def step(self, tau_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Advance the actuator state and return the realised wrench."""

        tau_cmd = self.saturate(np.asarray(tau_cmd, dtype=float).reshape(6))
        # Rate limit on the commanded wrench.
        max_delta = self.cfg.rate_limit * dt
        delta = np.clip(tau_cmd - self._tau, -max_delta, max_delta)
        target = self._tau + delta
        # First-order lag toward the (rate-limited) target.
        alpha = dt / max(self.cfg.time_constant, 1e-6)
        alpha = min(alpha, 1.0)
        self._tau = self._tau + alpha * (target - self._tau)
        # Apply actuator effectiveness (fault) and final saturation.
        return self.saturate(self.effectiveness * self._tau)


class AdaptiveAllocator:
    """Direction-preserving rescaling of an infeasible generalised force.

    If any channel of ``tau_cmd`` exceeds its limit the whole vector is scaled
    down so the worst channel sits exactly on its bound.  This keeps the
    *direction* of the commanded wrench, which matters far more than its
    magnitude for trajectory tracking and is the basis of the fault-tolerant
    allocation used by the supervisor.
    """

    def __init__(self, tau_max: np.ndarray):
        self.tau_max = np.asarray(tau_max, dtype=float)

    def __call__(
        self, tau_cmd: np.ndarray, effectiveness: np.ndarray | None = None
    ) -> np.ndarray:
        tau_cmd = np.asarray(tau_cmd, dtype=float).reshape(6)
        limit = self.tau_max.copy()
        if effectiveness is not None:
            # A degraded actuator can deliver less force; shrink its budget.
            limit = limit * np.clip(effectiveness, 1e-3, 1.0)
        ratios = np.abs(tau_cmd) / np.maximum(limit, 1e-9)
        worst = float(np.max(ratios))
        if worst > 1.0:
            tau_cmd = tau_cmd / worst
        return tau_cmd
