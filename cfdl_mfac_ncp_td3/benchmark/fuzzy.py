"""Fuzzy-logic (Mamdani PD) benchmark controller.

A per-DOF fuzzy proportional-derivative controller on the NED pose error.  Each
channel fuzzifies the error ``e`` and error-rate ``edot`` with three triangular
membership functions (Negative / Zero / Positive), applies the standard 3x3
PD rule base, and defuzzifies by centroid (weighted average).  The crisp output
is scaled by per-DOF gains ``ke`` and by the actuator limit.

This is a genuine, tunable baseline (its scaling gains are tuned by the same
fairness procedure as the other benchmarks, see :mod:`formation.tuning`).
"""

from __future__ import annotations

import numpy as np

from ..dynamics.remus6dof import jacobian
from .base import BaseController, pose_error


def _tri_memberships(x: float, width: float) -> np.ndarray:
    """Membership degrees (N, Z, P) for a normalized input over [-width, width]."""

    xn = np.clip(x / max(width, 1e-9), -1.0, 1.0)
    # Negative: 1 at -1 -> 0 at 0 ; Zero: triangle at 0 ; Positive: mirror.
    neg = max(0.0, -xn)
    pos = max(0.0, xn)
    zero = 1.0 - abs(xn)
    m = np.array([neg, zero, pos])
    s = m.sum()
    return m / s if s > 1e-9 else np.array([0.0, 1.0, 0.0])


# Rule consequent centroids for the 3x3 (e x edot) PD rule base, in [-1, 1].
# Rows = e in {N,Z,P}, cols = edot in {N,Z,P}; classic PD surface.
_RULE = np.array([
    [-1.0, -1.0, 0.0],
    [-1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
])


class FuzzyController(BaseController):
    """Per-DOF Mamdani fuzzy PD pose-tracking controller."""

    name = "Fuzzy"

    def __init__(self, ke=None, e_width=None, edot_width=None, tau_max=None):
        self.ke = np.array(ke if ke is not None else [40, 40, 40, 15, 25, 25], dtype=float)
        self.e_width = np.array(e_width if e_width is not None else [2.0] * 6, dtype=float)
        self.edot_width = np.array(edot_width if edot_width is not None else [1.5] * 6, dtype=float)
        self.tau_max = None if tau_max is None else np.asarray(tau_max, dtype=float)

    def _fuzzy_channel(self, e: float, edot: float, ew: float, dw: float) -> float:
        me = _tri_memberships(e, ew)          # (3,)
        md = _tri_memberships(edot, dw)        # (3,)
        weights = np.outer(me, md)             # (3,3) rule firing strengths
        num = float(np.sum(weights * _RULE))
        den = float(np.sum(weights))
        return num / den if den > 1e-9 else 0.0

    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float = 0.05) -> np.ndarray:
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, dtype=float)
        e = pose_error(eta_d, eta)
        edot = eta_d_dot - jacobian(eta) @ nu
        u = np.array([
            self._fuzzy_channel(e[i], edot[i], self.e_width[i], self.edot_width[i])
            for i in range(6)
        ])
        tau = self.ke * u
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau
