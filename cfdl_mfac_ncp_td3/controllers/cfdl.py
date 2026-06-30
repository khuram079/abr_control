"""Compact-Form Dynamic Linearization (CFDL) data model.

This thin wrapper couples the :class:`PseudoJacobianEstimator` with the bookkeeping
of past inputs/outputs needed to form the increments ``Delta y`` and ``Delta u``.
It exposes the locally-linearized model that the MFAC control law consumes, and
is deliberately separated from the controller so the same linearization can be
reused by observers or analysis tools.
"""

from __future__ import annotations

import numpy as np

from ..config import MFACConfig
from .pseudo_gradient import PseudoJacobianEstimator


class CFDLModel:
    """Maintains the CFDL linearization ``Delta y(k+1) = Phi(k) Delta u(k)``."""

    def __init__(self, n_out: int, n_in: int, config: MFACConfig | None = None):
        self.n_out = int(n_out)
        self.n_in = int(n_in)
        self.cfg = config or MFACConfig()
        self.estimator = PseudoJacobianEstimator(n_out, n_in, self.cfg)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self._y_prev = None
        self._u_prev = np.zeros(self.n_in)
        self._u_pprev = np.zeros(self.n_in)
        self._initialised = False

    @property
    def phi(self) -> np.ndarray:
        return self.estimator.phi

    def observe(self, y: np.ndarray) -> None:
        """Feed a new output measurement and update the PJM.

        The update uses the *previous* input increment ``Delta u(k-1)`` together
        with the resulting output increment ``Delta y(k)``.
        """

        y = np.asarray(y, dtype=float).reshape(self.n_out)
        if self._initialised:
            dy = y - self._y_prev
            du_prev = self._u_prev - self._u_pprev
            self.estimator.update(dy, du_prev)
        self._y_prev = y
        self._initialised = True

    def register_input(self, u: np.ndarray) -> None:
        """Record the input applied at step ``k`` (shifts the input history)."""

        u = np.asarray(u, dtype=float).reshape(self.n_in)
        self._u_pprev = self._u_prev
        self._u_prev = u

    @property
    def last_input(self) -> np.ndarray:
        return self._u_prev
