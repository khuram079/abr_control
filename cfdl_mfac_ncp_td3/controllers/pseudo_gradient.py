"""Pseudo-Jacobian Matrix (PJM) estimator for Compact-Form Dynamic Linearization.

Compact-Form Dynamic Linearization (CFDL) replaces an unknown nonlinear
discrete-time MIMO plant ``y(k+1) = f(y(k), ..., u(k), ...)`` with the local
input-output model

    Delta y(k+1) = Phi(k) Delta u(k),

where ``Delta y(k+1) = y(k+1) - y(k)``, ``Delta u(k) = u(k) - u(k-1)`` and
``Phi(k)`` is the (time-varying) *pseudo-Jacobian matrix* (PJM).  The PJM is
estimated online with a projection-type algorithm; no plant model is used.

Reference: Z. Hou and S. Jin, *Model Free Adaptive Control: Theory and
Applications*, CRC Press, 2013.
"""

from __future__ import annotations

import numpy as np

from ..config import MFACConfig


class PseudoJacobianEstimator:
    """Online estimator of the CFDL pseudo-Jacobian matrix.

    Parameters
    ----------
    n_out, n_in:
        Output and input dimensions (the PJM is ``n_out x n_in``).
    config:
        MFAC hyper-parameters (``eta``, ``mu``, ``epsilon``, ``phi_init``).
    """

    def __init__(self, n_out: int, n_in: int, config: MFACConfig | None = None):
        self.n_out = int(n_out)
        self.n_in = int(n_in)
        self.cfg = config or MFACConfig()
        self.reset()

    # ------------------------------------------------------------------ #
    def _initial_phi(self) -> np.ndarray:
        phi0 = np.zeros((self.n_out, self.n_in))
        # Seed the diagonal so the control law has authority from step 1.
        k = min(self.n_out, self.n_in)
        phi0[:k, :k] = self.cfg.phi_init * np.eye(k)
        return phi0

    def reset(self) -> None:
        self.phi = self._initial_phi()
        self._phi_init = self._initial_phi()

    # ------------------------------------------------------------------ #
    def update(self, dy: np.ndarray, du_prev: np.ndarray) -> np.ndarray:
        """Update the PJM with the most recent input/output increments.

        Parameters
        ----------
        dy:
            Output increment ``y(k) - y(k-1)`` (length ``n_out``).
        du_prev:
            Input increment ``u(k-1) - u(k-2)`` (length ``n_in``).
        """

        dy = np.asarray(dy, dtype=float).reshape(self.n_out)
        du = np.asarray(du_prev, dtype=float).reshape(self.n_in)

        denom = self.cfg.mu + float(du @ du)
        # Projection / gradient update of the PJM (Hou & Jin eq. for CFDL):
        #   Phi <- Phi + eta (dy - Phi du) du^T / (mu + ||du||^2)
        innovation = dy - self.phi @ du
        self.phi = self.phi + self.cfg.eta * np.outer(innovation, du) / denom

        self._apply_reset(du)
        return self.phi

    def _apply_reset(self, du: np.ndarray) -> None:
        """Reset the PJM to its initial value when the estimate degenerates.

        The reset conditions keep the estimator well-conditioned and preserve
        the controllability assumption that the diagonal of the PJM does not
        change sign.
        """

        eps = self.cfg.epsilon
        reset = False
        if np.linalg.norm(self.phi) <= eps:
            reset = True
        if np.linalg.norm(du) <= eps:
            reset = True
        # Diagonal sign-preservation (per the CFDL convergence assumptions).
        k = min(self.n_out, self.n_in)
        diag = np.diag(self.phi)[:k]
        diag0 = np.diag(self._phi_init)[:k]
        if np.any(np.sign(diag) != np.sign(diag0)):
            reset = True
        if reset:
            self.phi = self._phi_init.copy()

    # ------------------------------------------------------------------ #
    def predict(self, du: np.ndarray) -> np.ndarray:
        """One-step output-increment prediction ``Delta y = Phi Delta u``."""

        return self.phi @ np.asarray(du, dtype=float).reshape(self.n_in)
