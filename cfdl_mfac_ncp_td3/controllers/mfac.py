"""CFDL-based Model-Free Adaptive Controller (MFAC).

The controller is *data-driven*: it uses only measured input/output data and
the online CFDL pseudo-Jacobian estimate, never a plant model.  The control
law minimises the one-step cost

    J(u(k)) = || y*(k+1) - y(k+1) ||^2 + lambda || u(k) - u(k-1) ||^2,

which, under the CFDL model ``Delta y(k+1) = Phi(k) Delta u(k)``, yields the
scalar-normalised update

    u(k) = u(k-1) + rho * Phi(k)^T (y*(k+1) - y(k)) / (lambda + ||Phi(k)||^2).

Reference: Z. Hou and S. Jin, *Model Free Adaptive Control*, CRC Press, 2013.
"""

from __future__ import annotations

import numpy as np

from ..config import MFACConfig
from .cfdl import CFDLModel


class CFDLMFAC:
    """Compact-form MFAC controller for an ``n_out``-output AUV subsystem.

    Parameters
    ----------
    n_out, n_in:
        Output (controlled) and input dimensions.
    config:
        MFAC hyper-parameters.
    u_bounds:
        Optional ``(low, high)`` arrays clamping the absolute control signal.
    """

    def __init__(
        self,
        n_out: int,
        n_in: int,
        config: MFACConfig | None = None,
        u_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.cfg = config or MFACConfig()
        self.model = CFDLModel(n_out, n_in, self.cfg)
        self.n_out = int(n_out)
        self.n_in = int(n_in)
        self.u_bounds = u_bounds
        self.reset()

    def reset(self) -> None:
        self.model.reset()
        self._u = np.zeros(self.n_in)

    @property
    def phi(self) -> np.ndarray:
        return self.model.phi

    def control(self, y: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Compute the control signal for measurement ``y`` and reference ``y_ref``."""

        y = np.asarray(y, dtype=float).reshape(self.n_out)
        y_ref = np.asarray(y_ref, dtype=float).reshape(self.n_out)

        # 1) Update the CFDL linearization with the latest measurement.
        self.model.observe(y)
        phi = self.model.phi

        # 2) MFAC control-law increment.
        err = y_ref - y
        denom = self.cfg.lam + float(np.sum(phi * phi))
        du = self.cfg.rho * (phi.T @ err) / denom

        # 3) Per-channel increment clipping for robustness.
        du = np.clip(du, -self.cfg.u_limit, self.cfg.u_limit)
        u = self._u + du

        # 4) Absolute saturation (optional).
        if self.u_bounds is not None:
            low, high = self.u_bounds
            u = np.clip(u, low, high)

        # 5) Commit the input and update the model's input history.
        self._u = u
        self.model.register_input(u)
        return u
