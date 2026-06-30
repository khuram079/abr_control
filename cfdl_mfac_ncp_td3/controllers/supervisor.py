"""Confidence-guided hybrid supervisor (CFDL-MFAC + TD3, NCP-gated).

The supervisor fuses the model-free adaptive controller and the learned TD3
policy into a single command::

    u = (1 - alpha) * u_mfac + alpha * u_rl,

where ``alpha`` is the authority granted to the (robust) learned policy.  The
authority comes from the :class:`ConfidenceEstimator`; when the optional NCP
supervisor is enabled, its bounded output gates the analytic authority,
letting a trained liquid network shape the hand-over based on the full
operating context.  The blend reduces to pure MFAC under nominal conditions
and shifts toward the RL policy as error / disturbance / fault grow.
"""

from __future__ import annotations

import numpy as np

from ..config import SupervisorConfig, NCPConfig
from .confidence import ConfidenceEstimator

try:  # NCP is optional (torch dependency)
    import torch
    from ..ncp import LiquidNetwork

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class HybridSupervisor:
    """Blend MFAC and RL commands using confidence + optional NCP gating."""

    N_FEATURES = 8  # must match NCPConfig.n_sensory

    def __init__(self, config: SupervisorConfig | None = None,
                 ncp_config: NCPConfig | None = None,
                 error_scale: float = 1.0, dist_scale: float = 10.0):
        self.cfg = config or SupervisorConfig()
        self.confidence = ConfidenceEstimator(self.cfg, error_scale, dist_scale)
        self.use_ncp = self.cfg.use_ncp and _HAS_TORCH
        self.ncp = None
        self._ncp_state = None
        if self.use_ncp:
            ncp_config = ncp_config or NCPConfig(n_sensory=self.N_FEATURES, n_motor=1)
            self.ncp = LiquidNetwork(ncp_config)
        self.reset()

    def reset(self) -> None:
        self.confidence.reset()
        self._ncp_state = None
        self.last_alpha = self.cfg.blend_floor

    # ------------------------------------------------------------------ #
    def _features(self, error, d_hat, theta, u_mfac, u_rl, conf) -> np.ndarray:
        error = np.asarray(error, dtype=float).ravel()
        feats = np.array([
            np.tanh(np.linalg.norm(error)),
            np.tanh(np.max(np.abs(error)) if error.size else 0.0),
            np.tanh(np.linalg.norm(d_hat) / 10.0) if d_hat is not None else 0.0,
            float(np.min(theta)) if theta is not None else 1.0,
            float(np.mean(theta)) if theta is not None else 1.0,
            conf,
            np.tanh(np.linalg.norm(np.asarray(u_mfac, dtype=float)) / 10.0),
            np.tanh(np.linalg.norm(np.asarray(u_rl, dtype=float)) / 10.0),
        ], dtype=np.float32)
        return feats

    def authority(self, error, d_hat=None, theta=None, u_mfac=None, u_rl=None,
                  dt: float = 0.05) -> float:
        """Compute the RL authority ``alpha`` for the current context."""

        alpha = self.confidence.update(error, d_hat, theta)
        if self.use_ncp and u_mfac is not None and u_rl is not None:
            feats = self._features(error, d_hat, theta, u_mfac, u_rl,
                                   self.confidence.confidence)
            with torch.no_grad():
                gate, self._ncp_state = self.ncp(
                    torch.as_tensor(feats), self._ncp_state, dt=dt
                )
            gate = float(gate.item())
            # The learned gate multiplicatively shapes the analytic authority.
            alpha = float(np.clip(alpha * (0.5 + gate), self.cfg.blend_floor,
                                  self.cfg.blend_ceiling))
        self.last_alpha = alpha
        return alpha

    # ------------------------------------------------------------------ #
    def fuse(self, u_mfac: np.ndarray, u_rl: np.ndarray, error: np.ndarray,
             d_hat=None, theta=None, dt: float = 0.05):
        """Return the fused command and a diagnostics dict."""

        u_mfac = np.asarray(u_mfac, dtype=float).ravel()
        u_rl = np.asarray(u_rl, dtype=float).ravel()
        alpha = self.authority(error, d_hat, theta, u_mfac, u_rl, dt)
        u = (1.0 - alpha) * u_mfac + alpha * u_rl
        info = {
            "alpha": alpha,
            "confidence": self.confidence.confidence,
            "u_mfac": u_mfac,
            "u_rl": u_rl,
        }
        return u, info
