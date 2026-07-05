"""Residual-RL training environment.

The learned policy does **not** produce the full wrench; it outputs a bounded
*correction* added to the (strong) CFDL-MFAC command:

    tau = clip(tau_mfac + residual_scale * tau_max * a),   a in [-1, 1]^6.

The environment runs the model-free adaptive baseline internally each step and
exposes the same 18-D observation the deployed :class:`HybridController` uses,
so a policy trained here transfers directly to
``HybridController(residual_rl=True)``.  Because a zero action reproduces the
baseline exactly, the learned correction can only *improve* on it -- a much
easier and safer learning problem than learning a controller from scratch.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .auv_env import AUVEnv
from ..controllers import HybridController

#: Single source of truth for the tuned strong-hybrid baseline.  The
#: ``tune_hybrid_baseline`` experiment re-tunes the hybrid (now including the
#: nominal-model feed-forward) under the same fair energy-aware objective as the
#: baselines and writes the resolved kwargs here, so the residual-RL *training*
#: baseline and the *evaluation* strong hybrid are always the identical
#: controller.  The hard-coded fallback below is used only if the file is absent.
BASELINE_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "residual_v2", "hybrid_baseline.json")

_FALLBACK_BASELINE = dict(
    k_outer=0.507, trans_kp=150.0, mfac_trim_cap=0.5, trans_damping=0.0,
    att_lam=1.5 * 2.429, att_kd=[20.0 * 1.581, 30.0 * 1.581, 30.0 * 1.581],
    att_ks=[8.0 * 0.999, 12.0 * 0.999, 12.0 * 0.999],
)


def load_strong_baseline() -> dict:
    """Load the tuned strong-hybrid kwargs (from JSON if present)."""

    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON) as f:
            return json.load(f)
    return dict(_FALLBACK_BASELINE)


class AUVResidualEnv(AUVEnv):
    """AUV tracking env where the action is a residual on the MFAC command."""

    #: strong hybrid baseline (CFDL-MFAC + model feed-forward + SMC attitude +
    #: damping); the residual policy learns a bounded correction on top of THIS
    #: competitive, fairly-tuned controller, not the untuned default.
    STRONG_BASELINE = load_strong_baseline()

    def __init__(self, *args, residual_scale: float = 0.3,
                 baseline_kwargs: dict | None = None, **kwargs):
        self.residual_scale = float(residual_scale)
        self.baseline_kwargs = baseline_kwargs if baseline_kwargs is not None \
            else load_strong_baseline()
        super().__init__(*args, **kwargs)

    def _make_baseline(self) -> HybridController:
        # Strong hybrid baseline (no observers / supervisor / RL); the residual
        # correction is added on top via apply_residual().
        return HybridController(self.cfg, use_observers=False, use_supervisor=False,
                                residual_scale=self.residual_scale,
                                **self.baseline_kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.baseline = self._make_baseline()
        self.baseline.reset()
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(6), -1.0, 1.0)
        eta_d, eta_d_dot = self.traj.reference(self.t)

        # Model-free adaptive baseline command, then the bounded residual.
        tau_mfac = self.baseline.control(self.vehicle.eta, self.vehicle.nu,
                                         eta_d, eta_d_dot, self.dt)
        tau_cmd = self.baseline.apply_residual(tau_mfac, action)

        tau = self.thruster.step(tau_cmd, self.dt)
        nu_c = self._current_body()
        self.vehicle.step(tau, nu_c=nu_c)
        self.t += self.dt
        self.steps += 1

        eta_d, _ = self.traj.reference(self.t)
        from ..benchmark.base import pose_error
        err = pose_error(eta_d, self.vehicle.eta)
        err_norm = float(np.linalg.norm(err))
        # Reward tracks error and penalises the *residual* effort (not the full
        # wrench), so the policy is encouraged to correct only where it helps.
        reward = -(self.w_e * err_norm ** 2 + self.w_u * float(action @ action)) + 0.5

        terminated = err_norm > 25.0 or not np.all(np.isfinite(self.vehicle.state))
        truncated = self.t >= self.max_t
        obs = self._observe()
        info = {"error_norm": err_norm, "tau": tau, "eta_d": eta_d}
        return obs, float(reward), bool(terminated), bool(truncated), info
