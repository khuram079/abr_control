"""Stage 6 validation: confidence estimator and hybrid fusion supervisor.

Checks the monotone behaviour the fusion logic must guarantee: nominal
conditions keep authority with MFAC; growing error / disturbance / fault hand
authority to the (robust) RL policy; the blend stays a convex combination; and
the NCP-gated path runs and stays bounded.
"""

import numpy as np
import pytest

from cfdl_mfac_ncp_td3.config import SupervisorConfig, NCPConfig
from cfdl_mfac_ncp_td3.controllers import ConfidenceEstimator, HybridSupervisor


def _settle(est, error, d=None, theta=None, n=300):
    a = 0.0
    for _ in range(n):
        a = est.update(error, d, theta)
    return a


def test_confidence_high_when_nominal():
    est = ConfidenceEstimator(SupervisorConfig(smoothing=0.2))
    a = _settle(est, np.zeros(6))
    assert est.confidence > 0.95
    assert a < 0.05  # almost all authority stays with MFAC


def test_authority_increases_with_error():
    est = ConfidenceEstimator(SupervisorConfig(smoothing=0.3))
    a_small = _settle(est, np.full(6, 0.05))
    est.reset()
    a_large = _settle(est, np.full(6, 2.0))
    assert a_large > a_small
    assert a_large > 0.5


def test_authority_increases_with_fault():
    est = ConfidenceEstimator(SupervisorConfig(smoothing=0.3))
    a_healthy = _settle(est, np.zeros(6), theta=np.ones(6))
    est.reset()
    a_faulted = _settle(est, np.zeros(6), theta=np.array([1, 1, 0.2, 1, 1, 1.0]))
    assert a_faulted > a_healthy


def test_authority_increases_with_disturbance():
    est = ConfidenceEstimator(SupervisorConfig(smoothing=0.3), dist_scale=5.0)
    a_low = _settle(est, np.zeros(6), d=np.zeros(6))
    est.reset()
    a_high = _settle(est, np.zeros(6), d=np.full(6, 8.0))
    assert a_high > a_low


def test_authority_within_bounds():
    est = ConfidenceEstimator(SupervisorConfig(blend_floor=0.1, blend_ceiling=0.8,
                                               smoothing=0.5))
    a = _settle(est, np.full(6, 100.0))
    assert 0.1 - 1e-6 <= a <= 0.8 + 1e-6


def test_fusion_is_convex_combination():
    sup = HybridSupervisor(SupervisorConfig(use_ncp=False, smoothing=0.5))
    u_mfac = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    u_rl = np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    u, info = sup.fuse(u_mfac, u_rl, error=np.full(6, 0.5))
    a = info["alpha"]
    assert np.allclose(u, (1 - a) * u_mfac + a * u_rl)
    # Each component lies within the span of the two source commands.
    lo, hi = np.minimum(u_mfac, u_rl), np.maximum(u_mfac, u_rl)
    assert np.all(u >= lo - 1e-9) and np.all(u <= hi + 1e-9)


def test_fusion_prefers_mfac_when_nominal_and_rl_when_degraded():
    # Pure distress-based allocation (competence gate validated separately).
    sup = HybridSupervisor(SupervisorConfig(use_ncp=False, smoothing=0.4,
                                            competence_gating=False,
                                            directional_gate=False))
    u_mfac = np.ones(6)
    u_rl = -np.ones(6)
    # Nominal: result close to MFAC.
    for _ in range(200):
        u_nom, info_nom = sup.fuse(u_mfac, u_rl, error=np.zeros(6))
    sup.reset()
    # Severely degraded: result shifts toward RL.
    for _ in range(200):
        u_deg, info_deg = sup.fuse(u_mfac, u_rl, error=np.full(6, 3.0),
                                   theta=np.full(6, 0.1))
    assert info_nom["alpha"] < info_deg["alpha"]
    assert np.linalg.norm(u_nom - u_mfac) < np.linalg.norm(u_deg - u_mfac)


def test_competence_gate_suppresses_harmful_rl():
    """If error keeps growing while RL holds authority, trust -> low -> alpha -> low."""

    sup = HybridSupervisor(SupervisorConfig(use_ncp=False, competence_gating=True,
                                            directional_gate=False,
                                            trust_init=0.5, trust_rate=0.1,
                                            smoothing=1.0))
    # Sustained large error (distress high) with a worsening trend each step.
    err = 1.0
    last = None
    for _ in range(200):
        _, last = sup.fuse(np.ones(6), -np.ones(6), error=np.full(6, err))
        err += 0.02  # error grows -> RL is (implicitly) not helping
    assert sup.trust < 0.2, f"trust should collapse, got {sup.trust:.3f}"
    assert last["alpha"] < 0.25, f"authority should be suppressed, got {last['alpha']:.3f}"


def test_competence_gate_grants_authority_to_helpful_rl():
    """If error keeps falling while RL holds authority, trust -> high."""

    sup = HybridSupervisor(SupervisorConfig(use_ncp=False, competence_gating=True,
                                            directional_gate=False,
                                            trust_init=0.5, trust_rate=0.1,
                                            smoothing=1.0))
    err = 3.0
    for _ in range(200):
        _, last = sup.fuse(np.ones(6), -np.ones(6), error=np.full(6, err))
        err = max(0.05, err - 0.02)  # error steadily improves
    assert sup.trust > 0.85, f"trust should grow, got {sup.trust:.3f}"


def test_directional_gate_suppresses_opposed_rl():
    """RL opposed to the adaptive command (cos < 0) gets zero authority."""

    sup = HybridSupervisor(SupervisorConfig(use_ncp=False, competence_gating=False,
                                            directional_gate=True, smoothing=1.0))
    # u_rl points opposite to u_mfac -> cosine = -1 -> gate = 0.
    _, info = sup.fuse(np.ones(6), -np.ones(6), error=np.full(6, 2.0))
    assert info["agreement"] == 0.0
    assert info["alpha"] == 0.0


def test_directional_gate_admits_agreeing_rl():
    """RL aligned with the adaptive command keeps (scaled) authority."""

    sup = HybridSupervisor(SupervisorConfig(use_ncp=False, competence_gating=False,
                                            directional_gate=True, smoothing=1.0))
    # u_rl == 2 * u_mfac -> cosine = +1 -> gate = 1, authority preserved.
    _, info = sup.fuse(np.ones(6), 2 * np.ones(6), error=np.full(6, 2.0))
    assert info["agreement"] > 0.99
    assert info["alpha"] > 0.0


def test_ncp_gated_supervisor_runs_and_is_bounded():
    sup = HybridSupervisor(SupervisorConfig(use_ncp=True, blend_floor=0.0,
                                            blend_ceiling=1.0),
                           NCPConfig(n_sensory=8, n_motor=1))
    if not sup.use_ncp:
        pytest.skip("torch/NCP unavailable")
    for _ in range(50):
        u, info = sup.fuse(np.ones(6), -np.ones(6), error=np.random.rand(6),
                           d_hat=np.random.rand(6), theta=np.random.rand(6))
        assert 0.0 - 1e-6 <= info["alpha"] <= 1.0 + 1e-6
        assert np.all(np.isfinite(u))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
