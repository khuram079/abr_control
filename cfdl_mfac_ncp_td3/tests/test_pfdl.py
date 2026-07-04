"""Tests for PFDL-MFAC (Partial-Form Dynamic Linearization MFAC).

Validates the partial-form controller as a drop-in alternative to CFDL-MFAC:
L = 1 must reproduce CFDL exactly, and L > 1 must still track a stable plant.
"""

import numpy as np
import pytest

from cfdl_mfac_ncp_td3.config import MFACConfig
from cfdl_mfac_ncp_td3.controllers import CFDLMFAC, PFDLMFAC


def _track(ctrl, a=0.7, b=0.5, ref=1.0, n=200):
    y = 0.0
    errs = []
    for _ in range(n):
        u = ctrl.control([y], [ref])[0]
        y = a * y + b * u
        errs.append(abs(ref - y))
    return np.array(errs)


def test_pfdl_is_siso_only():
    with pytest.raises(ValueError):
        PFDLMFAC(2, 2)


def test_pfdl_L1_matches_cfdl():
    cfg = MFACConfig(rho=0.5, lam=1.0, phi_init=1.0, u_limit=5.0)
    c = CFDLMFAC(1, 1, cfg)
    p = PFDLMFAC(1, 1, cfg, L=1)
    # Same input sequence -> identical commands (L=1 PFDL == CFDL).
    y = 0.0
    for _ in range(50):
        uc = c.control([y], [1.0])[0]
        up = p.control([y], [1.0])[0]
        assert np.isclose(uc, up, atol=1e-9)
        y = 0.7 * y + 0.5 * up


def test_pfdl_tracks_stable_plant():
    cfg = MFACConfig(rho=0.5, lam=1.0, phi_init=1.0, u_limit=5.0)
    for L in (2, 3, 5):
        errs = _track(PFDLMFAC(1, 1, cfg, L=L))
        assert errs[-1] < 1e-2, f"PFDL(L={L}) failed to track: {errs[-1]}"


def test_pfdl_gain_is_leading_component():
    cfg = MFACConfig(phi_init=0.01)
    p = PFDLMFAC(1, 1, cfg, L=4)
    assert np.isclose(p.gain, 0.01)
    assert p.phi.shape == (4,)


def test_pfdl_respects_bounds():
    cfg = MFACConfig(rho=2.0, lam=0.01, phi_init=1.0, u_limit=100.0)
    p = PFDLMFAC(1, 1, cfg, L=3, u_bounds=(np.array([-0.5]), np.array([0.5])))
    y = 0.0
    for _ in range(50):
        u = p.control([y], [100.0])[0]
        assert -0.5 - 1e-9 <= u <= 0.5 + 1e-9
        y = 0.7 * y + 0.5 * u


def test_hybrid_accepts_pfdl_inner_law():
    from cfdl_mfac_ncp_td3.config import default_config
    from cfdl_mfac_ncp_td3.controllers import HybridController
    from cfdl_mfac_ncp_td3.evaluation import rollout

    cfg = default_config()
    ctrl = HybridController(cfg, use_observers=False, use_supervisor=False,
                            inner_law="pfdl", pfdl_L=3)
    res = rollout(ctrl, trajectory="setpoint", config=cfg, seed=0)
    assert not res["metrics"]["diverged"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
