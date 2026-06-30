"""Stage 8 validation: integration of env, trajectories, hybrid controller,
evaluation, statistics and visualization.
"""

import os

import numpy as np
import pytest

from cfdl_mfac_ncp_td3.config import default_config
from cfdl_mfac_ncp_td3.trajectories import make_trajectory, TRAJECTORY_REGISTRY
from cfdl_mfac_ncp_td3.environment import AUVEnv
from cfdl_mfac_ncp_td3.controllers import HybridController
from cfdl_mfac_ncp_td3.benchmark import PIDController
from cfdl_mfac_ncp_td3.evaluation import rollout
from cfdl_mfac_ncp_td3.evaluation import metrics as M
from cfdl_mfac_ncp_td3.statistics import monte_carlo, compare, cohens_d
from cfdl_mfac_ncp_td3 import visualization as viz


# ----------------------------- trajectories ------------------------------ #
@pytest.mark.parametrize("name", list(TRAJECTORY_REGISTRY))
def test_trajectory_outputs_finite_shapes(name):
    traj = make_trajectory(name)
    for t in np.linspace(0, traj.duration, 25):
        eta_d, eta_d_dot = traj.reference(float(t))
        assert eta_d.shape == (6,) and eta_d_dot.shape == (6,)
        assert np.all(np.isfinite(eta_d)) and np.all(np.isfinite(eta_d_dot))


# ------------------------------- gym env ---------------------------------- #
def test_env_api_and_rollout():
    env = AUVEnv(default_config(), trajectory="setpoint", randomize=False, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    total = 0.0
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample()
                                             if hasattr(env, "action_space") else np.zeros(6))
        total += r
        assert obs.shape == (env.obs_dim,)
        if term or trunc:
            break
    assert np.isfinite(total)


def test_env_zero_action_setpoint_is_finite():
    env = AUVEnv(default_config(), trajectory="setpoint", randomize=False, seed=1)
    env.reset()
    for _ in range(100):
        obs, r, term, trunc, info = env.step(np.zeros(6))
        assert np.all(np.isfinite(obs))


def test_env_difficulty_curriculum_setter():
    env = AUVEnv(default_config(), randomize=True, seed=2)
    env.set_difficulty(0.9)
    assert env.difficulty == 0.9


# -------------------------- hybrid controller ----------------------------- #
def test_hybrid_controller_tracks_setpoint():
    cfg = default_config()
    ctrl = HybridController(cfg)  # MFAC + observers + supervisor (no RL)
    res = rollout(ctrl, trajectory="setpoint", config=cfg, seed=0)
    assert not res["metrics"]["diverged"]
    # Final error should be a fraction of the initial setpoint distance.
    err = np.linalg.norm(res["log"]["error"], axis=1)
    assert err[-1] < 0.3 * err[0], f"hybrid failed to converge: {err[0]:.2f}->{err[-1]:.2f}"


def test_hybrid_controller_modes_run():
    cfg = default_config()
    for obs_flag, sup_flag in [(False, False), (True, False), (True, True)]:
        ctrl = HybridController(cfg, use_observers=obs_flag, use_supervisor=sup_flag)
        res = rollout(ctrl, trajectory="sinusoidal", config=cfg, seed=0)
        assert np.all(np.isfinite(res["log"]["eta"]))


def test_hybrid_robust_under_current_and_fault():
    cfg = default_config()
    ctrl = HybridController(cfg)
    res = rollout(ctrl, trajectory="setpoint", config=cfg, current=True,
                  fault=(0, 0.5), seed=3)
    assert not res["metrics"]["diverged"]


# ------------------------------ metrics ----------------------------------- #
def test_metrics_on_known_signal():
    errs = np.tile(np.array([[3, 4, 0, 0, 0, 0]], dtype=float), (10, 1))
    taus = np.ones((10, 6))
    assert np.isclose(M.rmse(errs), 5.0)  # 3-4-5 triangle
    assert np.isclose(M.iae(errs, 1.0), 5.0 * 10)
    assert M.control_energy(taus, 1.0) > 0


# ---------------------------- statistics ---------------------------------- #
def test_compare_detects_difference():
    a = np.random.default_rng(0).normal(1.0, 0.1, 30)
    b = np.random.default_rng(1).normal(2.0, 0.1, 30)
    rep = compare("A", a, "B", b)
    assert rep["significant"]
    assert abs(rep["ttest"]["cohens_d"]) > 1.0  # large effect


def test_monte_carlo_aggregates():
    cfg = default_config()
    mc = monte_carlo(lambda: PIDController(tau_max=list(cfg.thruster.tau_max)),
                     n_trials=4, trajectory="setpoint", config=cfg,
                     randomize=True, fault_prob=0.0)
    assert mc["n_trials"] == 4
    assert "rmse" in mc["summary"]
    assert mc["distributions"]["rmse"].shape == (4,)


# --------------------------- visualization -------------------------------- #
def test_plots_are_written(tmp_path):
    cfg = default_config()
    res = rollout(HybridController(cfg), trajectory="setpoint", config=cfg, seed=0)
    p1 = viz.plot_trajectory_3d(res["log"], str(tmp_path / "traj.png"))
    p2 = viz.plot_tracking_errors(res["log"], str(tmp_path / "err.png"))
    p3 = viz.plot_blending(res["log"], str(tmp_path / "alpha.png"))
    for p in (p1, p2, p3):
        assert os.path.exists(p) and os.path.getsize(p) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
