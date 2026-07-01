"""Real end-to-end experiment: TRAIN the TD3 policy, THEN evaluate.

This is not a smoke test.  It (1) trains the TD3 policy on the randomized AUV
environment with curriculum learning, saving the checkpoint and the learning
curve; then (2) evaluates the *trained-policy* hybrid controller against the
ablation variants and the classical benchmarks with a randomized Monte-Carlo
campaign (ocean current + actuator faults), reporting metric distributions and
pairwise statistical significance, plus trajectory/authority plots.

Run::

    python -m cfdl_mfac_ncp_td3.experiments.real_experiment \
        --steps 80000 --trials 40 --trajectory sinusoidal
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import default_config
from ..controllers import HybridController
from ..benchmark import (PIDController, SMCController,
                         BacksteppingController, MPCController)
from ..rl import TD3
from ..environment import AUVEnv
from ..training import train_td3
from ..evaluation import rollout
from ..statistics import monte_carlo, compare
from .. import visualization as viz

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "real_run")


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- #
def train(steps: int, trajectory: str, fault_prob: float, seed: int) -> tuple[TD3, dict]:
    _log(f"\n{'='*70}\nTRAINING TD3  |  steps={steps}  trajectory={trajectory}  "
         f"fault_prob={fault_prob}\n{'='*70}")
    t0 = time.time()
    out = train_td3(total_steps=steps, trajectory=trajectory, fault_prob=fault_prob,
                    curriculum=True, eval_every=max(2000, steps // 20), seed=seed,
                    verbose=True)
    dt = time.time() - t0
    agent, hist = out["agent"], out["history"]
    _log(f"Training done in {dt/60:.1f} min "
         f"({steps/dt:.0f} steps/s, {len(hist['episode_return'])} episodes).")

    # learning-curve plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if hist["episode_return"]:
        ep = np.array(hist["episode_return"])
        ax[0].plot(ep, alpha=0.35, color="C0", label="episode return")
        if len(ep) >= 10:
            k = max(1, len(ep) // 30)
            sm = np.convolve(ep, np.ones(k) / k, mode="valid")
            ax[0].plot(np.arange(len(sm)) + k // 2, sm, color="C0", lw=2, label="smoothed")
        ax[0].set_title("TD3 training return"); ax[0].set_xlabel("episode")
        ax[0].set_ylabel("return"); ax[0].legend()
    if hist["eval_return"]:
        ev = np.array(hist["eval_return"])
        ax[1].plot(ev[:, 0], ev[:, 1], "C2-o", ms=4)
        ax[1].set_title("TD3 deterministic eval return"); ax[1].set_xlabel("env step")
        ax[1].set_ylabel("mean return")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "training_curve.png"))
    plt.close(fig)

    ckpt = os.path.join(RESULTS_DIR, "td3_agent.pt")
    agent.save(ckpt)
    _log(f"Saved checkpoint -> {ckpt}")
    return agent, hist


# --------------------------------------------------------------------- #
def evaluate(agent: TD3, trials: int, trajectory: str, fault_prob: float) -> dict:
    _log(f"\n{'='*70}\nEVALUATION (trained policy in the loop)\n{'='*70}")
    cfg = default_config()
    tm = list(cfg.thruster.tau_max)

    factories = {
        "MFAC": lambda: HybridController(cfg, use_observers=False, use_supervisor=False),
        "Hybrid (no RL)": lambda: HybridController(cfg, use_observers=True, use_supervisor=True),
        "Hybrid (TD3)": lambda: HybridController(cfg, td3_agent=agent,
                                                 use_observers=True, use_supervisor=True),
        "TD3 only": lambda: _TD3Controller(agent, tm),
        "PID": lambda: PIDController(tau_max=tm),
        "SMC": lambda: SMCController(tau_max=tm),
        "Backstepping": lambda: BacksteppingController(tau_max=tm),
        "MPC": lambda: MPCController(tau_max=tm),
    }

    _log(f"\nMonte-Carlo ({trials} randomized trials, current + faults@p={fault_prob}), "
         f"RMSE mean +/- std:")
    summaries, dists = {}, {}
    for name, fac in factories.items():
        mc = monte_carlo(fac, n_trials=trials, trajectory=trajectory, config=cfg,
                         randomize=True, fault_prob=fault_prob)
        summaries[name] = mc["summary"]; dists[name] = mc["distributions"]
        s = mc["summary"]
        _log(f"  {name:<16s} rmse={s['rmse']['mean']:.4f} +/- {s['rmse']['std']:.4f}   "
             f"energy={s['control_energy']['mean']:.0f}")

    # significance vs the trained hybrid
    ref = "Hybrid (TD3)"
    _log(f"\nPairwise significance vs {ref} (Cohen's d, paired-t p):")
    comps = {}
    for name in dists:
        if name == ref:
            continue
        c = compare(ref, dists[ref]["rmse"], name, dists[name]["rmse"])
        comps[name] = c
        sig = "significant" if c["significant"] else "n.s."
        _log(f"  vs {name:<16s} d={c['ttest']['cohens_d']:+.2f} "
             f"p={c['ttest']['p']:.3g}  ({sig})")

    # comparison bar chart
    viz.plot_comparison_bars(summaries, "rmse",
                             os.path.join(RESULTS_DIR, "eval_rmse.png"),
                             title=f"Monte-Carlo RMSE ({trajectory}, current+faults)")

    # single trajectory demo: trained hybrid vs no-RL hybrid under a fault
    _log(f"\nSingle-episode demo under a forced fault (channel 0, 40% effectiveness):")
    demo = {}
    for name, fac in [("Hybrid (TD3)", factories["Hybrid (TD3)"]),
                      ("Hybrid (no RL)", factories["Hybrid (no RL)"])]:
        r = rollout(fac(), trajectory=trajectory, config=cfg, current=True,
                    fault=(0, 0.4), seed=7)
        demo[name] = r
        _log(f"  {name:<16s} rmse={r['metrics']['rmse']:.4f}  "
             f"mean_alpha={np.mean(r['log']['alpha']):.3f}")
    viz.plot_tracking_errors(demo["Hybrid (TD3)"]["log"],
                             os.path.join(RESULTS_DIR, "demo_hybrid_td3_errors.png"))
    viz.plot_blending(demo["Hybrid (TD3)"]["log"],
                      os.path.join(RESULTS_DIR, "demo_hybrid_td3_authority.png"))

    # persist a machine-readable summary
    with open(os.path.join(RESULTS_DIR, "eval_summary.json"), "w") as f:
        json.dump({"summaries": summaries,
                   "significance": {k: {"cohens_d": v["ttest"]["cohens_d"],
                                        "p": v["ttest"]["p"],
                                        "significant": v["significant"]}
                                    for k, v in comps.items()}}, f, indent=2)
    _log(f"\nArtifacts written to {RESULTS_DIR}/")
    return {"summaries": summaries, "comparisons": comps}


class _TD3Controller:
    """Adapter so the raw trained TD3 policy fits the rollout controller API."""

    name = "TD3 only"

    def __init__(self, agent: TD3, tau_max):
        from ..benchmark.base import pose_error
        self.agent = agent
        self.tau_max = np.asarray(tau_max, dtype=float)
        self._pose_error = pose_error
        self.last_info = {"alpha": 1.0}

    def reset(self):
        pass

    def control(self, eta, nu, eta_d, eta_d_dot=None, dt=0.05):
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, float)
        z1 = self._pose_error(eta_d, eta)
        obs = np.concatenate([z1, np.asarray(nu, float), eta_d_dot]).astype(np.float32)
        return np.clip(self.agent.select_action(obs, noise=None), -1, 1) * self.tau_max


# --------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Real train-then-evaluate pipeline")
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--trajectory", default="sinusoidal")
    p.add_argument("--fault-prob", type=float, default=0.2, dest="fault_prob")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent, _ = train(args.steps, args.trajectory, args.fault_prob, args.seed)
    evaluate(agent, args.trials, args.trajectory, args.fault_prob)


if __name__ == "__main__":
    main()
