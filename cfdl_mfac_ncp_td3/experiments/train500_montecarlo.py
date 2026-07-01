"""Full-scale experiment: train the hybrid RL policy for 500 episodes, then
run a 100-simulation Monte-Carlo ablation + benchmark comparison.

Pipeline
--------
1. TRAIN the TD3 policy for ``--episodes`` episodes on the randomized AUV env
   (curriculum annealed over episodes, domain randomization, actuator faults).
2. EVALUATE with a ``--trials``-simulation Monte-Carlo campaign (ocean current
   + faults) over the full controller set:
   MFAC, MFAC+Obs, Hybrid (no RL), Hybrid (TD3, competence-gated),
   TD3-only, PID, SMC, Backstepping, MPC.
3. ABLATION + statistics: metric distributions, box plots, pairwise
   significance (Cohen's d, paired-t) vs the trained hybrid AND vs Hybrid
   (no RL), plus a fault-scenario trajectory demo with authority/trust traces.

Run::

    python -m cfdl_mfac_ncp_td3.experiments.train500_montecarlo \
        --episodes 500 --trials 100 --trajectory sinusoidal
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
from ..training import train_td3
from ..evaluation import rollout
from ..statistics import monte_carlo, compare
from .. import visualization as viz
from .real_experiment import _TD3Controller

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "train500")


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- #
def train(episodes: int, trajectory: str, fault_prob: float, seed: int):
    _log(f"\n{'='*72}\nSTAGE 8a  |  TRAIN HYBRID RL POLICY (CFDL-MFAC-NCP-TD3)\n"
         f"           episodes={episodes}  trajectory={trajectory}  "
         f"fault_prob={fault_prob}\n{'='*72}")
    t0 = time.time()
    out = train_td3(trajectory=trajectory, fault_prob=fault_prob, curriculum=True,
                    max_episodes=episodes, eval_every_episodes=max(1, episodes // 25),
                    seed=seed, verbose=True)
    dt = time.time() - t0
    agent, hist = out["agent"], out["history"]
    _log(f"Training done: {out['episodes']} episodes / {out['steps']} steps in "
         f"{dt/60:.1f} min ({out['steps']/dt:.0f} steps/s).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ep = np.array(hist["episode_return"])
    ax[0].plot(ep, alpha=0.3, color="C0", label="episode return")
    if len(ep) >= 10:
        k = max(1, len(ep) // 40)
        sm = np.convolve(ep, np.ones(k) / k, mode="valid")
        ax[0].plot(np.arange(len(sm)) + k // 2, sm, color="C0", lw=2, label="smoothed")
    ax[0].set_title("TD3 training return (500 episodes)")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("return"); ax[0].legend()
    if hist["eval_return"]:
        ev = np.array(hist["eval_return"])
        ax[1].plot(ev[:, 0], ev[:, 1], "C2-o", ms=4)
        ax[1].set_title("Deterministic eval return"); ax[1].set_xlabel("env step")
        ax[1].set_ylabel("mean return")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "training_curve.png"))
    plt.close(fig)
    agent.save(os.path.join(RESULTS_DIR, "td3_agent.pt"))
    _log(f"Saved checkpoint + training_curve.png")
    return agent


# --------------------------------------------------------------------- #
def evaluate(agent: TD3, trials: int, trajectory: str, fault_prob: float):
    _log(f"\n{'='*72}\nSTAGE 8b  |  MONTE-CARLO EVALUATION ({trials} simulations) + ABLATION\n"
         f"{'='*72}")
    cfg = default_config(); tm = list(cfg.thruster.tau_max)

    factories = {
        "MFAC": lambda: HybridController(cfg, use_observers=False, use_supervisor=False),
        "MFAC+Obs": lambda: HybridController(cfg, use_observers=True, use_supervisor=False),
        "Hybrid (no RL)": lambda: HybridController(cfg, use_observers=True, use_supervisor=True),
        "Hybrid (TD3)": lambda: HybridController(cfg, td3_agent=agent,
                                                 use_observers=True, use_supervisor=True),
        "TD3 only": lambda: _TD3Controller(agent, tm),
        "PID": lambda: PIDController(tau_max=tm),
        "SMC": lambda: SMCController(tau_max=tm),
        "Backstepping": lambda: BacksteppingController(tau_max=tm),
        "MPC": lambda: MPCController(tau_max=tm),
    }

    _log(f"\nMonte-Carlo ({trials} randomized trials, current + faults@p={fault_prob}):")
    _log(f"  {'controller':<16s} {'RMSE mean':>10s} {'std':>8s} {'pos-RMSE':>9s} "
         f"{'energy':>10s}")
    summaries, dists = {}, {}
    for name, fac in factories.items():
        mc = monte_carlo(fac, n_trials=trials, trajectory=trajectory, config=cfg,
                         randomize=True, fault_prob=fault_prob)
        summaries[name] = mc["summary"]; dists[name] = mc["distributions"]
        s = mc["summary"]
        _log(f"  {name:<16s} {s['rmse']['mean']:>10.4f} {s['rmse']['std']:>8.4f} "
             f"{s['rmse_pos']['mean']:>9.4f} {s['control_energy']['mean']:>10.0f}")

    # significance vs both the trained hybrid and the no-RL hybrid
    for ref in ("Hybrid (TD3)", "Hybrid (no RL)"):
        _log(f"\nPairwise significance vs {ref} (Cohen's d, paired-t p):")
        for name in dists:
            if name == ref:
                continue
            c = compare(ref, dists[ref]["rmse"], name, dists[name]["rmse"])
            sig = "significant" if c["significant"] else "n.s."
            _log(f"  vs {name:<16s} d={c['ttest']['cohens_d']:+.2f} "
                 f"p={c['ttest']['p']:.3g}  ({sig})")

    # RMSE bar chart + box plot
    viz.plot_comparison_bars(summaries, "rmse", os.path.join(RESULTS_DIR, "eval_rmse_bars.png"),
                             title=f"Monte-Carlo RMSE ({trajectory}, current+faults)")
    fig, ax = plt.subplots(figsize=(11, 5))
    order = sorted(dists, key=lambda n: summaries[n]["rmse"]["mean"])
    ax.boxplot([dists[n]["rmse"] for n in order], showfliers=False)
    ax.set_xticks(np.arange(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("RMSE"); ax.set_title(f"RMSE distributions ({trials} MC trials)")
    plt.xticks(rotation=25, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_rmse_box.png")); plt.close(fig)

    # fault-scenario demo: trained hybrid vs no-RL, with authority + trust
    _log(f"\nFault-scenario demo (channel 0 @ 40% effectiveness, current on):")
    demo = {}
    for name, fac in [("Hybrid (TD3)", factories["Hybrid (TD3)"]),
                      ("Hybrid (no RL)", factories["Hybrid (no RL)"])]:
        r = rollout(fac(), trajectory=trajectory, config=cfg, current=True,
                    fault=(0, 0.4), seed=7)
        demo[name] = r
        _log(f"  {name:<16s} rmse={r['metrics']['rmse']:.4f}  "
             f"mean_alpha={np.mean(r['log']['alpha']):.3f}")
    viz.plot_trajectory_3d(demo["Hybrid (TD3)"]["log"],
                           os.path.join(RESULTS_DIR, "demo_trajectory_3d.png"))
    viz.plot_blending(demo["Hybrid (TD3)"]["log"],
                      os.path.join(RESULTS_DIR, "demo_authority.png"))

    # verdict on the competence gate
    rmse_td3 = summaries["Hybrid (TD3)"]["rmse"]["mean"]
    rmse_norl = summaries["Hybrid (no RL)"]["rmse"]["mean"]
    _log(f"\nCompetence-gate verdict: Hybrid(TD3) {rmse_td3:.3f} vs "
         f"Hybrid(no RL) {rmse_norl:.3f} -> "
         f"{'RL improves' if rmse_td3 < rmse_norl else 'RL neutral/worse; gate limits damage'}")

    with open(os.path.join(RESULTS_DIR, "eval_summary.json"), "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    _log(f"\nArtifacts -> {RESULTS_DIR}/")
    return summaries


# --------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--trajectory", default="sinusoidal")
    p.add_argument("--fault-prob", type=float, default=0.2, dest="fault_prob")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent = train(args.episodes, args.trajectory, args.fault_prob, args.seed)
    evaluate(agent, args.trials, args.trajectory, args.fault_prob)


if __name__ == "__main__":
    main()
