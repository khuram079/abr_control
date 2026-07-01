"""Re-evaluate the improved hybrid (CFDL feed-forward + competence gate) over a
100-simulation Monte-Carlo campaign, reusing the trained TD3 checkpoint.

The MFAC-based inner loop now includes a model-free CFDL inverse feed-forward
that removes most of the phase lag on moving references; the TD3 policy is
unchanged (it is trained on the environment independently of the MFAC loop), so
the existing ``train500/td3_agent.pt`` checkpoint is loaded rather than
retrained.
"""

from __future__ import annotations

import argparse
import json
import os

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
from ..statistics import monte_carlo, compare
from .. import visualization as viz
from .real_experiment import _TD3Controller

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "improved_eval")
CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "train500", "td3_agent.pt")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--trajectory", default="sinusoidal")
    p.add_argument("--fault-prob", type=float, default=0.2, dest="fault_prob")
    p.add_argument("--load", default=CKPT)
    args = p.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = default_config(); tm = list(cfg.thruster.tau_max)

    env = AUVEnv(cfg)
    agent = TD3(env.obs_dim, env.act_dim, config=cfg.td3)
    agent.load(args.load)
    print(f"Loaded trained TD3 checkpoint: {args.load}", flush=True)

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

    print(f"\nMonte-Carlo ({args.trials} trials, {args.trajectory}, current + "
          f"faults@p={args.fault_prob}) — IMPROVED hybrid (CFDL feed-forward):", flush=True)
    print(f"  {'controller':<16s} {'RMSE mean':>10s} {'std':>8s} {'pos-RMSE':>9s} "
          f"{'energy':>10s}", flush=True)
    summaries, dists = {}, {}
    for name, fac in factories.items():
        mc = monte_carlo(fac, n_trials=args.trials, trajectory=args.trajectory,
                         config=cfg, randomize=True, fault_prob=args.fault_prob)
        summaries[name] = mc["summary"]; dists[name] = mc["distributions"]
        s = mc["summary"]
        print(f"  {name:<16s} {s['rmse']['mean']:>10.4f} {s['rmse']['std']:>8.4f} "
              f"{s['rmse_pos']['mean']:>9.4f} {s['control_energy']['mean']:>10.0f}", flush=True)

    print(f"\nSignificance vs Hybrid (TD3):", flush=True)
    for name in dists:
        if name == "Hybrid (TD3)":
            continue
        c = compare("Hybrid (TD3)", dists["Hybrid (TD3)"]["rmse"], name, dists[name]["rmse"])
        sig = "significant" if c["significant"] else "n.s."
        print(f"  vs {name:<16s} d={c['ttest']['cohens_d']:+.2f} "
              f"p={c['ttest']['p']:.3g}  ({sig})", flush=True)

    viz.plot_comparison_bars(summaries, "rmse", os.path.join(RESULTS_DIR, "improved_rmse_bars.png"),
                             title=f"Improved hybrid — MC RMSE ({args.trajectory}, current+faults)")
    fig, ax = plt.subplots(figsize=(11, 5))
    order = sorted(dists, key=lambda n: summaries[n]["rmse"]["mean"])
    ax.boxplot([dists[n]["rmse"] for n in order], showfliers=False)
    ax.set_xticks(np.arange(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("RMSE"); ax.set_title(f"Improved hybrid — RMSE distributions ({args.trials} trials)")
    plt.xticks(rotation=25, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "improved_rmse_box.png")); plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "eval_summary.json"), "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    print(f"\nArtifacts -> {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
