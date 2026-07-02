"""Residual-RL hybrid: train 1000 episodes, then 500-simulation Monte-Carlo
with parameter testing.

The learned policy is trained as a *bounded residual* on the strong CFDL-MFAC
command (see :class:`AUVResidualEnv`), so it can only improve on the adaptive
baseline.  After training, the residual hybrid is evaluated against the
baseline and the classical benchmarks over a 500-trial randomized Monte-Carlo
campaign, plus a parameter-sensitivity sweep over current speed and fault
severity.

Run::

    python -m cfdl_mfac_ncp_td3.experiments.residual_experiment \
        --episodes 1000 --trials 500 --trajectory sinusoidal
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

from ..config import default_config, SimConfig
from ..controllers import HybridController
from ..benchmark import (PIDController, SMCController,
                         BacksteppingController, MPCController)
from ..rl import TD3
from ..environment import AUVResidualEnv
from ..training import train_td3
from ..evaluation import rollout
from ..statistics import monte_carlo, compare
from ..dynamics import REMUSParams
from .. import visualization as viz

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "final")
RESIDUAL_SCALE = 0.3


def _log(m: str) -> None:
    print(m, flush=True)


# --------------------------------------------------------------------- #
def train(episodes: int, trajectory: str, fault_prob: float, seed: int) -> TD3:
    _log(f"\n{'='*74}\nTRAIN RESIDUAL-RL HYBRID (CFDL-MFAC-NCP + residual TD3)\n"
         f"   episodes={episodes}  trajectory={trajectory}  fault_prob={fault_prob}  "
         f"residual_scale={RESIDUAL_SCALE}\n{'='*74}")
    cfg = default_config()
    cfg.sim = SimConfig(dt=0.05, horizon=30.0)  # 600-step training episodes

    def make_env(**kw):
        return AUVResidualEnv(residual_scale=RESIDUAL_SCALE, **kw)

    t0 = time.time()
    out = train_td3(trajectory=trajectory, config=cfg, fault_prob=fault_prob,
                    curriculum=True, max_episodes=episodes,
                    eval_every_episodes=max(1, episodes // 40),
                    make_env=make_env, seed=seed, verbose=True)
    dt = time.time() - t0
    agent, hist = out["agent"], out["history"]
    _log(f"Training done: {out['episodes']} episodes / {out['steps']} steps in "
         f"{dt/60:.1f} min ({out['steps']/dt:.0f} steps/s).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ep = np.array(hist["episode_return"])
    ax[0].plot(ep, alpha=0.25, color="C0")
    if len(ep) >= 20:
        k = max(1, len(ep) // 50)
        sm = np.convolve(ep, np.ones(k) / k, mode="valid")
        ax[0].plot(np.arange(len(sm)) + k // 2, sm, color="C0", lw=2)
    ax[0].set_title("Residual-RL training return (1000 episodes)")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("return")
    if hist["eval_return"]:
        ev = np.array(hist["eval_return"])
        ax[1].plot(ev[:, 0], ev[:, 1], "C2-o", ms=3)
        ax[1].set_title("Deterministic eval return"); ax[1].set_xlabel("env step")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "training_curve.png")); plt.close(fig)
    agent.save(os.path.join(RESULTS_DIR, "residual_agent.pt"))
    _log("Saved residual_agent.pt + training_curve.png")
    return agent


# --------------------------------------------------------------------- #
def evaluate(agent: TD3, trials: int, trajectory: str, fault_prob: float):
    _log(f"\n{'='*74}\nMONTE-CARLO EVALUATION ({trials} simulations) + PARAMETER TESTING\n{'='*74}")
    cfg = default_config(); tm = list(cfg.thruster.tau_max)

    factories = {
        "Hybrid (Residual-RL)": lambda: HybridController(cfg, td3_agent=agent,
                                                         residual_rl=True,
                                                         residual_scale=RESIDUAL_SCALE,
                                                         use_observers=False),
        "Hybrid (no RL)": lambda: HybridController(cfg, use_observers=False, use_supervisor=False),
        "PID": lambda: PIDController(tau_max=tm),
        "SMC": lambda: SMCController(tau_max=tm),
        "Backstepping": lambda: BacksteppingController(tau_max=tm),
        "MPC": lambda: MPCController(tau_max=tm),
    }

    _log(f"\nMonte-Carlo ({trials} randomized trials, current + faults@p={fault_prob}):")
    _log(f"  {'controller':<20s} {'RMSE mean':>10s} {'std':>8s} {'pos-RMSE':>9s} {'energy':>10s}")
    summaries, dists = {}, {}
    for name, fac in factories.items():
        mc = monte_carlo(fac, n_trials=trials, trajectory=trajectory, config=cfg,
                         randomize=True, fault_prob=fault_prob)
        summaries[name] = mc["summary"]; dists[name] = mc["distributions"]
        s = mc["summary"]
        _log(f"  {name:<20s} {s['rmse']['mean']:>10.4f} {s['rmse']['std']:>8.4f} "
             f"{s['rmse_pos']['mean']:>9.4f} {s['control_energy']['mean']:>10.0f}")

    ref = "Hybrid (no RL)"
    _log(f"\nDoes the residual RL help? Significance vs {ref}:")
    for name in ("Hybrid (Residual-RL)",):
        c = compare(ref, dists[ref]["rmse"], name, dists[name]["rmse"])
        improve = summaries[ref]["rmse"]["mean"] - summaries[name]["rmse"]["mean"]
        _log(f"  {name}: delta_rmse={improve:+.4f} "
             f"({100*improve/summaries[ref]['rmse']['mean']:+.1f}%)  "
             f"d={c['ttest']['cohens_d']:+.2f}  p={c['ttest']['p']:.3g}")

    # --- parameter-sensitivity sweep ---------------------------------- #
    _log(f"\nParameter testing (RMSE; 6 seeds each) — Residual vs no-RL vs Backstepping:")
    _log(f"  {'condition':<28s} {'Residual':>9s} {'no-RL':>9s} {'Backstep':>9s}")
    sweep = {}
    from ..config import CurrentConfig
    conditions = [
        ("nominal", dict()),
        ("current=0.6 m/s", dict(cur=0.6)),
        ("mass +40%", dict(mass=1.4)),
        ("drag +100%", dict(drag=2.0)),
        ("fault ch0=0.3", dict(fault=(0, 0.3))),
        ("current+fault+mass", dict(cur=0.5, fault=(0, 0.4), mass=1.3)),
    ]
    for label, cond in conditions:
        c2 = default_config()
        if "cur" in cond:
            c2.current = CurrentConfig(mean_velocity=(cond["cur"], cond["cur"] * 0.5, 0.0),
                                       turbulence_intensity=0.02)
        p = REMUSParams(mass=30.48 * cond.get("mass", 1.0))
        if "drag" in cond:
            for k in ("Xu", "Yv", "Zw", "Nr", "Mq"):
                setattr(p, k, getattr(p, k) * cond["drag"])
        fault = cond.get("fault")
        use_cur = "cur" in cond
        row = {}
        for cname, fac in [("Residual", factories["Hybrid (Residual-RL)"]),
                           ("no-RL", factories["Hybrid (no RL)"]),
                           ("Backstep", factories["Backstepping"])]:
            rs = [rollout(fac(), trajectory=trajectory, config=c2, current=use_cur,
                          fault=fault, seed=s, params=p)["metrics"]["rmse"]
                  for s in range(6)]
            row[cname] = float(np.mean(rs))
        sweep[label] = row
        _log(f"  {label:<28s} {row['Residual']:>9.3f} {row['no-RL']:>9.3f} {row['Backstep']:>9.3f}")

    # plots
    viz.plot_comparison_bars(summaries, "rmse", os.path.join(RESULTS_DIR, "mc_rmse_bars.png"),
                             title=f"Residual-RL hybrid — MC RMSE ({trials} trials)")
    fig, ax = plt.subplots(figsize=(11, 5))
    order = sorted(dists, key=lambda n: summaries[n]["rmse"]["mean"])
    ax.boxplot([dists[n]["rmse"] for n in order], showfliers=False)
    ax.set_xticks(np.arange(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("RMSE"); ax.set_title(f"RMSE distributions ({trials} MC trials)")
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mc_rmse_box.png")); plt.close(fig)

    # demo trajectory
    demo = rollout(factories["Hybrid (Residual-RL)"](), trajectory="helix", config=cfg,
                   current=True, seed=3)
    viz.plot_trajectory_3d(demo["log"], os.path.join(RESULTS_DIR, "demo_helix_3d.png"))
    viz.plot_tracking_errors(demo["log"], os.path.join(RESULTS_DIR, "demo_helix_errors.png"))

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"monte_carlo": summaries, "parameter_sweep": sweep,
                   "trials": trials, "residual_scale": RESIDUAL_SCALE}, f, indent=2)
    _log(f"\nArtifacts -> {RESULTS_DIR}/")
    return summaries, sweep


# --------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--trajectory", default="sinusoidal")
    p.add_argument("--fault-prob", type=float, default=0.2, dest="fault_prob")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent = train(args.episodes, args.trajectory, args.fault_prob, args.seed)
    evaluate(agent, args.trials, args.trajectory, args.fault_prob)


if __name__ == "__main__":
    main()
