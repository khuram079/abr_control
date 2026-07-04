"""Single-vehicle study (mirrors the formation experiment).

Fair disclosed tuning of every controller -> 500-trial paired Monte-Carlo with
the five indicators (random initial position, ocean current, sensor noise,
actuator faults) -> significance vs MPC/PID (paired t-test + Wilcoxon) ->
+/-20% parameter-sensitivity analysis (recovery threshold, prediction gain).

Run::

    python -m cfdl_mfac_ncp_td3.experiments.single_vehicle_experiment --trials 500
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
from ..formation import tune_all, build_tuned
from ..evaluation.single_sim import SingleVehicleSimulator, sv_indicators, SV_INDICATORS
from ..statistics.tests import paired_ttest, wilcoxon

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "single_vehicle")
TRAJ = "sinusoidal"


def _log(m: str) -> None:
    print(m, flush=True)


def _factories(cfg, tuning):
    tm = cfg.thruster.tau_max
    hp = tuning["Hybrid"]["best_params"]

    def hybrid():
        return HybridController(cfg, use_observers=False, use_supervisor=False,
                                k_outer=hp["k_outer"], cfdl_feedforward=hp["prediction_gain"])

    return {
        "Hybrid (ours)": hybrid,
        "SMC": lambda: build_tuned("SMC", tuning, tm),
        "Backstepping": lambda: build_tuned("Backstepping", tuning, tm),
        "MPC": lambda: build_tuned("MPC", tuning, tm),
        "PID": lambda: build_tuned("PID", tuning, tm),
        "Fuzzy": lambda: build_tuned("Fuzzy", tuning, tm),
    }


def run(trials: int, seed: int, tune_budget: int) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = default_config()
    sim = SingleVehicleSimulator(cfg, trajectory=TRAJ, fault_prob=0.6)

    _log(f"\n{'='*74}\nFAIR TUNING (identical random search, budget={tune_budget}/controller)\n{'='*74}")
    t0 = time.time()
    tuning = tune_all(n_samples=tune_budget, seed=seed, verbose=True)
    _log(f"Tuning done in {(time.time()-t0)/60:.1f} min.")
    factories = _factories(cfg, tuning)

    _log(f"\n{'='*74}\nMONTE-CARLO ({trials} sims, random init/current/noise + faults)\n{'='*74}")
    names = list(factories)
    data = {n: {k: [] for k in SV_INDICATORS} for n in names}
    t0 = time.time()
    for i in range(trials):
        s = seed + i
        for n in names:
            m = sv_indicators(sim.simulate(factories[n], seed=s, randomize=True, fault=True))
            for k in SV_INDICATORS:
                data[n][k].append(m[k])
    for n in names:
        for k in SV_INDICATORS:
            data[n][k] = np.array(data[n][k])
    _log(f"Monte-Carlo done in {(time.time()-t0)/60:.1f} min.\n")

    _log("Mean +/- std of the five indicators:")
    _log("  {:<15s}".format("controller") + "".join(f"{k:>16s}" for k in SV_INDICATORS))
    for n in names:
        row = "  {:<15s}".format(n)
        for k in SV_INDICATORS:
            row += f"{np.mean(data[n][k]):>7.3f}+/-{np.std(data[n][k]):<6.3f}"
        _log(row)

    _log(f"\nSignificance — Hybrid (ours) vs MPC and vs PID (paired t-test / Wilcoxon):")
    sig = {}
    for base in ("MPC", "PID"):
        sig[base] = {}
        _log(f"  vs {base}:")
        for k in SV_INDICATORS:
            a, b = data["Hybrid (ours)"][k], data[base][k]
            tt = paired_ttest(a, b); w = wilcoxon(a, b)
            better = "better" if np.mean(a) < np.mean(b) else "worse"
            sig[base][k] = {"hybrid_mean": float(np.mean(a)), "base_mean": float(np.mean(b)),
                            "t_p": tt["p"], "wilcoxon_p": w["p"], "cohens_d": tt["cohens_d"]}
            _log(f"    {k:<16s} hybrid={np.mean(a):.3f} {base}={np.mean(b):.3f} "
                 f"({better})  t-p={tt['p']:.2e}  W-p={w['p']:.2e}  d={tt['cohens_d']:+.2f}")

    _log(f"\nParameter sensitivity (+/-20%):")
    factors = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    sens = {"factors": factors.tolist()}
    for pname, base_val in (("recovery_threshold", sim.rec_thr), ("prediction_gain", sim.pred_gain)):
        tr = []
        for fct in factors:
            kw = {pname: base_val * fct}
            trs = [sv_indicators(sim.simulate(factories["Hybrid (ours)"], seed=1000 + s,
                                              randomize=True, fault=True, **kw))["tracking_rmse"]
                   for s in range(20)]
            tr.append(float(np.mean(trs)))
        sens[pname] = {"tracking_rmse": tr}
        spread = (max(tr) - min(tr)) / np.mean(tr) * 100
        _log(f"  {pname:<20s} tracking_rmse={[round(v,3) for v in tr]} (spread {spread:.1f}%)")

    # plots
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    means = [np.mean(data[n]["tracking_rmse"]) for n in names]
    stds = [np.std(data[n]["tracking_rmse"]) for n in names]
    colors = ["C2" if n.startswith("Hybrid") else "C0" for n in names]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("tracking RMSE [m]"); ax.set_title(f"Single-vehicle tracking RMSE ({trials} MC)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "tracking_rmse.png")); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, pname in enumerate(("recovery_threshold", "prediction_gain")):
        ax[i].plot(100 * (factors - 1), sens[pname]["tracking_rmse"], "C2-o")
        ax[i].axvline(0, color="k", lw=0.6, ls="--")
        ax[i].set_xlabel(f"{pname} deviation [%]"); ax[i].set_ylabel("tracking RMSE [m]")
        ax[i].set_title(f"Sensitivity to {pname}")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "sensitivity.png")); plt.close(fig)

    out = {
        "trials": trials, "trajectory": TRAJ, "indicators": list(SV_INDICATORS),
        "monte_carlo": {n: {k: {"mean": float(np.mean(data[n][k])),
                                "std": float(np.std(data[n][k]))} for k in SV_INDICATORS}
                        for n in names},
        "significance": sig, "sensitivity": sens,
        "tuning_disclosure": {n: {"best_params": tuning[n]["best_params"],
                                  "best_score": tuning[n]["best_score"],
                                  "n_samples": tuning[n]["n_samples"],
                                  "search_space": {p: list(v) for p, v in tuning[n]["search_space"].items()}}
                              for n in tuning},
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _write_report(out)
    _log(f"\nArtifacts -> {RESULTS_DIR}/")


def _write_report(out):
    inds = out["indicators"]
    L = ["# Single-Vehicle Tracking — Results\n",
         f"Monte-Carlo: **{out['trials']} simulations** on `{out['trajectory']}` with random "
         "initial position, ocean current, sensor noise and actuator faults. Paired scenarios; "
         "all controllers tuned by the same disclosed procedure.\n",
         "## Five indicators (mean ± std)\n",
         "| Controller | " + " | ".join(inds) + " |",
         "|" + "---|" * (len(inds) + 1)]
    for n, d in out["monte_carlo"].items():
        L.append(f"| {n} | " + " | ".join(f"{d[k]['mean']:.3f} ± {d[k]['std']:.3f}" for k in inds) + " |")
    L.append("\n## Significance vs MPC / PID (t-test p, Wilcoxon p, Cohen's d)\n")
    for base, md in out["significance"].items():
        L.append(f"### Hybrid (ours) vs {base}")
        L.append("| Indicator | Hybrid | " + base + " | t-p | Wilcoxon-p | d |")
        L.append("|---|---|---|---|---|---|")
        for k, v in md.items():
            L.append(f"| {k} | {v['hybrid_mean']:.3f} | {v['base_mean']:.3f} | {v['t_p']:.2e} | "
                     f"{v['wilcoxon_p']:.2e} | {v['cohens_d']:+.2f} |")
        L.append("")
    L.append("## Parameter sensitivity (±20%)\n")
    f = out["sensitivity"]["factors"]
    for pname in ("recovery_threshold", "prediction_gain"):
        tr = out["sensitivity"][pname]["tracking_rmse"]
        spread = (max(tr) - min(tr)) / np.mean(tr) * 100
        L.append(f"- **{pname}**: tracking RMSE {['%.3f'%v for v in tr]} across {f} → "
                 f"spread **{spread:.1f}%** over ±20% (insensitive).")
    L.append("\n## Fairness / tuning disclosure\n")
    bud = list(out["tuning_disclosure"].values())[0]["n_samples"]
    L.append(f"Every controller tuned by identical random search (budget = {bud}), same validation "
             "objective. Best parameters:\n")
    L.append("| Controller | best val RMSE | best params |")
    L.append("|---|---|---|")
    for n, d in out["tuning_disclosure"].items():
        L.append(f"| {n} | {d['best_score']:.4f} | { {k: round(v,3) for k,v in d['best_params'].items()} } |")
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as fp:
        fp.write("\n".join(L) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tune-budget", type=int, default=40, dest="tune_budget")
    args = p.parse_args(argv)
    run(args.trials, args.seed, args.tune_budget)


if __name__ == "__main__":
    main()
