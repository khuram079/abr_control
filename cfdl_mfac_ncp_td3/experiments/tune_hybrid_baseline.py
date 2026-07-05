"""Fairly re-tune the strong hybrid baseline (with the nominal-model feed-forward).

Runs the same disclosed, energy-aware random search used for every baseline
(:mod:`cfdl_mfac_ncp_td3.formation.tuning`) on the hybrid -- now including the
computed-torque ``model_ff_gain`` and re-optimising ``trans_damping`` around it
-- and writes the resolved controller kwargs to ``hybrid_baseline.json``.  Both
the residual-RL training environment and the evaluation load that file, so the
training baseline and the evaluated strong hybrid are the identical, freshly
and fairly tuned controller.

Run::

    python -m cfdl_mfac_ncp_td3.experiments.tune_hybrid_baseline --budget 60
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..config import default_config
from ..formation.tuning import tune_controller
from ..environment.residual_env import BASELINE_JSON


def resolve(bp: dict) -> dict:
    """Map the search-space multipliers to absolute HybridController kwargs."""

    return dict(
        k_outer=float(bp["k_outer"]),
        cfdl_feedforward=float(bp["prediction_gain"]),
        feedforward_cap=float(bp["feedforward_cap"]),
        att_lam=float(1.5 * bp["att_lam"]),
        att_kd=(np.array([20.0, 30.0, 30.0]) * bp["att_kd"]).tolist(),
        att_ks=(np.array([8.0, 12.0, 12.0]) * bp["att_ks"]).tolist(),
        trans_damping=float(bp["trans_damping"]),
        model_feedforward=True,
        model_ff_gain=float(bp["model_ff_gain"]),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = default_config()
    print(f"{'='*72}\nRE-TUNE STRONG HYBRID (model feed-forward), budget={args.budget}\n"
          f"{'='*72}", flush=True)
    rep = tune_controller("Hybrid", n_samples=args.budget, seed=args.seed, cfg=cfg)
    print(f"best energy-aware score = {rep['best_score']:.4f}", flush=True)
    print(f"best params = { {k: round(v, 3) for k, v in rep['best_params'].items()} }",
          flush=True)

    kwargs = resolve(rep["best_params"])
    os.makedirs(os.path.dirname(BASELINE_JSON), exist_ok=True)
    with open(BASELINE_JSON, "w") as f:
        json.dump(kwargs, f, indent=2)
    print(f"Resolved strong-hybrid kwargs -> {BASELINE_JSON}", flush=True)
    print(json.dumps(kwargs, indent=2), flush=True)


if __name__ == "__main__":
    main()
