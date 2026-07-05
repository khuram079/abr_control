"""Fairly re-tune the strong hybrid baseline (proportional + adaptive-trim law).

Runs the same disclosed, energy-aware random search used for every baseline
(:mod:`cfdl_mfac_ncp_td3.formation.tuning`) on the hybrid -- now over the
proportional velocity gain ``trans_kp`` and the bounded CFDL-MFAC trim cap
``mfac_trim_cap`` that replace the pure integrating MFAC surge loop -- and
writes the resolved controller kwargs to ``hybrid_baseline.json``.  Both the
residual-RL training environment and the evaluation load that file, so the
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
        trans_kp=float(bp["trans_kp"]),
        mfac_trim_cap=float(bp["mfac_trim_cap"]),
        trans_damping=float(bp["trans_damping"]),
        att_lam=float(1.5 * bp["att_lam"]),
        att_kd=(np.array([20.0, 30.0, 30.0]) * bp["att_kd"]).tolist(),
        att_ks=(np.array([8.0, 12.0, 12.0]) * bp["att_ks"]).tolist(),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = default_config()
    print(f"{'='*72}\nRE-TUNE STRONG HYBRID (proportional + adaptive trim), budget={args.budget}\n"
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
