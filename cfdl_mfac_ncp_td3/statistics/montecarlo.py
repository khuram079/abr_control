"""Monte-Carlo evaluation across randomized operating conditions."""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig, default_config
from ..dynamics import REMUSParams
from ..evaluation.evaluator import rollout


def monte_carlo(controller_factory, n_trials: int = 30,
                trajectory: str = "sinusoidal",
                config: ExperimentConfig | None = None,
                randomize: bool = True, fault_prob: float = 0.0,
                base_seed: int = 0) -> dict:
    """Evaluate a controller over ``n_trials`` randomized episodes.

    Parameters
    ----------
    controller_factory:
        Zero-argument callable returning a fresh controller (so per-trial
        randomization of internal model parameters is possible).
    randomize:
        Randomize vehicle mass and enable ocean current per trial.
    fault_prob:
        Probability of injecting a random actuator fault per trial.
    """

    cfg = config or default_config()
    rng = np.random.default_rng(base_seed)
    records = []
    for i in range(n_trials):
        seed = base_seed + i
        trng = np.random.default_rng(seed)
        params = REMUSParams()
        if randomize:
            params = REMUSParams(mass=30.48 * (1.0 + trng.uniform(-0.15, 0.15)))
        fault = None
        if trng.random() < fault_prob:
            fault = (int(trng.integers(6)), float(trng.uniform(0.2, 0.7)))
        ctrl = controller_factory()
        res = rollout(ctrl, trajectory=trajectory, config=cfg,
                      current=randomize, fault=fault, seed=seed, params=params)
        records.append(res["metrics"])

    # Aggregate metric distributions.
    keys = [k for k in records[0] if isinstance(records[0][k], (int, float))]
    dist = {k: np.array([r[k] for r in records], dtype=float) for k in keys}

    def _agg(v: np.ndarray) -> dict:
        # A metric may be all-NaN (e.g. settling_time when never held); report
        # NaNs without triggering numpy's empty-slice warnings.
        if not np.any(np.isfinite(v)):
            nan = float("nan")
            return {"mean": nan, "std": nan, "median": nan, "min": nan, "max": nan}
        return {
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v)),
            "median": float(np.nanmedian(v)),
            "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)),
        }

    summary = {k: _agg(v) for k, v in dist.items()}
    return {"records": records, "distributions": dist, "summary": summary,
            "n_trials": n_trials}
