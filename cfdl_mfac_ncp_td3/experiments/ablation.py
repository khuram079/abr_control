"""Ablation study and benchmark comparison for the hybrid controller.

Constructs controller *factories* (so each Monte-Carlo trial gets a fresh
instance), runs randomized Monte-Carlo evaluation for each, and reports metric
distributions plus pairwise statistical comparisons against the full hybrid
controller.

Ablation variants:

* ``MFAC``                 - inner adaptive loop only (no observers / fusion),
* ``MFAC+Obs``             - adds the observer suite,
* ``Hybrid (no RL)``       - confidence supervisor active, MFAC vs MFAC blend,
* ``Hybrid``               - full controller with the (optional) TD3 policy.

Classical baselines (PID, SMC, backstepping, MPC) are included for context.
"""

from __future__ import annotations

from ..config import ExperimentConfig, default_config
from ..controllers import HybridController
from ..benchmark import (
    PIDController,
    SMCController,
    BacksteppingController,
    MPCController,
)
from ..statistics import monte_carlo, compare


def build_factories(config: ExperimentConfig, td3_agent=None,
                    include_benchmarks: bool = True) -> dict:
    """Return a ``{name: factory}`` mapping for the ablation + baselines."""

    tau_max = list(config.thruster.tau_max)
    factories = {
        "MFAC": lambda: HybridController(config, use_observers=False,
                                         use_supervisor=False),
        "MFAC+Obs": lambda: HybridController(config, use_observers=True,
                                             use_supervisor=False),
        "Hybrid (no RL)": lambda: HybridController(config, use_observers=True,
                                                   use_supervisor=True),
        "Hybrid": lambda: HybridController(config, td3_agent=td3_agent,
                                           use_observers=True, use_supervisor=True),
    }
    if include_benchmarks:
        factories.update({
            "PID": lambda: PIDController(tau_max=tau_max),
            "SMC": lambda: SMCController(tau_max=tau_max),
            "Backstepping": lambda: BacksteppingController(tau_max=tau_max),
            "MPC": lambda: MPCController(tau_max=tau_max),
        })
    return factories


def run_ablation(config: ExperimentConfig | None = None, td3_agent=None,
                 n_trials: int = 20, trajectory: str = "sinusoidal",
                 fault_prob: float = 0.2, metric: str = "rmse",
                 reference: str = "Hybrid", include_benchmarks: bool = True,
                 verbose: bool = True) -> dict:
    """Run the full ablation + benchmark Monte-Carlo comparison."""

    cfg = config or default_config()
    factories = build_factories(cfg, td3_agent, include_benchmarks)

    results, summaries = {}, {}
    for name, factory in factories.items():
        mc = monte_carlo(factory, n_trials=n_trials, trajectory=trajectory,
                         config=cfg, randomize=True, fault_prob=fault_prob)
        results[name] = mc
        summaries[name] = mc["summary"]
        if verbose:
            s = mc["summary"][metric]
            print(f"  {name:<16s} {metric}={s['mean']:.4f} +/- {s['std']:.4f}")

    # Pairwise statistical comparison vs the reference controller.
    comparisons = {}
    if reference in results:
        ref_dist = results[reference]["distributions"][metric]
        for name, mc in results.items():
            if name == reference:
                continue
            comparisons[name] = compare(reference, ref_dist, name,
                                        mc["distributions"][metric])
    return {"results": results, "summaries": summaries,
            "comparisons": comparisons, "metric": metric}
