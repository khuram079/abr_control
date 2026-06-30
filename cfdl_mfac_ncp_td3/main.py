"""Command-line entry point for the CFDL-MFAC-NCP-TD3 framework.

Examples
--------
Quick end-to-end demo (no RL training), writes plots to ``outputs/``::

    python -m cfdl_mfac_ncp_td3.main demo --trajectory helix

Train the TD3 policy, then run the ablation study::

    python -m cfdl_mfac_ncp_td3.main train --steps 50000
    python -m cfdl_mfac_ncp_td3.main ablate --trials 20

Compare the hybrid controller against the classical benchmarks on one task::

    python -m cfdl_mfac_ncp_td3.main evaluate --trajectory lawnmower
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .config import default_config
from .controllers import HybridController
from .benchmark import PIDController, SMCController, BacksteppingController, MPCController
from .evaluation import rollout, evaluate_controllers
from . import visualization as viz


def _build_controllers(cfg, td3_agent=None):
    tau_max = list(cfg.thruster.tau_max)
    return {
        "Hybrid": HybridController(cfg, td3_agent=td3_agent),
        "PID": PIDController(tau_max=tau_max),
        "SMC": SMCController(tau_max=tau_max),
        "Backstepping": BacksteppingController(tau_max=tau_max),
        "MPC": MPCController(tau_max=tau_max),
    }


def cmd_demo(args):
    cfg = default_config()
    ctrl = HybridController(cfg)
    res = rollout(ctrl, trajectory=args.trajectory, config=cfg, current=True, seed=args.seed)
    m = res["metrics"]
    print(f"\n=== Hybrid controller on '{args.trajectory}' ===")
    for k, v in m.items():
        print(f"  {k:<16s}: {v}")
    out = args.out
    viz.plot_trajectory_3d(res["log"], os.path.join(out, "trajectory_3d.png"))
    viz.plot_tracking_errors(res["log"], os.path.join(out, "tracking_errors.png"))
    viz.plot_control_effort(res["log"], os.path.join(out, "control_effort.png"))
    viz.plot_blending(res["log"], os.path.join(out, "rl_authority.png"))
    print(f"\nPlots written to {os.path.abspath(out)}/")


def cmd_evaluate(args):
    cfg = default_config()
    controllers = _build_controllers(cfg)
    results = evaluate_controllers(controllers, trajectory=args.trajectory,
                                   config=cfg, current=True, seed=args.seed)
    print(f"\n=== Controller comparison on '{args.trajectory}' (RMSE) ===")
    rows = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["rmse"])
    for name, res in rows:
        m = res["metrics"]
        print(f"  {name:<14s} rmse={m['rmse']:.4f}  iae={m['iae']:.3f}  "
              f"energy={m['control_energy']:.1f}")


def cmd_train(args):
    from .training import train_td3
    cfg = default_config()
    out = train_td3(total_steps=args.steps, trajectory=args.trajectory,
                    config=cfg, curriculum=True)
    if args.save:
        out["agent"].save(args.save)
        print(f"Saved trained TD3 agent to {args.save}")


def cmd_ablate(args):
    from .experiments import run_ablation
    cfg = default_config()
    agent = None
    if args.load:
        from .rl import TD3
        from .environment import AUVEnv
        env = AUVEnv(cfg)
        agent = TD3(env.obs_dim, env.act_dim, config=cfg.td3)
        agent.load(args.load)
    print(f"\n=== Ablation study ({args.trials} trials, metric=rmse) ===")
    rep = run_ablation(cfg, td3_agent=agent, n_trials=args.trials,
                       trajectory=args.trajectory, fault_prob=args.fault_prob)
    print("\nPairwise comparison vs Hybrid:")
    for name, c in rep["comparisons"].items():
        sig = "*" if c["significant"] else " "
        print(f"  Hybrid vs {name:<14s} d={c['ttest']['cohens_d']:+.2f} "
              f"p={c['ttest']['p']:.3g} {sig}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CFDL-MFAC-NCP-TD3 hybrid AUV controller")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="quick end-to-end demo with plots")
    d.add_argument("--trajectory", default="helix")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="outputs")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("evaluate", help="compare hybrid vs benchmarks")
    e.add_argument("--trajectory", default="sinusoidal")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("train", help="train the TD3 policy")
    t.add_argument("--steps", type=int, default=50_000)
    t.add_argument("--trajectory", default="sinusoidal")
    t.add_argument("--save", default=None)
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("ablate", help="run the ablation study")
    a.add_argument("--trials", type=int, default=20)
    a.add_argument("--trajectory", default="sinusoidal")
    a.add_argument("--fault-prob", type=float, default=0.2, dest="fault_prob")
    a.add_argument("--load", default=None, help="trained TD3 checkpoint")
    a.set_defaults(func=cmd_ablate)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
