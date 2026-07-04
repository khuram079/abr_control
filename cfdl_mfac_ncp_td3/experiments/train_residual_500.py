"""Train the residual-RL policy (500 episodes) on top of the strong hybrid.

The learned TD3 policy outputs a bounded correction added to the strong
CFDL-MFAC + SMC-attitude + damping controller (see AUVResidualEnv.STRONG_BASELINE).
A zero policy reproduces the strong baseline exactly, so training can only
improve on an already-competitive controller.  Saves the checkpoint + learning
curve; evaluation and stability analysis are run separately.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import default_config, SimConfig
from ..environment import AUVResidualEnv
from ..training import train_td3

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "residual_v2")
RESIDUAL_SCALE = 0.25


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = default_config()
    cfg.sim = SimConfig(dt=0.05, horizon=30.0)  # 600-step training episodes

    def make_env(**kw):
        return AUVResidualEnv(residual_scale=RESIDUAL_SCALE, **kw)

    print(f"{'='*72}\nTRAIN RESIDUAL RL (500 episodes) on the strong hybrid baseline\n"
          f"   residual_scale={RESIDUAL_SCALE}\n{'='*72}", flush=True)
    t0 = time.time()
    out = train_td3(trajectory="sinusoidal", config=cfg, fault_prob=0.2, curriculum=True,
                    max_episodes=500, eval_every_episodes=20, make_env=make_env,
                    seed=0, verbose=True)
    dt = time.time() - t0
    agent, hist = out["agent"], out["history"]
    print(f"Training done: {out['episodes']} episodes / {out['steps']} steps in "
          f"{dt/60:.1f} min.", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ep = np.array(hist["episode_return"])
    ax[0].plot(ep, alpha=0.25, color="C0")
    if len(ep) >= 20:
        k = max(1, len(ep) // 40)
        ax[0].plot(np.arange(len(ep) - k + 1) + k // 2,
                   np.convolve(ep, np.ones(k) / k, mode="valid"), "C0", lw=2)
    ax[0].set_title("Residual-RL training return (500 ep)"); ax[0].set_xlabel("episode")
    if hist["eval_return"]:
        ev = np.array(hist["eval_return"])
        ax[1].plot(ev[:, 0], ev[:, 1], "C2-o", ms=3)
        ax[1].set_title("Deterministic eval return"); ax[1].set_xlabel("env step")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "training_curve.png")); plt.close(fig)
    agent.save(os.path.join(RESULTS_DIR, "residual_agent.pt"))
    print(f"Saved residual_agent.pt + training_curve.png -> {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
