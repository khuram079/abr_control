"""TD3 training loop with curriculum learning and domain randomization.

Trains the TD3 policy that serves as the robust component of the hybrid
controller.  Difficulty is annealed from easy to hard over training
(curriculum), and each episode randomizes mass / current / faults
(domain randomization) so the learned policy generalises to the conditions
the supervisor will hand authority to it under.
"""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig, default_config
from ..environment import AUVEnv
from ..rl import TD3


def train_td3(total_steps: int = 50_000, trajectory: str = "sinusoidal",
              config: ExperimentConfig | None = None, fault_prob: float = 0.1,
              curriculum: bool = True, eval_every: int = 5000,
              seed: int = 0, verbose: bool = True) -> dict:
    """Train a TD3 agent on the AUV environment.

    Returns a dict with the trained ``agent`` and the training history.
    """

    cfg = config or default_config()
    rng = np.random.default_rng(seed)
    env = AUVEnv(cfg, trajectory=trajectory, difficulty=0.1 if curriculum else 0.6,
                 randomize=True, fault_prob=fault_prob, seed=seed)
    agent = TD3(env.obs_dim, env.act_dim, max_action=1.0, config=cfg.td3)

    history = {"step": [], "episode_return": [], "eval_return": []}
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_idx = 0.0, 0, 0
    warmup = cfg.td3.warmup_steps

    for step in range(1, total_steps + 1):
        if step < warmup:
            action = rng.uniform(-1, 1, size=env.act_dim)
        else:
            action = agent.select_action(obs, noise=cfg.td3.exploration_noise)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, float(terminated))
        agent.train()
        obs = next_obs
        ep_ret += reward
        ep_len += 1

        if done:
            history["step"].append(step)
            history["episode_return"].append(ep_ret)
            ep_idx += 1
            if curriculum:
                # Linearly ramp difficulty across the run.
                env.set_difficulty(min(1.0, 0.1 + 0.9 * step / total_steps))
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if eval_every and step % eval_every == 0:
            er = evaluate_policy(agent, cfg, trajectory, n=3)
            history["eval_return"].append((step, er))
            if verbose:
                print(f"[train] step {step:>7d}  eval_return {er:8.2f}  "
                      f"episodes {ep_idx}")

    return {"agent": agent, "history": history, "env": env}


def evaluate_policy(agent: TD3, config: ExperimentConfig, trajectory: str,
                    n: int = 3) -> float:
    """Mean deterministic episode return over ``n`` randomized episodes."""

    env = AUVEnv(config, trajectory=trajectory, difficulty=0.6,
                 randomize=True, fault_prob=0.0, seed=12345)
    rets = []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i)
        done, ret = False, 0.0
        while not done:
            a = agent.select_action(obs, noise=None)
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))
