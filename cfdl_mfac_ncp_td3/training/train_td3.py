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
              max_episodes: int | None = None, eval_every_episodes: int | None = None,
              make_env=None, seed: int = 0, verbose: bool = True) -> dict:
    """Train a TD3 agent on the AUV environment.

    Training runs until ``total_steps`` environment steps, or until
    ``max_episodes`` completed episodes if that is given (episode-budget mode,
    with the curriculum annealed over episodes).  ``make_env(config, trajectory,
    difficulty, randomize, fault_prob, seed)`` overrides the environment factory
    (e.g. the residual-RL env).  Returns a dict with the trained ``agent`` and
    the training history.
    """

    cfg = config or default_config()
    rng = np.random.default_rng(seed)
    episode_budget = max_episodes is not None
    if episode_budget:
        total_steps = 10 ** 12  # effectively unbounded; episodes terminate it
    factory = make_env or (lambda **kw: AUVEnv(**kw))
    env = factory(config=cfg, trajectory=trajectory, difficulty=0.1 if curriculum else 0.6,
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
                # Anneal difficulty over the training budget (episodes or steps).
                frac = ep_idx / max_episodes if episode_budget else step / total_steps
                env.set_difficulty(min(1.0, 0.1 + 0.9 * frac))

            if eval_every_episodes and ep_idx % eval_every_episodes == 0:
                er = evaluate_policy(agent, cfg, trajectory, n=3)
                history["eval_return"].append((step, er))
                if verbose:
                    print(f"[train] episode {ep_idx:>5d}  step {step:>8d}  "
                          f"eval_return {er:10.2f}", flush=True)

            if episode_budget and ep_idx >= max_episodes:
                break
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if eval_every and not episode_budget and step % eval_every == 0:
            er = evaluate_policy(agent, cfg, trajectory, n=3, make_env=make_env)
            history["eval_return"].append((step, er))
            if verbose:
                print(f"[train] step {step:>7d}  eval_return {er:8.2f}  "
                      f"episodes {ep_idx}", flush=True)

    return {"agent": agent, "history": history, "env": env, "episodes": ep_idx,
            "steps": step}


def evaluate_policy(agent: TD3, config: ExperimentConfig, trajectory: str,
                    n: int = 3, make_env=None) -> float:
    """Mean deterministic episode return over ``n`` randomized episodes."""

    factory = make_env or (lambda **kw: AUVEnv(**kw))
    env = factory(config=config, trajectory=trajectory, difficulty=0.6,
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
