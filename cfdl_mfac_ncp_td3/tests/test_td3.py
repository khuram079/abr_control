"""Stage 5 validation: TD3 agent.

Verifies action bounds/shapes, replay-buffer mechanics, delayed policy
updates, and that the agent actually *learns* on a simple deterministic
control task (reach-the-target point mass) -- the standard smoke test for an
actor-critic implementation.
"""

import numpy as np
import torch
import pytest

from cfdl_mfac_ncp_td3.config import TD3Config
from cfdl_mfac_ncp_td3.rl import TD3, ReplayBuffer


def test_replay_buffer_roundtrip():
    buf = ReplayBuffer(3, 2, capacity=10)
    for i in range(15):  # overflow to test wrap-around
        buf.add(np.full(3, i), np.full(2, i), float(i), np.full(3, i + 1), i % 2)
    assert len(buf) == 10
    s, a, r, s2, d = buf.sample(4)
    assert s.shape == (4, 3) and a.shape == (4, 2) and r.shape == (4, 1)


def test_action_bounds_and_shape():
    agent = TD3(6, 3, max_action=2.0, config=TD3Config())
    a = agent.select_action(np.zeros(6), noise=0.5)
    assert a.shape == (3,)
    assert np.all(np.abs(a) <= 2.0 + 1e-9)


def test_train_returns_none_until_warm():
    agent = TD3(4, 2, config=TD3Config(batch_size=32))
    assert agent.train() is None  # empty buffer
    for _ in range(40):
        agent.store(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), 0.0)
    out = agent.train()
    assert out is not None and "critic_loss" in out


def test_delayed_policy_update():
    agent = TD3(4, 2, config=TD3Config(batch_size=16, policy_delay=2))
    for _ in range(64):
        agent.store(np.random.randn(4), np.random.randn(2), 1.0, np.random.randn(4), 0.0)
    actor_updates = 0
    for _ in range(10):
        out = agent.train()
        if out["actor_loss"] is not None:
            actor_updates += 1
    # With policy_delay=2 the actor updates roughly half as often as the critic.
    assert 3 <= actor_updates <= 7


def test_td3_learns_point_mass_reach():
    """1-D reach task: state=(pos,target), action=velocity, reward=-|pos-target|.

    A correct TD3 must learn to move toward the target, raising mean episode
    return well above a random policy baseline.
    """

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = TD3Config(batch_size=128, warmup_steps=500, exploration_noise=0.2,
                    actor_lr=1e-3, critic_lr=1e-3)
    agent = TD3(state_dim=2, action_dim=1, max_action=1.0, config=cfg)

    def run_episode(explore: bool):
        pos = np.random.uniform(-1, 1)
        target = np.random.uniform(-1, 1)
        total = 0.0
        s = np.array([pos, target], dtype=np.float32)
        for _ in range(20):
            if explore and agent.buffer.size < cfg.warmup_steps:
                a = np.random.uniform(-1, 1, size=1)
            else:
                a = agent.select_action(s, noise=cfg.exploration_noise if explore else None)
            pos = np.clip(pos + 0.1 * a[0], -1.5, 1.5)
            r = -abs(pos - target)
            s2 = np.array([pos, target], dtype=np.float32)
            if explore:
                agent.store(s, a, r, s2, 0.0)
                agent.train()
            s = s2
            total += r
        return total

    for _ in range(400):
        run_episode(explore=True)

    returns = [run_episode(explore=False) for _ in range(30)]
    mean_return = float(np.mean(returns))
    # Random policy averages well below -5 on this task; a learned policy that
    # drives toward the target scores substantially higher.
    assert mean_return > -5.0, f"TD3 did not learn (mean return {mean_return:.2f})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
