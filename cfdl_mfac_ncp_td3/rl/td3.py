"""Twin-Delayed Deep Deterministic Policy Gradient (TD3) agent.

Implements the three tricks of Fujimoto et al., "Addressing Function
Approximation Error in Actor-Critic Methods" (ICML 2018):

1. clipped double-Q targets (take the minimum of two critics),
2. target-policy smoothing (clipped noise added to the target action), and
3. delayed policy and target updates.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..config import TD3Config
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer


class TD3:
    """TD3 agent with a symmetric action box ``[-max_action, max_action]``."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 config: TD3Config | None = None):
        self.cfg = config or TD3Config()
        self.device = torch.device(self.cfg.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(max_action)

        self.actor = Actor(state_dim, action_dim, self.cfg.hidden_sizes, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)

        self.critic = Critic(state_dim, action_dim, self.cfg.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.buffer = ReplayBuffer(state_dim, action_dim, self.cfg.buffer_size, self.cfg.device)
        self.total_it = 0

    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, noise: float | None = None) -> np.ndarray:
        """Return an action; add exploration noise when ``noise`` is given."""

        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
        with torch.no_grad():
            a = self.actor(s).cpu().numpy().flatten()
        if noise:
            a = a + np.random.normal(0, noise * self.max_action, size=self.action_dim)
        return np.clip(a, -self.max_action, self.max_action)

    def store(self, s, a, r, s2, done) -> None:
        self.buffer.add(s, a, r, s2, done)

    # ------------------------------------------------------------------ #
    def train(self, batch_size: int | None = None) -> dict | None:
        """One optimisation step; returns loss diagnostics (or ``None``)."""

        bs = batch_size or self.cfg.batch_size
        if len(self.buffer) < bs:
            return None
        self.total_it += 1
        c = self.cfg
        state, action, reward, next_state, done = self.buffer.sample(bs)

        with torch.no_grad():
            # Target-policy smoothing: clipped noise on the target action.
            noise = (torch.randn_like(action) * c.policy_noise).clamp(-c.noise_clip, c.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            tq1, tq2 = self.critic_target(next_state, next_action)
            target_q = torch.min(tq1, tq2)  # clipped double-Q
            target_q = reward + (1.0 - done) * c.gamma * target_q

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_val = None
        if self.total_it % c.policy_delay == 0:
            # Delayed policy update: maximise Q1 of the current policy.
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}

    def _soft_update(self, net, target) -> None:
        tau = self.cfg.tau
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
