"""Twin critic networks for TD3 (clipped double-Q learning)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden, out_dim: int) -> nn.Sequential:
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Two independent state-action value networks ``Q1`` and ``Q2``."""

    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, hidden, 1)
        self.q2 = _mlp(state_dim + action_dim, hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))
