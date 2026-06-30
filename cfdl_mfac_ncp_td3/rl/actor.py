"""Deterministic actor network for TD3."""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """MLP policy mapping a state to a bounded continuous action.

    The output is ``max_action * tanh(.)`` so actions respect a symmetric box.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256), max_action=1.0):
        super().__init__()
        self.max_action = float(max_action)
        layers, last = [], state_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))
