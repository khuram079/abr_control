"""Fixed-size circular replay buffer for off-policy RL."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Pre-allocated ring buffer of transitions ``(s, a, r, s', done)``."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000,
                 device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done) -> None:
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)
        return to(self.state), to(self.action), to(self.reward), to(self.next_state), to(self.done)

    def __len__(self) -> int:
        return self.size
