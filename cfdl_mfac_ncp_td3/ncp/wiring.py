"""Neural Circuit Policy (NCP) wiring.

Builds the sparse, layered connectivity of an NCP following Lechner et al.,
"Neural circuit policies enabling auditable autonomy" (Nature MI, 2020):

    sensory --> inter --> command <--> command --> motor

Each synapse carries a polarity (+1 excitatory / -1 inhibitory).  The wiring
produces boolean/sign masks consumed by the :class:`LiquidNetwork` LTC cell so
that only the wired synapses have trainable conductances.
"""

from __future__ import annotations

import numpy as np

from ..config import NCPConfig


class NCPWiring:
    """Sparse four-population NCP wiring with signed adjacency masks."""

    def __init__(self, config: NCPConfig | None = None):
        self.cfg = config or NCPConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.n_sensory = self.cfg.n_sensory
        self.n_inter = self.cfg.n_inter
        self.n_command = self.cfg.n_command
        self.n_motor = self.cfg.n_motor
        self.n_units = self.n_inter + self.n_command + self.n_motor

        # Index ranges of each population within the state vector.
        self.inter_ids = np.arange(0, self.n_inter)
        self.command_ids = np.arange(self.n_inter, self.n_inter + self.n_command)
        self.motor_ids = np.arange(
            self.n_inter + self.n_command, self.n_units
        )

        # Signed adjacency: sensory->units and units->units.
        self.sensory_adj = np.zeros((self.n_sensory, self.n_units), dtype=np.float32)
        self.adj = np.zeros((self.n_units, self.n_units), dtype=np.float32)
        self._build()

    # ------------------------------------------------------------------ #
    def _polarity(self) -> int:
        return int(self.rng.choice([-1, 1]))

    def _connect_sensory(self, src: int, dst: int) -> None:
        self.sensory_adj[src, dst] = self._polarity()

    def _connect(self, src: int, dst: int) -> None:
        self.adj[src, dst] = self._polarity()

    def _build(self) -> None:
        c = self.cfg
        # 1) sensory -> inter (each sensory neuron fans out to a few inter).
        for s in range(self.n_sensory):
            targets = self.rng.choice(
                self.inter_ids, size=min(c.sensory_fanout, self.n_inter), replace=False
            )
            for t in targets:
                self._connect_sensory(s, int(t))
        # Guarantee every inter neuron receives at least one sensory synapse.
        for t in self.inter_ids:
            if not np.any(self.sensory_adj[:, t]):
                self._connect_sensory(int(self.rng.integers(self.n_sensory)), int(t))

        # 2) inter -> command.
        for s in self.inter_ids:
            targets = self.rng.choice(
                self.command_ids,
                size=min(c.inter_fanout, self.n_command),
                replace=False,
            )
            for t in targets:
                self._connect(int(s), int(t))
        for t in self.command_ids:
            if not np.any(self.adj[self.inter_ids, t]):
                self._connect(int(self.rng.choice(self.inter_ids)), int(t))

        # 3) command <-> command recurrence.
        for _ in range(c.recurrent_command * self.n_command):
            s = int(self.rng.choice(self.command_ids))
            t = int(self.rng.choice(self.command_ids))
            self._connect(s, t)

        # 4) command -> motor.
        for t in self.motor_ids:
            srcs = self.rng.choice(
                self.command_ids,
                size=min(c.motor_fanin, self.n_command),
                replace=False,
            )
            for s in srcs:
                self._connect(int(s), int(t))

    # ------------------------------------------------------------------ #
    def adjacency_masks(self):
        """Return ``(sensory_mask, recurrent_mask)`` sign matrices (float32)."""

        return self.sensory_adj.copy(), self.adj.copy()

    def summary(self) -> dict:
        return {
            "n_sensory": self.n_sensory,
            "n_inter": self.n_inter,
            "n_command": self.n_command,
            "n_motor": self.n_motor,
            "n_units": self.n_units,
            "sensory_synapses": int(np.count_nonzero(self.sensory_adj)),
            "recurrent_synapses": int(np.count_nonzero(self.adj)),
        }
