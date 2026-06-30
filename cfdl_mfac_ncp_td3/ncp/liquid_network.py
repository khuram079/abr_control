"""Liquid Time-Constant (LTC) network wired as a Neural Circuit Policy.

Implements the conductance-based LTC neuron model of Hasani et al., "Liquid
Time-constant Networks" (AAAI 2021), restricted to the sparse NCP wiring.  The
membrane dynamics of each neuron are

    cm dx/dt = gleak (vleak - x) + sum_j g_ij(x) (E_ij - x),
    g_ij(x)  = w_ij * sigmoid(sigma_ij (x_j - mu_ij)),

solved with the numerically stable fused semi-implicit Euler scheme used in
the reference implementation (several sub-steps per control step).  Only wired
synapses (per :class:`NCPWiring`) carry a conductance; the reversal potential
``E_ij`` is fixed to the synapse polarity.

The motor neuron(s) are squashed through a sigmoid so the network output is a
bounded supervisory signal (e.g. a controller-blending coefficient in [0, 1]).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import NCPConfig
from .wiring import NCPWiring


class LiquidNetwork(nn.Module):
    """Wired LTC cell producing a bounded supervisory output."""

    def __init__(self, config: NCPConfig | None = None, wiring: NCPWiring | None = None):
        super().__init__()
        self.cfg = config or NCPConfig()
        self.wiring = wiring or NCPWiring(self.cfg)
        n_units = self.wiring.n_units
        n_sensory = self.wiring.n_sensory
        torch.manual_seed(self.cfg.seed)

        sensory_mask, recurrent_mask = self.wiring.adjacency_masks()
        self.register_buffer("sensory_sign", torch.tensor(sensory_mask))
        self.register_buffer("recurrent_sign", torch.tensor(recurrent_mask))
        self.register_buffer("sensory_mask", (torch.tensor(sensory_mask) != 0).float())
        self.register_buffer("recurrent_mask", (torch.tensor(recurrent_mask) != 0).float())

        # Passive membrane parameters (kept positive via softplus at use time).
        self.gleak = nn.Parameter(torch.ones(n_units))
        self.vleak = nn.Parameter(torch.zeros(n_units))
        self.cm = nn.Parameter(torch.ones(n_units))

        # Recurrent synapse parameters.
        self.w = nn.Parameter(torch.rand(n_units, n_units) * 0.5 + 0.1)
        self.sigma = nn.Parameter(torch.rand(n_units, n_units) * 2.0 + 3.0)
        self.mu = nn.Parameter(torch.zeros(n_units, n_units))

        # Sensory synapse parameters.
        self.sensory_w = nn.Parameter(torch.rand(n_sensory, n_units) * 0.5 + 0.1)
        self.sensory_sigma = nn.Parameter(torch.rand(n_sensory, n_units) * 2.0 + 3.0)
        self.sensory_mu = nn.Parameter(torch.zeros(n_sensory, n_units))

        # Input/output affine maps for conditioning and read-out scaling.
        self.input_w = nn.Parameter(torch.ones(n_sensory))
        self.input_b = nn.Parameter(torch.zeros(n_sensory))

        self.n_units = n_units
        self.n_sensory = n_sensory

    # ------------------------------------------------------------------ #
    @staticmethod
    def _synapse(v_pre: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # v_pre: (batch, n_pre) -> (batch, n_pre, 1); mu/sigma: (n_pre, n_post)
        v = v_pre.unsqueeze(-1)
        return torch.sigmoid(sigma * (v - mu))

    def _ode_solve(self, inputs: torch.Tensor, state: torch.Tensor, dt: float) -> torch.Tensor:
        cm = torch.nn.functional.softplus(self.cm)
        gleak = torch.nn.functional.softplus(self.gleak)
        w = torch.nn.functional.softplus(self.w) * self.recurrent_mask
        sensory_w = torch.nn.functional.softplus(self.sensory_w) * self.sensory_mask
        erev = self.recurrent_sign
        sensory_erev = self.sensory_sign

        # Sensory contribution is constant across the sub-steps.
        sens_act = self._synapse(inputs, self.sensory_mu, self.sensory_sigma)
        sens_act = sens_act * sensory_w  # (batch, n_sensory, n_units)
        sens_num = torch.sum(sens_act * sensory_erev, dim=1)  # (batch, n_units)
        sens_den = torch.sum(sens_act, dim=1)

        sub_dt = dt / self.cfg.ode_unfolds
        cm_t = cm / sub_dt
        v = state
        for _ in range(self.cfg.ode_unfolds):
            rec_act = self._synapse(v, self.mu, self.sigma) * w  # (batch,n_units,n_units)
            rec_num = torch.sum(rec_act * erev, dim=1)
            rec_den = torch.sum(rec_act, dim=1)
            numerator = cm_t * v + gleak * self.vleak + sens_num + rec_num
            denominator = cm_t + gleak + sens_den + rec_den
            v = numerator / denominator
        return v

    # ------------------------------------------------------------------ #
    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_units)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor | None = None, dt: float = 0.05):
        """Advance one control step.

        Parameters
        ----------
        inputs:
            Sensory features ``(batch, n_sensory)``.
        state:
            Previous neuron state ``(batch, n_units)`` (zeros if ``None``).
        dt:
            Control step [s].

        Returns
        -------
        output, state:
            ``output`` is the squashed motor read-out in ``(0, 1)`` of shape
            ``(batch, n_motor)``; ``state`` is the new neuron state.
        """

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if state is None:
            state = self.init_state(inputs.shape[0])
        inputs = inputs * self.input_w + self.input_b
        state = self._ode_solve(inputs, state, dt)
        motor = state[:, self.wiring.motor_ids]
        output = torch.sigmoid(motor)
        return output, state
