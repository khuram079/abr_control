"""Stage 4 validation: NCP wiring and Liquid Time-Constant network.

Checks the structural invariants of the wiring (layered sparsity, full
coverage) and the functional properties of the LTC cell (output bounded in
(0,1), correct shapes, gradients flow, and the cell can learn a simple
monotone mapping from a context feature to the blending output).
"""

import numpy as np
import torch
import pytest

from cfdl_mfac_ncp_td3.config import NCPConfig
from cfdl_mfac_ncp_td3.ncp import NCPWiring, LiquidNetwork


def test_wiring_is_layered_and_covered():
    w = NCPWiring(NCPConfig())
    # Sensory neurons must not connect directly to motor neurons.
    assert np.all(w.sensory_adj[:, w.motor_ids] == 0)
    # Every inter neuron receives at least one sensory synapse.
    assert np.all(np.any(w.sensory_adj[:, w.inter_ids] != 0, axis=0))
    # Every command neuron receives at least one inter synapse.
    assert np.all(np.any(w.adj[np.ix_(w.inter_ids, w.command_ids)] != 0, axis=0))
    # Motor neurons are driven only by command neurons.
    non_command = np.concatenate([w.inter_ids])
    assert np.all(w.adj[np.ix_(non_command, w.motor_ids)] == 0)


def test_wiring_polarities_are_signed():
    w = NCPWiring(NCPConfig())
    vals = w.adj[w.adj != 0]
    assert set(np.unique(vals)).issubset({-1.0, 1.0})


def test_ltc_output_shape_and_bounds():
    net = LiquidNetwork(NCPConfig(n_sensory=8, n_motor=1))
    x = torch.randn(5, 8)
    out, state = net(x, dt=0.05)
    assert out.shape == (5, 1)
    assert state.shape == (5, net.n_units)
    assert torch.all(out > 0) and torch.all(out < 1)


def test_ltc_handles_unbatched_input():
    net = LiquidNetwork(NCPConfig(n_sensory=6, n_motor=1))
    out, state = net(torch.randn(6))
    assert out.shape == (1, 1)


def test_ltc_state_is_finite_under_large_input():
    net = LiquidNetwork(NCPConfig(n_sensory=8))
    out, state = net(torch.full((1, 8), 50.0))
    assert torch.all(torch.isfinite(state)) and torch.all(torch.isfinite(out))


def test_ltc_gradients_flow():
    net = LiquidNetwork(NCPConfig(n_sensory=8, n_motor=1))
    out, _ = net(torch.randn(4, 8))
    loss = out.mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_ltc_can_learn_monotone_mapping():
    """Train the LTC to map a scalar context onto the blend output.

    Target: high first feature -> output near 1, low -> near 0.  A modest
    number of steps must reduce the loss substantially.
    """

    torch.manual_seed(0)
    net = LiquidNetwork(NCPConfig(n_sensory=4, n_inter=8, n_command=6, n_motor=1))
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    losses = []
    for _ in range(400):
        x = torch.rand(64, 4) * 2 - 1
        target = (x[:, :1] > 0).float()  # 1 if first feature positive
        out, _ = net(x)
        loss = torch.mean((out - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # The conductance-based LTC has an early plateau near 0.5; once it breaks
    # free it must drive the variance-0.25 baseline well down.
    assert losses[-1] < 0.15, f"NCP failed to learn: {losses[0]:.3f}->{losses[-1]:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
