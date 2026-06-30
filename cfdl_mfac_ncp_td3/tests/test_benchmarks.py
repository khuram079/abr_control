"""Stage 7 validation: classical benchmark controllers.

Each controller is closed around the full nonlinear REMUS plant and must
regulate the vehicle to a constant pose setpoint with bounded effort.  These
are closed-loop stability/convergence checks, not tuning-optimality claims.
"""

import numpy as np
import pytest

from cfdl_mfac_ncp_td3.config import SimConfig, ThrusterConfig
from cfdl_mfac_ncp_td3.dynamics import REMUS6DOF, REMUSParams, ThrusterModel
from cfdl_mfac_ncp_td3.benchmark import (
    PIDController,
    SMCController,
    BacksteppingController,
    MPCController,
    pose_error,
)

SETPOINT = np.array([2.0, 1.0, 1.5, 0.0, 0.1, 0.5])


def _closed_loop(controller, steps=1200, dt=0.05, with_actuator=True):
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=dt))
    veh.reset()
    th = ThrusterModel(ThrusterConfig(tau_max=(80,) * 6, time_constant=0.05)) if with_actuator else None
    controller.reset() if hasattr(controller, "reset") else None
    errs = []
    for _ in range(steps):
        tau_cmd = controller.control(veh.eta, veh.nu, SETPOINT, dt=dt)
        tau = th.step(tau_cmd, dt) if th is not None else tau_cmd
        veh.step(tau)
        errs.append(np.linalg.norm(pose_error(SETPOINT, veh.eta)))
    return np.array(errs), veh


@pytest.mark.parametrize("make", [
    lambda: PIDController(tau_max=[80] * 6),
    lambda: SMCController(tau_max=[80] * 6),
    lambda: BacksteppingController(tau_max=[80] * 6),
    lambda: MPCController(tau_max=[80] * 6),
])
def test_benchmark_regulates_setpoint(make):
    ctrl = make()
    errs, veh = _closed_loop(ctrl)
    assert np.all(np.isfinite(errs)), f"{ctrl.name} diverged (non-finite)"
    # Error must shrink substantially and settle small.
    assert errs[-1] < 0.2, f"{ctrl.name} steady-state error {errs[-1]:.3f}"
    assert errs[-1] < 0.3 * errs[0], f"{ctrl.name} failed to reduce error"


@pytest.mark.parametrize("make", [
    lambda: PIDController(tau_max=[80] * 6),
    lambda: SMCController(tau_max=[80] * 6),
    lambda: BacksteppingController(tau_max=[80] * 6),
])
def test_benchmark_stable_no_actuator_model(make):
    """Without actuator lag the ideal-wrench loop must also stay stable."""

    ctrl = make()
    errs, _ = _closed_loop(ctrl, steps=800, with_actuator=False)
    assert np.all(np.isfinite(errs))
    assert errs[-1] < 0.2


def test_mpc_gain_is_cached():
    mpc = MPCController()
    k1 = mpc._gain(0, 0.05)
    k2 = mpc._gain(0, 0.05)
    assert k1 is k2  # cached object identity
    assert np.all(np.isfinite(k1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
