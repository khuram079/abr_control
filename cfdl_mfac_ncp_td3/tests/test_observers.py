"""Stage 3 validation: state / disturbance / fault observers.

Each observer is checked against a ground-truth simulation: the EKF reduces
measurement noise and reconstructs unmeasured velocities; the NDOB converges
to an injected constant wrench disturbance; the fault observer identifies an
injected actuator-effectiveness loss.
"""

import numpy as np
import pytest

from cfdl_mfac_ncp_td3.config import SimConfig, ObserverConfig
from cfdl_mfac_ncp_td3.dynamics import REMUS6DOF, REMUSParams
from cfdl_mfac_ncp_td3.observers import StateObserver, DisturbanceObserver, FaultObserver


def test_disturbance_observer_converges_to_constant_wrench():
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.02))
    veh.reset(nu=np.array([1.0, 0, 0, 0, 0, 0]))
    dob = DisturbanceObserver(veh.p, ObserverConfig(ndob_gain=8.0))
    d_true = np.array([3.0, -2.0, 1.0, 0.0, 0.5, -0.4])
    tau = np.zeros(6)
    for _ in range(2000):
        dob.update(veh.eta, veh.nu, tau, dt=0.02)
        veh.step(tau, tau_dist=d_true)
    err = np.linalg.norm(dob.d_hat - d_true) / np.linalg.norm(d_true)
    assert err < 0.1, f"NDOB failed to converge, rel-err={err:.3f}, est={dob.d_hat}"


def test_disturbance_observer_zero_when_no_disturbance():
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.02))
    veh.reset(nu=np.array([0.8, 0.1, 0, 0, 0, 0]))
    dob = DisturbanceObserver(veh.p, ObserverConfig(ndob_gain=8.0))
    tau = np.array([5.0, 0, 0, 0, 0, 0])
    for _ in range(1500):
        dob.update(veh.eta, veh.nu, tau, dt=0.02)
        veh.step(tau)
    assert np.linalg.norm(dob.d_hat) < 0.5, f"spurious disturbance {dob.d_hat}"


def test_fault_observer_identifies_effectiveness_loss():
    fo = FaultObserver(6, ObserverConfig(fault_adapt_gain=1.5, fault_forgetting=0.99))
    theta_true = np.array([1.0, 0.5, 0.8, 1.0, 0.3, 1.0])
    rng = np.random.default_rng(0)
    for _ in range(3000):
        tau_cmd = rng.uniform(-10, 10, size=6)
        tau_real = theta_true * tau_cmd  # realised (faulted) wrench
        fo.update(tau_cmd, tau_real)
    err = np.linalg.norm(fo.theta - theta_true)
    assert err < 0.1, f"fault estimate {fo.theta} vs {theta_true}"


def test_fault_observer_freezes_idle_channels():
    fo = FaultObserver(6)
    fo.theta[2] = 0.4
    # No command on any channel -> nothing should change.
    fo.update(np.zeros(6), np.zeros(6))
    assert np.isclose(fo.theta[2], 0.4)


def test_state_observer_denoises_full_measurement():
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.05))
    veh.reset(nu=np.array([1.0, 0, 0.05, 0, 0, 0.02]))
    obs = StateObserver(REMUS6DOF(REMUSParams(), SimConfig(dt=0.05)),
                        ObserverConfig(process_noise=1e-4, meas_noise=1e-2))
    rng = np.random.default_rng(1)
    tau = np.array([6.0, 0, 0, 0, 0, 0.5])
    raw_err, est_err = [], []
    for _ in range(300):
        veh.step(tau)
        meas = veh.state + rng.normal(0, 0.05, size=12)
        obs.step(tau, meas)
        raw_err.append(np.linalg.norm(meas - veh.state))
        est_err.append(np.linalg.norm(obs.x - veh.state))
    # The filtered estimate must be markedly better than the raw measurement.
    assert np.mean(est_err[50:]) < 0.6 * np.mean(raw_err[50:])


def test_state_observer_reconstructs_velocity_from_pose_only():
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.05))
    veh.reset(nu=np.array([1.0, 0, 0, 0, 0, 0]))
    H = np.zeros((6, 12))
    H[:6, :6] = np.eye(6)  # measure pose only
    obs = StateObserver(REMUS6DOF(REMUSParams(), SimConfig(dt=0.05)),
                        ObserverConfig(process_noise=1e-3, meas_noise=1e-3), H=H)
    rng = np.random.default_rng(2)
    tau = np.array([6.0, 0, 0, 0, 0, 0])
    for _ in range(400):
        veh.step(tau)
        meas = veh.eta + rng.normal(0, 0.01, size=6)
        obs.step(tau, meas)
    # Surge velocity should be recovered despite never being measured.
    assert abs(obs.nu[0] - veh.nu[0]) < 0.2, f"u est {obs.nu[0]} vs {veh.nu[0]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
