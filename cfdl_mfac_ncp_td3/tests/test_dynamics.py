"""Stage 1 validation: physical sanity of the 6-DOF REMUS simulator.

These tests do not check against experimental data; they verify structural
and energetic properties that any correct rigid-body marine model must
satisfy (symmetry, positive-definiteness, energy dissipation, equilibrium,
kinematic consistency).
"""

import numpy as np
import pytest

from cfdl_mfac_ncp_td3.config import SimConfig
from cfdl_mfac_ncp_td3.dynamics import (
    REMUSParams,
    REMUS6DOF,
    OceanCurrent,
    ThrusterModel,
    AdaptiveAllocator,
    jacobian,
    rotation_matrix,
    skew,
)
from cfdl_mfac_ncp_td3.config import CurrentConfig, ThrusterConfig


def test_mass_matrix_symmetric_pd():
    p = REMUSParams()
    M = p.mass_matrix()
    assert np.allclose(M, M.T, atol=1e-9), "mass matrix must be symmetric"
    eig = np.linalg.eigvalsh(M)
    assert np.all(eig > 0), f"mass matrix must be PD, got eigs {eig}"


def test_coriolis_skew_symmetric_energy():
    p = REMUSParams()
    rng = np.random.default_rng(0)
    for _ in range(20):
        nu = rng.normal(size=6)
        C = p.coriolis(nu)
        # nu^T C(nu) nu must vanish (no energy injection by Coriolis terms).
        assert abs(nu @ C @ nu) < 1e-8


def test_damping_positive_semidefinite():
    p = REMUSParams()
    rng = np.random.default_rng(1)
    for _ in range(20):
        nu = rng.normal(size=6)
        D = p.damping(nu)
        eig = np.linalg.eigvalsh(0.5 * (D + D.T))
        assert np.all(eig >= -1e-9), "damping must be dissipative"


def test_rotation_matrix_orthonormal():
    R = rotation_matrix(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_skew_identity():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-4.0, 0.5, 2.0])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_neutral_equilibrium_is_fixed_point():
    """At rest, neutrally buoyant, level: the state must not move."""

    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.05))
    veh.reset()
    for _ in range(200):
        veh.step(np.zeros(6))
    assert np.allclose(veh.state, np.zeros(12), atol=1e-6)


def test_free_decay_dissipates_energy():
    """With no input the kinetic energy must monotonically decrease."""

    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.02))
    veh.reset(nu=np.array([1.5, 0.3, 0.2, 0.0, 0.1, 0.2]))
    e0 = veh.kinetic_energy()
    e_prev = e0
    for _ in range(500):
        veh.step(np.zeros(6))
        e = veh.kinetic_energy()
        assert e <= e_prev + 1e-9, "energy increased without actuation"
        e_prev = e
    # After 10 s of free decay >99% of the kinetic energy must have dissipated.
    assert e_prev < 0.01 * e0, "vehicle should have shed nearly all its energy"


def test_surge_thrust_produces_forward_motion():
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.05))
    veh.reset()
    tau = np.zeros(6)
    tau[0] = 20.0  # surge force
    for _ in range(100):
        veh.step(tau)
    assert veh.nu[0] > 0.1, "positive surge thrust should drive vehicle forward"
    assert veh.eta[0] > 0.1, "vehicle should advance in +x (NED)"


def test_pitch_restoring_is_stable():
    """A small pitch perturbation must be restored (metacentric stability)."""

    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.01))
    veh.reset(eta=np.array([0, 0, 0, 0, 0.2, 0]))
    max_pitch = abs(veh.eta[4])
    settled = []
    for _ in range(3000):
        veh.step(np.zeros(6))
        settled.append(veh.eta[4])
    assert abs(settled[-1]) < 0.5 * max_pitch, "pitch should be restored toward 0"


def test_current_relative_velocity_effect():
    """A forward current should drag a passive vehicle forward over time."""

    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.05))
    veh.reset()
    cur = OceanCurrent(CurrentConfig(mean_velocity=(0.5, 0.0, 0.0),
                                     turbulence_intensity=0.0))
    for _ in range(400):
        R = rotation_matrix(*veh.eta[3:])
        nu_c = cur.body_velocity(R.T, 0.05)
        veh.step(np.zeros(6), nu_c=nu_c)
    assert veh.eta[0] > 0.05, "ambient current should advect the vehicle downstream"


def test_thruster_saturation_and_lag():
    th = ThrusterModel(ThrusterConfig(tau_max=(10,) * 6, time_constant=0.1))
    out = th.step(np.full(6, 100.0), dt=0.05)
    assert np.all(np.abs(out) <= 10.0 + 1e-9), "output must respect saturation"
    # First-order lag: a single step cannot reach the (saturated) target.
    assert np.all(out < 10.0)


def test_allocator_preserves_direction():
    alloc = AdaptiveAllocator(np.array([10.0, 10, 10, 10, 10, 10]))
    cmd = np.array([20.0, 10.0, 0, 0, 0, 0])
    out = alloc(cmd)
    assert np.all(np.abs(out) <= 10.0 + 1e-9)
    # Direction preserved: the ratio between active channels is unchanged.
    assert np.isclose(out[0] / out[1], cmd[0] / cmd[1])


def test_jacobian_block_structure():
    eta = np.array([1, 2, 3, 0.1, 0.2, 0.3])
    J = jacobian(eta)
    assert np.allclose(J[:3, 3:], 0.0)
    assert np.allclose(J[3:, :3], 0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
