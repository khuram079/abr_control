"""6-DOF REMUS AUV rigid-body simulator.

This module ties the hydrodynamics, ocean current and actuator models into a
single integrable plant.  The continuous-time model is::

    eta_dot = J(eta) nu
    M nu_dot = tau + tau_dist - C(nu_r) nu_r - D(nu_r) nu_r - g(eta)

where ``nu_r = nu - nu_c`` is the velocity relative to the (quasi-irrotational)
ocean current.  Integration uses fixed-step RK4 by default.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .hydrodynamics import REMUSParams


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body->NED rotation matrix for Z-Y-X (yaw-pitch-roll) Euler angles."""

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth, cth * sphi, cth * cphi],
        ]
    )


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """Transform body angular rates to Euler-angle rates (Fossen eq. 2.41)."""

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    # Guard against the gimbal-lock singularity at theta = +/- pi/2.
    cth = np.sign(cth) * max(abs(cth), 1e-4) if cth != 0 else 1e-4
    tth = np.tan(theta)
    return np.array(
        [
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth],
        ]
    )


def jacobian(eta: np.ndarray) -> np.ndarray:
    """Full 6x6 kinematic transform ``J(eta)``."""

    eta = np.asarray(eta, dtype=float).reshape(6)
    phi, theta, psi = eta[3], eta[4], eta[5]
    J = np.zeros((6, 6))
    J[:3, :3] = rotation_matrix(phi, theta, psi)
    J[3:, 3:] = euler_rate_matrix(phi, theta)
    return J


class REMUS6DOF:
    """Integrable 6-DOF REMUS plant.

    Parameters
    ----------
    params:
        Hydrodynamic parameters (defaults to a REMUS-100-class vehicle).
    sim:
        Simulation/integration settings.
    """

    DOF = 6

    def __init__(self, params: REMUSParams | None = None, sim: SimConfig | None = None):
        self.p = params or REMUSParams()
        self.sim = sim or SimConfig()
        self.M = self.p.mass_matrix()
        self.Minv = np.linalg.inv(self.M)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, eta: np.ndarray | None = None, nu: np.ndarray | None = None) -> np.ndarray:
        self.eta = np.zeros(6) if eta is None else np.asarray(eta, dtype=float).reshape(6)
        self.nu = np.zeros(6) if nu is None else np.asarray(nu, dtype=float).reshape(6)
        return self.state

    @property
    def state(self) -> np.ndarray:
        """Stacked state ``[eta; nu]`` (length 12)."""

        return np.concatenate([self.eta, self.nu])

    # ------------------------------------------------------------------ #
    def accel(
        self,
        eta: np.ndarray,
        nu: np.ndarray,
        tau: np.ndarray,
        nu_c: np.ndarray | None = None,
        tau_dist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Body-frame acceleration ``nu_dot`` for the given state and inputs."""

        nu = np.asarray(nu, dtype=float).reshape(6)
        nu_r = nu if nu_c is None else nu - np.asarray(nu_c, dtype=float).reshape(6)
        C = self.p.coriolis(nu_r, self.M)
        D = self.p.damping(nu_r)
        g = self.p.restoring(eta)
        rhs = np.asarray(tau, dtype=float).reshape(6) - C @ nu_r - D @ nu_r - g
        if tau_dist is not None:
            rhs = rhs + np.asarray(tau_dist, dtype=float).reshape(6)
        return self.Minv @ rhs

    def deriv(
        self,
        state: np.ndarray,
        tau: np.ndarray,
        nu_c: np.ndarray | None = None,
        tau_dist: np.ndarray | None = None,
    ) -> np.ndarray:
        """State derivative ``[eta_dot; nu_dot]`` for the stacked state."""

        eta, nu = state[:6], state[6:]
        eta_dot = jacobian(eta) @ nu
        nu_dot = self.accel(eta, nu, tau, nu_c=nu_c, tau_dist=tau_dist)
        return np.concatenate([eta_dot, nu_dot])

    # ------------------------------------------------------------------ #
    def step(
        self,
        tau: np.ndarray,
        dt: float | None = None,
        nu_c: np.ndarray | None = None,
        tau_dist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance the plant one step and return the new stacked state."""

        dt = self.sim.dt if dt is None else dt
        state = self.state

        def f(s):
            return self.deriv(s, tau, nu_c=nu_c, tau_dist=tau_dist)

        if self.sim.integrator == "euler":
            new = state + dt * f(state)
        else:  # RK4
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            new = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.eta = new[:6]
        self.nu = new[6:]
        # Wrap heading-type angles to (-pi, pi] for numerical hygiene.
        self.eta[3:] = (self.eta[3:] + np.pi) % (2 * np.pi) - np.pi
        return self.state

    # ------------------------------------------------------------------ #
    def kinetic_energy(self, nu: np.ndarray | None = None) -> float:
        """Total kinetic energy ``0.5 nu^T M nu`` (useful for sanity tests)."""

        nu = self.nu if nu is None else np.asarray(nu, dtype=float).reshape(6)
        return float(0.5 * nu @ self.M @ nu)
