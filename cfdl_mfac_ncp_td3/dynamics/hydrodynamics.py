"""Hydrodynamic parameters and matrices for a REMUS-100-class AUV.

The model follows the standard Fossen 6-DOF marine-craft formulation::

    M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + tau_dist

with the SNAME body-fixed state ``nu = [u, v, w, p, q, r]`` (surge, sway,
heave, roll-rate, pitch-rate, yaw-rate) and the earth-fixed (NED) pose
``eta = [x, y, z, phi, theta, psi]``.

Rigid-body inertia and added-mass coefficients are taken from Prestero's
REMUS-100 identification work [Prestero2001]_.  The damping model uses a
combined linear + quadratic (cross-flow) diagonal structure with
representative coefficients for a slender torpedo-shaped hull; the code is
organised so that off-diagonal coupling terms can be added without changing
the public interface.

.. [Prestero2001] T. Prestero, "Verification of a Six-Degree of Freedom
   Simulation Model for the REMUS Autonomous Underwater Vehicle", MIT/WHOI
   M.Sc. thesis, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def skew(a: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix ``S(a)`` such that ``S(a) b = a x b``."""

    a = np.asarray(a, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )


@dataclass
class REMUSParams:
    """Physical and hydrodynamic parameters of a REMUS-100-class AUV.

    All quantities are in SI units.  The defaults describe a neutrally
    buoyant vehicle (``W == B``) so that free-decay / equilibrium tests have
    a clean steady state; set :attr:`buoyancy` independently to study a
    net-buoyant vehicle.
    """

    # --- gross properties -------------------------------------------------
    mass: float = 30.48  # [kg]
    gravity: float = 9.81  # [m/s^2]
    rho: float = 1025.0  # sea-water density [kg/m^3] (informational)
    weight: float = field(default=None)  # [N]; defaults to mass * gravity
    buoyancy: float = field(default=None)  # [N]; defaults to weight (neutral)

    # centres of gravity / buoyancy in the body frame [m]
    r_g: tuple = (0.0, 0.0, 0.0196)
    r_b: tuple = (0.0, 0.0, 0.0)

    # rigid-body moments of inertia about the body axes [kg m^2]
    Ix: float = 0.177
    Iy: float = 3.45
    Iz: float = 3.45

    # --- added mass (Prestero 2001) [kg, kg m, kg m^2] --------------------
    X_udot: float = -0.93
    Y_vdot: float = -35.5
    Y_rdot: float = 1.93
    Z_wdot: float = -35.5
    Z_qdot: float = -1.93
    K_pdot: float = -0.0704
    M_wdot: float = -1.93
    M_qdot: float = -4.88
    N_vdot: float = 1.93
    N_rdot: float = -4.88

    # --- linear damping (representative) [N s/m, N m s/rad] ---------------
    Xu: float = -2.0
    Yv: float = -23.0
    Zw: float = -23.0
    Kp: float = -0.30
    Mq: float = -9.7
    Nr: float = -9.7

    # --- quadratic (cross-flow) damping (representative) ------------------
    Xuu: float = -1.62
    Yvv: float = -120.0
    Zww: float = -120.0
    Kpp: float = -0.13
    Mqq: float = -9.4
    Nrr: float = -9.4

    def __post_init__(self) -> None:
        if self.weight is None:
            self.weight = self.mass * self.gravity
        if self.buoyancy is None:
            self.buoyancy = self.weight  # neutral by default
        self.r_g = np.asarray(self.r_g, dtype=float)
        self.r_b = np.asarray(self.r_b, dtype=float)

    # ------------------------------------------------------------------ #
    # Matrix builders
    # ------------------------------------------------------------------ #
    def inertia_tensor(self) -> np.ndarray:
        """Body-frame inertia tensor (diagonal for REMUS)."""

        return np.diag([self.Ix, self.Iy, self.Iz])

    def rigid_body_mass(self) -> np.ndarray:
        """Rigid-body mass matrix ``M_RB`` (Fossen eq. 3.44)."""

        m = self.mass
        Sg = skew(self.r_g)
        Io = self.inertia_tensor()
        M = np.zeros((6, 6))
        M[:3, :3] = m * np.eye(3)
        M[:3, 3:] = -m * Sg
        M[3:, :3] = m * Sg
        M[3:, 3:] = Io
        return M

    def added_mass(self) -> np.ndarray:
        """Added-mass matrix ``M_A`` (symmetric, positive definite)."""

        # SNAME convention: M_A = -A, where A holds the hydrodynamic
        # derivatives.  The structure below is symmetric provided
        # Z_qdot == M_wdot and Y_rdot == N_vdot (true for REMUS).
        A = np.array(
            [
                [self.X_udot, 0, 0, 0, 0, 0],
                [0, self.Y_vdot, 0, 0, 0, self.Y_rdot],
                [0, 0, self.Z_wdot, 0, self.Z_qdot, 0],
                [0, 0, 0, self.K_pdot, 0, 0],
                [0, 0, self.M_wdot, 0, self.M_qdot, 0],
                [0, self.N_vdot, 0, 0, 0, self.N_rdot],
            ],
            dtype=float,
        )
        return -A

    def mass_matrix(self) -> np.ndarray:
        """Total system mass matrix ``M = M_RB + M_A``."""

        return self.rigid_body_mass() + self.added_mass()

    def coriolis(self, nu: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        """Combined Coriolis-centripetal matrix ``C(nu)``.

        Derived from the (symmetric) total mass matrix via Fossen's
        skew-symmetric parameterisation (Theorem 3.2), which guarantees
        ``nu^T C(nu) nu == 0`` (no spurious energy injection).
        """

        if M is None:
            M = self.mass_matrix()
        nu = np.asarray(nu, dtype=float).reshape(6)
        nu1, nu2 = nu[:3], nu[3:]
        M11, M12 = M[:3, :3], M[:3, 3:]
        M21, M22 = M[3:, :3], M[3:, 3:]
        C = np.zeros((6, 6))
        top = M11 @ nu1 + M12 @ nu2
        bot = M21 @ nu1 + M22 @ nu2
        C[:3, 3:] = -skew(top)
        C[3:, :3] = -skew(top)
        C[3:, 3:] = -skew(bot)
        return C

    def damping(self, nu: np.ndarray) -> np.ndarray:
        """Hydrodynamic damping matrix ``D(nu) = D_lin + D_quad(nu)`` (>= 0)."""

        nu = np.asarray(nu, dtype=float).reshape(6)
        d_lin = -np.array([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])
        quad = -np.array(
            [self.Xuu, self.Yvv, self.Zww, self.Kpp, self.Mqq, self.Nrr]
        )
        d_quad = quad * np.abs(nu)
        return np.diag(d_lin + d_quad)

    def restoring(self, eta: np.ndarray) -> np.ndarray:
        """Gravitational/buoyancy restoring vector ``g(eta)`` (Fossen eq. 4.6)."""

        eta = np.asarray(eta, dtype=float).reshape(6)
        phi, theta = eta[3], eta[4]
        W, B = self.weight, self.buoyancy
        xg, yg, zg = self.r_g
        xb, yb, zb = self.r_b
        sphi, cphi = np.sin(phi), np.cos(phi)
        sth, cth = np.sin(theta), np.cos(theta)
        g = np.array(
            [
                (W - B) * sth,
                -(W - B) * cth * sphi,
                -(W - B) * cth * cphi,
                -(yg * W - yb * B) * cth * cphi + (zg * W - zb * B) * cth * sphi,
                (zg * W - zb * B) * sth + (xg * W - xb * B) * cth * cphi,
                -(xg * W - xb * B) * cth * sphi - (yg * W - yb * B) * sth,
            ]
        )
        return g
