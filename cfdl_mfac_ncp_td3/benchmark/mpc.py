"""Receding-horizon linear MPC benchmark controller.

Each pose DOF is modelled as a discrete double integrator (position/velocity)
with an effective inertia taken from the diagonal of the AUV mass matrix.
A finite-horizon quadratic cost on tracking error and control effort is solved
in *condensed* form (decision variable = the input sequence); the first move
is applied (receding horizon) and saturated.  The per-DOF generalised forces
are assembled into a NED wrench and mapped to the body frame via ``J^T``.

The condensed unconstrained solution gives the optimal input sequence
``U* = -(Su^T Qbar Su + Rbar)^{-1} Su^T Qbar Sx x0``; input limits are then
imposed by saturation, which is the usual lightweight MPC approximation when a
full QP solver is not warranted.
"""

from __future__ import annotations

import numpy as np

from ..dynamics.hydrodynamics import REMUSParams
from ..dynamics.remus6dof import jacobian
from .base import BaseController, pose_error


class MPCController(BaseController):
    """Decoupled double-integrator linear MPC over the 6-DOF pose."""

    name = "MPC"

    def __init__(self, params: REMUSParams | None = None, horizon: int = 20,
                 q_pos: float = 10.0, q_vel: float = 1.0, r_u: float = 0.01,
                 tau_max=None):
        self.p = params or REMUSParams()
        M = self.p.mass_matrix()
        self.m_eff = np.clip(np.abs(np.diag(M)), 1e-3, None)  # per-DOF inertia
        self.N = int(horizon)
        self.q_pos, self.q_vel, self.r_u = q_pos, q_vel, r_u
        self.tau_max = None if tau_max is None else np.asarray(tau_max, dtype=float)
        self._cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def _gain(self, dof: int, dt: float) -> np.ndarray:
        """First-move condensed-MPC feedback gain for one DOF (cached)."""

        key = (dof, round(dt, 6))
        if key in self._cache:
            return self._cache[key]
        m = self.m_eff[dof]
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.0], [dt / m]])
        N = self.N
        # Build condensed prediction matrices: X = Sx x0 + Su U.
        Sx = np.zeros((2 * N, 2))
        Su = np.zeros((2 * N, N))
        Apow = np.eye(2)
        for i in range(N):
            Apow = Apow @ A if i > 0 else A  # A^(i+1)
            Sx[2 * i:2 * i + 2, :] = Apow
            acc = np.eye(2)
            for j in range(i, -1, -1):
                # contribution of u_j to x_{i+1}: A^(i-j) B
                power = np.linalg.matrix_power(A, i - j)
                Su[2 * i:2 * i + 2, j:j + 1] = power @ B
        Q = np.diag([self.q_pos, self.q_vel])
        Qbar = np.kron(np.eye(N), Q)
        Rbar = self.r_u * np.eye(N)
        H = Su.T @ Qbar @ Su + Rbar
        F = Su.T @ Qbar @ Sx
        # U* = -H^{-1} F x0; first move u0 = -(H^{-1} F)[0] x0 = -K x0.
        K = np.linalg.solve(H, F)[0]  # shape (2,)
        self._cache[key] = K
        return K

    # ------------------------------------------------------------------ #
    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float = 0.05) -> np.ndarray:
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, dtype=float)

        J = jacobian(eta)
        eta_dot = J @ nu  # NED pose rate
        e = pose_error(eta_d, eta)
        e_dot = eta_d_dot - eta_dot

        f_eta = np.zeros(6)
        for i in range(6):
            K = self._gain(i, dt)
            # State x0 = [pos_err_to_setpoint, vel_err]; optimal u0 = -K x0,
            # expressed directly via the (positive) tracking error e, e_dot.
            f_eta[i] = K[0] * e[i] + K[1] * e_dot[i]

        tau = J.T @ f_eta
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau
