"""Integrated CFDL-MFAC-NCP-TD3 hybrid controller.

Architecture (cascade + fusion)::

    eta_d --> [outer kinematic loop] --> nu_d --> [CFDL-MFAC inner loop] --> tau_mfac
                                                                              |
                       [TD3 policy] --> tau_rl --------------------------- [supervisor] --> tau
                                                                              ^
                       [observers: state / disturbance / fault] -------------/

* The **outer loop** turns the pose error into a desired body velocity
  ``nu_d = J^{-1}(eta_d_dot + K1 (eta_d - eta))``.  Velocity is a CFDL-valid
  controlled output (it settles for constant wrench), unlike pose.
* The **inner CFDL-MFAC loop** drives ``nu -> nu_d`` with the model-free
  adaptive law, learning the pseudo-Jacobian ``tau -> nu`` online.
* The **observer suite** estimates the lumped disturbance and actuator faults,
  which feed the confidence-guided **supervisor** that fuses ``tau_mfac`` with
  the (robust) **TD3** policy output.

The controller degrades gracefully: with no TD3 policy it is pure adaptive
control; with the supervisor disabled it is MFAC + feed-forward.
"""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig, default_config
from ..dynamics.hydrodynamics import REMUSParams
from ..dynamics.remus6dof import jacobian
from ..observers import DisturbanceObserver, FaultObserver
from ..benchmark.base import pose_error
from .mfac import CFDLMFAC
from .supervisor import HybridSupervisor


class HybridController:
    """Composed hybrid controller producing a 6-DOF wrench command."""

    name = "Hybrid"

    def __init__(self, config: ExperimentConfig | None = None,
                 params: REMUSParams | None = None,
                 td3_agent=None, k_outer=1.5, cfdl_feedforward=2.5,
                 nu_max=(2.0, 2.0, 2.0, 0.8, 0.8, 0.8),
                 use_observers: bool = True, use_supervisor: bool = True,
                 disturbance_feedforward: bool = False,
                 residual_rl: bool = False, residual_scale: float = 0.3):
        self.cfg = config or default_config()
        self.p = params or REMUSParams()
        self.k1 = np.full(6, float(k_outer))
        # CFDL inverse feed-forward gain.  A model-free anticipatory term that
        # uses MFAC's *own* learned pseudo-Jacobian Phi to compute the wrench
        # needed to realise the desired velocity change: tau_ff = g * dnu_d/Phi.
        # This supplies the anticipation a purely reactive MFAC lacks, sharply
        # reducing phase lag on moving references (no plant model used).
        self.cfdl_feedforward = float(cfdl_feedforward)
        # Feasible body-velocity envelope: the outer loop never commands a
        # velocity the (drag-limited) vehicle cannot achieve, which would
        # otherwise saturate the inner loop into a limit cycle.
        self.nu_max = np.asarray(nu_max, dtype=float)

        tau_max = np.asarray(self.cfg.thruster.tau_max, dtype=float)
        self.tau_max = tau_max
        # Decentralised per-DOF SISO MFAC.  The body-velocity dynamics are
        # diagonally dominant (mass and damping are near-diagonal), so a bank
        # of single-input/single-output adaptive loops is far more robust than
        # one coupled MIMO loop; the off-diagonal Coriolis coupling becomes a
        # disturbance that each loop's pseudo-gradient adapts to.
        self.mfac = [
            CFDLMFAC(1, 1, self.cfg.mfac,
                     u_bounds=(np.array([-tau_max[i]]), np.array([tau_max[i]])))
            for i in range(6)
        ]

        self.td3 = td3_agent
        self.use_observers = use_observers
        # Residual-RL mode: the policy outputs a *bounded correction* added to
        # the (strong) MFAC command, tau = clip(tau_mfac + scale*tau_max*a).
        # A zero policy reproduces the adaptive baseline exactly, so the learned
        # term can only improve on it -- unlike full-wrench blending it cannot
        # regress below the baseline.  Residual mode bypasses the supervisor.
        self.residual_rl = residual_rl
        self.residual_scale = float(residual_scale)
        self.use_supervisor = use_supervisor and not residual_rl
        self.disturbance_feedforward = disturbance_feedforward
        self.dob = DisturbanceObserver(self.p, self.cfg.observer) if use_observers else None
        self.fob = FaultObserver(6, self.cfg.observer) if use_observers else None
        self.supervisor = HybridSupervisor(self.cfg.supervisor) if self.use_supervisor else None
        self.reset()

    def reset(self) -> None:
        for m in self.mfac:
            m.reset()
        if self.dob is not None:
            self.dob.reset()
        if self.fob is not None:
            self.fob.reset()
        if self.supervisor is not None:
            self.supervisor.reset()
        self._tau_prev = np.zeros(6)
        self._nu_d_prev = np.zeros(6)
        self.last_info = {}

    # ------------------------------------------------------------------ #
    def _outer_loop(self, eta, nu, eta_d, eta_d_dot):
        J = jacobian(eta)
        try:
            Jinv = np.linalg.inv(J)
        except np.linalg.LinAlgError:  # pragma: no cover
            Jinv = np.linalg.pinv(J)
        z1 = pose_error(eta_d, eta)
        nu_d = Jinv @ (eta_d_dot + self.k1 * z1)
        nu_d = np.clip(nu_d, -self.nu_max, self.nu_max)  # feasible velocity envelope
        return nu_d, z1

    @staticmethod
    def rl_observation(z1, nu, eta_d_dot) -> np.ndarray:
        """The 18-D policy observation shared by training and deployment."""

        return np.concatenate([z1, nu, eta_d_dot]).astype(np.float32)

    def _rl_action(self, z1, nu, nu_d, eta_d_dot) -> np.ndarray:
        if self.td3 is None:
            return None
        action = self.td3.select_action(self.rl_observation(z1, nu, eta_d_dot), noise=None)
        return np.clip(action, -1.0, 1.0) * self.tau_max

    def apply_residual(self, tau_mfac: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Add a bounded learned correction to the adaptive command."""

        action = np.clip(np.asarray(action, dtype=float).reshape(6), -1.0, 1.0)
        tau = tau_mfac + self.residual_scale * self.tau_max * action
        return np.clip(tau, -self.tau_max, self.tau_max)

    # ------------------------------------------------------------------ #
    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float | None = None) -> np.ndarray:
        dt = self.cfg.sim.dt if dt is None else dt
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        eta_d = np.asarray(eta_d, dtype=float).reshape(6)
        eta_d_dot = np.zeros(6) if eta_d_dot is None else np.asarray(eta_d_dot, dtype=float)

        # Observers (use the previously applied wrench).
        d_hat = theta = None
        if self.use_observers:
            d_hat = self.dob.update(eta, nu, self._tau_prev, dt)
            tau_real = self._tau_prev + d_hat  # reconstructed realised wrench
            theta = self.fob.update(self._tau_prev, tau_real)

        # Outer kinematic loop -> desired body velocity.
        nu_d, z1 = self._outer_loop(eta, nu, eta_d, eta_d_dot)

        # Inner CFDL-MFAC loop: a bank of SISO adaptive loops drives each
        # body-velocity channel nu_i -> nu_d_i with a wrench command, plus a
        # model-free CFDL inverse feed-forward that anticipates the moving
        # velocity reference using the learned per-channel pseudo-Jacobian.
        tau_mfac = np.zeros(6)
        dnu_d = nu_d - self._nu_d_prev
        for i in range(6):
            u = self.mfac[i].control([nu[i]], [nu_d[i]])[0]
            if self.cfdl_feedforward:
                phi = float(self.mfac[i].model.phi[0, 0])
                if abs(phi) > 1e-4:
                    u += self.cfdl_feedforward * dnu_d[i] / phi
            tau_mfac[i] = np.clip(u, -self.tau_max[i], self.tau_max[i])
        self._nu_d_prev = nu_d

        # Learned policy: residual correction (preferred) or supervisor fusion.
        if self.residual_rl and self.td3 is not None:
            action = self.td3.select_action(
                self.rl_observation(z1, nu, eta_d_dot), noise=None)
            tau = self.apply_residual(tau_mfac, action)
            info = {"alpha": 0.0, "residual": self.residual_scale}
        else:
            tau_rl = self._rl_action(z1, nu, nu_d, eta_d_dot)
            if self.use_supervisor and tau_rl is not None:
                tau, info = self.supervisor.fuse(tau_mfac, tau_rl, error=z1,
                                                 d_hat=d_hat, theta=theta, dt=dt)
            else:
                tau, info = tau_mfac, {"alpha": 0.0}

        # Optional disturbance-observer feed-forward.  Off by default: the
        # CFDL-MFAC loop is already adaptive and rejects slowly-varying
        # disturbances on its own, so explicit feed-forward tends to
        # double-compensate.  The observer estimates instead inform the
        # confidence supervisor (fault/disturbance-aware RL hand-over).
        if self.disturbance_feedforward and self.use_observers and d_hat is not None:
            tau = tau - d_hat

        tau = np.clip(tau, -self.tau_max, self.tau_max)
        self._tau_prev = tau
        self.last_info = {"nu_d": nu_d, "z1": z1, "d_hat": d_hat, "theta": theta, **info}
        return tau
