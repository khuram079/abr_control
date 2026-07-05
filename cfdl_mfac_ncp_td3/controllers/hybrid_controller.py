"""Integrated CFDL-MFAC / SMC hybrid controller (+ optional NCP-gated TD3).

Architecture (per-DOF model selection + fusion)::

    eta_d --> [outer kinematic loop] --> nu_d --> [CFDL-MFAC]  translational --> tau[0:3]
                                     |                                                    \\
                                     +--> z1, e_dot --------> [SMC reaching law] attitude --> tau[3:6]
                                                                                              |
                       [TD3 policy] --> tau_rl ------------------------------------------ [supervisor] --> tau
                                                                                              ^
                       [observers: state / disturbance / fault] ---------------------------/

CFDL-MFAC remains the controller's core -- it is model-free and adaptive,
which is where it earns its keep: rejecting hydrodynamic/mass uncertainty and
thruster faults on the translational channels (surge/sway/heave).  Measured
across >1500 Monte-Carlo trials (single-vehicle + formation studies), the
per-DOF MFAC bank was consistently the weak point on the *rotational* channels
(attitude RMSE 2-8x worse than SMC/MPC): those channels have small inertia and
strongly nonlinear Euler kinematics, where a reactive, purely-adaptive law
converges slowly relative to a controller with an explicit sliding surface.
Rather than force MFAC to fix that empirically (already attempted via per-DOF
Phi rescaling -- it helped but did not close the gap), the attitude channels
are handed to a boundary-layer **sliding-mode law** (the same design used by
the standalone SMC benchmark), fed by the same pose-error/outer-loop state and
sharing the same (boostable) outer gain ``k1`` for a unified recovery
response.  ``attitude_law="mfac"`` reverts to the pure per-DOF MFAC bank for
ablation comparison.

* The **outer loop** turns the pose error into a desired body velocity
  ``nu_d = J^{-1}(eta_d_dot + K1 (eta_d - eta))``.  Velocity is a CFDL-valid
  controlled output (it settles for constant wrench), unlike pose.
* The **CFDL-MFAC** bank drives ``nu -> nu_d`` on surge/sway/heave with the
  model-free adaptive law, learning the pseudo-Jacobian ``tau -> nu`` online,
  plus a capped/filtered CFDL feed-forward for anticipation.
* The **SMC** law drives the NED attitude error/rate on roll/pitch/yaw via a
  sliding surface ``s = e_dot + k1 e``, boundary-layer reaching law, mapped to
  the body-frame wrench through the (block-diagonal) kinematic Jacobian
  transpose.
* The **observer suite** estimates the lumped disturbance and actuator faults,
  which feed the confidence-guided **supervisor** that fuses the combined
  command with the (optional, gated) **TD3** policy output.

The controller degrades gracefully: with no TD3 policy it is pure
CFDL-MFAC+SMC; with the supervisor disabled it is that plus feed-forward only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..config import ExperimentConfig, default_config
from ..dynamics.hydrodynamics import REMUSParams
from ..dynamics.remus6dof import jacobian
from ..observers import DisturbanceObserver, FaultObserver
from ..benchmark.base import pose_error
from .mfac import CFDLMFAC
from .pfdl import PFDLMFAC
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
                 residual_rl: bool = False, residual_scale: float = 0.3,
                 feedforward_cap: float = 0.5, feedforward_tau: float = 0.15,
                 attitude_law: str = "smc",
                 att_kd=(20.0, 30.0, 30.0), att_ks=(8.0, 12.0, 12.0),
                 att_phi: float = 0.1, att_lam: float = 1.5,
                 trans_damping: float = 0.0,
                 model_feedforward: bool = False, model_ff_gain: float = 1.0,
                 inner_law: str = "cfdl", pfdl_L: int = 3):
        self.cfg = config or default_config()
        self.p = params or REMUSParams()
        # k1[0:3] is the translational outer-loop gain (feasible-velocity
        # command for CFDL-MFAC); k1[3:6] is the attitude sliding-surface
        # slope (independently tunable -- the rotational Euler kinematics
        # warrant a different aggressiveness than the translational velocity
        # loop; a single shared value measurably hurt attitude tracking).
        # Both halves share one array so the external threshold-triggered
        # recovery boost (which scales k1 uniformly) accelerates translational
        # MFAC and attitude SMC together during fault/error recovery.
        self.k1 = np.array([k_outer] * 3 + [att_lam] * 3, dtype=float)
        # CFDL inverse feed-forward gain.  A model-free anticipatory term that
        # uses MFAC's *own* learned pseudo-Jacobian Phi to compute the wrench
        # needed to realise the desired velocity change: tau_ff = g * dnu_d/Phi.
        # This supplies the anticipation a purely reactive MFAC lacks, sharply
        # reducing phase lag on moving references (no plant model used).
        self.cfdl_feedforward = float(cfdl_feedforward)
        # The raw term is capped (as a fraction of tau_max) and its input is
        # low-pass filtered: an un-capped, un-filtered feed-forward spikes
        # whenever the online Phi estimate is transiently small, which was
        # burning ~2x the control energy of every baseline for little accuracy
        # gain (measured: single-vehicle/formation energy consistently worst
        # of all 6 controllers compared).
        self.ff_cap = float(feedforward_cap)
        self._ff_alpha = self.cfg.sim.dt / (feedforward_tau + self.cfg.sim.dt)
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
        #
        # Each loop's initial pseudo-Jacobian is physically scaled by the
        # rigid-body+added-mass diagonal dt/M_ii, anchored so surge keeps the
        # originally-calibrated value.  A single shared phi_init (the previous
        # behaviour) under-primes roll by ~40x (M_roll << M_surge).
        M_diag = np.diag(self.p.mass_matrix())
        true_gain = self.cfg.sim.dt / np.abs(M_diag)
        phi_scale = true_gain / true_gain[0]

        # attitude_law selects which model drives roll/pitch/yaw: "smc" hands
        # the rotational channels (small inertia, strongly nonlinear Euler
        # kinematics) to a boundary-layer sliding-mode law, while CFDL-MFAC
        # keeps surge/sway/heave -- the channels where hydrodynamic/mass
        # uncertainty and thruster faults are the dominant, model-free-suited
        # source of error.  "mfac" keeps the pure per-DOF MFAC bank on all 6
        # channels (the previous architecture) for ablation comparison.
        self.attitude_law = attitude_law
        self.mfac_dofs = list(range(6)) if attitude_law == "mfac" else [0, 1, 2]
        # inner_law selects the dynamic-linearization form of the adaptive core:
        # "cfdl" (compact form, single pseudo-gradient) or "pfdl" (partial form,
        # a length-L window of past control increments).  PFDL gives the control
        # law memory of recent inputs still propagating through the plant, which
        # curbs the compact-form integrator's surge limit cycle.  CFDL-MFAC
        # remains the default core; PFDL-MFAC is the drop-in alternative.
        self.inner_law = inner_law
        self.pfdl_L = int(pfdl_L)
        self.mfac = []
        for i in self.mfac_dofs:
            mfac_cfg = dataclasses.replace(self.cfg.mfac, phi_init=self.cfg.mfac.phi_init * phi_scale[i])
            bounds = (np.array([-tau_max[i]]), np.array([tau_max[i]]))
            if inner_law == "pfdl":
                self.mfac.append(PFDLMFAC(1, 1, mfac_cfg, u_bounds=bounds, L=self.pfdl_L))
            else:
                self.mfac.append(CFDLMFAC(1, 1, mfac_cfg, u_bounds=bounds))

        # SMC attitude reaching-law gains (unused when attitude_law="mfac").
        # The sliding-surface slope reuses self.k1[3:] so the same external
        # recovery-boost mechanism (which scales k1) accelerates both the
        # translational MFAC loop and the attitude SMC law during recovery.
        self.att_kd = np.asarray(att_kd, dtype=float)
        self.att_ks = np.asarray(att_ks, dtype=float)
        self.att_phi = float(att_phi)

        # Translational velocity-rate (acceleration-feedback) damping, applied
        # in parallel with the MFAC command: tau_i -= trans_damping * dnu_i/dt.
        # CFDL-MFAC is an *integrating* law, so its surge velocity loop is
        # under-damped and limit-cycles about the equilibrium thrust -- the
        # command swings +/-33 N (RMS) for a mean velocity a proportional law
        # (SMC) holds with ~11 N RMS, which was the entire ~2x control-energy
        # gap (localised to surge; see results/ENERGY_ANALYSIS.md).  In steady
        # tracking dnu/dt is small so this term is inert; during the limit
        # cycle it is large and damps it, cutting energy up to ~50% with a
        # graceful tracking trade-off.  MFAC remains the adaptive core; this is
        # a fixed inner damping loop, not a replacement.
        self.trans_damping = float(trans_damping)

        # Nominal-model (computed-torque) feed-forward on the translational
        # channels.  MPC's structural advantage is that it optimises over a
        # horizon using a *plant model*; a purely reactive CFDL-MFAC has no such
        # preview, so its integrating law has to build up the steady drag/inertia
        # wrench itself -- which on surge manifests as the +/-33 N limit cycle
        # that was the entire ~1.5x control-energy gap to MPC.  Here the
        # nominal REMUS dynamics supply that wrench directly and smoothly:
        #   tau_ff = M nu_dot_d + C(nu) nu + D(nu) nu + g(eta),
        # evaluated with the desired body acceleration nu_dot_d (the filtered
        # rate of the outer-loop velocity command).  CFDL-MFAC is left to adapt
        # only the *model error* (mass/hydro uncertainty, faults), so the hybrid
        # gains MPC-like anticipation while keeping its model-free robustness --
        # MPC degrades when its fixed model is wrong; the adaptive residual does
        # not.  ``model_ff_gain`` (tuned) scales the trust placed in the nominal
        # model.  Applied to surge/sway/heave only; attitude keeps the SMC law.
        self.model_feedforward = bool(model_feedforward)
        self.model_ff_gain = float(model_ff_gain)

        self.td3 = td3_agent
        self.use_observers = use_observers
        # Residual-RL mode: the policy outputs a *bounded correction* added to
        # the (strong) MFAC command, tau = clip(tau_mfac + scale*tau_max*a).
        # A zero policy reproduces the adaptive baseline exactly, so the learned
        # term can only improve on it -- unlike full-wrench blending it cannot
        # regress below the baseline.  Residual mode bypasses the supervisor.
        self.residual_rl = residual_rl
        self.residual_scale = float(residual_scale)
        self.supports_recovery = True  # threshold-triggered aggressive catch-up
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
        self._dnu_d_filt = np.zeros(6)
        self._nu_prev = np.zeros(6)
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
        J = jacobian(eta)

        # CFDL-MFAC: a bank of SISO adaptive loops drives each translational
        # body-velocity channel nu_i -> nu_d_i with a wrench command, plus a
        # model-free CFDL inverse feed-forward that anticipates the moving
        # velocity reference using the learned per-channel pseudo-Jacobian.
        # The reference-rate input is low-pass filtered and the resulting
        # feed-forward wrench is capped (fraction of tau_max) so a transiently
        # small Phi estimate cannot inject a large, energy-wasting spike.
        tau_mfac = np.zeros(6)
        dnu_d_raw = nu_d - self._nu_d_prev
        self._dnu_d_filt += self._ff_alpha * (dnu_d_raw - self._dnu_d_filt)
        for idx, i in enumerate(self.mfac_dofs):
            u = self.mfac[idx].control([nu[i]], [nu_d[i]])[0]
            if self.cfdl_feedforward:
                phi = self.mfac[idx].gain
                if abs(phi) > 1e-4:
                    u_ff = self.cfdl_feedforward * self._dnu_d_filt[i] / phi
                    ff_limit = self.ff_cap * self.tau_max[i]
                    u += np.clip(u_ff, -ff_limit, ff_limit)
            tau_mfac[i] = np.clip(u, -self.tau_max[i], self.tau_max[i])
        self._nu_d_prev = nu_d

        # Nominal-model computed-torque feed-forward (translational channels).
        # tau_ff = M nu_dot_d + C(nu) nu + D(nu) nu + g(eta); nu_dot_d is the
        # filtered desired body-acceleration.  This supplies the steady
        # drag/inertia wrench that a reactive MFAC would otherwise integrate up
        # (the surge limit cycle), so the MFAC loop is left to adapt only the
        # residual model error.  Added before clipping so it shares the wrench
        # budget with the adaptive command.
        if self.model_feedforward:
            M = self.p.mass_matrix()
            acc_d = self._dnu_d_filt / dt
            tau_ff = (M @ acc_d + self.p.coriolis(nu, M) @ nu
                      + self.p.damping(nu) @ nu + self.p.restoring(eta))
            tau_mfac[:3] = np.clip(tau_mfac[:3] + self.model_ff_gain * tau_ff[:3],
                                   -self.tau_max[:3], self.tau_max[:3])

        # Velocity-rate damping on the translational channels (breaks the
        # surge integrator limit cycle; inert in steady tracking).
        if self.trans_damping:
            acc = (nu[:3] - self._nu_prev[:3]) / dt
            tau_mfac[:3] = np.clip(tau_mfac[:3] - self.trans_damping * acc,
                                   -self.tau_max[:3], self.tau_max[:3])
        self._nu_prev = nu.copy()

        # SMC attitude law (roll/pitch/yaw): boundary-layer sliding-mode
        # reaching law on the NED attitude error, mapped to the body-frame
        # wrench via J^T.  J is block-diagonal (translational/rotational
        # blocks decouple), so zeroing f_eta[:3] isolates tau[3:6] exactly.
        if self.attitude_law == "smc":
            e_dot = eta_d_dot - J @ nu
            s_att = e_dot[3:] + self.k1[3:] * z1[3:]
            f_att = self.att_kd * s_att + self.att_ks * np.tanh(s_att / self.att_phi)
            f_eta = np.zeros(6)
            f_eta[3:] = f_att
            tau_att = J.T @ f_eta
            tau_mfac[3:] = np.clip(tau_att[3:], -self.tau_max[3:], self.tau_max[3:])

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
