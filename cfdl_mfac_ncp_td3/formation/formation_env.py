"""Leader--follower formation simulator (1 leader + N followers).

The leader tracks a reference trajectory with a fixed, strong controller
(identical across all experiments, so the leader path is held constant and only
the *follower* controller under test varies -- a fair comparison).  Each
follower is a full 6-DOF REMUS vehicle that must hold a slot defined in the
leader's body frame.

Supports the elements the study requires:

* **random initial positions**, **ocean-current disturbance** and **sensor
  noise** (for the Monte-Carlo campaign),
* **actuator-fault injection** on a follower at a random time, and
* **intelligent recovery**: when a follower's formation error exceeds
  ``recovery_threshold`` the method enters an aggressive catch-up mode
  (outer-loop gain boost); recovery time is measured with hysteresis.

Baseline followers (PID/SMC/MPC/fuzzy) run their standard feedback -- the
recovery boost only applies to controllers that expose an outer gain (``k1``),
i.e. the proposed hybrid; this is a feature of the method, not a handicap on
the baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ExperimentConfig, default_config
from ..dynamics import REMUS6DOF, REMUSParams, OceanCurrent, ThrusterModel
from ..dynamics.remus6dof import rotation_matrix, jacobian
from ..dynamics.hydrodynamics import skew
from ..trajectories import make_trajectory
from ..benchmark.base import pose_error
from ..benchmark import BacksteppingController


@dataclass
class FormationConfig:
    """Formation geometry and recovery settings."""

    # Follower slots in the leader body frame [m]: a compact V behind the leader.
    offsets: tuple = ((-2.0, -1.5, 0.0), (-2.0, 1.5, 0.0), (-3.5, 0.0, 0.0))
    recovery_threshold: float = 2.0     # formation-error norm that triggers recovery
    recovery_exit_ratio: float = 0.5    # hysteresis: exit when err < ratio*threshold
    recovery_boost: float = 2.0         # outer-gain multiplier in recovery
    prediction_gain: float = 2.5        # CFDL feed-forward ("prediction") gain
    fault_prob: float = 0.5             # per-episode probability of a follower fault
    fault_time_frac: float = 0.4        # fault onset as a fraction of the horizon


class FormationSimulator:
    """Simulate one formation episode and return per-follower logs + metrics."""

    def __init__(self, config: ExperimentConfig | None = None,
                 fcfg: FormationConfig | None = None, trajectory: str = "helix"):
        self.cfg = config or default_config()
        self.fcfg = fcfg or FormationConfig()
        self.trajectory_name = trajectory
        self.n = len(self.fcfg.offsets)
        self.dt = self.cfg.sim.dt

    # ------------------------------------------------------------------ #
    def _follower_slot(self, eta_L: np.ndarray, nu_L: np.ndarray, offset: np.ndarray):
        """Desired follower pose *and its velocity* from the leader's state.

        The leader is assumed to broadcast its state, so the slot velocity is
        computed analytically (noise-free) rather than by differentiating a
        noisy measured pose::

            slot     = p_L + R_L offset
            slot_dot = eta_L_dot + [R_L S(omega) offset ; 0]
        """

        R = rotation_matrix(eta_L[3], eta_L[4], eta_L[5])
        pos = eta_L[:3] + R @ offset
        att = eta_L[3:].copy()
        slot = np.concatenate([pos, att])

        eta_L_dot = jacobian(eta_L) @ nu_L
        pos_dot = eta_L_dot[:3] + (R @ skew(nu_L[3:])) @ offset
        slot_dot = np.concatenate([pos_dot, eta_L_dot[3:]])
        return slot, slot_dot

    def simulate(self, follower_factory, seed: int = 0, randomize: bool = True,
                 fault: bool = True, recovery_threshold: float | None = None,
                 prediction_gain: float | None = None) -> dict:
        rng = np.random.default_rng(seed)
        tm = np.asarray(self.cfg.thruster.tau_max, dtype=float)
        rec_thr = self.fcfg.recovery_threshold if recovery_threshold is None else recovery_threshold
        pred_gain = self.fcfg.prediction_gain if prediction_gain is None else prediction_gain

        # --- leader (fixed strong controller, constant across experiments) --
        leader = REMUS6DOF(REMUSParams(), self.cfg.sim)
        traj = make_trajectory(self.trajectory_name)
        eta_dL0, _ = traj.reference(0.0)
        leader.reset(eta=eta_dL0)  # start on-trajectory (no leader transient)
        leader_ctrl = BacksteppingController(tau_max=list(tm))
        leader_thr = ThrusterModel(self.cfg.thruster)
        n_steps = int(min(self.cfg.sim.horizon, traj.duration) / self.dt)

        # --- followers ------------------------------------------------------
        followers, fctrls, fthr, fcur, fparams = [], [], [], [], []
        base_k1, support_recovery = [], []
        meas_noise = 0.0
        for j in range(self.n):
            p = REMUSParams()
            if randomize:
                p = REMUSParams(mass=30.48 * (1.0 + rng.uniform(-0.2, 0.2)))
            veh = REMUS6DOF(p, self.cfg.sim)
            # Initialise near the assigned slot with a random perturbation
            # (random initial position is a Monte-Carlo factor).
            slot0, _ = self._follower_slot(leader.eta, leader.nu,
                                           np.asarray(self.fcfg.offsets[j]))
            eta0 = slot0.copy()
            if randomize:
                eta0[:3] += rng.uniform(-1.5, 1.5, size=3)
                eta0[5] += rng.uniform(-0.3, 0.3)
            veh.reset(eta=eta0)
            ctrl = follower_factory()
            if hasattr(ctrl, "cfdl_feedforward"):
                ctrl.cfdl_feedforward = pred_gain
            if hasattr(ctrl, "reset"):
                ctrl.reset()
            followers.append(veh)
            fctrls.append(ctrl)
            fthr.append(ThrusterModel(self.cfg.thruster))
            cur = OceanCurrent(self.cfg.current, rng=rng)
            if randomize:
                cur.reset(mean_velocity=rng.uniform(-0.4, 0.4, size=3))
            else:
                cur.reset(mean_velocity=(0.0, 0.0, 0.0))
            fcur.append(cur)
            fparams.append(p)
            base_k1.append(np.array(ctrl.k1).copy() if hasattr(ctrl, "k1") else None)
            support_recovery.append(bool(getattr(ctrl, "supports_recovery", False)))
        if randomize:
            meas_noise = 0.02

        # fault schedule
        fault_step, fault_follower, fault_ch, fault_eff = -1, -1, -1, 1.0
        if fault and rng.random() < self.fcfg.fault_prob:
            fault_step = int(self.fcfg.fault_time_frac * n_steps)
            fault_follower = int(rng.integers(self.n))
            fault_ch = int(rng.integers(6))
            fault_eff = float(rng.uniform(0.2, 0.5))

        # logs
        form_err = np.zeros((n_steps, self.n))       # per-follower formation error norm
        att_err = np.zeros((n_steps, self.n))        # attitude error norm
        energy = np.zeros(self.n)
        in_recovery = np.zeros(self.n, dtype=bool)
        recovery_time = np.zeros(self.n)
        leader_err = np.zeros(n_steps)
        prev_slot = [None] * self.n
        leader_log, foll_log = [], [[] for _ in range(self.n)]

        for k in range(n_steps):
            t = k * self.dt
            eta_dL, eta_dL_dot = traj.reference(t)
            # leader control + step
            tauL = leader_ctrl.control(leader.eta, leader.nu, eta_dL, eta_dL_dot, self.dt)
            leader.step(leader_thr.step(tauL, self.dt))
            leader_err[k] = np.linalg.norm(pose_error(eta_dL, leader.eta))
            leader_log.append(leader.eta[:3].copy())

            # inject fault
            if k == fault_step:
                fthr[fault_follower].set_fault(fault_ch, fault_eff)

            for j in range(self.n):
                # Slot from the leader's communicated (true) state; the follower
                # only measures *itself* with noise.
                slot, slot_dot = self._follower_slot(leader.eta, leader.nu,
                                                     np.asarray(self.fcfg.offsets[j]))
                veh = followers[j]
                eta_meas = veh.eta + rng.normal(0, meas_noise, 6)
                nu_meas = veh.nu + rng.normal(0, meas_noise, 6)
                e = pose_error(slot, veh.eta)
                enorm = float(np.linalg.norm(e[:3]))
                form_err[k, j] = enorm
                att_err[k, j] = float(np.linalg.norm(e[3:]))

                # Recovery-time metric: measured identically for EVERY
                # controller (error above threshold, with hysteresis).
                if not in_recovery[j] and enorm > rec_thr:
                    in_recovery[j] = True
                elif in_recovery[j] and enorm < rec_thr * self.fcfg.recovery_exit_ratio:
                    in_recovery[j] = False
                if in_recovery[j]:
                    recovery_time[j] += self.dt

                # Intelligent recovery ACTION (a feature of the proposed method):
                # a threshold-triggered outer-gain boost, applied only to
                # controllers that advertise recovery support.  Baselines run
                # their standard feedback -- they are not handicapped, they
                # simply lack this feature.
                if support_recovery[j] and base_k1[j] is not None:
                    fctrls[j].k1 = base_k1[j] * (self.fcfg.recovery_boost if in_recovery[j] else 1.0)

                tau = fctrls[j].control(eta_meas, nu_meas, slot, slot_dot, self.dt)
                tau_app = fthr[j].step(tau, self.dt)
                energy[j] += float(tau_app @ tau_app) * self.dt
                R = rotation_matrix(*veh.eta[3:])
                nu_c = fcur[j].body_velocity(R.T, self.dt)
                veh.step(tau_app, nu_c=nu_c)
                foll_log[j].append(veh.eta[:3].copy())

        return {
            "form_err": form_err, "att_err": att_err, "energy": energy,
            "recovery_time": recovery_time, "leader_err": leader_err,
            "fault": (fault_follower, fault_ch, fault_eff, fault_step),
            "n_steps": n_steps, "dt": self.dt,
            "leader_log": np.asarray(leader_log),
            "follower_logs": [np.asarray(f) for f in foll_log],
        }
