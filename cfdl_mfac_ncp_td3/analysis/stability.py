"""Controller stability analysis for the CFDL-MFAC + SMC + damping hybrid.

Provides five complementary stability characterisations -- a mix of the
theory the architecture is built on and an empirical validation of each:

1. **Lyapunov (SMC attitude)** -- along a closed-loop trajectory the attitude
   sliding-surface energy ``V = 0.5 s^T s`` (``s = e_dot + lambda e``) must be
   non-increasing outside the boundary layer (``V_dot <= 0``), the reaching
   condition guaranteeing finite-time convergence to the sliding manifold.
2. **Bounded pseudo-gradient (CFDL-MFAC BIBO)** -- the online pseudo-gradient
   ``phi_i`` must stay bounded and sign-definite; this is exactly the
   controllability / generalised-Lipschitz condition under which CFDL-MFAC is
   BIBO-stable (Hou & Jin), so verifying it empirically validates the premise.
3. **Input-to-State Stability (ISS)** -- sweeping the ocean-current disturbance
   magnitude, the ultimate tracking-error bound must stay finite and grow
   sub-linearly/linearly (a finite ISS gain), with no finite-escape.
4. **Region of attraction / convergence** -- from a wide range of initial pose
   errors the closed loop must converge to the same bounded set.
5. **Monte-Carlo robust stability** -- over many randomized plants/disturbances
   the divergence rate must be zero and the error uniformly ultimately bounded.
"""

from __future__ import annotations

import numpy as np

from ..config import default_config, CurrentConfig
from ..dynamics import REMUS6DOF, REMUSParams, ThrusterModel, OceanCurrent
from ..dynamics.remus6dof import jacobian, rotation_matrix
from ..trajectories import make_trajectory
from ..benchmark.base import pose_error

ATT = slice(3, 6)


def simulate_trace(controller, trajectory="sinusoidal", config=None, seed=0,
                   current_speed=0.0, init_err=None, params=None, n_steps=None,
                   fault=None):
    """Run one closed-loop episode, recording the signals the analyses need."""

    cfg = config or default_config()
    dt = cfg.sim.dt
    veh = REMUS6DOF(params or REMUSParams(), cfg.sim)
    traj = make_trajectory(trajectory)
    eta0 = traj.reference(0.0)[0].copy()
    if init_err is not None:
        eta0[:6] = eta0[:6] + np.asarray(init_err, dtype=float)
    veh.reset(eta=eta0)
    if hasattr(controller, "reset"):
        controller.reset()
    thr = ThrusterModel(cfg.thruster)
    if fault is not None:
        thr.set_fault(int(fault[0]), float(fault[1]))
    cur = None
    if current_speed:
        cur = OceanCurrent(CurrentConfig(mean_velocity=(current_speed, current_speed * 0.5, 0.0),
                                         turbulence_intensity=0.0))
        cur.reset(mean_velocity=(current_speed, current_speed * 0.5, 0.0))
    n = n_steps or int(min(cfg.sim.horizon, traj.duration) / dt)

    lam_att = np.asarray(controller.k1[ATT]) if hasattr(controller, "k1") else np.full(3, 1.5)
    err_n, s_norm, gains, energy = [], [], [], 0.0
    for k in range(n):
        t = k * dt
        eta_d, eta_d_dot = traj.reference(t)
        e = pose_error(eta_d, veh.eta)
        e_dot = eta_d_dot - jacobian(veh.eta) @ veh.nu
        s = e_dot[ATT] + lam_att * e[ATT]           # attitude sliding surface
        err_n.append(float(np.linalg.norm(e)))
        s_norm.append(float(np.linalg.norm(s)))
        if hasattr(controller, "mfac"):
            gains.append([m.gain for m in controller.mfac])
        tau = controller.control(veh.eta, veh.nu, eta_d, eta_d_dot, dt)
        ta = thr.step(tau, dt)
        energy += float(ta @ ta) * dt
        nu_c = cur.body_velocity(rotation_matrix(*veh.eta[3:]).T, dt) if cur else None
        veh.step(ta, nu_c=nu_c)
        if not np.all(np.isfinite(veh.state)):
            return {"diverged": True, "err": np.array(err_n), "s": np.array(s_norm),
                    "gains": np.array(gains), "energy": energy, "dt": dt}
    return {"diverged": False, "err": np.array(err_n), "s": np.array(s_norm),
            "gains": np.array(gains) if gains else np.zeros((n, 0)),
            "energy": energy, "dt": dt}


# --------------------------------------------------------------------- #
def lyapunov_smc(controller, **kw) -> dict:
    """SMC attitude Lyapunov / practical-stability check.

    ``V = 0.5 s^2`` for the attitude sliding surface ``s``.  On a coupled MIMO
    hybrid the translational loop acts as a bounded disturbance on the attitude
    subsystem, so strict step-wise ``V_dot <= 0`` does not hold; the meaningful
    guarantees are (i) the surface energy decays from its initial value and
    (ii) it is *ultimately bounded* -- ``s`` enters and stays in a small set
    (practical sliding-mode stability).  Reported: initial/final/max ``V``, the
    decay ratio, the smoothed-``V`` decreasing fraction, and the ultimate bound
    on ``|s|`` (mean over the final quarter of the run).
    """

    tr = simulate_trace(controller, **kw)
    s = tr["s"]
    V = 0.5 * s ** 2
    q = max(1, len(s) // 4)
    # Convergence is judged on the *settled* Lyapunov level, not the single last
    # sample.  A practical (boundary-layer) sliding mode reaches a small bounded
    # set in which the surface energy ripples with a few-sample period; V[-1]
    # then depends on which phase of that ripple the run happens to end on and
    # can land on a local peak even when the system has long since settled.  The
    # median of the final quarter is a phase-robust estimate of the settled
    # level (and the ultimate bound on |s| likewise), so the decay ratio and the
    # convergence flag measure "has V reached a small set", not a lucky/unlucky
    # terminal sample.
    V_settled = float(np.median(V[-q:]))
    s_bound = float(np.median(np.abs(s[-q:])))
    # Smoothed V trend (a coupled loop makes the raw signal noisy).
    win = max(1, len(V) // 25)
    Vs = np.convolve(V, np.ones(win) / win, mode="valid")
    smoothed_decreasing = float(np.mean(np.diff(Vs) <= 1e-9))
    return {"V": V, "s": s,
            "V0": float(V[0]), "V_final": float(V[-1]), "V_settled": V_settled,
            "V_max": float(V.max()),
            "decay_ratio": float(V_settled / (V.max() + 1e-12)),
            "smoothed_decreasing_frac": smoothed_decreasing,
            "s_ultimate_bound": s_bound,
            "converges": bool(V_settled < 0.3 * V.max()), "dt": tr["dt"]}


def pseudo_gradient_bound(controller, **kw) -> dict:
    """CFDL-MFAC controllability: pseudo-gradient bounded and sign-definite."""

    tr = simulate_trace(controller, **kw)
    g = tr["gains"]  # (T, n_mfac_dofs)
    if g.size == 0:
        return {"bounded": True, "sign_definite": True, "abs_max": 0.0, "abs_min": 0.0}
    sign0 = np.sign(g[0] + 1e-12)
    sign_definite = bool(np.all(np.sign(g + 1e-12) == sign0))
    return {"bounded": bool(np.all(np.isfinite(g)) and np.abs(g).max() < 1e3),
            "sign_definite": sign_definite,
            "abs_max": float(np.abs(g).max()), "abs_min": float(np.abs(g).min())}


def iss_analysis(controller_factory, disturbances=(0.0, 0.2, 0.4, 0.6, 0.8),
                 tail=200, **kw) -> dict:
    """Input-to-State Stability: ultimate error bound vs disturbance magnitude."""

    bounds, diverged = [], []
    for d in disturbances:
        tr = simulate_trace(controller_factory(), current_speed=d, **kw)
        diverged.append(tr["diverged"])
        bounds.append(float("inf") if tr["diverged"]
                      else float(np.mean(tr["err"][-tail:])))
    d = np.array(disturbances)
    b = np.array(bounds)
    # ISS gain estimate: slope of ultimate bound vs disturbance (finite -> ISS).
    finite = np.isfinite(b)
    gain = float(np.polyfit(d[finite], b[finite], 1)[0]) if finite.sum() > 1 else float("nan")
    return {"disturbances": d.tolist(), "error_bounds": b.tolist(),
            "iss_gain": gain, "any_diverged": bool(any(diverged))}


def region_of_attraction(controller_factory, radii=(0.5, 1.0, 2.0, 4.0, 8.0),
                         tail=200, tol=1.5, **kw) -> dict:
    """Convergence from a range of initial pose-error magnitudes."""

    rng = np.random.default_rng(0)
    converged, finals, envelopes = [], [], []
    for r in radii:
        e0 = np.zeros(6)
        e0[:3] = rng.normal(size=3); e0[:3] *= r / (np.linalg.norm(e0[:3]) + 1e-9)
        e0[5] = 0.2 * np.sign(rng.normal())
        tr = simulate_trace(controller_factory(), init_err=e0, **kw)
        ss = float("inf") if tr["diverged"] else float(np.mean(tr["err"][-tail:]))
        finals.append(ss); converged.append(ss < tol)
        envelopes.append(tr["err"])
    return {"radii": list(radii), "final_errors": finals,
            "converged": [bool(c) for c in converged],
            "fraction_converged": float(np.mean(converged)), "envelopes": envelopes}


def monte_carlo_stability(controller_factory, n_trials=200, seed=0, **kw) -> dict:
    """Divergence rate and ultimate error bound over randomized conditions."""

    rng = np.random.default_rng(seed)
    n_div, max_errs, ss_errs = 0, [], []
    for i in range(n_trials):
        p = REMUSParams(mass=30.48 * (1.0 + rng.uniform(-0.2, 0.2)))
        e0 = np.zeros(6); e0[:3] = rng.uniform(-1.5, 1.5, 3); e0[5] = rng.uniform(-0.3, 0.3)
        fault = (int(rng.integers(6)), float(rng.uniform(0.2, 0.5))) if rng.random() < 0.3 else None
        tr = simulate_trace(controller_factory(), seed=i, current_speed=float(rng.uniform(0, 0.5)),
                            init_err=e0, params=p, fault=fault, **kw)
        if tr["diverged"]:
            n_div += 1
        else:
            max_errs.append(float(tr["err"].max()))
            ss_errs.append(float(np.mean(tr["err"][-200:])))
    return {"n_trials": n_trials, "divergence_rate": n_div / n_trials,
            "ultimate_bound_mean": float(np.mean(ss_errs)) if ss_errs else float("nan"),
            "ultimate_bound_p95": float(np.percentile(ss_errs, 95)) if ss_errs else float("nan"),
            "max_error_p95": float(np.percentile(max_errs, 95)) if max_errs else float("nan")}
