"""Three robustness scenarios for the 6-DOF hybrid AUV controller.

Reproduces the standard AUV robustness battery (adapted to the REMUS 6-DOF plant
and this paper's controllers) used in the manuscript:

* **Scenario 1** - square trajectory; nominal case + measurement-noise robustness
  (additive white Gaussian noise on the velocity and position measurements at a
  range of signal-to-noise ratios, Tests 0-5).
* **Scenario 2** - lemniscate trajectory; sudden persistent external disturbances
  d = [d_u, d_v, d_r] active over 10-40 s (low-frequency stochastic surge,
  high-amplitude sinusoidal sway, step yaw moment).
* **Scenario 3** - circle trajectory; 100% time-varying parametric uncertainty on
  mass, inertia, added mass and (linear + quadratic) hydrodynamic damping.

The planar (x, y, psi) scenario references are embedded in the 6-DOF model with
zero depth/roll/pitch reference.  Disturbance magnitudes are expressed as the
same fraction of actuator authority as the source scenario (F_max = 2000 N) so
they are physically consistent with the REMUS thruster envelope
(tau_max = 50/30/30/10/20/20); the +/-400 N, 500 N, 400 N m literals would
otherwise exceed the REMUS actuator ~40x and merely saturate every controller.

Run::

    python -m cfdl_mfac_ncp_td3.experiments.scenario_experiment
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import default_config
from ..controllers import HybridController
from ..environment.residual_env import load_strong_baseline
from ..formation import tune_all, build_tuned
from ..dynamics import REMUS6DOF, REMUSParams, ThrusterModel
from ..dynamics.remus6dof import jacobian
from ..observers import DisturbanceObserver
from ..trajectories import make_trajectory
from ..benchmark.base import pose_error

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "scenarios")
F_MAX_REF = 2000.0            # source-scenario control range used for scaling
DIST_WINDOW = (10.0, 40.0)    # Scenario 2 disturbance active window [s]


def _log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# Core simulator (true plant noise-free; only the *measurement* is corrupted)
# --------------------------------------------------------------------------- #
def simulate(make_ctrl, trajectory, eta0, nu0, duration=60.0,
             tau_dist_fn=None, meas_noise=None, param_unc=None, seed=0, observe=False):
    cfg = default_config()
    dt = cfg.sim.dt
    veh = REMUS6DOF(REMUSParams(), cfg.sim)
    veh.reset(eta=np.asarray(eta0, float), nu=np.asarray(nu0, float))
    ctrl = make_ctrl(); ctrl.reset()
    thr = ThrusterModel(cfg.thruster)
    traj = make_trajectory(trajectory, duration=duration)
    rng = np.random.default_rng(seed)
    nom = REMUSParams()  # nominal coefficients for the uncertainty baseline
    # Disturbance/total-uncertainty estimator (Chen NDOB, i.e. an ESO for the
    # lumped body-frame disturbance wrench f_hat(k)); observes the closed loop
    # with the *nominal* model, so it also captures the parametric-uncertainty
    # mismatch in Scenario 3.  Runs alongside the controller for panel (e).
    ndob = DisturbanceObserver(REMUSParams(), cfg.observer) if observe else None
    if ndob is not None:
        ndob.reset()
    n = int(duration / dt)

    T, ETA, ETAD, NU, TAU, FHAT, DEXT = [], [], [], [], [], [], []
    pe, ye, energy = [], [], 0.0
    tau_prev = np.zeros(6)
    for k in range(n):
        t = k * dt
        if param_unc is not None:
            param_unc(veh, nom, t)                       # time-varying plant params
        if ndob is not None:                             # update ESO with last wrench
            FHAT.append(ndob.update(veh.eta, veh.nu, tau_prev, dt).copy())
        eta_d, eta_d_dot = traj.reference(t)
        eta_m, nu_m = veh.eta.copy(), veh.nu.copy()
        if meas_noise is not None:
            eta_m = eta_m + meas_noise["eta"] * rng.standard_normal(6)
            nu_m = nu_m + meas_noise["nu"] * rng.standard_normal(6)
        tau = ctrl.control(eta_m, nu_m, eta_d, eta_d_dot, dt)
        ta = thr.step(tau, dt)
        energy += float(ta @ ta) * dt
        td = tau_dist_fn(t) if tau_dist_fn is not None else None
        veh.step(ta, tau_dist=td)
        tau_prev = ta
        e = pose_error(eta_d, veh.eta)
        T.append(t); ETA.append(veh.eta.copy()); ETAD.append(eta_d.copy())
        NU.append(veh.nu.copy()); TAU.append(ta.copy())
        DEXT.append(td.copy() if td is not None else np.zeros(6))
        pe.append(float(np.hypot(e[0], e[1]))); ye.append(abs(float(e[5])))
        if not np.all(np.isfinite(veh.state)):
            break
    pe, ye = np.array(pe), np.array(ye)
    return {"t": np.array(T), "eta": np.array(ETA), "eta_d": np.array(ETAD),
            "nu": np.array(NU), "tau": np.array(TAU),
            "f_hat": np.array(FHAT) if FHAT else np.zeros((len(T), 6)),
            "d_ext": np.array(DEXT), "pos_err": pe, "yaw_err": ye,
            "pos_rmse": float(np.sqrt(np.mean(pe ** 2))),
            "yaw_rmse": float(np.sqrt(np.mean(ye ** 2))),
            "energy": energy, "diverged": not np.all(np.isfinite(veh.state))}


# --------------------------------------------------------------------------- #
# Controllers under test (SMC excluded, per the study's comparison set)
# --------------------------------------------------------------------------- #
def build_controllers(tune_budget=30):
    cfg = default_config(); tm = cfg.thruster.tau_max
    _log(f"Fair tuning baselines (budget={tune_budget}) ...")
    tuning = tune_all(n_samples=tune_budget, seed=0, verbose=True)
    BL = load_strong_baseline()
    return {
        "Hybrid": lambda: HybridController(cfg, use_observers=False, use_supervisor=False, **BL),
        "MPC": lambda: build_tuned("MPC", tuning, tm),
        "PID": lambda: build_tuned("PID", tuning, tm),
        "Fuzzy": lambda: build_tuned("Fuzzy", tuning, tm),
    }


# --------------------------------------------------------------------------- #
# Scenario 1 - square trajectory + measurement-noise robustness
# --------------------------------------------------------------------------- #
SNR_V = [None, 60, 40, 30, 25, 20]     # velocity-measurement SNR [dB], Tests 0-5
SNR_ETA = 80                           # position-measurement SNR [dB]


def _noise_std(ref_rms, snr_db):
    """AWGN std for a per-channel signal RMS at the given SNR (dB)."""
    return ref_rms / (10.0 ** (snr_db / 20.0))


def scenario1(ctrls, n_seeds=8):
    eta0 = np.zeros(6); nu0 = np.array([0.5, 0, 0, 0, 0, 0])
    _log("\n" + "=" * 72 + "\nSCENARIO 1 - square trajectory + measurement noise\n" + "=" * 72)
    # per-controller signal RMS from a nominal run (reference for the SNR noise)
    ref = {}
    nominal = {}
    for name, mk in ctrls.items():
        r = simulate(mk, "square", eta0, nu0)
        nominal[name] = r
        ref[name] = {"eta": np.sqrt(np.mean(r["eta"] ** 2, axis=0)) + 1e-6,
                     "nu": np.sqrt(np.mean(np.gradient(r["eta"], axis=0) ** 2, axis=0)) + 1e-3}
    table = {name: [] for name in ctrls}
    for ti, snr_v in enumerate(SNR_V):
        for name, mk in ctrls.items():
            if snr_v is None:
                r = nominal[name]
                table[name].append((r["pos_rmse"], r["yaw_rmse"]))
            else:
                pr, yr = [], []
                for s in range(n_seeds):
                    mn = {"eta": _noise_std(ref[name]["eta"], SNR_ETA),
                          "nu": _noise_std(ref[name]["nu"], snr_v)}
                    r = simulate(mk, "square", eta0, nu0, meas_noise=mn, seed=100 + s)
                    pr.append(r["pos_rmse"]); yr.append(r["yaw_rmse"])
                table[name].append((float(np.mean(pr)), float(np.mean(yr))))
    _report_noise_table(table)
    _plot_xy(nominal, "square", "scenario1_square_xy.png",
             "Scenario 1: square-trajectory tracking (nominal)")
    _plot_noise(table, "scenario1_noise.png")
    return {"snr_v": SNR_V, "snr_eta": SNR_ETA,
            "pos_rmse": {n: [v[0] for v in table[n]] for n in table},
            "yaw_rmse": {n: [v[1] for v in table[n]] for n in table}}


# --------------------------------------------------------------------------- #
# Scenario 2 - lemniscate trajectory + external disturbances
# --------------------------------------------------------------------------- #
def make_disturbance(seed=0):
    cfg = default_config(); tm = cfg.thruster.tau_max
    du_amp = 400.0 / F_MAX_REF * tm[0]     # 20% of surge authority  -> 10 N
    dv_amp = 500.0 / F_MAX_REF * tm[1]     # 25% of sway authority   -> 7.5 N
    dr_amp = 400.0 / F_MAX_REF * tm[5]     # 20% of yaw authority    -> 4 N m
    rng = np.random.default_rng(seed)
    t0, t1 = DIST_WINDOW
    samp_t = np.arange(t0, t1 + 1e-9, 0.6)
    samp_v = rng.uniform(-du_amp, du_amp, size=len(samp_t))   # low-freq stochastic surge

    def dist(t):
        d = np.zeros(6)
        if t0 <= t <= t1:
            d[0] = float(np.interp(t, samp_t, samp_v))          # d_u  stochastic
            d[1] = dv_amp * np.sin(t)                            # d_v  sinusoidal
            d[5] = dr_amp                                       # d_r  step
        return d
    return dist, (du_amp, dv_amp, dr_amp)


def scenario2(ctrls):
    eta0 = np.zeros(6); nu0 = np.array([0.3, 0.6, 0, 0, 0, np.pi / 15])
    dist, amps = make_disturbance()
    _log("\n" + "=" * 72 + "\nSCENARIO 2 - lemniscate + external disturbances (10-40 s)\n" + "=" * 72)
    _log(f"  scaled disturbances: d_u=+/-{amps[0]:.1f} N (stochastic), "
         f"d_v={amps[1]:.1f} sin(t) N, d_r={amps[2]:.1f} N m (step)")
    runs, rows = {}, {}
    for name, mk in ctrls.items():
        r = simulate(mk, "lemniscate", eta0, nu0, tau_dist_fn=dist)
        runs[name] = r
        m = (DIST_WINDOW[0] <= r["t"]) & (r["t"] <= DIST_WINDOW[1])
        rows[name] = {"pos_rmse": r["pos_rmse"], "yaw_rmse": r["yaw_rmse"],
                      "pos_rmse_dist": float(np.sqrt(np.mean(r["pos_err"][m] ** 2))),
                      "yaw_rmse_dist": float(np.sqrt(np.mean(r["yaw_err"][m] ** 2))),
                      "energy": r["energy"]}
    _report_scalar_table("Scenario 2 (overall / during-disturbance)", rows,
                         ["pos_rmse", "yaw_rmse", "pos_rmse_dist", "yaw_rmse_dist", "energy"])
    _plot_xy(runs, "lemniscate", "scenario2_lemniscate_xy.png",
             "Scenario 2: lemniscate tracking under disturbance")
    _plot_error_time(runs, "scenario2_error.png",
                     "Scenario 2: position error (disturbance 10-40 s shaded)", shade=DIST_WINDOW)
    return rows


# --------------------------------------------------------------------------- #
# Scenario 3 - circle trajectory + parametric uncertainty
# --------------------------------------------------------------------------- #
def param_uncertainty(veh, nom, t):
    """Apply 100% time-varying uncertainty to the plant coefficients."""
    p = veh.p
    # mass / inertia: constant +100%
    p.mass = 2.0 * nom.mass
    p.Iz = 2.0 * nom.Iz
    p.weight = p.mass * p.gravity
    # added mass (time-varying)
    p.X_udot = nom.X_udot * (1.0 + np.sin(t + np.pi / 6))
    p.Y_vdot = nom.Y_vdot * (1.0 + np.sin(t + np.pi / 5))
    p.N_rdot = nom.N_rdot * (1.0 + np.sin(0.8 * t))
    # linear damping (time-varying)
    p.Xu = nom.Xu * (1.0 + np.sin(0.8 * t + np.pi / 4))
    p.Yv = nom.Yv * (1.0 + np.sin(0.8 * t + np.pi / 3))
    p.Nr = nom.Nr * (1.0 + np.sin(0.8 * t))
    # quadratic damping (time-varying)  (D_u, D_v, D_r -> Xuu, Yvv, Nrr)
    p.Xuu = nom.Xuu * (1.0 + np.sin(t + np.pi / 6))
    p.Yvv = nom.Yvv * (1.0 + np.sin(t + np.pi / 5))
    p.Nrr = nom.Nrr * (1.0 + np.sin(0.8 * t + np.pi / 4))
    # mass changed -> refresh cached mass matrix
    veh.M = p.mass_matrix()
    veh.Minv = np.linalg.inv(veh.M)


def scenario3(ctrls):
    eta0 = np.array([3, 0, 0, 0, 0, 0]); nu0 = np.array([0, 0.5, 0, 0, 0, np.pi / 10])
    _log("\n" + "=" * 72 + "\nSCENARIO 3 - circle + 100% time-varying parametric uncertainty\n" + "=" * 72)
    runs, rows = {}, {}
    for name, mk in ctrls.items():
        r = simulate(mk, "circle", eta0, nu0, param_unc=param_uncertainty)
        runs[name] = r
        rows[name] = {"pos_rmse": r["pos_rmse"], "yaw_rmse": r["yaw_rmse"], "energy": r["energy"]}
    _report_scalar_table("Scenario 3 (parametric uncertainty)", rows,
                         ["pos_rmse", "yaw_rmse", "energy"])
    _plot_xy(runs, "circle", "scenario3_circle_xy.png",
             "Scenario 3: circle tracking under 100% parametric uncertainty")
    _plot_error_time(runs, "scenario3_error.png",
                     "Scenario 3: position error under parametric uncertainty")
    return rows


# --------------------------------------------------------------------------- #
# Reporting / plotting helpers
# --------------------------------------------------------------------------- #
def _report_noise_table(table):
    hdr = "  {:<8s}".format("Test") + "".join(f"{n:>22s}" for n in table)
    _log(hdr); _log("  " + "-" * (8 + 22 * len(table)))
    labels = ["0 (none)", "1 (60dB)", "2 (40dB)", "3 (30dB)", "4 (25dB)", "5 (20dB)"]
    for ti, lab in enumerate(labels):
        row = "  {:<8s}".format(lab)
        for n in table:
            pr, yr = table[n][ti]
            row += f"{pr:>10.4f}/{yr:<11.4f}"
        _log(row)
    _log("  (values: position RMSE [m] / yaw RMSE [rad])")


def _report_scalar_table(title, rows, keys):
    _log(title)
    hdr = "  {:<10s}".format("ctrl") + "".join(f"{k:>16s}" for k in keys)
    _log(hdr)
    for n, d in rows.items():
        _log("  {:<10s}".format(n) + "".join(f"{d[k]:>16.4f}" for k in keys))


def _plot_xy(runs, trajectory, fname, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ref = next(iter(runs.values()))
    ax.plot(ref["eta_d"][:, 0], ref["eta_d"][:, 1], "k--", lw=1.6, label="reference")
    colors = {"Hybrid": "#27ae60", "MPC": "#2c7fb8", "PID": "#c0392b", "Fuzzy": "#e67e22"}
    for n, r in runs.items():
        ax.plot(r["eta"][:, 0], r["eta"][:, 1], lw=1.3, color=colors.get(n), label=n)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=180); plt.close(fig)


def _plot_error_time(runs, fname, title, shade=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colors = {"Hybrid": "#27ae60", "MPC": "#2c7fb8", "PID": "#c0392b", "Fuzzy": "#e67e22"}
    for n, r in runs.items():
        ax.plot(r["t"], r["pos_err"], lw=1.1, color=colors.get(n), label=n)
    if shade is not None:
        ax.axvspan(shade[0], shade[1], color="grey", alpha=0.12)
    ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=180); plt.close(fig)


def _plot_noise(table, fname):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = [0, 1, 2, 3, 4, 5]
    xt = ["none", "60", "40", "30", "25", "20"]
    colors = {"Hybrid": "#27ae60", "MPC": "#2c7fb8", "PID": "#c0392b", "Fuzzy": "#e67e22"}
    for n in table:
        ax.plot(xs, [v[0] for v in table[n]], "-o", ms=4, color=colors.get(n), label=n)
    ax.set_xticks(xs); ax.set_xticklabels(xt)
    ax.set_xlabel("velocity-measurement SNR [dB]"); ax.set_ylabel("position RMSE [m]")
    ax.set_title("Scenario 1: robustness to velocity-measurement noise", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=180); plt.close(fig)


def _write_report(s1, s2, s3):
    L = ["# Robustness Scenarios (6-DOF hybrid AUV controller)\n",
         "Three robustness scenarios adapted to the REMUS 6-DOF plant and this "
         "study's controllers (Hybrid vs MPC, PID, Fuzzy-PID). Planar references "
         "(x, y, psi) are embedded with zero depth/roll/pitch. Disturbance "
         "magnitudes are scaled to the same fraction of actuator authority as the "
         "source scenario (F_max = 2000 N).\n",
         "## Scenario 1 - square trajectory + measurement noise\n",
         "Position RMSE [m] per velocity-measurement SNR (position SNR = 80 dB):\n",
         "| Controller | Test 0 (none) | 60 dB | 40 dB | 30 dB | 25 dB | 20 dB |",
         "|---|---|---|---|---|---|---|"]
    for n in s1["pos_rmse"]:
        L.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in s1["pos_rmse"][n]) + " |")
    L.append("\n## Scenario 2 - lemniscate + external disturbances (10-40 s)\n")
    L.append("| Controller | pos RMSE | yaw RMSE | pos RMSE (dist.) | yaw RMSE (dist.) | energy |")
    L.append("|---|---|---|---|---|---|")
    for n, d in s2.items():
        L.append(f"| {n} | {d['pos_rmse']:.4f} | {d['yaw_rmse']:.4f} | "
                 f"{d['pos_rmse_dist']:.4f} | {d['yaw_rmse_dist']:.4f} | {d['energy']:.1f} |")
    L.append("\n## Scenario 3 - circle + 100% time-varying parametric uncertainty\n")
    L.append("| Controller | pos RMSE | yaw RMSE | energy |")
    L.append("|---|---|---|---|")
    for n, d in s3.items():
        L.append(f"| {n} | {d['pos_rmse']:.4f} | {d['yaw_rmse']:.4f} | {d['energy']:.1f} |")
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Per-scenario 6-panel diagnostic figure for the HYBRID controller
#   (a) xy tracking  (b) tracking errors  (c) body velocity  (d) control input
#   (e) ESO total-disturbance estimate f_hat(k)  (f) external disturbance
# --------------------------------------------------------------------------- #
IDX = [0, 1, 5]                      # active planar DOFs: surge, sway, yaw
LBL = ["surge", "sway", "yaw"]
CL = ["#2c7fb8", "#e67e22", "#27ae60"]


def hybrid_panels(scenario, fname, title):
    cfg = default_config(); BL = load_strong_baseline()
    mk = lambda: HybridController(cfg, use_observers=False, use_supervisor=False, **BL)
    if scenario == "square":
        r = simulate(mk, "square", np.zeros(6), np.array([0.5, 0, 0, 0, 0, 0]), observe=True)
    elif scenario == "lemniscate":
        dist, _ = make_disturbance()
        r = simulate(mk, "lemniscate", np.zeros(6), np.array([0.3, 0.6, 0, 0, 0, np.pi / 15]),
                     tau_dist_fn=dist, observe=True)
    else:  # circle
        r = simulate(mk, "circle", np.array([3, 0, 0, 0, 0, 0]),
                     np.array([0, 0.5, 0, 0, 0, np.pi / 10]), param_unc=param_uncertainty, observe=True)

    t = r["t"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=13, y=0.995)

    # (a) reference and response in the xy-plane
    a = ax[0, 0]
    a.plot(r["eta_d"][:, 0], r["eta_d"][:, 1], "k--", lw=1.8, label="reference")
    a.plot(r["eta"][:, 0], r["eta"][:, 1], "#27ae60", lw=1.4, label="Hybrid")
    a.plot(r["eta"][0, 0], r["eta"][0, 1], "ko", ms=5)
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.axis("equal")
    a.set_title("(a) Reference and response (xy-plane)", fontsize=10)
    a.legend(fontsize=8, frameon=False)

    # (b) tracking errors (x, y, yaw); yaw wrapped to (-pi, pi]
    b = ax[0, 1]
    ex = r["eta"][:, 0] - r["eta_d"][:, 0]
    ey = r["eta"][:, 1] - r["eta_d"][:, 1]
    dpsi = r["eta"][:, 5] - r["eta_d"][:, 5]
    epsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))
    b.plot(t, ex, CL[0], lw=1.1, label="$x-x_d$")
    b.plot(t, ey, CL[1], lw=1.1, label="$y-y_d$")
    b.plot(t, epsi, CL[2], lw=1.1, label=r"$\psi-\psi_d$ [rad]")
    b.axhline(0, color="grey", lw=0.6)
    b.set_xlabel("t [s]"); b.set_ylabel("tracking error")
    b.set_title("(b) Tracking errors", fontsize=10); b.legend(fontsize=8, frameon=False)

    # (c) body-fixed velocities (u, v, r)
    c = ax[0, 2]
    for j, i in enumerate(IDX):
        lab = ["u [m/s]", "v [m/s]", "r [rad/s]"][j]
        c.plot(t, r["nu"][:, i], CL[j], lw=1.1, label=lab)
    c.set_xlabel("t [s]"); c.set_ylabel("body velocity")
    c.set_title("(c) AUV velocity (body frame)", fontsize=10); c.legend(fontsize=8, frameon=False)

    # (d) control input
    d = ax[1, 0]
    for j, i in enumerate(IDX):
        d.plot(t, r["tau"][:, i], CL[j], lw=1.0, label=r"$\tau_{%s}$" % LBL[j])
    d.set_xlabel("t [s]"); d.set_ylabel("control input [N, N·m]")
    d.set_title("(d) Control input", fontsize=10); d.legend(fontsize=8, frameon=False)

    # (e) ESO-estimated total (incremental) disturbance f_hat(k).  The observer
    # start-up transient (first ~1 s) is excluded from the y-scale so the steady
    # estimate is legible.
    e = ax[1, 1]
    for j, i in enumerate(IDX):
        e.plot(t, r["f_hat"][:, i], CL[j], lw=1.0, label=r"$\hat f_{%s}$" % LBL[j])
    warm = t > 1.0
    if np.any(warm):
        fw = r["f_hat"][warm][:, IDX]
        lo, hi = float(fw.min()), float(fw.max())
        pad = 0.15 * (hi - lo + 1e-6)
        e.set_ylim(lo - pad, hi + pad)
    e.set_xlabel("t [s]"); e.set_ylabel(r"$\hat f(k)$ [N, N·m]")
    e.set_title("(e) Estimated total disturbance (ESO)", fontsize=10); e.legend(fontsize=8, frameon=False)

    # (f) external disturbances
    f = ax[1, 2]
    if np.any(np.abs(r["d_ext"]) > 1e-9):
        for j, i in enumerate(IDX):
            f.plot(t, r["d_ext"][:, i], CL[j], lw=1.0, label=r"$d_{%s}$" % LBL[j])
        f.legend(fontsize=8, frameon=False)
    else:
        f.text(0.5, 0.5, "no external disturbance\n(nominal / parametric-uncertainty case)",
               ha="center", va="center", transform=f.transAxes, fontsize=9, color="grey")
    f.axhline(0, color="grey", lw=0.6)
    f.set_xlabel("t [s]"); f.set_ylabel("external disturbance [N, N·m]")
    f.set_title("(f) External disturbances", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=170); plt.close(fig)
    _log(f"  wrote {fname}  (pos_rmse={r['pos_rmse']:.4f}, yaw_rmse={r['yaw_rmse']:.4f})")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ctrls = build_controllers()
    s1 = scenario1(ctrls)
    s2 = scenario2(ctrls)
    s3 = scenario3(ctrls)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"scenario1": s1, "scenario2": s2, "scenario3": s3}, f, indent=2, default=float)
    _write_report(s1, s2, s3)

    _log("\n" + "=" * 72 + "\nHYBRID 6-panel diagnostic figures (per scenario)\n" + "=" * 72)
    hybrid_panels("square", "hybrid_scenario1_panels.png",
                  "Scenario 1 (square trajectory) - hybrid controller")
    hybrid_panels("lemniscate", "hybrid_scenario2_panels.png",
                  "Scenario 2 (lemniscate + external disturbances) - hybrid controller")
    hybrid_panels("circle", "hybrid_scenario3_panels.png",
                  "Scenario 3 (circle + parametric uncertainty) - hybrid controller")
    _log(f"\nArtifacts -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
