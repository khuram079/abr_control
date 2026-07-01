"""Stage-by-stage demonstration driver for the CFDL-MFAC-NCP-TD3 framework.

Each ``stage_k`` function exercises the corresponding subsystem end-to-end,
prints a concise results table and writes output artifacts (plots / logs) into
``cfdl_mfac_ncp_td3/results/stageK/``.  Run a single stage or all of them::

    python -m cfdl_mfac_ncp_td3.experiments.run_all_stages --stage 1
    python -m cfdl_mfac_ncp_td3.experiments.run_all_stages --all
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _dir(stage: int) -> str:
    d = os.path.join(RESULTS_ROOT, f"stage{stage}")
    os.makedirs(d, exist_ok=True)
    return d


def _banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ====================================================================== #
# Stage 1 - 6-DOF REMUS simulator
# ====================================================================== #
def stage_1() -> dict:
    _banner("STAGE 1  |  6-DOF REMUS AUV simulator")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..config import SimConfig
    from ..dynamics import REMUS6DOF, REMUSParams, rotation_matrix, OceanCurrent
    from ..config import CurrentConfig

    out = _dir(1)
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.02))

    # (a) free-decay energy dissipation
    veh.reset(nu=np.array([1.5, 0.3, 0.2, 0.0, 0.1, 0.2]))
    e0 = veh.kinetic_energy()
    energies = [e0]
    for _ in range(600):
        veh.step(np.zeros(6))
        energies.append(veh.kinetic_energy())
    print(f"(a) Free decay: KE {e0:.2f} J -> {energies[-1]:.4f} J "
          f"({100*(1-energies[-1]/e0):.2f}% dissipated), monotone="
          f"{all(np.diff(energies) <= 1e-9)}")

    # (b) surge-thrust step response
    veh.reset()
    surge = []
    for _ in range(300):
        veh.step(np.array([20.0, 0, 0, 0, 0, 0]))
        surge.append([veh.eta[0], veh.nu[0]])
    surge = np.array(surge)
    print(f"(b) 20 N surge for 6 s: u_terminal={surge[-1,1]:.3f} m/s, "
          f"x_travelled={surge[-1,0]:.3f} m")

    # (c) turn under yaw moment
    veh.reset()
    veh.step(np.array([15, 0, 0, 0, 0, 0]))  # give it speed
    for _ in range(60):
        veh.step(np.array([15, 0, 0, 0, 0, 3.0]))
    print(f"(c) Coordinated turn: heading psi={np.degrees(veh.eta[5]):.1f} deg, "
          f"yaw-rate r={veh.nu[5]:.3f} rad/s")

    # plot energy decay + step response
    t = np.arange(len(energies)) * 0.02
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, energies, "C0"); ax[0].set_title("Free-decay kinetic energy")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("KE [J]")
    ts = np.arange(len(surge)) * 0.02
    ax[1].plot(ts, surge[:, 1], "C1", label="surge u [m/s]")
    ax[1].plot(ts, surge[:, 0], "C2", label="position x [m]")
    ax[1].set_title("20 N surge step response"); ax[1].set_xlabel("t [s]")
    ax[1].legend()
    fig.tight_layout(); p = os.path.join(out, "stage1_dynamics.png")
    fig.savefig(p); plt.close(fig)
    print(f"    -> artifact: {p}")
    return {"dissipated_pct": 100 * (1 - energies[-1] / e0),
            "u_terminal": float(surge[-1, 1])}


# ====================================================================== #
# Stage 2 - CFDL-MFAC
# ====================================================================== #
def stage_2() -> dict:
    _banner("STAGE 2  |  CFDL pseudo-Jacobian + MFAC")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..config import MFACConfig
    from ..controllers import PseudoJacobianEstimator, CFDLMFAC

    out = _dir(2)
    # (a) PJM identification of a hidden static gain
    rng = np.random.default_rng(0)
    G = np.array([[2.0, 0.3], [-0.5, 1.5]])
    est = PseudoJacobianEstimator(2, 2, MFACConfig(eta=1.0, mu=0.1, phi_init=1.0))
    hist = []
    for _ in range(2000):
        du = rng.normal(size=2)
        est.update(G @ du, du)
        hist.append(np.linalg.norm(est.phi - G))
    print(f"(a) PJM identification: ||Phi - G|| {hist[0]:.3f} -> {hist[-1]:.4f}")
    print(f"    true G = {G.tolist()}")
    print(f"    est  Phi = {np.round(est.phi, 3).tolist()}")

    # (b) closed-loop tracking on a stable MIMO plant
    a = np.diag([0.7, 0.6, 0.8]); b = np.diag([0.4, 0.25, 0.6])
    ctrl = CFDLMFAC(3, 3, MFACConfig(rho=0.6, lam=0.5, phi_init=1.0, u_limit=2.0))
    y = np.zeros(3); y_ref = np.array([1.0, -0.5, 0.8]); errs = []
    for _ in range(400):
        u = ctrl.control(y, y_ref); y = a @ y + b @ u
        errs.append(np.linalg.norm(y - y_ref))
    print(f"(b) MIMO setpoint tracking: ||e|| {errs[0]:.3f} -> {errs[-1]:.5f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist, "C0"); ax[0].set_title("PJM identification error")
    ax[0].set_xlabel("update"); ax[0].set_ylabel("||Phi - G||")
    ax[1].semilogy(errs, "C3"); ax[1].set_title("MFAC tracking error")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("||y* - y||")
    fig.tight_layout(); p = os.path.join(out, "stage2_mfac.png")
    fig.savefig(p); plt.close(fig)
    print(f"    -> artifact: {p}")
    return {"pjm_final_err": hist[-1], "mfac_final_err": errs[-1]}


# ====================================================================== #
# Stage 3 - observers
# ====================================================================== #
def stage_3() -> dict:
    _banner("STAGE 3  |  Observer suite (state / disturbance / fault)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..config import SimConfig, ObserverConfig
    from ..dynamics import REMUS6DOF, REMUSParams
    from ..observers import DisturbanceObserver, FaultObserver, StateObserver

    out = _dir(3)
    # (a) NDOB convergence to a constant wrench disturbance
    veh = REMUS6DOF(REMUSParams(), SimConfig(dt=0.02))
    veh.reset(nu=np.array([1.0, 0, 0, 0, 0, 0]))
    dob = DisturbanceObserver(veh.p, ObserverConfig(ndob_gain=8.0))
    d_true = np.array([3.0, -2.0, 1.0, 0.0, 0.5, -0.4]); track = []
    for _ in range(2000):
        dob.update(veh.eta, veh.nu, np.zeros(6), 0.02)
        veh.step(np.zeros(6), tau_dist=d_true)
        track.append(np.linalg.norm(dob.d_hat - d_true))
    print(f"(a) NDOB: ||d_hat - d|| {track[0]:.3f} -> {track[-1]:.4f}")
    print(f"    d_true={d_true.tolist()}")
    print(f"    d_hat ={np.round(dob.d_hat, 2).tolist()}")

    # (b) fault observer effectiveness identification
    fo = FaultObserver(6, ObserverConfig(fault_adapt_gain=1.5, fault_forgetting=0.99))
    theta_true = np.array([1.0, 0.5, 0.8, 1.0, 0.3, 1.0]); rng = np.random.default_rng(0)
    ferr = []
    for _ in range(3000):
        tc = rng.uniform(-10, 10, size=6)
        fo.update(tc, theta_true * tc); ferr.append(np.linalg.norm(fo.theta - theta_true))
    print(f"(b) Fault obs: ||theta_hat - theta|| {ferr[0]:.3f} -> {ferr[-1]:.4f}")
    print(f"    theta_true={theta_true.tolist()}")
    print(f"    theta_hat ={np.round(fo.theta, 3).tolist()}")

    # (c) EKF denoising
    veh2 = REMUS6DOF(REMUSParams(), SimConfig(dt=0.05)); veh2.reset(nu=np.array([1, 0, .05, 0, 0, .02]))
    obs = StateObserver(REMUS6DOF(REMUSParams(), SimConfig(dt=0.05)),
                        ObserverConfig(process_noise=1e-4, meas_noise=1e-2))
    rng = np.random.default_rng(1); tau = np.array([6, 0, 0, 0, 0, 0.5]); raw, est = [], []
    for _ in range(300):
        veh2.step(tau); meas = veh2.state + rng.normal(0, 0.05, 12); obs.step(tau, meas)
        raw.append(np.linalg.norm(meas - veh2.state)); est.append(np.linalg.norm(obs.x - veh2.state))
    print(f"(c) EKF: raw meas err {np.mean(raw[50:]):.4f} -> filtered {np.mean(est[50:]):.4f} "
          f"({100*(1-np.mean(est[50:])/np.mean(raw[50:])):.1f}% reduction)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(track, "C0"); ax[0].set_title("NDOB error"); ax[0].set_xlabel("step")
    ax[1].plot(ferr, "C1"); ax[1].set_title("Fault-obs error"); ax[1].set_xlabel("step")
    ax[2].plot(raw, "C7", alpha=.5, label="raw"); ax[2].plot(est, "C2", label="EKF")
    ax[2].set_title("EKF denoising"); ax[2].legend(); ax[2].set_xlabel("step")
    fig.tight_layout(); p = os.path.join(out, "stage3_observers.png")
    fig.savefig(p); plt.close(fig)
    print(f"    -> artifact: {p}")
    return {"ndob_err": track[-1], "fault_err": ferr[-1]}


# ====================================================================== #
# Stage 4 - NCP
# ====================================================================== #
def stage_4() -> dict:
    _banner("STAGE 4  |  NCP wiring + Liquid Time-Constant network")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from ..config import NCPConfig
    from ..ncp import NCPWiring, LiquidNetwork

    out = _dir(4)
    wiring = NCPWiring(NCPConfig())
    print(f"(a) Wiring summary: {wiring.summary()}")

    torch.manual_seed(0)
    net = LiquidNetwork(NCPConfig(n_sensory=4, n_inter=8, n_command=6, n_motor=1))
    opt = torch.optim.Adam(net.parameters(), lr=0.05); losses = []
    for _ in range(400):
        x = torch.rand(64, 4) * 2 - 1; target = (x[:, :1] > 0).float()
        outp, _ = net(x); loss = torch.mean((outp - target) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    print(f"(b) LTC learns context->blend mapping: MSE {losses[0]:.4f} -> {losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, "C4"); ax.axhline(0.25, color="k", ls="--", lw=.8, label="baseline (0.25)")
    ax.set_title("NCP/LTC training loss"); ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.legend()
    fig.tight_layout(); p = os.path.join(out, "stage4_ncp.png")
    fig.savefig(p); plt.close(fig)
    print(f"    -> artifact: {p}")
    return {"ncp_final_loss": losses[-1]}


# ====================================================================== #
# Stage 5 - TD3
# ====================================================================== #
def stage_5(steps: int = 400) -> dict:
    _banner("STAGE 5  |  TD3 reinforcement-learning agent")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from ..config import TD3Config
    from ..rl import TD3

    out = _dir(5)
    torch.manual_seed(0); np.random.seed(0)
    cfg = TD3Config(batch_size=128, warmup_steps=500, exploration_noise=0.2,
                    actor_lr=1e-3, critic_lr=1e-3)
    agent = TD3(2, 1, 1.0, cfg)

    def episode(explore):
        pos = np.random.uniform(-1, 1); tgt = np.random.uniform(-1, 1); tot = 0.0
        s = np.array([pos, tgt], np.float32)
        for _ in range(20):
            if explore and agent.buffer.size < cfg.warmup_steps:
                a = np.random.uniform(-1, 1, 1)
            else:
                a = agent.select_action(s, noise=cfg.exploration_noise if explore else None)
            pos = np.clip(pos + 0.1 * a[0], -1.5, 1.5); r = -abs(pos - tgt)
            s2 = np.array([pos, tgt], np.float32)
            if explore:
                agent.store(s, a, r, s2, 0.0); agent.train()
            s = s2; tot += r
        return tot

    curve = []
    for i in range(steps):
        episode(True)
        if i % 20 == 0:
            curve.append(np.mean([episode(False) for _ in range(10)]))
    final = np.mean([episode(False) for _ in range(30)])
    rand = np.mean([-abs(np.random.uniform(-1, 1) - np.random.uniform(-1, 1)) * 20
                    for _ in range(30)])
    print(f"(a) Point-mass reach: random-policy return ~ {rand:.2f}, "
          f"TD3 return = {final:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(curve)) * 20, curve, "C0-o", ms=3)
    ax.axhline(rand, color="k", ls="--", lw=.8, label="random policy")
    ax.set_title("TD3 learning curve"); ax.set_xlabel("training episode")
    ax.set_ylabel("eval return"); ax.legend()
    fig.tight_layout(); p = os.path.join(out, "stage5_td3.png")
    fig.savefig(p); plt.close(fig)
    print(f"    -> artifact: {p}")
    return {"td3_return": float(final), "random_return": float(rand)}


# ====================================================================== #
# Stage 6 - supervisor
# ====================================================================== #
def stage_6() -> dict:
    _banner("STAGE 6  |  Confidence-guided fusion supervisor")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..config import SupervisorConfig
    from ..controllers import ConfidenceEstimator

    out = _dir(6)
    scenarios = {
        "nominal": dict(error=np.zeros(6)),
        "moderate error": dict(error=np.full(6, 0.5)),
        "large error": dict(error=np.full(6, 2.0)),
        "fault (theta=0.2)": dict(error=np.zeros(6), theta=np.array([1, 1, .2, 1, 1, 1.])),
        "disturbance": dict(error=np.zeros(6), d_hat=np.full(6, 8.0)),
        "error+fault+dist": dict(error=np.full(6, 1.5), theta=np.full(6, .3),
                                 d_hat=np.full(6, 6.0)),
    }
    print(f"(a) RL authority (alpha) vs operating condition:")
    print(f"    {'scenario':<20s} {'confidence':>11s} {'alpha (RL)':>11s}")
    rows = []
    for name, kw in scenarios.items():
        est = ConfidenceEstimator(SupervisorConfig(smoothing=1.0))
        a = est.update(kw.get("error"), kw.get("d_hat"), kw.get("theta"))
        print(f"    {name:<20s} {est.confidence:>11.3f} {a:>11.3f}")
        rows.append((name, est.confidence, a))

    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r[0] for r in rows]; alphas = [r[2] for r in rows]; confs = [r[1] for r in rows]
    x = np.arange(len(names))
    ax.bar(x - .2, confs, .4, label="MFAC confidence", color="C0")
    ax.bar(x + .2, alphas, .4, label="RL authority (alpha)", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend(); ax.set_title("Confidence-guided authority allocation")
    fig.tight_layout(); p = os.path.join(out, "stage6_supervisor.png")
    fig.savefig(p); plt.close(fig)
    print(f"    -> artifact: {p}")
    return {"rows": rows}


# ====================================================================== #
# Stage 7 - benchmarks
# ====================================================================== #
def stage_7() -> dict:
    _banner("STAGE 7  |  Classical benchmark controllers")
    from ..config import default_config
    from ..benchmark import (PIDController, SMCController,
                             BacksteppingController, MPCController)
    from ..evaluation import rollout

    cfg = default_config(); tm = list(cfg.thruster.tau_max)
    ctrls = {"PID": PIDController(tau_max=tm), "SMC": SMCController(tau_max=tm),
             "Backstepping": BacksteppingController(tau_max=tm),
             "MPC": MPCController(tau_max=tm)}
    print(f"(a) Closed-loop pose regulation (setpoint), RMSE / IAE / energy:")
    print(f"    {'controller':<14s} {'rmse':>8s} {'iae':>8s} {'energy':>10s}")
    res = {}
    for name, c in ctrls.items():
        r = rollout(c, trajectory="setpoint", config=cfg, seed=0)
        m = r["metrics"]; res[name] = m
        print(f"    {name:<14s} {m['rmse']:>8.4f} {m['iae']:>8.2f} {m['control_energy']:>10.1f}")
    return res


# ====================================================================== #
# Stage 8 - integrated hybrid + comparison + ablation
# ====================================================================== #
def stage_8(trials: int = 12) -> dict:
    _banner("STAGE 8  |  Integrated hybrid controller, comparison & ablation")
    from ..config import default_config
    from ..controllers import HybridController
    from ..benchmark import (PIDController, SMCController,
                             BacksteppingController, MPCController)
    from ..evaluation import rollout
    from ..experiments import run_ablation
    from .. import visualization as viz

    out = _dir(8); cfg = default_config(); tm = list(cfg.thruster.tau_max)

    # (a) hybrid tracking demo on a helix, with plots
    hres = rollout(HybridController(cfg), trajectory="helix", config=cfg,
                   current=True, seed=0)
    print(f"(a) Hybrid on helix (with current): rmse={hres['metrics']['rmse']:.4f}, "
          f"pos-rmse={hres['metrics']['rmse_pos']:.4f}, diverged={hres['metrics']['diverged']}")
    viz.plot_trajectory_3d(hres["log"], os.path.join(out, "hybrid_helix_3d.png"))
    viz.plot_tracking_errors(hres["log"], os.path.join(out, "hybrid_helix_errors.png"))
    print(f"    -> artifacts: hybrid_helix_3d.png, hybrid_helix_errors.png")

    # (b) head-to-head comparison on sinusoidal (single deterministic run)
    controllers = {"Hybrid": HybridController(cfg), "PID": PIDController(tau_max=tm),
                   "SMC": SMCController(tau_max=tm),
                   "Backstepping": BacksteppingController(tau_max=tm),
                   "MPC": MPCController(tau_max=tm)}
    print(f"\n(b) Head-to-head on sinusoidal (with current), RMSE:")
    comp = {}
    for name, c in controllers.items():
        r = rollout(c, trajectory="sinusoidal", config=cfg, current=True, seed=0)
        comp[name] = r["metrics"]["rmse"]
    for name, v in sorted(comp.items(), key=lambda kv: kv[1]):
        print(f"    {name:<14s} rmse={v:.4f}")

    # (c) Monte-Carlo ablation under faults
    print(f"\n(c) Monte-Carlo ablation + benchmarks ({trials} randomized trials, "
          f"fault_prob=0.3), RMSE mean +/- std:")
    rep = run_ablation(cfg, td3_agent=None, n_trials=trials, trajectory="setpoint",
                       fault_prob=0.3, verbose=True)

    # comparison bar chart
    viz.plot_comparison_bars(rep["summaries"], "rmse",
                             os.path.join(out, "ablation_rmse.png"),
                             title="Monte-Carlo RMSE (faults + current)")
    print(f"    -> artifact: ablation_rmse.png")

    print(f"\n    Pairwise significance vs Hybrid (no RL):")
    ref = "Hybrid (no RL)"
    ref_dist = rep["results"][ref]["distributions"]["rmse"]
    from ..statistics import compare
    for name, mc in rep["results"].items():
        if name == ref:
            continue
        c = compare(ref, ref_dist, name, mc["distributions"]["rmse"])
        sig = "significant" if c["significant"] else "n.s."
        print(f"      vs {name:<16s} d={c['ttest']['cohens_d']:+.2f} "
              f"p={c['ttest']['p']:.3g}  ({sig})")
    return {"comparison": comp, "ablation": rep["summaries"]}


STAGES = {1: stage_1, 2: stage_2, 3: stage_3, 4: stage_4,
          5: stage_5, 6: stage_6, 7: stage_7, 8: stage_8}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run framework stages one by one")
    p.add_argument("--stage", type=int, choices=list(STAGES))
    p.add_argument("--all", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if args.all or args.stage is None:
        t0 = time.time()
        for k in sorted(STAGES):
            STAGES[k]()
        print(f"\nAll stages complete in {time.time()-t0:.1f}s. "
              f"Artifacts under {RESULTS_ROOT}/")
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
