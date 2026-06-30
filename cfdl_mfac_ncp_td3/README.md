# CFDL-MFAC-NCP-TD3: Hybrid Intelligent Control for AUVs

A research-grade, modular framework for **fault-tolerant, disturbance-robust
6-DOF trajectory tracking** of a REMUS-class autonomous underwater vehicle
(AUV).  It fuses a **model-free adaptive controller** (CFDL-MFAC) with a
**learned policy** (TD3), arbitrated by a **confidence-guided supervisor** that
can be gated by a **Neural Circuit Policy** (NCP / liquid network), and backed
by an **observer suite** (state / disturbance / fault).

The framework was built in *verified stages*: every module is importable and
unit-tested in isolation (`pytest cfdl_mfac_ncp_td3/tests` — **68 tests**).

## Architecture

```
                         ┌──────────────────── observers ────────────────────┐
                         │  state (EKF)   disturbance (NDOB)   fault (RLS)     │
                         └───────────────┬─────────────────────┬──────────────┘
                                         │ d_hat               │ theta
 eta_d ──▶ outer kinematic loop ──▶ nu_d │                     │
                                         ▼                     ▼
                       ┌─────────── CFDL-MFAC (per-DOF SISO) ──────────┐  tau_mfac
                       │   Delta y = Phi Delta u, online PJM estimate  │────────┐
                       └───────────────────────────────────────────────┘        │
                                                                                 ▼
 state ───────────────────────▶ TD3 policy ─── tau_rl ──────────▶ confidence-guided
                                                                   supervisor (NCP-gated)
                                                                         │ alpha
                                                                         ▼
                                                  tau = (1-alpha) tau_mfac + alpha tau_rl
```

* **Outer loop** maps pose error → a *feasible* desired body velocity `nu_d`.
* **Inner CFDL-MFAC** is a bank of decentralised SISO adaptive loops driving
  `nu → nu_d` with no plant model (online pseudo-Jacobian identification).
* **TD3** provides a robust learned policy, trained with curriculum learning
  and domain randomization.
* **Supervisor** blends the two: nominal conditions keep authority with the
  adaptive controller; growing error / disturbance / fault hand authority to
  the learned policy.

## Module map

| Package | Contents |
|---|---|
| `dynamics/` | Fossen 6-DOF REMUS plant, hydrodynamics, ocean current, thruster/allocation |
| `controllers/` | CFDL pseudo-gradient, MFAC, confidence, supervisor, integrated `HybridController` |
| `observers/` | EKF state observer, nonlinear disturbance observer, adaptive fault observer |
| `ncp/` | NCP wiring + Liquid Time-Constant network |
| `rl/` | TD3 agent (actor, twin critics, replay buffer) |
| `environment/` | Gymnasium AUV tracking env (randomization, curriculum, faults) |
| `trajectories/` | setpoint, sinusoidal, helix, waypoint, lawnmower references |
| `benchmark/` | PID, SMC, backstepping, MPC baselines |
| `training/` | TD3 training loop with curriculum |
| `evaluation/` | metrics + closed-loop rollout harness |
| `statistics/` | Monte-Carlo evaluation + significance tests |
| `visualization/` | publication-quality plots |
| `experiments/` | ablation study + benchmark comparison |

## Quick start

```bash
pip install -r cfdl_mfac_ncp_td3/requirements.txt

# Quick end-to-end demo (no RL training) — writes plots to ./outputs
python -m cfdl_mfac_ncp_td3.main demo --trajectory helix

# Compare the hybrid controller against the classical benchmarks
python -m cfdl_mfac_ncp_td3.main evaluate --trajectory sinusoidal

# Train the TD3 policy, then run the ablation study
python -m cfdl_mfac_ncp_td3.main train --steps 50000 --save td3.pt
python -m cfdl_mfac_ncp_td3.main ablate --trials 20 --load td3.pt

# Run the test suite
pytest cfdl_mfac_ncp_td3/tests -q
```

## Design notes & honest caveats

* **Hydrodynamic parameters** (rigid-body inertia, added mass) follow
  Prestero's REMUS-100 identification; the **damping** model uses a
  representative linear + quadratic diagonal structure (off-diagonal
  cross-flow coupling can be added without API changes).
* The plant exposes a **fully-actuated generalised-force** interface so that
  every controller — adaptive, learned and classical — is compared on equal
  footing.
* The CFDL-MFAC inner loop is **decentralised per-DOF**: the body-velocity
  dynamics are diagonally dominant, which makes a bank of SISO adaptive loops
  far more robust than a single coupled MIMO loop (the Coriolis coupling is
  absorbed as a per-channel disturbance).
* Observer estimates inform the **confidence supervisor**; explicit
  disturbance feed-forward is available (`disturbance_feedforward=True`) but
  off by default because the adaptive loop already rejects slow disturbances.
* On *nominal* tasks, well-tuned model-based baselines (backstepping/SMC) can
  match or beat the model-free hybrid; the hybrid's value is **robustness**
  under model uncertainty, currents and actuator faults, and graceful
  hand-over to a learned policy when adaptation is insufficient.

## References

* T. Prestero, *Verification of a Six-Degree of Freedom Simulation Model for
  the REMUS AUV*, MIT/WHOI, 2001.
* T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*,
  Wiley, 2011.
* Z. Hou, S. Jin, *Model Free Adaptive Control: Theory and Applications*, 2013.
* S. Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic
  Methods* (TD3), ICML 2018.
* M. Lechner et al., *Neural Circuit Policies Enabling Auditable Autonomy*,
  Nature Machine Intelligence, 2020.
* R. Hasani et al., *Liquid Time-constant Networks*, AAAI 2021.
* W.-H. Chen et al., *A Nonlinear Disturbance Observer for Robotic
  Manipulators*, IEEE T-IE, 2000.
