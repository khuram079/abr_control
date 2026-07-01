# Stage-by-Stage Results

Reproduce with:

```bash
python -m cfdl_mfac_ncp_td3.experiments.run_all_stages --all
# or one stage at a time:
python -m cfdl_mfac_ncp_td3.experiments.run_all_stages --stage 3
pytest cfdl_mfac_ncp_td3/tests -q     # 68 tests
```

Every stage is unit-tested **and** produces the numerical output + plot below.
All plots are in `results/stageK/`.

---

## Stage 1 — 6-DOF REMUS simulator  (`test_dynamics.py`: 13 passed)

| Check | Result |
|---|---|
| Free-decay energy dissipation | 39.85 J → 0.0375 J (**99.91%** dissipated), monotone ✓ |
| 20 N surge step (6 s) | terminal surge **0.901 m/s**, travelled 6.476 m |
| Coordinated turn (yaw moment) | heading 9.5°, yaw-rate 0.226 rad/s |

Physical sanity confirmed: symmetric PD mass matrix, energy-conserving Coriolis,
dissipative damping, restoring stability. → `stage1/stage1_dynamics.png`

## Stage 2 — CFDL-MFAC  (`test_mfac.py`: 6 passed)

| Check | Result |
|---|---|
| PJM identification of hidden gain `G` | ‖Φ − G‖ 1.129 → **0.0000** (exact) |
| MIMO setpoint tracking | ‖e‖ 1.270 → **1e-5** |

Online pseudo-Jacobian converges to the true plant gain; model-free control law
drives the error to zero. → `stage2/stage2_mfac.png`

## Stage 3 — Observer suite  (`test_observers.py`: 6 passed)

| Observer | Result |
|---|---|
| Nonlinear disturbance observer | ‖d̂ − d‖ 208.7 → **0.0008** |
| Adaptive fault observer | ‖θ̂ − θ‖ 0.403 → **0.0000** (exact effectiveness ID) |
| EKF state observer | measurement noise reduced **75.7%** |

→ `stage3/stage3_observers.png`

## Stage 4 — NCP / Liquid network  (`test_ncp.py`: 7 passed)

- Wiring: 8 sensory / 12 inter / 6 command / 1 motor; 33 sensory + 67 recurrent synapses.
- LTC learns a context→blend mapping: MSE **0.25 (baseline) → 0.076**.

→ `stage4/stage4_ncp.png`

## Stage 5 — TD3  (`test_td3.py`: 5 passed)

- Point-mass reach task: random policy return ≈ **−12.89**, learned TD3 return = **−2.70** (≈4.8× better).

→ `stage5/stage5_td3.png`

## Stage 6 — Confidence-guided supervisor  (`test_supervisor.py`: 8 passed)

RL authority `alpha` vs operating condition (higher ⇒ more authority to the learned policy):

| Scenario | MFAC confidence | alpha (RL) |
|---|---|---|
| nominal | 1.000 | 0.000 |
| moderate error | 0.035 | 0.965 |
| large error | 0.018 | 0.982 |
| fault (θ=0.2) | 0.091 | 0.909 |
| disturbance | 0.146 | 0.854 |
| error+fault+dist | 0.000 | 1.000 |

Authority is monotone in error / fault / disturbance severity, as designed.
→ `stage6/stage6_supervisor.png`

## Stage 7 — Classical benchmarks  (`test_benchmarks.py`: 8 passed)

Closed-loop pose regulation (setpoint) around the full nonlinear plant:

| Controller | RMSE | IAE | Energy |
|---|---|---|---|
| PID | 0.696 | 18.96 | 9318 |
| SMC | 0.646 | 10.40 | 16743 |
| Backstepping | 0.859 | 17.05 | 18898 |
| MPC | 0.813 | 19.90 | 2846 |

All four stabilise and regulate the vehicle. → (no plot; see Stage 8 comparison)

## Stage 8 — Integrated hybrid + comparison + ablation  (`test_integration.py`: 15 passed)

**(a)** Hybrid on a helix (with current): RMSE **0.880**, position-RMSE 0.806, no divergence.
→ `stage8/hybrid_helix_3d.png`, `stage8/hybrid_helix_errors.png`

**(b)** Head-to-head on sinusoidal (with current), RMSE:
Backstepping 0.441 · SMC 0.681 · Hybrid 1.350 · PID 1.358 · MPC 1.823

**(c)** Monte-Carlo ablation (12 randomized trials, fault_prob = 0.3), RMSE mean ± std:

| Controller | RMSE |
|---|---|
| SMC | 0.666 ± 0.027 |
| PID | 0.684 ± 0.021 |
| Backstepping | 0.798 ± 0.062 |
| MPC | 0.948 ± 0.183 |
| MFAC / MFAC+Obs / Hybrid(no RL) / Hybrid | 0.977 ± 0.040 |

Pairwise significance vs *Hybrid (no RL)* (Cohen's d, paired-t p): PID d=+8.70 (p=4.5e-14),
SMC d=+8.63 (p=1.3e-14), Backstepping d=+3.29 (p=4.3e-7), MPC d=+0.21 (p=0.58, n.s.).
→ `stage8/ablation_rmse.png`

### Honest interpretation

- The ablation variants are **identical without a trained TD3** (`Hybrid == Hybrid (no RL)`):
  the supervisor blends MFAC with the RL policy, so with no policy loaded there is nothing
  to blend. Train first (`main.py train`) then pass the checkpoint to see the fusion effect.
- On these **mildly-randomized nominal tasks the well-tuned model-based controllers
  (SMC/backstepping) win** — they have structural model knowledge the model-free hybrid does
  not. The hybrid's intended advantage is **robustness under large model uncertainty, currents
  and actuator faults**, plus graceful hand-over to a learned policy; it is not expected to beat
  a tuned backstepping controller on a near-nominal plant. This is stated plainly rather than
  tuned away.
