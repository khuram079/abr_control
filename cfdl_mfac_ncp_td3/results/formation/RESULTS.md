# Formation Control — Results (leader + 3 followers)

Monte-Carlo: **500 simulations** on the `helix` trajectory with random initial positions, ocean current, sensor noise and actuator faults. All controllers evaluated on identical (paired) scenarios; all tuned by the same disclosed procedure.

## Five indicators (mean ± std)

| Controller | formation_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (ours) | 1.411 ± 0.760 | 0.085 ± 0.025 | 6.839 ± 5.783 | 287075.420 ± 19689.655 | 3.748 ± 1.831 |
| SMC | 1.306 ± 0.719 | 0.047 ± 0.016 | 6.255 ± 5.449 | 130193.418 ± 24206.582 | 3.536 ± 1.674 |
| Backstepping | 1.923 ± 1.251 | 0.654 ± 0.231 | 9.046 ± 7.400 | 197721.878 ± 48455.740 | 4.989 ± 2.817 |
| MPC | 1.435 ± 0.765 | 0.047 ± 0.023 | 7.198 ± 5.886 | 102191.049 ± 27583.263 | 3.816 ± 1.801 |
| PID | 13.874 ± 2.890 | 1.644 ± 0.102 | 36.835 ± 1.207 | 494156.014 ± 31872.999 | 29.273 ± 7.352 |
| Fuzzy | 14.932 ± 3.005 | 1.645 ± 0.141 | 37.184 ± 1.061 | 411248.668 ± 24570.304 | 30.968 ± 7.549 |

## Significance vs MPC / PID (paired t-test p, Wilcoxon p, Cohen's d)

### Hybrid (ours) vs MPC
| Indicator | Hybrid | MPC | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| formation_rmse | 1.411 | 1.435 | 3.28e-10 | 5.78e-08 | -0.03 |
| attitude_rmse | 0.085 | 0.047 | 1.88e-163 | 2.12e-81 | +1.60 |
| recovery_time | 6.839 | 7.198 | 4.09e-15 | 2.67e-30 | -0.06 |
| control_energy | 287075.420 | 102191.049 | 0.00e+00 | 1.26e-83 | +7.71 |
| max_deviation | 3.748 | 3.816 | 9.45e-09 | 1.92e-10 | -0.04 |

### Hybrid (ours) vs PID
| Indicator | Hybrid | PID | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| formation_rmse | 1.411 | 13.874 | 0.00e+00 | 1.26e-83 | -5.89 |
| attitude_rmse | 0.085 | 1.644 | 0.00e+00 | 1.26e-83 | -20.95 |
| recovery_time | 6.839 | 36.835 | 0.00e+00 | 1.26e-83 | -7.17 |
| control_energy | 287075.420 | 494156.014 | 0.00e+00 | 1.26e-83 | -7.81 |
| max_deviation | 3.748 | 29.273 | 1.32e-285 | 1.26e-83 | -4.76 |

## Parameter sensitivity (±20%)

- **recovery_threshold**: formation RMSE ['1.423', '1.397', '1.403', '1.393', '1.385'] across factors [0.8, 0.9, 1.0, 1.1, 1.2] → spread **2.7%** over the ±20% range (insensitive to parameter selection).
- **feedforward_cap**: formation RMSE ['1.392', '1.395', '1.403', '1.394', '1.404'] across factors [0.8, 0.9, 1.0, 1.1, 1.2] → spread **0.9%** over the ±20% range (insensitive to parameter selection).

## Fairness / tuning disclosure

Every controller was tuned by identical random search (budget = 40 samples), same validation objective (single-vehicle tracking RMSE over 2 trajectories × 2 seeds, current on). Best parameters:

| Controller | best validation RMSE | best params |
|---|---|---|
| PID | 8.1277 | {'kp': 0.769, 'ki': 2.617, 'kd': 1.537} |
| SMC | 0.5454 | {'lam': 0.622, 'kd': 1.721, 'ks': 2.803} |
| MPC | 0.5959 | {'q_pos': 32.93, 'q_vel': 2.764, 'r_u': 0.001} |
| Fuzzy | 10.1893 | {'ke': 2.072, 'e_width': 0.846, 'edot_width': 0.578} |
| Backstepping | 0.4552 | {'k1': 0.618, 'k2': 2.565} |
| Hybrid | 0.5803 | {'k_outer': 0.548, 'prediction_gain': 2.526, 'feedforward_cap': 0.082, 'att_lam': 2.097, 'att_kd': 1.427, 'att_ks': 2.969} |

## Honest verdict — CFDL-MFAC + SMC attitude architecture (v3)

Same architecture change as the single-vehicle rerun: CFDL-MFAC keeps
surge/sway/heave, roll/pitch/yaw now use a boundary-layer SMC law. See
`results/single_vehicle/RESULTS.md` for the full rationale.

**Ranking by formation RMSE:** SMC 1.306 < **Hybrid 1.411 < MPC 1.435** <
Backstepping 1.923 < PID 13.9 < Fuzzy 14.9.

- **The hybrid now ranks 2nd of 6**, beating MPC (1.411 vs 1.435, d=−0.03,
  p=3.3e-10 — small effect but real at N=500) on formation RMSE, recovery time
  (6.84 vs 7.20, d=−0.06) and max deviation (3.75 vs 3.82, d=−0.04). This
  mirrors the single-vehicle result and is the strongest formation result of
  any hybrid version so far (v1: 1.906, v2: 1.519, v3: 1.411).
- **Beats Backstepping on 4 of 5 indicators** (formation RMSE, attitude —
  0.085 vs 0.654, a large margin — recovery time, max deviation); loses only
  on control energy.
- **Sweeps PID and Fuzzy on all 5 indicators**, including control energy
  (287,075 vs Fuzzy's 411,249 and PID's 494,156) — the only baselines the
  hybrid beats on energy.
- **Attitude RMSE improved >2x** (v2: 0.203 → v3: 0.085), same pattern as
  single-vehicle, but still behind MPC/SMC (0.047 each).
- **Control energy improved modestly but remains the clear weak point**:
  287,075 (v2: 317,945) — still 2.8x MPC's and worse than every baseline
  except PID/Fuzzy.
- **Still loses to SMC on all 5 indicators**, though closely on formation
  RMSE, recovery time and max deviation (SMC remains the strongest single
  controller in both studies).

**Bottom line (consistent with single-vehicle):** the hybrid now beats MPC on
3/5 indicators, beats Backstepping on 4/5, and sweeps PID/Fuzzy on all 5 --
substantial, measured progress toward the "beat MPC/SMC/PID/Fuzzy" goal. It
does **not** beat SMC outright in either study, and control energy remains an
open weakness that this architecture change improved only modestly. The
honest overall standing: **2nd-of-6 (formation) / 3rd-of-6 (single-vehicle),
decisively ahead of PID/Fuzzy, ahead of or competitive with Backstepping and
MPC, behind SMC.**
