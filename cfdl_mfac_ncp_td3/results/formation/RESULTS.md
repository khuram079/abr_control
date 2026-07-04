# Formation Control — Results (leader + 3 followers)

Monte-Carlo: **500 simulations** on the `helix` trajectory with random initial positions, ocean current, sensor noise and actuator faults. All controllers evaluated on identical (paired) scenarios; all tuned by the same disclosed procedure.

## Five indicators (mean ± std)

| Controller | formation_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (ours) | 1.519 ± 0.817 | 0.203 ± 0.027 | 7.537 ± 5.984 | 317944.649 ± 17952.529 | 4.001 ± 1.962 |
| SMC | 1.306 ± 0.719 | 0.047 ± 0.016 | 6.255 ± 5.449 | 130193.418 ± 24206.582 | 3.536 ± 1.674 |
| Backstepping | 1.923 ± 1.251 | 0.654 ± 0.231 | 9.046 ± 7.400 | 197721.878 ± 48455.740 | 4.989 ± 2.817 |
| MPC | 1.435 ± 0.765 | 0.047 ± 0.023 | 7.198 ± 5.886 | 102191.049 ± 27583.263 | 3.816 ± 1.801 |
| PID | 13.874 ± 2.890 | 1.644 ± 0.102 | 36.835 ± 1.207 | 494156.014 ± 31872.999 | 29.273 ± 7.352 |
| Fuzzy | 14.932 ± 3.005 | 1.645 ± 0.141 | 37.184 ± 1.061 | 411248.668 ± 24570.304 | 30.968 ± 7.549 |

## Significance vs MPC / PID (paired t-test p, Wilcoxon p, Cohen's d)

### Hybrid (ours) vs MPC
| Indicator | Hybrid | MPC | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| formation_rmse | 1.519 | 1.435 | 1.88e-34 | 1.72e-36 | +0.11 |
| attitude_rmse | 0.203 | 0.047 | 0.00e+00 | 1.26e-83 | +6.19 |
| recovery_time | 7.537 | 7.198 | 7.77e-09 | 2.78e-16 | +0.06 |
| control_energy | 317944.649 | 102191.049 | 0.00e+00 | 1.26e-83 | +9.26 |
| max_deviation | 4.001 | 3.816 | 4.54e-20 | 1.33e-18 | +0.10 |

### Hybrid (ours) vs PID
| Indicator | Hybrid | PID | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| formation_rmse | 1.519 | 13.874 | 0.00e+00 | 1.26e-83 | -5.81 |
| attitude_rmse | 0.203 | 1.644 | 0.00e+00 | 1.26e-83 | -19.23 |
| recovery_time | 7.537 | 36.835 | 0.00e+00 | 1.26e-83 | -6.78 |
| control_energy | 317944.649 | 494156.014 | 0.00e+00 | 1.26e-83 | -6.81 |
| max_deviation | 4.001 | 29.273 | 8.16e-286 | 1.26e-83 | -4.69 |

## Parameter sensitivity (±20%)

- **recovery_threshold**: formation RMSE ['1.546', '1.543', '1.526', '1.538', '1.534'] across factors [0.8, 0.9, 1.0, 1.1, 1.2] → spread **1.3%** over the ±20% range (insensitive to parameter selection).
- **feedforward_cap**: formation RMSE ['1.548', '1.542', '1.526', '1.560', '1.551'] across factors [0.8, 0.9, 1.0, 1.1, 1.2] → spread **2.2%** over the ±20% range (insensitive to parameter selection).

## Fairness / tuning disclosure

Every controller was tuned by identical random search (budget = 40 samples), same validation objective (single-vehicle tracking RMSE over 2 trajectories × 2 seeds, current on). Best parameters:

| Controller | best validation RMSE | best params |
|---|---|---|
| PID | 8.1277 | {'kp': 0.769, 'ki': 2.617, 'kd': 1.537} |
| SMC | 0.5454 | {'lam': 0.622, 'kd': 1.721, 'ks': 2.803} |
| MPC | 0.5959 | {'q_pos': 32.93, 'q_vel': 2.764, 'r_u': 0.001} |
| Fuzzy | 10.1893 | {'ke': 2.072, 'e_width': 0.846, 'edot_width': 0.578} |
| Backstepping | 0.4552 | {'k1': 0.618, 'k2': 2.565} |
| Hybrid | 0.6610 | {'k_outer': 0.88, 'prediction_gain': 1.979, 'feedforward_cap': 0.054} |

## Honest verdict — architecture-fix rerun (v2)

Same three fixes as the single-vehicle rerun (per-DOF physically-scaled
`phi_init`, capped/filtered CFDL feed-forward, proportional recovery ramp);
see `results/single_vehicle/RESULTS.md` for the full rationale.

**Ranking by formation RMSE:** SMC 1.31 < **MPC 1.44 < Hybrid 1.52** <
Backstepping 1.92 < PID 13.9 < Fuzzy 14.9.

- **Formation RMSE improved substantially**: 1.519 (v1: 1.906, ~20% better).
  The hybrid now **beats Backstepping** (1.52 vs 1.92) and is much closer to
  MPC (d=+0.11, a small effect — still statistically significant at N=500,
  p≈1.9e-34, but no longer the clear loss seen in v1, d=+0.57). Still
  significantly worse than SMC/MPC, significantly better than PID/Fuzzy.
- **Recovery time improved**: 7.54 s (v1: 10.10 s) — the proportional-ramp fix
  reduced overshoot-driven recovery time as intended, and the hybrid is now
  close to MPC's 7.20 s.
- **Attitude RMSE**: 0.203 (v1: 0.217) — marginal improvement, still far
  behind MPC/SMC (0.047) but well ahead of Backstepping (0.654).
- **Control energy got WORSE again, confirming the single-vehicle finding**:
  317,945 (v1: 283,883) — the same mechanism as single-vehicle: the tuned
  `feedforward_cap=0.054` is tiny (feed-forward barely engages), so the
  accuracy gain comes from larger per-DOF Phi estimates letting MFAC command
  more torque per unit error, at a real and unresolved energy cost (now 3.1x
  MPC's, worse than v1's 2.8x).

**Bottom line (consistent with single-vehicle):** the fix delivered its
intended tracking/recovery improvement — the hybrid closed most of the gap to
MPC and now beats Backstepping on formation-keeping — but **energy is a
genuine, unresolved weakness that the fix made slightly worse, not better**.
Reported as-is. The supported claim across both studies is *"a model-free
adaptive controller, insensitive to tuning, that is now close to MPC and beats
Backstepping/PID/Fuzzy on tracking and recovery, at a real and larger energy
cost."* It does not beat MPC or SMC outright, and does not beat all models.
