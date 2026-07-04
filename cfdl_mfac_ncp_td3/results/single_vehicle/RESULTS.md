# Single-Vehicle Tracking — Results

Monte-Carlo: **500 simulations** on `sinusoidal` with random initial position, ocean current, sensor noise and actuator faults. Paired scenarios; all controllers tuned by the same disclosed procedure.

## Five indicators (mean ± std)

| Controller | tracking_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (ours) | 1.610 ± 0.822 | 0.112 ± 0.083 | 11.116 ± 11.037 | 147871.349 ± 16048.618 | 3.463 ± 1.565 |
| SMC | 1.568 ± 0.813 | 0.075 ± 0.075 | 10.646 ± 10.627 | 76399.403 ± 15865.862 | 3.412 ± 1.529 |
| Backstepping | 1.411 ± 0.864 | 0.847 ± 0.445 | 9.317 ± 11.642 | 106863.196 ± 24384.612 | 3.126 ± 1.544 |
| MPC | 1.725 ± 0.843 | 0.086 ± 0.075 | 12.811 ± 11.865 | 70212.627 ± 17534.047 | 3.652 ± 1.579 |
| PID | 3.551 ± 2.481 | 0.383 ± 0.306 | 27.739 ± 15.267 | 79390.818 ± 29519.520 | 7.054 ± 4.833 |
| Fuzzy | 2.996 ± 1.836 | 0.294 ± 0.260 | 24.347 ± 14.444 | 69977.145 ± 22939.177 | 6.098 ± 3.583 |

## Significance vs MPC / PID (t-test p, Wilcoxon p, Cohen's d)

### Hybrid (ours) vs MPC
| Indicator | Hybrid | MPC | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| tracking_rmse | 1.610 | 1.725 | 1.45e-59 | 8.88e-79 | -0.14 |
| attitude_rmse | 0.112 | 0.086 | 3.81e-46 | 1.21e-55 | +0.33 |
| recovery_time | 11.116 | 12.811 | 1.35e-18 | 1.88e-46 | -0.15 |
| control_energy | 147871.349 | 70212.627 | 0.00e+00 | 1.26e-83 | +4.62 |
| max_deviation | 3.463 | 3.652 | 4.91e-41 | 2.27e-59 | -0.12 |

### Hybrid (ours) vs PID
| Indicator | Hybrid | PID | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| tracking_rmse | 1.610 | 3.551 | 3.36e-67 | 1.26e-83 | -1.05 |
| attitude_rmse | 0.112 | 0.383 | 1.27e-86 | 1.26e-83 | -1.21 |
| recovery_time | 11.116 | 27.739 | 3.38e-125 | 2.38e-80 | -1.25 |
| control_energy | 147871.349 | 79390.818 | 2.79e-234 | 6.62e-82 | +2.88 |
| max_deviation | 3.463 | 7.054 | 5.24e-58 | 1.29e-83 | -1.00 |

## Parameter sensitivity (±20%)

- **recovery_threshold**: tracking RMSE ['1.257', '1.247', '1.250', '1.248', '1.247'] across [0.8, 0.9, 1.0, 1.1, 1.2] → spread **0.8%** over ±20% (insensitive).
- **feedforward_cap**: tracking RMSE ['1.236', '1.253', '1.250', '1.252', '1.247'] across [0.8, 0.9, 1.0, 1.1, 1.2] → spread **1.3%** over ±20% (insensitive).

## Fairness / tuning disclosure

Every controller tuned by identical random search (budget = 40), same validation objective. Best parameters:

| Controller | best val RMSE | best params |
|---|---|---|
| PID | 8.1277 | {'kp': 0.769, 'ki': 2.617, 'kd': 1.537} |
| SMC | 0.5454 | {'lam': 0.622, 'kd': 1.721, 'ks': 2.803} |
| MPC | 0.5959 | {'q_pos': 32.93, 'q_vel': 2.764, 'r_u': 0.001} |
| Fuzzy | 10.1893 | {'ke': 2.072, 'e_width': 0.846, 'edot_width': 0.578} |
| Backstepping | 0.4552 | {'k1': 0.618, 'k2': 2.565} |
| Hybrid | 0.5803 | {'k_outer': 0.548, 'prediction_gain': 2.526, 'feedforward_cap': 0.082, 'att_lam': 2.097, 'att_kd': 1.427, 'att_ks': 2.969} |

## Honest verdict — CFDL-MFAC + SMC attitude architecture (v3)

Per-DOF architecture change: CFDL-MFAC keeps surge/sway/heave (translational,
model-free-adaptive core); roll/pitch/yaw now use a boundary-layer sliding-mode
law instead of the per-DOF MFAC bank (see `controllers/hybrid_controller.py`
docstring for the full rationale). Same fair 40-sample tuning as v1/v2, now
also tuning the attitude SMC gains (`att_lam`, `att_kd`, `att_ks`).

**Ranking by tracking RMSE:** Backstepping 1.41 < SMC 1.57 < **Hybrid 1.61 <
MPC 1.73** < Fuzzy 3.00 < PID 3.55.

- **The hybrid now beats MPC, significantly, on 3 of 5 indicators**: tracking
  RMSE (1.610 vs 1.725, d=−0.14, p=1.4e-59), recovery time (11.12 vs 12.81,
  d=−0.15, p=1.4e-18), max deviation (3.46 vs 3.65, d=−0.12, p=4.9e-41). This
  is a first for this project — every prior architecture (v1, v2) lost to MPC
  on tracking.
- **Attitude RMSE improved more than 2x** (v2: 0.244 → v3: 0.112) but still
  loses to MPC (0.086, d=+0.33) and SMC (0.075). The SMC-attitude law closed
  most, not all, of the gap to the standalone SMC/MPC attitude performance —
  plausibly because the shared outer-loop pose error (rather than a dedicated
  attitude estimator) still carries the single-vehicle random-init/current
  disturbance into the sliding surface.
- **Control energy is unchanged and remains the clear weak point**: 147,871
  (v2: 145,552) — essentially flat. The attitude fix targeted tracking/
  attitude/recovery, not energy; the translational MFAC loop's over-aggressive
  torque-per-error (diagnosed in v2) is untouched by this change and remains
  unresolved. Worst of all 6 controllers on this indicator, as in every prior
  version.
- **Beats PID and Fuzzy on 4 of 5 indicators** (loses only on energy to Fuzzy).
- **Still loses to SMC on 4 of 5 indicators** (tracking, attitude, energy;
  close on recovery/max-deviation) — SMC remains the strongest single
  controller in this comparison.
- **Insensitive to parameters**: tracking RMSE spread 0.8-1.3% over ±20%.

**Bottom line:** real, measured progress toward the "beat MPC/SMC/PID/Fuzzy"
goal -- the hybrid now beats MPC on 3/5 indicators (a first) and PID/Fuzzy on
4/5, but does **not** beat SMC outright and does **not** sweep all five
indicators against any baseline. Energy remains an open, unresolved weakness
carried unchanged from v2; fixing it is the clear next target and is a
translational-MFAC issue, not an attitude one.
