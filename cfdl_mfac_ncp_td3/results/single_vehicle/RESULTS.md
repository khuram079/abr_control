# Single-Vehicle Tracking — Results

Monte-Carlo: **500 simulations** on `sinusoidal` with random initial position, ocean current, sensor noise and actuator faults. Paired scenarios; all controllers tuned by the same disclosed procedure.

## Five indicators (mean ± std)

| Controller | tracking_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (ours) | 1.740 ± 0.849 | 0.244 ± 0.082 | 13.071 ± 12.652 | 164694.797 ± 16220.373 | 3.634 ± 1.551 |
| SMC | 1.568 ± 0.813 | 0.075 ± 0.075 | 10.646 ± 10.627 | 76399.403 ± 15865.862 | 3.412 ± 1.529 |
| Backstepping | 1.411 ± 0.864 | 0.847 ± 0.445 | 9.317 ± 11.642 | 106863.196 ± 24384.612 | 3.126 ± 1.544 |
| MPC | 1.725 ± 0.843 | 0.086 ± 0.075 | 12.811 ± 11.865 | 70212.627 ± 17534.047 | 3.652 ± 1.579 |
| PID | 3.551 ± 2.481 | 0.383 ± 0.306 | 27.739 ± 15.267 | 79390.818 ± 29519.520 | 7.054 ± 4.833 |
| Fuzzy | 2.996 ± 1.836 | 0.294 ± 0.260 | 24.347 ± 14.444 | 69977.145 ± 22939.177 | 6.098 ± 3.583 |

## Significance vs MPC / PID (t-test p, Wilcoxon p, Cohen's d)

### Hybrid (ours) vs MPC
| Indicator | Hybrid | MPC | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| tracking_rmse | 1.740 | 1.725 | 6.03e-02 | 9.60e-06 | +0.02 |
| attitude_rmse | 0.244 | 0.086 | 0.00e+00 | 4.54e-83 | +2.01 |
| recovery_time | 13.071 | 12.811 | 1.82e-01 | 8.30e-01 | +0.02 |
| control_energy | 164694.797 | 70212.627 | 0.00e+00 | 1.26e-83 | +5.59 |
| max_deviation | 3.634 | 3.652 | 2.59e-01 | 5.76e-01 | -0.01 |

### Hybrid (ours) vs PID
| Indicator | Hybrid | PID | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| tracking_rmse | 1.740 | 3.551 | 9.36e-62 | 1.26e-83 | -0.98 |
| attitude_rmse | 0.244 | 0.383 | 3.90e-31 | 9.75e-79 | -0.62 |
| recovery_time | 13.071 | 27.739 | 4.35e-115 | 2.56e-80 | -1.05 |
| control_energy | 164694.797 | 79390.818 | 1.00e-270 | 3.84e-83 | +3.58 |
| max_deviation | 3.634 | 7.054 | 1.35e-54 | 1.32e-83 | -0.95 |

## Parameter sensitivity (±20%)

- **recovery_threshold**: tracking RMSE ['1.382', '1.365', '1.360', '1.366', '1.371'] across [0.8, 0.9, 1.0, 1.1, 1.2] → spread **1.6%** over ±20% (insensitive).
- **feedforward_cap**: tracking RMSE ['1.358', '1.358', '1.360', '1.357', '1.351'] across [0.8, 0.9, 1.0, 1.1, 1.2] → spread **0.6%** over ±20% (insensitive).

## Fairness / tuning disclosure

Every controller tuned by identical random search (budget = 40), same validation objective. Best parameters:

| Controller | best val RMSE | best params |
|---|---|---|
| PID | 8.1277 | {'kp': 0.769, 'ki': 2.617, 'kd': 1.537} |
| SMC | 0.5454 | {'lam': 0.622, 'kd': 1.721, 'ks': 2.803} |
| MPC | 0.5959 | {'q_pos': 32.93, 'q_vel': 2.764, 'r_u': 0.001} |
| Fuzzy | 10.1893 | {'ke': 2.072, 'e_width': 0.846, 'edot_width': 0.578} |
| Backstepping | 0.4552 | {'k1': 0.618, 'k2': 2.565} |
| Hybrid | 0.6610 | {'k_outer': 0.88, 'prediction_gain': 1.979, 'feedforward_cap': 0.054} |

## Honest verdict — architecture-fix rerun (v2)

This rerun follows three targeted fixes to the hybrid controller, diagnosed from
the v1 500-trial results (previously 4th-of-6, worst energy of all controllers):
(1) per-DOF `phi_init` physically scaled by the mass-matrix diagonal `dt/M_ii`
(v1 shared one value across all 6 DOFs, under-priming roll by ~40x), (2) the
CFDL feed-forward term is now low-pass filtered and capped (`feedforward_cap`,
a tuned parameter) instead of an uncapped raw-derivative spike, (3) the
fault-recovery gain boost now ramps proportionally with how far error exceeds
the threshold instead of a blunt on/off 2x step.

**Ranking by tracking RMSE:** Backstepping 1.41 < SMC 1.57 < **MPC 1.73 ≈
Hybrid 1.74** < Fuzzy 3.00 < PID 3.55.

- **Tracking: hybrid now statistically ties MPC** (Cohen's d = +0.02, a
  negligible effect — Wilcoxon detects a tiny but consistent difference only
  because N=500 gives it the power to; the paired t-test does not, p=0.06).
  This is a real change from v1, where the hybrid was clearly worse than MPC
  (d=+0.53). Hybrid still decisively beats PID (d=−0.98) and Fuzzy.
- **Attitude RMSE improved but remains the weak point**: 0.244 (v1: 0.272),
  still behind MPC (0.086) / SMC (0.075), though now *ahead* of Backstepping's
  0.847 — the per-DOF phi fix helped roll/pitch/yaw, but not enough to close
  the gap with model-based attitude control.
- **Control energy got WORSE, not better**: 164,695 (v1: 145,552) — the fair
  40-sample tuner (optimizing tracking RMSE only, no energy term, identical
  objective as v1) converged on `feedforward_cap=0.054` (feed-forward
  contributes almost nothing) with `k_outer=0.88`; the accuracy gain instead
  came from the larger per-DOF Phi estimates letting the adaptive law command
  more torque per unit error, without a corresponding re-derivation of `rho`
  or `u_limit` for the new Phi scale. This is reported as-is, not minimised:
  **energy remains the hybrid's principal weakness**, now 2.3x MPC's (was 2.1x
  in v1) despite the tracking improvement.
- **Insensitive to parameters**: tracking RMSE varies 1.6% (recovery
  threshold) / 0.6% (feedforward cap) over ±20% — even more insensitive than
  v1, since the tuned feedforward_cap is now so small its variation barely
  registers.

**Bottom line:** the architecture fix delivered its intended result on
*tracking* — the hybrid moved from clearly-behind-MPC to statistically tied —
but did **not** fix the energy weakness, which actually worsened as a
side-effect of the same fix. The supported claim is now *"a model-free
adaptive controller, insensitive to tuning, that matches MPC and beats
PID/fuzzy on tracking accuracy, at a real and unresolved energy cost, and with
a weaker attitude channel than model-based methods."* It is not a win on every
indicator, and does not beat MPC outright.
