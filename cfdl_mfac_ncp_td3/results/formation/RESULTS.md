# Formation Control — Results (leader + 3 followers)

Monte-Carlo: **500 simulations** on the `helix` trajectory with random initial positions, ocean current, sensor noise and actuator faults. All controllers evaluated on identical (paired) scenarios; all tuned by the same disclosed procedure.

## Five indicators (mean ± std)

| Controller | formation_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (ours) | 1.906 ± 0.896 | 0.217 ± 0.035 | 10.099 ± 6.843 | 283882.605 ± 19818.320 | 4.759 ± 2.199 |
| SMC | 1.306 ± 0.719 | 0.047 ± 0.016 | 6.255 ± 5.449 | 130193.418 ± 24206.582 | 3.536 ± 1.674 |
| Backstepping | 1.923 ± 1.251 | 0.654 ± 0.231 | 9.046 ± 7.400 | 197721.878 ± 48455.740 | 4.989 ± 2.817 |
| MPC | 1.435 ± 0.765 | 0.047 ± 0.023 | 7.198 ± 5.886 | 102191.049 ± 27583.263 | 3.816 ± 1.801 |
| PID | 13.874 ± 2.890 | 1.644 ± 0.102 | 36.835 ± 1.207 | 494156.014 ± 31872.999 | 29.273 ± 7.352 |
| Fuzzy | 14.932 ± 3.005 | 1.645 ± 0.141 | 37.184 ± 1.061 | 411248.668 ± 24570.304 | 30.968 ± 7.549 |

## Significance vs MPC / PID (paired t-test p, Wilcoxon p, Cohen's d)

### Hybrid (ours) vs MPC
| Indicator | Hybrid | MPC | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| formation_rmse | 1.906 | 1.435 | 7.98e-188 | 8.36e-83 | +0.57 |
| attitude_rmse | 0.217 | 0.047 | 0.00e+00 | 1.26e-83 | +5.70 |
| recovery_time | 10.099 | 7.198 | 9.76e-85 | 1.83e-75 | +0.45 |
| control_energy | 283882.605 | 102191.049 | 0.00e+00 | 1.26e-83 | +7.56 |
| max_deviation | 4.759 | 3.816 | 1.71e-118 | 4.94e-77 | +0.47 |

### Hybrid (ours) vs PID
| Indicator | Hybrid | PID | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| formation_rmse | 1.906 | 13.874 | 0.00e+00 | 1.26e-83 | -5.59 |
| attitude_rmse | 0.217 | 1.644 | 0.00e+00 | 1.26e-83 | -18.63 |
| recovery_time | 10.099 | 36.835 | 4.36e-308 | 1.26e-83 | -5.44 |
| control_energy | 283882.605 | 494156.014 | 0.00e+00 | 1.26e-83 | -7.92 |
| max_deviation | 4.759 | 29.273 | 7.10e-280 | 1.26e-83 | -4.51 |

## Parameter sensitivity (±20%)

- **recovery_threshold**: formation RMSE ['1.975', '1.912', '1.923', '1.875', '1.856'] across factors [0.8, 0.9, 1.0, 1.1, 1.2] → spread **6.2%** over the ±20% range (insensitive to parameter selection).
- **prediction_gain**: formation RMSE ['1.873', '1.894', '1.923', '1.917', '1.940'] across factors [0.8, 0.9, 1.0, 1.1, 1.2] → spread **3.5%** over the ±20% range (insensitive to parameter selection).

## Fairness / tuning disclosure

Every controller was tuned by identical random search (budget = 40 samples), same validation objective (single-vehicle tracking RMSE over 2 trajectories × 2 seeds, current on). Best parameters:

| Controller | best validation RMSE | best params |
|---|---|---|
| PID | 8.1277 | {'kp': 0.769, 'ki': 2.617, 'kd': 1.537} |
| SMC | 0.5454 | {'lam': 0.622, 'kd': 1.721, 'ks': 2.803} |
| MPC | 0.5959 | {'q_pos': 32.93, 'q_vel': 2.764, 'r_u': 0.001} |
| Fuzzy | 10.1893 | {'ke': 2.072, 'e_width': 0.846, 'edot_width': 0.578} |
| Backstepping | 0.4552 | {'k1': 0.618, 'k2': 2.565} |
| Hybrid | 0.5700 | {'k_outer': 1.859, 'prediction_gain': 3.773} |

## Honest verdict — what the data supports

**Ranking by formation RMSE (500 paired trials):**
SMC 1.31 < MPC 1.44 < **Hybrid 1.91 ≈ Backstepping 1.92** ≪ PID 13.9 < Fuzzy 14.9.

- **Beats PID and Fuzzy — significantly, on all five indicators** (Cohen's d = −4.5 to −18.6, p ≈ 0). Mechanism: PID/fuzzy have no feed-forward and lag the sustained curvature of the helix (tuned PID tracks a setpoint at 0.75 and a sinusoid at 1.43 RMSE, but the helix at 8.8 — a genuine structural limitation, not a weakened baseline). The hybrid's CFDL inverse feed-forward removes that lag.
- **Loses to MPC and SMC — significantly, on all five indicators** (the hybrid is worse; d = +0.45 to +7.56, p ≈ 0), and ties backstepping on formation RMSE.
- **Insensitive to parameter choice**: formation RMSE varies only 3.5 % (prediction gain) and 6.2 % (recovery threshold) across ±20 %.
- **Fair comparison**: every controller tuned by the same 40-sample random search, same objective; tuned gains disclosed above. Fair tuning *strengthened* the baselines (untuned MPC ≈ 3.9 → tuned 1.44), which makes the hybrid look worse, not better — the opposite of "weakening baselines".

**Bottom line:** on this *well-modeled* AUV formation task, fairly-tuned model-based
control (SMC, MPC) is near-optimal and the model-free hybrid does **not** beat all
methods — it is mid-pack: significantly better than PID/fuzzy, competitive with
backstepping, and behind SMC/MPC. The claim the data supports is *"a model-free
adaptive formation controller, insensitive to tuning, that significantly outperforms
PID and fuzzy control and is competitive with model-based methods"* — **not** "beats
all models". The regime where a model-free method can legitimately win is when an
accurate plant model is unavailable (severe unmodelled dynamics); that is not this
experiment, where the baselines are given the exact model structure.
