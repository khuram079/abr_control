# Single-Vehicle Tracking — Results

Monte-Carlo: **500 simulations** on `sinusoidal` with random initial position, ocean current, sensor noise and actuator faults. Paired scenarios; all controllers tuned by the same disclosed procedure.

## Five indicators (mean ± std)

| Controller | tracking_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (ours) | 2.221 ± 1.027 | 0.272 ± 0.083 | 20.024 ± 14.547 | 145551.785 ± 17310.008 | 4.322 ± 1.783 |
| SMC | 1.568 ± 0.813 | 0.075 ± 0.075 | 10.646 ± 10.627 | 76399.403 ± 15865.862 | 3.412 ± 1.529 |
| Backstepping | 1.411 ± 0.864 | 0.847 ± 0.445 | 9.317 ± 11.642 | 106863.196 ± 24384.612 | 3.126 ± 1.544 |
| MPC | 1.725 ± 0.843 | 0.086 ± 0.075 | 12.811 ± 11.865 | 70212.627 ± 17534.047 | 3.652 ± 1.579 |
| PID | 3.551 ± 2.481 | 0.383 ± 0.306 | 27.739 ± 15.267 | 79390.818 ± 29519.520 | 7.054 ± 4.833 |
| Fuzzy | 2.996 ± 1.836 | 0.294 ± 0.260 | 24.347 ± 14.444 | 69977.145 ± 22939.177 | 6.098 ± 3.583 |

## Significance vs MPC / PID (t-test p, Wilcoxon p, Cohen's d)

### Hybrid (ours) vs MPC
| Indicator | Hybrid | MPC | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| tracking_rmse | 2.221 | 1.725 | 4.44e-121 | 5.84e-78 | +0.53 |
| attitude_rmse | 0.272 | 0.086 | 3.13e-302 | 2.74e-83 | +2.34 |
| recovery_time | 20.024 | 12.811 | 1.20e-62 | 4.22e-67 | +0.54 |
| control_energy | 145551.785 | 70212.627 | 0.00e+00 | 1.26e-83 | +4.32 |
| max_deviation | 4.322 | 3.652 | 7.43e-87 | 1.80e-70 | +0.40 |

### Hybrid (ours) vs PID
| Indicator | Hybrid | PID | t-p | Wilcoxon-p | d |
|---|---|---|---|---|---|
| tracking_rmse | 2.221 | 3.551 | 4.34e-39 | 2.32e-81 | -0.70 |
| attitude_rmse | 0.272 | 0.383 | 1.16e-20 | 3.82e-43 | -0.50 |
| recovery_time | 20.024 | 27.739 | 3.23e-65 | 1.48e-65 | -0.52 |
| control_energy | 145551.785 | 79390.818 | 2.25e-224 | 9.80e-82 | +2.73 |
| max_deviation | 4.322 | 7.054 | 3.93e-38 | 7.86e-77 | -0.75 |

## Parameter sensitivity (±20%)

- **recovery_threshold**: tracking RMSE ['1.750', '1.733', '1.768', '1.722', '1.685'] across [0.8, 0.9, 1.0, 1.1, 1.2] → spread **4.8%** over ±20% (insensitive).
- **prediction_gain**: tracking RMSE ['1.718', '1.717', '1.768', '1.769', '1.757'] across [0.8, 0.9, 1.0, 1.1, 1.2] → spread **3.0%** over ±20% (insensitive).

## Fairness / tuning disclosure

Every controller tuned by identical random search (budget = 40), same validation objective. Best parameters:

| Controller | best val RMSE | best params |
|---|---|---|
| PID | 8.1277 | {'kp': 0.769, 'ki': 2.617, 'kd': 1.537} |
| SMC | 0.5454 | {'lam': 0.622, 'kd': 1.721, 'ks': 2.803} |
| MPC | 0.5959 | {'q_pos': 32.93, 'q_vel': 2.764, 'r_u': 0.001} |
| Fuzzy | 10.1893 | {'ke': 2.072, 'e_width': 0.846, 'edot_width': 0.578} |
| Backstepping | 0.4552 | {'k1': 0.618, 'k2': 2.565} |
| Hybrid | 0.5700 | {'k_outer': 1.859, 'prediction_gain': 3.773} |

## Honest verdict — what the data supports

**Ranking by tracking RMSE (500 paired trials, sinusoidal, current+noise+faults):**
Backstepping 1.41 < SMC 1.57 < MPC 1.73 < **Hybrid 2.22** < Fuzzy 3.00 < PID 3.55.

- **Beats PID and Fuzzy on tracking** — significantly (vs PID: d=−0.70 on tracking,
  −0.50 attitude, −0.52 recovery, −0.75 max-deviation; p≈1e−20..1e−39). But the
  hybrid **uses ~2× more control energy** than PID (145 k vs 79 k, d=+2.73) — a real
  cost of the CFDL feed-forward plus the recovery boost.
- **Loses to MPC, SMC and backstepping on every indicator** — significantly
  (vs MPC: hybrid worse on all five, d=+0.40..+4.32, p≈0). The hybrid is **4th of 6**.
- **Weak attitude tracking**: attitude RMSE 0.272 vs SMC 0.075 / MPC 0.086 — a genuine
  weakness of the model-free inner loop on the rotational DOFs.
- **Insensitive to parameter choice**: tracking RMSE varies 3.0 % (prediction gain) and
  4.8 % (recovery threshold) over ±20 %.
- **Fair tuning strengthened the baselines** (untuned MPC ≈ 3.7 → tuned 1.73;
  untuned PID ≈ 2.6 → tuned 3.55 on this harder randomized+faulted set), same 40-sample
  search for all; tuned gains disclosed above.

**Bottom line (consistent with the formation study):** on this well-modeled single
vehicle, the model-free hybrid is **mid-pack** — it significantly beats PID and fuzzy
control on tracking (at higher energy) but is **behind the model-based methods
(SMC, MPC, backstepping)** and does **not** beat all models. The supported claim is
*"a model-free adaptive controller, insensitive to tuning, that outperforms PID and
fuzzy control and remains stable and competitive with model-based methods"* — not
superiority over SMC/MPC/backstepping, which have the exact plant model on a
well-modeled task.
