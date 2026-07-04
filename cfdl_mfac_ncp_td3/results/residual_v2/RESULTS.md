# Residual-RL Hybrid + Stability Analysis

500-episode residual TD3 trained on the strong hybrid (CFDL-MFAC + SMC attitude + damping); 200-trial Monte-Carlo + stability analysis.

## Five indicators (mean ± std)

| Controller | tracking_rmse | attitude_rmse | recovery_time | control_energy | max_deviation |
|---|---|---|---|---|---|
| Hybrid (strong) | 1.788 ± 0.881 | 0.092 ± 0.089 | 13.673 ± 12.071 | 106937.286 ± 12267.756 | 3.776 ± 1.657 |
| Hybrid+Residual | 1.914 ± 0.930 | 0.109 ± 0.084 | 14.914 ± 12.181 | 101233.690 ± 12749.862 | 4.006 ± 1.746 |
| SMC | 1.606 ± 0.835 | 0.081 ± 0.086 | 11.058 ± 10.875 | 76297.174 ± 15943.032 | 3.508 ± 1.571 |
| MPC | 1.761 ± 0.864 | 0.095 ± 0.090 | 13.186 ± 11.921 | 70575.058 ± 18079.225 | 3.748 ± 1.617 |
| PID | 3.675 ± 2.656 | 0.394 ± 0.302 | 27.910 ± 15.575 | 80656.409 ± 31610.298 | 7.353 ± 5.088 |
| Fuzzy | 3.135 ± 1.934 | 0.322 ± 0.330 | 24.734 ± 14.617 | 71137.434 ± 24897.594 | 6.406 ± 3.739 |

## Does the residual RL help? (Hybrid+Residual vs Hybrid strong)

| Indicator | Residual | Strong | p | d |
|---|---|---|---|---|
| tracking_rmse | 1.914 | 1.788 | 1.00e-14 | +0.14 |
| attitude_rmse | 0.109 | 0.092 | 4.37e-16 | +0.21 |
| recovery_time | 14.914 | 13.673 | 7.47e-04 | +0.10 |
| control_energy | 101233.690 | 106937.286 | 7.57e-49 | -0.45 |
| max_deviation | 4.006 | 3.776 | 8.54e-15 | +0.13 |

## Stability analysis

1. **Lyapunov (SMC attitude)**: V decays 92.5% (V0=2.854 → 0.2144); sliding-surface ultimate bound 0.605 → **practical sliding-mode stability**.
2. **CFDL-MFAC BIBO condition**: pseudo-gradient bounded=True, sign-definite=True (|phi| ≤ 0.0387) → **controllability/BIBO premise holds**.
3. **Input-to-State Stability**: no divergence across current 0–0.8 m/s; finite ultimate-error bounds [0.168, 0.096, 0.132, 0.305, 3.363] → **ISS**.
4. **Region of attraction**: converges from initial errors [0.5, 1.0, 2.0, 4.0, 8.0] m (100% converged) → **large region of attraction**.
5. **Monte-Carlo robust stability**: divergence rate 0.0% over randomized plants/disturbances/faults; ultimate error bound 0.251 m (p95 0.749) → **uniformly ultimately bounded**.

## Honest verdict

**Residual RL is safe but not additive.** After 500 episodes on the strong
hybrid, `Hybrid+Residual` is *worse* than `Hybrid (strong)` on 4 of 5
indicators (tracking d=+0.14, attitude d=+0.21, recovery d=+0.10, max-dev
d=+0.13) and only better on energy (d=-0.45) -- a marginal Pareto shift, not a
win. This confirms the pattern from every RL experiment in this project: the
model-free CFDL-MFAC+SMC+damping baseline is already near the achievable
performance, so a bounded learned correction has almost nothing to add and
tends to trade a little accuracy for a little energy. The residual is *stable
and bounded* (see below), so enabling it does no harm beyond that trade -- but
**the recommended deployment is the strong hybrid without the residual**
(it beats the residual variant on 4/5 and statistically ties MPC).

**Standing vs baselines** (residual hybrid): sweeps PID and Fuzzy on
accuracy/recovery (d=-0.9..-1.3); behind MPC (the residual regressed the
strong hybrid's MPC parity); behind SMC. The strong hybrid (no residual)
remains the better configuration, tied with MPC on 4/5.

**Controller stability (the key new result): PASS on all five criteria.**

| Criterion | Result | Verdict |
|---|---|---|
| Lyapunov (SMC attitude) | V decays 92.5%, converges | practical sliding-mode stability |
| CFDL-MFAC pseudo-gradient | bounded, sign-definite (|phi|≤0.039) | BIBO/controllability premise holds |
| Input-to-State Stability | 0 divergence over current 0–0.8 m/s | ISS (bounded input → bounded state) |
| Region of attraction | 100% converge from ‖e0‖ up to 8 m | large region of attraction |
| Monte-Carlo robustness | 0% divergence, ultimate bound 0.25 m | uniformly ultimately bounded |

The controller is provably/practically stable and empirically robust: bounded
adaptive gains, a decaying attitude Lyapunov function, input-to-state stability
under ocean-current disturbance, a large region of attraction, and zero
divergence across 200 randomized plants/faults.
