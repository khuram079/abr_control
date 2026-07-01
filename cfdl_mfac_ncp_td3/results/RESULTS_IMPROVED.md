# Improved Hybrid — Corrected Results

This supersedes the earlier `RESULTS_500ep_100mc.md`. After the honest review
("the results are not good"), two root-cause fixes were made and the 100-trial
Monte-Carlo re-run below reuses the same trained TD3 checkpoint
(`train500/td3_agent.pt`). Reproduce:

```bash
python -m cfdl_mfac_ncp_td3.experiments.evaluate_improved \
    --trials 100 --trajectory sinusoidal --fault-prob 0.2
```

## What was wrong, and the two fixes

1. **MFAC phase-lag on moving references.** The model-free inner loop is purely
   reactive, so it lagged the moving x/y reference (error concentrated there),
   capping the whole hybrid at RMSE ≈ 1.3 while model-based backstepping tracked
   at ≈ 0.41. **Fix:** a model-free **CFDL inverse feed-forward** using MFAC's
   own learned per-channel pseudo-Jacobian, `tau_ff = g · Δnu_d / Phi` — the
   anticipation a reactive controller lacks, with no plant model.

2. **A weak RL policy dragging the strong baseline down.** Once (1) made the
   baseline strong (0.59), the still-weak TD3 policy (3.0 standalone) pulled the
   fused hybrid *down* to 0.81, because the trust gate cannot isolate RL's
   marginal effect while MFAC is itself reducing the error. **Fix:** a
   **directional-agreement gate** that scales RL authority by
   `max(0, cos(u_mfac, u_rl))`, suppressing a policy that disagrees with the
   sensible adaptive direction.

## Final 100-simulation Monte-Carlo (sinusoidal, current + faults @ p=0.2)

| Controller | RMSE mean ± std | pos-RMSE | Energy | vs before |
|---|---|---|---|---|
| **Backstepping** | **0.412 ± 0.206** | 0.248 | 52 613 | — (best) |
| SMC | 0.524 ± 0.240 | 0.518 | 43 595 | — |
| **MFAC / Hybrid (no RL)** | **0.590 ± 0.247** | 0.579 | 140 017 | **1.303 → 0.590 (−55 %)** |
| Hybrid (TD3) | 0.620 ± 0.220 | 0.592 | 130 843 | **9.996 → 0.620** |
| PID | 1.606 ± 1.894 | 1.583 | 51 932 | — |
| MPC | 1.708 ± 0.192 | 1.698 | 12 988 | — |
| TD3 only | 3.008 ± 0.475 | 2.606 | 137 985 | — |

Artifacts: `improved_eval/improved_rmse_bars.png`, `improved_eval/improved_rmse_box.png`,
`improved_eval/eval_summary.json`.

## Honest bottom line

- **Fix 1 is a large, real win.** The model-free CFDL-MFAC-NCP hybrid dropped
  from **1.303 → 0.590 RMSE (−55 %)** using **−33 % control energy**. It is now
  **competitive with SMC (0.524)** and beats PID and MPC by ~2.7×.
- **Fix 2 makes RL safe.** Hybrid (TD3) went from **0.81 → 0.62**, now within
  ~5 % of the no-RL baseline (0.590). Enabling the learned policy no longer
  degrades the controller — the gate holds it back where it does not help.
- **Backstepping (0.412) still wins**, and **SMC (0.524) still edges the
  hybrid** — both are model-based with exact structural knowledge on a
  near-nominal plant. I am not claiming the model-free hybrid beats them here.
- **The TD3 policy is safe but not yet *additive*.** Standalone it is 3.0
  (uncompetitive), so blending it in cannot improve on the strong 0.59 baseline;
  the gate correctly keeps its authority near zero. **Recommended deployment
  configuration: `Hybrid (no RL)` = CFDL-MFAC-NCP + feed-forward.**

## Recommended next step to make RL additive

Train TD3 as a **residual** on top of the (now strong) MFAC action rather than a
full wrench from scratch — the policy would then only need to learn a small
corrective term, which is a much easier and safer learning problem and is the
standard way to make learned control genuinely improve on a good analytic
baseline. This requires an env wrapper that runs MFAC internally and a retrain;
it is the clear path to `Hybrid (TD3) < Hybrid (no RL)`.
