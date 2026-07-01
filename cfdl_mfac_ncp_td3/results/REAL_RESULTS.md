# Real Train-then-Evaluate Results

**Not a demo.** TD3 was trained for real, the checkpoint saved, then the
trained-policy hybrid was evaluated against the ablation variants and the
classical benchmarks. Reproduce:

```bash
python -m cfdl_mfac_ncp_td3.experiments.real_experiment \
    --steps 80000 --trials 40 --trajectory sinusoidal --fault-prob 0.2
```

## Training

- 80,000 environment steps, curriculum difficulty 0.1→1.0, domain
  randomization (mass ±15 %, ocean current), actuator faults @ p=0.2.
- Wall-clock **7.6 min** (176 steps/s, 67 episodes) on CPU.
- Deterministic eval return (difficulty 0.6): noisy, **no clear convergence**;
  best −38 967 at 80 k, but bouncing between −40 k and −180 k throughout.
- Artifacts: `real_run/td3_agent.pt`, `real_run/training_curve.png`.

## Evaluation — 40-trial Monte-Carlo (current + faults), RMSE mean ± std

| Controller | RMSE | Control energy |
|---|---|---|
| **Backstepping** | **0.403 ± 0.168** | 51 948 |
| SMC | 0.530 ± 0.276 | 42 844 |
| MFAC / Hybrid (no RL) | 1.317 ± 0.278 | 212 658 |
| PID | 1.505 ± 1.685 | 48 568 |
| MPC | 1.706 ± 0.215 | 12 751 |
| TD3 only | 9.451 ± 2.166 | 168 027 |
| **Hybrid (TD3)** | **9.996 ± 2.849** | 151 980 |

Single-episode demo under a forced fault (channel 0, 40 % effectiveness):

| Controller | RMSE | mean α (RL authority) |
|---|---|---|
| Hybrid (no RL) | **0.933** | 0.000 |
| Hybrid (TD3) | 11.128 | 0.957 |

## Honest interpretation — this is a negative result, reported as-is

1. **The TD3 policy did not learn competent 6-DOF tracking in 80 k steps.**
   Standalone it scores RMSE ≈ 9.5 vs ≈ 1.3 for pure MFAC. Continuous 6-DOF
   control from a raw normalized-wrench action, under aggressive domain
   randomization + faults + a difficulty curriculum, typically needs
   **500 k–1 M+ steps**; 80 k is far too few, and the return curve shows no
   convergence.

2. **The confidence supervisor then makes things worse, not better.** Because
   pure MFAC has a non-trivial tracking error, the confidence drops and the
   supervisor hands ~96 % authority (α≈0.957) to the RL branch — but that
   branch is a *bad* policy, so `Hybrid (TD3)` (9.99) is worse than both
   `Hybrid (no RL)` (1.32) and the standalone bad policy. The fault demo shows
   this starkly: 0.93 (no RL) vs 11.13 (TD3).

   This exposes a genuine **design flaw**: the supervisor allocates authority
   from MFAC's *distress* alone, with **no measure of whether the RL policy is
   actually trustworthy**. A blind hand-over to an unproven policy is unsafe.

3. **Well-tuned model-based controllers win** (backstepping 0.40, SMC 0.53) —
   they have structural model knowledge the model-free branches lack, and the
   ±15 % randomization is not enough to erode that advantage.

## Recommended fixes (not yet applied)

- **Train much longer** (500 k–1 M steps) and reshape the reward (normalize by
  DOF scale, penalize energy less, add an alive/on-track shaping term) so TD3
  reaches a policy competitive with MFAC before it is trusted.
- **Make the supervisor competence-aware:** gate RL authority not only by MFAC
  distress but by an online estimate of the RL policy's own recent tracking
  performance (e.g. a running advantage of `tau_rl` vs `tau_mfac`), and cap α
  until the policy proves itself. This turns the fusion into a safe,
  performance-gated hand-over.
- Optionally warm-start / regularize TD3 toward the MFAC action (residual RL)
  so the learned policy can only *improve* on the adaptive baseline.
