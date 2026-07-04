# Control-Energy Analysis: why the hybrid uses more energy (and what fixes don't work)

Across every Monte-Carlo campaign, the one indicator on which the CFDL-MFAC
hybrid is consistently worst (behind SMC/MPC/Backstepping, ahead only of
PID/Fuzzy on some runs) is **control energy** (integrated `sum tau^2 dt`).
This note documents the root-cause investigation so the limitation is
disclosed rather than hidden.

## 1. The excess is almost entirely the SURGE channel

Per-DOF energy (integrated `tau_i^2`), hybrid vs the tuned SMC benchmark,
25-seed single-vehicle average (randomized mass/current/noise):

| DOF | Hybrid | SMC | ratio |
|---|---|---|---|
| surge | 67 740 | 10 137 | **6.7x** |
| sway | 36 543 | 36 337 | 1.0x |
| heave | 31 768 | 26 082 | 1.2x |
| roll | 1 027 | 750 | 1.4x |
| pitch | 7 044 | 2 792 | 2.5x |
| yaw | 7 319 | 3 132 | 2.3x |
| **total** | **151 441** | **79 230** | **1.9x** |

Sway/heave are essentially tied with SMC; the attitude channels are small in
absolute terms.  **Surge alone accounts for the entire energy gap.**

## 2. The surge command limit-cycles

On a smooth desired surge velocity `nu_d` (total variation 14.6), the *actual*
surge velocity oscillates (TV 55.7, swinging -1.3..-0.28 m/s around a -0.7
setpoint) and the surge thrust ramps +5 N/step to near-saturation (~50 N) then
reverses -- a self-sustained limit cycle.  Surge thrust RMS is **33.7 N** vs
SMC's **10.9 N** for the *same* mean velocity, i.e. ~9x the squared effort.

Mechanism: CFDL-MFAC is an **integrating** control law (`u(k) = u(k-1) +
Delta u`).  On this plant the surge velocity loop is under-damped, so the
integrator over-shoots and limit-cycles about the equilibrium thrust, whereas
a proportional model-based law (SMC computes `tau = J^T(kd s + ks sat(s))`
directly) settles on the small steady thrust without oscillating.

## 3. Every standard MFAC de-tuning knob was swept -- none reduce energy

All swept on the 25-seed single-vehicle randomized set; energy in the same
units as above:

| knob | direction tried | effect on energy | effect on RMSE |
|---|---|---|---|
| `lam` (control penalty) | 2e-4 -> 2e-2 | **worse** (147k -> 196k) | worse (1.49 -> 5.4) |
| `rho` (step size) | 2.0 -> 0.1 | **worse** (147k -> 178k) | worse (1.49 -> 2.7) |
| `u_limit` (increment clip) | 25 -> 0.5 | flat then worse | collapses below 3 |
| `phi_init` (surge, toward true gain) | 5e-3 -> 5e-4 | **worse** (147k -> 172k) | worse (1.49 -> 1.75) |
| output low-pass on tau | tau_lp 0 -> 0.8 | **worse** at moderate cutoff | worse |
| CFDL feed-forward | on -> off | no change (surge RMS 33.7 -> 33.9) | -- |

The current (v3) configuration is **Pareto-optimal** among these: any change
that reduces the oscillation also makes the loop sluggish, which raises both
tracking error *and* energy (the loop then chases accumulated error into
saturation).  Counter-intuitively, a *higher* effective surge gain gives lower
energy -- the loop is under-damped, not over-gained, so reducing gain is the
wrong direction.

## 4. The fix that works: parallel velocity-rate damping

None of the MFAC *hyper-parameters* help, but adding a **velocity-rate
(acceleration-feedback) damping term in parallel with the MFAC command** does:

    tau_i <- tau_i - trans_damping * dnu_i/dt   (translational channels)

CFDL-MFAC remains the adaptive core; this is a fixed inner damping loop.  The
key property is that in *steady* tracking `dnu/dt` is small, so the term is
inert and does not bias the adaptive law, but during the limit cycle `dnu/dt`
is large, so it damps the oscillation.  25-seed single-vehicle sweep:

| `trans_damping` | tracking RMSE | attitude RMSE | energy |
|---|---|---|---|
| 0 (v3) | 1.492 | 0.084 | 147 309 |
| 5 | 1.532 | 0.077 | 129 823 (-12%) |
| 15 | 1.657 | 0.070 | 109 005 (-26%) |
| 30 | 1.811 | 0.073 | 90 969 (-38%) |
| 60 | 2.072 | 0.075 | 72 871 (**-50%, ~= SMC's 76k**) |

Unlike every hyper-parameter in section 3, this exposes a genuine, monotone
energy/tracking Pareto frontier -- and it *improves* attitude RMSE as a
side-effect (a more damped translational loop disturbs the attitude SMC less).

## 5. Making the trade-off fair and automatic

`trans_damping` is added to the hybrid's fair-tuning search space, and the
shared tuning objective is changed from pure RMSE to

    score = rmse + 0.3 * energy / 1e5

applied **identically to every controller** (a purely-RMSE objective is
arguably *less* fair, since it lets a controller win on accuracy while ignoring
an actuator-effort blow-out).  The efficient baselines (SMC/MPC) are almost
unaffected and keep their ordering; the hybrid tuner is now pushed to pick a
damping value that balances the two -- resolving the previously-structural
energy gap through a principled, disclosed mechanism rather than leaving it as
an open limitation.

**Honest standing (v4):** the energy premium is no longer inherent -- it is a
tunable trade-off, and the energy-aware fair tuning selects the balance
automatically.  See `results/single_vehicle/RESULTS.md` and
`results/formation/RESULTS.md` for the v4 500-trial numbers.
