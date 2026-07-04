# PFDL-MFAC vs CFDL-MFAC ablation

**Question asked:** can Partial-Form Dynamic Linearization (PFDL-MFAC) improve
on the Compact-Form (CFDL-MFAC) core -- in particular, can its multi-step
"memory" of recent control increments cure the surge velocity-loop limit cycle
that drives the hybrid's control-energy premium (see `ENERGY_ANALYSIS.md`)?

**What PFDL is.** CFDL uses a single pseudo-gradient on the current increment,
`dy(k+1) = phi(k) du(k)`.  PFDL uses a length-`L` window,
`dy(k+1) = phi(k)^T [du(k), du(k-1), ..., du(k-L+1)]`, and its control law
subtracts the predicted effect of the recent increments before computing the
new one.  `L = 1` recovers CFDL exactly.  Implemented in
`controllers/pfdl.py` (`PFDLMFAC`), selectable via
`HybridController(inner_law="pfdl", pfdl_L=L)`; 6 unit tests in
`tests/test_pfdl.py`.

## Result: PFDL does not improve on CFDL for this plant

Surge thrust RMS (deterministic, no damping) -- lower = less limit cycle:

| core | surge thrust RMS |
|---|---|
| SMC benchmark (reference) | 10.9 |
| CFDL-MFAC | 33.6 |
| PFDL-MFAC L=2 | 33.7 |
| PFDL-MFAC L=3 | 33.7 |
| PFDL-MFAC L=5 | 33.8 |
| PFDL-MFAC L=8 | 33.8 |

30-seed randomized single-vehicle Monte-Carlo (5 indicators), with and without
the velocity-rate damping:

| core | tracking | attitude | recovery | energy | max-dev |
|---|---|---|---|---|---|
| CFDL (damping 12) | 1.751 | 0.076 | 13.18 | 106 403 | 3.689 |
| PFDL L=3 (damping 12) | 1.794 | 0.073 | 13.97 | 107 398 | 3.756 |
| PFDL L=5 (damping 12) | 1.793 | 0.076 | 13.97 | 105 469 | 3.746 |
| CFDL (no damping) | 1.612 | 0.083 | 10.96 | 137 709 | 3.449 |
| PFDL L=3 (no damping) | 1.644 | 0.089 | 11.14 | 140 396 | 3.508 |

**PFDL is statistically indistinguishable from CFDL (marginally worse on
tracking) at every `L`, with or without damping.**

## Why (the instructive part)

The surge limit cycle is *not* caused by CFDL's single-increment memory
limitation -- which is exactly what PFDL is designed to cure -- but by the
interaction of the MFAC **integrator** (`u(k) = u(k-1) + du`) with the plant's
slow, low-damped surge velocity dynamics.  PFDL shares that integrator, so it
inherits the limit cycle.  Moreover the higher-order pseudo-gradient components
`phi_2..phi_L` are barely excited on this loop (the increment window is nearly
collinear during the cycle), so the memory term `S ~= 0` and PFDL collapses to
CFDL behaviour.

## Conclusion

CFDL-MFAC remains the adaptive core: PFDL-MFAC is implemented, tested and
available as a drop-in alternative (`inner_law="pfdl"`), but it does **not**
improve tracking, energy or any other indicator on the REMUS AUV problem.  The
effective energy fix remains the parallel velocity-rate damping term, not the
linearization form.  This is a clean negative ablation -- it localises the
energy premium to the integrator/plant interaction rather than to the
compact-form approximation.
