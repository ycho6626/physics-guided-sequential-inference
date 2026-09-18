# Operational Architecture Contract

This benchmark is separate from the original Modules 03–06 demonstrator. It tests one specified
WDA/OT/filter/policy construction, not every possible implementation of those methods. Its
[public result record](../../docs/OPERATIONAL_BENCHMARK.md) describes the completed studies.

## Inputs and fitting

The [primary configuration](../configs/operational_architecture_v1.json) fixes 8,000 ten-frame
sequences; the [episode configuration](../configs/operational_architecture_episode_v1.json) fixes
4,000. Each has its own seeds, sequence partitions, complete training and calibration. The
episode simulator optionally zeros hazard weights outside known onset/duration windows; it does
not change the legacy generator when disabled. GB/VX are names for synthetic template shapes,
not validated spectra of real agents.

Outer sequence splitting precedes learned fitting. E1 consumes only sample/sequence IDs,
timestamps, wavelengths and observed spectra. It uses a fixed simulator-derived target/nuisance
design, train-estimated diagonal residual covariance, and a joint nonnegative GB/VX cone-GLR.
Its full configuration binding, canonical design and training-overlap guards remain enforced.
Two-fold sequence cross-fitting supplies training E1 features; held-out features use the full
training fit. Numerical scoring exclusions retain reasons and are never counted as negatives.

The nine observed channels are the eight Module-02 indicators and the E1 frame score. Simulator
latent fields supply training supervision and evaluation truth, never held-out detector inputs.
Whitening fits on eligible training frames only, using ddof=1 and an eigenvalue floor of 1e-8.

## Composed methods

1. **WDA:** orthonormal 9-to-4 projection; maximize between-class transport cost divided by the
   sum of within-class costs. Cost means `sum(plan * squared_distance)` under the entropic
   coupling, not a debiased Sinkhorn divergence. Each step uses one projected cross-cloud median
   scale for all three costs, differentiates through that scale and 30 scaling updates, and
   detaches warm starts between steps. Adam/QR settings and the 200-step budget are fixed in code
   and config. Inner-development chooses the first maximum; final fitting uses all outer-train.
2. **OT evidence:** two-sided smoothed c-transforms of fitted class-to-class dual potentials,
   centered by a serialized training mean. The mean absorbs the additive gauge shift. Reference
   identities are SHA256-ordered; epsilon is the selected multiplier times median cross-cost.
   The coordinate is applied row-locally, in blocks. It is not itself a posterior probability.
3. **Emission/filter:** supervised class means and pooled variance convert the coordinate to
   frame log-likelihood evidence. Realizable clipping fraction inflates variance by `1 + rho*c`.
   A two-state filter starts at equal odds and uses fixed symmetric switching probability 0.05.
   No EM-fitted persistence state or legacy grade-B gate is used.
4. **Policy:** per-sequence maxima of running log-odds are calibrated on benign calibration
   sequences. Upper ranks are `floor((m+1)/100)` for CONFIRM and `floor((m+1)/20)` for RESCAN,
   with strict crossing and conservative ties. Insufficient calibration produces null endpoints,
   not a top-up. Displayed CONFIRM persists after first crossing; timing uses the first crossing.
   Reporting quartiles never affect evidence or actions.

Selection uses an inner sequence split within outer-train. Every selection-time fit stays on
inner-fit; development chooses WDA step, kappa and rho. Each candidate must finish numerically;
failed candidates do not silently disappear from selection. The original primary result retained
its earlier scaling-solver selection and later refit OT with Newton-CG; see the provenance caveat.

## Controls

| Arm | Change relative to CAND |
|---|---|
| CAND | Complete nine-channel WDA / OT / switching-filter chain |
| −OT | Retain WDA and the same projected references; independent-coupling coordinate, refit emissions |
| −REP | Omit WDA, fit OT on all nine whitened channels with its own declared selection/refit |
| −TEMP | Share CAND fitted evidence, set switching probability to zero; recalibrate maxima |
| S1 | Nine-channel independent-coupling score without WDA; separate emissions/selection |
| S0 | Sum E1 frame scores over ten frames; sequence-only calibration |
| D′ | Original 2-D embedding plus class-conditional HMM readouts; primary only |

Primary uses all seven; episode uses CAND, −OT, −REP, −TEMP and S0. Removing the final OT
coordinate does not remove WDA's OT objective. These are the specified interventions, not
universal causal decompositions of mathematical tool families. D′ uses `compute_diagnostic: false`
in Module 03 to avoid an unused quadratic summary; regime supervision and frozen apply are unchanged.

## Numerical contract

Operational OT solves the entropic problem relative to uniform product marginals, using damped
semi-dual Newton-CG. Warm scaling halves epsilon down to the requested value, with 25 updates per
level. The gauge-free semi-dual Hessian action is `q*v - P.T@(P@v)/n`. Mean-zero CG directions,
damping 1e-12, relative CG tolerance 1e-3, 500 CG iterations, Armijo 1e-4 and at most 30 halvings
are fixed. The budget is 100 Newton steps, not 100 scaling updates. Acceptance requires the sum
of original-plan row and column L1 marginal errors to be at most 1e-6. No Gaussian fallback is
allowed for this operational score. WDA's differentiable finite-iteration objective is separate.

Marginal feasibility is not an out-of-sample coordinate error bound: two certified solutions in
development differed non-additively by about 0.066 on probes. The result record does not claim
score or ranking equivalence across numerical procedures. Tiny exact fixtures cover dependency,
gauge handling, independent coupling, class swaps, round trips, and independent residual checks.

## Evaluation and artifacts

All paired endpoints use the same E1-eligible, finite-across-arms test sequences. Report assigned,
eligible and excluded counts. Use 1,000 paired class-stratified sequence bootstrap draws for AUROC
and contrasts, linear 95% percentile intervals, and exact two-sided 95% binomial intervals for
rates. These are pointwise conditional intervals, without retraining or seed uncertainty.

Episode timing distinguishes early, in-window, late and absent confirmations. Delay is conditional
on detection, with null empty bootstrap draws and an explicit finite-draw count. No delay is
imputed to misses. Remaining occupancy sums predicted future activity probabilities including
re-entry; evaluate every eligible frame, with truth-absent/present MAE and all-frame Spearman.
This is neither contiguous dwell time nor time until the policy next emits HOLD.

`fitted.json` serializes E1, whitening, WDA, references/potentials/centering, emissions and state
links. `selection.json` and `training_e1_features.parquet` retain selection/cross-fit evidence.
`calibration.json` retains benign IDs/ranks/thresholds. `val/` and `test/` contain scores, decisions
and frame forecasts. `estimates.json` contains coverage and endpoints; `run_manifest.json` binds
config, inputs, outputs, seeds, revision/dirty state, environment and resource measurements.

Preparation is saved before final OT fitting. Explicit `--reuse-training` validates hashes,
ordered input values, config and implementation before reusing E1/WDA; it never resumes a failed
study directory or reuses old calibration. It is optional, with no automatic retry. Platform-specific
supervisors and historical preparation are not part of this release.

`episode_comparisons --source <completed-episode-bundle> --out <new-directory>` adds paired
timing/occupancy contrasts from saved predictions only. It verifies input hashes and reproduces
original per-arm endpoints before accepting contrasts. This is post-execution secondary reporting,
not a newly registered confirmatory test. Original artifacts are read-only.

## Execution modes

`python -m experiment_runner.operational_architecture` requires explicit `--smoke` or `--execute`
and a fresh `--out`. Smoke constructs 384 spectral sequences, uses the real indicator extractor,
16 references/class and three WDA steps, and exercises every arm. It does not sample the study
population. Full execution uses the selected frozen snapshot and may take hours. CI runs only
bounded fixtures. See [commands and limitations](../../docs/OPERATIONAL_BENCHMARK.md#reproduction-and-provenance).
