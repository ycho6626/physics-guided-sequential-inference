# Architecture Validation

## Two Distinct Implementations

This page preserves the original demonstrator's audit disposition. The later
[operational benchmark](OPERATIONAL_BENCHMARK.md) is also included: it makes OT decision-bearing,
uses WDA rather than regime-label distillation, and evaluates a supervised two-state filter with
independent episode truth. Its primary and episode comparisons found no demonstrated AUROC
advantage at the reported precision. That is distinct from the older underpowered battery below.

The public port retains the numerical contracts and synthetic regressions for E1 isolation,
WDA gradients, transport dependency/gauge handling, solver certificates, fixed filtering,
calibration, full action-label serialization, paired inference, and complete bounded smoke paths.
No study population is executed by these tests. Public-port tests are implementation checks,
not newly generated evidence supporting the archived aggregate results.

The curated update was validated under Python 3.12.13 with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`: 198 E1/readout/operational tests (including both
complete authored smokes), followed by 115 remaining experiment tests including portable
saved-prediction reporting and aggregate consistency. Module-01/03/05 non-slow suites passed
23/32/46 tests. Three slow tests across the module and experiment runs were not included.
The existing Matplotlib/NumPy deprecations and Intel/LLVM OpenMP coexistence warning remain
disclosed, not resolved by this port. JSON/config validation, compilation, scoped pyflakes,
local documentation links and whitespace checks also passed. No full study was rerun.

## Scope

This note records the disposition of the original demonstrator architecture after an independent
implementation audit. It concerns Modules 03–06 on the repository's original synthetic generator.
It does not establish, strengthen, or weaken the separate measurement-limit conclusions in the
calibrated case study.

## Transport

The implemented regime label, boundary, and risk score were found to depend on class-conditional
distance quantiles, not on the computed Wasserstein value. Counterfactual Wasserstein values leave
all decision-bearing outputs unchanged. Optimal transport is therefore a fit-time diagnostic, not a
load-bearing decision method.

The implementation now:

- names the distance-quantile construction directly;
- rejects Sinkhorn kernels with no representable mass;
- records an explicit Gaussian-Wasserstein fallback rather than a false successful solve; and
- regression-tests that changing the diagnostic Wasserstein value cannot change scores or actions.

## Fit and Apply

The historical experiment path fit Modules 03–05 before creating the outer split. That path is
retained only for explicit reproduction and refuses to run without `--allow-leaky-historical`.

The primary corrected path:

1. assigns complete sequences to outer train, validation, and test splits;
2. fits regime, embedding, and stability artifacts on outer-train only;
3. applies those serialized artifacts unchanged to held-out sequences; and
4. runs the deterministic policy separately on each split.

Tests require fit-artifact hashes to remain invariant when held-out rows are perturbed, relabelled,
or removed. They also check split integrity, row/sequence locality, configuration mismatch failures,
and serialization round trips.

## Representation-Learning Disposition

The representation-learning code is executable and its held-out application contract is tested.
That is an implementation result, not evidence that the representation improves inference.

In the corrected exploratory battery, the two-dimensional learned arms entered a never-confirm
failure pattern. A secondary four-dimensional arm escaped that pattern inconsistently, which is a
follow-up hypothesis rather than a validated mechanism. The battery contained 18 test events against
a frozen minimum of 20, so its formal status is `INCONCLUSIVE / UNDERPOWERED`. The pattern may guide a
future preregistered study but is not a citable architecture-negative or architecture-positive.

## Reproduction Boundary

New architecture evaluations must use:

```bash
python experiments/scripts/run_corrected_experiment.py \
  --config experiments/configs/nominal.yaml \
  --out /tmp/exp_corrected_nominal
```

Corrected records include the code revision, dirty-tree status, split provenance, and relative
artifact references. A dirty-tree run is exploratory. Generated exploratory batteries are omitted
from this public repository; the executable contracts and regression tests are retained.
