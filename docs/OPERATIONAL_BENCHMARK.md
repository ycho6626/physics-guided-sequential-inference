# Operational OT, Representation and Temporal Inference

## Question and implementation

Does a specified, decision-bearing OT/representation/filter chain improve held-out sequence
discrimination over simpler controls on the same observations? A separate episode population
also asks about onset detection and future active-frame occupancy.

The chain is **eight indicators + an E1 spectral cone-GLR → train-only whitening → WDA projection
→ centered entropic transport coordinate → supervised two-state filter → calibrated policy**.
OT enters both WDA's objective and the final evidence coordinate. This is a new empirical
construction, not the original demonstrator's Mahalanobis-quantile regimes or its diagnostic W2.
It also adds a raw-spectrum-derived channel absent from that original architecture.

The [implementation contract](../experiments/prd/operational_architecture.md) specifies the
methods, controls, numerical certificate and artifact interfaces. All data are synthetic. Template
names GB/VX identify configured Gaussian shapes, not validated real-agent spectra. Results do
not establish sensing capability or authorize physical deployment.

The preceding representation/temporal factorial comparison is also available in
[`representation_temporal.py`](../experiments/src/experiment_runner/representation_temporal.py):
raw indicators versus the shipped embedding, each with IID-mixture versus HMM readouts.
It supplies the reusable likelihood/calibration machinery and the D′ continuity arm here.
Its own [configuration](../experiments/configs/representation_temporal_v1.json) and authored
smoke tests remain separate; its arm definitions must not be confused with this WDA comparison.

## Held-out sequence results

The 8,000-sequence primary and 4,000-sequence episode populations have separate seeds, fits and
calibration sets. Each sequence has ten frames. Paired comparisons use 764 primary test sequences
(351 benign / 413 hazard), and 386 episode test sequences (189 / 197). E1 channel validity excludes
17/781 and 12/398 assigned test sequences, respectively; there is no additional paired-arm attrition.

| Arm | Primary AUROC [95% interval] | Episode AUROC [95% interval] |
|---|---|---|
| CAND | 0.5346 [0.4954, 0.5761] | 0.5552 [0.4994, 0.6082] |
| −OT | 0.5254 [0.4830, 0.5672] | 0.5470 [0.4912, 0.5961] |
| −REP | 0.5150 [0.4726, 0.5573] | 0.5571 [0.5037, 0.6088] |
| −TEMP | 0.5337 [0.4942, 0.5757] | 0.5527 [0.4963, 0.6049] |
| S1 | 0.5271 [0.4850, 0.5665] | Not in episode design |
| S0 | 0.5354 [0.4922, 0.5750] | 0.5417 [0.4864, 0.5959] |
| D′ | 0.5178 [0.4759, 0.5601] | Not in episode design |

−OT retains WDA; −REP retains the nine-channel input with its declared refitting differences;
−TEMP shares CAND's fitted evidence but removes switching. S1 is the simpler nine-channel
independent-coupling method; S0 sums E1 scores; D′ uses the shipped embedding and HMM readouts.

| CAND minus control | Primary paired AUROC difference [95% interval] | Episode paired difference [95% interval] |
|---|---|---|
| −OT | +0.0092 [−0.0410, +0.0569] | +0.0082 [−0.0223, +0.0422] |
| −REP | +0.0196 [−0.0227, +0.0594] | −0.0019 [−0.0359, +0.0325] |
| −TEMP | +0.0009 [−0.0051, +0.0068] | +0.0024 [−0.0048, +0.0111] |
| S1 | +0.0075 [−0.0453, +0.0623] | Not in episode design |
| S0 | −0.0008 [−0.0546, +0.0513] | +0.0135 [−0.0362, +0.0555] |
| D′ | +0.0168 [−0.0422, +0.0752] | Not in episode design |

Every prescribed paired AUROC interval includes zero. This establishes **no demonstrated CAND
advantage at the reported precision**, not equivalence. −REP's episode marginal interval excludes
0.5; it would be incorrect to say every arm is indistinguishable from chance.

At nominal 1% CONFIRM calibration, CAND detects 5/413 primary hazard sequences (1.21%) with
1/351 benign false positives (0.285%), and 3/197 episode-population hazard sequences (1.52%) with
0/189 false positives. The latter FPR interval reaches 1.93%: nominal calibration is not an
empirical guarantee that the population FPR is below 1%. Achieved FPR differs across arms.
All arms' exact count/rate intervals are in the [aggregate JSON](results/operational_benchmark.json).

## Episode timing and occupancy

Of 101 assigned episodes, 98 are eligible. All four timing-capable arms detect the **same single
episode** in-window, with 97 misses and no early or late confirmations. In-window detection is
1/98 = 1.02% [0.026%, 5.55%]. Sequence recall above counts any confirmation, irrespective of timing.

Conditional median delays are CAND 0, −OT 2, −REP 0 and −TEMP 1 frames. Only 644/1,000 paired
bootstrap draws contain a detection. CAND-minus-control delay intervals collapse to −2, 0 and −1,
respectively. These intervals repeat one observed detection and **do not establish reliable
population timing superiority**. Empty draws stay unevaluable; misses receive no imputed delay.

Future active occupancy is evaluated on all 3,860 eligible test frames, including terminal zeros:
2,372 truth-absent and 1,488 truth-present. CAND MAE is 2.000 / 1.835 frames by those strata;
all-frame Spearman is 0.367. Occupancy allows re-entry and is not contiguous dwell time.

The saved-prediction supplement reports paired secondary differences, CAND minus control:

| Control | Absent MAE difference [95% interval] | Present MAE difference [95% interval] | Spearman difference [95% interval] |
|---|---|---|---|
| −OT | −0.00263 [−0.01407, +0.00809] | +0.01011 [−0.00383, +0.02329] | −0.00063 [−0.00496, +0.00401] |
| −REP | +0.01063 [−0.00106, +0.02571] | +0.00909 [−0.00386, +0.02233] | −0.00289 [−0.00963, +0.00343] |
| −TEMP | +0.00281 [+0.00074, +0.00469] | +0.00531 [+0.00040, +0.01185] | −0.00160 [−0.00322, −0.00038] |

The three pointwise occupancy contrasts favor −TEMP by small amounts. There is no multiplicity
adjustment or predeclared practical-usefulness threshold. Positive occupancy correlation alone
does not establish observation-informed persistence: the shared horizon and terminal zeros
contribute structure, and no time-only comparator was evaluated. Nor was that structure shown
to explain the entire correlation.

## Interpretation

Operational dependency is established; empirical advantage is not. Further architecture investment
on this benchmark is paused. Weak observable evidence, model misspecification, finite-sample
estimation and generator artifacts remain possible explanations, not independently resolved causes.
A weak chosen estimator, even one given privileged information, is not an information ceiling.

Intervals use 1,000 paired class-stratified sequence bootstrap replicates or exact binomial
intervals. They are pointwise and conditional on each fitted deployment/calibration; they exclude
retraining and seed variability. The two populations are not independent retraining replications
of one identical population law. No universal conclusion about OT, representations or HMMs follows.
The [calibrated case study](CASE_STUDY.md) remains a separate generator and estimand.

## Reproduction and provenance

Install with `./scripts/bootstrap.sh`. From the repository root, use new output directories:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m experiment_runner.operational_architecture \
  --config experiments/configs/operational_architecture_v1.json \
  --out runs/primary-fixture --smoke
.venv/bin/python -m experiment_runner.operational_architecture \
  --config experiments/configs/operational_architecture_episode_v1.json \
  --out runs/episode-fixture --smoke
```

Each smoke uses 384 authored sequences and the real indicator extractor; it reduces references
and WDA steps, not the declared arm matrix. Its outputs validate execution, not scientific
performance. `--execute` instead runs the selected frozen study population once and may take hours.
No full study was rerun for this public port. `--reuse-training` requires a compatible complete
preparation bundle; no historical preparation is distributed or silently downloaded.

For an episode bundle generated locally, paired secondary reporting needs no refit or rescoring:

```bash
.venv/bin/python -m experiment_runner.episode_comparisons \
  --source runs/your-episode-study --out runs/your-episode-supplement
```

The [aggregate JSON](results/operational_benchmark.json) transcribes selected verified endpoints
without rounding, records original result hashes, config/seeds and execution revisions, and omits
row-level data. Source execution SHAs identify the research history, not commits in this curated
repository. Config contents and seed namespace strings are preserved to avoid changing RNG streams.
Original studies used Python 3.12.13 on Intel macOS; package versions are in the aggregate record.
The observed OpenMP-runtime coexistence warning does not establish cross-platform determinism.

Two provenance limitations are retained rather than hidden:

- **Primary:** selected settings were retained from the earlier scaling-solver procedure; final OT
  used Newton-CG. Cold execution with current code reselects under Newton-CG and is not promised
  to reproduce that historical selection bitwise. Marginal certification alone is not coordinate
  accuracy. The original primary bundle also truncated action strings to `CONFIR`; numerical
  endpoints independently reproduced unaffected. This port includes the full-label serialization fix.
- **Supplement:** post-execution reporting ran from a dirty candidate with its exact source snapshot
  retained and subsequently audited. It is not retrospectively described as registered or clean-code
  execution. The public `--source` argument replaces the original local path without changing arithmetic.

Large generated bundles, historical job supervisors, numerical-repair journals, licensed inputs
and abandoned study scaffolding are not included. The implementation and bounded tests run without
access to that private workspace; exact reanalysis of historical rows requires the original bundles.
