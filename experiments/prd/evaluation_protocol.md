# Evaluation Protocol

## Primary evaluation unit: alarm event
An alarm event is defined by contiguous timesteps where the system output is non-HOLD
(or where hazard posterior exceeds a low baseline threshold).

Each event is evaluated for:
- whether it should be CONFIRMED,
- time-to-confirm,
- stability (flicker / chatter),
- persistence estimate quality.

## Ground truth (synthetic)
Derived from simulator scenario tags:
- hazard present if hazard concentration > 0
- benign if hazard concentration == 0
- flicker event tagged by simulator

## Ground truth (real)
If real hazard labels are unavailable, evaluate:
- benign background stability (false alarm suppression),
- repeatability under controlled perturbations,
- correlation between predicted persistence and observed alarm duration.

## Steps (must be automated)
1) Prepare datasets + split manifest
2) Run pipeline: indicators → regimes → embeddings → stability → policies
3) Extract per-event summaries
4) Compute metrics and plots
5) Write run manifest with hashes

## Required temporal scenarios
- stable hazard sequences
- flicker false-alarm sequences
- gradual degradation sequences
- recovery sequences
- mixed-material sequences

## Event extraction
- Confirm time: first timestep action == CONFIRM
- Clear time: first timestep returns to HOLD after CONFIRM
- Flicker: event duration < T_flicker and does not meet persistence threshold

## Statistical reporting
Report for key metrics:
- mean ± std across sequences
- median and IQR
- bootstrap 95% CI for key claims (toggle reduction, false confirm rate, TTC)

## Reproducibility
Each run writes `run_manifest.json` containing:
- git commit hash
- dataset + split manifest hashes
- module config hashes
- experiment variant name
- random seeds
- metrics output hashes
