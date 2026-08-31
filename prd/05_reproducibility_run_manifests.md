# 05. Reproducibility & Run Manifests

## Why manifests are required
Published figures and reported metrics must be reproducible. This requires recording:
- what code ran,
- with what configuration,
- with what seeds,
- on what inputs,
- producing what outputs.

## Run Manifest (minimum schema)
A run must write `run_manifest.json` containing:

- `run_id`: stable identifier (timestamp + short hash)
- `created_at`: ISO8601
- `code_revision`: git SHA if available, else a content hash
- `python_version`
- `platform`
- `configs`: list of config file paths and their SHA256 hashes
- `seeds`: dict of all RNG seeds used
- `inputs`: list of input artifacts with SHA256 hashes
- `outputs`: list of output artifacts with SHA256 hashes
- `metrics_summary`: key numbers for quick audit (optional)

## Module-level manifest contract (Modules 01-03)
In addition to `run_manifest.json`, each implemented module emits a module-level manifest:
- Module 01: `sim_manifest.json`
- Module 02: `indicator_manifest.json`
- Module 03: `regimes_manifest.json`

These manifests use a shared top-level schema:
- `module_name`
- `schema_version`
- `created_at`
- `run_id`
- `config_path`
- `config_hash`
- `input_path`
- `input_hash`
- `code_revision`
- `artifacts`

Rules:
- The top-level key set above is exact and identical across Modules 01-03 (no extra module-specific top-level keys).
- `artifacts` is a deterministic sorted map of `relative_output_path -> sha256`.
- Manifest files must not hash themselves.
- Module 01 sets `input_path = null` and `input_hash = null`.
- `created_at` and `run_id` are time-based and expected to differ between runs.

## Determinism rules
- All randomness must be driven by explicit seeds.
- No dependence on system time for randomness.
- When using numerical libraries, document nondeterminism risks and provide mitigations.

## Reproducing a run
A reproducer must be able to:
1. load the same configs,
2. set the same seeds,
3. regenerate the same artifacts (within numeric tolerances),
4. regenerate the same figures.

## Tolerances
When floating point is involved:
- specify tolerances for equality checks (e.g., `atol`, `rtol`) in tests.
- store numeric summaries to detect drift.

## Storage convention
- `runs/<run_id>/` contains:
  - configs snapshot
  - manifests
  - artifacts
  - figures
  - report draft

## Corrected fit/apply experiment records

Corrected architecture-run manifests and ablation-grid indexes include top-level `code_revision`
and `working_tree_dirty` fields. A revision does not identify uncommitted code, so
`working_tree_dirty: true` marks the record as exploratory. Artifact references inside corrected
run, grid, and cell records are relative to the run or grid root so copied bundles remain portable.
