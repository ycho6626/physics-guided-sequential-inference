# Reporting & Figures (Paper Bundle)

## Standard results bundle structure
experiments/results/<run_name>/
  run_manifest.json
  metrics.json
  metrics.md
  figures/
  tables/
  configs/   (copies of module configs + experiment variant)
  logs/

## Table 1 (Main results)
Rows: methods (full pipeline + baselines)
Cols: FCR, MCR, median TTC, toggle rate, flicker confirms, suppression efficiency

## Table 2 (Ablations)
Rows: ablation variants
Cols: deltas vs full system

## Caption guidance
- Avoid internal jargon.
- Use: "risk regimes", "alarm persistence", "reliability gating", "false-alarm suppression".
- If real labels missing, avoid absolute accuracy claims; report stability metrics.

## Reproducibility statement
Every figure/table must cite:
- dataset split hash
- config hashes
- run id
