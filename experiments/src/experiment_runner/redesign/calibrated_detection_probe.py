"""Detection-gate assess-before-invest for the calibrated-instrument benchmark -> DETECTION-NEGATIVE.

The identity half (calibrated_instrument_benchmark.py) is a POSITIVE on the genuinely-trilinear co-located
stratum. This probe asks the assess-before-invest question for the DETECTION half, BEFORE building a detection
battery: can a realizable present/absent detector hold SPECIFICITY (analyte-absent false-positive rate <= alpha
under ONE global threshold) with a finite 95%-detection limit (C95), against an UNKNOWN co-located interferent,
on the calibrated ensemble? v12's binding negative was exactly specificity-under-overlap; calibration removes the
illumination nuisance, NOT the interferent, so the question is whether the multi-frame second-order advantage
rejects the interferent per-sample.

FAITHFUL DESIGN (target-independent interferent prior): the co-located interferent SHAPE VARIES per sample
(width WU ~ U(36,50), i.e. >= 20% wider than the analyte WB=30; a single FIXED shape would be trivially
blank-learnable = designing-to-pass), strength kappa ~ logU(0.3,3). Known pure-analyte template (the single
calibration standard the second-order advantage presupposes).

DETECTORS on the calibrated multi-frame ensemble (mean spectrum unless noted):
  single : matched-to-analyte-template response (v12-like; the interferent leaks into the analyte band).
  oracle : net-analyte-signal projecting out the TRUE per-sample interferent direction (CEILING; not realizable).
  blank  : NAS projecting out a blank-trained interferent SUBSPACE (top-K PCA of analyte-absent training means).
  cp     : per-sample second-order (CP) analyte-component ENERGY x template-match (the second-order advantage,
           realizable, blind; A,B unit-normalized so energy is read from the A(frame) x C(block) modes).

DECISION (Boque/Ferre/Faber/Olivieri two-error framing, mapped to a qualitative present/absent call): a global
critical level L_c from the analyte-absent (H0) distribution at alpha=0.05; detection power = P(score > L_c) vs
dose; C95 = smallest dose reaching 0.95 power. Specificity holds under one threshold iff analyte-absent FPR ~
alpha across the interferent prior AND C95 is finite.

RESULT (this configuration; cos(analyte,interferent-template) ~ 0.97, near linear dependence): the ORACLE detector
has C95 = 0.3 (the information IS present once the true interferent is removed), but ALL TESTED realizable /
off-the-shelf detectors (single, blank-subspace NAS, per-sample CP) NEVER reach 0.95 detection power at alpha=0.05
-> NO finite C95. This is NOT a formal bound over all conceivable detectors; it is a feasibility-level negative
over the tested realizable class, grounded in the second-order-advantage literature: under near-linear-dependence
the net analyte signal collapses and recovery degrades to a rotational-ambiguity BAND, so a single-threshold
present/absent decision is not defensible. Reproduces v12's specificity-under-overlap negative.

THE SPLIT: identity recovery (an ensemble RATIO) is co-linearity-ROBUST via the concentration-mode trilinearity
(identity benchmark: recovery holds to condition number ~1700); per-sample DETECTION loses the NAS as the analyte
and interferent shapes align, and cannot hold C95 at a fixed false-positive rate. Opposite co-linearity robustness
-> the complete calibrated-instrument benchmark is a POSITIVE identity half + a NEGATIVE detection half; a COMPLETE
calibrated positive is NOT achieved. Recommendation: document the split; do not build a detection battery toward a
foregone negative.

Run: PYTHONPATH=experiments/src python3 -m experiment_runner.redesign.calibrated_detection_probe
"""
from __future__ import annotations

import numpy as np

from experiment_runner.redesign.multiframe_abs_probe import (
    WL, _Z, _g, CA, CB, WB, _GA, _GB, _analyte, _ns, _cp3,
    I_FRAMES, K_BLOCKS, RHO_LO, RHO_HI, SIG_ILL, SIG_BASE, SIG_NOISE,
)
from experiment_runner.redesign.calibrated_feasibility_probe import _budget_stats, _CAL_BUDGET_LAB, ZSPAN

RAND = _budget_stats(_CAL_BUDGET_LAB)[4] / ZSPAN     # lab-primary random per-frame residual (calibrated)
_TMPL = _ns(_GA + _GB)                                # known analyte band template (width WB=30)
WU_LO, WU_HI = 36.0, 50.0                             # varying interferent width prior (co-located, >= 20% wider)
ALPHA = 0.05
DOSES = (0.3, 0.6, 1.0, 1.5, 2.5, 4.0)
DET_NAMES = ("single", "oracle", "blank", "cp")


def _gen(rng, dose, wu=None, kappa=None):
    wu = rng.uniform(WU_LO, WU_HI) if wu is None else wu
    kappa = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))) if kappa is None else kappa
    a = dose * _analyte(rng.uniform(RHO_LO, RHO_HI))
    u = _g(CA, wu) + _g(CB, wu)
    p = rng.uniform(0.4, 1.6, I_FRAMES); q = rng.uniform(0.4, 1.6, K_BLOCKS)
    r = rng.uniform(0.4, 1.6, I_FRAMES); t = rng.uniform(0.4, 1.6, K_BLOCKS)
    T = np.einsum("i,k,j->ijk", p, q, a) + kappa * np.einsum("i,k,j->ijk", r, t, u)
    logG = np.empty((I_FRAMES, len(WL)))
    for i in range(I_FRAMES):
        logG[i] = rng.normal(0, SIG_ILL) + rng.normal(0, SIG_ILL) * _Z
        T[i] *= np.exp(logG[i])[:, None]
    T += rng.normal(0, SIG_BASE * np.abs(T).mean(), T.shape)
    T += rng.normal(0, SIG_NOISE * np.abs(T).mean(), T.shape)
    Tc = np.empty_like(T)
    for i in range(I_FRAMES):
        Tc[i] = np.clip(T[i] / np.exp(logG[i] + rng.normal(0, RAND) * _Z)[:, None], 1e-9, None)
    return Tc, u


def _nas(sp, basis):
    """|<normalized sp, analyte template orthogonalized against the basis>|; the net analyte signal match."""
    tp = _TMPL.copy()
    for b in basis:
        tp = tp - (tp @ b) * b
    n = np.linalg.norm(tp)
    return abs(float(_ns(sp) @ (tp / n))) if n > 1e-9 else 0.0


def _blank_subspace(rng, n=80, K=3):
    means = np.array([_ns(_gen(rng, 0.0)[0].mean(axis=(0, 2))) for _ in range(n)])
    means = means - means.mean(0)
    _, _, Vt = np.linalg.svd(means, full_matrices=False)
    return [Vt[k] / np.linalg.norm(Vt[k]) for k in range(K)]


def _detect(Tc, u_true, blank_basis, rng):
    sp = Tc.mean(axis=(0, 2))
    single = abs(float(_ns(sp) @ _TMPL))
    oracle = _nas(sp, [_ns(u_true)])
    blank = _nas(sp, blank_basis)
    A, B, C = _cp3(Tc, rng, 2, 6, 80)
    specs = [_ns(B[:, 0]), _ns(B[:, 1])]
    energy = [float(np.linalg.norm(A[:, k]) * np.linalg.norm(C[:, k])) for k in (0, 1)]
    cp = max(energy[k] * max(0.0, float(specs[k] @ _TMPL)) for k in (0, 1))
    return {"single": single, "oracle": oracle, "blank": blank, "cp": cp}


def _scores(det, dose, n, rng, blank_basis):
    return np.array([_detect(*(_gen(rng, dose)), blank_basis, rng)[det] for _ in range(n)])


def run_probe():
    rng = np.random.default_rng(3)
    blank_basis = _blank_subspace(rng, n=80, K=3)
    cos = abs(float(_ns(_g(CA, (WU_LO + WU_HI) / 2) + _g(CB, (WU_LO + WU_HI) / 2)) @ _TMPL))
    print("=" * 96)
    print(f"CALIBRATED DETECTION assess-before-invest. interferent WU ~ U({WU_LO},{WU_HI}) (analyte WB={WB}); "
          f"kappa ~ logU(0.3,3); alpha={ALPHA}. cos(interferent,template) ~ {cos:.3f} (near linear dependence).")
    print("=" * 96)
    Lc, fpr = {}, {}
    for det in DET_NAMES:
        Lc[det] = float(np.percentile(_scores(det, 0.0, 200, np.random.default_rng(10), blank_basis), 95))
        fpr[det] = float((_scores(det, 0.0, 200, np.random.default_rng(50), blank_basis) > Lc[det]).mean())
        print(f"  {det:>7}: L_c={Lc[det]:.4f}  held-out analyte-absent FPR={fpr[det]:.3f} (target ~{ALPHA})")
    print("-" * 96)
    print(f"  {'dose':>5} | " + " ".join(f"{d:>8}" for d in DET_NAMES))
    power = {det: [] for det in DET_NAMES}
    for dose in DOSES:
        for det in DET_NAMES:
            power[det].append(float((_scores(det, dose, 120, np.random.default_rng(200 + int(dose * 100)),
                                              blank_basis) > Lc[det]).mean()))
        print(f"  {dose:>5.1f} | " + " ".join(f"{power[det][-1]:>8.3f}" for det in DET_NAMES))
    print("-" * 96)
    c95 = {}
    for det in DET_NAMES:
        c95[det] = next((d for d, p in zip(DOSES, power[det]) if p >= 0.95), None)
        print(f"  {det:>7}: C95 = {('%.2f' % c95[det]) if c95[det] else 'NOT REACHED (no dose gives 0.95 power at alpha)'}")
    realizable = ("single", "blank", "cp")
    verdict = ("DETECTION-NEGATIVE (all tested realizable detectors: no finite C95)"
               if all(c95[d] is None for d in realizable) and c95["oracle"] is not None
               else "INCONCLUSIVE -- re-examine")
    print("=" * 96)
    print(f"VERDICT: {verdict}. oracle C95={c95['oracle']} (info present); realizable {realizable} -> no finite C95. "
          "Feasibility-level negative over the TESTED realizable class (not a formal bound over all detectors); "
          "grounded in the second-order-advantage NAS-collapse under near-linear-dependence. SPLIT: identity half "
          "POSITIVE (co-linearity-robust), detection half NEGATIVE -> complete calibrated positive NOT achieved.")
    print("=" * 96)
    return {"cos": cos, "alpha": ALPHA, "Lc": Lc, "fpr": fpr, "power": {d: power[d] for d in DET_NAMES},
            "doses": list(DOSES), "c95": c95, "verdict": verdict}


if __name__ == "__main__":
    run_probe()
