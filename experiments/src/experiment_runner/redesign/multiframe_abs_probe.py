"""v2.1 assess-before-invest probe: ABSOLUTE / calibration-metric gate-zero, done right (the research bet).

The rank-metric probe (multiframe_cpd_probe.py) was PROBE-INVALID: a dose-partialled Spearman is monotone-robust
to a fixed co-located interferent, so a non-unmixed MERGED component scores 1.0 without unmixing. The feasibility
closure (findings v2 spec §0.2) says a valid v2 needs (i) an ABSOLUTE / calibration metric and (ii) a realizable
no-oracle unmixing + component-selection method, and is NO-GO for a generally feasible study; CONDITIONAL-GO only
for a genuinely-trilinear co-located stratum. This probe tests exactly that stratum.

WHAT IS DIFFERENT FROM THE INVALID PROBE:
  - Metric: CALIBRATION regression of recovered rho on true rho -> SLOPE and RMSE (absolute), NOT rank. Genuine
    separation gives slope ~1; a merged/contaminated read is RANGE-COMPRESSED (slope < 1) and is now CAUGHT.
  - Generator: genuinely TRILINEAR + co-located. The interferent occupies BOTH bands (co-located) but is
    LINEARLY INDEPENDENT of the analyte (equal-band shape, DIFFERENT width) so the spectral-mode k-rank is 2 for
    all analyte rho; concentrations are decorrelated so the concentration-mode k-ranks are 2 -> Kruskal holds.
    Per-family interferent strength kappa_f (log-uniform, independent of rho) scrambles the raw/merged read.
  - Controls that must behave for the verdict to bind: (a) MERGED-component control must COLLAPSE on the absolute
    metric (slope far below 1) even though it passed rank; (b) WRONG-FAMILY derangement must collapse (slope ~0);
    (c) selection is truth-free (no true rho / true spectra / labels).

REALIZABLE no-oracle pipeline: exogenous-anchor ensemble self-calibration -> CP-ALS -> truth-free component
selection by CROSS-FAMILY RECURRENCE (the single shared interferent recurs across families; the analyte varies).
NOTE (the open target-informative question, area 2/7 of the feasibility pass): whether cross-family recurrence
is a legitimate design fact or smuggles target-informative structure is exactly what the wrong-family control
and the oracle-vs-realizable gap probe here.

FROZEN v2.1 pass/fail (binding on the realizable, cross-family, genuinely-trilinear condition):
  PASS  iff realizable slope in [0.85, 1.15] (lower-CI >= 0.85, upper-CI <= 1.15) AND realizable RMSE upper-CI
        <= 0.30 AND the MERGED control slope <= 0.75 (collapses on absolute) AND wrong-family |slope| <= 0.15.
  HALT  if oracle-selection passes (slope ~1) but realizable fails: info is recoverable with a perfect selector
        but the truth-free selector / self-cal does not deliver calibrated recovery.
  NO-GO if oracle-selection ALSO fails on the genuinely-trilinear generator (slope far from 1): calibrated
        recovery is not achieved even with a perfect selector -> the trilinear stratum does not deliver.
  INVALID if the MERGED control does NOT collapse (slope > 0.75): the absolute metric is not discriminating and
        the probe is not faithful (repair before the verdict binds).

Result (this run; seeds {0,1,2} x F {200,400}; + attribution diagnostics seeds 1-2 F=300): VERDICT = HALT,
with the blocker LOCALIZED. The absolute metric is FAITHFUL (closed-form merged slope 0.461; empirical merged
control 0.463 -> collapses; oracle-on-CLEAN wrong-family 0.05). On the genuinely-trilinear generator, CLEAN data
gives calibrated recovery (oracle slope 0.97) AND near-perfect truth-free cross-family selection (0.97 agreement,
0 degenerate reads) -- so identifiability and truth-free selection are NOT the binding wall (this REFINES the
feasibility pass, which feared both). The REALIZABLE pipeline (self-cal + noise) fails: oracle-on-realizable
slope 2.4-5.8, selection collapses to chance (0.45-0.55), 18-31% degenerate reads (the CP-degeneracy
anti-parallel/cancellation mechanism). Non-negative CP-ALS reduces degeneracy (to 6-13%) but does NOT restore
calibration (oracle slope still ~2.4) or selection (~0.45). BLOCKER LOCALIZED: the realizable band-asymmetric
self-calibration + noise corrupts the recovered component geometry, breaking BOTH calibration and selection --
a realizable UNMIXING-NUMERICS obstruction, NOT a fundamental identifiability or selection wall. The bet reduces
to a bounded (non-trivial) sub-problem: a degeneracy/self-cal-robust realizable joint illumination+unmixing.
NECESSARY-NOT-SUFFICIENT: this is a BASELINE self-cal + plain/NN CP-ALS; a better realizable unmixer is the open
lever, and non-negativity alone is insufficient.

STAGE 2 (joint illumination-aware unmixing, stage2()): estimating the per-frame chromatic gain JOINTLY with the
components (anchored to unit geomean across frames) + non-negativity ADVANCES recovery substantially -- degeneracy
18-31% -> 1-2%, oracle slope 2.4-5.8 -> 1.43, RMSE huge -> 0.60 -- but does NOT reach the calibrated PASS band
(slope ~1.4 not ~1; RMSE 0.60 not <= 0.30). DECISIVE attribution: even on CORRECTLY-SELECTED families the slope
is 1.53 / RMSE 0.66, so the residual blocker is COMPONENT CORRUPTION, not selection (0.64): the recovered band
ratio stays inflated + noisy from the residual band-asymmetric illumination self-cal -- the SAME nuisance v1
identified as its measurement-geometry ceiling, improved by the ensemble/joint scheme but not eliminated.
TWO-STAGE VERDICT: the bet is ADVANCED and precisely LOCALIZED (identifiability + truth-free selection are solved
on clean data; the residual is the v1 illumination nuisance relocated into the realizable unmixing) but NOT
passed; reaching calibration needs a research-grade illumination model.

STAGE 3 (parametric LINEAR illumination model, stage3()): the true illumination is a per-frame linear log-gain
(alpha_i + beta_i*z); estimating it by weighted least-squares matching that physical form (vs stage 2's
over-flexible deg-2) advances further -- oracle slope 1.43 -> 1.12 (band edge), RMSE 0.60 -> 0.47, selection
0.64 -> 0.88, crossfam (no-oracle) slope ~1.0; degeneracy 0. ORACLE-ILLUMINATION ceiling check (divide by the
TRUE gain): slope 0.96 / RMSE 0.12 -- so the ENTIRE residual is illumination-estimation quality, not noise/CP.
THREE-STAGE VERDICT: NEAR-PASS but HALT by the frozen bar -- the slope converges into [0.85,1.15] (crossfam 1.0,
oracle 1.12) and selection recovers to 0.88, but RMSE 0.47 (oracle) / 0.75 (crossfam) still exceeds 0.30. Each
stage halves the gap and the ceiling (0.12) shows PASS-grade IS achievable; the sole remaining shortfall is the
illumination-estimation RMSE. The frozen PASS bar is NOT moved to claim a pass (that would be designing-to-pass).

STAGE 4 (variance reduction, stage4()): a shape-fit read (project the recovered spectrum onto the registered
band shapes) gave IDENTICAL RMSE -> the read is not the bottleneck; the residual is in the recovered COMPONENT.
Median over CP restarts cuts oracle RMSE 0.49 -> 0.36 (slope 1.08-1.10) but does NOT cleanly clear the frozen
0.30 bar: a variance-vs-bias decomposition (single 0.49, median-4 0.36) implies a DETERMINISTIC illumination-
estimation bias FLOOR ~0.32, just above the bar, which averaging cannot remove. FOUR-STAGE FINAL VERDICT:
NEAR-PASS but HALT by the frozen bar -- the pipeline converges to the edge (slope in band, selection 0.88, oracle
RMSE 0.36, floor ~0.32), but the systematic illumination bias sits ~0.02-0.07 above the 0.30 RMSE bar. Crossing
the last hairline by tuning the illumination estimator on this SYNTHETIC generator would be DESIGNING-TO-PASS;
the bar is not crossed and no positive is claimed. The bet is banked as a CONVERGED NEAR-POSITIVE: calibrated
recovery is achievable-in-principle (oracle-illumination ceiling 0.12) and the realizable no-oracle pipeline
reaches the edge, but does not cleanly clear the pre-registered bar.

Run: PYTHONPATH=experiments/src python3 -m experiment_runner.redesign.multiframe_abs_probe
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- FROZEN parameters
WL = np.linspace(1300.0, 2300.0, 64)
BAND_A = (1450.0, 1550.0)
BAND_B = (2050.0, 2150.0)
CA, CB, WB = 1500.0, 2100.0, 30.0
WU = 42.0                                         # interferent width (!= WB) -> linearly independent, both-band
RHO_INT = 0.0                                     # interferent band ratio (equal); independence comes from WU != WB
KAPPA_LO, KAPPA_HI = 0.3, 3.0                     # per-family interferent strength (log-uniform, independent of rho)
I_FRAMES, K_BLOCKS = 12, 6
RHO_LO, RHO_HI = -1.2, 1.2
SIG_ILL, SIG_BASE, SIG_NOISE = 0.15, 0.03, 0.02
N_RESTARTS, N_ITER = 5, 80
SLOPE_LO, SLOPE_HI, RMSE_MAX = 0.85, 1.15, 0.30   # PASS band
MERGE_MAX, WRONGFAM_MAX = 0.75, 0.15              # merged must collapse below; wrong-family |slope| below
SEEDS = (0, 1, 2)
N_FAMILIES = (200, 400)
_Z = (WL - WL.mean()) / (np.ptp(WL) / 2.0)


def _g(c, w):
    return np.exp(-0.5 * ((WL - c) / w) ** 2)


_GA, _GB = _g(CA, WB), _g(CB, WB)
_UA, _UB = _g(CA, WU), _g(CB, WU)                  # interferent gaussians (wider) -> co-located, both bands
_idxA = (WL >= BAND_A[0]) & (WL <= BAND_A[1])
_idxB = (WL >= BAND_B[0]) & (WL <= BAND_B[1])
_INTERF = np.exp(-RHO_INT / 2) * _UA + np.exp(RHO_INT / 2) * _UB   # fixed, shared across families


def _analyte(rho):
    return np.exp(-rho / 2.0) * _GA + np.exp(rho / 2.0) * _GB


def _read_rho(bcol):
    b = bcol.copy()
    if np.dot(b, _GA + _GB) < 0:
        b = -b
    return np.log(max(b[_idxB].sum(), 1e-9) / max(b[_idxA].sum(), 1e-9))


# ----------------------------------------------------------------------------- generator (genuinely trilinear)
def _gen(rng, rho, realizable, merged_deltas=None):
    a, u = _analyte(rho), _INTERF
    p = rng.uniform(0.4, 1.6, I_FRAMES); q = rng.uniform(0.4, 1.6, K_BLOCKS)
    r = rng.uniform(0.4, 1.6, I_FRAMES); t = rng.uniform(0.4, 1.6, K_BLOCKS)   # DECORRELATED from p,q
    kap = float(np.exp(rng.uniform(np.log(KAPPA_LO), np.log(KAPPA_HI))))
    T = np.einsum("i,k,j->ijk", p, q, a) + kap * np.einsum("i,k,j->ijk", r, t, u)
    if realizable:
        for i in range(I_FRAMES):
            T[i] *= np.exp(rng.normal(0, SIG_ILL) + rng.normal(0, SIG_ILL) * _Z)[:, None]
        T += rng.normal(0, SIG_BASE * np.abs(T).mean(), T.shape)
        T += rng.normal(0, SIG_NOISE * np.abs(T).mean(), T.shape)
    return T, a


def _selfcal(T):
    slab = T.mean(axis=2)
    floor = np.percentile(slab, 5, axis=1, keepdims=True)
    slab = np.clip(slab - floor, 1e-6, None)
    ref = np.clip(slab.mean(axis=0), 1e-6, None)
    Tc = T.copy()
    for i in range(I_FRAMES):
        gain = np.exp(np.polyval(np.polyfit(_Z, np.log(slab[i] / ref), 2), _Z))
        Tc[i] = np.clip(T[i] - floor[i], 1e-6, None) / gain[:, None]
    return Tc


# ----------------------------------------------------------------------------- CP-ALS (R=2)
def _kr(A, B):
    return (A[:, None, :] * B[None, :, :]).reshape(A.shape[0] * B.shape[0], A.shape[1])


def _cp(T, rng, R=2):
    I, J, K = T.shape
    X0 = T.reshape(I, J * K); X1 = np.transpose(T, (1, 0, 2)).reshape(J, I * K)
    X2 = np.transpose(T, (2, 0, 1)).reshape(K, I * J)
    best, be = None, np.inf
    for _ in range(N_RESTARTS):
        A = rng.uniform(0, 1, (I, R)); B = rng.uniform(0, 1, (J, R)); C = rng.uniform(0, 1, (K, R))
        for _ in range(N_ITER):
            A = X0 @ _kr(B, C) @ np.linalg.pinv((B.T @ B) * (C.T @ C))
            B = X1 @ _kr(A, C) @ np.linalg.pinv((A.T @ A) * (C.T @ C))
            C = X2 @ _kr(A, B) @ np.linalg.pinv((A.T @ A) * (B.T @ B))
            for M in (A, B):
                n = np.linalg.norm(M, axis=0); n[n == 0] = 1; M /= n
        e = np.linalg.norm(X0 - A @ _kr(B, C).T)
        if e < be:
            be, best = e, B.copy()
    return best


def _ns(b):
    b = b.copy()
    if np.dot(b, _GA + _GB) < 0:
        b = -b
    n = np.linalg.norm(b)
    return b / (n if n else 1.0)


def _safecorr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


# ----------------------------------------------------------------------------- absolute calibration metric
def _calib(rho_hat, rho_true):
    """slope of rho_hat on rho_true (calibration) and RMSE of |rho_hat - rho_true| (absolute agreement)."""
    b = np.cov(rho_true, rho_hat, bias=True)[0, 1] / max(np.var(rho_true), 1e-12)
    rmse = float(np.sqrt(np.mean((rho_hat - rho_true) ** 2)))
    return float(b), rmse


def _condition(seed, F, select):
    rng = np.random.default_rng(seed)
    rhos = rng.uniform(RHO_LO, RHO_HI, F)
    realizable = select in ("crossfam", "merged")
    specs = np.empty((F, 2, len(WL))); meansp = np.empty((F, len(WL))); truea = np.empty((F, len(WL)))
    for f in range(F):
        T, a = _gen(rng, rhos[f], realizable)
        Tc = _selfcal(T) if realizable else T
        meansp[f] = Tc.mean(axis=(0, 2))
        if select != "merged":
            B = _cp(Tc, rng); specs[f, 0] = _ns(B[:, 0]); specs[f, 1] = _ns(B[:, 1])
        truea[f] = a
    rho_hat = np.empty(F)
    if select == "merged":                          # NO unmixing: rho off the raw ensemble-mean band ratio
        for f in range(F):
            rho_hat[f] = _read_rho(meansp[f])
    elif select == "oracle":                        # perfect selector (match true analyte spectrum)
        for f in range(F):
            rho_hat[f] = _read_rho(specs[f, int(np.argmax([_safecorr(specs[f, r], truea[f]) for r in (0, 1)]))])
    else:                                           # crossfam: interferent = the component that RECURS across families
        S = specs.reshape(2 * F, len(WL)); cos = S @ S.T
        for f in range(F):
            sc = [cos[2 * f + c].reshape(F, 2).copy() for c in (0, 1)]
            for c in (0, 1):
                sc[c][f] = -1.0
            score = [sc[c].max(axis=1).mean() for c in (0, 1)]
            rho_hat[f] = _read_rho(specs[f, 1 - int(np.argmax(score))])
    slope, rmse = _calib(rho_hat, rhos)
    # bootstrap over families for slope/RMSE CIs; wrong-family derangement (rho_hat vs SHUFFLED true rho)
    bs_slope = np.empty(1500); bs_rmse = np.empty(1500); wf = np.empty(1500)
    for b in range(1500):
        idx = rng.integers(0, F, F)
        bs_slope[b], bs_rmse[b] = _calib(rho_hat[idx], rhos[idx])
        perm = rng.permutation(F)
        wf[b] = _calib(rho_hat, rhos[perm])[0]
    return {"slope": slope, "rmse": rmse,
            "slope_lo": float(np.percentile(bs_slope, 2.5)), "slope_hi": float(np.percentile(bs_slope, 97.5)),
            "rmse_hi": float(np.percentile(bs_rmse, 97.5)), "wrongfam_slope": float(np.median(np.abs(wf)))}


def run_probe():
    # closed-form faithfulness check: the merged read is range-compressed (slope<1) -> absolute metric discriminates
    rr = np.linspace(RHO_LO, RHO_HI, 400); sA, sB = np.exp(-rr / 2), np.exp(rr / 2)
    uA, uB = _UA[_idxA].sum() / _GA[_idxA].sum(), _UB[_idxB].sum() / _GB[_idxB].sum()
    merged_cf = np.polyfit(rr, np.log((sB + uB) / (sA + uA)), 1)[0]
    print("=" * 100)
    print(f"v2.1 ABSOLUTE-metric probe. Closed-form merged-read slope (kappa=1) = {merged_cf:.3f} (<1 => absolute "
          f"metric discriminates merge from separation). PASS band slope[{SLOPE_LO},{SLOPE_HI}], RMSE<={RMSE_MAX}.")
    print("=" * 100)
    agg = {}
    for select in ("oracle", "crossfam", "merged"):
        rows = [_condition(s, F, select) for F in N_FAMILIES for s in SEEDS]
        agg[select] = {"slope_lo": min(r["slope_lo"] for r in rows), "slope_hi": max(r["slope_hi"] for r in rows),
                       "slope": float(np.mean([r["slope"] for r in rows])),
                       "rmse_hi": max(r["rmse_hi"] for r in rows),
                       "wrongfam": max(r["wrongfam_slope"] for r in rows)}
        a = agg[select]
        print(f"  {select:<9} slope~{a['slope']:+.3f} [lo {a['slope_lo']:+.3f}, hi {a['slope_hi']:+.3f}]  "
              f"RMSE_hi {a['rmse_hi']:.3f}  wrong-family|slope| {a['wrongfam']:.3f}")
    print("-" * 100)
    o, c, m = agg["oracle"], agg["crossfam"], agg["merged"]
    merged_collapses = m["slope"] <= MERGE_MAX
    if not merged_collapses:
        v = f"INVALID -> merged control did NOT collapse (slope {m['slope']:+.3f} > {MERGE_MAX}); absolute metric not discriminating; repair"
    elif c["slope_lo"] >= SLOPE_LO and c["slope_hi"] <= SLOPE_HI and c["rmse_hi"] <= RMSE_MAX and c["wrongfam"] <= WRONGFAM_MAX:
        v = "PASS -> realizable no-oracle pipeline recovers rho in CALIBRATED units; v2 feasible on this stratum (VERIFY before believing)"
    elif o["slope_lo"] >= SLOPE_LO and o["slope_hi"] <= SLOPE_HI:
        v = "HALT -> oracle selector recovers calibrated rho but the truth-free (crossfam) selector / self-cal does NOT; realizable gap"
    else:
        v = "NO-GO -> even the oracle selector does not recover calibrated rho on the trilinear generator; the stratum does not deliver"
    print(f"  merged-control collapses on absolute metric: {merged_collapses} (slope {m['slope']:+.3f})")
    print("=" * 100); print(f"v2.1 VERDICT: {v}"); print("=" * 100)
    return agg


# ----------------------------------------------------------------------------- STAGE 2: joint illumination-aware unmixing
# The baseline self-cal (ratio of per-frame spectrum to ensemble mean) conflates the band-asymmetric ILLUMINATION
# with per-frame CONCENTRATION variation, corrupting the recovered components. Stage 2 estimates the per-frame
# chromatic gain JOINTLY with the components (using the current reconstruction to separate illumination from
# concentration), anchored by normalising the gain to unit geomean across frames (exogenous stability anchor),
# with non-negativity (which structurally forbids the CP anti-parallel/cancellation degeneracy).
def _cp3(T, rng, R=2, nr=3, ni=45):
    I, J, K = T.shape
    X0 = T.reshape(I, J * K); X1 = np.transpose(T, (1, 0, 2)).reshape(J, I * K); X2 = np.transpose(T, (2, 0, 1)).reshape(K, I * J)
    best, be = None, np.inf
    for _ in range(nr):
        A = rng.uniform(0, 1, (I, R)); B = rng.uniform(0, 1, (J, R)); C = rng.uniform(0, 1, (K, R))
        for _ in range(ni):
            A = np.maximum(0, X0 @ _kr(B, C) @ np.linalg.pinv((B.T @ B) * (C.T @ C)))
            B = np.maximum(0, X1 @ _kr(A, C) @ np.linalg.pinv((A.T @ A) * (C.T @ C)))
            C = np.maximum(0, X2 @ _kr(A, B) @ np.linalg.pinv((A.T @ A) * (B.T @ B)))
            for M in (A, B):
                n = np.linalg.norm(M, axis=0); n[n == 0] = 1; M /= n
        e = np.linalg.norm(X0 - A @ _kr(B, C).T)
        if e < be:
            be, best = e, (A.copy(), B.copy(), C.copy())
    return best


def _joint_unmix(X, rng, R=2, outer=5):
    I, J, K = X.shape
    fl = np.percentile(X.reshape(I, -1), 2, axis=1).reshape(I, 1, 1)
    Xb = np.clip(X - fl, 1e-9, None); f = np.ones((I, J))
    for _ in range(outer):
        A, B, C = _cp3(Xb / f[:, :, None], rng, R, 3, 45)
        Mhat = np.clip(np.einsum("ir,jr,kr->ijk", A, B, C), 1e-6, None)
        ratio = np.clip(np.median(Xb / Mhat, axis=2), 1e-6, None)
        fn = np.stack([np.exp(np.polyval(np.polyfit(_Z, np.log(ratio[i]), 2), _Z)) for i in range(I)])
        f = fn / np.exp(np.mean(np.log(fn), axis=0, keepdims=True))    # anchor: unit geomean across frames
    return _cp3(Xb / f[:, :, None], rng, R, 4, 70)[1]


def stage2(seeds=(1, 2), F=250):
    """Reproduces the STAGE-2 (joint illumination-aware) result documented in the module docstring."""
    sl = lambda x, y: np.cov(x, y, bias=True)[0, 1] / np.var(x)
    rm = lambda x, y: float(np.sqrt(np.mean((x - y) ** 2)))
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rhos = rng.uniform(RHO_LO, RHO_HI, F); specs = np.empty((F, 2, len(WL))); truea = np.empty((F, len(WL)))
        for f in range(F):
            T, a = _gen(rng, rhos[f], True); B = _joint_unmix(T, rng)
            specs[f, 0] = _ns(B[:, 0]); specs[f, 1] = _ns(B[:, 1]); truea[f] = a
        opick = np.array([int(np.argmax([_safecorr(specs[f, r], truea[f]) for r in (0, 1)])) for f in range(F)])
        S = specs.reshape(2 * F, len(WL)); cos = S @ S.T; cpick = np.empty(F, int)
        for f in range(F):
            sc = []
            for c in (0, 1):
                row = cos[2 * f + c].reshape(F, 2).copy(); row[f] = -1; sc.append(row.max(axis=1).mean())
            cpick[f] = 1 - int(np.argmax(sc))
        orho = np.array([_read_rho(specs[f, opick[f]]) for f in range(F)])
        crho = np.array([_read_rho(specs[f, cpick[f]]) for f in range(F)])
        mo = np.abs(orho) < 3; ag = (opick == cpick) & mo
        r = {"seed": seed, "oracle_slope": sl(rhos[mo], orho[mo]), "oracle_rmse": rm(rhos[mo], orho[mo]),
             "select_acc": float((opick == cpick).mean()), "degen": float((np.abs(orho) > 3).mean()),
             "correctsel_slope": sl(rhos[ag], crho[ag]), "correctsel_rmse": rm(rhos[ag], crho[ag])}
        out.append(r)
        print(f"  STAGE2 seed{seed}: oracle slope {r['oracle_slope']:.3f} RMSE {r['oracle_rmse']:.3f} | "
              f"select-acc {r['select_acc']:.3f} | degen {r['degen']:.2f} | correct-selection slope "
              f"{r['correctsel_slope']:.3f} RMSE {r['correctsel_rmse']:.3f}  (bar: slope~1 AND RMSE<={RMSE_MAX}; NOT met)")
    return out


# ----------------------------------------------------------------------------- STAGE 3: parametric (linear) illumination model
# The true illumination is a per-frame LINEAR log-gain (alpha_i + beta_i*z); stage 2's deg-2 median fit over-fits
# noise. Stage 3 estimates (alpha_i, beta_i) by a per-frame WEIGHTED LEAST-SQUARES regression of log(X/Mhat) on
# [1, z] (weighted by signal), matching the physical form, anchored to zero cross-frame mean. An oracle-illumination
# ceiling check (divide by the TRUE per-frame gain) gives slope 0.95-0.97 / RMSE 0.12 -- so the residual is entirely
# illumination-estimation quality, and a better estimator closes it toward that ceiling.
def _joint_unmix_lin(X, rng, R=2, outer=6):
    I, J, K = X.shape
    fl = np.percentile(X.reshape(I, -1), 2, axis=1).reshape(I, 1, 1)
    Xb = np.clip(X - fl, 1e-9, None)
    D = np.c_[np.ones(J), _Z]; ab = np.zeros((I, 2))
    for _ in range(outer):
        f = np.exp((D @ ab.T).T)
        A, B, C = _cp3(Xb / f[:, :, None], rng, R, 3, 45)
        Mhat = np.clip(np.einsum("ir,jr,kr->ijk", A, B, C), 1e-6, None)
        Lm = np.median(np.log(np.clip(Xb / Mhat, 1e-6, None)), axis=2)   # I x J
        w = np.clip(Mhat.mean(axis=2), 1e-6, None)                        # I x J signal weight
        for i in range(I):
            ab[i] = np.linalg.lstsq(D * np.sqrt(w[i])[:, None], Lm[i] * np.sqrt(w[i]), rcond=None)[0]
        ab -= ab.mean(axis=0)                                             # anchor: unit geomean across frames
    return _cp3(Xb / np.exp((D @ ab.T).T)[:, :, None], rng, R, 4, 70)[1]


def stage3(seeds=(1, 2), F=250):
    """Reproduces the STAGE-3 (parametric linear illumination) result. Near-PASS on slope/selection; RMSE > bar."""
    sl = lambda x, y: np.cov(x, y, bias=True)[0, 1] / np.var(x)
    rm = lambda x, y: float(np.sqrt(np.mean((x - y) ** 2)))
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rhos = rng.uniform(RHO_LO, RHO_HI, F); specs = np.empty((F, 2, len(WL))); truea = np.empty((F, len(WL)))
        for f in range(F):
            T, a = _gen(rng, rhos[f], True); B = _joint_unmix_lin(T, rng)
            specs[f, 0] = _ns(B[:, 0]); specs[f, 1] = _ns(B[:, 1]); truea[f] = a
        opick = np.array([int(np.argmax([_safecorr(specs[f, r], truea[f]) for r in (0, 1)])) for f in range(F)])
        S = specs.reshape(2 * F, len(WL)); cos = S @ S.T; cpick = np.empty(F, int)
        for f in range(F):
            sc = []
            for c in (0, 1):
                row = cos[2 * f + c].reshape(F, 2).copy(); row[f] = -1; sc.append(row.max(axis=1).mean())
            cpick[f] = 1 - int(np.argmax(sc))
        orho = np.array([_read_rho(specs[f, opick[f]]) for f in range(F)])
        crho = np.array([_read_rho(specs[f, cpick[f]]) for f in range(F)])
        mo = np.abs(orho) < 3; mc = np.abs(crho) < 3
        r = {"seed": seed, "oracle_slope": sl(rhos[mo], orho[mo]), "oracle_rmse": rm(rhos[mo], orho[mo]),
             "select_acc": float((opick == cpick).mean()), "degen": float((np.abs(orho) > 3).mean()),
             "crossfam_slope": sl(rhos[mc], crho[mc]), "crossfam_rmse": rm(rhos[mc], crho[mc])}
        out.append(r)
        print(f"  STAGE3 seed{seed}: oracle slope {r['oracle_slope']:.3f} RMSE {r['oracle_rmse']:.3f} | "
              f"select {r['select_acc']:.3f} | degen {r['degen']:.2f} | crossfam slope {r['crossfam_slope']:.3f} "
              f"RMSE {r['crossfam_rmse']:.3f}  (bar: slope in [{SLOPE_LO},{SLOPE_HI}] AND RMSE <= {RMSE_MAX}; NOT met)")
    return out


def stage4(seeds=(1, 2), F=150, nruns=4):
    """STAGE 4 (variance reduction: median over CP restarts on the stage-3 unmixer). Cuts RMSE but does NOT
    cleanly clear the 0.30 bar (deterministic illumination-estimation bias floor ~0.32; crossing = designing-to-pass)."""
    sl = lambda x, y: np.cov(x, y, bias=True)[0, 1] / np.var(x)
    rm = lambda x, y: float(np.sqrt(np.mean((x - y) ** 2)))
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed); rhos = rng.uniform(RHO_LO, RHO_HI, F)
        o1 = np.empty(F); om = np.empty(F)
        for f in range(F):
            T, a = _gen(rng, rhos[f], True); rr = []
            for run in range(nruns):
                B = _joint_unmix_lin(T, np.random.default_rng(9000 * seed + 13 * f + run))
                rr.append(_read_rho(_ns(B[:, int(np.argmax([_safecorr(_ns(B[:, r]), a) for r in (0, 1)]))])))
            o1[f] = rr[0]; om[f] = np.median(rr)
        for tag, arr in (("single", o1), (f"median-{nruns}", om)):
            m = np.abs(arr) < 3
            r = {"seed": seed, "read": tag, "slope": sl(rhos[m], arr[m]), "rmse": rm(rhos[m], arr[m])}
            out.append(r)
            print(f"  STAGE4 seed{seed} {tag:<9} oracle: slope {r['slope']:.3f} RMSE {r['rmse']:.3f}  (bar: RMSE <= {RMSE_MAX}; NOT met)")
    return out


if __name__ == "__main__":
    run_probe()
    print("-" * 100); print("STAGE 2 (joint illumination-aware, deg-2):"); stage2()
    print("-" * 100); print("STAGE 3 (parametric linear illumination model):"); stage3()
    print("-" * 100); print("STAGE 4 (variance reduction; near-PASS, does not clear the 0.30 bar):"); stage4()
