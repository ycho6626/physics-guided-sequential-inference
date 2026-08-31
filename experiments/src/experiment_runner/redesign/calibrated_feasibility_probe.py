"""Assess-before-invest for a CALIBRATED-INSTRUMENT benchmark (path B): can a realistic, instrument-side,
target-independent radiometric calibration remove the band-asymmetric illumination self-calibration nuisance
that caps the multi-frame co-located identity recovery, lifting it to a defensible calibrated positive?

CONTEXT. The single-spectrum study (v7-v13.1) and the v2 multi-frame study both stall on ONE root nuisance:
band-asymmetric illumination that, with NO clean reference, must be self-calibrated blind. On the multi-frame
generator the oracle-illumination ceiling is RMSE 0.12 (positive-grade) but the no-reference self-cal floor is
~0.32, just above the frozen 0.30 bar (a HALT). Real dispersive sensors, however, ARE radiometrically calibrated
(blackbody / integrating-sphere / transfer-standard references) - an instrument-side, target-independent step
that removes illumination WITHOUT touching the target or the co-located interferent. This probe asks whether the
achievable calibration precision is enough.

WHAT IS CALIBRATED, WHAT IS NOT (faithfulness). The calibration removes the per-frame illumination gain up to a
residual cross-band error. The co-located interferent REMAINS and must still be blindly unmixed (CP-ALS); the
component selection is truth-free (cross-family recurrence, no labels/oracle); the frozen bar is not moved. This
is a NEW calibrated-sensor benchmark, not a rescue of the no-reference one. Cheating would be: using clean target
spectra, known interferent spectra, target-informed anchors, or tuning the residual until it passes - none of
which is done here.

THE RESIDUAL MODEL. The two-band ratio is corrupted only by the DIFFERENTIAL (cross-band) gain error between the
1500 and 2100 nm band centers, parametrized as a residual log-gain slope of magnitude sigma_cal (units: per the
normalised z; the cross-band error is sigma_cal * (z_B - z_A) ~ sigma_cal * 1.2). Two limits are modeled:
  - RANDOM per frame (d_i ~ N(0, sigma_cal)): the HARDER case - it breaks trilinearity and corrupts the unmixing.
  - FIXED per instrument (d shared across all frames and families): a pure multiplicative offset on the recovered
    ratio -> rho_hat = rho_true - d*(z_B - z_A) -> slope preserved, |offset| ~ d*1.2 that COUNTS in absolute RMSE
    (it is NOT removed unless a target-independent absolute-scale step removes it; at lab-grade ~1% it is ~0.01).
The random model is conservative; a real within-ensemble residual (seconds-minutes, drift sub-0.2%/month) is
closer to fixed, i.e. easier. A single differential sigma_cal conservatively bounds the wavelength-specific
1500/2100 budget; the pre-registered benchmark must use explicit per-band correlated/uncorrelated terms.

RESULTS (this run; multiframe_abs_probe machinery; seeds/F as below):
  REQUIRED (required_tolerance, oracle-selected, random residual): band-ratio RMSE crosses the 0.30 bar at
    sigma_cal* ~ 0.083, i.e. residual cross-band radiometric gain uncertainty <~ 10% RMS (hard bar), and <~ 6-7%
    for margin on both gates (slope and RMSE). Perfect-calibration anchor reproduces the 0.12 oracle ceiling.
  REALIZABLE (realizable_confirmation, NO ORACLE anywhere - calibrated illumination + truth-free cross-family
    selection + blind CP unmixing): PASSES both gates at 0.6 / 1.2 / 2.4 / 4.8% cross-band (RMSE 0.15 / 0.15 /
    0.16 / 0.20; slope 0.966-0.995; truth-free selection 0.976-0.991). Calibration also nearly solves selection.
  FIXED-BIAS (fixed_bias_check): a fixed cross-band bias is slope-preserving and appears as an offset that COUNTS
    in absolute RMSE (RMSE ~ bias*1.2); at lab-grade ~1% it is ~0.01 (negligible), confirming it does not need to
    be excluded to pass, and must not be hand-waved as removed.
  FROZEN-MODEL (frozen_model_check): computes u_fixed / u_random FROM the frozen sub-term budget (single source of
    truth) via Var_term = u2100^2 + u1500^2 - 2*rho*u1500*u2100 (conservative rho: 0.9 only for the common-mode
    reference-scale term, 0 for all wavelength-specific + random terms), and stress-tests the fixed instrument bias
    DETERMINISTICALLY at {0, +-1sigma, +-2sigma}*u_fixed + random per frame. PASSES every case including +-2sigma:
    lab-primary (u1500 0.50% / u2100 1.00% -> u_diff 1.04%; slope 0.953-0.955 RMSE 0.162-0.165) and field-stress
    (u1500 2.10% / u2100 4.26% -> u_diff 4.55%; slope 0.954-0.956 RMSE 0.162-0.191, worst at +-2sigma = +-9.00%
    fixed). Degenerate-row invalid fraction 0.000 at every case (gate <= INVALID_FRAC_MAX, fail-closed). (Field is
    fixed-dominated hence benign; the pessimistic all-random reading of field k=2 ~9% approaches the ~10% crossing
    - the tightest reading, to be reported for any field/operational claim.)
  ACHIEVABLE (metrology literature, external): lab/NMI-grade cross-band responsivity-RATIO uncertainty ~0.3-1%
    (k=1) (the ratio is <= the absolute per-band figure because common-mode source/lamp/distance terms cancel);
    field/vicarious ~2-4% (k=1, over-states instrument-side); drift sub-0.2%/month.
  VERDICT: GO to SPECIFY / PRE-REGISTER a calibrated-instrument benchmark. Required <~6-7%; achievable ~0.3-1%
    (lab) to ~2-4% (field); the no-oracle pipeline passes even at 4.8% -> comfortably below, not barely. This is
    a feasibility GO, NOT a locked positive: the full pre-registered gate battery (shortcut-freeness, specificity,
    C95, monotone dose, ratio-recovery identity gate, seed x n robustness) must still be built and could fail.

Run: PYTHONPATH=experiments/src python3 -m experiment_runner.redesign.calibrated_feasibility_probe
"""
from __future__ import annotations

import numpy as np

from experiment_runner.redesign.multiframe_abs_probe import (
    WL, _Z, _idxA, _idxB, _analyte, _INTERF, _cp3, _ns, _read_rho, _safecorr,
    I_FRAMES, K_BLOCKS, KAPPA_LO, KAPPA_HI, RHO_LO, RHO_HI,
    SIG_ILL, SIG_BASE, SIG_NOISE, RMSE_MAX, SLOPE_LO, SLOPE_HI,
)

ZSPAN = float(_Z[_idxB].mean() - _Z[_idxA].mean())     # cross-band z separation (~1.2)


def _gen_cal(rng, rho, sigma_cal, fixed_d=None):
    """Genuinely-trilinear co-located ensemble with a KNOWN per-frame illumination gain, removed by radiometric
    calibration up to a residual cross-band log-gain slope error. fixed_d=None => random per frame N(0,sigma_cal)
    (harder); a scalar fixed_d => that fixed slope error on every frame (instrument bias, slope-preserving)."""
    a, u = _analyte(rho), _INTERF
    p = rng.uniform(0.4, 1.6, I_FRAMES); q = rng.uniform(0.4, 1.6, K_BLOCKS)
    r = rng.uniform(0.4, 1.6, I_FRAMES); t = rng.uniform(0.4, 1.6, K_BLOCKS)
    kap = float(np.exp(rng.uniform(np.log(KAPPA_LO), np.log(KAPPA_HI))))
    T = np.einsum("i,k,j->ijk", p, q, a) + kap * np.einsum("i,k,j->ijk", r, t, u)
    logG = np.empty((I_FRAMES, len(WL)))
    for i in range(I_FRAMES):
        logG[i] = rng.normal(0, SIG_ILL) + rng.normal(0, SIG_ILL) * _Z
        T[i] *= np.exp(logG[i])[:, None]
    T += rng.normal(0, SIG_BASE * np.abs(T).mean(), T.shape)
    T += rng.normal(0, SIG_NOISE * np.abs(T).mean(), T.shape)
    Tc = np.empty_like(T)
    for i in range(I_FRAMES):
        if fixed_d is None:
            d = rng.normal(0, sigma_cal)                                   # random per frame (harder case)
        else:
            d = fixed_d + (rng.normal(0, sigma_cal) if sigma_cal > 0 else 0.0)   # fixed instrument bias (+ optional random)
        Ghat = np.exp(logG[i] + d * _Z)                                    # calib estimate = true gain * residual
        Tc[i] = np.clip(T[i] / Ghat[:, None], 1e-9, None)
    return Tc, a


def _sl(x, y):
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def _rm(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _recover(seeds, F, sigma_cal, fixed_d=None, fixed_diff_std=None, seed0=1000):
    """Returns (oracle_slope, oracle_rmse, crossfam_slope, crossfam_rmse, select_acc, invalid_frac) over seeds.
    oracle = component picked by correlation with the true analyte; crossfam = truth-free cross-family recurrence.
    invalid_frac = fraction of crossfam rows that are degenerate (|rho_hat| >= 3); reported so degenerate rows are
    NEVER silently excluded (the full battery gates on a pre-registered max, fail-closed). fixed_diff_std, if set,
    draws a per-instrument FIXED cross-band bias once (kept for reference; the frozen-model grid uses deterministic
    fixed_d instead)."""
    o_sl, o_rm, c_sl, c_rm, acc, invf = [], [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed0 + seed)
        fd = fixed_d
        if fixed_diff_std is not None:
            fd = float(rng.normal(0, fixed_diff_std))         # instrument fixed cross-band bias, drawn once
        rhos = rng.uniform(RHO_LO, RHO_HI, F)
        specs = np.empty((F, 2, len(WL))); truea = np.empty((F, len(WL)))
        for f in range(F):
            Tc, a = _gen_cal(rng, rhos[f], sigma_cal, fd)
            _, B, _ = _cp3(Tc, rng, 2, 4, 70)
            specs[f, 0] = _ns(B[:, 0]); specs[f, 1] = _ns(B[:, 1]); truea[f] = a
        opick = np.array([int(np.argmax([_safecorr(specs[f, k], truea[f]) for k in (0, 1)])) for f in range(F)])
        S = specs.reshape(2 * F, len(WL)); cos = S @ S.T; cpick = np.empty(F, int)
        for f in range(F):
            sc = []
            for c in (0, 1):
                row = cos[2 * f + c].reshape(F, 2).copy(); row[f] = -1; sc.append(row.max(axis=1).mean())
            cpick[f] = 1 - int(np.argmax(sc))
        orho = np.array([_read_rho(specs[f, opick[f]]) for f in range(F)])
        crho = np.array([_read_rho(specs[f, cpick[f]]) for f in range(F)])
        mo = np.abs(orho) < 3; mc = np.abs(crho) < 3
        o_sl.append(_sl(rhos[mo], orho[mo])); o_rm.append(_rm(rhos[mo], orho[mo]))
        c_sl.append(_sl(rhos[mc], crho[mc])); c_rm.append(_rm(rhos[mc], crho[mc]))
        acc.append(float((opick == cpick).mean())); invf.append(float((~mc).mean()))
    return np.mean(o_sl), np.mean(o_rm), np.mean(c_sl), np.mean(c_rm), np.mean(acc), np.mean(invf)


def required_tolerance(seeds=(1, 2), F=200):
    """Sweep the random residual sigma_cal and find the 0.30 crossing (oracle-selected, isolating illumination)."""
    print(f"REQUIRED tolerance (z-sep {ZSPAN:.3f}; bar RMSE <= {RMSE_MAX}):")
    anch0 = _recover(seeds, F, 0.0)
    print(f"  perfect calibration (sigma_cal=0)   slope {anch0[0]:.3f}  RMSE {anch0[1]:.3f}   <- oracle-illum ceiling")
    rows = [(0.0, anch0[0], anch0[1])]
    for sig in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        o_sl, o_rm, *_ = _recover(seeds, F, sig)
        rows.append((sig, o_sl, o_rm))
        print(f"  sigma_cal {sig:.3f}  (cross-band ~{sig*ZSPAN*100:4.1f}%)   slope {o_sl:.3f}  RMSE {o_rm:.3f}"
              + ("   <= bar" if o_rm <= RMSE_MAX else ""))
    cross = None
    for (a0, _, y0), (a1, _, y1) in zip(rows, rows[1:]):
        if (y0 - RMSE_MAX) * (y1 - RMSE_MAX) <= 0 and y1 != y0:
            cross = a0 + (RMSE_MAX - y0) * (a1 - a0) / (y1 - y0); break
    if cross is not None:
        print(f"  --> RMSE crosses {RMSE_MAX} at sigma_cal* ~ {cross:.3f}  "
              f"(required cross-band uncertainty <~ {cross*ZSPAN*100:.1f}% RMS; <~6-7% for margin)")
    return cross


def realizable_confirmation(seeds=(1, 2, 3), F=250):
    """NO ORACLE anywhere: calibrated illumination + truth-free cross-family selection + blind CP unmixing."""
    print(f"REALIZABLE confirmation (no oracle; bar slope in [{SLOPE_LO},{SLOPE_HI}] AND RMSE <= {RMSE_MAX}):")
    for sig in (0.005, 0.010, 0.020, 0.040):
        osl, orm, csl, crm, acc, inv = _recover(seeds, F, sig, seed0=2000)
        ok = (SLOPE_LO <= csl <= SLOPE_HI) and (crm <= RMSE_MAX)
        print(f"  cross-band {sig*ZSPAN*100:4.1f}%  truth-free slope {csl:.3f} RMSE {crm:.3f}  "
              f"(oracle {osl:.3f}/{orm:.3f}) select {acc:.3f} invalid {inv:.3f}  -> {'PASS' if ok else 'below bar'}")


def fixed_bias_check(seeds=(1, 2, 3), F=250):
    """A FIXED cross-band bias is slope-preserving and appears as an offset that COUNTS in absolute RMSE."""
    print("FIXED-bias check (instrument bias, same on every frame; counts in absolute RMSE, not removed):")
    for b in (0.008, 0.020, 0.050):
        osl, orm, csl, crm, acc, inv = _recover(seeds, F, 0.0, fixed_d=b, seed0=3000)
        print(f"  fixed d={b:.3f} (cross-band {b*ZSPAN*100:4.1f}%)  truth-free slope {csl:.3f} RMSE {crm:.3f}  "
              f"(predicted offset ~{b*ZSPAN:.3f})")


# ------------------------------------------------------------------- FROZEN numeric calibration-residual lockfile
# Sub-term budget = the SINGLE SOURCE OF TRUTH (spec section 3.1): (name, u1500, u2100, cross-band rho, is_random).
# Conservative rho: positive ONLY for the genuinely common-mode reference-scale term (source/lamp/distance/
# alignment scale the whole spectrum together, so they cancel in a ratio); rho=0 for every wavelength-specific
# term (detector responsivity slope, stray-light/OSF/grating, wavelength-scale) and the per-frame random term.
INVALID_FRAC_MAX = 0.05                                      # degenerate-row gate (fail-closed above this)
_CAL_BUDGET_LAB = [                                          # lab / NMI-grade PRIMARY (k=1, relative)
    ("reference-scale (source/lamp/distance/alignment)", 0.0030, 0.0030, 0.9, False),
    ("detector responsivity + nonlinearity + dark",      0.0030, 0.0085, 0.0, False),
    ("stray-light / out-of-band / OSF / grating",        0.0015, 0.0035, 0.0, False),
    ("wavelength-scale / bandpass registration",         0.0010, 0.0015, 0.0, False),
    ("reproducibility (per-frame)",                      0.0020, 0.0020, 0.0, True),
]
_CAL_BUDGET_FIELD = [                                        # field / vicarious remote-sensing-grade STRESS (k=1)
    ("reference-scale (source/lamp/distance/alignment)", 0.0100, 0.0100, 0.9, False),
    ("detector responsivity + nonlinearity + dark",      0.0150, 0.0350, 0.0, False),
    ("stray-light / out-of-band / OSF / grating",        0.0080, 0.0200, 0.0, False),
    ("wavelength-scale / bandpass registration",         0.0050, 0.0080, 0.0, False),
    ("reproducibility (per-frame)",                      0.0050, 0.0050, 0.0, True),
]


def _budget_stats(budget):
    """Combine a sub-term budget into (u_1500, u_2100, u_diff, u_fixed, u_random) via the per-term covariance
    Var_term(diff) = u2100^2 + u1500^2 - 2*rho*u1500*u2100 (fixed and random summed separately)."""
    u15 = float(np.sqrt(sum(a ** 2 for _, a, _, _, _ in budget)))
    u21 = float(np.sqrt(sum(b ** 2 for _, _, b, _, _ in budget)))
    vfix = sum(b ** 2 + a ** 2 - 2 * r * a * b for _, a, b, r, rnd in budget if not rnd)
    vrnd = sum(b ** 2 + a ** 2 - 2 * r * a * b for _, a, b, r, rnd in budget if rnd)
    return u15, u21, float(np.sqrt(vfix + vrnd)), float(np.sqrt(vfix)), float(np.sqrt(vrnd))


def frozen_model_check(seeds=(1, 2, 3), F=250):
    """Instantiate the frozen sub-term budget (spec section 3.1/3.2) and confirm the no-oracle pipeline passes.
    u_fixed / u_random are computed FROM the sub-term table (single source of truth). The fixed instrument bias is
    stress-tested DETERMINISTICALLY at {0, +-1sigma, +-2sigma}*u_fixed (not a single lucky random draw), each with
    the random per-frame u_random overlaid; the +-2sigma case is the auditable worst case. Degenerate rows are
    gated (invalid fraction must be <= INVALID_FRAC_MAX), never silently excluded."""
    for name, budget in (("lab-primary", _CAL_BUDGET_LAB), ("field-stress", _CAL_BUDGET_FIELD)):
        u15, u21, ud, ufix, urnd = _budget_stats(budget)
        print(f"FROZEN-model {name}: u1500={u15*100:.2f}% u2100={u21*100:.2f}% -> u_diff={ud*100:.2f}% "
              f"(fixed {ufix*100:.2f}% + random {urnd*100:.2f}%); deterministic fixed grid +-1,2 sigma:")
        worst_ok = True
        for k in (0.0, 1.0, -1.0, 2.0, -2.0):
            osl, orm, csl, crm, acc, inv = _recover(seeds, F, urnd / ZSPAN, fixed_d=(k * ufix) / ZSPAN, seed0=4000)
            ok = (SLOPE_LO <= csl <= SLOPE_HI) and (crm <= RMSE_MAX) and (inv <= INVALID_FRAC_MAX)
            worst_ok = worst_ok and ok
            print(f"    fixed {k:+.0f}sigma ({k*ufix*100:+5.2f}%)  slope {csl:.3f} RMSE {crm:.3f} invalid {inv:.3f}"
                  f"  -> {'PASS' if ok else 'FAIL'}")
        print(f"    => {name}: {'PASS (all deterministic cases incl +-2 sigma)' if worst_ok else 'FAIL'}")


if __name__ == "__main__":
    print("=" * 100)
    required_tolerance()
    print("-" * 100)
    realizable_confirmation()
    print("-" * 100)
    fixed_bias_check()
    print("-" * 100)
    frozen_model_check()
    print("=" * 100)
    print("VERDICT: GO to SPECIFY / PRE-REGISTER a calibrated-instrument benchmark (feasibility GO, not a locked "
          "positive). Required <~6-7% cross-band; achievable ~0.3-1% (lab) to ~2-4% (field); no-oracle pipeline "
          "passes to 4.8%. Full pre-registered gate battery still to be built.")
