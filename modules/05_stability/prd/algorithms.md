# Algorithms (Alarm Stability)

This module models alarm stability as a **latent-state process** with observations derived from the
indicator/embedding space and regime scores.

Let:
- Hidden states: s_t ∈ {1..K} (e.g., trusted/ambiguous/degraded/high_risk)
- Observations: y_t (either discrete regime_label or continuous vector obs)
- Transition matrix: A where A[i,j] = P(s_{t+1}=j | s_t=i)
- Emission model: P(y_t | s_t)

The system performs:
- filtering: compute p(s_t | y_1:t)
- optional smoothing: p(s_t | y_1:T)
- persistence estimation based on A and current posterior.

## 1) State semantics and mapping
### Option 1 — States equal regimes (recommended)
K equals the number of risk regimes from Module 03.
Input regime_label can be used as a noisy observation of state.

### Option 2 — Coarsened states
Map multiple regimes into coarser stability states, e.g.:
- trusted = {trusted}
- degraded = {ambiguous, degraded}
- high_risk = {high_risk}

This can improve robustness when data is limited.

## 2) Emission models
### 2.1 Discrete emission (regime-as-observation)
Observation y_t is the discrete `regime_label` produced by Module 03.
Emission is a confusion matrix E where:
E[i, r] = P(observed_regime=r | hidden_state=i)

This captures that regime assignment can be noisy near boundaries.

Pros:
- very interpretable
- easy to train/fit with counts
- stable in low-data regimes

Cons:
- throws away continuous info in x/z

### 2.2 Continuous emission (Gaussian over z or x)
Observation is continuous vector (x_t or z_t).
Emission model:
y_t | s_t=i ~ Normal(μ_i, Σ_i) (diagonal Σ recommended for PoC)

Pros:
- uses continuous structure
- supports geometry-aware inference

Cons:
- requires more data and careful scaling

### 2.3 Hybrid emission (recommended when available)
Use a product emission:
P(y_t | s_t) = P(regime_label | s_t) * P(z_t | s_t)

This preserves interpretability while exploiting continuous structure.

## 3) Transition model and constraints
The transition matrix A is the core of stability modeling.

### 3.1 Structural constraints (recommended)
Enforce that transitions are mostly local in regime ordering:
trusted ↔ ambiguous ↔ degraded ↔ high_risk
Disallow long jumps unless configured:
- trusted → high_risk is rare (but not impossible)

Implement via:
- zeroing disallowed transitions then renormalizing, or
- strong Dirichlet priors pushing them to near-zero.

### 3.2 Geometry-aware transition priors (optional, publication-friendly)
Use indicator-space distance to modulate priors:
- if a sample is near boundary, allow higher transition probability.
- if deep inside a regime, favor self-transition.

This can be implemented by coupling a boundary-distance feature into a time-varying A_t.
For Phase-1 PoC, keep A time-homogeneous; include geometry-aware priors as an ablation.

### 3.3 Flicker modeling
To suppress flicker, you want:
- high self-transition probability in stable states (trusted)
- high probability of returning to benign/high_risk states after transient artifacts?

In alarm-reliability terms, flicker corresponds to:
- short dwell times in confirmable states
- repeated toggling between adjacent states

HMM explicitly models and quantifies this via dwell-time distributions.

## 4) Filtering and smoothing
Use standard forward algorithm:
α_t(i) ∝ P(y_t | s_t=i) Σ_j α_{t-1}(j) A[j,i]
Normalize α_t to sum to 1.

Optionally apply backward smoothing for offline evaluation.

## 5) Persistence / dwell-time estimation
Define confirmable set C (e.g., {trusted} or {trusted, ambiguous}).

Given current posterior p_t over states, define expected remaining time to exit C:
- For each state i∈C, compute expected steps to absorption into complement of C.
- Combine by posterior weight: E[T_exit | p_t] = Σ_i p_t(i) E_i[T_exit]

### Computing E_i[T_exit]
Let Q be submatrix of A restricted to states in C.
Then expected time-to-exit vector is:
t = (I - Q)^(-1) 1
where 1 is a vector of ones.

This quantity is finite only when the confirmable subchain is transient
(`spectral_radius(Q) < 1`). A closed confirmable class has infinite expected
exit time and must fail closed; a pseudoinverse must not turn it into zero.

If dt_seconds is known, convert steps → seconds.

This is a key output: **alarm persistence estimate**.

## 6) Stability grading
Define stability_grade as a deterministic function of:
- p_confirmable = Σ_{i∈C} p_t(i)
- persistence_seconds
- transition trend (optional)

Example:
- stable if p_confirmable ≥ p_thresh AND persistence_seconds ≥ T_min
- unstable otherwise

Provide ordinal grades if desired:
A (highly stable), B (stable), C (uncertain), D (unstable)

## 7) Derived “hazard posterior” (optional)
If hidden states are aligned with hazard reliability, hazard_posterior can be:
- p_confirmable, or
- sum over hazard-likely states.

Keep definition explicit and consistent with paper claims.

## 8) Training / fitting
Two supported modes:
- Configured: A, E, μ, Σ provided explicitly (for controlled demos)
- Fitted: estimate parameters from sequences via EM (Baum–Welch)

Training must be deterministic given seed and split.

### 8.1 EM fitting (discrete emissions)
- Initialize A with strong self-transition.
- Initialize E near identity.
- Run EM for N iterations; stop on log-likelihood improvement threshold.

### 8.2 EM fitting (Gaussian emissions)
- Initialize μ via k-means per state label heuristic.
- Σ as diagonal var with floor.
- Run EM; enforce covariance floors.

## 9) Auditability
For each timestep, emit reason_codes such as:
- `LOW_P_CONFIRMABLE`
- `SHORT_PERSISTENCE`
- `NEAR_BOUNDARY`
- `DEGRADING_FAST`

Reason codes are computed deterministically from intermediate quantities.
