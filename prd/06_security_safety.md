# 06. Security & Safety

This PRD section defines safety and compliance constraints for a defensive hazardous-material sensing software module.

## Security requirements
- No secrets committed to the repository (API keys, tokens, credentials).
- All configs must be safe to share internally (no classified parameters).
- If external services are used in the future, they must be optional and default-off for offline operation.
- The repository must not include proprietary or classified datasets, sensitive procedures, or site-specific emergency-response playbooks.

## Safety requirements (human-in-the-loop)
- The system must output **recommendations** and **confidence/risk annotations**, not autonomous actions.
- Outputs are defensive sensing decision support only; they must not trigger autonomous execution.
- Default decision posture is conservative:
  - uncertain or unstable conditions should lead to HOLD/RESCAN, not CONFIRM.
- Provide reason codes and traceable evidence:
  - which indicators drove the regime classification,
  - which transitions triggered instability,
  - which thresholds were applied by policy.

## Defensive-use boundaries
- Evaluation is synthetic/offline by default and does not require or describe hazardous-material handling.
- Documentation and artifacts must not provide weaponization guidance, release/dispersion guidance, evasion guidance, or operational targeting advice.
- Descriptions should stay within chemical/biological hazard sensing, defensive detection reliability, false-alarm suppression, auditability, and human-in-the-loop reporting.
- Negative results, failed gates, and insufficient separability findings must be reported directly; do not relabel or suppress failures to imply field readiness.

## Auditability
For any confirmed recommendation, log:
- input artifact IDs/hashes,
- regime label and risk score,
- stability grade and persistence estimate,
- policy thresholds and reason codes,
- run manifest pointer.

## Data governance
- Synthetic data may be committed only as small fixtures for tests.
- Large generated data must be gitignored and reproduced via manifests.

## Publication hygiene
- Ensure figures and descriptions do not disclose sensitive site-specific details.
- Keep parameter ranges generic unless explicitly approved for release.
- Distinguish production-grade evaluation infrastructure from publishable scientific claims; a bundle is not publication-ready unless strict gates pass and required criteria are fully evaluable.
