# Verifier fault-injection bench

cases: 1 · trials/arm: 7 · seed: 0 · actionable conf ≥ 0.7

## two_stage_off

**TPR (catch rate)**: 0.6 raw / 0.6 actionable (5/5 judged) — **FPR (clean refuted)**: 0.0 raw / 0.0 actionable — **degraded-evidence FP**: 1.0

| fault | expected | n | judged | skipped | confirmed | refuted | uncertain | rate | actionable | mean conf |
|---|---|---|---|---|---|---|---|---|---|---|
| artifact_leak | REFUTED | 1 | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.0 | 1.0 |
| clean | CONFIRMED | 1 | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.0 | 1.0 |
| constraint_violation | REFUTED | 1 | 1 | 0 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
| evidence_truncation | NOT_REFUTED | 1 | 1 | 0 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
| fabrication | REFUTED | 1 | 1 | 0 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
| fact_swap | REFUTED | 1 | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.0 | 1.0 |
| silent_failure | REFUTED | 1 | 1 | 0 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
