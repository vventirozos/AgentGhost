# Verifier fault-injection bench

cases: 2 · trials/arm: 16 · seed: 0 · actionable conf ≥ 0.7

## two_stage_on

**TPR (catch rate)**: 0.75 raw / 0.75 actionable (12/12 judged) — **FPR (clean refuted)**: 0.0 raw / 0.0 actionable — **degraded-evidence FP**: 1.0

| fault | expected | n | judged | skipped | confirmed | refuted | uncertain | rate | actionable | mean conf |
|---|---|---|---|---|---|---|---|---|---|---|
| artifact_leak | REFUTED | 2 | 2 | 0 | 2 | 0 | 0 | 0.0 | 0.0 | 1.0 |
| clean | CONFIRMED | 2 | 2 | 0 | 2 | 0 | 0 | 0.0 | 0.0 | 0.95 |
| constraint_violation | REFUTED | 2 | 2 | 0 | 0 | 2 | 0 | 1.0 | 1.0 | 1.0 |
| evidence_truncation | NOT_REFUTED | 2 | 2 | 0 | 0 | 2 | 0 | 1.0 | 1.0 | 0.95 |
| fabrication | REFUTED | 2 | 2 | 0 | 0 | 2 | 0 | 1.0 | 1.0 | 0.9 |
| fact_swap | REFUTED | 2 | 2 | 0 | 1 | 1 | 0 | 0.5 | 0.5 | 0.95 |
| silent_failure | REFUTED | 2 | 2 | 0 | 0 | 2 | 0 | 1.0 | 1.0 | 1.0 |
| wrong_topic | REFUTED | 2 | 2 | 0 | 0 | 2 | 0 | 1.0 | 1.0 | 1.0 |
