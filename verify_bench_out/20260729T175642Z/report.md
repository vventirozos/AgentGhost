# Verifier fault-injection bench

cases: 1 · trials/arm: 2 · seed: 0 · actionable conf ≥ 0.7

## two_stage_on

**TPR (catch rate)**: 0.0 raw / 0.0 actionable (1/1 judged) — **FPR (clean refuted)**: 0.0 raw / 0.0 actionable — **degraded-evidence FP**: None

| fault | expected | n | judged | skipped | confirmed | refuted | uncertain | rate | actionable | mean conf |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | CONFIRMED | 1 | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.0 | 1.0 |
| fact_swap | REFUTED | 1 | 1 | 0 | 1 | 0 | 0 | 0.0 | 0.0 | 1.0 |
