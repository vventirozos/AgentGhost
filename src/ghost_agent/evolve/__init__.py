"""Evolve (§4CN) — archive-based self-evolution of the scaffold.

The DGM-lineage results all reduce to three properties, and the
documented failure mode is always the same one:

* an **executed evaluator the candidate cannot write to** — DGM's agent
  faked test logs; the AI Scientist raised its own timeout;
* **archive/population selection** rather than greedy hill-climbing —
  HGM's finding is that a node's own score poorly predicts its
  descendants';
* **staged cheap-to-expensive gates**, so most candidates die for free.

⚠ WHAT IS ACTUALLY WIRED, as of 2026-08-23: the write fence (allow-list,
case-folded, and asked of the filesystem), the lineage archive, the
MUTATOR — which proposes one diff per cooldown and records it — and the
first two stages of the evaluation cascade (E2 stage 0 static, stage 1
pin smoke) together with the harness checksum's call sites, which now
run before and after every stage. **Stages 2 (bench smoke) and 3 (paired
confirm) are NOT built, and are absent rather than stubbed**: a stage
that returns a pass because nobody implemented it is the built-but-
unwired failure this package exists to avoid. The scheduled negative
controls (E3) are not built either. This paragraph once described all of
it in the present tense; a safety narrative that overstates itself is
worse than none.

This package is mostly *arranging* infrastructure this repo already has.
What it adds is the fence: an allow-list of what may be mutated, plus a
checksum over the harness that scores it — now called around every
cascade stage, so a candidate that edits the tests judging it voids the
generation rather than passing it — and, eventually, negative controls
on a schedule so a guard that never demonstrably fires is not presumed
alive.

**Promotion is operator-applied.** The loop produces evidence-carrying
patch proposals, never self-applied changes. Autonomy is earned in a
later version, not assumed in this one.
"""
