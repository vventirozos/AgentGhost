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
WHOLE evaluation cascade (E2 stages 0-4: static scope, pin smoke, bench
smoke, paired confirm on a held-out slice, operator packet) together
with the harness checksum's call sites, which run before and after
every stage, and the negative controls (E3), which check that the
cascade REFUSES three known-bad candidates.

⚠ AND WHAT IS NOT. **No production code calls the cascade or the
controls.** The mutator proposes and materialises a candidate; nothing
evaluates it. E2/E3 are operator-triggered tools, run by hand, by
design — the promotion path is deliberately not autonomous, because the
cascade's own threat model (see `evaluator.py`) says it guards against
MISTAKEN candidates, not hostile ones. Two consequences a reader should
carry: `autonomous_activity` registers `evolve_proposal` as EXPECT_GATED
for an emitter that never fires in production, and none of this has run
outside a test or an operator session.

This paragraph twice described planned work in the present tense, and
once described built work as missing. A safety narrative that overstates
itself is worse than none; one that understates itself gets the same
reader to the same wrong place by the other road.

This package is mostly *arranging* infrastructure this repo already has.
What it adds is the fence: an allow-list of what may be mutated, plus a
checksum over the harness that scores it — now called around every
cascade stage, so a candidate that edits the tests judging it voids the
generation rather than passing it — and the negative controls, which
demonstrate the cascade refusing three known-bad candidates, so a guard
that never visibly fires is not presumed alive. They are run on demand,
not on a schedule; nothing calls them automatically.

**Promotion is operator-applied.** The loop produces evidence-carrying
patch proposals, never self-applied changes. Autonomy is earned in a
later version, not assumed in this one.
"""
