"""Dream's completion message must report what its PATH can actually produce.

The defect (found 2026-08-09): the message always led with
"Synthesized N new meta-memories", but on the TRAJECTORY seed path
`consolidations` is forced empty by design (dream.py: trajectory digests are
not vector fragments, so there is nothing to merge; that path's value is the
heuristics). The headline number was therefore STRUCTURALLY ALWAYS ZERO on the
path dream almost always takes, while the real output sat in the tail clause.

Live evidence: 204 cycles produced 491 heuristics and 37 meta-memories, with
only 2 cycles producing nothing — yet the same log reads "Synthesized 0" 188
times. Three successive wrong diagnoses came out of that ("starved of input",
"lower the extraction threshold", "~92% no-op") before anyone read the code.
"""

import re
from pathlib import Path

SRC = Path("src/ghost_agent/core/dream.py").read_text()


def _completion_block() -> str:
    """The whole completion block, bounded by a real ANCHOR rather than a
    character count.

    ⚠ It used to slice a fixed 2200 chars, so adding a comment inside the
    block pushed the later message variants out of the window and the
    "every variant keeps the marker" check silently stopped covering two of
    the three paths (found 2026-08-21 when a queue-#11 comment did exactly
    that). A window measured in characters is a check that stops checking
    without failing."""
    i = SRC.index("Report what THIS PATH can actually produce")
    j = SRC.index('self.last_dream_outcome = {"phase": "ran"', i)
    return SRC[i:j]


def test_trajectory_path_does_not_headline_meta_memories():
    """On the trajectory path the meta-memory count is 0 by construction, so
    it must not be reported as though it measured anything."""
    seg = _completion_block()
    tp = seg[seg.index("if seeded_from_trajectories:"):seg.index("elif ")]
    assert "applied_consolidations" not in tp, (
        "the trajectory-path message still reports a count that is forced to "
        "zero a few hundred lines earlier")
    assert "heuristic" in tp, "the trajectory path must report its real output"


def test_every_variant_keeps_the_dream_complete_marker():
    """Callers and tests key off this marker; all branches must preserve it."""
    seg = _completion_block()
    variants = re.findall(r'"(Dream Complete[^"]*)"', seg)
    assert len(variants) >= 3, f"expected a message per path, found {variants}"


def test_a_genuinely_empty_cycle_says_so_explicitly():
    """The 2-in-204 case that really produced nothing should be legible as
    such, not indistinguishable from the by-design zero."""
    seg = _completion_block()
    assert "produced nothing this cycle" in seg


def test_the_old_unconditional_format_is_gone():
    """Red-on-revert guard for the exact string that caused the misreading."""
    assert 'f"Dream Complete. Synthesized {applied_consolidations} new meta-memories and extracted' not in SRC
