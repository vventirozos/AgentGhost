"""Queue #11 — the dream ledger recorded a constant for the busiest subsystem.

`core/dream.py` drives ~2/3 of live traffic. Its idle cycle writes one row to
the autonomous-activity ledger — the surface `introspect action='activity'`
reads to answer "what did you do while I was away" — and that row was a
CONSTANT string: measured on the live box, **919 dream records, every one
reading "REM cycle ran (memory consolidation / heuristic harvest)"**.

The information to tell them apart is right there. `Dreamer.dream()` returns
either "Dream Complete — consolidated N fragments into M new meta-memories and
extracted H heuristics" or "Dream Complete — produced nothing this cycle: no
consolidation met the compression bar", the call site binds it as
`_dream_msg`, uses it for skip classification, and then threw it away. A
productive cycle and an empty one were indistinguishable in the digest.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from types import SimpleNamespace

import pytest


class _Log:
    def __init__(self):
        self.rows = []

    def record(self, phase, summary, severity="info", **meta):
        self.rows.append((phase, summary, severity, meta))
        return True


def _record(msg):
    """Drive the REAL shaping function and the REAL ledger writer.

    ⚠ The first version of this helper re-implemented the truncation and
    prefix itself, so the "unbounded summary" mutant survived: the test was
    exercising its own copy, not the agent's. The shaping now lives in
    `agent.dream_ledger_summary` precisely so this can call it."""
    from ghost_agent.core.agent import GhostAgent, dream_ledger_summary
    log = _Log()
    fake = SimpleNamespace(context=SimpleNamespace(activity_log=log))
    GhostAgent._record_autonomous_activity(
        fake, "dream", dream_ledger_summary(msg))
    return log.rows[0][1]


class TestTheLedgerCarriesTheOutcome:
    def test_a_productive_cycle_and_an_empty_one_differ(self):
        """The whole defect in one assertion."""
        productive = _record("Dream Complete — consolidated 12 fragments into "
                             "3 new meta-memories and extracted 5 heuristics.")
        empty = _record("Dream Complete — produced nothing this cycle: no "
                        "consolidation met the compression bar and extracted "
                        "0 heuristics.")

        assert productive != empty
        assert "3 new meta-memories" in productive
        assert "produced nothing" in empty

    def test_the_prefix_introspect_matches_on_is_preserved(self):
        """`tests/test_selfhood_introspect_tool.py` and the digest match on
        "REM cycle ran"; changing the record must not orphan them."""
        out = _record("Dream Complete — consolidated 1 fragment.")

        assert out.startswith("REM cycle ran")

    def test_the_summary_is_bounded(self):
        out = _record("x" * 5000)

        assert len(out) < 300

    def test_whitespace_is_collapsed(self):
        """The Dreamer's message can carry newlines from a metrics note; the
        ledger is a one-line summary."""
        out = _record("Dream Complete —\n  consolidated 2\n\tfragments.")

        assert "\n" not in out and "\t" not in out
        assert "consolidated 2 fragments." in out

    def test_an_empty_message_falls_back_to_the_old_constant(self):
        """A mocked Dreamer returning "" must still ledger something
        recognisable rather than a bare prefix."""
        out = _record("")

        assert out == ("REM cycle ran (memory consolidation / heuristic "
                       "harvest)")


class TestTheCallSiteUsesIt:
    def test_the_constant_is_gone_from_the_dream_ledger_call(self):
        """Source-level: the call site must pass the Dreamer's message, not a
        literal. Pinned because the value was already in scope and discarded —
        the defect was one line of plumbing, and it can regress the same way."""
        from pathlib import Path
        import ghost_agent.core.agent as ag
        src = Path(ag.__file__).read_text()
        seg = src.split("if not _dream_skipped:", 1)[1][:2600]

        # A TOKEN pin, deliberately, and only for the plumbing: the behaviour
        # is executed above. What cannot be reached without driving a whole
        # biological tick is whether the CALL SITE still hands the message
        # over — and that is exactly what regressed, with the value already
        # in scope.
        assert "dream_ledger_summary(_dream_msg)" in seg, (
            "the ledger call must pass the Dreamer's outcome, not a literal")


# ──────────────────────────────────────────────────────────────────────
# Counterfactual replay: "0 generalized" must be readable
# ──────────────────────────────────────────────────────────────────────

class TestCounterfactualReportsWhatItSampled:
    """A "generalized" verdict can ONLY come from a challenge that originally
    FAILED. Live pool: 299 SUCCESS vs 15 FAILURE, and 178 of 185 replays were
    success-origin — so "0 generalized" was ledgered 84 times and means "we
    mostly did not TEST generalization", not "the learning does not
    generalize". Same rule as §4CE: a null result is evidence only when the
    design could have found something."""

    async def test_the_summary_counts_past_failures(self):
        # `async def` + asyncio_mode=auto: `get_event_loop().run_until_
        # complete` picked up a loop an earlier test had closed, so these
        # passed alone and failed in the suite (the known order-contamination
        # shape in this repo).
        from ghost_agent.core import counterfactual as CF

        cands = [{"id": "a", "status": "FAILURE", "challenge": "c",
                  "setup_script": "", "validation_script": "", "cluster": None},
                 {"id": "b", "status": "SUCCESS (in 1 attempts)",
                  "challenge": "c", "setup_script": "",
                  "validation_script": "", "cluster": None}]

        orig_load = CF.load_replay_candidates
        orig_gate = CF.should_replay
        orig_rec = CF.record_result
        CF.load_replay_candidates = lambda limit=2: cands
        CF.should_replay = lambda: (True, "")
        CF.record_result = lambda **k: None

        class _D:
            last_self_play_status = "SUCCESS"

            async def synthetic_self_play(self, **kw):
                return None

        try:
            out = await CF.run_counterfactual_batch(_D(), None, limit=2)
        finally:
            CF.load_replay_candidates = orig_load
            CF.should_replay = orig_gate
            CF.record_result = orig_rec

        assert out["replayed"] == 2
        assert out["past_failures"] == 1

    async def test_an_all_success_batch_reports_zero_past_failures(self):
        """The live shape: nothing sampled that could generalize."""
        from ghost_agent.core import counterfactual as CF

        cands = [{"id": "a", "status": "SUCCESS", "challenge": "c",
                  "setup_script": "", "validation_script": "",
                  "cluster": None}]
        orig = (CF.load_replay_candidates, CF.should_replay, CF.record_result)
        CF.load_replay_candidates = lambda limit=2: cands
        CF.should_replay = lambda: (True, "")
        CF.record_result = lambda **k: None

        class _D:
            last_self_play_status = "SUCCESS"

            async def synthetic_self_play(self, **kw):
                return None

        try:
            out = await CF.run_counterfactual_batch(_D(), None, limit=2)
        finally:
            (CF.load_replay_candidates, CF.should_replay,
             CF.record_result) = orig

        assert out["past_failures"] == 0

    def test_the_ledger_line_shows_the_composition(self):
        """Source pin on the plumbing (the behaviour is covered above): the
        activity line must carry `past_failures`, or "0 generalized" stays
        uninterpretable in the digest."""
        from pathlib import Path
        import ghost_agent.core.agent as ag
        src = Path(ag.__file__).read_text()
        seg = src.split("counterfactual replay: ", 1)[1][:400]

        assert "past_failures" in seg
        assert "past-failure" in seg


class TestTheCountsSurviveTheDigestClamp:
    """The ledger row is written at up to 220 chars, but
    `render_activity_digest` clamps each item to 140 — and the digest is what
    the operator reads between turns. The §LOG lesson is exactly this shape:
    a preview that died at 60 chars, right on the "why".

    ⚠ The fixtures are the REAL live messages, measured from 371 occurrences
    in the operator log. The first version of this test invented wording
    ("consolidated N fragments into M meta-memories") that the Dreamer never
    emits, so it proved the clamp was safe for a string that does not exist.
    The longest real variant is ~155 chars and DOES get clamped — what has to
    survive is the count, which `dream.py` deliberately keeps greppable in
    every variant ("extracted N heuristics")."""

    #: Verbatim shapes from the live log, most common first. The
    #: trajectory-seed variant is shown as CORRECTED: it read
    #: ". Extracted N heuristics" and so was invisible to the lowercase grep
    #: `dream.py` promises — 126 of 371 live messages (34%).
    REAL = [
        "Dream Complete. Synthesized 1 new meta-memories and extracted 0 "
        "heuristics. (RRF weights refit from usefulness ledger)",
        "Dream Complete (trajectory seed) — extracted 2 heuristics. "
        "(1 non-actionable heuristics dropped) (RRF weights refit from "
        "usefulness ledger)",
        "Dream Complete. Synthesized 0 new meta-memories and extracted 2 "
        "heuristics. (1 non-actionable heuristics dropped) (RRF weights "
        "refit from usefulness ledger)",
        "Dream Complete — produced nothing this cycle: no consolidation met "
        "the compression bar and extracted 0 heuristics.",
    ]

    def _digest(self, msg):
        from ghost_agent.core.agent import dream_ledger_summary
        from ghost_agent.core.autonomous_activity import (
            ActivityRecord, render_activity_digest)
        rec = ActivityRecord(ts=0.0, phase="dream",
                             summary=dream_ledger_summary(msg))
        return render_activity_digest([rec])

    @pytest.mark.parametrize("msg", REAL)
    def test_every_real_variant_keeps_its_heuristic_count(self, msg):
        out = self._digest(msg)
        import re
        want = re.search(r"extracted (\d+) heuristics", msg)

        assert want, "fixture must be a real variant"
        assert f"extracted {want.group(1)} heuristics" in out

    def test_the_meta_memory_count_survives_too(self):
        out = self._digest(self.REAL[0])

        assert "Synthesized 1 new meta-memories" in out

    def test_an_empty_cycle_still_reads_as_empty_after_the_clamp(self):
        out = self._digest(self.REAL[3])

        assert "produced nothing" in out

    def test_two_different_real_cycles_do_not_collapse_together(self):
        """`render_activity_digest` collapses IDENTICAL (phase, summary)
        rows into one "×N" line — which is precisely what the old constant
        did to 919 cycles. Different outcomes must stay different rows."""
        from ghost_agent.core.agent import dream_ledger_summary
        from ghost_agent.core.autonomous_activity import (
            ActivityRecord, render_activity_digest)
        recs = [ActivityRecord(ts=0.0, phase="dream",
                               summary=dream_ledger_summary(m))
                for m in (self.REAL[0], self.REAL[2])]
        out = render_activity_digest(recs)

        assert "×2" not in out
        assert out.count("[dream]") == 2 or out.count("REM cycle ran") == 2


class TestTheGreppabilityContractHolds:
    """`dream.py` states: "every variant still contains 'extracted N
    heuristics', because that count is what operators and tests key off."
    It was false for the trajectory-seed variant, which said ". Extracted" —
    34% of live messages were invisible to the documented grep, and the
    existing tests only asserted the lowercase ones. Pinned across ALL
    variants so the contract cannot drift again."""

    def test_every_terminal_message_template_uses_the_lowercase_form(self):
        import re
        from pathlib import Path
        import ghost_agent.core.dream as dm
        src = Path(dm.__file__).read_text()
        # Every f-string that reports a heuristic count in the dream summary.
        templates = re.findall(r'f"[^"]*[Ee]xtracted "?\s*\n?\s*f?"?[^"]*'
                               r'heuristics', src)

        assert templates, "the count templates must be findable"
        for t in templates:
            assert "Extracted" not in t, (
                f"a sentence-initial capital breaks the one-grep contract: {t}")
