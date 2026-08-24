"""§4CS item B — the LOOP YIELD surface: did the loop produce anything
anyone CONSUMED?

`core/liveness.py`'s existing probes answer "did this mechanism RUN?".
That is a different question, and the gap between them is where the macro
loop hid: dream fired on schedule for six weeks and minted 25 composed
skills that were invoked ZERO times, all time — every liveness probe would
have read FIRED throughout, and nothing anywhere said otherwise.

The load-bearing property is the FIVE states, in particular BARREN vs
UNMEASURED. A view that printed "invoked: 0" for both would say the same
thing about the macro loop (genuinely barren, and fixable) and the
graduated-skill loop (injected into the prompt on every matching turn, but
with no counter anywhere) — opposite problems with opposite remedies.
"""

import json
import time
from pathlib import Path

import pytest

from ghost_agent.core import liveness as L
from ghost_agent.core.liveness import (
    BARREN, EMPTY, GATED, NO_SOURCE, UNMEASURED, YIELDING,
    YIELD_PROBES, YieldProbe, YieldResult, _yield_status, render_yield,
    yield_all,
)


@pytest.fixture
def home(tmp_path):
    (tmp_path / "system" / "memory" / "composed_skills").mkdir(parents=True)
    (tmp_path / "system" / "memory" / "acquired_skills").mkdir(parents=True)
    (tmp_path / "system" / "foresight").mkdir(parents=True)
    (tmp_path / "system" / "evolve").mkdir(parents=True)
    return tmp_path


def _w(home, rel, obj):
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


# ── the five states ──────────────────────────────────────────────────────
class TestStateDerivation:
    def test_no_store_is_NO_SOURCE_not_zero(self):
        assert _yield_status(YieldResult()) == NO_SOURCE

    def test_store_without_a_use_counter_is_UNMEASURED_not_BARREN(self):
        """The expensive mistake is reading an unmeasured loop as dead."""
        assert _yield_status(
            YieldResult(minted=10, activated=10, invoked=None)) == UNMEASURED

    def test_a_measured_zero_is_BARREN(self):
        assert _yield_status(
            YieldResult(minted=25, activated=0, invoked=0)) == BARREN

    def test_a_measured_positive_is_YIELDING(self):
        assert _yield_status(
            YieldResult(minted=5, activated=5, invoked=1)) == YIELDING

    def test_precedence_missing_store_outranks_missing_counter(self):
        """A store that is gone must not be read as "has artifacts, no
        counter" — those lead opposite places. Distinguished from its
        neighbours by holding `invoked=None` FIXED and varying only
        `minted`, so the assertion is about precedence and not a
        restatement of the NO_SOURCE rule."""
        assert _yield_status(YieldResult(minted=None, invoked=None)) == NO_SOURCE
        assert _yield_status(YieldResult(minted=3, invoked=None)) == UNMEASURED

    def test_minted_NOTHING_is_not_the_same_as_artifacts_nobody_invokes(self):
        """REVIEW ROUND 1: BARREN covered `minted == 0`, so a home where
        every store existed and was legitimately empty rendered "5 loop(s)
        produce artifacts NOBODY INVOKES" about five loops that had
        produced nothing — the state of any fresh install."""
        assert _yield_status(YieldResult(minted=0, activated=0,
                                         invoked=0)) == EMPTY
        assert _yield_status(YieldResult(minted=1, activated=0,
                                         invoked=0)) == BARREN

    def test_an_explicit_status_wins(self):
        assert _yield_status(
            YieldResult(minted=1, invoked=0, status=GATED)) == GATED


# ── the macro loop: the case this was built for ──────────────────────────
class TestMacroYield:
    def _store(self, home, entries):
        _w(home, "system/memory/composed_skills/composed_skills.json", entries)

    def test_hand_written_macros_do_not_mask_a_barren_loop(self, home):
        """The live shape on 2026-08-23: 26 macros / 3 invocations, and
        every invocation belonged to the ONE hand-written macro. Counting
        them together reports a working loop."""
        mined = "Auto-discovered recurring sequence (a → b) seen in 3 turns."
        self._store(home, {
            "auto_a_b": {"status": "proposed", "usage_count": 0,
                         "success_count": 0, "last_used": 0.0,
                         "trigger_description": mined},
            "auto_c_d": {"status": "proposed", "usage_count": 0,
                         "success_count": 0, "last_used": 0.0,
                         "trigger_description": mined},
            "youtube_transcribe": {"status": "active", "usage_count": 3,
                                   "success_count": 0, "last_used": 1.0,
                                   "trigger_description": "hand-written"},
        })
        res = L._yield_macros(home)
        assert res.minted == 2, "hand-written macros are not this loop's output"
        assert res.activated == 0
        assert res.invoked == 0
        assert _yield_status(res) == BARREN
        assert "1 hand-written" in res.note

    def test_a_real_invocation_flips_it_to_yielding(self, home):
        self._store(home, {
            "auto_a_b": {"status": "active", "usage_count": 4,
                         "success_count": 3, "last_used": time.time(),
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."},
        })
        res = L._yield_macros(home)
        assert res.invoked == 4 and res.activated == 1
        assert _yield_status(res) == YIELDING
        assert "3/4 invocations succeeded" in res.note
        assert res.last_invoked is not None

    def test_a_missing_store_is_NO_SOURCE(self, home):
        assert _yield_status(L._yield_macros(home)) == NO_SOURCE

    def test_an_EMPTY_store_is_not_the_same_as_a_missing_one(self, home):
        self._store(home, {})
        res = L._yield_macros(home)
        assert res.minted == 0
        assert _yield_status(res) == EMPTY, \
            "an empty store parses — a measured zero, not a gap, and not a "\
            "loop whose artifacts nobody invokes"

    def test_an_operator_DEFINED_macro_named_auto_is_not_loop_output(self, home):
        """`manage_composed_skills(action='define')` accepts any valid
        identifier, `auto_*` included. Booking one as loop output would let
        ITS invocations make a barren loop read YIELDING — a lexical proxy
        for a provenance fact, which is a class this project has paid for
        repeatedly. Provenance is asked of the producer's own
        `trigger_description`."""
        self._store(home, {
            "auto_a_b": {"status": "proposed", "usage_count": 0,
                         "success_count": 0, "last_used": 0.0,
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."},
            "auto_my_helper": {"status": "active", "usage_count": 12,
                               "success_count": 12, "last_used": 1.0,
                               "trigger_description": "my hand-written macro"},
        })
        res = L._yield_macros(home)
        assert res.minted == 1, "only the producer-stamped row is loop output"
        assert res.invoked == 0, "the hand-defined macro's 12 uses must not count"
        assert _yield_status(res) == BARREN
        assert "1 hand-written" in res.note

    def test_both_producers_stamps_are_recognised(self, home):
        """If either producer's wording changes, this loop's output stops
        being counted and the row silently reads EMPTY. Checked against the
        producers' real strings."""
        self._store(home, {
            "auto_x_y": {"status": "proposed", "usage_count": 0,
                         "trigger_description":
                             "Auto-discovered recurring sequence (x → y) "
                             "seen in 4 past turns."},
            "auto_generic_p_q_abc123": {
                "status": "proposed", "usage_count": 0,
                "trigger_description":
                    "Proven 2-step sequence graduated from 5 successful runs"},
        })
        assert L._yield_macros(home).minted == 2

    def test_a_MALFORMED_row_is_reported_not_silently_booked(self, home):
        """`hand = len(data) - len(auto)` booked every corrupt row as
        hand-written, so the note asserted a count it had not measured."""
        self._store(home, {
            "auto_a_b": {"status": "proposed", "usage_count": 0,
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."},
            "broken": None, "also_broken": "corrupt",
            "youtube_transcribe": {"status": "active", "usage_count": 3},
        })
        res = L._yield_macros(home)
        assert res.minted == 1
        # Recomputed from the fixture, not restated: exactly one dict row
        # is neither loop-minted nor malformed. `hand = 1` frozen as a
        # constant survived a coincidental "1 hand-written" assertion.
        assert "1 hand-written" in res.note, res.note
        assert "2 unreadable row(s) skipped" in res.note

    def test_the_hand_written_count_is_MEASURED_not_assumed(self, home):
        mined = "Auto-discovered recurring sequence (a → b) seen in 3 turns."
        self._store(home, {
            "auto_a_b": {"status": "proposed", "usage_count": 0,
                         "trigger_description": mined},
            "h1": {"status": "active", "usage_count": 1},
            "h2": {"status": "active", "usage_count": 1},
            "h3": {"status": "active", "usage_count": 1},
        })
        res = L._yield_macros(home)
        assert res.minted == 1
        assert "3 hand-written" in res.note, res.note

    def test_a_home_with_ONLY_hand_written_macros_reports_EMPTY(self, home):
        self._store(home, {"h1": {"status": "active", "usage_count": 9}})
        res = L._yield_macros(home)
        assert res.minted == 0 and res.invoked == 0
        assert _yield_status(res) == EMPTY


# ── the loops whose zero would be a LIE ──────────────────────────────────
class TestUnmeasuredLoops:
    def test_graduated_skills_never_read_verifications_as_usage(self, home):
        """`verifications` is a PRODUCER-side re-verification counter, and
        mistaking it for usage would turn a silent loop into a
        healthy-looking one.

        ⚠ The `invoked is None` half of this test is RETIRED by §4CT, which
        wired the retrieval counter this row was missing. The row is now a
        MEASURED zero until a real turn surfaces a skill — see
        tests/test_4ct_graduated_skill_retrievals.py.
        """
        _w(home, "system/memory/auto_skills.json", {
            "h1": {"name": "a", "tool_sequence": ["x", "y"], "support": 14,
                   "verifications": 383},
        })
        res = L._yield_graduated_skills(home)
        assert res.minted == 1
        assert res.invoked == 0, "never surfaced — a measured zero"
        assert res.invoked != 383, "verifications are not usage"
        assert _yield_status(res) == BARREN

    def test_gepa_applies_are_per_process_so_not_claimed(self, home):
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/planning.decompose.json",
           {"optimized_instruction": "do the thing"})
        _w(home, "system/optim/broken.json", {"optimized_instruction": "  "})
        res = L._yield_gepa(home)
        assert res.minted == 2 and res.activated == 1
        assert res.invoked is None
        assert _yield_status(res) == UNMEASURED


# ── derived zeros, which ARE claimable ───────────────────────────────────
class TestForesightGateYield:
    def test_zero_enabled_buckets_derives_zero_steers(self, home):
        _w(home, "system/foresight/gate.json",
           {"buckets": {f"b{i}": {} for i in range(63)},
            "enabled_count": 0, "ledger_rows": 754})
        res = L._yield_foresight_gate(home)
        assert res.minted == 63 and res.activated == 0
        assert res.invoked == 0, "an allow-list with nothing allowed steers nothing"
        assert _yield_status(res) == BARREN
        assert "derived, not observed" in res.note

    def test_an_enabled_bucket_with_no_consumer_is_UNMEASURED(self, home):
        _w(home, "system/foresight/gate.json",
           {"buckets": {"b": {"enabled": True}}, "enabled_count": 1})
        res = L._yield_foresight_gate(home)
        assert res.invoked is None
        assert _yield_status(res) == UNMEASURED


class TestLessonYield:
    def test_iso_last_retrieved_is_parsed_not_dropped(self, home):
        """`_parse_ts` exists because these stores write ISO strings and an
        earlier probe threw ValueError on every row, reporting a healthy
        store as silent."""
        _w(home, "system/memory/skills_playbook.json", [
            {"retrievals": 10, "helpful_retrievals": 7,
             "last_retrieved_at": "2026-08-23T20:03:33.754724"},
            {"retrievals": 0, "helpful_retrievals": 0},
        ])
        res = L._yield_lessons(home)
        assert res.minted == 2 and res.activated == 1 and res.invoked == 10
        assert res.last_invoked is not None, "ISO timestamps must parse"
        assert "1 lesson(s) never retrieved" in res.note
        assert _yield_status(res) == YIELDING


# ── the view itself ──────────────────────────────────────────────────────
class TestYieldView:
    def test_a_raising_probe_is_NO_SOURCE_and_never_crashes(self, monkeypatch, home):
        def _boom(_h):
            raise RuntimeError("nope")
        monkeypatch.setattr(
            L, "YIELD_PROBES",
            [YieldProbe("x", "src", _boom, activated_means="a",
                        invoked_means="i")])
        r = yield_all(home)
        assert r["rows"][0]["status"] == NO_SOURCE
        assert "probe raised: RuntimeError" in r["rows"][0]["note"]

    def test_worst_news_sorts_first(self, monkeypatch, home):
        def _mk(st):
            return {NO_SOURCE: YieldResult(),
                    BARREN: YieldResult(minted=1, invoked=0),
                    UNMEASURED: YieldResult(minted=1, invoked=None),
                    YIELDING: YieldResult(minted=1, invoked=2)}[st]
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe("d_yield", "s", lambda _h: _mk(YIELDING),
                       activated_means="a", invoked_means="i"),
            YieldProbe("c_unmeas", "s", lambda _h: _mk(UNMEASURED),
                       activated_means="a", invoked_means="i"),
            YieldProbe("b_barren", "s", lambda _h: _mk(BARREN),
                       activated_means="a", invoked_means="i"),
            YieldProbe("a_nosrc", "s", lambda _h: _mk(NO_SOURCE),
                       activated_means="a", invoked_means="i"),
        ])
        names = [r["name"] for r in yield_all(home)["rows"]]
        assert names == ["a_nosrc", "b_barren", "c_unmeas", "d_yield"]

    def test_the_sort_is_by_STATUS_not_by_name(self, monkeypatch, home):
        """⚠ The test above names its probes alphabetically in status
        order, so `order = {}` — every status tying at the default and the
        rows falling back to a name sort — passed it. Reverse the names
        against the statuses so only a real status ordering can hold."""
        mk = {NO_SOURCE: YieldResult(),
              BARREN: YieldResult(minted=1, invoked=0),
              UNMEASURED: YieldResult(minted=1, invoked=None),
              EMPTY: YieldResult(minted=0, invoked=0),
              GATED: YieldResult(minted=0, invoked=None, status=GATED),
              YIELDING: YieldResult(minted=1, invoked=2)}
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe(n, "s", (lambda st: lambda _h: mk[st])(st),
                       activated_means="a", invoked_means="i")
            for n, st in (("a_yield", YIELDING), ("b_gated", GATED),
                          ("c_empty", EMPTY), ("d_unmeas", UNMEASURED),
                          ("e_barren", BARREN), ("f_nosrc", NO_SOURCE))])
        assert [r["name"] for r in yield_all(home)["rows"]] == [
            "f_nosrc", "e_barren", "d_unmeas", "c_empty", "b_gated",
            "a_yield"]

    def test_barren_and_unmeasured_are_reported_as_DIFFERENT_sets(
            self, monkeypatch, home):
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe("barren_one", "s",
                       lambda _h: YieldResult(minted=9, activated=0, invoked=0),
                       activated_means="a", invoked_means="i"),
            YieldProbe("unmeasured_one", "s",
                       lambda _h: YieldResult(minted=9, activated=9, invoked=None),
                       activated_means="a", invoked_means="i"),
        ])
        r = yield_all(home)
        assert r["barren"] == ["barren_one"]
        assert r["unmeasured"] == ["unmeasured_one"]
        assert r["gaps"] == []
        txt = render_yield(home)
        # The SUMMARY must reflect what was measured — an unconditional
        # "every probed loop has a live consumer" survived otherwise.
        assert "every probed loop has a live consumer" not in txt
        assert "produce artifacts NOBODY INVOKES: barren_one" in txt
        assert "NO durable use counter" in txt and "unmeasured_one" in txt
        # Visually distinguishable, which is the requirement.
        assert "✗ NIL" in txt and "? GAP" in txt
        bl = [l for l in txt.splitlines() if "barren_one" in l][0]
        ul = [l for l in txt.splitlines() if "unmeasured_one" in l][0]
        assert "✗ NIL" in bl and "? GAP" not in bl
        assert "? GAP" in ul and "✗ NIL" not in ul
        # A zero must be PRINTED as a zero, not as an unknown.
        assert "invoked    0" in bl
        assert "invoked    ?" in ul

    def test_every_probe_states_what_its_columns_MEAN(self, home):
        """The columns mean different things per loop; an operator reading
        one scale across all of them would draw wrong conclusions.

        ⚠ REVIEW ROUND 1: this was the ONLY test that read the real
        `YIELD_PROBES`, and it passed over an EMPTY list — the loop body
        simply never ran. That single vacuous loop is why the whole
        feature could be deleted with the suite green. It now asserts the
        registry is populated and that every declared probe appears.
        """
        assert len(YIELD_PROBES) >= 7, "the probe registry was emptied"
        rows = yield_all(home)["rows"]
        assert len(rows) == len(YIELD_PROBES), "rows were dropped or capped"
        assert {r["name"] for r in rows} == {p.name for p in YIELD_PROBES}
        for row in rows:
            assert row["activated_means"] and row["invoked_means"]
            assert row["source"]
            assert row["activated_means"] != row["invoked_means"]

    def test_the_loops_this_surface_EXISTS_for_are_probed(self, home):
        """A probe registry missing the macro loop is the §4CS defect
        with a telemetry view bolted beside it."""
        names = {p.name for p in YIELD_PROBES}
        for required in ("macros.auto_mined", "skills.acquired",
                         "skills.graduated", "lessons.playbook",
                         "prompts.gepa", "foresight.gate",
                         "evolve.candidates"):
            assert required in names, f"{required} is not probed"

    def test_no_row_is_silently_dropped_or_capped(self, monkeypatch, home):
        """The sibling section carried a top-N cut that structurally
        dropped exactly the zero rows worth acting on. `rows[:5]` here
        would hide 2 of 7 live loops."""
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe(f"p{i}", "s",
                       lambda _h, i=i: YieldResult(minted=i, invoked=i),
                       activated_means="a", invoked_means="i")
            for i in range(9)])
        out = yield_all(home)
        assert out["n_probes"] == 9
        assert len(out["rows"]) == 9
        assert len(render_yield(home).splitlines()) >= 9

    def test_the_learning_report_carries_a_REAL_yield_section(self, home):
        """⚠ REVIEW ROUND 1: this asserted `"LOOP YIELD" in out` and
        `"minted" in out.split(...)` — both satisfied by `render_yield`'s
        own HEADER LITERAL. Replacing the entire function body with
        `return "LOOP YIELD minted"` passed it, and so did reintroducing
        the `.parent` path bug this file already carries a comment about
        (it made all 8 sibling probes report NO_SOURCE). Assert on rows
        that only a real run can produce."""
        from ghost_agent.core.learning_health import render_learning_health
        _w(home, "system/memory/composed_skills/composed_skills.json", {
            "auto_a_b": {"status": "active", "usage_count": 5,
                         "success_count": 5, "last_used": time.time(),
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."}})
        out = render_learning_health(home / "system" / "memory")
        assert "LOOP YIELD" in out
        section = out.split("LOOP YIELD", 1)[1]
        # The probe actually ran, against the store we just wrote, and the
        # path arithmetic resolved to THIS home.
        assert "macros.auto_mined" in section
        assert YIELDING in section
        assert "invoked    5" in section, section[:800]

    def test_a_probe_cannot_report_yield_it_did_not_read(self, home):
        """Every probe must derive its numbers from its own store: with no
        store at all, none may claim a count."""
        for row in yield_all(home)["rows"]:
            assert row["status"] in (NO_SOURCE, EMPTY, GATED), (
                f"{row['name']} reported {row['status']} on an empty home")
            if row["status"] == NO_SOURCE:
                assert row["minted"] is None


# ── the two probes that had NO tests at all (review round 1) ─────────────
class TestAcquiredSkillsYield:
    def test_counts_usage_and_marks_the_missing_timestamp(self, home):
        _w(home, "system/memory/acquired_skills/skills_registry.json", {
            "a": {"name": "a", "status": "active", "usage_count": 7},
            "b": {"name": "b", "status": "degraded", "usage_count": 0},
            "c": "corrupt",
        })
        res = L._yield_acquired_skills(home)
        assert res.minted == 2 and res.activated == 1 and res.invoked == 7
        assert res.last_invoked is None
        assert _yield_status(res) == YIELDING

    def test_the_row_reports_the_skills_NOBODY_USES(self, home):
        """⚠ REVIEW ROUND 2: the row aggregated usage across skills, which
        is exactly the masking this whole surface exists to expose — live,
        24 invocations ALL on one of five skills, rendered "yielding" with
        no hint that four were dead. The sibling lesson probe reports
        "N never retrieved"; this one did not."""
        _w(home, "system/memory/acquired_skills/skills_registry.json",
           {"used": {"status": "active", "usage_count": 24},
            "d1": {"status": "active", "usage_count": 0},
            "d2": {"status": "active", "usage_count": 0},
            "d3": {"status": "degraded", "usage_count": 0}})
        res = L._yield_acquired_skills(home)
        assert res.minted == 4 and res.invoked == 24
        assert res.activated == 3, "a degraded skill is not advertised"
        assert "3 of 4 skill(s) have NEVER been used" in res.note
        assert "1 degraded" in res.note
        # ⚠ The earlier note claimed `status` is "a DEGRADATION flag, not
        # an activation gate" and that "activated == minted by
        # construction". Both are false: `status == "active"` gates
        # advertising, dispatch AND embedding, and a skill sits `degraded`
        # until a retirement pass runs.
        assert "DEGRADATION flag" not in res.note

    def test_a_missing_registry_is_NO_SOURCE(self, home):
        assert _yield_status(L._yield_acquired_skills(home)) == NO_SOURCE


class TestEvolveYield:
    def _led(self, home, rows):
        p = home / "system" / "evolve" / "mutations.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in rows))

    def test_runs_that_found_the_loop_OFF_are_GATED_not_barren(self, home):
        """The live shape: every ledger row is
        `{"outcome":"disabled","reason":"GHOST_EVOLVE is off"}` — four
        nightly runs that found the kill switch off. Counting them as
        `minted` reported 4 artifacts nobody invokes while the same report
        said "4 of 4 recorded run(s) found it off" twenty lines below."""
        self._led(home, [{"outcome": "disabled",
                          "reason": "GHOST_EVOLVE is off"}] * 4)
        res = L._yield_evolve(home)
        assert res.minted == 0, "a run that found the loop off minted nothing"
        assert _yield_status(res) == GATED
        assert "switched off" in res.note
        assert home.name and "evolve.candidates" not in yield_all(home)["barren"]

    def test_a_PACKET_is_not_evidence_that_anyone_consumed_it(self, home):
        """REVIEW ROUND 1, and it is this module's own failure mode rebuilt
        inside it: `invoked` was `packets` — the same number as `activated`,
        from the same glob. The row went green the moment a packet nobody
        had opened appeared on disk, which is exactly "dream minted 25 and
        every probe read FIRED"."""
        self._led(home, [{"outcome": "proposed", "node_id": "n1"},
                         {"outcome": "rejected"}])
        pdir = home / "system" / "evolve" / "proposals"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "n1.json").write_text("{}")
        res = L._yield_evolve(home)
        assert res.minted == 1, "only PROPOSED rows are minted candidates"
        assert res.activated == 1
        assert res.invoked is None, \
            "writing a packet is production, not consumption"
        assert _yield_status(res) == UNMEASURED
        assert res.invoked != res.activated

    def test_a_missing_ledger_is_NO_SOURCE(self, home):
        assert _yield_status(L._yield_evolve(home)) == NO_SOURCE


class TestGepaYield:
    def test_the_DURABLE_apply_signal_is_used(self, home):
        """REVIEW ROUND 1: this reported UNMEASURED — "apply counters are
        per-process" — while the loader's own log line sat on disk and a
        sibling probe in the SAME file already counted it, printing a live
        figure 390 lines above this row's "never"."""
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/planning.decompose.json",
           {"optimized_instruction": "do the thing"})
        log = home / "system" / "ghost-agent.log"
        import datetime
        now = datetime.datetime.now()
        log.write_text("\n".join(
            f"{(now - datetime.timedelta(hours=h)).strftime('%Y-%m-%d %H:%M:%S')}"
            f" - GhostStream - INFO - GEPA: loaded tuned instruction"
            for h in (1, 5, 20)))
        res = L._yield_gepa(home)
        assert res.minted == 1 and res.activated == 1
        assert res.invoked == 3, res
        assert res.last_invoked is not None
        assert _yield_status(res) == YIELDING
        assert "LOWER BOUND" in res.note

    def test_an_UNREADABLE_log_claims_no_count(self, home):
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/x.json", {"optimized_instruction": "y"})
        res = L._yield_gepa(home)
        assert res.invoked is None
        assert _yield_status(res) == UNMEASURED


class TestLessonQuarantine:
    def test_quarantined_rows_are_excluded_from_every_column(self, home):
        """Quarantine is RETENTION WITHOUT SERVICE: the row stays visible
        to an operator and `_filter_quarantined` keeps it out of both
        retrieval surfaces, so it can never be retrieved again. Counting
        them made 30% of the live `invoked` figure dead history."""
        _w(home, "system/memory/skills_playbook.json", [
            {"retrievals": 10, "helpful_retrievals": 7,
             "last_retrieved_at": "2026-08-23T20:03:33"},
            {"retrievals": 100, "helpful_retrievals": 90, "quarantined": True,
             "quarantine_reason": "noise", "last_retrieved_at":
                 "2026-08-23T21:00:00"},
        ])
        res = L._yield_lessons(home)
        # ⚠ REVIEW ROUND 2: `minted` was the SERVED count, which made a
        # store whose rows were ALL quarantined render "minted NOTHING
        # yet" — the most alarming state this store can be in shown as its
        # most benign — and made `minted` mean "produced AND still served"
        # here and "produced" in every other probe. Minted is produced.
        assert res.minted == 2, "a quarantined lesson was still minted"
        assert res.activated == 1, "but it can never be retrieved again"
        assert res.invoked == 10, "its 100 retrievals are dead history"
        assert "1 quarantined row(s) excluded" in res.note

    def test_an_ALL_QUARANTINED_store_is_BARREN_not_empty(self, home):
        """The most alarming state this store can be in must not render as
        the most benign. Reachable: this project has a history of mass
        lesson destruction."""
        _w(home, "system/memory/skills_playbook.json", [
            {"retrievals": 40, "helpful_retrievals": 30, "quarantined": True}
            for _ in range(44)])
        res = L._yield_lessons(home)
        assert res.minted == 44 and res.invoked == 0
        assert _yield_status(res) == BARREN
        assert res.derived_zero, "the zero follows from the quarantine"
        row = {r["name"]: r for r in yield_all(home)["rows"]}["lessons.playbook"]
        assert row["name"] in yield_all(home)["blocked"]
        assert row["name"] not in yield_all(home)["empty"]


class TestAgeUnits:
    def test_age_is_reported_in_HOURS(self, monkeypatch, home):
        """A minutes-for-hours slip renders a 5-day-old artifact as
        "2.0h ago" and reads as fresh."""
        six_h = time.time() - 6 * 3600
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe("x", "s",
                       lambda _h: YieldResult(minted=1, activated=1, invoked=1,
                                              last_invoked=six_h),
                       activated_means="a", invoked_means="i")])
        assert yield_all(home)["rows"][0]["age_h"] == pytest.approx(6.0, abs=0.1)


# ── ROUND 2: the fixes to the round-1 fixes ─────────────────────────────
class TestEmptyOutranksUnmeasured:
    def test_a_store_with_NO_artifacts_is_EMPTY_not_UNMEASURED(self, home):
        """⚠ ROUND 2: `minted == 0` sat BELOW the `invoked is None` test, so
        EMPTY was unreachable for every probe whose invocation channel is
        unmeasured. An empty `auto_skills.json` rendered "1 loop(s) have
        artifacts but NO durable use counter" — asserting artifacts for a
        store with none."""
        assert _yield_status(YieldResult(minted=0, invoked=None)) == EMPTY
        _w(home, "system/memory/auto_skills.json", {})
        res = L._yield_graduated_skills(home)
        assert res.minted == 0
        assert _yield_status(res) == EMPTY
        out = yield_all(home)
        assert "skills.graduated" in out["empty"]
        assert "skills.graduated" not in out["unmeasured"]

    def test_an_evolve_loop_that_RUNS_and_mints_nothing_is_not_UNMEASURED(
            self, home):
        p = home / "system" / "evolve" / "mutations.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps({"outcome": "rejected"})
                               for _ in range(40)))
        res = L._yield_evolve(home)
        assert res.minted == 0
        assert "NOT ONE proposed" in res.note
        assert _yield_status(res) == EMPTY


class TestDerivedZeroIsNotAMeasuredOne:
    def test_a_blocked_loop_is_reported_separately_from_an_ignored_one(
            self, monkeypatch, home):
        """⚠ ROUND 2: BARREN's contract is "measurable, and MEASURED at
        zero", and both live members violated it — the gate's zero is
        arithmetic and the macro loop's is structural (a `proposed` macro
        cannot be run). "Nobody invokes them" sends an operator looking for
        a consumer; the remedy is upstream."""
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe("ignored", "s",
                       lambda _h: YieldResult(minted=5, activated=5, invoked=0),
                       activated_means="a", invoked_means="i"),
            YieldProbe("blocked", "s",
                       lambda _h: YieldResult(minted=5, activated=0, invoked=0,
                                              derived_zero="nothing is approved"),
                       activated_means="a", invoked_means="i")])
        out = yield_all(home)
        assert out["barren"] == ["ignored"]
        assert out["blocked"] == ["blocked"]
        txt = render_yield(home)
        assert "NOBODY INVOKES: ignored" in txt
        assert "UPSTREAM blocks them" in txt and "nothing is approved" in txt
        assert "NOBODY INVOKES: blocked" not in txt

    def test_the_live_gate_and_macro_rows_declare_their_zero_derived(
            self, home):
        _w(home, "system/foresight/gate.json",
           {"buckets": {"a": {}}, "enabled_count": 0, "ledger_rows": 10})
        assert L._yield_foresight_gate(home).derived_zero

    def test_every_row_carries_the_derived_flag_field(self, home):
        for row in yield_all(home)["rows"]:
            assert "derived_zero" in row


class TestMarksAreUniform:
    def test_every_status_gets_an_equal_width_mark(self, monkeypatch, home):
        """A 4-char mark shifted the whole row's columns left, and
        NO_SOURCE — which sorts FIRST — had no mark at all."""
        mk = {NO_SOURCE: YieldResult(),
              BARREN: YieldResult(minted=1, invoked=0),
              UNMEASURED: YieldResult(minted=1, invoked=None),
              EMPTY: YieldResult(minted=0, invoked=0),
              GATED: YieldResult(minted=0, invoked=None, status=GATED),
              YIELDING: YieldResult(minted=1, invoked=2)}
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe(f"probe_{st}", "s", (lambda s_: lambda _h: mk[s_])(st),
                       activated_means="a", invoked_means="i")
            for st in mk])
        # Row lines only: skip the header (line 0) and the indented
        # continuation lines, which carry no "minted" column.
        # ROW lines only — the header and the summary lines also carry the
        # word "minted"; a row is the one with all three columns.
        lines = [ln for ln in render_yield(home).splitlines()[1:]
                 if "minted" in ln and "activated" in ln and "invoked" in ln]
        assert len(lines) == len(mk), lines
        cols = [ln.index("minted") for ln in lines]
        assert len(set(cols)) == 1, (
            "every row's columns must line up; a short mark shifts them: "
            + repr([(c, ln[:14]) for c, ln in zip(cols, lines)]))


class TestGepaCountsLoadsNotApplies:
    def _log(self, home, lines):
        import datetime
        now = datetime.datetime.now()
        (home / "system").mkdir(parents=True, exist_ok=True)
        (home / "system" / "ghost-agent.log").write_text("\n".join(
            f"{(now - datetime.timedelta(hours=h)).strftime('%Y-%m-%d %H:%M:%S')}"
            f" - GhostStream - INFO - {t}" for h, t in lines))

    def test_the_PREDATES_THE_GATE_SCHEMA_line_counts_too(self, home):
        """⚠ ROUND 2: `optim/loader.py` logs "loaded tuned instruction" only
        when the artifact carries a `gate_arm`, and logs a WARNING —
        "predates the gate schema" — for one that does not, THEN SERVES IT
        ANYWAY. Twenty such lines are on the live log. A fully-applied
        artifact read BARREN."""
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/verifier.enumerate.json",
           {"optimized_instruction": "x"})
        self._log(home, [
            (1, "GEPA: artifact 'verifier.enumerate' (sha abc12345) predates "
                "the gate schema — no gate identity/scores recorded"),
            (3, "GEPA: artifact 'verifier.enumerate' (sha abc12345) predates "
                "the gate schema — no gate identity/scores recorded"),
        ])
        res = L._yield_gepa(home)
        assert res.invoked == 2, res
        assert _yield_status(res) == YIELDING

    def test_the_count_and_the_age_come_from_the_SAME_window(self, home):
        """⚠ ROUND 2: `count` was windowed at 168h while `_log_probe` takes
        `last_ts` over EVERY match, so an artifact last served 300h ago
        rendered "invoked 0 | last 300.0h ago" — self-contradictory, in the
        loudest state on the view."""
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/a.json", {"optimized_instruction": "x"})
        self._log(home, [(300, "GEPA: loaded tuned instruction for 'a'")])
        res = L._yield_gepa(home)
        assert res.invoked == 1, "an old load is still a load"
        assert res.last_invoked is not None
        assert _yield_status(res) == YIELDING

    def test_the_column_says_LOADS_and_the_note_says_what_that_means(self, home):
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/a.json", {"optimized_instruction": "x"})
        self._log(home, [(1, "GEPA: loaded tuned instruction for 'a'")])
        note = L._yield_gepa(home).note
        assert "LOADS" in note and "NOT applies" in note
        assert "CACHE MISS" in note and "LOWER BOUND" in note
        means = {p.name: p.invoked_means for p in YIELD_PROBES}["prompts.gepa"]
        assert "load" in means.lower() and "appl" not in means.lower()


class TestGatedIsDecidedByTheLatestRun:
    def _led(self, home, outcomes):
        p = home / "system" / "evolve" / "mutations.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps({"outcome": o}) for o in outcomes))

    def test_old_proposals_then_a_recent_gate_off_reads_GATED(self, home):
        """⚠ ROUND 2: the predicate was "are ALL rows disabled?", so a
        ledger with old `proposed` rows and recent `disabled` ones — the
        loop gated off RIGHT NOW — reported UNMEASURED and never mentioned
        `disabled` at all."""
        self._led(home, ["proposed"] * 20 + ["disabled"] * 7)
        res = L._yield_evolve(home)
        assert _yield_status(res) == GATED
        assert "most recent" in res.note and "7 of them" in res.note

    def test_a_recent_run_that_was_NOT_gated_is_not_GATED(self, home):
        self._led(home, ["disabled"] * 7 + ["proposed"])
        res = L._yield_evolve(home)
        assert _yield_status(res) != GATED
        assert res.minted == 1

    def test_the_note_warns_that_a_frozen_ledger_looks_the_same(self, home):
        """A run that TIMES OUT writes no row and the ledger never rotates,
        so an operator who enabled the loop and whose runs all time out
        sees "a deliberate state, not a dead loop" forever."""
        self._led(home, ["disabled"] * 4)
        assert "TIMES OUT" in L._yield_evolve(home).note


class TestMacroProvenanceHasOneDefinition:
    def test_BOTH_producers_build_their_stamp_from_the_shared_constant(self):
        """⚠ ROUND 2: `_MACRO_LOOP_MARKS` held COPIES of the producers'
        strings with nothing linking them. Rewording either producer — a
        pure refactor — made the loop's own output invisible AND made the
        row assert a fabricated provenance fact, with the suite green.
        Asserted on the SOURCE importing the constant, not on a copy of
        its value.
        """
        from ghost_agent.tools import composed_skills as cs
        dream = Path("src/ghost_agent/core/dream.py").read_text()
        agent = Path("src/ghost_agent/core/agent.py").read_text()
        assert "MACRO_MARK_MINED" in dream, \
            "the dream producer must build its stamp from the constant"
        assert "MACRO_MARK_GRADUATED" in agent, \
            "the graduation producer must build its stamp from the constant"
        # ...and the constants must actually be what the READER matches.
        assert L._is_loop_minted_macro(
            "auto_x", {"trigger_description":
                       f"{cs.MACRO_MARK_MINED} (a → b) seen in 3 past turns."})
        assert L._is_loop_minted_macro(
            "auto_y", {"trigger_description":
                       f"Proven 2-step {cs.MACRO_MARK_GRADUATED} 5 runs"})
        # Provenance must NOT be claimed when it cannot be read.
        assert not L._is_loop_minted_macro("auto_z", {"trigger_description": ""})

    def test_the_REAL_dream_producer_output_is_recognised(self, home):
        """An EXECUTED pin: run the actual miner and feed its actual
        description to the actual reader."""
        from ghost_agent.core.dream import mine_recurring_tool_sequences
        from ghost_agent.distill.schema import Trajectory, ToolCall
        seq = [("file_system", {"operation": "replace", "path": "/f",
                                "content": "c"}),
               ("manage_services", {"action": "restart", "name": "s"})]
        trajs = [Trajectory(id=f"t{i}", outcome="passed",
                            tool_calls=[ToolCall(name=n, arguments=a)
                                        for n, a in seq]) for i in range(4)]
        props = mine_recurring_tool_sequences(trajs, min_support=3)
        assert props
        assert L._is_loop_minted_macro(
            props[0]["name"],
            {"trigger_description": props[0]["description"]}), \
            "the miner's real description must be recognised as loop output"

    def test_the_GRADUATION_producer_stamp_is_recognised(self):
        """The agent.py mint builds its description inline, deep in the
        idle tick. Reconstruct it from the SAME constant the source
        interpolates and feed it to the reader — the pin fails if either
        side moves."""
        from ghost_agent.tools import composed_skills as cs
        src = Path("src/ghost_agent/core/agent.py").read_text()
        assert 'f"{MACRO_MARK_GRADUATED} "' in src, \
            "the graduation mint must interpolate the shared constant"
        n, support = 3, 5
        desc = (f"Proven {n}-step {cs.MACRO_MARK_GRADUATED} "
                f"{support} successful runs")
        assert L._is_loop_minted_macro("auto_generic_x_y_abc123",
                                       {"trigger_description": desc})

    def test_provenance_is_NOT_claimed_when_the_marks_are_unreadable(
            self, monkeypatch):
        """A reader that cannot tell provenance must answer "not ours",
        never "ours" — claiming it would book someone else's macro as loop
        output and let ITS invocations make a barren loop read YIELDING."""
        monkeypatch.setattr(
            L, "_macro_marks",
            lambda: (_ for _ in ()).throw(ImportError("gone")))
        assert L._is_loop_minted_macro(
            "auto_x", {"trigger_description":
                       "Auto-discovered recurring sequence (a → b)"}) is False


class TestSummaryLinesAreAllPresent:
    def _probes(self, monkeypatch, **kinds):
        mk = {"empty": YieldResult(minted=0, invoked=0),
              "barren": YieldResult(minted=2, invoked=0),
              "blocked": YieldResult(minted=2, invoked=0,
                                     derived_zero="blocked upstream"),
              "unmeasured": YieldResult(minted=2, invoked=None),
              "gap": YieldResult()}
        monkeypatch.setattr(L, "YIELD_PROBES", [
            YieldProbe(n, "s", (lambda k: lambda _h: mk[k])(k),
                       activated_means="a", invoked_means="i")
            for n, k in kinds.items()])

    def test_an_EMPTY_loop_gets_its_own_summary_line(self, monkeypatch, home):
        self._probes(monkeypatch, only_empty="empty")
        txt = render_yield(home)
        assert "have minted NOTHING yet" in txt and "only_empty" in txt
        assert "NOBODY INVOKES" not in txt

    def test_each_state_produces_exactly_its_own_line(self, monkeypatch, home):
        self._probes(monkeypatch, e="empty", b="barren", k="blocked",
                     u="unmeasured", g="gap")
        txt = render_yield(home)
        for frag in ("have minted NOTHING yet",
                     "produce artifacts NOBODY INVOKES",
                     "UPSTREAM blocks them",
                     "NO durable use counter",
                     "absent or unreadable"):
            assert frag in txt, frag
        assert "every probed loop has a live consumer" not in txt


class TestGateReadsWhatTheConsumerReads:
    def test_a_bucket_flagged_enabled_with_no_COUNT_is_not_called_closed(
            self, home):
        """`gate_allows` reads `buckets[*].enabled`; the probe used to read
        the writer-derived `enabled_count`. A file where they disagree made
        the row claim "NO bucket is enabled" about a gate that says
        otherwise."""
        _w(home, "system/foresight/gate.json", {
            "buckets": {"a": {"enabled": True}, "b": {}},
            "enabled_count": 0, "ledger_rows": 99})
        res = L._yield_foresight_gate(home)
        assert res.activated == 1, "the probe must count enabled BUCKETS"
        assert "NO bucket is enabled" not in (res.note or "")
        assert _yield_status(res) == UNMEASURED

    def test_a_COUNT_with_no_enabled_buckets_does_not_invent_activation(
            self, home):
        _w(home, "system/foresight/gate.json", {
            "buckets": {"a": {}, "b": {}}, "enabled_count": 5,
            "ledger_rows": 99})
        res = L._yield_foresight_gate(home)
        assert res.activated == 0
        assert res.invoked == 0 and res.derived_zero


class TestMachineReadableYield:
    def test_collect_learning_health_carries_the_yield_sets(self, home):
        """⚠ ROUND 2: the axis existed only in the human RENDER, so nothing
        could alarm on it programmatically and `--json` had no yield key —
        the §4CS failure mode fixed only for a human who reads the report."""
        from ghost_agent.core.learning_health import collect_learning_health
        _w(home, "system/memory/composed_skills/composed_skills.json", {
            "auto_a_b": {"status": "proposed", "usage_count": 0,
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."}})
        r = collect_learning_health(home / "system" / "memory")
        y = r["loop_yield"]
        assert set(y) >= {"rows", "barren", "blocked", "unmeasured", "gaps",
                          "empty", "n_probes"}
        assert y["n_probes"] == len(YIELD_PROBES)
        assert "macros.auto_mined" in y["blocked"] + y["barren"]

    def test_an_all_PROPOSED_macro_store_is_BLOCKED_not_ignored(self, home):
        """The live shape: 25 minted, 0 approved, 0 invoked. The zero is
        structural — a proposed macro cannot be run — so the remedy is the
        approval queue, not a missing consumer."""
        _w(home, "system/memory/composed_skills/composed_skills.json", {
            f"auto_{i}": {"status": "proposed", "usage_count": 0,
                          "trigger_description":
                              "Auto-discovered recurring sequence (a → b) "
                              "seen in 3 past turns."} for i in range(25)})
        res = L._yield_macros(home)
        assert res.minted == 25 and res.activated == 0 and res.invoked == 0
        assert res.derived_zero, "an unapproved backlog is a derived zero"
        out = yield_all(home)
        assert "macros.auto_mined" in out["blocked"]
        assert "macros.auto_mined" not in out["barren"]

    def test_an_APPROVED_but_unused_macro_IS_measured_barren(self, home):
        """Once one is approved the zero becomes a real measurement: it
        could have been invoked and was not."""
        _w(home, "system/memory/composed_skills/composed_skills.json", {
            "auto_a_b": {"status": "active", "usage_count": 0,
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."}})
        res = L._yield_macros(home)
        assert res.activated == 1 and not res.derived_zero
        assert "macros.auto_mined" in yield_all(home)["barren"]
