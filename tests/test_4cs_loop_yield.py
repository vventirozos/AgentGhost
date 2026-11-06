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

(§4TA, 2026-11-05: the macro engine and its `macros.auto_mined` row are
removed. The axis and its states stay for the loops that are left; the
macro-row tests went with their subject, and a macro registry left on disk
is pinned as NOT read.)
"""

import json
import time

import pytest

from ghost_agent.core import liveness as L
from ghost_agent.core.liveness import (
    BARREN, EMPTY, GATED, NO_SOURCE, UNMEASURED, YIELDING,
    YIELD_PROBES, YieldProbe, YieldResult, _yield_status, render_yield,
    yield_all,
)


@pytest.fixture
def home(tmp_path):
    (tmp_path / "system" / "memory" / "acquired_skills").mkdir(parents=True)
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


# ── the loops whose zero would be a LIE ──────────────────────────────────
class TestUnmeasuredLoops:
    def test_gepa_applies_are_per_process_so_not_claimed(self, home):
        (home / "system" / "optim").mkdir(parents=True)
        _w(home, "system/optim/planning.decompose.json",
           {"optimized_instruction": "do the thing"})
        _w(home, "system/optim/broken.json", {"optimized_instruction": "  "})
        res = L._yield_gepa(home)
        assert res.minted == 2 and res.activated == 1
        assert res.invoked is None
        assert _yield_status(res) == UNMEASURED


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
        assert len(YIELD_PROBES) >= 3, "the probe registry was emptied"   # §4NM: mining.failure_envs left; §4TA: macros; §4TD: lessons
        rows = yield_all(home)["rows"]
        assert len(rows) == len(YIELD_PROBES), "rows were dropped or capped"
        assert {r["name"] for r in rows} == {p.name for p in YIELD_PROBES}
        for row in rows:
            assert row["activated_means"] and row["invoked_means"]
            assert row["source"]
            assert row["activated_means"] != row["invoked_means"]

    def test_the_loops_this_surface_EXISTS_for_are_probed(self, home):
        """A probe registry missing a producing loop is the §4CS defect
        with a telemetry view bolted beside it."""
        names = {p.name for p in YIELD_PROBES}
        # ("evolve.candidates" went with ghost_agent/evolve, §4NM;
        #  "macros.auto_mined" with the macro engine, §4TA)
        for required in ("skills.acquired",
                         "prompts.gepa"):
            assert required in names, f"{required} is not probed"
        assert "macros.auto_mined" not in names

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
        _w(home, "system/memory/acquired_skills/skills_registry.json", {
            "news_headlines": {"status": "active", "usage_count": 5}})
        # a macro registry left from before §4TA: not read, no row
        _w(home, "system/memory/composed_skills/composed_skills.json", {
            "auto_a_b": {"status": "active", "usage_count": 7,
                         "success_count": 7, "last_used": time.time(),
                         "trigger_description":
                             "Auto-discovered recurring sequence (a → b) "
                             "seen in 3 past turns."}})
        out = render_learning_health(home / "system" / "memory")
        assert "LOOP YIELD" in out
        section = out.split("LOOP YIELD", 1)[1]
        # The probe actually ran, against the store we just wrote, and the
        # path arithmetic resolved to THIS home.
        assert "skills.acquired" in section
        assert YIELDING in section
        assert "invoked    5" in section, section[:800]
        assert "macro" not in section.lower() and "invoked    7" not in section

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


# (TestEvolveYield went with the "evolve.candidates" probe — ghost_agent/evolve
# was removed in §4NM.)


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


class TestDerivedZeroIsNotAMeasuredOne:
    def test_a_blocked_loop_is_reported_separately_from_an_ignored_one(
            self, monkeypatch, home):
        """⚠ ROUND 2: BARREN's contract is "measurable, and MEASURED at
        zero", and both live members violated it — the gate's zero is
        arithmetic and the macro loop's is structural (a `proposed` macro
        cannot be run). "Nobody invokes them" sends an operator looking for
        a consumer; the remedy is upstream. (The macro row is gone, §4TA; the
        distinction is driven here with stand-in probes.)"""
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


# (TestGatedIsDecidedByTheLatestRun — the evolve ledger's GATED reading —
# went with ghost_agent/evolve, §4NM.)


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


class TestMachineReadableYield:
    def test_collect_learning_health_carries_the_yield_sets(self, home):
        """⚠ ROUND 2: the axis existed only in the human RENDER, so nothing
        could alarm on it programmatically and `--json` had no yield key —
        the §4CS failure mode fixed only for a human who reads the report."""
        from ghost_agent.core.learning_health import collect_learning_health
        _w(home, "system/memory/acquired_skills/skills_registry.json", {
            "unused_tool": {"status": "active", "usage_count": 0}})
        # a macro registry left from before §4TA: not read, in no set
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
        assert "skills.acquired" in y["barren"]
        assert not [r["name"] for r in y["rows"] if "macro" in r["name"]]


