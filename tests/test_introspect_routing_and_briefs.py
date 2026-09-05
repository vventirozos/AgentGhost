"""Pins for the 2026-09-05 introspect review fixes (journal §4EU).

Six findings, each with the world in which its pin FAILS:

  1. ROUTING — 48 introspective user prompts over two months, 9 reached
     the tool, 28 got no tool, 9 went to system_utility (the machine).
     Pins: the SELF SURFACE rule and the description name the phrases and
     the `overview` action; fail on the pre-fix prompt/description.
  2. ANSWER GIST — the experience record carried no trace of the reply, so
     recall found the question and never the answer. Pin: a query whose
     only match is in the ANSWER finds the record; fails without the gist.
  3. NOISE — boots in `recent`, "without a verdict either way" on 39% of
     rows, repeats rendered twice. Pins fail on the old renderers.
  4. BRIEFS — learning 20 KB / experiments 16 KB into the model's context.
     Pins: default output has no indented detail, verbose has it, a section
     is one block, the learning walk is cached; verdict LABEL is a prefix of
     the verdict TEXT for every branch (one authority, R5 table).
  5. ACTIVITY DIGEST — default leads with notify-severity changes, groups
     the rest by kind; verbose is the ledger.
  6. OVERVIEW — one bounded call composing six surfaces, each of which
     names its own absence.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.autonomous_activity import (
    ActivityLog, ActivityRecord, render_activity_brief,
)
from ghost_agent.core.experiments import (
    _MIN_VERDICT_N, ArmStats, MetricComparison, render_brief_report,
    render_headline,
)
from ghost_agent.core.prompts import SYSTEM_PROMPT
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.autobiographical import (
    AutobiographicalMemory, NO_VERDICT_CLAUSE, answer_gist,
    strip_no_verdict_clause,
)
from ghost_agent.selfhood.schema import Experience
from ghost_agent.tools import introspect as I
from ghost_agent.tools.introspect import tool_introspect
from ghost_agent.tools.registry import TOOL_DEFINITIONS


@pytest.fixture(autouse=True)
def _fresh_learning_cache():
    I._LEARNING_CACHE.clear()
    yield
    I._LEARNING_CACHE.clear()


def _tool(name: str) -> dict:
    return next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == name)


def _self_surface_rule() -> str:
    line = next((l for l in SYSTEM_PROMPT.split("\n")
                 if l.startswith("- SELF SURFACE:")), "")
    assert line, "the SELF SURFACE routing rule is missing from SYSTEM_PROMPT"
    return line


# ─────────────────────────────────────────────────────────── 1. routing

def test_prompt_self_surface_rule_routes_how_are_you_to_introspect():
    rule = _self_surface_rule().lower()
    for phrase in ("how are you", "how are you feeling", "how's things",
                   "while i was away", "tell me about yourself"):
        assert phrase in rule, phrase
    assert "`introspect`" in rule
    assert "`overview`" in rule
    assert "must" in rule


def test_prompt_self_surface_rule_disambiguates_the_machine():
    """9 of 48 live prompts went to system_utility — the rule has to say
    it is the machine, not the self."""
    rule = _self_surface_rule()
    assert "`system_utility`" in rule
    assert "MACHINE" in rule


def test_description_owns_how_are_you_feeling_and_lists_overview():
    fn = _tool("introspect")["function"]
    desc = fn["description"].lower()
    for phrase in ("how are you feeling", "how's things", "while i was away"):
        assert phrase in desc, phrase
    assert "system_utility" in desc
    assert "'overview'" in desc
    assert fn["parameters"]["properties"]["action"]["enum"][0] == "overview"


def test_description_declares_verbose_section_and_hours_for_recent():
    props = _tool("introspect")["function"]["parameters"]["properties"]
    assert props["verbose"]["type"] == "boolean"
    assert "section" in props
    assert "recent" in props["hours"]["description"].lower()


def test_description_does_not_claim_the_prefix_is_live():
    """The module docstring claimed the wake-up prefix was spliced into
    every prompt; it is OFF on the request path. The tool's own text must
    not repeat the claim — the live consequence was 28 no-tool answers."""
    from ghost_agent.core.agent import _SELFHOOD_PREFIX_ENABLED
    assert _SELFHOOD_PREFIX_ENABLED is False
    assert "already spliced" not in (I.__doc__ or "")


# ──────────────────────────────────────────────────────── 2. answer gist

def test_capture_stores_a_redacted_clipped_answer_gist(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    long_reply = ("<think>private reasoning</think>\n\nLeonidas was born on "
                  "12 March 2026, so he is about six months old.  Mail me at "
                  "dad@example.com if that is wrong. " + "x" * 300)
    exp = sm.capture_turn(trajectory_id="t-1",
                          user_request="how old is leonidas?",
                          tool_names=[], outcome="passed",
                          final_response=long_reply)
    assert exp is not None
    g = exp.answer_gist
    assert "private reasoning" not in g
    assert "12 March 2026" in g
    assert "dad@example.com" not in g and "[REDACTED_EMAIL]" in g
    assert len(g) <= 160 and g.endswith("…")
    assert "\n" not in g
    # And it is on disk, not only in memory.
    row = json.loads(sm.autobio.path.read_text().splitlines()[-1])
    assert row["answer_gist"] == g


def test_recall_finds_a_memory_by_its_answer_vocabulary(tmp_path: Path):
    """The world where this fails: rows without a gist. 'birth date' occurs
    only in the ANSWER; the old haystack (summary + request + cluster)
    cannot match it."""
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t-1", user_request="how old is he?",
                    tool_names=[], outcome="passed",
                    final_response="His birthdate is 12 March 2026.")
    sm.capture_turn(trajectory_id="t-2", user_request="weather please",
                    tool_names=["system_utility"], outcome="passed",
                    final_response="Sunny, 28 degrees.")
    hits = sm.recall_relevant("birthdate", limit=5)
    assert [h.trajectory_id for h in hits] == ["t-1"]


async def test_recall_renders_the_answer_line(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t-1", user_request="how old is he?",
                    tool_names=[], outcome="passed",
                    final_response="His birthdate is 12 March 2026.")
    out = await tool_introspect(action="recall", query="birthdate",
                                self_model=sm)
    assert "→ my answer: His birthdate is 12 March 2026." in out


def test_answer_gist_helper_edges():
    assert answer_gist("") == ""
    assert answer_gist(None) == ""
    assert answer_gist("  a \n\n b\t c ") == "a b c"
    assert answer_gist("<THINK>x</THINK>only this") == "only this"
    assert answer_gist("<think>only thinking</think>") == ""
    assert len(answer_gist("y" * 500)) == 160


def test_rows_written_before_the_field_still_load():
    exp = Experience.from_dict({"summary": "old row", "outcome": "passed"})
    assert exp.answer_gist == ""


# ─────────────────────────────────────────────────────────────── 3. noise

async def test_tool_recent_excludes_session_boots(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t-1", user_request="fix the parser",
                    tool_names=["execute"], outcome="passed",
                    final_response="done")
    sm.autobio.mark_session_boot(prior_session_at="2026-09-01T00:00:00Z")
    out = await tool_introspect(action="recent", self_model=sm, limit=5)
    assert "fix the parser" in out
    assert "Session resumed" not in out
    summary = await tool_introspect(action="summary", self_model=sm)
    assert "Session resumed" not in summary


def test_autobio_recent_keeps_boots_unless_told_otherwise(tmp_path: Path):
    """The narrative summariser reads boots on purpose — the substrate
    default is unchanged; only the tool opts out."""
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="turn", user_first_words="a"))
    mem.mark_session_boot(prior_session_at="")
    assert any(e.outcome == "boot" for e in mem.recent(limit=5))
    assert not any(e.outcome == "boot"
                   for e in mem.recent(limit=5, include_boots=False))


def test_autobio_recent_limit_still_binds_under_a_filter(tmp_path: Path):
    """A filter over-scans the tail (limit × 8) — the contract "the most
    recent N" must survive that: the mutant that returns the whole scanned
    tail passed every other pin (mutation batch, §4EU)."""
    mem = AutobiographicalMemory(tmp_path)
    for i in range(6):
        mem.append(Experience(summary=f"turn {i}", user_first_words=f"w{i}"))
    got = mem.recent(limit=2, include_boots=False)
    assert [e.summary for e in got] == ["turn 4", "turn 5"]
    got_h = mem.recent(limit=3, hours=24)
    assert [e.summary for e in got_h] == ["turn 3", "turn 4", "turn 5"]


def test_search_never_returns_boot_markers(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.mark_session_boot(prior_session_at="")
    mem.append(Experience(summary="I resumed the session work.",
                          user_first_words="resume the session"))
    hits = mem.search_my_past("session", limit=5)
    assert hits and all(h.outcome != "boot" for h in hits)


async def test_no_verdict_clause_is_stripped_in_render_but_kept_on_disk(
        tmp_path: Path):
    """Both halves: the reader never sees the clause; the backfill still
    finds it on disk (39% of live rows are waiting for a late verdict)."""
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t-1", user_request="ponder this",
                    tool_names=[], outcome="unknown", final_response="hm")
    out = await tool_introspect(action="recent", self_model=sm)
    assert NO_VERDICT_CLAUSE not in out
    assert 'I worked on "ponder this". I reasoned through it without tools.' in out
    on_disk = sm.autobio.path.read_text()
    assert NO_VERDICT_CLAUSE in on_disk
    assert sm.autobio.update_outcome("t-1", "passed") is True
    assert "and the answer landed" in sm.autobio.path.read_text()


def test_strip_only_touches_a_trailing_clause():
    s = ('I worked on "you said without a verdict either way, right?". '
         'I reasoned through it without tools and the answer landed.')
    assert strip_no_verdict_clause(s) == s
    t = 'I reached for x without a verdict either way.'
    assert strip_no_verdict_clause(t) == "I reached for x."
    assert strip_no_verdict_clause("") == ""
    assert strip_no_verdict_clause("plain") == "plain"


async def test_repeated_requests_collapse_with_a_count(tmp_path: Path):
    """Three identical requests → one line, "(×3)", and the NEWEST member
    represents the group on the chronological surface (its verdict tag
    proves which one was kept — the first two were never judged)."""
    sm = SelfModel(root=tmp_path)
    for i, outcome in enumerate(("unknown", "unknown", "passed")):
        sm.capture_turn(trajectory_id=f"t-{i}",
                        user_request="how old is leonidas now ?",
                        tool_names=[], outcome=outcome, final_response="six months")
    sm.capture_turn(trajectory_id="t-x", user_request="unrelated",
                    tool_names=[], outcome="passed", final_response="ok")
    recent = await tool_introspect(action="recent", self_model=sm, limit=5)
    assert recent.count("how old is leonidas now") == 1
    assert "(×3) [passed]" in recent
    recall = await tool_introspect(action="recall", query="leonidas",
                                   self_model=sm)
    assert recall.count("how old is leonidas now") == 1
    assert "(×3)" in recall


def _append_at(mem: AutobiographicalMemory, when: _dt.datetime, words: str):
    exp = Experience(summary=f"I worked on {words}.", user_first_words=words,
                     timestamp=when.replace(tzinfo=None).isoformat() + "Z")
    mem.append(exp)


async def test_recent_hours_window_excludes_old_rows(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    now = _dt.datetime.now(_dt.timezone.utc)
    _append_at(sm.autobio, now - _dt.timedelta(days=3), "three days ago")
    _append_at(sm.autobio, now - _dt.timedelta(hours=1), "one hour ago")
    sm.autobio.append(Experience(summary="I worked on undated.",
                                 user_first_words="undated", timestamp="garbage"))
    out = await tool_introspect(action="recent", self_model=sm, hours=24)
    assert "one hour ago" in out
    assert "three days ago" not in out
    assert "undated" in out, "unknown age is not old age"
    assert "from the last 24h" in out
    only_old = SelfModel(root=tmp_path / "old")
    _append_at(only_old.autobio, now - _dt.timedelta(days=3), "stale")
    empty = await tool_introspect(action="recent", self_model=only_old, hours=0.25)
    assert "no experiences on file from the last 0.25h" in empty


# ──────────────────────────────────────────────────────────────── 4. briefs

_FAKE_LEARNING = """### LEARNING HEALTH

LESSONS: 3 total (0 graduated)
  outcome ticks on 2 lessons
  ⚠ a warning worth keeping
COMPETENCE: 4 cells
  sql: 74%
CALIBRATION: 10 samples, Brier 0.1
  weights: entropy 0.1
    deeper detail line
"""


def _ctx(tmp_path: Path, **extra):
    md = tmp_path / "system" / "memory"
    md.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(memory_dir=md, args=None, **extra)


async def test_learning_default_is_brief_and_names_sections(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: _FAKE_LEARNING)
    out = await tool_introspect(action="learning", context=_ctx(tmp_path))
    assert "LESSONS: 3 total" in out and "CALIBRATION: 10 samples" in out
    assert "⚠ a warning worth keeping" in out
    assert "outcome ticks" not in out and "sql: 74%" not in out
    assert "sections: LEARNING HEALTH, LESSONS, COMPETENCE, CALIBRATION" in out
    assert len(out) < len(_FAKE_LEARNING) + 300


async def test_learning_verbose_and_section_views(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: _FAKE_LEARNING)
    full = await tool_introspect(action="learning", context=_ctx(tmp_path),
                                 verbose=True)
    assert "sql: 74%" in full and "deeper detail line" in full
    block = await tool_introspect(action="learning", context=_ctx(tmp_path),
                                  section="calibration")
    assert block.startswith("CALIBRATION: 10 samples")
    assert "deeper detail line" in block and "LESSONS" not in block
    missing = await tool_introspect(action="learning", context=_ctx(tmp_path),
                                    section="nope")
    assert "No section named 'nope'" in missing and "CALIBRATION" in missing
    # string booleans from a tool call coerce
    full2 = await tool_introspect(action="learning", context=_ctx(tmp_path),
                                  verbose="true")
    assert "sql: 74%" in full2


def test_learning_report_is_cached_within_ttl(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    calls = []
    monkeypatch.setattr(lh, "render_learning_health",
                        lambda md, a=None: calls.append(1) or _FAKE_LEARNING)
    md = tmp_path / "m"
    text, age = I._learning_report_cached(md, None, now=1000.0)
    assert text == _FAKE_LEARNING and age == 0.0 and len(calls) == 1
    text2, age2 = I._learning_report_cached(md, None, now=1000.0 + 30)
    assert len(calls) == 1 and age2 == 30.0
    I._learning_report_cached(md, None, now=1000.0 + I._LEARNING_CACHE_TTL_S + 1)
    assert len(calls) == 2, "expired cache must re-walk"
    I._learning_report_cached(tmp_path / "other", None, now=1000.0)
    assert len(calls) == 3, "a different memory_dir must not share numbers"


async def test_learning_trailer_says_how_old_cached_numbers_are(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: _FAKE_LEARNING)
    ctx = _ctx(tmp_path)
    first = await tool_introspect(action="learning", context=ctx)
    assert "computed" not in first
    second = await tool_introspect(action="learning", context=ctx)
    assert "learning numbers computed" in second


def _cmp(metric="failure_rate", lower=True, cm=0.2, tm=0.15, cn=100, tn=100,
         diff=-0.05, lo=-0.3, hi=0.2, confound=""):
    return MetricComparison(metric=metric, lower_is_better=lower,
                            control_mean=cm, treatment_mean=tm,
                            control_n=cn, treatment_n=tn, diff=diff,
                            diff_lo=lo, diff_hi=hi, confound=confound)


@pytest.mark.parametrize("cmp_, label, decided", [
    (_cmp(metric="human_label_rate", confound="DESCRIPTIVE"), "NO VERDICT", False),
    (_cmp(diff=None, lo=None, hi=None), "insufficient data", False),
    (_cmp(cn=_MIN_VERDICT_N - 1), f"insufficient data (n<{_MIN_VERDICT_N}/arm)", False),
    (_cmp(), "NO POWER for an improvement", False),
    (_cmp(cm=0.0, tm=0.0, diff=0.0, lo=-0.1, hi=0.1), "no improvement is POSSIBLE", False),
    (_cmp(metric="n_steps", lower=True, cm=3.0, tm=3.1, diff=0.1, lo=-1.0, hi=1.2),
     "no difference detected yet", False),
    (_cmp(diff=-0.1, lo=-0.15, hi=-0.05), "TREATMENT BETTER", True),
    (_cmp(diff=-0.1, lo=-0.15, hi=-0.05, confound="attrition"), "TREATMENT BETTER", True),
    (_cmp(diff=0.1, lo=0.05, hi=0.15), "TREATMENT WORSE", True),
])
def test_verdict_label_is_the_prefix_of_the_verdict_for_every_branch(cmp_, label, decided):
    """R5 table: one comparison, one story on both surfaces."""
    assert cmp_.verdict_label == label
    assert cmp_.verdict.startswith(label), cmp_.verdict
    assert cmp_.decided is decided
    if cmp_.confound:
        assert cmp_.verdict.endswith(f"⚠ {cmp_.confound}")


def _arms(n=40):
    import random
    rnd = random.Random(7)
    c, t = ArmStats(arm="control"), ArmStats(arm="treatment")
    for arm, p in ((c, 0.3), (t, 0.2)):
        for _ in range(n):
            arm.n += 1
            arm.add("failure_rate", 1.0 if rnd.random() < p else 0.0)
            arm.add("n_steps", rnd.randint(1, 6))
            # The descriptive denominator metric — present so the brief's
            # omission of it is a CHOICE the pin can see, not an absence
            # of data (the first version of this fixture never added it,
            # and the mutant that rendered it survived).
            arm.add("human_label_rate", 1.0 if rnd.random() < 0.25 else 0.0)
    return {"control": c, "treatment": t}


def test_brief_report_has_labels_headers_and_no_intervals():
    from ghost_agent.core.experiments import render_report
    arms = {"exp_a": _arms()}
    out = render_brief_report(arms,
                              coverage={"recent_admitted": 10, "recent_stamped": 8},
                              expected_names=["exp_a", "never_stamped"])
    assert "■ exp_a  (n=80)" in out
    assert "CS=" not in out and "control=" not in out
    assert "failure_rate" in out and "→ " in out
    assert "recent stamp coverage: 8/10 (80%)" in out and "⚠ below 90%" in out
    assert "■ never_stamped  (n=0)" in out
    assert "enabled in the registry but NO enrolled turn" in out
    # human_label_rate is descriptive — a line in the FULL report, never
    # in the brief.
    assert "human_label_rate" in render_report(arms)
    assert "human_label_rate" not in out
    # Empty corpus: same prose shape as the full report, no ■ rows.
    empty = render_brief_report({}, expected_names=["ghost"])
    assert "Enabled and waiting for traffic: ghost." in empty
    assert "■" not in empty


def test_headline_reports_enrollment_decisions_and_missing():
    arms = {"control": ArmStats(arm="control"), "treatment": ArmStats(arm="treatment")}
    for arm, v in (("control", 1.0), ("treatment", 0.0)):
        for _ in range(60):
            arms[arm].n += 1
            arms[arm].add("failure_rate", v)
    out = render_headline({"exp_a": arms}, expected_names=["exp_a", "ghost"])
    assert out.startswith("Experiments: 1 live arm(s) enrolled — exp_a n=120")
    assert "DECIDED: exp_a/failure_rate (all turns): TREATMENT BETTER" in out
    assert "⚠ enabled but unstamped: ghost" in out
    assert render_headline({}) == "Experiments: no live experiment has stamped a turn yet."
    unstamped = render_headline({}, expected_names=["ghost"])
    assert unstamped.startswith("Experiments: no live experiment")
    assert "⚠ enabled but unstamped: ghost" in unstamped
    quiet = render_headline({"exp_a": _arms(5)})
    assert "no decided verdict yet" in quiet


async def test_tool_experiments_default_is_brief_verbose_is_full(tmp_path, monkeypatch):
    import ghost_agent.core.experiments as ex
    seen = []

    def _fake(root, **kw):
        seen.append(kw.get("brief"))
        return ("BRIEF TEXT\n■ exp_a  (n=3)\n    failure_rate → x" if kw.get("brief")
                else "FULL TEXT\n■ exp_a  (n=3)\n    failure_rate CS=[a, b] → x\n■ exp_b  (n=1)\n    other")
    monkeypatch.setattr(ex, "report_from_trajectories", _fake)
    ctx = _ctx(tmp_path)
    out = await tool_introspect(action="experiments", context=ctx)
    assert out.startswith("BRIEF TEXT") and "verbose=true for everything" in out
    # No section-name trailer for experiments: the headers already show
    # every name, and a name list would leak bench-scoped names above the
    # bench banner (the DENY pin greps "■ <name>", which this must not add).
    assert "sections:" not in out
    assert True in seen and False in seen
    full = await tool_introspect(action="experiments", context=ctx, verbose=True)
    assert full.startswith("FULL TEXT") and "CS=[a, b]" in full
    block = await tool_introspect(action="experiments", context=ctx, section="exp_b")
    assert block == "■ exp_b  (n=1)\n    other"


# ─────────────────────────────────────────────────────── 5. activity digest

def test_activity_brief_leads_with_notify_then_groups_by_kind():
    now = 1_000_000.0
    recs = [ActivityRecord(ts=now - 60 * i, phase="calibration",
                           summary=f"refit {i}") for i in range(5)]
    recs.append(ActivityRecord(ts=now - 10, phase="scheduled_task",
                               summary="'netmon': host down", severity="notify"))
    recs.append(ActivityRecord(ts=now - 3600 * 30, phase="dream",
                               summary="too old"))
    out = render_activity_brief(recs, hours=24, now=now)
    lines = out.split("\n")
    assert lines[0].startswith("Background activity (last 24h): 6 record(s) across 2 kind(s)")
    assert "What changed" in lines[1]
    assert "! [" in lines[2] and "netmon" in lines[2]
    assert any("[calibration] ×5, newest just now — refit 0" in l for l in lines)
    assert "too old" not in out
    assert render_activity_brief([], hours=24, now=now).startswith("No background activity")


async def test_tool_activity_default_digest_and_verbose_ledger(tmp_path):
    log = ActivityLog(tmp_path / "activity.jsonl")
    for i in range(4):
        log.record("dream", "REM cycle ran")
    log.record("scheduled_task", "'netmon-check': all hosts up", severity="notify")
    ctx = SimpleNamespace(activity_log=log)
    brief = await tool_introspect(action="activity", context=ctx)
    assert "[dream] ×4" in brief and brief.count("REM cycle ran") == 1
    assert "What changed" in brief and "netmon-check" in brief
    assert "verbose=true for the line-by-line ledger" in brief
    ledger = await tool_introspect(action="activity", context=ctx, verbose=True)
    assert "newest first" in ledger and "(×4)" in ledger
    assert "What changed" not in ledger


# ────────────────────────────────────────────────────────────── 6. overview

class _Queue:
    def __init__(self, items):
        self._items = items

    def pending(self):
        return list(self._items)


async def test_overview_composes_every_surface_and_stays_bounded(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: _FAKE_LEARNING)
    sm = SelfModel(root=tmp_path / "self")
    sm.state.set_mood("satisfied", "my last 5 verdict-bearing turns all passed")
    sm.state.note_open_question("Why does the verifier land late?")
    sm.capture_turn(trajectory_id="t-1", user_request="fix the parser",
                    tool_names=["execute"], outcome="passed", final_response="fixed it")
    sm.autobio.mark_session_boot(prior_session_at="")
    log = ActivityLog(tmp_path / "activity.jsonl")
    log.record("dream", "REM cycle ran")
    log.record("scheduled_task", "'netmon': host down", severity="notify")
    ctx = _ctx(tmp_path, activity_log=log,
               defect_queue=_Queue([SimpleNamespace(title="loop never yields")]))
    out = await tool_introspect(action="overview", self_model=sm, context=ctx)
    assert "How I am:" in out
    assert "mood: satisfied — my last 5 verdict-bearing turns all passed" in out
    assert "Why does the verifier land late?" in out
    assert "fix the parser" in out and "Session resumed" not in out
    assert "What changed" in out and "netmon" in out
    assert "LESSONS: 3 total" in out and "⚠ a warning worth keeping" in out
    assert "Experiments:" in out
    assert "Post-mortem defects pending: 1 — worst first: loop never yields" in out
    assert len(out) <= I._OVERVIEW_MAX_CHARS


class _WsActivity:
    def __init__(self, events):
        self._events = events

    def recent(self, limit=10, *, kind=None):
        return list(self._events)[-limit:]


async def test_overview_counts_workspace_events_in_the_window(tmp_path):
    now = _dt.datetime.now(_dt.timezone.utc)

    def _ev(kind, age_h):
        ts = (now - _dt.timedelta(hours=age_h)).replace(tzinfo=None).isoformat() + "Z"
        return SimpleNamespace(kind=kind, timestamp=ts)
    wm = SimpleNamespace(enabled=True, activity=_WsActivity([
        _ev("file_changed", 1), _ev("file_changed", 2), _ev("research", 3),
        _ev("file_changed", 40)]))
    ctx = SimpleNamespace(memory_dir=None, args=None, activity_log=None,
                          workspace_model=wm)
    out = await tool_introspect(action="overview", context=ctx)
    assert "Workspace (24h): 2 file_changed, 1 research" in out
    quiet = SimpleNamespace(enabled=True, activity=_WsActivity([_ev("research", 40)]))
    out2 = await tool_introspect(action="overview", context=SimpleNamespace(
        memory_dir=None, args=None, activity_log=None, workspace_model=quiet))
    assert "Workspace (24h): no recorded events." in out2
    off = await tool_introspect(action="overview", context=SimpleNamespace(
        memory_dir=None, args=None, activity_log=None,
        workspace_model=SimpleNamespace(enabled=False)))
    assert "Workspace" not in off


async def test_overview_names_every_absent_surface(tmp_path):
    ctx = SimpleNamespace(memory_dir=None, args=None, activity_log=None)
    out = await tool_introspect(action="overview", self_model=None, context=ctx)
    assert "selfhood is disabled" in out
    assert "ledger is not attached" in out
    assert "Learning: memory_dir unavailable." in out
    assert "Experiments: memory_dir unavailable." in out
    assert "queue not attached" in out
    disabled = SelfModel(root=tmp_path, enabled=False)
    out2 = await tool_introspect(action="overview", self_model=disabled, context=ctx)
    assert "selfhood is disabled" in out2


async def test_overview_truncates_at_the_cap_and_says_so(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    huge = "LESSONS: x\n" + "\n".join(f"⚠ warning {i} " + "z" * 200 for i in range(40))
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: huge)
    monkeypatch.setattr(I, "_OVERVIEW_LEARNING_MAX_LINES", 40)
    out = await tool_introspect(action="overview", context=_ctx(tmp_path))
    assert len(out) <= I._OVERVIEW_MAX_CHARS + 120
    assert "overview truncated" in out


async def test_overview_surface_failure_is_named_not_swallowed(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh

    def _boom(md, a=None):
        raise RuntimeError("walk exploded")
    monkeypatch.setattr(lh, "render_learning_health", _boom)
    ctx = _ctx(tmp_path, defect_queue=_Queue([]))
    out = await tool_introspect(action="overview", context=ctx)
    assert "Learning: unavailable (RuntimeError: walk exploded)" in out
    assert "Post-mortem defects: none pending." in out


# ────────────────────────────────────────────────────────────── summary

async def test_summary_renders_open_question_and_thread_text(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.state.note_open_question("Why does the verifier land late?")
    sm.state.add_unfinished("the chess release")
    out = await tool_introspect(action="summary", self_model=sm)
    assert "Open questions I'm still carrying:" in out
    assert "Why does the verifier land late?" in out
    assert "Threads I left unfinished:" in out and "the chess release" in out


async def test_stats_renders_mood_evidence(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.state.set_mood("stuck", "3 of my last 5 verdict-bearing turns failed")
    out = await tool_introspect(action="stats", self_model=sm)
    assert "Last noted mood: stuck — 3 of my last 5 verdict-bearing turns failed (" in out


async def test_summary_principles_carry_their_age(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.note_principle("I verify before asserting")
    out = await tool_introspect(action="summary", self_model=sm)
    assert "- I verify before asserting (noted just now)" in out


# ──────────────────────────────────────────── cross-surface (R5) table

async def test_one_experience_renders_identically_on_recent_recall_and_summary(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t-1", user_request="size the toast tables",
                    tool_names=["postgres_admin"], outcome="unknown",
                    final_response="Total is 41 GB including toast and indexes.")
    recent = await tool_introspect(action="recent", self_model=sm)
    recall = await tool_introspect(action="recall", query="toast", self_model=sm)
    summary = await tool_introspect(action="summary", self_model=sm)
    line = ('  - I worked on "size the toast tables". I reached for postgres_admin.'
            ' (just now)\n      → my answer: Total is 41 GB including toast and indexes.')
    for out in (recent, recall, summary):
        assert line in out, out
