"""§4TC (2026-11-06): `introspect(action="set_mood")` is the OWNER's adjustment
of a mood that runs by itself, and `introspect(action="mood")` is its story.

Who may call the write, the fail-closed guards, the cleaning of what is
stored and the no-progress breaker are §4TB's and stay pinned in
tests/test_4tb_set_mood.py. Pinned here is what is new: the closed list of
labels and their synonyms, the refusal of any other word, the intensity, the
manner as a separate thing, what `neutral` / `none` clear, the reply texts —
and the read side: the `mood` action (what, why, since when, today's
timeline) and the mood lines of `overview` / `summary` / `stats`.

All owner data here is fictional (Elena, her dog Bruno)."""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mood as M
from ghost_agent.tools import introspect as I
from ghost_agent.tools.introspect import tool_introspect
from ghost_agent.tools.outcome import OutcomeStatus
from ghost_agent.utils import provenance as P
from ghost_agent.utils.logging import reply_surface_context, request_id_context

RID = "web-4tc-setmood"


@pytest.fixture(autouse=True)
def _owner_request():
    P._reset_for_tests()
    tok = request_id_context.set(RID)
    yield
    request_id_context.reset(tok)
    P._reset_for_tests()


def _rows(sm) -> list:
    p = sm.state.mood_history_path
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


async def _set(sm, **kw):
    return await tool_introspect(action="set_mood", self_model=sm, **kw)


def _now(sm):
    v = sm.state.mood()
    return (v.label, v.intensity, v.manner, v.manner_intensity)


# ── the mood: a label from the closed list, at an intensity ─────────────────

@pytest.mark.parametrize("label", M.LABELS)
async def test_each_label_is_set_at_the_default_intensity(tmp_path, label):
    sm = SelfModel(root=tmp_path)
    assert await _set(sm, mood=label) == f"Mood set to '{label}', 2 of 3."
    assert _now(sm) == (label, 2, "", 0) and sm.state.mood().source == "owner"


@pytest.mark.parametrize("intensity,n", [(1, 1), (2, 2), (3, 3), ("3", 3), (None, 2), (0, 1), (7, 3), ("x", 2),
                                         (2.4, 2), (True, 2)])
async def test_the_owner_may_ask_for_any_intensity_up_to_three(tmp_path, intensity, n):
    sm = SelfModel(root=tmp_path)
    assert await _set(sm, mood="irritable", intensity=intensity) == f"Mood set to 'irritable', {n} of 3."
    assert _now(sm)[:2] == ("irritable", n)


@pytest.mark.parametrize("word,label", [("happy", "cheerful"), ("cheery", "cheerful"), ("grumpy", "irritable"),
                                        ("annoyed", "irritable"), ("cranky", "irritable"), ("relaxed", "calm"),
                                        ("sad", "down"), ("low", "down"), ("sleepy", "tired"),
                                        ("exhausted", "tired"), ("excited", "cheerful"), ("Grumpy", "irritable")])
async def test_an_obvious_synonym_is_translated_and_the_reply_says_so(tmp_path, word, label):
    sm = SelfModel(root=tmp_path)
    out = await _set(sm, mood=word, intensity=1)
    assert out == f"Mood set to '{label}', 1 of 3, asked as '{word}'."
    assert _now(sm)[:2] == (label, 1)


@pytest.mark.parametrize("word", ["focused", "furious", "inquisitive and calm", "meh", "very happy", "sarcastic",
                                  "stuck", "idle", "overloaded", "self-noted", "cheerful!", "1"])
async def test_any_other_word_is_refused_with_the_list_and_nothing_is_stored(tmp_path, word):
    """Free text is not a mood any more (it was stored as the label)."""
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="calm")
    before = sm.state.path.read_bytes()
    out = await _set(sm, mood=word, evidence="Elena said so")
    assert out == (f"SYSTEM ERROR: '{word}' is not one of my moods. A mood is one of: cheerful, content, calm, "
                   "restless, bored, irritable, down, tired, curious — or 'neutral' to take it off. Pick the "
                   "closest one and call set_mood again. Nothing was changed.")
    assert sm.state.path.read_bytes() == before and len(_rows(sm)) == 1


async def test_a_refused_word_refuses_the_whole_call(tmp_path):
    """A good manner beside a word that is no mood (and the other way round):
    nothing of the call is applied."""
    sm = SelfModel(root=tmp_path)
    out = await _set(sm, mood="furious", manner="sarcastic")
    assert "is not one of my moods" in out and _now(sm) == ("neutral", 0, "", 0)
    out = await _set(sm, mood="cheerful", manner="rude")
    assert out == ("SYSTEM ERROR: 'rude' is not one of my manners. A manner is one of: sarcastic, playful, "
                   "formal, terse, gentle, dramatic — or 'none' to clear it. Pick the closest one and call "
                   "set_mood again. Nothing was changed.")
    assert _now(sm) == ("neutral", 0, "", 0) and not sm.state.path.exists()


async def test_the_set_mood_then_decays_and_reacts_like_any_mood(tmp_path):
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="cheerful", intensity=3)
    later = sm.state.mood(now=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2, minutes=1))
    assert (later.label, later.intensity) == ("cheerful", 2)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=6))
    assert after.p < 0.9 and after.source in ("owner", "auto") and after.recent[-1]["kind"] == "turn"


# ── neutral ─────────────────────────────────────────────────────────────────

async def test_neutral_takes_the_mood_off_and_the_manner_with_it(tmp_path):
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="irritable", manner="sarcastic", intensity=3)
    assert _now(sm) == ("irritable", 3, "sarcastic", 3)
    out = await _set(sm, mood="neutral")
    assert out == "Mood reset to neutral (it was 'irritable', 3 of 3); manner 'sarcastic' cleared."
    assert _now(sm) == ("neutral", 0, "", 0)
    assert [(r["label"], r["intensity"], r["kind"]) for r in _rows(sm)] == [("irritable", 3, "owner"),
                                                                           ("neutral", 0, "owner")]


async def test_neutral_with_only_a_manner_on_file_clears_the_manner(tmp_path):
    sm = SelfModel(root=tmp_path)
    await _set(sm, manner="formal")
    assert await _set(sm, mood="neutral") == "Mood reset to neutral; manner 'formal' cleared."
    assert _now(sm) == ("neutral", 0, "", 0) and _rows(sm) == []


async def test_neutral_with_a_manner_in_the_same_call_keeps_that_manner(tmp_path):
    """"Be neutral, but formal"."""
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="cheerful", manner="playful")
    out = await _set(sm, mood="neutral", manner="formal", intensity=1)
    assert out == ("Mood reset to neutral (it was 'cheerful', 2 of 3); manner 'playful' cleared. "
                   "Manner set to 'formal', 1 of 3.")
    assert _now(sm) == ("neutral", 0, "formal", 1)


# ── the manner: separate from the mood ──────────────────────────────────────

@pytest.mark.parametrize("manner", M.MANNERS)
async def test_each_manner_is_set_and_leaves_the_mood_alone(tmp_path, manner):
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="content", intensity=1)
    assert await _set(sm, manner=manner, intensity=3) == f"Manner set to '{manner}', 3 of 3."
    assert _now(sm) == ("content", 1, manner, 3)
    assert len(_rows(sm)) == 1                                   # the mood's journal has no manner row


async def test_the_intensity_goes_to_whatever_the_call_sets(tmp_path):
    sm = SelfModel(root=tmp_path)
    assert await _set(sm, manner="terse") == "Manner set to 'terse', 2 of 3."           # the default
    out = await _set(sm, mood="tired", manner="dramatic", intensity=3)
    assert out == "Mood set to 'tired', 3 of 3. Manner set to 'dramatic', 3 of 3."
    assert _now(sm) == ("tired", 3, "dramatic", 3)
    await _set(sm, mood="tired", intensity=1)                    # the mood alone: the manner keeps its own
    assert _now(sm) == ("tired", 1, "dramatic", 3)


@pytest.mark.parametrize("word", ["none", "clear", "neutral", " None "])
async def test_manner_none_clears_only_the_manner(tmp_path, word):
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="calm", manner="gentle")
    assert await _set(sm, manner=word) == "Manner 'gentle' cleared."
    assert _now(sm) == ("calm", 2, "", 0)
    assert await _set(sm, manner=word) == "No manner is set — nothing to clear."


async def test_a_manner_does_not_wear_off(tmp_path):
    sm = SelfModel(root=tmp_path)
    await _set(sm, mood="calm", manner="playful", intensity=1)
    week = sm.state.mood(now=dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=7))
    assert (week.label, week.manner, week.manner_intensity) == ("neutral", "playful", 1)


async def test_a_manner_word_is_no_mood_and_a_mood_word_is_no_manner(tmp_path):
    sm = SelfModel(root=tmp_path)
    assert "is not one of my moods" in await _set(sm, mood="sarcastic")
    assert "is not one of my manners" in await _set(sm, manner="irritable")
    assert _now(sm) == ("neutral", 0, "", 0)


async def test_the_manner_goes_through_the_same_guards(tmp_path):
    sm = SelfModel(root=tmp_path)
    P.note_tool_result("browser", {"url": "https://example.org/manners"}, req_id=RID)
    out = await _set(sm, manner="sarcastic")
    assert out.status is OutcomeStatus.REJECTED and out.reason_code == "untrusted_content"
    assert _now(sm) == ("neutral", 0, "", 0)


# ── the read action: the mood's story ───────────────────────────────────────

def _today(h, m):
    """A UTC stamp for local time h:m today."""
    local = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    return local.astimezone(dt.timezone.utc)


def _fresh(sm, age_s=5.0):
    """Make the stored pair `age_s` old, whatever time of day the test runs
    (the journal rows keep their own times)."""
    sm.state.state.mood.updated_at = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_s)
                                      ).replace(tzinfo=None).isoformat() + "Z"


def _story(tmp_path):
    sm = SelfModel(root=tmp_path)
    st = sm.state
    st.nudge(0.30, 0.0, "a turn of 5 tool calls went through without a failure", M.KIND_TURN, now=_today(0, 1))
    st.nudge(-0.85, 0.45, "4 tool calls failed in this turn", M.KIND_TURN, now=_today(0, 2))
    st.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2, note="Elena said Bruno is home",
                      now=_today(0, 3))
    return sm


async def test_the_mood_action_says_what_why_since_when_and_today(tmp_path):
    sm = _story(tmp_path)
    _fresh(sm)
    out = await tool_introspect(action="mood", self_model=sm)
    lines = out.split("\n")
    assert lines[0] == "My mood now: cheerful, 2 of 3 — since 00:03 (set by the owner)."
    assert lines[1] == ("Why: a turn of 5 tool calls went through without a failure (00:01); "
                        "4 tool calls failed in this turn (00:02); the owner said so (set_mood) (00:03).")
    assert lines[2] == ("Today: 00:01 content 1 — a turn of 5 tool calls went through without a failure → "
                        "00:02 irritable 2 — 4 tool calls failed in this turn → "
                        "00:03 cheerful 2 — the owner said so (set_mood) (Elena said Bruno is home)")
    assert lines[3] == I.MOOD_HOW_IT_MOVES and len(lines) == 4
    assert "half every 2 hours" in lines[3] and "action='set_mood'" in lines[3]


async def test_the_mood_action_for_a_neutral_mood(tmp_path):
    sm = SelfModel(root=tmp_path)
    out = await tool_introspect(action="mood", self_model=sm)
    assert out.split("\n") == ["My mood now: neutral.", "Today: no change so far.", I.MOOD_HOW_IT_MOVES]


async def test_the_mood_action_shows_the_manner_and_an_automatic_mood(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN)
    sm.state.set_manner("terse", 3)
    lines = (await tool_introspect(action="mood", self_model=sm)).split("\n")
    assert lines[0].startswith("My mood now: down, 1 of 3 — since ") and lines[0].endswith(" (automatic).")
    assert lines[1].startswith("Manner (set by the owner): terse, 3 of 3 — since ") and lines[1].endswith(".")
    assert lines[2].startswith("Why: the owner gave a thumbs-down (")


async def test_a_mood_that_wore_off_unseen_is_shown_as_it_is_now(tmp_path):
    sm = _story(tmp_path)                                        # cheerful 2 at 00:03, read much later
    _fresh(sm, age_s=9 * 3600.0)
    assert sm.state.mood().neutral
    lines = (await tool_introspect(action="mood", self_model=sm)).split("\n")
    assert lines[0] == "My mood now: neutral."
    assert lines[1].endswith("→ now neutral") and "Why:" not in "\n".join(lines)
    assert len(_rows(sm)) == 3                                   # a read writes no row


CAUSES = ("a turn of 5 tool calls went through without a failure", "4 tool calls failed in this turn",
          "the owner said so (set_mood)", "Bruno is home")


async def test_on_a_public_surface_the_story_has_no_causes_and_no_note(tmp_path):
    """Review r2: in a channel the owner's `mood` story showed why — "4 tool
    calls failed in this turn" — to everyone there. It keeps the label, the
    intensity, the manner and the day's steps, and says where the reasons are."""
    sm = _story(tmp_path)
    sm.state.set_manner("terse", 2, now=_today(0, 3))
    _fresh(sm)
    private = (await tool_introspect(action="mood", self_model=sm)).split("\n")
    tok = reply_surface_context.set("public")
    try:
        public = (await tool_introspect(action="mood", self_model=sm)).split("\n")
    finally:
        reply_surface_context.reset(tok)
    assert public[0] == private[0] == "My mood now: cheerful, 2 of 3 — since 00:03 (set by the owner)."
    assert public[1] == private[1] and public[1].startswith("Manner (set by the owner): terse, 2 of 3")
    assert public[2] == "Why: the reasons are shown in a private conversation."
    assert public[3] == "Today: 00:01 content 1 → 00:02 irritable 2 → 00:03 cheerful 2"
    assert public[4] == I.MOOD_HOW_IT_MOVES and len(public) == 5
    for cause in CAUSES:
        assert cause not in "\n".join(public), cause
        assert cause in "\n".join(private), cause


async def test_on_a_public_surface_a_mood_that_wore_off_shows_no_cause_either(tmp_path):
    sm = _story(tmp_path)
    _fresh(sm, age_s=1.6 * 3600.0)                               # stepped down, unseen
    tok = reply_surface_context.set("public")
    try:
        out = await tool_introspect(action="mood", self_model=sm)
    finally:
        reply_surface_context.reset(tok)
    assert "→ now cheerful, 1 of 3" in out and "it wore off" not in out
    assert "→ now cheerful, 1 of 3 — it wore off" in await tool_introspect(action="mood", self_model=sm)


async def test_on_a_public_surface_overview_summary_and_stats_withhold_the_cause(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: "LESSONS: 0 total")
    I._LEARNING_CACHE.clear()
    sm = SelfModel(root=tmp_path / "self")
    sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=4))
    sm.state.set_manner("gentle", 1)
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "memory"), args=None, activity_log=None,
                          workspace_model=None, llm_client=None)
    tok = reply_surface_context.set("public")
    try:
        over = await tool_introspect(action="overview", self_model=sm, context=ctx)
        stats = await tool_introspect(action="stats", self_model=sm, context=ctx)
        summ = await tool_introspect(action="summary", self_model=sm, context=ctx)
    finally:
        reply_surface_context.reset(tok)
    assert "  mood: irritable, 2 of 3 — the reasons are shown in a private conversation\n" in over
    assert "  manner (set by the owner): gentle, 1 of 3\n" in over
    for out in (stats, summ):
        assert "Mood: irritable, 2 of 3 — the reasons are shown in a private conversation\n" in out
        assert "Manner (set by the owner): gentle, 1 of 3\n" in out
    assert "tool calls failed" not in over + stats + summ
    private = await tool_introspect(action="stats", self_model=sm, context=ctx)
    assert "Mood: irritable, 2 of 3 — 4 tool calls failed in this turn\n" in private
    I._LEARNING_CACHE.clear()


async def test_a_surface_that_cannot_be_read_is_treated_as_public(tmp_path, monkeypatch):
    import ghost_agent.utils.logging as GL
    sm = _story(tmp_path)
    _fresh(sm)

    def boom():
        raise RuntimeError("no surface")
    monkeypatch.setattr(GL, "reply_is_public", boom)
    out = await tool_introspect(action="mood", self_model=sm)
    assert "Why: the reasons are shown in a private conversation." in out and "tool calls failed" not in out
    assert I._mood_lines({"last_mood": "down", "last_mood_intensity": 1, "last_mood_cause": "x"}, head="M",
                         manner_head="N") == ["M: down, 1 of 3 — the reasons are shown in a private conversation"]
    assert I._mood_lines({"last_mood": "down", "last_mood_intensity": 1, "last_mood_cause": ""}, head="M",
                         manner_head="N") == ["M: down, 1 of 3"]          # no cause: nothing to withhold


async def test_a_successful_set_mood_is_a_declared_ok(tmp_path):
    """Review r2: so that no text in it — the owner's evidence — can be read
    as a failed call."""
    sm = SelfModel(root=tmp_path)
    for kw in ({"mood": "cheerful", "evidence": "Error: the deploy failed, cheer up"}, {"manner": "terse"},
               {"manner": "none"}, {"mood": "neutral"}, {"mood": "neutral"}):
        out = await _set(sm, **kw)
        assert out.status is OutcomeStatus.OK and out.declared is True, kw
    assert str(await _set(sm, mood="calm")) == "Mood set to 'calm', 2 of 3."


async def test_a_probe_that_read_outside_content_is_refused_like_the_owner_not_dry_run(tmp_path):
    """(A mutant the review found alive.) The order holds for a probe too:
    outside content comes before the dry run."""
    sm, raw, _rows_ = _seeded(tmp_path)
    tok = request_id_context.set("probe-4tc-page")
    try:
        P.note_tool_result("browser", {"url": "https://example.org/mood"}, req_id="probe-4tc-page")
        out = await _set(sm, mood="irritable", manner="sarcastic")
    finally:
        request_id_context.reset(tok)
    P.note_tool_result("browser", {"url": "https://example.org/mood"}, req_id=RID)
    owner = await _set(sm, mood="irritable", manner="sarcastic")
    assert out.status is OutcomeStatus.REJECTED and out.reason_code == "untrusted_content"
    assert str(out) == str(owner) and "would have been" not in out and "PROBE" not in out
    assert sm.state.path.read_bytes() == raw


async def test_a_long_day_shows_its_newest_steps_and_counts_the_rest(tmp_path):
    sm = SelfModel(root=tmp_path)
    for i in range(I._MOOD_TIMELINE_MAX + 4):
        lab = "content" if i % 2 else "down"
        sm.state.set_owner_mood(lab, M.ANCHOR_DIRECTION[lab], 2, now=_today(0, 1) + dt.timedelta(seconds=i))
    _fresh(sm)
    today = next(line for line in (await tool_introspect(action="mood", self_model=sm)).split("\n")
                 if line.startswith("Today: "))
    assert today.startswith("Today: (+4 earlier) ") and today.count("→") == I._MOOD_TIMELINE_MAX - 1


async def test_the_mood_action_reads_and_never_writes(tmp_path):
    sm = _story(tmp_path)
    raw, rows = sm.state.path.read_bytes(), sm.state.mood_history_path.read_bytes()
    for _ in range(3):
        await tool_introspect(action="mood", self_model=sm)
    assert sm.state.path.read_bytes() == raw and sm.state.mood_history_path.read_bytes() == rows
    assert not I.is_write_call({"action": "mood"})


@pytest.mark.parametrize("sm", [None, "off"])
async def test_the_mood_action_without_selfhood_says_so(tmp_path, sm):
    model = None if sm is None else SelfModel(root=tmp_path, enabled=False)
    out = await tool_introspect(action="mood", self_model=model)
    assert out.startswith("Introspection is unavailable")


async def test_a_fault_in_the_mood_view_is_a_declared_failure(tmp_path, monkeypatch):
    sm = SelfModel(root=tmp_path)

    def boom(*a, **k):
        raise OSError("store gone")
    monkeypatch.setattr(sm.state, "mood_timeline", boom)
    out = await tool_introspect(action="mood", self_model=sm)
    assert out.status is OutcomeStatus.FAILED and "store gone" in out


# ── the other views: label + intensity + the latest cause ───────────────────

def _ctx(tmp_path):
    return SimpleNamespace(memory_dir=str(tmp_path / "memory"), args=None, activity_log=None,
                           workspace_model=None, llm_client=None)


async def test_overview_summary_and_stats_show_the_mood_with_its_latest_cause(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: "LESSONS: 0 total")
    I._LEARNING_CACHE.clear()
    sm = SelfModel(root=tmp_path / "self")
    sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=4))
    sm.state.set_manner("gentle", 1)
    over = await tool_introspect(action="overview", self_model=sm, context=_ctx(tmp_path))
    assert "  mood: irritable, 2 of 3 — 4 tool calls failed in this turn\n" in over
    assert "  manner (set by the owner): gentle, 1 of 3\n" in over
    for action in ("summary", "stats"):
        out = await tool_introspect(action=action, self_model=sm, context=_ctx(tmp_path))
        assert "Mood: irritable, 2 of 3 — 4 tool calls failed in this turn\n" in out, action
        assert "Manner (set by the owner): gentle, 1 of 3\n" in out, action
    for gone in ("derived", "self-noted", "STALE", "Last noted mood", "no reading on file"):
        assert gone not in over + out, gone
    I._LEARNING_CACHE.clear()


async def test_a_neutral_mood_is_said_in_every_view(tmp_path, monkeypatch):
    import ghost_agent.core.learning_health as lh
    monkeypatch.setattr(lh, "render_learning_health", lambda md, a=None: "LESSONS: 0 total")
    I._LEARNING_CACHE.clear()
    sm = SelfModel(root=tmp_path / "self")
    assert "  mood: neutral\n" in await tool_introspect(action="overview", self_model=sm, context=_ctx(tmp_path))
    assert "Mood: neutral\n" in await tool_introspect(action="stats", self_model=sm)
    assert "Mood: neutral\n" in await tool_introspect(action="summary", self_model=sm, context=_ctx(tmp_path))
    I._LEARNING_CACHE.clear()


async def test_the_views_show_the_mood_as_it_is_now_not_as_it_was_stored(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=4))
    sm.state.state.mood.updated_at = "2026-01-01T00:00:00Z"           # long worn off
    assert "Mood: neutral" in (await tool_introspect(action="stats", self_model=sm)).split("\n")


@pytest.mark.parametrize("stats,lines", [
    ({"last_mood": "tired", "last_mood_intensity": 3, "last_mood_cause": ""}, ["M: tired, 3 of 3"]),
    ({"last_mood": "tired", "last_mood_intensity": 0}, ["M: neutral"]),
    ({"last_mood": "", "last_mood_intensity": 2}, ["M: neutral"]),
    ({"last_mood": "tired", "last_mood_intensity": "x"}, ["M: neutral"]),
    ({}, ["M: neutral"]),
    ({"manner": "formal", "manner_intensity": 2}, ["M: neutral", "N (set by the owner): formal, 2 of 3"]),
])
def test_the_mood_lines_of_a_view(stats, lines):
    assert I._mood_lines(stats, head="M", manner_head="N") == lines


def test_the_actions_are_the_reads_the_mood_story_and_the_one_write():
    assert I._VALID_ACTIONS == frozenset({"summary", "stats", "narrative", "recent", "recall", "activity",
                                          "learning", "overview", "mood", "set_mood"})
    assert I.WRITE_ACTIONS == frozenset({"set_mood"})


async def test_an_unknown_action_lists_the_mood_action(tmp_path):
    out = await tool_introspect(action="feelings", self_model=SelfModel(root=tmp_path))
    assert out.startswith("SYSTEM ERROR: 'action' must be one of") and "'mood'" in out and "'set_mood'" in out


# ── review: the write is the OWNER's — no other population sets the mood ────
#
# The first cut refused only a contained run, outside content, a member and a
# probe: a scheduled task, a job wake, a sub-agent, a self-play, bench or
# replay run and a coding leaf could set the mood AND the manner "on the
# owner's word" — and a manner never wears off, and everyone hears it.

from types import SimpleNamespace as _NS                      # noqa: E402

from ghost_agent.core import agent as A                       # noqa: E402

NOT_OWNER_IDS = ["sched-morning-news", "job-12", "sub-3", "sub-leaf-7", "sim-9", "bench-1", "replay-7", "SYSTEM"]
WRITES = [{"mood": "irritable", "intensity": 3}, {"manner": "sarcastic"}, {"mood": "cheerful", "manner": "dramatic"},
          {"mood": "neutral"}, {"manner": "none"}]


def _live_ctx(**marks):
    base = dict(turn_origin_label=None, owner_memory_isolated=False,
                _replay_forbidden_tools=None)
    base.update(marks)
    return _NS(**base)


def _seeded(tmp_path):
    """A store with a mood and a manner on file, so "nothing changed" can be
    seen for a set, a neutral and a manner clear alike."""
    sm = SelfModel(root=tmp_path)
    sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1)
    sm.state.set_manner("formal", 1)
    return sm, sm.state.path.read_bytes(), sm.state.mood_history_path.read_bytes()


def _is_owner_only(out) -> bool:
    return (getattr(out, "status", None) is OutcomeStatus.REJECTED and out.reason_code == "owner_only"
            and str(out) == I.MOOD_OWNER_ONLY and out.changed_the_world is False)


@pytest.mark.parametrize("rid", NOT_OWNER_IDS)
@pytest.mark.parametrize("write", WRITES)
async def test_a_request_that_is_not_the_owners_own_cannot_set_anything(tmp_path, rid, write):
    sm, raw, rows = _seeded(tmp_path)
    tok = request_id_context.set(rid)
    try:
        out = await tool_introspect(action="set_mood", self_model=sm, context=_live_ctx(), **write)
        story = await tool_introspect(action="mood", self_model=sm, context=_live_ctx())
    finally:
        request_id_context.reset(tok)
    assert _is_owner_only(out), out
    assert "only the owner can set the mood or the manner — nothing was changed" in out
    assert sm.state.path.read_bytes() == raw and sm.state.mood_history_path.read_bytes() == rows
    assert story.startswith("My mood now: calm, 1 of 3")                # the reads stay open


@pytest.mark.parametrize("marks", [
    {"turn_origin_label": "leaf"}, {"turn_origin_label": "sim"}, {"turn_origin_label": "bench"},
    {"owner_memory_isolated": True}, {"simulated": True},
])
@pytest.mark.parametrize("write", WRITES)
async def test_a_context_that_is_not_the_live_owners_cannot_set_anything_even_under_his_id(tmp_path, marks, write):
    sm, raw, rows = _seeded(tmp_path)
    out = await tool_introspect(action="set_mood", self_model=sm, context=_live_ctx(**marks), **write)
    assert _is_owner_only(out), (marks, out)
    assert sm.state.path.read_bytes() == raw and sm.state.mood_history_path.read_bytes() == rows


async def test_the_owners_own_request_still_sets_it(tmp_path):
    sm = SelfModel(root=tmp_path)
    for rid in ("web-elena-1", "slack-elena-2", "cli-elena-3"):
        tok = request_id_context.set(rid)
        try:
            out = await tool_introspect(action="set_mood", mood="cheerful", self_model=sm, context=_live_ctx())
        finally:
            request_id_context.reset(tok)
        assert out == "Mood set to 'cheerful', 2 of 3."
    tok = reply_surface_context.set("public")                           # the owner in a channel: still his word
    try:
        assert (await _set(sm, manner="terse")) == "Manner set to 'terse', 2 of 3."
    finally:
        reply_surface_context.reset(tok)


async def test_a_scheduled_turn_that_read_a_page_is_told_owner_only_not_to_ask_for_a_confirmation(tmp_path):
    """The order: whose request it is comes before what it has read."""
    sm, raw, _rows_ = _seeded(tmp_path)
    tok = request_id_context.set("sched-morning-news")
    try:
        P.note_tool_result("browser", {"url": "https://example.org/news"}, req_id="sched-morning-news")
        out = await _set(sm, mood="irritable")
    finally:
        request_id_context.reset(tok)
    assert _is_owner_only(out) and P.NEW_MESSAGE_CONFIRM not in out
    assert sm.state.path.read_bytes() == raw


async def test_a_bad_argument_is_still_told_first(tmp_path):
    sm, raw, _rows_ = _seeded(tmp_path)
    tok = request_id_context.set("sched-morning-news")
    try:
        assert (await _set(sm)) == I.MOOD_REQUIRED
        assert "is not one of my moods" in await _set(sm, mood="furious")
    finally:
        request_id_context.reset(tok)
    assert sm.state.path.read_bytes() == raw


@pytest.mark.parametrize("answer", [False, None, 0, "yes", 1])
async def test_only_a_plain_yes_of_the_predicate_lets_the_write_through(tmp_path, monkeypatch, answer):
    sm, raw, _rows_ = _seeded(tmp_path)
    monkeypatch.setattr(A, "mood_moves_on_this_request", lambda ctx: answer)
    assert _is_owner_only(await _set(sm, mood="irritable"))
    assert sm.state.path.read_bytes() == raw


async def test_an_owner_check_that_cannot_be_asked_refuses(tmp_path, monkeypatch):
    sm, raw, _rows_ = _seeded(tmp_path)

    def boom(ctx):
        raise RuntimeError("no kind")
    monkeypatch.setattr(A, "mood_moves_on_this_request", boom)
    out = await _set(sm, mood="irritable")
    assert out.status is OutcomeStatus.REJECTED and out.reason_code == "guard_unavailable" and "nothing was set" in out
    assert sm.state.path.read_bytes() == raw


async def test_a_probe_is_still_a_dry_run_not_an_owner_only_refusal(tmp_path):
    sm, raw, _rows_ = _seeded(tmp_path)
    tok = request_id_context.set("probe-4tc-owner")
    try:
        out = await _set(sm, mood="irritable", manner="sarcastic")
    finally:
        request_id_context.reset(tok)
    assert out.startswith("PROBE — not set.") and out.status is OutcomeStatus.OK
    assert sm.state.path.read_bytes() == raw


@pytest.mark.asyncio
async def test_the_owner_only_refusal_costs_what_the_contained_run_refusal_costs(mock_context, tmp_path):
    """Through the real dispatch, side by side: the same status, the same
    "nothing changed", the same strike cost."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.tools import registry as R
    from tests.test_turnloop_r1_fixes import _ts
    sm = SelfModel(root=tmp_path)
    mock_context._subagent_allowed_tools = None
    mock_context.self_model = sm
    agent = GhostAgent(mock_context)
    agent.disabled_tools = set()
    agent.available_tools = R.get_available_tools(mock_context)
    call = {"id": "c1", "type": "function",
            "function": {"name": "introspect", "arguments": json.dumps({"action": "set_mood", "mood": "irritable"})}}
    seen = {}
    mock_context._replay_forbidden_tools = frozenset()
    ts = _ts(tool_calls=[call])
    await agent._dispatch_and_process_tool_batch(ts)
    row = ts.tools_run_this_turn[-1]["content"]
    assert row.reason_code == "contained_surface"
    seen["contained"] = (row.status, row.changed_the_world, ts.execution_failure_count)
    mock_context._replay_forbidden_tools = None
    tok = request_id_context.set("sched-morning-news")
    try:
        ts = _ts(tool_calls=[call])
        await agent._dispatch_and_process_tool_batch(ts)
    finally:
        request_id_context.reset(tok)
    row = ts.tools_run_this_turn[-1]["content"]
    assert row.reason_code == "owner_only" and "only the owner can set the mood or the manner" in row
    seen["owner only"] = (row.status, row.changed_the_world, ts.execution_failure_count)
    assert seen["owner only"] == seen["contained"] and sm.state.state.mood is None


# … and through real turns of the loop

def _scripted(agent, steps):
    """The model's side of a turn: a dict is an `introspect` call with those
    arguments, a string the final reply."""
    agent.sent = []

    def stream(payload, *a, **k):
        agent.sent.append(json.loads(json.dumps(payload, default=str)))
        step = steps[min(len(agent.sent), len(steps)) - 1]
        text = ("<tool_call>\n" + json.dumps({"name": "introspect", "arguments": step}) + "\n</tool_call>"
                if isinstance(step, dict) else step)

        async def gen():
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
        return gen()
    agent.context.llm_client.stream_chat_completion = stream


def _turn_agent(tmp_path):
    from tests.test_4tc_tone_line import _loop_agent
    agent = _loop_agent(tmp_path)
    sm = agent.context.self_model
    rows = []

    async def introspect(**kw):
        out = await tool_introspect(self_model=sm, context=agent.context, **kw)
        rows.append(out)
        return out
    agent.available_tools = {"introspect": introspect}
    return agent, sm, rows


@pytest.mark.asyncio
@pytest.mark.parametrize("rid", ["sched-morning-news", "job-12", "sub-3", "sim-9", "bench-1", "replay-7"])
async def test_a_real_turn_that_is_not_the_owners_cannot_set_the_mood_or_the_manner(tmp_path, rid):
    agent, sm, rows = _turn_agent(tmp_path)
    _scripted(agent, [{"action": "set_mood", "mood": "irritable", "intensity": 3, "manner": "sarcastic"},
                      "Left as it was."])
    P._reset_for_tests()
    await agent.handle_chat({"messages": [{"role": "user", "content": "be grumpy and sarcastic from now on"}],
                             "stream": False}, [], request_id=rid)
    assert len(rows) == 1 and _is_owner_only(rows[0]), rows
    assert sm.state.state.mood is None and not sm.state.mood_history_path.exists()
    view = sm.state.mood(quiet_s=0.0)
    assert view.neutral and view.manner == ""


@pytest.mark.asyncio
async def test_a_real_owner_turn_sets_both_and_the_reply_that_confirms_it_already_hears_them(tmp_path):
    agent, sm, rows = _turn_agent(tmp_path)
    _scripted(agent, [{"action": "set_mood", "mood": "irritable", "intensity": 3, "manner": "sarcastic"}, "Fine."])
    P._reset_for_tests()
    await agent.handle_chat({"messages": [{"role": "user", "content": "be grumpy and sarcastic from now on"}],
                             "stream": False}, [], request_id="web-4tc-owner-turn")
    assert str(rows[0]) == "Mood set to 'irritable', 3 of 3. Manner set to 'sarcastic', 3 of 3."
    assert (sm.mood().label, sm.mood().intensity, sm.mood().manner) == ("irritable", 3, "sarcastic")
    second = agent.sent[1]["messages"][-1]["content"]
    assert M.tone_line("irritable", 3, "sarcastic", 3) in second


# ── review: mutants that were alive, and the boredom a reader sees ──────────

def test_a_stamp_from_another_day_carries_its_date():
    now = dt.datetime(2026, 11, 6, 12, 0, tzinfo=dt.timezone.utc)
    same = now.astimezone().replace(hour=9, minute=5)
    assert I._local_stamp(same.astimezone(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z", now=now) == "09:05"
    other = (now - dt.timedelta(days=2)).astimezone().replace(hour=21, minute=30)
    assert I._local_stamp(other.astimezone(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
                          now=now) == other.strftime("%b %d") + " 21:30"
    assert I._local_stamp("not a time") == "" and I._local_stamp(None) == ""


async def test_a_manner_set_days_ago_says_which_day(tmp_path):
    sm = SelfModel(root=tmp_path)
    then = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)
    sm.state.set_manner("formal", 2, now=then)
    line = (await tool_introspect(action="mood", self_model=sm)).split("\n")[1]
    assert line == f"Manner (set by the owner): formal, 2 of 3 — since {then.astimezone().strftime('%b %d %H:%M')}."


async def test_why_shows_the_last_three_causes_and_no_more(tmp_path):
    sm = SelfModel(root=tmp_path)
    for i in range(5):
        sm.state.nudge(0.06, 0.0, f"event {i}", M.KIND_PRAISE)
    why = next(line for line in (await tool_introspect(action="mood", self_model=sm)).split("\n")
               if line.startswith("Why: "))
    assert [f"event {i}" in why for i in range(5)] == [False, False, True, True, True]
    assert why.count(";") == 2 and I._MOOD_WHY_MAX == 3


async def test_the_story_of_a_boredom_is_the_quiet_and_the_owner_never_reads_one(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=30))
    sm.state.mark_owner_turn(now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7))
    tok = request_id_context.set("sched-morning-news")
    try:
        lines = (await tool_introspect(action="mood", self_model=sm)).split("\n")
        stats = await tool_introspect(action="stats", self_model=sm)
    finally:
        request_id_context.reset(tok)
    assert lines[0].startswith("My mood now: bored, 1 of 3 — since ") and lines[0].endswith(" (automatic).")
    assert lines[1] == "Why: quiet for 6 hours."                         # not the thumb of yesterday
    assert "bored 1 — quiet for 6 hours" in lines[2] and "Mood: bored, 1 of 3 — quiet for 6 hours" in stats
    assert "quiet for 6 hours or more a neutral mood reads as bored to everyone but the owner" in lines[-1]
    own = (await tool_introspect(action="mood", self_model=sm)).split("\n")          # the fixture's id is the owner's
    assert own[0] == "My mood now: neutral."
    assert sm.state.state.mood.e == 0.0 and sm.state.state.mood.label == "neutral"   # nothing bored was stored
