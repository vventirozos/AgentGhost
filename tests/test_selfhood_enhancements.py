"""Unit tests for the selfhood enhancement bundle.

Covers all the changes in the rated-suggestions pass:
  #1  verdict backfill (autobio update_outcome from facade)
  #3  template-prompt rollup
  #4  utcnow() replacement (smoke check via timestamp shape)
  #5  user_handle propagation through capture_turn
  #6  stale-question gardener (removed §4TB with open questions; its tests went with it)
  #7  meta cluster keyword bin
  #8  narrative blends recent + relevant past
  #9  session-boundary markers
  #10 mood history JSONL
  #11 PII redaction
  #12 IDF cache for search_my_past
  #13 prefix-utility reference counter
  #14 (covered by tests/test_run_selfhood_probes_script.py)

Each test is independent and uses tmp_path so concurrent runs are safe.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import List

import pytest

from ghost_agent.selfhood.autobiographical import (
    AutobiographicalMemory,
    _derive_cluster,
    _template_marker_for,
    detect_referenced_experiences,
    redact_pii,
)
from ghost_agent.selfhood.model import SelfModel
from ghost_agent.selfhood.narrative import (
    NarrativeSummariser,
)
from ghost_agent.selfhood.schema import Experience, _utcnow_iso
from ghost_agent.selfhood.state import SelfStateThread


# ---------------------------------------------------------------------
# #4 — utcnow() replacement
# ---------------------------------------------------------------------

def test_utcnow_iso_uses_z_suffix_and_naive_iso():
    ts = _utcnow_iso()
    # Must round-trip via fromisoformat on the stripped string.
    assert ts.endswith("Z")
    raw = ts.rstrip("Z")
    parsed = datetime.datetime.fromisoformat(raw)
    # The function returns naive UTC text, so the parsed value has no tzinfo.
    assert parsed.tzinfo is None











# ---------------------------------------------------------------------
# #7 — meta cluster
# ---------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "what is the phenomenology of attention",
    "would you call yourself self-aware right now?",
    "describe your own subjective experience",
    "is there something subjective about your inner experience",
])
def test_meta_cluster_matches_introspection_prompts(prompt):
    assert _derive_cluster(prompt) == "meta"


def test_meta_cluster_does_not_eat_coding_prompts():
    # "Function" and "implement" should still resolve to coding,
    # not meta, even though both clusters share zero literal terms.
    assert _derive_cluster("implement a function that parses logs") == "coding"


# ---------------------------------------------------------------------
# #11 — PII redaction
# ---------------------------------------------------------------------

def test_redact_email():
    assert "alice@example.com" not in redact_pii("write to alice@example.com please")


def test_redact_phone():
    out = redact_pii("call me at 555-123-4567")
    assert "5551234567" not in re.sub(r"\D", "", out)
    assert "REDACTED_PHONE" in out


def test_redact_api_key():
    out = redact_pii("token sk_test_abcdef1234567890 leaked")
    assert "sk_test_abcdef1234567890" not in out


def test_redact_passthrough_when_clean():
    text = "summarise the readme"
    assert redact_pii(text) == text


@pytest.mark.parametrize("text", [
    "I went skateboarding yesterday",
    "photograph the skyscraper downtown",
    "it's a skeleton-key metaphor",
])
def test_redact_key_leaves_bare_sk_words_alone(text):
    # Empirical false positive: `sk` as a bare word-prefix ate ordinary
    # words ("skateboarding" → "[REDACTED_KEY] yesterday").
    assert redact_pii(text) == text


def test_redact_key_structured_prefixes_still_caught():
    out = redact_pii("leaked sk-live-abcdef1234567890 in the logs")
    assert "sk-live-abcdef1234567890" not in out
    assert "REDACTED_KEY" in out


def test_redact_key_value_assignment():
    # Empirical miss: the common real leak shape passed straight through
    # (`=` splits the token, so the prefix rule never saw it).
    out = redact_pii("run with api_key=abcd1234efgh please")
    assert "abcd1234efgh" not in out
    assert "REDACTED_KEY" in out


def test_redact_key_colon_assignment():
    out = redact_pii("config had password: hunter2hunter2 in it")
    assert "hunter2hunter2" not in out
    assert "REDACTED_KEY" in out


def test_capture_turn_redacts_user_first_words(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="please email me at vasilis@example.com about it",
        tool_names=[],
        outcome="unknown",
        final_response="ok",
    )
    raw = sm.autobio.path.read_text(encoding="utf-8").strip()
    assert "@example.com" not in raw
    assert "REDACTED_EMAIL" in raw


# ---------------------------------------------------------------------
# #5 — user_handle propagation
# ---------------------------------------------------------------------

def test_capture_turn_stores_user_handle(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="t-2",
        user_request="hello",
        tool_names=[],
        outcome="passed",
        final_response="hi",
        user_handle="Vasilis",
    )
    rec = json.loads(sm.autobio.path.read_text(encoding="utf-8").strip())
    assert rec["user_handle"] == "Vasilis"


def test_capture_turn_handle_blank_when_omitted(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="t-3",
        user_request="hello",
        tool_names=[],
        outcome="passed",
        final_response="hi",
    )
    rec = json.loads(sm.autobio.path.read_text(encoding="utf-8").strip())
    assert rec["user_handle"] == ""


# ---------------------------------------------------------------------
# #1 — verdict backfill via SelfModel.record_outcome
# ---------------------------------------------------------------------

def test_record_outcome_promotes_unknown_to_failed(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="traj-X",
        user_request="do the thing",
        tool_names=["execute"],
        outcome="unknown",
        final_response="",
    )
    line_before = sm.autobio.path.read_text(encoding="utf-8").strip()
    assert "without a verdict either way" in line_before
    changed = sm.record_outcome("traj-X", "failed", failure_reason="boom")
    assert changed is True
    line_after = sm.autobio.path.read_text(encoding="utf-8").strip()
    rec = json.loads(line_after)
    assert rec["outcome"] == "failed"
    assert "without a verdict either way" not in rec["summary"]
    assert "boom" in rec["summary"]


def test_record_outcome_noop_when_unchanged(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="traj-Y",
        user_request="x",
        tool_names=[],
        outcome="passed",
        final_response="y",
    )
    # First promotion: no-op because outcome is already passed.
    assert sm.record_outcome("traj-Y", "passed") is False


# ---------------------------------------------------------------------
# #3 — template-prompt rollup
# ---------------------------------------------------------------------

def test_template_marker_detects_synthetic_exercise():
    assert _template_marker_for(
        "### SYNTHETIC TRAINING EXERCISE Solve this challenge"
    ) == "### SYNTHETIC TRAINING EXERCISE"


def test_template_marker_returns_none_for_user_prompt():
    assert _template_marker_for("write me a haiku") is None


def test_template_rollup_collapses_consecutive_writes(tmp_path: Path):
    sm = SelfModel(tmp_path)
    for i in range(5):
        sm.capture_turn(
            trajectory_id=f"t-{i}",
            user_request="### SYNTHETIC TRAINING EXERCISE Solve this challenge",
            tool_names=["execute"],
            outcome="unknown",
            final_response="",
        )
    lines = sm.autobio.path.read_text(encoding="utf-8").strip().splitlines()
    # 5 inputs collapse into a single rollup record.
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["template_count"] == 5
    assert "5" in rec["summary"]
    assert "synthetic training exercises" in rec["summary"]


def test_template_rollup_separates_different_markers(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="t-a",
        user_request="### SYNTHETIC TRAINING EXERCISE foo",
        tool_names=[], outcome="unknown", final_response="",
    )
    sm.capture_turn(
        trajectory_id="t-b",
        user_request="SYSTEM JUDGE REJECTION: bar",
        tool_names=[], outcome="unknown", final_response="",
    )
    sm.capture_turn(
        trajectory_id="t-c",
        user_request="### SYNTHETIC TRAINING EXERCISE foo again",
        tool_names=[], outcome="unknown", final_response="",
    )
    lines = sm.autobio.path.read_text(encoding="utf-8").strip().splitlines()
    # Synthetic + judge + synthetic = three separate rollup records
    # because the second synthetic can't extend the first rollup once
    # an unrelated entry is between them.
    assert len(lines) == 3
    markers = [json.loads(l)["template_marker"] for l in lines]
    assert markers == [
        "### SYNTHETIC TRAINING EXERCISE",
        "SYSTEM JUDGE REJECTION",
        "### SYNTHETIC TRAINING EXERCISE",
    ]


def test_normal_user_prompt_not_rolled_up(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="hello",
        tool_names=[], outcome="passed", final_response="hi",
    )
    sm.capture_turn(
        trajectory_id="t-2",
        user_request="hello again",
        tool_names=[], outcome="passed", final_response="hi",
    )
    lines = sm.autobio.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------
# #9 — session boundary markers
# ---------------------------------------------------------------------

def test_mark_session_boot_writes_record(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.mark_session_boot()
    raw = sm.autobio.path.read_text(encoding="utf-8").strip()
    rec = json.loads(raw)
    assert rec["outcome"] == "boot"
    assert "resumed" in rec["summary"].lower()
    assert rec["cluster"] == "meta"


def test_mark_session_boot_dedup_within_same_minute(tmp_path: Path):
    sm = SelfModel(tmp_path)
    sm.mark_session_boot()
    sm.mark_session_boot()
    lines = sm.autobio.path.read_text(encoding="utf-8").strip().splitlines()
    # Same-minute restarts are noise — only one boot recorded.
    assert len(lines) == 1


def test_mark_session_boot_disabled(tmp_path: Path):
    sm = SelfModel(tmp_path, enabled=False)
    sm.mark_session_boot()
    # No autobio file should be created at all.
    assert not (tmp_path / "autobiographical.jsonl").exists()


# ---------------------------------------------------------------------
# #10 — mood history JSONL
# ---------------------------------------------------------------------

def _owner_sets(state, label, intensity=2):
    from ghost_agent.selfhood.mood import ANCHOR_DIRECTION
    return state.set_owner_mood(label, ANCHOR_DIRECTION[label], intensity)


def test_mood_history_appends_per_change(tmp_path: Path):
    """§4TC: one row per CHANGE of label or intensity (it was one per set)."""
    state = SelfStateThread(tmp_path)
    _owner_sets(state, "curious")
    _owner_sets(state, "irritable")
    _owner_sets(state, "irritable")            # the same again: no row
    _owner_sets(state, "irritable", 3)         # the intensity changed: a row
    _owner_sets(state, "content")
    hist = state.mood_history(limit=10)
    assert [(m["label"], m["intensity"]) for m in hist] == [
        ("curious", 2), ("irritable", 2), ("irritable", 3), ("content", 2)]


def test_mood_history_empty_when_no_writes(tmp_path: Path):
    state = SelfStateThread(tmp_path)
    assert state.mood_history() == []


def test_mood_history_handles_corrupt_lines(tmp_path: Path):
    state = SelfStateThread(tmp_path)
    _owner_sets(state, "curious")
    # Inject junk after the legitimate write: not JSON, a JSON value that is
    # no row, a row from before §4TC (a label with no pair), then a good row.
    with state.mood_history_path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write("[1, 2]\n")
        f.write(json.dumps({"label": "satisfied", "evidence": "ev2", "set_at": "t", "source": "derived"}) + "\n")
        f.write(json.dumps({"label": "content", "intensity": 1, "p": 0.3, "e": 0.0,
                            "at": "2026-11-06T10:00:00Z", "cause": "c", "kind": "turn"}) + "\n")
    hist = state.mood_history()
    assert [m["label"] for m in hist] == ["curious", "content"]


def test_mood_history_compaction_caps_file(tmp_path: Path):
    from ghost_agent.selfhood.state import (
        MOOD_HISTORY_KEEP_ROWS,
        MOOD_HISTORY_MAX_ROWS,
    )
    state = SelfStateThread(tmp_path)
    # Prefill the journal up to its cap.
    with state.mood_history_path.open("w", encoding="utf-8") as f:
        for i in range(MOOD_HISTORY_MAX_ROWS):
            f.write(json.dumps(
                {"label": "content", "intensity": 1, "p": 0.3, "e": 0.0,
                 "at": "2026-11-05T10:00:00Z", "cause": f"row {i}", "kind": "turn"}
            ) + "\n")

    _owner_sets(state, "tired")                # one more row: over the cap

    lines = [l for l in state.mood_history_path.read_text(
        encoding="utf-8").splitlines() if l]
    assert len(lines) == MOOD_HISTORY_KEEP_ROWS
    # Newest tail kept — the just-written mood is the last record, the
    # oldest prefilled records are gone.
    assert json.loads(lines[-1])["label"] == "tired"
    assert json.loads(lines[0])["cause"] == f"row {MOOD_HISTORY_MAX_ROWS + 1 - MOOD_HISTORY_KEEP_ROWS}"


def test_mood_history_under_the_cap_is_left_whole(tmp_path: Path):
    from ghost_agent.selfhood.state import MOOD_HISTORY_MAX_ROWS
    state = SelfStateThread(tmp_path)
    with state.mood_history_path.open("w", encoding="utf-8") as f:
        for i in range(MOOD_HISTORY_MAX_ROWS - 1):
            f.write(json.dumps({"label": "content", "intensity": 1, "p": 0.3, "e": 0.0,
                                "at": "2026-11-05T10:00:00Z", "cause": f"row {i}", "kind": "turn"}) + "\n")
    _owner_sets(state, "tired")
    lines = [l for l in state.mood_history_path.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == MOOD_HISTORY_MAX_ROWS and json.loads(lines[0])["cause"] == "row 0"


# ---------------------------------------------------------------------
# #12 — IDF cache
# ---------------------------------------------------------------------

def test_search_uses_cached_index(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I parsed nginx logs.", user_first_words="parse logs"))
    mem.append(Experience(summary="I baked a cake.", user_first_words="cake recipe"))
    _ = mem.search_my_past("cake", limit=5)
    # Second query should hit the cache. We can't observe that
    # directly without instrumentation, but we can verify the cache
    # dict is populated.
    assert mem._search_cache  # type: ignore[attr-defined]
    cached_key = next(iter(mem._search_cache))
    assert isinstance(cached_key, tuple) and len(cached_key) == 2


def test_search_cache_invalidates_on_append(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I baked a cake."))
    _ = mem.search_my_past("cake", limit=5)
    first_key = next(iter(mem._search_cache))  # type: ignore[attr-defined]
    # Force mtime / size delta.
    import time
    time.sleep(0.01)
    mem.append(Experience(summary="I parsed nginx logs."))
    _ = mem.search_my_past("nginx", limit=5)
    second_key = next(iter(mem._search_cache))  # type: ignore[attr-defined]
    assert first_key != second_key


# ---------------------------------------------------------------------
# #13 — prefix-utility reference counter
# ---------------------------------------------------------------------

def test_record_reference_persists_count(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(id="exp-1", summary="I solved X."))
    assert mem.record_reference("exp-1") == 1
    assert mem.record_reference("exp-1") == 2
    # Fresh instance reads from disk.
    mem2 = AutobiographicalMemory(tmp_path)
    assert mem2.reference_count("exp-1") == 2


def test_record_reference_empty_id_is_noop(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    assert mem.record_reference("") == 0


def test_detect_referenced_experiences_matches_trigram(tmp_path: Path):
    e = Experience(id="exp-A", summary="I parsed the malformed JSON file")
    matches = detect_referenced_experiences(
        prefix_text="recent: parsed malformed json",
        response_text=(
            "Earlier I parsed the malformed JSON file, and I want to revisit it."
        ),
        experiences=[e],
    )
    assert matches == ["exp-A"]


def test_detect_referenced_experiences_filters_stopword_trigrams():
    e = Experience(id="exp-A", summary="I worked on the thing")
    # The response shares "I worked on" but every token is a stopword
    # after filtering, so no trigram should match.
    matches = detect_referenced_experiences(
        prefix_text="x",
        response_text="I worked on it",
        experiences=[e],
    )
    assert matches == []


# ---------------------------------------------------------------------
# #8 — narrative blends recent + relevant past
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_narrative_blends_relevant_when_state_has_a_fresh_mood(tmp_path: Path):
    """The blend is driven by the state's recall query — the label of the
    mood as it is now (§4TB: open questions fed it too, until nothing could
    write them; §4TC: the label is one of the closed list)."""
    mem = AutobiographicalMemory(tmp_path)
    state = SelfStateThread(tmp_path)
    # Older entry that should resurface via IDF retrieval on the mood label.
    mem.append(Experience(
        summary="I was irritable about trapdoor functions in cryptography.",
        user_first_words="irritable trapdoor",
    ))
    # A wall of recent unrelated entries to push the trapdoor record
    # out of the recency window.
    for i in range(8):
        mem.append(Experience(summary=f"I did unrelated thing {i}."))
    _owner_sets(state, "irritable")        # §4TC: the query is the mood's label, from the closed list

    captured: List[str] = []

    async def fake_critique(prompt: str) -> str:
        captured.append(prompt)
        return "diary"

    nm = NarrativeSummariser(tmp_path, critique_fn=fake_critique,
                              max_recent_experiences=5)
    await nm.regenerate(autobio=mem, state=state)
    assert captured, "critique_fn should have been called"
    # The blended block should mention the older relevant entry.
    assert "older entries that connect" in captured[0]
    assert "I was irritable about trapdoor functions" in captured[0]


@pytest.mark.asyncio
async def test_narrative_no_blend_when_state_empty(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I did a single thing.",
                          user_first_words="x"))
    state = SelfStateThread(tmp_path)

    captured: List[str] = []

    async def fake_critique(prompt: str) -> str:
        captured.append(prompt)
        return "diary"

    nm = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    await nm.regenerate(autobio=mem, state=state)
    # The "older entries" header should not appear when there's no
    # query to drive the blend.
    assert captured
    assert "older entries that connect" not in captured[0]




def test_redact_pii_digit_run_inside_hex_id_survives():
    # 2026-07-20 corpus-scrub finding: digit runs embedded in the 32-hex
    # experience/trajectory ids the diary joins on Luhn-passed and got
    # eaten as card numbers, corrupting the ids. Letter-adjacent runs
    # are identifiers, not cards.
    from ghost_agent.selfhood.autobiographical import redact_pii
    s = "outcome backfilled for 411e4def63394fa28a9070199254740993a"
    assert redact_pii(s) == s


def test_redact_pii_bare_card_still_redacted():
    from ghost_agent.selfhood.autobiographical import redact_pii
    assert "[REDACTED_CC]" in redact_pii("paid with 4111111111111111 today")
