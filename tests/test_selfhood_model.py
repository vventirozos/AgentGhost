"""Unit tests for the SelfModel facade."""

from pathlib import Path

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.recognition import PREFIX_OPEN


def test_self_model_disabled_is_full_noop(tmp_path: Path):
    sm = SelfModel(root=tmp_path, enabled=False)
    assert sm.build_wakeup_prefix() == ""
    captured = sm.capture_turn(
        trajectory_id="abc", user_request="hi",
        tool_names=[], outcome="passed", final_response="ok",
    )
    assert captured is None
    stats = sm.stats()
    assert stats == {"enabled": False}


def test_self_model_capture_turn_writes_first_person(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    exp = sm.capture_turn(
        trajectory_id="t-1",
        user_request="count files in /tmp",
        tool_names=["execute"],
        outcome="passed",
        final_response="There are 47 files.",
    )
    assert exp is not None
    assert exp.trajectory_id == "t-1"
    assert exp.subject == "self"
    assert "count files in /tmp" in exp.summary
    assert "execute" in exp.tools_used
    assert "47 files" not in exp.summary  # the summary is request-focused


def test_self_model_wakeup_prefix_after_capture(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="explain memoisation",
        tool_names=[],
        outcome="passed",
        final_response="Cached results.",
    )
    sm.capture_turn(
        trajectory_id="t-2",
        user_request="rate my poem",
        tool_names=[],
        outcome="failed",
        final_response="",
        failure_reason="model timed out",
    )
    prefix = sm.build_wakeup_prefix(recent_experiences_n=3)
    assert PREFIX_OPEN in prefix
    assert "memoisation" in prefix
    assert "poem" in prefix


def test_self_model_capture_touches_session_timestamp(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    assert sm.state.state.last_session_at == ""
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="hi",
        tool_names=[],
        outcome="unknown",
        final_response="hello",
    )
    assert sm.state.state.last_session_at != ""


def test_self_model_stats_reports_counts(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="a thing",
        tool_names=[],
        outcome="passed",
        final_response="ok",
    )
    sm.state.note_open_question("Why?")
    sm.state.add_unfinished("rewrite the parser")
    sm.state.set_mood("focused", "deep in code")

    stats = sm.stats()
    assert stats["enabled"] is True
    assert stats["experience_count"] == 1
    assert stats["open_questions"] == 1
    assert stats["unfinished_threads"] == 1
    assert stats["last_mood"] == "focused"
    assert stats["narrative_present"] is False  # not regenerated yet


async def test_self_model_consolidate_narrative_uses_llm(tmp_path: Path):
    called_with = []

    async def fake_llm(prompt: str) -> str:
        called_with.append(prompt)
        return "I have been doing things. They went well. I am still curious."

    sm = SelfModel(root=tmp_path, narrative_critique_fn=fake_llm)
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="solve a maze",
        tool_names=["execute"],
        outcome="passed",
        final_response="solved",
    )
    out = await sm.consolidate_narrative()
    assert "doing things" in out
    assert len(called_with) == 1
    # The narrative is now reflected in the wake-up prefix
    prefix = sm.build_wakeup_prefix()
    assert "doing things" in prefix


async def test_self_model_consolidate_narrative_disabled(tmp_path: Path):
    sm = SelfModel(root=tmp_path, enabled=False)
    out = await sm.consolidate_narrative()
    assert out == ""


def test_wakeup_prefix_stamps_surfaced_experience_ids(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    assert sm._last_prefix_experience_ids == ()
    e1 = sm.capture_turn(
        trajectory_id="t-1",
        user_request="explain memoisation",
        tool_names=[],
        outcome="passed",
        final_response="ok",
    )
    e2 = sm.capture_turn(
        trajectory_id="t-2",
        user_request="rate my poem",
        tool_names=[],
        outcome="passed",
        final_response="ok",
    )
    sm.build_wakeup_prefix(recent_experiences_n=3)
    assert set(sm._last_prefix_experience_ids) == {e1.id, e2.id}


def test_note_referenced_credits_prefix_surfaced_older_experience(tmp_path: Path):
    # The wake-up prefix IDF-retrieves experiences from the ENTIRE log —
    # exactly the ones likely OLDER than note_referenced_experiences'
    # recent(50) window. They must still be creditable.
    sm = SelfModel(root=tmp_path)
    old = sm.capture_turn(
        trajectory_id="t-old",
        user_request="investigate the trapdoor cipher lattice weakness",
        tool_names=["execute"],
        outcome="passed",
        final_response="done",
    )
    assert old is not None
    # Push it far outside the recent(50) window.
    for i in range(60):
        sm.capture_turn(
            trajectory_id=f"t-{i}",
            user_request=f"unrelated filler chore number {i} with words",
            tool_names=["execute"],
            outcome="passed",
            final_response="ok",
        )
    assert all(e.id != old.id for e in sm.autobio.recent(limit=50))

    prefix = sm.build_wakeup_prefix(query="trapdoor cipher lattice")
    assert old.id in sm._last_prefix_experience_ids
    assert "trapdoor cipher lattice" in prefix

    n = sm.note_referenced_experiences(
        prefix_text=prefix,
        response_text=(
            "Earlier I investigated the trapdoor cipher lattice weakness "
            "— same approach applies here."
        ),
    )
    assert n >= 1
    assert sm.autobio.reference_count(old.id) >= 1


def test_self_model_capture_never_raises(tmp_path: Path, monkeypatch):
    """The capture hook is on the trajectory write path — a failure
    here must NEVER break a turn."""
    sm = SelfModel(root=tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(sm.autobio, "append", boom)
    # Must not raise
    result = sm.capture_turn(
        trajectory_id="t-1",
        user_request="hi",
        tool_names=[],
        outcome="passed",
        final_response="ok",
    )
    assert result is None
