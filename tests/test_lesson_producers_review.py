"""Fresh review of the lesson PRODUCERS (dream, self-play, bench, episodes,
distilled, learn_skill) — 2026-10-03. Each test names the world it fails in."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.lesson_quality import is_actionable_lesson
from ghost_agent.memory.skills import SkillMemory


@pytest.mark.parametrize("mistake", ["None observed; the solution was direct and efficient.", "None.",
                                     "N/A - no mistake", "No mistakes were made", "There was no error"])
def test_a_no_mistake_phrasing_needs_an_actionable_fix(mistake):
    """Fails in the world where only the exact words none/n/a counted, so an
    observation ("The solution correctly…") passed as a correction."""
    assert is_actionable_lesson(mistake, "The solution correctly parsed the log in two passes.", "x") is False
    assert is_actionable_lesson(mistake, "When parsing logs, always strip trailing whitespace first.", "x") is True


def test_a_fix_identical_to_the_mistake_is_no_lesson():
    assert is_actionable_lesson("x = a / b", "x = a / b", "Division") is False


@pytest.mark.parametrize("fix", ["Make stdout reference a token from input.txt so the validator passes.",
                                 "Satisfy the checker by printing the expected header first.",
                                 "Read the hidden test's expectations before coding."])
def test_a_rule_about_the_test_harness_is_refused(tmp_path, fix):
    """Fails in the world where self-play rules taught the agent to game
    its own validator."""
    sm = SkillMemory(tmp_path)
    assert sm.learn_lesson("When the output format is unclear", "Guessed the format", fix, source="self_play") is None


def test_a_bench_rule_restating_its_word_problem_is_refused(tmp_path):
    sm = SkillMemory(tmp_path)
    challenge = "Jan is 30 years old now and his son is 16. How old was Jan two years ago?"
    out = sm.learn_lesson("Age word problems involving relative time shifts", "Forgot to shift every age",
                          "If Jan is 30 now, two years ago Jan was 28 and his son was 14.",
                          source="bench", generality_context=challenge)
    assert out is None
    out = sm.learn_lesson("Age word problems involving relative time shifts", "Forgot to shift every age",
                          "Shift every person's age by the same interval before comparing them.",
                          source="bench", generality_context=challenge)
    assert out == "written"


def _archive(tmp_path, trigger, reason):
    with open(tmp_path / "skills_pruned_archive.jsonl", "a") as fh:
        fh.write(json.dumps({"reason": reason, "lesson": {"trigger": trigger}}) + "\n")


def test_a_retracted_lesson_is_not_relearned(tmp_path):
    """Fails in the world where the next dream cycle re-minted a lesson the
    operator had just retracted."""
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "When a final turn is reached, always answer directly", "removed_by_trigger")
    assert sm.learn_lesson("When a final turn is reached, always answer directly", "none",
                           "Always provide a direct answer on the final turn.", source="dream") is None


def test_a_cap_trimmed_lesson_may_come_back(tmp_path):
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "When parsing dates from logs", "playbook_cap_trim")
    assert sm.learn_lesson("When parsing dates from logs", "guessed", "Use ISO 8601.", source="dream") == "written"


def test_a_replaced_fix_brings_its_mistake(tmp_path):
    """Fails in the world where a distilled cluster's new pattern sat next to
    the previous pattern's mistake."""
    sm = SkillMemory(tmp_path)
    t = "distilled(tool/files): paths"
    sm.learn_lesson(t, "Assuming paths exist", "Check the path exists.", trigger=t, source="distilled")
    sm.learn_lesson(t, "Trusting one unverified source", "Always validate critical information against a second source first.",
                    trigger=t, source="distilled")
    row = sm._load_playbook()[0]
    assert row["mistake"] == "Trusting one unverified source" and row["solution"].startswith("Always validate")


@pytest.mark.parametrize("distance,rows", [(0.03, 1), (0.2, 2)])
def test_a_near_identical_twin_from_another_producer_merges(tmp_path, distance, rows):
    """Fails in the world where any cross-producer twin became its own row,
    however close."""
    sm = SkillMemory(tmp_path)
    a = "When a web page returns 403, try an archived copy"
    sm.save_playbook([{"trigger": a, "task": a, "mistake": "m", "solution": "Use archive.org", "source": "reflection"}])
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": a, "text": "", "distance": distance}
    sm.learn_lesson("If a site answers 403, use a cached copy", "m", "Use a cached copy.", MagicMock(), source="dream")
    assert len(sm._load_playbook()) == rows


def test_learn_skill_checks_the_request_in_hand():
    import asyncio
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.memory.lesson_scope import current_request
    skill = MagicMock()
    skill.learn_lesson.return_value = "written"
    bus = MemoryBus(skill_memory=skill)
    tok = current_request.set("count the lines in notes.txt")
    try:
        asyncio.run(bus.publish_fact("learn_skill", {"skill": {"task": "t", "mistake": "m", "solution": "s"}}))
    finally:
        current_request.reset(tok)
    kw = skill.learn_lesson.call_args.kwargs
    assert kw["source"] == "learn_skill" and kw["generality_context"] == "count the lines in notes.txt"


def test_a_full_fallback_never_mentions_the_validator():
    from ghost_agent.core.dream import _patch_with_fallback
    out = _patch_with_fallback({}, outcome="STRUGGLED_THEN_WON", cluster_key="sql", challenge="x",
                               attempt=1, solution_novelty=None)
    assert "validator" not in json.dumps(out).lower() and out["trigger"] and out["correct_pattern"]


def test_harness_lines_never_reach_the_dream_digest():
    """Fails in the world where "SYSTEM ALERT: this is the FINAL turn…" became
    the dream rule "When a final turn is reached…"."""
    from ghost_agent.core.dream import _strip_harness
    tail = ("ran the parser. SYSTEM ALERT: this is the FINAL turn, answer now. The output had 3 rows.\n"
            "[System] retry budget exhausted")
    assert _strip_harness(tail) == "ran the parser. The output had 3 rows."


def test_the_dream_window_cache_survives_a_restart(tmp_path):
    from ghost_agent.core.dream import _load_dream_cache, _save_dream_cache
    ctx = SimpleNamespace(skill_memory=SimpleNamespace(file_path=tmp_path / "skills_playbook.json"))
    _save_dream_cache(ctx, {"auto": frozenset({"a", "b"})})
    assert _load_dream_cache(SimpleNamespace(skill_memory=ctx.skill_memory)) == {"auto": frozenset({"a", "b"})}




def test_a_full_fallback_replaces_a_stray_mistake_too():
    """A leftover LLM anti_pattern next to the template trigger/fix would be
    a mismatched pair."""
    from ghost_agent.core.dream import _patch_with_fallback
    out = _patch_with_fallback({"anti_pattern": "some unrelated mistake"}, outcome="FAILED", cluster_key="sql",
                               challenge="x", attempt=2, solution_novelty=None)
    assert out["anti_pattern"] != "some unrelated mistake"
