"""Tests for the graduated-skill store (proposal item #9).

Verified, auto-acquired tool sequences are persisted here instead of
being discarded, and surfaced back into the turn prompt.
"""

from pathlib import Path

from ghost_agent.skills_auto import (
    GraduatedSkillStore,
    SkillCandidate,
    verify_candidate,
)


def _candidate(sig="sig-1", cluster="sql", support=4, confidence=0.8):
    return SkillCandidate(
        name=f"auto.{cluster}.x",
        cluster=cluster,
        tool_sequence=("postgres_admin", "execute"),
        support=support,
        confidence=confidence,
        trigger_examples=["optimize this slow database query"],
        signature_hash=sig,
    )


def test_graduate_persists_skill(tmp_path: Path):
    store = GraduatedSkillStore(tmp_path)
    store.graduate(_candidate())
    assert store.count() == 1
    skill = store.all_skills()[0]
    assert skill["tool_sequence"] == ["postgres_admin", "execute"]
    assert skill["cluster"] == "sql"
    assert skill["verifications"] == 1


def test_graduate_is_idempotent_on_signature(tmp_path: Path):
    store = GraduatedSkillStore(tmp_path)
    store.graduate(_candidate(sig="same"))
    store.graduate(_candidate(sig="same"))
    assert store.count() == 1
    assert store.all_skills()[0]["verifications"] == 2


def test_store_survives_reload(tmp_path: Path):
    GraduatedSkillStore(tmp_path).graduate(_candidate(sig="persist-me"))
    reloaded = GraduatedSkillStore(tmp_path)
    assert reloaded.count() == 1


def test_relevant_matches_on_keywords(tmp_path: Path):
    store = GraduatedSkillStore(tmp_path)
    store.graduate(_candidate(sig="sql", cluster="sql"))
    store.graduate(SkillCandidate(
        name="auto.bake", cluster="cooking",
        tool_sequence=("oven", "timer"), support=3, confidence=0.7,
        trigger_examples=["bake a sourdough loaf"], signature_hash="bake",
    ))
    rel = store.relevant("optimize a slow database query", limit=3)
    assert len(rel) == 1
    assert rel[0]["cluster"] == "sql"


def test_format_for_prompt(tmp_path: Path):
    store = GraduatedSkillStore(tmp_path)
    store.graduate(_candidate())
    block = store.format_for_prompt(query="slow database query")
    assert "PROVEN APPROACHES" in block
    assert "postgres_admin → execute" in block


def test_format_for_prompt_empty_when_no_skills(tmp_path: Path):
    store = GraduatedSkillStore(tmp_path)
    assert store.format_for_prompt(query="anything") == ""


def test_store_bounded(tmp_path: Path):
    store = GraduatedSkillStore(tmp_path)
    # 70 distinct skills, store cap is 60.
    for i in range(70):
        store.graduate(SkillCandidate(
            name=f"s{i}", cluster="c", tool_sequence=("a", "b"),
            support=3, confidence=i / 100.0, signature_hash=f"sig{i}",
        ))
    assert store.count() == 60
    # The lowest-confidence skills were dropped.
    confs = [s["confidence"] for s in store.all_skills()]
    assert min(confs) >= 0.10


def test_verify_then_graduate_pipeline(tmp_path: Path):
    """The exact pipeline biological phase 2.6 runs: verify, then
    graduate only the candidates that pass."""
    store = GraduatedSkillStore(tmp_path)
    good = _candidate(sig="good", support=4, confidence=0.8)
    weak = _candidate(sig="weak", support=1, confidence=0.4)

    def verify_fn(c):
        return c.support >= 3 and c.confidence >= 0.5

    vr_good = verify_candidate(good, verify_fn)
    vr_weak = verify_candidate(weak, verify_fn)
    assert vr_good.passed and vr_good.action == "keep"
    assert not vr_weak.passed

    for cand, vr in ((good, vr_good), (weak, vr_weak)):
        if vr.passed and vr.action == "keep":
            store.graduate(cand, confidence=vr.updated_confidence)

    assert store.count() == 1
    assert store.all_skills()[0]["signature_hash"] == "good"
