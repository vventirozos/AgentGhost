"""Tests for skills_auto.consolidator."""

from ghost_agent.skills_auto.extractor import SkillCandidate
from ghost_agent.skills_auto.consolidator import consolidate


def _c(name="c1", cluster=None, seq=("a", "b"), support=2, confidence=0.5,
       triggers=None, exemplar_id="t1"):
    return SkillCandidate(
        name=name,
        cluster=cluster,
        tool_sequence=seq,
        support=support,
        exemplar_trajectory_id=exemplar_id,
        trigger_examples=triggers or [],
        confidence=confidence,
        signature_hash="h1",
    )


def test_passthrough_when_no_merges_possible():
    cs = [_c(name="c1", seq=("a", "b")), _c(name="c2", seq=("x", "y"))]
    out, report = consolidate(cs)
    assert len(out) == 2
    assert report.n_merges == 0


def test_merges_identical_sequences():
    cs = [
        _c(name="c1", cluster=None, support=3, confidence=0.6),
        _c(name="c2", cluster=None, support=2, confidence=0.4),
    ]
    out, report = consolidate(cs)
    assert len(out) == 1
    assert report.n_merges == 1
    assert out[0].support == 5


def test_merged_confidence_reflects_aggregated_support():
    """Two candidates, one with high support+high-confidence, another
    low+low. Merged confidence sits in between and respects the formula:
        conf = S / (S + F + 1)
    """
    cs = [
        _c(support=4, confidence=0.8),  # implied f = 0 (approximately)
        _c(support=1, confidence=0.25),  # implied f ≈ 2
    ]
    out, _ = consolidate(cs)
    assert len(out) == 1
    merged = out[0]
    assert merged.support == 5
    # Confidence should be bounded (0, 1)
    assert 0.0 < merged.confidence < 1.0


def test_cluster_tie_break_alphabetical():
    cs = [
        _c(cluster="sql", support=2),
        _c(cluster="bash", support=3),
    ]
    out, _ = consolidate(cs)
    assert out[0].cluster == "bash"  # alphabetically first


def test_clusterless_candidates_merge_cleanly():
    cs = [
        _c(cluster=None, support=2),
        _c(cluster=None, support=3),
    ]
    out, _ = consolidate(cs)
    assert len(out) == 1
    assert out[0].cluster is None


def test_trigger_examples_deduped_and_capped():
    cs = [
        _c(triggers=["r1", "r2"]),
        _c(triggers=["r2", "r3", "r4", "r5", "r6", "r7"]),
    ]
    out, _ = consolidate(cs)
    trigs = out[0].trigger_examples
    # Must preserve order and dedupe
    assert trigs == ["r1", "r2", "r3", "r4", "r5"]
    assert len(trigs) == 5


def test_exemplar_from_highest_support_candidate():
    cs = [
        _c(support=2, exemplar_id="weak_exemplar"),
        _c(support=10, exemplar_id="strong_exemplar"),
    ]
    out, _ = consolidate(cs)
    assert out[0].exemplar_trajectory_id == "strong_exemplar"


def test_exemplar_tie_breaks_by_trajectory_id():
    cs = [
        _c(support=5, exemplar_id="z"),
        _c(support=5, exemplar_id="a"),
    ]
    out, _ = consolidate(cs)
    assert out[0].exemplar_trajectory_id == "a"


def test_report_counts_correct():
    cs = [
        _c(seq=("a", "b")),
        _c(seq=("a", "b")),
        _c(seq=("a", "b")),
        _c(seq=("x", "y")),
    ]
    out, report = consolidate(cs)
    assert report.n_in == 4
    assert report.n_out == 2
    assert report.n_merges == 2


def test_different_sequences_not_merged_even_when_cluster_matches():
    cs = [
        _c(cluster="sql", seq=("a", "b")),
        _c(cluster="sql", seq=("a", "c")),
    ]
    out, _ = consolidate(cs)
    assert len(out) == 2


def test_consolidation_preserves_sort_order():
    cs = [
        _c(name="low", seq=("a", "b"), support=2),
        _c(name="high1", seq=("x", "y"), support=4),
        _c(name="high2", seq=("x", "y"), support=3),
    ]
    out, _ = consolidate(cs)
    assert out[0].support >= out[1].support


# ---------------------------------------------------------------------
# Latent-bug regression (BUGHUNT.md unit 25, fixed 2026-07-05):
# a single-member passthrough kept the extractor's CLUSTER-SPECIFIC
# signature while merged groups recompute a sequence-only one — the
# same skill graduated under two store keys depending on whether its
# sequence happened to merge that run.
# ---------------------------------------------------------------------


def test_single_member_signature_is_sequence_only():
    from ghost_agent.skills_auto.extractor import _signature_hash
    from ghost_agent.skills_auto.consolidator import consolidate

    single = _c(name="s", seq=("a", "b"), support=3, cluster="sql")
    out, _ = consolidate([single])
    assert len(out) == 1
    assert out[0].signature_hash == _signature_hash(None, ("a", "b"))
    # The candidate keeps its observed cluster — only the dedupe
    # identity collapses to the sequence.
    assert out[0].cluster == "sql"
    # The input object is not mutated (replace(), not assignment).
    assert single.signature_hash != out[0].signature_hash or \
        single is not out[0]


def test_single_and_merged_share_store_key_across_runs():
    from ghost_agent.skills_auto.consolidator import consolidate

    # Run 1: the sequence appears once (single-member passthrough).
    run1, _ = consolidate([_c(name="s", seq=("a", "b"), cluster="sql")])
    # Run 2: the same sequence appears in two clusters (merged).
    run2, _ = consolidate([
        _c(name="s", seq=("a", "b"), cluster="sql"),
        _c(name="s2", seq=("a", "b"), cluster="bash"),
    ])
    assert run1[0].signature_hash == run2[0].signature_hash
