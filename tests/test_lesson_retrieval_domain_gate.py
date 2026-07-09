"""Domain-rescue gate for playbook lesson retrieval (B4 mediation fix).

Measured live on the B4 run (2026-07-09): the arm's real self-play lesson sat
at embedding distance 1.056 from a matching task probe against a strict 0.45
floor — so the skill tier filtered it on all 96 probe turns and the learning
loop was write-only. The fix admits a lesson whose `domains` explicitly
contain the query's cluster up to a relaxed bound (1.25); everything else
keeps the strict floor. `_explicit_query_cluster` deliberately requires a
keyword hit — `classify_cluster`'s python_general FALLBACK would have made
every small-talk turn "match" python_general lessons.
"""

from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.skills import (
    _DOMAIN_RELAXED_DISTANCE, DEFAULT_RETRIEVAL_DISTANCE,
    SkillMemory, _explicit_query_cluster,
)

# The REAL lesson doc shape from the B4 treatment arm.
_B4_LESSON_DOC = (
    "SITUATION: alternative idiom for python_general tasks (novelty=1.00)\n"
    "MISTAKE: \nSOLUTION: On a python_general task that has a familiar shape, "
    "a structurally different solution can be valid."
)


def _stub_vm(docs, dists, metas):
    vm = MagicMock()
    vm.collection.query.return_value = {
        "documents": [docs], "distances": [dists], "metadatas": [metas],
    }
    return vm


def _sm(tmp_path):
    return SkillMemory(tmp_path)


# ── _explicit_query_cluster ──────────────────────────────────────────────────

def test_explicit_cluster_requires_a_keyword_hit():
    assert _explicit_query_cluster("what's my favourite colour?") is None
    assert _explicit_query_cluster("") is None
    assert _explicit_query_cluster("read sales.csv and sum the amounts") == "data_analysis"
    assert _explicit_query_cluster("load it into sqlite and GROUP BY region") == "sql"
    assert _explicit_query_cluster("use python to count primes") == "python_general"


# ── the B4 reproduction ──────────────────────────────────────────────────────

def test_b4_lesson_surfaces_via_domain_rescue(tmp_path):
    """dist=1.056 + domains=python_general + a python-shaped query →
    the exact configuration that returned empty on all 96 B4 probes."""
    vm = _stub_vm([_B4_LESSON_DOC], [1.056],
                  [{"trigger": "alternative idiom for python_general tasks",
                    "domains": "python_general"}])
    items = _sm(tmp_path).get_playbook_items(
        "Use Python to count how many prime numbers are below 100", vm)
    assert items, "the B4 lesson must now surface via the domain gate"
    assert "structurally different solution" in items[0]["text"]


def test_no_cluster_signal_keeps_strict_floor(tmp_path):
    # small talk: no explicit cluster → dist 1.056 stays filtered
    vm = _stub_vm([_B4_LESSON_DOC], [1.056],
                  [{"trigger": "alternative idiom", "domains": "python_general"}])
    items = _sm(tmp_path).get_playbook_items("what's my favourite colour?", vm)
    assert items == []


def test_wrong_domain_stays_filtered(tmp_path):
    vm = _stub_vm([_B4_LESSON_DOC], [1.056],
                  [{"trigger": "alternative idiom", "domains": "sql"}])
    items = _sm(tmp_path).get_playbook_items(
        "Use Python to count how many prime numbers are below 100", vm)
    assert items == []


def test_relaxed_bound_is_still_a_bound(tmp_path):
    vm = _stub_vm([_B4_LESSON_DOC], [_DOMAIN_RELAXED_DISTANCE + 0.05],
                  [{"trigger": "alternative idiom", "domains": "python_general"}])
    items = _sm(tmp_path).get_playbook_items(
        "Use Python to count how many prime numbers are below 100", vm)
    assert items == []


def test_strict_path_unchanged(tmp_path):
    # near-verbatim match without any domains tag still surfaces
    vm = _stub_vm([_B4_LESSON_DOC], [DEFAULT_RETRIEVAL_DISTANCE - 0.1],
                  [{"trigger": "count primes below 100"}])
    items = _sm(tmp_path).get_playbook_items(
        "count primes below 100 again please", vm)
    assert items


def test_untagged_lesson_derives_domain_from_trigger(tmp_path):
    # reflection lessons carry no domains — the trigger text supplies one
    vm = _stub_vm(["SITUATION: fix the sqlite GROUP BY bug\nSOLUTION: quote it"],
                  [1.0],
                  [{"trigger": "fix the sqlite GROUP BY bug", "domains": ""}])
    items = _sm(tmp_path).get_playbook_items(
        "load the csv into sqlite and GROUP BY category", vm)
    assert items


def test_domain_rescued_items_rank_below_strict_hits(tmp_path):
    vm = _stub_vm(
        ["SITUATION: exact prior answer\nSOLUTION: verbatim",
         _B4_LESSON_DOC],
        [0.2, 1.056],
        [{"trigger": "count primes below 100"},
         {"trigger": "alternative idiom", "domains": "python_general"}])
    items = _sm(tmp_path).get_playbook_items(
        "use python: count primes below 100", vm)
    assert len(items) == 2
    assert "verbatim" in items[0]["text"]  # strict hit ranks first
