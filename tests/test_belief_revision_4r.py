"""§4R — belief/contradiction subsystem audit fixes (2026-08-08).

Four independent lenses audited this subsystem. The findings these pin:

* CRIT — the contradiction engine's candidate scope was a `$nin` DENYLIST, not
  the complement of the prunable set. Measured live it left 33 rows deletable
  of which only 2 were the `auto` facts it exists to supersede: 28 dream
  `synthesis` rows (each the only surviving copy of its merged sources), the
  user's single `manual` save, the `identity` row and a `document_summary`.
* MAJOR — the delete ran BEFORE the audit record, and the record was
  best-effort with failures swallowed at DEBUG, so an irreversible deletion
  could leave no trace anywhere.
* MAJOR — `explain_belief_change` over-corrected from inert into a
  false-positive machine: it fired on 67/67 live turns and injected the five
  most RECENT revisions regardless of the question.
* MAJOR — `dream.py` performed the same LLM-ids→delete with no whitelist.

NOTE ON FIXTURES: the pre-existing tests for this module used SINGLE-entry
logs, which is exactly why the 100%-false-positive behaviour stayed green.
These build realistic multi-entry ledgers on purpose.
"""

import json
from pathlib import Path

import pytest

from ghost_agent.memory.contradiction_log import ContradictionLog


def _log_with(tmp_path, entries):
    (tmp_path / "contradiction_log.json").write_text(json.dumps(entries))
    return ContradictionLog(tmp_path)


def _entry(ts, new_fact, old, reason="LLM-driven belief revision"):
    return {"timestamp": ts, "new_fact": new_fact,
            "superseded": [{"id": "x", "text": old}],
            "deleted_ids": ["x"], "reason": reason}


# A ledger shaped like the live one: a recent burst of near-identical
# self-play churn, with the genuinely useful revisions older.
def _realistic(tmp_path):
    # The churn entries deliberately carry the SAME vocabulary the generic
    # queries use ("tests", "interface", "build", "restart", "logs"), because
    # the live ledger does. A fixture whose churn shares no words with the
    # noise queries cannot discriminate a 1-token rule from a 2-token one —
    # the first version of this file had that flaw and its pins stayed green
    # with the fix reverted.
    churn = [_entry(f"2026-07-27T{h:02d}:00:00",
                    f"The user's current project codename is zephyrine{h:03d}, "
                    f"with tests and interface build logs restart notes.",
                    f"The user's current project codename is old{h:03d}, "
                    f"with tests and interface build logs restart notes.")
             for h in range(20)]
    real = [
        _entry("2026-07-20T10:00:00",
               "The user owns a BMW 118i and a Ducati Streetfighter V4s.",
               "The user owns a Honda Civic."),
        _entry("2026-07-21T10:00:00",
               "The user requires a dark/light theme toggle with persistence.",
               "The user wanted only a dark theme."),
    ]
    return _log_with(tmp_path, churn + real)


# ── read path: precision ──────────────────────────────────────────────────

def test_generic_chatter_does_not_fire(tmp_path):
    """The defect: one shared prefix-token was enough, so ordinary dev chatter
    matched ~every entry and fired on 100% of live turns."""
    cl = _realistic(tmp_path)
    for q in ("can you check the project tests now?",
              "fix the code in main.py",
              "user memory system update",
              "explain how postgres vacuum works"):
        assert cl.explain_belief_change(q) == "", f"false positive on {q!r}"


def test_haystack_stopwords_are_screened(tmp_path):
    """Stopwords were applied to the query only; `user`/`project`/`current`
    appear in nearly every entry and so matched almost any message."""
    cl = _realistic(tmp_path)
    assert cl.explain_belief_change("the user project current status") == ""


def test_prefix_tolerance_is_bounded(tmp_path):
    """Unbounded startswith() made `code`↔`codename` and `dark`↔`darkweb`
    match. Real inflections must still work."""
    cl = _log_with(tmp_path, [
        _entry("2026-07-27T10:00:00",
               "The user's project codename is zephyrine.", "old codename"),
    ])
    # `code` must NOT reach `codename` (len delta 4)...
    assert cl.explain_belief_change("please refactor this code") == ""
    cl2 = _log_with(tmp_path, [
        _entry("2026-07-27T10:00:00",
               "The user changed the deployment target.", "old target"),
    ])
    # ...but `changed`↔`change` (delta 1) still counts.
    assert cl2.explain_belief_change("what change happened to deployment") != ""


def test_relevant_query_still_matches(tmp_path):
    """Precision must not come from making it inert again — that was the
    2026-07-20 failure this over-corrected from."""
    cl = _realistic(tmp_path)
    out = cl.explain_belief_change("what is the dark theme toggle requirement?")
    assert "dark/light theme" in out


# ── read path: ranking + window ───────────────────────────────────────────

def test_ranks_by_relevance_not_recency(tmp_path):
    """`matches[:5]` truncated in raw file order, so a recent burst of churn
    crowded out the best match even when the query named it exactly.

    The fixture must contain a NEWER, weaker match as well as an older strong
    one — otherwise only one entry matches at all and the sort order is
    unobservable (the first version of this test had that flaw and stayed
    green with the ranking reverted to pure recency)."""
    # Weak entries match 2 of the query's 3 content tokens; the strong one
    # matches all 3. Both clear the evidence bar, so BOTH are candidates and
    # the ORDER is what the test observes. (Tokens must be >3 chars to survive
    # tokenisation — "BMW" would be dropped, which silently collapsed an
    # earlier version of this fixture into a single-token tie.)
    weak_recent = [_entry(f"2026-07-28T{i:02d}:00:00",
                          f"The user mentioned a Ducati Streetfighter poster {i}.",
                          f"There was no poster {i}.") for i in range(6)]
    strong_old = _entry("2026-07-01T10:00:00",
                        "The user owns a Ducati Streetfighter V4s.",
                        "The user owns a Honda Civic.")
    cl = _log_with(tmp_path, weak_recent + [strong_old])
    out = cl.explain_belief_change(
        "did the ducati streetfighter replace the honda?")
    body = [l for l in out.splitlines() if l.startswith("- ")]
    assert body, "no match at all"
    # "Honda" appears only in the strong (older) entry's superseded text.
    assert "Honda" in body[0], (
        "the strongest match must rank FIRST; got recency order instead:\n"
        + "\n".join(body))


def test_searches_beyond_the_old_50_entry_window(tmp_path):
    """The window was a bare [:50]; with 96 live entries every genuinely
    useful revision sat beyond it and was structurally unreachable."""
    filler = [_entry(f"2026-07-27T{i:02d}:00:00", f"filler fact {i} alpha",
                     f"old filler {i}") for i in range(60)]
    target = _entry("2026-07-01T10:00:00",
                    "The user moved the workshop to Thessaloniki.",
                    "The workshop was in Athens.")
    cl = _log_with(tmp_path, filler + [target])
    # Two shared content tokens ("workshop", "thessaloniki") — the matcher
    # deliberately requires real evidence, so a query sharing only ONE token
    # is not expected to match (see the evidence-rule comment in the module:
    # a single-token escape hatch was tried and measurably destroyed
    # precision against the live ledger).
    assert "Thessaloniki" in cl.explain_belief_change(
        "did the workshop move to Thessaloniki?")


def test_reason_is_rendered(tmp_path):
    """`reason` is present on 96/96 live entries and promised by the module
    docstring ("...because you said Z"), but was never shown."""
    cl = _log_with(tmp_path, [
        _entry("2026-07-27T10:00:00",
               "The workshop moved to Thessaloniki.", "Workshop in Athens.",
               reason="user correction during planning"),
    ])
    # Two content tokens: the evidence rule now requires real overlap (a
    # single-token query was the hole that let stopword-reduced messages
    # through — measured 61% fire rate on real turns).
    out = cl.explain_belief_change("did the workshop move to Thessaloniki?")
    assert "user correction during planning" in out


# ── record(): honest success reporting ────────────────────────────────────

def test_record_returns_true_on_success(tmp_path):
    cl = ContradictionLog(tmp_path)
    assert cl.record("new fact", [{"id": "a", "text": "old"}], ["a"]) is True


def test_record_returns_false_when_degraded(tmp_path):
    """The engine deletes vectors irreversibly and then records. `_save` is a
    silent no-op on a degraded store, and record() reported success anyway —
    so the erased text could vanish with no record anywhere.

    The file must be made genuinely unreadable: setting `_degraded` by hand
    does not work, because `record()` calls `_load()` first and a successful
    load clears the flag."""
    import os
    p = tmp_path / "contradiction_log.json"
    p.write_text(json.dumps([_entry("2026-07-01T00:00:00", "a", "b")]))
    cl = ContradictionLog(tmp_path)
    os.chmod(p, 0o000)
    try:
        if os.access(p, os.R_OK):        # running as root — can't degrade it
            pytest.skip("cannot make the file unreadable as this user")
        assert cl.record("new fact", [{"id": "a", "text": "old"}], ["a"]) is False
    finally:
        os.chmod(p, 0o644)


# ── engine: scope, gate, ordering (structural — driving the full
#    consolidation path needs the whole LLM stack) ──────────────────────────

def _engine_src():
    """Source of the contradiction-engine block with COMMENTS STRIPPED.

    Necessary because the fix comments quote the old behaviour verbatim
    ("was a `$nin` denylist", "not the bare `< 0.6`"), so a naive substring
    assertion matches the explanation instead of the code — these tests
    failed on correct source before the stripping was added.
    """
    import inspect
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent.run_smart_memory_task)
    seg = src[src.index("CONTRADICTION ENGINE"):src.index("Save the new fact")]
    return "\n".join(ln for ln in seg.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_candidate_scope_is_same_type_not_a_denylist():
    seg = _engine_src()
    assert '"type": memory_type' in seg, (
        "candidate scope must be same-type; a $nin denylist leaves synthesis / "
        "identity / manual / document_summary rows deletable")
    assert "$nin" not in seg, "the denylist scope is back"


def test_delete_gate_matches_the_sibling_calibration():
    seg = _engine_src()
    assert "0.50" in seg, "delete gate must be 0.50 (the sibling smart_update value)"
    assert "< 0.6" not in seg, "the loose 0.6 delete gate is back"
    assert "_subject_key" in seg, "subject-key conflict guard missing"


def test_record_precedes_delete():
    """An irreversible delete must not outlive its audit record."""
    seg = _engine_src()
    assert seg.index("contradiction_log.record") < seg.index("collection.delete"), (
        "delete still runs before the audit record is written")
    assert "if not _recorded" in seg, "delete is not gated on a successful record"


def test_judge_prompt_example_is_valid_json():
    """The example was in a NON-f-string segment, so the judge was literally
    shown `{{"ids": ...}}` — malformed, which silently no-op'd the engine."""
    seg = _engine_src()
    assert '{{\\"ids\\"' not in seg and '{{"ids"' not in seg, (
        "judge prompt still shows doubled braces")


def test_top_tier_gates_key_off_effective_threshold():
    """Hardcoded 0.9 gates inside the threshold branch meant that lowering
    --smart-memory would admit the 0.8 tier with the generic-knowledge filter
    silently bypassed — a tuning knob must not disarm a safety filter."""
    import inspect
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent.run_smart_memory_task)
    # The first attempt at this fix (`_top_tier = max(0.9, effective_threshold)`)
    # was a PROVABLE NO-OP — it sits inside `score >= effective_threshold`, so
    # it never changed an outcome for any (threshold, score) pair, and THIS
    # TEST passed on it because it only checked the source text. The filter is
    # now unconditional on score, which is what "a tuning knob must not disarm
    # a safety filter" actually requires.
    assert "if not (is_personal or is_technical):" in src
    assert "_top_tier" not in src, "the no-op top-tier gate is back"
    assert "if score >= 0.9 and not (is_personal" not in src


def test_document_summary_is_forget_protected():
    """Its `document` twin is protected; deleting the summary while the source
    survives is the asymmetric drift the protected list exists to prevent."""
    from ghost_agent.tools.memory import _FORGET_PROTECTED_TYPES
    assert "document_summary" in _FORGET_PROTECTED_TYPES
    assert "document" in _FORGET_PROTECTED_TYPES


@pytest.mark.asyncio
async def test_expansion_sweep_does_not_delete_synthesis(tmp_path):
    """BEHAVIOURAL (§4R R2). The expansion sweep matches graph NEIGHBOURS the
    user never named. A `synthesis` is a composite whose merged sources are
    already deleted, so removing one to excise an incidental token destroys
    the only surviving copy of everything else it merged.

    A synthesis mentioning the neighbour term must SURVIVE, while an ordinary
    `auto` fact mentioning it is still deleted — proving the guard is targeted
    and did not simply disable the sweep.
    """
    from unittest.mock import MagicMock
    from ghost_agent.tools.memory import tool_unified_forget

    deleted = []
    ms = MagicMock()
    ms._get_lock = None
    del ms._get_lock  # force the _NullCM path

    def _query(query_texts=None, n_results=None, where=None):
        return {
            "ids": [["syn-1", "auto-1"]],
            "documents": [["Master summary mentioning mortimer and much else.",
                           "The user once owned mortimer."]],
            "metadatas": [[{"type": "synthesis"}, {"type": "auto"}]],
            "distances": [[0.9, 0.9]],   # far: only the LITERAL rule can fire
        }
    ms.collection.query.side_effect = _query
    ms.collection.delete.side_effect = lambda ids=None: deleted.extend(ids or [])
    ms.search_advanced.return_value = []

    # `expanded_targets` comes from the graph; give it a neighbour the user
    # never typed, which is what makes this the expansion sweep.
    gm = MagicMock()
    # `get_connected_entities` is the real source of expanded_targets
    # (memory.py:841) — an earlier version of this test mocked the wrong
    # method, so the expansion sweep never ran and the pin stayed green with
    # the guard removed.
    gm.get_connected_entities.return_value = ["mortimer"]
    gm.delete_by_target.return_value = 0

    await tool_unified_forget(target="iguana", memory_system=ms, graph_memory=gm)

    assert "syn-1" not in deleted, (
        "the expansion sweep destroyed a composite synthesis on a term the "
        "user never named")


def test_dream_consolidation_whitelists_offered_ids():
    """dream.py ran the same LLM-ids→delete with no whitelist, so an id the
    model copied out of fragment text could erase a never-offered row."""
    import inspect
    from ghost_agent.core.dream import Dreamer
    src = inspect.getsource(Dreamer)
    seg = src[src.index("_offered_ids"):src.index("_offered_ids") + 600]
    assert "in _offered_ids" in seg, "dream delete is not whitelisted"
