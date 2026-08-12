"""Tool-ontology analysis (optim/tool_ontology.py).

Two claims are load-bearing and therefore pinned here: a confusion pair seen
in BOTH directions is a boundary problem (not a wording problem), and
"cohesion" means the calls share a TARGET — not that they share an action
enum, which is what made the first version read 0.89 on every same-tool run.
"""
from __future__ import annotations

import json
import pytest
from types import SimpleNamespace

from ghost_agent.optim import tool_ontology as ont


def _call(name, **args):
    return SimpleNamespace(name=name, arguments=dict(args), result="", error="")


def _traj(tid, calls, kind="user_request"):
    return SimpleNamespace(id=tid, task_kind=kind, tool_calls=calls)


# ── confusion ─────────────────────────────────────────────────────────

def _rows(*triples):
    return [{"truth": t, "picked": p, "err": e} for t, p, e in triples]


def test_fidelity_excludes_unreplayable_rows():
    """Unreplayable rows measure the replay plumbing, not the toolbox — the
    diagnostic denominator must exclude them (the GEPA metric scores them 0
    on purpose; that is a different question)."""
    rep = ont.analyze_confusion(_rows(
        ("file_system", "file_system", ""),
        ("browser", "browser", ""),
        ("execute", None, "unreplayable"),
    ))
    assert rep.n == 3 and rep.n_unreplayable == 1
    assert rep.fidelity == 1.0


def _noise_rows(n=200):
    """A background of correct picks plus scattered misses across many tools —
    so the marginal null has something to model. Without this the pair under
    test IS the whole corpus and every excess test is vacuous."""
    tools = ["file_system", "browser", "execute", "manage_projects",
             "web_search", "introspect"]
    rows = [{"truth": t, "picked": t, "err": ""}
            for i in range(n) for t in [tools[i % len(tools)]]]
    for i in range(24):
        a, b = tools[i % 6], tools[(i + 2) % 6]
        rows.append({"truth": a, "picked": b, "err": ""})
    return rows


def test_two_way_pairs_are_described_never_adjudicated():
    """Five statistical gates for this were each refuted by simulation — the
    last with BOTH error rates degrading as the corpus grew (FP 0%→50%, power
    75%→0%). The pair is surfaced with its raw shape and no verdict tier that
    could be mistaken for a mandate."""
    rows = _noise_rows() + _rows(*([("browser", "file_system", "")] * 25
                                   + [("file_system", "browser", "")] * 24))
    rep = ont.analyze_confusion(rows)
    v = next(v for v in rep.verdicts if v.kind == "merge_or_redraw"
             and set(v.tools) == {"browser", "file_system"})
    assert v.count == 49
    assert v.evidence == "observed"          # never "significant"
    assert "not adjudicated here" in v.detail
    assert "of all confusion" in v.detail
    # Each count must name ITS OWN direction — printing them against the
    # sorted pair transposed them whenever truth > picked lexicographically,
    # which sends an operator to rework the tool that was RIGHT.
    assert v.directions == {"browser->file_system": 25,
                            "file_system->browser": 24}
    assert "browser -> file_system 25x" in v.detail
    assert "file_system -> browser 24x" in v.detail
    # ...reported ONCE, not once per direction.
    assert sum(1 for x in rep.verdicts if x.kind == "merge_or_redraw") == 1


def test_no_corpus_size_can_make_a_two_way_pair_significant():
    """The regression that matters: the old gate's false-positive rate climbed
    to 50% at n=6000. No amount of data may promote a two-way pair now."""
    for scale in (1, 10, 50):
        rows = _noise_rows(200 * scale) + _rows(
            *([("browser", "file_system", "")] * (25 * scale)
              + [("file_system", "browser", "")] * (24 * scale)))
        assert not [v for v in ont.analyze_confusion(rows).verdicts
                    if v.kind == "merge_or_redraw" and v.evidence == "significant"]


def test_tiny_two_way_pairs_stay_out_of_the_verdict_list():
    """A 2-vs-1 pair (symmetry p=1.000) is raw-data material, not a proposal —
    but the data must survive where a reader can still find it."""
    rows = _noise_rows() + _rows(("a_tool", "b_tool", ""), ("a_tool", "b_tool", ""),
                                 ("b_tool", "a_tool", ""))
    rep = ont.analyze_confusion(rows)
    assert not [v for v in rep.verdicts if set(v.tools) == {"a_tool", "b_tool"}]
    assert any(pp.truth == "a_tool" for pp in rep.pairs)   # still in raw data


def test_unidirectional_confusion_is_a_description_verdict():
    """6-0 clears the exact binomial (p=0.031); 4-0 does not (p=0.125)."""
    rep = ont.analyze_confusion(_rows(*([("web_search", "manage_projects", "")] * 6)))
    v = next(v for v in rep.verdicts if v.kind == "describe")
    assert v.tools == ("web_search", "manage_projects")
    assert v.evidence == "significant"
    assert "one-way and statistically supported" in v.detail.lower()

    weak = ont.analyze_confusion(_rows(*([("web_search", "manage_projects", "")] * 4)))
    wv = next(v for v in weak.verdicts if v.kind == "describe")
    assert wv.evidence == "suggestive"
    assert "consistent with chance" in wv.detail


def test_symmetry_p_matches_the_exact_binomial():
    assert ont.binomial_symmetry_p(2, 1) == pytest.approx(1.0)
    assert ont.binomial_symmetry_p(3, 2) == pytest.approx(1.0)
    assert ont.binomial_symmetry_p(4, 0) == pytest.approx(0.125)
    assert ont.binomial_symmetry_p(6, 0) == pytest.approx(0.03125)
    assert ont.binomial_symmetry_p(0, 0) == 1.0


def test_adjudicated_verdicts_sort_above_merely_observed():
    rows = _rows(*([("a", "b", "")] * 6                    # significant one-way
                   + [("c", "d", "")] * 3 + [("d", "c", "")] * 3))  # two-way
    rep = ont.analyze_confusion(rows)
    assert rep.verdicts[0].evidence == "significant"
    assert {v.evidence for v in rep.verdicts} <= {"significant", "observed",
                                                  "suggestive", "counted"}


def test_no_tool_stalls_are_their_own_class():
    rep = ont.analyze_confusion(_rows(*([("notify_operator", None, "")] * 3)))
    assert rep.n_no_tool == 3
    v = next(v for v in rep.verdicts if v.kind == "missing_affordance")
    assert v.tools == ("notify_operator",)


def test_rare_pairs_stay_below_the_pattern_threshold():
    rep = ont.analyze_confusion(_rows(("a", "b", "")), min_pair=2)
    assert rep.verdicts == []
    assert rep.pairs and rep.pairs[0].count == 1  # still visible as raw data


def test_per_tool_recall_and_theft_bookkeeping():
    rep = ont.analyze_confusion(_rows(
        ("file_system", "file_system", ""),
        ("file_system", "browser", ""),
        ("file_system", "browser", ""),
    ))
    fs = rep.per_tool["file_system"]
    assert fs.n_truth == 3 and fs.n_correct == 1
    assert abs(fs.recall - 1 / 3) < 1e-9
    assert fs.stolen_by == {"browser": 2}
    assert rep.per_tool["browser"].steals_from == {"file_system": 2}


def test_confusion_tolerates_junk_rows():
    rep = ont.analyze_confusion([None, {}, {"truth": ""}, "nope",
                                 {"truth": "a", "picked": "a"}])
    assert rep.n == 1 and rep.n_correct == 1


def test_empty_report_renders_a_hint():
    assert "run the Phase 2b runner" in ont.render_confusion(ont.ConfusionReport())


# ── sequence mining ───────────────────────────────────────────────────

def test_mines_recurring_sequences_above_support():
    trajs = [_traj(f"t{i}", [_call("file_system", path="a.py"),
                             _call("execute", command="python a.py")])
             for i in range(4)]
    out = ont.mine_sequences(trajs, min_support=3)
    seqs = {c.sequence for c in out}
    assert ("file_system", "execute") in seqs


def test_support_is_per_trajectory_not_per_occurrence():
    """One pathological turn repeating a pair 50× must not mint a proposal —
    otherwise a single grind session redesigns the toolbox."""
    calls = [_call("browser", url="u"), _call("browser", url="u")] * 25
    out = ont.mine_sequences([_traj("only-one", calls)], min_support=3)
    assert out == []


def test_cohesion_ignores_the_action_enum():
    """Two reads of DIFFERENT files share operation='read' and nothing else.
    Counting that as a shared target is what inflated the first version."""
    trajs = [_traj(f"t{i}", [_call("file_system", operation="read", path=f"a{i}.py"),
                             _call("file_system", operation="read", path=f"b{i}.py")])
             for i in range(4)]
    out = ont.mine_sequences(trajs, min_support=3)
    cand = next(c for c in out if c.sequence == ("file_system", "file_system"))
    assert cand.cohesion == 0.0


def test_cohesion_detects_the_same_file_across_tools_by_containment():
    """`file_system(path="app.py")` → `execute(command="python3 app.py")` is one
    operation on one target, spelled two ways. Exact matching missed it."""
    trajs = [_traj(f"t{i}", [_call("file_system", operation="write", path="app.py"),
                             _call("execute", command="python3 app.py")])
             for i in range(4)]
    out = ont.mine_sequences(trajs, min_support=3)
    cand = next(c for c in out if c.sequence == ("file_system", "execute"))
    assert cand.cohesion == 1.0
    assert "app.py" in cand.example_targets


def test_redaction_sentinel_is_not_a_shared_target():
    trajs = [_traj(f"t{i}", [_call("manage_projects", secret="<REDACTED>"),
                             _call("manage_projects", secret="<REDACTED>")])
             for i in range(4)]
    out = ont.mine_sequences(trajs, min_support=3)
    cand = next(c for c in out if c.sequence == ("manage_projects",) * 2)
    assert cand.cohesion == 0.0


def test_numeric_arguments_are_never_targets():
    trajs = [_traj(f"t{i}", [_call("browser", timeout_ms=30000, url=f"u{i}"),
                             _call("browser", timeout_ms=30000, url=f"v{i}")])
             for i in range(4)]
    out = ont.mine_sequences(trajs, min_support=3)
    cand = next(c for c in out if c.sequence == ("browser", "browser"))
    assert cand.cohesion == 0.0


def test_steps_collapsed_and_priority_ranking():
    long_seq = [_call("file_system", path="x.py") for _ in range(4)]
    trajs = [_traj(f"t{i}", list(long_seq)) for i in range(4)]
    out = ont.mine_sequences(trajs, min_support=3)
    quad = next(c for c in out if len(c.sequence) == 4)
    assert quad.steps_collapsed == quad.occurrences * 3
    assert out[0].priority >= out[-1].priority  # ranked worst-first


def test_task_kind_filter_excludes_synthetic_work():
    synthetic = [_traj(f"s{i}", [_call("execute", command="c"),
                                 _call("execute", command="c")], kind="self_play")
                 for i in range(5)]
    assert ont.mine_sequences(synthetic, min_support=3) == []
    assert ont.mine_sequences(synthetic, min_support=3, task_kinds=None)


def test_mining_survives_hostile_records():
    class Hostile:
        id = "h"
        task_kind = "user_request"

        @property
        def tool_calls(self):
            raise RuntimeError("boom")

    good = [_traj(f"t{i}", [_call("a", path="p.py"), _call("b", path="p.py")])
            for i in range(3)]
    out = ont.mine_sequences([Hostile()] + good, min_support=3)
    assert any(c.sequence == ("a", "b") for c in out)


def test_unnamed_calls_do_not_form_sequences():
    trajs = [_traj(f"t{i}", [_call("", path="x"), _call("execute", command="x")])
             for i in range(4)]
    assert ont.mine_sequences(trajs, min_support=3) == []


# ── plumbing ──────────────────────────────────────────────────────────

def test_load_replay_rows_skips_torn_lines(tmp_path):
    p = tmp_path / "confusion.jsonl"
    p.write_text('{"truth": "a", "picked": "b"}\n{torn\n\n[1,2]\n')
    rows = ont.load_replay_rows(p)
    assert rows == [{"truth": "a", "picked": "b"}]
    assert ont.load_replay_rows(tmp_path / "missing.jsonl") == []


def test_report_to_dict_is_json_serializable():
    rep = ont.analyze_confusion(_rows(("a", "b", ""), ("a", "b", "")))
    macros = ont.mine_sequences(
        [_traj(f"t{i}", [_call("a", path="p.py"), _call("b", path="p.py")])
         for i in range(3)], min_support=3)
    blob = json.dumps(ont.report_to_dict(rep, macros))
    assert "macro_candidates" in blob and "confusion" in blob


def test_render_sequences_handles_empty():
    assert "No recurring tool sequences" in ont.render_sequences([])


# ── GEPA isolation (Phase 2b fixture corpus) ──────────────────────────

def test_steered_turns_are_excluded_from_tool_fixtures(tmp_path):
    """A live A/B treatment that rewrites the prompt context must not leak
    into the GEPA replay corpus: the optimizer replays recorded payloads
    verbatim, so those fixtures would tune descriptions against a context
    only one arm ever sees."""
    import json as _json
    from ghost_agent.optim.tool_fixtures import mine_fixtures

    traj_root = tmp_path / "trajectories" / "2026-08-05"
    traj_root.mkdir(parents=True)
    rec = tmp_path / "rec.jsonl"

    def _traj(sid, extra):
        return {"id": sid, "session_id": sid, "task_kind": "user_request",
                "outcome": "passed", "user_request": "do the thing",
                "tool_calls": [], "extra": extra,
                "timestamp": "2026-08-05T12:00:00Z"}

    (traj_root / "session-a.jsonl").write_text(
        _json.dumps(_traj("req-clean", {"experiments": {"risk_steer": "control"}}))
        + "\n"
        + _json.dumps(_traj("req-steered", {"experiments": {"risk_steer": "treatment"},
                                            "risk_steer_fired": True})) + "\n")

    def _record(req):
        return {"kind": "chat_completion_stream", "ts": "2026-08-05T12:00:00Z",
                "request_id": req, "session_id": "s1", "ordinal": 1,
                "payload": {"tools": [{"function": {"name": "file_system"}}],
                            "messages": []},
                "response": {"choices": [{"message": {"tool_calls": [
                    {"function": {"name": "file_system", "arguments": "{}"}}]}}]}}

    rec.write_text(_json.dumps(_record("req-clean")) + "\n"
                   + _json.dumps(_record("req-steered")) + "\n")

    fixtures, stats = mine_fixtures([rec], tmp_path / "trajectories",
                                    era_cutoff_local="2026-07-31T19:15")
    assert [f.request_id for f in fixtures] == ["req-clean"]
    assert stats["experiment_context_excluded"] == 1

    # ...and the exclusion is overridable, for a post-experiment re-mine.
    fixtures2, _ = mine_fixtures([rec], tmp_path / "trajectories",
                                 era_cutoff_local="2026-07-31T19:15",
                                 exclude_mutated_context=False)
    assert {f.request_id for f in fixtures2} == {"req-clean", "req-steered"}


def test_context_mutation_flag_reads_the_stamp():
    from ghost_agent.core.experiments import context_was_mutated
    assert context_was_mutated(SimpleNamespace(extra={"risk_steer_fired": True}))
    assert not context_was_mutated(SimpleNamespace(extra={"risk_steer_fired": False}))
    assert not context_was_mutated(SimpleNamespace(extra={}))
    assert not context_was_mutated(SimpleNamespace(extra=None))


# ── savings arithmetic (must not overstate) ───────────────────────────

def test_steps_collapsed_counts_non_overlapping_matches_only():
    """Five identical consecutive calls contain FOUR overlapping 2-grams but
    can only be collapsed into TWO macro calls, saving 2 steps. Counting
    windows overstated the saving by up to 3.4x."""
    calls = [_call("file_system", path="x.py") for _ in range(5)]
    trajs = [_traj(f"t{i}", list(calls)) for i in range(3)]
    out = ont.mine_sequences(trajs, sizes=(2,), min_support=3)
    pair = next(c for c in out if c.sequence == ("file_system", "file_system"))
    assert pair.occurrences == 12       # 4 windows x 3 turns (overlapping)
    assert pair.collapsible == 6        # 2 disjoint x 3 turns
    assert pair.steps_collapsed == 6    # 6 x (2-1)


def test_priority_is_not_dominated_by_one_grind_session():
    """A single 50-call grind (support 3) used to outrank a genuine 9-turn
    macro by 5.7x — the exact failure min_support exists to prevent, leaking
    back in through the ranking."""
    grind = [_traj(f"g{i}", [_call("browser", url="u"),
                             _call("browser", url="u")] * 25)
             for i in range(3)]
    real = [_traj(f"r{i}", [_call("file_system", path="app.py"),
                            _call("execute", command="python3 app.py")])
            for i in range(9)]
    out = ont.mine_sequences(grind + real, sizes=(2,), min_support=3)
    top = out[0].sequence
    assert top == ("file_system", "execute"), [
        (c.sequence, round(c.priority, 1)) for c in out]


def test_support_is_stable_for_a_streamed_corpus():
    """id() reuse on a lazily-consumed generator collapsed 8 distinct id-less
    records to support=3 — the same data answering differently depending only
    on whether the caller materialised the iterator."""
    def _mk():
        for _ in range(8):
            t = _traj("", [_call("a", path="p.py"), _call("b", path="p.py")])
            t.id = ""
            yield t

    streamed = ont.mine_sequences(_mk(), sizes=(2,), min_support=3)
    listed = ont.mine_sequences(list(_mk()), sizes=(2,), min_support=3)
    assert streamed[0].support == 8
    assert streamed[0].support == listed[0].support


def test_both_fidelity_numbers_are_reported_and_differ():
    """The runner's number and the replayable-only number are ~7 points apart
    on a real dump; claiming they are the same was the original defect."""
    rows = _rows(*([("a", "a", "")] * 44 + [("a", "b", "")] * 8
                   + [("a", None, "unreplayable")] * 5))
    rep = ont.analyze_confusion(rows)
    assert rep.fidelity == pytest.approx(44 / 52)
    assert rep.fidelity_runner == pytest.approx(44 / 57)
    out = ont.render_confusion(rep)
    assert "fidelity(replayable)" in out and "runner's number" in out


# ── re-review findings (2026-08-05, fixes-of-fixes) ───────────────────

def test_greedy_cursor_is_per_sequence_not_shared():
    """One shared cursor let a match of sequence X consume windows belonging
    to sequence Y. On the live corpus that understated the total saving by
    49.6%, and DIRECTIONALLY: same-tool runs claim the cursor first, starving
    exactly the cross-tool sequences a macro proposal is for."""
    trajs = [_traj(f"t{i}", [_call("a", path="p.py"), _call("b", path="p.py")] * 4)
             for i in range(3)]
    out = ont.mine_sequences(trajs, sizes=(2,), min_support=3)
    ba = next(c for c in out if c.sequence == ("b", "a"))
    # 4 A→B pairs and 3 B→A pairs per turn; the B→A macro is real.
    assert ba.collapsible == 9, [(c.sequence, c.collapsible) for c in out]
    assert ba.steps_collapsed == 9


def test_unrelated_neighbour_does_not_change_a_sequence_count():
    plain = [_traj(f"t{i}", [_call("r", path="p"), _call("e", path="p"),
                             _call("x", path="p")] * 4) for i in range(3)]
    prefixed = [_traj(f"u{i}", [_call("z", path="p")]
                      + [_call("r", path="p"), _call("e", path="p"),
                         _call("x", path="p")] * 4) for i in range(3)]
    a = next(c for c in ont.mine_sequences(plain, sizes=(2,), min_support=3)
             if c.sequence == ("r", "e"))
    b = next(c for c in ont.mine_sequences(prefixed, sizes=(2,), min_support=3)
             if c.sequence == ("r", "e"))
    assert a.collapsible == b.collapsible





def test_direction_counts_are_not_transposed_either_way():
    """The MAJOR defect: `tools` is sorted for stable identity while `count`
    belongs to truth->picked, so the two orderings are independent. Both
    matrices below must render distinguishably."""
    a = ont.analyze_confusion(_noise_rows() + _rows(
        *([("file_system", "browser", "")] * 25
          + [("browser", "file_system", "")] * 24)))
    b = ont.analyze_confusion(_noise_rows() + _rows(
        *([("browser", "file_system", "")] * 25
          + [("file_system", "browser", "")] * 24)))
    va = next(v for v in a.verdicts if v.kind == "merge_or_redraw")
    vb = next(v for v in b.verdicts if v.kind == "merge_or_redraw")
    assert va.directions == {"file_system->browser": 25, "browser->file_system": 24}
    assert vb.directions == {"browser->file_system": 25, "file_system->browser": 24}
    assert va.detail != vb.detail


def test_verdict_order_is_independent_of_row_arrival_order():
    """Equal-magnitude verdicts kept Counter insertion order — i.e. ROW order —
    so with a `top` cutoff the same corpus hid a different finding per run."""
    import random
    base = _noise_rows()
    for i in range(8):
        base += _rows(*([(f"t{i}a", f"t{i}b", "")] * 4))
    orders = []
    for seed in range(6):
        rows = list(base)
        random.Random(seed).shuffle(rows)
        orders.append([tuple(v.tools) for v in ont.analyze_confusion(rows).verdicts])
    assert len({tuple(o) for o in orders}) == 1


def test_evidence_defaults_to_the_weakest_claim():
    """A verdict built without an explicit tier must not assert that a null
    was rejected."""
    assert ont.OntologyVerdict(kind="x", tools=("a",), count=1,
                               detail="").evidence == "observed"


def test_inconclusive_sentence_counts_only_what_is_shown():
    rows = _noise_rows()
    for i in range(14):
        rows += _rows(*([(f"p{i}a", f"p{i}b", "")] * 3
                        + [(f"p{i}b", f"p{i}a", "")] * 3))
    rep = ont.analyze_confusion(rows)
    out = ont.render_confusion(rep, top=5)
    shown = sum(1 for v in rep.verdicts[:5]
                if v.evidence in ("observed", "suggestive"))
    assert f"({shown} PAIR verdict(s) above" in out
    assert "further verdict(s) not shown" in out


# ── fs_batch arm uptake (2026-08-11) ──────────────────────────────────
#
# §4F's stated verification for this arm is MECHANICAL — "re-run the ontology
# report; those n-grams must collapse. If they do not, the macro did not land,
# whatever the latency says." It needs no statistical power, so it is readable
# long before the outcome comparison is. What it must never do is confuse "the
# model refused the capability" with "the model was never offered a chance to
# use it": both render as zero `paths` calls, and only the ELIGIBLE denominator
# separates them. Measured on the live corpus the day this shipped: treatment
# had 0 eligible turns and 0 `paths` calls — no finding at all.

def _armed(tid, calls, arm):
    t = _traj(tid, calls)
    t.extra = {"experiments": {"fs_batch": arm}}
    return t


def _two_reads():
    return [_call("file_system", operation="read", path="a.py"),
            _call("file_system", operation="read", path="b.py")]


def test_eligibility_comes_from_the_SHIPPED_collapse_rule():
    """Not a hand-written '≥2 reads' test. `read_chunked` runs are pagination
    and deliberately OUT of the macro's scope, so a turn full of them offers it
    nothing — a second definition of eligibility would count them and measure a
    macro nobody deployed."""
    paging = [_call("file_system", operation="read_chunked", path="a.py")
              for _ in range(4)]
    rep = ont.fs_batch_arm_uptake([_armed("t1", paging, "treatment")])
    assert rep["arms"]["treatment"].eligible_turns == 0
    assert rep["corpus_eligible_turns"] == 0

    rep2 = ont.fs_batch_arm_uptake([_armed("t2", _two_reads(), "treatment")])
    assert rep2["arms"]["treatment"].eligible_turns == 1
    assert rep2["arms"]["treatment"].removable_steps == 1


def test_zero_uptake_on_zero_opportunity_is_NOT_a_verdict():
    """⚠ THE LIVE SHAPE, 2026-08-11. Treatment never emitted `paths` — and
    never once had two whole-file reads to merge. Calling that 'the macro did
    not land' would be a finding manufactured from an empty denominator."""
    trajs = [_armed(f"t{i}", [_call("web_search", query="x")], "treatment")
             for i in range(20)]
    rep = ont.fs_batch_arm_uptake(trajs)
    assert rep["arms"]["treatment"].paths_calls == 0
    assert rep["readable"] is False
    out = ont.render_fs_batch_arms(rep)
    assert "NOT READABLE" in out
    assert "DID NOT LAND" not in out
    assert "NO OPPORTUNITY, not refusal" in out


def test_but_it_MUST_convict_once_the_opportunity_is_real():
    """⚠ OVER-SUPPRESSION GUARD. A check that can only ever say 'not readable'
    is furniture. Give the treatment arm ample eligible turns and no `paths`
    call, and it has to say so — that is the §4F verdict this exists to
    deliver."""
    trajs = [_armed(f"t{i}", _two_reads(), "treatment") for i in range(12)]
    rep = ont.fs_batch_arm_uptake(trajs)
    assert rep["arms"]["treatment"].eligible_turns == 12
    assert rep["readable"] is True
    out = ont.render_fs_batch_arms(rep)
    assert "MACRO DID NOT LAND" in out


def test_a_used_capability_reads_as_exercised_not_condemned():
    """The other side of the same guard: a `paths` call must flip the verdict,
    or the report convicts a working macro."""
    used = [_call("file_system", operation="read",
                  paths=["a.py", "b.py"], path="a.py")]
    trajs = ([_armed(f"e{i}", _two_reads(), "treatment") for i in range(12)]
             + [_armed("u", used, "treatment")])
    rep = ont.fs_batch_arm_uptake(trajs)
    assert rep["arms"]["treatment"].paths_calls == 1
    out = ont.render_fs_batch_arms(rep)
    assert "DID NOT LAND" not in out
    assert "being exercised" in out


def test_no_eligible_turns_anywhere_invents_no_traffic_estimate():
    """An unknown requirement is not an infinite one. With a zero eligibility
    rate the projection is None rather than a fabricated turn count."""
    rep = ont.fs_batch_arm_uptake(
        [_armed("t", [_call("web_search", query="x")], "control")])
    assert rep["eligibility_rate"] == 0
    assert rep["enrolled_turns_needed"] is None
    assert "would be needed" not in ont.render_fs_batch_arms(rep)


def test_projection_uses_the_CORPUS_rate_not_the_arm_sample():
    """The 'how much traffic' answer must not inherit the arm's own small-n
    noise — it is computed over every trajectory, enrolled or not."""
    trajs = [_armed("a", _two_reads(), "treatment")]
    trajs += [_traj(f"u{i}", _two_reads()) for i in range(9)]   # unenrolled
    rep = ont.fs_batch_arm_uptake(trajs)
    assert rep["corpus_turns"] == 10 and rep["corpus_eligible_turns"] == 10
    assert rep["eligibility_rate"] == 1.0
    assert rep["enrolled_turns_needed"] == ont.FS_BATCH_MIN_ELIGIBLE * 2


def test_an_unenrolled_corpus_says_so_rather_than_rendering_an_empty_table():
    rep = ont.fs_batch_arm_uptake([_traj("t", _two_reads())])
    assert rep["arms"] == {}
    assert "not enrolled" in ont.render_fs_batch_arms(rep)


def test_one_corrupt_record_does_not_stop_the_walk():
    bad = SimpleNamespace(id="bad", task_kind="user_request")  # no tool_calls
    rep = ont.fs_batch_arm_uptake([bad, _armed("t", _two_reads(), "treatment")])
    assert rep["arms"]["treatment"].eligible_turns == 1
