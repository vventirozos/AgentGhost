"""§4FD pins: the post-mortem engine selects labelled failures and reflects
contrastively.

The engine ran 156 idle cycles in six weeks and analysed nothing while 124
labelled failures accumulated: its severity is a blend of loop shapes, so a
confabulation or a human 👎 on a clean transcript scored ~0.01 against a 0.4
bar. Now a human label bypasses the bar and ranks first, and every analysed
failure is shown its nearest PASSING sibling.
"""
import asyncio

import pytest

from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.reflection import postmortem as pm
from ghost_agent.reflection.postmortem_prompts import build_postmortem_prompt, render_contrast


def _traj(i, outcome, request, n_calls=1, cluster="chat", ts="2099-01-01T00:00:00"):
    calls = [ToolCall(name="web_search", arguments={"query": request}, result="ok", duration_s=0.01)
             for _ in range(n_calls)]
    return Trajectory(id=f"t{i}", session_id="s", task_kind="user_request", cluster=cluster,
                      user_request=request, outcome=outcome, tool_calls=calls, n_steps=n_calls,
                      duration_s=30.0, timestamp=ts, final_response=f"reply {i}")


def test_human_labelled_failure_is_selected_despite_near_zero_severity():
    """World where it fails: the severity bar is applied before the human
    label is consulted, so the 👎 turn is dropped as it was for six weeks."""
    clean_thumbs_down = _traj(1, Outcome.FAILED.value, "what is my project codename")
    sig = pm.compute_signature(clean_thumbs_down)
    assert sig.severity < 0.1
    picked = pm.select_failed_runs([clean_thumbs_down], min_severity=0.4,
                                   human_labeled=lambda tid: tid == "t1")
    assert [t.id for t, _ in picked] == ["t1"]
    assert picked[0][1].human_labeled is True
    # without the human channel, the same turn is still invisible
    assert pm.select_failed_runs([clean_thumbs_down], min_severity=0.4) == []


def test_human_labelled_failure_outranks_a_loop_shaped_machine_failure():
    """World where it fails: ranking ignores the label, so a noisy machine
    refute with a high structural score crowds out the scarce human signal."""
    loopy = _traj(2, Outcome.FAILED.value, "loop loop loop", n_calls=40)
    thumbs = _traj(3, Outcome.FAILED.value, "tell me the codename")
    picked = pm.select_failed_runs([loopy, thumbs], limit=1, min_severity=0.0,
                                   human_labeled=lambda tid: tid == "t3")
    assert [t.id for t, _ in picked] == ["t3"]


def test_two_clean_thumbs_down_are_not_deduplicated_against_each_other():
    """World where it fails: two structurally identical clean turns hash
    alike and the second 👎 is filed as a duplicate of the first."""
    a = _traj(4, Outcome.FAILED.value, "codename please")
    b = _traj(5, Outcome.FAILED.value, "drive time to athens")
    picked = pm.select_failed_runs([a, b], limit=5, min_severity=0.4,
                                   human_labeled=lambda tid: True)
    assert sorted(t.id for t, _ in picked) == ["t4", "t5"]


def test_machine_failures_keep_the_severity_bar():
    """World where it fails: the bar bypass leaks to machine labels — every
    late refute (a quarter of them wrong) would mint a post-mortem lesson."""
    quiet = _traj(6, Outcome.FAILED.value, "quiet machine refute")
    assert pm.select_failed_runs([quiet], min_severity=0.4, human_labeled=lambda tid: False) == []


def test_passing_sibling_is_the_most_similar_passed_request():
    """World where it fails: the sibling search returns a failed turn, a
    different task kind, or an unrelated request below the floor."""
    fail = _traj(7, Outcome.FAILED.value, "what is my project codename right now")
    good = _traj(8, Outcome.PASSED.value, "remind me of my project codename")
    # §4FH: the failed decoy is STRICTLY more similar (identical content
    # tokens, Jaccard 1.0 vs 0.5) and listed FIRST — only the outcome filter
    # keeps it out. The first version tied at 0.5 and the pool order decided,
    # so deleting the outcome check survived the battery.
    other_fail = _traj(9, Outcome.FAILED.value, "what is my project codename right now?")
    unrelated = _traj(10, Outcome.PASSED.value, "weather in athens tomorrow")
    bench = _traj(11, Outcome.PASSED.value, "what is my project codename right now")
    bench.task_kind = "bench"
    sib = pm.find_passing_sibling(fail, [fail, other_fail, bench, good, unrelated])
    assert sib is not None and sib.id == "t8"
    assert pm.find_passing_sibling(fail, [fail, unrelated]) is None
    assert pm.find_passing_sibling(fail, [fail, other_fail]) is None


def test_prompt_carries_the_contrast_block_only_when_a_sibling_exists():
    """World where it fails: the template lost its {contrast} slot (format
    silently ignores the kwarg), or the block renders for None."""
    fail = _traj(12, Outcome.FAILED.value, "codename?")
    sib = _traj(13, Outcome.PASSED.value, "my codename?")
    sig = pm.compute_signature(fail)
    with_sib = build_postmortem_prompt(fail, sig, sibling=sib)
    assert "CONTRAST — a SIMILAR request that SUCCEEDED" in with_sib
    assert "my codename?" in with_sib and "reply 13" in with_sib
    assert with_sib.index("CONTRAST") < with_sib.index("Full transcript:")
    without = build_postmortem_prompt(fail, sig)
    assert "CONTRAST" not in without
    assert render_contrast(None) == ""


def test_engine_run_uses_the_human_channel_and_records_sibling_and_counts(tmp_path):
    """Executed end to end through `PostMortemEngine.run`: the human channel
    is consulted, the report carries the sibling id, and the summary line
    shows the new counts. World where it fails: `run` does not pass
    `human_labeled` to the selector, or the sibling never reaches the
    report."""
    prompts = []

    async def analyze(prompt):
        prompts.append(prompt)
        return ("CATEGORY: BEHAVIOURAL\nTITLE: Abstain on empty recall\n"
                "ROOT CAUSE: answered without evidence\nLESSON: say not found")

    q = pm.DefectQueue(tmp_path)
    eng = pm.PostMortemEngine(analyze, queue=q, min_severity=0.4,
                              human_labeled=lambda tid: tid == "t20")
    fail = _traj(20, Outcome.FAILED.value, "what is my project codename")
    sib = _traj(21, Outcome.PASSED.value, "remind me my project codename")
    rep = asyncio.run(eng.run(source=[fail, sib]))
    assert rep.selected == 1 and rep.human_selected == 1 and rep.contrasted == 1
    assert rep.analysed_ok == 1
    assert rep.reports[0].source_trajectory_ids == ["t20", "t21"]
    assert rep.reports[0].signature_hash == "human:t20"
    assert "CONTRAST" in prompts[0]
    assert "1 human-labelled, 1 contrasted" in rep.summary()
    # a second tick must not re-file the same human turn
    rep2 = asyncio.run(eng.run(source=[fail, sib]))
    assert rep2.selected == 0


def test_failed_analysis_of_a_human_turn_is_keyed_like_its_selection(tmp_path):
    """§4FH M1. World where it fails: the failed-analysis key is the
    structural hash, so a perma-failing 👎 is re-selected every tick (an LLM
    call each) and its shared structural hash blocks unrelated machine
    failures of the same shape."""
    calls = []

    async def analyze(prompt):
        calls.append(prompt)
        return None                          # unparseable / empty reply
    q = pm.DefectQueue(tmp_path)
    eng = pm.PostMortemEngine(analyze, queue=q, min_severity=0.0,
                              human_labeled=lambda tid: tid == "t30")
    human = _traj(30, Outcome.FAILED.value, "what is my project codename")
    machine = _traj(31, Outcome.FAILED.value, "please summarise the weather report")
    r1 = asyncio.run(eng.run(source=[human, machine]))
    assert r1.selected == 2 and r1.analysed_errors == 2
    r2 = asyncio.run(eng.run(source=[human, machine]))
    assert r2.selected == 0, "both failed analyses are excluded on the next tick"
    assert "human:t30" in eng._failed_analysis_sigs
    # the human turn's structural hash was NOT used as its exclusion key
    assert pm.compute_signature(human).hash not in eng._failed_analysis_sigs or \
        pm.compute_signature(human).hash == pm.compute_signature(machine).hash


def test_sibling_search_ignores_function_words_and_old_turns():
    """§4FH M2. World where it fails: "stop it please." pairs with "please run
    a healthcheck." on the token "please", or a success from months ago is
    offered as the contrast."""
    fail = _traj(40, Outcome.FAILED.value, "stop it please.")
    decoy = _traj(41, Outcome.PASSED.value, "please run a healthcheck.")
    assert pm.find_passing_sibling(fail, [fail, decoy]) is None
    fail2 = _traj(42, Outcome.FAILED.value, "what is my project codename right now")
    old = _traj(43, Outcome.PASSED.value, "remind me of my project codename", ts="2026-01-01T00:00:00")
    assert pm.find_passing_sibling(fail2, [fail2, old]) is None
    assert pm.find_passing_sibling(fail2, [fail2, old], max_age_days=0) is old


def test_human_labelled_report_carries_top_severity_for_the_queue(tmp_path):
    """§4FH m5. World where it fails: the report stores the structural score
    (~0.01) and the operator queue, sorted by severity, renders it LAST."""
    async def analyze(prompt):
        return "CATEGORY: BEHAVIOURAL\nTITLE: t\nROOT CAUSE: r\nLESSON: l"
    q = pm.DefectQueue(tmp_path)
    eng = pm.PostMortemEngine(analyze, queue=q, min_severity=0.4,
                              human_labeled=lambda tid: tid == "t50")
    rep = asyncio.run(eng.run(source=[_traj(50, Outcome.FAILED.value, "codename?")]))
    assert rep.reports[0].severity == 1.0


def test_main_wires_the_collectors_human_label_channel():
    """World where it fails: main.py constructs the engine without the
    channel, so production keeps analysing nothing."""
    import ast, inspect
    from ghost_agent import main as main_mod
    tree = ast.parse(inspect.getsource(main_mod))
    ctor = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and ast.unparse(n.func).endswith("PostMortemEngine")]
    assert len(ctor) == 1
    kw = next((k for k in ctor[0].keywords if k.arg == "human_labeled"), None)
    assert kw is not None
    val = ast.unparse(kw.value)
    # §4FH: the VALUE must fetch `has_human_label` off the trajectory collector
    assert "has_human_label" in val and "trajectory_collector" in val, val
