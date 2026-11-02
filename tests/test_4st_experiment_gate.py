"""§4ST — an experiment task closes DONE only on a probe_model run that really happened.

Request b45ec99f: three "Run Experiment … probe_model runs …" tasks were built
by the coding executor, which wrote the transcripts itself, and closed DONE.
The gate reads the transcripts probe_model keeps OUTSIDE the sandbox.
"""

from tests._refused import refused_ids, refused_reasons  # §4SU Phase 4
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import probe_transcripts as PT
from ghost_agent.core import project_advancer as pa
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.experiment_gate import (
    unrun_experiment_problem, unrun_experiment_check, quarantine, _named_text_files, MIN_WAIVER_CHARS,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from tests.test_4qa_project_task_integrity import _ctx, _EchoLLM
from tests.test_4ss_b45ec99f_fixes import LIVE_GOAL, EXP1, EXP2, EXP3, ANALYZE, PDF

FAKE_RUN = ("# Experiment 1\n\nEach prompt is run once with probe_model against the language model.\n\n## 3. Transcripts & Scores\n\n### L1-sky\n\n**Transcript:**\n"
            "> The sky is blue because of Rayleigh scattering.\n\nScore: SA-1\n")
REAL_RUN_TEXT = "# Model probe transcript\n\n## 1\nPROMPT: why is the sky blue\n\nANSWER: Rayleigh scattering.\n"
HONEST = "# Experiment 1\n\nData: results/runs/l1.md (probe_model).\n\n| layer | score |\n|---|---|\n| L1 | SA-1 |\n"


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _proj(store, goal=LIVE_GOAL):
    pid = store.create_project("3-Layer", goal=goal)
    return pid, Path(store.ensure_workspace(pid))


def _w(ws, rel, text):
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _keep(store, pid, ws, rel="results/runs/l1.md"):
    _w(ws, rel, REAL_RUN_TEXT)
    assert PT.record(store.memory_dir, pid, rel, REAL_RUN_TEXT)


# ───────────────────────────── the rule ─────────────────────────────

def test_the_live_fake_run_is_refused(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _w(ws, "results/exp1-layer-ladder.md", FAKE_RUN)
    p, claimed = unrun_experiment_check(store, pid, EXP1, files=["results/exp1-layer-ladder.md"])
    assert p and "no probe_model transcript has been recorded" in p
    assert claimed == ["results/exp1-layer-ladder.md"]


def test_a_write_up_citing_a_kept_transcript_closes(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    _w(ws, "results/exp1.md", HONEST + FAKE_RUN)        # shaped too: the citation is what counts
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"]) is None


@pytest.mark.parametrize("cite", ["results/runs/l1.md", "see l1.md", "/workspace/projects/x/results/runs/l1.md"])
def test_the_close_result_can_name_the_transcript(store, cite):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    assert unrun_experiment_problem(store, pid, EXP1, result_text=f"ran it — {cite}") is None


def test_the_delivered_transcript_itself_backs_the_task(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws, "results/exp1-layer-ladder.md")
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1-layer-ladder.md"]) is None


def test_a_kept_transcript_nobody_cites_does_not_back_a_fake(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    _w(ws, "results/exp1.md", FAKE_RUN)
    p = unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"])
    assert p and "1 probe transcript(s) are kept" in p and "nothing this task delivers cites one" in p


def test_a_sandbox_file_at_a_transcript_path_is_no_run(store):
    # only probe_model writes the host-side copy: a file merely NAMED like a run is nothing
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _w(ws, "results/runs/l1.md", REAL_RUN_TEXT)
    _w(ws, "results/exp1.md", HONEST)
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"])


def test_an_edited_kept_copy_backs_nothing(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    d = PT.project_dir(store.memory_dir, pid)
    for f in d.glob("*.md"):
        f.write_text("forged")                                    # bytes no longer hash to the name
    _w(ws, "results/exp1.md", HONEST)
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"])


def test_one_hop_an_analysis_citing_a_cited_write_up(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    _w(ws, "results/exp1.md", HONEST)
    _w(ws, "results/analysis.md", "## Transcripts analysed\n\nFrom results/exp1.md: L1 = SA-1.\n")
    assert unrun_experiment_problem(store, pid, "Analyze the experiment transcripts",
                                    files=["results/analysis.md"]) is None


def test_a_quarantined_run_is_no_hop(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    _w(ws, "simulated/results/exp1.md", HONEST)                  # moved aside: no data
    _w(ws, "results/analysis.md", "## Transcripts analysed\n\nFrom exp1.md (probe_model): L1 = SA-1.\n")
    assert unrun_experiment_problem(store, pid, "Analyze the results", files=["results/analysis.md"])


def test_a_task_the_words_miss_is_caught_by_what_it_wrote(store):
    # §4SS r2's documented miss: no model name, a script named → the words say "code"
    desc = "Run Experiment 3: repeat the ladder on the worker node (second model); scoring via results/analyze.py"
    pid, ws = _proj(store, goal="Study recursion")
    store.add_task(pid, desc)
    assert pa.is_model_experiment(desc, "Study recursion", [desc]) is False
    _w(ws, "results/exp3.md", FAKE_RUN)
    assert unrun_experiment_problem(store, pid, desc, files=["results/exp3.md"])
    _w(ws, "results/exp3b.md", "| layer | score |\n|---|---|\n| L1 | 1 |\n")
    assert unrun_experiment_problem(store, pid, desc, files=["results/exp3b.md"]) is None


#: what makes a transcript a MODEL's (r1 C1): the file names a model
_LLM = "The prompts went to Qwen.\n"


@pytest.mark.parametrize("text,shaped", [
    ("## Transcripts\n" + _LLM, True), ("### 3. Transcript & Scores\n" + _LLM, True),
    ("**Transcript:** x\n" + _LLM, True), ("Transcript: x\nrun with probe_model\n", True),
    ("> **Model response:** x\n", True), ("Model answers: x\n", True),       # a model label alone
    ("## Transcripts\n", False),                                 # an interview / meeting transcript
    ("## Meeting transcripts reviewed\nTranscript: Ana: hi\n", False),

    ("We planned to run probe_model on L1.\n", False),          # a protocol planning the runs
    ("The transcripts are analysed below.\n", False),            # prose, not a heading or label
    ("Answer: 42\n", False),
])
def test_what_reads_like_model_output(store, text, shaped):
    pid, ws = _proj(store, goal="Ship the onboarding flow")
    store.add_task(pid, "Write the notes")
    _w(ws, "notes.md", text)
    assert bool(unrun_experiment_problem(store, pid, "Write the notes", files=["notes.md"])) is shaped


def test_a_file_the_description_names_is_an_input_not_an_output(store):
    # replay: the PDF task names research/protocol.md (with a worked "Model Output:")
    pid, ws = _proj(store)
    store.add_task(pid, PDF)
    _w(ws, "research/protocol.md", "**Model Output:**\n> example\n")
    assert unrun_experiment_problem(store, pid, PDF) is None


def test_ordinary_work_is_untouched(store):
    pid, ws = _proj(store, goal="Ship the onboarding flow")
    store.add_task(pid, "Test the signup form")
    _w(ws, "report.md", "# Results\nAll green.\n")
    assert unrun_experiment_problem(store, pid, "Test the signup form", files=["report.md"]) is None


def test_only_transcript_claims_are_moved(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _w(ws, "results/a.md", FAKE_RUN)
    _w(ws, "results/b.md", "**Model output:**\n> example\n")
    p, claimed = unrun_experiment_check(store, pid, EXP1, files=["results/a.md", "results/b.md"])
    assert p and claimed == ["results/a.md"]


def test_unreadable_store_never_raises(store):
    class Boom:
        memory_dir = store.memory_dir

        def ensure_workspace(self, _pid):
            raise RuntimeError("no")
    assert unrun_experiment_check(Boom(), "p", EXP1) == (None, [])


# ───────────────────────────── quarantine ─────────────────────────────

def test_quarantine_moves_and_never_overwrites(store):
    pid, ws = _proj(store)
    _w(ws, "results/exp1.md", "first")
    _w(ws, "simulated/results/exp1.md", "older")
    assert quarantine(store, pid, ["results/exp1.md"]) == ["simulated/results/exp1.2.md"]
    assert (ws / "simulated/results/exp1.md").read_text() == "older"
    assert (ws / "simulated/results/exp1.2.md").read_text() == "first"
    assert not (ws / "results/exp1.md").exists()


def test_quarantine_refuses_links_escapes_and_its_own_folder(store, tmp_path):
    pid, ws = _proj(store)
    out = _w(tmp_path, "outside.md", "x")
    (ws / "link.md").symlink_to(out)
    _w(ws, "simulated/x.md", "x")
    assert quarantine(store, pid, ["link.md", "../outside.md", "simulated/x.md", "missing.md"]) == []
    assert out.exists() and (ws / "link.md").is_symlink()


def test_named_text_files_skips_the_quarantine(store):
    pid, ws = _proj(store)
    _w(ws, "simulated/results/exp1.md", "x")
    _w(ws, "results/exp1.md", "x")
    assert _named_text_files(ws, ["exp1.md"]) == ["results/exp1.md"]


# ───────────────────────────── the owner's task_update ─────────────────────────────

async def _ctx_project(tmp_path, desc, goal=LIVE_GOAL):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="3-Layer", goal=goal)
    pid = ctx.current_project_id
    tid = json.loads(await tool_manage_projects(ctx, action="task_decompose", subtasks=[desc]))["created"][0]
    return ctx, store, pid, tid, Path(store.ensure_workspace(pid))


async def _close(ctx, tid, **kw):
    return await tool_manage_projects(ctx, action="task_update", task_id=tid, status="done", **kw)


async def test_task_update_refuses_an_unrun_experiment(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    _w(ws, "results/exp1-layer-ladder.md", FAKE_RUN)
    raw = await _close(ctx, tid, result="ran all five", deliverables=["results/exp1-layer-ladder.md"])
    out = json.loads(str(raw))
    assert refused_ids(out, "unrun_experiments") == [tid] and store.get_task(tid)["status"] != "DONE"
    assert "model_run" in out["agent_instruction_refused"]
    assert getattr(raw, "status", None) is not None and raw.status.name == "REJECTED"


async def test_task_update_closes_an_experiment_that_ran(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    _keep(store, pid, ws)
    _w(ws, "results/exp1-layer-ladder.md", HONEST)
    out = json.loads(str(await _close(ctx, tid, result="done", deliverables=["results/exp1-layer-ladder.md"])))
    assert not refused_ids(out, "unrun_experiments") and store.get_task(tid)["status"] == "DONE"


async def test_a_real_probe_run_end_to_end_backs_the_close(tmp_path):
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    ctx.llm_client = _EchoLLM()
    ctx.sandbox_dir = ws
    ctx.conversation_key = "c"
    out = await tool_probe_model(ctx, prompts=["why is the sky blue"], save_to="results/runs/l1.md")
    assert "Saved the transcript" in str(out)
    _w(ws, "results/exp1-layer-ladder.md", "Data: results/runs/l1.md\n")
    res = json.loads(str(await _close(ctx, tid, result="done", deliverables=["results/exp1-layer-ladder.md"])))
    assert store.get_task(tid)["status"] == "DONE", res


@pytest.mark.parametrize("reason", ["", "n/a", "x" * (MIN_WAIVER_CHARS - 1)])
async def test_a_short_waiver_is_no_waiver(tmp_path, reason):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    out = json.loads(str(await _close(ctx, tid, result="r", not_model_data=reason)))
    assert refused_ids(out, "unrun_experiments") == [tid]


async def test_a_waiver_closes_and_is_never_silent(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    why = "the protocol only describes the runs; nothing was asked of a model"
    out = json.loads(str(await _close(ctx, tid, result="wrote the protocol", not_model_data=why)))
    t = store.get_task(tid)
    assert t["status"] == "DONE" and out["experiment_gate_waived"] == [tid]
    assert f"[not model data: {why}]" in t["result_summary"]
    logs = store.recent_work_logs(pid, limit=20)
    assert any("experiment gate waived" in str(r) and why in str(r) for r in logs)


async def test_one_waiver_never_covers_a_bulk_close(tmp_path):
    # r1: one sentence waived three fabricated runs in one call
    ctx, store, pid, t1, ws = await _ctx_project(tmp_path, EXP1)
    t2 = json.loads(await tool_manage_projects(ctx, action="task_decompose", subtasks=[EXP2]))["created"][0]
    why = "both only describe the planned runs"
    out = json.loads(str(await tool_manage_projects(ctx, action="task_update", task_ids=[t1, t2], status="done",
                                                    result="r", not_model_data=why)))
    assert sorted(refused_ids(out, "unrun_experiments")) == sorted([t1, t2])
    assert "task_id, not task_ids" in out["agent_instruction_refused"]
    for t in (t1, t2):
        assert store.get_task(t)["status"] != "DONE"
        assert "[not model data:" not in (store.get_task(t)["result_summary"] or "")
    await _close(ctx, t1, result="r", not_model_data=why)
    assert store.get_task(t1)["result_summary"].count("[not model data:") == 1


async def test_a_rewrite_cannot_drop_the_words_the_gate_reads(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    _w(ws, "notes/x.md", "Plain notes about the layers, written as ordinary prose for the reader here.\n")
    out = json.loads(str(await _close(ctx, tid, result="r", description="Write notes → notes/x.md")))
    assert refused_ids(out, "unrun_experiments") == [tid]


async def test_a_re_close_of_a_done_task_is_not_gated(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    why = "the protocol only describes the runs; nothing was asked of a model"
    await _close(ctx, tid, result="r", not_model_data=why)
    out = json.loads(str(await _close(ctx, tid, result="again")))
    assert not refused_ids(out, "unrun_experiments")


# ───────────────────────────── autoadvance ─────────────────────────────

async def test_an_owner_build_of_a_word_missed_experiment_fails_and_moves_the_fake(tmp_path):
    desc = "Run Experiment 3: repeat the ladder on the worker node (second model); scoring via results/analyze.py"
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal="Study recursion")
    tid = store.add_task(pid, desc)
    ws = Path(store.ensure_workspace(pid))

    async def builder(*_a, **_k):
        _w(ws, "results/exp3.md", FAKE_RUN)
        _w(ws, "results/analyze.py", "print(1)\n")
        return CodingResult(ok=True, summary="ran the ladder", files=["results/exp3.md", "results/analyze.py"],
                            verified=True)

    async def runner(name, args):
        return "OK"

    async def cls(_d):
        return "coding"
    res = await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                                coding_executor=builder, owner_requested=True)
    t = store.get_task(tid)
    assert t["status"] == "FAILED" and "probe_model" in t["failure_reason"]
    assert "simulated/results/exp3.md" in t["failure_reason"]
    assert (ws / "simulated/results/exp3.md").exists() and not (ws / "results/exp3.md").exists()
    assert (ws / "results/analyze.py").exists()                  # code stays
    arts = [a.get("payload") for a in store.list_artifacts(project_id=pid)]
    assert "results/exp3.md" not in arts


async def test_a_build_that_cites_its_probe_run_closes(tmp_path):
    desc = "Run Experiment 3: repeat the ladder on the worker node (second model); scoring via results/analyze.py"
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal="Study recursion")
    tid = store.add_task(pid, desc)
    ws = Path(store.ensure_workspace(pid))
    _keep(store, pid, ws, "results/runs/exp3-worker.md")

    async def builder(*_a, **_k):
        _w(ws, "results/exp3.md", "## Transcripts\nSee results/runs/exp3-worker.md\n")
        return CodingResult(ok=True, summary="scored", files=["results/exp3.md"], verified=True)

    async def runner(name, args):
        return "OK"

    async def cls(_d):
        return "coding"
    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          coding_executor=builder, owner_requested=True)
    assert store.get_task(tid)["status"] == "DONE"


def test_the_experiment_stop_tells_the_model_the_close_is_checked():
    from ghost_agent.tools.projects import _advance_batch_instruction
    batch = SimpleNamespace(stop_reason="model_experiment", count=1, ran=0, advanced=[
        {"task_id": "t1", "summary": "model experiment: Run Experiment 1", "status": "PENDING"}],
        not_run=["t1"], project_done=False, held=[], gated=[])
    msg = _advance_batch_instruction(batch, None)
    assert "transcript's path" in msg
    assert '"model_run"' in msg and "override" in msg     # §4SU Phase 4: the waiver is an override


def test_the_schema_offers_the_waiver():
    # §4SU Phase 4: the waiver is `override={"model_run": why}` (the schema's
    # own param was dropped); the old kwarg is still accepted
    from ghost_agent.tools.projects import tool_manage_projects as T, MANAGE_PROJECTS_TOOL_DEF as D
    import inspect
    assert "not_model_data" in inspect.signature(T).parameters
    props = D["function"]["parameters"]["properties"]
    assert "not_model_data" not in props
    assert "model_run" in props["override"]["description"]


WORD_MISS = "Run Experiment 3: repeat the ladder on the worker node (second model); scoring via results/analyze.py"


def test_a_file_the_result_names_is_read(store):
    pid, ws = _proj(store, goal="Study recursion")
    store.add_task(pid, WORD_MISS)
    _w(ws, "results/exp3.md", FAKE_RUN)
    assert unrun_experiment_problem(store, pid, WORD_MISS, result_text="wrote results/exp3.md")


def test_the_declared_file_is_read(store):
    desc = "Repeat the ladder on the worker node → results/exp3.md"
    pid, ws = _proj(store, goal="Study recursion")
    store.add_task(pid, desc)
    _w(ws, "results/exp3.md", FAKE_RUN)
    assert unrun_experiment_problem(store, pid, desc)


def test_a_link_inside_is_read_a_link_out_is_not(store, tmp_path):
    pid, ws = _proj(store, goal="Study recursion")
    store.add_task(pid, WORD_MISS)
    _w(ws, "results/real.md", FAKE_RUN)
    (ws / "results/link.md").symlink_to(ws / "results/real.md")
    assert unrun_experiment_problem(store, pid, WORD_MISS, files=["results/link.md"])
    out = _w(tmp_path, "outside.md", FAKE_RUN)
    (ws / "results/out.md").symlink_to(out)
    assert unrun_experiment_problem(store, pid, WORD_MISS, files=["results/out.md"]) is None


async def test_the_rewrite_s_declared_file_is_read(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, WORD_MISS, goal="Study recursion")
    _w(ws, "results/exp3.md", FAKE_RUN + "More words here so the document gate is satisfied by length.\n")
    out = json.loads(str(await _close(ctx, tid, result="r", description="Score the ladder → results/exp3.md")))
    assert refused_ids(out, "unrun_experiments") == [tid]


async def test_a_waiver_a_later_gate_refuses_logs_nothing(tmp_path):
    ctx, store, pid, tid, ws = await _ctx_project(tmp_path, EXP1)
    t2 = store.add_task(pid, EXP2, postconditions=["[HUMAN_GATE: owner approval]"])
    why = "the protocol only describes the runs; nothing was asked of a model"
    out = json.loads(str(await _close(ctx, t2, result="r", not_model_data=why)))
    assert store.get_task(t2)["status"] != "DONE"
    assert "experiment_gate_waived" not in out
    assert not any("experiment gate waived" in str(r) for r in store.recent_work_logs(pid, limit=20))


async def _research_tick(tmp_path, owner, monkeypatch):
    """A research close whose brief is written up as a model run (the task
    words miss: only the shape triggers)."""
    from ghost_agent.core import project_research as PR
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal="Study recursion")
    tid = store.add_task(pid, "Repeat the ladder on the worker node and write up what it said")

    async def persist(context, project_id, topic, output, **kw):
        ws = Path(store.ensure_workspace(project_id))
        _w(ws, "research/ladder.md", FAKE_RUN)
        return PR.ResearchResult(ok=True, topic=topic, path="research/ladder.md", summary="the ladder run")
    monkeypatch.setattr(PR, "persist_research_from_output", persist)

    async def runner(name, args):
        return "Result 1: a page about recursion. Result 2: another page. " * 10

    async def cls(_d):
        return "research"
    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          coding_executor=None, owner_requested=owner)
    ws = Path(store.ensure_workspace(pid))
    return store.get_task(tid), ws


async def test_an_owner_research_write_up_of_a_run_fails(tmp_path, monkeypatch):
    t, ws = await _research_tick(tmp_path, True, monkeypatch)
    assert t["status"] == "FAILED" and "probe_model" in (t["failure_reason"] or "")
    # r1 M2: nothing is moved on this path (a moved brief could never close)
    assert (ws / "research/ladder.md").is_file() and not (ws / "simulated").exists()


async def test_an_unattended_research_write_up_of_a_run_is_held(tmp_path, monkeypatch):
    t, _ws = await _research_tick(tmp_path, False, monkeypatch)
    assert t["status"] == "NEEDS_USER" and "probe_model" in (t["result_summary"] or "")


# ─────────────────────── review r1 (fresh eye) ───────────────────────

@pytest.mark.parametrize("rel,text", [
    ("src/indexer.py", "# load the transcripts from disk (the language model reads them)\n"),
    ("src/model.py", "class Row:\n    transcript: str = ''   # Qwen output\n"),
    ("conf/app.yaml", "transcript: on\nmodel: qwen\n"),
])
def test_source_and_config_are_never_shaped(store, rel, text):
    # r1 C1: a code comment read as a heading, a typed field as a label — and
    # the build's source was quarantined
    pid, ws = _proj(store, goal="Build a CLI that indexes meeting audio")
    store.add_task(pid, "Build the indexer")
    _w(ws, rel, text)
    assert unrun_experiment_check(store, pid, "Build the indexer", files=[rel]) == (None, [])


def test_a_file_named_only_in_the_result_is_judged_never_moved(store):
    # r1 C2: the owner's input, named in the result, went to simulated/
    pid, ws = _proj(store, goal="Study recursion")
    store.add_task(pid, WORD_MISS)
    _w(ws, "inbox/run.md", FAKE_RUN)
    problem, claimed = unrun_experiment_check(store, pid, WORD_MISS, result_text="read inbox/run.md")
    assert problem and claimed == []
    problem, claimed = unrun_experiment_check(store, pid, WORD_MISS, files=["inbox/run.md"])
    assert problem and claimed == ["inbox/run.md"]


def test_a_declared_only_file_is_judged_never_moved(store):
    desc = "Repeat the ladder on the worker node → results/exp3.md"
    pid, ws = _proj(store, goal="Study recursion")
    store.add_task(pid, desc)
    _w(ws, "results/exp3.md", FAKE_RUN)
    problem, claimed = unrun_experiment_check(store, pid, desc)
    assert problem and claimed == []


def test_a_fake_over_a_kept_transcript_s_path_is_refused(store):
    # r1 C3: the path matched, the bytes were the executor's
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws, rel="results/exp1.md")
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"]) is None
    _w(ws, "results/exp1.md", FAKE_RUN)
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"])


def test_a_gate_crash_is_logged_at_warning(store, monkeypatch, caplog):
    # r1 M4: production logs INFO — a debug line hid a gate that was off
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    from ghost_agent.core import probe_transcripts as _PT

    def boom(*a, **k):
        raise RuntimeError("index unreadable")
    monkeypatch.setattr(_PT, "recorded", boom)
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        assert unrun_experiment_check(store, pid, EXP1) == (None, [])
    assert any(r.levelno == logging.WARNING and "model run gate" in r.getMessage() for r in caplog.records)


def test_quarantine_renames_the_file_not_its_folder(store):
    pid, ws = _proj(store)
    _w(ws, "simulated/results/v1.0/notes", "old")
    _w(ws, "results/v1.0/notes", "new")
    assert quarantine(store, pid, ["results/v1.0/notes"]) == ["simulated/results/v1.0/notes.2"]
    _w(ws, "simulated/results/v1.0/a.md", "old")
    _w(ws, "results/v1.0/a.md", "new")
    assert quarantine(store, pid, ["results/v1.0/a.md"]) == ["simulated/results/v1.0/a.2.md"]


def test_the_advancer_runs_the_gate_off_the_event_loop():
    # r1 M3 (an R1 enumeration): every advancer use of the gate is an argument
    # of asyncio.to_thread, never a direct call
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(pa))
    # §4SU Phase 1: the research close runs the gate inside check_close
    gate = {"_experiment_gate_build", "unrun_experiment_problem", "unrun_experiment_check"}
    _cc_threaded = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and getattr(n.func, "attr", "") == "to_thread" and n.args
                       and isinstance(n.args[0], ast.Name) and n.args[0].id == "check_close")
    assert _cc_threaded == 1
    seen = 0
    inner = {id(n) for fn in ast.walk(tree)
             if isinstance(fn, ast.FunctionDef) and fn.name == "_experiment_gate_build"
             for n in ast.walk(fn)}                      # it runs IN the thread
    assert inner
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and id(node) not in inner:
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            assert name not in gate, f"direct call of {name} at line {node.lineno}"
            if name == "to_thread" and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in gate:
                seen += 1
    assert seen == 2        # the build gate + the §4SU Phase 5 held-task re-check


def test_the_close_s_words_alone_can_cite_the_run(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    _w(ws, "results/exp1.md", "# Experiment 1\n\n| layer | score |\n|---|---|\n| L1 | SA-1 |\n")
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"])
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"],
                                    result_text="scored from results/runs/l1.md") is None


def test_quarantine_never_moves_a_link(store):
    pid, ws = _proj(store)
    _w(ws, "results/real.md", FAKE_RUN)
    (ws / "results/link.md").symlink_to(ws / "results/real.md")
    assert quarantine(store, pid, ["results/link.md"]) == []
    assert (ws / "results/link.md").is_symlink()


def test_a_refused_build_drops_the_moved_files_from_its_list(store):
    pid, ws = _proj(store)
    tid = store.add_task(pid, EXP1)
    _w(ws, "results/exp1.md", FAKE_RUN)
    _w(ws, "results/score.py", "print(1)\n")
    cres = CodingResult(ok=True, files=["results/exp1.md", "results/score.py"], summary="ran it")
    out = pa._experiment_gate_build(store, pid, SimpleNamespace(id=tid, description=EXP1), cres)
    assert out.ok is False and out.files == ["results/score.py"]
    assert (ws / "simulated/results/exp1.md").is_file()


# ─────────────────────── review r2 (fresh eye) ───────────────────────

def test_a_quarantined_run_is_never_a_build_input(store):
    # r2 M1: simulated/results/exp1.md was fed back as the analysis' data
    from ghost_agent.core.experiment_gate import QUARANTINE_DIR
    assert pa._QUARANTINE_DIR == QUARANTINE_DIR
    pid, ws = _proj(store)
    _w(ws, "simulated/results/exp1.md", FAKE_RUN)
    _w(ws, "simulated/research/brief.md", "# brief\n" + "words " * 40)
    _w(ws, "results/notes.md", "real notes\n")
    files = pa._gather_project_files(store, pid, description="Analyze the per-layer scores in results/exp1.md → results/analysis.md")
    assert not any(k.startswith("simulated/") for k in files) and "results/notes.md" in files
    assert not any(k.startswith("simulated/") for k in pa._gather_research_briefs(store, pid))


def test_a_heading_alone_never_moves_a_file(store):
    # r2 M2: a protocol's "## Transcripts" section moved its own declared file
    pid, ws = _proj(store)
    desc = "Write the experiment protocol → protocol.md"
    store.add_task(pid, desc)
    _w(ws, "protocol.md", "# Protocol\n\n## Transcripts\n\nEach run is made with probe_model(save_to=…).\n")
    problem, claimed = unrun_experiment_check(store, pid, desc, files=["protocol.md"])
    assert claimed == []
    _w(ws, "results/exp1.md", FAKE_RUN)
    assert unrun_experiment_check(store, pid, EXP1, files=["results/exp1.md"])[1] == ["results/exp1.md"]


def test_a_dot_slash_path_is_the_same_file(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws, rel="results/exp1.md")
    assert unrun_experiment_problem(store, pid, EXP1, files=["./results/exp1.md"]) is None


def test_the_close_s_words_cite_a_run_whose_sandbox_copy_is_gone(store):
    # the kept copy outside the sandbox is the proof — the sandbox file may be deleted
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _keep(store, pid, ws)
    (ws / "results/runs/l1.md").unlink()
    _w(ws, "results/exp1.md", "# Experiment 1\n\n| layer | score |\n|---|---|\n| L1 | SA-1 |\n")
    assert unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"],
                                    result_text="scored from results/runs/l1.md") is None


def test_a_build_gate_crash_is_logged_at_warning(store, monkeypatch, caplog):
    from ghost_agent.core import experiment_gate as EG

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(EG, "unrun_experiment_check", boom)
    pid, ws = _proj(store)
    cres = CodingResult(ok=True, files=[], summary="s")
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        assert pa._experiment_gate_build(store, pid, SimpleNamespace(id="t", description=EXP1), cres) is cres
    assert any(r.levelno == logging.WARNING and "model run gate" in r.getMessage() for r in caplog.records)


def test_quarantine_moves_the_record_and_no_recovery_promotes_it(store):
    # audit (storage): the moved fake stayed a registered deliverable, and the
    # DONE sweep's recovery registered simulated/… as built work
    from ghost_agent.core import workspace_cleanup as WC
    from ghost_agent.core.experiment_gate import QUARANTINE_DIR
    assert WC._QUARANTINE_DIR == QUARANTINE_DIR
    pid, ws = _proj(store)
    tid = store.add_task(pid, EXP1)
    _w(ws, "results/exp1.md", FAKE_RUN)
    store.register_file_artifact(tid, "results/exp1.md")
    assert quarantine(store, pid, ["results/exp1.md"]) == ["simulated/results/exp1.md"]
    listed = lambda: [str(a) for a in store.list_deliverables(pid)]
    assert not any("exp1.md" in a for a in listed())
    WC.sweep_project_workspace(store, pid)
    assert (ws / "simulated/results/exp1.md").is_file()
    assert not any("simulated" in a for a in listed())


def test_the_refusal_names_the_probe_call_to_make(store):
    pid, ws = _proj(store)
    store.add_task(pid, EXP1)
    _w(ws, "results/exp1.md", FAKE_RUN)
    problem = unrun_experiment_problem(store, pid, EXP1, files=["results/exp1.md"])
    assert 'probe_model(save_to="results/' in problem


def test_a_partial_recovery_never_registers_a_quarantined_file(store):
    from ghost_agent.core import workspace_cleanup as WC
    pid, ws = _proj(store)
    tid = store.add_task(pid, EXP1)
    _w(ws, "results/analysis.md", "real\n")
    store.register_file_artifact(tid, "results/analysis.md")          # a PARTIAL keep-set
    _w(ws, "simulated/results/exp1.md", FAKE_RUN)
    WC.sweep_project_workspace(store, pid)
    assert (ws / "simulated/results/exp1.md").is_file()
    assert not any("simulated" in str(a) for a in store.list_deliverables(pid))


async def test_an_owner_research_refusal_is_labelled_by_its_gate(tmp_path, monkeypatch):
    # §4SU Phase 1 r1 L3: a model_run refusal was logged as postcondition_failed
    t, _ws = await _research_tick(tmp_path, True, monkeypatch)
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ev = store.latest_task_event(t["project_id"], t["id"], ("autoadvance_step_failed",))
    assert ev["payload"].get("refused_gate") == "model_run"
    assert not ev["payload"].get("postcondition_failed")
