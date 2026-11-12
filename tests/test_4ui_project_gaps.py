"""§4UI (2026-11-12) — project workflow gaps found by the review of req 8646c6c8
and the live logs (10-07..11-12)."""

import pytest

from ghost_agent.core import project_advancer as pa
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.memory.projects import ProjectStore
from tests.test_4qa_project_task_integrity import GOAL, _ctx


# ── autoadvance carries on past an experiment, never onto its data ──────────

async def test_a_batch_past_an_experiment_holds_the_later_undeclared_tasks(tmp_path):
    """Req c6b9f65d: a flat plan, the batch skipped E1 and then built and
    closed the analysis and findings on zero transcripts."""
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal=GOAL)
    t0 = store.add_task(pid, "Set up the repo → README.md")
    exp = store.add_task(pid, "Task 3.1: Run Layer 0 — Base State → results/layer0-base.md")
    later = store.add_task(pid, "Write the analysis → analyze.py")              # no depends_on
    indep = store.add_task(pid, "Write the glossary → glossary.md", depends_on=[t0])
    built = []

    async def runner(name, args):
        return "OK"

    from pathlib import Path
    ws = Path(store.ensure_workspace(pid))

    async def builder(ctx, desc, **_k):
        built.append(desc)
        f = desc.split("→")[-1].strip()
        (ws / f).parent.mkdir(parents=True, exist_ok=True)
        (ws / f).write_text("# " + desc + "\n\nReal content written by the build for this task.\n")
        return CodingResult(ok=True, summary="wrote it", files=[f], verified=True)

    async def cls(_d):
        return "coding"

    batch = await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=None, tool_runner=runner,
                                  llm_classifier=cls, coding_executor=builder, owner_requested=True)
    assert batch.stop_reason == "model_experiment"
    assert store.get_task(exp)["status"] == "PENDING"
    assert store.get_task(later)["status"] == "PENDING"         # waits for the experiment
    assert not any("analysis" in d for d in built)
    assert any("glossary" in d for d in built)                  # declared independent → ran


def test_the_shadow_is_only_later_undeclared_peers(tmp_path):
    store = ProjectStore(tmp_path / "m2", sandbox_root=tmp_path / "sb2")
    pid = store.create_project("P", goal="g")
    before = store.add_task(pid, "before")
    exp = store.add_task(pid, "experiment")
    after = store.add_task(pid, "after")
    declared = store.add_task(pid, "declared", depends_on=[before])
    assert pa._waits_on_experiments(store, pid, []) == set()
    assert pa._waits_on_experiments(store, pid, [exp]) == {exp, after}
    assert declared not in pa._waits_on_experiments(store, pid, [exp])


from tests.test_4su_phase0 import _project, _done, owner_turn  # noqa: E402,F401


# ── a changed plan may retract a declared file, with a recorded reason ──────

async def test_a_declared_file_is_retracted_with_a_recorded_reason(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the report → out/report.md")
    out = await _done(ctx, tid, result="folded into README", description="Fold the report into README",
                      override={"deliverable": "the report became a README section, no separate file"})
    assert store.get_task(tid)["status"] == "DONE", out
    assert "→" not in store.get_task(tid)["description"]          # no re-appended promise
    assert any(o["gate"] == "deliverable" for o in out.get("overrides") or []), out
    assert "OVERRIDE deliverable" in store.get_ledger(pid)


async def test_a_retraction_outside_the_closing_call_keeps_the_promise(tmp_path, owner_turn):
    """r1 MAJOR 2: a path-less rewrite + deliverable override in a call that
    does not close dropped the "→ path" with nothing recorded; the next DONE
    then had no file to check."""
    from ghost_agent.tools.projects import tool_manage_projects
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the report → out/report.md")
    await tool_manage_projects(ctx, action="task_update", task_id=tid, description="Fold the report into README",
                               override={"deliverable": "the report became a README section, no separate file"})
    assert "→ out/report.md" in store.get_task(tid)["description"]
    out = await _done(ctx, tid, result="done")
    assert store.get_task(tid)["status"] != "DONE"
    assert any(r["gate"] == "deliverable" for r in out.get("refused") or []), out


async def test_a_short_retraction_reason_is_refused(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the report → out/report.md")
    out = await _done(ctx, tid, result="x", override={"deliverable": "nah"})
    assert store.get_task(tid)["status"] != "DONE"
    assert any(r["gate"] == "deliverable" for r in out.get("refused") or []), out


async def test_the_deliverable_refusal_names_the_retraction():
    from ghost_agent.core.task_close import gate_deliverable
    r = gate_deliverable("/nonexistent-ws", "Write the report → out/report.md")
    assert r is not None and 'override={"deliverable"' in r.fix


# ── a constraint override is recorded (§4UK: no approval), never a skipped audit ──

async def _constrained(tmp_path, owner_turn, monkeypatch, verdict=(False, "uses an external library")):
    from pathlib import Path
    owner_turn("a1b2c3d4")
    ctx, store, pid, tid = await _project(tmp_path, desc="Write the parser → parser.py")
    store.update_project(pid, metadata={"constraints": ["no external libraries"]})
    (Path(store.ensure_workspace(pid)) / "parser.py").write_text("import lark\n\ndef parse(s):\n    return s\n")
    calls = []

    async def gate(_ctx, _c, _files, **_k):
        calls.append(1)
        return verdict
    monkeypatch.setattr("ghost_agent.core.build_gates.constraint_gate", gate)
    return ctx, store, pid, tid, calls


async def test_an_unattended_override_is_refused_and_not_parked(tmp_path, owner_turn, monkeypatch):
    ctx, store, pid, tid, _ = await _constrained(tmp_path, owner_turn, monkeypatch)
    owner_turn("sched-1234")
    out = await _done(ctx, tid, result="CONSTRAINT-OVERRIDE: lark is the only parser that fits")
    assert store.get_task(tid)["status"] not in ("DONE", "NEEDS_USER")
    assert any(r["gate"] == "constraints" for r in out["refused"])


async def test_an_override_over_a_passing_audit_records_nothing(tmp_path, owner_turn, monkeypatch):
    ctx, store, pid, tid, _ = await _constrained(tmp_path, owner_turn, monkeypatch, verdict=(True, ""))
    out = await _done(ctx, tid, result="wrote it", override={"constraints": "just in case it is needed"})
    assert store.get_task(tid)["status"] == "DONE"
    assert not out.get("overrides") and not store.list_events(pid, event_type="close_override")


# ── a write-up the plan declared is not a model run by its words ────────────

def _close(store, pid, tid, **kw):
    from ghost_agent.core import task_close as TC
    from ghost_agent.core.planning import ProjectPlan
    return TC.check_close(store, pid, ProjectPlan(store, pid).tree, tid, actor="model", **kw)


def test_a_planned_write_up_about_experiments_closes(tmp_path):
    from pathlib import Path
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="Evaluate Qwen on GSM8K")
    tid = store.add_task(pid, "Write how to run the 50 GSM8K prompts on Qwen and record its answers"
                              " → docs/protocol.md", kind="write")
    ws = Path(store.ensure_workspace(pid))
    (ws / "docs").mkdir()
    (ws / "docs/protocol.md").write_text("# Protocol\n\n" + "Each GSM8K prompt is sent once at temperature zero, "
                                         "the final number is parsed and compared with the reference. " * 4 + "\n")
    assert _close(store, pid, tid, result="wrote the protocol", files=["docs/protocol.md"]).ok


@pytest.mark.parametrize("desc", [
    "Run the 4-layer prompts on Gemma and record its answers → results/answers.json",   # not prose
    "Run the 50 GSM8K prompts on Qwen and record its answers",                          # no file
])
def test_a_planned_write_that_delivers_no_prose_is_still_judged_by_its_words(tmp_path, desc):
    """r1 MAJOR 1: kind=write/research skipped the word test whatever it
    delivered — a .json of "answers", or only the result text, closed with no
    transcript and no waiver."""
    from pathlib import Path
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="Evaluate models on prompts")
    tid = store.add_task(pid, desc, kind="write")
    ws = Path(store.ensure_workspace(pid))
    (ws / "results").mkdir()
    (ws / "results/answers.json").write_text('{"answers": ["4", "7"]}')
    files = ["results/answers.json"] if ".json" in desc else []
    v = _close(store, pid, tid, result="recorded the answers", files=files)
    assert v.first and v.first.gate == "model_run", v


def test_the_advancer_holds_a_planned_write_without_a_prose_file(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    t1 = store.add_task(pid, "Run the prompts on Gemma → results/answers.json", kind="write")
    t2 = store.add_task(pid, "Write the methods → docs/methods.md", kind="write")
    nodes = {t["id"]: t for t in store.list_tasks(pid)}
    from types import SimpleNamespace as N
    assert pa._planned_not_experiment(store, pid, N(id=t1, kind="write", description=nodes[t1]["description"])) is False
    assert pa._planned_not_experiment(store, pid, N(id=t2, kind="write", description=nodes[t2]["description"])) is True


def test_a_retype_to_write_by_the_model_is_not_trusted(tmp_path, owner_turn):
    import asyncio
    from ghost_agent.tools.projects import tool_manage_projects
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    asyncio.run(tool_manage_projects(ctx, action="create", title="Q", goal="Evaluate Qwen on GSM8K"))
    pid = ctx.current_project_id
    tid = store.add_task(pid, "Run the 50 GSM8K prompts on Qwen and record its answers")
    asyncio.run(tool_manage_projects(ctx, action="task_update", task_id=tid, task_kind="write"))
    assert store.get_task(tid)["kind"] == "write"
    assert _close(store, pid, tid, result="done").first.gate == "model_run"


def test_a_planned_write_up_that_writes_a_transcript_is_still_refused(tmp_path):
    from pathlib import Path
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    tid = store.add_task(pid, "Write the summary → results/l1.md", kind="write")
    ws = Path(store.ensure_workspace(pid))
    (ws / "results").mkdir()
    (ws / "results/l1.md").write_text("# Run\n\n## Transcript\n\nModel response: the sky is blue (Qwen said).\n")
    v = _close(store, pid, tid, result="wrote it", files=["results/l1.md"])
    assert v.first and v.first.gate == "model_run"


# ── r1 M1: with the one-task stop off, nothing says "task complete, I stopped" ─

async def test_a_forced_final_after_a_close_is_not_called_a_designed_stop(tmp_path, monkeypatch):
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_NOTE
    from ghost_agent.core import agent as A
    from tests.test_4qa_project_task_integrity import _closed_then
    assert A._CLOSE_ENDS_TURN is False
    first = "The notes are collected: 12 items saved to notes.md, grouped by topic."
    reply, n = await _closed_then(tmp_path, monkeypatch, first,
                                  lambda r, t: r("", [t("t1", "manage_projects", {"action": "task_next"})]))
    # (_closed_then forces the final right after the close — standing in
    # for a budget / breaker / deadline final)
    assert TASK_CLOSED_NOTE not in reply and "say *proceed*" not in reply, reply
    assert "finalized before" in reply, reply          # the dropped call is said honestly


@pytest.mark.parametrize("decl,files,result", [
    ("results/answers.html", ["results/answers.html"], "wrote the table"),          # tags hide labels
    ("docs/notes.md", ["docs/notes.md"], "answers are in results/answers.json"),   # result-named file
    ("docs/notes.md", ["docs/notes.md"], "ran 40 prompts, answers in results/answers.jsonl"),  # r3: any suffix
])
def test_a_planned_write_with_a_non_plain_text_candidate_is_judged_by_its_words(tmp_path, decl, files, result):
    """review r2 m1."""
    from pathlib import Path
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P", goal="Evaluate models")
    tid = store.add_task(pid, f"Run the 50 GSM8K prompts on Qwen and record its answers → {decl}", kind="write")
    ws = Path(store.ensure_workspace(pid))
    for d in ("docs", "results"):
        (ws / d).mkdir()
    body = "Notes on the run. " * 30
    (ws / decl).write_text(f"<html><body><p>{body}</p></body></html>" if decl.endswith(".html") else body)
    (ws / "results/answers.json").write_text('{"answers": ["4", "7"]}')
    v = _close(store, pid, tid, result=result, files=files)
    assert v.first and v.first.gate == "model_run", v


def test_the_held_recheck_passes_the_trusted_kind():
    """review r2 M2: the owner-run re-check of a held task passed the raw kind.
    Parsed: the experiment check's `kind=` is a name bound from planned_kind()."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(pa))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and any(isinstance(a, ast.Name) and a.id == "unrun_experiment_problem" for a in n.args)]
    assert calls
    for c in calls:
        kw = next(k for k in c.keywords if k.arg == "kind")
        assert isinstance(kw.value, ast.Name), ast.dump(kw.value)
        binds = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == kw.value.id for t in n.targets)]
        assert any(isinstance(b.value, ast.Call) and getattr(b.value.func, "id", "") == "planned_kind"
                   for b in binds)
