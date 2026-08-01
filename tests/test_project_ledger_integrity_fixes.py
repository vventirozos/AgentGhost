"""Fixes from the 2026-08-01 "Mini AI" incident (request b7e516b9).

One request chained five project-management defects into a corrupted build
and a false completion report:

  1. task_decompose's dedup filter keyed on the generic type prefix
     ("Implement: …"), silently eating 2 of 5 planned tasks — the agent then
     burned turns updating PHANTOM task ids.
  2. autoadvance re-picked a task whose deliverable an interactive turn had
     already built and appended a SECOND full implementation into the
     working core.py → SyntaxError on disk → task FAILED → project FAILED.
  3. The FAILED task's recorded "does NOT parse" evidence was stale within
     minutes (the agent rewrote and verified the file) but nothing ever
     re-checked it, so every close attempt wedged.
  4. A later autoadvance call on the already-FAILED project reported
     "All tasks are complete — the project is done."
  5. The turn verifier CONFIRMED the completion claim at 95% because its
     evidence never included the ledger that said FAILED.

Each test names the defect it pins down.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import (
    tool_manage_projects, _feature_key, _filter_duplicate_subtasks,
    _recheck_stale_parse_failure,
)
from ghost_agent.core.planning import ProjectPlan, TaskStatus, TaskTree
from ghost_agent.core.coding_executor import (
    build_coding_task, _py_parse_error, _py_append_guard,
    _task_named_existing, _render_already_built, _generate_build_spec,
)


VALID_PY = (
    "import math\n\n"
    "def sigmoid(x):\n    return 1.0 / (1.0 + math.exp(-x))\n\n"
    "class MiniMLP:\n    def forward(self, x):\n        return sigmoid(x)\n\n"
    "def demo():\n    print(MiniMLP().forward(0.5))\n\n"
    "if __name__ == \"__main__\":\n    demo()\n"
)
BROKEN_PY = "def demo():\n    print('x')\n      return 1\n"

STALE_PARSE_REASON = (
    "code_executor: core.py is on disk but does NOT parse — SYNTAX CHECK "
    "FAILED: 'core.py' was written but does NOT parse. Fix this BEFORE any "
    "other step: unindent does not match any outer indentation level "
    "(line 379, col 31)"
)


@pytest.fixture
def context(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=None, llm_client=None,
    )


@pytest.fixture
def stub_subtools(monkeypatch):
    """Stub the registry so autoadvance ticks never need real sub-tools."""
    import ghost_agent.tools.registry as reg

    async def _web_search(query="", **_kw):
        return f"stub results for {query}"

    monkeypatch.setattr(reg, "get_available_tools",
                        lambda ctx: {"web_search": _web_search})


def _workspace_write(store, pid, rel, content):
    ws = store.ensure_workspace(pid)
    p = Path(ws) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


async def _mini_ai_with_failed_task(context, file_content,
                                    constraints=None):
    """Project + one implement-task wedged exactly like the incident:
    task FAILED with the stale parse evidence, project rolled up FAILED,
    core.py on disk in whatever state the test wants."""
    store = context.project_store
    await tool_manage_projects(context, action="create", title="Mini AI",
                               kind="CODING", constraints=constraints)
    pid = context.current_project_id
    out = json.loads(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["Implement: core.py - The AI model core (weights, "
                  "forward pass, training step)"]))
    tid = out["created"][0]
    _workspace_write(store, pid, "core.py", file_content)
    store.update_task(tid, status="FAILED",
                      failure_reason=STALE_PARSE_REASON)
    assert store.get_project(pid)["status"] == "FAILED"
    return pid, tid


# ===================================================================== D4a
# decompose dedup must not collapse distinct deliverables onto a generic
# type prefix ("Implement: core.py" vs "Implement: train.py").

def test_feature_key_generic_head_does_not_collapse():
    k1 = _feature_key("Implement: core.py - The AI model core")
    k2 = _feature_key("Implement: train.py - Training loop with progress")
    k3 = _feature_key("Research: minimal neural network architectures")
    k4 = _feature_key("Research: training-time benchmarks for tiny nets")
    assert k1 != k2
    assert k3 != k4


def test_feature_key_specific_head_still_dedups():
    # The original intent survives: same FEATURE, different wording → same key.
    assert _feature_key("Core Shell: HTML structure") == \
        _feature_key("Core Shell: index.html skeleton")


def test_filter_returns_kept_and_dropped(context):
    store = context.project_store
    pid = store.create_project("P")
    store.add_task(pid, "Implement: core.py - model core")
    kept, dropped = _filter_duplicate_subtasks(
        store, pid,
        ["Implement: core.py - model core",       # true duplicate
         "Implement: train.py - training loop"])  # distinct deliverable
    assert kept == ["Implement: train.py - training loop"]
    assert dropped == ["Implement: core.py - model core"]


async def test_decompose_persists_all_incident_tasks(context):
    """The exact Mini AI plan shape: 5 planned → 5 persisted (was 3)."""
    await tool_manage_projects(context, action="create", title="Mini AI")
    out = json.loads(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=[
            "Research: minimal neural network architectures",
            "Design: choose the approach and interfaces",
            "Implement: core.py - The AI model core",
            "Implement: train.py - Training loop with progress display",
            "Implement: demo.py - Interactive demo",
        ]))
    assert len(out["created"]) == 5
    assert "dropped_duplicates" not in out


async def test_decompose_reports_dropped_duplicates(context):
    await tool_manage_projects(context, action="create", title="P2")
    await tool_manage_projects(context, action="task_decompose",
                               subtasks=["Implement: core.py - model core"])
    out = json.loads(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["Implement: core.py - model core",
                  "Implement: demo.py - the demo"]))
    assert out["dropped_duplicates"] == ["Implement: core.py - model core"]
    assert "NOT created" in out["agent_instruction_dropped"]
    assert len(out["created"]) == 1


# ===================================================================== D4b
# task_update with ids that don't exist must name the REAL ledger, not
# return a success-shaped empty payload.

async def test_task_update_missing_id_names_valid_tasks(context):
    await tool_manage_projects(context, action="create", title="P3")
    real = json.loads(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["Implement: core.py - model core"]))["created"][0]
    out = json.loads(await tool_manage_projects(
        context, action="task_update", task_id="feedbeef0000",
        status="done", result="phantom"))
    assert out["count"] == 0
    assert out["missing"] == ["feedbeef0000"]
    assert "do NOT exist" in out["agent_instruction_missing"]
    assert any(t["id"] == real for t in out["valid_tasks"])


# ===================================================================== D2
# Stale mechanical failure evidence is re-checked LIVE on a DONE attempt;
# a DONE that sticks clears failure_reason.

def test_update_status_done_clears_stale_failure_reason():
    tree = TaskTree()
    tid = tree.add_task("Implement: core.py")
    tree.update_status(tid, TaskStatus.FAILED, failure_reason="does NOT parse")
    assert tree.nodes[tid].failure_reason
    tree.update_status(tid, TaskStatus.DONE, result="fixed and verified")
    assert tree.nodes[tid].status == TaskStatus.DONE
    assert tree.nodes[tid].failure_reason == ""


async def test_recheck_helper_verdicts(context):
    store = context.project_store
    pid = store.create_project("P4")
    tid = store.add_task(pid, "Implement: core.py")
    # not parse-shaped → not applicable
    verdict, _ = await _recheck_stale_parse_failure(
        store, pid, tid, "web_search failed: empty results")
    assert verdict is None
    # parse-shaped but the file doesn't exist → not applicable
    verdict, _ = await _recheck_stale_parse_failure(
        store, pid, tid, STALE_PARSE_REASON)
    assert verdict is None
    # file parses now → stale, evidence text returned
    _workspace_write(store, pid, "core.py", VALID_PY)
    verdict, detail = await _recheck_stale_parse_failure(
        store, pid, tid, STALE_PARSE_REASON)
    assert verdict is True
    assert "core.py" in detail and "stale" in detail
    # file still broken → current diagnostic, never the stale text
    _workspace_write(store, pid, "core.py", BROKEN_PY)
    verdict, detail = await _recheck_stale_parse_failure(
        store, pid, tid, STALE_PARSE_REASON)
    assert verdict is False
    assert "STILL fails" in detail
    assert "line 379" not in detail          # stale line never echoed


async def test_done_on_stale_parse_failure_uses_live_recheck(context):
    """The incident wedge: file rewritten+verified, ledger still says
    'does NOT parse', project FAILED. One task_update status=done — even
    with NO result evidence — must re-check live, close the task, clear
    the stale reason, and roll the project DONE."""
    store = context.project_store
    pid, tid = await _mini_ai_with_failed_task(
        context, VALID_PY,
        constraints=["implement your own tiny AI"])
    out = json.loads(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="done"))
    assert out["count"] == 1
    row = store.get_task(tid)
    assert row["status"] == "DONE"
    assert row["failure_reason"] == ""
    assert "live syntax re-check" in row["result_summary"]
    assert store.get_project(pid)["status"] == "DONE"


async def test_done_refused_when_file_still_broken(context):
    """A result string cannot talk past the mechanical re-check: the file
    is still broken, so the DONE is refused with the CURRENT diagnostic."""
    store = context.project_store
    pid, tid = await _mini_ai_with_failed_task(context, BROKEN_PY)
    out = json.loads(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="done",
        result="I fixed it, trust me"))
    assert out["count"] == 0
    assert out["still_failing"] == [tid]
    assert "STILL fails" in out["agent_instruction_still_failing"]
    assert store.get_task(tid)["status"] == "FAILED"
    assert store.get_project(pid)["status"] == "FAILED"


# ===================================================================== D3
# An already-FAILED project must never be reported as "done"; parked
# tasks are named with a revival path.

async def test_autoadvance_on_failed_project_reports_project_failed(
        context, stub_subtools):
    store = context.project_store
    pid, tid = await _mini_ai_with_failed_task(context, VALID_PY)
    out = json.loads(await tool_manage_projects(
        context, action="autoadvance", count="all"))
    assert out["stop_reason"] == "project_failed"
    assert "FAILED state" in out["agent_instruction"]
    assert "Do NOT report the project as complete" in out["agent_instruction"]
    assert out["failed_tasks"] and out["failed_tasks"][0]["id"] == tid
    assert "does NOT parse" in out["failed_tasks"][0]["failure_reason"]


async def test_task_next_names_parked_failed_tasks(context):
    pid, tid = await _mini_ai_with_failed_task(context, VALID_PY)
    out = json.loads(await tool_manage_projects(context, action="task_next"))
    assert out["next"] is None
    assert out["parked_tasks"][0]["id"] == tid
    assert out["parked_tasks"][0]["status"] == "FAILED"
    assert "NOT done" in out["agent_instruction"]


# ===================================================================== D1
# The coding executor must never corrupt a working Python file.

def test_py_parse_error_roundtrip():
    assert _py_parse_error(VALID_PY) is None
    assert "indent" in (_py_parse_error(BROKEN_PY) or "")


def test_py_append_guard_refuses_duplicate_implementation():
    snippet = "def demo():\n    print('again')\n"
    reason = _py_append_guard(VALID_PY, snippet, "core.py")
    assert reason and "RE-DEFINES" in reason and "demo" in reason


def test_py_append_guard_refuses_second_main_block():
    snippet = "if __name__ == \"__main__\":\n    print('run')\n"
    reason = _py_append_guard(VALID_PY, snippet, "core.py")
    assert reason and "__main__" in reason


def test_py_append_guard_refuses_merge_that_breaks_parse():
    snippet = "def extra():\n      return 1\n    return 2\n"
    reason = _py_append_guard(VALID_PY, snippet, "core.py")
    assert reason and "would not parse" in reason


def test_py_append_guard_allows_clean_extension():
    snippet = "def save_model(path):\n    return path\n"
    assert _py_append_guard(VALID_PY, snippet, "core.py") is None


def test_py_append_guard_skips_broken_base_and_non_py():
    # Mid-repair file: keep the old flow (post-write check reports it).
    assert _py_append_guard(BROKEN_PY, "def demo():\n    pass\n",
                            "core.py") is None
    # Non-Python: HTML appends are isolated by design, not guarded here.
    assert _py_append_guard("<html></html>", "def x():\n    pass\n",
                            "index.html") is None


class _SpecLLM:
    """Scripted spec responses; optionally with a reasoning channel."""
    def __init__(self, *contents, reasoning=""):
        self.contents = list(contents)
        self.reasoning = reasoning
        self.prompts = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.prompts.append(payload["messages"][-1]["content"])
        c = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        msg = {"content": c}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        return {"choices": [{"message": msg}]}


class _DiskRunner:
    """In-memory disk mirroring file_system read/write shapes."""
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "file_system":
            op = args.get("operation")
            p = args.get("path")
            if op == "read":
                if p in self.files:
                    return f"--- {p} CONTENTS ---\n{self.files[p]}"
                return f"Error: '{p}' does not exist in the sandbox."
            if op == "write":
                self.files[p] = args.get("content")
                return "SUCCESS: Wrote file."
        if name == "execute":
            return "--- EXECUTION RESULT ---\nEXIT CODE: 0\nOK"
        return ""

    def writes(self):
        return [a for (n, a) in self.calls
                if n == "file_system" and a.get("operation") == "write"]


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


async def test_build_refuses_duplicate_append_and_disk_survives():
    """The incident's destructive step, replayed: a spec that appends a
    second implementation into a working core.py must be refused BEFORE
    the write — the working file stays byte-identical on disk."""
    dup = json.dumps({"files": [{"path": "core.py",
                                 "append": "def demo():\n    print('v2')\n"}],
                      "verify": "", "summary": "re-add demo"})
    runner = _DiskRunner(files={"core.py": VALID_PY})
    res = await build_coding_task(
        _ctx(_SpecLLM(dup)), "Implement: core.py - The AI model core",
        tool_runner=runner, existing_files={"core.py": VALID_PY})
    assert not res.ok
    assert runner.writes() == []
    assert runner.files["core.py"] == VALID_PY


async def test_build_refuses_broken_overwrite_of_working_py():
    bad = json.dumps({"files": [{"path": "core.py", "content": BROKEN_PY * 40}],
                      "verify": "", "summary": "rewrite"})
    runner = _DiskRunner(files={"core.py": VALID_PY})
    res = await build_coding_task(
        _ctx(_SpecLLM(bad)), "Implement: core.py",
        tool_runner=runner, existing_files={"core.py": VALID_PY})
    assert not res.ok
    assert runner.writes() == []
    assert runner.files["core.py"] == VALID_PY


async def test_verify_only_spec_survives_reasoning_channel():
    """A verify-only spec ('the deliverable already exists — prove it') in
    the content channel must not be clobbered by reasoning-channel noise,
    and must close the build with no writes."""
    verify_only = json.dumps({"files": [], "verify": "python3 core.py",
                              "summary": "core.py already implements this"})
    llm = _SpecLLM(verify_only, reasoning="thinking about it…")
    runner = _DiskRunner(files={"core.py": VALID_PY})
    res = await build_coding_task(
        _ctx(llm), "Implement: core.py - The AI model core",
        tool_runner=runner, existing_files={"core.py": VALID_PY})
    assert res.ok
    assert res.files == []
    assert runner.writes() == []


def test_task_named_existing_matches_basename():
    existing = {"core.py": VALID_PY, "research/notes.md": "notes"}
    assert _task_named_existing(
        "Implement: core.py - The AI model core", existing) == ["core.py"]
    assert _task_named_existing("Implement: train.py", existing) == []
    assert "ALREADY ON DISK" in _render_already_built(
        "Implement: core.py - the core", existing)
    assert _render_already_built("Design the architecture", existing) == ""


async def test_spec_prompt_carries_already_built_steer():
    verify_only = json.dumps({"files": [], "verify": "python3 core.py",
                              "summary": "exists"})
    llm = _SpecLLM(verify_only)
    spec, was_empty = await _generate_build_spec(
        llm, "m", "Implement: core.py - The AI model core", "",
        existing_files={"core.py": VALID_PY})
    assert not was_empty
    assert spec.get("verify") == "python3 core.py"
    assert "ALREADY ON DISK" in llm.prompts[0]
    assert "core.py" in llm.prompts[0]


# ===================================================================== D5
# Project-touching turns feed the LIVE ledger to the verifier.

def test_project_ledger_evidence_block(context):
    from ghost_agent.core.agent import _project_ledger_evidence
    store = context.project_store
    pid = store.create_project("Mini AI")
    tid = store.add_task(pid, "Implement: core.py - The AI model core")
    store.update_task(tid, status="FAILED",
                      failure_reason=STALE_PARSE_REASON)
    context.current_project_id = pid
    tools = [{"name": "manage_projects", "content": "…"}]
    block = _project_ledger_evidence(context, tools)
    assert block.startswith("[project ledger (live)]")
    assert "status=FAILED" in block
    assert "does NOT parse" in block
    # non-project turn → nothing
    assert _project_ledger_evidence(
        context, [{"name": "web_search", "content": "…"}]) == ""
    # synthetic manage_projects entries don't count
    assert _project_ledger_evidence(
        context, [{"name": "manage_projects", "_synthetic": True}]) == ""


def test_project_ledger_evidence_capped(context):
    from ghost_agent.core.agent import _project_ledger_evidence
    store = context.project_store
    pid = store.create_project("Big")
    for i in range(20):
        store.add_task(pid, f"task number {i} with a long description " * 3)
    context.current_project_id = pid
    block = _project_ledger_evidence(
        context, [{"name": "manage_projects", "content": "…"}])
    assert len(block) <= 1000


def test_project_ledger_evidence_resolves_explicit_project(context):
    """The tool accepts project_id != current_project_id; the evidence must
    cover the project the outputs actually mention, not just `current`."""
    from ghost_agent.core.agent import _project_ledger_evidence
    store = context.project_store
    pid_b = store.create_project("Other Project")
    store.add_task(pid_b, "Implement: thing.py")
    context.current_project_id = None      # no switched project at all
    block = _project_ledger_evidence(
        context, [{"name": "manage_projects",
                   "content": json.dumps({"advanced": [],
                                          "project_id": pid_b,
                                          "stop_reason": "project_done"})}])
    assert pid_b in block
    assert "Other Project" in block


# ================================================== review-pass hardenings
# (fresh-eyes review of these fixes found the holes below; each is pinned)

async def test_recheck_skips_uncheckable_extensions(context, monkeypatch):
    """No node binary → a .js failure CANNOT be re-verified; the recheck
    must return not-applicable, never a fabricated pass (which would
    force-close a still-broken task on manufactured evidence)."""
    import ghost_agent.tools.file_system as fs
    monkeypatch.setattr(fs, "_find_node", lambda: None)
    store = context.project_store
    pid = store.create_project("JS")
    tid = store.add_task(pid, "Implement: app.js")
    _workspace_write(store, pid, "app.js", "function broken( {")
    verdict, detail = await _recheck_stale_parse_failure(
        store, pid, tid,
        "app.js is on disk but does NOT parse — SYNTAX CHECK FAILED")
    assert verdict is None
    assert detail == ""


def test_parse_candidates_survive_marker_prefix_phrasing(context):
    """'py_compile failed for core.py' must extract core.py — the old
    space-tolerant regex consumed the whole phrase as one bogus path."""
    from ghost_agent.tools.projects import _parse_failure_candidates
    store = context.project_store
    pid = store.create_project("P5")
    tid = store.add_task(pid, "Implement: core.py")
    _workspace_write(store, pid, "core.py", VALID_PY)
    paths = _parse_failure_candidates(
        store, pid, tid, "py_compile failed for core.py: unexpected indent")
    assert [p.name for p in paths] == ["core.py"]


async def test_verify_only_spec_with_kill_command_is_refused():
    """files:[] + a kill-shaped verify would otherwise close the task with
    ZERO checks (_run_verify skips kill verifies as a pass)."""
    killer = json.dumps({"files": [],
                         "verify": "pkill -f server; curl localhost:8000",
                         "summary": "already running"})
    runner = _DiskRunner(files={"core.py": VALID_PY})
    res = await build_coding_task(
        _ctx(_SpecLLM(killer)), "Implement: core.py",
        tool_runner=runner, existing_files={"core.py": VALID_PY})
    assert not res.ok
    assert "verify-only spec refused" in res.summary
    assert runner.writes() == []


def test_parent_cascade_done_clears_stale_failure_reason():
    """A revived child closing DONE promotes a once-FAILED parent; the
    parent's dead diagnostic must not survive on the DONE row."""
    tree = TaskTree()
    parent = tree.add_task("Build the feature")
    child = tree.add_task("Implement: part.py", parent_id=parent)
    tree.nodes[parent].status = TaskStatus.FAILED
    tree.nodes[parent].failure_reason = "part.py does NOT parse"
    tree.update_status(child, TaskStatus.DONE, result="fixed and verified")
    assert tree.nodes[parent].status == TaskStatus.DONE
    assert tree.nodes[parent].failure_reason == ""


def test_autoadvance_batch_outcome_is_substantive_for_verifier():
    """A manage_projects output carrying a batch outcome (stop_reason) must
    reach the verifier — the incident turn's ONLY tool was autoadvance and
    the run gate skipped it, so the false completion was never judged."""
    from ghost_agent.core.agent import _find_substantive_tool_for_verifier
    batch = {"name": "manage_projects",
             "content": json.dumps({"advanced": [], "count": 0,
                                    "stop_reason": "project_failed"})}
    assert _find_substantive_tool_for_verifier([batch]) is batch
    # plain confirmations keep skipping (run-gate/evidence-set split)
    plain = {"name": "manage_projects",
             "content": json.dumps({"task_id": "a" * 12,
                                    "status": "PENDING"})}
    assert _find_substantive_tool_for_verifier([plain]) is None
