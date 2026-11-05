"""§4QA — project 448d5afa86f0 (2026-10-18): autoadvance self-stall, tasks
closed DONE without their work, numbered search queries, wrong deliverables.

Each test names the live failure it pins.
"""

from tests._refused import refused_ids, refused_reasons  # §4SU Phase 4
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import project_advancer as pa
from ghost_agent.core.coding_executor import CodingResult, build_coding_task
from ghost_agent.core.task_deliverable import (
    declared_deliverable, deliverable_problem, hollow_document_reason, task_topic,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects

CLEAN_PASS = "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nmarker"
LIVE_13 = ("Task 1.3: Analyze mathematical & computational models of recursion depth "
           "(Church-Turing, type theory, fixed-point combinator) → research/recursion-depth-models.md")
LIVE_62 = ("Task 6.2: Compare with existing frameworks (Hofstadter, Dennett, Buddhist "
           "philosophy, AI recursion) → theory/comparison-with-existing-frameworks.md")
LIVE_HUMAN = "Human-subject replication protocol: design introspection journal format for 3-5 humans"


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(store, tmp_path, llm=None):
    return SimpleNamespace(
        project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None, current_project_id=None,
        llm_client=llm, args=SimpleNamespace(model="m"),
    )


# ------------------------------------------------------------------ #1 self-stall

class _ParkingLLM:
    """The real gate's shape: a BACKGROUND call while a user request is live
    parks (here: forever); a foreground call answers."""

    def __init__(self, foreground_requests):
        self.foreground_requests = foreground_requests
        self.kwargs = None

    async def chat_completion(self, payload, **kw):
        self.kwargs = kw
        if kw.get("is_background") and self.foreground_requests > 0:
            await asyncio.sleep(3600)
        return {"choices": [{"message": {"content": "## Memory\nMy context window holds it."}}]}


async def test_self_analysis_inside_a_live_request_does_not_park_behind_it():
    # live 13:04:34 → 13:14:35: parked 600 s behind its own request, then deferred
    llm = _ParkingLLM(foreground_requests=1)
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    out = await asyncio.wait_for(pa._generate_self_analysis(ctx, "describe your memory"), 5)
    assert "context window" in out
    assert llm.kwargs["is_background"] is False
    assert llm.kwargs["task_label"] == "project-self-analysis"


async def test_self_analysis_from_the_idle_loop_stays_background():
    llm = _ParkingLLM(foreground_requests=0)
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    await pa._generate_self_analysis(ctx, "describe your memory")
    assert llm.kwargs["is_background"] is True


def test_no_self_analysis_call_hardcodes_background():
    # class guard: every advancer LLM call decides background from the context
    import ast
    tree = ast.parse(Path(pa.__file__).read_text())
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
           for k in n.keywords if k.arg == "is_background"
           and isinstance(k.value, ast.Constant) and k.value.value is True]
    assert bad == []


# --------------------------------------------------------- task text helpers

def test_topic_and_declared_path_of_the_live_tasks():
    assert task_topic(LIVE_13) == ("Analyze mathematical & computational models of recursion "
                                   "depth (Church-Turing, type theory, fixed-point combinator)")
    assert declared_deliverable(LIVE_13) == "research/recursion-depth-models.md"
    assert declared_deliverable(LIVE_62) == "theory/comparison-with-existing-frameworks.md"
    # prose after the arrow is no path, and is kept in the topic
    d = "Run recursive_harness.py on all outputs → get actual scores"
    assert declared_deliverable(d) is None and task_topic(d) == d
    assert task_topic("1.2. Do x -> `a/b.py`") == "Do x"
    assert declared_deliverable("Do x -> /workspace/a/b.py") == "a/b.py"


@pytest.mark.parametrize("bad", ["x → ../etc/passwd.md", "x → /etc/passwd.md", "x → a/./b.md",
                                 "x → a b.md", "x → noext"])
def test_declared_path_never_escapes(bad):
    assert declared_deliverable(bad) is None


def test_hollow_documents():
    assert hollow_document_reason("...", "results/meta-reflection.md")
    assert hollow_document_reason("# Title\n\n…\n", "a.md")
    assert hollow_document_reason("TODO", "a.rst")
    assert hollow_document_reason("flask\nrequests", "requirements.txt") is None
    assert hollow_document_reason("# Meta-reflection\n\nObserving the recursion changed it in "
                                  "three measurable ways.", "a.md") is None
    assert hollow_document_reason("", "pkg/__init__.py") is None       # code is not judged


# --------------------------------------------------------------- #5 routing

@pytest.mark.parametrize("desc", [
    "Update final report with actual quantitative data from harness runs",
    "Replace lexicon metrics with semantic embeddings (sentence-transformers) for scoring",
    LIVE_HUMAN,
    LIVE_62,
])
def test_project_work_is_built_not_researched(desc):
    assert pa._route_project_work(desc, "research") == "coding"


@pytest.mark.parametrize("desc", [
    LIVE_13,                                         # research/ deliverable stays research
    "Research how teams update their reports",
    "Find benchmarks comparing Qwen and Llama on code generation",
])
def test_research_stays_research(desc):
    assert pa._route_project_work(desc, "research") == "research"


def test_route_never_touches_other_buckets_or_introspection():
    assert pa._route_project_work("Update final report", "needs_user") == "needs_user"
    assert pa._route_project_work("Update your own memory model analysis", "research") == "research"


def test_human_subject_introspection_is_not_self_analysis():
    assert pa.is_self_referential(LIVE_HUMAN) is False
    assert pa.is_self_referential("Run an introspection pass over the agent's processing") is True
    assert pa.is_self_referential("Self-introspection on recursion") is True


async def test_research_query_is_the_topic_and_brief_takes_the_declared_name(store, tmp_path):
    # live: "task 1.3: analyze mathematic.." — off-topic everywhere, task FAILED
    pid = store.create_project("Recursion")
    tid = store.add_task(pid, LIVE_13)
    queries = []

    async def runner(name, args):
        queries.append((name, args))
        return "[Source: https://example.org/y]\nThe Y combinator gives fixed points; type theory bounds recursion depth."

    ctx = _ctx(store, tmp_path)

    async def cls(_d):
        return "research"

    await pa.advance_once(ctx, pid, tool_runner=runner, llm_classifier=cls, owner_requested=True)
    assert queries[0] == ("web_search", {"query": task_topic(LIVE_13)})
    ws = Path(store.ensure_workspace(pid))
    assert (ws / "research" / "recursion-depth-models.md").is_file()
    assert not list((ws / "research").glob("task-1-3*"))
    assert store.get_task(tid)["status"] == "DONE"


async def test_declared_non_research_deliverable_is_not_web_searched(store, tmp_path):
    # live: 6.2 wrote research/task-6-2-….md; the declared theory/… never existed
    pid = store.create_project("Recursion")
    tid = store.add_task(pid, LIVE_62)
    calls = []

    async def runner(name, args):
        calls.append(name)
        return "results"

    async def cls(_d):
        return "research"

    res = await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner,
                                llm_classifier=cls, owner_requested=True)
    assert "web_search" not in calls
    assert res.classification == "coding"
    assert store.get_task(tid)["status"] != "DONE"


# ------------------------------------------- #2 hollow writes, existence verify

class _SpecLLM:
    def __init__(self, spec):
        self.spec = json.dumps(spec)

    async def chat_completion(self, payload, **_kw):
        return {"choices": [{"message": {"content": self.spec}}]}


def _disk_runner(ws, verify_output=CLEAN_PASS):
    calls = []

    async def runner(name, args):
        calls.append((name, args))
        if name == "file_system":
            p = Path(ws) / args["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args.get("content", ""), encoding="utf-8")
            return f"Successfully wrote {args.get('path')}"
        if name == "execute":
            return verify_output
        return "OK"

    return runner, calls


async def test_three_byte_document_is_refused(store, tmp_path):
    # live: results/meta-reflection.md = "..." closed DONE on `test -f`
    pid = store.create_project("P")
    tid = store.add_task(pid, "Task 7.2: Reflect on observation → results/meta-reflection.md")
    ws = store.ensure_workspace(pid)
    ctx = _ctx(store, tmp_path, _SpecLLM({"files": [{"path": "results/meta-reflection.md", "content": "..."}],
                                          "verify": "test -f results/meta-reflection.md",
                                          "summary": "wrote it", "ledger": ""}))
    runner, _ = _disk_runner(ws)

    async def cls(_d):
        return "coding"

    await pa.advance_once(ctx, pid, tool_runner=runner, llm_classifier=cls,
                          coding_executor=build_coding_task, owner_requested=True)
    t = store.get_task(tid)
    assert t["status"] == "FAILED"
    assert "only a placeholder" in t["failure_reason"]


def test_executor_floor_is_elision_only():
    from ghost_agent.core.task_deliverable import elided_document_reason
    assert elided_document_reason("...\n\n…", "a.md")
    assert elided_document_reason("", "a.md")
    assert elided_document_reason("# Mini AI", "README.md") is None   # weak, not a refusal
    assert elided_document_reason("...", "a.py") is None


@pytest.mark.parametrize("verify,verified", [
    ("test -f notes.md", False),
    ("test -s notes.md && echo 'OK'", False),
    ("[ -f notes.md ] && ls notes.md", False),
    ("grep -q 'measurable' notes.md", True),
])
async def test_existence_only_verify_is_not_verification(store, tmp_path, verify, verified):
    pid = store.create_project("P")
    ws = store.ensure_workspace(pid)
    ctx = _ctx(store, tmp_path, _SpecLLM({
        "files": [{"path": "notes.md", "content": "Observing the recursion changed it in three measurable ways."}],
        "verify": verify, "summary": "wrote notes", "ledger": ""}))
    runner, calls = _disk_runner(ws)
    cres = await build_coding_task(ctx, "Write the notes", tool_runner=runner)
    assert cres.ok is True
    assert cres.verified is verified
    assert any(n == "execute" for n, _ in calls)       # it still RAN


async def test_existence_only_verify_still_fails_on_a_missing_file(store, tmp_path):
    pid = store.create_project("P")
    ws = store.ensure_workspace(pid)
    ctx = _ctx(store, tmp_path, _SpecLLM({
        "files": [{"path": "notes.md", "content": "Observing the recursion changed it in three measurable ways."}],
        "verify": "test -f other.md", "summary": "wrote", "ledger": ""}))
    runner, _ = _disk_runner(ws, "--- COMMAND RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n")
    cres = await build_coding_task(ctx, "Write the notes", tool_runner=runner)
    assert cres.ok is False


async def test_a_build_that_missed_the_declared_file_is_not_done(store, tmp_path):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Task 6.4: Write final report → report/final-report.md")

    async def builder(*_a, **_k):
        return CodingResult(ok=True, summary="wrote the report", files=["final-report.md"], verified=True)

    async def cls(_d):
        return "coding"

    async def runner(*_a):
        return "OK"

    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          coding_executor=builder, owner_requested=True)
    t = store.get_task(tid)
    assert t["status"] == "FAILED"
    assert "report/final-report.md does not exist" in t["failure_reason"]


# --------------------------------------------------- #5f task_update DONE gate

async def _project_with(tmp_path, desc):
    store = ProjectStore(tmp_path / "mem2", sandbox_root=tmp_path / "sb2")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Recursion")
    pid = ctx.current_project_id
    tid = json.loads(await tool_manage_projects(ctx, action="task_decompose", subtasks=[desc]))["created"][0]
    return ctx, store, pid, tid, Path(store.ensure_workspace(pid))


async def _done(ctx, tid, **kw):
    return json.loads(await tool_manage_projects(ctx, action="task_update", task_id=tid,
                                                 status="done", result="wrote it", **kw))


async def test_done_refused_when_the_declared_file_is_one_folder_off(tmp_path):
    # live: 1.1 wrote recursive-thinking-experiment/research/… and closed DONE
    ctx, store, pid, tid, ws = await _project_with(
        tmp_path, "Task 1.1: Research recursive self-reference → research/recursive-self-reference.md")
    nested = ws / "recursive-thinking-experiment" / "research" / "recursive-self-reference.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("# Survey\n\nKant, Hofstadter, Dennett, GWT and IIT each describe self-reference.")
    out = await _done(ctx, tid)
    assert refused_ids(out, "undelivered") == [tid]
    assert "recursive-thinking-experiment/research/recursive-self-reference.md" in refused_reasons(out, "undelivered")
    assert store.get_task(tid)["status"] != "DONE"


async def test_done_refused_on_a_hollow_declared_file(tmp_path):
    ctx, store, pid, tid, ws = await _project_with(
        tmp_path, "Task 7.2: Reflect on observation → results/meta-reflection.md")
    (ws / "results").mkdir()
    (ws / "results" / "meta-reflection.md").write_text("...")
    out = await _done(ctx, tid)
    assert refused_ids(out, "undelivered") == [tid]
    assert "no content" in refused_reasons(out, "undelivered")


async def test_done_accepted_when_the_declared_file_has_content(tmp_path):
    ctx, store, pid, tid, ws = await _project_with(
        tmp_path, "Task 7.2: Reflect on observation → results/meta-reflection.md")
    (ws / "results").mkdir()
    (ws / "results" / "meta-reflection.md").write_text(
        "# Meta-reflection\n\nObserving the recursion changed it in three measurable ways.")
    out = await _done(ctx, tid)
    assert not refused_ids(out, "undelivered")
    assert store.get_task(tid)["status"] == "DONE"


async def test_a_rewritten_description_is_what_the_gate_judges(tmp_path):
    ctx, store, pid, tid, ws = await _project_with(
        tmp_path, "Task 6.2: Compare frameworks → theory/comparison.md")
    (ws / "theory").mkdir()
    (ws / "theory" / "frameworks.md").write_text(
        "# Frameworks\n\nHofstadter and Dennett both treat the self as a pattern, not a thing.")
    out = await _done(ctx, tid, description="Task 6.2: Compare frameworks → theory/frameworks.md")
    assert not refused_ids(out, "undelivered")
    assert store.get_task(tid)["status"] == "DONE"


async def test_a_task_without_a_declared_file_is_unaffected(tmp_path):
    ctx, store, pid, tid, ws = await _project_with(tmp_path, "Add falsification tests")
    out = await _done(ctx, tid)
    assert not refused_ids(out, "undelivered")
    assert store.get_task(tid)["status"] == "DONE"


def test_symlinked_deliverable_is_refused(tmp_path):
    (tmp_path / "secret.md").write_text("word " * 20)
    (tmp_path / "ws" / "results").mkdir(parents=True)
    os.symlink(tmp_path / "secret.md", tmp_path / "ws" / "results" / "x.md")
    assert "symbolic link" in deliverable_problem(tmp_path / "ws", "t → results/x.md")


# ------------------------------------------------- #4 ownership on reference

from tests.focus_helpers import scope_reset  # noqa: F401,E402


def _scope_ctx(tmp_path):
    store = ProjectStore(tmp_path / "mem3", sandbox_root=tmp_path / "sb3")
    ctx = _ctx(store, tmp_path)
    ctx.last_user_content = ""
    return ctx, store


def _request(ctx, conv, message):
    """A request start of conversation ``conv`` (§4SX B0: its focus row)."""
    from tests.focus_helpers import start_request
    ctx.last_user_content = message
    start_request(ctx, conv)
    return ctx.current_project_id


# ------------------------------------------------ #6 pre-project folder prefix

def _two_level_sandbox(tmp_path):
    root = tmp_path / "sandbox"
    (root / "recursive-thinking-experiment").mkdir(parents=True)
    (root / "recursive-thinking-experiment" / "tasks.md").write_text("tasks")
    ws = root / "projects" / "448d5afa86f0"
    ws.mkdir(parents=True)
    return root, ws


def _age(path, seconds=3600):
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


@pytest.mark.parametrize("name", ["recursive-thinking-experiment/research/x.md",
                                  "/workspace/recursive-thinking-experiment/research/x.md"])
async def test_a_pre_project_prefix_inside_the_project_gets_a_note(tmp_path, name):
    # fresh-eye M1: a NOTE on the write, judged on the resolved path (the
    # /workspace/ form was missed by the first version's refusal)
    from ghost_agent.tools.file_system import tool_write_file
    root, ws = _two_level_sandbox(tmp_path)
    _age(root / "recursive-thinking-experiment")
    out = await tool_write_file(name, "# Survey\n\ntext", ws)
    assert str(out).startswith("SUCCESS")
    assert "sandbox ROOT from before this project" in str(out)
    assert "the file belongs at 'research/x.md'" in str(out)
    again = await tool_write_file("recursive-thinking-experiment/research/y.md", "# S\n\ntext", ws)
    assert "sandbox ROOT" not in str(again)          # once per folder


async def test_a_projects_own_first_src_file_is_written(tmp_path):
    # fresh-eye M1: a root src/ refused a new CODING project's first file
    from ghost_agent.tools.file_system import tool_write_file
    root, ws = _two_level_sandbox(tmp_path)
    (root / "src").mkdir()
    _age(root / "src")
    out = await tool_write_file("src/main.py", "print('hi')\n", ws)
    assert str(out).startswith("SUCCESS") and (ws / "src" / "main.py").is_file()
    # a root folder made AFTER the project is not "from before" it
    (root / "results").mkdir()
    future = ws.stat().st_mtime + 3600
    os.utime(root / "results", (future, future))
    out = await tool_write_file("results/r.md", "# R\n\ntext", ws)
    assert "sandbox ROOT" not in str(out)


async def test_the_prefix_is_fine_once_the_project_has_that_folder(tmp_path):
    from ghost_agent.tools.file_system import tool_write_file
    root, ws = _two_level_sandbox(tmp_path)
    (ws / "recursive-thinking-experiment").mkdir()
    await tool_write_file("recursive-thinking-experiment/research/x.md", "# S\n\ntext", ws)
    assert (ws / "recursive-thinking-experiment" / "research" / "x.md").is_file()


async def test_outside_a_project_the_root_folder_is_simply_used(tmp_path):
    from ghost_agent.tools.file_system import tool_write_file
    root, ws = _two_level_sandbox(tmp_path)
    await tool_write_file("recursive-thinking-experiment/research/x.md", "# S\n\ntext", root)
    assert (root / "recursive-thinking-experiment" / "research" / "x.md").is_file()


async def test_a_root_file_with_that_name_is_no_reason_to_refuse(tmp_path):
    from ghost_agent.tools.file_system import tool_write_file
    root, ws = _two_level_sandbox(tmp_path)
    (root / "notes").write_text("a file, not a folder")
    await tool_write_file("notes/x.md", "# S\n\ntext", ws)
    assert (ws / "notes" / "x.md").is_file()


# ------------------------------------------ #7 the one-task stop on the stream

def _sse(text):
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode()


async def _drive_one_task_final(task_closed):
    from unittest.mock import AsyncMock, MagicMock
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finalize_stream_pins import make_stream_agent, _make_stream_state
    a = make_stream_agent()
    a._write_project_work_log_safe = AsyncMock()
    pulled = []
    body = [_sse("Task 1.1 is complete. The next task is 1.2."),
            _sse("\n\n<tool_call>\n<function=file_system>\n<parameter=content>\n")]
    body += [_sse("A long research document line that the scrub eats. " * 4) for _ in range(60)]
    body += [_sse("</parameter>\n</function>\n</tool_call>"), b"data: [DONE]\n\n"]

    async def final_stream(payload, use_coding=False):
        for c in body:
            pulled.append(c)
            yield c

    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    st = _make_stream_state(reg)
    st._proj_task_closed_this_req = task_closed
    gen, _, _ = a._stream_final_generation(st)
    out = []
    async for c in gen:
        out.append(c.decode("utf-8", "replace"))
    text = "".join(json.loads(x[6:])["choices"][0]["delta"].get("content", "")
                   for x in "".join(out).split("\n\n")
                   if x.startswith("data: {") and '"choices"' in x)
    _wl = a._write_project_work_log_safe.call_args
    durable = (_wl.kwargs.get("final_ai_content") if _wl else "") or ""
    return text, "".join(out), len(pulled), len(body), durable


async def test_a_designed_stop_says_how_to_continue_and_stops_generating():
    text, raw, pulled, total, durable = await _drive_one_task_final(True)
    assert "say *proceed*" in durable and "NOT executed" not in durable   # the record says what the stream said
    assert "Task 1.1 is complete" in text
    assert "NOT executed" not in text
    assert "say *proceed*" in text
    assert "<tool_call>" not in raw
    assert pulled < total // 2                 # stopped once the call opened
    assert raw.rstrip().endswith("data: [DONE]")


async def test_a_real_unparsed_call_still_gets_the_note():
    text, raw, pulled, total, durable = await _drive_one_task_final(False)
    assert "NOT executed" in text
    assert pulled == total


# ------------------------------------------------------ #8 "on your own"

@pytest.mark.parametrize("msg,mode", [
    ("proceed the project on your own.", "all"),          # live 25a54a0a
    ("continue autonomously", "all"),
    ("keep going without asking me", "all"),
    ("run the remaining work by yourself", "all"),
    ("do 2 tasks on your own", "n"),                       # a count still wins
    ("can you write task 3 on your own?", "one"),          # no continue-verb
    ("proceed", "one"),
    ("I can't proceed on this alone", "one"),
])
def test_on_your_own_is_the_rest(msg, mode):
    assert pa.classify_advance_intent(msg)["mode"] == mode


# --------------------------------------------- #9 where an unread link came from

def test_snippet_links_are_not_blamed_on_an_earlier_request(tmp_path):
    # live 25a54a0a: 20 links from THIS request's search snippets were
    # captioned "an earlier request may have read them"
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from tests.test_4oh_open_items import _write_row
    from ghost_agent.core.agent import unopened_citation_note
    body = "# AI\n\n- https://grokipedia.com/page/Recursive_Meta-Cognition\n- https://never.example/p\n"
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "meta.md").write_text(body)
    search = {"name": "web_search",
              "content": "[Source: https://grokipedia.com/page/Recursive_Meta-Cognition]\nRecursive meta-cognition is …"}
    note = unopened_citation_note([search, _write_row("research/meta.md", body)], (tmp_path,))
    assert "2 links" in note
    assert "1 only from this request's search-result snippets" in note
    assert "1 in no result of this request (an earlier request may have read it)" in note
    (tmp_path / "research" / "meta.md").write_text(body.replace("- https://never.example/p\n", ""))
    only_snip = unopened_citation_note(
        [search, _write_row("research/meta.md", body.replace("- https://never.example/p\n", ""))], (tmp_path,))
    assert "earlier request" not in only_snip and "search-result snippets" in only_snip


# ------------------------------------------------------------- #10 memory

def test_the_memory_arc_skips_the_loops_own_user_rows():
    from ghost_agent.core.agent import _build_memory_arc
    hist = [{"role": "user", "content": "add this picture to the project"},
            {"role": "assistant", "content": "Adding it."},
            {"role": "user", "content": "AUTO-DIAGNOSTIC: ERROR: task_id is required for action=artifact_add"},
            {"role": "user", "content": "SYSTEM ALERT: retry with a description"}]
    arc = _build_memory_arc(hist, "Done — the image is registered.", tools_run=[])
    assert "AUTO-DIAGNOSTIC" not in arc and "SYSTEM ALERT" not in arc
    assert "USER: add this picture to the project" in arc


@pytest.mark.parametrize("trigger,indexed", [
    ("proceed all tasks", False),
    ("proceed.", False),
    ("proceed the project on your own.", False),
    ("ok continue with the next task", False),
    ("proceed will all tasks", False),
    ("is there any way to improve your findings ?", True),
    ("proceed with the PostgreSQL migration plan", True),
    ("add this picture to the project", True),
])
def test_content_free_triggers_get_no_episode_vector(trigger, indexed):
    from ghost_agent.memory.episodes import EpisodicMemory
    added = []
    vm = SimpleNamespace(add=lambda text, meta: added.append(text))
    EpisodicMemory._ingest_episode_vector(SimpleNamespace(), 7, trigger, "", vm)
    assert bool(added) is indexed
    if not indexed:   # a LESSON still makes it worth recalling
        EpisodicMemory._ingest_episode_vector(SimpleNamespace(), 7, trigger, "retry the search", vm)
        assert added


# --------------------------------------------- #11 dates, #12 artifact_add hint

async def test_list_and_get_show_readable_dates(tmp_path):
    import time as _t
    ctx, store = _scope_ctx(tmp_path)
    _request(ctx, "convA", "create a project called Recursion")
    ctx.conversation_key = "convA"
    await tool_manage_projects(ctx, action="create", title="Recursion")
    today = _t.strftime("%Y-%m-%d")
    out = json.loads(await tool_manage_projects(ctx, action="list", metadata={"verbose": True}))
    assert out["projects"][0]["created"].startswith(today)
    assert f"(created {today}" in out["display"]
    got = json.loads(await tool_manage_projects(ctx, action="get", project_id=ctx.current_project_id))
    assert got["created"].startswith(today) and got["updated"].startswith(today)


async def test_artifact_add_without_a_task_names_describe_file(tmp_path):
    ctx, store = _scope_ctx(tmp_path)
    _request(ctx, "convA", "create a project called Recursion")
    ctx.conversation_key = "convA"
    await tool_manage_projects(ctx, action="create", title="Recursion")
    out = await tool_manage_projects(ctx, action="artifact_add", payload="gen_7ee20fb9.png")
    assert "describe_file" in out and "description" in out


# --------------------------------------- #3 real model experiments (probe_model)

class _EchoLLM:
    def __init__(self):
        self.calls = []

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        n = len(self.calls)
        return {"choices": [{"message": {"content": f"ANSWER {n} to: {payload['messages'][-1]['content']}"}}]}


def _probe_ctx(tmp_path):
    ctx = SimpleNamespace(llm_client=_EchoLLM(), args=SimpleNamespace(model="qwen3.6"),
                          sandbox_dir=tmp_path / "sb", current_project_id="448d5afa86f0",
                          scratchpad=None, conversation_key="c")
    (tmp_path / "sb" / "projects" / "448d5afa86f0").mkdir(parents=True)
    return ctx


async def test_probe_runs_each_prompt_for_real_in_the_foreground(tmp_path):
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    out = await tool_probe_model(ctx, prompts=["Describe recursion.", "Reflect on that answer."],
                                 temperature=0.3)
    assert "ANSWER 1 to: Describe recursion." in out and "ANSWER 2" in out
    for payload, kw in ctx.llm_client.calls:
        assert kw["is_background"] is False                 # never parks behind its own turn
        assert payload["temperature"] == 0.3
        assert len(payload["messages"]) == 1                # fresh context, independent
    assert ctx.llm_client.calls[0][1]["use_worker"] is False


async def test_a_chained_probe_carries_the_previous_answers(tmp_path):
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    await tool_probe_model(ctx, prompts=["L0", "L1", "L2"], chain=True, system="Be brief.")
    msgs = ctx.llm_client.calls[2][0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "ANSWER 1 to: L0"


async def test_the_system_writes_the_transcript_and_never_overwrites_it(tmp_path):
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    out = await tool_probe_model(ctx, prompts=["Describe recursion."], save_to="results/runs/run1.md")
    f = tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "runs" / "run1.md"
    body = f.read_text()
    assert "the system wrote this file" in body and "ANSWER 1 to: Describe recursion." in body
    assert "temperature: 0.7" in body and "model: qwen3.6" in body
    n = len(ctx.llm_client.calls)
    again = await tool_probe_model(ctx, prompts=["x"], save_to="results/runs/run1.md")
    assert again.startswith("REJECTED") and "already exists" in again and f.read_text() == body
    assert len(ctx.llm_client.calls) == n                     # refused before any call
    # fresh-eye MAJOR-1: the prefixed form names the SAME file
    pref = await tool_probe_model(ctx, prompts=["x"],
                                  save_to="/workspace/projects/448d5afa86f0/results/runs/run1.md")
    assert pref.startswith("REJECTED") and f.read_text() == body


async def test_the_worker_node_never_falls_back_to_main(tmp_path):
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    await tool_probe_model(ctx, prompts=["x"], node="worker")
    kw = ctx.llm_client.calls[0][1]
    assert kw["use_worker"] is True and kw["off_main_only"] is True


@pytest.mark.parametrize("kw", [{"prompts": []}, {"prompts": ["a"] * 9}, {"prompts": [""]},
                                {"prompts": ["a"], "save_to": "../x.md"},
                                {"prompts": ["a"], "save_to": "results/x.py"}])
async def test_probe_refuses_bad_input_before_any_call(tmp_path, kw):
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    out = await tool_probe_model(ctx, **kw)
    assert out.startswith("REJECTED") and ctx.llm_client.calls == []


def test_probe_model_is_closed_to_members_replays_and_macros():
    from ghost_agent.core.agent import _MEMBER_ALLOWED_TOOLS
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    from ghost_agent.tools.composed_skills import MACRO_FORBIDDEN_STEP_TOOLS
    assert "probe_model" not in _MEMBER_ALLOWED_TOOLS
    assert "probe_model" in REPLAY_FORBIDDEN_TOOLS and "probe_model" in MACRO_FORBIDDEN_STEP_TOOLS


GOAL = ("Test the hypothesis that 3-layer recursive thinking culminates in self-affirming "
        "self-awareness, by running recursive introspection loops on an LLM")


@pytest.mark.parametrize("desc,exp", [
    ("Task 3.1: Run Layer 0 — Base State (direct experience, no meta) → results/layer0-base.md", True),
    ("Task 4.1: Test with abstract concepts (time, infinity) → results/abstract-start.md", True),
    ("Create and run control condition: same prompts but non-recursive → compare outputs", True),
    ("Add falsification tests: noise injection, contradictory self-references", True),
    ("Run recursive_harness.py on all 5 layer outputs → get actual scores", False),
    ("Task 6.1: Draft the recursive self-awareness theory → theory/t.md", False),
])
def test_model_experiments_are_recognised(desc, exp):
    assert pa.is_model_experiment(desc, GOAL) is exp


def test_no_model_in_the_goal_no_experiment():
    assert pa.is_model_experiment("Run the load test on the API", "Build a web service") is False


async def test_autoadvance_pauses_at_an_experiment_and_says_how_to_run_it(tmp_path):
    from ghost_agent.tools.projects import _advance_batch_instruction
    store = ProjectStore(tmp_path / "mem4", sandbox_root=tmp_path / "sb4")
    pid = store.create_project("Recursion", goal=GOAL)
    t1 = store.add_task(pid, "Task 3.1: Run Layer 0 — Base State → results/layer0-base.md")
    calls = []

    async def runner(name, args):
        calls.append(name)
        return "OK"

    async def builder(*_a, **_k):
        calls.append("build")
        return CodingResult(ok=True, summary="wrote it", files=["results/layer0-base.md"], verified=True)

    async def cls(_d):
        return "coding"

    batch = await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=None, tool_runner=runner,
                                  llm_classifier=cls, coding_executor=builder, owner_requested=True)
    assert batch.stop_reason == "model_experiment"
    assert calls == []                                     # nothing was built or searched
    assert store.get_task(t1)["status"] == "PENDING"
    msg = _advance_batch_instruction(batch, None)
    assert "probe_model" in msg and t1 in msg and "never write a model response yourself" in msg


@pytest.mark.parametrize("call_text", [
    # parsed → dropped by the final-generation guard
    "<tool_call>\n<function=manage_projects>\n<parameter=action>\ntask_next\n</parameter>\n</function>\n</tool_call>",
    # unclosed → unparseable → scrubbed at finalize (FinalizeState.task_closed_this_req)
    "<tool_call>\n<function=file_system>\n<parameter=content>\n# Next task notes, cut off mid-write",
])
async def test_the_one_task_stop_reaches_the_non_stream_reply_through_handle_chat(tmp_path, monkeypatch, call_text):
    # FinalizeState must carry the gate: live the reply ended "[A tool call … NOT executed]"
    from unittest.mock import AsyncMock
    from tests.helpers import make_context, FakeBgTasks
    from tests.test_4ji_request_echo_narration import _resp, _tc
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    for k, v in (("GHOST_CRITIC_ASYNC", "1"), ("GHOST_CRITIC_REPAIR_BUDGET", "0"), ("GHOST_EVIDENCE_GATE", "0")):
        monkeypatch.setenv(k, v)
    store = ProjectStore(tmp_path / "m5", sandbox_root=tmp_path / "sb5")
    pid = store.create_project("Recursion", goal="notes")
    tid = store.add_task(pid, "Collect notes")
    store.add_task(pid, "Summarise notes")
    ctx = make_context()
    ctx.project_store = store
    store.set_focus("session:s5", pid)          # §4SX B0: the conversation focuses it
    agent = GhostAgent(ctx)

    async def _mp(**kw):
        return await tool_manage_projects_for_model(ctx, **kw)

    agent.available_tools = {"manage_projects": _mp}
    final = "Task 'Collect notes' is complete. The next task is 'Summarise notes'.\n\n" + call_text
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("t0", "manage_projects",
                       {"action": "task_update", "task_id": tid, "status": "done", "result": "notes collected"})]),
    ] + [_resp(final)] * 4)
    out = await agent.handle_chat({"messages": [{"role": "user", "content": f"proceed with {pid}"}],
                                   "session_id": "s5"}, FakeBgTasks())
    reply = out[0] if isinstance(out, tuple) else str(out)
    assert store.get_task(tid)["status"] == "DONE"
    assert "say *proceed*" in reply, reply
    assert "NOT executed" not in reply


# =================================================== fresh-eye review (r1)

@pytest.mark.parametrize("desc", ["Port frontend → Vue.js", "Add realtime -> socket.io",
                                  "Upgrade → Python3.12", "Use a framework → e.g.",
                                  "Rewrite the server → Node.js"])
def test_technology_names_are_not_deliverables(desc):
    # M4: they parsed as files, failed good builds and refused DONE
    assert declared_deliverable(desc) is None


@pytest.mark.parametrize("desc,path", [
    ("Write it → `report/final.md`.", "report/final.md"),       # m11
    ("Write it → notes.md", "notes.md"),
    ("Build it → app.js", "app.js"),
])
def test_real_bare_and_punctuated_deliverables(desc, path):
    assert declared_deliverable(desc) == path


def test_a_range_is_not_a_numbering_prefix():
    assert task_topic("5-7 year outlook for LLM costs") == "5-7 year outlook for LLM costs"    # m11
    assert task_topic("Step 2 - collect data") == "collect data"


def test_a_cjk_document_is_not_hollow():
    # m9: a Chinese paragraph counted as one word
    zh = "这是一个关于递归思维的中文段落，包含了足够多的内容来说明问题。"
    assert hollow_document_reason(zh, "a.md") is None
    assert hollow_document_reason("这是", "a.md")


def test_a_symlinked_folder_out_of_the_workspace_is_not_delivered(tmp_path):
    # m10: O_NOFOLLOW guards only the last component
    ws, out = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir(); out.mkdir()
    (out / "z.md").write_text("A real looking document with many words in it for the gate to pass.")
    (ws / "linked").symlink_to(out)
    assert "symbolic link" in deliverable_problem(ws, "x → linked/z.md")


GOAL_RUN = GOAL


@pytest.mark.parametrize("desc,goal", [
    ("Measure price elasticity", "Refine the model of regional housing prices"),   # names no model
    ("Research: run-time costs of GPT-4", GOAL_RUN),                               # not experiment-shaped
    ("Run recursive_harness.py on all outputs", GOAL_RUN),                         # a code file is code
])
def test_experiment_false_positives(desc, goal):
    # M2: "the model" is no model name; a topic or a code run is no experiment.
    # (r5 inversion: an experiment-shaped task in ANY model project is held —
    # "Measure market share" in "Survey adoption of LLMs" now waits for the owner)
    assert pa.is_model_experiment(desc, goal) is False


async def test_an_unattended_tick_holds_the_experiment_for_the_owner(tmp_path):
    # M2: released to PENDING it was re-picked every tick → "no progress"
    store = ProjectStore(tmp_path / "mem6", sandbox_root=tmp_path / "sb6")
    pid = store.create_project("Recursion", goal=GOAL)
    t1 = store.add_task(pid, "Task 3.1: Run Layer 0 — Base State → results/layer0-base.md")
    t2 = store.add_task(pid, "Task 6.1: Draft the theory notes")
    calls = []

    async def runner(name, args):
        calls.append(name)
        return "[Source: https://example.org]\nTheory notes on recursion and awareness, long enough to keep."

    async def cls(_d):
        return "research"

    ctx = _ctx(store, tmp_path)
    res = await pa.advance_once(ctx, pid, tool_runner=runner, llm_classifier=cls, owner_requested=False)
    assert res.classification == "needs_user" and res.task_id == t1
    t = store.get_task(t1)
    assert t["status"] == "NEEDS_USER" and t["result_summary"].endswith(pa._HELD_MARK)
    assert "probe_model" in t["result_summary"]
    res2 = await pa.advance_once(ctx, pid, tool_runner=runner, llm_classifier=cls, owner_requested=False)
    assert res2.task_id == t2                                  # the independent leaf runs


async def test_the_owners_turn_keeps_the_hold_until_it_runs_the_experiment(tmp_path):
    store = ProjectStore(tmp_path / "mem7", sandbox_root=tmp_path / "sb7")
    pid = store.create_project("Recursion", goal=GOAL)
    t1 = store.add_task(pid, "Task 3.1: Run Layer 0 — Base State → results/layer0-base.md")

    async def cls(_d):
        return "research"

    async def runner(*_a):
        return "OK"

    ctx = _ctx(store, tmp_path)
    await pa.advance_once(ctx, pid, tool_runner=runner, llm_classifier=cls, owner_requested=False)
    held = store.get_task(t1)["result_summary"]
    res = await pa.advance_once(ctx, pid, tool_runner=runner, llm_classifier=cls, owner_requested=True)
    assert res.summary.startswith(pa.MODEL_EXPERIMENT_PREFIX)
    t = store.get_task(t1)
    assert t["status"] == "NEEDS_USER" and t["result_summary"] == held


@pytest.mark.parametrize("owner,status", [(True, "FAILED"), (False, "NEEDS_USER")])
async def test_a_research_close_without_the_declared_file_is_not_done(store, tmp_path, monkeypatch, owner, status):
    # M3: the brief did not land at the declared name, yet the leaf closed DONE.
    # r2 MAJOR-3: owner-run it FAILS (a hold was re-picked by every owner
    # advance and re-searched forever); unattended it is held for the owner
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("Recursion")
    tid = store.add_task(pid, "Survey depth models → research/depth.md")

    async def no_brief(*_a, **_k):
        return PRM.ResearchResult(False, "t", error="write failed: disk full")

    monkeypatch.setattr(PRM, "persist_research_from_output", no_brief)

    async def runner(name, args):
        return "[Source: https://example.org/y]\nThe Y combinator gives fixed points; type theory bounds depth."

    async def cls(_d):
        return "research"

    res = await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                                owner_requested=owner)
    t = store.get_task(tid)
    assert t["status"] == status
    if owner:                                          # r3 m: the summary names the reason
        assert "research/depth.md does not exist" in res.summary
    assert ("research/depth.md does not exist" in t["failure_reason"] if owner
            else pa._HELD_MARK in t["result_summary"])


@pytest.mark.parametrize("desc", [
    "Survey depth models → research/models/depth.md",           # nested: no brief lands there
    "Survey depth models → research/depth.txt",
    "Survey depth models → research/v1.2-notes.md",              # a name the writer slugifies
    "Survey depth models → research/INDEX.md",                   # the brief writer's own index
])
def test_a_research_path_the_brief_writer_cannot_produce_is_built(desc):
    assert pa._route_project_work(desc, "research") == "coding"


def test_research_brief_slug():
    from ghost_agent.core.project_research import research_brief_slug
    assert research_brief_slug("research/recursion_depth.md") == "recursion_depth"
    assert research_brief_slug("research/a/b.md") is None
    assert research_brief_slug("research/x.MD") is None
    assert research_brief_slug("research/index.md") is None
    assert research_brief_slug(None) is None


async def test_a_snake_case_research_name_is_kept_exactly(store, tmp_path):
    pid = store.create_project("Recursion")
    tid = store.add_task(pid, "Survey depth models → research/recursion_depth.md")

    async def runner(name, args):
        return "[Source: https://example.org/y]\nThe Y combinator gives fixed points; type theory bounds depth."

    async def cls(_d):
        return "research"

    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          owner_requested=True)
    assert (Path(store.ensure_workspace(pid)) / "research" / "recursion_depth.md").is_file()
    assert store.get_task(tid)["status"] == "DONE"


@pytest.mark.parametrize("msg,mode", [
    ("proceed with task 3 on your own", "one"),               # M5
    ("run the tests by yourself please", "one"),
    ("can you work on this on your own?", "one"),
    ("continue without me", "one"),
    ("continue with the project on your own", "all"),
    ("do the remaining work", "all"),
])
def test_on_your_own_needs_the_project_as_its_object(msg, mode):
    assert pa.classify_advance_intent(msg)["mode"] == mode


@pytest.mark.parametrize("desc,cls", [
    ("Research: update on the EU AI act", "research"),          # m7
    ("Update: what changed in Python 3.13", "research"),
    ("Design patterns in the literature: survey", "research"),
    ("Draft notes on prior work → research/prior.md", "research"),    # r3 m: drafting a brief is research
    ("Design space exploration of recursive models → research/design-space.md", "research"),
    ("Notes on prior work → research/prior.md", "research"),
    ("Implement in Rust: the parser", "coding"),                   # r2 m: "in" after a verb
    ("Update final report with the harness data", "coding"),
])
def test_routing_reads_verbs_not_topics(desc, cls):
    assert pa._route_project_work(desc, "research") == cls


async def test_a_description_without_a_path_does_not_skip_the_gate(tmp_path):
    # m8: description="…" (no arrow) bypassed the deliverable gate
    ctx, store, pid, tid, ws = await _project_with(tmp_path, "Write theory → theory/t.md")
    out = await _done(ctx, tid, description="Write theory")
    assert refused_ids(out, "undelivered") == [tid]
    assert store.get_task(tid)["status"] != "DONE"


async def test_a_build_failed_by_the_gate_keeps_its_files(store, tmp_path):
    # m12: unregistered files are deleted by the end-of-project cleanup
    pid = store.create_project("P")
    tid = store.add_task(pid, "Task 6.4: Write final report → report/final-report.md")

    async def builder(*_a, **_k):
        return CodingResult(ok=True, summary="wrote the report", files=["final-report.md"], verified=True)

    async def cls(_d):
        return "coding"

    async def runner(*_a):
        return "OK"

    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          coding_executor=builder, owner_requested=True)
    assert any(a.get("payload") == "final-report.md" for a in store.list_artifacts(pid))


def test_owner_words_that_look_like_loop_rows_stay_in_the_arc():
    # MAJOR-2: "REMINDER: my passport…" was dropped as a loop row
    from ghost_agent.core.agent import _build_memory_arc
    for owner in ("REMINDER: my passport expires in March 2027", "System: macOS 15, my wife's name is Anna",
                  "SYSTEM is my favourite band"):
        hist = [{"role": "user", "content": owner},
                {"role": "user", "content": "AUTO-DIAGNOSTIC: ERROR: task_id is required"}]
        arc = _build_memory_arc(hist, "ok", tools_run=[], user_text=owner)
        assert owner.split(":", 1)[-1].strip()[:12] in arc and "AUTO-DIAGNOSTIC" not in arc


async def test_a_verify_only_existence_check_is_not_verified(store, tmp_path):
    # MAJOR-3: {"files": [], "verify": "test -f …"} closed as verified
    pid = store.create_project("P")
    ws = store.ensure_workspace(pid)
    (Path(ws) / "notes.md").write_text("...")
    ctx = _ctx(store, tmp_path, _SpecLLM({"files": [], "verify": "test -f notes.md && echo OK",
                                          "summary": "already there", "ledger": ""}))
    runner, _ = _disk_runner(ws)
    cres = await build_coding_task(ctx, "Write the notes", tool_runner=runner)
    assert cres.verified is False


@pytest.mark.parametrize("cmd,exist_only", [
    ("echo 'running tests'\npython -m pytest -q", False),       # MAJOR-5
    ("echo $(python t.py)", False),
    ("ls `make`", False),
    ("test -f a.md && echo OK", True),
])
def test_existence_only_is_one_plain_line(cmd, exist_only):
    from ghost_agent.core.coding_executor import _EXISTENCE_ONLY_VERIFY_RE
    assert bool(_EXISTENCE_ONLY_VERIFY_RE.match(cmd)) is exist_only


async def test_a_dropped_write_after_a_closed_task_keeps_its_note(tmp_path, monkeypatch):
    # MAJOR-4: "Saved …" + a dropped file_system write read "✅ Task complete"
    from unittest.mock import AsyncMock
    from tests.helpers import make_context, FakeBgTasks
    from tests.test_4ji_request_echo_narration import _resp, _tc
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    for k, v in (("GHOST_CRITIC_ASYNC", "1"), ("GHOST_CRITIC_REPAIR_BUDGET", "0"), ("GHOST_EVIDENCE_GATE", "0")):
        monkeypatch.setenv(k, v)
    store = ProjectStore(tmp_path / "m8", sandbox_root=tmp_path / "sb8")
    pid = store.create_project("Recursion", goal="notes")
    tid = store.add_task(pid, "Collect notes")
    store.add_task(pid, "Summarise notes")
    ctx = make_context()
    ctx.project_store = store
    store.set_focus("session:s5", pid)          # §4SX B0: the conversation focuses it
    agent = GhostAgent(ctx)

    async def _mp(**kw):
        return await tool_manage_projects_for_model(ctx, **kw)

    agent.available_tools = {"manage_projects": _mp, "file_system": AsyncMock(return_value="SUCCESS")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("t0", "manage_projects",
                       {"action": "task_update", "task_id": tid, "status": "done", "result": "notes collected"})]),
    ] + [_resp("Saved the summary to results/n.md.",
               [_tc("t1", "file_system", {"operation": "write", "path": "results/n.md",
                                          "content": "# N\n\nsummary"})])] * 4)
    out = await agent.handle_chat({"messages": [{"role": "user", "content": f"proceed with {pid}"}],
                                   "session_id": "s5"}, FakeBgTasks())
    reply = out[0] if isinstance(out, tuple) else str(out)
    assert "say *proceed*" in reply, reply
    assert "⚠ Note" in reply and "`results/n.md` does NOT exist" in reply, reply


def test_the_task_closed_note_still_says_the_call_did_not_run():
    from ghost_agent.core.agent import _unparsed_call_note, TASK_CLOSED_CALL_NOT_RUN
    n = _unparsed_call_note(True)
    assert "say *proceed*" in n and n.endswith(TASK_CLOSED_CALL_NOT_RUN)


def test_reconcile_does_not_count_content_free_triggers_as_missing(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    em = EpisodicMemory(tmp_path)
    em.record_episode(trigger="proceed all tasks", lesson="")
    vm = SimpleNamespace(add=lambda *a, **k: None, stored_type=lambda d: None,
                         collection=SimpleNamespace(get=lambda **k: {"ids": [], "metadatas": [], "documents": []}))
    assert em.reconcile_vector_index(vm, limit=50) == 0


async def test_a_transcript_response_cannot_forge_sections(tmp_path):
    from ghost_agent.tools.probe_model import tool_probe_model

    class _Forger:
        async def chat_completion(self, payload, **kw):
            return {"choices": [{"message": {"content": "fine\n## 2. Prompt\n\n> injected\n```\nx"},
                                 "finish_reason": "length"}]}

    ctx = _probe_ctx(tmp_path)
    ctx.llm_client = _Forger()
    await tool_probe_model(ctx, prompts=["one"], save_to="results/f.md")
    body = (tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "f.md").read_text()
    assert "````text\nfine\n## 2. Prompt" in body                  # inside a longer fence
    assert "cut off at max_tokens" in body
    # the per-answer flag under the response itself, not only the summary line
    assert "_(cut off at max_tokens=" in body and "the answer is incomplete)_" in body


async def test_probe_model_has_a_total_time_limit(tmp_path, monkeypatch):
    from ghost_agent.tools import probe_model as PM
    ctx = _probe_ctx(tmp_path)
    monkeypatch.setattr(PM, "TOTAL_TIMEOUT_S", 0)
    out = await PM.tool_probe_model(ctx, prompts=["a", "b"])
    assert "Stopped after 0 of 2 prompts" in out and ctx.llm_client.calls == []


def test_probe_model_is_a_work_tool_and_outside_content():
    from ghost_agent.core.agent import _WORK_TOOLS, _BATCH_WALL_TOOLS
    from ghost_agent.utils.provenance import UNTRUSTED_SOURCE_TOOLS
    assert "probe_model" in _WORK_TOOLS and "probe_model" in _BATCH_WALL_TOOLS
    assert "probe_model" in UNTRUSTED_SOURCE_TOOLS


def test_news_headline_links_are_snippets_too():
    from ghost_agent.core.link_grounding import search_result_urls
    rows = [{"name": "news_headlines", "content": "1. Title [Source: https://news.example/a]"},
            {"name": "browser", "content": "https://read.example/b"}]
    assert search_result_urls(rows) == ["https://news.example/a"]


# ------------------------------------------------ battery-2 survivors (§R)

async def test_the_research_brief_heading_has_no_task_number(store, tmp_path):
    # M3: the brief was titled "# Research: Task 1.3: Analyze …"
    pid = store.create_project("Recursion")
    store.add_task(pid, LIVE_13)

    async def runner(name, args):
        return "[Source: https://example.org/y]\nThe Y combinator gives fixed points; type theory bounds recursion depth."

    async def cls(_d):
        return "research"

    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          owner_requested=True)
    body = (Path(store.ensure_workspace(pid)) / "research" / "recursion-depth-models.md").read_text()
    assert "# Research: Analyze mathematical" in body and "Task 1.3" not in body


def test_todo_lines_are_not_words():
    # M24: eight TODO lines passed the eight-word floor
    assert hollow_document_reason("TODO\n" * 8, "results/a.md")
    assert hollow_document_reason("...\nTBD\n- TODO\nsome words here", "results/a.md")


def test_a_symlinked_deliverable_is_named_as_one(tmp_path):
    # M25: the link was reported as missing, which sends the model to write it again
    ws = tmp_path / "ws"
    (ws / "results").mkdir(parents=True)
    (ws / "results" / "real.md").write_text("# Real\n\n" + "word " * 20)
    os.symlink(ws / "results" / "real.md", ws / "results" / "r.md")
    assert "symbolic link" in deliverable_problem(ws, "Write it → results/r.md")


async def test_an_append_that_leaves_only_a_placeholder_is_refused():
    # M33: append "..." onto a missing .md wrote a 3-byte document
    from ghost_agent.core.coding_executor import _apply_file
    calls = []

    async def runner(name, args):
        calls.append(args.get("operation"))
        return "Error: file not found" if args.get("operation") == "read" else "Successfully wrote"

    written, why = await _apply_file(runner, {"path": "results/m.md", "append": "..."}, {})
    assert written is None and "only a placeholder" in why
    assert "write" not in calls


async def test_an_old_root_file_named_like_the_folder_gets_no_note(tmp_path):
    # M62: a root FILE "notes" made the note claim a root "notes/" folder
    from ghost_agent.tools.file_system import tool_write_file
    root, ws = _two_level_sandbox(tmp_path)
    (root / "notes").write_text("a file, not a folder")
    _age(root / "notes")
    out = await tool_write_file("notes/x.md", "# S\n\ntext", ws)
    assert str(out).startswith("SUCCESS") and "sandbox ROOT" not in str(out)


async def test_a_dotdot_save_path_is_refused_before_any_call(tmp_path):
    # M94: "results/../r.md" ran every prompt before the path was judged
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    out = await tool_probe_model(ctx, prompts=["a"], save_to="results/../r.md")
    assert "REJECTED" in out and ctx.llm_client.calls == []


def test_an_autonomous_phrase_naming_one_task_is_one_task():
    # M16b: "continue on your own with task 3" ran the whole project
    assert pa.classify_advance_intent("continue on your own with task 3")["mode"] == "one"
    assert pa.classify_advance_intent("continue on your own")["mode"] == "all"


def test_every_memory_arc_call_is_anchored_on_the_users_words():
    # M78c/d: a call site without user_text falls back to the wording regexes
    # that dropped "REMINDER: my passport…" (class guard over agent.py)
    import ast
    from ghost_agent.core import agent as A
    calls = [n for n in ast.walk(ast.parse(Path(A.__file__).read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_build_memory_arc"]
    assert len(calls) >= 2
    assert all(any(k.arg == "user_text" for k in c.keywords) for c in calls)


@pytest.mark.parametrize("closed", [True, False])
async def test_call_markup_after_a_closed_task_gets_the_task_closed_note(closed):
    # M74: the non-stream finalize said "could not be parsed" after a task close
    from tests.test_finalize_extraction import _fs, _make_agent
    from ghost_agent.core.agent import TASK_CLOSED_CALL_NOT_RUN
    agent = _make_agent()
    out, _, _ = await agent._finalize_and_return(_fs(
        final_ai_content='Notes collected.\n\n<tool_call>\n<function=manage_projects>\n'
                         '<parameter=action>\nautoadvance\n</parameter>\n</function>\n</tool_call>',
        task_closed_this_req=closed))
    assert ("say *proceed*" in out and TASK_CLOSED_CALL_NOT_RUN in out) is closed, out


def test_the_turn_loop_tells_finalize_whether_a_task_closed():
    # M76: FinalizeState(task_closed_this_req=False) at the call site
    import ast
    from ghost_agent.core import agent as A
    calls = [n for n in ast.walk(ast.parse(Path(A.__file__).read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "FinalizeState"]
    kw = [k.value for c in calls for k in c.keywords if k.arg == "task_closed_this_req"]
    assert kw and all(not isinstance(v, ast.Constant) for v in kw)
    assert all("_proj_task_closed_this_req" in ast.unparse(v) for v in kw)


# ------------------------------------------------------- r2 fresh-eye fixes

async def _closed_then(tmp_path, monkeypatch, first_text, later):
    from unittest.mock import AsyncMock
    from tests.helpers import make_context, FakeBgTasks
    from tests.test_4ji_request_echo_narration import _resp, _tc
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    for k, v in (("GHOST_CRITIC_ASYNC", "1"), ("GHOST_CRITIC_REPAIR_BUDGET", "0"), ("GHOST_EVIDENCE_GATE", "0")):
        monkeypatch.setenv(k, v)
    store = ProjectStore(tmp_path / "m5", sandbox_root=tmp_path / "sb5")
    pid = store.create_project("Recursion", goal="notes")
    tid = store.add_task(pid, "Collect notes")
    store.add_task(pid, "Summarise notes")
    ctx = make_context()
    ctx.project_store = store
    store.set_focus("session:s5", pid)          # §4SX B0: the conversation focuses it
    agent = GhostAgent(ctx)

    async def _mp(**kw):
        return await tool_manage_projects_for_model(ctx, **kw)

    agent.available_tools = {"manage_projects": _mp}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp(first_text, [_tc("t0", "manage_projects",
                               {"action": "task_update", "task_id": tid, "status": "done",
                                "result": "notes collected"})]),
    ] + [later(_resp, _tc)] * 6)
    out = await agent.handle_chat({"messages": [{"role": "user", "content": f"proceed with {pid}"}],
                                   "session_id": "s5"}, FakeBgTasks())
    return (out[0] if isinstance(out, tuple) else str(out)), ctx.llm_client.chat_completion.await_count


async def test_a_bare_next_task_call_after_a_close_is_the_designed_stop(tmp_path, monkeypatch):
    # r2 MAJOR-2: empty text + a dropped task_next call → retry → "ran out of budget"
    first = "The notes are collected: 12 items saved to notes.md, grouped by topic."
    reply, n = await _closed_then(tmp_path, monkeypatch, first,
                                  lambda r, t: r("", [t("t1", "manage_projects", {"action": "task_next"})]))
    assert "say *proceed*" in reply, reply
    assert "12 items" in reply and "budget" not in reply.lower(), reply
    assert n == 2, n


@pytest.mark.parametrize("first", ["", "Let me close task 1."])
async def test_a_closed_task_never_reads_not_finished(tmp_path, monkeypatch, first):
    # r2 MAJOR-2 / r3 MAJOR-2: nothing (or only narration), then a bare next-task
    # call twice → one retry, then the digest said "the task is NOT finished"
    reply, n = await _closed_then(tmp_path, monkeypatch, first,
                                  lambda r, t: r("", [t("t1", "manage_projects", {"action": "task_next"})]))
    from ghost_agent.core.reply_shape_check import refute_no_answer_fallback
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_NOTE
    assert n == 3, n                                         # the one retry is kept
    # r4 MAJOR-1: no "I ran out of this turn's budget" head (booked FAILED)
    assert reply.startswith(TASK_CLOSED_NOTE), reply
    assert "NOT finished" not in reply and "budget" not in reply.lower(), reply
    assert not refute_no_answer_fallback(reply)
    assert "Let me close task 1." not in reply               # narration still dropped


@pytest.mark.parametrize("work", [False, True])
@pytest.mark.parametrize("dropped_write", [False, True])
def test_our_notes_are_not_the_rest_that_lets_an_answer_be_dropped(work, dropped_write):
    # r2 MAJOR-1 / r3 MAJOR-1: the answer written beside task_update was
    # recorded as a segment; the task-closed note (and the dropped-write note)
    # made "what is left" look like an answer, so the answer was dropped
    from ghost_agent.core.agent import _dropped_mutation_note, _unparsed_call_note
    from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
    answer = "Task 1 is done: 12 notes saved to notes.md, grouped by topic."
    text = answer + "\n\n" + _unparsed_call_note(True)
    if dropped_write:
        text += _dropped_mutation_note(["file_system"], ["results/n.md"], creating=["results/n.md"])
    seg = WorkSegment(answer) if work else answer
    assert drop_pre_tool_segments(text, [seg]) == text


async def test_a_path_less_rewrite_then_done_does_not_skip_the_gate(tmp_path):
    # r2 m: rewrite "Write theory" (no arrow) in one call, close DONE in the next
    ctx, store, pid, tid, ws = await _project_with(tmp_path, "Write theory → theory/t.md")
    await tool_manage_projects(ctx, action="task_update", task_id=tid, description="Write theory")
    assert store.get_task(tid)["description"] == "Write theory → theory/t.md"
    out = await _done(ctx, tid)
    assert refused_ids(out, "undelivered") == [tid] and store.get_task(tid)["status"] != "DONE"
    # a rewrite naming a NEW path is how a changed plan names its real file
    await tool_manage_projects(ctx, action="task_update", task_id=tid, description="Write theory → theory/u.md")
    assert store.get_task(tid)["description"] == "Write theory → theory/u.md"


async def test_a_brief_never_takes_the_research_index_name(store, tmp_path):
    # r2 m: a topic slugified to "index" overwrote research/INDEX.md (macOS folds case)
    from ghost_agent.core.project_research import persist_research_from_output
    pid = store.create_project("P")

    async def summ(*_a, **_k):
        return "Indexes speed up lookups by keeping sorted keys beside the table rows."

    rr = await persist_research_from_output(_ctx(store, tmp_path), pid, "Index", "[Source: https://e.org/i]\nbody",
                                            llm_summarizer=summ)
    assert rr.ok and rr.path.endswith("research/index-brief.md")


async def test_an_unattended_research_close_without_its_file_is_held(store, tmp_path, monkeypatch):
    # N9: the brief "landed" (persist ok) but the declared file is not there
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("Recursion")
    tid = store.add_task(pid, "Survey depth models → research/depth.md")

    async def elsewhere(*_a, **_k):
        return PRM.ResearchResult(True, "t", slug="other", path="research/other.md",
                                  summary="The Y combinator gives fixed points; type theory bounds depth.")

    monkeypatch.setattr(PRM, "persist_research_from_output", elsewhere)

    async def runner(name, args):
        return "[Source: https://example.org/y]\nThe Y combinator gives fixed points; type theory bounds depth."

    async def cls(_d):
        return "research"

    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                          owner_requested=False)
    t = store.get_task(tid)
    assert t["status"] == "NEEDS_USER" and "research/depth.md does not exist" in t["result_summary"]


@pytest.mark.parametrize("goal", [
    "Evaluate Qwen on GSM8K arithmetic",                # r5 MAJOR-2: the model as direct object
    "Benchmark Llama 3 on multi-step reasoning",
    "Have Qwen answer 50 questions",
    "See if ChatGPT can count",
    "Test whether small LLMs can play chess",           # r4 MAJOR-3
    "Ask several LLMs 20 trivia questions",
    "Run the 200 items through the LLM",
    "Test whether Qwen can do multi-digit arithmetic",  # r3 MAJOR-3
    "Running introspection loops on an LLM",            # the live goal
    "Build and test an LLM chatbot for support",        # a held false positive — waits for the owner
])
def test_any_experiment_in_a_model_project_is_held(goal):
    # INVERTED (r5): three rounds of goal-phrase lists each missed common goals
    assert pa.is_model_experiment("Run the arithmetic test set", goal) is True


async def _brief(store, tmp_path, pid, slug="lit"):
    from ghost_agent.core.project_research import persist_research_from_output

    async def summ(*_a, **_k):
        return "Recursion depth is bounded by the fixed-point construction in typed calculi."

    rr = await persist_research_from_output(_ctx(store, tmp_path), pid, "Prior work", "[Source: https://e.org/a]\nb",
                                            llm_summarizer=summ, slug=slug)
    assert rr.ok
    return Path(store.ensure_workspace(pid)) / "research" / f"{slug}.md"


@pytest.mark.parametrize("state,cls", [
    ("absent", "research"),                  # a new brief
    ("ours", "research"),                    # our own unchanged brief: re-researched
    ("ours+notes", "coding"),                # r5 MAJOR-1: owner notes appended under OUR heading
    ("curated", "coding"),                   # someone's work
])
@pytest.mark.parametrize("desc", [
    "Integrate the benchmark results into the literature review → research/lit.md",   # r4 MAJOR-2
    "Update the literature review → research/lit.md",                                # r2 MAJOR-4
    "Literature review → research/lit.md",
])
async def test_a_research_path_holding_someones_work_is_built_not_overwritten(store, tmp_path, state, cls, desc):
    # decided on the BYTES: ours only while the file still hashes to what the writer wrote
    from ghost_agent.core.project_research import is_generated_brief
    pid = store.create_project("P")
    if state.startswith("ours"):
        f = await _brief(store, tmp_path, pid)
        if state == "ours+notes":
            f.write_text(f.read_text() + "\n## My notes (owner, 6 weeks)\n\nKeep these.\n")
    elif state == "curated":
        f = Path(store.ensure_workspace(pid)) / "research" / "lit.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# Research: recursion\n\nCurated by the owner over weeks.")   # even with our heading
    assert pa._route_project_work(desc, "research",
                                  curated=lambda rel: not is_generated_brief(store, pid, rel)) == cls


async def test_the_brief_writer_never_overwrites_someones_work(store, tmp_path):
    # r5: the route-then-write window — the writer itself refuses a named file
    # that is not its own brief, and moves a topic-derived name to a free one
    from ghost_agent.core.project_research import persist_research_from_output
    pid = store.create_project("P")
    rdir = Path(store.ensure_workspace(pid)) / "research"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "lit.md").write_text("# My review\n\nweeks of work")
    (rdir / "prior-work.md").write_text("# Prior work, by hand")

    async def summ(*_a, **_k):
        return "Recursion depth is bounded by the fixed-point construction in typed calculi."

    rr = await persist_research_from_output(_ctx(store, tmp_path), pid, "Prior work", "[Source: https://e.org/a]\nb",
                                            llm_summarizer=summ, slug="lit")
    assert not rr.ok and "not overwritten" in rr.error
    assert (rdir / "lit.md").read_text() == "# My review\n\nweeks of work"
    rr = await persist_research_from_output(_ctx(store, tmp_path), pid, "Prior work", "[Source: https://e.org/a]\nb",
                                            llm_summarizer=summ)
    assert rr.ok and rr.path.endswith("research/prior-work-2.md")
    assert (rdir / "prior-work.md").read_text() == "# Prior work, by hand"
    rr2 = await persist_research_from_output(_ctx(store, tmp_path), pid, "Prior work", "[Source: https://e.org/a]\nb",
                                             llm_summarizer=summ)
    assert rr2.path == rr.path                 # our own brief there is simply refreshed


async def test_autoadvance_does_not_overwrite_a_curated_research_file(store, tmp_path):
    pid = store.create_project("Recursion")
    store.add_task(pid, "Integrate the benchmark results into the literature review → research/lit.md")
    ws = Path(store.ensure_workspace(pid))
    (ws / "research").mkdir(parents=True, exist_ok=True)
    (ws / "research" / "lit.md").write_text("# Literature review\n\nCurated by the owner over weeks.")
    calls = []

    async def runner(name, args):
        calls.append(name)
        return "[Source: https://e.org/a]\nbody"

    async def cls(_d):
        return "research"

    res = await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner, llm_classifier=cls,
                                owner_requested=True)
    assert "web_search" not in calls and res.classification == "coding"
    assert (ws / "research" / "lit.md").read_text().startswith("# Literature review")


async def test_the_stream_never_says_task_complete_twice(monkeypatch):
    # r4 m1: narration + a next-task call after a close, the retry misses too —
    # the fallback carries the note, and the scrub's note chunk repeated it
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_NOTE
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finalize_stream_pins import make_stream_agent, _make_stream_state
    a = make_stream_agent()
    a._write_project_work_log_safe = AsyncMock()
    body = [_sse("Let me start task 1.2 now."),
            _sse("\n\n<tool_call>\n<function=manage_projects>\n<parameter=action>\n"),
            _sse("task_next\n</parameter>\n</function>\n</tool_call>"), b"data: [DONE]\n\n"]

    async def final_stream(payload, use_coding=False):
        for c in body:
            yield c

    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {
        "role": "assistant", "content": "Let me start task 1.2 now."}}]})
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    st = _make_stream_state(reg)
    st._proj_task_closed_this_req = True
    gen, _, _ = a._stream_final_generation(st)
    out = [c.decode("utf-8", "replace") async for c in gen]
    text = "".join(json.loads(x[6:])["choices"][0]["delta"].get("content", "")
                   for x in "".join(out).split("\n\n")
                   if x.startswith("data: {") and '"choices"' in x)
    assert text.count(TASK_CLOSED_NOTE) == 1, text


def test_a_greek_request_does_not_repair_the_one_task_stop_reply():
    # r5 MAJOR-3: the English task-closed reply was "repaired" into a tools-off
    # regeneration of "continue with the next task"
    from ghost_agent.core.agent import _no_answer_fallback_reply
    from ghost_agent.core.reply_language import reply_language_mismatch
    greek = "Συνέχισε με την επόμενη εργασία του έργου σε παρακαλώ"
    rows = [{"role": "tool", "tool_call_id": "c", "name": "web_search",
             "content": "### 1. Reuters\nRevolut confirmed the spoofed domain in a long English statement."}]
    reply = _no_answer_fallback_reply(rows, ask=greek, task_closed=True)
    assert reply_language_mismatch(greek, reply) is None
    # a model reply in the wrong language is still caught
    assert reply_language_mismatch(greek, "The task is complete and the next one is about the report "
                                          "structure, which I will now describe in some detail.")


@pytest.mark.parametrize("kw,line", [
    ({"stopped_by": "the edit-churn brake"}, "stopped by a check: the edit-churn brake"),
    ({"head_key": "no_answer_loop"}, "kept repeating itself"),
])
def test_a_closed_task_reply_still_names_what_stopped_it(kw, line):
    from ghost_agent.core.agent import _no_answer_fallback_reply
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_NOTE
    reply = _no_answer_fallback_reply([], task_closed=True, **kw)
    assert reply.startswith(TASK_CLOSED_NOTE) and line in reply


def test_a_long_work_segment_beside_more_own_text_is_kept():
    # R1b: the inner work-segment rule — what is left (our notes off) must
    # outweigh 1.5× the segment for it to go
    from ghost_agent.core.agent import _unparsed_call_note
    from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
    answer = "Task 1 is done. " + "The notes are grouped by topic and each cites its source. " * 5
    more = "Next I would summarise them. " + "Each summary keeps the citations intact for the report. " * 4
    text = answer.strip() + "\n\n" + more.strip() + "\n\n" + _unparsed_call_note(True)
    assert drop_pre_tool_segments(text, [WorkSegment(answer.strip())]) == text


async def test_a_planted_link_at_a_named_brief_is_refused(store, tmp_path):
    # R4j: the writer's own refusal, not only write_text_nofollow's
    from ghost_agent.core.project_research import persist_research_from_output
    pid = store.create_project("P")
    rdir = Path(store.ensure_workspace(pid)) / "research"
    rdir.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    os.symlink(victim, rdir / "lit.md")

    async def summ(*_a, **_k):
        return "Recursion depth is bounded by the fixed-point construction in typed calculi."

    rr = await persist_research_from_output(_ctx(store, tmp_path), pid, "Prior work", "[Source: https://e.org/a]\nb",
                                            llm_summarizer=summ, slug="lit")
    assert not rr.ok and "is a link or not a regular file" in rr.error
    assert victim.read_text() == "original"


async def test_brief_ownership_survives_a_slug_collision_and_a_full_index(store, tmp_path):
    # r6 m2/m3: the sha lived in the slug-keyed, 50-entry index
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("P")
    f = await _brief(store, tmp_path, pid, slug="research-transformer-basics")
    rel = "research/research-transformer-basics.md"
    rdir = f.parent
    (rdir / "transformer-basics.md").write_text("# Notes\n\nwritten directly")   # reconcile slug collides
    for i in range(60):
        (rdir / f"direct-{i}.md").write_text(f"# Note {i}\n\nbody")
    for _ in range(3):
        PRM.reconcile_research_dir(store, pid)
    assert PRM.is_generated_brief(store, pid, rel)


@pytest.mark.parametrize("before,after,ours", [
    ("", "", True),                                   # unchanged old brief: ours once backfilled
    ("", "\n## My notes\n\nkeep\n", False),         # edited AFTER the backfill: someone's
    ("\n## My notes\n\nkeep\n", "", False),         # notes appended before it: no footer, never recorded
])
async def test_old_briefs_are_ours_only_through_the_backfill(store, tmp_path, before, after, ours):
    # r6 m4 → r7 m1: no runtime heading+footer rule (a mid-file edit kept both);
    # a one-time backfill at deploy records the hash of each unchanged old brief
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("P")
    f = await _brief(store, tmp_path, pid)
    store._atomic_metadata_update(pid, lambda m: {**m, PRM.BRIEF_SHAS_META_KEY: {}})
    f.write_text(f.read_text() + before)
    assert PRM.is_generated_brief(store, pid, "research/lit.md") is False       # unhashed: someone's
    assert PRM.backfill_legacy_brief_shas(store, pid) == ([] if before else ["research/lit.md"])
    text = f.read_text()
    f.write_text(text.replace("\n\n", "\n\n" + after, 1) if after else text)   # a MID-file edit
    assert PRM.is_generated_brief(store, pid, "research/lit.md") is ours


async def test_the_backfill_skips_direct_writes_hashed_files_and_links(store, tmp_path):
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("P")
    f = await _brief(store, tmp_path, pid)
    store._atomic_metadata_update(pid, lambda m: {**m, PRM.BRIEF_SHAS_META_KEY: {}})
    store._atomic_metadata_update(pid, lambda m: {**m, PRM.INDEX_META_KEY: [
        {**e, "origin": "direct_write"} for e in m.get(PRM.INDEX_META_KEY) or []]})
    assert PRM.backfill_legacy_brief_shas(store, pid) == []              # a direct write is never ours
    store._atomic_metadata_update(pid, lambda m: {**m, PRM.INDEX_META_KEY: [
        {k: v for k, v in e.items() if k != "origin"} for e in m.get(PRM.INDEX_META_KEY) or []]})
    real = f.with_name("real.md")
    f.rename(real)
    f.symlink_to(real)
    assert PRM.backfill_legacy_brief_shas(store, pid) == []              # a link is never hashed
    f.unlink()
    real.rename(f)
    assert PRM.backfill_legacy_brief_shas(store, pid) == ["research/lit.md"]
    assert PRM.backfill_legacy_brief_shas(store, pid) == []              # idempotent


async def test_a_clone_keeps_its_briefs_own(store, tmp_path):
    # r7 m2: clone copies the workspace byte for byte but dropped the hashes
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("P")
    await _brief(store, tmp_path, pid)
    out = json.loads(await tool_manage_projects(_ctx(store, tmp_path), action="clone",
                                                project_id=pid, title="Copy"))
    new = (out.get("project") or {}).get("id") or out.get("new_project_id") or out.get("project_id")
    meta = store.get_project(new)["metadata"]
    assert meta[PRM.BRIEF_SHAS_META_KEY] == store.get_project(pid)["metadata"][PRM.BRIEF_SHAS_META_KEY]
    assert meta[PRM.INDEX_META_KEY]
    assert PRM.is_generated_brief(store, new, "research/lit.md")


async def test_the_model_never_reads_the_brief_hash_map(store, tmp_path):
    # r7 m3: list and the no-project reply dumped the uncapped sha map
    from ghost_agent.core import project_research as PRM
    from ghost_agent.tools.projects import _no_active_project
    pid = store.create_project("P")
    await _brief(store, tmp_path, pid)
    assert store.get_project(pid)["metadata"][PRM.BRIEF_SHAS_META_KEY]
    listed = await tool_manage_projects(_ctx(store, tmp_path), action="list")
    assert PRM.BRIEF_SHAS_META_KEY not in listed and "research/lit.md" in json.dumps(store.get_project(pid))
    assert PRM.BRIEF_SHAS_META_KEY not in _no_active_project(store)
    assert store.get_project(pid)["metadata"][PRM.BRIEF_SHAS_META_KEY]      # the store keeps it


async def test_a_new_version_keeps_its_briefs_own(store, tmp_path):
    from tests.test_projects_round1_fixes import _released_project_async
    from ghost_agent.core import project_research as PRM
    pid = await _released_project_async(store)
    store._atomic_metadata_update(pid, lambda m: {**m, PRM.BRIEF_SHAS_META_KEY: {"research/a.md": "x"}})
    out = json.loads(await tool_manage_projects(_ctx(store, tmp_path), action="create_version",
                                                project_id=pid, description="change A"))
    child = store.get_project(out["project"]["id"])
    assert child["metadata"][PRM.BRIEF_SHAS_META_KEY] == {"research/a.md": "x"}


def test_the_experiment_stop_lets_the_turn_decide():
    # r6 MAJOR-1: "run its prompts with probe_model" steered a false positive
    # ("Measure market share of LLM vendors") into asking the model for facts
    from ghost_agent.tools.projects import _advance_batch_instruction
    batch = SimpleNamespace(stop_reason="model_experiment", count=2, advanced=[
        {"task_id": "t9", "summary": "model experiment: Measure market share of LLM vendors"}])
    text = _advance_batch_instruction(batch, 1)
    assert "If they are NOT" in text and "never ask a model for facts it should look up" in text
    assert "a build), do it the normal way — web_search / read / write the file" in text


async def test_the_model_cannot_forge_a_brief_hash(store, tmp_path):
    # r8 m4: a forged hash would make an owner's file "ours" to the brief writer
    from ghost_agent.core import project_research as PRM
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    pid = store.create_project("P")
    f = Path(store.ensure_workspace(pid)) / "research" / "lit.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# Owner's literature review\n")
    import hashlib
    forged = {"research/lit.md": hashlib.sha256(f.read_bytes()).hexdigest()}
    await tool_manage_projects_for_model(_ctx(store, tmp_path), action="update", project_id=pid,
                                         metadata={PRM.BRIEF_SHAS_META_KEY: forged, "note": "x"})
    meta = store.get_project(pid)["metadata"]
    assert PRM.BRIEF_SHAS_META_KEY not in meta and meta.get("note") == "x"
    assert PRM.is_generated_brief(store, pid, "research/lit.md") is False


async def test_the_backfill_dry_run_names_what_it_would_record_and_records_nothing(store, tmp_path):
    # r8 m2: the repair's dry run listed every unhashed entry, not what --apply records
    from ghost_agent.core import project_research as PRM
    pid = store.create_project("P")
    await _brief(store, tmp_path, pid)
    await _brief(store, tmp_path, pid, slug="other")
    store._atomic_metadata_update(pid, lambda m: {**m, PRM.BRIEF_SHAS_META_KEY: {}})
    o = Path(store.ensure_workspace(pid)) / "research" / "other.md"
    o.write_text(o.read_text() + "\nmy notes\n")                       # no footer at the end
    assert PRM.backfill_legacy_brief_shas(store, pid, dry_run=True) == ["research/lit.md"]
    assert PRM._brief_shas(store, pid) == {}
    assert PRM.backfill_legacy_brief_shas(store, pid) == ["research/lit.md"]


# ── r9 (final review) ────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [
    "<tool_call>\n<function=manage_projects>\n<parameter=action>",
    '<tool_call>\n<function name="manage_projects">',
    '<tool_call>{"name": "manage_projects", "arguments": {"action"',
    '<tool name="x">{"a":',
])
def test_a_call_cut_open_keeps_what_is_appended_after_it(cut):
    # r9 A-MAJOR-1: an unclosed block ran to the end, so the scrub of the
    # record ate the retry's answer and every note appended after the cut
    from ghost_agent.core.reply_smoothing import close_open_calls, treat_reply
    full = close_open_calls("Now let me start task 4.\n\n" + cut)
    rec = treat_reply(full + "\n\nThe notes are grouped by topic.\n\n_Source check: 2 links_", n_real_tools=1)
    assert "grouped by topic" in rec and "Source check" in rec and "<tool" not in rec
    assert close_open_calls("plain text") == "plain text"
    closed = "a <tool_call>{}</tool_call> b"
    assert close_open_calls(closed) == closed


async def test_the_streamed_record_keeps_the_retry_answer_after_a_cut_call(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finalize_stream_pins import make_stream_agent, _make_stream_state
    a = make_stream_agent()
    a._write_project_work_log_safe = AsyncMock()
    seen = []
    a._finish_turn_scratch = lambda _rid, text, *a_, **k: seen.append(text)
    body = [_sse("Let me start task 1.2 now."),
            _sse("\n\n<tool_call>\n<function=file_system>\n<parameter=content>\n")]
    body += [_sse("A long research document line. " * 4) for _ in range(40)]
    body += [_sse("</parameter>\n</function>\n</tool_call>"), b"data: [DONE]\n\n"]

    async def final_stream(payload, use_coding=False):
        for c in body:
            yield c

    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {
        "role": "assistant", "content": "Task 1.1 collected 12 notes, grouped by topic in notes.md."}}]})
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    st = _make_stream_state(reg)
    st._proj_task_closed_this_req = True
    gen, _, _ = a._stream_final_generation(st)
    async for _c in gen:
        pass
    assert seen, "the record sink was not reached"
    treated = seen[-1].split("\n<")[0]
    assert "grouped by topic" in seen[-1]
    assert "grouped by topic" in treated or "grouped by topic" in seen[-1].split("<tool_call>")[0]


def test_the_turn_state_check_never_judges_a_system_written_reply():
    # r9 A-MAJOR-2: judged after the note was stripped, the task-closed
    # fallback (English digest) was refuted on language under a Greek request
    from ghost_agent.core import agent as A
    from ghost_agent.core.turn_state_check import refute_turn_state
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_NOTE
    req = "Προχώρα με την επόμενη εργασία του έργου και πες μου τι έγινε"
    tools = [{"name": "manage_projects", "content": "SUCCESS: task_3 marked DONE"},
             {"name": "web_search", "content": "Results for the query about the parser library and the line "
              "ending handling in the module. The library normalises carriage returns before tokenising."},
             {"name": "execute", "content": "EXIT CODE: 1\nSTDERR: Traceback (most recent call last):\n"
              "ValueError: the configuration file could not be parsed because a required section is missing"}]
    fb = A._no_answer_fallback_reply(tools, ask=req, task_closed=True)
    assert refute_turn_state(request=req, reply=fb, tools_run=tools) == []
    # the same text as the MODEL's own reply is still judged
    own = fb.replace(TASK_CLOSED_NOTE, "The parser task is finished and the result is described below.", 1)
    assert refute_turn_state(request=req, reply=own, tools_run=tools) != []


def test_a_loop_after_a_closed_task_carries_no_abort_marker():
    # r9 A-m1: "✅ Task complete" beside "not a finished answer" + an abort marker
    from ghost_agent.core import agent as A
    from ghost_agent.core.agent import reply_carries_abort_marker
    fb = A._no_answer_fallback_reply([{"name": "manage_projects", "content": "SUCCESS: done"}],
                                     ask="proceed", task_closed=True)
    out = A._task_closed_loop_reply(fb)
    assert not reply_carries_abort_marker(out) and "not a finished" not in out
    assert out.count("The reply after it") == 1
    stopped = A._no_answer_fallback_reply([], ask="proceed", stopped_by="loop breaker", task_closed=True)
    assert A._task_closed_loop_reply(stopped) == stopped


def test_a_link_seen_in_a_page_this_request_read_is_not_blamed_on_an_earlier_one(tmp_path):
    # r9 A-m4
    from ghost_agent.core.agent import unopened_citation_note
    doc = tmp_path / "report.md"
    doc.write_text("See https://example.org/paper and https://other.org/x\n")
    tools = [
        {"name": "browser", "content": "SUCCESS page text … cites https://example.org/paper …",
         "args": {"url": "https://a.org"}},
        {"name": "file_system", "content": "SUCCESS: wrote report.md",
         "args": {"operation": "write", "path": "report.md", "content": doc.read_text()}},
    ]
    note = unopened_citation_note(tools, [tmp_path])
    assert "only in the text of a page or result this request read" in note
    assert "1 in no result of this request" in note


@pytest.mark.parametrize("desc", [
    "Task 6.1: Draft the recursive self-awareness theory → theory/recursive-self-awareness-theory.md",
    "Task 2.2: Introspection report on your own layers → results/introspection.md",
])
def test_self_analysis_that_promises_a_file_it_cannot_write_is_built(desc):
    # r9 B-MAJOR-1: it was saved as a brief elsewhere and FAILED on every advance
    assert pa.is_self_referential(desc)
    assert pa._route_project_work(desc, "research") == "coding"
    assert pa._route_project_work("Analyse your own recursion depth", "research") == "research"
    assert pa._route_project_work("Analyse your own recursion → research/depth.md", "research") == "research"
    assert pa._route_project_work("Analyse your own recursion → research/depth.md", "research",
                                  curated=lambda _r: True) == "coding"


async def test_a_self_referential_experiment_is_held_too(tmp_path):
    # r9 B-MAJOR-1: "Run Layer 3 … self-awareness" took the self-analysis path,
    # which wrote the "responses" itself
    store = ProjectStore(tmp_path / "mem9", sandbox_root=tmp_path / "sb9")
    pid = store.create_project("Recursion", goal=GOAL)
    t1 = store.add_task(pid, "Task 3.4: Run Layer 3 — test for self-awareness in your own replies")
    assert pa.is_self_referential(store.get_task(t1)["description"])

    async def cls(_d):
        return "research"

    batch = await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=None,
                                  tool_runner=lambda *a: None, llm_classifier=cls,
                                  coding_executor=None, owner_requested=True)
    assert batch.stop_reason == "model_experiment"
    assert store.get_task(t1)["status"] == "PENDING"


def test_a_declared_path_with_the_project_prefix_is_the_projects_file():
    # r9 B-m1
    assert declared_deliverable("Write it → /workspace/projects/448d5afa86f0/results/x.md") == "results/x.md"


async def test_a_worker_run_is_recorded_under_the_model_that_answered(tmp_path):
    # r9 C-MAJOR: the transcript said the main model's name for a worker run
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    orig = ctx.llm_client.chat_completion

    async def served(payload, **kw):
        d = await orig(payload, **kw)
        return {**d, "model": "worker-model-7b", "_ghost_leg": {"served_by": "worker", "fell_back_from": ""}}

    ctx.llm_client.chat_completion = served
    await tool_probe_model(ctx, prompts=["x"], node="worker", save_to="results/w.md")
    body = (tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "w.md").read_text()
    assert "model: worker-model-7b" in body and "qwen3.6" not in body

    async def fell_back(payload, **kw):
        d = await orig(payload, **kw)
        return {**d, "_ghost_leg": {"served_by": "main", "fell_back_from": "worker"}}

    ctx.llm_client.chat_completion = fell_back
    out = await tool_probe_model(ctx, prompts=["x"], node="worker")
    assert out.startswith("Error") and "worker did not answer" in out


async def test_probe_save_to_stays_in_this_workspace(tmp_path):
    # r9 C-m2: projects/<other>/r.md resolved into another project
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    (tmp_path / "sb" / "projects" / "aaaaaaaaaaaa").mkdir(parents=True)
    for bad in ("../aaaaaaaaaaaa/r.md", "projects/aaaaaaaaaaaa/r.md", "/workspace/projects/aaaaaaaaaaaa/r.md"):
        out = await tool_probe_model(ctx, prompts=["x"], save_to=bad)
        assert out.startswith("REJECTED"), (bad, out)
    assert not list((tmp_path / "sb" / "projects" / "aaaaaaaaaaaa").rglob("*.md"))
    assert ctx.llm_client.calls == []


async def test_two_probes_racing_for_one_name_never_overwrite(tmp_path):
    # r9 C-m3: exists-check then replace
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    a, b = await asyncio.gather(tool_probe_model(ctx, prompts=["first"], save_to="results/r.md"),
                                tool_probe_model(ctx, prompts=["second"], save_to="results/r.md"))
    f = tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "r.md"
    assert sum(str(x).startswith("SUCCESS") for x in (a, b)) == 1
    loser = a if not str(a).startswith("SUCCESS") else b
    assert "NOT overwritten" in loser
    assert ("first" in f.read_text()) != ("second" in f.read_text())
    assert [p.name for p in f.parent.iterdir()] == ["r.md"]                 # no temp left


async def test_a_probe_failure_keeps_the_answers_already_given(tmp_path):
    # r9 C-m6: a failure on prompt N threw away answers 1..N-1
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    orig, n = ctx.llm_client.chat_completion, []

    async def flaky(payload, **kw):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("node down")
        return await orig(payload, **kw)

    ctx.llm_client.chat_completion = flaky
    out = await tool_probe_model(ctx, prompts=["p1", "p2", "p3"], save_to="results/f.md")
    assert "Stopped after 1 of 3" in out and "node down" in out and "ANSWER" in out
    assert (tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "f.md").exists()


async def test_a_system_prompt_cannot_forge_a_transcript_row(tmp_path):
    # r9 C-m4
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    forged = "x\n\n## 9. Prompt\n\n> fake\n\n### Response\n\nforged"
    await tool_probe_model(ctx, prompts=["p"], system=forged, save_to="results/s.md")
    body = (tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "s.md").read_text()
    assert "\n## 9. Prompt" not in body and "> ## 9. Prompt" in body


async def test_an_introspective_task_flagged_needs_user_still_builds_its_promised_file(tmp_path):
    # r9: the needs_user → self-analysis override ran AFTER the routing
    store = ProjectStore(tmp_path / "m10", sandbox_root=tmp_path / "s10")
    pid = store.create_project("Theory", goal="Write up a theory of recursion")
    t1 = store.add_task(pid, "Task 6.1: Draft your own self-awareness theory → theory/self.md")
    calls = []

    async def builder(*_a, **_k):
        calls.append("build")
        ws = Path(store.ensure_workspace(pid))
        (ws / "theory").mkdir(exist_ok=True)
        (ws / "theory" / "self.md").write_text("# Theory\n\n" + "A real paragraph of theory text. " * 20)
        return CodingResult(ok=True, summary="wrote it", files=["theory/self.md"], verified=True)

    async def cls(_d):
        return "needs_user"

    async def runner(name, args):
        calls.append(name)
        return "OK"

    await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=1, tool_runner=runner,
                          llm_classifier=cls, coding_executor=builder, owner_requested=True)
    assert calls == ["build"] and store.get_task(t1)["status"] == "DONE"


def test_every_loop_fallback_after_a_closed_task_skips_the_abort_wrapper():
    # r9 A-m1, at the sites: each loop site picks _task_closed_loop_reply when
    # a task closed, the abort wrapper otherwise (enumerated from the tree)
    import ast
    from ghost_agent.core import agent as A
    picks = []
    for n in ast.walk(ast.parse(Path(A.__file__).read_text())):
        if isinstance(n, ast.IfExp) and getattr(n.test, "id", "") == "_proj_task_closed_this_req":
            names = lambda e: {getattr(x, "id", "") for x in ast.walk(e)}
            picks.append(("_task_closed_loop_reply" in names(n.body),
                          bool(names(n.orelse) & {"cross_turn_loop_fallback", "forced_final_loop_fallback"})))
    assert picks.count((True, True)) == 3, picks


async def test_the_streamed_record_says_task_complete_once_when_the_retry_misses_too(monkeypatch):
    # r9: the scrubbed call became the task note while the fallback carried it too
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_NOTE
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finalize_stream_pins import make_stream_agent, _make_stream_state
    a = make_stream_agent()
    a._write_project_work_log_safe = AsyncMock()
    seen = []
    a._finish_turn_scratch = lambda _rid, text, *a_, **k: seen.append(text)
    body = [_sse("Let me start task 1.2 now."),
            _sse("\n\n<tool_call>\n<function=file_system>\n<parameter=content>\n")]
    body += [_sse("A long research document line. " * 4) for _ in range(40)]
    body += [_sse("</parameter>\n</function>\n</tool_call>"), b"data: [DONE]\n\n"]

    async def final_stream(payload, use_coding=False):
        for c in body:
            yield c

    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {
        "role": "assistant", "content": "Let me start task 1.2 now."}}]})
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    st = _make_stream_state(reg)
    st._proj_task_closed_this_req = True
    gen, _, _ = a._stream_final_generation(st)
    async for _c in gen:
        pass
    assert seen
    treated, _, raw = seen[-1].partition("\n" + "Let me start task 1.2 now.")
    # the sink gets "<treated>\n<raw>": once in each, never twice in the record
    assert seen[-1].count(TASK_CLOSED_NOTE) <= 2 and treated.count(TASK_CLOSED_NOTE) <= 1, seen[-1]


async def test_a_write_cut_by_the_one_task_stop_never_runs_as_a_report_write(monkeypatch):
    # r10 MAJOR-1: closing the cut call at the break made it PARSE, and the
    # forced-final report path wrote the half file
    from unittest.mock import AsyncMock, MagicMock
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finalize_stream_pins import make_stream_agent, _make_stream_state
    a = make_stream_agent()
    a._write_project_work_log_safe = AsyncMock()
    ran = []

    async def _writes(calls, *_a, **_k):
        ran.append(calls)
        return True, []

    a._run_streamed_report_writes = _writes
    body = [_sse("Task 1.1 is complete."),
            _sse("\n\n<tool_call>\n<function=file_system>\n<parameter=operation>\nwrite\n</parameter>\n"
                 "<parameter=path>\nresults/r.md\n</parameter>\n<parameter=content>\n# Report\n")]
    body += [_sse("half of the report line. " * 4) for _ in range(40)]
    body += [b"data: [DONE]\n\n"]

    async def final_stream(payload, use_coding=False):
        for c in body:
            yield c

    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {
        "role": "assistant", "content": "Task 1.1 is complete."}}]})
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    st = _make_stream_state(reg)
    st._proj_task_closed_this_req = True
    st.ff_breaker = True
    gen, _, _ = a._stream_final_generation(st)
    async for _c in gen:
        pass
    assert all(not calls for calls in ran), ran         # no half write reached the writer


def test_close_open_calls_skips_a_fenced_example():
    # r10 m: a fenced <tool_call> example got the closer; the real cut call leaked
    from ghost_agent.core.reply_smoothing import close_open_calls, treat_reply
    text = ("Example:\n```\n<tool_call>{\"name\": \"x\"}</tool_call>\n```\n\nNow:\n\n"
            "<tool_call>\n<function=file_system>\n<parameter=content>\nhalf")
    closed = close_open_calls(text)
    assert closed.endswith("</tool_call>") and closed.count("</tool_call>") == 2
    rec = treat_reply(closed + "\n\nAfter the cut.", n_real_tools=1)
    assert "half" not in rec and "After the cut." in rec
    # an UNCLOSED fenced example hides the rest from the scrub too: left alone, no stray closer
    odd = "```\n<tool_call>{\"name\": \"x\"}\n```\n\n<tool_call>\n<function=f>\nhalf"
    assert close_open_calls(odd) == odd


async def test_a_folder_swapped_for_a_link_during_the_probe_is_not_written_through(tmp_path):
    # r10 MAJOR-4: containment was checked once, before minutes of model calls
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    ws = tmp_path / "sb" / "projects" / "448d5afa86f0"
    (ws / "results").mkdir()
    outside = tmp_path / "OUTSIDE"
    outside.mkdir()
    orig = ctx.llm_client.chat_completion

    async def swap(payload, **kw):
        (ws / "results").rmdir()
        (ws / "results").symlink_to(outside, target_is_directory=True)
        return await orig(payload, **kw)

    ctx.llm_client.chat_completion = swap
    out = await tool_probe_model(ctx, prompts=["x"], save_to="results/run.md")
    assert not out.startswith("SUCCESS") and "saving failed" in out
    assert list(outside.iterdir()) == []


async def test_a_lone_surrogate_never_loses_the_answers(tmp_path):
    # r10 m: UnicodeEncodeError is a ValueError — the tool raised
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    out = await tool_probe_model(ctx, prompts=["bad \ud800 char"], save_to="results/u.md")
    assert out.startswith("SUCCESS")
    assert (tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / "u.md").exists()


def test_a_flood_after_a_closed_task_is_called_a_flood():
    from ghost_agent.core import agent as A
    fb = A._no_answer_fallback_reply([], ask="proceed", task_closed=True)
    assert "runaway burst of tool calls" in A._task_closed_loop_reply(fb, flood=True)
    assert "kept repeating itself" in A._task_closed_loop_reply(fb)


def test_a_link_is_seen_only_when_that_exact_link_was_shown(tmp_path):
    # r10 m: a substring test counted https://x.com/a as seen in https://x.com/abc
    from ghost_agent.core.agent import unopened_citation_note
    doc = tmp_path / "report.md"
    doc.write_text("See https://x.com/a and https://y.org/page.\n")
    tools = [{"name": "browser", "content": "SUCCESS text … https://x.com/abc … (https://y.org/page)",
              "args": {"url": "https://q.org"}},
             {"name": "file_system", "content": "SUCCESS: wrote report.md",
              "args": {"operation": "write", "path": "report.md", "content": doc.read_text()}}]
    note = unopened_citation_note(tools, [tmp_path])
    assert "1 only in the text of a page" in note and "1 in no result of this request" in note


async def test_an_introspective_experiment_flagged_needs_user_is_held(tmp_path):
    # r10 MAJOR-2: "choose" → needs_user → the override → self-analysis wrote the "responses"
    store = ProjectStore(tmp_path / "m11", sandbox_root=tmp_path / "s11")
    pid = store.create_project("Recursion", goal=GOAL)
    t1 = store.add_task(pid, "Run Layer 3 self-awareness prompts on the model: test whether I truly "
                             "choose my responses")
    assert pa.is_self_referential(store.get_task(t1)["description"])

    async def cls(_d):
        return "needs_user"

    async def runner(name, args):
        return "OK"

    batch = await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=None, tool_runner=runner,
                                  llm_classifier=cls, coding_executor=None, owner_requested=True)
    assert batch.stop_reason == "model_experiment"
    assert store.get_task(t1)["status"] == "PENDING"


async def test_the_record_keeps_the_retry_when_quote_markers_preceded_the_cut(monkeypatch):
    # r11 MAJOR: the record-time close matched the RAW prefix, but quote
    # markers are stripped from full_content after the loop — no match, and
    # the scrub ate the retry answer again
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core import quote_bound as QB
    sys.path.insert(0, os.path.dirname(__file__))
    from test_finalize_stream_pins import make_stream_agent, _make_stream_state
    tok = QB.quote_bound_context.set("append")
    try:
        a = make_stream_agent()
        a._write_project_work_log_safe = AsyncMock()
        seen = []
        a._finish_turn_scratch = lambda _rid, text, *a_, **k: seen.append(text)
        body = [_sse('Let me start task 2 now [src: "start task 2"].'),
                _sse("\n\n<tool_call>\n<function=file_system>\n<parameter=content>\n")]
        body += [_sse("A long research document line. " * 4) for _ in range(40)]
        body += [b"data: [DONE]\n\n"]

        async def final_stream(payload, use_coding=False):
            for c in body:
                yield c

        a.context.llm_client.stream_chat_completion = final_stream
        a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {
            "role": "assistant", "content": "Task 1.1 collected 12 notes, grouped by topic in notes.md."}}]})
        reg = MagicMock()
        reg.is_cancelled.return_value = False
        st = _make_stream_state(reg)
        st._proj_task_closed_this_req = True
        gen, _, _ = a._stream_final_generation(st)
        async for _c in gen:
            pass
    finally:
        QB.quote_bound_context.reset(tok)
    assert seen
    treated = seen[-1].split("\nLet me start task 2")[0] if seen[-1].count("Let me start task 2") > 1 else seen[-1]
    assert "grouped by topic" in treated, seen[-1]


def test_the_cut_close_helper_only_closes_at_the_cut():
    from ghost_agent.core.agent import _qa_closed_at_cut
    cut = "Done.\n\n<tool_call>\n<function=file_system>\n<parameter=content>\nhalf"
    full = cut + "\n\nThe answer after the cut."
    out = _qa_closed_at_cut(full, cut)
    assert out.startswith(cut) and "</tool_call>" in out and out.endswith("The answer after the cut.")
    assert _qa_closed_at_cut(full, None) == full
    assert _qa_closed_at_cut("something else", cut) == "something else"


def test_a_link_with_brackets_shown_by_a_page_counts_as_seen(tmp_path):
    # r11 m2: the URL reader cut "Python_(programming_language)" at "("
    from ghost_agent.core.agent import unopened_citation_note
    u = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    doc = tmp_path / "report.md"
    doc.write_text(f"See {u}\n")
    tools = [{"name": "browser", "content": f"SUCCESS page … {u} …", "args": {"url": "https://q.org"}},
             {"name": "file_system", "content": "SUCCESS: wrote report.md",
              "args": {"operation": "write", "path": "report.md", "content": doc.read_text()}}]
    note = unopened_citation_note(tools, [tmp_path])
    assert "1 only in the text of a page" in note and "no result of this request" not in note


async def test_a_workspace_swapped_for_a_link_during_the_probe_is_not_written_through(tmp_path):
    # r12 MAJOR: the walk's root (the project folder) was opened following links
    import shutil
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    ws = tmp_path / "sb" / "projects" / "448d5afa86f0"
    outside = tmp_path / "OUTSIDE"
    outside.mkdir()
    orig = ctx.llm_client.chat_completion

    async def swap(payload, **kw):
        shutil.rmtree(ws)
        ws.symlink_to(outside, target_is_directory=True)
        return await orig(payload, **kw)

    ctx.llm_client.chat_completion = swap
    out = await tool_probe_model(ctx, prompts=["x"], save_to="results/run.md")
    assert not out.startswith("SUCCESS") and "saving failed" in out
    assert list(outside.rglob("*")) == []


async def test_a_long_save_to_name_still_saves(tmp_path):
    # r13 m2: the temp name added ~30 bytes to the target's — ENAMETOOLONG after the calls were paid
    from ghost_agent.tools.probe_model import tool_probe_model
    ctx = _probe_ctx(tmp_path)
    name = "r" * 240 + ".md"
    out = await tool_probe_model(ctx, prompts=["x"], save_to=f"results/{name}")
    assert out.startswith("SUCCESS"), out[:200]
    assert (tmp_path / "sb" / "projects" / "448d5afa86f0" / "results" / name).exists()


def test_the_announcement_check_ignores_the_one_task_notes():
    """§4QA N25: the one-task stop note and the "call was not run" note are
    the SYSTEM's — an announcement followed by either is still only an
    announcement (the raw text with the note is not narration)."""
    from ghost_agent.core.agent import _announced_then_called
    from ghost_agent.core.reply_smoothing import TASK_CLOSED_CALL_NOT_RUN, TASK_CLOSED_NOTE, narration_only
    said = "Let me check the next task now."
    for note in (TASK_CLOSED_NOTE, TASK_CLOSED_CALL_NOT_RUN):
        assert not narration_only(said + "\n\n" + note, tool_attempt=True)   # the world the strip matters in
        assert _announced_then_called(said + "\n\n" + note)


def test_probe_model_is_advertised_and_dispatched(mock_context):
    """§4QA M100/M101: the experiment instrument is a built-in — in the schema
    the model sees AND in the dispatch table."""
    from ghost_agent.tools import registry as R
    names = {t["function"]["name"] for t in R.get_active_tool_definitions(mock_context, None, serve_tuned=False)}
    assert "probe_model" in names
    assert "probe_model" in {t["function"]["name"] for t in R.TOOL_DEFINITIONS}
    assert callable(R.get_available_tools(mock_context).get("probe_model"))


async def test_the_project_list_hides_the_brief_hash_map(store, tmp_path):
    """§4QA R23: the brief hashes are system state — `list` shows each project
    through `_model_view`, like the no-project listing (R23b)."""
    store.create_project("Hashes", metadata={"research_brief_shas": {"research/a.md": "f" * 64}, "keep": "me"})
    out = await tool_manage_projects(_ctx(store, tmp_path), action="list", metadata={"verbose": True})
    assert '"keep": "me"' in out                       # the verbose listing carries the metadata
    assert "research_brief_shas" not in out and "f" * 64 not in out


async def test_a_needs_user_introspective_task_with_a_promised_file_is_built_at_the_override(tmp_path):
    """§4QA S6b: the wording here KEEPS needs_user through the gate (a
    "choose" task about decision-making), so the self-analysis override is
    what routes it — and a promised theory/ file is built, never briefed."""
    store = ProjectStore(tmp_path / "m11", sandbox_root=tmp_path / "s11")
    pid = store.create_project("Agency", goal="Study agency")
    desc = ("Illusion of Agency: Evaluate whether I truly choose responses or merely predict them. "
            "Analyze decision-making → theory/agency.md")
    from ghost_agent.core.project_advancer import settle_gate
    assert settle_gate(desc, "needs_user", default="research") == "needs_user"   # the override's world
    t1 = store.add_task(pid, desc)
    calls = []

    async def builder(*_a, **_k):
        calls.append("build")
        ws = Path(store.ensure_workspace(pid))
        (ws / "theory").mkdir(exist_ok=True)
        (ws / "theory" / "agency.md").write_text("# Agency\n\n" + "A real paragraph of analysis text. " * 20)
        return CodingResult(ok=True, summary="wrote it", files=["theory/agency.md"], verified=True)

    async def cls(_d):
        return "needs_user"

    async def runner(name, args):
        calls.append(name)
        return "OK"

    await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=1, tool_runner=runner,
                          llm_classifier=cls, coding_executor=builder, owner_requested=True)
    assert calls == ["build"] and store.get_task(t1)["status"] == "DONE"
