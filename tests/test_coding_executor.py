"""Real coding executor for autonomous batches (build-and-verify), plus the
classification fix that routes file/build leaves to it instead of web_search.

Replaces the single-command path that produced "theatrical completion" — a
task marked DONE having written nothing.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.coding_executor import build_coding_task, CodingResult
from ghost_agent.core import project_advancer as PA
from ghost_agent.core.project_advancer import classify_task, advance_once
from ghost_agent.memory.projects import ProjectStore


# ----------------------------------------------------------------- fakes

class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    async def chat_completion(self, payload):
        self.calls += 1
        return {"choices": [{"message": {"content": self.content}}]}


class FakeRunner:
    """Records tool calls; canned outputs per tool."""
    def __init__(self, write_out="Successfully wrote file.", verify_out="OK"):
        self.calls = []
        self.write_out = write_out
        self.verify_out = verify_out

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "file_system":
            return self.write_out
        if name == "execute":
            return self.verify_out
        return ""

    def writes(self):
        return [a for (n, a) in self.calls if n == "file_system"]

    def execs(self):
        return [a for (n, a) in self.calls if n == "execute"]


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


SPEC_OK = json.dumps({
    "files": [
        {"path": "parser.py", "content": "def parse(p):\n    return []\n"},
        {"path": "README.md", "content": "# parser\n"},
    ],
    "verify": "python3 -c 'import ast; ast.parse(open(\"parser.py\").read())'",
    "summary": "wrote parser.py with parse(path) -> list",
    "ledger": "parser.py exposes parse(path) -> list[dict]",
})


# ----------------------------------------------------------------- executor

@pytest.mark.asyncio
async def test_build_happy_path_writes_and_verifies():
    runner = FakeRunner(verify_out="OK")
    res = await build_coding_task(_ctx(FakeLLM(SPEC_OK)), "build the parser",
                                  tool_runner=runner)
    assert res.ok
    assert res.files == ["parser.py", "README.md"]
    assert "parse(path)" in res.ledger_note
    # both files written, then the verify command executed
    written = [w["path"] for w in runner.writes()]
    assert written == ["parser.py", "README.md"]
    assert all(w["operation"] == "write" for w in runner.writes())
    assert len(runner.execs()) == 1


@pytest.mark.asyncio
async def test_build_fails_when_verify_fails():
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n SyntaxError")
    res = await build_coding_task(_ctx(FakeLLM(SPEC_OK)), "build x", tool_runner=runner)
    assert not res.ok
    assert "verify failed" in res.summary
    # files were still written before verify caught the breakage
    assert res.files == ["parser.py", "README.md"]


@pytest.mark.asyncio
async def test_build_fails_on_write_rejection():
    runner = FakeRunner(write_out="SYSTEM ERROR: path escapes sandbox")
    res = await build_coding_task(_ctx(FakeLLM(SPEC_OK)), "build x", tool_runner=runner)
    assert not res.ok
    assert "write rejected" in res.summary
    assert len(runner.execs()) == 0  # never reached verify


@pytest.mark.asyncio
async def test_build_fails_when_no_files_in_spec():
    res = await build_coding_task(_ctx(FakeLLM('{"files":[],"verify":""}')),
                                  "x", tool_runner=FakeRunner())
    assert not res.ok
    assert "no file" in res.summary.lower()


@pytest.mark.asyncio
async def test_build_no_verify_is_ok():
    spec = json.dumps({"files": [{"path": "a.txt", "content": "a"}],
                       "verify": "", "summary": "wrote a.txt"})
    runner = FakeRunner()
    res = await build_coding_task(_ctx(FakeLLM(spec)), "x", tool_runner=runner)
    assert res.ok and res.files == ["a.txt"]
    assert len(runner.execs()) == 0


@pytest.mark.asyncio
async def test_build_unavailable_without_llm_or_runner():
    res = await build_coding_task(SimpleNamespace(llm_client=None), "x", tool_runner=FakeRunner())
    assert not res.ok
    res2 = await build_coding_task(_ctx(FakeLLM(SPEC_OK)), "x", tool_runner=None)
    assert not res2.ok


@pytest.mark.asyncio
async def test_build_tolerates_markdown_fenced_json():
    fenced = "Sure!\n```json\n" + SPEC_OK + "\n```\n"
    res = await build_coding_task(_ctx(FakeLLM(fenced)), "x", tool_runner=FakeRunner())
    assert res.ok and res.files == ["parser.py", "README.md"]


# ----------------------------------------------------------------- classification

@pytest.mark.parametrize("desc", [
    "create a file hello.txt containing the word hello",
    "build the parser module",
    "write app.js for the frontend",
    "add a style.css with the layout",
    "implement the solver in solver.py",
])
def test_build_tasks_classify_as_coding(desc):
    assert classify_task(desc) == "coding"


def test_research_still_wins_over_coding_keyword():
    # "research" beats a coding signal in the same description
    assert classify_task("research how to write a .py parser") == "research"


def test_default_bucket_biases_coding_projects():
    # an unlabelled leaf is research by default, coding in a CODING project
    assert classify_task("the central thing") == "research"
    assert classify_task("the central thing", default="coding") == "coding"


# ----------------------------------------------------------------- advance_once integration

@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.mark.asyncio
async def test_advance_once_uses_coding_executor_and_registers_files(store):
    from ghost_agent.core.planning import ProjectPlan, TaskStatus
    pid = store.create_project("Builder", kind="CODING")
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("build the parser module")
    runner = FakeRunner(verify_out="OK")

    res = await advance_once(
        SimpleNamespace(project_store=store, llm_client=None),
        pid,
        tool_runner=runner,
        coding_executor=lambda ctx, d, *, tool_runner, ledger: build_coding_task(
            _ctx(FakeLLM(SPEC_OK)), d, tool_runner=tool_runner, ledger=ledger),
    )
    assert res.classification == "coding"
    assert store.get_task(tid)["status"] == "DONE"
    # produced files are registered as deliverables, and the ledger updated
    arts = [a["payload"] for a in store.list_artifacts(project_id=pid) if a["kind"] == "file"]
    assert "parser.py" in arts
    assert "parse(path)" in store.get_ledger(pid)


@pytest.mark.asyncio
async def test_advance_once_failed_build_marks_failed(store):
    from ghost_agent.core.planning import ProjectPlan
    pid = store.create_project("Builder", kind="CODING")
    tid = ProjectPlan(store, pid).add_task("build the broken thing")
    runner = FakeRunner(verify_out="ERROR: broken")

    res = await advance_once(
        SimpleNamespace(project_store=store, llm_client=None),
        pid, tool_runner=runner,
        coding_executor=lambda ctx, d, *, tool_runner, ledger: build_coding_task(
            _ctx(FakeLLM(SPEC_OK)), d, tool_runner=tool_runner, ledger=ledger),
    )
    assert store.get_task(tid)["status"] == "FAILED"
    assert "failed" in res.summary.lower()
