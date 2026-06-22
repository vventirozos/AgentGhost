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
        self.bg_flags = []
        self.payloads = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.calls += 1
        self.bg_flags.append(is_background)
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.content}}]}

    def last_user_prompt(self):
        msgs = self.payloads[-1]["messages"]
        return next(m["content"] for m in msgs if m["role"] == "user")


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
async def test_oversized_content_fails_loudly_not_silently_truncated():
    # A full file larger than the cap must FAIL (so the retry can split it),
    # never be silently sliced mid-code into a broken file.
    from ghost_agent.core.coding_executor import MAX_CONTENT_CHARS
    huge = "x" * (MAX_CONTENT_CHARS + 50)
    spec = json.dumps({"files": [{"path": "big.py", "content": huge}],
                       "verify": ""})
    runner = FakeRunner()
    res = await build_coding_task(_ctx(FakeLLM(spec)), "x", tool_runner=runner)
    assert not res.ok
    assert "split it across multiple files" in res.summary
    # nothing was written (no silent truncated file on disk)
    assert runner.writes() == []


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


# ------------------------------------------------- reasoning-channel spec (Bug 1)

class FakeReasoningLLM:
    """A reasoning model (Qwen) that routes its whole reply into the separate
    `reasoning_content` field and leaves `content` empty — the live failure
    mode that logged `len=0` and killed coding leaves with 'no file spec'."""
    def __init__(self, *, content="", reasoning=""):
        self.content = content
        self.reasoning = reasoning
        self.calls = 0
        self.bg_flags = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.calls += 1
        self.bg_flags.append(is_background)
        return {"choices": [{"message": {
            "content": self.content, "reasoning_content": self.reasoning}}]}


@pytest.mark.asyncio
async def test_build_recovers_spec_from_reasoning_content():
    # content is empty; the JSON spec lives entirely in reasoning_content.
    llm = FakeReasoningLLM(content="", reasoning=SPEC_OK)
    runner = FakeRunner(verify_out="OK")
    res = await build_coding_task(_ctx(llm), "build the parser", tool_runner=runner)
    assert res.ok, res.summary
    assert res.files == ["parser.py", "README.md"]
    # recovered on the FIRST attempt — no wasteful retry
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_build_reasoning_spec_after_unclosed_think():
    # The closing </think> never arrived (budget exhausted mid-think) so the
    # parser kept everything in reasoning; spec still recoverable.
    reasoning = "let me think about the files I need...\n" + SPEC_OK
    llm = FakeReasoningLLM(content="", reasoning=reasoning)
    res = await build_coding_task(_ctx(llm), "build x", tool_runner=FakeRunner())
    assert res.ok and res.files == ["parser.py", "README.md"]


@pytest.mark.asyncio
async def test_build_prefers_content_when_present():
    # When content holds the spec, reasoning noise must not override it.
    llm = FakeReasoningLLM(content=SPEC_OK, reasoning="some unrelated musing")
    res = await build_coding_task(_ctx(llm), "build x", tool_runner=FakeRunner())
    assert res.ok and res.files == ["parser.py", "README.md"]


async def _instant_sleep(*_a, **_kw):
    return None


@pytest.mark.asyncio
async def test_empty_response_fails_with_honest_contention_message(monkeypatch):
    # A fully empty upstream response (content=0 reasoning=0) is contention, not
    # the model failing to spec — report it honestly and stop after a bounded
    # number of backoff retries rather than burning every attempt.
    from ghost_agent.core import coding_executor as ce
    monkeypatch.setattr(ce.asyncio, "sleep", _instant_sleep)
    llm = FakeReasoningLLM(content="", reasoning="")
    res = await build_coding_task(_ctx(llm), "x", tool_runner=FakeRunner())
    assert not res.ok
    assert "empty responses" in res.summary.lower()
    assert "contention" in res.summary.lower()
    assert "no file" not in res.summary.lower()        # NOT the mislabel
    # stopped at the empty-retry cap, not all MAX_ATTEMPTS
    assert llm.calls == ce.MAX_EMPTY_RETRIES


@pytest.mark.asyncio
async def test_empty_then_valid_recovers_after_backoff(monkeypatch):
    # First call empty (transient), second returns a valid spec → success.
    from ghost_agent.core import coding_executor as ce
    monkeypatch.setattr(ce.asyncio, "sleep", _instant_sleep)

    class FlakyLLM:
        def __init__(self):
            self.calls = 0
        async def chat_completion(self, payload, is_background=False, **_kw):
            self.calls += 1
            body = "" if self.calls == 1 else SPEC_OK
            return {"choices": [{"message": {"content": body}}]}

    res = await build_coding_task(_ctx(FlakyLLM()), "build the parser",
                                  tool_runner=FakeRunner(verify_out="OK"))
    assert res.ok, res.summary
    assert res.files == ["parser.py", "README.md"]


# ------------------------------------------------- background lane (contention)

@pytest.mark.asyncio
async def test_spec_call_is_foreground_by_default():
    # The user-initiated tool path: the user is waiting, so spec generation
    # runs foreground.
    llm = FakeLLM(SPEC_OK)
    await build_coding_task(_ctx(llm), "build x", tool_runner=FakeRunner())
    assert llm.bg_flags == [False]


@pytest.mark.asyncio
async def test_spec_call_uses_background_when_requested():
    # The idle autoadvancer passes is_background=True so its 8K-token spec call
    # defers to a user who starts typing mid-build.
    llm = FakeLLM(SPEC_OK)
    await build_coding_task(_ctx(llm), "build x", tool_runner=FakeRunner(),
                            is_background=True)
    assert llm.bg_flags == [True]


# ------------------------------------- empty files + verify (already-built leaf)

_SPEC_NO_FILES_WITH_VERIFY = json.dumps({
    "files": [],
    "verify": "python3 src/model.py",
    "summary": "model.py already exists and runs",
    "ledger": "model.py: decoder-only transformer",
})


@pytest.mark.asyncio
async def test_empty_files_with_passing_verify_succeeds():
    # The deliverable was built by a prior turn; the model emits no files + a
    # verify. The executor must run the verify and PASS, not fail on no-files
    # (observed live: Model Architecture task FAILED in autoadvance though
    # model.py existed and ran).
    runner = FakeRunner(verify_out="OK")
    res = await build_coding_task(_ctx(FakeLLM(_SPEC_NO_FILES_WITH_VERIFY)),
                                  "build the model", tool_runner=runner)
    assert res.ok, res.summary
    assert res.files == []
    assert len(runner.execs()) == 1                 # the verify ran
    assert "already exists" in res.summary


@pytest.mark.asyncio
async def test_empty_files_with_failing_verify_fails_with_verify_reason():
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n NameError")
    res = await build_coding_task(_ctx(FakeLLM(_SPEC_NO_FILES_WITH_VERIFY)),
                                  "build the model", tool_runner=runner)
    assert not res.ok
    assert "verify failed" in res.summary
    # the failure carries the real verify error, not the generic "no file spec"
    assert "no file spec" not in res.summary


@pytest.mark.asyncio
async def test_empty_files_without_verify_still_fails():
    spec = json.dumps({"files": [], "verify": "", "summary": "nothing"})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "x", tool_runner=FakeRunner())
    assert not res.ok
    assert "no file" in res.summary.lower()


# ----------------------------------------- research reference reaches the build

@pytest.mark.asyncio
async def test_research_context_is_injected_into_the_spec_prompt():
    llm = FakeLLM(SPEC_OK)
    await build_coding_task(
        _ctx(llm), "build the model", tool_runner=FakeRunner(),
        research_context={"research/transformer.md": "d_model=128, RoPE, RMSNorm."})
    prompt = llm.last_user_prompt()
    assert "PROJECT RESEARCH" in prompt
    assert "research/transformer.md" in prompt
    assert "d_model=128" in prompt
    # framed as reference, not an editable file
    assert "NOT files to edit" in prompt


@pytest.mark.asyncio
async def test_no_research_context_omits_the_section():
    llm = FakeLLM(SPEC_OK)
    await build_coding_task(_ctx(llm), "build x", tool_runner=FakeRunner())
    assert "PROJECT RESEARCH" not in llm.last_user_prompt()


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
        coding_executor=lambda ctx, d, **kw: build_coding_task(
            _ctx(FakeLLM(SPEC_OK)), d, tool_runner=kw["tool_runner"],
            ledger=kw.get("ledger", ""), existing_files=kw.get("existing_files")),
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
        coding_executor=lambda ctx, d, **kw: build_coding_task(
            _ctx(FakeLLM(SPEC_OK)), d, tool_runner=kw["tool_runner"],
            ledger=kw.get("ledger", ""), existing_files=kw.get("existing_files")),
    )
    assert store.get_task(tid)["status"] == "FAILED"
    assert "failed" in res.summary.lower()
