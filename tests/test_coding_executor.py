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
        # §4PG: the smoke gate now deletes its own script after the run — a delete is not a write
        # §4QA-d: a file outside the snapshot is read before a full overwrite — a read is not a write
        return [a for (n, a) in self.calls if n == "file_system" and a.get("operation") not in ("delete", "read")]

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








# --------------------------------------------- think-ate-the-budget retry

class ThinkHogLLM:
    """First call burns the whole budget inside <think> (content empty,
    reasoning is prose with NO spec — the live 2026-07-06 failure); the
    no-think retry then answers with a clean spec."""
    def __init__(self, retry_content=None):
        self.calls = 0
        self.payloads = []
        self.retry_content = SPEC_OK if retry_content is None else retry_content

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.calls += 1
        self.payloads.append(payload)
        if self.calls == 1:
            return {"choices": [{"message": {
                "content": "",
                "reasoning_content": "The user wants me to build... " * 200}}]}
        return {"choices": [{"message": {
            "content": self.retry_content, "reasoning_content": ""}}]}






async def _instant_sleep(*_a, **_kw):
    return None






# ------------------------------------------------- background lane (contention)





# ------------------------------------- empty files + verify (already-built leaf)

_SPEC_NO_FILES_WITH_VERIFY = json.dumps({
    "files": [],
    "verify": "python3 src/model.py",
    "summary": "model.py already exists and runs",
    "ledger": "model.py: decoder-only transformer",
})








# ----------------------------------------- research reference reaches the build







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






# ---------------------------------------------------------------------------
# Verify must TEST, not manage processes (2026-07-25, req 8ec512cf: the
# spec's verify ran `fuser -k 8101/tcp; node server.js &`, murdered the
# supervised service, and the operator's browser hit a dead port).
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# Raw-newline-in-JSON-string repair (2026-07-25: three malformed-spec
# retries in one afternoon, all the same shape — literal newlines inside
# find/append/content string values).
# ---------------------------------------------------------------------------







