"""Fixes for autonomous-advance misbehaviour observed live: a CODING project
whose build leaves ("File Explorer app", "Snake game") were classified as
research and web-searched, marked DONE with no code.

Covers:
  1a. _infer_kind  — a build-flavoured goal creates a CODING project.
  1b. CODING projects bypass the (mislabelling) LLM classifier.
  2a. a coding leaf with no build path FAILS instead of being web-searched.
  2b. research briefs are NOT registered as deliverables in CODING projects.
  3.  _gather_project_files / existing_files feed the executor for cumulative
      builds; the build spec prompt instructs "extend, don't recreate".
  4.  file_system tool description warns against monolithic writes.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import project_advancer as PA
from ghost_agent.core.project_advancer import (
    advance_once, _gather_project_files,
)
from ghost_agent.core.coding_executor import CodingResult, _generate_build_spec
from ghost_agent.core.planning import ProjectPlan
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import _infer_kind, MANAGE_PROJECTS_TOOL_DEF


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


async def _dummy_runner(name, args):
    return "OK"


# ---------------------------------------------------------------- 1a infer kind

@pytest.mark.parametrize("title,goal,expected", [
    ("browser OS", "Create a single-file browser OS with desktop and 5 apps", "CODING"),
    ("snake", "build a snake game with canvas", "CODING"),
    ("site", "implement a portfolio website", "CODING"),
    ("parser", "a python module that parses CSV", "CODING"),
    ("market notes", "research the EV market and summarize trends", "GENERAL"),
    ("trip", "plan a vacation itinerary for Japan", "GENERAL"),
])
def test_infer_kind(title, goal, expected):
    assert _infer_kind(title, goal, "") == expected


def test_infer_kind_explicit_wins():
    # an explicit non-default kind is never overridden
    assert _infer_kind("x", "build an app", "GENERAL") == "CODING"  # inferred
    assert _infer_kind("research foo", "summarize bar", "CODING") == "CODING"  # explicit


# ---------------------------------------------------------------- 1b bypass LLM

@pytest.mark.asyncio
async def test_coding_project_ignores_llm_research_mislabel(store):
    pid = store.create_project("P", kind="CODING")
    tid = ProjectPlan(store, pid).add_task("the central component")  # no keyword
    seen = {}

    async def exec_(ctx, d, **kw):
        seen["called"] = True
        return CodingResult(True, "built it", files=["x.py"])

    async def mislabel(_d):
        return "research"  # the small model's wrong guess — must be IGNORED

    res = await advance_once(
        SimpleNamespace(project_store=store, llm_client=None), pid,
        tool_runner=_dummy_runner, llm_classifier=mislabel, coding_executor=exec_)

    assert res.classification == "coding"
    assert seen.get("called") is True          # executor ran, not web_search
    assert store.get_task(tid)["status"] == "DONE"


@pytest.mark.asyncio
async def test_general_project_still_honors_llm_label(store):
    pid = store.create_project("P", kind="GENERAL")
    tid = ProjectPlan(store, pid).add_task("the central component")
    seen = {}

    async def exec_(ctx, d, **kw):
        seen["called"] = True
        return CodingResult(True, "x", files=[])

    async def classifier(_d):
        return "research"

    res = await advance_once(
        SimpleNamespace(project_store=store, llm_client=None), pid,
        tool_runner=_dummy_runner, llm_classifier=classifier,
        coding_executor=exec_)

    assert res.classification == "research"
    assert "called" not in seen                # coding executor NOT run


# ---------------------------------------------------------------- 2a fail, don't research

@pytest.mark.asyncio
async def test_coding_leaf_without_build_path_fails(store):
    pid = store.create_project("P", kind="CODING")
    tid = ProjectPlan(store, pid).add_task("build the thing")
    # no coding_executor, no code_generator → no way to build
    res = await advance_once(
        SimpleNamespace(project_store=store, llm_client=None), pid,
        tool_runner=_dummy_runner)
    assert res.classification == "coding"
    assert store.get_task(tid)["status"] == "FAILED"
    # it must NOT have been marked DONE via a web_search
    assert "could not be built" in res.summary


# ---------------------------------------------------------------- 2b no research-brief deliverable

@pytest.mark.asyncio
async def test_research_brief_not_a_deliverable_in_coding_project(store):
    from ghost_agent.core.project_research import _persist

    async def fake_summary(topic, output):
        return f"summary of {topic}"

    ctx = SimpleNamespace(project_store=store, llm_client=None)

    # CODING project: brief written + indexed, but NOT a file artifact
    cpid = store.create_project("C", kind="CODING")
    ctid = ProjectPlan(store, cpid).add_task("research an edge case")
    r1 = await _persist(ctx, cpid, "topic A", "some search text",
                        task_id=ctid, llm_summarizer=fake_summary)
    assert r1.ok
    file_arts = [a for a in store.list_artifacts(project_id=cpid) if a["kind"] == "file"]
    assert file_arts == []

    # GENERAL project: brief IS a deliverable
    gpid = store.create_project("G", kind="GENERAL")
    gtid = ProjectPlan(store, gpid).add_task("research a topic")
    await _persist(ctx, gpid, "topic B", "some search text",
                   task_id=gtid, llm_summarizer=fake_summary)
    gfile_arts = [a for a in store.list_artifacts(project_id=gpid) if a["kind"] == "file"]
    assert len(gfile_arts) == 1


# ---------------------------------------------------------------- 3 cumulative builds

def test_gather_project_files_reads_and_skips_research(store):
    pid = store.create_project("P", kind="CODING")
    base = store.sandbox_root / "projects" / pid
    base.mkdir(parents=True, exist_ok=True)
    (base / "index.html").write_text("<html><body>shell</body></html>")
    (base / "research").mkdir()
    (base / "research" / "x.md").write_text("# brief")
    files = _gather_project_files(store, pid)
    assert "index.html" in files
    assert "shell" in files["index.html"]
    assert not any(k.startswith("research/") for k in files)  # briefs excluded


def test_gather_large_file_is_not_emptied(store):
    # A grown file (>20KB, the old budget) must NOT be passed as an empty
    # marker — that blinded the non-regression guard and let later tasks
    # clobber it (observed live: apps overwritten and lost).
    pid = store.create_project("P", kind="CODING")
    base = store.sandbox_root / "projects" / pid
    base.mkdir(parents=True, exist_ok=True)
    big = "<html><body>" + ("<div id='x'></div>" * 3000) + "</body></html>"
    (base / "index.html").write_text(big)
    files = _gather_project_files(store, pid)
    assert files.get("index.html")                 # non-empty → guard sees it
    assert len(files["index.html"]) > 20_000        # the real (large) content


def test_gather_project_files_skips_binary_media(store):
    # PNG/WAV read with errors="replace" are pure noise to the executor, yet
    # they consumed the file cap and char budget — crowding real sources out
    # of existing_files and blinding the non-regression guard to them.
    pid = store.create_project("P", kind="CODING")
    base = store.sandbox_root / "projects" / pid
    base.mkdir(parents=True, exist_ok=True)
    (base / "index.html").write_text("<html><body>app</body></html>")
    (base / "sprite.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 256)
    (base / "theme.wav").write_bytes(b"RIFF" + b"\x00" * 128)
    files = _gather_project_files(store, pid)
    assert "index.html" in files
    assert not any(k.endswith((".png", ".wav")) for k in files)


def test_gather_binary_files_do_not_consume_the_cap(store):
    # A dozen media files sort ahead of the real source and used to fill the
    # 12-file cap before it was ever reached.
    pid = store.create_project("P", kind="CODING")
    base = store.sandbox_root / "projects" / pid
    base.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        (base / f"frame_{i:02d}.png").write_bytes(b"\x89PNG" + b"\x00" * 64)
    (base / "z_app.js").write_text("console.log('app')")
    files = _gather_project_files(store, pid)
    assert files == {"z_app.js": "console.log('app')"}


@pytest.mark.asyncio
async def test_build_spec_prompt_instructs_extend_when_files_exist():
    captured = {}

    class LLM:
        async def chat_completion(self, payload, is_background=False, **_kw):
            captured["user"] = payload["messages"][-1]["content"]
            return {"choices": [{"message": {"content": "{}"}}]}

    await _generate_build_spec(LLM(), "m", "add the editor app", "",
                               existing_files={"index.html": "<html>shell</html>"})
    u = captured["user"]
    assert "EXISTING PROJECT FILES" in u
    assert "index.html" in u and "<html>shell</html>" in u
    # steers toward the incremental primitives instead of re-emitting the file
    assert "append" in u and "ADDS to these" in u


# ---------------------------------------------------------------- 4 large-file guidance

def test_file_system_description_warns_against_mega_write():
    defs = MANAGE_PROJECTS_TOOL_DEF  # sanity that tool defs import
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    fs = [t for t in TOOL_DEFINITIONS
          if t.get("function", {}).get("name") == "file_system"][0]
    desc = fs["function"]["description"].lower()
    assert "large file" in desc
    assert "skeleton" in desc and "replace" in desc
