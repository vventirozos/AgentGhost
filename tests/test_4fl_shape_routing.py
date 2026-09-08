"""§4FL pins: the coding executor is chosen by LEAF SHAPE. A leaf that grows
an existing file of GHOST_LEAF_GROW_LINES (80) lines or more runs the agentic
loop; a fresh or small leaf keeps the spec executor; project metadata and
the env override both ways; the rule has a kill switch; the seam at the top
of build_coding_task passes the shape inputs through.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from ghost_agent.core import coding_loop as cl
from ghost_agent.core import coding_executor as ce


def _files(lines, path="app.py"):
    return {path: "\n".join(f"line {i}" for i in range(lines)) + "\n"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GHOST_CODING_EXECUTOR", "GHOST_LEAF_ROUTE_BY_SHAPE", "GHOST_LEAF_GROW_LINES"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("desc,files,expected", [
    ("Create app.py with a Flask app", None, ("fresh", "", 0)),
    ("Create app.py with a Flask app", {}, ("fresh", "", 0)),
    ("Extend app.py: PUT /api/expenses/<id>", _files(200), ("growing", "app.py", 200)),
    ("Extend src/app.py: add a route", _files(200, "src/app.py"), ("growing", "src/app.py", 200)),
    ("Extend app.py: add a route", _files(200, "src/app.py"), ("growing", "src/app.py", 200)),   # basename
    ("Extend app.py: add a route", _files(20), ("fresh", "", 0)),                               # too small
    ("Extend myapp.py: add a route", _files(200), ("fresh", "", 0)),                            # not the same file
    ("Add tests for the summary endpoint", _files(200), ("fresh", "", 0)),                      # names no file
    ("Extend app.py and utils.py", {**_files(120), **_files(300, "utils.py")}, ("growing", "utils.py", 300)),
])
def test_leaf_shape_names_the_largest_existing_file_at_or_above_the_threshold(desc, files, expected):
    assert cl.leaf_shape(desc, files) == expected


def test_threshold_is_the_env_knob_with_the_measured_default(monkeypatch):
    assert cl.GROW_LINES_DEFAULT == 80
    assert cl.leaf_shape("Extend app.py", _files(80)) == ("growing", "app.py", 80)
    assert cl.leaf_shape("Extend app.py", _files(79)) == ("fresh", "", 0)
    monkeypatch.setenv("GHOST_LEAF_GROW_LINES", "300")
    assert cl.leaf_shape("Extend app.py", _files(200)) == ("fresh", "", 0)


class _Store:
    def __init__(self, executor=None):
        self.executor = executor

    def get_project(self, pid):
        return {"id": pid, "metadata": ({"executor": self.executor} if self.executor else {})}


def _ctx(executor=None):
    return SimpleNamespace(project_store=_Store(executor), current_project_id=None)


def test_precedence_env_then_metadata_then_shape_then_spec(monkeypatch):
    grow = dict(description="Extend app.py: add CSV export", existing_files=_files(200))
    fresh = dict(description="Create fib.py", existing_files=None)
    # shape decides when nothing else says
    assert cl.executor_kind(_ctx(), "p1", **grow) == "agentic"
    assert cl.executor_kind(_ctx(), "p1", **fresh) == "spec"
    # metadata overrides the shape BOTH ways
    assert cl.executor_kind(_ctx("spec"), "p1", **grow) == "spec"
    assert cl.executor_kind(_ctx("agentic"), "p1", **fresh) == "agentic"
    # env overrides metadata
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "spec")
    assert cl.executor_kind(_ctx("agentic"), "p1", **grow) == "spec"
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "agentic")
    assert cl.executor_kind(_ctx("spec"), "p1", **fresh) == "agentic"


def test_kill_switch_turns_the_shape_rule_off(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO)
    grow = dict(description="Extend app.py: add CSV export", existing_files=_files(200))
    assert cl.executor_kind(_ctx(), "p1", **grow) == "agentic"
    assert "routing leaf to the agentic loop" in caplog.text and "app.py" in caplog.text
    monkeypatch.setenv("GHOST_LEAF_ROUTE_BY_SHAPE", "0")
    assert cl.executor_kind(_ctx(), "p1", **grow) == "spec"


@pytest.mark.asyncio
async def test_seam_routes_a_growing_leaf_to_the_loop_and_a_fresh_one_to_spec(monkeypatch):
    """Executed at the REAL seam: build_coding_task with the advancer's
    existing_files dict. World where it fails: the seam calls executor_kind
    without the shape inputs (every leaf stays on the spec path)."""
    calls = []

    async def fake_agentic(context, description, **kw):
        calls.append(("agentic", description, kw.get("project_id")))
        return ce.CodingResult(True, "loop", files=["app.py"])
    monkeypatch.setattr(cl, "build_coding_task_agentic", fake_agentic)

    from tests.test_coding_executor import FakeLLM, FakeRunner, SPEC_OK
    ctx = SimpleNamespace(llm_client=FakeLLM(SPEC_OK), project_store=_Store(), current_project_id=None,
                          args=SimpleNamespace(model="m"))
    runner = FakeRunner()
    res = await ce.build_coding_task(ctx, "Extend app.py: add CSV export", tool_runner=runner,
                                     existing_files=_files(200), project_id="p9")
    assert res.ok and res.files == ["app.py"]
    assert calls == [("agentic", "Extend app.py: add CSV export", "p9")]
    # a fresh leaf takes the spec path (the fake spec writes parser.py + README.md)
    res2 = await ce.build_coding_task(ctx, "Create fib.py with tests", tool_runner=runner,
                                      existing_files=None, project_id="p9")
    assert res2.ok and res2.files == ["parser.py", "README.md"] and len(calls) == 1
    # a small existing file stays on the spec path too
    res3 = await ce.build_coding_task(ctx, "Extend app.py: add CSV export", tool_runner=runner,
                                      existing_files=_files(20), project_id="p9")
    assert res3.ok and res3.files == ["parser.py", "README.md"] and len(calls) == 1
