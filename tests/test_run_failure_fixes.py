"""Tests for the fixes prompted by the live Minecraft-clone failure run:

  A. Browser render verification — an objective "did anything render?" check
     on the screenshot (kills the false "it works" over a blank/sky frame)
     plus a click_center/settle interaction for canvas games.
  B. file_system replace robustness — anchor-based block match (rescues an
     edit whose boundaries are stable but the middle drifted) + param-name
     aliases (the live run's first replace failed purely on `old_text=`).
  C. manage_projects DONE-gate — a visual/runnable-artifact task can't be
     marked DONE with no verification evidence.
  D. create returns the project working-directory note (so the model writes
     files into the project dir instead of the sandbox root).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.tools.browser import (
    analyze_screenshot_render, _build_op_payload, _runner_script,
    _pre_interaction_line,
)
from ghost_agent.tools.file_system import tool_file_system, _anchor_block_match
from ghost_agent.tools.projects import (
    tool_manage_projects, _is_visual_artifact_task,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad


# ===================================================== A: browser render check

def _solid(path, color, size=(120, 90)):
    from PIL import Image
    Image.new("RGB", size, color).save(path)


def test_render_analyzer_flags_uniform_frame(tmp_path):
    p = tmp_path / "sky.png"
    _solid(p, (135, 206, 235))  # all blue sky
    r = analyze_screenshot_render(p)
    assert r and r["verdict"] == "uniform"
    assert r["dominant_pct"] >= 0.8


def test_render_analyzer_passes_varied_frame(tmp_path):
    from PIL import Image
    import random
    random.seed(7)
    img = Image.new("RGB", (120, 90))
    # A populated scene has many distinct colours and no single dominant one.
    for x in range(120):
        for y in range(90):
            img.putpixel((x, y), (random.randint(30, 90),
                                  random.randint(110, 200),
                                  random.randint(20, 80)))
    p = tmp_path / "terrain.png"
    img.save(p)
    r = analyze_screenshot_render(p)
    assert r and r["verdict"] == "has_content"
    assert r["dominant_pct"] < 0.8 and r["distinct_colors"] > 6


def test_render_analyzer_missing_file_returns_none(tmp_path):
    assert analyze_screenshot_render(tmp_path / "nope.png") is None


def test_build_op_payload_threads_interaction_params():
    pl = _build_op_payload(
        op="screenshot", url=None, selector=None, out_path="/workspace/s.png",
        wait_until="load", full_page=True, max_chars=None, timeout_ms=30000,
        tor_proxy=None, click_center=True, settle_ms=1500,
    )
    assert pl["click_center"] is True and pl["settle_ms"] == 1500
    # Omitted → not present (back-compat with old runner payloads).
    pl2 = _build_op_payload(
        op="screenshot", url=None, selector=None, out_path="/workspace/s.png",
        wait_until="load", full_page=True, max_chars=None, timeout_ms=30000,
        tor_proxy=None,
    )
    assert "click_center" not in pl2 and "settle_ms" not in pl2


def test_op_screenshot_supports_interaction_source():
    # op_screenshot lives inside the runner script (it executes in the
    # sandbox subprocess), so assert the interaction code is present there.
    src = _runner_script()
    assert "click_center" in src and "settle_ms" in src and "mouse.click" in src


# --- A-v2: pre-interaction / start-screen detection ---

def test_runner_has_pre_interaction_probe():
    import ast
    src = _runner_script()
    ast.parse(src)  # runner must stay valid Python (runs in the sandbox)
    assert "_probe_pre_interaction" in src
    assert "click to (play|start)" in src  # the KW regex
    # both navigate and screenshot wire the probe into their result
    assert src.count("_probe_pre_interaction(page)") >= 2


def test_pre_interaction_line_fires_on_start_control():
    parsed = {"pre_interaction": {"pre_interaction": True,
                                  "controls": ["Click to Play", "Minecraft Clone"]}}
    line = _pre_interaction_line(parsed)
    assert "PRE_INTERACTION" in line
    assert "Click to Play" in line
    assert "click_center=true" in line


def test_pre_interaction_line_silent_when_running():
    assert _pre_interaction_line({"url": "x"}) == ""
    assert _pre_interaction_line({"pre_interaction": {"pre_interaction": False}}) == ""


def test_visual_verifier_prompt_rejects_start_screens():
    from ghost_agent.core.verifier import _VERIFY_VISUAL_PROMPT
    p = _VERIFY_VISUAL_PROMPT.lower()
    assert "start" in p and "menu" in p
    assert "not started" in p or "not running" in p


# ===================================================== B: replace robustness

ADDFACE_FILE = """\
class Mesher {
  addFace(vertices, colors, normals, x, y, z, dir, color, colorMod) {
    const [dx, dy, dz] = dir;
    const v = [
      [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z]
    ];
    const indices = [];
    if (dy !== 0) { indices.push(0, 1, 2, 0, 2, 3); }
    for (const i of indices) { vertices.push(v[i][0], v[i][1], v[i][2]); }
  }
  other() { return 1; }
}
"""


def test_anchor_block_match_brace_strategy():
    drifted = (
        "  addFace(vertices, colors, normals, x, y, z, dir, color, colorMod) {\n"
        "    TOTALLY DIFFERENT MIDDLE THAT DOES NOT MATCH;\n"
        "  }"
    )
    res = _anchor_block_match(ADDFACE_FILE, drifted)
    assert res is not None
    matched, info = res
    assert "addFace(" in matched and "other()" not in matched
    assert info["strategy"] in ("brace", "first_last")


def test_anchor_block_match_opener_with_leading_context():
    # old_text has a leading comment before the unique signature — the
    # brace anchor should still find the block by its opener line.
    file_src = ("noise\nfunction buildMeshGeometry(chunk) {\n"
                "  const a = 1;\n  const b = 2;\n}\ntail\n")
    old = ("// trying to fix this\nfunction buildMeshGeometry(chunk) {\n"
           "  WRONG BODY here;\n}")
    res = _anchor_block_match(file_src, old)
    assert res is not None and res[1]["strategy"] == "brace"
    assert "buildMeshGeometry" in res[0] and "tail" not in res[0]


def test_anchor_block_match_first_last_strategy():
    text = "func start_unique_anchor()\n  junk line A\n  junk line B\nreturn end_unique_anchor()"
    src = ("x\nfunc start_unique_anchor()\n  REAL A\n  REAL B\nreturn end_unique_anchor()\ny\n")
    res = _anchor_block_match(src, text)
    assert res is not None
    matched, info = res
    assert "REAL A" in matched


def test_anchor_block_match_rejects_short_or_ambiguous():
    # anchors too short → no confident match
    assert _anchor_block_match("a {\n}\nb {\n}\n", "x {\n}") is None


async def test_replace_anchor_rescues_drifted_block(tmp_path):
    f = tmp_path / "index.html"
    f.write_text(ADDFACE_FILE)
    drifted = (
        "  addFace(vertices, colors, normals, x, y, z, dir, color, colorMod) {\n"
        "    const v = [ WRONG MIDDLE not in file at all ];\n"
        "    blah();\n"
        "  }"
    )
    new = ("  addFace(vertices, colors, normals, x, y, z, dir, color, colorMod) {\n"
           "    fixedImplementation();\n"
           "  }")
    res = await tool_file_system(operation="replace", filename="index.html",
                                 old_text=drifted, replace_with=new,
                                 sandbox_dir=tmp_path)
    assert "SUCCESS" in res
    body = f.read_text()
    assert "fixedImplementation()" in body
    assert "other() { return 1; }" in body  # rest untouched


async def test_replace_accepts_param_aliases(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world\n")
    # old_text/replace_with are NOT the canonical content/replace_with names
    res = await tool_file_system(operation="replace", filename="a.txt",
                                 old_text="hello world", replace_with="goodbye",
                                 sandbox_dir=tmp_path)
    assert "SUCCESS" in res
    assert f.read_text().strip() == "goodbye"


# ===================================================== C/D: project tool

@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None, current_project_id=None,
    )


def test_is_visual_artifact_task():
    assert _is_visual_artifact_task("Core Three.js setup and render loop")
    assert _is_visual_artifact_task("Build the WebGL canvas game")
    assert _is_visual_artifact_task("UI/HUD with crosshair")
    assert not _is_visual_artifact_task("research local regulations")
    assert not _is_visual_artifact_task("write the database migration")


async def _new_task(context, store, desc):
    await tool_manage_projects(context, action="create", title="P", kind="CODING")
    pid = context.current_project_id
    out = json.loads(await tool_manage_projects(
        context, action="task_add", description=desc))
    return pid, out["task_id"]


async def test_done_gate_blocks_visual_task_without_evidence(context, store):
    pid, tid = await _new_task(context, store, "Render the Three.js game scene")
    out = json.loads(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE"))
    assert out.get("gated_unverified") == [tid]
    assert "RENDER_CHECK" in out["agent_instruction"]
    # NOT marked done
    assert store.get_task(tid)["status"] != "DONE"


async def test_done_gate_allows_visual_task_with_result(context, store):
    pid, tid = await _new_task(context, store, "Render the game scene")
    out = json.loads(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE",
        result="browser screenshot RENDER_CHECK=HAS_CONTENT, terrain visible"))
    assert "gated_unverified" not in out
    assert store.get_task(tid)["status"] == "DONE"


async def test_done_gate_ignores_nonvisual_task(context, store):
    pid, tid = await _new_task(context, store, "Write the config parser")
    out = json.loads(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE"))
    assert "gated_unverified" not in out
    assert store.get_task(tid)["status"] == "DONE"


async def test_create_returns_workspace_note(context, store):
    out = json.loads(await tool_manage_projects(
        context, action="create", title="Fresh Proj", kind="CODING"))
    pid = out["created"]
    assert out["workspace"] == f"projects/{pid}"
    assert "note" in out and f"projects/{pid}/" in out["note"]
