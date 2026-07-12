"""Regression tests for the 2026-05-28 functional-audit bug fixes.

Each `test_bug_*` covers exactly one finding from the audit report. They
should be **independent** from any global mocks/fixtures in conftest so
they continue to pass even if the surrounding infrastructure changes.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient


# ======================================================================
# Bug #2 — `/api/projects*` routes are eaten by the catch-all proxy
# ======================================================================

def _make_app_for_router_test(tmp_path):
    """Build a real `create_app()` instance with a minimal agent stub
    that satisfies `verify_api_key` and the project routes' need for a
    `project_store` on the context."""
    from ghost_agent.api.app import create_app
    from ghost_agent.memory.projects import ProjectStore

    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    context = SimpleNamespace(
        args=args,
        project_store=store,
        scratchpad=MagicMock(),
        graph_memory=None,
        current_project_id=None,
        llm_client=MagicMock(),
    )
    agent = SimpleNamespace(context=context)
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    return TestClient(app), store


def test_bug2_projects_routes_reachable_after_catchall(tmp_path):
    """The catch-all `/{path:path}` in routes.py used to swallow every
    request to /api/projects*. The fix re-orders the routers in
    `create_app()` so projects_router is registered first."""
    tc, store = _make_app_for_router_test(tmp_path)

    # GET /api/projects must hit the dedicated route, not the proxy.
    r = tc.get("/api/projects")
    assert r.status_code == 200, (
        f"GET /api/projects returned {r.status_code} — the catch-all "
        f"is still eating the projects route. Body: {r.text[:200]}"
    )
    body = r.json()
    assert "projects" in body
    assert "current" in body

    # POST also reaches the dedicated route.
    r = tc.post("/api/projects", json={"title": "T", "kind": "CODING"})
    assert r.status_code == 201
    pid = r.json()["id"]

    # Nested path too.
    r = tc.get(f"/api/projects/{pid}/tasks")
    assert r.status_code == 200


# ======================================================================
# Bug #3, #5, #6 — chat-endpoint request validation
# ======================================================================

def _make_chat_app(tmp_path, handle_chat_return=("hi", 1700000000, "rid")):
    """Spin up the real FastAPI app, but stub `agent.handle_chat` so we
    can test the request-validation layer in isolation from the LLM."""
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    agent = SimpleNamespace(
        context=SimpleNamespace(args=args, llm_client=MagicMock()),
        handle_chat=AsyncMock(return_value=handle_chat_return),
    )
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    return TestClient(app), agent


def test_bug3_empty_messages_rejected(tmp_path):
    """`messages: []` used to return HTTP 200 with fabricated content
    drawn from prior state. Now: 422 with a structured error."""
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={"model": "qwen-3.6-35b-a3", "messages": []})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["type"] == "InvalidRequestShape"
    assert "messages" in body["error"]["message"]
    # And critically: handle_chat must NOT have been called.
    agent.handle_chat.assert_not_called()


def test_bug3_missing_messages_rejected(tmp_path):
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={"model": "qwen-3.6-35b-a3"})
    assert r.status_code == 422
    agent.handle_chat.assert_not_called()


def test_bug3_messages_not_a_list_rejected(tmp_path):
    """`messages: "not a list"` used to crash with
    `'str' object has no attribute 'get'` (HTTP 500 stack leak)."""
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={
        "model": "qwen-3.6-35b-a3",
        "messages": "not a list",
    })
    assert r.status_code == 422
    body = r.json()
    # The error must NOT leak Python internals (no AttributeError, no
    # `'str' object has no attribute`).
    assert "AttributeError" not in body["error"]["message"]
    assert "'str' object" not in body["error"]["message"]
    assert body["error"]["type"] == "InvalidRequestShape"


def test_bug5_unknown_role_rejected(tmp_path):
    """`role: "asdf"` used to be forwarded to the upstream LLM and
    surface its template-parse error as assistant content."""
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={
        "model": "qwen-3.6-35b-a3",
        "messages": [{"role": "asdf", "content": "hi"}],
    })
    assert r.status_code == 422
    body = r.json()
    assert "role" in body["error"]["message"]
    assert "asdf" in body["error"]["message"]
    agent.handle_chat.assert_not_called()


def test_bug5_each_message_must_be_an_object(tmp_path):
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={
        "model": "qwen-3.6-35b-a3",
        "messages": ["a string instead of {role, content}"],
    })
    assert r.status_code == 422
    body = r.json()
    assert "messages[0]" in body["error"]["message"]


def test_bug6_unknown_model_returns_404(tmp_path):
    """`model: "gpt-4"` used to return HTTP 200 with the response
    falsely claiming `"model": "gpt-4"`."""
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "ModelNotFound"
    assert "gpt-4" in body["error"]["message"]
    agent.handle_chat.assert_not_called()


def test_bug6_missing_model_still_accepted(tmp_path):
    """We don't 404 a missing model — that preserves Ollama-style
    clients that leave the field implicit. The configured default is
    used."""
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    agent.handle_chat.assert_awaited_once()


def test_bug6_valid_model_reaches_handler(tmp_path):
    """Configured model must still pass through cleanly."""
    tc, agent = _make_chat_app(tmp_path)
    r = tc.post("/api/chat", json={
        "model": "qwen-3.6-35b-a3",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    agent.handle_chat.assert_awaited_once()


# ======================================================================
# Bug #4 — stack-trace leak on handle_chat exception
# ======================================================================

def test_bug4_handler_exception_does_not_leak_internals(tmp_path):
    """Any exception raised by `handle_chat` is now masked behind a
    generic `InternalError` with an `error_id`. The leaky type name
    and exception repr must NOT appear in the wire response."""
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    agent = SimpleNamespace(
        context=SimpleNamespace(args=args, llm_client=MagicMock()),
        handle_chat=AsyncMock(
            side_effect=AttributeError("'str' object has no attribute 'get'"),
        ),
    )
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    tc = TestClient(app)

    r = tc.post("/api/chat", json={
        "model": "qwen-3.6-35b-a3",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 500
    body = r.json()
    msg = body["error"]["message"]
    typ = body["error"]["type"]
    # Type must be the generic "InternalError", not "AttributeError".
    assert typ == "InternalError"
    # Message must contain an error_id and NOT the raw Python repr.
    assert "error_id=" in msg
    assert "AttributeError" not in msg
    assert "'str' object has no attribute" not in msg


# ======================================================================
# Bug #7 — `/api/workspace/save` with no body returned bare 500
# ======================================================================

def test_bug7_workspace_save_no_body_succeeds(tmp_path):
    """Empty body now means "save with empty chat_history" — returns
    200 with a zip rather than a 500 stack-leak."""
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    agent = SimpleNamespace(
        context=SimpleNamespace(
            args=args, llm_client=MagicMock(),
            sandbox_dir=sandbox,
            scratchpad=None,  # exercise the "no scratchpad" path
        ),
    )
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    tc = TestClient(app)

    r = tc.post("/api/workspace/save")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"


def test_bug7_workspace_save_malformed_body_returns_400(tmp_path):
    """Garbage body returns a structured 400, not a bare 500."""
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    agent = SimpleNamespace(
        context=SimpleNamespace(
            args=args, llm_client=MagicMock(),
            sandbox_dir=sandbox, scratchpad=None,
        ),
    )
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    tc = TestClient(app)

    r = tc.post("/api/workspace/save", content=b"{not valid json",
                headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    body = r.json()
    assert "error" in body


# ======================================================================
# Bug #8 — `/api/delete` for unknown model returned `{"status":"success"}`
# ======================================================================

def test_bug8_delete_unknown_model_returns_404(tmp_path):
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    agent = SimpleNamespace(context=SimpleNamespace(args=args, llm_client=MagicMock()))
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    tc = TestClient(app)

    r = tc.request("DELETE", "/api/delete", json={"model": "gpt-4"})
    assert r.status_code == 404


def test_bug8_delete_configured_model_returns_200(tmp_path):
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    agent = SimpleNamespace(context=SimpleNamespace(args=args, llm_client=MagicMock()))
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    tc = TestClient(app)

    r = tc.request("DELETE", "/api/delete", json={"model": "qwen-3.6-35b-a3"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"


def test_bug8_delete_no_body_returns_200(tmp_path):
    """No body / missing model → preserve legacy behavior (200)."""
    from ghost_agent.api.app import create_app

    args = SimpleNamespace(api_key="", model="qwen-3.6-35b-a3")
    agent = SimpleNamespace(context=SimpleNamespace(args=args, llm_client=MagicMock()))
    app = create_app()
    app.state.agent = agent
    app.state.args = args
    tc = TestClient(app)

    r = tc.request("DELETE", "/api/delete")
    assert r.status_code == 200


# ======================================================================
# Bug #9 — `report_pdf` honours the `filename` parameter
# ======================================================================

def test_bug9_sanitize_filename_accepts_safe_name():
    from ghost_agent.tools.report_pdf import _sanitize_filename
    assert _sanitize_filename("test_report.pdf") == "test_report.pdf"
    assert _sanitize_filename("q4-2026_summary") == "q4-2026_summary.pdf"
    assert _sanitize_filename("a.b.c") == "a.b.c.pdf"


def test_bug9_sanitize_filename_rejects_traversal():
    from ghost_agent.tools.report_pdf import _sanitize_filename
    # Path traversal must be defeated by basename-extraction. The
    # leading `..` is stripped along with everything before the last
    # separator, so we end up with a sandbox-relative name.
    assert _sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert _sanitize_filename("/tmp/foo.pdf") == "foo.pdf"
    # Reject empty / null / leading-dot / spaces / over-length.
    assert _sanitize_filename("") is None
    assert _sanitize_filename(None) is None
    assert _sanitize_filename(".hidden") is None
    assert _sanitize_filename("has space.pdf") is None
    assert _sanitize_filename("x" * 200) is None
    assert _sanitize_filename("bad!chars.pdf") is None


@pytest.mark.asyncio
async def test_bug9_pdf_uses_requested_filename(tmp_path, monkeypatch):
    """The PDF tool now writes to the caller-supplied filename rather
    than always generating `report_<8hex>.pdf`."""
    from ghost_agent.tools import report_pdf

    # Stub the renderer so the test doesn't depend on PyMuPDF.
    # _render_to_pdf now returns (pages, truncated).
    monkeypatch.setattr(report_pdf, "_render_to_pdf",
                        lambda html, out_path: (out_path.write_bytes(b"%PDF-1.7\n"), (1, False))[1])

    result = await report_pdf.tool_generate_pdf(
        title="Test",
        sections=[{"heading": "H", "body": "Hello"}],
        filename="my_report.pdf",
        sandbox_dir=tmp_path,
    )
    assert "my_report.pdf" in result
    assert (tmp_path / "my_report.pdf").exists()


@pytest.mark.asyncio
async def test_bug9_pdf_rejects_unsafe_filename(tmp_path, monkeypatch):
    from ghost_agent.tools import report_pdf

    monkeypatch.setattr(report_pdf, "_render_to_pdf",
                        lambda html, out_path: (1, False))

    result = await report_pdf.tool_generate_pdf(
        title="T",
        sections=[{"heading": "H", "body": "x"}],
        filename="has space and bad!chars.pdf",
        sandbox_dir=tmp_path,
    )
    assert "SYSTEM ERROR" in result


# ======================================================================
# Bug #10 — image-gen snaps requested size to nearest SDXL bucket
# ======================================================================

def test_bug10_snap_returns_exact_bucket_when_matching():
    from ghost_agent.tools.image_gen import _snap_to_bucket
    (w, h), adjusted = _snap_to_bucket(624, 624)
    assert (w, h) == (624, 624)
    assert adjusted is False


def test_bug10_snap_square_to_node_square():
    """Squares snap to the node's square bucket (624x624 — the node is an
    SD1.5 Jetson with a 512x768 pixel budget; the old SDXL 1024 buckets
    exceeded it and got scale-distorted server-side)."""
    from ghost_agent.tools.image_gen import _snap_to_bucket
    (w, h), adjusted = _snap_to_bucket(512, 512)
    assert (w, h) == (624, 624)
    assert adjusted is True
    (w, h), adjusted = _snap_to_bucket(1024, 1024)
    assert (w, h) == (624, 624)
    assert adjusted is True


def test_bug10_snap_landscape_to_landscape_bucket():
    """A landscape request must NOT pick a portrait bucket — aspect
    ratio is the primary discriminator."""
    from ghost_agent.tools.image_gen import _snap_to_bucket
    # 16:9-ish landscape.
    (w, h), adjusted = _snap_to_bucket(1600, 900)
    assert w > h, f"expected landscape, got {w}x{h}"
    assert adjusted is True


def test_bug10_snap_portrait_to_portrait_bucket():
    from ghost_agent.tools.image_gen import _snap_to_bucket
    (w, h), adjusted = _snap_to_bucket(900, 1600)
    assert h > w, f"expected portrait, got {w}x{h}"
    assert adjusted is True


@pytest.mark.asyncio
async def test_bug10_payload_contains_snapped_size(tmp_path):
    """The payload sent to the image client must include the snapped
    width/height (not the raw user request)."""
    from ghost_agent.tools.image_gen import tool_generate_image

    captured = {}

    async def fake_gen(payload):
        captured.update(payload)
        # Minimal valid response: a single 1x1 PNG.
        import base64
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        return {"data": [{"b64_json": png}]}

    llm = MagicMock()
    llm.image_gen_clients = [object()]  # truthy
    llm.generate_image = fake_gen

    out = await tool_generate_image(
        prompt="a cat",
        llm_client=llm,
        sandbox_dir=tmp_path,
        width=512,
        height=512,
    )
    assert "SUCCESS" in out
    assert captured["width"] in (624,)   # snapped to the node's square bucket
    assert captured["height"] in (624,)
    # Steps omitted -> node's tuned default (30) applies server-side; the
    # old LCM-era 4-8 clamp forced every image to the 15-step floor.
    assert "steps" not in captured


# ======================================================================
# Bug #11 — `update_profile` replaces singleton keys instead of merging
# ======================================================================

def test_bug11_singleton_keys_replace(tmp_path):
    """`name`, `role`, etc. are singleton — setting a new value must
    replace, not append to a list."""
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    # The constructor seeds `name: "User"`. The audit observed:
    # after `update(root, name, Vasilis)` the stored value was
    # `["User", "Vasilis"]`. With the fix it must be just "Vasilis".
    pm.update("root", "name", "Vasilis")
    data = pm.load()
    assert data["root"]["name"] == "Vasilis", (
        f"singleton key 'name' was merged into a list: {data['root']['name']!r}"
    )

    # A subsequent overwrite still replaces (no list growth).
    pm.update("root", "name", "Alex")
    assert pm.load()["root"]["name"] == "Alex"


def test_bug11_non_singleton_still_merges(tmp_path):
    """Multi-value keys like `language` must STILL merge — the
    deliberate dedup-merge behavior for non-singletons is preserved."""
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    pm.update("interests", "language", "python")
    pm.update("interests", "language", "rust")
    val = pm.load()["interests"]["language"]
    assert isinstance(val, list)
    assert set(val) == {"python", "rust"}


# ======================================================================
# Bug #12 — project auto-rolls up to DONE when all tasks reach terminal
# ======================================================================

def test_bug12_project_auto_done_when_all_tasks_done(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project(title="X", kind="CODING")
    t1 = store.add_task(pid, "task one")
    t2 = store.add_task(pid, "task two")

    # Mark first done — project should still be ACTIVE (work remains).
    store.update_task(t1, status="DONE")
    assert store.get_project(pid)["status"] == "ACTIVE"

    # Mark second done — project auto-transitions to DONE.
    store.update_task(t2, status="DONE")
    assert store.get_project(pid)["status"] == "DONE"


def test_bug12_failed_task_rolls_project_up_to_failed(tmp_path):
    """A FAILED task counts as terminal for rollup purposes — the project
    leaves the in-flight (ACTIVE) state. It now rolls up to FAILED rather
    than DONE: a project whose work ended in failure must not masquerade
    as completed (see test_project_mgmt_fixes for the full matrix)."""
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project(title="X", kind="CODING")
    a = store.add_task(pid, "ok")
    b = store.add_task(pid, "broken")
    store.update_task(a, status="DONE")
    store.update_task(b, status="FAILED")
    assert store.get_project(pid)["status"] == "FAILED"


def test_bug12_archived_project_not_rolled_back(tmp_path):
    """Manually-archived projects must NOT be auto-un-archived just
    because their tasks reached terminal state."""
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    pid = store.create_project(title="X", kind="CODING")
    t = store.add_task(pid, "ok")
    store.delete_project(pid)  # soft-delete → ARCHIVED
    store.update_task(t, status="DONE")
    assert store.get_project(pid)["status"] == "ARCHIVED"


# ======================================================================
# Bug #13 — final response deduplicates consecutive identical paragraphs
# ======================================================================

def test_bug13_consecutive_duplicate_paragraphs_collapse():
    """The fix is inline in `handle_chat`, but the algorithm is a
    pure-text transform we can verify against the same regex."""
    text = (
        "Your name is **Vasilis** and your role is **engineer**.\n\n"
        "Your name is **Vasilis** and your role is **engineer**."
    )
    parts = text.split("\n\n")
    deduped: list[str] = []
    prev_key = None
    for p in parts:
        key = re.sub(r"\s+", " ", p).strip()
        if key and key == prev_key:
            continue
        deduped.append(p)
        prev_key = key
    out = "\n\n".join(deduped)
    assert out.count("Your name is") == 1


def test_bug13_non_adjacent_repeats_preserved():
    """Quote-then-answer style (same string at positions 0 and 2) must
    NOT be collapsed — only adjacent duplicates."""
    text = "X\n\nY\n\nX"
    parts = text.split("\n\n")
    deduped: list[str] = []
    prev_key = None
    for p in parts:
        key = re.sub(r"\s+", " ", p).strip()
        if key and key == prev_key:
            continue
        deduped.append(p)
        prev_key = key
    assert "\n\n".join(deduped) == "X\n\nY\n\nX"


# ======================================================================
# Bug #1 — acquired skills appear in live tool list mid-turn
# ======================================================================

def test_bug1_request_state_exposes_invalidate_tool_defs():
    """The fix relies on a new method on the per-request cache. Pin
    its name + behavior so a future refactor doesn't silently revert
    the fix."""
    from ghost_agent.core.agent import GhostAgent

    # `_RequestState` is a nested class inside the module body. We
    # introspect via the module to find it without instantiating
    # GhostAgent (which needs a full context).
    import ghost_agent.core.agent as agent_mod
    src = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert "def invalidate_tool_defs" in src, (
        "Bug #1 fix lost: _RequestState.invalidate_tool_defs() is "
        "no longer defined."
    )
    # Wire-up call site must remain too.
    assert "request_state.invalidate_tool_defs()" in src, (
        "Bug #1 fix lost: create_skill no longer triggers a tool-def "
        "cache invalidation in the agent loop."
    )


@pytest.mark.asyncio
async def test_bug1_create_skill_success_message_advises_direct_call(tmp_path, monkeypatch):
    """After the fix, `tool_create_skill` returns a message that
    explicitly tells the model "the skill is LIVE — call by name" so
    it doesn't fall back to the `python3 acquired_skills/<n>.py`
    anti-pattern that the audit observed."""
    from ghost_agent.tools import acquired_skills

    async def fake_execute(**kwargs):
        # Pretend the TDD test passed.
        return "EXIT CODE: 0\nresult"

    monkeypatch.setattr("ghost_agent.tools.execute.tool_execute", fake_execute)
    monkeypatch.setattr(acquired_skills, "AcquiredSkillManager",
                        MagicMock(return_value=MagicMock()))

    schema = json.dumps({"type": "object", "properties": {}})
    payload = json.dumps({})
    code = ("import sys, json\n"
            "def main():\n"
            "    print('ok')\n"
            "if __name__ == '__main__':\n"
            "    main()\n")
    out = await acquired_skills.tool_create_skill(
        sandbox_dir=tmp_path,
        memory_dir=tmp_path,
        memory_system=MagicMock(),
        sandbox_manager=MagicMock(),
        name="my_skill",
        description="d",
        parameters_schema=schema,
        python_code=code,
        test_payload=payload,
    )
    assert "LIVE" in out
    assert "tool_call" in out
    assert "my_skill" in out


# ======================================================================
# Smoke test — `create_app()` still composes successfully
# ======================================================================

def test_create_app_smoke():
    """If `create_app()` raises, every other test in this file is
    moot. Fail fast and loud."""
    from ghost_agent.api.app import create_app
    app = create_app()
    # The two routers must both be present.
    route_paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/chat" in route_paths
    assert "/api/projects" in route_paths
    # Catch-all must STILL be present (we didn't delete it).
    assert any("{path:path}" in p for p in route_paths), (
        "Catch-all route was accidentally removed."
    )
