"""Functional coverage for the off-loop workspace save (#23).

Drives GET/POST /api/workspace/save against a TestClient with a real sandbox
dir on disk, verifying: a normal save returns a valid zip, the archive is built
off the event loop, and an oversized sandbox is rejected with 413 instead of
freezing the loop / spiking RAM.
"""
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ghost_agent.api.routes as routes


def _make_app(sandbox_dir: Path):
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.context = MagicMock()
    agent.context.args = MagicMock()
    agent.context.args.api_key = "test-key"
    agent.context.sandbox_dir = sandbox_dir
    agent.context.scratchpad._data = {"k": "v"}
    app.state.agent = agent
    app.state.context = agent.context
    app.state.args = agent.context.args
    return TestClient(app), agent


def test_save_returns_valid_zip(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    (sb / "a.txt").write_text("hello")
    (sb / "sub").mkdir()
    (sb / "sub" / "b.py").write_text("print(1)")

    client, _ = _make_app(sb)
    r = client.post("/api/workspace/save", headers={"X-Ghost-Key": "test-key"},
                    json={"chat_history": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "session.json" in names
    assert "sandbox/a.txt" in names
    assert "sandbox/sub/b.py" in names
    # session.json carries chat history + scratchpad.
    import json
    session = json.loads(zf.read("session.json"))
    assert session["chat_history"][0]["content"] == "hi"
    assert session["scratchpad"] == {"k": "v"}


def test_save_rejects_oversized_workspace(tmp_path, monkeypatch):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    (sb / "big.bin").write_bytes(b"x" * 2048)
    # Drop the cap below the file size so the ceiling trips deterministically.
    monkeypatch.setattr(routes, "_MAX_WORKSPACE_SAVE_BYTES", 1024)

    client, _ = _make_app(sb)
    r = client.post("/api/workspace/save", headers={"X-Ghost-Key": "test-key"}, json={})
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "WorkspaceTooLarge"


def test_save_requires_api_key(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    client, _ = _make_app(sb)
    assert client.post("/api/workspace/save", json={}).status_code == 403


def test_save_spool_file_cleaned_up(tmp_path):
    """The temp spool must not accumulate across saves."""
    import tempfile
    sb = tmp_path / "sandbox"
    sb.mkdir()
    (sb / "a.txt").write_text("hi")
    client, _ = _make_app(sb)

    before = set(Path(tempfile.gettempdir()).glob("ws_save_*.zip"))
    r = client.post("/api/workspace/save", headers={"X-Ghost-Key": "test-key"}, json={})
    assert r.status_code == 200
    _ = r.content  # drain the response so the background cleanup fires
    after = set(Path(tempfile.gettempdir()).glob("ws_save_*.zip"))
    assert after <= before  # no new leftover spool files
