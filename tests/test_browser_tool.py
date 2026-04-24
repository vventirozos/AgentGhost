"""Tests for the native browser tool and related Playwright hardening.

Covers:
  1. Sandbox fail-loud: if `playwright install chromium --with-deps`
     returns non-zero, `.supercharged` is NOT touched and the install
     raises.
  2. tool_browser arg validation — unknown op / missing fields return
     a structured error, not an exception.
  3. tool_browser runs the runner with a DNS-leak-safe proxy payload
     (Chromium `socks5://` + `--host-resolver-rules` in the runner's
     `_chromium_args`).
  4. tool_browser parse logic — [BROWSER_OK] / [BROWSER_ERR] sentinel
     handling, robustness to stray Chromium stderr.
  5. The runner's `_chromium_args` always includes `--no-sandbox` and
     `--disable-dev-shm-usage`, and adds `--host-resolver-rules` when
     a proxy is configured.
  6. Each tool_browser op type formats the runner's OK payload into
     the expected LLM-facing string.
  7. The prompt section was updated to mention the native tool first
     and to include the host-resolver-rules DNS-leak guard.
  8. The web_automation template produces a well-formed triple at
     every tier, and advanced+ triggers the JS-decoy twist.
  9. The `browser` tool is registered in TOOL_DEFINITIONS.
"""

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- 1. Sandbox install fail-loud -------------------------------------------

@pytest.mark.asyncio
async def test_docker_sandbox_raises_when_playwright_install_fails():
    """If Chromium install exits non-zero, the sandbox must raise and
    NOT touch /root/.supercharged — otherwise a future boot will
    silently believe Playwright is ready."""
    from ghost_agent.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox(host_workspace=Path("/tmp/ws"))
    sandbox.container = MagicMock()
    sandbox.container.status = "running"

    calls = []

    def exec_side_effect(cmd, **kwargs):
        calls.append(cmd)
        if "test -f /root/.supercharged" in cmd:
            return (1, b"")  # not installed — run install
        if "apt-get" in cmd or "sudoers" in cmd or "pysocks" in cmd.lower():
            return (0, b"")
        if cmd.startswith("pip install --no-cache-dir"):
            return (0, b"")
        if "from playwright" in cmd:
            return (1, b"ModuleNotFoundError")  # not cached → force install
        if "playwright install chromium" in cmd:
            return (13, b"Download failed: network timeout")  # fail loud
        if cmd.startswith("touch "):
            return (0, b"")
        return (0, b"")

    sandbox.container.exec_run.side_effect = exec_side_effect

    with patch("sys.modules", {**sys.modules, "docker": MagicMock(), "docker.errors": MagicMock()}):
        with patch.object(sandbox, "_is_container_ready", return_value=True):
            with pytest.raises(Exception) as exc_info:
                sandbox.ensure_running()

    # Must mention the failure
    assert "playwright" in str(exc_info.value).lower() or "chromium" in str(exc_info.value).lower()
    assert "13" in str(exc_info.value), "Exit code should be surfaced"

    # Crucially: /root/.supercharged must NEVER be touched on a failed install
    touches = [c for c in calls if c.startswith("touch /root/.supercharged")]
    assert not touches, f".supercharged marker must not be touched on failure, but was: {touches}"


@pytest.mark.asyncio
async def test_docker_sandbox_succeeds_when_playwright_install_ok():
    """Happy path: when both pip install AND `playwright install`
    return 0, /root/.supercharged IS touched exactly once."""
    from ghost_agent.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox(host_workspace=Path("/tmp/ws"))
    sandbox.container = MagicMock()
    sandbox.container.status = "running"

    calls = []

    def exec_side_effect(cmd, **kwargs):
        calls.append(cmd)
        if "test -f /root/.supercharged" in cmd:
            return (1, b"")
        if "find /root/.cache/ms-playwright" in cmd:
            # Post-install Chromium verification passes.
            return (0, b"/root/.cache/ms-playwright/chromium-9999/chrome-linux/headless_shell\n")
        return (0, b"")

    sandbox.container.exec_run.side_effect = exec_side_effect

    with patch("sys.modules", {**sys.modules, "docker": MagicMock(), "docker.errors": MagicMock()}):
        with patch.object(sandbox, "_is_container_ready", return_value=True):
            sandbox.ensure_running()

    touches = [c for c in calls if c.startswith("touch /root/.supercharged")]
    assert len(touches) == 1, f"Expected exactly one .supercharged touch, got: {touches}"


@pytest.mark.asyncio
async def test_docker_sandbox_always_runs_playwright_install_on_first_boot():
    """The previous `from playwright.sync_api import sync_playwright`
    probe was faulty: the pip-install step above it makes the library
    importable, so the probe always passed and the Chromium BINARY
    install was silently skipped on the very first boot. The eval
    run on 2026-04-23 hit 'headless_shell not found' for exactly
    this reason. After the fix, `playwright install chromium
    --with-deps` must ALWAYS run inside the first-boot bootstrap
    block — regardless of whether the library probe would have
    succeeded."""
    from ghost_agent.sandbox.docker import DockerSandbox

    sandbox = DockerSandbox(host_workspace=Path("/tmp/ws"))
    sandbox.container = MagicMock()
    sandbox.container.status = "running"

    calls = []

    def exec_side_effect(cmd, **kwargs):
        calls.append(cmd)
        if "test -f /root/.supercharged" in cmd:
            return (1, b"")  # first boot
        # Simulate the previous-bug scenario: library probe WOULD
        # return 0 (importable). A correct implementation must ignore
        # this and still run the binary install.
        if "from playwright" in cmd:
            return (0, b"OK")
        if "find /root/.cache/ms-playwright" in cmd:
            return (0, b"/root/.cache/ms-playwright/chromium-9999/chrome-linux/headless_shell\n")
        return (0, b"")

    sandbox.container.exec_run.side_effect = exec_side_effect

    with patch("sys.modules", {**sys.modules, "docker": MagicMock(), "docker.errors": MagicMock()}):
        with patch.object(sandbox, "_is_container_ready", return_value=True):
            sandbox.ensure_running()

    chromium_installs = [c for c in calls if "playwright install chromium --with-deps" in c]
    assert len(chromium_installs) == 1, (
        f"Chromium install must always run on first boot (no probe gate), "
        f"but ran {len(chromium_installs)} times. All calls:\n"
        + "\n".join(f"  {c}" for c in calls)
    )
    # And /root/.supercharged must still get touched once
    touches = [c for c in calls if c.startswith("touch /root/.supercharged")]
    assert len(touches) == 1


# --- 2/3/4/6. tool_browser semantics ----------------------------------------

def _make_sandbox_stub(output: str, exit_code: int = 0):
    """Fake sandbox_manager that captures the last command it was
    given and returns canned output on `.execute()`."""
    stub = MagicMock()
    stub.last_command = None
    stub.tor_proxy = None

    def _execute(cmd, timeout=300, **kwargs):
        stub.last_command = cmd
        return output, exit_code

    stub.execute = _execute
    return stub


@pytest.mark.asyncio
async def test_tool_browser_rejects_missing_operation(tmp_path):
    from ghost_agent.tools.browser import tool_browser

    stub = _make_sandbox_stub("")
    result = await tool_browser(sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "ERROR" in result
    assert "operation" in result.lower()


@pytest.mark.asyncio
async def test_tool_browser_rejects_unknown_operation(tmp_path):
    from ghost_agent.tools.browser import tool_browser

    stub = _make_sandbox_stub("")
    result = await tool_browser(operation="drive_to_store", sandbox_dir=tmp_path, sandbox_manager=stub)
    assert "ERROR" in result
    assert "Unknown operation" in result


@pytest.mark.asyncio
async def test_tool_browser_requires_sandbox(tmp_path):
    from ghost_agent.tools.browser import tool_browser

    result = await tool_browser(operation="navigate", url="http://x", sandbox_dir=tmp_path, sandbox_manager=None)
    assert "ERROR" in result
    assert "Sandbox" in result


@pytest.mark.asyncio
async def test_tool_browser_navigate_happy_path(tmp_path):
    """Navigate op: runner returns an OK sentinel; tool formats the
    LLM-facing summary and writes the runner script to the sandbox."""
    from ghost_agent.tools.browser import tool_browser, _BROWSER_RUNNER_FILENAME

    payload = {"status": 200, "url": "https://example.com/", "title": "Example"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(
        operation="navigate", url="https://example.com/",
        sandbox_dir=tmp_path, sandbox_manager=stub, tor_proxy=None,
    )

    # The runner script must have been written
    assert (tmp_path / _BROWSER_RUNNER_FILENAME).exists()

    # Command must invoke the runner with a JSON payload
    assert stub.last_command is not None
    assert _BROWSER_RUNNER_FILENAME in stub.last_command

    # Output formatting
    assert "STATUS: OK" in result
    assert "example.com" in result
    assert "HTTP_STATUS: 200" in result
    assert "TITLE: Example" in result


@pytest.mark.asyncio
async def test_tool_browser_navigate_propagates_socks5h_rewrite(tmp_path):
    """Chromium does not accept `socks5h://`. If the caller supplies
    that scheme (some helpers in the repo do), the tool must rewrite
    to `socks5://` before launching — DNS-over-proxy is then enforced
    at the runner level via --host-resolver-rules, not URL scheme."""
    from ghost_agent.tools.browser import tool_browser

    payload = {"status": 200, "url": "x", "title": "t"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    await tool_browser(
        operation="navigate", url="http://example.com/",
        sandbox_dir=tmp_path, sandbox_manager=stub,
        tor_proxy="socks5h://127.0.0.1:9050",
    )
    # The JSON arg encodes the payload — look for the rewritten scheme
    assert stub.last_command is not None
    assert "socks5://127.0.0.1:9050" in stub.last_command
    # And NOT the original socks5h:// scheme (would confuse Chromium)
    assert "socks5h://" not in stub.last_command


@pytest.mark.asyncio
async def test_tool_browser_extract_text_reports_truncation(tmp_path):
    from ghost_agent.tools.browser import tool_browser

    payload = {
        "url": "http://x", "title": "T",
        "text": "hello", "truncated": True, "length": 5,
    }
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(
        operation="extract_text", url="http://x", max_chars=5,
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "STATUS: OK" in result
    assert "truncated" in result.lower()
    assert "hello" in result


@pytest.mark.asyncio
async def test_tool_browser_screenshot_formats_download_link(tmp_path):
    from ghost_agent.tools.browser import tool_browser

    payload = {"path": "/workspace/page.png", "url": "http://x"}
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    result = await tool_browser(
        operation="screenshot", url="http://x", out_path="page.png",
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "STATUS: OK" in result
    assert "SAVED: page.png" in result
    # Download path is what the agent hands to the user
    assert "/api/download/page.png" in result


@pytest.mark.asyncio
async def test_tool_browser_click_requires_selector(tmp_path):
    """Runner's own validation fails with a sentinel error — the tool
    should surface it rather than hide under a generic error."""
    from ghost_agent.tools.browser import tool_browser

    # We simulate the runner's own error path
    stub = _make_sandbox_stub("[BROWSER_ERR] click requires 'selector'\n", exit_code=1)
    result = await tool_browser(
        operation="click", sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "ERROR" in result
    assert "selector" in result


@pytest.mark.asyncio
async def test_tool_browser_no_sentinel_means_runner_crash(tmp_path):
    """If Chromium crashed hard and printed neither OK nor ERR sentinel,
    the tool must not silently succeed — it should surface the raw tail
    so the agent has something to debug with."""
    from ghost_agent.tools.browser import tool_browser

    stub = _make_sandbox_stub("segfault somewhere\ncore dumped\n", exit_code=139)
    result = await tool_browser(
        operation="navigate", url="http://x",
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "ERROR" in result
    assert "139" in result
    assert "segfault" in result


@pytest.mark.asyncio
async def test_tool_browser_tolerates_stderr_noise_before_sentinel(tmp_path):
    """Chromium prints ALSA / libpci warnings to stdout before the
    runner's sentinel line. The parser must still pick up the OK."""
    from ghost_agent.tools.browser import tool_browser

    payload = {"status": 200, "url": "http://x", "title": "T"}
    noise = (
        "Gtk-Message: Failed to load module ...\n"
        "ALSA lib confmisc.c:767:(parse_card)...\n"
        f"[BROWSER_OK] {json.dumps(payload)}\n"
    )
    stub = _make_sandbox_stub(noise)
    result = await tool_browser(
        operation="navigate", url="http://x",
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "STATUS: OK" in result


# --- 5. Chromium args include DNS-leak-safe flags ---------------------------

def _load_runner_module(monkeypatch):
    """Extract and import the runner script as a standalone module so
    we can call `_chromium_args` directly without needing a container.
    """
    from ghost_agent.tools.browser import _runner_script

    src = _runner_script()
    # Stub out playwright (not installed in the test env's Python) so
    # the module loads; we only want `_chromium_args` anyway.
    fake_pw_module = types.ModuleType("playwright.async_api")
    fake_pw_module.async_playwright = MagicMock()
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_pw_module)

    mod = types.ModuleType("runner_under_test")
    exec(compile(src, "runner", "exec"), mod.__dict__)
    return mod


def test_runner_chromium_args_baseline(monkeypatch):
    mod = _load_runner_module(monkeypatch)
    args = mod._chromium_args(None)
    # Docker-safe flags must always be present
    assert "--no-sandbox" in args
    assert "--disable-dev-shm-usage" in args
    # Without a proxy, no DNS-leak guard is needed (direct egress)
    joined = " ".join(args)
    assert "--host-resolver-rules" not in joined


def test_runner_chromium_args_dns_leak_guard_with_proxy(monkeypatch):
    mod = _load_runner_module(monkeypatch)
    args = mod._chromium_args("socks5://127.0.0.1:9050")
    joined = " ".join(args)
    # DNS-over-proxy guard
    assert "--host-resolver-rules=MAP * ~NOTFOUND" in joined
    # Localhost must be excluded (self-play + container services)
    assert "EXCLUDE localhost" in joined
    # WebRTC IP-leak guards
    assert "--webrtc-ip-handling-policy=disable_non_proxied_udp" in joined


# --- 5b. last_url sidecar: cross-op continuity ------------------------------

def test_runner_sidecar_read_write_roundtrip(monkeypatch, tmp_path):
    """After a successful navigation writes the sidecar, a subsequent
    op with no explicit url must resolve to that URL."""
    mod = _load_runner_module(monkeypatch)
    profile = str(tmp_path / "profile")
    assert mod._read_last_url(profile) is None  # nothing yet

    mod._write_last_url(profile, "https://example.com/foo")
    assert mod._read_last_url(profile) == "https://example.com/foo"

    # Overwrite, not append — the NEXT nav's URL is the new source of truth.
    mod._write_last_url(profile, "https://example.com/bar")
    assert mod._read_last_url(profile) == "https://example.com/bar"


def test_runner_sidecar_write_ignores_empty(monkeypatch, tmp_path):
    """A None / empty URL must not overwrite the sidecar — otherwise
    a failed op could blow away the last-known good URL."""
    mod = _load_runner_module(monkeypatch)
    profile = str(tmp_path / "profile")
    mod._write_last_url(profile, "https://example.com/seed")

    mod._write_last_url(profile, None)
    assert mod._read_last_url(profile) == "https://example.com/seed"

    mod._write_last_url(profile, "")
    assert mod._read_last_url(profile) == "https://example.com/seed"


def test_runner_resolve_url_prefers_explicit(monkeypatch, tmp_path):
    """Explicit `url` in the op dict wins over the sidecar."""
    mod = _load_runner_module(monkeypatch)
    profile = str(tmp_path / "profile")
    mod._write_last_url(profile, "https://stale.example/")

    url, used_fallback = mod._resolve_url_or_error(
        {"url": "https://fresh.example/", "profile_dir": profile},
        "extract_text",
    )
    assert url == "https://fresh.example/"
    assert used_fallback is False


def test_runner_resolve_url_falls_back_to_sidecar(monkeypatch, tmp_path):
    mod = _load_runner_module(monkeypatch)
    profile = str(tmp_path / "profile")
    mod._write_last_url(profile, "https://last.example/")

    url, used_fallback = mod._resolve_url_or_error(
        {"profile_dir": profile}, "extract_text",
    )
    assert url == "https://last.example/"
    assert used_fallback is True


def test_runner_resolve_url_errors_when_no_source(monkeypatch, tmp_path):
    """When neither an explicit url nor a sidecar is available, the
    runner must raise a clear actionable error rather than silently
    querying a blank about:blank page."""
    mod = _load_runner_module(monkeypatch)
    profile = str(tmp_path / "profile")
    with pytest.raises(ValueError) as exc:
        mod._resolve_url_or_error({"profile_dir": profile}, "extract_text")
    msg = str(exc.value)
    # Error must name the op AND point the LLM at the fix (navigate or url=...)
    assert "extract_text" in msg
    assert "url=" in msg or "`url=`" in msg or "url" in msg
    assert "navigate" in msg.lower()


def test_runner_close_wipes_sidecar_via_profile_rmtree(monkeypatch, tmp_path):
    """The close op rmtree's the whole profile dir; the sidecar lives
    inside, so this must also clear it."""
    mod = _load_runner_module(monkeypatch)
    profile = str(tmp_path / "profile")
    mod._write_last_url(profile, "https://example.com/")
    assert Path(profile, ".last_url").exists()

    # Invoke the op_close coroutine directly
    import asyncio
    asyncio.run(mod.op_close({"profile_dir": profile}))

    assert not Path(profile).exists(), "profile dir (and sidecar) must be gone"


# --- 5c. interact op: multi-step flows in one Chromium context -------------

@pytest.mark.asyncio
async def test_tool_browser_interact_rejects_empty_actions(tmp_path):
    """The whole point of interact is the sequence — refuse silently
    'succeeding' on an empty list, which would mislead the LLM."""
    from ghost_agent.tools.browser import tool_browser

    stub = _make_sandbox_stub("")
    result = await tool_browser(
        operation="interact", url="http://x", actions=[],
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "ERROR" in result
    assert "actions" in result.lower()


@pytest.mark.asyncio
async def test_tool_browser_interact_rejects_non_dict_actions(tmp_path):
    from ghost_agent.tools.browser import tool_browser

    stub = _make_sandbox_stub("")
    result = await tool_browser(
        operation="interact", url="http://x",
        actions=[{"action": "click", "selector": "a"}, "garbage"],
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "ERROR" in result
    assert "actions[1]" in result


@pytest.mark.asyncio
async def test_tool_browser_interact_passes_actions_through(tmp_path):
    """The runner JSON payload must carry the full `actions` list
    verbatim. This test captures the command and verifies it."""
    from ghost_agent.tools.browser import tool_browser

    # Simulate a successful interact response from the runner
    runner_payload = {
        "actions": [
            {"index": 0, "action": "click", "ok": True, "selector": "[data-app='calc']"},
            {"index": 1, "action": "click", "ok": True, "selector": "#calc-btn-7"},
            {"index": 2, "action": "extract_text", "ok": True, "selector": "#calc-display",
             "text": "7", "length": 1, "truncated": False},
        ],
        "final_url": "file:///workspace/webos/index.html",
        "final_title": "WebOS",
        "used_last_url": False,
    }
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(runner_payload)}\n")
    result = await tool_browser(
        operation="interact", url="file:///workspace/webos/index.html",
        actions=[
            {"action": "click", "selector": "[data-app='calc']"},
            {"action": "click", "selector": "#calc-btn-7"},
            {"action": "extract_text", "selector": "#calc-display"},
        ],
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    # Runner got the JSON payload with all three actions
    assert stub.last_command is not None
    assert '"op": "interact"' in stub.last_command
    assert '"click"' in stub.last_command
    assert '"extract_text"' in stub.last_command
    # Tool output summary format
    assert "STATUS: OK" in result
    assert "OP: interact" in result
    assert "3 OK, 0 errors" in result
    # Per-action breakdown
    assert "[0] OK click '[data-app=\\'calc\\']'" in result or "[0] OK click" in result
    assert "[2] OK extract_text" in result
    assert "TEXT: 7" in result


@pytest.mark.asyncio
async def test_tool_browser_interact_reports_partial_failure(tmp_path):
    """When stop_on_error is false (default), a failing action doesn't
    abort the whole sequence — the summary must show mixed OK/ERR."""
    from ghost_agent.tools.browser import tool_browser

    runner_payload = {
        "actions": [
            {"index": 0, "action": "click", "ok": True, "selector": "#a"},
            {"index": 1, "action": "click", "ok": False,
             "error": "TimeoutError: Timeout 30000ms exceeded"},
            {"index": 2, "action": "extract_text", "ok": True,
             "selector": "#x", "text": "hi", "length": 2, "truncated": False},
        ],
        "final_url": "http://x", "final_title": "t", "used_last_url": False,
    }
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(runner_payload)}\n")
    result = await tool_browser(
        operation="interact", url="http://x",
        actions=[{"action": "click", "selector": "#a"}] * 3,
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "2 OK, 1 error" in result
    # The error message from the failing action must be surfaced, not swallowed
    assert "TimeoutError" in result
    assert "Timeout 30000ms" in result


@pytest.mark.asyncio
async def test_tool_browser_interact_rewrites_screenshot_subaction_paths(tmp_path):
    """Screenshot actions nested inside `actions` need the same
    sandbox-safety + host→container path translation as the top-level
    screenshot op. Without it, paths would skip the safety check and
    either escape the sandbox or fail opaquely."""
    from ghost_agent.tools.browser import tool_browser

    runner_payload = {
        "actions": [
            {"index": 0, "action": "screenshot", "ok": True,
             "path": "/workspace/step.png"},
        ],
        "final_url": "http://x", "final_title": "t", "used_last_url": False,
    }
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(runner_payload)}\n")
    result = await tool_browser(
        operation="interact", url="http://x",
        actions=[{"action": "screenshot", "out_path": "step.png"}],
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    # The command sent to the runner must have the container-absolute path
    assert stub.last_command is not None
    assert "/workspace/step.png" in stub.last_command
    # Traversal must be blocked
    traversal_result = await tool_browser(
        operation="interact", url="http://x",
        actions=[{"action": "screenshot", "out_path": "../../etc/passwd.png"}],
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "ERROR" in traversal_result
    assert "actions[0]" in traversal_result


def test_browser_tool_definition_includes_interact():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    browser_def = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "browser")
    op_enum = browser_def["function"]["parameters"]["properties"]["operation"]["enum"]
    assert "interact" in op_enum
    # actions parameter documented
    props = browser_def["function"]["parameters"]["properties"]
    assert "actions" in props
    assert "stop_on_error" in props


def test_runner_op_interact_registered(monkeypatch):
    """Runner's dispatch table must route 'interact' to op_interact."""
    mod = _load_runner_module(monkeypatch)
    assert "interact" in mod.OPS
    assert mod.OPS["interact"] is mod.op_interact


def test_runner_op_interact_validates_actions_shape(monkeypatch):
    """Empty list / missing key / wrong type must raise with a clear
    message — the runner's own guardrail, before Chromium ever spins up."""
    mod = _load_runner_module(monkeypatch)
    import asyncio
    # Empty actions
    with pytest.raises(ValueError) as exc:
        asyncio.run(mod.op_interact({"actions": [], "profile_dir": "/tmp/prof",
                                     "timeout_ms": 30000}))
    assert "non-empty" in str(exc.value) or "actions" in str(exc.value)

    # Missing actions
    with pytest.raises(ValueError):
        asyncio.run(mod.op_interact({"profile_dir": "/tmp/prof",
                                     "timeout_ms": 30000}))


@pytest.mark.asyncio
async def test_tool_browser_surfaces_last_url_error_cleanly(tmp_path):
    """Integration check: when the runner emits the 'needs URL' error
    because neither explicit url nor sidecar was available, the tool
    must surface it as a STATUS: ERROR with the full message, not hide
    it behind a generic failure."""
    from ghost_agent.tools.browser import tool_browser

    stub = _make_sandbox_stub(
        "[BROWSER_ERR] extract_text needs a URL: pass `url=...` or call "
        "`operation=\"navigate\"` first (this op has no recorded last "
        "URL in the persistent profile)\n",
        exit_code=1,
    )
    result = await tool_browser(
        operation="extract_text", selector="#target",
        sandbox_dir=tmp_path, sandbox_manager=stub,
    )
    assert "ERROR" in result
    assert "extract_text needs a URL" in result
    assert "navigate" in result


# --- 7. Prompt updates ------------------------------------------------------

def test_prompt_mentions_native_browser_tool():
    """The user-facing prompt should advertise the `browser(...)` tool
    as the preferred path, not raw Playwright."""
    prompt_path = Path(__file__).parent.parent / "src/ghost_agent/core/prompts.py"
    content = prompt_path.read_text()
    # Native tool mentioned
    assert "browser(operation=" in content
    # Raw Playwright is still documented but secondary
    assert "RAW-PLAYWRIGHT FALLBACK" in content


def test_prompt_contains_dns_leak_guard():
    """The raw-Playwright fallback in the prompt must now teach the
    --host-resolver-rules DNS-over-proxy flag, not just the proxy URL."""
    prompt_path = Path(__file__).parent.parent / "src/ghost_agent/core/prompts.py"
    content = prompt_path.read_text()
    assert "--host-resolver-rules" in content
    assert "MAP * ~NOTFOUND" in content


# --- 8. web_automation template --------------------------------------------

def test_web_automation_template_registered():
    from ghost_agent.core.challenge_templates import TEMPLATES
    assert "web_automation" in TEMPLATES


def test_web_automation_template_basic_tier_triple_is_valid():
    from ghost_agent.core.challenge_templates import TEMPLATES

    fn = TEMPLATES["web_automation"]
    prompt, setup, validator = fn(tier="basic")
    # Structural checks
    assert "page.html" in prompt
    assert "playwright" in prompt.lower()
    assert "#secret" in prompt
    # Basic tier: setup is plain HTML
    assert "<div id=\"secret\">" in setup
    # Validator runs solution.py and compares a 12-char secret
    assert "solution.py" in validator
    assert "expected" in validator


def test_web_automation_template_hard_mode_twist():
    from ghost_agent.core.challenge_templates import TEMPLATES

    fn = TEMPLATES["web_automation"]
    _, setup_adv, _ = fn(tier="advanced")
    _, setup_exp, _ = fn(tier="expert")
    # Hard mode must have the JS-injected twist
    assert "DOMContentLoaded" in setup_adv
    assert "DOMContentLoaded" in setup_exp
    # And include a decoy via noscript / comment
    assert "<noscript>" in setup_adv
    # Basic tier must NOT have the twist (otherwise tier ramp is broken)
    _, setup_basic, _ = fn(tier="basic")
    assert "DOMContentLoaded" not in setup_basic


def test_web_automation_validator_can_recompute_expected():
    """The validator reads the `expected` secret out of the challenge
    triple itself (baked into the validator body). Sanity-check the
    triple is self-consistent."""
    from ghost_agent.core.challenge_templates import TEMPLATES

    fn = TEMPLATES["web_automation"]
    prompt, setup, validator = fn(tier="basic")
    # Extract the secret from setup script body
    import re
    m = re.search(r'id="secret">([A-Z0-9]{12})<', setup)
    assert m, "Could not find secret in basic-tier setup"
    secret = m.group(1)
    # Validator must be looking for that same 12-char token
    assert f"'{secret}'" in validator or f'"{secret}"' in validator


def test_web_automation_classified_by_cluster_keywords():
    """If a challenge talks about scraping / headless browser, it
    should route into the `web_automation` cluster so the frontier
    tracker + template router line up."""
    from ghost_agent.memory.frontier import classify_cluster
    assert classify_cluster("Write a playwright scraper") == "web_automation"
    assert classify_cluster("Use a headless browser to click through ...") == "web_automation"


# --- 9. Registry exposure ---------------------------------------------------

def test_browser_tool_registered_in_definitions():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "browser" in names, f"browser tool missing from TOOL_DEFINITIONS (have: {sorted(names)})"


def test_browser_tool_definition_advertises_all_ops():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    browser_def = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "browser")
    op_enum = browser_def["function"]["parameters"]["properties"]["operation"]["enum"]
    assert set(op_enum) == {
        "navigate", "extract_text", "click", "screenshot", "close", "interact",
    }


def test_browser_tool_in_forbidden_import_list():
    """Native tools must not be importable as Python modules — make
    sure `browser` is on the execute-guard blocklist."""
    exec_path = Path(__file__).parent.parent / "src/ghost_agent/tools/execute.py"
    content = exec_path.read_text()
    # The forbidden_modules list is in the file — check `browser` is on it
    assert '"browser"' in content and "forbidden_modules" in content
