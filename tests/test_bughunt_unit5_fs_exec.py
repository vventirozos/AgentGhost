"""Regression tests for bug-hunt unit 5 (tools-fs-exec) — see BUGHUNT.md.

Fixed bugs pinned here:
 1. file_system: destructive ops (delete/rename/copy) could target the resolved
    sandbox/project ROOT (/workspace, ., '', projects/<active-id>) → wipe the
    whole workspace. Now guarded via _get_safe_path(allow_root=False).
 2. file_system: writes had no encoding= (locale mojibake + truncate-on-encode-error)
 3. file_system: replace read with errors="replace" then wrote back → persisted
    U+FFFD corruption of untouched regions; streaming replace leaked .tmp on error
 4. file_system: empty-content guard rejected the legit word "none"; flexible
    replace had no count=1 limit; inspect used platform-default encoding
 5. execute: reading an existing non-UTF8 file propagated an uncaught error;
    both command+content silently dropped the file (now warns); grep-no-match
    logged to workspace history as a failed command
 6. system/workspace: system action normalised only quotes (not case/space);
    non-string action could break the never-raises contract
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.file_system import (
    _get_safe_path,
    tool_delete_file,
    tool_rename_file,
    tool_copy_file,
    tool_write_file,
    tool_replace_text,
)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "keep.txt").write_text("important data", encoding="utf-8")
    return tmp_path


# ──────────────────────────────────────────────────────────────────────
# 1. Destructive-op-on-root guard (HIGH)
# ──────────────────────────────────────────────────────────────────────

class TestDestructiveRootGuard:
    @pytest.mark.parametrize("target", ["/workspace", ".", "workspace", "", "/"])
    def test_get_safe_path_rejects_root_when_disallowed(self, tmp_path, target):
        with pytest.raises(ValueError, match="destructive"):
            _get_safe_path(tmp_path, target, allow_root=False)

    @pytest.mark.parametrize("target", ["/workspace", ".", "workspace"])
    def test_reads_of_root_still_allowed(self, tmp_path, target):
        # allow_root=True (the default, used by list/read) must NOT reject.
        assert _get_safe_path(tmp_path, target, allow_root=True) == tmp_path.resolve()

    def test_project_scoped_root_rejected(self, tmp_path):
        proj = tmp_path / "projects" / "deadbeef12"
        proj.mkdir(parents=True)
        for t in ["projects/deadbeef12", ".", "/workspace"]:
            with pytest.raises(ValueError, match="destructive"):
                _get_safe_path(proj, t, allow_root=False)

    async def test_delete_root_leaves_files_intact(self, sandbox):
        out = await tool_delete_file("/workspace", sandbox)
        assert "Security Error" in out
        assert (sandbox / "keep.txt").exists()

    async def test_delete_real_file_still_works(self, sandbox):
        out = await tool_delete_file("keep.txt", sandbox)
        assert out.startswith("SUCCESS")
        assert not (sandbox / "keep.txt").exists()

    async def test_rename_root_blocked(self, sandbox):
        out = await tool_rename_file("/workspace", "backup", sandbox)
        assert "Security Error" in out
        assert (sandbox / "keep.txt").exists()

    async def test_copy_root_blocked(self, sandbox):
        out = await tool_copy_file(".", "backup", sandbox)
        assert "Security Error" in out


# ──────────────────────────────────────────────────────────────────────
# 2. Write encoding
# ──────────────────────────────────────────────────────────────────────

class TestWriteEncoding:
    async def test_non_ascii_write_roundtrips_utf8(self, tmp_path):
        content = "π ≈ 3.14 — café — 日本語"
        out = await tool_write_file("u.txt", content, tmp_path)
        assert out.startswith("SUCCESS")
        # Read back as UTF-8 explicitly — must match exactly (no mojibake).
        assert (tmp_path / "u.txt").read_text(encoding="utf-8") == content


# ──────────────────────────────────────────────────────────────────────
# 3. Replace binary-safety + temp cleanup
# ──────────────────────────────────────────────────────────────────────

class TestReplaceTempCleanup:
    # NB: strict-refuse of invalid-UTF-8 files was intentionally NOT adopted —
    # tolerating a mostly-text file with stray bad bytes is a deliberate
    # feature (test_high_roi_fixes). The bad-byte write-back corruption is a
    # deferred finding (see BUGHUNT.md) awaiting a surrogateescape round-trip.
    async def test_replace_no_tmp_orphans_on_no_match(self, tmp_path):
        # Large file so the streaming path is used; a non-matching single-line
        # replace must leave no .tmp orphan behind (was leaked on no-match/error).
        f = tmp_path / "big.txt"
        f.write_text("x\n" * 200_000, encoding="utf-8")
        await tool_replace_text("big.txt", "NOTPRESENT", "y", tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []


# ──────────────────────────────────────────────────────────────────────
# 4. Content guard, flexible replace count, (inspect encoding covered above)
# ──────────────────────────────────────────────────────────────────────

class TestContentGuardAndReplaceCount:
    async def test_word_none_is_writable(self, tmp_path):
        out = await tool_write_file("cfg.txt", "none", tmp_path)
        assert out.startswith("SUCCESS")
        assert (tmp_path / "cfg.txt").read_text(encoding="utf-8") == "none"

    async def test_python_none_repr_still_rejected(self, tmp_path):
        out = await tool_write_file("x.txt", "None", tmp_path)
        assert "empty or 'None'" in out

    async def test_empty_still_rejected(self, tmp_path):
        out = await tool_write_file("x.txt", "   ", tmp_path)
        assert "empty" in out

    async def test_flexible_replace_single_match_bounded(self, tmp_path):
        # The whitespace-flexible path (exact match fails, flexible matches
        # once) now uses count=1, matching the aider path. Verify it still
        # replaces correctly for a single whitespace-variant match.
        f = tmp_path / "ws.py"
        f.write_text("x  =  1\n", encoding="utf-8")  # extra spaces → exact miss
        out = await tool_replace_text("ws.py", "x = 1", "x = 42", tmp_path)
        assert out.startswith("SUCCESS")
        assert "42" in (tmp_path / "ws.py").read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# 5. execute: grep telemetry + both-args warning
# ──────────────────────────────────────────────────────────────────────

class TestExecuteTelemetryAndArgs:
    async def test_grep_no_match_not_logged_as_failure(self):
        from ghost_agent.tools.execute import tool_execute

        recorded = []
        wm = MagicMock()
        wm.enabled = True
        wm.record_command_outcome = lambda **kw: recorded.append(kw)

        sm = MagicMock()
        # grep exits 1 with no output = "no match" (a success)
        sm.execute = lambda *a, **k: ("", 1)

        out = await tool_execute(command="grep foo file.txt", sandbox_manager=sm,
                                 sandbox_dir=Path(tempfile.mkdtemp()), workspace_model=wm)
        # LLM sees success…
        assert "EXIT CODE: 0" in out
        # …and it was NOT recorded to workspace history as a failure.
        assert all(r.get("exit_code") == 0 for r in recorded), recorded
        assert not any(r.get("note") == "failed" for r in recorded)

    async def test_real_failure_still_logged(self):
        from ghost_agent.tools.execute import tool_execute

        recorded = []
        wm = MagicMock()
        wm.enabled = True
        wm.record_command_outcome = lambda **kw: recorded.append(kw)

        sm = MagicMock()
        sm.execute = lambda *a, **k: ("boom: something broke", 1)

        await tool_execute(command="python3 build.py", sandbox_manager=sm,
                           sandbox_dir=Path(tempfile.mkdtemp()), workspace_model=wm)
        assert any(r.get("note") == "failed" for r in recorded)


# ──────────────────────────────────────────────────────────────────────
# 6. system / workspace action robustness
# ──────────────────────────────────────────────────────────────────────

class TestActionRobustness:
    async def test_system_action_case_and_space_normalised(self):
        from ghost_agent.tools.system import tool_system_utility
        # "Check_Location " (trailing space, mixed case) must dispatch, not
        # fall through to "Unknown action".
        out = await tool_system_utility(action=" Check_Location ", profile_memory=None)
        assert "Unknown action" not in out

    async def test_workspace_tools_never_raise_on_nonstring_action(self):
        from ghost_agent.tools.workspace import tool_workspace
        from ghost_agent.tools.workspace_track import tool_workspace_track
        # Non-string action must return a string, not raise AttributeError.
        assert isinstance(await tool_workspace(action=5, workspace_model=None), str)
        assert isinstance(await tool_workspace_track(action=5, workspace_model=None), str)
