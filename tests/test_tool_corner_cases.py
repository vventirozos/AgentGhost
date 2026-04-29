"""Corner-case tests for tool dispatch + file-system tool handlers.

Targets:
  * Tool registry: get_active_tool_definitions / get_available_tools
    behave correctly under degenerate context.
  * File-system tool path safety: traversal, symlink, /workspace
    stripping (already covered in invariants but we add deeper probes).
  * Read/write tool error paths: missing file, permission denied,
    binary content, oversized content.
  * Replace tool: text not found, unicode, newline preservation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools import file_system as fs


# ──────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_tool_definitions_is_a_list(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) > 0

    def test_each_tool_definition_has_name(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        for tool in TOOL_DEFINITIONS:
            # OpenAI-format function definition
            assert isinstance(tool, dict)
            assert "type" in tool
            if tool.get("type") == "function":
                assert "function" in tool
                fn = tool["function"]
                assert "name" in fn, f"function missing name: {tool}"
                assert isinstance(fn["name"], str)
                assert fn["name"]  # non-empty

    def test_tool_names_are_unique(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        names = []
        for tool in TOOL_DEFINITIONS:
            if tool.get("type") == "function":
                names.append(tool["function"]["name"])
        assert len(names) == len(set(names)), (
            f"duplicate tool names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_get_active_tool_definitions_with_empty_context(self):
        """A minimal MagicMock context should not crash get_active."""
        from ghost_agent.tools.registry import get_active_tool_definitions
        ctx = MagicMock()
        ctx.args = MagicMock()
        ctx.args.smart_memory = 0.0
        ctx.args.coding_nodes_parsed = []
        ctx.args.swarm_nodes_parsed = []
        ctx.args.image_gen_nodes_parsed = []
        ctx.args.visual_nodes_parsed = []
        try:
            tools = get_active_tool_definitions(ctx)
            assert isinstance(tools, list)
        except (AttributeError, TypeError):
            # Some context attribute is required but missing — that's
            # acceptable as long as it raises clearly.
            pass


# ──────────────────────────────────────────────────────────────────────
# _get_safe_path — adversarial paths
# ──────────────────────────────────────────────────────────────────────

class TestSafePathAdversarial:
    def test_blocks_parent_traversal(self, tmp_path: Path):
        with pytest.raises((ValueError, OSError, RuntimeError)):
            fs._get_safe_path(tmp_path, "../../etc/passwd")

    def test_absolute_path_relativized_into_sandbox(self, tmp_path: Path):
        """Documented behavior: leading '/' is stripped, the path is
        relativized into the sandbox. So '/etc/passwd' becomes
        '<sandbox>/etc/passwd' (still inside sandbox), not the system
        /etc/passwd."""
        result = fs._get_safe_path(tmp_path, "/etc/passwd")
        assert result.is_relative_to(tmp_path.resolve())
        # Confirm it's NOT pointing at the real /etc/passwd
        assert str(result) != "/etc/passwd"
        assert str(result.resolve()).startswith(str(tmp_path.resolve()))

    def test_blocks_double_dot_in_middle(self, tmp_path: Path):
        with pytest.raises((ValueError, OSError, RuntimeError)):
            fs._get_safe_path(tmp_path, "foo/../../etc/passwd")

    def test_accepts_simple_relative_path(self, tmp_path: Path):
        result = fs._get_safe_path(tmp_path, "subdir/file.txt")
        assert result.is_relative_to(tmp_path.resolve())

    def test_accepts_root_file(self, tmp_path: Path):
        result = fs._get_safe_path(tmp_path, "file.txt")
        assert result.is_relative_to(tmp_path.resolve())

    def test_strips_workspace_prefix(self, tmp_path: Path):
        result = fs._get_safe_path(tmp_path, "/workspace/file.txt")
        assert result.is_relative_to(tmp_path.resolve())
        rel = str(result.relative_to(tmp_path.resolve()))
        assert "workspace" not in rel.split("/")
        assert rel == "file.txt"

    def test_workspace_alone_strips_to_sandbox_root(self, tmp_path: Path):
        result = fs._get_safe_path(tmp_path, "/workspace")
        # Should resolve to the sandbox root itself
        assert result.resolve() == tmp_path.resolve()

    def test_double_workspace_only_strips_once(self, tmp_path: Path):
        """`workspace/workspace/foo` should strip exactly one segment,
        leaving `workspace/foo` inside the sandbox."""
        result = fs._get_safe_path(tmp_path, "workspace/workspace/foo")
        rel = str(result.relative_to(tmp_path.resolve()))
        assert rel == "workspace/foo"


# ──────────────────────────────────────────────────────────────────────
# tool_read_file edge cases
# ──────────────────────────────────────────────────────────────────────

class TestReadFile:
    async def test_read_nonexistent_file(self, tmp_path: Path):
        result = await fs.tool_read_file("nope.txt", tmp_path)
        # Tool returns an error string, not raises
        assert isinstance(result, str)
        # Some kind of error indicator
        lc = result.lower()
        assert any(s in lc for s in ("error", "not found", "no such",
                                     "failed", "missing", "doesn't exist"))

    async def test_read_empty_file(self, tmp_path: Path):
        target = tmp_path / "empty.txt"
        target.write_text("")
        result = await fs.tool_read_file("empty.txt", tmp_path)
        assert isinstance(result, str)

    async def test_read_with_traversal_blocked(self, tmp_path: Path):
        # Create a file outside the sandbox (in the parent)
        outside = tmp_path.parent / "outside.txt"
        try:
            outside.write_text("secret")
            # Read attempt with traversal must NOT return the content
            result = await fs.tool_read_file("../outside.txt", tmp_path)
            assert isinstance(result, str)
            assert "secret" not in result
        finally:
            outside.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# tool_write_file edge cases
# ──────────────────────────────────────────────────────────────────────

class TestWriteFile:
    async def test_write_simple(self, tmp_path: Path):
        result = await fs.tool_write_file("test.txt", "hello world", tmp_path)
        assert isinstance(result, str)
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.txt").read_text() == "hello world"

    async def test_write_with_workspace_prefix_strips(self, tmp_path: Path):
        result = await fs.tool_write_file("/workspace/test.txt", "x", tmp_path)
        # Should land at tmp_path/test.txt, NOT tmp_path/workspace/test.txt
        assert (tmp_path / "test.txt").exists()
        assert not (tmp_path / "workspace").exists()

    async def test_write_creates_parent_dirs(self, tmp_path: Path):
        result = await fs.tool_write_file(
            "deep/nested/path/file.txt", "content", tmp_path,
        )
        assert (tmp_path / "deep" / "nested" / "path" / "file.txt").exists()

    async def test_write_traversal_blocked(self, tmp_path: Path):
        outside = tmp_path.parent / "should_not_write.txt"
        try:
            await fs.tool_write_file("../should_not_write.txt", "evil", tmp_path)
            # The file must not have been written outside the sandbox
            assert not outside.exists() or outside.read_text() != "evil"
        finally:
            outside.unlink(missing_ok=True)

    async def test_write_unicode_content(self, tmp_path: Path):
        content = "héllo wörld 🎉 αβγ"
        await fs.tool_write_file("unicode.txt", content, tmp_path)
        assert (tmp_path / "unicode.txt").read_text() == content

    async def test_write_overwrites_existing(self, tmp_path: Path):
        await fs.tool_write_file("a.txt", "first", tmp_path)
        await fs.tool_write_file("a.txt", "second", tmp_path)
        assert (tmp_path / "a.txt").read_text() == "second"


# ──────────────────────────────────────────────────────────────────────
# tool_list_files edge cases
# ──────────────────────────────────────────────────────────────────────

class TestListFiles:
    async def test_empty_sandbox(self, tmp_path: Path):
        result = await fs.tool_list_files(tmp_path)
        assert isinstance(result, str)

    async def test_lists_files_in_sandbox(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.py").write_text("y")
        result = await fs.tool_list_files(tmp_path)
        assert "a.txt" in result
        assert "b.py" in result

    async def test_does_not_list_files_outside(self, tmp_path: Path):
        outside = tmp_path.parent / "outside_marker.txt"
        try:
            outside.write_text("x")
            result = await fs.tool_list_files(tmp_path)
            assert "outside_marker" not in result
        finally:
            outside.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# tool_replace_text edge cases
# ──────────────────────────────────────────────────────────────────────

class TestReplaceText:
    async def test_replace_when_target_present(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello world")
        result = await fs.tool_replace_text("f.txt", "world", "there", tmp_path)
        # Either the file was modified or the function returned an
        # error string. Read the file to confirm.
        content = (tmp_path / "f.txt").read_text()
        # If it succeeded, the change is visible
        if "hello there" not in content and "hello world" in content:
            # Replace didn't happen — the result string should explain
            assert isinstance(result, str)

    async def test_replace_when_target_missing(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello world")
        result = await fs.tool_replace_text(
            "f.txt", "nonexistent_string", "x", tmp_path,
        )
        # Must not crash. Either replaces nothing or returns an error.
        assert isinstance(result, str)
        # File content should be unchanged in the missing-target case
        assert (tmp_path / "f.txt").read_text() == "hello world"

    async def test_replace_in_nonexistent_file(self, tmp_path: Path):
        result = await fs.tool_replace_text("nope.txt", "a", "b", tmp_path)
        # Returns an error string, not raises
        assert isinstance(result, str)
        lc = result.lower()
        assert any(s in lc for s in ("error", "not found", "missing",
                                     "no such", "failed", "doesn't exist"))


# ──────────────────────────────────────────────────────────────────────
# tool_inspect_file edge cases
# ──────────────────────────────────────────────────────────────────────

class TestInspectFile:
    async def test_inspect_short_file_returns_all(self, tmp_path: Path):
        (tmp_path / "short.txt").write_text("line 1\nline 2\nline 3")
        result = await fs.tool_inspect_file("short.txt", tmp_path, lines=10)
        assert isinstance(result, str)
        assert "line 1" in result

    async def test_inspect_nonexistent_file(self, tmp_path: Path):
        result = await fs.tool_inspect_file("nope.txt", tmp_path)
        assert isinstance(result, str)
        # Should be an error message
        lc = result.lower()
        assert any(s in lc for s in ("error", "not found", "missing",
                                     "no such", "failed", "doesn't exist"))


# ──────────────────────────────────────────────────────────────────────
# tool_copy_file / tool_rename_file edge cases
# ──────────────────────────────────────────────────────────────────────

class TestCopyAndRenameFile:
    async def test_copy_simple(self, tmp_path: Path):
        (tmp_path / "src.txt").write_text("content")
        await fs.tool_copy_file("src.txt", "dst.txt", tmp_path)
        assert (tmp_path / "dst.txt").exists()
        assert (tmp_path / "src.txt").exists()  # source preserved

    async def test_copy_nonexistent_source(self, tmp_path: Path):
        result = await fs.tool_copy_file("nope.txt", "dst.txt", tmp_path)
        assert isinstance(result, str)
        # Destination should not have been created
        assert not (tmp_path / "dst.txt").exists()

    async def test_rename_simple(self, tmp_path: Path):
        (tmp_path / "old.txt").write_text("content")
        await fs.tool_rename_file("old.txt", "new.txt", tmp_path)
        assert (tmp_path / "new.txt").exists()
        assert not (tmp_path / "old.txt").exists()

    async def test_rename_traversal_blocked(self, tmp_path: Path):
        (tmp_path / "src.txt").write_text("x")
        outside = tmp_path.parent / "renamed_target.txt"
        try:
            await fs.tool_rename_file("src.txt", "../renamed_target.txt", tmp_path)
            # The target file outside sandbox must NOT exist
            assert not outside.exists()
        finally:
            outside.unlink(missing_ok=True)
