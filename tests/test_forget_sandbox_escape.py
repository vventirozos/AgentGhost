"""Regression tests for the sandbox-escape in `tool_unified_forget`.

The disk-deletion guard used `str(resolved).startswith(str(sandbox_root))`,
which treats a *sibling* directory sharing the prefix (``/x/sandbox_evil``
vs root ``/x/sandbox``) as "inside". Combined with `resolve()` following
symlinks, an LLM-placed symlink inside the sandbox could escape and delete
a sibling-prefixed file. The fix uses path-component containment
(`_is_within_root`) and refuses to delete through symlinks.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.memory import tool_unified_forget, _is_within_root


def _mk_memsys():
    """A memory_system stub whose vector/sweep paths are no-ops, so the
    test exercises only the disk-deletion logic."""
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.collection.query.return_value = {"ids": None}
    return ms


# -----------------------------------------------------------------
# _is_within_root — the containment primitive
# -----------------------------------------------------------------

def test_is_within_root_rejects_sibling_prefix():
    root = Path("/x/sandbox")
    # The exact case str.startswith got wrong:
    assert _is_within_root(Path("/x/sandbox_evil/secret.txt"), root) is False
    assert _is_within_root(Path("/x/sandbox/sub/f.txt"), root) is True
    assert _is_within_root(Path("/x/sandbox"), root) is True
    assert _is_within_root(Path("/y/other/f.txt"), root) is False


# -----------------------------------------------------------------
# tool_unified_forget — end-to-end on the filesystem
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_forget_deletes_normal_file_inside_sandbox(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    f = sandbox / "notes.txt"
    f.write_text("hi")
    await tool_unified_forget("notes.txt", sandbox_dir=sandbox, memory_system=_mk_memsys())
    assert not f.exists()


@pytest.mark.asyncio
async def test_forget_symlink_to_sibling_prefixed_dir_is_refused(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    evil = tmp_path / "sandbox_evil"          # name shares the 'sandbox' prefix
    evil.mkdir()
    target = evil / "secret.txt"
    target.write_text("DO NOT DELETE")
    # Symlink inside the sandbox whose RESOLVED path is the sibling-prefixed
    # file. Under the old str.startswith check this passed containment and
    # was unlinked (sandbox-escape). Now it must be refused.
    link = sandbox / "secret.txt"
    link.symlink_to(target)

    await tool_unified_forget("secret.txt", sandbox_dir=sandbox, memory_system=_mk_memsys())
    assert target.exists(), "sandbox-escape: sibling-prefixed file was deleted via symlink"


@pytest.mark.asyncio
async def test_forget_refuses_symlink_pointing_outside(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("DO NOT DELETE")
    link = sandbox / "secret.txt"
    link.symlink_to(outside)

    await tool_unified_forget("secret.txt", sandbox_dir=sandbox, memory_system=_mk_memsys())
    assert outside.exists() and outside.read_text() == "DO NOT DELETE"
