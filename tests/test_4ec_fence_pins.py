"""§4EC — `evolve/fence.py` survivors of the §R re-verification of §4CP (2026-09-02):
the filesystem-alias guard `resolves_to_immutable`, `is_mutable`'s fail-closed
arms, the digest's UNREADABLE marker and the BYTECODE label. Driven on temp
repos so every arm decides alone."""
import os
from pathlib import Path

import pytest

from ghost_agent.evolve import fence


def _repo(tmp_path):
    root = tmp_path / "repo"
    for d in ("src/ghost_agent/tools", "src/ghost_agent/memory", "scripts", "tests"):
        (root / d).mkdir(parents=True)
    (root / "src/ghost_agent/core").mkdir(parents=True)
    (root / "src/ghost_agent/core/agent.py").write_text("# the immutable FILE entry\n")
    (root / "src/ghost_agent/memory/store.py").write_text("x = 1\n")
    (root / "src/ghost_agent/tools/plain.py").write_text("y = 1\n")
    (root / "tests/t.py").write_text("t\n")
    return root


class TestResolvesToImmutable:
    def test_a_link_into_an_immutable_tree_is_named(self, tmp_path):
        root = _repo(tmp_path)
        (root / "src/ghost_agent/tools/alias.py").symlink_to(root / "tests/t.py")
        why = fence.resolves_to_immutable("src/ghost_agent/tools/alias.py", root)
        assert "tests/t.py" in why and "immutable tree tests/" in why

    def test_a_link_onto_an_immutable_file_entry_is_named(self, tmp_path):
        """The file-shaped entries of IMMUTABLE_PREFIXES (no trailing slash) are
        compared by samefile; the dir-shaped ones by rglob. L215/L227 keep them apart."""
        root = _repo(tmp_path)
        (root / "src/ghost_agent/tools/alias.py").symlink_to(root / "src/ghost_agent/core/agent.py")
        why = fence.resolves_to_immutable("src/ghost_agent/tools/alias.py", root)
        assert "IS src/ghost_agent/core/agent.py" in why

    def test_a_later_prefix_is_still_checked_when_earlier_ones_are_absent(self, tmp_path):
        """L225/L228/L231 `continue` → `break`: a repo without `tests/` (an
        earlier entry) must still catch a link into `src/ghost_agent/memory/`."""
        root = _repo(tmp_path)
        import shutil; shutil.rmtree(root / "tests"); shutil.rmtree(root / "scripts")
        (root / "src/ghost_agent/tools/alias.py").symlink_to(root / "src/ghost_agent/memory/store.py")
        why = fence.resolves_to_immutable("src/ghost_agent/tools/alias.py", root)
        assert "memory/store.py" in why

    @pytest.mark.parametrize("path", ["src/ghost_agent/tools/plain.py", "src/ghost_agent/tools/does_not_exist.py"])
    def test_a_plain_or_missing_mutable_path_is_clean(self, tmp_path, path):
        assert fence.resolves_to_immutable(path, _repo(tmp_path)) == ""

    @pytest.mark.parametrize("path", ["../outside.py", "/etc/hosts"])
    def test_an_escaping_path_is_refused_not_resolved(self, tmp_path, path):
        assert "escapes the repo" in fence.resolves_to_immutable(path, _repo(tmp_path))


class TestIsMutableFailsClosed:
    @pytest.mark.parametrize("raw,reason", [
        ("", "empty path"), ("   ", "empty path"),
        ("/etc/hosts", "absolute path"), ("~/x.py", "absolute path"),
        ("../x.py", "escapes"), (".", "escapes"), ("a/../../x.py", "escapes"),
    ])
    def test_refusals(self, raw, reason):
        ok, why = fence.is_mutable(raw)
        assert ok is False and reason in why

    def test_a_mutable_path_is_allowed_and_an_immutable_one_is_not(self):
        assert fence.is_mutable("src/ghost_agent/tools/plain.py")[0] is True
        assert fence.is_mutable("tests/t.py")[0] is False


def test_an_unreadable_file_is_marked_in_the_digest(tmp_path):
    """L352: an OSError while hashing must leave a visible UNREADABLE marker,
    not an omitted entry — a file made unreadable is a change, not nothing."""
    if os.geteuid() == 0:
        pytest.skip("root reads everything")
    root = tmp_path / "repo"; (root / "scripts").mkdir(parents=True)
    f = root / "scripts" / "locked.py"; f.write_text("x\n"); f.chmod(0)
    try:
        d = fence.harness_digest(root, trees=("scripts",))
    finally:
        f.chmod(0o644)
    assert any(v.startswith("UNREADABLE:") for v in d.values()), d


def test_bytecode_label_applies_to_pyc_anywhere_and_to_anything_under_pycache():
    before = {"a/__pycache__/m.cpython-310.pyc": "1", "top.pyc": "1", "a/__pycache__/notes.txt": "1", "a/m.py": "1"}
    after = {k: "2" for k in before}
    out = fence.compare_harness(before, after)
    assert "BYTECODE MODIFIED a/__pycache__/m.cpython-310.pyc" in out
    assert "BYTECODE MODIFIED top.pyc" in out
    assert "BYTECODE MODIFIED a/__pycache__/notes.txt" in out
    assert "MODIFIED a/m.py" in out and "BYTECODE MODIFIED a/m.py" not in out
