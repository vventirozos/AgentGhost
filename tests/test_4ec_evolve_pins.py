"""§4EC — `evolve/mutator.py` survivors of the §R re-verification of §4CP (2026-09-02).

The §4CP fixes were described as "mutation-tested to confirm it can fail" with no
score quoted; a 40% sample of a whole-file battery (354 mutants) left 82 alive, all
of them after the wide tier too. These pins cover the §4CP fix sites among them:
the model-controlled anchor path, snapshot containment, the `+++`/`---` ambiguity,
the guard-flag prefix rule and the relocation objection's control flow. Each names
the world in which it fails (the mutant line).
"""
import os
from pathlib import Path

import pytest

from ghost_agent.evolve import mutator as M


# ── _read_anchor_source: the path is MODEL-CONTROLLED and read BEFORE the fence ──
class TestAnchorSourceContainment:
    def test_an_escaping_relative_path_is_not_read(self, tmp_path):
        """L1142 `startswith(str(base) + os.sep)` → False: `../secret.txt` is read."""
        root = tmp_path / "repo"; root.mkdir()
        (root / "in.py").write_text("inside\n")
        (tmp_path / "secret.txt").write_text("outside\n")
        assert M._read_anchor_source(root, "in.py") == ["inside"]
        assert M._read_anchor_source(root, "../secret.txt") is None

    def test_absolute_outside_symlink_and_oversized_targets_are_refused(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"; root.mkdir()
        (root / "big.py").write_text("x" * 10)
        (tmp_path / "secret.txt").write_text("outside\n")
        (root / "out_link.py").symlink_to(tmp_path / "secret.txt")
        (root / "in_link.py").symlink_to(root / "big.py")
        assert M._read_anchor_source(root, str(root / "big.py")) is None      # absolute
        assert M._read_anchor_source(root, "out_link.py") is None             # symlink OUT: containment
        # a symlink INSIDE the repo resolves before the lstat, so the S_ISREG
        # arm never sees a link — the docstring's "symlink" refusal is carried
        # by containment alone (recorded in §4EC, not a defect: the target is
        # in-tree either way)
        assert M._read_anchor_source(root, "in_link.py") == ["x" * 10]
        monkeypatch.setattr(M, "MAX_ANCHOR_SOURCE_BYTES", 5)
        assert M._read_anchor_source(root, "big.py") is None                   # bounded


# ── containment_violation: the fence tests strings; the filesystem can escape ──
class TestSnapshotContainment:
    DIFF = ("--- a/src/ghost_agent/tools/link/x.py\n+++ b/src/ghost_agent/tools/link/x.py\n"
            "@@ -1 +1 @@\n-a\n+b\n")

    def test_a_symlink_escaping_the_snapshot_is_a_violation(self, tmp_path):
        """L1489-1491: the probe resolves outside `dest` → a message, never ''."""
        dest = tmp_path / "snap"; (dest / "src" / "ghost_agent" / "tools").mkdir(parents=True)
        outside = tmp_path / "outside"; outside.mkdir()
        (dest / "src" / "ghost_agent" / "tools" / "link").symlink_to(outside)
        why = M.containment_violation(dest, self.DIFF)
        assert why and ("symlink" in why or "outside the snapshot" in why)

    def test_a_dotdot_path_resolving_outside_is_a_violation(self, tmp_path):
        """L1489-1491 exactly: no symlink, the path itself walks out."""
        dest = tmp_path / "snap"; (dest / "src").mkdir(parents=True)
        (tmp_path / "escape.py").write_text("x\n")
        diff = "--- a/../escape.py\n+++ b/../escape.py\n@@ -1 +1 @@\n-x\n+y\n"
        why = M.containment_violation(dest, diff)
        assert why and ("outside the snapshot" in why or "symlink" in why), why

    def test_a_contained_path_is_clean(self, tmp_path):
        dest = tmp_path / "snap"; (dest / "src" / "ghost_agent" / "tools" / "link").mkdir(parents=True)
        (dest / "src" / "ghost_agent" / "tools" / "link" / "x.py").write_text("b\n")
        assert M.containment_violation(dest, self.DIFF) == ""


# ── _is_file_header: `--- `/`+++ ` are ambiguous with content ────────────────
class TestFileHeaderAmbiguity:
    def test_only_a_pair_is_a_header(self):
        lines = ["--- a/f.py", "+++ b/f.py", "@@ -1 +1 @@", "--- x", " ctx", "+++ /dev/null", "--- y", "+++ z"]
        assert M._is_file_header(lines, 0) and M._is_file_header(lines, 1)
        assert not M._is_file_header(lines, 3)      # `--- x` followed by context: content
        assert not M._is_file_header(lines, 5)      # `+++ /dev/null` not preceded by `--- `: content
        assert M._is_file_header(lines, 6) and M._is_file_header(lines, 7)   # a pair, wherever it sits
        assert not M._is_file_header(lines, -1) and not M._is_file_header(lines, 99)

    def test_a_plus_header_needs_the_minus_header_immediately_before(self):
        lines = ["--- a/f.py", "context", "+++ b/f.py"]
        assert not M._is_file_header(lines, 2)


# ── guard_flags: headers are not removed lines ───────────────────────────────
def test_guard_flags_ignores_the_minus_header_even_when_its_path_looks_guardy():
    diff = "--- a/tools/refuse_guard.py\n+++ b/tools/refuse_guard.py\n@@ -1 +1 @@\n+x = 1\n"
    assert M.guard_flags(diff) == []
    diff2 = "--- a/tools/x.py\n+++ b/tools/x.py\n@@ -1 +1 @@\n-    raise PermissionError('refused')\n+    pass\n"
    assert M.guard_flags(diff2) != []


# ── applied_where_it_said: control flow of the relocation objection ──────────
def _repo(tmp_path, text):
    d = tmp_path / "snap"; d.mkdir(parents=True, exist_ok=True)
    (d / "f.py").write_text(text)
    return d


class TestRelocationObjection:
    def test_an_objection_in_the_first_hunk_is_not_lost_behind_a_clean_second_hunk(self, tmp_path):
        """L1417 `if bad: return bad` → False: only the LAST flush counts."""
        # file after patching: the first hunk's post-image is NOT at line 1
        repo = _repo(tmp_path, "\n".join(["pad"] * 5 + ["new1", "k1", "k2", "k3", "new2"]) + "\n")
        diff = ("--- a/f.py\n+++ b/f.py\n"
                "@@ -1,1 +1,1 @@\n-old1\n+new1\n"          # declared at 1, landed at 6
                "@@ -6,1 +10,1 @@\n-old2\n+new2\n")        # declared at 10: correct
        assert "hunk declared line 1" in M.applied_where_it_said(repo, diff)

    def test_an_objection_is_not_lost_at_a_file_boundary(self, tmp_path):
        """L1435 `return bad` → None: the objection raised at the next `+++` is dropped."""
        repo = _repo(tmp_path, "\n".join(["pad"] * 5 + ["new1"]) + "\n")
        (repo / "g.py").write_text("gnew\n")
        diff = ("--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old1\n+new1\n"
                "--- a/g.py\n+++ b/g.py\n@@ -1,1 +1,1 @@\n-gold\n+gnew\n")
        assert "f.py: hunk declared line 1" in M.applied_where_it_said(repo, diff)

    def test_index_lines_before_the_first_hunk_do_not_stop_the_scan(self, tmp_path):
        """L1450 `continue` → `break`: a git `index` line ends the scan and the
        relocation below it is never checked."""
        repo = _repo(tmp_path, "\n".join(["pad"] * 5 + ["new1"]) + "\n")
        diff = ("diff --git a/f.py b/f.py\nindex 0000000..1111111 100644\n"
                "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old1\n+new1\n")
        assert "hunk declared line 1" in M.applied_where_it_said(repo, diff)

    def test_an_added_line_reading_plus_plus_is_content_of_the_post_image(self, tmp_path):
        """L1428-1431: a hunk that ADDS the text `++ /dev/null` (diff line
        `+++ /dev/null`, not preceded by `--- `) is part of the post-image."""
        repo = _repo(tmp_path, "++ /dev/null\nk\n")
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,2 @@\n+++ /dev/null\n k\n"
        assert M.applied_where_it_said(repo, diff) == ""
        repo2 = _repo(tmp_path / "b", "k\n")
        assert "declared line 1" in M.applied_where_it_said(repo2, diff)

    def test_a_correctly_placed_hunk_is_clean(self, tmp_path):
        repo = _repo(tmp_path, "new1\nk\n")
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old1\n+new1\n"
        assert M.applied_where_it_said(repo, diff) == ""


class TestRelocationObjectionEdges:
    def test_an_unreadable_patched_file_is_an_objection_not_silence(self, tmp_path):
        """L1403 `return f"{current} is unreadable after patching"` → None: the
        objection would read as 'no objection'. A directory where the file
        should be is unreadable."""
        repo = tmp_path / "snap"; (repo / "f.py").mkdir(parents=True)
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old1\n+new1\n"
        assert "unreadable after patching" in M.applied_where_it_said(repo, diff)

    def test_a_plus_plus_content_line_mid_hunk_does_not_end_the_post_image(self, tmp_path):
        """L1431 `continue` → `break`: the lines after an added `++ x` would be
        dropped from the post-image and a relocated tail would pass."""
        repo = _repo(tmp_path, "a\n++ x\nzzz\n")
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,3 @@\n+a\n+++ x\n+b\n"
        assert "declared line 1" in M.applied_where_it_said(repo, diff)      # file has zzz, not b
        repo2 = _repo(tmp_path / "ok", "a\n++ x\nb\n")
        assert M.applied_where_it_said(repo2, diff) == ""


def test_repair_hunk_counts_keeps_a_diff_without_a_trailing_newline_byte_identical():
    """L1342 `if diff.endswith("\\n")` → True: a correct diff that lacked the
    trailing newline would come back with one — §4CP's '31/31 correct diffs
    left byte-identical' rests on this."""
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old1\n+new1"
    out = M.repair_hunk_counts(diff)
    out = out[0] if isinstance(out, tuple) else out
    assert out == diff
