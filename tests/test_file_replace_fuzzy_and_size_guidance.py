"""Tests for file_replace fuzzy fallback + size-aware failure guidance.

Regression target: a one-line edit (`spawn` → `spawnPos`, `RENDER_DIST
4` → `2`) on a large index.html repeatedly missed the exact/flexible
matcher, and the tool's guidance told the model to "overwrite the whole
file" — a 200s+ regeneration that also reintroduced bugs. Two fixes:
  1. A conservative difflib fuzzy fallback rescues a near-unique block
     so a tiny edit lands surgically instead of forcing a rewrite.
  2. The replace-miss guidance is size-aware: large files must NOT be
     wholesale-rewritten; the model is steered to a tight single-line
     replace anchored on the exact current text.
"""
import pytest

from ghost_agent.tools.file_system import (
    _fuzzy_block_match,
    tool_replace_text,
)


@pytest.fixture
def sandbox_dir(tmp_path):
    return tmp_path


# ── _fuzzy_block_match unit behaviour ────────────────────────────────
def test_fuzzy_matches_single_char_drift():
    content = "alpha\nlet spawnPoint = new Vector3(0, 30, 0);\nbeta\n"
    # capitalisation drift the flexible (token-exact) matcher can't bridge
    target = "let spawnpoint = new Vector3(0, 30, 0);"
    res = _fuzzy_block_match(content, target)
    assert res is not None
    matched, ratio = res
    assert "spawnPoint" in matched and ratio >= 0.92


def test_fuzzy_refuses_ambiguous_match():
    # two near-identical candidate lines → refuse to guess
    content = "x = computeValue(1)\n# sep\nx = computeValue(2)\n"
    target = "x = computeValue(3)"
    assert _fuzzy_block_match(content, target) is None


def test_fuzzy_refuses_when_nothing_close():
    content = "completely\nunrelated\nlines\n"
    assert _fuzzy_block_match(content, "def quantum_entangle(state):") is None


def test_fuzzy_refuses_all_whitespace_target():
    assert _fuzzy_block_match("a\nb\nc\n", "   \n  ") is None


# ── tool_replace_text integration ────────────────────────────────────
@pytest.mark.asyncio
async def test_replace_uses_fuzzy_when_exact_and_flexible_miss(sandbox_dir):
    f = sandbox_dir / "game.js"
    f.write_text("const A = 1;\nlet spawnPoint = new V3(0, 30, 0);\nconst B = 2;\n")
    # old_text has a single-character typo → exact + flexible both miss
    res = await tool_replace_text(
        "game.js",
        "let spawnPont = new V3(0, 30, 0);",
        "let spawnPoint = new V3(0, 40, 0);",
        sandbox_dir,
    )
    assert "SUCCESS" in res and "Fuzzy" in res
    body = f.read_text()
    assert "0, 40, 0" in body
    # the surrounding lines survive intact — no welding / corruption
    assert "const A = 1;" in body and "const B = 2;" in body
    assert body.count("\n") == 3  # newline structure preserved


@pytest.mark.asyncio
async def test_large_file_miss_guidance_forbids_full_rewrite(sandbox_dir):
    f = sandbox_dir / "index.html"
    # > 250 lines → "large" branch
    f.write_text("<html>\n" + "\n".join(f"  <div>row {i}</div>" for i in range(400)) + "\n</html>\n")
    res = await tool_replace_text(
        "index.html", "THIS_BLOCK_DOES_NOT_EXIST_ANYWHERE_42", "x", sandbox_dir
    )
    assert "NOT found" in res
    assert "LARGE" in res
    assert "DO NOT use operation='write'" in res
    # must NOT tell it to overwrite the whole file
    assert "overwrite the whole file" not in res


@pytest.mark.asyncio
async def test_small_file_miss_guidance_allows_write(sandbox_dir):
    f = sandbox_dir / "tiny.py"
    f.write_text("a = 1\nb = 2\n")
    res = await tool_replace_text("tiny.py", "z = 99", "q = 0", sandbox_dir)
    assert "NOT found" in res
    # small file: a full write is still an acceptable suggestion
    assert "overwrite this small file" in res
    assert "LARGE" not in res


@pytest.mark.asyncio
async def test_exact_match_still_preferred_over_fuzzy(sandbox_dir):
    f = sandbox_dir / "c.txt"
    f.write_text("keep\nexact target line\nkeep2\n")
    res = await tool_replace_text("c.txt", "exact target line", "new line", sandbox_dir)
    assert "Exact match" in res
    assert "new line" in f.read_text()
