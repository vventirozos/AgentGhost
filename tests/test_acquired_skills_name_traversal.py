"""Security regression test for AcquiredSkillManager.save_skill.

Before the fix, `save_skill(name=..., python_code=...)` wrote the
Python code to ``<skills_dir>/<name>.py`` with NO name validation. A
name containing ``..`` (e.g. hallucinated by the LLM or injected via
a crafted prompt) escaped the skills_dir: ``save_skill(name="../../pwn")``
wrote to the sandbox's PARENT directory on disk.

Confirmed exploitable on 2026-04-24: a two-level `../..` traversal
wrote a file into the ephemeral tmpdir's parent. A more aggressive
traversal would land anywhere the Python process can write.

Fix: `_validate_skill_name` enforces an identifier-shape regex
(`[A-Za-z_][A-Za-z0-9_]{0,63}`) before any filesystem write, and
`save_skill` also defensively verifies the resolved path stays inside
`self.skills_dir` (belt + braces — any future regex widening that
reintroduces separators still hits the second guard).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.acquired_skills import (
    AcquiredSkillManager,
    SkillNameError,
    _validate_skill_name,
)


# ------------------------------------------------------------------
# Traversal is rejected
# ------------------------------------------------------------------

@pytest.mark.parametrize("bad_name", [
    "../evil",
    "../../evil",
    "../../../tmp/evil",
    "/absolute/path",
    "path/with/slash",
    "has spaces",
    "has.dot",
    "..",
    ".",
    "",
    None,
    "a/../b",
    "foo\\bar",   # windows-style separator
    "foo\x00bar", # NUL byte
    "foo\nbar",   # newline
    "42leading_digit",
    "-leading_dash",
    "skill!bang",
])
def test_unsafe_name_raises_validator(bad_name):
    with pytest.raises((SkillNameError, ValueError, TypeError)):
        _validate_skill_name(bad_name)


@pytest.mark.parametrize("bad_name", [
    "../evil",
    "../../evil",
    "../../../tmp/evil",
    "/absolute/escape",
    "a/../b",
])
def test_unsafe_name_never_writes_outside_sandbox(bad_name):
    """Exercise the full save_skill path — even without raising, the
    filesystem must remain untouched outside skills_dir."""
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td).resolve()
        mgr = AcquiredSkillManager(sandbox_dir=sandbox, memory_system=MagicMock())

        parent_before = {p.name for p in sandbox.parent.iterdir()}

        mgr.save_skill(
            name=bad_name, description="x",
            parameters_schema={"type": "object"},
            python_code="# payload",
        )

        parent_after = {p.name for p in sandbox.parent.iterdir()}
        escaped = parent_after - parent_before
        assert not escaped, (
            f"traversal write escaped to parent dir: {escaped}"
        )


def test_unsafe_name_not_registered():
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td).resolve()
        mgr = AcquiredSkillManager(sandbox_dir=sandbox, memory_system=MagicMock())
        mgr.save_skill(
            name="../injected_name", description="x",
            parameters_schema={"type": "object"},
            python_code="# payload",
        )
        registry = mgr.get_all_skills()
        assert "../injected_name" not in registry
        assert "injected_name" not in registry  # did NOT silently heal


# ------------------------------------------------------------------
# Valid names still work
# ------------------------------------------------------------------

@pytest.mark.parametrize("good_name", [
    "compute_tip",
    "CamelCase",
    "under_score_snake_case",
    "mixedCase123",
    "skill42",
    "_leading_underscore",
    "a",
    "A" * 64,  # boundary
])
def test_valid_name_accepted(good_name):
    assert _validate_skill_name(good_name) == good_name


def test_valid_name_registers_and_writes():
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td).resolve()
        mgr = AcquiredSkillManager(sandbox_dir=sandbox, memory_system=MagicMock())
        mgr.save_skill(
            name="compute_tip", description="tip calc",
            parameters_schema={"type": "object"},
            python_code="def compute_tip(): return 42",
        )
        assert "compute_tip" in mgr.get_all_skills()
        code_file = sandbox / "acquired_skills" / "compute_tip.py"
        assert code_file.exists()
        assert "compute_tip" in code_file.read_text()


def test_64_char_name_accepted():
    name = "a" * 64
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td).resolve()
        mgr = AcquiredSkillManager(sandbox_dir=sandbox, memory_system=MagicMock())
        mgr.save_skill(
            name=name, description="x",
            parameters_schema={"type": "object"},
            python_code="# ok",
        )
        assert name in mgr.get_all_skills()


def test_65_char_name_rejected():
    name = "a" * 65
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td).resolve()
        mgr = AcquiredSkillManager(sandbox_dir=sandbox, memory_system=MagicMock())
        mgr.save_skill(
            name=name, description="x",
            parameters_schema={"type": "object"},
            python_code="# ok",
        )
        assert name not in mgr.get_all_skills()


# ------------------------------------------------------------------
# Belt-and-braces: path-resolution check still catches regex widening
# ------------------------------------------------------------------

def test_path_resolution_guard_doc_is_present():
    """The belt-and-braces `skill_path.relative_to(skills_dir)` check
    guards future regex widening. Pin the comment so ripping it out
    requires updating this test too."""
    import inspect
    src = inspect.getsource(AcquiredSkillManager.save_skill)
    assert "relative_to" in src, (
        "belt-and-braces path resolve check missing — if the name regex "
        "ever loosens this is the second line of defence"
    )
    assert "escapes skills_dir" in src
