"""Meta-test: pin the "no module-scope sys.modules patch" invariant.

`tests/test_database_tool.py` used to set ``sys.modules["tabulate"] =
MagicMock()`` at import time. Pytest would then load this file before
any test that needed a real `tabulate`, poisoning the module cache
for every subsequent test in the run. The failure mode was pytest
collection raising:

    ValueError: tabulate.__spec__ is not set

…in a completely unrelated test file (`test_vision_integration.py`,
`test_image_gen_integration.py`, etc.). The failure was **pytest-order
dependent**: with the "right" discovery order the full suite passed,
but any reordering (new tests, renamed files, a `-k` filter) flipped
it red. A real CI that parallelized tests or sharded them would have
tripped over this immediately.

Rule: **every sys.modules patch in this tests/ directory MUST be
scoped to a function or fixture using `monkeypatch.setitem` or
`unittest.mock.patch.dict(sys.modules, ...)`.** A bare
``sys.modules["x"] = ...`` at module scope is forbidden — that
permanently rewrites the interpreter's import cache for the rest
of the pytest process.

This test grep-asserts the invariant so a future contributor can't
silently reintroduce the bug.
"""

import ast
import re
from pathlib import Path

import pytest


_TESTS_DIR = Path(__file__).resolve().parent


def _module_scope_sys_modules_writes(py_path: Path) -> list[tuple[int, str]]:
    """Return (line_no, line_text) for every module-scope sys.modules
    write in `py_path`. An assignment inside a function or class is
    fine; only top-level ones pollute the import cache for downstream
    tests.
    """
    try:
        src = py_path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return []

    findings: list[tuple[int, str]] = []
    lines = src.splitlines()

    for node in tree.body:  # module-body nodes only — function bodies ignored
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        for t in targets:
            # Match `sys.modules["x"] = ...` and `sys.modules.update({...})`
            if isinstance(t, ast.Subscript):
                # sys.modules[key] = value
                if (
                    isinstance(t.value, ast.Attribute)
                    and t.value.attr == "modules"
                    and isinstance(t.value.value, ast.Name)
                    and t.value.value.id == "sys"
                ):
                    line_no = node.lineno
                    findings.append((line_no, lines[line_no - 1] if line_no <= len(lines) else ""))

    # Also catch `sys.modules.update({...})` and `sys.modules.pop(...)`
    # at module scope — same pollution risk.
    for node in tree.body:
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in {"update", "pop", "clear", "setdefault"}:
            continue
        if (
            isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "modules"
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "sys"
        ):
            line_no = node.lineno
            findings.append((line_no, lines[line_no - 1] if line_no <= len(lines) else ""))

    return findings


def test_no_module_scope_sys_modules_writes_in_tests():
    bad: dict[str, list[tuple[int, str]]] = {}
    for py_path in _TESTS_DIR.glob("test_*.py"):
        if py_path.name == Path(__file__).name:
            continue  # skip the meta-test itself
        findings = _module_scope_sys_modules_writes(py_path)
        if findings:
            bad[py_path.name] = findings

    if bad:
        lines = ["Module-scope sys.modules writes found (they poison subsequent tests):"]
        for fname, hits in sorted(bad.items()):
            for line_no, text in hits:
                lines.append(f"  {fname}:{line_no}: {text.strip()}")
        lines.append("")
        lines.append(
            "Fix: move the patch inside a test function or fixture using "
            "`monkeypatch.setitem(sys.modules, ...)` or "
            "`patch.dict(sys.modules, {...})` as a context manager."
        )
        pytest.fail("\n".join(lines))


def test_meta_test_detector_flags_module_scope_writes(tmp_path):
    """Sanity: the detector actually catches the pattern it claims to."""
    sample = tmp_path / "test_example.py"
    sample.write_text(
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "\n"
        "sys.modules['tabulate'] = MagicMock()  # BAD: module scope\n"
        "\n"
        "def test_something():\n"
        "    sys.modules['fine'] = MagicMock()  # tolerated: inside a function\n"
    )
    findings = _module_scope_sys_modules_writes(sample)
    # Exactly one hit — the module-scope assignment on line 4.
    assert len(findings) == 1
    assert findings[0][0] == 4


def test_meta_test_detector_ignores_function_scope_writes(tmp_path):
    sample = tmp_path / "test_ok.py"
    sample.write_text(
        "import sys\n"
        "from unittest.mock import MagicMock, patch\n"
        "\n"
        "def test_ok():\n"
        "    with patch.dict(sys.modules, {'x': MagicMock()}):\n"
        "        sys.modules['y'] = MagicMock()\n"
    )
    findings = _module_scope_sys_modules_writes(sample)
    assert findings == []


def test_meta_test_detector_catches_update_and_pop(tmp_path):
    sample = tmp_path / "test_mixed.py"
    sample.write_text(
        "import sys\n"
        "\n"
        "sys.modules.update({'a': None})  # BAD\n"
        "sys.modules.pop('x', None)       # BAD\n"
    )
    findings = _module_scope_sys_modules_writes(sample)
    assert len(findings) == 2
