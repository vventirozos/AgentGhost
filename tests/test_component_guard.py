"""The real-component guard must tolerate BOTH import shapes.

Production launches `python -m src.ghost_agent.main` from the repo root, so
modules are `src.ghost_agent.*`. The test suite runs with `PYTHONPATH=src` and
gets `ghost_agent.*`. A guard written as
`type(obj).__module__.startswith("ghost_agent")` is therefore always FALSE in
production and always TRUE under test — which silently disabled five idle
subsystems for weeks while their tests stayed green.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from ghost_agent.utils.component_guard import _is_real_component


def _obj_from_module(module_name: str):
    cls = type("Thing", (), {})
    cls.__module__ = module_name
    return cls()


def test_accepts_the_test_import_shape():
    assert _is_real_component(_obj_from_module("ghost_agent.memory.skills"))


def test_accepts_the_PRODUCTION_import_shape():
    """The whole point. This is the shape `bin/start-ghost-agent.sh` produces."""
    assert _is_real_component(_obj_from_module("src.ghost_agent.memory.skills"))


def test_rejects_test_doubles():
    assert not _is_real_component(MagicMock())
    assert not _is_real_component(_obj_from_module("unittest.mock"))
    assert not _is_real_component(_obj_from_module("builtins"))
    assert not _is_real_component(object())


def test_no_ghost_agent_module_guard_survives_in_the_tree():
    """Regression fence: any NEW `__module__.startswith("ghost_agent")` is the
    same production-dead bug. (The chromadb/unittest variants are unaffected —
    those packages import under one name.)"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "component_guard.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if '__module__.startswith("ghost_agent")' in line:
                offenders.append(f"{path.relative_to(root)}:{i}")
    assert not offenders, (
        "use utils.component_guard._is_real_component instead: " + ", ".join(offenders))
