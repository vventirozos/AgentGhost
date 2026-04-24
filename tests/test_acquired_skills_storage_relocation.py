"""Regression tests for acquired-skills storage relocation.

Before: ``AcquiredSkillManager`` saved skills under
``<sandbox_dir>/acquired_skills/``. That's inside the Docker bind-mount
— a ``docker volume rm`` or a stray ``rm -rf $GHOST_SANDBOX_DIR``
would destroy every learned skill.

After: skills live under ``<memory_dir>/acquired_skills/`` (resolving
to ``$GHOST_HOME/system/memory/acquired_skills/`` when wired from
main.py). Execution still happens inside the sandbox — the
registry's skill-runner closure reads the canonical file and passes
``content=`` to ``tool_execute``, so the "all code runs sandboxed"
invariant is preserved.

These tests pin:
  1. The manager stores at ``base_dir/acquired_skills/``; swapping
     sandbox_dir → memory_dir just changes WHERE it lands, the class
     doesn't care.
  2. One-time migration: if a legacy ``sandbox_dir/acquired_skills/``
     exists and the new location is empty, the manager copies skill
     files + registry over. Idempotent — a second construction finds
     the new location populated and does nothing.
  3. Execution path reads from memory_dir, not sandbox_dir. Deleting
     the sandbox copy doesn't affect skill availability.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.tools.acquired_skills import (
    AcquiredSkillManager,
    tool_create_skill,
)


# ---------------------------------------------------------------------------
# 1. Storage location follows the base_dir argument
# ---------------------------------------------------------------------------


def test_manager_writes_under_base_dir_acquired_skills(tmp_path):
    """Canonical contract: `<base>/acquired_skills/<name>.py`."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    mgr = AcquiredSkillManager(mem_dir, memory_system=None)
    mgr.save_skill(
        "greet", "say hi", {"type": "object", "properties": {}},
        "print('hi')\n",
    )
    assert (mem_dir / "acquired_skills" / "greet.py").is_file()
    # Registry lands in the same dir.
    assert (mem_dir / "acquired_skills" / "skills_registry.json").is_file()


def test_manager_does_not_write_outside_base_dir(tmp_path):
    """Negative: if memory_dir is memory/, nothing must land in
    sandbox/ accidentally."""
    mem_dir = tmp_path / "memory"
    sandbox_dir = tmp_path / "sandbox"
    mem_dir.mkdir()
    sandbox_dir.mkdir()
    mgr = AcquiredSkillManager(mem_dir, memory_system=None)
    mgr.save_skill(
        "noop", "do nothing", {"type": "object", "properties": {}},
        "pass\n",
    )
    assert (mem_dir / "acquired_skills" / "noop.py").is_file()
    # Sandbox must stay clean — no phantom acquired_skills/ dir.
    assert not (sandbox_dir / "acquired_skills").exists()


# ---------------------------------------------------------------------------
# 2. Legacy sandbox → memory migration
# ---------------------------------------------------------------------------


def _seed_legacy_sandbox(sandbox_dir: Path, skill_name: str, source: str,
                        schema: dict = None, description: str = "legacy"):
    """Mimic a pre-relocation on-disk layout so we can exercise the
    migration path end-to-end."""
    sandbox_skills = sandbox_dir / "acquired_skills"
    sandbox_skills.mkdir(parents=True, exist_ok=True)
    (sandbox_skills / f"{skill_name}.py").write_text(source, encoding="utf-8")
    reg = {
        skill_name: {
            "name": skill_name,
            "description": description,
            "parameters_schema": schema or {"type": "object", "properties": {}},
            "usage_count": 3,
            "status": "active",
            "content_hash": "abc",
        }
    }
    (sandbox_skills / "skills_registry.json").write_text(
        json.dumps(reg, indent=2), encoding="utf-8"
    )


def test_migration_copies_legacy_sandbox_skills_to_memory(tmp_path):
    sandbox_dir = tmp_path / "sandbox"
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    _seed_legacy_sandbox(sandbox_dir, "pwgen", "print('pw')\n")

    mgr = AcquiredSkillManager(
        mem_dir, memory_system=None, legacy_sandbox_dir=sandbox_dir,
    )

    # File + registry entry now in the canonical memory location.
    assert (mem_dir / "acquired_skills" / "pwgen.py").is_file()
    assert (mem_dir / "acquired_skills" / "pwgen.py").read_text() == "print('pw')\n"
    all_skills = mgr.get_all_skills()
    assert "pwgen" in all_skills
    # Preserves description / usage count from the legacy registry.
    assert all_skills["pwgen"]["description"] == "legacy"
    assert all_skills["pwgen"]["usage_count"] == 3


def test_migration_does_not_clobber_populated_memory_store(tmp_path):
    """If the canonical store already has skills, the migration is a
    no-op — we never overwrite newer data."""
    sandbox_dir = tmp_path / "sandbox"
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    # Pre-populate memory_dir with a skill.
    mgr1 = AcquiredSkillManager(mem_dir, memory_system=None)
    mgr1.save_skill(
        "canonical", "new-path owner", {"type": "object", "properties": {}},
        "print('new')\n",
    )

    # Seed a DIFFERENT skill into the legacy sandbox location.
    _seed_legacy_sandbox(sandbox_dir, "legacy_only", "print('legacy')\n")

    # Reconstruct with legacy pointer. Must NOT pull legacy in, because
    # the canonical store is non-empty.
    mgr2 = AcquiredSkillManager(
        mem_dir, memory_system=None, legacy_sandbox_dir=sandbox_dir,
    )
    skills = mgr2.get_all_skills()
    assert "canonical" in skills
    assert "legacy_only" not in skills, (
        "must not clobber a populated canonical store"
    )


def test_migration_is_idempotent(tmp_path):
    """Constructing the manager repeatedly with the same legacy
    pointer must not duplicate or corrupt the migrated data."""
    sandbox_dir = tmp_path / "sandbox"
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    _seed_legacy_sandbox(sandbox_dir, "gen", "print('gen')\n")

    for _ in range(3):
        mgr = AcquiredSkillManager(
            mem_dir, memory_system=None, legacy_sandbox_dir=sandbox_dir,
        )

    # One skill, one file, one registry entry.
    skill_files = list((mem_dir / "acquired_skills").glob("*.py"))
    assert len(skill_files) == 1
    assert skill_files[0].name == "gen.py"
    assert list(mgr.get_all_skills().keys()) == ["gen"]


def test_migration_skips_legacy_entries_with_unsafe_names(tmp_path):
    """Defense-in-depth: a malformed legacy registry containing a
    traversal-shaped name must NOT be migrated (the name validator
    would refuse it on save). The rest of the legacy batch should
    still migrate."""
    sandbox_dir = tmp_path / "sandbox"
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    legacy_skills = sandbox_dir / "acquired_skills"
    legacy_skills.mkdir(parents=True)
    (legacy_skills / "good.py").write_text("print('g')\n", encoding="utf-8")
    reg = {
        "good": {"name": "good", "description": "", "parameters_schema": {},
                 "status": "active"},
        "../evil": {"name": "../evil", "description": "", "parameters_schema": {},
                    "status": "active"},
    }
    (legacy_skills / "skills_registry.json").write_text(
        json.dumps(reg), encoding="utf-8"
    )

    mgr = AcquiredSkillManager(
        mem_dir, memory_system=None, legacy_sandbox_dir=sandbox_dir,
    )
    skills = mgr.get_all_skills()
    assert "good" in skills
    assert "../evil" not in skills
    # And no file escaped.
    assert not (mem_dir.parent / "evil.py").exists()


def test_migration_is_noop_when_legacy_dir_missing(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    sandbox_dir = tmp_path / "nonexistent_sandbox"
    mgr = AcquiredSkillManager(
        mem_dir, memory_system=None, legacy_sandbox_dir=sandbox_dir,
    )
    assert mgr.get_all_skills() == {}


def test_migration_is_noop_when_base_equals_legacy(tmp_path):
    """If someone passes the same dir as base_dir and legacy — a
    degenerate case — nothing should break."""
    same = tmp_path / "same"
    same.mkdir()
    mgr = AcquiredSkillManager(
        same, memory_system=None, legacy_sandbox_dir=same,
    )
    assert mgr.get_all_skills() == {}


# ---------------------------------------------------------------------------
# 3. tool_create_skill routes canonical save to memory_dir
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_create_skill_saves_to_memory_dir_not_sandbox(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    mem = tmp_path / "memory"
    sandbox.mkdir()
    mem.mkdir()

    # Stub tool_execute so we don't need a real container.
    from ghost_agent.tools import acquired_skills as mod

    async def fake_execute(**kwargs):
        return "EXIT CODE: 0\n\nHello"

    monkeypatch.setattr(mod, "tool_execute", fake_execute, raising=False)
    # Route the import inside tool_create_skill to the stub too.
    import ghost_agent.tools.execute as execute_mod
    monkeypatch.setattr(execute_mod, "tool_execute", fake_execute)

    result = await tool_create_skill(
        sandbox_dir=sandbox,
        memory_dir=mem,
        memory_system=None,
        sandbox_manager=MagicMock(),
        name="greet",
        description="say hi",
        parameters_schema='{"type": "object", "properties": {}}',
        python_code="print('hi')\n",
        test_payload="{}",
    )
    assert "Success" in result
    # Canonical file under memory_dir.
    assert (mem / "acquired_skills" / "greet.py").is_file()
    # NOT under sandbox (sandbox held only the transient test_skill.py,
    # which the tool also unlinks on pass).
    assert not (sandbox / "acquired_skills" / "greet.py").exists()
    # The transient test file is cleaned up.
    assert not (sandbox / "test_skill.py").exists()


@pytest.mark.asyncio
async def test_tool_create_skill_normalises_cdata_wrapped_python(tmp_path, monkeypatch):
    """The 2026-04-24 in_gr_news failure: the LLM sent python_code
    wrapped in `<![CDATA[...]]>`. Without the entry-point
    normalization, CDATA landed in test_skill.py verbatim and the
    parser rejected line 1. With the fix, the envelope is stripped
    and the normalized body is what hits disk (and later hits the
    canonical storage)."""
    sandbox = tmp_path / "sandbox"
    mem = tmp_path / "memory"
    sandbox.mkdir()
    mem.mkdir()

    from ghost_agent.tools import acquired_skills as mod
    import ghost_agent.tools.execute as execute_mod

    # Capture what actually lands in test_skill.py.
    captured = {"written": None}

    async def fake_execute(**kwargs):
        # Read the file that tool_create_skill just wrote so we can
        # pin that it's CDATA-clean.
        sd = kwargs.get("sandbox_dir")
        if sd:
            captured["written"] = (Path(sd) / "test_skill.py").read_text()
        return "EXIT CODE: 0\n\nhello"

    monkeypatch.setattr(mod, "tool_execute", fake_execute, raising=False)
    monkeypatch.setattr(execute_mod, "tool_execute", fake_execute)

    wrapped_source = (
        "<![CDATA[\n"
        "#!/usr/bin/env python3\n"
        '"""Fetch headlines from in.gr."""\n'
        "import sys, json\n"
        "def main(args): return 'ok'\n"
        "if __name__ == '__main__':\n"
        "    print(main(json.loads(sys.argv[1])))\n"
        "]]>"
    )

    result = await tool_create_skill(
        sandbox_dir=sandbox,
        memory_dir=mem,
        memory_system=None,
        sandbox_manager=MagicMock(),
        name="in_gr_news",
        description="fetch headlines",
        parameters_schema='{"type":"object","properties":{}}',
        python_code=wrapped_source,
        test_payload="{}",
    )

    assert "Success" in result
    # CDATA markers never reached test_skill.py.
    assert "<![CDATA[" not in captured["written"]
    assert "]]>" not in captured["written"]
    # Body survived.
    assert "def main(args)" in captured["written"]
    # Canonical save is also clean.
    canonical = (mem / "acquired_skills" / "in_gr_news.py").read_text()
    assert "<![CDATA[" not in canonical
    assert "def main(args)" in canonical


@pytest.mark.asyncio
async def test_tool_create_skill_rejects_unparseable_python_with_actionable_error(tmp_path, monkeypatch):
    """When the payload is broken beyond repair, we don't silently
    write a corrupt test_skill.py — we return an error to the LLM
    that names the common causes so the retry is pointed."""
    sandbox = tmp_path / "sandbox"
    mem = tmp_path / "memory"
    sandbox.mkdir()
    mem.mkdir()

    # tool_execute must NOT be reached.
    from ghost_agent.tools import acquired_skills as mod
    import ghost_agent.tools.execute as execute_mod

    called = {"n": 0}

    async def fake_execute(**kwargs):
        called["n"] += 1
        return "EXIT CODE: 0\n\nhello"

    monkeypatch.setattr(mod, "tool_execute", fake_execute, raising=False)
    monkeypatch.setattr(execute_mod, "tool_execute", fake_execute)

    # Neither CDATA-envelope-strip nor HTML-entity-rescue nor markdown
    # extraction can recover this: just garbage.
    broken = "this is not valid python at all !!!\nbroken {syntax"

    result = await tool_create_skill(
        sandbox_dir=sandbox,
        memory_dir=mem,
        memory_system=None,
        sandbox_manager=MagicMock(),
        name="broken",
        description="x",
        parameters_schema='{"type":"object","properties":{}}',
        python_code=broken,
        test_payload="{}",
    )

    assert "python_code didn't parse as valid Python" in result
    # The message names common remedies so the LLM's retry is focused.
    assert "CDATA" in result or "wrapper" in result
    # tool_execute was never reached — we refused before writing.
    assert called["n"] == 0
    # And no test_skill.py got written.
    assert not (sandbox / "test_skill.py").exists()


@pytest.mark.asyncio
async def test_tool_create_skill_falls_back_to_sandbox_when_memory_dir_missing(tmp_path, monkeypatch):
    """Legacy path for callers that never threaded memory_dir. The
    final save still works — it just lands in sandbox as before,
    which the migration path will pick up on next boot."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    from ghost_agent.tools import acquired_skills as mod
    import ghost_agent.tools.execute as execute_mod

    async def fake_execute(**kwargs):
        return "EXIT CODE: 0\n\nok"

    monkeypatch.setattr(mod, "tool_execute", fake_execute, raising=False)
    monkeypatch.setattr(execute_mod, "tool_execute", fake_execute)

    result = await tool_create_skill(
        sandbox_dir=sandbox,
        memory_dir=None,  # legacy caller
        memory_system=None,
        sandbox_manager=MagicMock(),
        name="legacy_caller",
        description="x",
        parameters_schema='{"type":"object","properties":{}}',
        python_code="print('x')\n",
        test_payload="{}",
    )
    assert "Success" in result
    assert (sandbox / "acquired_skills" / "legacy_caller.py").is_file()
