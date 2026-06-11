"""Root-cause fix for the `MagicMock/` directory leak.

`Path(MagicMock())` does not raise — it stringifies the mock repr into a
real relative path (`MagicMock/mock.memory_dir/<id>`), and
AcquiredSkillManager.__init__ then `mkdir`s that tree into the CWD. The
manager now rejects a mock base dir up front so the misconfiguration
fails at the call site instead of accreting junk directories.
"""

from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.acquired_skills import AcquiredSkillManager


def test_rejects_bare_magicmock_base(tmp_path):
    with pytest.raises(TypeError):
        AcquiredSkillManager(MagicMock(), MagicMock())


def test_rejects_magicmock_attribute_base():
    ctx = MagicMock()
    with pytest.raises(TypeError):
        # the exact footgun: a bare-mock context's .memory_dir
        AcquiredSkillManager(ctx.memory_dir, MagicMock())


def test_real_path_still_accepted(tmp_path):
    mgr = AcquiredSkillManager(tmp_path, MagicMock())
    assert (tmp_path / "acquired_skills").is_dir()


def test_no_magicmock_dir_created_on_rejection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError):
        AcquiredSkillManager(MagicMock().memory_dir, MagicMock())
    assert not (tmp_path / "MagicMock").exists()
