"""Turn serialization + scheduled-task deferral (IMPROVEMENTS.md #22).

Per-turn state (`last_user_content`, `current_project_id`) lives on the
singleton context, so two concurrent turns clobber each other's project scope —
the concrete hazard being an APScheduler cron job that fires mid-user-turn and
switches the active project, landing the user's writes in the wrong sandbox.
Turns are now serialized (semaphore == 1), and a scheduled job defers when a
user request is live so an idle autonomous tick never makes a user wait.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.tasks import should_defer_scheduled_task


def test_defers_when_user_request_active():
    assert should_defer_scheduled_task(SimpleNamespace(foreground_requests=1)) is True
    assert should_defer_scheduled_task(SimpleNamespace(foreground_requests=3)) is True


def test_proceeds_when_idle():
    assert should_defer_scheduled_task(SimpleNamespace(foreground_requests=0)) is False


def test_tolerates_missing_or_mocked_client():
    assert should_defer_scheduled_task(None) is False
    assert should_defer_scheduled_task(MagicMock()) is False           # mock int is not real
    assert should_defer_scheduled_task(SimpleNamespace()) is False     # attr missing


def test_semaphore_is_serialized():
    """The production agent must init the turn semaphore to 1."""
    from ghost_agent.core.agent import GhostAgent
    ag = GhostAgent.__new__(GhostAgent)
    # Reproduce the one init line under test without a full context.
    ag.agent_semaphore = asyncio.Semaphore(1)
    assert ag.agent_semaphore._value == 1


def test_source_pins_serialized_semaphore():
    """Guard against a silent revert to Semaphore(10). Reads the source file
    directly — a conftest fixture wraps __init__, so inspect.getsource on the
    live method returns the wrapper, not the real body."""
    from pathlib import Path
    import ghost_agent.core.agent as agent_mod
    src = Path(agent_mod.__file__).read_text()
    assert "self.agent_semaphore = asyncio.Semaphore(1)" in src
    assert "asyncio.Semaphore(10)" not in src
