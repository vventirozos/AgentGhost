"""Conversation-scoped project binding.

Fix for cross-conversation project leakage: `context.current_project_id`
is process-global, so a project activated in one chat stayed active for
every other conversation hitting the same process — their file writes
landed in `<sandbox>/projects/<id>/` (observed in production: an SQL
snippet request's `migration.sql` was written into an unrelated
"Memory Garden" project, then several turns were burned undoing it).

Covers:
  - conversation_fingerprint identity semantics
  - _set_current records the owning conversation next to the id sentinel
  - reconcile_conversation: foreign conversation deactivates (binding
    preserved), owning conversation reactivates, legacy unbound sentinel
    never re-attaches, exit clears both sentinels
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import (
    tool_manage_projects,
    conversation_fingerprint,
    reconcile_conversation,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    sp = Scratchpad(persist_path=tmp_path / "sp.db")
    return SimpleNamespace(
        project_store=store,
        scratchpad=sp,
        graph_memory=None,
        current_project_id=None,
    )


CONV_A = conversation_fingerprint(
    [{"role": "user", "content": "let's build a memory garden"}])
CONV_B = conversation_fingerprint(
    [{"role": "user", "content": "I have the following table in postgres"}])


async def _create(context, title="Memory Garden"):
    res = await tool_manage_projects(context, action="create",
                                     title=title, kind="CODING", goal="x")
    return json.loads(res)["created"]


# ------------------------------------------------- conversation_fingerprint

def test_fingerprint_stable_as_conversation_grows():
    first = {"role": "user", "content": "hello world"}
    fp1 = conversation_fingerprint([first])
    fp2 = conversation_fingerprint([
        first,
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "another turn"},
    ])
    assert fp1 == fp2


def test_fingerprint_differs_across_conversations():
    assert CONV_A != CONV_B
    assert CONV_A and CONV_B


def test_fingerprint_ignores_leading_system_message():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"}]
    assert conversation_fingerprint(msgs) == conversation_fingerprint(
        [{"role": "user", "content": "hello world"}])


def test_fingerprint_multimodal_first_message():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]}]
    assert conversation_fingerprint(msgs) != ""


def test_fingerprint_empty_inputs():
    assert conversation_fingerprint([]) == ""
    assert conversation_fingerprint(None) == ""
    assert conversation_fingerprint(
        [{"role": "assistant", "content": "no user turn"}]) == ""


# ------------------------------------------------------- binding lifecycle

async def test_create_records_owning_conversation(context):
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    assert context.scratchpad.get("__current_project__") == pid
    assert context.scratchpad.get("__current_project_conv__") == CONV_A


async def test_foreign_conversation_does_not_inherit_project(context):
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    # a different conversation's request starts
    reconcile_conversation(context, CONV_B)
    assert context.current_project_id is None
    # ...but the binding is preserved for the owner
    assert context.scratchpad.get("__current_project__") == pid
    assert context.scratchpad.get("__current_project_conv__") == CONV_A


async def test_owning_conversation_reactivates_project(context):
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    reconcile_conversation(context, CONV_B)
    assert context.current_project_id is None
    reconcile_conversation(context, CONV_A)
    assert context.current_project_id == pid


async def test_legacy_unbound_sentinel_never_reattaches(context):
    """A bare `__current_project__` written before the conv sentinel
    existed must not capture ANY conversation — including the next one
    to arrive."""
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    context.scratchpad.delete("__current_project_conv__")
    context.current_project_id = pid
    reconcile_conversation(context, CONV_A)
    assert context.current_project_id is None


async def test_empty_conversation_key_owns_nothing(context):
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    reconcile_conversation(context, "")
    assert context.current_project_id is None
    assert context.scratchpad.get("__current_project__") == pid


async def test_exit_clears_both_sentinels(context):
    reconcile_conversation(context, CONV_A)
    await _create(context)
    await tool_manage_projects(context, action="exit")
    assert context.scratchpad.get("__current_project__") is None
    assert context.scratchpad.get("__current_project_conv__") is None


async def test_reconcile_without_scratchpad_is_noop(store):
    ctx = SimpleNamespace(project_store=store, scratchpad=None,
                          current_project_id=None)
    reconcile_conversation(ctx, CONV_A)  # must not raise
    assert ctx.conversation_key == CONV_A
