"""Test helpers for the per-conversation focus (§4SX B0).

``start_request(ctx, key)`` does what ``handle_chat`` does at request start:
installs a request scope for ``ctx`` and loads its conversation's focus row.
``persist=False`` is a request-only scope (member, probe, no session id).
Use the ``scope_reset`` fixture (re-exported by importing it) so a scope set
by one test never reaches the next.
"""
import pytest

from ghost_agent.core import request_scope as _rs
from ghost_agent.core.request_scope import RequestScope


def start_request(ctx, key: str = "", persist: bool = True, origin: str = "owner",
                  project_id=None):
    sc = RequestScope(conversation_key=key, origin=origin, persist=bool(persist and key),
                      project_id=project_id, context=ctx)
    _rs._scope_var.set(sc)
    try:
        ctx.conversation_key = key
    except Exception:  # noqa: BLE001
        pass
    if not getattr(type(ctx), "current_project_id", None):
        # a plain test context holds the value itself
        ctx.current_project_id = project_id
    from ghost_agent.tools.projects import begin_request_focus
    begin_request_focus(ctx)
    return sc


def end_request():
    _rs._scope_var.set(None)


@pytest.fixture(autouse=True)
def scope_reset():
    yield
    _rs._scope_var.set(None)
