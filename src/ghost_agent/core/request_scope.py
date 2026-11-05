"""Per-request project scope (§4SX B0, UNIFIED_DESIGN §10.3).

One :class:`RequestScope` is built at request start and carried in a
ContextVar. ``GhostContext.current_project_id`` / ``conversation_key`` are
properties over it, so the ~110 readers keep working while the value stops
being process-global state: a task created during a request (late verdict,
spawn_bg, a gathered tool call) shares the request's scope object; the next
request installs a new one and never sees the old one's project.

The scope object is MUTABLE on purpose. A ``manage_projects switch`` that runs
inside a gathered/child task must change the request's project, and a child
task's ``ContextVar.set`` would not reach its parent — mutating the shared
holder does.

A scope applies to the context it was built for only (``scope.context is
self``). A copy (``copy.copy`` of a context for a leaf / sub-agent) gets the
value at copy time as its own plain value (``GhostContext.__copy__``), and a
leaf's ``handle_chat`` installs its own scope for the leaf context.

Who writes the per-conversation focus row (``conversation_focus`` in the
project store): only a scope with ``persist=True`` — an owner request that
carries a ``session_id``. Members, replays, probes, unattended requests
(sched-/sub-/job-), leaves and clients that send no session id get a
request-only scope: they may switch for the request, nothing is stored.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional

#: origins (who the request is)
ORIGIN_OWNER = "owner"
ORIGIN_MEMBER = "member"
ORIGIN_REPLAY = "replay"
ORIGIN_PROBE = "probe"
ORIGIN_UNATTENDED = "unattended"
ORIGIN_LEAF = "leaf"


@dataclass
class RequestScope:
    conversation_key: str = ""
    project_id: Optional[str] = None
    origin: str = ORIGIN_OWNER
    #: True only for an owner request with a session id: switch / create /
    #: resume / exit write this conversation's focus row
    persist: bool = False
    #: the context this scope applies to (identity, not equality)
    context: Any = field(default=None, repr=False, compare=False)


_scope_var: contextvars.ContextVar = contextvars.ContextVar("ghost_request_scope", default=None)


def current_scope(context: Any = None) -> Optional[RequestScope]:
    """The installed scope; with ``context``, only when it scopes that context."""
    s = _scope_var.get()
    if s is None:
        return None
    if context is not None and s.context is not context:
        return None
    return s


def install(scope: RequestScope):
    """Install ``scope`` for this task (and the tasks it creates); returns the
    token for :func:`reset`."""
    return _scope_var.set(scope)


def reset(token) -> None:
    try:
        _scope_var.reset(token)
    except (ValueError, RuntimeError):
        # a token from another Context (a cross-context finally): clear instead
        _scope_var.set(None)


def conversation_key_for_session(session_id: Any) -> str:
    """``session:<id>`` for a client session id; the Slack bot's ``slack:<…>``
    ids are already conversation keys. "" when there is none."""
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    if sid.startswith("slack:"):
        return sid
    return f"session:{sid}"
