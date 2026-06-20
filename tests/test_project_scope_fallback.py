"""Project file-scoping resilience (Fix B).

`current_project_id` is process-global and can be cleared MID-REQUEST by a
concurrent conversation's reconcile. Observed live: a ~700s autoadvance +
manual build had its scoping wiped, so the agent's file writes landed in the
sandbox ROOT and it thrashed for minutes. `project_scoped_sandbox` now falls
back to THIS conversation's binding (which a concurrent conversation does not
stomp), so file ops stay scoped to the right project.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.tools.file_system import (
    project_scoped_sandbox, _conversation_bound_project,
    _PROJECT_BIND_PID, _PROJECT_BIND_CONV,
)


class _SP:
    """Minimal scratchpad stand-in exposing .get()."""
    def __init__(self, data):
        self.data = data

    def get(self, k, default=None):
        return self.data.get(k, default)


def _ctx(tmp_path, *, current, conv, bound_pid, bound_conv):
    return SimpleNamespace(
        sandbox_dir=tmp_path,
        current_project_id=current,
        conversation_key=conv,
        scratchpad=_SP({_PROJECT_BIND_PID: bound_pid, _PROJECT_BIND_CONV: bound_conv}),
    )


def test_scoping_uses_current_project_id_when_set(tmp_path):
    ctx = _ctx(tmp_path, current="abc123", conv="c1", bound_pid="abc123", bound_conv="c1")
    host, _ = project_scoped_sandbox(ctx)
    assert host == tmp_path / "projects" / "abc123"


def test_scoping_falls_back_to_binding_when_global_cleared(tmp_path):
    # current_project_id was wiped mid-request, but THIS conversation owns abc123
    ctx = _ctx(tmp_path, current=None, conv="c1", bound_pid="abc123", bound_conv="c1")
    host, workdir = project_scoped_sandbox(ctx)
    assert host == tmp_path / "projects" / "abc123"
    assert workdir.endswith("/projects/abc123")


def test_scoping_ignores_binding_owned_by_other_conversation(tmp_path):
    # the binding belongs to a DIFFERENT conversation → do not borrow it
    ctx = _ctx(tmp_path, current=None, conv="c1", bound_pid="abc123", bound_conv="other")
    host, workdir = project_scoped_sandbox(ctx)
    assert host == tmp_path          # sandbox root, not the other conv's project
    assert workdir is None


def test_scoping_no_binding_returns_root(tmp_path):
    ctx = _ctx(tmp_path, current=None, conv="c1", bound_pid=None, bound_conv=None)
    host, workdir = project_scoped_sandbox(ctx)
    assert host == tmp_path
    assert workdir is None


def test_stateful_never_scopes_even_with_binding(tmp_path):
    # stateful kernel sessions are pinned to /workspace — no fallback scoping
    ctx = _ctx(tmp_path, current=None, conv="c1", bound_pid="abc123", bound_conv="c1")
    host, workdir = project_scoped_sandbox(ctx, stateful=True)
    assert host == tmp_path
    assert workdir is None


def test_bound_helper_blank_when_no_conversation(tmp_path):
    ctx = _ctx(tmp_path, current=None, conv="", bound_pid="abc123", bound_conv="")
    assert _conversation_bound_project(ctx) == ""
