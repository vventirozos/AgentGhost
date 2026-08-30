"""Tests for autoadvance project-workspace pinning (2026-07-08).

Live failure: idle autoadvance built TinyAI's model.py/train.py/evaluate.py
at the sandbox ROOT — idle ticks carry no conversation, so the process-
global ``current_project_id`` was parked (None) and every file write fell
back to the root. The follow-up interactive demo task (after ``switch``)
saw only ``projects/<id>/`` and had to recreate the deliverable from
scratch, detached from the trained checkpoint.

Fix: tool runners for autoadvance are built from ``pinned_project_context``
— a proxy that pins ``current_project_id`` to the target project while
delegating everything else (reads AND writes) to the real context.
"""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from ghost_agent.core.project_advancer import pinned_project_context
from ghost_agent.tools.file_system import project_scoped_sandbox


class TestPinnedProjectContext:
    def test_pins_current_project_id(self):
        base = SimpleNamespace(current_project_id=None, foo="bar")
        ctx = pinned_project_context(base, "abc123")
        assert ctx.current_project_id == "abc123"
        # The base context's global is untouched — no race with the
        # conversation reconciler.
        assert base.current_project_id is None

    def test_reads_delegate_to_base(self):
        base = SimpleNamespace(current_project_id="other", foo="bar")
        ctx = pinned_project_context(base, "abc123")
        assert ctx.foo == "bar"

    def test_writes_delegate_to_base(self):
        base = SimpleNamespace(current_project_id=None)
        ctx = pinned_project_context(base, "abc123")
        ctx.some_counter = 7
        assert base.some_counter == 7

    def test_pin_survives_concurrent_reconcile_clearing_global(self):
        # The exact hazard: another conversation's reconcile clears the
        # global mid-build. The pinned view must not change.
        base = SimpleNamespace(current_project_id="abc123")
        ctx = pinned_project_context(base, "abc123")
        base.current_project_id = None  # concurrent reconcile parks it
        assert ctx.current_project_id == "abc123"

    def test_empty_project_id_returns_raw_context(self):
        base = SimpleNamespace(current_project_id=None)
        assert pinned_project_context(base, "") is base


class TestSandboxScopingThroughPin:
    def test_project_scoped_sandbox_lands_in_project_dir(self, tmp_path):
        # End-to-end through the real single-source-of-truth scoper: an
        # idle-shaped context (global parked, no conversation binding)
        # yields the ROOT without the pin, and projects/<id>/ with it.
        base = SimpleNamespace(
            sandbox_dir=tmp_path, current_project_id=None, scratchpad=None,
        )
        host, workdir = project_scoped_sandbox(base)
        assert Path(host) == tmp_path
        assert workdir is None

        pinned = pinned_project_context(base, "f36f04d446a6")
        host, workdir = project_scoped_sandbox(pinned)
        assert Path(host) == tmp_path / "projects" / "f36f04d446a6"
        assert Path(host).is_dir()  # created on demand
        assert workdir.endswith("/projects/f36f04d446a6")


class TestCallersArePinned:
    """Pin the wiring by source inspection: both autoadvance tool-runner
    construction sites must build tools from the pinned context."""

    # ⚠ Both of these were SUBSTRING pins and both were proven vacuous.
    #
    #  * the interactive one asserted `"pinned_project_context(context,
    #    project_id)" in src` over the whole module — satisfied by
    #    `_pinned = pinned_project_context(context, project_id)` followed by
    #    `get_available_tools(context)`. The literal survives; the value
    #    never reaches the consumer. It passed 1,515 tests.
    #  * the idle one sliced a 1,400-char window and asserted
    #    `"get_available_tools(_pin_ctx(ctx, pid))" in window` — and
    #    `inspect.getsource` returns COMMENTS, so commenting the call out and
    #    writing `get_available_tools(ctx)` beneath it satisfied both
    #    assertions. Also 1,515 tests.
    #
    # Parse the call instead. A comment is not in the AST, and an unused
    # assignment is not the call's argument.
    @staticmethod
    def _pinned_arg(mod, fn_name=None):
        """The argument `get_available_tools` is actually called with."""
        tree = ast.parse(inspect.getsource(mod))
        scope = tree
        if fn_name:
            scope = next((n for n in ast.walk(tree)
                          if isinstance(n, (ast.FunctionDef,
                                            ast.AsyncFunctionDef))
                          and n.name == fn_name), None)
            assert scope is not None, f"{fn_name} moved"
        calls = [n for n in ast.walk(scope)
                 if isinstance(n, ast.Call)
                 and ast.unparse(n.func).endswith("get_available_tools")]
        assert calls, f"no get_available_tools call in {fn_name or mod}"
        return [ast.unparse(c.args[0]) if c.args else "" for c in calls]

    def test_idle_tick_pins_target_project(self):
        import ghost_agent.core.agent as agent_mod

        args = self._pinned_arg(agent_mod, "_aa_tool_runner")
        assert args, "the idle tick's tool runner moved"
        for a in args:
            assert "_pin_ctx" in a or "pinned_project_context" in a, (
                "the idle autoadvancer builds its tool map from an UNPINNED "
                f"context ({a!r}) — idle ticks carry no conversation, so the "
                "process-global project id is parked at None and every file "
                "write lands at the sandbox ROOT")

    def test_interactive_autoadvance_pins_target_project(self):
        import ghost_agent.tools.projects as projects_mod

        args = self._pinned_arg(projects_mod)
        assert any("pinned_project_context" in a for a in args), (
            "the interactive autoadvance batch builds its tool map from an "
            f"unpinned context ({args!r}) — a concurrent conversation's "
            "reconcile can point the process-global id at a DIFFERENT "
            "project, and the batch then writes into its workspace")
