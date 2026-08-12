"""Composed-skill INVOCATION fixes (2026-08-12 postmortem).

Backstory: asked to *invoke* the active macro `youtube_transcribe`, the worker
model never called it by name — it reached for the MANAGEMENT tool
`manage_composed_skills(action='define')`, hit "already exists", struck out,
DELETED the macro to break the loop, then deadlocked because the pre-flight
guard kept blocking the re-`define` with the stale "already exists" error even
though the macro was gone.

Two fixes, pinned here:
  A. An explicit `action='run'` on manage_composed_skills that dispatches an
     existing macro through the SAME runner a direct `name(...)` call uses, so
     the model's instinct (reach for the management tool) actually works. Plus
     the advertised macro schema now marks runtime params REQUIRED and tells
     the model to call the macro directly.
  B. A successful REGISTRY-mutating `manage_composed_skills` action
     (define/approve/delete) counts as a world-change, so the guard's global
     reset fires and a delete unblocks a subsequently-legal define — no more
     deadlock. `run` and `list` are excluded: a macro can be entirely
     read-only, so a successful run must NOT blanket-clear the guard (that
     would let a model interleave a read-only run between failing calls and
     defeat the guard permanently — the read-only-probe hole the `execute`
     branch already guards against).
"""

import pytest

from ghost_agent.tools.composed_skills import (
    ComposedSkillRegistry, tool_manage_composed_skills, _registry_from_context,
)
from ghost_agent.core.agent import _call_mutated_world


class _FakeCtx:
    def __init__(self, base):
        self.memory_dir = base
        self.sandbox_dir = base


def _define(ctx, name="ping", steps=None):
    return tool_manage_composed_skills(
        context=ctx, action="define", name=name, description="d",
        mode="sequential",
        steps=steps or [{"tool": "web_search", "params": {"query": "$q"}}],
        known_tools={"web_search"},
    )


# ─────────────────────────────────────────────────────────────────────────
# Fix A — action='run'
# ─────────────────────────────────────────────────────────────────────────
class TestRunAction:
    async def test_run_requires_name(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="run")
        assert r.lower().startswith("error")
        assert "name" in r.lower()

    async def test_run_unknown_macro(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="run", name="ghost")
        assert "not found" in r.lower()

    async def test_run_inactive_macro_refused(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        # Flip the cached registry's copy to a proposed (inactive) draft.
        _registry_from_context(ctx).skills["ping"].status = "proposed"
        r = await tool_manage_composed_skills(context=ctx, action="run", name="ping")
        assert "not active" in r.lower()
        assert "approve" in r.lower()  # points at the recovery path

    async def test_run_bad_params_type(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        r = await tool_manage_composed_skills(
            context=ctx, action="run", name="ping", params="not-a-dict")
        assert r.lower().startswith("error")
        assert "params" in r.lower()

    async def test_run_dispatches_through_live_runner_with_params(
            self, tmp_path, monkeypatch):
        """`run` must call the SAME runner get_available_tools wires, passing
        the macro's runtime inputs through."""
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")

        seen = {}

        async def _stub_runner(**kwargs):
            seen.update(kwargs)
            return "RAN-OK"

        # Patch the registry-module symbol the lazy import resolves.
        import ghost_agent.tools.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_available_tools",
                            lambda _c: {"ping": _stub_runner})

        r = await tool_manage_composed_skills(
            context=ctx, action="run", name="ping", params={"q": "hello"})
        assert r == "RAN-OK"
        assert seen == {"q": "hello"}

    async def test_run_accepts_bare_kwargs_as_params(self, tmp_path, monkeypatch):
        """A weak model often passes inputs as top-level kwargs (q='x') rather
        than under params={...}; both must reach the macro."""
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")

        seen = {}

        async def _stub_runner(**kwargs):
            seen.update(kwargs)
            return "OK"

        import ghost_agent.tools.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_available_tools",
                            lambda _c: {"ping": _stub_runner})

        # No `params=` — q rides in **_extra.
        await tool_manage_composed_skills(
            context=ctx, action="run", name="ping", q="hello")
        assert seen == {"q": "hello"}

    async def test_run_explicit_params_win_over_bare_kwargs(
            self, tmp_path, monkeypatch):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        seen = {}

        async def _stub_runner(**kwargs):
            seen.update(kwargs)
            return "OK"

        import ghost_agent.tools.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_available_tools",
                            lambda _c: {"ping": _stub_runner})

        await tool_manage_composed_skills(
            context=ctx, action="run", name="ping",
            q="from_bare", params={"q": "from_params"})
        assert seen == {"q": "from_params"}

    async def test_run_no_runner_reports_shadow(self, tmp_path, monkeypatch):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        import ghost_agent.tools.registry as reg_mod
        # Registered+active but absent from the dispatch map ⇒ shadowed.
        monkeypatch.setattr(reg_mod, "get_available_tools", lambda _c: {})
        r = await tool_manage_composed_skills(context=ctx, action="run", name="ping")
        assert "not" in r.lower() and "dispatchable" in r.lower()

    async def test_unknown_action_message_lists_run(self, tmp_path):
        r = await tool_manage_composed_skills(
            context=_FakeCtx(tmp_path), action="frobnicate")
        assert "run" in r.lower()


# ─────────────────────────────────────────────────────────────────────────
# Fix A — advertised schema sharpening
# ─────────────────────────────────────────────────────────────────────────
class TestAdvertisedSchema:
    def _def_for(self, ctx, name):
        reg = _registry_from_context(ctx)
        for d in reg.to_tool_definitions():
            if d["function"]["name"] == name:
                return d["function"]
        return None

    async def test_runtime_params_marked_required(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping",
                      steps=[{"tool": "web_search", "params": {"query": "$q"}}])
        fn = self._def_for(ctx, "ping")
        assert fn is not None
        assert fn["parameters"]["required"] == ["q"]

    async def test_produced_values_not_required(self, tmp_path):
        """A value BOUND by an earlier step (save_as) is produced internally —
        it must not be advertised, let alone required."""
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="pipe", steps=[
            {"tool": "web_search", "params": {"query": "$q"}, "save_as": "hits"},
            {"tool": "web_search", "params": {"query": "$hits"}},
        ])
        fn = self._def_for(ctx, "pipe")
        assert set(fn["parameters"]["required"]) == {"q"}  # not "hits"

    async def test_description_steers_direct_call(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        fn = self._def_for(ctx, "ping")
        assert "CALL THIS TOOL DIRECTLY" in fn["description"]


# ─────────────────────────────────────────────────────────────────────────
# Fix B — world-changed reset covers manage_composed_skills
# ─────────────────────────────────────────────────────────────────────────
class TestWorldMutationPredicate:
    @pytest.mark.parametrize("action", ["define", "approve", "delete"])
    def test_mutating_composed_actions_reset_guard(self, action):
        assert _call_mutated_world(
            "manage_composed_skills", {"action": action}, False) is True

    @pytest.mark.parametrize("action", ["list", "run"])
    def test_readonly_composed_actions_do_not_reset_guard(self, action):
        # `run` is EXCLUDED on purpose: a macro can be entirely read-only, so a
        # successful run must not blanket-clear the failure guard (that would
        # let a model interleave a read-only macro between failing calls and
        # defeat the guard permanently). `list` is read-only too.
        assert _call_mutated_world(
            "manage_composed_skills", {"action": action}, False) is False

    def test_operation_alias_does_not_mutate_for_composed(self):
        # The tool reads only its named `action` param; an `operation='delete'`
        # (which the tool rejects) must NOT be scored a registry mutation.
        assert _call_mutated_world(
            "manage_composed_skills", {"operation": "delete"}, False) is False

    def test_empty_action_does_not_reset(self):
        assert _call_mutated_world(
            "manage_composed_skills", {}, False) is False

    # Behaviour-preserving refactor: the three pre-existing branches must still
    # decide exactly as before.
    def test_file_system_uses_is_mutating(self):
        assert _call_mutated_world("file_system", {"operation": "write"}, True) is True
        assert _call_mutated_world("file_system", {"operation": "read"}, False) is False

    def test_manage_services_readonly_ops_inert(self):
        assert _call_mutated_world("manage_services", {"action": "status"}, False) is False
        assert _call_mutated_world("manage_services", {"action": "start"}, False) is True

    def test_execute_uses_command_heuristic(self):
        assert _call_mutated_world("execute", {"command": "rm -rf build"}, True) is True
        assert _call_mutated_world("execute", {"command": "ls -la"}, True) is False
        # script-content execution (no command arg) never counts
        assert _call_mutated_world("execute", {}, True) is False

    def test_unknown_tool_never_resets(self):
        assert _call_mutated_world("web_search", {"query": "x"}, False) is False
