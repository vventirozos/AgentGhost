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


def _define(ctx, name="ping", steps=None, known_tools=None):
    return tool_manage_composed_skills(
        context=ctx, action="define", name=name, description="d",
        mode="sequential",
        steps=steps or [{"tool": "web_search", "params": {"query": "$q"}}],
        known_tools=known_tools if known_tools is not None else {"web_search"},
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
# Fix A — duplicate-define message leads with the execution path
# ─────────────────────────────────────────────────────────────────────────
class TestDuplicateDefineMessage:
    """The old "already exists … delete it first" wording steered a RUN-intent
    model into deleting an active macro it only wanted to invoke (the exact
    on-ramp to the guard deadlock). The message must now lead with how to USE
    the existing macro and demote delete to the replace-only path.

    PRODUCTION SHAPE (fresh-eye review, 2026-08-12): the registry call site
    derives known_tools from the fully-populated dispatch table, which
    register_composed_skill_runners has already seeded with every ACTIVE
    macro's own name — so the active-duplicate tests pass the macro's own
    name in known_tools. With the shadow check ordered before the duplicate
    check, the duplicate branch was unreachable in exactly that scenario
    (the refusal misdiagnosed the duplicate as a built-in collision)."""

    async def test_active_duplicate_points_at_run_before_delete(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        # Production shape: the active macro's runner is registered, so its
        # own name is in the call-time known_tools set.
        r = await _define(ctx, name="ping", known_tools={"web_search", "ping"})
        assert "already exists" in r
        assert "ACTIVE" in r
        assert "action='run'" in r
        assert "Do NOT re-define" in r
        # The duplicate diagnosis must WIN over the shadow check — the
        # registry's own macro is not "shadowing" a tool, it IS the tool.
        assert "Choose a different name" not in r
        # Delete guidance survives, but only AFTER the execution guidance and
        # framed as replacement — never as the first way out.
        assert "REPLACE" in r
        assert r.index("action='run'") < r.index("action='delete'")

    async def test_proposed_duplicate_points_at_approve_not_run(self, tmp_path):
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        _registry_from_context(ctx).skills["ping"].status = "proposed"
        # Proposed macros are never dispatchable, so their name is NOT in the
        # call-time known_tools — the default fixture IS the production shape.
        r = await _define(ctx, name="ping")
        assert "already exists" in r
        # A non-active macro can't be run — the message must not claim it can.
        assert "action='approve'" in r
        assert "action='run'" not in r
        assert "action='delete'" in r

    async def test_genuine_builtin_collision_still_shadow_refused(self, tmp_path):
        # A name that is a real built-in (NOT a registry macro) must still be
        # refused by the shadow check after the reorder.
        r = await _define(_FakeCtx(tmp_path), name="web_search",
                          known_tools={"web_search"})
        assert "Choose a different name" in r
        assert "already exists and is ACTIVE" not in r

    async def test_volatile_counters_outside_guard_prefix(self, tmp_path):
        """Steps/usage counters live at the message TAIL: two duplicate-define
        failures that differ only in usage_count must normalize to the SAME
        RecentFailureGuard key (first 80 lowercased chars), or for short macro
        names the repeat guard never accumulates an identical failure."""
        from ghost_agent.core.triggers import RecentFailureGuard
        ctx = _FakeCtx(tmp_path)
        await _define(ctx, name="ping")
        r1 = await _define(ctx, name="ping", known_tools={"web_search", "ping"})
        _registry_from_context(ctx).skills["ping"].usage_count = 37
        r2 = await _define(ctx, name="ping", known_tools={"web_search", "ping"})
        assert r1 != r2  # the counters really do differ...
        assert (RecentFailureGuard._norm_err(r1)
                == RecentFailureGuard._norm_err(r2))  # ...outside the key


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
