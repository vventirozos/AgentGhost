"""Turn-loop review, Round 4 (§ turn-loop R4, 2026-08-19) — behavioral pins.

R4's converging findings: the static `_PURE_TRIGGER_TOOLS` was blind to
RUNTIME-registered tools (a dynamic no-arg macro's native call lost to an
echoed tag — the recurring proxy class, fixed via `_STATIC_TOOL_NAMES`), one
unadvertised kb action dodged `is_mutating`, and four R3-diff sites were
unpinned. All pins drive the REAL methods; each mutation-verified.
"""

import argparse
import json

import pytest

from ghost_agent.core.agent import (GhostAgent, TurnState,
                                    _PURE_TRIGGER_TOOLS, _STATIC_TOOL_NAMES)
from ghost_agent.core.strikes import StrikeLedger
from unittest.mock import AsyncMock, MagicMock


def _parser_agent(names=("dream_mode", "web_search", "recall", "execute",
                         "morning_briefing")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in names}
    return agent


def _names(tcs):
    return [(t.get("function") or {}).get("name") for t in tcs]


def _args0(tcs):
    x = tcs[0]["function"]["arguments"]
    return json.loads(x) if isinstance(x, str) else x


_ECHO = ('I will call the <function name="web_search">'
         '<parameter name="query">weather</parameter></function> helper.')
_XML_RECALL = ('<tool_call><function name="recall">'
               '<parameter name="query">real</parameter></function></tool_call>')


class TestDynamicNoArgToolsUsable:
    """MAJOR (lens A+C): a runtime-registered tool (composed macro / acquired
    skill) is never in the static trigger set, but its empty-args native call
    must still win over an echoed tag — dispatching it errors recoverably at
    worst; executing an echo (possibly mutating) is unrecoverable."""

    def test_static_names_set_is_populated(self):
        assert "execute" in _STATIC_TOOL_NAMES
        assert "web_search" in _STATIC_TOOL_NAMES
        # a runtime macro name is, by construction, not static
        assert "morning_briefing" not in _STATIC_TOOL_NAMES

    def test_dynamic_no_arg_macro_wins_over_echoed_tag(self):
        agent = _parser_agent()
        msg = {"content": _ECHO, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "morning_briefing", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_ECHO, msg)
        assert _names(tcs) == ["morning_briefing"]

    def test_static_non_trigger_empty_args_still_yields_to_xml(self):
        # control: web_search is STATIC and takes params — empty native args
        # must still yield to the rich XML call (R3 semantics preserved).
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]


class TestUsableNativeGateEdgePins:
    """Lens C-c: the R3-diff sites that survived neutering."""

    def test_null_string_args_trigger_still_usable(self):
        # the '"null"' fast-path must classify as empty (→ trigger usable),
        # not as "has args" nor as unparseable.
        agent = _parser_agent()
        msg = {"content": _ECHO, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "dream_mode", "arguments": "null"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_ECHO, msg)
        assert _names(tcs) == ["dream_mode"]

    def test_list_literal_string_args_trigger_still_usable(self):
        agent = _parser_agent()
        msg = {"content": _ECHO, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "dream_mode", "arguments": "[]"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_ECHO, msg)
        assert _names(tcs) == ["dream_mode"]

    def test_scalar_args_are_degenerate(self):
        # arguments=42 (non-str, non-dict, non-None) → degenerate → XML wins.
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": 42}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]

    def test_padded_raw_json_is_not_healed(self):
        # `_looks_raw_json` strips leading whitespace: a padded raw-JSON call
        # whose command contains a real fn-tag literal must recover unmangled.
        agent = _parser_agent(("execute",))
        cmd = "grep -n '<function name=' x.py"
        content = "\n  " + json.dumps({"name": "execute",
                                       "arguments": {"command": cmd}})
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["execute"]
        assert _args0(tcs)["command"] == cmd


# ── dispatch harness ─────────────────────────────────────────────────────────
def _dispatch_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args = argparse.Namespace(smart_memory=0.0, enable_preflight_guard=True)
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    agent.disabled_tools = set()
    return agent


def _ts(**over):
    fields = dict(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=True, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0, tool_calls=[],
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content="do the thing", char_budget=4000,
        strikes=StrikeLedger(), task_tree=MagicMock(), _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=set(), messages=[],
        seen_tools=set(), executed_idempotent=set(), raw_tools_called=set(),
        tool_usage={}, tools_run_this_turn=[], request_state=MagicMock(),
    )
    fields.update(over)
    return TurnState(**fields)


def _ok_tool():
    async def _t(**kwargs):
        return "ok"
    return _t


async def _run_dup(tool_name, args):
    agent = _dispatch_agent()
    calls = {"n": 0}

    async def t(**kwargs):
        calls["n"] += 1
        return "ok"

    agent.available_tools = {tool_name: t}
    tc = [{"id": f"c{i}", "type": "function",
           "function": {"name": tool_name, "arguments": json.dumps(args)}}
          for i in range(2)]
    ts = _ts(tool_calls=tc)
    await agent._dispatch_and_process_tool_batch(ts)
    return calls["n"]


class TestIsMutatingCompletenessPins:
    """Lens B MINOR + lens C-c case-insensitivity: aliased / unadvertised /
    case-variant kb mutations must not collapse."""

    @pytest.mark.asyncio
    async def test_kb_update_profile_action_not_collapsed(self):
        # ⚠ The original reason is GONE: this was an unadvertised
        # pass-through to tool_update_profile (a write), and §4DK made it a
        # redirect that writes nothing. The entry stays for a second reason
        # this test protects — collapsing a byte-identical repeat replaces
        # the result with a "no new information" note, which would swallow
        # the redirect telling the model which actions this tool actually
        # has. A corrective error is worth repeating.
        assert await _run_dup("knowledge_base",
                              {"action": "update_profile", "category": "root",
                               "key": "city", "value": "Athens"}) == 2

    @pytest.mark.asyncio
    async def test_kb_uppercase_transcribe_not_collapsed(self):
        assert await _run_dup("knowledge_base",
                              {"action": "TRANSCRIBE",
                               "filename": "t.mp4"}) == 2

    @pytest.mark.asyncio
    async def test_kb_padded_ingest_not_collapsed(self):
        assert await _run_dup("knowledge_base",
                              {"action": " ingest \n",
                               "filename": "a.pdf"}) == 2


class TestActionHealingAndStrikePins:
    """Two guards in the same block that still read a raw value."""

    @pytest.mark.asyncio
    async def test_a_padded_define_still_invalidates_the_tool_cache(self):
        """⚠ REPLACES `test_composed_skill_define_is_case_healed`, which
        could not fail on its stated subject: it counted dispatches of
        `manage_composed_skills`, and that tool is not in
        `_COLLAPSE_READSAFE`, so a collapse was impossible regardless of
        `is_mutating`. `assert == 2` was satisfied by the allowlist alone
        and its causal docstring was false.

        The observable that DOES depend on healing the action is the
        tool-definition cache: `invalidate_tool_defs` lowered the action
        without stripping it, so `action=" define "` registered the macro
        and left the cache stale — the model then narrates "now invoking X"
        against a schema that does not carry X.
        """
        for action in ("define", " define ", " approve \n", "DELETE"):
            agent = _dispatch_agent()
            invalidated = {"n": 0}
            agent.available_tools = {"manage_composed_skills": _ok_tool()}
            agent._rebuild_available_tools = lambda *a, **k: None
            ts = _ts(tool_calls=[{
                "id": "c0", "type": "function",
                "function": {"name": "manage_composed_skills",
                             "arguments": json.dumps({"action": action,
                                                      "name": "s"})}}])
            rs = getattr(ts, "request_state", None)
            if rs is not None and hasattr(rs, "invalidate_tool_defs"):
                orig = rs.invalidate_tool_defs

                def _spy(*a, _o=orig, **k):
                    invalidated["n"] += 1
                    return _o(*a, **k)

                rs.invalidate_tool_defs = _spy
            await agent._dispatch_and_process_tool_batch(ts)
            if rs is not None and hasattr(rs, "invalidate_tool_defs"):
                assert invalidated["n"] == 1, (
                    f"action={action!r} registered the macro without "
                    f"invalidating the tool-definition cache"
                )

    @pytest.mark.asyncio
    async def test_an_uninvokable_call_burns_a_strike(self):
        """`describe_invocation_error` exists because models REPEAT a call
        they cannot enter. The branch set `last_was_failure` but never
        incremented `execution_failure_count`, so the 6-strike cap — the
        only backstop on that loop — was never armed. Driven through the
        real batch with a tool that cannot accept the arguments."""
        agent = _dispatch_agent()

        # The REAL shape: the registry dispatch lambda already supplies
        # `model_name`, so a model that also passes it makes the binding
        # itself fail — synchronously, at call time.
        async def inner(action=None, model_name=None, **kw):
            return "ok"

        agent.available_tools = {
            "knowledge_base": lambda **kw: inner(model_name="system", **kw)}
        tc = [{"id": "c0", "type": "function",
               "function": {"name": "knowledge_base",
                            "arguments": json.dumps({"action": "forget",
                                                     "model_name": "x"})}}]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)

        assert ts.execution_failure_count >= 1, (
            "an un-enterable call did not burn a strike; a model repeating "
            "it loops until the turn cap instead of the strike cap"
        )
        body = "".join(str(m.get("content", "")) for m in ts.messages)
        assert "REMOVE" in body and "'model_name'" in body

    @pytest.mark.asyncio
    async def test_an_exception_from_inside_the_tool_is_described_too(self):
        """Two exception paths reach the model: one at BINDING time (the
        registry lambda's kwargs collide, raised synchronously) and one from
        the AWAITED coroutine (`str_res = f"Error: {result}"`). Only the
        first was rewritten, so an argument error raised a frame deeper
        still read as a bare TypeError with no instruction attached."""
        agent = _dispatch_agent()

        async def t(**kwargs):
            raise TypeError(
                "inner() got multiple values for keyword argument 'sandbox_dir'")

        agent.available_tools = {"knowledge_base": t}
        tc = [{"id": "c0", "type": "function",
               "function": {"name": "knowledge_base",
                            "arguments": json.dumps({"action": "forget"})}}]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)

        body = "".join(str(m.get("content", "")) for m in ts.messages)
        assert "'sandbox_dir'" in body and "REMOVE" in body, (
            f"the awaited-coroutine path still surfaces a bare TypeError: "
            f"{body[:200]!r}"
        )
        # ...and it still reads as a failure to the turn loop.
        assert body.lstrip().startswith(("Error", "SYSTEM ERROR"))

    @pytest.mark.asyncio
    async def test_a_crashing_tool_is_booked_as_a_FAILURE(self):
        """The turn loop books a failure on
        `startswith(("Error:", "ERROR", "SYSTEM ERROR", "Critical Tool
        Error"))`. `describe_invocation_error`'s fallback returned "Error
        invoking tool …", which matches NONE of them — so every
        non-argument exception from every tool became a clean success: no
        strike, no classifier, no diagnostic, the clean-success streak
        advanced, and a crashed idempotent setter had its hash recorded as
        applied so the model's retry was refused with "the intended state is
        already applied"."""
        agent = _dispatch_agent()

        async def t(**kwargs):
            raise ValueError("kernel exploded")

        agent.available_tools = {"knowledge_base": t}
        tc = [{"id": "c0", "type": "function",
               "function": {"name": "knowledge_base",
                            "arguments": json.dumps({"action": "query"})}}]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)

        body = "".join(str(m.get("content", "")) for m in ts.messages)
        assert body.lstrip().startswith(
            ("Error:", "ERROR", "SYSTEM ERROR", "Critical Tool Error")), (
            f"a crashing tool produced a result the turn loop reads as a "
            f"SUCCESS: {body[:120]!r}"
        )
        assert "kernel exploded" in body
        assert ts.execution_failure_count >= 1

    @pytest.mark.asyncio
    async def test_a_crashed_setter_is_not_recorded_as_applied(self):
        """The consequence that makes the prefix load-bearing: with the
        crash booked as a success, `executed_idempotent` stored the hash and
        the model's legitimate retry was blocked as a duplicate — the tool
        having run zero times."""
        agent = _dispatch_agent()
        runs = {"n": 0}

        async def t(**kwargs):
            runs["n"] += 1
            raise RuntimeError("profile store locked")

        agent.available_tools = {"knowledge_base": t}
        args = json.dumps({"action": "insert_fact", "fact": "Athens"})
        # ⚠ ONE SHARED SET, as the real loop has. `_ts()` builds a fresh
        # `executed_idempotent` per call, so the cross-call guard could never
        # fire and this test — written to pin the CONSEQUENCE that makes the
        # failure prefix load-bearing — could not fail. Reverting the prefix
        # fix left it green.
        shared = set()
        for _ in range(2):
            tc = [{"id": "c0", "type": "function",
                   "function": {"name": "knowledge_base", "arguments": args}}]
            await agent._dispatch_and_process_tool_batch(
                _ts(tool_calls=tc, executed_idempotent=shared))

        assert runs["n"] == 2, (
            "the second attempt was refused as already-applied, although the "
            "first one crashed and applied nothing"
        )

    @pytest.mark.asyncio
    async def test_a_binding_failure_survives_a_sibling_success(self):
        """`execution_failure_count` is decayed on a turn with no RECORDED
        failure, and this branch never reaches the classifier that records
        one — so pairing an un-enterable call with any trivial read ended
        the turn on zero strikes."""
        agent = _dispatch_agent()

        async def ok(**kwargs):
            return "fine"

        async def inner(action=None, model_name=None, **kw):
            return "ok"

        agent.available_tools = {
            "recall": ok,
            "knowledge_base": lambda **kw: inner(model_name="system", **kw)}
        tc = [
            {"id": "c0", "type": "function",
             "function": {"name": "knowledge_base",
                          "arguments": json.dumps({"action": "forget",
                                                   "model_name": "x"})}},
            {"id": "c1", "type": "function",
             "function": {"name": "recall",
                          "arguments": json.dumps({"query": "anything"})}},
        ]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)

        assert ts.execution_failure_count >= 1, (
            "the sibling success cancelled the strike; a model that pairs "
            "its broken call with a trivial read never trips the cap"
        )


class TestNameResolutionBehaviour:
    """The name-resolution guards were pinned almost entirely by AST shape.
    A mutation audit showed the cost: reverting a guard to
    `tool["function"]["name"]` — the same raw read, spelled differently —
    walked past every one of those pins, and deleting the identity
    short-circuit that stops a skill hijacking a built-in left all 115
    tests green. These drive the real dispatch instead."""

    @pytest.mark.asyncio
    async def test_an_aliased_name_does_not_collapse_two_real_ingests(self):
        """`is_mutating` keyed on the raw name classified two identical
        `knowledgebase` ingests as read-safe, so the second was
        dedup-collapsed and a real ingest was dropped."""
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def t(**kwargs):
            calls["n"] += 1
            return "ok"

        agent.available_tools = {"knowledge_base": t}
        # The dispatch rebuilds the tool map on a miss (an alias IS a miss),
        # which would replace this stub with the real registry. The subject
        # here is name resolution, not registry loading.
        agent._rebuild_available_tools = lambda *a, **k: None
        args = json.dumps({"action": "ingest_document", "filename": "a.pdf"})
        tc = [{"id": f"c{i}", "type": "function",
               "function": {"name": "knowledgebase", "arguments": args}}
              for i in range(2)]
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=tc))
        assert calls["n"] == 2, (
            "an aliased ingest was collapsed as a read; the second call — a "
            "real ingest — never ran"
        )

    @pytest.mark.asyncio
    async def test_an_aliased_setter_is_still_idempotency_blocked(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def t(**kwargs):
            calls["n"] += 1
            return "ok"

        agent.available_tools = {"knowledge_base": t}
        agent._rebuild_available_tools = lambda *a, **k: None
        args = json.dumps({"action": "insert_fact", "fact": "x"})
        tc = [{"id": f"c{i}", "type": "function",
               "function": {"name": "kb", "arguments": args}}
              for i in range(2)]
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=tc))
        assert calls["n"] == 1

    def test_a_skill_cannot_take_a_builtin_name(self):
        """Acquired skills are appended AFTER the built-ins, so a skill
        legally named `filesystem` won the normalised key of `file_system`
        and every guard keyed on the result asked about the wrong tool.
        Deleting the identity short-circuit that fixes this left the whole
        suite green."""
        from ghost_agent.core.agent import GhostAgent

        av = ["file_system", "knowledge_base", "filesystem", "knowledgebase"]
        assert GhostAgent._canonicalise_tool_name("file_system", av) == "file_system"
        assert GhostAgent._canonicalise_tool_name("knowledge_base", av) == "knowledge_base"
        # ...and the skill still resolves to ITSELF, which is the other half.
        assert GhostAgent._canonicalise_tool_name("filesystem", av) == "filesystem"

    def test_a_shadowing_skill_does_not_disarm_the_wipe_flag(self):
        """The same defect reached through `call_runs_a_memory_wipe`: with a
        skill named `knowledgebase` present, a real `knowledge_base` forget
        resolved to the skill and the flag stayed False while the wipe
        ran — §4DK's tombstone resurrection, restored."""
        from ghost_agent.core.agent import call_runs_a_memory_wipe

        av = ["knowledge_base", "file_system", "knowledgebase"]
        assert call_runs_a_memory_wipe(
            "knowledge_base", '{"action": "forget", "target": "atlas"}',
            available=av) is True


class TestTheFailureContractIsOneThing:
    """Nine live tool returns say "Error " with a SPACE — `tool_remember`'s
    "Error storing memory: …", `file_system`'s "Error 404 - Failed to
    download …", and seven more. A colon-only gate booked every one as a
    clean success, and the test that claimed to pin the contract hard-coded
    its own COPY of the loop's prefix tuple and applied it only to
    `knowledge_base`'s own refusals — never to the inner tool `insert_fact`
    actually runs. The contract is one exported predicate now."""

    @pytest.mark.parametrize("text", [
        "Error storing memory: boom",                     # tool_remember
        "Error 404 - Failed to download from http://x",   # file_system
        "Error: invoking tool 'x' failed",                # the describer
        "SYSTEM ERROR: The 'target' parameter is MANDATORY",
        "ERROR unhandled",
        "Critical Tool Error: sandbox gone",
        "  Error storing memory: boom",                   # leading space
    ])
    def test_every_failure_shape_reads_as_a_failure(self, text):
        from ghost_agent.tools.tool_failure import result_is_failure
        assert result_is_failure(text) is True

    @pytest.mark.parametrize("text", [
        "Success: stored", "Report: Memory disabled.", "OK",
        "Errors were avoided", "", None,
    ])
    def test_a_success_is_not_a_failure(self, text):
        from ghost_agent.tools.tool_failure import result_is_failure
        assert result_is_failure(text) is False

    @pytest.mark.asyncio
    async def test_a_failing_setter_is_not_recorded_as_applied(self):
        """The consequence, driven: `insert_fact` dispatches to
        `tool_remember`, which RETURNS "Error storing memory: …" rather than
        raising. Booked as a success, its hash went into
        `executed_idempotent` and the retry was refused as a duplicate — the
        fact never stored."""
        agent = _dispatch_agent()
        runs = {"n": 0}

        async def t(**kwargs):
            runs["n"] += 1
            return "Error storing memory: disk full"

        agent.available_tools = {"knowledge_base": t}
        args = json.dumps({"action": "insert_fact", "fact": "Athens"})
        shared = set()
        for _ in range(2):
            tc = [{"id": "c0", "type": "function",
                   "function": {"name": "knowledge_base", "arguments": args}}]
            await agent._dispatch_and_process_tool_batch(
                _ts(tool_calls=tc, executed_idempotent=shared))

        assert runs["n"] == 2, (
            "the retry was refused as already-applied although the first "
            "attempt returned an error and stored nothing"
        )
        assert not shared, "a failed setter was recorded as applied"

    def test_the_loop_asks_the_outcome_not_the_text(self):
        """The dispatch loop must decide from ONE `ToolOutcome`, not from
        the result string.

        An audit found TEN vocabularies asking "did this fail?", five of
        them inlined in this very method and disagreeing with each other
        about tracebacks, refusals and non-zero exits. This asserts the
        method contains no inline prefix tuple and no second classification
        — every decision reads the outcome built once at the top.
        """
        import ast
        import inspect

        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        for inlined in ('startswith(("Error:"', 'startswith(("Error"',
                        '"Error" in str_res'):
            assert inlined not in src, (
                f"the failure prefixes are inlined again ({inlined}); the "
                f"loop must read the outcome"
            )

        tree = ast.parse(src.lstrip())
        coerced = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and "ToolOutcome" in ast.unparse(n.func)]
        assert coerced, "the loop no longer builds a ToolOutcome"

        # ...and `_res_is_error` is derived from it, not re-derived.
        derivations = [ast.unparse(n.value) for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == "_res_is_error"]
        assert derivations, "_res_is_error was renamed or removed"
        for d in derivations:
            assert "_outcome" in d, (
                f"a second classification of the result: {d[:80]}"
            )


class TestForesightStillRecognisesRejections:
    def test_the_describer_output_is_synthetic(self):
        """`foresight._SYNTHETIC_RESULT_PREFIXES` listed the wording the
        describer used BEFORE it was reworded, and the foresight test pinned
        that literal — so the guard was inoperative while its test was
        green. Call the producer, do not copy its string."""
        from ghost_agent.core.agent import describe_invocation_error
        from ghost_agent.core.foresight import is_synthetic_result

        for exc in (ValueError("kernel exploded"),
                    TypeError("f() got multiple values for keyword argument 'x'"),
                    TypeError("f() got an unexpected keyword argument 'y'")):
            msg = describe_invocation_error("knowledge_base", exc)
            assert is_synthetic_result(msg), (
                f"a rejection the live hook never resolves would be seeded "
                f"into the trajectory corpus as a real FAILED transition: "
                f"{msg[:80]!r}"
            )


class TestARefusalIsNotAWorldChange:
    """`file_system` answers a malformed mutating call with "SYSTEM
    INSTRUCTION: you used operation='replace' but forgot 'replace_with'" or
    "REPLACE REJECTED (byte-identical)". Neither matched any failure
    predicate, so a call that changed NOTHING was credited with changing the
    world: every recorded pre-flight failure cleared, the loop-breaker's
    memory wiped, the strike count DECREMENTED (so a run of rejected
    replaces erases earlier strikes and the cap can never fire), the file
    marked modified in the work log, and the foresight model taught that the
    call succeeds. 63 live occurrences across 36 requests."""

    @pytest.mark.parametrize("text", [
        "SYSTEM INSTRUCTION: You used operation='replace' but forgot 'replace_with'.",
        "REPLACE REJECTED (byte-identical): the file already contains that text.",
        "REJECTED: none of the SEARCH/REPLACE blocks matched.",
        "PARTIAL: Wiped 3 entries; 1 batch(es) failed.",
    ])
    def test_a_refusal_is_recognised(self, text):
        # `result_changed_nothing` was deleted — it had no product
        # consumer, and the question it named is now `changed_the_world` on
        # the outcome, which is where the loop actually asks it.
        from ghost_agent.tools.outcome import ToolOutcome
        from ghost_agent.tools.tool_failure import result_is_rejection

        assert result_is_rejection(text) is True
        o = ToolOutcome.coerce(text)
        assert o.is_failure is True
        # a refusal touched nothing; a PARTIAL did
        assert o.changed_the_world is text.startswith("PARTIAL:")

    @pytest.mark.parametrize("text", [
        "SUCCESS: Applied 1 SEARCH/REPLACE blocks to 'x.js'.",
        "Wrote 42 bytes.", "",
    ])
    def test_a_real_write_is_not_a_refusal(self, text):
        from ghost_agent.tools.tool_failure import result_is_rejection
        assert result_is_rejection(text) is False

    @pytest.mark.asyncio
    async def test_a_rejected_write_does_not_decay_a_strike(self):
        """The consequence that makes this load-bearing: a rejection scored
        as a clean success ran the decay, so repeated malformed replaces
        erased the strikes their own failures had earned."""
        agent = _dispatch_agent()

        async def rejecting(**kwargs):
            return ("SYSTEM INSTRUCTION: You used operation='replace' but "
                    "forgot 'replace_with'.")

        async def failing(**kwargs):
            return "Error: something genuinely broke"

        agent.available_tools = {"file_system": rejecting, "recall": failing}
        tc = [
            {"id": "c0", "type": "function",
             "function": {"name": "recall",
                          "arguments": json.dumps({"query": "x"})}},
            {"id": "c1", "type": "function",
             "function": {"name": "file_system",
                          "arguments": json.dumps({"operation": "replace",
                                                   "path": "a.py"})}},
        ]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)

        assert ts.execution_failure_count >= 1, (
            "the rejected write was credited as a success and decayed the "
            "strike the real failure earned"
        )

    @pytest.mark.asyncio
    async def test_a_nonzero_exit_is_not_reported_as_SUCCEEDED(self):
        """`op_outcomes` feeds the MULTI-STEP OUTCOME line, which tells the
        model "the successful operations DID take effect … AUTHORITATIVE
        over any prior context". Measured, it called a shell command that
        exited 127 SUCCEEDED in 15 live turns, because the exit-code rescue
        lives in a different predicate three hundred lines below."""
        agent = _dispatch_agent()

        async def exited(**kwargs):
            return "stdout:\nbash: frobnicate: command not found\nEXIT CODE: 127"

        async def broke(**kwargs):
            return "Error: 'x.py' not found"

        agent.available_tools = {"execute": exited, "file_system": broke}
        tc = [
            {"id": "c0", "type": "function",
             "function": {"name": "execute",
                          "arguments": json.dumps({"command": "frobnicate"})}},
            {"id": "c1", "type": "function",
             "function": {"name": "file_system",
                          "arguments": json.dumps({"operation": "read",
                                                   "path": "x.py"})}},
        ]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)

        body = "".join(str(m.get("content", "")) for m in ts.messages)
        if "MULTI-STEP OUTCOME" in body:
            head = body.split("SUCCEEDED:")[1].split("\n")[0]
            assert "execute" not in head, (
                f"a command that exited 127 was reported to the model as "
                f"SUCCEEDED under an AUTHORITATIVE banner: {head!r}"
            )

    @pytest.mark.asyncio
    async def test_a_rejected_write_does_not_credit_a_world_change(self):
        """The two consequence sites, driven. A rejection scored as a
        successful mutation cleared every recorded pre-flight failure and
        wiped the loop-breaker's memory — so a model repeating a malformed
        replace kept resetting the guards that exist to stop it."""
        agent = _dispatch_agent()

        async def rejecting(**kwargs):
            return ("SYSTEM INSTRUCTION: You used operation='replace' but "
                    "forgot 'replace_with'.")

        agent.available_tools = {"file_system": rejecting}
        ts = _ts(tool_calls=[{
            "id": "c0", "type": "function",
            "function": {"name": "file_system",
                         "arguments": json.dumps({"operation": "replace",
                                                  "path": "a.py"})}}])
        strikes = ts.strikes
        seen = {"world": 0}
        if hasattr(strikes, "note_world_changed"):
            orig = strikes.note_world_changed

            def _spy(*a, _o=orig, **k):
                seen["world"] += 1
                return _o(*a, **k)

            strikes.note_world_changed = _spy

        await agent._dispatch_and_process_tool_batch(ts)

        assert seen["world"] == 0, (
            "a refusal that changed nothing was credited with changing the "
            "world; every recorded failure was cleared"
        )

    @pytest.mark.asyncio
    async def test_a_real_write_still_credits_a_world_change(self):
        """...and the guard must not swallow the real case."""
        agent = _dispatch_agent()

        async def writing(**kwargs):
            return "SUCCESS: Wrote 42 bytes to a.py."

        agent.available_tools = {"file_system": writing}
        ts = _ts(tool_calls=[{
            "id": "c0", "type": "function",
            "function": {"name": "file_system",
                         "arguments": json.dumps({"operation": "write",
                                                  "path": "a.py",
                                                  "content": "x"})}}])
        strikes = ts.strikes
        seen = {"world": 0}
        if hasattr(strikes, "note_world_changed"):
            orig = strikes.note_world_changed

            def _spy(*a, _o=orig, **k):
                seen["world"] += 1
                return _o(*a, **k)

            strikes.note_world_changed = _spy

        await agent._dispatch_and_process_tool_batch(ts)
        assert seen["world"] == 1

    def test_exit_code_zero_is_a_success(self):
        """The non-zero-exit guard must not read a clean exit as a failure —
        `EXIT CODE: 0` is how a successful command reports itself."""
        from ghost_agent.core.agent import _EXIT_CODE_FAIL_RE

        assert not _EXIT_CODE_FAIL_RE.search("stdout: hi\nEXIT CODE: 0")
        assert _EXIT_CODE_FAIL_RE.search("EXIT CODE: 1")
        assert _EXIT_CODE_FAIL_RE.search("EXIT CODE: 127")
        assert not _EXIT_CODE_FAIL_RE.search("EXIT CODE: 00")

    @pytest.mark.asyncio
    async def test_a_rejected_write_does_not_clear_the_preflight_guard(self):
        """The OTHER world-changed site. `_failure_guard.note_world_changed()`
        clears every recorded pre-flight failure on the premise that a
        successful mutation invalidates them — a refusal mutated nothing, so
        clearing them disarms the guard on the strength of work that did not
        happen. Spying on the strike ledger alone left this one alive."""
        agent = _dispatch_agent()
        seen = {"cleared": 0}

        guard = getattr(agent, "_failure_guard", None)
        if guard is None or not hasattr(guard, "note_world_changed"):
            pytest.skip("no pre-flight guard on this agent")
        orig = guard.note_world_changed

        def _spy(*a, _o=orig, **k):
            seen["cleared"] += 1
            return _o(*a, **k)

        guard.note_world_changed = _spy

        async def rejecting(**kwargs):
            return "REPLACE REJECTED (byte-identical): nothing to change."

        agent.available_tools = {"file_system": rejecting}
        try:
            await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[{
                "id": "c0", "type": "function",
                "function": {"name": "file_system",
                             "arguments": json.dumps({"operation": "replace",
                                                      "path": "a.py"})}}]))
        finally:
            guard.note_world_changed = orig

        assert seen["cleared"] == 0, (
            "a refusal cleared every recorded pre-flight failure"
        )
