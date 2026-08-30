"""Round-4 (§4DQ): the fixes that round 3's fixes needed.

Two review lenses ran against the round-3 tree. One replayed the 4,391-call
corpus through every reader OUTSIDE the dispatch loop; the other attacked
round 3's own changes. Between them they found three CRITICALs inside round
3's fixes and five unswept consumers — and, separately, that **1,009 tests
passed with and without three of those fixes applied**. So every rule below
is pinned here, and every pin was mutation-checked.
"""

import ast
import inspect


def _loop_expr(name):
    """EXTRACT the named assignment out of the live dispatch method and
    return it as a callable.

    Executed, not asserted-about. Every pin in this area that checked for a
    TOKEN (`"UNRESOLVED" in <unparsed test>`, `"declared" in <gate>`) was
    walked straight through by a whole-file mutant that kept the token and
    changed the meaning — six of them in one review round.
    """
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.tools.outcome import OutcomeStatus

    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    tree = ast.parse(src.lstrip())
    exprs = [n.value for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and getattr(n.targets[0], "id", None) == name]
    assert len(exprs) == 1, (
        f"expected exactly ONE `{name}` definition in the dispatch loop, "
        f"found {len(exprs)} — a second definition is how seven consumers "
        f"of one call came to disagree")
    code = compile(ast.Expression(exprs[0]), "<extracted>", "eval")

    def _run(outcome, str_res=None, fname="file_system"):
        ns = {
            "_outcome": outcome,
            "str_res": str_res if str_res is not None else str(outcome),
            "_OutcomeStatus": OutcomeStatus,
            "fname": fname,
            "_op_shell_failed": (outcome.shell_failed if fname == "execute"
                                 else outcome.exit_code_failed),
        }
        ns["_res_is_error"] = (
            _loop_expr("_res_is_error")(outcome, str_res, fname)
            if name != "_res_is_error" else None)
        return bool(eval(code, {}, ns))
    return _run


class TestTheLoopDoesNotDeclareOnTheProducersBehalf:
    """CRITICAL. `ToolOutcome.__new__` defaults `declared=True`, so a
    construction that forwards `status` but not `declared` re-labels a
    DERIVED ok — what `coerce` produces for `execute`, `jobs`,
    `manage_services` — as a producer's own declaration. A declared ok
    settles the corpus label, so the sniffer becomes unreachable: measured,
    **151 of 4,391 calls lost the structured `ToolCall.error` flag**, 116 of
    them `execute`. That is the −198 regression rebuilt from the other end,
    and the "execute never declares" guard cannot see it because the
    offending code is in the LOOP, in a file that pin never opens."""

    def test_every_status_forwarding_construction_forwards_declared(self):
        import ghost_agent.core.agent as A
        import ghost_agent.tools.execute as E
        import ghost_agent.tools.file_system as F

        missing = []
        for mod in (A, E, F):
            tree = ast.parse(inspect.getsource(mod))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if ast.unparse(node.func) not in ("ToolOutcome", "_TO"):
                    continue
                kw = {k.arg for k in node.keywords if k.arg}
                if "status" in kw and "declared" not in kw:
                    missing.append(
                        f"{mod.__name__}:{node.lineno} {ast.unparse(node)[:70]}")
        assert not missing, (
            "constructions that forward a status but silently DECLARE it:\n  "
            + "\n  ".join(missing))

    def test_the_relayed_message_is_BUILT_with_declared_intact(self):
        """EXECUTED, not token-matched. The sibling above asserts the keyword
        NAME appears; both `declared=True` and `declared=not
        _outcome.declared` keep that name and pass it, while 197 corpus calls
        lose their `ToolCall.error` flag. So extract the real construction
        and run it."""
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        content_expr = None
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None) == "tool_msg"):
                continue
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == "content":
                    content_expr = v
        assert content_expr is not None, "tool_msg moved — re-point this pin"
        code = compile(ast.Expression(content_expr), "<extracted>", "eval")

        for src_outcome, want in (
                (ToolOutcome.coerce("xxd: not found\nEXIT CODE: 127"), False),
                (ToolOutcome.coerce("SUCCESS: done"), False),
                (ToolOutcome.rejected("SYSTEM BLOCK: x"), True),
                (ToolOutcome.ok("declared success"), True)):
            built = eval(code, {"ToolOutcome": ToolOutcome},
                         {"_outcome": src_outcome, "safe_res": str(src_outcome)})
            assert built.status is src_outcome.status
            assert built.declared is want, (
                f"the loop relayed a {'declared' if want else 'DERIVED'} "
                f"outcome as declared={built.declared} — a derived status "
                "that claims to be declared settles the corpus label and "
                "makes the sniffer unreachable"
            )

    def test_a_derived_crash_still_reaches_the_corpus_as_a_failure(self):
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import ToolOutcome

        # exactly what the loop builds: coerce, then re-wrap for `\r` and
        # truncation, forwarding the fields
        src = ToolOutcome.coerce("xxd: command not found\nEXIT CODE: 127")
        relayed = ToolOutcome(str(src), status=src.status,
                              world_changed=src.world_changed,
                              reason_code=src.reason_code,
                              declared=src.declared)
        assert relayed.declared is False
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "execute", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "execute",
             "content": relayed},
        ]
        assert GhostAgent._reconstruct_tool_calls(msgs)[0].error, (
            "a command that exited 127 reached the corpus as a SUCCESS "
            "because the loop declared its derived status"
        )


class TestUnresolvedIsNotAVerdictAnywhere:
    def test_an_in_flight_call_is_not_a_failed_corpus_row(self):
        """CRITICAL, and NEW in round 3: `"unresolved" != "ok"`, so the
        first clause of the corpus label fired. That contradicts
        `outcome_heuristics`' own contract ("callers must SKIP an unresolved
        call rather than label it") and seeds the foresight world model with
        "this command shape FAILS" while the live grader refuses to grade
        the same row."""
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import ToolOutcome

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "execute", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "execute",
             "content": ToolOutcome.unresolved(
                 "--- COMMAND RESULT --- [sandbox job promoted]\n"
                 "EXIT CODE: 0 (STILL RUNNING, NOT finished)")},
        ]
        assert not GhostAgent._reconstruct_tool_calls(msgs)[0].error

    def test_the_strike_branch_reads_the_ONE_verdict(self):
        """MAJOR. The elif chain was a FOURTH reading of "is this a failure"
        — narrower than both of the two `_res_is_error` definitions that
        then existed. A detached job still drew a strike (making `swarm`'s
        PARTIAL → UNRESOLVED change a no-op at the one site it was made
        for), and a non-`execute` tool whose only signal was a non-zero
        `EXIT CODE:` drew none at all: 8 calls with `op_outcomes.ok = False`
        and zero strikes, and because `summarize_multi_op_outcomes` runs
        inside `if turn_has_failure:`, 6 turns where the MULTI-STEP line
        never reached the model."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        tests = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body = ast.unparse(node.body)
            if "turn_has_failure = True" in body and "failed_fname = fname" \
                    in body and "last_error_preview" in body:
                tests.append(ast.unparse(node.test))
        assert tests, "the strike branch moved — re-point this pin"
        assert any(t.strip() == "_res_is_error" for t in tests), (
            "the strike branch keeps its own reading of failure instead of "
            f"the loop's one verdict: {tests}")

    def test_the_third_state_predicate_sees_every_producer(self):
        """It said "today that means exactly one thing" and was blind to
        `swarm`'s still-running branch, so `tool_failure_flags` emitted a
        clean SUCCESS for a swarm await in flight."""
        from ghost_agent.distill.outcome_heuristics import (
            is_unresolved_tool_result)
        from ghost_agent.tools.outcome import ToolOutcome

        assert is_unresolved_tool_result(
            ToolOutcome.unresolved("PARTIAL: 2/4 done; 2 still running")) is True
        assert is_unresolved_tool_result("SUCCESS: done") is False


class TestPartialIsItsOwnThing:
    def test_a_partial_does_not_arm_the_preflight_guard(self):
        """The guard's premise is "re-running this unchanged will fail the
        same way". A PARTIAL partly SUCCEEDED, so the premise is false.
        `search.py` builds `partial(world_changed=False)` when the research
        landed and only the verification call hiccupped — two of those in a
        row blocked the third re-issue of that query pre-dispatch."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        checked = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            t = ast.unparse(node.test)
            # the INNERMOST guard, not any ancestor that transitively
            # contains the call
            if "_failure_guard.record" not in ast.unparse(node.body):
                continue
            if "_res_is_error" not in t:
                continue
            checked = True
            assert "PARTIAL" in t, (
                "a half-landed call arms the guard against its own "
                "corrected retry: " + t)
        assert checked, "the guard record moved — re-point this pin"

    def test_a_repeated_partial_is_not_told_to_stop(self):
        """Round 3 branched the steer for REJECTED and left PARTIAL on the
        hard-failure arm, so a write that landed but does not parse got
        "STOP repeating it — retrying will not change the result… if a file
        is missing, CREATE it" — for a syntax error, where the correct move
        is exactly the retry it forbids."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        # ⚠ NOT `"failure_was_partial" in src`: rewriting the branch test to
        # `if False:` leaves both that name and the steer text in the source
        # and the pin stayed green on the exact mutant it exists to catch.
        # The branch has to be REACHABLE and it has to build the steer.
        reachable = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if ast.unparse(node.test).strip() != "failure_was_partial":
                continue
            body = ast.unparse(node.body)
            assert "_steer" in body, (
                "the PARTIAL branch no longer builds a steer")
            assert "PART OF THIS" in body.upper(), (
                "the PARTIAL branch gives the hard-failure advice")
            reachable = True
        assert reachable, (
            "no reachable `if failure_was_partial:` branch — a half-landed "
            "call falls through to \"STOP repeating it\", which for a "
            "syntax error forbids exactly the retry that fixes it")


class TestTheBannerTheModelActuallyReads:
    def test_the_failure_banner_reads_the_outcome(self):
        """The `[FAILURE BANNER]` prepend and the fallback-hint scan were
        two byte-identical copies of a text rule 17 lines apart, neither
        reading `_outcome` — which is in scope on the same line. Measured:
        **220 results the loop books as failures got no banner** (file_system
        67, execute 60, browser 42, manage_projects 15) and 9 clean results
        got one, through the unanchored `"Traceback" in str_res` substring
        that `_res_is_error` already guards 140 lines below. This is the
        last thing the model reads before deciding to retry or pivot."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        from ghost_agent.tools.outcome import ToolOutcome

        gate = _loop_expr("_failure_shaped")
        # a refusal — invisible to every text rule
        assert gate(ToolOutcome.rejected("SYSTEM INSTRUCTION: forgot 'x'")), (
            "the banner still cannot see a refusal")
        # a non-zero exit on a non-shell tool
        assert gate(ToolOutcome.coerce("--- job ---\nEXIT CODE: 1"),
                    fname="jobs")
        # a DECLARED success whose body quotes someone else's crash
        assert not gate(
            ToolOutcome.ok('{"events": [{"reason": "Traceback ..."}]}')), (
            "a SUCCESSFUL result that quotes a traceback gets a FAILURE "
            "banner")
        # ...and NEITHER does an undeclared one: the banner gate reads the
        # verdict, which no longer carries an unanchored traceback arm. A
        # file read of a log quoting a crash used to get a FAILURE banner
        # and a "the path doesn't exist" fallback hint.
        assert not gate(ToolOutcome.coerce(
            "--- ci.log CONTENTS ---\nTraceback (most recent call last):"))
        # the SHELL still gets it, through `shell_failed`
        assert gate(ToolOutcome.coerce("Traceback (most recent call last):"),
                    fname="execute")
        # a plain success does not
        assert not gate(ToolOutcome.coerce("SUCCESS: wrote 10 chars"))
        # and the hint scan must share it, not keep a second copy
        assert src.count("_failure_shaped") >= 3


class TestTheUnsweptConsumers:
    def test_the_foresight_seed_sees_a_refusal(self):
        """Fully unattended. Foresight kept a PRIVATE, diverging copy of the
        failure vocabulary and no rejection vocabulary at all, so **39 of
        the 82 corpus refusals were seeded into the shadow world model as
        SUCCESSES**. The index is re-seeded from a rolling window on every
        boot, and the LIVE grade is status-aware — so the seed and the live
        resolver disagreed inside one subsystem."""
        from ghost_agent.core.foresight import offline_call_failed

        class _TC:
            def __init__(self, r):
                self.result = r
                self.error = ""

        for text in ("SYSTEM INSTRUCTION: you forgot 'replace_with'",
                     "REPLACE REJECTED (byte-identical)",
                     "REJECTED: that edit would introduce a syntax error",
                     "  Error: leading whitespace used to hide this"):
            assert offline_call_failed(_TC(text)) is True, text
        assert offline_call_failed(_TC("SUCCESS: wrote 10 chars")) is False

    def test_the_turn_shape_rule_keeps_the_status(self):
        """`tool_failure_flags`'s dict path is the turn loop's
        `tools_run_this_turn`, whose `content` IS a `ToolOutcome`. `str()`
        killed the status check inside the shared sniffer before it ran —
        61 of 82 refusals lost, in the THIRD reader of that same list to
        have this exact defect."""
        from ghost_agent.distill.outcome_heuristics import tool_failure_flags
        from ghost_agent.tools.outcome import ToolOutcome

        assert tool_failure_flags(
            [{"content": ToolOutcome.rejected("SYSTEM BLOCK: aborted")}]) == [True]
        assert tool_failure_flags([{"content": "SUCCESS: stored"}]) == [False]
        # an in-flight call is SKIPPED, not labelled
        assert tool_failure_flags(
            [{"content": ToolOutcome.unresolved("still running")}]) == []

    def test_the_verifier_run_gate_sees_a_failed_bookkeeping_call(self):
        """`startswith(("Error", "SYSTEM BLOCK", "REJECTED"))` was
        case-SENSITIVE and inlined at three sites, so `ERROR:` and
        `SYSTEM ERROR:` — which `manage_projects` and `knowledge_base` both
        emit — matched nothing: **15 of 22 failed bookkeeping calls (68%)
        invisible**, 2 turns where the verifier never runs at all and 5
        where the unverified-mutation guard is disarmed. Exactly the blind
        spot the 2026-07-25 error carve-out closed, reopened by letter
        case."""
        from ghost_agent.core.agent import _bookkeeping_call_failed
        from ghost_agent.tools.outcome import ToolOutcome

        for text in ("ERROR: project not found",
                     "SYSTEM ERROR: The 'target' parameter is MANDATORY.",
                     "Error: no such task",
                     "CRITICAL ERROR: nothing ran",
                     "  system block — pre-flight guard"):
            assert _bookkeeping_call_failed(text) is True, text
        assert _bookkeeping_call_failed(
            ToolOutcome.rejected("nothing prose-shaped at all")) is True
        assert _bookkeeping_call_failed("SUCCESS: Profile updated.") is False
        assert _bookkeeping_call_failed(
            ToolOutcome.unresolved("still running")) is False

    def test_the_bench_instrument_has_not_desynced(self):
        """`derive_high_stakes` replicates a production predicate that is
        now status-aware; a packed digest carries only text. 61 of 82
        refusals and 15 turns (7.7% of the production high-stakes
        population) were invisible to the instrument that calibrates the
        CONFIRM escalation."""
        from ghost_agent.eval.verify_bench import derive_high_stakes

        assert derive_high_stakes(
            "[file_system] SYSTEM INSTRUCTION: you forgot 'replace_with'")
        assert derive_high_stakes(
            "[manage_services] SYSTEM BLOCK — pre-flight guard: this call")
        assert derive_high_stakes(
            "[execute] --- EXECUTION RESULT ---\nEXIT CODE: 1")
        assert not derive_high_stakes(
            "[recall] SYSTEM: Found 10 highly relevant memories.")

    def test_a_failed_autoadvance_does_not_declare_success(self):
        """`_ok(_adv_payload)` declared OK on a failure-shaped stop — and a
        declared ok now settles the corpus label, so 5 live rows reporting
        FAILED tasks were exempted from the sniffer while
        `_turn_had_tool_failure` still flagged them: a fresh three-reader
        split."""
        import ghost_agent.tools.projects as P

        tree = ast.parse(inspect.getsource(P))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            t = ast.unparse(node.test)
            if "project_failed" not in t or "stop_reason" not in t:
                continue
            body = ast.unparse(node.body)
            assert "ToolOutcome.partial" in body or "_TO.partial" in body, (
                "a failure-shaped autoadvance stop still declares OK")
            return
        raise AssertionError("the autoadvance stop check moved")


class TestTheDeclaredArmAnswersBothWays:
    def test_a_declared_failure_still_reads_as_an_exit_failure(self):
        """The `declared` arm of `exit_code_failed` could be replaced with
        `return False` and every assertion stayed green: all of them asserted
        the FALSE side on a declared outcome, and the TRUE assertions all
        went through `coerce` (derived), which never takes the branch."""
        from ghost_agent.tools.outcome import ToolOutcome

        assert ToolOutcome.failed("anything at all").exit_code_failed is True
        assert ToolOutcome.ok("EXIT CODE: 1 quoted").exit_code_failed is False
        assert ToolOutcome.rejected("nothing ran").exit_code_failed is False
        assert ToolOutcome.unresolved("EXIT CODE: 0").exit_code_failed is False

    def test_a_promoted_result_keeps_its_status_through_the_probe_note(self):
        """`ToolOutcome` is a `str` SUBCLASS, so `_promoted_result(...) +
        note` is `str.__add__` and returns a plain `str` — the UNRESOLVED
        status was destroyed on the file-mode path while the command-mode
        path kept it. The same event had two verdicts depending on which
        argument the model used."""
        import ghost_agent.tools.execute as E

        tree = ast.parse(inspect.getsource(E))
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value,
                                                           ast.BinOp):
                u = ast.unparse(node.value)
                if "_promoted_result" in u:
                    bad.append(f"line {node.lineno}: {u[:70]}")
        assert not bad, (
            "a promoted result is concatenated, which strips its status:\n  "
            + "\n  ".join(bad))
