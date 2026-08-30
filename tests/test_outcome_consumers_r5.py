"""Round-5 (§4DR): one verdict per call, and the fixes nobody had pinned.

Two lenses drove the REAL dispatch over the 4,391-call corpus. Between them:
the loop held **three** different answers to "did this call fail" in a single
iteration (seven consumers disagreed on 56 calls); the per-ACTION success flag
in the episode store booked 145 of 291 declared non-OK results as successes;
and — the finding that shaped this file — a reviewer reverted the single change
behind the headline "66 refusals credited with changing the world → 0" and ran
the full suite: **18,052 passed, 0 failed**. Measuring a fix is not pinning it.
"""

import ast
import inspect

from tests.test_outcome_consumers_r4 import _loop_expr


def _branch_test(pred):
    """The INNERMOST `if`/`elif` in the dispatch loop whose body satisfies
    `pred`.

    Innermost matters: every ancestor `if` transitively contains the call, so
    taking the first match walked all the way out to `if tool_tasks:`.
    """
    from ghost_agent.core.agent import GhostAgent

    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    tree = ast.parse(src.lstrip())
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not pred(ast.unparse(node.body)):
            continue
        size = len(list(ast.walk(node)))
        if best is None or size < best[0]:
            best = (size, ast.unparse(node.test))
    return best[1] if best else None


class TestOneVerdictPerCall:
    def test_the_loop_has_exactly_one_definition_of_failure(self):
        """There were three in one iteration: a status-only verdict feeding
        `op_outcomes.ok` and the pre-flight guard, a wider rebind 660 lines
        down feeding metacog and the work log, and the strike branch, which
        was narrower than both. Driven through the real dispatch: 18 calls
        where four readers said FAILED and three said success, and 8 with
        `op_ok=False` and zero strikes — so `summarize_multi_op_outcomes`,
        which runs inside `if turn_has_failure:`, never reached the model on
        6 turns."""
        from ghost_agent.tools.outcome import ToolOutcome

        verdict = _loop_expr("_res_is_error")   # asserts there is only one
        # A non-shell tool whose OWN envelope carries a non-zero exit.
        # ⚠ `jobs` was the example here and is the wrong one: it QUOTES the
        # jobs it reports on, so a successful read of a FAILED job carries
        # `EXIT CODE: 1` that is not its own — which is why it declares now.
        assert verdict(ToolOutcome.coerce(
            "Skill creation failed: --- EXECUTION RESULT ---\nEXIT CODE: 1"),
            fname="create_skill") is True, (
            "a non-shell tool's non-zero exit is not a failure to the loop")
        assert verdict(ToolOutcome.ok(
            "--- job-1 [FAILED] ---\nexited 1\nEXIT CODE: 1"),
            fname="jobs") is False, (
            "reading a FAILED job is booked as the `jobs` tool failing, so "
            "the model draws a strike every time it re-reads the same job")
        assert verdict(ToolOutcome.rejected("SYSTEM BLOCK: x")) is True
        assert verdict(ToolOutcome.unresolved("still running")) is False
        assert verdict(ToolOutcome.coerce("SUCCESS: done")) is False


class TestTheWorldChangedResets:
    """CRITICAL. Both resets clear recorded failures on the strength of a
    call having changed something. Reverting the `file_system` one to its
    pre-refactor predicate passed 18,052 tests — the fix behind "66 → 0" was
    measured and never pinned."""

    def test_a_refused_mutation_does_not_clear_the_strike_ledger(self):
        from ghost_agent.tools.outcome import ToolOutcome

        t = _branch_test(lambda b: "strikes.note_world_changed()" in b)
        assert t is not None, "the strike world-changed reset moved"
        code = compile(ast.parse(t, mode="eval"), "<extracted>", "eval")

        def run(o):
            return bool(eval(code, {}, {"_outcome": o, "fname": "file_system",
                                        "is_mutating": True}))

        assert run(ToolOutcome.rejected(
            "SYSTEM INSTRUCTION: you forgot 'replace_with'")) is False, (
            "a refusal that touched NOTHING clears every recorded pre-flight "
            "failure, wipes the loop-breaker's memory and decrements the "
            "strike count — 66 live occurrences")
        assert run(ToolOutcome.failed("Error: could not open 'a.py'")) is False
        assert run(ToolOutcome.ok("SUCCESS: Wrote 40 chars")) is True, (
            "a genuine mutation no longer invalidates stale failures — the "
            "guard deadlocks: a blocked call can never demonstrate the fix")
        assert run(ToolOutcome.partial("SUCCESS: wrote; SYNTAX CHECK FAILED",
                                       world_changed=True)) is True, (
            "a half-landed write DID change the world")

    def test_a_failed_mutation_does_not_disarm_the_preflight_guard(self):
        """It fired on 9 live calls another reader books as failures — the
        operator line reads "World changed (successful manage_services
        mutation) — cleared N recorded failure(s)" for a restart that failed
        to bind. The branch's own comment says "a SUCCESSFUL state-mutating
        call"."""
        t = _branch_test(lambda b: "_failure_guard.note_world_changed()" in b)
        assert t is not None, "the pre-flight world-changed reset moved"
        assert "_res_is_error" in t, (
            "a demonstrably failed mutating call still clears every recorded "
            f"pre-flight failure: {t}")
        assert "_pf_exec_failed" in t and "_pf_promoted" in t


class TestTheFlagsAreNotStaleAcrossTheBatch:
    def test_the_execute_branch_resets_BOTH_failure_flags(self):
        """`failure_was_rejection` was reset there and `failure_was_partial`
        was not — the exact last-write-wins defect the sibling's own comment
        documents. The PARTIAL steer is the FIRST branch of the chain, so a
        shell command that exited non-zero and did nothing was told "PART OF
        THIS LANDED — do NOT re-run the whole operation"."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body = ast.unparse(node.body)
            if "exit_code_val != 0" not in ast.unparse(node.test):
                continue
            # BOTH must be RE-DERIVED here, not left stale and not
            # hard-coded: `execute` has three declared rejection sites whose
            # envelope carries `EXIT CODE: 1`, so `= False` gave an execute
            # refusal the "STOP repeating it" steer while an identical
            # file_system refusal got "re-issue the SAME tool".
            for flag in ("failure_was_rejection", "failure_was_partial"):
                assert f"{flag} = " in body, (
                    f"the execute branch leaves `{flag}` stale from an "
                    "earlier result in the same batch")
                assert f"{flag} = False" not in body, (
                    f"`{flag}` is hard-coded here instead of read from the "
                    "outcome — an execute REFUSAL gets the hard-failure steer")
            return
        raise AssertionError("the execute exit-code branch moved")


class TestTheStoresAndGatesOutsideTheLoop:
    def test_no_stored_row_reader_keeps_a_PRIVATE_prose_bank(self):
        """Superseded in part by `test_a_stored_row_is_judged_the_way_the_LOOP
        _judges_it` (r6), which executes the shared predicate. What stays
        here is the anti-duplication rule that produced three wrong answers
        in three rounds: the episode store and the info-gathering gate must
        not carry their own head-prefix tuple at all.

        Round 4: a five-prefix bank booked 145 of 291 declared non-OK results
        as successes. Round 5: an `if/else` over the status shadowed that
        bank, so 118 non-zero `execute` exits were still stored successful.
        Both were private re-derivations of a question the loop already
        answers.
        """
        import ghost_agent.core.agent as A

        tree = ast.parse(inspect.getsource(A))
        # Scoped to the two ASSIGNMENTS, not the enclosing method: these
        # live inside very large functions that legitimately contain other
        # prefix rules (the reply banner, where the exit-code banner decides).
        offenders = []
        seen = set()
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1):
                continue
            name = getattr(n.targets[0], "id", None)
            if name not in ("_ok", "_ran_info"):
                continue
            seen.add(name)
            u = ast.unparse(n.value)
            # resolve ONE level of delegation: `_ran_info` calls `_info_ok`
            for helper in ("_info_ok",):
                if f"{helper}(" not in u:
                    continue
                for h in ast.walk(tree):
                    if (isinstance(h, ast.FunctionDef)
                            and h.name == helper):
                        u += "\n" + ast.unparse(h)
            if "startswith" in u:
                offenders.append(f"{name}: private prose bank -> {u[:60]}")
            if "_action_failed" not in u:
                offenders.append(f"{name}: does not ask the shared question")
        assert seen == {"_ok", "_ran_info"}, f"a reader moved: {seen}"
        assert not offenders, offenders

    def test_a_bookkeeping_PARTIAL_is_not_a_failure(self):
        """`update_profile` returns PARTIAL when the canonical write LANDED
        and only a secondary index lagged. Counting it failed made it shadow
        the real action in the verifier's run gate — which its own docstring
        says "silently disables the untested-write guard"."""
        from ghost_agent.core.agent import _bookkeeping_call_failed
        from ghost_agent.tools.outcome import ToolOutcome

        assert _bookkeeping_call_failed(
            ToolOutcome.partial("PARTIAL: profile written, index lagged")) is False
        assert _bookkeeping_call_failed(
            ToolOutcome.rejected("SYSTEM BLOCK: x")) is True
        assert _bookkeeping_call_failed("ERROR: project not found") is True

    def test_the_offline_seed_does_not_fail_an_in_flight_call(self):
        """`result_is_rejection`'s vocabulary includes `PARTIAL:`, which is
        the literal head of swarm's UNRESOLVED "still running, NOT cancelled"
        branch — the one every other reader exempts. The seed and the live
        resolver disagreed inside one subsystem."""
        from ghost_agent.core.foresight import offline_call_failed
        from ghost_agent.tools.outcome import ToolOutcome

        class _TC:
            def __init__(self, r):
                self.result = r
                self.error = ""

        running = ToolOutcome.unresolved(
            "PARTIAL: 2/4 completed; 2 still running in the background "
            "(t3, t4). They were NOT cancelled")
        assert offline_call_failed(_TC(running)) is False
        assert offline_call_failed(
            _TC("SYSTEM INSTRUCTION: forgot 'replace_with'")) is True


class TestTheCallSitesPassTheObject:
    def test_a_composed_step_is_classified_from_the_OBJECT(self):
        """Both call sites did `result_str = str(result)` one line before
        asking the status-aware question, so `_step_result_ok`'s status arm
        was unreachable from either — browser FAILED, memory PARTIAL, swarm
        UNRESOLVED and file_system PARTIAL all counted as successful steps
        and inflated the macro success_rate the model is shown."""
        import ghost_agent.tools.composed_skills as C

        tree = ast.parse(inspect.getsource(C))
        calls = [ast.unparse(n) for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and ast.unparse(n.func).endswith("_step_result_ok")]
        assert len(calls) >= 2, f"expected both step paths, got {calls}"
        for c in calls:
            assert "result_str" not in c, (
                f"a step is classified from the stringified result: {c}")

    def test_the_shell_failure_flag_is_not_gated_on_the_verdict(self):
        """`_pf_exec_failed` was set only `if not _res_is_error` — so a
        crashed command left it False and the world-changed reset then
        cleared every recorded pre-flight failure on the strength of the
        crash."""
        t = _branch_test(lambda b: "_pf_exec_failed = _outcome.shell_failed" in b)
        assert t is not None, "the _pf_exec_failed branch moved"
        assert "_res_is_error" not in t, (
            f"a crashed shell command cannot set the flag that protects the "
            f"guard from itself: {t}")


class TestTheProducersRoundFiveMigrated:
    def test_manage_services_declares(self):
        """Its output legitimately QUOTES crashes — `logs` returns a log
        tail — and the loop's traceback rule is an unanchored whole-body
        substring, so 13 live successful reads were booked failures. The rule
        defers to a DECLARED status precisely so a tool that quotes a crash
        can say the crash is not its own."""
        import ghost_agent.tools.sandbox_services as S

        tree = ast.parse(inspect.getsource(S))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_declare")
        u = ast.unparse(fn)
        # `rejected` for an `Error:` head reaching the wrapper (a validation
        # refusal that launched nothing), `ok` otherwise, and an outcome the
        # supervisor already declared passes straight through.
        assert "ToolOutcome.rejected" in u and "ToolOutcome.ok" in u
        assert "isinstance(res, ToolOutcome)" in u
        assert "Error:" in u
        # every dispatch return must go through it
        outer = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.AsyncFunctionDef)
                     and n.name == "tool_manage_services")
        bare = [ast.unparse(r.value)[:60] for r in ast.walk(outer)
                if isinstance(r, ast.Return) and r.value is not None
                and "asyncio.to_thread" in ast.unparse(r.value)
                and "_declare" not in ast.unparse(r.value)]
        assert not bare, f"undeclared manage_services returns: {bare}"

    def test_self_play_and_vision_declare_their_failures(self):
        from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
        from ghost_agent.tools.tool_failure import result_is_failure

        # neither head is visible to the anchored predicate
        assert result_is_failure(
            "Synthetic challenge generation failed: setup script") is False
        assert result_is_failure("Vision API Error: 500") is False
        # ...which is exactly why the producers must say so
        import importlib
        # No BARE failure-shaped return may survive in either module — a
        # token check passed while one of two sites had been reverted.
        for mod, heads in (
                ("ghost_agent.core.dream",
                 ("Synthetic challenge generation failed",)),
                ("ghost_agent.tools.vision", ("Vision API Error",))):
            tree = ast.parse(inspect.getsource(importlib.import_module(mod)))
            bare = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                u = ast.unparse(node.value)
                if not any(h in u for h in heads):
                    continue
                if "ToolOutcome." not in u:
                    bare.append(f"{mod}:{node.lineno}")
            assert not bare, f"bare failure strings still returned: {bare}"
        assert ToolOutcome.failed("x").status is OutcomeStatus.FAILED

    def test_a_browser_argument_refusal_is_a_REJECTION(self):
        """Booking it FAILED armed the pre-flight guard against the model's
        own corrected re-issue — the pathology the guard's docstring
        records."""
        import ghost_agent.tools.browser as B

        src = inspect.getsource(B.tool_browser)
        tree = ast.parse(src.lstrip())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_reject")
        assert "ToolOutcome.rejected" in ast.unparse(fn)
        assert "_reject(f\"Missing 'operation'" in src
        assert "_reject(f'Unknown operation" in src or \
               '_reject(f"Unknown operation' in src

    def test_appending_a_note_never_strips_a_status(self):
        """`ToolOutcome` is a `str` subclass, so `res + note` is
        `str.__add__` and returns a plain `str`. Three sites did that; one
        was found live and its two siblings only by enumeration."""
        import ghost_agent.tools.execute as E
        from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

        out = E._append_note(ToolOutcome.unresolved("EXIT CODE: 0"), "\nnote")
        assert out.status is OutcomeStatus.UNRESOLVED
        assert out.endswith("note")
        assert E._append_note("plain", "\nx") == "plain\nx"

        tree = ast.parse(inspect.getsource(E))
        bad = [f"line {n.lineno}" for n in ast.walk(tree)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.BinOp)
               and any(k in ast.unparse(n.value)
                       for k in ("_promoted_result", "_format_error"))]
        assert not bad, f"a status is concatenated away at {bad}"

    def test_a_zero_advance_batch_is_not_PARTIAL(self):
        import ghost_agent.tools.projects as P

        src = inspect.getsource(P)
        assert "_TO.partial if batch.count else _TO.failed" in src, (
            "a batch that advanced NOTHING still tells the model "
            '"PART OF THIS LANDED" over an empty list')

    def test_a_refusal_gets_no_path_hint(self):
        """`_FALLBACK_HINTS["file_system"]` matches a bare "not found", so a
        REJECTED replace whose message says "the search block was NOT found
        in 'x.py'" was told "the path doesn't exist — run list_files" about a
        file that exists."""
        from ghost_agent.core.agent import GhostAgent

        # the PER-RESULT injection (it appends to `safe_res`), not the
        # turn-level one in the failure block
        t = _branch_test(
            lambda b: "get_fallback_hint" in b and "safe_res" in b)
        assert t is not None, "the hint gate moved — re-point this pin"
        assert "is_rejection" in t, (
            f"the hint scan reaches refusals and diagnoses the wrong thing: {t}")
