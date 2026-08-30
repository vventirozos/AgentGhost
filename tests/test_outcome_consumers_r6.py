"""Round-6 (§4DS): the producers say it, and the pins can fail.

Two lenses. One enumerated by AST every reader OUTSIDE the dispatch loop and
found the measuring instruments were the worst offenders. The other attacked
round 5 and found **twelve pins that could not fail** — the common shape
being a token check (`"ToolOutcome." in <expr>`, `"Error:" in <fn>`) that a
mutant satisfies while inverting the meaning. Everything here executes the
code it guards, or extracts the real expression and runs it.
"""

import ast
import inspect

from tests.test_outcome_consumers_r4 import _loop_expr


def _returns_calling(mod, needle):
    """Every `return` in `mod` whose value mentions `needle`, unparsed."""
    return [ast.unparse(n.value) for n in ast.walk(ast.parse(
        inspect.getsource(mod)))
        if isinstance(n, ast.Return) and n.value is not None
        and needle in ast.unparse(n.value)]


def _extract_fn(mod, outer, inner, **closure):
    """Compile a nested function out of the live source and return it.

    `closure` supplies the enclosing locals the nested function reads (it is
    a closure in the real code; extracted, those become globals).
    """
    src = inspect.getsource(getattr(mod, outer))
    fn = next((n for n in ast.walk(ast.parse(src.lstrip()))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == inner), None)
    assert fn is not None, f"{inner} moved out of {outer}"
    ns = dict(vars(mod))
    ns.update(closure)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<x>", "exec"), ns)
    return ns[inner]


class TestTheProducersDeclareTHEIRactualOutcome:
    """A pin that accepts any expression containing `ToolOutcome.` passes
    while a mutant rewrites `.failed(` to `.ok(`. Four did."""

    def test_vision_and_self_play_declare_FAILURE_not_success(self):
        import ghost_agent.core.dream as D
        import ghost_agent.tools.vision as V

        for mod, head in ((V, "Vision API Error"),
                          (D, "Synthetic challenge generation failed")):
            rets = [r for r in _returns_calling(mod, head)]
            assert rets, f"{mod.__name__}: no {head!r} return found"
            for r in rets:
                assert "ToolOutcome.failed" in r, (
                    f"{mod.__name__} announces a failure as "
                    f"{r.split('(')[0]!r}")

    def test_browser_errors_declare_FAILED_and_refusals_REJECTED(self):
        import ghost_agent.tools.browser as B

        src = inspect.getsource(B.tool_browser)
        for name, want in (("_err", "ToolOutcome.failed"),
                           ("_reject", "ToolOutcome.rejected")):
            fn = next(n for n in ast.walk(ast.parse(src.lstrip()))
                      if isinstance(n, ast.FunctionDef) and n.name == name)
            rets = [ast.unparse(r.value) for r in ast.walk(fn)
                    if isinstance(r, ast.Return) and r.value is not None]
            assert rets and all(want in r for r in rets), (
                f"browser {name} no longer produces {want}: {rets}")

        # ...and the ARGUMENT refusals must route through `_reject`. Checking
        # only what `_reject` returns left three sites (`actions` not a list,
        # `actions[N]` not a dict, a bad sub-path) on `_err`, so the
        # pre-flight guard recorded them and blocked the corrected re-issue.
        _ARG_HEADS = ("interact requires a non-empty", "actions[{idx}]")
        arg_refusals = [
            n for n in ast.walk(ast.parse(src.lstrip()))
            if isinstance(n, ast.Return) and n.value is not None
            and any(h in ast.unparse(n.value) for h in _ARG_HEADS)]
        assert len(arg_refusals) >= 3, (
            f"the argument-refusal sites moved: {len(arg_refusals)}")
        for r in arg_refusals:
            assert "_reject(" in ast.unparse(r.value), (
                f"an argument refusal arms the pre-flight guard against its "
                f"own corrected re-issue: {ast.unparse(r.value)[:70]}")

    def test_manage_services_actually_classifies(self):
        """EXECUTED. The pin used to accept any `_declare` containing the
        token `"Error:"`, so a mutant that can never take the failed branch
        passed all 4,476 tests."""
        import ghost_agent.tools.sandbox_services as S
        from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

        declare = _extract_fn(S, "tool_manage_services", "_declare",
                              action="start")
        # An `Error:` head reaching the WRAPPER is a validation refusal —
        # nothing was launched. The three paths that actually spawned a
        # process, and the two bind failures, declare `failed` themselves in
        # `sandbox/services.py` and pass through untouched. Booking a refusal
        # FAILED armed the pre-flight guard against the model's corrected
        # re-issue: 13 live rows, and the guard then emitted 4 SYSTEM BLOCK
        # blocks across 3 sessions.
        assert declare("Error: no service named 'x'").status \
            is OutcomeStatus.REJECTED
        assert declare("Service 'x' RUNNING (pid 12)").status \
            is OutcomeStatus.OK
        assert declare(ToolOutcome.failed(
            "Error: service 'x' exited immediately")).status \
            is OutcomeStatus.FAILED
        # a supervisor that already SAID keeps its answer
        pre = ToolOutcome.failed("Service 'x' started BUT port 8080 is "
                                 "answered by a DIFFERENT process")
        assert declare(pre) is pre, (
            "the wrapper re-classifies the supervisor's own verdict — and "
            "its head rule reads a failed-to-bind start as a SUCCESS")

    def test_a_service_that_failed_to_bind_is_a_failure(self):
        """The supervisor returns two genuine failures that do NOT lead with
        `Error:` — a service that started and then failed to bind. The
        wrapper declared them OK, and a declared ok short-circuits the banner
        rule AND the guard's `not _res_is_error`, so nothing downstream could
        recover them: 18 live rows over 9 turns, 9.4% of all start/restart
        calls, each also CLEARING the pre-flight guard as a 'successful
        mutation'."""
        import ghost_agent.sandbox.services as S

        for needle in ("answered by a DIFFERENT process",
                       "likely FAILED to\"\n                f\" bind"):
            pass
        rets = _returns_calling(S, "failed to bind")
        rets += _returns_calling(S, "answered by a DIFFERENT process")
        assert len(rets) >= 2, f"the bind-failure returns moved: {rets}"
        for r in rets:
            assert "ToolOutcome.failed" in r, (
                f"a service that failed to bind reports success: {r[:80]}")

    def test_jobs_reads_are_successes(self):
        """A `jobs` report QUOTES the jobs it describes, including a finished
        job's `EXIT CODE: 1`. The banner rule read that as the `jobs` call
        failing, drawing a strike every time the model re-read it."""
        import ghost_agent.tools.delegate as D
        from ghost_agent.tools.outcome import ToolOutcome

        src = inspect.getsource(D.tool_jobs)
        assert "ToolOutcome.ok" in src and "ToolOutcome.failed" in src
        assert ToolOutcome.ok(
            "--- job-1 [FAILED] ---\nexited 1\nEXIT CODE: 1"
        ).exit_code_failed is False


class TestTheStatusSurvivesEveryDecoration:
    def test_append_note_preserves_declared_rather_than_forcing_it(self):
        """A mutant that forced `declared=True` here survived: nothing
        asserted the flag, only the status."""
        from ghost_agent.tools.outcome import ToolOutcome, append_note

        derived = ToolOutcome.coerce("boom\nEXIT CODE: 127")
        assert derived.declared is False
        out = append_note(derived, "\nnote")
        assert out.declared is False, (
            "decorating a DERIVED result re-labelled it as a producer's "
            "declaration — that makes the sniffer unreachable")
        assert out.status is derived.status
        assert append_note("plain", "\nx") == "plain\nx"

    def test_the_self_play_tool_boundary_keeps_the_status(self):
        """Six declarations in `dream.py` were destroyed one function later
        by ``return f"{result}\\n\\nSYSTEM: SELF PLAY DONE."`` — an f-string
        on a `str` subclass. The same defect as the `+` sites, one boundary
        out, missed by an enumeration scoped to the module being migrated."""
        import ghost_agent.tools.memory as M

        tree = ast.parse(inspect.getsource(M))
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            u = ast.unparse(node.value)
            if not isinstance(node.value, ast.JoinedStr):
                continue
            if "result" in u and ("SELF PLAY" in u or "SESSION FINISHED" in u):
                bad.append(u[:70])
        assert not bad, f"a status is f-string'd away at: {bad}"


class TestTheMeasuringInstruments:
    def test_the_eval_error_count_uses_the_shared_predicate(self):
        """It counted `"ERROR" in result.upper()`: 987 against 387 true
        failures — 2.55x, 654 fabricated error credits — and this number is
        `mean_tool_errors` in the FROZEN regression baseline."""
        import ghost_agent.eval.behavioral as B

        src = inspect.getsource(B)
        assert '"ERROR" in str(t.get("result"' not in src, (
            "the private whole-body substring is back")
        assert "_action_failed" in src, (
            "the eval instrument keeps its own predicate instead of the "
            "shared question")
        assert 't.get("name")' in src, (
            "the tool name is in scope and not passed, so `execute` is "
            "judged by the generic sniffer")

    def test_self_play_scoring_counts_a_refusal_whatever_the_tool(self):
        """287 of 408 failures and 83 of 83 refusals were missed — the
        tool-name exclusion is there so a `file_system` READ of a log
        containing "ERROR:" is not counted, and only the status separates
        that from a REFUSED write. All 9 UNRESOLVED were counted as errors,
        which the third-state contract forbids."""
        from ghost_agent.core.self_play_scoring import count_tool_errors
        from ghost_agent.tools.outcome import ToolOutcome

        def _m(content, name="file_system"):
            return [{"role": "tool", "name": name, "content": content}]

        assert count_tool_errors(_m(ToolOutcome.rejected(
            "SYSTEM INSTRUCTION: forgot 'replace_with'"))) == 1
        assert count_tool_errors(_m(ToolOutcome.failed("Error: nope"))) == 1
        # ⚠ UNRESOLVED deliberately COUNTS here and only here. A reviewer
        # read the third-state contract ("callers must SKIP an unresolved
        # call") as universal; it is advice for a LABELLER, and a reward is
        # not a label — `test_promoted_result_graders` pins the opposite,
        # with the reason: "otherwise an unfinished run scores as a clean one
        # and the reward is computed on work that never happened."
        assert count_tool_errors(_m(ToolOutcome.unresolved(
            "--- COMMAND RESULT --- [SANDBOX JOB PROMOTED]\n"
            "EXIT CODE: 0 (STILL RUNNING, NOT finished)"),
            name="execute")) == 1
        # the prose exclusion still protects a fixture read
        assert count_tool_errors(
            _m("--- ci.log ---\nERROR: something in the log")) == 0

    def test_the_smoke_gate_cannot_pass_on_a_result_that_never_ran(self):
        """`if not m: return None` fails this gate OPEN, and passing marks a
        task DONE unattended: 7 of 8 non-OK shapes passed."""
        import ghost_agent.core.build_gates as G

        src = inspect.getsource(G.smoke_gate)
        tree = ast.parse(src.lstrip())
        # a REACHABLE branch that tests the status and RETURNS — `_st = None`
        # followed by `if False:` keeps the word "status" and passes a
        # substring check
        good = [n for n in ast.walk(tree)
                if isinstance(n, ast.If)
                and ("_st" in ast.unparse(n.test)
                     or "status" in ast.unparse(n.test))
                and ast.unparse(n.test).strip() not in ("False", "None")
                and any(isinstance(x, ast.Return) for x in ast.walk(n))]
        assert good, (
            "the smoke gate has no reachable status branch that can fail the "
            "gate — `if not m: return None` fails it OPEN, and passing marks "
            "a task DONE unattended")


class TestTheReplyAndTheBanner:
    def test_the_final_reply_does_not_call_a_refusal_a_success(self):
        """148 of 408 failures — including 83 of 83 refusals — would be
        announced as "Process finished successfully.", and that string is
        also the verifier's evidence and the recorded reply."""
        import ghost_agent.core.agent as A

        src = inspect.getsource(A.GhostAgent)
        tree = ast.parse(src.lstrip())
        arms = [ast.unparse(n.value) for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and getattr(n.targets[0], "id", None) == "_looks_failed"]
        assert len(arms) >= 2, f"the reply failure check moved: {arms}"
        missing = [a for a in arms if "_fb_declared_bad" not in a]
        assert not missing, (
            "a branch of the reply's failure check still judges a full "
            f"ToolOutcome by a prefix tuple alone: {missing}")

    def test_a_failure_shaped_result_always_gets_a_banner(self):
        """The gate was widened and the TEXT was not: `first_err_line` uses
        an ANCHORED per-line predicate, so 82 of 82 refusals entered the
        branch and emitted nothing."""
        import ghost_agent.core.agent as A

        src = inspect.getsource(A.GhostAgent._dispatch_and_process_tool_batch)
        i = src.index("first_err_line = next(")
        assert "if not first_err_line:" in src[i:i + 900], (
            "a failure-shaped result with no Error-headed line still emits "
            "no banner at all")

    def test_an_in_flight_call_is_not_listed_as_SUCCEEDED(self):
        """`op_outcomes` had no third state, so a detached job was reported
        under the one message that calls itself AUTHORITATIVE as having
        taken effect."""
        from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes

        out = summarize_multi_op_outcomes([
            {"tool": "execute", "ok": True, "unresolved": True},
            {"tool": "file_system", "ok": False, "preview": "nope"},
            {"tool": "manage_tasks", "ok": True},
        ])
        assert "STILL RUNNING" in out, out
        assert "SUCCEEDED: manage_tasks" in out
        assert "SUCCEEDED: execute" not in out


class TestTheStoresReadTheStatusWITHOUTshadowingTheProse:
    def test_a_stored_row_is_judged_the_way_the_LOOP_judges_it(self):
        """Three private prose banks answered this, each wrong in its own
        direction: a five-prefix head bank booked 145 of 291 declared non-OK
        results as successes; replacing it with an `if/else` over the status
        then SHADOWED the bank, so 118 non-zero `execute` exits were still
        stored successful — neither half can see an `EXIT CODE:` banner
        under an `--- EXECUTION RESULT ---` head.

        EXECUTED. Every token-shaped version of this pin was walked through
        by a mutant that kept the tokens and inverted the meaning."""
        from ghost_agent.core.agent import _action_failed
        from ghost_agent.tools.outcome import ToolOutcome

        cases = [
            (("--- EXECUTION RESULT ---\nEXIT CODE: 127", "execute"), True),
            ((ToolOutcome.rejected("SYSTEM BLOCK: aborted"), ""), True),
            ((ToolOutcome.failed("--- BROWSER RESULT ---\nSTATUS: ERROR"),
              "browser"), True),
            ((ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)"), ""),
             False),
            (("--- a.py CONTENTS ---\nimport os", "file_system"), False),
            ((ToolOutcome.ok('{"reason": "... EXIT CODE: 1 ..."}'),
              "manage_projects"), False),
        ]
        for args, want in cases:
            assert _action_failed(*args) is want, (args[0][:50], want)

    def test_both_stored_row_readers_actually_call_it(self):
        """A shared predicate nothing calls is decoration."""
        import ghost_agent.core.agent as A

        tree = ast.parse(inspect.getsource(A))
        callers = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            b = ast.unparse(fn)
            if "_action_failed(" not in b:
                continue
            if '"success": _ok' in b or "'success': _ok" in b:
                callers.add("episode_store")
            if "_ran_info" in b:
                callers.add("ran_info")
        assert callers == {"episode_store", "ran_info"}, (
            f"a stored-row reader stopped asking the shared question: "
            f"{callers}")

    def test_a_canonical_write_failure_is_not_hidden_behind_PARTIAL(self):
        """PARTIAL is exempted from the verifier's bookkeeping gate because
        it means "the canonical write LANDED and only an index lagged". On
        the bus path the same PARTIAL was emitted when the CANONICAL leg
        errored, so a total write failure disarmed the unverified-mutation
        guard and skipped the verifier."""
        from ghost_agent.tools.memory import _bus_canonical_failed

        # Canonicality is PER-OPERATION. A flat list was wrong both ways: it
        # omitted `vector`, which IS the canonical store for `insert_fact`
        # (so a fact that was never stored emitted a guard-exempt PARTIAL),
        # and adding `vector` flatly would make a retrieval-index lag on
        # `update_profile` a total failure.
        assert _bus_canonical_failed(
            {"profile": "error: disk full", "vector": "ok"},
            "update_profile") is True
        assert _bus_canonical_failed(
            {"profile": "ok", "vector": "error: index lag"},
            "update_profile") is False
        assert _bus_canonical_failed(
            {"vector": "error: store down"}, "insert_fact") is True
        assert _bus_canonical_failed(
            {"skill": "error: dropped by the playbook"},
            "learn_skill") is True
        assert _bus_canonical_failed({}, "insert_fact") is False

    def test_compression_does_not_fabricate_a_failure_label(self):
        """All three helpers returned a plain `str` (status destroyed) AND
        the keyword filter hoists an `[error]` line into the sniffer's
        120-char window: 67 corpus labels flipped on 42 turns, 64 of them
        SUCCESS -> FAILED, written into the training corpus."""
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.context_manager import ContextManager
        from ghost_agent.tools.outcome import ToolOutcome

        keep = ContextManager._keep_outcome
        # DERIVED, not `ok(...)`: a declared ok settles the label on its own,
        # so the pin passed with the compressed branch deleted entirely.
        # `browser` successes are bare strings on the live path.
        msg = {"role": "tool", "tool_call_id": "c0", "name": "browser",
               "content": ToolOutcome.coerce(
                   "--- BROWSER RESULT ---\nSTATUS: OK\nrendered")}
        assert msg["content"].declared is False
        out = keep(msg, "--- BROWSER RESULT ---\n[error] console noise\n…")
        assert out["content"].status is msg["content"].status
        assert out["content"].reason_code == "context_compressed"

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "browser", "arguments": "{}"}}]},
            dict(out, tool_call_id="c0"),
        ]
        assert not GhostAgent._reconstruct_tool_calls(msgs)[0].error, (
            "a SUCCESSFUL call acquired a failure label from text the LOOP "
            "rewrote")


class TestTheSystemThreeCounterStepsByOne:
    def test_the_pivot_counter_increments_by_exactly_one(self):
        """A mutant that incremented by 2 survived: nothing pinned the step
        size, and at +2 the second trigger arm (`== 1`) can never match."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        bumps = [ast.unparse(n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and getattr(n.targets[0], "id", None)
                 == "_request_sys3_fired_once"
                 # the ARITHMETIC one, not the `ts.` unpack or the reset
                 # the ARITHMETIC assignment, not the `ts.` unpack or the
                 # per-request reset
                 and isinstance(n.value, ast.BinOp)]
        assert bumps, "the pivot counter moved"
        for b in bumps:
            v = eval(b, {}, {"_request_sys3_fired_once": 0})
            assert v == 1, (
                f"the pivot counter steps by {v}, so the second trigger arm "
                f"(== 1) can never match: {b}")


class TestTheThirdStateOffline:
    def test_the_offline_reader_sees_a_still_running_swarm_by_TEXT(self):
        """The exemption read `.status`, but the seeder iterates rows loaded
        from JSONL where `result` is a plain `str` — so it was dead on
        exactly the data it exists for."""
        from ghost_agent.distill.outcome_heuristics import (
            is_unresolved_tool_result)

        text = ("PARTIAL: 2/4 task(s) completed; 2 still running in the "
                "background (t3, t4). They were NOT cancelled — do not "
                "re-dispatch them.")
        assert is_unresolved_tool_result(text) is True
        assert is_unresolved_tool_result("SUCCESS: all done") is False

    def test_the_foresight_seed_does_not_fail_it_either(self):
        from ghost_agent.core.foresight import offline_call_failed

        class _TC:
            def __init__(self, r):
                self.result = r
                self.error = ""

        assert offline_call_failed(_TC(
            "PARTIAL: 2/4 completed; 2 still running in the background "
            "(t3, t4). They were NOT cancelled")) is False
