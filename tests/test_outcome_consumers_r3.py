"""The CONSUMERS read the status — round 3 (§4DP).

WHY THIS FILE EXISTS
====================
§4DO migrated the PRODUCERS and pinned the dispatch loop. A reviewer then
replayed all 4,391 recorded live tool calls through every "did this fail?"
reader in the tree, independently, and asked of each: *how many of the 82
corpus refusals does THIS one still miss?* The answer was that the status was
minted in `tools/` and died at the dispatch loop — nothing downstream read
it, and two of the readers that missed 82 out of 82 sit on the UNATTENDED
path, where a refused edit marks its task DONE with nothing done.

Every test here pins one consumer against the status. They are all
ADD-only rules: an `ok` status never suppresses the evidence a text rule
already had (that inversion silently killed the exit-127 detection once
already — measured at -198 `execute` failures).
"""

import ast
import inspect

import pytest


# ─────────────────────────────────────────────── the declared/derived line

class TestADeclaredStatusIsNotReSniffed:
    def test_a_quoted_exit_banner_does_not_fail_a_successful_read(self):
        """`manage_projects` returns the project ledger as JSON, and a
        stored `autoadvance_failed` event quotes the failing tool's
        `--- EXECUTION RESULT --- EXIT CODE: 1` verbatim. The banner rule
        found that quote and booked the READ as a failed shell command: 4
        live rows, all successes, reported to the model under the
        AUTHORITATIVE multi-step banner and recorded as incompetence.

        Position cannot separate the two — a JSON payload is one line, so
        "the banner heads the result" is true for a real envelope AND for a
        quote. Only the producer can, so the producer says it.
        """
        from ghost_agent.tools.outcome import ToolOutcome

        payload = (
            '{"project": {"id": "f36f04d446a6", "title": "TinyAI", '
            '"events": [{"type": "autoadvance_failed", "payload": '
            '{"reason": "verify failed: --- EXECUTION RESULT --- '
            'EXIT CODE: 1 ..."}}]}}'
        )
        assert ToolOutcome.ok(payload).exit_code_failed is False
        # ...and an UNMIGRATED tool's identical text still gets the banner
        # rule, because nobody has answered the question for it.
        assert ToolOutcome.coerce(payload).exit_code_failed is True

    def test_manage_projects_declares_its_successes(self):
        from ghost_agent.tools.outcome import OutcomeStatus
        from ghost_agent.tools.projects import _ok

        out = _ok({"switched_to": "30d5d5b65c38",
                   "note": "EXIT CODE: 1 quoted from an old event"})
        assert out.status is OutcomeStatus.OK
        assert out.declared is True
        assert out.exit_code_failed is False

    def test_a_shell_cannot_be_talked_out_of_its_exit_code(self):
        """The `declared` short-circuit is deliberately NOT in
        `shell_failed`: that question is asked only of `execute`, and there
        the banner is evidence, not a claim."""
        from ghost_agent.tools.outcome import ToolOutcome

        assert ToolOutcome.ok("boom\nEXIT CODE: 127").shell_failed is True


# ───────────────────────────────────────────────────── unattended consumers

class TestTheUnattendedPathReadsTheStatus:
    def test_the_autoadvancer_does_not_mark_a_refused_edit_DONE(self):
        """`project_advancer._looks_like_failure` is what lets an idle tick
        call `update_status(..., DONE)`. It did `str(output)`, so all 82
        corpus refusals reached it as clean successes — unattended."""
        from ghost_agent.core.project_advancer import _looks_like_failure
        from ghost_agent.tools.outcome import ToolOutcome

        refusal = ToolOutcome.rejected(
            "SYSTEM INSTRUCTION: You used operation='replace' but forgot to "
            "specify 'replace_with'. The file was NOT modified.")
        assert _looks_like_failure(refusal) is True
        # an unfinished job is not evidence of success either
        assert _looks_like_failure(
            ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)")) is True
        # ADD-only: an ok status does not suppress the banner rule
        assert _looks_like_failure(
            ToolOutcome.ok("--- EXECUTION RESULT ---\nEXIT CODE: 1")) is True
        assert _looks_like_failure(
            ToolOutcome.ok("--- EXECUTION RESULT ---\nEXIT CODE: 0")) is False

    def test_skill_graduation_does_not_count_a_refusal_as_a_success(self):
        """`_acquired_skill_result_class` feeds the graduation
        `success_rate` — the number that decides whether a skill is kept."""
        from ghost_agent.tools.outcome import ToolOutcome
        from ghost_agent.tools.registry import _acquired_skill_result_class

        assert _acquired_skill_result_class(
            ToolOutcome.rejected("SYSTEM BLOCK: aborted before execution")
        ) == "fail"
        # no verdict yet ⇒ "infra", which callers skip telemetry on
        assert _acquired_skill_result_class(
            ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)")) == "infra"
        # ADD-only
        assert _acquired_skill_result_class(
            ToolOutcome.ok("--- EXECUTION RESULT ---\nEXIT CODE: 1")) == "fail"
        assert _acquired_skill_result_class(
            ToolOutcome.ok("--- EXECUTION RESULT ---\nEXIT CODE: 0")) == "ok"

    def test_the_verifier_high_stakes_gate_reads_the_status(self):
        """`_turn_had_tool_failure` gates the verifier's CONFIRM escalation.
        It did `str(tool.get("content"))`, throwing away the status of a
        `ToolOutcome` it was handed — the same defect
        `_reconstruct_tool_calls` was fixed for, in the same file, over the
        same `tools_run_this_turn` list. Measured: 15 turns, 8% of the true
        high-stakes population, where the escalation should fire and did
        not."""
        from ghost_agent.core.agent import _turn_had_tool_failure
        from ghost_agent.tools.outcome import ToolOutcome

        refusal = ToolOutcome.rejected(
            "SYSTEM INSTRUCTION: REPLACE REJECTED — the file was NOT modified.")
        assert _turn_had_tool_failure(
            [{"role": "tool", "content": refusal}]) is True
        # a SYSTEM BLOCK has no marker in the sniffer at all
        assert _turn_had_tool_failure([{"role": "tool", "content":
            ToolOutcome.rejected("SYSTEM BLOCK — pre-flight guard: …")}]) is True
        # ADD-only: the sniffer's own evidence survives an ok status
        assert _turn_had_tool_failure([{"role": "tool", "content":
            ToolOutcome.ok("xxd: not found\nEXIT CODE: 127")}]) is True
        # UNRESOLVED is not a verdict
        assert _turn_had_tool_failure([{"role": "tool", "content":
            ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)")}]) is False
        # and a plain success is still a success
        assert _turn_had_tool_failure(
            [{"role": "tool", "content": "SUCCESS: stored"}]) is False

    def test_the_coding_executor_does_not_mark_a_refused_write_applied(self):
        """The FIFTH reader of "did this write land?", and the only one the
        first sweep missed — four siblings were migrated and this was not,
        which is the round-over-round pattern this work keeps reproducing.

        36 live refusals slipped through on the typed path (14 `REJECTED:
        that replace would introduce a syntax error`, 11 pre-flight guard
        blocks, 5 empty-write blocks, 4 rejected SQL). The caller then does
        `touched.add(path)` and advances the task — unattended — with the
        file never written."""
        from ghost_agent.core.coding_executor import (
            _looks_like_write_error, _op_ok)
        from ghost_agent.tools.outcome import ToolOutcome

        for text in (
            "REJECTED: that replace would introduce a syntax error",
            "SYSTEM BLOCK — pre-flight guard: this exact call",
            "SYSTEM BLOCK: SQL statement rejected by the validator",
            "SYSTEM BLOCK: You invoked file_system operation='write' ...",
        ):
            refusal = ToolOutcome.rejected(text)
            assert _looks_like_write_error(refusal) is True, text
            assert _op_ok(refusal) is False, text

        # a write that LANDED but does not parse is not a write ERROR — the
        # file did change, and the syntax diagnostic is what reports it
        landed = ToolOutcome.partial("SUCCESS: Wrote 10 chars to 'a.py'.",
                                     world_changed=True)
        assert _looks_like_write_error(landed) is False
        assert _op_ok(landed) is True

        # ADD-only: plain strings behave exactly as before
        assert _looks_like_write_error("SUCCESS: Wrote 10 chars") is False
        assert _op_ok("SUCCESS: Wrote 10 chars") is True
        assert _looks_like_write_error("SYSTEM INSTRUCTION: not found") is True

    def test_a_composed_skill_step_sees_the_search_module_head(self):
        """`CRITICAL ERROR:` was added to `_FAILURE_PREFIX_RE` and to none
        of the five other prefix banks, so a `deep_research` that never ran
        scored a composed-skill step SUCCESS."""
        from ghost_agent.tools.composed_skills import _step_result_ok
        from ghost_agent.tools.outcome import ToolOutcome

        assert _step_result_ok("CRITICAL ERROR: no research was run") is False
        assert _step_result_ok(
            ToolOutcome.rejected("SYSTEM INSTRUCTION: bad args")) is False
        assert _step_result_ok("SUCCESS: done") is True

    def test_the_shared_sniffer_reads_the_status_too(self):
        """One place closes the remaining callers — `tool_failure_flags`,
        the browser-navigate gate and the eval harness all route through
        it. ADD-only: preferring the status here once cost -198 `execute`
        failures."""
        from ghost_agent.distill.outcome_heuristics import (
            _looks_like_tool_error)
        from ghost_agent.tools.outcome import ToolOutcome

        assert _looks_like_tool_error(
            ToolOutcome.rejected("SYSTEM BLOCK: aborted")) is True
        # ...and the prose evidence is untouched by an ok status
        assert _looks_like_tool_error(
            ToolOutcome.ok("boom\nEXIT CODE: 127")) is True
        # UNRESOLVED is not a verdict
        assert _looks_like_tool_error(
            ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)")) is False
        assert _looks_like_tool_error("SUCCESS: stored") is False

    def test_the_verify_gate_can_never_PASS_on_a_refusal(self):
        """`classify_verify_result` is what lets a tick mark a task DONE.
        It is fail-closed on prose; it must be fail-closed on the status."""
        from ghost_agent.core.project_advancer import classify_verify_result
        from ghost_agent.tools.outcome import ToolOutcome

        assert classify_verify_result(
            ToolOutcome.rejected("SYSTEM BLOCK: nothing ran")) != "pass"
        assert classify_verify_result(
            ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)")) != "pass"
        assert classify_verify_result(
            ToolOutcome.failed("--- EXECUTION RESULT ---\nEXIT CODE: 0")
        ) != "pass", "a FAILED status read as a pass because it exited 0"
        assert classify_verify_result(
            "--- EXECUTION RESULT ---\nEXIT CODE: 0\nall good") == "pass"

    def test_every_result_classifier_reads_the_status(self):
        """THE ANTI-PATTERN PIN.

        Four readers were migrated in one pass and a fifth
        (`coding_executor._looks_like_write_error`, 36 live refusals, the
        unattended path) was missed — which is the round-over-round pattern
        this whole body of work keeps reproducing. Enumerate them instead of
        remembering them.

        Two things this pin got wrong before, both found by review:
        it walked a RELATIVE `src/ghost_agent` (0 files and a vacuous pass
        from any cwd but the repo root), and its heuristics reached 5 of the
        13 documented readers — `"ok("` can never match an identifier.

        DELEGATION is detected, not listed: a wrapper that forwards to a
        checked classifier is covered by it. A hand-written exempt list
        guesses at someone else's dependency and goes stale.
        """
        import os
        from pathlib import Path

        pkg = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
        assert pkg.is_dir(), pkg

        hint = ("fail", "error", "_ok", "success", "reject", "classif",
                "_class", "verdict", "unresolved", "informational",
                "high_stakes")
        result_args = {"out", "output", "result", "result_str", "res", "body",
                       "content", "tc", "evidence", "tool_result"}
        # These do not answer "did a TOOL CALL fail" at all: they parse an
        # LLM verdict, or build a grouping key from an error that has
        # already been classified.
        not_about_tool_results = {
            "_parse_verdict", "_verdict_score_probe", "_error_key",
            "_syntax_fail_reason", "_replace_failure_kind",
            "_log_verify_outcome", "_normalize_tool_error",
        }

        found = {}          # name -> (path, lineno, body)
        for root, _, files in os.walk(pkg):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                try:
                    tree = ast.parse(open(path, encoding="utf-8").read())
                except SyntaxError:
                    continue
                for n in ast.walk(tree):
                    if not isinstance(n, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                        continue
                    if n.name in not_about_tool_results:
                        continue
                    if not {a.arg for a in n.args.args} & result_args:
                        continue
                    if not any(h in n.name.lower() for h in hint):
                        continue
                    found[n.name] = (path, n.lineno, ast.unparse(n))

        assert len(found) >= 12, (
            f"the enumeration only found {len(found)} classifiers — it is "
            "passing because it scanned almost nothing, which is exactly "
            "what a broken scanner looks like")

        def _reads_status(body):
            return ".status" in body or "'status'" in body

        direct = {k for k, (_, _, b) in found.items() if _reads_status(b)}
        # transitive closure over delegation
        changed = True
        while changed:
            changed = False
            for name, (_, _, body) in found.items():
                if name in direct:
                    continue
                if any(f"{d}(" in body for d in direct):
                    direct.add(name)
                    changed = True

        missing = [f"{found[n][0]}:{found[n][1]} {n}"
                   for n in sorted(found) if n not in direct]
        assert not missing, (
            "result classifiers that judge a tool result by its prose alone, "
            "and do not delegate to one that reads the status:\n  "
            + "\n  ".join(missing))

    def test_the_batch_classifier_sees_it_too(self):
        from ghost_agent.tools.file_system import _batch_result_failed

        assert _batch_result_failed("CRITICAL ERROR: gone") is True
        assert _batch_result_failed("SUCCESS: wrote 10 chars") is False


# ─────────────────────────────────────────────────── newly-migrated writers

class TestTheNewlyMigratedProducers:
    def test_a_browser_error_is_a_failure_to_the_loop(self):
        """`--- BROWSER RESULT ---` heads the string, so the loop's anchored
        failure-prefix rule never matched: 42 live rows over 32 turns, 0/42
        booked as failures — no strike, no guard record, no competence
        signal, and `STATUS: ERROR` reported to the model as a SUCCEEDED
        operation. Largest remaining loop-vs-corpus disagreement class."""
        import ghost_agent.tools.browser as B
        from ghost_agent.tools.outcome import ToolOutcome

        src = inspect.getsource(B.tool_browser)
        tree = ast.parse(src.lstrip())
        errs = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_err"]
        assert errs, "browser's error helper moved — re-point this pin"
        # The RETURN, not the function body: `_err` imports ToolOutcome on
        # its first line, so "is ToolOutcome mentioned anywhere in here"
        # stayed true with `return out` — this pin survived the exact
        # mutant it exists to catch.
        rets = [n for n in ast.walk(errs[0]) if isinstance(n, ast.Return)]
        assert rets, "_err returns nothing?"
        for r in rets:
            u = ast.unparse(r.value) if r.value is not None else ""
            assert "ToolOutcome." in u, (
                f"browser errors are bare strings again ({u[:60]!r}); the "
                "loop cannot see them — 42 live rows booked as SUCCEEDED"
            )
        # and the shape it produces is a failure that changed nothing
        probe = ToolOutcome.failed(
            "--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed",
            world_changed=False)
        assert probe.is_failure and probe.changed_the_world is False

    def test_a_fact_check_that_could_not_verify_is_partial(self):
        """`FACT CHECK PARTIAL:` is not a head any predicate in this tree
        recognised — a half-answer was a clean success."""
        import ghost_agent.tools.search as S
        from ghost_agent.tools.outcome import ToolOutcome

        src = inspect.getsource(S)
        assert src.count("FACT CHECK PARTIAL") >= 2
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Return) and node.value is not None:
                u = ast.unparse(node.value)
                if "FACT CHECK PARTIAL" in u:
                    assert "ToolOutcome.partial" in u, (
                        f"a FACT CHECK PARTIAL is still a bare string: {u[:80]}"
                    )
        # nothing in the tree could see it before
        assert ToolOutcome.coerce(
            "FACT CHECK PARTIAL: the verifier returned no text"
        ).is_failure is False, (
            "if this now classifies by TEXT, the status is no longer what "
            "is carrying the distinction — re-read the test above"
        )

    def test_the_dead_return_below_a_return_is_gone(self):
        """`search.py` had a `return "SYSTEM ERROR: …"` directly after an
        unconditional `return` — the one head in that function any predicate
        recognised was unreachable."""
        import ghost_agent.tools.search as S

        tree = ast.parse(inspect.getsource(S))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for block in [fn.body]:
                for k, stmt in enumerate(block[:-1]):
                    if isinstance(stmt, ast.Return):
                        nxt = block[k + 1]
                        assert not isinstance(nxt, ast.Return), (
                            f"{fn.name}: unreachable return at line "
                            f"{nxt.lineno}"
                        )

    def test_swarm_still_running_is_unresolved_not_partial(self):
        """PARTIAL is a failure status, so it drew a strike and fed the
        same-failure breaker for tasks the message itself says "were NOT
        cancelled — do not re-dispatch them"."""
        import ghost_agent.tools.swarm as SW

        src = inspect.getsource(SW)
        assert "swarm_await_still_running" in src, (
            "the still-running branch is a PARTIAL again — that is a "
            "failure verdict for work explicitly still in flight"
        )
        assert "ToolOutcome.unresolved" in src

    @pytest.mark.asyncio
    async def test_a_write_that_lands_but_does_not_parse_is_partial(
            self, tmp_path):
        """"SUCCESS: Wrote …" headed the string, so every classifier in the
        tree read it as a clean success — 11 live rows — and the only thing
        that noticed was an ELEVENTH private sniffer in coding_executor."""
        from ghost_agent.tools.file_system import tool_write_file
        from ghost_agent.tools.outcome import OutcomeStatus

        res = await tool_write_file("broken.py", "def f(:\n", tmp_path)
        assert getattr(res, "status", None) is OutcomeStatus.PARTIAL, (
            f"a written-but-unparseable file is still a SUCCESS: {res[:120]!r}"
        )
        assert res.changed_the_world is True, "the write DID land"
        assert res.may_record_as_applied is False

        ok = await tool_write_file("fine.py", "def f():\n    return 1\n",
                                   tmp_path)
        assert getattr(ok, "status", OutcomeStatus.OK) is OutcomeStatus.OK

    def test_the_pre_dispatch_refusals_carry_a_status(self):
        """22 live `SYSTEM BLOCK …` rows reached the trajectory corpus as
        clean SUCCESSES: `_looks_like_tool_error` has no marker for them and
        there was no status to read. Eleven of the twelve `{"role": "tool"}`
        append sites in the dispatch method were bare strings."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        bare = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant)]
            if "role" not in keys or "content" not in keys:
                continue
            roles = [v.value for k, v in zip(node.keys, node.values)
                     if isinstance(k, ast.Constant) and k.value == "role"
                     and isinstance(v, ast.Constant)]
            if roles != ["tool"]:
                continue
            content = [v for k, v in zip(node.keys, node.values)
                       if isinstance(k, ast.Constant) and k.value == "content"]
            u = ast.unparse(content[0]) if content else ""
            if "ToolOutcome" not in u and "_TO." not in u:
                bare.append((node.lineno, u[:70]))
        assert not bare, (
            "pre-dispatch tool messages that carry no status:\n"
            + "\n".join(f"  line {ln}: {u}" for ln, u in bare)
        )


# ──────────────────────────────────────────────── previously-unpinned rules

class TestTheRulesNobodyHadPinned:
    def test_critical_error_is_a_failure_prefix(self):
        """Round-2 finding #7 shipped with no test anywhere: deleting
        `CRITICAL ERROR:` from `_FAILURE_PREFIX_RE` kept the suite green."""
        from ghost_agent.tools.tool_failure import result_is_failure

        assert result_is_failure("CRITICAL ERROR: research produced nothing")
        assert result_is_failure("  CRITICAL ERROR: leading space")

    def test_the_shell_prose_rule_is_asked_only_of_the_shell(self):
        """Applying `shell_failed`'s unanchored marker fallback to every
        tool booked 226 live successes as incompetence. The narrowing is the
        fix; this pins that it is still narrow."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        found = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None)
                    == "_op_shell_failed"):
                u = ast.unparse(node.value)
                found = True
                assert "shell_failed" in u and "exit_code_failed" in u, u
                assert "execute" in u, (
                    "the shell prose rule is no longer scoped to the shell: "
                    + u
                )
        assert found, "_op_shell_failed moved — re-point this pin"

    def test_a_world_changing_failure_is_not_credited_as_untouched(self):
        """`world_changed=True` exists for exactly one site — a truncating
        write that then hit ENOSPC, reproduced under RLIMIT_FSIZE: 5,434
        bytes in, 8,192 bytes of a different file out, original gone. It had
        no test: ignoring the override kept the suite green."""
        from ghost_agent.tools.outcome import ToolOutcome

        half = ToolOutcome.failed("Error: ENOSPC mid-write", world_changed=True)
        assert half.is_failure is True
        assert half.changed_the_world is True, (
            "a half-landed write claims it touched nothing — the pre-flight "
            "guard would then record 're-running this will fail the same "
            "way' for a file that has already been truncated"
        )
        assert half.may_record_as_applied is False
        # the default for a plain failure is still 'touched nothing'
        assert ToolOutcome.failed("Error: no such file").changed_the_world is False

    def test_an_exit_code_failure_is_named_in_the_multi_step_preview(self):
        """`preview` was gated on `_res_is_error` alone, so a non-zero exit
        was listed under FAILED with the bare word "failed" and no text."""
        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        previews = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value == "preview":
                        previews.append(ast.unparse(v))
        assert previews, "op_outcomes preview moved — re-point this pin"
        assert any("_op_shell_failed" in p for p in previews), (
            "the preview is gated on the error verdict alone again: a "
            "non-zero exit gets named to the model as the bare word 'failed'"
        )
