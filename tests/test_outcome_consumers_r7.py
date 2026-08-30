"""Round-7 (§4DT): the defects live where two modules meet.

Two lenses. One attacked round 6's fixes; the other audited every DECLARATION
in the tree — a surface no earlier round had looked at, and one that only
became dangerous because rounds 3-6 quintupled the number of declaring
producers. A *wrong* declaration is strictly worse than none: a declared `ok`
short-circuits the banner rule, the corpus sniffer and several guards.

The reviewer's closing observation is the organising idea here: **every defect
sat at the EDGE of a migration** — the last `return` of a dispatcher, the
wrapper one boundary out from the declared producer, the flag set one line
after the awaited call it guards, the `+=` one function later. The migrations
were scoped to the module being changed; the defects live between modules.
"""

import ast
import inspect
import os
from pathlib import Path

from tests.test_outcome_consumers_r4 import _loop_expr

PKG = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


def _iter_py():
    for root, _, files in os.walk(PKG):
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


class TestNothingDecoratesAStatusAway:
    """`ToolOutcome` is a `str` subclass, so `res + note`, ``f"{res}…"`` and
    `str(res).strip()` all go through `str` and return a plain `str`. Found
    at NINE sites in six modules across four rounds — three of them AFTER
    the same defect had been fixed one boundary away."""

    def test_the_tree_wide_sweep_that_the_journal_claimed_existed(self):
        """⚠ §4DS said "there is one `append_note` now, and a pin that walks
        every module." That was FALSE: the pin walked ONE module and matched
        two literal phrases. A reviewer's real sweep found five live sites it
        could not see. This is the sweep.

        For every function that can return a `ToolOutcome`, no local bound
        from that value may then be decorated with `+`/`+=`/an f-string and
        returned.
        """
        offenders = []
        for path in _iter_py():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # locals bound from something that yields an outcome
                tainted = set()
                for n in ast.walk(fn):
                    if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                        continue
                    val = getattr(n, "value", None)
                    if val is None:
                        continue
                    u = ast.unparse(val)
                    if not any(k in u for k in (
                            "ToolOutcome", "_write_replace_guarded",
                            "_format_error", "_promoted_result",
                            "_declare(", "dreamer.dream", "_ok(")):
                        continue
                    tgts = (n.targets if isinstance(n, ast.Assign)
                            else [n.target])
                    for t in tgts:
                        if isinstance(t, ast.Name):
                            tainted.add(t.id)
                if not tainted:
                    continue
                for n in ast.walk(fn):
                    bad = None
                    if (isinstance(n, ast.AugAssign)
                            and isinstance(n.target, ast.Name)
                            and n.target.id in tainted
                            and isinstance(n.op, ast.Add)):
                        bad = ast.unparse(n)
                    elif isinstance(n, ast.JoinedStr):
                        u = ast.unparse(n)
                        if any("{" + t in u or "{" + t + "}" in u
                               for t in tainted):
                            bad = u
                    if bad:
                        offenders.append(
                            f"{path.relative_to(PKG)}:{n.lineno} "
                            f"{fn.name}: {bad[:60]}")
        assert not offenders, (
            "a declared status is decorated away (use `outcome.append_note` "
            "or `outcome.with_text`):\n  " + "\n  ".join(sorted(set(offenders))))

    def test_with_text_keeps_everything_including_provenance(self):
        from ghost_agent.tools.outcome import (OutcomeStatus, ToolOutcome,
                                               with_text)

        d = ToolOutcome.coerce("boom\nEXIT CODE: 127")
        out = with_text(d, "shorter")
        assert out.status is d.status and out.declared is False
        p = ToolOutcome.partial("SUCCESS: wrote", world_changed=True)
        assert with_text(p, "x").world_changed is True
        assert with_text(p, "x").status is OutcomeStatus.PARTIAL
        assert with_text("plain", "y") == "y"
        assert not isinstance(with_text("plain", "y"), ToolOutcome)


class TestTheCorpusAndTheContextPath:
    def test_a_compressed_failure_keeps_its_label(self):
        """The compressed arm was status-ONLY, and `execute` never declares —
        while its `EXIT CODE:` banner SURVIVES compression (the summariser
        always keeps the first three lines; the banner is line 2). Measured:
        the arm removed 83 false positives and created 32 false negatives, 31
        of them `execute` — the loop-vs-corpus split rebuilt sign-flipped
        inside the fix meant to close it."""
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.context_manager import ContextManager
        from ghost_agent.tools.outcome import ToolOutcome

        raw = ("--- EXECUTION RESULT ---\nEXIT CODE: 127\nSTDOUT/STDERR:\n"
               + "noise\n" * 400)
        msg = {"role": "tool", "tool_call_id": "c0", "name": "execute",
               "content": ToolOutcome.coerce(raw)}
        small = ContextManager._keep_outcome(msg, "\n".join(
            raw.split("\n")[:3] + ["[... compressed]"]))
        assert small["content"].reason_code == "context_compressed"
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "execute", "arguments": "{}"}}]},
            dict(small, tool_call_id="c0"),
        ]
        assert GhostAgent._reconstruct_tool_calls(msgs)[0].error, (
            "a command that exited 127 reached the corpus as a SUCCESS "
            "because compression suppressed the whole sniffer")

    def test_in_place_truncation_keeps_the_verdict(self):
        """`_cut_message` mutates the SAME dicts `tools_run_this_turn` holds,
        so every declared verdict was destroyed in place under history
        pressure — 8 of 10 statuses lost, in both directions."""
        import ghost_agent.core.agent as A

        # the ENCLOSING function, resolved from the AST — a fixed character
        # window around `_cut_str` does not reach its call sites
        tree = ast.parse(inspect.getsource(A.GhostAgent).lstrip())
        owner = next((f for f in ast.walk(tree)
                      if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and any(isinstance(g, ast.FunctionDef)
                              and g.name == "_cut_str"
                              for g in ast.walk(f))
                      and f.name != "_cut_str"), None)
        assert owner is not None, "the oversized-tail truncation moved"
        body = ast.unparse(owner)
        # Only assignments to the MESSAGE content: `best["text"]` truncates a
        # dict element inside block-style content, where the outer value is a
        # list and no status is at risk.
        assigns = [ast.unparse(n) for n in ast.walk(owner)
                   if isinstance(n, ast.Assign)
                   and "_cut_str(" in ast.unparse(n.value)
                   and any(ast.unparse(t).endswith(("['content']",
                                                    '["content"]'))
                           for t in n.targets)]
        assert assigns, "no message-content truncation found"
        for a in assigns:
            assert "_keep(" in a, (
                "the oversized-tail truncation writes a plain str back over "
                f"a declared outcome — and it mutates the SAME dicts "
                f"`tools_run_this_turn` holds: {a[:70]}")

    def test_the_deterministic_fast_path_keeps_the_status(self):
        """`memory.py` was migrated to `append_note` so `dream.py`'s six
        declared failures survive; they died 90 lines away in a fast path
        that `str()`s the answer AND bypasses the dispatch loop entirely."""
        import ghost_agent.core.agent as A

        tree = ast.parse(inspect.getsource(A.GhostAgent).lstrip())
        assigns = [ast.unparse(n.value) for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and getattr(n.targets[0], "id", None) == "_det_raw"]
        assert assigns, "the deterministic dispatch moved"
        assert any("_wt(" in a or "with_text(" in a for a in assigns), (
            "the deterministic dispatch stringifies the status away — and "
            f"this path BYPASSES the loop entirely: {assigns}")


class TestTheBannerActuallyEmits:
    def test_a_failure_shaped_result_gets_a_banner(self):
        """Round 6 added a fallback and then always declined to use it: the
        de-duplication guard asks "is this text already visible", which is
        trivially true of a HEAD line. 382 of 400 failures still reached the
        model unmarked."""
        import ghost_agent.core.agent as A

        src = inspect.getsource(A.GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        # the `if` that actually PREPENDS the banner must accept the
        # fallback, not just the de-duplication test
        gates = [ast.unparse(n.test) for n in ast.walk(tree)
                 if isinstance(n, ast.If)
                 and "[FAILURE BANNER]" in ast.unparse(n.body)]
        assert gates, "the banner emit site moved"
        assert any("_banner_is_label" in g for g in gates), (
            "the fallback computes a banner it can never emit: the "
            "de-duplication guard asks whether the text is already visible, "
            f"which is trivially true of a HEAD line — {gates}")


class TestEveryDeclarationIsTheRightOne:
    def test_the_dispatcher_tail_is_a_refusal(self):
        """Eight sibling argument refusals were migrated; the last `return`
        of the dispatcher was not, so a malformed call coerced to OK — the
        model told it SUCCEEDED under the AUTHORITATIVE banner, with a
        metacog competence success, for a call that did nothing. 9 live rows
        on the highest-traffic tool in the corpus."""
        from ghost_agent.tools.outcome import OutcomeStatus
        import ghost_agent.tools.file_system as F

        rets = [ast.unparse(n.value) for n in ast.walk(
            ast.parse(inspect.getsource(F)))
            if isinstance(n, ast.Return) and n.value is not None
            and "Unknown operation" in ast.unparse(n.value)]
        assert rets, "the dispatcher tail moved"
        for r in rets:
            assert "ToolOutcome.rejected" in r, r
        assert OutcomeStatus.REJECTED

    def test_the_truncating_write_flag_is_armed_before_the_write(self):
        """`Path.write_text` opens with 'w' — truncate first, write second —
        so an OSError raised BY the write left the flag False and took the
        "nothing was touched" arm: a DECLARED rejection over a half-written
        file, with the `failed(world_changed=True)` arm written for exactly
        that case unreachable."""
        import ghost_agent.tools.file_system as F

        src = inspect.getsource(F)
        i = src.index("_wrote = True")
        j = src.index("path.write_text", i - 400)
        assert i < j, (
            "`_wrote` is armed AFTER the call that truncates the file")

    def test_a_rolled_back_replace_does_not_claim_a_mutation(self):
        """The guard can roll the whole edit back and return REJECTED;
        relabelling that PARTIAL with `world_changed=True` fires
        `strikes.note_world_changed()` and wipes the loop-breaker's memory on
        a call that touched nothing."""
        import ghost_agent.tools.file_system as F

        src = inspect.getsource(F)
        i = src.index("some_replace_blocks_failed")
        window = src[i - 900:i + 200]
        assert "is_rejection" in window, (
            "a rolled-back multi-block replace still declares a mutation")

    def test_the_supervisor_declares_its_launch_failures(self):
        """Three paths spawn a process and then fail; every other `Error:`
        return is a validation refusal that launched nothing. Declaring the
        three is what lets the wrapper treat the rest as refusals."""
        import ghost_agent.sandbox.services as S

        src = inspect.getsource(S)
        for needle in ("service_launch_failed", "service_pid_unknown",
                       "service_exited_immediately", "service_failed_to_bind",
                       "service_port_hijacked"):
            assert needle in src, f"{needle} declaration missing"

    def test_browser_refusals_and_post_execution_failures(self):
        import ghost_agent.tools.browser as B

        src = inspect.getsource(B.tool_browser)
        # every pre-execution refusal goes through _reject
        for needle in ("_reject(f\"Refused navigation", "_reject(f\"Refused goto",
                       "_reject(str(ve))"):
            assert needle in src, f"a refusal still arms the guard: {needle}"
        # ...and _err can say the runner already ran
        assert "world_changed=ran" in src, (
            "`_err` declares 'nothing changed' for failures that happen "
            "AFTER the runner navigated, clicked and filled")
        assert "ran=True" in src
        # interact reports what actually happened
        tree = ast.parse(src.lstrip())
        # the TEST, not the body: the guard is
        # `if _interact_status == "ok": return _txt`, whose body never
        # mentions the variable at all.
        gates = [ast.unparse(n.test) for n in ast.walk(tree)
                 if isinstance(n, ast.If)]
        assert any("_interact_status" in g for g in gates), (
            "the interact status is computed and never consulted — the "
            "envelope says STATUS: OK whatever the per-action results")

    def test_manage_projects_does_not_declare_a_held_update_a_success(self):
        import ghost_agent.tools.projects as P

        from ghost_agent.tools.outcome import OutcomeStatus
        from ghost_agent.tools.projects import _err, _ok

        held = _ok({"updated": [], "count": 0,
                    "gated_constraints": ["c1"],
                    "agent_instruction_constraints": "Held 2 task(s)"})
        assert held.status is OutcomeStatus.REJECTED, (
            "a task_update that landed NOTHING is declared a success, and a "
            "declared ok settles the corpus label")
        assert _ok({"updated": ["t1"], "count": 1}).status is OutcomeStatus.OK
        assert _err("no such project").status is OutcomeStatus.REJECTED, (
            "the success path declares and the error path returns a bare "
            "string — a split inside one tool")

    def test_the_skill_leg_reads_what_learn_lesson_returned(self):
        """`learn_lesson` returns None on every drop path and catches its own
        exceptions, so discarding the return made the leg report ok whenever
        nothing escaped — and the caller's failure branch was dead code that
        looked like coverage."""
        import ghost_agent.core.bus as BUS

        src = inspect.getsource(BUS)
        i = src.index("async def _skill()")
        window = src[i:i + 1400]
        assert "_written" in window and "if _written" in window, (
            "the skill leg discards `learn_lesson`'s answer")

    def test_the_enveloped_execute_refusals_declare(self):
        """`_format_error` wraps a refusal in `--- EXECUTION RESULT ---`, and
        the rejection predicate is ANCHORED, so the SYSTEM BLOCK head was
        unreachable — the metacog exemption whose comment names this exact
        case could not fire. 27 live rows."""
        import ghost_agent.tools.execute as E

        src = inspect.getsource(E)
        for needle in ("inline_shell_form_rejected", "shell_command_rejected"):
            assert needle in src, needle

    def test_no_skill_creation_failure_is_a_bare_string(self):
        """Eight `Skill creation failed:` heads match no predicate in the
        tree, and two also f-string a `ToolOutcome` away. A token check
        passes while any ONE of them is reverted."""
        import ghost_agent.tools.acquired_skills as AS

        # ⚠ the STATUS, not the type. "does it mention ToolOutcome" is the
        # token check that has now been walked through by a `.failed(` ->
        # `.ok(` mutant in four separate rounds.
        bad = []
        for n in ast.walk(ast.parse(inspect.getsource(AS))):
            if not isinstance(n, ast.Return) or n.value is None:
                continue
            u = ast.unparse(n.value)
            if "Skill creation failed" not in u:
                continue
            if not ("ToolOutcome.failed" in u or "ToolOutcome.rejected" in u):
                bad.append(f"line {n.lineno}: {u.split('(')[0][:40]}")
        assert not bad, (
            "a skill-creation FAILURE is announced as something other than a "
            f"failure: {bad}")


class TestTheInstrumentsAgain:
    def test_the_smoke_gate_sees_an_exit_code(self):
        """`execute` never declares, so round 6's status read was inert for
        the only tool this gate calls, and `if not m: return None` fails it
        OPEN — 7 of 8 non-OK shapes PASSED, and passing marks a task DONE."""
        import ghost_agent.core.build_gates as G

        src = inspect.getsource(G.smoke_gate)
        i = src.index("re.search(r'SMOKE_RESULT" if "re.search(r'SMOKE_RESULT"
                      in src else "SMOKE_RESULT")
        assert "shell_failed" in src[:i], (
            "the gate reaches its fail-open branch without ever asking the "
            "exit code")

    def test_the_still_running_notice_survives_every_shape(self):
        """Round 6 filtered the running ops out BEFORE the mixed-outcome
        gate, so "do NOT re-dispatch these" was unreachable in 4 of 5
        shapes — including both of the most likely ones."""
        from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes

        run = {"tool": "execute", "ok": True, "unresolved": True}
        ok = {"tool": "manage_tasks", "ok": True}
        bad = {"tool": "file_system", "ok": False, "preview": "nope"}
        for label, ops in (("1 ok + 1 running", [ok, run]),
                           ("1 fail + 1 running", [bad, run]),
                           ("2 ok + 1 running", [ok, ok, run]),
                           ("mixed + running", [ok, bad, run])):
            assert "STILL RUNNING" in summarize_multi_op_outcomes(ops), label
        # ...and a turn with no in-flight work is unchanged
        assert "STILL RUNNING" not in summarize_multi_op_outcomes([ok, bad])

    def test_the_eval_instrument_passes_the_tool_name(self):
        import ghost_agent.eval.behavioral as B

        src = inspect.getsource(B)
        assert "_action_failed" in src and 't.get("name")' in src, (
            "a third predicate, and tool-name-blind: `execute` judged by the "
            "generic sniffer, 61 REJECTED refusals missed")

    def test_canonicality_is_per_operation(self):
        from ghost_agent.tools.memory import _bus_canonical_failed as f

        assert f({"vector": "error: down"}, "insert_fact") is True
        assert f({"profile": "ok", "vector": "error: lag"},
                 "update_profile") is False
        assert f({"profile": "error: disk"}, "update_profile") is True
        assert f({"skill": "error: dropped"}, "learn_skill") is True
