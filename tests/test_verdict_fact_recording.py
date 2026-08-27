"""The verifier's verdict is RECORDED on the trajectory (§4BQ follow-up).

WHY THIS EXISTS. The router's difficulty signal can be evaluated for COST
but not for QUALITY. Trajectory labels derive from outcome / n_steps /
tool_calls, so any correctness comparison against them is circular — and
the corpus carries no independent correctness signal at all: measured, 0
of 1,701 trajectories had a user correction or verifier verdict in
`extra`. The router's confident-hard slice was measured to cost 1.85x
wall-clock [+13.9s, +34.7s], but "longer" is not "wronger" and nothing on
disk could distinguish them.

The verifier already runs on every final answer, so stamping its verdict
costs nothing and makes the question answerable from ordinary traffic.

RECORDING ONLY. Nothing consumes this yet, deliberately: §4AN's lesson is
that a signal must be shown informative in the region where it acts
BEFORE anything acts on it. These tests therefore pin that the value is
WRITTEN and REACHES the trajectory — not that any decision changes.
"""

import pytest

from ghost_agent.core import turn_facts


class _Ctx:
    """Minimal context carrying the fact ring."""


def _ring_ctx():
    return _Ctx()


class TestVerdictFactsAreRecordable:
    def test_verdict_and_confidence_round_trip_through_the_ledger(self):
        ctx = _ring_ctx()
        turn_facts.record(ctx, "req-1", verifier_verdict="REFUTED",
                          verifier_confidence=0.83)
        facts = turn_facts.facts_for(ctx, "req-1")
        assert facts["verifier_verdict"] == "REFUTED"
        assert facts["verifier_confidence"] == pytest.approx(0.83)

    def test_it_merges_with_the_router_facts_rather_than_replacing_them(self):
        """Both land on the same turn; a correctness study needs the pair,
        and `record` merges rather than replaces."""
        ctx = _ring_ctx()
        turn_facts.record(ctx, "req-2", router_label="hard",
                          router_confidence=0.71)
        turn_facts.record(ctx, "req-2", verifier_verdict="CONFIRMED",
                          verifier_confidence=0.9)
        facts = turn_facts.facts_for(ctx, "req-2")
        assert facts["router_label"] == "hard"
        assert facts["verifier_verdict"] == "CONFIRMED"

    def test_an_empty_req_id_is_a_no_op(self):
        ctx = _ring_ctx()
        turn_facts.record(ctx, "", verifier_verdict="CONFIRMED")
        assert not turn_facts.facts_for(ctx, "")


class TestTheWriterActuallyRuns:
    """EXECUTES `_record_verdict_sidecar`. Every other test in this file
    is a source or AST assertion, and a review proved all of them stay
    green when the writer's body is replaced with `return` — the same
    instrument failure that let the previous version of this feature
    record nothing at all on both live paths."""

    class _Coll:
        def __init__(self, root):
            self.root = root

    class _Ctx:
        pass

    def _agent(self, tmp_path):
        from ghost_agent.core.agent import GhostAgent
        a = GhostAgent.__new__(GhostAgent)          # no __init__ machinery
        ctx = self._Ctx()
        ctx.trajectory_collector = self._Coll(tmp_path / "trajectories")
        a.context = ctx
        return a

    @staticmethod
    def _rows(tmp_path):
        import json
        out = tmp_path / "verdicts"
        return [json.loads(l) for f in sorted(out.glob("*.jsonl"))
                for l in f.read_text().splitlines() if l.strip()]

    def test_a_row_is_written_and_is_joinable(self, tmp_path):
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-abc", "REFUTED", 0.83)
        rows = self._rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["trajectory_id"] == "traj-abc"
        assert rows[0]["verdict"] == "REFUTED"
        assert rows[0]["confidence"] == pytest.approx(0.83)
        assert rows[0]["at"]

    def test_the_vote_counters_reach_DISK(self, tmp_path):
        """§4BR R18 CRITICAL-2. The verify_depth arm's FIRST gate is "if
        a healthy arm records ~11% non-unanimous votes, so too FEW of
        them means the judge does not vary on this traffic and the arm
        retires". Its inputs shipped as dynamic
        attributes on VerifyResult — absent from the dataclass, to_dict,
        this sidecar and turn-facts — so the gate had nothing to read. A
        decision rule whose instrument does not exist is worse than no
        rule, because it reads as measured."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-v", "REFUTED", 0.8,
                                  sc_n=3, sc_agree=2, sc_drawn=3)
        row = self._rows(tmp_path)[0]
        assert row["sc_n"] == 3 and row["sc_agree"] == 2 and row["sc_drawn"] == 3

    def test_the_carry_works_END_TO_END_through_the_real_verdict_path(
            self, tmp_path, monkeypatch):
        """§4BR R23 — the test below asserts the SEAM; this one asserts the
        PRODUCTION PATH, and only this one can fail when the fix is removed.

        The seam test re-implements the carry in its own body, so it passes
        whatever `_compute_verifier_verdict` does. Four mutations proved it:
        disabling the re-stamp guard, swapping n/agree, reading `[0]`
        instead of `[2]`, and inverting the guard all left the suite green
        while fully reinstating the R22 defect. That is the fourth vacuous
        test in this feature, and the shape is always the same — the test
        does the work instead of watching the work.

        Here the vote is produced by the real sampler, the file-artifact
        override really replaces the result, and the assertion reads the
        sidecar row that lands on disk.
        """
        import asyncio
        from ghost_agent.core import experiments as ex
        from ghost_agent.core import verifier as V
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        import ghost_agent.core.agent as AG
        from ghost_agent.core.agent import GhostAgent

        class _Args:
            no_verifier = False

        class _SkillMem:
            is_read_only = False

        a = GhostAgent.__new__(GhostAgent)
        ctx = self._Ctx()
        ctx.args, ctx.skill_memory = _Args(), _SkillMem()
        ctx.trajectory_collector = self._Coll(tmp_path / "trajectories")

        v = V.Verifier(llm_client=object())
        state = {"n": 0}
        verdicts = ["CONFIRMED", "REFUTED", "REFUTED"]   # contested -> 3 drawn

        async def _call(prompt, *ar, **kw):
            state["n"] += 1
            if kw.get("route_out") is not None:
                kw["route_out"]["route"] = "critic"
            if state["n"] == 1:
                return {"suspects": [{"quote": "q", "check": "support",
                                      "reason": "r"}]}
            i = min(state["n"] - 2, len(verdicts) - 1)
            return {"verdict": verdicts[i], "confidence": 0.8,
                    "reasoning": "cheap", "issues": []}

        object.__setattr__(v, "_call_llm", _call)
        ctx.verifier = v
        a.context = ctx

        ex.enroll_request(ctx, "R", eligible=True, origin="user")
        ex.mark_trigger(ctx, "R", "verify_depth_fired", True)   # deep turn

        # Force the FILE-ARTIFACT override to win — it builds a replacement
        # VerifyResult, which is what drops the counters when the carry is
        # missing.
        # ⚠ **kwargs, deliberately: this stub stands in for a real
        # signature, and when that signature gained a keyword the
        # positional-only lambda raised TypeError INSIDE the block's
        # blanket `except`, which logged a warning and moved on. Both
        # tests kept passing while the override they exist to watch
        # had stopped running.
        a._verify_file_artifacts = (
            lambda claimed, host_dir, *a_, **k_: VerifyResult(
                verdict=VerifyVerdict.REFUTED, confidence=0.9,
                reasoning="FILE-ARTIFACT", issues=["missing"]))
        monkeypatch.setattr(AG, "_claimed_deliverable_files",
                            lambda *x, **k: ["report.md"], raising=False)
        import ghost_agent.tools.file_system as fs
        monkeypatch.setattr(fs, "project_scoped_sandbox",
                            lambda c: (str(tmp_path), None), raising=False)

        final = "Done — report.md is written."
        asyncio.run(a._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "file_system", "content": "z" * 400,
                                  "arguments": {}}],
            messages=[{"role": "user", "content": "make it"},
                      {"role": "assistant", "content": final}],
            final_ai_content=final, last_user_content="make it",
            lc="make it", req_id="R", trajectory_id="T"))

        rows = self._rows(tmp_path)
        assert rows, "no verdict row was written at all"
        row = rows[-1]
        assert row["verdict"] == "REFUTED", "the override must still win"
        assert row["sc_n"] == 3 and row["sc_agree"] == 2, (
            f"the vote behind the overridden verdict was lost: {row}")
        assert row["sc_drawn"] == 3

    def test_an_UNPARSEABLE_vote_survives_an_override_too(self, tmp_path,
                                                          monkeypatch):
        """The case that separates `_vote_carry[2]` from `_vote_carry[0]`.

        A contested vote has parsed==3 and drawn==3, so both indices are
        truthy and the test above cannot tell the guards apart — that
        mutation survived it. When every adjudication sample is unparseable
        the record is (n=0, agree=0, drawn=3): guarding on PARSED drops it,
        guarding on DRAWN keeps it. Dropping it files a turn that paid for
        three samples as a control turn that never voted, which is the
        exact confusion `sc_drawn` was added to prevent.
        """
        import asyncio
        from ghost_agent.core import experiments as ex
        from ghost_agent.core import verifier as V
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        import ghost_agent.core.agent as AG
        from ghost_agent.core.agent import GhostAgent

        class _Args:
            no_verifier = False

        class _SkillMem:
            is_read_only = False

        a = GhostAgent.__new__(GhostAgent)
        ctx = self._Ctx()
        ctx.args, ctx.skill_memory = _Args(), _SkillMem()
        ctx.trajectory_collector = self._Coll(tmp_path / "trajectories")

        v = V.Verifier(llm_client=object())
        state = {"n": 0}

        async def _call(prompt, *ar, **kw):
            state["n"] += 1
            if kw.get("route_out") is not None:
                kw["route_out"]["route"] = "critic"
            if state["n"] == 1:                       # stage 1 succeeds
                return {"suspects": [{"quote": "q", "check": "support",
                                      "reason": "r"}]}
            if state["n"] <= 4:                       # 3 unparseable samples
                return {}
            return {"verdict": "CONFIRMED", "confidence": 0.7,   # classic
                    "reasoning": "fallback", "issues": []}

        object.__setattr__(v, "_call_llm", _call)
        ctx.verifier = v
        a.context = ctx
        ex.enroll_request(ctx, "R", eligible=True, origin="user")
        ex.mark_trigger(ctx, "R", "verify_depth_fired", True)

        # ⚠ **kwargs, deliberately: this stub stands in for a real
        # signature, and when that signature gained a keyword the
        # positional-only lambda raised TypeError INSIDE the block's
        # blanket `except`, which logged a warning and moved on. Both
        # tests kept passing while the override they exist to watch
        # had stopped running.
        a._verify_file_artifacts = (
            lambda claimed, host_dir, *a_, **k_: VerifyResult(
                verdict=VerifyVerdict.REFUTED, confidence=0.9,
                reasoning="FILE-ARTIFACT", issues=["missing"]))
        monkeypatch.setattr(AG, "_claimed_deliverable_files",
                            lambda *x, **k: ["report.md"], raising=False)
        import ghost_agent.tools.file_system as fs
        monkeypatch.setattr(fs, "project_scoped_sandbox",
                            lambda c: (str(tmp_path), None), raising=False)

        final = "Done — report.md is written."
        asyncio.run(a._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "file_system", "content": "z" * 400,
                                  "arguments": {}}],
            messages=[{"role": "user", "content": "make it"},
                      {"role": "assistant", "content": final}],
            final_ai_content=final, last_user_content="make it",
            lc="make it", req_id="R", trajectory_id="T"))

        row = self._rows(tmp_path)[-1]
        assert row["sc_drawn"] == 3, (
            f"three samples were paid for and the row lost them: {row}")
        assert row["sc_n"] == 0, "none of them parsed"

    def test_vote_counters_survive_a_ground_truth_OVERRIDE(self, tmp_path):
        """§4BR R22 MAJOR-1 — the fourth layer of the same defect.

        `verify_claim` re-stamps the counters after its internal escalation
        ladder, and its comment claims that makes it "immune to the sixth
        replacement site". The immunity stops at the function boundary:
        `_compute_verifier_verdict` substitutes a DIFFERENT VerifyResult
        three more times after it returns — the visual, WEB-EXEC and
        file-artifact ground-truth overrides — and the sidecar then reads
        the counters off the replacement, finds None, and omits the keys.
        The row becomes byte-identical to a control turn that never voted.
        Measured on the live log: ~3% of claim verdicts take an override.

        Simulated here at the seam that matters: a result carrying a vote,
        replaced by a fresh object, then recorded.
        """
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        voted = VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.8,
                             reasoning="cheap", issues=[])
        voted.self_consistency_n = 3
        voted.self_consistency_agree = 2
        voted.self_consistency_drawn = 3
        carry = (voted.self_consistency_n, voted.self_consistency_agree,
                 voted.self_consistency_drawn)

        override = VerifyResult(verdict=VerifyVerdict.REFUTED,
                                confidence=0.9,
                                reasoning="FILE-ARTIFACT", issues=["missing"])
        assert override.self_consistency_n is None      # the defect's source
        if carry[2] and override.self_consistency_drawn is None:
            (override.self_consistency_n, override.self_consistency_agree,
             override.self_consistency_drawn) = carry

        a = self._agent(tmp_path)
        a._record_verdict_sidecar(
            "traj-ovr", override.verdict.value, override.confidence,
            sc_n=override.self_consistency_n,
            sc_agree=override.self_consistency_agree,
            sc_drawn=override.self_consistency_drawn)
        row = self._rows(tmp_path)[0]
        assert row["verdict"] == "REFUTED", "the override must still win"
        assert row["sc_n"] == 3 and row["sc_drawn"] == 3, (
            "the vote that produced the overridden verdict was lost")

    def test_the_override_carry_is_WIRED_in_the_verdict_path(self):
        """The execution test above proves the seam works; this proves the
        production path actually contains it. Both halves are needed — the
        three override sites are what the wire has to cross."""
        import ast
        from pathlib import Path
        import ghost_agent.core.agent as m
        src = Path(m.__file__).read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        cap = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_vote_carry"
                       for t in n.targets)]
        assert cap, "_vote_carry is not captured at all"
        assigns = [n.lineno for n in cap]
        # It must capture FROM v_result, not from constants: replacing the
        # getattr calls with None preserves every structural property here
        # while making the carry inert, and that mutation survived the
        # first version of this test.
        dumped = ast.dump(cap[0])
        assert "v_result" in dumped, "the carry does not read v_result"
        for attr in ("self_consistency_n", "self_consistency_agree",
                     "self_consistency_drawn"):
            assert attr in dumped, f"the carry does not capture {attr}"
        restamp = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(getattr(t, "attr", "") == "self_consistency_drawn"
                           for t in ast.walk(n) if isinstance(t, ast.Attribute))]
        assert restamp, "_vote_carry is captured but never re-applied"
        assert min(assigns) < max(restamp), "captured after it is applied"
        # And the overrides must sit BETWEEN the two, or the carry is moot.
        overrides = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
                     for t in n.targets
                     if isinstance(t, ast.Name) and t.id == "v_result"]
        crossed = [ln for ln in overrides if min(assigns) < ln < max(restamp)]
        assert len(crossed) >= 3, (
            f"the carry spans {len(crossed)} v_result reassignments; the "
            "visual, WEB-EXEC and file-artifact overrides must all be inside")

    def test_a_single_sample_verdict_carries_NO_vote_keys(self, tmp_path):
        """Control rows stay byte-shaped as before, so "no vote was taken"
        stays distinguishable from "a vote of one" in the corpus."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-c", "CONFIRMED", 0.9)
        row = self._rows(tmp_path)[0]
        assert "sc_n" not in row and "sc_agree" not in row

    def test_the_verification_ROUTE_reaches_disk(self, tmp_path):
        """§4BR R23 finding 2. `verify_route` was written ONLY to turn-facts
        — three lines above the comment explaining that turn-facts records
        nothing on either live delivery path. Measured after six days: 154
        router labels in the corpus and ZERO routes, while two comments
        claimed the arm's analysis could condition on it.

        It matters because ~10.8% of the arm's population takes the
        code-output route, which has no two-stage leg and therefore cannot
        receive the treatment at all. Without the route those turns are
        indistinguishable from turns the treatment simply failed to help,
        and they dilute the effect toward zero."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-r", "CONFIRMED", 0.9,
                                  route="code_output")
        assert self._rows(tmp_path)[0]["route"] == "code_output"

    def test_the_route_is_FORWARDED_from_the_verdict_path(self):
        """The other end of that wire — the half that was missing."""
        import ast
        import inspect
        from pathlib import Path
        import ghost_agent.core.agent as m
        from ghost_agent.core.agent import GhostAgent
        tree = ast.parse(Path(m.__file__).read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        call = next(c for c in ast.walk(fn) if isinstance(c, ast.Call)
                    and getattr(c.func, "attr", "") == "_record_verdict_sidecar")
        kw = {k.arg: k.value for k in call.keywords}
        assert "route" in kw, "the route is not forwarded to the sidecar"
        assert "_verify_route" in ast.dump(kw["route"])
        params = inspect.signature(GhostAgent._record_verdict_sidecar).parameters
        assert "route" in params

    def test_a_repaired_turn_keeps_BOTH_rows_with_seq_ordering(self, tmp_path):
        """An auto-repaired turn verifies twice. Both rows are real
        signal, so the join contract is last-wins — asserted here rather
        than left for an analysis to discover."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-1", "REFUTED", 0.9)
        a._record_verdict_sidecar("traj-1", "CONFIRMED", 0.8)
        rows = [r for r in self._rows(tmp_path) if r["trajectory_id"] == "traj-1"]
        assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)
        assert max(rows, key=lambda r: r["seq"])["verdict"] == "CONFIRMED"

    def test_max_seq_stays_correct_across_many_other_trajectories(self, tmp_path):
        """`seq` was a per-trajectory map bounded at 512 entries, so after
        eviction the second verdict for a trajectory restarted at 0 and a
        max-seq join returned the PRE-repair verdict."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("T", "REFUTED", 0.9)
        for i in range(600):
            a._record_verdict_sidecar(f"other-{i}", "CONFIRMED", 0.5)
        a._record_verdict_sidecar("T", "CONFIRMED", 0.8)
        rows = [r for r in self._rows(tmp_path) if r["trajectory_id"] == "T"]
        assert max(rows, key=lambda r: r["seq"])["verdict"] == "CONFIRMED"

    def test_it_never_raises_when_the_sink_is_unwritable(self, tmp_path, caplog):
        import logging
        a = self._agent(tmp_path)
        (tmp_path / "verdicts").write_text("not a directory")
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            a._record_verdict_sidecar("traj-x", "CONFIRMED", 0.5)   # must not raise
        assert any("UNWRITABLE" in r.getMessage() for r in caplog.records)

    def test_no_collector_is_a_silent_no_op(self, tmp_path):
        a = self._agent(tmp_path)
        a.context.trajectory_collector = None
        a._record_verdict_sidecar("traj-y", "CONFIRMED", 0.5)
        assert not (tmp_path / "verdicts").exists()

    def test_it_records_only_the_verdict_and_a_number(self, tmp_path):
        """No reasoning text, no claim, no evidence — this file
        accumulates unattended for weeks."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-z", "CONFIRMED", 0.5)
        assert set(self._rows(tmp_path)[0]) == {
            "trajectory_id", "verdict", "confidence", "at", "seq"}


class TestTheStampIsWiredAtTheChokePoint:
    """Source pins. The turn loop is not unit-testable here, and the three
    consumer sites are exactly where a wrapper split would reappear — this
    feature has already produced three of those."""

    @staticmethod
    def _src():
        from pathlib import Path
        import ghost_agent.core.agent as m
        return Path(m.__file__).read_text()

    @staticmethod
    def _sidecar_call():
        """The sidecar call node inside `_compute_verifier_verdict`.

        By AST rather than by substring: the call grew keyword arguments
        (§4BR vote counters) and wrapped onto several lines, which broke a
        pin that was only ever testing the formatting.
        """
        import ast
        from pathlib import Path
        import ghost_agent.core.agent as m
        tree = ast.parse(Path(m.__file__).read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        return next(c for c in ast.walk(fn) if isinstance(c, ast.Call)
                    and getattr(c.func, "attr", "") == "_record_verdict_sidecar")

    def test_the_verdict_is_recorded(self):
        src = self._src()
        assert "verifier_verdict=_vs, verifier_confidence=_vc" in src
        call = self._sidecar_call()
        assert len(call.args) >= 3, "trajectory_id, verdict and confidence"

    def test_the_vote_counters_are_FORWARDED_from_the_verdict(self):
        """The other end of the wire. `TestTheWriterActuallyRuns` proves the
        sidecar PERSISTS the counters when handed them; this proves the
        verdict path HANDS them over. Renaming the kwarg here raises a
        TypeError swallowed by the broad `except` — a warning in the log and
        silently no counters on disk — and that mutation survived a suite
        that tested only the writer."""
        import ast
        import inspect
        from ghost_agent.core.agent import GhostAgent
        call = self._sidecar_call()
        kw = {k.arg: k.value for k in call.keywords}
        for name, attr in (("sc_n", "self_consistency_n"),
                           ("sc_agree", "self_consistency_agree"),
                           ("sc_drawn", "self_consistency_drawn")):
            assert name in kw, f"{name} is not forwarded to the sidecar"
            src = ast.dump(kw[name])
            assert attr in src, f"{name} must come from v_result.{attr}: {src}"
        # …and the writer must actually accept them under those names.
        params = inspect.signature(GhostAgent._record_verdict_sidecar).parameters
        assert {"sc_n", "sc_agree", "sc_drawn"} <= set(params)

    def test_it_is_recorded_ONCE_at_the_shared_choke_point(self):
        """`_compute_verifier_verdict` is documented as running exactly
        once per final answer, so recording there covers the streamed and
        non-streamed paths alike.

        Checked by AST, not by string offsets. The offset version compared
        `index("async def ...") < index(stamp)` — which stays true when the
        stamp is moved to the END OF THE FILE, outside the method
        completely. Verified: relocating the whole block there left all
        three source pins green."""
        import ast
        from pathlib import Path
        import ghost_agent.core.agent as m
        tree = ast.parse(Path(m.__file__).read_text())
        owners = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seg = ast.dump(node)
                if "_record_verdict_sidecar" in seg and node.name != "_record_verdict_sidecar":
                    owners.append(node.name)
        assert owners == ["_compute_verifier_verdict"], owners

    def test_recording_can_never_fail_a_turn(self):
        """A durable write on the answer path must be strictly optional.

        By AST containment rather than a character window: the window
        version measured DISTANCE from the call and broke when the call
        grew keyword arguments, which has nothing to do with the property
        it guards.
        """
        import ast
        from pathlib import Path
        import ghost_agent.core.agent as m
        tree = ast.parse(Path(m.__file__).read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        call = next(c for c in ast.walk(fn) if isinstance(c, ast.Call)
                    and getattr(c.func, "attr", "") == "_record_verdict_sidecar")
        guarding = [t for t in ast.walk(fn) if isinstance(t, ast.Try)
                    and any(c is call for c in ast.walk(t))
                    and any(isinstance(h.type, ast.Name)
                            and h.type.id == "Exception" for h in t.handlers)]
        assert guarding, "the durable write is not inside a try/except Exception"
        # ...and the recording must never fail a turn: no bare re-raise.
        for t in guarding:
            for h in t.handlers:
                assert not any(isinstance(n, ast.Raise) for n in ast.walk(h)), (
                    "the handler re-raises; recording would fail the turn")

    def test_the_docstring_admits_the_write(self):
        """It previously said "NO side effects". A docstring that
        contradicts the code is the defect class this repo keeps finding."""
        src = self._src()
        i = src.index("async def _compute_verifier_verdict")
        assert "Pure verifier verdict computation — NO side effects." not in src[i:i + 2000]
        assert "the verdict is RECORDED onto" in src[i:i + 2000]

    def test_nothing_consumes_it_yet(self):
        """Deliberate: §4AN acted on a signal without showing it was
        informative where it acted. If this ever fails, the consumer must
        arrive WITH the measurement that justifies it."""
        import subprocess
        from pathlib import Path
        import ghost_agent
        # ABSOLUTE path and an asserted exit code. The relative-path
        # version passed vacuously from any cwd but the repo root: grep
        # returned rc=1 with zero hits and the test read that as "no
        # readers". A search that cannot fail proves nothing.
        srcdir = str(Path(ghost_agent.__file__).parent)
        probe = subprocess.run(["grep", "-rn", "def ", "--include=*.py", srcdir],
                               capture_output=True, text=True)
        assert probe.returncode == 0 and probe.stdout, (
            "control probe found nothing — the search itself is broken")
        # Look for READS of the FACT, not for the identifier. `calibration
        # .grade_turn_outcome(verifier_verdict=...)` is an unrelated
        # parameter of the same name, and matching it would make this test
        # fail for a reason that has nothing to do with what it claims.
        patterns = [r'facts.*\["verifier_verdict"\]',
                    r'facts.*\.get\("verifier_verdict"',
                    r'extra.*\["verifier_verdict"\]',
                    r'extra.*\.get\("verifier_verdict"']
        readers = []
        for pat in patterns:
            out = subprocess.run(["grep", "-rnE", pat, "--include=*.py",
                                  srcdir],
                                 capture_output=True, text=True).stdout
            readers += [l for l in out.splitlines() if l.strip()]
        assert not readers, f"something now reads the verdict fact: {readers}"
