"""§4BQ's FIRST LIVE CONSUMER: verification depth routed by the router.

On a turn the complexity router called confidently HARD (label=hard, not
escalated), the cheap leg's adjudication runs self-consistency n=3
(majority vote) instead of a single sample.

WHY THIS CONSUMER AND NOT THE OBVIOUS ONE. The MCTS depth gate is the
router's only historical consumer and it is switched off pending an
execution-grounded value function (no trained PRM exists), disabled by a
paired ablation measuring 78% vs 80% at ~1.8x latency. Re-enabling it
would re-introduce a measured-harmful mechanism. Self-consistency is a
different trade entirely:

  * it targets the MISS side — the judge's own confidence carries NO
    signal about false-CONFIRM (AUC 0.5087, chance), while cross-sample
    disagreement does;
  * majority-of-3 was measured to fix `artifact_leak` (36% of the miss
    mass), and `clean` never varied, so the false-positive gate survives;
  * the samples run on the CRITIC node, so none of it lands on the main
    model — and a sample that degrades onto the main route stops the vote.

Routing by difficulty is what makes it affordable: n=3 on the ~58% of
turns the router calls confidently hard, rather than on all of them.

⚠ THE COST CLAIM HERE WAS ONCE WRONG, AND WRONG BECAUSE IT WAS REASONED
RATHER THAN MEASURED. This file previously said the samples ran under
`asyncio.gather` so the added wall-clock was "~one sample", and that the
trigger fired on ~41% of turns. Timed against the live critic (§4BR R17):
gather cost 24.9s vs 10.8s for one sample — and 17.8s for the SEQUENTIAL
loop its comment dismissed, because llama.cpp's per-slot prompt cache
makes a repeat of the same prompt nearly free while concurrent samples
land on different slots and each re-prefills 2,285 tokens. The live
corpus put the trigger rate at 58.1% (122/210), not 41%. Both numbers
now come from measurements; the sampler is sequential and early-stops
once the majority is decided (~13.0s in the common case).

⚠ THE PREMISE IS UNTESTED. "Harder turns are likelier wrong" has never
been shown here — the corpus carries no independent correctness signal,
which is why the verdict sidecar exists. So this ships as a RANDOMISED
ARM, not a default: both arms compute and stamp the trigger, and the
deploy is the measurement. §4AN's lesson was that this router was
accurate overall and worthless where it acted.

These tests pin the SAFETY properties, not the benefit (the benefit is
the arm's job):
  1. depth reaches ONLY the cheap leg — never the main-model escalations;
  2. the code-output route, which has no two-stage leg, is untouched and
     is NAMED so the analysis can condition on it;
  3. routing raises a floor and never lowers an operator's setting;
  4. it is killable;
  5. a failure in the routing can never fail a verdict.
"""

import os
import re
from pathlib import Path

import pytest

from ghost_agent.core import verifier as V

_AGENT = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "core" / "agent.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFY_DEPTH_ROUTING", raising=False)
    monkeypatch.delenv("GHOST_VERIFY_SELF_CONSISTENCY", raising=False)


class TestTheDepthKnob:
    def test_a_normal_turn_is_unchanged(self):
        assert V._self_consistency_n() == 1

    def test_a_confident_hard_turn_gets_majority_of_three(self):
        assert V._self_consistency_n(deep=True) == 3

    def test_routing_raises_a_floor_and_never_lowers(self, monkeypatch):
        """An operator who set 5 globally keeps 5 on hard turns."""
        monkeypatch.setenv("GHOST_VERIFY_SELF_CONSISTENCY", "5")
        assert V._self_consistency_n(deep=True) == 5
        assert V._self_consistency_n() == 5

    def test_it_is_killable(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_DEPTH_ROUTING", "0")
        assert V._self_consistency_n(deep=True) == 1

    def test_the_kill_switch_is_read_per_call(self, monkeypatch):
        """Same idiom as _two_stage_enabled: flippable without a restart."""
        assert V._self_consistency_n(deep=True) == 3
        monkeypatch.setenv("GHOST_VERIFY_DEPTH_ROUTING", "false")
        assert V._self_consistency_n(deep=True) == 1


class TestTheDecisionRuleExecutes:
    """§4BR R17 MAJOR-7. The first version of this suite was 19 source-text
    greps: cutting `deep=_deep` to `deep=False` in agent.py, and
    `_self_consistency_n(deep=deep)` to `(deep=False)` in verifier.py, both
    left 19/19 GREEN. The rule now lives in a callable predicate so it can
    be asserted by behaviour rather than by description."""

    def test_a_confident_hard_treatment_turn_is_deep(self):
        assert V.depth_for_turn(router_label="hard", router_escalated=False,
                                arm="treatment") is True

    def test_the_control_arm_is_never_deep(self):
        assert V.depth_for_turn(router_label="hard", router_escalated=False,
                                arm="control") is False

    def test_an_unenrolled_turn_is_never_deep(self):
        """`arm_for` returns "" for internal/background/sim requests, and
        consumers must treat "" as control."""
        assert V.depth_for_turn(router_label="hard", router_escalated=False,
                                arm="") is False

    def test_an_easy_turn_is_never_deep(self):
        assert V.depth_for_turn(router_label="easy", router_escalated=False,
                                arm="treatment") is False

    def test_an_ESCALATED_hard_verdict_is_not_deep(self):
        """An untrained or unsure router escalates WITH label='hard'.
        Counting those would fire the treatment on essentially every turn —
        the §4BQ finding that the router was accurate overall and worthless
        exactly where it acted."""
        assert V.depth_for_turn(router_label="hard", router_escalated=True,
                                arm="treatment") is False

    def test_the_kill_switch_is_folded_INTO_the_rule(self, monkeypatch):
        """R17 MAJOR-4: the switch was read only in the verifier while the
        stamp was written unconditionally, so killing it mid-window would
        have recorded treatment turns that ran control behaviour — a
        treatment filed as a withheld one, poisoning the block the report
        tells the operator to read first. One predicate, one answer."""
        monkeypatch.setenv("GHOST_VERIFY_DEPTH_ROUTING", "0")
        assert V.depth_for_turn(router_label="hard", router_escalated=False,
                                arm="treatment") is False

    def test_depth_is_FALSE_when_two_stage_is_off(self, monkeypatch):
        """§4BR R20 MAJOR-3. `GHOST_VERIFY_TWO_STAGE=0` makes `verify_claim`
        take the classic single-prompt path — one LLM call, control's exact
        shape, no voted leg for `deep` to reach. Stamping deep=True under it
        files a treatment turn that ran CONTROL behaviour: the identical
        defect R17 found for `GHOST_VERIFY_DEPTH_ROUTING`, for a second
        switch. Both switches now live inside the one predicate."""
        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
        assert V.depth_for_turn(router_label="hard", router_escalated=False,
                                arm="treatment") is False

    def test_the_trigger_is_arm_and_switch_INDEPENDENT(self, monkeypatch):
        """Eligibility must be recorded for BOTH arms and even when the
        feature is switched off, or the triggered-only comparison loses its
        denominator."""
        monkeypatch.setenv("GHOST_VERIFY_DEPTH_ROUTING", "0")
        assert V.router_called_hard("hard", False) is True
        assert V.router_called_hard("hard", True) is False
        assert V.router_called_hard("easy", False) is False


class TestTheWireIsConnectedEndToEnd:
    """Execution, not grep: does `deep` actually change how many
    adjudications happen?"""

    @staticmethod
    def _verifier(calls):
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        v = V.Verifier(llm_client=None)

        async def _call(prompt, *a, **k):
            # Stage 1 is the FIRST call by construction; everything after it
            # is an adjudication sample. Keyword-sniffing the prompt made
            # this test pass vacuously when stage 1 failed.
            calls.append("enumerate" if not calls else "adjudicate")
            if k.get("route_out") is not None:
                k["route_out"]["route"] = "critic"
            return {"suspects": [{"quote": "q", "check": "support",
                                  "reason": "r"}],
                    "verdict": "CONFIRMED", "confidence": 0.8,
                    "reasoning": "r", "issues": []}

        object.__setattr__(v, "_call_llm", _call)
        return v

    def _run(self, deep):
        import asyncio
        calls = []
        v = self._verifier(calls)
        asyncio.run(v._verify_claim_two_stage("c", "e", "ctx", deep=deep))
        return calls

    def test_a_normal_turn_adjudicates_once(self):
        assert self._run(deep=False).count("adjudicate") == 1

    def test_a_deep_turn_adjudicates_more_than_once(self):
        """The mutation that survived 19 green tests: if `deep` stops
        reaching `_self_consistency_n`, this drops back to 1."""
        assert self._run(deep=True).count("adjudicate") > 1

    def test_stage_one_still_runs_exactly_once(self):
        """Depth samples the ADJUDICATION only. Re-running the enumeration
        would multiply the expensive half for no vote."""
        assert self._run(deep=True).count("enumerate") == 1


class TestDepthReachesTheCheapLegOnly:
    """The two `force_main=True` call sites are main-model escalations.
    Multiplying those would put 3x on the 35B for the most expensive turns
    — the exact cost profile this consumer exists to avoid."""

    @staticmethod
    def _src():
        return Path(V.__file__).read_text()

    def test_the_cheap_leg_receives_deep(self):
        """Structural: the textual version pinned the whole argument list
        and broke when an unrelated kwarg was added, which is a test
        guarding formatting rather than behaviour."""
        import ast
        import inspect
        import textwrap
        src = textwrap.dedent(inspect.getsource(V.Verifier.verify_claim))
        calls = [c for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "_verify_claim_two_stage"]
        kws = [{k.arg: k.value for k in c.keywords} for c in calls]
        cheap = [k for k in kws if "force_main" not in k]
        main = [k for k in kws if "force_main" in k]
        assert len(cheap) == 1, f"expected one cheap-leg call, got {len(cheap)}"
        assert isinstance(cheap[0].get("deep"), ast.Name)
        assert cheap[0]["deep"].id == "deep"
        # `verify_claim` also contains the `_reverify_on_main` closure. That
        # leg is the main-model escalation and must never be given depth.
        for k in main:
            assert "deep" not in k, "an escalation leg was given depth"

    def test_no_force_main_call_site_receives_deep(self):
        src = self._src()
        for m in re.finditer(r"_verify_claim_two_stage\((.*?)\)", src, re.S):
            args = m.group(1)
            if "force_main=True" in args:
                assert "deep" not in args, args

    def test_the_scope_is_explained_where_it_is_enforced(self):
        """A future reader must not 'fix' the asymmetry by threading deep
        into the escalations."""
        src = self._src()
        assert "CHEAP leg ONLY" in src


class TestTheUntreatableRouteIsExcludedAndNamed:
    """`verify_code_output` has no two-stage leg (one _call_llm, by
    design), so there is no adjudication to sample. Passing `deep` there
    would be an accepted-and-ignored parameter."""

    @staticmethod
    def _agent_src():
        return _AGENT.read_text()

    def test_code_output_is_not_given_deep(self):
        src = self._agent_src()
        call = src[src.index("verifier.verify_code_output("):]
        call = call[:call.index(")")]
        assert "deep=" not in call, call

    def test_verify_code_output_does_not_accept_deep(self):
        """If it ever grows the parameter without a two-stage leg, that is
        the dead-mechanism shape."""
        import inspect
        params = inspect.signature(V.Verifier.verify_code_output).parameters
        assert "deep" not in params

    def test_the_route_is_recorded_for_the_analysis(self):
        """A confident-hard turn that took the code route gets NO
        treatment. Diluting the arm with those silently would understate
        the effect."""
        src = self._agent_src()
        assert 'verify_route=str(_verify_route)' in src
        assert '_verify_route = "code_output"' in src


class TestTheDecisionPoint:
    @staticmethod
    def _agent_src():
        return _AGENT.read_text()

    def test_it_is_computed_beside_high_stakes(self):
        """`_high_stakes` is the RETROSPECTIVE signal and carries a comment
        explaining that this is the ONE place every verdict is produced —
        so depth can never be live on one delivery path and dark on
        another. The prospective signal belongs in the same place."""
        src = self._agent_src()
        hs = src.index("_high_stakes = _turn_had_tool_failure")
        deep = src.index("_deep = False")
        assert 0 < deep - hs < 2000, deep - hs

    def test_the_decision_uses_the_ONE_predicate(self):
        """Not re-derived at the call site. Two copies of one rule is the
        wrapper-split shape this project keeps re-growing (§4BN), and it is
        also what made the rule untestable — see TestTheDecisionRuleExecutes."""
        src = self._agent_src()
        assert "depth_for_turn as _depth_rule" in src
        assert "_vd_deep = _depth_rule(" in src

    def test_it_is_a_randomised_arm_not_a_default(self):
        src = self._agent_src()
        assert '_experiments_mod.arm_for(' in src
        assert '"verify_depth",' in src

    def test_both_arms_stamp_the_trigger_via_mark_trigger(self):
        """Arm-independent stamping is what makes a triggered-only
        comparison possible; without it the all-enrolled average dilutes
        the effect over turns the trigger never touched (§4AN).

        It must go through `mark_trigger` (the experiments RING), not
        turn-facts alone: the ring is what survives a streamed drain, and
        `trigger_fired` reads nothing else."""
        src = self._agent_src()
        i = src.index('"verify_depth_fired",')
        assert "mark_trigger(" in src[i - 200:i], src[i - 200:i]
        assert "verify_depth_deep=bool(_vd_deep)" in src

    def test_EVERY_verify_claim_call_forwards_the_ROUTED_value(self):
        """§4BR R17 MAJOR-7: mutating `deep=_deep` to `deep=False` left the
        whole suite green — the feature was inert and nothing noticed.

        Structural rather than textual on purpose: this fails on a literal
        (`deep=False`), on a dropped kwarg, and on a THIRD call site added
        later that forgets it — which is the realistic way this wire gets
        cut, since there are already two.
        """
        import ast
        tree = ast.parse(self._agent_src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        calls = [c for c in ast.walk(fn)
                 if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "verify_claim"]
        assert len(calls) == 2, f"expected 2 verify_claim call sites, got {len(calls)}"
        for c in calls:
            kw = {k.arg: k.value for k in c.keywords}
            assert "deep" in kw, "a verify_claim call forwards no depth at all"
            assert isinstance(kw["deep"], ast.Name), (
                f"deep must forward the routed variable, not "
                f"{ast.dump(kw['deep'])}")
            assert kw["deep"].id == "_deep"

    def test_the_deep_flag_is_READ_from_the_ledger_not_recomputed(self):
        """One definition. handle_chat decides (while the ring slot is live
        and before either delivery path writes its trajectory); this method
        only reads. Recomputing here is what put the stamp after the
        streamed drain — 2/169 rows, measured."""
        src = self._agent_src()
        assert '_deep = bool(_flags.get("verify_depth_fired"))' in src
        i = src.index("async def _compute_verifier_verdict")
        j = src.index("async def _compute_verifier_verdict_gated")
        assert "arm_for" not in src[i:j], (
            "the arm must not be consulted here — it is decided in handle_chat")

    def test_the_flag_is_read_from_the_ENROLLED_ONLY_ring(self):
        """§4BR R22 MINOR-3. The same boolean lives in two rings, and they
        have different eviction exposure. `turn_facts` is a 16-slot ring
        that EVERY request writes to — including sub-agent and dream turns,
        which run the same router block — while the experiments ring only
        ever holds enrolled requests and refreshes LRU position on each
        mark. A 16-wide sub-agent fan-out evicts the parent's turn-facts
        entry but not its experiments entry, which would read `_deep=False`
        on a turn whose trajectory still stamps `verify_depth_fired=True`:
        a control run filed as treatment, i.e. the exact inversion
        `depth_for_turn`'s docstring promises cannot happen."""
        from ghost_agent.core import experiments as _exp
        from ghost_agent.core import turn_facts as _tf

        class _Ctx:
            pass
        ctx = _Ctx()
        _exp.enroll_request(ctx, "parent", eligible=True, origin="user")
        _exp.mark_trigger(ctx, "parent", "verify_depth_fired", True)
        _tf.record(ctx, "parent", verify_depth_deep=True)

        # 20 sub-agent turns: they record router facts (same block) but are
        # never enrolled, so they take turn_facts slots and no arm slots.
        for i in range(20):
            _tf.record(ctx, f"sub-{i}", router_label="easy")

        assert not _tf.facts_for(ctx, "parent").get("verify_depth_deep"), (
            "if turn-facts survived this, the test no longer guards anything")
        assert _exp.trigger_flags(ctx, "parent").get(
            "verify_depth_fired") is True, (
            "the enrolled-only ring must survive a sub-agent fan-out")

    def test_the_stamp_is_passed_the_REQUEST_ID(self):
        """§4BR R18. `mark_trigger` returns SILENTLY when req_id is empty or
        has no ring slot (experiments.py), and `turn_facts.record` keys on
        it — so a wrong or empty id loses the stamp with no error anywhere.
        Both mutations (empty string, wrong variable) survived the suite.
        """
        import ast
        tree = ast.parse(self._agent_src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
        marks = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "mark_trigger"
                 and any(isinstance(a, ast.Constant)
                         and a.value == "verify_depth_fired" for a in c.args)]
        assert len(marks) == 1, f"expected 1 verify_depth mark_trigger, got {len(marks)}"
        rid = marks[0].args[1]
        assert isinstance(rid, ast.Name) and rid.id == "req_id", ast.dump(rid)

        recs = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                and getattr(c.func, "attr", "") == "record"
                and any(k.arg == "verify_depth_deep" for k in c.keywords)]
        assert len(recs) == 1
        rid2 = recs[0].args[1]
        assert isinstance(rid2, ast.Name) and rid2.id == "req_id", ast.dump(rid2)

    def test_enrolment_happens_BEFORE_the_stamp(self):
        """THE ordering invariant this whole fix rests on.

        `mark_trigger` refuses to create a ring slot — it returns silently
        when none exists, deliberately, because sub-agents share a
        shallow-copied context and creating slots there re-opened an
        eviction bug. So if `enroll_request` ever moves after the §4BR
        block, the trigger is dropped for every turn and nothing complains.
        """
        import ast
        src = self._agent_src()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
        enrol = [c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "enroll_request"]
        mark = [c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
                and getattr(c.func, "attr", "") == "mark_trigger"
                and any(isinstance(a, ast.Constant)
                        and a.value == "verify_depth_fired" for a in c.args)]
        assert enrol and mark
        assert min(enrol) < min(mark), (
            f"enroll_request at {enrol} must precede the verify_depth "
            f"stamp at {mark}, or mark_trigger silently no-ops")

    def test_the_trigger_is_stamped_ONLY_when_it_FIRED(self):
        """§4BR R18 CRITICAL-1. `trigger_fired` tests for the key's
        PRESENCE, not its value, so an unconditional `mark_trigger(...,
        False)` on every routed turn makes "triggered only" mean "all
        traffic" — the block the operator is told to read FIRST becomes
        identical to the all-enrolled one, and the effect is diluted over
        the ~42% of turns the treatment cannot reach. Every other call site
        guards the CALL; this one did not."""
        import ast
        src = self._agent_src()
        tree = ast.parse(src)
        call = next(c for c in ast.walk(tree) if isinstance(c, ast.Call)
                    and getattr(c.func, "attr", "") == "mark_trigger"
                    and any(isinstance(a, ast.Constant)
                            and a.value == "verify_depth_fired" for a in c.args))
        # The nearest enclosing If must be the trigger test itself.
        guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and n.lineno < call.lineno
                  and any(c is call for c in ast.walk(n))]
        assert guards, "mark_trigger is unconditional"
        inner = max(guards, key=lambda n: n.lineno)
        assert isinstance(inner.test, ast.Name) and inner.test.id == "_rhard", (
            ast.dump(inner.test))
        # And the VALUE follows the module contract: did the treatment run.
        assert isinstance(call.args[3], ast.Name) and call.args[3].id == "_vd_deep"

    def test_a_context_mutation_key_is_stamped_for_the_GEPA_corpus(self):
        """§4BR R18 MAJOR-2. A deep turn can change the verdict, which can
        append the verifier note to the final text and add a repair round —
        both enter conversation history and the recorded trajectory. The
        experiments module states the contract: any treatment that alters
        what the model sees MUST stamp a compliance key and register it, or
        the tool-description optimizer replays a context only half of
        production ever saw."""
        from ghost_agent.core.experiments import CONTEXT_MUTATING_KEYS
        assert "verify_depth_context" in CONTEXT_MUTATING_KEYS
        src = self._agent_src()
        assert '"verify_depth_context": True' in src

        from ghost_agent.core.experiments import context_was_mutated

        class _T:
            extra = {"verify_depth_context": True}
        assert context_was_mutated(_T()) is True

    def test_the_context_key_is_stamped_ONLY_on_a_deep_turn(self):
        """§4BR R21 MINOR-6. Stamping it unconditionally survives the suite
        and would exclude every confidently-hard CONTROL turn from the GEPA
        fixture corpus — silently shrinking the population the tool-
        description optimizer trains on, in the arm that changes nothing."""
        import ast
        src = self._agent_src()
        tree = ast.parse(src)
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.Dict)
                    and any(isinstance(k, ast.Constant)
                            and k.value == "verify_depth_context"
                            for k in n.keys))
        # The dict is the body of a conditional expression gated on _vd_deep.
        owner = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.IfExp)
                     and any(c is node for c in ast.walk(n)))
        assert isinstance(owner.test, ast.Name) and owner.test.id == "_vd_deep", (
            ast.dump(owner.test))

    def test_routing_can_never_fail_a_verdict(self):
        src = self._agent_src()
        i = src.index("_deep = False")
        window = src[i:i + 2200]
        assert "except Exception" in window
        assert "never fail a verdict" in window

    def test_the_router_verdict_is_READ_not_re_threaded(self):
        """Threading a new parameter through three call sites is the
        wrapper-split shape this feature keeps producing; a per-request
        ledger already carries the decision, keyed by req_id.

        §4BR R23 finding 4: the first version asserted a substring
        containing a LINE BREAK, so it failed on a pure reformat, and its
        other half (`'self.context, str(req_id or ""))'`) was satisfied by
        two unrelated call sites elsewhere in the file. Meanwhile the thing
        that actually matters — that the READ is keyed on req_id — was
        unpinned: keying it on trajectory_id instead makes EVERY treatment
        turn run control, silently, which is the R18 defect class exactly.
        """
        import ast
        tree = ast.parse(self._agent_src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "trigger_flags"]
        assert len(calls) == 1, f"expected one trigger_flags read, got {len(calls)}"
        arg = calls[0].args[1]
        assert "req_id" in ast.dump(arg), (
            f"the depth read must be keyed on req_id, not {ast.dump(arg)}")
        assert "trajectory_id" not in ast.dump(arg)

    def test_the_deep_read_and_the_stamp_use_the_SAME_key(self):
        """The write is in handle_chat and the read is in the verdict path;
        if they ever key differently the feature goes dark with no error
        anywhere. Asserted as an equality between the two sites, not as two
        independent spellings."""
        import ast
        tree = ast.parse(self._agent_src())

        def _key_of(fnname, attr, flag):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.AsyncFunctionDef) and n.name == fnname)
            call = next(c for c in ast.walk(fn) if isinstance(c, ast.Call)
                        and getattr(c.func, "attr", "") == attr
                        and (flag is None or any(
                            isinstance(x, ast.Constant) and x.value == flag
                            for x in c.args)))
            return ast.dump(call.args[1])

        write = _key_of("handle_chat", "mark_trigger", "verify_depth_fired")
        read = _key_of("_compute_verifier_verdict", "trigger_flags", None)
        assert "req_id" in write and "req_id" in read, (write, read)


class TestTheExperimentIsRegistered:
    def test_the_spec_exists_with_both_arms(self):
        from ghost_agent.core.experiments import DEFAULT_SPECS
        spec = next((s for s in DEFAULT_SPECS if s.name == "verify_depth"), None)
        assert spec is not None
        assert set(spec.arms) == {"control", "treatment"}
        assert spec.enabled is True

    def test_the_description_states_the_untested_premise(self):
        """A future reader must not promote this to a default without the
        measurement — that is precisely the §4AN mistake."""
        from ghost_agent.core.experiments import DEFAULT_SPECS
        spec = next(s for s in DEFAULT_SPECS if s.name == "verify_depth")
        assert "UNTESTED" in spec.description
        assert "GHOST_VERIFY_DEPTH_ROUTING=0" in spec.description

    def test_an_ON_DISK_registry_REPLACES_the_defaults_entirely(self,
                                                                tmp_path):
        """⚠ THE MECHANISM THAT MADE THIS SHIP 100% INERT (§4BR R17
        CRITICAL-1), pinned by execution.

        `DEFAULT_SPECS` is NOT what production reads. When
        `system/experiments.json` exists it replaces the defaults wholesale,
        so a spec added only to the Python constant is never assigned:
        `arm_for` returns "", every turn takes the control path, and the
        feature is dark while every test stays green. The live file's own
        `_comment` records this exact lesson from Track 1c — and the second
        version of the mistake was made anyway.

        Note this test could not be written against the ambient registry:
        conftest gives every test an isolated GHOST_HOME, so `load_registry()`
        here always falls back to DEFAULT_SPECS and would report success no
        matter what the operator's live file says. The omission has to be
        constructed.
        """
        import json
        from ghost_agent.core.experiments import load_registry
        p = tmp_path / "experiments.json"
        p.write_text(json.dumps({"experiments": [
            {"name": "risk_steer", "arms": ["control", "treatment"],
             "traffic": 1.0, "enabled": True, "scope": "live"}]}))
        specs = getattr(load_registry(p), "specs", {})
        assert "risk_steer" in specs
        assert "verify_depth" not in specs, (
            "a file that omits a spec must DROP it, not merge with the "
            "defaults — if this ever starts merging, the drift check below "
            "is no longer load-bearing and can be deleted")

    def test_the_LIVE_registry_lists_every_live_default(self):
        """The drift check the previous test makes necessary.

        Generalised past this one feature on purpose: it fails for the NEXT
        spec someone adds to DEFAULT_SPECS and forgets to list on disk,
        which is the actual recurring defect (twice now).
        """
        import json
        from pathlib import Path
        from ghost_agent.core.experiments import DEFAULT_SPECS
        live = Path(os.getenv("GHOST_LIVE_REGISTRY")
                    or "/Users/vasilis/Data/AI/Data/system/experiments.json")
        if not live.exists():
            pytest.skip(f"no live registry at {live}")
        on_disk = {e.get("name") for e in
                   json.loads(live.read_text()).get("experiments", [])}
        missing = {s.name for s in DEFAULT_SPECS
                   if s.enabled} - on_disk
        assert not missing, (
            f"live-scoped defaults missing from {live}: {sorted(missing)} — "
            "they are INERT in production until listed there")

    def test_the_trigger_key_is_registered(self):
        """Without a TRIGGER_KEYS entry `trigger_fired` is False for every
        trajectory, so `summarize(triggered_only=True)` renders an EMPTY
        block — and the "trigger has not fired yet" warning is suppressed
        too, because that warning is itself gated on membership. The arm
        would look like a population with no eligible turns rather than
        like a missing registration (§4BR R17 CRITICAL-2a)."""
        from ghost_agent.core.experiments import TRIGGER_KEYS, trigger_fired
        assert TRIGGER_KEYS.get("verify_depth") == "verify_depth_fired"

        class _T:
            extra = {"verify_depth_fired": True}
        assert trigger_fired(_T(), "verify_depth") is True

    def test_a_live_enrolment_actually_assigns_both_arms(self, tmp_path,
                                                         monkeypatch):
        """EXECUTION, not registration: run real enrolments and require the
        arm to be present and to SPLIT. A spec that is registered but never
        assigned is the same dark feature by another route."""
        from collections import Counter
        from ghost_agent.core.experiments import enroll_request

        class _Ctx:
            pass
        ctx = _Ctx()
        seen = Counter()
        for i in range(400):
            arms = enroll_request(ctx, f"req{i:05d}", eligible=True,
                                  origin="user")
            seen[arms.get("verify_depth", "<missing>")] += 1
        assert seen["<missing>"] == 0, seen
        assert seen["treatment"] > 100 and seen["control"] > 100, seen
