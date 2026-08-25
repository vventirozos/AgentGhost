"""§4CV — mining failed turns into verifiable-reward tasks for GEPA.

The load-bearing property is the ORACLE SELF-TEST, and specifically its
NEGATIVE half. A validator that accepts a correct answer proves nothing
on its own: one that `sys.exit(0)`s unconditionally accepts it too, and
then reports PASSED forever while teaching nothing. §4AO measured
skill-prune deciding 52% of its victims on noise — an oracle that cannot
fail is how noise gets manufactured at scale.

Everything else here is a refusal: containment inherited rather than
re-invented, `origin="bench"` so bench may teach and never grade,
text-graded items only for a metric that scores TEXT, staging separate
from arming the live flywheel, and a `None` (not 0.0) when the checker
could not run.
"""

import json
from pathlib import Path

import pytest

from ghost_agent.optim import env_mining as EM
from ghost_agent.optim.env_mining import (
    GRADED_ARTIFACT, GRADED_TEXT, MINING_EPOCH, PROBE_ARBITRARY,
    PROBE_SEPARATES, PROBE_TRIVIAL, PROBE_UNRUN, MineReport, MinedItem, Seed,
    mine, mine_seeds, oracle_is_sound, oracle_score, promote_to_bank,
    read_staging, solvability_probe, staging_path, synthesize,
    trainset_from_items, validator_static_defects, write_staging,
)


# ── validators used throughout ────────────────────────────────────────
GOOD = (
    "import sys\n"
    "a = open('answer.txt').read().strip()\n"
    "sys.exit(0 if a == '42' else 1)\n"
)
ALWAYS_PASSES = (
    "import sys\n"
    "open('answer.txt').read()\n"
    "sys.exit(0)\n"
)
PASSES_EMPTY = (
    "import sys\n"
    "a = open('answer.txt').read().strip()\n"
    "sys.exit(0 if a in ('42', '') else 1)\n"
)
REJECTS_ITS_OWN_REFERENCE = (
    "import sys\n"
    "a = open('answer.txt').read().strip()\n"
    "sys.exit(0 if a == 'something else' else 1)\n"
)
# ⚠ Genuinely unparseable. The first fixture here was
# `this is not python`, which — as ast.parse pointed out — is VALID
# Python: an `is not` comparison between two names. It reached the
# "no failure path" branch instead of the syntax branch, and the test
# asserting a syntax refusal was passing on the wrong guard.
CRASHES = (
    "import sys\n"
    "a = open('answer.txt').read()\n"
    "if a ==: sys.exit(1)\n"
)
NETWORKED = (
    "import sys, socket\n"
    "a = open('answer.txt').read().strip()\n"
    "sys.exit(0 if a == '42' else 1)\n"
)


def _item(validator=GOOD, ref="42", **kw):
    return MinedItem(item_id=kw.pop("item_id", "mined-test"),
                     challenge=kw.pop("challenge", "What is 6*7?"),
                     validation_script=validator,
                     reference_answer=ref, **kw)


class FakeLLM:
    def __init__(self, *payloads):
        self.queue = list(payloads)
        self.prompts = []

    async def chat_completion(self, payload, **kw):
        self.prompts.append(payload["messages"][0]["content"])
        if not self.queue:
            return {"choices": [{"message": {"content": ""}}]}
        nxt = self.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        body = nxt if isinstance(nxt, str) else json.dumps(nxt)
        return {"choices": [{"message": {"content": body}}]}


def _synth(challenge="What is 6*7?", validator=GOOD, ref="42"):
    return {"challenge": challenge, "validation_script": validator,
            "reference_answer": ref}


class _Traj:
    def __init__(self, **kw):
        self.id = kw.get("id", "t1")
        self.task_kind = kw.get("task_kind", "user_request")
        self.outcome = kw.get("outcome", "failed")
        self.tool_calls = kw.get("tool_calls", [{"name": "file_system"}])
        self.user_request = kw.get("user_request", "do a thing")
        self.final_response = kw.get("final_response", "I could not")
        self.failure_reason = kw.get("failure_reason", "")
        self.n_steps = kw.get("n_steps", 3)


# ══════════════════════════════════════════════════════════════════════
# 1. THE ORACLE SELF-TEST — both directions
# ══════════════════════════════════════════════════════════════════════
class TestOracleSoundness:
    def test_a_real_oracle_is_accepted(self):
        ok, why = oracle_is_sound(_item())
        assert ok, why
        assert "rejects both negative controls" in why

    def test_a_validator_that_ALWAYS_PASSES_is_refused(self):
        """THE central case. Envs-FORGE verifies the oracle accepts its
        reference; that alone admits this validator, which then reports
        PASSED forever and teaches nothing."""
        ok, why = oracle_is_sound(_item(ALWAYS_PASSES))
        assert not ok
        assert "cannot fail" in why and "label noise" in why

    def test_a_validator_that_accepts_the_EMPTY_string_is_refused(self):
        """The subtler half: this one DOES discriminate, and would pass a
        single-sentinel control, but an empty answer.txt is what a
        crashed or silent candidate produces."""
        ok, why = oracle_is_sound(_item(PASSES_EMPTY))
        assert not ok
        assert "empty string" in why

    def test_a_validator_that_rejects_its_own_reference_is_refused(self):
        ok, why = oracle_is_sound(_item(REJECTS_ITS_OWN_REFERENCE))
        assert not ok
        assert "REJECTS its own reference" in why

    def test_an_UNPARSEABLE_validator_is_refused_as_broken_not_strict(self):
        """⚠ A validator that does not compile exits non-zero on EVERY
        input, which through an exit code alone is indistinguishable from
        a very strict oracle. Parsing turns it into a static refusal that
        names the real problem."""
        ok, why = oracle_is_sound(_item(CRASHES))
        assert not ok
        assert "does not parse" in why and "strict oracle" in why

    def test_the_syntax_check_runs_BEFORE_any_execution(self):
        ran = []
        real = EM._run_validator
        EM._run_validator = lambda *a, **k: ran.append(1) or 0
        try:
            oracle_is_sound(_item(CRASHES))
        finally:
            EM._run_validator = real
        assert ran == [], "an unparseable validator was EXECUTED"

    def test_both_negative_controls_are_actually_exercised(self):
        """Negative control ON THE TEST: if only one control ran, a
        validator rejecting the sentinel but accepting "" would pass."""
        seen = []
        real = EM._run_validator

        def _spy(validator, answer, timeout_s=20.0):
            seen.append(answer)
            return real(validator, answer, timeout_s)
        EM._run_validator = _spy
        try:
            oracle_is_sound(_item())
        finally:
            EM._run_validator = real
        assert "42" in seen and "" in seen
        assert any(s.startswith("NOT-AN-ANSWER") for s in seen)

    def test_the_controls_are_mechanical_not_model_authored(self):
        """A model asked for a 'wrong answer' can produce an accidentally
        correct one, and the resulting rejection would be a real oracle
        failing a real answer — exactly backwards."""
        assert "" in EM._NEGATIVE_CONTROLS
        assert any(c.startswith("NOT-AN-ANSWER") for c in EM._NEGATIVE_CONTROLS)


class TestStaticRefusals:
    def test_a_networked_validator_is_refused_WITHOUT_running_it(self):
        """Static first: the execution IS the damage. A socket import has
        already reached the network by the time an execution-based check
        could notice."""
        ran = []
        real = EM._run_validator
        EM._run_validator = lambda *a, **k: ran.append(1) or 0
        try:
            ok, why = oracle_is_sound(_item(NETWORKED))
        finally:
            EM._run_validator = real
        assert not ok
        assert "socket" in why
        assert ran == [], "a network-importing validator was EXECUTED"

    @pytest.mark.parametrize("mod", ["urllib", "requests", "httpx", "aiohttp"])
    def test_every_listed_network_module_is_caught(self, mod):
        """⚠ COMMA-SEPARATED, deliberately. The first implementation used
        `\\bimport\\s+{mod}\\b`, which does not match `import sys, socket`
        — the most natural way to write the very thing being refused.
        Four of five modules sailed through a green suite."""
        v = f"import sys, {mod}\nopen('answer.txt')\nsys.exit(1)\n"
        assert any(mod in d for d in validator_static_defects(v))

    def test_an_ALIASED_import_is_caught(self):
        v = "import sys\nimport socket as s\nopen('answer.txt')\nsys.exit(1)\n"
        assert any("socket" in d for d in validator_static_defects(v))

    def test_a_SUBMODULE_import_is_caught(self):
        v = "import sys\nimport urllib.request\nopen('answer.txt')\nsys.exit(1)\n"
        assert any("urllib" in d for d in validator_static_defects(v))

    @pytest.mark.parametrize("dyn", ["__import__", "importlib"])
    def test_a_DYNAMIC_import_is_refused(self, dyn):
        """The AST name-set cannot see these, so the textual backstop
        stays for exactly these two forms."""
        v = f"import sys\nm = {dyn}\nopen('answer.txt')\nsys.exit(1)\n"
        assert any(dyn in d for d in validator_static_defects(v))

    def test_from_imports_are_caught_too(self):
        v = "import sys\nfrom urllib import request\nopen('answer.txt')\nsys.exit(1)\n"
        assert any("urllib" in d for d in validator_static_defects(v))

    def test_a_validator_that_never_reads_answer_txt_is_refused(self):
        v = "import sys\nsys.exit(0)\n"
        assert any("answer.txt" in d for d in validator_static_defects(v))

    def test_a_validator_with_no_failure_path_is_refused(self):
        v = "print(open('answer.txt').read())\n"
        assert any("no failure path" in d for d in validator_static_defects(v))

    def test_an_empty_validator_is_refused(self):
        assert validator_static_defects("") == ["empty validator"]

    def test_a_clean_validator_has_no_static_defects(self):
        """Negative control: without it every assertion above would pass
        on a function that always returns a defect."""
        assert validator_static_defects(GOOD) == []


# ══════════════════════════════════════════════════════════════════════
# 2. Seeds and containment
# ══════════════════════════════════════════════════════════════════════
class TestMineSeeds:
    def test_a_failed_tool_using_real_turn_is_a_seed(self):
        assert len(mine_seeds([_Traj()])) == 1

    @pytest.mark.parametrize("kw", [
        {"outcome": "passed"},
        {"outcome": "unknown"},
        {"task_kind": "reflection"},
        {"task_kind": "bench"},
        {"tool_calls": []},
        {"user_request": "   "},
    ])
    def test_everything_else_is_refused(self, kw):
        assert mine_seeds([_Traj(**kw)]) == []

    def test_a_forbidden_tool_disqualifies_the_whole_trajectory(self):
        """Containment is inherited from the module that FAILS CLOSED on
        it. A failure that needed the live world is not environment-able,
        exactly as it is not replayable."""
        t = _Traj(tool_calls=[{"name": "file_system"}, {"name": "browser"}])
        assert mine_seeds([t]) == []

    def test_the_containment_list_is_IMPORTED_not_copied(self):
        from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
        assert EM._forbidden() == frozenset(REPLAY_FORBIDDEN_TOOLS)

    def test_an_unreadable_containment_list_FAILS_CLOSED(self, monkeypatch):
        """Being unable to read the list is not permission to ignore it —
        an empty set would admit every seed the list exists to refuse."""
        import builtins
        real = builtins.__import__

        def _boom(name, *a, **k):
            if name.endswith("isolation") or "isolation" in name:
                raise ImportError("nope")
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", _boom)
        with pytest.raises(RuntimeError, match="containment"):
            EM._forbidden()

    def test_tool_names_are_normalised_before_the_check(self):
        t = _Traj(tool_calls=[{"name": "Web-Search"}])
        assert mine_seeds([t]) == []

    def test_the_limit_is_honoured(self):
        assert len(mine_seeds([_Traj(id=str(i)) for i in range(9)],
                              limit=4)) == 4

    def test_outcome_enums_are_read_by_value(self):
        class _O:
            value = "failed"
        assert len(mine_seeds([_Traj(outcome=_O())])) == 1

    def test_junk_rows_do_not_raise(self):
        assert mine_seeds([None, 7, "traj", object()]) == []


# ══════════════════════════════════════════════════════════════════════
# 3. Synthesis
# ══════════════════════════════════════════════════════════════════════
class TestSynthesis:
    async def test_a_well_formed_reply_becomes_an_item(self):
        item = await synthesize(Seed(trajectory_id="t1",
                                     user_request="q",
                                     tool_names=["execute"]),
                                FakeLLM(_synth()))
        assert item is not None
        assert item.validation_script == GOOD
        assert item.graded_on == GRADED_TEXT
        assert item.source_trajectory_id == "t1"
        assert item.epoch == MINING_EPOCH

    async def test_the_prompt_carries_the_request_and_the_tools(self):
        llm = FakeLLM(_synth())
        await synthesize(Seed(user_request="CANARY-REQ",
                              tool_names=["execute", "file_system"]), llm)
        assert "CANARY-REQ" in llm.prompts[0]
        assert "execute" in llm.prompts[0]

    async def test_the_failure_reason_is_included_when_present(self):
        llm = FakeLLM(_synth())
        await synthesize(Seed(user_request="q", tool_names=["execute"],
                              failure_reason="CANARY-REASON"), llm)
        assert "CANARY-REASON" in llm.prompts[0]

    @pytest.mark.parametrize("bad", [
        {"challenge": "", "validation_script": GOOD, "reference_answer": "42"},
        {"challenge": "c", "validation_script": "", "reference_answer": "42"},
        {"challenge": "c", "validation_script": GOOD, "reference_answer": "  "},
        {"nope": 1},
    ])
    async def test_an_incomplete_triple_is_a_reject_not_an_error(self, bad):
        assert await synthesize(Seed(user_request="q"), FakeLLM(bad)) is None

    async def test_unparseable_output_is_a_reject(self):
        assert await synthesize(Seed(user_request="q"),
                                FakeLLM("no json here")) is None

    async def test_an_llm_exception_is_a_reject_not_a_raise(self):
        assert await synthesize(Seed(user_request="q"),
                                FakeLLM(RuntimeError("down"))) is None

    async def test_a_missing_client_is_a_reject(self):
        assert await synthesize(Seed(user_request="q"), None) is None

    async def test_fenced_json_parses(self):
        llm = FakeLLM("```json\n" + json.dumps(_synth()) + "\n```")
        assert await synthesize(Seed(user_request="q"), llm) is not None

    async def test_item_ids_are_stable_and_distinct(self):
        a = await synthesize(Seed(trajectory_id="t1"), FakeLLM(_synth()))
        b = await synthesize(Seed(trajectory_id="t1"), FakeLLM(_synth()))
        c = await synthesize(Seed(trajectory_id="t2"), FakeLLM(_synth()))
        assert a.item_id == b.item_id and a.item_id != c.item_id


# ══════════════════════════════════════════════════════════════════════
# 4. Orchestration
# ══════════════════════════════════════════════════════════════════════
class TestMine:
    async def test_good_items_are_accepted_and_bad_ones_recorded(self):
        seeds = [Seed(trajectory_id="a"), Seed(trajectory_id="b")]
        llm = FakeLLM(_synth(), _synth(validator=ALWAYS_PASSES))
        rep = await mine(seeds, llm, probe=False)
        assert len(rep.accepted) == 1
        assert len(rep.rejected) == 1
        assert "cannot fail" in rep.rejected[0][1]

    async def test_every_rejection_carries_a_REASON(self):
        """A miner that reports only its successes cannot be debugged."""
        rep = await mine([Seed(trajectory_id="a")], FakeLLM("garbage"),
                         probe=False)
        assert rep.rejected and rep.rejected[0][1]

    async def test_a_synthesis_reject_does_not_count_as_synthesized(self):
        """Acceptance rate is the headline number; a failed synthesis in
        the denominator would deflate it and mask a weak oracle gate."""
        rep = await mine([Seed(trajectory_id="a")], FakeLLM("garbage"),
                         probe=False)
        assert rep.synthesized == 0 and rep.acceptance_rate is None

    async def test_a_high_acceptance_rate_is_WARNED_about(self):
        """⚠ Near-100% acceptance means the oracle self-test is not
        biting — the silent failure this module exists to prevent."""
        rep = await mine([Seed(trajectory_id=str(i)) for i in range(4)],
                         FakeLLM(*[_synth() for _ in range(4)]), probe=False)
        assert rep.acceptance_rate == pytest.approx(1.0)
        assert "⚠" in rep.summary() and "NOT BITING" in rep.summary()

    async def test_a_normal_acceptance_rate_is_NOT_warned_about(self):
        rep = MineReport(seeds=10, synthesized=10,
                         accepted=[_item() for _ in range(3)])
        assert "⚠" not in rep.summary()

    async def test_the_on_item_callback_never_breaks_the_run(self):
        rep = await mine([Seed(trajectory_id="a")], FakeLLM(_synth()),
                         probe=False, on_item=lambda *a: 1 / 0)
        assert len(rep.accepted) == 1


# ══════════════════════════════════════════════════════════════════════
# 3b. THE DETERMINACY PROBE — the gate the first LIVE run proved necessary
# ══════════════════════════════════════════════════════════════════════
class _Answers:
    """An LLM that returns a fixed sequence of ANSWERS to the challenge."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0
        self.temps = []

    async def chat_completion(self, payload, **kw):
        self.calls += 1
        self.temps.append(payload.get("temperature"))
        a = self.answers[(self.calls - 1) % len(self.answers)]
        if isinstance(a, Exception):
            raise a
        return {"choices": [{"message": {"content": a}}]}


class TestSolvabilityProbe:
    """⚠ THIS GATE EXISTS BECAUSE OF A REAL ITEM THAT PASSED EVERY OTHER
    CHECK. `mined-1399212a06bc` asked for a chess move and validated with
    `if data['move'] != 'e7e6': sys.exit(1)`. That oracle accepts its
    reference and rejects both negative controls — it demonstrably
    discriminates — but `e7e6` is one of MANY reasonable moves. The task
    is indeterminate and the validator encodes a preference as a fact.
    `oracle_is_sound` answers "can this checker fail?"; it cannot answer
    "is there one right answer?"."""

    async def test_an_ARBITRARY_item_is_refused(self):
        """No independent attempt reproduces the preferred answer."""
        llm = _Answers("d7d5", "g8f6", "c7c5", "b8c6")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_ARBITRARY
        assert not pr.usable
        assert "ARBITRARY" in pr.why

    async def test_a_TRIVIAL_item_is_refused(self):
        """Every candidate scores 1.0, so the item cannot discriminate
        between prompts — a constant column, which the optimizer can only
        lose by weighting."""
        llm = _Answers("42")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_TRIVIAL
        assert not pr.usable
        assert "DISCRIMINATE" in pr.why

    async def test_an_item_that_SEPARATES_is_accepted(self):
        llm = _Answers("42", "41", "42", "43")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_SEPARATES
        assert pr.usable and pr.passes == 2 and pr.attempts == 4

    async def test_the_probe_samples_at_NONZERO_temperature(self):
        """At temperature 0 the k samples are near-identical and the probe
        would measure the decoder rather than the task."""
        llm = _Answers("42", "41", "42", "43")
        await solvability_probe(_item().to_bank_row(), llm)
        assert all(t and t > 0 for t in llm.temps)

    async def test_it_makes_K_INDEPENDENT_attempts(self):
        llm = _Answers("42", "41", "42", "43")
        await solvability_probe(_item().to_bank_row(), llm, k=4)
        assert llm.calls == 4

    async def test_a_fenced_answer_is_extracted(self):
        """A determinate item must not be failed for formatting."""
        llm = _Answers("Here you go:\n```\n42\n```", "41", "```\n42\n```", "43")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.passes == 2

    async def test_an_UNRUNNABLE_probe_is_NOT_usable(self):
        """⚠ Unknown is not permission. An item admitted because its probe
        crashed is an item admitted for no reason."""
        llm = _Answers(RuntimeError("node down"))
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_UNRUN
        assert not pr.usable

    async def test_no_client_is_NOT_usable(self):
        pr = await solvability_probe(_item().to_bank_row(), None)
        assert pr.verdict == PROBE_UNRUN and not pr.usable

    async def test_an_empty_challenge_is_NOT_usable(self):
        row = _item().to_bank_row()
        row["challenge"] = "  "
        pr = await solvability_probe(row, _Answers("42"))
        assert not pr.usable


class TestMineRunsBothGates:
    async def test_a_sound_but_ARBITRARY_item_is_rejected_by_mine(self):
        """The end-to-end version of the live finding: the oracle gate
        passes it and the determinacy gate must not."""
        llm = FakeLLM(_synth(), "d7d5", "g8f6", "c7c5", "b8c6")
        rep = await mine([Seed(trajectory_id="a")], llm, probe=True)
        assert rep.accepted == []
        assert PROBE_ARBITRARY in rep.rejected[0][1]

    async def test_a_sound_AND_determinate_item_is_accepted(self):
        llm = FakeLLM(_synth(), "42", "41", "42", "43")
        rep = await mine([Seed(trajectory_id="a")], llm, probe=True)
        assert len(rep.accepted) == 1
        assert "probe" in rep.rejected[0][1] if rep.rejected else True

    async def test_the_report_SAYS_when_the_determinacy_gate_was_skipped(self):
        """"accepted" means something weaker without it, and a reader
        cannot tell from the count alone."""
        skipped = MineReport(seeds=1, synthesized=1, accepted=[_item()],
                             probed=False)
        assert "SKIPPED" in skipped.summary()
        both = MineReport(seeds=1, synthesized=1, accepted=[_item()],
                          probed=True)
        assert "oracle + determinacy" in both.summary()

    async def test_an_unprobed_run_does_not_call_the_model_extra_times(self):
        llm = FakeLLM(_synth())
        await mine([Seed(trajectory_id="a")], llm, probe=False)
        assert len(llm.prompts) == 1


class TestTheProbeCanActuallyPASS:
    """⚠ EVERY TEST ABOVE PASSED WHILE THE LIVE PROBE COULD NOT ACCEPT
    ANYTHING. 12 of 12 mined items were rejected as "arbitrary" by a gate
    that had never once asked the task successfully — the numbers looked
    like a finding about the items and were a finding about the probe.

    Two independent causes, both pinned here:

      1. `max_tokens: 1200` with thinking left ON. The reasoning phase
         consumed the whole budget and content came back EMPTY. That is
         VERBATIM the defect `scripts/run_gepa.py`'s A/B runner already
         carries a paragraph about ("a gate that can only ever reject"),
         rebuilt in a module whose docstring cites it.
      2. The synthesis prompt told the model the validator reads
         `answer.txt`, so every generated challenge instructed the solver
         to CREATE FILES — unanswerable by a GEPA rollout, which has no
         tools and no filesystem.

    The fakes above could not see either: they always returned non-empty
    text and never inspected the payload.
    """

    async def test_the_probe_disables_thinking_and_uses_the_full_budget(self):
        seen = {}

        class _Capture:
            async def chat_completion(self, payload, **kw):
                seen.update(payload)
                return {"choices": [{"message": {"content": "42"}}]}
        await solvability_probe(_item().to_bank_row(), _Capture(), k=1)
        assert seen.get("chat_template_kwargs", {}).get(
            "enable_thinking") is False, (
            "thinking left ON starves the content phase and every reply "
            "comes back empty")
        assert seen.get("max_tokens", 0) >= 8192

    async def test_an_EMPTY_reply_is_NOT_a_failed_attempt(self):
        """An empty generation means the task was never actually asked."""
        llm = _Answers("", "", "", "")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.attempts == 0
        assert pr.no_output == 4
        assert pr.verdict == PROBE_UNRUN
        assert pr.verdict != PROBE_ARBITRARY

    async def test_a_whitespace_only_reply_is_also_unscored(self):
        pr = await solvability_probe(_item().to_bank_row(),
                                     _Answers("   \n  ", "", " ", "\t"))
        assert pr.verdict == PROBE_UNRUN and not pr.usable

    async def test_a_MIXED_run_scores_only_the_real_replies(self):
        """One empty sample must not deflate the denominator into a
        verdict the data does not support."""
        pr = await solvability_probe(_item().to_bank_row(),
                                     _Answers("42", "", "41", "42"))
        # ⚠ TWO COUNTERS, NOT ONE. An empty generation never enters
        # `attempts`; folding it into `unscored` too would subtract it
        # from the denominator TWICE, and this exact run then read
        # TRIVIAL (scored=2, passes=2) — a mixed item promoted into a
        # rejection band by a counting bug in the fix for a counting bug.
        assert pr.attempts == 3 and pr.passes == 2
        assert pr.unscored == 0 and pr.no_output == 1
        assert pr.verdict == PROBE_SEPARATES

    async def test_a_CHECKER_failure_and_an_EMPTY_reply_are_counted_apart(self):
        import ghost_agent.optim.env_mining as _EM
        real = _EM.oracle_score
        calls = {'n': 0}
        def _flaky(row, text, **kw):
            calls['n'] += 1
            return None if calls['n'] == 1 else real(row, text, **kw)
        _EM.oracle_score = _flaky
        try:
            pr = await solvability_probe(_item().to_bank_row(),
                                         _Answers("42", "", "41", "42"))
        finally:
            _EM.oracle_score = real
        assert pr.no_output == 1      # the empty reply
        assert pr.unscored == 1       # the checker failure
        assert pr.attempts == 3       # both empties excluded from attempts
        assert pr.attempts - pr.unscored == 2

    async def test_the_unrun_reason_says_the_task_was_never_asked(self):
        pr = await solvability_probe(_item().to_bank_row(), _Answers(""))
        assert "never actually asked and answered" in pr.why


class TestAgentShapedChallengesAreRefused:
    """The solver in a GEPA rollout answers in ONE reply with no tools.
    A challenge that tells it to create files is unanswerable, and the
    probe would score its prose against a checker expecting a value."""

    @pytest.mark.parametrize("challenge", [
        "Compute the total and write it to answer.txt",
        "Create a file called items.txt with this content, then sum it",
        "Create a directory called inventory and populate it",
        "Parse the data and write the result to results.json",
        "Create a script that prints the answer",
    ])
    async def test_a_filesystem_instruction_is_refused(self, challenge):
        item = await synthesize(Seed(user_request="q"),
                                FakeLLM(_synth(challenge=challenge)))
        assert item is None, f"agent-shaped challenge accepted: {challenge!r}"

    @pytest.mark.parametrize("challenge", [
        "What is 6*7? Reply with only the number.",
        "Count the pieces on this board and reply with the count.",
        "Given this CSV, reply with the total value to 2 decimal places.",
    ])
    async def test_a_TEXT_challenge_is_accepted(self, challenge):
        """Negative control — without it the refusal above could be a
        function that rejects everything."""
        item = await synthesize(Seed(user_request="q"),
                                FakeLLM(_synth(challenge=challenge)))
        assert item is not None, f"text challenge refused: {challenge!r}"

    async def test_the_synthesis_prompt_TELLS_the_model_about_the_solver(self):
        """The refusal above is a backstop; the prompt is the fix. If the
        prompt stops saying the solver has no tools, the model goes back
        to writing filesystem tasks and the backstop eats the whole run."""
        llm = FakeLLM(_synth())
        await synthesize(Seed(user_request="q"), llm)
        p = llm.prompts[0].lower()
        assert "no tools" in p
        assert "one reply" in p
        assert "never instruct the solver to create files" in p
        # And the JSON schema at the tail must not contradict it — it
        # described "the exact correct contents of answer.txt" while the
        # rules above said the solver has no filesystem.
        assert "contents of answer.txt" not in p

    async def test_the_prompt_demands_determinacy_explicitly(self):
        llm = FakeLLM(_synth())
        await synthesize(Seed(user_request="q"), llm)
        assert "exactly ONE correct answer" in llm.prompts[0]


# ══════════════════════════════════════════════════════════════════════
# 5. Staging is NOT arming
# ══════════════════════════════════════════════════════════════════════
class TestStagingVsPromotion:
    def test_write_staging_does_NOT_touch_the_bank_directory(self, tmp_path):
        """`pick_next_item` walks EVERY bank and the biological watchdog
        calls it in production — writing there arms a live loop."""
        write_staging([_item()], "ghost_failures", str(tmp_path))
        banks = tmp_path / "system" / "bench" / "banks"
        assert not banks.exists() or list(banks.glob("*.jsonl")) == []

    def test_staging_lands_under_system_optim(self, tmp_path):
        p = write_staging([_item()], "ghost_failures", str(tmp_path))
        assert p.is_file()
        assert "mined_envs" in str(p) and "banks" not in str(p)

    def test_a_round_trip_preserves_the_bank_schema(self, tmp_path):
        write_staging([_item()], "ghost_failures", str(tmp_path))
        rows = read_staging("ghost_failures", str(tmp_path))
        assert len(rows) == 1
        for k in ("bank", "item_id", "cluster", "challenge",
                  "setup_script", "validation_script", "graded_on"):
            assert k in rows[0], f"{k} missing — eval.banks needs it"

    def test_rows_from_another_epoch_are_dropped(self, tmp_path):
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        row = _item().to_bank_row()
        old = dict(row, mining_epoch="e0", item_id="old")
        p.write_text(json.dumps(old) + "\n" + json.dumps(row) + "\n")
        rows = read_staging("ghost_failures", str(tmp_path))
        assert [r["item_id"] for r in rows] == ["mined-test"]

    def test_malformed_lines_are_skipped(self, tmp_path):
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("NOT JSON\n[1,2]\n" + json.dumps(_item().to_bank_row()) + "\n")
        assert len(read_staging("ghost_failures", str(tmp_path))) == 1

    def test_promote_is_what_writes_the_bank(self, tmp_path):
        write_staging([_item()], "ghost_failures", str(tmp_path))
        out = promote_to_bank("ghost_failures", str(tmp_path))
        assert out is not None and out.is_file()
        assert "banks" in str(out)

    def test_promote_with_nothing_staged_returns_None(self, tmp_path):
        assert promote_to_bank("ghost_failures", str(tmp_path)) is None

    def test_the_promoted_bank_is_loadable_by_eval_banks(self, tmp_path):
        from ghost_agent.eval.banks import load_bank
        write_staging([_item()], "ghost_failures", str(tmp_path))
        promote_to_bank("ghost_failures", str(tmp_path))
        assert len(load_bank("ghost_failures", str(tmp_path))) == 1


# ══════════════════════════════════════════════════════════════════════
# 6. The GEPA bridge
# ══════════════════════════════════════════════════════════════════════
class TestTrainsetBridge:
    def test_a_text_graded_item_becomes_an_example(self):
        ex = trainset_from_items([_item().to_bank_row()], "planning.decompose")
        assert len(ex) == 1
        assert ex[0].signature_name == "planning.decompose"
        assert ex[0].inputs["user_request"] == "What is 6*7?"
        assert ex[0].expected_output["final_response"] == "42"

    def test_origin_is_bench_so_it_can_TEACH_but_never_GRADE(self):
        """§4BH doctrine, enforced by one field: `real_only_gate` evicts
        bench examples from the PRIVATE ship-gate tier."""
        ex = trainset_from_items([_item().to_bank_row()], "s")
        assert ex[0].origin == "bench"

    def test_the_real_only_gate_ACTUALLY_evicts_them(self):
        """Executed, not asserted — the tag only matters because that
        function reads it, so drive the real function."""
        from ghost_agent.optim.trainset import TrainExample, real_only_gate
        mined = trainset_from_items([_item().to_bank_row()], "s")
        real = [TrainExample(signature_name="s", inputs={"user_request": "r"},
                             expected_output={"final_response": "x"},
                             origin="user_request")]
        # The public tier must hold real examples: `real_only_gate`
        # re-caps bench at EQUAL MASS after the move, so moving into an
        # empty public tier correctly drops it (bench must never
        # outnumber real). Driving it with a realistic public tier is
        # what exercises the eviction rather than the cap.
        pub_real = [TrainExample(signature_name="s",
                                 inputs={"user_request": f"r{i}"},
                                 expected_output={"final_response": "x"},
                                 origin="user_request") for i in range(3)]
        pub, priv, moved = real_only_gate(pub_real, mined + real)
        assert moved == 1
        assert all(e.origin != "bench" for e in priv)
        assert any(e.origin == "bench" for e in pub)

    def test_bench_can_never_OUTNUMBER_real_in_the_public_tier(self):
        """The equal-mass re-cap, driven: mining 50 items must not turn
        the optimizer's trainset bench-majority."""
        from ghost_agent.optim.trainset import TrainExample, real_only_gate
        mined = trainset_from_items(
            [_item(item_id=f"m{i}").to_bank_row() for i in range(50)], "s")
        real = [TrainExample(signature_name="s",
                             inputs={"user_request": f"r{i}"},
                             expected_output={"final_response": "x"},
                             origin="user_request") for i in range(4)]
        pub, _priv, _m = real_only_gate(real, mined)
        n_bench = sum(1 for e in pub if e.origin == "bench")
        n_real = sum(1 for e in pub if e.origin != "bench")
        assert n_bench <= n_real, f"bench {n_bench} > real {n_real}"

    def test_an_ARTIFACT_graded_item_is_refused(self):
        """GEPA optimises an instruction that produces TEXT; there is no
        agent loop in a rollout to write a solution.py, so an artifact
        item would be a metric that can only reject."""
        row = _item().to_bank_row()
        row["graded_on"] = GRADED_ARTIFACT
        assert trainset_from_items([row], "s") == []

    def test_an_item_with_no_graded_on_defaults_to_REFUSED(self):
        """Absent must mean artifact (eval.banks' own default), so an old
        row cannot silently become a text target."""
        row = _item().to_bank_row()
        row.pop("graded_on")
        assert trainset_from_items([row], "s") == []

    @pytest.mark.parametrize("row", [None, 7, "row", {}, {"challenge": ""}])
    def test_junk_rows_are_skipped(self, row):
        assert trainset_from_items([row], "s") == []

    def test_an_item_without_a_reference_is_skipped(self):
        row = _item(ref="   ").to_bank_row()
        assert trainset_from_items([row], "s") == []


class TestOracleScore:
    def test_a_correct_answer_scores_one(self):
        assert oracle_score(_item().to_bank_row(), "42") == 1.0

    def test_a_wrong_answer_scores_zero(self):
        assert oracle_score(_item().to_bank_row(), "43") == 0.0

    def test_an_UNRUNNABLE_checker_scores_NONE_not_zero(self, monkeypatch):
        """⚠ Scoring an infrastructure failure as 'the candidate was
        wrong' optimises against noise. None forces the caller to decide;
        0.0 would quietly punish a prompt for a broken box.

        ⚠ AND THE HONEST LIMIT, stated rather than faked: a validator
        that COMPILES and then raises at runtime exits non-zero, which a
        single call cannot tell from a rejection. That is why the gate is
        `oracle_is_sound`, which observes the validator on THREE inputs —
        a crash there fails the reference check and the item never
        ships. This test drives the case that genuinely cannot run."""
        monkeypatch.setattr(EM, "_run_validator", lambda *a, **k: None)
        assert oracle_score(_item().to_bank_row(), "42") is None

    def test_a_missing_validator_scores_NONE(self):
        assert oracle_score({"validation_script": ""}, "42") is None
        assert oracle_score({}, "42") is None

    def test_it_beats_token_overlap_on_the_case_that_matters(self):
        """The reason the module exists. A reply that LOOKS like the
        reference but is wrong scores 0; one that looks nothing like it
        but is right scores 1. Token overlap gets both backwards."""
        row = _item().to_bank_row()
        assert oracle_score(row, "The answer is 43") == 0.0
        assert oracle_score(row, "42") == 1.0


# ══════════════════════════════════════════════════════════════════════
# 7. The feature cannot be silently deleted
# ══════════════════════════════════════════════════════════════════════
class TestAbsenceIsDetectable:
    def test_the_oracle_gate_is_reachable_in_BOTH_directions(self):
        assert oracle_is_sound(_item())[0] is True
        assert oracle_is_sound(_item(ALWAYS_PASSES))[0] is False

    def test_the_negative_control_list_is_not_empty(self):
        """A reviewer emptying `_NEGATIVE_CONTROLS` must not leave the
        suite green — that is §4CS item B's own blind spot."""
        assert len(EM._NEGATIVE_CONTROLS) >= 2

    def test_the_forbidden_import_list_is_not_empty(self):
        assert len(EM._FORBIDDEN_VALIDATOR_IMPORTS) >= 5

    async def test_the_end_to_end_path_produces_a_GEPA_example(self, tmp_path):
        """Seeds → synthesis → oracle gate → staging → trainset, driven
        through the real functions with real subprocess validation."""
        seeds = mine_seeds([_Traj(id="t1", user_request="compute 6*7")])
        assert seeds
        rep = await mine(seeds, FakeLLM(_synth()), probe=False)
        assert rep.accepted
        write_staging(rep.accepted, "ghost_failures", str(tmp_path))
        rows = read_staging("ghost_failures", str(tmp_path))
        ex = trainset_from_items(rows, "planning.decompose")
        assert len(ex) == 1
        assert oracle_score(rows[0], "42") == 1.0


# ══════════════════════════════════════════════════════════════════════
# 8. CONFINEMENT — the round-1 CRITICAL, found by two reviewers
# ══════════════════════════════════════════════════════════════════════
class TestValidatorsRunUnderTheKernelSandbox:
    """⚠ ROUND 1 RAN MODEL-AUTHORED PYTHON WITH THE AGENT'S OWN
    PRIVILEGES. `subprocess.run(..., cwd=tmpdir)` is not confinement, and
    the containment was a tuple of 11 module names. Two reviewers
    independently walked 13 of 15 bypasses straight through it —
    `os.system("curl …")`, `subprocess.run(["/usr/bin/curl"])`,
    `import http.client` (top-level name is "http"), `ctypes`,
    `webbrowser`, `shutil.rmtree` of the live bank directory, reading
    `~/.ghost_api_key` — while `oracle_is_sound` reported "accepts its
    reference, rejects both negative controls".

    A deny-list of module NAMES is a lexical proxy for "cannot touch the
    box". §4CI: every exemption in a deny-list becomes the next bypass.
    §4CR: closing this class needs the child UNDER the sandbox, not
    beside it — and `evolve/confine.py` already existed for exactly this.

    The static checks stay as a cheap pre-filter. They are no longer the
    containment.
    """

    ESCAPER = ("import sys, os\n"
               "a = open('answer.txt').read().strip()\n"
               "try:\n"
               "    open({path!r}, 'w').write('escaped')\n"
               "except Exception:\n"
               "    pass\n"
               "sys.exit(0 if a == '42' else 1)\n")

    def _run(self, tmp_path):
        esc = tmp_path / "ESCAPED.txt"
        v = self.ESCAPER.format(path=str(esc))
        it = MinedItem(item_id="x", challenge="c", validation_script=v,
                       reference_answer="42")
        ok, why = oracle_is_sound(it)
        return it, ok, why, esc

    def test_the_denylist_does_NOT_catch_this_at_all(self, tmp_path):
        """The pre-filter is honestly weak, and the test says so rather
        than implying the list is the defence."""
        v = self.ESCAPER.format(path=str(tmp_path / "x"))
        assert validator_static_defects(v) == []

    @pytest.mark.skipif(not __import__(
        "ghost_agent.evolve.confine", fromlist=["available"]).available()[0],
        reason="sandbox-exec unavailable on this platform")
    def test_a_write_OUTSIDE_the_workdir_is_BLOCKED(self, tmp_path):
        it, ok, why, esc = self._run(tmp_path)
        assert it.confined is True
        assert not esc.exists(), "model-authored code escaped its workdir"
        assert ok, why

    def test_the_NEGATIVE_CONTROL_escapes_when_confinement_is_off(
            self, tmp_path, monkeypatch):
        """⚠ THE HALF THAT MAKES THE OTHER HALF MEAN ANYTHING (§4CR). If
        the escape does not succeed unconfined, the confined result
        proves nothing about the sandbox."""
        from ghost_agent.evolve.confine import ENV_FLAG
        monkeypatch.setenv(ENV_FLAG, "0")
        it, ok, why, esc = self._run(tmp_path)
        assert it.confined is False
        assert esc.exists(), (
            "the escape did not succeed even unconfined — this test can no "
            "longer prove the sandbox does anything")
        assert "UNCONFINED" in why

    def test_an_unconfined_run_SAYS_SO_on_the_item(self, tmp_path,
                                                   monkeypatch):
        """A cascade that silently ran unconfined while claiming
        otherwise would be worse than never having this."""
        from ghost_agent.evolve.confine import ENV_FLAG
        monkeypatch.setenv(ENV_FLAG, "0")
        it = MinedItem(item_id="x", challenge="c", validation_script=GOOD,
                       reference_answer="42")
        ok, why = oracle_is_sound(it)
        assert ok and it.confined is False
        assert "UNCONFINED" in why and "is off" in why
        assert it.to_bank_row()["verified_confined"] is False

    def test_REQUIRE_CONFINE_refuses_rather_than_running_unconfined(
            self, tmp_path, monkeypatch):
        from ghost_agent.evolve.confine import ENV_FLAG
        monkeypatch.setenv(ENV_FLAG, "0")
        monkeypatch.setenv("GHOST_MINE_REQUIRE_CONFINE", "1")
        it = MinedItem(item_id="x", challenge="c", validation_script=GOOD,
                       reference_answer="42")
        ok, why = oracle_is_sound(it)
        assert ok is False
        assert "unconfined" in why

    def test_the_policy_file_is_NOT_writable_by_the_child(self, tmp_path):
        """⚠ `confine`'s own docstring: the policy must sit somewhere the
        child cannot write, or the confinement is advisory."""
        from ghost_agent.optim import env_mining as _EM
        seen = {}
        real = _EM._confined_cmd

        def _spy(cmd, workdir):
            out = real(cmd, workdir)
            seen["workdir"] = workdir
            seen["cmd"] = out[0]
            return out
        _EM._confined_cmd = _spy
        try:
            oracle_is_sound(_item())
        finally:
            _EM._confined_cmd = real
        pol = [a for a in seen.get("cmd", []) if str(a).endswith(".sb")]
        if pol:
            assert not str(pol[0]).startswith(str(seen["workdir"]) + "/"), (
                "the policy file sits inside the directory the child may "
                "write — the confinement would be advisory")


# ══════════════════════════════════════════════════════════════════════
# 9. THE CONSUMER — a reviewer found this loop was built but UNWIRED
# ══════════════════════════════════════════════════════════════════════
class TestTheOracleActuallyReachesGEPA:
    """⚠ ROUND 1 SHIPPED §4CV's ONE NOVEL MECHANISM WITH NO CALLER.
    `grep` over src/, scripts/ and tests/ found `oracle_score` only in
    its own module and its own test file; `trainset_from_items` was
    called once, by the miner script, which printed a count and
    DISCARDED the list. `scripts/run_gepa.py` had no flag that accepts a
    mined set, so a promoted item would still have been scored by the
    token-overlap metric the module's opening paragraph exists to
    replace. That is `built-but-unwired-loops`, in a module whose scope
    argument is "this is a GEPA trainset producer".
    """

    def _src(self):
        from pathlib import Path
        return Path("scripts/run_gepa.py").read_text()

    def test_run_gepa_accepts_a_mined_bank(self):
        assert "--mined-bank" in self._src()

    def test_run_gepa_CALLS_the_oracle_metric(self):
        s = self._src()
        assert "oracle_score" in s, (
            "the executable oracle is the point of the module; a mined "
            "bank scored by token overlap is the thing it replaces")
        assert "trainset_from_items" in s

    def test_the_metric_does_NOT_score_an_unrunnable_checker_as_zero(self):
        s = self._src()
        assert "if x is not None" in s, (
            "oracle_score returns None when the checker could not RUN; "
            "calling that 0.0 optimises against noise")
        assert "_extract_answer" in s, (
            "the metric must score the SAME extracted string the probe "
            "scored, or the gate that admitted the item and the metric "
            "that trains on it ask different questions")

    def test_run_gepa_REPORTS_whether_the_oracle_fired(self):
        """A mined bank whose checker never ran is a trainset scored by
        overlap wearing a verifiable-reward label."""
        s = self._src()
        assert "NEVER" in s and "FIRED" in s

    def test_the_flag_is_OPT_IN(self):
        """A trainset that silently changed shape would invalidate every
        comparison against a previous run."""
        assert 'parser.add_argument("--mined-bank", default=None' in self._src()

    def test_run_gepa_still_parses_and_advertises_the_flag(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "scripts/run_gepa.py", "--help"],
            capture_output=True, text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin",
                 "HOME": str(__import__("pathlib").Path.home())})
        assert r.returncode == 0, r.stderr[-500:]
        assert "--mined-bank" in r.stdout

    def test_the_join_key_survives_the_bridge(self):
        """The metric looks the row up by REQUEST TEXT, so the challenge
        that lands in `inputs["user_request"]` must be the same string
        the bank row carries — pinned because a silent mismatch would
        make the oracle never fire while everything looked wired."""
        row = _item().to_bank_row()
        ex = trainset_from_items([row], "planning.decompose")[0]
        assert ex.inputs["user_request"] == row["challenge"]
        assert ex.source_trajectory_id == row["item_id"]


class TestNondeterministicOraclesAreRefused:
    """⚠ `random` is not on the import denylist, and round 1 ran the
    reference check ONCE — so a validator that coin-flips after a
    correct-answer check was admitted in roughly half of runs (a
    reviewer measured 7 of 12). An oracle that disagrees with itself
    writes a different verdict for the same answer on different days."""

    COINFLIP = (
        "import sys, random\n"
        "a = open('answer.txt').read().strip()\n"
        "if a != '42':\n"
        "    sys.exit(1)\n"
        "sys.exit(0 if random.random() < 0.5 else 1)\n"
    )

    def test_the_KNOWN_LEAK_is_written_down_and_bounded(self):
        """⚠ NO FINITE REPEAT COUNT EXCLUDES AN ADVERSARIAL COIN-FLIP,
        and two earlier versions of this test tried to pretend otherwise.

        The first asserted the NONDETERMINISTIC message within 12 tries —
        flaky at ~3%, because that message needs run 1 to pass AND run 2
        to fail. The second asserted a coin-flip is ALWAYS refused, which
        is simply FALSE: with N identical runs a fair coin ships at
        0.5**(N-1).

        A pin that fails 3% of the time gets muted, and a pin that
        asserts a false property gets deleted. So the honest pin is on
        the BOUND: the constant exists, it is at least 4, and the module
        says out loud that this is a filter with a known leak. The
        mechanism itself is pinned deterministically below."""
        assert EM._REFERENCE_REPEATS >= 4
        src = __import__("inspect").getsource(EM)
        assert "known leak, not a proof" in src, (
            "the residual must stay documented — a filter presented as a "
            "proof is how a leak becomes a surprise")

    def test_the_reference_is_scored_REFERENCE_REPEATS_times(self,
                                                             monkeypatch):
        seen = []
        real = EM._run_validator
        monkeypatch.setattr(
            EM, "_run_validator",
            lambda v, a, timeout_s=20.0: (seen.append(a),
                                          real(v, a, timeout_s))[1])
        oracle_is_sound(_item())
        assert seen.count("42") == EM._REFERENCE_REPEATS

    def test_the_NONDETERMINISM_branch_is_driven_deterministically(
            self, monkeypatch):
        """The mechanism itself, with the randomness removed: the same
        reference answer scoring 0 then 1 must be refused as
        nondeterministic rather than accepted on the first result."""
        codes = iter([0] + [1] * EM._REFERENCE_REPEATS)
        monkeypatch.setattr(EM, "_run_validator",
                            lambda v, a, timeout_s=20.0: next(codes))
        ok, why = oracle_is_sound(_item())
        assert ok is False
        assert "NONDETERMINISTIC" in why
        assert "scored 0 then 1" in why

    def test_a_stable_reference_passes_the_repeat(self, monkeypatch):
        """Negative control: identical codes must NOT trip the check, or
        it eats the whole (already thin) funnel."""
        monkeypatch.setattr(
            EM, "_run_validator",
            lambda v, a, timeout_s=20.0: 0 if a == "42" else 1)
        ok, why = oracle_is_sound(_item())
        assert ok is True, why

    def test_a_DETERMINISTIC_validator_is_not_refused_by_the_repeat(self):
        """Negative control: the repeat check must not reject sound
        oracles, or it eats the whole (already thin) funnel."""
        for _ in range(5):
            ok, why = oracle_is_sound(_item())
            assert ok, why

    def test_the_reference_is_actually_run_TWICE(self):
        seen = []
        real = EM._run_validator

        def _spy(v, a, timeout_s=20.0):
            seen.append(a)
            return real(v, a, timeout_s)
        EM._run_validator = _spy
        try:
            oracle_is_sound(_item())
        finally:
            EM._run_validator = real
        assert seen.count("42") == EM._REFERENCE_REPEATS, seen


# ══════════════════════════════════════════════════════════════════════
# 10. The guards reviewer D showed were UNPINNED
# ══════════════════════════════════════════════════════════════════════
class TestTheSandboxCannotLIE:
    """⚠ A reviewer made `_confined_cmd`'s except path return
    `confined=True` and 128 of 128 passed — while the validator WROTE
    OUTSIDE ITS WORKDIR and `oracle_is_sound` reported it sound. Both
    existing confinement tests drive `GHOST_EVOLVE_CONFINE=0`; the
    FAILURE path (import error, unwritable policy, non-macOS) was
    untested. `silent-inoperative-subsystems` inside the fix for the
    round-1 critical."""

    def test_a_failed_confine_reports_UNCONFINED_not_confined(
            self, monkeypatch):
        import ghost_agent.optim.env_mining as _EM

        def _boom(*a, **k):
            raise RuntimeError("confine module exploded")
        monkeypatch.setattr(_EM, "_confined_cmd",
                            lambda cmd, wd: _EM._confined_cmd.__wrapped__(cmd, wd)
                            if False else (list(cmd), False, "boom"))
        it = _item()
        ok, why = oracle_is_sound(it)
        assert it.confined is False
        assert "UNCONFINED" in why

    def test_an_IMPORT_failure_does_not_claim_confinement(self, monkeypatch):
        """Drives the real except path rather than stubbing past it."""
        import builtins
        import ghost_agent.optim.env_mining as _EM
        real = builtins.__import__

        def _no_confine(name, *a, **k):
            if "confine" in name:
                raise ImportError("nope")
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", _no_confine)
        cmd, confined, why = _EM._confined_cmd(["python3", "x.py"],
                                               __import__("pathlib").Path("/tmp/w"))
        assert confined is False
        assert cmd == ["python3", "x.py"]
        assert why

    def test_an_unconfined_run_is_REFUSED_when_required(self, monkeypatch):
        import ghost_agent.optim.env_mining as _EM
        monkeypatch.setattr(_EM, "_confined_cmd",
                            lambda cmd, wd: (list(cmd), False, "stubbed"))
        monkeypatch.setenv("GHOST_MINE_REQUIRE_CONFINE", "1")
        ok, why = oracle_is_sound(_item())
        assert ok is False
        assert "REQUIRE_CONFINE" in why and "unconfined" in why


class TestTheSandboxPolicyDeniesNetwork:
    """A reviewer flipped `allow_network=False` to `True` and 128 of 128
    passed — the generated policy really did lose `(deny network*)`."""

    def test_the_generated_policy_denies_network(self):
        import ghost_agent.optim.env_mining as _EM
        from pathlib import Path
        seen = {}
        real = _EM._confined_cmd

        def _spy(cmd, workdir):
            out = real(cmd, workdir)
            pol = [a for a in out[0] if str(a).endswith(".sb")]
            if pol:
                seen["policy"] = Path(pol[0]).read_text()
            return out
        _EM._confined_cmd = _spy
        try:
            oracle_is_sound(_item())
        finally:
            _EM._confined_cmd = real
        if "policy" not in seen:
            pytest.skip("sandbox-exec unavailable on this platform")
        assert "(deny network*)" in seen["policy"], (
            "the validator sandbox permits network access")
        assert "(deny file-write*)" in seen["policy"]


class TestAnUnrunnableNegativeControlIsNotAPass:
    """⚠ A reviewer replaced `return False` with `continue` on the
    negative-control run and 128 of 128 passed — an always-exit-0
    validator then returned the literally FALSE string 'accepts its
    reference, rejects both negative controls'. This module's own
    most-cited doctrine ('None and a non-zero exit are DIFFERENT'), and
    it was unpinned at the one site that matters."""

    def test_an_unscorable_control_refuses_the_item(self, monkeypatch):
        import ghost_agent.optim.env_mining as _EM
        real = _EM._run_validator
        calls = {"n": 0}

        def _flaky(v, a, timeout_s=20.0):
            calls["n"] += 1
            if calls["n"] <= EM._REFERENCE_REPEATS:   # reference runs OK
                return real(v, a, timeout_s)
            return None                  # every control run is unrunnable
        monkeypatch.setattr(_EM, "_run_validator", _flaky)
        ok, why = oracle_is_sound(_item())
        assert ok is False
        assert "negative control" in why

    def test_and_the_reason_is_not_a_false_claim(self, monkeypatch):
        import ghost_agent.optim.env_mining as _EM
        real = _EM._run_validator
        calls = {"n": 0}

        def _flaky(v, a, timeout_s=20.0):
            calls["n"] += 1
            return (real(v, a, timeout_s)
                    if calls["n"] <= EM._REFERENCE_REPEATS else None)
        monkeypatch.setattr(_EM, "_run_validator", _flaky)
        _ok, why = oracle_is_sound(_item(ALWAYS_PASSES))
        assert "rejects both negative controls" not in why


class TestMineRefusesEVERY_UnusableVerdict:
    """A reviewer narrowed `if not pr.usable:` to
    `if pr.verdict == PROBE_ARBITRARY:` and 128 of 128 passed — a TRIVIAL
    item went from accepted=0 to accepted=1. "Unknown is not permission"
    was asserted only on `ProbeResult.usable`, never THROUGH `mine`."""

    async def test_a_TRIVIAL_item_is_not_accepted_by_mine(self):
        llm = FakeLLM(_synth(), "42", "42", "42", "42")
        rep = await mine([Seed(trajectory_id="a")], llm, probe=True)
        assert rep.accepted == []
        assert PROBE_TRIVIAL in rep.rejected[0][1]

    async def test_an_UNPROBEABLE_item_is_not_accepted_by_mine(self):
        llm = FakeLLM(_synth(), "", "", "", "")
        rep = await mine([Seed(trajectory_id="a")], llm, probe=True)
        assert rep.accepted == []
        assert PROBE_UNRUN in rep.rejected[0][1]

    async def test_an_unprobeable_item_is_COUNTED_as_an_instrument_failure(self):
        llm = FakeLLM(_synth(), "", "", "", "")
        rep = await mine([Seed(trajectory_id="a")], llm, probe=True)
        assert rep.unprobed == 1
        assert "NOT the gate being strict" in rep.summary()


class TestTheProbeHonoursK:
    """`range(max(1, int(k)))` -> `range(DEFAULT_PROBE_K)` survived,
    because the only k-test passed k=4 — identical to the default.
    A verification that cannot distinguish."""

    @pytest.mark.parametrize("k", [1, 2, 3, 6, 9])
    async def test_exactly_k_samples_are_taken(self, k):
        llm = _Answers("42", "41")
        await solvability_probe(_item().to_bank_row(), llm, k=k)
        assert llm.calls == k

    async def test_k_is_clamped_to_at_least_one(self):
        llm = _Answers("42")
        await solvability_probe(_item().to_bank_row(), llm, k=0)
        assert llm.calls == 1


class TestTheValidatorTimeoutIsEnforced:
    """A model-authored `while True:` would hang the miner, which runs
    under idle orchestration. Removing `timeout=timeout_s` survived 128
    tests."""

    SLOW = ("import sys, time\n"
            "open('answer.txt').read()\n"
            "time.sleep(30)\n"
            "sys.exit(0)\n")

    def test_a_slow_validator_is_cut_off_and_reads_as_UNRUNNABLE(self):
        import time
        t0 = time.time()
        rc = EM._run_validator(self.SLOW, "42", timeout_s=1.0)
        dt = time.time() - t0
        assert rc is None, "a hanging validator did not read as unrunnable"
        assert dt < 12, f"the timeout did not fire ({dt:.1f}s)"

    def test_a_hanging_validator_is_refused_by_the_gate(self):
        ok, why = oracle_is_sound(_item(self.SLOW), timeout_s=1.0)
        assert ok is False


class TestSelfConsistencyCatchesIndeterminateItems:
    """⚠ THE PASS-RATE BAND ALONE COULD NOT DO THIS, AND A REVIEWER
    DEMONSTRATED IT END TO END: a "favourite colour" item was ACCEPTED at
    1/4 while a determinate 6*7 item was REJECTED as trivial at 4/4. The
    LIVE staged item validated `data['move'] != 'c8g4'` — one of 23 legal
    moves — the exact indeterminate shape §3b was written to catch,
    sitting accepted in `system/optim/mined_envs/`.

    The distinguishing signal is not the pass rate, it is whether the
    independent samples AGREE WITH EACH OTHER. A determinate task pulls
    samples toward one answer; an indeterminate one scatters them, and
    the checker's preferred value is merely one of the scattered answers.
    """

    CHESS = ("import sys, json\n"
             "a = open('answer.txt').read().strip()\n"
             "d = json.loads(a)\n"
             "sys.exit(0 if d.get('move') == 'c8g4' else 1)\n")

    async def test_ALL_DIFFERENT_answers_read_as_ARBITRARY(self):
        row = {"challenge": "pick a move", "validation_script": self.CHESS}
        llm = _Answers('{"move":"c8g4"}', '{"move":"d7d5"}',
                       '{"move":"g8f6"}', '{"move":"e7e6"}')
        pr = await solvability_probe(row, llm)
        assert pr.passes == 1, "one sample DID satisfy the checker"
        assert pr.verdict == PROBE_ARBITRARY, (
            "1-of-4 landed in the accepted band on pass rate alone")
        assert "no single right answer" in pr.why

    async def test_a_RECURRING_modal_answer_reads_as_SEPARATES(self):
        """Negative control: a hard-but-determinate item must still be
        accepted, or the gate eats the whole funnel."""
        llm = _Answers("42", "41", "42", "42")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_SEPARATES
        assert "modal answer recurred 3x" in pr.why

    async def test_the_mode_only_needs_to_recur_TWICE(self):
        llm = _Answers("42", "41", "42", "39")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_SEPARATES

    async def test_answers_are_compared_after_normalisation(self):
        """'42' and ' 42 ' are the same answer; a whitespace difference
        must not read as disagreement."""
        llm = _Answers("42", "41", "  42  ", "39")
        pr = await solvability_probe(_item().to_bank_row(), llm)
        assert pr.verdict == PROBE_SEPARATES

    async def test_the_consistency_check_needs_at_least_two_samples(self):
        """At k=1 there is nothing to be consistent WITH, and calling
        that 'all samples differed' would be a verdict on one draw."""
        llm = _Answers("42")
        pr = await solvability_probe(_item().to_bank_row(), llm, k=1)
        assert pr.verdict == PROBE_TRIVIAL      # 1/1 passed
        assert "no single right answer" not in pr.why


class TestBankNamesCannotTraverse:
    """The staging/promotion separation was a PATH CONVENTION, not a
    check: `--name ../../bench/banks/live_bank` wrote straight into the
    live bank directory, arming `pick_next_item` without `--promote`."""

    @pytest.mark.parametrize("bad", [
        "../../bench/banks/live", "/etc/passwd", "a/b", "..",
        "ghost_failures\n", "x" * 65, "", "with space", "dot.name",
    ])
    def test_a_bad_name_is_refused_by_staging(self, bad, tmp_path):
        with pytest.raises(ValueError):
            staging_path(bad, str(tmp_path))

    @pytest.mark.parametrize("bad", ["../../x", "a/b", "ghost\n"])
    def test_write_bank_ALSO_refuses_it(self, bad, tmp_path):
        """Defence in depth — `write_bank` is public and was safe only
        because `_check_name` happened to run upstream."""
        from ghost_agent.eval.banks import write_bank
        with pytest.raises(ValueError):
            write_bank([{"bank": "x", "item_id": "i"}], bad, str(tmp_path))

    @pytest.mark.parametrize("good", ["ghost_failures", "a", "A-1_b",
                                      "x" * 64])
    def test_a_good_name_is_accepted(self, good, tmp_path):
        """Negative control: the check must not refuse legitimate names."""
        assert staging_path(good, str(tmp_path)).name == f"{good}.jsonl"

    def test_a_trailing_newline_cannot_slip_through(self, tmp_path):
        """`$` also matches BEFORE a trailing newline, so the first
        version allowed `"ghost_failures\\n"`. No traversal was possible,
        but a boundary with a known hole is not a boundary."""
        with pytest.raises(ValueError):
            staging_path("ghost_failures\n", str(tmp_path))


# ══════════════════════════════════════════════════════════════════════
# 11. ROUND 2 — the round-1 fixes that were themselves wrong
# ══════════════════════════════════════════════════════════════════════
class TestMinedExamplesSURVIVE_TheRealPipeline:
    """⚠ ROUND 1'S CONSUMER FIX HAD EXACTLY THE DEFECT IT WAS ADDED TO
    REMOVE — the third time this loop shipped unwired.

    `trainset_from_items` stamped `{"final_response": ref, "plan": ""}`,
    and `run_gepa`'s `keyed` filter twelve lines later keeps only
    examples with a TRUTHY field named in `sig.outputs`. `plan` was
    empty and `final_response` is not a signature output, so **100% of
    mined examples were dropped**. Measured on the live corpus: 0 of 1
    survived, the oracle never fired, and nothing said so.
    """

    def _sig(self):
        from ghost_agent.optim.signatures import PLANNING_SIGNATURE
        return PLANNING_SIGNATURE

    def test_the_reference_is_stamped_into_the_SIGNATURES_output_fields(self):
        sig = self._sig()
        ex = trainset_from_items([_item().to_bank_row()], sig.name,
                                 outputs=sorted(sig.outputs))[0]
        assert set(ex.expected_output) == set(sig.outputs)
        assert all(v == "42" for v in ex.expected_output.values())

    def test_mined_examples_SURVIVE_the_keyed_filter(self):
        """The exact filter, driven."""
        sig = self._sig()
        mined = trainset_from_items(
            [_item(item_id=f"m{i}").to_bank_row() for i in range(12)],
            sig.name, outputs=sorted(sig.outputs))
        keyed = [e for e in mined
                 if any((e.expected_output or {}).get(f) for f in sig.outputs)]
        assert len(keyed) == 12, "the keyed filter dropped mined examples"

    def test_an_EMPTY_output_field_would_be_dropped(self):
        """Negative control on the filter itself: it must actually reject
        the shape round 1 produced, or the pin above proves nothing."""
        sig = self._sig()
        broken = trainset_from_items([_item().to_bank_row()], sig.name,
                                     outputs=sorted(sig.outputs))[0]
        broken.expected_output = {f: "" for f in sig.outputs}
        assert not any((broken.expected_output or {}).get(f)
                       for f in sig.outputs)

    def test_the_DEFAULT_stamping_also_carries_a_target(self):
        """Without an explicit `outputs` the fallback must still populate
        `plan` — round 1 left it "" and that alone dropped every mined
        example for planning.decompose."""
        ex = trainset_from_items([_item().to_bank_row()], "s")[0]
        assert ex.expected_output["plan"] == "42"

    def test_a_signature_without_user_request_is_REFUSED(self):
        """`_metric` keys on `gold.user_request` and `_to_dspy_examples`
        copies only DECLARED inputs, so for these signatures the
        challenge is discarded and the golds arrive ALL-EMPTY, scoring
        0.0 in both arms. Adding noise that looks like data is worse
        than adding nothing."""
        from ghost_agent.optim.env_mining import signature_can_use_mined
        from ghost_agent.optim.signatures import (
            PLANNING_SIGNATURE, REFLECTION_SIGNATURE,
            TOOL_SELECTION_SIGNATURE,
        )
        assert signature_can_use_mined(PLANNING_SIGNATURE) is True
        assert signature_can_use_mined(TOOL_SELECTION_SIGNATURE) is False
        assert signature_can_use_mined(REFLECTION_SIGNATURE) is False

    def test_run_gepa_ABORTS_rather_than_silently_losing_them(self):
        src = Path("scripts/run_gepa.py").read_text()
        assert "mined example(s) " in src and "were dropped by the" in src
        assert "Aborting rather than running" in src

    def test_run_gepa_reports_the_oracle_BEFORE_the_early_returns(self):
        """`_report_oracle_use()` was unreachable on `--no-ab-gate` — the
        'did the oracle fire' guard skipped on exactly the path that
        adopts a prompt UNVERIFIED."""
        src = Path("scripts/run_gepa.py").read_text()
        # the DEFINITION precedes both; compare the first CALL.
        first_call = src.index("    _report_oracle_use()\n")
        assert first_call < src.index("    if args.no_ab_gate:")


class TestStagingPreservesOtherEpochs:
    """⚠ `write_staging` read through `read_staging` (current epoch only)
    and then rewrote the WHOLE file — so the first miner run after an
    epoch bump silently DELETED every earlier row, plus any row with no
    epoch. `MINING_EPOCH` is documented as a thing you bump; bumping it
    must supersede the old corpus, not erase it. Data loss in the fix
    whose entire purpose is accumulation."""

    def test_rows_from_another_epoch_are_PRESERVED_on_disk(self, tmp_path):
        from ghost_agent.optim.env_mining import _read_raw
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        old = dict(_item(item_id="old-1").to_bank_row(), mining_epoch="e0")
        noep = _item(item_id="noep").to_bank_row()
        noep.pop("mining_epoch")
        p.write_text(json.dumps(old) + "\n" + json.dumps(noep) + "\n")
        write_staging([_item(item_id="new-1")], "ghost_failures",
                      str(tmp_path))
        on_disk = {r["item_id"] for r in _read_raw("ghost_failures",
                                                   str(tmp_path))}
        assert on_disk == {"old-1", "noep", "new-1"}, on_disk

    def test_but_they_are_still_not_SERVED(self, tmp_path):
        from ghost_agent.optim.env_mining import _read_raw
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            dict(_item(item_id="old-1").to_bank_row(),
                 mining_epoch="e0")) + "\n")
        write_staging([_item(item_id="new-1")], "ghost_failures",
                      str(tmp_path))
        assert [r["item_id"] for r in
                read_staging("ghost_failures", str(tmp_path))] == ["new-1"]
        assert len(_read_raw("ghost_failures", str(tmp_path))) == 2

    def test_accumulation_across_runs(self, tmp_path):
        for i in range(3):
            write_staging([_item(item_id=f"m{i}")], "ghost_failures",
                          str(tmp_path))
        assert len(read_staging("ghost_failures", str(tmp_path))) == 3

    def test_an_id_less_item_is_refused_not_collapsed(self, tmp_path):
        write_staging([_item(item_id=""), _item(item_id="")],
                      "ghost_failures", str(tmp_path))
        assert read_staging("ghost_failures", str(tmp_path)) == []

    def test_the_write_is_atomic(self, tmp_path):
        """A crash mid-write truncated the corpus this function exists to
        protect."""
        write_staging([_item(item_id="a")], "ghost_failures", str(tmp_path))
        leftovers = list(staging_path("ghost_failures",
                                      str(tmp_path)).parent.glob("*.tmp"))
        assert leftovers == [], leftovers


class TestTheDangerousNondeterminismLegIsGuarded:
    """⚠ Round 1 repeated the REFERENCE run and left the negative
    controls at one run each. Round 2 measured the asymmetry over 200
    trials: a validator coin-flipping on a CORRECT answer was admitted
    4% of the time (down from ~50%), but one coin-flipping on WRONG
    answers was still admitted 26% — UNCHANGED. The second shape
    manufactures false PASSes, which is label noise pointed straight at
    the learning corpus."""

    def test_the_controls_are_scored_REFERENCE_REPEATS_times_each(self,
                                                                  monkeypatch):
        seen = []
        real = EM._run_validator
        monkeypatch.setattr(
            EM, "_run_validator",
            lambda v, a, timeout_s=20.0: (seen.append(a),
                                          real(v, a, timeout_s))[1])
        oracle_is_sound(_item())
        assert seen.count("") == EM._REFERENCE_REPEATS
        assert sum(1 for s in seen
                   if s.startswith("NOT-AN-ANSWER")) == EM._REFERENCE_REPEATS

    def test_a_validator_that_sometimes_ACCEPTS_a_control_is_refused(
            self, monkeypatch):
        """Deterministic drive of the leg that matters: reject the
        control on the first run, accept it on a later one."""
        codes = iter([0] * EM._REFERENCE_REPEATS      # reference: stable
                     + [1, 0])                        # control: flips to 0
        monkeypatch.setattr(EM, "_run_validator",
                            lambda v, a, timeout_s=20.0: next(codes))
        ok, why = oracle_is_sound(_item())
        assert ok is False
        assert "cannot fail" in why


class TestAnswerExtractionPrefersTheANSWER:
    """⚠ BOTH earlier rules were wrong, and round 2 showed the second did
    not fix its own cited example: `"```python\\nprint(6*7)\\n```\\nSo the
    answer is 42."` has ONE fence, so last-fence still returned the
    WORKING. And on the commoner "answer, then code" shape it was
    strictly worse than first-fence. The mistake was picking a POSITION
    at all."""

    def test_working_in_a_fence_prose_answer_after(self):
        assert EM._extract_answer(
            "```python\nprint(6*7)\n```\nSo the answer is 42."
        ).endswith("42.")

    def test_answer_first_code_after(self):
        assert EM._extract_answer(
            "The answer is 42.\n```python\nprint(6*7)\n```"
        ).startswith("The answer is 42")

    def test_a_bare_fenced_value_is_still_taken(self):
        assert EM._extract_answer("Here:\n```\n42\n```") == "42"

    def test_the_LAST_non_code_fence_wins(self):
        assert EM._extract_answer(
            "```python\nx=1\n```\nthen\n```\n42\n```") == "42"

    def test_no_fence_returns_the_whole_reply(self):
        assert EM._extract_answer("42") == "42"

    @pytest.mark.parametrize("code", [
        "import os", "def f():\n    pass", "print(6*7)",
        "for i in range(3):\n    pass", "# a comment",
    ])
    def test_code_shapes_are_recognised(self, code):
        assert EM._looks_like_code(code) is True

    @pytest.mark.parametrize("ans", ["42", "157.50", "e7e6",
                                     '{"move": "e7e6"}', "the answer"])
    def test_answer_shapes_are_NOT_mistaken_for_code(self, ans):
        assert EM._looks_like_code(ans) is False


# ══════════════════════════════════════════════════════════════════════
# 12. §4CV GETS ITS YIELD ROW — the loop that can go dark
# ══════════════════════════════════════════════════════════════════════
class TestTheMinerHasAYieldRow:
    """⚠ §4CV shipped WITHOUT one and a round-1 reviewer said why it
    matters: 'the loop that can actually go dark is the one without a
    yield row'. §4CS is the whole precedent — the macro loop ran for six
    weeks producing nothing consumable while every liveness probe read
    FIRED."""

    def _row(self, home):
        from ghost_agent.core.liveness import yield_all
        return next(r for r in yield_all(home)["rows"]
                    if r["name"] == "mining.failure_envs")

    def test_it_is_registered(self, tmp_path):
        from ghost_agent.core.liveness import YIELD_PROBES
        assert any(p.name == "mining.failure_envs" for p in YIELD_PROBES)

    def test_an_empty_staging_reads_GATED_not_a_gap(self, tmp_path):
        """Operator-triggered by design, so "nothing staged" is the
        expected state of a correctly-configured box — a permanent ⚠
        would be noise.

        ⚠ AND IT MUST NOT CLAIM "NEVER RUN". The state is derived from
        the staging FILE's absence, and `mine_failure_envs.py` returns
        before writing when a run accepts nothing — so a real run that
        mined 12 candidates and refused all 12 produces exactly this.
        Asserting "never run" about a loop that ran and refused
        everything is the opposite finding."""
        r = self._row(tmp_path)
        assert r["status"] == "gated"
        assert "perator-triggered" in r["note"]
        assert "INDISTINGUISHABLE" in r["note"]
        assert "a run accepted nothing" in r["note"]
        assert "has never been run on this box" not in r["note"]

    def test_invoked_is_UNMEASURED_because_nothing_records_consumption(
            self, tmp_path):
        """⚠ ROUND 3: the first version set `invoked` to "staged items a
        GEPA metric COULD train on" — a property of the ROWS, i.e.
        production wearing a consumption label. That is the exact defect
        §4CS item B's round 1 built into `evolve.candidates`, rebuilt in
        the probe whose docstring cites it. Nothing durable records a
        GEPA run touching this bank: `run_gepa` PRINTS its oracle counts
        and writes nothing. UNMEASURED is the honest state, and its
        remedy (wire a counter) differs from BARREN's (find a
        consumer)."""
        good = _item(item_id="txt").to_bank_row()
        art = _item(item_id="art").to_bank_row()
        art["graded_on"] = GRADED_ARTIFACT
        write_staging([], "ghost_failures", str(tmp_path))
        p = staging_path("ghost_failures", str(tmp_path))
        p.write_text(json.dumps(good) + "\n" + json.dumps(art) + "\n")
        r = self._row(tmp_path)
        assert r["minted"] == 2
        assert r["invoked"] is None, (
            "nothing records a GEPA run consuming the bank, so any number "
            "here is a claim about the rows, not about consumption")
        assert r["status"] == "unmeasured"
        # the trainable count is a FACT ABOUT THE STORE and belongs in
        # the note, where it cannot be read as consumption.
        assert "1/2 staged item(s) are text-graded" in r["note"]

    def test_staged_but_unpromoted_is_a_DERIVED_zero(self, tmp_path):
        """Staging is deliberately not arming, so 'nobody uses these' is
        arithmetic, not neglect."""
        write_staging([_item()], "ghost_failures", str(tmp_path))
        r = self._row(tmp_path)
        assert r["activated"] == 0
        assert "deliberately not arming" in r["derived_zero"]

    def test_an_UNCONFINED_item_is_called_out(self, tmp_path):
        row = _item().to_bank_row()
        row["verified_confined"] = False
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row) + "\n")
        assert "NOT kernel-sandboxed" in self._row(tmp_path)["note"]

    def test_superseded_epoch_rows_are_reported_not_hidden(self, tmp_path):
        cur = _item(item_id="new").to_bank_row()
        old = dict(_item(item_id="old").to_bank_row(), mining_epoch="e0")
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(old) + "\n" + json.dumps(cur) + "\n")
        r = self._row(tmp_path)
        assert r["minted"] == 1
        assert "superseded mining epoch" in r["note"]

    def test_promotion_shows_up_as_ACTIVATED(self, tmp_path):
        write_staging([_item()], "ghost_failures", str(tmp_path))
        promote_to_bank("ghost_failures", str(tmp_path))
        assert self._row(tmp_path)["activated"] == 1


# ══════════════════════════════════════════════════════════════════════
# 13. ROUND 3 — the round-2 fixes that were themselves wrong
# ══════════════════════════════════════════════════════════════════════
class TestAWrapperFailureIsNotAValidatorVerdict:
    """⚠ Round 2 made `_confined_cmd` the choke point for every caller,
    which widened this: `sandbox-exec`'s OWN exit codes were returned as
    the validator's. Executed with a missing policy file, rc=65 on a
    CORRECT answer — so `oracle_score` returned 0.0 ("the candidate was
    WRONG"), the thing its own docstring forbids, and in the probe one
    such hiccup flips a TRIVIAL item (4/4) to SEPARATES (3/4): an
    infrastructure failure moving an item into the ACCEPTED band."""

    class _P:
        def __init__(self, rc, err=""):
            self.returncode, self.stderr = rc, err

    def test_a_sandbox_diagnostic_reads_as_UNRUNNABLE(self):
        assert EM._wrapper_failed(
            self._P(65, "sandbox-exec: /x.sb: No such file or directory")
        ) is True

    def test_a_plain_nonzero_exit_is_the_VALIDATORS_verdict(self):
        assert EM._wrapper_failed(self._P(1, "")) is False
        assert EM._wrapper_failed(
            self._P(1, "Expected 42, got 43")) is False

    def test_a_clean_exit_is_never_second_guessed(self):
        assert EM._wrapper_failed(
            self._P(0, "sandbox-exec: noise")) is False

    def test_a_validator_cannot_FORGE_the_diagnostic_mid_line(self):
        """Anchored to the start of a line, so printing the string in
        prose does not let a validator claim its own failure was the
        sandbox's."""
        assert EM._wrapper_failed(
            self._P(1, "note: the sandbox-exec: wrapper is fine")) is False

    def test_the_end_to_end_effect(self, monkeypatch):
        """The reason this matters: unrunnable must not score 0.0."""
        monkeypatch.setattr(EM, "_run_validator",
                            lambda v, a, timeout_s=20.0: None)
        assert oracle_score(_item().to_bank_row(), "42") is None


class TestABucketThatDISCRIMINATESVetoesRetirement:
    """⚠ Round 3: `_live_bucket_can_still_qualify` read only
    `fail_n`/`fail_hits` and never a bucket's own `ok_*`, so it could not
    detect bucket-level discrimination at all. A Simpson-reversed gate —
    every bucket +spread, pooled negative — was RETIRED as 'nothing
    owed' while every stratum discriminated correctly. That is the exact
    scenario `_anti_predictive`'s docstring calls disqualifying, applied
    to the pooled figure and not to the strata."""

    def _fn(self):
        from ghost_agent.core.liveness import _live_bucket_can_still_qualify
        return _live_bucket_can_still_qualify

    def test_a_bucket_UNDER_the_bar_but_DISCRIMINATING_is_alive(self):
        # precision 0.30 (under 0.60) but ok_fail_rate 0.05 -> +0.25
        assert self._fn()({"hard": {"fail_n": 40, "fail_hits": 12,
                                    "ok_n": 200, "ok_hits": 10}}) is True

    def test_a_bucket_that_does_NOT_discriminate_stays_dead(self):
        assert self._fn()({"dead": {"fail_n": 40, "fail_hits": 2,
                                    "ok_n": 200, "ok_hits": 40}}) is False

    def test_a_bucket_with_no_ok_arm_falls_back_to_the_bar(self):
        assert self._fn()({"b": {"fail_n": 40, "fail_hits": 2}}) is False
        assert self._fn()({"b": {"fail_n": 40, "fail_hits": 34}}) is True

    def test_a_THIN_bucket_cannot_veto(self, ):
        """`_evaluate_bucket` refuses a bucket with fewer than
        `min_bucket_n` RESOLVED rows outright, so one that can never
        enable must not block RETIRED forever."""
        b = {"b": {"n": 12, "fail_n": 11, "fail_hits": 8}}
        assert self._fn()(b, min_fail_n=10, min_bucket_n=30) is False
        assert self._fn()(b, min_fail_n=10, min_bucket_n=0) is True

    def test_an_UNREADABLE_fail_hits_is_not_counted_assessable(self):
        """It was counted assessable and then silently skipped by the
        qualify check, so the note claimed a check it never ran."""
        from ghost_agent.core.liveness import _assessable_buckets
        assert _assessable_buckets(
            {"b": {"fail_n": 40, "fail_hits": None}}, 10) == 0
        assert _assessable_buckets(
            {"b": {"fail_n": 40, "fail_hits": 2}}, 10) == 1


class TestCodeDetectionAfterRound3:
    @pytest.mark.parametrize("code", [
        "a = 6\nb = 7\na * b", "sys.exit(0)", "console.log(6*7)",
        "let a = 6\nlet b = 7", "return 42", "def f():\n    pass",
        "import os", "// a comment",
    ])
    def test_working_shapes_round_3_MISSED_are_caught(self, code):
        assert EM._looks_like_code(code) is True, code

    @pytest.mark.parametrize("ans", [
        "SELECT name FROM users;", "y = sin(x)",
        "Search for John in the directory", "42", "157.50",
        '{"move": "e7e6"}', "The answer is 42.", "e7e6",
    ])
    def test_answers_round_3_MISCLASSIFIED_are_not_code(self, ans):
        assert EM._looks_like_code(ans) is False, ans

    def test_the_docstrings_cited_example_without_print(self):
        """Round 2's fix handled its own example only because it
        contained `print(`."""
        assert EM._extract_answer(
            "```python\na = 6\nb = 7\na * b\n```\n\nSo the answer is 42."
        ).endswith("42.")


class TestStagingIsHonestAboutWhatItDrops:
    def test_it_WARNS_when_rewriting_would_drop_rows(self, tmp_path, caplog):
        """A rewrite that silently discards id-less rows and unparseable
        lines is data loss with no trace, in the function whose purpose
        is preservation."""
        import logging
        p = staging_path("ghost_failures", str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        noid = _item().to_bank_row()
        noid["item_id"] = ""
        p.write_text(json.dumps(_item(item_id="keep").to_bank_row()) + "\n"
                     + json.dumps(noid) + "\n" + "NOT JSON\n")
        with caplog.at_level(logging.WARNING, logger="GhostOptim"):
            write_staging([_item(item_id="new")], "ghost_failures",
                          str(tmp_path))
        assert any("unusable row" in r.message for r in caplog.records), \
            [r.message for r in caplog.records]

    def test_it_is_SILENT_when_nothing_is_dropped(self, tmp_path, caplog):
        import logging
        write_staging([_item(item_id="a")], "ghost_failures", str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="GhostOptim"):
            write_staging([_item(item_id="b")], "ghost_failures",
                          str(tmp_path))
        assert not [r for r in caplog.records if "unusable row" in r.message]

    def test_the_temp_file_is_per_process(self, tmp_path):
        """One fixed `.tmp` shared by every writer makes the 'atomic'
        guarantee hold only for a single process — two concurrent
        writers tore each other's temp file and one trial lost 400
        items."""
        import os
        import ghost_agent.optim.env_mining as _EM
        seen = {}
        real = _EM.staging_path

        def _spy(name, home=None):
            out = real(name, home)
            seen["p"] = out
            return out
        _EM.staging_path = _spy
        try:
            write_staging([_item()], "ghost_failures", str(tmp_path))
        finally:
            _EM.staging_path = real
        assert str(os.getpid()) in \
            f"{seen['p'].name}.{os.getpid()}.tmp"
        assert list(seen["p"].parent.glob("*.tmp")) == []


class TestWrapperFailureIntegration:
    """⚠ `_wrapper_failed` was tested in ISOLATION and a mutation showed
    the integration unpinned: deleting the branch in `_run_validator`
    left 233 tests green. A helper nobody's caller is tested against is
    a helper, not a guard."""

    def test_run_validator_returns_NONE_on_a_wrapper_failure(self,
                                                             monkeypatch):
        import ghost_agent.optim.env_mining as _EM
        monkeypatch.setattr(
            _EM, "_confined_cmd",
            lambda cmd, wd: (["/bin/sh", "-c",
                              "echo 'sandbox-exec: /x.sb: no such file' >&2; "
                              "exit 65"], True, ""))
        rc = _EM._run_validator(GOOD, "42")
        assert rc is None, (
            f"a sandbox wrapper failure was returned as the validator's "
            f"verdict (rc={rc}) — a correct answer scored as WRONG")

    def test_a_REAL_validator_failure_still_returns_its_code(self,
                                                             monkeypatch):
        """Negative control: the guard must not swallow genuine
        rejections, or every wrong answer reads as unrunnable."""
        import ghost_agent.optim.env_mining as _EM
        monkeypatch.setattr(
            _EM, "_confined_cmd",
            lambda cmd, wd: (["/bin/sh", "-c",
                              "echo 'Expected 42, got 43' >&2; exit 1"],
                             True, ""))
        assert _EM._run_validator(GOOD, "43") == 1

    def test_it_does_not_fire_when_UNCONFINED(self, monkeypatch):
        """Unconfined there is no wrapper, so a stderr line mentioning
        sandbox-exec is just the validator's own output."""
        import ghost_agent.optim.env_mining as _EM
        monkeypatch.setattr(
            _EM, "_confined_cmd",
            lambda cmd, wd: (["/bin/sh", "-c",
                              "echo 'sandbox-exec: x' >&2; exit 1"],
                             False, "off"))
        assert _EM._run_validator(GOOD, "43") == 1

    def test_the_temp_file_NAME_carries_the_pid(self, tmp_path, monkeypatch):
        """⚠ The earlier pin only checked the tmp glob was empty
        afterwards, which is true of a SHARED name too — it could not
        distinguish the fix from the defect."""
        import os
        from pathlib import Path as _P
        seen = []
        real_replace = _P.replace

        def _spy(self, target):
            seen.append(self.name)
            return real_replace(self, target)
        monkeypatch.setattr(_P, "replace", _spy)
        write_staging([_item()], "ghost_failures", str(tmp_path))
        assert seen, "no atomic replace happened"
        assert str(os.getpid()) in seen[-1], (
            f"the temp file {seen[-1]!r} is shared by every process — two "
            f"concurrent writers tear each other's file")


class TestTheGEPAGateHasASeedArm:
    """⚠ THE GATE RATCHETED AND NOBODY CHECKED WHERE TO. `_live_incumbent()`
    makes every run "new candidate vs PREVIOUS ARTIFACT" — right for
    measuring an improvement, blind to a slow drift away from the
    hand-written instruction the chain started from.

    Measured 2026-08-24 on `planning.decompose`: the 2026-07-29 artifact
    scored 0.071, the 2026-08-07 candidate 0.393 (+0.321 — a real
    improvement, correctly promoted) — and the HAND-WRITTEN SEED, never
    in either comparison, scores 0.496. Every promotion was honest and
    the live artifact still ended up significantly WORSE than the thing
    it replaced (McNemar p=0.0059 over 123 examples)."""

    def _src(self):
        return Path("scripts/run_gepa.py").read_text()

    def test_the_gate_runs_a_SEED_arm(self):
        s = self._src()
        assert "_seed_cmp" in s and "result.baseline_instruction" in s
        assert "the arm the ratchet cannot see" in s

    def test_it_REFUSES_to_promote_a_candidate_that_loses_to_the_seed(self):
        """⚠ TOKEN PIN, KEPT ONLY AS A POINTER. A round-4 reviewer deleted
        this entire guard with all six of these assertions green. The
        REAL pins are executed, in
        `tests/test_gepa_optim_reaudit.py::TestTheSeedArmIsDrivenNotAsserted`,
        which drives `main()` and kills the guard-deletion mutants."""
        s = self._src()
        assert "NOT PROMOTING" in s
        assert "_seed_loses" in s
        # the refusal must carry a NOISE FLOOR, not fire on any negative
        # delta — written as `delta < 0` it threw away a +0.50 candidate
        # over ONE flipped example.
        assert "-args.ab_min_delta" in s
        assert "_seed_p" in s

    def test_the_refusal_has_an_explicit_override(self):
        """A hard refusal with no override becomes the next thing someone
        deletes. `--allow-seed-loss` makes overriding a recorded act."""
        assert "--allow-seed-loss" in self._src()

    def test_the_seed_arm_only_runs_when_the_candidate_would_SHIP(self):
        """A rejected candidate does not need a second N-example pass to
        stay rejected."""
        assert "cmd.candidate_ships" in self._src() or \
            "cmp.candidate_ships and _seed" in self._src()

    def test_the_seed_result_is_recorded_in_the_artifact(self):
        assert '"seed_arm"' in self._src()

    def test_seed_cmp_is_defined_before_promote_staging_can_read_it(self):
        """⚠ The provenance field made `_promote_staging` close over
        `_seed_cmp`, which is computed near the END of `main()` — so the
        `--no-ab-gate` path, which promotes long before that, raised
        NameError. A provenance field that crashes the one path adopting
        a prompt UNVERIFIED is the worst place to put one."""
        import ast as _ast
        tree = _ast.parse(self._src())
        for node in _ast.walk(tree):
            if isinstance(node, _ast.AsyncFunctionDef) and node.name == "main":
                st = [n.lineno for n in _ast.walk(node)
                      if isinstance(n, _ast.Name)
                      and isinstance(n.ctx, _ast.Store) and n.id == "_seed_cmp"]
                ld = [n.lineno for n in _ast.walk(node)
                      if isinstance(n, _ast.Name)
                      and isinstance(n.ctx, _ast.Load) and n.id == "_seed_cmp"]
                assert st and ld and min(st) < min(ld), (st, ld)
                return
        raise AssertionError("main() not found")


class TestTheRecheckRefusesAZeroZeroComparison:
    """⚠ `scripts/recheck_gepa_incumbent.py` printed a confident "the
    incumbent no longer clears its own bar" on its FIRST run, off a
    result where BOTH arms scored 0.0000 on all 31 examples. The cause:
    `_expected_target` was nested inside `run_gepa.main()`, so the runner
    raised AttributeError every time and `ab_eval._run_one`'s broad
    except turned that into `passed=False` for both arms."""

    def _src(self):
        return Path("scripts/recheck_gepa_incumbent.py").read_text()

    def test_it_refuses_when_both_arms_score_zero(self):
        s = self._src()
        assert "NO VERDICT" in s and "BOTH ARMS SCORED ZERO" in s
        assert "instrument failure until proven" in s

    def test_the_metric_is_IMPORTED_not_reimplemented(self):
        s = self._src()
        assert "rg._overlap" in s and "rg._expected_target" in s
        assert "def _overlap" not in s, (
            "a second private notion of 'did this prompt win' is how two "
            "answers to the same question come to disagree")

    def test_the_metric_functions_are_reachable_from_outside_run_gepa(self):
        """The bug was that they were NESTED. Import and call them."""
        import importlib.util
        import sys as _s
        spec = importlib.util.spec_from_file_location(
            "_rg_probe", "scripts/run_gepa.py")
        rg = importlib.util.module_from_spec(spec)
        _s.modules["_rg_probe"] = rg
        spec.loader.exec_module(rg)
        assert callable(getattr(rg, "_overlap", None))
        assert callable(getattr(rg, "_expected_target", None))
        assert rg._overlap("a b c", "a b c") == 1.0
        assert rg._overlap("a b c", "x y z") == 0.0

    def test_expected_target_takes_the_signature_EXPLICITLY(self):
        """It closed over `sig`, which is what made it uncallable."""
        import inspect as _i
        import importlib.util
        import sys as _s
        spec = importlib.util.spec_from_file_location(
            "_rg_probe2", "scripts/run_gepa.py")
        rg = importlib.util.module_from_spec(spec)
        _s.modules["_rg_probe2"] = rg
        spec.loader.exec_module(rg)
        assert "sig" in _i.signature(rg._expected_target).parameters

    def test_it_reports_McNemar_not_just_a_pass_rate_delta(self):
        """A pass-rate delta on 31 paired examples is a direction, not a
        verdict — and the private-tier run measured p=0.289 while the
        full run measured p=0.0059."""
        assert "McNemar" in self._src()
