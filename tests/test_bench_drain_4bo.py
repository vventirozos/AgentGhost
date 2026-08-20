"""§4BO — the operator-armed BENCH DRAIN.

WHY THIS EXISTS. The bench flywheel yields ~14 items/day (measured: 18
rows over 29.9 h) because an item needs a deep-idle window, a self-play
dice miss, and a 45-minute cooldown. Against a 2,291-task bank that is
roughly a year to consume, and every downstream verdict is metered by it.
A drain lifts all three limiters on request.

WHAT IS NOT HERE, AND WHY. A `select="random"` sampling mode was built
alongside this and REMOVED before it shipped. Its premise — "the cursor
starts at index 0 and MBPP/GSM8K are ordered easiest-first, so 18/18 is an
artifact of where the walk started" — was checked against the actual bank
files and refuted: MBPP index 0 is a 2-D DP problem, index 967 is "find
the minimum of two numbers", and GSM8K shows corr(index, length) = +0.04.
The banks are unordered, so the sequential prefix is already a fair sample
and a uniform draw would buy nothing while costing with-replacement
duplicates and a two-stage draw over a population where two of the three
banks are the same 1,319 questions. The pins below therefore cover
THROUGHPUT and SAFETY, which is what the feature actually is.

The pins follow the §4BN rule: assert the thing EQUALS a value the test
recomputes, rather than asserting a property of it.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import asyncio
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ghost_agent.eval import banks


def _mk_items(bank, n):
    return [{"bank": bank, "item_id": f"{bank}-{i}", "cluster": "algo",
             "challenge": f"solve {i}", "setup_script": "# none",
             "validation_script": "import sys; sys.exit(0)"}
            for i in range(n)]


# ──────────────────────────────────────────────────────────────────────
# The bank filter
# ──────────────────────────────────────────────────────────────────────

class TestBankFilter:

    def test_the_filter_actually_restricts_the_walk(self, tmp_path):
        home = str(tmp_path)
        banks.write_bank(_mk_items("aa", 10), "aa", home=home)
        banks.write_bank(_mk_items("bb", 10), "bb", home=home)
        picks = {banks.pick_next_item(home=home, banks=["bb"])["bank"]
                 for _ in range(4)}
        assert picks == {"bb"}

    def test_an_unmatched_filter_returns_NOTHING_not_everything(self,
                                                               tmp_path):
        """A filter that silently falls back to 'all banks' answers a
        scoped question with unscoped data."""
        home = str(tmp_path)
        banks.write_bank(_mk_items("aa", 10), "aa", home=home)
        assert banks.pick_next_item(home=home, banks=["does-not-exist"]) is None

    def test_no_filter_still_walks_every_bank(self, tmp_path):
        home = str(tmp_path)
        banks.write_bank(_mk_items("aa", 5), "aa", home=home)
        banks.write_bank(_mk_items("bb", 5), "bb", home=home)
        seen = {banks.pick_next_item(home=home)["bank"] for _ in range(6)}
        assert seen == {"aa", "bb"}


# ──────────────────────────────────────────────────────────────────────
# Provenance: idle vs drain
# ──────────────────────────────────────────────────────────────────────

class TestTheLedgerRecordsWhichRegimeRanIt:
    """Not a sampling scheme — a machine-conditions one. A drain runs
    back-to-back at a 60s floor; the idle walk runs once per deep-idle
    window after a 45-minute cooldown. Pooled, 200 drained rows swamp the
    ~14/day organic ones and silently redefine the number."""

    def _row(self, home, item_id, passed, source):
        banks.record_result({"bank": "aa", "item_id": item_id},
                            passed=passed, status="SUCCESS", attempts=1,
                            home=home, source=source)

    def test_the_regime_rides_the_row(self, tmp_path):
        home = str(tmp_path)
        self._row(home, "aa-1", True, "drain")
        row = json.loads(
            Path(home, "system", "bench", "results.jsonl").read_text()
            .splitlines()[0])
        assert row["source"] == "drain"

    def test_the_default_is_idle(self, tmp_path):
        home = str(tmp_path)
        banks.record_result({"bank": "aa", "item_id": "x"}, passed=True,
                            status="SUCCESS", home=home)
        row = json.loads(
            Path(home, "system", "bench", "results.jsonl").read_text()
            .splitlines()[0])
        assert row["source"] == "idle"

    def test_a_legacy_row_with_no_field_reads_as_idle(self, tmp_path):
        """Every row written before 2026-08-15 came from the organic
        cadence by construction — they must not vanish from an idle-scoped
        query, nor pollute a drain-scoped one."""
        home = str(tmp_path)
        p = Path(home, "system", "bench", "results.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"ts": "2026-08-14T00:00:00Z", "bank": "aa",
                                 "item_id": "aa-0", "passed": True,
                                 "status": "SUCCESS"}) + "\n")
        assert banks.stats(home=home, source="idle")["aa"]["runs"] == 1
        assert banks.stats(home=home, source="drain") == {}

    def test_the_two_rates_are_computed_separately(self, tmp_path):
        home = str(tmp_path)
        for i in range(4):
            self._row(home, f"aa-{i}", True, "idle")
        for i in range(4):
            self._row(home, f"aa-{100+i}", i < 2, "drain")
        assert banks.stats(home=home, source="idle")["aa"]["pass_rate"] == 1.0
        assert banks.stats(home=home, source="drain")["aa"]["pass_rate"] == 0.5
        assert banks.stats(home=home)["aa"]["runs"] == 8

    def test_an_unrecognised_regime_is_normalised(self, tmp_path):
        home = str(tmp_path)
        self._row(home, "x", True, "../evil")
        row = json.loads(
            Path(home, "system", "bench", "results.jsonl").read_text()
            .splitlines()[0])
        assert row["source"] == "idle"


class TestTheOperatorFacingReportSplitsThem:
    """`stats()` grew the filter and had NO caller — the exact
    'instrument that never actually runs' class the learning-health
    report exists to prevent, re-created one level down."""

    def _ledger(self, tmp_path):
        home = tmp_path / "home"
        md = home / "system" / "memory"
        md.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            banks.record_result({"bank": "aa", "item_id": f"aa-{i}"},
                                passed=True, status="SUCCESS",
                                home=str(home), source="idle")
        for i in range(4):
            banks.record_result({"bank": "aa", "item_id": f"aa-{9+i}"},
                                passed=(i == 0), status="SUCCESS",
                                home=str(home), source="drain")
        return home, md

    def test_the_rendered_block_breaks_out_both_regimes(self, tmp_path):
        from ghost_agent.core.learning_health import _bench_health_lines
        _home, md = self._ledger(tmp_path)
        text = "\n".join(_bench_health_lines(md))
        assert "idle" in text and "drain" in text, text
        # The pooled line must NOT be the only rate on screen.
        assert "100.0%" in text, f"idle rate missing: {text}"
        assert "25.0%" in text, f"drain rate missing: {text}"

    def test_a_DRAIN_ONLY_bank_is_labelled_as_such(self, tmp_path):
        """R2 MAJOR: the first version printed the split only when BOTH
        regimes had rows, which excluded the most likely case — a
        bank-scoped drain produces a drain-ONLY bank, which then rendered
        as one unlabelled line whose number is 100% operator-armed. That
        is the confusion the split exists to prevent, reintroduced by the
        guard meant to reduce noise."""
        from ghost_agent.core.learning_health import _bench_health_lines
        home = tmp_path / "home"
        md = home / "system" / "memory"
        md.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            banks.record_result({"bank": "solo", "item_id": f"solo-{i}"},
                                passed=(i == 0), status="SUCCESS",
                                home=str(home), source="drain")
        text = "\n".join(_bench_health_lines(md))
        assert "drain" in text, text
        assert "no organic runs" in text, (
            f"a bank whose entire number is operator-armed renders "
            f"unlabelled: {text}")

    def test_an_undrained_box_gets_no_duplicate_line(self, tmp_path):
        """Until someone arms a drain, the pooled line already IS the idle
        line; printing it twice is noise."""
        from ghost_agent.core.learning_health import _bench_health_lines
        home = tmp_path / "home"
        md = home / "system" / "memory"
        md.mkdir(parents=True, exist_ok=True)
        banks.record_result({"bank": "aa", "item_id": "aa-0"}, passed=True,
                            status="SUCCESS", home=str(home), source="idle")
        lines = _bench_health_lines(md)
        assert len([ln for ln in lines if "runs —" in ln]) == 1

    def test_the_json_report_carries_the_split(self, tmp_path):
        from ghost_agent.core.learning_health import collect_learning_health
        _home, md = self._ledger(tmp_path)
        rep = collect_learning_health(str(md))
        assert rep["bench_by_source"]["idle"]["aa"]["runs"] == 3
        assert rep["bench_by_source"]["drain"]["aa"]["runs"] == 4


# ──────────────────────────────────────────────────────────────────────
# The drain, driven through the REAL tick
# ──────────────────────────────────────────────────────────────────────

_ITEM = {"bank": "aa", "item_id": "aa-0", "cluster": "algo",
         "challenge": "do it", "setup_script": "# none",
         "validation_script": "import sys; sys.exit(0)"}


def _agent(idle_seconds=4000, no_bench=False, foreground=0):
    from ghost_agent.core.agent import GhostAgent, GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.no_dream = True
    ctx.args.no_self_play = True
    ctx.args.no_bench = no_bench
    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = 0
    ctx.llm_client.foreground_requests = foreground
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": []}
    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.journal = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=idle_seconds))
    return GhostAgent(ctx)


def _dreamer(passed=True, raises=False, hangs=False):
    d = MagicMock()
    if hangs:
        async def _forever(*a, **k):
            await asyncio.sleep(3600)
        d.synthetic_self_play = AsyncMock(side_effect=_forever)
    elif raises:
        d.synthetic_self_play = AsyncMock(side_effect=RuntimeError("boom"))
    else:
        d.synthetic_self_play = AsyncMock(return_value="ok")
    d.last_bench_result = {"passed": passed, "status": "SUCCESS",
                           "attempts": 1}
    return d


class TestTheDrainBypassesTheIdleEconomy:

    @pytest.mark.asyncio
    async def test_a_drain_runs_when_NOT_deeply_idle(self):
        agent = _agent(idle_seconds=200)         # far below the 3600 gate
        agent._bench_drain_remaining = 3
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)) as pick, \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        pick.assert_called_once()
        assert agent._bench_drain_remaining == 2

    @pytest.mark.asyncio
    async def test_a_drain_ignores_the_45_minute_cooldown(self):
        agent = _agent()
        agent._last_bench_at = datetime.datetime.now()   # just ran
        agent._bench_drain_remaining = 2
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)) as pick, \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        pick.assert_called_once()

    @pytest.mark.asyncio
    async def test_the_drain_passes_its_BANK_filter_through(self):
        agent = _agent()
        agent._bench_drain_remaining = 1
        agent._bench_drain_banks = ["bb"]
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)) as pick, \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert pick.call_args.kwargs["banks"] == ["bb"]

    @pytest.mark.asyncio
    async def test_the_IDLE_path_never_inherits_a_stale_drain_filter(self):
        """No drain armed → the phase must behave exactly as before §4BO.
        A leftover filter leaking into the organic walk would silently
        change what the flywheel has been measuring for days."""
        agent = _agent()
        agent._bench_drain_banks = ["bb"]        # stale from a past drain
        agent._bench_drain_remaining = 0
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)) as pick, \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert pick.call_args.kwargs["banks"] is None

    @pytest.mark.asyncio
    async def test_the_regime_reaches_the_LEDGER(self):
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True) as rec:
            await agent._biological_tick()
        assert rec.call_args.kwargs["source"] == "drain"

    @pytest.mark.parametrize("regime", ["idle", "drain"])
    @pytest.mark.asyncio
    @patch("ghost_agent.sandbox.docker.DockerSandbox")
    @patch("ghost_agent.core.agent.GhostAgent")
    async def test_the_regime_reaches_the_REAL_solve_context(
            self, mock_agent_cls, mock_sandbox_cls, regime, tmp_path,
            monkeypatch, disable_self_play_templates):
        """THE pin that matters, third attempt — and the first two were
        both proxies that a wrong implementation satisfied.

        v1 read the argument back off the MagicMock it had just fed, and
        passed while `dream.py` dropped the key entirely. v2 walked the
        AST for a `bench_meta.get("source")` call ANYWHERE inside the
        value, and passed on
        `("idle" if bench_meta.get("source") else "idle")` — a constant
        with a decorative call — while ALSO failing a behaviour-preserving
        refactor that moved the dict to a local. Three versions, same
        defect class: a structural proxy standing in for a value.

        So drive the REAL `Dreamer.synthetic_self_play` and read the
        regime off the isolated context it actually builds — the object
        `_record_turn_trajectory` writes from. No proxy left: this fails
        for a constant, passes for a refactor, and is indifferent to how
        the dict is spelled.
        """
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.core.dream import Dreamer
        from tests.test_bench_solve_loop_1c import (
            _make_context, _stateful_sandbox, _wire_agent, _ITEM as _RI)

        ctx = _make_context(tmp_path)
        _wire_agent(mock_agent_cls)
        mock_sandbox_cls.return_value = _stateful_sandbox([("Success", 0)])
        dreamer = Dreamer(ctx)
        await dreamer.synthetic_self_play(
            "test-model", injected_challenge=dict(_RI),
            bench_meta={"bank": "mbpp", "item_id": "mbpp-7",
                        "cluster": "algo", "source": regime})

        solve_ctx = mock_agent_cls.call_args_list[0].args[0]
        extras = solve_ctx.trajectory_extra_static
        assert extras["bench_source"] == regime, (
            f"the corpus row says {extras.get('bench_source')!r} for a "
            f"{regime!r} item — every admitted consumer reads THIS dict")

    @pytest.mark.asyncio
    async def test_the_regime_is_HANDED_to_that_consumer(self):
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        d = _dreamer()
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert d.synthetic_self_play.await_args.kwargs[
            "bench_meta"]["source"] == "drain"

    @pytest.mark.asyncio
    async def test_an_organic_run_is_tagged_idle_on_both(self):
        agent = _agent()
        d = _dreamer()
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True) as rec:
            await agent._biological_tick()
        assert rec.call_args.kwargs["source"] == "idle"
        assert d.synthetic_self_play.await_args.kwargs[
            "bench_meta"]["source"] == "idle"


class TestTheDrainStillObeysTheThingsThatMatter:

    @pytest.mark.asyncio
    async def test_no_bench_beats_an_armed_drain(self):
        agent = _agent(no_bench=True)
        agent._bench_drain_remaining = 5
        with patch("ghost_agent.eval.banks.pick_next_item") as pick:
            await agent._biological_tick()
        pick.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_live_request_in_flight_defers_the_drain(self):
        agent = _agent(foreground=1)
        agent._bench_drain_remaining = 5
        with patch("ghost_agent.eval.banks.pick_next_item") as pick:
            await agent._biological_tick()
        pick.assert_not_called()
        assert agent._bench_drain_remaining == 5   # deferred, not consumed

    @pytest.mark.asyncio
    async def test_a_request_ARRIVING_MID_TICK_defers_the_drain(self):
        """R2 review: the test above passes at the tick's pre-existing
        HARD LOCK (which returns before phase 3b is ever reached), so
        deleting the drain's own `foreground_requests` term left the whole
        suite green. The guard that matters covers a request that arrives
        DURING the tick — the phases above burn minutes — and only a
        counter that flips 0→1 mid-tick can see it."""
        agent = _agent(idle_seconds=4000)
        agent._bench_drain_remaining = 3
        agent._bench_drain_banks = ["bb"]      # marks a drain-origin pick
        class _LiveTurnMidTick:
            """Idle when the tick's hard lock samples it; busy by the
            time phase 3b asks."""
            foreground_tasks = 0

            def __init__(self):
                self.reads = 0

            @property
            def foreground_requests(self):
                self.reads += 1
                return 0 if self.reads <= 1 else 1

        fg = _LiveTurnMidTick()
        agent.context.llm_client = fg
        with patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=None) as pick:
            await agent._biological_tick()
        assert fg.reads >= 2, (
            "phase 3b never re-read foreground_requests — a request "
            "arriving mid-tick cannot defer the drain")
        assert agent._bench_drain_remaining == 3
        for call in pick.call_args_list:
            assert call.kwargs.get("banks") is None, \
                "that pick came from the DRAIN, mid user request"

    @pytest.mark.asyncio
    async def test_the_idle_floor_keeps_it_out_of_a_conversation_gap(self):
        agent = _agent(idle_seconds=5)
        agent._bench_drain_remaining = 5
        with patch("ghost_agent.eval.banks.pick_next_item") as pick:
            await agent._biological_tick()
        pick.assert_not_called()
        assert agent._bench_drain_remaining == 5

    @pytest.mark.asyncio
    async def test_the_idle_floor_reads_the_CLOCK_NOW_not_the_tick_start(
            self):
        """R1 MAJOR-1, proven live with a probe. `idle_secs` is sampled
        ONCE at the top of the tick, and the phases above can burn minutes
        of real time. A user turn that completed one second before phase
        3b evaluated was greeted with the stale sample (`idle = 2000s`)
        and a multi-minute solve started on top of it — exactly what the
        floor exists to forbid. Probe output at the time: "real idle at
        the moment phase 3b ran: 0.01s (floor is 60s), bench pick called:
        True".

        Drive the real shape: the activity clock is deeply idle when the
        tick samples it, and the operator speaks before phase 3b runs.
        The first read (top of tick) sees the old value; every later read
        sees the new one. Under the stale-variable version the drain
        fires; only re-reading the clock at the gate skips it."""
        from unittest.mock import PropertyMock
        agent = _agent(idle_seconds=4000)
        agent._bench_drain_remaining = 3
        old = datetime.datetime.now() - datetime.timedelta(seconds=4000)
        reads = {"n": 0}

        def _clock():
            reads["n"] += 1
            # Read 1 is the tick's own `idle_secs` sample; the user turn
            # lands immediately after it.
            return old if reads["n"] <= 1 else datetime.datetime.now()

        agent._bench_drain_banks = ["bb"]   # makes a drain call identifiable
        type(agent.context).last_activity_time = PropertyMock(
            side_effect=lambda: _clock())
        try:
            with patch("ghost_agent.eval.banks.pick_next_item",
                       return_value=None) as pick:
                await agent._biological_tick()
            assert reads["n"] >= 2, (
                "phase 3b never re-read the clock — it is still deciding "
                "on the sample taken at the top of the tick")
            # The organic walk may legitimately run (its own gate uses the
            # tick-start sample by design, bounded by the 45-min cooldown).
            # What must NOT happen is the DRAIN firing: it would carry the
            # bank filter and spend a budget slot.
            assert agent._bench_drain_remaining == 3, (
                "the drain fired one second after the operator spoke — it "
                "decided on the stale tick-start clock")
            for call in pick.call_args_list:
                assert call.kwargs.get("banks") is None, (
                    "that pick came from the DRAIN, not the idle walk")
        finally:
            del type(agent.context).last_activity_time


class TestTheBudgetAlwaysTerminates:

    @pytest.mark.asyncio
    async def test_the_budget_is_spent_even_when_the_solve_RAISES(self):
        agent = _agent()
        agent._bench_drain_remaining = 2
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer(raises=True)), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert agent._bench_drain_remaining == 1

    @pytest.mark.asyncio
    async def test_the_budget_is_spent_when_the_SETUP_raises_above_the_await(
            self):
        """R1 MAJOR-2, proven with a probe. Six statements used to sit
        between the branch and the `try` — a `pretty_log` on a full disk,
        or the `from .dream import Dreamer`. A raise there escaped the
        tick entirely, was swallowed as a generic 'watchdog tick failed',
        and left the budget untouched while the cursor had ALREADY been
        advanced: three ticks, three burned items, no ledger rows, budget
        still 3."""
        agent = _agent()
        agent._bench_drain_remaining = 3
        with patch("ghost_agent.core.dream.Dreamer",
                   side_effect=RuntimeError("import blew up")), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()     # must not raise
        assert agent._bench_drain_remaining == 2, \
            "the budget was not spent, so this item will retry forever"

    @pytest.mark.asyncio
    async def test_the_budget_reaches_zero_and_the_phase_stands_down(self):
        agent = _agent(idle_seconds=200)   # only a drain can run here
        agent._bench_drain_remaining = 1
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)) as pick, \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
            assert agent._bench_drain_remaining == 0
            await agent._biological_tick()      # second tick: nothing left
        assert pick.call_count == 1

    @pytest.mark.asyncio
    async def test_a_REPEATEDLY_topped_up_budget_still_spends(self):
        """R2 MAJOR: the generation guard that fixed the re-arm race
        created a worse bug — every arm bumped the generation, so a
        script topping the budget up on a cadence shorter than a solve
        meant NO item ever decremented. Measured: 12 items solved, budget
        still 2, bank items burning forever. Spending the slot at CLAIM
        time removes the race instead of guarding it."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 2
        d = _dreamer()

        async def _topped_up(*a, **k):
            agent._bench_drain_remaining += 1      # a supervisor re-arms
            return "ok"

        d.synthetic_self_play = AsyncMock(side_effect=_topped_up)
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            for _ in range(3):
                await agent._biological_tick()
        # 3 claims spent, 3 top-ups added: net 2. The failure mode was a
        # budget that never moved at all.
        assert agent._bench_drain_remaining == 2
        assert d.synthetic_self_play.await_count == 3

    @pytest.mark.asyncio
    async def test_a_cancel_and_REARM_does_not_lose_an_item(self):
        """R1 MAJOR-4: the decrement runs in a `finally` minutes after the
        gate. Cancel mid-solve, arm a fresh budget, and the old item's
        decrement would silently eat one of the NEW slots."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 4

        d = _dreamer()

        async def _cancel_and_rearm(*a, **k):
            agent._bench_drain_remaining = 5      # operator re-armed
            return "ok"

        d.synthetic_self_play = AsyncMock(side_effect=_cancel_and_rearm)
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert agent._bench_drain_remaining == 5, \
            "the finished item spent a slot of the budget armed after it"

    @pytest.mark.asyncio
    async def test_a_RAISING_log_write_cannot_kill_the_idle_loop(self):
        """The widened guard's own comment names "a `pretty_log` that
        hits a full disk" as the motivating failure — and the first fix
        then left three of the drain's own log calls outside every guard.
        A raise there escapes the tick, is swallowed as a generic
        "watchdog tick failed", and the drain retries forever."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 3
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True), \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=OSError(28, "No space left on device")):
            await agent._biological_tick()      # must not raise
        assert agent._bench_drain_remaining == 2, \
            "the slot was not spent, so this item retries every tick"

    @pytest.mark.asyncio
    async def test_a_raising_log_still_RUNS_the_item(self):
        """R3 MAJOR-5: the drain's narration line was raw `pretty_log`
        INSIDE the guarded region, so a full disk consumed the cursor
        position and wrote a NO_RESULT row without ever running the item
        — 200 burned items and zero bench work over ~3.3 h. Spending the
        budget is not enough; the ITEM must still run."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 2
        d = _dreamer()
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True) as rec, \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=OSError(28, "No space left on device")):
            await agent._biological_tick()
        assert d.synthetic_self_play.await_count == 1, \
            "the item was consumed but never solved — a log write ate it"
        assert rec.call_args.kwargs["status"] == "SUCCESS"

    @pytest.mark.asyncio
    async def test_the_START_stamp_is_written_before_the_solve(self):
        """R3 MAJOR-4: without it, a wedged multi-hour drain is
        indistinguishable from one deferring on the idle floor."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        seen = {}

        async def _capture(*a, **k):
            seen["at"] = agent._bench_item_started_at
            return "ok"

        d = _dreamer()
        d.synthetic_self_play = AsyncMock(side_effect=_capture)
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert seen.get("at") is not None, \
            "the start stamp is not set while the item is in flight"

    @pytest.mark.asyncio
    async def test_the_START_stamp_is_CLEARED_when_the_item_ends(self):
        """R4 MAJOR-2. Left set, the stamp says "an item started at T and
        this field never says it finished" — which reads IDENTICALLY for
        a drain that completed hours ago and one wedged right now. An
        operator checking health at 03:00 sees a six-hour-old stamp,
        concludes "wedged", restarts, and destroys the in-memory budget —
        over a misread the field exists to prevent. Cleared, non-null
        means exactly "an item is in flight"."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert agent._bench_item_started_at is None, (
            "a finished drain still reports an item in flight — "
            "indistinguishable from a wedge")
        assert agent._bench_last_item_at is not None

    @pytest.mark.asyncio
    async def test_an_EXHAUSTED_drain_clears_its_bank_filter(self):
        """The cancel path clears it for a stated reason — "else health
        shows a filter for a drain that is over" — and that reason
        applies verbatim to a drain that simply finished. Otherwise
        /api/health advertises the filter indefinitely, until restart."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        agent._bench_drain_banks = ["mbpp"]
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert agent._bench_drain_remaining == 0
        assert agent._bench_drain_banks is None

    @pytest.mark.asyncio
    async def test_a_MID_drain_item_still_reports_in_flight(self):
        """The clear must not fire while items remain: a drain with work
        left and an item running is exactly the state the stamp exists
        to show."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 3
        agent._bench_drain_banks = ["mbpp"]
        seen = {}
        d = _dreamer()

        async def _capture(*a, **k):
            seen["at"] = agent._bench_item_started_at
            return "ok"

        d.synthetic_self_play = AsyncMock(side_effect=_capture)
        with patch("ghost_agent.core.dream.Dreamer", return_value=d), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result", return_value=True):
            await agent._biological_tick()
        assert seen["at"] is not None          # in flight, while running
        assert agent._bench_drain_banks == ["mbpp"]   # still 2 to go

    @pytest.mark.asyncio
    async def test_a_raising_log_does_not_strand_an_unpickable_drain(self):
        """Same class on the cancel path: if the CANCELLED line raises,
        the budget must still be zeroed rather than spinning forever."""
        agent = _agent()
        agent._bench_drain_remaining = 5
        agent._bench_drain_banks = ["ghost-bank"]
        with patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=None), \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=OSError(28, "No space left on device")):
            await agent._biological_tick()      # must not raise
        assert agent._bench_drain_remaining == 0

    @pytest.mark.asyncio
    async def test_an_unpickable_drain_CANCELS_loudly(self):
        agent = _agent()
        agent._bench_drain_remaining = 5
        agent._bench_drain_banks = ["ghost-bank"]
        emitted = []
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer()), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=None), \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=lambda *a, **k: emitted.append((a, k))):
            await agent._biological_tick()
        assert agent._bench_drain_remaining == 0, \
            "an unpickable drain stayed armed — it will spin forever"
        assert any(k.get("level") == "WARNING"
                   for a, k in emitted if a and a[0] == "Bench Drain")
        # §4CB R2 B-MIN-2: the cancel must clear the bank filter too — the
        # drained-to-zero path clears it so /api/health stops serving a
        # filter for a drain that is over; this path zeroed the budget but
        # left banks set until the next restart.
        assert agent._bench_drain_banks is None

    @pytest.mark.asyncio
    async def test_a_TRANSIENT_pick_failure_keeps_the_budget(self):
        """R1 MINOR-1: an OSError reading a bank file is not 'you armed
        something impossible'. Discarding 199 queued items over one bad
        read would be the wrong cure."""
        agent = _agent()
        agent._bench_drain_remaining = 5
        with patch("ghost_agent.eval.banks.pick_next_item",
                   side_effect=OSError("disk hiccup")):
            await agent._biological_tick()
        assert agent._bench_drain_remaining == 5

    @pytest.mark.asyncio
    async def test_a_quiet_idle_skip_does_NOT_cancel_an_unarmed_phase(self):
        agent = _agent()
        emitted = []
        with patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=None), \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=lambda *a, **k: emitted.append((a, k))):
            await agent._biological_tick()
        assert not [a for a, k in emitted if a and a[0] == "Bench Drain"]


class TestOneItemCannotWedgeTheWholeIdleLoop:
    """R1 CRITICAL. The biological watchdog is the process's ONLY
    long-lived background task and `synthetic_self_play` has no internal
    timeout. An item parked on the shared inference slot takes journal,
    dream, reflection, skills, PRM/router retrain, calibration,
    autoadvance and self-play down with it until a restart — while
    /api/health still reports `biological_watchdog_alive: true`, because
    the task IS alive, parked inside one await."""

    @pytest.mark.asyncio
    async def test_a_hanging_item_is_cancelled_and_the_tick_returns(self):
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 2
        agent._BENCH_ITEM_TIMEOUT = 0.05
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer(hangs=True)), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True) as rec:
            await asyncio.wait_for(agent._biological_tick(), timeout=10)
        rec.assert_called_once()
        assert agent._bench_drain_remaining == 1, \
            "a wedged item must still spend its budget slot"

    @pytest.mark.asyncio
    async def test_a_wedge_is_recorded_as_INFRA_not_as_a_failed_task(self):
        """A box that wedged is not the agent getting the answer wrong;
        counting it as a loss would quietly deflate the pass rate."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        agent._BENCH_ITEM_TIMEOUT = 0.05
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer(hangs=True)), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True) as rec:
            await asyncio.wait_for(agent._biological_tick(), timeout=10)
        status = rec.call_args.kwargs["status"]
        assert status.startswith("INFRA_ABORT"), status
        # …and INFRA_ABORT is what banks.stats() excludes from the
        # pass-rate denominator, recomputed here rather than asserted.
        assert status.startswith(banks._UNRESOLVED_STATUS_PREFIXES)

    @pytest.mark.asyncio
    async def test_a_wedge_is_never_recorded_as_a_PASS(self):
        """R2 MAJOR: `stats()` tests `passed` BEFORE the unresolved
        prefix, so {passed: True, status: INFRA_ABORT} lands in the pass
        NUMERATOR — and that row is constructible, because
        `last_bench_result` is set before the sandbox-teardown finally, so
        a run that passed and then wedged during cleanup produces exactly
        it. Recompute the consequence rather than trusting the status."""
        agent = _agent(idle_seconds=200)
        agent._bench_drain_remaining = 1
        agent._BENCH_ITEM_TIMEOUT = 0.05
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=_dreamer(passed=True, hangs=True)), \
             patch("ghost_agent.eval.banks.pick_next_item",
                   return_value=dict(_ITEM)), \
             patch("ghost_agent.eval.banks.record_result",
                   return_value=True) as rec:
            await asyncio.wait_for(agent._biological_tick(), timeout=10)
        assert rec.call_args.kwargs["passed"] is False

    def test_a_wedge_row_lands_OUTSIDE_the_pass_rate(self, tmp_path):
        """The property the test above protects, recomputed through the
        real `stats()` rather than asserted about the status string."""
        home = str(tmp_path)
        banks.record_result({"bank": "aa", "item_id": "aa-0"}, passed=False,
                            status="INFRA_ABORT (item exceeded 900s)",
                            home=home, source="drain")
        banks.record_result({"bank": "aa", "item_id": "aa-1"}, passed=True,
                            status="SUCCESS", home=home, source="drain")
        st = banks.stats(home=home)["aa"]
        assert st["unresolved"] == 1 and st["passed"] == 1
        assert st["pass_rate"] == 1.0     # 1/(1+0), the wedge excluded

    def test_the_timeout_is_generous_enough_to_be_a_wedge_guard(self):
        """Measured per-item wall clock on this box is 17.8–54.9 s, and a
        3-attempt item ~3 min. A ceiling near that would truncate honest
        slow work instead of ending wedges."""
        from ghost_agent.core.agent import GhostAgent
        # Measured worst honest item is 179 s (request-start → ledger
        # row). Recompute the ratio rather than asserting a magic floor.
        assert GhostAgent._BENCH_ITEM_TIMEOUT >= 5 * 179


# ──────────────────────────────────────────────────────────────────────
# The endpoint
# ──────────────────────────────────────────────────────────────────────

def _request(body, bio_done=False):
    req = MagicMock()

    async def _json():
        if isinstance(body, Exception):
            raise body
        return body
    req.json = _json
    bio = MagicMock()
    bio.done.return_value = bio_done
    req.app.state.biological_task = bio
    return req


def _api_agent(no_bench=False):
    agent = MagicMock()
    agent.context.args.no_bench = no_bench
    agent._bench_drain_remaining = 0
    agent._bench_drain_gen = 0
    return agent


async def _drain(body, agent, bio_done=False):
    from ghost_agent.api.routes import bench_drain
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        return await bench_drain(_request(body, bio_done=bio_done))


def _payload(resp):
    return json.loads(bytes(resp.body).decode("utf-8"))


class TestTheEndpointArmsTheWatchdog:

    @pytest.mark.asyncio
    async def test_arming_sets_the_budget_the_TICK_reads(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        agent = _api_agent()
        resp = await _drain({"count": 7}, agent)
        assert resp.status_code == 200
        assert agent._bench_drain_remaining == 7

    @pytest.mark.asyncio
    async def test_the_arm_reply_estimates_the_wall_clock(self, tmp_path,
                                                          monkeypatch):
        """200 items is hours of the box's only inference slot; the
        operator should see that before walking away."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        body = _payload(await _drain({"count": 200}, _api_agent()))
        lo, hi = body["estimated_hours"]
        # Recomputed, not eyeballed: 92 s and 240 s per item.
        assert [lo, hi] == [round(200 * 92 / 3600.0, 1),
                            round(200 * 240 / 3600.0, 1)]
        # …and the wedge tail must be stated, not implied.
        from ghost_agent.core.agent import GhostAgent
        # timeout + the 60 s tick gap + the ~60 s teardown tail that
        # `wait_for` waits out after cancelling. Omitting the tail
        # understated the 200-item worst case by ~6%.
        assert body["worst_case_hours"] == round(
            200 * (GhostAgent._BENCH_ITEM_TIMEOUT + 120) / 3600.0, 1)

    @pytest.mark.asyncio
    async def test_count_zero_cancels(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent = _api_agent()
        agent._bench_drain_remaining = 9
        resp = await _drain({"count": 0}, agent)
        assert agent._bench_drain_remaining == 0
        assert _payload(resp)["cancelled"] == 9

    @pytest.mark.asyncio
    async def test_cancel_works_even_with_an_INVALID_other_field(
            self, tmp_path, monkeypatch):
        """R2 m4: validation used to run ahead of the cancel branch, so
        `{"count": 0, "banks": "typo"}` returned 400 and the drain kept
        running. An operator watching the box thrash must never be
        refused a stop because of an unrelated field."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent = _api_agent()
        agent._bench_drain_remaining = 40
        resp = await _drain({"count": 0, "banks": "not-a-list"}, agent)
        assert resp.status_code == 200
        assert agent._bench_drain_remaining == 0


class TestTheArmWarnsAboutCollisions:

    @pytest.mark.asyncio
    async def test_a_live_bench_scoped_arm_is_named_at_ARM_time(
            self, tmp_path, monkeypatch):
        """R3 MAJOR-2: a bench-scoped arm accrues from the population a
        drain floods — ~50 nights of `tts_bon` accrual in one night, and
        nothing yet stratifies on the regime. The arm's own rule gates on
        "no confound annotation", i.e. on a human noticing."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))

        class _Reg:
            def names_for_scope(self, scope):
                return ("tts_bon",)

        import ghost_agent.core.experiments as _ex
        monkeypatch.setattr(_ex, "load_registry", lambda *a, **k: _Reg())
        body = _payload(await _drain({"count": 30}, _api_agent()))
        assert body["bench_scoped_arms"] == ["tts_bon"]
        assert "tts_bon" in body["note"]

    @pytest.mark.asyncio
    async def test_no_warning_when_no_arm_is_live(self, tmp_path,
                                                  monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))

        class _Reg:
            def names_for_scope(self, scope):
                return ()

        import ghost_agent.core.experiments as _ex
        monkeypatch.setattr(_ex, "load_registry", lambda *a, **k: _Reg())
        body = _payload(await _drain({"count": 30}, _api_agent()))
        assert body["bench_scoped_arms"] == []
        assert "stratifies" not in body["note"]


class TestHealthReadbackCannotBreakTheProbe:
    """The health route's standing contract: a partial or mocked agent
    must never 500 it. Both §4BO fields went in un-coerced first and did
    exactly that — ten health tests turned red on
    `Object of type MagicMock is not JSON serializable`."""

    def _health_fields(self, agent):
        from ghost_agent.api.routes import (
            _bench_drain_int, _bench_drain_banks, _bench_last_item_iso)
        return (_bench_drain_int(agent), _bench_drain_banks(agent),
                _bench_last_item_iso(agent))

    def test_a_mocked_agent_yields_JSON_SERIALIZABLE_values(self):
        n, b, ts = self._health_fields(MagicMock())
        json.dumps([n, b, ts])     # the actual failure was here
        # The contract is the TYPE, not the value: a MagicMock satisfies
        # __int__, so the count is some int — what must never happen is a
        # MagicMock reaching the JSON encoder.
        assert isinstance(n, int) and b is None and ts is None

    def test_a_missing_agent_is_survivable(self):
        assert self._health_fields(None) == (0, None, None)

    def test_the_START_stamp_separates_deferring_from_wedged(self):
        """R3 MAJOR-4: `bench_last_item_at` is written in the `finally`,
        so it moves when an item ENDS — and "deferring on the idle floor"
        and "parked inside a wedged solve" BOTH look like a stamp that
        stopped. Only a START stamp tells them apart."""
        from ghost_agent.api.routes import (_bench_started_iso,
                                            _bench_last_item_iso)
        # Since R4 the start stamp is CLEARED when an item ends, so it
        # is non-null iff an item is in flight. Both states below have
        # ALREADY completed items — which is what defeated the earlier
        # version of this test, where both arms had ended=None.
        deferring = MagicMock()
        deferring._bench_item_started_at = None            # nothing running
        deferring._bench_last_item_at = datetime.datetime(2026, 8, 16, 6, 0)
        wedged = MagicMock()
        wedged._bench_item_started_at = datetime.datetime(2026, 8, 16, 6, 5)
        wedged._bench_last_item_at = datetime.datetime(2026, 8, 16, 6, 0)
        assert _bench_started_iso(deferring) is None
        assert _bench_started_iso(wedged) == "2026-08-16T06:05:00"
        # The END stamp alone cannot separate them — identical in both.
        assert (_bench_last_item_iso(deferring)
                == _bench_last_item_iso(wedged) == "2026-08-16T06:00:00")

    def test_real_values_survive_intact(self):
        agent = MagicMock()
        agent._bench_drain_remaining = 12
        agent._bench_drain_banks = ["mbpp", "gsm8k"]
        agent._bench_last_item_at = datetime.datetime(2026, 8, 15, 9, 30)
        n, b, ts = self._health_fields(agent)
        assert (n, b) == (12, ["mbpp", "gsm8k"])
        assert ts == "2026-08-15T09:30:00"
        json.dumps([n, b, ts])


class TestTheEndpointCannotFailAnArmItAlreadyMade:

    @pytest.mark.asyncio
    async def test_a_raising_log_does_not_500_an_armed_drain(
            self, tmp_path, monkeypatch):
        """R3 MAJOR-3: the arm handler's narration ran AFTER the budget
        was written and was unwrapped, so a log write that raised turned
        a SUCCESSFUL arm into an HTTP 500 — the operator reads that as
        "nothing happened" and walks away while the box drains 200
        items behind it."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        agent = _api_agent()
        with patch("ghost_agent.api.routes.pretty_log",
                   side_effect=OSError(28, "No space left on device")):
            resp = await _drain({"count": 200}, agent)
        assert resp.status_code == 200, \
            "the operator is told the arm failed while it is armed"
        assert agent._bench_drain_remaining == 200

    @pytest.mark.asyncio
    async def test_cancel_clears_the_bank_filter(self, tmp_path,
                                                 monkeypatch):
        """Else /api/health reports a filter for a drain that is over."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent = _api_agent()
        agent._bench_drain_remaining = 20
        agent._bench_drain_banks = ["mbpp"]
        await _drain({"count": 0}, agent)
        assert agent._bench_drain_banks is None


class TestTheEndpointRefusesBadInput:

    @pytest.mark.asyncio
    async def test_an_absurd_count_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent = _api_agent()
        resp = await _drain({"count": 100000}, agent)
        assert resp.status_code == 400
        assert agent._bench_drain_remaining == 0

    @pytest.mark.asyncio
    async def test_an_infinite_count_is_a_400_not_a_500(self, tmp_path,
                                                        monkeypatch):
        """`1e999` decodes to float('inf'); int(inf) raises OverflowError,
        which is not a subclass of TypeError or ValueError."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        resp = await _drain({"count": float("inf")}, _api_agent())
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_a_missing_count_is_NOT_a_silent_cancel(self, tmp_path,
                                                          monkeypatch):
        """A typo'd arm request and 'stop the running drain' must not be
        the same 200 response."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent = _api_agent()
        agent._bench_drain_remaining = 9
        resp = await _drain({"banks": ["aa"]}, agent)
        assert resp.status_code == 400
        assert agent._bench_drain_remaining == 9    # untouched
        # The message is the whole value of the dedicated guard — the type
        # check below it would also 400, but on "count must be a number",
        # which does not tell an operator how to cancel.
        assert "required" in _payload(resp)["error"], _payload(resp)
        assert "0 to cancel" in _payload(resp)["error"]

    @pytest.mark.asyncio
    async def test_a_bare_string_banks_is_refused_not_read_as_all(
            self, tmp_path, monkeypatch):
        """What a hand-typed curl produces. Silently meaning 'all banks'
        would spend hours answering a different question."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        agent = _api_agent()
        resp = await _drain({"count": 5, "banks": "aa"}, agent)
        assert resp.status_code == 400
        assert agent._bench_drain_remaining == 0

    @pytest.mark.asyncio
    async def test_an_empty_banks_list_is_refused(self, tmp_path,
                                                  monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        resp = await _drain({"count": 5, "banks": []}, _api_agent())
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_a_boolean_count_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        resp = await _drain({"count": True}, _api_agent())
        assert resp.status_code == 400


class TestTheEndpointRefusesToArmSomethingInert:
    """Armed-but-permanently-inert is this project's most-repeated defect.
    Answer it at ARM time, while the operator is still looking at the
    terminal, instead of leaving a WARNING in a log."""

    @pytest.mark.asyncio
    async def test_no_banks_on_disk_is_refused_with_the_fix(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent = _api_agent()
        resp = await _drain({"count": 5}, agent)
        assert resp.status_code == 409
        assert "import_bench_banks" in _payload(resp)["error"]
        assert agent._bench_drain_remaining == 0

    @pytest.mark.asyncio
    async def test_an_unknown_bank_is_refused_and_names_the_real_ones(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        agent = _api_agent()
        resp = await _drain({"count": 5, "banks": ["nope"]}, agent)
        assert resp.status_code == 409
        assert "aa" in _payload(resp)["error"]
        assert agent._bench_drain_remaining == 0

    @pytest.mark.asyncio
    async def test_no_bench_is_refused_at_the_door(self, tmp_path,
                                                   monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        agent = _api_agent(no_bench=True)
        resp = await _drain({"count": 5}, agent)
        assert resp.status_code == 409
        assert agent._bench_drain_remaining == 0

    @pytest.mark.asyncio
    async def test_a_DEAD_watchdog_is_refused(self, tmp_path, monkeypatch):
        """The watchdog is the only thing that spends the budget. Arming
        while it is dead returns a cheerful 200 for work that can never
        run — the precise outcome this block exists to rule out."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        banks.write_bank(_mk_items("aa", 3), "aa", home=str(tmp_path))
        agent = _api_agent()
        resp = await _drain({"count": 5}, agent, bio_done=True)
        assert resp.status_code == 409
        assert _payload(resp)["code"] == "no_consumer"
        assert agent._bench_drain_remaining == 0


# ──────────────────────────────────────────────────────────────────────
# The FD leak — found by running a real 20-item drain, not by review
# ──────────────────────────────────────────────────────────────────────

class TestTheSolveDoesNotLeakDockerClients:
    """LIVE FINDING (2026-08-15). A 20-item drain ran 15 items and then
    the last 5 died on `OSError: [Errno 24] Too many open files`: they
    failed to solve AND failed to append their ledger rows, each leaving
    a container stuck in `Created`. Measured on the box: 228 unix FDs to
    /var/run/docker.sock after ~20 items.

    Cause: `DockerSandbox.__init__` opens a client via `from_env()`,
    dream.py builds a FRESH sandbox per solve, and nothing ever closed
    the connection pool. Four review rounds did not find this; one real
    drain did — the organic cadence (one item per 45-72 min, with
    restarts between) never accumulates enough to show it.
    """

    def _sandbox(self):
        from ghost_agent.sandbox import docker as _d
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container = None
        sb.container_name = "ghost-agent-sandbox-deadbeef"
        sb.NotFound = _d.docker.errors.NotFound if hasattr(
            _d, "docker") else Exception

        class _NF(Exception):
            pass
        sb.NotFound = _NF
        sb.APIError = Exception
        sb.client.containers.get.side_effect = _NF()
        return sb

    def test_discarding_a_sandbox_CLOSES_its_docker_client(self):
        sb = self._sandbox()
        sb.close(remove=True)
        assert sb.client.close.called, (
            "the connection pool survives the sandbox — every solve then "
            "leaks its docker sockets until EMFILE")

    def test_a_KEPT_sandbox_stays_usable(self):
        """remove=False means "stop but keep it warm for a fast resume".
        Closing the client there would break the long-lived sandbox the
        agent reuses for real turns."""
        sb = self._sandbox()
        sb.close(remove=False)
        assert not sb.client.close.called

    def test_close_NEVER_raises(self):
        """Its docstring promises this ("expected to run from signal
        handlers / shutdown hooks where exceptions are disruptive") and
        the promise was false — an exception in the container step
        propagated straight through. Both callers are teardown paths, so
        a throwing close() is how one bad container takes the idle loop
        with it."""
        from ghost_agent.sandbox import docker as _d
        # A RuntimeError from `containers.get` is caught INSIDE
        # `_close_container`, so it proves nothing about the outer
        # handler — the first version of this test made that mistake and
        # a mutation removing the outer `except` stayed green. Drive a
        # failure that genuinely ESCAPES: `except self.NotFound` raises
        # AttributeError while handling an exception when the attribute
        # is missing, and that propagates out of `_close_container`.
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container = None
        sb.container_name = "ghost-agent-sandbox-nonf"
        sb.APIError = Exception
        sb.client.containers.get.side_effect = RuntimeError("docker is gone")
        # deliberately NO `sb.NotFound`
        sb.close(remove=True)          # must not raise
        assert sb.client.close.called, \
            "the client was not closed on the failure path"

    def test_close_survives_a_half_built_instance(self):
        """`self.container` was read outside any guard, so an instance
        whose __init__ was bypassed raised AttributeError — inconsistent
        with this file's own getattr-default convention."""
        from ghost_agent.sandbox import docker as _d
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container_name = "ghost-agent-sandbox-halfbuilt"

        class _NF(Exception):
            pass
        sb.NotFound = _NF
        sb.APIError = Exception
        sb.client.containers.get.side_effect = _NF()
        sb.close(remove=True)          # no `container` attribute at all
        assert sb.client.close.called
        # THE discriminating assertion: with the getattr default the code
        # proceeds to the by-name lookup (which is what removes an
        # orphan); reading `self.container` directly raises before ever
        # getting there, and the outer handler would swallow it — leaving
        # this test green while orphan cleanup silently stopped.
        sb.client.containers.get.assert_called_once_with(sb.container_name)

    def test_teardown_removes_a_container_it_never_BOUND(self):
        """The old per-solve teardown required `.container` to be truthy,
        so a sandbox that failed midway through provisioning — exactly
        what EMFILE causes — orphaned its container forever. Five such
        containers were left on the live box, one per failed item."""
        sb = self._sandbox()
        orphan = MagicMock()
        orphan.status = "created"
        sb.client.containers.get.side_effect = None
        sb.client.containers.get.return_value = orphan
        sb.close(remove=True)
        sb.client.containers.get.assert_called_with(sb.container_name)
        assert orphan.remove.called

    @pytest.mark.asyncio
    @patch("ghost_agent.sandbox.docker.DockerSandbox")
    @patch("ghost_agent.core.agent.GhostAgent")
    async def test_the_REAL_solve_loop_closes_its_sandbox(
            self, mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
            disable_self_play_templates):
        """Drive the real solve loop and assert the teardown HAPPENED.

        The first version of this pin walked the AST for a `.close()`
        call — and failed against the correct implementation, because the
        call goes through `asyncio.to_thread(sandbox.close, remove=True)`
        where `close` is an attribute reference, not a Call node. Same
        lesson as the provenance pin: drive it, do not pattern-match it.
        """
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.core.dream import Dreamer
        from tests.test_bench_solve_loop_1c import (
            _make_context, _stateful_sandbox, _wire_agent, _ITEM as _RI)

        ctx = _make_context(tmp_path)
        _wire_agent(mock_agent_cls)
        mock_sandbox_cls.return_value = _stateful_sandbox([("Success", 0)])
        await Dreamer(ctx).synthetic_self_play(
            "test-model", injected_challenge=dict(_RI),
            bench_meta={"bank": "mbpp", "item_id": "mbpp-7",
                        "cluster": "algo", "source": "drain"})

        sb = mock_sandbox_cls.return_value
        assert sb.close.called, (
            "the solve finished without closing its sandbox — the docker "
            "client outlives it and every solve leaks its sockets")
        assert sb.close.call_args.kwargs.get("remove") is True, (
            "close() without remove=True leaves the container behind")


# ──────────────────────────────────────────────────────────────────────
# Boot sweep of containers orphaned by a kill mid-solve
# ──────────────────────────────────────────────────────────────────────

class TestTheBootSweepOnlyRemovesProvableOrphans:
    """A `finally` cannot run through SIGKILL, so a daemon killed
    mid-solve leaves a container running against a TemporaryDirectory
    Python already deleted. Two were found live (3 days and 43 min old).

    This code DELETES containers, so every test here is about what it
    must NOT touch."""

    def _mgr(self, tmp_path, containers):
        from ghost_agent.sandbox import docker as _d
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container = None
        sb.container_name = "ghost-agent-sandbox-self"
        sb.client.containers.list.return_value = containers
        return sb

    def _c(self, name, sources, status="running", age_s=99999):
        import datetime as _dt
        c = MagicMock()
        c.name = name
        c.status = status
        created = (_dt.datetime.now(_dt.timezone.utc)
                   - _dt.timedelta(seconds=age_s))
        c.attrs = {"Mounts": [{"Source": s} for s in sources],
                   "Created": created.isoformat().replace("+00:00", "Z")}
        return c

    def _tmpws(self, name="tmpabcdefgh"):
        """A path shaped like `tempfile.TemporaryDirectory()` — the ONLY
        shape the sweep treats as a per-solve workspace."""
        import tempfile
        return os.path.join(tempfile.gettempdir(), name)

    def test_it_removes_an_old_PER_SOLVE_sandbox(self, tmp_path):
        dead = self._c("ghost-agent-sandbox-dead", [self._tmpws()])
        sb = self._mgr(tmp_path, [dead])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-dead"]
        assert dead.remove.called

    def test_a_SURVIVING_workspace_does_not_save_an_orphan(self, tmp_path):
        """THE correction that made this feature real. The first version
        swept "containers whose mount no longer exists", and a dry run
        against the live box refuted it: SIGKILL is exactly what orphans
        a container AND exactly what prevents
        `TemporaryDirectory.cleanup()` from running, so the workspace
        survives. That criterion spared every orphan it targeted."""
        ws = self._tmpws("tmpstillhere")
        os.makedirs(ws, exist_ok=True)
        try:
            dead = self._c("ghost-agent-sandbox-dead", [ws])
            sb = self._mgr(tmp_path, [dead])
            assert sb.sweep_orphaned_containers() == \
                ["ghost-agent-sandbox-dead"]
        finally:
            os.rmdir(ws)

    def test_it_NEVER_removes_the_agents_OWN_sandbox(self, tmp_path):
        """The long-lived sandbox mounts $GHOST_HOME/sandbox — a stable
        path outside the temp root, so it can never match."""
        live = tmp_path / "sandbox"
        live.mkdir()
        alive = self._c("ghost-agent-sandbox-live", [str(live)])
        sb = self._mgr(tmp_path, [alive])
        assert sb.sweep_orphaned_containers() == []
        assert not alive.remove.called

    def test_it_NEVER_removes_a_DETACHED_JOB_container(self, tmp_path):
        """`ghostjobs-*` workspaces belong to commands deliberately
        detached to survive agent restarts (§4AX). One was live on the
        box while this was written; sweeping it would destroy running
        work. It sits under the temp root, so only the `tmp` prefix test
        separates it."""
        import tempfile
        job = os.path.join(tempfile.gettempdir(), "ghostjobs-bq1yp821")
        c = self._c("ghost-agent-sandbox-job", [job])
        sb = self._mgr(tmp_path, [c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_tmp_NAMED_path_outside_the_temp_root_is_spared(self,
                                                              tmp_path):
        """The `tmp` prefix alone is not the test — containment in the
        system temp root is. A project directory that happens to be
        called `tmpdata` must never be read as a throwaway workspace."""
        # NOT pytest's tmp_path — on macOS that lives INSIDE the system
        # temp root, so it would legitimately qualify. A path genuinely
        # outside it is the case under test.
        decoy = os.path.join(os.path.expanduser("~"), "projects", "tmpdata")
        c = self._c("ghost-agent-sandbox-decoy", [decoy])
        sb = self._mgr(tmp_path, [c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_YOUNG_per_solve_sandbox_is_spared(self, tmp_path):
        """It may belong to a solve genuinely in flight in another
        process. True orphans persist for hours."""
        young = self._c("ghost-agent-sandbox-young", [self._tmpws()],
                        age_s=60)
        sb = self._mgr(tmp_path, [young])
        assert sb.sweep_orphaned_containers() == []

    def test_an_UNDATED_container_is_spared(self, tmp_path):
        c = self._c("ghost-agent-sandbox-nodate", [self._tmpws()])
        c.attrs["Created"] = ""
        sb = self._mgr(tmp_path, [c])
        assert sb.sweep_orphaned_containers() == []

    def test_a_container_with_ANY_non_per_solve_mount_is_spared(self,
                                                                tmp_path):
        mixed = self._c("ghost-agent-sandbox-mixed",
                        [self._tmpws(), str(tmp_path / "project")])
        sb = self._mgr(tmp_path, [mixed])
        assert sb.sweep_orphaned_containers() == []

    def test_a_container_with_NO_mounts_is_spared(self, tmp_path):
        """'Cannot tell' must never read as 'safe to delete'."""
        unknown = self._c("ghost-agent-sandbox-unknown", [])
        sb = self._mgr(tmp_path, [unknown])
        assert sb.sweep_orphaned_containers() == []
        assert not unknown.remove.called

    def test_it_never_removes_ITSELF(self, tmp_path):
        me = self._c("ghost-agent-sandbox-self", [self._tmpws()])
        sb = self._mgr(tmp_path, [me])
        assert sb.sweep_orphaned_containers() == []

    def test_it_ignores_containers_that_are_not_ours(self, tmp_path):
        other = self._c("postgres-prod", [self._tmpws()])
        sb = self._mgr(tmp_path, [other])
        assert sb.sweep_orphaned_containers() == []
        assert not other.remove.called

    def test_the_kill_switch_disables_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_SWEEP", "0")
        dead = self._c("ghost-agent-sandbox-dead", [self._tmpws()])
        sb = self._mgr(tmp_path, [dead])
        assert sb.sweep_orphaned_containers() == []
        assert not dead.remove.called

    def test_one_unremovable_container_does_not_stop_the_sweep(self,
                                                               tmp_path):
        bad = self._c("ghost-agent-sandbox-bad", [self._tmpws("tmpg1")])
        bad.remove.side_effect = RuntimeError("docker said no")
        good = self._c("ghost-agent-sandbox-good", [self._tmpws("tmpg2")])
        sb = self._mgr(tmp_path, [bad, good])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-good"]

    def test_a_listing_failure_is_survivable(self, tmp_path):
        sb = self._mgr(tmp_path, [])
        sb.client.containers.list.side_effect = RuntimeError("no docker")
        assert sb.sweep_orphaned_containers() == []      # must not raise

    def test_the_cap_bounds_one_sweep(self, tmp_path):
        many = [self._c(f"ghost-agent-sandbox-{i}",
                        [self._tmpws(f"tmpcap{i}")]) for i in range(40)]
        sb = self._mgr(tmp_path, many)
        assert len(sb.sweep_orphaned_containers()) == sb._SWEEP_CAP
