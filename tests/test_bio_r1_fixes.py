"""§4CB idle/biological slice R1 (2026-08-20) — pins for the round's fixes.

Lens A: the /advance route's missing foreground bracket, the phase-2/3
preambles outside their trys (cooldown-starvation identity), the unmarked
scheduled/job-resume turns, and the 6 phases missing from _idle_ran.
Lens B: dream/self-play failures ledgered as success via string proxies
(fixed with the last_dream_outcome / last_self_play_status OUTCOME
SURFACES), the entropy-skip "digest" false positive that disarmed the
backoff, the unpinned phase-3 idle-clock reset, and the crash-truncated
ledger tail merge.
Lens C: the six mutation-sweep survivors, adopted as driven pins (each was
verified red-on-mutant in the bio1c mirror): skip-streak backoff, bio
deterministic override, write-side severity coercion, the 600-char summary
cap, the is_read_only sentinel identity, and the nulled trajectory
collector on the self-play isolate.
(§4NM: the dream is housekeeping only — its surface is "skipped"/"error",
never "ran" — and the reflection / post-mortem / PRM idle phases are gone.)
"""

import datetime
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_biological_watchdog import _make_agent
# Importing the fixture + helper registers them for this module too.
from tests.test_projects_api import client, _fake_tools_map  # noqa: F401


# ── B-M1: dream outcome surface beats the string proxy ──────────────────────

class TestDreamOutcomeSurface:
    @pytest.mark.asyncio
    async def test_error_surface_blocks_ledger_row_and_backs_off(self):
        # The dream RETURNS a success-looking message but the surface says
        # error — the surface must win (this is the discriminator against
        # any string-probe revert).
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        mock_dreamer = MagicMock()
        mock_dreamer.dream = AsyncMock(
            return_value="Dream Complete. Synthesized 2 new meta-memories "
                         "and extracted 3 heuristics.")
        mock_dreamer.last_dream_outcome = {"phase": "error",
                                           "side_output": False}
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()
        phases = [c.args[0] for c in
                  agent._record_autonomous_activity.call_args_list]
        assert "maintenance" not in phases, (
            "a dream whose outcome surface says ERROR must not mint a "
            "'REM cycle ran' ledger row (EXPECT_PERIODIC blinding)")
        assert agent._dream_skip_streak == 1, (
            "an erroring dream must feed the backoff streak, not reset it "
            "to max-cadence refire")

    @pytest.mark.asyncio
    async def test_ran_surface_mints_row_even_when_message_lies(self):
        # Inverse discriminator: skip-looking message, surface says ran.
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        agent._dream_skip_streak = 2
        mock_dreamer = MagicMock()
        mock_dreamer.dream = AsyncMock(
            return_value="Not enough entropy to dream.")
        mock_dreamer.last_dream_outcome = {"phase": "ran",
                                           "side_output": True}
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()
        phases = [c.args[0] for c in
                  agent._record_autonomous_activity.call_args_list]
        assert "maintenance" in phases          # §4NM: was "dream"
        assert agent._dream_skip_streak == 0

    @pytest.mark.asyncio
    async def test_string_fallback_recognizes_error_shapes(self):
        # Mocked Dreamer without a dict surface (auto-vivified MagicMock
        # attr) → the string fallback must classify "Dream error:" as
        # not-ran (the pre-fix probes could not see it at all).
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        mock_dreamer = MagicMock()   # last_dream_outcome = child mock ≠ dict
        mock_dreamer.dream = AsyncMock(
            return_value="Dream error: database is locked")
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()
        phases = [c.args[0] for c in
                  agent._record_autonomous_activity.call_args_list]
        assert "dream" not in phases
        assert agent._dream_skip_streak == 1

    @pytest.mark.asyncio
    async def test_entropy_skip_bare_message_feeds_streak(self):
        # B-MINOR-5: the bare entropy message contains "digests", which the
        # old side-output probe ("digest") matched — the backoff streak
        # never advanced on this path. The narrowed probe must let it.
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        mock_dreamer = MagicMock()
        mock_dreamer.dream = AsyncMock(return_value=(
            "Not enough entropy to dream. (Need ≥3 auto-memories or ≥3 "
            "trajectory/self-play digests to form heuristics)"))
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()
        assert agent._dream_skip_streak == 1, (
            "'digests' in the bare skip message must not read as side "
            "output — that kept the churn backoff permanently disarmed")


def _housekeeping_ctx():
    """A context the housekeeping dream can run on with no side work: a
    store exists, no project store / collector / graph, no real memory dir,
    and the 6-hourly store maintenance is not due."""
    import time as _time
    return SimpleNamespace(memory_system=SimpleNamespace(collection=MagicMock()),
                           project_store=None, trajectory_collector=None,
                           graph_memory=None, memory_dir=None,
                           _last_store_maintenance=_time.time())


class TestDreamerStampsTheSurface:
    """The real Dreamer must stamp last_dream_outcome at its terminal sites
    (the tick-side classification is only as honest as these stamps)."""

    @pytest.mark.asyncio
    async def test_memory_unavailable_stamps_error(self):
        from ghost_agent.core.dream import Dreamer
        ctx = MagicMock()
        ctx.memory_system = None
        d = Dreamer(ctx)
        out = await d.dream(model_name="test-model")
        assert "Memory system not available" in str(out)
        assert {k: d.last_dream_outcome[k] for k in ("phase", "side_output")} == {
            "phase": "error", "side_output": False}
        assert d.last_dream_outcome["summary"].startswith("maintenance:")

    @pytest.mark.asyncio
    async def test_housekeeping_with_no_side_work_stamps_skipped(self):
        from ghost_agent.core.dream import Dreamer
        d = Dreamer(_housekeeping_ctx())
        out = await d.dream(model_name="test-model")
        assert "Housekeeping cycle done" in str(out)
        assert d.last_dream_outcome["phase"] == "skipped"
        # §4CB R2 A-MAJ-3: `isinstance(..., bool)` was vacuous — hardcoding
        # side_output=True (which permanently resets the backoff streak on
        # every skip, reintroducing B-MINOR-5 on the surface path) passed
        # it. This harness produces NO digest/manifest/maintenance side
        # work, so the truth is exactly False.
        assert d.last_dream_outcome["side_output"] is False

    @pytest.mark.asyncio
    async def test_housekeeping_with_side_work_stamps_side_output_true(self):
        # The True leg of the same stamp: a cycle that still did side work
        # must NOT feed the backoff streak.
        from ghost_agent.core.dream import Dreamer
        d = Dreamer(_housekeeping_ctx())
        with patch.object(Dreamer, "_backfill_file_manifests",
                          AsyncMock(return_value=2)):
            out = await d.dream(model_name="test-model")
        assert "Described 2 project file(s)" in str(out)
        assert {k: d.last_dream_outcome[k] for k in ("phase", "side_output")} == {
            "phase": "skipped", "side_output": True}
        assert "2 project file descriptions written" in d.last_dream_outcome["summary"]   # §4PQ


# ── B-M2: self-play ledger row requires a CONCLUDED session ─────────────────



# ── B-M3: the load-bearing phase-3 idle-clock reset, executed ────────────────

# ── A-F2: preamble raises can no longer starve the cooldowns ────────────────

def _raise_on_biological_hook(real_pretty_log=None):
    def _plog(*args, **kwargs):
        if args and args[0] == "Idle Maintenance":   # §4NN: was "Biological Hook"
            raise OSError(28, "No space left on device")
    return _plog


class TestPreambleRaiseContained:
    @pytest.mark.asyncio
    async def test_phase2_preamble_raise_advances_cooldown(self):
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        before_tick = datetime.datetime.now()
        with patch("ghost_agent.core.dream.Dreamer") as MockDreamer, \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=_raise_on_biological_hook()), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            # must NOT raise: the log line sits inside the phase try
            # (§4NM: it now reports what the pass DID, so it follows the pass)
            await agent._biological_tick()
        assert agent._last_dream_at >= before_tick, (
            "a raising preamble must still consume the cooldown — "
            "otherwise the failing phase refires every 60s tick and "
            "starves every later phase while the watchdog reports alive")



# ── A-F1: /advance marks the whole run as a foreground request ──────────────

class TestAdvanceRouteForeground:
    def test_advance_marks_and_unmarks(self, client, monkeypatch):
        tc, store, context = client
        context.llm_client = SimpleNamespace(foreground_requests=0)

        async def _search(**kwargs):
            return "results"
        _fake_tools_map(monkeypatch, {"web_search": _search})

        seen = {}

        async def _fake_advance(ctx, pid, **kw):
            seen["during"] = context.llm_client.foreground_requests
            return SimpleNamespace(ok=True, task_id=None,
                                   classification="idle", summary="",
                                   artifact_id=None)
        monkeypatch.setattr(
            "ghost_agent.api.projects_routes.advance_once", _fake_advance)

        pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
        r = tc.post(f"/api/projects/{pid}/advance")
        assert r.status_code == 200
        assert seen["during"] == 1, (
            "the advance (LLM classifier + 4096-token code gen on the main "
            "slot) must run marked — unmarked, the biological tick ran "
            "journal consolidation MID-advance")
        assert context.llm_client.foreground_requests == 0

    def test_advance_unmarks_on_raise(self, client, monkeypatch):
        tc, store, context = client
        context.llm_client = SimpleNamespace(foreground_requests=0)

        async def _search(**kwargs):
            return "results"
        _fake_tools_map(monkeypatch, {"web_search": _search})

        async def _boom(ctx, pid, **kw):
            raise RuntimeError("advance blew up")
        monkeypatch.setattr(
            "ghost_agent.api.projects_routes.advance_once", _boom)

        pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
        with pytest.raises(RuntimeError):
            tc.post(f"/api/projects/{pid}/advance")
        assert context.llm_client.foreground_requests == 0, (
            "try/finally: a raise must never leak the counter — a stuck "
            "positive count parks ALL background LLM work forever")


# ── A-F3: job-resume turns run marked ────────────────────────────────────────

class TestJobResumeForeground:
    @pytest.mark.asyncio
    async def test_resume_marks_during_handle_chat(self):
        from ghost_agent.main import _resume_after_job
        llm = SimpleNamespace(foreground_requests=0, foreground_tasks=0)
        seen = {}

        async def _hc(body, bg, request_id=None):
            seen["during"] = llm.foreground_requests
            seen["request_id"] = request_id
            return ("ok", None, None)

        agent = SimpleNamespace(handle_chat=_hc, context=None)
        context = SimpleNamespace(args=SimpleNamespace(model="m"),
                                  llm_client=llm, agent=agent,
                                  sandbox_manager=None)
        agent.context = context
        ok = await _resume_after_job(
            context, {"id": "job-pin-af3", "state": "done", "exit_code": 0})
        assert ok is True
        assert seen.get("during") == 1, (
            "a job-resume wake is a full agent turn on the main slot — "
            "unmarked, idle phases and the RSS execv restart treat the "
            "process as idle mid-turn")
        assert llm.foreground_requests == 0


# ── A-F4: mid phases report into _idle_ran ───────────────────────────────────

class TestIdleRanCoverage:
    @pytest.mark.asyncio
    async def test_tidy_phase_reports_in_idle_ran(self, caplog):
        # 6 of 15 phases never appended to _idle_ran, so the tick-end
        # summary (and the "no phase ran" diagnostic, #40) lied for them.
        # Tidy is roll-less: at 20 min idle with all dice lost it is the
        # phase that fires — the summary line must now name it.
        agent = _make_agent(idle_seconds=1200, memory_ids=0)
        _store = MagicMock()
        _store.list_projects.return_value = []
        agent.context.project_store = _store
        with patch("ghost_agent.core.dream.Dreamer"), \
             patch("ghost_agent.core.agent.random.random",
                   return_value=0.99), \
             caplog.at_level(logging.INFO, logger="GhostAgent"):
            await agent._biological_tick()
        msgs = [r.getMessage() for r in caplog.records
                if "idle cycle: ran" in r.getMessage()]
        assert msgs and "tidy" in msgs[0], (
            "the tidy phase ran (cooldown consumed, work attempted) but "
            "never reported into _idle_ran")


# ── lens C P-M10: skip-streak backoff actually stretches the cooldown ───────

class TestSkipStreakBackoff:
    @pytest.mark.asyncio
    async def test_backoff_defers_redream(self):
        agent = _make_agent(idle_seconds=1200, memory_ids=5)
        agent._bio_deterministic = True          # win every dice roll
        now = datetime.datetime.now()
        agent._dream_skip_streak = 3             # eff cooldown 1800*(1+3)=7200
        agent._last_dream_at = now - datetime.timedelta(seconds=3600)
        with patch("ghost_agent.core.dream.Dreamer") as MockDreamer:
            MockDreamer.return_value.dream = AsyncMock(return_value="")
            await agent._biological_tick()
            # streak=3 must hold the dream until 7200s — the backoff was
            # the 2026-07-29 churn fix and had NO executed pin (M10).
            MockDreamer.assert_not_called()


# ── lens C P-M11: --bio-deterministic must override the dice ────────────────

def test_bio_deterministic_overrides_random():
    import ghost_agent.core.agent as agent_mod
    ag = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    ag._bio_deterministic = True
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        assert ag._bio_roll(0.2) is True
        assert ag._bio_roll(0.0) is True


# ── lens C P-M13/P-M15: ledger write-side coercion + summary cap ────────────

class TestLedgerWriteSide:
    def test_bad_severity_is_coerced_on_disk(self, tmp_path):
        from ghost_agent.core.autonomous_activity import ActivityLog
        p = tmp_path / "act.jsonl"
        ActivityLog(p).record("a", "x", severity="bogus")
        raw = json.loads(p.read_text().splitlines()[0])
        assert raw["severity"] == "info", (
            "write-side coercion — the read-path re-coercion in from_dict "
            "must not be the only guard (external readers parse the file)")

    def test_midsize_summary_clamped_to_600(self, tmp_path):
        from ghost_agent.core.autonomous_activity import ActivityLog
        p = tmp_path / "act.jsonl"
        log = ActivityLog(p)
        log.record("a", "y" * 5000)   # under the 16KB LINE cap
        recs, _ = log.read_since(0)
        assert len(recs[0].summary) <= 600, (
            "the 16KB line-cap fallback shadowed this in the old test — "
            "probe BETWEEN the two caps")

    def test_partial_tail_is_healed_on_next_append(self, tmp_path):
        # B-M6: a crash mid-write leaves a partial line without "\n"; the
        # next record must not merge into it (the merged line was skipped
        # by read_since and the first post-restart record vanished).
        from ghost_agent.core.autonomous_activity import ActivityLog
        p = tmp_path / "act.jsonl"
        log = ActivityLog(p)
        log.record("a", "before crash")
        with open(p, "a", encoding="utf-8") as f:
            f.write('{"ts": 1, "phase": "torn')   # torn tail, no newline
        log2 = ActivityLog(p)
        assert log2.record("b", "first after restart") is True
        recs, _ = log2.read_since(0)
        summaries = [r.summary for r in recs]
        assert "first after restart" in summaries, (
            "the post-restart record merged into the torn tail and was "
            "skipped as malformed — heal with a leading newline")
        assert "before crash" in summaries


# ── lens C P-M17/P-M19: self-play isolate invariants ─────────────────────────


# ═════════════════════════════ §4CB ROUND 2 ═════════════════════════════════
# R2 lens A proved the R1 pattern held (13th consecutive round): the worst
# defect was inside R1's B-M2 gate — counterfactual.py stamps "" on the
# SHARED Dreamer, and `"" is not None` walked through it. These pins cover
# the R2 fixes: the "" boundary (gate truthiness + the missing pre-clear),
# the raising-dream streak, the reflection/postmortem preamble containment
# (lens B: the A-F2 class had two unfixed instances), the shared foreground
# helper, the bench cancel banks clear, and the last_dream_outcome pre-clear.


class TestSelfPlayEmptyStatusBoundary:
    @pytest.mark.asyncio
    async def test_empty_string_status_mints_no_row(self):
        # The counterfactual arm plants "" on the same Dreamer instance the
        # fresh self-play then uses; "" must read as "never concluded".
        agent = _make_agent(idle_seconds=4000, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        mock_dreamer = MagicMock()
        mock_dreamer.synthetic_self_play = AsyncMock(
            return_value="Self-Play encountered an error: generation failed")
        mock_dreamer.last_self_play_status = ""   # the counterfactual leftover
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()
        phases = [c.args[0] for c in
                  agent._record_autonomous_activity.call_args_list]
        assert "self_play" not in phases, (
            'the counterfactual leftover "" walked through the R1 identity '
            "gate and re-minted the false 'session ran' row")



class TestRaisingDreamFeedsBackoff:
    @pytest.mark.asyncio
    async def test_dream_raise_increments_streak_and_mints_no_row(self):
        # §4CB R2 A-MAJ-2: the surface stamps RETURN sites only; a raising
        # dream skipped the classification wholesale, freezing the streak —
        # a permanently RAISING dream refired at base cadence forever.
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        mock_dreamer = MagicMock()
        mock_dreamer.dream = AsyncMock(
            side_effect=OSError(28, "No space left on device"))
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()   # must not raise (phase-2 except)
        phases = [c.args[0] for c in
                  agent._record_autonomous_activity.call_args_list]
        assert "dream" not in phases
        assert agent._dream_skip_streak == 1, (
            "the RAISE flavor of a failing dream must feed the backoff too")


class TestDreamOutcomePreclear:
    @pytest.mark.asyncio
    async def test_escaping_raise_leaves_surface_none_not_stale(self):
        # A raise that escapes dream() must leave the surface at the
        # pre-cleared None — never at a STALE prior verdict (the surface's
        # stated identity: None = did not conclude).
        from ghost_agent.core.dream import Dreamer
        dreamer = Dreamer(_housekeeping_ctx())
        dreamer.last_dream_outcome = {"phase": "skipped", "side_output": True}
        with patch.object(Dreamer, "_daily_store_care",
                          AsyncMock(side_effect=RuntimeError("store care boom"))):
            with pytest.raises(RuntimeError):
                await dreamer.dream(model_name="test-model")
        assert dreamer.last_dream_outcome is None, (
            "the entry pre-clear is the only thing standing between a raise "
            "and a reused Dreamer reading a stale verdict")


class TestSharedForegroundHelper:
    @pytest.mark.asyncio
    async def test_helper_marks_during_and_unmarks_after(self):
        from ghost_agent.main import _handle_chat_foreground
        llm = SimpleNamespace(foreground_requests=0)
        seen = {}

        async def _hc(body, bg, request_id=None):
            seen["during"] = llm.foreground_requests
            seen["request_id"] = request_id
            return ("ok", None, None)

        agent = SimpleNamespace(handle_chat=_hc, context=None)
        context = SimpleNamespace(llm_client=llm, agent=agent)
        agent.context = context
        out = await _handle_chat_foreground(context, {"messages": []},
                                            "sched-pin")
        assert out[0] == "ok"
        assert seen["during"] == 1
        assert seen["request_id"] == "sched-pin"
        assert llm.foreground_requests == 0

    @pytest.mark.asyncio
    async def test_helper_unmarks_on_raise(self):
        from ghost_agent.main import _handle_chat_foreground
        llm = SimpleNamespace(foreground_requests=0)

        async def _boom(body, bg, request_id=None):
            raise RuntimeError("turn blew up")

        agent = SimpleNamespace(handle_chat=_boom, context=None)
        context = SimpleNamespace(llm_client=llm, agent=agent)
        agent.context = context
        with pytest.raises(RuntimeError):
            await _handle_chat_foreground(context, {}, "sched-x")
        assert llm.foreground_requests == 0, (
            "a leaked +1 parks ALL background LLM work forever")


# ═════════════════════════════ §4CB ROUND 3 ═════════════════════════════════
# R3 lens A found NO live defect in the R2 fixes — the streak broke at the
# code level. What it found instead were verification gaps: the R2 gate-
# tightening made "concluded ⟹ truthy" newly load-bearing with only a
# token pin guarding it (a feature-killing tail-clear walked through 356
# tests), the streak pin couldn't tell increment from set-to-1, and two R2
# MINOR fixes (consumers-off no-append, skip-line _safe_pretty_log — both
# on the PRM phase, removed §4NM) were revertible unseen. These are the
# executed pins for each.




class TestRaiseStreakIncrements:
    @pytest.mark.asyncio
    async def test_raise_path_increments_not_sets(self):
        # R3 MINOR-2: the 0→1 pin passed with `streak = 1` hardcoded,
        # which caps a permanently-raising dream's backoff at 2x instead
        # of 4x. Start above 1 so only a true increment passes.
        agent = _make_agent(idle_seconds=900, memory_ids=4)
        agent._record_autonomous_activity = MagicMock()
        agent._dream_skip_streak = 2
        mock_dreamer = MagicMock()
        mock_dreamer.dream = AsyncMock(
            side_effect=OSError(28, "No space left on device"))
        with patch("ghost_agent.core.dream.Dreamer",
                   return_value=mock_dreamer), \
             patch("ghost_agent.core.agent.random.random", return_value=0.1):
            await agent._biological_tick()
        assert agent._dream_skip_streak == 3
