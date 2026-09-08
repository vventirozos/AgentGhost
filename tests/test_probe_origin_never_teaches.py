"""Diagnostic probes never teach (§4FB, 2026-09-06).

Playbook lesson [49] — trigger "Use deep_research ONCE on the query: llama.cpp
prompt prefill speed apple silicon. Then reply with one short sentence … Do
not investigate anything else, do not read files." — was minted by reflection
from five repeats of a perf-audit PROBE sent through the ordinary user path,
then surfaced 37 times into unrelated requests (req 2422eb25, an image
description, among them). A diagnostic is indistinguishable from the
operator unless the sender says so; once it says so, nothing may learn from
it.

The mark is one header (``X-Ghost-Origin: probe``) → one request-id prefix
(``probe-``) → one population answer from ``turn_origin`` (the predicate the
calibration booking, selfhood, foresight, feedback and lesson-credit gates
already consult with ``== "user"``) → one trajectory ``task_kind`` (``probe``)
that no consumer's ``admitted_task_kinds`` lists, plus an explicit admitted-
kinds filter on the one consumer that had none (reflection).

Worlds where these fail: drop the header mapping; derive the origin from the
context only; stamp the trajectory ``user_request``; let reflection reflect
a FAILED probe (or bench / self_play) row; run the hydration judge or the
lesson-outcome credit on a probe turn; give probes the user family's colours.
"""

import inspect
import json
import logging
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import (
    GhostAgent,
    GhostContext,
    book_graduated_retrieval,
    rubric_shadow_eligible,
    turn_origin,
)
from ghost_agent.core.calibration import CalibrationTracker
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, Trajectory
from ghost_agent.utils import logging as glog
from ghost_agent.utils.logging import (
    ORIGIN_PROBE,
    PROBE_REQUEST_PREFIX,
    _ORIGIN_PALETTES,
    _req_color,
    is_probe_request_id,
    request_id_context,
)

PROBE_REQUEST = ("Use deep_research ONCE on the query: llama.cpp prompt prefill "
                 "speed apple silicon. Then reply with one short sentence about "
                 "what you found. Do not investigate anything else, do not read files.")


class _rid:
    """Set the request-id contextvar for a block, restoring it after."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.tok = request_id_context.set(self.value)

    def __exit__(self, *a):
        request_id_context.reset(self.tok)


def _user_ctx():
    return types.SimpleNamespace(
        skill_memory=types.SimpleNamespace(is_read_only=False))


# ── 1. the predicate ────────────────────────────────────────────────────────

class TestTurnOrigin:

    def test_probe_prefix_makes_the_turn_a_probe(self):
        with _rid(PROBE_REQUEST_PREFIX + "abcd1234"):
            assert turn_origin(_user_ctx()) == ORIGIN_PROBE

    def test_plain_ids_keep_user_sim_and_bench(self):
        with _rid("abcd1234"):
            assert turn_origin(_user_ctx()) == "user"
            assert turn_origin(types.SimpleNamespace(
                skill_memory=types.SimpleNamespace(is_read_only=True))) == "sim"
            assert turn_origin(types.SimpleNamespace(
                turn_origin_label="bench")) == "bench"

    def test_probe_wins_over_an_explicit_label(self):
        """Documented precedence: a probe is a probe whatever context it
        runs in (no bench/sim context ever carries a probe id)."""
        with _rid("probe-x"):
            assert turn_origin(types.SimpleNamespace(
                turn_origin_label="bench")) == ORIGIN_PROBE

    def test_prefix_helper_is_exact(self):
        assert is_probe_request_id("probe-1")
        assert not is_probe_request_id("probe1")
        assert not is_probe_request_id("xprobe-1")
        assert not is_probe_request_id(None)


# ── 2. the gates that already key on the predicate ──────────────────────────

class TestExistingUserGatesExcludeProbes:

    def test_rubric_shadow_gate(self):
        traj = Trajectory(user_request="q", final_response="a",
                          outcome=Outcome.UNKNOWN.value)
        with patch("ghost_agent.core.rubric_grader.shadow_enabled",
                   return_value=True):
            with _rid("user1234"):
                assert rubric_shadow_eligible(_user_ctx(), traj) is True  # live
            with _rid("probe-1234"):
                assert rubric_shadow_eligible(_user_ctx(), traj) is False

    def test_graduated_retrieval_booking(self):
        with _rid("user1234"):
            store = MagicMock()
            book_graduated_retrieval(_user_ctx(), store, ["h1"])
            assert store.mock_calls, "control: a user turn books"
        with _rid("probe-1234"):
            store = MagicMock()
            assert book_graduated_retrieval(_user_ctx(), store, ["h1"]) == 0
            assert not store.mock_calls


# ── 3. the route ────────────────────────────────────────────────────────────

class TestRouteHeader:

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ghost_agent.api import routes as routes_module

        app = FastAPI()
        fake = MagicMock()
        fake.context.args.api_key = None
        fake.context.args.model = "test-model"
        # Capture the call, then fail fast — the error path is the shortest
        # route through the handler and the kwargs are all this pin needs.
        fake.handle_chat = AsyncMock(side_effect=RuntimeError("stop"))
        app.state.agent = fake
        app.state.args = MagicMock()
        app.state.args.model = "test-model"
        app.include_router(routes_module.router)
        return TestClient(app), fake

    BODY = {"messages": [{"role": "user", "content": PROBE_REQUEST}],
            "stream": False}

    def _rid_for(self, headers):
        client, fake = self._client()
        client.post("/api/chat", json=self.BODY, headers=headers)
        assert fake.handle_chat.called
        return fake.handle_chat.call_args.kwargs.get("request_id")

    def test_header_prefixes_a_fresh_request_id(self):
        rid = self._rid_for({"X-Ghost-Origin": "probe"})
        assert is_probe_request_id(rid)
        assert len(rid) > len(PROBE_REQUEST_PREFIX)

    def test_header_keeps_a_client_request_id_under_the_prefix(self):
        rid = self._rid_for({"X-Ghost-Origin": "Probe ", "X-Request-ID": "slack-77"})
        assert rid == "probe-slack-77"

    def test_an_already_prefixed_id_is_not_doubled(self):
        rid = self._rid_for({"X-Ghost-Origin": "probe", "X-Request-ID": "probe-77"})
        assert rid == "probe-77"

    def test_without_the_header_ids_pass_through(self):
        assert self._rid_for({"X-Request-ID": "slack-77"}) == "slack-77"
        assert self._rid_for({}) is None
        assert self._rid_for({"X-Ghost-Origin": "user"}) is None


# ── 4. the real turn loop stamps the population and credits nothing ─────────

class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


def _real_user_context(tmp_path, record_calls):
    """A minimal REAL-enough USER context (mirrors the bench harness in
    tests/test_bench_handle_chat_1c.py minus the bench markers)."""
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "One short sentence: prefill is fast.",
                                 "tool_calls": []}}]})
    context.sandbox_dir = str(tmp_path)
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "test-model"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.memory_bus = None
    context.skill_memory = types.SimpleNamespace(
        is_read_only=False,
        last_playbook_triggers=["real lesson trigger"],
        _playbook_turn_key="",
        get_recent_failures=lambda *a, **k: "No failures",
        record_surfaced_outcomes=lambda *a, **k: record_calls.append((a, k)) or 0,
    )
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_dir = tmp_path / "system" / "memory"
    context.calibration_tracker = CalibrationTracker(
        tmp_path / "system" / "calibration")
    root = tmp_path / "system" / "trajectories"
    context.trajectory_collector = TrajectoryCollector(
        root=root, session_id="probe-test")
    context.trajectory_task_kind = None
    context.turn_origin_label = None
    context.trajectory_extra_static = None
    context.trajectory_user_request_override = None
    return context, root


def _rows(root):
    rows = []
    if not root.exists():
        return rows
    for day in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(day.glob("session-*.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines()
                     if l.strip()]
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id,kind", [
    ("itest0001", "user_request"),          # control: the harness is live
    ("probe-itest01", "probe"),
])
async def test_real_handle_chat_stamps_the_population(
        tmp_path, monkeypatch, request_id, kind):
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    from ghost_agent.core import experiments as ex
    ex.reset_registry_cache()
    calls = []
    context, root = _real_user_context(tmp_path, calls)
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": PROBE_REQUEST}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(), request_id=request_id)
    rows = _rows(root)
    assert rows, "the real finalize wrote no trajectory"
    assert rows[-1]["task_kind"] == kind
    assert rows[-1]["extra"]["req_id"] == request_id
    # (Calibration is pinned separately in TestCalibrationGate: this harness
    # carries no metacog bundle, so the recorder has no reading to pair and
    # records nothing for EITHER run — a row-count assertion here would be
    # vacuous in the direction that matters.)


class TestCalibrationGate:
    """Calibration is a learned instrument. Live-measured 2026-09-06: a
    header-marked probe passed the recorder's read-only carve-out (its skill
    memory is the live one) and landed a row stamped origin="user" — the
    population the instrument exists to score. Executed pin on the real
    recorder with a stashed reading, so the gate is the only variable."""

    def _agent(self, tmp_path, calls, rid):
        context, _root = _real_user_context(tmp_path, [])
        reading = types.SimpleNamespace(
            composite=0.8, entropy_component=0.9, competence_component=0.9,
            below_threshold=False, threshold=0.8)
        context._calib_pending = (rid, reading)
        context.calibration_tracker = types.SimpleNamespace(
            record=lambda **kw: calls.append(kw))
        return GhostAgent(context)

    async def _run(self, tmp_path, rid):
        calls = []
        agent = self._agent(tmp_path, calls, rid)
        with _rid(rid), patch("ghost_agent.core.agent.pretty_log"):
            await agent._record_calibration_safe(
                req_id=rid, tools_run=[], verifier_backfill=None,
                execution_failure_count=0, budget_exhausted=False,
                final_ai_content="PONG", user_request="reply PONG")
        return calls

    @pytest.mark.asyncio
    async def test_a_user_turn_records_a_sample(self, tmp_path):
        calls = await self._run(tmp_path, "abcd0001")
        assert calls and calls[0]["req_id"] == "abcd0001"
        assert calls[0]["origin"] == "user"

    @pytest.mark.asyncio
    async def test_a_probe_turn_records_nothing(self, tmp_path):
        assert await self._run(tmp_path, "probe-0001") == []


class TestCreditGates:

    def _agent(self, bus):
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = types.SimpleNamespace(
            memory_bus=bus, llm_client=None,
            args=types.SimpleNamespace(model="m"),
            skill_memory=types.SimpleNamespace(is_read_only=False))
        return agent

    def _bus(self, turn_id):
        bus = MagicMock()
        bus.last_hydration = {"turn_id": turn_id,
                              "survivors": [{"source": "skill", "trigger": "t"}]}
        bus.judge_hydration_usefulness = MagicMock(return_value=None)
        return bus

    def test_hydration_judge_skips_a_probe_turn(self):
        with patch("ghost_agent.utils.logging.spawn_bg") as spawn:
            bus = self._bus("abcd0001")
            with _rid("abcd0001"):
                self._agent(bus)._judge_hydration_safe("reply", turn_id="abcd0001")
            assert bus.judge_hydration_usefulness.called, "control: user turns judge"
            bus = self._bus("probe-0001")
            with _rid("probe-0001"):
                self._agent(bus)._judge_hydration_safe("reply", turn_id="probe-0001")
            assert not bus.judge_hydration_usefulness.called
            assert spawn.call_count == 1

    @pytest.mark.asyncio
    async def test_lesson_outcome_credit_skips_a_probe_turn(self):
        seen = []
        sm = types.SimpleNamespace(
            record_surfaced_outcomes=lambda *a, **k: seen.append((a, k)) or 0)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = types.SimpleNamespace(
            skill_memory=sm,
            skill_memory_is_read_only=False)
        agent.context.skill_memory.is_read_only = False
        kwargs = dict(surfaced_triggers=["t"], execution_failure_count=0,
                      verifier_backfill=("passed", 0.9),
                      trajectory_id="tj")
        with _rid("abcd0001"):
            await agent._record_lesson_outcomes(**kwargs)
        assert seen, "control: a user turn credits its surfaced lessons"
        seen.clear()
        with _rid("probe-0001"):
            await agent._record_lesson_outcomes(**kwargs)
        assert seen == []


# ── 5. reflection admits only the real population ───────────────────────────

class TestReflectionAdmitsOnlyUserRequests:

    def _reflector(self):
        from ghost_agent.reflection.loop import Reflector
        return Reflector(critique_fn=lambda x: "diag")

    def _traj(self, kind, outcome=Outcome.FAILED.value):
        return Trajectory(user_request=PROBE_REQUEST, final_response="x",
                          outcome=outcome, task_kind=kind)

    @pytest.mark.parametrize("kind,expected", [
        ("user_request", True), ("probe", False), ("bench", False),
        ("self_play", False), ("reflection", False), ("", False),
    ])
    def test_failed_rows_by_kind(self, kind, expected):
        assert self._reflector()._is_reflectable(self._traj(kind)) is expected

    def test_a_passed_user_request_is_still_not_reflectable(self):
        assert self._reflector()._is_reflectable(
            self._traj("user_request", Outcome.PASSED.value)) is False

    def test_the_kinds_come_from_admissibility_not_a_private_list(self, monkeypatch):
        from ghost_agent.core import admissibility
        monkeypatch.setattr(admissibility, "admitted_task_kinds",
                            lambda consumer: ("probe",))
        assert self._reflector()._is_reflectable(self._traj("probe")) is True
        assert self._reflector()._is_reflectable(self._traj("user_request")) is False

    def test_admissibility_failure_reflects_nothing(self, monkeypatch, caplog):
        from ghost_agent.core import admissibility

        def _boom(consumer):
            raise RuntimeError("table missing")
        monkeypatch.setattr(admissibility, "admitted_task_kinds", _boom)
        caplog.set_level(logging.WARNING)
        assert self._reflector()._is_reflectable(self._traj("user_request")) is False
        assert any("admissibility unavailable" in r.getMessage()
                   for r in caplog.records)


# ── 6. the stream shows the population ──────────────────────────────────────

class TestStreamFamily:

    def test_probe_has_its_own_family_disjoint_from_user(self):
        assert "probe" in _ORIGIN_PALETTES
        assert not set(_ORIGIN_PALETTES["probe"]) & set(_ORIGIN_PALETTES["user"])
        assert len(set(_ORIGIN_PALETTES["probe"])) >= 2

    def test_req_color_draws_from_the_probe_family(self, monkeypatch):
        monkeypatch.setattr(glog, "_USE_COLOR", True)
        code = int(_req_color("probe-abcd", origin="probe").split(";")[-1].rstrip("m"))
        assert code in _ORIGIN_PALETTES["probe"]
        user = int(_req_color("probe-abcd", origin="user").split(";")[-1].rstrip("m"))
        assert user in _ORIGIN_PALETTES["user"]


# ── 7. the memory bus names its fan-out queries ─────────────────────────────

class TestBusSubQueriesAreLogged:

    @pytest.mark.asyncio
    async def test_fan_out_queries_land_in_the_log(self, caplog):
        from ghost_agent.core.bus import MemoryBus
        bus = MemoryBus()
        llm = MagicMock()
        llm.route = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(
            ["auth middleware architecture", "schema migration patterns"])}}]})
        caplog.set_level(logging.INFO)
        await bus.hydrate_context(
            "How should I handle the authentication migration given compliance requirements",
            llm_client=llm)
        lines = [r.getMessage() for r in caplog.records
                 if "memory bus sub-queries" in r.getMessage()]
        assert lines, [r.getMessage() for r in caplog.records][-10:]
        assert "auth middleware architecture" in lines[0]
        assert "schema migration patterns" in lines[0]
        assert lines[0].startswith("memory bus sub-queries (2)")

    @pytest.mark.asyncio
    async def test_a_short_query_logs_no_fan_out(self, caplog):
        from ghost_agent.core.bus import MemoryBus
        caplog.set_level(logging.INFO)
        await MemoryBus().hydrate_context("hello world", llm_client=None)
        assert not [r for r in caplog.records
                    if "memory bus sub-queries" in r.getMessage()]
