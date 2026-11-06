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
that no consumer's ``admitted_task_kinds`` lists. (§4NM: the reflection
consumer and the hydration-usefulness judge were removed; their pins went
with them.)

Worlds where these fail: drop the header mapping; derive the origin from the
context only; stamp the trajectory ``user_request``; run the lesson-outcome
credit on a probe turn; give probes the user family's colours.
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
    rubric_shadow_eligible,
    turn_origin,
)
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
        )


# ── 1. the predicate ────────────────────────────────────────────────────────

class TestTurnOrigin:

    def test_probe_prefix_makes_the_turn_a_probe(self):
        with _rid(PROBE_REQUEST_PREFIX + "abcd1234"):
            assert turn_origin(_user_ctx()) == ORIGIN_PROBE

    def test_plain_ids_keep_user_sim_and_bench(self):
        with _rid("abcd1234"):
            assert turn_origin(_user_ctx()) == "user"
            assert turn_origin(types.SimpleNamespace(
                simulated=True)) == "sim"
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
    context.args.model = "test-model"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.memory_bus = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_dir = tmp_path / "system" / "memory"
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
        caplog.set_level(logging.INFO)
        await bus.hydrate_context(
            # §4NQ: only the keyword split fans out now
            "how should i handle the authentication migration and also what does compliance require")
        lines = [r.getMessage() for r in caplog.records
                 if "memory bus sub-queries" in r.getMessage()]
        assert lines, [r.getMessage() for r in caplog.records][-10:]
        assert "how should i handle the authentication migration" in lines[0]
        assert "what does compliance require" in lines[0]
        assert lines[0].startswith("memory bus sub-queries (2)")

    @pytest.mark.asyncio
    async def test_a_short_query_logs_no_fan_out(self, caplog):
        from ghost_agent.core.bus import MemoryBus
        caplog.set_level(logging.INFO)
        await MemoryBus().hydrate_context("hello world")
        assert not [r for r in caplog.records
                    if "memory bus sub-queries" in r.getMessage()]
