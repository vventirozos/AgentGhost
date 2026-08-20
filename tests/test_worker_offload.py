"""Tier-2 worker-node offload (2026-07-11).

The main model has ONE inference slot (`-np 1`) and turns are serialized, so
every auxiliary LLM call either blocks the user's turn or queues behind it. A
secondary box (e.g. a small model on a Mac Mini) is a SECOND SLOT. These tests
pin the offload wiring and — crucially — that everything still behaves exactly
as before when NO worker pool is configured.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.build_gates import constraint_gate, _parse_verdict


_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

_CLEAN = json.dumps({"violates": False, "constraint": "", "evidence": ""})
_DIRTY = json.dumps({"violates": True, "constraint": "no verbatim quotes",
                     "evidence": "a.md: 'be concise'"})


class FakeLLM:
    """Records the requested pool AND models which leg actually served it.

    ⚠ This used to record only `use_worker` — the REQUEST — and answer from a
    script regardless, so it passed identically in the broken world: with the
    worker pool down, the real client answers a `use_worker=True` call from
    the MAIN model, and `constraint_gate` then "independently confirmed" its
    own screen with the same model at temperature 0.0. The test asserted
    `llm.calls == [True, False]`, i.e. what was ASKED (LLM review R2 lens C).

    `worker_up` models the real fallback: a requested pool that is configured
    but failing is served by main, and the result is stamped accordingly —
    which is what the real client does via `_stamp_leg`.
    """

    def __init__(self, replies, worker_clients=None, worker_up=True):
        self.replies = list(replies)
        self.worker_clients = worker_clients or []
        self.worker_up = worker_up
        self.calls = []            # list[bool] — use_worker per call
        self.served = []           # list[str]  — the leg that answered

    async def chat_completion(self, payload, use_worker=False,
                              is_background=False, **kw):
        self.calls.append(bool(use_worker))
        content = self.replies.pop(0) if self.replies else ""
        if isinstance(content, Exception):
            raise content
        if use_worker and self.worker_clients and self.worker_up:
            leg, fell = "worker", ""
        elif use_worker and self.worker_clients:
            leg, fell = "main", "worker"      # configured but failing
        else:
            leg, fell = "main", ""
        self.served.append(leg)
        return {"choices": [{"message": {"content": content}}],
                "_ghost_leg": {"served_by": leg, "fell_back_from": fell}}


def _ctx(llm):
    return SimpleNamespace(llm_client=llm,
                           args=SimpleNamespace(model="qwen"))


def _gate(llm, files=None):
    return asyncio.run(constraint_gate(
        _ctx(llm), ["no verbatim quotes"],
        files if files is not None else {"a.md": "body"}))


# ══════════════════════════════════════════════════════════════════════
# _parse_verdict
# ══════════════════════════════════════════════════════════════════════

class TestParseVerdict:
    def test_parses_clean(self):
        assert _parse_verdict(_CLEAN)["violates"] is False

    def test_parses_dirty(self):
        assert _parse_verdict(_DIRTY)["violates"] is True

    def test_parses_embedded_in_prose(self):
        assert _parse_verdict(f"Sure!\n{_DIRTY}\nHope that helps")["violates"] \
            is True

    def test_unparseable_is_none(self):
        assert _parse_verdict("no json here") is None
        assert _parse_verdict("") is None
        assert _parse_verdict("{}") is None      # no "violates" key


# ══════════════════════════════════════════════════════════════════════
# No worker pool → byte-identical to the old behaviour
# ══════════════════════════════════════════════════════════════════════

class TestNoWorkerPoolUnchanged:
    def test_clean_verdict_single_call_on_main(self):
        llm = FakeLLM([_CLEAN])                   # no worker_clients
        ok, reason = _gate(llm)
        assert ok is True and reason == ""
        assert llm.calls == [True]                # use_worker=True …
        # …but with no pool it falls back to main (LLMClient contract), and
        # crucially there is NO second confirm call.
        assert len(llm.calls) == 1

    def test_violation_blocks_without_a_confirm_pass(self):
        """No worker pool ⇒ the screen WAS the main model ⇒ don't re-ask it."""
        llm = FakeLLM([_DIRTY])
        ok, reason = _gate(llm)
        assert ok is False
        assert "CONSTRAINT VIOLATION" in reason
        assert "no verbatim quotes" in reason
        assert len(llm.calls) == 1                # no wasteful double call

    def test_unparseable_fails_open(self):
        ok, _ = _gate(FakeLLM(["garbage"]))
        assert ok is True

    def test_llm_error_fails_open(self):
        ok, _ = _gate(FakeLLM([RuntimeError("upstream down")]))
        assert ok is True

    def test_no_files_skips_entirely(self):
        llm = FakeLLM([_DIRTY])
        ok, _ = _gate(llm, files={})
        assert ok is True and llm.calls == []     # no LLM call at all


# ══════════════════════════════════════════════════════════════════════
# Worker pool → screen off-main, confirm a veto on main
# ══════════════════════════════════════════════════════════════════════

WORKER = [{"url": "http://mini:8088", "model": "gemma"}]


class TestScreenThenConfirm:
    def test_clean_screen_costs_the_main_model_nothing(self):
        """The common case: the small model answers, the 35B is never touched."""
        llm = FakeLLM([_CLEAN], worker_clients=WORKER)
        ok, _ = _gate(llm)
        assert ok is True
        assert llm.calls == [True]                # worker only — no main call

    def test_worker_veto_confirmed_by_main_blocks(self):
        llm = FakeLLM([_DIRTY, _DIRTY], worker_clients=WORKER)
        ok, reason = _gate(llm)
        assert ok is False and "CONSTRAINT VIOLATION" in reason
        assert llm.calls == [True, False]         # screen on worker, confirm on main

    def test_worker_veto_NOT_confirmed_passes(self):
        """A weak screening model must never deadlock work on its own — an
        unconfirmed veto is exactly how a false positive blocks a project."""
        llm = FakeLLM([_DIRTY, _CLEAN], worker_clients=WORKER)
        ok, reason = _gate(llm)
        assert ok is True and reason == ""
        assert llm.calls == [True, False]

    def test_confirm_uses_main_models_evidence(self):
        main = json.dumps({"violates": True, "constraint": "C-main",
                           "evidence": "E-main"})
        llm = FakeLLM([_DIRTY, main], worker_clients=WORKER)
        ok, reason = _gate(llm)
        assert ok is False
        assert "C-main" in reason and "E-main" in reason
        assert "no verbatim quotes" not in reason   # not the screen's text

    def test_confirm_unparseable_fails_open(self):
        llm = FakeLLM([_DIRTY, "garbage"], worker_clients=WORKER)
        assert _gate(llm)[0] is True

    def test_confirm_error_fails_open(self):
        llm = FakeLLM([_DIRTY, RuntimeError("main down")],
                      worker_clients=WORKER)
        assert _gate(llm)[0] is True


# ══════════════════════════════════════════════════════════════════════
# Other offloaded call sites + observability
# ══════════════════════════════════════════════════════════════════════

class TestOffloadWiring:
    def test_autoadvance_classifier_uses_worker(self):
        src = (_SRC / "core" / "agent.py").read_text()
        block = src.split("Classify this task into EXACTLY one word", 1)[1][:600]
        assert "use_worker=True" in block

    def test_smart_memory_already_offloaded(self):
        src = (_SRC / "core" / "agent.py").read_text()
        # Window widened 2500→3200 (2026-07-22): the extract call gained a
        # bounded `timeout=90.0` + an explanatory comment, pushing the
        # `use_worker=True` kwarg further into the function body.
        block = src.split("async def run_smart_memory_task", 1)[1][:3200]
        assert "use_worker=True" in block

    def test_constraint_gate_screens_on_worker(self):
        src = (_SRC / "core" / "build_gates.py").read_text()
        assert "_ask(use_worker=True)" in src
        assert "screened_off_main" in src

    def test_quality_critical_paths_stay_on_main(self):
        """Reflection + post-mortem WRITE lessons and classify defects — a
        weak judge poisons the learning stack, and they run at idle (when the
        main model is free anyway). Deliberately NOT offloaded."""
        src = (_SRC / "main.py").read_text()
        for marker in ("_critique_fn", "_analyze_fn"):
            block = src.split(marker, 1)[1][:1800]
            assert "use_worker=True" not in block, (
                f"{marker} must stay on the main model")


class TestHealthExposesNodes:
    def test_health_reports_pools(self):
        fastapi = pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ghost_agent.api.routes import router

        app = FastAPI()
        app.include_router(router)
        llm = SimpleNamespace(
            worker_clients=WORKER, critic_clients=[], swarm_clients=[],
            coding_clients=[], vision_clients=[], image_gen_clients=[],
            foreground_requests=0, foreground_tasks=0,
        )
        context = SimpleNamespace(
            args=SimpleNamespace(api_key=""), llm_client=llm,
            memory_system=None,
        )
        app.state.agent = SimpleNamespace(context=context)
        app.state.resolved_config = {}

        with TestClient(app) as c:
            nodes = c.get("/api/health").json()["nodes"]
        assert nodes["worker"] == ["http://mini:8088"]
        assert nodes["critic"] == []
        assert set(nodes) == {"worker", "critic", "swarm", "coding",
                              "vision", "image_gen"}


class TestScreenThenConfirmUnderADEGRADEDWorker:
    """The property the old fake could not see: when the screen itself was
    served by MAIN (worker configured but failing), there is no independent
    second opinion to be had — asking the same model the same question again
    is not a confirmation."""

    def test_a_main_served_screen_is_not_confirmed_by_main(self):
        import asyncio
        from ghost_agent.core.build_gates import constraint_gate

        llm = FakeLLM(
            replies=['{"violates": true, "constraint": "c", "evidence": "e"}',
                     '{"violates": true, "constraint": "c", "evidence": "e"}'],
            worker_clients=[{"url": "http://n", "model": "m"}],
            worker_up=False)
        ok, reason = _gate(llm)
        assert llm.served == ["main"], (
            f"the gate asked the main model {llm.served.count('main')} times "
            f"and called it screen-then-confirm: {llm.served}")
        assert ok is False, "a confirmed violation must still block"

    def test_a_worker_served_screen_IS_confirmed_on_main(self):
        import asyncio
        from ghost_agent.core.build_gates import constraint_gate

        llm = FakeLLM(
            replies=['{"violates": true, "constraint": "c", "evidence": "e"}',
                     '{"violates": false}'],
            worker_clients=[{"url": "http://n", "model": "m"}],
            worker_up=True)
        ok, reason = _gate(llm)
        assert llm.served == ["worker", "main"], llm.served
        assert ok is True, "an unconfirmed worker veto must not block"


class TestTheGateCannotBeSkippedByAnOutage:
    """⚠ A SAFETY REGRESSION INTRODUCED AND CAUGHT IN THE SAME ROUND.

    Adding `off_main_only=is_background` (to stop a background gate from
    dogpiling the single main slot) combined with the client's "unconfigured
    or exhausted pool + off_main_only → RAISE" and this function's fail-open
    posture. Result, measured: with the worker pool down and
    `is_background=True` (the DEFAULT), the constraint gate returned
    `passed=True` having made ZERO model calls — a violating build shipped,
    logging one debug line.

    Fail-open is right for "the judge is broken"; it is NOT right for "we
    declined to ask". The gate now falls back to main deliberately, without
    the pool flag, so it is queued like any other main-targeted call."""

    def _violating(self):
        return '{"violates": true, "constraint": "no X", "evidence": "used X"}'

    def test_a_background_gate_still_blocks_when_the_worker_is_down(self):
        """⚠ DRIVEN BY THE REAL CLIENT, not FakeLLM.

        The first version of this test used `FakeLLM`, which swallows
        `off_main_only` into `**kw` and can therefore never raise
        `OffMainNodeUnavailable` — so it could not enter the code path it is
        named for. Deleting the entire fix passed 140 tests across nine files
        (R4 lens A, MAJOR-2). Only the real `LLMClient` produces the raise
        that the gate's fail-open then swallowed."""
        import asyncio as _aio
        from unittest.mock import AsyncMock, MagicMock
        from ghost_agent.core.llm import LLMClient

        violating = self._violating()
        c = LLMClient("http://127.0.0.1:8088")
        dead = MagicMock()
        dead.post = AsyncMock(side_effect=RuntimeError("worker down"))
        c.worker_clients = [{"url": "http://100.0.0.9:8088", "model": "Nova",
                             "client": dead, "name": "Nova"}]
        c._worker_index = 0
        main_calls = []

        async def _main(*a, **kw):
            main_calls.append(1)
            r = MagicMock()
            r.json = lambda: {"choices": [{"message": {"content": violating}}]}
            r.raise_for_status = lambda: None
            r.status_code, r.text = 200, "{}"
            return r

        c.http_client = MagicMock()
        c.http_client.post = AsyncMock(side_effect=_main)

        ok, reason = _aio.run(constraint_gate(
            _ctx(c), ["no X"], {"a.md": "uses X"}, is_background=True))
        assert ok is False, (
            "a background build with the worker pool down skipped the "
            "constraint gate entirely and shipped a violating artifact")
        assert main_calls, (
            "the gate declined to ask ANY model — fail-open is right for a "
            "broken judge, not for a judge we chose not to consult")

    def test_a_foreground_gate_is_unchanged(self):
        llm = FakeLLM(replies=[self._violating(), self._violating()],
                      worker_clients=[{"url": "http://n", "model": "m"}],
                      worker_up=False)
        ok, _ = asyncio.run(constraint_gate(
            _ctx(llm), ["no X"], {"a.md": "uses X"}, is_background=False))
        assert ok is False
