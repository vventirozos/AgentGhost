"""Tests for core/failure_dimension.py — harness-dimension failure
attribution (MemoHarness adaptation, 2026-07-19).

These tests pin:
  * one realistic classification per dimension (regex tables)
  * precedence: harness dimensions before `model`; network timeouts land
    on tool_interaction while LLM ReadTimeouts land on generation_control
  * empty/unmatched input → unknown
  * env-toggle helpers (per-call re-read, kill-switch semantics)
  * adjudicate_dimension: valid label wins; garbage/absent client falls
    back to the heuristic; the CLASSIFY_FAILURE routing label is used
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.failure_dimension import (
    DIM_UNKNOWN,
    DIMENSIONS,
    adjudicate_dimension,
    adjudicate_enabled,
    classify_failure_dimension,
    distill_enabled,
    distill_max,
    failure_dim_enabled,
)


class _RouteStub:
    """Duck-typed llm_client exposing only route(); records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def route(self, task, payload, **kw):
        self.calls.append((task, payload, kw))
        return self.responses.pop(0) if self.responses else None


# ------------------------------------------------------------- classifier

class TestClassify:
    @pytest.mark.parametrize("text,expected", [
        ("SEARCH/REPLACE block failed to parse; markers written to file",
         "output_processing"),
        ("retrieved a quarantined lesson from the playbook", "memory"),
        ("verifier evidence was truncated at 3400 chars", "context_assembly"),
        ("verifier REFUTED the claim but the file content was correct",
         "orchestration"),
        ("browser click failed: element not found for selector #submit",
         "tool_interaction"),
        ("finish_reason length — the completion was cut off",
         "generation_control"),
        ("the model hallucinated a function name that does not exist",
         "model"),
    ])
    def test_realistic_signals(self, text, expected):
        dim, signal = classify_failure_dimension(text)
        assert dim == expected
        assert signal

    def test_network_timeout_is_tool_interaction(self):
        dim, _ = classify_failure_dimension(
            "connection error: ECONNRESET while fetching url")
        assert dim == "tool_interaction"

    def test_llm_read_timeout_is_generation_control(self):
        dim, _ = classify_failure_dimension(
            "httpx.ReadTimeout while waiting for the completion tokens")
        assert dim == "generation_control"

    def test_empty_input(self):
        assert classify_failure_dimension("") == (DIM_UNKNOWN, "empty")
        assert classify_failure_dimension(None) == (DIM_UNKNOWN, "empty")

    def test_unmatched_input(self):
        dim, signal = classify_failure_dimension(
            "zzz qqq completely unrelated words")
        assert dim == DIM_UNKNOWN
        assert signal == "unclassified"

    def test_dimension_roster(self):
        assert len(DIMENSIONS) == 8
        assert DIM_UNKNOWN in DIMENSIONS
        assert "model" in DIMENSIONS


# ------------------------------------------------------------- env toggles

class TestEnvToggles:
    def test_defaults_on(self, monkeypatch):
        for var in ("GHOST_FAILURE_DIM", "GHOST_FAILURE_DISTILL",
                    "GHOST_FAILURE_ADJUDICATE"):
            monkeypatch.delenv(var, raising=False)
        assert failure_dim_enabled()
        assert distill_enabled()
        assert adjudicate_enabled()

    @pytest.mark.parametrize("value", ["0", "false", "no"])
    def test_kill_switches(self, monkeypatch, value):
        monkeypatch.setenv("GHOST_FAILURE_DIM", value)
        monkeypatch.setenv("GHOST_FAILURE_DISTILL", value)
        monkeypatch.setenv("GHOST_FAILURE_ADJUDICATE", value)
        assert not failure_dim_enabled()
        assert not distill_enabled()
        assert not adjudicate_enabled()

    def test_distill_max(self, monkeypatch):
        monkeypatch.delenv("GHOST_FAILURE_DISTILL_MAX", raising=False)
        assert distill_max() == 2
        monkeypatch.setenv("GHOST_FAILURE_DISTILL_MAX", "5")
        assert distill_max() == 5
        monkeypatch.setenv("GHOST_FAILURE_DISTILL_MAX", "-3")
        assert distill_max() == 0
        monkeypatch.setenv("GHOST_FAILURE_DISTILL_MAX", "bogus")
        assert distill_max() == 2


# ------------------------------------------------------------- adjudication

class TestAdjudicate:
    async def test_valid_label_wins(self):
        stub = _RouteStub(["memory"])
        result = await adjudicate_dimension(
            stub, "recalled something wrong", "unknown")
        assert result == "memory"
        task, payload, _ = stub.calls[0]
        assert task == "CLASSIFY_FAILURE"
        assert "HEURISTIC GUESS: unknown" in payload["messages"][1]["content"]

    async def test_label_normalization(self):
        stub = _RouteStub(['  "Memory" '])
        assert await adjudicate_dimension(stub, "text", "unknown") == "memory"

    async def test_garbage_falls_back_to_heuristic(self):
        stub = _RouteStub(["flibbertigibbet"])
        assert await adjudicate_dimension(
            stub, "text", "orchestration") == "orchestration"

    async def test_none_reply_falls_back(self):
        stub = _RouteStub([None])
        assert await adjudicate_dimension(stub, "text", "memory") == "memory"

    async def test_missing_client_falls_back(self):
        assert await adjudicate_dimension(None, "text", "memory") == "memory"

    async def test_invalid_heuristic_falls_back_to_unknown(self):
        stub = _RouteStub(["not-a-label"])
        assert await adjudicate_dimension(
            stub, "text", "made-up-dim") == DIM_UNKNOWN
