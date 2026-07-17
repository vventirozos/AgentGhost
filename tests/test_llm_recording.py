"""LLM-boundary record & stub-replay (core/llm_recording.py, 2026-07-17).

Recording is OFF by default (payloads carry unredacted memory/profile
text — enabling it is an explicit operator act per debugging session).
Replay is order-based with fingerprint verification: drifted harnesses
are served anyway but flagged, exhaustion fails loudly.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.llm_recording import (
    LLMRecorder, ReplayLLMClient, maybe_record, payload_fingerprint,
    recording_enabled,
)


def _payload(text):
    return {"messages": [{"role": "user", "content": text}],
            "temperature": 0.6}


class TestRecorder:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("GHOST_LLM_RECORD", raising=False)
        assert not recording_enabled()

    def test_record_roundtrip(self, tmp_path):
        rec = LLMRecorder(root=tmp_path)
        assert rec.record("chat_completion", _payload("hi"),
                          {"choices": [{"message": {"content": "hello"}}]},
                          task_label="verify")
        assert rec.record("route", _payload("judge"), "CONFIRMED",
                          task="VERIFY")
        [day_file] = list(tmp_path.glob("*.jsonl"))
        rows = [json.loads(l) for l in day_file.read_text().splitlines()]
        assert [r["ordinal"] for r in rows] == [1, 2]
        assert rows[0]["kind"] == "chat_completion"
        assert rows[0]["meta"]["task_label"] == "verify"
        assert rows[1]["response"] == "CONFIRMED"
        assert rows[0]["fingerprint"] == payload_fingerprint(_payload("hi"))

    def test_no_ghost_home_is_safe_noop(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        rec = LLMRecorder()
        assert rec.root is None
        assert rec.record("chat_completion", _payload("x"), {}) is False

    def test_module_hook_respects_kill_switch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GHOST_LLM_RECORD", "0")
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        maybe_record("chat_completion", _payload("x"), {})
        assert not list(tmp_path.rglob("*.jsonl"))

    def test_fingerprint_ignores_sampling_params(self):
        a = {"messages": [{"role": "user", "content": "q"}],
             "temperature": 0.6}
        b = {"messages": [{"role": "user", "content": "q"}],
             "temperature": 1.0, "max_tokens": 99}
        assert payload_fingerprint(a) == payload_fingerprint(b)


class TestReplay:
    def _recording(self, tmp_path):
        rec = LLMRecorder(root=tmp_path)
        rec.record("chat_completion", _payload("first"),
                   {"choices": [{"message": {"content": "one"}}]})
        rec.record("route", _payload("second"), "two")
        return next(tmp_path.glob("*.jsonl"))

    async def test_replay_serves_in_order_from_file(self, tmp_path):
        client = ReplayLLMClient(self._recording(tmp_path))
        r1 = await client.chat_completion(_payload("first"))
        r2 = await client.route("VERIFY", _payload("second"), fallback=None)
        assert r1["choices"][0]["message"]["content"] == "one"
        assert r2 == "two"
        assert client.mismatches == []

    async def test_mismatch_flagged_not_fatal(self, tmp_path):
        client = ReplayLLMClient(self._recording(tmp_path))
        out = await client.chat_completion(_payload("DIFFERENT"))
        assert out["choices"][0]["message"]["content"] == "one"
        assert len(client.mismatches) == 1

    async def test_exhaustion_fails_loudly(self, tmp_path):
        client = ReplayLLMClient(self._recording(tmp_path))
        await client.chat_completion(_payload("first"))
        await client.route("VERIFY", _payload("second"))
        with pytest.raises(IndexError):
            await client.chat_completion(_payload("third"))


class TestWiring:
    def test_llm_client_records_both_branches_and_route(self):
        src = (Path(__file__).resolve().parents[1]
               / "src" / "ghost_agent" / "core" / "llm.py").read_text()
        assert src.count("_maybe_record_call(") >= 3  # fg + bg + route
        assert 'kind="route"' in src
