"""LLM-boundary record & stub-replay (core/llm_recording.py, 2026-07-17).

Recording is OFF by default (payloads carry unredacted memory/profile
text — enabling it is an explicit operator act per debugging session).
Replay is order-based with fingerprint verification: drifted harnesses
are served anyway but flagged, exhaustion fails loudly.
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.llm_recording import (
    LLMRecorder, ReplayLLMClient, elide_image_data, maybe_record,
    payload_fingerprint, recording_enabled,
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


class TestCrossRestartReplay:
    """Fix 2026-07-22: the per-process ordinal restarts at 1 while the day
    file appends across boots; a global ordinal sort interleaved sessions
    record-by-record. Records now carry a per-boot session_id and replay
    groups by it."""

    def test_records_carry_session_id(self, tmp_path):
        rec = LLMRecorder(root=tmp_path)
        assert rec.session_id  # minted at init, process-lifetime stable
        rec.record("chat_completion", _payload("a"), {})
        row = json.loads(
            next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
        assert row["session_id"] == rec.session_id

    async def test_day_file_spanning_two_boots_replays_contiguously(
            self, tmp_path):
        # Boot 1 records two calls; boot 2 (fresh recorder = fresh process)
        # appends to the SAME day file with its ordinal restarted at 1.
        boot1 = LLMRecorder(root=tmp_path)
        boot1.record("chat_completion", _payload("b1-c1"),
                     {"choices": [{"message": {"content": "b1-r1"}}]})
        boot1.record("chat_completion", _payload("b1-c2"),
                     {"choices": [{"message": {"content": "b1-r2"}}]})
        boot2 = LLMRecorder(root=tmp_path)
        assert boot2.session_id != boot1.session_id
        boot2.record("chat_completion", _payload("b2-c1"),
                     {"choices": [{"message": {"content": "b2-r1"}}]})
        day_file = next(tmp_path.glob("*.jsonl"))
        rows = [json.loads(l) for l in day_file.read_text().splitlines()]
        assert [r["ordinal"] for r in rows] == [1, 2, 1]  # the restart

        # Pre-fix, the stable ordinal sort served b2-r1 to boot 1's second
        # call. Now boot 1's records stay contiguous and in order.
        client = ReplayLLMClient(day_file)
        outs = [await client.chat_completion(_payload(p))
                for p in ("b1-c1", "b1-c2", "b2-c1")]
        assert [o["choices"][0]["message"]["content"] for o in outs] \
            == ["b1-r1", "b1-r2", "b2-r1"]
        assert client.mismatches == []

    async def test_legacy_records_without_session_id_keep_file_order(
            self, tmp_path):
        # A pre-fix multi-boot day file: no session_id, ordinals restart.
        # Backward compat = file order (never the interleaving sort).
        rows = []
        for n, (ordinal, text) in enumerate(
                [(1, "r1c1"), (2, "r1c2"), (1, "r2c1")], start=1):
            p = _payload(text)
            rows.append({"ordinal": ordinal, "kind": "chat_completion",
                         "fingerprint": payload_fingerprint(p),
                         "payload": p, "response": {"n": n}})
        day = tmp_path / "legacy.jsonl"
        day.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        client = ReplayLLMClient(day)
        served = [await client.chat_completion(_payload(t))
                  for t in ("r1c1", "r1c2", "r2c1")]
        assert [s["n"] for s in served] == [1, 2, 3]
        assert client.mismatches == []


class TestTornLineTolerance:
    """Fix 2026-07-22: a torn last line (file copied mid-append, or a
    crash) used to raise JSONDecodeError and kill the whole replay."""

    async def test_torn_last_line_skipped_not_fatal(self, tmp_path, caplog):
        rec = LLMRecorder(root=tmp_path)
        rec.record("chat_completion", _payload("ok"),
                   {"choices": [{"message": {"content": "fine"}}]})
        day = next(tmp_path.glob("*.jsonl"))
        with day.open("a", encoding="utf-8") as f:
            f.write('{"ts": "2026-07-22T00:00:00Z", "ordinal": 2, "resp')
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            client = ReplayLLMClient(day)
        assert client.skipped_lines == 1
        assert len(client.records) == 1
        assert any("skipped 1 unparseable" in r.getMessage()
                   for r in caplog.records)
        out = await client.chat_completion(_payload("ok"))
        assert out["choices"][0]["message"]["content"] == "fine"
        assert client.mismatches == []

    def test_clean_file_reports_zero_skipped(self, tmp_path):
        rec = LLMRecorder(root=tmp_path)
        rec.record("route", _payload("q"), "CONFIRMED")
        client = ReplayLLMClient(next(tmp_path.glob("*.jsonl")))
        assert client.skipped_lines == 0


class TestGhostHomeReResolution:
    """Fix 2026-07-22: GHOST_HOME was frozen into the recorder at first
    touch — unset-then-set left it silently poisoned for the process
    lifetime, contradicting the flip-without-restart claim."""

    def test_recorder_heals_when_ghost_home_appears_later(
            self, monkeypatch, tmp_path):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        rec = LLMRecorder()
        assert rec.root is None
        assert rec.record("chat_completion", _payload("early"), {}) is False
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        assert rec.record("chat_completion", _payload("late"), {}) is True
        files = list(
            (tmp_path / "system" / "llm_recordings").glob("*.jsonl"))
        assert len(files) == 1

    def test_explicit_root_still_wins_over_env(self, monkeypatch, tmp_path):
        env_home = tmp_path / "env-home"
        explicit = tmp_path / "explicit"
        monkeypatch.setenv("GHOST_HOME", str(env_home))
        rec = LLMRecorder(root=explicit)
        assert rec.record("chat_completion", _payload("x"), {}) is True
        assert list(explicit.glob("*.jsonl"))
        assert not list(env_home.rglob("*.jsonl"))

    def test_module_hook_singleton_not_poisoned(self, monkeypatch, tmp_path):
        import ghost_agent.core.llm_recording as mod
        monkeypatch.setattr(mod, "_recorder", None)
        monkeypatch.setenv("GHOST_LLM_RECORD", "1")
        monkeypatch.delenv("GHOST_HOME", raising=False)
        maybe_record("chat_completion", _payload("early"), {})  # no-op
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        maybe_record("chat_completion", _payload("late"), {"ok": True})
        files = list(
            (tmp_path / "system" / "llm_recordings").glob("*.jsonl"))
        assert len(files) == 1


class TestImageElision:
    """Improvement 2026-07-22: base64 image bodies are elided from stored
    payloads (size guard — NOT redaction; text is untouched). Fingerprints
    hash the original payload so replay matching still works."""

    def _vision_payload(self, url):
        return {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this screenshot?"},
                {"type": "image_url", "image_url": {"url": url}},
            ]}], "temperature": 0.1}

    def test_big_data_uri_elided_in_written_record(self, tmp_path):
        big = "data:image/png;base64," + "A" * 5000
        payload = self._vision_payload(big)
        rec = LLMRecorder(root=tmp_path)
        assert rec.record("chat_completion", payload, {"choices": []})
        row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
        stored = row["payload"]["messages"][0]["content"]
        assert stored[1]["image_url"]["url"].startswith(
            "data:image/png;base64,<elided 5000 chars sha256:")
        assert "A" * 100 not in stored[1]["image_url"]["url"]
        # Text content is untouched (no-redaction contract stands).
        assert stored[0]["text"] == "what is in this screenshot?"
        # Fingerprint was taken from the ORIGINAL payload.
        assert row["fingerprint"] == payload_fingerprint(payload)
        # Caller's payload was not mutated.
        assert payload["messages"][0]["content"][1]["image_url"]["url"] \
            == big

    async def test_replay_matches_live_payload_with_full_image(
            self, tmp_path):
        payload = self._vision_payload(
            "data:image/jpeg;base64," + "B" * 4096)
        rec = LLMRecorder(root=tmp_path)
        rec.record("chat_completion", payload,
                   {"choices": [{"message": {"content": "a screenshot"}}]})
        client = ReplayLLMClient(next(tmp_path.glob("*.jsonl")))
        out = await client.chat_completion(payload)  # full image, live-side
        assert out["choices"][0]["message"]["content"] == "a screenshot"
        assert client.mismatches == []

    def test_small_data_uri_kept_verbatim(self):
        small = "data:image/png;base64,AAAA"
        payload = self._vision_payload(small)
        out = elide_image_data(payload)
        assert out["messages"][0]["content"][1]["image_url"]["url"] == small

    def test_helper_is_noop_without_images(self):
        p = _payload("plain text only")
        assert elide_image_data(p) is p

    def test_anthropic_base64_source_shape_elided(self):
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": "C" * 2048}},
        ]}]}
        out = elide_image_data(payload)
        data = out["messages"][0]["content"][0]["source"]["data"]
        assert data.startswith("<elided 2048 chars sha256:")
        # original untouched
        assert payload["messages"][0]["content"][0]["source"]["data"] \
            == "C" * 2048
