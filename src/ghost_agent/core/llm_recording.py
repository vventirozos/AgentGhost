# src/ghost_agent/core/llm_recording.py
"""LLM-boundary record & stub-replay (2026-07-17).

Recording: with ``GHOST_LLM_RECORD=1``, every ``LLMClient.chat_completion``
result and every ``route()`` verdict is appended as one JSONL line under
``$GHOST_HOME/system/llm_recordings/<YYYY-MM-DD>.jsonl`` — request id (from
the pretty-log context), a per-process ordinal, the exact payload, the
response, and the routing flags. That is the raw material for minting
regression tests from real failures: every fix this project has shipped
started with log archaeology and a hand-reconstructed fixture; a recording
replaces the reconstruction step with a copy.

**OFF by default, deliberately.** Payloads contain the rendered system
prompt — memory contents, the user profile — VERBATIM, bypassing the
corpus redaction contract. Enable for a debugging session, harvest, turn
it off. This module never redacts (a redacted prompt is not what was
sent, which defeats replay); retention is the operator's call.

Replay: ``ReplayLLMClient`` serves a recording back in order —
byte-exact, no model, fully deterministic. Calls are matched by a hash of
``payload["messages"]``; a mismatch is served anyway (order-based) but
flagged on ``mismatches`` so a drifted harness is visible, not silent.
This is stub-replay for harness-logic tests and forensics; it is NOT
live re-generation (llama.cpp on Metal with prefix-cache reuse is not
byte-stable run-to-run, so "re-run with the same seed" was rejected as a
goal — see PROJECT_JOURNAL 2026-07-17).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")


def recording_enabled() -> bool:
    return (os.getenv("GHOST_LLM_RECORD", "0").strip().lower()
            in ("1", "true", "yes"))


def _recordings_dir() -> Optional[Path]:
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        return None
    return Path(home) / "system" / "llm_recordings"


def payload_fingerprint(payload: Dict[str, Any]) -> str:
    """Stable hash of the conversational content — messages only, so
    sampling-param tweaks between record and replay don't break matching."""
    try:
        blob = json.dumps(payload.get("messages", []),
                          sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = str(payload.get("messages"))
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


class LLMRecorder:
    """Append-only JSONL sink. Thread-safe, never raises into a caller —
    a broken recording layer must never break an LLM call."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root if root is not None else _recordings_dir()
        self._lock = threading.Lock()
        self._ordinal = 0

    def record(self, kind: str, payload: Dict[str, Any], response: Any,
               **meta) -> bool:
        if self.root is None:
            return False
        try:
            with self._lock:
                self._ordinal += 1
                ordinal = self._ordinal
            try:
                from ..utils.logging import request_id_context
                req_id = request_id_context.get()
            except Exception:
                req_id = ""
            rec = {
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "ordinal": ordinal,
                "request_id": str(req_id or ""),
                "kind": kind,
                "fingerprint": payload_fingerprint(payload),
                "payload": payload,
                "response": response,
                "meta": {k: v for k, v in meta.items() if v},
            }
            day = datetime.date.today().isoformat()
            path = self.root / f"{day}.jsonl"
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return True
        except Exception as e:  # noqa: BLE001 — recording is best-effort
            logger.debug("llm recording skipped: %s: %s",
                         type(e).__name__, e)
            return False


_recorder: Optional[LLMRecorder] = None
_recorder_lock = threading.Lock()


def maybe_record(kind: str, payload: Dict[str, Any], response: Any,
                 **meta) -> None:
    """Module-level hook for LLMClient — resolves the enabled flag and the
    singleton lazily so the env can be flipped without a restart."""
    if not recording_enabled():
        return
    global _recorder
    with _recorder_lock:
        if _recorder is None:
            _recorder = LLMRecorder()
    _recorder.record(kind, payload, response, **meta)


class ReplayLLMClient:
    """Serve a recording back, in order, without a model.

    Duck-types the two LLMClient surfaces the harness exercises:
    ``chat_completion`` (returns the recorded response dict) and
    ``route`` (returns the recorded content / the caller's fallback).
    Exhaustion raises ``IndexError`` — a replay that needs more calls
    than were recorded IS a behavior change worth failing on.
    """

    def __init__(self, records):
        if isinstance(records, (str, Path)):
            loaded: List[dict] = []
            with Path(records).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        loaded.append(json.loads(line))
            records = loaded
        self.records: List[dict] = sorted(
            list(records), key=lambda r: r.get("ordinal", 0))
        self._cursor = 0
        self.mismatches: List[dict] = []

    def _next(self, kind: str, payload: Dict[str, Any]) -> dict:
        if self._cursor >= len(self.records):
            raise IndexError(
                f"replay exhausted after {len(self.records)} call(s) — "
                f"the harness now makes more LLM calls than the recording")
        rec = self.records[self._cursor]
        self._cursor += 1
        fp = payload_fingerprint(payload)
        if rec.get("fingerprint") != fp or rec.get("kind") != kind:
            self.mismatches.append({
                "ordinal": rec.get("ordinal"),
                "expected_kind": rec.get("kind"), "got_kind": kind,
                "expected_fp": rec.get("fingerprint"), "got_fp": fp,
            })
        return rec

    async def chat_completion(self, payload: Dict[str, Any],
                              **_kw) -> Dict[str, Any]:
        return self._next("chat_completion", payload).get("response") or {}

    async def route(self, task: str, payload: Dict[str, Any],
                    fallback: Any = None, **_kw) -> Any:
        resp = self._next("route", payload).get("response")
        return resp if resp is not None else fallback
