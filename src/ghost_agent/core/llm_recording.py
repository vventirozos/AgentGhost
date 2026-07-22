# src/ghost_agent/core/llm_recording.py
"""LLM-boundary record & stub-replay (2026-07-17).

Recording: with ``GHOST_LLM_RECORD=1``, every ``LLMClient.chat_completion``
result and every ``route()`` verdict is appended as one JSONL line under
``$GHOST_HOME/system/llm_recordings/<YYYY-MM-DD>.jsonl`` — request id (from
the pretty-log context), a per-boot session id, a per-process ordinal, the
exact payload (base64 image bodies elided — a size guard, not redaction),
the response, and the routing flags. That is the raw material for minting
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
import uuid
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


# Data-URI image bodies below this size are kept verbatim (tiny stubs are
# harmless and occasionally informative); anything larger is bulk that
# balloons a screenshot-heavy day file into GBs.
_IMAGE_ELIDE_MIN = 256


def _elide_marker(data: str) -> str:
    digest = hashlib.sha256(data.encode("utf-8", "replace")).hexdigest()[:12]
    return f"<elided {len(data)} chars sha256:{digest}>"


def _elide_image_part(part: Any) -> Any:
    """Return ``part`` with any large base64 image body replaced by a short
    marker, or the SAME object when nothing needs eliding (copy-on-write)."""
    if not isinstance(part, dict):
        return part
    # OpenAI vision shape: {"type": "image_url", "image_url": {"url": "data:..."}}
    img = part.get("image_url")
    if isinstance(img, dict):
        url = img.get("url")
        if (isinstance(url, str) and url.startswith("data:")
                and len(url) >= _IMAGE_ELIDE_MIN):
            head, _, body = url.partition(",")
            new_img = dict(img)
            new_img["url"] = f"{head},{_elide_marker(body)}"
            new_part = dict(part)
            new_part["image_url"] = new_img
            return new_part
    # Flat variant: {"type": "image_url", "image_url": "data:..."}
    elif (isinstance(img, str) and img.startswith("data:")
          and len(img) >= _IMAGE_ELIDE_MIN):
        head, _, body = img.partition(",")
        new_part = dict(part)
        new_part["image_url"] = f"{head},{_elide_marker(body)}"
        return new_part
    # Anthropic shape: {"type": "image", "source": {"type": "base64", "data": ...}}
    src = part.get("source")
    if (isinstance(src, dict) and src.get("type") == "base64"
            and isinstance(src.get("data"), str)
            and len(src["data"]) >= _IMAGE_ELIDE_MIN):
        new_src = dict(src)
        new_src["data"] = _elide_marker(src["data"])
        new_part = dict(part)
        new_part["source"] = new_src
        return new_part
    return part


def elide_image_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``payload`` with base64 image bodies (data-URIs in
    vision message parts) replaced by ``<elided N chars sha256:...>``.

    This is a SIZE guard, not redaction — text content is untouched (the
    no-redaction contract in the module docstring stands), and it does not
    affect replay matching because ``record()`` fingerprints the ORIGINAL
    payload before eliding. Returns the original object unchanged when
    there is nothing to elide; never mutates the caller's payload."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    changed = False
    new_messages: List[Any] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_parts: List[Any] = []
        msg_changed = False
        for part in content:
            new_part = _elide_image_part(part)
            if new_part is not part:
                msg_changed = True
            new_parts.append(new_part)
        if msg_changed:
            msg = dict(msg)
            msg["content"] = new_parts
            changed = True
        new_messages.append(msg)
    if not changed:
        return payload
    out = dict(payload)
    out["messages"] = new_messages
    return out


class LLMRecorder:
    """Append-only JSONL sink. Thread-safe, never raises into a caller —
    a broken recording layer must never break an LLM call.

    Every record is stamped with ``session_id`` — a per-boot token (pid +
    uuid4, minted once at recorder init). The per-process ``ordinal``
    restarts at 1 on every boot while the day file appends across boots,
    so ordinals alone cannot order a multi-restart day; the session id is
    what lets ``ReplayLLMClient`` keep each boot's records contiguous."""

    def __init__(self, root: Optional[Path] = None):
        self._explicit_root = root
        self.session_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._lock = threading.Lock()
        self._ordinal = 0

    @property
    def root(self) -> Optional[Path]:
        """Explicit root if one was passed in, else GHOST_HOME re-resolved
        on every touch — a recorder first touched before GHOST_HOME was
        exported heals once it appears instead of staying poisoned (the
        'env can be flipped without a restart' claim)."""
        if self._explicit_root is not None:
            return self._explicit_root
        return _recordings_dir()

    def record(self, kind: str, payload: Dict[str, Any], response: Any,
               **meta) -> bool:
        root = self.root
        if root is None:
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
                "session_id": self.session_id,
                "ordinal": ordinal,
                "request_id": str(req_id or ""),
                "kind": kind,
                # Fingerprint the ORIGINAL payload, store the elided one:
                # replay matching hashes messages whole, so the stored
                # fingerprint must match what a live caller will send.
                "fingerprint": payload_fingerprint(payload),
                "payload": elide_image_data(payload),
                "response": response,
                "meta": {k: v for k, v in meta.items() if v},
            }
            day = datetime.date.today().isoformat()
            path = root / f"{day}.jsonl"
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                root.mkdir(parents=True, exist_ok=True)
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
        self.skipped_lines = 0
        if isinstance(records, (str, Path)):
            loaded: List[dict] = []
            with Path(records).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded.append(json.loads(line))
                    except ValueError:
                        # Torn line — copied mid-append or truncated by a
                        # crash. Skip it; the rest of the day is still good.
                        self.skipped_lines += 1
            if self.skipped_lines:
                logger.warning(
                    "llm replay: skipped %d unparseable line(s) in %s "
                    "(torn/partial writes)", self.skipped_lines, records)
            records = loaded
        # Group by recording session (one id per boot) so a day file that
        # spans agent restarts replays each boot contiguously — a global
        # ordinal sort would interleave boots record-by-record, since the
        # per-process ordinal restarts at 1 on every boot. Legacy records
        # without a session_id keep file order (for a single-boot file
        # that IS ordinal order; for a multi-boot legacy file it is the
        # only ordering that keeps boots contiguous).
        groups: Dict[str, List[dict]] = {}
        for rec in records:
            groups.setdefault(str(rec.get("session_id") or ""), []).append(rec)
        for sid, grp in groups.items():
            if sid:
                grp.sort(key=lambda r: r.get("ordinal", 0))
        self.records: List[dict] = [r for grp in groups.values() for r in grp]
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
