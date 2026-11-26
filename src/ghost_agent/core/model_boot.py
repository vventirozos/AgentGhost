"""§4VM: line the agent up with the model the main server actually runs.

One step, ``sync``, run at boot (``main.py``) and again from every head
warm-up (``GhostAgent._model_resync``, at most once a minute), so a llama
restart on another model — or a profile file the operator just added — is
picked up without restarting the agent:

1. read the profile files again if the directory changed;
2. ask the server which model file it runs and how large its window is
   (``/props``: read-only);
3. choose the main model's profile from that (``model_profile.choose_main``);
4. the tokenizer: the server's own, built from the model file and checked id
   for id against the server (``utils/gguf_tokenizer``); when that cannot be
   had (a server on another machine, an unknown vocabulary), the profile's;
5. check the token counter against the server's own count
   (``token_counter.calibrate``);
6. fit ``args.max_context`` into the server's window
   (``model_profile.context_budget``): the operator's ``--max-context`` is
   kept as the ceiling and lowered only when the window cannot hold it with
   room for a reply; ``--max-context 0`` takes the budget from the server.

A server that does not answer AT BOOT leaves the profile to ``--model``, the
profile's tokenizer still loads and the flag stands. A RE-SYNC changes
something only when the server answered with a model file — one that does not
answer, or answers without naming a model, changes nothing — and a re-sync
that fails half-way puts the counting back as it was and is tried again.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import model_profile as _mp
from .head_saves import server_base
from ..utils import gguf_tokenizer as _gguf
from ..utils import token_counter as _tc

logger = logging.getLogger("GhostAgent")

#: The budget when ``--max-context 0`` (from the server) and no server answers.
FALLBACK_CONTEXT = 65536

#: A re-sync asks the server at most this often (the re-warm loop ticks every
#: 10–120 s; one ``/props`` read costs a millisecond, but it need not be every tick).
RESYNC_EVERY_S = 60.0

_lock = threading.Lock()
#: one sync at a time: a second caller (a slow server outliving the throttle) gets the last summary
_running = threading.Lock()
_FRESH = {"booted": False, "system_dir": None, "flag": None, "profile": None, "seen": {},
          "files": None, "calibrated": False, "build_again": False, "asked_at": 0.0, "last": {}}
_state: Dict[str, Any] = dict(_FRESH)


def booted() -> bool:
    """``sync`` ran at boot in this process. The warm-up re-sync is a no-op
    otherwise (a test's agent, a script: no boot, no server to ask)."""
    return bool(_state["booted"])


def last() -> Dict[str, Any]:
    """The summary of the state in force (health, the log): what the last
    sync that finished left behind."""
    return dict(_state["last"])


def reset() -> None:
    """Forget the boot (tests)."""
    with _lock:
        _state.clear()
        _state.update(_FRESH)


def _load_counting(profile, model_path: str, sysdir: Optional[Path], base: str, timeout: float) -> str:
    """Step 4. Returns where the tokenizer came from: "model file", "profile" or "none"."""
    enc = _gguf.from_model_file(model_path, sysdir, base, timeout) if model_path else None
    if enc is not None:
        _tc.set_encoder(enc)
        _tc.reset_calibration()
        return "model file"
    loaded = None
    if sysdir is not None:
        loaded = _tc.load_tokenizer(_tc.tokenizer_path(sysdir, profile), repo=profile.tokenizer_repo,
                                    profile_name=profile.name)
    if loaded is None:
        _tc.set_encoder(None)                       # never the previous model's tokenizer
    _tc.reset_calibration()
    return "profile" if loaded is not None else "none"


def sync(args, system_dir: Optional[Path] = None, *, boot: bool = False, timeout: float = 3.0,
         now: Optional[float] = None) -> Dict[str, Any]:
    """Run the steps above. ``boot=True`` is the first call: it records the
    operator's ``--max-context`` and always loads a tokenizer.

    A later call (a RE-SYNC) needs a boot before it and asks the server at
    most every ``RESYNC_EVERY_S``. It does the full work when the server named
    another model file, the profile files changed, or the last attempt did
    not finish; it refits the budget when only the window changed; it tries
    the token check again when the last one got no answer. A server that is
    down, loading, or answering without a model name changes nothing.

    Returns the summary of the state in force; never raises."""
    now = time.monotonic() if now is None else now
    with _lock:
        if not boot and not _state["booted"]:
            return {}
        if not boot and now - _state["asked_at"] < RESYNC_EVERY_S:
            return dict(_state["last"])
    if not _running.acquire(blocking=boot):
        return last()
    try:
        return _sync(args, system_dir, boot, timeout, now)
    finally:
        _running.release()


def _sync(args, system_dir, boot: bool, timeout: float, now: float) -> Dict[str, Any]:
    with _lock:
        _state["asked_at"] = now
        if boot:
            _state["system_dir"] = Path(system_dir) if system_dir is not None else None
            _state["flag"] = max(0, int(getattr(args, "max_context", 0) or 0))
        sysdir, flag, before, seen_before, files_before, calibrated, build_again = (
            _state["system_dir"], _state["flag"], _state["profile"], _state["seen"],
            _state["files"], _state["calibrated"], _state["build_again"])
    out: Dict[str, Any] = dict(_state["last"]) if not boot else {}
    counting, why_before = _tc.counting_state(), _mp.main_why()
    profile = before or _mp.main()
    seen: Dict[str, Any] = {}
    done = False
    try:
        _mp.reload_if_changed()
        files = _mp.files_signature()
        base = server_base(getattr(args, "upstream_url", ""))
        seen = _mp.probe_server(base, timeout) if base else {}
        if not boot and not seen:
            return last()                               # silent, or it named no model: nothing to decide
        # `build_again`: last time the model file's tokenizer could not be CHECKED (the
        # server's /tokenize did not answer) — the fallback is in use; the build is owed
        model_changed = (boot or seen.get("model_path") != seen_before.get("model_path")
                         or files != files_before or build_again)
        if model_changed:
            profile, why, sure = _mp.choose_main(str(getattr(args, "model", "") or ""), seen.get("model_path", ""))
            source = _load_counting(profile, seen.get("model_path", ""), sysdir, base, timeout)
            build_again = source != "model file" and _gguf.last_outcome == _gguf.SERVER_SILENT
            _mp.announce_main(profile, why, sure)
            out = {"profile": profile.name, "why": why, "tokenizer": source,
                   "server_model": os.path.basename(seen.get("model_path", ""))}
            calibrated = False
        if seen and not calibrated:
            out["tokens"] = _tc.calibrate(base, timeout)
            calibrated = bool(out["tokens"])
        # the window: a server that gave none (0) leaves the budget where it is
        n_ctx = seen.get("n_ctx", 0)
        if n_ctx or boot or model_changed:
            budget = _mp.context_budget(flag, n_ctx, profile) or FALLBACK_CONTEXT
            if flag and budget < flag and budget != int(getattr(args, "max_context", 0) or 0):
                logger.warning(
                    "context: --max-context %d does not fit the server's %d-token window with "
                    "%d left for a reply — using %d", flag, n_ctx, min(profile.reply_reserve, n_ctx // 4), budget)
            args.max_context = budget
            out.update(n_ctx=n_ctx, max_context=budget, max_context_flag=flag)
        done = True
    except Exception as exc:  # noqa: BLE001 — never stop a boot or a warm-up on this
        logger.warning("model sync failed (%s: %s) — %s", type(exc).__name__, exc,
                       "booting on the default profile" if boot else "everything stays as it was; tried again in a minute")
        if boot:
            # a boot must end with a profile, a tokenizer when one can be had, and a budget
            profile = _mp.main()
            try:
                if _tc.TOKEN_ENCODER is None and sysdir is not None:
                    _tc.load_tokenizer(_tc.tokenizer_path(sysdir, profile), repo=profile.tokenizer_repo,
                                       profile_name=profile.name)
            except Exception:  # noqa: BLE001
                pass
            if not int(getattr(args, "max_context", 0) or 0):
                args.max_context = FALLBACK_CONTEXT
            out = {"profile": profile.name, "why": _mp.main_why() or "default",
                   "max_context": int(args.max_context), "max_context_flag": flag}
        else:
            _tc.restore_counting(counting)              # the old profile keeps the old tokenizer and scale
            if before is not None and _mp.main() is not before:
                _mp.set_main(before, why_before)
            profile = before or _mp.main()
            out = dict(_state["last"])
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    with _lock:
        if boot:
            _state["booted"] = True
        if done:
            # only a sync that FINISHED is remembered: a failed one is tried again
            _state["seen"], _state["files"], _state["calibrated"] = dict(seen), files, calibrated
            _state["build_again"] = build_again
            out.pop("error", None)
        _state["profile"], _state["last"] = profile, dict(out)
    return dict(out)


def describe(summary: Dict[str, Any]) -> str:
    """One line for the boot banner."""
    if not summary or "profile" not in summary:
        return "model profile: unknown"
    tok = summary.get("tokens") or {}
    if tok.get("exact") or summary.get("tokenizer") == "model file":
        count = "tokens counted with the model file's own tokenizer (matches the server)"
    elif tok.get("local_tokens") and tok.get("encoder_scale") == 1.0:
        count = f"tokenizer within {abs(tok.get('drift') or 0.0) * 100:.1f}% of the server's count"
    elif tok.get("local_tokens"):
        count = f"tokenizer is NOT the server's (counts ×{tok.get('encoder_scale')})"
    elif tok:
        count = f"tokens estimated at {tok.get('chars_per_token')} chars each"
    else:
        count = "token counts not checked against the server"
    return f"model profile: {summary['profile']} ({summary.get('why') or 'default'}); {count}"
