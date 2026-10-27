"""§4RO: the byte-stable request head parked on disk.

The 27B re-processes the 28k-token head in ~166 s whenever llama-server has
lost it (a llama restart, or a call that displaced it while the RAM cache
could not hold it). llama-server's slot save/restore (`--slot-save-path`)
writes the slot's whole state — attention KV, the hybrid model's recurrent
state, the context checkpoints and the MTP draft — to one file. Measured on
build b11669 (2026-10-25): save 0.68 s / 1.73 GB, restore 0.41 s, and the
next request reused 28,213 of 28,219 tokens; a request that diverged 5
tokens before the end still reused 28,209 (the restored checkpoints work).

The agent saves the head right after a warm-up that had to prefill it, and
restores it in place of every re-warm prefill (r3), judged at dispatch: a
freshly started server is restored straight away; a slot that holds exactly the
head is left alone; anything else is PARKED first (a 1-token prompt that shares
nothing makes llama save the slot's state to its RAM cache, as the old re-warm
prefill did) and then restored — nothing is lost that the old prefill kept. The
park and the restore happen only while no request is live and the slot is idle,
checked again before each (r4).

Correctness never depends on this module: llama compares exact tokens, so a
stale file only shortens the reused prefix, and the warm-up that follows a
restore re-prefills (and re-saves) whatever did not match.

§4RU: two kinds of head, each its own file — the owner's (tools + system
prompt + profile) and the channel members' (the 6 allowed tools + the system
prompt with the member placeholder). The meta file names the kind; llama-start
restores only the owner's, and pruning keeps each kind's newest separately.

Off unless GHOST_HEAD_SAVES=1. The directory is $GHOST_HOME/system/head_saves
and MUST be llama-server's --slot-save-path (llama only takes a file name).
One slot (-np 1): slot id 0.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Tuple

ENV = "GHOST_HEAD_SAVES"
#: newest head files kept per kind (a profile or effort change makes a new key; ~1.7 GB each)
KEEP = 3
#: §4RU: the head kinds; a meta without one is the owner's (saved before §4RU)
OWNER = "owner"
MEMBER = "member"
#: §4RV: an owner conversation saved before another call displaced it (its key
#: is `conv_key`; llama-start never restores one, `newest_head_for` reads the kind)
CONV = "conv"
#: §4RV: conversation files kept (each ~2-4 GB: the KV of the whole conversation
#: plus up to --ctx-checkpoints checkpoints of ~150 MB)
KEEP_CONV = 6     # §4RW review: 3 lost a 4-conversation rotation
SLOT_ID = 0
#: r1: short — llama defers a restore while the slot is busy; giving up cancels it.
#: 10-26: 10 s timed out 3× while the Mac was short of memory (normal reads
#: 0.2-0.9 s of the 1.7 GB file, slow ones 5-9 s); each time-out cost an 18-22 s
#: re-prefill for a restore that was nearly done. 30 s (operator).
RESTORE_TIMEOUT_S = 30.0
#: §4RW review: seconds allowed per GB of a saved state (10-26: 1.7 GB took 5-9 s under memory pressure)
RESTORE_S_PER_GB = 10.0
SAVE_TIMEOUT_S = 60.0
#: a .part file older than this is a save that never finished
STALE_PART_S = 3600.0
#: r3: a prompt that shares no token with the slot: llama selects the slot by LRU
#: and parks what it holds in its RAM cache (`get_available_slot` → prompt_save)
#: before it processes this, as the old re-warm prefill did. r4: llama reads that
#: entry back only for a request that matches the slot poorly (f_keep < 0.5); an
#: owner turn matches the restored head, so a finished conversation's follow-up
#: re-processes its own tail — the same as after the old prefill
PARK_PROMPT = "."
#: r5: llama copies the slot's state into its RAM cache before it runs the park
#: prompt — 28 s for a 43k-token conversation (2.2 GB) on a box in swap, past
#: the old 30 s, after which the agent took the park for failed and prefilled
PARK_TIMEOUT_S = 120.0
#: the park prompt's size in the slot once it ran ("." is 1 token; headroom
#: for a BOS) — far below any conversation or the head
PARK_MAX_TOKENS = 4

#: servers that answered "does not support slots action" (no --slot-save-path)
_unsupported: set = set()


def enabled() -> bool:
    return (os.getenv(ENV) or "").strip() == "1"


def head_dir() -> Optional[Path]:
    home = (os.getenv("GHOST_HOME") or "").strip()
    return Path(home) / "system" / "head_saves" if home else None


def server_base(upstream_url: Any) -> str:
    return re.sub(r"/v1(/.*)?$", "", str(upstream_url or "").strip().rstrip("/"))


def supported(base: str) -> bool:
    return bool(base) and base not in _unsupported


def _http(url: str, body: Optional[dict] = None, timeout: float = 5.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    return json.loads(raw) if raw.lstrip().startswith(("{", "[")) else raw


def _error_text(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            return e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
    return f"{type(e).__name__}: {e}"


def _note_unsupported(base: str, e: Exception, text: str) -> None:
    if isinstance(e, urllib.error.HTTPError) and "does not support slots action" in text:
        _unsupported.add(base)


def server_identity(base: str) -> str:
    """Model file, build and chat template of the running server — part of the
    file key, so a model, build or template change never restores old state."""
    p = _http(base + "/props")
    tmpl = hashlib.sha1(str(p.get("chat_template") or "").encode("utf-8", "ignore")).hexdigest()[:16]
    return f"{p.get('model_path') or ''}|{p.get('build_info') or ''}|{tmpl}"


def render(base: str, payload: dict) -> str:
    """The prompt text llama renders for this payload (``/apply-template``:
    text only, it does not touch the slot or the cache)."""
    body = {k: payload[k] for k in ("messages", "tools", "tool_choice", "parallel_tool_calls",
                                    "chat_template_kwargs") if k in payload}
    prompt = (_http(base + "/apply-template", body, timeout=10.0) or {}).get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("no rendered prompt")
    return prompt


def head_key(identity: str, rendered: str) -> str:
    """Live r1: keyed on the RENDERED head, not the payload bytes — a payload
    that differed only in a field the template does not render (the same
    effort level sent explicitly) made a second, identical 1.7 GB file and
    missed the restore after a llama restart."""
    blob = identity + "\n" + rendered
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:16]


def conv_key(identity: str, session: str) -> str:
    """§4RV: one file per conversation (its slot-session key) per server."""
    blob = identity + "\nconversation\n" + str(session or "")
    return "c" + hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:15]


def _bin(key: str) -> str:
    return f"head-{key}.bin"


def _part(key: str) -> str:
    return f"head-{key}.part.bin"


def _meta(key: str) -> str:
    return f"head-{key}.json"


def stored_tokens(d: Path, key: str) -> Optional[int]:
    """The verified token count of the saved head, None when there is no
    complete save (both the state file and its meta) for this key."""
    try:
        if not (d / _bin(key)).is_file():
            return None
        n = int(json.loads((d / _meta(key)).read_text()).get("n_tokens") or 0)
        return n if n > 0 else None
    except Exception:  # noqa: BLE001
        return None


def meta_field(d: Path, key: str, name: str):
    """§4RW-f: one field of a save's meta (None when absent or unreadable)."""
    try:
        return json.loads((d / _meta(key)).read_text()).get(name)
    except Exception:  # noqa: BLE001
        return None


def kind_of(d: Path, key: str) -> str:
    """§4RU: the saved head's kind (owner when the meta does not say)."""
    try:
        return str(json.loads((d / _meta(key)).read_text()).get("kind") or OWNER)
    except Exception:  # noqa: BLE001
        return OWNER


def touch(d: Path, key: str) -> None:
    """§4RU-r: mark a saved head as used now (its meta's mtime)."""
    try:
        os.utime(d / _meta(key))
    except Exception:  # noqa: BLE001
        pass


def has_kind(d: Path, kind: str, within_s: Optional[float] = None) -> bool:
    """§4RU: a complete save of this kind exists (any key) — saved or used
    (`touch`) within ``within_s`` seconds when given."""
    now = time.time()
    for m in d.glob("head-*.json"):
        try:
            key = m.name[len("head-"):-len(".json")]
            meta = json.loads(m.read_text())
            if (stored_tokens(d, key) is None or str(meta.get("kind") or OWNER) != kind
                    or (within_s is not None
                        and now - max(float(meta.get("saved_at") or 0), m.stat().st_mtime) > within_s)):
                continue
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def discard(d: Path, key: str) -> None:
    for name in (_bin(key), _meta(key), _part(key)):
        try:
            (d / name).unlink()
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass


def server_fresh(base: str) -> bool:
    """llama-server has served nothing since it started: its slot and RAM
    cache are empty, so the head is certainly gone."""
    try:
        text = _http(base + "/metrics", timeout=3.0)
        vals = {}
        for line in str(text).splitlines():
            for name in ("prompt_tokens_total", "prompt_tokens_cached_total", "tokens_predicted_total"):
                if line.startswith(f"llamacpp:{name} "):
                    vals[name] = float(line.split()[1])
        return len(vals) == 3 and not any(vals.values())
    except Exception:  # noqa: BLE001
        return False


def slot_busy(base: str) -> bool:
    """r4: llama is processing a task on the slot now (True when unknown) — a
    restore posted then is held back until a turn's call ends and lands in
    its tool gap, replacing its conversation."""
    try:
        return bool((_http(base + "/slots", timeout=3.0) or [{}])[SLOT_ID].get("is_processing", True))
    except Exception:  # noqa: BLE001
        return True


def slot_tokens(base: str) -> Optional[int]:
    """How many tokens the slot holds now (llama's /slots ``n_prompt_tokens``:
    absent until the first completion since llama started); None when unknown."""
    try:
        v = (_http(base + "/slots", timeout=3.0) or [{}])[SLOT_ID].get("n_prompt_tokens")
        return int(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def park_landed(base: str) -> bool:
    """r5: after a park whose reply never came (a timeout, a dropped
    connection): True when the slot is idle and holds only the park prompt —
    llama parked the state before the client gave up. Unknown → False."""
    try:
        s = (_http(base + "/slots", timeout=3.0) or [{}])[SLOT_ID]
        n = s.get("n_prompt_tokens")
        return (not s.get("is_processing", True)) and n is not None and int(n) <= PARK_MAX_TOKENS
    except Exception:  # noqa: BLE001
        return False


def park(base: str, timeout: float = PARK_TIMEOUT_S) -> Tuple[bool, str]:
    """r3: make llama park the slot's state in its RAM cache (a conversation
    someone may resume) by sending a 1-token prompt that shares nothing with
    it. A state larger than --cache-ram is not parked (llama logs "exceeds
    cache size limit") — the same loss as the re-warm prefill had."""
    try:
        _http(base + "/completion", {"prompt": PARK_PROMPT, "n_predict": 1, "temperature": 0.0,
                                     "cache_prompt": True}, timeout=timeout)
        return (True, "parked")
    except Exception as e:  # noqa: BLE001
        return (False, _error_text(e))


def restore_timeout_s(d: Path, key: str) -> float:
    """§4RW review: the restore's bound grows with the file — 30 s was sized
    for the 1.7 GB head; a 6 GB conversation read under swap passed it, and
    the head then went over the conversation llama had just loaded."""
    try:
        gb = (d / _bin(key)).stat().st_size / 1e9
    except Exception:  # noqa: BLE001
        return RESTORE_TIMEOUT_S
    return max(RESTORE_TIMEOUT_S, RESTORE_S_PER_GB * gb)


def restore(base: str, d: Path, key: str) -> Tuple[bool, str, float]:
    """(restored, detail, seconds). A restore whose token count is not the
    saved one is discarded with its file."""
    expected = stored_tokens(d, key)
    if expected is None:
        return (False, "no saved head", 0.0)
    t0 = time.monotonic()
    try:
        r = _http(f"{base}/slots/{SLOT_ID}?action=restore", {"filename": _bin(key)},
                  timeout=restore_timeout_s(d, key))
    except Exception as e:  # noqa: BLE001
        # r1: the file stays — llama's "Unable to restore" also covers a full
        # KV cache, and the warm-up that follows re-prefills and overwrites it
        text = _error_text(e)
        _note_unsupported(base, e, text)
        return (False, text, time.monotonic() - t0)
    dt = time.monotonic() - t0
    n = int((r or {}).get("n_restored") or 0)
    if n != expected:
        discard(d, key)
        return (False, f"restored {n} tokens, saved head has {expected}", dt)
    touch(d, key)       # §4RW-e: every restore is a use (prune keeps by last use)
    return (True, f"{n} tokens", dt)


def save(base: str, d: Path, key: str, expected_n: int, identity: str = "",
         kind: str = OWNER, timeout: float = SAVE_TIMEOUT_S,
         at_least: bool = False, extra: Optional[dict] = None) -> Tuple[bool, str]:
    """Save the slot as this key's head when it holds exactly ``expected_n``
    tokens (the warm-up prompt). llama writes a .part name; only a verified
    save is renamed into place, so a save that caught a live conversation
    (the slot moved on before llama ran it) never becomes the head.
    §4RV: ``kind=CONV`` saves a conversation the same way; ``at_least``:
    any count from ``expected_n`` up is the state wanted (llama's /slots
    count and its saved count differ by the generated tokens)."""
    if expected_n <= 0:
        return (False, "no prompt token count")
    prune(d)            # review 10-27: a run of failed saves left .part files behind
    try:
        r = _http(f"{base}/slots/{SLOT_ID}?action=save", {"filename": _part(key)},
                  timeout=timeout)
    except Exception as e:  # noqa: BLE001
        text = _error_text(e)
        _note_unsupported(base, e, text)
        return (False, text)
    n = int((r or {}).get("n_saved") or 0)
    part = d / _part(key)
    if n < expected_n if at_least else n != expected_n:
        try:
            part.unlink()
        except Exception:  # noqa: BLE001
            pass
        return (False, f"slot held {n} tokens, expected {expected_n}")
    if not part.is_file():
        return (False, f"llama saved outside {d} — its --slot-save-path must be this directory")
    os.replace(part, d / _bin(key))
    meta = {"n_tokens": n, "bytes": int((r or {}).get("n_written") or 0),
            "save_ms": ((r or {}).get("timings") or {}).get("save_ms"),
            "identity": identity, "saved_at": time.time(), "kind": kind, **(extra or {})}
    tmp = d / (_meta(key) + ".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, d / _meta(key))
    prune(d)
    return (True, f"{n} tokens, {meta['bytes'] / 1e9:.2f} GB")


def prune(d: Path, keep: int = KEEP) -> None:
    """Keep the ``keep`` newest heads of each kind (§4RU: owner effort
    changes never push out the members' head; §4RV: conversations keep
    ``KEEP_CONV``); drop stale .part files. Touches only this module's own
    file names."""
    try:
        def _last_use(p):
            # §4RW review: saved (.bin) or used (`touch` on the .json) — by the
            # save time alone a restored conversation counted as old and went first
            try:
                return max(p.stat().st_mtime, p.with_suffix(".json").stat().st_mtime)
            except Exception:  # noqa: BLE001
                return p.stat().st_mtime
        heads = sorted((p for p in d.glob("head-*.bin")
                        if not p.name.endswith(".part.bin")),
                       key=_last_use, reverse=True)
        seen: dict = {}
        for p in heads:
            key = p.name[len("head-"):-len(".bin")]
            kind = kind_of(d, key)
            seen[kind] = seen.get(kind, 0) + 1
            if seen[kind] > (KEEP_CONV if kind == CONV else keep):
                discard(d, key)
        now = time.time()
        for p in d.glob("head-*.part.bin"):
            if now - p.stat().st_mtime > STALE_PART_S:
                p.unlink()
    except Exception:  # noqa: BLE001
        pass


# ── at llama start (§4RO r2) ────────────────────────────────────────────
# The agent restores the head only at its next re-warm tick (≤180 s) after a
# llama restart, and anything that reaches llama first makes it non-fresh: the
# head is then re-processed (~166 s). The llama launcher runs this file in the
# background instead (`python -I head_saves.py --restore-on-start …`), so the
# head is back ~1 s after llama listens — on every start, the agent up or not.

def newest_head_for(d: Path, identity: str, kind: str = OWNER) -> Optional[str]:
    """Key of the most recently saved head of ``kind`` made on this server
    identity (§4RU: llama-start puts the owner's back, never the members')."""
    best_key: Optional[str] = None
    best_at = float("-inf")
    for meta in d.glob("head-*.json"):
        try:
            m = json.loads(meta.read_text())
            key = meta.name[len("head-"):-len(".json")]
            if (m.get("identity") != identity or str(m.get("kind") or OWNER) != kind
                    or stored_tokens(d, key) is None):
                continue
            at = float(m.get("saved_at") or 0)
            if at > best_at:
                best_key, best_at = key, at
        except Exception:  # noqa: BLE001
            continue
    return best_key


def wait_ready(base: str, wait_s: float, sleep=time.sleep) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            _http(base + "/health", timeout=3.0)
            return True
        except Exception:  # noqa: BLE001 — 503 while the model loads
            sleep(1.0)
    return False


def restore_on_start(base: str, d: Path, wait_s: float = 600.0, sleep=time.sleep) -> str:
    """One line for llama's log. Restores only while llama has served nothing
    (the agent's rule: a restore replaces the slot without parking it)."""
    if not wait_ready(base, wait_s, sleep):
        return f"llama not ready after {wait_s:.0f} s — nothing restored"
    try:
        ident = server_identity(base)
    except Exception as e:  # noqa: BLE001
        return f"cannot read /props ({_error_text(e)}) — nothing restored"
    key = newest_head_for(d, ident)
    if key is None:
        return "no saved head for this model/build/template — the agent's warm-up will process it"
    if not server_fresh(base):
        return "llama already served a request — not restoring over it"
    ok, detail, dt = restore(base, d, key)
    if ok:
        return f"restored head-{key} ({detail}) in {dt:.1f} s"
    return f"restore of head-{key} failed: {detail}"


def _main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="§4RO: restore the saved request head at llama start")
    ap.add_argument("--restore-on-start", action="store_true", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8088")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--wait", type=float, default=600.0)
    a = ap.parse_args(argv)
    print(f"head-restore: {restore_on_start(a.base.rstrip('/'), Path(a.dir), a.wait)}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
