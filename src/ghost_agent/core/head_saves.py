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

§4UD (review 11-11): a file is kept only when it is whole and the running
server's (`save`: free space first, ``n_written`` equal to the file's size,
llama's identity read again before the rename); any slot action that succeeds
lifts the "saves off" mark; the directory is pruned by age as well as by count
and bytes, another server's saves leave by age instead of at once, and files
are owner-only. This file runs standalone at llama start (`python -I`): it
imports the standard library only.

Off unless GHOST_HEAD_SAVES=1. The directory is $GHOST_HOME/system/head_saves
and MUST be llama-server's --slot-save-path (llama only takes a file name).
One slot (-np 1): slot id 0.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
#: 10-27: conversations also have a byte budget — the count alone let 6 files
#: grow with the conversation (196k context: ~10 GB each near the end); the
#: newest conversation is always kept
CONV_BYTES_MAX = 24 << 30
SLOT_ID = 0
#: r1: short — llama defers a restore while the slot is busy; giving up cancels it.
#: 10-26: 10 s timed out 3× while the Mac was short of memory (normal reads
#: 0.2-0.9 s of the 1.7 GB file, slow ones 5-9 s); each time-out cost an 18-22 s
#: re-prefill for a restore that was nearly done. 30 s (operator).
RESTORE_TIMEOUT_S = 30.0
#: §4RW review: seconds allowed per GB of a saved state (10-26: 1.7 GB took 5-9 s under memory pressure)
RESTORE_S_PER_GB = 10.0
SAVE_TIMEOUT_S = 60.0
#: §4SA: a save's bound grows with the state like the restore's — a flat 60 s
#: can time a multi-GB write out under memory pressure, and the .part llama
#: goes on writing is never kept. ~35 KB of KV per token (§4RL) plus the
#: checkpoints (--ctx-checkpoints 12 × ~150 MiB)
STATE_BYTES_PER_TOKEN = 35_000
CHECKPOINTS_BYTES = 12 * 150 * 1024 * 1024
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

#: §4UD (H8): a save made on ANOTHER model, build or template is kept
#: this long after its last use — trying another model once must not delete the
#: usual model's saves (it did: llama start dropped every foreign file at once)
FOREIGN_KEEP_S = 3 * 86400.0
#: §4UD (H14): a save not used (saved or restored) for this long is dropped,
#: per kind — the directory was bounded by count and bytes only, so a members'
#: head for a prompt that no longer exists never left
MAX_AGE_S = {OWNER: 30 * 86400.0, MEMBER: 14 * 86400.0, CONV: 14 * 86400.0}
#: §4UD (H4): free space a save needs beyond the state's own estimated size
FREE_MARGIN_BYTES = 2 << 30
#: §4UD (H14): the states hold the owner's tokens — owner-only files and directory
FILE_MODE = 0o600
DIR_MODE = 0o700

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


def restore_timed_out(detail: Any) -> bool:
    """§4UD r2: did a POSTED restore end in a client timeout? Read from the
    detail `restore` returned — `_error_text` above writes it, and this is the
    one place that reads it back (llama may still be applying such a restore)."""
    t = str(detail or "").lower()
    return "timeout" in t or "timed out" in t


def _note_unsupported(base: str, e: Exception, text: str) -> None:
    if isinstance(e, urllib.error.HTTPError) and "does not support slots action" in text:
        _unsupported.add(base)


def _note_supported(base: str) -> None:
    """§4UD (H7): llama just answered a slot action — it takes them, whatever
    an earlier 501 said (the mark was lifted only by an idle tick's probe, so
    a working owner kept "saves off" for the whole session)."""
    _unsupported.discard(base)


def secure_dir(d: Path) -> None:
    """§4UD (H14): the save directory is the owner's alone (0700). Best
    effort: a directory another user owns is left as it is."""
    try:
        if (d.stat().st_mode & 0o777) != DIR_MODE:
            os.chmod(d, DIR_MODE)
    except OSError:
        pass


def _secure_file(p: Path) -> None:
    try:
        if (p.stat().st_mode & 0o777) != FILE_MODE:
            os.chmod(p, FILE_MODE)
    except OSError:
        pass


def free_bytes(d: Path) -> Optional[int]:
    """Free bytes on the save directory's volume; None when unknown."""
    try:
        return int(shutil.disk_usage(d).free)
    except Exception:  # noqa: BLE001
        return None


def state_bytes(n_tokens: int) -> int:
    """The estimated size of a saved slot holding ``n_tokens`` (§4SA's figures)."""
    try:
        n = max(0, int(n_tokens or 0))
    except (TypeError, ValueError):
        n = 0
    return n * STATE_BYTES_PER_TOKEN + CHECKPOINTS_BYTES


def recheck_supported(base: str, timeout: float = 3.0) -> bool:
    """§4TJ: True when a server marked unsupported now takes slot actions (the
    mark is lifted). Req c6751cb9: one llama without --slot-save-path at the
    agent's boot left saves off through two llama restarts WITH the flag, and a
    picture look cost a re-read of 80k tokens. An empty filename saves nothing:
    without the flag llama answers 501 before it reads the name, with it 400
    "Invalid filename" at once, even while the slot is busy (measured live). No
    answer says nothing, so the mark stays."""
    if base not in _unsupported:
        return False
    try:
        _http(f"{base}/slots/{SLOT_ID}?action=save", {"filename": ""}, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            return False
    except Exception:  # noqa: BLE001
        return False
    _unsupported.discard(base)
    return True


def server_identity(base: str) -> str:
    """What the RUNNING server says it is, read from its /props now: model
    file path, build and chat template — part of every file key and stored in
    every meta, so a model, build or template change never restores old state.

    A value held in a plan is only what the server WAS when the plan was made
    (§4UD, H2: llama can restart on another model between a plan and a save).
    `save` therefore reads it again after llama answered and before the file
    is kept, and the agent's restore reads it again before it posts.

    Limit (§4UD r2): the model is known by its PATH. A model file replaced in
    place under the same path is not detected. (Round 1 added the file's size
    and mtime, read from the path: that is the file on disk, not the one llama
    has open, so a replaced file filed the old model's state under the new
    one's identity — worse than not looking. Removed; the format is the one
    every existing file was saved under.)"""
    p = _http(base + "/props")
    tmpl = hashlib.sha1(str(p.get("chat_template") or "").encode("utf-8", "ignore")).hexdigest()[:16]
    return f"{p.get('model_path') or ''}|{p.get('build_info') or ''}|{tmpl}"


def confirmed_identity(base: str) -> Tuple[Optional[str], str]:
    """§4UD r3: the running server's identity, asked TWICE before "no answer"
    — ``(identity, "")``, or ``(None, why)``. The one rule for both sides: the
    save's check after llama answered and the agent's check before a restore
    (it asked once there, so one slow /props cost a whole restore)."""
    err = ""
    for _try in (1, 2):
        try:
            return (server_identity(base), "")
        except Exception as e:  # noqa: BLE001
            err = _error_text(e)
    return (None, err)


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


def conv_key(identity: str, session: str, level: Optional[str] = None) -> str:
    """§4RV: one file per conversation (its slot-session key) per server.
    §4SA review: and per effort LEVEL when known — the effort sentence is in
    the head, so a state built at one level cannot serve a prompt at another;
    one file per conversation made every level flip overwrite the other
    level's state (A→B→A re-read the whole conversation). A level-less key is
    the plain one (files saved before §4SA, an unknown server level)."""
    blob = identity + "\nconversation\n" + str(session or "") + (f"\n@{level}" if level else "")
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


def discard(d: Path, key: str, keep_part: bool = False) -> None:
    """``keep_part``: prune's use — a .part may be a re-save of this key that
    llama is writing right now (review r3); the stale-.part sweep takes old ones."""
    for name in (_bin(key), _meta(key)) + (() if keep_part else (_part(key),)):
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


def _count(r: Any, field: str) -> int:
    """A count from llama's reply; 0 when the reply is not the JSON object
    expected (review r4: a string or list reply raised out of save/restore,
    and the llama-start helper died with a traceback)."""
    try:
        return int((r if isinstance(r, dict) else {}).get(field) or 0)
    except (TypeError, ValueError):
        return 0


def restore_timeout_s(d: Path, key: str) -> float:
    """§4RW review: the restore's bound grows with the file — 30 s was sized
    for the 1.7 GB head; a 6 GB conversation read under swap passed it, and
    the head then went over the conversation llama had just loaded."""
    try:
        gb = (d / _bin(key)).stat().st_size / 1e9
    except Exception:  # noqa: BLE001
        return RESTORE_TIMEOUT_S
    return max(RESTORE_TIMEOUT_S, RESTORE_S_PER_GB * gb)


def save_timeout_s(n_tokens: int) -> float:
    """§4SA: seconds a save of a slot holding ``n_tokens`` may take —
    max(SAVE_TIMEOUT_S, RESTORE_S_PER_GB per estimated GB)."""
    try:
        n = max(0, int(n_tokens or 0))
    except (TypeError, ValueError):
        n = 0
    gb = state_bytes(n) / 1e9
    return max(SAVE_TIMEOUT_S, RESTORE_S_PER_GB * gb)


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
    if not isinstance(r, dict):
        # review r5: an odd reply says nothing about the file — keep it
        return (False, f"llama answered {type(r).__name__}, not the restore result", dt)
    _note_supported(base)       # §4UD (H7): llama takes slot actions
    n = _count(r, "n_restored")
    if n != expected:
        discard(d, key)
        return (False, f"restored {n} tokens, saved head has {expected}", dt)
    touch(d, key)       # §4RW-e: every restore is a use (prune keeps by last use)
    return (True, f"{n} tokens", dt)


def _drop(part: Path) -> None:
    try:
        part.unlink()
    except Exception:  # noqa: BLE001
        pass


def save(base: str, d: Path, key: str, expected_n: int, identity: str = "",
         kind: str = OWNER, timeout: float = SAVE_TIMEOUT_S,
         at_least: bool = False, extra: Optional[dict] = None) -> Tuple[bool, str]:
    """Save the slot as this key's head when it holds exactly ``expected_n``
    tokens (the warm-up prompt). llama writes a .part name; only a verified
    save is renamed into place, so a save that caught a live conversation
    (the slot moved on before llama ran it) never becomes the head.
    §4RV: ``kind=CONV`` saves a conversation the same way; ``at_least``:
    any count from ``expected_n`` up is the state wanted (llama's /slots
    count and its saved count differ by the generated tokens).

    §4UD — what "verified" means, all checked HERE (the one place a file is kept):
    H4: the volume must hold the estimated state plus `FREE_MARGIN_BYTES`
    before anything is posted; llama must report ``n_written`` > 0 and the
    .part must be exactly that long — an empty or short file (a full disk)
    never replaces a good one. H2: ``identity`` is what the caller's key was
    built from; the running server's is read again after llama answered, and
    a different one drops the .part (llama restarted on another model between
    the caller's plan and this save — the state is that model's). When /props
    does not answer (asked twice) the .part is dropped too (r2): a state whose
    server could not be confirmed is not kept."""
    if expected_n <= 0:
        return (False, "no prompt token count")
    prune(d)            # review 10-27: a run of failed saves left .part files behind
    need, free = state_bytes(expected_n) + FREE_MARGIN_BYTES, free_bytes(d)
    if free is not None and free < need:
        return (False, f"the disk has {free / 1e9:.1f} GB free, the state needs about "
                       f"{need / 1e9:.1f} GB with its margin — nothing posted")
    try:
        r = _http(f"{base}/slots/{SLOT_ID}?action=save", {"filename": _part(key)},
                  timeout=timeout)
    except Exception as e:  # noqa: BLE001
        text = _error_text(e)
        _note_unsupported(base, e, text)
        return (False, text)
    n = _count(r, "n_saved")
    if isinstance(r, dict):
        _note_supported(base)       # §4UD (H7): llama takes slot actions
    r = r if isinstance(r, dict) else {}
    part = d / _part(key)
    if n < expected_n if at_least else n != expected_n:
        _drop(part)
        return (False, f"slot held {n} tokens, expected {expected_n}")
    if not part.is_file():
        return (False, f"llama saved outside {d} — its --slot-save-path must be this directory")
    written = _count(r, "n_written")
    try:
        size = part.stat().st_size
    except OSError:
        size = -1
    if written <= 0 or size != written:
        _drop(part)
        return (False, f"llama reported {written} bytes written and the file holds {max(size, 0)} — "
                       f"not kept (a full disk?); the previous save, if any, stays")
    if identity:
        running, err = confirmed_identity(base)
        if running is None:
            _drop(part)
            return (False, f"llama's /props did not answer after the save ({err}) — the state's server "
                           f"could not be confirmed, not kept")
        if running != identity:
            _drop(part)
            return (False, "llama now runs another model, build or template than the one "
                           "this save was planned for — not kept")
    _secure_file(part)
    try:
        os.replace(part, d / _bin(key))
    except FileNotFoundError:            # the llama-start helper's prune took it meanwhile
        return (False, "the saved file was removed before it was kept")
    except OSError as e:
        return (False, f"the saved file was not kept: {type(e).__name__}: {e}")
    meta = {"n_tokens": n, "bytes": written,
            "save_ms": (r.get("timings") if isinstance(r.get("timings"), dict) else {}).get("save_ms"),
            "identity": identity,
            "saved_at": time.time(), "kind": kind, **(extra or {})}
    tmp = d / (_meta(key) + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(meta))
        os.replace(tmp, d / _meta(key))
    except (OSError, TypeError, ValueError) as e:
        # review r3 (a full disk): the new state beside the OLD meta would
        # restore as a token mismatch — drop both, report, never raise
        # (§4UD: a meta field json cannot write is the same loss, not a raise)
        discard(d, key)
        try:
            tmp.unlink()
        except OSError:
            pass
        return (False, f"meta not written: {type(e).__name__}: {e}")
    # §4UD (H8): the identity given to prune was read from the running server
    # just above and found equal — never a caller's plan identity unchecked
    # (fresh-eye review 10-27: read before a llama restart on another model, it
    # marked the running server's files as the foreign ones)
    prune(d, identity=identity)
    return (True, f"{n} tokens, {written / 1e9:.2f} GB")


def _last_use(p: Path) -> float:
    # §4RW review: saved (.bin) or used (`touch` on the .json) — by the
    # save time alone a restored conversation counted as old and went first
    try:
        return max(p.stat().st_mtime, p.with_suffix(".json").stat().st_mtime)
    except Exception:  # noqa: BLE001
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0


def prune(d: Path, keep: int = KEEP, identity: str = "") -> None:
    """Keep the ``keep`` newest heads of each kind (§4RU: owner effort
    changes never push out the members' head; §4RV: conversations keep
    ``KEEP_CONV`` within ``CONV_BYTES_MAX``); drop stale .part files. Touches
    only this module's own file names. No llama call: stats and the small
    meta files only, so it can run on every idle tick.

    §4UD (H14): every save also has an AGE cap by kind (`MAX_AGE_S`, counted
    from its last save or restore), and files and the directory are made
    owner-only.

    Given the running server's ``identity`` — only one just read from its
    /props, never a cached plan's — a save made on another model, build
    or template is FOREIGN. §4UD (H8): a foreign save is no longer
    dropped at once (10-27 did that, and trying another model once deleted
    every save of the usual one); it is kept `FOREIGN_KEEP_S` after its last
    use, and it ranks BEHIND every save of the running server in its kind's
    count and in the conversations' byte budget — so the bounds of the
    directory are the same as before, and the running server's saves are never
    the ones pushed out by another model's. A meta that names no identity (a
    save from before identities were stored) is not foreign.
    A meta whose state file is gone and a stale ``.json.tmp`` are dropped too.
    One file vanishing under us (the agent and the llama-start helper both
    prune) skips that file only."""
    try:
        now = time.time()
        secure_dir(d)
        for m in list(d.glob("head-*.json")) + list(d.glob("head-*.json.tmp")):
            try:
                orphan = m.name.endswith(".json") and not (d / (m.name[:-len(".json")] + ".bin")).exists()
                if (orphan or m.name.endswith(".tmp")) and now - m.stat().st_mtime > STALE_PART_S:
                    m.unlink()
            except Exception:  # noqa: BLE001
                continue

        def _foreign(p: Path) -> bool:
            if not identity:
                return False
            other = meta_field(d, p.name[len("head-"):-len(".bin")], "identity")
            return bool(other) and other != identity
        ranked = sorted(((not _foreign(p), _last_use(p), p) for p in d.glob("head-*.bin")
                         if not p.name.endswith(".part.bin")),
                        key=lambda t: (t[0], t[1]), reverse=True)
        kept: dict = {}
        conv_seen: set = set()  # §4SA r2: conversations (the meta's `conv`), not their per-level files
        conv_bytes = 0          # of the conversations kept so far, newest first
        for ours, used_at, p in ranked:
            if _last_use(p) != used_at:
                # review r4: saved or used again since the ranking (a re-save
                # by another thread or process) — the next prune judges it
                continue
            try:
                key = p.name[len("head-"):-len(".bin")]
                size = p.stat().st_size
                if not (d / _meta(key)).exists():
                    # review r2: a state file without its meta can never be
                    # restored, yet counted as an OWNER head (kind_of's
                    # default) — a fresh one is a save between its two writes
                    if now - p.stat().st_mtime > STALE_PART_S:
                        discard(d, key, keep_part=True)
                    continue
                kind = kind_of(d, key)
            except OSError:
                continue
            if now - used_at > (MAX_AGE_S.get(kind, MAX_AGE_S[OWNER]) if ours else FOREIGN_KEEP_S):
                discard(d, key, keep_part=True)       # §4UD: H14 (age), H8 (another server's)
                continue
            n = kept.get(kind, 0)
            # §4SA review r2: conversation files are per conversation AND effort
            # level — the count is of conversations (the meta's `conv`, the
            # plain key; a pre-§4SA file's key IS that plain key, so it groups
            # with the conversation's level files), else an effort-switching
            # owner's rotation shrank to two conversations
            cid = (str(meta_field(d, key, "conv") or key)) if kind == CONV else None
            same_conv = kind == CONV and cid in conv_seen
            if ((n >= (KEEP_CONV if kind == CONV else keep) and not same_conv)
                    or (kind == CONV and n >= 1 and conv_bytes + size > CONV_BYTES_MAX)):
                discard(d, key, keep_part=True)   # a dropped big one leaves room for an older, smaller one
                continue
            _secure_file(p)
            _secure_file(d / _meta(key))
            if kind == CONV:
                if not same_conv:
                    conv_seen.add(cid)
                    kept[kind] = n + 1
                conv_bytes += size
            else:
                kept[kind] = n + 1
        for p in d.glob("head-*.part.bin"):
            try:
                if now - p.stat().st_mtime > STALE_PART_S:
                    p.unlink()
            except OSError:
                continue
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
    identity (§4RU: llama-start puts the owner's back, never the members').
    By SAVE time, not last use (operator 10-27: "after a llama restart the
    most recently saved head should be loaded not the most recently used")."""
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
    prune(d, identity=ident)        # §4UD: another server's saves leave by age (FOREIGN_KEEP_S), not at once
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
