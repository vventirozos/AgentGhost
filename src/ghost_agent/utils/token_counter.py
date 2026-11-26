import hashlib
import logging
import os
import threading
from pathlib import Path
from functools import lru_cache

# Imported on first use, not at module scope: `transformers` pulls in torch,
# sklearn, scipy and pandas (~300 MB per process), and every module that
# counts tokens imports this one. Only load_tokenizer needs it; the agent
# calls that once at boot. Tests replace this attribute directly.
AutoTokenizer = None


def _auto_tokenizer():
    global AutoTokenizer
    if AutoTokenizer is None:
        from transformers import AutoTokenizer as _cls
        AutoTokenizer = _cls
    return AutoTokenizer

# Tokenizer source for token-budget accounting. §4VM: the repo is the MAIN
# model's profile's (`core/model_profile.py`, `tokenizer_repo`); the built-in
# Qwen profile names a repo that ships the `tokenizer.json` of the Qwen3
# family (shared by the 35B-A3B and 27B derivatives — base, instruct and
# finetunes). A profile with no tokenizer, or a failed load, leaves
# `estimate_tokens` on a characters-per-token estimate: `len(text) // 4`
# until `calibrate` has asked the server how its tokenizer counts.
from ..core import model_profile as _mp

QWEN_MODEL_ID = _mp.QWEN38.tokenizer_repo          # the built-in default's, for readers and tests
TOKEN_ENCODER = None

# §4VM — what `calibrate` measured against the running server:
#   _CHARS_PER_TOKEN  the estimate when no tokenizer is loaded (4.0 = the old heuristic);
#   _ENCODER_SCALE    multiplies the loaded tokenizer's count when it is NOT the
#                     server's (1.0 = it is, counts are exact and untouched).
# The Inkling 2 Mini trial (2026-11-22) ran with the Qwen tokenizer counting a
# vocabulary that needs ~45 % more tokens for the same text; nothing said so.
_CHARS_PER_TOKEN = 4.0
_ENCODER_SCALE = 1.0
_calibration: dict = {}

# §4VM: bumped by every clear. The tokenizer or its scale can change in another
# thread (a model re-sync) while a count is being computed; that count is
# returned but not cached, so the cache never holds the old model's number.
_CACHE_GENERATION = 0

# One-shot warning gate: the first time we silently fall back to the
# `len(text) // 4` heuristic, we want a single visible WARNING log so the
# operator notices the tokenizer never loaded. Without this gate every
# token-budgeting decision in the agent runs on rough char/4 estimates
# with zero indication, causing weird context-window math.
_FALLBACK_WARNED = False
_logger = logging.getLogger("GhostAgent")


def _warn_fallback_once(reason: str = "tokenizer not loaded"):
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        _logger.warning(
            "estimate_tokens: falling back to a characters-per-token estimate "
            "(%.2f chars per token; len(text)//4 until the server was asked) (%s). "
            "Token-budget math will be approximate. Check tokenizer load logs "
            "above for the root cause.",
            _CHARS_PER_TOKEN, reason,
        )


def _reset_fallback_warning():
    """Re-arm the one-shot warning. Called by `load_tokenizer` on success
    and exposed for tests."""
    global _FALLBACK_WARNED
    _FALLBACK_WARNED = False


def clear_token_cache():
    """Drop the bounded token-count cache. Safe to call after a tokenizer
    becomes available so previously-cached approximate counts get recomputed
    accurately. Kept as a public symbol for backward compatibility — the
    underlying cache implementation has changed (was an unbounded-key LRU
    over full text, which acted as a memory sinkhole for long prompts)."""
    global _CACHE_GENERATION
    try:
        with _SHORT_TOKEN_CACHE_LOCK:
            _CACHE_GENERATION += 1      # §4VM: a count being computed now is not stored afterwards
            _SHORT_TOKEN_CACHE.clear()
    except Exception:
        pass

def tokenizer_path(system_dir: Path, profile=None) -> Path:
    """§4VM: where the main model's tokenizer is cached under
    ``$GHOST_HOME/system`` (the Qwen profile keeps the old ``tokenizer``)."""
    sub = (profile or _mp.main()).tokenizer_dir or "tokenizer"
    return Path(system_dir) / sub


def set_encoder(encoder) -> None:
    """§4VM: count with ``encoder`` from now on (None = estimate). The cache
    is dropped: its counts were the previous tokenizer's."""
    global TOKEN_ENCODER
    TOKEN_ENCODER = encoder
    if encoder is not None:
        _reset_fallback_warning()
    clear_token_cache()


def counting_state() -> tuple:
    """§4VM: everything a count depends on, to put back with ``restore_counting``
    when a change of model fails half-way."""
    return (TOKEN_ENCODER, _CHARS_PER_TOKEN, _ENCODER_SCALE, _calibration)


def restore_counting(state: tuple) -> None:
    global TOKEN_ENCODER, _CHARS_PER_TOKEN, _ENCODER_SCALE, _calibration
    TOKEN_ENCODER, _CHARS_PER_TOKEN, _ENCODER_SCALE, _calibration = state
    clear_token_cache()


def load_tokenizer(local_tokenizer_path: Path, repo=None, profile_name=None):
    """
    Robust loading strategy: LOCAL DISK -> TOR NETWORK -> FALLBACK

    §4VM: ``repo`` defaults to the main model's profile's tokenizer; a profile
    with none loads nothing (tokens are then estimated, see ``calibrate``).
    """
    global TOKEN_ENCODER
    repo = _mp.main().tokenizer_repo if repo is None else repo
    if not repo:
        TOKEN_ENCODER = None
        clear_token_cache()
        print(f"📐 No tokenizer for model profile '{profile_name or _mp.main().name}' — token counts are estimated")
        return None
    # Import transformers (and huggingface_hub) BEFORE the offline override
    # below: the library freezes HF_HUB_OFFLINE at its first import, and since
    # the import became lazy (2026-10-26) that import would otherwise happen
    # under the temporary "1" — offline for the whole process even under
    # --no-mandatory-tor (review: the embedder's first download then fails).
    _auto_tokenizer()
    # 1. Try Local Disk (Offline Mode) - PREFERRED
    if local_tokenizer_path.exists() and (local_tokenizer_path / "tokenizer.json").exists():
        # Restore (not pop) the prior value afterwards: --mandatory-tor sets
        # HF_HUB_OFFLINE=1 deliberately (see _env.py) and an operator may
        # have set it too — unconditionally deleting it would re-enable
        # cleartext HF model-resolution calls the egress guard then blocks.
        _prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            print(f"📂 Loading Tokenizer from local cache: {local_tokenizer_path}")
            TOKEN_ENCODER = _auto_tokenizer().from_pretrained(str(local_tokenizer_path), local_files_only=True)
            # Tokenizer became available — re-arm the warn-once gate AND
            # drop the LRU cache so previously-approximated counts are
            # recomputed accurately on next access.
            _reset_fallback_warning()
            clear_token_cache()
            return TOKEN_ENCODER
        except Exception as e:
            print(f"⚠️ Local tokenizer corrupted: {e}")
        finally:
            if _prev_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev_hf_offline

    # 2. Try Network Download (Direct Mode) - FALLBACK
    print(f"⏳ Local missing. Downloading {repo} via Direct Network...")

    import threading
    import queue

    def _download_hf_tokenizer(q):
        import huggingface_hub
        original_timeout = getattr(huggingface_hub.constants, "HF_HUB_DOWNLOAD_TIMEOUT", 10)
        huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 10
        try:
            enc = _auto_tokenizer().from_pretrained(repo)
            huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = original_timeout
            q.put(("SUCCESS", enc))
        except Exception as err:
            huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = original_timeout
            q.put(("ERROR", err))

    q = queue.Queue()
    t = threading.Thread(target=_download_hf_tokenizer, args=(q,), daemon=True)
    t.start()

    try:
        status, result = q.get(timeout=15.0)
        if status == "SUCCESS":
            TOKEN_ENCODER = result
            _reset_fallback_warning()
            clear_token_cache()
        else:
            print(f"❌ Network download failed (Thread Error): {result}")
            return None

        # Save it immediately so we never have to download again
        print(f"💾 Caching tokenizer to {local_tokenizer_path}...")
        local_tokenizer_path.mkdir(parents=True, exist_ok=True)
        TOKEN_ENCODER.save_pretrained(str(local_tokenizer_path))
        return TOKEN_ENCODER

    except queue.Empty:
        print(f"❌ Network download failed: Hard 15s Timeout Reached. HuggingFace might be blocked (daemon dropped).")
        return None

# Tiny bounded cache keyed on (sha1-prefix, length) instead of the full
# text. The previous implementation was `@lru_cache(maxsize=10240)` keyed on
# the full string — every distinct prompt up to 10K chars stayed pinned in
# memory by the LRU, turning a long-running daemon into a slow sinkhole.
# We cap entries at 1024 and drop the oldest on overflow.
_SHORT_TOKEN_CACHE: dict = {}
_SHORT_TOKEN_CACHE_MAX = 1024
# estimate_tokens is called from the event-loop thread and from work run
# under asyncio.to_thread; the eviction path iterates the dict, so an
# unlocked concurrent insert can raise "dict changed size during iteration".
_SHORT_TOKEN_CACHE_LOCK = threading.Lock()


def _short_cache_key(text: str) -> tuple:
    # Hash prefix + length is cheap and effectively collision-free for the
    # purpose of caching small token counts. Collisions only cost accuracy
    # on the rare colliding pair, never correctness elsewhere.
    digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]
    return (digest, len(text))


# Above this count, log the oversized payload ourselves (durable log).
# transformers' own "Token indices sequence length is longer than…" warning
# escaped ONLY to raw stderr (2026-07-25 audit: a 194308-token estimate
# left no trace in either log) — and at this layer it is noise anyway (we
# only COUNT; no model forward pass happens). We silence it per-call and
# emit the actually-useful signal through our own logger instead.
_HUGE_ESTIMATE_TOKENS = 131072


def _encoder_count(text: str) -> int:
    if TOKEN_ENCODER:
        try:
            # verbose=False suppresses the sequence-length warning that
            # bypassed our logging (it goes through transformers' own
            # stderr handler).
            try:
                n = len(TOKEN_ENCODER.encode(text, verbose=False))
            except TypeError:  # tokenizer without verbose kwarg
                n = len(TOKEN_ENCODER.encode(text))
            if _ENCODER_SCALE != 1.0:       # §4VM: not the server's tokenizer — its measured ratio
                n = int(n * _ENCODER_SCALE + 0.5)
            if n > _HUGE_ESTIMATE_TOKENS:
                _logger.info(
                    "estimate_tokens: %d-token payload exceeds the model "
                    "window (%d) — whatever built this string is oversized "
                    "(head: %.80s…)", n, _HUGE_ESTIMATE_TOKENS, text[:80])
            return n
        except Exception as e:
            _warn_fallback_once(f"encoder error: {type(e).__name__}")
            return _chars_estimate(text)
    if _mp.main().tokenizer_repo:       # the profile HAS a tokenizer and it is not loaded: say so
        _warn_fallback_once("tokenizer not loaded")
    return _chars_estimate(text)


def _chars_estimate(text: str) -> int:
    if _CHARS_PER_TOKEN == 4.0:
        return len(text) // 4
    return int(len(text) / _CHARS_PER_TOKEN)


#: §4VM: what a calibration may set. A server's answer outside these is not a
#: measurement (a stub's one token for the whole sample would make every count
#: 0 and nothing would ever be pruned; 200,000 tokens would prune everything).
CHARS_PER_TOKEN_RANGE = (1.0, 12.0)
ENCODER_SCALE_RANGE = (0.5, 2.0)


#: §4VM: what `calibrate` has the server tokenize — the kinds of text a request
#: is made of (prose, a tool schema, code, a list with paths and numbers), in
#: roughly the head's proportions. Fixed, so two calibrations are comparable.
CALIBRATION_SAMPLE = (
    "You are a careful assistant. Read the request, decide which tool answers it, and say plainly "
    "what you found. When a step fails, say so and try the next most likely cause; never report "
    "work you did not do. Keep replies short unless the owner asks for detail.\n\n"
    '{"type": "function", "function": {"name": "file_system", "description": "Read, write, edit or '
    'list files in the workspace.", "parameters": {"type": "object", "properties": {"operation": '
    '{"type": "string", "enum": ["read", "write", "edit", "list", "delete"]}, "path": {"type": '
    '"string", "description": "Path relative to the workspace root."}, "content": {"type": "string"}, '
    '"max_bytes": {"type": "integer", "minimum": 1}}, "required": ["operation", "path"]}}}\n\n'
    "def load_rows(path: str, limit: int = 100) -> list[dict]:\n"
    '    """Rows of a CSV file as dicts; at most `limit`."""\n'
    "    rows = []\n"
    "    with open(path, newline='', encoding='utf-8') as fh:\n"
    "        for i, row in enumerate(csv.DictReader(fh)):\n"
    "            if i >= limit:\n"
    "                break\n"
    "            rows.append({k.strip(): v for k, v in row.items()})\n"
    "    return rows\n\n"
    "- 2026-03-14 09:26:53 | /srv/app/logs/worker-3.log | 1,204 lines | exit code 0\n"
    "- http://127.0.0.1:8088/v1/chat/completions -> 200 in 3.42 s (cache_n=28213, n_ctx=196608)\n"
    "- TODO: re-run `pytest -q tests/test_rows.py::test_limit` after the fix; 3 of 17 still fail.\n"
)

_BAD_CALIBRATION_SAID = False

#: A loaded tokenizer whose count is within this share of the server's is left
#: alone. NOT "identical": measured 2026-11-26 on the live Qwen3.8 finetune, the
#: Qwen3 tokenizer counted 2.6 % under the server on this sample (0.6 % on a
#: prompt, 4 % on code, 16 % on indented JSON — the two split text differently
#: where a line ends). `calibration()["drift"]` reports the figure either way.
CALIBRATION_SAME = 0.03


def _server_token_count(base_url: str, text: str, timeout: float) -> int:
    import json
    import urllib.request
    req = urllib.request.Request(str(base_url).rstrip("/") + "/tokenize",
                                 data=json.dumps({"content": text}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        toks = (json.loads(r.read().decode("utf-8", "replace")) or {}).get("tokens")
    if not isinstance(toks, list) or not toks:
        raise ValueError("no tokens in the /tokenize answer")
    return len(toks)


def _local_sample_count() -> int:
    if not TOKEN_ENCODER:
        return 0
    try:
        try:
            return len(TOKEN_ENCODER.encode(CALIBRATION_SAMPLE, verbose=False))
        except TypeError:
            return len(TOKEN_ENCODER.encode(CALIBRATION_SAMPLE))
    except Exception:  # noqa: BLE001
        return 0


def calibrate(base_url: str, timeout: float = 3.0) -> dict:
    """§4VM: ask the running server how many tokens ``CALIBRATION_SAMPLE`` is
    (``/tokenize``: text only, no slot is touched) and set the estimate to it.

    * no tokenizer loaded → ``_CHARS_PER_TOKEN`` becomes the server's ratio;
    * a tokenizer that counts the sample within ``CALIBRATION_SAME`` of the
      server → nothing changes (the measured drift is reported);
    * a tokenizer that does not → its counts are scaled by the measured ratio
      and one WARNING says the tokenizer is not this model's.

    Returns what it found (also ``calibration()``); ``{}`` and no change when
    the server does not answer, or answers a count no tokenizer gives
    (``CHARS_PER_TOKEN_RANGE``, ``ENCODER_SCALE_RANGE``) — said once."""
    global _CHARS_PER_TOKEN, _ENCODER_SCALE
    try:
        n_server = _server_token_count(base_url, CALIBRATION_SAMPLE, timeout)
    except Exception:  # noqa: BLE001 — a server that is down leaves the estimate as it was
        return {}
    n_local = _local_sample_count()
    chars, scale = len(CALIBRATION_SAMPLE) / n_server, 1.0
    if n_local and abs(n_local - n_server) > CALIBRATION_SAME * n_server:
        scale = n_server / n_local
    lo, hi = CHARS_PER_TOKEN_RANGE
    slo, shi = ENCODER_SCALE_RANGE
    if not lo <= chars <= hi or not slo <= scale <= shi:
        global _BAD_CALIBRATION_SAID
        if not _BAD_CALIBRATION_SAID:
            _BAD_CALIBRATION_SAID = True
            _logger.warning("token counts: the server counted %d tokens for the %d-character calibration "
                            "sample (%d here) — not believed; counts are left as they were",
                            n_server, len(CALIBRATION_SAMPLE), n_local)
        return {}
    out = {"server_tokens": n_server, "local_tokens": n_local,
           "chars_per_token": round(chars, 3), "encoder_scale": round(scale, 4),
           # how far the loaded tokenizer is from the server on the sample (-0.026 = 2.6 % under)
           "drift": round(n_local / n_server - 1.0, 4) if n_local else None,
           # the tokenizer was built from the server's model file and checked id for id
           "exact": bool(getattr(TOKEN_ENCODER, "exact", False)) and n_local == n_server}
    global _calibration
    changed = (chars != _CHARS_PER_TOKEN and not n_local) or scale != _ENCODER_SCALE
    was = _calibration
    _CHARS_PER_TOKEN, _ENCODER_SCALE = chars, scale
    _calibration = dict(out)            # rebound whole: a reader in another thread sees old or new
    if changed:
        clear_token_cache()
    if scale != 1.0 and was.get("encoder_scale") != out["encoder_scale"]:
        _logger.warning(
            "token counts: the loaded tokenizer is NOT the server's (%d vs %d tokens on the "
            "calibration sample) — counts are scaled by %.3f. Set the model profile's tokenizer.",
            n_local, n_server, scale)
    elif not n_local and was.get("chars_per_token") != out["chars_per_token"]:
        _logger.info("token counts: estimated at %.2f characters per token (the server's own count)", chars)
    return dict(out)


def calibration() -> dict:
    """What the last ``calibrate`` found ({} = none yet)."""
    return dict(_calibration)


def reset_calibration() -> None:
    """Back to the uncalibrated estimate (tests; a change of main model)."""
    global _CHARS_PER_TOKEN, _ENCODER_SCALE, _calibration, _BAD_CALIBRATION_SAID
    _CHARS_PER_TOKEN, _ENCODER_SCALE = 4.0, 1.0
    _calibration = {}
    _BAD_CALIBRATION_SAID = False
    clear_token_cache()


def estimate_tokens(text: str) -> int:
    """
    Accurately estimates tokens using the Qwen tokenizer.
    Falls back to a `len(text)//4` heuristic if the tokenizer failed to
    load. The first fallback per process emits a single WARNING log so the
    operator knows context-budget math is approximate.
    """
    if not text:
        return 0

    # Massive texts: skip the cache entirely. Hashing megabytes per call
    # would dwarf the encoder cost we're trying to amortize.
    if len(text) > 10000:
        return _encoder_count(text)

    key = _short_cache_key(text)
    with _SHORT_TOKEN_CACHE_LOCK:
        cached = _SHORT_TOKEN_CACHE.get(key)
        generation = _CACHE_GENERATION
    if cached is not None:
        return cached

    # Encode outside the lock — it is the expensive part.
    count = _encoder_count(text)
    with _SHORT_TOKEN_CACHE_LOCK:
        if generation != _CACHE_GENERATION:
            return count
        if len(_SHORT_TOKEN_CACHE) >= _SHORT_TOKEN_CACHE_MAX:
            # Dict preserves insertion order — drop the oldest entry. Cheaper
            # than wiring a full LRU and good enough for a 1024-entry cap.
            try:
                oldest = next(iter(_SHORT_TOKEN_CACHE))
                _SHORT_TOKEN_CACHE.pop(oldest, None)
            except StopIteration:
                pass
        _SHORT_TOKEN_CACHE[key] = count
    return count


def check_budget(messages: list, max_tokens: int, system_prompt: str = "") -> dict:
    """Pre-flight validation of message payload against token budget.

    Returns a dict with:
    - fits (bool): whether the payload fits within budget
    - total_tokens (int): estimated total tokens
    - overflow (int): tokens over budget (0 if fits)
    - per_message (list): per-message token counts for debugging
    - system_tokens (int): tokens charged to the system prompt (0 if none)

    The optional ``system_prompt`` argument lets callers include the
    upstream system prompt — which is prepended by the LLM client and is
    NOT in ``messages`` — in the budget calculation. Without it, callers
    were silently underestimating context usage by however many tokens the
    system prompt takes (often 1–4K).

    Usage::

        result = check_budget(payload["messages"], max_context, system_prompt=sp)
        if not result["fits"]:
            # truncate or reject
    """
    per_message = []
    total = 0

    if messages:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal: extract text parts
                text_parts = [
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                text = " ".join(text_parts)
            else:
                text = str(content)
            # Add overhead per message (role token, delimiters) ≈ 4 tokens
            msg_tokens = estimate_tokens(text) + 4
            per_message.append(msg_tokens)
            total += msg_tokens

    system_tokens = 0
    if system_prompt:
        # Mirror the per-message overhead — the system prompt rides as a
        # role-tagged message on the wire too.
        system_tokens = estimate_tokens(str(system_prompt)) + 4
        total += system_tokens

    overflow = max(0, total - max_tokens)
    return {
        "fits": total <= max_tokens,
        "total_tokens": total,
        "overflow": overflow,
        "per_message": per_message,
        "system_tokens": system_tokens,
    }
