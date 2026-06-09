import hashlib
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer
from functools import lru_cache

# Tokenizer source for token-budget accounting. Must come from a repo
# that ships `tokenizer.json` matching the Qwen3 35B-A3B family (the
# tokenizer is shared across all 35B-A3B derivatives — base, instruct,
# and uncensored/finetuned variants including qwen-3.6-35b-a3). If the
# AutoTokenizer load fails, `estimate_tokens` falls back to a `len(text)
# // 4` heuristic and logs a one-shot warning.
QWEN_MODEL_ID = "Qwen/Qwen3-30B-A3B"
TOKEN_ENCODER = None

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
            "estimate_tokens: falling back to len(text)//4 heuristic (%s). "
            "Token-budget math will be approximate. Check tokenizer load logs "
            "above for the root cause.",
            reason,
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
    try:
        _SHORT_TOKEN_CACHE.clear()
    except Exception:
        pass

def load_tokenizer(local_tokenizer_path: Path):
    """
    Robust loading strategy: LOCAL DISK -> TOR NETWORK -> FALLBACK
    """
    global TOKEN_ENCODER
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
            TOKEN_ENCODER = AutoTokenizer.from_pretrained(str(local_tokenizer_path), local_files_only=True)
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
    print(f"⏳ Local missing. Downloading {QWEN_MODEL_ID} via Direct Network...")

    import threading
    import queue

    def _download_hf_tokenizer(q):
        import huggingface_hub
        original_timeout = getattr(huggingface_hub.constants, "HF_HUB_DOWNLOAD_TIMEOUT", 10)
        huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 10
        try:
            enc = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
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


def _short_cache_key(text: str) -> tuple:
    # Hash prefix + length is cheap and effectively collision-free for the
    # purpose of caching small token counts. Collisions only cost accuracy
    # on the rare colliding pair, never correctness elsewhere.
    digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]
    return (digest, len(text))


def _encoder_count(text: str) -> int:
    if TOKEN_ENCODER:
        try:
            return len(TOKEN_ENCODER.encode(text))
        except Exception as e:
            _warn_fallback_once(f"encoder error: {type(e).__name__}")
            return len(text) // 4
    _warn_fallback_once("tokenizer not loaded")
    return len(text) // 4


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
    cached = _SHORT_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    count = _encoder_count(text)
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


def estimate_payload_tokens(payload: dict) -> int:
    """Estimate the total tokens in an LLM payload (messages + tools).

    Accounts for messages, tool definitions (which consume tokens),
    and system prompt overhead.
    """
    total = 0
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            text = str(content)
        total += estimate_tokens(text) + 4  # per-message overhead

    # Tool definitions consume tokens too
    tools = payload.get("tools", [])
    if tools:
        import json
        tools_text = json.dumps(tools, default=str)
        total += estimate_tokens(tools_text)

    return total
