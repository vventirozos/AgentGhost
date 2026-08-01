"""Web Push for the ghost web client (2026-08-01).

iOS delivers web push ONLY to Home-Screen-installed PWAs over trusted
HTTPS (APNs-backed, standard Web Push API + VAPID) — the in-page bell
cannot reach a locked phone. This module owns the transport:

- VAPID keys: `$GHOST_VAPID_FILE` (default `~/Data/AI/.ghost_vapid.json`,
  0600, generated once out-of-band; NEVER regenerate casually — existing
  subscriptions are bound to the public key and all die with it).
- Subscriptions: `$GHOST_PUSH_SUBS_FILE` (default
  `~/Data/AI/.ghost_push_subs.json`, 0600), keyed by endpoint URL —
  re-subscribing the same device is an upsert, not a duplicate.
- `broadcast()` fans out to every subscription; a 404/410 from the push
  service means the subscription is dead and it is pruned in place.

pywebpush is synchronous (requests-based) — callers on the event loop
must use `broadcast_async` (thread offload), never `broadcast` directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

_VAPID_FILE = Path(os.environ.get(
    "GHOST_VAPID_FILE", str(Path.home() / "Data/AI/.ghost_vapid.json")))
_SUBS_FILE = Path(os.environ.get(
    "GHOST_PUSH_SUBS_FILE", str(Path.home() / "Data/AI/.ghost_push_subs.json")))

_lock = threading.Lock()
_vapid_cache: Optional[Dict[str, str]] = None
_vapid_signer = None  # py_vapid.Vapid built from the PEM, cached


def vapid_config() -> Optional[Dict[str, str]]:
    """{'private_key_pem', 'public_key_b64url', 'sub'} or None (push off)."""
    global _vapid_cache
    if _vapid_cache is not None:
        return _vapid_cache
    try:
        data = json.loads(_VAPID_FILE.read_text())
        if data.get("private_key_pem") and data.get("public_key_b64url"):
            _vapid_cache = data
            return data
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"webpush: VAPID file unreadable: {e}")
    return None


def public_key() -> Optional[str]:
    cfg = vapid_config()
    return cfg["public_key_b64url"] if cfg else None


def _vapid_key_object():
    """py_vapid.Vapid built from the stored PEM — pywebpush accepts a
    Vapid INSTANCE (or a raw-base64 string or a file path) but NOT PEM
    text: `Vapid.from_string()` b64-decodes and dies on a PEM blob, and
    that ValueError was being swallowed per-endpoint into a debug-level
    "send failed" — push silently dead (2026-08-01 review CRITICAL)."""
    global _vapid_signer
    if _vapid_signer is not None:
        return _vapid_signer
    cfg = vapid_config()
    if cfg is None:
        return None
    try:
        from py_vapid import Vapid
        _vapid_signer = Vapid.from_pem(cfg["private_key_pem"].encode())
    except Exception as e:
        logger.warning(f"webpush: VAPID PEM unusable: {e}")
        return None
    return _vapid_signer


def _load_subs() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_SUBS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"webpush: subscriptions file unreadable: {e}")
        return {}


def _save_subs(subs: Dict[str, Dict[str, Any]]) -> None:
    # 0600 from the first byte: the endpoint+auth+p256dh triple is a
    # bearer capability to push to that device — a create-then-chmod
    # left a world-readable window.
    _SUBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SUBS_FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(subs, indent=1))
    tmp.replace(_SUBS_FILE)


def subscription_count() -> int:
    with _lock:
        return len(_load_subs())


def add_subscription(subscription: Dict[str, Any]) -> bool:
    """Store (upsert by endpoint). Returns False on a malformed payload."""
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys") or {}
    if (not isinstance(endpoint, str) or not endpoint.startswith("https://")
            or not keys.get("p256dh") or not keys.get("auth")):
        return False
    with _lock:
        subs = _load_subs()
        subs[endpoint] = subscription
        _save_subs(subs)
    return True


def remove_subscription(endpoint: str) -> bool:
    with _lock:
        subs = _load_subs()
        if endpoint in subs:
            subs.pop(endpoint)
            _save_subs(subs)
            return True
    return False


def broadcast(title: str, body: str, *, url: str = "/", tag: str = "") -> int:
    """Send to every subscription; prune dead ones. Returns sent count.
    BLOCKING (pywebpush/requests) — use broadcast_async on the loop."""
    cfg = vapid_config()
    if cfg is None:
        return 0
    with _lock:
        subs = _load_subs()
    if not subs:
        return 0
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        logger.warning("webpush: pywebpush not installed — push disabled")
        return 0
    signer = _vapid_key_object()
    if signer is None:
        return 0
    payload = json.dumps({
        "title": title[:120], "body": body[:400], "url": url, "tag": tag})
    sent = 0
    dead: List[str] = []
    for endpoint, sub in subs.items():
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=signer,
                vapid_claims={"sub": cfg.get("sub", "mailto:ghost@localhost")},
                ttl=3600,
            )
            sent += 1
        except WebPushException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (404, 410):
                dead.append(endpoint)  # unsubscribed/expired — prune
            else:
                logger.warning(f"webpush: send failed ({status}): {e}")
        except Exception as e:
            logger.warning(f"webpush: send failed: {e}")
    if dead:
        with _lock:
            fresh = _load_subs()
            for ep in dead:
                fresh.pop(ep, None)
            _save_subs(fresh)
    return sent


async def broadcast_async(title: str, body: str, *, url: str = "/",
                          tag: str = "") -> int:
    return await asyncio.to_thread(broadcast, title, body, url=url, tag=tag)
