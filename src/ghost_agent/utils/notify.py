"""Outbound push notifications — the delivery half of the agent's "mouth".

Transports (both optional, both may be configured together):

* **Generic webhook** (``--notify-webhook`` / ``GHOST_NOTIFY_WEBHOOK``) —
  one JSON POST per record: ``{title, body, severity, phase, ts}``.
  Points at anything that accepts JSON (ntfy in JSON mode, a Matrix/Gotify
  bridge, a home-lab relay...).
* **ntfy** (``--notify-ntfy`` / ``GHOST_NOTIFY_NTFY``) — a full topic URL
  (e.g. ``http://ghost.lan:8090/ghost-agent``); plain-text POST with the
  ``Title``/``Priority`` headers ntfy expects.

Egress posture: this module must respect fail-closed Tor. A public target
is only ever reached through the Tor SOCKS proxy (socks5h — DNS inside
Tor); loopback / RFC1918 / CGNAT-Tailscale (100.64/10) / link-local /
``.local``-style LAN hostnames go direct, which the egress guard permits.
With mandatory-tor active and NO tor proxy available, a public target is
skipped outright rather than attempting a direct connect the guard would
(rightly) kill.

Delivery is best-effort by contract: one retry per transport, failures are
logged and swallowed — the durable record already lives in the activity
ledger, and pull consumers (digest, /api/notifications) don't depend on
push succeeding.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import threading
from typing import Optional
from urllib.parse import urlsplit

logger = logging.getLogger("GhostAgent")

_TIMEOUT_S = 8.0
_LAN_SUFFIXES = (".local", ".lan", ".home", ".internal", ".arpa")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")  # Tailscale lives here


def url_needs_tor(url: str) -> bool:
    """True when ``url`` targets a PUBLIC host (must route via Tor under
    the fail-closed posture); False for loopback/LAN/Tailscale targets
    that the egress guard allows directly. Unparseable → treat as public
    (fail toward Tor, never toward a direct public connect)."""
    try:
        host = (urlsplit(url).hostname or "").strip("[]").lower()
        if not host:
            return True
        try:
            ip = ipaddress.ip_address(host)
            return not (ip.is_loopback or ip.is_private or ip.is_link_local
                        or ip in _CGNAT)
        except ValueError:
            pass  # not an IP literal — classify by hostname shape
        if host == "localhost" or host.endswith(_LAN_SUFFIXES):
            return False
        if "." not in host:
            return False  # bare intranet hostname
        return True
    except Exception:  # noqa: BLE001
        return True


class OutboundNotifier:
    """Push an :class:`ActivityRecord`-shaped payload to the configured
    transports. ``send()`` never raises; ``send_soon()`` is the
    fire-and-forget entry the activity log's ``on_notify`` uses."""

    def __init__(self, webhook_url: Optional[str] = None,
                 ntfy_url: Optional[str] = None,
                 tor_proxy: Optional[str] = None,
                 timeout: float = _TIMEOUT_S,
                 transport=None):
        self.webhook_url = (webhook_url or "").strip() or None
        self.ntfy_url = (ntfy_url or "").strip() or None
        self.tor_proxy = (tor_proxy or "").strip() or None
        self.timeout = float(timeout)
        # Test seam: an httpx-compatible transport injected into every
        # client this notifier builds (httpx.MockTransport in tests).
        self._transport = transport
        self.sent_count = 0
        self.failed_count = 0

    @property
    def configured(self) -> bool:
        return bool(self.webhook_url or self.ntfy_url)

    # -- egress routing ----------------------------------------------------

    def _proxy_for(self, url: str) -> Optional[str]:
        """SOCKS proxy to use for ``url``, or None for direct. Raises
        ``PermissionError`` when the target is public, Tor is mandatory,
        and no proxy exists (caller skips the transport)."""
        if not url_needs_tor(url):
            return None
        if self.tor_proxy:
            proxy = self.tor_proxy
            if proxy.startswith("socks5://"):
                # socks5h — resolve DNS inside Tor, no cleartext lookup.
                proxy = proxy.replace("socks5://", "socks5h://", 1)
            return proxy
        raise PermissionError(
            f"public notify target {url!r} with no Tor proxy available")

    # -- delivery ----------------------------------------------------------

    async def _post(self, url: str, *, json_body=None, content=None,
                    headers=None) -> bool:
        import httpx
        proxy = self._proxy_for(url)
        kwargs = {"timeout": self.timeout}
        if proxy:
            kwargs["proxy"] = proxy
        if self._transport is not None:
            kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**kwargs) as client:
            r = await client.post(url, json=json_body, content=content,
                                  headers=headers or {})
            return 200 <= r.status_code < 300

    async def send(self, *, title: str, body: str, severity: str = "notify",
                   phase: str = "", ts: float = 0.0) -> bool:
        """Deliver to every configured transport; True if ANY succeeded.
        One retry per transport. Never raises."""
        if not self.configured:
            return False
        delivered = False
        targets = []
        if self.webhook_url:
            targets.append((self.webhook_url, {
                "json_body": {"title": title, "body": body,
                              "severity": severity, "phase": phase,
                              "ts": ts},
            }))
        if self.ntfy_url:
            targets.append((self.ntfy_url, {
                "content": body.encode("utf-8", "ignore"),
                "headers": {
                    "Title": title.encode("ascii", "ignore").decode("ascii"),
                    "Priority": "high" if severity == "notify" else "default",
                },
            }))
        for url, kwargs in targets:
            for attempt in (1, 2):
                try:
                    if await self._post(url, **kwargs):
                        delivered = True
                        break
                except PermissionError as e:
                    logger.debug("notify skipped: %s", e)
                    break  # routing verdict won't change on retry
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        logger.debug("notify delivery failed (%s): %s",
                                     url, e)
        if delivered:
            self.sent_count += 1
        else:
            self.failed_count += 1
        return delivered

    def send_soon(self, record) -> None:
        """Fire-and-forget delivery of an ActivityRecord (or anything with
        ``phase``/``summary``/``severity``/``ts`` attributes). Safe to call
        from sync code with or without a running event loop; never raises."""
        try:
            phase = getattr(record, "phase", "") or ""
            title = f"Ghost Agent — {phase or 'notification'}"
            kwargs = {
                "title": title,
                "body": getattr(record, "summary", "") or "",
                "severity": getattr(record, "severity", "notify") or "notify",
                "phase": phase,
                "ts": float(getattr(record, "ts", 0.0) or 0.0),
            }
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(self.send(**kwargs))
            else:
                # No loop (sync caller, e.g. a thread) — one daemon thread
                # per push is fine at notification volume.
                threading.Thread(
                    target=lambda: asyncio.run(self.send(**kwargs)),
                    daemon=True,
                ).start()
        except Exception as e:  # noqa: BLE001
            logger.debug("send_soon failed: %s", e)


def notifier_from_config(args=None, *, tor_proxy: Optional[str] = None,
                         transport=None) -> OutboundNotifier:
    """Build a notifier from CLI args (flags win) with env fallbacks
    (``GHOST_NOTIFY_WEBHOOK`` / ``GHOST_NOTIFY_NTFY``)."""
    webhook = getattr(args, "notify_webhook", None) if args is not None else None
    ntfy = getattr(args, "notify_ntfy", None) if args is not None else None
    if not isinstance(webhook, str):
        webhook = None
    if not isinstance(ntfy, str):
        ntfy = None
    return OutboundNotifier(
        webhook_url=webhook or os.getenv("GHOST_NOTIFY_WEBHOOK", ""),
        ntfy_url=ntfy or os.getenv("GHOST_NOTIFY_NTFY", ""),
        tor_proxy=tor_proxy,
        transport=transport,
    )


__all__ = ["OutboundNotifier", "notifier_from_config", "url_needs_tor"]
