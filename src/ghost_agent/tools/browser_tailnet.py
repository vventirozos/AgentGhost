"""§4UU-4 — the sandbox browser cannot reach the tailnet; a tailnet URL is refused with the way that works."""


#: §4UU-4: what reaches a tailnet address instead of the sandbox browser
TAILNET_HINT = ("The browser runs inside the sandbox behind Tor and cannot reach the tailnet (a kept app's URL, a "
                "lab, another machine). For a page in the workspace load it as file:// or through its running sandbox "
                "service on 127.0.0.1; for a kept app or a lab, if you have the `lab` tool, use `lab http` (lab_id + "
                "path; the member defaults to the kept app's) or `lab screenshot` — otherwise say it cannot be "
                "reached from here.")


def tailnet_host(url) -> str:
    """The host of `url` when it is a tailnet address (a `*.ts.net` name, an IP in 100.64.0.0/10 or
    fd7a:115c:a1e0::/48), else "". (A bare MagicDNS name like `nova` is not caught: a single-label host is also
    how a sandbox service or a container is named.)"""
    import ipaddress
    from urllib.parse import urlparse
    u = str(url or "").strip()
    if not u:
        return ""
    try:
        host = (urlparse(u if "://" in u else "http://" + u).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""
    if not host:
        return ""
    if host == "ts.net" or host.endswith(".ts.net"):
        return host
    try:
        ip = ipaddress.ip_address(host)
        if (ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10")) or (
                ip.version == 6 and ip in ipaddress.ip_network("fd7a:115c:a1e0::/48")):
            return host
    except ValueError:
        pass
    return ""


def tailnet_refusal(url, actions=None) -> str:
    """The refusal for a browser call whose url (or an interact goto/navigate) is a tailnet address, else "".
    ERR_SOCKS_CONNECTION_FAILED ×9 in 5 owner turns, 30–60 s each: refuse at once and say what reaches it."""
    for u in [url] + [a.get("url") for a in (actions or [])
                      if isinstance(a, dict) and a.get("action") in ("goto", "navigate")]:
        host = tailnet_host(u)
        if host:
            return f"Refused navigation: {host} is a tailnet address. {TAILNET_HINT}"
    return ""
