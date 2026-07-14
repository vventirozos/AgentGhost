import datetime
import os
import asyncio
import httpx
import subprocess
import platform
import ipaddress
from typing import List, Optional

import socket

# Hostnames that always resolve to the local machine, regardless of DNS.
_BLOCKED_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def _ip_is_internal(ip_str: str) -> bool:
    """True if `ip_str` is a loopback / private / link-local / reserved /
    multicast / unspecified address — i.e. an SSRF-relevant internal target
    (covers 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16 incl. the
    169.254.169.254 cloud-metadata endpoint, ::1, fc00::/7, etc.)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def url_ssrf_reason(url: str, *, resolve: bool = True) -> Optional[str]:
    """Return None if `url` is a safe EXTERNAL http(s) target, else a short
    human-readable refusal reason.

    The single SSRF guard shared by every tool that fetches an
    LLM-supplied URL (web fetch, file download, vision, browser). It blocks:
      * non-http(s) schemes — file://, gopher://, dict://, … (local-file read)
      * hosts that ARE, or (best-effort) DNS-RESOLVE TO, a loopback /
        private / link-local / reserved address — the SSRF + cloud-metadata
        (169.254.169.254) + internal-service class.
    DNS resolution failures are treated as "allow" so a transient resolver
    problem can't block a legitimate public fetch.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(str(url))
    except Exception:
        return f"invalid URL: {url!r}"
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return (
            f"refused non-http(s) URL (scheme={scheme!r}); only http/https are "
            f"allowed (blocks file://, gopher://, dict://, …)."
        )
    host = (parsed.hostname or "").lower()
    if not host:
        return f"refused URL with no host: {url!r}"
    if host in _BLOCKED_HOSTNAMES:
        return f"refused local host {host!r} (SSRF guard)."
    if _ip_is_internal(host):
        return f"refused internal/loopback/link-local host {host!r} (SSRF guard)."
    if resolve:
        try:
            port = parsed.port or (443 if scheme == "https" else 80)
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except Exception:
            infos = []
        for info in infos:
            ip = info[4][0]
            if _ip_is_internal(ip):
                return (
                    f"refused host {host!r} resolving to internal address {ip} "
                    f"(SSRF guard)."
                )
    return None

def request_new_tor_identity(control_port=9051, password=""):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", control_port))
            if password:
                s.sendall(f'AUTHENTICATE "{password}"\r\n'.encode())
            else:
                s.sendall(b'AUTHENTICATE\r\n')
            
            resp = s.recv(1024).decode()
            if not resp.startswith("250"):
                return False, f"Tor Auth failed: {resp.strip()}"
                
            s.sendall(b'SIGNAL NEWNYM\r\n')
            resp = s.recv(1024).decode()
            if not resp.startswith("250"):
                return False, f"Tor NEWNYM failed: {resp.strip()}"
                
            return True, "Identity renewed successfully"
    except Exception as e:
        # Fallback: try restarting the tor service directly
        try:
            if platform.system() == "Darwin":
                # macOS Homebrew
                subprocess.run(["brew", "services", "restart", "tor"], check=True, capture_output=True)
                return True, "Identity renewed successfully via brew services restart"
            else:
                # Linux systemd (might require sudo NOPASSWD or run as root, but worth trying)
                subprocess.run(["sudo", "-n", "systemctl", "restart", "tor"], check=True, capture_output=True)
                return True, "Identity renewed successfully via systemctl restart"
        except Exception as fallback_e:
            return False, f"Tor control port error: {e}. Fallback restart also failed: {fallback_e}"


def socks_url_with_identity(tor_proxy: Optional[str], identity: str) -> Optional[str]:
    """Inject a SOCKS ``username:password`` into a SOCKS URL so Tor's
    ``IsolateSOCKSAuth`` (on by default) assigns a SEPARATE CIRCUIT per
    ``identity`` tag — circuit-per-identity isolation without the global
    NEWNYM thrash, and without needing the control port.

    Distinct identities → distinct circuits → a colluding set of exit
    nodes / sites can't trivially link a sequence of differently-tagged
    requests into one session. Returns ``tor_proxy`` unchanged when it is
    falsy, has no parseable host, or already carries credentials.
    """
    if not tor_proxy or not identity:
        return tor_proxy
    import re as _re
    from urllib.parse import urlparse, urlunparse
    try:
        p = urlparse(tor_proxy)
        if not p.hostname or p.username:
            return tor_proxy  # unparseable or already has creds
        tag = _re.sub(r"[^A-Za-z0-9]", "", str(identity))[:32] or "ghost"
        netloc = f"{tag}:isolate@{p.hostname}:{p.port or 9050}"
        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return tor_proxy

async def helper_fetch_url_content(
    url: str, *, proxy_override: Optional[str] = None, renew_identity: bool = True,
) -> str:
    """Fetch and text-extract a URL over Tor.

    ``proxy_override`` — use this SOCKS proxy instead of the env default. The
    deep-research fetch passes its per-query identity-tagged proxy so the
    fetch rides the same isolated circuit as the search, instead of the bare
    env proxy (which dropped the per-query circuit isolation).

    ``renew_identity=False`` — do NOT fire a global Tor NEWNYM on 503 / error.
    A global identity renewal from one fetch re-circuits Tor for every
    concurrently-running sibling fetch (deep_research runs several at once),
    sabotaging them; callers that fan out should pass False.
    """
    # 1. Setup Tor Proxy (computed BEFORE the SSRF guard so the guard knows
    # whether to do a host-side DNS lookup — see below).
    proxy_url = proxy_override or os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    if proxy_url and proxy_url.startswith("socks5://"):
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    # --- URL VALIDATION (shared SSRF guard) ---
    # Reject non-http(s) schemes and any host that is/resolves to an
    # internal address, so an LLM tool call can't fetch `file:///etc/passwd`,
    # hit cloud-metadata (169.254.169.254), or reach internal services.
    #
    # resolve=False when fetching over Tor: the default resolve=True does a
    # HOST-SIDE getaddrinfo of the target hostname, which LEAKS the DNS query
    # for the very site we're about to visit anonymously — defeating the
    # DNS-over-SOCKS anonymity this fetch exists to provide (and, for a
    # .onion, leaking which hidden service is being visited). Tor routes/
    # resolves at the exit node anyway, so a host lookup buys no protection
    # here; literal-IP internal targets are still blocked without resolving.
    # Mirrors the browser/download tools' `resolve=not anonymous` handling.
    _ssrf = url_ssrf_reason(url, resolve=not bool(proxy_url))
    if _ssrf:
        return f"Error: {_ssrf}"

    try:
        import curl_cffi.requests
    except ImportError:
        curl_cffi = None

    # Hard body-size cap. A malicious / misconfigured server can stream
    # multi-GB content under any timeout; we count bytes ourselves.
    MAX_BODY_BYTES = 5 * 1024 * 1024  # 5 MB ceiling

    for attempt in range(3):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

            if curl_cffi:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                async with curl_cffi.requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=20.0) as client:
                    resp = await client.get(url, headers=headers)
            else:
                # Fallback to httpx if curl_cffi is missing for some reason
                async with httpx.AsyncClient(proxy=proxy_url, timeout=20.0, follow_redirects=True) as client:
                    resp = await client.get(url, headers=headers)

            content_type = resp.headers.get("content-type", "").lower()
            # Check the URL *path*, not the raw URL — "/report.pdf?dl=1"
            # must still trip the binary short-circuit.
            from urllib.parse import urlparse
            url_path = urlparse(url).path.lower()
            if "application/pdf" in content_type or "application/octet-stream" in content_type or url_path.endswith((".pdf", ".zip")):
                return "Error: URL points to a binary file. To read PDFs, download them using file_system and ingest them using knowledge_base."

            # Enforce Content-Length cap before reading the body if the
            # server tells us the size up front.
            try:
                content_length = int(resp.headers.get("content-length", "0") or "0")
            except (TypeError, ValueError):
                content_length = 0
            if content_length and content_length > MAX_BODY_BYTES:
                return f"Error: response from {url} is {content_length // (1024*1024)} MB; refusing to read more than {MAX_BODY_BYTES // (1024*1024)} MB."

            status_code = resp.status_code
            text = resp.text
            # And cap the actual decoded text in case the server lied about
            # Content-Length or sent chunked transfer encoding.
            if isinstance(text, str) and len(text) > MAX_BODY_BYTES:
                text = text[:MAX_BODY_BYTES] + "\n[... TRUNCATED at 5 MB ceiling ...]"
            
            if status_code != 200:
                # Only renew the Tor identity on 503. 401/403 are
                # APPLICATION-level "you are forbidden" responses (auth
                # token, geo block, account state) — rotating the exit
                # node won't change them, and burning identities on every
                # 401 just slows the next legitimate request. Reserve
                # rotation for 503, which often signals the exit node
                # itself is being rate-limited or filtered.
                if status_code == 503 and proxy_url:
                    if attempt < 2 and renew_identity:
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    return f"Error: Access Denied (503) via Tor. The site {url} likely blocks Tor exit nodes. Try a different source."
                if status_code in (401, 403):
                    return f"Error: Access Denied ({status_code}) from {url}. Application-level forbidden — Tor rotation will not help."
                return f"Error: Received status {status_code} from {url}"
            
            def _parse_html(html_content):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "iframe", "svg"]):
                    script.decompose()
                text_content = soup.get_text(separator=' ', strip=True)
                return " ".join(text_content.split()) if text_content else "Error: No text content extracted from page."
            
            return await asyncio.to_thread(_parse_html, text)
            
        except Exception as e:
            if attempt < 2 and proxy_url and renew_identity:
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            return f"Error reading {url}: {str(e)}"
            
    return f"Error fetching {url} after 3 retries."

# --------------------------------------------------------------------------
# Removal / negation detection
# --------------------------------------------------------------------------
# Phrases that mark a fact as a REMOVAL or NON-OWNERSHIP statement rather than
# a positive thing to remember. Consolidation must NOT store these: doing so
# manufactures self-perpetuating tombstones — e.g. asking "what pets do I
# own?" makes the model answer "you previously had an iguana that was
# removed", which then gets consolidated back into a fresh "user previously
# had an iguana" fact, keeping the deleted entity alive forever. A removal
# should DELETE knowledge, never insert a new memory about the deletion.
_REMOVAL_NEGATION_PHRASES = (
    "no longer", "previously had", "previously owned", "used to have",
    "used to own", "was removed", "were removed", "has been removed",
    "have been removed", "had been removed", "removed from", "deleted from",
    "does not own", "doesn't own", "do not own", "don't own",
    "does not have", "doesn't have", "do not have", "don't have",
    "did not have", "didn't have", "never had", "never owned",
    "no longer has", "no longer have", "no longer owns", "no longer own",
    "got rid of", "is gone", "are gone", "is no more",
    "not own", "not have an", "not have a",
    # Parenthetical tombstone markers an earlier soft-delete left behind,
    # e.g. "Mortimer the iguana (removed)".
    "(removed)", "[removed]", "(deleted)", "[deleted]", "(former)",
)

#: Graph-predicate fragments that encode the same removal/past-tense
#: semantics. Predicates are uppercase verbs (e.g. ``PREVIOUSLY_OWNED``,
#: ``NO_LONGER_HAS``, ``REMOVED``) — matched case-insensitively as substrings.
_REMOVAL_PREDICATE_FRAGMENTS = (
    "PREVIOUSLY", "FORMER", "REMOVED", "NO_LONGER", "NOLONGER",
    "NOT_", "NEVER", "DELETED", "USED_TO", "PAST_",
)


def is_removal_or_negation_text(text) -> bool:
    """True iff ``text`` reads as a removal / non-ownership statement.

    Used to stop the memory consolidator from re-storing "X was removed"
    tombstones (see `_REMOVAL_NEGATION_PHRASES`)."""
    if not text:
        return False
    t = " ".join(str(text).lower().split())
    return any(p in t for p in _REMOVAL_NEGATION_PHRASES)


def is_removal_triplet(triplet) -> bool:
    """True iff a graph triplet encodes removal / past-ownership and so must
    not be ingested. Checks the predicate for tombstone verbs and the
    subject/object text for removal phrasing."""
    if not isinstance(triplet, dict):
        return False
    pred = str(triplet.get("predicate", "")).upper()
    if any(frag in pred for frag in _REMOVAL_PREDICATE_FRAGMENTS):
        return True
    for field in ("subject", "object"):
        if is_removal_or_negation_text(triplet.get(field, "")):
            return True
    return False


def get_utc_timestamp():
    """Returns strict ISO8601 UTC timestamp for Go/iOS clients.

    The `Z` suffix is intentional — Go and iOS Decoder strict-mode requires
    it. Python consumers that want to round-trip the value should use
    `parse_utc_timestamp()` below rather than doing their own `strptime`,
    which avoids the trap of forgetting to strip the `Z`.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc_timestamp(s: str) -> datetime.datetime:
    """Robust parser for the strings produced by `get_utc_timestamp()`.

    Accepts the canonical ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` form, its Z-less
    variant, and ``+00:00``-suffixed variants. Raises `ValueError` on
    genuinely malformed input. Use this helper wherever you parse a
    Ghost-emitted timestamp so a future format change lands in one place.
    """
    if not isinstance(s, str):
        raise ValueError(f"timestamp must be a str, got {type(s).__name__}")
    stripped = s.strip()
    if not stripped:
        raise ValueError("empty timestamp")
    # Normalise the common Z-suffix to the +00:00 form that
    # `datetime.fromisoformat` understands on every supported Python.
    if stripped.endswith("Z"):
        stripped = stripped[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(stripped)
    except ValueError:
        # Fallback for the legacy `%f` strftime shape (some callers produce
        # exactly 6-digit microseconds, which fromisoformat handles, but we
        # keep this branch in case of edge cases like `.%f` without tz).
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(stripped, fmt)
                break
            except ValueError:
                continue
        else:
            raise
    # Always return a naive UTC datetime so arithmetic like
    # `(now - mem_time).total_seconds()` works without tz mixing.
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt

def recursive_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 70) -> List[str]:
    if not text: return []
    # The range() loops below step by `chunk_size - chunk_overlap`. When
    # chunk_overlap >= chunk_size that step is <= 0, so range() yields
    # NOTHING and the text is silently DROPPED (a 0 step raises
    # ValueError). Clamp so the step is always >= 1. Reachable from
    # semantic_split_text when a long header shrinks the effective
    # chunk_size below the overlap.
    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    if len(text) <= chunk_size: return [text]

    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    final_chunks = []
    stack = [text]

    while stack:
        current_text = stack.pop()

        if len(current_text) <= chunk_size:
            final_chunks.append(current_text)
            continue

        found_sep = ""
        for sep in separators:
            if sep in current_text:
                found_sep = sep
                break

        if not found_sep:
            for i in range(0, len(current_text), chunk_size - chunk_overlap):
                final_chunks.append(current_text[i:i+chunk_size])
            continue

        parts = current_text.split(found_sep)
        buffer = ""
        temp_chunks = []

        for p in parts:
            fragment = p + found_sep if found_sep.strip() else p
            if len(buffer) + len(fragment) <= chunk_size:
                buffer += fragment
            else:
                if buffer:
                    temp_chunks.append(buffer.strip())
                buffer = fragment

        if buffer:
            temp_chunks.append(buffer.strip())

        # Push everything back through the stack (reversed, so LIFO pops in
        # document order) — appending fitting chunks to final_chunks here
        # while iterating reversed would emit them backwards.
        for chunk in reversed(temp_chunks):
            if len(chunk) > chunk_size and (found_sep == "" or chunk == current_text):
                # Can't be reduced by separators — hard character split.
                pieces = [chunk[i:i+chunk_size]
                          for i in range(0, len(chunk), chunk_size - chunk_overlap)]
                stack.extend(reversed(pieces))
            else:
                stack.append(chunk)

    return final_chunks


def semantic_split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """Structure-aware chunking that respects document semantics.

    Unlike ``recursive_split_text`` which splits purely by character count,
    this function:
    1. Detects document structure (markdown headers, code blocks, list items)
    2. Groups content by section when possible
    3. Never splits mid-sentence or mid-code-block
    4. Prepends section headers to each chunk for retrieval context

    Falls back to ``recursive_split_text`` for unstructured content.
    """
    import re

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Detect if content has markdown structure
    has_headers = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
    has_code_blocks = '```' in text

    if not has_headers and not has_code_blocks:
        return recursive_split_text(text, chunk_size, chunk_overlap)

    # Split by sections (markdown headers)
    sections = []
    if has_headers:
        # Split on headers, keeping the header with its content
        parts = re.split(r'(^#{1,6}\s[^\n]*\n)', text, flags=re.MULTILINE)
        current_section = ""
        current_header = ""
        for part in parts:
            if re.match(r'^#{1,6}\s', part):
                if current_section.strip():
                    sections.append((current_header, current_section.strip()))
                current_header = part.strip()
                current_section = ""
            else:
                current_section += part
        if current_section.strip():
            sections.append((current_header, current_section.strip()))
    else:
        sections = [("", text)]

    # Process each section
    chunks = []
    for header, section_text in sections:
        # Prepend header context to each chunk from this section
        header_prefix = f"[Section: {header}] " if header else ""
        prefix_len = len(header_prefix)
        effective_size = chunk_size - prefix_len

        if effective_size <= 50:
            effective_size = chunk_size

        if len(section_text) <= effective_size:
            chunks.append(f"{header_prefix}{section_text}" if header_prefix else section_text)
            continue

        # For code blocks, try to keep them intact
        if has_code_blocks and '```' in section_text:
            code_parts = re.split(r'(```[^\n]*\n.*?```)', section_text, flags=re.DOTALL)
            for part in code_parts:
                if part.startswith('```'):
                    # Code block — keep intact up to 2x chunk_size
                    if len(part) <= chunk_size * 2:
                        chunks.append(f"{header_prefix}{part}" if header_prefix else part)
                    else:
                        # Very long code block — split by lines
                        sub_chunks = recursive_split_text(part, effective_size, chunk_overlap)
                        for sc in sub_chunks:
                            chunks.append(f"{header_prefix}{sc}" if header_prefix else sc)
                else:
                    # Prose between code blocks
                    if part.strip():
                        sub_chunks = recursive_split_text(part.strip(), effective_size, chunk_overlap)
                        for sc in sub_chunks:
                            chunks.append(f"{header_prefix}{sc}" if header_prefix else sc)
        else:
            # No code blocks in this section — use recursive split
            sub_chunks = recursive_split_text(section_text, effective_size, chunk_overlap)
            for sc in sub_chunks:
                chunks.append(f"{header_prefix}{sc}" if header_prefix else sc)

    return [c for c in chunks if c.strip()]