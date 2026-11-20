"""§4UN-fetch: "is this the real content?" — one place for the checks every internet fetch needs.

* ``body_mismatch(filename, content_type, head)`` — a download's first bytes against the type its name
  promises (the §4LL "saved the error page as the photo" class): an HTML page, a JSON error or an empty body
  under a binary name, or an HTML page under a text/data name (``data.csv``), is not the file.
* ``challenge_reason(status, title, text, url)`` — a bot check, consent wall or JavaScript-only shell served
  as a page (Cloudflare, Anubis, DDoS-Guard, Incapsula, Google/YouTube consent), so a fetch never stores or
  cites it as the document.
* ``extract_main_text(html)`` / ``page_meta(html, headers)`` / ``pdf_text(data)`` — pure helpers for readers:
  the page's main text without menus, banners and forms; its title and published/modified date; a PDF's text.

Pure functions (no network, no logging) so they can be pinned over their inputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# ── file types ─────────────────────────────────────────────────────────

#: suffix → the sniffed kind (`quarantine.sniff_type`) its bytes must have. ftyp containers (mp4, mov, m4a,
#: heic, avif) sniff as "mp4"; zip containers (docx, xlsx, pptx, epub, jar, apk, whl, odt) as "zip".
_EXPECTED_KIND: Dict[str, str] = {
    ".pdf": "pdf",
    ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "riff (webp/avi/wav)",
    ".avif": "mp4", ".heic": "mp4", ".heif": "mp4", ".mp4": "mp4", ".m4a": "mp4", ".mov": "mp4", ".m4v": "mp4",
    ".zip": "zip", ".docx": "zip", ".xlsx": "zip", ".pptx": "zip", ".epub": "zip", ".jar": "zip", ".apk": "zip",
    ".whl": "zip", ".odt": "zip", ".ods": "zip", ".odp": "zip",
    ".gz": "gzip", ".tgz": "gzip", ".bz2": "bzip2", ".xz": "xz", ".7z": "7z", ".rar": "rar",
    ".doc": "ole-compound (legacy Office / msi)", ".xls": "ole-compound (legacy Office / msi)",
    ".ppt": "ole-compound (legacy Office / msi)", ".msi": "ole-compound (legacy Office / msi)",
    ".mkv": "matroska/webm", ".webm": "matroska/webm", ".wav": "riff (webp/avi/wav)",
    ".avi": "riff (webp/avi/wav)", ".sqlite": "sqlite", ".db": "sqlite", ".exe": "pe-executable",
}
#: binary formats whose magic the sniffer does not know — still never text or HTML
#: (not `.bin`: it is also the name of a download whose URL has none — any content)
_BINARY_ONLY = frozenset({".bmp", ".tif", ".tiff", ".ico", ".mp3", ".ogg", ".flac", ".aac", ".dmg",
                          ".iso", ".so", ".dylib", ".woff", ".woff2", ".ttf", ".otf", ".parquet", ".npy",
                          ".pkl", ".pt", ".onnx", ".safetensors", ".gguf", ".tar"})
#: text formats an HTML page is not (an HTML body under these names is an error/login/challenge page)
_TEXT_NOT_HTML = frozenset({".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".md", ".py", ".js", ".ts",
                            ".css", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".sql", ".sh", ".r",
                            ".tex", ".bib", ".srt", ".vtt", ".geojson", ".ipynb"})
_HTML_OK = frozenset({".html", ".htm", ".xhtml", ".xml", ".svg", ".rss", ".atom", ".php", ".asp", ".aspx", ".jsp"})
_TEXTLIKE = frozenset({"html/xml", "json-like text", "text", "empty"})
#: §4UN-fetch r1: legacy Office names that real exporters fill with an HTML table or TSV (Excel and Word open
#: both) — such a body is the file, said in a note. r2: only with the export's own marks (`_office_export`); a
#: login, 404 or "access denied" page under these names stays PARTIAL
_OFFICE_TEXT_OK = frozenset({".xls", ".doc"})
_OFFICE_NS_RE = re.compile(rb"urn:schemas-microsoft-com:office:(?:excel|word)|\bmso-[a-z-]+\s*:", re.I)
#: r3: a login form is never an export, even laid out in a <table>
_LOGIN_RE = re.compile(rb"<input\b[^>]*\btype\s*=\s*[\"']?password|<form\b[^>]*(?:log-?in|sign-?in|logon|auth)",
                       re.I)
_MHTML_RE = re.compile(rb"mime-version:\s*1\.0", re.I)
#: r4: table cells must hold this share of the visible text for an HTML body to be a table export
_TABLE_SHARE = 0.6
#: r4: delimited text needs this many rows with the same separator count
_MIN_ROWS = 3


def _office_export(suffix: str, kind: str, head: bytes) -> bool:
    """r4 (simplified): the body is an HTML/text export Excel or Word opens.
    HTML: the Office namespace or `mso-` styles, or table cells holding at least 60% of the visible text (a
    homepage with a layout table is not an export) — never with a password field or login form.
    .xls text: at least 3 rows with the same count of one separator (tab, comma or semicolon); one title line
    may precede them. .doc text: RTF or MHTML. ``head`` is the first 64 KB."""
    b = bytes(head or b"")
    if kind == "html/xml":
        if _LOGIN_RE.search(b):
            return False
        if _OFFICE_NS_RE.search(b):
            return True
        try:
            soup = _soup(b)
            for t in soup(list(_INVISIBLE_TAGS)):
                t.decompose()
            total = len(_text_of(soup.find("body") or soup))
            cells = sum(len(_text_of(c)) for c in soup.find_all(["td", "th"]) if c.find(["td", "th"]) is None)
        except Exception:  # noqa: BLE001
            return False
        return total > 0 and cells >= _TABLE_SHARE * total
    if suffix == ".doc":
        s0 = b.lstrip()
        return s0.startswith(b"{\\rtf") or bool(
            _MHTML_RE.search(s0[:2048]) and re.search(rb"multipart/related", s0[:4096], re.I))
    lines = [ln for ln in b.decode("utf-8", "replace").splitlines()[:40] if ln.strip()]
    if len(lines) and len(b) >= 65536:
        lines = lines[:-1]                       # the head may cut the last row
    for rows in (lines, lines[1:]):              # one title line may precede the rows
        if len(rows) < _MIN_ROWS:
            continue
        for sep in ("\t", ",", ";"):
            n = rows[0].count(sep)
            if n >= 1 and all(ln.count(sep) == n for ln in rows):
                return True
    return False


#: §4UY N8: native code — a binary the loader runs or imports by its name (a `sitecustomize.so` beside a script)
NATIVE_CODE_SUFFIXES = frozenset({".exe", ".dll", ".so", ".dylib", ".jar", ".apk", ".whl", ".msi"})


def server_name_may_stand(name: str) -> bool:
    """§4UY N8: may a SERVER-chosen file name (Content-Disposition) be the saved name? Only for a document or
    media type — a format whose bytes the download checks against the name (`body_mismatch`) and that no
    toolchain loads by name. Code, config and text (`conftest.py`, `sitecustomize.py`, `pytest.ini`,
    `package.json`, `.npmrc`) and native code are loaded BY NAME from the project folder: for those the
    server gives the type only, never the name. A principle (document/media vs everything else), not a list
    of dangerous names."""
    suf = Path(str(name or "")).suffix.lower()
    return bool(suf) and (suf in _EXPECTED_KIND or suf in _BINARY_ONLY) and suf not in NATIVE_CODE_SUFFIXES


#: §4UY N8 r3: names a toolchain loads from the folder they sit in (test runners, interpreters, package managers,
#: build tools) — lower case
TOOLCHAIN_NAMES = frozenset({
    "conftest.py", "sitecustomize.py", "usercustomize.py", "setup.py", "noxfile.py", "manage.py", "__init__.py",
    "__main__.py", "pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "pipfile", "requirements.txt",
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "makefile", "gnumakefile", "dockerfile",
    "containerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml", "gemfile",
    "rakefile", "cargo.toml", "go.mod", "build.gradle", "settings.gradle", "pom.xml", "cmakelists.txt",
    "justfile", "procfile", "vagrantfile", "jenkinsfile", "brewfile"})
#: …and the suffixes of code: a file the shell, an interpreter or a loader runs or imports by name
LOADED_SUFFIXES = frozenset({
    ".py", ".pyw", ".pyc", ".pyo", ".pyx", ".pth", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts",
    ".sh", ".bash", ".zsh", ".ksh", ".csh", ".fish", ".command", ".rb", ".pl", ".pm", ".php", ".phtml", ".phar",
    ".lua", ".tcl", ".ps1", ".psm1", ".bat", ".cmd", ".vbs", ".wsf", ".applescript", ".scpt", ".r", ".jl",
    ".go", ".rs", ".java", ".class", ".kt", ".kts", ".scala", ".groovy", ".gradle", ".swift", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".el", ".vim", ".awk", ".desktop", ".service", ".plist", ".mk", ".cmake",
    ".ini", ".cfg", ".toml", ".env"}) | NATIVE_CODE_SUFFIXES
#: the suffix appended to such a name
LANDING_SUFFIX = ".download"


def toolchain_loads(name: str) -> bool:
    """§4UY N8 r3: is ``name`` (its final component) one a toolchain loads by name — a dotfile (`.env`,
    `.npmrc`), a known build/test/package file (`TOOLCHAIN_NAMES`), or code (`LOADED_SUFFIXES`)?"""
    n = Path(str(name or "")).name.lower()
    if not n:
        return False
    if n.startswith(".") or n in TOOLCHAIN_NAMES:
        return True
    return Path(n).suffix in LOADED_SUFFIXES


def landing_name(name: str) -> str:
    """§4UY N8 r3: THE final name of a path-less download (the caller named no path; the URL or the server
    chose it): a name a toolchain loads (`toolchain_loads`) never lands as itself — `conftest.py` is saved as
    `conftest.py.download`. Every path-less download site asks this, after every name source is combined."""
    name = str(name or "")
    return name + LANDING_SUFFIX if toolchain_loads(name) else name


#: Content-Type → extension, for a download whose URL name has none (`arxiv.org/pdf/2401.12345`)
CONTENT_TYPE_EXT: Dict[str, str] = {
    "application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
    "image/webp": ".webp", "image/avif": ".avif", "image/svg+xml": ".svg", "application/zip": ".zip",
    "application/gzip": ".gz", "application/x-gzip": ".gz", "application/json": ".json", "text/csv": ".csv",
    "text/plain": ".txt", "text/html": ".html", "application/xml": ".xml", "text/xml": ".xml",
    "audio/mpeg": ".mp3", "audio/wav": ".wav", "video/mp4": ".mp4", "video/webm": ".webm",
    "application/epub+zip": ".epub", "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

_COMMENT_RE = re.compile(rb"^\s*(?:<!--.*?-->\s*)+", re.S)


def sniff_kind(head: bytes) -> str:
    """The kind of a body from its first bytes (`quarantine.sniff_type`), reading past leading whitespace and
    HTML comments — an error page that opens with 600 bytes of comments is still HTML."""
    from ..tools.quarantine import sniff_type
    b = bytes(head or b"")[:4096]       # r3: callers keep 64 KB (for `_office_export`); the kind is the first 4 KB
    if b.startswith(b"\xef\xbb\xbf"):
        b = b[3:]
    kind = sniff_type(b)
    if kind in ("text", "binary (unknown)", "empty"):
        stripped = _COMMENT_RE.sub(b"", b, count=1).lstrip()
        if stripped != b:
            if not stripped:
                return "html/xml" if b.lstrip().startswith(b"<!--") else kind
            k2 = sniff_type(stripped)
            if k2 in ("html/xml", "json-like text"):
                return k2
        if re.match(rb"\s*<(?:!doctype|html|head|body)\b", b, re.I):
            return "html/xml"
    if kind == "binary (unknown)":
        # a UTF-8 page cut mid-character at the head's end is still text
        try:
            b[: max(0, len(b) - 4)].decode("utf-8")
            if re.match(rb"\s*<(?:!doctype|html|\?xml)", b, re.I):
                return "html/xml"
        except UnicodeDecodeError:
            pass
    return kind


def body_mismatch(filename: str, content_type: Optional[str], head: bytes, total: Optional[int] = None) -> str:
    """Why a downloaded body is NOT the file its name promises — "" when it may be. Reads the first bytes,
    never only the Content-Type (servers mislabel both ways)."""
    suffix = Path(str(filename or "")).suffix.lower()
    ct = str(content_type or "").split(";")[0].strip().lower()
    kind = sniff_kind(head)
    if total == 0 or kind == "empty":
        return "the body is EMPTY (0 bytes)"
    if kind == "html/xml":
        # §4UY N13: an HTML body that is a WALL (a bot check, consent page, paywall stub) is not the file under
        # ANY name — saved as `.html` it read SUCCESS. THE page classifier judges it (`classify_page`).
        _title, _text = _head_title_text(head)
        _wall = challenge_reason(None, _title, _text)
        if _wall:
            return f"the body is an HTML page ({_wall}), not the {suffix or 'file'} content"
    expected = _EXPECTED_KIND.get(suffix)
    if suffix in _OFFICE_TEXT_OK and kind in ("html/xml", "text") and _office_export(suffix, kind, head):
        return ""
    if expected or suffix in _BINARY_ONLY:
        if kind == "html/xml" or (ct == "text/html" and kind in _TEXTLIKE):
            return f"the body is an HTML page, not a {suffix} file"
        if kind == "json-like text":
            return f"the body is a JSON/text message (likely an error), not a {suffix} file"
        if kind == "text" and expected:            # a format whose magic is known: text is never it
            return f"the body is plain text, not a {suffix} file"
        return ""
    if suffix in _TEXT_NOT_HTML and kind == "html/xml" and re.match(
            rb"\s*(?:<!--.*?-->\s*)*<(?:!doctype\s+html|html|head|body)\b", bytes(head or b""), re.I | re.S):
        return f"the body is an HTML page, not a {suffix} file"
    return ""


def kind_note(filename: str, head: bytes) -> str:
    """A note when the bytes are a DIFFERENT binary type than the name says (a PNG saved as .jpg) — usable,
    but worth saying. "" otherwise."""
    suffix = Path(str(filename or "")).suffix.lower()
    expected = _EXPECTED_KIND.get(suffix)
    kind = sniff_kind(head)
    if suffix in _OFFICE_TEXT_OK and kind in ("html/xml", "text") and _office_export(suffix, kind, head):
        return (f"the bytes are {'an HTML table/page' if kind == 'html/xml' else 'plain text (CSV/TSV)'}, not the "
                f"binary {suffix} format — Excel/Word open it; read it as text")
    if expected and kind not in _TEXTLIKE and kind not in ("binary (unknown)",) and kind != expected:
        return f"the bytes are {kind}, not {suffix}"
    return ""


def _head_title_text(head: bytes):
    """(title, visible text) of an HTML head — for judging a body by its first bytes."""
    try:
        soup = _soup(bytes(head or b""))
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for t in soup(["script", "style", "noscript", "template"]):
            t.decompose()
        return title, _text_of(soup.find("body") or soup)
    except Exception:  # noqa: BLE001
        return "", ""


# ── challenge / interstitial pages ──────────────────────────────────────
# §4UY N5: the ONE classifier of "a wall in front of the content". Three used to disagree on the same page
# (`search.is_bot_check_page`, `challenge_reason`, `browser_routes.blocked_page_reason`): the live 51-char
# HTTP-200 "Please wait while your request is being verified..." was a bot check to the first, a page to the
# other two, and the browser booked it STATUS: OK and cleared the host's memo. Every reader now asks
# `classify_page` (the search page reads, deep research, dark-web research, `helpers.fetch`, download's
# body check, the quarantine and the browser); the two old names are thin renderers of its answer.

#: a TITLE that is the wall itself — anchored, so a story whose title only MENTIONS one is a story; it counts
#: only over a page with little text (`_TITLE_WALL_MAX_CHARS`)
CHALLENGE_TITLE_RE = re.compile(
    r"^\s*(?:just a moment|security verification|attention required|"
    r"access (?:to this page has been )?denied|verify you are human|"
    r"are you a human|checking your browser|please enable cookies|"
    r"403 forbidden|401 authorization required|429 too many requests|"
    r"one moment, please|bot verification|"
    # §4UN-fetch r1: a bare BRAND word is a wall only as the WHOLE title — "Cloudflare outage takes down X"
    # and "CAPTCHA farms are booming" are news stories
    r"(?:cloudflare|ddos-guard|captcha|incapsula)\s*[.!]?\s*$|"
    # §4HV (req d121dcc6): Hetzner's "Heray" gate; word-bounded so "Security Checklist …" is an article
    r"security check\b|checking that you are not a robot|robot check\b|"
    r"human verification|"
    # §4UN-fetch: Anubis, Google/YouTube's EU consent wall, Imperva, the "one more step" interstitials
    r"making sure you(?:'|’)?re not a bot|before you continue(?: to (?:google|youtube)[\w ]*)?\s*$|"
    r"request unsuccessful|one more step\b|please wait while (?:we verify|your request)|"
    r"pardon our interruption|performing security verification)",
    re.IGNORECASE)
#: a URL that is the wall (Cloudflare's challenge tokens and path, Heray's proof-of-work, Google's consent host)
CHALLENGE_URL_RE = re.compile(r"__cf_chl|/cdn-cgi/challenge|/_ray/pow\b|^https?://consent\.(?:google|youtube)\.",
                              re.I)
#: body phrases of a page that stands in front of the content: a bot check / refusal. Like every wall phrase
#: they count only as the page's MAIN content (`_wall_phrase`), each kind under its own size bound.
#: The interstitial's own words about itself (review r2) — a challenge page explains itself at length
#: (Anubis: 1,276 chars; a Cloudflare "Just a moment…" page ~2k), so these count on pages under
#: `_CHALLENGE_TEXT_MAX`:
_BOT_INTERSTITIAL_RE = re.compile(
    r"making sure you(?:'|’)?re not a bot|anubis\b.*\bproof[- ]of[- ]work|"
    r"checking (?:if the site connection is secure|your browser(?: before accessing)?)|"
    r"verify(?:ing)? you are (?:a )?human|verifies you are not a bot|performing security verification|"
    r"request is being verified|just a moment(?:\.\.\.|…)|enable javascript and cookies to continue|"
    r"press (?:&|and) hold(?: (?:the|this) button)? to (?:confirm|verify|prove)",
    re.I | re.S)
#: …and the words a page may also say ABOUT a wall (a how-to: "Error 1020: Access Denied is shown when…") —
#: these count only on a short page (`_PHRASE_WALL_MAX_CHARS`), like the consent and paywall phrases
_BOT_PHRASE_RE = re.compile(
    r"incapsula incident id|request unsuccessful\. incapsula|ddos-guard|ddos protection by|"
    r"attention required!|are you a robot\??|unusual traffic from your computer network|"
    r"blocked by network security|error(?: code)?:?\s*1020\b|access denied[^\n]{0,40}\b1020\b",
    re.I | re.S)
#: a consent wall: Google/YouTube's, or a short page that is only a cookie/privacy choice
_CONSENT_PHRASE_RE = re.compile(
    r"before you continue to (?:google|youtube)|we use cookies and data to deliver and maintain google|"
    r"we value your privacy|(?:accept|agree to)(?: all)? cookies to continue|"
    r"manage (?:your )?(?:cookie|privacy) (?:preferences|choices|settings)", re.I)
#: a paywall stub: the page says the content is for subscribers / members
_PAYWALL_PHRASE_RE = re.compile(
    r"subscribe (?:now )?to (?:continue|keep) reading|to (?:continue|keep) reading,? (?:please )?"
    r"(?:subscribe|log in|sign in|become a (?:subscriber|member))|"
    r"this (?:article|content|story|page) is (?:only )?(?:available )?(?:exclusively )?(?:for|to) "
    r"(?:paid )?(?:subscribers|members)|already a subscriber\?|subscribers only\b", re.I)
#: a JavaScript-only shell (no content without a browser)
_JS_PHRASE_RE = re.compile(
    r"you need to enable javascript to run this app|please (?:turn on|enable) javascript|"
    r"javascript is (?:required|disabled)|this (?:site|page) requires javascript", re.I)
#: §4UN-fetch r1: a wall TITLE counts only over this little text (a 1,100-char news story titled "Access
#: denied: court rules…" or "CAPTCHA farms are booming" is the story)
_TITLE_WALL_MAX_CHARS = 600
#: §4UY review: a wall PHRASE counts only as the page's MAIN content — the page is SHORT (under this many
#: characters: a wall has little else to say) AND the phrase opens it (starts within its opening,
#: `_PHRASE_OPENING_CHARS` or the first half, whichever is longer). Review r2: the bound is per kind — an
#: interstitial's own words (`_BOT_INTERSTITIAL_RE`) under `_CHALLENGE_TEXT_MAX`, everything else under this. A long how-to ("Error 1020: Access Denied is shown when…", "Press and hold the volume up
#: button…") is content, and so is a short page whose footer links "privacy choices".
_PHRASE_WALL_MAX_CHARS = 800
_PHRASE_OPENING_CHARS = 160
_CHALLENGE_TEXT_MAX = 3000


def _wall_phrase(rx, body: str, size: int, max_chars: int = _PHRASE_WALL_MAX_CHARS) -> str:
    """The wall phrase of ``rx`` that is ``body``'s main content (a page under ``max_chars`` that opens with
    it), or ""."""
    if size >= max_chars:
        return ""
    lead = len(body) - len(body.lstrip())
    for m in rx.finditer(body):
        if m.start() - lead > max(_PHRASE_OPENING_CHARS, size / 2):
            continue
        if size >= _PHRASE_WALL_MAX_CHARS and _quoted(body, m.start()):
            continue               # review r2: a longer page that QUOTES the phrase is about the wall
        return m.group(0).strip()[:40]
    return ""


def _quoted(body: str, at: int) -> bool:
    """The match at ``at`` sits inside a quotation on its line (an odd number of quote marks before it)."""
    line = body[body.rfind("\n", 0, at) + 1:at]
    straight = sum(line.count(q) for q in "\"'") - len(re.findall(r"\w'\w", line))   # not an apostrophe
    # r3: a curly apostrophe between word characters ("you’re") closes no quotation
    curly = sum(line.count(q) for q in "“‘«") - (sum(line.count(q) for q in "”’»")
                                                  - len(re.findall(r"\w’\w", line)))
    return straight % 2 == 1 or curly > 0
#: a refusal status (the document answered it) — the page under it is not the content
REFUSAL_STATUSES = frozenset({401, 403, 407, 429, 451, 503})
#: under this many characters a page has no readable content (a JavaScript shell, an empty frame)
NEAR_EMPTY_CHARS = 120

#: the kinds `classify_page` returns
WALL_BOT, WALL_CONSENT, WALL_PAYWALL, WALL_JS, WALL_REFUSAL = (
    "bot challenge", "consent wall", "paywall", "javascript-only", "refusal")


def classify_page(status: Optional[int] = None, title: str = "", text: Optional[str] = None, url: str = "",
                  length: Optional[int] = None) -> tuple:
    """THE wall classifier: ``(kind, detail)`` — ``("", "")`` when the page may be read.

    ``text`` is the page's visible text (None when the reader did not get it); ``length`` the page's full
    text size when the reader knows it and ``text`` is only a preview (the browser). With neither, the size
    is UNKNOWN and a title decides nothing (§4LN: a screenshot of a page titled "Cloudflare…" banned the host).
    A refusal ``status`` decides on its own — the kind says whether the page under it is a challenge."""
    title = str(title or "")
    url = str(url or "")
    txt = None if text is None else str(text)
    try:
        size = int(length) if length is not None else (len(txt.strip()) if txt is not None else None)
    except (TypeError, ValueError):
        size = len(txt.strip()) if txt is not None else None
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if CHALLENGE_URL_RE.search(url):
        if "consent." in url.lower():
            return WALL_CONSENT, "consent url"
        return WALL_BOT, "challenge url"
    if size is not None and size < _TITLE_WALL_MAX_CHARS and CHALLENGE_TITLE_RE.match(title):
        return WALL_BOT, title.strip()[:40]
    body = (txt or "")[:_CHALLENGE_TEXT_MAX]
    if body and size is not None:
        for kind, rx, bound in ((WALL_BOT, _BOT_INTERSTITIAL_RE, _CHALLENGE_TEXT_MAX),
                                (WALL_BOT, _BOT_PHRASE_RE, _PHRASE_WALL_MAX_CHARS),
                                (WALL_CONSENT, _CONSENT_PHRASE_RE, _PHRASE_WALL_MAX_CHARS),
                                (WALL_PAYWALL, _PAYWALL_PHRASE_RE, _PHRASE_WALL_MAX_CHARS),
                                (WALL_JS, _JS_PHRASE_RE, _PHRASE_WALL_MAX_CHARS)):
            hit = _wall_phrase(rx, body, size, bound)
            if hit:
                return kind, hit
    if status in REFUSAL_STATUSES:
        return WALL_REFUSAL, f"HTTP {status}"
    return "", ""


def challenge_reason(status: Optional[int], title: str = "", text: Optional[str] = "", url: str = "",
                     length: Optional[int] = None) -> str:
    """Why this fetched page is a wall in front of the content — "" when it may be read (`classify_page`,
    worded for a fetch's error line). A refusal status counts here only over a short page (a fetch reader
    that got a long 503 body has the body)."""
    kind, detail = classify_page(status, title, text, url, length)
    if not kind:
        return ""
    if kind == WALL_REFUSAL:
        size = length if length is not None else len(str(text or "").strip())
        return f"{detail} refusal page" if size < _CHALLENGE_TEXT_MAX else ""
    if kind == WALL_JS:
        return "a JavaScript-only page (no content without a browser)"
    if kind == WALL_CONSENT:
        return "consent wall"
    if kind == WALL_PAYWALL:
        return f"paywall ({detail.lower()})"
    return f"bot challenge ({detail.lower()})"


def is_wall_text(text: str) -> bool:
    """A fetched page's TEXT alone is a wall (the readers that have no title, url or status)."""
    return bool(classify_page(None, "", str(text or ""), "")[0])


# ── readers ─────────────────────────────────────────────────────────────

# r4 — the MINIMAL conservative reader (four review rounds churned on exceptions: class-name rules took
# stories, tag rules took FAQ questions and article headers). Principle: never lose the page's content;
# leftover menu or banner text is the lesser harm. Only these go:
#: never visible text
_INVISIBLE_TAGS = ("script", "style", "noscript", "template", "svg", "iframe")
#: page furniture — removed only OUTSIDE an <article>/<main> (an article's own header/footer is content)
_FURNITURE_TAGS = ("nav", "header", "footer", "aside")
#: a consent banner (a cookie/consent word and a banner-type word in its class list or id: `govuk-cookie-
#: banner`, `js-cookie-consent`, `cookie-notice-container`, cookieconsent's `cc-window cc-banner`, Cookiebot)
#: — removed only when SHORT and without headings (a cookie/privacy POLICY page is long and titled)
_CONSENT_WORD_RE = re.compile(r"(?:^|[-_])(?:cookies?|consent|gdpr)(?:[-_]|$)|cookiebot|cookieconsent|"
                              r"cookiebanner|cookienotice|^cc[-_](?:window|banner|revoke)$", re.I)
_BANNER_WORD_RE = re.compile(r"(?:^|[-_])(?:banner|notice|consent|window|bar|popup|modal|container|dialog|"
                             r"overlay|wall|revoke)(?:[-_]|$)|cookiebot|cookieconsent|cookiebanner|cookienotice",
                             re.I)
_CONSENT_MAX_CHARS = 1500
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "article", "main"]
#: r5: furniture holding these is the article's title/byline (`<header class="article-header">` beside a div)
_TITLE_TAGS = ["h1", "h2", "time"]
#: r5: an <aside> that is a footnote is content
_FOOTNOTE_RE = re.compile(r"footnote|endnote", re.I)


def _is_main(t) -> bool:
    return t.name == "main" or str(t.get("role") or "").lower() == "main"


def _in_main_or_article(t) -> bool:
    return t.find_parent(lambda p: p.name == "article" or _is_main(p)) is not None


def _keep_furniture(t) -> bool:
    """r5: furniture that is content — it holds a title/byline, or it is a footnote."""
    if t.find(_TITLE_TAGS) is not None:
        return True
    if t.name == "aside":
        role = str(t.get("role") or "").lower()
        if role in ("note", "doc-footnote", "doc-endnote", "doc-endnotes"):
            return True
        return any(_FOOTNOTE_RE.search(x) for x in _tokens(t.attrs or {}))
    return False
#: r6: a <main> (or the page's <article>s together) is returned alone only when the cleaned text OUTSIDE it is
#: under this share of the page's cleaned text — anything more may be the story (24 teaser cards joined as
#: "the articles" hid a 1,500-char story in a plain div); otherwise the whole cleaned body is returned
_OUTSIDE_MAX_SHARE = 0.2
#: a cleaned page under this share of the RAW visible text lost its content — the raw text is returned
_CLEAN_MIN_SHARE = 0.2


def _soup(html: Any):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")


def _tokens(attrs):
    return [t.strip() for t in [str(attrs.get("id") or "")] + [str(c) for c in (attrs.get("class") or [])]
            if t and t.strip()]


def _is_consent_banner(attrs) -> bool:
    toks = _tokens(attrs)
    return (any(_CONSENT_WORD_RE.search(t) for t in toks) and any(_BANNER_WORD_RE.search(t) for t in toks))


def _text_of(node) -> str:
    return " ".join(node.get_text(separator=" ", strip=True).split()) if node is not None else ""


def extract_main_text(html: Any) -> str:
    """The page's main text (r4, minimal): scripts/styles, the nav/header/footer/aside OUTSIDE any
    <article>/<main>, and short heading-less consent banners removed — nothing else (forms, buttons and
    class-named blocks stay). An <article>/<main> holding half the cleaned text is returned alone. When the
    removals leave under a fifth of the raw visible text, the raw text is returned. Plain text, whitespace
    collapsed; "" for an empty page. ``html`` may be str or bytes (bytes honour a <meta charset>)."""
    try:
        soup = _soup(html)
    except Exception:  # noqa: BLE001
        return ""
    for t in soup(list(_INVISIBLE_TAGS)):
        t.decompose()
    raw_text = _text_of(soup.find("body") or soup)
    titles = []                         # r7: kept title furniture outside the region (page title, date)
    for t in soup(list(_FURNITURE_TAGS)):
        if t.decomposed or _in_main_or_article(t):
            continue
        if not _keep_furniture(t):
            t.decompose()
        elif t.find(_TITLE_TAGS) is not None and not any(p in titles for p in t.parents):
            titles.append(t)
    for t in soup.find_all(True):
        if t.decomposed or t.name in ("html", "body", "main", "article"):
            continue
        attrs = getattr(t, "attrs", None) or {}
        if (attrs and _is_consent_banner(attrs) and t.find(_HEADING_TAGS) is None
                and len(_text_of(t)) < _CONSENT_MAX_CHARS):
            t.decompose()
    body_text = _text_of(soup.find("body") or soup)
    if len(body_text) < _CLEAN_MIN_SHARE * len(raw_text):
        return raw_text
    mains = [n for n in soup.find_all(True) if _is_main(n) and not any(_is_main(p) for p in n.parents)]
    # r5: role="main" is <main>; several articles (a question and its answers) are read together, never alone
    region = mains or [n for n in soup.find_all("article") if n.find_parent("article") is None]
    inside = " ".join(t for t in (_text_of(n) for n in region) if t)
    # r6: ONE rule for both — the region alone only when what lies OUTSIDE it is under a fifth of the page
    if inside and len(body_text) - len(inside) < _OUTSIDE_MAX_SHARE * len(body_text):
        # r7: the page's title/date furniture outside the region goes in front of it
        head = " ".join(x for x in (_text_of(n) for n in titles if not n.decomposed) if x)
        return f"{head} {inside}" if head else inside
    return body_text


_DATE_META = ("article:published_time", "og:published_time", "datePublished", "pubdate", "publishdate",
              "date", "dc.date", "dc.date.issued", "dcterms.created", "sailthru.date", "parsely-pub-date",
              "article:modified_time", "og:updated_time", "dateModified", "last-modified", "dcterms.modified")
_ISO_DATE_RE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}(?:[T ][0-9:.+\-Z]*)?")


def _iso_date(v: Any) -> str:
    m = _ISO_DATE_RE.search(str(v or ""))
    return m.group(0)[:10] if m else ""


def page_meta(html: Any, headers: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """{"title", "published", "modified", "date"} of a page — ``date`` is the best single as-of date
    (modified, else published, else the Last-Modified header), "" when the page states none. ISO YYYY-MM-DD."""
    out = {"title": "", "published": "", "modified": "", "date": ""}
    try:
        soup = _soup(html)
    except Exception:  # noqa: BLE001
        return out
    t = soup.find("meta", attrs={"property": "og:title"}) or None
    title = (t.get("content") if t is not None else "") or (soup.title.get_text(" ", strip=True)
                                                            if soup.title else "")
    out["title"] = " ".join(str(title or "").split())[:200]
    pub = mod = ""
    for m in soup.find_all("meta"):
        key = str(m.get("property") or m.get("name") or m.get("itemprop") or "").strip()
        if not key:
            continue
        val = _iso_date(m.get("content"))
        if not val:
            continue
        k = key.lower()
        if any(x in k for x in ("modified", "updated")):
            mod = mod or val
        elif key in _DATE_META or k in [d.lower() for d in _DATE_META]:
            pub = pub or val
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(s.string or s.get_text() or "")
        except Exception:  # noqa: BLE001
            continue
        stack = [data]
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
            elif isinstance(d, dict):
                pub = pub or _iso_date(d.get("datePublished"))
                mod = mod or _iso_date(d.get("dateModified"))
                if "@graph" in d:
                    stack.append(d["@graph"])
    if not pub:
        tm = soup.find("time", attrs={"datetime": True})
        if tm is not None:
            pub = _iso_date(tm.get("datetime"))
    out["published"], out["modified"] = pub, mod
    lm = ""
    if headers:
        try:
            from email.utils import parsedate_to_datetime
            raw = next((v for k, v in dict(headers).items() if str(k).lower() == "last-modified"), None)
            if raw:
                lm = parsedate_to_datetime(str(raw)).date().isoformat()
        except Exception:  # noqa: BLE001
            lm = ""
    out["date"] = mod or pub or lm
    return out


def pdf_text(data: bytes, max_pages: int = 20, max_chars: int = 200_000) -> str:
    """The text of a PDF's first ``max_pages`` pages ("" when it has none — a scanned PDF — or cannot be
    read, and "" for a password-protected PDF). Pure and SYNCHRONOUS (it blocks on the parse); call it in a thread (`asyncio.to_thread`) from async code. Each call opens
    its own document, so concurrent calls do not share state."""
    if not data or not bytes(data[:5]).startswith(b"%PDF-"):
        return ""
    parts = []
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=bytes(data), filetype="pdf") as doc:
            if doc.needs_pass:                     # §4UN-fetch r1: a password-protected PDF has no readable text
                return ""
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                parts.append(page.get_text("text") or "")
                if sum(len(p) for p in parts) >= max_chars:
                    break
    except Exception:  # noqa: BLE001
        parts = []
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(bytes(data)))
            if reader.is_encrypted and not reader.decrypt(""):
                return ""
            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    break
                parts.append(page.extract_text() or "")
                if sum(len(p) for p in parts) >= max_chars:
                    break
        except Exception:  # noqa: BLE001
            return ""
    text = "\n".join(" ".join(p.split()) for p in parts if p and p.strip())
    return text[:max_chars]
