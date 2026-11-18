"""§4UU-1 — load a web page headless RIGHT AFTER the write that changed it.

Measured (25 days, 38 owner coding turns): 21 turns drove the browser 120
times to test what they built, 27 of those calls failed, and a real bug
("build is not a function", req bcd86f13) surfaced only after 23 browser
calls. The WEB-EXEC arm already loads every written page headless — but only
after the reply, as a verdict the model never reads.

The rule: a successful `file_system` write / edit of a web file
(.html/.htm/.js/.mjs/.css) — the LAST such write of its tool batch, and not a
file still open for parts — loads the page it belongs to (the page itself, or
the `index.html` next to an edited script) with the same probe WEB-EXEC uses,
and the result is added to that write's tool result:

* `[PAGE CHECK] ✓` — loaded, no uncaught JS exception;
* `[PAGE CHECK] ✗` — the page throws on load, with the exceptions (and the
  console errors after them);
* nothing for a script / stylesheet with no page to load, and a short
  "could not be loaded here" for an HTML page the probe could not run.

The write keeps its own status: this is information, not a block. The note
names the page and the scripts it loads, and the claim arm
(`core/code_claims.py`) reads it back from the write's row — live and in the
census alike — as a RUN of each of them (a clean load) or a FAILED run (a
page that throws).

`GHOST_PAGE_CHECK=0` turns it off.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

#: a write to one of these may change what a page does on load
WEB_TRIGGER_EXTS = (".html", ".htm", ".js", ".mjs", ".css")
_HTML_EXTS = (".html", ".htm")

#: `<script src=…>` — quoted or not; never `data-src` (a lazy-load attribute runs nothing by itself)
_SCRIPT_SRC_RE = re.compile(r"<script\b[^>]*?(?<![\w-])src\s*=\s*(?:[\"']([^\"'#?]+)|([^\s>\"'#?]+))", re.I)
#: an inline `<script>` body (module imports inside it count)
_INLINE_SCRIPT_RE = re.compile(r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script>", re.I | re.S)
_MODULE_IMPORT_RE = re.compile(
    r"(?:\bimport\s+(?:[\w$*{}\s,]+\s+from\s+)?|\bimport\s*\(\s*|\bexport\s+[\w$*{}\s,]+\s+from\s+)"
    r"[\"'`](\.{1,2}/[^\"'`?#]+)")

def enabled() -> bool:
    return os.environ.get("GHOST_PAGE_CHECK", "1").strip().lower() not in ("0", "off", "false", "no")


def is_web_path(path) -> bool:
    return str(path or "").lower().endswith(WEB_TRIGGER_EXTS)


def is_html_path(path) -> bool:
    return str(path or "").lower().endswith(_HTML_EXTS)


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def loaded_names(page: Path, root: Path, *, depth: int = 4) -> set:
    """Basenames (casefold) of the page and of the local scripts it loads that EXIST: `<script src>` and,
    `depth` levels down, relative ES-module imports (from script files and inline module scripts). Only files
    inside `root` are read; a missing file is not named (it never ran)."""
    names = {page.name.casefold()}
    try:
        html = page.read_text(encoding="utf-8", errors="replace")[:400_000]
    except OSError:
        return names
    frontier = []
    for q, bare in _SCRIPT_SRC_RE.findall(html):
        src = q or bare
        if re.match(r"(?i)^(?:[a-z][a-z0-9+.\-]*:)?//", src):
            continue                                   # a remote script is not this request's code
        frontier.append((page.parent / src.lstrip("/")) if not src.startswith("/") else (root / src.lstrip("/")))
    for body in _INLINE_SCRIPT_RE.findall(html):
        frontier += [page.parent / rel for rel in _MODULE_IMPORT_RE.findall(body)]
    seen = set()
    for _ in range(max(1, depth)):
        nxt = []
        for f in frontier:
            key = str(f)
            if key in seen or not _inside(f, root) or not f.is_file():
                continue
            seen.add(key)
            names.add(f.name.casefold())
            try:
                text = f.read_text(encoding="utf-8", errors="replace")[:400_000]
            except OSError:
                continue
            nxt += [f.parent / rel for rel in _MODULE_IMPORT_RE.findall(text)]
        frontier = nxt
    return names


#: the check's own lines — whole lines, so file text the tool echoes elsewhere never reads as one
_CLEAN_LINE = "[PAGE CHECK] ✓ loaded '{pages}' headless: no uncaught JS exception. Ran: {names}."
_THROW_LINE = ("[PAGE CHECK] ✗ '{pages}' FAILS on load (an uncaught exception, or its own files did not load) — the "
               "page is broken; fix this before anything else. Ran: {names}.")
_LINE_RE = re.compile(r"^\[PAGE CHECK\] (✓|✗) .*? Ran: ([^\n]*?)\.$", re.M)


def note_for(result, wrote_html: bool, names=(), *, local_fetch: bool = False) -> str:
    """The text added to the write's tool result for the probe's `result` (`_execute_web_artifact`'s shape:
    `(pages, "")` clean, `(page, block)` throws, None could not run). `names` = the files the load ran."""
    listed = ", ".join(sorted({str(n) for n in names if n})) or "-"
    if result is None:
        if not wrote_html:
            return ""
        if local_fetch:                       # review r2: the serve advice must not out-rank the double-click case
            return ("\n\n[PAGE CHECK] not run — the page calls the network and no running sandbox service covers it. "
                    "If it is meant to be SERVED, start it with the `run` tool and exercise it over http before you say "
                    "it works; if it must open from disk, see the note below. Otherwise say it is untested.")
        return ("\n\n[PAGE CHECK] not run — the page could not be loaded headless here (it calls the network and "
                "no running sandbox service covers it, or the load itself failed). Start it as a server with the `run` "
                "tool and exercise it over http (browser interact, or its tests) before you say it works, or say it "
                "is untested.")
    pages, block = result
    if block:
        return "\n\n" + _THROW_LINE.format(pages=pages, names=listed) + "\n" + str(block).strip()
    return "\n\n" + _CLEAN_LINE.format(pages=pages, names=listed)


#: `fetch('sales.json')` / `fetch("./data/x.csv")` — a page's own data file, by a relative URL
_LOCAL_FETCH_RE = re.compile(r"""\bfetch\s*\(\s*(['"`])(?!https?:|//|/api/|\$\{)([^'"`\s?#]+\.(?:jsonl?|ndjson|csv|tsv|txt|md|xml|geojson))"""
                             r"""(?:[?#][^'"`]*)?\1""", re.I)
_LOCAL_FETCH_NOTE = (
    "\n[PAGE CHECK] note: the page fetches its own data file(s) {files} by a relative URL. Opened by double-click "
    "(file://) a browser blocks that fetch, so the page shows no data there — and an http check does not test that "
    "case. If it must open from disk, put the data in the page (inline JSON, or a `<script src=\"data.js\">` that "
    "sets a variable) and check it over file://; otherwise tell the user it needs a server.")


def local_fetch_note(page_texts) -> str:
    """ft live 11-18 (w4_dash): the note for pages that fetch a relative data file ("" when none does)."""
    files = []
    for text in page_texts or ():
        text = str(text or "")
        for m in _LOCAL_FETCH_RE.finditer(text):
            lead = text[text.rfind("\n", 0, m.start()) + 1:m.start()]
            lead = re.sub(r"'[^'\n]*'|\"[^\"\n]*\"|`[^`\n]*`", "''", lead)     # strings are not comments
            if re.search(r"(?<![:\\])//|/\*|<!--", lead) or lead.lstrip().startswith("*"):
                continue                      # a commented-out line (final review: a regex strip ate real code)
            if m.group(2) not in files:
                files.append(m.group(2))
    return _LOCAL_FETCH_NOTE.format(files=", ".join(f"'{f}'" for f in files[:4])) if files else ""


def checks_in(text) -> List[Tuple[bool, set]]:
    """The page checks a write's tool result carries: [(ok, {basename casefold, …}), …]."""
    m = _LINE_RE.search(str(text or ""))        # the FIRST only: what follows a ✗ line is the page's own text
    if not m:
        return []
    keys = {n.strip().casefold() for n in m.group(2).split(",") if n.strip() and n.strip() != "-"}
    return [(m.group(1) == "✓", keys)] if keys else []
