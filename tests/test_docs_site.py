"""Guards for the docs/ site.

The documentation drifted badly before the 2026-08-04 rebuild: dead internal
links, pages for modules that no longer existed, and whole packages with no
coverage at all. Nothing caught any of it, because nothing checked.

These tests are cheap (pure file parsing, no browser) and pin the things that
silently rot:

  * internal links resolve;
  * the guide pages all carry the same navigation;
  * every reference page links back to the guide, so a reader arriving from a
    search engine is never stranded in expert material;
  * every ``<table>`` sits inside a ``.scroll`` wrapper — without one a wide
    table pushes the whole page sideways on a phone (80 pages did);
  * no NEW orphan page appears for a module that has been deleted.
"""
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest


_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_SRC = _REPO / "src" / "ghost_agent"

# The 12 hand-written guide pages. Everything else under docs/ is reference.
_GUIDE_PAGES = (
    "index.html", "getting-started.html", "interfaces.html", "how-it-works.html",
    "memory.html", "tools.html", "self-improvement.html", "safety.html",
    "configuration.html", "troubleshooting.html", "glossary.html", "reference.html",
)

# Pages that describe a module that is gone, kept only so old links don't 404.
# A page appearing here is a decision; a page NOT here whose module vanished is
# drift, and `test_no_new_orphan_reference_pages` fails on it.
_KNOWN_TOMBSTONES = {
    "core/agent_qwen.html",     # deleted 2026-07-27, page is an explicit tombstone
    "core/workspace_model.html",  # code moved to workspace/model.py
    "core/delegation.html",     # covers core/subagent.py + core/jobs.py + tools/delegate.py
    "tools/qwen_bridge.html",   # module removed
    "tools/memory_tools.html",  # covers tools/memory.py
}

# Doc directories that mirror a source package one-to-one.
_MIRRORED = ("core", "memory", "tools", "sandbox")


def _pages():
    return sorted(p for p in _DOCS.rglob("*.html") if p.parent.name != "assets")


def _strip_code(text: str) -> str:
    """Drop <code>/<pre> spans so escaped href examples aren't read as links."""
    text = re.sub(r"<code>.*?</code>", "", text, flags=re.S)
    return re.sub(r"<pre[^>]*>.*?</pre>", "", text, flags=re.S)


def _rel(path: Path) -> str:
    return path.relative_to(_DOCS).as_posix()


def test_docs_directory_exists():
    assert _DOCS.is_dir(), "docs/ is the published documentation set"
    assert (_DOCS / "assets" / "style.css").exists(), "the shared theme must exist"


def test_no_broken_internal_links():
    broken = []
    for page in _pages():
        body = _strip_code(page.read_text())
        for href in re.findall(r'href="([^"#][^"]*)"', body):
            if href.startswith(("http://", "https://", "mailto:", "//")):
                continue
            target = href.split("#")[0]
            if not target:
                continue
            if not (page.parent / target).resolve().exists():
                broken.append(f"{_rel(page)} -> {href}")
    assert not broken, "broken internal links:\n  " + "\n  ".join(broken)


@pytest.mark.parametrize("name", _GUIDE_PAGES)
def test_guide_page_exists_and_has_full_nav(name):
    page = _DOCS / name
    assert page.exists(), f"guide page {name} is missing"
    body = page.read_text()
    for other in _GUIDE_PAGES:
        assert f'href="{other}"' in body, f"{name} does not link to {other}"
    # Exactly one entry is marked current, and it is this page.
    current = re.findall(r'<a class="current" href="([^"]+)"', body)
    assert current == [name], f"{name} should mark itself current, got {current}"


def test_home_page_body_links_to_every_guide_page():
    """The sidebar is not discovery.

    Every guide page was reachable from every sidebar, yet `interfaces.html`
    was still hard to find because the home page's own "Start here" grid
    omitted it — a reader working down the page never saw it. So the home
    page BODY, not just its nav, has to reach all of them.
    """
    home = (_DOCS / "index.html").read_text()
    body = home.split("</aside>")[-1]
    missing = [name for name in _GUIDE_PAGES
               if name != "index.html" and f'href="{name}"' not in body]
    assert not missing, (
        "index.html body (outside the sidebar) does not link to:\n  "
        + "\n  ".join(missing))


def test_every_reference_page_links_back_to_the_guide():
    stranded = []
    for page in _pages():
        rel = _rel(page)
        if rel in _GUIDE_PAGES:
            continue
        body = page.read_text()
        depth = len(page.relative_to(_DOCS).parts) - 1
        prefix = "../" * depth
        if f'href="{prefix}index.html"' not in body:
            stranded.append(rel)
    assert not stranded, (
        "reference pages with no link back to the guide:\n  " + "\n  ".join(stranded))


def test_every_table_is_inside_a_scroll_wrapper():
    """A bare <table> makes the whole page scroll sideways on a phone."""
    bare = []
    for page in _pages():
        body = page.read_text()
        for m in re.finditer(r"<table(?=[\s>])[^>]*>", body):
            before = body[max(0, m.start() - 60):m.start()].rstrip()
            if not before.endswith('<div class="scroll">'):
                line = body[:m.start()].count("\n") + 1
                bare.append(f"{_rel(page)}:{line}")
    assert not bare, (
        'tables not wrapped in <div class="scroll">:\n  ' + "\n  ".join(bare))


def test_html_div_tags_are_balanced():
    class _Counter(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth = 0

        def handle_starttag(self, tag, attrs):
            if tag == "div":
                self.depth += 1

        def handle_endtag(self, tag):
            if tag == "div":
                self.depth -= 1

    unbalanced = []
    for page in _pages():
        counter = _Counter()
        counter.feed(page.read_text())
        if counter.depth != 0:
            unbalanced.append(f"{_rel(page)} ({counter.depth:+d})")
    assert not unbalanced, "unbalanced <div>:\n  " + "\n  ".join(unbalanced)


def test_no_new_orphan_reference_pages():
    """A page whose module was deleted is drift unless it's a known tombstone."""
    orphans = []
    for sub in _MIRRORED:
        doc_dir = _DOCS / sub
        src_dir = _SRC / sub
        if not (doc_dir.is_dir() and src_dir.is_dir()):
            continue
        for page in sorted(doc_dir.glob("*.html")):
            rel = f"{sub}/{page.name}"
            if rel in _KNOWN_TOMBSTONES:
                continue
            if not (src_dir / f"{page.stem}.py").exists():
                orphans.append(rel)
    assert not orphans, (
        "reference pages for modules that no longer exist — document the move or "
        "add to _KNOWN_TOMBSTONES:\n  " + "\n  ".join(orphans))


def test_tombstone_list_stays_accurate():
    """A tombstone whose module came BACK should leave the list."""
    resurrected = []
    for rel in sorted(_KNOWN_TOMBSTONES):
        sub, name = rel.split("/")
        if (_SRC / sub / f"{Path(name).stem}.py").exists():
            resurrected.append(rel)
    assert not resurrected, (
        "these modules exist again — drop them from _KNOWN_TOMBSTONES:\n  "
        + "\n  ".join(resurrected))


def test_theme_is_self_contained():
    """No external requests: the docs fail closed the same way the agent does."""
    css = (_DOCS / "assets" / "style.css").read_text()
    assert "@import" not in css, "the theme must not pull a remote stylesheet or font"
    for page in _pages():
        body = page.read_text()
        remote = re.findall(
            r'<(?:link|script|img)[^>]+(?:href|src)="(https?://[^"]+)"', body)
        assert not remote, f"{_rel(page)} loads external resources: {remote}"


def test_images_resolve_and_are_sized():
    """A missing image is invisible until someone loads the page; an unsized
    one reflows the article as it arrives."""
    problems = []
    for page in _pages():
        for tag in re.findall(r"<img\b[^>]*>", page.read_text()):
            src = re.search(r'src="([^"]+)"', tag)
            if not src:
                problems.append(f"{_rel(page)}: <img> with no src")
                continue
            if not (page.parent / src.group(1)).resolve().exists():
                problems.append(f"{_rel(page)} -> missing {src.group(1)}")
            for attr in ("width", "height", "alt"):
                if f"{attr}=" not in tag:
                    problems.append(f"{_rel(page)}: {src.group(1)} has no {attr}")
    assert not problems, "image problems:\n  " + "\n  ".join(problems)
