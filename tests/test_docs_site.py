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

# The site map lives in scripts/docs_site.py (2026-11-10: by feature, not by code module).
import sys as _sys
_sys.path.insert(0, str(_REPO / "scripts"))
import docs_site as _SITE  # noqa: E402
_FEATURE_PAGES = tuple(f"features/{slug}.html" for slug, _t, _s in _SITE.FEATURES)
_START_PAGES = tuple(p for p, _t in _SITE.START)
_TOP_DIRS = {"start", "features", "reference", "assets"}


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


@pytest.mark.parametrize("name", ("index.html",) + _START_PAGES + _FEATURE_PAGES + ("reference/index.html",))
def test_every_site_map_page_exists_and_marks_itself_current(name):
    page = _DOCS / name
    assert page.exists(), f"page {name} in the site map is missing"
    body = page.read_text()
    current = re.findall(r'<a class="current" href="([^"]+)"', body)
    assert len(current) == 1 and (page.parent / current[0]).resolve() == page.resolve(), (
        f"{name} should mark itself current, got {current}")


def test_every_page_is_what_the_build_makes_it():
    """The sidebar, the feature pages' reference lists, the home cards and the reference index are generated:
    `python scripts/docs_site.py build` after adding, moving or retitling a page."""
    stale = _SITE.build(False)
    assert not stale, "pages out of date — run `python scripts/docs_site.py build`:\n  " + "\n  ".join(stale)


def test_home_page_body_links_to_every_feature_and_the_start():
    """The sidebar is not discovery: the home page BODY reaches every feature and the getting-started page."""
    body = (_DOCS / "index.html").read_text().split("</aside>")[-1]
    missing = [n for n in _FEATURE_PAGES + ("start/getting-started.html",) if f'href="{n}"' not in body]
    assert not missing, "index.html body does not link to:\n  " + "\n  ".join(missing)


def test_every_page_links_back_to_the_home_page():
    stranded = []
    for page in _pages():
        prefix = "../" * (len(page.relative_to(_DOCS).parts) - 1)
        if f'href="{prefix}index.html"' not in page.read_text():
            stranded.append(_rel(page))
    assert not stranded, "pages with no link back to the home page:\n  " + "\n  ".join(stranded)


def test_pages_live_only_where_the_site_map_puts_them():
    """index.html at the top; everything else under start/, features/ or reference/<feature>/."""
    slugs = {slug for slug, _t, _s in _SITE.FEATURES}
    stray = []
    for page in _DOCS.rglob("*"):
        if not page.is_file() or page.suffix not in (".html", ".md"):
            continue
        parts = page.relative_to(_DOCS).parts
        if parts == ("index.html",) or parts[0] in ("start", "features", "assets"):
            continue
        if parts[0] == "reference" and (parts == ("reference", "index.html") or
                                        (len(parts) == 3 and parts[1] in slugs)):
            continue
        stray.append(_rel(page))
    assert not stray, "pages outside the site map:\n  " + "\n  ".join(sorted(stray))


def test_every_feature_page_has_its_sections():
    missing = []
    for name in _FEATURE_PAGES:
        body = (_DOCS / name).read_text()
        for anchor in ("what", "try", "how", "where", "limits"):
            if f'id="{anchor}"' not in body:
                missing.append(f"{name}#{anchor}")
    assert not missing, "feature pages missing a section:\n  " + "\n  ".join(missing)


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


def test_readme_doc_links_point_at_pages_that_exist():
    """The README links into the published site by URL, so a page rename
    breaks it silently — the site keeps working and only the README rots."""
    readme = _REPO / "README.md"
    assert readme.exists()
    base = "https://vventirozos.github.io/AgentGhost/"
    broken = []
    for url in re.findall(rf"\(({re.escape(base)}[^)\s]*)\)", readme.read_text()):
        rel = url[len(base):] or "index.html"
        if not (_DOCS / rel).exists():
            broken.append(url)
    assert not broken, "README links to missing doc pages:\n  " + "\n  ".join(broken)


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
