"""browser: hand back the page's own main image URL (§4JY).

Live failure 2026-09-22: asked to fetch a photo of a person, the agent could
not obtain one. It GUESSED Wikimedia thumbnail widths — `800px-…` and
`640px-…` both return HTTP 400 because only some widths are pre-rendered,
while `500px-…` returns 200 — then fell back to screenshotting the file PAGE
and passed a 1280x2290 picture of a web page to the image model as the
"photo". The page had declared the right image in `og:image` all along, and
that exact URL fetched 200 over Tor.

Two things must hold, and the second is the subtle one:
  * navigate AND extract_text surface the URL with its SOURCE, because
    `og:image` is the site's own declaration while `largest-rendered` is a
    guess the caller should verify;
  * the fallback ranks by RENDERED geometry. The naive "biggest <img>" reads
    HTML width/height, which are INTRINSIC — on the live page that selected a
    7651x5103 photo of a different politician entirely.
"""
import json
import os
import re
import shlex
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.tools.browser import tool_browser


def _make_sandbox_stub(output: str, exit_code: int = 0):
    stub = MagicMock()
    stub.last_command = None

    def _execute(cmd, timeout=300, **kwargs):
        stub.last_command = cmd
        return output, exit_code

    stub.execute = _execute
    return stub


async def _run(tmp_path, op, payload):
    stub = _make_sandbox_stub(f"[BROWSER_OK] {json.dumps(payload)}\n")
    return await tool_browser(operation=op, url="http://x/",
                              sandbox_dir=tmp_path, sandbox_manager=stub)


_OG = {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/K.jpg/960px-K.jpg",
       "source": "og:image"}
_GUESS = {"url": "https://site/photo.jpg", "source": "largest-rendered", "w": 640, "h": 480}


class TestTheUrlReachesTheModel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("op", ["navigate", "extract_text"])
    async def test_both_read_operations_surface_it(self, tmp_path, op):
        out = await _run(tmp_path, op, {"status": 200, "url": "http://x/", "title": "T",
                                        "text": "body", "length": 4, "truncated": False,
                                        "main_image": _OG})
        assert f"MAIN_IMAGE: {_OG['url']}" in out
        assert "og:image" in out
        # and it must say what to DO with it — the affordance that was missing
        assert "file_system" in out and "download" in out

    @pytest.mark.asyncio
    async def test_the_source_is_reported_so_a_guess_can_be_checked(self, tmp_path):
        out = await _run(tmp_path, "navigate", {"status": 200, "url": "http://x/",
                                                "title": "T", "main_image": _GUESS})
        assert "largest-rendered" in out
        assert "640x480" in out                      # rendered size, so it can be judged

    @pytest.mark.asyncio
    async def test_a_page_without_one_says_nothing(self, tmp_path):
        out = await _run(tmp_path, "navigate", {"status": 200, "url": "http://x/", "title": "T"})
        assert "MAIN_IMAGE" not in out

    @pytest.mark.asyncio
    async def test_a_malformed_entry_is_not_rendered(self, tmp_path):
        # The runner returns {} on any failure; a dict without a url must not
        # print a dangling "MAIN_IMAGE: None".
        out = await _run(tmp_path, "navigate", {"status": 200, "url": "http://x/",
                                                "title": "T", "main_image": {"source": "og:image"}})
        assert "MAIN_IMAGE" not in out

    @pytest.mark.asyncio
    async def test_the_page_text_still_renders_alongside_it(self, tmp_path):
        out = await _run(tmp_path, "extract_text", {"status": 200, "url": "http://x/", "title": "T",
                                                    "text": "Hello body", "length": 10,
                                                    "truncated": False, "main_image": _OG})
        assert "Hello body" in out and "--- TEXT ---" in out and "MAIN_IMAGE:" in out


class TestTheFallbackRanksByWhatIsShown:
    """The extraction runs as JS inside the page, so assert on the source the
    runner ships — the property that made the naive version pick the wrong
    person is that it must not trust HTML width/height attributes."""

    def _js(self):
        src = (os.path.join(os.path.dirname(__file__), '..', 'src', 'ghost_agent',
                            'tools', 'browser_runner.py'))
        import ast
        tree = ast.parse(open(src).read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_probe_main_image")
        return next(n.value.value for n in ast.walk(fn)
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str) and "querySelectorAll" in n.value.value)

    def test_it_prefers_the_sites_own_declaration(self):
        js = self._js()
        assert 'og:image' in js and 'twitter:image' in js
        # ⚠ Assert the BRANCH, not the word order: a first draft compared
        # js.index('og:image') < js.index('querySelectorAll'), which stays
        # true when the early return is disabled — the mutant that did
        # exactly that survived.
        assert re.search(r"if\s*\(\s*declared\s*\)\s*return", js), \
            "a declared og:image must return BEFORE the <img> scan"

    def test_the_fallback_measures_rendered_geometry(self):
        js = self._js()
        assert "getBoundingClientRect" in js
        # the trap: intrinsic attributes must NOT be what ranks candidates
        assert "naturalWidth" not in js
        assert re.search(r"r\.width\s*\*\s*r\.height", js), "area must come from the rect"

    def test_the_fallback_skips_what_is_never_a_photo(self):
        js = self._js()
        assert "data:" in js and "svg" in js
        assert "128" in js                      # icons/spacers/tracking pixels

    def test_urls_are_absolute(self):
        # A relative src is useless to file_system download.
        assert "new URL(" in self._js()


class TestTheRunnerActuallyShipsIt:
    """The formatter tests feed a hand-made payload, so they pass even if the
    runner never collects the image — the mutant that deleted the call from
    op_navigate survived until this existed."""

    def _fn(self, name):
        import ast
        src = os.path.join(os.path.dirname(__file__), '..', 'src', 'ghost_agent',
                           'tools', 'browser_runner.py')
        tree = ast.parse(open(src).read())
        return next(n for n in ast.walk(tree)
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == name)

    def _calls_probe(self, fn):
        import ast
        return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_probe_main_image"
                   for n in ast.walk(fn))

    @pytest.mark.parametrize("op", ["op_navigate", "op_extract_text"])
    def test_the_read_ops_collect_the_image(self, op):
        assert self._calls_probe(self._fn(op)), f"{op} never asks for the page's main image"

    @pytest.mark.parametrize("op", ["op_navigate", "op_extract_text"])
    def test_and_put_it_in_the_result_under_main_image(self, op):
        import ast
        assert any(isinstance(n, ast.Constant) and n.value == "main_image"
                   for n in ast.walk(self._fn(op)))


def test_the_schema_tells_the_model_not_to_guess_urls():
    from unittest.mock import MagicMock as MM
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MM()
    ctx.llm_client.image_gen_clients = ["x"]
    desc = next(t for t in get_active_tool_definitions(ctx)
                if t["function"]["name"] == "browser")["function"]["parameters"]["properties"]["operation"]["description"]
    assert "MAIN_IMAGE" in desc
    assert "guessing thumbnail widths" in desc      # names the exact failure
    assert "og:image" in desc and "largest-rendered" in desc
    assert "twitter:image" in desc          # it is a distinct source, not mislabelled og:image
    # …and the CUE that makes the model reach for it when it needs a picture.
    # Without this the mutant that gutted the lead sentence survived, because
    # the rest of the paragraph still mentioned MAIN_IMAGE.
    assert "picture on a page" in desc.lower()


# ============================================ the JS, EXECUTED (not read)
from tests.helpers import eval_js  # noqa: E402
from tests.test_browser_probe_js_execution import _arrow_from  # noqa: E402


def _pick(meta=None, imgs=(), href="https://site/page"):
    """Run the real _probe_main_image arrow fn against a fake DOM.

    Reading the JS proves nothing about what it does — the sibling probe's
    regex was broken in production for exactly that reason. `imgs` entries are
    {src, w, h, top} where w/h are RENDERED (getBoundingClientRect), which is
    the distinction the fallback turns on.
    """
    metas = ",".join(
        "{getAttribute(){return null;},content:%r,__sel:%r}" % (v, k)
        for k, v in (meta or {}).items()
    )
    images = ",".join(
        "{src:%r,currentSrc:'',getBoundingClientRect(){return {width:%d,height:%d,top:%d};}}"
        % (i["src"], i.get("w", 300), i.get("h", 300), i.get("top", 0))
        for i in imgs
    )
    harness = (
        "const __metas=[%s];\n"
        "const location={href:%r};\n"
        "const window={scrollY:0};\n"
        "const document={\n"
        "  querySelector:(sel)=>{ const m=__metas.find(x=>sel.includes(x.__sel)); return m||null; },\n"
        "  querySelectorAll:(sel)=>[%s]\n"
        "};\n"
        "const pick = %s;\n" % (metas, href, images, _arrow_from("_probe_main_image").rstrip().rstrip(";"))
    )
    return eval_js(harness, "pick()")


class TestTheHeuristicExecuted:
    def test_og_image_wins_and_is_labelled(self):
        r = _pick(meta={"og:image": "https://cdn/photo.jpg"},
                  imgs=[{"src": "https://cdn/other.jpg", "w": 900, "h": 900}])
        assert r["url"] == "https://cdn/photo.jpg"
        assert r["source"] == "og:image"

    def test_a_twitter_card_is_not_labelled_og_image(self):
        # Review finding: both meta tags returned source 'og:image', so the
        # schema's "og:image means the site's own og:image tag" was false.
        r = _pick(meta={"twitter:image": "https://cdn/card.jpg"})
        assert r["url"] == "https://cdn/card.jpg"
        assert r["source"] == "twitter:image"

    def test_og_image_is_preferred_over_a_twitter_card(self):
        r = _pick(meta={"twitter:image": "https://cdn/card.jpg",
                        "og:image": "https://cdn/og.jpg"})
        assert r["url"] == "https://cdn/og.jpg" and r["source"] == "og:image"

    def test_it_returns_before_scanning_images(self):
        # A huge <img> must NOT beat the site's own declaration.
        r = _pick(meta={"og:image": "https://cdn/small-but-declared.jpg"},
                  imgs=[{"src": "https://cdn/huge.jpg", "w": 4000, "h": 4000}])
        assert r["url"].endswith("small-but-declared.jpg")

    def test_without_a_declaration_the_biggest_rendered_wins(self):
        r = _pick(imgs=[{"src": "https://cdn/a.jpg", "w": 200, "h": 200},
                        {"src": "https://cdn/b.jpg", "w": 800, "h": 600}])
        assert r["url"].endswith("b.jpg") and r["source"] == "largest-rendered"
        assert r["w"] == 800 and r["h"] == 600

    def test_the_intrinsic_size_trap(self):
        """The live bug this guards: ranking by HTML width/height picked a
        7651x5103 photo of a DIFFERENT person, because those attributes are
        intrinsic. Rendered small must lose to rendered large."""
        r = _pick(imgs=[{"src": "https://cdn/wrong-person-huge-intrinsic.jpg", "w": 90, "h": 60},
                        {"src": "https://cdn/the-actual-lead-photo.jpg", "w": 640, "h": 480}])
        assert r["url"].endswith("the-actual-lead-photo.jpg")

    def test_near_the_top_is_preferred_over_far_below(self):
        r = _pick(imgs=[{"src": "https://cdn/lead.jpg", "w": 400, "h": 400, "top": 0},
                        {"src": "https://cdn/footer.jpg", "w": 430, "h": 430, "top": 9000}])
        assert r["url"].endswith("lead.jpg")

    @pytest.mark.parametrize("src", ["data:image/png;base64,AAA", "https://cdn/icon.svg",
                                     "https://cdn/logo.svg?v=2"])
    def test_non_photos_are_skipped(self, src):
        r = _pick(imgs=[{"src": src, "w": 900, "h": 900}])
        assert r == {} or not r.get("url")

    def test_icons_below_the_floor_are_skipped(self):
        r = _pick(imgs=[{"src": "https://cdn/tiny.jpg", "w": 64, "h": 64}])
        assert r == {} or not r.get("url")

    def test_a_relative_src_comes_back_absolute(self):
        # file_system download cannot use "/media/x.jpg".
        r = _pick(imgs=[{"src": "/media/x.jpg", "w": 500, "h": 500}],
                  href="https://site/a/b/page.html")
        assert r["url"] == "https://site/media/x.jpg"

    def test_a_page_with_nothing_usable_returns_empty(self):
        assert _pick() == {}


class TestHostilePagesCannotSteerTheDownload:
    """Review findings against the first cut of MAIN_IMAGE. The affordance told
    the model to download with NO `path`, so `file_system`'s auto-heal took the
    filename from the URL — a page declaring
    `<meta property="og:image" content="https://evil/x/main.py">` overwrote the
    agent's own main.py, which a later `execute` runs. And the og:image branch
    applied none of the `<img>` filters, so `data:`, `javascript:` and `file://`
    URLs were printed verbatim under a "download this" instruction."""

    @pytest.mark.asyncio
    async def test_the_line_names_a_safe_destination(self, tmp_path):
        out = await _run(tmp_path, "navigate", {"status": 200, "url": "http://x/", "title": "T",
                                                "main_image": {"url": "https://evil/x/main.py",
                                                               "source": "og:image"}})
        assert 'path="main_image' in out, "the model is not told where to put it"
        assert "main.py" not in out.split("MAIN_IMAGE_SOURCE")[1], "the page still names the file"
        assert "ALWAYS pass an explicit `path`" in out

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_ext_url,expected", [
        ("https://evil/x/main.py", "main_image.jpg"),
        ("https://cdn/a/photo.PNG", "main_image.png"),
        ("https://cdn/a/shot.webp", "main_image.webp"),
        ("https://cdn/a/x.sh", "main_image.jpg"),
    ])
    async def test_only_a_whitelisted_extension_survives(self, tmp_path, bad_ext_url, expected):
        out = await _run(tmp_path, "navigate", {"status": 200, "url": "http://x/", "title": "T",
                                                "main_image": {"url": bad_ext_url, "source": "og:image"}})
        assert f'path="{expected}"' in out

    @pytest.mark.parametrize("hostile", [
        "data:image/png;base64,AAAA",
        "javascript:alert(1)",
        "file:///etc/passwd",
    ])
    def test_a_declared_non_http_url_is_refused(self, hostile):
        assert _pick(meta={"og:image": hostile}) == {}

    def test_an_enormous_declared_url_is_refused(self):
        assert _pick(meta={"og:image": "https://cdn/" + "a" * 4000 + ".jpg"}) == {}

    def test_a_normal_declaration_still_works(self):
        r = _pick(meta={"og:image": "https://cdn/photo.jpg"})
        assert r["url"] == "https://cdn/photo.jpg"

    def test_the_probe_cannot_hang_the_operation(self):
        """It is a nicety; it must never cost the caller the page."""
        import ast
        from pathlib import Path as _P
        src = (_P(os.path.dirname(__file__)).parent / "src" / "ghost_agent"
               / "tools" / "browser_runner.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_probe_main_image")
        assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "wait_for"
                   for n in ast.walk(fn)), "page.evaluate is unbounded"
