"""§4JP — the other two fd89fd6d fixes: the rewrite hint and the WebGL note.

Req fd89fd6d rewrote a 15–30 KB file from scratch seven times (126 KB
generated, ~15 of its 30 minutes) while `replace` was used twice; and it
abandoned a WebGL2 renderer because its own page printed "WebGL2
required" — the sandbox browser HAS WebGL2 (ANGLE on SwiftShader, measured
through the real runner with the proxy and the persistent profile).

World where each pin fails: the hint fires before the third rewrite, on a
small file, on a first write, or never; it leaks across requests; the
WebGL note stops firing on a page-side denial or fires on a healthy page;
the tool description stops stating the capability.
"""
import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.browser import WEBGL_CAPABILITY_NOTE, webgl_denial_note
from ghost_agent.tools.file_system import REWRITE_HINT_AT, REWRITE_HINT_MIN_BYTES, tool_write_file
from ghost_agent.utils.logging import request_id_context

BIG = "<!doctype html><html><body>" + ("<p>sponza atrium global illumination</p>\n" * 250) + "</body></html>\n"   # > 8 KB, no scripts (the post-write node check is per script block)


@pytest.fixture
def sandbox(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def _own_request():
    tok = request_id_context.set("rw-test-req")
    fs._REWRITES.clear()
    yield
    fs._REWRITES.clear()
    request_id_context.reset(tok)


async def test_hint_from_the_third_full_rewrite_of_a_large_existing_file(sandbox):
    assert len(BIG) >= REWRITE_HINT_MIN_BYTES
    r0 = await tool_write_file("sponza_renderer.html", BIG, sandbox)              # first write: file did not exist
    r1 = await tool_write_file("sponza_renderer.html", BIG + "a", sandbox)        # rewrite #1
    r2 = await tool_write_file("sponza_renderer.html", BIG + "b", sandbox)        # rewrite #2
    r3 = await tool_write_file("sponza_renderer.html", BIG + "c", sandbox)        # rewrite #3 → hint
    r4 = await tool_write_file("sponza_renderer.html", BIG + "d", sandbox)        # and every one after
    assert all("SUCCESS" in str(r) for r in (r0, r1, r2, r3, r4))
    assert "NOTE: this is full rewrite" not in str(r0) + str(r1) + str(r2)
    assert f"full rewrite #{REWRITE_HINT_AT} of an existing" in str(r3) and "operation='edit'" in str(r3)
    assert f"full rewrite #{REWRITE_HINT_AT + 1}" in str(r4)


async def test_small_files_and_other_paths_do_not_count(sandbox):
    small = "x = 1\n" * 50                                                        # < 8 KB
    for _ in range(5):
        r = await tool_write_file("tiny.py", small, sandbox)
    assert "full rewrite" not in str(r)
    await tool_write_file("a.html", BIG, sandbox)
    for _ in range(3):
        await tool_write_file("a.html", BIG + "!", sandbox)
    r = await tool_write_file("b.html", BIG, sandbox)                           # a different path: its own count
    assert "full rewrite" not in str(r)


async def test_the_count_is_per_request(sandbox):
    await tool_write_file("f.html", BIG, sandbox)
    for _ in range(3):
        r = await tool_write_file("f.html", BIG + "!", sandbox)
    assert "full rewrite #3" in str(r)
    tok = request_id_context.set("another-request")
    try:
        r = await tool_write_file("f.html", BIG + "?", sandbox)
        assert "full rewrite" not in str(r)                                       # the new request starts at 1
    finally:
        request_id_context.reset(tok)


@pytest.mark.parametrize("text, fires", [
    ("WebGL2 required", True),
    ("Your browser does not support WebGL", False),           # "does not support WebGL" — not in the denial grammar
    ("WebGL not supported by your browser", True),
    ("WebGL unavailable — falling back to CPU", True),
    ("Sponza Atrium — rendering 3 FPS, 307 objects", False),
    ("", False),
])
def test_webgl_note_fires_only_on_a_page_side_denial(text, fires):
    note = webgl_denial_note(text)
    assert (note == WEBGL_CAPABILITY_NOTE) is fires
    if fires:
        assert "DOES have WebGL2" in note and "SwiftShader" in note


def test_the_browser_tool_states_the_capability():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    browser = next(d for d in TOOL_DEFINITIONS if (d.get("function") or d).get("name") == "browser")
    desc = (browser.get("function") or browser)["description"]
    assert "HAS WebGL2" in desc and "SwiftShader" in desc
