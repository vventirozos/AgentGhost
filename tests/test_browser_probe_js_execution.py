"""§5 — the pre-interaction probe JS is EXECUTED under node, not string-matched.

The embedded-script-string-trap: `_probe_pre_interaction`'s page JS
(browser_runner.py) was only ever asserted by `"_probe_pre_interaction" in
src` and driven with a fake `page.evaluate` that ignores the JS string. Two
independent mutations proved it unpinned (lens B/C): corrupt the KW regex, or
break the visibility helper, and every browser test stayed green. This
extracts the arrow function and RUNS it under node against DOM fixtures — the
discipline `tests/helpers.py::eval_js` exists for.
"""

import re
from pathlib import Path

from tests.helpers import eval_js

_RUNNER = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "tools" / "browser_runner.py").read_text(encoding="utf-8")


def _probe_arrow() -> str:
    """The `() => {...}` arrow fn literal from _probe_pre_interaction."""
    m = re.search(r'js = """(\(\) => \{.*?return \{ pre_interaction.*?\};?\s*)"""',
                  _RUNNER, re.S)
    assert m, "could not locate the probe arrow fn in browser_runner.py"
    return m.group(1)


def _run(elements):
    """Run the probe against a fixture of {text, w, h, visibility, display,
    opacity} elements; returns {pre_interaction, controls}."""
    fixture = ",".join(
        "{innerText:%r,textContent:'',_s:{visibility:%r,display:%r,opacity:%r},"
        "getBoundingClientRect(){return {width:%d,height:%d};}}"
        % (e["text"], e.get("visibility", "visible"), e.get("display", "block"),
           str(e.get("opacity", "1")), e.get("w", 100), e.get("h", 40))
        for e in elements
    )
    harness = (
        "const getComputedStyle = (el) => el._s;\n"
        "const document = { querySelectorAll: (sel) => [%s] };\n"
        "const probe = %s;\n" % (fixture, _probe_arrow().rstrip().rstrip(";"))
    )
    return eval_js(harness, "probe()")


def test_a_visible_start_control_is_detected():
    r = _run([{"text": "Click to Play"}])
    assert r["pre_interaction"] is True
    assert r["controls"] == ["Click to Play"]


def test_a_hidden_control_is_ignored():
    for hidden in ({"display": "none"}, {"visibility": "hidden"},
                   {"opacity": "0"}):
        r = _run([{"text": "Start", **hidden}])
        assert r["pre_interaction"] is False, (
            f"a control hidden by {hidden} was treated as a live start "
            f"control — the probe's vis() helper is broken")


def test_a_tiny_control_is_ignored():
    r = _run([{"text": "Play", "w": 2, "h": 2}])
    assert r["pre_interaction"] is False


def test_long_text_is_not_a_control():
    r = _run([{"text": "Start " + "x" * 90}])
    assert r["pre_interaction"] is False


def test_restart_and_start_your_journey_do_NOT_match():
    # The keywords are anchored (^start$); these must not trip it.
    for txt in ("Restart", "Start your journey", "Get started here now"):
        r = _run([{"text": txt}])
        assert r["pre_interaction"] is False, f"{txt!r} wrongly matched"


def test_the_keyword_regex_actually_matches_the_documented_controls():
    """The mutation that broke the KW regex survived every string-match test;
    this fails on it."""
    for txt in ("Click to Play", "Press Start", "Tap to play", "Play",
                "Begin", "Enter Game"):
        r = _run([{"text": txt}])
        assert r["pre_interaction"] is True, (
            f"the probe no longer recognises {txt!r} — the KW regex is broken")


def test_a_real_loading_screen_matches_but_download_words_do_not():
    """§5 lens B MINOR: `loading` was unanchored, so "Downloading files",
    "Reloading", "Uploading" all tripped the loading-screen warning. Anchored
    to a word boundary: a genuine "Loading..." still matches; substrings do
    not."""
    assert _run([{"text": "Loading..."}])["pre_interaction"] is True
    for word in ("Downloading files", "Reloading page", "Uploading photo"):
        assert _run([{"text": word}])["pre_interaction"] is False, (
            f"{word!r} wrongly tripped the loading-screen detector")
