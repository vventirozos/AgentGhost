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


def _arrow_from(func_name: str) -> str:
    """The `() => {...}` arrow fn literal assigned inside ``func_name``.

    ⚠ Scoped to ONE function on purpose. This used to be a file-wide regex
    anchored on the literal js-assignment opener plus `return { pre_interaction`,
    which silently spanned two functions the moment another JS probe was added
    ABOVE this one — node then received the tail of one function plus the head
    of the next and failed to parse. Locating the assignment inside the target
    function's own AST cannot drift that way.
    """
    import ast
    fn = next((n for n in ast.walk(ast.parse(_RUNNER))
               if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
               and n.name == func_name), None)
    assert fn is not None, f"{func_name} not found in browser_runner.py"
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and node.value.value.lstrip().startswith("()")):
            return node.value.value
    raise AssertionError(f"no arrow-fn JS literal assigned in {func_name}")


def _probe_arrow() -> str:
    """The `() => {...}` arrow fn literal from _probe_pre_interaction."""
    return _arrow_from("_probe_pre_interaction")


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


# ---------------------------------------------------------------- the class
def test_no_embedded_js_literal_loses_its_escapes():
    """R1 enumeration, not a one-site fix (§4JY).

    `_probe_pre_interaction`'s JS was a NON-raw Python string, so the regex's
    `\\b` word boundaries evaluated to U+0008 BACKSPACE and `\\bloading\\b`
    could never match — the loading-screen false-positive the probe exists to
    prevent. It survived because the test read the SOURCE TEXT (where the
    backslash is still there) while production ran the evaluated string.

    Any JS literal that Python has silently rewritten shows up as a control
    character in the evaluated value. Walk every string constant in the runner
    that looks like page JS and refuse them all.
    """
    import ast
    bad = []
    for node in ast.walk(ast.parse(_RUNNER)):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        v = node.value
        if "=>" not in v and "document." not in v:      # not page JS
            continue
        ctrl = {c for c in v if ord(c) < 32 and c not in "\n\r\t"}
        if ctrl:
            bad.append((getattr(node, "lineno", "?"), sorted(hex(ord(c)) for c in ctrl)))
    assert not bad, (
        "embedded JS with Python-eaten escapes (use an r-string): "
        f"{bad} — e.g. a regex \\\\b became a backspace, so it matches nothing"
    )
