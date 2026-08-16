"""Web UI: the redesigned splash, and the iOS composer-at-the-top bug.

Both changes are operator-reported, 2026-08-16.

**1 — the splash.** The empty state carried a 2.2rem/700-weight wordmark
with 10px tracking, two stacked glows, an infinitely sweeping gradient,
and three suggestion chips ("What are you capable of?" …). It sits on top
of a LIVE animated constellation, so that was two animations and a button
row competing with the most striking thing on the screen. The operator's
verdict was "ugly"; the chips are gone and the wordmark is a hairline.

**2 — the iOS keyboard.** After a long reply, the composer jumped to the
very TOP of the screen, overlapping the status bar (operator screenshot).
Cause: TWO keyboard-avoidance mechanisms both firing.

  * `syncBodyHeight` pins `document.body` to `visualViewport.height`
    whenever `_vvKeyboardOpen()` — which ALREADY lifts the footer to sit
    just above the keyboard;
  * `--keyboard-height` ADDITIONALLY translates `#ui-layer` up by the
    keyboard height.

Applied together that is the same correction twice, and the composer
leaves the top of the screen. The transform is still needed for the
window the body pin does not cover (`_vvKeyboardOpen` also requires a
learned max height and a >80px drop), so the fix is mutual exclusion,
not deletion — the same "one definition, one mechanism" shape as §4BQ's
looks control.
"""

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"
_APP = _STATIC / "app.js"
_CSS = _STATIC / "style.css"
_HTML = _STATIC / "index.html"


def _strip_comments(js: str) -> str:
    """Code only — the history of a removed feature lives in comments and
    SHOULD. Asserting over raw text made this file's own explanation of
    what was deleted read as the deleted thing still being present."""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"^\s*//.*$", "", js, flags=re.M)


@pytest.fixture(scope="module")
def app_js():
    return _strip_comments(_APP.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def app_js_raw():
    return _APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css():
    return _CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html():
    return _HTML.read_text(encoding="utf-8")


class TestTheSuggestionChipsAreGone:
    def test_no_example_prompt_strings_survive(self, app_js):
        """The literal prompts the operator objected to."""
        for gone in ("What are you capable of",
                     "Summarise the latest files",
                     "Write a python script to sort CSV",
                     "EXAMPLE_PROMPTS"):
            assert gone not in app_js, gone

    def test_no_chip_element_is_built(self, app_js):
        assert "empty-hero-chip" not in app_js
        assert "empty-hero-chips" not in app_js

    def test_the_deletion_is_still_EXPLAINED_in_the_source(self, app_js_raw):
        """The prompts must be gone from the code and remembered in the
        comment — a silent deletion invites re-adding them."""
        assert "suggestion chips" in app_js_raw

    def test_no_chip_styling_is_left_behind(self, css):
        """Dead CSS for a removed element is how a 'removed' feature comes
        back looking half-styled."""
        assert "empty-hero-chip" not in css


class TestTheSplashIsQuiet:
    def test_the_wordmark_is_hairline_not_heavy(self, css):
        block = css[css.index(".empty-hero-title {"):]
        block = block[:block.index("}")]
        weight = re.search(r"font-weight:\s*(\d+)", block)
        assert weight and int(weight.group(1)) <= 300, block

    def test_the_infinite_gradient_sweep_is_gone(self, css):
        """A moving gradient on top of a moving background was two
        animations competing for the same glance."""
        assert "heroSweep" not in css
        assert "-webkit-text-fill-color: transparent" not in css

    def test_the_hero_still_renders_a_wordmark_rule_and_one_line(self, app_js):
        hero = app_js[app_js.index("function renderEmptyStateHero"):]
        hero = hero[:hero.index("function dismissEmptyStateHero")]
        assert "empty-hero-title" in hero
        assert "empty-hero-rule" in hero
        assert "empty-hero-sub" in hero

    def test_the_wordmark_is_split_per_letter_for_the_cascade(self, app_js):
        hero = app_js[app_js.index("function renderEmptyStateHero"):]
        hero = hero[:hero.index("function dismissEmptyStateHero")]
        assert "for (const ch of 'GHOST')" in hero

    def test_trailing_letter_tracking_is_trimmed(self, css):
        """Per-letter tracking pushes the word off centre unless the last
        letter's trailing space is pulled back."""
        assert ".empty-hero-title span:last-child" in css
        assert "margin-right: -0.42em" in css

    def test_motion_is_disabled_under_the_OS_preference(self, css):
        rm = css[css.index("@media (prefers-reduced-motion: reduce)"):]
        assert ".empty-hero-title span" in rm
        assert ".empty-hero-rule" in rm


class TestTheIOSComposerBug:
    """The screenshot: composer at the top of the screen, over the status
    bar, keyboard up."""

    def test_the_two_mechanisms_are_mutually_exclusive(self, app_js):
        """THE fix. `updateKeyboardOffset` must zero the transform exactly
        when `syncBodyHeight` will pin the body — otherwise the same
        correction is applied twice."""
        fn = app_js[app_js.index("const updateKeyboardOffset"):]
        fn = fn[:fn.index("window.visualViewport.addEventListener")]
        assert "if (_vvKeyboardOpen()) kb = 0;" in fn, fn[-600:]

    def test_the_body_pin_still_exists(self, app_js):
        """The fix relies on the pin doing the avoidance; if the pin were
        removed the transform would be the only mechanism and zeroing it
        would leave the composer UNDER the keyboard."""
        assert "document.body.style.height = window.visualViewport.height" in app_js

    def test_the_transform_is_still_wired_for_the_uncovered_window(self, css):
        """`_vvKeyboardOpen` needs a learned max height and a >80px drop,
        so a just-focused input still needs the transform."""
        layer = css[css.index("#ui-layer {"):]
        layer = layer[:layer.index("}")]
        assert "--keyboard-height" in layer
        assert "translateY" in layer

    def test_the_stale_offset_guard_is_intact(self, app_js):
        """Unfocusing must clear the offset, or a stale lift survives the
        keyboard closing."""
        fn = app_js[app_js.index("const updateKeyboardOffset"):]
        fn = fn[:fn.index("window.visualViewport.addEventListener")]
        assert "if (!isEditing()) kb = 0;" in fn


class TestCacheBustVersionsMovedTogether:
    """Stale-module skew: a browser holding old CSS with new JS renders a
    half-applied redesign. This repo has been bitten by it before."""

    def test_both_assets_were_bumped(self, html):
        css_v = re.search(r"style\.css\?v=([\d.]+)", html)
        app_v = re.search(r"app\.js\?v=([\d.]+)", html)
        assert css_v and app_v
        assert float(css_v.group(1)) >= 5.3, css_v.group(1)
        assert float(app_v.group(1)) >= 9.4, app_v.group(1)


class TestTheStylesheetIsWellFormed:
    def test_braces_balance(self, css):
        """The redesign replaced two large blocks; an orphaned keyframe
        tail survived the first attempt and would have broken every rule
        after it."""
        assert css.count("{") == css.count("}")

    def test_no_orphaned_keyframe_fragments(self, css):
        assert not re.search(r"\*/\s*\n\s*(to|from|\d+%)\s*\{", css)
