"""WCAG contrast of the interface's quiet metadata text (2026-09-04).

⚠ WHY THIS FILE EXISTS. A headless audit of the live UI measured three
text styles below the WCAG AA 4.5:1 floor, and all three were metadata
the operator reads constantly rather than decoration:

    .session-meta      ~3.1:1 at 9.9px   "1h ago · 48 msgs"
    .msg-time          ~3.2:1 at 9.6px   per-message timestamp
    #planner-monologue ~3.6:1 at 12.8px  live planner narration

None was pinned, because the suite's existing style assertions match
TEXT ("the rule contains rgba(...)") and a text match cannot tell 0.55
from 0.75 — it only tells you somebody wrote *an* rgba. So this file
RECOMPUTES the ratio from whatever value is shipped. Lowering an alpha
back toward the old value fails here with the number it produced.

⚠ WHAT THIS TEST CANNOT PROVE, and why it is still worth having. Both
metadata styles sit on TRANSLUCENT surfaces (`#session-rail` is
`rgba(8,5,14,.62)`, `.message.agent` is `rgba(22,13,38,.20)`) with a
`backdrop-filter` over the animated face. The real backdrop therefore
varies frame to frame, and a brighter face REDUCES the contrast of
light-grey text further. The first draft of this file calibrated
against my own in-browser measurements and failed on arrival — the two
selectors declare the SAME colour yet measured 3.09 and 3.19, because
the face was lit differently behind each. That discrepancy is the
finding, not a rounding error: there is no single true number here.

So the model composites over the DECLARED surfaces only, which is the
deterministic part of the stack and the *optimistic* bound. Passing is
necessary, not sufficient. It still catches the defect it was written
for: the pre-fix values fail even this generous model, by a wide
margin, which `test_the_shipped_values_would_fail_if_reverted` pins.

Calibration uses published WCAG reference pairs rather than my own
field measurements, for the same reason (R6: verify the instrument can
fail before trusting that it passed).

#planner-monologue additionally had a SEMANTIC defect: it was
`#c22745`, i.e. `--error-color` exactly. Ordinary planner narration was
painted in the colour reserved for failure. That is pinned separately
and does not depend on the ratio.
"""

import re
from pathlib import Path

import pytest

_CSS_PATH = (Path(__file__).resolve().parent.parent
             / "interface" / "static" / "style.css")

# WCAG 2.x AA for text below 18pt / 14pt-bold. Every element here is
# under 13px, so the large-text exemption (3:1) does not apply to any
# of them.
_AA = 4.5

# selector -> the declared surface stack beneath it, outermost LAST.
# `#000000` is `--bg-color`, the page ground under everything.
_SURFACES = {
    ".session-meta":      ["rgba(8, 5, 14, 0.62)", "#000000"],     # #session-rail
    ".msg-time":          ["rgba(22, 13, 38, 0.20)", "#000000"],   # .message.agent
    "#planner-monologue": ["#000000"],                             # header, no fill
}


@pytest.fixture(scope="module")
def css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


def _rule_body(css: str, selector: str) -> str:
    """Body of the FIRST `selector { … }`, brace-matched."""
    m = re.search(re.escape(selector) + r"\s*\{", css)
    assert m, f"{selector} not found in style.css"
    depth, i = 1, m.end()
    while i < len(css) and depth:
        depth += (css[i] == "{") - (css[i] == "}")
        i += 1
    return css[m.end():i - 1]


def _declared_color(css: str, selector: str) -> str:
    """The `color:` declaration of a rule, comments stripped.

    Comment stripping is load-bearing: the fixes ship with comments
    quoting the OLD rgba values as evidence, and a naive scan would read
    the documented defect instead of the shipped value — the test would
    then fail on a correct file and pass on a reverted one.
    """
    body = re.sub(r"/\*.*?\*/", "", _rule_body(css, selector), flags=re.S)
    m = re.search(r"(?<![-\w])color\s*:\s*([^;]+);", body)
    assert m, f"{selector} declares no color"
    return m.group(1).strip()


def _srgb_to_linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(rgb) -> float:
    r, g, b = (_srgb_to_linear(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def parse_color(value: str):
    """-> (r, g, b, a). Accepts #rgb / #rrggbb / rgb() / rgba()."""
    value = value.strip()
    m = re.match(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$", value)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0)
    m = re.match(r"rgba?\(([^)]+)\)$", value)
    assert m, f"unparseable color: {value!r}"
    parts = [float(p) for p in m.group(1).replace("/", ",").split(",")]
    r, g, b = parts[:3]
    a = parts[3] if len(parts) > 3 else 1.0
    return (r, g, b, a)


def flatten(layers):
    """Composite `layers` (topmost first) to an opaque rgb triple.

    The bottom layer must be opaque; `_SURFACES` always ends in
    `#000000` so that holds by construction.
    """
    r, g, b, a = parse_color(layers[-1])
    assert a == 1.0, "the bottom surface layer must be opaque"
    out = (r, g, b)
    for layer in reversed(layers[:-1]):
        lr, lg, lb, la = parse_color(layer)
        out = tuple(lc * la + oc * (1 - la) for lc, oc in zip((lr, lg, lb), out))
    return out


def contrast(fg_css: str, surface_layers) -> float:
    """WCAG contrast of `fg_css` painted on `surface_layers`.

    Public (no leading underscore) because the arithmetic IS the
    assertion — a helper the test cannot exercise independently is the
    kind of instrument that passes by not running.
    """
    bg = flatten(surface_layers)
    r, g, b, a = parse_color(fg_css)
    fg = tuple(c * a + bc * (1 - a) for c, bc in zip((r, g, b), bg))
    l1, l2 = _luminance(fg), _luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


# ── the instrument, before anything trusts it (R6) ──────────────────

def test_the_formula_matches_published_wcag_pairs():
    """Calibrated on reference pairs with a documented answer, NOT on my
    own screenshots of this UI — those were taken over a live animated
    backdrop and are not reproducible (see the module docstring)."""
    # The two ends the formula cannot get wrong.
    assert round(contrast("#ffffff", ["#000000"]), 1) == 21.0
    assert round(contrast("#000000", ["#000000"]), 1) == 1.0
    # WCAG's own worked example: #767676 is the darkest grey that still
    # passes AA on white, at 4.54:1.
    assert round(contrast("#767676", ["#ffffff"]), 2) == 4.54
    assert contrast("#767676", ["#ffffff"]) >= _AA
    # One step lighter and it fails — so the floor genuinely bites.
    assert contrast("#777777", ["#ffffff"]) < _AA


def test_alpha_compositing_actually_composites():
    """A fully transparent foreground must vanish into its surface
    (ratio 1.0), and a fully opaque one must ignore the surface. Without
    this, an alpha bug that silently treated 0.55 as 1.0 would make
    every assertion below pass for the wrong reason."""
    assert round(contrast("rgba(255,255,255,0)", ["#000000"]), 3) == 1.0
    assert (round(contrast("rgba(255,255,255,1)", ["#000000"]), 3)
            == round(contrast("#ffffff", ["#000000"]), 3))
    # Lowering alpha on light text over a dark surface must LOWER contrast.
    assert (contrast("rgba(158,170,192,0.55)", ["#000000"])
            < contrast("rgba(158,170,192,0.75)", ["#000000"]))


def test_the_shipped_values_would_fail_if_reverted():
    """Names the world where this file fails. The pre-fix values must
    miss AA on their own surfaces — by enough that the variable face
    backdrop cannot explain it away."""
    for selector, old in ((".session-meta", "rgba(158, 170, 192, 0.55)"),
                          (".msg-time", "rgba(158, 170, 192, 0.55)"),
                          ("#planner-monologue", "#c22745")):
        ratio = contrast(old, _SURFACES[selector])
        assert ratio < _AA, (
            f"{selector}'s pre-fix value {old} measures {ratio:.2f}:1 — if "
            f"that now passes, the model drifted and the pins below are "
            f"no longer testing anything")


# ── the shipped values ──────────────────────────────────────────────

@pytest.mark.parametrize("selector", sorted(_SURFACES))
def test_metadata_text_meets_wcag_aa(css, selector):
    value = _declared_color(css, selector)
    ratio = contrast(value, _SURFACES[selector])
    assert ratio >= _AA, (
        f"{selector} ships {value} = {ratio:.2f}:1 over its declared "
        f"surface, below the WCAG AA floor of {_AA}:1. This text is under "
        f"13px, so the large-text exemption does not apply. Note the real "
        f"backdrop can be BRIGHTER than modelled (the face renders behind "
        f"these translucent surfaces), so this number is the optimistic "
        f"bound.")


def test_planner_monologue_is_not_the_error_colour(css):
    """Live planner narration ("considering X…") was `#c22745`, which is
    `--error-color` verbatim — so the one colour reserved for something
    being WRONG fired on every healthy turn. Errors keep it to
    themselves."""
    root = _rule_body(css, ":root")
    m = re.search(r"--error-color\s*:\s*([^;]+);", root)
    assert m, "--error-color is no longer declared in :root"
    error = parse_color(m.group(1).strip())
    mono = parse_color(_declared_color(css, "#planner-monologue"))
    assert mono[:3] != error[:3], (
        "#planner-monologue is painted --error-color; ordinary planner "
        "narration must not use the failure signal")


def test_the_modelled_surfaces_are_the_ones_the_css_ships(css):
    """The surface stack above is hand-written, so it can silently drift
    from the CSS and leave every ratio computed against a fiction."""
    rail = _rule_body(css, "#session-rail")
    assert "rgba(8, 5, 14, 0.62)" in rail, (
        "#session-rail's background changed — update _SURFACES, the "
        "contrast numbers above are computed against the old one")
    agent = _rule_body(css, ".message.agent")
    assert "rgba(22, 13, 38, 0.20)" in agent, (
        ".message.agent's background changed — update _SURFACES")
    root = _rule_body(css, ":root")
    assert re.search(r"--bg-color\s*:\s*#000000", root), (
        "--bg-color is no longer #000000 — every surface stack here "
        "bottoms out on it")
