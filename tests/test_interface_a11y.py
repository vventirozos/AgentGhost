"""Every control has a name, and every control shows keyboard focus
(2026-09-04).

⚠ WHY THIS FILE EXISTS — two defects of the same shape, both found by
auditing the live UI rather than by any test here:

1. `#send-btn` — the single most-used control in the app — carried no
   `title`, no `aria-label` and only an inline SVG, so assistive tech
   announced it as "button". Fifteen other icon-only controls had a
   `title` and nothing else; `title` is not surfaced by touch at all,
   which is where this UI is half used.

2. The `:focus-visible` rule was a hand-kept ENUMERATION of ten
   selectors, and `#send-btn`, `#mic-btn`, `#palette-input`,
   `.face-form-item` and `#delete-all-sessions-btn` were not on it. A
   keyboard user tabbing to the primary action saw nothing at all.

Both are the same failure: a per-instance list that only covers what
somebody remembered on the day they wrote it. So both tests here walk
index.html and enumerate the CLASS — every button, every focusable
element — rather than checking the specific controls that were broken.
A test naming those would pass today and rot immediately.

⚠ AND THEN THE FIX HAD THE SAME BUG. Generalising the rule to
`:where(button, input, textarea, …):focus-visible` gives it specificity
0-1-0, which four `outline: none` declarations on IDs (#chat-input,
#session-search, #palette-input, #memory-match/#memory-replacement)
outrank. Every selector-coverage test in this file passed while a real
browser computed `outline-style: none` on all four. Asking "does a rule
match?" is a PROXY for "is a ring visible", and the proxy is what
shipped. `test_no_control_silently_suppresses_its_focus_ring` reads the
suppressions instead; #chat-input remains the one declared exemption,
with its `#input-area:focus-within` indicator asserted separately.

The dynamic labels (`#send-btn` flips Send/Stop/Busy, `#workspace-btn`
flips Load/Save) are checked in app.js: a STATIC label on a control
whose meaning changes is worse than none, because it is confidently
wrong.
"""

import re
from pathlib import Path

import pytest

from tests.helpers import strip_js_comments

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def html() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return (_STATIC / "style.css").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js() -> str:
    return strip_js_comments((_STATIC / "app.js").read_text(encoding="utf-8"))


def _strip_html_comments(html: str) -> str:
    """index.html carries long explanatory comments that mention element
    ids and markup; scanning them as if they were live DOM invents
    controls that do not exist."""
    return re.sub(r"<!--.*?-->", " ", html, flags=re.S)


def _elements(html: str, tag: str):
    """-> [(attrs_text, inner_text)] for every non-void `tag`."""
    out = []
    for m in re.finditer(rf"<{tag}(\s[^>]*)?>(.*?)</{tag}>",
                         _strip_html_comments(html), re.S):
        out.append((m.group(1) or "", m.group(2)))
    return out


def _attr(attrs: str, name: str):
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', attrs)
    return m.group(1) if m else None


def _visible_text(inner: str) -> str:
    """Inner text with markup removed — an SVG is not a name."""
    return re.sub(r"<[^>]+>", " ", inner).strip()


def _accessible_name(attrs: str, inner: str):
    """The name AT would announce, by the parts of the algorithm that
    apply to this markup: aria-label, then contents, then title."""
    for candidate in (_attr(attrs, "aria-label"),
                      _visible_text(inner) or None,
                      _attr(attrs, "title")):
        if candidate and candidate.strip():
            return candidate.strip()
    return None


# ── the instrument, before anything trusts it (R6) ──────────────────

def test_the_scanner_finds_buttons_and_reads_names():
    sample = ('<button id="a" aria-label="Alpha"><svg><path/></svg></button>'
              '<button id="b">Beta</button>'
              '<button id="c" title="Gamma"><svg/></button>'
              '<button id="d"><svg><path/></svg></button>')
    found = _elements(sample, "button")
    assert len(found) == 4, found
    names = [_accessible_name(a, i) for a, i in found]
    assert names == ["Alpha", "Beta", "Gamma", None], names


def test_the_scanner_ignores_commented_out_markup():
    """index.html documents removed controls inside comments (the TTS
    toggle, for one). Counting those as live buttons would fail this
    file on markup that does not exist."""
    sample = '<!-- <button id="ghost"><svg/></button> --><button>Real</button>'
    assert len(_elements(sample, "button")) == 1


def test_an_svg_only_button_is_reported_nameless():
    """The exact shape of the #send-btn defect must be detectable."""
    assert _accessible_name("", "<svg><path d='M1 1'/></svg>") is None


# ── every button has a name ─────────────────────────────────────────

def test_every_button_has_an_accessible_name(html):
    nameless = [_attr(a, "id") or a.strip()[:60]
                for a, i in _elements(html, "button")
                if _accessible_name(a, i) is None]
    assert not nameless, (
        "these controls announce as bare \"button\": "
        f"{nameless}. An inline SVG is not an accessible name.")


def test_icon_only_buttons_do_not_rely_on_title_alone(html):
    """`title` produces no tooltip on touch and is an unreliable name
    source. Any control whose only content is an SVG needs an explicit
    `aria-label`."""
    bad = []
    for attrs, inner in _elements(html, "button"):
        if _visible_text(inner):
            continue                      # has real text content
        if not _attr(attrs, "aria-label"):
            bad.append(_attr(attrs, "id") or attrs.strip()[:60])
    assert not bad, (
        f"icon-only controls with no aria-label: {bad}")


def test_role_button_elements_have_names_too(html):
    """`#status-indicator` is a div with role=button; the enumeration has
    to follow the ROLE, not the tag, or a control moves out of scope by
    changing element."""
    for tag in ("div", "span"):
        for attrs, inner in _elements(html, tag):
            if _attr(attrs, "role") != "button":
                continue
            assert _accessible_name(attrs, inner), (
                f"role=button {tag} #{_attr(attrs, 'id')} has no name")


# ── every focusable control shows a focus ring ──────────────────────

def _focus_rule_selectors(css: str):
    """The selector list inside `:where( … ):focus-visible`, or None.

    ⚠ Paren-MATCHED, not `[^)]*`. The shipped rule ends with
    `[tabindex]:not([tabindex="-1"])`, so a naive character class stops
    at the `)` of `:not(` and reads a truncated list — the first version
    of this helper reported the whole rule as missing and failed on a
    correct stylesheet.
    """
    i = css.find(":where(")
    while i != -1:
        j = i + len(":where(")
        depth = 1
        while j < len(css) and depth:
            depth += (css[j] == "(") - (css[j] == ")")
            j += 1
        if css[j:j + len(":focus-visible")] == ":focus-visible":
            return _split_top_level(css[i + len(":where("):j - 1])
        i = css.find(":where(", j)
    return None


def _split_top_level(sel_list: str):
    """Split on commas that are not inside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in sel_list:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += (ch == "(") - (ch == ")")
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _focus_ring_covers(css: str, tag: str, attrs: str) -> bool:
    """Does some `:focus-visible` rule reach this element?"""
    selectors = _focus_rule_selectors(css)
    if selectors:
        for sel in selectors:
            if sel == tag:
                return True
            am = re.match(r'\[(\w[\w-]*)(?:="([^"]*)")?\]', sel)
            if am:
                key, want = am.group(1), am.group(2)
                have = _attr(attrs, key)
                if have is not None and (want is None or have == want):
                    return True
            if sel.startswith(f"{tag}["):
                inner = sel[len(tag):]
                am = re.match(r'\[(\w[\w-]*)\]', inner)
                if am and _attr(attrs, am.group(1)) is not None:
                    return True
    # Fall back to the older per-selector enumeration, if any survives.
    ident = _attr(attrs, "id")
    if ident and f"#{ident}:focus-visible" in css:
        return True
    for cls in (_attr(attrs, "class") or "").split():
        if f".{cls}:focus-visible" in css:
            return True
    return False


def test_every_interactive_control_gets_a_focus_ring(css, html):
    uncovered = []
    for tag in ("button", "textarea", "select"):
        for attrs, _ in _elements(html, tag):
            if not _focus_ring_covers(css, tag, attrs):
                uncovered.append(f"<{tag} id={_attr(attrs, 'id')}>")
    # Inputs are void elements, so they need their own scan.
    for m in re.finditer(r"<input(\s[^>]*?)/?>", _strip_html_comments(html)):
        attrs = m.group(1)
        if _attr(attrs, "type") == "file":
            continue                      # display:none proxy for a button
        if not _focus_ring_covers(css, "input", attrs):
            uncovered.append(f"<input id={_attr(attrs, 'id')}>")
    assert not uncovered, (
        "no :focus-visible rule reaches these, so a keyboard user sees "
        f"nothing when they land on them: {uncovered}")


def test_the_focus_rule_is_general_not_a_hand_kept_list(css):
    """The defect was the LIST, not its contents. A rule over element
    types covers a control the day it is added; an enumeration covers it
    the day somebody remembers."""
    selectors = _focus_rule_selectors(css)
    assert selectors, (
        ":focus-visible is no longer declared over element types — if it "
        "went back to a selector enumeration, the next control added will "
        "silently ship with no focus ring, which is exactly how "
        "#send-btn and #mic-btn came to have none")
    covered = set(selectors)
    for required in ("button", "input", "textarea"):
        assert required in covered, f"{required} dropped out of the rule"


def test_the_focus_rule_parser_survives_nested_parens():
    """R6 for the parser itself: the shipped rule's last selector
    contains a `)`, which is what defeated the first version."""
    css = (':where(button, [role="button"], '
           '[tabindex]:not([tabindex="-1"])):focus-visible { outline: 1px; }')
    sels = _focus_rule_selectors(css)
    assert sels == ["button", '[role="button"]',
                    '[tabindex]:not([tabindex="-1"])'], sels
    assert _focus_rule_selectors("button:focus-visible { outline: 1px; }") is None


def test_the_focus_coverage_check_can_fail(css):
    """R6. A tag the rule does not name must come back uncovered, or the
    test above passes by matching everything."""
    assert _focus_ring_covers(css, "button", ' id="send-btn"')
    assert not _focus_ring_covers(css, "marquee", ' id="nope"')


# ── the CASCADE, not just the selector ──────────────────────────────
#
# ⚠ The tests above were the first version of this file, and they were a
# PROXY. They ask "does a :focus-visible selector match this element?"
# and every one passed while four text inputs — #chat-input,
# #session-search, #palette-input and #memory-match/#memory-replacement
# — rendered NO ring at all, because each carried `outline: none` at ID
# specificity (1-0-0), which outranks the `:where(…):focus-visible` rule
# (0-1-0). Measured in a real browser after the "fix" had shipped:
# `outline-style: none` on all four.
#
# A matching selector is not a visible ring. These read the suppressions.

_OUTLINE_NONE = re.compile(
    r"([^{}]+?)\{[^{}]*?(?<![-\w])outline\s*:\s*none", re.S)

# The one control allowed to suppress its ring, and the rule that has to
# exist for that to be legitimate. `#input-area:focus-within` restyles
# the whole composer pill — border colour, fill, glow and a 2px lift —
# a stronger indicator than a ring around a borderless textarea inside
# it. Anything else appearing here is a regression.
_ALLOWED_SUPPRESSION = {"#chat-input": "#input-area:focus-within"}


def _suppressing_selectors(css: str):
    """Selectors that set `outline: none`, comments stripped."""
    stripped = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    out = []
    for m in _OUTLINE_NONE.finditer(stripped):
        for sel in m.group(1).split(","):
            sel = sel.strip().splitlines()[-1].strip()
            if sel and not sel.startswith("@"):
                out.append(sel)
    return out


def test_no_control_silently_suppresses_its_focus_ring(css):
    """`outline: none` on an id beats the general rule. Every such
    suppression must be a DECLARED exemption with its own indicator."""
    unexpected = [s for s in _suppressing_selectors(css)
                  if s not in _ALLOWED_SUPPRESSION]
    assert not unexpected, (
        "these rules suppress the keyboard focus ring at a specificity "
        f"the general :focus-visible rule cannot beat: {unexpected}. A "
        "control that does this needs its own visible focus indicator "
        "and an entry in _ALLOWED_SUPPRESSION explaining it.")


@pytest.mark.parametrize("suppressor,indicator",
                         sorted(_ALLOWED_SUPPRESSION.items()))
def test_each_exempted_control_still_has_its_indicator(css, suppressor,
                                                       indicator):
    """An exemption nobody checks is a hole with a comment on it. If the
    composer's focus-within treatment is ever deleted, its `outline:
    none` stops being a considered trade-off and becomes a control with
    no focus indicator at all."""
    assert suppressor in _suppressing_selectors(css), (
        f"{suppressor} no longer suppresses its outline — remove it from "
        f"_ALLOWED_SUPPRESSION so the general rule is what covers it")
    m = re.search(re.escape(indicator) + r"\s*\{([^{}]*)\}", css)
    assert m, (
        f"{suppressor} suppresses its focus ring and its replacement "
        f"indicator `{indicator}` is gone — the control now shows nothing "
        f"on keyboard focus")
    declared = {d.split(":")[0].strip()
                for d in m.group(1).split(";") if ":" in d}
    assert declared & {"border-color", "box-shadow", "background",
                       "outline", "transform"}, (
        f"`{indicator}` no longer changes anything visible: {declared}")


def test_the_suppression_scanner_can_fail():
    """R6 for the scanner. It must SEE a suppression, ignore a commented
    one, and not invent them."""
    assert _suppressing_selectors("#x { outline: none; }") == ["#x"]
    assert _suppressing_selectors("#x { /* outline: none; */ color: red; }") == []
    assert _suppressing_selectors("#x { outline-offset: none; }") == []
    assert _suppressing_selectors("#x { color: red; }") == []
    assert set(_suppressing_selectors("#a,\n#b { outline: none; }")) == {"#a", "#b"}


# ── labels that must MOVE with the control's meaning ────────────────

class TestDynamicLabels:
    """A static name on a control that changes meaning is worse than no
    name: it is confidently wrong."""

    def test_send_button_relabels_for_every_state(self, app_js):
        m = re.search(r"function toggleSendButtonUI\(.*?\n\}", app_js, re.S)
        assert m, "toggleSendButtonUI is gone"
        body = m.group(0)
        labels = re.findall(r"aria-label',\s*\n?\s*(.+?)\);", body, re.S)
        assert labels, (
            "the send button's accessible name never changes — it "
            "announces 'Send message' while showing a Stop icon")
        joined = " ".join(labels)
        assert "Stop" in joined, "no Stop label for the cancellable state"
        assert "Working" in joined or "Busy" in joined, (
            "no label for the non-cancellable busy state, which is a "
            "DISABLED button showing a send icon")

    def test_workspace_button_relabels_with_its_title(self, app_js):
        m = re.search(r"function updateWorkspaceBtnState\(.*?\n\}",
                      app_js, re.S)
        assert m, "updateWorkspaceBtnState is gone"
        body = m.group(0)
        titles = len(re.findall(r"\.title\s*=", body))
        labels = len(re.findall(r"aria-label'", body))
        assert labels == titles, (
            f"the title flips between Load/Save on {titles} branches but "
            f"the accessible name on {labels} — one glyph, two actions, "
            f"and screen readers hear only one of them")
