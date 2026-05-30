"""Regression guard for the interface chat-area layout (style.css).

The real cause of "the top third isn't showing chats" was the header, not
the chat container: `#header-controls` stacked its icon buttons in a
`flex-direction: column`, making the (flex-shrink:0) header ~300px tall
and reserving the top third above the chat. Secondary issues were a top
mask gradient that faded the top 15% of messages until hovered, and a
`max-height` cap that left a band empty even once the header was fixed.

These assertions catch regressions that reintroduce any of those:
- header buttons stacked vertically,
- the top-fade mask (and its hover toggle),
- a restrictive max-height cap on the chat container.
"""

import re
from pathlib import Path

import pytest

_CSS_PATH = Path(__file__).resolve().parent.parent / "interface" / "static" / "style.css"


def _rule_body(css: str, selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, re.DOTALL)
    assert m, f"{selector} rule not found in style.css"
    return m.group(1)


def _media_block(css: str, condition_substr: str):
    """Return (header, body) of the first @media block whose condition
    contains condition_substr, brace-matched so nested rule braces are
    handled. None if not found."""
    idx = 0
    while True:
        m = re.search(r"@media([^{]*)\{", css[idx:])
        if not m:
            return None
        header = m.group(1)
        start = idx + m.end()
        depth, i = 1, start
        while i < len(css) and depth > 0:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        body = css[start : i - 1]
        if condition_substr in header:
            return header, body
        idx = i


@pytest.fixture(scope="module")
def css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_container_block(css) -> str:
    return _rule_body(css, "main#chat-container")


@pytest.fixture(scope="module")
def header_controls_block(css) -> str:
    return _rule_body(css, "#header-controls")


@pytest.fixture(scope="module")
def header_block(css) -> str:
    return _rule_body(css, "header")


def test_header_buttons_are_horizontal(header_controls_block):
    """The icon buttons must lay out in a row, not a tall vertical stack.

    A column made the header ~300px tall and blanked the top third.
    """
    body = re.sub(r"/\*.*?\*/", "", header_controls_block, flags=re.DOTALL)
    assert re.search(r"flex-direction:\s*row", body)
    assert "flex-direction: column" not in body


def test_no_top_mask_fade(chat_container_block):
    """The top-fade mask was removed so top messages aren't hidden."""
    body = re.sub(r"/\*.*?\*/", "", chat_container_block, flags=re.DOTALL)
    assert "mask-image" not in body


def test_no_restrictive_height_cap(chat_container_block):
    """No max-height cap: the chat fills from the header down to the input.

    Any cap below ~90vh reintroduces an empty band at the top.
    """
    body = re.sub(r"/\*.*?\*/", "", chat_container_block, flags=re.DOTALL)
    m = re.search(r"max-height:\s*(\d+)vh", body)
    assert m is None or int(m.group(1)) >= 90


def test_chat_uses_robust_fill_pattern(chat_container_block):
    """Chat fills via flex + min-height:0, with no vestigial auto margin.

    The old `margin-top:auto` + flex-grow combo overlapped the header on
    iOS Safari (buttons couldn't be tapped); the fill pattern can't.
    """
    body = re.sub(r"/\*.*?\*/", "", chat_container_block, flags=re.DOTALL)
    assert re.search(r"flex:\s*1\s+1\s+0", body) or re.search(r"flex-grow:\s*1", body)
    assert re.search(r"min-height:\s*0", body)
    assert not re.search(r"margin-top:\s*auto", body)


def test_header_sits_above_chat(header_block):
    """The header must establish a stacking layer above the chat so its
    buttons stay visible and tappable even if regions meet."""
    body = re.sub(r"/\*.*?\*/", "", header_block, flags=re.DOTALL)
    assert re.search(r"position:\s*relative", body)
    assert re.search(r"z-index:\s*\d+", body)


def test_header_has_no_fixed_height(css):
    """No `header` rule may set a fixed `height` (min/max-height only).

    A fixed height clipped the mobile header — where the <=480 rule stacks
    status + buttons into a column — so the buttons overflowed below it
    into the chat. The uConsole media query used `height:50px`; it must be
    `min-height` so the header grows to contain its content.
    """
    # Match `header {` but not `#header-controls {` / `render-header {`.
    for m in re.finditer(r"(?<![-\w])header\s*\{(.*?)\}", css, re.DOTALL):
        body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.DOTALL)
        assert not re.search(r"(?<!min-)(?<!max-)\bheight\s*:", body), (
            f"fixed height in a header rule (use min-height): {body.strip()!r}"
        )


def test_uconsole_query_does_not_degrade_phones(css):
    """The broad uConsole query (max-width:1280 and max-height:720) also
    matches phones, so it must NOT shrink touch targets to 36px nor set
    the chat input below 16px (which makes iOS Safari auto-zoom on focus).
    """
    block = _media_block(css, "max-width: 1280px")
    assert block, "uConsole media query (max-width:1280px) not found"
    _, body = block
    assert "36px" not in body, "36px tap targets leak to phones in the broad query"
    # #chat-input must not carry a sub-1rem font-size here (iOS focus-zoom).
    assert not re.search(
        r"#chat-input\b[^}]*font-size:\s*0?\.[0-9]+rem", body, re.DOTALL
    ), "chat input font < 16px leaks to phones (iOS auto-zoom)"


def test_uconsole_shrink_is_width_gated(css):
    """The 36px shrink lives in a min-width-gated, uConsole-only query."""
    block = _media_block(css, "min-width: 1000px")
    assert block, "expected a min-width-gated uConsole-only media query"
    assert "36px" in block[1]


def test_no_mask_hover_rule_remains(css):
    """The :hover rule that toggled the mask should be gone too."""
    assert "main#chat-container:hover" not in css
