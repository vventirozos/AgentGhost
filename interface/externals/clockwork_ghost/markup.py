"""Reply text → the HTML a chat bubble draws. Qt-free, so it is unit-tested.

Three jobs, all of which used to live inline in ``client.py``/``chatlog.py``
where nothing could execute them off the device:

* :func:`render_reply` — markdown → HTML, with the agent's ``/api/download``
  images turned into a tappable glyph (a QLabel cannot fetch a remote image).
* :func:`style_markup` — chat-sized typography injected per tag, because Qt's
  rich text applies its own (enormous) heading defaults and a QLabel offers no
  document stylesheet.
* :func:`soften_long_tokens` — break opportunities inside long unbroken runs.

**Why the last one exists (measured on the device, Qt 6.4.2, 2026-10-01).** A
word-wrapped QLabel wraps at WORD boundaries only, and a ``<pre>`` does not
wrap at all, so anything wider than the bubble was clipped with no way to
scroll to it: a 1,078 px code line in a 670 px column, a 1,614 px URL in
prose. ``white-space: pre-wrap`` fixes the code line (629 px, indentation
kept) and a zero-width space every few characters fixes the unbroken run
(558 px). Both were measured before being adopted — see ``device_probe.py``,
which re-measures them on every deploy.
"""

from __future__ import annotations

import re

# Prose is set in a SANS face; only the operator's own input, inline code and
# code blocks stay monospace. This mirrors the web UI, where agent messages
# inherit the sans body font and `.message.user` overrides to mono — reading a
# briefing in 21px monospace is what made long answers feel wrong.
SANS = "'DejaVu Sans', 'Liberation Sans', 'Cantarell', 'Noto Sans', sans-serif"

# Markdown → HTML arrives as bare <h2>/<p>/<ol>, and Qt's rich text applies its
# OWN defaults to those: <h1> renders at roughly 2x the base size and <h2> at
# 1.5x, so a briefing's headings came out enormous next to 21px body text.
# QLabel gives no hook for a document stylesheet, so the styles are injected
# inline, per tag. Headings sit only slightly above body size — in a chat
# bubble a heading is a label, not a page title.
_TAG_STYLES = {
    "h1": "font-size:22px; font-weight:700; margin:12px 0 6px 0;",
    "h2": "font-size:21px; font-weight:700; margin:11px 0 5px 0;",
    "h3": "font-size:20px; font-weight:700; margin:10px 0 4px 0;",
    "h4": "font-size:19px; font-weight:700; margin:9px 0 4px 0;",
    "h5": "font-size:19px; font-weight:600; margin:8px 0 3px 0;",
    "h6": "font-size:19px; font-weight:600; margin:8px 0 3px 0;",
    "p": "margin:0 0 9px 0; line-height:148%;",
    "ul": "margin:2px 0 9px 0; -qt-list-indent:1;",
    "ol": "margin:2px 0 9px 0; -qt-list-indent:1;",
    "li": "margin:0 0 4px 0; line-height:145%;",
    "blockquote": "margin:6px 0 8px 6px; padding-left:11px;",
    "hr": "margin:10px 0;",
    "table": "margin:6px 0 9px 0;",
    "th": "padding:3px 11px 3px 0; font-weight:700;",
    "td": "padding:3px 11px 3px 0;",
}
_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:\s[^>]*?)?)(/?)>")
# A real `style` attribute — not `data-style=`, not `?style=dark` in an href.
_STYLE_ATTR_RE = re.compile(r"""(\sstyle\s*=\s*)(["'])(.*?)\2""", re.I | re.S)

# A run of non-whitespace longer than this cannot wrap and is clipped by the
# bubble. 40 is under the narrowest column the client draws (a 0.56-width
# bubble at 20px holds ~60 characters), so a run this long is always at risk.
LONG_TOKEN = 40
BREAK_EVERY = 20
ZWSP = "\u200b"
# A run of non-space that has no break opportunity YET — a ZWSP ends a run,
# so softening an already-softened string changes nothing.
_LONG_RUN_RE = re.compile(r"[^\s\u200b]{%d,}" % (LONG_TOKEN + 1))
# One displayed character. An entity is ONE unit and must never be split —
# `&am` + ZWSP + `p;` would be drawn as literal text — and a base character
# keeps its combining marks, variation selector and joiners (a break between
# `e` and U+0301 un-accents the letter; one inside a flag pair splits the flag).
_UNIT_RE = re.compile(
    r"(?:&#?\w+;|[\U0001F1E6-\U0001F1FF]{2}|.)"
    r"(?:[\u0300-\u036f\ufe0e\ufe0f]|\u200d.)*", re.S)
_SPLIT_TAGS_RE = re.compile(r"(<[^>]*>)")

# The alt text cannot contain `]` (a lazy `.*?` ran from one image's `![` to
# the NEXT image's target and swallowed the prose between), and the target
# stops at whitespace so an optional "title" is not taken for part of the path.
_IMG_RE = re.compile(r'!\[([^\]\n]*)\]\((/api/download/[^)\s]+)(?:\s+"[^"]*")?\)')
_FENCED_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.S)


def _img_link(m: "re.Match") -> str:
    # The alt text is the model's; it goes inside an ATTRIBUTE, so quotes are
    # escaped too — an alt of `x" href="…` must not rewrite the link.
    alt = (m.group(1).replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;").replace('"', "&quot;"))
    return (f'<br><a href="{m.group(2)}" style="text-decoration:none; '
            f'font-size:28px;" title="View Image: {alt}">🖼️</a>')
# A tool-call BLOCK is the tag followed by its JSON. A reply that merely
# mentions the tag ("the <tool_call> tag wraps…") is prose, and matching the
# bare tag cut everything after it out of a restored message.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*\{[\s\S]*?(?:</tool_call>|$)", re.I)


def _soften_run(m: "re.Match") -> str:
    units = _UNIT_RE.findall(m.group(0))
    return ZWSP.join("".join(units[i:i + BREAK_EVERY])
                     for i in range(0, len(units), BREAK_EVERY))


def soften_long_tokens(html: str) -> str:
    """Insert zero-width break opportunities into long unbroken runs.

    Only TEXT is touched — never a tag, so an ``href`` keeps working — and an
    entity is never cut in half. Text that already wraps is returned
    unchanged, byte for byte.
    """
    if not html:
        return html
    parts = _SPLIT_TAGS_RE.split(html)
    for i in range(0, len(parts), 2):          # even slots are text nodes
        if len(parts[i]) > LONG_TOKEN:
            parts[i] = _LONG_RUN_RE.sub(_soften_run, parts[i])
    return "".join(parts)


def style_markup(html: str, mono_font: str, accent: str, dim: str) -> str:
    """Give markdown-generated HTML sane, chat-sized typography."""
    code_style = (f"font-family:{mono_font}; font-size:18px;"
                  f" background-color:rgba(0,0,0,0.30);")
    per_tag = dict(_TAG_STYLES)
    per_tag["code"] = code_style + " padding:0 3px;"
    # `white-space: pre-wrap`: a code line wider than the bubble wraps (and
    # keeps its indentation) instead of running off the right edge.
    per_tag["pre"] = (f"font-family:{mono_font}; font-size:17px;"
                      f" white-space:pre-wrap;"
                      f" background-color:rgba(0,0,0,0.32); margin:7px 0 9px 0;")
    per_tag["a"] = f"color:{accent}; text-decoration:none;"
    per_tag["blockquote"] = (_TAG_STYLES["blockquote"] + f" color:{dim};"
                             f" border-left:2px solid {accent};")

    def _inject(m):
        tag, attrs, close = m.group(1).lower(), m.group(2) or "", m.group(3)
        style = per_tag.get(tag)
        if not style:
            return m.group(0)
        has = _STYLE_ATTR_RE.search(attrs)
        if has:
            # MERGE, ours first so the element's own declarations win. An
            # aligned table cell arrives as `<td style="text-align: left;">`;
            # skipping every already-styled tag left such tables unpadded.
            merged = f"{has.group(1)}{has.group(2)}{style} {has.group(3)}{has.group(2)}"
            return f"<{m.group(1)}{attrs[:has.start()]}{merged}{attrs[has.end():]}{close}>"
        return f"<{m.group(1)}{attrs} style=\"{style}\"{close}>"

    return soften_long_tokens(_TAG_RE.sub(_inject, html))


def _prose_parts(text: str) -> list:
    """``text`` split so that odd slots are fenced code blocks, even slots
    prose. An image written INSIDE a code block is an example, not an image."""
    return _FENCED_RE.split(text or "")


def reply_images(text: str) -> list:
    """The ``/api/download/…`` image paths a reply embeds, in order."""
    return [path for part in _prose_parts(text)[0::2]
            for _alt, path in _IMG_RE.findall(part)]


def render_reply(text: str, markdown_fn=None) -> str:
    """An assistant reply as bubble HTML (before :func:`style_markup`).

    ``markdown_fn`` is injectable for tests; by default the ``markdown``
    package the client already depends on.
    """
    if markdown_fn is None:
        import markdown as _md
        markdown_fn = lambda t: _md.markdown(t, extensions=["fenced_code", "tables"])  # noqa: E731
    parts = _prose_parts(text)
    parts[0::2] = [_IMG_RE.sub(_img_link, part) for part in parts[0::2]]
    return markdown_fn("".join(parts))


def strip_tool_calls(text: str) -> str:
    """A stored assistant message without its raw ``<tool_call>`` blocks."""
    return _TOOL_CALL_RE.sub("", text or "").strip()


def transcript_items(messages) -> list:
    """A stored conversation as ``[(role, text)]`` rows the transcript can draw.

    Shared by the workspace restore and the session restore, which used to be
    one inline loop. Roles other than user/assistant (system prompts, tool
    results) are not conversation and are skipped; a multimodal user turn
    shows its text part, or a placeholder when it was an image alone.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):
            # Multimodal: every text part, in order. A turn that was an
            # attachment alone still gets a row, so the reply below it is not
            # an answer to nothing.
            text = "\n".join(str(p.get("text")) for p in content
                             if isinstance(p, dict) and p.get("type") == "text"
                             and str(p.get("text") or "").strip())
            if role == "user":
                text = text or "[attachment]"
        else:
            text = content if isinstance(content, str) else ""
        if role == "user":
            if text.strip():
                out.append(("user", text))
        elif role == "assistant":
            text = strip_tool_calls(text)
            if text:
                out.append(("assistant", text))
    return out


def escape_user(text: str) -> str:
    """The operator's own words as bubble HTML: escaped, line breaks kept.

    User bubbles are drawn as rich text too, so an unescaped ``<`` typed by
    the operator used to be swallowed as markup.
    """
    return (str(text or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", "<br>"))
