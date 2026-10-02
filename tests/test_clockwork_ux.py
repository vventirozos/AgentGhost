"""uConsole client — the 2026-10-01 UI/UX pass (journal §4KU).

The logic of that pass was written into Qt-FREE modules precisely so it could
be executed here: ``markup`` (what a bubble draws), ``speech`` (what is said
aloud), ``devstatus`` (battery, link, backlight, volume, panel, face rate),
``agentapi`` (sessions, cancel, feedback, notifications) and ``commands``
(slash commands). These tests import the real modules.

What cannot be imported here — ``client.py`` and ``chatlog.py`` need PyQt6,
which lives on the handheld — is exercised by ``device_probe.py``, which
``deploy.sh`` runs on the device before it installs anything. The few
``client.py`` checks below are AST enumerations of a CLASS of mistake (a
second way to start a turn, a second way to power off), not text matches.

World where each pin fails is named in its docstring or its assertion message.
"""

import ast
import asyncio
import io
import json
import os
import re
import stat
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function

_ROOT = Path(__file__).resolve().parent.parent
_DIR = _ROOT / "interface" / "externals" / "clockwork_ghost"
_APP_JS = _ROOT / "interface" / "static" / "app.js"

sys.path.insert(0, str(_DIR))
import agentapi  # noqa: E402
import commands  # noqa: E402
import devstatus  # noqa: E402
import markup  # noqa: E402
import speech  # noqa: E402
import turnstatus  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# markup — nothing is clipped, nothing is swallowed
# ════════════════════════════════════════════════════════════════════════════

def test_a_long_unbroken_run_gets_break_opportunities():
    """A word-wrapped QLabel wraps at word boundaries only: a 160-character
    URL was drawn 1,614 px wide in a 670 px column and clipped."""
    run = "a" * 100
    out = markup.soften_long_tokens(f"<p>see {run} ok</p>")
    pieces = out[len("<p>see "):-len(" ok</p>")].split(markup.ZWSP)
    assert "".join(pieces) == run
    assert len(pieces) == 5 and all(len(p) == markup.BREAK_EVERY for p in pieces)


def test_text_that_already_wraps_is_returned_byte_for_byte():
    html = "<p>An ordinary sentence, with a <a href=\"https://x.test/a\">link</a>.</p>"
    assert markup.soften_long_tokens(html) == html
    edge = "<p>" + "b" * markup.LONG_TOKEN + "</p>"          # exactly at the limit
    assert markup.soften_long_tokens(edge) == edge
    # …also when the text node itself is longer than the limit
    inside = "<p>hello " + "b" * markup.LONG_TOKEN + " there</p>"
    assert markup.soften_long_tokens(inside) == inside
    over = "<p>" + "b" * (markup.LONG_TOKEN + 1) + "</p>"    # one past it
    assert markup.ZWSP in markup.soften_long_tokens(over)


def test_softening_twice_changes_nothing_more():
    """A bubble is re-rendered on every streamed token; a pass that adds
    breaks to its own output would grow them without bound."""
    once = markup.soften_long_tokens("<p>" + "q" * 130 + "</p>")
    assert markup.soften_long_tokens(once) == once


def test_a_tag_is_never_broken_so_a_link_still_works():
    href = "https://example.test/" + "p" * 120
    out = markup.soften_long_tokens(f'<a href="{href}">{href}</a>')
    assert f'href="{href}"' in out, "the href was softened — the link is dead"
    assert markup.ZWSP in out.split(">", 1)[1], "the visible text was not softened"


def test_an_entity_is_one_unit_and_is_never_cut():
    """`&am` + break + `p;` is drawn as the literal characters. (`x&amp;` is
    six characters, so a break every 20 CHARACTERS lands inside an entity.)"""
    run = "x&amp;" * 50
    out = markup.soften_long_tokens(f"<p>{run}</p>")
    assert out.replace(markup.ZWSP, "") == f"<p>{run}</p>"
    pieces = out[3:-4].split(markup.ZWSP)
    assert len(pieces) > 1
    for piece in pieces:
        assert re.fullmatch(r"(?:x|&amp;)+", piece), piece


def test_a_letter_keeps_its_accent_and_a_flag_its_other_half():
    # The leading "x" puts the pairs OFF the 20-character grid: without it a
    # break every 20 characters happens to fall between pairs, and a version
    # that knows nothing about marks passes.
    out = markup.soften_long_tokens("<p>x" + "e\u0301" * 60 + "</p>")
    pieces = out[3:-4].split(markup.ZWSP)
    assert len(pieces) > 2 and all(not piece.startswith("\u0301") for piece in pieces)
    flags = "x" + "\U0001F1EC\U0001F1F7" * 45
    for piece in markup.soften_long_tokens(f"<p>{flags}</p>")[3:-4].split(markup.ZWSP):
        indicators = sum(1 for ch in piece if "\U0001F1E6" <= ch <= "\U0001F1FF")
        assert indicators % 2 == 0, "a break fell inside a regional-indicator pair"


def test_every_tag_gets_chat_sized_type_and_long_runs_are_softened():
    out = markup.style_markup(
        "<h1>T</h1><p>p</p><ul><li>i</li></ul><table><tr><td>c</td></tr></table>"
        '<a href="https://x.test">l</a><blockquote>q</blockquote><code>k</code><p>' + "z" * 90 + "</p>",
        "MONO", "#acc", "#dim")
    for tag, needle in (("h1", "font-size:22px"), ("p", "line-height:148%"), ("li", "line-height:145%"),
                        ("td", "padding:3px 11px"), ("a", "color:#acc"), ("blockquote", "#dim"),
                        ("code", "font-family:MONO")):
        style = re.search(rf"<{tag}\b[^>]*style=\"([^\"]*)\"", out).group(1)
        assert needle in style, (tag, style)
    assert markup.ZWSP in out, "style_markup must soften long runs (it is the bubble's only pass)"


def test_code_blocks_wrap():
    out = markup.style_markup("<pre><code>x = 1</code></pre>", "mono", "#acc", "#dim")
    pre = re.search(r"<pre style=\"([^\"]*)\"", out).group(1)
    assert "white-space:pre-wrap" in pre, "a code line wider than the bubble is clipped"


def test_an_element_that_arrives_styled_is_merged_not_skipped():
    """An aligned table cell arrives as `<td style="text-align: left;">`.
    Skipping every already-styled tag left such tables unpadded; and a
    `style=` inside an href or another attribute is not a style attribute."""
    out = markup.style_markup(
        '<td style="text-align: left;">x</td>'
        '<a href="https://x.test/?style=dark">l</a><p data-style="1">p</p>', "mono", "#acc", "#dim")
    assert '<td style="padding:3px 11px 3px 0; text-align: left;">' in out      # ours first, theirs wins
    assert 'href="https://x.test/?style=dark" style="color:#acc;' in out
    assert '<p data-style="1" style="margin:' in out


def test_a_reply_image_becomes_a_tappable_glyph_and_is_listed():
    text = ("First ![ext](http://x/y.png) and text then ![gen](/api/download/z.png) "
            'and ![t](/api/download/a.png "My title").')
    html = markup.render_reply(text, markdown_fn=lambda t: t)
    # the foreign image and the prose between are left exactly as written
    assert html.startswith("First ![ext](http://x/y.png) and text then <br><a href=\"/api/download/z.png\"")
    assert 'href="/api/download/a.png"' in html and html.count("🖼️") == 2 and html.endswith("</a>.")
    assert markup.reply_images(text) == ["/api/download/z.png", "/api/download/a.png"]


def test_an_image_inside_a_code_block_is_an_example():
    text = "Write it like this:\n```\n![alt](/api/download/no.png)\n```\nand ~~~\n![x](/api/download/no2.png)\n~~~"
    assert markup.reply_images(text) == []
    assert markup.render_reply(text, markdown_fn=lambda t: t) == text


def test_the_alt_text_cannot_rewrite_the_link():
    html = markup.render_reply('![x" href="https://evil.test" a="<b>](/api/download/p.png)',
                               markdown_fn=lambda t: t)
    assert html.count('href="') == 1 and "&quot;" in html and "<b>" not in html


def test_render_reply_uses_the_markdown_package_by_default():
    html = markup.render_reply("**bold**\n\n```\ncode\n```")
    assert "<strong>bold</strong>" in html and "<pre>" in html


def test_a_stored_conversation_becomes_transcript_rows():
    rows = markup.transcript_items([
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "text", "text": ""},
                                     {"type": "text", "text": "what is this"},
                                     {"type": "image_url", "image_url": {"url": "data:..."}},
                                     {"type": "text", "text": "and this"}]},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]},
        {"role": "assistant", "content": 'done <tool_call>{"name":"x"}</tool_call>'},
        {"role": "assistant", "content": "<tool_call>{\"name\":\"only\"}</tool_call>"},
        {"role": "assistant", "content": [{"type": "text", "text": "listed"}]},
        {"role": "assistant", "content": "The <tool_call> tag wraps a JSON call. The rest stays."},
        {"role": "tool", "content": "raw tool output"},
        "not a message",
    ])
    assert rows == [("user", "plain"), ("user", "what is this\nand this"), ("user", "[attachment]"),
                    ("assistant", "done"), ("assistant", "listed"),
                    ("assistant", "The <tool_call> tag wraps a JSON call. The rest stays.")]


def test_the_operators_own_text_is_escaped():
    assert markup.escape_user("if a < b && c > d:\n  go") == \
        "if a &lt; b &amp;&amp; c &gt; d:<br>  go"


# ════════════════════════════════════════════════════════════════════════════
# speech — what is said aloud
# ════════════════════════════════════════════════════════════════════════════

def _stream(text, sizes=(7,)):
    """Feed `text` in uneven chunks, as a token stream arrives."""
    ch = speech.SpeechChunker()
    out, i, k = [], 0, 0
    while i < len(text):
        n = sizes[k % len(sizes)]
        out += ch.feed(text[i:i + n])
        i += n
        k += 1
    return out + ch.flush()


def test_a_sentence_is_spoken_as_soon_as_it_is_complete():
    ch = speech.SpeechChunker()
    assert ch.feed("Hello there. And th") == ["Hello there."]
    assert ch.feed("en more") == []
    assert ch.flush() == ["And then more"]


SPOKEN = [
    # code is never read aloud
    ("Run this.\n```python\nimport os\nprint(os.getcwd())  # a comment. With a stop.\n```\nThen check the output.",
     ["Run this.", "Then check the output."]),
    # a fence indented under a list item is still a fence
    ("1. Do this:\n   ```\n   rm -rf build.\n   ```\n2. Then that.", ["Do this:", "Then that."]),
    # a line that merely STARTS with inline code is prose — it used to open
    # a "fence" that silenced the rest of the reply
    ("```ls -la``` lists files. It is useful.\nSecond paragraph is prose.\n",
     ["ls -la lists files.", "It is useful.", "Second paragraph is prose."]),
    # only the same marker, at least as long, closes a block
    ("Before.\n````md\n```py\nx = compute(1).\n```\ny = 2\n````\nAfter.", ["Before.", "After."]),
    ("Run:\n```sh\ncat <<EOF\n~~~\nEOF\n```\nThen done.", ["Run:", "Then done."]),
    ("A.\n~~~\nraw. text.\n~~~\nB.", ["A.", "B."]),
    # a closing fence stands ALONE on its line: ```js inside a block is content
    ("A.\n```\n```not a close\nstill code.\n```\nB.", ["A.", "B."]),
    # a fence marker in the MIDDLE of a line is not a fence
    ("Use ``` to open a block. Then go on.", ["Use to open a block.", "Then go on."]),
    ("First point. ``` is the marker. Next.", ["First point.", "is the marker.", "Next."]),
    # tables, with and without outer pipes; a rule line is never spoken
    ("Results:\n\n| name | score |\n|------|-------|\n| a | 1. |\n\n---\nDone.", ["Results:", "Done."]),
    ("Results:\n\nName | Value\n---|---\ncpu | 93.5 percent\nmem | 40\n\nThat is all.",
     ["Results:", "That is all."]),
    # …but a pipe in prose is prose
    ("|x| is the absolute value of x. It is never negative.\n",
     ["|x| is the absolute value of x.", "It is never negative."]),
    ("Use a | b to pipe. It works.\n| x | y |\n|---|---|\n| 1 | 2 |\nEnd.",
     ["Use a | b to pipe.", "It works.", "End."]),
    # a table ENDS: prose after it — or after a code block that follows it —
    # is prose again, pipe or no pipe
    ("| a | b |\n|---|---|\n| 1 | 2 |\nEnd of table.\nUse a | b here.\nOk.",
     ["End of table.", "Use a | b here.", "Ok."]),
    ("| a | b |\n```\ncode\n```\nx | y is prose.\nEnd.", ["x | y is prose.", "End."]),
    # a link is its label, whatever punctuation the label holds; an image is nothing
    ("Source: [Is Rust faster than C? - Stack Overflow](https://so.test/q/1). Next one.",
     ["Source: Is Rust faster than C?", "Stack Overflow.", "Next one."]),
    ("![A cat on a mat. Photorealistic.](/api/download/img_1.png)\nDone.", ["Done."]),
    ("See https://x.test/a. Next sentence here.", ["See a link.", "Next sentence here."]),
    # list markers and headings are not utterances
    ("1. First thing.\n2. Second thing.\n", ["First thing.", "Second thing."]),
    ("## Plan\n- one\n- two\n", ["Plan", "one", "two"]),
    # an unclosed bracket does not hold the reply hostage past its line
    ("See [1. Then go on.\nNew line.", ["See [1.", "Then go on.", "New line."]),
    ("One.\r\nTwo.\r\n```\r\ncode.\r\n```\r\nThree.", ["One.", "Two.", "Three."]),
    ("no terminator and no newline", ["no terminator and no newline"]),
    ("Wait... — !!\nOk.", ["Wait...", "Ok."]),
]


@pytest.mark.parametrize("sizes", [(1,), (2,), (3,), (5,), (7,), (11,), (400,)])
@pytest.mark.parametrize("text, want", SPOKEN)
def test_what_is_spoken_however_the_stream_is_cut(text, want, sizes):
    """The old splitter was a regex over a rolling buffer; it read code, table
    rows and URLs aloud. Every case is fed at every chunk size, because a
    marker can arrive split across chunks (`` ` `` then `` `` ``)."""
    assert _stream(text, sizes) == want


_FRAGMENTS = ["Hello. ", "World? ", "ok ", "`x` ", "```", "```py", "~~~", "\n", "\n\n", "| a | b |",
              "a | b", "---|---", "[lab. el](http://u.test/a.b) ", "![al. t](/api/download/x.png) ",
              "https://a.test/b. ", "[open ", "] ", "1. ", "- ", "**b** ", "    code. ", "|x| y. ",
              "\r\n", "!", "(", ")", "####", " ", "](", "[", "www.x.test/y? ", "e.g. ", "... ", "|",
              "\n|---|---|\n"]


def test_the_utterances_do_not_depend_on_how_the_stream_is_cut():
    """The chunker's contract. 4,000 random documents, each fed whole, one
    character at a time, and in random chunks: the three must agree. (A
    count-based "is a link still open?" check and a table-header hold both
    failed this before they were fixed.)"""
    import random
    rnd = random.Random(20261001)
    for _ in range(4000):
        text = "".join(rnd.choice(_FRAGMENTS) for _ in range(rnd.randint(1, 16)))
        whole = _stream(text, (len(text) + 1,))
        assert _stream(text, (1,)) == whole, repr(text)
        sizes = tuple(rnd.choice((1, 2, 3, 5, 8, 13)) for _ in range(7))
        assert _stream(text, sizes) == whole, (repr(text), sizes)


def test_links_images_and_marks_are_spoken_as_words():
    assert speech.speakable("See [the docs](https://x.test/a_b) or https://y.test/z?q=1 now.") == \
        "See the docs or a link now."
    assert speech.speakable("![chart](/api/download/c.png)") == ""
    assert speech.speakable("## **Big** _news_ about `snake_case`") == "Big news about snake case"
    assert speech.speakable("- item one") == "item one"
    # a `#` after a letter is a name and a lone `*` is arithmetic: both stay
    assert speech.speakable("C# and F# are languages") == "C# and F# are languages"
    assert speech.speakable("2 * 3 = 6") == "2 * 3 = 6"
    assert speech.speakable("~~gone~~ and *kept*") == "gone and kept"
    # nothing left to pronounce
    for nothing in ("***", "", "...", "— !", "`  `", "(https://x.test)"[:1]):
        assert speech.speakable(nothing) == "", nothing


def test_an_unclosed_fence_swallows_the_tail_and_the_next_reply_starts_clean():
    ch = speech.SpeechChunker()
    assert ch.feed("Before.\n```\ncode with no end. Really.") == ["Before."]
    assert ch.flush() == []
    assert ch.feed("A new reply. ") == ["A new reply."]
    ch = speech.SpeechChunker()                       # …and a table left open at the end
    assert ch.feed("| a | b |\n") == [] and ch.flush() == []
    assert ch.feed("fresh | prose. ok\n") + ch.flush() == ["fresh | prose.", "ok"]


def test_the_chime_is_a_playable_wav_that_fades():
    raw = speech.chime_wav()
    with wave.open(io.BytesIO(raw)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        frames = w.readframes(w.getnframes())
    samples = [int.from_bytes(frames[i:i + 2], "little", signed=True)
               for i in range(0, len(frames), 2)]
    assert 0.15 < len(samples) / 16000 < 0.4
    assert max(abs(s) for s in samples) > 3000, "inaudible"
    # an unfaded sine starts and stops with a click louder than the tone
    assert abs(samples[0]) < 200 and abs(samples[-1]) < 200
    assert max(abs(s) for s in speech.chime_wav(volume=0.0)[44:]) == 0


@pytest.mark.parametrize("elapsed, off, tts, want", [
    (5, False, False, False),      # a quick answer the operator is watching
    (19.9, False, False, False),
    (20, False, False, True),      # the threshold itself counts as long
    (45, False, False, True),      # a long wait
    (5, True, False, True),        # the panel went dark — they have stopped watching
    (45, True, True, False),       # replies are spoken: the voice is the notification
])
def test_when_a_reply_chimes(elapsed, off, tts, want):
    assert speech.should_chime(elapsed, off, tts) is want


def test_clock_label():
    assert [speech.clock_label(s) for s in (0, 7.9, 60, 125, -3)] == \
        ["0:00", "0:07", "1:00", "2:05", "0:00"]


# ════════════════════════════════════════════════════════════════════════════
# devstatus — the handheld's own state
# ════════════════════════════════════════════════════════════════════════════

# /proc/net/wireless, verbatim from the device (2026-10-01).
WIRELESS = (
    "Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE\n"
    " face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\n"
    " wlan0: 0000   53.  -57.  -256        0      0      0      3      0        0\n")


def _supply(root, name, capacity=None, status=None):
    d = root / name
    d.mkdir()
    if capacity is not None:
        (d / "capacity").write_text(f"{capacity}\n")
    if status is not None:
        (d / "status").write_text(f"{status}\n")


def test_battery_is_read_from_the_supply_that_reports_one(tmp_path):
    _supply(tmp_path, "axp22x-ac")                          # the charger: no capacity
    _supply(tmp_path, "axp20x-battery", 87, "Discharging")
    assert devstatus.read_battery(str(tmp_path)) == (87, "discharging")
    (tmp_path / "axp20x-battery" / "status").write_text("Charging\n")
    assert devstatus.read_battery(str(tmp_path)) == (87, "charging")
    for status in ("Full", "Not charging"):                  # on the charger, topped up
        (tmp_path / "axp20x-battery" / "status").write_text(f"{status}\n")
        assert devstatus.read_battery(str(tmp_path)) == (87, "full")
    (tmp_path / "axp20x-battery" / "status").write_text("Unknown\n")
    assert devstatus.read_battery(str(tmp_path)) == (87, "unknown")


def test_a_missing_or_garbled_battery_is_unknown_not_a_crash(tmp_path):
    assert devstatus.read_battery(str(tmp_path / "nope")) == (None, "unknown")
    for i, garbage in enumerate(("n/a", "inf", "")):
        _supply(tmp_path, f"BAT{i}", garbage, "Charging")
    assert devstatus.read_battery(str(tmp_path)) == (None, "unknown")


def test_the_battery_is_this_machines_and_is_a_percentage(tmp_path):
    """A paired headset is a power supply too (scope "Device"), and sorts
    first; and a gauge that over-reads must not print 103%."""
    _supply(tmp_path, "AirPods-battery", 3, "Discharging")
    (tmp_path / "AirPods-battery" / "scope").write_text("Device\n")
    _supply(tmp_path, "AAA-mains", 77, "Charging")            # sorts first; not a battery at all
    _supply(tmp_path, "BAT0", 103, "Charging")
    assert devstatus.read_battery(str(tmp_path)) == (100, "charging")
    (tmp_path / "BAT0" / "capacity").write_text("-4\n")
    assert devstatus.read_battery(str(tmp_path)) == (0, "charging")


def test_wifi_quality_is_read_from_proc(tmp_path):
    f = tmp_path / "wireless"
    f.write_text(WIRELESS)
    assert devstatus.read_wifi(str(f)) == 76                 # 53 of 70
    f.write_text(WIRELESS.replace("   53.  ", "   83.  "))
    assert devstatus.read_wifi(str(f)) == 100                # a driver that over-reports
    f.write_text(WIRELESS)
    f.write_text(WIRELESS.splitlines(keepends=True)[0] + WIRELESS.splitlines(keepends=True)[1])
    assert devstatus.read_wifi(str(f)) is None               # no interface line
    assert devstatus.read_wifi(str(tmp_path / "absent")) is None


def test_wifi_bars():
    assert [devstatus.wifi_bars(p) for p in (None, 0, 10, 30, 60, 90)] == \
        ["", "▁", "▂", "▂▄", "▂▄▆", "▂▄▆█"]


def test_the_status_readout_says_what_changed():
    kw = dict(ok="#ok", danger="#bad", dim="#dim")
    up = devstatus.status_html(True, 76, 100, "charging", "10:30 AM", **kw)
    assert "color:#ok;'>●" in up and "▂▄▆█" in up and "⚡ 100%" in up and "10:30 AM" in up
    # the bolt used to be printed ALWAYS, so it said nothing
    on_battery = devstatus.status_html(True, 76, 60, "discharging", "t", **kw)
    assert "⚡" not in on_battery and "60%" in on_battery and "#bad" not in on_battery
    low = devstatus.status_html(True, None, 9, "discharging", "t", **kw)
    assert "color:#bad;'>9%" in low
    at_limit = devstatus.status_html(True, None, devstatus.LOW_BATTERY_PCT, "discharging", "t", **kw)
    assert "#bad" not in at_limit and f"{devstatus.LOW_BATTERY_PCT}%" in at_limit
    assert "color:#bad" not in devstatus.status_html(True, None, 9, "charging", "t", **kw)
    down = devstatus.status_html(False, None, None, "unknown", "t", **kw)
    assert "color:#bad;'>●" in down and "--%" in down
    assert "color:#dim;'>●" in devstatus.status_html(None, None, 50, "full", "t", **kw)


@pytest.mark.parametrize("arg, want", [
    ("", 3), ("+", 4), ("-", 2), ("9", 9), ("42", 9), ("0", 1),
    ("+2", 5), ("-1", 2), ("-5", 1), ("+40", 9),          # a SIGNED number is a change
    ("bright", None), ("3.5", None), ("+-1", None),
])
def test_brightness_steps_and_never_reaches_zero(tmp_path, arg, want):
    """At zero the panel is black, and the command that would turn it back up
    has to be typed blind."""
    d = tmp_path / "backlight@0"
    d.mkdir()
    (d / "brightness").write_text("3\n")
    (d / "max_brightness").write_text("9\n")
    got = devstatus.set_backlight(arg, str(tmp_path))
    if want is None:
        assert got is None and (d / "brightness").read_text().strip() == "3"
    else:
        assert got == (want, 9) and (d / "brightness").read_text().strip() == str(want)


def test_no_backlight_is_reported_not_raised(tmp_path):
    assert devstatus.set_backlight("+", str(tmp_path)) is None
    assert devstatus.read_backlight(str(tmp_path / "absent")) is None
    d = tmp_path / "bl"                                    # present, but max is garbage
    d.mkdir()
    (d / "brightness").write_text("3\n")
    (d / "max_brightness").write_text("?\n")
    assert devstatus.read_backlight(str(tmp_path)) is None
    assert devstatus.set_backlight("+", str(tmp_path)) is None


def test_a_brightness_write_that_is_refused_is_reported(tmp_path):
    d = tmp_path / "bl"
    d.mkdir()
    (d / "brightness").write_text("3\n")
    (d / "max_brightness").write_text("9\n")
    (d / "brightness").chmod(0o444)
    try:
        if os.access(d / "brightness", os.W_OK):
            pytest.skip("running as a user that ignores file modes")
        assert devstatus.set_backlight("+", str(tmp_path)) is None, "claimed a level it did not set"
        assert devstatus.set_backlight("", str(tmp_path)) == (3, 9)      # a query needs no write
    finally:
        (d / "brightness").chmod(0o644)


def test_volume_is_absolute_and_capped():
    """A relative `10%+` with no limit walks PipeWire past unity into clipping."""
    assert devstatus.parse_volume("Volume: 1.00\n") == 100          # verbatim wpctl output
    assert devstatus.parse_volume("Volume: 0.45 [MUTED]") == 45
    assert devstatus.volume_muted("Volume: 0.45 [MUTED]") and not devstatus.volume_muted("Volume: 0.45")
    assert devstatus.parse_volume("Volume: 0,45") == 45              # a comma-decimal locale
    assert devstatus.parse_volume("") is None
    sink = devstatus.SINK
    assert devstatus.volume_command("+", 50) == ["wpctl", "set-volume", sink, "60%"]
    assert devstatus.volume_command("-10", 60) == ["wpctl", "set-volume", sink, "50%"]   # not "to 0"
    assert devstatus.volume_command("+", 95) == ["wpctl", "set-volume", sink, "100%"]
    assert devstatus.volume_command("-", 5) == ["wpctl", "set-volume", sink, "0%"]
    assert devstatus.volume_command("35", 80) == ["wpctl", "set-volume", sink, "35%"]
    assert devstatus.volume_command("250", 80) == ["wpctl", "set-volume", sink, "100%"]
    assert devstatus.volume_command("", 80) is None and devstatus.volume_command("loud", 80) is None


def test_the_panel_helpers():
    env = devstatus.panel_env({"PATH": "/bin"}, uid=1000)
    assert env["XDG_RUNTIME_DIR"] == "/run/user/1000" and env["WAYLAND_DISPLAY"] == "wayland-0"
    kept = devstatus.panel_env({"WAYLAND_DISPLAY": "wayland-1", "XDG_RUNTIME_DIR": "/x"}, uid=1)
    assert (kept["WAYLAND_DISPLAY"], kept["XDG_RUNTIME_DIR"]) == ("wayland-1", "/x")
    assert devstatus.panel_is_off("DSI-1 on\n") is False             # verbatim wlopm output
    assert devstatus.panel_is_off("DSI-1 off\n") is True
    assert devstatus.panel_is_off("DSI-1 off\nHDMI-A-1 on\n") is False
    # unknown is not "off": a failed wlopm must not wake (or chime for) a lit panel
    assert devstatus.panel_is_off("") is None
    assert devstatus.panel_is_off("failed to connect to display") is None


@pytest.mark.parametrize("idle, busy, battery, env, want", [
    (0, False, False, {}, 0),            # just typed
    (44, False, False, {}, 0),
    (45, False, False, {}, 10),          # nobody is looking
    (45, False, True, {}, 5),            # …and it is costing battery
    (599, False, False, {}, 10),
    (600, False, False, {}, -1),         # the panel has blanked by now
    (9999, True, True, {}, 0),           # a turn in flight always animates
    (100, False, False, {"GHOST_FACE_IDLE_FPS": "0"}, 0),        # slow tier off
    (100, False, False, {"GHOST_FACE_IDLE_S": "300"}, 0),
    (700, False, False, {"GHOST_FACE_SLEEP_S": "0"}, 10),        # never pause
    (100, False, False, {"GHOST_FACE_IDLE_FPS": "banana"}, 10),  # a typo keeps the default
    (100, False, False, {"GHOST_FACE_IDLE_FPS": "nan"}, 10),     # …and so does a non-number
    (100, False, False, {"GHOST_FACE_IDLE_FPS": "1e999"}, 10),
    (100, False, False, {"GHOST_FACE_IDLE_S": "inf"}, 10),
    (100, False, True, {"GHOST_FACE_BATTERY_FPS": "0.5"}, 1),    # never truncated to 0 = UNCAPPED
    (100, False, False, {"GHOST_FACE_IDLE_FPS": "-3"}, 0),
    (100, False, False, {"GHOST_FACE_IDLE_FPS": "12.9"}, 12),
])
def test_the_face_slows_when_idle_and_pauses_when_left(idle, busy, battery, env, want):
    assert devstatus.face_rate(idle, busy, battery, environ=env) == want


# ════════════════════════════════════════════════════════════════════════════
# agentapi — sessions, cancel, feedback, notifications
# ════════════════════════════════════════════════════════════════════════════

class _Resp:
    def __init__(self, status=200, data=None, text=None):
        self.status_code = status
        self._data = data
        self.text = text if text is not None else json.dumps(data if data is not None else {})

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class _Http:
    """An httpx.AsyncClient double: answers from a script, records calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def _next(self, method, url, **kw):
        self.calls.append((method, url, kw.get("json"), kw.get("params"), kw.get("headers")))
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    async def get(self, url, **kw):
        return await self._next("GET", url, **kw)

    async def post(self, url, **kw):
        return await self._next("POST", url, **kw)


def _run(coro):
    return asyncio.run(coro)


async def _no_sleep(_s):
    return None


def test_a_session_id_survives_a_restart(tmp_path):
    path = tmp_path / "deep" / "session_id"
    sid = agentapi.new_session_id()
    assert agentapi.valid_session_id(sid) and sid != agentapi.new_session_id()
    assert agentapi.save_session_id(sid, str(path)) is True
    assert agentapi.load_session_id(str(path)) == sid
    assert [p.name for p in path.parent.iterdir()] == ["session_id"], "temp litter left behind"


@pytest.mark.parametrize("bad", ["", "has space", "a/b", "x" * 65, "trailing\n", "dot.ted", None, 7])
def test_an_id_the_agent_would_not_persist_is_never_saved_or_loaded(tmp_path, bad):
    """The agent accepts a turn with such an id and silently does NOT store
    it — the conversation would look durable and not be."""
    path = tmp_path / "s"
    assert agentapi.save_session_id(bad, str(path)) is False and not path.exists()
    if not isinstance(bad, str) or bad != bad.strip():
        return          # the FILE always ends in a newline; loading strips it, by design
    path.write_text(bad)
    assert agentapi.load_session_id(str(path)) is None


def test_the_session_id_guard_is_the_agents_own():
    from ghost_agent.core import sessions
    assert agentapi._SESSION_ID_RE.pattern == sessions._ID_RE.pattern


@pytest.mark.parametrize("resp, want", [
    (_Resp(200, {"id": "s", "messages": [{"role": "user", "content": "q"}]}),
     ([{"role": "user", "content": "q"}], "ok")),
    (_Resp(404, {"detail": "session not found"}), ([], "missing")),
    (_Resp(503, {"detail": "sessions are not enabled"}), ([], "disabled")),
    (_Resp(500, None, "boom"), ([], "error")),
    (_Resp(200, None, "<html>"), ([], "error")),
    (_Resp(200, {"messages": "nope"}), ([], "ok")),
    (ConnectionRefusedError("down"), ([], "error")),
])
def test_fetching_a_session_names_what_happened(resp, want):
    http = _Http([resp])
    assert _run(agentapi.fetch_session(http, "http://a", "k", "cw-1")) == want
    assert http.calls[0][1] == "http://a/api/sessions/cw-1"
    assert http.calls[0][4] == {"X-Ghost-Key": "k"}


def test_history_is_replayed_as_stored():
    """The agent aligns a replayed history by (role, str(content)); a content
    object re-serialised here would be appended to the conversation again."""
    image_turn = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "d"}}]
    stored = [{"role": "system", "content": "p"},
              {"role": "user", "content": image_turn},
              {"role": "assistant", "content": "seen", "prefixLen": 3},
              {"role": "tool", "content": "raw"}]
    # a stored assistant turn that was only tool calls has no content: it is
    # not replayed as an assistant message that says nothing
    stored += [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
               {"role": "assistant", "content": ""}]
    got = agentapi.history_for_model(stored)
    assert got == [{"role": "user", "content": image_turn}, {"role": "assistant", "content": "seen"}]
    assert got[0]["content"] is image_turn
    from ghost_agent.core.sessions import merge_history_detail
    merged, new = merge_history_detail(stored, got + [{"role": "user", "content": "next"}])
    assert new == [{"role": "user", "content": "next"}] and len(merged) == len(stored) + 1


def test_listing_sessions():
    rows = [{"id": "cw-1", "title": "one", "updated_at": 5.0, "message_count": 4},
            {"title": "no id"}, "garbage", {"id": "web-2", "title": "two"}]
    http = _Http([_Resp(200, {"enabled": True, "sessions": rows})])
    assert _run(agentapi.list_sessions(http, "http://a", "k", limit=8)) == [rows[0], rows[3]]
    assert http.calls[0][1] == "http://a/api/sessions" and http.calls[0][3] == {"limit": 8}
    assert _run(agentapi.list_sessions(_Http([_Resp(200, {"enabled": True, "sessions": [rows[0]] * 20})]),
                                       "http://a", "k", limit=3)) == [rows[0]] * 3
    for bad in (_Resp(503, {}), _Resp(200, None, "x"), _Resp(200, {"sessions": "x"}), OSError()):
        assert _run(agentapi.list_sessions(_Http([bad]), "http://a", "k")) == []


@pytest.mark.parametrize("age, want", [
    (0, "just now"), (89, "just now"), (90, "1 min ago"), (5399, "89 min ago"),
    (5400, "1 h ago"), (129599, "35 h ago"), (129600, "1 d ago"), (10 * 86400, "10 d ago"),
])
def test_age_text(age, want):
    assert agentapi.age_text(1_000_000 - age, now=1_000_000) == want
    assert agentapi.age_text("garbage") == "" and agentapi.age_text(None) == ""


def test_the_request_id_comes_from_the_frames():
    assert agentapi.frame_request_id({"id": "chatcmpl-4f6dc15d"}) == "4f6dc15d"
    assert agentapi.frame_request_id({"id": "chatcmpl-4f6dc15d#2"}) == "4f6dc15d#2"   # uniquified
    for frame in ({"id": "4f6dc15d"}, {"id": "chatcmpl-"}, {"id": 7}, {}, None, "x"):
        assert agentapi.frame_request_id(frame) is None
    assert re.fullmatch(r"[0-9a-f]{8}", agentapi.new_request_id())
    assert agentapi.frame_unlabelable({"ghost": {"labelable": False}}) is True
    for frame in ({"ghost": {"labelable": True}}, {"ghost": {}}, {"ghost": None}, {}, None):
        assert agentapi.frame_unlabelable(frame) is False


@pytest.mark.parametrize("frame, want", [
    ({"choices": [{"index": 0, "delta": {"content": "hi"}}]}, "hi"),
    ({"message": {"content": "ollama style"}}, "ollama style"),
    # the usage frame: `choices` is EMPTY. Indexing it raised IndexError out
    # of the stream — the reply ended in a fault and a held reply was lost.
    ({"choices": [], "usage": {"prompt_tokens": 9}}, ""),
    ({"choices": [{"index": 0, "delta": {}}]}, ""),
    ({"choices": [{"index": 0, "delta": {"content": None}}]}, ""),
    ({"choices": [{"index": 0}]}, ""), ({"choices": [None]}, ""), ({"choices": "x"}, ""),
    ({"message": None, "choices": [{"delta": {"content": "falls through"}}]}, "falls through"),
    ({"message": {"content": ""}, "choices": [{"delta": {"content": "second"}}]}, "second"),
    ({}, ""), (12, ""), (None, ""), ([], ""),
])
def test_the_text_of_any_frame(frame, want):
    assert agentapi.frame_content(frame) == want


def test_an_error_frame_is_one_with_no_choices():
    assert agentapi.frame_error({"error": {"message": "boom"}}) == "boom"
    assert agentapi.frame_error({"error": "stalled"}) == "stalled"
    # an OpenAI-style frame may carry both; then it is a content frame
    assert agentapi.frame_error({"error": "x", "choices": []}) is None
    for frame in ({"error": None}, {"error": ""}, {}, 12, None):
        assert agentapi.frame_error(frame) is None


TURNS = {"turns": [
    {"request_id": "dream1", "running": True, "session_id": None, "preview": "### SYNTHETIC"},
    {"request_id": "mine01", "running": False, "session_id": "cw-me", "preview": "what is the time"},
    {"request_id": "web001", "running": False, "session_id": "web-x", "preview": "what is the time in Paris"},
]}


def test_background_busy_is_a_running_turn_that_is_not_ours():
    assert agentapi.background_busy(TURNS, "cw-me") is True              # the dream holds the lock
    assert agentapi.background_busy(TURNS, "cw-me", own_request_id="dream1") is False
    ours = {"turns": [{"request_id": "r1", "running": True, "session_id": "cw-me"}]}
    assert agentapi.background_busy(ours, "cw-me") is False
    assert agentapi.background_busy({"turns": [{"request_id": "q", "running": False}]}, "cw-me") is False
    assert agentapi.background_busy(None, "cw-me") is False
    assert agentapi.background_busy({"turns": "garbage"}, "cw-me") is False


def test_fetching_turns_is_also_the_reachability_probe():
    assert _run(agentapi.fetch_turns(_Http([_Resp(200, TURNS)]), "http://a", "k")) == TURNS
    for bad in (_Resp(503, {}), _Resp(200, None, "x"), _Resp(200, ["x"]), ConnectionRefusedError()):
        assert _run(agentapi.fetch_turns(_Http([bad]), "http://a", "k")) is None


def test_stop_cancels_the_id_minted_at_send_time():
    http = _Http([_Resp(200, {"cancelled": True})])
    assert _run(agentapi.cancel_turn(http, "http://a", "k", "abcd1234", sleep=_no_sleep)) == \
        ("cancelled", "abcd1234")
    assert http.calls == [("POST", "http://a/api/turn/cancel", {"request_id": "abcd1234"},
                           None, {"X-Ghost-Key": "k"})]


def test_a_forced_stop_says_hard():
    http = _Http([_Resp(200, {"cancelled": True})])
    _run(agentapi.cancel_turn(http, "http://a", "k", "abcd1234", hard=True, sleep=_no_sleep))
    assert http.calls[0][2] == {"request_id": "abcd1234", "hard": True}


def test_stop_never_reaches_for_another_turn():
    """Ownership is the request id and nothing else. The first version fell
    back, on a 404, to "the turn in my session" and "the turn whose text
    starts like mine" — and cancelled a browser's turn (run against the real
    registry by a reviewer). A 404 now only ever leads to the SAME id again."""
    http = _Http([_Resp(404, {"cancelled": False}), _Resp(404, {"cancelled": False})])
    assert _run(agentapi.cancel_turn(http, "http://a", "k", "cafe0001", sleep=_no_sleep)) == ("finished", "")
    assert [c[0] for c in http.calls] == ["POST", "POST"], "something other than the cancel was called"
    assert all(c[2] == {"request_id": "cafe0001"} for c in http.calls)


def test_a_stop_pressed_before_the_turn_registers_is_retried_once():
    """A 404 is "already over" OR "not registered yet". Believing the first
    at once reported a turn as stopped that then registered and ran."""
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    http = _Http([_Resp(404, {}), _Resp(200, {"cancelled": True})])
    assert _run(agentapi.cancel_turn(http, "http://a", "k", "abcd1234", sleep=sleep)) == \
        ("cancelled", "abcd1234")
    assert slept == [agentapi.CANCEL_RETRY_S] and len(http.calls) == 2


def test_with_no_id_nothing_is_cancelled():
    for missing in (None, ""):
        http = _Http([])
        assert _run(agentapi.cancel_turn(http, "http://a", "k", missing, sleep=_no_sleep)) == ("unknown", "")
        assert http.calls == [], "a cancel with no id means 'whatever holds the lock'"


@pytest.mark.parametrize("script, want", [
    ([_Resp(500, {})], ("failed", "HTTP 500")),
    ([_Resp(200, {"cancelled": False})], ("failed", "HTTP 200")),      # answered, did not take it
    ([_Resp(200, None, "<html>")], ("failed", "HTTP 200")),
    ([_Resp(404, {}), _Resp(503, {})], ("failed", "HTTP 503")),
    ([OSError("no route")], ("failed", "OSError: no route")),
])
def test_a_stop_the_agent_did_not_take_is_reported(script, want):
    assert _run(agentapi.cancel_turn(_Http(script), "http://a", "k", "abcd1234", sleep=_no_sleep)) == want


def test_feedback_is_posted_with_this_devices_source():
    http = _Http([_Resp(200, {"ok": True})])
    assert _run(agentapi.send_feedback(http, "http://a", "k", "4f6dc15d", "negative",
                                       note="wrong year", sleep=_no_sleep)) == (True, "")
    assert http.calls[0][1:3] == ("http://a/api/feedback", {
        "request_id": "4f6dc15d", "signal": "negative", "source": "clockwork", "note": "wrong year"})


@pytest.mark.parametrize("first", [404, 429, 500, 502, 503, 504])
def test_a_thumb_pressed_the_instant_the_reply_lands_is_retried_once(first):
    """The trajectory is written moments AFTER the stream ends, so the first
    attempt can 404; and a 503 is the agent restarting."""
    slept = []

    async def sleep(s):
        slept.append(s)

    http = _Http([_Resp(first, {"error": "no trajectory found"}), _Resp(200, {"ok": True})])
    assert _run(agentapi.send_feedback(http, "http://a", "k", "r", "positive", sleep=sleep)) == (True, "")
    assert slept == [agentapi.FEEDBACK_RETRY_S] and len(http.calls) == 2
    http = _Http([_Resp(first, {"error": "still nothing"}), _Resp(first, {"error": "still nothing"})])
    assert _run(agentapi.send_feedback(http, "http://a", "k", "r", "positive", sleep=_no_sleep)) == \
        (False, "still nothing")
    assert len(http.calls) == 2, "retried more than once"


def test_a_refused_label_is_not_retried():
    http = _Http([_Resp(400, {"error": "signal must be one of ['positive', 'negative']"})])
    ok, detail = _run(agentapi.send_feedback(http, "http://a", "k", "r", "positive", sleep=_no_sleep))
    assert ok is False and "signal must be" in detail and len(http.calls) == 1
    for rid, sig in (("", "positive"), ("r", "meh")):
        http = _Http([])
        assert _run(agentapi.send_feedback(http, "http://a", "k", rid, sig, sleep=_no_sleep))[0] is False
        assert http.calls == []


def test_the_signals_this_client_sends_are_ones_the_agent_accepts():
    """`send_feedback` refuses anything else before the network; this pins
    that what it lets through is what the agent's route validates against."""
    from ghost_agent.core.feedback import VALID_SIGNALS
    for signal in ("positive", "negative"):
        assert signal in VALID_SIGNALS
        http = _Http([_Resp(200, {"ok": True})])
        assert _run(agentapi.send_feedback(http, "http://a", "k", "r", signal, sleep=_no_sleep))[0] is True
        assert http.calls[0][2]["signal"] == signal
    assert "note" not in http.calls[0][2], "an empty note is sent as a field"
    http = _Http([_Resp(200, {"ok": True})])
    _run(agentapi.send_feedback(http, "http://a", "k", "r", "negative", note="x" * 900, sleep=_no_sleep))
    assert len(http.calls[0][2]["note"]) == 500           # the route's own cap


REC = {"ts": 1_000.0, "phase": "scheduled_task", "summary": "backup  finished\nok", "severity": "notify"}


def test_a_notification_is_one_plain_line_with_its_age_only_when_stale():
    assert agentapi.format_notification(REC, now=1_060.0) == "[scheduled task] backup finished ok"
    assert agentapi.format_notification(REC, now=1_000.0 + 7200) == \
        "[scheduled task] backup finished ok · 2 h ago"
    assert agentapi.format_notification({"phase": "some_new_phase", "summary": "x"}) == "[some new phase] x"


def test_notification_labels_match_the_slack_bots():
    src = (_ROOT / "interface" / "externals" / "slack_bot").glob("*.py")
    text = "".join(p.read_text() for p in src)
    block = text[text.index("_PHASE_LABELS = {"):]
    slack = dict(re.findall(r'"(\w+)":\s*"([^"]+)"', block[:block.index("}")]))
    assert slack and agentapi._PHASE_LABELS == slack


def _cycle(script, last_acked=None):
    poller = agentapi.NotifyPoller("http://a", "k")
    poller.last_acked = last_acked
    http = _Http(script)
    delivered = []
    n = _run(poller.cycle(http, delivered.extend))
    return poller, http, delivered, n


def test_notifications_are_delivered_then_acked():
    poller, http, delivered, n = _cycle([_Resp(200, {"enabled": True, "records": [REC], "watermark": 41}),
                                         _Resp(200, {"ok": True})])
    assert n == 1 and delivered == [REC] and poller.last_acked == 41
    assert http.calls[0][3] == {"consumer": "clockwork", "limit": 20}
    assert http.calls[1][1:3] == ("http://a/api/notifications/ack", {"consumer": "clockwork", "watermark": 41})


def test_a_disabled_ledger_is_never_acked():
    """Its watermark is a literal 0: acking it overwrites the stored offset,
    and when the ledger returns the whole history replays."""
    poller, http, delivered, n = _cycle([_Resp(200, {"enabled": False, "records": [], "watermark": 0})])
    assert n == 0 and len(http.calls) == 1 and poller.last_acked is None


def test_an_empty_poll_whose_watermark_moved_is_still_acked():
    """The scan window was all non-notify lines. Skipping this ack once wedged
    the Slack consumer for two days."""
    poller, http, _, n = _cycle([_Resp(200, {"enabled": True, "records": [], "watermark": 90}),
                                 _Resp(200, {"ok": True})], last_acked=41)
    assert n == 0 and len(http.calls) == 2 and poller.last_acked == 90


def test_an_idle_watermark_is_not_re_acked():
    _, http, _, _ = _cycle([_Resp(200, {"enabled": True, "records": [], "watermark": 41})], last_acked=41)
    assert len(http.calls) == 1


def test_a_failed_ack_is_not_recorded_as_done():
    poller, _, delivered, _ = _cycle([_Resp(200, {"enabled": True, "records": [REC], "watermark": 50}),
                                      _Resp(500, {})], last_acked=41)
    assert delivered == [REC] and poller.last_acked == 41, "a failed ack would never be retried"


def test_records_that_could_not_be_shown_are_not_acked():
    poller = agentapi.NotifyPoller("http://a", "k")
    http = _Http([_Resp(200, {"enabled": True, "records": [REC], "watermark": 50})])

    def boom(_records):
        raise RuntimeError("the transcript is gone")

    with pytest.raises(RuntimeError):
        _run(poller.cycle(http, boom))
    assert len(http.calls) == 1 and poller.last_acked is None


def test_a_reply_that_cannot_be_acked_delivers_nothing():
    """Records shown without an ack are served again on the next poll, and
    the one after — the same line in the transcript every minute. A reply
    whose watermark is not a number is malformed: nothing is shown."""
    for wm in (None, "41", 41.0, True):
        poller, http, delivered, n = _cycle([_Resp(200, {"enabled": True, "records": [REC], "watermark": wm})])
        assert (n, delivered, len(http.calls), poller.last_acked) == (0, [], 1, None), wm


@pytest.mark.parametrize("resp", [_Resp(500, {}), _Resp(200, None, "<html>"), _Resp(200, ["x"]),
                                  ConnectionRefusedError()])
def test_an_unreadable_poll_delivers_and_acks_nothing(resp):
    poller, http, delivered, n = _cycle([resp])
    assert (n, delivered, len(http.calls), poller.last_acked) == (0, [], 1, None)


def test_errors_are_said_in_words():
    class ConnectError(Exception):
        pass

    class ReadTimeout(Exception):
        pass

    class RemoteProtocolError(Exception):
        pass

    assert agentapi.describe_error(ConnectError("All connection attempts failed"), "http://eva:8000") == \
        "cannot reach eva:8000 — is it up, and is this device on the network?"
    assert agentapi.is_unreachable(ConnectError()) and not agentapi.is_unreachable(ReadTimeout())
    assert agentapi.describe_error(ReadTimeout(), "http://eva:8000") == "eva:8000 did not answer in time"
    assert "dropped the connection" in agentapi.describe_error(RemoteProtocolError(), "http://eva:8000")
    assert agentapi.describe_error(ValueError("odd"), "") == "ValueError: odd"
    assert agentapi.describe_error(KeyError(), "") == "KeyError"


def test_an_http_error_carries_the_servers_own_message():
    assert agentapi.describe_http_error(503, '{"error": {"message": "model is loading"}}') == \
        "HTTP 503 — model is loading — the agent is starting up or busy"
    assert agentapi.describe_http_error(401, '{"detail": "Invalid API key"}') == \
        "HTTP 401 — Invalid API key — the API key was refused (~/.ghost_api_key)"
    assert agentapi.describe_http_error(500, "Internal   Server\nError") == "HTTP 500 — Internal Server Error"
    assert agentapi.describe_http_error(502, "") == "HTTP 502"
    # a 422's `detail` is a LIST of validation errors — one bounded line, not its repr
    long_detail = json.dumps({"detail": [{"loc": ["body", "x"], "msg": "field required " * 40}] * 5})
    out = agentapi.describe_http_error(422, long_detail)
    assert out.startswith("HTTP 422 — ") and len(out) <= 220 and "\n" not in out
    assert len(agentapi.describe_error(ValueError("x" * 5000), "")) <= 220


# ════════════════════════════════════════════════════════════════════════════
# commands
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text, want", [
    ("/clear", ("clear", "")),
    ("  /NEW  ", ("new", "")),
    ("/bad it got the year wrong", ("bad", "it got the year wrong")),
    ("/bright +", ("bright", "+")),
    ("/open 3", ("open", "3")),
    ("/vol -10", ("vol", "-10")),
    # attached arguments and stray end punctuation are still the command
    ("/vol+", ("vol", "+")), ("/vol50", ("vol", "50")), ("/bright5", ("bright", "5")),
    ("/stop.", ("stop", "")), ("/help?", ("help", "")),
    # a command word followed by a SENTENCE is a message. `startswith('/clear')`
    # wiped the conversation on "/clearly…"; "a whole first word" then started
    # a new one on "/new idea: …" — the same defect one space later.
    ("/clearly this is a sentence", None),
    ("/new idea: use a queue instead", None),
    ("/clear this up for me", None),
    ("/help me fix the parser", None),
    ("/stop doing that please", None),
    ("/shutdown now please", None),
    # …and followed by ONE stray word it runs nothing
    ("/stop it", ("extra", "stop")), ("/shutdown now", ("extra", "shutdown")),
    ("/new 2", ("extra", "new")),
    # a command that takes an argument keeps whatever follows (its handler validates)
    ("/open the door please", ("open", "the door please")),
    # paths and prose
    ("/etc/hosts looks odd", None), ("/tmp/x", None), ("/face/ok", None),
    ("/vol5/x", None),                    # an attached argument never holds a slash
    ("not a /clear command", None), ("", None), ("/", None), ("//", None),
    ("/s that was sarcasm", None),
    ("/news about greece", None), ("/boot is 90% full", None), ("/tests pass now", None),
    # a typo is answered locally, not sent to the agent as a prompt
    ("/hlep", ("unknown", "hlep")), ("/zzzz", ("unknown", "zzzz")),
    ("/brigth 5", ("unknown", "brigth")), ("/shutdownx", ("unknown", "shutdownx")),
    ("/foo 5", None),                     # not near any command: a message
])
def test_what_counts_as_a_command(text, want):
    got = commands.parse(text)
    assert (tuple(got) if got else None) == want


def test_a_near_miss_is_suggested():
    assert commands.suggestion("hlep") == "help" and commands.suggestion("zzzzzz") == ""
    # the near-miss rule is neither "anything" nor "nothing"
    assert commands._near("brigth") and commands._near("sesions") and commands._near("reboto")
    assert commands._near("hlep")         # two letters swapped in a four-letter word: 0.75 exactly
    assert not commands._near("foo") and not commands._near("s") and not commands._near("volume")


def test_which_commands_take_an_argument_is_read_off_the_help_table():
    assert commands.TAKES_ARG == {"open", "bad", "bright", "vol"}


def test_ending_the_session_takes_typing_it_twice_in_a_row():
    now = [0.0]
    c = commands.Confirmer(window_s=10, clock=lambda: now[0])
    assert c.confirm("shutdown") is False
    now[0] = 9.9
    assert c.confirm("shutdown") is True
    assert c.confirm("shutdown") is False, "one confirmation must not arm the next"
    now[0] = 30.0
    assert c.confirm("shutdown") is False, "the window expired"
    assert c.confirm("reboot") is False, "a DIFFERENT command does not confirm"
    assert c.confirm("shutdown") is False
    c.disarm()                                  # anything else typed in between
    assert c.confirm("shutdown") is False
    now[0] = 5.0                                # the clock stepped BACKWARDS (an NTP sync)
    assert c.confirm("shutdown") is False, "a negative interval is not 'within the window'"


def test_every_command_is_listed_and_the_dangerous_ones_confirm():
    html = commands.help_html("#dim", "#acc", "mono")
    for name in commands.NAMES:
        assert f"/{name}" in html, f"/{name} exists but /help does not say so"
    assert set(commands.CONFIRM) == {"shutdown", "reboot", "exit"} <= set(commands.NAMES)


def test_every_command_is_handled_by_the_client():
    """A command that parses and then falls into the `unknown` branch."""
    src = (_DIR / "client.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_run_command")
    handled = {c.value for n in ast.walk(fn) if isinstance(n, ast.Compare)
               for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert set(commands.NAMES) <= handled, sorted(set(commands.NAMES) - handled)
    # …and the reverse: a branch for a name the parser can never produce is
    # a command that silently stopped existing.
    assert handled <= set(commands.NAMES) | {"extra"}, sorted(handled - set(commands.NAMES))


# ════════════════════════════════════════════════════════════════════════════
# the face: signals, and the frame-rate gear
# ════════════════════════════════════════════════════════════════════════════

STEPS = [
    ("web search", "🌐", "qwen release"), ("web read", "🌎", "https://x"), ("browser", "🌐", ""),
    ("memory search", "🔎", "who"), ("hydrated context", "📚", ""), ("file read", "📖", "notes.md"),
    ("sandbox exec", "🐚", "ls"), ("file write", "📝", "a.py"), ("tool call", "🔧", "execute"),
    ("something", "🧰", ""), ("recall", "📍", ""), ("verifier", "🧪", "CONFIRMED: grounded"),
    ("verifier", "🧪", "  refuted: no evidence"), ("verify depth", "🧭", "full"),
    ("verifier", "🧪", "UNCERTAIN"), ("planning", "📋", "3 steps"), ("", "", ""),
]


_RUN_THE_ORIGINAL = r"""
// Node reads app.js ITSELF, lifts the table and the function out of it by
// brace matching, and runs them — so what is compared below is the browser
// code's behaviour, executed, not its text.
import fs from 'node:fs';
const src = fs.readFileSync(%s, 'utf8');
function lift(needle) {
    const i = src.indexOf(needle);
    if (i < 0) throw new Error('app.js no longer has: ' + needle);
    let depth = 0;
    for (let k = src.indexOf('{', i); k < src.length; k++) {
        if (src[k] === '{') depth++;
        else if (src[k] === '}' && --depth === 0) return src.slice(i, k + 1);
    }
    throw new Error('unbalanced braces after: ' + needle);
}
const STEPS = %s;
const result = (0, eval)(
    lift('const _FACE_PHASE_BY_TITLE = {') + ';\n' + lift('function faceSignalsForTicker(') +
    ';\n({ table: _FACE_PHASE_BY_TITLE,' +
    '    signals: ' + JSON.stringify(STEPS) + '.map(s => faceSignalsForTicker(s[0], s[1], s[2])) })');
"""


def test_face_signals_match_the_browsers():
    """`face_signals_for_ticker` is a port of app.js's `faceSignalsForTicker`;
    nothing at runtime can catch drift (the device never sees app.js), so the
    original is RUN here, under node, and compared step for step."""
    want = eval_js(_RUN_THE_ORIGINAL % (json.dumps(str(_APP_JS)), json.dumps(STEPS)), "result")
    got = [turnstatus.face_signals_for_ticker(*s) for s in STEPS]
    assert got == want["signals"]
    assert {g["phase"] for g in got} >= {"search", "read", "tool", "verify", None}
    assert [g["verdict"] for g in got if g["verdict"]] == ["pass", "refute"]
    # the table itself, key for key — the steps above reach only some of it
    assert turnstatus.FACE_PHASE_BY_TITLE == want["table"] and len(want["table"]) >= 15


HEADER = "┌─ 99 9961f364  request started  15:16:26 ─────────────"
TOOL = "│  99  🔧  +8.06s  tool call           execute"
LONG_VERDICT = "│  99  🧪  +9.10s  verifier            CONFIRMED: every claim is grounded in the fetched page"


def test_every_step_reaches_the_face_even_when_the_caption_does_not_move():
    """Two identical tool calls are ONE caption and TWO kicks."""
    t = turnstatus.TurnTicker()
    seen = []
    t.on_step = lambda title, icon, detail: seen.append((title, icon, detail))
    t.start()
    t.note_line(HEADER)
    assert t.note_line(TOOL) is True
    assert t.note_line(TOOL) is False                       # the caption did not change…
    assert seen == [("tool call", "🔧", "execute")] * 2     # …the face was told twice
    t.note_line("│  98  🔧  +1.00s  tool call           other corridor")
    t.note_line("│  99  💭  +9.00s  thinking            …")
    t.note_line("│  99  ⚡  +9.00s  llm request         plumbing")
    assert len(seen) == 2, "another corridor, raw thought or plumbing reached the face"
    # the face gets the WHOLE detail (the caption shows 30 characters of it)
    t.note_line(LONG_VERDICT)
    assert seen[-1] == ("verifier", "🧪", "CONFIRMED: every claim is grounded in the fetched page")


def test_a_face_hook_that_raises_does_not_break_the_caption():
    t = turnstatus.TurnTicker()
    t.on_step = lambda *a: 1 / 0
    t.start()
    t.note_line(HEADER)
    assert t.note_line(TOOL) is True and t.desc.startswith("tool call")


_FACE_HTML = (_DIR / "webface" / "face.html").read_text()
_FACE_DRIVER = """
const calls = [];
const throttle = { setFps(n) { calls.push(['throttle.setFps', n]); return n; } };
let face = new Proxy({}, { get: (_t, name) => (...a) => {
    calls.push([name, ...a]);
    return name === 'errorKindFor' ? 'network' : undefined;
} });
"""


def _apply(ops, face_decl=""):
    """The calls `apply` makes, as JSON TEXT — compared as text because in
    Python `1 == True`, and "the page passed 1 where the face wants a
    boolean" is exactly what would otherwise slip through."""
    src = _FACE_DRIVER + face_decl + extract_js_function(_FACE_HTML, "apply")
    return json.dumps(eval_js(src, f"({json.dumps(ops)}.forEach(o => apply(o[0], o[1])), calls)"))


def test_the_page_maps_each_signal_to_the_face_modules_function():
    calls = _apply([["phase", "read"], ["phase", ""], ["tool", None], ["recall", None],
                    ["verdict", "pass"], ["busy", 1], ["gaze", 0], ["error", "connection refused"]])
    assert calls == json.dumps(
        [["setPhase", "read"], ["setPhase", None], ["noteToolCall"], ["noteRecall"],
         ["noteVerdict", "pass"], ["setBackgroundBusy", True], ["setComposerGaze", False],
         ["errorKindFor", "connection refused", ""], ["noteError", "network"]])


def test_the_frame_rate_op_caps_pauses_and_resumes():
    assert _apply([["rate", 10]]) == json.dumps([["throttle.setFps", 10], ["setAnimationPaused", False]])
    assert _apply([["rate", -1]]) == json.dumps([["setAnimationPaused", True]])
    # resuming must UNPAUSE as well as lift the cap
    assert _apply([["rate", 0]]) == json.dumps([["throttle.setFps", 0], ["setAnimationPaused", False]])


def test_a_face_module_without_the_new_hooks_is_tolerated():
    """An older matrix_graph.js: a missing function is skipped, and the
    error op still flinches with the generic kind."""
    old = "face = { noteError(kind) { calls.push(['noteError', kind]); } };\n"
    assert _apply([["tool", None], ["phase", "read"], ["rate", 5], ["error", "x"]], old) == \
        json.dumps([["throttle.setFps", 5], ["noteError", "generic"]])


def test_every_signal_the_python_side_sends_exists_on_the_page():
    """webface.py calls `ghostFace.<op>(…)`; an op the page does not define
    is a silent no-op (`window.ghostFace && ghostFace.x()` throws inside a
    fire-and-forget runJavaScript)."""
    tree = ast.parse((_DIR / "webface.py").read_text())
    # every `self._call("<op>", …)` …
    sent = {n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_call"
            and n.args and isinstance(n.args[0], ast.Constant)}
    # … and every op spelt out inside a JavaScript string the widget sends
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            sent |= set(re.findall(r"ghostFace\.(\w+)\(", node.value))
    # the three change-only states go through `_sync` by NAME
    sync = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sync")
    sent |= {c.value for c in ast.walk(sync) if isinstance(c, ast.Constant)
             and c.value in ("rate", "busy", "gaze")}
    assert len(sent) == 15, sorted(sent)
    api = _FACE_HTML[_FACE_HTML.index("const API = {"):]
    defined = set(re.findall(r"^\s{2}(\w+)\(", api[:api.index("};")], re.M))
    handled = set(re.findall(r"case '(\w+)':", extract_js_function(_FACE_HTML, "apply")))
    assert sent == defined, sorted(sent ^ defined)       # nothing unsendable, nothing undefined
    assert defined == handled, sorted(defined ^ handled)


_THROTTLE = (_DIR / "webface" / "throttle.js").read_text()
_VSYNC = """
// A fake window with a 60 Hz display. An animation-frame callback runs at the
// NEXT refresh after it was requested — so one requested by a timer that
// fires during this tick runs on the following tick, as in a browser. (The
// first version of this model ran it in the same tick, and reported exactly
// the cap; Chromium on the real page draws a little under it.)
let clock = 0, rafQ = [], timers = [], frames = 0, nativeRaf = 0, cancelled = [];
const win = {
    requestAnimationFrame(cb) { nativeRaf++; rafQ.push(cb); return 1000 + rafQ.length; },
    cancelAnimationFrame(id) { cancelled.push(id); rafQ[id - 1001] = null; },
    setTimeout(cb, ms) { timers.push({ cb, at: clock + ms }); return timers.length; },
    clearTimeout(id) { timers[id - 1] = null; },
    performance: { now: () => clock },
};
const api = installThrottle(win);
function tick() {
    clock += 1000 / 60;
    const run = rafQ; rafQ = [];
    for (let i = 0; i < timers.length; i++) {
        const t = timers[i];
        if (t && t.at <= clock) { timers[i] = null; t.cb(); }
    }
    for (const cb of run) if (cb) cb(clock);
}
// The face's loop, as matrix_graph.js writes it: re-arm first, then draw.
let handle = null;
function animate() { handle = win.requestAnimationFrame(animate); frames++; }
function second() { const f0 = frames, n0 = nativeRaf; for (let i = 0; i < 60; i++) tick();
                    return [frames - f0, nativeRaf - n0]; }
"""


def test_the_throttle_caps_frames_and_lets_the_renderer_sleep():
    full, cap10, cap5, full_again = eval_js(_THROTTLE + _VSYNC, """(() => {
        animate();
        const a = second();
        api.setFps(10); second();          // settle
        const b = second();
        api.setFps(5); second();
        const c = second();
        api.setFps(0); second();
        const d = second();
        return [a, b, c, d];
    })()""")
    assert full == [60, 60]
    # A cap is a CEILING: the timer is followed by a wait for the next refresh.
    # These are the exact counts of this deterministic model (Chromium on the
    # real page measured 9.2 and 4.8) — pinned exactly, so a change to the
    # pacing arithmetic shows up as a different number.
    assert cap10[0] == 8, f"cap 10 drew {cap10[0]} frames in a second"
    assert cap5[0] == 4, f"cap 5 drew {cap5[0]} frames in a second"
    # a capped frame waits on a TIMER: the renderer is not woken 60 times a
    # second to decline to draw (one request may straddle the second)
    assert cap10[1] <= cap10[0] + 1 and cap5[1] <= cap5[0] + 1, (cap10, cap5)
    assert full_again == [60, 60], "lifting the cap did not restore the full rate"


def test_the_cap_counts_from_the_last_frame_not_from_the_request():
    """A loop that asks for its next frame AFTER its work (40 ms here) must
    still get its frames 100 ms apart at a cap of 10, not 140."""
    frames = eval_js(_THROTTLE + _VSYNC, """(() => {
        function late() { frames++; clock += 40; handle = win.requestAnimationFrame(late); }
        api.setFps(10); late(); second();
        return second()[0];
    })()""")
    assert frames == 12, frames


def test_lifting_the_cap_takes_effect_at_once():
    """A frame already waiting on the old cap's timer (200 ms at cap 5) is
    rescheduled — "the operator touched a key" must not wait out a slow frame."""
    ticks = eval_js(_THROTTLE + _VSYNC, """(() => {
        api.setFps(5); animate(); second();
        const f0 = frames; api.setFps(0);
        let n = 0; while (frames === f0 && n < 60) { tick(); n++; }
        return n;
    })()""")
    assert ticks == 1, f"{ticks} refreshes passed before the next frame"


@pytest.mark.parametrize("cap", [0, 5])
def test_the_faces_own_pause_still_stops_the_loop(cap):
    """matrix_graph pauses by cancelling the id IT holds — which is the
    throttle's id. If the wrapped cancel did not resolve it, the loop would
    keep running through every pause (including the tab-hidden one)."""
    after, pending, native = eval_js(_THROTTLE + _VSYNC, f"""(() => {{
        api.setFps({cap}); animate(); second();
        const before = cancelled.length;
        win.cancelAnimationFrame(handle);
        const f0 = frames; second(); second();
        return [frames - f0, api.pending(), cancelled.slice(before)];
    }})()""")
    assert after == 0 and pending == 0
    if cap == 0:
        # the browser's OWN id for the pending frame is cancelled — not ours
        assert len(native) == 1 and native[0] > 1000, native


def test_the_throttle_loads_before_the_face_module():
    assert _FACE_HTML.index('<script src="./throttle.js">') < _FACE_HTML.index("import('./matrix_graph.js')")


# ════════════════════════════════════════════════════════════════════════════
# the launcher and the deploy
# ════════════════════════════════════════════════════════════════════════════

def _launch(tmp_path, codes, run_seconds=None, log_setup=None):
    """Run the REAL launch_ghost.sh with `python3` replaced by a stub that
    exits with the next code in `codes` (and `sleep`/`xset` by no-ops).

    `run_seconds[i]` is how long run i appears to last: `date +%s` is stubbed
    to advance by that much while the stub client "runs". The stub also
    writes to stderr and records the environment it was started with.
    """
    bins = tmp_path / "bin"
    bins.mkdir()
    (tmp_path / "codes").write_text(" ".join(map(str, codes)))
    (tmp_path / "secs").write_text(" ".join(map(str, run_seconds or [])))
    (tmp_path / "n").write_text("0")
    (tmp_path / "t").write_text("1000")
    stubs = {
        "python3": f"""#!/bin/bash
n=$(cat "{tmp_path}/n"); echo $((n + 1)) > "{tmp_path}/n"
echo "run $n unbuffered=$PYTHONUNBUFFERED platform=$QT_QPA_PLATFORM args=$*" >> "{tmp_path}/starts"
echo "TRACEBACK-ON-STDERR run $n" >&2
set -- $(cat "{tmp_path}/secs"); shift $n
t=$(cat "{tmp_path}/t"); echo $((t + ${{1:-0}})) > "{tmp_path}/t"
set -- $(cat "{tmp_path}/codes"); shift $n
[ -z "$1" ] && exit 0
exit "$1"
""",
        "sleep": f"#!/bin/bash\necho $1 >> '{tmp_path}/sleeps'\nexit 0\n",
        "xset": "#!/bin/bash\nexit 0\n",
        "date": f"""#!/bin/bash
if [ "$1" = "+%s" ]; then cat "{tmp_path}/t"; else echo "2026-10-01 00:00:00"; fi
""",
    }
    for name, body in stubs.items():
        p = bins / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "ui.log"
    if log_setup:
        log_setup(log)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PATH=f"{bins}:{os.environ['PATH']}", GHOST_UI_LOG=str(log), HOME=str(home))
    env.pop("PYTHONUNBUFFERED", None)
    subprocess.run(["bash", str(_DIR / "launch_ghost.sh")], env=env, timeout=60, check=False)
    return int((tmp_path / "n").read_text()), log


@pytest.mark.parametrize("codes, runs", [
    ([0], 1),                 # /exit — the operator asked to leave
    ([143], 1),               # SIGTERM: deploy.sh, a deliberate pkill
    ([130], 1),               # SIGINT
    ([1, 0], 2),              # a Python traceback → back up
    ([139, 134, 0], 3),       # a segfault, an abort → back up
    ([137, 0], 2),            # SIGKILL: the OOM killer → back up
    ([129, 0], 2),            # SIGHUP → back up
])
def test_the_launcher_restarts_a_crash_and_only_a_crash(tmp_path, codes, runs):
    n, log = _launch(tmp_path, codes)
    assert n == runs, log.read_text()
    assert log.read_text().count("CRASHED") == runs - 1


def test_the_launcher_gives_up_on_a_client_that_cannot_start(tmp_path):
    n, log = _launch(tmp_path, [1] * 20)
    assert n == 5 and "giving up" in log.read_text(), log.read_text()
    # the wait doubles: five tries span half a minute, not ten seconds — a
    # display that was not up yet at boot gets a real chance
    assert (tmp_path / "sleeps").read_text().split() == ["2", "4", "8", "16"]


def test_crashes_far_apart_do_not_add_up_to_giving_up(tmp_path):
    """Five crashes IN A ROW mean it cannot start; a crash a day does not."""
    n, log = _launch(tmp_path, [1] * 8 + [0], run_seconds=[100] * 9)
    assert n == 9 and "giving up" not in log.read_text(), log.read_text()


def test_a_slow_crash_resets_the_count_to_zero_not_below(tmp_path):
    """One long run, then a client that cannot start: it still gets exactly
    five tries (the long run's own crash is the first of them)."""
    n, log = _launch(tmp_path, [1] * 20, run_seconds=[100] + [0] * 19)
    assert n == 5 and "giving up" in log.read_text(), log.read_text()
    n2, _ = _launch(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path,
                    [1] * 20, run_seconds=[0, 0, 100] + [0] * 17)
    assert n2 == 7        # two fast crashes, the slow one resets, then five more


def test_every_start_has_a_log_with_the_clients_stderr_in_it(tmp_path):
    """The autostart entry ran the script with its output going nowhere: the
    client's [tls]/[face] diagnostics, every voice error and every TRACEBACK
    existed only after a deploy had restarted it by hand."""
    def previous(log):
        log.write_text("the previous run\n")
    _, log = _launch(tmp_path, [1, 0], log_setup=previous)
    text = log.read_text()
    assert "[launch]" in text and "the previous run" not in text
    assert "TRACEBACK-ON-STDERR run 0" in text, "stderr is not in the log — a crash leaves no traceback"
    assert (tmp_path / "ui.log.1").read_text() == "the previous run\n"


def test_the_client_is_started_unbuffered_on_xcb(tmp_path):
    """Buffered, the last lines before a crash never reach the log."""
    _launch(tmp_path, [0])
    assert (tmp_path / "starts").read_text().strip() == \
        "run 0 unbuffered=1 platform=xcb args=/home/vasilis/bin/client.py"


def test_a_log_that_cannot_be_written_falls_back_to_home(tmp_path):
    """/tmp is sticky: a log left by another user can be neither moved nor
    appended to — and the launcher used to carry on with no log at all."""
    def lock(log):
        log.mkdir()                   # a directory where the file should be: unwritable as a log
    n, log = _launch(tmp_path, [1, 0], log_setup=lock)
    fallback = tmp_path / "home" / "ghost_ui.log"
    assert n == 2 and fallback.exists(), "no fallback log was written"
    assert "CRASHED" in fallback.read_text() and "TRACEBACK-ON-STDERR" in fallback.read_text()


def _local_imports(path, seen):
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            dep = _DIR / f"{name.split('.')[0]}.py"
            if dep.exists() and dep.name not in seen:
                seen.add(dep.name)
                _local_imports(dep, seen)
    return seen


def test_deploy_ships_every_module_the_client_imports():
    """A module the client imports and deploy.sh does not copy is an
    ImportError on the device and a client that will not start. The list is
    derived from the imports, so adding a module cannot be forgotten."""
    deploy = (_DIR / "deploy.sh").read_text()
    shipped = set(re.search(r'^MODULES="([^"]+)"', deploy, re.M).group(1).split())
    needed = _local_imports(_DIR / "client.py", {"client.py"})
    needed |= _local_imports(_DIR / "device_probe.py", {"device_probe.py"})
    assert len(needed) >= 10
    assert needed <= shipped, f"not deployed: {sorted(needed - shipped)}"
    assert all((_DIR / m).exists() for m in shipped), "deploy.sh lists a file that does not exist"


def _deploy(tmp_path, probe_rc=0, compile_rc=0, pids="4242 4242", crash_line="", old_backups=0):
    """Run the REAL deploy.sh end to end against a fake device.

    `ssh` is replaced by a stub that runs the remote command LOCALLY, with
    HOME pointing at a sandbox — so the script's own control flow is what is
    executed, on both sides of the connection: the `&&` chain around the
    probe, the install here-doc, the restart here-doc. (An earlier version
    only recorded the ssh calls and decided their exit code itself; it could
    not see `|| true` after the probe, because that runs on the far side.)
    Two path constants are redirected into the sandbox; nothing else in the
    script is touched.
    """
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    (home / "gui_env" / "bin").mkdir(parents=True)
    (home / "gui_env" / "bin" / "activate").write_text("")
    (home / "bin" / "client.py").write_text("OLD BUILD\n")           # what is live
    (home / "bin" / "launch_ghost.sh").write_text("# old launcher\n")
    for i in range(old_backups):
        b = home / "bin" / f"client.py.bak-2026010{i}-000000"
        b.write_text(f"backup {i}\n")
        os.utime(b, (1_700_000_000 + i, 1_700_000_000 + i))
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "launcher_inode").write_text(str((home / "bin" / "launch_ghost.sh").stat().st_ino))
    log = tmp_path / "ghost_ui.log"

    tree = tmp_path / "repo" / "interface" / "externals" / "clockwork_ghost"
    (tree / "webface").mkdir(parents=True)
    (tmp_path / "repo" / "interface" / "static").mkdir(parents=True)
    (tmp_path / "repo" / "interface" / "static" / "matrix_graph.js").write_text("// canonical\n")
    deploy = (_DIR / "deploy.sh").read_text()
    assert deploy.count("STAGE=/home/vasilis/bin/.stage") == 1
    (tree / "deploy.sh").write_text(
        deploy.replace("STAGE=/home/vasilis/bin/.stage", f"STAGE={home}/bin/.stage")
        .replace("/tmp/ghost_ui.log", str(log)))
    modules = re.search(r'^MODULES="([^"]+)"', deploy, re.M).group(1).split()
    for mod in modules + ["launch_ghost.sh"]:
        (tree / mod).write_text(f"NEW BUILD {mod}\n")
    (tree / "webface" / "face.html").write_text("face\n")

    bins = tmp_path / "bin"
    bins.mkdir()
    stubs = {
        # the "device": run the command here, with the sandbox as its HOME
        "ssh": f"""#!/bin/bash
shift
printf '%s\\n' "$*" >> "{rec}/ssh"
HOME="{home}" bash -c "$*"
""",
        "scp": f"""#!/bin/bash
args=(); for a in "$@"; do case "$a" in -q|-r|-qr) ;; *) args+=("$a") ;; esac; done
dest="${{args[${{#args[@]}}-1]}}"; unset "args[${{#args[@]}}-1]"
printf '%s\\n' "$dest" >> "{rec}/scp"
cp -R "${{args[@]}}" "${{dest#*:}}"
""",
        "python3": f"""#!/bin/bash
printf '%s\\n' "$*" >> "{rec}/python3"
case "$*" in *py_compile*) exit {compile_rc} ;; *device_probe.py*) exit {probe_rc} ;; esac
exit 0
""",
        "timeout": '#!/bin/bash\nshift\nexec "$@"\n',
        "sleep": "#!/bin/bash\nexit 0\n",
        "pkill": f"""#!/bin/bash
printf '%s\\n' "$*" >> "{rec}/pkill"
exit 0
""",
        "setsid": f"""#!/bin/bash
printf '%s\\n' "$*" >> "{rec}/setsid"
printf '%s' "{crash_line}" >> "{log}"
exit 0
""",
        "pgrep": f"""#!/bin/bash
if [ "$1" = "-fc" ]; then echo 1; exit 0; fi
n=$(cat "{rec}/pg" 2>/dev/null || echo 0); echo $((n + 1)) > "{rec}/pg"
set -- {pids}; shift $n; echo "$1"
""",
    }
    for name, body in stubs.items():
        (bins / name).write_text(body)
        (bins / name).chmod(0o755)
    env = dict(os.environ, PATH=f"{bins}:{os.environ['PATH']}")
    proc = subprocess.run(["bash", str(tree / "deploy.sh"), "testhost"], env=env, timeout=120,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)

    def read(name):
        return (rec / name).read_text() if (rec / name).exists() else ""
    return proc, home, read


def test_a_deploy_stages_probes_installs_and_restarts(tmp_path):
    proc, home, read = _deploy(tmp_path, old_backups=5)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    live = home / "bin"
    # the new build is live, compiled and probed IN STAGING first
    assert (live / "client.py").read_text() == "NEW BUILD client.py\n"
    assert (live / "device_probe.py").exists() and (live / "webface" / "face.html").exists()
    assert (live / "webface" / "matrix_graph.js").read_text() == "// canonical\n"     # the face re-synced
    assert not (live / ".stage").exists()
    assert all(".stage/" in line for line in read("scp").splitlines()), "copied over the LIVE build"
    runs = read("python3").splitlines()
    assert runs[0].startswith("-m py_compile client.py") and runs[1] == "device_probe.py"
    # The launcher was swapped by RENAME (a new inode) and is executable. The
    # old one is a RUNNING bash script; written over in place, the running
    # copy carries on at its old offset in the new text.
    assert (live / "launch_ghost.sh").read_text() == "NEW BUILD launch_ghost.sh\n"
    assert os.access(live / "launch_ghost.sh", os.X_OK) and not (live / ".launch_ghost.sh.new").exists()
    assert str((live / "launch_ghost.sh").stat().st_ino) != read("launcher_inode"), \
        "the running launcher's file was overwritten in place"
    # the old build is backed up, and only the three newest backups are kept
    backups = sorted(p.name for p in live.glob("client.py.bak-*"))
    assert len(backups) == 3, backups
    assert any((live / b).read_text() == "OLD BUILD\n" for b in backups)
    assert "client.py.bak-20260100-000000" not in backups          # the oldest went
    # the launcher is stopped BEFORE the client (it restarts a crashed one)
    kills = read("pkill").splitlines()
    assert kills[0].endswith("launch_ghost.sh") and "client" in kills[1], kills
    assert "launch_ghost.sh" in read("setsid") and "WAYLAND_DISPLAY=wayland-0" in read("setsid")
    assert "client running (pid 4242, stable)" in proc.stdout


def test_the_relaunch_is_detached_so_ssh_can_return():
    """The one thing the fake device cannot show: a launcher left attached to
    the ssh session's stdio keeps the session — and the deploy — open."""
    deploy = (_DIR / "deploy.sh").read_text()
    relaunch = next(ln for ln in deploy.splitlines() if "launch_ghost.sh >" in ln)
    assert relaunch.rstrip().endswith("> /dev/null 2>&1 < /dev/null &"), relaunch


@pytest.mark.parametrize("kw", [{"probe_rc": 1}, {"probe_rc": 124}, {"compile_rc": 1}])
def test_a_build_that_fails_its_probe_is_never_installed(tmp_path, kw):
    """The point of staging: a failed compile or probe (or one that timed out,
    124) leaves ~/bin and the running client exactly as they were. `|| true`
    after the probe, or `;` in place of `&&`, would install it anyway — and
    both run on the far side of the ssh, where only executing them shows."""
    proc, home, read = _deploy(tmp_path, **kw)
    assert proc.returncode != 0
    assert (home / "bin" / "client.py").read_text() == "OLD BUILD\n", "a build that failed was installed"
    assert not list((home / "bin").glob("client.py.bak-*"))
    assert read("pkill") == "" and read("setsid") == "", "the running client was touched"
    if "compile_rc" in kw:
        assert "device_probe.py" not in read("python3").splitlines(), "probed a build that does not compile"


def test_the_probe_is_bounded_by_a_timeout(tmp_path):
    """A probe that hangs must fail the deploy, not hang it."""
    deploy = (_DIR / "deploy.sh").read_text()
    code = "\n".join(ln for ln in deploy.splitlines() if not ln.lstrip().startswith("#"))
    assert re.search(r"timeout \d+ python3 device_probe\.py", code)


@pytest.mark.parametrize("kw", [
    {"pids": "4242 4301"},                                              # a new pid 6 s later: it restarted
    {"crash_line": "[launch] client CRASHED rc=1 after 3s (crash 1 of 5)\n"},
])
def test_a_client_that_crashes_on_start_fails_the_deploy(tmp_path, kw):
    """"One client is alive" is not "the client started": the launcher
    restarts a crash, so a build that dies in three seconds was alive again
    at the six-second look, and the deploy said OK over a crash loop."""
    proc, _home, _read = _deploy(tmp_path, **kw)
    assert proc.returncode != 0 and "CRASHING ON START" in proc.stdout, proc.stdout


def test_a_face_that_failed_to_start_fails_the_deploy(tmp_path):
    proc, _home, _read = _deploy(tmp_path, crash_line="[face] ERROR Error creating WebGL context.\n")
    assert proc.returncode != 0 and "THE FACE FAILED TO START" in proc.stdout, proc.stdout


# ════════════════════════════════════════════════════════════════════════════
# client.py — enumerations of a class of mistake (it cannot be imported here)
# ════════════════════════════════════════════════════════════════════════════

_CLIENT_SRC = (_DIR / "client.py").read_text()
_CLIENT_TREE = ast.parse(_CLIENT_SRC)


def _enclosing_functions(tree, pred):
    """Names of the functions that directly contain a node matching `pred`."""
    out = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if node is not fn and pred(node):
                    out.append(fn.name)
    return out


def _callers(tree, attr):
    def calls(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr)
    return sorted(_enclosing_functions(tree, calls))


def test_there_is_exactly_one_way_to_send_a_message():
    """Two streams used to share one reply bubble. Every message now goes
    through `_submit`, which refuses while a turn runs — AT the moment of
    commitment, because the camera dialog is modal and a turn can start
    underneath it. A new caller of anything below `_submit` would bypass it."""
    assert _callers(_CLIENT_TREE, "send_chat_request") == ["_start_turn"]
    assert _callers(_CLIENT_TREE, "_start_turn") == ["_submit"]
    assert _callers(_CLIENT_TREE, "_submit") == ["handle_input", "take_picture"]
    # …and nothing else puts a user message into the conversation
    def appends_user(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
                and any(isinstance(c, ast.Constant) and c.value == "user" for c in ast.walk(n)))
    assert _enclosing_functions(_CLIENT_TREE, appends_user) == ["_submit"]
    fn = next(n for n in ast.walk(_CLIENT_TREE) if isinstance(n, ast.FunctionDef) and n.name == "_submit")
    first = fn.body[1] if isinstance(fn.body[0], ast.Expr) else fn.body[0]
    assert isinstance(first, ast.If) and "_busy" in ast.unparse(first.test), \
        "_submit must refuse a second turn before it does anything else"


def _send(lines):
    """The REAL `send_chat_request`, run with Qt and the network replaced by
    doubles (the harness lives with the §4KP tests, which pin the same loop)."""
    from tests.test_4kp_think_split_and_clients import _clockwork_send
    return _clockwork_send(lines)


def _send_with(lines, **attrs):
    from tests.test_4kp_think_split_and_clients import _clockwork_send
    return _clockwork_send(lines, **attrs)


def test_a_frame_that_carries_no_text_does_not_abort_the_reply():
    """The agent asks its model server for usage, and the usage frame's
    `choices` is EMPTY. `data["choices"][0]` raised IndexError out of the
    stream: every such reply ended in "fault → IndexError" and its last
    sentence was never spoken."""
    shown, spoken, events = _send([
        'data: {"choices":[{"delta":{"content":"It is four. "}}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}',
        "data: 12",
        'data: {"choices":[null]}',
        ": keepalive",
        "data: [DONE]"])
    assert shown == "It is four. " and spoken == ["It is four."]
    assert [e for e in events if e[0] == "error"] == []


def test_a_reply_is_not_spoken_into_an_open_microphone():
    """Barge-in silences what was queued when recording starts; a reply still
    streaming must not then start talking again — it would be transcribed
    and sent back as the operator's own words."""
    lines = ['data: {"choices":[{"delta":{"content":"Still talking. "}}]}', "data: [DONE]"]
    shown, spoken, _ = _send_with(lines, is_recording=True)
    assert shown == "Still talking. " and spoken == []
    assert _send_with(lines, is_recording=False)[1] == ["Still talking."]
    assert _send_with(lines, tts_enabled=False)[1] == []


def test_what_a_turn_leaves_in_the_conversation():
    """The reply the operator read — once; and nothing at all for a turn that
    produced no text (an empty assistant message is sent back to the model
    on every later turn)."""
    from tests.test_4kp_think_split_and_clients import _clockwork_send
    _send(['data: {"choices":[{"delta":{"content":"Four."}}]}', "data: [DONE]"])
    assert _clockwork_send.window.conversation_history == [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "Four."}]
    _send(['data: {"error": "the model is gone"}', "data: [DONE]"])
    assert _clockwork_send.window.conversation_history == [{"role": "user", "content": "q"}]
    # a reply cut off before [DONE] is still what was read: kept, once
    _send(['data: {"choices":[{"delta":{"content":"Half an ans"}}]}'])
    assert _clockwork_send.window.conversation_history[1:] == [
        {"role": "assistant", "content": "Half an ans"}]


def test_nothing_after_done_is_shown():
    """The loop reads on past [DONE] (it must not `break`), but what follows
    is not part of the reply."""
    shown, spoken, _ = _send(['data: {"choices":[{"delta":{"content":"Done. "}}]}', "data: [DONE]",
                              'data: {"choices":[{"delta":{"content":"late"}}]}'])
    assert shown == "Done. " and spoken == ["Done."]


def test_there_is_exactly_one_way_to_power_the_device_off():
    def runs_power(n):
        if not isinstance(n, ast.Call):
            return False
        words = {c.value for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        return bool(words & {"shutdown", "reboot", "poweroff", "halt"})
    callers = set(_enclosing_functions(_CLIENT_TREE, runs_power))
    assert callers == {"_run_command"}, callers
    fn = next(n for n in ast.walk(_CLIENT_TREE) if isinstance(n, ast.FunctionDef) and n.name == "_run_command")
    power_calls = [n for n in ast.walk(fn) if runs_power(n)]
    assert power_calls and all(isinstance(n.func, ast.Name) and n.func.id == "_power" for n in power_calls), \
        "a power command bypasses _power() — the device probe would really shut the device down"


def test_the_stream_is_read_to_its_end():
    """A `break` out of the SSE loop leaves httpx's generator chain to the
    garbage collector, and qasync installs no async-generator finalizer:
    "async generator ignored GeneratorExit" on every turn (found by
    device_probe.py)."""
    fn = next(n for n in ast.walk(_CLIENT_TREE)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "send_chat_request")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.AsyncFor))

    def breaks(node):                       # a `break` that belongs to THIS loop
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Break):
                return True
            if not isinstance(child, (ast.For, ast.AsyncFor, ast.While,
                                      ast.FunctionDef, ast.AsyncFunctionDef)) and breaks(child):
                return True
        return False
    assert not breaks(loop)


def _main_window():
    return next(n for n in ast.walk(_CLIENT_TREE) if isinstance(n, ast.ClassDef) and n.name == "MainWindow")


def test_no_shortcut_repeats_while_its_key_is_held():
    """Every shortcut here is a TOGGLE, and a QShortcut auto-repeats while
    its key is held. Holding Esc — the natural way to use a chip that says
    "PTT" — started and stopped the recording some twenty-five times a
    second, each stop uploading a file the next start was already rewriting:
    "STT Error: Too much data for declared Content-Length", eleven in a row,
    seen on the device. Enumerated, so a shortcut added later cannot be the
    one that was forgotten."""
    cls = _main_window()
    made = {t.attr for n in ast.walk(cls) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "QShortcut"
            for t in n.targets if isinstance(t, ast.Attribute)}
    quiet = set()
    for n in ast.walk(cls):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "setAutoRepeat"
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value is False):
            if isinstance(n.func.value, ast.Attribute):            # self.x_shortcut.setAutoRepeat(False)
                quiet.add(n.func.value.attr)
    for loop in (n for n in ast.walk(cls) if isinstance(n, ast.For)):
        calls = [c for c in ast.walk(loop) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                 and c.func.attr == "setAutoRepeat" and c.args
                 and isinstance(c.args[0], ast.Constant) and c.args[0].value is False]
        if calls and isinstance(loop.iter, ast.Tuple):             # for _sc in (self.a, self.b): …
            quiet |= {e.attr for e in loop.iter.elts if isinstance(e, ast.Attribute)}
    assert len(made) >= 7 and made == quiet, sorted(made ^ quiet)


def test_a_recording_is_uploaded_from_memory_never_from_the_open_file():
    """httpx declares Content-Length from the file's size when the request is
    built and then streams the file: if it changes in between, the request
    dies. The clip is read whole first, and the file is gone before the
    network is touched."""
    fn = next(n for n in ast.walk(_CLIENT_TREE)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_stt_audio")
    net = next(n for n in ast.walk(fn) if isinstance(n, ast.AsyncWith))
    opens = [c for c in ast.walk(net) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "open"]
    assert opens == [], "a file is opened inside the HTTP block"
    reads = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "read"]
    assert reads and max(reads) < net.lineno, "the clip is read after the request began"
    # …and every recording is written to a path of its own
    start = next(n for n in ast.walk(_CLIENT_TREE) if isinstance(n, ast.FunctionDef) and n.name == "start_recording")
    path = next(n for n in ast.walk(start) if isinstance(n, ast.Assign)
                and any(getattr(t, "attr", "") == "_rec_path" for t in n.targets))
    assert isinstance(path.value, ast.JoinedStr) and "_rec_seq" in ast.unparse(path.value), \
        "recordings share one file again"


def test_the_transcript_drops_an_empty_bubble_by_its_own_row():
    """`bubble.parentWidget().layout()` is the transcript's whole column, not
    the bubble's row; unparenting it segfaulted the client on the next
    message — i.e. whenever a turn failed before its first token."""
    tree = ast.parse((_DIR / "chatlog.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "end_agent")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "parentWidget" not in attrs and "removeItem" in attrs
