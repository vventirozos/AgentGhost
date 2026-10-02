"""What the handheld says aloud, and the small sounds around it. Qt-free.

**The chunker.** Replies arrive as a token stream and are spoken sentence by
sentence, so speech starts before the reply finishes. The old splitter was a
regex over a rolling buffer, and it read EVERYTHING: a fenced code block was
spoken symbol by symbol, a table row as a run of pipes, a URL letter by letter.
:class:`SpeechChunker` keeps the low latency (a sentence is released as soon as
it is complete) and adds the one thing a regex over a buffer cannot know —
where it is in the document: inside a code fence, in a table, inside a link.

Its contract, which the tests fuzz: **however the stream is cut into chunks,
the utterances are the same.** Everything below that looks like caution (a
line starting with a backtick is held until its newline; a sentence is not
released while a ``[link`` is still open) exists to keep that true.

**The chime.** A long turn used to end in silence; with the panel asleep the
operator had no way to know the answer had landed. :func:`chime_wav` builds a
short two-tone WAV in memory (no asset to deploy, no dependency) that the
client pipes to ``aplay``.
"""

from __future__ import annotations

import io
import math
import re
import struct
import wave

_SENTENCE_END_RE = re.compile(r"[.?!]+[\"')\]]*\s")

# ── fences (CommonMark's rules, the two that matter) ────────────────────────
# An opener is three or more backticks or tildes; a BACKTICK opener's info
# string may not contain a backtick — which is what makes a line that starts
# with inline code (```ls -la``` lists files.) prose and not a fence. The first
# version called any line starting with ``` a fence toggle: that line silenced
# the rest of the reply.
_FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


def _fence_open(line: str):
    m = _FENCE_OPEN_RE.match(line)
    if not m:
        return None
    marks, info = m.group(1), m.group(2)
    if marks[0] == "`" and "`" in info:
        return None
    return marks[0], len(marks)


def _fence_close(line: str, fence) -> bool:
    """Only the SAME character, at least as long, alone on its line, closes:
    a ``~~~`` inside a backtick block, or a shorter run inside a longer one,
    is content."""
    ch, n = fence
    return bool(re.match(r"^\s*%s{%d,}\s*$" % (re.escape(ch), n), line))


# ── tables ──────────────────────────────────────────────────────────────────
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")          # | a | b |
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
# A partial line that may still turn out to be a fence or a table row: nothing
# on it is spoken until its newline arrives and says which.
_HOLD_PREFIX_RE = re.compile(r"^\s*[`~]")

# ── links, images, bare URLs ────────────────────────────────────────────────
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# The URL ends BEFORE trailing sentence punctuation, or "see https://x.test/a.
# Next" would swallow the full stop that ends the sentence.
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>]*[^\s<>.,;:!?)\]'\"]", re.I)
_UNFINISHED_TARGET_RE = re.compile(r"\]\([^)]*$")

_LEAD_MARK_RE = re.compile(r"^\s*(?:#{1,6}\s+|>+\s*|[-*+]\s+|\d+[.)]\s+)")
# Emphasis marks hug a word (`**bold**`, `~~gone~~`); a `*` standing alone is
# arithmetic ("2 * 3") and a `#` after a letter is a name ("C#") — both stay.
_EMPHASIS_RE = re.compile(r"(?<!\s)[*~]+|[*~]+(?!\s)")
_EDGE_UNDERSCORE_RE = re.compile(r"(?<!\w)_+|_+(?!\w)")
_WS_RE = re.compile(r"\s+")


def _resolve(text: str) -> str:
    """Images dropped, a link read as its label, a bare URL as "a link"."""
    s = _IMAGE_RE.sub(" ", text)
    s = _LINK_RE.sub(r"\1", s)
    return _URL_RE.sub(" a link", s)


def _link_open(prefix: str) -> bool:
    """Does ``prefix`` end inside a link or image that has not closed yet?"""
    # Scanned, not counted: a stray `]` earlier in the line must not cancel
    # a `[` that opens later (counting them said "balanced" and cut a link
    # in half — found by the chunk-invariance fuzz). A link that has already
    # closed scans to zero and its target is not "unfinished", so complete
    # links need no special handling.
    depth = 0
    for ch in prefix:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
    return depth > 0 or bool(_UNFINISHED_TARGET_RE.search(prefix))


def speakable(text: str) -> str:
    """One sentence of reply markdown as the words to say, or ``""``.

    Images are dropped, a link is read as its label, a bare URL as "a link"
    (nobody wants ``h t t p s colon slash slash``), emphasis and heading marks
    are removed, and ``snake_case`` is read as two words rather than fused
    into one. A string with nothing left to pronounce returns ``""``.
    """
    s = _resolve(str(text or ""))
    s = _LEAD_MARK_RE.sub("", s)
    s = s.replace("`", "")
    s = _EMPHASIS_RE.sub("", s)
    s = _EDGE_UNDERSCORE_RE.sub("", s)
    s = s.replace("_", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s if any(ch.isalnum() for ch in s) else ""


class SpeechChunker:
    """Streamed reply text in, utterances out.

    ``feed(text)`` returns the utterances that became complete; ``flush()``
    returns whatever is left when the reply ends. Code fences and tables are
    never spoken.
    """

    def __init__(self):
        self._line = ""            # the current, still-incomplete line
        self._line_start = True    # does `_line` begin at a line start?
        self._fence = None         # (char, length) while inside a code fence
        self._in_table = False
        self._held = None          # a line with a `|` waiting to learn if it heads a table

    def feed(self, text: str) -> list:
        out = []
        # (A CRLF needs no handling: the "\r" is trailing whitespace on its
        # line, and every line pattern here tolerates that.)
        self._line += str(text or "")
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            out.extend(self._end_line(line))
            self._line_start = True
        out.extend(self._partial())
        return out

    def flush(self) -> list:
        line, self._line = self._line, ""
        out = self._end_line(line) if line else []
        out.extend(self._release_held())
        self._line_start = True
        self._fence = None
        self._in_table = False
        return out

    # ── internals ────────────────────────────────────────────────────────
    def _release_held(self) -> list:
        held, self._held = self._held, None
        return self._sentences(held) if held is not None else []

    def _end_line(self, line: str) -> list:
        """A line is complete (its newline arrived, or the reply ended)."""
        start = self._line_start
        if self._fence is not None:
            if start and _fence_close(line, self._fence):
                self._fence = None
            return []
        if start:
            fence = _fence_open(line)
            if fence is not None:
                out = self._release_held()
                self._fence = fence
                self._in_table = False
                return out
            if _TABLE_RULE_RE.match(line):
                # The line above was the table's header, not prose.
                self._held = None
                self._in_table = True
                return []
            if _TABLE_ROW_RE.match(line) or (self._in_table and "|" in line):
                # A row. Only a RULE line makes the line above a header, so a
                # held line is prose and is said.
                out = self._release_held()
                self._in_table = True
                return out
        out = self._release_held()
        self._in_table = False
        if start and "|" in line and not self._cut_before(line, line.index("|")):
            # "name | score" — prose, or the header of a table written
            # without outer pipes? The NEXT line says which. (Not held when a
            # sentence ends before the first pipe: a stream cut finely enough
            # has already spoken that sentence, and the utterances must not
            # depend on how the stream was cut.)
            self._held = line
            return out
        return out + self._sentences(line)

    @staticmethod
    def _cut_before(text: str, limit: int) -> bool:
        """Is there a releasable sentence end within ``text[:limit]``?"""
        return any(m.end() <= limit and not _link_open(text[:m.end()])
                   for m in _SENTENCE_END_RE.finditer(text[:limit]))

    def _partial(self) -> list:
        """Release finished sentences from the line still being written."""
        if self._fence is not None:
            return []
        if self._line_start and (_HOLD_PREFIX_RE.match(self._line) or "|" in self._line
                                 or self._in_table or self._held is not None):
            return []          # structure is decided when the line is whole
        cut = None
        for m in _SENTENCE_END_RE.finditer(self._line):
            if not _link_open(self._line[:m.end()]):
                cut = m.end()
        if cut is None:
            return []
        done, self._line = self._line[:cut], self._line[cut:]
        self._line_start = False
        return self._sentences(done)

    @staticmethod
    def _sentences(text: str) -> list:
        # Links are resolved BEFORE the split: "[Is Rust faster than C? - SO]
        # (https://…)" used to be cut at the "?" and read out as markup.
        text = _resolve(text)
        out, pos = [], 0
        for m in _SENTENCE_END_RE.finditer(text):
            out.append(text[pos:m.end()])
            pos = m.end()
        if text[pos:].strip():          # a line's unterminated tail
            out.append(text[pos:])
        return [s for s in (speakable(x) for x in out) if s]


# ── sounds ──────────────────────────────────────────────────────────────────

def chime_wav(tones=(880.0, 1318.5), tone_ms: int = 110, rate: int = 16000,
              volume: float = 0.22) -> bytes:
    """A short rising two-tone chime as a mono 16-bit WAV.

    Each tone is faded in and out over 12 ms — an unfaded sine starts and
    stops with a click that is louder than the tone itself on a small speaker.
    """
    volume = max(0.0, min(1.0, float(volume)))
    n = max(1, int(rate * tone_ms / 1000))
    fade = max(1, int(rate * 0.012))
    frames = bytearray()
    for freq in tones:
        for i in range(n):
            env = min(1.0, i / fade, (n - 1 - i) / fade)
            sample = math.sin(2.0 * math.pi * freq * i / rate) * env * volume
            frames += struct.pack("<h", int(sample * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def should_chime(elapsed_s: float, panel_was_off: bool, tts_on: bool,
                 min_elapsed_s: float = 20.0) -> bool:
    """Chime when a reply lands that the operator has stopped watching for.

    Not when replies are spoken (the voice IS the notification), and not after
    a quick turn — a beep on every answer would be turned off within a day.
    """
    if tts_on:
        return False
    return bool(panel_was_off) or float(elapsed_s) >= float(min_elapsed_s)


def clock_label(seconds: float) -> str:
    """``7`` → ``0:07`` — the recording timer and the review countdown."""
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"
