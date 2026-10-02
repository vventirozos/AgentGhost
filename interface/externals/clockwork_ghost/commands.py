"""Slash commands for the handheld — what counts as one, and which ask twice.

Qt-free. The old handling was ``text.startswith('/shutdown')`` inline in the
send path, which had two properties nobody chose: ``/clearly this is wrong``
wiped the conversation, and ``/shutdown`` powered the device off the instant
Enter was pressed, with no way back. Both are decided here instead, where the
rules can be executed in tests.
"""

from __future__ import annotations

import difflib
import re
import time
from collections import namedtuple

Command = namedtuple("Command", "name arg")

# (name, argument hint, what it does) — also the /help text, so a command
# cannot exist without being listed.
COMMANDS = (
    ("help", "", "this list"),
    ("new", "", "start a fresh conversation"),
    ("clear", "", "same as /new"),
    ("stop", "", "stop the running turn"),
    ("sessions", "", "recent conversations, from any client"),
    ("open", "N", "open conversation N from /sessions"),
    ("good", "", "the last reply was right"),
    ("bad", "[why]", "the last reply was wrong"),
    ("bright", "1-9 | + | -", "screen brightness"),
    ("vol", "0-100 | + | -", "speaker volume"),
    ("tts", "", "spoken replies on / off"),
    ("face", "", "next face form"),
    ("shutdown", "", "power the device off"),
    ("reboot", "", "restart the device"),
    ("exit", "", "quit to the desktop"),
)
NAMES = tuple(c[0] for c in COMMANDS)
# Asked twice: each ends the session and cannot be taken back from inside it.
CONFIRM = ("shutdown", "reboot", "exit")
CONFIRM_WINDOW_S = 10.0

SHORTCUTS = (
    ("Esc", "talk (press again to send)"),
    ("Alt+Esc", "spoken replies on / off"),
    ("Ctrl+Esc", "camera"),
    ("Shift+Esc", "stop the running turn"),
    ("PgUp / PgDn", "scroll (also Shift+↑ / Shift+↓)"),
    ("Ctrl+↑ / Ctrl+↓", "rate the last reply good / bad"),
    ("F11", "face only (any key returns)"),
)

# After the word: nothing, an argument after a space, an argument attached
# with no space (`/vol+`, `/vol50`, `/bright5`), or stray end punctuation
# (`/help?`, `/stop.`). A second slash is none of those, so a path never
# matches.
_CMD_RE = re.compile(r"/([A-Za-z]+)(?:[ \t]+(.*)|([+\-0-9][^/\s]*)|[.?!]*)\Z", re.S)
# Commands that take something after them (their /help hint says what).
TAKES_ARG = frozenset(name for name, hint, _what in COMMANDS if hint)
NEAR = 0.75


def _near(word: str) -> bool:
    return bool(difflib.get_close_matches(word, NAMES, n=1, cutoff=NEAR))


def parse(text: str):
    """A :class:`Command`, or None when ``text`` is an ordinary message.

    The old test was ``text.startswith('/clear')``, so ``/clearly that is
    wrong`` wiped the conversation. The first rewrite made a command "a whole
    first word" — and a reviewer then started a new conversation with ``/new
    idea: use a queue instead``, the same defect one space later. The rule
    now turns on what FOLLOWS the word:

    * a known command, alone or with the argument it takes → that command;
    * a known command that takes NO argument, followed by one stray word
      (``/stop it``) → ``Command("extra", word)``: nothing runs, the text
      stays in the input;
    * a known no-argument command followed by a SENTENCE (``/new idea: …``,
      ``/help me fix the parser``) → None: it is a message, and is sent;
    * an unknown word alone, or a near miss with one argument (``/hlep``,
      ``/brigth 5``) → ``Command("unknown", word)``: answered locally with a
      suggestion, where a typo used to be sent to the agent as a prompt;
    * an unknown word followed by a sentence (``/news about greece``,
      ``/s that was sarcasm``), and anything with a second slash
      (``/etc/hosts``) → None.
    """
    m = _CMD_RE.match((text or "").strip())
    if not m:
        return None
    word = m.group(1).lower()
    arg = (m.group(2) or m.group(3) or "").strip()
    words = len(arg.split())
    if word in NAMES:
        if arg and word not in TAKES_ARG:
            return Command("extra", word) if words == 1 else None
        return Command(word, arg)
    if words > 1:
        return None
    if not arg or _near(word):
        return Command("unknown", word)
    return None


def suggestion(word: str) -> str:
    near = difflib.get_close_matches(word, NAMES, n=1, cutoff=0.6)
    return near[0] if near else ""


class Confirmer:
    """"Type it again to confirm", keyboard-only and with a deadline.

    ``confirm(name)`` is False the first time (and arms), True when the SAME
    command arrives again inside the window. Anything else in between —
    another command, an ordinary message — must call :meth:`disarm`, so
    ``/shutdown``, a sentence, ``/shutdown`` does not power the device off.
    """

    def __init__(self, window_s: float = CONFIRM_WINDOW_S, clock=time.monotonic):
        self.window_s = float(window_s)
        self._clock = clock
        self._armed = None
        self._at = 0.0

    def confirm(self, name: str) -> bool:
        now = self._clock()
        # `0 <=`: a clock that stepped backwards must not confirm.
        if self._armed == name and 0 <= now - self._at <= self.window_s:
            self._armed = None
            return True
        self._armed, self._at = name, now
        return False

    def disarm(self) -> None:
        self._armed = None


def help_html(dim: str, accent: str, mono: str) -> str:
    """The /help bubble."""
    def row(left, right):
        return (f"<tr><td style='padding:1px 14px 1px 0; font-family:{mono};"
                f" color:{accent};'>{left}</td>"
                f"<td style='padding:1px 0; color:{dim};'>{right}</td></tr>")
    cmds = "".join(row(f"/{n}" + (f" {hint}" if hint else ""), what)
                   for n, hint, what in COMMANDS)
    keys = "".join(row(k, what) for k, what in SHORTCUTS)
    return (f"<table>{cmds}</table>"
            f"<div style='margin-top:8px;'><table>{keys}</table></div>")
