"""§4UD — the mood on the clients (review findings F10, F11, F12, F15, W1, W4).

What reaches the web console and the handheld of the agent's mood, and what
they do when nothing does:

* ``/api/health`` sends the manner's level and end, the hold's end, ONE label
  for the look (``face``) and ``problem`` when the state file was unreadable;
* both clients read the block through one function each (status.js
  ``moodView`` / agentapi.py ``mood_view``) and the two must give the same look
  and the same words for every payload — one table, executed in node and in
  Python;
* three polls in a row without a usable block: the face is told neutral and
  the words say "mood unknown";
* the words are visible text, not a tooltip, and do not need the face;
* the end of a turn and a recorded thumb trigger one extra health read.

JS is EXECUTED: node loads ``interface/static/status.js`` itself and calls its
exports (and runs ``initStatus`` against stubs of the page).
"""
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_HANDHELD = _ROOT / "interface" / "externals" / "clockwork_ghost"

sys.path.insert(0, str(_HANDHELD))
try:
    import agentapi
    import devstatus
finally:
    sys.path.pop(0)

TZ = "Asia/Kolkata"          # +05:30 all year: a zone conversion that also moves the minutes

#: status.js's module-level mood functions, in dependency order. Harnesses that run an EXTRACTED
#: ``initStatus`` (tests/test_interface_face_prefs_and_status_chip.py, …batch3…) prepend these.
STATUS_MOOD_FNS = ("moodLabels", "moodStalePolls", "moodLevel", "moodClock", "moodView", "mannerWords",
                   "moodWords", "faceMoodFrom", "moodTrack", "faceLookFor", "moodNote")


def status_mood_scope(status_js: str) -> str:
    """What an extracted ``initStatus`` needs from the module around it since §4UD."""
    from tests.helpers import extract_js_function
    return ("const MOOD_REFRESH_DELAY_MS = 1500;\n"
            + "\n".join(extract_js_function(status_js, n) for n in STATUS_MOOD_FNS) + "\n")


# ── node: status.js as the browser loads it ─────────────────────────────────

_PRELUDE = r"""
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[1], 'utf8');
const S = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const DATA = JSON.parse(process.argv[2]);
const out = v => process.stdout.write(JSON.stringify(v === undefined ? null : v));
"""


def _node(body: str, data=None, tz: str = TZ):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not on PATH — JS EXECUTION pins are inert here")
    proc = subprocess.run(
        [node, "--input-type=module", "-e", _PRELUDE + body, str(_STATIC / "status.js"), json.dumps(data)],
        capture_output=True, text=True, timeout=40, env=dict(os.environ, TZ=tz))
    if proc.returncode != 0:
        raise RuntimeError(f"node failed: {proc.stderr.strip()[:3000]}")
    return json.loads(proc.stdout or "null")


@pytest.fixture
def local_tz():
    """This process in the same zone node is started in. Restored by hand: ``monkeypatch.undo()`` here
    would also undo the suite's own isolation (GHOST_HOME) for the rest of the test."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = TZ
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


def _py(mood):
    """The handheld's look and words for one block, shaped like the JS answer."""
    view = agentapi.mood_view(mood)
    look = [view["face"], view["strength"]] if view else [None, None]
    return [look, agentapi.mood_words(view)]


_JS_LOOK_AND_WORDS = "out(DATA.map(m => [S.faceMoodFrom(m), S.moodWords(S.moodView(m), false)]));"


# ── W1 / W4: one look, and the words name the look ──────────────────────────

# (payload, look, words) — the expected column keeps "both sides wrong the same way" out
TABLE = [
    # an older agent (no `face`): each client's own rule
    ({"label": "irritable", "intensity": 2, "rest": "calm"}, ["irritable", 2], "irritable, 2 of 3"),
    ({"label": "neutral", "intensity": 0, "rest": "calm"}, ["calm", "rest"], "rests calm"),
    ({"label": "neutral", "intensity": 0, "rest": "neutral"}, ["neutral", 0], "neutral"),
    ({"label": "neutral", "intensity": 0}, ["neutral", 0], "neutral"),
    ({"label": "neutral", "intensity": 0, "rest": "Calm!"}, ["neutral", 0], "neutral"),
    ({"label": "neutral", "intensity": 0, "rest": 7}, ["neutral", 0], "neutral"),
    # the engine's `face` is THE look, whatever this client's rule would say
    ({"label": "neutral", "intensity": 0, "rest": "calm", "face": "down"}, ["down", "rest"], "rests down"),
    ({"label": "neutral", "intensity": 0, "rest": "calm", "face": "neutral"}, ["neutral", 0], "neutral"),
    # r2: a mood has ONE label — a `face` naming another (or neutral beside a level) is a miss, not a look
    ({"label": "irritable", "intensity": 3, "face": "bored"}, [None, None], ""),
    ({"label": "irritable", "intensity": 2, "face": "neutral"}, [None, None], ""),
    ({"label": "content", "intensity": 1, "face": "calm", "rest": "calm"}, [None, None], ""),
    ({"label": "content", "intensity": 1, "face": "content"}, ["content", 1], "content, 1 of 3"),
    # a `face` this client has no look for: its own rule, never a throw
    ({"label": "neutral", "intensity": 0, "rest": "calm", "face": "smug"}, ["calm", "rest"], "rests calm"),
    ({"label": "cheerful", "intensity": 2, "face": 3}, ["cheerful", 2], "cheerful, 2 of 3"),
    ({"label": "cheerful", "intensity": 2, "face": ""}, ["cheerful", 2], "cheerful, 2 of 3"),
    # F10: the manner's level and end
    ({"label": "neutral", "intensity": 0, "manner": "evil", "manner_intensity": 2,
      "manner_until": "2026-11-11T22:30:00Z"}, ["neutral", 0], "neutral · manner evil, 2 of 3 — until 04:00"),
    ({"label": "down", "intensity": 1, "manner": "terse", "manner_intensity": 3,
      "manner_until": "2026-11-12T04:00:00"}, ["down", 1], "down, 1 of 3 · manner terse, 3 of 3 — until 04:00"),
    ({"label": "neutral", "intensity": 0, "manner": "terse"}, ["neutral", 0], "neutral · manner terse"),
    ({"label": "neutral", "intensity": 0, "manner": "terse", "manner_intensity": 9, "manner_until": "soon"},
     ["neutral", 0], "neutral · manner terse"),
    ({"label": "neutral", "intensity": 0, "manner": "<b>x</b>", "manner_intensity": 2}, ["neutral", 0], "neutral"),
    ({"label": "neutral", "intensity": 0, "manner": "", "manner_intensity": 2, "manner_until": "2026-11-11T22:30:00Z"},
     ["neutral", 0], "neutral"),
    # F12: an unreadable state file is said, not shown as a healthy neutral
    ({"label": "neutral", "intensity": 0, "problem": "state.json unreadable: JSONDecodeError"},
     ["neutral", 0], "mood unreadable"),
    # r2: … and BESIDE a mood, a rest or a manner the block does carry, never instead of it
    ({"label": "cheerful", "intensity": 2, "face": "cheerful", "problem": "unreadable: JSONDecodeError"},
     ["cheerful", 2], "cheerful, 2 of 3 · state file was unreadable at start"),
    ({"label": "neutral", "intensity": 0, "face": "calm", "problem": "unreadable: x"},
     ["calm", "rest"], "rests calm · state file was unreadable at start"),
    ({"label": "neutral", "intensity": 0, "manner": "terse", "manner_intensity": 1, "problem": "unreadable: x"},
     ["neutral", 0], "neutral · manner terse, 1 of 3 · state file was unreadable at start"),
    ({"label": "neutral", "intensity": 0, "problem": "  \t\n"}, ["neutral", 0], "neutral"),
    ({"label": "neutral", "intensity": 0, "problem": 7}, ["neutral", 0], "neutral"),
    # malformed: no look, no words (the caller counts a poll without a mood)
    ({"label": "x'); alert(1); //", "intensity": 2}, [None, None], ""),
    ({"label": "smug", "intensity": 1}, [None, None], ""),
    ({"label": "calm", "intensity": 9}, [None, None], ""),
    ({"label": "calm", "intensity": -1}, [None, None], ""),
    ({"label": "calm", "intensity": 1.5}, [None, None], ""),
    ({"label": "calm", "intensity": "2"}, [None, None], ""),
    ({"label": "calm", "intensity": True}, [None, None], ""),
    ({"label": "calm", "intensity": None}, [None, None], ""),
    ({"label": "calm", "intensity": 0}, [None, None], ""),          # a label that contradicts its level
    ({"label": "neutral", "intensity": 2}, [None, None], ""),
    ({"label": 4, "intensity": 2}, [None, None], ""),
    ({"intensity": 2}, [None, None], ""),
    ({}, [None, None], ""),
    (None, [None, None], ""),
    ([], [None, None], ""),
    (["calm", 1], [None, None], ""),
    ("calm", [None, None], ""),
    (3, [None, None], ""),
    (True, [None, None], ""),
]


def test_the_web_face_and_the_handheld_show_the_same_look_and_say_the_same_words(local_tz):
    """W1 + W4. Fails where a client keeps its own rule beside the server's `face` (a neutral mood with
    `rest: calm, face: down` → one face calm, the other down), or where the words are made from the mood's
    label while the look is made from the rest ("neutral" beside a calm look)."""
    payloads = [row[0] for row in TABLE]
    web = _node(_JS_LOOK_AND_WORDS, payloads)
    for (mood, look, words), got_web in zip(TABLE, web):
        assert got_web == [look, words], mood
        assert _py(mood) == [look, words], mood
        got = asyncio.run(agentapi.fetch_mood(_Http({"mood": mood}), "http://a", "k"))
        assert (list(got) if got else [None, None]) == look, mood


def _garbage(rng):
    atoms = [None, True, False, 0, 1, 2, 3, 4, -1, 2.0, 1.5, 1e308, "", " ", "calm", "neutral", "irritable",
             "Calm", "smug", "evil", "terse", "x" * 300, "<i>", "2026-11-11T22:30:00Z", "2026-11-11T22:30:00.123456+02:00",
             "2026-13-40T99:99:99Z", "2026-02-30T03:00:00-11:30", "1999-01-01T00:00:00Z", "2026-11-11T04:00",
             "2026-11-11 04:00:00", "２０２６-11-11T04:00:00Z", [], {}, ["calm"], {"label": "calm"}]
    keys = ["label", "intensity", "rest", "face", "manner", "manner_intensity", "manner_until", "held_until",
            "problem", "cause", "source", "set_at"]
    if rng.random() < 0.08:
        return rng.choice(atoms)
    mood = {}
    for k in keys:
        if rng.random() < 0.7:
            mood[k] = rng.choice(atoms)
    if rng.random() < 0.6:                       # often a valid core, so the optional fields get exercised
        n = rng.choice([0, 1, 2, 3])
        mood["label"], mood["intensity"] = ("neutral" if n == 0 else rng.choice(agentapi.MOOD_LABELS[1:])), n
    return mood


def test_garbage_mood_blocks_never_throw_and_both_clients_agree(local_tz):
    """F11: 600 blocks of wrong types, unknown labels, impossible dates. Neither parser throws (node exits
    non-zero on a throw, Python raises) and they give the same look and words for every one."""
    rng = random.Random(4)
    payloads = [_garbage(rng) for _ in range(600)]
    web = _node(_JS_LOOK_AND_WORDS + "", payloads)
    mine = [_py(m) for m in payloads]
    diff = [(m, w, p) for m, w, p in zip(payloads, web, mine) if w != p]
    assert not diff, diff[:3]
    assert sum(1 for w in web if w[0] == [None, None]) > 100 and sum(1 for w in web if w[0][0]) > 100
    # every look either side returns is one the face can be told
    for look, _words in web:
        assert look == [None, None] or (look[0] in agentapi.MOOD_LABELS and look[1] in (0, 1, 2, 3, "rest"))


def test_numbers_json_cannot_carry_are_refused_by_each_parser():
    for bad in (float("nan"), float("inf"), -float("inf")):
        assert agentapi.mood_view({"label": "calm", "intensity": bad}) is None
        assert agentapi.mood_view({"label": "neutral", "intensity": 0, "manner": "evil",
                                   "manner_intensity": bad})["manner_level"] == 0
    assert agentapi.mood_view(json.loads('{"label": "calm", "intensity": NaN}')) is None
    assert _node("out([NaN, Infinity, -Infinity, 1e400, 2n, new Date(), () => 1, Symbol.iterator].map(x => "
                 "[S.moodView({label: 'calm', intensity: x}), S.moodClock(x), "
                 "S.moodView({label: 'neutral', intensity: 0, manner: 'evil', manner_intensity: x}).mannerLevel]));"
                 ) == [[None, "", 0]] * 8


def test_the_extraction_list_names_every_mood_function_the_module_exports():
    """A helper added to status.js must be added to STATUS_MOOD_FNS, or the chip harnesses that run an
    extracted ``initStatus`` die on a ReferenceError."""
    assert sorted(_node("out(Object.keys(S));")) == sorted(STATUS_MOOD_FNS + ("initStatus",))


def test_both_clients_know_the_same_labels_and_they_are_the_engines():
    from ghost_agent.selfhood import mood as M
    web = _node("out([S.moodLabels(), S.moodStalePolls()]);")
    assert sorted(web[0]) == sorted(agentapi.MOOD_LABELS) == sorted(set(M.LABELS) | {M.NEUTRAL})
    assert web[1] == agentapi.MOOD_STALE_POLLS == 3


@pytest.mark.parametrize("stamp,want", [
    ("2026-11-11T22:30:00Z", "04:00"),                     # the engine's stamps: UTC with a Z
    ("2026-11-11T22:30:00.123456Z", "04:00"),
    ("2026-11-11T22:30Z", "04:00"),
    ("2026-11-12T00:00:00+01:30", "04:00"),
    ("2026-11-11T12:30:00-10:00", "04:00"),
    ("2026-11-12T04:00:00", "04:00"),                      # no zone: already wall-clock
    ("2026-11-12T23:59", "23:59"),
    ("", ""), ("04:00", ""), ("2026-11-12", ""), ("2026-11-12 04:00:00Z", ""), ("2026-11-12T24:00:00Z", ""),
    ("2026-13-01T04:00:00Z", ""), ("2026-11-32T04:00:00Z", ""), ("2026-11-12T04:60:00Z", ""),
    ("2026-11-12T04:00:00+24:00", ""), ("1999-11-12T04:00:00Z", ""), ("2026-11-12T04:00:00Z\n", ""),
    (None, ""), (4, ""), (["2026-11-12T04:00:00Z"], ""),
])
def test_an_end_stamp_is_read_as_a_clock_time_here(stamp, want, local_tz):
    """Fails where the stamp is printed raw ("until 2026-11-11T22:30:00Z"), in UTC ("22:30" at +05:30), or
    where one client reads a stamp the other refuses."""
    assert agentapi.mood_clock(stamp) == want
    assert _node("out(S.moodClock(DATA));", stamp) == want


def test_the_tooltip_adds_the_cause_the_hold_and_the_manners_end():
    notes = _node("out(DATA.map(([m, unknown]) => S.moodNote({mood: m}, unknown)));", [
        [{"label": "calm", "intensity": 1, "cause": "the owner said so (set_mood)",
          "held_until": "2026-11-11T22:30:00Z", "manner": "evil", "manner_intensity": 2,
          "manner_until": "2026-11-11T22:30:00Z"}, False],
        [{"label": "calm", "intensity": 1, "held_until": "not a stamp"}, False],
        [{"label": "neutral", "intensity": 0, "rest": "calm", "face": "content"}, False],
        [{"label": "neutral", "intensity": 0, "problem": "state.json unreadable: JSONDecodeError"}, False],
        [{"label": "neutral", "intensity": 0}, False],
        [{"label": "calm", "intensity": 1}, True],
        [None, False],
        [{"label": "cheerful", "intensity": 2, "problem": "unreadable: JSONDecodeError"}, False],
    ])
    assert notes == [
        "\nmood: calm, 1 of 3 — the owner said so (set_mood) (held as the owner set it until 04:00)"
        "\nmanner: evil, 2 of 3 — until 04:00",
        "\nmood: calm, 1 of 3 (held as the owner set it)",
        "\nmood: neutral — rests content",
        "\nmood: unreadable — state.json unreadable: JSONDecodeError",
        "",
        "\nmood: unknown — no reading for 3 polls",
        "",
        "\nmood: unreadable — unreadable: JSONDecodeError\nmood: cheerful, 2 of 3",
    ]


# ── F11: a mood nobody has heard of for three polls is not shown ────────────

_PAGE = r"""
const mk = () => ({ textContent: '', title: null, classList: { toggle() {} },
                    getAttribute: () => 'Live log', removeAttribute(n) { this[n] = null; } });
const chip = mk(), moodEl = mk();
globalThis.document = { visibilityState: 'visible', addEventListener() {},
    getElementById: id => (id === 'status-indicator' ? chip : id === 'mood-text' ? moodEl : null) };
globalThis.window = { dispatchEvent() {} };
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push([fn, ms]); return timers.length; };
let tick = null;
globalThis.setInterval = fn => { tick = fn; return 1; };
const script = DATA.polls.slice();
let healthReads = 0;
globalThis.fetch = async url => {
    if (url === '/api/health') {
        healthReads++;
        const r = script.length ? script.shift() : 'DOWN';
        if (r === 'DOWN') throw new Error('connect');
        if (r === 'HTTP500') return { ok: false, status: 500, json: async () => ({}) };
        return { ok: true, status: 200, json: async () => r };
    }
    if (url === '/api/turns') return { ok: true, json: async () => ({ turns: [] }) };
    return { ok: false, status: 404, json: async () => ({}) };
};
const told = [];
const events = new EventTarget();
const face = DATA.face ? { setMoodHue: (label, n) => { told.push([label, n]); } } : null;
const settle = async () => { for (let i = 0; i < 25; i++) await new Promise(r => setImmediate(r)); };
S.initStatus({ Core: { events, activeFace: face } });
await settle();
const seen = [];
const snap = () => seen.push([moodEl.textContent, told.length ? told[told.length - 1] : null, moodEl.title]);
snap();
"""
_POLL_THE_REST = "while (script.length) { await tick(); await settle(); snap(); }\n"

_CALM = {"label": "calm", "intensity": 1, "face": "calm"}
_RESTS = {"label": "neutral", "intensity": 0, "rest": "calm", "face": "content"}


def test_three_polls_without_a_mood_ease_the_face_to_neutral_and_say_unknown():
    """Fails in the reviewed world, where the last look stays for ever: the agent goes down (or sends health
    without a usable mood block) and the face still reads calm an hour later."""
    polls = [{"status": "ok", "mood": _CALM}, "DOWN", {"status": "ok"},
             {"status": "ok", "mood": {"label": "calm", "intensity": "1"}}, "HTTP500",
             {"status": "ok", "mood": _RESTS}, "DOWN"]
    seen = _node(_PAGE + _POLL_THE_REST + "out(seen);", {"polls": polls, "face": True})
    words = [s[0] for s in seen]
    looks = [s[1] for s in seen]
    assert words == ["calm, 1 of 3", "calm, 1 of 3", "calm, 1 of 3", "mood unknown", "mood unknown",
                     "rests content", "rests content"]
    assert looks == [["calm", 1], ["calm", 1], ["calm", 1], ["neutral", 0], ["neutral", 0],
                     ["content", "rest"], ["content", "rest"]]
    assert seen[3][2] == "mood: unknown — no reading for 3 polls" and seen[0][2] == "mood: calm, 1 of 3"


def test_before_the_first_reading_nothing_is_claimed_then_unknown():
    seen = _node(_PAGE + _POLL_THE_REST + "out(seen);", {"polls": ["DOWN", "DOWN", "DOWN", "DOWN"], "face": True})
    assert [s[0] for s in seen] == ["", "", "mood unknown", "mood unknown"]
    assert [s[1] for s in seen] == [[None, None], [None, None], ["neutral", 0], ["neutral", 0]]


def test_the_words_do_not_need_the_face():
    """F11 (b)(c): the mood is visible text in the page, written whether or not a face takes it."""
    polls = [{"status": "ok", "mood": _CALM},
             {"status": "ok", "mood": {"label": "neutral", "intensity": 0, "problem": "state.json unreadable"}}]
    seen = _node(_PAGE + _POLL_THE_REST + "out(seen);", {"polls": polls, "face": False})
    assert [s[0] for s in seen] == ["calm, 1 of 3", "mood unreadable"]
    assert seen[1][2] == "mood: unreadable — state.json unreadable"


def test_the_end_of_a_turn_and_a_thumb_trigger_one_extra_read_not_a_faster_poll():
    """F11 (d). Fails where a mood set this turn waits for the 25 s poll (no read is scheduled), and where
    each trigger makes its own read (four reads for one turn)."""
    got = _node(_PAGE + r"""
const before = healthReads;
for (const name of ['turn-complete', 'mood-changed', 'turn-complete', 'mood-changed']) events.dispatchEvent(new Event(name));
events.dispatchEvent(new Event('conversation-cleared'));
const queued = timers.map(t => t[1]);
await settle();
const notYet = healthReads - before;
timers[0][0](); await settle();
const afterFire = healthReads - before;
// each trigger alone schedules a read too
events.dispatchEvent(new Event('mood-changed'));
const afterThumb = timers.length;
timers[1][0](); await settle();
events.dispatchEvent(new Event('turn-complete'));
out({ queued, notYet, afterFire, afterThumb, afterTurn: timers.length, words: moodEl.textContent, told });
""", {"polls": [{"status": "ok", "mood": _CALM}, {"status": "ok", "mood": {"label": "irritable", "intensity": 3}}],
      "face": True})
    assert got["queued"] == [1500] and got["notYet"] == 0 and got["afterFire"] == 1
    assert (got["afterThumb"], got["afterTurn"]) == (2, 3)
    assert got["words"] == "irritable, 3 of 3" and got["told"][-1] == ["irritable", 3]


_RACE_PAGE = r"""
const mk = () => ({ textContent: '', title: null, classList: { toggle() {} },
                    getAttribute: () => 'Live log', removeAttribute(n) { this[n] = null; } });
const chip = mk(), moodEl = mk();
const listeners = {};
globalThis.document = { visibilityState: 'visible', addEventListener(name, fn) { listeners[name] = fn; },
    getElementById: id => (id === 'status-indicator' ? chip : id === 'mood-text' ? moodEl : null) };
globalThis.window = { dispatchEvent() {} };
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push(fn); return timers.length; };
let tick = null;
globalThis.setInterval = fn => { tick = fn; return 1; };
const asked = [];                      // one resolver per /api/health request, in the order asked
let open = 0, most = 0;
globalThis.fetch = url => {
    if (url === '/api/health') {
        open++; most = Math.max(most, open);
        return new Promise((res, rej) => asked.push(m => {
            open--;
            if (m === 'DOWN') rej(new Error('connect')); else res({ ok: true, status: 200, json: async () => ({ mood: m }) });
        }));
    }
    return Promise.resolve({ ok: true, json: async () => ({ turns: [] }) });
};
const told = [];
const events = new EventTarget();
const settle = async () => { for (let i = 0; i < 25; i++) await new Promise(r => setImmediate(r)); };
S.initStatus({ Core: { events, activeFace: { setMoodHue: (l, n) => told.push([l, n]) } } });
await settle();                        // read #1 (the poll) is in flight, unanswered
"""


def test_an_older_health_answer_cannot_land_over_a_newer_one():
    """R2-6 (the reviewer's js1.mjs). Fails with two reads in flight: the slow poll asked BEFORE the turn
    ended answers last and puts the old mood back for 25 s. With one read at a time the trigger waits, so
    the last answer applied is the last one asked."""
    got = _node(_RACE_PAGE + r"""
events.dispatchEvent(new Event('turn-complete'));
timers.shift()(); await settle();                  // the extra read is due while read #1 is still out
tick(); listeners.visibilitychange(); await settle();   // … and the poll and the tab, too
const whileOut = asked.length;
asked[0](DATA.old); await settle();                // the slow poll answers with the mood from before the turn
const afterOld = [moodEl.textContent, asked.length];
asked[1](DATA.fresh); await settle();
out({ whileOut, afterOld, words: moodEl.textContent, told, asked: asked.length, most });
""", {"old": {"label": "irritable", "intensity": 2, "face": "irritable"},
      "fresh": {"label": "cheerful", "intensity": 3, "face": "cheerful"}})
    assert got["whileOut"] == 1 and got["most"] == 1, got      # never a second request beside the first
    assert got["afterOld"] == ["irritable, 2 of 3", 2]         # then ONE re-read for the three triggers
    assert got["words"] == "cheerful, 3 of 3" and got["told"] == [["irritable", 2], ["cheerful", 3]]
    assert got["asked"] == 2


def test_a_burst_of_triggers_against_a_dead_agent_is_one_missed_poll():
    """R2-6. Fails where the interval poll, the tab coming back and the extra read each fail on their own:
    three misses in seconds, "mood unknown" 50 s early."""
    got = _node(_RACE_PAGE + r"""
asked[0](DATA.calm); await settle();
tick(); await settle();                            // a poll goes out …
events.dispatchEvent(new Event('mood-changed')); timers.shift()(); listeners.visibilitychange(); tick();
await settle();
asked[1]('DOWN'); await settle();                  // … and fails with three more triggers waiting
const afterBurst = [moodEl.textContent, asked.length, told[told.length - 1]];
tick(); await settle(); asked[2]('DOWN'); await settle();
const second = moodEl.textContent;
tick(); await settle(); asked[3]('DOWN'); await settle();
out({ afterBurst, second, third: moodEl.textContent, most, last: told[told.length - 1] });
""", {"calm": {"label": "calm", "intensity": 1, "face": "calm"}})
    assert got["afterBurst"] == ["calm, 1 of 3", 2, ["calm", 1]] and got["second"] == "calm, 1 of 3"
    assert got["third"] == "mood unknown" and got["last"] == ["neutral", 0] and got["most"] == 1


def test_a_recorded_thumb_announces_the_mood_may_have_moved():
    """app.js's own ``sendFeedback`` run by node (the §4SZ harness), with the requests recorded.
    R2-11: thumb → the same thumb again → the other thumb. Fails where the second click on the active thumb
    returns without sending (the reviewed world: a thumb could not be taken back), and where a thumb or a
    retract does not ask for the mood re-read."""
    from tests.helpers import eval_js
    from tests.test_4sz_thumbs_down_hint import _APP, _HARNESS
    harness = _HARNESS.replace("fetch: async () => {", "fetch: async (url, opts) => { sent.push(JSON.parse(opts.body));")
    assert harness != _HARNESS
    src = (f"const APP = {json.dumps(str(_APP))};\n"
           "const sent = []; let heard = 0; const events = new EventTarget();\n"
           "events.addEventListener('mood-changed', () => heard++);\n"
           "globalThis.window = { GhostCore: { events } };\n" + harness
           + "replies = [{ok: true}, {ok: true, retracted: true}, {ok: true}, {ok: true}];\n"
             "page_stubs.chatHistory.push({reqId: 'r1'});\n"
             "const d = bubble('r1'); const steps = [];\n"
             "for (const sig of ['positive', 'positive', 'positive', 'negative']) {\n"
             "  await sendFeedback(d, sig);\n"
             "  steps.push([d.dataset.feedback || '', page_stubs.chatHistory[0].feedback || '', heard]);\n"
             "}\n")
    got = eval_js(src, "{sent, steps, shown}")
    assert [(b["request_id"], b["signal"], b["source"]) for b in got["sent"]] == [
        ("r1", "positive", "web"), ("r1", "retract:positive", "web"), ("r1", "positive", "web"),
        ("r1", "negative", "web")]
    # the latch on the bubble and in the stored history follow; every one of them asks for the re-read
    assert got["steps"] == [["positive", "positive", 1], ["", "", 2], ["positive", "positive", 3],
                            ["negative", "negative", 4]]
    assert got["shown"] == []


def test_a_thumb_that_could_not_be_taken_back_stays_lit_and_says_so():
    from tests.helpers import eval_js
    from tests.test_4sz_thumbs_down_hint import _APP, _HARNESS
    harness = _HARNESS.replace(
        "fetch: async () => {",
        "fetch: async (url, opts) => { sent.push(JSON.parse(opts.body).signal);"
        " if (sent.length === 2) return { ok: false, status: 503, json: async () => ({error: 'agent restarting'}) };")
    src = (f"const APP = {json.dumps(str(_APP))};\nconst sent = []; const armed = [];\n"
           "globalThis.setTimeout = (fn, ms) => { armed.push(ms); return 1; };\n" + harness
           + "replies = [{ok: true}];\nconst d = bubble('r1');\n"
             "await sendFeedback(d, 'positive'); await sendFeedback(d, 'positive');\n")
    got = eval_js(src, "{sent, armed, latched: d.dataset.feedback, shown}")
    assert got == {"sent": ["positive", "retract:positive"], "armed": [], "latched": "positive",
                   "shown": [["failure", "Thumb not taken back: agent restarting"]]}


def test_the_interface_proxy_passes_a_retract_through_unchanged():
    """The real ``feedback_proxy``: it forwards the body as sent and has no signal list of its own."""
    os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")
    from unittest.mock import AsyncMock, patch
    import interface.server as server
    from fastapi.testclient import TestClient
    resp = MagicMock(status_code=200, text="raw")
    resp.json = MagicMock(return_value={"ok": True, "retracted": True, "why": ""})
    upstream = MagicMock()
    upstream.request = AsyncMock(return_value=resp)
    body = b'{"request_id": "r1", "signal": "retract:positive", "source": "web"}'
    with patch.object(server, "_get_http_client", return_value=upstream):
        r = TestClient(server.app).post("/api/feedback", content=body, headers={
            "Content-Type": "application/json", "X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200 and r.json() == {"ok": True, "retracted": True, "why": ""}
    args, kwargs = upstream.request.call_args
    assert args[0] == "POST" and args[1].endswith("/api/feedback") and kwargs["content"] == body


# ── the handheld ────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, body, code=200):
        self.status_code, self._b = code, body

    def json(self):
        if self._b == "NOT JSON":
            raise ValueError("no json")
        return self._b


class _Http:
    """An ``httpx.AsyncClient`` double: every get/post answers with the next scripted body."""

    def __init__(self, *bodies, **_kw):
        self.bodies, self.calls = list(bodies), []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def _next(self, url):
        self.calls.append(url)
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        if body == "DOWN":
            raise ConnectionError("connect")
        return body if isinstance(body, _Resp) else _Resp(body)

    async def get(self, url, headers=None):
        return await self._next(url)

    async def post(self, url, json=None, headers=None):
        return await self._next(url)


def test_the_handheld_drops_a_look_nobody_confirmed_for_three_polls():
    """Same scenario as the web page's, on `MoodWatch`: fails where a failed poll keeps the look for ever."""
    watch = agentapi.MoodWatch()
    http = _Http({"mood": _CALM}, "DOWN", {"status": "ok"}, {"mood": {"label": "calm", "intensity": "1"}},
                 _Resp({}, 500), "NOT JSON", {"mood": _RESTS}, ["not", "a", "dict"])
    seen = []
    for _ in range(8):
        seen.append(watch.step(asyncio.run(agentapi.fetch_mood_view(http, "http://a", "k"))))
    assert seen == [(("calm", 1), "calm, 1 of 3"), (("calm", 1), "calm, 1 of 3"), (("calm", 1), "calm, 1 of 3"),
                    (("neutral", 0), "mood unknown"), (("neutral", 0), "mood unknown"),
                    (("neutral", 0), "mood unknown"), (("content", "rest"), "rests content"),
                    (("content", "rest"), "rests content")]
    assert all(url == "http://a/api/health" for url in http.calls)
    fresh = agentapi.MoodWatch()
    assert [fresh.step(None) for _ in range(3)] == [(None, ""), (None, ""), (("neutral", 0), "mood unknown")]


def test_the_handhelds_status_line_carries_the_mood_in_words():
    """F11 (b)(c): the handheld had no surface for the mood but the face. The words go first on the status
    line (rich text: escaped), cut before they can push the chips, and nothing is added for no words."""
    kw = dict(ok="#ok", danger="#bad", dim="#dim")
    plain = devstatus.status_html(True, 76, 60, "discharging", "10:30 AM", **kw)
    assert devstatus.status_html(True, 76, 60, "discharging", "10:30 AM", mood="", **kw) == plain
    got = devstatus.status_html(True, 76, 60, "discharging", "10:30 AM", mood="rests calm", **kw)
    assert got == "<span style='color:#dim; font-size:13px;'>rests calm</span>&nbsp;&nbsp;&nbsp;" + plain
    hostile = devstatus.status_html(True, 76, 60, "discharging", "t", mood="<img src=x> & 'q'", **kw)
    assert "<img" not in hostile and "&lt;img src=x&gt; &amp; &#x27;q&#x27;" in hostile
    long = devstatus.status_html(True, 76, 60, "discharging", "t", mood="m" * 200 + "\n\tz", **kw)
    assert "m" * (devstatus.MOOD_WORDS_MAX - 1) + "…<" in long and "m" * devstatus.MOOD_WORDS_MAX not in long


def _window(*names, **ns):
    """The named methods of the handheld's window run without PyQt6 (the mood-tesseract tests' harness)."""
    from tests.test_interface_face_mood_tesseract_2026_11_06 import _client_tree, _window_methods
    scope = {"agentapi": agentapi, "asyncio": asyncio, "AGENT_BASE": "http://a", "GHOST_API_KEY": "k",
             "MOOD_POLL_S": 25, "MOOD_REFRESH_DELAY_S": 0, "print": lambda *a, **k: None, **ns}
    exec(compile(_window_methods(_client_tree(), *names), "client.py", "exec"), scope)
    return scope


def _mood_window(http, face=True):
    told, drawn = [], []
    scope = _window("_mood_once", "_mood_read", "refresh_mood", "mood_loop",
                    httpx=SimpleNamespace(AsyncClient=lambda **_kw: http))
    win = SimpleNamespace(_mood_watch=agentapi.MoodWatch(), _mood_words="", _mood_refresh_due=False,
                          _mood_reading=False, _mood_again=False)
    win._mood_read = lambda: scope["_mood_read"](win)
    win.web_face = (SimpleNamespace(set_agent_mood=lambda *look: told.append(look)) if face
                    else SimpleNamespace(set_agent_mood=lambda *look: (_ for _ in ()).throw(RuntimeError("no view"))))
    win.update_stats = lambda: drawn.append(win._mood_words)
    win._mood_once = lambda: scope["_mood_once"](win)
    win.refresh_mood = lambda: scope["refresh_mood"](win)
    return scope, win, told, drawn


def test_the_handheld_client_feeds_the_face_and_the_status_line_from_one_read():
    http = _Http({"mood": _CALM}, "DOWN", "DOWN", "DOWN", {"mood": {"label": "neutral", "intensity": 0,
                                                                     "problem": "state.json unreadable"}})
    scope, win, told, drawn = _mood_window(http)

    async def five():
        for _ in range(5):
            await win._mood_once()
    asyncio.run(five())
    assert told == [("calm", 1), ("calm", 1), ("calm", 1), ("neutral", 0), ("neutral", 0)]
    assert drawn == ["calm, 1 of 3", "mood unknown", "mood unreadable"]         # redrawn only when the words change

    # animations off / a face that cannot take it: the words still arrive
    scope, win, told, drawn = _mood_window(_Http({"mood": _RESTS}), face=False)
    asyncio.run(win._mood_once())
    assert drawn == ["rests content"]

    # a transport that cannot even be opened is one poll without a mood, not a dead loop
    def boom(**_kw):
        raise OSError("no network stack")
    scope = _window("_mood_read", httpx=SimpleNamespace(AsyncClient=boom))
    win = SimpleNamespace(_mood_watch=agentapi.MoodWatch(), _mood_words="", update_stats=lambda: None,
                          web_face=SimpleNamespace(set_agent_mood=lambda *a: None))
    asyncio.run(scope["_mood_read"](win))
    assert win._mood_watch.misses == 1


def test_the_handheld_reads_the_mood_once_more_at_the_end_of_a_turn():
    """F11 (d) on the handheld: several triggers inside the wait share one read; with the mood poll switched
    off there is none."""
    http = _Http({"mood": _CALM})
    scope, win, told, _drawn = _mood_window(http)

    async def turn():
        for _ in range(4):
            win.refresh_mood()
        assert win._mood_refresh_due is True
        for _ in range(6):
            await asyncio.sleep(0)
        first = len(http.calls)
        win.refresh_mood()
        for _ in range(6):
            await asyncio.sleep(0)
        return first, len(http.calls)
    assert asyncio.run(turn()) == (1, 2) and told == [("calm", 1), ("calm", 1)]

    scope["MOOD_POLL_S"] = 0
    http.calls.clear()

    async def off():
        win.refresh_mood()
        for _ in range(6):
            await asyncio.sleep(0)
    asyncio.run(off())
    assert http.calls == [] and win._mood_refresh_due is False


class _SlowHttp:
    """Each GET waits for its own release, so a test decides the order the answers arrive in."""

    def __init__(self):
        self.gates, self.open, self.most = [], 0, 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, url, headers=None):
        fut = asyncio.get_running_loop().create_future()
        self.gates.append(fut)
        self.open += 1
        self.most = max(self.most, self.open)
        try:
            body = await fut
        finally:
            self.open -= 1
        if body == "DOWN":
            raise ConnectionError("connect")
        return _Resp({"mood": body})


async def _spin():
    for _ in range(8):
        await asyncio.sleep(0)


def test_the_handheld_reads_the_mood_one_read_at_a_time():
    """R2-6 on the handheld. Fails with two reads in flight (the slow poll's old mood lands last), and
    where a burst against a dead agent counts as three missed polls."""
    http = _SlowHttp()
    scope, win, told, drawn = _mood_window(http)

    async def race():
        poll = asyncio.ensure_future(win._mood_once())          # the 25 s poll, slow
        await _spin()
        extra = [asyncio.ensure_future(win._mood_once()) for _ in range(3)]   # the turn ended meanwhile
        await _spin()
        assert len(http.gates) == 1                              # nobody asked beside the read in flight
        http.gates[0].set_result({"label": "irritable", "intensity": 2})      # the mood from BEFORE the turn
        await _spin()
        assert len(http.gates) == 2 and win._mood_words == "irritable, 2 of 3"
        http.gates[1].set_result({"label": "cheerful", "intensity": 3})
        await asyncio.gather(poll, *extra)
    asyncio.run(race())
    assert told == [("irritable", 2), ("cheerful", 3)] and win._mood_words == "cheerful, 3 of 3"
    assert http.most == 1 and (win._mood_reading, win._mood_again) == (False, False)

    async def outage():
        poll = asyncio.ensure_future(win._mood_once())
        await _spin()
        extra = [asyncio.ensure_future(win._mood_once()) for _ in range(3)]
        await _spin()
        http.gates[2].set_result("DOWN")
        await asyncio.gather(poll, *extra)
    asyncio.run(outage())
    assert len(http.gates) == 3 and win._mood_watch.misses == 1 and win._mood_words == "cheerful, 3 of 3"


NOTE = "Your set mood holds until 04:00 — this thumb is saved but doesn't move the mood."


@pytest.mark.parametrize("body,want", [
    ({"ok": True, "mood_note": NOTE}, (True, NOTE)),
    ({"ok": True, "mood_note": "  spaced  "}, (True, "spaced")),
    ({"ok": True}, (True, "")), ({"ok": True, "mood_note": 7}, (True, "")),
    ({"ok": True, "mood_note": None}, (True, "")), ("NOT JSON", (True, "")), (["x"], (True, "")),
])
def test_the_agents_note_comes_back_with_a_recorded_thumb(body, want):
    async def no_sleep(_s):
        return None
    got = asyncio.run(agentapi.send_feedback(_Http(_Resp(body)), "http://a", "k", "r1", "positive", sleep=no_sleep))
    assert got == want


def test_the_handheld_takes_the_lit_thumb_back_when_it_is_pressed_again():
    """R2-11 on the handheld: the chip that is lit, pressed again, sends the retract and goes dark; the
    other chip still switches; a note makes it a (re)label, never a retract."""
    async def no_sleep(_s):
        return None
    for sig in ("retract:positive", "retract:negative"):
        http = _Http(_Resp({"ok": True, "retracted": True}))
        assert asyncio.run(agentapi.send_feedback(http, "http://a", "k", "r1", sig, sleep=no_sleep)) == (True, "")
    for bad in ("retract", "retract:", "retract:up", "Positive", ""):
        assert asyncio.run(agentapi.send_feedback(_Http({}), "http://a", "k", "r1", bad, sleep=no_sleep)) == \
            (False, "bad signal")

    sent = []

    class _Rec(_Http):
        async def post(self, url, json=None, headers=None):
            sent.append(json["signal"])
            return await self._next(url)
    http = _Rec(_Resp({"ok": True}))
    scope = _window("rate_last", "_send_rating", httpx=SimpleNamespace(AsyncClient=lambda **_kw: http),
                    NOTE_ERR="ERR")
    lit, rereads, said = [], [], []
    win = SimpleNamespace(_last_reply_rid="r1", _rated=None, _mood_note_shown=False, _note_activity=lambda: None,
                          _note=lambda text, tone=None: said.append(text),
                          _refresh_rating_chips=lambda: lit.append(win._rated),
                          refresh_mood=lambda: rereads.append(1))
    win._send_rating = lambda rid, signal, note: scope["_send_rating"](win, rid, signal, note)

    async def presses():
        for signal, note in (("positive", ""), ("positive", ""), ("negative", ""), ("negative", "wrong year"),
                             ("negative", "")):
            scope["rate_last"](win, signal, note)
            await _spin()
    asyncio.run(presses())
    assert sent == ["positive", "retract:positive", "negative", "negative", "retract:negative"]
    assert lit == ["positive", None, "negative", "negative", None] and len(rereads) == 5 and said == []

    fail = _Rec(_Resp({"error": "agent restarting"}, 400))
    scope = _window("_send_rating", httpx=SimpleNamespace(AsyncClient=lambda **_kw: fail), NOTE_ERR="ERR")
    win._rated = "positive"
    asyncio.run(scope["_send_rating"](win, "r1", "retract:positive", ""))
    assert win._rated == "positive" and said == ["thumb not taken back: agent restarting"]


def test_the_handheld_says_once_that_a_set_mood_holds_and_rereads_the_mood():
    """F15: the web page showed the agent's "your set mood holds" line; on the handheld the thumb was
    recorded and nothing was said."""
    http = _Http(_Resp({"ok": True, "mood_note": NOTE}))
    scope = _window("_send_rating", httpx=SimpleNamespace(AsyncClient=lambda **_kw: http), NOTE_ERR="ERR")
    said, rereads, chips = [], [], []
    win = SimpleNamespace(_last_reply_rid="r1", _rated=None, _mood_note_shown=False,
                          _note=lambda text, tone=None: said.append((text, tone)),
                          _refresh_rating_chips=lambda: chips.append(1), refresh_mood=lambda: rereads.append(1))
    asyncio.run(scope["_send_rating"](win, "r1", "positive", ""))
    asyncio.run(scope["_send_rating"](win, "r1", "negative", ""))
    assert said == [(NOTE, None)] and win._rated == "negative" and len(rereads) == 2 and len(chips) == 2

    # a failed rating says why, in the error tone, and does not re-read
    fail = _Http(_Resp({"error": "no trajectory found"}, 400))
    scope = _window("_send_rating", httpx=SimpleNamespace(AsyncClient=lambda **_kw: fail), NOTE_ERR="ERR")
    said.clear(), rereads.clear()
    asyncio.run(scope["_send_rating"](win, "r1", "positive", ""))
    assert said == [("rating not recorded: no trajectory found", "ERR")] and rereads == []


def test_the_handheld_window_starts_with_a_mood_watch_and_ends_a_turn_with_a_reread():
    """Wiring, walked on the AST: the mood state is set before the window's first status draw, the
    status line is drawn WITH the words, and a finished turn asks for the re-read."""
    import ast
    from tests.test_interface_face_mood_tesseract_2026_11_06 import _client_tree
    tree = _client_tree()
    methods = {n.name: n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # set where the window is built (`initUI`, the method that makes the status line), before its first draw
    build = methods["initUI"]
    set_at = {t.attr: n.lineno for n in ast.walk(build) if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Attribute)}
    first_draw = min(c.lineno for c in ast.walk(build) if isinstance(c, ast.Call)
                     and ast.unparse(c.func) == "self.update_stats")
    for name in ("_mood_watch", "_mood_words", "_mood_refresh_due", "_mood_note_shown", "_mood_reading",
                 "_mood_again"):
        assert set_at[name] < first_draw, name
    drawn = [kw for c in ast.walk(methods["update_stats"]) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "status_html" for kw in c.keywords if kw.arg == "mood"]
    assert len(drawn) == 1 and ast.unparse(drawn[0].value) == "self._mood_words"
    callers = sorted(name for name, fn in methods.items()
                     if any(isinstance(c, ast.Call) and ast.unparse(c.func) == "self.refresh_mood"
                            for c in ast.walk(fn)))
    assert len(callers) == 2 and "_send_rating" in callers     # the turn's end and the thumb


# ── /api/health ─────────────────────────────────────────────────────────────

KEYS = {"label", "intensity", "manner", "manner_intensity", "manner_until", "set_at", "source", "cause",
        "held_until", "rest", "face", "problem", "p", "e", "shade", "shade_at", "shade_age_s"}


def _health(context):
    from ghost_agent.api import routes
    return routes._health_mood(context)


def _real(tmp_path, rest=None):
    from ghost_agent.selfhood.model import SelfModel
    sm = SelfModel(root=tmp_path / "self")
    if rest is not None:
        sm.state.attach_rest(lambda: rest)
    return sm


def test_health_sends_the_manners_level_and_end_and_the_holds_end(tmp_path):
    """F10 on today's engine. Fails in the reviewed world: the block had the manner's name only."""
    from ghost_agent.selfhood import mood as M
    sm = _real(tmp_path)
    sm.state.set_manner("evil", 3)
    sm.state.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 2)
    now = sm.state.mood()
    got = _health(SimpleNamespace(self_model=sm))
    assert set(got) == KEYS
    assert (got["label"], got["intensity"], got["face"], got["source"]) == ("irritable", 2, "irritable", "owner")
    assert got["manner"] == "evil" and got["manner_intensity"] == now.manner_intensity and 1 <= got["manner_intensity"] <= 3
    assert got["manner_until"] == now.manner_steps[-1][0] and got["held_until"] == now.held_until != ""
    # both end at the day's end, 04:00 on this machine's clock — and that is what the clients print
    assert agentapi.mood_clock(got["manner_until"]) == agentapi.mood_clock(got["held_until"]) == "04:00"
    assert agentapi.mood_words(agentapi.mood_view(got)) == \
        f"irritable, 2 of 3 · manner evil, {got['manner_intensity']} of 3 — until 04:00"
    assert got["problem"] == ""


def test_health_sends_one_face_label_and_the_fallback_is_the_rest(tmp_path):
    """W1/W4 on today's engine (no ``stats()["face"]``): neutral with a resting point → the rest's region."""
    got = _health(SimpleNamespace(self_model=_real(tmp_path, rest=(0.12, -0.06))))
    assert (got["label"], got["intensity"], got["rest"], got["face"]) == ("neutral", 0, "calm", "calm")
    assert (got["manner"], got["manner_intensity"], got["manner_until"], got["held_until"]) == ("", 0, "", "")
    assert agentapi.mood_words(agentapi.mood_view(got)) == "rests calm"
    plain = _health(SimpleNamespace(self_model=_real(tmp_path / "z")))
    assert (plain["face"], plain["rest"]) == ("neutral", "neutral")


class _View(SimpleNamespace):
    pass


def _engine(*views, load_problem="", rest=(0.12, -0.06)):
    """A state thread whose mood is a different view at every read, and a self-model whose ``stats`` must
    never be asked (it rebuilds a search index: ~50 ms on the event loop)."""
    reads, left = [], list(views)

    def mood():
        reads.append(1)
        return left.pop(0) if len(left) > 1 else left[0]

    def stats(**_kw):
        raise AssertionError("health asked SelfModel.stats()")
    state = SimpleNamespace(mood=mood, load_problem=load_problem, rest=lambda: rest)
    return SimpleNamespace(self_model=SimpleNamespace(state=state, stats=stats)), reads


def _v(label, n, **kw):
    base = dict(label=label, intensity=n, manner="", manner_intensity=0, source="auto", set_at="", cause="",
                held_until="", manner_steps=())
    return _View(**{**base, **kw})


def test_health_builds_the_block_from_one_read_and_never_asks_stats():
    """R2-5. Fails with a second read: a thumb lands between them (`content, 1` was read, then the mood is
    `calm`) and the block says label content with face calm. And fails where health calls ``stats()``."""
    ctx, reads = _engine(_v("content", 1, cause="a thumbs-up", held_until="2026-11-12T02:00:00Z", source="owner"),
                         _v("calm", 2), _v("neutral", 0))
    got = _health(ctx)
    assert reads == [1]
    assert (got["label"], got["intensity"], got["face"], got["cause"], got["held_until"], got["source"]) == \
        ("content", 1, "content", "a thumbs-up", "2026-11-12T02:00:00Z", "owner")
    assert got["rest"] == "calm"
    # every block health can send is one both clients can show: 300 states, each read exactly once
    from ghost_agent.selfhood import mood as M
    rng = random.Random(7)
    for _ in range(300):
        label = rng.choice(list(M.LABELS) + ["neutral", "smug", "", 7, None])
        n = rng.choice([0, 0, 1, 2, 3, 9, -2, True, "2", None])
        rest = rng.choice([(0.0, 0.0), (0.12, -0.06), (-0.12, 0.0), (0.0, -0.12), ("x", 1), None, (float("nan"), 0)])
        view = _v(label, n, manner=rng.choice(["", "evil", 7]), manner_intensity=rng.choice([0, 2, 9, "1"]),
                  manner_steps=rng.choice([(), (("2026-11-12T02:00:00Z", 0),), "soon", ((1, 2),), None]),
                  held_until=rng.choice(["", "2026-11-12T02:00:00Z", 5]))
        ctx, reads = _engine(view, _v("irritable", 3), rest=rest)
        got = json.loads(json.dumps(_health(ctx)))
        assert reads == [1] and set(got) == KEYS, (label, n)
        assert got["face"] == M.face_label(got["label"], got["intensity"], rest), (label, n, rest, got)
        assert (got["intensity"] > 0) == (got["label"] != "neutral") and got["rest"] == M.rest_label(rest)
        assert agentapi.mood_view(got) is not None, got                  # never a block a client must drop


def test_the_face_rule_is_the_engines_and_health_keeps_no_copy():
    """W1: one rule, one implementation. ``routes`` has no rest/face rule of its own any more."""
    import ast
    import inspect
    from ghost_agent.api import routes
    from ghost_agent.selfhood import mood as M
    assert not hasattr(routes, "_rest_label")
    fn = next(n for n in ast.parse(inspect.getsource(routes)).body
              if isinstance(n, ast.FunctionDef) and n.name == "_health_mood")
    called = {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert {"_M.face_label", "_M.rest_label"} <= called and "_M.region" not in called
    # and the block follows that rule where it differs from a naive one: an unhurried rest is calm, not bored
    got = _health(_engine(_v("neutral", 0), rest=(0.0, -0.12))[0])
    assert (got["face"], got["rest"]) == (M.rest_label((0.0, -0.12)), "calm") == ("calm", "calm")


def test_the_manners_end_comes_from_the_views_last_step():
    got = _health(_engine(_v("neutral", 0, manner="evil", manner_intensity=2,
                             manner_steps=(("2026-11-12T00:00:00Z", 1), ("2026-11-12T02:00:00Z", 0))))[0])
    assert (got["manner"], got["manner_intensity"], got["manner_until"]) == ("evil", 2, "2026-11-12T02:00:00Z")
    assert got["held_until"] == ""
    got = _health(_engine(_v("neutral", 0, held_until="2026-11-12T03:00:00Z"))[0])
    assert got["held_until"] == ""                      # a neutral mood is not held


@pytest.mark.parametrize("load_problem,want", [
    ("unreadable: JSONDecodeError: Expecting value", "unreadable: JSONDecodeError: Expecting value"),
    ("  truncated file  ", "truncated file"),
    ("p" * 400, "p" * 120), ("", ""), (None, ""), (7, ""), (lambda: "x", ""),
])
def test_health_says_when_the_state_file_could_not_be_read(load_problem, want):
    """F12. Fails where an unreadable state reads as a healthy neutral: both clients then say so in words."""
    got = _health(_engine(_v("neutral", 0), load_problem=load_problem, rest=(0.0, 0.0))[0])
    assert got["problem"] == want
    words = agentapi.mood_words(agentapi.mood_view(got))
    assert (words == "mood unreadable") is bool(want)
    # R2-4: beside a real mood it is added, and the look stays the mood's
    got = _health(_engine(_v("cheerful", 2), load_problem=load_problem)[0])
    assert agentapi.mood_words(agentapi.mood_view(got)) == \
        "cheerful, 2 of 3" + (" · state file was unreadable at start" if want else "")


def test_the_real_state_thread_says_an_unreadable_file_through_health(tmp_path):
    """End to end on the engine's own ``load_problem``: a truncated state.json."""
    from ghost_agent.selfhood.model import SelfModel
    sm = SelfModel(root=tmp_path / "self")
    sm.state.set_manner("terse", 2)
    sm.state.path.write_text(sm.state.path.read_text(encoding="utf-8")[:40], encoding="utf-8")
    got = _health(SimpleNamespace(self_model=SelfModel(root=tmp_path / "self")))
    assert got["problem"].startswith("unreadable: ") and (got["label"], got["manner"]) == ("neutral", "")
    assert agentapi.mood_words(agentapi.mood_view(got)) == "mood unreadable"


def test_health_without_a_self_model_or_with_a_mock_is_the_neutral_block():
    from ghost_agent.api import routes
    for ctx in (None, SimpleNamespace(), SimpleNamespace(self_model=None), MagicMock(),
                SimpleNamespace(self_model=SimpleNamespace(state=SimpleNamespace(
                    mood=lambda: (_ for _ in ()).throw(RuntimeError("boom")), load_problem="unreadable")))):
        got = _health(ctx)
        assert set(got) == KEYS and (got["label"], got["intensity"], got["face"]) == ("neutral", 0, "neutral")
    assert got["problem"] == "unreadable"                # kept even when the mood read itself failed
    assert _health(None) is not routes._MOOD_NEUTRAL and _health(None) == routes._MOOD_NEUTRAL


def test_the_health_endpoint_serves_that_block(tmp_path):
    from ghost_agent.api import routes
    sm = _real(tmp_path, rest=(0.12, -0.06))
    sm.state.set_manner("terse", 2)
    agent = SimpleNamespace(context=SimpleNamespace(self_model=sm, llm_client=None, scheduler=None,
                                                     memory_system=None))
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(agent=agent, biological_task=None, boot_monotonic=None,
                                                    resolved_config={}))
    body = json.loads(asyncio.new_event_loop().run_until_complete(routes.api_health(req)).body)
    assert body["mood"] == routes._health_mood(agent.context)
    assert (body["mood"]["face"], body["mood"]["manner"], body["mood"]["manner_intensity"] >= 1) == ("calm", "terse", True)
    look, words = _node("out([S.faceMoodFrom(DATA), S.moodWords(S.moodView(DATA), false)]);", body["mood"], tz="UTC")
    assert look == ["calm", "rest"] and words.startswith("rests calm · manner terse, ")


# ── rendered: where the words sit (WebKit) ──────────────────────────────────

@pytest.fixture(scope="module")
def browser():
    from tests.test_interface_header_simplify import _real_playwright
    try:
        sync_playwright = _real_playwright().sync_playwright
    except ImportError:               # pragma: no cover
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        try:
            b = p.webkit.launch(timeout=120_000)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                pytest.skip(f"WebKit not installed: {str(e)[:80]}")
            raise
        yield b
        b.close()


_MEASURE = """() => {
    const box = e => { const r = e.getBoundingClientRect();
                       return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, h: r.height, w: r.width}; };
    const el = id => document.getElementById(id);
    const range = document.createRange(); range.selectNodeContents(el('token-count'));
    return {pill: box(el('input-area')), mood: box(el('mood-text')), count: box(range), row: box(el('token-count')),
            field: box(el('chat-input')), mic: box(el('mic-btn')), send: box(el('send-btn')),
            clipped: el('mood-text').scrollWidth > el('mood-text').clientWidth,
            shown: getComputedStyle(el('mood-text')).visibility + '/' + getComputedStyle(el('mood-text')).display};
}"""


@pytest.mark.parametrize("width,height", [(375, 667), (430, 932), (820, 1180), (1280, 720), (1440, 900)])
def test_the_mood_words_sit_on_the_counters_line_and_move_nothing(browser, width, height):
    """Rendered. The words are visible at every width (a phone included: no hover there), on the token
    counter's line, clear of the counter's digits and of the controls, and the composer is the same size
    with and without them."""
    from tests.test_interface_header_simplify import _serve
    ctx = browser.new_context(viewport={"width": width, "height": height},
                              has_touch=width < 900, is_mobile=width < 600)
    page = ctx.new_page()
    try:
        _serve(page)
        page.goto("http://ghost.test/", wait_until="load")
        page.evaluate("""() => { document.getElementById('effort-select').classList.remove('hidden');
            const t = document.getElementById('token-count'); t.textContent = '↑~29.6k ↓12.4k';
            t.classList.remove('hidden'); }""")
        empty = page.evaluate(_MEASURE)
        for words in ("rests calm", "irritable, 2 of 3 · manner sarcastic, 3 of 3 — until 04:00 " + "and more " * 20):
            page.evaluate("(w) => { document.getElementById('mood-text').textContent = w; }", words)
            m = page.evaluate(_MEASURE)
            assert m["shown"] == "visible/block" and m["mood"]["w"] > 20 and m["mood"]["h"] >= 11, m
            # on the counter's line …
            assert abs(m["mood"]["top"] - m["row"]["top"]) <= 1.5 and abs(m["mood"]["bottom"] - m["row"]["bottom"]) <= 1.5, m
            # … inside the pill, left of the counter's digits, under the field and the controls
            assert m["mood"]["left"] >= m["pill"]["left"] + 8 and m["mood"]["right"] <= m["count"]["left"] - 8, m
            assert m["mood"]["top"] >= max(m["field"]["bottom"], m["mic"]["bottom"], m["send"]["bottom"]) - 1, m
            # … and nothing moved for it
            for part in ("pill", "field", "mic", "send", "row", "count"):
                assert m[part] == empty[part], (part, m[part], empty[part])
        assert m["clipped"] is True, m                   # a phrase too long for the line is cut, at every width
    finally:
        ctx.close()
