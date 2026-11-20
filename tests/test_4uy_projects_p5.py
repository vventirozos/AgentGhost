"""§4UY P5 (finding §4UX-3 P5, req eb25d165): a QUESTION about a ready project was answered by rewriting a data
file and `lab put`-ing it to the live site. The rule pinned here:

* `provenance.is_pure_question` — structure only: every sentence of the message ends with a question mark
  (`?`, U+037E, or `;` in a message that holds Greek letters); quoted and fenced text is not read;
* `provenance.question_only_request` — the request's message and every message added while it runs are pure
  questions (one instruction, first or queued, lifts it for the rest of the request);
* `labs._question_wall` — in the OWNER's own turn of such a request every publish action of the `lab` tool
  (`labs.PUBLISH_ACTIONS`; `put` only into a KEPT lab) is refused before any host write, with a who-may-act code.
  Reads, disposable-lab work, workspace edits and every other kind of turn are untouched.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import time

import pytest

from ghost_agent.tools import labs as L
from ghost_agent.tools.outcome import WHO_MAY_ACT_REASONS, OutcomeStatus, is_who_may_act_wall
from ghost_agent.utils import provenance as PV
from ghost_agent.utils.logging import request_id_context

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_4sx_d3_kept import LAB, LAB2, PID, _app, _det, _lab, ctx, fake, scope  # noqa: E402,F401

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "ghost_agent"
CODE = "question_not_instruction"
QUESTION = "What's in Elena's shop project?"
INSTRUCTION = "Put today's prices on the shop site."


# ── the predicate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "What's in the shop project?",
    "what is in the shop?   ",
    "Is it live? Which version serves?",
    "Is it live?\nWhich version serves?",
    "Hey Ghost, what's in the shop?",                       # a vocative inside the sentence
    "So... what's in the shop?",                            # a bare ellipsis ends nothing
    "So… what's in the shop?",
    "1. What's in it?\n2. Is it live?",                     # list numbers are not sentences
    "Is v1.2 live? Is index.html there?",                   # a dot inside a word ends nothing
    "really?!", "Is it up!?", "Is it live? :)", "Is it live? (the shop one?)",
    "What does \"deploy it now!\" mean?",                   # the quoted instruction is not the owner's
    "What does `rm -rf dist; make deploy.` do?",
    "Why does this fail?\n```\nmake deploy.\nexit 1\n```",
    "Τι έχει το κατάστημα της Έλενας;",                     # Greek question mark typed as ;
    "Τι έχει το κατάστημα\u037e",                           # … and as U+037E
    "Τι έχει το κατάστημα; Είναι ανεβασμένο;",
    "Τι κάνει το `x = 1;`;",                                # a span standing alone after a Greek word
    "Τι έχει το (κατάστημα);", "Τι έχει το 'κατάστημα';", "Είναι σωστό το κατάστημα(παλιό);",
    "Είναι το «παλιό» (κατάστημα);", "Είναι το «shop» live;",    # the word before: spans are stepped over
    # review r2 N-2: a Latin last word inside a Greek sentence
    "Είναι live;", "Τρέχει το site;", "Είναι online το site;", "Είναι το shop της Elena;", "Είναι OK;",
    "Τι έχει το κατάστημα; Είναι live;", "Is it live?(the shop one)",
    "What about 'it's broken?",                             # no closing quote: nothing is masked
    "Is the rock 'n' roll page live?", "Is Elena's and Markos' shop live?",    # apostrophes mask nothing
    "Is it live (the shop one)?", "[Elena] why is the shop down?",
    "Τι έχει το «κατάστημα»; Ποια έκδοση τρέχει;",
    "Is it live\u037e",                                     # U+037E is a question mark in any text
    "Is the 5\" screen supported?",                         # an unclosed quote is left as it is
    "Can you deploy it?",                                   # a request worded as a question IS a question here
])
def test_a_message_whose_every_sentence_ends_with_a_question_mark_is_a_pure_question(text):
    assert PV.is_pure_question(text) is True


@pytest.mark.parametrize("text", [
    "", "   ", "??", "?", ":)",
    "what is in the shop",                                  # no question mark
    "Put today's prices on the shop site.",
    "Deploy it!",
    "What's in the shop? Deploy it.",                       # mixed: question + instruction
    "Deploy it. What's in the shop?",
    "What's in the shop? Update the prices",                # words after the last end mark
    "What's in the shop?\nthanks",
    "Update the prices\nIs it live?",                       # a line with no end mark is a sentence
    "Hi! What's in the shop?",                              # a greeting as its own sentence: not PURE
    "Check it; what is in the shop?",                       # `;` ends a sentence, and is no question mark in English
    "Is it e.g. live?",                                     # an abbreviation's dot before a space ends a sentence
    "Change the heading to \"What's new?\"",                # the question mark is inside the quote
    "Change the heading to \u201cWhat's new?\u201d",
    "Άλλαξε τον τίτλο σε «Τι νέα;»",
    "Rename it to `ready?`",
    "```\nis it live?\n```",                                # only fenced text
    "\"What's in the shop?\"",                              # only quoted text
    "Τι έχει το κατάστημα; Ανέβασέ το.",                    # Greek mixed
    "Ανέβασε τις τιμές στο κατάστημα.",
    "Ανέβασέ το\u0387 τι έχει;",                            # the Greek semicolon (ano teleia) ends a sentence
    "Deploy it; then rest;",                                # `;` is a question mark only on a Greek word
    # review r1 A-1: an instruction that ENDS in quoted / bracketed / code text
    "Rename the button to 'Ready?'", "Change the heading to 'What's new?'",
    "Change the heading to \u2018What\u2019s new?\u2019", "Άλλαξε τον τίτλο σε 'Τι νέα;'",
    "Deploy Elena's shop. Is it the kids' one?",            # an apostrophe inside a word opens no span
    "Set the title to (Ready?)", "Set the title to [Ready?]", "Deploy it (ok?)",
    "Πρόσθεσε τη γραμμή x = 1;", "Τρέξε: SELECT * FROM t;", "Βάλε στο css το color: red;",
    # review r2 N-1: a Greek instruction ending in code that closes with a bracket / quote before the `;`
    "Πρόσθεσε τη γραμμή foo();", "Βάλε στο αρχείο print(\"hi\");", "Πρόσθεσε const a = [1, 2];",
    "Βάλε το x = 'a';", "Βάλε color: (red);", "Τρέξε: «SELECT 1»;", "Τρέξε: deploy;",
    "Είναι έτοιμο το v2;", "Είναι σωστό 100%;",            # limits: the last word is neither Greek nor a plain word
    "Fix it; deploy;", "live;",
    "Γιατί δεν δουλεύει; Fix it; deploy;",
])
def test_anything_else_is_not_a_pure_question(text):
    assert PV.is_pure_question(text) is False


def test_the_predicate_takes_any_value_and_long_text():
    assert PV.is_pure_question(None) is False and PV.is_pure_question(7) is False
    assert PV.is_pure_question(["Is it live?"]) is False            # not a string: never a question

    class _Says:
        def __str__(self):
            return "Is it live?"
    assert PV.is_pure_question(_Says()) is False
    assert PV.is_pure_question("Is it live? " * 20000) is True
    assert PV.is_pure_question("is it \ue000 live?") is True and PV.is_pure_question("\ue000?") is False   # the mask itself


#: every character that opens a masked span, alone and after a space, and the other marks the scan reads
_OPENERS = ["`", "```", '"', "\u201c", "\u201e", "\u00ab", "(", "[", "'", "\u2018"]


@pytest.mark.parametrize("unit", _OPENERS + [" " + o for o in _OPENERS]
                         + ["'a ", " 'a", "' ", "( ) ", "\u201e\u201c", "? ", ". ", "; ", "\n", "\ue000"])
def test_the_predicate_is_linear_in_unclosed_openers(unit):
    """Review r1 A-3: 20k unclosed « took 1.5 s and 500k never ended (a regex per opener). 500k of anything now
    returns at once; the bound is wide because the battery runs this under coverage."""
    t0 = time.monotonic()
    PV.is_pure_question(unit * 500_000)
    PV.is_pure_question("Is it live? " + unit * 500_000 + " is it?")
    assert time.monotonic() - t0 < 10.0
    assert PV.is_pure_question("\"" * 50001 + " is it live?") is True    # the odd quote out stays as it is
    assert PV.is_pure_question("`" * 50001) is False and PV.is_pure_question("." * 50001 + " x") is False


# ── the request ──────────────────────────────────────────────────────────────

class _req:
    """One request: its id in the context and its user message recorded as the agent loop does."""

    def __init__(self, rid, text):
        self.rid, self.text = rid, text

    def __enter__(self):
        self.tok = request_id_context.set(self.rid)
        PV.note_user_message(self.text, self.rid)
        return self.rid

    def __exit__(self, *a):
        request_id_context.reset(self.tok)


def test_the_request_is_question_only_until_an_instruction_arrives():
    with _req("req-elena-q1", QUESTION) as rid:
        assert PV.question_only_request(rid) and PV.question_only_request()     # by id and from the context
        PV.note_added_user_message("And is it live?", rid)                       # a queued question (§4RZ)
        assert PV.question_only_request(rid)
        PV.note_added_user_message("Yes, deploy it.", rid)                       # a queued instruction lifts it
        assert not PV.question_only_request(rid)
        PV.note_added_user_message("Is it done?", rid)                           # … for the rest of the request
        assert not PV.question_only_request(rid)
    with _req("req-elena-q1", QUESTION) as rid:                                  # a NEW request under the same id
        assert PV.question_only_request(rid)


def test_a_question_added_to_an_instruction_never_makes_the_request_question_only():
    with _req("req-elena-q2", INSTRUCTION) as rid:
        assert not PV.question_only_request(rid)
        PV.note_added_user_message("How long will it take?", rid)
        assert not PV.question_only_request(rid)


def test_no_recorded_message_is_not_a_question():
    assert not PV.question_only_request("req-kosmas-none")
    with _req("req-elena-q3", "   ") as rid:
        assert not PV.question_only_request(rid)
        PV.note_added_user_message("Is it live?", rid)       # the blank first message is no message
        assert PV.question_only_request(rid)


def test_the_verdict_does_not_depend_on_which_map_was_evicted():
    """Review r1 A-2: the message and its additions lived in two LRU maps of 512 — after 600 other requests an
    instruction request with a queued question was walled (the additions survived alone). One entry now; a
    dropped entry is "unknown", never question-only."""
    with _req("req-elena-evict-1", INSTRUCTION) as rid:
        PV.note_added_user_message("How long will it take?", rid)
        for i in range(PV._CAP + 90):
            PV.note_user_message("Is it live?", f"req-other-a{i}")
        assert not PV.question_only_request(rid)
        PV.note_added_user_message("Is it done?", rid)                 # an addition never rebuilds a dropped entry
        assert not PV.question_only_request(rid)
        assert rid not in PV._ASKED


def test_a_request_that_is_still_judged_keeps_its_verdict_under_load():
    with _req("req-elena-evict-2", QUESTION) as rid:
        for i in range(PV._CAP + 90):
            PV.note_user_message(INSTRUCTION, f"req-other-b{i}")
            if i % 100 == 0:
                assert PV.question_only_request(rid)                   # a read refreshes the entry
        assert PV.question_only_request(rid)
        PV.note_added_user_message("Deploy it.", rid)
        assert not PV.question_only_request(rid)
    assert len(PV._ASKED) <= PV._CAP


def test_a_dropped_entry_is_unknown_not_a_question():
    with _req("req-elena-evict-3", QUESTION) as rid:
        for i in range(PV._CAP + 1):
            PV.note_user_message("Is it live?", f"req-other-c{i}")
        assert not PV.question_only_request(rid)                       # never read meanwhile: dropped → no wall


def _slack_file_notes():
    """Every note the Slack bot writes into the user's own message for an attached file (its f-strings that open
    with "[SYSTEM NOTE: The user attached"), rendered with a fictional file name."""
    src = (SRC.parent.parent / "interface" / "externals" / "slack_bot" / "main.py").read_text(encoding="utf-8")
    notes = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant) \
                and str(node.values[0].value).startswith("[SYSTEM NOTE: The user attached"):
            notes.append("".join(v.value if isinstance(v, ast.Constant) else "prices (final?).csv"
                                 for v in node.values))
    return notes


@pytest.mark.parametrize("unit", ["[SYSTEM NOTE: The user attached a file named ", "<@U1|", "<@U1", "<@",
                                  "(File just uploaded to the sandbox: ", "vision_analysis (target='",
                                  "[SYSTEM NOTE: The user attached a file named x vision_analysis (target='"])
def test_reading_the_client_notes_is_linear(unit):
    """Review r2 N-5: the note patterns were quadratic on a note opened again and again and never closed (500k
    chars: 25 s and 130 s), and the wall reads them at every publish attempt."""
    t0 = time.monotonic()
    assert PV._owner_words((unit * (500_000 // len(unit) + 1))[:500_000])
    assert time.monotonic() - t0 < 10.0


def test_the_bounded_note_patterns_still_take_out_whole_notes():
    from ghost_agent.selfhood.sentiment import message_text
    long_name = "prices " + "x" * 240 + ".csv"
    for note in _slack_file_notes():
        assert message_text("Is it live? " + note.replace("prices (final?).csv", long_name)) == "Is it live?"
    assert message_text(f"Is it live? (File just uploaded to the sandbox: {long_name})") == "Is it live?"
    assert message_text("<@U0123ABCD|elena> is it live? <@W9>") == "is it live?"


def test_a_slack_file_note_is_not_the_users_sentence():
    """Review r1 A-4: the bot glues `[SYSTEM NOTE: …]` onto the message, so every Slack question with a file
    read as "not a question". The notes are taken out by their one reader (`sentiment.message_text`)."""
    notes = _slack_file_notes()
    assert len(notes) >= 4, notes
    for n, note in enumerate(notes):
        assert PV._owner_words(note) == "", note                       # the shared pattern knows every one
        with _req(f"req-elena-slack-{n}", QUESTION + "\n\n" + note) as rid:
            assert PV.question_only_request(rid), note
        with _req(f"req-elena-slack-{n}", INSTRUCTION + "\n\n" + note) as rid:
            assert not PV.question_only_request(rid), note
        with _req(f"req-elena-slack-{n}", note) as rid:                # a file with no caption asks nothing
            assert not PV.question_only_request(rid)
    with _req("req-elena-slack-m", "<@U0123ABCD> what's in the shop?") as rid:
        assert PV.question_only_request(rid)


# ── the wall ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_kept_cache(monkeypatch):
    monkeypatch.setattr(L, "_KEPT_LAB_SEEN", {})


def _workspace(c):
    d = c.sandbox_dir / "projects" / PID
    (d / "js" / "data").mkdir(parents=True, exist_ok=True)
    (d / "js" / "data" / "prices.json").write_text('{"asOf": "today"}')
    (d / "index.html").write_text("<h1>shop</h1>")
    return d


def _routes(fake, lifetime="kept"):
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime=lifetime, app=_app(workdir="/app") if lifetime == "kept"
                                                 else None)))
    fake.on("PUT", f"/v1/labs/{LAB}/files", (200, {"bytes": 17}))
    fake.on("PUT", f"/v1/labs/{LAB}/tree", (200, {"entries": 3, "bytes": 40}))
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    fake.on("POST", f"/v1/labs/{LAB}/switch", (200, {"switched": True, "from_lab": LAB2, "app": _app(),
                                                     "probe": {"ok": True}}))
    fake.on("POST", f"/v1/labs/{LAB}/rollback", (200, {"lab_id": LAB, "rolled_back_from": LAB2, "app": _app(),
                                                       "probe": {"ok": True}}))
    fake.on("POST", f"/v1/labs/{LAB}/restore", (200, {"restored": "b1", "app": _app(), "probe": {"ok": True}}))
    fake.on("POST", f"/v1/labs/{LAB}/keep", (200, _det(app=_app(state="stopped"))))


_F = f"projects/{PID}"
#: one call per way of publishing: {name: (kwargs, the host write it makes)}
PUBLISH_CALLS = {
    "put": (dict(action="put", lab_id=LAB, member="web", sandbox_path=f"{_F}/js/data/prices.json",
                 path="/app/js/data/prices.json"), ("PUT", f"/v1/labs/{LAB}/files")),
    "put-data": (dict(action="put", lab_id=LAB, member="web", sandbox_path=f"{_F}/js/data/prices.json",
                      path="/data/js/data/prices.json"), ("PUT", f"/v1/labs/{LAB}/files")),
    "put-folder": (dict(action="put", lab_id=LAB, member="web", sandbox_path=f"{_F}/js", path="/app/js"),
                   ("PUT", f"/v1/labs/{LAB}/tree")),
    "deploy": (dict(action="deploy", lab_id=LAB, params={"folder": f"/workspace/{_F}"}),
               ("POST", f"/v1/labs/{LAB}/deploy")),
    "switch": (dict(action="switch", lab_id=LAB, params={"from_lab": LAB2}), ("POST", f"/v1/labs/{LAB}/switch")),
    "rollback": (dict(action="rollback", lab_id=LAB), ("POST", f"/v1/labs/{LAB}/rollback")),
    "restore": (dict(action="restore", lab_id=LAB, params={"backup_id": "b1"}),
                ("POST", f"/v1/labs/{LAB}/restore")),
    "keep": (dict(action="keep", lab_id=LAB, params={"app": {"name": "shop", "member": "web", "port": 8080,
                                                             "cmd": "python3 app.py"}}),
             ("POST", f"/v1/labs/{LAB}/keep")),
}


def _writes(fake):
    return [(c.method, c.path) for c in fake.calls if c.method != "GET"]


def test_the_publish_calls_cover_every_publish_action():
    assert {kw["action"] for kw, _ in PUBLISH_CALLS.values()} == set(L.PUBLISH_ACTIONS)


@pytest.mark.parametrize("name", sorted(PUBLISH_CALLS))
async def test_no_publish_action_reaches_the_host_in_a_request_that_only_asked(ctx, fake, scope, name):
    kw, write = PUBLISH_CALLS[name]
    scope(ctx)
    _workspace(ctx)
    _routes(fake, lifetime="disposable" if name == "keep" else "kept")
    if name == "keep":
        fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable")))
    with _req(f"req-elena-{name}", QUESTION):
        out = await _lab(ctx, **kw)
    assert out.status is OutcomeStatus.REJECTED and out.reason_code == CODE, str(out)
    assert is_who_may_act_wall(out) and out.world_changed is False
    assert _writes(fake) == []
    # the same call when the owner asked for it: the host gets the write
    fake.calls.clear()
    with _req(f"req-elena-{name}", INSTRUCTION):
        out = await _lab(ctx, **kw)
    assert getattr(out, "reason_code", None) != CODE and write in _writes(fake), str(out)


async def test_the_eb25d165_shape_question_then_file_edit_then_put(ctx, fake, scope):
    """The live shape: a question, the data file rewritten in the project, two `lab put`s into the kept lab. The
    edit stays (files are not walled); neither put is sent; reading the lab still works."""
    scope(ctx)
    d = _workspace(ctx)
    _routes(fake)
    fake.on("POST", f"/v1/labs/{LAB}/http", (200, {"status": 200, "body": "{}", "headers": {}}))
    with _req("req-elena-eb", "Τι έχει μέσα το κατάστημα της Έλενας;"):
        (d / "js" / "data" / "prices.json").write_text('{"asOf": "rewritten"}')
        a = await _lab(ctx, **PUBLISH_CALLS["put"][0])
        b = await _lab(ctx, **PUBLISH_CALLS["put-data"][0])
        st = await _lab(ctx, action="status", lab_id=LAB)
        ht = await _lab(ctx, action="http", lab_id=LAB, member="web", path="/js/data/prices.json")
    assert a.reason_code == CODE and b.reason_code == CODE
    assert [w for w in _writes(fake) if w[0] == "PUT"] == []
    assert (d / "js" / "data" / "prices.json").read_text() == '{"asOf": "rewritten"}'
    assert getattr(st, "reason_code", None) != CODE and "LAB ERROR (not_allowed)" not in str(st)
    assert getattr(ht, "reason_code", None) != CODE and ("POST", f"/v1/labs/{LAB}/http") in _writes(fake)


async def test_the_refusal_steers_and_never_asks_for_a_yes(ctx, fake, scope):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    with _req("req-elena-text", QUESTION):
        s = str(await _lab(ctx, **PUBLISH_CALLS["deploy"][0]))
    low = s.lower()
    assert s.startswith("LAB ERROR (not_allowed): NOT done")
    assert "question" in low and "say in the reply what you would deploy" in low and "nothing was sent" in low
    assert "the user deploys by asking for it" in low and "not live" in low
    for word in ("confirm", "yes", "approve", "permission", "shall i", "should i", "ok?"):   # §4UK
        assert word not in low, word


async def test_a_queued_instruction_lifts_the_wall_within_the_request(ctx, fake, scope):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    kw, write = PUBLISH_CALLS["put"]
    with _req("req-elena-queued", QUESTION) as rid:
        first = await _lab(ctx, **kw)
        PV.note_added_user_message("Is that the new file?", rid)
        second = await _lab(ctx, **kw)
        assert _writes(fake) == []
        PV.note_added_user_message("Fine — put it live.", rid)
        third = await _lab(ctx, **kw)
    assert first.reason_code == CODE and second.reason_code == CODE
    assert third.status is OutcomeStatus.OK and _writes(fake) == [write]


async def test_an_instruction_request_is_not_walled_after_its_entry_was_dropped(ctx, fake, scope):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    kw, write = PUBLISH_CALLS["deploy"]
    with _req("req-elena-evict-w", INSTRUCTION) as rid:
        PV.note_added_user_message("Is it building?", rid)
        for i in range(PV._CAP + 90):
            PV.note_user_message("Is it live?", f"req-other-w{i}")
        out = await _lab(ctx, **kw)
    assert out.status is OutcomeStatus.OK and _writes(fake) == [write]


async def test_the_next_request_is_judged_by_its_own_message(ctx, fake, scope):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    kw, write = PUBLISH_CALLS["deploy"]
    with _req("req-elena-a", QUESTION):
        assert (await _lab(ctx, **kw)).reason_code == CODE
    with _req("req-elena-b", "Ανέβασε το κατάστημα."):
        assert (await _lab(ctx, **kw)).status is OutcomeStatus.OK
    with _req("req-elena-c", "Και τώρα, είναι ανεβασμένο;"):               # the earlier instruction does not carry over
        assert (await _lab(ctx, **kw)).reason_code == CODE
    assert _writes(fake) == [write]


async def test_put_into_a_disposable_lab_runs_in_a_question_request(ctx, fake, scope):
    """Trying something on a throwaway machine is part of answering — only a KEPT lab is the live site."""
    scope(ctx)
    _workspace(ctx)
    _routes(fake, lifetime="disposable")
    kw, write = PUBLISH_CALLS["put"]
    with _req("req-elena-disp", "Does prices.json parse on that machine?"):
        out = await _lab(ctx, **kw)
        tree = await _lab(ctx, **PUBLISH_CALLS["put-folder"][0])
    assert out.status is OutcomeStatus.OK and tree.status is OutcomeStatus.OK, (str(out), str(tree))
    assert _writes(fake) == [write, PUBLISH_CALLS["put-folder"][1]]


@pytest.mark.parametrize("answer", [(500, {"error": "boom"}), (200, "not a dict"), RuntimeError("down"),
                                    L.LabsNoAnswer("timeout")])
async def test_put_is_refused_when_the_host_cannot_say_whether_the_lab_is_kept(ctx, fake, scope, answer):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    fake.on("GET", f"/v1/labs/{LAB}", answer)
    with _req("req-elena-unsure", QUESTION):
        out = await _lab(ctx, **PUBLISH_CALLS["put"][0])
    assert out.reason_code == CODE and "could not be checked" in str(out) and _writes(fake) == []
    assert "no answer" not in str(out)                     # review r1 A-6: a 500 IS an answer


async def test_put_into_a_lab_the_host_does_not_know_gets_the_ordinary_answer(ctx, fake, scope):
    scope(ctx)
    _workspace(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "not_found", "message": "no such lab"}))
    fake.on("PUT", f"/v1/labs/{LAB}/files", (404, {"error": "not_found", "message": "no such lab"}))
    with _req("req-elena-404", QUESTION):
        out = await _lab(ctx, **PUBLISH_CALLS["put"][0])
    assert getattr(out, "reason_code", None) != CODE and "not_found" in str(out)


async def test_a_kept_lab_seen_before_is_walled_without_asking_the_host_again(ctx, fake, scope):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    L._KEPT_LAB_SEEN[LAB] = (f"project:{PID}", "https://labs.example/", "web", "/app")   # as `_note_project_push` keeps it
    with _req("req-elena-seen2", QUESTION):
        out = await _lab(ctx, **PUBLISH_CALLS["put"][0])
    assert out.reason_code == CODE and fake.calls == []


async def test_a_publish_call_with_bad_arguments_gets_the_wall_not_an_argument_puzzle(ctx, fake, scope):
    """The wall comes before the argument errors: the model is not led to repair a call that cannot run."""
    scope(ctx)
    with _req("req-elena-args", QUESTION):
        no_lab = await _lab(ctx, action="deploy", params={"files": ["a.js"]})
        no_folder = await _lab(ctx, action="deploy", lab_id=LAB)
        put_no_lab = await _lab(ctx, action="put", member="web", sandbox_path="x", path="/app/x")
    assert no_lab.reason_code == CODE and no_folder.reason_code == CODE
    assert put_no_lab.reason_code == "lab_bad_request" and fake.calls == []      # no lab named: nothing to wall


@pytest.mark.parametrize("action", sorted(L.NOT_PUBLISH_ACTIONS))
async def test_no_other_action_meets_the_question_wall(ctx, fake, scope, action):
    scope(ctx)
    _workspace(ctx)
    _routes(fake)
    with _req(f"req-elena-other-{action}", QUESTION):
        out = await _lab(ctx, action=action, lab_id=LAB, member="web", cmd="ls", path="/app/x",
                         sandbox_path=f"{_F}/got.txt", op="restart", template="static-site")
    assert getattr(out, "reason_code", None) != CODE, str(out)
    assert "asked a QUESTION" not in str(out)


# ── the other kinds of turn: unchanged ───────────────────────────────────────

#: (request id, scope origin) of every kind of turn that is not the owner's own interactive one
OTHER_TURNS = [("probe-3f34a566", "probe"), ("sched-autopilot-7", "autopilot"), ("sched-autopilot-8", "owner"),
               ("sub-ptask-1a2b3c4d", "owner"), ("sub-ptask-1a2b3c4e", "autopilot"), ("sub-agent-9", "owner"),
               ("SYSTEM", "owner"), ("req-markos-1", "member"),
               ("req-kosmas-bg", "autopilot"), ("req-kosmas-probe", "probe")]   # the scope alone says "not the owner"


@pytest.mark.parametrize("rid,origin", OTHER_TURNS)
@pytest.mark.parametrize("name", sorted(PUBLISH_CALLS))
async def test_other_turns_get_the_answer_they_got_before(ctx, fake, scope, rid, origin, name):
    """A probe, a scheduled run, a project task's turn, a sub-agent, a call outside any request, a channel member:
    the question wall never answers — the result is the same whether the root message asks or instructs."""
    kw, _write = PUBLISH_CALLS[name]
    scope(ctx, origin=origin)
    _workspace(ctx)
    seen = []
    for text in (QUESTION, INSTRUCTION):
        fake.calls.clear()
        _routes(fake, lifetime="disposable" if name == "keep" else "kept")
        L._KEPT_LAB_SEEN.clear()
        ctx.memory_dir = ctx.memory_dir.parent / f"memory-{len(seen)}"      # a `keep` records a run: fresh each time
        ctx.memory_dir.mkdir()
        with _req(rid, text):
            out = await _lab(ctx, **kw)
        assert getattr(out, "reason_code", None) != CODE, str(out)
        seen.append((str(out), getattr(out, "status", None), getattr(out, "reason_code", None),
                     [(c.method, c.path) for c in fake.calls]))
    assert seen[0] == seen[1]


async def test_a_task_turn_under_a_question_root_is_not_the_owners_turn(ctx, fake, scope):
    """The root's message is read through the lineage — and still the wall is the OWNER's own turn only."""
    scope(ctx)
    PV.note_user_message(QUESTION, "req-elena-root")
    PV.link_child("sub-ptask-77aa88bb", "req-elena-root")
    tok = request_id_context.set("sub-ptask-77aa88bb")
    try:
        assert PV.question_only_request() is True          # the root only asked …
        assert L._asked_only_a_question(ctx) is False      # … but this turn is not the owner's own
    finally:
        request_id_context.reset(tok)


# ── no way round: the registry and the source ────────────────────────────────

def test_every_lab_action_is_publish_or_not_and_the_code_is_declared():
    assert L.PUBLISH_ACTIONS | L.NOT_PUBLISH_ACTIONS == set(L.ACTIONS)
    assert not (L.PUBLISH_ACTIONS & L.NOT_PUBLISH_ACTIONS)
    assert CODE in WHO_MAY_ACT_REASONS
    assert set(L.LAB_TOOL_DEFINITION["function"]["parameters"]["properties"]["action"]["enum"]) == set(L.ACTIONS)


def _functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _called(node):
    return getattr(node.func, "id", None) or getattr(node.func, "attr", None)


#: the last path segment of a host route that changes what a kept lab serves
_PUBLISH_ROUTES = {"files", "tree", "deploy", "switch", "rollback", "restore", "keep"}


def _route_tail(arg):
    """The constant tail of a `_call` path argument (`f"/v1/labs/{lab_id}/files"` → "files"), or None."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value.rstrip("/").rsplit("/", 1)[-1]
    if isinstance(arg, ast.JoinedStr) and arg.values:
        last = arg.values[-1]
        if isinstance(last, ast.Constant) and isinstance(last.value, str) and "/" in last.value:
            return last.value.rstrip("/").rsplit("/", 1)[-1]
        return "<computed>"
    return None


def _labs_imports(tree):
    """Names this module binds to tools/labs.py or to something in it (`from .labs import _dispatch as d`,
    `from ..tools import labs as L`, `import ghost_agent.tools.labs as L`)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = str(node.module or "")
            for al in node.names:
                if mod.split(".")[-1] == "labs" or al.name == "labs":
                    names.add(al.asname or al.name)
        elif isinstance(node, ast.Import):
            names |= {al.asname or al.name for al in node.names if al.name.split(".")[-1] == "labs"}
    return names


def _is_from_labs(call, labs_names):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id in labs_names
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in labs_names


def test_every_host_write_that_publishes_sits_behind_the_one_gate():
    """A registry / AST walk, so a new deploy path cannot go round `_question_wall`:
    * every host call (`_call` / `_long_call`) to a publishing route — or to a route computed at run time, with a
      PUT / POST — is made inside tools/labs.py, in the action table (`_dispatch`, `_kept_action`,
      `_put_tree`);
    * that table is entered only through `tool_lab`, which asks `_question_wall` before it dispatches;
    * every "this went live" record (`_note_project_push(how=…)`) names a publish action."""
    hows, entered_from = set(), {"_dispatch": set(), "_kept_action": set(), "_put_tree": set()}
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC).as_posix()
        labs_names = _labs_imports(tree)
        for fn in _functions(tree):
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                name = _called(node)
                if name in entered_from and (rel == "tools/labs.py" or _is_from_labs(node, labs_names)):
                    entered_from[name].add((rel, fn.name))
                if name == "_note_project_push":
                    hows |= {kw.value.value for kw in node.keywords if kw.arg == "how"}
                if name in ("_call", "_long_call") and len(node.args) >= 3:
                    method = node.args[1].value if isinstance(node.args[1], ast.Constant) else "<computed>"
                    tail = _route_tail(node.args[2])
                    if method not in ("PUT", "POST", "<computed>") or tail is None:
                        continue
                    if tail in _PUBLISH_ROUTES or (tail == "<computed>" and rel == "tools/labs.py"):
                        assert rel == "tools/labs.py" and fn.name in entered_from, (rel, fn.name, method, tail)
    assert hows and hows <= L.PUBLISH_ACTIONS, hows
    assert entered_from["_dispatch"] == {("tools/labs.py", "tool_lab")}
    assert entered_from["_kept_action"] == {("tools/labs.py", "_dispatch")}
    assert entered_from["_put_tree"] == {("tools/labs.py", "_dispatch")}
    # … and `tool_lab` asks the wall before the table
    src = ast.parse((SRC / "tools" / "labs.py").read_text(encoding="utf-8"))
    tool = next(f for f in _functions(src) if f.name == "tool_lab")
    order = [(_called(n), n.lineno) for n in ast.walk(tool) if isinstance(n, ast.Call)
             and _called(n) in ("_question_wall", "_dispatch")]
    assert min(ln for nm, ln in order if nm == "_question_wall") < min(ln for nm, ln in order if nm == "_dispatch")


def test_the_registry_serves_the_lab_tool_through_tool_lab_only():
    """No second model-facing entry to the labs host: `tool_lab` is referenced by the registry alone."""
    users = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called(node) == "tool_lab":
                users.add(path.relative_to(SRC).as_posix())
    assert users == {"tools/registry.py"}, users
