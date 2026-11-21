"""§4VE — the intent reading (what the owner's message is DOING) beside the tone and the emotions: how the agent
keeps it (``sentiment._answer``, the readings store), what it does to the mood (one tone per message, through the
reaction ledger) and what journal mode writes instead. The service is faked by a tiny HTTP server that answers a
body the test sets, so the real client, ``_answer`` and ``score_message`` run. Each test names the wrong
implementation it catches. All text is fictional (Elena, Markos, Kosmas).
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ghost_agent.selfhood import intent as I
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import sentiment as C
from ghost_agent.selfhood.state import SelfStateThread

KEY = "test-key"
OLD_ROW_KEYS = {"ts", "request_id", "label", "score", "probs", "reason", "nudge", "held", "chars", "text"}


def _body(label="neutral", score=0.9, intent=None, urgent=None, emotions=None, reason=""):
    probs = {"negative": 0.05, "neutral": 0.05, "positive": 0.05}
    if label in probs:
        probs[label] = score
    else:
        probs = {}
    b = {"ok": True, "label": label, "score": score, "probs": probs, "reason": reason}
    if emotions is not None:
        b["emotions"] = emotions
    if intent is not None:
        b["intent"] = intent
    if urgent is not None:
        b["urgent"] = urgent
    return b


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.server.seen.append(json.loads(self.rfile.read(n) or b"{}").get("text"))
        data = json.dumps(self.server.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.body, server.seen = _body(), []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    monkeypatch.setenv("GHOST_SENTIMENT_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(key_file))
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    C.BREAKER.reset()
    for k in ("GHOST_INTENT", "GHOST_INTENT_MOOD", "GHOST_INTENT_REGISTER", "GHOST_INTENT_MEMBERS",
              "GHOST_INTENT_JUDGE", "GHOST_INTENT_CHARACTER"):
        monkeypatch.delenv(k, raising=False)
    yield
    C.BREAKER.reset()


def _model(tmp_path):
    from ghost_agent.selfhood import SelfModel
    return SelfModel(tmp_path / "selfhood")


def _events(sm):
    p = sm.root / "mood.events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _logged(monkeypatch):
    from ghost_agent.utils import logging as L
    calls = []
    monkeypatch.setattr(L, "pretty_log", lambda title, content=None, icon="", **kw: calls.append((title, content, icon)))
    return calls


#: a real-shaped reading (11-20 measurement): one INDEPENDENT score per label — they do not sum to 1
VENT = {"venting": 0.93, "confiding": 0.41, "correcting": 0.22, "banter": 0.03, "question": 0.04, "advice": 0.06,
        "instruction": 0.02, "greeting": 0.01, "reply": 0.05, "thanks": 0.01}


def _only(label, p=0.95):
    """``label`` at ``p`` (above every floor), every other label low — independent scores, all ten labels."""
    return {k: (p if k == label else 0.05) for k in I.LABELS}


# ── parse ───────────────────────────────────────────────────────────────────

def test_parse_keeps_a_good_reading():
    assert I.parse({"intent": VENT, "urgent": 0.8}) == {"intent": VENT, "urgent": 0.8}


@pytest.mark.parametrize("intent", [
    {"venting": 0.9, "rant": 0.1},                 # an unknown label (catches: unknown labels kept or skipped)
    {"venting": True},                             # a bool is not a number
    {"venting": 1.2}, {"venting": -0.1}, {"venting": float("nan")}, {"venting": "0.9"},
    [0.9, 0.1], "venting", {},                     # wrong type, empty
])
def test_a_bad_intent_field_drops_only_itself(intent):
    # catches: one bad value keeping the rest of the field, or dropping the urgency with it
    assert I.parse({"intent": intent, "urgent": 0.75}) == {"urgent": 0.75}


def test_parse_missing_and_urgency_alone():
    assert I.parse({"ok": True}) is None and I.parse(None) is None and I.parse([1]) is None
    assert I.parse({"urgent": 0.9}) == {"urgent": 0.9}
    assert I.parse({"urgent": True}) is None and I.parse({"urgent": 2}) is None
    assert I.parse({"urgent": "0.9"}) is None                    # catches: a string read as a number


def test_a_huge_number_drops_only_its_own_field():
    # review r1 NIT 8: 10**400 overflowed float() and threw the valid intent away with it
    assert I.parse({"intent": VENT, "urgent": 10 ** 400}) == {"intent": VENT}
    assert I.parse({"intent": {"venting": 10 ** 400}, "urgent": 0.8}) == {"urgent": 0.8}


def test_an_unvalidated_reading_with_string_numbers_reads_as_none():
    # catches: top / log_part / register trusting a reading that did not come through parse
    bad = {"intent": {"advice": "0.9"}, "urgent": "0.9"}
    assert I.top(bad) == ("", 0.0) and I.log_part(bad) == "" and not I.urgent(bad)
    assert I.mood_event(bad, "what now?")[0] is None


def test_the_answer_keeps_the_intent_for_an_unknown_tone_and_drops_only_a_bad_one():
    # catches: copying the emotions' `label != "unknown"` condition (operator decision 8: Greek keeps it)
    got = C._answer(_body("unknown", 0.0, intent=_only("advice"), reason="not English",
                          emotions={"joy": 0.9}))
    assert got["intent"] == _only("advice") and "emotions" not in got
    # catches: a malformed intent field throwing the whole sentiment answer away
    bad = C._answer(_body("positive", 0.9, intent={"rant": 0.9}, urgent=0.8))
    assert bad["label"] == "positive" and "intent" not in bad and bad["urgent"] == 0.8


def test_mode_0_does_not_even_parse(monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "0")
    # catches: parsing the field whatever the switch says
    assert set(C._answer(_body("positive", 0.9, intent=_only("banter"), urgent=0.9))) == {
        "label", "score", "probs", "reason"}


def test_mode_and_switches(monkeypatch):
    assert I.mode() == "journal" and not I.acts("mood") and I.would("mood") and not I.would("members")
    monkeypatch.setenv("GHOST_INTENT", "1")
    assert I.acts("mood") and I.acts("register") and not I.acts("members") and not I.acts("nonsense")
    monkeypatch.setenv("GHOST_INTENT_MOOD", "0")
    assert not I.acts("mood") and I.acts("judge")
    monkeypatch.setenv("GHOST_INTENT", "garbage")          # unreadable: the safe default
    assert I.mode() == "journal"


# ── top ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent,want", [
    ({"advice": 0.90, "question": 0.89}, "advice"),        # at its floor; a close second is no reason (no margin)
    ({"advice": 0.8999, "question": 0.10}, ""),            # under its own floor
    ({"question": 0.60, "advice": 0.59}, "question"),      # question's floor is 0.6
    ({"question": 0.5999}, ""),
    ({"greeting": 0.85}, "greeting"), ({"greeting": 0.8499}, ""),
    ({"instruction": 0.11, "question": 0.05, "advice": 0.04}, "instruction"),   # no floor: wins at 0.11
    ({"banter": 0.90}, "banter"), ({"banter": 0.89}, ""),  # the provisional floor of the unmeasured ones
    ({"confiding": 0.89}, ""), ({"venting": 0.89}, ""), ({"correcting": 0.89}, ""),
    ({"reply": 0.99, "advice": 0.5}, ""),                  # reply never — and no falling through to advice
    ({"thanks": 1.0, "instruction": 0.3}, ""),             # thanks never — not even to instruction (no floor)
    ({"advice": 0.95, "question": 0.95}, ""),              # a tie at the top
])
def test_top_takes_the_highest_label_at_its_own_floor(intent, want):
    assert I.top({"intent": intent})[0] == want
    assert I.counted({"intent": intent}) == want           # the day records: one rule
    assert I.top({}) == ("", 0.0) and I.top(None) == ("", 0.0)


def test_instruction_has_a_small_floor_under_the_lowest_real_score():
    # catches: instruction's floor back at 0.0 (a catch-all) or raised over the measured low-but-real 0.11
    assert I.top({"intent": {**{k: 0.01 for k in I.LABELS}, "instruction": 0.11}})[0] == "instruction"
    assert I.top({"intent": {**{k: 0.01 for k in I.LABELS}, "instruction": 0.02}}) == ("", 0.0)
    assert 0.0 < I.LABEL_MIN["instruction"] <= 0.10


def test_every_label_has_a_floor_or_is_never_taken():
    # catches: a label added without deciding its floor; "never" said with a magic number instead of NEVER
    assert set(I.LABEL_MIN) == set(I.LABELS)
    assert {lab for lab, f in I.LABEL_MIN.items() if f is I.NEVER} == {"reply", "thanks"}
    assert all(f is I.NEVER or 0.0 <= f <= 1.0 for f in I.LABEL_MIN.values())


@pytest.mark.parametrize("name,intent,want", [
    ("greeting with banter", {"greeting": 1.00, "banter": 0.83, "reply": 0.40}, "greeting"),
    ("instruction low", {"instruction": 0.11, "question": 0.08, "advice": 0.03, "reply": 0.02}, "instruction"),
    ("advice under its floor", {"advice": 0.89, "question": 0.30, "instruction": 0.20}, ""),
    ("reply at 0.99", {"reply": 0.99, "thanks": 0.70, "greeting": 0.60}, ""),
    # review r2: nothing scores — instruction is no catch-all (its floor 0.10 sits under the lowest real 0.11)
    ("nothing scores", {**{k: 0.01 for k in I.LABELS}, "instruction": 0.02}, ""),
])
def test_real_shaped_readings(name, intent, want):
    """The 11-20 model's shape: independent scores, several high at once."""
    reading = {"label": "positive", "emotions": {"joy": 0.9, "amusement": 0.8}, "intent": intent}
    assert I.top(reading)[0] == want, name
    ev, why = I.mood_event(reading, "haha hi Ghost, how are you?")
    assert ev is None, name                                # greeting moves nothing; banter (second) never acts
    assert I.register(reading, audience="owner", text="haha hi Ghost, how are you?") == ("", ""), name
    if not want:
        assert why == I.WHY_UNCLEAR, name


# ── each effect needs its second witness ────────────────────────────────────

@pytest.mark.parametrize("label,reading,text,moves", [
    ("advice", {}, "what would you do about Markos's garden?", True),
    ("advice", {"emotions": {"curiosity": 0.7}}, "tell me what you think of the plan", True),
    ("advice", {}, "tell me what you think of the plan", False),          # no question, no curiosity
    ("confiding", {}, "I have been feeling lost since Kosmas moved away", True),
    ("confiding", {}, "the weather has been strange all week", False),   # not about the owner
    ("confiding", {}, "delete my old notes about Kosmas", False),         # a task's wording
    # review r1 MINOR 4: a possessive in an instruction is not the owner speaking of themselves
    ("confiding", {}, "send my report to Elena", False),
    ("confiding", {}, "check my code", False),
    ("confiding", {}, "honestly the move has me stressed out", True),    # a feeling in the words
    ("banter", {}, "haha you are a goofball", True),
    ("banter", {"emotions": {"amusement": 0.8}}, "you are a goofball", True),
    ("banter", {}, "you are a goofball", False),                          # no laugh, no amusement
    ("banter", {}, "I'm not kidding, you are a goofball", False),        # "kidding" is no laugh
    ("banter", {}, "stop joking around, goofball", False),
    ("banter", {"emotions": {"annoyance": 0.7}}, "haha you are a goofball", False),
    ("banter", {}, "great, you broke it again lol", False),               # sarcasm (trouble words)
    # review r1 MAJOR 2: banter under an unhappy tone (any score) is refused
    ("banter", {"label": "negative"}, "haha this is awful", False),
    ("venting", {"emotions": {"sadness": 0.8}}, "the whole week at work has been awful", True),
    ("venting", {"label": "negative"}, "the whole week at work has been awful", True),   # no emotion reading
    ("venting", {}, "the whole week at work has been awful", False),      # nothing unhappy backs it
    ("venting", {"label": "negative"}, "delete the failed jobs", False),  # a task's wording
    # review r1 MAJOR 2: with an emotion reading the tone label is no witness — the tone gate refused it
    ("venting", {"label": "negative", "emotions": {"confusion": 0.8}},
     "the dashboard is still empty after the update", False),
    ("venting", {"label": "neutral", "emotions": {"confusion": 0.8}}, "the dashboard is still empty", False),
])
def test_each_mood_effect_needs_its_witness(label, reading, text, moves):
    # catches: a classifier alone moving the mood (operator decision 10)
    ev, why = I.mood_event({**reading, "intent": _only(label)}, text)
    assert (ev is not None) is moves, why
    if moves:
        assert ev.kind == M.KIND_SENTIMENT and ev.cause in I.EVENT_CAUSES
    else:
        assert why


#: one reading + text per EVENT label that makes its event with nothing doubtful about it
EVENT_FIXTURES = {
    "advice": ({}, "what would you do about Markos's garden?"),
    "confiding": ({}, "I have been feeling lost since Kosmas moved away"),
    "banter": ({}, "haha you are a goofball"),
    "venting": ({"emotions": {"sadness": 0.8}}, "the whole week at work has been awful"),
}


def test_every_event_label_is_refused_when_its_tone_is_doubted():
    """Review r1 MAJOR 1 — the CLASS: "great, you broke it again, what would you do now?" after the agent's
    trouble (positive 0.7, under its gate) lifted the mood as advice. Every label that can make an event is
    refused by the tone's doubt and by the tone's missing backing; a new event label must be added here."""
    event_labels = {lab for lab, cause in I.MOOD_OUTCOMES.items() if cause is not None}
    assert set(EVENT_FIXTURES) == event_labels           # catches: a new event label with no doubt checked
    for lab, (extra, text) in EVENT_FIXTURES.items():
        ev, why = I.mood_event({**extra, "intent": _only(lab)}, text)
        assert ev is not None, (lab, why)                                   # the control: it does move
        sarcastic = {**extra, "intent": _only(lab), "label": "positive", "emotions": {"joy": 0.8}}
        ev, why = I.mood_event(sarcastic, "great, you broke it again. " + text, after_trouble=True)
        assert ev is None and why in (C.HELD_SARCASM, I.WHY_TONE_AGAINST), (lab, why)
        ev, why = I.mood_event(sarcastic, text, after_trouble=True)        # upbeat right after trouble
        assert ev is None and why in (C.HELD_SARCASM, I.WHY_TONE_AGAINST), (lab, why)
        # review r3 (item 2): the doubt comes FIRST — the reason kept is the doubt whatever the emotion model read
        # (none, nothing on the tone's side, or the label's own emotion); the day close counts from that reason
        for emo in (None, {"neutral": 0.9}, {"amusement": 0.8}, {"joy": 0.8}, {"sadness": 0.8}):
            r = {k: v for k, v in sarcastic.items() if k != "emotions"}
            if emo is not None:
                r["emotions"] = emo
            assert I.mood_event(r, "great, you broke it again. " + text, after_trouble=True) == (
                None, C.HELD_SARCASM), (lab, emo)
            assert I.refusal(r, lab, "great, you broke it again. " + text, after_trouble=True) == C.HELD_SARCASM
        # … and the same refusal stops every register clause of that label that can offend (patience cannot:
        # its witness, an unsure emotion, is absent here)
        assert I.register(sarcastic, audience="owner", text="great, you broke it again. " + text,
                          after_trouble=True) == ("", "")


#: under an UNBACKED tone (an emotion reading with nothing on the tone's side): each label that is NOT a tone and
#: has an event or a clause, with a reading that reaches them — (reading, text, event cause or None, clause key)
UNBACKED_REACHABLE = {
    "advice": ({"label": "negative", "emotions": {"confusion": 0.8}}, "evaluate my plan first", I.CAUSE_ASKED,
               "patient"),
    "confiding": ({"label": "positive", "emotions": {"sadness": 0.7}},
                  "I have been feeling lost since Kosmas moved away", I.CAUSE_CONFIDED, "acknowledge"),
    "instruction": ({"label": "positive", "emotions": {"curiosity": 0.6}}, "make the chart for Elena bigger", None,
                    "brief"),
}
#: … and each label that IS a tone, under an unbacked tone of its own side
UNBACKED_TONE = {
    "banter": ({"label": "positive", "emotions": {"curiosity": 0.8}}, "haha you are a goofball"),
    "venting": ({"label": "negative", "emotions": {"confusion": 0.8}}, "the whole week at work has been awful"),
}


def test_an_unbacked_tone_refuses_only_the_labels_that_are_a_tone():
    """Coordinator 11-20 (the corrected rule): an unbacked tone says nothing about a feeling — exactly when what
    the message is DOING is the useful reading. Catches the over-broad rule (backing refusing every label), and
    a new label with an event or clause whose fate under an unbacked tone was never decided."""
    with_effect = {lab for lab in I.LABELS if I.MOOD_OUTCOMES.get(lab) or I.REGISTER_OUTCOMES.get(lab)}
    tone_labels = {lab for lab, side in I.INTENT_TONE.items() if side}
    assert set(I.INTENT_TONE) == set(I.LABELS)
    assert set(UNBACKED_REACHABLE) == with_effect - tone_labels
    assert set(UNBACKED_TONE) == tone_labels & with_effect
    for lab, (extra, text, cause, key) in UNBACKED_REACHABLE.items():
        reading = {**extra, "intent": _only(lab)}
        assert M.sentiment_backing(extra["label"], extra["emotions"]) is False          # the premise
        ev, why = I.mood_event(reading, text)
        assert (ev.cause if ev else None) == cause, (lab, why)
        assert I.register(reading, audience="owner", text=text)[0] == key, lab
    for lab, (extra, text) in UNBACKED_TONE.items():
        reading = {**extra, "intent": _only(lab)}
        assert I.mood_event(reading, text) == (None, C.HELD_UNBACKED), lab
        assert I.register(reading, audience="owner", text=text) == ("", ""), lab


#: fictional, the same shape as the case the feature exists for: self-deprecating, unsure, asking for an evaluation
UNSURE_PLAN = ("I'm a complete beginner with vegetable beds, so I'm not even sure tomatoes next to the fennel is "
               "a good idea … look over my planting plan first")
UNSURE_PLAN_READING = {"label": "negative", "emotions": {"confusion": 0.81, "annoyance": 0.11, "disapproval": 0.10},
                      "intent": _only("advice")}


@pytest.mark.parametrize("name,reading,text,after_trouble,key,cause", [
    ("unsure plan", UNSURE_PLAN_READING, UNSURE_PLAN, False, "patient", I.CAUSE_ASKED),
    ("build, instruction", {"label": "neutral", "emotions": {"curiosity": 0.6}, "intent": _only("instruction")},
     "can you check why the build fails?", False, "", None),                       # a question in form: no brief
    ("build, question", {"label": "neutral", "emotions": {"curiosity": 0.6}, "intent": _only("question")},
     "can you check why the build fails?", False, "", None),
    ("cron", {"label": "positive", "emotions": {"curiosity": 0.6}, "intent": _only("advice")},
     "what do you think about moving the cron to systemd?", False, "", I.CAUSE_ASKED),   # sure of itself: no patient
    ("hospital", {"label": "negative", "emotions": {"sadness": 0.8}, "intent": _only("confiding")},
     "my father is in hospital since yesterday", False, "acknowledge", None),   # the tone's own −0.08 books it
    ("sarcastic", {"label": "positive", "emotions": {}, "intent": _only("advice")},
     "great, you broke it again, what would you do now?", True, "", None),
    ("dashboard", {"label": "negative", "emotions": {"confusion": 0.8}, "intent": _only("venting")},
     "the dashboard is still empty after the update", False, "", None),
    # doubt refuses the EVENT; patience cannot offend, so a sarcastic but unsure owner still gets it
    ("sarcastic, unsure", {"label": "positive", "emotions": {"confusion": 0.7}, "intent": _only("advice")},
     "great, you broke it again, what would you do now?", True, "patient", None),
])
def test_realistic_readings(name, reading, text, after_trouble, key, cause):
    assert I.register(reading, audience="owner", text=text, after_trouble=after_trouble)[0] == key, name
    ev, why = I.mood_event(reading, text, after_trouble=after_trouble)
    assert (ev.cause if ev else None) == cause, (name, why)


async def test_the_unsure_plan_message_gets_patience_and_the_asked_event(svc, tmp_path, monkeypatch):
    """The case the feature exists for: negative 0.81 with confusion only (unbacked → no sentiment event)."""
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("negative", 0.81, intent=_only("advice"), emotions=UNSURE_PLAN_READING["emotions"])
    sm = _model(tmp_path)
    await C.score_message(UNSURE_PLAN, sm.root, self_model=sm, request_id="web-ve-plan", register_audience="owner")
    row = C.journal_rows(sm.root)[0]
    assert row["held"] == "below 0.85" and row["register"] == "patient"
    assert row["intent_effect"] == I.CAUSE_ASKED and row["intent_held"] == ""
    assert sm.mood().p == pytest.approx(I.EV_ASKED[0])


#: a reading + text per CLAUSE that offers it, with nothing doubtful about the message
CLAUSE_FIXTURES = {
    I.REG_URGENT: ({"urgent": 0.9}, "restart the server now, asap"),
    I.REG_PATIENT: ({"intent": _only("advice"), "emotions": {"confusion": 0.8}}, "what should I do about it?"),
    I.REG_ACKNOWLEDGE: ({"intent": _only("confiding"), "emotions": {"sadness": 0.8}}, "I miss Kosmas"),
    I.REG_JOKE: ({"intent": _only("banter")}, "haha you goof"),
    I.REG_BRIEF: ({"intent": _only("instruction")}, "make the chart for Elena bigger"),
}


def test_every_clause_is_either_doubt_free_or_refused_under_doubt():
    """Review r2 (coordinator): EVERY clause of CLAUSES is decided — listed doubt-free, or refused when the message
    is doubted (upbeat words about trouble right after the agent's own trouble). A clause added later without a
    fixture here fails; so does one whose code and declaration disagree (the urgent clause skipped the doubt
    without being listed)."""
    assert set(CLAUSE_FIXTURES) == set(I.CLAUSES)
    for clause, (reading, text) in CLAUSE_FIXTURES.items():
        key, _line = I.register(reading, audience="owner", text=text)
        assert clause in key.split("+"), (clause, key)                          # the control: it is offered
        doubted = {**reading, "label": "positive", "emotions": {**reading.get("emotions", {}), "joy": 0.8}}
        key, _line = I.register(doubted, audience="owner", text="great, you broke it again. " + text,
                                after_trouble=True)
        assert (clause in key.split("+")) is (clause in I.DOUBT_FREE_CLAUSES), (clause, key)
    # a tone label's clause never skips the doubt (its refusal is whether the feeling is there at all)
    tone_clauses = {I.REGISTER_OUTCOMES[lab] for lab, side in I.INTENT_TONE.items() if side}
    assert not (I.DOUBT_FREE_CLAUSES & tone_clauses) and I.REG_JOKE not in I.DOUBT_FREE_CLAUSES


def test_banter_right_after_trouble_is_doubted():
    # catches: a joke right after the agent's own trouble read as banter (it may be sarcasm)
    ev, why = I.mood_event({"intent": _only("banter")}, "haha nice one", after_trouble=True)
    assert ev is None and why == C.HELD_SARCASM


def test_an_unclear_or_absent_reading_moves_nothing():
    assert I.mood_event({}, "haha")[1] == I.WHY_NO_READING
    assert I.mood_event({"intent": {"banter": 0.5, "advice": 0.4}}, "haha")[1] == I.WHY_UNCLEAR
    for lab in ("instruction", "question", "correcting", "greeting"):
        assert I.mood_event({"intent": _only(lab)}, "please do it?") == (None, I.WHY_NO_EFFECT)
    for lab in ("reply", "thanks"):                         # never taken: no reading at all
        assert I.mood_event({"intent": _only(lab, 1.0)}, "please do it?") == (None, I.WHY_UNCLEAR)


@pytest.mark.parametrize("reading,text,kw,key", [
    ({"intent": _only("advice"), "emotions": {"confusion": 0.8}}, "what should I do?", {}, "patient"),
    ({"intent": _only("advice")}, "what should I do?", {}, ""),                       # no confusion
    ({"intent": _only("instruction")}, "make the chart for Elena bigger", {}, "brief"),
    ({"intent": _only("instruction")}, "can you make the chart bigger?", {}, ""),    # a question in form
    ({"intent": _only("venting"), "emotions": {"sadness": 0.7}}, "this week was awful", {}, "acknowledge"),
    ({"intent": _only("venting")}, "this week was awful", {}, ""),
    ({"intent": _only("confiding"), "label": "negative"}, "I miss Kosmas", {}, "acknowledge"),
    ({"intent": _only("confiding")}, "I miss Kosmas", {}, ""),
    ({"intent": _only("banter")}, "haha you goof", {}, "joke"),
    ({"intent": _only("banter")}, "you goof", {}, ""),
    ({"intent": _only("banter")}, "haha you goof", {"humour": -1}, ""),              # the character wins
    ({"intent": _only("banter")}, "haha you goof", {"humour": 0}, "joke"),
    ({"intent": _only("banter")}, "haha you goof", {"humour": -2, "audience": "others"}, "joke"),
    ({"intent": _only("banter")}, "haha you goof", {"after_trouble": True}, ""),
    # brief is the weakest clause (form only): never said beside another one
    ({"intent": _only("instruction"), "urgent": 0.9}, "restart the server now, asap", {}, "urgent"),
    ({"intent": _only("instruction"), "urgent": 0.9}, "restart the server", {}, "brief"),  # no hurry witness
    ({"urgent": 0.9, "emotions": {"nervousness": 0.8}}, "the server is down", {}, "urgent"),
    ({"urgent": 0.9}, "quick question about the chart", {}, ""),          # "quick" is not hurry
    ({"urgent": 0.9}, "explain quicksort", {}, ""),
    ({"intent": _only("question")}, "what time is it?", {}, ""),
    # an instruction with distress behind it is not to be briefed
    ({"intent": _only("instruction"), "emotions": {"sadness": 0.8}}, "move Kosmas's photos to the archive", {}, ""),
    # venting in a task's wording gets no "acknowledge"
    ({"intent": _only("venting"), "emotions": {"annoyance": 0.8}}, "delete the failed jobs", {}, ""),
    # with an emotion reading, a negative label alone is no unhappiness to acknowledge
    ({"intent": _only("venting"), "label": "neutral", "emotions": {"confusion": 0.8}}, "the page is empty", {}, ""),
    ({"intent": _only("banter")}, "haha you goof", {"audience": "public"}, "joke"),
    ({"intent": _only("banter"), "label": "negative"}, "haha this is awful", {}, ""),
])
def test_each_register_clause_needs_its_witness(reading, text, kw, key):
    args = {"audience": "owner", "text": text, **kw}
    got, line = I.register(reading, **args)
    assert got == key
    assert (line != "") is bool(key)
    if line:
        assert line.startswith(I.REGISTER_HEAD) and line.endswith("\n\n")
        assert "%" not in line and "intent" not in line.lower()      # decision 9: no numbers, no classifier


@pytest.mark.parametrize("humour,offered", [
    (None, True), (0, True), (2, True), (1, True), (-1, False), (-2, False),
    (float("nan"), False), (True, False), ("1", False), (1.0, False), (3, False), (-3, False)])
def test_the_joke_needs_a_real_humour_dial_or_none(humour, offered):
    # review r2: NaN / True / a string read as "fine, joke" — anything not an int in −2..2 (or None, no character)
    # withholds the joke; members and public never read the dial
    reading = {"intent": _only("banter")}
    assert (I.register(reading, audience="owner", humour=humour, text="haha you goof")[0] == "joke") is offered
    assert I.register(reading, audience="others", humour=humour, text="haha you goof")[0] == "joke"


def test_a_clause_taken_off_the_doubt_free_list_is_refused_under_doubt(monkeypatch):
    # review r2: the urgent clause's doubt runs through the same helper — take it off the list and it is refused
    doubted = {"urgent": 0.9, "label": "positive", "emotions": {"joy": 0.8}}
    text = "great, you broke it again. restart it now, asap"
    assert I.register(doubted, audience="owner", text=text, after_trouble=True)[0] == "urgent"
    monkeypatch.setattr(I, "DOUBT_FREE_CLAUSES", I.DOUBT_FREE_CLAUSES - {I.REG_URGENT})
    assert I.register(doubted, audience="owner", text=text, after_trouble=True) == ("", "")
    assert I.register({"urgent": 0.9}, audience="owner", text="restart it now, asap")[0] == "urgent"


def test_the_register_line_keeps_the_work_and_says_no_number_or_classifier():
    """Decision 9: the line limits PROSE only. Properties, not the bytes: it says the facts, checks and tool use
    stay the same, it never tells the model to skip or shorten them, and carries no number or classifier name."""
    import re
    for reading, text in (({"intent": _only("instruction"), "urgent": 0.9}, "restart it now, asap"),
                          ({"intent": _only("banter")}, "haha you goof"),
                          ({"intent": _only("venting"), "emotions": {"sadness": 0.9}}, "awful week")):
        _k, line = I.register(reading, audience="owner", text=text)
        low = line.lower()
        assert re.search(r"\bthe same facts, checks, tool use\b", low), line
        assert not re.search(r"\b(skip|omit|drop|fewer|less|without|instead of)\b", low), line
        assert not re.search(r"[0-9%]", line), line
        assert not re.search(r"\b(classifier|model|intent|sentiment|emotion|probab\w*|score|mood of)\b", low), line


def test_no_line_for_an_audience_it_does_not_know():
    assert I.register({"intent": _only("banter")}, audience="slack", text="haha") == ("", "")


def test_says_correcting():
    # only nominates to the correction judge: correcting as the TOP label at its floor (0.90)
    assert I.says_correcting({"intent": _only("correcting")})
    assert I.says_correcting({"intent": {"correcting": 0.90, "instruction": 0.5}})
    assert not I.says_correcting({"intent": {"correcting": 0.89}})
    assert not I.says_correcting({"intent": {"correcting": 0.95, "instruction": 0.97}})   # high, but not the top


# ── the class: every label has a decided outcome ────────────────────────────

def test_every_label_has_a_decided_outcome_in_the_mood_and_the_register():
    # catches: a label added to LABELS (or renamed by the measurement) with no decision on what it does
    assert set(I.MOOD_OUTCOMES) == set(I.LABELS) and set(I.REGISTER_OUTCOMES) == set(I.LABELS)
    for lab in I.LABELS:
        ev, why = I.mood_event({"intent": _only(lab), "emotions": {"sadness": 0.9, "curiosity": 0.9}},
                               "I wonder, haha, what you think?")
        assert why != I.WHY_UNDECIDED, lab
        assert I.REGISTER_OUTCOMES[lab] in (None,) + tuple(I.CLAUSES), lab


def test_an_undecided_label_is_said_as_undecided(monkeypatch):
    # the enumeration above can fail: a label with no entry is reported, not silently treated as "nothing"
    monkeypatch.setattr(I, "LABELS", I.LABELS + ("gossip",))
    monkeypatch.setitem(I.LABEL_MIN, "gossip", 0.5)
    assert I.mood_event({"intent": {"gossip": 0.9}}, "x")[1] == I.WHY_UNDECIDED
    # … and one with no floor decided is never taken (``test_every_label_has_a_floor_or_is_never_taken``)
    monkeypatch.delitem(I.LABEL_MIN, "gossip")
    assert I.top({"intent": {"gossip": 0.9}}) == ("", 0.0)


def test_the_decisions_per_label_are_these():
    """Every label's decided outcome, listed — greeting and reply explicitly as "decided: none". Catches a
    label added, renamed or re-decided without this table being looked at."""
    assert I.MOOD_OUTCOMES == {
        "advice": I.CAUSE_ASKED, "confiding": I.CAUSE_CONFIDED, "banter": I.CAUSE_BANTER, "venting": I.CAUSE_VENTED,
        "instruction": None, "question": None, "correcting": None, "thanks": None,
        "greeting": None, "reply": None}                   # decided: none
    assert I.REGISTER_OUTCOMES == {
        "advice": I.REG_PATIENT, "instruction": I.REG_BRIEF, "confiding": I.REG_ACKNOWLEDGE,
        "venting": I.REG_ACKNOWLEDGE, "banter": I.REG_JOKE, "question": None, "correcting": None, "thanks": None,
        "greeting": None, "reply": None}                   # decided: none
    assert I.INTENT_TONE == {
        "banter": "positive", "venting": "negative", "advice": None, "confiding": None, "instruction": None,
        "question": None, "correcting": None, "thanks": None, "greeting": None, "reply": None}


# ── score_message: journal mode, mode 0, mode 1 ─────────────────────────────

async def test_journal_mode_moves_nothing_and_says_what_it_would_have_done(svc, tmp_path, monkeypatch):
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    calls = _logged(monkeypatch)
    assert await C.score_message("the whole week at work has been awful", sm.root, self_model=sm,
                                 register_audience="owner") is True
    # catches: journal mode booking the event (or a held row) anywhere in the mood
    assert (sm.mood().p, sm.mood().e) == (0.0, 0.0) and _events(sm) == []
    (row,) = C.journal_rows(sm.root)
    assert row["intent"] == VENT and row["nudge"] == 0.0 and row["held"] == "neutral"
    assert (row["intent_effect"], row["intent_held"], row["register"]) == (I.CAUSE_VENTED, C.INTENT_HELD_JOURNAL,
                                                                          "acknowledge")
    (_t, line, icon), = calls
    assert icon == "💗" and line == ("neutral 90% (negative 5%, positive 5%) · emotion: sadness 80% · intent: "
                                     "venting 93% · pitch: acknowledge (journal only) · mood unchanged; intent "
                                     "would nudge down (−0.05 before caps, journal only)")
    assert "awful" not in line                                            # never the words


async def test_mode_0_rows_and_log_lines_are_as_before(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "0")
    svc.body = _body("positive", 0.85, intent=_only("banter"), urgent=0.9)
    calls = _logged(monkeypatch)
    await C.score_message("Markos loved the new garden", tmp_path)
    (row,) = C.journal_rows(tmp_path)
    # catches: a new key on the row or the line with the switch off
    assert set(row) == OLD_ROW_KEYS
    assert calls == [("Sentiment", "positive 85% (negative 5%, neutral 5%) · mood unchanged: no mood to move",
                      "💗")]


async def test_the_row_keeps_every_score_the_service_sent(svc, tmp_path):
    # all ten scores stay on the row (the floors are re-fitted from the journal) — none filtered or cut
    full = {"instruction": 0.0312, "question": 0.4471, "advice": 0.8912, "greeting": 0.0021, "reply": 0.1203,
            "thanks": 0.0005, "correcting": 0.0731, "banter": 0.0044, "confiding": 0.2518, "venting": 0.0102}
    svc.body = _body("neutral", 0.9, intent=full, urgent=0.0213)
    await C.score_message("what would you do about Markos's garden?", tmp_path)
    row = C.journal_rows(tmp_path)[0]
    assert row["intent"] == full and row["urgent"] == 0.0213
    # … and a partial set is kept as it came
    svc.body = _body("neutral", 0.9, intent={"advice": 0.91, "question": 0.2})
    await C.score_message("what would you do?", tmp_path)
    assert C.journal_rows(tmp_path)[1]["intent"] == {"advice": 0.91, "question": 0.2}


#: an intent row's keys: today's, the answer's own intent fields, and what the intent did / would have done
INTENT_ROW_KEYS = OLD_ROW_KEYS | {"intent", "urgent", "intent_effect", "intent_held", "register", "intent_judge"}


async def test_an_intent_rows_shape(svc, tmp_path):
    svc.body = _body("neutral", 0.9, intent=_only("advice"), urgent=0.2)
    await C.score_message("what would you do?", tmp_path)
    assert set(C.journal_rows(tmp_path)[0]) == INTENT_ROW_KEYS      # catches: a key added or lost unannounced


@pytest.mark.parametrize("env,want", [
    ({}, C.INTENT_HELD_JOURNAL),                                      # journal mode: what it would have done
    ({"GHOST_INTENT": "1"}, C.INTENT_JUDGE_NOMINATED),
    ({"GHOST_INTENT": "1", "GHOST_INTENT_JUDGE": "0"}, C.INTENT_JUDGE_OFF),
    ({"GHOST_INTENT_JUDGE": "0"}, C.INTENT_JUDGE_OFF),
])
async def test_the_row_says_whether_the_correction_judge_was_nominated(svc, tmp_path, monkeypatch, env, want):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    svc.body = _body("negative", 0.7, intent=_only("correcting"))
    await C.score_message("that is not the file I meant", tmp_path)
    svc.body = _body("negative", 0.7, intent=_only("instruction"))
    await C.score_message("open the other file", tmp_path)
    first, second = C.journal_rows(tmp_path)
    # catches: the nomination never written, written for every row, or ignoring the mode and the switch
    assert first["intent_judge"] == want and second["intent_judge"] == ""


async def test_a_row_without_an_intent_reading_is_as_before_in_journal_mode(svc, tmp_path):
    svc.body = _body("positive", 0.85)
    await C.score_message("Markos loved the new garden", tmp_path)
    assert set(C.journal_rows(tmp_path)[0]) == OLD_ROW_KEYS


async def test_mode_1_books_the_intent_as_the_messages_one_tone(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    await C.score_message("the whole week at work has been awful", sm.root, self_model=sm, request_id="web-ve1")
    v = sm.mood()
    assert (v.p, v.e) == (pytest.approx(I.EV_VENTING[0]), pytest.approx(I.EV_VENTING[1]))
    (row,) = C.journal_rows(sm.root)
    assert row["nudge"] == pytest.approx(-0.05) and row["intent_held"] == "" and row["held"] == "neutral"
    (ev,) = _events(sm)
    assert ev["kind"] == M.KIND_SENTIMENT and ev["cause"] == I.CAUSE_VENTED
    # … a TONE: members never hear it (the private share, not the pair members read)
    assert sm.state.private_sides()[2] == pytest.approx(0.05)
    assert C.BREAKER.moved == 1


async def test_mode_1_with_the_mood_switch_off_moves_nothing(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "1")
    monkeypatch.setenv("GHOST_INTENT_MOOD", "0")
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    await C.score_message("the whole week has been awful", sm.root, self_model=sm)
    assert sm.mood().p == 0.0 and _events(sm) == []
    assert C.journal_rows(sm.root)[0]["intent_held"] == C.INTENT_HELD_OFF


async def test_greek_keeps_its_intent_and_can_move_the_mood(svc, tmp_path, monkeypatch):
    # catches: the early return for an unknown tone running before the intent (all of Greek lost)
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("unknown", 0.0, intent=_only("advice"), reason="not English")
    sm = _model(tmp_path)
    calls = _logged(monkeypatch)
    await C.score_message("τι λες, να το κάνω;", sm.root, self_model=sm, request_id="web-ve-gr")
    assert sm.mood().p == pytest.approx(I.EV_ASKED[0])
    (row,) = C.journal_rows(sm.root)
    assert row["label"] == "unknown" and row["held"] == "not English" and row["intent_effect"] == I.CAUSE_ASKED
    (_t, line, _i), = calls
    assert line == "not rated: not English · intent: advice 95% · mood nudged up +0.02 (asked for my opinion)"


async def test_one_verdict_praise_and_banter(svc, tmp_path, monkeypatch):
    # "haha thanks, perfect": the praise check counted it — the banter is not a second lift
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("neutral", 0.6, intent=_only("banter"))
    sm = _model(tmp_path)
    sm.note_owner_reaction(M.KIND_PRAISE, request_id="web-ve-p")
    await C.score_message("haha thanks, perfect", sm.root, self_model=sm, praised=True, request_id="web-ve-p")
    assert sm.mood().p == pytest.approx(M.EV_PRAISE[0])
    assert C.journal_rows(sm.root)[0]["intent_held"] == C.HELD_PRAISED


async def test_one_verdict_sentiment_and_venting(svc, tmp_path, monkeypatch):
    # catches: the venting booked beside the unhappy tone of the same message (−0.13 instead of −0.08)
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("negative", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    await C.score_message("my flight got cancelled again", sm.root, self_model=sm, request_id="web-ve-s")
    assert sm.mood().p == pytest.approx(-0.08)
    row = C.journal_rows(sm.root)[0]
    assert row["intent_held"] == C.INTENT_HELD_TONE and row["held"] == ""
    assert [e["cause"] for e in _events(sm)] == ["the owner's message sounded unhappy"]


async def test_one_verdict_correction_and_intent(svc, tmp_path, monkeypatch):
    # catches: the intent's tone and the correction of the same reply counting twice
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    await C.score_message("this whole week has been awful", sm.root, self_model=sm, request_id="web-ve-c")
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="web-ve-c")
    assert sm.mood().p == pytest.approx(M.EV_CORRECTION[0])


async def test_a_doubted_tone_under_its_gate_silences_advice(svc, tmp_path, monkeypatch):
    """Review r1 MAJOR 1 (probe2 case 2): after the agent's trouble, "great, you broke it again, what would you
    do now?" reads positive 0.7 — under the tone's gate, so no sentiment event and no sentiment doubt — and
    advice has no doubt of its own. Only the shared refusal decides here."""
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("positive", 0.7, intent=_only("advice"))
    sm = _model(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="web-ve-prev")      # trouble the owner saw
    before = sm.mood().p
    await C.score_message("great, you broke it again, what would you do now?", sm.root, self_model=sm)
    assert sm.mood().p == before
    row = C.journal_rows(sm.root)[0]
    assert row["held"] == "below 0.85" and row["intent_held"] == C.HELD_SARCASM and row["intent_effect"] == ""


async def test_an_unbacked_unhappy_tone_is_not_vented(svc, tmp_path, monkeypatch):
    # review r1 MAJOR 2 (probe2 case 1): the tone gate refused it as unbacked; venting must not re-admit it
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("negative", 0.95, intent=VENT, emotions={"confusion": 0.8})
    sm = _model(tmp_path)
    await C.score_message("the dashboard is still empty after the update", sm.root, self_model=sm)
    assert sm.mood().p == 0.0 and _events(sm) == []
    row = C.journal_rows(sm.root)[0]
    assert row["held"] == C.HELD_UNBACKED and row["intent_held"] == C.HELD_UNBACKED


async def test_the_intent_reason_says_what_happened_to_the_tone_it_deferred_to(svc, tmp_path, monkeypatch):
    # review r1 MINOR 6a: the unhappy tone was HELD by the owner's hold — "already counted" was false
    import datetime as dt
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("negative", 0.95, intent=VENT, emotions={"sadness": 0.9})
    sm = _model(tmp_path)
    sm.state.set_owner_mood("calm", M.resolve_label("calm")[1], 1,
                            now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5))
    await C.score_message("my flight got cancelled", sm.root, self_model=sm)
    row = C.journal_rows(sm.root)[0]
    assert row["held"] == SelfStateThread.HELD_OWNER_HOLD and row["intent_held"] == SelfStateThread.HELD_OWNER_HOLD
    # … and without a mood to move at all
    await C.score_message("my flight got cancelled", tmp_path / "bare", self_model=None)
    assert C.journal_rows(tmp_path / "bare")[0]["intent_held"] == C.HELD_NO_MOOD


# ── the row's register key says what the reply would really hear ────────────

async def _journal_register(svc, tmp_path, body, text, **kw):
    svc.body = body
    sm = kw.pop("sm", None) or _model(tmp_path)
    kw.setdefault("register_audience", "owner")          # the owner's own reply, as the request path passes it
    await C.score_message(text, sm.root, self_model=sm, **kw)
    return C.journal_rows(sm.root)[-1]["register"]


async def test_the_rows_register_follows_the_switch_audience_humour_and_trouble(svc, tmp_path, monkeypatch):
    banter = _body("neutral", 0.7, intent=_only("banter"))
    assert await _journal_register(svc, tmp_path / "a", banter, "haha you goof") == "joke"
    # catches: the row ignoring GHOST_INTENT_REGISTER=0
    monkeypatch.setenv("GHOST_INTENT_REGISTER", "0")
    assert await _journal_register(svc, tmp_path / "b", banter, "haha you goof") == ""
    monkeypatch.delenv("GHOST_INTENT_REGISTER")
    # catches: the row computed with after_trouble=False
    sm = _model(tmp_path / "c")
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="web-ve-prev")
    assert await _journal_register(svc, tmp_path / "c", banter, "haha you goof", sm=sm) == ""
    # catches: the row ignoring the owner's humour dial (read as today by default)
    from ghost_agent.selfhood import character as CH
    monkeypatch.setattr(CH, "dials", lambda home: {"warmth": 0, "humour": -1, "directness": 0})
    assert await _journal_register(svc, tmp_path / "d", banter, "haha you goof") == ""
    # … the caller says what the reply hears (catches: the audience fixed to "owner"): a public reply ignores
    # the dial (still −1 here); no character heard (None) offers the joke; an audience it does not know, nothing
    assert await _journal_register(svc, tmp_path / "e", banter, "haha you goof", register_audience="public") == "joke"
    assert await _journal_register(svc, tmp_path / "g", banter, "haha you goof", register_humour=None) == "joke"
    assert await _journal_register(svc, tmp_path / "f", banter, "haha you goof", register_audience="slack",
                                   register_humour=None) == ""
    # review r2: "" = this reply hears no register (the owner on a public surface with the members switch off)
    assert await _journal_register(svc, tmp_path / "h", banter, "haha you goof", register_audience="",
                                   register_humour=None) == ""


async def test_banter_after_the_agents_trouble_is_held(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("neutral", 0.7, intent=_only("banter"))
    sm = _model(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="web-ve-prev")      # trouble the owner saw
    before = sm.mood().p
    await C.score_message("haha nice one", sm.root, self_model=sm)
    assert sm.mood().p == before and C.journal_rows(sm.root)[0]["intent_held"] == C.HELD_SARCASM


async def test_the_owners_hold_holds_the_intent(svc, tmp_path, monkeypatch):
    import datetime as dt
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    sm.state.set_owner_mood("calm", M.resolve_label("calm")[1], 1,
                            now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5))
    before = sm.mood().p
    await C.score_message("this whole week has been awful", sm.root, self_model=sm)
    assert sm.mood().p == before
    assert C.journal_rows(sm.root)[0]["intent_held"] == SelfStateThread.HELD_OWNER_HOLD


async def test_a_message_set_before_the_owners_word_does_not_move_the_mood(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    sm = _model(tmp_path)
    sm.state.nudge(-0.5, 0.2, "trouble", M.KIND_TURN)
    real = C.classify

    async def _slow(text, **kw):
        sm.state.reset_mood()                       # the owner says "neutral" while the reading is on its way
        return await real(text, **kw)
    monkeypatch.setattr(C, "classify", _slow)
    await C.score_message("this whole week has been awful", sm.root, self_model=sm)
    assert sm.mood().neutral
    assert C.journal_rows(sm.root)[0]["intent_held"] == SelfStateThread.HELD_OWNER_LATER


def test_only_a_tone_is_taken_as_the_messages_event(tmp_path):
    # catches: a non-tone event handed to the tone path moving the pair members hear
    sm = _model(tmp_path)
    after, dp, why = sm.note_owner_sentiment("neutral", 0.0, request_id="r1",
                                             event=M.Event(0.3, 0.0, "x", M.KIND_THUMB_UP))
    assert (after, dp, why) == (None, 0.0, sm.HELD_NOT_A_TONE) and sm.mood().p == 0.0


# ── the readings store ──────────────────────────────────────────────────────

def test_reading_for_keeps_its_default_and_gives_the_whole_reading_on_request():
    async def go():
        C.expect_reading("web-ve-r")
        C._note_reading("web-ve-r", {"sadness": 0.8},
                        reading={"label": "neutral", "emotions": {"sadness": 0.8}, "intent": VENT, "urgent": 0.2,
                                 "probs": {"neutral": 0.9}})
        # catches: the default changed from the bare emotions dict (the judge's gate reads it)
        assert await C.reading_for("web-ve-r") == {"sadness": 0.8}
        assert await C.reading_for("web-ve-r", part="all") == {
            "label": "neutral", "emotions": {"sadness": 0.8}, "intent": VENT, "urgent": 0.2}
        assert await C.reading_for("web-ve-r", part="words") is None
        C.expect_reading("web-ve-none")
        C._note_reading("web-ve-none", None)                       # scored, no answer
        assert await C.reading_for("web-ve-none", part="all") is None
    asyncio.run(go())


async def test_a_queued_message_keeps_the_requests_own_reading(svc, tmp_path, monkeypatch):
    # two messages under one request id: the slot keeps the first; each row carries its own intent
    monkeypatch.setenv("GHOST_INTENT", "1")
    sm = _model(tmp_path)
    C.expect_reading("web-ve-q")
    svc.body = _body("neutral", 0.9, intent=VENT, emotions={"sadness": 0.8})
    await C.score_message("this week has been awful", sm.root, self_model=sm, request_id="web-ve-q")
    svc.body = _body("neutral", 0.7, intent=_only("banter"))
    await C.score_message("haha anyway", sm.root, self_model=sm, request_id="web-ve-q")
    assert (await C.reading_for("web-ve-q", part="all"))["intent"] == VENT
    assert [r["intent_effect"] for r in C.journal_rows(sm.root)] == [I.CAUSE_VENTED, I.CAUSE_BANTER]


async def test_read_only_keeps_the_reading_and_writes_no_row_and_no_mood(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "1")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(home))         # where a journal of the agent's would land
    svc.body = _body("neutral", 0.9, intent=_only("instruction"))
    calls = _logged(monkeypatch)
    C.expect_reading("probe-ve-1")
    assert await C.read_only("make Elena's chart bigger", request_id="probe-ve-1") is True
    assert (await C.reading_for("probe-ve-1", part="all"))["intent"] == _only("instruction")
    # catches: the read-only path journalling the words anywhere under the agent's home (or here)
    for root in (home, tmp_path):
        for f in root.rglob("*"):
            if f.is_file():
                assert "chart bigger" not in f.read_text(errors="replace"), f
    (_t, line, _i), = calls
    assert line == "read for the reply only: neutral · intent: instruction · nothing kept"


async def test_read_only_asks_nothing_unless_the_register_acts(svc, monkeypatch):
    # journal mode (the default): the service is not asked, and a waiter is not left waiting — the slot is
    # filled (empty) AT ONCE, not left to time out (catches: the early exit forgetting `_note_reading`)
    C.expect_reading("probe-ve-2")
    assert await C.read_only("make the chart bigger", request_id="probe-ve-2") is False
    assert svc.seen == []
    assert await asyncio.wait_for(C.reading_for("probe-ve-2", wait_s=30, part="all"), 1.0) is None


async def test_forgetting_a_read_only_reading_at_the_end_of_its_request_is_safe(svc, monkeypatch):
    # review r1 MINOR 7: the request path forgets the slot when the member's / probe's request ends; a reading
    # that lands after that is not kept, and a waiter returns at once
    C.expect_reading("probe-ve-3")
    C.forget_reading("probe-ve-3")
    C._note_reading("probe-ve-3", {"joy": 0.9}, reading={"label": "positive", "intent": VENT})
    assert "probe-ve-3" not in C._READINGS
    assert await asyncio.wait_for(C.reading_for("probe-ve-3", wait_s=30, part="all"), 1.0) is None


async def test_read_only_never_sends_a_members_words_with_the_members_switch_off(svc, monkeypatch):
    # catches: a member's words reaching the service against the written rule (operator decision 3)
    monkeypatch.setenv("GHOST_INTENT", "1")
    import ghost_agent.utils.logging as GL
    # review r2: a member's PROBE is kind "probe" — the member test must still hold it
    monkeypatch.setattr(GL, "request_kind", lambda rid=None: "probe")
    monkeypatch.setattr(GL, "requester_is_member", lambda: True)
    assert await C.read_only("haha nice", request_id="probe-ve-m") is False and svc.seen == []
    monkeypatch.setattr(GL, "requester_is_member", lambda: False)
    monkeypatch.setattr(GL, "request_kind", lambda rid=None: "member")
    assert await C.read_only("haha nice", request_id="web-ve-m") is False and svc.seen == []
    monkeypatch.setenv("GHOST_INTENT_MEMBERS", "1")
    C.expect_reading("web-ve-m")
    assert await C.read_only("haha nice", request_id="web-ve-m") is True and svc.seen == ["haha nice"]


# ── the health part and the log line ────────────────────────────────────────

async def test_health_counts_intent_readings_only_once_one_was_seen(svc, tmp_path):
    svc.body = _body("positive", 0.85)
    await C.score_message("Markos loved the garden", tmp_path)
    assert C.health_text() == "sentiment service: 1 scored since start, 0 moved the mood"
    svc.body = _body("positive", 0.85, intent=_only("thanks"))
    await C.score_message("Markos loved the garden", tmp_path)
    assert C.health_text() == "sentiment service: 2 scored since start, 0 moved the mood, 1 with an intent reading"


def test_the_rating_line_says_both_readings():
    # the tone moved the mood; the venting deferred to it — the line says nothing of the intent's mood effect
    answer = {"label": "negative", "score": 0.9, "probs": {"negative": 0.9}, "emotions": {"sadness": 0.8},
              "intent": VENT, "urgent": 0.8}
    line = C.rating_line(answer, -0.08, "", intent_effect=I.CAUSE_VENTED, intent_held=C.INTENT_HELD_TONE)
    assert line == "negative 90% · emotion: sadness 80% · intent: venting 93%, urgent 80% · mood nudged down −0.08"
    unclear = {"label": "neutral", "score": 0.9, "probs": {"neutral": 0.9}, "intent": {"advice": 0.5, "question": 0.4}}
    assert " · intent: unclear (advice 50% is under the 90% it needs; next question 40%) · " in C.rating_line(
        unclear, 0.0, "neutral")


# ── one message, one story: the journal's register key = the request path's ─────────────────────────────

@pytest.mark.parametrize("mode", ["journal", "1"])
@pytest.mark.parametrize("name,body,text,trouble", [
    ("banter, no emotion model", _body("positive", 0.95, intent=_only("banter")), "haha you goof", False),
    ("banter with emotions", _body("positive", 0.95, intent=_only("banter"), emotions={"amusement": 0.8}),
     "haha you goof", False),
    ("venting, no emotion model", _body("negative", 0.95, intent=_only("venting")), "this whole week was awful",
     False),
    ("venting, unbacked", _body("negative", 0.95, intent=_only("venting"), emotions={"confusion": 0.8}),
     "this whole week was awful", False),
    ("unsure advice", _body("negative", 0.81, intent=_only("advice"), emotions={"confusion": 0.81}),
     UNSURE_PLAN, False),
    ("urgent instruction", _body("neutral", 0.9, intent=_only("instruction"), urgent=0.9),
     "restart the server now, asap", False),
    ("unknown tone", _body("unknown", 0.0, intent=_only("banter"), reason="not English"), "haha you goof", False),
    ("banter after trouble", _body("neutral", 0.7, intent=_only("banter")), "haha you goof", True),
])
@pytest.mark.parametrize("audience,humour", [("owner", None), ("owner", -1), ("public", None), ("", None)])
async def test_the_rows_register_is_what_the_request_path_computes_from_the_stored_reading(
        svc, tmp_path, monkeypatch, mode, name, body, text, trouble, audience, humour):
    """Review r2 MAJOR 1 (the class): the journal row's ``register`` key and the key the request path gets by
    calling ``intent.register`` on ``reading_for(rid, part="all")`` with the same audience, humour, text and
    trouble must agree for every answer shape — the store once turned "no emotion model" into "nothing backs
    the tone", so the row said "joke" and the reply heard nothing."""
    monkeypatch.setenv("GHOST_INTENT", mode)
    svc.body = body
    sm = _model(tmp_path)
    if trouble:
        sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="web-ve-prev")
    rid = f"web-ve-one-{abs(hash((name, mode, audience, humour))) % 10 ** 8}"
    C.expect_reading(rid)
    await C.score_message(text, sm.root, self_model=sm, request_id=rid, register_audience=audience,
                          register_humour=humour)
    row = C.journal_rows(sm.root)[-1]
    stored = await C.reading_for(rid, part="all")
    path_key, _line = I.register(stored, audience=audience, humour=humour, text=C.prepare_text(text),
                                 after_trouble=sm.recent_trouble())
    assert row["register"] == path_key, (name, row["register"], path_key)
    # … and the stored reading is the parsed answer's shape (no "emotions" key when the answer had none)
    assert ("emotions" in stored) is ("emotions" in body), name


async def test_mode_0_stores_no_intent_for_the_request_path(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "0")
    svc.body = _body("positive", 0.95, intent=_only("banter"))
    C.expect_reading("web-ve-m0")
    await C.score_message("haha you goof", tmp_path, request_id="web-ve-m0")
    stored = await C.reading_for("web-ve-m0", part="all")
    assert "intent" not in (stored or {})


# ── one 💗 line per message: tone · emotion · intent · pitch · mood · judge (operator 11-20) ─────────────

def _plan_body(intent=None):
    b = _body("negative", 0.81, intent=intent if intent is not None else _only("advice", 0.97),
              emotions={"confusion": 0.81})
    b["probs"] = {"negative": 0.81, "neutral": 0.18, "positive": 0.02}
    return b


_SAD = {"sadness": 0.8}
_HOLD = "hold"           # a row that sets the owner's hold first
_THUMB = "thumb"         # … a row with trouble the owner saw just before
_NO_MODEL = "no model"

#: (name, env, body, text, extra) → the exact line. ``extra``: score_message kwargs, or a setup tag
LINE_TABLE = [
    ("approved: mode 1, intent nudged", {"GHOST_INTENT": "1"}, _plan_body(), UNSURE_PLAN, {},
     "negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · intent: advice 97% · pitch: patient · "
     "mood nudged up +0.02 (asked for my opinion)"),
    ("approved: journal", {}, _plan_body(), UNSURE_PLAN, {},
     "negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · intent: advice 97% · pitch: patient "
     "(journal only) · mood unchanged; intent would nudge up (+0.02 before caps, journal only)"),
    ("tone nudged, the venting deferred (1)", {"GHOST_INTENT": "1"}, _body("negative", 0.9, intent=VENT, emotions=_SAD),
     "my flight got cancelled again", {},
     "negative 90% (neutral 5%, positive 5%) · emotion: sadness 80% · intent: venting 93% · pitch: acknowledge · "
     "mood nudged down −0.08"),
    ("tone nudged, the venting deferred (journal)", {}, _body("negative", 0.9, intent=VENT, emotions=_SAD),
     "my flight got cancelled again", {},
     "negative 90% (neutral 5%, positive 5%) · emotion: sadness 80% · intent: venting 93% · pitch: acknowledge "
     "(journal only) · mood nudged down −0.08"),
    ("tone held as praise, banter too", {"GHOST_INTENT": "1"},
     _body("positive", 0.95, intent=_only("banter"), emotions={"joy": 0.9}), "haha thanks, perfect",
     {"praised": True},
     "positive 95% (negative 5%, neutral 5%) · emotion: joy 90% (counts as warm) · intent: banter 95% · pitch: joke · "
     "mood unchanged: the praise check counted it"),
    ("neutral tone, banter held as praise", {"GHOST_INTENT": "1"}, _body("neutral", 0.6, intent=_only("banter")),
     "haha thanks, perfect", {"praised": True},
     "neutral 60% (negative 5%, positive 5%) · intent: banter 95% · pitch: joke · mood unchanged: a neutral "
     "message; intent held: the praise check counted it"),
    ("sarcasm under the tone's gate", {"GHOST_INTENT": "1"}, _body("positive", 0.7, intent=_only("advice")),
     "great, you broke it again, what would you do now?", _THUMB,
     "positive 70% (negative 5%, neutral 5%) · intent: advice 95% · mood unchanged: a nudge needs 85% or more; "
     "intent held: upbeat words about trouble (it may be sarcasm)"),
    ("unbacked tone, the intent acted", {"GHOST_INTENT": "1"}, _plan_body(), UNSURE_PLAN, {},
     "negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · intent: advice 97% · pitch: patient · "
     "mood nudged up +0.02 (asked for my opinion)"),
    ("unbacked tone, the venting refused for it", {"GHOST_INTENT": "1"},
     _body("negative", 0.95, intent=VENT, emotions={"confusion": 0.8}), "the dashboard is still empty", {},
     "negative 95% (neutral 5%, positive 5%) · emotion: confusion 80% · intent: venting 93% · mood unchanged: no "
     "emotion backs the tone (politeness, not a feeling)"),
    ("not rated, the intent acted", {"GHOST_INTENT": "1"},
     _body("unknown", 0.0, intent=_only("advice"), reason="not English"), "what would you plant here?", {},
     "not rated: not English · intent: advice 95% · mood nudged up +0.02 (asked for my opinion)"),
    ("not rated, journal", {}, _body("unknown", 0.0, intent=_only("advice"), reason="not English"),
     "what would you plant here?", {},
     "not rated: not English · intent: advice 95% · mood unchanged; intent would nudge up (+0.02 before caps, "
     "journal only)"),
    ("the owner's hold", {"GHOST_INTENT": "1"}, _body("neutral", 0.9, intent=VENT, emotions=_SAD),
     "the whole week at work has been awful", _HOLD,
     "neutral 90% (negative 5%, positive 5%) · emotion: sadness 80% · intent: venting 93% · pitch: acknowledge · "
     "mood unchanged: a neutral message; intent held: " + SelfStateThread.HELD_OWNER_HOLD),
    ("no mood to move", {"GHOST_INTENT": "1"}, _body("neutral", 0.9, intent=VENT, emotions=_SAD),
     "the whole week at work has been awful", _NO_MODEL,
     "neutral 90% (negative 5%, positive 5%) · emotion: sadness 80% · intent: venting 93% · pitch: acknowledge · "
     "mood unchanged: a neutral message; intent held: no mood to move"),
    ("no witness", {"GHOST_INTENT": "1"}, _body("neutral", 0.9, intent=_only("confiding")),
     "send the garden report to Elena", {},
     "neutral 90% (negative 5%, positive 5%) · intent: confiding 95% · mood unchanged: a neutral message; intent "
     "held: nothing else in the message backs the intent"),
    ("the mood switch off", {"GHOST_INTENT": "1", "GHOST_INTENT_MOOD": "0"},
     _body("neutral", 0.9, intent=VENT, emotions=_SAD), "the whole week at work has been awful", {},
     "neutral 90% (negative 5%, positive 5%) · emotion: sadness 80% · intent: venting 93% · pitch: acknowledge · "
     "mood unchanged: a neutral message; intent held: the intent's mood effect is switched off"),
    ("instruction, urgent: no mood word from the intent", {"GHOST_INTENT": "1"},
     _body("neutral", 0.9, intent=_only("instruction", 0.8), urgent=0.9), "restart the server now, asap", {},
     "neutral 90% (negative 5%, positive 5%) · intent: instruction 80%, urgent 90% · pitch: urgent · mood "
     "unchanged: a neutral message"),
    ("urgency under its floor is not said", {"GHOST_INTENT": "1"},
     _body("neutral", 0.9, intent=_only("question"), urgent=0.5), "where is the garden report?", {},
     "neutral 90% (negative 5%, positive 5%) · intent: question 95% · mood unchanged: a neutral message"),
    ("unclear: under its floor", {"GHOST_INTENT": "1"},
     _body("neutral", 0.9, intent={**_only("advice", 0.89), "question": 0.4}), "what about the fennel", {},
     "neutral 90% (negative 5%, positive 5%) · intent: unclear (advice 89% is under the 90% it needs; next question "
     "40%) · mood unchanged: a neutral message"),
    ("unclear: never taken", {}, _body("neutral", 0.9, intent={**_only("reply"), "thanks": 0.6}), "yes, that one",
     {}, "neutral 90% (negative 5%, positive 5%) · intent: unclear (reply is never taken; next thanks 60%) · mood "
     "unchanged: a neutral message"),
    ("unclear: a tie", {}, _body("neutral", 0.9, intent={**_only("advice", 0.6), "question": 0.6}),
     "the fennel then", {},
     "neutral 90% (negative 5%, positive 5%) · intent: unclear (advice and question tie at 60%) · mood unchanged: "
     "a neutral message"),
    ("judge nomination (1)", {"GHOST_INTENT": "1"}, _body("negative", 0.7, intent=_only("correcting")),
     "that is not the file I meant", {},
     "negative 70% (neutral 5%, positive 5%) · intent: correcting 95% · mood unchanged: a nudge needs 85% or more"
     " · nominated to the correction judge"),
    ("judge nomination (journal)", {}, _body("negative", 0.7, intent=_only("correcting")),
     "that is not the file I meant", {},
     "negative 70% (neutral 5%, positive 5%) · intent: correcting 95% · mood unchanged: a nudge needs 85% or more"
     " · would be nominated to the correction judge (journal only)"),
    ("the reply hears no register", {"GHOST_INTENT": "1"}, _plan_body(), UNSURE_PLAN, {"register_audience": ""},
     "negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · intent: advice 97% · mood nudged up +0.02 "
     "(asked for my opinion)"),
    ("the register switch off", {"GHOST_INTENT": "1", "GHOST_INTENT_REGISTER": "0"}, _plan_body(), UNSURE_PLAN, {},
     "negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · intent: advice 97% · mood nudged up +0.02 "
     "(asked for my opinion)"),
    ("mode 0: as before, whatever the intent field", {"GHOST_INTENT": "0"}, _plan_body(), UNSURE_PLAN, {},
     "negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · mood unchanged: no unhappy emotion behind "
     "the tone (with one, a nudge needs 60% or more)"),
    ("mode 0: garbage in the intent field", {"GHOST_INTENT": "0"},
     _body("neutral", 0.9, intent={"rant": 7, "advice": "lots"}, urgent="very"), "the fennel then", {},
     "neutral 90% (negative 5%, positive 5%) · mood unchanged: a neutral message"),
    ("journal: garbage in the intent field is no intent", {},
     _body("neutral", 0.9, intent={"rant": 7, "advice": "lots"}, urgent="very"), "the fennel then", {},
     "neutral 90% (negative 5%, positive 5%) · mood unchanged: a neutral message"),
]


async def _line_of(svc, tmp_path, monkeypatch, env, body, text, extra):
    import datetime as dt
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    svc.body = body
    sm = _model(tmp_path)
    kw = dict(extra) if isinstance(extra, dict) else {}
    if extra == _HOLD:
        sm.state.set_owner_mood("calm", M.resolve_label("calm")[1], 1,
                                now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5))
    if extra == _THUMB:
        sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="web-ve-prev")
    if kw.get("praised"):
        sm.note_owner_reaction(M.KIND_PRAISE, request_id="web-ve-line")
    kw.setdefault("register_humour", None)
    kw.setdefault("register_audience", "owner")
    calls = _logged(monkeypatch)
    await C.score_message(text, sm.root if extra != _NO_MODEL else tmp_path / "bare",
                          self_model=None if extra == _NO_MODEL else sm, request_id="web-ve-line", **kw)
    assert [(t, i) for t, _l, i in calls] == [("Sentiment", "💗")]        # ONE line, the 💗 one
    return calls[0][1]


@pytest.mark.parametrize("name,env,body,text,extra,want", LINE_TABLE, ids=[r[0] for r in LINE_TABLE])
async def test_one_line_per_message_says_everything(svc, tmp_path, monkeypatch, name, env, body, text, extra, want):
    # catches: the runner-ups shown for a taken label, a pitch the reply did not hear, two mood statements, the
    # intent part in mode 0, the cause's long form, the judge part missing or in the wrong mode
    assert await _line_of(svc, tmp_path, monkeypatch, env, body, text, extra) == want


@pytest.mark.parametrize("want", [r[5] for r in LINE_TABLE], ids=[r[0] for r in LINE_TABLE])
def test_the_line_states_the_mood_once(want):
    # catches: "mood unchanged" beside "mood nudged", or the mood spread over two parts
    assert not ("mood unchanged" in want and "mood nudged" in want)
    assert sum("mood" in part for part in want.split(" · ")) == 1


def test_every_intent_cause_has_a_short_form():
    assert set(I.CAUSE_SHORT) == set(I.EVENT_CAUSES)
    assert all(I.CAUSE_SHORT[c] and I.CAUSE_SHORT[c] != c for c in I.EVENT_CAUSES)


#: answers with no intent reading: their line is the pre-feature line in every mode
_PRE_FEATURE = [
    (_body("positive", 0.9), 0.08, "", "positive 90% (negative 5%, neutral 5%) · mood nudged up +0.08"),
    (_body("negative", 0.81, emotions={"confusion": 0.81}), 0.0, "below 0.85",
     "negative 81% (neutral 5%, positive 5%) · emotion: confusion 81% · mood unchanged: no unhappy emotion behind "
     "the tone (with one, a nudge needs 60% or more)"),
    (_body("unknown", 0.0, reason="not English"), 0.0, "not English", "not rated: not English · mood unchanged"),
]


@pytest.mark.parametrize("mode", ["0", "journal", "1"])
@pytest.mark.parametrize("body,nudge,held,want", _PRE_FEATURE)
def test_a_line_without_an_intent_reading_is_the_pre_feature_line(monkeypatch, mode, body, nudge, held, want):
    monkeypatch.setenv("GHOST_INTENT", mode)
    answer = C._answer(body)
    # every intent keyword at its loudest changes nothing without a reading
    assert C.rating_line(answer, nudge, held, intent_effect=I.CAUSE_ASKED, intent_held=C.INTENT_HELD_JOURNAL,
                         register="patient", judge=C.INTENT_JUDGE_NOMINATED) == want


def test_mode_0_ignores_an_intent_reading_given_to_the_line(monkeypatch):
    # catches: the line trusting the answer's keys instead of the mode (a reading kept from before a switch)
    monkeypatch.setenv("GHOST_INTENT", "0")
    answer = {"label": "neutral", "score": 0.9, "probs": {"neutral": 0.9}, "intent": _only("correcting"),
              "urgent": 0.9}
    assert C.rating_line(answer, 0.0, "neutral", intent_effect=I.CAUSE_ASKED, intent_held=C.INTENT_HELD_JOURNAL,
                         register="urgent", judge=C.INTENT_JUDGE_NOMINATED) == "neutral 90% · mood unchanged: a " \
                                                                                 "neutral message"


@pytest.mark.parametrize("mode", ["0", "journal", "1"])
async def test_the_line_never_says_the_messages_words(svc, tmp_path, monkeypatch, mode):
    text = "Kosmas zucchini quarantine paperwork marmalade asap?"
    words = {w.strip("?,.").lower() for w in text.split()}
    for body in (_body("negative", 0.95, intent=_only("advice"), urgent=0.9, emotions={"sadness": 0.8}),
                 _body("unknown", 0.0, intent=_only("venting"), reason="not English"),
                 _body("neutral", 0.9, intent={**_only("advice", 0.6), "question": 0.6})):
        line = await _line_of(svc, tmp_path, monkeypatch, {"GHOST_INTENT": mode}, body, text, {})
        assert not words & {w.strip("();:,%").lower() for w in line.split()}, line


@pytest.mark.parametrize("p,floor,shown", [(0.8996, 0.9, "89.9%"), (0.89, 0.9, "89%"), (0.0951, 0.1, "9.5%")])
def test_a_score_under_its_floor_never_reads_as_the_floor(p, floor, shown):
    lab = "advice" if floor == 0.9 else "instruction"
    reading = {"intent": {lab: p}}
    assert I.unclear_why(reading) == f"{lab} {shown} is under the {floor * 100:.0f}% it needs"


# ── review r3: the doubt before the backing, whatever the emotion model read ───────────────────────────

@pytest.mark.parametrize("emotions", [None, {"neutral": 0.9}, {"amusement": 0.8}], ids=["none", "neutral", "amused"])
async def test_banter_right_after_trouble_keeps_the_sarcasm_reason_whatever_the_emotions(svc, tmp_path, monkeypatch,
                                                                                         emotions):
    # catches: the backing checked before the doubt — {neutral} kept "no emotion backs the tone" and lost the
    # sarcasm, so the day record's words-only banter count depended on which messages had emotions
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _body("positive", 0.9, intent=_only("banter"), emotions=emotions)
    sm = _model(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="web-ve-prev")
    await C.score_message("lol nice one", sm.root, self_model=sm, register_audience="owner")
    row = C.journal_rows(sm.root)[0]
    assert row["intent_held"] == C.HELD_SARCASM and row["intent_effect"] == ""


# ── review r3: the reply stopped waiting before the reading landed ─────────────────────────────────────

#: the rows of ``_late``: an emotion reading, an intent reading, no urgency
_LATE_ROW_KEYS = (INTENT_ROW_KEYS | {"emotions"}) - {"urgent"}


async def _late(svc, tmp_path, monkeypatch, mode, *, rid="web-ve-late", mark=None, body=None):
    monkeypatch.setenv("GHOST_INTENT", mode)
    svc.body = body or _plan_body()
    sm = _model(tmp_path)
    C.expect_reading(rid)
    for m in (mark or []):
        C.note_reply_stopped_waiting(m)
    calls = _logged(monkeypatch)
    await C.score_message(UNSURE_PLAN, sm.root, self_model=sm, request_id=rid, register_audience="owner",
                          register_humour=None)
    C.forget_reading(rid)
    return C.journal_rows(sm.root)[0], calls[0][1]


async def test_mode_1_a_late_reading_says_its_pitch_was_not_offered(svc, tmp_path, monkeypatch):
    row, line = await _late(svc, tmp_path, monkeypatch, "1", mark=["web-ve-late", "web-ve-late"])  # twice: safe
    assert line == ("negative 81% (neutral 18%, positive 2%) · emotion: confusion 81% · intent: advice 97% · pitch: "
                    "patient (too late, not offered) · mood nudged up +0.02 (asked for my opinion)")
    # ``register`` keeps meaning what the reading implies; the lateness is its own key
    assert row["register"] == "patient" and row["register_late"] is True


async def test_mode_1_in_time_is_unchanged(svc, tmp_path, monkeypatch):
    row, line = await _late(svc, tmp_path, monkeypatch, "1")
    assert set(row) == _LATE_ROW_KEYS
    assert " · pitch: patient · " in line


async def test_a_mark_for_one_request_never_touches_another(svc, tmp_path, monkeypatch):
    C.expect_reading("web-ve-other")
    try:
        row, line = await _late(svc, tmp_path, monkeypatch, "1", mark=["web-ve-other", "never-expected"])
    finally:
        C.forget_reading("web-ve-other")
    assert "register_late" not in row and "too late" not in line


async def test_journal_mode_the_line_never_claims_lateness_and_the_row_only_when_marked_before(svc, tmp_path,
                                                                                               monkeypatch):
    row, line = await _late(svc, tmp_path / "a", monkeypatch, "journal", mark=["web-ve-late"])
    assert row["register_late"] is True and "too late" not in line and "pitch: patient (journal only)" in line
    assert set(row) == _LATE_ROW_KEYS | {"register_late"}
    row, line = await _late(svc, tmp_path / "b", monkeypatch, "journal")
    assert set(row) == _LATE_ROW_KEYS


async def test_mode_0_rows_keep_their_keys_even_when_marked(svc, tmp_path, monkeypatch):
    row, line = await _late(svc, tmp_path, monkeypatch, "0", mark=["web-ve-late"])
    assert set(row) == OLD_ROW_KEYS | {"emotions"} and "pitch" not in line


def test_a_mark_with_no_reading_ever_landing_leaks_nothing():
    C.note_reply_stopped_waiting("never-expected")                          # unknown id: a no-op
    assert "never-expected" not in C._LATE
    for i in range(C.READINGS_MAX * 2):
        C.expect_reading(f"web-ve-leak-{i}")
        C.note_reply_stopped_waiting(f"web-ve-leak-{i}")
    # catches: marks kept past their slot (the store's own bound)
    assert len(C._LATE) <= C.READINGS_MAX and set(C._LATE) <= set(C._READINGS)
    C.forget_reading(f"web-ve-leak-{C.READINGS_MAX * 2 - 1}")
    assert f"web-ve-leak-{C.READINGS_MAX * 2 - 1}" not in C._LATE
    C.expect_reading("web-ve-leak-0x")
    C.note_reply_stopped_waiting("web-ve-leak-0x")
    C.expect_reading("web-ve-leak-0x")                                      # a new message under the same id
    assert "web-ve-leak-0x" not in C._LATE
    for rid in list(C._READINGS):
        C.forget_reading(rid)


# ── review r3: the applied step in mode 1, the event before caps in journal mode ──────────────────────

@pytest.mark.parametrize("nudge,shown", [(0.015, "+0.01"), (0.02, "+0.02"), (0.004, "+0.004"), (-0.004, "−0.004"),
                                         (-0.0125, "−0.01")])
def test_mode_1_reports_the_applied_step(monkeypatch, nudge, shown):
    # catches: the raw event shown instead of what the ledger let through, or a small step shown as "+0.00"
    monkeypatch.setenv("GHOST_INTENT", "1")
    answer = C._answer(_plan_body())
    line = C.rating_line(answer, nudge, "below 0.85", intent_effect=I.CAUSE_ASKED, intent_held="")
    assert line.endswith(f"· mood nudged {'up' if nudge > 0 else 'down'} {shown} (asked for my opinion)")


async def test_the_capped_step_in_mode_1_is_what_the_row_moved(svc, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_INTENT", "1")
    sm = _model(tmp_path)
    calls = _logged(monkeypatch)
    for i in range(6):
        svc.body = _plan_body()
        await C.score_message(UNSURE_PLAN, sm.root, self_model=sm, request_id=f"web-ve-cap{i}",
                              register_audience="owner")
    for row, (_t, line, _i) in zip(C.journal_rows(sm.root), calls):
        if row["nudge"]:
            assert f"mood nudged up {C._step(row['nudge'])} (asked for my opinion)" in line
        else:
            assert "mood nudged" not in line


async def test_score_message_without_an_audience_states_no_register(svc, tmp_path, monkeypatch):
    # review r3 (item 5): the default is "not stated" — a future caller never claims an owner-private pitch
    monkeypatch.setenv("GHOST_INTENT", "1")
    svc.body = _plan_body()
    calls = _logged(monkeypatch)
    await C.score_message(UNSURE_PLAN, tmp_path, register_humour=None)
    assert C.journal_rows(tmp_path)[0]["register"] == "" and "pitch" not in calls[0][1]
