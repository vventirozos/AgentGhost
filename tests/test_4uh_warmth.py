"""§4UH-warmth / §4UH-lean (2026-11-12, operator: "separate them", "make it stronger", "proceed" on the emotion model).

* the sentiment service answers ``emotions`` from a second, optional model (GoEmotions, sigmoid per label);
  without it, the answer is exactly as before;
* the agent keeps ``emotions`` from a valid answer only, journals it, and names the felt ones in the log line;
* the day record counts the owner's rated messages, the warm ones and the annoyed ones (``owner_tones``) —
  never a word; sarcasm and the wording of a task are not counted as felt;
* warmth reads those counts (thanks and tone on days before them); thumbs and corrections are directness only;
* the lean is 0.25 a step.
"""
from __future__ import annotations

import datetime
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interface" / "sentiment-service"))

import sentiment_service as S  # noqa: E402
from ghost_agent.selfhood import character as C  # noqa: E402
from ghost_agent.selfhood import days as D  # noqa: E402
from ghost_agent.selfhood import mood as M  # noqa: E402
from ghost_agent.selfhood import sentiment as SE  # noqa: E402

UTC = datetime.timezone.utc
KEY = "k"


class _Clf:
    def __init__(self, labels, probs):
        self.labels, self._p, self.seen = labels, probs, []

    def probs(self, text):
        self.seen.append(text)
        return list(self._p)


class _Lid:
    def top(self, text):
        return ("el", 0.9) if text.startswith("γ") else ("en", 0.95)


SENT = _Clf(["negative", "neutral", "positive"], [0.02, 0.08, 0.90])
EMO_LABELS = ["gratitude", "admiration", "annoyance", "neutral"]


# ── the service ─────────────────────────────────────────────────────────────

def test_the_answer_carries_emotions_at_or_above_the_floor():
    emo = _Clf(EMO_LABELS, [0.987, 0.15, 0.049, 0.05])
    got = S.classify(SENT, _Lid(), "thanks, that worked perfectly", emo)
    assert got["label"] == "positive" and got["probs"]["positive"] == 0.9
    assert got["emotions"] == {"gratitude": 0.987, "admiration": 0.15, "neutral": 0.05}   # 0.049 < EMOTION_MIN


def test_without_the_emotion_model_the_answer_is_as_before():
    got = S.classify(SENT, _Lid(), "thanks", None)
    assert "emotions" not in got and set(got) == {"ok", "label", "score", "probs", "reason"}


def test_an_unknown_text_reaches_neither_model():
    emo = _Clf(EMO_LABELS, [0.9, 0, 0, 0])
    got = S.classify(SENT, _Lid(), "γεια σου", emo)
    assert got["label"] == S.UNKNOWN and "emotions" not in got and emo.seen == []


def test_the_handler_passes_the_emotion_model_through():
    import urllib.request
    emo = _Clf(EMO_LABELS, [0.99, 0.0, 0.0, 0.0])
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(SENT, _Lid(), KEY, emo))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/classify",
                                     data=json.dumps({"text": "thank you"}).encode(),
                                     headers={"X-Sentiment-Service-Key": KEY})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
    finally:
        server.shutdown()
        server.server_close()
    assert body["emotions"] == {"gratitude": 0.99}


def test_the_sigmoid_classifier_and_its_parity_activation(tmp_path):
    """A sigmoid model's parity file must say so; a softmax file refuses a sigmoid model and the other way round."""
    pytest.importorskip("numpy")
    c = S.Classifier.__new__(S.Classifier)
    import numpy as np
    c._np, c.activation, c.labels = np, "sigmoid", ["x"]
    c.probs = lambda text: [0.5]
    (tmp_path / "parity.json").write_text(json.dumps({"samples": ["x"], "probs": [[0.5]], "activation": "sigmoid"}))
    c._check_parity(tmp_path / "parity.json")
    (tmp_path / "parity.json").write_text(json.dumps({"samples": ["x"], "probs": [[0.5]]}))
    with pytest.raises(SystemExit):
        c._check_parity(tmp_path / "parity.json")
    c.activation = "softmax"
    c._check_parity(tmp_path / "parity.json")                         # the sentiment model's file: no key


# ── the agent: the answer, the journal, the log line ───────────────────────

def _body(**extra):
    return {"ok": True, "label": "positive", "score": 0.9,
            "probs": {"negative": 0.02, "neutral": 0.08, "positive": 0.9}, "reason": "", **extra}


def test_a_valid_emotions_field_is_kept():
    got = SE._answer(_body(emotions={"gratitude": 0.99, "neutral": 0.06}))
    assert got["emotions"] == {"gratitude": 0.99, "neutral": 0.06}


@pytest.mark.parametrize("emo", [{"gratitude": "high"}, {"not_an_emotion": 0.9}, {"gratitude": True},
                                 {"gratitude": 1.5}, {"gratitude": -0.1}, "gratitude", [0.9]])
def test_a_bad_emotions_field_is_dropped_and_the_answer_kept(emo):
    got = SE._answer(_body(emotions=emo))
    assert got is not None and got["label"] == "positive" and "emotions" not in got


def test_an_unknown_answer_keeps_no_emotions():
    got = SE._answer({"ok": True, "label": "unknown", "score": 0.0, "probs": {}, "reason": "not English",
                      "emotions": {"gratitude": 0.9}})
    assert got is not None and "emotions" not in got


def test_the_emotion_labels_are_the_models_28():
    assert len(SE.EMOTION_LABELS) == 28 and "neutral" in SE.EMOTION_LABELS
    assert SE.WARM_EMOTIONS <= SE.EMOTION_LABELS and SE.COLD_EMOTIONS <= SE.EMOTION_LABELS
    assert not (SE.WARM_EMOTIONS & SE.COLD_EMOTIONS)
    # questions and the owner's own news are not how the owner is with Ghost
    for k in ("curiosity", "confusion", "sadness", "fear", "grief", "nervousness", "neutral"):
        assert k not in SE.WARM_EMOTIONS | SE.COLD_EMOTIONS


def test_the_log_line_names_the_felt_emotions_never_the_words():
    ans = SE._answer(_body(emotions={"gratitude": 0.99, "admiration": 0.6, "joy": 0.3, "neutral": 0.7}))
    line = SE.rating_line(ans, 0.08, "")
    assert "emotion: gratitude 99% (counts as warm), admiration 60% (counts as warm)" in line and "joy" not in line and "neutral 70%" not in line


def test_the_log_line_shows_a_weaker_emotion_and_says_what_is_counted():
    """req 9cbdf9aa: a question read curiosity 0.44 and the line said nothing."""
    line = SE.rating_line(SE._answer(_body(emotions={"curiosity": 0.44, "neutral": 0.19, "caring": 0.08})), 0.0, "")
    assert "· emotion: curiosity 44% ·" in line and "caring" not in line and "counts as" not in line
    line = SE.rating_line(SE._answer(_body(emotions={"annoyance": 0.7, "joy": 0.3})), 0.0, "")
    assert "emotion: annoyance 70% (counts as annoyed), joy 30% ·" in line
    assert "curiosity 60% ·" in SE.rating_line(SE._answer(_body(emotions={"curiosity": 0.6})), 0.0, "")
    assert "· emotion: none clear ·" in SE.rating_line(SE._answer(_body(emotions={"neutral": 0.9})), 0.0, "")
    assert SE.rating_line(SE._answer(_body()), 0.08, "") == "positive 90% (neutral 8%, negative 2%) · mood nudged up +0.08"


@pytest.mark.asyncio
async def test_the_journal_row_holds_the_emotions(tmp_path, monkeypatch):
    async def fake(text, **k):
        return SE._answer(_body(emotions={"gratitude": 0.99}))
    monkeypatch.setattr(SE, "classify", fake)
    assert await SE.score_message("thank you so much, that worked", tmp_path, request_id="r1")
    row = SE.journal_rows(tmp_path)[-1]
    assert row["emotions"] == {"gratitude": 0.99} and row["label"] == "positive"


# ── the day record: how the owner talked ───────────────────────────────────

def _journal(home, *rows):
    p = home / D.SELFHOOD_REL / SE.JOURNAL_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(ts, emotions, held="", label="positive"):
    return {"ts": ts, "label": label, "emotions": emotions, "held": held, "text": "SECRET WORDS"}


DAY = datetime.date(2026, 11, 13)


def test_owner_tones_counts_rated_warm_and_cold_messages_of_the_mood_day(tmp_path):
    home = tmp_path
    _journal(home,
             _row("2026-11-13T09:00:00+00:00", {"gratitude": 0.99}),
             _row("2026-11-13T10:00:00+00:00", {"annoyance": 0.7, "approval": 0.6}),     # both sides
             _row("2026-11-13T11:00:00+00:00", {"curiosity": 0.8, "neutral": 0.3}),     # read, neither side
             _row("2026-11-13T12:00:00+00:00", {"gratitude": 0.49}),                    # under the line
             _row("2026-11-14T03:30:00+00:00", {"gratitude": 0.9}),                     # before 04:00: still the 13th
             _row("2026-11-14T04:30:00+00:00", {"gratitude": 0.9}),                     # the next mood day
             _row("2026-11-13T13:00:00+00:00", {"gratitude": 0.9}, held=SE.HELD_SARCASM),
             _row("2026-11-13T13:30:00+00:00", {"anger": 0.9}, held=SE.HELD_TASK),
             _row("2026-11-13T14:00:00+00:00", None, label="unknown"),                    # not read by the model
             {"ts": "2026-11-13T15:00:00+00:00", "label": "positive"},                    # before §4UH-warmth
             _row("2026-11-13T16:00:00+00:00", {"gratitude": "x"}),                       # a damaged number
             _row("not a time", {"gratitude": 0.9}))
    # rated: 09, 10, 11, 12, 03:30, sarcasm (13:00), damaged (16:00) = 7; warm: 09, 10, 03:30 = 3 (sarcasm's
    # gratitude is not warmth); cold: 10 = 1; thanks: 09, 03:30 = 2
    assert D.owner_tones(home, DAY, tz=UTC) == (7, 3, 1, 2)


def test_sarcasm_keeps_its_annoyance_and_a_task_counts_for_nothing(tmp_path):
    _journal(tmp_path,
             _row("2026-11-13T09:00:00+00:00", {"gratitude": 0.9, "annoyance": 0.6}, held=SE.HELD_SARCASM),
             _row("2026-11-13T09:05:00+00:00", {"anger": 0.9}, held=SE.HELD_TASK))
    assert D.owner_tones(tmp_path, DAY, tz=UTC) == (1, 0, 1, 0)


def test_the_rotated_journal_is_read_too(tmp_path):
    _journal(tmp_path, _row("2026-11-13T09:00:00+00:00", {"gratitude": 0.9}))
    p = tmp_path / D.SELFHOOD_REL / SE.JOURNAL_FILENAME
    p.rename(p.with_name(p.name + ".1"))
    _journal(tmp_path, _row("2026-11-13T10:00:00+00:00", {"joy": 0.9}))
    assert D.owner_tones(tmp_path, DAY, tz=UTC) == (2, 2, 0, 1)


def test_owner_tones_without_a_journal_or_a_home(tmp_path):
    assert D.owner_tones(tmp_path, DAY, tz=UTC) == (0, 0, 0, 0)
    assert D.owner_tones(None, DAY, tz=UTC) == (0, 0, 0, 0)


def test_the_day_record_carries_the_counts_and_no_words(tmp_path):
    home = tmp_path
    _journal(home, _row("2026-11-13T09:00:00+00:00", {"gratitude": 0.99}))
    rec = D.build_day(home, DAY, tz=UTC)
    assert (rec["msgs_rated"], rec["msgs_warm"], rec["msgs_cold"], rec["msgs_thanks"]) == (1, 1, 0, 1)
    assert "SECRET" not in json.dumps(rec)


# ── warmth and directness read different things ────────────────────────────

def _rec(**kw):
    base = {"date": "2026-11-13", "thumbs_up": 0, "thumbs_down": 0, "corrections": 0, "praise": 0, "tone_up": 0,
            "tone_down": 0, "own": {}, "shades": {}}
    base.update(kw)
    return base


def test_no_signal_feeds_both_warmth_and_directness():
    for key in ("thumbs_up", "thumbs_down", "corrections"):
        s = C.day_signals(_rec(**{key: 3}))
        assert s["warmth"] is None and s["directness"] is not None, key
    for key, kw in (("praise", {"praise": 3}), ("tone", {"tone_up": 3}), ("emotions", {"msgs_rated": 3, "msgs_warm": 3})):
        s = C.day_signals(_rec(**kw))
        assert s["warmth"] is not None and s["directness"] is None, key


def test_a_good_thumbs_day_moves_directness_and_not_warmth():
    start = datetime.date(2026, 10, 1)
    usual = [dict(_rec(thumbs_up=1, thumbs_down=1, msgs_rated=4, msgs_warm=1),
                  date=(start + datetime.timedelta(days=i)).isoformat()) for i in range(6)]
    good = dict(_rec(thumbs_up=6, msgs_rated=4, msgs_warm=1), date=(start + datetime.timedelta(days=6)).isoformat())
    pulls = C.day_pulls(usual + [good])
    assert pulls == {good["date"]: {"directness": 1}}


def test_the_signal_words_say_what_each_dial_reads():
    assert "thumbs" not in C.SIGNAL_WORDS["warmth"] and "talked" in C.SIGNAL_WORDS["warmth"]
    assert "thumbs" in C.SIGNAL_WORDS["directness"]


# ── the lean is stronger ───────────────────────────────────────────────────

def test_the_lean_step_is_a_quarter_and_bounded_at_half():
    assert C.LEAN_STEP == 0.25
    ev = M.Event(-0.2, 0.0, "a call failed", M.KIND_TURN)
    assert C.lean(ev, {"warmth": 0, "humour": 2, "directness": 0}).dp == pytest.approx(-0.1)
    assert C.lean(ev, {"warmth": 0, "humour": -9, "directness": 0}).dp == pytest.approx(-0.3)
    thanks = M.Event(0.2, 0.0, "thanks", M.KIND_PRAISE)
    assert C.lean(thanks, {"warmth": 2, "humour": 0, "directness": 0}).dp == pytest.approx(0.3)
    assert C.lean(thanks, {"warmth": -2, "humour": 0, "directness": 0}).dp == pytest.approx(0.1)


# ── review: the two warmth scales are never compared with each other ────────

def _dated(recs, start=datetime.date(2026, 10, 1)):
    return [dict(r, date=(start + datetime.timedelta(days=i)).isoformat()) for i, r in enumerate(recs)]


def test_ordinary_new_days_do_not_read_worse_than_an_old_norm_of_good_days():
    """The review's case: old days carry warmth only when they had thanks (None otherwise), so their norm is
    high; new days are read every day. Judged only against their own scale, a week of ordinary new days pulls
    nothing until it has ``BASE_MIN`` of its own, and then only against itself."""
    old = [_rec(praise=2) if i % 2 else _rec() for i in range(21)]
    new = [_rec(msgs_rated=8, msgs_warm=w) for w in (2, 1, 2, 0, 1, 2, 1)]
    days = _dated(old + new)
    pulls = C.day_pulls(days)
    first_new = {d["date"] for d in days[21:21 + C.BASE_MIN]}
    assert not any(p.get("warmth") for date, p in pulls.items() if date in first_new)
    downs = sum(1 for d in days[21:] if (pulls.get(d["date"]) or {}).get("warmth") == -1)
    assert downs <= 1
    state = {"dials": {"warmth": -1, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    assert [m for m in C.propose(days, state, [], days[-1]["date"]) if m["dial"] == "warmth"] == []


def test_a_new_day_is_judged_against_new_days():
    days = _dated([_rec(msgs_rated=6, msgs_warm=1)] * 6 + [_rec(msgs_rated=6, msgs_warm=3)])
    assert C.day_pulls(days) == {days[-1]["date"]: {"warmth": 1}}
    assert C.signal_scales(days[0])["warmth"] == "emotions" and C.signal_scales(_rec())["warmth"] == "counts"
    assert C.signal_scales({"msgs_rated": True})["warmth"] == "counts"


def test_thanks_the_model_did_not_read_still_count():
    """Greek thanks: the praise check counts them, the English-only emotion model does not."""
    greek_day = _rec(praise=3, msgs_rated=1, msgs_warm=0, msgs_thanks=0)
    assert C.day_signals(greek_day)["warmth"] == 3.0
    english_day = _rec(praise=2, msgs_rated=2, msgs_warm=2, msgs_thanks=2)     # the same thanks, read: once
    assert C.day_signals(english_day)["warmth"] == 2.0


# ── review: the emotions never cost the sentiment ──────────────────────────

class _Boom:
    labels = EMO_LABELS

    def probs(self, text):
        raise RuntimeError("onnx went away")


def test_an_emotion_failure_keeps_the_sentiment_answer():
    got = S.classify(SENT, _Lid(), "thanks", _Boom())
    assert got["label"] == "positive" and "emotions" not in got


def test_an_emotion_model_whose_outputs_do_not_match_its_labels_answers_none():
    got = S.classify(SENT, _Lid(), "thanks", _Clf(EMO_LABELS, [0.9, 0.1]))
    assert got["label"] == "positive" and "emotions" not in got


# ── the logs (operator: "would be nice to have it") ──────────────────────────

def test_the_day_record_line_says_how_the_owner_talked_in_counts():
    line = D.day_line(dict(_rec(msgs_rated=6, msgs_warm=2, msgs_cold=1, msgs_thanks=1), shades={}))
    assert "your messages: 6 read for emotion, 2 warm, 1 annoyed" in line
    assert "your messages: 1 read for emotion, 1 warm, 0 annoyed" in D.day_line(_rec(msgs_rated=1, msgs_warm=1))
    assert "your messages" not in D.day_line(_rec())


def test_the_pull_line_matches_the_rule_and_says_when_there_is_no_norm():
    days = _dated([_rec(msgs_rated=6, msgs_warm=1)] * 6 + [_rec(msgs_rated=6, msgs_warm=3)])
    line = C.pull_line(days, days[-1]["date"])
    assert line.startswith(days[-1]["date"] + " vs usual: ")
    assert "warmth higher (3.0 vs 1.0), pulls up" in line and "directness no signal today" in line
    assert C.day_pulls(days) == {days[-1]["date"]: {"warmth": 1}}
    first = C.pull_line(days, days[0]["date"])
    assert "warmth 1.0, not enough days yet to compare (needs 5)" in first
    usual = C.pull_line(days, days[5]["date"])
    assert "warmth about usual (1.0 vs 1.0)" in usual


def test_the_pull_line_never_raises():
    assert isinstance(C.pull_line(None, "2026-11-13"), str)
    assert isinstance(C.pull_line([{"date": "2026-11-13", "msgs_rated": "x", "praise": object()}], "2026-11-13"), str)
    assert C.pull_line([], "2026-11-13") == "2026-11-13 vs usual: warmth no signal today · humour no signal today · directness no signal today"
