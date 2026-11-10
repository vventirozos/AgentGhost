"""§4TW — the sentiment of the owner's messages, from nova's sentiment-service, in SHADOW (a journal row; the mood
does not move). The service runs its real HTTP layer with a fake classifier; the agent side calls the real client
against it. Each test names the world where it would fail. All text is fictional.
"""
import asyncio
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from ghost_agent.selfhood import mood as M

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interface" / "sentiment-service"))

import sentiment_service as S  # noqa: E402
from ghost_agent.selfhood import sentiment as C  # noqa: E402

KEY = "test-key"


class FakeClassifier:
    labels = ["negative", "neutral", "positive"]

    def __init__(self, probs=(0.05, 0.10, 0.85), fail=None):
        self.seen, self._probs, self.fail = [], list(probs), fail

    def probs(self, text):
        self.seen.append(text)
        if self.fail:
            raise self.fail
        return self._probs


class FakeLangId:
    """English at 0.95, unless the text is in ``answers``."""

    def __init__(self, answers=None):
        self.answers = answers or {}

    def top(self, text):
        return self.answers.get(text, ("en", 0.95))


GREEKLISH = {"den douleuei pali, ti ekanes?": ("la", 0.09), "efxaristo poli, teleio": ("pt", 0.76),
             "great": ("en", 0.12)}


@pytest.fixture
def service():
    clf = FakeClassifier()
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(clf, FakeLangId(GREEKLISH), KEY))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", clf
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def fresh_breaker():
    C.BREAKER.reset()
    yield
    C.BREAKER.reset()


@pytest.fixture
def client(service, tmp_path, monkeypatch):
    url, clf = service
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    monkeypatch.setenv("GHOST_SENTIMENT_URL", url + "/")
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(key_file))
    return url, clf


def call(url, path, body=None, key=KEY, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data=data, headers={"X-Sentiment-Service-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ── the service: the language gate ──────────────────────────────────────────

@pytest.mark.parametrize("text", ["thanks, that worked", "Café déjà vu, merci", "ok 👍 nice"])
def test_latin_text_is_read(text):
    assert S.script_gate(text) == ""


@pytest.mark.parametrize("text", ["ευχαριστώ πολύ, τέλειο", "Ευχαριστώ ok", "спасибо большое", "谢谢你的帮助"])
def test_mostly_non_latin_text_is_not_english(text):
    # fails if Greek (or any non-Latin script) reaches the English-only model
    assert S.script_gate(text) == S.REASON_LANGUAGE


def test_a_mixed_message_follows_its_majority_script():
    assert S.script_gate("τέλεια δουλειά με το server") == S.REASON_LANGUAGE        # 18 Greek letters, 6 Latin
    assert S.script_gate("the deploy on nova worked, ευχαριστώ") == ""              # 24 Latin, 9 Greek


@pytest.mark.parametrize("text", ["", "👍", "?!", "x", "12345", "   "])
def test_text_without_words_is_not_classified(text):
    assert S.script_gate(text) == S.REASON_NO_TEXT


@pytest.mark.parametrize("lang,prob,reason", [
    ("en", 0.95, ""), ("en", S.MIN_ENGLISH_PROB, ""),
    ("en", 0.12, S.REASON_UNCLEAR),                 # the prior every unknown word gets
    ("pt", 0.76, S.REASON_LANGUAGE), ("la", 0.09, S.REASON_LANGUAGE), ("el", 0.99, S.REASON_LANGUAGE),
    ("", 0.0, S.REASON_LANGUAGE)])
def test_the_language_gate_lets_through_confident_english_only(lang, prob, reason):
    assert S.language_gate(lang, prob) == reason


@pytest.mark.parametrize("text,reason", [("ευχαριστώ πολύ", S.REASON_LANGUAGE),
                                         ("den douleuei pali, ti ekanes?", S.REASON_LANGUAGE),
                                         ("great", S.REASON_UNCLEAR)])
def test_unknown_never_reaches_the_model(text, reason):
    # fails if Greek letters, Greek in Latin letters (Greeklish) or a word too short to tell reach the model
    clf = FakeClassifier()
    out = S.classify(clf, FakeLangId(GREEKLISH), text)
    assert out == {"ok": True, "label": "unknown", "score": 0.0, "probs": {}, "reason": reason}
    assert clf.seen == []


def test_the_script_gate_runs_before_the_language_identifier():
    class Boom:
        def top(self, text):
            raise AssertionError("asked the language identifier about Greek letters")
    assert S.classify(FakeClassifier(), Boom(), "ευχαριστώ πολύ")["reason"] == S.REASON_LANGUAGE


def test_the_answer_is_the_most_probable_label_with_all_three_probabilities():
    out = S.classify(FakeClassifier((0.7, 0.2, 0.1)), FakeLangId(), "this is broken again")
    assert out["label"] == "negative" and out["score"] == 0.7
    assert out["probs"] == {"negative": 0.7, "neutral": 0.2, "positive": 0.1}


# ── the service: the HTTP layer ─────────────────────────────────────────────

def test_health_needs_no_key(service):
    url, _ = service
    assert call(url, "/health", key=None) == (200, {"ok": True})


@pytest.mark.parametrize("key", [None, "", "wrong"])
def test_classify_refuses_a_missing_or_wrong_key(service, key):
    url, clf = service
    status, body = call(url, "/classify", {"text": "great"}, key=key)
    assert status == 401 and body["ok"] is False and clf.seen == []


def test_an_empty_server_key_refuses_everyone():
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(FakeClassifier(), FakeLangId(), ""))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert call(url, "/classify", {"text": "great"}, key="")[0] == 401
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("raw,status", [(b"not json", 400), (b"[1]", 400), (b'{"text": 5}', 400),
                                        (json.dumps({"text": "a" * (S.MAX_TEXT_CHARS + 1)}).encode(), 413)])
def test_bad_bodies_are_refused(service, raw, status):
    url, clf = service
    assert call(url, "/classify", raw=raw)[0] == status and clf.seen == []


def test_a_body_over_the_cap_is_refused_before_it_is_read(service):
    url, _ = service
    req = urllib.request.Request(url + "/classify", data=b"{}", headers={
        "X-Sentiment-Service-Key": KEY, "Content-Length": str(S.MAX_BODY_BYTES + 1)})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 413


def test_a_model_failure_is_a_500_without_the_text(service, caplog):
    url, clf = service
    clf.fail = RuntimeError("Elena's secret message")
    status, body = call(url, "/classify", {"text": "Elena's secret message"})
    assert status == 500 and "secret" not in json.dumps(body)
    assert "secret" not in caplog.text


def test_the_log_carries_the_label_never_the_text(service, caplog):
    url, _ = service
    caplog.set_level("DEBUG", logger="sentiment-service")
    assert call(url, "/classify", {"text": "Markos loved the new garden"})[0] == 200
    assert "positive" in caplog.text and "Markos" not in caplog.text and "garden" not in caplog.text


# ── the agent's client ──────────────────────────────────────────────────────

def test_configured_needs_both_url_and_key(tmp_path, monkeypatch):
    key = tmp_path / "k"
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(key))
    monkeypatch.setenv("GHOST_SENTIMENT_URL", "http://100.64.0.5:8097")
    assert C.configured() is False                                  # no key file
    key.write_text("  \n")
    assert C.configured() is False                                  # empty key
    key.write_text("k")
    assert C.configured() is True
    monkeypatch.setenv("GHOST_SENTIMENT_URL", " ")
    assert C.configured() is False


async def test_the_client_returns_the_services_answer(client):
    _, clf = client
    out = await C.classify("thanks, that worked")
    assert out == {"label": "positive", "score": 0.85,
                   "probs": {"negative": 0.05, "neutral": 0.1, "positive": 0.85}, "reason": ""}
    assert clf.seen == ["thanks, that worked"] and C.BREAKER.scored == 1


@pytest.mark.parametrize("text", ["ευχαριστώ πολύ", "efxaristo poli, teleio"])
async def test_greek_in_either_alphabet_comes_back_unknown(client, text):
    _, clf = client
    out = await C.classify(text)
    assert out["label"] == "unknown" and out["probs"] == {} and clf.seen == []


async def test_text_is_cut_to_the_services_limit(client):
    _, clf = client
    assert (await C.classify("a" * (C.MAX_TEXT_CHARS + 50)))["label"] == "positive"
    assert len(clf.seen[0]) == C.MAX_TEXT_CHARS


async def test_a_wrong_key_is_a_failed_call(client, tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.write_text("nope")
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(bad))
    assert await C.classify("great") is None
    assert C.BREAKER.last_fail_why == "HTTP 401"


@pytest.mark.parametrize("body", [
    {"ok": False}, {"ok": True, "label": "joy", "score": 0.9, "probs": {}},
    {"ok": True, "label": "positive", "score": True, "probs": {}},
    {"ok": True, "label": "positive", "score": 0.9, "probs": {"joy": 1.0}},
    {"ok": True, "label": "positive", "score": 0.9, "probs": {"positive": "x"}},
    [], "positive"])
def test_an_answer_that_is_not_one_is_refused(body):
    assert C._answer(body) is None


async def test_nothing_is_asked_when_not_configured(monkeypatch):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    assert await C.classify("great") is None and C.BREAKER.last_fail == 0


async def test_a_dead_service_pauses_after_three_failures_and_retries_by_itself(tmp_path, monkeypatch):
    key = tmp_path / "k"
    key.write_text(KEY)
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(key))
    monkeypatch.setenv("GHOST_SENTIMENT_URL", "http://127.0.0.1:9")          # nothing listens
    clock = [1000.0]
    now = lambda: clock[0]  # noqa: E731
    for _ in range(C.BREAKER_FAILS):
        assert await C.classify("great", now_fn=now) is None
    assert C.BREAKER.may_ask(now()) is False
    # paused: no call is made at all (the failure stamp does not move)
    stamp = C.BREAKER.last_fail
    clock[0] += 10
    assert await C.classify("great", now_fn=now) is None and C.BREAKER.last_fail == stamp
    assert "PAUSED" in C.health_text(now_fn=now)
    # after the pause it tries again on its own
    clock[0] += C.BREAKER_PAUSE_S
    assert C.BREAKER.may_ask(now()) is True
    assert await C.classify("great", now_fn=now) is None and C.BREAKER.last_fail == clock[0]
    assert C.BREAKER.may_ask(now()) is True          # one failure after a pause does not pause again


async def test_a_success_clears_the_failure_count(client):
    C.BREAKER.failed("ConnectError", 1.0)
    C.BREAKER.failed("ConnectError", 2.0)
    assert await C.classify("great") is not None
    assert C.BREAKER.fails == 0


async def test_a_slow_service_counts_as_a_failure(tmp_path, monkeypatch):
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)                                      # accepts, never answers
    key = tmp_path / "k"
    key.write_text(KEY)
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(key))
    monkeypatch.setenv("GHOST_SENTIMENT_URL", f"http://127.0.0.1:{sock.getsockname()[1]}")
    monkeypatch.setattr(C, "TIMEOUT_S", 0.3)
    try:
        assert await asyncio.wait_for(C.classify("great"), 5) is None
        assert "Timeout" in C.BREAKER.last_fail_why
    finally:
        sock.close()


# ── the journal ──────────────────────────────────────────────────────

@pytest.mark.parametrize("content,text", [
    ("hi there", "hi there"),
    ([{"type": "text", "text": "look at this"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
     "look at this"),
    (None, ""), (5, ""), ([{"type": "image_url"}], ""),
    # a channel member's row (the bot labels it) is never the owner's words
    ("[message from another channel member — not the owner; Kosmas]\nthis bot is useless", ""),
    ("  [message from the owner of this assistant]\nfine", ""),
    # the clients' notes are taken out
    ("thanks!\n\n[SYSTEM NOTE: The user attached a file named 'plan.pdf'. It has been uploaded to your sandbox. "
     "Use your file_system or knowledge_base tools to interact with it.]", "thanks!"),
    ("[SYSTEM NOTE: The user attached a file named 'big.mov' but it could NOT be ingested (too large or the "
     "fetch failed).]", ""),
    ("read this\n\n(File just uploaded to the sandbox: notes.txt)", "read this"),
    ("<@U0ABC123> perfect, ship it", "perfect, ship it"), ("<@U0ABC123|ghost>", "")])
def test_message_text(content, text):
    assert C.message_text(content) == text


async def test_a_scored_message_writes_one_private_row(client, tmp_path):
    assert await C.score_message("thanks, that worked", tmp_path, request_id="web-abc") is True
    path = tmp_path / C.JOURNAL_FILENAME
    assert path.stat().st_mode & 0o777 == 0o600
    (row,) = C.journal_rows(tmp_path)
    assert row["label"] == "positive" and row["request_id"] == "web-abc"
    assert row["text"] == "thanks, that worked" and row["chars"] == 19 and row["ts"].endswith("+00:00")


async def test_the_row_keeps_only_the_start_of_a_long_message(client, tmp_path):
    await C.score_message("great " * 200, tmp_path)
    (row,) = C.journal_rows(tmp_path)
    assert len(row["text"]) == C.JOURNAL_TEXT_CHARS and row["chars"] == len(("great " * 200).strip())


async def test_greek_is_journalled_as_unknown(client, tmp_path):
    await C.score_message("ευχαριστώ πολύ", tmp_path)
    (row,) = C.journal_rows(tmp_path)
    assert row["label"] == "unknown" and row["reason"] == S.REASON_LANGUAGE


async def test_no_answer_writes_no_row(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    assert await C.score_message("great", tmp_path) is False
    assert not (tmp_path / C.JOURNAL_FILENAME).exists()


async def test_an_empty_message_is_not_sent(client, tmp_path):
    _, clf = client
    assert await C.score_message("   ", tmp_path) is False and clf.seen == []


async def test_a_full_journal_is_moved_aside_and_a_new_one_started(client, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "JOURNAL_MAX_BYTES", 100)
    for _ in range(3):
        await C.score_message("thanks, that worked", tmp_path)
    assert (tmp_path / (C.JOURNAL_FILENAME + ".1")).exists()
    assert 1 <= len(C.journal_rows(tmp_path)) < 3


async def test_an_unwritable_journal_never_raises(client, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    assert await C.score_message("great", blocker / "sub") is False


def test_journal_rows_skips_damaged_lines(tmp_path):
    (tmp_path / C.JOURNAL_FILENAME).write_text('{"label": "positive"}\nnot json\n[1]\n{"label": "neutral"}\n')
    assert [r["label"] for r in C.journal_rows(tmp_path)] == ["positive", "neutral"]
    assert [r["label"] for r in C.journal_rows(tmp_path, limit=1)] == ["neutral"]


# ── feeding the mood (operator 11-09: option 1, the owner's sentiment whatever it is about) ──

def _model(tmp_path):
    from ghost_agent.selfhood import SelfModel
    return SelfModel(tmp_path / "selfhood")


@pytest.mark.parametrize("probs,dp", [((0.05, 0.10, 0.85), +0.08), ((0.90, 0.07, 0.03), -0.08)])
async def test_a_confident_score_nudges_the_pleasantness_a_third_of_a_thumb(client, tmp_path, probs, dp):
    _, clf = client
    clf._probs = list(probs)
    sm = _model(tmp_path)
    assert await C.score_message("my flight got cancelled", sm.root, self_model=sm) is True
    v = sm.mood()
    assert (v.p, v.e) == (pytest.approx(dp), pytest.approx(dp / 4))        # §4TX: a little energy each way
    (row,) = C.journal_rows(sm.root)
    assert row["nudge"] == pytest.approx(dp) and row["held"] == ""
    assert v.recent[-1]["kind"] == "sentiment"
    assert C.BREAKER.moved == 1


@pytest.mark.parametrize("probs,held", [
    ((0.10, 0.06, 0.84), "below 0.85"),          # just under the floor
    ((0.40, 0.20, 0.40), "below 0.85"),          # mixed
    ((0.04, 0.90, 0.06), "neutral")])            # an instruction
async def test_an_unsure_or_neutral_score_moves_nothing(client, tmp_path, probs, held):
    _, clf = client
    clf._probs = list(probs)
    sm = _model(tmp_path)
    await C.score_message("list the files in the sandbox", sm.root, self_model=sm)
    assert (sm.mood().p, sm.mood().e) == (0.0, 0.0)
    (row,) = C.journal_rows(sm.root)
    assert row["nudge"] == 0.0 and row["held"] == held


async def test_greek_moves_nothing(client, tmp_path):
    sm = _model(tmp_path)
    await C.score_message("ευχαριστώ πολύ, τέλειο", sm.root, self_model=sm)
    assert sm.mood().p == 0.0
    (row,) = C.journal_rows(sm.root)
    assert row["label"] == "unknown" and row["nudge"] == 0.0 and row["held"] == "not English"


async def test_a_message_the_praise_check_counted_is_not_counted_twice(client, tmp_path):
    sm = _model(tmp_path)
    await C.score_message("thanks, perfect", sm.root, self_model=sm, praised=True)
    assert sm.mood().p == 0.0
    (row,) = C.journal_rows(sm.root)
    assert row["label"] == "positive" and row["held"] == C.HELD_PRAISED


async def test_without_a_mood_the_score_is_still_journalled(client, tmp_path):
    await C.score_message("great news today", tmp_path, self_model=None)
    (row,) = C.journal_rows(tmp_path)
    assert row["nudge"] == 0.0 and row["held"] == C.HELD_NO_MOOD


async def test_a_mood_that_fails_to_move_never_breaks_scoring(client, tmp_path):
    class Broken:
        def note_owner_sentiment(self, *a, **k):
            raise RuntimeError("state file locked")
    assert await C.score_message("great news today", tmp_path, self_model=Broken()) is True
    assert C.journal_rows(tmp_path)[0]["held"] == C.HELD_NO_MOOD


def test_the_mood_event():
    from ghost_agent.selfhood import mood as M
    ev = M.sentiment_event("positive", 0.85)
    assert (ev.dp, ev.de, ev.kind) == (0.08, 0.02, M.KIND_SENTIMENT)
    neg = M.sentiment_event("negative", 0.99)
    assert (neg.dp, neg.de) == (-0.08, -0.02)          # §4TX: bad news reads "down" (with him), not "irritable"
    for label, score in (("positive", 0.849), ("neutral", 0.99), ("unknown", 1.0), ("positive", True),
                         ("positive", "0.9"), ("positive", float("nan")), ("joy", 0.99)):
        assert M.sentiment_event(label, score) is None, (label, score)


def test_the_nudge_is_smaller_than_a_thumb_and_no_bigger_than_thanks():
    from ghost_agent.selfhood import mood as M
    assert M.EV_SENTIMENT_POSITIVE[0] <= M.EV_PRAISE[0] < M.EV_THUMB_UP[0]
    assert abs(M.EV_SENTIMENT_NEGATIVE[0]) < abs(M.EV_THUMB_DOWN[0])


async def test_upbeat_messages_tint_the_mood_up_to_the_sentiment_share(client, tmp_path):
    """§4UB: the owner's tone alone takes the pair no further than SENTIMENT_SHARE_MAX (under level 1)."""
    sm = _model(tmp_path)
    for _ in range(3):
        await C.score_message("finally got the promotion today!", sm.root, self_model=sm)
    v = sm.mood()
    assert v.p == pytest.approx(M.SENTIMENT_SHARE_MAX) and v.intensity == 0
    assert v.recent[-1]["cause"] == "the owner's message sounded upbeat"


# ── the health line ─────────────────────────────────────────────────────────

def test_health_says_nothing_when_not_configured(monkeypatch):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    assert C.health_text() is None


async def test_health_counts_scores_and_names_a_failed_last_call(client):
    await C.classify("great")
    assert C.health_text() == "sentiment service: 1 scored since start, 0 moved the mood"
    C.BREAKER.failed("ConnectError", C.BREAKER.last_ok + 30)
    assert "last call failed (ConnectError" in C.health_text(now_fn=lambda: C.BREAKER.last_ok + 60)


# ── the hook in handle_chat: owner messages only, never in the reply's way ──

def _hooked_agent(monkeypatch, tmp_path, client_env=True, shadow=None):
    from ghost_agent.selfhood import SelfModel
    from tests.test_requester_role import _marked_agent, _resp
    agent, ctx, _ = _marked_agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Here you go."))
    ctx.self_model = SelfModel(tmp_path / "selfhood")
    seen = []

    async def _fake_shadow(content, root, *, request_id="", tor_proxy=None, self_model=None, praised=None):
        seen.append((content, Path(root), request_id))
        calls.append({"self_model": self_model, "praised": praised})
        return True
    calls = []
    agent.sentiment_calls = calls
    monkeypatch.setattr(C, "score_message", shadow or _fake_shadow)
    if client_env:
        monkeypatch.setenv("GHOST_SENTIMENT_URL", "http://100.64.0.5:8097")
    else:
        monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    return agent, ctx, seen


async def _chat(agent, text, rid, role="owner"):
    from tests.helpers import FakeBgTasks
    await agent.handle_chat({"messages": [{"role": "user", "content": text}]},
                            FakeBgTasks(), request_id=rid, requester_role=role)
    await asyncio.sleep(0)


async def test_an_owner_message_is_scored_in_the_background(monkeypatch, tmp_path):
    agent, ctx, seen = _hooked_agent(monkeypatch, tmp_path)
    await _chat(agent, "thanks, the garden plan for Elena is perfect", "web-4tw-owner")
    assert seen == [("thanks, the garden plan for Elena is perfect", ctx.self_model.root, "web-4tw-owner")]
    assert agent.sentiment_calls == [{"self_model": ctx.self_model, "praised": False}]


async def test_the_hook_tells_the_scorer_when_the_praise_check_counted_the_message(monkeypatch, tmp_path):
    agent, ctx, _ = _hooked_agent(monkeypatch, tmp_path)
    await _chat(agent, "thanks, perfect", "web-4tw-praise")
    assert agent.sentiment_calls == [{"self_model": ctx.self_model, "praised": True}]


async def test_a_members_message_is_not_scored(monkeypatch, tmp_path):
    agent, _, seen = _hooked_agent(monkeypatch, tmp_path)
    await _chat(agent, "thanks, that is perfect", "slack-4tw-member", role="member")
    assert seen == []


async def test_nothing_is_scored_when_the_service_is_not_configured(monkeypatch, tmp_path):
    agent, _, seen = _hooked_agent(monkeypatch, tmp_path, client_env=False)
    await _chat(agent, "thanks, that is perfect", "web-4tw-off")
    assert seen == []


@pytest.mark.parametrize("rid", ["probe-4tw", "sched-4tw", "sub-4tw", "sim-4tw", "job-4tw", "bench-4tw",
                                 "replay-4tw"])
async def test_only_a_real_owner_request_is_scored(monkeypatch, tmp_path, rid):
    # fails if the predicate is any weaker than the one that moves the mood (e.g. "not a member", or "hears
    # the tone", which is True for probes, scheduled tasks and job wake-ups)
    agent, _, seen = _hooked_agent(monkeypatch, tmp_path)
    await _chat(agent, "thanks, that is perfect", rid)
    assert seen == []


async def test_the_reply_never_waits_for_the_score(monkeypatch, tmp_path):
    # fails if the hook awaits the call instead of starting it in the background
    gate = asyncio.Event()
    started = []

    async def _hanging(content, root, *, request_id="", tor_proxy=None, **_kw):
        started.append(request_id)
        await gate.wait()
        return True
    agent, _, _ = _hooked_agent(monkeypatch, tmp_path, shadow=_hanging)
    from tests.helpers import FakeBgTasks
    final, _, _ = await asyncio.wait_for(agent.handle_chat(
        {"messages": [{"role": "user", "content": "thanks, that is perfect"}]},
        FakeBgTasks(), request_id="web-4tw-hang", requester_role="owner"), 60)
    assert "Here you go." in str(final)
    await asyncio.sleep(0)
    assert started == ["web-4tw-hang"]
    gate.set()
    await asyncio.sleep(0)


async def test_a_failing_hook_never_breaks_the_turn(monkeypatch, tmp_path):
    agent, ctx, _ = _hooked_agent(monkeypatch, tmp_path)

    def _boom():
        raise RuntimeError("config read failed")
    monkeypatch.setattr(C, "settings", _boom)
    from tests.helpers import FakeBgTasks
    final, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "thanks"}]},
                                          FakeBgTasks(), request_id="web-4tw-boom", requester_role="owner")
    assert "Here you go." in str(final)


# ── the health check carries the line ───────────────────────────────────────

async def test_the_health_check_models_line_carries_the_sentiment_part(client):
    from types import SimpleNamespace
    from ghost_agent.tools import health_internals as H
    await C.classify("great")
    out = H._Parts()
    H._models(SimpleNamespace(llm_client=object()), out)
    assert "sentiment service: 1 scored since start, 0 moved the mood" in out.parts
    assert out.problems == []


# ── where the agent finds the service: --sentiment-url / --sentiment-key-file, env fallback ──

def _parse(monkeypatch, argv, env=None):
    from ghost_agent import main as gm
    for k in ("GHOST_SENTIMENT_URL", "GHOST_SENTIMENT_KEY_FILE"):
        monkeypatch.setenv(k, "")            # recorded, so parse_args' own writes are undone after the test
        monkeypatch.delenv(k)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["ghost"] + argv)
    return gm.parse_args()


def test_the_flag_sets_where_the_service_is(monkeypatch):
    args = _parse(monkeypatch, ["--sentiment-url", "http://100.64.0.5:8097", "--sentiment-key-file", "~/k"])
    assert args.sentiment_url == "http://100.64.0.5:8097"
    assert C.settings() == ("http://100.64.0.5:8097", str(Path("~/k").expanduser()))


def test_the_environment_is_the_fallback(monkeypatch):
    _parse(monkeypatch, [], env={"GHOST_SENTIMENT_URL": "http://100.64.0.6:8097"})
    assert C.settings() == ("http://100.64.0.6:8097", str(Path(C.DEFAULT_KEY_FILE).expanduser()))


def test_a_flag_beats_the_environment(monkeypatch):
    _parse(monkeypatch, ["--sentiment-url", "http://100.64.0.7:8097"], env={"GHOST_SENTIMENT_URL": "http://x:1"})
    assert C.settings()[0] == "http://100.64.0.7:8097"


def test_without_flag_or_environment_it_is_off(monkeypatch):
    _parse(monkeypatch, [])
    assert C.settings()[0] == "" and C.health_text() is None


# ── the log line: the rating with its own icon, never the words ─────────────

def test_the_sentiment_icon_is_its_own_and_wide():
    from ghost_agent.utils.logging import Icons, _icon_display_width
    others = [v for k, v in vars(Icons).items() if k.isupper() and k != "SENTIMENT" and isinstance(v, str)]
    assert Icons.SENTIMENT == "💗" and Icons.SENTIMENT not in others
    assert _icon_display_width(Icons.SENTIMENT) == 2


# (both clients knowing the icon: test_web_icon_map_drift.py and test_clockwork_turnstatus.py, for every icon)


@pytest.mark.parametrize("answer,line", [
    ({"label": "positive", "score": 0.9529, "probs": {"negative": 0.0088, "neutral": 0.0383, "positive": 0.9529}},
     "positive 0.95 (neg 0.01 · neu 0.04) · mood +0.08"),
    ({"label": "negative", "score": 0.907, "probs": {"negative": 0.907, "neutral": 0.0874, "positive": 0.0056}},
     "negative 0.91 (neu 0.09 · pos 0.01) · mood −0.08"),
    ({"label": "unknown", "score": 0.0, "probs": {}, "reason": "not English"}, "unknown — not English · mood unmoved"),
])
def test_the_rating_line(answer, line):
    nudge = {"positive": 0.08, "negative": -0.08}.get(answer["label"], 0.0)
    assert C.rating_line(answer, nudge, "" if nudge else "not English") == line


def test_the_rating_line_says_why_the_mood_did_not_move():
    answer = {"label": "positive", "score": 0.84, "probs": {"negative": 0.06, "neutral": 0.10, "positive": 0.84}}
    assert C.rating_line(answer, 0.0, "below 0.85") == "positive 0.84 (neg 0.06 · neu 0.10) · mood unmoved (below 0.85)"


async def test_a_scored_message_logs_its_rating_with_the_icon_and_not_its_words(client, tmp_path, monkeypatch):
    from ghost_agent.utils import logging as L
    calls = []
    monkeypatch.setattr(L, "pretty_log", lambda title, content=None, icon="", **kw: calls.append((title, content, icon)))
    await C.score_message("Markos loved the new garden", tmp_path)
    assert calls == [("Sentiment", "positive 0.85 (neg 0.05 · neu 0.10) · mood unmoved (no mood to move)", "💗")]


async def test_no_answer_logs_no_rating(tmp_path, monkeypatch):
    from ghost_agent.utils import logging as L
    calls = []
    monkeypatch.setattr(L, "pretty_log", lambda *a, **k: calls.append(a))
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    await C.score_message("great", tmp_path)
    assert calls == []


def test_a_failing_log_line_never_raises(monkeypatch):
    from ghost_agent.utils import logging as L

    def _boom(*a, **k):
        raise RuntimeError("stream closed")
    monkeypatch.setattr(L, "pretty_log", _boom)
    C.log_rating({"label": "positive", "score": 0.9, "probs": {}})


# ── round-2 review fixes ────────────────────────────────────────────────────

async def test_a_missing_key_is_said_on_the_health_line(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_SENTIMENT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(tmp_path / "absent"))
    assert await C.classify("great") is None
    assert "last call failed (no key in the key file" in C.health_text(now_fn=lambda: C.BREAKER.last_fail + 1)


@pytest.mark.parametrize("name", ["v1.].txt", "a.] b.].pdf", "x'.].png"])
def test_a_file_name_holding_dot_bracket_leaves_no_note_behind(name):
    for note in (f"[SYSTEM NOTE: The user attached a file named '{name}'. It has been uploaded to your sandbox. "
                 f"Use your file_system or knowledge_base tools to interact with it.]",
                 f"[SYSTEM NOTE: The user attached a file named '{name}'. If it is an image, inspect it with "
                 f"vision_analysis (target='{name}').]",
                 f"[SYSTEM NOTE: The user attached a file named '{name}' but it could NOT be ingested (too "
                 f"large or the fetch failed).]"):
        assert C.message_text(f"ok\n\n{note}") == "ok", note


def test_journal_writers_take_turns(tmp_path):
    # fails if two writers can reach the size check at once (both moved the full file over ".1")
    done = threading.Event()
    with C._JOURNAL_LOCK:
        t = threading.Thread(target=lambda: (C._append(tmp_path / "j.jsonl", {"a": 1}), done.set()))
        t.start()
        assert not done.wait(0.3)
    assert done.wait(5)
    t.join()


def test_the_key_is_redacted_from_stored_text(monkeypatch, tmp_path):
    from ghost_agent.distill import redact
    secret = "sentiment-service-key-abcdefghijklmnop"
    key = tmp_path / "key"
    key.write_text(secret)
    monkeypatch.setenv("GHOST_SENTIMENT_KEY_FILE", str(key))
    rule = redact._sentiment_key_rule()
    assert rule and rule[1].sub(rule[2], f"header {secret}") == "header <REDACTED_SENTIMENT_KEY>"


# ── round-3 review fixes ────────────────────────────────────────────────────

async def test_a_score_that_lands_after_the_owner_set_the_mood_does_not_move_it(client, tmp_path, monkeypatch):
    # "cheer up, you're grumpy": the owner's set_mood lands while the score is still on its way
    from ghost_agent.selfhood import mood as M
    _, clf = client
    clf._probs = [0.90, 0.07, 0.03]
    sm = _model(tmp_path)
    real = C.classify

    async def _slow(text, **kw):
        sm.state.set_owner_mood("cheerful", M.resolve_label("cheerful")[1], 2)
        return await real(text, **kw)
    monkeypatch.setattr(C, "classify", _slow)
    await C.score_message("cheer up, you're grumpy", sm.root, self_model=sm)
    v = sm.mood()
    assert (v.label, v.intensity, v.source) == ("cheerful", 2, M.SOURCE_OWNER)
    assert v.recent[-1]["kind"] == M.KIND_OWNER
    from ghost_agent.selfhood.state import SelfStateThread
    assert C.journal_rows(sm.root)[0]["held"] == SelfStateThread.HELD_OWNER_HOLD


async def test_a_score_that_lands_after_the_owner_said_neutral_does_not_undo_it(client, tmp_path, monkeypatch):
    """§4TX: "neutral" holds nothing, but a message sent BEFORE it must not move the mood after it."""
    from ghost_agent.selfhood import mood as M
    from ghost_agent.selfhood.state import SelfStateThread
    _, clf = client
    clf._probs = [0.90, 0.07, 0.03]
    sm = _model(tmp_path)
    sm.state.nudge(-0.5, 0.2, "trouble", M.KIND_TURN)
    real = C.classify

    async def _slow(text, **kw):
        sm.state.reset_mood()
        return await real(text, **kw)
    monkeypatch.setattr(C, "classify", _slow)
    await C.score_message("ugh, just stop, be normal", sm.root, self_model=sm)
    assert sm.mood().neutral
    assert C.journal_rows(sm.root)[0]["held"] == SelfStateThread.HELD_OWNER_LATER


async def test_a_mood_the_owner_set_is_held_and_later_messages_move_it_once_the_hold_is_over(client, tmp_path):
    """§4TX: the owner's set mood holds until the day ends — a score meanwhile is journalled as held."""
    from ghost_agent.selfhood import mood as M
    from ghost_agent.selfhood.state import SelfStateThread
    import datetime as dt
    sm = _model(tmp_path)
    earlier = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)
    sm.state.set_owner_mood("calm", M.resolve_label("calm")[1], 1, now=earlier)
    before = sm.mood().p
    await C.score_message("finally got the promotion today!", sm.root, self_model=sm)
    assert sm.mood().p == before
    assert C.journal_rows(sm.root)[0]["held"] == SelfStateThread.HELD_OWNER_HOLD
    sm.state.state.mood.held_until = ""                                   # the day is over
    await C.score_message("finally got the promotion today!", sm.root, self_model=sm, request_id="r2")
    assert sm.mood().p == pytest.approx(before + 0.08, abs=0.02)
    assert C.journal_rows(sm.root)[1]["held"] == ""


async def test_the_whole_call_has_a_cap_whatever_httpx_does_per_stage(client, monkeypatch):
    import httpx
    monkeypatch.setattr(C, "TOTAL_TIMEOUT_S", 0.2)

    async def _trickle(self, *a, **k):
        await asyncio.sleep(10)
    monkeypatch.setattr(httpx.AsyncClient, "post", _trickle)
    assert await asyncio.wait_for(C.classify("great"), 5) is None
    assert "Timeout" in C.BREAKER.last_fail_why
