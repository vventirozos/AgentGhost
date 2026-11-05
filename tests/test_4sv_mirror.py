"""§4SV: the Mirror — a daily I-sentence (toy strange loop), inert, main model at background priority."""
import asyncio
import datetime
import json
import random
from pathlib import Path

import pytest

from ghost_agent.selfhood import mirror as M

NOW = datetime.datetime(2026, 11, 3, 14, 0, 0)


class _LLM:
    def __init__(self, content="I am a careful helper who checks the news."):
        self.content = content
        self.calls = []

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        return {"choices": [{"message": {"content": self.content}}]}


def _reply(text, at="2026-11-03T09:00:00Z"):
    return {"text": text, "at": at}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "pick_reply", lambda h, now=None, rng=None: _reply(
        "Here's the news — past 24 hours. Greece: heavy rain and thunderstorms across the country."))
    return tmp_path


def test_first_look_starts_from_the_seed_and_writes_one_row(home):
    llm = _LLM()
    note = asyncio.run(M.look(home, llm, now=NOW))
    assert note == "new I-sentence: I am a careful helper who checks the news."
    rows = M.load(home)
    assert len(rows) == 1 and rows[0]["date"] == "2026-11-03"
    assert rows[0]["prev"] == M.SEED_SENTENCE
    prompt = llm.calls[0][0]["messages"][-1]["content"]
    assert M.SEED_SENTENCE in prompt and "heavy rain" in prompt


def test_second_day_feeds_yesterdays_sentence_back(home):
    asyncio.run(M.look(home, _LLM("I am a news reader."), now=NOW))
    llm = _LLM("I am a news reader who also builds games.")
    asyncio.run(M.look(home, llm, now=NOW + datetime.timedelta(days=1)))
    assert "«I am a news reader.»" in llm.calls[0][0]["messages"][-1]["content"]
    assert [r["sentence"] for r in M.load(home)] == ["I am a news reader.",
                                                     "I am a news reader who also builds games."]


def test_once_a_day(home):
    asyncio.run(M.look(home, _LLM(), now=NOW))
    llm = _LLM()
    assert asyncio.run(M.look(home, llm, now=NOW + datetime.timedelta(hours=3))) == "already looked today"
    assert llm.calls == [] and len(M.load(home)) == 1


def test_unchanged_sentence_is_said(home):
    asyncio.run(M.look(home, _LLM("I am steady."), now=NOW))
    note = asyncio.run(M.look(home, _LLM("I am steady."), now=NOW + datetime.timedelta(days=1)))
    assert note == "unchanged: I am steady."


def test_main_model_background_call_thinking_off(home):
    """Operator 11-02: the MAIN model (the one that wrote the replies) reads
    them — at background priority, so the conversation is saved first (§4RV)."""
    llm = _LLM()
    asyncio.run(M.look(home, llm, now=NOW, model="qwen3.8-27b"))
    payload, kw = llm.calls[0]
    assert kw["is_background"] and not kw.get("use_worker") and not kw.get("off_main_only")
    assert payload["model"] == "qwen3.8-27b"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][-1]["content"].endswith("/no_think")


def test_no_model_no_call(home):
    assert asyncio.run(M.look(home, None, now=NOW)).startswith("skipped — no model")
    assert M.load(home) == []


def test_no_reply_no_call(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "pick_reply", lambda h, now=None, rng=None: None)
    llm = _LLM()
    assert "no reply" in asyncio.run(M.look(tmp_path, llm, now=NOW))
    assert llm.calls == []


def test_empty_model_answer_writes_nothing(home):
    assert asyncio.run(M.look(home, _LLM("   "), now=NOW)).startswith("no sentence")
    assert M.load(home) == []


@pytest.mark.parametrize("raw,want", [
    ("«I am a lamp.»", "I am a lamp."),
    ('"I am a lamp."\n\nHope this helps!', "I am a lamp."),
    ("Sentence: I am a lamp.", "I am a lamp."),
    ("<think>hmm</think>I am a lamp.", "I am a lamp."),
    ("", ""),
])
def test_clean_sentence(raw, want):
    assert M.clean_sentence(raw) == want


def test_clean_sentence_caps_length():
    s = M.clean_sentence("I am " + "very " * 200 + "long.")
    assert len(s) <= M.SENTENCE_MAX_CHARS + 1 and s.endswith("…")


def test_pick_reply_skips_short_aborted_and_old(tmp_path, monkeypatch):
    import ghost_agent.core.failure_census as FC
    now_utc = NOW.astimezone(datetime.timezone.utc)

    def ts(hours_ago):
        return (now_utc - datetime.timedelta(hours=hours_ago)).isoformat()
    turns = [{"final_response": "ok", "timestamp": ts(1)},
             {"final_response": "x" * 100 + " [ATTEMPT_ABORTED_TIMEOUT]", "timestamp": ts(1)},
             {"final_response": "An old answer about a museum visit, long enough to count. " * 2,
              "timestamp": ts(60)},                                   # outside the 48 h
             {"final_response": "A real answer about the weather in Athens, rain all week. " * 2,
              "timestamp": ts(3)}]
    seen = {}

    def fake(sysd, days):
        seen["days"] = days
        return turns, []
    monkeypatch.setattr(FC, "_owner_turns", fake)
    for seed in range(5):
        got = M.pick_reply(tmp_path, now=NOW, rng=random.Random(seed))
        assert "Athens" in got["text"]
    # day folders are UTC dates, read with one day to spare (review m3)
    d = now_utc.date()
    assert seen["days"] == [(d - datetime.timedelta(days=k)).isoformat() for k in range(3)]


def test_pick_reply_reads_real_day_folders_owner_only(tmp_path):
    """Real trajectory files: a member's, a probe's and a job's reply are never picked."""
    now_utc = NOW.astimezone(datetime.timezone.utc)
    day = tmp_path / "system" / "trajectories" / now_utc.date().isoformat()
    day.mkdir(parents=True)
    at = (now_utc - datetime.timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    body = " — a reply long enough to be picked by the mirror: rain in Athens all week, 14 °C."

    def row(i, text, **extra):
        return {"id": f"t{i}", "timestamp": at, "session_id": "s", "task_kind": "user_request",
                "user_request": "q", "final_response": text + body, "extra": extra}
    rows = [row(1, "OWNER", req_id="abc123", origin="user"),
            row(2, "MEMBER", req_id="slack-1", requester_role="member"),
            row(3, "PROBE", req_id="probe-1"),
            row(4, "JOB", req_id="job-1"),
            row(5, "SIM", req_id="x", origin="sim")]
    (day / "session-s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    picks = {M.pick_reply(tmp_path, now=NOW, rng=random.Random(k))["text"].split(" ")[0] for k in range(12)}
    assert picks == {"OWNER"}


def test_torn_diary_line_is_skipped(tmp_path):
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-11-01", "sentence": "I am one.", "prev": "x"}) + "\n{\"date\": \"20")
    assert [r["sentence"] for r in M.load(tmp_path)] == ["I am one."]
    assert M.current_sentence(tmp_path) == "I am one."


@pytest.mark.parametrize("text,yes", [("mirror", True), ("Mirror.", True), (" /mirror ", True),
                                      ("mirror the repo", False), ("show the mirror", False)])
def test_command_matcher(text, yes):
    assert M.is_mirror_command(text) is yes


def test_banner_empty_and_filled(tmp_path):
    assert M.SEED_SENTENCE in M.mirror_banner(tmp_path)
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    rows = [{"date": f"2026-10-{d:02d}", "sentence": f"I am day {d}.", "prev": f"I am day {d - 1}.",
             "reply_excerpt": "the news", "reply_at": "2026-10-20T09:00"} for d in range(1, 11)]
    rows[-1]["prev"] = rows[-1]["sentence"]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    b = M.mirror_banner(tmp_path)
    assert "I am day 10." in b and "I am day 3." not in b      # last 7 only
    assert "(unchanged)" in b and "10 day(s) so far" in b and "the news" in b
    assert "latest look (2026-10-10)" in b and "Today's" not in b


@pytest.mark.parametrize("kind,public,want", [("owner", False, True), ("member", False, False),
                                               ("owner", True, False)])
def test_owner_command_is_answered_by_the_system(tmp_path, monkeypatch, kind, public, want):
    """The `mirror` command rides the §4PY path: a note whose banner ends the
    turn — the owner's private surface only (never a member, never public)."""
    from types import SimpleNamespace
    from ghost_agent.core import agent as A
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(L, "request_kind", lambda: kind)
    monkeypatch.setattr(A, "reply_is_public", lambda: public)
    monkeypatch.setattr(A.logger, "warning", lambda *a, **k: pytest.fail(f"note raised: {a}"))
    fake = SimpleNamespace(context=SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory")))
    # a real diary row at the real path: the command must find it (r3: an empty
    # diary shows the seed whatever home the command resolves)
    diary = tmp_path / "system" / "selfhood" / "mirror.jsonl"
    diary.parent.mkdir(parents=True)
    diary.write_text(json.dumps({"date": "2026-11-02", "sentence": "I am day 1.", "prev": "x"}) + "\n")
    note = asyncio.run(A.GhostAgent._owner_rule_note(fake, {"messages": [{"role": "user", "content": "mirror"}]}))
    assert bool(note) is want
    if want:
        assert "The Mirror" in note.banner and A.system_command_reply(note.banner).startswith("**The Mirror**")
        assert "I am day 1." in note.banner


def test_registered_as_an_idle_job_with_a_purpose():
    from ghost_agent.core import agent as A
    assert "mirror" in [lbl for lbl, _t, _e in A.IDLE_SCHEDULER_JOBS]
    assert "mirror" in A.IDLE_JOB_PURPOSE


def test_the_sentence_reaches_no_prompt():
    """Inert by design: only core/agent.py (the idle phase and the command) imports the module."""
    import re
    src = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    pat = re.compile(r"^\s*(?:from\s+\S*selfhood(?:\.mirror)?\s+import\s+[^\n]*\bmirror\b"
                     r"|from\s+\S*selfhood\.mirror\s+import|import\s+\S*selfhood\.mirror"
                     r"|from\s+\.\s+import\s+[^\n]*\bmirror\b)", re.M)
    users = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                   if p.name != "mirror.py" and pat.search(p.read_text(encoding="utf-8")))
    assert users == ["core/agent.py"]


# ── the idle phase (review m5) ──────────────────────────────────────────────

def _agent(tmp_path, llm):
    from types import SimpleNamespace
    from ghost_agent.core import agent as A
    a = A.GhostAgent.__new__(A.GhostAgent)
    a._bio_cooldown = lambda s: s
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory"), llm_client=llm)
    return a, ctx


def test_phase_looks_once_then_waits_for_its_cooldown(home):
    llm = _LLM()
    a, ctx = _agent(home, llm)
    ran, notes = [], {}
    asyncio.run(a._mirror_phase(ctx, ran, notes))
    assert ran == ["mirror"] and notes["mirror"].startswith("new I-sentence")
    ran2, notes2 = [], {}
    asyncio.run(a._mirror_phase(ctx, ran2, notes2))          # inside the hour: nothing
    assert ran2 == [] and len(llm.calls) == 1
    a._last_mirror_try_at = datetime.datetime.min             # hour passed, same day: not listed
    asyncio.run(a._mirror_phase(ctx, ran2, notes2))
    assert ran2 == [] and len(llm.calls) == 1


def test_phase_books_a_deferred_call_and_survives(home):
    class _Down(_LLM):
        async def chat_completion(self, payload, **kw):
            raise RuntimeError("BackgroundDeferred: a request started")
    a, ctx = _agent(home, _Down())
    ran, notes = [], {}
    asyncio.run(a._mirror_phase(ctx, ran, notes))
    assert ran == ["mirror(failed)"] and M.load(home) == []


def test_phase_books_a_deferral_as_a_deferral_and_retries_next_tick(home):
    """Review r2 m1: a request starting is a guard, not a failure (§4PM)."""
    from ghost_agent.core.llm import BackgroundDeferred

    class _Busy(_LLM):
        async def chat_completion(self, payload, **kw):
            raise BackgroundDeferred("background call deferred: mirror (a request started)")
    a, ctx = _agent(home, _Busy())
    ran, notes = [], {}
    asyncio.run(a._mirror_phase(ctx, ran, notes))
    assert ran == ["mirror"] and notes["mirror"].startswith("deferred")
    assert a._last_mirror_try_at == datetime.datetime.min      # the next tick may try again
    ctx.llm_client = _LLM()
    asyncio.run(a._mirror_phase(ctx, [], {}))
    assert len(M.load(home)) == 1


def test_file_work_runs_off_the_event_loop(home, monkeypatch):
    """Review m2: the diary/trajectory reads and the append go through to_thread."""
    called = []
    real = asyncio.to_thread

    async def counting(fn, *a, **k):
        called.append(fn.__name__)
        return await real(fn, *a, **k)
    monkeypatch.setattr(asyncio, "to_thread", counting)
    asyncio.run(M.look(home, _LLM(), now=NOW))
    assert {"looked_today", "<lambda>", "current_sentence", "_append"} <= set(called)


def test_home_is_two_levels_above_memory_dir(tmp_path):
    """Live bug 11-02: memory_dir is $GHOST_HOME/system/memory — its parent is system/."""
    assert M.home_from_memory_dir(str(tmp_path / "system" / "memory")) == tmp_path
    assert M.home_from_memory_dir(None) is None and M.home_from_memory_dir("") is None


def test_phase_finds_replies_in_the_real_layout(tmp_path):
    """End to end over the real directory layout (no pick_reply stub): the
    phase reads $GHOST_HOME/system/trajectories and writes system/selfhood/mirror.jsonl."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    day = tmp_path / "system" / "trajectories" / now_utc.date().isoformat()
    day.mkdir(parents=True)
    at = (now_utc - datetime.timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    r = {"id": "t1", "timestamp": at, "session_id": "s", "task_kind": "user_request", "user_request": "q",
         "final_response": "Here is the weather for Athens this week: rain every day, around 14 °C, with a strong wind from the north.",
         "extra": {"req_id": "abc123", "origin": "user"}}
    (day / "session-s.jsonl").write_text(json.dumps(r) + "\n")
    llm = _LLM("I am a weather reader.")
    a, ctx = _agent(tmp_path, llm)
    ran, notes = [], {}
    asyncio.run(a._mirror_phase(ctx, ran, notes))
    assert notes["mirror"] == "new I-sentence: I am a weather reader.", notes
    assert (tmp_path / "system" / "selfhood" / "mirror.jsonl").is_file()
    assert "Athens" in llm.calls[0][0]["messages"][-1]["content"]
