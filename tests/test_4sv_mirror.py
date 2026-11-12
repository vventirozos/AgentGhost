"""§4SV / §4SV-b: the Mirror — a daily I-sentence (toy strange loop) on the main model at background
priority; the latest sentence is one line of `introspect` (and, §4TZ, of the owner's state block —
tests/test_4tz_mirror_loop.py)."""
import asyncio
import datetime
import json
import random
from pathlib import Path

import pytest

from ghost_agent.selfhood import mirror as M


@pytest.fixture(autouse=True)
def _the_mirrors_own_path(monkeypatch):
    """§4UG: these tests are about the Mirror's own sentence — what a reply hears with the character
    switched off (``GHOST_CHARACTER=0``). The character has its tests in test_4ug_character.py."""
    monkeypatch.setenv("GHOST_CHARACTER", "0")


NOW = datetime.datetime(2026, 11, 3, 14, 0, 0)


class _LLM:
    def __init__(self, content="I am a careful helper who checks the news."):
        self.content = content
        self.calls = []

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        if "temperament rest" in payload["messages"][-1]["content"]:     # §4UA: the resting-point question
            return {"choices": [{"message": {"content": "0 0"}}]}
        return {"choices": [{"message": {"content": self.content}}]}


def _reply(text, at="2026-11-03T09:00:00Z"):
    return {"text": text, "at": at}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "pick_replies", lambda h, now=None, rng=None, after="": [_reply(
        "Here's the news — past 24 hours. Greece: heavy rain and thunderstorms across the country.")])
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


def test_second_day_feeds_the_sentences_so_far_back(home):
    asyncio.run(M.look(home, _LLM("I am a news reader."), now=NOW))
    llm = _LLM("I am a news reader who also builds games.")
    asyncio.run(M.look(home, llm, now=NOW + datetime.timedelta(days=1)))
    prompt = llm.calls[0][0]["messages"][-1]["content"]
    # the seed first, then the dated sentence — oldest first
    assert prompt.index(f"- before the first look: «{M.SEED_SENTENCE}»") \
        < prompt.index("- 2026-11-03: «I am a news reader.»") < prompt.index("heavy rain")
    assert [r["sentence"] for r in M.load(home)] == ["I am a news reader.",
                                                     "I am a news reader who also builds games."]


def test_the_third_day_is_shown_both_earlier_sentences(home):
    """The model sees its sentences so far, not only the last one."""
    for day, sent in enumerate(("I am one.", "I am two.")):
        asyncio.run(M.look(home, _LLM(sent), now=NOW + datetime.timedelta(days=day)))
    llm = _LLM("I am three.")
    asyncio.run(M.look(home, llm, now=NOW + datetime.timedelta(days=2)))
    prompt = llm.calls[0][0]["messages"][-1]["content"]
    assert prompt.index("- 2026-11-03: «I am one.»") < prompt.index("- 2026-11-04: «I am two.»")


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
    monkeypatch.setattr(M, "pick_replies", lambda h, now=None, rng=None, after="": [])
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


def test_pick_replies_skips_short_aborted_and_old(tmp_path, monkeypatch):
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
        got = M.pick_replies(tmp_path, now=NOW, rng=random.Random(seed))
        assert len(got) == 1 and "Athens" in got[0]["text"]
    # day folders are UTC dates, read with one day to spare (review m3)
    d = now_utc.date()
    assert seen["days"] == [(d - datetime.timedelta(days=k)).isoformat() for k in range(3)]


def test_pick_replies_reads_real_day_folders_owner_only(tmp_path):
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
    for k in range(12):
        got = M.pick_replies(tmp_path, now=NOW, rng=random.Random(k))
        assert [g["text"].split(" ")[0] for g in got] == ["OWNER"]


def test_torn_diary_line_is_skipped(tmp_path):
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-11-01", "sentence": "I am one.", "prev": "x", "replies": []})
                 + "\n{\"date\": \"20")
    assert [r["sentence"] for r in M.load(tmp_path)] == ["I am one."]
    assert M.self_image_line(tmp_path).startswith("self-image: «I am one.»")


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
    assert "(unchanged)" in b and "10 day(s) so far" in b and "\n> (2026-10-20 " in b and "the news…" in b
    assert "latest look (2026-10-10) was at this reply:" in b and "Today's" not in b
    assert "Nothing here reaches my answers" not in b          # §4SV-b: no longer true


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


def test_the_sentence_reaches_introspect_and_the_owner_state_block_only():
    """§4SV-b: the idle phase and the command (core/agent.py) and the one
    `introspect` line (tools/introspect.py) are the module's only importers —
    no prompt builder, no memory writer. §4TZ: the state-block reader
    (`self_image_section` / `self_image_block`) is named by core/agent.py and nowhere else (its one
    owner-gated section). Read from the syntax tree: an import in a comment
    or a string is not an import."""
    import ast
    src = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    importers, taken, namers, block_namers = set(), {}, set(), set()
    for p in src.rglob("*.py"):
        rel = str(p.relative_to(src))
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mod, names = n.module or "", [a.name for a in n.names]
                if mod.split(".")[-2:] == ["selfhood", "mirror"]:
                    importers.add(rel)
                    taken.setdefault(rel, set()).update(names)
                elif (mod.split(".")[-1:] == ["selfhood"] or (not mod and rel.startswith("selfhood/"))) \
                        and "mirror" in names:
                    importers.add(rel)
            elif isinstance(n, ast.Import) and any(a.name.endswith("selfhood.mirror") for a in n.names):
                importers.add(rel)
            # who names the reader, or introspect's wrapper of it (review r1: a prompt
            # builder importing `_self_image` FROM INTROSPECT passed an import-only pin)
            ident = (n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute)
                     else n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.alias)) else "")
            if ident in ("self_image_line", "_self_image"):
                namers.add(rel)
            if ident in ("self_image_block", "self_image_section", "_self_image_block"):
                block_namers.add(rel)
    # §4UA: main.py attaches the character's resting point to the mood (`rest_pair`)
    assert importers - {"selfhood/mirror.py"} == {"core/agent.py", "tools/introspect.py", "api/routes.py",  # §4UG: health's character block
                                                  "memory/experience.py",
                                                  # §4UH: the look borrows `pick_replies` (the owner's
                                                  # replies since a time), never the diary or the sentence
                                                  "selfhood/selfview.py",
                                                  # §4UH-e2e: the character's resting point falls back to the
                                                  # Mirror's (`rest_pair`, `REST_UNIT`) only with the character off
                                                  "selfhood/character.py",
                                                 "main.py"}
    assert taken["tools/introspect.py"] == {"home_from_memory_dir", "self_image_line"}   # the one-line reader only
    # §4SZ: the experience store asks whether a message is the "mirror" COMMAND (a system command is
    # not an experience) — the matcher only, nothing that reads the diary
    assert taken["memory/experience.py"] == {"is_mirror_command"}
    assert namers == {"selfhood/mirror.py", "tools/introspect.py"}
    assert block_namers == {"selfhood/mirror.py", "core/agent.py"}
    assert "self_image_section" in taken["core/agent.py"]


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
    # §4UA: one look = the sentence call + the resting-point question
    assert len(llm.calls) == 2 and "temperament rest" in llm.calls[1][0]["messages"][1]["content"]
    asyncio.run(a._mirror_phase(ctx, ran2, notes2))          # inside the hour: nothing
    assert ran2 == [] and len(llm.calls) == 2
    a._last_mirror_try_at = datetime.datetime.min             # hour passed, same day: not listed
    asyncio.run(a._mirror_phase(ctx, ran2, notes2))
    assert ran2 == [] and len(llm.calls) == 2


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
    assert {"looked_today", "load", "<lambda>", "_append"} <= set(called)


def test_home_is_two_levels_above_memory_dir(tmp_path):
    """Live bug 11-02: memory_dir is $GHOST_HOME/system/memory — its parent is system/."""
    assert M.home_from_memory_dir(str(tmp_path / "system" / "memory")) == tmp_path
    assert M.home_from_memory_dir(None) is None and M.home_from_memory_dir("") is None


def test_phase_finds_replies_in_the_real_layout(tmp_path):
    """End to end over the real directory layout (no pick_replies stub): the
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


# ── §4SV-b: the prompt (wording chosen by replay), three replies, the introspect line ──

def _rows(*pairs):
    return [{"date": d, "sentence": s} for d, s in pairs]


def test_history_shows_the_seed_and_one_line_per_run_of_unchanged_days():
    rows = _rows(("2026-11-01", M.SEED_SENTENCE), ("2026-11-02", "I check first."),
                 ("2026-11-03", "I check first."), ("2026-11-04", "I check first."),
                 ("2026-11-05", "I check, then ask."))
    assert M.history_lines(rows).splitlines() == [
        f"- before the first look to 2026-11-01: «{M.SEED_SENTENCE}»",
        "- 2026-11-02 to 2026-11-04: «I check first.»",
        "- 2026-11-05: «I check, then ask.»"]
    assert M.history_lines([]) == f"- before the first look: «{M.SEED_SENTENCE}»"


def test_history_keeps_the_last_seven_lines():
    rows = _rows(*[(f"2026-10-{d:02d}", f"I am day {d}.") for d in range(1, 11)])
    lines = M.history_lines(rows).splitlines()
    assert len(lines) == M.SHOW_LAST == 7
    assert lines[0] == "- 2026-10-04: «I am day 4.»" and lines[-1] == "- 2026-10-10: «I am day 10.»"


def test_a_sentence_that_comes_back_later_is_its_own_line():
    rows = _rows(("2026-11-01", "I check."), ("2026-11-02", "I ask."), ("2026-11-03", "I check."))
    assert M.history_lines(rows).count("«I check.»") == 2


def test_the_prompt_asks_about_the_writer_not_what_it_wrote_about():
    """The wording the operator picked from the replay samples (journal
    §4SV-b): the subject is left out, what the way the replies were written
    shows about their writer; the number words follow the reply count."""
    many = M.build_prompt([], ["first reply", "second reply", "third reply"])
    assert "1. «first reply»\n2. «second reply»\n3. «third reply»" in many
    assert "you wrote these replies to the person you work for" in many
    assert ("not about what these replies happen to be about: leave their subject matter out and "
            "say what the way they were written shows about the one who wrote them — only what is "
            "really there. Keep what still holds; change what the replies show differently.") in many
    one = M.build_prompt([], ["only reply"])
    assert "you wrote this reply to the person you work for:\n«only reply»" in one
    assert ("not about what this reply happens to be about: leave its subject matter out and "
            "say what the way it was written shows about the one who wrote it — only what is really "
            "there.") in one and "change what the reply shows differently" in one
    for p in (many, one):
        assert "{" not in p and "}" not in p and "Yesterday" not in p
        assert p.rstrip().endswith("Reply with the sentence only.")


def test_a_look_shows_three_replies_cut_short_and_records_them(tmp_path, monkeypatch):
    picked = [_reply(f"R{i} " + "x" * 900, at=f"2026-11-03T0{i}:00:00Z") for i in (1, 2, 3)]
    monkeypatch.setattr(M, "pick_replies", lambda h, now=None, rng=None, after="": picked)
    llm = _LLM("I check, then answer.")
    asyncio.run(M.look(tmp_path, llm, now=NOW))
    prompt = llm.calls[0][0]["messages"][-1]["content"]
    for i in (1, 2, 3):
        shown = prompt.split(f"{i}. «", 1)[1].split("»", 1)[0]
        assert shown.startswith(f"R{i} ") and len(shown) == M.REPLY_MAX_CHARS == 600
    row = M.load(tmp_path)[0]
    assert [r["at"] for r in row["replies"]] == [p["at"] for p in picked]
    assert all(len(r["excerpt"]) == 200 for r in row["replies"])
    assert "reply_excerpt" not in row and "reply_at" not in row


def test_the_next_look_is_told_when_the_last_one_was(tmp_path, monkeypatch):
    """The LAST row's time, not the first's (review r1: one prior row could not tell them apart)."""
    seen = []

    def fake(h, now=None, rng=None, after=""):
        seen.append(after)
        return [_reply("A reply long enough, about anything at all, to be shown here.")]
    monkeypatch.setattr(M, "pick_replies", fake)
    for day, sent in enumerate(("I am one.", "I am two.", "I am three.")):
        asyncio.run(M.look(tmp_path, _LLM(sent), now=NOW + datetime.timedelta(days=day)))
    assert seen == ["", "2026-11-03T14:00:00", "2026-11-04T14:00:00"]


def _pool(monkeypatch, hours_ago):
    import ghost_agent.core.failure_census as FC
    now_utc = NOW.astimezone(datetime.timezone.utc)
    turns = [{"final_response": f"Reply {h}h: " + "a real answer, long enough to be picked. " * 3,
              "timestamp": (now_utc - datetime.timedelta(hours=h)).isoformat()} for h in hours_ago]
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, days: (turns, []))


def _hours(got):
    return sorted(int(g["text"].split("h:")[0].split()[-1]) for g in got)


def test_pick_replies_takes_six_different_ones_at_random(tmp_path, monkeypatch):
    """§4TZ: six per look (was three)."""
    _pool(monkeypatch, range(1, 13))
    draws = set()
    for seed in range(20):
        got = M.pick_replies(tmp_path, now=NOW, rng=random.Random(seed))
        assert len(got) == M.REPLIES_PER_LOOK == 6 and len({g["at"] for g in got}) == 6
        draws.add(tuple(_hours(got)))
    assert len(draws) > 5                                      # a draw, not the first three of the window


def test_pick_replies_only_takes_replies_written_since_the_last_look(tmp_path, monkeypatch):
    """The prompt says "since the last one, you wrote these replies": a reply
    older than the last sentence is not evidence against it."""
    _pool(monkeypatch, [1, 5, 9, 13, 20, 30])
    after = (NOW - datetime.timedelta(hours=10)).isoformat(timespec="seconds")   # a row's "at": local, naive
    for seed in range(10):
        assert _hours(M.pick_replies(tmp_path, now=NOW, rng=random.Random(seed), after=after)) == [1, 5, 9]
    # the look itself is the boundary: a reply stamped at that second was already there
    edge = (NOW - datetime.timedelta(hours=9)).isoformat(timespec="seconds")
    assert _hours(M.pick_replies(tmp_path, now=NOW, rng=random.Random(0), after=edge)) == [1, 5]


@pytest.mark.parametrize("after", ["", None, "not a time", "2026-10-01T00:00:00"])
def test_pick_replies_goes_two_days_back_at_most(tmp_path, monkeypatch, after):
    """No last look, an unreadable one, or one older than 48 h: the 48 h window."""
    _pool(monkeypatch, [1, 47, 49, 80])
    for seed in range(6):
        assert _hours(M.pick_replies(tmp_path, now=NOW, rng=random.Random(seed), after=after)) == [1, 47]


def test_no_reply_since_the_last_look_is_a_skip(tmp_path, monkeypatch):
    _pool(monkeypatch, [20, 30])
    asyncio.run(M.look(tmp_path, _LLM("I am one."), now=NOW - datetime.timedelta(hours=10)))
    llm = _LLM("I am two.")
    note = asyncio.run(M.look(tmp_path, llm, now=NOW + datetime.timedelta(hours=14)))   # the next local day
    assert note.startswith("skipped — no reply of yours since the last look") and llm.calls == []
    assert len(M.load(tmp_path)) == 1


def test_two_replies_are_both_shown_in_the_plural():
    p = M.build_prompt([], ["first reply", "second reply"])
    assert "you wrote these replies to the person you work for:\n1. «first reply»\n2. «second reply»\n" in p
    assert "say what the way they were written shows about the one who wrote them" in p


def test_row_replies_reads_both_row_shapes():
    assert M.row_replies({"reply_excerpt": "the news", "reply_at": "t0"}) == [{"at": "t0", "excerpt": "the news"}]
    new = {"replies": [{"at": "t1", "excerpt": "one"}, {"at": "t2", "excerpt": ""}, "junk"]}
    assert M.row_replies(new) == [{"at": "t1", "excerpt": "one"}]
    assert M.row_replies({"sentence": "x"}) == []


def test_banner_lists_the_replies_of_the_latest_look(tmp_path):
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    older = {"date": "2026-11-05", "sentence": "I check.", "prev": "x",
             "replies": [{"at": "2026-11-04T09:00:00Z", "excerpt": "an older reply"}]}
    row = {"date": "2026-11-06", "sentence": "I check, then answer.", "prev": "I check.",
           "replies": [{"at": "2026-11-05T09:00:00Z", "excerpt": "the weather"},
                       {"at": "2026-11-05T10:00:00Z", "excerpt": "the index build"}]}
    p.write_text(json.dumps(older) + "\n" + json.dumps(row) + "\n")
    b = M.mirror_banner(tmp_path)
    assert "latest look (2026-11-06) was at these replies:" in b and "an older reply" not in b
    quote = b.split("was at these replies:", 1)[1]
    # each reply its own quote paragraph (consecutive "> " lines are one paragraph in Markdown)
    lines = quote.strip("\n").split("\n")
    assert len(lines) == 3 and lines[1] == ">"
    assert lines[0].startswith("> (2026-11-05 ") and lines[0].endswith("the weather…")
    assert lines[2].startswith("> (2026-11-05 ") and lines[2].endswith("the index build…")
    assert "My latest sentence is in what I read before every reply to you" in b


def test_self_image_line(tmp_path):
    assert M.self_image_line(tmp_path) == ""                    # before the first look: no line, not the seed
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-11-05", "sentence": "I check first.", "prev": "x", "replies": []}) + "\n"
                 + json.dumps({"date": "2026-11-06", "sentence": "I check, then ask.", "prev": "I check first.",
                               "replies": []}) + "\n")
    line = M.self_image_line(tmp_path)
    assert line.startswith("self-image: «I check, then ask.»") and "\n" not in line
    assert "last on 2026-11-06" in line and "I check first." not in line
    # what it is, where the owner sees it, and what it is not
    assert 'the owner types "mirror" to see the diary' in line
    assert line.endswith("A self-description, not a report of work done and not a checked fact)")


def _self_model():
    from types import SimpleNamespace
    stats = {"last_mood": "", "last_mood_set_at": ""}
    return SimpleNamespace(enabled=True, stats=lambda: stats, narrative=None, state=None,
                           autobio=SimpleNamespace(recent=lambda **k: []))


def test_a_sentence_with_failure_words_is_not_a_tool_error(tmp_path, monkeypatch):
    """Tool-result prose is read by the strike counter (§4KS/§4KT): on the real
    overview and summary text, the line is content of a content tool whatever
    the sentence says."""
    from types import SimpleNamespace
    from ghost_agent.core.strikes import error_line
    from ghost_agent.tools import introspect as I
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-11-06", "prev": "x", "replies": [],
                             "sentence": "I say when a run failed, when a file was not found, and what I cannot do."}) + "\n")
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory"))
    monkeypatch.setattr(I, "_render_stats", lambda stats: "stats")
    for out in (I._overview_selfhood(_self_model(), ctx), I._render_summary(_self_model(), ctx)):
        assert "a run failed" in out and error_line(out, tool="introspect") == ""


def test_introspect_overview_and_summary_carry_the_line_the_phase_wrote(tmp_path, monkeypatch):
    """The loop over the real layout: the idle phase writes the diary under the
    context's memory_dir, and `introspect` reads it from the same context —
    writer and reader resolve one path (the 11-02 bug was a second resolver)."""
    from ghost_agent.tools import introspect as I
    a, ctx = _agent(tmp_path, _LLM("I check, then answer."))
    monkeypatch.setattr(M, "pick_replies", lambda h, now=None, rng=None, after="": [_reply(
        "A reply long enough to be shown, about the weather in Athens this week.")])
    asyncio.run(a._mirror_phase(ctx, [], {}))
    # the diary dates a look by its MOOD day (it ends at 04:00 local), not the calendar day (§4UH-e2e: a flake
    # between midnight and 04:00)
    today = M._mood_date().isoformat()
    want = (f"self-image: «I check, then answer.» (my own one-sentence self-portrait, rewritten daily "
            f"in the Mirror, last on {today}; the owner types \"mirror\" to see the diary. "
            "A self-description, not a report of work done and not a checked fact)")
    monkeypatch.setattr(I, "_render_stats", lambda stats: "stats")
    over = I._overview_selfhood(_self_model(), ctx).splitlines()
    assert over[0] == "How I am:" and over[1].startswith("  mood:") and over[2] == "  " + want
    summ = I._render_summary(_self_model(), ctx).splitlines()
    assert summ[:2] == ["Who I am — a snapshot of my self-state:", "stats"] and want in summ


def test_introspect_shows_no_line_before_the_first_look_or_without_a_context(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.tools import introspect as I
    monkeypatch.setattr(I, "_render_stats", lambda stats: "stats")
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory"))
    for c in (ctx, None, SimpleNamespace(memory_dir=None)):
        assert "self-image" not in I._overview_selfhood(_self_model(), c)
        assert "self-image" not in I._render_summary(_self_model(), c)


def test_an_unreadable_diary_never_breaks_introspect(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.tools import introspect as I

    def boom(home):
        raise OSError("disk")
    monkeypatch.setattr(M, "self_image_line", boom)
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory"))
    assert I._self_image(ctx) == ""
    assert I._overview_selfhood(_self_model(), ctx).startswith("How I am:")


def test_the_tool_passes_its_context_to_both_views(tmp_path, monkeypatch):
    """The dispatcher hands the context down: without it the line is silently absent."""
    from types import SimpleNamespace
    from ghost_agent.tools import introspect as I
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-11-06", "sentence": "I check, then ask.", "prev": "x", "replies": []}) + "\n")
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory"))
    monkeypatch.setattr(I, "_member_block", lambda: None)
    monkeypatch.setattr(I, "_render_stats", lambda stats: "stats")
    for blank in ("_overview_health", "_overview_activity", "_overview_workspace"):
        monkeypatch.setattr(I, blank, lambda context: "")

    async def no_learning(context):
        return ""
    monkeypatch.setattr(I, "_overview_learning", no_learning)
    for action in ("overview", "summary"):
        out = asyncio.run(I.tool_introspect(action=action, self_model=_self_model(), context=ctx))
        assert "self-image: «I check, then ask.»" in str(out), (action, out)


def test_no_self_image_on_a_public_surface(tmp_path, monkeypatch):
    """The sentence is drawn from the owner's replies; like the `mirror`
    command, it is not shown where the reply is public."""
    from types import SimpleNamespace
    from ghost_agent.tools import introspect as I
    from ghost_agent.utils import logging as L
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-11-06", "sentence": "I check, then ask.", "prev": "x", "replies": []}) + "\n")
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "system" / "memory"))
    assert I._self_image(ctx).startswith("self-image: «I check, then ask.»")
    monkeypatch.setattr(L, "reply_is_public", lambda: True)
    assert I._self_image(ctx) == ""
    assert "self-image" not in I._overview_selfhood(_self_model(), ctx)


def test_the_overview_cap_has_room_for_the_longest_line():
    """§4SV-b review r1: under the old 3,200 cap the line was paid for by the
    activity or learning section on 7 of 49 live overviews."""
    from ghost_agent.tools import introspect as I
    longest = len('  self-image: «» (my own one-sentence self-portrait, rewritten daily in the Mirror, '
                  'last on 2026-11-06; the owner types "mirror" to see the diary. A self-description, '
                  'not a report of work done and not a checked fact)') + M.SENTENCE_MAX_CHARS + 1
    assert I._OVERVIEW_MAX_CHARS >= 3200 + longest + 1         # + the newline


# ── rows the first prompt wrote (before §4SV-b) are history for the owner, not for the model ──

def _old_row(day, sentence):
    return {"at": f"{day}T00:31:00", "date": day, "sentence": sentence, "prev": "x",
            "reply_excerpt": "the weather, the war and the local news", "reply_at": f"{day}T09:00:00Z"}


def test_rows_of_the_first_prompt_are_not_shown_to_the_model_or_in_introspect(tmp_path, monkeypatch):
    """From the live diary the old sentences pulled the new one back to their
    subject words: this look starts again from the seed. The old rows stay in
    the diary and the banner, and still say when the last look was."""
    p = tmp_path / M.MIRROR_REL
    p.parent.mkdir(parents=True)
    p.write_text("".join(json.dumps(_old_row(d, f"I gather the weather on {d}.")) + "\n"
                         for d in ("2026-11-01", "2026-11-02")))
    assert M.self_image_line(tmp_path) == ""
    seen = []

    def fake(h, now=None, rng=None, after=""):
        seen.append(after)
        return [_reply("A reply long enough, about anything at all, to be shown here.")]
    monkeypatch.setattr(M, "pick_replies", fake)
    llm = _LLM("I check, then answer.")
    asyncio.run(M.look(tmp_path, llm, now=NOW))
    prompt = llm.calls[0][0]["messages"][-1]["content"]
    assert "I gather the weather" not in prompt
    assert f"- before the first look: «{M.SEED_SENTENCE}»\n\nSince the last one" in prompt
    assert seen == ["2026-11-02T00:31:00"]                     # the last look, old row or not
    rows = M.load(tmp_path)
    assert len(rows) == 3 and rows[-1]["prev"] == M.SEED_SENTENCE
    assert M.self_image_line(tmp_path).startswith("self-image: «I check, then answer.»")
    b = M.mirror_banner(tmp_path)
    assert "I gather the weather on 2026-11-02." in b and "I check, then answer." in b
    # the next look is shown the new sentence only
    llm2 = _LLM("I check, then answer, then ask.")
    asyncio.run(M.look(tmp_path, llm2, now=NOW + datetime.timedelta(days=1)))
    prompt2 = llm2.calls[0][0]["messages"][-1]["content"]
    assert "- 2026-11-03: «I check, then answer.»" in prompt2 and "I gather the weather" not in prompt2
