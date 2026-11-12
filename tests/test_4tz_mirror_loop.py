"""§4TZ: the Mirror's loop closed — the latest self-image is one section of the
per-turn state block on every reply to the OWNER on a private surface (never
the system slot, a tool description, a member, a public channel or a run that
answers nobody); six replies per look; a long sentence must replace, not add;
the `mirror` banner shows how much each sentence kept of the day before."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mirror as M
from ghost_agent.utils.logging import (


    ORIGIN_PROBE, reply_surface_context, request_id_context, request_origin_context, requester_role_context)

SENTENCE = "I am the one who checks twice and then says it plainly."



# dates relative to the mood day (§4UH-e2e: these were fixed dates, written on 11-10 — after 11-12 the diary read stale and the block wording changed)
import datetime as _dt_rel  # noqa: E402
from ghost_agent.selfhood import mirror as _MI_rel  # noqa: E402
D04 = (_MI_rel._mood_date() + _dt_rel.timedelta(days=-6)).isoformat()
D05 = (_MI_rel._mood_date() + _dt_rel.timedelta(days=-5)).isoformat()
D08 = (_MI_rel._mood_date() + _dt_rel.timedelta(days=-2)).isoformat()
D09 = (_MI_rel._mood_date() + _dt_rel.timedelta(days=-1)).isoformat()
D10 = (_MI_rel._mood_date() + _dt_rel.timedelta(days=0)).isoformat()

@pytest.fixture(autouse=True)
def _the_mirrors_own_path(monkeypatch):
    """§4UG: these tests are about the Mirror's own sentence — what a reply hears with the character
    switched off (``GHOST_CHARACTER=0``). The character has its tests in test_4ug_character.py."""
    monkeypatch.setenv("GHOST_CHARACTER", "0")


def _home(tmp_path) -> Path:
    """GHOST_HOME laid out as production lays it out: memory_dir = $HOME/system/memory."""
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write(home, *rows):
    p = home / M.MIRROR_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row(sentence, prev=M.SEED_SENTENCE, date=D09, replies=True):
    r = {"at": f"{date}T00:30:00", "date": date, "sentence": sentence, "prev": prev}
    if replies:
        r["replies"] = [{"at": f"{date}T09:00:00Z", "excerpt": "Bruno's match is at six."}]
    else:
        r["reply_excerpt"] = "the weather in Kalopetres"
    return r


def _ctx(home):
    return SimpleNamespace(memory_dir=str(home / "system" / "memory"), turn_origin_label=None,
                           owner_memory_isolated=False, _replay_forbidden_tools=None)


class _req:
    def __init__(self, rid="web-4tz", role="", surface="", origin=""):
        self.v = [(request_id_context, rid), (requester_role_context, role),
                  (reply_surface_context, surface), (request_origin_context, origin)]

    def __enter__(self):
        self.t = [(var, var.set(val)) for var, val in self.v]

    def __exit__(self, *exc):
        for var, tok in reversed(self.t):
            var.reset(tok)


EXPECT = f"{M.SELF_IMAGE_HEADER}\n«{SENTENCE}»\n{M.SELF_IMAGE_RULE}\n\n"


# ── the section's text ──────────────────────────────────────────────────────

def test_no_diary_no_home_or_only_first_prompt_rows_is_zero_bytes(tmp_path):
    home = _home(tmp_path)
    assert M.self_image_block(home) == ""
    assert M.self_image_block(None) == ""
    assert M.self_image_block("") == ""
    _write(home, _row("I am a faithful sentinel who gathers the weather.", replies=False))
    assert M.self_image_block(home) == ""          # a §4SV-v1 row is not this look's sentence


def test_the_section_is_the_latest_sentence_header_and_rule(tmp_path):
    home = _home(tmp_path)
    _write(home, _row("I am older.", date=D08), _row(SENTENCE, prev="I am older."))
    assert M.self_image_block(home) == EXPECT
    assert M.self_image_block(home) == EXPECT      # same bytes on every turn


def test_a_new_look_is_heard_at_once_despite_the_memo(tmp_path):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    assert SENTENCE in M.self_image_block(home)
    _write(home, _row("I am the one who asks before guessing.", prev=SENTENCE, date=D10))
    got = M.self_image_block(home)
    assert "asks before guessing" in got and SENTENCE not in got


def test_whitespace_inside_the_sentence_is_one_line(tmp_path):
    home = _home(tmp_path)
    _write(home, _row("I am\n  the one\twho checks."))
    assert "«I am the one who checks.»" in M.self_image_block(home)


def test_an_unreadable_diary_is_zero_bytes_not_an_error(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(M, "load", boom)
    M._BLOCK_MEMO.clear()
    with _req():
        assert A._self_image_block(_ctx(home)) == ""  # a fault is zero bytes, never a broken turn


# ── who hears it (owner only) ───────────────────────────────────────────────

@pytest.mark.parametrize("who", [{"rid": "web-4tz"}, {"rid": "slack-4tz"}, {"rid": "probe-4tz"},
                                 {"rid": "web-4tz", "origin": ORIGIN_PROBE}])
def test_the_owner_on_a_private_surface_and_a_probe_of_one_hear_it(tmp_path, who):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    with _req(**who):
        assert A.self_image_is_heard_on_this_request(_ctx(home)) is True
        assert A._self_image_block(_ctx(home)) == EXPECT


@pytest.mark.parametrize("who", [
    {"rid": "slack-4tz", "surface": "public"},                     # the owner in a public channel
    {"rid": "slack-4tz-m", "role": "member"},                      # a member
    {"rid": "slack-4tz-m", "role": "member", "surface": "public"},
    {"rid": "probe-4tz", "role": "member"},                        # a member's probe (request_kind says "probe")
    {"rid": "sched-morning-news"}, {"rid": "job-12"}, {"rid": "sub-3"}, {"rid": "sim-9"},
    {"rid": "bench-1"}, {"rid": "replay-7"}, {"rid": "SYSTEM"}, {"rid": ""},
])
def test_nobody_else_hears_it(tmp_path, who):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    with _req(**who):
        assert A.self_image_is_heard_on_this_request(_ctx(home)) is False
        assert A._self_image_block(_ctx(home)) == ""


@pytest.mark.parametrize("mark", ["sim", "bench", "leaf", "contained", "isolated"])
def test_a_leaf_a_bench_or_a_contained_copy_does_not(tmp_path, mark):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    ctx = _ctx(home)
    if mark in ("sim", "bench", "leaf"):
        ctx.turn_origin_label = mark
    elif mark == "contained":
        ctx._replay_forbidden_tools = frozenset({"deploy"})
    else:
        ctx.owner_memory_isolated = True
    with _req():
        assert A._self_image_block(ctx) == ""


def test_the_real_leaf_context_does_not_hear_it(tmp_path):
    from ghost_agent.core.coding_loop import build_leaf_context
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    live = _ctx(home)
    live.args = SimpleNamespace(native_tools=True)
    live.self_model = SelfModel(root=tmp_path / "self")
    leaf = build_leaf_context(live, leaf_id="leaf-1", project_id="p1")
    with _req():
        assert A._self_image_block(live) == EXPECT
        assert A._self_image_block(leaf) == ""


# ── through the real loop: after the head, never in it ──────────────────────

def _loop_agent(home):
    context = MagicMock()
    context.args.model = "test-model"
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 32000
    context.args.perfect_it = False
    context.args.native_tools = True
    context.sandbox_dir = "/tmp"
    context.memory_dir = str(home / "system" / "memory")
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.scratchpad.list_all.return_value = "nothing yet"
    context.turn_origin_label = None
    context.owner_memory_isolated = False
    context._replay_forbidden_tools = None
    context.self_model = SelfModel(root=home / "self")
    agent = GhostAgent(context)
    agent.sent = []

    def stream(payload, *a, **k):
        agent.sent.append(json.loads(json.dumps(payload, default=str)))

        async def gen():
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": "Bruno is nine."}}]}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
        return gen()
    context.llm_client.stream_chat_completion = stream
    return agent


async def _send(agent, rid="web-4tz-loop"):
    agent.sent.clear()
    await agent.handle_chat({"messages": [{"role": "user", "content": "how old is Bruno?"}], "stream": False},
                            [], request_id=rid)
    assert len(agent.sent) == 1
    return agent.sent[0]


def _head(payload) -> str:
    return hashlib.sha256((payload["messages"][0]["content"] + "\x00"
                           + json.dumps(payload.get("tools") or [], sort_keys=True)).encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", ["0", "1"])
async def test_the_owner_hears_it_in_the_state_block_and_the_head_does_not_move(tmp_path, monkeypatch, anchor):
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, anchor)
    home = _home(tmp_path)
    agent = _loop_agent(home)
    before = await _send(agent)
    assert M.SELF_IMAGE_HEADER not in json.dumps(before)          # no sentence yet: zero bytes
    _write(home, _row(SENTENCE))
    with _req(rid="web-4tz-loop-2"):                 # a new request (ids are per request: frozen per id)
        after = await _send(agent, rid="web-4tz-loop-2")
    assert _head(after) == _head(before)
    assert SENTENCE not in after["messages"][0]["content"] and SENTENCE not in json.dumps(after.get("tools") or [])
    state = after["messages"][-1]["content"]
    line = EXPECT.rstrip("\n")                  # the block trims the blank lines that end its last section
    assert after["messages"][-1]["role"] == "user" and state.count(line) == 1
    assert state.index("### DYNAMIC SYSTEM STATE") < state.index(line) < state.index("</system_state_update>")
    for m in after["messages"][:-1]:                 # review r1: not in the stable first message either
        assert SENTENCE not in (m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]))
    assert A.self_image_heard("web-4tz-loop-2") == D09     # the turn record's field


@pytest.mark.asyncio
@pytest.mark.parametrize("who", [{"rid": "slack-4tz-m", "role": "member"},
                                 {"rid": "slack-4tz", "surface": "public"},
                                 {"rid": "sched-morning-news"}])
async def test_a_member_a_public_channel_or_a_scheduled_turn_does_not_get_it_through_the_loop(tmp_path, who):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    agent = _loop_agent(home)
    with _req(**who):
        payload = await _send(agent, rid=who["rid"])
    assert SENTENCE not in json.dumps(payload) and M.SELF_IMAGE_HEADER not in json.dumps(payload)


# ── the look's prompt ───────────────────────────────────────────────────────

def _words(n):
    return " ".join(["I"] + ["word"] * (n - 1))


@pytest.mark.parametrize("n,long", [(29, False), (30, True), (40, True)])
def test_a_long_last_sentence_asks_for_a_replacement_not_an_addition(n, long):
    p = M.build_prompt([_row(_words(n))], ["first reply", "second reply"])
    note = M.LONG_NOTE.format(n=n)
    assert (note in p) is long
    assert p.rstrip().endswith("Reply with the sentence only.")
    if long:
        assert p.index("at most 40 words.") < p.index(note) < p.index("Reply with the sentence only.")
    assert "{" not in p and "}" not in p


def test_the_seed_alone_is_not_long():
    p = M.build_prompt([], ["only reply"])
    assert "words long" not in p


def test_the_note_counts_the_last_sentence_not_an_earlier_one():
    p = M.build_prompt([_row(_words(35), date=D08), _row("I am short now.", prev=_words(35))],
                       ["a", "b"])
    assert "words long" not in p


# ── watching the loop: the banner's kept share ─────────────────────────────

def test_kept_share():
    assert M.kept_share("I check twice, then answer.", "I check twice, then answer.") == 1.0
    assert M.kept_share("I check twice, then answer.", "Quiet nights suit me.") == 0.0
    assert M.kept_share("I am.", "anything") is None                  # no word of 4+ letters
    assert M.kept_share("Check twice, ANSWER plainly", "answer plainly") == 0.5             # 2 of 4 words


def test_the_banner_shows_how_much_each_sentence_kept(tmp_path):
    home = _home(tmp_path)
    a = "I check twice and answer plainly."
    b = "I check twice and answer warmly."
    _write(home, _row(a, date=D08), _row(b, prev=a, date=D09),
           _row(b, prev=b, date=D10))
    banner = M.mirror_banner(home)
    kept = round(M.kept_share(a, b) * 100)
    assert f"**{D09}** (kept {kept}% of the day before's words) — {b}" in banner
    assert f"**{D10}** (unchanged) — {b}" in banner
    assert "before every reply to you" in banner


# ── one request, one self-image (review r1) ─────────────────────────────────

def test_the_section_is_frozen_for_the_request_that_first_heard_it(tmp_path, monkeypatch):
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    ctx = _ctx(home)
    with _req(rid="web-4tz-freeze"):
        assert A._self_image_block(ctx) == EXPECT
        _write(home, _row("I am the one who asks before guessing.", prev=SENTENCE, date=D10))
        assert A._self_image_block(ctx) == EXPECT                     # a look landed mid-request
        monkeypatch.setattr(M, "_path", lambda h: None)               # … or the diary went unreadable
        assert A._self_image_block(ctx) == EXPECT
        monkeypatch.undo()
        monkeypatch.setenv("GHOST_CHARACTER", "0")          # §4UG: undo() also took the file's own switch back
    assert A.self_image_heard("web-4tz-freeze") == D09
    with _req(rid="web-4tz-freeze-2"):
        assert "asks before guessing" in A._self_image_block(ctx)    # the next request hears the new one
    assert A.self_image_heard("web-4tz-freeze-2") == D10
    with _req(rid="slack-4tz-m", role="member"):
        assert A._self_image_block(ctx) == ""
    assert A.self_image_heard("slack-4tz-m") == ""                     # nothing recorded for a member


def test_a_reused_client_request_id_hears_the_new_sentence_after_the_freeze_window(tmp_path, monkeypatch):
    import time as _t
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    ctx = _ctx(home)
    clock = [1000.0]
    monkeypatch.setattr(_t, "monotonic", lambda: clock[0])
    with _req(rid="slack-reused-id"):
        assert A._self_image_block(ctx) == EXPECT
        _write(home, _row("I am the one who asks before guessing.", prev=SENTENCE, date=D10))
        clock[0] += A._SELF_IMAGE_FREEZE_S - 1
        assert A._self_image_block(ctx) == EXPECT                     # still the same request
        clock[0] += 2
        assert "asks before guessing" in A._self_image_block(ctx)    # a later request under the same id


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", ["0", "1"])
async def test_a_look_that_lands_mid_request_changes_nothing_the_model_sees(tmp_path, monkeypatch, anchor):
    """Four calls; the diary gets a new sentence during the second. Every
    call carries the first sentence once, never the new one, and no "no
    longer applies"; the head does not move."""
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, anchor)
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", anchor)
    home = _home(tmp_path)
    _write(home, _row(SENTENCE))
    agent = _loop_agent(home)
    new = "I am the one who asks before guessing."

    async def introspect(**kw):
        if kw.get("action") == "recent":
            _write(home, _row(new, prev=SENTENCE, date=D10))
        return "ok"
    agent.available_tools = {"introspect": introspect}
    steps = [{"action": "stats"}, {"action": "recent"}, {"action": "narrative"}, "Done."]

    def stream(payload, *a, **k):
        agent.sent.append(json.loads(json.dumps(payload, default=str)))
        step = steps[min(len(agent.sent), len(steps)) - 1]
        text = ("<tool_call>\n" + json.dumps({"name": "introspect", "arguments": step}) + "\n</tool_call>"
                if isinstance(step, dict) else step)

        async def gen():
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
        return gen()
    agent.context.llm_client.stream_chat_completion = stream
    await agent.handle_chat({"messages": [{"role": "user", "content": "how are things?"}], "stream": False}, [],
                            request_id="web-4tz-four")
    assert len(agent.sent) == 4
    for p in agent.sent:
        everything = json.dumps(p["messages"][1:])
        assert new not in everything and "no longer applies" not in everything
        assert everything.count(SENTENCE) >= 1
    assert len({_head(p) for p in agent.sent}) == 1
    if anchor == "1":
        # one copy in the anchored first block; later blocks say nothing changed about it
        assert all(json.dumps(p["messages"][1:]).count(SENTENCE) == 1 for p in agent.sent)


def test_the_banner_measures_a_looks_first_row_against_the_seed(tmp_path):
    home = _home(tmp_path)
    _write(home, _row("I am a faithful sentinel.", replies=False, date=D04),
           _row("I answer what is asked, plainly.", date=D05))
    banner = M.mirror_banner(home)
    assert f"**{D05}** (kept " in banner and "of the starting sentence's words)" in banner
