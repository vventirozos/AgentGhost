"""§4TC (2026-11-06): the mood is audible — ONE line per turn, AFTER the head.

The operator's hard requirement: the cached / saved head (system prompt +
tool definitions) is byte-identical whatever the mood; the line sits in the
per-turn state block and nowhere else, is zero bytes for a neutral mood with
no manner, and depends only on (label, intensity, manner, manner intensity) —
no time, no decimal, no cause.

Pinned here: the line's text per level, who hears it and who does not, where
it is placed in what the real turn loop sends to the model, and the head
being the same bytes under every mood.

All owner data here is fictional (Elena, her dog Bruno)."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import llm as LLM
from ghost_agent.core.agent import GhostAgent
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mood as M
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import (
    ORIGIN_PROBE, reply_surface_context, request_id_context, request_origin_context, requester_role_context)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ghost_agent"
ASK = "how old is Bruno?"

HOW = (" Speak in that mood: it must be audible in HOW you phrase this reply - word choice, rhythm - "
       "the way a person in that mood would say the same thing. ")
LIMITS = ("WHAT you say stays exactly as it would be: the same facts, numbers, tool use and every point "
          "the answer needs (a manner trims only padding), no new topics or asides; the mood changes no answer and no refusal. Code, file contents, tool "
          "arguments and commands carry no tone. Precedence: the owner's rules, then correctness, then the "
          "manner, then the mood. This applies to every reply, serious ones included.")
#: §4TX: the owner is pointed at the real cause; anyone else is told it is not about them
WHY = 'Asked why, call introspect(action="mood") first - never guess a cause.'
WHY_OTHERS = "It is not about the person you are talking to; never invent a cause."
LEVEL = {1: "At 1 it is a faint tint: word choice only, no remarks about it.",
         2: "At 2 it is clearly noticeable, with a remark or two in character.",
         3: "At 3 it is strong, and you may say outright that you are in this mood."}


# ── the text ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_line_for_a_mood_at_each_level(n):
    assert M.tone_line("irritable", n) == (f"YOUR MOOD RIGHT NOW: irritable, {n} of 3." + HOW + LEVEL[n] + " "
                                           + WHY + " " + LIMITS)
    assert M.tone_line("irritable", n, audience="others") == (
        f"YOUR MOOD RIGHT NOW: irritable, {n} of 3." + HOW + LEVEL[n] + " " + WHY_OTHERS + " " + LIMITS)


@pytest.mark.parametrize("n,level", [
    (1, "At 1 it is a faint tint: word choice only, no remarks about it."),
    (2, "At 2 it is clearly noticeable in the wording; it already showed earlier in this conversation, so no "
        "remark about it."),
    (3, "At 3 it is strong; you already let it show in this conversation, so do not announce it again."),
])
def test_a_mood_already_heard_in_the_conversation_keeps_its_tint_without_a_new_remark(n, level):
    """§4TX (review: "a remark or two" every reply became a verbal tic)."""
    assert M.tone_line("down", n, remarked=True) == (f"YOUR MOOD RIGHT NOW: down, {n} of 3." + HOW + level + " "
                                                     + WHY + " " + LIMITS)


def test_level_one_is_a_tint_and_only_level_three_may_name_the_mood():
    one, two, three = (M.tone_line("tired", n) for n in (1, 2, 3))
    assert "no remarks about it" in one and "no remarks about it" not in two + three
    assert "a remark or two in character" in two and "a remark or two" not in one + three
    assert "say outright that you are in this mood" in three and "say outright" not in one + two


def test_the_line_with_a_manner_names_it_between_the_mood_and_the_instruction():
    line = M.tone_line("cheerful", 2, "sarcastic", 3)
    assert line == ("YOUR MOOD RIGHT NOW: cheerful, 2 of 3. MANNER (set by the owner): sarcastic, 3 of 3."
                    + HOW + LEVEL[2] + " " + WHY + " " + M.MANNER_WHY_OWNER + " " + LIMITS)  # the level sentence is the MOOD's


@pytest.mark.parametrize("m,level", [
    (1, "At 1 it is a faint tint: word choice only, no remarks about it."),
    (2, "At 2 it is clearly noticeable, with a remark or two in character."),
    (3, "At 3 it is strong throughout."),
])
def test_the_line_for_a_manner_alone(m, level):
    line = M.tone_line("neutral", 0, "formal", m)
    assert line == (f"MANNER (set by the owner): formal, {m} of 3. Speak in that manner: it must be audible in "
                    "HOW you phrase this reply - word choice, rhythm - the way a person with that manner would "
                    "say the same thing. " + level + " " + M.MANNER_WHY_OWNER + " " + LIMITS)
    assert "MOOD" not in line and "in this mood" not in line


@pytest.mark.parametrize("args", [
    ("neutral", 0), ("neutral", 0, "", 0), ("", 2), ("irritable", 0), ("neutral", 0, "rude", 2),
    ("irritable", 9), ("irritable", None), ("irritable", "strong"), (None, 2),
])
def test_nothing_to_say_is_no_bytes(args):
    assert M.tone_line(*args) == ""


def test_a_manner_that_is_no_manner_is_left_out_of_a_mood_line():
    assert M.tone_line("down", 1, "rude", 2) == M.tone_line("down", 1)
    assert M.tone_line("down", 1, "gentle", "x") == M.tone_line("down", 1, "gentle", 2)       # default level


def test_every_line_is_one_line_with_the_limits_and_no_number_of_the_pair():
    for label in M.LABELS:
        for n in (1, 2, 3):
            for manner in ("",) + M.MANNERS:
                line = M.tone_line(label, n, manner, 2)
                assert "\n" not in line and line.endswith(LIMITS)
                assert not re.search(r"\d\.\d|\d:\d\d|ago|because|since", line), line
    # §4TX: + the "why" sentence, no tone in code / tool arguments, the precedence (was < 600)
    # §4UB review r1: + the manner's "why" for the owner too (was < 800)
    assert len(M.tone_line("irritable", 3, "sarcastic", 3)) < 850
    # §4UB: a member's line also carries the manner's "why" rule
    assert len(M.tone_line("irritable", 3, "sarcastic", 3, audience="others")) < 900


def test_the_line_depends_on_the_label_the_intensity_and_the_manner_only(tmp_path):
    """Two moods that differ in everything else — the pair, the cause, the
    time, who set it — give the same bytes."""
    a, b = SelfModel(root=tmp_path / "a"), SelfModel(root=tmp_path / "b")
    a.state.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 2, note="Elena said so")
    b.state.nudge(-0.50, 0.33, "4 tool calls failed in this turn", M.KIND_TURN)
    b.state.nudge(-0.02, 0.0, "the owner corrected my last reply", M.KIND_CORRECTION)
    va, vb = a.mood(), b.mood()
    assert (va.p, va.cause, va.source) != (vb.p, vb.cause, vb.source)
    assert va.tone() == vb.tone() == M.tone_line("irritable", 2)
    for hidden in ("Elena", "failed", "corrected", "owner said", "0.5", "0.6"):
        assert hidden not in va.tone() + vb.tone()


# ── who hears it ────────────────────────────────────────────────────────────

def _ctx(tmp_path, label="irritable", n=2, manner=""):
    sm = SelfModel(root=tmp_path)
    if label:
        sm.state.set_owner_mood(label, M.ANCHOR_DIRECTION[label], n)
    if manner:
        sm.state.set_manner(manner, 2)
    return SimpleNamespace(self_model=sm, turn_origin_label=None,
                           owner_memory_isolated=False, _replay_forbidden_tools=None)


class _req:
    """A request of one kind, as the context variables carry it."""
    def __init__(self, rid="web-4tc-tone", role="", surface="", origin=""):
        self.v = [(request_id_context, rid), (requester_role_context, role),
                  (reply_surface_context, surface), (request_origin_context, origin)]

    def __enter__(self):
        self.t = [(var, var.set(val)) for var, val in self.v]

    def __exit__(self, *exc):
        for var, tok in reversed(self.t):
            var.reset(tok)


EXPECT = M.tone_line("irritable", 2) + "\n\n"
#: §4TX: what a member hears (not told the owner's "why")
EXPECT_OTHERS = M.tone_line("irritable", 2, audience="others") + "\n\n"
#: §4TX r2: the owner in a public channel (others read it; the owner may be its cause)
EXPECT_PUBLIC = M.tone_line("irritable", 2, audience="public") + "\n\n"


def _expected(who):
    if who.get("role") == "member":
        return EXPECT_OTHERS
    return EXPECT_PUBLIC if who.get("surface") == "public" else EXPECT


@pytest.mark.parametrize("who", [
    {"rid": "web-4tc-tone"},                                  # the owner
    {"rid": "slack-4tc", "surface": "public"},                # the owner in a public channel
    {"rid": "slack-4tc-m", "role": "member"},                 # a channel member
    {"rid": "slack-4tc-m", "role": "member", "surface": "public"},
    {"rid": "probe-4tc"},                                     # a probe sees what a real turn sees
    {"rid": "web-4tc-tone", "origin": ORIGIN_PROBE},
    {"rid": "sched-morning-news"},                            # a scheduled task's turn
    {"rid": "job-12"},                                        # the wake-up of a job the owner started
])
def test_every_main_loop_reply_carries_the_line(tmp_path, who):
    ctx = _ctx(tmp_path)
    with _req(**who):
        assert A.mood_is_heard_on_this_request(ctx) is True
        assert A._mood_tone_block(ctx) == _expected(who)


@pytest.mark.parametrize("rid", ["sub-3", "sim-9", "bench-1", "replay-7", "SYSTEM", ""])
def test_a_run_that_answers_nobody_does_not(tmp_path, rid):
    ctx = _ctx(tmp_path)
    with _req(rid=rid):
        assert A.mood_is_heard_on_this_request(ctx) is False
        assert A._mood_tone_block(ctx) == ""


@pytest.mark.parametrize("mark", ["sim", "bench", "leaf", "contained", "isolated", "read_only"])
@pytest.mark.parametrize("rid", ["web-4tc-tone", "probe-4tc", "sched-morning-news"])
def test_a_sub_agent_a_leaf_or_a_contained_copy_does_not(tmp_path, mark, rid):
    ctx = _ctx(tmp_path)
    if mark in ("sim", "bench", "leaf"):
        ctx.turn_origin_label = mark
    elif mark == "contained":
        ctx._replay_forbidden_tools = frozenset({"deploy"})
    elif mark == "isolated":
        ctx.owner_memory_isolated = True
    else:
        ctx.simulated = True
    with _req(rid=rid):
        if not (rid.startswith("probe-") and mark in ("sim", "bench", "leaf", "read_only")):
            # (a probe id is a probe whatever its context is labelled — `turn_origin`)
            assert A._mood_tone_block(ctx) == ""
    with _req(rid="web-4tc-tone"):
        assert A._mood_tone_block(ctx) == ""


def test_the_real_leaf_and_replay_contexts_do_not_hear_it(tmp_path):
    """The two builders themselves, not their marks re-typed."""
    from ghost_agent.core.coding_loop import build_leaf_context
    from ghost_agent.core.isolation import is_contained_context
    live = _ctx(tmp_path)
    live.args = SimpleNamespace(native_tools=True)
    leaf = build_leaf_context(live, leaf_id="leaf-1", project_id="p1")
    with _req():
        assert A._mood_tone_block(live) == EXPECT
        assert A._mood_tone_block(leaf) == ""
        assert A.mood_moves_on_this_request(leaf) is False
    contained = SimpleNamespace(**{**vars(live), "_replay_forbidden_tools": frozenset({"deploy"})})
    assert is_contained_context(contained)
    with _req():
        assert A._mood_tone_block(contained) == ""


def test_a_neutral_mood_with_no_manner_is_zero_bytes(tmp_path):
    with _req():
        assert A._mood_tone_block(_ctx(tmp_path / "none", label="")) == ""
        worn = _ctx(tmp_path / "worn")
        worn.self_model.state.state.mood.updated_at = "2026-01-01T00:00:00Z"       # long worn off
        worn.self_model.state.state.mood.held_until = ""                          # §4TX: and its hold over
        assert A._mood_tone_block(worn) == ""
        assert A._mood_tone_block(_ctx(tmp_path / "manner", label="", manner="terse")) == (
            M.tone_line("neutral", 0, "terse", 2) + "\n\n")


def test_no_self_model_a_disabled_one_or_a_mock_is_zero_bytes(tmp_path):
    with _req():
        assert A._mood_tone_block(SimpleNamespace()) == ""
        assert A._mood_tone_block(None) == ""
        off = _ctx(tmp_path)
        off.self_model = SelfModel(root=tmp_path / "off", enabled=False)
        assert A._mood_tone_block(off) == ""
        assert A._mood_tone_block(MagicMock()) == ""


def test_a_fault_in_the_mood_is_zero_bytes_not_a_broken_turn(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)

    def boom(*a, **k):
        raise OSError("store gone")
    with _req():
        monkeypatch.setattr(ctx.self_model.state, "mood", boom)
        assert A._mood_tone_block(ctx) == ""
        monkeypatch.setattr(A, "turn_origin", boom)
        assert A._mood_tone_block(ctx) == "" and A.mood_is_heard_on_this_request(ctx) is False


def test_building_the_line_writes_nothing(tmp_path):
    """A probe (and every other reader) sees the mood and never changes it."""
    ctx = _ctx(tmp_path)
    st = ctx.self_model.state
    st.state.mood.updated_at = "2026-11-01T00:00:00Z"            # it has worn off since
    raw, rows = st.path.read_bytes(), st.mood_history_path.read_bytes()
    for who in ({"rid": "probe-4tc"}, {"rid": "slack-m", "role": "member"}, {"rid": "sched-x"}, {"rid": "web-1"}):
        with _req(**who):
            A._mood_tone_block(ctx)
    assert st.path.read_bytes() == raw and st.mood_history_path.read_bytes() == rows


@pytest.mark.parametrize("who", [
    {"rid": "sched-morning-news"}, {"rid": "probe-4tc"}, {"rid": "job-3"},
    {"rid": "slack-4tc-m", "role": "member"},
    {"rid": "web-4tc-tone"}, {"rid": "slack-4tc", "surface": "public"},       # the owner
])
def test_the_boredom_of_a_running_quiet_is_heard_by_no_reply(tmp_path, who):
    """§4TX (review: every night the morning report and the members were answered "bored"): a quiet is not
    taken out on whoever writes next. Read-time only, still shown on the face and by introspect."""
    import datetime as dt
    ctx = _ctx(tmp_path, label="")
    st = ctx.self_model.state
    for hours, level in ((7, 1), (13, 2)):
        st.mark_owner_turn(now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours))
        with _req(**who):
            assert A._mood_tone_block(ctx) == ""
        if who["rid"].startswith(("sched-", "probe-", "job-")) or who.get("role") == "member":
            with _req(**who):
                assert (st.mood().label, st.mood().intensity) == ("bored", level)        # the face still sees it
    assert st.state.mood is None                                 # never in the pair


# ── where it is placed: the real turn loop ──────────────────────────────────

def _loop_agent(tmp_path, *, native=True, self_model=None):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    context = MagicMock()
    context.args.model = "test-model"
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 32000
    context.args.perfect_it = False
    context.args.native_tools = native
    context.sandbox_dir = "/tmp"
    context.memory_dir = str(tmp_path / "memory")
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.scratchpad.list_all.return_value = "nothing yet"
    context.turn_origin_label = None
    context.owner_memory_isolated = False
    context._replay_forbidden_tools = None
    context.self_model = self_model if self_model is not None else SelfModel(root=tmp_path / "self")
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


async def _send(agent, rid="web-4tc-loop", text=ASK):
    agent.sent.clear()
    await agent.handle_chat({"messages": [{"role": "user", "content": text}], "stream": False}, [],
                            request_id=rid)
    assert len(agent.sent) == 1, "one model call for a plain question"
    return agent.sent[0]


def _system(payload) -> str:
    assert payload["messages"][0]["role"] == "system"
    return payload["messages"][0]["content"]


def _head_bytes(payload) -> bytes:
    """The head as sent: the system slot and the tool definitions."""
    return (_system(payload) + "\x00" + json.dumps(payload.get("tools") or [], sort_keys=True)).encode("utf-8")


def _rest(payload) -> str:
    return "\n\x00\n".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                           for m in payload["messages"][1:])


def _no_clock(text: str) -> str:
    return re.sub(r"CURRENT TIME: [^\n]*", "CURRENT TIME: -", text)


STATES = {
    "neutral": lambda st: None,
    "irritable 2": lambda st: st.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 2),
    "cheerful 3": lambda st: st.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 3),
    "tired 1 + sarcastic": lambda st: (st.set_owner_mood("tired", M.ANCHOR_DIRECTION["tired"], 1),
                                       st.set_manner("sarcastic", 2)),
    "manner only": lambda st: (st.reset_mood(), st.set_manner("formal", 3)),
}
LINES = {
    "neutral": "",
    "irritable 2": M.tone_line("irritable", 2),
    "cheerful 3": M.tone_line("cheerful", 3),
    "tired 1 + sarcastic": M.tone_line("tired", 1, "sarcastic", 2),
    "manner only": M.tone_line("neutral", 0, "formal", 3),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("pin", ["0", "1"])
async def test_the_head_is_the_same_bytes_under_every_mood(tmp_path, monkeypatch, native, pin):
    """THE requirement. One agent, five mood states, the request sent through
    the real loop each time: the system slot and the tool definitions do not
    move by a byte; the line is in the trailing state block and nowhere else."""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", pin)
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, pin)
    agent = _loop_agent(tmp_path, native=native)
    heads, shapes = {}, {}
    for name, put in STATES.items():
        put(agent.context.self_model.state)
        payload = await _send(agent)
        heads[name] = hashlib.sha256(_head_bytes(payload)).hexdigest()
        shapes[name] = LLM.head_shape(payload)
        system, tools, rest = _system(payload), json.dumps(payload.get("tools") or []), _rest(payload)
        for mark in ("YOUR MOOD RIGHT NOW", "MANNER (set by the owner)", "Speak in that"):
            assert mark not in system and mark not in tools, (name, mark)
        line = LINES[name]
        if line:
            assert rest.count(line) == 1, name
            state_block = payload["messages"][-1]["content"]
            assert payload["messages"][-1]["role"] == "user" and "<system_state_update>" in state_block
            at = state_block.index(line)
            assert state_block.index("### DYNAMIC SYSTEM STATE") < at < state_block.index("</system_state_update>")
        else:
            assert "YOUR MOOD" not in rest and "MANNER (set by" not in rest
    assert len(set(heads.values())) == 1, heads
    assert len(set(shapes.values())) == 1 and None not in shapes.values()
    assert native is False or shapes["neutral"][2] > 0           # the tool definitions were really sent


@pytest.mark.asyncio
async def test_a_neutral_mood_sends_what_no_mood_at_all_sends(tmp_path):
    """Zero bytes: the request of an agent whose mood is neutral is the
    request of an agent with selfhood switched off."""
    on = _loop_agent(tmp_path / "on")
    off = _loop_agent(tmp_path / "off", self_model=SelfModel(root=tmp_path / "off" / "self", enabled=False))
    a, b = await _send(on), await _send(off)
    assert _head_bytes(a) == _head_bytes(b)
    assert _no_clock(_rest(a)) == _no_clock(_rest(b))
    # … and a mood adds exactly the line (after the blank line that ends the section before it)
    on.context.self_model.state.set_owner_mood("down", M.ANCHOR_DIRECTION["down"], 2)
    c = await _send(on)
    assert _no_clock(_rest(c)).replace("\n\n" + M.tone_line("down", 2), "") == _no_clock(_rest(a))
    assert _head_bytes(c) == _head_bytes(a)


@pytest.mark.asyncio
@pytest.mark.parametrize("who", [
    {"rid": "slack-4tc-m", "role": "member"},
    {"rid": "slack-4tc", "surface": "public"},
    {"rid": "probe-4tc-loop"},
    {"rid": "sched-morning-news"},
])
async def test_everyone_who_is_answered_hears_it_through_the_real_loop(tmp_path, who):
    agent = _loop_agent(tmp_path)
    agent.context.self_model.state.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2)
    raw = agent.context.self_model.state.path.read_bytes()
    with _req(**who):
        payload = await _send(agent, rid=who["rid"])
    audience = ("others" if who.get("role") == "member"
                else "public" if who.get("surface") == "public" else "owner")
    assert M.tone_line("cheerful", 2, audience=audience) in payload["messages"][-1]["content"]
    assert "YOUR MOOD" not in _system(payload)
    mood_now = json.loads(agent.context.self_model.state.path.read_text())["mood"]
    assert mood_now == json.loads(raw)["mood"]                   # hearing it does not move it


@pytest.mark.asyncio
async def test_the_saved_head_is_built_without_the_mood(tmp_path):
    """The warm-up / saved head (`_head_payload`), for the owner and for the
    members: the same shape under every mood."""
    class _RS:
        async def get_profile_str(self):
            return "OWNER PROFILE"

        def get_active_tool_defs(self, query):
            return list(R.TOOL_DEFINITIONS)

    agent = _loop_agent(tmp_path)
    agent._warm_up_effort = lambda p: p
    seen = {}
    for name, put in STATES.items():
        put(agent.context.self_model.state)
        for member in (False, True):
            payload, _base, warmed = await agent._head_payload(_RS(), member=member)
            seen.setdefault(member, set()).add((LLM.head_shape(payload), hashlib.sha256(warmed.encode()).hexdigest()))
            assert "YOUR MOOD" not in json.dumps(payload) and "MANNER (set by" not in json.dumps(payload), name
    assert len(seen[False]) == 1 and len(seen[True]) == 1


def test_the_tool_definition_is_static_text():
    """The `introspect` description is part of the head: it names the labels
    and the manners — never the mood of the moment."""
    before = json.dumps(R.TOOL_DEFINITIONS, sort_keys=True)
    sm = SelfModel(root=Path(__import__("tempfile").mkdtemp()))
    sm.state.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 3)
    sm.state.set_manner("dramatic", 3)
    assert json.dumps(R.TOOL_DEFINITIONS, sort_keys=True) == before
    d = next(t for t in R.TOOL_DEFINITIONS if t["function"]["name"] == "introspect")["function"]["description"]
    for label in M.LABELS:
        assert label in d
    for manner in M.MANNERS:
        assert manner in d
    assert '"cheer up"' in d and '"don\'t be so grumpy"' in d          # the two examples of a relative ask
    assert "RIGHT NOW" not in d and " of 3" not in d


def test_the_introspect_definition_grew_by_no_more_than_the_allowance():
    size = len(json.dumps(next(t for t in R.TOOL_DEFINITIONS if t["function"]["name"] == "introspect")))
    assert 3508 < size <= 3508 + 500          # §4TX: + manner_intensity and step


# ── the state block mechanics (§4RC) ────────────────────────────────────────

def test_under_the_anchor_a_mood_that_changed_mid_request_is_sent_as_a_change():
    base = {"time": "### DYNAMIC SYSTEM STATE\nCURRENT TIME: x\n\n", "owner_rules": "",
            "mood_tone": M.tone_line("irritable", 2) + "\n\n"}
    same = A._state_changes(base, dict(base))
    assert same == A._STATE_NO_CHANGE
    changed = A._state_changes(base, {**base, "mood_tone": M.tone_line("cheerful", 2) + "\n\n"})
    assert changed.startswith(A._STATE_CHANGES_INTRO)
    assert M.tone_line("irritable", 2) not in changed
    # review: the request's first block still holds the old line — the new one says it replaces it
    assert ("UPDATED — this replaces the earlier mood line. " + M.tone_line("cheerful", 2)) in changed
    assert A.MOOD_LINE_UPDATED == "UPDATED — this replaces the earlier mood line. "
    gone = A._state_changes(base, {**base, "mood_tone": ""})
    assert '(The earlier "YOUR MOOD RIGHT NOW: irritable, 2 of 3.' in gone and "no longer applies" in gone
    new = A._state_changes({**base, "mood_tone": ""}, base)
    assert M.tone_line("irritable", 2) in new and "UPDATED" not in new        # no earlier line to replace
    # … also when what changed is the MANNER on the line, or the line became a manner-only one
    for before, after in ((M.tone_line("irritable", 2), M.tone_line("irritable", 2, "sarcastic", 3)),
                          (M.tone_line("irritable", 2, "formal", 1), M.tone_line("irritable", 2, "sarcastic", 1)),
                          (M.tone_line("irritable", 2, "formal", 2), M.tone_line("neutral", 0, "formal", 2)),
                          (M.tone_line("neutral", 0, "formal", 2), M.tone_line("neutral", 0, "formal", 3))):
        got = A._state_changes({**base, "mood_tone": before + "\n\n"}, {**base, "mood_tone": after + "\n\n"})
        assert (A.MOOD_LINE_UPDATED + after) in got and before not in got
    other = A._state_changes({**base, "owner_rules": "### OWNER RULES\n1. be brief\n\n"},
                             {**base, "owner_rules": "### OWNER RULES\n1. be brief\n2. no emoji\n\n"})
    assert "UPDATED" not in other                                             # only the mood line says so


@pytest.mark.asyncio
async def test_a_mood_set_in_the_turn_is_in_the_very_next_state_block(tmp_path):
    """`set_mood` through the real dispatch, then the block the next loop
    iteration would build."""
    from tests.test_turnloop_r1_fixes import _ts
    from ghost_agent.utils import provenance as P
    P._reset_for_tests()
    sm = SelfModel(root=tmp_path)
    ctx = MagicMock()
    ctx._subagent_allowed_tools = None
    ctx.self_model = sm
    ctx.turn_origin_label = None
    ctx.owner_memory_isolated = False
    ctx._replay_forbidden_tools = None
    agent = GhostAgent(ctx)
    agent.disabled_tools = set()
    agent.available_tools = R.get_available_tools(ctx)
    with _req():
        assert A._mood_tone_block(ctx) == ""
        call = {"id": "c1", "type": "function", "function": {
            "name": "introspect", "arguments": json.dumps({"action": "set_mood", "mood": "grumpy", "intensity": 3})}}
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[call]))
        assert A._mood_tone_block(ctx) == M.tone_line("irritable", 3) + "\n\n"
    P._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_a_mood_set_at_the_third_call_of_a_five_call_turn_replaces_the_anchored_line(tmp_path, monkeypatch, native):
    """Review: under the state anchor (on in production) the request's first
    block kept saying "irritable, 2" while the trailing block said "cheerful,
    3" — two lines that contradict each other for calls 4 and 5. The trailing
    one now says it replaces the earlier one; the head still does not move."""
    from ghost_agent.tools.introspect import tool_introspect
    from ghost_agent.utils import provenance as P
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, "1")
    P._reset_for_tests()
    agent = _loop_agent(tmp_path, native=native)
    sm = agent.context.self_model
    sm.state.set_owner_mood("irritable", M.ANCHOR_DIRECTION["irritable"], 2)

    async def introspect(**kw):
        return await tool_introspect(self_model=sm, context=agent.context, **kw)
    agent.available_tools = {"introspect": introspect}
    steps = [{"action": "stats"}, {"action": "recent"},
             {"action": "set_mood", "mood": "cheerful", "intensity": 3},
             {"action": "narrative"}, "Done — cheerful now."]
    agent.sent = []

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
    await agent.handle_chat({"messages": [{"role": "user", "content": "cheer up, a lot"}], "stream": False}, [],
                            request_id="web-4tc-five")
    assert len(agent.sent) == 5 and (sm.mood().label, sm.mood().intensity) == ("cheerful", 3)
    old, new = M.tone_line("irritable", 2), M.tone_line("cheerful", 3)

    def lines(payload):
        out = []
        for m in payload["messages"][1:]:
            t = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            if old in t:
                out.append("old")
            if new in t:
                out.append("updated" if (A.MOOD_LINE_UPDATED + new) in t else "new, unmarked")
        return out
    assert [lines(p) for p in agent.sent] == [["old"], ["old"], ["old"], ["old", "updated"], ["old", "updated"]]
    for p in agent.sent[3:]:
        assert A.MOOD_LINE_UPDATED in p["messages"][-1]["content"]            # in the TRAILING block
    assert len({hashlib.sha256(_head_bytes(p)).hexdigest() for p in agent.sent}) == 1
    P._reset_for_tests()


# ── nobody else builds or reads the line ────────────────────────────────────

def _calls(tree, names):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in names:
                out.append(name)
    return out


def test_the_line_is_built_in_one_place_and_enters_one_block():
    """Side calls (the verifier, summaries, the diary, the Mirror, judges)
    build their own prompts: none of them calls the tone line. Read off the
    tree."""
    users = {}
    for path in SRC.rglob("*.py"):
        got = _calls(ast.parse(path.read_text(encoding="utf-8")), {"tone_line", "tone", "_mood_tone_block"})
        if got:
            users[str(path.relative_to(SRC))] = sorted(got)
    # tools/introspect.py (2026-11-08, operator: "add the tone to the mood"):
    # the `mood` view SHOWS the line verbatim to the owner (a tool result in
    # the owner's own turn, not a side call's prompt) — `mood_tone_text`.
    assert users == {"core/agent.py": ["_mood_tone_block", "tone"],
                     "selfhood/state.py": ["tone_line"],
                     "tools/introspect.py": ["tone"]}
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    keys = [n.args[0].elts[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "_ds_parts"
            and n.args and isinstance(n.args[0], ast.Tuple) and isinstance(n.args[0].elts[0], ast.Constant)]
    assert keys.count("mood_tone") == 1
    assert keys.index("mood_tone") == keys.index("owner_rules") + 1            # right after the owner's rules


def test_the_system_prompt_names_no_mood_of_the_moment():
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert "YOUR MOOD RIGHT NOW" not in SYSTEM_PROMPT and "MANNER (set by" not in SYSTEM_PROMPT
    rule = next(line for line in SYSTEM_PROMPT.split("\n") if line.startswith("- SELF SURFACE:"))
    assert "`set_mood` (with `mood` and/or `manner`)" in rule and "`mood` for why-are-you-in-this-mood" in rule
