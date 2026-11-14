"""§4UH-loop (2026-11-12): the Mirror is removed; the strange loop is the character's.

* the Mirror's module, diary and switch-back path are gone (`selfhood.mirror` does not import);
* what the new loop still needs lives in `selfhood.loop`: the home, a probe's self-image, the resting point's
  unit and switch, the `mirror` command (the reply picker went with the daily self-look, §4UH-noself);
* only the owner on a private surface (and a probe of one) hears the character's sentence — the §4TZ gating,
  now over the character instead of the diary; frozen per request;
* `mirror` shows the path the loop has taken: each sentence, what moved it, and where it is now.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import loop as L
from ghost_agent.utils.logging import (
    ORIGIN_PROBE, reply_surface_context, request_id_context, request_origin_context, requester_role_context)

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
NOW = datetime.datetime(2026, 11, 3, 14, 0, 0)


@pytest.fixture(autouse=True)
def _character_on(monkeypatch):
    for env in (C.ENV, L.REST_ENV):
        monkeypatch.delenv(env, raising=False)
    C._MEMO.clear()
    yield
    C._MEMO.clear()


def _home(tmp_path) -> Path:
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _ctx(home):
    return SimpleNamespace(memory_dir=str(home / "system" / "memory"), turn_origin_label=None,
                           owner_memory_isolated=False, _replay_forbidden_tools=None)


class _req:
    def __init__(self, rid="web-4uh-loop", role="", surface="", origin=""):
        self.v = [(request_id_context, rid), (requester_role_context, role),
                  (reply_surface_context, surface), (request_origin_context, origin)]

    def __enter__(self):
        self.t = [(var, var.set(val)) for var, val in self.v]

    def __exit__(self, *exc):
        for var, tok in reversed(self.t):
            var.reset(tok)


def _expect(home):
    return C.section_text(C.load(home)["dials"])


START_SENTENCE = C.sentence(C.START)


# ── the Mirror is gone ──────────────────────────────────────────────────────

def test_the_mirror_module_is_gone():
    assert not (SRC / "selfhood" / "mirror.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ghost_agent.selfhood.mirror")


def test_nothing_in_src_imports_it():
    """ast walk over every module: no import of `selfhood.mirror` (relative or absolute)."""
    bad = []
    for p in SRC.rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                names = {a.name for a in n.names}
                if mod.endswith("selfhood.mirror") or mod == "mirror" or (mod.endswith("selfhood") and "mirror" in names) \
                        or (n.level and mod == "" and "mirror" in names and p.parent.name == "selfhood"):
                    bad.append(f"{p.relative_to(SRC)}:{n.lineno}")
            elif isinstance(n, ast.Import):
                if any(a.name.endswith("selfhood.mirror") for a in n.names):
                    bad.append(f"{p.relative_to(SRC)}:{n.lineno}")
    assert bad == []


def test_the_idle_phase_writes_no_diary(tmp_path):
    """The character phase runs (day close, rule) and nothing writes a mirror file."""
    import asyncio
    home = _home(tmp_path)
    agent = GhostAgent.__new__(GhostAgent)
    agent._last_mirror_try_at = datetime.datetime.min
    agent._bio_cooldown = lambda s: s
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=None, args=None)
    ran, notes = [], {}
    asyncio.run(agent._self_phase(ctx, ran, notes))
    assert agent._last_mirror_try_at > datetime.datetime.min
    assert not list((home / "system").rglob("mirror*"))
    assert "mirror" not in ran and "mirror" not in notes


def test_the_idle_job_is_the_character():
    assert "mirror" not in A.IDLE_JOB_PURPOSE and "character" in A.IDLE_JOB_PURPOSE
    assert [j for j, _t, _e in A.IDLE_SCHEDULER_JOBS if j == "mirror"] == []
    assert any(j == "character" for j, _t, _e in A.IDLE_SCHEDULER_JOBS)


# ── with the character off: nothing is heard, nothing rests ────────────────

def test_with_the_character_off_no_section_no_line_and_no_rest(tmp_path, monkeypatch):
    from ghost_agent.tools import introspect as I
    home = _home(tmp_path)
    C.set_dial(home, "humour", 2)
    with _req():
        assert A._self_image_block(_ctx(home)) != ""
    monkeypatch.setenv(C.ENV, "0")
    with _req(rid="web-4uh-loop-off"):
        assert A._self_image_block(_ctx(home)) == ""
        assert I._self_image(_ctx(home)) == ""
    assert A.self_image_heard("web-4uh-loop-off") == ""
    assert C.rest_pair(home) == (0.0, 0.0)
    assert "switched off" in L.banner(home)


# ── who hears it (owner only; §4TZ, over the character) ─────────────────────

@pytest.mark.parametrize("who", [{"rid": "web-4uh"}, {"rid": "slack-4uh"}, {"rid": "probe-4uh"},
                                 {"rid": "web-4uh", "origin": ORIGIN_PROBE}])
def test_the_owner_on_a_private_surface_and_a_probe_of_one_hear_it(tmp_path, who):
    home = _home(tmp_path)
    with _req(**who):
        assert A.self_image_is_heard_on_this_request(_ctx(home)) is True
        got = A._self_image_block(_ctx(home))
    assert got == _expect(home) and f"«{START_SENTENCE}»" in got


@pytest.mark.parametrize("who", [
    {"rid": "slack-4uh", "surface": "public"},
    {"rid": "slack-4uh-m", "role": "member"},
    {"rid": "slack-4uh-m", "role": "member", "surface": "public"},
    {"rid": "probe-4uh", "role": "member"},
    {"rid": "sched-morning-news"}, {"rid": "job-12"}, {"rid": "sub-3"}, {"rid": "sim-9"},
    {"rid": "bench-1"}, {"rid": "replay-7"}, {"rid": "SYSTEM"}, {"rid": ""},
])
def test_nobody_else_hears_it(tmp_path, who):
    home = _home(tmp_path)
    with _req(**who):
        assert A.self_image_is_heard_on_this_request(_ctx(home)) is False
        assert A._self_image_block(_ctx(home)) == ""


@pytest.mark.parametrize("mark", ["sim", "bench", "leaf", "contained", "isolated"])
def test_a_leaf_a_bench_or_a_contained_copy_does_not(tmp_path, mark):
    home = _home(tmp_path)
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
    live = _ctx(home)
    live.args = SimpleNamespace(native_tools=True)
    live.self_model = SelfModel(root=tmp_path / "self")
    leaf = build_leaf_context(live, leaf_id="leaf-1", project_id="p1")
    with _req():
        assert A._self_image_block(live) == _expect(home)
        assert A._self_image_block(leaf) == ""


def test_the_section_is_frozen_for_the_request_that_first_heard_it(tmp_path):
    home = _home(tmp_path)
    ctx = _ctx(home)
    first = _expect(home)
    with _req(rid="web-4uh-freeze"):
        assert A._self_image_block(ctx) == first
        C.set_dial(home, "humour", 2)                             # a dial moved mid-request
        assert A._self_image_block(ctx) == first
    assert A.self_image_heard("web-4uh-freeze") == C.stamp(C.START)
    with _req(rid="web-4uh-freeze-2"):
        got = A._self_image_block(ctx)
    assert got == _expect(home) and got != first
    assert A.self_image_heard("web-4uh-freeze-2") == C.stamp(C.load(home)["dials"])
    with _req(rid="slack-4uh-m", role="member"):
        assert A._self_image_block(ctx) == ""
    assert A.self_image_heard("slack-4uh-m") == ""


def test_a_reused_client_request_id_hears_the_new_sentence_after_the_freeze_window(tmp_path, monkeypatch):
    import time as _t
    home = _home(tmp_path)
    ctx = _ctx(home)
    clock = [1000.0]
    monkeypatch.setattr(_t, "monotonic", lambda: clock[0])
    first = _expect(home)
    with _req(rid="slack-reused-id-4uh"):
        assert A._self_image_block(ctx) == first
        C.set_dial(home, "warmth", 2)
        clock[0] += A._SELF_IMAGE_FREEZE_S - 1
        assert A._self_image_block(ctx) == first
        clock[0] += 2
        assert A._self_image_block(ctx) == _expect(home) != first


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


async def _send(agent, rid):
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
async def test_the_owner_hears_it_after_the_head_and_a_move_does_not_move_the_head(tmp_path, monkeypatch, anchor):
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, anchor)
    home = _home(tmp_path)
    agent = _loop_agent(home)
    with _req(rid="web-4uh-loop-1"):
        before = await _send(agent, "web-4uh-loop-1")
    C.set_dial(home, "humour", 2)
    sentence = C.sentence(C.load(home)["dials"])
    with _req(rid="web-4uh-loop-2"):
        after = await _send(agent, "web-4uh-loop-2")
    assert _head(after) == _head(before)
    state = after["messages"][-1]["content"]
    line = _expect(home).rstrip("\n")
    assert after["messages"][-1]["role"] == "user" and state.count(line) == 1
    assert state.index("### DYNAMIC SYSTEM STATE") < state.index(line) < state.index("</system_state_update>")
    for m in after["messages"][:-1]:
        assert sentence not in (m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("who", [{"rid": "slack-4uh-m", "role": "member"},
                                 {"rid": "slack-4uh", "surface": "public"},
                                 {"rid": "sched-morning-news"}])
async def test_a_member_a_public_channel_or_a_scheduled_turn_does_not_get_it_through_the_loop(tmp_path, who):
    home = _home(tmp_path)
    agent = _loop_agent(home)
    with _req(**who):
        payload = await _send(agent, who["rid"])
    assert START_SENTENCE not in json.dumps(payload) and C.HEADER not in json.dumps(payload)


# ── the helpers the loop kept ───────────────────────────────────────────────

def test_home_is_two_levels_above_memory_dir(tmp_path):
    assert L.home_from_memory_dir(str(tmp_path / "system" / "memory")) == tmp_path
    assert L.home_from_memory_dir(None) is None and L.home_from_memory_dir("") is None


@pytest.mark.parametrize("text,yes", [("mirror", True), ("Mirror.", True), (" /mirror ", True),
                                      ("mirror the repo", False), ("show the mirror", False)])
def test_command_matcher(text, yes):
    assert L.is_mirror_command(text) is yes


def test_a_probe_section_has_the_characters_header_and_rule():
    s = L.probe_section("I am  a test\nsentence.")
    assert s == f"{C.HEADER}\n«I am a test sentence.»\n{C.RULE}\n\n"
    assert L.probe_section("I am.", "my own rule") == f"{C.HEADER}\n«I am.»\nmy own rule\n\n"
    assert L.probe_section("  ") == ""
    assert L.probe_stamp(s).startswith(L.PROBE_DATE + ":") and len(L.probe_stamp(s)) == len(L.PROBE_DATE) + 7


def test_the_rest_switch_and_unit(tmp_path, monkeypatch):
    home = _home(tmp_path)
    C.set_dial(home, "humour", 2)
    a, b = C.rest_units(C.load(home)["dials"])
    assert (a, b) != (0, 0) and C.rest_pair(home) == pytest.approx((a * L.REST_UNIT, b * L.REST_UNIT))
    monkeypatch.setenv(L.REST_ENV, "0")
    assert C.rest_pair(home) == (0.0, 0.0) and "mood rests" not in L.banner(home)


# ── `mirror`: the path the loop has taken ──────────────────────────────────

def _log(home, *rows):
    p = home / C.SELFHOOD_REL / C.LOG_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_state(home, dials, pins=()):
    p = home / C.SELFHOOD_REL / C.CHARACTER_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"dials": dials, "pins": list(pins), "changed": {}}))
    C._MEMO.clear()


START_ROW = {"at": "2026-11-11T19:02:04Z", "kind": "start", "dials": dict(C.START), "why": "the starting character"}
MOVE_W = {"at": "2026-11-12T02:38:55Z", "kind": "move", "dial": "warmth", "from": -1, "to": 0,
          "why": "7 days clearly better than its usual against 4 the other way, in what came from the owner"}
MOVE_D = {"at": "2026-11-12T02:38:55Z", "kind": "move", "dial": "directness", "from": 0, "to": 1,
          "why": "8 days clearly better than its usual against 3 the other way, in how its answers were judged"}


def test_the_path_groups_one_close_into_one_step_and_says_each_sentence(tmp_path):
    home = _home(tmp_path)
    _log(home, START_ROW, MOVE_W, MOVE_D)
    steps = L.path_steps(C, C.log_rows(home))
    assert [s["by"] for s in steps] == ["start", "itself"]
    assert steps[0]["sentence"] == START_SENTENCE
    assert steps[1]["changes"] == ["warmth -1 → 0", "directness 0 → +1"]
    assert steps[1]["sentence"] == C.sentence({"warmth": 0, "humour": 1, "directness": 1}) and steps[1]["changed"]
    assert [w.split(":")[0] for w in steps[1]["why"]] == ["warmth", "directness"]


def test_the_owners_word_a_pin_and_a_veto_are_their_own_steps(tmp_path):
    home = _home(tmp_path)
    _log(home, START_ROW, MOVE_W,
         {"at": "2026-11-13T10:00:00Z", "kind": "owner_set", "dial": "humour", "from": 1, "to": 2, "why": "x"},
         {"at": "2026-11-13T10:05:00Z", "kind": "pin", "dial": "humour"},
         {"at": "2026-11-14T10:00:00Z", "kind": "veto", "dial": "warmth", "from": 0, "to": -1},
         {"at": "2026-11-14T11:00:00Z", "kind": "move", "dial": "nonsense", "from": 0, "to": 1},
         {"at": "2026-11-14T12:00:00Z", "kind": "move", "dial": "warmth", "from": "x", "to": None})
    steps = L.path_steps(C, C.log_rows(home))
    assert [s["by"] for s in steps] == ["start", "itself", "you", "you", "you"]
    assert steps[2]["changes"] == ["humour +1 → +2"] and steps[2]["changed"]
    assert steps[3]["changes"] == ["humour pinned"] and not steps[3]["changed"]
    assert steps[4]["changes"] == ["warmth 0 → -1 (your veto)"]
    assert steps[4]["sentence"] == C.sentence({"warmth": -1, "humour": 2, "directness": 0})
    # an unknown dial and a hand-edited row with no whole numbers are not steps (and never raise)
    assert len(steps) == 5


def test_the_banner_shows_the_loop_its_path_and_now(tmp_path):
    home = _home(tmp_path)
    _log(home, START_ROW, MOVE_W, MOVE_D)
    _write_state(home, {"warmth": 0, "humour": 1, "directness": 1})
    now_s = C.sentence({"warmth": 0, "humour": 1, "directness": 1})
    b = L.banner(home)
    assert b.startswith("**The strange loop**") and "self → replies → days → self" in b
    assert "**Where it has gone** (2 moves by itself so far):" in b             # dials moved, not closes
    assert f"started as «{START_SENTENCE}»" in b
    assert f"warmth -1 → 0, directness 0 → +1, by itself, from how its days went → «{now_s}»" in b
    assert "  - warmth: 7 days clearly better" in b
    assert f"**Now:** «{now_s}»" in b and "my mood rests at pleasantness" in b
    assert "diary" not in b.lower() and "Mirror" not in b
    assert b.index("**Where it has gone**") < b.index("**Now:**")


def test_the_banner_before_any_change_and_with_an_unreadable_character(tmp_path):
    home = _home(tmp_path)
    b = L.banner(home)
    assert "**Where it has gone:** nowhere yet" in b and f"**Now:** «{START_SENTENCE}»" in b
    p = home / C.SELFHOOD_REL / C.CHARACTER_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    C._MEMO.clear()
    assert "cannot be read" in L.banner(home)


def test_the_banner_never_raises(tmp_path, monkeypatch):
    home = _home(tmp_path)

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(C, "log_rows", boom)
    b = L.banner(home)
    assert "could not be read just now" in b


def test_the_owner_command_is_answered_with_the_banner(tmp_path):
    """The `mirror` command path in the agent reads `loop.banner` (ast: the call names it)."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("selfhood.loop")
             and {"is_mirror_command", "banner"} <= {a.name for a in n.names}]
    assert found


def test_one_close_whose_rows_cross_a_second_is_still_one_step(tmp_path):
    home = _home(tmp_path)
    _log(home, START_ROW, MOVE_W, dict(MOVE_D, at="2026-11-12T02:38:56Z"))
    steps = L.path_steps(C, C.log_rows(home))
    assert len(steps) == 2 and steps[1]["changes"] == ["warmth -1 → 0", "directness 0 → +1"]
    later = dict(MOVE_D, at="2026-11-12T02:40:00Z")                       # another minute: its own step
    home2 = _home(tmp_path / "b")
    _log(home2, START_ROW, MOVE_W, later)
    assert len(L.path_steps(C, C.log_rows(home2))) == 3


def test_now_is_shown_when_the_path_cannot_be_read(tmp_path, monkeypatch):
    home = _home(tmp_path)

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(C, "log_rows", boom)
    b = L.banner(home)
    assert f"**Now:** «{START_SENTENCE}»" in b and "path could not be read just now" in b
