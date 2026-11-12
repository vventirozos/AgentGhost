"""§4UD — the head save-restore review's fixes (findings H1–H15).

The agent-level tests here run the REAL `head_saves.save` / `restore` /
`server_identity` / `prune` against a fake llama at the HTTP function
(`head_saves._http`): §4RO's shared `warm` fixture replaces save and restore
with fakes that always succeed, so a defect inside them (H2: a save filed
under a stale identity, H4: an empty file kept) was invisible to every
agent-level test (H15).

No network (the fake is the only `_http`), a temp GHOST_HOME per test.
"""
from __future__ import annotations

import ast
import asyncio
import io
import json
import os
import pathlib
import stat
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L
from ghost_agent.core.agent import GhostAgent
from ghost_agent.utils.logging import request_id_context, slot_session_context
from tests.test_4ro_head_saves import _clean, _dir, _resp, _store  # noqa: F401
from tests.test_4rp_slot_keep import _bare_client

BASE = "http://127.0.0.1:8088"
FLAG_501 = ('{"error":{"message":"This server does not support slots action. '
            'Start it with `--slot-save-path`"}}')


def _http_error(code, text):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(text.encode()))


class FakeLlama:
    """llama-server as the agent sees it over HTTP: /props, /apply-template,
    /slots, /metrics, the slot save and restore actions, the park prompt."""

    def __init__(self, d: pathlib.Path, model: pathlib.Path):
        self.d = d
        self.model_path, self.build, self.template, self.n_ctx = str(model), "b1", "T", 196608
        self.slot_n = 28213          # None: llama does not count the slot yet
        self.processing = False
        self.has_processing = True
        self.fresh = False
        self.down = None             # an exception every call raises
        self.no_flag = False         # started without --slot-save-path
        self.on_save = None          # called when a save is posted, before the file is written
        self.written = None          # the n_written llama reports (None: the truth)
        self.size = None             # the bytes it really writes (None: 1000)
        self.calls, self.saves, self.restores = [], [], []

    def swap_model(self, name="other"):
        p = pathlib.Path(self.model_path).with_name(f"{name}.gguf")
        p.write_bytes(b"another model of the same shape")
        self.model_path = str(p)

    def http(self, url, body=None, timeout=5.0):
        self.calls.append(url)
        if self.down is not None:
            raise self.down
        assert url.startswith(BASE), url
        path = url[len(BASE):]
        if path == "/props":
            return {"model_path": self.model_path, "build_info": self.build, "chat_template": self.template,
                    "default_generation_settings": {"n_ctx": self.n_ctx}}
        if path == "/apply-template":
            return {"prompt": "RENDERED " + json.dumps(body, sort_keys=True, default=str)}
        if path == "/slots":
            slot = {"id": 0}
            if self.has_processing:
                slot["is_processing"] = self.processing
            if self.slot_n is not None:
                slot["n_prompt_tokens"] = self.slot_n
            return [slot]
        if path == "/metrics":
            z = "0" if self.fresh else "5"
            return (f"llamacpp:prompt_tokens_total {z}\nllamacpp:prompt_tokens_cached_total 0\n"
                    f"llamacpp:tokens_predicted_total {z}\n")
        if path == "/completion":
            self.slot_n = 1
            return {"content": ""}
        if path.endswith("action=save"):
            if self.no_flag:
                raise _http_error(501, FLAG_501)
            if not body["filename"]:
                raise _http_error(400, '{"error":{"message":"Invalid filename"}}')
            if self.on_save is not None:
                self.on_save()
            size = 1000 if self.size is None else self.size
            (self.d / body["filename"]).write_bytes(b"s" * size)
            self.saves.append(body["filename"])
            return {"n_saved": self.slot_n or 0, "n_written": size if self.written is None else self.written,
                    "timings": {"save_ms": 12.0}}
        if path.endswith("action=restore"):
            if self.no_flag:
                raise _http_error(501, FLAG_501)
            self.restores.append(body["filename"])
            key = body["filename"][len("head-"):-len(".bin")]
            self.slot_n = int(json.loads((self.d / f"head-{key}.json").read_text())["n_tokens"])
            return {"n_restored": self.slot_n}
        raise AssertionError(f"the fake llama has no {path}")


@pytest.fixture
def real(mock_context, monkeypatch, tmp_path):
    """An agent whose llama is the fake above; `head_saves` itself is real."""
    for k in (H.ENV, L.CONV_KEEP_ENV, A.STATE_ANCHOR_ENV):
        monkeypatch.setenv(k, "1")
    monkeypatch.delenv("GHOST_MAIN_PREFIX_REWARM_S", raising=False)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 3.0)   # a busy slot is waited 1 s
    monkeypatch.setattr(A, "CONV_SAVE_MIN_S", 0.0)
    d = _dir(tmp_path)
    model = tmp_path / "usual.gguf"
    model.write_bytes(b"the usual model")
    fake = FakeLlama(d, model)
    monkeypatch.setattr(H, "_http", fake.http)
    llm = MagicMock()
    llm.upstream_url = BASE + "/v1"
    llm.foreground_requests = 0
    llm._main_streams_open = 0
    llm._head_displaced = llm._head_displaced_early = llm._slot_moved_on = False
    llm._slot_session = llm._slot_request = llm._conv_saved = llm._conv_last_saved = None
    llm._slot_head_tag = llm._restore_check = None
    llm._outside_slot_dirty = False
    mock_context.llm_client = llm
    st = {"lines": [], "info": [], "sent": 0, "head_n": 28213}
    monkeypatch.setattr(A, "pretty_log", lambda title, msg=None, **kw: st["lines"].append(str(msg)))
    monkeypatch.setattr(L, "pretty_log", lambda title, msg=None, **kw: st["lines"].append(str(msg)))
    monkeypatch.setattr(A.logger, "info", lambda fmt, *a: st["info"].append(fmt % a if a else str(fmt)))

    async def cc(payload, **kw):
        st["payload"] = payload
        st["kw"] = kw
        if "before_dispatch" in kw:
            await kw["before_dispatch"]()
        if st.get("during_call"):
            st["during_call"]()
        st["sent"] += 1
        fake.slot_n = st["head_n"]                       # llama processed the head
        return _resp(st["head_n"], 0)
    llm.chat_completion = cc
    return GhostAgent(mock_context), llm, st, fake


def _turn(st, text="new question"):
    p = dict(st["payload"])
    p["messages"] = list(p.get("messages") or []) + [{"role": "user", "content": text}]
    return p


def _owner_call(llm, st, key="sess-a", rid="req-a1"):
    """The conversation's own call just ran on the slot (the real note)."""
    t1, t2 = slot_session_context.set(key), request_id_context.set(rid)
    try:
        L.note_main_slot_prompt(llm, _turn(st, "its message"), "turn", usage=(40000, 50))
    finally:
        slot_session_context.reset(t1)
        request_id_context.reset(t2)


async def _switch(agent, payload, key, rid):
    t1, t2 = slot_session_context.set(key), request_id_context.set(rid)
    try:
        await agent._slot_switch_before(payload)
    finally:
        slot_session_context.reset(t1)
        request_id_context.reset(t2)


def _names(d):
    return sorted(p.name for p in d.iterdir())


# ════════ H15: the real save and restore, at the agent level ════════

async def test_a_warm_up_saves_the_head_through_the_real_save(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    key = agent._head_plan_last["key"]
    assert fake.saves == [f"head-{key}.part.bin"] and H.stored_tokens(fake.d, key) == 28213
    meta = json.loads((fake.d / f"head-{key}.json").read_text())
    assert meta["identity"] == H.server_identity(BASE) and "identity_checked" not in meta
    assert meta["bytes"] == 1000 and (fake.d / f"head-{key}.bin").stat().st_size == 1000


async def test_a_conversation_goes_to_disk_and_comes_back_through_the_real_calls(real):
    """The whole loop with nothing stubbed but HTTP: saved before another
    conversation's call, restored before its own next request, bound to what
    its own call opened with, and its reuse judged on the part past the head."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)                                             # its prompt was 40,000 tokens
    fake.slot_n = 40050                                              # … and it generated 50
    await _switch(agent, _turn(st), "sess-b", "req-b1")              # another conversation's first call
    ckey = H.conv_key(H.server_identity(BASE), "sess-a")
    meta = json.loads((fake.d / f"head-{ckey}.json").read_text())
    assert meta["kind"] == H.CONV and meta["n_tokens"] == 40050
    assert meta["head"] == L.head_bind_tag(_turn(st)) and meta["head_n"] == 28213 and "prompt_n" not in meta
    assert fake.restores[-1] == f"head-{agent._head_plan_last['key']}.bin"   # sess-b starts on the head
    fake.restores.clear()
    back = _turn(st)
    await _switch(agent, back, "sess-a", "req-a2")                   # sess-a returns
    assert fake.restores == [f"head-{ckey}.bin"] and fake.slot_n == 40050
    assert llm._slot_session == "sess-a" and llm._slot_head_tag == meta["head"]
    assert llm._restore_check == {"payload": back, "what": "conversation", "key": ckey, "n": 40050, "head_n": 28213}
    assert llm._restore_check["payload"] is back


# ════════ H1: a conversation file is bound to what its state opens with (r2: by content) ════════

def _stage_conv(fake, session, *, head, level=None, n=41000, effort=None, plain=False):
    ident = H.server_identity(BASE)
    ckey = H.conv_key(ident, session, None if plain else level)
    meta = {"n_tokens": n, "kind": H.CONV, "identity": ident, "head_in_slot": True,
            "conv": H.conv_key(ident, session)}
    if head is not None:
        meta["head"] = head
    if effort:
        meta["effort"] = effort
    (fake.d / f"head-{ckey}.bin").write_bytes(b"state")
    (fake.d / f"head-{ckey}.json").write_text(json.dumps(meta))
    return ckey


def _with_tools(payload, names=("search",)):
    p = dict(payload)
    p["tools"] = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]
    return p


def _call(head, system=None, text="its message"):
    """A request's call on ``head``'s tool list, with ``system`` as its system
    text (None: the head's own)."""
    p = dict(head)
    sys0 = head["messages"][0]["content"] if system is None else system
    p["messages"] = [{"role": "system", "content": sys0}, {"role": "user", "content": text}]
    return p


#: the state × event table of the bound rule (r2: by CONTENT). The state on
#: disk was built by a call with (system text, tool list); the request about to
#: run sends (system text, tool list); the warm-up's reference is whatever the
#: last tick saw — the rule must not depend on it.
#:   S1 = the warmed system text, S2 = after a profile edit, PUB = the owner on Slack's public text
#:   T1 = the tool list, T2 = after a deploy changed it
#:   ref: which system text the last tick's reference holds ("S1" or "S2")
#:   file: None (no file) / (system, tools) of the call that built it / "untagged" (saved before §4UD)
#:   level: the file was saved at this request's effort level ("same") or another ("other")
#: → what goes into the slot ("conv" / "head") and whether the file is on disk afterwards
BOUND_TABLE = [
    # name                                  file            request        ref    level    slot    kept
    ("no file",                             None,           ("S1", "T1"),  "S1",  "same",  "head", None),
    ("same head",                           ("S1", "T1"),   ("S1", "T1"),  "S1",  "same",  "conv", True),
    ("same head, another level",            ("S1", "T1"),   ("S1", "T1"),  "S1",  "other", "head", True),
    ("profile edited, before the tick",     ("S2", "T1"),   ("S2", "T1"),  "S1",  "same",  "conv", True),   # r2-F2
    ("profile edited, saved before the tick, back after it",
                                            ("S2", "T1"),   ("S2", "T1"),  "S2",  "same",  "conv", True),   # r2-F2
    ("profile edited since the save",       ("S1", "T1"),   ("S2", "T1"),  "S2",  "same",  "head", False),  # r2-F4
    ("profile edited since, tick not yet",  ("S1", "T1"),   ("S2", "T1"),  "S1",  "same",  "head", False),  # r2-F4
    ("tool list changed since the save",    ("S1", "T2"),   ("S1", "T1"),  "S1",  "same",  "head", False),  # H1, req eaf6e7e8
    ("both changed, another level too",     ("S2", "T2"),   ("S1", "T1"),  "S1",  "other", "head", False),
    ("the owner on Slack, its own file",    ("PUB", "T1"),  ("PUB", "T1"), "S1",  "same",  "conv", True),
    ("the owner on Slack, a web state",     ("S1", "T1"),   ("PUB", "T1"), "S1",  "same",  "head", False),
    ("the web owner, a Slack state",        ("PUB", "T1"),  ("S1", "T1"),  "S1",  "same",  "head", False),
    ("saved before §4UD (no field)",        "untagged",     ("S1", "T1"),  "S1",  "same",  "head", False),
    ("saved before §4UD, another level",    "untagged",     ("S1", "T1"),  "S1",  "other", "head", False),
]


@pytest.mark.parametrize("name,file,request_,ref,level,want,kept", BOUND_TABLE, ids=[r[0] for r in BOUND_TABLE])
async def test_the_bound_head_table(real, monkeypatch, name, file, request_, ref, level, want, kept):
    agent, llm, st, fake = real
    monkeypatch.setattr(A._effort, "known_server_default", lambda: "medium")
    monkeypatch.setattr(A.GhostAgent, "_call_level", staticmethod(lambda payload: "medium"))
    await agent.warm_up_main_prefix(quiet=True)
    s1 = st["payload"]["messages"][0]["content"]
    systems = {"S1": s1, "S2": s1[:200] + " [profile: a new fact] " + s1[200:], "PUB": "Public Ghost, no profile."}
    tools = {"T1": ("search",), "T2": ("search", "a_new_tool")}

    def call(sys_name, tools_name):
        return _call(_with_tools(st["payload"], tools[tools_name]), systems[sys_name])
    L.note_head_reference(llm, call(ref, "T1"))                      # what the last tick saw
    req = call(*request_)
    ckey = None
    if file is not None:
        tag = None if file == "untagged" else L.head_bind_tag(call(*file))
        ckey = _stage_conv(fake, "sess-b", head=tag, level="medium",
                           effort=("xhigh" if level == "other" else "medium"), plain=(level == "other"))
    head_key = agent._head_plan_last["key"]
    fake.slot_n = 33000                                              # someone else's state
    fake.restores.clear()
    st["lines"].clear()
    await _switch(agent, req, "sess-b", "req-b9")
    assert fake.restores == [f"head-{ckey if want == 'conv' else head_key}.bin"]
    if kept is not None:
        assert (fake.d / f"head-{ckey}.bin").exists() is kept and (fake.d / f"head-{ckey}.json").exists() is kept
    said = [m for m in st["lines"] if "not restored — its file was saved on" in m and "discarded" in m]
    assert len(said) == (1 if kept is False else 0)                  # one line exactly when a file is discarded
    assert (llm._slot_session == "sess-b") is (want == "conv")


def test_the_tag_reads_the_calls_own_tools_and_system_text_and_nothing_else():
    head = _with_tools({"messages": [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "ok"}]})
    a = L.head_bind_tag(_call(head))
    assert a == L.head_bind_tag(_call(head, text="another message, a longer history"))   # not the conversation
    longer = _call(head)
    longer["messages"] += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "state block …"}]
    assert L.head_bind_tag(longer) == a
    assert L.head_bind_tag(_call(head, "SYSTEM ")) != a                      # the whole system text
    assert L.head_bind_tag(_call(_with_tools(head, ("search", "x")))) != a   # the tool list
    shape = L.head_shape(_call(head))
    assert a == f"{shape[3]}:{shape[1]}"                                     # `head_shape`'s own two hashes
    assert L.head_bind_tag(None) is None


async def test_a_request_between_a_profile_edit_and_the_tick_keeps_its_file(real):
    """r2-F2, end to end with the real note and save: the conversation's
    request runs after the edit and BEFORE the tick re-takes the reference;
    the tick re-takes it; the conversation is displaced and saved; its
    follow-up sends exactly what the state opens with. (Round 1 tagged the
    state by the stale reference and discarded the file.)"""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    head1 = _with_tools(st["payload"])
    L.note_head_reference(llm, head1)
    sys1 = head1["messages"][0]["content"]
    sys2 = sys1[:200] + " [profile: a new fact] " + sys1[200:]
    turn2 = _call(head1, sys2)
    t1, t2 = slot_session_context.set("sess-a"), request_id_context.set("req-a1")
    try:
        L.note_main_slot_prompt(llm, turn2, "turn", usage=(40000, 50))
    finally:
        slot_session_context.reset(t1)
        request_id_context.reset(t2)
    assert llm._slot_session == "sess-a"
    fake.slot_n = 40050
    L.note_head_reference(llm, _call(head1, sys2, "ok"))             # the next tick: the reference follows the edit
    assert await agent._conv_save(llm, "a gap job") is True
    ckey = H.conv_key(H.server_identity(BASE), "sess-a")
    L.release_slot_session(llm)
    fake.slot_n = 500
    follow = dict(turn2)
    follow["messages"] = turn2["messages"] + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "next"}]
    fake.restores.clear()
    await _switch(agent, follow, "sess-a", "req-a2")
    assert fake.restores == [f"head-{ckey}.bin"] and llm._slot_session == "sess-a"
    assert not any("saved on another head" in m for m in st["lines"])


async def test_the_owners_own_look_does_not_change_what_its_state_opens_with(real):
    """A call of the owner that is not the conversation's (its picture look)
    refreshes the record but built nothing of the state."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    L.note_head_reference(llm, _with_tools(st["payload"]))
    own = _call(_with_tools(st["payload"]))
    t1, t2 = slot_session_context.set("sess-a"), request_id_context.set("req-a1")
    try:
        L.note_main_slot_prompt(llm, own, "turn", usage=(40000, 50))
        tag = llm._slot_head_tag
        assert tag == L.head_bind_tag(own)
        L.note_main_slot_prompt(llm, {"messages": [{"role": "user", "content": "describe the picture"}]},
                                "look", usage=(900, 30))
    finally:
        slot_session_context.reset(t1)
        request_id_context.reset(t2)
    assert llm._slot_session == "sess-a" and llm._slot_moved_on is True
    assert llm._slot_head_tag == tag


async def test_a_state_with_no_record_of_what_it_opens_with_is_not_written(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    llm._slot_head_tag = None
    fake.saves.clear()
    assert await agent._conv_save(llm, "a test") is False and fake.saves == []
    assert any("no record of what its state opens with" in m for m in st["info"])


async def test_a_request_whose_payload_cannot_be_read_never_takes_an_untagged_file(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    ckey = _stage_conv(fake, "sess-b", head=None)
    fake.restores.clear()
    assert await agent._conv_restore(llm, agent._head_plan_last, "sess-b", "r", time.monotonic() + 5,
                                     payload=None) is False
    assert fake.restores == [] and not (fake.d / f"head-{ckey}.bin").exists()


def test_a_release_forgets_the_tag(real):
    agent, llm, st, fake = real
    llm._slot_head_tag = "a:b"
    L.release_slot_session(llm)
    assert llm._slot_head_tag is None


# ════════ H12: was the restore reused? ════════

@pytest.mark.parametrize("obj,want", [
    ({"usage": {"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 7}}}, 7),
    ({"usage": {"prompt_tokens": 9}, "timings": {"cache_n": 5, "prompt_n": 4}}, 5),
    ('data: {"usage": {"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 0}}}', 0),
    (b'data: {"choices": [], "timings": {"cache_n": 31}}', 31),
    ({"usage": {"prompt_tokens": 9}}, None),
    ("data: [DONE]", None), ("not json", None), (None, None),
])
def test_the_cached_count_is_read_from_usage_or_timings(obj, want):
    assert L.usage_cached(obj) == want


HEAD = {"messages": [{"role": "system", "content": "SYSTEM HEAD"}, {"role": "user", "content": "ok"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}]}
LOOK = {"messages": [{"role": "user", "content": "describe the picture"}]}
#: live sizes (11-08 … 11-11): the head 24,375 tokens; conversation files of 28,869 – 33,536
LIVE_HEAD = 24375


def _turn_of(head=HEAD):
    return {**head, "messages": head["messages"] + [{"role": "user", "content": "more"}]}


def _armed(tmp_path, *, what="conversation", n=33536, head_n=LIVE_HEAD):
    d = _dir(tmp_path)
    _store(d, "ck", n)
    payload = _turn_of()
    llm = SimpleNamespace(_head_ref=L.head_shape(HEAD), _restore_check=None,
                          _conv_saved=("sess-b", 1.0), _conv_last_saved=("sess-b", n, 1.0, "ck"))
    L.expect_restore_reuse(llm, payload, "ck", what, n, head_n=head_n)
    return d, llm, payload


def _check(monkeypatch, llm, payload, cached):
    lines, info = [], []
    monkeypatch.setattr(L, "pretty_log", lambda title, msg=None, **kw: lines.append((str(msg), kw.get("level"))))
    monkeypatch.setattr(L.logger, "info", lambda fmt, *a: info.append(fmt % a if a else str(fmt)))
    L.check_restore_reuse(llm, payload, cached)
    return lines, info


def _untouched(d, llm, n=33536):
    """r3: the reuse check deletes nothing and clears no mark — both kinds."""
    return (H.stored_tokens(d, "ck") == n and (d / "head-ck.bin").exists()
            and llm._conv_saved == ("sess-b", 1.0) and llm._conv_last_saved == ("sess-b", n, 1.0, "ck"))


@pytest.mark.parametrize("n", [33536, 30745, 28869])
@pytest.mark.parametrize("world,cached_of,reused", [
    ("resumed at its last message", lambda n: n - 1300, True),       # the live re-read: 0.9k–3.3k
    ("a small part past the head, a 3k state block re-read", lambda n: n - 3000, True),
    ("one token more than the margin", lambda n: LIVE_HEAD + L.RESTORE_PAST_HEAD_MIN + 1, True),
    ("the head and the margin", lambda n: LIVE_HEAD + L.RESTORE_PAST_HEAD_MIN, False),
    ("only the head was reused", lambda n: LIVE_HEAD, False),
    ("far below the head", lambda n: 9000, False),
    ("nothing was reused", lambda n: 0, False),
])
def test_a_conversation_that_reused_nothing_past_the_head_is_said_and_nothing_is_deleted(
        monkeypatch, tmp_path, n, world, cached_of, reused):
    """r2-F3 / r3: a share of the WHOLE file passed a conversation re-read from
    its first message (24.4k of 28.9k is 84 %); a share of the part past the
    head fired on a correct resume of a small conversation, which re-reads its
    last state block. The wrong file lands at or below the head: that, plus a
    256-token margin, is the line. LOG ONLY: the bound tag refuses the wrong
    file before the restore, and a deleted file came back under the same key."""
    assert L.RESTORE_PAST_HEAD_MIN == 256
    d, llm, payload = _armed(tmp_path, n=n)
    cached = cached_of(n)
    lines, info = _check(monkeypatch, llm, payload, cached)
    assert _untouched(d, llm, n) and llm._restore_check is None
    if reused:
        assert not lines and any("was reused" in m for m in info)
    else:
        assert len(lines) == 1 and lines[0][1] == "WARNING" and "head-ck" in lines[0][0] and "kept" in lines[0][0]
        assert f"restored {n:,} tokens, reused {cached:,}" in lines[0][0] and "nothing past the head" in lines[0][0]


@pytest.mark.parametrize("head_n", [None, 0, -5, "24375", True])
def test_a_file_that_does_not_record_the_heads_size_is_not_judged(monkeypatch, tmp_path, head_n):
    d, llm, payload = _armed(tmp_path, head_n=head_n)
    lines, info = _check(monkeypatch, llm, payload, 0)
    assert _untouched(d, llm) and not lines and llm._restore_check is None
    assert any("not judged" in m for m in info)


def test_no_cached_count_is_no_verdict(monkeypatch, tmp_path):
    d, llm, payload = _armed(tmp_path)
    lines, info = _check(monkeypatch, llm, payload, None)
    assert _untouched(d, llm) and not lines and llm._restore_check is None
    assert any("no cached-token count" in m for m in info)


def test_only_the_call_the_restore_was_made_for_answers(monkeypatch, tmp_path):
    """r2-F3c: the record is held by the payload object — another call (a
    look, another request, a later request that reuses a `sched-` id, an equal
    payload built again) neither answers for it nor ends it; it ends with its
    own call, with or without a reply."""
    d, llm, payload = _armed(tmp_path)
    tok = request_id_context.set("sched-daily")
    try:
        for other in (LOOK, _turn_of(), dict(payload)):
            lines, _ = _check(monkeypatch, llm, other, 0)
            assert not lines and llm._restore_check["payload"] is payload
        L.drop_restore_check(llm, _turn_of())                        # not its call: nothing dropped
        assert llm._restore_check is not None
        L.drop_restore_check(llm, payload)                           # its call failed
        assert llm._restore_check is None
        lines, _ = _check(monkeypatch, llm, payload, 0)              # a later call, same id: nothing armed
        assert not lines and _untouched(d, llm)
    finally:
        request_id_context.reset(tok)


@pytest.mark.parametrize("n,cached,loud", [(28213, 0, True), (28213, 14106, True), (28213, 14107, False),
                                           (28212, 14105, True), (28212, 14106, False),   # an even head: exactly half
                                           (28213, 28213, False)])
def test_a_head_that_gave_under_half_is_said_and_nothing_is_deleted(monkeypatch, tmp_path, n, cached, loud):
    """r2-F3a: a head file's key is a hash of its content — right for the calls
    that open with it whatever one call reused. A low count is one line."""
    d, llm, payload = _armed(tmp_path, what="head", n=n, head_n=None)
    lines, info = _check(monkeypatch, llm, payload, cached)
    assert _untouched(d, llm, n) and llm._restore_check is None
    assert len(lines) == (1 if loud else 0)
    if loud:
        assert "not reused" in lines[0][0] and "kept" in lines[0][0] and lines[0][1] == "WARNING"
        assert f"restored {n:,} tokens, reused {cached:,}" in lines[0][0] and "under half of the head" in lines[0][0]


async def test_one_low_count_does_not_cost_the_next_conversation_its_head(real):
    """The loop round 1 opened: discard → no head file → every first call
    re-reads the head until an idle gap re-makes it → the next low count …"""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    key = agent._head_plan_last["key"]
    call = _turn(st)
    fake.slot_n = 33000
    await _switch(agent, call, "sess-c", "req-c1")
    assert llm._restore_check["what"] == "head" and llm._restore_check["payload"] is call
    L.check_restore_reuse(llm, call, 0)                              # an outside call took the slot in between
    assert H.stored_tokens(fake.d, key) == 28213
    L.release_slot_session(llm)
    fake.slot_n = 33000
    fake.restores.clear()
    await _switch(agent, _turn(st), "sess-d", "req-d1")
    assert fake.restores == [f"head-{key}.bin"]


def _client(tmp_path, monkeypatch):
    d, _, _ = _armed(tmp_path)
    c = _bare_client()
    c._head_ref = L.head_shape(HEAD)
    c._conv_saved = c._conv_last_saved = None
    c._main_streams_open = 0
    c._note_usage = lambda r: None
    c._served_on_main_box = lambda served: True
    payload = _turn_of()
    L.expect_restore_reuse(c, payload, "ck", "conversation", 33536, head_n=LIVE_HEAD)
    lines = []
    monkeypatch.setattr(L, "pretty_log", lambda title, msg=None, **kw: lines.append(str(msg)))
    return d, c, payload, lines


def _reply(cached):
    return {"choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 33500, "completion_tokens": 3,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


@pytest.mark.parametrize("cached,said", [(LIVE_HEAD, True), (31900, False)])
async def test_a_finished_main_call_reports_its_reuse(monkeypatch, tmp_path, cached, said):
    """The wire: `chat_completion` hands the reply's cached count to the check
    (no extra llama call)."""
    d, c, payload, lines = _client(tmp_path, monkeypatch)

    async def do(p, *a, **k):
        return _reply(cached)
    c._do_chat_completion = do
    await c.chat_completion(payload)
    assert bool(lines) is said and c._restore_check is None and H.stored_tokens(d, "ck") == 33536


async def test_a_reply_served_by_another_node_is_no_verdict(monkeypatch, tmp_path):
    d, c, payload, lines = _client(tmp_path, monkeypatch)
    c._served_on_main_box = lambda served: False

    async def do(p, *a, **k):
        return _reply(0)
    c._do_chat_completion = do
    await c.chat_completion(payload)
    assert not lines and c._restore_check is None


@pytest.mark.parametrize("exc", [RuntimeError("upstream 500"), asyncio.CancelledError()])
async def test_an_expectation_ends_with_a_call_that_failed_or_was_cancelled(monkeypatch, tmp_path, exc):
    d, c, payload, lines = _client(tmp_path, monkeypatch)

    async def do(p, *a, **k):
        raise exc
    c._do_chat_completion = do
    with pytest.raises(type(exc)):
        await c.chat_completion(payload)
    assert c._restore_check is None and not lines


def _usage_line(cached=None, where="timings"):
    body = {"choices": [], "usage": {"prompt_tokens": 33500, "completion_tokens": 3}}
    if cached is not None and where == "timings":
        body["timings"] = {"cache_n": cached}
    elif cached is not None:
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
    return "data: " + json.dumps(body) + "\n\n"


def _timings_line(cached):
    return ("data: " + json.dumps({"choices": [{"delta": {"content": "x"}}], "timings": {"cache_n": cached}})
            + "\n\n").encode()


@pytest.mark.parametrize("chunks,said", [
    ([_usage_line(LIVE_HEAD)], True),
    ([_usage_line(LIVE_HEAD, "usage")], True),
    ([_usage_line(31900)], False),
    ([_usage_line(LIVE_HEAD), _usage_line(None)], True),             # a later usage line without a count
    ([_timings_line(LIVE_HEAD)], True),                              # r3: a chunk with timings and no usage
    ([_timings_line(31900), _usage_line(None)], False),
    ([_usage_line(None)], False),                                    # no count: no verdict
    ([], False),                                                     # cut before any count: no verdict
])
async def test_a_stream_reports_its_reuse_from_a_usage_or_timings_chunk(monkeypatch, tmp_path, chunks, said):
    d, c, payload, lines = _client(tmp_path, monkeypatch)

    async def stream(p, use_coding):
        yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
        for ch in chunks:
            yield ch
    c._do_stream_chat_completion = stream
    async for _ in c.stream_chat_completion(payload):
        pass
    assert bool(lines) is said and H.stored_tokens(d, "ck") == 33536
    assert c._restore_check is None                                  # it ended with its stream either way


async def test_a_stream_that_fails_ends_its_expectation_and_another_stream_does_not(monkeypatch, tmp_path):
    d, c, payload, lines = _client(tmp_path, monkeypatch)

    async def other(p, use_coding):
        yield _usage_line(0)
    c._do_stream_chat_completion = other
    async for _ in c.stream_chat_completion(_turn_of()):             # another request's stream: count 0
        pass
    assert c._restore_check["payload"] is payload and not lines

    async def broken(p, use_coding):
        raise RuntimeError("upstream closed")
        yield ""  # pragma: no cover
    c._do_stream_chat_completion = broken
    with pytest.raises(RuntimeError):
        async for _ in c.stream_chat_completion(payload):
            pass
    assert c._restore_check is None and not lines


async def test_what_arms_the_check(real):
    """A conversation restore and a head restore for a call that opens with
    that head arm it, for that call's payload. A state CLAIMED in the slot
    (nothing was restored), a tools-only call's head and the nearest head do not."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    key = agent._head_plan_last["key"]
    st["payload"] = _with_tools(st["payload"])
    L.note_head_reference(llm, st["payload"])
    # claimed: the save just made is still what the slot holds
    _owner_call(llm, st)
    fake.slot_n = 40050
    assert await agent._conv_save(llm, "a gap job that then deferred") is True
    L.release_slot_session(llm)
    fake.restores.clear()
    st["lines"].clear()
    await _switch(agent, _turn(st), "sess-a", "req-a2")
    assert fake.restores == [] and llm._slot_session == "sess-a" and llm._restore_check is None
    assert any("already in the slot" in m for m in st["lines"])
    # a tools-only call (the owner on Slack), no file of its own: the head, not judged
    L.release_slot_session(llm)
    fake.slot_n = 33000
    await _switch(agent, _call(st["payload"], "Public Ghost, no profile."), "slack:t1:U0", "req-s1")
    assert fake.restores == [f"head-{key}.bin"] and llm._restore_check is None
    # the nearest head: this plan has no file of its own
    L.release_slot_session(llm)
    real_plan = agent._head_plan_last
    agent._head_plan_last = {**real_plan, "key": "0123456789abcdef", "have": False}
    fake.slot_n = 33000
    await _switch(agent, _turn(st), "sess-n", "req-n1")
    assert fake.restores == [f"head-{key}.bin"] * 2 and llm._restore_check is None
    # this head, for a call that opens with it
    agent._head_plan_last = real_plan
    L.release_slot_session(llm)
    fake.slot_n = 33000
    call = _turn(st)
    await _switch(agent, call, "sess-c", "req-c1")
    rec = llm._restore_check
    assert rec == {"payload": call, "what": "head", "key": key, "n": 28213, "head_n": None} and rec["payload"] is call


# ════════ H2: the identity a save is filed under is the running server's ════════

def test_a_save_made_after_llama_changed_model_is_dropped(real):
    """llama restarted on model B between the caller's plan and its save."""
    agent, llm, st, fake = real
    planned = H.server_identity(BASE)
    _store(fake.d, "k", 28213)
    (fake.d / "head-k.bin").write_bytes(b"the previous good save")
    fake.on_save = fake.swap_model                                   # the swap lands before llama answers
    ok, detail = H.save(BASE, fake.d, "k", 28213, planned)
    assert not ok and "another model" in detail
    assert _names(fake.d) == ["head-k.bin", "head-k.json"]           # no .part left
    assert (fake.d / "head-k.bin").read_bytes() == b"the previous good save"


def test_a_save_on_the_planned_server_is_kept(real):
    agent, llm, st, fake = real
    ok, detail = H.save(BASE, fake.d, "k", 28213, H.server_identity(BASE))
    assert ok, detail
    meta = json.loads((fake.d / "head-k.json").read_text())
    assert meta["identity"] == H.server_identity(BASE) and "identity_checked" not in meta


@pytest.mark.parametrize("fails,kept", [(1, True), (2, False)])
def test_a_state_whose_server_cannot_be_confirmed_is_not_kept(real, fails, kept):
    """r2-F10: /props is asked twice after the save. One miss is asked again;
    no answer at all drops the save (round 1 kept it with a field nobody read)."""
    agent, llm, st, fake = real
    planned = H.server_identity(BASE)
    _store(fake.d, "k", 28213)
    (fake.d / "head-k.bin").write_bytes(b"the previous good save")
    real_http, seen = fake.http, {"props": 0}

    def http(url, body=None, timeout=5.0):
        if url.endswith("/props"):
            seen["props"] += 1
            if seen["props"] <= fails:
                raise TimeoutError("timed out")
        return real_http(url, body, timeout)
    H._http = http                                                   # (the fixture's monkeypatch restores it)
    ok, detail = H.save(BASE, fake.d, "k", 28213, planned)
    assert ok is kept and seen["props"] == min(fails + 1, 2)
    if kept:
        assert (fake.d / "head-k.bin").stat().st_size == 1000
    else:
        assert "/props did not answer" in detail and "not kept" in detail
        assert _names(fake.d) == ["head-k.bin", "head-k.json"]       # no .part left
        assert (fake.d / "head-k.bin").read_bytes() == b"the previous good save"


async def test_a_conversation_save_with_props_down_is_said_and_not_kept(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    real_http = fake.http

    def http(url, body=None, timeout=5.0):
        if url.endswith("/props"):
            raise TimeoutError("timed out")
        return real_http(url, body, timeout)
    H._http = http
    st["lines"].clear()
    assert await agent._conv_save(llm, "a background call") is False
    ckey = H.conv_key(agent._head_plan_last["identity"], "sess-a")
    assert not (fake.d / f"head-{ckey}.bin").exists() and not list(fake.d.glob("*.part.bin"))
    assert sum("not saved before a background call" in m and "/props did not answer" in m
               for m in st["lines"]) == 1


async def test_the_owner_head_is_not_filed_under_the_model_before_a_swap(real):
    """The warm-up waits between its plan and its save; llama is swapped then."""
    agent, llm, st, fake = real
    st["during_call"] = fake.swap_model
    await agent.warm_up_main_prefix(quiet=True)
    assert [p for p in _names(fake.d) if p.endswith(".bin")] == []
    assert any("head not saved" in m and "another model" in m for m in st["lines"])


async def test_a_conversation_is_not_filed_under_the_model_before_a_swap(real):
    """The reviewer's scenario: the plan carries model A, llama runs B."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    plan_a = agent._head_plan_last["identity"]
    _owner_call(llm, st)
    fake.slot_n = 40050
    fake.swap_model()
    st["lines"].clear()
    fake.calls.clear()
    assert await agent._conv_save(llm, "another conversation's request") is False
    assert not (fake.d / f"head-{H.conv_key(plan_a, 'sess-a')}.bin").exists()
    assert any("not saved" in m and "another model" in m for m in st["lines"])
    assert not [c for c in fake.calls if c.endswith("action=save")]  # refused before the multi-GB write


async def test_a_swap_during_the_save_itself_is_caught_at_the_sink(real):
    """The pre-check passed (model A), the swap lands while llama saves."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    plan_a = agent._head_plan_last["identity"]
    _owner_call(llm, st)
    fake.slot_n = 40050
    fake.on_save = fake.swap_model
    st["lines"].clear()
    assert await agent._conv_save(llm, "another conversation's request") is False
    assert not (fake.d / f"head-{H.conv_key(plan_a, 'sess-a')}.bin").exists()
    assert not list(fake.d.glob("*.part.bin"))
    assert any("not saved" in m and "another model" in m for m in st["lines"])


async def test_the_head_is_not_restored_over_an_outside_call_on_another_model(real):
    """`_outside_slot_restore` had no identity re-read."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    fake.slot_n = 3300
    llm._outside_slot_dirty = True
    fake.swap_model()
    fake.restores.clear()
    await agent._outside_slot_restore(llm, hs, _turn(st), time.monotonic() + 5)
    assert fake.restores == []
    assert any("another model" in m for m in st["info"])


#: /props answers → the identity commit 4500c7f's `server_identity` returns for
#: them (computed by running that file's function; every head and conversation
#: file on disk was saved under this format, so it must not move by a byte)
BASELINE_IDENTITIES = [
    ({"model_path": "/models/usual-27b.gguf", "build_info": "b11960-abc",
      "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
      "default_generation_settings": {"n_ctx": 196608}},
     "/models/usual-27b.gguf|b11960-abc|c5670fa26baec931"),
    ({"model_path": "/models/usual-27b.gguf", "build_info": "b11960-abc", "chat_template": "T",
      "default_generation_settings": {"n_ctx": 65536}},
     "/models/usual-27b.gguf|b11960-abc|c2c53d6694821425"),
    ({"build_info": None, "chat_template": None}, "||da39a3ee5e6b4b0d"),
    ({"model_path": "", "build_info": "b1"}, "|b1|da39a3ee5e6b4b0d"),
]


@pytest.mark.parametrize("props,want", BASELINE_IDENTITIES)
def test_the_identity_is_the_baseline_format_byte_for_byte(monkeypatch, props, want):
    """r2-F1: round 1 added the model file's size and mtime and the window.
    The stamp was read from the PATH, not from the file llama has open, so a
    GGUF replaced under a running llama filed the old model's state under the
    new one's identity; and every existing file became another server's."""
    monkeypatch.setattr(H, "_http", lambda url, body=None, timeout=5.0: props)
    assert H.server_identity(BASE) == want


def test_the_identity_does_not_read_the_model_file_or_the_window(real):
    agent, llm, st, fake = real
    a = H.server_identity(BASE)
    fake.n_ctx = 65536
    p = pathlib.Path(fake.model_path)
    p.write_bytes(b"replaced in place under the same path: not detected (a documented limit)")
    assert H.server_identity(BASE) == a
    p.unlink()
    assert H.server_identity(BASE) == a


def test_the_launchers_helper_finds_a_head_saved_under_the_baseline_identity(real):
    """The llama launcher runs this module's `restore_on_start`; the agent
    saves through `save`. Both must name the server the same way — the way the
    files already on disk were named."""
    agent, llm, st, fake = real
    baseline = f"{fake.model_path}|b1|" + __import__("hashlib").sha1(b"T").hexdigest()[:16]
    (fake.d / "head-old.bin").write_bytes(b"state")
    (fake.d / "head-old.json").write_text(json.dumps({"n_tokens": 28213, "identity": baseline,
                                                      "kind": H.OWNER, "saved_at": time.time() - 86400}))
    fake.fresh, fake.slot_n = True, None
    real_http = fake.http
    H._http = lambda url, body=None, timeout=5.0: ({} if url.endswith("/health") else real_http(url, body, timeout))
    line = H.restore_on_start(BASE, fake.d, wait_s=1, sleep=lambda s: None)
    assert line.startswith("restored head-old") and fake.restores == ["head-old.bin"]
    assert (fake.d / "head-old.bin").exists()                        # not pruned as another server's


# ════════ H3: no plan ════════

def _props_down(fake, exc=None):
    real_http = fake.http

    def http(url, body=None, timeout=5.0):
        if url.endswith("/props"):
            raise exc or TimeoutError("timed out")
        return real_http(url, body, timeout)
    H._http = http
    return real_http


async def test_after_a_tick_whose_plan_failed_the_next_request_plans_and_restores(real):
    """H3, the restore half: one /props timeout in a tick turned every restore
    off until the next good tick. No plan is KEPT over a failure (r2); the
    request that needs one makes it."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    key = agent._head_plan_last["key"]
    real_http = _props_down(fake)
    st["lines"].clear()
    await agent.warm_up_main_prefix(quiet=True)                      # the tick whose plan failed
    assert agent._head_plan_last is None
    H._http = real_http                                              # llama answers again
    fake.slot_n = 41000
    fake.restores.clear()
    await _switch(agent, _turn(st), "sess-b", "req-b1")
    assert fake.restores == [f"head-{key}.bin"] and agent._head_plan_last["key"] == key
    assert any("save plan made" in m for m in st["lines"])


async def test_a_tick_with_no_plan_sends_nothing_over_a_kept_conversation(real):
    """r2-F6: the plan failed, so nothing could save the conversation that
    holds the slot — and the warm-up's bare prefill went over it."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    _props_down(fake)
    sent, saves = st["sent"], list(fake.saves)
    st["lines"].clear()
    st["info"].clear()
    agent._head_rewarm_retry = False
    await agent.warm_up_main_prefix(quiet=True)
    await agent.warm_up_main_prefix(quiet=True)                      # and the tick after it
    assert st["sent"] == sent and fake.saves == saves                # nothing reached llama's slot
    assert llm._slot_session == "sess-a" and fake.slot_n == 40050
    assert agent._head_rewarm_retry is True                          # the loop will not skip the next tick
    said = [m for m in st["lines"] + st["info"] if "no save plan this tick" in m]
    assert len(said) == 1 and "TimeoutError" in said[0]              # one line for the same reason
    assert sum("no save plan (" in m for m in st["lines"]) == 1


async def test_saves_off_for_good_still_warms_the_head_and_posts_no_restore(real):
    """r2-F7: llama without --slot-save-path. The agent then works as with
    head saves off: the tick prefills the head; no plan is kept, so no request
    posts a restore llama would refuse; one loud line, not one per request."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)                      # a good plan from before
    fake.no_flag = True
    H._unsupported.add(BASE)
    sent = st["sent"]
    st["lines"].clear()
    st["info"].clear()
    await agent.warm_up_main_prefix(quiet=True)
    await agent.warm_up_main_prefix(quiet=True)
    assert agent._head_plan_last is None and st["sent"] == sent + 2  # the prefill goes out, as with saves off
    assert "before_dispatch" not in st["kw"]
    fake.calls.clear()
    for i in range(5):
        fake.slot_n = 33000
        for _call_no in range(3):                                    # three calls of each request
            await _switch(agent, _turn(st), f"sess-{i}", f"req-{i}")
    assert [c for c in fake.calls if "action=restore" in c] == []
    # each request's one attempt re-checks the mark (an empty-name save llama answers at once), no more
    assert len([c for c in fake.calls if "action=save" in c]) == 5
    loud = [m for m in st["lines"] if "--slot-save-path" in m]
    assert len(loud) == 1 and "no save plan" in loud[0]
    assert sum("save plan attempt" in m for m in st["info"]) == 0    # the reason on record did not change


async def test_a_members_plan_does_not_rewrite_the_owners_reason(real):
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    why = agent._head_plan_why
    assert "HTTPError" in why
    fake.down = TimeoutError("timed out")
    assert await agent._member_head_plan(llm) is None                # fails for another reason
    assert agent._head_plan_why == why
    fake.down = None
    assert isinstance(await agent._member_head_plan(llm), dict)      # … or succeeds
    assert agent._head_plan_why == why and agent._head_plan_last is None


async def test_a_boot_while_llama_loads_installs_the_hooks_and_the_first_request_plans(real):
    """Scenario B (11-11 07:45): no hook for a whole tick, silently."""
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    llm.before_main_dispatch = llm.before_side_main_dispatch = llm.before_background_dispatch = None
    await agent.warm_up_main_prefix()
    assert agent._head_plan_last is None and st["sent"] == 0
    assert llm.before_main_dispatch == agent._slot_switch_before
    assert llm.before_side_main_dispatch == agent._side_call_before
    assert llm.before_background_dispatch == agent._conv_before_background
    assert sum("no save plan (" in m and "asked again" in m for m in st["lines"]) == 1
    # llama is up; an earlier run saved this head
    fake.down = None
    plan = await agent._head_save_plan(llm, agent._head_payload_last)
    _store(fake.d, plan["key"], 28213)
    (fake.d / f"head-{plan['key']}.json").write_text(json.dumps(
        {"n_tokens": 28213, "identity": plan["identity"], "kind": H.OWNER}))
    fake.slot_n = 500
    st["lines"].clear()
    await _switch(agent, _turn({"payload": agent._head_payload_last}), "sess-b", "req-b1")
    assert isinstance(agent._head_plan_last, dict)
    assert fake.restores == [f"head-{plan['key']}.bin"]
    assert any("save plan made" in m for m in st["lines"])


async def test_with_head_saves_off_a_warm_up_installs_no_hook_and_keeps_no_plan(real, monkeypatch):
    agent, llm, st, fake = real
    monkeypatch.delenv(H.ENV)
    llm.before_main_dispatch = llm.before_side_main_dispatch = llm.before_background_dispatch = None
    await agent.warm_up_main_prefix(quiet=True)
    assert st["sent"] == 1 and agent._head_plan_last is None
    assert (llm.before_main_dispatch, llm.before_side_main_dispatch, llm.before_background_dispatch) == (None,) * 3
    assert not any("no save plan" in m for m in st["lines"])


async def test_plan_attempts_are_bounded_back_off_and_say_a_reason_once(real, monkeypatch):
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    clock = [1000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: clock[0])
    attempts = []
    real_plan = agent._head_save_plan

    async def plan(llm_, payload):
        attempts.append(clock[0])
        return await real_plan(llm_, payload)
    agent._head_save_plan = plan
    st["info"].clear()
    for _ in range(50):                                              # a request's 50 calls in one instant
        assert await agent._head_plan_retry(llm) is None
    assert len(attempts) == 1
    for _ in range(400):                                             # 400 s, asked every second
        clock[0] += 1.0
        await agent._head_plan_retry(llm)
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert gaps[:4] == [10.0, 20.0, 40.0, 80.0] and all(g <= A.PLAN_RETRY_MAX_S for g in gaps)
    assert len(attempts) <= 8
    assert sum("save plan attempt" in m for m in st["info"]) == 0    # the boot's reason, already said
    fake.down = TimeoutError("timed out")                            # another reason: one line, then silence
    for _ in range(3):
        clock[0] += A.PLAN_RETRY_MAX_S
        await agent._head_plan_retry(llm)
    assert sum("save plan attempt" in m and "TimeoutError" in m for m in st["info"]) == 1
    fake.down = None
    clock[0] += A.PLAN_RETRY_MAX_S
    made = await agent._head_plan_retry(llm)
    assert isinstance(made, dict) and agent._head_plan_tries == 0
    # a later failure starts over: its first attempt is not held back by the old back-off
    _props_down(fake)
    await agent.warm_up_main_prefix(quiet=True)
    assert agent._head_plan_last is None
    n = len(attempts)
    await agent._head_plan_retry(llm)
    assert len(attempts) == n + 1


async def test_the_polls_back_off_does_not_shut_a_request_out(real, monkeypatch):
    """r3 scenario A: llama was down for minutes, the 5 s poll ratcheted its
    gate to 120 s; llama answers; a request arrives INSIDE the gate. It has its
    own attempt (one per request id) — it plans and restores."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)                      # a head file exists
    key = agent._head_plan_last["key"]
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix(quiet=True)
    assert agent._head_plan_last is None
    clock = [5000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: clock[0])
    for _ in range(300):                                             # the poll, every 5 s for 25 min
        clock[0] += 5.0
        assert await agent._head_plan_retry(llm) is None
    assert agent._head_plan_retry_at - clock[0] > 60                 # the gate is far away
    fake.down = None                                                 # llama answers
    assert await agent._head_plan_retry(llm) is None      # the poll itself still waits
    fake.slot_n = 41000
    fake.restores.clear()
    await _switch(agent, _turn(st), "sess-b", "req-b1")
    assert isinstance(agent._head_plan_last, dict) and fake.restores == [f"head-{key}.bin"]


async def test_a_request_gets_one_attempt_not_one_per_call(real):
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    attempts = []
    real_plan = agent._head_save_plan

    async def plan(llm_, payload):
        attempts.append(1)
        return await real_plan(llm_, payload)
    agent._head_save_plan = plan
    tok = request_id_context.set("req-1")
    try:
        for _ in range(30):                                          # its 30 calls
            assert await agent._head_plan_retry(llm) is None
    finally:
        request_id_context.reset(tok)
    assert len(attempts) == 1
    tok = request_id_context.set("req-2")                            # the next request: its own
    try:
        await agent._head_plan_retry(llm)
    finally:
        request_id_context.reset(tok)
    assert len(attempts) == 2


async def test_whoever_asks_during_an_attempt_gets_its_plan(real):
    """r3 scenario B: the poll's attempt is in flight; a request asks 0.1 s
    in. It awaits that attempt (it got None, and nothing was restored). A
    waiter that is cancelled does not cancel the attempt for the others."""
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    fake.down = None
    gate, attempts = asyncio.Event(), []
    real_plan = agent._head_save_plan

    async def slow(llm_, payload):
        attempts.append(1)
        await gate.wait()
        return await real_plan(llm_, payload)
    agent._head_save_plan = slow

    async def request(rid):
        tok = request_id_context.set(rid)
        try:
            return await agent._head_plan_retry(llm)
        finally:
            request_id_context.reset(tok)
    poll = asyncio.ensure_future(agent._head_plan_retry(llm))
    await asyncio.sleep(0.01)
    gone = asyncio.ensure_future(request("req-gone"))
    r1 = asyncio.ensure_future(request("req-1"))
    r2 = asyncio.ensure_future(request("req-2"))
    await asyncio.sleep(0.01)
    gone.cancel()                                                    # its user hung up
    with pytest.raises(asyncio.CancelledError):
        await gone
    gate.set()
    got = await asyncio.gather(poll, r1, r2)
    assert len(attempts) == 1                                        # one attempt served all
    assert all(isinstance(g, dict) and g is agent._head_plan_last for g in got)


async def test_a_plan_a_warm_up_made_during_the_attempt_is_the_one_kept(real):
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    fake.down = None
    real_plan = agent._head_save_plan
    newer = {}

    async def plan(llm_, payload):
        mine = await real_plan(llm_, payload)
        newer.update({**mine, "key": "the-warm-ups-plan"})
        agent._head_plan_last = newer                                # a tick's warm-up planned meanwhile
        return mine
    agent._head_save_plan = plan
    st["lines"].clear()
    assert await agent._head_plan_retry(llm) is newer and agent._head_plan_last is newer
    assert not any("save plan made" in m for m in st["lines"])


async def test_a_plan_made_on_a_retry_also_serves_the_conversation_save(real):
    """§4RW-e: a conversation save outlives a failed plan — the plan a retry
    made counts as the last good one for it too."""
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    fake.down = None
    assert isinstance(await agent._head_plan_retry(llm), dict)
    _owner_call(llm, {"payload": agent._head_payload_last})
    fake.slot_n = 40050
    agent._head_plan_last = None                                     # a later tick's plan failed …
    agent._head_plan_retry_at = time.monotonic() + 999               # … and the bounded retry is not due
    assert await agent._conv_save(llm, "a background call") is True


async def test_the_rewarm_wait_asks_for_a_missing_plan(real):
    agent, llm, st, fake = real
    fake.down = _http_error(503, "Loading model")
    await agent.warm_up_main_prefix()
    assert agent._head_plan_last is None
    fake.down = None
    slept = []

    async def sleep(s):
        slept.append(s)
    await agent._rewarm_wait(10.0, sleep)
    assert isinstance(agent._head_plan_last, dict) and slept[0] == A.HEAD_POLL_S


async def test_a_side_call_with_no_plan_makes_one_and_saves(real, monkeypatch):
    """`_conv_save` with no plan asked nobody; the conversation was lost."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    agent._head_plan_last = agent._conv_plan_last = None
    assert await agent._conv_save(llm, "this request's picture look") is True
    assert isinstance(agent._head_plan_last, dict)


# ════════ r3: the same two asks before a restore as after a save ════════

@pytest.mark.parametrize("fails,posted", [(1, True), (2, False)])
async def test_one_slow_props_does_not_cost_the_restore(real, fails, posted):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    real_http, seen = fake.http, {"props": 0}

    def http(url, body=None, timeout=5.0):
        if url.endswith("/props"):
            seen["props"] += 1
            if seen["props"] <= fails:
                raise TimeoutError("timed out")
        return real_http(url, body, timeout)
    H._http = http
    fake.restores.clear()
    ok, detail, _dt = await agent._restore_file(hs)
    assert seen["props"] == min(fails + 1, 2)
    if posted:
        assert ok is True and fake.restores == [f"head-{hs['key']}.bin"]
    else:
        assert (ok, detail) == (None, A.RESTORE_NO_PROPS) and fake.restores == []


async def test_the_save_and_the_restore_ask_through_the_one_helper(real, monkeypatch):
    """One rule in one place, so the two sides cannot drift again."""
    agent, llm, st, fake = real
    asked = []
    monkeypatch.setattr(H, "confirmed_identity", lambda base: (asked.append(base), ("X", ""))[1])
    assert "another model" in H.save(BASE, fake.d, "k", 28213, "planned-for-Y")[1]
    _store(fake.d, "k2", 28213)
    res = await agent._restore_file({"base": BASE, "dir": fake.d, "key": "k2", "identity": "planned-for-Y"})
    assert res[:2] == (None, A.RESTORE_OTHER_SERVER) and asked == [BASE, BASE]


# ════════ r3: a deadline's room goes to the save, not to a /props that hangs ════════

@pytest.mark.parametrize("deadline", [True, False])
async def test_a_save_with_a_deadline_does_not_read_props_first(real, deadline):
    """The shutdown save has 25 s and the save itself needs 20: the pre-read's
    5 s timeout on a hanging /props left it "5s left of the deadline" — not
    posted. With a deadline the save is posted at once; the identity is checked
    after llama answered, where it always was."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    fake.calls.clear()
    late = time.monotonic() + A.SHUTDOWN_CONV_SAVE_S if deadline else None
    assert await agent._conv_save(llm, "the agent's shutdown", late) is True
    before_save = fake.calls[:next(i for i, c in enumerate(fake.calls) if c.endswith("action=save"))]
    assert any(c.endswith("/props") for c in before_save) is (not deadline)
    assert any(c.endswith("/props") for c in fake.calls)             # after it: always


async def test_the_shutdown_save_is_posted_though_props_hangs_and_then_not_kept(real, monkeypatch):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    real_http = fake.http
    clock = {"now": time.monotonic()}
    monkeypatch.setattr(A.time, "monotonic", lambda: clock["now"])

    def http(url, body=None, timeout=5.0):
        if url.endswith("/props"):
            clock["now"] += timeout                                  # it hangs to its timeout
            raise TimeoutError("timed out")
        return real_http(url, body, timeout)
    monkeypatch.setattr(H, "_http", http)
    fake.saves.clear()
    st["lines"].clear()
    assert await agent.save_slot_conversation_at_shutdown() is False
    assert len(fake.saves) == 1                                      # posted: the 25 s were the save's
    assert not list(fake.d.glob("*.part.bin"))
    assert sum("not saved before the agent's shutdown" in m and "/props did not answer" in m
               for m in st["lines"]) == 1


# ════════ r3-F6: only "llama did not answer" stops a warm-up ════════

@pytest.mark.parametrize("exc,sends", [
    (TimeoutError("timed out"), False),
    (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")), False),
    (ConnectionResetError(54, "Connection reset by peer"), False),
    (_http_error(503, "Loading model"), False),
    (_http_error(400, '{"error":{"message":"template error"}}'), True),
    (PermissionError(13, "Permission denied"), True),
    (ValueError("no rendered prompt"), True),
])
async def test_only_an_unanswering_llama_stops_the_warm_up(real, exc, sends):
    """A PermissionError on the save directory or a 4xx from /apply-template
    said nothing about the slot, yet stopped every warm-up, the boot's too:
    the head was never warmed. Such a tick prefills without a plan, as before
    head saves had plans, and names the error once."""
    agent, llm, st, fake = real
    real_http = fake.http

    def http(url, body=None, timeout=5.0):
        if url.endswith("/apply-template"):
            raise exc
        return real_http(url, body, timeout)
    H._http = http
    await agent.warm_up_main_prefix()
    await agent.warm_up_main_prefix(quiet=True)
    assert st["sent"] == (2 if sends else 0) and agent._head_plan_last is None
    said = [m for m in st["lines"] if "no save plan (" in m]
    assert len(said) == 1 and type(exc).__name__ in said[0]
    assert ("the plan failed" in said[0]) is sends
    assert llm.before_main_dispatch == agent._slot_switch_before     # the hooks are there either way


# ════════ r2-F5: a restore that was never posted is not a restore that timed out ════════

@pytest.mark.parametrize("detail,want", [("TimeoutError: timed out", True), ("URLError: <urlopen error timed out>", True),
                                         ("socket.timeout", True), ("Unable to restore slot", False),
                                         (A.RESTORE_NO_PROPS, False), (A.RESTORE_OTHER_SERVER, False),
                                         ("too late to post", False), (None, False)])
def test_what_counts_as_a_posted_restore_that_timed_out(detail, want):
    assert H.restore_timed_out(detail) is want


async def test_props_down_on_the_second_try_is_not_a_second_restore_timeout(real, monkeypatch):
    """The conversation's restore timed out once; before the second try
    /props timed out too, so NOTHING was posted. Round 1 read the words
    "TimeoutError" in that reason as llama still loading the file: it claimed
    the slot for the conversation and armed the reuse check."""
    agent, llm, st, fake = real
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 200.0)
    await agent.warm_up_main_prefix(quiet=True)
    call = _turn(st)
    ckey = _stage_conv(fake, "sess-b", head=L.head_bind_tag(call))
    fake.slot_n = 33000
    real_http = fake.http
    state = {"restores": 0}

    def http(url, body=None, timeout=5.0):
        if url.endswith("action=restore"):
            state["restores"] += 1
            raise TimeoutError("timed out")
        if url.endswith("/props") and state["restores"] >= 1:
            raise TimeoutError("timed out")                          # the box is in swap
        return real_http(url, body, timeout)
    H._http = http
    st["lines"].clear()
    await _switch(agent, call, "sess-b", "req-b1")
    assert state["restores"] == 1                                    # one restore was posted
    assert llm._slot_session != "sess-b" and llm._restore_check is None
    assert not any("after two timeouts" in m or "restored from disk" in m for m in st["lines"])
    assert any("restore failed" in m and "the head instead" in m for m in st["lines"])
    assert (fake.d / f"head-{ckey}.bin").exists()                    # the file stays for the next request


async def test_a_second_posted_restore_that_times_out_is_still_a_landing(real, monkeypatch):
    """The control: llama WAS asked twice and is reading the file."""
    agent, llm, st, fake = real
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 200.0)
    await agent.warm_up_main_prefix(quiet=True)
    call = _turn(st)
    _stage_conv(fake, "sess-b", head=L.head_bind_tag(call))
    fake.slot_n = 33000
    real_http = fake.http
    state = {"restores": 0}

    def http(url, body=None, timeout=5.0):
        if url.endswith("action=restore"):
            state["restores"] += 1
            raise TimeoutError("timed out")
        return real_http(url, body, timeout)
    H._http = http
    st["lines"].clear()
    await _switch(agent, call, "sess-b", "req-b1")
    assert state["restores"] == 2 and llm._slot_session == "sess-b"
    assert any("after two timeouts" in m for m in st["lines"])


async def test_a_restore_llama_refused_is_posted_once_and_arms_nothing(real, monkeypatch):
    """Only a TIMED-OUT restore is asked for again (llama may be applying it).
    A refusal is final: the conversation's file is posted once, the head's
    once, and a head restore that failed leaves no expectation behind."""
    agent, llm, st, fake = real
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 200.0)
    await agent.warm_up_main_prefix(quiet=True)
    call = _turn(st)
    ckey = _stage_conv(fake, "sess-b", head=L.head_bind_tag(call))
    fake.slot_n = 33000
    real_http, posts = fake.http, []

    def http(url, body=None, timeout=5.0):
        if url.endswith("action=restore"):
            posts.append(body["filename"])
            raise _http_error(400, '{"error":{"message":"Unable to restore slot"}}')
        return real_http(url, body, timeout)
    H._http = http
    await _switch(agent, call, "sess-b", "req-b1")
    assert posts == [f"head-{ckey}.bin", f"head-{agent._head_plan_last['key']}.bin"]
    assert llm._restore_check is None and llm._slot_session != "sess-b"


async def test_a_warm_up_restore_that_was_never_posted_is_not_reported_as_restored(real):
    """Between the plan and the dispatch llama restarted on another model."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    fake.slot_n, fake.fresh = None, True
    real_http, seen = fake.http, {"props": 0}

    def http(url, body=None, timeout=5.0):
        if url.endswith("/props"):
            seen["props"] += 1
            if seen["props"] == 2:                                   # 1: the plan; 2: the restore's own read
                fake.swap_model()
        return real_http(url, body, timeout)
    H._http = http
    fake.restores.clear()
    st["lines"].clear()
    await agent.warm_up_main_prefix(quiet=True)
    assert fake.restores == []
    assert not any("restored the head" in m for m in st["lines"])
    assert any("head restore failed" in m and "another model" in m for m in st["lines"])


# ════════ H4: only a whole file is kept ════════

@pytest.mark.parametrize("written,size,why", [(0, 0, "empty"), (1000, 400, "short"), (0, 1000, "no count"),
                                              (1000, 1000, "")])
def test_an_empty_or_short_state_never_replaces_a_good_one(real, written, size, why):
    agent, llm, st, fake = real
    _store(fake.d, "k", 28213)
    (fake.d / "head-k.bin").write_bytes(b"the previous good save")
    fake.written, fake.size = written, size
    ok, detail = H.save(BASE, fake.d, "k", 28213)
    assert ok is (why == "")
    if why:
        assert "not kept" in detail
        assert (fake.d / "head-k.bin").read_bytes() == b"the previous good save"
        assert H.stored_tokens(fake.d, "k") == 28213 and not list(fake.d.glob("*.part.bin"))
    else:
        assert (fake.d / "head-k.bin").stat().st_size == 1000


def test_a_disk_that_cannot_hold_the_state_is_not_written_to(real, monkeypatch):
    agent, llm, st, fake = real
    need = H.state_bytes(28213) + H.FREE_MARGIN_BYTES
    monkeypatch.setattr(H, "free_bytes", lambda d: need - 1)
    ok, detail = H.save(BASE, fake.d, "k", 28213)
    assert not ok and "free" in detail and fake.saves == []
    monkeypatch.setattr(H, "free_bytes", lambda d: need)
    assert H.save(BASE, fake.d, "k", 28213)[0] and len(fake.saves) == 1
    monkeypatch.setattr(H, "free_bytes", lambda d: None)             # unknown: not a refusal
    assert H.save(BASE, fake.d, "k2", 28213)[0]


def test_free_bytes_reads_the_volume(tmp_path):
    assert H.free_bytes(tmp_path) > 0 and H.free_bytes(tmp_path / "missing") is None


# ════════ H5: why the wait ended ════════

async def test_the_slot_wait_says_why(real):
    agent, llm, st, fake = real
    assert await agent._slot_wait(BASE, time.monotonic() + 5) == ("", {"id": 0, "is_processing": False,
                                                                       "n_prompt_tokens": 28213})
    fake.processing = True
    assert (await agent._slot_wait(BASE, time.monotonic() - 1))[0] == "busy"
    assert (await agent._slot_wait(BASE, time.monotonic() + 30, lambda: True))[0] == "stopped"
    fake.has_processing = False
    assert (await agent._slot_wait(BASE, time.monotonic() + 30))[0] == "silent"
    fake.down = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    t0 = time.monotonic()
    assert (await agent._slot_wait(BASE, time.monotonic() + 30))[0] == "unreachable"
    assert time.monotonic() - t0 < 5
    fake.down = TimeoutError("timed out")                            # a long batch: busy, to the deadline
    assert (await agent._slot_wait(BASE, time.monotonic() + 0.6))[0] == "busy"
    assert set(A.SLOT_WAIT_WHY) == {"busy", "unreachable", "stopped", "silent"}


async def test_an_unreachable_llama_is_not_logged_as_a_busy_slot(real):
    """11-11 07:38: the shutdown save gave up 1 s into its budget, "stayed busy"."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    refused = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    real_http = fake.http

    def http(url, body=None, timeout=5.0):
        if url.endswith("/slots"):
            raise refused
        return real_http(url, body, timeout)
    H._http = http
    st["lines"].clear()
    assert await agent._conv_save(llm, "the agent's shutdown", time.monotonic() + 25.0) is False
    line = [m for m in st["lines"] if "not saved before the agent's shutdown" in m]
    assert len(line) == 1 and "did not answer /slots" in line[0] and "stayed busy" not in line[0]


async def test_a_request_that_started_is_not_logged_as_a_busy_slot(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.processing = True
    st["lines"].clear()
    assert await agent._conv_save(llm, "a background call", stop=lambda: True) is False
    assert any("a request started" in m for m in st["lines"]) and not any("stayed busy" in m for m in st["lines"])
    st["lines"].clear()
    assert await agent._conv_save(llm, "a background call", time.monotonic() - 1) is False
    assert any("the main slot stayed busy" in m for m in st["lines"])


# ════════ H6: a slot llama does not count yet ════════

@pytest.mark.parametrize("fresh", [True, False])
async def test_a_slot_without_a_count_is_restored_into_only_on_a_fresh_server(real, fresh):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    call = _turn(st)
    ckey = _stage_conv(fake, "sess-b", head=L.head_bind_tag(call))
    fake.slot_n, fake.fresh = None, fresh                            # llama restarted: no count until a completion
    fake.restores.clear()
    st["info"].clear()
    await _switch(agent, call, "sess-b", "req-b1")
    if fresh:
        assert fake.restores == [f"head-{ckey}.bin"]
        assert any("served nothing since it started" in m for m in st["info"])
    else:
        assert fake.restores == []
        assert any("did not say how many tokens the slot holds" in m for m in st["info"])


# ════════ H7: the "saves off" mark ════════

def test_a_slot_action_that_succeeds_lifts_the_mark(real):
    agent, llm, st, fake = real
    _store(fake.d, "k", 28213)
    fake.no_flag = True
    assert H.restore(BASE, fake.d, "k")[0] is False and not H.supported(BASE)
    fake.no_flag = False                                             # llama restarted WITH the flag
    assert H.restore(BASE, fake.d, "k")[0] is True and H.supported(BASE)
    H._unsupported.add(BASE)
    assert H.save(BASE, fake.d, "k2", 28213)[0] is True and H.supported(BASE)


def test_a_refused_action_keeps_the_mark(real):
    agent, llm, st, fake = real
    _store(fake.d, "k", 28213)
    fake.no_flag = True
    H.restore(BASE, fake.d, "k")
    assert H.save(BASE, fake.d, "k2", 28213)[0] is False and not H.supported(BASE)


async def test_the_recheck_runs_on_a_tick_that_a_live_request_skips(real):
    """It sat after the busy `continue`s: a working owner kept saves off for
    the whole session. The directory's prune runs on the same tick."""
    agent, llm, st, fake = real
    H._unsupported.add(BASE)
    llm.foreground_requests = 1                                      # a request is live on every tick
    _store(fake.d, "dead", 9000)
    (fake.d / "head-dead.json").write_text(json.dumps({"n_tokens": 9000, "kind": H.MEMBER}))
    old = time.time() - H.MAX_AGE_S[H.MEMBER] - 60
    for n in ("head-dead.bin", "head-dead.json"):
        os.utime(fake.d / n, (old, old))
    _kfile(fake.d, "current", H.OWNER, "A", 60)                      # the head in use: a tick never takes it
    ticks, warms = [], []

    async def wait(period, sleep, early=True):
        ticks.append(1)
        if len(ticks) > 2:
            raise RuntimeError("stop")
        return ""

    async def warm(**kw):
        warms.append(kw)
    agent._rewarm_wait, agent.warm_up_main_prefix = wait, warm
    with pytest.raises(RuntimeError):
        await agent.rewarm_main_prefix_loop(60)
    assert warms == []                                               # the ticks were skipped
    assert H.supported(BASE) and any("head saves back on" in m for m in st["lines"])
    assert not (fake.d / "head-dead.bin").exists()                   # H14: one prune per tick
    assert (fake.d / "head-current.bin").exists() and (fake.d / "head-current.json").exists()


async def test_a_recheck_answered_on_a_skipped_tick_still_warms_up_on_the_next(real):
    """The answer is carried: the tick that found saves back was skipped (a
    request was live), the next one warms up though llama's total never moved."""
    agent, llm, st, fake = real
    ticks, warms = [], []

    async def wait(period, sleep, early=True):
        ticks.append(1)
        if len(ticks) == 2:
            H._unsupported.add(BASE)
            llm.foreground_requests = 1
        elif len(ticks) == 3:
            llm.foreground_requests = 0
        elif len(ticks) > 4:
            raise RuntimeError("stop")
        return ""

    async def warm(**kw):
        warms.append(len(ticks))

    async def metrics():
        return 0, 100

    async def no_member():
        return False
    agent._rewarm_wait, agent.warm_up_main_prefix, agent._member_head_refresh = wait, warm, no_member
    with pytest.raises(RuntimeError):
        await agent.rewarm_main_prefix_loop(60, metrics=metrics)
    assert warms == [1, 3]                                           # tick 2 skipped, tick 4: nothing ran


async def test_a_request_with_no_plan_lifts_the_mark_itself(real):
    """Not only an idle tick: the request that needs the save asks."""
    agent, llm, st, fake = real
    fake.no_flag = True
    H._unsupported.add(BASE)
    await agent.warm_up_main_prefix()
    assert agent._head_plan_last is None
    assert any("no save plan" in m and "--slot-save-path" in m for m in st["lines"])
    fake.no_flag = False                                             # restarted with the flag
    await _switch(agent, _turn(st), "sess-b", "req-b1")
    assert H.supported(BASE) and isinstance(agent._head_plan_last, dict)


# ════════ H8 / H14: the directory ════════

def _kfile(d, key, kind, identity, age_s, size=5):
    (d / f"head-{key}.bin").write_bytes(b"x" * size)
    (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": kind, "identity": identity}))
    at = time.time() - age_s
    for n in (f"head-{key}.bin", f"head-{key}.json"):
        os.utime(d / n, (at, at))


def test_trying_another_model_once_keeps_the_usual_models_saves(tmp_path):
    """llama start on model B dropped every save of model A at once."""
    d = _dir(tmp_path)
    for i, key in enumerate(["a-medium", "a-high"]):
        _kfile(d, key, H.OWNER, "A", 3600 + i)
    _kfile(d, "a-members", H.MEMBER, "A", 3600)
    _kfile(d, "ca-conv", H.CONV, "A", 3600)
    H.prune(d, identity="B")                                         # llama starts on B
    assert len(list(d.glob("head-*.bin"))) == 4
    _kfile(d, "a-ancient", H.OWNER, "A", H.FOREIGN_KEEP_S + 60)
    _kfile(d, "b-recent-old", H.OWNER, "B", H.FOREIGN_KEEP_S + 60)   # the running server's: its own age cap
    H.prune(d, identity="B")
    assert not (d / "head-a-ancient.bin").exists() and (d / "head-b-recent-old.bin").exists()


def test_another_models_saves_rank_behind_the_running_servers(tmp_path):
    """Kept by age, but inside the same bounds: a foreign save never pushes
    out one of the running server's, however recently it was used."""
    d = _dir(tmp_path)
    for i, key in enumerate(["ours-1", "ours-2", "ours-3"]):
        _kfile(d, key, H.OWNER, "A", 5000 + i)
    _kfile(d, "theirs", H.OWNER, "B", 10)                            # the newest of all
    H.prune(d, identity="A")
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-ours-1.bin", "head-ours-2.bin",
                                                            "head-ours-3.bin"]
    _kfile(d, "theirs", H.OWNER, "B", 10)
    H.prune(d)                                                       # nobody said which runs: by use alone
    assert (d / "head-theirs.bin").exists() and not (d / "head-ours-3.bin").exists()


def test_a_meta_that_names_no_server_is_not_another_servers(tmp_path):
    """A save from before identities were stored: bounded by its kind's count
    and age like the running server's, not by the 3 days of a foreign one."""
    d = _dir(tmp_path)
    _kfile(d, "legacy", H.OWNER, "", H.FOREIGN_KEEP_S + 60)
    _kfile(d, "theirs", H.OWNER, "B", H.FOREIGN_KEEP_S + 60)
    H.prune(d, identity="A")
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-legacy.bin"]


async def test_a_fresh_server_is_not_parked_before_a_members_restore(real):
    """Its slot is empty: the park prompt would only make llama not fresh."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    fake.slot_n, fake.fresh = None, True
    fake.calls.clear()
    res = await agent._restore_when_idle(llm, {**hs}, time.monotonic() + 30, skip_n=-1, park=True)
    assert res is not None and res[0] is True
    assert not any(c.endswith("/completion") for c in fake.calls)
    fake.slot_n, fake.fresh = 33000, False                           # the control: a state worth parking
    fake.calls.clear()
    res = await agent._restore_when_idle(llm, {**hs, "key": hs["key"]}, time.monotonic() + 30, skip_n=-1, park=True)
    assert res is not None and any(c.endswith("/completion") for c in fake.calls)


def test_a_save_prunes_with_the_identity_it_just_confirmed(real):
    """The one place besides llama start that knows which server runs: another
    model's saves past their 3 days leave when the running one saves."""
    agent, llm, st, fake = real
    _kfile(fake.d, "theirs-old", H.OWNER, "B", H.FOREIGN_KEEP_S + 60)
    _kfile(fake.d, "theirs-recent", H.OWNER, "B", 60)
    assert H.save(BASE, fake.d, "k", 28213, H.server_identity(BASE))[0]
    assert sorted(p.name for p in fake.d.glob("head-*.bin")) == ["head-k.bin", "head-theirs-recent.bin"]


@pytest.mark.parametrize("kind", [H.OWNER, H.MEMBER, H.CONV])
def test_every_kind_has_an_age_cap(tmp_path, kind):
    d = _dir(tmp_path)
    cap = H.MAX_AGE_S[kind]
    _kfile(d, "cold", kind, "A", cap + 60)
    _kfile(d, "warm", kind, "A", cap - 60)
    H.prune(d)
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-warm.bin"]
    assert not (d / "head-cold.json").exists()


def test_a_restore_counts_as_a_use_for_the_age_cap(real):
    agent, llm, st, fake = real
    _kfile(fake.d, "k", H.CONV, "A", H.MAX_AGE_S[H.CONV] + 60)
    assert H.restore(BASE, fake.d, "k")[0] is True                   # touches the meta
    H.prune(fake.d)
    assert (fake.d / "head-k.bin").exists()


def test_a_members_head_for_a_prompt_that_is_gone_ages_out_while_the_owners_stays(tmp_path):
    d = _dir(tmp_path)
    age = H.MAX_AGE_S[H.MEMBER] + 60
    assert age < H.MAX_AGE_S[H.OWNER]
    _kfile(d, "dead-prompt", H.MEMBER, "A", age)
    _kfile(d, "owner", H.OWNER, "A", age)
    H.prune(d)
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-owner.bin"]


def _mode(p):
    return stat.S_IMODE(p.stat().st_mode)


def test_saved_states_are_the_owners_alone(real):
    """Conversation files hold the owner's tokens; llama writes them 0644."""
    agent, llm, st, fake = real
    os.chmod(fake.d, 0o755)
    _store(fake.d, "older")
    os.chmod(fake.d / "head-older.bin", 0o644)
    os.chmod(fake.d / "head-older.json", 0o644)
    assert H.save(BASE, fake.d, "k", 28213, H.server_identity(BASE))[0]
    assert _mode(fake.d / "head-k.bin") == 0o600 and _mode(fake.d / "head-k.json") == 0o600
    assert _mode(fake.d) == 0o700
    assert _mode(fake.d / "head-older.bin") == 0o600 and _mode(fake.d / "head-older.json") == 0o600


def test_a_save_makes_its_own_files_owner_only_before_any_prune(real, monkeypatch):
    """Not left to the prune that follows: the state is 0600 when it gets its name."""
    agent, llm, st, fake = real
    monkeypatch.setattr(H, "prune", lambda *a, **k: None)
    real_http = fake.http

    def http(url, body=None, timeout=5.0):
        r = real_http(url, body, timeout)
        if url.endswith("action=save"):
            os.chmod(fake.d / body["filename"], 0o644)               # as llama creates it
        return r
    monkeypatch.setattr(H, "_http", http)
    old = os.umask(0o022)
    try:
        assert H.save(BASE, fake.d, "k", 28213)[0]
    finally:
        os.umask(old)
    assert _mode(fake.d / "head-k.bin") == 0o600 and _mode(fake.d / "head-k.json") == 0o600


async def test_the_plan_makes_the_directory_owner_only(real):
    agent, llm, st, fake = real
    os.chmod(fake.d, 0o755)
    await agent._head_save_plan(llm, {"messages": [{"role": "system", "content": "x"}]})
    assert _mode(fake.d) == 0o700


# ════════ H10: one wait, one restore ════════

def _functions(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _calls_self(fn, name):
    return any(isinstance(n, ast.Attribute) and n.attr == name
               and isinstance(n.value, ast.Name) and n.value.id == "self" for n in ast.walk(fn))


def _src_files():
    root = pathlib.Path(A.__file__).resolve().parents[1]             # src/ghost_agent
    return sorted(p for p in root.rglob("*.py") if p.name != "head_saves.py")


#: llama's two slot actions as `head_saves` exposes them, and the module's
#: back doors to them
SINK_NAMES = {"restore", "restore_on_start", "save"}
BACK_DOORS = {"_http", "_main", "__dict__"}


def _slot_action_uses(tree):
    """Every way code outside `head_saves.py` can reach llama's slot save or
    restore, as (enclosing functions, what):
    - an attribute named restore / restore_on_start / save on the module or on
      a local alias of it, and `module._http` / `_main` / `__dict__`;
    - a bare name restore / restore_on_start (a from-import), a from-import of
      any sink or back door, and a star-import of the module;
    - the sink's name as a string handed to getattr;
    - a string that holds llama's own `action=restore` / `action=save`;
    - the module used as a value (bound to another name, passed on, `vars()`),
      and `sys.modules` (the module by its name)."""
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == "head_saves":
                    mods.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.endswith("head_saves"):
                    mods.add(a.asname or a.name.split(".")[0])
    found = []

    def walk(node, chain, parent):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chain = chain + (node.name,)
        what = None
        on_module = (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                     and node.value.id in mods)
        if isinstance(node, ast.Attribute) and node.attr in ("restore", "restore_on_start"):
            what = f".{node.attr}"                                   # on ANY object: an alias is caught
        elif on_module and node.attr in SINK_NAMES | BACK_DOORS:
            what = f".{node.attr}"
        elif isinstance(node, ast.Name) and node.id in ("restore", "restore_on_start"):
            what = f"name {node.id}"
        elif isinstance(node, ast.ImportFrom) and str(node.module or "").endswith("head_saves") \
                and any(a.name in SINK_NAMES | BACK_DOORS | {"*"} for a in node.names):
            what = "from-import"
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and any(isinstance(a, ast.Constant) and (
                  a.value in ("restore", "restore_on_start")         # on any object
                  or (a.value in SINK_NAMES | BACK_DOORS and isinstance(node.args[0], ast.Name)
                      and node.args[0].id in mods)) for a in node.args[1:2])):
            what = "getattr"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and ("action=restore" in node.value or "action=save" in node.value):
            what = "action string"
        elif isinstance(node, ast.Attribute) and node.attr == "modules" \
                and isinstance(node.value, ast.Name) and node.value.id == "sys":
            what = "sys.modules"
        elif (isinstance(node, ast.Name) and node.id in mods
              and not (isinstance(parent, ast.Attribute) and parent.value is node)):
            what = f"module {node.id} as a value"
        if what:
            found.append((chain, what))
        for child in ast.iter_child_nodes(node):
            walk(child, chain, node)
    walk(tree, (), None)
    return found


#: every legal use, by file and enclosing function. ONE restore. Saves have
#: four callers, each of a different state (a conversation, the members' head
#: made by a request, by the idle tick, the owner's head). `_http` is used for
#: three reads that are not slot actions (the slot's record, the window, the
#: members' prefill), each named here.
ALLOWED_USES = sorted([
    ("agent.py", ("_restore_file", "_restore_in_time"), ".restore"),
    ("agent.py", ("_conv_save_locked",), ".save"),
    ("agent.py", ("_member_head_make",), ".save"),
    ("agent.py", ("_member_head_refresh",), ".save"),
    ("agent.py", ("_head_save_after",), ".save"),
    ("agent.py", ("_slot_wait", "_read"), "._http"),
    ("agent.py", ("_member_head_make",), "._http"),
    ("agent.py", ("_main_window_tokens",), "._http"),
    ("coding_executor.py", ("_window_tokens",), "._http"),
])


def test_every_slot_action_in_the_package_is_one_of_the_named_sites():
    uses = []
    for p in _src_files():
        for chain, what in _slot_action_uses(ast.parse(p.read_text(encoding="utf-8"))):
            if p.name == "staleness.py" and what == "sys.modules":
                continue                                             # (its own module-reload watch: no head_saves)
            uses.append((p.name, chain, what))
    assert sorted(uses) == ALLOWED_USES


#: other spellings of a second restore site — none of them the plain
#: `_head_saves.restore(…)` the one legal site uses
OTHER_SPELLINGS = [
    "ok, detail, rdt = await asyncio.to_thread(getattr(_head_saves, 'restore'), 1, 2, 3)",
    "_r = _head_saves\n                ok, detail, rdt = await asyncio.to_thread(_r.restore, 1, 2, 3)",
    "from .head_saves import restore as _put_back\n                ok, detail, rdt = _put_back(1, 2, 3)",
    "from . import head_saves as _head_saves_again\n                ok, detail, rdt = _head_saves_again.restore(1, 2, 3)",
    "ok, detail, rdt = await asyncio.to_thread(_head_saves.restore_on_start, 1, 2), '', 0.0",
    "ok, detail, rdt = await asyncio.to_thread(vars(_head_saves)['rest' + 'ore'], 1, 2, 3)",
    # r3: the five that walked past the name walk
    "ok, detail, rdt = _head_saves._http(owner_hs['base'] + '/slots/0?action=restore', {'filename': 'x'}), '', 0.0",
    "_u = owner_hs['base'] + '/slots/0?action=restore'\n                ok, detail, rdt = await llm.post(_u), '', 0.0",
    "ok, detail, rdt = _head_saves._main(['--restore-on-start', '--dir', 'x']), '', 0.0",
    "ok, detail, rdt = _head_saves.__dict__['rest' + 'ore'](1, 2, 3)",
    "import sys\n                ok, detail, rdt = sys.modules['ghost_agent.core.head_saves'].__dict__['rest' + 'ore'](1, 2, 3)",
    "from .head_saves import *\n                ok, detail, rdt = globals()['rest' + 'ore'](1, 2, 3)",
    "ok, detail, rdt = _head_saves.save(1, 2, 3, 4), 0.0",          # a fifth save site
]


@pytest.mark.parametrize("spelling", OTHER_SPELLINGS)
def test_the_enumeration_fires_on_every_spelling_of_a_second_site(spelling):
    tree = ast.parse(pathlib.Path(A.__file__).read_text(encoding="utf-8").replace(
        "ok, detail, rdt = await self._restore_file(owner_hs)     # §4UD (H10): the one sink", spelling))
    uses = [("agent.py", chain, what) for chain, what in _slot_action_uses(tree)]
    extra = [u for u in uses if u not in ALLOWED_USES or uses.count(u) > ALLOWED_USES.count(u)]
    assert extra and all(u[1][-1] == "_member_head_refresh" for u in extra), spelling


def test_one_wait_for_the_slot_and_every_path_goes_through_the_two_sinks():
    tree = ast.parse(pathlib.Path(A.__file__).read_text(encoding="utf-8"))
    fns = _functions(tree)
    assert "_slot_idle" not in fns                                   # the bool wait is gone, not wrapped
    reads_slots = sorted(
        name for name, fn in fns.items()
        if any(isinstance(n, ast.Constant) and n.value == "/slots" for n in ast.walk(fn)))
    assert reads_slots == ["_read", "_slot_wait"]                    # the one poll loop (and its reader)
    for name in ("_restore_when_idle", "_outside_slot_restore"):
        assert _calls_self(fns[name], "_slot_wait") and _calls_self(fns[name], "_restore_file"), name
    for name in ("_conv_restore", "_member_head_before", "_slot_switch_before"):
        assert _calls_self(fns[name], "_restore_when_idle"), name
    for name in ("_conv_save_locked", "_member_head_make"):
        assert _calls_self(fns[name], "_slot_wait"), name


async def test_the_owners_head_comes_back_after_the_members_head_is_made(real, monkeypatch):
    """Behaviour, not spelling: `_member_head_refresh` prefills the members'
    head in a gap, saves it, and puts the owner's head back through the sink."""
    agent, llm, st, fake = real
    monkeypatch.delenv(L.CONV_KEEP_ENV)
    await agent.warm_up_main_prefix(quiet=True)
    owner_key = agent._head_plan_last["key"]
    agent._member_head_wanted = True
    agent._idle_s = lambda: 9999.0
    st["head_n"] = 9000                                              # the members' head is smaller
    fake.restores.clear()
    assert await agent._member_head_refresh() is True
    member = [p for p in fake.d.glob("head-*.json") if json.loads(p.read_text()).get("kind") == H.MEMBER]
    assert len(member) == 1 and json.loads(member[0].read_text())["n_tokens"] == 9000
    assert fake.restores == [f"head-{owner_key}.bin"] and fake.slot_n == 28213
    assert llm._head_displaced is False


async def test_the_outside_restore_gives_up_on_a_llama_that_refuses(real):
    """Its own copy of the wait had no unread limit: polled to the deadline."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    llm._outside_slot_dirty = True
    fake.down = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    t0 = time.monotonic()
    await agent._outside_slot_restore(llm, hs, _turn(st), time.monotonic() + 60)
    assert time.monotonic() - t0 < 5
    assert any("outside call" in m and "did not answer /slots" in m for m in st["info"])


async def test_the_one_restore_reads_the_identity_and_the_files_own(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    assert (await agent._restore_file(hs))[0] is True
    # a file whose meta names another server, under this plan's key (H2's mislabel)
    meta_p = fake.d / f"head-{hs['key']}.json"
    meta = json.loads(meta_p.read_text())
    meta_p.write_text(json.dumps({**meta, "identity": "model-B|b|t"}))
    fake.restores.clear()
    assert (await agent._restore_file(hs))[:2] == (None, A.RESTORE_OTHER_SERVER) and fake.restores == []
    meta_p.write_text(json.dumps(meta))
    assert (await agent._restore_file(hs, time.monotonic() - 1))[:2] == (False, "too late to post")
    fake.down = TimeoutError("timed out")
    assert (await agent._restore_file(hs))[:2] == (None, A.RESTORE_NO_PROPS)   # /props unread: nothing posted
    assert not H.restore_timed_out(A.RESTORE_NO_PROPS)               # … and never read as a restore's timeout


async def test_the_warm_ups_restore_goes_through_the_same_sink(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    fake.slot_n, fake.fresh = None, True                             # llama restarted: the head comes from disk
    fake.restores.clear()
    st["during_call"] = None
    seen = []
    real_sink = agent._restore_file

    async def sink(hs, late=None):
        seen.append(hs["key"])
        return await real_sink(hs, late)
    agent._restore_file = sink
    await agent.warm_up_main_prefix(quiet=True)
    assert seen == [agent._head_plan_last["key"]] and fake.restores == [f"head-{seen[0]}.bin"]


# ════════ r3-F9: the tag of the warm-up IS the tag of a real owner request ════════

async def test_a_real_owner_request_opens_with_what_the_warm_up_warmed(monkeypatch):
    """The bound rule compares tags of real payloads. This one comes from
    `handle_chat` itself (the function that builds an owner web request's
    system slot and tool list), next to the warm-up's own payload for the
    same agent: equal tags, on the first request and on its follow-up — so a
    conversation saved by one request is restored for the next."""
    from tests.test_kv_pin_stable_prefix import _make_agent
    for k in (L.CONV_KEEP_ENV, H.ENV, A.STATE_ANCHOR_ENV):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    agent = _make_agent()
    warm, _base, _sys = await agent._head_payload(agent._RequestState(agent))
    tags = []
    q1 = "Write a python script that parses my csv file and prints totals"
    for msgs in ([{"role": "user", "content": q1}],
                 [{"role": "user", "content": q1}, {"role": "assistant", "content": "Done: totals.py"},
                  {"role": "user", "content": "what is my dog's name?"}]):
        await agent.handle_chat({"messages": msgs, "model": "m", "stream": False}, MagicMock())
        sent = agent.context.llm_client.chat_completion.await_args.args[0]
        assert sent["messages"][0]["role"] == "system" and len(sent["messages"]) > 1
        tags.append(L.head_bind_tag(sent))
    assert tags[0] and tags == [L.head_bind_tag(warm)] * 2
    assert L.opens_with_head(sent, L.head_shape(warm))


# ════════ H11: every exit says why ════════

async def test_the_switch_says_why_it_restores_nothing(real, monkeypatch):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    call = _turn(st)
    fake.slot_n = 33000

    def said(text):
        return sum(text in m for m in st["info"])
    await _switch(agent, call, "", "probe-1")                        # no conversation key
    await _switch(agent, call, "", "probe-1")                        # … its second call: not said twice
    assert said("no conversation key") == 1
    agent._turn_lock_rid = "another-turn"
    await _switch(agent, call, "sess-b", "req-b1")
    assert said("another turn holds the lock") == 1
    agent._turn_lock_rid = None
    H.discard(fake.d, agent._head_plan_last["key"])                  # no head file at all
    await _switch(agent, call, "sess-c", "req-c1")
    assert said("no head file for this server yet") == 1
    agent._head_plan_last = None
    agent._head_plan_asked_rid = "req-d1"                            # (this request had its one attempt)
    await _switch(agent, call, "sess-d", "req-d1")
    assert said("no save plan") == 1
    assert fake.restores == []


async def test_a_slot_that_holds_the_file_already_says_so(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    fake.slot_n = 28213                                              # the bare head
    await _switch(agent, _turn(st), "sess-b", "req-b1")
    assert fake.restores == [] and any("already holds exactly its 28213 tokens" in m for m in st["info"])


async def test_a_save_that_is_not_made_says_why(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    assert await agent._conv_save(llm, "a test") is True
    st["info"].clear()
    assert await agent._conv_save(llm, "a second displacement") is True       # once per state
    assert any("already on disk before a second displacement" in m for m in st["info"])
    llm._slot_moved_on = True                                        # a background call ran since
    llm._conv_saved = None
    assert await agent._conv_save(llm, "a third") is False
    assert any("not saved before a third" in m and "last call was not this conversation's" in m
               for m in st["info"])


# ════════ H13: a save that lost the race ════════

async def test_a_conversation_save_that_lost_the_race_is_discarded(real):
    """A call reached the slot between the last check and the save: the file
    may be that call's state. It was kept and reported as saved."""
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    ckey = H.conv_key(H.server_identity(BASE), "sess-a")

    def another_call_lands():
        llm._slot_moved_on = True
        llm._last_main_call_end = time.monotonic() + 1
    fake.on_save = another_call_lands
    st["lines"].clear()
    assert await agent._conv_save(llm, "a background call") is False
    assert H.stored_tokens(fake.d, ckey) is None and not (fake.d / f"head-{ckey}.bin").exists()
    assert llm._conv_saved is None
    assert any("another call reached the slot during the save" in m for m in st["lines"])


async def test_a_conversation_save_that_kept_the_slot_is_kept(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    _owner_call(llm, st)
    fake.slot_n = 40050
    assert await agent._conv_save(llm, "a background call") is True
    ckey = H.conv_key(H.server_identity(BASE), "sess-a")
    assert H.stored_tokens(fake.d, ckey) == 40050 and llm._conv_saved[0] == "sess-a"
    assert llm._conv_last_saved[3] == ckey


# ════════ a cancelled request passes through ════════

async def test_a_request_cancelled_in_the_hook_leaves_the_lock_and_the_flags(real):
    agent, llm, st, fake = real
    await agent.warm_up_main_prefix(quiet=True)
    call = _turn(st)
    _stage_conv(fake, "sess-b", head=L.head_bind_tag(call))
    fake.slot_n, fake.processing = 33000, True                       # the hook waits for the slot

    async def run():
        await _switch(agent, call, "sess-b", "req-b1")
    task = asyncio.ensure_future(run())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not agent._conv_lock.locked()
    assert llm._slot_session is None and llm._restore_check is None and llm._slot_moved_on is False
    fake.processing = False                                          # the next request is served normally
    await _switch(agent, call, "sess-b", "req-b2")
    assert llm._slot_session == "sess-b"
