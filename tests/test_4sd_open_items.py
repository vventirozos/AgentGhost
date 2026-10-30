"""§4SD open items (operator 10-30: "fix all the remaining open items").

1. A background spec call (the executor's planning call) held the one main
   slot for minutes while the owner waited: it now yields as the file
   streams do, and is called again after the owner.
2. A foreground (owner-started) build's calls displaced the owner's
   conversation without saving it: they take the §4SC side path.
3. A web/CLI loop rewriting one file with different text every time earned
   deadline extensions forever: past `MAX_COUNTED_REWRITES` whole rewrites of
   a file, a rewrite is not progress.
4. Slack's 30 minutes were a target (the running turn finished): a client
   that sent a finite timeout has a hard limit — the turn is stopped.

Each test names the world it fails in.
"""
import ast
import asyncio
import json
import math
import pathlib
import time
from types import SimpleNamespace

import pytest

import ghost_agent.core.coding_executor as CE
from ghost_agent.core import agent as A
from ghost_agent.core import llm as LLM
from ghost_agent.core.turns import TurnRegistry
from ghost_agent.utils import logging as L

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = "4sdopen1"
SPEC = json.dumps({"files": [{"path": "a.py", "content": "print(1)\n"}], "verify": "", "summary": "s", "ledger": ""})


def _frame(delta):
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


class _SpecStream:
    """Streams a spec; on the calls in `owner_at` the owner's request starts
    after the first chunk."""
    def __init__(self, owner_at=(), text=SPEC, think=""):
        self.owner_at, self.text, self.think = set(owner_at), text, think
        self.calls, self.kw, self.payloads = 0, [], []
        self.foreground_requests = 0
        self.foreground_tasks = 0

    async def stream_chat_completion(self, payload, is_background=False, **kw):
        self.calls += 1
        self.kw.append(kw)
        self.payloads.append(payload)
        call = self.calls
        if self.think:
            yield _frame({"reasoning_content": self.think})
        yield _frame({"content": self.text[:5]})
        if call in self.owner_at:
            self.foreground_requests += 1
        yield _frame({"content": self.text[5:]})
        yield b"data: [DONE]\n\n"


# ── 1. a background spec call yields the slot ─────────────────────────────

async def test_a_background_spec_stream_gives_the_slot_to_the_owner_and_runs_again():
    # world: the spec stream had no yield — the owner waited behind minutes of planning
    llm = _SpecStream(owner_at={1})
    content, reasoning, aborted = await CE._spec_stream(llm, {"messages": []}, True)
    assert llm.calls == 2 and aborted is None and content == SPEC


async def test_a_yielded_spec_call_is_not_a_think_abort():
    # world: a yield counted as an aborted think phase — the policy turned thinking off
    llm = _SpecStream(owner_at={1}, think="plan: one file")
    CE._think_state(llm)["aborts"] = 0
    spec, _ = await CE._generate_build_spec(llm, "m", "build a.py", "", is_background=True)
    assert spec.get("files") and CE._think_state(llm)["aborts"] == 0 and llm.calls == 2


async def test_the_spec_yield_baseline_is_the_first_chunk():
    # world: read at the call's start, an owner request already live at admission yielded at once, forever
    llm = _SpecStream()
    llm.foreground_requests = 3
    content, _, aborted = await CE._stream_spec_completion(llm, {}, True)
    assert aborted is None and content == SPEC and llm.calls == 1


async def test_a_waiting_foreground_call_also_makes_the_spec_yield():
    class _FgWaits(_SpecStream):
        async def stream_chat_completion(self, payload, is_background=False, **kw):
            self.calls += 1
            yield _frame({"content": "{"})
            if self.calls == 1:
                self.foreground_tasks = 1
            else:
                self.foreground_tasks = 0
            yield _frame({"content": SPEC[1:]})
            yield b"data: [DONE]\n\n"
    llm = _FgWaits()
    content, _, aborted = await CE._spec_stream(llm, {}, True)
    assert llm.calls == 2 and content == SPEC


async def test_spec_yields_are_bounded():
    # world: an owner who keeps arriving starves the build — after _MAX_YIELDS it runs on
    llm = _SpecStream(owner_at=set(range(1, 100)))
    content, _, aborted = await CE._spec_stream(llm, {}, True)
    assert llm.calls == CE._MAX_YIELDS + 1 and aborted is None and content == SPEC


async def test_a_foreground_spec_stream_never_yields():
    llm = _SpecStream(owner_at={1})
    content, _, aborted = await CE._spec_stream(llm, {}, False)
    assert llm.calls == 1 and aborted is None and content == SPEC


async def test_the_background_no_think_spec_call_streams_and_yields():
    # world: the no-think call was one non-streamed call — it could not yield at all
    llm = _SpecStream(owner_at={1})
    content, _ = await CE._spec_nothink_call(llm, "m", "sys", "user", True)
    assert llm.calls == 2 and content == SPEC
    assert all("stream" not in p for p in llm.payloads)
    assert llm.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_an_empty_answer_under_the_grammar_is_retried_without_it(monkeypatch):
    # world: streamed, a rejected grammar is an error frame (no raise) — the retry never ran
    monkeypatch.setattr(CE, "_spec_response_format", lambda: {"type": "json_object"})

    class _GrammarRejected(_SpecStream):
        async def stream_chat_completion(self, payload, is_background=False, **kw):
            self.calls += 1
            self.payloads.append(payload)
            if "response_format" in payload:
                yield f"data: {json.dumps({'error': 'grammar'})}\n\n".encode()
                return
            yield _frame({"content": SPEC})
            yield b"data: [DONE]\n\n"
    llm = _GrammarRejected()
    content, _ = await CE._spec_nothink_call(llm, "m", "sys", "user", True)
    assert content == SPEC and llm.calls == 2 and "response_format" not in llm.payloads[1]


# ── 2. a foreground build's calls take the side path ──────────────────────

async def test_foreground_executor_streams_ask_for_the_side_path():
    # world: an owner's build displaced the owner's conversation unsaved (the 28k head re-read)
    llm = _SpecStream()
    await CE._stream_spec_completion(llm, {}, False)
    assert llm.kw[0] == {"side_main": True, "task_label": "coding spec"}
    await CE._stream_file_text(llm, {}, False)
    assert llm.kw[1] == {"side_main": True, "task_label": "generated file"}


async def test_background_executor_streams_do_not():
    llm = _SpecStream()
    await CE._stream_spec_completion(llm, {}, True)
    await CE._stream_file_text(llm, {}, True)
    assert llm.kw == [{}, {}]


async def test_the_foreground_no_think_call_asks_for_the_side_path():
    seen = []

    class _NonStream:
        async def chat_completion(self, payload, is_background=False, **kw):
            seen.append(kw)
            return {"choices": [{"message": {"content": SPEC}}]}
    content, _ = await CE._spec_nothink_call(_NonStream(), "m", "s", "u", False)
    assert content == SPEC and seen == [{"side_main": True, "task_label": "coding spec"}]


class _Client:
    """The real `stream_chat_completion` over a fake upstream."""
    def __init__(self, fail=False):
        self.events, self.fail = [], fail
        self.coding_clients = None
        self._foreground_lock = asyncio.Lock()
        self.foreground_tasks = 0
        self._main_streams_open = 0

    async def _before_side_main_dispatch(self, payload):
        self.events.append(("side", LLM.side_call_what()))

    async def _before_main_dispatch(self, payload):
        self.events.append(("main", ""))

    async def _do_stream_chat_completion(self, payload, use_coding=False):
        if self.fail:
            raise RuntimeError("prefill cut")
        yield _frame({"content": "x"})


async def _drain(client, **kw):
    out = []
    async for c in LLM.LLMClient.stream_chat_completion(client, {"messages": []}, **kw):
        out.append(c)
    return out


async def test_a_side_stream_runs_the_side_hook_named_by_its_label(monkeypatch):
    marks = []
    monkeypatch.setattr(LLM, "note_main_slot_prompt", lambda *a, **k: marks.append((a[2], k)))
    c = _Client()
    await _drain(c, side_main=True, task_label="coding spec")
    assert c.events == [("side", "this request's coding spec call")] and len(marks) == 1
    c2 = _Client()
    await _drain(c2)
    assert c2.events == [("main", "")]                       # the head's own streams are unchanged


async def test_a_side_stream_that_failed_mid_prefill_is_counted_as_displaced(monkeypatch):
    # world: a cut side stream left its tokens in the slot unmarked — the next head call did not restore
    marks = []
    monkeypatch.setattr(LLM, "note_main_slot_prompt", lambda *a, **k: marks.append(a[2]))
    with pytest.raises(RuntimeError):
        await _drain(_Client(fail=True), side_main=True, task_label="generated file")
    assert marks == ["stream"]
    marks.clear()
    with pytest.raises(RuntimeError):
        await _drain(_Client(fail=True))                    # not a side stream: unchanged (no mark)
    assert marks == []


def test_the_project_advancers_foreground_calls_take_the_side_path():
    # world: the classify / command / self-analysis calls inside the owner's request displaced it unsaved
    src = (ROOT / "src/ghost_agent/core/project_advancer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "chat_completion"]
    assert len(calls) == 3
    for c in calls:
        kws = {k.arg: k.value for k in c.keywords}
        assert isinstance(kws.get("side_main"), ast.Constant) and kws["side_main"].value is True
        assert isinstance(kws.get("task_label"), ast.Constant) and kws["task_label"].value


# ── 3. rewrites of one file stop counting ─────────────────────────────────

@pytest.fixture
def req(monkeypatch):
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "3600")
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "inf")
    tok = L.client_deadline_context.set(math.inf)
    with L._REQ_STATE_LOCK:
        L._REQ_STATE[RID] = {"started": time.monotonic() - 100, "origin": None}
    yield
    L.client_deadline_context.reset(tok)
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID, None)


def test_whole_rewrites_of_one_file_count_only_up_to_the_cap(req):
    # world: a model rewriting app.js with new text each time earned extensions forever
    n = L.MAX_COUNTED_REWRITES
    got = [L.note_code_progress(RID, "w", key=f"k{i}", rewrite_of="app.js") for i in range(n + 2)]
    assert got == [True] * n + [False, False]
    assert L.note_code_progress(RID, "w", key="other", rewrite_of="lib.js") is True     # per file
    assert L.note_code_progress(RID, "w", key="edit1") is True                          # edits still count


def test_a_capped_rewrite_does_not_book_its_key():
    # world: a refused rewrite's key was booked — nothing; but a repeat must still be refused as a repeat
    with L._REQ_STATE_LOCK:
        L._REQ_STATE[RID] = {"started": time.monotonic(), "origin": None}
    try:
        assert L.note_code_progress(RID, key="same") is True
        assert L.note_code_progress(RID, key="same", rewrite_of="x.js") is False        # repeat: not a rewrite count
        assert L._REQ_STATE[RID].get("rewrites", {}).get("x.js", 0) == 0
    finally:
        with L._REQ_STATE_LOCK:
            L._REQ_STATE.pop(RID, None)


def test_spellings_of_one_file_share_the_rewrite_cap(req):
    # world (review r1): game.py / ./game.py / /workspace/game.py alternated past the cap
    for i, t in enumerate(["game.py", "./game.py", "/workspace/game.py", "Game.py", "workspace/game.py"]):
        _earn_hook("write", {}, t, f'file_system:{{"content": "v{i}"}}')
    L._REQ_STATE[RID]["progress_t"] = t0 = time.monotonic() - 900
    _earn_hook("write", {}, "./game.py", 'file_system:{"content": "v9"}')
    assert L._REQ_STATE[RID]["progress_t"] == t0


@pytest.mark.parametrize("op,args,want", [
    (" write ", {}, True),
    ("write", {}, True), ("write", {"append": False}, True), ("WRITE", {}, True),
    ("write", {"append": True}, False), ("write", {"append": "true"}, False),
    ("append", {}, False), ("edit", {}, False), ("replace", {}, False), ("write", None, True),
])
def test_a_whole_rewrite_is_a_write_without_append(op, args, want):
    # world: the next part of a file written in parts (write append=true) counted as a rewrite
    assert A.is_whole_rewrite(op, args) is want


def _earn_hook(op, args, target, a_hash):
    from tests.test_4ra_deadline_extension import _Ctx, _hook_code
    self = _Ctx()
    self.context = _Ctx()
    self.context._script_iter = {}
    outcome = _Ctx()
    outcome.is_rejection = False
    g = {**vars(A), "self": self, "fname": "file_system", "ptool_op": op, "_recorded_args": args,
         "is_mutating": True, "ptarget": target, "a_hash": a_hash, "_res_is_error": False,
         "_res_unresolved": False, "_outcome": outcome}
    tok = L.request_id_context.set(RID)
    try:
        exec(_hook_code(), g)
    finally:
        L.request_id_context.reset(tok)


def test_the_turn_loop_stops_counting_the_sixth_rewrite(req):
    # world: the loop never passed `rewrite_of` — the cap was dead
    for i in range(L.MAX_COUNTED_REWRITES):
        _earn_hook("write", {}, "app.js", f'file_system:{{"content": "v{i}", "path": "app.js"}}')
    L._REQ_STATE[RID]["progress_t"] = t0 = time.monotonic() - 900
    _earn_hook("write", {}, "app.js", 'file_system:{"content": "v99", "path": "app.js"}')
    assert L._REQ_STATE[RID]["progress_t"] == t0
    _earn_hook("write", {"append": True}, "app.js", 'file_system:{"content": "more", "append": true}')
    assert L._REQ_STATE[RID]["progress_t"] > t0                       # the next part still counts


def test_a_capped_rewrite_rearms_no_run():
    # world: rewrite + run of a .py file earned through the run's verdict instead
    import ast as _ast
    tree = _ast.parse((ROOT / "src/ghost_agent/core/agent.py").read_text(encoding="utf-8"))
    fn = next(n for n in _ast.walk(tree) if isinstance(n, _ast.AsyncFunctionDef)
              and n.name == "_dispatch_and_process_tool_batch")
    blk = next(n for n in _ast.walk(fn) if isinstance(n, _ast.If) and isinstance(n.test, _ast.BoolOp)
               and any(isinstance(v, _ast.Compare) and isinstance(v.left, _ast.Name) and v.left.id == "fname"
                       and any(isinstance(c, _ast.Constant) and c.value == "file_system" for c in v.comparators)
                       for v in n.test.values)
               and any(isinstance(x, _ast.Name) and x.id == "_bn" for x in _ast.walk(n)))
    code = compile(_ast.fix_missing_locations(_ast.Module(body=blk.body, type_ignores=[])), "<fut>", "exec")
    si = {}
    for i in range(L.MAX_COUNTED_REWRITES + 3):
        g = {**vars(A), "_si": si, "fname": "file_system", "ptool_op": "write", "_recorded_args": {},
             "is_mutating": True, "ptarget": "main.py", "_res_is_error": False,
             "a_hash": f"file_system:v{i}", "_CODE_EXTS": (".py",)}
        exec(code, g)
    assert si["main.py"]["new_writes"] == L.MAX_COUNTED_REWRITES
    assert si["main.py"]["writes"] == L.MAX_COUNTED_REWRITES + 3          # the churn count sees all
    g["ptool_op"], g["_recorded_args"], g["a_hash"] = "edit", {}, "file_system:edit1"
    exec(code, g)
    assert si["main.py"]["new_writes"] == L.MAX_COUNTED_REWRITES + 1      # an edit still re-arms


# ── 4. a finite client timeout is a hard limit ────────────────────────────

@pytest.mark.parametrize("d,want", [(1800, True), (1800.0, True), (0, False), (None, False),
                                    (math.inf, False), (-5, False), ("junk", False), ("60", True)])
def test_which_clients_get_a_hard_limit(d, want):
    # world: web/CLI (inf) stopped at a number; probes (no header) stopped at the server cap
    assert A.hard_limit_applies(d) is want


@pytest.fixture
def slack_req(monkeypatch):
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "1800")
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "inf")
    tok = L.client_deadline_context.set(1800.0)

    def _open(elapsed):
        with L._REQ_STATE_LOCK:
            L._REQ_STATE[RID] = {"started": time.monotonic() - elapsed, "origin": None}
    yield _open
    L.client_deadline_context.reset(tok)
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID, None)


async def test_the_watch_stops_the_turn_when_the_limit_passes(slack_req):
    # world: the 30 minutes were a target — a 16k turn begun in the reserve ran ~13 min over
    slack_req(1800 - 0.2)
    reg = TurnRegistry()
    turn = reg.register(RID, session_id="slack:t:u", kind="owner")
    turn.running = True
    turn.inbox.append({"text": "and also this", "ts": time.time()})
    from ghost_agent.utils.tool_stop import ToolStop
    stop = ToolStop()
    stop._running[0] = "execute"                      # a 10-min execute is running
    turn.tool_stop = stop
    assert await A.watch_hard_limit(RID, reg, poll_s=0.05) is True
    assert turn.cancelled and turn.reason.startswith(A.HARD_LIMIT_REASON) and "30 min" in turn.reason
    # world (review r1 MAJOR): cancel first — interrupt_tools refused the cancelled turn, the tool ran on
    assert stop.requested()
    assert reg.take_leftovers("slack:t:u", kind="owner") == ["and also this"]     # not lost


async def test_the_watch_waits_while_time_remains(slack_req):
    slack_req(100)
    reg = TurnRegistry()
    reg.register(RID).running = True
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.05))
    await asyncio.sleep(0.2)
    assert not t.done() and not reg.is_cancelled(RID)
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID, None)                          # the request closed
    assert await asyncio.wait_for(t, 2) is False and not reg.is_cancelled(RID)


async def test_the_watch_reads_the_deadline_in_force_each_time(slack_req, monkeypatch):
    # world: the deadline read once — an extension earned meanwhile was cut anyway
    slack_req(100)
    reg = TurnRegistry()
    reg.register(RID).running = True
    left = iter([5.0, 0.01, -1.0])
    monkeypatch.setattr(L, "request_remaining_s", lambda rid: next(left))
    assert await A.watch_hard_limit(RID, reg, poll_s=0.01) is True


async def test_a_turn_already_stopped_is_not_stopped_again(slack_req):
    slack_req(1900)
    reg = TurnRegistry()
    t = reg.register(RID)
    t.running = True
    reg.cancel(RID, reason="cancelled by user")
    assert await A.watch_hard_limit(RID, reg, poll_s=0.01) is False
    assert t.reason == "cancelled by user"


def test_the_stopped_turns_reply_names_the_files():
    # world: a Slack owner stopped at 30 min got "No output was produced" and nothing else
    rows = [{"name": "file_system", "role": "tool", "content": "SUCCESS: Wrote 12 chars to 'app.js'. "}]
    assert A.hard_limit_files_line(rows) == "Files written so far: `app.js`."
    assert A.hard_limit_files_line([]) == "" and A.hard_limit_files_line(None) == ""


def _handle_chat_src():
    tree = ast.parse((ROOT / "src/ghost_agent/core/agent.py").read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")


def test_handle_chat_starts_the_watch_for_a_finite_client_and_ends_it_in_the_finally():
    # world: the watch was written but never started (or outlived its turn)
    fn = _handle_chat_src()
    src = ast.unparse(fn)
    assert "if hard_limit_applies(_cdl):" in src
    assert "watch_hard_limit(str(req_id), _turn_reg, client_s=_cdl)" in src
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody
             and any(isinstance(h.type, ast.Name) and h.type.id == "TurnCancelled" for h in n.handlers)]
    assert tries
    fin = ast.unparse(ast.Module(body=tries[0].finalbody, type_ignores=[]))
    assert "_hw.cancel()" in fin
    handler = next(h for h in tries[0].handlers if isinstance(h.type, ast.Name) and h.type.id == "TurnCancelled")
    hsrc = ast.unparse(handler)
    assert "HARD_LIMIT_REASON" in hsrc and "hard_limit_files_line" in hsrc


async def test_the_watch_never_stops_before_the_clients_own_timeout(monkeypatch):
    # world (review r1): a 4 h non-streamed client under the server's 1 h cap was stopped at 1 h
    monkeypatch.setattr(L, "request_remaining_s", lambda rid: -10.0)
    monkeypatch.setattr(L, "request_elapsed_s", lambda rid: 3600.0)
    reg = TurnRegistry()
    reg.register(RID).running = True
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02, client_s=14400.0))
    await asyncio.sleep(0.15)
    assert not t.done() and not reg.is_cancelled(RID)
    monkeypatch.setattr(L, "request_elapsed_s", lambda rid: 14400.5)
    assert await asyncio.wait_for(t, 2) is True


async def test_a_finished_spec_is_kept_when_the_owner_arrives_in_its_trailing_frames():
    # world (review r1): the owner check ran on the usage/[DONE] frames — a whole spec regenerated
    class _Late(_SpecStream):
        async def stream_chat_completion(self, payload, is_background=False, **kw):
            self.calls += 1
            yield _frame({"content": SPEC})
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
            self.foreground_requests += 1
            yield f"data: {json.dumps({'usage': {'completion_tokens': 5}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
    llm = _Late()
    content, _, aborted = await CE._spec_stream(llm, {}, True)
    assert llm.calls == 1 and content == SPEC and aborted is None


@pytest.mark.parametrize("bg", [True, False])
async def test_a_deferral_is_not_retried_as_a_grammar_failure(monkeypatch, bg):
    # world (review r1): BackgroundDeferred was caught as a grammar error — parked twice, misleading warning
    monkeypatch.setattr(CE, "_spec_response_format", lambda: {"type": "json_object"})
    calls = []

    class _Defers:
        async def chat_completion(self, payload, is_background=False, **kw):
            calls.append(payload)
            raise LLM.BackgroundDeferred("owner busy")
    with pytest.raises(LLM.BackgroundDeferred):
        await CE._spec_nothink_call(_Defers(), "m", "s", "u", bg)
    assert len(calls) == 1


async def test_the_reply_names_the_limit_that_fired(monkeypatch):
    # world (review r2): a 4 h client stopped at 4 h was told "(30 min)" — the server cap
    monkeypatch.setattr(L, "request_remaining_s", lambda rid: -10.0)
    monkeypatch.setattr(L, "request_elapsed_s", lambda rid: 14401.0)
    monkeypatch.setattr(L, "request_deadline_s", lambda rid: 1800.0)
    reg = TurnRegistry()
    t = reg.register(RID)
    t.running = True
    assert await A.watch_hard_limit(RID, reg, poll_s=0.01, client_s=14400.0) is True
    assert "(240 min)" in t.reason


async def test_a_finish_in_its_own_frame_keeps_the_spec():
    # world (review r2): the owner check ran before the frame was read — the finish frame itself yielded
    class _SepFinish(_SpecStream):
        async def stream_chat_completion(self, payload, is_background=False, **kw):
            self.calls += 1
            yield _frame({"content": SPEC})
            self.foreground_requests += 1
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"
    llm = _SepFinish()
    content, _, aborted = await CE._spec_stream(llm, {}, True)
    assert llm.calls == 1 and content == SPEC


# ── r3: a stopped build writes the files it finished ──────────────────────

from tests.test_4sd_executor_generate import ScriptLLM, FakeRunner, _spec, _ctx, _written, JS  # noqa: E402
from ghost_agent.core.write_continuation import END_MARKER  # noqa: E402


@pytest.fixture
def _exec_env(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "spec")
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "-1")

    async def _no_window(llm):
        return 0
    monkeypatch.setattr(CE, "_window_tokens", _no_window)


async def test_a_build_stopped_mid_file_writes_the_files_it_finished(_exec_env):
    # world (review r3 MAJOR): the hard stop's CancelledError passed `_write_pending` — a.js lost
    class _Stopped(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW the complete file: b.js" in str(payload["messages"]):
                raise asyncio.CancelledError()
            return await super().chat_completion(payload, is_background, **kw)
    llm = _Stopped([(_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
                    (JS + END_MARKER, "stop")])
    runner = FakeRunner()
    with pytest.raises(asyncio.CancelledError):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner)
    assert _written(runner, "a.js") == [JS]


async def test_a_build_stopped_in_its_second_spec_call_writes_what_a_refusal_left(_exec_env):
    spec1 = json.dumps({"files": [{"path": "style.css", "content": "body {\n...\n}\n"},
                                  {"path": "b.js", "generate": "b"}], "verify": "", "summary": "s", "ledger": ""})

    class _StopsOnSecondSpec(ScriptLLM):
        specs = 0

        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW" not in str(payload["messages"]):
                self.specs += 1
                if self.specs == 2:
                    raise asyncio.CancelledError()
            return await super().chat_completion(payload, is_background, **kw)
    llm = _StopsOnSecondSpec([(spec1, "stop"), ("export const b = 1;\n" + END_MARKER, "stop")])
    runner = FakeRunner()
    with pytest.raises(asyncio.CancelledError):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert _written(runner, "b.js") == ["export const b = 1;\n"]


async def test_a_flush_that_hangs_does_not_hold_the_stop(_exec_env, monkeypatch):
    monkeypatch.setattr(CE, "STOP_FLUSH_S", 0.2)

    class _Slow(FakeRunner):
        async def __call__(self, name, args):
            if name == "file_system" and args.get("path") == "a.js" and args.get("operation") != "read":
                await asyncio.sleep(30)
            return await super().__call__(name, args)

    class _Stopped(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW the complete file: b.js" in str(payload["messages"]):
                raise asyncio.CancelledError()
            return await super().chat_completion(payload, is_background, **kw)
    llm = _Stopped([(_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
                    (JS + END_MARKER, "stop")])
    t0 = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=_Slow())
    assert time.monotonic() - t0 < 5


async def test_a_stop_inside_the_batch_writes_the_files_still_pending(_exec_env):
    # world (review r4): a cancel on a.js's write lost the fully generated b.js
    class _StopOnA(FakeRunner):
        async def __call__(self, name, args):
            if (name == "file_system" and args.get("path") == "a.js" and args.get("operation") != "read"
                    and not getattr(self, "_stopped", False)):
                self._stopped = True
                raise asyncio.CancelledError()
            return await super().__call__(name, args)
    llm = ScriptLLM([(_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
                     (JS + END_MARKER, "stop"), ("export const b = 1;\n" + END_MARKER, "stop")])
    runner = _StopOnA()
    with pytest.raises(asyncio.CancelledError):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner)
    assert _written(runner, "b.js") == ["export const b = 1;\n"]


async def test_a_stopped_background_build_does_not_flush_behind_its_reopen(_exec_env):
    # world (review r5): the unattended step reopens its leaf without waiting — a late flush
    # landed inside the next build of the same leaf
    class _Stopped(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW the complete file: b.js" in str(payload["messages"]):
                raise asyncio.CancelledError()
            return await super().chat_completion(payload, is_background, **kw)
    llm = _Stopped([(_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
                    (JS + END_MARKER, "stop")])
    runner = FakeRunner()
    with pytest.raises(asyncio.CancelledError):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, is_background=True)
    assert _written(runner, "a.js") == []


# ── r5: a verify the stop killed is a stop, not a failed attempt ──────────

async def test_a_verify_killed_by_the_stop_ends_the_build_as_a_stop(_exec_env):
    # world (review r5 MAJOR): the killed verify (exit 130) read as a failed attempt; the build
    # returned FAILED before the cancel landed and the reopen found a FAILED leaf
    from ghost_agent.utils import tool_stop as TS
    spec = json.dumps({"files": [{"path": "a.py", "content": "print(1)\n"}], "verify": "python3 a.py",
                       "summary": "s", "ledger": ""})
    stop = TS.ToolStop()

    class _KilledVerify(FakeRunner):
        async def __call__(self, name, args):
            if name == "execute":
                stop.request()
                return "EXIT CODE: 130\nkilled"
            return await super().__call__(name, args)
    llm = ScriptLLM([(spec, "stop")] * 3)
    tok = TS.bind(stop)
    try:
        with pytest.raises(asyncio.CancelledError):
            await CE.build_coding_task(_ctx(llm), "build", tool_runner=_KilledVerify(), max_attempts=1)
    finally:
        TS.unbind(tok)


async def test_a_build_without_a_stop_still_fails_normally(_exec_env):
    spec = json.dumps({"files": [{"path": "a.py", "content": "print(1)\n"}], "verify": "python3 a.py",
                       "summary": "s", "ledger": ""})

    class _Fails(FakeRunner):
        async def __call__(self, name, args):
            if name == "execute":
                return "EXIT CODE: 1\nTraceback: boom"
            return await super().__call__(name, args)
    res = await CE.build_coding_task(_ctx(ScriptLLM([(spec, "stop")] * 3)), "build", tool_runner=_Fails(),
                                     max_attempts=1)
    assert res.ok is False


async def test_a_build_whose_caller_does_not_wait_never_flushes_late(_exec_env):
    # world (review r5): a scheduled batch's build is foreground AND unattended — it flushed
    # after the step had reopened its leaf
    class _Stopped(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW the complete file: b.js" in str(payload["messages"]):
                raise asyncio.CancelledError()
            return await super().chat_completion(payload, is_background, **kw)
    llm = _Stopped([(_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
                    (JS + END_MARKER, "stop")])
    runner = FakeRunner()
    tok = CE.caller_waits_context.set(False)
    try:
        with pytest.raises(asyncio.CancelledError):
            await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner)
    finally:
        CE.caller_waits_context.reset(tok)
    assert _written(runner, "a.js") == []


def test_the_unattended_step_says_it_does_not_wait():
    import ghost_agent.core.project_advancer as PA
    seen = []

    async def probe():
        seen.append(CE.caller_waits_context.get())
    asyncio.run(PA._with_leaf_deadline(probe(), time.monotonic() + 60))
    assert seen == [False] and CE.caller_waits_context.get() is True


# ── the simplification (operator 10-30): a running project build is waited for ──

async def test_the_hard_limit_waits_for_a_running_project_build(slack_req):
    # world: the stop cut a build mid-way — six review rounds of stuck leaves and lost files
    from ghost_agent.utils.tool_stop import ToolStop
    slack_req(1900)
    reg = TurnRegistry()
    turn = reg.register(RID, session_id="slack:t:u", kind="owner")
    turn.running = True
    stop = ToolStop()
    stop._running[0] = "manage_projects"
    turn.tool_stop = stop
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02, report_grace_s=0.3))
    await asyncio.sleep(0.2)
    assert not t.done() and not stop.requested() and not turn.cancelled       # the build runs on
    stop._running.clear()                                                      # it returned
    await asyncio.sleep(0.15)
    assert not turn.cancelled                                                  # the report's grace
    assert await asyncio.wait_for(t, 2) is True
    assert turn.cancelled and turn.reason.startswith(A.HARD_LIMIT_REASON)      # then the turn stops


async def test_other_tools_beside_a_build_are_not_cut_either_until_it_returns(slack_req):
    from ghost_agent.utils.tool_stop import ToolStop
    slack_req(1900)
    reg = TurnRegistry()
    turn = reg.register(RID)
    turn.running = True
    stop = ToolStop()
    stop._running.update({0: "manage_projects", 1: "execute"})
    turn.tool_stop = stop
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02, report_grace_s=0.0))
    await asyncio.sleep(0.15)
    assert not stop.requested()
    stop._running.pop(0)
    assert await asyncio.wait_for(t, 2) is True and stop.requested()           # the rest is cut


async def test_a_user_stop_during_the_wait_ends_the_watch(slack_req):
    from ghost_agent.utils.tool_stop import ToolStop
    slack_req(1900)
    reg = TurnRegistry()
    turn = reg.register(RID)
    turn.running = True
    stop = ToolStop()
    stop._running[0] = "manage_projects"
    turn.tool_stop = stop
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02))
    await asyncio.sleep(0.1)
    reg.cancel(RID, reason="cancelled by user")
    assert await asyncio.wait_for(t, 2) is False and turn.reason == "cancelled by user"


async def test_without_a_spared_build_there_is_no_grace(slack_req):
    # world: the grace leaked to every stop — Slack ran 5 min over on any turn
    slack_req(1800 - 0.1)
    reg = TurnRegistry()
    t = reg.register(RID)
    t.running = True
    t0 = time.monotonic()
    assert await A.watch_hard_limit(RID, reg, poll_s=0.02, report_grace_s=30.0) is True
    assert time.monotonic() - t0 < 2


async def test_a_build_that_returned_just_before_the_limit_still_gets_the_report_grace(slack_req, monkeypatch):
    # world (review r7): the grace started only for a build running AT the limit — one that
    # returned 0.1 s before it had its report cut at once
    from ghost_agent.utils.tool_stop import ToolStop
    left = {"v": 0.3}
    monkeypatch.setattr(L, "request_remaining_s", lambda rid: left["v"])
    reg = TurnRegistry()
    turn = reg.register(RID)
    turn.running = True
    stop = ToolStop()
    stop._running[0] = "manage_projects"
    turn.tool_stop = stop
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02, report_grace_s=0.5))
    await asyncio.sleep(0.1)
    stop._running.clear()                      # the build returned before the limit
    await asyncio.sleep(0.05)
    left["v"] = -1.0                           # the limit passes during the report
    await asyncio.sleep(0.15)
    assert not turn.cancelled                  # the report has its grace
    assert await asyncio.wait_for(t, 2) is True


async def test_a_build_wedged_past_the_spare_cap_is_cut(slack_req):
    # world (review r7): a hung build held the one turn lock with no bound
    from ghost_agent.utils.tool_stop import ToolStop
    slack_req(1900)
    reg = TurnRegistry()
    turn = reg.register(RID)
    turn.running = True
    stop = ToolStop()
    stop._running[0] = "manage_projects"
    turn.tool_stop = stop
    assert await asyncio.wait_for(A.watch_hard_limit(RID, reg, poll_s=0.02, spare_max_s=0.2), 2) is True
    assert stop.requested() and turn.cancelled


async def test_a_build_long_before_the_limit_gives_no_grace(slack_req):
    slack_req(1800 - 0.1)
    reg = TurnRegistry()
    t = reg.register(RID)
    t.running = True
    t0 = time.monotonic()
    assert await A.watch_hard_limit(RID, reg, poll_s=0.02, report_grace_s=30.0) is True
    assert time.monotonic() - t0 < 2


async def test_the_default_grace_is_the_requests_own_report_reserve(slack_req, monkeypatch):
    # world (review r7): a fixed 300 s grace was below the measured report turn
    from ghost_agent.utils.tool_stop import ToolStop
    seen = []
    monkeypatch.setattr(A, "effective_report_floor", lambda d, rid=None: seen.append(d) or 0.4)
    monkeypatch.setattr(A, "HARD_LIMIT_REPORT_GRACE_S", 0.1)
    slack_req(1900)
    reg = TurnRegistry()
    turn = reg.register(RID)
    turn.running = True
    stop = ToolStop()
    stop._running[0] = "manage_projects"
    turn.tool_stop = stop
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02, client_s=1800))
    await asyncio.sleep(0.06)
    stop._running.clear()
    await asyncio.sleep(0.25)
    assert not turn.cancelled and seen and seen[0] >= 1800      # 0.4 s, not the 0.1 constant
    assert await asyncio.wait_for(t, 2) is True


async def test_an_owner_stop_during_the_spare_ends_the_watch_without_a_second_cancel(slack_req):
    from ghost_agent.utils.tool_stop import ToolStop
    slack_req(1900)
    reg = TurnRegistry()
    turn = reg.register(RID)
    turn.running = True
    stop = ToolStop()
    stop._running[0] = "manage_projects"
    turn.tool_stop = stop
    t = asyncio.ensure_future(A.watch_hard_limit(RID, reg, poll_s=0.02, spare_max_s=0.3))
    await asyncio.sleep(0.06)
    reg.cancel(RID, reason="the owner pressed stop")
    assert await asyncio.wait_for(t, 2) is False
    assert not stop.requested() and turn.reason == "the owner pressed stop"
