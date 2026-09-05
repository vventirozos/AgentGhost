"""§4EX (2026-09-05): every streamed frame a client sees carries the AGENT's
request id — and a failed label is reported once per bubble.

The defect: on the streamed-final-generation path the chat route yielded
llama-server's SSE frames verbatim, so the web UI captured llama's own
``chatcmpl-<32 chars>`` completion id, POSTed it to /api/feedback, and the
agent — whose trajectory row is filed under its 8-hex ``req_id`` — answered
"no trajectory found" for EVERY streamed turn. Operator thumbs were lost
silently for the corpus and loudly (once per tap) for the operator.

Three layers, each with a world in which it fails:
  1. the helper, as a matrix;
  2. the real FastAPI route with a stub agent streaming foreign-id frames —
     what the CLIENT receives is asserted, not what the helper returns;
  3. the cross-surface table (R5): the id the web client's capture rule
     extracts from those frames must be the id ``core.feedback`` resolves
     against a trajectory stamped the way ``_record_turn_trajectory``
     stamps it — with the pre-fix frames as the control that FAILS.
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ghost_agent.api.routes import _restamp_sse_request_id, _sse_delta_text  # noqa: E402
from ghost_agent.core.feedback import _matches_request, normalize_request_id  # noqa: E402
from tests.helpers import eval_js, extract_js_function, strip_js_comments  # noqa: E402

_STATIC = _ROOT / "interface" / "static"
REQ = "4f6dc15d"
LLAMA = "chatcmpl-GtcVXPW50mYbDGv0R32ReWFoTsTECgxK"     # llama.cpp's shape


def _frame(obj) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")


def _llama(delta, **extra) -> bytes:
    d = {"id": LLAMA, "object": "chat.completion.chunk", "created": 1,
         "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    d.update(extra)
    return _frame(d)


def _ids(body: bytes):
    out = []
    for frame in body.decode("utf-8").split("\n\n"):
        s = frame.strip()
        if s.startswith("data:") and s[5:].strip().startswith("{"):
            out.append(json.loads(s[5:].strip()).get("id"))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  1. the helper
# ═══════════════════════════════════════════════════════════════════════

class TestRestampHelper:

    def test_a_foreign_id_becomes_ours_and_the_text_is_untouched(self):
        out = _restamp_sse_request_id(_llama({"content": "héllo — ✓"}), REQ)
        assert isinstance(out, bytes)
        assert _ids(out) == [f"chatcmpl-{REQ}"]
        assert _sse_delta_text(out) == "héllo — ✓"
        assert out.endswith(b"\n\n")

    def test_an_id_less_frame_gets_one(self):
        """The client captures the FIRST id it sees; an id-less first frame
        would leave it holding llama's id from the second."""
        out = _restamp_sse_request_id(_frame({"choices": [{"delta": {"content": "x"}}]}), REQ)
        assert _ids(out) == [f"chatcmpl-{REQ}"]

    @pytest.mark.parametrize("chunk", [
        b"data: [DONE]\n\n",
        b": processing request...\n\n",
        b'event: error\ndata: {"error": {"message": "m", "type": "T"}}\n\n',
        b"data: not json at all\n\n",
        b"data: [1, 2, 3]\n\n",
        b"\xff\xfe not utf-8",
        b"",
    ])
    def test_non_object_frames_pass_through_byte_identical(self, chunk):
        assert _restamp_sse_request_id(chunk, REQ) is chunk

    def test_an_error_event_frame_is_left_alone_but_its_data_object_is_not_ours_to_stamp(self):
        """`event: error` frames are `event:`-first; the helper only touches
        frames whose first line is `data:` — the error frame's shape is a
        client contract (the UI reads `.error`)."""
        chunk = b'event: error\ndata: {"error": {"message": "m"}}\n\n'
        assert _restamp_sse_request_id(chunk, REQ) is chunk

    def test_multi_frame_chunk_stamps_every_object(self):
        chunk = _llama({"content": "a"}) + b"data: [DONE]\n\n" + _frame({"choices": []})
        out = _restamp_sse_request_id(chunk, REQ)
        assert _ids(out) == [f"chatcmpl-{REQ}", f"chatcmpl-{REQ}"]
        assert b"data: [DONE]" in out

    def test_already_ours_is_returned_as_is(self):
        chunk = _frame({"id": f"chatcmpl-{REQ}", "choices": [{"delta": {"content": "x"}}]})
        assert _restamp_sse_request_id(chunk, REQ) is chunk

    def test_empty_req_id_never_rewrites(self):
        chunk = _llama({"content": "x"})
        assert _restamp_sse_request_id(chunk, "") is chunk
        assert _restamp_sse_request_id(chunk, None) is chunk

    def test_str_in_str_out(self):
        chunk = _llama({"content": "x"}).decode("utf-8")
        out = _restamp_sse_request_id(chunk, REQ)
        assert isinstance(out, str) and f"chatcmpl-{REQ}" in out

    def test_usage_and_finish_reason_survive(self):
        chunk = _llama({}, usage={"prompt_tokens": 3, "completion_tokens": 4})
        out = json.loads(_restamp_sse_request_id(chunk, REQ)[6:].strip())
        assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}
        assert out["choices"][0]["finish_reason"] is None


# ═══════════════════════════════════════════════════════════════════════
#  2. the route — what the client receives
# ═══════════════════════════════════════════════════════════════════════

def _make_request(body):
    from fastapi import Request
    req = MagicMock(spec=Request)
    req.method = "POST"
    req.headers = {"content-type": "application/json"}
    req.body = AsyncMock(return_value=json.dumps(body).encode("utf-8"))
    req.json = AsyncMock(return_value=body)
    return req


async def _drive_route(frames, req_id=REQ):
    from ghost_agent.api.routes import chat_proxy

    async def _streamed_final():
        for f in frames:
            yield f

    async def fake_handle_chat(*args, **kwargs):
        return (_streamed_final(), 1, req_id)

    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"
    body = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}
    req = _make_request(body)
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await chat_proxy(req, MagicMock())
    out = b""
    async for chunk in resp.body_iterator:
        out += chunk if isinstance(chunk, (bytes, bytearray)) else str(chunk).encode("utf-8")
    return out


LLAMA_FRAMES = [
    _llama({"role": "assistant"}),
    _llama({"content": "Hey "}),
    _llama({"content": "Vasilis!"}),
    _frame({"id": LLAMA, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
    b"data: [DONE]\n\n",
]


class TestRouteReStampsWhatTheClientSees:

    @pytest.mark.asyncio
    async def test_every_frame_the_client_receives_carries_the_agents_id(self):
        body = await _drive_route(LLAMA_FRAMES)
        ids = _ids(body)
        assert ids and all(i == f"chatcmpl-{REQ}" for i in ids), ids
        assert LLAMA.encode() not in body, "llama's id leaked to the client"
        assert _sse_delta_text(body) == "Hey Vasilis!"
        assert body.rstrip().endswith(b"data: [DONE]")
        assert body.startswith(b": processing request...")

    @pytest.mark.asyncio
    async def test_the_control_world_without_the_restamp_leaks_llamas_id(self):
        """R6: the pin above must be able to fail — with the helper made a
        no-op, the client receives llama's id."""
        with patch("ghost_agent.api.routes._restamp_sse_request_id", side_effect=lambda c, r: c):
            body = await _drive_route(LLAMA_FRAMES)
        assert LLAMA.encode() in body


# ═══════════════════════════════════════════════════════════════════════
#  3. one input, one story: client capture == corpus match (R5)
# ═══════════════════════════════════════════════════════════════════════

def _client_captured_id(body: bytes):
    """Replay the web client's capture rule over the frames it received.
    The rule is lifted from app.js by text so a change there fails HERE."""
    app_js = strip_js_comments((_STATIC / "app.js").read_text(encoding="utf-8"))
    rule = re.search(r"if \(!currentReqId && typeof data\.id === 'string' && data\.id\) \{\s*"
                     r"currentReqId = data\.id\.replace\(/\^chatcmpl-/, ''\);", app_js)
    assert rule, "the client's id capture rule moved — update the replay"
    for frame in body.decode("utf-8").split("\n\n"):
        s = frame.strip()
        if not s.startswith("data:") or not s[5:].strip().startswith("{"):
            continue
        d = json.loads(s[5:].strip())
        if isinstance(d.get("id"), str) and d["id"]:
            return re.sub(r"^chatcmpl-", "", d["id"])
    return None


def _trajectory_stamped_like_the_agent(req_id):
    """`_record_turn_trajectory` stamps session_id=req_id and extra.req_id."""
    return SimpleNamespace(session_id=req_id, extra={"req_id": req_id})


class TestClientAndCorpusAgree:

    @pytest.mark.asyncio
    async def test_the_id_the_client_captures_resolves_the_trajectory(self):
        body = await _drive_route(LLAMA_FRAMES)
        captured = _client_captured_id(body)
        assert captured == REQ
        rid = normalize_request_id("chatcmpl-" + captured)
        assert _matches_request(_trajectory_stamped_like_the_agent(REQ), rid)

    @pytest.mark.asyncio
    async def test_the_pre_fix_frames_did_NOT_resolve(self):
        """The control: raw llama frames → the captured id matches no row.
        This is the exact production failure ('no trajectory found for
        request_id GtcVXPW…'), reproduced so the table can distinguish."""
        with patch("ghost_agent.api.routes._restamp_sse_request_id", side_effect=lambda c, r: c):
            body = await _drive_route(LLAMA_FRAMES)
        captured = _client_captured_id(body)
        assert captured != REQ and len(captured) == 32
        assert not _matches_request(_trajectory_stamped_like_the_agent(REQ), normalize_request_id(captured))

    def test_the_restamp_is_on_the_streamed_final_branch(self):
        src = (_ROOT / "src/ghost_agent/api/routes.py").read_text(encoding="utf-8")
        branch = src[src.index("if hasattr(content, '__aiter__'):"):src.index("_persist_session(\"\".join(_acc))")]
        assert "yield _restamp_sse_request_id(chunk, req_id)" in branch
        assert "yield chunk\n" not in branch, "a raw yield survived on the streamed-final branch"


# ═══════════════════════════════════════════════════════════════════════
#  4. the notice: once per bubble
# ═══════════════════════════════════════════════════════════════════════

_NOTICE_HARNESS = """
const said = [];
function addMessage(role, text) { said.push(role + ':' + text); }
"""


class TestFailureNoticeOnce:

    @pytest.fixture(scope="class")
    def fn(self):
        app_js = strip_js_comments((_STATIC / "app.js").read_text(encoding="utf-8"))
        return _NOTICE_HARNESS + extract_js_function(app_js, "_noteFeedbackFailure")

    def test_same_failure_speaks_once_different_speaks_again(self, fn):
        out = eval_js(fn + """
const div = { dataset: {} };
const r = [
  _noteFeedbackFailure(div, 'Feedback not recorded: no trajectory found'),
  _noteFeedbackFailure(div, 'Feedback not recorded: no trajectory found'),
  _noteFeedbackFailure(div, 'Feedback not recorded: no trajectory found'),
  _noteFeedbackFailure(div, 'Feedback not recorded: network error — tap again.'),
];
""", "({ r, said, latch: div.dataset.fbFailure })")
        assert out["r"] == [True, False, False, True]
        assert out["said"] == ["system:Feedback not recorded: no trajectory found",
                               "system:Feedback not recorded: network error — tap again."]
        assert out["latch"] == "Feedback not recorded: network error — tap again."

    def test_a_cleared_latch_speaks_again(self, fn):
        out = eval_js(fn + """
const div = { dataset: {} };
_noteFeedbackFailure(div, 'x'); delete div.dataset.fbFailure; _noteFeedbackFailure(div, 'x');
""", "said.length")
        assert out == 2

    def test_a_second_bubble_has_its_own_latch(self, fn):
        out = eval_js(fn + """
const a = { dataset: {} }, b = { dataset: {} };
_noteFeedbackFailure(a, 'x'); _noteFeedbackFailure(b, 'x');
""", "said.length")
        assert out == 2

    def test_a_missing_bubble_still_reports(self, fn):
        assert eval_js(fn + "\n_noteFeedbackFailure(null, 'x');\n", "said.length") == 1

    def test_both_failure_sites_route_through_it_and_success_clears_it(self):
        app_js = strip_js_comments((_STATIC / "app.js").read_text(encoding="utf-8"))
        fn = extract_js_function(app_js, "sendFeedback")
        assert fn.count("_noteFeedbackFailure(div,") == 2, "a failure site bypasses the latch"
        assert "addMessage('system', `Feedback not recorded" not in fn
        assert "addMessage('system', 'Feedback not recorded" not in fn
        assert "delete div.dataset.fbFailure;" in fn
        assert fn.index("delete div.dataset.fbFailure;") > fn.index("if (!r.ok) {")


# ═══════════════════════════════════════════════════════════════════════
#  5. the SECOND cause: a trivial-fast-path reply has no trajectory at all
# ═══════════════════════════════════════════════════════════════════════
#
# "hello" → "Hey Vasilis!" bypasses the turn loop and writes no trajectory
# by design (a greeting is not corpus material), so a thumb on it can never
# land — with or without the id fix. The agent remembers those req_ids, the
# route marks their frames `ghost.labelable: false`, and the web UI renders
# no thumbs on them instead of an error per tap.

class TestAgentRemembersTrivialReplies:

    def _agent_like(self):
        from ghost_agent.core.agent import GhostAgent as Agent
        # A REAL instance with no constructor run: the helpers are lazy on
        # purpose (the constructor is untouched), and the cap is a class attr.
        ns = Agent.__new__(Agent)
        return Agent, ns

    def test_noted_ids_are_trivial_unknown_and_empty_are_not(self):
        Agent, ns = self._agent_like()
        assert Agent.is_trivial_reply(ns, "abc") is False          # nothing noted yet
        Agent._note_trivial_reply(ns, "abc")
        Agent._note_trivial_reply(ns, "")                          # ignored
        assert Agent.is_trivial_reply(ns, "abc") is True
        assert Agent.is_trivial_reply(ns, "abd") is False
        assert Agent.is_trivial_reply(ns, "") is False
        assert Agent.is_trivial_reply(ns, None) is False

    def test_the_set_is_bounded_and_evicts_the_oldest(self):
        Agent, ns = self._agent_like()
        cap = Agent._TRIVIAL_REQ_IDS_CAP
        for i in range(cap + 5):
            Agent._note_trivial_reply(ns, f"r{i}")
        assert len(ns._trivial_req_ids) == cap
        assert Agent.is_trivial_reply(ns, "r0") is False
        assert Agent.is_trivial_reply(ns, "r4") is False
        assert Agent.is_trivial_reply(ns, "r5") is True
        assert Agent.is_trivial_reply(ns, f"r{cap + 4}") is True

    def test_re_noting_refreshes_age(self):
        Agent, ns = self._agent_like()
        cap = Agent._TRIVIAL_REQ_IDS_CAP
        Agent._note_trivial_reply(ns, "keep")
        for i in range(cap - 1):
            Agent._note_trivial_reply(ns, f"r{i}")
        Agent._note_trivial_reply(ns, "keep")      # refreshed → youngest
        Agent._note_trivial_reply(ns, "new")       # evicts r0, not keep
        assert Agent.is_trivial_reply(ns, "keep") is True
        assert Agent.is_trivial_reply(ns, "r0") is False

    def test_the_fast_path_return_notes_the_id(self):
        src = (_ROOT / "src/ghost_agent/core/agent.py").read_text(encoding="utf-8")
        i = src.index("fast_result = await self._handle_trivial_chat(")
        window = src[i:i + 900]
        assert "if fast_result is not None:" in window
        assert window.index("self._note_trivial_reply(req_id)") < window.index("return fast_result")


class TestStreamOpenaiExtra:

    @pytest.mark.asyncio
    async def test_extra_is_merged_into_every_json_frame_and_none_changes_nothing(self):
        from ghost_agent.core.llm import LLMClient
        client = LLMClient.__new__(LLMClient)
        plain = b"".join([c async for c in LLMClient.stream_openai(client, "m", "hello world, twenty chars+", 7, "abc12345")])
        marked = b"".join([c async for c in LLMClient.stream_openai(
            client, "m", "hello world, twenty chars+", 7, "abc12345", extra={"ghost": {"labelable": False}})])
        assert b'"ghost"' not in plain
        frames = [json.loads(f.strip()[5:]) for f in marked.decode().split("\n\n")
                  if f.strip().startswith("data:") and f.strip()[5:].strip().startswith("{")]
        assert len(frames) >= 4                                   # start + ≥2 slices + stop
        assert all(f["ghost"] == {"labelable": False} for f in frames)
        assert all(f["id"] == "chatcmpl-abc12345" for f in frames)
        assert _sse_delta_text(marked) == "hello world, twenty chars+"
        assert marked.rstrip().endswith(b"data: [DONE]")


async def _drive_string_route(content, agent):
    from ghost_agent.api.routes import chat_proxy

    async def fake_handle_chat(*args, **kwargs):
        return (content, 1, REQ)
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"
    from ghost_agent.core.llm import LLMClient
    llm = LLMClient.__new__(LLMClient)
    agent.context.llm_client.stream_openai = lambda *a, **k: LLMClient.stream_openai(llm, *a, **k)
    body = {"stream": True, "messages": [{"role": "user", "content": "hello"}]}
    req = _make_request(body)
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await chat_proxy(req, MagicMock())
    out = b""
    async for chunk in resp.body_iterator:
        out += chunk if isinstance(chunk, (bytes, bytearray)) else str(chunk).encode("utf-8")
    return out


def _ghost_flags(body: bytes):
    return [json.loads(f.strip()[5:]).get("ghost") for f in body.decode().split("\n\n")
            if f.strip().startswith("data:") and f.strip()[5:].strip().startswith("{")]


class TestRouteMarksTrivialReplies:

    @pytest.mark.asyncio
    async def test_a_trivial_reply_is_marked_on_every_frame(self):
        agent = MagicMock()
        agent.is_trivial_reply = lambda rid: rid == REQ
        body = await _drive_string_route("Hey Vasilis! Good to hear from you.", agent)
        flags = _ghost_flags(body)
        assert flags and all(f == {"labelable": False} for f in flags), flags
        assert _sse_delta_text(body) == "Hey Vasilis! Good to hear from you."

    @pytest.mark.asyncio
    async def test_a_real_reply_is_not_marked(self):
        agent = MagicMock()
        agent.is_trivial_reply = lambda rid: False
        flags = _ghost_flags(await _drive_string_route("The weather is fine.", agent))
        assert flags and all(f is None for f in flags), flags

    @pytest.mark.asyncio
    async def test_a_magicmock_agent_does_not_mark(self):
        """`is True`, not truthiness: a MagicMock answers truthy to any call,
        which would have marked every reply in every existing route test."""
        flags = _ghost_flags(await _drive_string_route("x" * 40, MagicMock()))
        assert flags and all(f is None for f in flags), flags


class TestClientRendersNoThumbsOnUnlabelable:

    @pytest.fixture(scope="class")
    def app_js(self):
        return strip_js_comments((_STATIC / "app.js").read_text(encoding="utf-8"))

    def test_stamp_sets_and_clears_the_flag_with_the_id(self, app_js):
        fn = extract_js_function(app_js, "_stampReqId")
        src = fn + """
const mk = () => ({ dataset: {}, querySelector: () => null, _fbTimer: null });
const a = mk(); _stampReqId(a, 'r1', true);
const b = mk(); _stampReqId(b, 'r2', false);
const c = mk(); _stampReqId(c, 'r3', true); _stampReqId(c, 'r4', false);   // reused bubble, new turn
const d = mk(); d.dataset.reqId = 'old'; d.dataset.feedback = 'positive'; _stampReqId(d, 'new', true);
"""
        out = eval_js(src, "({a: a.dataset, b: b.dataset, c: c.dataset, d: d.dataset})")
        assert out["a"] == {"reqId": "r1", "unlabelable": "1"}
        assert out["b"] == {"reqId": "r2"}
        assert out["c"] == {"reqId": "r4"}, "a reused bubble kept the old turn's no-thumbs flag"
        assert out["d"] == {"reqId": "new", "unlabelable": "1"}, "the stale feedback latch survived a new id"

    def test_the_feedback_row_is_skipped_for_an_unlabelable_bubble(self, app_js):
        fn = extract_js_function(app_js, "_ensureFeedbackRow")
        head = fn[:fn.index("const fbRow")]
        assert "if (div.dataset.unlabelable) return;" in head
        assert head.index("!div.dataset.reqId) return;") < head.index("div.dataset.unlabelable")

    def test_the_parser_reads_the_marker_and_every_reset_clears_it(self, app_js):
        assert "if (data.ghost && data.ghost.labelable === false) currentTurnUnlabelable = true;" in app_js
        # Declaration + every reset: wherever the id is cleared, the flag is
        # cleared beside it (a stale flag would strip thumbs from the NEXT
        # real reply; a stale id would label the wrong turn).
        assert app_js.count("currentReqId = null;") == 3
        assert app_js.count("currentTurnUnlabelable = false;") == app_js.count("currentReqId = null;"), (
            "a reset site clears currentReqId without clearing currentTurnUnlabelable")

    def test_history_and_wire_shape(self, app_js):
        # both assistant pushes carry the flag, next to the reqId they belong to
        assert app_js.count("unlabelable: currentTurnUnlabelable || undefined") == 2
        assert app_js.count("_stampReqId(currentAgentMessageDiv, currentReqId, currentTurnUnlabelable)") == 2
        # restore + server-history adoption keep it; the wire never sees it
        assert "if (msg.unlabelable) div.dataset.unlabelable = '1';" in app_js
        assert "if (loc.unlabelable) out[i].unlabelable = true;" in app_js
        fn = extract_js_function(app_js, "toWireMessage")
        out = eval_js(fn, "toWireMessage({role: 'assistant', content: 'x', reqId: 'r', feedback: 'positive', unlabelable: true})")
        assert out == {"role": "assistant", "content": "x"}
