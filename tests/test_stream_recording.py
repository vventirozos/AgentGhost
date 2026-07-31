"""Tests for the streaming recording hook (§4F Phase 2b).

The main tool loop streams every generation, so the recorder previously
never saw a tool-bearing call (7 fixtures in 21 h of heavy traffic).
Contracts: SSE deltas fold tolerantly into the accumulator; one
OpenAI-shaped record is reassembled on clean completion; empty/error
streams produce no record; the off path never touches the accumulator.
"""

import json

from ghost_agent.core.llm import LLMClient


def _acc():
    return {"content": [], "reasoning": [], "tool_calls": {}, "finish": None}


def _line(delta=None, finish=None):
    choice = {}
    if delta is not None:
        choice["delta"] = delta
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]})


class TestAccumulate:
    def test_content_deltas_fold_in_order(self):
        acc = _acc()
        LLMClient._stream_rec_accumulate(_line({"content": "Hel"}), acc)
        LLMClient._stream_rec_accumulate(_line({"content": "lo"}), acc)
        assert "".join(acc["content"]) == "Hello"

    def test_reasoning_and_finish_captured(self):
        acc = _acc()
        LLMClient._stream_rec_accumulate(
            _line({"reasoning_content": "think"}), acc)
        LLMClient._stream_rec_accumulate(_line({}, finish="stop"), acc)
        assert acc["reasoning"] == ["think"]
        assert acc["finish"] == "stop"

    def test_tool_syntax_content_preserved_verbatim(self):
        # Native-tools stacks embed tool calls IN content — the miner
        # depends on the reassembled text carrying them untouched.
        acc = _acc()
        frag = '<tool_call>{"name": "execute", "argu'
        LLMClient._stream_rec_accumulate(_line({"content": frag}), acc)
        assert acc["content"] == [frag]

    def test_garbage_lines_ignored(self):
        acc = _acc()
        for junk in ("", ": keepalive", "data: [DONE]", "data: not json",
                     'data: {"choices": "weird"}', "event: ping"):
            LLMClient._stream_rec_accumulate(junk, acc)
        assert acc == _acc()


class TestReassemble:
    def test_reassembled_shape(self):
        acc = _acc()
        LLMClient._stream_rec_accumulate(_line({"content": "hi "}), acc)
        LLMClient._stream_rec_accumulate(_line({"content": "there"}), acc)
        LLMClient._stream_rec_accumulate(_line({}, finish="stop"), acc)
        resp = LLMClient._stream_rec_response(acc)
        msg = resp["choices"][0]["message"]
        assert msg["content"] == "hi there"
        assert msg["role"] == "assistant"
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert "reasoning_content" not in msg

    def test_reasoning_included_when_present(self):
        acc = _acc()
        LLMClient._stream_rec_accumulate(
            _line({"reasoning_content": "t"}), acc)
        LLMClient._stream_rec_accumulate(_line({"content": "x"}), acc)
        resp = LLMClient._stream_rec_response(acc)
        assert resp["choices"][0]["message"]["reasoning_content"] == "t"

    def test_empty_stream_yields_no_record(self):
        assert LLMClient._stream_rec_response(_acc()) is None


class TestToolCallDeltas:
    """Native-tools streaming: the parsed call arrives as indexed
    delta.tool_calls fragments with EMPTY content — the whole reason the
    first hook version recorded blank tool fixtures."""

    def test_fragmented_tool_call_reassembles(self):
        acc = _acc()
        LLMClient._stream_rec_accumulate("data: " + json.dumps({"choices": [
            {"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                       "function": {"name": "execute"}}]}}
        ]}), acc)
        LLMClient._stream_rec_accumulate("data: " + json.dumps({"choices": [
            {"delta": {"tool_calls": [{"index": 0,
                                       "function": {"arguments": '{"cmd": '}}]}}
        ]}), acc)
        LLMClient._stream_rec_accumulate("data: " + json.dumps({"choices": [
            {"delta": {"tool_calls": [{"index": 0,
                                       "function": {"arguments": '"ls"}'}}]},
             "finish_reason": "tool_calls"}
        ]}), acc)
        resp = LLMClient._stream_rec_response(acc)
        tc = resp["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "execute"
        assert tc["function"]["arguments"] == '{"cmd": "ls"}'
        assert tc["id"] == "c1"
        assert resp["choices"][0]["finish_reason"] == "tool_calls"

    def test_multiple_indexed_calls_kept_in_order(self):
        acc = _acc()
        for idx, name in ((1, "second"), (0, "first")):
            LLMClient._stream_rec_accumulate(
                "data: " + json.dumps({"choices": [
                    {"delta": {"tool_calls": [
                        {"index": idx, "function": {"name": name}}]}}]}), acc)
        resp = LLMClient._stream_rec_response(acc)
        names = [t["function"]["name"]
                 for t in resp["choices"][0]["message"]["tool_calls"]]
        assert names == ["first", "second"]

    def test_tool_calls_alone_produce_a_record(self):
        acc = _acc()
        LLMClient._stream_rec_accumulate("data: " + json.dumps({"choices": [
            {"delta": {"tool_calls": [{"index": 0,
                                       "function": {"name": "recall"}}]}}
        ]}), acc)
        resp = LLMClient._stream_rec_response(acc)
        assert resp is not None
        assert resp["choices"][0]["message"]["content"] == ""
