"""Tests for stylometric egress scrubbing (utils/stylometry.py) and its
wiring into the search tools."""

import pytest

from ghost_agent.utils.stylometry import scrub_query, neutralize_query
from ghost_agent.tools import search as search_mod


# ──────────────────────────────────────────────────────────────────────
# scrub_query
# ──────────────────────────────────────────────────────────────────────

def test_strips_request_frame_and_politeness():
    out = scrub_query("Please can you find me the best Python ORM?")
    assert "please" not in out
    assert "can you find me" not in out
    assert out == out.lower()
    assert not out.endswith("?")
    assert "python orm" in out


def test_preserves_technical_symbols():
    out = scrub_query("How to use C++ and C# with .NET?")
    assert "c++" in out
    assert "c#" in out
    assert ".net" in out
    # "how to" is a universal idiom, not a personal fingerprint — kept.
    assert out.startswith("how to")


def test_removes_standalone_politeness_tokens():
    out = scrub_query("reverse a linked list please thanks")
    assert "please" not in out.split()
    assert "thanks" not in out.split()
    assert "reverse a linked list" in out


def test_first_person_opener_stripped():
    out = scrub_query("I'm looking for async patterns in Rust")
    assert not out.startswith("i'm looking for")
    assert "async patterns in rust" in out


def test_empty_and_none():
    assert scrub_query("") == ""
    assert scrub_query(None) == ""
    assert scrub_query("   ") .strip() == ""


def test_idempotent():
    q = "Could you please tell me about gradient descent?"
    once = scrub_query(q)
    assert scrub_query(once) == once


def test_never_empties_content():
    # A query that is ONLY a request frame still returns something usable.
    out = scrub_query("search for")
    assert out  # not empty


# ──────────────────────────────────────────────────────────────────────
# neutralize_query (LLM tier + fallback)
# ──────────────────────────────────────────────────────────────────────

class _StubLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    async def chat_completion(self, payload):
        self.calls += 1
        return {"choices": [{"message": {"content": self.content}}]}


class _BoomLLM:
    async def chat_completion(self, payload):
        raise RuntimeError("upstream down")


async def test_neutralize_no_client_falls_back_to_lexical():
    q = "Please find me the fastest sorting algorithm!"
    assert await neutralize_query(q, llm_client=None) == scrub_query(q)


async def test_neutralize_uses_llm_output():
    llm = _StubLLM("fastest sorting algorithm")
    out = await neutralize_query("Please find me the fastest sorting algorithm!",
                                 llm_client=llm, model="m")
    assert llm.calls == 1
    assert out == "fastest sorting algorithm"


async def test_neutralize_strips_label_and_quotes():
    llm = _StubLLM('Query: "python asyncio gather"')
    out = await neutralize_query("how do I gather coroutines", llm_client=llm)
    assert out == "python asyncio gather"


async def test_neutralize_llm_failure_falls_back():
    q = "Could you please look up CRDT conflict resolution?"
    out = await neutralize_query(q, llm_client=_BoomLLM())
    assert out == scrub_query(q)


# ──────────────────────────────────────────────────────────────────────
# wiring into tool_search (monkeypatch DDGS layer — no network)
# ──────────────────────────────────────────────────────────────────────

async def test_tool_search_scrubs_when_anonymous(monkeypatch):
    captured = {}

    async def fake_ddgs(query, tor_proxy):
        captured["query"] = query
        return "ok"

    monkeypatch.setattr(search_mod, "tool_search_ddgs", fake_ddgs)
    await search_mod.tool_search(query="Please find me C++ tutorials!",
                                 anonymous=True, tor_proxy=None)
    q = captured["query"]
    assert "please find me" not in q
    assert "c++" in q
    assert q == q.lower()


async def test_tool_search_passes_through_when_not_anonymous(monkeypatch):
    captured = {}

    async def fake_ddgs(query, tor_proxy):
        captured["query"] = query
        return "ok"

    monkeypatch.setattr(search_mod, "tool_search_ddgs", fake_ddgs)
    original = "Please find me C++ tutorials!"
    await search_mod.tool_search(query=original, anonymous=False, tor_proxy=None)
    assert captured["query"] == original  # untouched when not anonymous
