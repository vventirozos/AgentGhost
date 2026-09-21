"""The language rule (2026-09-21) and its measurement.

Operator: "while speaking to the agent in English, the agent sometimes
translates my search to Greek, why?" Measured on the recorded turns: 18 of
167 English requests that searched issued a Greek-script query (all
Greece-local subjects — the model's own retrieval judgment, no rule told
it to; "jiujitsu north athens" in English returned Missouri), 46 of 983
queries mixed the scripts, and 14 of 2,112 English requests were ANSWERED
mostly in Greek (0.66% on prose lines — quoted headlines excluded). There
was no language rule in the prompt at all. Now rule 5 under COGNITIVE
ARCHITECTURE: reply in the user's language, search in the language that
finds the sources, never translate a proper name, one script per query.

Pins: the rule reaches the MODEL (the system message of a real
`handle_chat` turn carries it — a consumer pin, not a grep), and the
measurement script classifies the shapes it is meant to count.
"""
import json
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.agent import GhostAgent
from scripts import measure_reply_language as M
from tests.helpers import FakeBgTasks, make_context

RULE_HEAD = "LANGUAGE: Reply in the language the user wrote their message in"


def _resp(content):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": []}}]}


async def _system_text(monkeypatch, user_text):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Understood."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": user_text}]}, FakeBgTasks())
    call = ctx.llm_client.chat_completion.call_args_list[0]
    msgs = call.kwargs.get("messages")
    if msgs is None and call.args:
        msgs = call.args[0].get("messages") if isinstance(call.args[0], dict) else call.args[0]
    return "\n".join(str(m.get("content")) for m in msgs if m.get("role") == "system")


async def test_the_language_rule_reaches_the_model(monkeypatch):
    system = await _system_text(monkeypatch, "Who is the actual father of Hector Koufontinas? Check the Greek press.")
    assert RULE_HEAD in system
    assert "NEVER translate a proper name" in system and "one script" in system


async def test_the_trivial_chat_prompt_carries_the_language_line_too(monkeypatch):
    """"hello there" → "Γεια σου!" (b2f61b94, 2026-08-03): the trivial-chat
    path has its OWN short prompt, which had no language line either."""
    system = await _system_text(monkeypatch, "hello there")
    assert "concise conversational AI" in system
    assert "in the language the user wrote in" in system


# ── the measurement's classifier ────────────────────────────────────

def _row(req, rep, queries=(), kind="user_request"):
    return {"id": "abcdef0123", "task_kind": kind, "user_request": req, "final_response": rep,
            "tool_calls": [{"name": "web_search", "arguments": json.dumps({"query": q})} for q in queries]}


def test_an_english_request_answered_in_greek_prose_is_a_mismatch():
    f = M.classify(_row("Who is the father of Hector Koufontinas?",
                        "Έχεις απόλυτο δίκιο — ο πατέρας του είναι ο Δημήτρης Κουφοντίνας, ηγετικό στέλεχος της 17Ν.\n\nΔεν υπάρχει καμία αμφιβολία."))
    assert f["req_greek"] < 0.05 and f["rep_greek_prose"] > 0.5


def test_quoted_greek_headlines_under_an_english_lead_are_not_a_mismatch():
    rep = ("Here are today's top headlines from Naftemporiki:\n\n"
           "1. **«Οδύσσεια» του Κρίστοφερ Νόλαν: Υπερπαραγωγή στην Ελλάδα**\n"
           "2. **Μελόνι: Ζητά χάρη για τον κοσμηματοπώλη**\n"
           "| # | Headline |\n|---|---|\n| 3 | ΔΕΗ: Νέα τιμολόγια από Σεπτέμβριο |\n"
           "Let me know if you want any of these expanded.")
    f = M.classify(_row("give me the news", rep))
    assert f["rep_greek_prose"] < 0.2, f


def test_query_switch_and_mixed_script_are_counted():
    f = M.classify(_row("is there any jiujitsu school at north athens?", "Yes — three schools: …" * 4,
                        queries=["jiujitsu north athens", "jiu jitsu athens Βόρεια Προάστια", "BJJ γυμναστήριο Χολαργός"]))
    assert f["n_queries"] == 3 and f["greek_queries"] == 2 and f["mixed_queries"] == 2


def test_internal_kinds_and_short_turns_are_excluded():
    assert M.classify(_row("Τι ξέρεις;", "DIAGNOSIS: the model failed to …" * 3, kind="reflection")) is None
    assert M.classify(_row("hi", "ok")) is None


def test_summary_rates():
    facts = [
        M.classify(_row("english one", "An English answer, long enough to count for the rate here.", queries=["q one", "Ελληνικό ερώτημα"])),
        M.classify(_row("english two", "Ελληνική απάντηση σε αγγλική ερώτηση, αρκετά μεγάλη ώστε να μετρήσει.")),
        M.classify(_row("Ελληνική ερώτηση εδώ", "An English answer to a Greek question, long enough to count.")),
    ]
    s = M.summarize(facts)
    assert s["english_requests"] == 2 and s["english_with_searches"] == 1
    assert s["query_switch"] == {"n": 1, "rate": 1.0}
    assert s["reply_mismatch_en"]["n"] == 1 and s["reply_mismatch_en"]["ids"] == ["abcdef01"]
    assert s["greek_requests"] == 1 and s["reply_mismatch_el"]["n"] == 1
