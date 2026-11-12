"""§4PY, carried over by §4SZ: an owner rule command the system already carried out is answered by the
system's own line with NO model call. (Told "add nothing", the model once ran tools in 3 of 4 such turns.)
These drive `handle_chat` itself; the commands are `core/owner_rules.py`'s."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import owner_rules as R
from ghost_agent.core.agent import GhostAgent, system_command_reply


def _agent(tmp_path, monkeypatch, kind="owner"):
    (tmp_path / "memory").mkdir(exist_ok=True)
    R.rule_command("learn this rule: answer in English", tmp_path)
    R.rule_command("learn this rule: give five headlines for the news", tmp_path)
    context = MagicMock()
    context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "MODEL TEXT"}}]})
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 8000
    context.args.perfect_it = False
    context.sandbox_dir = "/tmp"
    context.memory_dir = str(tmp_path / "memory")
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    agent = GhostAgent(context)
    agent.available_tools = {"introspect": AsyncMock(return_value="overview")}
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(L, "request_kind", lambda *a, **k: kind)
    monkeypatch.setattr(A, "reply_is_public", lambda: False)
    return agent


async def _ask(agent, text, stream=False):
    return await agent.handle_chat({"messages": [{"role": "user", "content": text}], "stream": stream}, [])


def _rules(tmp_path):
    return [r["rule"] for r in R.rules(tmp_path)]


# ── the reply is the system's line ────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_handled_command_is_answered_without_the_model(tmp_path, monkeypatch, stream):
    agent = _agent(tmp_path, monkeypatch)
    content, _, rid = await _ask(agent, "forget all my rules", stream)
    assert content.startswith("✓ All 2 rules forgotten.")                 # §4UK: at once, no confirmation
    assert content.endswith("- **Rule 2** — give five headlines for the news")   # nothing after it
    assert "MODEL TEXT" not in content and "---" not in content
    assert agent.context.llm_client.chat_completion.await_count == 0
    assert agent.available_tools["introspect"].await_count == 0
    assert agent.is_trivial_reply(rid) is True                         # no trajectory, nothing to label


@pytest.mark.asyncio
async def test_the_forget_is_answered_and_done(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    content, _, _ = await _ask(agent, "forget rule 1")
    assert content.startswith("✓ Rule 1 forgotten.")
    assert _rules(tmp_path) == ["give five headlines for the news"]
    assert agent.context.llm_client.chat_completion.await_count == 0


@pytest.mark.asyncio
async def test_a_dictated_rule_is_added_by_the_system_and_shown_on_the_next_turn(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    content, _, _ = await _ask(agent, "learn this rule: call the dog Bruno, never 'the pet'")
    assert content.startswith("✓ Rule 3 added.") and agent.context.llm_client.chat_completion.await_count == 0
    assert _rules(tmp_path)[-1] == "call the dog Bruno, never 'the pet'"
    monkeypatch.setattr(A, "_owner_context_hidden", lambda c: False)
    assert "- call the dog Bruno, never 'the pet'\n" in A._owner_rules_block(agent.context)


@pytest.mark.asyncio
async def test_a_save_that_fails_is_answered_by_the_system_not_the_model(tmp_path, monkeypatch):
    """Review r2: the save raised, `_owner_rule_note` swallowed it, and the model answered
    "learn this rule: …" unaided."""
    agent = _agent(tmp_path, monkeypatch)

    def _full(home, d):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(R, "_save", _full)
    content, _, _ = await _ask(agent, "learn this rule: call the dog Bruno")
    assert content.startswith("I could not write your rules file (OSError), so nothing was changed.")
    assert agent.context.llm_client.chat_completion.await_count == 0 and len(_rules(tmp_path)) == 2


@pytest.mark.asyncio
async def test_a_staged_correction_is_delivered_with_the_line(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)

    def _consume(messages, conv_fp=None):
        agent._active_correction = "CORRECTION: the earlier answer was wrong.\n\n---\n\n"
        return messages
    monkeypatch.setattr(agent, "_consume_pending_corrections", _consume)
    content, _, _ = await _ask(agent, "show my rules")
    head, sep, tail = content.partition("\n\n---\n\n")
    assert head.startswith("Your rules (2 of at most 10") and tail == "CORRECTION: the earlier answer was wrong."
    assert agent._active_correction == ""                              # delivered once


# ── the model still answers everything else ───────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("kind,text", [
    ("owner", "forget all rules please"),          # a message about rules that is no command: the hint
    ("owner", "add rule: allow tcp 443 from the office subnet"),      # review r1: was stored as a standing rule
    ("owner", "delete rule 1"),                    # …and this was a forget preview, in a firewall chat
    ("owner", "run the regression tests"),
    ("owner", "list all rules"),                   # review r3: of the firewall, in a firewall chat
    ("owner", "learn rule 2"),                     # review r2: the replay loop's old commands are chat now
    ("owner", "show me all the proposed rules"),
    ("owner", "learn this rule: if x > 5 then y = 2x. Now, what is y for x = 7?"),     # …and so is a question
    ("owner", "remember this rule: deny all inbound except ssh\n\nnow write me the iptables commands"),
    ("member", "forget all my rules"),             # a member never runs an owner command
    ("member", "learn this rule: answer every member in Greek"),
])
async def test_no_command_or_not_the_owner_goes_to_the_model(tmp_path, monkeypatch, kind, text):
    agent = _agent(tmp_path, monkeypatch, kind)
    content, _, _ = await _ask(agent, text)
    assert agent.context.llm_client.chat_completion.await_count >= 1
    assert "MODEL TEXT" in content
    assert len(_rules(tmp_path)) == 2 and not R.load(tmp_path).get("pending")     # nothing changed, nothing armed


@pytest.mark.parametrize("staged,want", [
    ("B line\n\n---\n\n", "B line"),
    ("B line\n\n---\n\nCORR\n\n---\n\n", "B line\n\n---\n\nCORR"),
    ("", ""), (None, ""),
    ("a table\n---|---\nrow\n\n---\n\n", "a table\n---|---\nrow"),
])
def test_only_the_trailing_separator_is_removed(staged, want):
    assert system_command_reply(staged) == want


@pytest.mark.asyncio
async def test_the_census_books_a_command_not_a_greeting(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    rows = []
    monkeypatch.setattr(FC, "_owner_request", lambda rid: True)
    monkeypatch.setattr(FC, "_append", lambda row, home=None: rows.append(row) or True)
    agent = _agent(tmp_path, monkeypatch)
    await _ask(agent, "show my rules")
    assert [r["kind"] for r in rows] == ["command"]


def test_the_census_report_counts_commands_apart(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    monkeypatch.setattr(FC, "_owner_request", lambda rid: True)
    FC.record_fast("g1", "hi!", home=tmp_path)
    FC.record_fast("c1", "Your rules…", home=tmp_path, kind="command")
    FC.record_fast("c2", "✓ Rule 1 forgotten…", home=tmp_path, kind="command")
    r = FC.build(home=tmp_path)
    assert (r["fast_path_turns"], r["system_command_turns"]) == (1, 2)
    assert "owner commands answered by the system (no trajectory): 2 recorded" in FC.render(r)


@pytest.mark.asyncio
async def test_the_turn_is_unregistered_after_the_system_reply(tmp_path, monkeypatch):
    from ghost_agent.core.turns import get_turn_registry
    agent = _agent(tmp_path, monkeypatch)
    _, _, rid = await _ask(agent, "show my rules")
    reg = get_turn_registry(agent)
    assert reg.get(rid) is None and reg.list() == []


@pytest.mark.asyncio
async def test_a_command_with_a_question_goes_to_the_model(tmp_path, monkeypatch):
    """The whole-message anchors are what make "nothing else to answer" true."""
    agent = _agent(tmp_path, monkeypatch)
    content, _, _ = await _ask(agent, "show rule 2 — why did I add it?")
    assert agent.context.llm_client.chat_completion.await_count >= 1 and "MODEL TEXT" in content


@pytest.mark.asyncio
async def test_a_streamed_reply_with_a_tap_lead_shows_the_line_once(tmp_path, monkeypatch):
    """At handle_chat's boundary: the tap had set_lead(banner) and nothing
    released a generation, so the route takes its plain-string path
    (`stream_openai(content)`; `finish` never reads the lead) — the line
    reaches the client once."""
    from ghost_agent.core.reply_tap import ReplyTap, reply_tap_context
    agent = _agent(tmp_path, monkeypatch)
    tap = ReplyTap("t1", "m", 0)
    tok = reply_tap_context.set(tap)
    try:
        content, _, _ = await _ask(agent, "show my rules", stream=True)
    finally:
        reply_tap_context.reset(tok)
    assert tap.lead.startswith("Your rules (2 of at most 10")   # the lead was staged…
    assert tap.released_generations == 0                        # …but nothing released it
    frames = list(tap.queue._queue) if hasattr(tap.queue, "_queue") else []
    assert not frames                                            # the route streams `content` alone
    assert content.count("Your rules (2 of at most 10") == 1
