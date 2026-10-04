"""§4LK (2026-10-04): the streamed reply path — what the web UI receives.
Each test names the world it fails in."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_stream_forced_final_retry import _drive
from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE


# ── a streamed final that tried to call a tool did not answer ─────────────────
@pytest.mark.asyncio
async def test_a_short_sentence_plus_a_tool_call_gets_the_answer_retry():
    """Fails where req b46518f0's reply was "Let me check what I was actually
    carrying before you came back." plus the "could not be parsed" note — the
    lexical narration test read "you" as addressed to the user."""
    turn = ("Let me check what I was actually carrying before you came back."
            "\\n\\n<tool_call>\\n<function=self_state>\\n<parameter=action>read</parameter>\\n</function>\\n</tool_call>")
    client, durable, retries = await _drive([turn.replace("\\\\n", "\\n")],
                                            retry_reply="I was carrying the question about your birthday plans.",
                                            prefix="", tools=[])
    assert len(retries) == 1 and "birthday plans" in client and "birthday plans" in durable


@pytest.mark.asyncio
async def test_a_full_answer_with_a_stray_call_is_not_retried():
    answer = ("## Summary\\n\\n" + "PostgreSQL 18.6 is the current minor release, published on 13 August 2026; "
              "it fixes several planner bugs and a replication slot leak. " * 3)
    turn = answer + "\\n\\n<tool_call>\\n<function=web_search>\\n<parameter=query>x</parameter>\\n</function>\\n</tool_call>"
    client, durable, retries = await _drive([turn.replace("\\\\n", "\\n")], retry_reply="SECOND", prefix="", tools=[])
    assert len(retries) == 0 and "PostgreSQL 18.6" in client and "SECOND" not in client


@pytest.mark.parametrize("text,tool,narr", [
    ("Let me check what I was actually carrying before you came back.", True, True),
    ("Let me check what I was actually carrying before you came back.", False, False),   # no call: addressed
    ("The capital of Australia is Canberra.", True, False),                               # an answer
])
def test_an_announcement_that_says_you_is_narration_only_when_a_call_followed(text, tool, narr):
    from ghost_agent.core.reply_smoothing import narration_only
    assert narration_only(text, tool_attempt=tool) is narr


@pytest.mark.asyncio
async def test_the_non_streamed_final_retries_an_announcement_that_says_you(monkeypatch):
    """The same world on the internal path: a tools-off final whose text only
    announces ("…before you came back.") and whose call was dropped."""
    from tests.helpers import FakeBgTasks
    from tests.test_forced_final_no_answer import _agent, _resp, _tc, FIRST
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me close task 1.", FIRST),
        _resp("Let me check what I was actually carrying before you came back.",
              [_tc("c2", "self_state", {"action": "read"})]),
        _resp("I was carrying one thing: the open question about your birthday plans."),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "start task 1"}]},
                                        FakeBgTasks())
    assert "birthday plans" in out


# ── what the streamed record and the stream itself carry ─────────────────────
from tests.test_finalize_stream_pins import make_stream_agent, sse
from tests.test_stream_forced_final_retry import _state


async def _drive2(deltas, *, prefix="", tools=(), pre_tool=(), request="q", verify=False):
    a = make_stream_agent()
    a.context.args.no_verifier = not verify
    if verify:
        a.context.verifier = MagicMock()

    async def _fake_verdict(**kw):
        return None
    a._compute_verifier_verdict = _fake_verdict
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock()
    a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock()
    a._write_project_work_log_safe = AsyncMock()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "X"}}]})
    reg = MagicMock(); reg.is_cancelled.return_value = False
    st = _state(reg, list(tools), prefix)
    st.pre_tool_segments = tuple(pre_tool)
    st.last_user_content = request
    gen, _, _ = a._stream_final_generation(st)
    chunks = [c async for c in gen]
    client = "".join(
        (json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    kw = a._record_turn_trajectory.call_args.kwargs if a._record_turn_trajectory.called else {}
    return client, kw, a


_WRITE = {"name": "file_system", "content": "SUCCESS: wrote 1204 bytes to 'app.py'."}
_ANSWER = ("I rewrote app.py so the parser handles the empty-header case and the date column, "
           "and added the retry around the upload step; the CLI flags are unchanged. " * 2)


@pytest.mark.asyncio
async def test_a_streamed_final_after_an_untested_write_is_flagged_and_failed():
    """Fails where the streamed path — unlike finalize — said nothing and
    recorded no failure for a write that was never run."""
    client, kw, _ = await _drive2([_ANSWER], tools=[_WRITE], request="fix app.py")
    assert "⚠ Unverified" in client and "⚠ Unverified" in kw["final_content"]
    assert kw["verifier"] == "failed"


@pytest.mark.asyncio
async def test_a_write_the_user_asked_not_to_run_is_said_not_failed():
    client, kw, _ = await _drive2([_ANSWER], tools=[_WRITE], request="edit app.py but don't run it")
    assert "Not run, as you asked" in client and kw["verifier"] is None


@pytest.mark.asyncio
async def test_the_previous_turns_banner_is_not_recorded_as_this_answer():
    """Fails where the record (trajectory, post-mortem lessons, calibration)
    stored the previous turn's correction banner as this turn's answer."""
    banner = "⚠️ **Correction to my previous answer:** the year was 2017.\n\n---\n\n"
    client, kw, _ = await _drive2([_ANSWER], prefix=banner)
    assert client.startswith("⚠️") and "Correction to my previous answer" not in kw["final_content"]


@pytest.mark.asyncio
async def test_the_streamed_record_drops_text_written_beside_lookups():
    narration = "Let me pull the release notes first."
    client, kw, _ = await _drive2([_ANSWER], prefix=narration + "\n\n", pre_tool=[narration],
                                  tools=[{"name": "web_search", "content": "results"}])
    assert narration in client and narration not in kw["final_content"]


@pytest.mark.asyncio
async def test_a_streamed_answers_late_correction_is_bound_to_that_answer():
    """Fails where the stream gate handed the late handler the bare
    first-message fingerprint."""
    from ghost_agent.core.agent import _reply_tag
    _, _, a = await _drive2([_ANSWER], tools=[{"name": "web_search", "content": "results"}], verify=True)
    assert a._attach_late_verdict_handler.called
    conv = a._attach_late_verdict_handler.call_args.args[2]
    assert conv.startswith("fp|r") and conv.endswith(_reply_tag(_ANSWER))


# ── corrections reach the owner ──────────────────────────────────────────────
from tests.test_critic_async import agent  # noqa: E402,F401 — fixture


def test_a_bound_correction_waits_a_day_and_an_unbound_one_fifteen_minutes(agent):
    from ghost_agent.core.agent import _CORRECTION_TTL, _reply_tag
    old = time.monotonic() - _CORRECTION_TTL - 60
    tag = _reply_tag("It launched in 2016.")
    first = [{"role": "user", "content": "when?"}]
    fp = agent._conversation_fingerprint(first)
    agent._pending_corrections = [
        {"note": "BOUND", "conv": f"{fp}|r{tag}", "ts": old, "traj": "a"},
        {"note": "UNBOUND", "conv": fp, "ts": old, "traj": "b"}]
    agent._consume_pending_corrections(first + [{"role": "assistant", "content": "It launched in 2016."},
                                                {"role": "user", "content": "sure?"}], conv_fp=fp)
    banner = agent._take_active_correction()
    assert "BOUND" in banner and "UNBOUND" not in banner


def test_the_correction_queue_survives_a_restart(tmp_path):
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context
    ctx = make_context()
    ctx.memory_dir = tmp_path
    a = GhostAgent(ctx)
    a._pending_corrections = [{"note": "the year is 2017", "conv": "fp|rabc", "ts": time.monotonic() - 30,
                               "traj": "t"}]
    a._save_pending_corrections()
    b = GhostAgent(ctx)
    assert [c["note"] for c in b._pending_corrections] == ["the year is 2017"]
    assert 25 < time.monotonic() - b._pending_corrections[0]["ts"] < 120


@pytest.mark.asyncio
async def test_an_owners_late_correction_is_bound_to_its_answer(agent, monkeypatch):
    """Fails where the owner's correction carried only the first-message
    fingerprint, so chat A's banner opened chat B ("hello ghost" twice)."""
    from ghost_agent.core.verifier import VerifyVerdict
    from tests.test_critic_async import _final, _make_verifier, _verdict
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    verifier, _ = _make_verifier([_verdict(VerifyVerdict.REFUTED, conf=0.97, issues=["the year is 2017"])])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch"}'}}]}}]},
        _final("It launched in 2016.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "hello ghost, when did it launch?"}]},
                                background_tasks=MagicMock())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if agent._pending_corrections:
            break
    assert agent._pending_corrections and all("|r" in c["conv"] for c in agent._pending_corrections)


# ── the review of §4LK ───────────────────────────────────────────────────────
def test_a_delegate_neither_loads_nor_saves_the_owners_corrections(tmp_path):
    """Fails where a sub-agent (same memory_dir) loaded the owner's queue and
    saved a stale copy back — a shown correction returned after a restart."""
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context
    (tmp_path / "pending_corrections.json").write_text(json.dumps(
        [{"note": "OWNER", "conv": "fp|rabc", "traj": "t", "wall": time.time()}]))
    ctx = make_context()
    ctx.memory_dir = tmp_path
    ctx.owner_memory_isolated = True
    d = GhostAgent(ctx)
    assert d._pending_corrections == []
    d._pending_corrections = [{"note": "STALE", "conv": "x", "ts": time.monotonic()}]
    d._save_pending_corrections()
    assert "OWNER" in (tmp_path / "pending_corrections.json").read_text()


@pytest.mark.parametrize("text", [
    "I'll double-check, but you should be fine to upgrade now.",
    "Let me confirm, but yes, you can delete the old branch.",
    "I'll check — you're right that the port is wrong.",
    "Let me check that for you — should I also update the lockfile?",
    "Θα το ελέγξω, αλλά μπορείς να το αναβαθμίσεις χωρίς πρόβλημα.",
    "Ας το ελέγξω, αλλά ναι, μπορείτε να διαγράψετε το παλιό branch.",
])
def test_an_answer_addressed_to_the_user_is_not_narration_even_with_a_call(text):
    from ghost_agent.core.reply_smoothing import narration_only
    assert narration_only(text, tool_attempt=True) is False


_ANS = "It launched in 2016, per the site."


@pytest.mark.parametrize("queued_from,history_has", [
    (_ANS, "Which version do you mean?\n\n" + _ANS),                                  # a clarify head before it
    (_ANS, _ANS + "\n\n---\n*Not run, as you asked — the change is untested.*"),     # a note after it
    ("![chart](/api/download/gen_1.png)\n\n" + _ANS, _ANS),                          # Slack drops the image
    ("⚠️ **Correction to my previous answer:** x\n\n---\n\n" + _ANS,
     ":warning: *Correction to my previous answer:* x\n\n---\n\n" + _ANS),          # Slack's shortcode banner
    (_ANS + " :tada:", _ANS),                                                          # an emoji shortcode
])
def test_a_correction_finds_its_answer_through_what_is_added_around_it(agent, queued_from, history_has):
    """Fails where the tag was an exact hash of the shipped opening: a header
    before the answer, a note after it, a Slack-dropped image or a shortcode
    banner meant the correction never surfaced."""
    from ghost_agent.core.agent import _reply_tag
    first = [{"role": "user", "content": "when?"}]
    fp = agent._conversation_fingerprint(first)
    agent._pending_corrections = [{"note": "the year is 2017", "ts": time.monotonic(), "traj": "t",
                                   "conv": f"{fp}|r{_reply_tag(queued_from)}"}]
    hist = first + [{"role": "assistant", "content": history_has}, {"role": "user", "content": "sure?"}]
    agent._consume_pending_corrections(hist, conv_fp=fp)
    assert "the year is 2017" in agent._take_active_correction()


def test_reset_all_clears_the_correction_queue(tmp_path):
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context
    ctx = make_context()
    ctx.memory_dir = tmp_path
    a = GhostAgent(ctx)
    a._pending_corrections = [{"note": "x", "conv": "fp|rabc", "ts": time.monotonic()}]
    a._save_pending_corrections()
    a.clear_pending_corrections()
    assert a._pending_corrections == [] and json.loads((tmp_path / "pending_corrections.json").read_text()) == []



def test_reset_all_through_the_tool_clears_the_agents_queue(tmp_path):
    import ghost_agent.tools.memory as M
    from ghost_agent.utils.logging import request_id_context
    owner = MagicMock()
    vec = MagicMock()
    vec.collection.count.return_value = 1
    vec.collection.get.return_value = {"ids": ["a"], "metadatas": [{}]}
    vec.library_file = None
    kb = lambda **kw: asyncio.run(M.tool_knowledge_base(memory_system=vec, graph_memory=MagicMock(),
                                                        owner_agent=owner, **kw))
    t = request_id_context.set("req-preview")
    try:
        tok = str(kb(action="reset_all")).split("confirm='")[1].split("'")[0]
    finally:
        request_id_context.reset(t)
    t = request_id_context.set("req-next")
    try:
        kb(action="reset_all", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert owner.clear_pending_corrections.called


def test_the_registry_hands_the_agent_to_knowledge_base(monkeypatch):
    import ghost_agent.tools.registry as R
    from tests.helpers import make_context
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return "ok"
    monkeypatch.setattr(R, "tool_knowledge_base", fake)
    ctx = make_context()
    ctx.agent = object()
    ctx.tor_proxy = None
    tools = R.get_available_tools(ctx)
    asyncio.run(tools["knowledge_base"](action="list_docs"))
    assert seen.get("owner_agent") is ctx.agent
