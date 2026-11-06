"""§4KD (2026-09-24): clarify-first after a costly tool turn; the seed line is
information; Slack never teaches.

Source: the 2026-09-23 Slack open-channel diagnosis — a member's `emp1` after
an image turn produced a SECOND image (the tool result ended with a ready
next action and the generic "if you lack information, ASK" prompt line lost),
and `skills_playbook.json` had ingested lessons whose task IS a stranger's
Slack prompt. Every test names the world it fails in.
"""
import ast
import asyncio
import base64
import inspect
import json
import struct
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import (COSTLY_TOOLS, _clarify_first_block,
                                    _is_low_information_message,
                                    _previous_turn_asked_a_question,
                                    turn_may_teach)
from ghost_agent.utils.logging import (requester_role_context,
                                       request_id_context)

REPO = Path(__file__).resolve().parents[1]
BOT_PATH = REPO / "interface" / "externals" / "slack_bot" / "main.py"
from tests.test_4kd_image_pipeline_review import _png_bytes   # the seed-line pin's fixture (shared)


# ---------------------------------------------------------------------------
# 1. Clarify-first: the pure rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,low", [
    ("emp1", True), ("?", True), ("...", True), ("ok", True), ("k", True), ("", True),
    ("   ", True), ("!!!!", True), ("123 456", True), ("ok ok", True),
    ("make it blue", False), ("another one please", False),
    ("κάνε το μπλε", False),                      # Greek is a word too
    ("no, the cat", False),
])
def test_low_information_messages(text, low):
    assert _is_low_information_message(text) is low, text


def _hist(prev_assistant, user="emp1"):
    return [{"role": "user", "content": "draw a bakery"},
            {"role": "assistant", "content": prev_assistant},
            {"role": "user", "content": user}]


IMG_REPLY = "![generated image](/api/download/gen_1a2b3c4d.png)\n\nA warm bakery at dusk."


def test_a_one_token_follow_up_after_an_image_turn_blocks_the_image_tool():
    """The live case. Fails in the world where the guard does not exist."""
    msgs = _hist(IMG_REPLY)
    block = _clarify_first_block("image_generation", "emp1", msgs, costly_record=False)
    assert block and "clarify first" in block and "Nothing was generated" in block


def test_an_intelligible_follow_up_is_not_blocked():
    assert _clarify_first_block("image_generation", "make the sign say CLOSED",
                                _hist(IMG_REPLY, "make the sign say CLOSED"), False) is None


def test_an_answer_to_our_own_question_is_not_blocked():
    """'ok' after we asked 'shall I render it?' is consent, not noise — even
    right after an image turn (the case the exemption decides; without the
    image the costly check would clear it anyway, and a mutant that deleted
    the exemption survived the first version of this pin)."""
    msgs = _hist(IMG_REPLY + "\n\nWant another take at a different angle?", "ok")
    assert _previous_turn_asked_a_question(msgs)
    assert _clarify_first_block("image_generation", "ok", msgs, costly_record=True) is None
    # and with no question the same "ok" is blocked
    assert _clarify_first_block("image_generation", "ok", _hist(IMG_REPLY, "ok"), True)


def test_only_costly_tools_are_guarded_and_only_after_a_costly_turn():
    assert _clarify_first_block("file_system", "emp1", _hist(IMG_REPLY), False) is None
    assert _clarify_first_block("image_generation", "emp1",
                                _hist("Here is the recipe you asked for."), False) is None
    # the in-process record stands in for a client that resends text only
    assert _clarify_first_block("image_generation", "emp1",
                                _hist("Here is the recipe you asked for."), True)
    assert "image_generation" in COSTLY_TOOLS


def test_the_current_user_message_is_not_mistaken_for_the_previous_assistant_turn():
    """The loop appends tool rows after the current user message; the guard
    must read the assistant message BEFORE the current user turn."""
    msgs = _hist(IMG_REPLY) + [{"role": "tool", "content": "SUCCESS: Image generated …"}]
    assert _clarify_first_block("image_generation", "emp1", msgs, False)
    assert agent_mod._last_assistant_text([{"role": "user", "content": "hi"}]) == ""


# ---------------------------------------------------------------------------
# 2. Clarify-first: the wiring (AST — a correct rule that is unfed is the
#    §4KB lesson; source-text pins are rejected by the ratchet)
# ---------------------------------------------------------------------------

def _func(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _dispatch_tree():
    return _func(ast.parse(inspect.getsource(agent_mod)), "_dispatch_and_process_tool_batch")


def test_the_dispatch_loop_calls_the_guard_with_the_users_message_and_the_history():
    tree = _dispatch_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_clarify_first_block"]
    assert len(calls) == 1, "the guard is called exactly once per tool call"
    args = [getattr(a, "id", None) for a in calls[0].args]
    assert args[1] == "last_user_content" and args[2] == "messages", args
    # its refusal is a REJECTED outcome with the reason code…
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "clarify_first" in consts
    # …and NOT a strike: the block that asks is the correct turn (live probe:
    # a strike booked the clarification `failed · 0.85`)
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and any(isinstance(c, ast.Name) and c.id == "_cf_block"
                                         for c in ast.walk(n.test)):
            struck = [c for c in ast.walk(n) if isinstance(c, ast.Call)
                      and getattr(c.func, "id", "") == "_strike_synthetic"]
            assert not struck, "the clarify-first block strikes the turn"


def test_finalize_records_a_costly_turn():
    """The costly record is written from the tools that ran. (§4NM: the
    `post_mortem` journal item this gate once guarded is gone — nothing in
    `_finalize_and_return` enqueues one any more.)"""
    tree = _func(ast.parse(inspect.getsource(agent_mod)), "_finalize_and_return")
    assert not any(isinstance(c, ast.Constant) and c.value == "post_mortem" for c in ast.walk(tree))
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_note_costly_turn"
               for n in ast.walk(tree))


def test_the_costly_record_is_written_for_real_calls_only_and_is_bounded():
    a = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    a._costly_turns = {}
    assert a._note_costly_turn("conv-1", [{"name": "image_generation", "content": "SUCCESS: …"}]) is True
    assert "conv-1" in a._costly_turns
    assert a._note_costly_turn("conv-2", [{"name": "image_generation", "_synthetic": True}]) is False
    assert a._note_costly_turn("conv-3", [{"name": "file_system", "content": "x"}]) is False
    assert a._note_costly_turn("", [{"name": "image_generation"}]) is False
    for i in range(agent_mod._COSTLY_TURNS_MAX + 5):
        a._note_costly_turn(f"c{i}", [{"name": "image_generation"}])
    assert len(a._costly_turns) <= agent_mod._COSTLY_TURNS_MAX + 1


# ---------------------------------------------------------------------------
# 3. A member never teaches (de-Slacked 2026-09-24 evening: the agent knows
#    "owner" and "member", never a client's name — the role header is the
#    only multi-user signal, and no request-id prefix carries meaning)
# ---------------------------------------------------------------------------

def _as(role):
    return requester_role_context.set(role)


def test_turn_may_teach_by_population():
    ctx = MagicMock()
    ctx.turn_origin_label = None
    tok = request_id_context.set("slack-1a2b3c4d"); rt = _as("member")
    try:
        assert turn_may_teach(ctx) is False
        # …and the population is still USER: a member's turn is real traffic
        assert agent_mod.turn_origin(ctx) == "user"
    finally:
        requester_role_context.reset(rt); request_id_context.reset(tok)
    # the same id with NO role is the owner's: the prefix means nothing
    tok = request_id_context.set("slack-1a2b3c4d")
    try:
        assert turn_may_teach(ctx) is True
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("1a2b3c4d")
    try:
        assert turn_may_teach(ctx) is True
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("probe-1a2b3c4d")
    try:
        assert turn_may_teach(ctx) is False
    finally:
        request_id_context.reset(tok)


def _playbook(tmp_path, lessons):
    (tmp_path / "skills_playbook.json").write_text(json.dumps(lessons))
    return SkillMemory(tmp_path)


def test_the_write_origin_derivation_names_the_member():
    from ghost_agent.memory import population as P
    tok = request_id_context.set("web-1a2b3c4d"); rt = _as("member")
    try:
        assert P.write_origin() == P.ORIGIN_MEMBER
    finally:
        requester_role_context.reset(rt); request_id_context.reset(tok)
    tok = request_id_context.set("slack-1a2b3c4d")          # a prefix alone is nobody
    try:
        assert P.write_origin() == P.ORIGIN_USER
    finally:
        request_id_context.reset(tok)


def test_the_bot_mints_its_own_request_id():
    """AST pin on the bot's request-id minting: the id is the bot's, for its
    feedback correlation — it carries no meaning in the agent."""
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
    mints = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") == "request_id" for t in n.targets)
             and isinstance(n.value, ast.BinOp)
             and isinstance(n.value.left, ast.Constant) and n.value.left.value == "slack-"]
    assert mints, "the bot does not mint a slack- prefixed request id"


def test_no_client_name_keys_a_rule_in_the_agent():
    """The agent's src carries no Slack request-id predicate, prefix or
    origin constant any more: the role header is the only multi-user
    signal (2026-09-24)."""
    import ghost_agent.utils.logging as lg
    for name in ("is_slack_request_id", "SLACK_REQUEST_PREFIX", "ORIGIN_SLACK"):
        assert not hasattr(lg, name), name
    from ghost_agent.memory import population as P
    assert not hasattr(P, "ORIGIN_SLACK")


def test_the_seed_line_states_a_record_and_suggests_nothing(tmp_path):
    """Fails in the world where the result ends with 'Reuse seed=N … for
    ANOTHER TAKE' — the ready next action an unintelligible follow-up
    resolved into."""
    from ghost_agent.tools.image_gen import tool_generate_image
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]

    async def gen(payload):
        return {"data": [{"b64_json": base64.b64encode(_png_bytes(8, 8)).decode()}],
                "seed": 123456, "width": 8, "height": 8}
    llm.generate_image = gen
    out = asyncio.run(tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path))
    assert out.startswith("SUCCESS:")
    assert "Seed 123456" in out                        # the record is kept
    low = out.lower()
    for verb in ("reuse seed", "another take", "tweaked prompt", "seed=123456"):
        assert verb not in low, verb
    assert "this request is complete" in low


def test_the_seed_advice_lives_in_the_description_where_actions_are_chosen():
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = ["http://gpu"]
    t = next(x for x in get_active_tool_definitions(ctx)
             if x.get("function", {}).get("name") == "image_generation")
    d = t["function"]["parameters"]["properties"]["seed"]["description"]
    assert "Never re-run the tool because a result mentioned a seed" in d
    assert "reference_images" in d and "different composition" in d
