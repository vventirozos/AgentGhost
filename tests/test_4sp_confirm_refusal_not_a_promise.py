"""§4SP (req a62cf750): a write refused with "ask the user to confirm in a new
message" (reason_code ``untrusted_content``) makes "Reply yes and I'll create
it" the correct closing sentence. The trailing-promise guard steered "DO it
NOW", the model re-sent the 6k-char create_skill and the same guard refused
it again (+146 s, turn booked failed).

Review r1 MAJOR: the first cut exempted on ANY such row of the request — an
early refusal silenced the guard for the rest of it, and the usual owner-fact
refusal (remember/update_profile, same code: "they will say so") asks for no
new message. Now: the LATEST row only, carrying one of the refusal's own
new-message instructions. r2: the mid-turn owner-fact refusal ("send it again
as a new message") counts; the file deletion refusal keeps its instruction."""
from ghost_agent.core.agent import _awaits_new_message_confirm, _ends_with_action_promise
from ghost_agent.tools.outcome import ToolOutcome
from ghost_agent.utils.provenance import content_refusal, owner_fact_refusal

SRC = ["darkweb_search", "execute"]


def _refusal():
    return ToolOutcome.rejected(content_refusal("the skill", SRC), world_changed=False,
                                reason_code="untrusted_content")


def _owner_fact():
    return ToolOutcome.rejected(owner_fact_refusal("the fact", SRC), world_changed=False,
                                reason_code="untrusted_content")


def _row(outcome, name="create_skill"):
    return {"role": "tool", "tool_call_id": "t1", "name": name, "content": outcome}


def test_a_latest_new_message_refusal_awaits_the_user():
    assert _awaits_new_message_confirm([_row(ToolOutcome.ok("200"), "execute"), _row(_refusal())]) is True


def test_an_early_refusal_followed_by_other_work_does_not_exempt():
    assert _awaits_new_message_confirm([_row(_refusal()), _row(ToolOutcome.ok("200"), "execute")]) is False


def test_an_owner_fact_refusal_does_not_exempt():
    """Same reason code, but its text asks for no new message."""
    assert _awaits_new_message_confirm([_row(_owner_fact(), "remember")]) is False


def test_a_mid_turn_owner_fact_refusal_asks_for_a_new_message_and_exempts():
    """r2: "ask them to send it again as a new message" is the same designed stop."""
    out = ToolOutcome.rejected(owner_fact_refusal("the fact", SRC, mid_turn=True), world_changed=False,
                               reason_code="untrusted_content")
    assert _awaits_new_message_confirm([_row(out, "update_profile")]) is True


def test_the_file_deletion_refusal_keeps_its_new_message_instruction(tmp_path):
    """r2: the deletion sink rewrites content_refusal's text — through the real sink."""
    from ghost_agent.tools import file_system as FS
    from ghost_agent.utils import provenance as PV
    from ghost_agent.utils.logging import request_id_context
    (tmp_path / "uploads").mkdir()
    tok = request_id_context.set("req-4sp-del")
    try:
        PV.note_untrusted_content("browser", req_id="req-4sp-del")
        out = FS._protected_dir_refusal(tmp_path, tmp_path / "uploads", "uploads")
    finally:
        request_id_context.reset(tok)
    assert getattr(out, "reason_code", None) == "untrusted_content" and "was not done" in str(out)
    assert _awaits_new_message_confirm([_row(out, "file_system")]) is True


def test_the_instruction_alone_without_the_code_does_not_exempt():
    out = ToolOutcome.failed(content_refusal("the skill", SRC), world_changed=False, reason_code="raised")
    assert _awaits_new_message_confirm([_row(out)]) is False


def test_other_refusals_and_successes_do_not_exempt():
    rows = [_row(ToolOutcome.ok("done"), "execute"),
            _row(ToolOutcome.rejected("blocked", reason_code="untrusted_egress"), "browser"),
            _row(ToolOutcome.failed("boom", world_changed=False, reason_code="raised"))]
    assert _awaits_new_message_confirm(rows) is False


def test_plain_string_and_odd_rows_are_safe():
    assert _awaits_new_message_confirm(None) is False
    assert _awaits_new_message_confirm([]) is False
    assert _awaits_new_message_confirm([{"content": "Error: x"}]) is False
    assert _awaits_new_message_confirm(["junk"]) is False


def test_the_live_closing_sentence_still_reads_as_a_promise():
    # the detector is unchanged — the exemption lives at the guard
    reply = ("The skill was not created yet.\n\n"
             "**Reply \"yes\" (or anything confirming) and I'll create it.**")
    assert _ends_with_action_promise(reply)


CONFIRM_ENDING = ("The skill was not created: the creation guard needs your confirmation in a new message.\n\n"
                  "Reply \"yes\" and I'll create it.")
UNRELATED_PROMISE = "The skill was not created. I found the cause in the config. I'll fix the config next."


async def _drive_with(monkeypatch, tmp_path, results, script, tool="create_skill"):
    """``results``: what each call of the one tool returns, in order."""
    from unittest.mock import AsyncMock
    from tests import test_4kv_slack12_fixes as K
    from tests.test_4kw_announced_work_and_outcome import _drive
    monkeypatch.setattr(K, "AsyncMock", lambda return_value=None: AsyncMock(side_effect=list(results)))
    return await _drive(monkeypatch, tmp_path, script,
                        request="create a new skill called torportal_news", tools=(tool,), role="owner")


def _one_call(ending, tool="create_skill"):
    return [("call", tool, {"name": "torportal_news"}), ("say", ending), ("say", "x")]


def _steered(logged):
    return any(t == "Pending-Promise Guard" for t, _, _ in logged)


async def test_a_confirm_in_a_new_message_refusal_is_not_steered(monkeypatch, tmp_path):
    """The live shape: refused create_skill, then the "reply yes" ending ships
    as is — one generation after the call, no act-or-admit steer."""
    final, model, logged, _, _ = await _drive_with(monkeypatch, tmp_path, [_refusal()], _one_call(CONFIRM_ENDING))
    assert not _steered(logged)
    assert len(model.payloads) == 2
    assert "I'll create it" in final


async def test_the_same_ending_after_an_ordinary_result_is_still_steered(monkeypatch, tmp_path):
    """Control: the guard is untouched when nothing asked for a new message."""
    _, model, logged, _, _ = await _drive_with(monkeypatch, tmp_path, [ToolOutcome.ok("skill draft checked")],
                                               _one_call(CONFIRM_ENDING))
    assert _steered(logged) and len(model.payloads) == 3


async def test_an_early_refusal_then_work_then_an_unrelated_promise_is_steered(monkeypatch, tmp_path):
    """r1 MAJOR: one early refusal no longer silences the guard for the rest of the request."""
    script = [("call", "create_skill", {"name": "a"}), ("call", "create_skill", {"name": "b"}),
              ("say", UNRELATED_PROMISE), ("say", "The config was NOT fixed."), ("say", "x")]
    _, model, logged, _, _ = await _drive_with(monkeypatch, tmp_path, [_refusal(), ToolOutcome.ok("checked")], script)
    assert _steered(logged)


async def test_an_owner_fact_refusal_keeps_the_guard(monkeypatch, tmp_path):
    """r1 MAJOR: remember/update_profile refusals share the code but ask for no new message."""
    _, model, logged, _, _ = await _drive_with(monkeypatch, tmp_path, [_owner_fact()],
                                               _one_call(UNRELATED_PROMISE, tool="remember"), tool="remember")
    assert _steered(logged)
