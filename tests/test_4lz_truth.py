"""§4LZ lens B — truthfulness: behaviour pins."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome


# ── B1: a forget PREVIEW is never "already applied"; duplicate blocks are budgeted ──

async def test_a_repeated_forget_preview_is_not_blocked_as_applied():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    calls = []

    async def kb(**kw):
        calls.append(kw)
        return "PREVIEW — nothing was deleted. 1. document x"
    agent.available_tools = {"knowledge_base": kb}
    args = {"action": "forget", "target": "x"}
    ts = H._ts([("knowledge_base", args), ("knowledge_base", dict(args))], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert len(calls) == 2


async def test_duplicate_setter_blocks_end_the_turn_on_the_second():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()

    async def upd(**kw):
        return "Profile updated."
    agent.available_tools = {"update_profile": upd}
    args = {"category": "a", "key": "b", "value": "c"}
    ts = H._ts([("update_profile", args), ("update_profile", args), ("update_profile", args)],
               StrikeLedger(), set())
    ts.executed_idempotent = set()
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.force_final_response is True


# ── B2: only a notification that went out counts as delivered ──

@pytest.mark.parametrize("row,delivered", [
    ({"name": "notify_operator", "content": ToolOutcome.ok("Notification sent.")}, True),
    ({"name": "notify_operator", "content": "Notification sent."}, True),
    ({"name": "notify_operator", "content": "SYSTEM BLOCK: preflight", "_synthetic": True}, False),
    ({"name": "notify_operator", "content": "Notification sent.", "_synthetic": True}, False),
    ({"name": "notify_operator", "content": ToolOutcome.rejected("disabled tool")}, False),
    ({"name": "notify_operator", "content": ToolOutcome.ok("PROBE — not sent. Would have sent: x")}, False),
    ({"name": "notify_operator", "content": "Error: rate limit"}, False),
    ({"name": "web_search", "content": "ok"}, False),
])
def test_notify_delivered(row, delivered):
    from ghost_agent.core.agent import _notify_delivered
    assert _notify_delivered(row) is delivered


# ── B3: content that mentions exceptions is not a failure ──

@pytest.mark.parametrize("tool", ["web_search", "recall", "knowledge_base", "deep_research"])
def test_search_content_mentioning_exceptions_is_not_an_error(tool):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert not _looks_like_tool_error("### 1. Exceptional performance of Python exception handling\n...", tool)


def test_a_declared_success_is_believed():
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert not _looks_like_tool_error(ToolOutcome.ok("Traceback mentioned in the doc"), "introspect")
    assert _looks_like_tool_error(ToolOutcome.failed("boom"), "introspect")


# ── B4: an incomplete report is partial ──

async def test_a_report_with_missing_sources_is_partial(tmp_path):
    pytest.importorskip("fitz")
    from ghost_agent.tools.report_pdf import tool_generate_pdf
    (tmp_path / "a.md").write_text("# A\nhello")
    out = await tool_generate_pdf(title="R", sections=[{"heading": "H", "body": "text"}],
                                  sandbox_dir=tmp_path, source_files=["a.md", "missing1.md", "missing2.md"])
    assert getattr(out, "status", None) is OutcomeStatus.PARTIAL and str(out).startswith("PARTIAL")


# (B5 — a refused macro step is not a succeeded step — went with the macro engine, §4TA)


# ── B7: the unattended advancer reads failure heads the shared way ──

@pytest.mark.parametrize("text", ["CRITICAL ERROR: x", "SYSTEM ERROR: y", "Security Error: z",
                                  "REJECTED: no"])
def test_the_advancer_sees_every_failure_head(text):
    from ghost_agent.core.project_advancer import _looks_like_failure
    assert _looks_like_failure(text)


# ── B8: exception arms and unavailable refusals are declared ──

async def test_an_unavailable_mood_write_is_rejected():
    """§4TB: was `self_state` (removed); its one surviving write is
    `introspect(action="set_mood")`, and with no self-model it is REJECTED,
    not booked OK."""
    from ghost_agent.tools.introspect import tool_introspect
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("web-4lz-owner")           # §4TC (review): the write is the owner's own request's
    try:
        out = await tool_introspect(action="set_mood", mood="calm", self_model=None)
    finally:
        request_id_context.reset(tok)
    assert getattr(out, "status", None) is OutcomeStatus.REJECTED
    assert getattr(out, "reason_code", None) == "tool_unavailable"
    # … and outside the owner's own request it is REJECTED too, one step earlier
    out = await tool_introspect(action="set_mood", mood="calm", self_model=None)
    assert getattr(out, "status", None) is OutcomeStatus.REJECTED
    assert getattr(out, "reason_code", None) == "owner_only"
