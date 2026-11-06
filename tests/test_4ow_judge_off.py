"""§4OW (2026-10-14, operator "turn off the judge, keep the mechanical
checks"): the LLM judge — verify_claim / verify_code_output and all inside
them (claim binder, main-model escalation, votes) — is off by default; the
mechanical arms still run. §4OV measured 9 of 17 real judge refutes false."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


@pytest.fixture(autouse=True)
def _judge_default(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFIER_JUDGE", raising=False)


@pytest.mark.parametrize("raw,on", [(None, False), ("0", False), ("1", True), ("on", True), ("x", False)])
def test_the_switch(monkeypatch, raw, on):
    if raw is not None:
        monkeypatch.setenv("GHOST_VERIFIER_JUDGE", raw)
    assert A.verifier_judge_enabled() is on


def _agent(tmp_path, web=None):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.sandbox_dir = str(tmp_path)
    agent.available_tools = {}
    agent._is_strict_trivial_chat = lambda lc: False
    agent._scoped_sandbox_for = lambda pid: tmp_path
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    verifier.verify_claim = AsyncMock(return_value=VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="judge"))
    verifier.verify_code_output = AsyncMock(return_value=VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="judge"))
    agent.context.verifier = verifier
    agent._execute_web_artifact = AsyncMock(return_value=web)
    return agent, verifier


TOOLS = [{"role": "tool", "name": "file_system", "content": "SUCCESS: Wrote 1000 chars to 'index.html'."}]


async def _run(agent, tools=TOOLS, reply="Done! The game is ready.", request="build me a web game"):
    return await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=[{"role": "user", "content": request}],
        final_ai_content=reply, last_user_content=request, lc=request)


async def test_no_judge_call_and_no_verdict_without_a_mechanical_finding(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    agent, verifier = _agent(tmp_path, web=("index.html", ""))
    v, _ = await _run(agent)
    verifier.verify_claim.assert_not_awaited()
    verifier.verify_code_output.assert_not_awaited()
    assert v is None


async def test_a_page_that_throws_is_still_refuted(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    agent, verifier = _agent(tmp_path, web=("index.html", "UNCAUGHT JS EXCEPTIONS (1)\n  • SyntaxError: x"))
    v, _ = await _run(agent)
    verifier.verify_claim.assert_not_awaited()
    assert v is not None and v.verdict == VerifyVerdict.REFUTED and "SyntaxError" in v.reasoning
    assert A.refute_is_grounded(v)
    assert any("SyntaxError: x" in i for i in v.issues)          # r1: the repair round reads the error


async def test_a_code_turn_skips_the_code_judge_too(tmp_path):
    agent, verifier = _agent(tmp_path)
    tools = [{"role": "tool", "name": "execute", "content": "EXIT CODE: 0\nhello"}]
    msgs = [{"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "execute", "arguments": '{"command": "echo hello"}'}}]}]
    v, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=msgs, final_ai_content="It printed hello.",
        last_user_content="run echo hello", lc="run echo hello")
    verifier.verify_code_output.assert_not_awaited()
    verifier.verify_claim.assert_not_awaited()
    assert v is None


async def test_the_switch_brings_the_judge_back(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_VERIFIER_JUDGE", "1")
    (tmp_path / "index.html").write_text("<html></html>")
    agent, verifier = _agent(tmp_path, web=("index.html", ""))
    v, _ = await _run(agent)
    verifier.verify_claim.assert_awaited_once()
    assert v is not None and v.reasoning == "judge"


async def test_no_evidence_pack_is_built_for_a_judge_that_is_off(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(A, "_collect_verifier_evidence", lambda *a, **k: built.append(1) or "x")
    monkeypatch.setattr(A, "_written_documents_for_audit", lambda *a, **k: built.append(2) or "")
    (tmp_path / "index.html").write_text("<html></html>")
    agent, _ = _agent(tmp_path, web=("index.html", ""))
    await _run(agent)
    assert built == []


def test_an_empty_late_verdict_is_quiet_with_the_judge_off(monkeypatch):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.verifier.llm_client = MagicMock()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda t, m, **k: lines.append((m, k.get("level", "INFO"))))
    agent._record_late_verdict(None, "tid", last_tool={"name": "file_system"})
    assert lines == []
    monkeypatch.setenv("GHOST_VERIFIER_JUDGE", "1")
    agent._record_late_verdict(None, "tid", last_tool={"name": "file_system"})
    assert lines and lines[-1][1] == "WARNING"


async def test_a_missing_written_file_is_still_refuted(tmp_path):
    agent, verifier = _agent(tmp_path, web=None)
    tools = [{"role": "tool", "name": "file_system", "content": "SUCCESS: Wrote 900 chars to 'report.md'."}]
    v, _ = await _run(agent, tools=tools, reply="The report is in report.md.", request="write a report to report.md")
    verifier.verify_claim.assert_not_awaited()
    assert v is not None and v.verdict == VerifyVerdict.REFUTED
    assert "file-artifact" in str(getattr(v, "override", ""))


# ── r1 review ──────────────────────────────────────────────────────────

async def test_no_depth_routing_without_a_judge(tmp_path):
    # §4PJ: the low-confidence depth trigger went with the confidence score,
    # so no path asks for depth at all — judge on or off.
    assert not hasattr(GhostAgent, "_verify_depth_for_turn")
    from tests import pin_ast as P
    t = P.tree(GhostAgent._compute_verifier_verdict)
    assert P.assigned_values(t, "_deep") == ["False"]


# ── §4OX: the mechanical repair window, and lines that say what runs ──

@pytest.mark.parametrize("raw,want", [(None, 15.0), ("", 15.0), ("8", 8.0), ("0", 0.0), ("-3", 0.0),
                                      ("500", 60.0), ("nan", 15.0), ("x", 15.0)])
def test_the_mechanical_budget(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("GHOST_MECHANICAL_REPAIR_BUDGET", raising=False)
    else:
        monkeypatch.setenv("GHOST_MECHANICAL_REPAIR_BUDGET", raw)
    assert A.mechanical_repair_budget_s() == want


@pytest.mark.parametrize("judge,critic_budget,want", [(None, "0", 15.0), (None, "65", 15.0), ("1", "0", 0.0),
                                                      ("1", "25", 25.0)])
def test_the_repair_await_takes_the_mechanical_window_when_the_judge_is_off(monkeypatch, judge, critic_budget, want):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", critic_budget)
    monkeypatch.delenv("GHOST_MECHANICAL_REPAIR_BUDGET", raising=False)
    if judge:
        monkeypatch.setenv("GHOST_VERIFIER_JUDGE", judge)
    agent = GhostAgent.__new__(GhostAgent)
    assert agent._critic_repair_await_budget() == want


def test_no_async_no_await(monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    assert GhostAgent.__new__(GhostAgent)._critic_repair_await_budget() == 0.0


def test_the_finalize_line_says_what_runs(monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent = GhostAgent.__new__(GhostAgent)
    agent._late_verdict_running_map = {"t1": 1}
    kind, text = agent._no_verdict_reason("t1")
    assert kind == "deferred" and "mechanical checks running" in text and "LLM judge is off" in text
    kind, text = agent._no_verdict_reason("t-none")
    assert kind == "empty" and "mechanical checks found nothing" in text


def test_the_judge_on_keeps_the_old_wording(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFIER_JUDGE", "1")
    agent = GhostAgent.__new__(GhostAgent)
    agent._late_verdict_running_map = {"t1": 1}
    assert "verifying asynchronously" in agent._no_verdict_reason("t1")[1]


def test_liveness_counts_mechanical_lines(tmp_path):
    import time as _t
    from ghost_agent.core import liveness as LV
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    (tmp_path / "system").mkdir()
    now = _t.strftime("%Y-%m-%d %H:%M:%S")
    (tmp_path / "system" / "ghost-agent.log").write_text(
        f"{now} - GhostStream - INFO - [ab12] verifier — mechanical checks running after the reply\n"
        f"{now} - GhostStream - INFO - [SYSTEM] verifier — mechanical checks on (page load, written files)\n"
        f"{now} - GhostStream - INFO - [ab12] verifier — mechanical checks found nothing to refute\n"
        f"{now} - GhostStream - INFO - [ab12] verifier — FILE-ARTIFACT clean: 1 of 1 written file(s) read\n"
        f"{now} - GhostStream - DEBUG - [ab12] thinking — the verifier mechanical checks\n")
    r = probe.fn(tmp_path)
    assert getattr(r, "count", None) == 2, r


def test_the_boot_banner_names_the_mechanical_checks():
    from ghost_agent import main as M
    from tests import pin_ast as P
    t = P.tree(M)
    import ast
    ifs = [n for n in ast.walk(t) if isinstance(n, ast.If) and P.U(n.test) == "verifier_judge_enabled()"]
    assert ifs, "the boot banner must branch on the judge switch"
    off = " ".join(P.U(c.args[1]) for n in ifs for o in n.orelse for c in P.calls(o, "pretty_log") if len(c.args) > 1)
    on = " ".join(P.U(c.args[1]) for n in ifs for b in n.body for c in P.calls(b, "pretty_log") if len(c.args) > 1)
    assert "mechanical checks on" in off and "the LLM judge is off" in off
    assert "mechanical" not in on



@pytest.mark.parametrize("block,want", [
    ("UNCAUGHT JS EXCEPTIONS (2)\n  • SyntaxError: Unexpected number\n  • TypeError: x", "SyntaxError: Unexpected number"),
    ("UNCAUGHT JS EXCEPTIONS (1)", ""), ("", "")])
def test_first_js_error(block, want):
    assert A._first_js_error(block) == want


@pytest.mark.parametrize("override,only", [("visual", True), ("visual+image-unseen", True), ("", False),
                                           ("visual+file-artifact", False), ("web-exec", False)])
def test_image_only_refutes(override, only):
    from types import SimpleNamespace
    assert A._refute_is_image_only(SimpleNamespace(override=override)) is only


def test_the_repair_branch_leaves_images_and_dont_run_requests_to_the_late_path():
    from tests import pin_ast as P
    import ast
    t = P.tree(GhostAgent)
    tests = [P.U(n.test) for n in ast.walk(t) if isinstance(n, ast.If) and "_refute_is_image_only(_vr)" in P.U(n.test)]
    assert tests and "request_forbids_running(last_user_content)" in tests[0] and "_refuted and" in tests[0]



@pytest.mark.parametrize("judge,kind,level", [(None, "empty", "INFO"), ("1", "empty", "WARNING"),
                                              (None, "lost", "WARNING"), (None, "deferred", "INFO")])
def test_no_verdict_levels(monkeypatch, judge, kind, level):
    if judge:
        monkeypatch.setenv("GHOST_VERIFIER_JUDGE", judge)
    assert A._no_verdict_level(kind) == level


@pytest.mark.parametrize("text,forbids", [
    ("Do not open or test the page, just write it and reply with the path.", True),
    ("just write it, do not execute.", True),
    ("do not run or test it", True),
    ("don't test the app", True),
    ("don't forget to run the tests", False),
    ("don't hesitate to run it", False),
    ("do not run the migration on prod without a backup", False),
])
def test_dont_run_phrasings(text, forbids):
    from ghost_agent.utils.constraints import request_forbids_running
    assert request_forbids_running(text) is forbids
