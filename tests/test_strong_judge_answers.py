"""§4HO — the strong judge must ANSWER, and the digest must reach the source.

Measured on the Revolut series (2026-09-15/16): of 45 refute escalations,
17 came back "unavailable" — the strong judge (thinking ON) hit its
2,048-token cap with ZERO content (finish_reason=length), and the branch
was silent, so 17 cheap REFUTEs stood unchecked; replayed with thinking
OFF the same call answered CONFIRMED 0.85 in 3 s. The verdict digest
(4,000 chars / 5 items) omitted the tool that supported the claim on a
21-tool turn, and "78%" — the agent's OWN confidence score — was refuted
as a fabricated fact.

Pins (in the consumer's words):
  A. the last-resort MAIN call carries /no_think + enable_thinking=False
     when it is the strong judge (force_main, classic prompt) — and only
     then; GHOST_VERIFY_MAIN_NO_THINK=0 restores thinking.
  B. an empty main answer is LOUD (WARNING) and leaves its cause where the
     escalation site can read it.
  C. an escalation whose strong judge returned nothing says WHY at WARNING
     and writes the cause into the ledger row.
  D. the digest budget/items scale with the candidate count; the verdict
     site wires that rule to the packer; dead (BLOCKED) candidates never
     take a positional slot from a live one; verify_claim's cap IS the
     packer's max.
  E. the judge prompts tell the judge a stated confidence is an assessment.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
from unittest.mock import patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import verifier as vmod
from ghost_agent.core.verifier import (
    Verifier, VerifyResult, VerifyVerdict, _VERIFY_ADJUDICATE_PROMPT,
    _VERIFY_CLAIM_PROMPT, _VERIFY_ENUMERATE_PROMPT,
)


@pytest.fixture(autouse=True)
def _classic_flow(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    monkeypatch.delenv("GHOST_VERIFY_MAIN_NO_THINK", raising=False)


class _MainClient:
    """A main model that answers exactly as configured; records payloads."""
    def __init__(self, content="", finish="length", worker=False):
        self.worker_clients = [object()] if worker else []
        self.critic_clients = []
        self.content, self.finish = content, finish
        self.payloads = []

    async def chat_completion(self, payload, **_kw):
        self.payloads.append(copy.deepcopy(payload))
        return {"choices": [{"finish_reason": self.finish,
                             "message": {"content": self.content}}]}


_GOOD = json.dumps({"verdict": "CONFIRMED", "confidence": 0.85, "reasoning": "ok"})


def _res(verdict, conf=0.9, issues=None):
    return VerifyResult(verdict=verdict, confidence=conf,
                        reasoning="r", issues=list(issues or []))


# ── A. the strong judge is asked without thinking ──────────────────────
class TestStrongJudgeNoThink:
    @pytest.mark.asyncio
    async def test_force_main_classic_call_disables_thinking(self):
        c = _MainClient(_GOOD, "stop")
        v = Verifier(llm_client=c)
        out = await v._call_llm("judge this", force_main=True)
        assert out.get("verdict") == "CONFIRMED"
        p = c.payloads[-1]
        assert p["messages"][0]["content"].rstrip().endswith("/no_think")
        assert p.get("chat_template_kwargs") == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_a_non_strong_classic_call_keeps_thinking(self):
        c = _MainClient(_GOOD, "stop")
        v = Verifier(llm_client=c)
        await v._call_llm("judge this", force_main=False)
        p = c.payloads[-1]
        assert "/no_think" not in p["messages"][0]["content"]
        assert "chat_template_kwargs" not in p

    @pytest.mark.asyncio
    async def test_kill_switch_restores_thinking(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_MAIN_NO_THINK", "0")
        c = _MainClient(_GOOD, "stop")
        v = Verifier(llm_client=c)
        await v._call_llm("judge this", force_main=True)
        p = c.payloads[-1]
        assert "/no_think" not in p["messages"][0]["content"]
        assert "chat_template_kwargs" not in p


# ── B. an empty main answer is loud and leaves its cause ───────────────
class TestEmptyMainAnswer:
    @pytest.mark.asyncio
    async def test_empty_content_warns_and_stashes_the_cause(self):
        c = _MainClient("", "length")
        v = Verifier(llm_client=c)
        with patch("ghost_agent.utils.logging.pretty_log") as plog:
            out = await v._call_llm("judge this", force_main=True)
        assert out == {}
        assert v._last_main_call == {"finish_reason": "length", "content_chars": 0}
        warnings = [c_ for c_ in plog.call_args_list
                    if c_.kwargs.get("level") == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].args[1]
        assert "NO content" in msg and "finish_reason=length" in msg

    @pytest.mark.asyncio
    async def test_a_real_answer_does_not_warn(self):
        c = _MainClient(_GOOD, "stop")
        v = Verifier(llm_client=c)
        with patch("ghost_agent.utils.logging.pretty_log") as plog:
            await v._call_llm("judge this", force_main=True)
        assert not [c_ for c_ in plog.call_args_list
                    if c_.kwargs.get("level") == "WARNING"]
        assert v._last_main_call == {"finish_reason": "stop",
                                     "content_chars": len(_GOOD)}


# ── C. escalation with no strong verdict says why, in the ledger too ───
def _ledger(home):
    p = home / "system" / "verifier" / "escalations.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class TestEscalationUnavailableIsLoud:
    @pytest.mark.asyncio
    async def test_capped_strong_judge_names_its_cause(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        c = _MainClient("", "length", worker=True)   # cheap route exists
        v = Verifier(llm_client=c)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["fabricated 78%"])
        with patch("ghost_agent.utils.logging.pretty_log") as plog:
            out = await v._escalate_refute(cheap, "claim", "evidence", "ctx",
                                           trace={"req_id": "r1"})
        assert out is cheap                                   # refute stands
        rows = [r for r in _ledger(tmp_path) if r["outcome"] == "unavailable"]
        assert len(rows) == 1
        assert rows[0]["rebuttal"] == "strong_none:length:0c"
        msgs = [c_.args[1] for c_ in plog.call_args_list
                if c_.kwargs.get("level") == "WARNING"]
        assert any("strong judge returned no verdict (strong_none:length:0c)" in m
                   and "stands UNCHECKED" in m for m in msgs)
        # the strong call itself was asked without thinking
        assert c.payloads and c.payloads[-1].get("chat_template_kwargs") == {
            "enable_thinking": False}

    @pytest.mark.asyncio
    async def test_a_stale_cause_from_an_earlier_call_is_not_reported(
            self, tmp_path, monkeypatch):
        """The stash is reset before the strong call: a strong call that
        never reaches the main route (raises → the except branch) must not
        report the previous call's finish_reason."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        c = _MainClient("", "length", worker=True)
        v = Verifier(llm_client=c)
        v._last_main_call = {"finish_reason": "length", "content_chars": 0}

        async def _none(*_a, **_k):
            return {}
        monkeypatch.setattr(v, "_call_llm", _none)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["x"])
        with patch("ghost_agent.utils.logging.pretty_log"):
            await v._escalate_refute(cheap, "c", "e", "ctx", trace={"req_id": "r2"})
        rows = [r for r in _ledger(tmp_path) if r["outcome"] == "unavailable"]
        assert rows and rows[0]["rebuttal"] == "strong_none:unparsed:?c"

    @pytest.mark.asyncio
    async def test_an_answering_strong_judge_still_overturns(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        c = _MainClient(_GOOD, "stop", worker=True)
        v = Verifier(llm_client=c)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["fabricated 78%"])
        out = await v._escalate_refute(cheap, "claim", "evidence", "ctx",
                                       trace={"req_id": "r3"})
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert not [r for r in _ledger(tmp_path) if r["outcome"] == "unavailable"]


# ── D. the digest reaches the source ───────────────────────────────────
def _t(name, content):
    return {"name": name, "content": content}


class TestDigestBudget:
    @pytest.mark.parametrize("n,expected", [
        (1, (4000, 3)), (6, (4000, 3)), (7, (4200, 4)), (12, (7200, 4)),
        (13, (7800, 5)), (20, (12000, 5)), (21, (12000, 6)), (40, (12000, 6)),
    ])
    def test_budget_and_items_scale_with_the_candidate_count(self, n, expected):
        tools = [_t("web_search", f"### {i}. result {i} about revolut") for i in range(n)]
        assert A._evidence_budget_for(tools) == expected

    def test_bookkeeping_confirmations_are_not_candidates(self):
        tools = [_t("web_search", "### 1. result")] + [
            _t("manage_tasks", '{"ok": true}') for _ in range(30)]
        assert A._evidence_budget_for(tools) == (4000, 3)

    def test_the_claim_source_at_index_15_of_21_reaches_the_digest(self):
        """The recorded shape: 21 substantive tools, the supporting source
        six from the end, every output ~1,500 chars. FAILS IF: the caller
        packs with the old flat 4000/3 (or 5) — the pull then has no room
        beyond the newest slots."""
        filler = " ".join(f"coverage{i}" for i in range(220))       # ~1.5k, no claim tokens
        tools = [_t("web_search", f"### {i}. {filler}") for i in range(21)]
        tools[15] = _t("browser", "The sender domain was giustizia-cert.it per the "
                                  "screenshot; InfoCert PEC branch. " + filler)
        budget, items = A._evidence_budget_for(tools)
        ev = A._collect_verifier_evidence(tools, max_items=items, budget=budget,
                                          claim_text="the sender domain was "
                                          "giustizia-cert.it via the InfoCert PEC branch")
        assert "giustizia-cert.it" in ev
        assert ev.count("[web_search]") + ev.count("[browser]") >= 5
        flat = A._collect_verifier_evidence(tools, claim_text="the sender domain was "
                                            "giustizia-cert.it via the InfoCert PEC branch")
        assert flat.count("[web_search]") + flat.count("[browser]") < 5

    def test_the_verdict_site_packs_with_the_scaled_budget(self):
        """AST pin on the ONE verdict site: `_collect_verifier_evidence` is
        called with max_items/budget derived from `_evidence_budget_for`
        over the same tools list — not literals."""
        tree = ast.parse(inspect.getsource(A))
        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "_collect_verifier_evidence"
                 and any(k.arg == "max_items" for k in n.keywords)]
        assert len(sites) == 1
        kw = {k.arg: k.value for k in sites[0].keywords}
        assert isinstance(kw["max_items"], ast.Name)
        budget_names = {n.id for n in ast.walk(kw["budget"]) if isinstance(n, ast.Name)}
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == "_evidence_budget_for"]
        assert len(assigns) == 1
        tgt = assigns[0].targets[0]
        assert isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2
        b_name, i_name = tgt.elts[0].id, tgt.elts[1].id
        assert kw["max_items"].id == i_name
        assert b_name in budget_names
        # same tools list on both sides
        src_arg = assigns[0].value.args[0]
        assert isinstance(src_arg, ast.Name) and isinstance(sites[0].args[0], ast.Name)
        assert src_arg.id == sites[0].args[0].id


class TestDeadCandidates:
    def test_a_blocked_page_never_takes_a_positional_slot_from_a_live_one(self):
        tools = [_t("browser", "Live page A: the notification names the request number"),
                 _t("web_search", "### 1. Live result B about the breach coverage"),
                 _t("browser", "STATUS: BLOCKED — the site refused the fetch (403)")]
        ev = A._collect_verifier_evidence(tools, max_items=2)
        assert "Live page A" in ev and "Live result B" in ev
        assert "BLOCKED" not in ev

    def test_all_dead_still_packs_something(self):
        tools = [_t("browser", "STATUS: BLOCKED — refused (403)"),
                 _t("browser", "STATUS: BLOCKED — refused (429)")]
        ev = A._collect_verifier_evidence(tools, max_items=2)
        assert ev.count("BLOCKED") == 2

    def test_dead_is_a_head_property(self):
        live = _t("browser", "x" * 300 + " STATUS: BLOCKED mentioned in the body")
        assert A._evidence_is_dead(live) is False
        assert A._evidence_is_dead(_t("browser", "  STATUS: BLOCKED\nnothing")) is True


class TestVerifyClaimCap:
    @pytest.mark.asyncio
    async def test_the_cap_is_the_packers_max(self, monkeypatch):
        seen = {}

        async def _cap(prompt, *_a, **_k):
            seen["prompt"] = prompt
            return {"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "ok"}
        v = Verifier(llm_client=_MainClient(_GOOD, "stop"))
        monkeypatch.setattr(v, "_call_llm", _cap)
        mx = A._EVIDENCE_BUDGET_MAX
        assert mx > 4000
        evidence = "e" * (mx - 20) + " INSIDE-MARK " + "f" * 200 + " OUTSIDE-MARK"
        await v.verify_claim("the report names the sender", evidence, "ctx")
        assert "INSIDE-MARK" in seen["prompt"]
        assert "OUTSIDE-MARK" not in seen["prompt"]


# ── E. the judge is told a stated confidence is an assessment ──────────
class TestConfidenceIsAnAssessment:
    def test_claim_prompt(self):
        p = _VERIFY_CLAIM_PROMPT.format(claim="c", evidence="e", context="x")
        assert "OWN stated confidence" in p and "never a fabrication" in p

    def test_adjudicate_prompt(self):
        assert "OWN stated confidence" in _VERIFY_ADJUDICATE_PROMPT
        assert "FALSE ALARM" in _VERIFY_ADJUDICATE_PROMPT.split("OWN stated confidence")[1][:300]

    def test_enumerate_prompt(self):
        assert "not its own confidence score" in _VERIFY_ENUMERATE_PROMPT
