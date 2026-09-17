"""§4HS — instruments the last session could not read.

(a) The verdict site logs what the judge was handed: items packed of how
    many candidates, chars against the budget.
(b) Every escalation row carries the strong judge's finish_reason and
    content length — not only the `unavailable` rows.
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
    Verifier, VerifyResult, VerifyVerdict, record_escalation,
)


@pytest.fixture(autouse=True)
def _classic_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))


def _ledger(home):
    p = home / "system" / "verifier" / "escalations.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


class _MainClient:
    def __init__(self, content="", finish="length", worker=True):
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
    return VerifyResult(verdict=verdict, confidence=conf, reasoning="r", issues=list(issues or []))


# ── (b) the ledger row ─────────────────────────────────────────────────
class TestStrongCallOnTheRow:
    def test_record_writes_strong_fields_when_given(self, tmp_path):
        assert record_escalation(kind="refute", route="claim", outcome="overturned",
                                 strong_call={"finish_reason": "stop", "content_chars": 123},
                                 trace={"req_id": "r1"})
        row = _ledger(tmp_path)[-1]
        assert row["strong_finish"] == "stop" and row["strong_chars"] == 123

    def test_record_omits_the_fields_when_not_given(self, tmp_path):
        record_escalation(kind="refute", route="claim", outcome="upheld", trace={"req_id": "r2"})
        record_escalation(kind="refute", route="claim", outcome="upheld", strong_call={},
                          trace={"req_id": "r3"})
        for row in _ledger(tmp_path):
            assert "strong_finish" not in row and "strong_chars" not in row

    @pytest.mark.asyncio
    async def test_an_answered_refute_escalation_names_its_strong_call(self, tmp_path):
        c = _MainClient(_GOOD, "stop")
        v = Verifier(llm_client=c)
        cheap = _res(VerifyVerdict.REFUTED, 0.9, ["x"])
        out = await v._escalate_refute(cheap, "claim", "evidence", "ctx", trace={"req_id": "r4"})
        assert out.verdict == VerifyVerdict.CONFIRMED
        row = [r for r in _ledger(tmp_path) if r["outcome"] == "overturned"][-1]
        assert row["strong_finish"] == "stop"
        assert row["strong_chars"] == len(_GOOD)

    @pytest.mark.asyncio
    async def test_the_confirm_path_describes_its_own_strong_call(self, tmp_path, monkeypatch):
        """The stash is reset before the confirm retry: a stale
        finish_reason from an earlier main call must not be reported."""
        monkeypatch.setenv("GHOST_VERIFY_CONFIRM_ESCALATION", "1")
        c = _MainClient(_GOOD, "stop")
        v = Verifier(llm_client=c)
        v._last_main_call = {"finish_reason": "length", "content_chars": 0}   # stale

        async def _retry():
            data = await v._call_llm("re-judge", temperature=0.1, force_main=True)
            return v._build_verify_result(data)
        cheap = _res(VerifyVerdict.CONFIRMED, 0.9)
        out = await v._escalate_confirm(cheap, high_stakes=True, retry=_retry,
                                        trace={"req_id": "r5"})
        rows = [r for r in _ledger(tmp_path) if r["kind"] == "confirm"]
        assert rows, "confirm escalation did not run (gate?)"
        assert rows[-1]["outcome"] == "upheld"
        assert rows[-1]["strong_finish"] == "stop" and rows[-1]["strong_chars"] == len(_GOOD)
        assert out.verdict == VerifyVerdict.CONFIRMED

    @pytest.mark.asyncio
    async def test_a_confirm_retry_that_never_reaches_the_main_route_reports_no_stale_call(
            self, tmp_path, monkeypatch):
        """FAILS IF: the confirm path does not reset the stash before its
        strong call — an `unavailable` row would then describe an EARLIER
        main call as if it were this one."""
        monkeypatch.setenv("GHOST_VERIFY_CONFIRM_ESCALATION", "1")
        v = Verifier(llm_client=_MainClient(_GOOD, "stop"))
        v._last_main_call = {"finish_reason": "length", "content_chars": 0}   # stale

        async def _retry():
            raise RuntimeError("route down before any main call")
        cheap = _res(VerifyVerdict.CONFIRMED, 0.9)
        await v._escalate_confirm(cheap, high_stakes=True, retry=_retry, trace={"req_id": "r6"})
        rows = [r for r in _ledger(tmp_path) if r["kind"] == "confirm"]
        assert rows and rows[-1]["outcome"] == "unavailable"
        assert "strong_finish" not in rows[-1] and "strong_chars" not in rows[-1]

    def test_every_escalation_site_in_the_three_methods_passes_the_strong_call(self):
        tree = ast.parse(inspect.getsource(vmod))
        seen = 0
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in (
                    "_escalate_refute_impl", "_resolve_rebuttal", "_escalate_confirm_impl"):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                            and n.func.id == "record_escalation":
                        seen += 1
                        assert any(k.arg == "strong_call" for k in n.keywords), \
                            f"{fn.name}:{n.lineno} records without strong_call"
        assert seen >= 15


# ── (a) the digest line ────────────────────────────────────────────────
def _t(name, content):
    return {"name": name, "content": content}


class TestEvidenceDigestLine:
    def test_counts_only_candidate_labels(self):
        tools = [_t("browser", "page A"), _t("web_search", "### 1. hit"), _t("file_system", "SUCCESS wrote")]
        digest = "[browser] page A\n\n[web_search] ### 1. hit\n[x] not an item\n\n[file_system] SUCCESS wrote"
        with patch("ghost_agent.utils.logging.pretty_log") as plog:
            line = A._log_evidence_digest(digest, tools, 4000, 3)
        assert line == ("evidence digest — 3 item(s) of 3 candidate(s), "
                        f"{len(digest)} chars (budget 4000, positional ≤3 + claim pull)")
        assert plog.call_args.args[1] == line

    def test_empty_digest_logs_nothing(self):
        with patch("ghost_agent.utils.logging.pretty_log") as plog:
            assert A._log_evidence_digest("", [_t("browser", "x")], 4000, 3) == ""
        assert not plog.called

    def test_candidates_are_the_packers_candidates(self):
        tools = [_t("web_search", "### 1. hit")] + [_t("manage_tasks", '{"ok": true}')] * 5
        line = A._log_evidence_digest("[web_search] ### 1. hit", tools, 4000, 3)
        assert "1 item(s) of 1 candidate(s)" in line

    def test_the_verdict_site_logs_the_digest_it_packed(self):
        """AST pin: the ONE site that packs with the scaled budget also
        calls the digest logger on the same evidence/budget names."""
        tree = ast.parse(inspect.getsource(A))
        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_log_evidence_digest"]
        assert len(sites) == 1
        args = sites[0].args
        assert isinstance(args[0], ast.Name) and args[0].id == "claim_evidence"
        assert isinstance(args[2], ast.Name) and args[2].id == "_ev_budget"
        assert isinstance(args[3], ast.Name) and args[3].id == "_ev_items"
