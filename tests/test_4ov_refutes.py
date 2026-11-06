"""§4OV (2026-10-14, operator "proceed"): verifier refutes on real traffic.
Labelled 10-07…10-14 (system/eval/4ov/labels.json): 17 refutes, 5 right,
9 false. Fixes: the judged text drops the away-digest; the written-file pack
keeps the part the claim refers to and covers scripts; the binder reads
"ERC-20" as a name; a refute only a judge produced is a measurement."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.reply_smoothing import strip_system_notes
from ghost_agent.core import claim_binding as C


# ── the judged text is the model's ──────────────────────────────────────

DIGEST = ("**Background activity while you were away:**\n  - [self-play] Candidate rule 4 — replays "
          "confirmed the fix\n\n---\n\n")


def test_the_away_digest_is_not_the_models_claim():
    assert strip_system_notes(DIGEST + "**Rule 1 (adopted):** be brief.") == "**Rule 1 (adopted):** be brief."
    two = DIGEST + "**While you were away** - project X advanced\n\n---\n\nThe answer."
    assert strip_system_notes(two) == "The answer."


def test_a_correction_banner_after_the_digest_goes_too():
    banner = "⚠️ **Correction to my previous answer:** it was 5.\n\n---\n\n"
    assert strip_system_notes(DIGEST + banner + "Body.") == "Body."


def test_a_reply_that_opens_with_its_own_short_section_is_kept():
    reply = "Here is the cascade:\n\n---\n\n**Layer 0** thinking"
    assert strip_system_notes(reply) == reply


# ── the binder reads names as names ─────────────────────────────────────

@pytest.mark.parametrize("text,want", [
    ("| USDT (ERC-20) | Part of ~$90M |", ["$90M"]),
    ("COVID-19 cases rose 12%", ["12%"]),
    ("GPT-4 scored 86 on it", ["86"]),
    ("F-35 jets and UH-60M helicopters", []),
    ("top-5 list of 30 items", ["5", "30"]),          # lower case: not a name
    ("Top-10 list has 12 items", ["10", "12"]),        # r1: a word, not an acronym
    ("Revenue-12% YoY", ["12%"]),
    ("Final: Lakers-110 Celtics-102", ["110", "102"]),
    ("ERC-20%", ["20%"]),
    ("pages 10-20", ["10-20"]),                        # a range stays a range
    ("Q-3.5 ratio", ["3.5"]),                          # a decimal is not a name's number
])
def test_hyphen_names_are_not_figures(text, want):
    assert [q.text for q in C.extract_quantities(text)] == want


# ── the written-file pack keeps what the claim refers to ────────────────

def _write_turn(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    tools = [{"name": "file_system", "content": "OK", "_args": {"operation": "write", "path": name}}]
    return tools


def test_a_big_file_keeps_the_part_the_claim_names(tmp_path, monkeypatch):
    body = "body { color: red; }\n" * 2000 + "\nconst APPS = ['Calculator', 'Notes', 'Terminal'];\n" + "x;\n" * 4000
    monkeypatch.setattr(A, "_files_mutated_this_turn", lambda tools: ["webos/index.html"])
    (tmp_path / "webos").mkdir()
    (tmp_path / "webos" / "index.html").write_text(body)
    pack = A._written_documents_for_audit([{"name": "file_system"}], tmp_path,
                                          claim_text="NebulaOS ships a Calculator, Notes and a Terminal app")
    assert "'Calculator', 'Notes', 'Terminal'" in pack
    assert "the part the reply's claim refers to" in pack
    plain = A._written_documents_for_audit([{"name": "file_system"}], tmp_path)
    assert "'Calculator'" not in plain                        # without a claim: head and tail, as before


def test_scripts_are_written_files_the_claim_lens_sees(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_files_mutated_this_turn", lambda tools: ["watch_pg_backup.sh"])
    (tmp_path / "watch_pg_backup.sh").write_text('SERVICE_NAME="postgresql@17-iboe_production"\nCHECK_INTERVAL=30\n')
    pack = A._written_documents_for_audit([{"name": "file_system"}], tmp_path, claim_text="polls every 30s")
    assert "postgresql@17-iboe_production" in pack and "CHECK_INTERVAL=30" in pack


def test_the_verdict_hands_the_claim_to_the_file_pack():
    from tests import pin_ast as P
    t = P.tree(A.GhostAgent._compute_verifier_verdict)
    calls = P.calls(t, "_written_documents_for_audit")
    assert calls and P.U(P.call_kw(calls[0], "claim_text")) == "_claim_src"


# ── a judge-only refute is a measurement ────────────────────────────────

@pytest.mark.parametrize("override,grounded", [
    ("", False), ("visual", True), ("web-exec", True), ("reply-shape+turn-state", True),
    ("file-artifact", True), ("memory-claim", True), ("image-unseen", False), ("claim_binding", False),
    ("visual+file-artifact", True), ("WEB-EXEC", True),
])
def test_grounded_means_a_mechanical_arm(override, grounded):
    assert A.refute_is_grounded(SimpleNamespace(override=override)) is grounded


@pytest.fixture
def late_agent(monkeypatch):
    monkeypatch.delenv("GHOST_JUDGE_REFUTE_CONSEQUENCES", raising=False)
    from ghost_agent.core.agent import GhostAgent
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = SimpleNamespace(verifier=None, trajectory_collector=None)
    ag._human_label_locked = lambda tid: False
    ag._strong_verdict_for = lambda tid: None
    ag._queue_source_caveat = lambda *a, **k: None
    ag._backfill_trajectory_outcome = MagicMock()
    ag._emit_late_outcome_correction = MagicMock()
    ag._relabel_episode = MagicMock()
    ag._record_withheld_verdict = MagicMock()
    return ag


def _refute(override=""):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    r = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="judge said so",
                     issues=["The claim states all services are stopped"])
    r.override = override
    return r


def test_a_judge_only_refute_changes_nothing_but_is_recorded(late_agent, monkeypatch):
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda t, m, **k: lines.append((m, k.get("level", "INFO"))))
    late_agent._record_late_verdict(_refute(), "tid-1", conv_fp="c")
    late_agent._backfill_trajectory_outcome.assert_not_called()
    late_agent._relabel_episode.assert_not_called()
    late_agent._record_withheld_verdict.assert_called_once()
    assert late_agent._record_withheld_verdict.call_args.args[:2] == ("tid-1", "failed")
    assert any("by the judge alone" in m and lvl == "INFO" for m, lvl in lines)


def test_a_grounded_refute_keeps_its_consequences(late_agent, monkeypatch):
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: None)
    late_agent._record_late_verdict(_refute("web-exec"), "tid-2", conv_fp="c")
    late_agent._backfill_trajectory_outcome.assert_called_once()
    assert late_agent._backfill_trajectory_outcome.call_args.args[:2] == ("tid-2", "failed")


def test_the_switch_restores_the_consequences(late_agent, monkeypatch):
    monkeypatch.setenv("GHOST_JUDGE_REFUTE_CONSEQUENCES", "1")
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: None)
    late_agent._record_late_verdict(_refute(), "tid-3", conv_fp="c")
    late_agent._backfill_trajectory_outcome.assert_called_once()


def test_a_confirm_is_untouched(late_agent, monkeypatch):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: None)
    late_agent._record_late_verdict(VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9,
                                                 reasoning="ok", issues=[]), "tid-4", conv_fp="c")
    assert late_agent._backfill_trajectory_outcome.call_args.args[:2] == ("tid-4", "passed")
    late_agent._record_withheld_verdict.assert_not_called()


@pytest.mark.parametrize("raw,on", [(None, False), ("0", False), ("1", True), ("yes", True), ("x", False)])
def test_the_switch(monkeypatch, raw, on):
    if raw is None:
        monkeypatch.delenv("GHOST_JUDGE_REFUTE_CONSEQUENCES", raising=False)
    else:
        monkeypatch.setenv("GHOST_JUDGE_REFUTE_CONSEQUENCES", raw)
    assert A.judge_refute_consequences_enabled() is on


def test_a_low_confidence_refute_is_not_recorded_as_a_failure(late_agent, monkeypatch):
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: None)
    r = _refute()
    r.confidence = 0.5
    late_agent._record_late_verdict(r, "tid-5", conv_fp="c")
    late_agent._record_withheld_verdict.assert_not_called()
    late_agent._backfill_trajectory_outcome.assert_not_called()


def test_r1_the_pack_never_takes_the_whole_evidence_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_files_mutated_this_turn", lambda tools: ["a.py"])
    (tmp_path / "a.py").write_text("x = 1\n" * 3000)
    pack = A._written_documents_for_audit([{"name": "file_system"}], tmp_path, claim_text="sets x", budget=2000)
    assert len(pack) <= 2100
    from tests import pin_ast as P
    t = P.tree(A.GhostAgent._compute_verifier_verdict)
    c = P.calls(t, "_written_documents_for_audit")[0]
    assert P.U(P.call_kw(c, "budget")) == "_ev_budget // 2"


def test_r1_no_claim_window_keeps_head_and_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_files_mutated_this_turn", lambda tools: ["b.html"])
    (tmp_path / "b.html").write_text("HEAD\n" + "filler\n" * 5000 + "TAILMARK\n")
    pack = A._written_documents_for_audit([{"name": "file_system"}], tmp_path, claim_text="zzzzzz qqqqqq")
    assert "TAILMARK" in pack and "head and tail" in pack


def test_r1_a_model_heading_is_not_a_digest():
    reply = "**While you were away** I fixed three bugs: " + "details " * 300 + "\n\n---\n\nThe rest"
    assert strip_system_notes(reply) == reply
