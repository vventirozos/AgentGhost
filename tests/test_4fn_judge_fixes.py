"""§4FN pins — the judge, fixed where the 121 human labels said it was wrong.

Three of six false REFUTEs came from one parser bug: a compound age ("5
months and 23 days old") was read as its last component ("23 days old").
The seven false PASSes were replies that were not answers; the one shape
that is mechanical and clean on the labelled corpus (a raw tool dump) is
now refuted before any evidence question. And a machine verdict that a
fast human label WITHHELD is recorded on a measurement-only channel so the
judge stays measurable (84 of 121 labelled turns had no machine verdict).
"""
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.core import memory_claim_check as mcc
from ghost_agent.core import reply_shape_check as rsc
from ghost_agent.core.memory_claim_check import refute_age_claims
from tests.test_memory_claim_check import PROFILE, NOW


# --- compound ages ------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    # the three live false refutes of 2026-09-04 (cli feedback: correct answers)
    "Leonidas is **5 months and 23 days old** today (born March 12, 2026 → Sept 4, 2026 = 176 days).",
    "Thodoris is **9 years, 9 months and 10 days old** today (born November 25, 2016 → Sept 4, 2026).",
    "You're right — good catch. Let me recompute: March 12 → September 4. Leonidas is 5 months and 23 days old.",
])
def test_a_correct_compound_age_is_not_refuted(reply):
    """World where it fails: the tail component is read as the whole age
    ("Leonidas is stated as 23 day(s) old, but the stored birth date
    2026-03-12 makes Leonidas 5.9 months old today")."""
    assert refute_age_claims(reply=reply, profile=PROFILE, now=dt.date(2026, 9, 4)) == []


def test_compound_claims_are_one_claim_in_months_with_their_span():
    claims = mcc._age_claims("Leonidas is 5 months and 23 days old.")
    assert len(claims) == 1
    count, unit, start, end = claims[0]
    assert unit == "month" and abs(count - (5 + 23 * 12 / 365.25)) < 0.02
    assert start == len("Leonidas ") and end == len("Leonidas is 5 months and 23 days old")   # copula included
    claims = mcc._age_claims("Thodoris is 9 years, 9 months and 10 days old.")
    assert len(claims) == 1 and claims[0][1] == "month"
    assert abs(claims[0][0] - (9 * 12 + 9 + 10 * 12 / 365.25)) < 0.02
    # separators and spellings the corpus uses (review mutants 1–4)
    for text in ("9 years, 9 months, 10 days old", "3 weeks and 2 days old",
                 "1 year, 2 months and 3 weeks old", "5 months and 23 days-old"):
        cl = mcc._age_claims("Leonidas is " + text + ".")
        assert len(cl) == 1 and cl[0][1] == "month", text
    assert mcc._age_claims("Leonidas is 3 weeks and 2 days old.")[0][0] == pytest.approx(3 * 12 / 52 + 2 * 12 / 365.25, abs=0.01)
    assert mcc._age_claims("Leonidas is 1 year, 2 months and 3 weeks old.")[0][0] == pytest.approx(14 + 3 * 12 / 52, abs=0.01)


def test_compound_tolerance_is_the_month_window_not_a_three_year_band():
    """§4FN review M1: '1 year and 2 months old' for a 5.9-month-old must
    refute exactly as '14 months old' does."""
    now = dt.date(2026, 9, 4)
    assert refute_age_claims(reply="Leonidas is 14 months old.", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Leonidas is 1 year and 2 months old.", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Leonidas is 1 year, 2 months and 3 days old.", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Thodoris is 9 years, 9 months and 10 days old.", profile=PROFILE, now=now) == []


def test_true_months_is_exact_to_the_day():
    """§4FN review minor 2: (days % 30)/30 read 117.0 for 117.33 and 5.87
    for 5.77 — the remainder must be the days past the last whole month."""
    assert mcc._true_months(dt.date(2026, 3, 12), dt.date(2026, 9, 4)) == pytest.approx(5 + 23 / 30.4375, abs=0.01)
    assert mcc._true_months(dt.date(2016, 11, 25), dt.date(2026, 9, 4)) == pytest.approx(117 + 10 / 30.4375, abs=0.01)
    assert mcc._true_months(dt.date(2026, 3, 12), dt.date(2026, 3, 12)) == 0.0


CORRECT_TWO_SUBJECT_REPLIES = (
    # the docstring's live tie case and the two §4FN C1 compound ties
    "Your sons are 9 (Thodoris) and 5 months old (Leonidas).",
    "Your sons are 9 years and 9 months old (Thodoris) and 5 months and 23 days old (Leonidas).",
    "Your sons are 9 years, 9 months and 10 days old (Thodoris) and 5 months and 23 days old (Leonidas).",
    # the six appositive / colon / dash / comma phrasings of review C1-R2 (all correct)
    "Thodoris (9 years old) and Leonidas (5 months old)",
    "Leonidas (5 months old) and Thodoris (9 years old)",
    "Thodoris: 9 years old; Leonidas: 5 months old",
    "Thodoris — 9 years old, Leonidas — 5 months old",
    "Thodoris, 9 years old, and Leonidas, 5 months old",
    "Thodoris (9 years, 9 months and 10 days old) and Leonidas (5 months and 23 days old)",
    # a following name that is NOT a subject: only a symmetric gap keeps the
    # true subject (asymmetric-by-name-length bound "9 years" to Leonidas)
    "Thodoris is 9 years old, like Leonidas.",
    "Thodoris is 9 years, 9 months and 10 days old, like Leonidas.",
    "Leonidas is 5 months old, unlike Thodoris.",
    # copula and "aged" subjects, both orders
    "Thodoris is 9 years old and Leonidas is 5 months old.",
    "Leonidas is 5 months old and Thodoris is 9 years old.",
    "Thodoris is 9 years old and Leonidas aged 5 months.",
    "Leonidas is 5 months and 23 days old; Thodoris is 9 years, 9 months and 10 days old.",
)


@pytest.mark.parametrize("reply", CORRECT_TWO_SUBJECT_REPLIES)
def test_correct_two_subject_replies_are_never_refuted(reply):
    """§4FN C1 → C1-R2 → round 3. World where it fails: the subject is the
    NEAREST name by distance (any rule: from the phrase start, from its
    nearest edge, with or without a punctuation cue) — six of these refute
    under each of the three distance rules that preceded the link rule."""
    assert refute_age_claims(reply=reply, profile=PROFILE, now=dt.date(2026, 9, 4)) == []


def test_wrong_ages_in_the_same_shapes_still_refute():
    now = dt.date(2026, 9, 4)
    assert refute_age_claims(reply="Leonidas is 9 years and 9 months old today.", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Thodoris (5 months old) and Leonidas (9 years old)", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Leonidas: 9 years old; Thodoris: 5 months old", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Leonidas is 3 months old.", profile=PROFILE, now=now)   # upper bound: 5 < 5.8


def test_two_subjects_sharing_one_compound_are_ambiguous_not_a_claim():
    """§4FN review M-B → round 3: 'Thodoris and Leonidas are 9 years and 5
    months old' is two ages glued together. The compound is ONE claim (its
    tail, '5 months old', is never a second one) that binds to NOBODY —
    the ambiguity is decided at the subject, where every other plural
    surface ("Thodoris and Leonidas: …", "… are, respectively, …") is
    decided too, not by a copula test on the phrase."""
    now = dt.date(2026, 9, 4)
    reply = "Thodoris and Leonidas are 9 years and 5 months old."
    claims = mcc._age_claims(reply)
    assert [(c[0], c[1]) for c in claims] == [(113.0, "month")], claims
    assert refute_age_claims(reply=reply, profile=PROFILE, now=now) == []


def test_an_implausible_compound_shadows_its_tail():
    """§4FN review M-A: a declined compound must not fall through to the
    single-unit patterns ('999 years and 11 months old' → '11 months old')."""
    claims = mcc._age_claims("The artifact is 999 years and 11 months old, Leonidas is 5 months old.")
    assert [(c[1], c[0]) for c in claims] == [("month", 5)]


def test_a_future_birth_date_is_bad_data_not_evidence():
    profile = {"family": {"sons": "Leonidas (born 2027-03-12)"}}
    assert refute_age_claims(reply="Leonidas is 9 years old.", profile=profile, now=dt.date(2026, 9, 4)) == []


def test_the_profile_writer_anchors_a_compound_as_a_whole(monkeypatch):
    """§4FN review M2: `temporal.anchor` read the tail component and anchored
    a nine-year-old to a birth date ten days ago. A compound is known to at
    least the month, so it anchors in the full-date form (round 3 m1: the
    "smallest unit" expression could never yield a year and was dead)."""
    from ghost_agent.memory import temporal as tp
    said = dt.date(2026, 9, 4)
    out = tp.anchor("Thodoris is 9 years, 9 months and 10 days old", said_at=said)
    assert out == "Thodoris born ~2016-11-25", out                    # the copula goes with the phrase, as for single units
    out = tp.anchor("Leonidas is 5 months and 23 days old", said_at=said)
    assert out == "Leonidas born ~2026-03-12", out
    out = tp.anchor("Leonidas is 1 year and 2 months old", said_at=said)
    assert out == "Leonidas born ~2025-07-04", out                       # months → the day
    # declined (implausible) and plural-subject compounds are left whole:
    # the tail must NOT be anchored on its own (review M-A / M-B)
    assert tp.anchor("the artifact is 999 years and 11 months old", said_at=said) == "the artifact is 999 years and 11 months old"
    assert tp.anchor("Thodoris and Leonidas are 9 years and 5 months old", said_at=said) == "Thodoris and Leonidas are 9 years and 5 months old"
    # idempotent, and a quoted span is left alone
    assert tp.anchor(out, said_at=said) == out
    assert tp.anchor('he wrote "5 months and 23 days old"', said_at=said) == 'he wrote "5 months and 23 days old"' or "2026" not in tp.anchor('he wrote "5 months and 23 days old"', said_at=said)
    # the checker and the writer share ONE compound authority
    assert mcc._COMPOUND_AGE_RE is tp._COMPOUND_AGE_RE


def test_a_wrong_compound_age_still_refutes():
    """The compound path must not become a free pass: 9 months and 23 days
    for a 5.8-month-old is off by four months, outside the window."""
    issues = refute_age_claims(reply="Leonidas is 9 months and 23 days old.",
                               profile=PROFILE, now=dt.date(2026, 9, 4))
    assert len(issues) == 1 and "Leonidas" in issues[0] and "9.76 month(s)" in issues[0]


def test_single_unit_claims_are_unchanged():
    assert refute_age_claims(reply="Leonidas is 9 years old.", profile=PROFILE, now=NOW)
    assert refute_age_claims(reply="Leonidas is 6 months old.", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply="Leonidas is 23 days old.", profile=PROFILE, now=dt.date(2026, 9, 4))


# --- reply shape ---------------------------------------------------------------

LIVE_DUMP = ("Process finished successfully.\n\n### Final Output:\n```text\n### 1. Γαστρονομία"
             "Φουρφουρά,ΓαστρονομίαΚρήτης\n[Source: https://www.visitfourfouras.gr/]\n")


@pytest.mark.parametrize("reply,flag", [
    (LIVE_DUMP, True),                                       # the live false pass 70d04ea6
    (rsc.FALLBACK_HEADS["failed"] + "\n\n### Final Output:\n```text\nTraceback\n```", True),   # sibling banner
    (rsc.FALLBACK_HEADS["running"] + "\n\n### Final Output:\n```text\n…\n```", True),
    ("### Final Output:\n```text\nx\n```", True),
    ("--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT: 3\n", True),
    ("--- COMMAND RESULT ---\nok", True),
    ("[sandbox job 7 finished — EXIT CODE: 0]\nout", True),
    ("```\nEXIT CODE: 1\nTraceback…\n```", True),
    ("The file has 3 lines. (I ran `wc -l`; the tool said: --- EXECUTION RESULT --- EXIT CODE: 0)", False),
    ("Here are the results of the research:\n\n### Final Output: none needed", False),
    ("PONG", False),
    ("", False),
])
def test_raw_tool_dump_is_refuted_only_when_the_reply_opens_as_a_dump(reply, flag):
    issues = rsc.refute_raw_tool_dump(reply)
    assert bool(issues) is flag
    if flag:
        assert "raw tool output" in issues[0] and "not answered" in issues[0]


def test_a_request_for_the_raw_output_is_not_refuted():
    assert rsc.refute_raw_tool_dump(LIVE_DUMP, "show me the raw output of the command") == []
    assert rsc.refute_raw_tool_dump(LIVE_DUMP, "paste the full stdout as-is") == []
    assert rsc.refute_raw_tool_dump(LIVE_DUMP, "what did the research find?")


def test_the_finalize_fallback_and_the_check_share_one_literal():
    """§4FN review M5: two copies of one banner is how a check goes dark.
    The finalize site must build its banner from FALLBACK_HEADS."""
    import inspect
    from ghost_agent.core import agent as agent_mod
    src = inspect.getsource(agent_mod)
    assert 'FALLBACK_HEADS["success"]' in src and 'FALLBACK_HEADS["failed"]' in src and 'FALLBACK_HEADS["running"]' in src
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert code.count('"Process finished successfully."') == 0     # comments may quote it; code may not
    assert rsc.FALLBACK_HEADS["success"] == "Process finished successfully."


def test_narration_is_deliberately_not_a_refute():
    """Measured: 4 human-approved replies for every 1 rejected — a worse
    judge than none. World where it fails: someone adds it back to the
    pattern (asserted on the compiled pattern, not on source lines)."""
    pat = rsc._DUMP_HEAD_RE.pattern.lower()
    for word in ("let me", "i now have", "i have enough", "i'll now"):
        assert word not in pat
    narr = ("The searches were noisy. Let me run focused deep research. I now have solid data. "
            "Let me write it up.\n\n# Olive Oil Price Forecast")
    assert rsc.refute_raw_tool_dump(narr) == []


@pytest.mark.asyncio
async def test_verdict_computation_refutes_a_dump_before_asking_the_verifier(mock_context, tmp_path):
    """At the REAL site: a tool turn whose reply is a raw dump is REFUTED
    mechanically and the LLM verifier is never asked; a normal reply falls
    through to the verifier; an ablation with no verifier gets no verdict."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = GhostAgent(mock_context)
    # the ground-truth overrides now RUN on these turns (§4FN M3): give them
    # real, empty paths so nothing mock-derived is created on disk
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    # the verdict sidecar is written beside the collector's root (proof that
    # the shape verdict is RECORDED, review M3): a real collector on tmp_path
    from ghost_agent.distill.collector import TrajectoryCollector
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    called = {"n": 0}

    async def fake_verify(*a, **k):
        called["n"] += 1
        return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    # whichever verifier entry the site awaits, count it
    VERIFY_NAMES = ("verify_response", "verify", "verify_turn", "run", "verify_claim",
                    "verify_code_output", "verify_visual")
    for name in VERIFY_NAMES:
        setattr(verifier, name, fake_verify)
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    tools = [{"name": "execute", "arguments": {"command": "python x.py"}, "content": "EXIT CODE: 0\nout", "result": "EXIT CODE: 0\nout"}]
    async def run(reply, rid, **over):
        kw = dict(tools_run_this_turn=tools, messages=[{"role": "user", "content": "run it"}],
                  final_ai_content=reply, last_user_content="run it", lc="run it", req_id=rid, trajectory_id=rid)
        kw.update(over)
        return await agent._compute_verifier_verdict(**kw)
    res, last_tool = await run(LIVE_DUMP, "r1")
    assert res is not None and res.verdict == VerifyVerdict.REFUTED and res.confidence >= 0.7
    assert "raw tool output" in res.issues[0]
    # …and it reached the verdict sidecar the instruments read (review M3)
    side = list((tmp_path / "verdicts").glob("*.jsonl"))
    rows = [json.loads(l) for f in side for l in f.read_text().splitlines() if l.strip()]
    assert rows and rows[-1]["trajectory_id"] == "r1"
    assert str(rows[-1]["verdict"]).lower().startswith("refut") and "reply-shape" in str(rows[-1].get("override") or "")
    # the control leg: a normal reply reaches the verifier (once) and its verdict stands
    called["n"] = 0
    res_ok, _ = await run("The file has 3 lines.", "r1b")
    assert called["n"] == 1 and res_ok is not None and res_ok.verdict == VerifyVerdict.CONFIRMED
    # a refute that already stands keeps its grounded issue in the first slot
    async def fake_refute(*a, **k):
        return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                            issues=["the page throws a SyntaxError on load", "x", "y"])
    for name in VERIFY_NAMES:
        setattr(verifier, name, fake_refute)
    res_m, _ = await run(LIVE_DUMP, "r1c")
    assert res_m.issues[0] == "the page throws a SyntaxError on load" and any("raw tool output" in i for i in res_m.issues)
    # the user asked for the raw output: not refuted by shape
    for name in VERIFY_NAMES:
        setattr(verifier, name, fake_verify)
    res_raw, _ = await run(LIVE_DUMP, "r1d", last_user_content="show me the raw output")
    assert res_raw.verdict == VerifyVerdict.CONFIRMED
    # ablation: no verifier attached → no verdict at all; a verifier with no client → none either
    agent.context.verifier = None
    res2, _ = await run(LIVE_DUMP, "r2")
    assert res2 is None
    v2 = MagicMock(); v2.llm_client = None
    agent.context.verifier = v2
    res3, _ = await run(LIVE_DUMP, "r3")
    assert res3 is None


# --- the withheld-verdict measurement channel ---------------------------------

def _collector(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    return TrajectoryCollector(tmp_path / "sys" / "trajectories")   # root.parent is per-test


def test_withheld_verdicts_fill_the_machine_slot_only_when_corrections_gave_none(tmp_path):
    """Identity pin over the real files: a withheld verdict is the machine
    half of the pair when the corrections file has no machine row; a
    corrections-file verdict (the one that shipped) stays authoritative;
    the outcome overlay never sees the withheld file."""
    c = _collector(tmp_path)
    c.update_outcome("t-human-first", "passed", "thumbs up", source="human_feedback:web")
    assert c.record_withheld_verdict("t-human-first", "failed", "refuted late") is True
    c.update_outcome("t-both", "passed", "", source="verifier_late")
    c.update_outcome("t-both", "failed", "thumbs down", source="human_feedback:cli")
    assert c.record_withheld_verdict("t-both", "failed", "later re-verify") is True
    assert c.record_withheld_verdict("", "failed") is False
    assert c.record_withheld_verdict("t-x", "unknown") is False
    # a retraction stays retracted; an id the corrections file never saw mints no pair
    c.update_outcome("t-retracted", "failed", "", source="verifier_late")
    c.update_outcome("t-retracted", "unknown", "false positive", source="operator_overlay")
    assert c.record_withheld_verdict("t-retracted", "failed", "again") is True
    assert c.record_withheld_verdict("t-orphan", "failed", "no human row") is True
    # last write wins among withheld rows
    assert c.record_withheld_verdict("t-human-first", "passed", "re-verified") is True
    mh = c.machine_and_human_outcomes()
    assert mh["t-human-first"] == ("passed", "passed")     # measurable pair (last write), human still gold
    assert mh["t-both"] == ("passed", "failed")            # the shipped verdict wins the slot
    assert mh["t-retracted"] == (None, None)
    assert "t-x" not in mh and "t-orphan" not in mh
    # the overlay that decides shipped outcomes reads corrections only
    assert c.latest_correction("t-human-first")["outcome"] == "passed"
    wfile = tmp_path / "sys" / "judge" / "withheld_verdicts.jsonl"
    assert c._withheld_path() == wfile and wfile.exists()      # its own directory: not the corpus root, not verdicts/
    assert not list((tmp_path / "sys" / "verdicts").glob("*.jsonl")) if (tmp_path / "sys" / "verdicts").exists() else True
    assert not any("withheld" in l for l in (tmp_path / "sys" / "trajectories" / "corrections.jsonl").read_text().splitlines())
    # a disabled (read-only) collector never appends; the reason is redacted
    from ghost_agent.distill.collector import TrajectoryCollector
    ro = TrajectoryCollector(tmp_path / "sys" / "trajectories", enabled=False)
    assert ro.record_withheld_verdict("t-human-first", "failed") is False
    c.record_withheld_verdict("t-human-first", "failed", "token sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 leaked")
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345" not in wfile.read_text()


def test_withheld_rows_are_readable_without_a_corrections_file(tmp_path):
    c = _collector(tmp_path)
    assert c.record_withheld_verdict("t1", "failed") is True
    assert c.machine_and_human_outcomes() == {}              # no human label known → no pair (never an orphan)


def test_a_malformed_withheld_row_never_fills_the_machine_slot(tmp_path):
    """The reader validates the outcome itself (review mutant 6): a hand-
    edited or corrupted row with outcome "unknown" must not become the
    judge's answer."""
    c = _collector(tmp_path)
    c.update_outcome("t-h", "passed", "thumbs up", source="human_feedback:web")
    wf = c._withheld_path(); wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(json.dumps({"trajectory_id": "t-h", "outcome": "unknown", "source": "verifier_late"}) + "\n"
                  + json.dumps({"trajectory_id": "t-h", "outcome": "", "source": "verifier_late"}) + "\n")
    assert c.machine_and_human_outcomes()["t-h"] == (None, "passed")
    wf.write_text(wf.read_text() + json.dumps({"trajectory_id": "t-h", "outcome": "failed", "source": "verifier_late"}) + "\n")
    assert c.machine_and_human_outcomes()["t-h"] == ("failed", "passed")


def test_all_three_withhold_sites_record_the_verdict(mock_context):
    """§4FN review M7: the writer-side yield and the cached-label guard
    withheld silently too."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(mock_context)
    recorded = []
    coll = MagicMock()
    coll.record_withheld_verdict = lambda tid, oc, reason="": recorded.append((tid, oc)) or True
    agent.context.trajectory_collector = coll
    agent._record_withheld_verdict("t-a", "failed", "r")
    agent._record_withheld_verdict("t-b", "unknown", "r")     # never an outcome that is not a verdict
    assert recorded == [("t-a", "failed")]
    import inspect
    from ghost_agent.core import agent as agent_mod
    src = inspect.getsource(agent_mod)
    i = src.index("def _backfill_trajectory_outcome"); seg = src[i:i + 6000]
    j = seg.index('WITHHELD — a human label already resolved this turn')
    assert "self._record_withheld_verdict(trajectory_id, outcome, reason)" in seg[j:j + 400]
    k = src.index('if ok == "withheld":')
    assert "self._record_withheld_verdict(trajectory_id, outcome, reason)" in src[k:k + 200]


@pytest.mark.parametrize("verdict_name,conf,expect", [
    ("REFUTED", 0.9, "failed"), ("CONFIRMED", 0.8, "passed"),
    ("REFUTED", 0.6, None),                   # below the consequence bar: not measured either
    ("UNCERTAIN", 0.9, None),
])
def test_locked_late_verdict_is_recorded_as_withheld_at_the_real_handler(mock_context, verdict_name, conf, expect):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    if not hasattr(VerifyVerdict, verdict_name):
        pytest.skip(f"no {verdict_name} verdict in this build")
    agent = GhostAgent(mock_context)
    agent._human_label_locked = lambda tid: True
    recorded = []
    coll = MagicMock()
    coll.record_withheld_verdict = lambda tid, oc, reason="": recorded.append((tid, oc, reason)) or True
    agent.context.trajectory_collector = coll
    backfilled = []
    agent._backfill_trajectory_outcome = lambda *a, **k: backfilled.append(a)
    agent._emit_late_outcome_correction = lambda *a, **k: backfilled.append(a)
    v = VerifyResult(verdict=getattr(VerifyVerdict, verdict_name), confidence=conf,
                     reasoning="r", issues=["Stiva's is not in the evidence"])
    agent._record_late_verdict(v, "t-locked")
    assert backfilled == []                                   # withheld from every consequence
    assert [r[:2] for r in recorded] == ([("t-locked", expect)] if expect else [])
    if expect:
        assert "Stiva" in recorded[0][2]


def test_unlocked_late_verdict_is_not_written_to_the_withheld_channel(mock_context):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = GhostAgent(mock_context)
    agent._human_label_locked = lambda tid: False
    recorded = []
    coll = MagicMock()
    coll.record_withheld_verdict = lambda *a, **k: recorded.append(a) or True
    agent.context.trajectory_collector = coll
    agent._backfill_trajectory_outcome = lambda *a, **k: None
    agent._emit_late_outcome_correction = lambda *a, **k: None
    agent._record_late_verdict(VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9,
                                            reasoning="r", issues=[]), "t-open")
    assert recorded == []



def test_a_shape_issue_is_never_filed_as_a_project_task(mock_context, monkeypatch):
    """§4FN review M4: a delivery-shape complaint is not project work and
    must not consume a DONE project's reopen slot; a grounded issue still is."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    monkeypatch.delenv("GHOST_REFUTE_FOLLOWUP_TASKS", raising=False)
    agent = GhostAgent(mock_context)
    added = []

    class Store:
        def get_project(self, pid):
            return {"id": pid, "status": "ACTIVE", "metadata": {}}

        def add_task(self, pid, description, **kw):
            added.append(description); return {"id": "t1"}

        def list_tasks(self, pid):
            return []
    agent.context.project_store = Store()
    shape_issue = rsc.refute_raw_tool_dump(LIVE_DUMP)[0]
    assert len(shape_issue) >= agent._REFUTE_TASK_MIN_CHARS
    agent._file_refute_followup_tasks(
        VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r", issues=[shape_issue]), "p1")
    assert added == []
    agent._file_refute_followup_tasks(
        VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                     issues=["the CSV export omits the header row the request asked for"]), "p1")
    assert len(added) == 1 and "CSV export" in added[0]



def test_override_provenance_is_chained_by_one_helper():
    """§4FN review minor 5: the WEB-EXEC arm stamped a bare tag and the
    FILE-ARTIFACT replace path dropped the stamp with the old object. One
    helper, an identity pin, and no bare `override = "…"` assignment left
    in the verdict computation."""
    import inspect
    from types import SimpleNamespace
    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core.agent import GhostAgent
    v = SimpleNamespace()
    GhostAgent._chain_override(v, "reply-shape", "")
    assert v.override == "reply-shape"
    GhostAgent._chain_override(v, "web-exec", v.override)
    assert v.override == "reply-shape+web-exec"
    GhostAgent._chain_override(v, "file-artifact", v.override)
    assert v.override == "reply-shape+web-exec+file-artifact"
    import ast, textwrap
    src = textwrap.dedent(inspect.getsource(GhostAgent._compute_verifier_verdict))
    tree = ast.parse(src)
    bare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "override" for t in n.targets)]
    assert bare == [], f"bare override assignments at {bare}: every arm must stamp through _chain_override"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self._chain_override"]
    tags = {n.args[1].value for n in calls if len(n.args) > 1 and isinstance(n.args[1], ast.Constant)}
    # reply-shape (tool and tool-free), memory-claim (tool-free, round 4 M1), VISUAL (round 3 m5: a
    # visual refute reported as "(text judge)"), web-exec, file-artifact merge + replace
    assert tags == {"reply-shape", "memory-claim", "visual", "web-exec", "file-artifact"}, tags
    assert len(calls) >= 7


# --- round 3: the subject is read from the LINK, not from the distance -------
#
# Three rounds of "nearest name within N characters" each moved the defect
# (C1 → C1-R2 → round 3 C2/M1). A name is the subject of an age phrase when
# everything between them is a link (copula / age cue / hedge / punctuation)
# and it does not close a list; the one backward form is a bracketed name
# right after the phrase. Every case below was REFUTED, or lost its refute,
# under the distance rules.

NOT_ABOUT_THE_NAMED_PERSON = (
    # round 3 C2: a name in possessive / attributive position captured any age nearby
    "Thodoris's laptop is 3 years old.",
    "Thodoris goes to a school that is 40 years old.",
    "Thodoris plays for Panellinios, a club 130 years old.",
    "Leonidas has a cot that is 20 years old.",
    "Leonidas Street is named after a statue that is 90 years old.",
    "I looked it up for Thodoris — the tree is 300 years old.",
    "Thodoris and Maria are 3 years old.",             # an unanchored name in the gap is a word too
    "The lease is 9 years old (Leonidas signed it)",   # a bracket that does not close on the name is prose
)


@pytest.mark.parametrize("reply", NOT_ABOUT_THE_NAMED_PERSON)
def test_an_age_the_name_is_not_predicated_of_is_unbound(reply):
    """World where it fails: the closest name is the subject (round-1/2/3
    distance rules — all six of the first shapes REFUTE under each)."""
    assert refute_age_claims(reply=reply, profile=PROFILE, now=dt.date(2026, 9, 4)) == []


LINKED_CORRECT = (
    # round 3 M1: one word between a name and its own age broke the punctuation cue
    "Thodoris — now 9 years old, Leonidas — now 5 months old",
    "Thodoris' age: 9 years old. Leonidas' age: 5 months old.",
    "Thodoris's age is 9 years old; Leonidas's age is 5 months old",
    "Thodoris = 9 years old, Leonidas = 5 months old",
    "Thodoris -> 9 years old; Leonidas -> 5 months old",
    "Thodoris, who is 9 years old, and Leonidas, who is 5 months old",
    "**Thodoris** is currently about 9 years old",
    "Thodoris is exactly 9 years old; Leonidas is just 5 months old",
)
LINKED_WRONG = (
    "Thodoris — now 5 months old, Leonidas — now 9 years old",
    "Thodoris' age: 5 months old. Leonidas' age: 9 years old.",
    "Thodoris's age is 5 months old; Leonidas's age is 9 years old",
    "Thodoris = 5 months old, Leonidas = 9 years old",
    "Thodoris -> 5 months old; Leonidas -> 9 years old",
    "Thodoris, who is 5 months old, and Leonidas, who is 9 years old",
    "**Thodoris** is currently about 5 months old; Leonidas is 9 years old",
    "Thodoris is exactly 5 months old; Leonidas is just 9 years old",
)


@pytest.mark.parametrize("reply", LINKED_CORRECT)
def test_a_link_word_between_a_name_and_its_age_keeps_them_together(reply):
    """World where it fails: the link admits only punctuation (the round-2
    cue) — the following name is then the previous claim's nearest
    candidate and every one of these refutes."""
    assert refute_age_claims(reply=reply, profile=PROFILE, now=dt.date(2026, 9, 4)) == []


@pytest.mark.parametrize("reply", LINKED_WRONG)
def test_the_same_links_carry_a_wrong_age_to_its_subject(reply):
    """The mirror: each link word must BIND, or the rule is 'abstain always'.
    Both children are wrong here, so both are named — identity, not
    'something refuted'."""
    issues = refute_age_claims(reply=reply, profile=PROFILE, now=dt.date(2026, 9, 4))
    assert {i.split()[0] for i in issues} == {"Thodoris", "Leonidas"}, issues


def test_a_birth_date_between_the_name_and_its_age_is_a_link():
    """The agent's arithmetic shape and a table row: the date in between is
    the computation, not another subject. Both directions — the correct
    value is consistent, the wrong one is refuted — so the date tokens are
    proven to BIND, not merely to be tolerated."""
    now = dt.date(2026, 9, 4)
    live = "- **Leonidas:** born March 12, 2026 → today (Sep 4, 2026) is **about {} months old** (5 months and 3 weeks)."
    assert refute_age_claims(reply=live.format(6), profile=PROFILE, now=now) == []
    assert refute_age_claims(reply=live.format(16), profile=PROFILE, now=now)[0].startswith("Leonidas is stated as 16")
    assert refute_age_claims(reply="| Leonidas | 2026-03-12 | 5 months old |", profile=PROFILE, now=now) == []
    assert refute_age_claims(reply="| Leonidas | 2026-03-12 | 40 years old |", profile=PROFILE, now=now)
    assert refute_age_claims(reply="Thodoris (b. 25/11/2016) is 9 years old", profile=PROFILE, now=now) == []
    assert refute_age_claims(reply="Thodoris (b. 25/11/2016) is 3 years old", profile=PROFILE, now=now)


def test_a_name_that_closes_a_list_is_not_one_persons_subject():
    """Round 3 C1 (checker half): 'Thodoris and Leonidas: 9 years and 5
    months old' binds Leonidas under any link rule that ignores the list.
    A clause-level 'and' is not a list."""
    now = dt.date(2026, 9, 4)
    for reply in ("Thodoris and Leonidas: 9 years and 5 months old.",
                  "Your sons Thodoris and Leonidas are aged 9 years and 5 months.",
                  "Thodoris, Leonidas and Vasilis are 9, 5 months and 46 years old.",
                  "Thodoris & Leonidas — 9 years and 5 months old"):
        assert refute_age_claims(reply=reply, profile=PROFILE, now=now) == [], reply
    # control: ", and Thodoris is 12 years old" is a new clause, and wrong
    issues = refute_age_claims(reply="Leonidas is 5 months old, and Thodoris is 12 years old.", profile=PROFILE, now=now)
    assert len(issues) == 1 and issues[0].startswith("Thodoris is stated as 12"), issues


def test_a_bracketed_name_after_the_phrase_is_its_subject():
    """The agent's own disambiguation form, and the ONLY backward binding:
    '9 years old (Thodoris)'. The link forward wins when it exists."""
    now = dt.date(2026, 9, 4)
    assert refute_age_claims(reply="Your sons are 9 years old (Thodoris) and 5 months old (Leonidas).", profile=PROFILE, now=now) == []
    swapped = refute_age_claims(reply="Your sons are 9 years old (Leonidas) and 5 months old (Thodoris).", profile=PROFILE, now=now)
    assert {i.split()[0] for i in swapped} == {"Thodoris", "Leonidas"}, swapped
    assert refute_age_claims(reply="Your sons are 9 years old [Thodoris] and 5 months old [Leonidas].", profile=PROFILE, now=now) == []
    # forward first: the phrase is predicated of Thodoris; the bracket is an aside
    assert refute_age_claims(reply="Thodoris is 9 years old (Leonidas)", profile=PROFILE, now=now) == []
    assert refute_age_claims(reply="Thodoris is 5 months old (Leonidas)", profile=PROFILE, now=now)[0].startswith("Thodoris")


def test_a_link_never_crosses_a_line():
    """A heading and a list item are two lines; whitespace is a link
    character, so the line rule is what keeps 'Leonidas:\\n- 9 years old'
    unbound (measured: the three live false refutes were all cross-line)."""
    assert refute_age_claims(reply="Leonidas:\n- 9 years old", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply="Leonidas: 9 years old", profile=PROFILE, now=NOW)   # same text, one line


def test_the_reader_skips_a_date_glued_onto_a_list():
    """Round 3 C1, the poisoned loop: a stored 'Thodoris and Leonidas: born
    ~2017-04-04' read as Leonidas's date refutes the CORRECT answer about
    him forever. The reader must survive data the old writer stored."""
    glued = {"sons": "Thodoris and Leonidas: born ~2017-04-04"}
    assert mcc.anchored_subjects(glued) == []
    assert refute_age_claims(reply="Leonidas is 5 months old.", profile=glued, now=NOW) == []
    # the live shape — each name with its own anchor — still yields both
    assert [n for n, _ in mcc.anchored_subjects(PROFILE)] == ["Thodoris", "Leonidas"]
    # a lowercase word before an anchor is not a name (unchanged from §4EQ)
    assert mcc.anchored_subjects({"k": "my twins born ~2026-04-19"}) == []


def test_a_greek_spelled_name_is_a_subject():
    """Round 3 m4: `[A-Z]` is ASCII; the operator's family is Greek."""
    prof = {"k": "Θοδωρής (born 2016-11-25)"}
    assert mcc.anchored_subjects(prof) == [("Θοδωρής", dt.date(2016, 11, 25))]
    issues = refute_age_claims(reply="Θοδωρής is 5 months old.", profile=prof, now=NOW)
    assert issues and issues[0].startswith("Θοδωρής is stated as 5"), issues


@pytest.mark.parametrize("text", [
    # round 3 C1 (writer half): every one of these anchored BOTH children to one date
    "Thodoris and Leonidas: 9 years and 5 months old",
    "Thodoris and Leonidas are, respectively, 9 years and 5 months old",
    "Thodoris and Leonidas are: 9 years and 5 months old",
    "Thodoris and Leonidas are about 9 years and 5 months old",
    "Thodoris and Leonidas were 9 years and 5 months old",
    "The boys are: 9 years and 5 months old",              # plural copula, no list
    "The boys were, respectively, 9 years and 5 months old",  # "were" is a tense marker: left alone before any plural rule
    "Thodoris and Leonidas are 9 and 5 months old",        # single unit at the tail of a list
    "Θοδωρής και Λεωνίδας: 9 years and 5 months old",
])
def test_the_writer_leaves_a_plural_subject_whole(text):
    from ghost_agent.memory import temporal as tp
    assert tp.anchor(text, said_at=dt.date(2026, 9, 4)) == text


def test_the_writer_still_anchors_one_persons_age():
    """The plural guards are not a blanket mute: a copula two clauses back,
    a twins phrase, and a clause-level 'and' all anchor."""
    from ghost_agent.memory import temporal as tp
    said = dt.date(2026, 9, 4)
    assert tp.anchor("The kids are great, and Leonidas is 5 months and 23 days old", said_at=said) == "The kids are great, and Leonidas born ~2026-03-12"
    assert tp.anchor("My twins are 4 months old", said_at=said).startswith("My twins born ~2026-0")
    out = tp.anchor("Leonidas is 5 months old, and Thodoris is 9 years, 9 months and 10 days old", said_at=said)
    assert out.endswith("and Thodoris born ~2016-11-25"), out


def test_the_writer_counts_a_compound_exactly():
    """Round 3 m3: the day part went through a months fraction and a
    30.4375 constant; 40 days came back as 10 days after a month. Exact
    parts: Sep 4 − 40 days = Jul 26, − 12 months = 2025-07-26."""
    from ghost_agent.memory import temporal as tp
    assert tp.anchor("Thodoris is 1 year and 40 days old", said_at=dt.date(2026, 9, 4)) == "Thodoris born ~2025-07-26"
    assert tp.anchor("Leonidas is 5 months and 3 weeks old", said_at=dt.date(2026, 9, 4)) == "Leonidas born ~2026-03-14"


def test_trailing_whitespace_does_not_stall_the_writer():
    """Rounds 3–4 m6: the gloss stripper's leading `\\s*` and the
    `\\s*[-\\s]\\s*old` idiom of the compound and single-unit age patterns
    were quadratic on a whitespace run — measured 11.8 s (compound) and
    0.6 s (single) on 16 000 spaces NOT followed by "old", and 7.5 s for
    the "9-year old" pattern (the run before a group). A run followed by
    "old" lets the engine skip the backtracking, which is why round 3's
    version of this pin (spaces after "old") could not tell the idioms
    apart — the round-4 reviewer's inputs can. Linear now: ~10 ms each."""
    import time
    from ghost_agent.memory import temporal as tp
    for text, head in ((("Leonidas is 5 months and 23 days" + " " * 16000 + "olx"), "Leonidas is 5 months and 23 days"),
                       (("Leonidas is 5 months" + " " * 8000 + "x"), "Leonidas is 5 months"),
                       (("Leonidas is 9" + " " * 16000 + "x"), "Leonidas is 9"),            # a count, a run, no unit: the "9-year old" idiom
                       (("Leonidas is 5 months and 23 days old" + " " * 8000), "Leonidas born ~2026-03-12")):
        t0 = time.perf_counter()
        out = tp.anchor(text, said_at=dt.date(2026, 9, 4))
        assert time.perf_counter() - t0 < 0.25, text[:30]
        assert out.startswith(head), out[:60]


def test_the_override_report_counts_a_chain_under_each_tag(tmp_path, capsys, monkeypatch):
    """Round 3 m5: the report grouped by the exact override string, so a
    chained row ('reply-shape+web-exec') left both buckets. Executed on a
    scratch home."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "verdict_override_report",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "verdict_override_report.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    vdir = tmp_path / "system" / "verdicts"; vdir.mkdir(parents=True)
    rows = [{"trajectory_id": "t1", "verdict": "REFUTED", "override": "reply-shape+web-exec", "at": "2026-09-08T10:00:00", "seq": 1},
            {"trajectory_id": "t2", "verdict": "CONFIRMED", "override": "", "at": "2026-09-08T10:01:00", "seq": 2}]
    (vdir / "2026-09-08.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["verdict_override_report.py", "--home", str(tmp_path)])
    assert mod.main() == 0
    out = capsys.readouterr().out
    buckets = {line.split()[0]: line for line in out.splitlines() if line and not line.startswith(("override", "NOTE")) and line[0] != " "}
    assert "reply-shape" in buckets and "web-exec" in buckets and "(text" in buckets, out
    assert "reply-shape+web-exec" not in buckets and "1 chained (reply-shape+web-exec=1)" in out, out
    assert mod.override_tags({"override": "reply-shape+web-exec"}) == ["reply-shape", "web-exec"]
    assert mod.override_tags({"override": None}) == ["(text judge)"]


@pytest.mark.asyncio
async def test_a_visual_refute_carries_its_provenance(mock_context, tmp_path, monkeypatch):
    """Round 3 m5: the VISUAL arm replaced the verdict and stamped nothing,
    so a visual refute was reported as '(text judge)'. At the real site: a
    raw-dump reply whose vision check REFUTES reaches the sidecar as
    'visual+reply-shape' — the VISUAL arm runs first at the site and the
    reply-shape override merges into it, and the chain records that order."""
    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    from ghost_agent.distill.collector import TrajectoryCollector
    agent = GhostAgent(mock_context)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")
    verifier = MagicMock(); verifier.llm_client = MagicMock()
    async def fake_verify(*a, **k):
        return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    for name in ("verify_response", "verify", "verify_turn", "run", "verify_claim", "verify_code_output"):
        setattr(verifier, name, fake_verify)
    async def fake_visual(**k):
        return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="the button is still red", issues=["the button is still red"])
    verifier.verify_visual = fake_visual
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    agent._scoped_sandbox_for = lambda pid: None
    monkeypatch.setattr(agent_mod, "_is_visual_intent", lambda t: True)
    monkeypatch.setattr(agent_mod, "_select_visual_evidence", lambda *a, **k: (None, "after.png"))
    tools = [{"name": "execute", "arguments": {"command": "python x.py"}, "content": "EXIT CODE: 0\nout", "result": "EXIT CODE: 0\nout"}]
    res, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=[{"role": "user", "content": "make the button green"}],
        final_ai_content=LIVE_DUMP, last_user_content="make the button green", lc="make the button green",
        req_id="v1", trajectory_id="v1")
    assert res is not None and res.verdict == VerifyVerdict.REFUTED
    assert getattr(res, "override", "") == "visual+reply-shape", getattr(res, "override", "")
    rows = [json.loads(l) for f in (tmp_path / "verdicts").glob("*.jsonl") for l in f.read_text().splitlines() if l.strip()]
    assert rows and rows[-1]["trajectory_id"] == "v1" and rows[-1].get("override") == "visual+reply-shape", rows[-1:]


@pytest.mark.asyncio
async def test_a_web_exec_refute_keeps_the_reply_shape_stamp(mock_context, tmp_path, monkeypatch):
    """The chain at the real site: reply-shape stamps first, then WEB-EXEC
    replaces the verdict with its own and must carry the earlier tag —
    'reply-shape+web-exec'. World where it fails: the WEB-EXEC arm stamps a
    bare tag (the pre-§4FN code) or reads the previous tag from the new
    object (always empty)."""
    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    from ghost_agent.distill.collector import TrajectoryCollector
    agent = GhostAgent(mock_context)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")
    verifier = MagicMock(); verifier.llm_client = MagicMock()
    async def fake_verify(*a, **k):
        return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    for name in ("verify_response", "verify", "verify_turn", "run", "verify_claim", "verify_code_output", "verify_visual"):
        setattr(verifier, name, fake_verify)
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    monkeypatch.setattr(agent_mod, "_web_artifacts_written", lambda tools: ["index.html"])
    async def fake_exec(written, project_id=None):
        return ("index.html", "TypeError: Cannot read properties of undefined")
    agent._execute_web_artifact = fake_exec
    tools = [{"name": "write_file", "arguments": {"path": "index.html", "content": "<html>"}, "content": "written", "result": "written"}]
    res, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=[{"role": "user", "content": "build the page"}],
        final_ai_content=LIVE_DUMP, last_user_content="build the page", lc="build the page",
        req_id="w1", trajectory_id="w1")
    assert res is not None and res.verdict == VerifyVerdict.REFUTED
    assert any("uncaught JS exception" in i for i in res.issues), res.issues
    assert getattr(res, "override", "") == "reply-shape+web-exec", getattr(res, "override", "")
    rows = [json.loads(l) for f in (tmp_path / "verdicts").glob("*.jsonl") for l in f.read_text().splitlines() if l.strip()]
    assert rows and rows[-1]["trajectory_id"] == "w1" and rows[-1].get("override") == "reply-shape+web-exec", rows[-1:]


# --- round 4: what the link must NOT admit -----------------------------------
#
# The round-4 reviewer generated 63 phrasings of its own and executed them.
# Seven false-refute classes came out of the whitelist itself: tense words
# ("was", "turns", "at", "in", "on" — a past or future age compared with
# today), "of" (the age belongs to the noun BEFORE "of Thodoris"), a bare
# possessive ("Thodoris's 5 month old brother"), "was born … is" chains,
# the forward candidate outranking the phrase's own bracket, decimals read
# from their fraction, and week/day claims compared on the month scale.
# Every word left on the list is pinned in BOTH directions below, and the
# catalogue is asserted to cover the list — a word added without a pin
# fails the suite.

LINK_CATALOGUE = {
    "is": ("Thodoris is 9 years old", "Thodoris is 5 months old"),
    "'s age": ("Thodoris's age is 9 years old", "Thodoris's age is 5 months old"),
    "’s age": ("Thodoris’s age is 9 years old", "Thodoris’s age is 5 months old"),
    "' age": ("Thodoris' age: 9 years old", "Thodoris' age: 5 months old"),
    "age": ("Thodoris age 9 years old", "Thodoris age 5 months old"),
    "aged": ("Thodoris aged 9 years old", "Thodoris aged 5 months old"),
    "now": ("Thodoris — now 9 years old", "Thodoris — now 5 months old"),
    "currently": ("Thodoris is currently 9 years old", "Thodoris is currently 5 months old"),
    "today": ("Thodoris is today 9 years old", "Thodoris is today 5 months old"),
    "just": ("Leonidas is just 5 months old", "Leonidas is just 9 years old"),
    "already": ("Thodoris is already 9 years old", "Thodoris is already 5 months old"),
    "still": ("Leonidas is still 5 months old", "Leonidas is still 9 years old"),
    "only": ("Leonidas is only 5 months old", "Leonidas is only 9 years old"),
    "about": ("Thodoris is about 9 years old", "Thodoris is about 5 months old"),
    "around": ("Thodoris is around 9 years old", "Thodoris is around 5 months old"),
    "roughly": ("Thodoris is roughly 9 years old", "Thodoris is roughly 5 months old"),
    "approximately": ("Thodoris is approximately 9 years old", "Thodoris is approximately 5 months old"),
    "nearly": ("Thodoris is nearly 10 years old", "Thodoris is nearly 5 months old"),
    "almost": ("Thodoris is almost 10 years old", "Thodoris is almost 5 months old"),
    "exactly": ("Thodoris is exactly 9 years old", "Thodoris is exactly 5 months old"),
    "over": ("Thodoris is over 9 years old", "Thodoris is over 5 months old"),
    "under": ("Thodoris is under 10 years old", "Thodoris is under 5 months old"),
    "who": ("Thodoris, who is 9 years old", "Thodoris, who is 5 months old"),
    "born": ("Thodoris, born 2016-11-25, is 9 years old", "Thodoris, born 2016-11-25, is 5 months old"),
    "b": ("Thodoris (b. 2016-11-25) is 9 years old", "Thodoris (b. 2016-11-25) is 5 months old"),
}


def test_the_link_catalogue_covers_every_link_word():
    """A word on `_LINK_WORDS` without a two-direction pin is the round-3
    survivor list (31 words, none exercised)."""
    import re as _re
    words = {w.replace("\\s+", " ") for w in mcc._LINK_WORDS.split("|")}
    assert words == set(LINK_CATALOGUE), words ^ set(LINK_CATALOGUE)


@pytest.mark.parametrize("word", sorted(LINK_CATALOGUE))
def test_each_link_word_binds_in_both_directions(word):
    correct, wrong = LINK_CATALOGUE[word]
    assert refute_age_claims(reply=correct, profile=PROFILE, now=NOW) == [], (word, correct)
    issues = refute_age_claims(reply=wrong, profile=PROFILE, now=NOW)
    import re as _re
    assert issues and issues[0].startswith(_re.match(r"\W*([^\W\d_]+)", wrong).group(1)), (word, issues)


@pytest.mark.parametrize("sep", [":", " —", " –", " -", ",", " =", " ->", " →", " |"])
def test_each_link_separator_binds_in_both_directions(sep):
    assert refute_age_claims(reply=f"Thodoris{sep} 9 years old", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply=f"Thodoris{sep} 5 months old", profile=PROFILE, now=NOW)[0].startswith("Thodoris")


DATE_SHAPES = ("2016-11-25", "25/11/2016", "25.11.2016", "November 25, 2016", "Nov. 25 2016",
               "25 November 2016", "25th Nov 2016", "2016")


@pytest.mark.parametrize("date", DATE_SHAPES)
def test_each_date_shape_is_a_link(date):
    assert refute_age_claims(reply=f"Thodoris (born {date}) is 9 years old", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply=f"Thodoris (born {date}) is 5 months old", profile=PROFILE, now=NOW)[0].startswith("Thodoris")


@pytest.mark.parametrize("month", ["January", "Jan", "February", "Feb", "March", "Mar", "April", "Apr", "May",
                                   "June", "Jun", "July", "Jul", "August", "Aug", "September", "Sep", "Sept",
                                   "October", "Oct", "November", "Nov", "December", "Dec"])
def test_each_month_name_is_a_link(month):
    """Round-3 survivors: ten month names unexercised (only March and Sep
    were pinned)."""
    assert refute_age_claims(reply=f"Thodoris, born {month} 25, 2016, is 5 months old", profile=PROFILE, now=NOW)[0].startswith("Thodoris")
    assert refute_age_claims(reply=f"Thodoris (b. {month} 2016) is 9 years old", profile=PROFILE, now=NOW) == []


def test_a_bare_count_in_the_gap_is_not_a_date():
    """Round 4 M4: "| Thodoris | 2 | 3 years old |" is a count of bikes."""
    assert refute_age_claims(reply="| Thodoris | 2 | 3 years old |", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply="| Thodoris | 2016-11-25 | 3 years old |", profile=PROFILE, now=NOW)


PAST_OR_FUTURE = (
    "Thodoris was about 4 years old when you moved in 2021.",
    "When Thodoris was 3 years old you moved to Athens.",
    "Leonidas was 2 months old in May.",
    "Thodoris turned 5 years old in November 2021.",
    "Leonidas at 3 months old probably started smiling.",
    "Here is a photo of Thodoris at 2 years old.",
    "Leonidas turns 7 months old on October 12.",
    "Thodoris turns 11 years old in November 2027.",
    "Thodoris in 2036: 19 years old.",
    "By March Leonidas is 12 months old.",
    "Leonidas at 12 months old will start walking.",
    "Thodoris will be 10 years old soon.",
    "The house in which Thodoris was born is 40 years old.",
    "The hospital Leonidas was born in is 30 years old.",
)


@pytest.mark.parametrize("reply", PAST_OR_FUTURE)
def test_a_past_or_future_age_is_not_compared_with_today(reply):
    """Round 4 C1/C4. World where it fails: "was", "turns", "at", "in", "on"
    are links, or the marker before the name is invisible to a gap rule."""
    assert refute_age_claims(reply=reply, profile=PROFILE, now=NOW) == []


def test_a_present_age_beside_a_date_word_still_binds():
    """"today" is not a tense marker; the tense rule is not a blanket mute."""
    assert refute_age_claims(reply="Leonidas is 5 months old today", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply="Thodoris is 5 months old today", profile=PROFILE, now=NOW)[0].startswith("Thodoris")


@pytest.mark.parametrize("text", [
    "Thodoris was 5 years old when they moved in 2021",
    "Leonidas was 3 months old in June",
    "Leonidas turns 7 months old on October 12",
    "Thodoris will be 10 years old soon",                   # "will" alone
    "the photo of Thodoris at 2 years old",                  # "at N" alone
    "Thodoris in 2036: 19 years old",                        # "in YYYY" alone
    "Leonidas in June 2026: 3 months old",                   # "in Month YYYY" alone
    "Thodoris was 9 years and 2 months old when they moved",
])
def test_the_writer_never_anchors_a_past_or_future_age(text):
    """Round 4 C1 (writer half): "Thodoris was 5 years old when they moved
    in 2021" was stored as `born ~2021-03` and then refuted the correct
    "Thodoris is 9 years old" forever."""
    from ghost_agent.memory import temporal as tp
    assert tp.anchor(text, said_at=NOW) == text


NOT_THIS_PERSONS_AGE = (
    "The father of Thodoris is 46 years old.",
    "The older brother of Leonidas is 9 years old.",
    "The father of Thodoris, who is 46 years old, works from home.",
    "Vasilis, father of Leonidas, is 46 years old.",
    "Leonidas is the baby brother of Thodoris, 5 months old.",
    "Thodoris's 5 month old brother Leonidas is doing great.",
    "Thodoris's 5 months old brother is teething.",
)


@pytest.mark.parametrize("reply", NOT_THIS_PERSONS_AGE)
def test_of_and_a_bare_possessive_do_not_hand_the_age_to_the_name(reply):
    """Round 4 C2/C3."""
    assert refute_age_claims(reply=reply, profile=PROFILE, now=NOW) == []


def test_the_reader_skips_a_date_stored_after_of_or_a_possessive():
    """Round 4 C2/C3, the loop: what the writer stores for "The brother of
    Leonidas is 9 years old" must not become Leonidas's birth date."""
    from ghost_agent.memory import temporal as tp
    for text, correct in (("The brother of Leonidas is 9 years old", "Leonidas is 5 months old."),
                          ("The father of Thodoris is 46 years old", "Thodoris is 9 years old."),
                          ("Vasilis, father of Thodoris, is 46 years old", "Thodoris is 9 years old."),
                          ("Thodoris's 5 month old brother", "Thodoris's age is 9 years old.")):
        stored = tp.anchor(text, said_at=NOW)
        assert "born ~" in stored, stored
        assert mcc.anchored_subjects({"k": stored}) == [], stored
        assert refute_age_claims(reply=correct, profile={"k": stored}, now=NOW) == [], stored


def test_one_name_with_two_stored_dates_is_bad_data_not_evidence():
    """Round 4 C2: the verdict depended on dict order."""
    a, b = "Thodoris (born 2016-11-25)", "Thodoris (born 1980-01-29)"
    for prof in ({"a": a, "b": b}, {"b": b, "a": a}):
        assert refute_age_claims(reply="Thodoris is 9 years old.", profile=prof, now=NOW) == []
        assert refute_age_claims(reply="Thodoris is 46 years old.", profile=prof, now=NOW) == []


def test_a_bracketed_name_is_an_annotation_of_the_phrase_before_it():
    """Round 4 C5: "9 years old (Thodoris), 5 months old (Leonidas)" bound
    the second age to Thodoris because the forward link won."""
    for reply in ("Your sons are 9 years old (Thodoris), 5 months old (Leonidas).",
                  "| 9 years old (Thodoris) | 5 months old (Leonidas) |",
                  "Ages: 9 years old (Thodoris); 5 months old (Leonidas).",
                  "9 years old (Thodoris) / 5 months old (Leonidas)"):
        assert refute_age_claims(reply=reply, profile=PROFILE, now=NOW) == [], reply
    swapped = refute_age_claims(reply="Your sons are 5 months old (Thodoris), 9 years old (Leonidas).", profile=PROFILE, now=NOW)
    assert {i.split()[0] for i in swapped} == {"Thodoris", "Leonidas"}, swapped
    # a markdown link and an aside are not annotations
    assert refute_age_claims(reply="The team page is 2 years old [Thodoris](https://example.org/t)", profile=PROFILE, now=NOW) == []
    assert refute_age_claims(reply="The cot is 20 years old (a hand-me-down) (Leonidas)", profile=PROFILE, now=NOW) == []


def test_a_decimal_age_is_one_number():
    """Round 4 C6: "9.5 years old" was read as FIVE years and "5.8 months"
    — the checker's own rendering, quoted back — as eight."""
    for reply in ("Thodoris is 9.5 years old.", "Leonidas is 5.8 months old.", "Leonidas is about 5.8 months old.",
                  "Thodoris is 9,5 years old.", "Thodoris is 9 1/2 years old.", "Thodoris is 9-10 years old."):
        assert refute_age_claims(reply=reply, profile=PROFILE, now=NOW) == [], reply
    issues = refute_age_claims(reply="Leonidas is 15.5 months old.", profile=PROFILE, now=NOW)
    assert issues and "15.5" in issues[0], issues
    from ghost_agent.memory import temporal as tp
    for text in ("Leonidas is 5.8 months old", "Thodoris is 9-10 years old", "Thodoris is 9 1/2 years old",
                 "Thodoris is 1.5 years and 2 months old"):
        assert tp.anchor(text, said_at=NOW) == text, text


def test_week_and_day_claims_are_compared_on_their_own_scale():
    """Round 4 C7: a 176-day-old child "is 26 weeks old" (25 w 1 d) and
    "177 days old" were refuted by the calendar-month scale."""
    for reply in ("Leonidas is 26 weeks old.", "Leonidas is 25 weeks old.", "Leonidas is 177 days old.",
                  "Leonidas is 176 days old.", "Leonidas is 180 days old.", "Leonidas is roughly 180 days old."):
        assert refute_age_claims(reply=reply, profile=PROFILE, now=NOW) == [], reply
    for reply in ("Leonidas is 40 weeks old.", "Leonidas is 100 days old."):
        issues = refute_age_claims(reply=reply, profile=PROFILE, now=NOW)
        assert issues and issues[0].endswith("176 days old today"), issues


def test_one_unit_old_does_not_admit_a_newborn():
    """Round 4 M5: N=1 gave a lower bound of zero."""
    assert refute_age_claims(reply="Leonidas is 1 year old.", profile=PROFILE, now=NOW)[0].startswith("Leonidas")
    assert refute_age_claims(reply="Leonidas is 1 month old.", profile=PROFILE, now=dt.date(2026, 4, 2)) == []   # 3 weeks


def test_a_sentence_opener_before_a_comma_is_not_a_list():
    """Round 4 M2: "Yes, Thodoris", "Today, Leonidas" were read as lists
    and muted the checker AND the writer."""
    assert refute_age_claims(reply="Yes, Thodoris is 15 years old.", profile=PROFILE, now=NOW)[0].startswith("Thodoris")
    assert refute_age_claims(reply="Today, Leonidas is 9 years old.", profile=PROFILE, now=NOW)[0].startswith("Leonidas")
    assert refute_age_claims(reply="Hi Vasilis, Thodoris is 15 years old.", profile=PROFILE, now=NOW)[0].startswith("Thodoris")
    # two ANCHORED names joined by a comma are a list
    assert refute_age_claims(reply="Thodoris, Leonidas: 9 years and 5 months old", profile=PROFILE, now=NOW) == []
    from ghost_agent.memory import temporal as tp
    assert tp.anchor("Today, Leonidas is 5 months old", said_at=NOW) == "Today, Leonidas born ~2026-03-20"
    # the writer keeps no comma lists; the reader is the defence for what it stores
    stored = tp.anchor("Thodoris, Leonidas: 9 years and 5 months old", said_at=NOW)
    assert stored == "Thodoris, Leonidas: born ~2017-04-04"
    assert mcc.anchored_subjects({"k": stored}) == []


def test_a_plural_list_far_back_on_the_line_still_guards_a_compound():
    """Round 4 M3: an 80-character look-back let a list 90 characters back
    glue two children to one date; a single-unit age later on such a line
    is still one person's."""
    from ghost_agent.memory import temporal as tp
    text = ("Thodoris and Leonidas are, as of today (September 4, 2026, which is a Friday afternoon), "
            "respectively 9 years and 5 months old")
    assert tp.anchor(text, said_at=NOW) == text
    assert tp.anchor("Thodoris and Leonidas are brothers and the younger one Leonidas is 5 months old", said_at=NOW) \
        == "Thodoris and Leonidas are brothers and the younger one Leonidas born ~2026-03-20"
    assert tp.anchor("Thodoris and Leonidas\nMax is 3 years old", said_at=NOW) == "Thodoris and Leonidas\nMax born ~2023-03"
    assert tp.anchor("Thodoris and Leonidas: ages 9 years and 5 months old", said_at=NOW) == "Thodoris and Leonidas: ages 9 years and 5 months old"
    assert tp.anchor("The boys are now about 9 years and 5 months old", said_at=NOW) == "The boys are now about 9 years and 5 months old"
    # a list with NO plural copula after it is not a plural line: grandma's age is grandma's
    out = tp.anchor("Thodoris and Leonidas love their grandma, who is 70 years and 2 months old", said_at=NOW)
    assert out.endswith("grandma, who born ~1956-07-04"), out


def test_names_match_case_insensitively_and_whole():
    assert refute_age_claims(reply="leonidas is 9 years old.", profile=PROFILE, now=NOW)[0].startswith("Leonidas")
    assert refute_age_claims(reply="Leonidas is 5 months old.", profile={"k": "Nidas (born 2016-11-25)"}, now=NOW) == []


@pytest.mark.asyncio
async def test_a_tool_free_verdict_reaches_the_sidecar(mock_context, tmp_path):
    """Round 4 M1: the tool-free branch returned before the recorder, so the
    arithmetic route — the one the override report exists to measure — and
    the tool-free reply-shape refute never reached the sidecar or carried
    an override stamp. At the real site, with no tools run."""
    from types import SimpleNamespace
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyVerdict
    from ghost_agent.distill.collector import TrajectoryCollector
    agent = GhostAgent(mock_context)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")
    agent.context.profile_memory = SimpleNamespace(load=lambda: PROFILE)
    verifier = MagicMock(); verifier.llm_client = MagicMock()
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    async def run(reply, rid):
        return await agent._compute_verifier_verdict(
            tools_run_this_turn=[], messages=[{"role": "user", "content": "how old are they?"}],
            final_ai_content=reply, last_user_content="how old are they?", lc="how old are they?",
            req_id=rid, trajectory_id=rid)
    def rows():
        return [json.loads(l) for f in (tmp_path / "verdicts").glob("*.jsonl") for l in f.read_text().splitlines() if l.strip()]
    res, _ = await run("Thodoris is 5 months old.", "m1")
    assert res is not None and res.verdict == VerifyVerdict.REFUTED
    assert getattr(res, "override", "") == "memory-claim"
    last = rows()[-1]
    assert last["trajectory_id"] == "m1" and last.get("override") == "memory-claim", last
    res2, _ = await run(LIVE_DUMP, "m2")
    assert res2 is not None and res2.verdict == VerifyVerdict.REFUTED and getattr(res2, "override", "") == "reply-shape"
    last = rows()[-1]
    assert last["trajectory_id"] == "m2" and last.get("override") == "reply-shape", last
    # a correct tool-free reply records nothing (refute-only, and no verdict to record)
    n = len(rows())
    res3, _ = await run("Thodoris is 9 years old.", "m3")
    assert res3 is None and len(rows()) == n
