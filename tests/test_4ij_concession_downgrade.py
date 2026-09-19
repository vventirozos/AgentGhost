"""§4IJ (part 3) — a CONFIRMED that names a discrepancy is not a confirm.

The strong judge WRITES the problem and confirms over it. Run-7 IFS reply
(radians labelled as degrees): "the only minor issue is a slightly loose
'~10 km spacing' characterization given the nearest point is 86 km away,
but this does not refute the core answer" → CONFIRMED 0.9, issues [].
Bench weather case under the new rules: "two evidence rows report 34°C
and 35°C … the CLAIM picked 34°C without noting the disagreement" →
CONFIRMED 0.95, and that sentence sat in `issues`. Bench arm A: 12
overturns were CONFIRMED with non-empty issues — every one on a fault
trial, none on a clean one.

Now both deciding prompts ask for the concession as a structured
`conceded` list, and `_build_verify_result` — the ONE builder every path
uses — turns a CONFIRMED that carries a conceded discrepancy or an issue
into UNCERTAIN capped at `_CONFIRM_WITHHELD_CONF_CAP`: never an
actionable positive, never a punitive refute. On an escalation the
strong UNCERTAIN takes the existing `replaced_uncertain` path.

World where each pin fails: the builder ships the contradiction as a
0.9+ CONFIRMED again, the cap moves above the actionable gate, a REFUTED
is touched, the field is dropped from a prompt's contract, the flag goes
invisible to the bench, or the code route (its own contract) gets a
field it was never asked to fill.
"""
import ast
import inspect

import pytest

from ghost_agent.core import verifier as V
from ghost_agent.core.verifier import VerifyVerdict
from ghost_agent.eval import verify_bench as B

CAP = V._CONFIRM_WITHHELD_CONF_CAP


def _v():
    return V.Verifier.__new__(V.Verifier)


@pytest.mark.parametrize("data,expect_verdict,expect_conf,expect_conceded", [
    # the run-7 shape: confirmed, issues empty, concession structured
    ({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "answers it",
      "issues": [], "conceded": ["'~10 km spacing' beside a nearest point 86 km away"]},
     VerifyVerdict.UNCERTAIN, CAP, ["'~10 km spacing' beside a nearest point 86 km away"]),
    # the weather shape: confirmed with the discrepancy in issues, no conceded field
    ({"verdict": "CONFIRMED", "confidence": 0.95, "reasoning": "supported",
      "issues": ["Minor conflict: two rows report 34°C and 35°C; the CLAIM picked 34°C."]},
     VerifyVerdict.UNCERTAIN, CAP, ["Minor conflict: two rows report 34°C and 35°C; the CLAIM picked 34°C."]),
    # a string instead of a list still counts
    ({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "r", "issues": [], "conceded": "34 vs 35"},
     VerifyVerdict.UNCERTAIN, CAP, ["34 vs 35"]),
    # a clean confirm is untouched
    ({"verdict": "CONFIRMED", "confidence": 0.98, "reasoning": "r", "issues": [], "conceded": []},
     VerifyVerdict.CONFIRMED, 0.98, None),
    ({"verdict": "CONFIRMED", "confidence": 0.98, "reasoning": "r"},
     VerifyVerdict.CONFIRMED, 0.98, None),
    # blank entries are not concessions
    ({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "r", "issues": ["  "], "conceded": [None, ""]},
     VerifyVerdict.CONFIRMED, 0.9, None),
    # a REFUTED keeps its verdict and confidence whatever it conceded
    ({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "bad", "issues": ["x"], "conceded": ["y"]},
     VerifyVerdict.REFUTED, 0.9, None),
    # an UNCERTAIN is left alone (already below the gate)
    ({"verdict": "UNCERTAIN", "confidence": 0.5, "reasoning": "?", "issues": [], "conceded": ["z"]},
     VerifyVerdict.UNCERTAIN, 0.5, None),
    # a low confirm is not raised by the cap
    ({"verdict": "CONFIRMED", "confidence": 0.4, "reasoning": "r", "conceded": ["w"]},
     VerifyVerdict.UNCERTAIN, 0.4, ["w"]),
])
def test_builder_downgrades_a_confirm_that_names_a_discrepancy(data, expect_verdict, expect_conf, expect_conceded):
    r = _v()._build_verify_result(dict(data), strong=True)
    # `.value`, not identity: another test file reloads `verifier` in the
    # same worker and leaves two VerifyVerdict classes alive.
    assert r.verdict.value == expect_verdict.value
    assert r.confidence == pytest.approx(expect_conf)
    assert r.conceded == expect_conceded
    if expect_conceded:
        assert r.reasoning.startswith("conceded discrepancy: ")
        assert r.to_dict()["conceded"] == expect_conceded
    else:
        assert "conceded" not in r.to_dict()


def test_cap_sits_below_the_actionable_gate():
    assert CAP < 0.7


def test_flag_off_restores_the_raw_verdict(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_CONCESSION_DOWNGRADE", "0")
    r = _v()._build_verify_result({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "r",
                                   "issues": [], "conceded": ["34 vs 35"]}, strong=True)
    assert r.verdict.value == "CONFIRMED" and r.confidence == 0.9 and r.conceded is None


def test_the_cheap_judge_is_never_downgraded():
    """The cheap judge concedes nitpicks on half of the CLEAN replies
    (bench arm D, first cut: 3 of the first 6 clean turns went UNCERTAIN
    with no second opinion). Only a STRONG verdict is downgraded; a cheap
    CONFIRMED that concedes stays CONFIRMED, as it did before."""
    r = _v()._build_verify_result({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "r",
                                   "issues": [], "conceded": ["omits the wind gusts"]})
    assert r.verdict.value == "CONFIRMED" and r.confidence == 0.9 and r.conceded is None
    r = _v()._build_verify_result({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "r",
                                   "issues": ["minor"]}, strong=False)
    assert r.verdict.value == "CONFIRMED"


def test_only_the_main_model_sites_pass_strong():
    """AST: `strong=True` is passed at the classic re-judgement sites that
    follow a `force_main=True` call (refute escalation, the truncation /
    confirm re-judge); the two-stage stage-2 builder passes
    `strong=bool(force_main)`; the cheap paths pass nothing."""
    tree = ast.parse(inspect.getsource(V))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "_build_verify_result"]
    strong_true = [c for c in calls if any(k.arg == "strong" and isinstance(k.value, ast.Constant)
                                           and k.value.value is True for k in c.keywords)]
    strong_dyn = [c for c in calls if any(k.arg == "strong" and not isinstance(k.value, ast.Constant)
                                          for k in c.keywords)]
    plain = [c for c in calls if not any(k.arg == "strong" for k in c.keywords)]
    assert len(strong_true) == 3                       # refute escalation, confirm re-judge, code re-verify
    assert len(strong_dyn) == 1 and "force_main" in ast.unparse(strong_dyn[0])
    assert len(plain) >= 3


@pytest.mark.parametrize("value,expected", [("1", True), ("", True), ("0", False), ("no", False)])
def test_flag_parsing(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_VERIFY_CONCESSION_DOWNGRADE", raising=False)
    else:
        monkeypatch.setenv("GHOST_VERIFY_CONCESSION_DOWNGRADE", value)
    assert V._concession_downgrade_enabled() is expected


def test_every_deciding_prompt_asks_for_the_conceded_list():
    """Claim, adjudicate AND code (run 8 of the IFS ask went through the
    code route: `verify code — CONFIRMED 0.75` on a reply listing a
    latitude of 128°). The visual prompt judges pixels and is left alone."""
    for name in ("_VERIFY_CLAIM_PROMPT", "_VERIFY_ADJUDICATE_PROMPT", "_VERIFY_CODE_PROMPT"):
        assert '"conceded": [' in getattr(V, name), name
        assert "judged too minor to refute" in getattr(V, name), name
    assert '"conceded"' not in V._VERIFY_VISUAL_PROMPT
    assert "INTERNAL CONTRADICTION or IMPOSSIBLE VALUES" in V._VERIFY_CODE_PROMPT
    assert "a latitude of 128°" in V._VERIFY_CODE_PROMPT
    assert V._template_reject_reason("verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT) == ""
    assert V._VERIFY_CODE_PROMPT.format(intent="i", code="c", output="o", response="r")


def test_the_builder_is_the_one_authority():
    """Every VerifyResult constructed from judge JSON goes through
    `_build_verify_result` — no site builds one from `data` directly, so
    the downgrade cannot be bypassed by a route."""
    tree = ast.parse(inspect.getsource(V))
    direct = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "VerifyResult":
            kws = {k.arg: k.value for k in n.keywords}
            v = kws.get("verdict")
            # a construction whose verdict is read straight off a `data`-like dict
            if v is not None and "data" in ast.unparse(v) and "get(" in ast.unparse(v):
                direct.append(ast.unparse(n)[:80])
    assert direct == [], direct


def test_bench_provenance_records_the_flag():
    src = inspect.getsource(B.bench_provenance)
    consts = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "GHOST_VERIFY_CONCESSION_DOWNGRADE" in consts
