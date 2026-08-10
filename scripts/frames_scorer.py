#!/usr/bin/env python3
"""Grade FRAMES answers — the way FRAMES is actually graded.

⚠ WHY NOT THE GAIA SCORER. It was the first choice, and the readiness smoke
caught it producing a FABRICATED ZERO. Measured 2026-08-10, both scored wrong,
both semantically correct:

    ground truth "Mulona barnesi and mulona schausi"
    agent        "Mulona barnesi, Mulona schausi"        -> WRONG
    ground truth "5 minutes and 31 seconds"
    agent        "5:31"                                  -> WRONG

Reported accuracy 0.0 on two correct answers. The cause is not a bug in the
GAIA scorer: it assumes GAIA's mandated answer FORMAT on *both* sides (comma
lists, no units, digits plain), and FRAMES ground truth is ordinary natural
language. FRAMES' own paper grades with an LLM judge, so exact match was never
the protocol — using it would have understated the agent and published a
number that measured the ruler.

⚠ AND WHY NOT HAND-TUNED NORMALISATION. The tempting fix is a pile of rules —
treat " and " as a comma, canonicalise durations, strip units. Every one of
those rules would have been written by looking at MY agent's failures, which
makes it a scorer tuned to raise my agent's score. That is the instrument
flattering the thing it measures. One normalisation survives, because it is
defensible a priori rather than derived from a failure: " and " is the
natural-language equivalent of the list separator GAIA already splits on. Case
2 above genuinely requires semantic judgement, and no defensible amount of
string munging reaches it.

JUDGE INDEPENDENCE. The answering model is Qwen3.6-35B (main); the judge is
the critic node (Gemma 4 E4B) — a DIFFERENT model family, so this is not
self-grading. That matters more than usual here: the answerer cannot vouch for
itself. The judge task ("do these two mean the same thing, given the reference")
is also far narrower than the 5.4KB adjudication prompt whose capacity limits
§4Z documented.

CONSERVATIVE BY CONSTRUCTION. The judge is instructed to answer NO when
uncertain, and any unparseable reply counts as NO. A scoreboard that errs
should err DOWNWARD — an understated number invites a re-measure, an
overstated one gets quoted.

Every judgement is returned with its reason so a run is auditable after the
fact, and `validate_judge()` measures the judge against hand labels before any
headline number is quoted.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaia_scorer import _normalize_str, _split_string  # noqa: E402

JUDGE_PROMPT = (
    "You are grading one short factual answer against a reference answer.\n\n"
    "QUESTION: {question}\n"
    "REFERENCE ANSWER: {truth}\n"
    "CANDIDATE ANSWER: {model}\n\n"
    "Does the CANDIDATE convey the same factual content as the REFERENCE?\n"
    "Formatting differences do not matter: word order, punctuation, case, "
    "units written out vs abbreviated, '5:31' vs '5 minutes and 31 seconds', "
    "'A, B' vs 'A and B'.\n"
    "A candidate that is missing part of the reference, adds a contradictory "
    "claim, or names a different entity is NOT a match.\n"
    "If you are uncertain, answer NO.\n\n"
    "Reply with exactly one word: YES or NO."
)

_VERDICT = re.compile(r"\b(yes|no)\b", re.I)


def nova_judge_ask(base_url: str = "http://100.83.184.117:8088",
                   timeout: float = 90.0) -> Callable[[str], str]:
    """Transport for the INDEPENDENT judge (critic node, Gemma 4 E4B).

    ⚠⚠ BOTH NO-THINK SWITCHES ARE LOAD-BEARING. Measured 2026-08-10: without
    them E4B returns **empty content with finish_reason='length'** at every
    max_tokens tried (8 and 200) — it spends the budget on thinking tokens
    that never surface. Every judgement then parsed as unreadable and, per the
    conservative rule, became NO.

    The damage that does is subtle and is why this comment is long: the
    validation still LOOKED sane. Zero false positives, agreement 0.69 — an
    unremarkable "the small judge is a bit weak" reading. It was not weak; it
    was ABSENT, and the number was entirely `strict_match` doing the work.
    With the switches, agreement on the same 16 hand-labelled pairs is
    **1.0 (0 FP, 0 FN)**.

    So if these are ever removed, the scoreboard does not break loudly — it
    quietly reports a much lower score and reads as the AGENT regressing.
    `tests/test_frames_scorer.py` pins both.
    """
    import urllib.request

    def ask(prompt: str) -> str:
        body = json.dumps({
            "model": "local",
            # soft switch
            "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
            "max_tokens": 16, "temperature": 0,
            # hard switch — the repo pairs these everywhere (tools/vision.py,
            # main.py); either alone has been observed to be insufficient.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    return ask


def strict_match(model_answer, ground_truth: str) -> bool:
    """Deterministic pre-pass. Cheap, reproducible, and never the last word.

    The ONLY normalisation beyond GAIA's is treating " and " as a list
    separator — the natural-language form of the comma GAIA already splits on,
    and defensible without reference to any observed failure.
    """
    if model_answer is None:
        return False

    def parts(s: str) -> list[str]:
        s = re.sub(r"\s+and\s+", ",", str(s), flags=re.I)
        return sorted(_normalize_str(p) for p in _split_string(s) if p.strip())

    if parts(model_answer) == parts(ground_truth):
        return True
    return _normalize_str(str(model_answer)) == _normalize_str(str(ground_truth))


def judge_match(model_answer, ground_truth: str, question: str,
                ask: Callable[[str], str]) -> tuple[bool, str]:
    """Semantic grade via an INDEPENDENT model. Returns (is_match, reason).

    `ask` takes a prompt and returns the model's raw reply, so the transport
    stays injectable and every test here runs offline.
    """
    prompt = JUDGE_PROMPT.format(question=str(question)[:1500],
                                 truth=str(ground_truth)[:500],
                                 model=str(model_answer)[:500])
    try:
        reply = str(ask(prompt) or "")
    except Exception as e:                                    # noqa: BLE001
        # ⚠ FAIL DOWNWARD. A judge that errored graded nothing; counting it as
        # a match would silently inflate the score by the error rate.
        return False, f"judge error: {type(e).__name__}: {e}"[:200]

    # ⚠ The LAST verdict token, not "is there a yes anywhere". A thinking model
    # restates the question first — "The candidate says NO such thing... YES"
    # — so any whole-reply scan reads the reasoning instead of the conclusion,
    # and a fixed tail window still straddles both. Scanning from the end also
    # makes "YES and NO" resolve to NO, which is the conservative direction.
    tail = reply.strip()[-400:]
    last = None
    for m in _VERDICT.finditer(tail):
        last = m.group(0).lower()
    if last == "yes":
        return True, tail
    if last == "no":
        return False, tail
    return False, f"unparseable judge reply: {tail!r}"


def score_one(model_answer, ground_truth: str, question: str,
              ask: Optional[Callable[[str], str]] = None) -> dict:
    """Grade one answer. Strict first; the judge only sees the residual.

    Reported per-answer so a run can be re-analysed without re-running: how
    much of the score came from string equality and how much from a model's
    opinion is exactly the question a sceptical reader should ask.
    """
    # ⚠⚠ AN ABSENT ANSWER IS NEVER A MATCH — and this must SHORT-CIRCUIT the
    # judge rather than be left to it.
    #
    # MEASURED 2026-08-10 on the first real run: asked to compare ground truth
    # "10.81" against an EMPTY candidate, the judge replied **YES**. All three
    # no-answer tasks (2 empty replies + 1 ReadTimeout) scored CORRECT, which
    # inflated the headline from 0.767 to 0.817 — five points of pure credit
    # for answering nothing, in exactly the direction this scorer was built to
    # never err in.
    #
    # It was caught only because dropping the errored rows made accuracy go
    # DOWN, which is arithmetically impossible if those rows were misses. A
    # single consistency check between two views of the same number; without
    # it the inflated figure was completely plausible.
    if model_answer is None or not str(model_answer).strip():
        return {"correct": False, "how": "empty",
                "reason": "no answer produced"}
    if strict_match(model_answer, ground_truth):
        return {"correct": True, "how": "strict", "reason": ""}
    if ask is None:
        return {"correct": False, "how": "strict", "reason": "no judge"}
    ok, why = judge_match(model_answer, ground_truth, question, ask)
    return {"correct": bool(ok), "how": "judge", "reason": why}


def validate_judge(labelled: list[dict],
                   ask: Callable[[str], str]) -> dict:
    """Measure the JUDGE before quoting anything it graded.

    `labelled` items are {question, ground_truth, model_answer, correct} with
    `correct` a HAND label. An unvalidated judge is an unmeasured instrument,
    and this project's whole failure history is instruments nobody checked.

    False-POSITIVE rate is the number to watch: it is the direction that
    inflates the headline.
    """
    tp = fp = tn = fn = 0
    rows = []
    for it in labelled:
        got = score_one(it["model_answer"], it["ground_truth"],
                        it.get("question", ""), ask)["correct"]
        want = bool(it["correct"])
        tp += got and want
        fp += got and not want
        tn += (not got) and (not want)
        fn += (not got) and want
        rows.append({**it, "judged": got, "agrees": got == want})
    n = len(labelled) or 1
    return {
        "n": len(labelled), "agreement": round((tp + tn) / n, 4),
        "true_pos": tp, "false_pos": fp, "true_neg": tn, "false_neg": fn,
        # The inflating direction.
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
        "rows": rows,
    }


def rescore_details(details_path: str | Path,
                    ask: Optional[Callable[[str], str]] = None) -> dict:
    """Re-grade a finished `details.jsonl` without re-running the agent.

    The agent's answers are the expensive artifact; the grading is cheap and
    was wrong once already. Decoupling them means a scorer fix never costs
    another multi-hour run.
    """
    rows = [json.loads(l) for l in Path(details_path).read_text().splitlines()
            if l.strip()]
    out, n_ok, by_how = [], 0, {}
    for r in rows:
        s = score_one(r.get("model_answer"), r.get("ground_truth", ""),
                      r.get("question", ""), ask)
        n_ok += bool(s["correct"])
        by_how[s["how"]] = by_how.get(s["how"], 0) + bool(s["correct"])
        out.append({**r, "correct": s["correct"], "graded_by": s["how"],
                    "grade_reason": s["reason"]})
    return {"n": len(rows), "n_correct": n_ok,
            "accuracy": round(n_ok / len(rows), 4) if rows else None,
            "correct_by_method": by_how, "rows": out}
