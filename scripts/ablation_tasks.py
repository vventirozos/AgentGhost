"""Discriminating single-session task suite for the ablation harness.

The default eval suite (`ghost_agent.eval.tasks._load_curated_tasks`) is three
trivial prompts — fine as a CI smoke test, useless for telling whether the
cognitive layers earn their keep. This suite is built to *discriminate*: tasks
spread across capability clusters, each with a robust validator that tolerates
a chatty model but still requires the RIGHT answer.

IMPORTANT — read before trusting the numbers:

  * These tasks exercise the IN-SESSION layers (raw reasoning, deep-reason/MCTS,
    metacog/arbiter, swarm, tool loop, pre-flight guard). A clean ablation runs
    each config with a FRESH, EMPTY GHOST_HOME, so the CROSS-SESSION layers
    (memory/RRF, selfhood, reflection, dream, skills_auto, the cross-project
    map) start blank and CANNOT help here. If those show "no effect" on this
    suite, that is NOT evidence they are useless — it means single-shot is the
    wrong probe. Use the retention protocol (see scripts/ABLATION.md, Track B)
    for the cross-session layers.

  * REPLACE these with YOUR real workload as soon as you can. A suite that
    doesn't look like what you actually use the agent for measures the wrong
    thing. Keep the validator discipline: objective, automatic, stable.

Each task is a CuratedRequestTask(prompt, validator) where validator is a
callable (output, ctx) -> (bool, reason).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, List, Tuple

from ghost_agent.eval.tasks import CuratedRequestTask


# --------------------------------------------------------------------------
# Validator helpers — tolerant of prose, strict on the answer.
# --------------------------------------------------------------------------

def contains_number(n: int) -> Callable[[str, Any], Tuple[bool, str]]:
    """True if the integer n appears as a standalone token (commas/spaces ok)."""
    plain = str(n)
    grouped = f"{n:,}"
    pat = re.compile(rf"(?<![\d.]){re.escape(plain)}(?![\d.])")
    patg = re.compile(rf"(?<![\d.]){re.escape(grouped)}(?![\d])")

    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        ok = bool(pat.search(out or "")) or bool(patg.search(out or ""))
        return ok, "" if ok else f"expected the number {n}"
    return _v


def contains_any(words: List[str]) -> Callable[[str, Any], Tuple[bool, str]]:
    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        low = (out or "").lower()
        hit = [w for w in words if w.lower() in low]
        return bool(hit), "" if hit else f"none of {words} present"
    return _v


def contains_all(words: List[str]) -> Callable[[str, Any], Tuple[bool, str]]:
    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        low = (out or "").lower()
        missing = [w for w in words if w.lower() not in low]
        return (not missing), "" if not missing else f"missing {missing}"
    return _v


def equals_token(token: str) -> Callable[[str, Any], Tuple[bool, str]]:
    """The output, stripped of surrounding whitespace/punctuation/quotes/
    markdown, equals `token` (case-sensitive). For strict format tasks."""
    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        cleaned = (out or "").strip().strip("`*_\"'.").strip()
        ok = cleaned == token
        return ok, "" if ok else f"expected exactly {token!r}, got {cleaned[:40]!r}"
    return _v


def json_object_has(expected: dict) -> Callable[[str, Any], Tuple[bool, str]]:
    """Parse the first {...} block in the output and check it contains every
    key/value in `expected`."""
    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        m = re.search(r"\{.*\}", out or "", re.DOTALL)
        if not m:
            return False, "no JSON object found"
        try:
            obj = json.loads(m.group(0))
        except Exception as e:
            return False, f"invalid JSON: {e}"
        for k, v in expected.items():
            if obj.get(k) != v:
                return False, f"expected {k}={v!r}, got {obj.get(k)!r}"
        return True, ""
    return _v


def _T(task_id: str, cluster: str, prompt: str, validator) -> CuratedRequestTask:
    t = CuratedRequestTask(task_id=task_id, category="curated", prompt=prompt,
                           validator=validator, cluster=cluster)
    return t


# --------------------------------------------------------------------------
# The suite. Clusters: reasoning, tooluse, format, instruction, robustness,
# anchor (easy floor so a thin baseline still scores > 0).
# --------------------------------------------------------------------------

def load_ablation_suite() -> List[CuratedRequestTask]:
    return [
        # --- anchors: trivial; every config should pass → sanity floor ---
        _T("anchor:capital", "anchor",
           "What is the capital of France? Answer in one word.",
           contains_any(["paris"])),
        _T("anchor:arith", "anchor",
           "What is 7 times 6? Reply with just the number.",
           contains_number(42)),

        # --- reasoning: careful multi-step; stresses deep-reason / metacog ---
        _T("reason:bat_ball", "reasoning",
           "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
           "than the ball. How much does the ball cost? Give the amount in "
           "dollars.",
           contains_any(["0.05", "$.05", "5 cent", "five cent", "0,05"])),
        _T("reason:apples", "reasoning",
           "I start with 3 apples, then buy 2 bags that each contain 4 apples. "
           "How many apples do I have in total? Reply with just the number.",
           contains_number(11)),
        _T("reason:ages", "reasoning",
           "Anna is twice as old as Ben. In 5 years Anna will be 25. How old "
           "is Ben now? Reply with just the number.",
           contains_number(10)),

        # --- tool use: forces sandbox execution; wrong arithmetic = fail ---
        _T("tool:pow", "tooluse",
           "Use your code execution tool to compute 2 to the power of 20, and "
           "report the resulting number.",
           contains_number(1048576)),
        _T("tool:count_a", "tooluse",
           "Use your code execution tool to count how many times the letter "
           "'a' appears in the word 'abracadabra'. Report the count.",
           contains_number(5)),
        _T("tool:sum100", "tooluse",
           "Use your code execution tool to compute the sum of all integers "
           "from 1 to 100 inclusive. Report the number.",
           contains_number(5050)),

        # --- format / instruction-following: strict output shape ---
        _T("format:exact_word", "format",
           "Reply with exactly this single word in uppercase and nothing else: "
           "BANANA",
           equals_token("BANANA")),
        _T("format:json", "format",
           'Output ONLY a JSON object with keys "status" set to "ok" and '
           '"count" set to 7. No prose, no code fence.',
           json_object_has({"status": "ok", "count": 7})),
        _T("format:primary_colors", "format",
           "Name the three additive primary colors of light, as a single "
           "comma-separated line.",
           contains_all(["red", "green", "blue"])),

        # --- robustness: false premises & non-looping failure handling ---
        _T("robust:false_premise", "robustness",
           "In what year was the number seven elected president of Brazil? If "
           "the question doesn't make sense, say so plainly.",
           contains_any(["no sense", "doesn't make", "does not make", "not a",
                         "isn't", "cannot", "can't", "nonsensical", "invalid"])),
        _T("robust:missing_file", "robustness",
           "Try to read the file 'definitely_missing_probe_9z.txt' once. Then "
           "tell me plainly whether it exists. Do not keep retrying.",
           contains_any(["not exist", "does not", "doesn't exist", "no such",
                         "not found", "could not", "couldn't", "missing"])),
    ]
