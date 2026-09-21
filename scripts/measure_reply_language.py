#!/usr/bin/env python3
"""Reply-language and query-language measurement over the recorded turns.

Two rates, both on USER requests written in Latin script (English):
  * query switch — requests that ran web searches and issued at least one
    Greek-script query;
  * reply mismatch — requests whose reply is mostly Greek script, measured
    on the reply's PROSE lines (list items, table rows and «quoted»
    headlines are excluded: a news digest quoting Naftemporiki is not a
    language drift).
And the reverse for Greek requests (reply mostly Latin). Internal kinds
(reflection, bench, self-play) are excluded. `--since` splits the corpus
so a prompt change can be re-measured against its own baseline:

    PYTHONPATH=src python scripts/measure_reply_language.py --since 2026-09-21
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from pathlib import Path

GREEK = re.compile(r"[Ͱ-Ͽ]")
LATIN = re.compile(r"[A-Za-z]")
_SKIP_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\||#{1,6}\s|\*\*[^*]*«|«)")
INTERNAL_KINDS = ("reflection", "bench", "self", "dream", "probe_internal")


def script_share(text: str) -> float:
    """Greek letters over Greek+Latin letters; 0.0 when there are none."""
    g = len(GREEK.findall(text or "")); l = len(LATIN.findall(text or ""))
    return g / (g + l) if g + l else 0.0


def prose_lines(reply: str) -> str:
    """The reply minus list items, table rows, headings and «quoted» lines —
    the parts whose language is the model's own choice."""
    keep = [ln for ln in (reply or "").splitlines() if ln.strip() and not _SKIP_LINE.match(ln)]
    return "\n".join(keep)


def query_of(call) -> str:
    a = call.get("arguments")
    if isinstance(a, dict):
        return str(a.get("query") or "")
    try:
        return str(ast.literal_eval(str(a)).get("query") or "")
    except Exception:
        m = re.search(r"'query':\s*'([^']+)'", str(a))
        return m.group(1) if m else ""


def classify(row: dict) -> dict | None:
    """One recorded turn → its language facts, or None when not a user turn."""
    kind = str(row.get("task_kind") or "")
    if any(kind.startswith(k) for k in INTERNAL_KINDS):
        return None
    req = row.get("user_request") or ""; rep = row.get("final_response") or ""
    if len(req) < 8 or len(rep) < 40:
        return None
    queries = [query_of(t) for t in (row.get("tool_calls") or []) if t.get("name") == "web_search"]
    queries = [q for q in queries if q]
    return {
        "id": str(row.get("id", ""))[:8],
        "req_greek": script_share(req),
        "rep_greek_prose": script_share(prose_lines(rep)),
        "n_queries": len(queries),
        "greek_queries": sum(1 for q in queries if GREEK.search(q)),
        "mixed_queries": sum(1 for q in queries if GREEK.search(q) and LATIN.search(q)),
    }


def summarize(facts: list[dict]) -> dict:
    eng = [f for f in facts if f["req_greek"] < 0.05]
    eng_s = [f for f in eng if f["n_queries"]]
    gr = [f for f in facts if f["req_greek"] > 0.5]
    def rate(n, d):
        return round(n / d, 4) if d else None
    return {
        "english_requests": len(eng),
        "english_with_searches": len(eng_s),
        "query_switch": {"n": sum(1 for f in eng_s if f["greek_queries"]), "rate": rate(sum(1 for f in eng_s if f["greek_queries"]), len(eng_s))},
        "mixed_script_queries": {"n": sum(f["mixed_queries"] for f in eng_s), "of_queries": sum(f["n_queries"] for f in eng_s)},
        "reply_mismatch_en": {"n": sum(1 for f in eng if f["rep_greek_prose"] > 0.5), "rate": rate(sum(1 for f in eng if f["rep_greek_prose"] > 0.5), len(eng)),
                              "ids": [f["id"] for f in eng if f["rep_greek_prose"] > 0.5]},
        "greek_requests": len(gr),
        "reply_mismatch_el": {"n": sum(1 for f in gr if f["rep_greek_prose"] < 0.2), "rate": rate(sum(1 for f in gr if f["rep_greek_prose"] < 0.2), len(gr))},
    }


def load(root: str, since: str = "", until: str = "") -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(root, "20*", "*.jsonl"))):
        day = Path(f).parent.name
        if since and day < since or until and day >= until:
            continue
        for line in open(f, errors="replace"):
            try:
                fact = classify(json.loads(line))
            except Exception:
                continue
            if fact:
                out.append(fact)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/Data/AI/Data/system/trajectories"))
    ap.add_argument("--since", default="", help="YYYY-MM-DD: measure this day onward; the days before are the baseline")
    args = ap.parse_args()
    if args.since:
        print("BASELINE (before %s):" % args.since, json.dumps(summarize(load(args.root, until=args.since)), indent=1))
        print("SINCE %s:" % args.since, json.dumps(summarize(load(args.root, since=args.since)), indent=1))
    else:
        print(json.dumps(summarize(load(args.root)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
