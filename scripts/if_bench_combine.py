#!/usr/bin/env python3
"""Combine one or more if_bench ledgers (chunked runs) into one paired summary.

usage: if_bench_combine.py LEDGER.jsonl [LEDGER2.jsonl ...]
Pairs rows by (rep, item); a pair needs BOTH variants. Exact McNemar on the
paired outcomes; per-variant pass rate, narration and tool-syntax counts.
"""
import json
import sys
from collections import defaultdict
from math import comb


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def main(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by = defaultdict(dict)
    for r in rows:
        by[(r["rep"], r["item"])][r["variant"]] = r
    variants = sorted({r["variant"] for r in rows})
    a, b_ = (variants + [None, None])[:2]
    pairs = [v for v in by.values() if a in v and b_ in v]
    ok = {v: sum(1 for p in pairs if p[v]["passed"]) for v in variants}
    narr = {v: sum(p[v]["narration"] for p in pairs) for v in variants}
    leak = {v: sum(p[v]["tool_syntax_leak"] for p in pairs) for v in variants}
    secs = {v: round(sum((p[v]["seconds"] or 0) for p in pairs) / max(1, len(pairs)), 1) for v in variants}
    bb = sum(1 for p in pairs if p[a]["passed"] and not p[b_]["passed"])
    cc = sum(1 for p in pairs if p[b_]["passed"] and not p[a]["passed"])
    disagreements = [(k, {v: p[v]["passed"] for v in variants}, {v: p[v]["reply"][:60] for v in variants})
                     for k, p in by.items() if a in p and b_ in p and p[a]["passed"] != p[b_]["passed"]]
    out = {"pairs": len(pairs), "pass_rate": {v: ok[v] / len(pairs) for v in variants} if pairs else {},
           "passed": ok, "narration": narr, "tool_syntax_leak": leak, "mean_seconds": secs,
           "mcnemar": {f"{a}_only": bb, f"{b_}_only": cc, "p": mcnemar_exact(bb, cc)},
           "disagreements": disagreements}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
