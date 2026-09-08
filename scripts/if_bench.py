#!/usr/bin/env python3
"""§4FF — instruction-following bench: SYSTEM_PROMPT vs SYSTEM_PROMPT_COMPILED.

Drives a fixed bank of format-constrained requests through the LIVE agent as
DIAGNOSTIC PROBES (`X-Ghost-Origin: probe` — probes never teach, never enrol
in arms, and are recorded as `task_kind=probe`), once per prompt variant per
repeat, paired per item. The variant is chosen by the probe-only header
`X-Ghost-Prompt-Variant: control|compiled` (ignored on any other origin).

Each item carries a deterministic checker over the final reply text (exact,
regex, word-count, starts-with, JSON, yes/no, language script), so the score
is mechanical. Secondary counts: narration beats ("Let me…"), tool-call XML in
the reply, reply length. Paired outcomes → exact McNemar per pair of arms.

The items are seeded from REAL requests in the trajectory corpus that carried
an explicit constraint (2026-07 → 2026-09; e.g. 5a90ff10 "count the lines …
reply with just the number" failed; 88d1692d "capital of Australia? One
word."), plus synthetic siblings of the same shapes.

usage: if_bench.py [--repeats 2] [--limit N] [--variants control,compiled] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GHOST_HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data"))
AGENT = os.getenv("GHOST_AGENT_URL", "http://127.0.0.1:8000")
KEY = os.getenv("GHOST_API_KEY") or (Path.home() / "Data/AI/.ghost_api_key").read_text().strip()

# ── checkers ─────────────────────────────────────────────────────────────
def _strip(s: str) -> str:
    return re.sub(r"[*_`\"'“”‘’.!]", "", str(s or "")).strip()

def ck_exact(want):
    return lambda r: _strip(r).lower() == _strip(want).lower()

def ck_regex(pat):
    rx = re.compile(pat, re.I | re.S)
    return lambda r: bool(rx.fullmatch(str(r or "").strip()))

def ck_one_word():
    return lambda r: len(_strip(r).split()) == 1

def ck_max_words(n):
    return lambda r: 0 < len(_strip(r).split()) <= n

def ck_one_sentence():
    return lambda r: len([s for s in re.split(r"(?<=[.!?])\s+", str(r or "").strip()) if s.strip()]) == 1

def ck_number_only():
    return lambda r: bool(re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?", _strip(r)))

def ck_yes_no():
    return lambda r: _strip(r).lower() in ("yes", "no")

def ck_json_keys(*keys):
    def f(r):
        txt = str(r or "").strip()
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
        try:
            obj = json.loads(txt)
        except Exception:
            return False
        return isinstance(obj, dict) and all(k in obj for k in keys)
    return f

def ck_starts_with(prefix):
    return lambda r: str(r or "").lstrip().lower().startswith(prefix.lower())

def ck_greek():
    return lambda r: bool(re.search(r"[α-ωΑ-Ω]", str(r or ""))) and not re.search(r"\b(the|and|is|are)\b", str(r or "").lower())

def ck_bullets(n):
    return lambda r: len([l for l in str(r or "").splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", l)]) == n

def ck_no_tool_syntax():
    return lambda r: "<tool_call" not in str(r or "") and "<tool_response" not in str(r or "")

# ── the bank ─────────────────────────────────────────────────────────────
ITEMS = [
    # (id, request, checker, needs_tools)
    ("num-1", "What is 17 times 4? Answer with just the number.", ck_number_only(), False),
    ("num-2", "How many days are in a leap year? Reply with just the number, nothing else.", ck_number_only(), False),
    ("word-1", "What is the capital of Australia? One word.", ck_one_word(), False),
    ("word-2", "Which planet is closest to the Sun? Reply with a single word.", ck_one_word(), False),
    ("exact-1", "Reply with exactly the word READY and nothing else.", ck_exact("READY"), False),
    ("exact-2", "Reply with exactly: PONG2. Nothing else.", ck_exact("PONG2"), False),
    ("exact-3", "Do not investigate or use any tools, just reply with exactly: NOTED", ck_exact("NOTED"), False),
    ("sent-1", "In one sentence: what is a hash table?", ck_one_sentence(), False),
    ("sent-2", "Reply with one short sentence: what is 2+2?", ck_one_sentence(), False),
    ("yn-1", "Is Python dynamically typed? Answer yes or no only.", ck_yes_no(), False),
    ("yn-2", "Is 91 a prime number? Yes or no, nothing else.", ck_yes_no(), False),
    ("json-1", "Give me the boiling point of water at sea level in Celsius and Fahrenheit as JSON with keys celsius and fahrenheit. Output only the JSON.", ck_json_keys("celsius", "fahrenheit"), False),
    ("json-2", "Return a JSON object with keys name and year for the first person on the Moon. JSON only, no prose.", ck_json_keys("name", "year"), False),
    ("start-1", "Answer starting with the words 'Short answer:' — why is the sky blue? Keep it under 40 words.", ck_starts_with("Short answer:"), False),
    ("words-1", "Explain recursion in at most 12 words.", ck_max_words(12), False),
    ("words-2", "Describe the colour of the sea in five words or fewer.", ck_max_words(5), False),
    ("greek-1", "Απάντησε μόνο στα ελληνικά: τι είναι ο ήλιος; Μία πρόταση.", ck_greek(), False),
    ("bul-1", "List exactly three benefits of unit tests as three bullet points, nothing else.", ck_bullets(3), False),
    ("bul-2", "Name four primary colours as a numbered list of four items and nothing more.", ck_bullets(4), False),
    ("code-1", "Reply with only a Python one-liner that prints hello, no explanation, no code fence.", ck_regex(r"print\((['\"])hello\1\)"), False),
    # tool-using items: format constraint after a tool call
    ("tool-num-1", "List the top-level entries of the workspace with file_system, then reply with just the number of entries the listing reported.", ck_number_only(), True),
    ("tool-exact-1", "Run exactly this in the sandbox: echo if-probe-ok — then reply with exactly the word: DONE", ck_exact("DONE"), True),
    ("tool-word-1", "Use system_utility to check the weather for Athens, then reply with one word describing the sky (e.g. clear, cloudy, rain).", ck_one_word(), True),
    ("tool-json-1", "Run `python3 -c \"print(6*7)\"` in the sandbox and reply with JSON only: {\"result\": <the number>}.", ck_json_keys("result"), True),
    ("tool-sent-1", "Check the system health with system_utility and reply in exactly one sentence saying whether it is healthy.", ck_one_sentence(), True),
    ("tool-yn-1", "Look at the workspace listing with file_system and answer yes or no only: is there a file named README.md at the top level?", ck_yes_no(), True),
]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def _chat(text: str, variant: str, rid: str, timeout=400.0) -> dict:
    body = {"messages": [{"role": "user", "content": text}], "stream": False}
    req = urllib.request.Request(
        f"{AGENT}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Ghost-Key": KEY,
                 "X-Ghost-Origin": "probe", "X-Ghost-Prompt-Variant": variant,
                 "X-Request-ID": rid})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    reply = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
    if reply is None:
        reply = d.get("response") or d.get("content") or ""
    return {"reply": str(reply or ""), "seconds": round(time.time() - t0, 1),
            "usage": d.get("usage") or {}}


NARRATION_RE = re.compile(r"^\s*(let me|now (?:let|i'll|i will)|i'll (?:now|start)|first, i)", re.I | re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N items (chunked runs)")
    ap.add_argument("--items", default="", help="comma-separated item ids to run (overrides offset/limit)")
    ap.add_argument("--variants", default="control,compiled")
    ap.add_argument("--no-tools", action="store_true", help="skip tool-using items")
    ap.add_argument("--out", default=str(GHOST_HOME / "system" / "eval" / "if_bench"))
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    items = [it for it in ITEMS if not (args.no_tools and it[3])]
    if args.items:
        want = {x.strip() for x in args.items.split(",") if x.strip()}
        items = [it for it in items if it[0] in want]
    else:
        items = items[args.offset:]
        if args.limit:
            items = items[: args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ledger = out / f"{stamp}.jsonl"
    ok = {v: 0 for v in variants}
    narr = {v: 0 for v in variants}
    leak = {v: 0 for v in variants}
    pair = {"b": 0, "c": 0}   # for the first two variants: b = v0 ok & v1 not; c = v1 ok & v0 not
    total = 0
    t_start = time.time()
    with ledger.open("w") as f:
        for rep in range(args.repeats):
            for iid, text, check, needs_tools in items:
                res = {}
                for v in variants:
                    rid = f"ifb-{stamp}-{iid}-{v}-r{rep}"
                    try:
                        r = _chat(text, v, rid)
                    except Exception as e:  # noqa: BLE001
                        r = {"reply": "", "seconds": None, "usage": {}, "error": str(e)}
                    passed = bool(check(r["reply"]))
                    n_narr = 1 if NARRATION_RE.search(r["reply"]) else 0
                    n_leak = 0 if ck_no_tool_syntax()(r["reply"]) else 1
                    res[v] = passed
                    ok[v] += passed; narr[v] += n_narr; leak[v] += n_leak
                    rec = {"rep": rep, "item": iid, "variant": v, "passed": passed,
                           "narration": n_narr, "tool_syntax_leak": n_leak,
                           "seconds": r.get("seconds"), "reply": r["reply"][:400],
                           "error": r.get("error"), "usage": r.get("usage")}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    print(f"[r{rep} {iid:<12} {v:<8}] {'PASS' if passed else 'FAIL'} "
                          f"{(r.get('seconds') or 0):5.1f}s  {r['reply'][:70]!r}", flush=True)
                total += 1
                if len(variants) >= 2:
                    a, b_ = variants[0], variants[1]
                    if res.get(a) and not res.get(b_): pair["b"] += 1
                    if res.get(b_) and not res.get(a): pair["c"] += 1
    summary = {"items": len(items), "repeats": args.repeats, "pairs": total,
               "pass_rate": {v: ok[v] / total for v in variants} if total else {},
               "narration": narr, "tool_syntax_leak": leak,
               "mcnemar": {"b_first_only": pair["b"], "c_second_only": pair["c"],
                           "p": mcnemar_exact(pair["b"], pair["c"])} if len(variants) >= 2 else None,
               "seconds": round(time.time() - t_start), "ledger": str(ledger)}
    (out / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
