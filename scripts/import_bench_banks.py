#!/usr/bin/env python3
"""Import external graded task sets into bench banks (§4BF Track 1b).

Reads raw dataset files ALREADY ON DISK (the agent's egress is
Tor-only-keyless by doctrine — downloading is an operator/build-time act,
never an agent-runtime one), normalizes them through
``ghost_agent.eval.banks``, and writes ``$GHOST_HOME/system/bench/banks/``.

Usage:
  PYTHONPATH=src python scripts/import_bench_banks.py \
      --mbpp /path/to/mbpp.jsonl --gsm8k /path/to/gsm8k_test.jsonl \
      [--limit N] [--home /path/to/GHOST_HOME]

Sources (keyless, for the operator's curl):
  MBPP:  https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
  GSM8K: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghost_agent.eval import banks  # noqa: E402


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mbpp", help="path to raw mbpp.jsonl")
    ap.add_argument("--gsm8k", help="path to raw GSM8K test.jsonl")
    ap.add_argument("--gsm8k-text",
                    help="path to raw GSM8K test.jsonl, imported as the "
                         "final_response-graded gsm8k_text bank (§4BF "
                         "Track 2 flip ii): the validator reads the "
                         "harness-written answer.txt instead of running "
                         "a solver-written solution.py")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap items per bank (0 = all)")
    ap.add_argument("--home", default=None,
                    help="GHOST_HOME override (default: env)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the oracle self-test against reference "
                         "solutions (MBPP; ~0.1s/item)")
    args = ap.parse_args()

    if not args.mbpp and not args.gsm8k and not args.gsm8k_text:
        ap.error("nothing to import — pass --mbpp, --gsm8k and/or "
                 "--gsm8k-text")

    total = 0
    if args.mbpp:
        items, skipped, broken = [], 0, 0
        for row in _read_jsonl(args.mbpp):
            it = banks.normalize_mbpp(row)
            if it is None:
                skipped += 1
                continue
            # Oracle self-test: the validator must pass the dataset's own
            # reference solution — MBPP has known-flaky rows, and a broken
            # oracle turns every future FAIL into label noise.
            if not args.no_verify:
                ref = str(row.get("code") or "")
                if not ref or not banks.verify_item_against_reference(it, ref):
                    broken += 1
                    continue
            items.append(it)
            if args.limit and len(items) >= args.limit:
                break
        path = banks.write_bank(items, "mbpp", home=args.home)
        print(f"mbpp: wrote {len(items)} items "
              f"({skipped} malformed, {broken} failed oracle self-test) "
              f"→ {path}")
        total += len(items)

    if args.gsm8k:
        items, skipped, broken = [], 0, 0
        for i, row in enumerate(_read_jsonl(args.gsm8k)):
            it = banks.normalize_gsm8k(row, i)
            if it is None:
                skipped += 1
                continue
            # Both-direction oracle self-test (R1 review — this branch
            # previously had NONE, which is how gsm8k-611/796 shipped
            # with oracles that passed gold±1): a solution printing the
            # gold must pass, gold+1 must fail.
            if not args.no_verify:
                gold = banks._gsm8k_gold(str(row.get("answer") or ""))
                ok_ref = f"print({gold!r})"
                bad_ref = f"print({gold + 1!r})"
                if not banks.verify_item_against_reference(it, ok_ref) \
                        or banks.verify_item_against_reference(it, bad_ref):
                    broken += 1
                    continue
            items.append(it)
            if args.limit and len(items) >= args.limit:
                break
        path = banks.write_bank(items, "gsm8k", home=args.home)
        print(f"gsm8k: wrote {len(items)} items "
              f"({skipped} malformed, {broken} failed oracle self-test) "
              f"→ {path}")
        total += len(items)

    if args.gsm8k_text:
        items, skipped, broken = [], 0, 0
        for i, row in enumerate(_read_jsonl(args.gsm8k_text)):
            it = banks.normalize_gsm8k_text(row, i)
            if it is None:
                skipped += 1
                continue
            # Oracle self-test, BOTH directions (a validator that passes
            # everything is worse than none — new filters need a
            # false-positive check): a reply ending in the gold number
            # must pass; the same reply with gold+1 must fail.
            if not args.no_verify:
                gold = banks._gsm8k_gold(str(row.get("answer") or ""))
                ok_reply = f"Working through it step by step.\n{gold}\n"
                bad_reply = f"Working through it step by step.\n{gold + 1}\n"
                if not banks.verify_item_against_reference(it, ok_reply) \
                        or banks.verify_item_against_reference(it, bad_reply):
                    broken += 1
                    continue
            items.append(it)
            if args.limit and len(items) >= args.limit:
                break
        path = banks.write_bank(items, "gsm8k_text", home=args.home)
        print(f"gsm8k_text: wrote {len(items)} items "
              f"({skipped} malformed, {broken} failed oracle self-test) "
              f"→ {path}")
        total += len(items)

    print(f"banks now on disk: {banks.list_banks(args.home)}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
