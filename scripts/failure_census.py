#!/usr/bin/env python3
"""§4PL: build the silent failure census over the last N days of owner turns.

    GHOST_HOME=~/Data/AI/Data PYTHONPATH=src python scripts/failure_census.py [--days 14] [--json] [--no-write]
    ... --draw-gold                 # §4PM: draw 30 blind packets into system/census/gold/<date>/pool
    ... --ingest-gold grades.jsonl  # join a blind grader's output to the sample (newest, or --gold-date)
    ... --draw-split                # §4QY: bucket the knowledge WRONG/PARTIAL turns (A + blind B/C packets)
    ... --ingest-split r1.jsonl r2.jsonl  # join the bucket readers, apply the pre-registered rule

Read-only over the agent's stores; writes the report to
$GHOST_HOME/system/census/report-<date>.json unless --no-write. The gold
sample is graded by someone who did not see how the replies were made (a
fresh agent with gold/<date>/brief.md, or the owner), never by the agent.
"""
import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", type=float, default=14.0)
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument("--no-write", action="store_true", help="do not save the report file")
    p.add_argument("--draw-gold", action="store_true", help="draw the fortnightly blind sample and stop")
    p.add_argument("--gold-n", type=int, default=30)
    p.add_argument("--ingest-gold", metavar="GRADES_JSONL", help="ingest a blind grader's output and stop")
    p.add_argument("--gold-date", help="the sample folder to ingest into (default: the newest)")
    p.add_argument("--draw-split", action="store_true", help="split a graded sample's knowledge failures and stop")
    p.add_argument("--ingest-split", nargs="+", metavar="BUCKETS_JSONL", help="ingest the split readers and stop")
    ns = p.parse_args(argv)
    from ghost_agent.core import failure_census as FC
    if ns.draw_split or ns.ingest_split:
        res = (FC.draw_split(date=ns.gold_date) if ns.draw_split
               else FC.ingest_split(ns.ingest_split, date=ns.gold_date))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 2 if res.get("error") else 0
    if ns.draw_gold or ns.ingest_gold:
        res = (FC.draw_gold(days=ns.days, n=ns.gold_n) if ns.draw_gold
               else FC.ingest_gold(ns.ingest_gold, date=ns.gold_date))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 2 if res.get("error") else 0
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    report = FC.build(days=ns.days, tool_names=[t["function"]["name"] for t in TOOL_DEFINITIONS])
    if report.get("error"):
        print(FC.render(report), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1, ensure_ascii=False) if ns.json else FC.render(report))
    if not ns.no_write:
        path = FC.write_report(report)
        if path:
            print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
