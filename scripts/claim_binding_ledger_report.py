#!/usr/bin/env python3
"""Daily read of the claim-binding ledger (§4IM/§4IN/§4IP).

    GHOST_HOME=~/Data/AI/Data PYTHONPATH=src python scripts/claim_binding_ledger_report.py [--days 7] [--all]

Answers, for the window: how many verified turns the binder saw, how often
it agreed with the incumbent, how many verdicts it DECIDED (refute-first /
confirm-first overrides) — and prints every override with both quotes, so a
rule defect on live traffic is readable the day it lands. Also lists the
binder failures (rows with an error) and the escalation ledger's outcomes
for the same window, so a silent subsystem is visible.

Rows are data: nothing here changes a verdict.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
from pathlib import Path


def _home() -> Path:
    return Path(os.getenv("GHOST_HOME", str(Path.home() / "Data" / "AI" / "Data")))


def load_rows(path: Path, since: _dt.datetime | None):
    """Rows of the ledger AND its rotated predecessor (`<name>.1`), oldest first."""
    out = []
    parts = [p for p in (Path(str(path) + ".1"), path) if p.exists()]
    if not parts:
        return out
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in parts)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ts = str(row.get("ts") or "")
        if since is not None:
            # a row that cannot be placed in time is not in the window
            # (`--all` still lists it); review §4IP R7 instruments
            try:
                when = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=_dt.timezone.utc)
            if when < since:
                continue
        out.append(row)
    return out


def summarize(rows) -> dict:
    """The counts the daily read needs; pure, for the test."""
    s = collections.Counter()
    overrides, failures, disagreements, capped = [], [], [], []
    for r in rows:
        cb = r.get("claim_binding") or {}
        s["rows"] += 1
        if cb.get("error") or cb.get("verdict") is None:
            s["binder_failed"] += 1
            failures.append(r)
            continue
        s[f"binder_{cb.get('verdict')}"] += 1
        if r.get("agree") is True:
            s["agree"] += 1
        elif r.get("agree") is False:
            s["disagree"] += 1
            disagreements.append(r)
        if r.get("decided") == "claim_binding":
            s["decided_by_binder"] += 1
            overrides.append(r)
        if r.get("capped"):
            s["capped"] += 1                  # §4IR: a cheap CONFIRMED shipped at the withheld confidence
            capped.append(r)
    return {"counts": s, "overrides": overrides, "failures": failures, "disagreements": disagreements, "capped": capped}


def _short(text, n=160):
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def render(summary: dict, escalations) -> str:
    c = summary["counts"]
    lines = [f"claim-binding ledger: {c['rows']} rows | binder failed {c['binder_failed']} | "
             f"REFUTED {c['binder_REFUTED']} CONFIRMED {c['binder_CONFIRMED']} UNCERTAIN {c['binder_UNCERTAIN']} | "
             f"agree {c['agree']} disagree {c['disagree']} | DECIDED BY BINDER {c['decided_by_binder']} | CAPPED {c['capped']}"]
    if summary["overrides"]:
        lines.append("\nOVERRIDES (read each one — a live catch or a rule defect):")
        for r in summary["overrides"]:
            cb = r["claim_binding"]
            appeal = f" [appeal: {r['appeal']}, shipped {r.get('shipped_confidence')}]" if r.get("appeal") else ""
            lines.append(f"- {str(r.get('ts'))[:19]} req={r.get('trace', {}).get('req_id')} incumbent={r['incumbent'].get('verdict')} "
                         f"→ binder={cb.get('verdict')}{appeal} ({_short(cb.get('reasoning'), 100)})")
            for i in (cb.get("issues") or [])[:3]:
                lines.append(f"    · {_short(i, 220)}")
    if summary.get("capped"):
        lines.append("\nCAPPED (a cheap CONFIRMED shipped at 0.6 on a name the session never carried — read each: a fabrication or a name the reply knew):")
        for r in summary["capped"]:
            lines.append(f"- {str(r.get('ts'))[:19]} req={r.get('trace', {}).get('req_id')} names={r.get('capped')}")
    if summary["failures"]:
        lines.append(f"\nBINDER FAILURES ({len(summary['failures'])}):")
        for r in summary["failures"][:10]:
            lines.append(f"- {str(r.get('ts'))[:19]} req={r.get('trace', {}).get('req_id')} error={_short((r.get('claim_binding') or {}).get('error'), 120)}")
    dis = [r for r in summary["disagreements"] if r.get("decided") != "claim_binding"]
    if dis:
        kinds = collections.Counter(f"incumbent {r['incumbent'].get('verdict')} / binder {r['claim_binding'].get('verdict')}" for r in dis)
        lines.append("\nDISAGREEMENTS the incumbent decided: " + ", ".join(f"{k} ×{v}" for k, v in kinds.most_common()))
    if escalations:
        kinds = collections.Counter(str(r.get("outcome")) for r in escalations)
        lines.append("\nescalation ledger, same window: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7, help="window in days (default 7)")
    ap.add_argument("--all", action="store_true", help="the whole ledger")
    args = ap.parse_args(argv)
    since = None if args.all else _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=args.days)
    home = _home()
    rows = load_rows(home / "system" / "verifier" / "claim_binding_shadow.jsonl", since)
    esc = load_rows(home / "system" / "verifier" / "escalations.jsonl", since)
    print(render(summarize(rows), esc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
