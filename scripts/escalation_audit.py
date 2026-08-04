#!/usr/bin/env python3
"""Audit verifier escalations: is escalation REPAIRING labels or CORRUPTING them?

The 84% claim-refute overturn rate has been read one way throughout §4F — "the
cheap judge false-alarms and the strong model corrects it" — and that reading is
why refute escalation is on and its churn is treated as latency-only. A live
turn on 2026-08-04 (req 03b96c28) is the counter-example: every tool failed, the
agent answered a fabricated ``0``, the cheap judge REFUTED it correctly, the
main model overturned to CONFIRMED, and the trajectory was backfilled
``failed -> passed`` into the learning corpus.

One case is not a rate. This script makes the rate measurable: it joins the
escalation ledger to the trajectories the verdicts landed on and prints an
adjudication card per overturn, so a human (or a judge model) can score which
side was right. The signal to watch is `tools_failed` — an overturn on a turn
where tools failed is where a fabrication can be laundered into a pass.

Read-only. Writes nothing, changes no verdict.

    GHOST_HOME=... PYTHONPATH=src python scripts/escalation_audit.py
    ... --outcome overturned --limit 20
    ... --json > cards.jsonl        # for a judge pass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _ghost_home() -> Path:
    return Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))


def _load_ledger(path: Path) -> List[Dict[str, Any]]:
    """Ledger rows, newest last. Malformed lines are COUNTED, not dropped
    silently — a shrinking audit population must never look like a clean one."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    if bad:
        print(f"⚠ {bad} malformed ledger line(s) skipped", file=sys.stderr)
    return rows


def _index_trajectories(root: Path, wanted: set) -> Dict[str, Any]:
    """Map trajectory_id -> Trajectory for the ids we actually need.

    Uses `iter_trajectories()` so the corrections sidecar is overlaid — the
    whole question is what the corpus ENDED UP believing, which is the
    overlaid outcome, not the original write.
    """
    out: Dict[str, Any] = {}
    if not wanted:
        return out
    try:
        from ghost_agent.distill.collector import TrajectoryCollector
    except Exception as e:  # pragma: no cover - import shape guard
        print(f"⚠ cannot import TrajectoryCollector ({e}) — cards will carry "
              "ledger fields only", file=sys.stderr)
        return out
    try:
        collector = TrajectoryCollector(root=root)
        for t in collector.iter_trajectories():
            if t.id in wanted:
                out[t.id] = t
                if len(out) == len(wanted):
                    break
    except Exception as e:
        print(f"⚠ trajectory scan failed ({e})", file=sys.stderr)
    return out


def _card(row: Dict[str, Any], traj) -> Dict[str, Any]:
    """One adjudication card. `tools_failed` is the load-bearing field: an
    overturn on a turn where tools failed is where a fabricated answer can be
    laundered into a `passed` label."""
    card: Dict[str, Any] = {
        "ts": row.get("ts", ""),
        "req_id": row.get("req_id", ""),
        "trajectory_id": row.get("trajectory_id", ""),
        "route": row.get("route", ""),
        "kind": row.get("kind", ""),
        "outcome": row.get("outcome", ""),
        "cheap_verdict": row.get("cheap_verdict", ""),
        "cheap_confidence": row.get("cheap_confidence"),
        "strong_verdict": row.get("strong_verdict", ""),
        "final_confidence": row.get("final_confidence"),
        "joined": traj is not None,
    }
    if traj is None:
        return card
    calls = list(getattr(traj, "tool_calls", []) or [])
    failed = [c for c in calls if not _call_ok(c)]
    card.update({
        "task_kind": getattr(traj, "task_kind", ""),
        "user_request": (getattr(traj, "user_request", "") or "")[:600],
        "final_response": (getattr(traj, "final_response", "") or "")[:600],
        "final_response_chars": len(getattr(traj, "final_response", "") or ""),
        "corpus_outcome": getattr(traj, "outcome", ""),
        "failure_reason": getattr(traj, "failure_reason", ""),
        "n_steps": getattr(traj, "n_steps", 0),
        "tools": [getattr(c, "name", "") for c in calls],
        "tools_failed": len(failed),
        "failed_tools": [getattr(c, "name", "") for c in failed][:8],
    })
    return card


def _call_ok(call) -> bool:
    """Best-effort success read across ToolCall shapes."""
    for attr in ("ok", "success"):
        v = getattr(call, attr, None)
        if isinstance(v, bool):
            return v
    err = getattr(call, "error", None)
    if isinstance(err, str) and err.strip():
        return False
    return True


def _render(card: Dict[str, Any]) -> str:
    head = (f"{card['ts']}  {card['route']}/{card['kind']} "
            f"→ {card['outcome'].upper()}  req={card['req_id']}")
    lines = [head, "  " + "-" * (len(head) - 2),
             f"  cheap : {card['cheap_verdict']} "
             f"(conf {card.get('cheap_confidence')})",
             f"  strong: {card['strong_verdict']} "
             f"(final conf {card.get('final_confidence')})"]
    if not card.get("joined"):
        lines.append("  ⚠ trajectory not found — cannot adjudicate this one")
        return "\n".join(lines)
    flag = ""
    if card.get("tools_failed"):
        flag = ("   ⚠ TOOLS FAILED — a fabricated answer here would be "
                "laundered into a pass")
    lines += [
        f"  corpus outcome: {card['corpus_outcome']}"
        f"{('  (' + card['failure_reason'] + ')') if card['failure_reason'] else ''}",
        f"  tools: {', '.join(card['tools']) or '(none)'} · "
        f"{card['tools_failed']} failed{flag}",
        f"  asked : {card['user_request'][:160]!r}",
        f"  answer: {card['final_response'][:160]!r} "
        f"({card['final_response_chars']} chars)",
        "  VERDICT (fill in): was the cheap judge right? [yes / no / unclear]",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=None,
                    help="default $GHOST_HOME/system/verifier/escalations.jsonl")
    ap.add_argument("--trajectories", type=Path, default=None,
                    help="default $GHOST_HOME/system/trajectories")
    ap.add_argument("--outcome", default="overturned",
                    help="filter: overturned|upheld|withheld|unavailable|all "
                         "(default overturned — the population in question)")
    ap.add_argument("--kind", default="all", help="refute|confirm|all")
    ap.add_argument("--route", default="all", help="claim|code|all")
    ap.add_argument("--limit", type=int, default=20,
                    help="newest N after filtering (default 20)")
    ap.add_argument("--json", action="store_true",
                    help="emit JSONL cards for a judge pass instead of text")
    args = ap.parse_args()

    home = _ghost_home()
    ledger = args.ledger or home / "system" / "verifier" / "escalations.jsonl"
    traj_root = args.trajectories or home / "system" / "trajectories"

    rows = _load_ledger(ledger)
    if not rows:
        print(f"No escalation rows at {ledger}.\n"
              "The file is created on the FIRST escalation after a boot, so "
              "absence shortly after a restart is expected — not a broken "
              "instrument.", file=sys.stderr)
        return 1

    def _keep(r: Dict[str, Any]) -> bool:
        return (
            (args.outcome == "all" or r.get("outcome") == args.outcome)
            and (args.kind == "all" or r.get("kind") == args.kind)
            and (args.route == "all" or r.get("route") == args.route)
        )

    sel = [r for r in rows if _keep(r)]
    dropped = len(rows) - len(sel)
    sel = sel[-max(1, args.limit):]

    traj_map = _index_trajectories(
        traj_root, {r.get("trajectory_id") for r in sel if r.get("trajectory_id")})

    cards = [_card(r, traj_map.get(r.get("trajectory_id", ""))) for r in sel]

    if args.json:
        for c in cards:
            print(json.dumps(c, ensure_ascii=False))
        return 0

    print(f"ESCALATION AUDIT — {len(cards)} card(s) from {len(rows)} ledger row(s)"
          f"{f' ({dropped} filtered out)' if dropped else ''}")
    print(f"ledger: {ledger}\n")
    for c in cards:
        print(_render(c))
        print()

    joined = sum(1 for c in cards if c.get("joined"))
    risky = sum(1 for c in cards if c.get("tools_failed"))
    print("=" * 68)
    print(f"joined to a trajectory: {joined}/{len(cards)}")
    print(f"overturns on turns where a TOOL FAILED: {risky}/{len(cards)}"
          "  ← the population where a fabrication becomes a pass")
    print("\nScore the VERDICT lines by hand (or feed --json to a judge). If the "
          "cheap judge was right\nin a material fraction, refute-escalation is "
          "corrupting outcome labels, not repairing them —\nand that outranks "
          "any optimizer work, because the corpus has been absorbing false\n"
          "passes for as long as escalation has been on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
