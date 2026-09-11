#!/usr/bin/env python3
"""§4FY — replay the state-aware judge tier over the trajectory corpus.

MEASURE-ONLY. Runs `core.turn_state_check.refute_turn_state` on every real
user turn (probes excluded, as every learner/report excludes them) and
cross-tabulates the fires against the overlaid outcome and the HUMAN label
where one exists. Prints every fire on a passed / human-approved turn in
full, because those are the false refutes this tier must not write.

    GHOST_HOME=/path/to/Data PYTHONPATH=src python3 scripts/turn_state_replay.py
    [--rule strict_json] [--show-all] [--days N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path


def _home() -> Path:
    h = os.getenv("GHOST_HOME", "").strip()
    if not h:
        sys.exit("GHOST_HOME is not set — refusing to guess the data root.")
    p = Path(h)
    if not (p / "system" / "trajectories").is_dir():
        sys.exit(f"{p}/system/trajectories does not exist.")
    return p


def _human_labels(home: Path) -> dict:
    """trajectory_id → (outcome, source) for HUMAN rows only."""
    out = {}
    corr = home / "system" / "trajectories" / "corrections.jsonl"
    if not corr.is_file():
        return out
    for line in corr.open(encoding="utf-8", errors="replace"):
        try:
            c = json.loads(line)
        except Exception:
            continue
        src = str(c.get("source") or "")
        if src.startswith("human") and c.get("outcome") in ("passed", "failed"):
            tid = c.get("trajectory_id") or c.get("id")
            if tid:
                out[tid] = (c["outcome"], src)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="", help="only this rule tag")
    ap.add_argument("--show-all", action="store_true", help="print every fire, not only the suspicious ones")
    ap.add_argument("--days", type=int, default=0, help="only the last N day partitions")
    ap.add_argument("--brief", action="store_true", help="short request/reply excerpts")
    args = ap.parse_args()
    rq, rp = (140, 260) if args.brief else (300, 500)
    home = _home()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.core.turn_state_check import refute_turn_state, mechanical_constraints

    human = _human_labels(home)
    coll = TrajectoryCollector(root=str(home / "system" / "trajectories"))
    days = None
    if args.days:
        days = sorted(p.name for p in (home / "system" / "trajectories").iterdir() if p.is_dir())[-args.days:]

    n = 0
    fires = collections.Counter()               # rule → count
    by_outcome = collections.defaultdict(collections.Counter)   # rule → outcome → count
    by_human = collections.defaultdict(collections.Counter)     # rule → human → count
    constrained = collections.Counter()         # kind → turns carrying the constraint
    outcomes = collections.Counter()
    suspicious = []
    # ⚠ Named so the headline cannot be read as precision when it is coverage:
    # "0 fires on a human-approved turn" with 0 human-labelled turns carrying
    # a constraint is 0/0 (§4FY review). Both counts are printed.
    human_joined = 0
    human_constrained = 0

    def walk():
        if days:
            for d in days:
                yield from coll.iter_trajectories(day=d)
        else:
            yield from coll.iter_trajectories()

    for t in walk():
        if str(getattr(t, "task_kind", "") or "") != "user_request":
            continue
        n += 1
        req = str(getattr(t, "user_request", "") or "")
        reply = str(getattr(t, "final_response", "") or "")
        tools = []
        for tc in (getattr(t, "tool_calls", None) or []):
            if isinstance(tc, dict):
                tools.append(tc)
            else:
                tools.append({"name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", {}),
                              "result": getattr(tc, "result", ""), "error": getattr(tc, "error", None)})
        outcome = str(getattr(t, "outcome", "") or "")
        outcomes[outcome] += 1
        if t.id in human:
            human_joined += 1
        cons = mechanical_constraints(req)
        for c in cons:
            constrained[c.kind] += 1
        if cons and t.id in human:
            human_constrained += 1
        issues = refute_turn_state(request=req, reply=reply, tools_run=tools)
        if args.rule:
            issues = [i for i in issues if i[0] == args.rule]
        if not issues:
            continue
        h = human.get(t.id)
        for rule, msg in issues:
            fires[rule] += 1
            by_outcome[rule][outcome] += 1
            by_human[rule][h[0] if h else "-"] += 1
        flag = (outcome == "passed") or (h and h[0] == "passed")
        if flag or args.show_all:
            suspicious.append((t.id, outcome, h, req, reply, issues, [x.get("name") for x in tools]))

    print(f"{n} real user turns; outcomes {dict(outcomes)}; human-labelled turns walked "
          f"{human_joined} (of {len(human)} human label rows)")
    print(f"turns carrying a mechanical constraint: {dict(constrained)}; "
          f"of the human-labelled turns: {human_constrained}\n")
    print(f"{'rule':16} {'fires':>5}  by overlaid outcome                 by human label")
    for rule, k in fires.most_common():
        print(f"{rule:16} {k:>5}  {dict(by_outcome[rule])!s:36} {dict(by_human[rule])}")
    print(f"\n{len(suspicious)} fire(s) on a passed / human-approved turn"
          + (" (or all fires, --show-all)" if args.show_all else "") + ":\n")
    for tid, outcome, h, req, reply, issues, names in suspicious:
        print("=" * 100)
        print(f"{tid}  outcome={outcome}  human={h}")
        print(f"  REQ:   {req[:rq]!r}")
        if not args.brief:
            print(f"  TOOLS: {names[:12]}")
        print(f"  REPLY: {reply[:rp]!r}")
        for rule, msg in issues:
            print(f"  >> {rule}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
