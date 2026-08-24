#!/usr/bin/env python3
"""§4CS item D — WHY is each undecidable turn undecidable?

MEASURE-ONLY. This builds nothing and changes nothing; it exists so the
breakdown is re-derivable instead of being a number someone once quoted.

§4CM measured what the 877 undecided turns cost DOWNSTREAM (of them, 7
would ever become replayable). This asks the upstream question §4CN named
as the only lever still open: what stage failed, per turn?

  A  a verdict was PRODUCED but never attached to the corpus
  A2 a correction row exists yet the outcome is still undecided
  B  the turn made NO tool call — nothing was executed to check
  C  the turn used tools and no verdict was ever recorded

Run:
    GHOST_HOME=/path/to/Data PYTHONPATH=src python3 scripts/measure_undecidable_turns.py
"""
from __future__ import annotations

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


def _rows(path: Path):
    if not path.is_file():
        return
    for ln in path.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def main() -> int:
    home = _home()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.core.replay_engine import triage

    trajs = list(TrajectoryCollector(
        root=str(home / "system" / "trajectories")).iter_trajectories())
    user = [t for t in trajs
            if str(getattr(t, "task_kind", "") or "") == "user_request"]

    def decided(t):
        return str(getattr(t, "outcome", "") or "").lower() in ("passed", "failed")

    def ntc(t):
        return len(getattr(t, "tool_calls", None) or [])

    und = [t for t in user if not decided(t)]

    print(f"corpus: {len(trajs)} trajectories, {len(user)} user_request, "
          f"{len(user) - len(und)} decided ({100*(len(user)-len(und))/max(len(user),1):.1f}%), "
          f"{len(und)} undecidable")

    print("\nTRIAGE over user_request turns (replay_engine.triage):")
    tri = collections.Counter()
    for t in user:
        r = triage(t)
        tri[("REPLAYABLE" if r.replayable else r.reason)] += 1
    for k, v in tri.most_common():
        print(f"  {k:38} {v:5}")

    # Evidence stores.
    verdicts: dict = {}
    for p in sorted((home / "system" / "verdicts").glob("*.jsonl")):
        for r in _rows(p):
            verdicts.setdefault(r.get("trajectory_id"), []).append(r.get("verdict"))
    corrections: dict = {}
    for r in _rows(home / "system" / "trajectories" / "corrections.jsonl"):
        corrections.setdefault(r.get("trajectory_id"), []).append(r)

    cat = collections.Counter()
    verdict_kinds = collections.Counter()
    era = collections.Counter()
    for t in und:
        tid = getattr(t, "id", None)
        if tid in verdicts:
            cat["A  verdict PRODUCED but never attached"] += 1
            for v in verdicts[tid]:
                verdict_kinds[v] += 1
        elif tid in corrections:
            cat["A2 correction row exists, outcome still undecided"] += 1
        elif ntc(t) == 0:
            cat["B  no tool call — nothing was executed to check"] += 1
        else:
            cat["C  tool-using, no verdict ever recorded"] += 1
            ts = (getattr(t, "timestamp", "") or "")[:10]
            era["   C before 2026-08-13 (pre human-feedback channel)"
                if ts < "2026-08-13" else
                "   C 2026-08-13..08-20"
                if ts < "2026-08-21" else
                "   C on/after 2026-08-21 (post late-verdict backfill)"] += 1

    print(f"\nWHY UNDECIDABLE  (n={len(und)}):")
    for k, v in cat.most_common():
        print(f"  {k:52} {v:5}  {100*v/max(len(und),1):5.1f}%")
    if verdict_kinds:
        print(f"     └ bucket A verdict kinds: {dict(verdict_kinds)} "
              f"(UNCERTAIN is correctly NOT decisive — not a lost verdict)")
    for k, v in sorted(era.items()):
        print(f"  {k:52} {v:5}")

    print("\nDECIDED-RATE BY TURN SHAPE — the ceiling, and where it comes from:")
    for label, sel in (("chat  (0 tool calls)", lambda t: ntc(t) == 0),
                       ("tools (>=1 call)   ", lambda t: ntc(t) > 0)):
        g = [t for t in user if sel(t)]
        d = sum(1 for t in g if decided(t))
        print(f"  {label}: {d}/{len(g)} = {100*d/max(len(g),1):.1f}% decided")

    print("\nWHO DECIDES, by turn shape (source of the winning sidecar row):")
    for label, sel in (("chat ", lambda t: ntc(t) == 0),
                       ("tools", lambda t: ntc(t) > 0)):
        src = collections.Counter()
        for t in user:
            if not sel(t) or not decided(t):
                continue
            rows = corrections.get(getattr(t, "id", None))
            src[rows[-1].get("source") if rows else "inline at write time"] += 1
        print(f"  {label}: " + ", ".join(f"{k}={v}" for k, v in src.most_common()))

    print("\nBY MONTH (decided / total):")
    for label, sel in (("chat ", lambda t: ntc(t) == 0),
                       ("tools", lambda t: ntc(t) > 0)):
        by = collections.defaultdict(lambda: [0, 0])
        for t in user:
            if not sel(t):
                continue
            m = (getattr(t, "timestamp", "") or "")[:7]
            by[m][0] += 1 if decided(t) else 0
            by[m][1] += 1
        print(f"  {label}: " + "  ".join(
            f"{m} {by[m][0]}/{by[m][1]}" for m in sorted(by)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
