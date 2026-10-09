#!/usr/bin/env python3
"""§4MN: take (or verify) a store snapshot by hand.

The agent takes one itself, at most daily, in its idle (dream) cycle. Use this
before a repair or to check a snapshot:

  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/snapshot_stores.py [--tag pre-repair] [--dest DIR]
  ... scripts/snapshot_stores.py --verify <snapshot dir>

RESTORE (agent STOPPED — `sudo launchctl bootout system/com.local.ghost-agent`):
for each store folder you restore (the whole `memory/` folder together —
chroma.sqlite3 and its segment folder belong to each other), MOVE the live
folder aside first (`mv memory memory.broken-<date>`), then copy the
snapshot's folder into its place. Never copy INTO a live folder: a crash
leaves `*-wal`/`*-shm` files there, and SQLite replays that WAL onto the
restored database and corrupts it (§4MR, reproduced). Then bootstrap the
agent. Check first with `--verify <snapshot dir>`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="manual")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--verify", default=None, help="verify an existing snapshot dir and exit")
    a = ap.parse_args(argv)
    from ghost_agent.memory.snapshot import take_snapshot, verify_snapshot
    if a.verify:
        problems = verify_snapshot(Path(a.verify))
        print(json.dumps({"snapshot": a.verify, "problems": problems}, indent=1))
        return 1 if problems else 0
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    res = take_snapshot(home, Path(a.dest) if a.dest else None, a.tag)
    print(json.dumps(res, indent=1))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
