"""§4LL follow-up 2 (2026-10-04, operator: "remove them"): the three diary rows
`memory_repair_4ll_b.py` left — the two non-git §4KB test turns of 09-23
(dup_probe, r5) and the 09-22 row of the image request whose lesson was
retired. By id; each row must still carry its expected text, or nothing is
applied. Agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4ll_c.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

APPLY = "--apply" in sys.argv
SYSTEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system"
DIARY = SYSTEM / "selfhood" / "autobiographical.jsonl"
DIARY_IDS = {  # id prefix -> text the row's summary must contain
    "1d578f23": 'path="dup_probe.py"',
    "f72fe9f5": 'path="r5/n.py"',
    "04e1deb4": "Create a photorealistic image that looks like a real photograph",
}


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def diary_split(lines: list) -> tuple:
    keep, hits = [], {p: 0 for p in DIARY_IDS}
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            keep.append(ln)
            continue
        p = next((p for p in DIARY_IDS if str(r.get("id", "")).startswith(p)), None)
        if p is None:
            keep.append(ln)
            continue
        if DIARY_IDS[p] not in (r.get("summary") or ""):
            raise SystemExit(f"diary row {p} is not the expected row — nothing applied")
        hits[p] += 1
    bad = {p: n for p, n in hits.items() if n != 1}
    if bad:
        raise SystemExit(f"diary ids not matching exactly one row: {bad} — nothing applied")
    return keep, len(DIARY_IDS)


def main():
    lines = DIARY.read_text(encoding="utf-8").splitlines()
    keep, n = diary_split(lines)
    print(f"drop {n} diary rows ({len(lines)} -> {len(keep)}){'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    shutil.copy2(DIARY, DIARY.with_name(f"autobiographical.jsonl.pre-4ll-c-{time.strftime('%Y%m%dT%H%M%S')}.bak"))
    tmp = DIARY.with_suffix(".jsonl.tmp-4ll")
    tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
    tmp.replace(DIARY)
    print(f"diary rows dropped {n}")


# §4MR: a store writer must be the ONLY writer — refuse beside the running
# agent (two chroma writers left the store segfaulting on open, §4MN)
if __name__ == "__main__" and "--apply" in __import__("sys").argv:
    import os as _os4mr, sys as _sys4mr
    from pathlib import Path as _P4mr
    _sys4mr.path.insert(0, str(_P4mr(__file__).resolve().parents[1] / "src"))
    from ghost_agent.memory.store_lock import assert_no_other_writer as _no_other_writer
    _no_other_writer(_P4mr(_os4mr.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory",
                     _P4mr(__file__).name)


if __name__ == "__main__":
    main()
