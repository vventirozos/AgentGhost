#!/usr/bin/env python3
"""§4ML one-off (2026-10-08): channel-member turns from before the member wall
that the stores still count as the owner's.

`member_data_cleanup_4kj.py` stamped only trajectories whose id starts with
`slack-`. Before 2026-09-24 Slack turns had plain 8-hex ids, so the members'
turns from 08-16 → 09-23 carried no role and `trajectory_may_teach` read them
as the owner's. Five of the owner's own 09-24 turns went the other way (stamped
member by the `slack-` rule).

The truth is the bot's reply index (`ghost-slack-reply-index.json`): Slack
itself says who asked each request. Per request id it names the requester:
  1. trajectories: `extra.requester_role` set to what the index says
     (member → member; owner → owner), only where it differs;
  2. lessons sourced from member trajectories: retracted from the JSON
     playbook AND the vector store (correctives too — the turn was not ours);
  3. foresight ledger rows (`req_id`), calibration rows (`req_id`) and memory
     ranking observations (`turn`) of member requests: removed;
  4. episodes whose trigger IS a member request's text (normalised, first 200
     chars) and which were recorded within a day of it: forgotten (archived,
     `delete_episodes`). Episodes carry no request id before §4MJ.
NOT touched: knowledge-graph triplets (no provenance at all — a term-based
guess could delete the owner's own facts); verifier logs (diagnostic, probes
write them too).

Run ONLY with the agent stopped (the vector store has one writer). Backups next
to every file touched. Dry run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/member_data_repair_4ml.py [--apply]
"""
import datetime as _dt
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
SYS = HOME / "system"
MEM = SYS / "memory"
INDEX = Path(os.environ.get("GHOST_SLACK_REPLY_INDEX_PATH",
                            "/Users/vasilis/Data/AI/Logs/ghost-slack-reply-index.json"))
OWNER = os.environ.get("GHOST_SLACK_OWNER", "U56CVBHHQ")
STAMP = time.strftime("%Y%m%dT%H%M%S")


def backup(p: Path) -> None:
    shutil.copy2(p, p.with_name(p.name + f".pre-4ml-{STAMP}.bak"))


def _rewrite_jsonl(path: Path, keep_fn) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    keep, drop = [], 0
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001 — never drop what we cannot read
            keep.append(line)
            continue
        if keep_fn(row):
            keep.append(line)
        else:
            drop += 1
    if drop and APPLY:
        backup(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        os.replace(tmp, path)
    return drop


def requesters() -> dict:
    """request id → 'owner' | 'member' from the reply index (skips the test
    fixture rows a pre-guard test wrote: channel ids that are not Slack's)."""
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    out = {}
    for key, v in idx.items():
        rid, who = v.get("req_id"), v.get("requester")
        if not rid or not who or key.startswith("C1234567"):
            continue
        out[rid] = "owner" if who == OWNER else "member"
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()[:200]


def stamp_trajectories(who: dict):
    changed, member_tids, member_reqs, owner_texts = 0, set(), {}, set()
    for f in sorted(glob.glob(str(SYS / "trajectories" / "*" / "*.jsonl"))):
        lines = open(f, encoding="utf-8").read().splitlines()
        out, n = [], 0
        for line in lines:
            try:
                t = json.loads(line)
            except Exception:  # noqa: BLE001
                out.append(line)
                continue
            role = who.get(str(t.get("session_id") or ""))
            # the owner's texts, by the INDEX first (5 owner turns carry a
            # wrong `member` stamp from the §4KJ rule), then by the stamp
            if role == "owner" or (role is None and (t.get("extra") or {}).get("requester_role") != "member"):
                owner_texts.add(_norm(t.get("user_request")))
            if role is None:
                out.append(line)
                continue
            if role == "member":
                tid = t.get("id") or t.get("trajectory_id")
                if tid:
                    member_tids.add(tid)
                member_reqs[str(t.get("session_id"))] = (t.get("user_request") or "", t.get("timestamp") or t.get("ts"))
            ex = t.get("extra") or {}
            if ex.get("requester_role") != role:
                ex["requester_role"] = role
                ex["requester_role_repaired_4ml"] = True
                t["extra"] = ex
                n += 1
                out.append(json.dumps(t, ensure_ascii=False))
            else:
                out.append(line)
        if n:
            changed += n
            if APPLY:
                backup(Path(f))
                tmp = f + ".tmp"
                open(tmp, "w", encoding="utf-8").write("\n".join(out) + "\n")
                os.replace(tmp, f)
    return changed, member_tids, member_reqs, owner_texts


def _ts(v):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


def member_episode_ids(member_reqs: dict, owner_texts: set = frozenset()) -> list:
    import sqlite3
    by_text = {}
    for text, ts in member_reqs.values():
        k = _norm(text)
        # a text the OWNER also sent ("hi", "status?") is never matched — the
        # episode could be the owner's
        if len(k) >= 8 and k not in owner_texts:
            by_text.setdefault(k, []).append(_ts(ts))
    con = sqlite3.connect(f"file:{MEM / 'episodic_memory.db'}?mode=ro", uri=True)
    ids = []
    for eid, trig, ts in con.execute("SELECT id, trigger, timestamp FROM episodes"):
        times = by_text.get(_norm(trig))
        # an episode is written at the end of its turn: within the hour (every
        # real match was within ±1 s; a day let a later same-text turn in)
        if times and any(t is not None and abs(float(ts) - t) <= 3600 for t in times):
            ids.append(eid)
    con.close()
    return ids


def main() -> int:
    who = requesters()
    member = {r for r, w in who.items() if w == "member"}
    changed, member_tids, member_reqs, owner_texts = stamp_trajectories(who)
    report = {"index_requests": len(who), "member_requests": len(member),
              "trajectory_rows_restamped": changed, "member_trajectories": len(member_tids)}
    for name, path, key in (("foresight", SYS / "foresight" / "predictions.jsonl", "req_id"),
                            ("foresight.1", SYS / "foresight" / "predictions.jsonl.1", "req_id"),
                            ("calibration", SYS / "calibration" / "calibration.jsonl", "req_id"),
                            ("rrf_observations", SYS / "rrf" / "observations.jsonl", "turn")):
        report[name + "_rows_removed"] = (_rewrite_jsonl(path, lambda r, k=key: str(r.get(k) or "") not in member)
                                          if path.exists() else 0)
    eids = member_episode_ids(member_reqs, owner_texts)
    report["episodes_forgotten"] = len(eids)
    from ghost_agent.memory.skills import SkillMemory
    playbook = json.loads((MEM / "skills_playbook.json").read_text(encoding="utf-8"))
    lessons = [x for x in playbook if x.get("source_trajectory_id") in member_tids]
    report["lessons_retracted"] = len(lessons)
    report["lesson_tasks"] = [str(x.get("task", ""))[:80] for x in lessons]
    if APPLY:
        from ghost_agent.memory.episodes import EpisodicMemory
        from ghost_agent.memory.vector import VectorMemory
        shutil.copy2(MEM / "skills_playbook.json", MEM / f"skills_playbook.json.pre-4ml-{STAMP}.bak")
        shutil.copy2(MEM / "episodic_memory.db", MEM / f"episodic_memory.db.pre-4ml-{STAMP}.bak")
        vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")   # deletes by metadata only
        sm = SkillMemory(MEM)
        n = sum(sm.retract_lessons_from_trajectory(t, memory_system=vm, include_correctives=True)
                for t in sorted(member_tids))
        report["lessons_retracted"] = n
        if eids:
            em = EpisodicMemory(MEM)
            report["episodes_forgotten"] = em.delete_episodes(eids, vm, reason="§4ML: a channel member's turn")
    report["applied"] = APPLY
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
