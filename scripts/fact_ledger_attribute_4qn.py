"""§4QN — attribute the ledger's UNATTRIBUTED seeded rows to the owner's own
words, and delete the seeded graph rows that twin a profile value.

The §4QM backfill attributed a seeded fact only to a first-person STATEMENT
CLAUSE carrying every word of the value. Thirteen live rows had none: the
clause was split at a comma ("…, Athens, Greece"), the owner stated the fact
about someone else without "my" ("elena was born at 7 Jan 1982"), the value
is the extractor's paraphrase ("High memory usage is normal …"), or the owner
confirmed the fact in person (§4KY/§4LC, 2026-10-03) and never typed it.

This script takes an EVIDENCE file (JSON, kept OUTSIDE the repo — it quotes
the owner's words): one entry per unattributed row with the owner's clause,
the turn's local time ("YYYY-MM-DD HH:MM"), the request id when the
trajectory has one (a prefix is enough) and an optional `how`. Nothing is
trusted from the file alone: a clause is applied only when an owner turn in
the trajectory corpus at that minute (±2 min) or under that request id
CONTAINS it (case/accent-folded, whitespace-normalised); a row whose clause
is not found stays UNATTRIBUTED and is listed. Containment is the check, not
the statement rule: the operator curates the file, and a clause inside a
question sentence ("the doctor prescribed me ventolin, what is it?") is
accepted on their word. A request-id prefix must be at least 8 characters.
An entry with no clause must carry a `how` that starts with "confirmed by
you" — the ledger renders it. Rows are matched by (store, category, key)
and a value PREFIX among the rows marked `unattributed`; only those are
touched, so a second --apply is a no-op.

Twins: a seeded graph row that carries the SAME value as a seeded profile
row is the same fact dated twice (§4QM replay: "you told me on 2026-09-21"
for a July fact); the backfill no longer seeds them and `--delete-twins`
removes the ones it seeded before that rule — but only when the graph row
adds no evidence of its own: its `said` is empty or equal to the profile
row's and it carries no confirmation. A graph row with its own clause, a
`how` that starts "confirmed by you" (review r2: a confirmation IS
evidence, and a later --delete-twins run has no evidence file to name it),
or one named in the evidence file, is KEPT and listed (review r1: value-only
matching deleted `born_in = Athens` beside `location = Athens`, and the only
row carrying the owner's clause). The ledger is append-only for the live writers — this is the
one-off cleanup the operator approved on 2026-10-22.

    python3 scripts/fact_ledger_attribute_4qn.py --evidence E.json [--delete-twins] [--apply]

Report mode by default. `--apply` needs the memory directory's writer lock
(§4MN: run it in the restart window) and first copies fact_ledger.db to
system/backups/<stamp>-4qn/memory/.
"""
import argparse
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EVIDENCE_KEYS = ("store", "category", "key", "value")


def _norm(text) -> str:
    from ghost_agent.memory.profile import _fold
    return " ".join(_fold(str(text or "")).split())


def _local_epoch(when: str) -> float:
    return time.mktime(_dt.datetime.strptime(when.strip(), "%Y-%m-%d %H:%M").timetuple())


def load_turns(sysd: Path) -> list:
    """(epoch, req_id, request) for every delivered owner turn, oldest first."""
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.core.failure_census import is_owner_turn, _traj_dict
    from ghost_agent.memory.fact_ledger import _epoch
    out = []
    col = TrajectoryCollector(root=sysd / "trajectories", enabled=False)
    for tr in col.iter_trajectories(include_archive=True):
        t = _traj_dict(tr)
        if not is_owner_turn(t):
            continue
        req = str(t.get("user_request") or "")
        if req.strip():
            out.append((_epoch(t.get("timestamp")) or 0.0, str((t.get("extra") or {}).get("req_id") or ""), req))
    out.sort(key=lambda r: r[0])
    return out


def find_turn(turns: list, said: str, when: str, req_id: str):
    """The owner turn that CONTAINS the clause: under the request id when one
    is given, else within ±2 min of the stated local time. None when no
    turn contains it (fail closed: the evidence file is not the corpus)."""
    want = _norm(said)
    if not want:
        return None
    cands = []
    if req_id:
        cands = [t for t in turns if t[1] and t[1].startswith(req_id)]
    if not cands and when:
        at = _local_epoch(when)
        cands = [t for t in turns if abs(t[0] - at) <= 120]
    for t in cands:
        if want in _norm(t[2]):
            return t
    return None


def twins_of(mdir: Path, rows: list, keep_ids=()) -> tuple:
    """(twins, kept): seeded graph rows whose value equals a seeded PROFILE
    row's value in the ledger (the backfill's twin rule applied to what it
    seeded before it). A graph row is a twin only when it adds no evidence of
    its own — `said` empty or equal to the profile row's; otherwise, or when
    the evidence file names it, it is kept and reported."""
    prof = {}
    for r in rows:
        if r["store"] == "profile" and r["event"] == "backfilled":
            prof.setdefault(r["value"].lower(), []).append(r)
    twins, kept = [], []
    for r in rows:
        if r["store"] != "graph" or r["event"] != "backfilled" or r["value"].lower() not in prof:
            continue
        mates = prof[r["value"].lower()]
        if r["id"] in keep_ids:
            kept.append((r, "named in the evidence file"))
        elif r["how"].startswith("confirmed by you"):
            kept.append((r, r["how"]))
        elif r["said"] and all(r["said"] != m["said"] for m in mates):
            kept.append((r, f"carries its own clause {r['said'][:50]!r}"))
        else:
            twins.append(r)
    return twins, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME", ""))
    ap.add_argument("--evidence", help="JSON list of {store, category, key, value(prefix), said, when, req_id, how}")
    ap.add_argument("--delete-twins", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not a.home:
        print("GHOST_HOME not set", file=sys.stderr); return 2
    if not a.evidence and not a.delete_twins:
        print("nothing asked: give --evidence and/or --delete-twins", file=sys.stderr); return 2
    sysd = Path(a.home).expanduser() / "system"; mdir = sysd / "memory"
    from ghost_agent.memory.fact_ledger import FactLedger, clip_said, redact_said
    led = FactLedger(mdir)
    with led._conn() as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute("SELECT * FROM ledger ORDER BY id")]

    updates = []   # (id, said, req_id, how, label)
    missing = []
    named = set()  # every row an evidence entry names — never deleted as a twin, this run or a later one
    if a.evidence:
        evidence = json.loads(Path(a.evidence).read_text(encoding="utf-8"))
        turns = load_turns(sysd)
        print(f"owner turns in the corpus: {len(turns)}; evidence entries: {len(evidence)}")
        for ev in evidence:
            for k in EVIDENCE_KEYS:
                if not str(ev.get(k) or "").strip():
                    raise SystemExit(f"evidence entry without {k!r}: {ev}")
            field = f"{ev['category']}.{ev['key']} = {ev['value']}"
            same = [r for r in rows if r["store"] == ev["store"] and r["category"] == ev["category"].strip().lower()
                    and r["key"] == ev["key"].strip().lower() and r["value"].lower().startswith(ev["value"].strip().lower())]
            named.update(r["id"] for r in same)
            hits = [r for r in same if r["how"] == "unattributed"]
            if not hits:
                if same:
                    print(f"ALREADY  id={same[0]['id']} {field} (how={same[0]['how']!r}, said={same[0]['said'][:40]!r})")
                else:
                    missing.append(f"NO ROW {field}")
                continue
            if len(hits) > 1:
                missing.append(f"AMBIGUOUS {field} ({len(hits)} unattributed rows)"); continue
            r = hits[0]
            said = str(ev.get("said") or "").strip()
            how = str(ev.get("how") or "").strip()
            if how.startswith("unattributed"):
                raise SystemExit(f"evidence entry re-marks a row unattributed: {field}")
            req_pref = str(ev.get("req_id") or "").strip()
            if req_pref and len(req_pref) < 8:
                raise SystemExit(f"request-id prefix shorter than 8 characters: {field} {req_pref!r}")
            if not said:
                if not how.startswith("confirmed by you"):
                    missing.append(f"NO CLAUSE, NO CONFIRMATION {field}"); continue
                updates.append((r["id"], "", "", how, f"CONFIRM  id={r['id']} {field} ← {how}")); continue
            turn = find_turn(turns, said, str(ev.get("when") or ""), req_pref)
            if turn is None:
                missing.append(f"NOT IN CORPUS {field} ← {said[:60]!r} @ {ev.get('when')} {ev.get('req_id') or ''}"); continue
            day = _dt.datetime.fromtimestamp(turn[0]).strftime("%Y-%m-%d %H:%M")
            updates.append((r["id"], clip_said(redact_said(said)), turn[1][:64], how,
                            f"ATTRIBUTE id={r['id']} {field} ← {day} req={turn[1][:12] or '-'} said={said[:70]!r}"
                            + (f" how={how!r}" if how else "")))
    for u in updates:
        print(u[4])
    for m in missing:
        print("MISSING ", m)
    still = [r for r in rows if r["how"] == "unattributed" and r["id"] not in {u[0] for u in updates}]
    for r in still:
        print(f"STILL UNATTRIBUTED id={r['id']} {r['category']}.{r['key']} = {r['value'][:50]}")

    twins, kept = twins_of(mdir, rows, keep_ids=named) if a.delete_twins else ([], [])
    for r in twins:
        why = f"same clause as the profile row: {r['said'][:50]!r}" if r["said"] else "no clause of its own"
        print(f"TWIN     id={r['id']} graph.{r['key']} = {r['value']} (a seeded profile row holds it; {why})")
    for r, why in kept:
        print(f"KEPT     id={r['id']} graph.{r['key']} = {r['value']} — {why}")

    if not a.apply:
        print(f"report mode: {len(updates)} to attribute, {len(missing)} missing, {len(twins)} twins to delete — nothing written")
        return 0
    if not updates and not twins:
        print("nothing to do"); return 0
    from ghost_agent.memory.store_lock import assert_no_other_writer
    assert_no_other_writer(mdir, "fact_ledger_attribute_4qn.py")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bdir = sysd / "backups" / f"{stamp}-4qn" / "memory"
    bdir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(led.path)) as src, sqlite3.connect(str(bdir / led.path.name)) as dst:
        src.backup(dst)
    print(f"backup: {bdir / led.path.name}")
    with led._lock, led._conn() as c:
        n_upd = 0
        for rid, said, req_id, how, _ in updates:
            n_upd += c.execute("UPDATE ledger SET said=?, req_id=?, how=? WHERE id=? AND how='unattributed'",
                               (said, req_id, how[:80], rid)).rowcount
        n_del = 0
        if twins:
            n_del = c.execute(f"DELETE FROM ledger WHERE id IN ({','.join('?' * len(twins))})",
                              [r["id"] for r in twins]).rowcount
    print(f"applied: {n_upd} rows attributed, {n_del} twin rows deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
