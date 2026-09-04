#!/usr/bin/env python3
"""One-time repair: convert stored age SNAPSHOTS into temporal anchors.

``memory/temporal.py`` stops new snapshots from being written, but it is
dead code against everything already in the stores — the profile line that
triggered this work ("Leonidas (4 months old)", stated 2026-07-07) is still
sitting on disk being injected into every system prompt. This script
rewrites the existing corpus.

WHAT IT WILL NOT DO
    * Guess. Every rewrite needs a said_at — the date the age was TRUE —
      and a rewrite anchored to the wrong date is worse than the snapshot,
      because it looks authoritative. Rows carrying their own timestamp
      (vector, graph) use it; the profile has no timestamps at all, so the
      date must be supplied with --said-at or inferred and REVIEWED.
    * Write anything without --apply. Default is a dry run.
    * Open Chroma directly. A second PersistentClient against the live
      Chroma dir risks HNSW corruption, so vector rewrites go through the
      owning process via POST /api/memory/correct (see
      VectorMemory.correct_fragment). Requires the agent to be running.
    * Sweep blindly. An age is not always a claim about a person: the live
      graph holds ``wilson evolution youth IS_BEST_FOR "9-year-old"``,
      where the age is a product CATEGORY that does not decay. Candidates
      are printed for review and selected per target, not applied en masse.

USAGE
    # See everything that would change, and where its said_at came from
    python3 scripts/repair_temporal_anchors.py

    # Repair the profile (the load-bearing store) with an explicit date
    python3 scripts/repair_temporal_anchors.py --targets profile \\
        --said-at 2026-07-07 --apply

    # Repair vector rows through the running agent
    python3 scripts/repair_temporal_anchors.py --targets vector --apply

    # Recover per-value as_of provenance (RUN AFTER DEPLOYING THE CODE —
    # an agent on the older ProfileMemory renders a stamped value as a raw
    # dict into every system prompt)
    python3 scripts/repair_temporal_anchors.py --targets profile-stamps --apply
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.memory.temporal import anchor  # noqa: E402

DEFAULT_HOME = Path(os.getenv("GHOST_HOME", str(Path.home() / "Data/AI/Data")))


def _memory_dir(home: Path) -> Path:
    return home / "system" / "memory"


# ── candidate discovery ─────────────────────────────────────────────────

# A date far enough in the past that any age phrase anchors to something
# different from itself. Used ONLY to answer "does this text carry an age?"
# — never to build a rewrite.
_PROBE = datetime.date(2000, 1, 1)


def _carries_age(text: str) -> bool:
    return isinstance(text, str) and anchor(text, _PROBE) != text


def _strings_in(obj):
    """Every string leaf of a nested JSON structure.

    Detection must walk the VALUES, never a json.dumps() of the container:
    anchor() deliberately refuses to rewrite quoted spans (a quoted string
    is a record, not a claim), and in a JSON blob every value is inside
    quotes — so a dump-based probe reports "no ages found" on a document
    that is full of them. That false negative is what made the first run of
    this script print "profile (0 candidates)" for the very row the whole
    change exists to fix.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings_in(v)


def _profile_candidates(mem: Path, said_at):
    """(path-in-json, old, new|None, said_at_note) for the profile store.

    The profile holds no timestamps, so said_at cannot come from the row.
    Candidates are DETECTED regardless — a missing date must not hide a
    broken row — and ``new`` is None when no date is available, which the
    caller reports as work still to do rather than as nothing to do.
    """
    pf = mem / "user_profile.json"
    if not pf.exists():
        return []
    data = json.loads(pf.read_text(encoding="utf-8"))
    if said_at:
        when, src = said_at, "--said-at"
    else:
        when, src = _infer_said_at(mem)

    out = []
    for cat, fields in data.items():
        if not isinstance(fields, dict):
            continue
        for key, val in fields.items():
            values = val if isinstance(val, list) else [val]
            for i, v in enumerate(values):
                if not _carries_age(v):
                    continue
                idx = i if isinstance(val, list) else None
                new = anchor(v, when) if when else None
                note = f"{when} ({src})" if when else f"UNKNOWN — {src}"
                out.append(((cat, key, idx), v, new, note))
    return out


def _infer_said_at(mem: Path):
    """Earliest contradiction-log entry carrying an age, as a fallback
    said_at. Returns (date|None, source-description).

    The contradiction log stamps every entry, so it is the only store that
    can date a profile fact after the fact. The EARLIEST match is the right
    pick: it is when the age was first recorded, and therefore closest to
    when it was true.
    """
    log = mem / "contradiction_log.json"
    if not log.exists():
        return None, "no contradiction_log.json"
    try:
        entries = json.loads(log.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"unreadable contradiction_log.json ({e})"
    best = None
    for e in entries if isinstance(entries, list) else []:
        if not any(_carries_age(s) for s in _strings_in(e)):
            continue
        try:
            d = datetime.date.fromisoformat(str(e.get("timestamp", ""))[:10])
        except Exception:
            continue
        best = d if best is None or d < best else best
    if best is None:
        return None, "no age-bearing entry in contradiction_log.json"
    return best, "inferred: earliest age-bearing contradiction_log entry"


def _stamp_candidates(mem: Path):
    """(cat, key, value, recovered_as_of|None, evidence) for profile values
    that carry no provenance.

    The profile never recorded write times, so the dates have to be
    RECOVERED, not invented. The vector store mints a derived fact for
    every profile write (``User <key> is <value>``) and stamps it, so the
    EARLIEST row that mentions a value is the best evidence of when that
    value was learned. A value with no matching row gets None and is
    reported unrecovered — back-filling it with "now" would fabricate
    provenance, which is worse than admitting there is none.
    """
    pf = mem / "user_profile.json"
    db = mem / "chroma.sqlite3"
    if not pf.exists():
        return []
    data = json.loads(pf.read_text(encoding="utf-8"))

    rows = []
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            cur = con.cursor()
            cur.execute("SELECT id, key, string_value FROM embedding_metadata "
                        "WHERE key IN ('chroma:document', 'timestamp', 'type')")
            by_id: dict = {}
            for rid, k, v in cur.fetchall():
                by_id.setdefault(rid, {})[k] = v
        finally:
            con.close()
        for m in by_id.values():
            doc, ts = m.get("chroma:document"), m.get("timestamp")
            # Documents, episodes and skill lessons are not records of a
            # profile WRITE — they merely contain words.
            if not doc or not ts or (m.get("type") or "") in {
                    "document", "episode", "skill"}:
                continue
            rows.append((" ".join(doc.lower().split()), ts))

    # Second evidence source: the contradiction log. It is admissible for
    # the same reason the vector rows are and a document is not — it is a
    # record of facts being ASSERTED or SUPERSEDED, each with a timestamp.
    # A PostgreSQL manual that happens to contain the word "nova" records
    # nothing about the profile.
    log_entries = []
    log_path = mem / "contradiction_log.json"
    if log_path.exists():
        try:
            for e in json.loads(log_path.read_text(encoding="utf-8")) or []:
                ts = str(e.get("timestamp", ""))
                if not ts:
                    continue
                blob = " ".join(_strings_in(e)).lower()
                log_entries.append((blob, ts))
        except Exception:
            pass

    # A short value ("nova", "dtrace", "Vasilis") matches by accident. Only
    # a value distinctive enough to identify itself is attributable.
    _MIN_ATTRIBUTABLE = 12

    def _recover(key: str, value: str):
        """Earliest row that IS the derived profile fact for this key.

        Matched STRUCTURALLY, on the exact shape `tool_update_profile`
        mints — ``User <key> is <value>`` — not by looking for the value's
        text somewhere in the corpus. The first version of this did the
        latter and the dry run showed what that buys: `name = Vasilis` was
        dated from a chess-coaching prompt that happens to say "Vasilis",
        `debugging_tool_macos = dtrace` and `home_lab_worker_node = nova`
        from a PostgreSQL manual that happens to contain "dtrace" and
        "nova". A coincidental mention is not evidence of a write, and a
        confidently wrong date is worse than an admitted gap.
        """
        prefix = f"user {str(key).strip().lower()} is "
        hits = [(t, d) for d, t in rows if d.startswith(prefix)]
        if hits:
            t, d = min(hits)
            return t, f"vector fact :: {d[:80]}"

        needle = " ".join(str(value).lower().split())
        if len(needle) < _MIN_ATTRIBUTABLE:
            return None, ("no minted vector fact; value too short to "
                          "attribute from the contradiction log")

        # The log preserves superseded vector facts verbatim, so the SAME
        # structural form is often still findable there. Prefer it, and say
        # which kind of match produced the date: "user <key> is <value>" is
        # a record of this key being written, while a bare value match only
        # proves the text was around at that time.
        exact = f"user {str(key).strip().lower()} is {needle}"
        for label, probe in (("log, minted-fact form", exact),
                             ("log, value mention", needle)):
            hits = [(t, b) for b, t in log_entries if probe in b]
            if hits:
                t, b = min(hits)
                at = b.find(probe)
                return t, f"{label} :: …{b[max(0, at - 20):at + 60]}…"
        return None, "no minted vector fact and no contradiction-log entry"

    out = []
    for cat, fields in data.items():
        if not isinstance(fields, dict):
            continue
        for key, val in fields.items():
            items = val if isinstance(val, list) else [val]
            for item in items:
                if isinstance(item, dict) and "v" in item:
                    continue  # already stamped
                if not isinstance(item, str):
                    continue
                when, evidence = _recover(key, item)
                out.append((cat, key, item, when, evidence))
    return out


def _vector_candidates(mem: Path):
    """(id, old, new, ts) for Chroma rows, read-only. Document, episode and
    skill rows are EXCLUDED: a document's ages belong to the document's own
    timeline, and a skill lesson is a record of what was done."""
    db = mem / "chroma.sqlite3"
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cur.execute("SELECT id, key, string_value FROM embedding_metadata "
                    "WHERE key IN ('chroma:document', 'timestamp', 'type')")
        rows: dict = {}
        for rid, k, v in cur.fetchall():
            rows.setdefault(rid, {})[k] = v
    finally:
        con.close()

    out = []
    for meta in rows.values():
        doc, ts = meta.get("chroma:document"), meta.get("timestamp")
        if not doc or not ts:
            continue
        if (meta.get("type") or "") in {"document", "episode", "skill"}:
            continue
        new = anchor(doc, ts)
        if new != doc:
            out.append((meta.get("type"), doc, new, ts))
    return out


_AGE_PREDICATES = {"IS_AGE", "HAS_AGE", "AGE", "IS_AGED", "AGED"}


def _graph_candidates(mem: Path):
    """(rowid, subject, old_pred, new_pred, old_obj, new_obj, ts, verdict).

    An age in a triplet is only a claim about the subject when the
    PREDICATE says so. The live graph holds both shapes:

        thodoris             IS_AGE      "9 years old"   <- a decaying claim
        wilson evolution …   IS_BEST_FOR "9-year-old"    <- a product category

    The second does not decay — a ball stays suitable for nine-year-olds —
    and rewriting it would be a false repair. So only an age-ish predicate
    whose object is essentially JUST an age gets repaired, and that repair
    changes the predicate too: leaving ``IS_AGE`` pointing at a birth date
    would be a worse record than the one it replaced.
    """
    db = mem / "knowledge_graph.db"
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cur.execute("SELECT rowid, subject, predicate, object, timestamp "
                    "FROM triplets WHERE valid_until IS NULL")
        rows = cur.fetchall()
    finally:
        con.close()

    out = []
    for rowid, s, p, o, ts in rows:
        if not isinstance(o, str):
            continue
        pred_is_age = str(p or "").upper() in _AGE_PREDICATES
        probe = _with_predicate_cue(o) if pred_is_age else o
        if not _carries_age(probe):
            continue
        anchored = anchor(probe, ts)
        # "essentially just an age": anchoring consumed the whole object.
        obj_is_pure_age = anchored.lower().startswith("born ")
        if pred_is_age and obj_is_pure_age:
            out.append((rowid, s, p, "BORN", o,
                        anchored[len("born "):].strip(), ts, "repair"))
        else:
            out.append((rowid, s, p, p, o, anchored, ts,
                        "SKIP — age is a category here, not a claim about "
                        "the subject; it does not decay"))
    return out


def _with_predicate_cue(obj: str) -> str:
    """Supply the age cue an age-PREDICATE already implies.

    anchor() deliberately requires an explicit cue in the text ("old",
    "age", "yo") because a bare "4 months" is far more often a duration.
    In a triplet that rule is too strict in one direction: the live graph
    holds ``thodoris IS_AGE "9 years"``, where the cue lives in the
    predicate and the object alone reads as a duration. A bare number
    under an age predicate means years.

    Only ever applied when the predicate is age-ish, so free text keeps
    the conservative rule.
    """
    text = str(obj or "").strip()
    if not text or _carries_age(text):
        return text
    if re.fullmatch(r"\d{1,3}", text):
        return f"{text} years old"
    return f"{text} old"


# ── application ─────────────────────────────────────────────────────────

def _backup(path: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = path.with_suffix(path.suffix + f".pre-temporal-{stamp}")
    shutil.copy2(path, dest)
    return dest


def _apply_profile(mem: Path, cands) -> int:
    pf = mem / "user_profile.json"
    print(f"  backup: {_backup(pf).name}")
    data = json.loads(pf.read_text(encoding="utf-8"))
    for (cat, key, idx), old, new, _src in cands:
        if idx is None:
            data[cat][key] = new
        else:
            data[cat][key][idx] = new
    tmp = pf.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, pf)
    return len(cands)


def _apply_graph(mem: Path, cands) -> int:
    """Rewrite each repairable edge, or RETIRE it when the repaired form
    already exists.

    ``triplets`` has a UNIQUE(subject, predicate, object) constraint, so an
    edge whose repaired form is already present cannot be updated into
    place — and it must not be, because that row IS the duplicate. The
    store's own mechanism for a superseded edge is a ``valid_until`` stamp
    (graph.py expires rather than deletes, and every read path filters on
    ``valid_until IS NULL``), so that is what a collision gets. Found by
    running this on the live graph after an earlier pass had already
    created the target edge: the first version raised IntegrityError and
    rolled back.
    """
    import time

    db = mem / "knowledge_graph.db"
    print(f"  backup: {_backup(db).name}")
    con = sqlite3.connect(str(db))
    now = time.time()
    done = 0
    try:
        for rowid, s, _p, new_p, _old, new_o, _ts, _v in cands:
            try:
                with con:
                    con.execute(
                        "UPDATE triplets SET predicate = ?, object = ? "
                        "WHERE rowid = ?", (new_p, new_o, rowid))
                print(f"  rewrote: {s} -{new_p}-> {new_o}")
            except sqlite3.IntegrityError:
                with con:
                    con.execute(
                        "UPDATE triplets SET valid_until = ? WHERE rowid = ?",
                        (now, rowid))
                print(f"  retired (repaired form already present): "
                      f"{s} -{new_p}-> {new_o}")
            done += 1
    finally:
        con.close()
    return done


def _apply_vector(cands, base_url: str, api_key: str) -> int:
    """Rewrite via the OWNING process. Opening Chroma from here would risk
    HNSW corruption against the live index."""
    import urllib.request

    done = 0
    for _type, old, new, _ts in cands:
        payload = json.dumps({"match": old, "replacement": new}).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/memory/correct", data=payload,
            headers={"Content-Type": "application/json", "X-Ghost-Key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
            print(f"  ok: {body[:160]}")
            done += 1
        except Exception as e:
            print(f"  FAILED ({e}): {old[:80]}")
    return done


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME)
    ap.add_argument("--targets", default="profile",
                    help="comma-separated: profile,profile-stamps,vector,graph "
                         "(default: profile). `profile` rewrites age snapshots "
                         "into anchors; `profile-stamps` recovers per-value "
                         "as_of provenance from the vector store.")
    ap.add_argument("--said-at", default=None,
                    help="YYYY-MM-DD the profile's ages were true. The "
                         "profile stores no timestamps, so without this the "
                         "date is inferred and must be reviewed.")
    ap.add_argument("--apply", action="store_true", help="write the changes")
    ap.add_argument("--base-url", default=os.getenv("GHOST_BASE_URL",
                                                    "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    mem = _memory_dir(args.home)
    if not mem.is_dir():
        print(f"no memory dir at {mem}", file=sys.stderr)
        return 2

    said_at = None
    if args.said_at:
        try:
            said_at = datetime.date.fromisoformat(args.said_at)
        except ValueError:
            print(f"--said-at must be YYYY-MM-DD, got {args.said_at!r}",
                  file=sys.stderr)
            return 2

    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    unknown = targets - {"profile", "profile-stamps", "vector", "graph"}
    if unknown:
        print(f"unknown target(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    total = 0

    if "profile" in targets:
        cands = _profile_candidates(mem, said_at)
        print(f"\n=== profile ({len(cands)} candidate(s)) ===")
        for (cat, key, idx), old, new, src in cands:
            where = f"{cat}.{key}" + (f"[{idx}]" if idx is not None else "")
            print(f"  {where}\n    said_at: {src}\n    OLD: {old}\n"
                  f"    NEW: {new if new else '(needs --said-at)'}")
        undated = [c for c in cands if c[2] is None]
        if undated:
            print(f"  {len(undated)} row(s) carry an age but have no date to "
                  f"anchor to — pass --said-at YYYY-MM-DD. NOT repaired.")
        writable = [c for c in cands if c[2] is not None]
        if writable and args.apply:
            total += _apply_profile(mem, writable)

    if "profile-stamps" in targets:
        cands = _stamp_candidates(mem)
        recoverable = [c for c in cands if c[3]]
        print(f"\n=== profile-stamps ({len(cands)} unstamped value(s), "
              f"{len(recoverable)} with recoverable provenance) ===")
        for cat, key, value, when, evidence in cands:
            print(f"  {cat}.{key} = {str(value)[:70]}")
            print(f"    as_of: {when or 'UNRECOVERED'}  [{evidence}]")
        if recoverable and args.apply:
            from ghost_agent.memory.profile import ProfileMemory
            # ⚠ ORDER MATTERS, and only in this direction. New code reads
            # legacy bare values fine (load() unwraps, which is identity on
            # a string), but OLD code reading a STAMPED value renders the
            # dict repr straight into the system prompt — "{'v': 'Athens,
            # Greece', 'as_of': …}" — and hands the same dict to
            # tool_check_location. So: deploy the code, THEN stamp.
            print("\n  ⚠ Deploy the layer-3 code BEFORE stamping. An agent "
                  "still running the older\n    ProfileMemory renders a "
                  "stamped value as a raw dict into every prompt.\n"
                  "    (New code reads unstamped values fine, so "
                  "code-then-data is always safe.)\n")
            pf = mem / "user_profile.json"
            print(f"  backup: {_backup(pf).name}")
            pm = ProfileMemory(mem)
            # Earliest recovered date per key wins: stamp() only fills
            # UNSTAMPED items, so the first write for a key is the one that
            # lands and a later, newer attribution cannot overwrite it.
            for cat, key, _v, when, _e in sorted(recoverable, key=lambda c: c[3]):
                n = pm.stamp(cat, key, when)
                if n:
                    print(f"  stamped {cat}.{key} ({n} item(s)) as_of {when[:10]}")
                    total += n
        elif not recoverable:
            print("  nothing to stamp — no value could be dated from the "
                  "vector store. NOT back-filled with today.")

    if "vector" in targets:
        cands = _vector_candidates(mem)
        print(f"\n=== vector ({len(cands)} candidate(s)) ===")
        for _t, old, new, ts in cands:
            print(f"  said_at: {ts} (row timestamp)\n    OLD: {old}\n    NEW: {new}")
        if cands and args.apply:
            key = args.api_key or _read_api_key()
            if not key:
                print("  SKIPPED: no API key (set GHOST_API_KEY or --api-key); "
                      "vector rewrites must go through the running agent.")
            else:
                total += _apply_vector(cands, args.base_url, key)

    if "graph" in targets:
        cands = _graph_candidates(mem)
        repairs = [c for c in cands if c[7] == "repair"]
        print(f"\n=== graph ({len(cands)} candidate(s), "
              f"{len(repairs)} repairable) ===")
        for _rid, subj, old_p, new_p, old_o, new_o, ts, verdict in cands:
            print(f"  {subj}\n    said_at: {ts} (edge timestamp)"
                  f"\n    OLD: -{old_p}-> {old_o}")
            if verdict == "repair":
                print(f"    NEW: -{new_p}-> {new_o}")
            else:
                print(f"    {verdict}")
        if repairs and args.apply:
            total += _apply_graph(mem, repairs)

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: "
          f"{total if args.apply else 'no'} change(s) written.")
    if not args.apply:
        print("Re-run with --apply once the said_at attributions above look right.")
    return 0


def _read_api_key():
    for p in (Path.home() / "Data/AI/.ghost_api_key",):
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return os.getenv("GHOST_API_KEY")


if __name__ == "__main__":
    raise SystemExit(main())
