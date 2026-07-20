#!/usr/bin/env python3
"""One-shot scrub of persisted stores whose writes predate the 2026-07-20
redaction fixes (docs/audit_fixes.html Round 10): lowercase/colon secret
assignments (`password: hunter2`, `db_password = hunter2`, `api_key: sk_…`)
passed every rule, so the trajectory corpus, LLM recordings, and the
selfhood diary may hold raw secrets.

Usage:
    PYTHONPATH=src python scripts/scrub_secrets.py            # dry-run report
    PYTHONPATH=src python scripts/scrub_secrets.py --apply    # rewrite in place

Behavior:
- Distill-owned stores (trajectories/, llm_recordings/) are scrubbed with
  distill.redact.redact_text — their native writer-side redactor.
- Selfhood stores use selfhood.autobiographical.redact_pii (redact_text's
  path/onion rules would mangle diary prose that legitimately names paths).
- JSONL lines are parsed and every string value is redacted recursively;
  unparseable lines are left byte-identical (counted, never dropped).
- --apply copies each to-be-modified file into
  $GHOST_HOME/system/pre_scrub_backup_<ts>/ (0700/0600 — it CONTAINS the
  secrets; delete it once satisfied) then writes tmp + os.replace.
- Safe against a live agent appending to the active day file: the file is
  re-statted just before replace and any appended tail is scrubbed and
  carried over.
"""

import argparse
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path


def _ghost_home() -> Path:
    home = os.environ.get("GHOST_HOME", "").strip()
    if not home:
        sys.exit("GHOST_HOME is not set — refusing to guess at live data.")
    return Path(home)


def _walk_redact(obj, redact_fn, stats):
    if isinstance(obj, str):
        red = redact_fn(obj)
        if red != obj:
            stats["strings_changed"] += 1
        return red
    if isinstance(obj, dict):
        return {k: _walk_redact(v, redact_fn, stats) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_redact(v, redact_fn, stats) for v in obj]
    return obj


def scrub_jsonl_bytes(raw: bytes, redact_fn, stats) -> bytes:
    out_lines = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["unparseable_lines"] += 1
            out_lines.append(line)
            continue
        before = stats["strings_changed"]
        rec = _walk_redact(rec, redact_fn, stats)
        if stats["strings_changed"] != before:
            stats["lines_changed"] += 1
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        else:
            out_lines.append(line)
    tail_nl = "\n" if raw.endswith(b"\n") else ""
    return ("\n".join(out_lines) + tail_nl).encode("utf-8")


def scrub_text_bytes(raw: bytes, redact_fn, stats) -> bytes:
    text = raw.decode("utf-8", "replace")
    red = redact_fn(text)
    if red != text:
        stats["strings_changed"] += 1
        stats["lines_changed"] += 1
    return red.encode("utf-8")


def process_file(path: Path, redact_fn, apply: bool, backup_root: Path,
                 gh: Path, totals: dict) -> None:
    stats = {"strings_changed": 0, "lines_changed": 0, "unparseable_lines": 0}
    raw = path.read_bytes()
    if path.suffix in (".jsonl", ".json"):
        scrubbed = scrub_jsonl_bytes(raw, redact_fn, stats) \
            if path.suffix == ".jsonl" else scrub_text_bytes(raw, redact_fn, stats)
    else:
        scrubbed = scrub_text_bytes(raw, redact_fn, stats)

    totals["files_scanned"] += 1
    totals["unparseable_lines"] += stats["unparseable_lines"]
    if scrubbed == raw:
        return
    totals["files_changed"] += 1
    totals["lines_changed"] += stats["lines_changed"]
    totals["strings_changed"] += stats["strings_changed"]
    rel = path.relative_to(gh)
    print(f"  {'WOULD SCRUB' if not apply else 'SCRUB'} {rel}: "
          f"{stats['lines_changed']} line(s), "
          f"{stats['strings_changed']} string(s)"
          + (f", {stats['unparseable_lines']} unparseable kept"
             if stats['unparseable_lines'] else ""))
    if not apply:
        return

    bpath = backup_root / rel
    bpath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, bpath)
    os.chmod(bpath, stat.S_IRUSR | stat.S_IWUSR)

    tmp = path.with_suffix(path.suffix + ".scrub-tmp")
    tmp.write_bytes(scrubbed)
    # Live-append guard: carry over (scrubbed) anything the agent appended
    # between our read and this replace.
    grown = path.stat().st_size - len(raw)
    if grown > 0:
        with path.open("rb") as f:
            f.seek(len(raw))
            delta = f.read()
        with tmp.open("ab") as f:
            f.write(scrub_jsonl_bytes(delta, redact_fn, stats)
                    if path.suffix == ".jsonl" else delta)
        print(f"    carried over {grown} live-appended byte(s)")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite files (default: dry-run report)")
    args = ap.parse_args()

    from ghost_agent.distill.redact import redact_text
    from ghost_agent.selfhood.autobiographical import redact_pii

    gh = _ghost_home()
    backup_root = gh / "system" / f"pre_scrub_backup_{int(time.time())}"
    if args.apply:
        backup_root.mkdir(parents=True, exist_ok=True)
        os.chmod(backup_root, stat.S_IRWXU)

    groups = [
        ("trajectories", sorted((gh / "system" / "trajectories").rglob("*.jsonl")),
         redact_text),
        ("llm_recordings", sorted((gh / "system" / "llm_recordings").rglob("*.jsonl")),
         redact_text),
        ("selfhood", [p for p in [
            gh / "system" / "selfhood" / "autobiographical.jsonl",
            gh / "system" / "selfhood" / "narrative.history.jsonl",
            gh / "system" / "selfhood" / "narrative.md",
            gh / "system" / "selfhood" / "state.json",
            gh / "system" / "selfhood" / "values.json",
        ] if p.exists()], redact_pii),
    ]

    totals = {"files_scanned": 0, "files_changed": 0,
              "lines_changed": 0, "strings_changed": 0,
              "unparseable_lines": 0}
    for name, files, fn in groups:
        print(f"[{name}] {len(files)} file(s)")
        for p in files:
            process_file(p, fn, args.apply, backup_root, gh, totals)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"\n{mode}: {totals['files_changed']}/{totals['files_scanned']} "
          f"file(s) with secrets — {totals['lines_changed']} line(s), "
          f"{totals['strings_changed']} string value(s)"
          + (f"; {totals['unparseable_lines']} unparseable line(s) left as-is"
             if totals['unparseable_lines'] else ""))
    if args.apply and totals["files_changed"]:
        print(f"Backups (contain the ORIGINAL secrets): {backup_root}\n"
              f"Delete after review: rm -rf {backup_root}")
    elif args.apply:
        backup_root.rmdir()


if __name__ == "__main__":
    main()
