#!/usr/bin/env python3
"""Fetch Google's FRAMES benchmark into a GAIA-shaped tasks JSONL — no account.

WHY THIS EXISTS. Everything about this agent is measured internally and nothing
comparatively, so "how good is it?" has no falsifiable answer. The GAIA harness
was built and piloted 8/8 and is blocked forever on `huggingface-cli login`:
GAIA is a GATED dataset, and the operator's standing constraint is no keyed
APIs and no accounts, ever. Measured 2026-08-10 — `datasets-server` returns
HTTP 401 for `gaia-benchmark/GAIA` and HTTP 200 for the datasets below, so the
block is real and not a tooling problem.

⚠ The gate is a LICENSE AGREEMENT, not an inconvenience. Hunting for an
ungated GAIA mirror would circumvent the terms the gate exists to enforce, so
this goes to a genuinely open benchmark instead.

WHY FRAMES, having probed the alternatives:
  AssistantBench  OPEN, and the closest analogue to GAIA — but its tasks need
                  live browsing of arbitrary commercial sites ("gyms near
                  Tompkins Square Park with classes before 7am"). Under
                  Tor-only egress that measures TOR REACHABILITY, not agent
                  quality, and its answers age as real-world schedules change.
  GPQA            gated (401). Same wall as GAIA.
  FRAMES          OPEN, 824 multi-hop questions over Wikipedia, and crucially
                  it ships the GOLD DOCUMENT LINKS with each question. That
                  allows an ORACLE mode which removes the retrieval confound
                  and measures the multi-hop reasoning and synthesis this
                  agent actually does. Published baselines exist, so the
                  number is comparable rather than merely internal.

Verified before relying on it: Wikipedia is reachable over Tor (HTTP 200 in
3.2s via 127.0.0.1:9050), so oracle mode can actually run under the operator's
egress rules.

OUTPUT is GAIA-shaped (`task_id` / `Level` / `Question` / `Final answer`) so
`scripts/gaia_eval.py --tasks-file` runs it unchanged, and `gaia_scorer`'s
official normalisation grades it — FRAMES answers are the same short-factual
shape the scorer was written for. No second runner, no second scorer.

    python scripts/frames_fetch.py --out $GHOST_HOME/system/eval/frames.jsonl
    python scripts/frames_fetch.py --limit 100 --seed 7 --oracle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "google/frames-benchmark"
CONFIG, SPLIT = "default", "test"
_BASE = "https://datasets-server.huggingface.co"
# The rows endpoint caps a page at 100.
_PAGE = 100


def _get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ghost-frames/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        if e.code in (401, 403):
            # ⚠ LOUD, not a fallback. A gated dataset silently degrading to a
            # partial or empty pull is how a benchmark number gets published
            # against data nobody actually fetched.
            raise SystemExit(
                f"AUTH REQUIRED for {DATASET} (HTTP {e.code}). This script "
                f"exists precisely because accounts are not permitted; it "
                f"will not authenticate.\n  {body}")
        raise SystemExit(f"HTTP {e.code} from datasets-server: {body}")


def fetch_rows(limit: int | None = None) -> list[dict]:
    """Page through the split. Anonymous; no token is ever sent."""
    q = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG,
                                "split": SPLIT, "offset": 0, "length": 1})
    total = int(_get(f"{_BASE}/rows?{q}").get("num_rows_total") or 0)
    if not total:
        raise SystemExit("datasets-server reported 0 rows — refusing to write "
                         "an empty task set")
    want = total if limit is None else min(limit, total)
    rows: list[dict] = []
    for off in range(0, total, _PAGE):
        if len(rows) >= want and limit is not None:
            break
        q = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG,
                                    "split": SPLIT, "offset": off,
                                    "length": _PAGE})
        page = _get(f"{_BASE}/rows?{q}").get("rows") or []
        if not page:
            break
        rows.extend(r["row"] for r in page)
        print(f"  fetched {len(rows)}/{total}", file=sys.stderr)
        time.sleep(0.2)          # be a polite anonymous client
    return rows


def _links(row: dict) -> list[str]:
    """Gold Wikipedia articles. `wiki_links` is a stringified list in the
    served rows, with numbered columns as the fallback."""
    raw = row.get("wiki_links")
    out: list[str] = []
    if isinstance(raw, list):
        out = [str(x) for x in raw]
    elif isinstance(raw, str) and raw.strip().startswith("["):
        try:
            out = [str(x) for x in json.loads(raw.replace("'", '"'))]
        except Exception:                                    # noqa: BLE001
            out = []
    if not out:
        for i in range(1, 12):
            v = row.get(f"wikipedia_link_{i}")
            if v and str(v).lower() not in ("none", "nan"):
                out.append(str(v))
    seen, uniq = set(), []
    for u in out:
        if u.startswith("http") and u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def to_task(row: dict, idx: int, *, oracle: bool) -> dict | None:
    q = str(row.get("Prompt") or "").strip()
    a = str(row.get("Answer") or "").strip()
    if not q or not a:
        return None
    docs = _links(row)
    question = q
    if oracle:
        # ORACLE MODE. Naming the gold articles converts "can you find it"
        # into "can you reason across it" — the retrieval leg is exactly the
        # part Tor-only egress would confound, and it is not the part under
        # test. The agent must still FETCH and read them.
        question = (q + "\n\nUse these source articles:\n"
                    + "\n".join(f"- {u}" for u in docs))
    return {
        # Stable and content-derived, so a re-fetch produces identical ids and
        # two runs remain comparable.
        "task_id": "frames-" + hashlib.sha256(q.encode()).hexdigest()[:12],
        # ⚠ ALWAYS 1. FRAMES has no GAIA-style level, and inventing one from
        # hop count would let `gaia_eval --level` silently filter on a scale
        # this dataset never defined. Difficulty lives in the preserved
        # fields below.
        "Level": 1,
        "Question": question,
        "Final answer": a,
        "frames_reasoning_type": str(row.get("reasoning_types") or ""),
        "frames_n_docs": len(docs),
        "frames_docs": docs,
        "frames_oracle": bool(oracle),
        "frames_index": idx,
    }


def sample_tasks(tasks: list[dict], limit: int | None, seed: int) -> list[dict]:
    """A seeded sample of the FULL set — never a head slice.

    The split is ordered, so `tasks[:N]` is a biased subset wearing the name
    of a random one; any difficulty or topic gradient in the file silently
    becomes the result. Output stays in dataset order so two runs at the same
    seed are diffable line-by-line.
    """
    if limit is None or limit >= len(tasks):
        return tasks
    rnd = random.Random(seed)
    return sorted(rnd.sample(tasks, limit), key=lambda t: t["frames_index"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="tasks JSONL to write")
    ap.add_argument("--limit", type=int, default=None,
                    help="sample N tasks (seeded) instead of all 824")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--oracle", action="store_true",
                    help="append the gold Wikipedia links to each question — "
                         "removes the retrieval confound under Tor-only egress")
    args = ap.parse_args()

    rows = fetch_rows(None)
    tasks = [t for i, r in enumerate(rows)
             if (t := to_task(r, i, oracle=args.oracle))]
    if not tasks:
        raise SystemExit("no usable tasks parsed — refusing to write")

    tasks = sample_tasks(tasks, args.limit, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(
        "|".join(sorted(t["task_id"] for t in tasks)).encode()).hexdigest()[:16]
    prov = {
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_available": len(rows), "n_written": len(tasks),
        "oracle": bool(args.oracle), "seed": args.seed,
        "limit": args.limit, "tasks_sha256": digest,
        "auth": "anonymous — no token, no account",
    }
    Path(str(out) + ".provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"wrote {len(tasks)} tasks -> {out}")
    print(f"  tasks_sha256 {digest}   oracle={bool(args.oracle)}")
    print(f"  provenance   {out}.provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
