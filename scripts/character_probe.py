#!/usr/bin/env python3
"""The character's acceptance run (§4UG): is a character sentence AUDIBLE in replies, and does it leave
what the agent DOES alone?

Sends each prompt to the live agent as a labelled probe, with the character and without it (the probe arm
`no_self_image`), several samples each, in a seeded random order, waiting before every call until no owner
request is running. Then reads the probes' own turn records and reports:

  * length, tool calls and replies that NAME a trait, per arm — the character must not lower tool use
    (§4UF: "short and dry" cut it by a third) and must never be announced;
  * a blind sheet (pairs in random order, no arm names) for a reader to pick, per pair, the reply written
    with the character in view — and the score of such picks against the key.

    python scripts/character_probe.py smoke  OUT                 one prompt, both arms; proves the arms differ
    python scripts/character_probe.py run    OUT [--samples 2]   the whole set (resumable)
    python scripts/character_probe.py report OUT                 measures, and writes OUT/blind.json + key.json
    python scripts/character_probe.py score  OUT PICKS.json      PICKS = {"<pair id>": "A"|"B" or {"pick": ...}}

  --dials=W,H,D   test THESE dial positions (e.g. --dials=-2,2,1) instead of the stored character: the probe
                  brings their sentence and the character's rule (X-Ghost-Probe-Self-Image / -Rule).
OUT is a directory outside the repository: it holds reply text. The prompts are generic on purpose.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = os.environ.get("GHOST_AGENT_URL", "http://127.0.0.1:8000")
KEY_FILE = Path(os.path.expanduser(os.environ.get("GHOST_API_KEY_FILE", "~/Data/AI/.ghost_api_key")))
ARMS = ("with", "without")
SEED = 20261111
CALL_TIMEOUT_S = 900
PROMPTS = [
    "all good here, do you have any topic you'd like to discuss?",
    "good job today!",
    "it's still early, I have some time before bed.",
    "good thing I don't have plans to go out today",
    "I can't decide between learning Rust or Go next, what do you think?",
    "give me three ideas for a weekend project",
    "do you, as an AI, believe in death?",
    "what's the difference between a mutex and a semaphore?",
    "can I play a PlayStation exclusive on a Mac?",
    "do the batteries of electric cars have a lifespan?",
    "how do I detach a partition in PostgreSQL?",
    "I need to wash my car. The car wash is 50 meters away. Should I drive or walk?",
    "explain to me what a tesseract is",
    "I plan to invest a fixed amount every month, how should I think about spreading it?",
    "how's the weather today?",
    "is my PostgreSQL up?",
]
#: a reply that announces a trait instead of having it
NAMES_TRAIT = re.compile(r"\b(?:i(?:'ll| will)? be (?:blunt|straight|dry|warm|playful)|to be blunt|"
                         r"(?:wry|dry|light) (?:remark|note|joke|aside)\b.{0,12}(?:allow|permit|only|one)|"
                         r"my (?:character|dials?)\b|as a (?:dry|warm|playful|blunt) )", re.I)


def _key() -> str:
    return KEY_FILE.read_text().strip()


def _health() -> dict:
    req = urllib.request.Request(BASE + "/api/health", headers={"X-Ghost-Key": _key()})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def wait_idle() -> bool:
    """Block while an owner request runs. True when one was seen."""
    seen = False
    while int(_health().get("foreground_requests") or 0) > 0:
        seen = True
        time.sleep(5)
    return seen


def brought(dials):
    """(sentence, rule, the stamp the turn record must carry) for --dials, or None for the stored character."""
    if dials is None:
        return None
    from ghost_agent.selfhood import character as C, loop as M
    d = dict(zip(C.DIALS, dials))
    s = C.sentence(d)
    if not s:
        raise SystemExit("every dial at 0 has no sentence: nothing to test")
    return s, C.RULE, M.probe_stamp(M.probe_section(s, C.RULE))


def chat(prompt: str, rid: str, arm: str, bring) -> str:
    h = {"Content-Type": "application/json", "X-Ghost-Key": _key(), "User-Agent": "ghost-functional-test",
         "X-Ghost-Origin": "probe", "X-Request-ID": rid}
    if arm == "without":
        h["X-Ghost-Arm"] = "no_self_image"
    elif bring is not None:
        h["X-Ghost-Probe-Self-Image"] = quote(bring[0])
        h["X-Ghost-Probe-Self-Image-Rule"] = quote(bring[1])
    body = {"messages": [{"role": "user", "content": prompt}], "stream": False}
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
        d = json.loads(r.read())
    if isinstance(d.get("message"), dict):
        return str(d["message"].get("content") or "")
    try:
        return str(d["choices"][0]["message"]["content"] or "")
    except Exception:  # noqa: BLE001
        return ""


def _rows(out: Path):
    p = out / "replies.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _append(out: Path, row: dict) -> None:
    fd = os.open(out / "replies.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_job(out: Path, tag: str, pi: int, arm: str, s: int, bring) -> dict:
    owner = wait_idle()
    rid = f"probe-chr{tag}-{pi:02d}{arm[:4]}{s}-{int(time.time()) % 100000}"
    t0 = time.time()
    try:
        reply, err = chat(PROMPTS[pi], rid, arm, bring), ""
    except Exception as e:  # noqa: BLE001
        reply, err = "", type(e).__name__
    row = {"job": f"{tag}-p{pi:02d}-{arm}-s{s}", "rid": rid, "prompt_i": pi, "arm": arm, "sample": s,
           "reply": reply, "error": err, "seconds": round(time.time() - t0, 1), "owner_active": owner,
           "at": datetime.datetime.now().isoformat(timespec="seconds")}
    _append(out, row)
    return row


def records(prefix: str) -> dict:
    """{rid: {"self_image", "mood", "tools"}} from the probes' own turn records (the authority)."""
    from ghost_agent.distill.collector import TrajectoryCollector
    home = Path(os.path.expanduser(os.environ.get("GHOST_HOME", "~/Data/AI/Data")))
    col = TrajectoryCollector(root=home / "system" / "trajectories", enabled=False)
    got = {}
    for back in range(3):
        day = (datetime.date.today() - datetime.timedelta(days=back)).isoformat()
        # the run's own probes, by request id: a probe record is the only place that says what the reply
        # heard and which tools it called (nothing here teaches or reports on user traffic)
        for tr in col.iter_trajectories(day=day, include_probes=True):
            ex = getattr(tr, "extra", None) or {}
            rid = str(ex.get("req_id") or "")
            if rid.startswith(prefix):
                got[rid] = {"self_image": str(ex.get("self_image") or ""), "mood": ex.get("mood_tone"),
                            "tools": [(c.get("name") if isinstance(c, dict) else getattr(c, "name", "")) or ""
                                      for c in (getattr(tr, "tool_calls", None) or [])]}
    return got


def expected_stamp(arm: str, bring) -> str:
    if arm == "without":
        return ""
    if bring is not None:
        return bring[2]
    from ghost_agent.selfhood import character as C
    home = Path(os.path.expanduser(os.environ.get("GHOST_HOME", "~/Data/AI/Data")))
    return C.section(home)[1]


def binom(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    pmf = lambda i: math.comb(n, i) / 2 ** n  # noqa: E731
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= pmf(k) + 1e-12))


def clean_pairs(out: Path, bring):
    recs = records("probe-chrrun-")
    aside = collections.Counter()
    by = collections.defaultdict(dict)
    for r in _rows(out):
        if not r["job"].startswith("run-"):
            continue
        rec = recs.get(r["rid"])
        why = ("failed or empty" if r["error"] or not r["reply"] else "no turn record" if rec is None
               else "the record disagrees with the arm" if rec["self_image"] != expected_stamp(r["arm"], bring)
               else "heard a mood line" if rec["mood"] else
               "called introspect (it shows the character)" if "introspect" in rec["tools"] else "")
        if why:
            aside[why] += 1
            continue
        by[(r["prompt_i"], r["sample"])][r["arm"]] = {**r, "tools": rec["tools"]}
    pairs = [(k, v["with"], v["without"]) for k, v in sorted(by.items()) if len(v) == 2]
    return pairs, aside


def cmd_smoke(out: Path, bring) -> int:
    ok = True
    for arm in ARMS:
        row = run_job(out, "smoke", 0, arm, 0, bring)
        time.sleep(3)
        rec = records("probe-chrsmoke-").get(row["rid"])
        si = None if rec is None else rec["self_image"]
        print(f"{arm:8s} {row['seconds']:6.1f}s err={row['error'] or '-'} chars={len(row['reply'])} heard={si!r}")
        ok &= bool(row["reply"]) and si == expected_stamp(arm, bring)
    print("DISCRIMINATING: " + ("yes" if ok else "NO — do not run (a record is missing or disagrees with its arm)"))
    return 0 if ok else 1


def cmd_run(out: Path, samples: int, bring) -> int:
    jobs = [(pi, arm, s) for pi in range(len(PROMPTS)) for arm in ARMS for s in range(samples)]
    random.Random(SEED).shuffle(jobs)
    done = {r["job"] for r in _rows(out)}
    t0 = time.time()
    for n, (pi, arm, s) in enumerate(jobs, 1):
        if f"run-p{pi:02d}-{arm}-s{s}" in done:
            continue
        row = run_job(out, "run", pi, arm, s, bring)
        (out / "progress.json").write_text(json.dumps({
            "done": n, "total": len(jobs), "last": row["job"], "last_seconds": row["seconds"],
            "elapsed_s": round(time.time() - t0), "updated": row["at"]}))
        print(f"[{n}/{len(jobs)}] {row['job']} {row['seconds']}s err={row['error'] or '-'}", flush=True)
    (out / "progress.json").write_text(json.dumps({"done": len(jobs), "total": len(jobs), "finished": True}))
    return 0


def cmd_report(out: Path, bring) -> int:
    pairs, aside = clean_pairs(out, bring)
    print(f"{len(pairs)} clean pairs; set aside: {dict(aside) or 'none'}")
    if not pairs:
        return 1
    w, wo = [p[1] for p in pairs], [p[2] for p in pairs]
    for name, fn in (("characters", lambda r: len(r["reply"])), ("tool calls", lambda r: len(r["tools"]))):
        a, b = [fn(r) for r in w], [fn(r) for r in wo]
        up, down = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
        print(f"{name:11s} with: median {statistics.median(a):7.1f} total {sum(a):6d} | without: median "
              f"{statistics.median(b):7.1f} total {sum(b):6d} | with>without in {up}, < in {down} "
              f"(sign test p={binom(up, up + down):.3f})")
    named = [sum(bool(NAMES_TRAIT.search(r["reply"])) for r in rs) for rs in (w, wo)]
    print(f"replies that name a trait: with {named[0]} of {len(w)}, without {named[1]} of {len(wo)}")
    tools_w, tools_wo = sum(len(r["tools"]) for r in w), sum(len(r["tools"]) for r in wo)
    gates = [("tool use not lowered (with ≥ 80 % of without)", tools_wo == 0 or tools_w >= 0.8 * tools_wo),
             ("no reply names a trait", named[0] == 0)]
    for label, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    rnd = random.Random(SEED + 1)
    blind, key = [], {}
    for (pi, s), a, b in pairs:
        pid = f"p{pi:02d}s{s}"
        first = rnd.random() < 0.5
        x, y = (a, b) if first else (b, a)
        blind.append({"id": pid, "prompt": PROMPTS[pi], "A": x["reply"], "B": y["reply"]})
        key[pid] = "A" if first else "B"
    rnd.shuffle(blind)
    if bring is not None:
        sentence = bring[0]
    else:
        from ghost_agent.selfhood import character as C
        home = Path(os.path.expanduser(os.environ.get("GHOST_HOME", "~/Data/AI/Data")))
        sentence = C.sentence(C.load(home)["dials"])
    for name, data in (("blind.json", {"sentence": sentence, "pairs": blind}), ("key.json", key)):
        fd = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"blind sheet: {out / 'blind.json'} (give a reader ONLY this file); key: {out / 'key.json'}")
    return 0 if all(ok for _, ok in gates) else 2


def cmd_score(out: Path, picks_path: Path) -> int:
    key = json.loads((out / "key.json").read_text())
    picks = json.loads(picks_path.read_text())
    n = k = 0
    for pid, truth in key.items():
        got = picks.get(pid)
        pick = got.get("pick") if isinstance(got, dict) else got
        if pick in ("A", "B"):
            n += 1
            k += pick == truth
    if not n:
        print("no picks")
        return 1
    ok = k / n >= 0.70
    print(f"the reader picked the reply written with the character in {k} of {n} pairs ({k / n:.0%}); "
          f"exact two-sided p vs chance = {binom(k, n):.4f}")
    print(f"  [{'PASS' if ok else 'FAIL'}] audible (a reader is right in at least 70 % of pairs)")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("smoke", "run", "report", "score"))
    ap.add_argument("out", type=Path)
    ap.add_argument("picks", nargs="?", type=Path)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--dials", default="")
    a = ap.parse_args()
    out = a.out.expanduser().resolve()
    if REPO in out.parents or out == REPO:
        raise SystemExit("OUT must be outside the repository: it holds reply text")
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    dials = None
    if a.dials:
        try:
            dials = [int(x) for x in a.dials.replace("−", "-").split(",")]
            assert len(dials) == 3 and all(-2 <= x <= 2 for x in dials)
        except Exception:  # noqa: BLE001
            raise SystemExit("--dials wants three whole numbers from -2 to 2: warmth,humour,directness")
    bring = brought(dials)
    if a.command == "smoke":
        return cmd_smoke(out, bring)
    if a.command == "run":
        return cmd_run(out, a.samples, bring)
    if a.command == "report":
        return cmd_report(out, bring)
    if not a.picks:
        raise SystemExit("score wants the picks file")
    return cmd_score(out, a.picks)


if __name__ == "__main__":
    sys.exit(main())
