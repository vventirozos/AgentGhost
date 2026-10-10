"""§4NE (2026-10-10, operator: "finish §4ND then build it"): a regression
suite built from the owner's CONFIRMED failures.

Why: background self-improvement without independent ground truth made the
agent worse or changed nothing (§4NA–§4ND), and the 2026 literature agrees —
self-graded memory inflates its own errors; the gains reported all came from
executable checks or labels. This agent's one ground truth is the owner.

How:
  * a failure carrying the OWNER's evidence (a 👎, or the reaction judge's
    reading of the owner's next message, with that message) is drafted into
    a test PROPOSAL: the request + deterministic checks on the reply text and
    the tools the turn used — shown once ("Test proposal N …"), expiring
    after `PROPOSAL_TTL_S`; nothing enters the suite without "keep test N";
  * kept tests are replayed as labelled probes (`probe-rt-…`, never teach)
    while the owner is idle — after every code change and daily — and a
    failure is reported in the activity digest;
  * the failure-replay loop runs the kept suite with a candidate rule
    injected, and its proposal says how the owner's tests fared.

No model judges pass/fail: every check is deterministic. The case store is
written only by the agent process, under one lock, read-modify-write.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("GhostAgent")

STORE = "regression/cases.json"
RT_PREFIX = "probe-rt-"
PROPOSAL_TTL_S = 7 * 86400
MAX_OPEN_PROPOSALS = 3
RUN_EVERY_S = 24 * 3600
MAX_CASES_PER_RUN = 2          # full agent turns inside one capped idle job (§4NE r1: was 5)
DRAFT_BACKOFF_S = 3600         # r4: a busy model is retried later, doubling, never a verdict
DRAFT_BACKOFF_MAX_S = 24 * 3600
CANDIDATE_DAYS = 14.0
CHECK_KINDS = ("contains_any", "contains_all", "not_contains", "tool_used",
               "tool_not_used", "opened_domain")
_MAX_VALUE = 80
_MAX_VALUES = 5
_MAX_CHECKS = 4

_LOCK = threading.Lock()


# ── the store ─────────────────────────────────────────────────────────

def _path(home) -> Optional[Path]:
    return None if home is None else Path(home) / STORE


def load(home) -> dict:
    p = _path(home)
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p is not None and p.exists() else {}
    except Exception:  # noqa: BLE001
        d = {}
    if not isinstance(d, dict):
        d = {}
    d.setdefault("cases", [])
    d.setdefault("meta", {})
    return d


def update(home, fn: Callable[[dict], object]):
    """Load, apply `fn(store)` and save — one lock, so a command and the idle
    runner never overwrite each other. Returns `fn`'s result."""
    p = _path(home)
    if p is None:
        return None
    with _LOCK:
        store = load(home)
        out = fn(store)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
        return out


def _by_n(store: dict, n: int) -> Optional[dict]:
    return next((c for c in store["cases"] if int(c.get("n") or 0) == n), None)


def _next_n(store: dict) -> int:
    return 1 + max([int(c.get("n") or 0) for c in store["cases"]] + [0])


# ── checks: deterministic, on the reply and the turn's tools ──────────

def clean_checks(raw) -> list:
    """Only known kinds, short values, a bounded number — or []."""
    out = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict) or c.get("kind") not in CHECK_KINDS:
            continue
        vals = c.get("values")
        vals = [str(v).strip()[:_MAX_VALUE] for v in (vals if isinstance(vals, list) else [vals])
                if str(v or "").strip()][:_MAX_VALUES]
        if c["kind"] == "opened_domain":
            vals = [h for h in (_host(v) for v in vals) if h]
        if vals:
            out.append({"kind": c["kind"], "values": vals})
    return out[:_MAX_CHECKS]


def _host(v: str) -> str:
    """'https://www.PostgreSQL.org/docs' → 'postgresql.org' (§4NE r1); ""
    for a value no page can match or every page would (r2: no dot, spaces,
    a bare TLD)."""
    v = str(v or "").strip().lower()
    if not v or any(ch.isspace() for ch in v):
        return ""
    try:
        h = (urlparse(v if "://" in v else "//" + v).hostname or "").rstrip(".")
    except ValueError:
        return ""
    h = h[4:] if h.startswith("www.") else h
    return h if "." in h and all(h.split(".")) else ""


def _domains(tools: list) -> List[str]:
    hosts = []
    for t in tools or []:
        args = t.get("args") if isinstance(t, dict) else None
        text = json.dumps(args) if not isinstance(args, str) else args
        for u in re.findall(r"https?://[^\s\"'\\]+", text or ""):
            h = (urlparse(u).hostname or "").lower()
            if h:
                hosts.append(h)
    return hosts


def evaluate(checks: list, reply: str, tools: list) -> List[str]:
    """The checks that FAILED (empty = pass). `tools`: [{"tool", "args"}]."""
    text = str(reply or "").casefold()
    names = {str(t.get("tool") or "").lower() for t in tools or [] if isinstance(t, dict)}
    hosts = _domains(tools)
    failed = []
    for c in checks or []:
        k, vals = c.get("kind"), [str(v) for v in c.get("values") or []]
        lv = [v.casefold() for v in vals]
        ok = True
        if k == "contains_any":
            ok = any(v in text for v in lv)
        elif k == "contains_all":
            ok = all(v in text for v in lv)
        elif k == "not_contains":
            ok = not any(v in text for v in lv)
        elif k == "tool_used":
            ok = bool(names & {v.lower() for v in vals})
        elif k == "tool_not_used":
            ok = not (names & {v.lower() for v in vals})
        elif k == "opened_domain":
            ok = any(h == d or h.endswith("." + d) for h in hosts for d in lv)
        if not ok:
            failed.append(f"{k}: {', '.join(vals)}")
    return failed


def describe(checks: list) -> str:
    words = {"contains_any": "mentions one of", "contains_all": "mentions all of",
             "not_contains": "does not say", "tool_used": "uses", "tool_not_used": "does not use",
             "opened_domain": "opens"}
    return "; ".join(f"{words.get(c['kind'], c['kind'])} {' / '.join(c['values'])}" for c in checks or [])


# ── candidates: the owner's evidence only ─────────────────────────────

def candidates(collector, home_system, *, days: float = CANDIDATE_DAYS) -> List[dict]:
    """Owner failures with the OWNER's evidence: a 👎 (human label) or the
    reaction judge's verdict with the owner's next message. Newest first.
    Replayable (read-only) requests only — a probe cannot write."""
    from .owner_seeds import load_reactions, reaction_pairs
    from .failure_replay import replayable
    from ..memory.skills import iter_teachable
    from ..distill.collector import HUMAN_SOURCE_PREFIX
    try:
        trajs = list(iter_teachable(collector.iter_trajectories(since_days=days), consumer="frontier"))
    except Exception:  # noqa: BLE001
        return []
    reactions = load_reactions(home_system)
    nxt = {str(getattr(t, "id", "")): u for t, u in reaction_pairs(trajs)}
    out = []
    for t in reversed(trajs):
        tid = str(getattr(t, "id", "") or "")
        ex = getattr(t, "extra", None) or {}
        evidence = ""
        src = str(ex.get("outcome_source") or "")
        if src.startswith(HUMAN_SOURCE_PREFIX) and str(getattr(t, "outcome", "")).lower() == "failed":
            evidence = "👎" + (f" — {str(getattr(t, 'failure_reason', '') or '')[:200]}"
                               if getattr(t, "failure_reason", "") else "")
        elif reactions.get(tid) is True and nxt.get(tid) is not None:
            evidence = f"your next message: “{str(nxt[tid].user_request or '')[:300]}”"
        if not evidence or not tid:
            continue
        ok, _why = replayable(t)
        if not ok:
            continue
        out.append({"source_id": tid, "request": str(t.user_request or "")[:1500],
                    "reply": str(t.final_response or "")[:1500], "evidence": evidence})
    return out


# ── drafting ──────────────────────────────────────────────────────────

_DRAFT_SYSTEM = (
    "The owner of an AI assistant said an answer was wrong. Write what a CORRECT answer to the same "
    "request must satisfy, as at most 4 deterministic checks a program can run on the reply text and "
    "the tools the assistant used. Kinds: contains_any, contains_all, not_contains (case-insensitive "
    "substrings of the reply), tool_used, tool_not_used (tool names: web_search, browser, introspect, "
    "system_utility, file_system, knowledge_base, deep_research, recall …), opened_domain (a site the "
    "assistant must open, e.g. postgresql.org). Use only what the owner's evidence supports; prefer "
    "short exact values. Reply with JSON only: {\"expectation\": \"one sentence in plain words\", "
    "\"checks\": [{\"kind\": \"…\", \"values\": [\"…\"]}]}")


def draft_prompt(cand: dict, owner_words: str = "") -> list:
    parts = [f"REQUEST:\n{cand.get('request', '')[:1500]}",
             f"THE ASSISTANT'S ANSWER (judged wrong):\n{cand.get('reply', '')[:1500]}",
             f"THE OWNER'S EVIDENCE:\n{cand.get('evidence', '')[:600]}"]
    if owner_words:
        parts.append(f"THE OWNER'S OWN EXPECTATION (follow it exactly):\n{owner_words[:400]}")
    return [{"role": "system", "content": _DRAFT_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_draft(text: str) -> Optional[dict]:
    t = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    m = re.search(r"\{.*\}", t, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        return None
    checks = clean_checks(d.get("checks"))
    exp = str(d.get("expectation") or "").strip()[:200]
    if not checks or not exp:
        return None
    return {"expectation": exp, "checks": checks}


def proposal_text(case: dict) -> str:
    n = case.get("n")
    return (f"Test proposal {n} — say “keep test {n}”, “edit test {n}: …”, or ignore: for "
            f"“{case.get('request', '')[:60]}”, the answer {case.get('expectation', '')} "
            f"(checks: {describe(case.get('checks'))}).")


# ── running the suite ─────────────────────────────────────────────────

def _source_fingerprint() -> str:
    """Path, size and mtime of every source file, hashed (r2: count plus
    newest mtime missed a deploy that kept older mtimes)."""
    import hashlib
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        try:
            st = f.stat()
        except OSError:
            continue
        h.update(f"{f.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


# §4NE r1: the code THIS process runs — taken when the agent imports the
# module at boot, so an edit on disk before a restart does not count as run
_PROCESS_FP = _source_fingerprint()


def code_fingerprint() -> str:
    """Changes with each deploy (process start on changed code)."""
    return _PROCESS_FP


class RunInterrupted(Exception):
    """The test turn was cancelled (the owner arrived) — not a result."""


# r3: the agent's note starts a line in the reply's last paragraph — a model
# QUOTING it earlier in a reply is not a cancel
_CANCELLED_RE = re.compile(r"(?m)^_\(Turn cancelled[:.]")
MAX_STARTS = 2      # r2: a test cut by the idle cap this often is booked "could not run"


async def run_case(agent, case: dict, run_id: str, rule: str = "") -> dict:
    """One labelled-probe replay of a kept case; its checks evaluated."""
    import asyncio
    from ..utils.logging import probe_rule_context
    rid = f"{RT_PREFIX}{case['n']}-{run_id}"
    tok = probe_rule_context.set(str(rule or "")[:800])
    try:
        body = {"model": getattr(getattr(agent.context, "args", None), "model", "default"),
                "messages": [{"role": "user", "content": case["request"]}], "stream": False}
        out = await agent.handle_chat(body, background_tasks=None, request_id=rid)
        reply = str(getattr(out[0] if isinstance(out, tuple) else out, "content",
                            out[0] if isinstance(out, tuple) else out) or "")
    finally:
        probe_rule_context.reset(tok)
    if _CANCELLED_RE.search(reply.rstrip()[-400:]):     # r2: also after partial output (agent.py appends the note)
        raise RunInterrupted("the test run was cancelled")
    tools = await asyncio.to_thread(_tools_for, getattr(agent.context, "trajectory_collector", None), rid)
    failed = evaluate(case.get("checks"), reply, tools)
    return {"n": case["n"], "req_id": rid, "passed": not failed, "failed": failed,
            "reply": reply[:600], "at": time.time()}


def _tools_for(collector, rid: str) -> list:
    if collector is None:
        return []
    try:
        # its own run is a labelled probe (probe-rt-…): looked up by id, never taught
        for t in collector.iter_trajectories(since_days=1, include_probes=True):
            if str((getattr(t, "extra", None) or {}).get("req_id") or "") == rid:
                return [{"tool": tc.name, "args": tc.arguments or {}} for tc in (t.tool_calls or [])]
    except Exception:  # noqa: BLE001
        pass
    return []


def kept(store: dict) -> List[dict]:
    return [c for c in store["cases"] if c.get("state") == "kept" and c.get("checks")]


def due_cases(store: dict, now: float) -> List[dict]:
    """Kept cases not yet run on this code, or not run for a day, or all of
    them when the owner asked ("run regression tests") — most overdue first."""
    fp = code_fingerprint()
    req = float(store["meta"].get("run_requested") or 0)

    def _last(c):
        return (c.get("history") or [{}])[-1]
    due = [c for c in kept(store)
           if c.get("last_fp") != fp or now - float(_last(c).get("at") or 0) >= RUN_EVERY_S
           or float(_last(c).get("at") or 0) < req]
    return sorted(due, key=lambda c: float(_last(c).get("at") or 0))


async def run_suite(agent, home, *, rule: str = "", limit: int = MAX_CASES_PER_RUN,
                    record: bool = True) -> dict:
    """Replay up to `limit` cases: the DUE ones (`record=True`, the history is
    written), or — for a rule trial (the replay loop's gate, `record=False`)
    — the most recently kept, results only."""
    import asyncio
    store = await asyncio.to_thread(load, home)
    cases = (due_cases(store, time.time()) if record
             else sorted(kept(store), key=lambda c: -float(c.get("kept_at") or 0)))[:limit]
    run_id = f"{int(time.time()) % 10**8:x}"
    results, errors, interrupted = [], [], False
    fp = code_fingerprint()
    for c in cases:
        starts = 0
        if record:
            # r2 (fresh reader, MAJOR): a test the idle cap cuts every time
            # was never recorded, stayed first in line and starved the rest
            def _start(store, n=c["n"]):
                cc = _by_n(store, n)
                if cc is None:
                    return 0
                cc["starts"] = int(cc.get("starts") or 0) + 1
                store["meta"]["running"] = n
                return cc["starts"]
            starts = await asyncio.to_thread(update, home, _start)
        try:
            if record and starts > MAX_STARTS:
                raise RuntimeError("it did not finish within an idle slot's time limit")
            r = await run_case(agent, c, run_id, rule)
        except RunInterrupted:
            interrupted = True
            if record:          # the owner's arrival is not the test's fault
                await asyncio.to_thread(owner_stopped, home)
            break                   # the owner is here: what finished is kept
        except Exception as e:  # noqa: BLE001 — one broken run is not the suite
            r = {"n": c["n"], "req_id": "", "passed": None, "error": f"{type(e).__name__}: {e}"[:200],
                 "failed": [], "at": time.time()}
            errors.append(r)
        else:
            results.append(r)
        if record:              # §4NE r1: saved as each finishes — a cut keeps the rest

            def _apply(store, r=r):
                cc = _by_n(store, r["n"])
                if cc is not None:
                    cc.setdefault("history", []).append(
                        {k: r.get(k) for k in ("at", "passed", "failed", "req_id", "error") if k in r})
                    cc["history"] = cc["history"][-20:]
                    cc["last_fp"] = fp
                    cc.pop("starts", None)
                store["meta"].pop("running", None)
                if not due_cases(store, time.time()):
                    store["meta"].pop("run_requested", None)
            await asyncio.to_thread(update, home, _apply)
    return {"run": len(results), "passed": sum(1 for r in results if r["passed"]),
            "failed": [r for r in results if not r["passed"]], "errors": errors,
            "interrupted": interrupted}


def owner_stopped(home) -> None:
    """The owner's arrival stopped the test in flight: that start does not
    count toward MAX_STARTS (only the idle cap's cuts do). Called here on a
    cancelled turn and by the agent when the idle job was cancelled for the
    owner."""
    def _f(store):
        n = store["meta"].pop("running", None)
        c = _by_n(store, int(n)) if n is not None else None
        if c is not None and c.get("starts"):
            c["starts"] = int(c["starts"]) - 1
    if _path(home) is not None:
        update(home, _f)


def run_due(store: dict, now: float) -> bool:
    return bool(due_cases(store, now))


def suite_line(res: dict) -> str:
    bad, err = res.get("failed") or [], res.get("errors") or []
    if not res.get("run") and not err:
        return ""
    return ((f"Regression suite: {res['passed']}/{res['run']} of your tests pass" if res.get("run")
             else "Regression suite: no test finished")
            + (" — failing: " + "; ".join(f"test {r['n']} ({r['failed'][0]})" for r in bad[:3]) if bad else "")
            + (" — could not run: " + ", ".join(f"test {r['n']}" for r in err[:3]) if err else ""))


# ── the idle step ─────────────────────────────────────────────────────

async def advance(agent, ctx, home, *, now: Optional[float] = None) -> str:
    """One unit of work per idle slot: expire stale proposals, redraft an
    owner-edited test, run the suite when due, or draft one new proposal.
    Returns a one-line outcome ("" when there was nothing to do)."""
    import asyncio
    now = now or time.time()
    if _path(home) is None:
        return ""                   # no data home → no store to work on
    # r3: a run pointer left by an idle-cap cut is history — only a run this
    # step starts can be forgiven by `owner_stopped`
    await asyncio.to_thread(update, home, lambda s: s["meta"].pop("running", None))

    def _expire(store):
        for c in store["cases"]:
            if c.get("state") == "proposed" and now - float(c.get("proposed_at") or now) > PROPOSAL_TTL_S:
                c["state"] = "expired"
            # r4: a redraft of a KEPT test the owner never adopted lapses; the
            # test itself stays kept on its old checks
            if (c.get("pending") or {}).get("at") and now - float(c["pending"]["at"]) > PROPOSAL_TTL_S:
                c.pop("pending", None)
        wait = store["meta"].get("draft_wait") or {}
        for k in [k for k, (t, _n) in wait.items() if now - float(t) > CANDIDATE_DAYS * 86400]:
            wait.pop(k, None)
    store = await asyncio.to_thread(update, home, lambda s: (_expire(s), s)[1])

    edited = next((c for c in store["cases"]
                   if (c.get("state") == "edited" or c.get("redraft"))
                   and float(c.get("draft_wait") or 0) <= now), None)
    if edited is not None:
        words = edited.get("owner_expectation", "")
        d, transient = await _draft(ctx, edited, owner_words=words)

        def _redraft(s):
            c = _by_n(s, edited["n"])
            # r1: a newer "edit test N" while this one drafted wins
            if (c is None or not (c.get("state") == "edited" or c.get("redraft"))
                    or c.get("owner_expectation", "") != words):
                return None
            if d is None and transient:
                # r4: the model was busy — retry later (doubling), never a verdict
                c["draft_tries"] = int(c.get("draft_tries") or 0) + 1
                c["draft_wait"] = now + min(DRAFT_BACKOFF_MAX_S, DRAFT_BACKOFF_S * 2 ** (c["draft_tries"] - 1))
                return None
            c.pop("draft_tries", None)
            c.pop("draft_wait", None)
            if c.pop("redraft", None):
                # r4: a KEPT test keeps running on its old checks; the new
                # draft waits for "keep test N"
                if d is None:
                    return {"failed": True, "n": c["n"]}
                c["pending"] = dict(d, at=now)
                return dict(c, **d, pending_of_kept=True)
            if d is None:
                # r1/r4: never strand the owner's test — exactly what it was comes back
                prev = c.pop("prev_state", "proposed")
                checks = c.pop("prev_checks", []) or []
                if not checks:          # r3: nothing to restore — not a check-less proposal
                    prev = prev if prev in ("undraftable", "forgotten", "expired") else "undraftable"
                c.update(state=prev, checks=checks,
                         expectation=c.pop("prev_expectation", c.get("expectation", "")))
                return {"failed": True, "n": c["n"]}
            c.update(d, state="proposed", proposed_at=now)
            c.pop("prev_checks", None)
            c.pop("prev_expectation", None)
            c.pop("prev_state", None)
            return dict(c)
        c = await asyncio.to_thread(update, home, _redraft)
        if c and c.get("failed"):
            agent._record_autonomous_activity(
                "self_play", f"Test {c['n']}: your words could not be turned into checks a program can run — "
                f"the test is unchanged; try “edit test {c['n']}: …” with an exact word the answer must contain.",
                severity="notify")
            return f"test {c['n']} could not be drafted from your words"
        if c:
            agent._record_autonomous_activity(
                "self_play", proposal_text(c) + (" Until then the kept test runs on its current checks."
                                                 if c.get("pending_of_kept") else ""), severity="notify")
            return f"redrafted test {c['n']} from your words"
        return ""

    if run_due(store, now):
        res = await run_suite(agent, home)
        line = suite_line(res)
        if line:
            agent._record_autonomous_activity("self_play", line,
                                              severity="notify" if (res["failed"] or res.get("errors")) else "info")
        return line or ""

    if sum(1 for c in store["cases"] if c.get("state") == "proposed") >= MAX_OPEN_PROPOSALS:
        return ""
    # r2: an undraftable case was never shown to the owner, so it does not
    # block a NEW failure on the same request; r3: but the same failure with
    # the same evidence is not redrafted every slot (it starved older ones)
    shown = [c for c in store["cases"] if c.get("state") != "undraftable"]
    known = {c.get("source_id") for c in shown}
    asked = [c.get("request", "") for c in shown]
    tried = {(c.get("source_id"), c.get("evidence")) for c in store["cases"] if c.get("state") == "undraftable"}
    waiting = {k for k, (t, _n) in (store["meta"].get("draft_wait") or {}).items() if float(t) > now}
    from ..memory.lesson_scope import same_request
    cands = await asyncio.to_thread(candidates, getattr(ctx, "trajectory_collector", None), home)
    # r1: a re-asked request is one failure — one test
    cand = next((c for c in cands if c["source_id"] not in known
                 and (c["source_id"], c.get("evidence")) not in tried
                 and c["source_id"] not in waiting      # r4: a busy model's backoff
                 and not any(same_request(c["request"], a) for a in asked if a)), None)
    if cand is None:
        return ""
    d, transient = await _draft(ctx, cand)

    def _add(s):
        if any(c.get("source_id") == cand["source_id"] and c.get("state") != "undraftable"
               for c in s["cases"]):
            return None
        wait = s["meta"].setdefault("draft_wait", {})
        if d is None and transient:
            # r1/r4: a busy model is not a verdict on the evidence — retry
            # later, doubling the wait; other failures are drafted meanwhile
            n_tries = int((wait.get(cand["source_id"]) or (0, 0))[1]) + 1
            wait[cand["source_id"]] = (now + min(DRAFT_BACKOFF_MAX_S, DRAFT_BACKOFF_S * 2 ** (n_tries - 1)), n_tries)
            return None
        wait.pop(cand["source_id"], None)
        # a re-drafted failure replaces its unseen undraftable row (after the
        # transient return above, r3: so a busy model does not drop the row)
        s["cases"][:] = [c for c in s["cases"]
                         if not (c.get("source_id") == cand["source_id"] and c.get("state") == "undraftable")]
        c = dict(cand, n=_next_n(s), created=now)
        if d is None:
            c["state"] = "undraftable"
        else:
            c.update(d, state="proposed", proposed_at=now)
        s["cases"].append(c)
        return dict(c)
    c = await asyncio.to_thread(update, home, _add)
    if c and c.get("state") == "proposed":
        agent._record_autonomous_activity("self_play", proposal_text(c), severity="notify")
        return f"proposed test {c['n']}"
    return ""


async def _draft(ctx, cand: dict, owner_words: str = ""):
    """(draft or None, transient): transient = the model could not be asked
    (no client, timeout, busy) — not a judgement on the evidence."""
    llm = getattr(ctx, "llm_client", None)
    if llm is None:
        return None, True
    try:
        r = await llm.chat_completion({"model": getattr(getattr(ctx, "args", None), "model", "default"),
                                       "messages": draft_prompt(cand, owner_words), "temperature": 0.0,
                                       "max_tokens": 600, "stream": False,
                                       "chat_template_kwargs": {"enable_thinking": False}},
                                      is_background=True, timeout=180.0, task_label="regression draft")
    except Exception as e:  # noqa: BLE001
        logger.debug("regression draft failed: %s", e)
        return None, True
    return parse_draft(((r or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")), False


# ── the owner's commands ──────────────────────────────────────────────

_LEAD = r"^[\s\"'“”‘’«»]*(?:(?:ok|okay|yes|sure|please|pls)[\s,!.]+)*"
_END = r"[\s\"'“”‘’«».!?]*$"
_KEEP_RE = re.compile(_LEAD + r"keep\s+test\s*#?\s*(\d+)" + _END, re.I)
_FORGET_RE = re.compile(_LEAD + r"forget\s+test\s*#?\s*(\d+)" + _END, re.I)
_EDIT_RE = re.compile(_LEAD + r"edit\s+test\s*#?\s*(\d+)\s*:\s*(.+)$", re.I | re.S)
_SHOW_RE = re.compile(_LEAD + r"show\s+test\s*#?\s*(\d+)" + _END, re.I)
# r1 (fresh reader, MAJOR): "run tests" / "list the tests" are everyday
# coding requests — the suite-wide commands name the REGRESSION tests
_LIST_RE = re.compile(_LEAD + r"(?:show|list)\s+(?:me\s+)?(?:all\s+)?(?:of\s+)?(?:my\s+|the\s+)?regression\s+tests" + _END, re.I)
_RUN_RE = re.compile(_LEAD + r"run\s+(?:all\s+)?(?:the\s+|my\s+)?regression\s+tests" + _END, re.I)


def _pending_line(c: dict) -> str:
    """r5: a redraft of a kept test waiting for "keep test N" is shown."""
    p = c.get("pending") or {}
    if not p:
        return ""
    from ..utils.prompt_safety import defuse_text as _dfz
    return (f"\nPending redraft (say “keep test {c.get('n')}” to use it): the answer "
            f"{_dfz(p.get('expectation', ''))[:200]} — checks: {_dfz(describe(p.get('checks')))}")


def _last_run(last: dict, short: bool = False) -> str:
    """r2: an infra error (passed None) is "could not run", never a FAIL."""
    if last.get("passed") is None:
        why = "" if short else f" ({str(last.get('error') or 'error')[:120]})"
        return (" — last run: could not run" if short else "could not run") + why
    if last.get("passed"):
        return " — last run: pass" if short else "pass"
    return " — last run: FAIL" if short else "FAIL — " + "; ".join(last.get("failed") or [])


def is_test_command(text: str) -> bool:
    t = str(text or "")
    return any(r.match(t) for r in (_KEEP_RE, _FORGET_RE, _EDIT_RE, _SHOW_RE, _LIST_RE, _RUN_RE))


def owner_test_command(text: str, home):
    """The OWNER's test commands. Returns a note (str subclass carrying the
    outcome `banner` the agent prepends — see failure_replay.RuleNote), or "".
    Call ONLY for an owner turn."""
    from .failure_replay import _rn, _SAID
    from ..utils.prompt_safety import defuse_text as _dfz
    t = str(text or "").strip()

    def said(b):
        return _rn(_SAID.format(b=b), b)

    if _LIST_RE.match(t):
        store = load(home)
        rows = {"kept": [], "proposed": []}
        for c in store["cases"]:
            if c.get("state") in rows:
                last = (c.get("history") or [{}])[-1]
                res = "" if not last else _last_run(last, short=True)
                rows[c["state"]].append(f"- **Test {c['n']}** — “{_dfz(c.get('request', ''))[:60]}”: "
                                        f"{_dfz(c.get('expectation', ''))[:120]}{res}"
                                        + (" — a redraft waits for “keep test N”" if c.get("pending") else ""))
        parts = []
        if rows["kept"]:
            parts.append("**Kept** (checked after every change; “forget test N” to drop one):\n" + "\n".join(rows["kept"][-30:]))
        if rows["proposed"]:
            parts.append("**Proposed** (“keep test N” to add one):\n" + "\n".join(rows["proposed"][-10:]))
        return said("\n\n".join(parts) if parts else "There are no regression tests yet — none has been proposed or kept.")
    if _RUN_RE.match(t):
        update(home, lambda s: s["meta"].__setitem__("run_requested", time.time()))
        return said("Your regression tests will run at the next idle moment; the result comes in the activity digest.")
    for rx, verb in ((_KEEP_RE, "keep"), (_FORGET_RE, "forget"), (_SHOW_RE, "show"), (_EDIT_RE, "edit")):
        m = rx.match(t)
        if m:
            n = int(m.group(1))
            break
    else:
        return ""

    if _by_n(load(home), n) is None and not any(c.get("state") != "undraftable" for c in load(home)["cases"]):
        return ""       # r1: no suite at all — "show test 3" is someone's own test

    def _act(store):
        c = _by_n(store, n)
        if c is None:
            return f"There is no regression test {n} (“show regression tests” lists them)."
        if verb == "show":
            last = (c.get("history") or [{}])[-1]
            return (f"**Test {n}** ({c.get('state')}): for “{_dfz(c.get('request', ''))[:200]}”, the answer "
                    f"{_dfz(c.get('expectation', ''))[:200]}\nChecks: {_dfz(describe(c.get('checks')))}\n"
                    f"Evidence: {_dfz(c.get('evidence', ''))[:200]}"
                    + (f"\nLast run: {_last_run(last)}" if last else "\nNot run yet.")
                    + _pending_line(c))
        if verb == "keep":
            if c.get("redraft"):       # r5: the owner's new words are still being drafted
                return (f"Test {n} is still being redrafted from your words — it keeps running on its "
                        f"current checks, and you will be asked to keep the new ones.")
            if c.get("state") == "kept" and not c.get("pending"):
                return f"Test {n} is already kept."      # r5: no re-stamp, no suite-wide re-run
            pend = c.pop("pending", None)
            if pend:                # r4: adopt the redraft of a kept test
                c.update(expectation=pend["expectation"], checks=pend["checks"])
            if not c.get("checks"):
                return f"Test {n} has no checks yet — it cannot be kept."
            c["state"], c["kept_at"] = "kept", time.time()
            store["meta"]["run_requested"] = time.time()
            return f"✓ Test {n} kept — it will be checked after every change: {_dfz(c.get('expectation', ''))[:160]}"
        if verb == "forget":
            c["state"] = "forgotten"
            for k in ("pending", "redraft", "draft_wait", "draft_tries"):
                c.pop(k, None)
            return f"✓ Test {n} forgotten — it will no longer be run."
        # edit: the owner's words; redrafted into checks at the next idle moment
        c["owner_expectation"] = m.group(2).strip()[:400]
        for k in ("draft_tries", "draft_wait"):
            c.pop(k, None)
        if c.get("state") == "kept":
            # r4: a kept test stays in the suite on its current checks while
            # the owner's words are drafted; "keep test N" adopts the redraft
            c["redraft"] = True
            c.pop("pending", None)
            return (f"Test {n} will be redrafted from your words and shown again — it keeps running on "
                    f"its current checks until you say “keep test {n}”.")
        if c.get("state") != "edited":      # r1: keep what to restore if the words cannot be drafted
            c["prev_checks"], c["prev_expectation"] = c.get("checks") or [], c.get("expectation", "")
            c["prev_state"] = c.get("state")
        c["state"], c["checks"] = "edited", []
        return f"Test {n} will be redrafted from your words and shown again for “keep test {n}”."
    return said(update(home, _act))
