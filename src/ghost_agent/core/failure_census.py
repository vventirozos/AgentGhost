"""Silent failure census (§4PL): which mechanical failure detectors fire on
the owner's real turns, and which of them the owner's own signals agree with.

Why. Two features in one day were justified by failure rates measured on
older history, and the second (§4PK, the unbacked-draft hold) met a model
that had already stopped making the mistake: it fired on 3.4 % of replays
where the history said ~14 %. A feature has to be chosen from the CURRENT
rate of its failure class, on real traffic — this module is that instrument.

It changes no reply and calls no model. LIVE rows (from §4PL on) carry what
only the turn itself can see — the quote checker's full result, and the
no-tool detector run over the whole context the model was given (the
composed prompt with its injected state and the conversation's tool
results; a stored trajectory keeps 8,000 chars of the system prompt and no
earlier turns, so the same detector over HISTORY is an approximation and is
reported as one). The rest is joined afterwards from stores that exist:

  * trajectories (+ full tool results)        — the turn itself
  * verifier/verdicts sidecar                  — the mechanical arms' refutes
  * trajectories/corrections.jsonl             — the owner's 👍/👎 and corrections
  * selfplay/reaction_verdicts.json            — the reaction judge on the owner's
                                                 next message (an LLM signal; tool turns only)

Owner turns only: no members, probes, sub-agents, jobs, scheduled or
simulated turns. Read-only except the quote ledger and the report it writes.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("GhostAgent")

QUOTE_REL = "system/census/turns.jsonl"          # live rows: kind "quote" | "turn"
#: rotate the live ledger past this size (keeps the newest half)
LEDGER_MAX_BYTES = 4 * 1024 * 1024
REPORT_DIR_REL = "system/census"
#: request-id prefixes that are not the owner talking (probes, replays, sub-agents, jobs, schedules, sims)
_NOT_OWNER_PREFIXES = ("probe", "sim-", "sub-", "job-", "sched", "bench", "replay")


def _home(home=None) -> Optional[Path]:
    root = home or (os.environ.get("GHOST_HOME") or "").strip()
    return Path(root) if root else None


# ── the one live write ──────────────────────────────────────────────────────

def _owner_request(req_id) -> bool:
    """Write only for the owner's own turns (review m6: the ledger held
    probes' and members' quoted text too)."""
    try:
        from ..utils.logging import requester_is_member
        rid = str(req_id or "")
        # (`probe-` is in _NOT_OWNER_PREFIXES; a separate probe test was dead code, battery 4pl M2)
        return bool(rid) and not requester_is_member() and not rid.startswith(_NOT_OWNER_PREFIXES)
    except Exception:  # noqa: BLE001
        return False


import threading as _threading
#: r3 m3: rotation (read → .tmp → replace) raced concurrent appends and lost rows
_LEDGER_LOCK = _threading.Lock()


def _append(row: dict, home=None) -> bool:
    base = _home(home)
    if base is None:
        return False
    with _LEDGER_LOCK:
        return _append_locked(row, base)


def _append_locked(row: dict, base: Path) -> bool:
    from ..utils.json_store import open_append
    p = base / QUOTE_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.exists() and p.stat().st_size > LEDGER_MAX_BYTES:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            tmp = p.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")
            os.replace(tmp, p)
    except OSError:
        pass
    with open_append(p) as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def _redact(t: str) -> str:
    try:
        from ..distill.redact import redact_text
        return redact_text(str(t or ""))[:200]
    except Exception:  # noqa: BLE001
        return ""


def record_quote(req_id: str, stats: dict, home=None, path: str = "finalize") -> bool:
    """Append this turn's quote-check result (both delivery paths: the
    finalize step and the streamed filter). Never raises; no home or not
    the owner's turn → no write."""
    try:
        if not isinstance(stats, dict) or not _owner_request(req_id):
            return False
        row = {"kind": "quote", "req_id": str(req_id), "path": path,
               "at": datetime.datetime.utcnow().isoformat() + "Z"}
        for k in ("quotes", "found", "figure_mismatch", "markers", "failed"):
            if isinstance(stats.get(k), (int, float)) and not isinstance(stats.get(k), bool):
                row[k] = int(stats[k])
        miss = stats.get("missing")
        if isinstance(miss, (list, tuple)):
            row["missing"] = [_redact(m) for m in miss[:20]]
        return _append(row, home)
    except Exception as e:  # noqa: BLE001 — the census never costs a turn
        logger.debug("failure census: quote row not recorded: %s", e)
        return False


def record_turn(req_id: str, reply: str, request: str, context_text: str, tools_ran: bool,
                tool_names: Iterable[str] = (), home=None, path: str = "finalize",
                tools: Iterable[dict] = (), delivered_reply: Optional[str] = None) -> bool:
    """Append the live no-tool detector's verdict for this turn, run over the
    FULL context (review M2), and (§4PM) each tool call's class read off its
    live outcome plus the turn's task kind and outcome. One row per finished
    turn; never raises."""
    try:
        if not _owner_request(req_id):
            return False
        f = unbacked_no_tool(reply, request, context_text, tools_ran, tool_names)
        row = {"kind": "turn", "req_id": str(req_id), "path": path, "tools_ran": bool(tools_ran),
               "at": datetime.datetime.utcnow().isoformat() + "Z",
               "unsupported_n": len(f["unsupported"]) if f else 0,
               "moving": bool(f and f["moving"]), "claims_n": len(f["claims"]) if f else 0}
        try:
            rows = [t for t in (tools or ()) if isinstance(t, dict)]
            if rows:
                row["classes"] = [tool_class(t) for t in rows]
                row["call_names"] = [_tname(t) for t in rows]
            # the task outcome reads what the OWNER got (a system-appended
            # image link included, r1 M2); the detectors read the model's reply
            to = task_outcome(rows, reply if delivered_reply is None else delivered_reply)
            row["task"] = {"kind": to["kind"], "outcome": to["outcome"]}
        except Exception as e:  # noqa: BLE001 — the detector row still lands
            logger.debug("failure census: task outcome not recorded: %s", e)
        return _append(row, home)
    except Exception as e:  # noqa: BLE001
        logger.debug("failure census: turn row not recorded: %s", e)
        return False


def record_fast(req_id: str, reply: str, home=None, kind: str = "fast") -> bool:
    """§4PM: a turn the trivial fast path answered (a greeting) writes no
    trajectory by design — this row is the only trace that it happened.
    §4PY: ``kind="command"`` — an owner command the system answered itself.
    Owner turns only; never raises."""
    try:
        if not _owner_request(req_id):
            return False
        return _append({"kind": "command" if kind == "command" else "fast", "req_id": str(req_id),
                        "chars": len(str(reply or "")),
                        "at": datetime.datetime.utcnow().isoformat() + "Z"}, home)
    except Exception as e:  # noqa: BLE001
        logger.debug("failure census: fast row not recorded: %s", e)
        return False


# ── detectors (pure; mechanical) ────────────────────────────────────────────

# The §4PK trigger v4 (pre-registered, sha256 94f9185e…3e9db), kept here as a
# DETECTOR only — nothing is held. Its tests hold it to the frozen copy.
_FENCE = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`\n]+`")
_SYS_PREFIX = re.compile(r"^\*\*Background activity while you were away:\*\*.*?(?:\n\n|\Z)", re.S)
_FIN = re.compile(r"\b(?:ETF|ETFs|fund|stock|stocks|shares|ticker|ISIN|UCITS|index|portfolio|broker|dividend)\b", re.I)
_SPEC = (
    re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w]|\.\d)|(?<![\w.:])\d+\.\d+(?![\w:]|\.\d)|(?<![\w.])\d+(?:\.\d+)?\s?%"),
    re.compile(r"[$€£]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:USD|EUR|GBP|dollars|euros)\b"),
    re.compile(r"(?<![\w.])v?\d+\.\d+\.\d+(?![\w]|\.\d)|\bversion\s+\d+(?:\.\d+)*"),
    re.compile(r"(?<![\w./-])[a-z]+_[a-z0-9_]+\(\)|(?<![\w./-])[a-z][a-z0-9]*_[a-z0-9_]+(?![\w./-])|(?<![\w.])[A-Za-z_]\w*\(\)"),
)
_TICKER = re.compile(r"(?<![\w-])[A-Z]{3,5}(?![\w-])")
_CLAIM = re.compile(
    r"\b(?:I(?:'ve| have)? just (?:checked|looked|ran|run|pulled|searched|verified|tested|generated|fetched)|"
    r"(?:no|nothing has) change[ds]? since (?:the )?last (?:check|time)|"
    r"I (?:checked|verified|tested|looked it up|searched)\b|the (?:data|results?) I (?:just )?(?:pulled|fetched|got)|"
    r"as of (?:today|right now|this morning)|"
    r"I (?:just )?(?:ran|executed|pulled|fetched|looked up|checked) (?:it|the|this|that|a|your)\b)", re.I)
_LIVE = re.compile(r"\b(?:weather|temperature|forecast|right now|at the moment|today's news|headlines|score (?:of|in) the)\b", re.I)
_LITERAL = re.compile(r"\b(?:reply|respond|answer|say|repeat|echo)\b[^.\n]{0,30}\b(?:exactly|verbatim|only with|nothing else|word for word)\b", re.I)
_CREATIVE = re.compile(r"\b(?:write|compose|draft|make up|invent|imagine)\b.{0,40}\b(?:poem|story|song|joke|haiku|limerick|fiction|lyrics|toast|speech)\b", re.I)
_SMALLTALK = re.compile(
    r"^\W*(?:hi|hello|hey|good (?:morning|evening|afternoon)|yo)\b|\bhow (?:are|r) (?:you|u)\b"
    r"|how(?:'s| is| are) (?:things|it going|you doing|life)|\bhow do you feel\b|\bfeeling today\b"
    r"|^\W*(?:thanks|thank you|very good|great|nice|cool|ok|okay)\b[^?]{0,40}$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def unbacked_no_tool(reply: str, request: str, context_text: str, tools_ran: bool,
                     tool_names: Iterable[str] = ()) -> Optional[dict]:
    """The §4PK v4 decision: None, or {"unsupported", "claims", "moving"}."""
    if tools_ran:
        return None
    req = request or ""
    if _CREATIVE.search(req) or _SMALLTALK.search(req) or _LITERAL.search(req):
        return None
    t = _INLINE.sub(" ", _FENCE.sub(" ", _SYS_PREFIX.sub("", reply or "")))
    known = _norm(req + " " + (context_text or "") + " " + " ".join(tool_names or ()))
    unsup = set()
    for rx in _SPEC:
        for m in rx.finditer(t):
            v = m.group(0).strip().rstrip(".,;:")
            if v and _norm(v) not in known and _norm(v.rstrip("()")) not in known:
                unsup.add(v)
    if _FIN.search(t) or _FIN.search(req):
        for m in _TICKER.finditer(t):
            if _norm(m.group(0)) not in known:
                unsup.add(m.group(0))
    claims = [m.group(0) for m in _CLAIM.finditer(t)]
    from ..memory.episodes import is_moving_target_question, _QUESTION_SHAPE_RE
    moving = is_moving_target_question(req) or bool(_LIVE.search(req) and _QUESTION_SHAPE_RE.search(req))
    if claims or unsup or moving:
        return {"unsupported": sorted(unsup), "claims": claims, "moving": moving}
    return None


#: the detector columns, in report order (name → what it reads; which turns it can fire on)
DETECTORS = {
    "quote_unverified": "the quote checker could not verify ≥1 quoted claim — over turns with a live quote row "
                        "(its marks were right 10 of 52 times in §4OU: an upper bound, not a count of errors)",
    "unbacked_no_tool": "no tool ran, and the reply states specifics nothing in its context backs (§4PK trigger v4)",
    "moving_no_lookup": "no tool ran on a latest/current/live question",
    "claimed_check_no_tool": "no tool ran, and the reply says it checked/ran/pulled something",
    "mechanical_refute": "a CODE-only verifier arm refuted the reply (reply-shape, turn-state, memory-claim, "
                         "file-artifact, web-exec); not the LLM judge, not the vision model",
    "tool_error_streak": "≥2 consecutive FAILED tool calls (§4PM: guard refusals — idempotency, egress, probe "
                         "blocks — and questions to the user are skipped, not failures)",
    "outcome_failed": "the MACHINE label is FAILED: the newest non-human correction (the late verifier — an LLM until "
                      "2026-10-14 — or an operator script), else the label the turn was written with; a turn whose "
                      "only correction is yours is not measured (your 👎 is what it is compared with)",
}
_NO_TOOL = ("unbacked_no_tool", "moving_no_lookup", "claimed_check_no_tool")
_MECH_ROUTES = ("mechanical", "reply-shape", "memory-claim", "turn-state")
#: a "mechanical" verdict the VISION MODEL produced (review M6) — a model, not code
_MODEL_OVERRIDES = ("visual",)
_UNDELIVERED = re.compile(r"\[TURN_INTERRUPTED\]|^\s*_\(Turn cancelled")
#: history approximation: earlier turns in the same channel within this window
HISTORY_WINDOW_S = 60 * 60


def delivered(t: dict) -> bool:
    """False when the reply never reached the owner (client gone, turn
    cancelled) — review M5. An honest "[ATTEMPT_ABORTED_*]" reply WAS sent."""
    return not _UNDELIVERED.search(str(t.get("final_response") or ""))


def detect(traj: dict, turn_row: Optional[dict], quote_rows: List[dict], verdicts: List[dict],
           history_context: str = "", tool_names: Iterable[str] = ()) -> Dict[str, Optional[bool]]:
    """Every detector over one owner turn. ``None`` = not measured on this
    turn (outside the detector's population), never "did not fire"."""
    tools = [t for t in (traj.get("tool_calls") or []) if isinstance(t, dict)]
    ran = bool(tools) if turn_row is None else bool(turn_row.get("tools_ran"))
    out: Dict[str, Optional[bool]] = {}
    if quote_rows:
        out["quote_unverified"] = any(int(q.get("failed") or 0) + len(q.get("missing") or []) > 0
                                      for q in quote_rows)
    else:
        out["quote_unverified"] = None
    if ran:
        for k in _NO_TOOL:
            out[k] = None
    elif turn_row is not None:                       # LIVE: the full context, recorded by the turn
        out["unbacked_no_tool"] = int(turn_row.get("unsupported_n") or 0) > 0
        out["moving_no_lookup"] = bool(turn_row.get("moving"))
        out["claimed_check_no_tool"] = int(turn_row.get("claims_n") or 0) > 0
    else:                                            # HISTORY: an approximation of that context
        reply = str(traj.get("final_response") or "")
        try:
            from .reply_smoothing import strip_system_notes
            reply = strip_system_notes(reply)
        except Exception:  # noqa: BLE001
            pass
        ub = unbacked_no_tool(reply, str(traj.get("user_request") or ""),
                              str(traj.get("system_prompt") or "") + " " + history_context, False, tool_names)
        out["unbacked_no_tool"] = bool(ub and ub["unsupported"])
        out["moving_no_lookup"] = bool(ub and ub["moving"])
        out["claimed_check_no_tool"] = bool(ub and ub["claims"])
    out["mechanical_refute"] = any(
        v.get("verdict") == "REFUTED"
        and (str(v.get("route") or "") in _MECH_ROUTES or v.get("override"))
        and str(v.get("override") or "") not in _MODEL_OVERRIDES
        for v in verdicts)
    # §4PM: real FAILED calls only — guard refusals and questions to the user
    # are not failures (39 of 53 knowledge_base "failures" were guards). A
    # live row carries each call's class from its outcome; history reads text.
    try:
        live_classes = (turn_row or {}).get("classes")
        classes = (list(live_classes) if isinstance(live_classes, list)
                   else [tool_class(tc) for tc in tools])
        best = failed_streak(classes)
    except Exception:  # noqa: BLE001
        best = 0
    out["tool_error_streak"] = best >= 2
    # needs the corrections, so `build` sets it (see `machine_failed`)
    out["outcome_failed"] = None
    return out


def machine_failed(traj: dict, corrections: List[dict]) -> Optional[bool]:
    """The non-human FAILED label, or None when a human label replaced it.
    The collector's overlay writes the owner's 👎 into `outcome`; comparing
    that with the owner's 👎 is circular (it read 6/6 on its first run)."""
    machine = [c for c in corrections
               if not str(c.get("source") or "").startswith(("human_feedback", "user_correction"))]
    if machine:
        last = sorted(machine, key=lambda c: str(c.get("timestamp") or c.get("at") or ""))[-1]
        return str(last.get("outcome") or "").lower() == "failed"
    if corrections:
        return None                     # only human labels: the machine's own was overwritten
    return str(traj.get("outcome") or "").lower() == "failed"


def owner_signal_source(tid: str, corrections: Dict[str, List[dict]], reactions: Dict[str, Optional[bool]]) -> str:
    """Which signal `owner_signal` used: "human" (👍/👎, a correction),
    "judge" (the reaction judge on the next message — an LLM; tool turns
    with a next owner message only), or ""."""
    if any(str(c.get("source") or "").startswith(("human_feedback", "user_correction"))
           for c in corrections.get(tid, [])):
        return "human"
    return "judge" if reactions.get(tid) in (True, False) else ""


def owner_signal(tid: str, corrections: Dict[str, List[dict]], reactions: Dict[str, Optional[bool]]) -> str:
    """"negative" / "positive" / "" — the newest human label first; the
    reaction judge only when no human label exists."""
    human = [c for c in corrections.get(tid, [])
             if str(c.get("source") or "").startswith(("human_feedback", "user_correction"))]
    if human:
        last = sorted(human, key=lambda c: str(c.get("timestamp") or c.get("at") or ""))[-1]
        return "negative" if str(last.get("outcome") or "").lower() == "failed" else "positive"
    r = reactions.get(tid)
    if r is True:
        return "negative"
    if r is False:
        return "positive"
    return ""


# ── §4PM: tool classes and task outcomes (pure; mechanical) ─────────────────

#: a refusal the SYSTEM made before the call did anything — a guard, not the
#: tool failing (review of 10-16: 39 of 53 knowledge_base "failures" were
#: these — 30 idempotency repeats, 8 "the user has not answered yet").
#: Read off a stored row's text; a live row answers by its outcome first.
_GUARD_HEAD = re.compile(
    r"^\s*(?:\[FAILURE BANNER\]\s*)?(?:ERROR:\s*)?(?:"
    r"SYSTEM IDEMPOTENCY\b|SYSTEM BLOCK\b|SYSTEM: \S+ is a COMPOSED address"
    r"|Not done: this \w+ call would send|SYSTEM SEARCH CAP\b|SYSTEM PREFLIGHT\b"
    r"|Not run: the user asked for a STATUS"
    r"|NOT (?:deleted|done|executed): that preview was made outside"
    r"|Error: a probe (?:or background )?request does not)", re.I)
#: a confirmation the OWNER has not given yet (preview → confirm, §4KX) —
#: the tool asking the user, like a designed stop (r3 M2: the forget flow
#: read "guard" while manage_projects' same wait read "asked_user")
_ASK_HEAD = re.compile(r"^\s*(?:\[FAILURE BANNER\]\s*)?(?:ERROR:\s*)?NOT (?:deleted|done|executed): "
                       r"the user has not answered", re.I)
ASK_REASONS = frozenset({"forget_not_confirmed", "reset_not_confirmed"})
#: live reason codes that are guards. Review r2 M1: ONLY codes whose text
#: the stored-row heads above also read as a guard — so a live row and the
#: stored row of the same call agree. A blocked page (`browser_blocked`), a
#: repeated fetch (served or failed), a disabled or unknown tool, an empty
#: write, a constraint breach are the call's real result or the model's own
#: error (`_STRUCK_BLOCK_CODES`), not a guard.
GUARD_REASONS = frozenset({"idempotency_guard", "owner_data_blocked", "lesson_channel_blocked", "status_only",
                           "search_cap", "preflight_block", "deadline_wait"})
TOOL_CLASSES = ("ok", "failed", "guard", "asked_user", "partial", "unresolved")

_IMAGE_TOOLS = ("image_generation", "imagegen", "generate_image", "image")
_RESEARCH_TOOLS = frozenset({"web_search", "deep_research", "fact_check", "darkweb_search", "darkweb_research",
                             "browser", "news_headlines", "youtube", "video"})
_MEMORY_TOOLS = frozenset({"knowledge_base", "recall", "update_profile", "scratchpad", "list_lessons"})
#: the order a turn's kind is chosen in: the first that applies names it
TASK_KINDS = ("image", "build", "project", "research", "memory", "other_tool", "no_tool")


def _tname(t: dict) -> str:
    return str(t.get("name") or "").strip().lower().replace("-", "_")


def tool_class(t: dict) -> str:
    """One tool row → one of TOOL_CLASSES. A LIVE loop row (``content`` is a
    ``ToolOutcome``) answers by its status and reason code; a STORED row (a
    plain ``result`` string, ``error`` text) by the recorder's designed-stop
    mark, the guard heads, then the corpus failure sniffer."""
    from ..distill.outcome_heuristics import is_designed_stop, tool_call_failed, is_unresolved_tool_result
    c = t.get("content") if "content" in t else t.get("result")
    text = str(c or "")
    st = getattr(getattr(c, "status", None), "value", None)
    if is_designed_stop(c):                 # before anything: a clarify-first row is minted too (r1 M1)
        return "asked_user"
    if st is not None:
        # a live row — minted (`_synthetic`) or not — answers by its code and
        # status: a stored row has no synthetic flag, so the flag must not decide
        rc = str(getattr(c, "reason_code", "") or "")
        if rc in ASK_REASONS or (st in ("rejected", "failed") and _ASK_HEAD.search(text)):
            return "asked_user"
        if rc in GUARD_REASONS or (st in ("rejected", "failed") and _GUARD_HEAD.search(text)):
            return "guard"
        if st == "rejected":
            return "failed"
        return {"ok": "ok", "failed": "failed", "partial": "partial", "unresolved": "unresolved"}.get(st, "ok")
    if _ASK_HEAD.search(text):
        return "asked_user"
    if _GUARD_HEAD.search(text):
        return "guard"
    if is_unresolved_tool_result(text):
        return "unresolved"
    from types import SimpleNamespace
    ns = SimpleNamespace(error=t.get("error") or "", result=text, name=t.get("name") or "")
    return "failed" if tool_call_failed(ns) else "ok"


def failed_streak(classes: Iterable[str]) -> int:
    """Longest run of FAILED calls. Guard refusals and questions to the user
    are skipped (neither breaks nor extends a run): they are not failures."""
    best = run = 0
    for c in classes:
        if c in ("guard", "asked_user"):
            continue
        # r3 M3: a live PARTIAL is a stored "failed" (the sniffer reads its
        # text) — both count, so live and history give one streak
        run = run + 1 if c in ("failed", "partial") else 0
        best = max(best, run)
    return best


def loop_shape(t: dict) -> dict:
    """A stored row in the shape the agent's own readers take (``content`` =
    the result text, ``arguments``, ``error``) — so the same helpers judge
    history and live turns."""
    if "content" in t:
        return t
    return {"name": t.get("name") or "", "content": str(t.get("result") or ""),
            "arguments": t.get("arguments") if isinstance(t.get("arguments"), dict) else {},
            "error": t.get("error") or ""}


def _status_outcome(classes: List[str]) -> str:
    if not classes:
        return "none"
    if classes[-1] == "asked_user":
        return "asked_user"        # r3 M2: preview ok → "not answered yet": the action did NOT happen
    if "failed" in classes:
        return "mixed" if ("ok" in classes or "partial" in classes) else "failed"
    if "asked_user" in classes:
        return "asked_user"
    if all(c == "guard" for c in classes):
        return "blocked"
    return "ok"


def task_outcome(tools: Iterable[dict], reply: str = "") -> Dict[str, object]:
    """{kind, outcome, classes} for one turn, from its tool rows and reply —
    the agent's OWN judgements of each kind of task, reused: the image
    tool's success line (`_turn_generated_images`), the code ledger
    (`code_claims.unbacked_code_writes`), the evidence assessment
    (`evidence_gate.assess_turn_evidence`), and each call's class. A
    mechanical outcome, not a grade: "delivered" says the image link
    reached the reply, not that it is a good image."""
    rows = [loop_shape(t) for t in (tools or []) if isinstance(t, dict)]
    real = [t for t in rows if not t.get("_synthetic")]
    classes = [tool_class(t) for t in rows]
    if not rows:
        return {"kind": "no_tool", "outcome": "none", "classes": classes}
    # the KIND is what the model asked for — minted rows count (a clarify-first
    # image turn is an image turn; a stored row carries no synthetic flag, r1 M1)
    names = [_tname(t) for t in rows]
    imgs: List[str] = []
    try:
        from .agent import _turn_generated_images
        imgs = _turn_generated_images(real)
    except Exception:  # noqa: BLE001
        pass
    if imgs or any(n in _IMAGE_TOOLS for n in names):
        ic = [tool_class(t) for t in rows if _tname(t) in _IMAGE_TOOLS] or classes
        if imgs:
            # shown = the download LINK, as the agent's own check reads it (r1 m2)
            out = ("delivered" if any(f"/api/download/{i}" in str(reply or "") for i in imgs)
                   else "made_not_shown")
        else:
            out = "asked_user" if "asked_user" in ic else ("blocked" if ic and all(c == "guard" for c in ic) else "failed")
        return {"kind": "image", "outcome": out, "classes": classes}
    try:
        from .code_claims import code_evidence, unbacked_code_writes
        if code_evidence(real):
            never, failed = unbacked_code_writes(real)
            out = "last_run_failed" if failed else ("never_run" if never else "ran_ok")
            return {"kind": "build", "outcome": out, "classes": classes}
    except Exception:  # noqa: BLE001
        pass
    for kind, members in (("project", {"manage_projects"}), ("research", _RESEARCH_TOOLS),
                          ("memory", _MEMORY_TOOLS)):
        sel = [t for t in rows if _tname(t) in members]
        if not sel:
            continue
        if kind == "research":
            try:
                from .evidence_gate import assess_turn_evidence
                a = assess_turn_evidence(sel)
                if a.consulted:
                    return {"kind": kind, "outcome": "no_evidence" if a.fires else "evidence", "classes": classes}
            except Exception:  # noqa: BLE001
                pass
        return {"kind": kind, "outcome": _status_outcome([tool_class(t) for t in sel]), "classes": classes}
    return {"kind": "other_tool", "outcome": _status_outcome(classes), "classes": classes}


_ABORT_CAUSE = re.compile(r"\[TURN_INTERRUPTED\]\s*(?:Turn aborted:\s*)?([^\n]{0,120})|_\(Turn cancelled:\s*([^)\n]{0,120})")


def undelivered_cause(reply: str) -> str:
    """Why a turn never reached the owner, from its own marker line."""
    m = _ABORT_CAUSE.search(str(reply or ""))
    if not m:
        return "unknown"
    return (m.group(1) or m.group(2) or "").strip().rstrip(".)_ ") or "interrupted"


def _pct(xs: List[float], p: float) -> Optional[float]:
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return round(float(xs[k]), 1)


# ── the offline builder ─────────────────────────────────────────────────────

def is_owner_turn(t: dict) -> bool:
    if t.get("task_kind") not in (None, "user_request"):
        return False
    ex = t.get("extra") or {}
    rid = str(ex.get("req_id") or "")
    if str(ex.get("requester_role") or "") == "member" or rid.startswith(_NOT_OWNER_PREFIXES):
        return False
    return str(ex.get("origin") or "user") in ("user", "")


def _traj_dict(tr) -> dict:
    """A collector Trajectory as the plain dict the detectors read."""
    if isinstance(tr, dict):
        return tr
    try:
        d = tr.to_dict()
    except Exception:  # noqa: BLE001
        d = dict(getattr(tr, "__dict__", {}) or {})
    d["tool_calls"] = [c if isinstance(c, dict) else
                       {"name": getattr(c, "name", ""), "result": getattr(c, "result", ""),
                        "error": getattr(c, "error", ""), "arguments": getattr(c, "arguments", None)}
                       for c in (d.get("tool_calls") or [])]
    return d


def _load_jsonl(p: Path) -> List[dict]:
    out = []
    try:
        from ..tools.file_system import read_text_nofollow
        text = read_text_nofollow(p)
    except Exception:  # noqa: BLE001 — absent, a link, unreadable: nothing to read
        return out
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def _epoch(ts) -> float:
    try:
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=datetime.timezone.utc)
        return d.timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def _channel(t: dict) -> str:
    ex = t.get("extra") or {}
    rid = str(ex.get("req_id") or "")
    return ("slack" if rid.startswith("slack-") else "other") + (":public" if ex.get("surface") == "public" else "")


def _history_contexts(turns: List[dict], standing: str) -> Dict[str, str]:
    """For each turn: the tool results and user words of the same channel's
    earlier turns within HISTORY_WINDOW_S, plus standing state (profile,
    owner rules) — what a stored row cannot show of the live context."""
    out: Dict[str, str] = {}
    by_ch: Dict[str, List[dict]] = {}
    for t in sorted(turns, key=lambda x: _epoch(x.get("timestamp"))):
        by_ch.setdefault(_channel(t), []).append(t)
    for seq in by_ch.values():
        for i, t in enumerate(seq):
            now = _epoch(t.get("timestamp"))
            parts = [standing]
            for u in seq[:i]:
                if now - _epoch(u.get("timestamp")) <= HISTORY_WINDOW_S:
                    parts.append(str(u.get("user_request") or ""))
                    parts += [str(c.get("result") or "") for c in (u.get("tool_calls") or []) if isinstance(c, dict)]
            out[str(t.get("id"))] = " ".join(parts)
    return out


# ── §4PM: the gold sample — a blind grade every fortnight ───────────────────

#: the reaction judge's verdicts on NO-TOOL turns (owner_seeds writes it) —
#: a separate file so the failure-replay seeds, which read the tool-turn
#: cache, are unchanged
REACTIONS_NOTOOL_FILENAME = "reaction_verdicts_notool.json"
GOLD_DIR_REL = "system/census/gold"
#: the packet id → trajectory key, OUTSIDE the folder the grader is given (r1 m1)
GOLD_KEYS_REL = "system/census/gold_keys"
#: finalize's machine lines a blind grader must not see (r1 m1: the quote
#: checker's own closing line is the quote_unverified detector's output)
#: (r2 m1: not end-anchored — the image note and the "Not run" note follow it)
_BLIND_STRIP = (re.compile(r"\n*_Could not verify against the sources read: [^\n]*\._(?=\n|$)"),
                re.compile(r"\n*-{3}\n\*Not run, as you asked\b[^\n]*\*(?=\n|$)"))
GOLD_EVERY_DAYS = 14
GOLD_N = 30
GRADES = ("RIGHT", "PARTIAL", "WRONG", "FAILED-HONEST", "N/A")
#: §4QY: the grader's kind label (the 10-21 census graded kinds, but ingest dropped them)
GOLD_KINDS = ("image", "build", "bookkeeping", "status", "knowledge", "chat")
#: task outcomes that are a mechanical sign the task did not get done
BAD_TASK_OUTCOMES = frozenset({"failed", "mixed", "blocked", "made_not_shown", "never_run",
                               "last_run_failed", "no_evidence"})
GOLD_BRIEF = """# Blind grading brief (§4PM gold sample)

You grade replies an AI assistant gave its owner. Each file in `pool/` is one
item: `id`, `asked_on` (the date of the reply), `earlier_turns` (the same
conversation's previous messages within the hour, oldest first: the owner's
`question` and the assistant's `reply`, cut to 1,500 characters), `question`
(the owner's message), `tools` (each tool the assistant called, in order,
with what it RETURNED, cut to 1,500 characters), and `reply`. Grade every
item on its own; never compare items. `<REDACTED…>` marks stand for the
owner's private details: treat them as correct unless the item contradicts
itself about them.

Grade the REPLY strictly, as of its `asked_on` date, against what the
assistant could see — the earlier turns and the tool results — and the world:
- **RIGHT**: does what was asked; every specific claim is correct, or is
  what a tool returned (a reply that faithfully reports a tool's output is
  right on that point, even if the tool was wrong), or is plainly flagged
  as an estimate; an action it says it took is one a listed tool took.
- **PARTIAL**: mostly right, with a material omission or one wrong or
  unsupported specific.
- **WRONG**: wrong answer, a different question answered, invented
  specifics (not in the tool results, the earlier turns or the world), a
  live reading with no basis, a claim that a tool showed something it did
  not, or an action it claims that no listed tool took (or that a tool
  REFUSED — read the tool results).
- **FAILED-HONEST**: did not do it and said plainly why.
- **N/A**: nothing gradable (small talk). Grade correctness only: whether a
  request should have been refused is not part of this grade.

Also give each item ONE `kind`, the first that applies: **image** (made or
edited a picture), **build** (wrote or ran code), **bookkeeping** (files,
projects, rules, memory, tasks), **status** (about the assistant itself),
**knowledge** (facts about the world: explanations, how-to, news, products,
versions, people, places), **chat** (anything else).

Check world facts you can with web search. Never put a name, address,
account or any other personal detail from the items into a search. Write one
JSON object per line to the output path you were given:
`{"id": ..., "grade": ..., "kind": ..., "reason": "..."}`. Final message:
counts per grade only. You do not know how the replies were produced; do not
guess.
"""


def _owner_turns(sysd: Path, window: List[str]):
    """(delivered owner turns, undelivered owner turns) over the date
    folders in ``window`` — THE COLLECTOR is the one reader of day files
    (§4EE), so `outcome` carries the corrections overlay. Read-only."""
    from ..distill.collector import TrajectoryCollector
    col = TrajectoryCollector(root=sysd / "trajectories", enabled=False)
    seen, good, bad = set(), [], []
    for day in sorted(window):
        try:
            got = list(col.iter_trajectories(day=day))
        except Exception:  # noqa: BLE001
            got = []
        for tr in got:
            t = _traj_dict(tr)
            tid = t.get("id")
            if not tid or tid in seen or not is_owner_turn(t):
                continue
            seen.add(tid)
            (good if delivered(t) else bad).append(t)
    return good, bad


def _window(today: datetime.date, days: float) -> List[str]:
    return [(today - datetime.timedelta(days=k)).isoformat() for k in range(int(days))]


def _redact_packet(text: str) -> str:
    """The owner's identity values and the built-in secret/PII rules out of
    a packet that leaves the agent for a grader (r3 minor). Packets are never
    read back by the agent, so a placeholder cannot become a fact here."""
    try:
        from ..distill.redact import redact_text, redact_profile_values
        return redact_profile_values(redact_text(str(text or "")))
    except Exception:  # noqa: BLE001
        return str(text or "")


def _blind_reply(reply: str) -> str:
    try:
        from .reply_smoothing import strip_system_notes
        reply = strip_system_notes(reply)
    except Exception:  # noqa: BLE001
        pass
    for rx in _BLIND_STRIP:
        reply = rx.sub("", reply)
    return reply.rstrip()


def draw_gold(home=None, days: float = GOLD_EVERY_DAYS, n: int = GOLD_N,
              today: Optional[datetime.date] = None) -> dict:
    """Draw ``n`` delivered owner turns at random into blind packets under
    gold/<date>/pool, with the grading brief beside them and the id →
    trajectory key under gold_keys/ (outside what the grader is given).
    The window is the ``days`` COMPLETED days before ``today`` (r1 M3: with
    today inside, a same-day re-draw met a bigger population and remapped
    18 of 30 ids), and a sample already drawn for a date is never redrawn.
    Returns {dir, drawn, of} or {error}."""
    import random
    base = _home(home)
    if base is None:
        return {"error": "GHOST_HOME is not set"}
    today = today or datetime.datetime.utcnow().date()
    keyp = base / GOLD_KEYS_REL / f"{today.isoformat()}.json"
    d = base / GOLD_DIR_REL / today.isoformat()
    if keyp.exists() or d.exists():
        return {"error": f"a gold sample was already drawn for {today.isoformat()}: grade it, or remove both "
                         f"{d} and {keyp} to draw again"}
    turns, _ = _owner_turns(base / "system", _window(today - datetime.timedelta(days=1), days))
    if not turns:
        return {"error": "no delivered owner turns in the window — nothing to draw"}
    pick = random.Random(today.isoformat()).sample(turns, min(n, len(turns)))
    # r3 M1: the grader saw neither the tool results nor the conversation, so
    # it graded tool-grounded and follow-up turns blind to what the reply
    # was based on — each packet now carries both
    by_ch: Dict[str, List[dict]] = {}
    for t in sorted(turns, key=lambda x: _epoch(x.get("timestamp"))):
        by_ch.setdefault(_channel(t), []).append(t)

    def _earlier(t):
        seq = by_ch.get(_channel(t), [])
        now = _epoch(t.get("timestamp"))
        prev = [u for u in seq if u is not t and 0 <= now - _epoch(u.get("timestamp")) <= HISTORY_WINDOW_S]
        return [{"question": _redact_packet(str(u.get("user_request") or ""))[:1500],
                 "reply": _redact_packet(_blind_reply(str(u.get("final_response") or "")))[:1500]}
                for u in prev[-3:]]
    import shutil
    import tempfile
    d.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{today.isoformat()}-", dir=d.parent))   # r2 m3: all or nothing
    pool = stage / "pool"
    pool.mkdir()
    key = {}
    for i, t in enumerate(pick, 1):
        gid = f"g{today.strftime('%m%d')}-{i:02d}"
        key[gid] = str(t.get("id"))
        reply = _redact_packet(_blind_reply(str(t.get("final_response") or "")))
        (pool / f"{gid}.json").write_text(json.dumps({
            "id": gid, "asked_on": str(t.get("timestamp") or "")[:10],
            "earlier_turns": _earlier(t),
            "question": _redact_packet(str(t.get("user_request") or "")),
            "tools": [{"name": _tname(c), "returned": _redact_packet(str(c.get("result") or c.get("error") or ""))[:1500]}
                      for c in (t.get("tool_calls") or []) if isinstance(c, dict)],
            "reply": reply}, ensure_ascii=False, indent=1), encoding="utf-8")
    (stage / "brief.md").write_text(GOLD_BRIEF, encoding="utf-8")
    # r3 minor: the sample's own window, so the report reads exactly these days
    (stage / "meta.json").write_text(json.dumps({"days": _window(today - datetime.timedelta(days=1), days)}),
                                     encoding="utf-8")
    try:
        keyp.parent.mkdir(parents=True, exist_ok=True)
        keyp.write_text(json.dumps(key, indent=1), encoding="utf-8")
        os.replace(stage, d)
    except OSError:
        shutil.rmtree(stage, ignore_errors=True)
        keyp.unlink(missing_ok=True)
        raise
    return {"dir": str(d), "drawn": len(pick), "of": len(turns)}


def ingest_gold(grades_path, home=None, date: Optional[str] = None) -> dict:
    """Join a grader's output (one {"id", "grade", "reason"} per line) to
    the trajectories through the sample's key, into gold/<date>/grades.jsonl.
    ``date`` = the sample's folder; default the newest. Rows with an unknown
    id or grade are counted, never written."""
    base = _home(home)
    if base is None:
        return {"error": "GHOST_HOME is not set"}
    gdir = base / GOLD_DIR_REL
    date = _gold_date(gdir, date)
    if date is None:
        return {"error": "no gold sample drawn yet"}
    d = gdir / date
    try:
        from ..tools.file_system import read_text_nofollow     # §4GI: a listed folder, a planted link
        key = json.loads(read_text_nofollow(base / GOLD_KEYS_REL / f"{date}.json"))
    except Exception as e:  # noqa: BLE001
        return {"error": f"no key for the {date} sample: {e}"}
    good, bad, counts = [], 0, {}
    for r in _load_jsonl(Path(grades_path)):
        g = str(r.get("grade") or "").strip().upper()
        tid = key.get(str(r.get("id") or ""))
        if not tid or g not in GRADES:
            bad += 1
            continue
        counts[g] = counts.get(g, 0) + 1
        kind = str(r.get("kind") or "").strip().lower()
        good.append({"id": r["id"], "tid": tid, "grade": g, "reason": str(r.get("reason") or "")[:500],
                     "kind": kind if kind in GOLD_KINDS else "",
                     "graded_at": datetime.datetime.utcnow().isoformat() + "Z"})
    if not good:                       # r2 m2: a typo'd path must not wipe the grades
        return {"error": f"no gradable rows in {grades_path} ({bad} rejected) — nothing written", "rejected": bad}
    (d / "grades.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in good),
                                    encoding="utf-8")
    kinds: Dict[str, int] = {}
    for x in good:                     # §4QY r1 m13: a grader who skipped `kind` shows here
        kinds[x["kind"] or "none"] = kinds.get(x["kind"] or "none", 0) + 1
    return {"date": date, "ingested": len(good), "rejected": bad, "grades": counts, "kinds": kinds, "of": len(key)}


# ── §4QY: why knowledge answers fail ────────────────────────────────────────
# Every knowledge turn graded WRONG/PARTIAL lands in one bucket: A = no lookup
# returned anything (answered from memory; read mechanically off the tool
# rows), B = looked up, but the faulted specific is in no tool result, C = it
# IS in a tool result (the source was wrong). B/C need a reader who sees the
# tool results; the packets reuse the gold ids, so no second key is written.
SPLIT_GRADES = ("WRONG", "PARTIAL")
SPLIT_BUCKETS = ("B", "C", "OTHER")
SPLIT_TOOL_CHARS = 50_000
#: the owner's memory text stays at the gold sample's 1,500 (privacy parity);
#: knowledge_base is the owner's DOCUMENTS, evidence a reader must see whole (r2 m3)
SPLIT_MEMORY_CHARS = 1_500
_SPLIT_CAPPED = _MEMORY_TOOLS - {"knowledge_base"}
#: pre-registered (journal §4QY): below this share of A, a "should I look it
#: up first?" decision is the wrong lever (§4PK fired on 3.4 %)
SPLIT_A_FLOOR = 0.20
SPLIT_MIN_N = 8
SPLIT_BRIEF = """# Knowledge-failure split brief (§4QY)

Each file in `pool/` is one reply an AI assistant gave its owner that a blind
grader marked WRONG or PARTIAL; `grader_reason` says why. `tools` holds what
each tool the assistant called RETURNED, in order; `earlier_turns` holds the
same conversation's previous turns within the hour, with their tools — the
assistant could see those too. `cut: true` = the text shown here is not the
whole result. Judge every item on its own.

Find the specific the grader faulted (if several, the main one) and put the
item in ONE bucket:
- **B**: that specific is NOT in any tool result — the reply added it beyond
  what was read (from memory, a guess, or a misreading that changes it).
- **C**: that specific IS in a tool result (this turn's or an earlier
  turn's) — the reply reported its source
  faithfully and the source was wrong, stale, or about something else.
- **OTHER**: the fault is not a wrong specific (an omission, a different
  question answered, a format problem), or you cannot tell — say why. A
  `cut` result that might hold the specific counts here.

Judge only against the tool results; do not search the web. Write one JSON
object per line to the output path you were given:
`{"id": ..., "bucket": "B" | "C" | "OTHER", "claim": "<the faulted specific, short>", "where": "<tool + short quote if C, else empty>"}`.
Final message: counts per bucket only. You do not know how the replies were
produced; do not guess.
"""


def _split_rows(grades: Path) -> List[dict]:
    """The knowledge WRONG/PARTIAL rows of a grades file, one per id (the
    last line for an id wins: a grader's correction, r2 m2)."""
    last = {str(r.get("id")): r for r in _load_jsonl(grades)}
    return [r for r in last.values() if r.get("kind") == "knowledge" and r.get("grade") in SPLIT_GRADES]


def _grades_digest(rows: List[dict]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(sorted([str(r.get("id")), str(r.get("grade")), str(r.get("reason") or "")]
                                            for r in rows)).encode()).hexdigest()


def _gold_date(gdir: Path, date: Optional[str]) -> Optional[str]:
    if date is not None:
        return date
    dates = sorted(p.name for p in gdir.iterdir()
                   if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")) if gdir.is_dir() else []
    return dates[-1] if dates else None


#: tools whose result is evidence about the world: the owner's documents and
#: files count, recall (owner facts) does not (r1 M3: transcripts, weather)
_LOOKUP_TOOLS = _RESEARCH_TOOLS | {"knowledge_base", "file_system", "system_utility", "youtube_transcribe",
                                   "vision_analysis"}


def _row_has_evidence(c: dict) -> bool:
    """One stored row brought back evidence: the agent's own empty-result
    reading (`evidence_gate`, r1 m8: "No results" or a bot-check page is no
    evidence) where it knows the tool, else the row's class."""
    if not isinstance(c, dict) or _tname(c) not in _LOOKUP_TOOLS:
        return False
    args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
    if _tname(c) == "system_utility" and args.get("action") != "check_weather":
        return False                   # r2 m4: a health or location check is not world evidence
    try:
        from .evidence_gate import EVIDENCE_TOOLS, assess_turn_evidence
        if _tname(c) in EVIDENCE_TOOLS:
            return assess_turn_evidence([loop_shape(c)]).substantive > 0
    except Exception:  # noqa: BLE001
        pass
    return tool_class(c) in ("ok", "partial")


def _looked_up(t: dict, earlier: Iterable[dict] = ()) -> bool:
    """A lookup returned evidence in this turn or in an earlier turn the
    reply could see (r1 M2); a failed or empty search is no evidence: A."""
    return any(_row_has_evidence(c) for u in (t, *earlier) for c in (u.get("tool_calls") or []))


def _full_results(base: Path, days: Iterable[str]) -> Dict[tuple, dict]:
    """(trajectory id, tool index) → the §4LH sidecar row: a trajectory row
    keeps 4,000 chars per result, the full text lives beside it (r1 C1)."""
    out: Dict[tuple, dict] = {}
    for day in days:
        for f in sorted((base / "system" / "trajectory_results" / day).glob("*.jsonl")):
            for r in _load_jsonl(f):
                if isinstance(r.get("index"), int):
                    out[(str(r.get("trajectory_id")), r["index"])] = r
    return out


def _split_tools(t: dict, full: Dict[tuple, dict]) -> List[dict]:
    rows = []
    for i, c in enumerate(t.get("tool_calls") or []):
        if not isinstance(c, dict):
            continue
        side = full.get((str(t.get("id")), i))
        txt = str((side or {}).get("result") or c.get("result") or c.get("error") or "")
        cap = SPLIT_MEMORY_CHARS if _tname(c) in _SPLIT_CAPPED else SPLIT_TOOL_CHARS
        txt = _redact_packet(txt)
        rows.append({"name": _tname(c), "returned": txt[:cap],
                     "cut": len(txt) > cap or bool((side or {}).get("truncated"))
                            or (side is None and len(str(c.get("result") or "")) >= 4000)})
    return rows


def _earlier_in_channel(turns: List[dict], t: dict, k: int = 3) -> List[dict]:
    """The same conversation's previous turns within HISTORY_WINDOW_S, oldest
    first — the window the gold packets show."""
    now, ch = _epoch(t.get("timestamp")), _channel(t)
    prev = sorted((u for u in turns if u is not t and _channel(u) == ch
                   and 0 <= now - _epoch(u.get("timestamp")) <= HISTORY_WINDOW_S),
                  key=lambda u: _epoch(u.get("timestamp")))
    return prev[-k:]


def draw_split(home=None, date: Optional[str] = None) -> dict:
    """Bucket the knowledge WRONG/PARTIAL turns of a graded gold sample: A
    mechanically, the rest into blind packets under gold/<date>/split/pool
    with SPLIT_BRIEF beside them. Never redrawn. Returns {dir, a, drawn,
    missing, of} or {error}."""
    import shutil
    import tempfile
    base = _home(home)
    if base is None:
        return {"error": "GHOST_HOME is not set"}
    gdir = base / GOLD_DIR_REL
    date = _gold_date(gdir, date)
    if date is None:
        return {"error": "no gold sample drawn yet"}
    d = gdir / date
    out = d / "split"
    if out.exists():
        return {"error": f"the {date} split was already drawn: read it, or remove {out} to draw again"}
    rows = _split_rows(d / "grades.jsonl")
    if not rows:
        return {"error": f"no knowledge WRONG/PARTIAL rows in the {date} grades (were they graded with a kind?)"}
    try:
        days = json.loads((d / "meta.json").read_text(encoding="utf-8"))["days"]
    except Exception as e:  # noqa: BLE001
        return {"error": f"no window for the {date} sample: {e}"}
    turns, _ = _owner_turns(base / "system", days)
    by_id = {str(t.get("id")): t for t in turns}
    full = _full_results(base, days)
    a, drawn, missing = [], [], []
    stage = Path(tempfile.mkdtemp(prefix=".split-", dir=d))
    try:
        pool = stage / "pool"
        pool.mkdir()
        for r in rows:
            t = by_id.get(str(r.get("tid")))
            if t is None:
                missing.append(r["id"])
                continue
            earlier = _earlier_in_channel(turns, t)
            if not _looked_up(t, earlier):
                a.append(r["id"])
                continue
            drawn.append(r["id"])
            (pool / f"{r['id']}.json").write_text(json.dumps({
                "id": r["id"], "asked_on": str(t.get("timestamp") or "")[:10],
                "earlier_turns": [{"question": _redact_packet(str(u.get("user_request") or ""))[:1500],
                                   "reply": _redact_packet(_blind_reply(str(u.get("final_response") or "")))[:1500],
                                   "tools": _split_tools(u, full)} for u in earlier],
                "question": _redact_packet(str(t.get("user_request") or "")),
                "tools": _split_tools(t, full),
                "reply": _redact_packet(_blind_reply(str(t.get("final_response") or ""))),
                "grader_reason": _redact_packet(str(r.get("reason") or ""))}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        (stage / "brief.md").write_text(SPLIT_BRIEF, encoding="utf-8")
        (stage / "meta.json").write_text(json.dumps({"a": a, "drawn": drawn, "missing": missing, "of": len(rows),
                                                     "grades": _grades_digest(rows)}), encoding="utf-8")
        os.replace(stage, out)
    except BaseException:              # r1 m10: no stage folder left behind, whatever failed
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"dir": str(out), "a": len(a), "drawn": len(drawn), "missing": len(missing), "of": len(rows)}


def split_decision(counts: Dict[str, int]) -> dict:
    """The pre-registered rule over {A, B, C, OTHER, DISAGREE} counts: A is a
    candidate only at SPLIT_A_FLOOR of the known buckets; the largest
    candidate names the lever; a tie goes to B (the lever kind that has moved
    answers here, §4OJ), then A (its first step is a silent shadow), then C."""
    n = sum(counts.get(k, 0) for k in ("A", "B", "C", "OTHER", "DISAGREE"))
    a_share = counts.get("A", 0) / n if n else 0.0
    cands = {"B": counts.get("B", 0), "C": counts.get("C", 0)}
    if a_share >= SPLIT_A_FLOOR:
        cands["A"] = counts.get("A", 0)
    order = ("B", "A", "C")
    lever = (max(order, key=lambda k: (cands.get(k, -1), -order.index(k)))
             if any(cands.values()) else None)
    return {"n": n, "a_share": round(a_share, 3), "lever": lever, "underpowered": n < SPLIT_MIN_N}


def ingest_split(reader_paths, home=None, date: Optional[str] = None) -> dict:
    """Join the readers' bucket files to a drawn split. With two or more
    readers an item counts in a bucket only when ALL of them rated it and
    agree (DISAGREE otherwise; rated by fewer = PARTLY_READ, outside n, r1
    M4). A reader file with no usable row is refused. Writes
    split/result.json; returns it, or {error}."""
    base = _home(home)
    if base is None:
        return {"error": "GHOST_HOME is not set"}
    gdir = base / GOLD_DIR_REL
    date = _gold_date(gdir, date)
    if date is None:
        return {"error": "no gold sample drawn yet"}
    sd = gdir / date / "split"
    try:
        meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"error": f"no split drawn for {date}: {e}"}
    drawn = set(meta.get("drawn") or [])
    # r1 m9 / r2 m6: grades re-ingested after the draw (ids, grade or reason) → stale
    if _grades_digest(_split_rows(gdir / date / "grades.jsonl")) != meta.get("grades"):
        return {"error": f"the {date} grades changed after the split was drawn — remove {sd} and draw it again"}
    paths = [Path(p) for p in reader_paths]
    # r2 m1: one file twice is one reader, not an agreement; two files with the
    # same bytes may be honest agreement on a short sample, so only flagged
    if len({p.resolve() for p in paths}) < len(paths):
        return {"error": "the same reader file was given twice — nothing written"}
    blobs = []
    for p in paths:
        try:
            blobs.append(p.read_bytes())
        except OSError:
            blobs.append(None)
    identical = len(paths) > 1 and None not in blobs and len(set(blobs)) < len(blobs)
    readers, rejected, conflicts = [], 0, 0
    for p in paths:
        got, bad_ids = {}, set()
        for r in _load_jsonl(Path(p)):
            gid = str(r.get("id") or "")
            b = str(r.get("bucket") or "").strip().upper()
            if gid not in drawn or b not in SPLIT_BUCKETS:
                rejected += 1
            elif got.get(gid, b) != b:            # r1 m5: one reader, two buckets → that rating is void
                bad_ids.add(gid)
            else:
                got[gid] = b
        conflicts += len(bad_ids)
        for gid in bad_ids:
            got.pop(gid, None)
        if not got:
            return {"error": f"no usable rows in {p} ({rejected} rejected so far) — nothing written"}
        readers.append(got)
    counts = {"A": len(meta.get("a") or []), "B": 0, "C": 0, "OTHER": 0, "DISAGREE": 0,
              "PARTLY_READ": 0, "UNREAD": 0}
    for gid in sorted(drawn):
        seen = [r[gid] for r in readers if gid in r]
        if not seen:
            counts["UNREAD"] += 1
        elif len(seen) < len(readers):
            counts["PARTLY_READ"] += 1
        else:
            counts[seen[0] if len(set(seen)) == 1 else "DISAGREE"] += 1
    rated = counts["B"] + counts["C"] + counts["OTHER"] + counts["DISAGREE"]
    missing = len(meta.get("missing") or [])
    res = {"date": date, "readers": len(readers), "rejected": rejected, "conflicts": conflicts,
           "identical_files": identical, "counts": counts,
           "agreement": {"agreed": rated - counts["DISAGREE"], "of": rated} if len(readers) > 1 else None,
           "missing": missing,
           "complete": counts["UNREAD"] == counts["PARTLY_READ"] == missing == 0 and len(readers) > 1,
           "rule": {"a_floor": SPLIT_A_FLOOR, "min_n": SPLIT_MIN_N}, **split_decision(counts)}
    (sd / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _graded_sample_days(base: Path, days: float = GOLD_EVERY_DAYS) -> List[str]:
    """The date folders every GRADED sample was drawn from (r2 M2: a sample
    covers the 14 days BEFORE its draw, so the report's own window at the
    next re-read no longer holds it — the builder reads these days too)."""
    out = set()
    gdir = base / GOLD_DIR_REL
    try:
        for p in (gdir.iterdir() if gdir.is_dir() else ()):
            if p.is_dir() and not p.is_symlink() and not p.name.startswith(".") and (p / "grades.jsonl").exists():
                try:
                    d = datetime.date.fromisoformat(p.name)
                except ValueError:
                    continue
                try:
                    from ..tools.file_system import read_text_nofollow
                    meta = json.loads(read_text_nofollow(p / "meta.json"))
                    out.update(str(x) for x in meta.get("days") or [])
                    continue
                except Exception:  # noqa: BLE001 — a sample drawn before meta.json existed
                    pass
                out.update(_window(d - datetime.timedelta(days=1), days))
    except OSError:
        pass
    return sorted(out)


def gold_agreement(base: Path, rows: List[dict], today: datetime.date) -> dict:
    """How often the machine signals catch what the blind grader marked
    WRONG or PARTIAL, and how often they fire on what it marked RIGHT —
    the instruments' own error rate. ``due`` when the newest graded sample
    is older than GOLD_EVERY_DAYS (or none exists)."""
    gdir = base / GOLD_DIR_REL
    grades: Dict[str, str] = {}
    graded_dates = []
    try:
        for p in sorted(gdir.iterdir()) if gdir.is_dir() else []:
            if not p.is_dir() or p.is_symlink() or p.name.startswith("."):
                continue
            got = _load_jsonl(p / "grades.jsonl")
            if got:
                graded_dates.append(p.name)
            for g in got:
                grades[str(g.get("tid") or "")] = str(g.get("grade") or "")
    except OSError:
        pass
    last = graded_dates[-1] if graded_dates else None
    try:
        age = (today - datetime.date.fromisoformat(last)).days if last else None
    except ValueError:
        age = None
    out = {"last_graded": last, "due": age is None or age > GOLD_EVERY_DAYS, "graded_total": len(grades)}
    gr = [r for r in rows if r["tid"] in grades]
    out["in_window"] = len(gr)
    if not gr:
        return out

    # each signal by its own source (r1 M4): code detectors, the machine
    # label (an LLM late verifier until 10-14), your 👎, the reaction judge
    # (an LLM), the mechanical task outcome — never folded together
    def sig(r):
        neg = r["owner"] == "negative"
        return {"detectors": any(v for k, v in r["det"].items() if k != "outcome_failed"),
                "machine_label": bool(r["det"].get("outcome_failed")),
                "yours": neg and r["source"] == "human",
                "judge": neg and r["source"] == "judge",
                "task_outcome": str(r["task"].get("outcome")) in BAD_TASK_OUTCOMES}

    keys = ("detectors", "machine_label", "yours", "judge", "task_outcome")
    bad = [r for r in gr if grades[r["tid"]] in ("WRONG", "PARTIAL")]
    right = [r for r in gr if grades[r["tid"]] == "RIGHT"]
    out["grades"] = {g: sum(1 for r in gr if grades[r["tid"]] == g) for g in GRADES}
    out["wrong_or_partial"] = len(bad)
    out["caught"] = {k: sum(1 for r in bad if sig(r)[k]) for k in keys}
    out["caught"]["any"] = sum(1 for r in bad if any(sig(r).values()))
    out["false_alarms_on_right"] = {"of": len(right), "any": sum(1 for r in right if any(sig(r).values()))}
    out["false_alarms_on_right"].update({k: sum(1 for r in right if sig(r)[k]) for k in keys})   # r3 M4
    return out


def build(home=None, days: float = 14.0, today: Optional[datetime.date] = None,
          tool_names: Iterable[str] = ()) -> dict:
    """The census over the last ``days`` days (date folders strictly after
    today − days, so 14 days = 14 folders)."""
    base = _home(home)
    if base is None:
        return {"error": "GHOST_HOME is not set"}
    sysd = base / "system"
    today = today or datetime.datetime.utcnow().date()
    cutoff = (today - datetime.timedelta(days=days)).isoformat()
    window = _window(today, days)
    turns, bad = _owner_turns(sysd, window)
    in_window = {str(t.get("id")) for t in turns}
    # r2 M2: the graded samples' own days, outside this window, for the gold table only
    extra_days = [d for d in _graded_sample_days(base) if d not in window]
    if extra_days:
        more, _ = _owner_turns(sysd, extra_days)
        turns = turns + [t for t in more if str(t.get("id")) not in in_window]
    undelivered: List[dict] = []
    for t in bad:
        fr = str(t.get("final_response") or "")
        undelivered.append({"tid": str(t.get("id")), "day": str(t.get("timestamp") or "")[:10],
                            "cause": undelivered_cause(fr), "duration_s": t.get("duration_s"),
                            "kind": task_outcome(t.get("tool_calls") or [], fr)["kind"]})
    verdicts: Dict[str, List[dict]] = {}
    for day in list(window) + extra_days:              # by name: no directory walk (§4GI)
        for v in _load_jsonl(sysd / "verdicts" / f"{day}.jsonl"):
            verdicts.setdefault(str(v.get("trajectory_id") or ""), []).append(v)
    corrections: Dict[str, List[dict]] = {}
    for c in _load_jsonl(sysd / "trajectories" / "corrections.jsonl"):
        corrections.setdefault(str(c.get("trajectory_id") or ""), []).append(c)
    try:
        from ..tools.file_system import read_text_nofollow
        reactions = json.loads(read_text_nofollow(sysd / "selfplay" / "reaction_verdicts.json"))
        reactions = reactions if isinstance(reactions, dict) else {}
    except Exception:  # noqa: BLE001
        reactions = {}
    try:                       # §4PM: the judge on NO-TOOL turns, its own cache
        from ..tools.file_system import read_text_nofollow
        nt = json.loads(read_text_nofollow(sysd / "selfplay" / REACTIONS_NOTOOL_FILENAME))
        for k, v in (nt.items() if isinstance(nt, dict) else ()):
            reactions.setdefault(k, v)
    except Exception:  # noqa: BLE001
        pass
    live_turn: Dict[str, dict] = {}
    live_quote: Dict[str, List[dict]] = {}
    fast = commands = 0
    for r in _load_jsonl(base / QUOTE_REL):
        rid = str(r.get("req_id") or "")
        if r.get("kind") == "fast":
            fast += cutoff < str(r.get("at") or "")[:10] <= today.isoformat()
        elif r.get("kind") == "command":       # §4PY
            commands += cutoff < str(r.get("at") or "")[:10] <= today.isoformat()
        elif r.get("kind") == "turn":
            live_turn[rid] = r
        elif r.get("kind") == "quote":
            live_quote.setdefault(rid, []).append(r)
    standing = ""
    try:
        from ..tools.file_system import read_text_nofollow
        standing += read_text_nofollow(sysd / "memory" / "user_profile.json")
        pb = json.loads(read_text_nofollow(sysd / "memory" / "skills_playbook.json"))
        standing += " ".join(str(x.get("lesson") or x.get("text") or "") for x in pb
                             if isinstance(x, dict) and x.get("origin") == "owner_rule")
    except Exception:  # noqa: BLE001
        pass
    hist = _history_contexts(turns, standing)

    rows = []
    for t in turns:
        tid = str(t.get("id"))
        rid = str((t.get("extra") or {}).get("req_id") or "")
        lt = live_turn.get(rid)
        d = detect(t, lt, live_quote.get(rid, []), verdicts.get(tid, []), hist.get(tid, ""), tool_names)
        d["outcome_failed"] = machine_failed(t, corrections.get(tid, []))
        stored = [c for c in (t.get("tool_calls") or []) if isinstance(c, dict)]
        if lt is not None and isinstance(lt.get("task"), dict):
            task = dict(lt["task"])
            classes = list(lt.get("classes") or [])
            cnames = list(lt.get("call_names") or [])
        else:
            to = task_outcome(stored, str(t.get("final_response") or ""))
            task = {"kind": to["kind"], "outcome": to["outcome"]}
            classes = to["classes"]
            cnames = [_tname(c) for c in stored]
        rows.append({"tid": tid, "day": str(t.get("timestamp") or "")[:10],
                     "tools": bool(t.get("tool_calls")), "live": lt is not None, "det": d,
                     "owner": owner_signal(tid, corrections, reactions),
                     "source": owner_signal_source(tid, corrections, reactions),
                     "task": task, "task_live": lt is not None and isinstance(lt.get("task"), dict),
                     "guards": [n for n, c in zip(cnames, classes) if c == "guard"],
                     "failed_calls": sum(1 for c in classes if c in ("failed", "partial")),
                     "duration_s": t.get("duration_s"), "tokens_in": t.get("tokens_in"),
                     "tokens_out": t.get("tokens_out")})

    all_rows = rows                                    # the gold table may reach outside the window
    rows = [r for r in all_rows if r["tid"] in in_window]

    def _share(xs):
        res = {"n": len(xs)}
        for src in ("human", "judge"):
            lab = [r for r in xs if r["source"] == src]
            res[src] = {"labelled": len(lab), "negative": sum(1 for r in lab if r["owner"] == "negative")}
        return res

    by_det = {}
    for name in DETECTORS:
        pop = [r for r in rows if r["det"][name] is not None]          # review M4: one population
        fired = [r for r in pop if r["det"][name]]
        quiet = [r for r in pop if not r["det"][name]]
        entry = {"fired": len(fired), "of": len(pop), "rate": round(len(fired) / len(pop), 3) if pop else None,
                 "of_all_owner_turns": round(len(fired) / len(rows), 3) if rows else None,
                 "when_fired": _share(fired), "when_quiet": _share(quiet)}
        if name in _NO_TOOL:
            entry["measured_live"] = sum(1 for r in pop if r["live"])
            entry["approximated_from_history"] = sum(1 for r in pop if not r["live"])
        by_det[name] = entry
    negs = [r for r in rows if r["owner"] == "negative"]
    by_task = {}
    for kind in TASK_KINDS:
        xs = [r for r in rows if r["task"].get("kind") == kind]
        if not xs:
            continue
        oc: Dict[str, int] = {}
        for r in xs:
            oc[str(r["task"].get("outcome"))] = oc.get(str(r["task"].get("outcome")), 0) + 1
        by_task[kind] = {"n": len(xs), "outcomes": dict(sorted(oc.items(), key=lambda kv: -kv[1])),
                         "measured_live": sum(1 for r in xs if r["task_live"]),
                         "signals": _share(xs),
                         "duration_s": {"p50": _pct([r["duration_s"] for r in xs], .5),
                                        "p90": _pct([r["duration_s"] for r in xs], .9)},
                         "tokens_in_p50": _pct([r["tokens_in"] for r in xs], .5),
                         "tokens_out_p50": _pct([r["tokens_out"] for r in xs], .5)}
    gtools: Dict[str, int] = {}
    for r in rows:
        for n in r["guards"]:
            gtools[n] = gtools.get(n, 0) + 1
    ucause: Dict[str, int] = {}
    for u in undelivered:
        ucause[u["cause"]] = ucause.get(u["cause"], 0) + 1
    return {"built": datetime.datetime.utcnow().isoformat() + "Z", "days": days, "since": cutoff,
            "owner_turns": len(rows), "undelivered_excluded": len(undelivered),
            "undelivered": {"by_cause": ucause, "turns": undelivered,
                            "over_4_min": sum(1 for u in undelivered if (u["duration_s"] or 0) > 240)},
            "fast_path_turns": fast,
            "system_command_turns": commands,
            "latency": {"p50": _pct([r["duration_s"] for r in rows], .5),
                        "p90": _pct([r["duration_s"] for r in rows], .9),
                        "max": _pct([r["duration_s"] for r in rows], 1.0)},
            "tasks": by_task,
            "guard_refusals": {"calls": sum(gtools.values()), "turns": sum(1 for r in rows if r["guards"]),
                               "by_tool": dict(sorted(gtools.items(), key=lambda kv: -kv[1])),
                               "failed_calls": sum(r["failed_calls"] for r in rows)},
            "gold": gold_agreement(base, all_rows, today),
            "no_tool_turns": sum(1 for r in rows if not r["tools"]),
            "live_rows": sum(1 for r in rows if r["live"]),
            "signal_sources": {src: {"labelled": sum(1 for r in rows if r["source"] == src),
                                     "negative": sum(1 for r in rows if r["source"] == src and r["owner"] == "negative"),
                                     "no_tool_labelled": sum(1 for r in rows if r["source"] == src and not r["tools"])}
                               for src in ("human", "judge")},
            "detectors": by_det,
            "negatives": {"tool_turns": sum(1 for r in negs if r["tools"]),
                          "no_tool_turns": sum(1 for r in negs if not r["tools"]),
                          "caught_by_a_detector": sum(1 for r in negs if any(v for v in r["det"].values())),
                          "uncaught_ids": [r["tid"] for r in negs if not any(v for v in r["det"].values())]}}


def render(report: dict) -> str:
    """The report as a short text table. Human labels and the reaction
    judge's are kept apart (review M7: 18 of 24 "owner negatives" were the
    judge's)."""
    if report.get("error"):
        return f"failure census: {report['error']}"
    ss = report["signal_sources"]
    lines = [f"FAILURE CENSUS — owner turns after {report['since']} ({report['days']:g} days): "
             f"{report['owner_turns']} delivered turns ({report['undelivered_excluded']} undelivered left out), "
             f"{report['no_tool_turns']} with no tool, {report['live_rows']} with a live row",
             f"labels: yours {ss['human']['labelled']} ({ss['human']['negative']} 👎, {ss['human']['no_tool_labelled']} on no-tool turns); "
             f"reaction judge {ss['judge']['labelled']} ({ss['judge']['negative']} negative, "
             f"{ss['judge']['no_tool_labelled']} on no-tool turns — an LLM, not you)",
             f"{'detector':<24}{'fired':>11}{'rate':>8}  {'of all':>7}   yours 👎 fired|quiet    judge − fired|quiet"]
    for name, d in report["detectors"].items():
        wf, wq = d["when_fired"], d["when_quiet"]
        rate = f"{100 * d['rate']:.1f}%" if d["rate"] is not None else "—"
        allr = f"{100 * d['of_all_owner_turns']:.1f}%" if d["of_all_owner_turns"] is not None else "—"
        h = f"{wf['human']['negative']}/{wf['human']['labelled']} | {wq['human']['negative']}/{wq['human']['labelled']}"
        j = f"{wf['judge']['negative']}/{wf['judge']['labelled']} | {wq['judge']['negative']}/{wq['judge']['labelled']}"
        tag = ""
        if name in _NO_TOOL and d.get("approximated_from_history"):
            tag = f"  ({d['approximated_from_history']} approximated from history)"
        lines.append(f"{name:<24}{d['fired']:>5}/{d['of']:<5}{rate:>8}  {allr:>7}   {h:<20} {j}{tag}")
    if not report["detectors"]["quote_unverified"]["of"]:
        lines.append("(quote_unverified: no live quote rows yet — measured from §4PL on)")
    nb = report["negatives"]
    lines.append(f"negative turns (yours or the judge's): {nb['tool_turns']} with tools, {nb['no_tool_turns']} without; "
                 f"{nb['caught_by_a_detector']} caught by a detector or the machine label, {len(nb['uncaught_ids'])} by none")
    if "tasks" in report:
        lines.append("")
        lines.append(f"{'task kind':<12}{'turns':>6}  {'p50 s':>6} {'p90 s':>6} {'tok in':>7}  "
                     f"yours 👎/lab  judge −/lab  outcomes (mechanical)")
        for kind, d in report["tasks"].items():
            sg = d["signals"]
            oc = ", ".join(f"{k} {v}" for k, v in d["outcomes"].items())
            live = f" [{d['measured_live']} live]" if d.get("measured_live") else ""
            lines.append(f"{kind:<12}{d['n']:>6}  {_fmt(d['duration_s']['p50']):>6} {_fmt(d['duration_s']['p90']):>6} "
                         f"{_fmt(d['tokens_in_p50']):>7}  {sg['human']['negative']:>5}/{sg['human']['labelled']:<6}"
                         f"{sg['judge']['negative']:>5}/{sg['judge']['labelled']:<6} {oc}{live}")
        lt = report["latency"]
        lines.append(f"latency, all delivered turns: p50 {_fmt(lt['p50'])} s · p90 {_fmt(lt['p90'])} s · max {_fmt(lt['max'])} s")
        gr = report["guard_refusals"]
        if gr["calls"]:
            top = ", ".join(f"{k} {v}" for k, v in list(gr["by_tool"].items())[:5])
            lines.append(f"guard refusals (not failures): {gr['calls']} calls in {gr['turns']} turns ({top}); "
                         f"real failed calls {gr['failed_calls']}")
        ud = report["undelivered"]
        if report["undelivered_excluded"]:
            cs = ", ".join(f"{k} {v}" for k, v in ud["by_cause"].items())
            lines.append(f"undelivered: {report['undelivered_excluded']} ({cs}); {ud['over_4_min']} ran over 4 min first")
        lines.append(f"fast-path greetings (no trajectory): {report['fast_path_turns']} recorded")
        lines.append(f"owner commands answered by the system (no trajectory): "
                     f"{report.get('system_command_turns', 0)} recorded")
        g = report.get("gold") or {}
        if g.get("in_window"):
            c = g["caught"]
            fa = g["false_alarms_on_right"]
            lines.append(f"gold sample ({g['in_window']} graded here, last {g['last_graded']}): "
                         + ", ".join(f"{k} {v}" for k, v in g["grades"].items() if v)
                         + f" · of {g['wrong_or_partial']} wrong/partial, any signal caught {c['any']} "
                         f"(detectors {c['detectors']}, machine label {c['machine_label']}, yours {c['yours']}, "
                         f"judge {c['judge']}, task outcome {c['task_outcome']})"
                         f" · fired on {fa['any']} of {fa['of']} RIGHT (detectors {fa['detectors']}, machine label "
                         f"{fa['machine_label']}, yours {fa['yours']}, judge {fa['judge']}, task outcome {fa['task_outcome']})")
        if g.get("due"):
            lines.append("GOLD SAMPLE DUE: draw one with  scripts/failure_census.py --draw-gold  and grade it blind "
                         f"(last graded: {g.get('last_graded') or 'never'})")
    return "\n".join(lines)


def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")


def write_report(report: dict, home=None) -> Optional[Path]:
    base = _home(home)
    if base is None:
        return None
    d = base / REPORT_DIR_REL
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"report-{datetime.datetime.utcnow().date().isoformat()}.json"
    p.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return p


__all__ = ["record_quote", "record_turn", "record_fast", "tool_class", "failed_streak", "task_outcome",
           "undelivered_cause", "draw_gold", "ingest_gold", "gold_agreement", "TASK_KINDS", "GRADES", "detect", "owner_signal", "owner_signal_source", "build", "render",
           "write_report", "unbacked_no_tool", "DETECTORS", "is_owner_turn", "delivered", "machine_failed"]
