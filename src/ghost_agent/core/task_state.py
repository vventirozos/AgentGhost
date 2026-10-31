"""§4OE-1 — what a LONG task needs from the system, not the model.

Three things the §4OD evaluation found missing on the forced-final path:

* the report reserve was a constant (300 s) smaller than a measured report
  turn (240–253 s + a retry), so every deadline-forced report landed after
  the client had closed — the reserve is now sized from THIS request's own
  longest reply (`note_reply_seconds` / `sized_report_floor`), and the forced
  report itself is capped at `FORCED_REPORT_MAX_TOKENS`;
* nothing carried the plan across 30 tool calls (the per-turn planner left in
  §4NM; the turn-15/30 checkpoints keep 200-char text heads) — `render_task_state`
  writes a factual TASK STATE block from the tool records every turn: files
  written, the last failed command and its error, what ran;
* a forced or aborted turn left nothing a later turn could resume from — the
  same block is persisted as the session's STATE OF WORK (`persist_state_of_work`)
  and shown once at the next turn of that session (`recall_state_of_work`).

Everything here is built from the tool records with the parsers the verifier
uses, so it can name nothing the request did not do.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

#: cap for the forced report turn — the §4OA post-cut cap; a report never
#: needs more, and an uncapped one ran 253 s past a 300 s reserve (§4OD)
FORCED_REPORT_MAX_TOKENS = 8192
#: the reserve is the longest reply so far ×2 (a report + its retry) + slack
REPLY_FACTOR = 2.0
REPLY_SLACK_S = 60.0
#: the state-of-work note expires — a stale "resume" is worse than none
STATE_OF_WORK_TTL_S = 6 * 3600.0
#: bounds for the rendered blocks
TASK_STATE_MAX_FILES = 12
TASK_STATE_MAX_CHARS = 1800
EVIDENCE_HEAD_CHARS = 300
EVIDENCE_MAX_CHARS = 6000
EVIDENCE_TOOLS = ("web_search", "search", "deep_research", "browser", "execute",
                  "vision_analysis", "knowledge_base", "darkweb")

_REPLY_SECONDS: Dict[str, float] = {}
_REPLY_SECONDS_MAX = 512
#: §4OG-5 (f): per request — the slowest generation rate seen (tok/s, replies
#: of ≥ MIN_RATE_TOKENS tokens) and the longest tool batch wall (s)
_REPLY_RATE: Dict[str, float] = {}
_BATCH_WALL: Dict[str, Dict[str, float]] = {}
MIN_RATE_TOKENS = 150
RATE_FLOOR_TOK_S = 5.0
#: the report and the retry the forced-final path may spend on it (the retry
#: is rarely a full report: 1.5 reports, not 2)
REPORT_RETRY_FACTOR = 1.5



#: §4RZ step 5: a file call the user's stop never ran (or stopped) wrote nothing
_STOPPED_HEADS = ("NOT RUN: the user interrupted", "INTERRUPTED BY THE USER:")

def note_reply_seconds(req_id: Any, seconds: float) -> None:
    """Record a reply's wall time; the longest one sizes the report reserve."""
    rid = str(req_id or "")
    if not rid:
        return
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return
    if len(_REPLY_SECONDS) >= _REPLY_SECONDS_MAX and rid not in _REPLY_SECONDS:
        for old in list(_REPLY_SECONDS)[: _REPLY_SECONDS_MAX // 4]:
            _REPLY_SECONDS.pop(old, None)
    _REPLY_SECONDS[rid] = max(_REPLY_SECONDS.get(rid, 0.0), s)


def max_reply_seconds(req_id: Any) -> float:
    return float(_REPLY_SECONDS.get(str(req_id or ""), 0.0))


def _evict(table: Dict[str, float], rid: str) -> None:
    if len(table) >= _REPLY_SECONDS_MAX and rid not in table:
        for old in list(table)[: _REPLY_SECONDS_MAX // 4]:
            table.pop(old, None)


def note_reply_tokens(req_id: Any, tokens: Any, seconds: Any) -> None:
    """§4OG-5 (f): record a reply's generation RATE; the slowest one sizes the
    report (a report is a known number of tokens — up to the 8k cap — at
    this request's own speed, 22–48 tok/s measured, §4OF)."""
    rid = str(req_id or "")
    try:
        n, sec = int(tokens or 0), float(seconds or 0.0)
    except (TypeError, ValueError):
        return
    if not rid or n < MIN_RATE_TOKENS or sec <= 0:
        return
    rate = max(RATE_FLOOR_TOK_S, n / sec)
    _evict(_REPLY_RATE, rid)
    prev = _REPLY_RATE.get(rid)
    _REPLY_RATE[rid] = rate if prev is None else min(prev, rate)


def reply_rate_tok_s(req_id: Any) -> float:
    """The slowest generation rate seen this request, or 0.0 when unknown."""
    return float(_REPLY_RATE.get(str(req_id or ""), 0.0))


def note_batch_wall(req_id: Any, seconds: Any, tool: Any = "") -> None:
    """§4OG-5 (f): the wall time of one tool CALL, kept per tool kind (reader 3
    M5: one 600 s execute must not refuse every later browser call); the
    longest call of a kind is what a call of that kind started just outside
    the reserve can still cost."""
    rid = str(req_id or "")
    try:
        sec = float(seconds or 0.0)
    except (TypeError, ValueError):
        return
    if not rid or sec <= 0:
        return
    _evict(_BATCH_WALL, rid)
    per = _BATCH_WALL.setdefault(rid, {})
    name = str(tool or "*")
    per[name] = max(float(per.get(name, 0.0)), sec)


def max_batch_wall(req_id: Any, tool: Any = None) -> float:
    """The longest call of `tool` this request (or of any tool when None)."""
    per = _BATCH_WALL.get(str(req_id or "")) or {}
    if tool:
        return float(per.get(str(tool), 0.0))
    return float(max(per.values())) if per else 0.0


def forget_reply_seconds(req_id: Any) -> None:
    rid = str(req_id or "")
    _REPLY_SECONDS.pop(rid, None)
    _REPLY_RATE.pop(rid, None)
    _BATCH_WALL.pop(rid, None)


def report_turn_estimate_s(req_id: Any) -> float:
    """§4OG-5 (f): what the forced report and its retry will cost at THIS
    request's measured rate — the cap's worth of tokens ÷ the slowest tok/s
    seen, × the retry factor — or 0.0 before any rate is known."""
    rate = reply_rate_tok_s(req_id)
    if rate <= 0:
        return 0.0
    return REPORT_RETRY_FACTOR * FORCED_REPORT_MAX_TOKENS / rate


def sized_report_floor(base_floor: float, deadline_s: float, req_id: Any) -> float:
    """The report reserve for a deadline: the constant floor, or more when
    this request is slow — the report turn's estimate at its measured rate
    (§4OG-5; before a rate is known, longest reply ×2) plus the longest tool
    batch so far plus slack — never more than a quarter of the deadline (a
    small deadline must not force a report on its first turn, §4ME)."""
    try:
        d = float(deadline_s or 0.0)
    except (TypeError, ValueError):
        d = 0.0
    est = report_turn_estimate_s(req_id)
    if est <= 0 and max_reply_seconds(req_id) > 0:
        est = REPLY_FACTOR * max_reply_seconds(req_id)          # the §4OE sizing until a rate is measured
    want = float(base_floor)
    if est > 0:
        want = max(want, est + REPLY_SLACK_S)         # the batch wall is the pre-dispatch check's (reader 3 M5: not twice)
    if d <= 0:
        return want
    return min(want, max(30.0, 0.25 * d))


# ── the task state block ──────────────────────────────────────────────────── #

def _runs(tools_run) -> List[dict]:
    return [t for t in (tools_run or []) if isinstance(t, dict) and not t.get("_synthetic")]


def _args_of(t: dict) -> dict:
    a = getattr(t.get("content"), "call_args", None)
    if isinstance(a, dict):
        return a
    a = t.get("args") or t.get("call_args")
    return a if isinstance(a, dict) else {}


def _target_of(t: dict) -> str:
    a = _args_of(t)
    return str(a.get("path") or a.get("url") or a.get("query") or a.get("command") or "").strip()


_TEST_PATH_RE = re.compile(r"(?:^|/)(?:tests?|spec|__tests__)/|(?:^|/)test_[^/]*\.\w+$|_test\.\w+$|\.(?:test|spec)\.\w+$", re.I)
_CODE_EXT_RE = re.compile(r"\.(?:py|js|mjs|ts|tsx|jsx|go|rs|java|rb|sh)$", re.I)
_MUTATING_OPS = ("write", "append", "edit", "replace")


def is_test_path(path) -> bool:
    return bool(_TEST_PATH_RE.search(str(path or "").replace("\\", "/")))


def edit_sides(tools_run) -> dict:
    """§4OG-1 (j): which SIDE the fix loop edits — test files vs source — over
    the request and since the last failing run. 18 of 26 edits in the §4OF
    coding probe went to tests ("the test should use absolute values"), and
    nothing noticed."""
    from .strikes import error_line
    out = {"tests": 0, "source": 0, "since_fail_tests": 0, "since_fail_source": 0, "failing_run_seen": False}
    for t in _runs(tools_run):
        n = str(t.get("name") or "")
        a = _args_of(t)
        if n == "execute":
            body = str(t.get("content") or "")
            from ..tools.tool_failure import exec_exit_code
            _c = exec_exit_code(body)
            if error_line(t.get("content") or "", tool="execute") or (_c is not None and int(_c) != 0):
                out["failing_run_seen"] = True
                out["since_fail_tests"] = out["since_fail_source"] = 0
            continue
        if n != "file_system" or str(a.get("operation") or "").lower() not in _MUTATING_OPS:
            continue
        _head = str(t.get("content") or "").lstrip()[:40]
        if error_line(t.get("content") or "", tool="file_system") or _head.startswith(
                ("REJECTED", "REFUSED", "Error", "ERROR", "SYSTEM") + _STOPPED_HEADS):
            continue                                    # a refused (or §4RZ not-run) mutation changed nothing
        p = str(a.get("path") or a.get("filename") or "")
        if not _CODE_EXT_RE.search(p):
            continue
        side = "tests" if is_test_path(p) else "source"
        out[side] += 1
        if out["failing_run_seen"]:
            out["since_fail_" + side] += 1
    return out


def tests_only_note(sides: dict) -> str:
    """The TASK STATE line when, after the last failing run, only test files
    were edited — the model must say so in its reply rather than report a
    green suite as a fix."""
    if not sides.get("failing_run_seen"):
        return ""
    if sides.get("since_fail_tests", 0) >= 1 and sides.get("since_fail_source", 0) == 0:
        return (f"Since the last failing run only TEST files were edited ({sides['since_fail_tests']}); "
                "the source is untouched. If a test was changed to make it pass, say so explicitly in the "
                "reply (which test, what it asserted before) — do not report the suite as fixed otherwise.")
    return ""


_EXERCISE_ACTIONS = ("click", "dblclick", "fill", "type", "press", "select_option", "check", "evaluate")
_LOCAL_HOST_RE = re.compile(r"^https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0)(?::\d+)?(?:/|$)", re.I)


def _page_key(path: str) -> str:
    p = str(path or "").strip().replace("\\", "/")
    p = re.sub(r"^(?:file://)?/?(?:workspace|sandbox)/", "", p)
    return p.lstrip("./").lstrip("/")


def _exercised_pages(url: str, pages) -> set:
    """Which written pages a successful click/interact on `url` exercised
    (r1 M1): a file:// URL exercises the page it names; a local server
    (127.0.0.1/localhost) cannot be mapped to a file, so it counts for every
    page; any other host exercises none."""
    u = str(url or "").strip()
    if _LOCAL_HOST_RE.match(u):
        return set(pages)
    if u.lower().startswith("file://"):
        k = _page_key(re.sub(r"[?#].*$", "", u[len("file://"):]))
        return {p for p in pages if p == k or k.endswith("/" + p) or p.endswith("/" + k)}
    return set()


def untested_pages(tools_run) -> List[str]:
    """§4OJ: .html pages this request wrote with a script and never EXERCISED
    after their last change — no successful browser click/fill/evaluate on
    that page came later. The §4OI census's top error class was a build
    called "working" on a load alone (the calculator probe: navigate, then
    "all buttons functional"). r1: an interact that only loads (goto +
    screenshot) is not a test; a click on another site tests nothing; a
    PARTIAL interact that clicked is a test; a page rewritten static, deleted
    or renamed follows its file."""
    from .strikes import error_line
    changed: Dict[str, int] = {}
    tested: Dict[str, int] = {}
    scripted: set = set()
    last_url = ""
    for i, t in enumerate(_runs(tools_run)):
        n = str(t.get("name") or "")
        a = _args_of(t)
        op = str(a.get("operation") or "").lower()
        body = str(t.get("content") or "")
        if n == "browser":
            from ..tools.browser_support import browser_effective_call
            op, acts = browser_effective_call(a)              # r1: evaluate/fill ran as interact
            acts = acts if op == "interact" else []
            url = str(a.get("url") or a.get("link") or a.get("href") or next((x.get("url") for x in acts if str(x.get("action")) == "goto"
                                            and x.get("url")), "") or last_url)
            last_url = url or last_url
            head = body.lstrip()[:200]
            ok = "STATUS: OK" in head or "STATUS: PARTIAL" in head
            uses = op == "click" or any(str(x.get("action") or "").lower() in _EXERCISE_ACTIONS for x in acts)
            if ok and uses:
                for p in _exercised_pages(url, changed):
                    tested[p] = i
            continue
        if n != "file_system":
            continue
        if error_line(body, tool=n) or body.lstrip()[:40].startswith(("REJECTED", "REFUSED", "Error", "ERROR", "SYSTEM") + _STOPPED_HEADS):
            continue                                           # a refused / failed call changed nothing
        p = _page_key(a.get("path") or a.get("filename") or "")
        if op in ("delete",):
            changed.pop(p, None)
            scripted.discard(p)
            continue
        if op in ("rename", "move"):
            dst = _page_key(a.get("destination") or "")
            if p in changed:
                changed[dst] = changed.pop(p)
                if p in scripted:
                    scripted.discard(p)
                    scripted.add(dst)
            continue
        if op not in _MUTATING_OPS or not p.lower().endswith((".html", ".htm")):
            continue
        _app = a.get("append")
        if op == "write" and a.get("content") == "" and (
                _app is True or str(_app).strip().lower() in ("true", "1", "yes")):
            continue                                           # r1/r4: an empty finish changes nothing
        changed[p] = i
        text = " ".join(str(a.get(k) or "") for k in ("content", "new_string", "replace_with"))
        if re.search(r"<script|<[a-z][^>]*\son[a-z]+\s*=", text, re.I):
            scripted.add(p)
        elif op == "write" and not a.get("append") and not a.get("more_parts") and "part" not in body[:200].lower():
            scripted.discard(p)                                # rewritten whole as a static page
    return [p for p, i in changed.items() if p in scripted and tested.get(p, -1) < i]


def task_state_facts(tools_run, *, ask: str = "") -> dict:
    """The facts a long task needs restated: files touched, the last failure,
    what ran. Pure data; `render_task_state` makes the prose."""
    from .agent import _fs_path_ledger
    from .strikes import error_line
    runs = _runs(tools_run)
    facts: dict = {"ask": " ".join(str(ask or "").split())[:300], "calls": len(runs),
                   "files": [], "retired": [], "last_failure": None, "counts": {}, "failures": 0,
                   "sides": {"tests": 0, "source": 0}}
    if not runs:
        return facts
    try:
        facts["sides"] = edit_sides(runs)
    except Exception:  # noqa: BLE001
        pass
    try:
        left, retired = _fs_path_ledger(runs)
    except Exception:  # noqa: BLE001 — a state block never raises
        left, retired = [], []
    facts["files"] = list(left)[:TASK_STATE_MAX_FILES]
    facts["retired"] = list(retired)[:6]
    for t in runs:
        n = str(t.get("name") or "?")
        facts["counts"][n] = facts["counts"].get(n, 0) + 1
        line = error_line(t.get("content") or "", tool=n or None)     # the OBJECT: a declared status counts (§4KT)
        if line:
            facts["failures"] += 1
            facts["last_failure"] = {"tool": n, "target": _target_of(t)[:160],
                                     "error": " ".join(str(line).split())[:240]}
    return facts


_SCRIPT_SIDE_RE = re.compile(r"Script-side path \(from sandbox cwd\): '(.+?)'\.")


def _download_links(tools_run, files, prefix: str = "") -> List[str]:
    """§4OI (iv) r1 M2: the link the download ROUTE resolves — the tool's own
    script-side spelling (not the model's `/workspace/...`), under the
    project prefix when the request is project-bound, URL-encoded."""
    from urllib.parse import quote
    rel_of: Dict[str, str] = {}
    for t in _runs(tools_run):
        if str(t.get("name") or "") != "file_system":
            continue
        m = _SCRIPT_SIDE_RE.search(str(t.get("content") or ""))
        p = str(_args_of(t).get("path") or _args_of(t).get("filename") or "")
        if m and p:
            rel_of[p] = m.group(1)
    out = []
    for f in files:
        rel = rel_of.get(f) or re.sub(r"^/?(?:workspace|sandbox)/", "", str(f)).lstrip("/")
        pre = str(prefix or "")
        full = rel if (pre and rel.startswith(pre)) else pre + rel    # r2 m2: never doubled; r3 m2: a project's own projects/ folder
        out.append(f"[{rel.rsplit('/', 1)[-1]}](/api/download/{quote(full, safe='/')})")
    return out


def _open_part_line(path) -> str:
    """§4SH: an open parts file as the model needs to judge it — its size and
    how it ends (req cb27058e: the file ended `</html>` and the bare "the LAST
    part has not been sent" read as stale to the model, which then ignored it)."""
    name = str(path).rsplit("/", 1)[-1]
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 160))
            tail = " ".join(fh.read().decode("utf-8", "replace").split())[-60:]
    except OSError:
        return f"`{name}`"
    return f"`{name}` ({size:,} bytes, ends «…{tail}»)"


def render_task_state(tools_run, *, ask: str = "", turn: int = 0, note: str = "", link_prefix: str = "") -> str:
    """The TASK STATE block for DYNAMIC STATE — empty until a tool has run."""
    facts = task_state_facts(tools_run, ask=ask)
    if not facts["calls"]:
        return ""
    lines = ["### TASK STATE (written by the system from this request's tool records)"]
    if turn:
        lines.append(f"Tool calls so far: {facts['calls']} over {turn} turn(s).")
    if facts["files"]:
        lines.append("Files this request wrote or edited (they exist — do not rewrite them from memory, "
                     "read or edit them): " + ", ".join(f"`{p}`" for p in facts["files"]))
    if facts["retired"]:
        lines.append("Deleted this request: " + ", ".join(f"`{p}`" for p in facts["retired"]))
    lf = facts["last_failure"]
    if lf:
        where = f"`{lf['tool']}`" + (f" ({lf['target']})" if lf["target"] else "")
        lines.append(f"Last failure ({facts['failures']} so far): {where}: {lf['error']}")
    if facts["counts"]:
        lines.append("Ran: " + ", ".join(f"{n} ×{c}" for n, c in
                                        sorted(facts["counts"].items(), key=lambda kv: -kv[1])))
    try:                                                              # §4OI (iii)
        from ..tools.file_system import files_left_in_progress
        from ..utils.logging import request_id_context
        _parts = files_left_in_progress(request_id_context.get())
    except Exception:  # noqa: BLE001
        _parts = []
    if _parts:
        lines.append("Open for parts — more is expected, so these files are not checked yet: "
                     + ", ".join(_open_part_line(x) for x in _parts[:6])
                     + ". Append the next part — or, if a file is already complete, finish it: file_system "
                       "operation='write', append=true, content='' (nothing is added; the finish says what "
                       "was checked).")
    _sides = facts.get("sides") or {}
    if _sides.get("tests") or _sides.get("source"):
        lines.append(f"Code edits this request: {_sides.get('source', 0)} to source files, "
                     f"{_sides.get('tests', 0)} to test files.")
    _tn = tests_only_note(_sides)
    if _tn:
        lines.append(_tn)
    try:
        _ut = untested_pages(tools_run)
    except Exception:  # noqa: BLE001 — a state block never raises
        _ut = []
    _ut_line = ("Pages with scripts NOT exercised since their last change (loaded at most — no click or "
                "interact): " + ", ".join(f"`{p}`" for p in _ut[:3]) + ". Loading a page does not test it: "
                "before calling it working, use browser interact to click a control and check what changes, "
                "or say plainly in the reply that it is untested.") if _ut else ""
    if note:
        lines.append(note)
    if facts["files"]:
        # §4OI (iv): "give me the download link" was answered with the VENDOR's download
        # page (postgresql.org/download/) — the model was never shown the file's own link.
        # Last in the block, and only WHOLE links that fit the cap (r1 m7: a cut link gets copied).
        head = "To give the user one of these files, link it exactly like this (a website's own download page is not the file): "
        room = TASK_STATE_MAX_CHARS - len("\n".join(lines)) - 1 - len(head)
        kept = []
        for lk in _download_links(tools_run, facts["files"][:6], link_prefix):
            if len(", ".join(kept + [lk])) > room:
                break
            kept.append(lk)
        if kept:
            lines.append(head + ", ".join(kept))
    if _ut_line and len("\n".join(lines)) + 1 + len(_ut_line) <= TASK_STATE_MAX_CHARS:
        lines.insert(len(lines) - (1 if facts["files"] and lines[-1].startswith("To give the user") else 0), _ut_line)
    text = "\n".join(lines)
    return (text[:TASK_STATE_MAX_CHARS] + " …" if len(text) > TASK_STATE_MAX_CHARS else text) + "\n\n"


# ── state of work across turns ────────────────────────────────────────────── #

def _sow_key(session_id: Any) -> str:
    return f"_state_of_work:{str(session_id or '').strip()}"


def state_of_work_text(tools_run, *, ask: str = "", why: str = "") -> str:
    facts = task_state_facts(tools_run, ask=ask)
    if not facts["calls"]:
        return ""
    parts = [f"The previous turn of this conversation did not finish ({why or 'it was stopped'})."]
    if facts["ask"]:
        parts.append(f"It was asked: {facts['ask']}")
    if facts["files"]:
        parts.append("It wrote or edited: " + ", ".join(f"`{p}`" for p in facts["files"])
                     + " — these files exist now; read them before writing them again.")
    lf = facts["last_failure"]
    if lf:
        where = f"`{lf['tool']}`" + (f" ({lf['target']})" if lf["target"] else "")
        parts.append(f"Its last failure: {where}: {lf['error']}")
    parts.append("Ran: " + ", ".join(f"{n} ×{c}" for n, c in
                                     sorted(facts["counts"].items(), key=lambda kv: -kv[1])))
    return "\n".join(parts)


def persist_state_of_work(scratchpad, session_id: Any, text: str) -> bool:
    """Store the note for the session (no project scope — it must survive a
    project switch). Returns True when stored."""
    sid = str(session_id or "").strip()
    if scratchpad is None or not sid or not str(text or "").strip():
        return False
    try:
        scratchpad.set(_sow_key(sid), {"text": str(text)[:4000], "at": time.time()}, namespace=None)
        return True
    except Exception:  # noqa: BLE001
        return False


def recall_state_of_work(scratchpad, session_id: Any) -> str:
    """The note for this session, or "" (expired notes count as none)."""
    sid = str(session_id or "").strip()
    if scratchpad is None or not sid:
        return ""
    try:
        v = scratchpad.get(_sow_key(sid))
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(v, dict):
        return ""
    try:
        if time.time() - float(v.get("at") or 0) > STATE_OF_WORK_TTL_S:
            return ""
    except (TypeError, ValueError):
        return ""
    return str(v.get("text") or "")


def clear_state_of_work(scratchpad, session_id: Any) -> None:
    sid = str(session_id or "").strip()
    if scratchpad is None or not sid:
        return
    try:
        scratchpad.delete(_sow_key(sid))
    except Exception:  # noqa: BLE001
        pass


def render_state_of_work(text: str) -> str:
    if not str(text or "").strip():
        return ""
    return ("### STATE OF WORK (from the previous, unfinished turn — written by the system)\n"
            f"{text}\nContinue from here if the user asks to; do not redo what is done.\n\n")


# ── evidence for the fallback ─────────────────────────────────────────────── #

def evidence_heads(tools_run, *, limit_chars: int = EVIDENCE_MAX_CHARS) -> str:
    """Heads of EVERY successful evidence-tool output, newest first — the
    §4II digest kept only the last one (49 fetched pages → one ad, §4OD)."""
    from .strikes import error_line
    out: List[str] = []
    used = 0
    for t in reversed(_runs(tools_run)):
        n = str(t.get("name") or "")
        if n not in EVIDENCE_TOOLS:
            continue
        raw = t.get("content") or ""
        content = str(raw)
        if not content.strip() or error_line(raw, tool=n or None):     # the OBJECT, not its text
            continue
        flat = " ".join(content.split())
        # an `execute` output's head is the banner — its TAIL holds the result
        body = flat[-EVIDENCE_HEAD_CHARS:] if n == "execute" and len(flat) > EVIDENCE_HEAD_CHARS else flat[:EVIDENCE_HEAD_CHARS]
        target = _target_of(t)[:100]
        line = f"- `{n}`" + (f" ({target})" if target else "") + f": {body}"
        if used + len(line) > limit_chars:
            break
        out.append(line)
        used += len(line)
    if not out:
        return ""
    return "Evidence gathered (newest first, heads only):\n" + "\n".join(out)
