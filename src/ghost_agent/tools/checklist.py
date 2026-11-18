"""§4UU-2 — the build checklist: one item per feature, ticked only by its own check.

Measured (25 days, 38 owner coding turns): big builds went in one shot — the
web OS 11 times from scratch, 10.2 h — and 37 of 38 replies said "done /
tested / live" while about 5 turns had actually run the program after the
last write. The best-evidenced harness change for a ~30B model is a plan the
model keeps and sees again while it works (FAN-2026, arXiv 2609.20804: a todo
tool took Nemotron-3 30B from 13.6 % to 25.2 % on SWE-bench Verified, with an
ablation; without it the model quit after a median of 5 turns). Anthropic's
long-running harness adds the other half: the model may only flip a
feature's `passes` flag, and only by testing it.

So: `checklist` — the model writes the list (title + check per item); ONLY
the check decides an item's state. There is no "mark done".

* `set` / `add` — items `[{title, check}]`; `set` replaces the list (an item
  with the same title AND check keeps its state).
* `check` — RUNS the checks (all items, or `ids`) and records each result:
  - `page:<file.html>` — the page loads headless with no uncaught JS
    exception (the system's own probe, as after every web write);
  - `file:<path>` — the file exists and is not empty;
  - anything else — a shell command in the sandbox: exit 0 = passed.
* `drop` — remove items (`ids`); `view`; `clear`.

The list lives in the HOST data dir (`system/checklists/<scope>.json`), never
in the sandbox — code running in the sandbox cannot tick an item. Scope: the
project the turn works in, else the conversation (a conversation list idles
out after 24 h; a project's stays until it is cleared).

While items are open, the agent adds one status line to the last tool result
of each tool batch; a reply that says the work is done while items have not
passed gets one repair round (`checklist_directive`).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_ITEMS = 20
MAX_TITLE = 120
MAX_CHECK = 400
CONVERSATION_TTL_S = 24 * 3600
_OUT_TAIL = 300
#: one `check` call stops starting new checks after this long (each check has execute's own time limit)
CHECK_BUDGET_S = 600
_LOCKS: "dict" = {}             # one checklist operation at a time per event loop (load → change → save)
#: the row separator the claim arm parses — never allowed inside a title or a check
_ROW_SEP = " — check: "

_STATUS_MARK = {"passed": "✓", "failed": "✗", "open": "○", "stale": "↻"}


# ── scope + storage ───────────────────────────────────────────────────

def _system_dir(context) -> Optional[Path]:
    md = getattr(context, "memory_dir", None)
    return Path(md).parent if isinstance(md, (str, Path)) else None


def scope_key(context) -> str:
    """'project-<id>' when the turn works in a project, else 'conv-<hash>' for the conversation, else
    'req-<hash>' for the request alone; "" when none can be told."""
    pid = None
    try:
        from .file_system import _safe_project_id
        pid = _safe_project_id(getattr(context, "current_project_id", None))
    except Exception:  # noqa: BLE001
        pid = None
    if pid:
        return "project-" + re.sub(r"[^A-Za-z0-9_-]", "_", str(pid))[:64]
    try:
        from ..utils.logging import conversation_key_context, request_id_context
        ck = conversation_key_context.get() or ""
        if ck:
            return "conv-" + hashlib.sha256(ck.encode("utf-8")).hexdigest()[:20]
        rid = request_id_context.get() or ""
        if rid:
            return "req-" + hashlib.sha256(rid.encode("utf-8")).hexdigest()[:20]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _path_for(context, scope: str) -> Optional[Path]:
    sd = _system_dir(context)
    if sd is None or not scope:
        return None
    return sd / "checklists" / f"{scope}.json"


def load(context, scope: Optional[str] = None) -> Dict[str, Any]:
    """The scope's list ({"items": [...], "updated": ts}); an empty one when absent, unreadable or idled out."""
    scope = scope if scope is not None else scope_key(context)
    p = _path_for(context, scope)
    empty = {"items": [], "updated": 0.0}
    if p is None or not p.is_file():
        return empty
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return empty
    try:
        updated = float(data.get("updated") or 0)
    except (TypeError, ValueError):
        updated = time.time()                          # unreadable: keep the list (expiry is a convenience)
    if not scope.startswith("project-") and time.time() - updated > CONVERSATION_TTL_S:
        try:
            p.unlink()
        except OSError:
            pass
        return empty
    items = []
    for i in data["items"]:                             # a hand-edited / damaged row is dropped, never fatal
        if not (isinstance(i, dict) and i.get("title") and isinstance(i.get("check"), str) and i["check"]):
            continue
        try:
            i["id"] = int(i.get("id"))
        except (TypeError, ValueError):
            continue
        items.append(i)
    data["items"] = items
    data["updated"] = updated
    return data


def _save(context, scope: str, data: Dict[str, Any]) -> bool:
    p = _path_for(context, scope)
    if p is None:
        return False
    data["updated"] = time.time()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return True


# ── items ─────────────────────────────────────────────────────────────

def _item_problem(title: str, check: str) -> str:
    """Why one item cannot be saved ("" when it can)."""
    if "\n" in check or "\r" in check or _ROW_SEP.strip() in check:
        return (f"item {title!r}: a check is ONE line (join commands with && or put them in a script) "
                f"and does not contain '— check:'")
    if not title:
        return "every item needs a title"
    if not check:
        return (f"item {title!r} has no check — every item needs one: a shell command that exits 0 when "
                f"the feature works (e.g. `python3 tests/check_page.py clock`), `page:<file.html>` or "
                f"`file:<path>`")
    if _SWALLOWS_FAILURE_RE.search(check):
        return (f"item {title!r}: the check `{check[:80]}` always exits 0 (`|| true`, `; exit 0` …) — it can "
                f"never fail, so it cannot show the feature works. Drop that part.")
    if not check.lower().startswith(("page:", "file:")) and not runs_something(check):
        return (f"item {title!r}: the check `{check[:80]}` runs nothing — a syntax check, `ls`, `cat`, "
                f"`grep`, `test -f` or `echo` cannot show the feature works. Give a command that RUNS it "
                f"(a test, the program, `python3 <folder>/tests/check_page.py <feature>`, `curl -f` to a "
                f"service running here), or page:<file.html>; a file that only has to exist is "
                f"`file:<path>` (next to at least one check that runs something)")
    return ""


def _coerce_items(items) -> Tuple[List[Dict[str, str]], str]:
    """[{title, check}] from what the model sent (a list of objects, or of "title :: check" strings, or a JSON
    string of either) — or ([], why)."""
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except ValueError:
            items = [ln for ln in items.splitlines() if ln.strip()]
    if not isinstance(items, list) or not items:
        return [], "items must be a non-empty list of {title, check}"
    out, problems = [], []
    for it in items:
        if isinstance(it, str):
            title, _, check = it.partition("::")
            it = {"title": title, "check": check}
        if not isinstance(it, dict):
            return [], "each item is {title, check}"
        title = " ".join(str(it.get("title") or it.get("name") or "").split())[:MAX_TITLE]
        title = title.replace("—", "-")                 # the row separator never appears in a title
        check = str(it.get("check") or it.get("test") or it.get("command") or "").strip()[:MAX_CHECK]
        why = _item_problem(title, check)
        if why:
            problems.append(why)                        # every bad item in ONE answer (one fix round, not one each)
            continue
        out.append({"title": title, "check": check})
    if problems:
        head = f"{len(problems)} items need a different check (nothing was saved):\n" if len(problems) > 1 else ""
        return [], head + "\n".join(("- " + p) if head else p for p in problems)
    if out and all(o["check"].lower().startswith("file:") for o in out):
        return [], ("every check is file: — a file existing does not show a feature works; give at least one "
                    "check that RUNS something")
    return out, ""


#: curl / wget options that take a value (their value is never read as a flag)
_HTTP_VALUE_OPTS = frozenset({"-d", "--data", "--data-raw", "--data-binary", "-H", "--header", "-o", "--output",
                              "-X", "--request", "-u", "--user", "-w", "--write-out", "-A", "--user-agent", "-e",
                              "-m", "--max-time", "-O", "--output-document", "--post-data", "--header"})
_LOCAL_HOST_RE = re.compile(r"^(?:https?://)?(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?(?:/\S*)?$")
_ANSWER_CHECKERS = frozenset({"grep", "egrep", "jq", "python", "python3", "node", "test"})


#: a check whose own failure is swallowed always passes
_SWALLOWS_FAILURE_RE = re.compile(
    r"(?:\|\||;)\s*(?:true|:|exit\s+0|echo\b[^;&|]*|printf\b[^;&|]*)\s*$|\bset\s+\+e\b")


def _service_check(command: str) -> bool:
    """A request to a service running HERE that fails on a bad answer: `curl -f localhost:5000/api` or
    `curl -s 127.0.0.1:8080/ | grep ok` (and the pipeline's failure is not swallowed by a later `| true`)."""
    import shlex
    try:
        lex = shlex.shlex(str(command or ""), posix=True, punctuation_chars=";&|")
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return False
    pipelines, cur, seg = [], [], []
    for tok in toks + [";"]:
        if tok == "|":
            cur.append(seg)
            seg = []
        elif tok and set(tok) <= set(";&|"):
            cur.append(seg)
            pipelines.append(cur)
            cur, seg = [], []
        else:
            seg.append(tok)
    for stages in pipelines:
        stages = [st for st in stages if st]
        if not stages or stages[0][0] not in ("curl", "wget"):
            continue
        words, flags, local, j = stages[0][1:], set(), False, 0
        while j < len(words):
            w = words[j]
            if w in _HTTP_VALUE_OPTS:
                j += 2
                continue
            if w.startswith("--"):
                flags.add(w)
            elif w.startswith("-"):
                flags |= {"-" + c for c in w[1:]}
            elif _LOCAL_HOST_RE.match(w):
                local = True
            j += 1
        if not local:
            continue
        tail = [st[0] for st in stages[1:]]
        if any(t in ("true", ":", "cat", "echo") for t in tail):
            continue
        fails = bool(flags & {"-f", "--fail", "--fail-with-body"}) if stages[0][0] == "curl" else False
        if fails or (tail and tail[0] in _ANSWER_CHECKERS):
            return True
    return False


def runs_something(command: str) -> bool:
    """Does the shell `command` run a program (by the claim arm's own rule, §4UU-3)? A request to a service
    running here counts when a bad answer fails it (`curl -f localhost:8080/api`, `curl -s localhost/x | grep ok`):
    that exercises the running code."""
    if _service_check(command):
        return True
    try:
        from ..core.code_claims import _shell_segments, segment_kind
        return any(segment_kind(w) == "run" for w in _shell_segments(command))
    except Exception:  # noqa: BLE001 — never refuse on a parser failure
        return True


def _ids(ids) -> List[int]:
    if ids is None or ids == "":
        return []
    if isinstance(ids, (int, float)):
        return [int(ids)]
    if isinstance(ids, str):
        return [int(x) for x in re.findall(r"\d+", ids)]
    out = []
    for x in ids if isinstance(ids, list) else []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def counts(data: Dict[str, Any]) -> Tuple[int, int]:
    items = data.get("items") or []
    return sum(1 for i in items if i.get("status") == "passed"), len(items)


#: a row a `check` call printed with its OLD state (it did not run that item) — the claim arm skips it
NOT_RUN_NOW = " (not run in this call)"


def render(data: Dict[str, Any], *, with_output: bool = False, ran_ids=None) -> str:
    items = data.get("items") or []
    if not items:
        return "Checklist: empty."
    passed, total = counts(data)
    lines = [f"Checklist: {passed}/{total} passed."]
    for it in items:
        st = it.get("status") or "open"
        mark = _STATUS_MARK.get(st, "○")
        tail = {"open": " (not checked yet)", "stale": " (passed before the last code change — check again)"}.get(st, "")
        if ran_ids is not None and it["id"] not in ran_ids:
            tail += NOT_RUN_NOW
        lines.append(f"{mark} #{it['id']} {it['title']} — check: {it['check']}{tail}")
        if with_output and st == "failed" and (it.get("last") or {}).get("out"):
            out = str(it["last"]["out"]).strip().replace("\n", "\n    ")
            lines.append(f"    {out}")
    return "\n".join(lines)


def status_line(context) -> str:
    """One line for the end of a tool batch while items are not all passed — "" otherwise."""
    try:
        data = load(context)
    except Exception:  # noqa: BLE001
        return ""
    items = data.get("items") or []
    if not items:
        return ""
    passed, total = counts(data)
    if passed == total:
        return ""
    nxt = [i for i in items if i.get("status") != "passed"][:2]
    shown = "; ".join(f"#{i['id']} {i['title'][:40]}" + {"failed": " ✗", "stale": " ↻"}.get(i.get("status"), "")
                      for i in nxt)
    return f"\n\n[CHECKLIST {passed}/{total} passed · next: {shown}]"


# ── running a check ───────────────────────────────────────────────────

async def _run_one(check: str, *, run_execute, page_probe, sandbox_dir) -> Tuple[bool, str]:
    """(passed, output tail) for one check."""
    c = check.strip()
    if c.lower().startswith("page:"):
        target = c[5:].strip()
        if page_probe is None:
            return False, "page check unavailable here (no browser)"
        res = await page_probe([target])
        if res is None:
            return False, f"'{target}' could not be loaded headless (missing, or it needs a running service)"
        page, block = res
        return (not block), (str(block).strip()[:_OUT_TAIL] if block else f"loaded '{page}' clean")
    if c.lower().startswith("file:"):
        target = c[5:].strip()
        try:
            from .file_system import _get_safe_path
            p = _get_safe_path(sandbox_dir, target)
            if p.is_dir():
                n = sum(1 for _ in p.iterdir())
                return n > 0, f"{target}/: a folder with {n} item(s)"
            ok = p.is_file() and p.stat().st_size > 0
            return ok, (f"{target}: {p.stat().st_size} bytes" if p.is_file() else f"{target}: missing")
        except Exception as e:  # noqa: BLE001
            return False, f"{target}: {e}"
    if run_execute is None:
        return False, "no sandbox to run the check in"
    from ..core import env_limits as _el
    from ..utils.logging import request_id_context as _ridc
    _rid = _ridc.get()
    _blocked = _el.refusal(_rid, c)
    if _blocked:
        return False, _blocked
    res = await run_execute(command=c)
    text = str(res or "")
    from .tool_failure import exec_exit_code
    code = exec_exit_code(text)
    st = str(getattr(getattr(res, "status", None), "value", "") or "")
    ok = code == 0 and st not in ("failed", "rejected", "unresolved")
    try:
        _el.note_result(_rid, c, text, code)
    except Exception:  # noqa: BLE001
        pass
    tail = text.strip()[-_OUT_TAIL:]
    return ok, tail


# ── the tool ──────────────────────────────────────────────────────────

async def tool_checklist(context=None, action: str = "", items=None, ids=None, run_execute=None, page_probe=None,
                         sandbox_dir=None, **kwargs):
    loop = asyncio.get_running_loop()
    lock = _LOCKS.get(id(loop))
    if lock is None:
        if len(_LOCKS) > 8:
            _LOCKS.clear()
        lock = _LOCKS[id(loop)] = asyncio.Lock()
    async with lock:
        return await _tool_checklist(context, action, items, ids, run_execute, page_probe, sandbox_dir, **kwargs)


async def _tool_checklist(context, action, items, ids, run_execute, page_probe, sandbox_dir, **kwargs):
    from .outcome import ToolOutcome
    action = str(action or kwargs.get("operation") or "view").strip().lower()
    scope = scope_key(context)
    if not scope or _path_for(context, scope) is None:
        return ToolOutcome.rejected("CHECKLIST ERROR: no place to keep a checklist in this context.",
                                    reason_code="checklist_unavailable")
    data = load(context, scope)
    lst: List[Dict[str, Any]] = list(data.get("items") or [])

    def _next_id() -> int:
        return max([int(i.get("id") or 0) for i in lst] + [0]) + 1

    if action in ("set", "add"):
        new, why = _coerce_items(items)
        if why:
            return ToolOutcome.rejected(f"CHECKLIST ERROR: {why}.", reason_code="checklist_bad_items")
        if action == "set":
            old = {(i["title"], i["check"]): i for i in lst}
            lst = []
            for n in new:
                kept = old.get((n["title"], n["check"]))
                lst.append({"id": len(lst) + 1, "title": n["title"], "check": n["check"],
                            "status": (kept or {}).get("status", "open"), "last": (kept or {}).get("last")})
        else:
            for n in new:
                lst.append({"id": _next_id(), "title": n["title"], "check": n["check"], "status": "open",
                            "last": None})
        if len(lst) > MAX_ITEMS:
            return ToolOutcome.rejected(f"CHECKLIST ERROR: at most {MAX_ITEMS} items — group small features.",
                                        reason_code="checklist_too_long")
        data["items"] = lst
        _save(context, scope, data)
        return ("Checklist saved. Build ONE item at a time, then `checklist check` — only a passing check "
                "ticks an item.\n" + render(data))
    if action == "check":
        if not lst:
            return ToolOutcome.rejected("CHECKLIST ERROR: the checklist is empty — `checklist set` first.",
                                        reason_code="checklist_empty")
        want = set(_ids(ids))
        chosen = [i for i in lst if not want or int(i["id"]) in want]
        if not chosen:
            return ToolOutcome.rejected(f"CHECKLIST ERROR: no item with id {sorted(want)}.",
                                        reason_code="checklist_bad_ids")
        started = time.monotonic()
        ran_ids = set()
        for it in chosen:
            if time.monotonic() - started > CHECK_BUDGET_S:
                break                                   # the rest keep their state; `check` them next
            ran_ids.add(it["id"])
            try:
                ok, out = await _run_one(it["check"], run_execute=run_execute, page_probe=page_probe,
                                         sandbox_dir=sandbox_dir)
            except Exception as e:  # noqa: BLE001 — a check that cannot run has not passed
                ok, out = False, f"the check could not run: {type(e).__name__}: {e}"
            it["status"] = "passed" if ok else "failed"
            it["last"] = {"at": time.time(), "out": out}
        data["items"] = lst
        _save(context, scope, data)
        left = [i["id"] for i in chosen if i["id"] not in ran_ids]
        note = (f"\n(Time budget of {CHECK_BUDGET_S // 60} min for one check call used up — items {left} were not run; "
                f"`checklist check` them next.)") if left else ""
        return render(data, with_output=True, ran_ids=ran_ids) + note
    if action == "drop":
        gone = set(_ids(ids))
        if not gone:
            return ToolOutcome.rejected("CHECKLIST ERROR: drop needs ids.", reason_code="checklist_bad_ids")
        have = {int(i["id"]) for i in lst}
        if not gone & have:
            return ToolOutcome.rejected(f"CHECKLIST ERROR: no item with id {sorted(gone)} — the list has "
                                        f"{sorted(have) or 'no items'}.", reason_code="checklist_bad_ids")
        data["items"] = [i for i in lst if int(i["id"]) not in gone]
        _save(context, scope, data)
        missing = sorted(gone - have)
        return (f"Dropped {sorted(gone & have)}." + (f" (no item {missing})" if missing else "") + "\n"
                + render(data))
    if action == "clear":
        data["items"] = []
        _save(context, scope, data)
        return "Checklist cleared."
    if action == "view":
        return render(data, with_output=True)
    return ToolOutcome.rejected("CHECKLIST ERROR: action must be set, add, check, drop, view or clear.",
                                reason_code="checklist_bad_action")


#: written files that are not code: they never make a passed item stale
_NOT_CODE_EXTS = (".md", ".markdown", ".txt", ".rst", ".log", ".csv", ".tsv", ".png", ".jpg", ".jpeg", ".gif",
                  ".webp", ".ico", ".pdf", ".docx", ".xlsx", ".pptx", ".mp3", ".wav", ".mp4", ".zip", ".tar", ".gz")
_RAN_ROW_RE = re.compile(r"^(?:✓|✗) #(\d+) .* — check: .*$", re.M)


def ran_ids_in(text) -> set:
    """The item ids a `checklist check` result says it RAN (rows printed with their old state are skipped)."""
    return {int(m.group(1)) for m in _RAN_ROW_RE.finditer(str(text or "")) if not m.group(0).endswith(NOT_RUN_NOW)}


def exempt_ids_after(writer: str, i: int, metas, results) -> set:
    """The item ids a `check` call of the same batch RAN after the write at index `i`. A file_system write runs
    in the batch's first phase, before EVERY checklist call; an execute runs before the checklist calls that come
    LATER in the batch (they wait for it). `metas[j]` is the batch's tool_call_metadata row (name first, the
    call's arguments 10th); `results[j]` that call's result text. Only `check` (or its `operation=` spelling)
    runs items — view/set/add print rows too."""
    later = range(len(metas)) if writer == "file_system" else range(i + 1, len(metas))
    out = set()
    for j in later:
        m = metas[j]
        if not m or m[0] != "checklist" or j >= len(results):
            continue
        a = (m[9] if len(m) > 9 else None) or {}
        if str(a.get("action") or a.get("operation") or "").strip().lower() == "check":
            out |= ran_ids_in(str(results[j] or ""))
    return out


def note_code_changed(context, written, exempt_ids=()) -> int:
    """§4UU review: code was written after the checks ran — every PASSED item must be checked again (which
    feature a file serves is unknown, so all of them). Docs, data tables and images do not count.
    `exempt_ids`: items a checklist call of the SAME batch ran after this write (the agent knows the batch
    order) — they stay as that call left them. Returns how many items went stale."""
    code = [w for w in (written or []) if not str(w).lower().endswith(_NOT_CODE_EXTS)]
    if not code:
        return 0
    scope = scope_key(context)
    p = _path_for(context, scope)
    if p is None or not p.is_file():
        return 0
    exempt = {int(x) for x in (exempt_ids or ())}
    data = load(context, scope)
    n = 0
    for it in data.get("items") or []:
        if it.get("status") != "passed" or int(it.get("id") or 0) in exempt:
            continue
        it["status"] = "stale"
        n += 1
    if n:
        _save(context, scope, data)
    return n


# ── the end-of-turn nudge ─────────────────────────────────────────────

#: the reply presents the work as finished
_DONE_RE = re.compile(
    r"\b(?:(?:it'?s|it is|everything(?:'s| is)?|all|the (?:app|game|site|page|build|project)(?: is)?)\s+"
    r"(?:now\s+)?(?:done|complete|finished|ready|live|working)|all (?:features|items|tasks) (?:are )?"
    r"(?:done|complete|implemented|working)|(?:fully|now) (?:implemented|complete|working)|"
    r"(?:is|are) (?:now )?(?:live|deployed|implemented|ready)|built and (?:tested|working|deployed)|"
    r"I(?:'ve| have)? (?:now )?(?:finished|completed)(?: (?:the|it|everything|all|your)\b)?|"
    r"I(?:'ve| have)? (?:now )?(?:built|implemented) (?:it|everything|all of it|the (?:whole|full|complete))\b|"
    r"^\W*(?:all )?done\b)\b", re.I | re.M)
#: the reply itself says something is NOT finished — nothing to repair in its wording
_ADMITS_RE = re.compile(
    r"\bnot (?:all|every|yet|fully|working|finished|done|complete)|\bexcept\b|\bstill (?:fails?|failing|broken|"
    r"red|missing|not)|\b(?:does ?n[o']t|do ?n[o']t|is ?n[o']t|are ?n[o']t|cannot|can't) (?:yet )?(?:work|pass|run)|"
    r"\buntested\b|\bremaining\b|\bfail(?:s|ing|ed)\b", re.I)


#: "no remaining issues", "0 failed", "nothing failing", "X failed, I fixed it" — these say it IS finished
_DENIED_ADMIT_RE = re.compile(
    r"\b(?:no|zero|0|none|nothing)\s+(?:\w+\s+){0,2}(?:remaining|failing|failures?|failed|issues?|errors?|broken)\b"
    r"|\b0 failed\b|\bfail(?:s|ed|ing)?\b[^.;\n]{0,40}\b(?:I|we) (?:fixed|corrected|resolved)\b", re.I)
#: a progress report, not a finish: "Next I'll …", "so far", "step 1 of 3", "ready for you to add …"
_PROGRESS_RE = re.compile(
    r"\b(?:next|now),? (?:I'?ll|I will|I'm going to|we'?ll) (?:build|add|implement|write|create|start|work|"
    r"fix|finish|make|do|tackle|move|wire|handle)\b|\bnext (?:is|up|step)\b|\bnow moving on\b|"
    r"\b(?:done|finished|completed|built)\s+so far\b|\bstep \d+ (?:of|/) \d+|"
    r"\bready for (?:you to (?:add|build|implement|write|extend)|the next (?:step|part|feature|phase))\b", re.I | re.M)


def presents_as_finished(reply, claims_code_works=None) -> bool:
    """Does the reply present the work as finished (and not say, itself, what is still open)?"""
    text = str(reply or "")
    if not (_DONE_RE.search(text) or (claims_code_works is not None and claims_code_works(text))):
        return False
    if _PROGRESS_RE.search(text):
        return False
    return not _ADMITS_RE.search(_DENIED_ADMIT_RE.sub(" ", text))


def checklist_directive(reply: str, tools_run, context) -> str:
    """The repair directive for a reply that presents the work as finished while items THIS request worked on
    (`checklist` was called) have not passed — "" otherwise."""
    touched = any(isinstance(t, dict) and not t.get("_synthetic")
                  and str(t.get("name", "")).lower() == "checklist" for t in (tools_run or []))
    if not touched:
        return ""
    from ..core.code_claims import claims_code_works
    if not presents_as_finished(reply, claims_code_works):
        return ""
    data = load(context)
    items = data.get("items") or []
    bad = [i for i in items if i.get("status") != "passed"]
    if not bad:
        return ""
    _why = {"failed": "failed", "stale": "code changed since it passed"}
    names = "; ".join(f"#{i['id']} {i['title']} ({_why.get(i.get('status'), 'not checked')})" for i in bad[:5])
    return (f"SYSTEM ALERT — your answer presents the work as finished, but {len(bad)} of {len(items)} checklist "
            f"item(s) have not passed: {names}. Run `checklist check`, fix what fails and check again; if an item "
            f"cannot be done, say so plainly in your answer (or `checklist drop` it and say why in your answer). THEN give "
            f"your final answer.")


CHECKLIST_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "checklist",
        "description": (
            "Your plan for a build with several features — one item per feature, each with a CHECK that proves it "
            "works. Only the check ticks an item: there is no 'mark done'. Use it for any app/site/tool with more "
            "than ~2 features: `set` the items first, build ONE item, run `check`, fix until it passes, then the "
            "next. While items are open you see `[CHECKLIST n/m passed · next: …]` after your tool calls.\n"
            "Actions: set (items; replaces the list, unchanged items keep their state) · add (items) · "
            "check (ids, or none = all: RUNS each check and records passed/failed with its output) · "
            "drop (ids) · view · clear.\n"
            "A check is one of: a shell command run in the sandbox — exit 0 = passed (e.g. "
            "`cd tetris && python3 tests/check_page.py rotate`, `python3 -m pytest -q tests/test_api.py`); "
            "`page:<file.html>` — the page loads headless with no uncaught JS error; `file:<path>` — exists and is "
            "not empty. A check must exercise the feature: `node --check`, `ls`, `cat` or `grep` prove nothing. "
            "The list belongs to the project you work in (else this conversation)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "add", "check", "drop", "view", "clear"]},
                "items": {
                    "type": "array",
                    "description": "For set / add: the items, in build order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "The feature, in a few words."},
                            "check": {"type": "string", "description": "Shell command (exit 0 = works), "
                                                                       "page:<file.html> or file:<path>."},
                        },
                        "required": ["title", "check"],
                    },
                },
                "ids": {"type": "array", "items": {"type": "integer"},
                        "description": "For check / drop: item numbers (check with none = every item)."},
            },
            "required": ["action"],
        },
    },
}
