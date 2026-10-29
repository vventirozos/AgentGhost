"""§4PH — the claim-keyed "did this turn RUN what it wrote?" arm.

The one mechanical guard between a coding turn and a false "it works" used
to be `_is_unverified_mutation(last_tool)`: it fired only when the turn's
LAST action was a successful whole-file write. A trailing `ls`, a read-back
of the file, `python --version` — anything — cleared it, and no arm anywhere
read the reply's own claim (§4PH readers 1 and 3: write 3,000 chars →
`execute ls` → "tested, it works" shipped clean; a file whose only run
FAILED shipped under "all tests pass"). The corpus agreed: 27 of 82
works/tested/verified replies ran nothing, and 1 of 158 coding turns ran a
test runner.

This module answers the question from the turn's tool records instead of
its last action:

* `code_evidence(tools_run)` — for every CODE file the request wrote (a
  `file_system` write/edit/append confirmation, OR a shell heredoc /
  redirect / `tee` through `execute`, OR `execute(filename=, content=)`),
  the LAST write and the LAST run after it: an `execute` that names the file
  (command, filename, args or an ephemeral script's text / imports), a test
  suite run (`is_suite_command`) for non-web code, or a browser load / vision
  check that names a web page. ORDER matters: a run before the final edit
  tested an earlier version.
* `unbacked_code_writes(tools_run)` → `(never_run, failed_last)`.
* `claims_code_works(reply)` — the reply affirms works / tested / verified /
  passes (negation-guarded: "I could not test", "untested", "should work" are
  not claims).
* `unbacked_claim_note(reply, tools_run, forbids_running)` — the note and
  failed-label for a reply that makes such a claim about a file with no
  backing run, ONE wording for both delivery paths (as §4LK required of the
  last-tool note). The operator's 2026-09-23 decision (no footer on a correct
  SMALL write; targeted edits exempt) is about a write with no claim — a
  reply that affirmatively says "tested" about a file nothing ran is a false
  statement at any size, so this arm has no size or shape exemption.

Pure functions over the turn's tool records; no LLM, no I/O.
"""
from __future__ import annotations

import os
import re
from typing import Optional

#: a file the request can RUN (or render) — the claim arm cares about these
CODE_EXTS = (".py", ".sh", ".bash", ".js", ".mjs", ".cjs", ".ts", ".go", ".rs",
             ".rb", ".php", ".pl", ".c", ".cpp", ".java", ".html", ".htm")
WEB_EXTS = (".html", ".htm")
_CODE_NAME_RE = re.compile(
    r"[\w.\-/]+\.(?:py|sh|bash|js|mjs|cjs|ts|go|rs|rb|php|pl|c|cpp|java|html?)\b", re.I)
#: `cat > app.py <<'EOF'`, `cat <<EOF > app.py`, `tee app.py`, `printf … > x.sh`
_SHELL_WRITE_RE = re.compile(
    r"(?:(?<![<>])>{1,2}\s*|\btee\s+(?:-a\s+)?)['\"]?([\w.\-/]+\.(?:py|sh|bash|js|mjs|cjs|ts|go|rs|rb|php|pl|c|cpp|java|html?))\b",
    re.I)
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)
#: r3 (reader M2): the common "run it" shapes that name no `x.py` — `python -m
#: app`, `python -c 'import calc; …'`, `uvicorn main:app`, `gunicorn app:app`
_IMPORT_ANY_RE = re.compile(r"(?<![\w.])(?:from\s+([\w.]+)\s+import\b|import\s+([\w.]+))")
_MODULE_RUN_RE = re.compile(
    r"(?<![\w-])-m\s+([\w.]+)"
    r"|\b(?:uvicorn|gunicorn|hypercorn|daphne|waitress-serve)\s+(?:-{1,2}\S+(?:\s+\S+)?\s+)*([\w.]+):\w+")
_NOT_A_MODULE = ("pytest", "unittest", "pip", "venv", "http.server", "json.tool", "nose", "nose2", "tox", "coverage")

#: the reply affirms the code works / was tested. Present-tense "works",
#: "tested", "verified", "passes" — not "should work" / "will work".
#: r3 (reader M1): "This works BY splitting…" / "a regex which works ON both" are
#: mechanism prose, not claims — the subject list drops `that|which` and a
#: mechanism preposition after `works` disqualifies; "I ran the tests and 2
#: of 5 failed" is honest reporting — an `I ran/tested X` needs an OUTCOME
#: word to be a claim ("I ran it — works", "I tested it successfully").
_CLAIM_RE = re.compile(
    r"\b(?:it|this|everything|the\s+(?:script|code|app|page|site|server|build|"
    r"function|program|file|fix|change|patch|module|tests?|suite|game|tool|cli|bot)s?)\s+"
    r"(?:now\s+|all\s+)?(?:works|is\s+working|runs\s+(?:fine|correctly|cleanly|without\s+errors?)"
    r"|passes|is\s+(?:tested|verified|functional|working|passing))\b"
    r"(?!\s+(?:by|like|as\s+follows|because|via|through|with|on|in|at|when|once|if|only|after|before|until)\b)"
    r"|\b(?:tested|verified)\s*(?:and|,|—|-|:)?\s*(?:it\s+)?(?:works|working|passes|passing)\b"
    r"|\ball\s+tests?\s+(?:pass|passed|passing|are\s+passing|succeed)\b"
    r"|\btests?\s+(?:pass|passed|are\s+(?:passing|green)|all\s+pass)\b"
    r"|\b(?:fully|successfully|thoroughly)\s+(?:tested|verified|working|functional)\b"
    r"|\bI\s+(?:have\s+|also\s+)?(?:tested|verified|ran|run|executed)\s+(?:it|this|that|the\s+\w+)\s*"
    r"(?:[—,:\-]|and|;)?\s*(?:it\s+)?(?:works|passes|passed|successfully|fine|ok|without\s+errors?|with\s+no\s+errors?|clean(?:ly)?)\b"
    r"|\bworks\s+(?:as\s+expected|correctly|fine|perfectly|end[- ]to[- ]end)\b"
    r"|\bworking\s+(?:correctly|as\s+expected|fine|now|end[- ]to[- ]end)\b"
    r"|\bconfirmed\s+(?:working|that\s+it\s+works|it\s+works)\b"
    r"|\b(?:is|are)\s+(?:tested|verified)\s+and\s+(?:working|functional)\b"
    r"|\bδουλε[υύ]ει\b|\bτο\s+(?:δοκ[ιί]μασα|τ[εέ]σταρα|[εέ]τρεξα)\b|\bπερν[αά](?:ει|νε)\s+(?:[οό]λα\s+)?τα\s+τεστ\b",
    re.I)
#: a claim governed by a negation / a conditional / a future is not a claim
_NEGATION_RE = re.compile(
    r"\b(?:not|n[’']t|never|cannot|can[’']t|unable|without|untested|unverified|no|none|"
    r"should|would|could|might|may|will|if|once|until|unless|after|when|before|assuming|"
    r"haven[’']t|hasn[’']t|didn[’']t|wasn[’']t|isn[’']t|aren[’']t|couldn[’']t|how|"
    r"δεν|μην|χωρις|χωρίς|αν|θα|πως|πώς)\b|\bun(?=tested|verified)", re.I)


def claims_code_works(text: str) -> str:
    """The first affirmative works/tested/verified claim in `text`, or ""."""
    s = str(text or "")
    for m in _CLAIM_RE.finditer(s):
        before = s[max(0, m.start() - 45):m.start()]
        # a negation / conditional in the same clause governs the claim
        _clause = re.split(r"[.;:!?\n]", before)[-1]
        if _NEGATION_RE.search(_clause):
            continue
        return m.group(0)
    return ""


# ── tool-record readers ──────────────────────────────────────────────

def _args_of(tool: dict) -> dict:
    c = tool.get("content")
    a = getattr(c, "call_args", None)
    if isinstance(a, dict):
        return a
    for k in ("call_args", "args", "arguments"):
        v = tool.get(k)
        if isinstance(v, dict):
            return v
    return {}


def _name(tool: dict) -> str:
    return str(tool.get("name", "")).lower().replace("-", "_").replace(" ", "_")


def _key(path) -> str:
    bn = os.path.basename(str(path or "").replace("\\", "/").rstrip("/"))
    return bn.casefold()


def _is_code(path) -> bool:
    return str(path or "").casefold().endswith(CODE_EXTS)


def _is_web(path) -> bool:
    return str(path or "").casefold().endswith(WEB_EXTS)


def _exec_ok(tool: dict) -> Optional[bool]:
    """True/False for a completed `execute`; None for a detached (promoted)
    job or an unreadable result — neither a pass nor a failure."""
    c = tool.get("content")
    text = str(c or "")
    try:
        from ..sandbox.jobs import is_promoted_result
        if is_promoted_result(text):
            return None
    except Exception:  # noqa: BLE001
        pass
    st = str(getattr(getattr(c, "status", None), "value", "") or "")
    if st in ("failed", "rejected", "partial"):
        return False
    if st == "unresolved":
        return None
    try:
        from ..utils.tool_stop import is_user_interrupted
        if is_user_interrupted(text):        # a stored row: its status is gone
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..tools.tool_failure import exec_exit_code, result_is_failure
        code = exec_exit_code(text)
        if code is not None:
            return code == 0
        return not result_is_failure(text)
    except Exception:  # noqa: BLE001
        return st == "ok"


def _fs_written(tool: dict) -> list:
    """Paths a file_system SUCCESS confirmation says were written/edited."""
    c = tool.get("content")
    raw = str(c or "").lstrip()
    if raw.startswith("[FAILURE BANNER]"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
    if not raw.lstrip().upper().startswith("SUCCESS"):
        return []
    from .agent import _written_paths_from_confirmation
    return [p for p in _written_paths_from_confirmation(raw) if _is_code(p)]


def _exec_written(tool: dict, args: dict) -> list:
    """Code files an `execute` call itself wrote: a heredoc/redirect/tee in
    the command (§4PH R3-4 — a route execute's own text offers), or the
    `filename` of an `execute(filename=, content=)` call (written, then run)."""
    out = []
    if _exec_ok(tool) is None:
        return out
    cmd = str(args.get("command") or "")
    for m in _SHELL_WRITE_RE.finditer(cmd):
        out.append(m.group(1))
    fn = str(args.get("filename") or "")
    if fn and args.get("content") and _is_code(fn):
        out.append(fn)
    return out


def _exec_run_keys(tool: dict, args: dict) -> tuple:
    """(keys the run exercised, is_suite). Keys are basenames (casefold) of
    the code files the command / filename / args / script text name, plus
    `x.py` for an `import x` in an ephemeral script."""
    from .agent import is_suite_command
    # a heredoc / redirect TARGET is a write, not a run: `cat > app.py <<EOF`
    # names app.py without running it (the same command may run it later:
    # `…EOF\npython app.py`)
    cmd = _SHELL_WRITE_RE.sub(" ", str(args.get("command") or ""))
    fn = str(args.get("filename") or "")
    content = str(args.get("content") or "")[:6000]
    extra = " ".join(str(a) for a in (args.get("args") or []) if a is not None)
    keys = set()
    for blob in (cmd, fn, extra, content):
        for m in _CODE_NAME_RE.findall(blob):
            keys.add(_key(m))
    for blob in (cmd, content):
        for a, b in _IMPORT_ANY_RE.findall(blob):
            mod = (a or b).split(".")[0]
            if mod:
                keys.add(f"{mod}.py".casefold())
    for a, b in _MODULE_RUN_RE.findall(cmd):
        mod = (a or b)
        if mod and mod.lower() not in _NOT_A_MODULE:
            keys.add(f"{mod.split('.')[-1]}.py".casefold())
    return keys, bool(is_suite_command(cmd) or is_suite_command(content))


def _browser_keys(tool: dict, args: dict) -> set:
    """Web pages a browser / vision call named (url, path, target)."""
    keys = set()
    for k in ("url", "path", "target", "file", "page", "image", "out_path"):
        v = args.get(k)
        if isinstance(v, str) and v:
            v = v.split("?", 1)[0].split("#", 1)[0]
            if re.match(r"(?i)^(?:https?|file)://", v):
                # r3 (reader M2): a served ROOT names the index page
                from urllib.parse import urlparse
                p = urlparse(v).path or ""
                if p in ("", "/") or p.endswith("/"):
                    keys.add("index.html")
                    continue
                v = p
            if _is_web(v):
                keys.add(_key(v))
    return keys


def _browser_ok(tool: dict) -> Optional[bool]:
    """True/False for a page load; None for one that did not finish (§4RZ
    step 5: the user stopped it, or it never ran) — not a run at all."""
    c = tool.get("content")
    st = str(getattr(getattr(c, "status", None), "value", "") or "")
    if st in ("failed", "rejected"):
        return False
    text = str(c or "")
    if st == "unresolved":
        return None
    try:
        from ..utils.tool_stop import is_user_interrupted
        if is_user_interrupted(text):        # a stored row: its status is gone
            return None
    except Exception:  # noqa: BLE001
        pass
    m = re.match(r"\A\s*--- BROWSER RESULT ---[ \t]*\r?\nSTATUS:[ \t]*([A-Z]+)", text)
    if m:
        return m.group(1) == "OK"
    return not text.lstrip().lower().startswith(("error", "[failure banner]"))


# ── the ledger ───────────────────────────────────────────────────────

def code_evidence(tools_run: Optional[list]) -> dict:
    """key → {path, write_idx, run_idx, run_ok, via}. `run_idx` is the last
    run AFTER the last write (None when nothing ran the final version);
    `run_ok` is that run's verdict (None = detached / unknown)."""
    led: dict = {}
    for i, t in enumerate(tools_run or []):
        if not isinstance(t, dict) or t.get("_synthetic"):
            continue
        n = _name(t)
        args = _args_of(t)
        if n in ("file_system", "filesystem", "file"):
            for p in _fs_written(t):
                led[_key(p)] = {"path": p, "write_idx": i, "run_idx": None,
                                "run_ok": None, "via": "file_system"}
            continue
        if n == "execute":
            ok = _exec_ok(t)
            for p in _exec_written(t, args):
                led[_key(p)] = {"path": p, "write_idx": i, "run_idx": None,
                                "run_ok": None, "via": "execute"}
            if ok is None:
                continue
            keys, suite = _exec_run_keys(t, args)
            # every record here was written at an index <= i (a later write
            # REPLACES the record, which is how order is kept)
            for k, rec in led.items():
                if k in keys or (suite and not _is_web(rec["path"])):
                    rec["run_idx"], rec["run_ok"] = i, ok
            # `execute(filename=x, content=…)` writes AND runs x in one call
            fn = str(args.get("filename") or "")
            if fn and args.get("content") and _key(fn) in led and led[_key(fn)]["write_idx"] == i:
                led[_key(fn)]["run_idx"], led[_key(fn)]["run_ok"] = i, ok
            continue
        if n in ("browser", "vision_analysis"):
            keys = _browser_keys(t, args)
            if not keys:
                continue
            ok = _browser_ok(t)
            if ok is None:
                continue
            for k, rec in led.items():
                if k in keys:
                    rec["run_idx"], rec["run_ok"] = i, ok
    return led


def unbacked_code_writes(tools_run: Optional[list]) -> tuple:
    """(never_run, failed_last): code files written this request whose final
    version nothing ran / loaded, and those whose last run after the final
    write FAILED. Paths in first-write order."""
    led = code_evidence(tools_run)
    never, failed = [], []
    for rec in sorted(led.values(), key=lambda r: r["write_idx"]):
        if rec["run_idx"] is None:
            never.append(rec["path"])
        elif rec["run_ok"] is False:
            failed.append(rec["path"])
    return never, failed


def _names(paths: list, limit: int = 3) -> str:
    shown = [f"`{os.path.basename(str(p))}`" for p in paths[:limit]]
    more = len(paths) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


UNBACKED_CLAIM_REASON = "unbacked code claim: the reply says it works/was tested but the file was never run (or its last run failed)"


def unbacked_claim(reply: str, tools_run: Optional[list]) -> tuple:
    """(claim, never_run, failed_last) when `reply` affirms the code works /
    was tested AND a code file written this request has no backing run —
    ("", [], []) otherwise."""
    claim = claims_code_works(reply)
    if not claim:
        return "", [], []
    never, failed = unbacked_code_writes(tools_run)
    if not never and not failed:
        return "", [], []
    return claim, never, failed


def unbacked_claim_note(reply: str, tools_run: Optional[list],
                        forbids_running: bool = False) -> tuple:
    """(note, failed) for a reply whose works/tested claim has no backing run
    — ("", False) when the claim is backed or absent. ONE wording for both
    delivery paths. A request that said "don't run it" is said, not failed
    (§4KW) — but the claim is still named as unbacked."""
    claim, never, failed = unbacked_claim(reply, tools_run)
    if not claim:
        return "", False
    if failed:
        what = (f"the last run of {_names(failed)} after it was written FAILED"
                + (f", and {_names(never)} was never run" if never else ""))
        verdict = "Treat it as BROKEN until a run passes."
    else:
        what = f"{_names(never)} was written this turn and never run"
        verdict = "Treat it as UNTESTED — run it before relying on it."
    if forbids_running:
        return (f"\n\n---\n*Not run, as you asked — but note the reply's \"{claim}\" is not "
                f"backed by a run this turn: {what}.*"), False
    return (f"\n\n---\n**⚠ Unverified claim:** this reply says \"{claim}\", but {what}. "
            f"{verdict}"), True


def unbacked_claim_directive(reply: str, tools_run: Optional[list]) -> str:
    """The in-loop repair directive for an unbacked claim, or ""."""
    claim, never, failed = unbacked_claim(reply, tools_run)
    if not claim:
        return ""
    if failed:
        what = f"the last run of {_names(failed)} after you wrote it FAILED"
    else:
        what = f"{_names(never)} was written this turn and NEVER RUN"
    return (f"SYSTEM ALERT — your answer says \"{claim}\", but {what}. "
            f"Actually RUN it now: execute the file or its tests (for a web page, load it in "
            f"the browser) and read the result. If it fails, FIX it or say so plainly — do not "
            f"claim it works. THEN give your final answer, naming what you ran.")
