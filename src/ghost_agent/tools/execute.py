import asyncio
import os
from .outcome import ToolOutcome
import shlex
import sys
import re
import logging
import uuid
import base64
import datetime
import ast
import contextvars
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional
from ..sandbox.jobs import PROMOTED_RESULT_MARKER
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import sanitize_code
from .file_system import _get_safe_path


# Commands for which exit code 1 means "no matches found" — a successful
# query with an empty result — rather than a failure. grep & friends signal
# genuine errors (bad regex, unreadable file) with exit 2 + stderr output.
_EXIT1_MEANS_NO_MATCH = {"grep", "egrep", "fgrep", "zgrep", "rg", "pgrep"}

# Released-workspace shell guard (see call site in tool_execute).
# case-insensitively: the sandbox volume is APFS (third review)
_RELEASED_PROJ_RE = re.compile(r"projects/+(?:\./)*([0-9a-fA-F]{12})(?:/|\b)")


# Backgrounded-server guard (2026-07-30, §4G). A `… &`-detached process
# re-parents to the container's PID 1 and SURVIVES the exec — holding its
# port invisibly, outside the service registry's sight (the solar-sim
# incident: an autoadvance task ran `python3 server.py &`, and the stray
# holder of port 8102 cost three requests of thrash). Block only when BOTH
# signals fire — a detach (`&` background / setsid) AND a server-ish
# command — so short-lived background jobs and foreground servers (which
# die with the exec timeout anyway) pass untouched. False negatives are
# acceptable: the supervisor's reconcile/status names surviving orphans.
_BG_AMP_RE = re.compile(r"(?<![&>])&(?!\s*[&>])")
_SERVERISH_RE = re.compile(
    r"\bhttp\.server\b|\bserver\.py\b|\bflask\s+run\b|\buvicorn\b"
    r"|\bgunicorn\b|\bnode\s+\S*(?:server|app)\.js\b"
    r"|\bnpm\s+(?:start|run\s+(?:dev|serve|start))\b"
    r"|\byarn\s+(?:start|dev)\b|\bstreamlit\s+run\b"
    r"|\bmanage\.py\s+runserver\b"
    r"|\bphp\s+-S\b|--port[=\s]\d|\bhttpd\b|\bnginx\b",
    re.IGNORECASE)
_SH_C_PAYLOAD_RE = re.compile(
    # review B2-M4: eval too; B3-M1: a quoted / escaped / pathed shell (`'bash'`, `\\bash`, `/bin/sh`) or a `$VAR` head
    r"(?:(?:(?<![\w-])['\"\\]?(?:/[\w./-]*/)?(?:ba|z|da)?sh['\"]?|\$\{?[A-Za-z_]\w*\}?)\s+(?:-\S+\s+)*-[A-Za-z]*c|\beval)"   # B4-M1: `-lc`, `-ec`
    r"\s+(?:'([^']*)'|\"([^\"]*)\")")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_DAEMON_BLOCK_MSG = (
    "SYSTEM BLOCK: this command backgrounds a long-lived server "
    "('… &' / setsid) inside the sandbox — NOT executed. Detached "
    "processes survive the exec and hold their port INVISIBLY: the "
    "service registry cannot see, stop, or report them (this exact "
    "leak caused the port-8102 incident). Start it as a SUPERVISED "
    "service instead: run action='start' lifetime='server' "
    "where='workspace' name='<name>' "
    "command='<the same command without the & and without setsid/"
    "nohup>' — a free port is assigned automatically (exported as "
    "$PORT), logs are captured, and it keeps running across turns. "
    "A one-off smoke test that ends its own server in the same command passes: "
)


def _detached_server_signals(text: str) -> bool:
    return bool(_SERVERISH_RE.search(text)
                and (_BG_AMP_RE.search(text)
                     or re.search(r"\bsetsid\b", text)
                     or re.search(r"\bgunicorn\b[^;&|\n]*\s(?:--daemon|-D)(?=\s|$)", text)))   # §4UY C14


#: §4UY C7: the smoke-test forms the server guards point to — a background start the SAME command ends AFTER it
#: started it: `kill %N`, `kill $!`, `PID=$!; …; kill $PID` (any signal), `kill $(jobs -p)`, `pkill`/`killall`
#: naming the started program, a `trap '… kill …' EXIT`, or a start bounded by its own `timeout N` (it dies on
#: its own). Review r1: a kill BEFORE the start, a `kill 1234` naming nothing started, or a second server left
#: running beside a killed first one is no smoke test — EVERY backgrounded start must be covered.
_SMOKE_FORMS = ("`timeout 15 <server> & sleep 1; curl …`, `<server> & PID=$!; sleep 1; curl …; kill $PID`, "
                "`<server> & sleep 1; curl …; kill %1` or a `trap 'kill $!' EXIT` — every `&` start needs its own")
#: live probe 11-19: `cd d && server & P=$!; …; kill $P` backgrounds the whole LIST in a subshell — `$!` is that
#: subshell, the kill leaves the server running. A job kill ends a server only when the job IS its process.
_JOB_STEER = (" A `$!` / `%1` / `$(jobs -p)` kill is credited only for the forms proven to end the server: "
              "a plain `server &` (env / nohup / nice prefixes and redirects are fine) — with `cd dir` first on its "
              "own line (`cd dir; server & P=$!`) — or `(cd dir && exec server > log 2>&1) &`, or a pipeline with "
              "`kill %1`. Anything else — `cd dir && server &` or `(cd dir && server) &` without exec (its kill can "
              "leave the server running), `{ …; }`, `bash -c` / `eval`, a loop or function, `pkill -P` — is refused: "
              "use one of those forms or end it by name (`pkill -f '<server>'`)")
_DAEMON_BLOCK_MSG += _SMOKE_FORMS + "." + _JOB_STEER + "."

_TIMEOUT_BOUNDED_RE = re.compile(
    r"^\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:\(\s*)?timeout\s+(?:-\S+\s+(?:\S+\s+)?)*[\d.]+[smhd]?\s")


#: a kill whose signal ends nothing: `-0` (is it alive?), `-STOP`/`-TSTP`/`-CONT` (pause/resume), `-l`/`-L` (list)
_NON_ENDING_SIGNAL_RE = re.compile(
    r"^\s*(?:-(?:0|l|L|(?:SIG)?(?:STOP|TSTP|CONT|TTIN|TTOU|WINCH|CHLD|URG)|1[789]|2[0-3]|28)"
    r"|-[sn]\s+(?:0|(?:SIG)?(?:STOP|TSTP|CONT|TTIN|TTOU|WINCH|CHLD|URG)))(?=\s|$)", re.I)


def _background_starts(text: str) -> list:
    """[(start, end, segment)] of every `… &` start, in job order (one split — §4UY C19: a re-split of the
    prefix per `&` was quadratic)."""
    parts = re.split(r"(\n|;|&&|\|\||(?<![&>|])&(?![&>]))", text)            # `|&` is a pipe
    out, pos = [], 0
    for i in range(0, len(parts), 2):
        seg = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if sep == "&":
            out.append((pos, pos + len(seg), seg))
        pos += len(seg) + len(sep)
    return out


def _shape_view(text: str) -> str:
    """`text` with the same length, for reading its SHAPE: quoted spans (`'…'`, `"…"`, `$'…'`) and comments
    become `x`, a backslash-newline becomes two spaces (review C7: a quoted `;` / `|` / `(` / newline and a
    continued line were read as structure)."""
    out, i, n, q = list(text), 0, len(text), None
    while i < n:
        c = text[i]
        if q:
            if c == "\\" and q != "'" and i + 1 < n:
                out[i] = out[i + 1] = "x"
                i += 2
                continue
            if c == q:
                q = None
            else:
                out[i] = "x"                            # a quoted newline is no structure either
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            if text[i + 1] == "\n":
                out[i] = out[i + 1] = " "
            i += 2
            continue
        if c in "'\"":
            q = c
        elif c == "#" and (i == 0 or text[i - 1] in " \t\n;&|("):
            while i < n and text[i] != "\n":
                out[i] = "x"
                i += 1
            continue
        i += 1
    depth = 0                                         # a `)` closing nothing is a `case` pattern's end: a separator
    for k, ch in enumerate(out):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                out[k] = ";"
            else:
                depth -= 1
    return "".join(out)


def _job_text(view: str, end: int) -> str:
    """The whole job a `&` at `end` backgrounds, read on the shape view: back to the previous `;` / newline /
    lone `&` outside parentheses and braces. An and-or list `a && b &` is ONE job, and a newline after `&&`,
    `||` or `|` continues it."""
    depth, i = 0, end - 1
    while i >= 0:
        c = view[i]
        if c in ")}":
            depth += 1
        elif c in "({":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and c == "\n":
            if not re.search(r"(?:&&|\|\||\|)\s*$", view[max(0, i - 80):i]):
                break
        elif depth == 0 and (c == ";" or (c == "&" and view[i - 1:i] != "&" and view[i + 1:i + 2] != "&"
                                           and view[i - 1:i] not in (">", "<"))):
            break
        i -= 1
    return view[i + 1:end]


def _subshell_closes(view: str, ends) -> list:
    """For each `&` at `ends[k]`: where the `( … )` subshell it runs inside closes (len(view) in the top shell).
    `$!`, `%N`, `jobs` after that point belong to the outer shell, which never started it (`(srv &); kill $!`).
    One pass with a bracket stack (linear: §4UY C19)."""
    out, stack, k = [len(view)] * len(ends), [], 0
    pending = {}                                      # open-paren index → starts waiting for its close
    for i, c in enumerate(view):
        while k < len(ends) and ends[k] <= i:
            if stack and view[stack[-1]] == "(":
                pending.setdefault(stack[-1], []).append(k)
            k += 1
        if c in "({":
            stack.append(i)
        elif c in ")}" and stack:
            o = stack.pop()
            for w in pending.pop(o, ()):
                out[w] = i
    return out


#: a function the command defines: `name()` / `function name`
_FN_DEF_RE = re.compile(r"(?:^|[\s;&|(])(?:function\s+([\w.:-]+)|([\w.:-]+)\s*\(\s*\))")
#: heads whose job is NOT the program after them — or not proven to be in the sandbox: a nested shell / `eval`
#: (judging its payload is not trivial), reserved words, `time`/`sudo`/`setsid`/`xargs`/`watch`. Refuse when unsure.
_UNPROVEN_HEADS = frozenset("bash sh zsh dash ksh fish eval source . time sudo setsid xargs watch command builtin "
                            "if then elif else fi while until for do done case esac select function ! [[ coproc".split())
_SUBST_RE = re.compile(r"\$\(\([^()]*\)\)|\$\([^()]*\)|\$\{[^{}]*\}")       # `$((…))`, `$(…)`, `${…}`: words
_REDIR_AMP_RE = re.compile(r"\d*>&\d*-?|&>>?|<&\d*")
_GROUP_RE = re.compile(r"^([({])(.*)([)}])((?:\s*(?:\d*>>?|&>>?|\d*<)\s*&?\d*\s*[^\s()<>&|;]*)*)\s*$", re.S)


def _defined_functions(view: str) -> set:
    return {a or b for a, b in _FN_DEF_RE.findall(view)}


def _plain(cmd: str) -> str:
    """Redirect `&`s blanked and substitutions (`$(…)` nested, `` `…` ``, `${…}`) made one word `v`."""
    prev = None
    while prev != cmd:
        prev, cmd = cmd, _SUBST_RE.sub("v", cmd)
    return _REDIR_AMP_RE.sub(" ", re.sub(r"`[^`]*`", "v", cmd))


def _resolve_head(words: list):
    """THE head resolver (review B3-M1) — every head is judged here: the program a simple command runs, past
    `VAR=` words, `exec` (and its flags), `env` (its flags and `VAR=` words), `nice` (its flags), `nohup`, `--`,
    as a basename. None when unprovable: a quoted / escaped / substituted / `$VAR` head."""
    w, i, n = words, 0, len(words)                    # an index walk: linear (review: pop(0) was quadratic)
    while i < n:
        h = w[i]
        if re.match(r"^[A-Za-z_]\w*=", h) or h == "--":
            i += 1
            continue
        base = h.rsplit("/", 1)[-1]
        if base in ("exec", "env", "nice", "nohup"):
            i += 1
            while i < n and (w[i].startswith("-") and w[i] != "--"
                             or base == "env" and re.match(r"^[A-Za-z_]\w*=", w[i])):
                i += 1                                # a flag's argument then reads as the head: a program
            continue
        if re.search(r"[\\'\"$`]", h) or h == "v":
            return None
        return base
    return None


def _is_one_program(cmd: str, fns) -> bool:
    """PROVEN form (1): ONE simple command — `VAR=x` prefixes, `nohup` / `env` / `nice` / `exec` wrappers,
    redirects and substitutions in its ARGUMENTS allowed — whose resolved head is not a nested shell, a reserved
    word, a function the command defined, or unprovable."""
    c = _plain(cmd).strip()
    if not c or re.search(r"[(){}|;&\n]", c):
        return False
    head = _resolve_head(c.split())                   # `$B …` / `` `x` `` / `$(x)` heads resolve to None
    return bool(head) and head not in _UNPROVEN_HEADS and head not in fns


def _job_kind(job: str, fns) -> str:
    """What a job kill ends, for the job a `&` backgrounds (read on the shape view). REFUSE WHEN UNSURE (operator
    11-19): only forms PROVEN in the sandbox's bash 5.2.15 are credited —
    "own" (`$!`, `$VAR`, `%N`, `$(jobs -p)` end the server): (1) one simple command that is the server, bare or
    alone in `( … )`; (2) a job of MORE than one command (`a && …`, `( … )`, `{ … }`, redirects anywhere) only when
    its last command is `exec` server with no other server before it — holds in bash 3.2 and 5.2 alike;
    "pipe" (only `%N` / `$(jobs -p)`, which signal every process of it): (3) a bare pipeline of simple commands;
    "" for everything else: `a && srv`, `{ a; srv; }`, `||` lists, loops, functions, `bash -c` / `eval`.
    Known limit — kill bookkeeping is not traced: `P=1; kill $P`, `echo kill $P` are read as kills."""
    j = job.strip()
    g = _GROUP_RE.match(j)
    if g and (g.group(1), g.group(3)) in (("(", ")"), ("{", "}")):
        inner = _plain(g.group(2))
        if re.search(r"[(){}`]|(?<!&)&(?!&)", inner):
            return ""                                 # nested groups (also `(a) && (b)`), a `&` inside: not proven
        parts = [x.strip() for x in re.split(r"&&|;|\n", inner) if x.strip()]
        if not parts or any(_SERVERISH_RE.search(x) or x.startswith("trap") for x in parts[:-1]):
            return ""
        last = parts[-1]
        if g.group(1) == "(" and len(parts) == 1:    # `(srv …) &`: a single command, as bare
            return "own" if _is_one_program(re.sub(r"^exec\s+", "", last), fns) else ""
        # more than one command (or a `{ }` group): only an `exec`'d last command is the job's process — bash 5's
        # own exec of a subshell's last command is silently lost to a redirect on it (measured 11-19)
        return "own" if last.startswith("exec ") and _is_one_program(last[5:], fns) else ""
    p = _plain(j)
    if re.search(r"[(){}`]|\|\|", p):
        return ""
    if "&&" in p:
        parts = [x.strip() for x in p.split("&&")]
        if any(_SERVERISH_RE.search(x) for x in parts[:-1]):
            return ""
        return "own" if parts[-1].startswith("exec ") and _is_one_program(parts[-1][5:], fns) else ""
    if "|" in p:
        stages = [x for x in re.split(r"\|&?", j)]
        if any(_SERVERISH_RE.search(x) for x in stages[1:]):
            return ""                                 # review B3-M2: a job spec ends a pipeline only from its head
        return "pipe" if all(_is_one_program(x, fns) for x in stages) else ""
    return "own" if _is_one_program(j, fns) else ""


_HEREDOC_ANY_RE = re.compile(r"(<<-?\s*(['\"]?)(\w+)\2[^\n]*)\n.*?\n[ \t]*\3[ \t]*(?=\n|$)", re.S)


def _ends_its_background(cmd: str) -> bool:
    """Does `cmd` end EVERY server it backgrounds, after starting it (see `_SMOKE_FORMS`)?"""
    # review B2-M3: every heredoc body is data to this reading (an apostrophe in it would open a "quote")
    text = _HEREDOC_ANY_RE.sub(lambda m: m.group(1), str(cmd or ""))
    view = _shape_view(text)                          # quotes / comments / continuations read as shape only
    starts = [(a, e, text[a:e]) for a, e, _seg in _background_starts(view)]
    if not starts:
        return False
    covered = [bool(_TIMEOUT_BOUNDED_RE.match(seg)) for _s, _e, seg in starts]
    # a job kill (`$!`, `$VAR`, `%N`, `$(jobs -p)`) ends a start only when its job IS the process (live 11-19)
    jobs = [_job_text(view, e) for _s, e, _seg in starts]
    fns = _defined_functions(view)
    # review B4-M1, wrapper-proof: a job carrying a shell `-c` payload is never its server's process, whatever
    # wraps it (`stdbuf -oL bash -c`, `caffeinate -i bash -c`, `busybox sh -c`)
    kinds = ["" if _SH_C_PAYLOAD_RE.search(j) else _job_kind(j, fns) for j in jobs]
    closes = _subshell_closes(view, [e for _s, e, _seg in starts])
    own = [k == "own" for k in kinds]
    # a job SPEC (`%N`, `$(jobs -p)`) also ends a bare pipeline: bash signals every process of it (verified);
    # `$!` names only its last process
    spec = [k in ("own", "pipe") for k in kinds]

    import bisect as _bisect
    ends = [e for _s, e, _t in starts]                # ascending: starts are in text order

    def last_started(at: int) -> int:                 # index of the newest start before `at`, or -1
        return _bisect.bisect_right(ends, at) - 1

    pid_vars = {}                                     # VAR=$! → (the start it holds, where it was set)
    for m in re.finditer(r"\b([A-Za-z_]\w*)=[\"']?\$(?:!|\{!\})", text):
        pid_vars[m.group(1)] = (last_started(m.start()), m.start())
    traps = [(m.start(), m.end()) for m in
             re.finditer(r"\btrap\s+(['\"])(?:(?!\1).)*\1\s+(?:EXIT|INT|TERM|0)\b", text, re.S)]
    for m in re.finditer(r"\b(kill|pkill|killall)\b([^;&|\n]{0,160})", text):
        at = len(text) if any(a <= m.start() < b for a, b in traps) else m.start()   # a trap runs at the end
        n = last_started(at)
        if n < 0:
            continue                                  # a kill before any start ends nothing it started
        args = m.group(2)
        if _NON_ENDING_SIGNAL_RE.match(args):
            continue                                  # review r2 m3: `kill -0` probes, -STOP pauses, -l lists
        if m.group(1) == "kill":
            if m.start() > closes[n]:
                continue                              # outside the subshell that started it: not its job
            if re.search(r"\$\(\s*jobs\s+-p", args):
                covered[:n + 1] = [c or (spec[k] and m.start() < closes[k]) for k, c in enumerate(covered[:n + 1])]
            if re.search(r"\$!|\$\{!\}", args):
                covered[n] = covered[n] or own[n]
            for jm in re.finditer(r"%(\d+|[%+-]?)(?![\w])", args):
                g = jm.group(1)
                idx = int(g) - 1 if g.isdigit() else (n - 1 if g == "-" else n)
                if 0 <= idx <= n and m.start() < closes[idx]:
                    covered[idx] = covered[idx] or spec[idx]
            for var in re.findall(r"\$\{?([A-Za-z_]\w*)", args):
                held, set_at = pid_vars.get(var, (-1, len(text)))
                if 0 <= held and set_at < at and m.start() < closes[held]:
                    covered[held] = covered[held] or own[held]
        else:
            names = [w.strip("'\"") for w in args.split() if not w.startswith("-")]
            for idx in range(n + 1):
                if any(nm and nm in starts[idx][2] for nm in names):
                    covered[idx] = True
    # every server-shaped start must be covered (a `sleep 30 &` beside it is no server); with none, every start
    need = [i for i, (_s, _e, seg) in enumerate(starts) if _SERVERISH_RE.search(seg)] or range(len(starts))
    return all(covered[i] for i in need)


# ⚠ THE INSTALL VERB IS REQUIRED, and `uninstall` is excluded — my first
# version matched installer+package with no verb, so `pip uninstall
# openai-whisper` was refused. Blocking the CLEANUP of the very package this
# guard exists to prevent is the opposite of the intent, and the
# over-blocking test caught it.
def _daemonized_server_block(command: str):
    """Refusal message when a shell command would detach a long-lived
    server inside the sandbox, else None.

    Precision rails (review 2026-07-30, same class the egress guard hit on
    2026-07-08): heredoc BODIES are data being written, not commands —
    stripped via the shared ``_strip_data_heredocs``; quoted strings are
    data too (`git commit -m "fix server.py & restart"`), EXCEPT a nested
    shell's ``-c`` payload, which is itself a command and gets its own
    scan. A command that job-kills its background child (`… & curl …;
    kill %1` — the smoke-test idiom) does not outlive the exec and passes."""
    cmd = str(command or "")
    if not cmd:
        return None
    try:
        cmd = _strip_data_heredocs(cmd)
    except Exception:  # noqa: BLE001 — guard must not crash the tool
        pass
    if _ends_its_background(cmd):             # §4UY C7: the command ends what it started
        return None
    for m in _SH_C_PAYLOAD_RE.finditer(cmd):
        inner = m.group(1) or m.group(2) or ""
        if inner and _detached_server_signals(inner) and not _ends_its_background(inner):
            return _DAEMON_BLOCK_MSG
    if _detached_server_signals(_QUOTED_RE.sub(" ", cmd)):
        return _DAEMON_BLOCK_MSG
    return None


#: ft live (probe-75a957e4): `python3 -m http.server 8099` run in the FOREGROUND never exits, so the exec waited its
#: whole 600 s budget and was killed — ten minutes of a coding turn for nothing. Only commands that are servers by
#: definition (no exit of their own) are refused; a `timeout N` bound, a backgrounded/job-killed start (the §4G guard
#: owns those) and `--help`/`--version` pass.
_END = r"(?=\s|$|[;&|)])"
_FG_SERVER_RE = re.compile(
    r"^(?:sudo\s+(?:-[A-Za-z]\s+\S+\s+|-\S+\s+)*|env\s+(?:\S+=\S*\s+)*|nohup\s+|exec\s+|time\s+|nice\s+(?:-n\s*-?\d+\s+)?|[A-Za-z_]\w*=\S*\s+)*"
    r"(?:python[\d.]*\s+-m\s+(?:http\.server|SimpleHTTPServer|flask\s+run|uvicorn|gunicorn)" + _END +
    r"|python[\d.]*\s+-m\s+streamlit\s+run" + _END +
    # §4UY C14: `flask --app x run`, `fastapi dev|run`, `nodemon`; `gunicorn --daemon` detaches (the §4G guard's)
    r"|flask\s+(?:(?:--app|-A|--debug|--no-debug|-e|--env-file)(?:[\s=]\S+)?\s+)*run" + _END +
    r"|fastapi\s+(?:dev|run)" + _END + r"|(?:npx\s+)?nodemon" + _END +
    r"|uvicorn\s+(?!--version)|gunicorn\s+(?!--check-config|--version|(?:.*\s)?(?:--daemon|-D)(?:\s|$))|hypercorn\s"
    r"|(?:npm|pnpm|bun)\s+(?:(?:--prefix|-C)\s+\S+\s+)?(?:run(?:-script)?\s+)?(?:start|dev|serve|preview)" + _END +
    r"|pnpm\s+(?:start|dev|serve)" + _END + r"|yarn\s+(?:run\s+)?(?:start|dev|serve|preview)" + _END +
    r"|streamlit\s+run" + _END + r"|(?:python[\d.]*\s+)?manage\.py\s+runserver" + _END + r"|php\s+-S" + _END +
    r"|(?:npx\s+(?:-y\s+)?)?(?:serve|http-server|live-server)(?=\s|$)(?!\s+--?(?:h|help|v|version)\b)"
    r"|(?:npx\s+(?:-y\s+)?)?vite(?:\s+(?:dev|serve|preview))?" + _END + r"(?!\s*(?:build|optimize))"
    r"|(?:npx\s+)?next\s+(?:dev|start)" + _END +
    # §4UY C14: `jupyter notebook list|stop|password` and `--version` exit
    r"|jupyter\s+(?:notebook|lab)" + _END + r"(?!\s+(?:list|stop|password|--version|--help|-h)\b)" +
    r"|hugo\s+server" + _END + r"|rails\s+s(?:erver)?" + _END + r")", re.I)
_FG_SERVER_BLOCK_MSG = (
    "SYSTEM BLOCK: this command starts a server in the FOREGROUND — NOT executed. A server never exits on its own, so "
    "execute would wait its whole 600 s budget and then kill it. Start it with the `run` tool instead: "
    "run action='start' lifetime='server' where='workspace' name='<name>' command='<the same command>' (it gets a "
    "free port as $PORT and keeps running across turns), then request it with curl or the browser. For a one-off "
    "smoke test in ONE command, background it and end it yourself: " + _SMOKE_FORMS + "."
)


def _foreground_server_block(command: str):
    """Refusal message when a segment of the command runs a definite server in the foreground, else None."""
    cmd = str(command or "")
    if not cmd or (_ends_its_background(cmd) and re.search(r"[^&]&(?!&)", cmd)):
        return None
    # final review: EVERY heredoc body is data to this guard (a python body's `serve = …` is not a shell line)
    cmd = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n[ \t]*\2[ \t]*(?=\n|$)", "<<HEREDOC", cmd, flags=re.S)
    for m in _SH_C_PAYLOAD_RE.finditer(cmd):            # `bash -c 'python3 -m http.server'` is a command too
        inner = m.group(1) or m.group(2) or ""
        if inner and _foreground_server_block(inner):
            return _FG_SERVER_BLOCK_MSG
    cmd = _QUOTED_RE.sub("''", cmd)          # final review: quoted text (a commit message, a README) is data
    for seg in re.split(r"\n|;|&&|\|\|", cmd):
        seg = seg.strip().lstrip("( ").rstrip(") ").strip()
        # §4UY C14: a server inside `if … then` / `for … do` / `{ …; }` / `! …` is still a foreground server
        seg = re.sub(r"^(?:(?:then|do|else|elif|if|while|until|\{|!)\s+)+", "", seg).strip()
        if not seg or seg.endswith("&"):
            continue                          # backgrounded: the §4G guard owns it
        seg = seg.split("|", 1)[0].strip()    # `npm start | tee log` still never exits
        seg = re.sub(r"^(?:cd\s+\S+\s*)", "", seg)
        _usage = r"\s(?:--help|--version|-V)(?:\s|$)" if re.search(r"\bflask\s+run\b", seg) else \
            r"\s(?:--help|-h|--version|-V)(?:\s|$)"                    # flask's `-h` is --host
        if re.match(r"^(?:\S+=\S*\s+)*timeout\b", seg) or re.search(_usage, seg):
            continue                          # bounded by the model itself / only prints its usage
        if _FG_SERVER_RE.search(seg):
            return _FG_SERVER_BLOCK_MSG
    return None


#: §4PI (R3-M5): the model CAN `pip install yt-dlp && yt-dlp <url>` in the
#: sandbox (pip rides Tor) — without the agent's PO token every exit is
#: bot-walled and the attempt becomes a 600 s promoted job (the 2026-08-11
#: loop in sandbox/jobs.py). Refused, with the route it should use.
_YT_TOOL_RE = re.compile(r"(?<![\w./-])(?:yt-dlp|yt_dlp|youtube-dl|youtube_dl|pytube|pytubefix)(?![\w-])", re.I)
_YT_BLOCK_MSG = (
    "SYSTEM BLOCK: this command downloads from YouTube inside the sandbox (yt-dlp / youtube-dl / "
    "pytube) — NOT executed. The sandbox leaves only over Tor, and YouTube refuses every Tor exit "
    "without the agent's PO-token route ('Sign in to confirm you're not a bot' on every circuit, "
    "measured). Use the route instead: knowledge_base(action='transcribe', filename='<the YouTube "
    "URL>') fetches the captions or the audio over Tor, transcribes it on the audio node and indexes "
    "it with timestamps; read it in order with knowledge_base(action='transcript', filename=...)."
)


_YT_IMPORT_RE = re.compile(
    r"\b(?:import\s+(?:yt_dlp|pytube|pytubefix)\b|from\s+(?:yt_dlp|pytube|pytubefix)\s+import\b"
    r"|pip3?\s+(?:install|download)\s+[^\n;|&]*?(?:yt-dlp|yt_dlp|youtube-dl|pytube))", re.I)


def _youtube_download_block(command: str):
    cmd = str(command or "")
    try:
        cmd = _strip_data_heredocs(cmd)
    except Exception:  # noqa: BLE001
        pass
    # r2 (reader m2): the natural shapes live INSIDE quotes — `python -c
    # 'import yt_dlp'`, `pip install "yt-dlp"` — so the import/install forms
    # are read on the heredoc-stripped text; the bare tool names on the
    # quote-stripped text (a note that mentions yt-dlp is not a download)
    if _YT_IMPORT_RE.search(cmd):
        return _YT_BLOCK_MSG
    cmd = _QUOTED_RE.sub(" ", cmd)
    return _YT_BLOCK_MSG if _YT_TOOL_RE.search(cmd) else None


from .shell_analysis import _shell_segments, _bulk_destructive, _released_written_ids  # noqa: E402


def _keep_versions_before_run(command: str, sandbox_dir, container_workdir: str = "") -> int:
    """§4UY C6/C21 — the shell side of `file_system.keep_before_replace`: before a command runs, the previous
    version of every existing workspace file it would REMOVE (`rm`, `find -delete`, `mv` sources, `rsync
    --delete`, …) or OVERWRITE (`>`, `sed -i`, `cp` onto it, `tee`, …) is kept in the version store. The
    paths come from `shell_analysis` (quotes, comments and heredoc bodies set aside, nested shells read,
    `cd` followed). A directory is kept only when it is REMOVED (a build tool writing into `.` is not a
    removal). Returns how many files were kept; never raises, never blocks."""
    try:
        if not command or sandbox_dir is None:
            return 0
        from .file_versions import keep_previous
        from .shell_analysis import _destructive_targets, _written_paths, _norm_shell_arg
        sd = Path(sandbox_dir)
        root = sd.parent.parent if (sd.parent.name == "projects" and re.fullmatch(r"[0-9a-f]{12}", sd.name)) else sd
        cwd = str(container_workdir or "/workspace")
        removed, written = [], []

        def _host(p: str):
            p = _norm_shell_arg(p)
            if not p or "$" in p or "<unknown>" in p or p.startswith("~"):
                return None
            if not p.startswith("/") and not cwd:
                return None                       # relative to a directory nobody knows
            full = os.path.normpath(p if p.startswith("/") else cwd + "/" + p)
            if not (full == "/workspace" or full.startswith("/workspace/")):
                return None
            hp = root / full[len("/workspace"):].lstrip("/")
            return hp

        _prev_find = None                         # §4UY r3: the find a following `| xargs rm` reads
        for head, args, raw, via_xargs in _shell_segments(command):
            if head in ("cd", "pushd"):
                tgt = next((a for a in args if not a.startswith("-")), None)
                if tgt is None or "$" in tgt or tgt.startswith("~"):
                    cwd = ""
                elif cwd or tgt.startswith("/"):
                    cwd = os.path.normpath(tgt if tgt.startswith("/") else cwd + "/" + _norm_shell_arg(tgt))
                continue
            dt = _destructive_targets(head, args, via_xargs) or []
            _fstarts = set()
            _fargs, _frm = None, False
            if head == "find" and not dt and not via_xargs:
                _fargs = args             # §4UY r2 m4: a narrowed `find … -name '*.py' -delete`
            elif via_xargs and head in ("rm", "unlink", "shred", "srm", "mv") and _prev_find is not None:
                _fargs, _frm = _prev_find, True   # §4UY r3: `find … -print0 | xargs -0 rm` removes its matches
            _prev_find = args if (head == "find" and not via_xargs) else None
            if _fargs is not None:
                # keep what the find's filter can match (a superset of what it selects)
                from .shell_analysis import _find_kept_match
                for _start, _match in (_find_kept_match(_fargs, removes=_frm) or []):
                    _fstarts.add(_start)              # its start is filtered, not "written" whole below
                    hp = _host(_start)
                    if hp is not None and not any(c in str(hp) for c in "*?["):
                        removed.append((hp, _match))
            for t in dt:
                import glob as _glob
                hp = _host(t)
                if hp is None:
                    continue
                if any(c in str(hp) for c in "*?["):
                    removed += [Path(x) for x in _glob.glob(str(hp))[:500]]
                else:
                    removed.append(hp)
            for t in _written_paths(head, args, raw):
                if t in dt or t in _fstarts:
                    continue
                hp = _host(t)
                if hp is not None and hp.is_file():
                    written.append(hp)
        if not removed and not written:
            return 0
        kept, skipped = keep_previous(removed + written, root)
        if kept or skipped:
            # §4UY r1: a spent budget says "kept N of M"; the store is outside the sandbox (file_versions)
            pretty_log("File versions", f"{len(kept)} kept before the command ran"
                       + (f"; not kept: {'; '.join(skipped[-2:])}" if skipped else ""), icon=Icons.TOOL_FILE_W)
        return len(kept)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("GhostAgent").debug("keep versions before run: %s", e)
        return 0


def _released_shell_block(project_store, command: str, workdir: str = ""):
    """Refusal message when a shell command would mutate a RELEASED
    project's workspace, else None. Heuristic by necessity (shell), so it
    only fires on the conjunction: released-project path referenced AND a
    mutation token present. Never raises."""
    try:
        if not command:
            return None
        ids = {i.lower() for i in _RELEASED_PROJ_RE.findall(str(command))}
        try:
            _rel = [str(p.get("id") or "").lower()
                    for p in ((project_store.list_projects("RELEASED") if project_store is not None else None) or [])]
        except Exception:                                   # noqa: BLE001
            _rel = []
        # Fresh review: `rm -rf projects`, `rm -rf *`, `find . -delete` name
        # no project id and removed released workspaces with the rest. Fifth
        # review: refused ALWAYS — the projects folder holds every project's
        # workspace, released or not (with none released, `rm -rf projects`
        # passed).
        if _bulk_destructive(str(command), [r for r in _rel if r], workdir):
            _held = (f" — and RELEASED (immutable) project(s) {', '.join(str(r) for r in _rel[:5])}" if _rel else "")
            return ToolOutcome.rejected(
                f"SYSTEM BLOCK: this command removes or moves files in bulk (the projects folder, a wildcard at "
                f"the workspace root, the whole workspace, or everything in a project's own folder — "
                f"`rm -rf *` / `find . -delete` at its top){_held} — NOT executed. Name the files or folders to "
                f"remove explicitly (`rm -rf build dist`); a whole project is deleted with manage_projects "
                f"action='delete'.",
                reason_code="released_write_blocked")
        if project_store is None:
            return None
        # the working directory IS a released workspace (fourth review:
        # `manage_projects switch` there, then `rm -rf *`, `sed -i`, `> f`
        # all passed — the ids were read from the command text only)
        _wm = re.search(r"/projects/([0-9a-fA-F]{12})(?:/|$)", str(workdir or ""))
        if _wm:
            ids.add(_wm.group(1).lower())
        if not ids:
            return None
        written = _released_written_ids(str(command), ids, workdir)
        for pid in sorted(written):
            proj = project_store.get_project(pid)
            if proj and str(proj.get("status", "")).upper() == "RELEASED":
                from .projects import FORK_ONLY_IF_ASKED
                return (
                    ToolOutcome.rejected(f"SYSTEM BLOCK: project {pid} is RELEASED (immutable) and "
                    f"this command contains a mutation (rm/mv/redirect/"
                    f"sed -i/…) targeting its workspace — NOT executed. Fork "
                    f"a development copy first: manage_projects "
                    f"action=create_version project_id={pid}, then work in "
                    f"the new version's workspace. Read-only commands "
                    f"(cat/grep/ls) on released files are fine. {FORK_ONLY_IF_ASKED}"))
    except Exception:
        return None
    return None

# Execution budget for sandboxed runs (seconds). The sandbox layer wraps
# every run in `timeout -k 5s <budget>s`, so hitting the budget surfaces
# as exit 124 (or 137/143 when the -k SIGKILL / a SIGTERM landed).
# A run that is still ALIVE and still PROGRESSING at the budget is promoted
# to a background job instead of killed (sandbox/jobs.py); exit 124 now means
# "the budget expired AND it had gone quiet", i.e. genuinely wedged.
_EXEC_TIMEOUT_S = 600
_TIMEOUT_KILL_CODES = (124, 137, 143)

#: §4HZ — exit 137 is SIGKILL, and two hands send it: `timeout -k 5s` once
#: the budget expired, and the kernel OOM killer at any moment. The hint used
#: to name the timeout for both. Live (req 0e6cf008): a plot script died with
#: 137 after FOUR SECONDS, was told it "timed out / killed after 600s", and
#: the model spent four turns timing imports before guessing memory. A 137
#: that lands this many seconds before the budget cannot be the timeout.
_OOM_KILL_MARGIN_S = 30


def _kill_is_oom(exit_code, elapsed_s) -> bool:
    """True when an exit-137 kill happened too early to be the budget kill."""
    try:
        return (int(exit_code) == 137 and elapsed_s is not None
                and float(elapsed_s) < _EXEC_TIMEOUT_S - _OOM_KILL_MARGIN_S)
    except (TypeError, ValueError):
        return False


def _sandbox_mem_limit() -> str:
    """The container memory cap the OOM hint names (same env the sandbox
    layer reads at container creation — one truth, not a second constant)."""
    return os.environ.get("GHOST_SANDBOX_MEM", "4g")


#: §4UY C9/C10 — ONE exit-status rule: the REAL exit codes. The shell reports the status of the LAST command
#: of a list, so `python x.py; echo done`, `gcc …; ./a`, `pytest > out.log; tail -3 out.log` read as EXIT CODE 0
#: over a failing program. The old cover was output signatures (a Python traceback, `N failed`, `npm ERR!`) and
#: a printed `exit=N` — prose, wrong both ways. Now the shell itself records every top-level command that
#: failed: an ERR trap (bash fires it exactly for an unhandled failure — never for the left side of `&&`/`||`,
#: an `if`/`while` test or a `!`) appends `<code>:<stage codes>:<command>`, and an EXIT trap writes the list to
#: a STATUS FILE (`/workspace/.ghost_status/<id>`, read back and removed by the host — review r1: a marker on
#: stdout went into the user's own file under `exec > out.txt`). The run's status is then:
#:   * the shell's status when it is non-zero (pipe / find forgiveness as before);
#:   * else the LAST recorded failure of a program that RUNS CODE (`_RUNNER_HEAD_RE`: an interpreter, a test
#:     runner, a compiler or build tool, `./x`) — review r1 inverted the rule: `rm` of an absent file, `lsof`,
#:     `nc -z`, `cp 2>/dev/null`, `curl`, `pip` failing keep the shell's status, and so does a `timeout N`
#:     run that the timeout ended (124/137/143: a bounded smoke run);
#:   * the failing STAGE of a pipeline is named (from the stage codes), and a pipeline the reader closed early
#:     (`… | head`) is forgiven by the same pipe rule as the last one.
#: No trap is installed when the command reads statuses itself or brings its own handling (PIPESTATUS, its own
#: `trap`, `set -e` / `-o pipefail` / `errexit`, `exec` redirection, a `source` / `.` of a file whose traps and
#: PIPESTATUS reads would meet ours — not a virtualenv `bin/activate` or conda's `conda.sh`, which set no traps
#: — `_reads_own_status`): it then behaves as plain bash. `$?` keeps the trap
#: (bash restores `$?` after an ERR trap — pinned on the host's bash 3.2, the sandbox's 5.2 assumed the same);
#: a runner failure whose status the NEXT list member reads (`…; echo "rc=$?"`) is the model's own probe.
STATUS_DIRNAME = ".ghost_status"
_CONTAINER_STATUS_DIR = "/workspace/" + STATUS_DIRNAME
#: the legacy stdout marker (r0): stripped wherever a stored log still carries one, never read as a status
_FAILED_MARK = "__GHOST_FAILED="
_FAILED_LINE_RE = re.compile(r"\n?" + re.escape(_FAILED_MARK) + r"([^\n]*)(?:\n|\Z)")
_STATUS_FILE_RE = re.compile(r"__g_sf=['\"]?(" + re.escape(_CONTAINER_STATUS_DIR) + r"/[0-9a-f]{32})\b")
_STATUS_FILE_MAX = 64 * 1024

_ERR_TRAP = ('__g_lp=("$?" "${PIPESTATUS[@]}"); printf -v __g_ps "%s," "${__g_lp[@]:1}"; '
             '[ ${#__g_f[@]} -lt 32 ] && __g_f+=("${__g_lp[0]}:${__g_ps}:${BASH_COMMAND//[$\'\\t\\n\']/ }"); :')
_EXIT_TRAP = ('[ ${#__g_f[@]} -gt 0 ] && { mkdir -p "${__g_sf%/*}" && (IFS=$\'\\t\'; '
              'printf "%s\\n" "${__g_f[*]}") > "$__g_sf"; } 2>/dev/null')

#: the command reads statuses itself — our trap would change what it sees (PIPESTATUS is reset by the trap's own
#: commands) or fight its own handling: no trap, the shell's status stands
_CMD_POS = r"(?:^|[;&|(){]|\b(?:then|do|else)\b)\s*"          # a word in command position
#: an environment activator sets no traps and reads no statuses — sourcing one keeps the status rule
#: (`. .venv/bin/activate`, `source venv/bin/activate.sh`, conda's `…/etc/profile.d/conda.sh`)
_ACTIVATE_SCRIPT = (r"['\"]?(?:\S*/)?(?:bin/activate(?:\.sh|\.bash)?|etc/profile\.d/conda\.sh)['\"]?"
                    r"(?=$|[\s;&|)])")
_OWN_STATUS_RE = re.compile(
    r"PIPESTATUS|" + _CMD_POS + r"trap\s|" + _CMD_POS + r"(?:source|\.)\s+(?!" + _ACTIVATE_SCRIPT + r")\S"
    r"|" + _CMD_POS + r"set\s+(?:-[a-zA-Z]*e|[-+]o\s+(?:errexit|pipefail)|\+e)"
    r"|" + _CMD_POS + r"exec\s+(?:\d*[<>]|&>)", re.M)


def _reads_own_status(command: str) -> bool:
    return bool(_OWN_STATUS_RE.search(str(command or "")))


def _status_traps(container_path: str) -> str:
    # `set -E`: the ERR trap reaches inside shell functions (review r2 m5: `f(){ python3 x.py; echo t; }; f`
    # recorded nothing) — a subshell's record stays in the subshell (its EXIT trap is not ours)
    return (f"set -E; __g_f=(); __g_sf={shlex.quote(container_path)}; trap {shlex.quote(_ERR_TRAP)} ERR; "
            f"trap {shlex.quote(_EXIT_TRAP)} EXIT; ")


def new_status_file(host_root):
    """(container path, host path) of a fresh status file, or (None, None) without a host workspace."""
    if not host_root:
        return None, None
    try:
        import uuid as _uuid
        name = _uuid.uuid4().hex
        return f"{_CONTAINER_STATUS_DIR}/{name}", Path(str(host_root)) / STATUS_DIRNAME / name
    except Exception:  # noqa: BLE001
        return None, None


def _parse_status(text: str) -> list:
    out = []
    for line in str(text or "").splitlines():
        for item in line.split("\t"):
            code, _, rest = item.partition(":")
            stages, _, cmd = rest.partition(":")
            try:
                out.append((int(code), [int(c) for c in stages.split(",") if c.strip()], cmd.strip()))
            except ValueError:
                continue
    return out


def read_status_file(host_path, *, delete: bool = True) -> list:
    """The failures a run's ERR trap recorded — [(code, [stage codes], command), …] — read from the host side
    and removed. The directory lives in the bind mount, so the model can plant links there: the directory must
    be a real directory directly under the workspace, the file a REGULAR file (review r2 M1: a FIFO planted at
    the path blocked the open forever — on the event loop), opened O_NOFOLLOW|O_NONBLOCK and read up to 64 KB.
    Blocking I/O: async callers run it in a thread."""
    if not host_path:
        return []
    p = Path(str(host_path))
    try:
        import stat as _stat
        if not _stat.S_ISDIR(os.lstat(p.parent).st_mode):
            return []
        if not _stat.S_ISREG(os.lstat(p).st_mode):
            return []
        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return []
    try:
        import stat as _stat
        from .file_system import _block_if_regular       # §4SD: a regular file reads BLOCKING (virtiofs EAGAIN)
        if not _stat.S_ISREG(_block_if_regular(fd).st_mode):
            return []
        data = os.read(fd, _STATUS_FILE_MAX)
    except OSError:
        return []
    finally:
        os.close(fd)
        if delete:
            try:
                os.unlink(p)
            except OSError:
                pass
        _prune_status_dir(p.parent)
    return _parse_status(data.decode("utf-8", "replace"))


_STATUS_KEEP_S = 24 * 3600
_STATUS_PRUNE_MAX = 2000          # entries looked at per read (the model can fill the directory)


def _prune_status_dir(d) -> None:
    """Status files a job wrote and nobody read (a timed-out run, a job collected twice) leave after a day."""
    try:
        cut = time.time() - _STATUS_KEEP_S
        for n, e in enumerate(os.scandir(d)):
            if n >= _STATUS_PRUNE_MAX:
                break
            if re.fullmatch(r"[0-9a-f]{32}", e.name) and e.is_file(follow_symlinks=False) \
                    and e.stat(follow_symlinks=False).st_mtime < cut:
                os.unlink(e.path)
    except OSError:
        pass


def status_failures_of_job(command_text, host_root, *, delete: bool = False) -> list:
    """A run's recorded failures: the status file its wrapped command names. A foreground run's file is
    removed once read; a job's stays for its other readers (the wake, `run collect`) and ages out."""
    m = _STATUS_FILE_RE.search(str(command_text or ""))
    if not m or not host_root:
        return []
    return read_status_file(Path(str(host_root)) / STATUS_DIRNAME / m.group(1).rsplit("/", 1)[-1],
                            delete=delete)


_RUNNER_WRAPPERS = ("env", "nohup", "time", "sudo", "xvfb-run", "timeout", "nice", "stdbuf", "exec", "command")
_SHELL_LEAD_WORDS = ("do", "then", "else", "elif", "!", "{", "(", "if", "while", "until")


#: THE table of programs that RUN CODE — interpreters, test runners, compilers, build tools, `./x`, a script by
#: its extension. A failure behind `;` counts only for these (§4UY r1); it is also the claims ledger's reading
#: of "a runner".
_RUNNER_HEAD_RE = re.compile(
    r"^(?:python[\d.]*|py|node|nodejs|deno|bun|pytest|py\.test|npm|npx|yarn|pnpm|cargo|go|make|java|javac|ruby"
    r"|php|perl|bash|sh|zsh|uv|poetry|pipenv|tox|nox|coverage|gunicorn|uvicorn|flask|django-admin|manage\.py"
    r"|dotnet|mvn|gradle|swift|swiftc|rustc|gcc|g\+\+|cc|c\+\+|clang|clang\+\+|tsc|ts-node|tsx|jest|vitest|mocha"
    r"|streamlit|lua|Rscript|julia|kotlinc|scala|ghc|elixir|mix|dart|cmake|ninja|bazel)$"
    r"|^\./|\.(?:py|sh|js|mjs|cjs|ts|rb|pl)$")


def _command_head(text: str) -> str:
    """The program a simple command (or a pipeline stage) runs: env assignments, wrappers (`timeout -s KILL
    5`, `nice -n 5`) and shell keywords that only lead into it skipped; a path's basename except `./x`."""
    toks = str(text or "").strip().lstrip("({!").split()
    while toks:
        t = toks[0]
        if re.fullmatch(r"[A-Za-z_]\w*=.*", t) or t in _SHELL_LEAD_WORDS:
            toks = toks[1:]
            continue
        # a function definition leads into its body (§4UY r3): `name(){`, `name() {`, `function name {`
        if t == "function" and len(toks) > 1:
            toks = toks[2:]
            continue
        if re.fullmatch(r"[\w.:-]+\(\)\{?", t) or t == "()" or (t == "{" and len(toks) > 1):
            toks = toks[1:]
            continue
        if t in _RUNNER_WRAPPERS:
            toks = toks[1:]
            while toks and toks[0].startswith("-"):
                toks = toks[2:] if toks[0] in ("-s", "--signal", "-k", "--kill-after", "-n", "-u") else toks[1:]
            if t == "timeout" and toks and re.fullmatch(r"[\d.]+[smhd]?", toks[0]):
                toks = toks[1:]
            continue
        break
    if not toks:
        return ""
    return toks[0] if toks[0].startswith("./") else toks[0].rsplit("/", 1)[-1]


#: a runner's invocations that run no code of the project (review r2 m2): `python -m pip|venv|ensurepip|
#: http.server|json.tool`, `make clean`, `npx kill-port` — read by the module / target / package
_NOT_A_RUN_RE = re.compile(
    r"^python[\d.]*\s+(?:-\S+\s+)*-m\s+(?:pip|venv|virtualenv|ensurepip|http\.server|json\.tool|site|pydoc)\b"
    r"|^(?:make|gmake)\s+(?:-[Cf]\s+\S+\s+|-\S+\s+)*(?:clean|distclean|mostlyclean|maintainer-clean)\b"
    r"|^(?:npx|pnpx)\s+(?:-\S+\s+)*(?:kill-port|fkill|rimraf)\b"
    r"|^(?:pip[\d.]*|npm|yarn|pnpm)\s+(?:install|uninstall|i|add|remove|ci|cache)\b")


def _runs_code(text: str) -> bool:
    """Is this command a program that runs code (`_RUNNER_HEAD_RE`, minus `_NOT_A_RUN_RE`)?"""
    head = _command_head(text)
    if not head or not _RUNNER_HEAD_RE.match(head):
        return False
    toks = str(text or "").split()
    i = next((k for k, w in enumerate(toks) if w.rsplit("/", 1)[-1] == head or w == head), None)
    return not (i is not None and _NOT_A_RUN_RE.match(" ".join([head] + toks[i + 1:])))


_TIMEOUT_LED_RE = re.compile(r"^\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:(?:sudo|nice|nohup|env)\s+(?:-\S+\s+)*)*timeout\b")


def _take_status(output):
    """(output without our marker lines, [pipeline stage statuses] or None, []) — the legacy stdout failure
    marker is stripped wherever a stored log still has one, never read (failures come from the status file:
    `read_status_file`)."""
    text = output if isinstance(output, str) else str(output or "")
    if _FAILED_MARK in text:
        text = _FAILED_LINE_RE.sub("\n", text)
    elif output is None or output == "":
        text = output
    clean, codes = _take_pipestatus_only(text)
    return clean, codes, []


def _list_members(command: str) -> list:
    """The command's top-level list members (heredoc data bodies out), each as its pipeline stages."""
    try:
        text = _strip_data_heredocs(str(command or ""))
    except Exception:  # noqa: BLE001
        text = str(command or "")
    out = []
    for line in text.split("\n"):
        out.extend(pl for pl in _shell_pipelines(line) if "".join(pl).strip())
    return out


def _failure_parts(item):
    """(code, [stage codes], command) from a recorded failure (an old 2-tuple has no stage codes)."""
    if len(item) == 3:
        return int(item[0]), list(item[1] or []), str(item[2] or "")
    return int(item[0]), [], str(item[1] or "")


def _list_failure(command: str, failures, output: str = ""):
    """(code, program) of the failure that is the run's status when the shell said 0 (see the rule above), or
    (0, "")."""
    if not failures:
        return 0, ""
    members = _list_members(command)
    norm = [[" ".join(st.split()) for st in pl] for pl in members]
    pos, found = 0, (0, "")
    for item in failures:
        try:
            code, stage_codes, text = _failure_parts(item)
        except (TypeError, ValueError, IndexError):
            continue
        if not code:
            continue
        t = " ".join(text.split())
        idx = None
        # review r4: an EXACT stage match first — `echo 'step: python3 t.py'` must not stand for `python3 t.py`
        for exact, rng in ((True, range(pos, len(norm))), (True, range(0, len(norm))),
                           (False, range(pos, len(norm))), (False, range(0, len(norm)))):
            idx = next((i for i in rng if t and any(t == st if exact else t in st for st in norm[i])), None)
            if idx is not None:
                break
        stages = members[idx] if idx is not None else [t]
        if idx is not None:
            pos = idx + 1
            # §4UY r3: the model read THIS status itself in the very next member (`…; echo "rc=$?"`) — it
            # asked for that code and sees it; a later program's failure still counts
            nxt = members[idx + 1] if idx + 1 < len(members) else []
            if any(re.search(r"\$\?|\$\{\?\}", st) for st in nxt):
                continue
        if len(stages) > 1 and _normalise_pipe_exit(" | ".join(stages), code, output) == 0:
            continue                                 # an early-closed pipe (`… | head`), as for the last one
        # the failing stage: the rightmost non-zero stage that runs code (review r1 (d): BASH_COMMAND names a
        # pipeline's LAST stage, `tee log` for `python x.py | tee log`)
        # one stage: the recorded simple command itself (inside a function body or a loop the list member is
        # `f(){ …` / `do …`, the trap's BASH_COMMAND is the program that failed)
        cands = ([(t, code)] if len(stages) == 1
                 else [(stages[i], c) for i, c in enumerate(stage_codes) if c] if len(stage_codes) == len(stages)
                 else [(stages[-1], code)])
        cands = [(st, c) for st, c in cands if _runs_code(st)]
        if not cands:
            continue                                 # a reader / `rm` / `curl` / `pip` failing keeps the status
        stage, scode = cands[-1]
        if _TIMEOUT_LED_RE.match(stage) and scode in (124, 137, 143):
            continue                                 # a bounded run its own `timeout` ended
        found = (scode, " ".join(re.sub(r"^\s*(?:(?:do|then|else|elif|if|while|until|\{|\(|!)\s+)+", "", stage).split()))
    return found


def _list_failure_note(code: int, text: str) -> str:
    return (f"\n[NOTE: the shell's own status was 0 (the LAST command of the list succeeded), but `{text[:120]}` "
            f"earlier in it exited {code} — reported as EXIT CODE {code}: fix that program, do not re-run the "
            f"same list.]")


def _normalise_exit(command: str, exit_code, output: str = "", failures=None) -> int:
    """THE exit-code normaliser for a sandbox run — pipe forgiveness (§4GE), a quiet find (§4IE), then the
    list rule (§4UY C9/C10). Every adoption of a sandbox result in the command path goes through here; the
    enumeration in `tests/test_execute_self_reported_exit.py` fails when one does not."""
    code = _normalise_find_exit(command,
                                _normalise_pipe_exit(command, exit_code, output), output)
    try:
        if int(code or 0) == 0 and failures:
            got, _t = _list_failure(command, failures, output)
            if got:
                return got
    except (TypeError, ValueError):
        pass
    return code


#: §4PH (R2-3): a streamed docker exec inspected before the daemon recorded
#: its exit reads ``None`` — and ``int(None or 0)`` made it a success over a
#: traceback. An unknown status is a failure the model is told about.
_NO_STATUS_NOTE = ("\n[NOTE: the sandbox reported NO exit status for this run — treated as a "
                   "failure. If the output above looks complete, re-run it once for a real status.]")


def _adopt_exit(exit_code, output):
    """(exit_code, output) with a missing status made an explicit failure."""
    if exit_code is None:
        return 1, str(output or "") + _NO_STATUS_NOTE
    return exit_code, output


async def _run_failures(wrapped: str, sandbox_manager) -> list:
    """The failures the run of ``wrapped`` (a `_bash_c` string) recorded in its status file, or [] — read in a
    thread (review r2 M1: file I/O on a path the model controls never runs on the event loop)."""
    try:
        return await asyncio.to_thread(status_failures_of_job, wrapped,
                                       getattr(sandbox_manager, "host_workspace", None), delete=True)
    except Exception:  # noqa: BLE001
        return []


def _read_shell_run(command: str, exit_code, output, failures=None):
    """(exit_code, output, stage statuses) — THE reading of one finished sandbox run of `command` (the
    primary run and every heal retry): our marker lines out, a missing status made a failure, the exit
    normalised (§4UY C9/C10: an earlier program's failure under a 0 — ``failures`` from its status file — is
    named in a note)."""
    output, codes, _legacy = _take_status(output)
    exit_code, output = _adopt_exit(exit_code, output)
    raw = exit_code
    exit_code = _normalise_exit(command, exit_code, output, failures)
    try:
        if exit_code and int(raw or 0) == 0:
            _c, _t = _list_failure(command, failures, output)
            if _c == exit_code:
                output = str(output or "") + _list_failure_note(_c, _t)
    except (TypeError, ValueError):
        pass
    return exit_code, output, codes


#: §4PH (R2-4): `nohup python3 app.py &` — a backgrounded process that is not
#: server-shaped (those are refused) returns 0 the instant it STARTED; its
#: outcome and output are not in the result.
_BG_NOTE = ("\n[NOTE: `&` detached a process — EXIT CODE 0 here means only that it STARTED. Its "
            "outcome and output are NOT in this result (unless you redirected them to a file you "
            "then read). Check it in a later command (its log, `pgrep`) before saying it works.]")


def _bg_detached(command: str) -> bool:
    """A bare `&` OUTSIDE heredoc bodies and quotes (r3 reader M4: `Tom &amp;
    Jerry` in a heredoc is data), not server-shaped (those are refused) and
    not job-killed in the same command (`… & …; kill %1`)."""
    cmd = str(command or "")
    try:
        cmd = _strip_data_heredocs(cmd)
    except Exception:  # noqa: BLE001
        pass
    if _ends_its_background(cmd):             # §4UY C7: one reading of "the command ends what it started"
        return False
    cmd = _QUOTED_RE.sub(" ", cmd)
    return bool(_BG_AMP_RE.search(cmd)) and not _detached_server_signals(cmd)


#: §4IE: `find … 2>/dev/null` exits 1 whenever ANY directory under its root
#: was unreadable — with stderr discarded that status carries nothing the
#: model can act on, and the paths it printed are valid. The FAILURE banner
#: it earned read as "the command failed": probe ifs18371… re-ran eight
#: `find / …` variants in a row (turns 31–38, up to 166 s each), each
#: banner-flagged, each a strike. Forgiven ONLY when the last command of the
#: line is a `find` whose stderr is discarded and something was printed.
_FIND_QUIET_RE = re.compile(r"^\s*find\b.*2>\s*/dev/null", re.S)


def _shell_pipelines(command: str) -> list:
    """`command` as ``[[stage, …], …]`` — the `;`/`&&`/`||`-separated
    pipelines, each split into its `|`/`|&` stages, QUOTE-AWARE (§4LP
    follow-up): `… | awk '{print $1; exit}'` and `grep 'a|b'` were cut at the
    quoted `;`/`|`, so the reader was not seen and the pattern became a stage.
    Blank pipelines are dropped; at least one (possibly empty) is returned.
    A whole-command `bash -c '…'` wrapper (a detached job's recorded
    command) is read as the script it runs."""
    s = str(command or "")
    try:
        _w = shlex.split(s)
    except ValueError:
        _w = []
    if len(_w) == 3 and Path(_w[0]).name in ("bash", "sh") and _w[1] == "-c":
        return _shell_pipelines(_w[2])
    out, stages, cur, q, i = [], [], [], None, 0
    while i < len(s):
        c = s[i]
        if q:
            cur.append(c)
            if c == "\\" and q == '"' and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if c == q:
                q = None
            i += 1
            continue
        if c == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == "#" and (i == 0 or s[i - 1] in " \t\n;|&("):
            # a comment runs to the end of its line; an apostrophe in it is
            # not a quote (review N5: `# don't … # it's fine` paired up)
            nl = s.find("\n", i)
            i = len(s) if nl < 0 else nl
            continue
        if c in "'\"":
            q = c
        elif s.startswith(("&&", "||"), i) or c == ";":
            stages.append("".join(cur))
            out.append(stages)
            stages, cur = [], []
            i += 1 if c == ";" else 2
            continue
        elif c == "|":
            stages.append("".join(cur))
            cur = []
            i += 2 if s.startswith("|&", i) else 1
            continue
        cur.append(c)
        i += 1
    if q:
        # an unpaired quote — an apostrophe in a comment or a heredoc body
        # ("# don't …") is not shell quoting; the quote-blind split is the
        # better reading than one stage swallowing the rest (review m1)
        out = [[st.strip().lstrip("&").strip() for st in pl.split("|")]
               for pl in re.split(r";|&&|\|\|", s) if pl.strip()]
        return out or [[""]]
    stages.append("".join(cur))
    out.append(stages)
    out = [[st.strip() for st in pl] for pl in out if "".join(pl).strip()]
    return out or [[""]]


def _last_simple_command(command: str) -> str:
    """The last `;`/`&&`/`||`-separated command, minus a trailing pipe tail
    of pure filters (`| head`, `| sort`, `| grep -v …` keep find's status
    under pipefail only when THEY fail)."""
    return _shell_pipelines(command)[-1][0]


def _normalise_find_exit(command: str, exit_code, output: str = "") -> int:
    try:
        code = int(exit_code or 0)
    except (TypeError, ValueError):
        return exit_code
    if code != 1:
        return code
    if not _FIND_QUIET_RE.match(_last_simple_command(command)):
        return code
    _o = str(output or "").strip()
    # the sandbox's own "no output" placeholder is not output (§4LO: two live
    # quiet finds were turned from exit 1 into exit 0 by it)
    if not _o or re.fullmatch(r"\[SYSTEM ERROR\]: Process failed \(Exit \d+\) with no output\.", _o):
        return code
    if re.search(r"^find: ", str(output or ""), re.M):
        return code
    return 0


#: "The DOWNSTREAM of a pipe closed early", not "the command failed".
#: Without this, turning pipefail on would report every `… | head -N` that
#: actually truncated as a FAILURE — a far commoner shape than the one
#: pipefail exists to catch.
#:
#: ⚠ THE EXIT CODE ALONE IS NOT ENOUGH, and the host lied about it.
#: Measured on macOS: standard tools report 141 (SIGPIPE) and Python 120 (a
#: failed stdout flush at shutdown). Measured in the REAL sandbox
#: (Debian, bash 5.2, CPython 3.11) the same Python pipeline reports
#: **1** — it raises `BrokenPipeError` as an ordinary uncaught exception —
#: which is indistinguishable by code from a script that genuinely failed.
#: So the second signal is the runtime NAMING the broken pipe itself, in
#: output that is captured anyway. A real traceback keeps its strike; a
#: pipe closed by `head` does not.
_PIPE_CLOSED_EARLY_CODES = (141, 120)
_BROKEN_PIPE_RE = re.compile(r"BrokenPipeError|Broken pipe|SIGPIPE",
                             re.IGNORECASE)


def _bash_c(command: str, status_path: str = None) -> str:
    """The sandbox shell invocation for a raw command — ONE home, because
    both call sites (the primary run and the project-path remap retry) must
    agree about pipefail or one of them keeps the old blindness.

    WHY PIPEFAIL (request e0f4a8bd, 2026-09-08). `python3 probe.py 2>&1 |
    head -200` raised a Traceback and the tool logged **execution ok, exit
    0** — twice — because bash returns the exit status of the LAST command
    in a pipeline and `head` succeeded. The strike counter therefore never
    moved off 1/6, the failure taught the model nothing, and it re-ran the
    same broken probe. A pipeline's failure is the pipeline's failure.
    """
    # §4UY C9/C10: the ERR/EXIT traps record every top-level failure in a status file the host reads back
    # (`read_status_file`) — none when the command reads statuses itself (`_reads_own_status`)
    traps = "" if _reads_own_status(command) else _status_traps(
        status_path or f"{_CONTAINER_STATUS_DIR}/{uuid.uuid4().hex}")
    cmd = "set -o pipefail; " + traps + (command or "")
    if _grep_tail_after_pipe(command):
        cmd += _PIPESTATUS_TRAILER
    return "bash -c " + shlex.quote(cmd)


#: §4LO: `python3 x.py 2>&1 | grep X` that CRASHED and `… | grep X` that just
#: found nothing both exit 1 with no output under pipefail — and the second
#: is rewritten to "no matches, exit 0", so the crash read as success. For
#: that one shape the shell reports every stage's status on a marker line.
_PIPESTATUS_MARK = "__GHOST_PIPESTATUS="
_PIPESTATUS_TRAILER = (
    '\n__g_x=("$?" "${PIPESTATUS[@]}"); trap - ERR; __g_ps=("${__g_x[@]:1}"); '
    # §4UY: an ERR trap that ran on the last pipeline reset PIPESTATUS to its own `:` — its saved copy is the
    # pipeline's (a non-zero status with a lone 0 stage cannot be the pipeline's own)
    'if [ "${__g_x[0]}" -ne 0 ] && [ "${#__g_x[@]}" -eq 2 ] && [ "${__g_x[1]}" -eq 0 ] && [ "${__g_lp[0]:-0}" = "${__g_x[0]}" ]; '
    'then __g_ps=("${__g_lp[@]:1}"); fi; '
    'printf "\\n%s%s\\n" "' + _PIPESTATUS_MARK + '" "${__g_ps[*]}"; '
    # §4PH (R2-8): the FIRST non-zero stage is the one that failed — the
    # producer's code, not the grep's 1 that followed it
    '__g_r=0; for __g_s in "${__g_ps[@]}"; do [ "$__g_s" -ne 0 ] && [ "$__g_r" -eq 0 ] && __g_r=$__g_s; done; exit $__g_r')
_PIPESTATUS_LINE_RE = re.compile(r"\n?" + re.escape(_PIPESTATUS_MARK) + r"([0-9 ]*)\n?\s*\Z")


def _grep_tail_after_pipe(command: str) -> bool:
    stages = _shell_pipelines(command)[-1]
    if len(stages) < 2:
        return False
    toks = stages[-1].split()
    while toks and re.fullmatch(r"[A-Za-z_]\w*=.*", toks[0]):
        toks = toks[1:]
    return bool(toks) and Path(toks[0]).name in _EXIT1_MEANS_NO_MATCH


def _an_upstream_stage_failed(command: str, codes) -> bool:
    """Did a stage BEFORE the trailing grep really fail? 141 is a stage the
    downstream closed early, and a quiet `find … 2>/dev/null` exits 1 for any
    unreadable directory (review: `find / … 2>/dev/null | grep x` with no
    match read "1 1" and became a failure). An upstream grep's 1 is "no
    match" too — `grep a f | grep b` is a search that found nothing."""
    if not codes or len(codes) < 2:
        return False
    stages = _shell_pipelines(command)[-1]
    for i, c in enumerate(codes[:-1]):
        if c in (0, 141):
            continue
        if c == 1 and i < len(stages) and (_FIND_QUIET_RE.match(stages[i])
                                           or _stage_head(stages[i]) in _EXIT1_MEANS_NO_MATCH):
            continue
        return True
    return False


def _stage_head(stage: str) -> str:
    """The program name of one pipeline stage (env assignments skipped)."""
    toks = str(stage or "").split()
    while toks and re.fullmatch(r"[A-Za-z_]\w*=.*", toks[0]):
        toks = toks[1:]
    return Path(toks[0]).name if toks else ""


def _take_pipestatus(output):
    """(output without our marker lines, [stage statuses] or None) — the ERR-trap failure line is dropped too
    (`_take_status` keeps it)."""
    clean, codes, _failures = _take_status(output)
    return clean, codes


def _take_pipestatus_only(output):
    """(output without the stage-status line, [stage statuses] or None)."""
    text = output or ""
    m = _PIPESTATUS_LINE_RE.search(text)
    if not m:
        return output, None
    try:
        codes = [int(x) for x in m.group(1).split()]
    except ValueError:
        codes = None
    return text[:m.start()], codes


def _normalise_pipe_exit(command: str, exit_code, output: str = "") -> int:
    """Undo pipefail's one false alarm: an early-closing downstream.

    Narrowed to commands that actually contain a PIPE, so a bare
    `python3 x.py` exiting 1 stays the failure it is. `||` is stripped
    before the test: it is an OR, not a pipeline, and nothing in
    `a || b` can close a pipe early. Within a pipeline the run is forgiven
    only when the failure IDENTIFIES ITSELF as a broken pipe — by exit code
    (141/120) or by the runtime's own error name in the output. Anything
    else, including a `ModuleNotFoundError` through `| head`, is a failure
    and costs its strike.
    """
    try:
        code = int(exit_code or 0)
    except (TypeError, ValueError):
        return exit_code
    if code == 0 or "|" not in str(command or "").replace("||", ""):
        return code
    # §4LO: only a pipeline whose READER can close early can break a pipe —
    # `| tail` reads to the end, and a pipe elsewhere in the command does not
    # excuse its last pipeline (`pip install … | tail` failing, a DataLoader's
    # own BrokenPipeError, `cat a | sort > b; python3 client.py`).
    if not _has_early_closing_reader(command):
        return code
    if code in _PIPE_CLOSED_EARLY_CODES:
        return 0
    # §4PH (R2-5): the TEXT arm forgives a runtime's own exit 1 for the broken
    # pipe it named — never a kill (124/137) or an unrelated failure whose
    # message merely contains the words
    if code == 1 and _BROKEN_PIPE_RE.search(str(output or "")):
        return 0
    return code


_EARLY_READER_RE = re.compile(
    r"^(?:head\b|grep\b.*\s-(?:[a-zA-Z]*[qm][a-zA-Z0-9]*|-max-count\b|-quiet\b)|"
    r"sed\b.*\d*q\b|awk\b.*\bexit\b|less\b|more\b|read\b)")


def _has_early_closing_reader(command: str) -> bool:
    """The LAST pipeline of `command` has a downstream stage that may stop
    reading before its input ends (head, grep -q/-m, sed …q, awk … exit)."""
    segs = _shell_pipelines(command)
    # a trailing `echo "exit=$?"` reports the pipeline BEFORE it (§4HZ)
    while len(segs) > 1 and re.match(r"\s*(?:echo|printf)\b.*\$\?", "|".join(segs[-1])):
        segs.pop()
    stages = segs[-1]
    for st in stages[1:]:
        toks = st.split(None, 1)
        if toks:
            st = Path(toks[0]).name + ((" " + toks[1]) if len(toks) > 1 else "")   # /usr/bin/head
        if _EARLY_READER_RE.match(st):
            return True
    return False


async def _run_in_sandbox(sandbox_manager, cmd_str, *, timeout=_EXEC_TIMEOUT_S,
                          label=None, project_id=None, cleanup_paths=None,
                          identity=None, **kwargs):
    """Run a command for the execute TOOL, preferring the promotable path.

    Returns ``(output, exit_code, job_entry_or_None)``. Falls back to the
    plain 2-tuple ``execute`` for any manager that does not implement jobs.

    The capability is probed through an explicit class-level flag, NOT
    ``hasattr(mgr, "execute_promotable")``: most sandbox stubs in the suite
    (and in any caller's tests) are ``MagicMock``s, which auto-create every
    attribute — a hasattr probe routes them all down the promotable path and
    then unpacks a Mock as a 3-tuple. ``is True`` is the one check a
    duck-typed mock cannot accidentally satisfy.

    ``cleanup_paths`` are HOST files the run OWNS — the ephemeral script an
    `execute(content=…)` call generates. They cannot be deleted while a
    promoted job is still executing them, so the supervisor takes ownership
    and unlinks them when the job ends.

    ``identity`` is what the supervisor's re-run guard compares. It defaults
    to the command, which is right for a `command=` run (the command IS the
    content) but WRONG for a script run: `python3 -u build.py` is stable
    across every rewrite of build.py, so a model that reads the promotion
    banner, fixes the script and re-runs it would be handed the OLD job and
    blocked from running the fix. The script path folds in a content hash.
    """
    runner = getattr(sandbox_manager, "execute_promotable", None)
    try:
        from ..utils.logging import request_id_context as _ridc
        _delegate_run = str(_ridc.get() or "").startswith("sub-job-")
    except Exception:  # noqa: BLE001
        _delegate_run = False
    if getattr(sandbox_manager, "supports_job_promotion", False) is not True \
            or runner is None or _delegate_run:
        # §4NS: a `sub-job-` turn's command is never promoted — its landing
        # woke the MAIN agent with the full tool surface, survived a cancel,
        # and a probe's job woke a real turn. (The delegate sub-agents that
        # first used the prefix were removed in §4SI; `main._resume_after_job`
        # still starts background wake turns as `sub-job-…`.)
        output, exit_code = await asyncio.to_thread(
            sandbox_manager.execute, cmd_str, timeout=timeout, **kwargs)
        return output, exit_code, None
    return await asyncio.to_thread(
        runner, cmd_str, timeout=timeout, label=label,
        project_id=project_id, cleanup_paths=cleanup_paths,
        identity=identity, **kwargs)


def _promotion_banner(job: dict, ran_s: int, run_name: str = "") -> str:
    """The message that turns a timeout into a promotion.

    It has one job beyond informing: FORECLOSING THE RETRY. The retry is what
    turned 10 minutes of loss into 40 in the incident this replaces, so the
    "do NOT re-run" line is not politeness — it is the fix. (The supervisor
    enforces it too: an identical command that is already running as a job
    returns that job instead of starting a second copy.)

    The closing instruction deliberately avoids "I will report back" phrasing:
    that reads as a PROMISE OF FUTURE ACTION, which the pending-promise guard
    rejects. Since 2026-08-12 the reaper DOES wake the model when the job
    lands (main._resume_after_job), so the work genuinely continues — but the
    promise still belongs to the system, not to this reply, and the guard is
    right that a reply cannot keep it."""
    jid = job.get("id")
    now = time.time()

    def _num(key):
        try:
            return float(job.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # REMAINING lifetime, not the configured one. For a fresh promotion the
    # two are the same (now == promoted_at); for an already-running job hit
    # by the re-run guard they are not, and quoting the full TTL would
    # overstate how long the model can wait.
    ttl_min = max(1, int((_num("deadline_at") - now) / 60)) \
        if _num("deadline_at") > now else 1
    if job.get("duplicate"):
        # …and its age is the ORIGINAL command's, not the ~0s this call took.
        age = int(now - _num("started_at")) if _num("started_at") else ran_s
        head = (f"▎ NOT re-run: this exact command is ALREADY running as "
                f"background job {jid} (started {age}s ago). A second copy "
                f"was NOT started — it would compete with the first for the "
                f"same sandbox CPU.")
    else:
        head = (f"▎ Still running at {ran_s}s — promoted to background job "
                f"{jid}. Do NOT re-run this command — it was NOT killed and "
                f"is still working.")
    if run_name:
        # §4SX D5: the promoted job is a `job` run named after its id
        how = (f"▎ Poll it with run(action='status', name='{run_name}'); read its full "
               f"result with run(action='collect', name='{run_name}') once it "
               f"finishes. Stop it with run(action='stop', name='{run_name}').\n")
    else:
        how = (f"▎ Poll it with run(action='status', name='{jid}'); read its full "
               f"result with run(action='collect', name='{jid}') once it "
               f"finishes. Stop it with run(action='stop', name='{jid}').\n")
    return (
        f"{head}\n"
        f"{how}"
        f"▎ It keeps running in the sandbox (up to {ttl_min} more minutes, "
        f"then it is reaped) and survives this turn; its live output is in "
        f"'{job.get('log')}' (file_system operation='read').\n"
        f"▎ FINISH YOUR TURN NOW: state plainly that the work is STILL "
        f"RUNNING in the background and where its result will appear — do "
        f"not wait for it here and do not claim it finished. You will be "
        f"woken with the result when it lands, so there is nothing to poll "
        f"for and nothing to promise."
    )


def _project_id_from_workdir(container_workdir) -> str:
    """Owning project id from a scoped container workdir, else ''. Only the
    canonical 12-hex form is accepted — a garbage workdir must not fabricate
    ownership."""
    m = re.search(r"/projects/([0-9a-f]{12})(?:/|$)",
                  str(container_workdir or ""))
    return m.group(1) if m else ""


def _record_promoted_command(workspace_model, command, duration_s, job):
    """Workspace ledger entry for a promoted run. Exit code 0 with an
    explicit note — recording it as a failure would put the same wrong label
    into the ledger that the strike accounting no longer takes."""
    _run = _RUN_STATE.get()                    # §4UY N6 r3: both promotion sites pass here — the run's record
    if _run is not None and isinstance(job, dict) and job.get("id"):
        _run["job"] = str(job.get("id"))
    if workspace_model is None or not getattr(workspace_model, "enabled", False):
        return
    try:
        workspace_model.record_command_outcome(
            command=str(command), exit_code=0,
            duration_seconds=float(duration_s),
            note=f"still running — promoted to background job {job.get('id')}",
            job_id=str(job.get("id") or ""),                       # §4PF (W7)
        )
    except Exception:  # noqa: BLE001 — telemetry must never break a tool
        pass


def _register_promoted_job(job_registry, job, label):
    """Mirror a promoted sandbox job into the in-process job registry so the
    existing `jobs` tool lists it alongside delegated sub-agents. The sandbox
    supervisor stays the source of truth for its state (this row carries no
    asyncio task); `tools/delegate.py` syncs the two before answering."""
    if job_registry is None or not isinstance(job, dict):
        return
    sid = job.get("id")
    if not sid:
        return
    try:
        # Dedupe. Two writers can mirror the same sandbox job: this one, and
        # the `jobs` tool's adoption pass for a job it has not seen. A second
        # row would be reported as its own "still running" job for ever — and
        # the re-run guard means one command legitimately returns the SAME
        # job id twice, which lands here twice by design.
        for existing in job_registry.list(kind="sandbox"):
            if (existing.meta or {}).get("sandbox_job_id") == sid:
                return
        job_registry.register(
            "sandbox", str(label or job.get("command") or "")[:200],
            sandbox_job_id=sid, log=job.get("log"),
            command=job.get("command"))
    except Exception:  # noqa: BLE001 — never break the tool over bookkeeping
        logging.getLogger("GhostAgent").debug(
            "promoted-job registry mirror failed", exc_info=True)


def _append_note(res, note: str):
    """Append trailing text WITHOUT dropping a `ToolOutcome`'s status.

    `ToolOutcome` is a `str` SUBCLASS, so `res + note` is `str.__add__` and
    returns a plain `str` — silently discarding the status. Three sites in
    this module did exactly that; one was found live and its two siblings
    only by enumeration.
    """
    from .outcome import append_note as _shared
    return _shared(res, note)


def _promoted_result(job: dict, ran_s: int, output: str, run_name: str = "") -> "ToolOutcome":
    """Tool result for a promoted command. EXIT CODE 0 on purpose: the turn
    loop scores a non-zero `execute` exit as a FAILURE and a transient
    strike, and a legitimately-long task is not a failure — miscounting it
    poisons the outcome labels the learning loop trains on. The annotation
    rides AFTER the digits so `EXIT CODE:\\s*(\\d+)` parsers keep working.

    ``PROMOTED_RESULT_MARKER`` rides in the FIRST LINE because success-shaped
    is exactly what the other readers of an execute result must NOT take this
    for: the verify gate that marks a task DONE and the TDD gate that
    installs a skill both key on exit 0, and neither would otherwise know the
    command has not finished."""
    if not isinstance(job, dict):
        job = {}
    body = (output or "").strip()
    _state = ("ALREADY RUNNING as background job"
              if job.get("duplicate") else
              "STILL RUNNING — promoted to background job")
    # UNRESOLVED, not OK: this is THE unfinished-call case, and the status
    # says so where the text cannot. The text is byte-identical to before
    # (still `EXIT CODE: 0`), so every downstream reader that keys on the
    # banner — the verify gate, the TDD skill gate, project_advancer — is
    # unaffected; only the dispatch loop's status-aware predicates change,
    # and they now correctly decline to give a running job a verdict.
    return ToolOutcome.unresolved(
        f"--- COMMAND RESULT --- {PROMOTED_RESULT_MARKER}\n"
        f"EXIT CODE: 0 ({_state} "
        f"{job.get('id')}, NOT finished)\n"
        f"STDOUT/STDERR:\n{_promotion_banner(job, ran_s, run_name)}\n"
        f"--- output so far ---\n{body or '(nothing printed yet)'}",
        reason_code="promoted_to_background_job",
    )

# SANDBOX EGRESS GUARD (agent's own ports). The sandbox container has its
# own loopback, so 127.0.0.1:8000 in there is NOT the agent's API and
# 127.0.0.1:8088 is NOT the upstream LLM. Every observed probe of these
# ports from inside the sandbox ended in the same misdiagnosis chain:
# connection failure → "the server is down on the user's machine" → write
# a mock/stand-in server (the forbidden engine, three incidents:
# 2026-07-02, 07-04, 07-05 request C5). Prompt-side warnings did not stop
# it — the model trusts its own curl over the prompt — so the probe itself
# is intercepted before execution and answered with the ground truth.
_AGENT_PORT_PROBE_RE = re.compile(
    r"(?:127\.0\.0\.1|localhost|0\.0\.0\.0)\s*:\s*(?:8000|8088)\b",
    re.IGNORECASE,
)

# Heredoc bodies are DATA being written, not commands being run. Observed
# false positive 2026-07-08 (chess session): `cat > app.py <<'EOF' …` was
# blocked because the FILE CONTENT legitimately references the agent's API
# (the app calls Ghost by design) — sealing off the model's only remaining
# write path. Strip heredoc bodies before matching; a real probe puts the
# URL in the executed part of the command.
_HEREDOC_BODY_RE = re.compile(
    r"<<-?\s*(['\"]?)(\w+)\1.*?(?:\n\2(?=\s|$)|\Z)", re.DOTALL)

# BUT a heredoc fed to an interpreter is EXECUTED code, not data:
# `python3 <<'EOF' … urlopen(…) … EOF` runs the body exactly like a
# script, so stripping it as "data" was a clean bypass of the egress
# block. The opener line (everything on the physical line carrying `<<`,
# including a trailing `| python3`) names the consumer — strip only when
# no interpreter appears there.
_HEREDOC_INTERP_RE = re.compile(
    r"(?:^|[|&;(`\s])(?:python[0-9.]*|node(?:js)?|sh|bash|zsh|dash|ksh|"
    r"ruby|perl)\b")


def _strip_data_heredocs(command: str) -> str:
    """Replace heredoc bodies that are pure data (file writes) with a
    placeholder; keep bodies an interpreter will execute so the probe
    scan sees them."""
    def _sub(m):
        line_start = command.rfind("\n", 0, m.start()) + 1
        line_end = command.find("\n", m.start())
        if line_end == -1:
            line_end = len(command)
        if _HEREDOC_INTERP_RE.search(command[line_start:line_end]):
            return m.group(0)
        return "<<HEREDOC-BODY>>"
    return _HEREDOC_BODY_RE.sub(_sub, command)

# A probe needs a network client. Plain text manipulation that mentions the
# URL (echo > file, sed on a config) proves nothing and is allowed.
_NET_CLIENT_RE = re.compile(
    r"(?:\bcurl\b|\bwget\b|\bnc\b|\bncat\b|\bnetcat\b|\btelnet\b|\bhttpx\b|"
    r"\baiohttp\b|\brequests\b|\burllib\b|\burlopen\b|http\.client|"
    r"\bsocket\b|/dev/tcp/|\bfetch\s*\(|\baxios\b|\bhttpie\b)",
    re.IGNORECASE)


def _command_probes_agent_port(command: str) -> bool:
    """True iff a SHELL command actually probes the agent's ports: the
    loopback URL must appear OUTSIDE data heredoc bodies AND a
    network-client token must be present. Inline ``content`` (code handed
    to an interpreter) is judged by the caller with the strict any-match
    rule — executed code that mentions the URL is always suspect."""
    stripped = _strip_data_heredocs(str(command))
    if not _AGENT_PORT_PROBE_RE.search(stripped):
        return False
    return bool(_NET_CLIENT_RE.search(stripped))

_AGENT_PORT_PROBE_MSG = (
    "SANDBOX EGRESS BLOCKED (known blind spot — command NOT executed): "
    "this references 127.0.0.1:8000 / :8088, but inside your sandbox that "
    "loopback is the CONTAINER'S OWN, not the host's. The agent API "
    "(:8000) and upstream LLM (:8088) are running fine on the host — a "
    "connection failure from in here would prove NOTHING, and acting on "
    "one has repeatedly led to writing a forbidden mock server. Ground "
    "truth: from the USER'S machine, 127.0.0.1:8000 reaches the agent. "
    "What to do instead: (a) verify endpoints with the `browser` tool "
    "(it runs on the host), (b) ask the user to run the command on THEIR "
    "machine, or (c) if you need an in-sandbox server for something "
    "unrelated, bind a DIFFERENT port (e.g. 8081). Writing a FILE whose "
    "text merely mentions these URLs is fine — use file_system "
    "operation='write' (or a shell heredoc), which is not blocked. NEVER "
    "write a mock or stand-in for the agent's API."
)


# HOST-PROCESS BLIND SPOT (2026-07-08). The sandbox has its own PID
# namespace: a process the USER started on the host (their `python app.py`)
# is invisible and unkillable from in here. `pkill -f app.py` therefore
# "succeeds" (exit 0, no output, nothing killed) and the model concludes it
# restarted the server — then keeps debugging against stale code. Observed
# twice in the chess session. Same family as the loopback blind spot: the
# tool must tell the truth, because the exit code lies.
_HOST_PROCESS_RE = re.compile(
    r"\b(?:pkill|killall)\b|\bkill\s+(?:-\w+\s+)?\$?\(?\s*(?:pgrep|pidof)\b",
    re.IGNORECASE)

#: a database CLIENT that found no server here: libpq (psql, psycopg, pg_dump…) and the MySQL client. 11-05 prod
#: probe PG: told "postgres_admin" was gone, the model ran `psql` in the sandbox, read the socket error, then spent 6
#: minutes trying to initdb/start its own server (su, rm -rf, rmtree guards) to the strike cap — the prompt's
#: DATABASES line was in the prompt and did not stop it. The fact goes where the model is looking: on the result.
#: Line-anchored to how the CLIENTS print it (review r19: a `grep`/`cat` of a log quoting the phrase, or two unrelated
#: lines joined by DOTALL, fired it), and a TCP failure only on loopback (a user's REMOTE database is not "absent").
#: Review r20: a pytest `E   ` line, SQLAlchemy's `(psycopg2.OperationalError) ` and psycopg 3's `connection failed: `
#: prefixes count; the REASON must say nothing answered (a `FATAL: role/password` failure is a server that DID answer).
_DB_LINE_PREFIX = (r"^(?:E\s+)?(?:psql: (?:error: )?|pg_\w+: (?:error: )?|[\w.]*(?:Error|Exception): "
                   r"|\([\w.]*(?:Error|Exception)\) |- host: '(?:localhost|127\.0\.0\.1|::1)'[^\n]{0,80}?: "
                   r"|connection failed: |Multiple exceptions: ){0,4}")
_LOOPBACK = r"(?:localhost|127\.0\.0\.1|::1)"
_NO_DB_SERVER_RE = re.compile(
    _DB_LINE_PREFIX +
    r"(?:connection to server (?:on socket [^\n]{0,160}?|at \"" + _LOOPBACK + r"\"[^\n]{0,80}?)failed: "
    r"(?:No such file or directory|Connection refused|Cannot assign requested address)"
    r"|could not connect to server: (?:No such file or directory|Connection refused)"
    r"|(?:\[Errno 111\] )?Connect call failed \('" + _LOOPBACK + r"', (?:5432|3306)"
    r"|(?:ERROR 2002 \(HY000\): |\(2002, \"?)Can't connect to local (?:MySQL )?server"
    r"|(?:/[\w./-]+|" + _LOOPBACK + r"):\d+ - no response\s*$"
    # review r21: MariaDB Connector/C drops "MySQL" and may say 2002 … (115) on TCP; mysqlclient prints it as a tuple
    r"|\(?(?:ERROR )?200[23],? (?:\(HY000\): )?\"?Can't connect to (?:MySQL )?server on '" + _LOOPBACK +
    r"(?::\d+)?' \((?:\[Errno (?:111|99)\]|111\b|115\b|10061\b)"
    r"|connect ECONNREFUSED (?:127\.0\.0\.1|::1):(?:5432|3306)\b)",
    re.IGNORECASE | re.MULTILINE)


def _no_db_server_note(output, context=None) -> str:
    """The note for a client that found no database server in the sandbox ("" when the output shows none)."""
    if not _NO_DB_SERVER_RE.search(str(output or "")):
        return ""
    _lab_route = ("create a disposable machine with the `lab` tool — `lab create` template pg-single (PostgreSQL 18, "
                  "member pg1, `psql -U postgres`) or mysql-single (member db1, `mysql -h127.0.0.1 -uroot -plab`) — and "
                  "run the SQL there with `lab exec`")
    if context is None:
        # review r19: a caller with no context cannot tell — never state "not configured" as a fact
        route = f"if the `lab` tool is available, {_lab_route}; if it is not, say that no database server is available"
    else:
        try:
            from .labs import labs_configured
            _labs = bool(labs_configured(context))
        except Exception:  # noqa: BLE001  — review r20: unknown is not "not configured"
            _labs = None
        route = (_lab_route if _labs else
                 f"if the `lab` tool is available, {_lab_route}; if it is not, say that no database server is available"
                 if _labs is None else
                 "no labs host is configured either, so a real PostgreSQL/MySQL server is not available here — say so")
    return ("\n\n--- 💡 NO DATABASE SERVER HERE ---\n"
            "There is no database server in this sandbox, by design. Do NOT try to install, initdb or start one here. "
            f"To run SQL on a real server: {route}. Plain data work: sqlite3 or pandas in `execute`.\n")


_HOST_PROCESS_NOTE = (
    "\n\n--- 💡 SANDBOX PID NAMESPACE ---\n"
    "You just tried to kill a process by name. The sandbox has its OWN pid "
    "namespace: processes the USER runs on their machine (their "
    "`python app.py`, their dev server) are NOT visible here, so this "
    "command killed nothing even if it exited 0 — do NOT conclude the "
    "server restarted. If the user runs the process, tell them: \"I've "
    "changed <file>; restart it to pick up the fix.\" Only processes YOU "
    "started inside this sandbox can be killed from here.\n"
    "------------------------"
)

# Non-blocking sibling of _AGENT_PORT_PROBE_MSG for the run-existing-file
# path. The egress guard vets `command` and inline `content`, but a script
# WRITTEN earlier (via file_system, whose writes are deliberately not gated)
# and executed by name was never checked — and hard-blocking it here would
# seal off legitimate apps whose source references the agent's URL by
# design. So the run proceeds, but when the source matches the probe
# signature (loopback URL + a network client) the result carries the ground
# truth, cutting off the "connection refused → the agent is down → write a
# mock server" misdiagnosis chain at the moment it would start.
_AGENT_PORT_FILE_NOTE = (
    "\n\n--- 💡 SANDBOX LOOPBACK BLIND SPOT ---\n"
    "This script references 127.0.0.1:8000 / :8088 and uses a network "
    "client. Inside the sandbox that loopback is the CONTAINER'S OWN — a "
    "connection failure to those ports here proves NOTHING about the "
    "agent/LLM on the host (both are fine). Do NOT conclude the server is "
    "down, and NEVER write a mock/stand-in for it. Verify endpoints with "
    "the `browser` tool or ask the user to run the check on THEIR machine.\n"
    "------------------------"
)


_MISSING_NAME_RES = (
    re.compile(r"can't open file ['\"]([^'\"]+)['\"]"),
    re.compile(r"No such file or directory:? ['\"]([^'\"]+)['\"]"),
    re.compile(r"(?m)^(?:[\w./-]+: )?([^\s:]+): No such file or directory"),
    re.compile(r"ENOENT[^'\"]*['\"]([^'\"]+)['\"]"),
    re.compile(r"Cannot find module ['\"]([^'\"]+)['\"]"),
)


def _missing_target_named_by(command: str, out: str) -> bool:
    """§4LO: the root re-run is for a WRONG-CWD miss — the file the COMMAND
    names was not where it looked. When the error names a file the command
    does not mention (a data file the program looked for at runtime: node's
    ENOENT, a script printing its own error), re-running elsewhere repeats
    the program's side effects and the root's exit 0 replaced the real
    failure (reproduced). Unknown (no name extractable) keeps the old rule."""
    names = [m.group(1) for rx in _MISSING_NAME_RES for m in rx.finditer(out or "")]
    if not names:
        return True
    cmd = str(command or "")
    if re.match(r"\s*(?:npm|yarn|pnpm)\b", cmd) and any(Path(n).name == "package.json" for n in names):
        return True                  # `npm start` names its package.json implicitly
    return any(Path(n).name and Path(n).name in cmd for n in names)


def _looks_like_file_not_found(out) -> bool:
    """Heuristic: did a command fail because the target file wasn't where it
    looked? Used to trigger the project-scoped → root cwd retry. Matches the
    common interpreter/shell messages (python, node, bash, cat, ...).

    Guard (2026-07-22): a Python TRACEBACK means the interpreter FOUND and RAN
    the script (it got past startup), so a "no such file" inside it is a
    RUNTIME data-file error raised AFTER the script's side effects — not a
    wrong-cwd miss. Re-running from another cwd would just repeat those side
    effects (DB inserts, appends, API calls). A genuine wrong-cwd miss is a
    startup error (`can't open file '…'`) with NO traceback, so this only
    suppresses the unsafe re-run."""
    if not isinstance(out, str):
        return False
    o = out.lower()
    if "traceback (most recent call last)" in o:
        return False
    return (
        "can't open file" in o
        or "no such file or directory" in o
        or "cannot find" in o
        or "not found" in o and ".py" in o
    )


# Command heads that MUTATE state (create/remove/write/clone). If one of
# these leads a non-final segment of a COMPOUND command, the file-not-found
# heals must NOT re-run the whole command: the earlier segment already
# committed its side effect, so the re-run fails differently (e.g.
# `mkdir out && cat out/x` → the re-run's `mkdir out` fails "File exists",
# `&&` short-circuits, and that non-file-not-found output is ADOPTED,
# replacing the real cat error with a misleading one).
_MUTATING_HEADS = frozenset({
    "mkdir", "touch", "rm", "rmdir", "mv", "cp", "ln", "git", "wget",
    "curl", "tee", "dd", "truncate", "install", "unzip", "tar", "chmod",
    "chown", "sed", "npm", "pip", "pip3", "apt", "apt-get", "make",
})


#: A final segment headed by one of these is never re-run elsewhere.
_DESTRUCTIVE_HEADS = frozenset({"rm", "rmdir", "unlink", "shred", "mv", "cp", "ln", "truncate", "dd",
                                "rsync", "chmod", "chown", "tee"})


def _rerun_unsafe(command: str) -> bool:
    """True when re-running the WHOLE command could repeat a side effect: a
    write redirect anywhere, a removing/moving verb anywhere (the final
    segment too — a failed `rm notes.md` re-run from the root deleted the
    root's copy), or a mutating verb in a non-final segment. Verbs are read
    per simple command through wrappers (`FOO=1`, `sudo -u x`, `timeout 5`,
    `nice`, `exec`, `xargs -0 -r`, `bash -c '…'` — fourth review: `xargs -0
    rm` hid the verb). A read-only compound (cd/pwd/ls/cat/echo…) is safe."""
    if not isinstance(command, str):
        return False
    segs = _shell_segments(command)
    for i, (head, args, raw, _x) in enumerate(segs):
        # a redirect that WRITES a file (`2>&1`, `>/dev/null` are not;
        # a `>` inside quotes is not a redirect)
        _unq = re.sub(r"'[^']*'|\"[^\"]*\"", "''", raw)
        if re.search(r"&>>?\s*(?!/dev/null)\S|(?<![0-9&])>>?(?!&)\s*(?!/dev/null)\S|(?<=[0-9])>>?\s*(?!&|/dev/null)\S", _unq):
            return True
        if head == "find":
            if any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir") for a in args):
                return True
        elif head in _DESTRUCTIVE_HEADS:
            return True
        elif i < len(segs) - 1 and head in _MUTATING_HEADS:
            return True
    return False


# §4BX/#5 — raw-Playwright anonymity backstop. The `browser` tool enforces the
# Tor proxy on every launch; `execute(stateful=True)` running raw Playwright
# does NOT — the model must thread `proxy=os.environ['TOR_PROXY']` itself
# (prompts.py teaches it; TOR_PROXY is injected into the cell env). A launch
# that omits it and then navigates to a PUBLIC host dials cleartext from the
# real IP — the exact leak the browser tool exists to prevent. We cannot
# block (a `file://`/loopback launch legitimately needs no proxy, and Tor
# can't route file://), so this is a loud NON-BLOCKING notice on the result,
# visible to the model, the verifier and the operator's stream.
_PW_LAUNCH_RE = re.compile(
    r"(async_playwright|(chromium|firefox|webkit)\s*\.\s*launch)", re.I)


def _proxyless_browser_launch(code: str) -> bool:
    """True if `code` launches Playwright WITHOUT referencing a proxy."""
    if not code or not _PW_LAUNCH_RE.search(code):
        return False
    low = code.lower()
    return ("proxy" not in low) and ("tor_proxy" not in low)


_PROXYLESS_BROWSER_NOTE = (
    "\n\n⚠ ANONYMITY: this cell launched Playwright with NO proxy while a "
    "Tor proxy was available (TOR_PROXY is set in the cell env). A `file://` "
    "or loopback navigation is fine, but any PUBLIC navigation just dialled "
    "cleartext from the real IP — the Tor-only rule. Add "
    "`proxy={'server': os.environ['TOR_PROXY']}` (+ the "
    "`--host-resolver-rules` DNS arg) to the launch, or use the native "
    "`browser` tool which enforces it."
)



#: The kernel liveness probe (§4GI, 2026-09-13). `sandbox_manager.execute`
#: wraps every command in `timeout -k 5s Ns sh -c '<cmd>'`, so the bare
#: `pgrep -f ipykernel_launcher` matched ITS OWN wrapper's command line —
#: exit 0 with no kernel, verified live by the review: a stale
#: `/workspace/.kernel.json` (the bind mount survives every container
#: restart) then read as "alive" forever and every stateful run failed
#: "Kernel did not become ready". The bracket idiom (memory
#: `the-probe-carries-its-own-pattern`): the regex `[i]pykernel_launcher`
#: matches the kernel's argv but not the literal `[i]pykernel_launcher`
#: that appears in the wrapper's own argv.
KERNEL_LIVENESS_PROBE = "pgrep -f '[i]pykernel_launcher'"


async def _kernel_alive(sandbox_manager, conn_file: str) -> bool:
    """True when the connection file exists AND a kernel process is alive."""
    _out, check_code = await asyncio.to_thread(sandbox_manager.execute, f"test -f {conn_file}")
    if check_code != 0:
        return False
    _out, pgrep_code = await asyncio.to_thread(sandbox_manager.execute, KERNEL_LIVENESS_PROBE)
    return pgrep_code == 0


_SELF_OVERWRITE_MSG = (
    "SYSTEM BLOCK: execute(filename='{filename}', content=…) would OVERWRITE '{filename}' with this script — and the "
    "script itself loads '{base}', so it is a test of that file, not its new source. Nothing was written or run. Put "
    "the test in its own file (filename='test_{base}') or omit `filename` to run it as a temporary script. (To "
    "REWRITE '{base}' itself, use file_system write.)")


def _self_loading_overwrite(filename, content, host_path) -> str:
    """ft live 11-18 (w6_rest): `execute(filename='ft/app.py', content=<a test that loads ft/app.py>)` WRITES the
    content to that filename before running it — the model's Flask app was replaced by its own test script. Refused
    only when the new script loads that very file (names its path, or imports its module) AND the file exists with
    other content: a script that tests a file is never meant to become it. The script text is read first, so a
    plain rewrite-and-run never touches the disk here."""
    new = str(content or "")
    rel = str(filename).replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    base = rel.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    imports = base.endswith(".py") and re.search(
        rf"^\s*(?:from\s+{re.escape(stem)}\s+import|import\s+{re.escape(stem)}\b)", new, re.M)
    # final review: a usage string, argparse `prog=`, a log line or FLASK_APP naming the file is not a LOAD of it —
    # only a loader call on that path is (importlib / runpy / exec(open) / SourceFileLoader)
    _q = r"""['"](?:\./)?(?:[\w./-]*/)?""" + re.escape(base) + r"""['"]"""
    loads = (re.search(r"(?:spec_from_file_location|run_path|SourceFileLoader|exec\s*\(\s*open|load_source)"
                       r"\s*\([^)\n]*" + _q, new) or imports
             or (base.endswith(".js") and re.search(r"""require\(\s*['"]\./""" + re.escape(stem) + r"""(?:\.js)?['"]""", new)))
    if not loads:
        return ""
    if re.fullmatch(r"test_\w+\.py|\w+_test\.py|\w+\.(?:test|spec)\.js", base) and not imports:
        return ""                 # review r1: a test file naming itself (`pytest.main(['test_app.py'])`) is its own run
    try:
        if not host_path.is_file():
            return ""
        old = host_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if not old.strip() or old.strip() == new.strip():
        return ""
    return _SELF_OVERWRITE_MSG.format(filename=filename, base=base)


async def tool_execute(filename: str = None, content: str = None, sandbox_dir: Path = None, sandbox_manager=None, scrapbook=None, args: list = None, memory_dir: Path = None, stateful: bool = False, command: str = None, workspace_model=None, container_workdir: str = None, **kwargs):
    # When a project is active, run from /workspace/projects/<id> so files
    # written via file_system (also scoped) read back. Passed ONLY when set,
    # so sandbox managers without a `workdir` param keep working unchanged.
    _workdir_kw = {"workdir": container_workdir} if container_workdir else {}
    # §4GI: the sandbox's egress must be Tor-only or cut off before anything
    # runs — the consumer the enforcement code never had.
    from ..sandbox.egress_gate import network_refusal as _egress_refusal
    _egress_block = _egress_refusal(sandbox_manager)
    if _egress_block is not None:
        pretty_log("Sandbox Egress", "refusing execute — egress unavailable",
                   level="ERROR", icon=Icons.SHIELD)
        return _egress_block
    # --- PARAMETER HALLUCINATION HEALING ---
    command = command or kwargs.get("cmd")
    filename = filename or kwargs.get("file") or kwargs.get("script") or kwargs.get("name")
    content = content or kwargs.get("code") or kwargs.get("script_content") or kwargs.get("text")

    # --- 🛡️ SANDBOX EGRESS GUARD (agent's own ports) ---
    # Checked BEFORE any execution, over both the shell command and inline
    # code content. Returns a plain instruction (not an EXIT CODE failure)
    # so the block teaches without burning a strike.
    _is_probe = bool(
        (command and _command_probes_agent_port(command))
        # Inline code EXECUTES — keep the strict any-match rule there; a
        # probe script always carries the URL, and file-writing belongs in
        # file_system.write (which is not gated on content).
        or (content and _AGENT_PORT_PROBE_RE.search(str(content)))
    )
    if _is_probe:
        pretty_log(
            "Sandbox Egress Guard",
            "Blocked in-sandbox probe of the agent's own port "
            "(127.0.0.1:8000/:8088 is the container's loopback, not "
            "the host) — returned ground-truth explanation instead",
            level="WARNING", icon=Icons.SHIELD,
        )
        # A REFUSAL — its own text says "command NOT executed". Returned as
        # a bare string it classified OK, so it booked competence success,
        # `op_outcomes.ok=True`, a foresight `ok=True`, and on a mutating
        # command it would have taken the world-changed branch and cleared
        # the pre-flight guard for work that never ran.
        return ToolOutcome.rejected(_AGENT_PORT_PROBE_MSG,
                                    reason_code="sandbox_egress_blocked")

    # --- 🛡️ HIJACK LAYER: CODE SANITIZATION ---
    
    # Helper for consistent error reporting. Emits the REAL exit code —
    # hardcoding 1 made a 124 timeout kill indistinguishable from a
    # program failure, so the model re-ran identical >=10-min commands.
    # The annotation rides AFTER the digits: downstream parsers match
    # `EXIT CODE:\s*(\d+)` and must keep working.
    def _format_error(msg, hint=None, exit_code=1, elapsed_s=None):
        _code_note = ""
        # the sandbox layer already said the CONTAINER died (§4MD M10): the
        # OOM advice ("reduce peak memory") would send the model to rewrite
        # correct code
        if _kill_is_oom(exit_code, elapsed_s) and not str(msg or "").lstrip().startswith("[SANDBOX INFRA ERROR]"):
            # §4HZ: a 137 this far under the budget is the OOM killer. Say
            # so — and say "memory", because the timeout advice (fewer
            # iterations, checkpoint) does nothing for a peak-memory kill.
            # ⚠ No "timeout"/"timed out" on THIS line: the strike classifier
            # (`tool_failure._RETRYABLE_PATTERNS`) reads the banner and would
            # book an OOM as a transient retry; the hint block below is
            # stripped before classification, the code note is not.
            _code_note = (f" (KILLED after {float(elapsed_s):.0f}s — far under the "
                          f"{_EXEC_TIMEOUT_S}s budget: out of memory, not the time limit)")
            _t_hint = (
                f"Exit 137 after only {float(elapsed_s):.0f}s means the kernel "
                f"OOM killer stopped the process — the sandbox is capped at "
                f"{_sandbox_mem_limit()} of memory — NOT the {_EXEC_TIMEOUT_S}s "
                f"timeout, and NOT slowness: timing imports or the computation "
                f"will not explain it. Re-running the identical command will "
                f"die the same way. Reduce PEAK MEMORY: process in chunks, "
                f"avoid full-grid meshgrid/broadcast temporaries, subsample "
                f"before plotting, `del` large arrays you are done with, or "
                f"use float32.")
            hint = f"{_t_hint}\n\n{hint}" if hint else _t_hint
        elif (exit_code == 124 and elapsed_s is not None
              and float(elapsed_s) < _EXEC_TIMEOUT_S - _OOM_KILL_MARGIN_S):
            # §4PH (R2-2): a 124 this far under the budget is the command's OWN
            # `timeout N` (or a program that exits 124) — not the sandbox
            # budget, and not transient. No "timed out / killed after" on this
            # line: that wording is the budget kill's.
            # r3 (reader): the jobs path also emits 124 for its own kills (a
            # wedged run, the log cap) — only a command that CARRIES a
            # `timeout N` is told it was its own wrapper
            if re.search(r"\btimeout\s+(?:-\S+\s+)*\d", str(command or "")):
                _code_note = (f" (exit 124 after {float(elapsed_s):.0f}s — the command's own `timeout` "
                              f"wrapper or the program itself, NOT the {_EXEC_TIMEOUT_S}s sandbox budget)")
                _t_hint = (f"Exit 124 after only {float(elapsed_s):.0f}s is the `timeout N` in your own "
                           f"command (or a program that exits 124), not the {_EXEC_TIMEOUT_S}s sandbox "
                           f"budget. Raise or drop that wrapper if the work legitimately needs longer; "
                           f"otherwise the program hung — re-running the identical command will time out "
                           f"the same way.")
            else:
                _code_note = (f" (exit 124 after {float(elapsed_s):.0f}s — under the {_EXEC_TIMEOUT_S}s "
                              f"sandbox budget, so NOT the budget kill)")
                _t_hint = (f"Exit 124 after only {float(elapsed_s):.0f}s is not the {_EXEC_TIMEOUT_S}s "
                           f"sandbox budget: a `timeout` inside the script, the program exiting 124 itself, "
                           f"or the job supervisor's own kill (a wedged run, the log cap) — read the output "
                           f"above. Re-running the identical command will end the same way.")
            hint = f"{_t_hint}\n\n{hint}" if hint else _t_hint
        elif exit_code in _TIMEOUT_KILL_CODES:
            _code_note = f" (timed out / killed after {_EXEC_TIMEOUT_S}s)"
            _t_hint = (
                f"Exit {exit_code} means the process was KILLED — most "
                f"likely it exceeded the {_EXEC_TIMEOUT_S}s execution "
                f"budget (137 can also be the kernel OOM killer). This is "
                f"NOT a bug in the code, and re-running the identical "
                f"command will die the same way. Shrink the workload "
                f"(fewer iterations / a data subset), checkpoint "
                f"intermediate results, or split the run into stages.")
            hint = f"{_t_hint}\n\n{hint}" if hint else _t_hint
        out = (f"--- EXECUTION RESULT ---\nEXIT CODE: {exit_code}{_code_note}"
               f"\nSTDOUT/STDERR:\n{msg}")
        if hint:
            out += f"\n\n--- 💡 DIAGNOSTIC HINT ---\n{hint}\n------------------------"
        # CARRY THE STATUS THROUGH THE ENVELOPE. `msg` may itself be a
        # `ToolOutcome` — three `execute` refusals are built as one and then
        # passed here — and interpolating it into an f-string produced a
        # plain `str`, so the status never reached the loop and the
        # migration was inert. It survived only because the envelope happens
        # to carry `EXIT CODE: 1`, which is the same accident that covered
        # it before the refactor.
        if isinstance(msg, ToolOutcome):
            return ToolOutcome(out, status=msg.status,
                               world_changed=msg.world_changed,
                               reason_code=msg.reason_code,
                               declared=msg.declared)
        return out

    if command:
        if not sandbox_manager: return _format_error("Error: Sandbox manager not initialized.")

        # §4UY C15: `command` AND `content` in one call — the file was silently dropped (only a log line said
        # so) while the model believed it written; the run ledger even credited it as written and run. Nothing
        # runs and nothing is written: the refusal says how to do both. A `filename` with no content beside a
        # command loses nothing and is ignored.
        if content:
            pretty_log("Execute Ambiguous Args", f"`command` and `content` (filename={filename!r}) in one call — "
                       f"refused, nothing written or run", level="WARNING", icon=Icons.WARN)
            return ToolOutcome.rejected(_format_error(
                "SYSTEM ERROR: execute got BOTH `command` and `content` — nothing was written and nothing ran. "
                "Either run a script: execute(filename=\"x.py\", content=...) (written, then run), or write the "
                "file with file_system(operation=\"write\") and then execute(command=...)."),
                reason_code="execute_ambiguous_args")

        # §4UY C1: the leading-`cd` rewrite is gone. It stripped `cd /opt|/home|/root|/app|/sandbox|/usr/src… &&`
        # and ran the rest in the project folder — `cd /opt/scratch && rm -rf *` became `rm -rf *` in the project,
        # while the original is safe (a failed `cd` stops the `&&` chain). 0 firings in the live log since 11-12.

        # Inline `-c` handling. Inline `python -c "<body>"` / `bash -c
        # "<body>"` is a recurring failure source: bash quote-escaping mangles
        # f-strings / nested quotes (observed live: a one-line fix spiralled
        # into ~15 turns of sed/-c quoting errors), and the model otherwise
        # crams whole scripts into one `-c`.
        #
        # We USED to REJECT every substantive inline body (>= 120 chars, >1
        # `;`, or import+nested-quotes) and force the model to re-emit as
        # file_system(write)+execute. That cost a turn EACH and fired on
        # perfectly well-formed scripts — a multi-statement one-liner or a long
        # body would run fine, yet got bounced. So now we split the action:
        #
        #   * AUTO-CONVERT a well-formed body into an in-sandbox file run. The
        #     body is extracted with `shlex` (POSIX word-splitting — byte-
        #     identical to what bash would hand the interpreter) and shipped
        #     into the container via base64, so bash quoting can't corrupt it.
        #     This is the CWD-Auto-fix philosophy applied one block down.
        #   * BLOCK only the two shapes a redirect genuinely helps: an
        #     acquired-skill call wrapped in `-c` (steer to the skill name),
        #     and a body bash can't even parse / a piped-or-redirected inline
        #     form we can't safely rewrite — re-emitting cleanly is the fix.
        #
        # The `-c` invocation is matched after an OPTIONAL command-separator
        # prefix (`cd <dir> && …`, `… ; …`) because the model almost always
        # works inside a project subdir (`cd projects/<id>/app && python3 -c
        # "<body>"`). True one-liners (short, single-statement, no import)
        # never enter this block and still run inline.
        _inline_py_match = re.search(
            r'(?P<sep>^|&&|\|\||;)\s*'
            r'(?P<interp>python3?|bash)\s+-c\s+'
            r'(?P<q>[\'"])(?P<body>.*)(?P=q)'
            r'(?P<post>\s*(?:\|.*)?)$',
            command, re.DOTALL,
        )
        if _inline_py_match:
            _body = _inline_py_match.group("body")
            _body_compact = _body.strip()
            _too_long = len(_body_compact) >= 120
            _too_many_stmts = _body_compact.count(";") >= 2
            # Acquired-skill wrap detection: the LLM trying to call a skill as a
            # module/file — `from foo import foo` or `acquired_skills/foo.py`.
            # Acquired skills are top-level tools; this ALWAYS blocks (with a
            # skill-name hint), never auto-converts, regardless of length.
            _skill_pattern = re.search(
                r'from\s+([a-zA-Z_][\w]*)\s+import\s+\1\b', _body_compact)
            _subprocess_pattern = re.search(
                r'acquired_skills/([a-zA-Z_][\w]*)\.py', _body_compact)
            _candidate_name = (_skill_pattern.group(1) if _skill_pattern
                               else _subprocess_pattern.group(1) if _subprocess_pattern
                               else None)
            _nested_quotes = ("'" in _body_compact and '"' in _body_compact)
            _has_import = bool(re.search(r'\b(?:from|import)\s+\w', _body_compact))
            _risky_import = _has_import and _nested_quotes
            # Only intervene when the body is substantive enough to deserve a
            # file; otherwise fall through and run it inline.
            if _too_long or _too_many_stmts or _risky_import or _candidate_name:
                reason = []
                if _too_long:
                    reason.append(f"body is {len(_body_compact)} chars (>= 120)")
                if _too_many_stmts:
                    reason.append(f"{_body_compact.count(';')} semicolons (multi-statement)")
                if _risky_import:
                    reason.append("import + nested quotes (bash-escape corruption risk)")
                if _candidate_name and not reason:
                    reason.append("looks like an acquired-skill call wrapped in -c")
                reason_str = "; ".join(reason)

                # Decide whether the body is SAFE to auto-run as a file.
                #
                # The corruption mode is an UNESCAPED copy of the outer
                # delimiter quote inside the body: bash splits the string there
                # (close + reopen), silently changing the body's meaning — e.g.
                # `-c "x = "literal""` turns the string literal into bare words.
                # If the captured body contains the delimiter quote NOT preceded
                # by a backslash, we cannot trust the inline form, so we BLOCK
                # and let the model re-emit cleanly. A properly ESCAPED `\"` is
                # fine (shlex unescapes it correctly), as is any body that never
                # embeds its delimiter. A skill wrap is never auto-run either.
                _delim = _inline_py_match.group("q")
                _quote_safe = (
                    not _candidate_name
                    and len(re.findall(r'(?<!\\)' + re.escape(_delim), _body)) == 0
                )

                # Extract the body exactly as bash would unquote it. A
                # ValueError from shlex means the quotes are unbalanced — the
                # body is genuinely corrupt, so we cannot safely auto-run it. We
                # also require the `-c` body to be the LAST token (no trailing
                # pipe/redirect we'd have to faithfully reconstruct); anything
                # fancier falls back to BLOCK.
                _bash_body = None
                _body_is_last = False
                if _quote_safe:
                    try:
                        _toks = shlex.split(command)
                    except ValueError:
                        _toks = None
                    if _toks:
                        for _i, _t in enumerate(_toks):
                            if (_t in ("python", "python3", "bash")
                                    and _i + 1 < len(_toks)
                                    and _toks[_i + 1] == "-c"
                                    and _i + 2 < len(_toks)):
                                _bash_body = _toks[_i + 2]
                                _body_is_last = (_i + 3 == len(_toks))
                                break

                # AST-RESCUE (2026-07-14). Mixed/unescaped quotes defeat the
                # shlex path above (`_quote_safe` False, or unbalanced-quote
                # ValueError), so a long repair one-liner with both quote
                # types got BLOCKED — in a live code-fix turn that cost a
                # strike plus a ~4-step write-probe detour, twice. But the
                # corruption the block protects against is bash mangling the
                # inline form; the auto-convert transport (base64 → file)
                # never lets bash see the body at all. So when the RAW
                # regex-captured body parses as valid Python it is almost
                # certainly the code the model intended — a transport-mangled
                # or mis-captured body essentially never parses — and we can
                # ship it as a file instead of blocking. Python only (bash
                # has no cheap host-side parse), never a skill wrap, and no
                # trailing pipe (regex `post` must be empty).
                if (_bash_body is None
                        and not _candidate_name
                        and _inline_py_match.group("interp") != "bash"
                        and not (_inline_py_match.group("post") or "").strip()):
                    try:
                        ast.parse(_body)
                    except (SyntaxError, ValueError):
                        pass
                    else:
                        _bash_body = _body
                        _body_is_last = True

                if _bash_body is not None and _body_is_last:
                    # --- AUTO-CONVERT → in-sandbox file run -------------------
                    _interp = _inline_py_match.group("interp")
                    _ext = "sh" if _interp == "bash" else "py"
                    # Each converted script gets its OWN EMPTY directory. A
                    # shared /tmp is an import minefield: Python puts the
                    # SCRIPT'S directory at sys.path[0] — ahead of PYTHONPATH
                    # — so any stale module in /tmp (e.g. an ai_model.py a
                    # previous scaffolding script left there) silently
                    # SHADOWS the workspace file of the same name. Observed
                    # live 2026-08-01 (req 50855398): two consecutive
                    # converted probes imported /tmp/ai_model.py instead of
                    # the project's, struck out the turn at 6/6, and the
                    # request aborted. An empty per-script dir has nothing
                    # to shadow with.
                    _dir = f"/tmp/_ghost_inline_{uuid.uuid4().hex[:8]}"
                    _path = f"{_dir}/inline.{_ext}"
                    _b64 = base64.b64encode(_bash_body.encode("utf-8")).decode("ascii")
                    _prefix = command[:_inline_py_match.start()]
                    _sep = _inline_py_match.group("sep")
                    # `cd proj && python3 /tmp/.../inline.py` — cd (if any) is
                    # preserved so the script's relative paths resolve like
                    # the inline form did; the file is written to an absolute
                    # /tmp path so the write itself is cwd-independent.
                    #
                    # For Python, `python -c "<body>"` puts the CWD on
                    # sys.path ('' as sys.path[0]); running the converted
                    # file puts the script dir there instead, so a body that
                    # imports a module sitting in the working directory
                    # breaks with ModuleNotFoundError (observed live,
                    # 2026-07 chess trace: `from chess_engine import Board`
                    # failed only BECAUSE of this conversion). Prepend the
                    # runtime CWD to PYTHONPATH to keep the inline form's
                    # import semantics; the empty script dir (above) keeps
                    # sys.path[0] harmless.
                    if _ext == "py":
                        _invoke = ('PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" '
                                   f"{_interp} {_path}")
                    else:
                        _invoke = f"{_interp} {_path}"
                    _run = f"{_prefix}{_sep} {_invoke}".strip()
                    command = (
                        f"mkdir -p {_dir} && "
                        f"printf %s {shlex.quote(_b64)} | base64 -d > {_path} "
                        f"&& {_run}"
                    )
                    pretty_log(
                        "Inline Script Auto-fixed",
                        f"Converted inline `-c` body → {_path} via base64 "
                        f"({reason_str}); running as a file to dodge bash "
                        f"quote corruption.",
                        icon=Icons.SHIELD,
                    )
                    # fall through to execution with the rewritten command
                else:
                    # --- BLOCK (skill-wrap, or a body we can't safely run) ----
                    pretty_log(
                        "Inline Script Blocked",
                        f"Rejected inline `-c` body ({reason_str})",
                        level="WARNING", icon=Icons.SHIELD,
                    )

                    # Targeted hint when the blocked body looks like a failed
                    # attempt to call an acquired skill (2026-04-24 EA
                    # incident). Acquired skills are TOP-LEVEL callable tools —
                    # steer the retry to invoke by name, not write+execute.
                    _skill_hint = ""
                    if _candidate_name:
                        _skill_hint = (
                            f"\n\nHINT: The body looks like an attempt to call "
                            f"the `{_candidate_name}` skill as a module / file. "
                            f"Acquired skills are TOP-LEVEL TOOLS. Invoke it "
                            f"directly as `{_candidate_name}(...)` — the same way "
                            f"you'd call a built-in like `web_search`. Don't "
                            f"import it, don't run its .py file, don't write a "
                            f"wrapper script."
                        )

                    return _format_error(ToolOutcome.rejected(
                        f"SYSTEM BLOCK: this inline `python -c '...'` / `bash -c "
                        f"'...'` form was rejected. Trigger: {reason_str}. Bash "
                        f"quote-escape corrupts f-strings and imported-symbol "
                        f"calls — this is the pattern that caused the previous "
                        f"run's SyntaxError. USE THIS PATTERN INSTEAD (both "
                        f"tool_calls can be stacked in the SAME turn):\n\n"
                        f"  1. file_system(operation=\"write\", path=\"probe.py\", "
                        f"content=...)\n"
                        f"  2. execute(filename=\"probe.py\")\n\n"
                        f"For TRUE one-liners (<120 chars, single statement, no "
                        f"imports) like `python3 -c \"import sys; print(sys.path)\"` "
                        f"the inline form is still allowed."
                        f"{_skill_hint}",
                        reason_code="inline_shell_form_rejected"))

        pretty_log("Shell Command", command, icon=Icons.TOOL_SHELL)

        # Pre-execution shell validator (roadmap phase 1.4). Runs
        # UNCONDITIONALLY — the destructive-command deny-list (`rm -rf /`,
        # `curl | sh`, fork bombs, etc.) is a safety boundary that must not
        # depend on the optional ``--enable-metacog`` uplift. Extra metacog
        # telemetry is emitted only when the bundle is wired. Validator
        # failures are non-fatal: a crashing validator must never break a
        # turn, so we fall through to running the command.
        _metacog = kwargs.get("_metacog_bundle")
        try:
            from .validators import validate_shell
            _shell_ok, _shell_reason = validate_shell(command)
            # Released-workspace shell guard (2026-07-25 round 2): the
            # file-tool write block was bypassable via `execute` (redirects,
            # rm/mv/sed -i). A command that references a RELEASED project's
            # path AND carries a mutation token is refused with the same
            # create_version steer. Read-only commands pass untouched.
            if _shell_ok:
                _rb = _released_shell_block(
                    kwargs.get("project_store"), command, workdir=str(container_workdir or ""))
                if _rb:
                    return _format_error(_rb)
                # Backgrounded-server guard (2026-07-30, §4G): a detached
                # server must go through manage_services, or it becomes an
                # invisible port-holder the registry can't see or stop.
                _yb = _youtube_download_block(command)          # §4PI (R3-M5)
                if _yb:
                    return ToolOutcome.rejected(_format_error(_yb), reason_code="youtube_download_blocked")
                _db = _daemonized_server_block(command) or _foreground_server_block(command)
                if _db:
                    # A refusal: the command is not run. Left as a string it
                    # classified OK, so a blocked daemon start booked a
                    # success and would have credited a world change.
                    return ToolOutcome.rejected(
                        _format_error(_db),
                        reason_code="daemonized_server_blocked")
                # §4UY C6/C21: nothing it removes or overwrites is lost (one rule with file_system)
                await asyncio.to_thread(_keep_versions_before_run, command, sandbox_dir,
                                        str(container_workdir or ""))
        except Exception as _vexc:
            logging.getLogger("GhostAgent").debug("shell validator crashed: %s", _vexc)
            _shell_ok, _shell_reason = True, ""
        if not _shell_ok:
            if _metacog is not None and getattr(_metacog, "enabled", False):
                try:
                    from ..core.metacog_log import (
                        emit as _mc_emit, Subsystem as _mc_ss, LEVEL_WARN,
                    )
                    _mc_emit(
                        _mc_ss.VALID, level=LEVEL_WARN,
                        verdict="block", tool="shell",
                        reason=_shell_reason, command_head=command[:60],
                    )
                    _metacog.count(validator_block=True)
                except Exception:
                    pass
            return _format_error(ToolOutcome.rejected(
                f"SYSTEM BLOCK: shell command rejected by pre-execution "
                f"validator: {_shell_reason}. The command was not run. "
                f"IMPORTANT: retry the SAME verification intent in an allowed "
                f"form — do NOT substitute a weaker check. Allowed patterns: "
                f"write the command's output to a file first and inspect it "
                f"(`curl … -o /tmp/out.json` then `python3 -m json.tool "
                f"/tmp/out.json`), or put the logic in a script via "
                f"file_system + execute (urllib for HTTP checks works well).",
                reason_code="shell_command_rejected"))

        # Execute the command securely using the sandbox manager's built-in execute wrapper.
        # 600s (not 300s): real project work — training a small model, running a
        # full test suite, installing a package, a multi-stage build — routinely
        # needs more than five minutes, and a premature kill burned turns
        # re-running from scratch. The hard kill (-k 5s) still guarantees the
        # process can't hang the container indefinitely.
        import time as _time
        _t0 = _time.time()
        cmd_str = _bash_c(command)
        # spill_large_output: same small-view + full-log-to-file policy as the
        # script path (the bash branch previously had NO tool-level trim, so a
        # noisy direct command dumped its whole 256 KB into context).
        output, exit_code, promoted_job = await _run_in_sandbox(
            sandbox_manager, cmd_str, timeout=_EXEC_TIMEOUT_S,
            label=command[:120],
            project_id=_project_id_from_workdir(container_workdir),
            spill_large_output=True, **_workdir_kw)
        if promoted_job is None:
            exit_code, output, _pipe_codes = _read_shell_run(                             # §4UY C9/C10
                command, exit_code, output, await _run_failures(cmd_str, sandbox_manager))
        else:
            output, _pipe_codes = _take_pipestatus(output)
            exit_code = _normalise_exit(command, exit_code, output)
        if promoted_job is None and _bg_detached(command):
            output = str(output or "") + _BG_NOTE                        # §4PH (R2-4)
        # Root fallback for project-scoped commands. When a project is active
        # the command runs from /workspace/projects/<id>, but the model may
        # reference a file that lives at the sandbox ROOT — e.g. one it wrote
        # in the SAME turn it switched into the project, before the switch
        # took effect (observed: file_system write emitted just before the
        # `switch` tool call → file at root, then `python3 chart.py` from the
        # scoped cwd fails). If the scoped run failed with a file-not-found
        # signature, retry once from the root. Safe: "can't open file" means
        # nothing executed, so there are no partial side effects to repeat.
        # Scope-flap heal (2026-07-18): the INVERSE of the heal below. This
        # call ran WITHOUT a project workdir — stateful sessions opt out of
        # scoping by design (kernel conn file pinned to /workspace), and a
        # transient scope clear does the same — while a project is actually
        # bound. A relative `python3 extract_data.py` then resolved against
        # /workspace and died even though the file sits in the project dir.
        # Live request 9c9b75aa burned FOUR strikes on this exact shape (the
        # model even ran `pwd` — a later, scoped call — saw the right cwd,
        # retried, failed again): the remap heal below is gated on the
        # workdir kwarg, i.e. disabled precisely when this happens. Re-derive
        # the project dir from the workspace-model mirror and retry once
        # from there, with a note naming the actual mechanism.
        if exit_code != 0 and exit_code not in _TIMEOUT_KILL_CODES and not _workdir_kw and _looks_like_file_not_found(output) and _missing_target_named_by(command, output) and not _rerun_unsafe(command):
            _flap_pid = ""
            try:
                _flap_pid = str(getattr(
                    workspace_model, "current_project_id", "") or "").strip().lower()
            except Exception:
                _flap_pid = ""
            # Canonical 12-hex project ids only — a mocked/garbage mirror
            # must not fabricate a workdir.
            if not re.fullmatch(r"[0-9a-f]{12}", _flap_pid or ""):
                _flap_pid = ""
            if _flap_pid:
                _proj_wd = f"/workspace/projects/{_flap_pid}"
                pretty_log(
                    "Project Path Heal",
                    f"file-not-found with no project cwd while project "
                    f"'{_flap_pid}' is bound — retrying from {_proj_wd}",
                    level="INFO", icon=Icons.SHIELD,
                )
                _fh_out, _fh_code = await asyncio.to_thread(
                    sandbox_manager.execute, cmd_str, timeout=_EXEC_TIMEOUT_S,
                    spill_large_output=True, workdir=_proj_wd)
                _fh_code, _fh_out, _fh_pipe = _read_shell_run(command, _fh_code, _fh_out,
                                                               await _run_failures(cmd_str, sandbox_manager))
                if _fh_code == 0 or not _looks_like_file_not_found(_fh_out):
                    output, exit_code = _fh_out, _fh_code
                    output = (output or "") + (
                        f"\n[SYSTEM NOTE: the original attempt ran at /workspace "
                        f"because this call opted out of project scoping (most "
                        f"often stateful=true — the kernel session is pinned to "
                        f"/workspace). It was retried from {_proj_wd} (the active "
                        f"project) and that is the output above. For project "
                        f"files either drop stateful, or use the absolute "
                        f"project path.]")

        if exit_code != 0 and exit_code not in _TIMEOUT_KILL_CODES and _workdir_kw and _looks_like_file_not_found(output) and _missing_target_named_by(command, output) and not _rerun_unsafe(command):
            # Absolute-path variant first: when the command names files under
            # `/workspace/...` a workdir change can't help (absolute paths are
            # cwd-independent) — but the model almost always means the ACTIVE
            # PROJECT's copy: file_system heals `/workspace/game.py` into the
            # project dir, so the file the model just wrote/read really lives
            # at /workspace/projects/<id>/game.py. Left alone, this asymmetry
            # produced a 10+ turn ENOENT loop (2026-07 chess trace: `cd
            # /workspace && python3 game.py`, `python3 /workspace/game.py`).
            # Retry once with `/workspace` remapped to the scoped workdir.
            # Safe: file-not-found means nothing executed. Skipped when the
            # command already targets the scoped dir explicitly.
            _remapped = None
            if (container_workdir and container_workdir != "/workspace"
                    and container_workdir not in command
                    and re.search(r"/workspace(?![\w-])", command)):
                _remapped = re.sub(r"/workspace(?![\w-])", container_workdir, command)
            if _remapped is not None:
                pretty_log(
                    "Project Path Remap",
                    f"File not found under /workspace — retrying with paths "
                    f"remapped to {container_workdir}. Original: {command[:80]}",
                    level="INFO", icon=Icons.SHIELD,
                )
                # spill_large_output on the retry too — dropping it meant a
                # noisy retried command dumped its full output into context,
                # the exact flood the primary call's spill mode prevents.
                _re_cmd = _bash_c(_remapped)
                _re_out, _re_code = await asyncio.to_thread(
                    sandbox_manager.execute,
                    _re_cmd, timeout=_EXEC_TIMEOUT_S,
                    spill_large_output=True, **_workdir_kw)
                _re_code, _re_out, _re_pipe = _read_shell_run(_remapped, _re_code, _re_out,
                                                              await _run_failures(_re_cmd, sandbox_manager))
                if _re_code == 0 or not _looks_like_file_not_found(_re_out):
                    output, exit_code, _pipe_codes = _re_out, _re_code, _re_pipe
                    # Teach on EVERY adopted remap, not just clean exits.
                    # The note used to ride only exit_code == 0 — so when a
                    # remapped run failed for its own reasons (req A3: the
                    # parser crashed on assembly expressions), the model was
                    # never told its paths were being rewritten and kept
                    # misaddressing /workspace for all 22 turns. The lesson
                    # is about the PATH, not the run outcome.
                    output = (output or "") + (
                        f"\n[SYSTEM NOTE: paths under /workspace were remapped to "
                        f"{container_workdir} (the active project's workspace) and "
                        f"the command ran there"
                        + ("" if exit_code == 0 else
                           " (it failed for reasons UNRELATED to the path)")
                        + ". Reference project files by BARE relative path "
                          "instead.]")
            else:
                # Root-cwd retry (relative paths, file at the sandbox root).
                _re_out, _re_code = await asyncio.to_thread(
                    sandbox_manager.execute, cmd_str, timeout=_EXEC_TIMEOUT_S,
                    spill_large_output=True)
                _re_code, _re_out, _re_pipe = _read_shell_run(command, _re_code, _re_out,
                                                              await _run_failures(cmd_str, sandbox_manager))
                # Same adoption guard as the two heals above: a scoped
                # script that RAN and then died on a missing data file
                # matches the file-not-found signature too, and adopting
                # unconditionally replaced its real traceback with the
                # root run's bogus "can't open file '<script>'".
                if _re_code == 0 or not _looks_like_file_not_found(_re_out):
                    output, exit_code = _re_out, _re_code
                    # Say WHERE it ran — silently succeeding from the root
                    # taught the model nothing, so its next command used the
                    # scoped-relative path again and failed again.
                    output = (output or "") + (
                        "\n[SYSTEM NOTE: the target was not in the active "
                        "project's workspace; this ran from the sandbox ROOT "
                        "(/workspace)"
                        + ("" if exit_code == 0 else
                           " and failed there for reasons UNRELATED to the path")
                        + ". Reference it as /workspace/<path>, or "
                          "move it into the project.]")
        _dt = _time.time() - _t0

        # Budget reached with the command still alive and still working: it
        # was DETACHED, not killed. Return before every failure-shaped path
        # below — none of them apply to a run that has not finished.
        if promoted_job is not None:
            _record_promoted_command(workspace_model, command, _dt, promoted_job)
            _register_promoted_job(kwargs.get("job_registry"), promoted_job,
                                   command)
            from .runs import record_promoted_job
            return _promoted_result(promoted_job, int(_dt), output,
                                    record_promoted_job(kwargs.get("run_context"), promoted_job))

        # Match-style commands (grep family, pgrep) exit 1 to mean "no
        # matches" — a perfectly successful query with an empty result, not a
        # failure. Reporting it as `EXIT CODE: 1` made the agent loop count it
        # as an execution strike ("[SYSTEM ERROR]: Process failed (Exit 1)
        # with no output", observed live 2026-07: a grep proving a fix had
        # landed was scored as a failure). Normalize to a success-shaped
        # result that says what the exit code actually meant. Only fires on
        # exit EXACTLY 1 with NO output — grep signals real errors with exit 2
        # and prints to stderr. NB: the docker sandbox layer substitutes
        # "[SYSTEM ERROR]: Process failed (Exit N) with no output." for empty
        # output before we see it, so that sentinel counts as no output too.
        #
        # Computed BEFORE the telemetry write below so a successful empty
        # grep isn't logged to workspace history as a failed command.
        _out_stripped = (output or "").strip()
        _no_output = (not _out_stripped
                      or _out_stripped == f"[SYSTEM ERROR]: Process failed (Exit {exit_code}) with no output.")
        _grep_no_match = False
        # …and only when every stage BEFORE the grep succeeded (§4LO)
        if exit_code == 1 and _no_output and not _an_upstream_stage_failed(command, _pipe_codes):
            _tail_seg = _shell_pipelines(command)[-1][-1]
            try:
                _tail_toks = shlex.split(_tail_seg)
            except ValueError:
                _tail_toks = _tail_seg.split()
            while _tail_toks and re.fullmatch(r"[A-Za-z_]\w*=.*", _tail_toks[0]):
                _tail_toks = _tail_toks[1:]
            _tail_head = Path(_tail_toks[0]).name if _tail_toks else ""
            _grep_no_match = _tail_head in _EXIT1_MEANS_NO_MATCH

        # Workspace command-outcome capture: record significant runs so
        # the user can later ask "what commands did I run yesterday?" or
        # "did that pipeline ever finish?". Significance gate: failed
        # commands always, or successful runs that took >5s. Skip noise.
        # A grep-no-match counts as a SUCCESS (exit 0) for this purpose.
        _eff_exit = 0 if _grep_no_match else exit_code
        if workspace_model is not None and getattr(workspace_model, "enabled", False):
            try:
                if _eff_exit != 0 or _dt >= 5.0:
                    workspace_model.record_command_outcome(
                        command=command, exit_code=int(_eff_exit),
                        duration_seconds=float(_dt),
                        note=("failed" if _eff_exit != 0 else "long-running"),
                    )
            except Exception:  # noqa: BLE001
                pass

        if _grep_no_match:
            return (
                "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n"
                f"(no matches — `{_tail_head}` exited 1, which for this command "
                "means the pattern/target was NOT FOUND, not that the command "
                "failed.)"
            )

        # A name-based kill in the sandbox can never touch a host process.
        # Append the ground truth to BOTH outcomes — a "successful" pkill
        # that killed nothing is the dangerous case (the model believes it
        # restarted the user's server).
        _host_proc_note = (_HOST_PROCESS_NOTE
                           if _HOST_PROCESS_RE.search(command) else "")
        if _host_proc_note:
            pretty_log(
                "Sandbox PID Namespace",
                "Name-based kill in the sandbox cannot reach host processes "
                "— appended ground truth to the result",
                level="WARNING", icon=Icons.SHIELD,
            )

        _db_note = _no_db_server_note(output, kwargs.get("run_context"))      # 11-05: also on exit 0 (`psql … || true`)
        if _db_note:
            pretty_log("Sandbox No DB", "a database client found no server — appended the lab route to the result",
                       level="INFO", icon=Icons.SHIELD)

        if exit_code != 0:
            return _append_note(
                _format_error(
                    output or f"Process failed (Exit {exit_code}) with no output.",
                    exit_code=exit_code, elapsed_s=_dt,
                ), _host_proc_note + _db_note)

        return (f"--- COMMAND RESULT ---\nEXIT CODE: {exit_code}\n"
                f"STDOUT/STDERR:\n{output}{_host_proc_note}{_db_note}")

    is_ephemeral = False
    if content and not filename and not command:
        filename = f".ephemeral_{uuid.uuid4().hex[:8]}.py"
        is_ephemeral = True
    if not filename and not command:
        return _format_error("SYSTEM ERROR: You must provide either a 'command' or a 'filename'. If you just want to run code, provide 'content' and a temporary filename will be used.")
    if not sandbox_dir or not sandbox_manager: return _format_error("Error: Sandbox manager not initialized.")

    # 0. VALIDATION: Ensure we are only executing scripts
    ext = str(filename).split('.')[-1].lower()
    if ext not in ["py", "sh", "js"]:
        pretty_log("Execution Blocked", f"Invalid extension: .{ext}", level="WARNING", icon=Icons.SHIELD)
        tip = "To save data files or web pages (HTML/CSS/JS), use 'file_system(operation=\"write\", ...)' instead."
        return _format_error(f"SYSTEM ERROR: The 'execute' tool is ONLY for running scripts (.py, .sh, .js).\nSYSTEM TIP: {tip}")

    try:
        host_path = _get_safe_path(sandbox_dir, filename)
    except ValueError as ve:
        return _format_error(str(ve))

    _clobber = _self_loading_overwrite(filename, content, host_path) if (content and not is_ephemeral) else ""
    if _clobber:
        pretty_log("Execution Blocked", f"'{filename}' would be overwritten by a script that loads it",
                   level="WARNING", icon=Icons.SHIELD)
        return ToolOutcome.rejected(_format_error(_clobber), reason_code="self_loading_overwrite")   # review r1

    is_new_code = True
    _probe_note = ""
    # Non-blocking anonymity backstop for raw-Playwright in stateful execute.
    _proxyless_browser_note = ""
    if (stateful and content and getattr(sandbox_manager, "tor_proxy", None)
            and _proxyless_browser_launch(content)):
        _proxyless_browser_note = _PROXYLESS_BROWSER_NOTE
        pretty_log(
            "Anonymity",
            "raw Playwright launched with no proxy while Tor is available "
            "— flagged on the result (public navigation would leak)",
            level="WARNING", icon=Icons.WARN)
    if not content:
        if not host_path.exists():
            return _format_error(f"SYSTEM ERROR: File '{filename}' does not exist. You must provide 'content' to create it.")
        # Read existing source UTF-8-tolerantly: the main call path has no
        # try/except around tool_execute, so a raw UnicodeDecodeError here
        # would propagate out of the tool instead of a formatted error.
        try:
            content = await asyncio.to_thread(
                lambda: host_path.read_text(encoding="utf-8", errors="replace"))
        except OSError as _read_err:
            return _format_error(f"SYSTEM ERROR: could not read '{filename}': {_read_err}")
        is_new_code = False
        # Existing-file source was never seen by the egress guard above
        # (it only vets `command`/inline `content`). Don't block — a legit
        # app may reference the agent's URL by design — but make the
        # result carry the loopback ground truth so a failed connect can't
        # start the mock-server misdiagnosis chain.
        if (_AGENT_PORT_PROBE_RE.search(content)
                and _NET_CLIENT_RE.search(content)):
            _probe_note = _AGENT_PORT_FILE_NOTE
            pretty_log(
                "Sandbox Loopback Note",
                f"'{filename}' references the agent's ports and a net "
                f"client — running it, with the loopback ground truth "
                f"appended to the result",
                level="INFO", icon=Icons.SHIELD,
            )

    # 1. Holistic Sanitization
    content, syntax_error = await asyncio.to_thread(sanitize_code, content, str(filename))
    
    if syntax_error:
        pretty_log("Sanitization Failed", syntax_error, level="WARNING", icon=Icons.BUG)
        
        # HTML/Web guard for when the LLM pastes raw HTML into a Python script
        if "<html" in content.lower() or "body {" in content.lower() or "<div" in content.lower() or "font-family:" in content.lower():
            html_tip = "SYSTEM TIP: It looks like you are trying to write HTML, CSS, or JS intended for the browser. DO NOT use the 'execute' tool to create web pages. Use the 'file_system' tool with operation='write' to save the code directly to a file."
            return _format_error(f"Syntax Error Detected: {syntax_error}\n\n{html_tip}")
            
        # We block execution if syntax is clearly invalid to save a roundtrip
        return _format_error(f"Syntax Error Detected: {syntax_error}\nPlease fix the code and try again.")
        
    # 2. Hard Sandbox Guard against Native Tool Imports
    # The LLM frequently hallucinates that native JSON tools are importable Python modules.
    # The list holds distinctive tool names that are never plausible local
    # module names. The generic ones that collided with legitimate code —
    # `file_system` (very common module name), `scratchpad`, `recall`
    # — were REMOVED: a strong model importing a local `file_system.py` helper
    # was getting its valid code hard-blocked. `browser` is KEPT: it's a
    # prominent native tool the model does hallucinate importing, and a dedicated
    # guard test pins it (the Pyodide-`browser` false positive is rare in this
    # sandbox). The rest are unambiguous tool names.
    if ext == "py":
        forbidden_modules = [
            "knowledge_base", "system_utility", "manage_tasks",
            "web_search", "fact_check", "deep_research",
            "vision_analysis",
            "update_profile", "browser",
        ]
        
        # Check for direct imports or pip installs
        for mod in forbidden_modules:
            # We look for simple patterns: import mod, from mod import, !pip install mod
            if re.search(rf"\bimport\s+{mod}\b", content) or re.search(rf"\bfrom\s+{mod}\s+import\b", content) or re.search(rf"pip\s+install\s+{mod}\b", content):
                pretty_log("Sandbox Guard Invoked", f"Blocked hallucinated import: {mod}", level="WARNING", icon=Icons.SHIELD)
                return _format_error(
                    f"SYSTEM ERROR: FORBIDDEN IMPORT DETECTED -> '{mod}'\n"
                    f"CRITICAL: '{mod}' is a Native JSON Tool, NOT a Python module.\n"
                    f"You CANNOT import it or install it in this sandbox.\n"
                    f"To use '{mod}', you MUST stop writing code and call the JSON tool directly!"
                )

    # 3. Final Trim
    clean_content = content.strip()
    
    # Extract rel_path early for wrapper filename generation.
    # Derived from the HEALED host path so the container-side run targets the
    # SAME file the write below lands on. The old hand-rolled strip
    # (lstrip("/") + drop a "sandbox/" prefix) had diverged from
    # _get_safe_path's healing on both sides: filename="/workspace/game.py"
    # wrote <proj>/game.py but ran `python3 -u workspace/game.py` → an ENOENT
    # loop (2026-07 chess trace), and "sandbox/foo.py" wrote
    # <sandbox>/sandbox/foo.py but ran "foo.py" (a stale or missing file).
    if not filename: return _format_error("Error: filename is required.")
    if isinstance(host_path, Path):
        try:
            rel_path = host_path.relative_to(Path(sandbox_dir).resolve()).as_posix()
        except (TypeError, ValueError, OSError):
            # Root-anchored resolution (file_system 2026-07-14): under a
            # project-scoped sandbox, _get_safe_path may legitimately resolve
            # the filename to the OUTER root (e.g. "/workspace/x.py" whose
            # file exists only at the root — an execute-created tree). The
            # scoped relative_to then raises, and the legacy lstrip fallback
            # minted a phantom "workspace/x.py" relative path → ENOENT from
            # the scoped cwd. Run such a file via its container-ABSOLUTE
            # path instead, which is cwd-independent and names the same file
            # the read/write above touched.
            rel_path = None
            try:
                _sd = Path(sandbox_dir).resolve()
                if _sd.parent.name == "projects":
                    _root_rel = host_path.relative_to(
                        _sd.parent.parent.resolve()).as_posix()
                    rel_path = f"/workspace/{_root_rel}"
            except (TypeError, ValueError, OSError):
                pass
            if rel_path is None:
                rel_path = str(filename).lstrip("/")
    else:
        # _get_safe_path was stubbed (unit tests) — legacy derivation.
        rel_path = str(filename).lstrip("/")

    # 4. Stateful Execution Wrapper (Jupyter Kernel)
    exec_content = clean_content
    exec_rel_path = rel_path
    wrapper_line_offset = 0
    is_jupyter = False

    if ext == "py" and stateful:
        pretty_log("Stateful Execution", "Routing to Persistent Jupyter Kernel", icon=Icons.TOOL_CODE)
        conn_file = "/workspace/.kernel.json"

        # Check if kernel is running (prevent stale file deadlocks)
        check_code = 0 if await _kernel_alive(sandbox_manager, conn_file) else 1

        if check_code != 0:
            # Clean up dead connection file if it exists
            await asyncio.to_thread(sandbox_manager.execute, f"rm -f {conn_file}")
            pretty_log("Jupyter Kernel", "Booting up persistent kernel...", icon=Icons.SYSTEM_BOOT)
            user_id = os.getuid() if hasattr(os, 'getuid') else 1000
            group_id = os.getgid() if hasattr(os, 'getgid') else 1000
            
            proxy_env = {}
            if getattr(sandbox_manager, 'tor_proxy', None):
                # The Sandbox runs an isolated Tor daemon internally to bypass Docker bridge network routing issues
                proxy_env = {"TOR_PROXY": "socks5://127.0.0.1:9050"}
                
            exec_kwargs = {
                "workdir": "/workspace",
                "detach": True,
                "environment": proxy_env
            }
            if sys.platform != "darwin":
                exec_kwargs["user"] = f"{user_id}:{group_id}"
                
            # Guarded: `container` is None when the sandbox died or was
            # never started; the bare attribute access raised an unhandled
            # AttributeError out of tool_execute instead of the formatted
            # tool error every other path returns.
            _container = getattr(sandbox_manager, "container", None)
            if _container is None:
                return _format_error(
                    "Error: Sandbox container is not running — cannot boot the stateful Jupyter kernel.",
                    hint="Retry without stateful=True, or re-run the command so the sandbox restarts.",
                )
            try:
                # ⚠ OFF THE EVENT LOOP, WITH A DEADLINE (§4BW MAJOR). This
                # was a synchronous `_container.exec_run(...)` inside an async
                # function — a wedged daemon here froze the WHOLE agent event
                # loop (every tool, every turn), the exact class the
                # 2026-07-22 `_exec_run` wrapper removed elsewhere; the fix
                # never reached this call. Route through that wrapper (hard
                # client-side deadline) when the manager exposes it, and run
                # it on a thread so a hang cannot block the loop.
                _boot = getattr(sandbox_manager, "_exec_run", None)
                if callable(_boot):
                    await asyncio.to_thread(
                        _boot,
                        f"python3 -m ipykernel_launcher -f {conn_file}",
                        deadline_s=60, **exec_kwargs)
                else:
                    await asyncio.to_thread(
                        _container.exec_run,
                        f"python3 -m ipykernel_launcher -f {conn_file}",
                        **exec_kwargs)
            except Exception as _boot_err:
                return _format_error(
                    f"Error: Failed to boot the stateful Jupyter kernel: {_boot_err}",
                    hint="Retry without stateful=True, or re-run the command so the sandbox restarts.",
                )
            
            for _ in range(20):
                out_chk, c_code = await asyncio.to_thread(sandbox_manager.execute, f"test -f {conn_file}")
                if c_code == 0:
                    break
                await asyncio.sleep(0.5)

        jupyter_runner_code = """import sys, json, re, time
from jupyter_client import BlockingKernelClient
import queue

with open(sys.argv[1], 'r') as f:
    code = f.read()

kc = BlockingKernelClient(connection_file='/workspace/.kernel.json')
kc.load_connection_file()
kc.start_channels()

# Block until the kernel heartbeat is established before doing anything
# else. Without this call, `kc.is_alive()` returns False during the
# first ~1s of every brand-new kernel's lifetime (heartbeat hasn't
# had a chance to exchange yet). The `except queue.Empty` path below
# reads `kc.is_alive()` as a "kernel died?" probe, so the startup
# race produced a spurious `[SYSTEM ERROR: Kernel Terminated Abruptly]`
# on the FIRST stateful execute of every session — agent would
# conclude its code killed the kernel when in fact the code hadn't
# run yet. `wait_for_ready` closes the race; if it times out the
# kernel really is broken.
try:
    kc.wait_for_ready(timeout=10)
except RuntimeError as _ready_err:
    # wait_for_ready raises RuntimeError when the kernel didn't
    # handshake in time. Distinct from "kernel died mid-execution" —
    # surface a separate error so the agent knows to check the
    # kernel launch, not the code it tried to run.
    sys.stdout.write("\\n[SYSTEM ERROR: Kernel did not become ready within 10s: " + str(_ready_err) + "]\\n")
    sys.stdout.flush()
    try: kc.stop_channels()
    except: pass
    sys.exit(1)

msg_id = kc.execute(code)
has_error = False
start_time = time.time()

def _interrupt_kernel():
    # SIGINT to THIS kernel (the one started on /workspace/.kernel.json) — a
    # timed-out cell otherwise ran on and queued every later stateful call
    # behind it (§4LO). KeyboardInterrupt in the cell; state survives.
    import os, signal
    for _pid in os.listdir('/proc'):
        if not _pid.isdigit():
            continue
        try:
            with open('/proc/' + _pid + '/cmdline', 'rb') as _f:
                _cl = _f.read()
        except OSError:
            continue
        if b'ipykernel' in _cl and b'/workspace/.kernel.json' in _cl:
            try:
                os.kill(int(_pid), signal.SIGINT)
            except OSError:
                pass

while True:
    # the 5-minute wall clock is checked on EVERY pass — a cell that keeps
    # printing never reached the queue.Empty branch and was never stopped
    if time.time() - start_time > 295:
        _interrupt_kernel()
        sys.stdout.write("\\n[SYSTEM ERROR: Kernel Timeout. Execution exceeded 5 minutes — the cell was interrupted; kernel state is kept]\\n")
        sys.stdout.flush()
        has_error = True
        break
    try:
        # Short timeout so we can check if the kernel died (e.g. os._exit) or if we hit the 5 min limit
        msg = kc.get_iopub_msg(timeout=1)
        if msg['parent_header'].get('msg_id') != msg_id:
            continue

        msg_type = msg['header']['msg_type']
        content = msg['content']

        if msg_type == 'stream':
            sys.stdout.write(content['text'])
            sys.stdout.flush()
        elif msg_type in ('execute_result', 'display_data'):
            if 'text/plain' in content.get('data', {}):
                sys.stdout.write(content['data']['text/plain'] + "\\n")
                sys.stdout.flush()
        elif msg_type == 'error':
            tb = "\\n".join(content['traceback'])
            clean_tb = re.sub(r'\\x1b\\[[0-9;]*[a-zA-Z]', '', tb) # Strip ANSI color codes
            sys.stdout.write(clean_tb + "\\n")
            sys.stdout.flush()
            has_error = True
        elif msg_type == 'status' and content['execution_state'] == 'idle':
            break
    except queue.Empty:
        if not kc.is_alive():
            sys.stdout.write("\\n[SYSTEM ERROR: Kernel Terminated Abruptly (Did the script call os._exit()?)]\\n")
            sys.stdout.flush()
            has_error = True
            break
        continue

try:
    kc.stop_channels()
except:
    pass

if has_error:
    sys.exit(1)
"""
        runner_path = _get_safe_path(sandbox_dir, ".jupyter_runner.py")
        await asyncio.to_thread(runner_path.write_text, jupyter_runner_code)
        
        exec_content = clean_content
        exec_rel_path = str(Path(rel_path).parent / f"._{Path(rel_path).name}")
        wrapper_line_offset = 0
        is_jupyter = True

    # ----------------------------------------
    if stateful:
        pretty_log("Execution Task", f"{filename} [STATEFUL]", icon=Icons.TOOL_CODE)
    else:
        pretty_log("Execution Task", filename, icon=Icons.TOOL_CODE)
        
    try:
        exec_host_path = _get_safe_path(sandbox_dir, exec_rel_path)
    except ValueError as ve:
        return _format_error(str(ve))
        
    # Stubbornness Guard checks the ORIGINAL file
    if is_new_code and host_path.exists():
        if host_path.stat().st_size < 1_000_000:
            try:
                # For Python files, indentation is semantic — two scripts that
                # differ only in internal whitespace are NOT the same program.
                # So for .py we do an (almost-)exact comparison: only outer
                # whitespace is trimmed (because the caller pipeline strips
                # `clean_content` above, so a trailing "\n" on disk would
                # otherwise always force a rewrite).
                # For other languages (shell, JS, …) the whitespace-
                # insensitive heuristic still catches the LLM re-submitting
                # cosmetically-reformatted but identical code.
                ext_for_compare = rel_path.split('.')[-1].lower()
                def _check_same():
                    # errors='replace' keeps this safe on near-binary files.
                    existing = host_path.read_text(errors="replace")
                    if ext_for_compare == "py":
                        return existing.strip() == clean_content.strip()
                    return "".join(existing.split()) == "".join(clean_content.split())
                is_same = await asyncio.to_thread(_check_same)
                if is_same:
                    is_new_code = False
            except Exception as e:
                # Don't pass silently — the previous bare except hid real
                # errors (encoding issues, race deletions). Log + treat as
                # new code so the write-then-execute path proceeds normally.
                logging.getLogger("GhostAgent").debug(
                    f"execute stubbornness check failed for {filename}: {type(e).__name__}: {e}"
                )
        
    # Async Directory Creation
    if is_new_code:
        if not is_ephemeral and isinstance(host_path, Path) and host_path.is_file():   # §4UY C6/C21
            try:
                from .file_system import keep_before_replace
                await asyncio.to_thread(keep_before_replace, host_path, str(filename), sandbox_dir)
            except Exception:  # noqa: BLE001
                pass
        await asyncio.to_thread(host_path.parent.mkdir, parents=True, exist_ok=True)
        try: 
            await asyncio.to_thread(host_path.write_text, clean_content)
        except Exception as e: 
            return _format_error(f"Error writing script: {e}")
            
    if stateful and ext == "py":
        try:
            await asyncio.to_thread(exec_host_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(exec_host_path.write_text, exec_content)
        except Exception as e: 
            return _format_error(f"Error writing stateful wrapper: {e}")

    # Removed: pre-execution `python3 -m black` reformat. It added ~15s
    # to every .py execute, and the reformatted file then diverged from
    # the model's mental image — a subsequent `file_system replace` would
    # miss its SEARCH block and force a re-read round-trip. Black stays
    # available in the sandbox image if the model wants to call it
    # explicitly; it's just no longer imposed on every execution.

    # Set when this run is detached as a background job; read by the `finally`
    # cleanup, which must NOT delete a file the promoted process is still
    # executing. Initialised here so the cleanup is safe even if the try below
    # raises before the run.
    _promoted_job_landed = None

    try:
        ext_runner = rel_path.split('.')[-1].lower()
        runtime_map = {"py": "python3 -u", "js": "node", "sh": "bash"}
        runner = runtime_map.get(ext_runner, "")
        
        if is_jupyter:
            cmd = f"python3 -u .jupyter_runner.py {shlex.quote(exec_rel_path)}"
        else:
            cmd = f"{runner} {shlex.quote(exec_rel_path)}" if runner else f"./{shlex.quote(exec_rel_path)}"
            
        if args:
             # SECURITY FIX: Use shlex.quote to safely escape all arguments
             if isinstance(args, str):
                 try:
                     args = json.loads(args)
                 except (json.JSONDecodeError, ValueError):
                     # Not valid JSON — treat the whole string as a single argument.
                     args = [args]
             if not isinstance(args, list):
                 args = [args]
             cmd += " " + " ".join(shlex.quote(str(a)) for a in args)

        # Output truncation is owned by the sandbox layer now (one shared
        # head+tail policy via utils.text_truncate). The execute tool opts into
        # SPILL mode: the returned view is small (~24 KB head+tail) and the FULL
        # output is written to a workspace run-log the model can read with
        # file_system — so a noisy pip-install / test-run can no longer inject
        # ~70 KB of tokens that persist in history, and nothing is lost. The old
        # 512 KB trim here was dead code (docker capped at 256 KB before it saw
        # the output) and a fourth divergent budget; it is removed.
        import time as _time
        _t0 = _time.time()
        # STATEFUL runs opt OUT of promotion: the jupyter runner talks to a
        # persistent kernel, so detaching it would leave the kernel busy for
        # every later cell, and the wrapper/runner files this branch deletes
        # in `finally` are still being read by the detached process.
        if stateful and ext == "py":
            output, exit_code = await asyncio.to_thread(
                sandbox_manager.execute, cmd, spill_large_output=True,
                **_workdir_kw)
            promoted_job = None
        else:
            output, exit_code, promoted_job = await _run_in_sandbox(
                sandbox_manager, cmd, timeout=_EXEC_TIMEOUT_S,
                label=f"{runner} {rel_path}".strip(),
                project_id=_project_id_from_workdir(container_workdir),
                # Content-sensitive: a REWRITTEN script must not be mistaken
                # for the job that is still running the old one.
                identity=f"{cmd}#{hashlib.sha1(clean_content.encode()).hexdigest()[:16]}",
                # If this run is promoted the `finally` below must NOT delete
                # the script the detached process is executing — the job
                # supervisor deletes it when the job ends instead.
                cleanup_paths=([str(host_path)] if is_ephemeral else None),
                spill_large_output=True, **_workdir_kw)
        _dt = _time.time() - _t0

        if promoted_job is not None:
            # Still running at the budget → detached. Leave the script on
            # disk (the `finally` cleanup is skipped via this flag) — a
            # detached `bash` reads its script incrementally, and deleting it
            # mid-run would break the very job we just promoted.
            _promoted_job_landed = promoted_job
            _record_promoted_command(
                workspace_model, f"{runner} {filename}" if runner else str(filename),
                _dt, promoted_job)
            _register_promoted_job(kwargs.get("job_registry"), promoted_job,
                                   f"{runner} {rel_path}".strip())
            # `ToolOutcome` is a `str` SUBCLASS, so `+` is `str.__add__` and
            # returns a plain `str` — the UNRESOLVED status the line above
            # just built was destroyed by concatenation. The command-mode
            # promotion (`if command:`) kept its status and this one did not,
            # so the same event had two different verdicts depending on which
            # argument the model used.
            from .runs import record_promoted_job
            return _append_note(
                _promoted_result(promoted_job, int(_dt), output,
                                 record_promoted_job(kwargs.get("run_context"), promoted_job)), _probe_note)

        exit_code, output = _adopt_exit(exit_code, output)          # §4PH (R2-3)
        diagnostic_info = ""
        if exit_code != 0:
            if stateful and ext == "py":
                # Hide the temporary execution filename from the output
                output = output.replace(exec_rel_path, rel_path)
            
            tb_match = re.findall(r'File "([^"]+)", line (\d+),', output)
            if tb_match:
                # Only frames from the EXECUTED file may index
                # `clean_content` — the last workspace frame can belong to
                # an imported module, and slicing the executed file at THAT
                # line number showed the wrong file's region (or nothing).
                _exec_names = {Path(rel_path).name, Path(str(filename)).name}
                exec_matches = [m for m in tb_match
                                if Path(m[0]).name in _exec_names]
                try:
                    if exec_matches:
                        line_num = int(exec_matches[-1][1])
                        if stateful and ext == "py":
                            line_num = max(1, line_num - wrapper_line_offset)

                        lines = clean_content.splitlines()
                        start_l = max(0, line_num - 3)
                        end_l = min(len(lines), line_num + 2)
                        snippet = "\n".join([f"{i+1}: {l}" for i, l in enumerate(lines) if start_l <= i < end_l])
                        diagnostic_info = f"Error detected at Line {line_num} of '{rel_path}':\n{snippet}\n\nSUGGESTION: Review the snippet above line {line_num}."
                    else:
                        # The failing frame is in another file (an imported
                        # module or runner) — label it honestly instead of
                        # slicing the executed file at a foreign line number.
                        script_matches = [m for m in tb_match if "/workspace/" in m[0] or m[0].startswith("./")]
                        err_file, err_line = (script_matches or tb_match)[-1]
                        diagnostic_info = (
                            f"Error detected at Line {err_line} of '{err_file}' "
                            f"(NOT in '{rel_path}' itself — a file it imports/"
                            f"calls). Read that file around line {err_line} for "
                            f"the failing code.")
                except (ValueError, TypeError) as de:
                    # Diagnostic snippet is best-effort; don't let a parse
                    # error here mask the actual execution failure.
                    logging.getLogger("GhostAgent").debug(
                        f"execute diagnostic snippet build failed: {de}"
                    )

        if not output.strip():
            if exit_code != 0:
                output = f"[SYSTEM ERROR]: Process failed (Exit {exit_code}) with no output."
            else:
                output = "(Process executed successfully, but no output was printed to stdout. If you expected output, ensure you use print() statements.)"

        # Workspace command-outcome capture for the script-execution
        # branch. Significance gate is the same as the bash branch:
        # failures always, successes only when they took noticeable
        # wall time. (The gate was previously MISSING here despite this
        # comment claiming parity — every fast successful script run was
        # recorded with duration 0.0, spamming the activity ledger.)
        # Non-fatal — never block a real return.
        if workspace_model is not None and getattr(workspace_model, "enabled", False):
            try:
                if int(exit_code) != 0 or _dt >= 5.0:
                    workspace_model.record_command_outcome(
                        command=f"{runner} {filename}" if runner else str(filename),
                        exit_code=int(exit_code),
                        duration_seconds=float(_dt),
                        note=("failed" if exit_code != 0 else "long-running"),
                    )
            except Exception:  # noqa: BLE001
                pass

        _db_note = _no_db_server_note(output, kwargs.get("run_context"))      # 11-05: a script's psycopg/asyncpg/pymysql error too
        if exit_code != 0:
             return _append_note(
                 _format_error(output, hint=diagnostic_info,
                               exit_code=exit_code, elapsed_s=_dt),
                 _probe_note + _proxyless_browser_note + _db_note)

        return (f"--- EXECUTION RESULT ---\nEXIT CODE: {exit_code}\n"
                f"STDOUT/STDERR:\n{output}{_probe_note}{_proxyless_browser_note}{_db_note}")
    except Exception as e:
        return _format_error(f"Error: {e}")
    finally:
        # Cleanup paths — log on failure but don't raise. These leaks would
        # accumulate over time without diagnostic visibility.
        _logger = logging.getLogger("GhostAgent")
        # A PROMOTED script is still executing from this file. Deleting it
        # would break the job we just detached (bash reads its script
        # incrementally; python has already loaded it, but the traceback
        # would name a missing file either way). The path was handed to the
        # job supervisor as a `cleanup_path`, which unlinks it when the job
        # reaches a terminal state — including after a restart, since the
        # path is persisted on the job's registry row.
        if is_ephemeral and _promoted_job_landed is None:
            try:
                await asyncio.to_thread(host_path.unlink, missing_ok=True)
            except Exception as ce:
                _logger.debug(f"execute cleanup (ephemeral): {ce}")
        elif is_ephemeral:
            _logger.info(
                "execute: kept ephemeral %s — still running as job %s",
                filename, _promoted_job_landed.get("id"))
        if stateful and ext == "py":
            try:
                await asyncio.to_thread(exec_host_path.unlink, missing_ok=True)
            except Exception as ce:
                _logger.debug(f"execute cleanup (stateful wrapper): {ce}")
            if 'runner_path' in locals() and runner_path:
                try:
                    await asyncio.to_thread(runner_path.unlink, missing_ok=True)
                except Exception as ce:
                    _logger.debug(f"execute cleanup (jupyter runner): {ce}")


# ── §4UL (projects hook) ────────────────────────────────────────────────────
_PROJECT_PATH_RE = re.compile(r"(?<![\w/.-])(?:\./)?(projects/[A-Za-z0-9_-]+/[^\s'\"`;|&<>()]+)")
_PROJECT_DIR_RE = re.compile(r"(?:\./)?projects/[A-Za-z0-9_-]+/?")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", ">&"})
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", ";;", "|&"})


def _strip_heredocs(command: str) -> str:
    """The command without its here-document BODIES (their text is data, not shell)."""
    out, rest = [], command
    while True:
        m = _HEREDOC_RE.search(rest)
        if not m:
            out.append(rest)
            return "".join(out)
        nl = rest.find("\n", m.end())
        if nl < 0:
            out.append(rest)
            return "".join(out)
        out.append(rest[:nl + 1])
        body = rest[nl + 1:]
        end = re.search(rf"(?m)^\s*{re.escape(m.group(2))}\s*$", body)
        rest = body[end.end():] if end else ""


#: §4UL re-review m3: a command prefix that runs the rest — with the flags that take a value
_SHELL_PREFIX_FLAGS = {
    "sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U", "--user", "--group", "--chdir"},
    "doas": {"-u", "-C"},
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "--class", "--classdata"},
    "stdbuf": {"-i", "-o", "-e"},
    "nohup": set(), "time": set(), "command": set(), "exec": set(), "xvfb-run": {"-s", "-n", "-a"},
}
_INPUT_REDIRECTS = frozenset({"<", "<<", "<<<", "<<-", "<&", "<>"})


def _shell_tokens(command: str) -> Optional[List[str]]:
    """The command's shell tokens, here-document bodies dropped, a glued punctuation run (``)>``, ``;>``) split
    into its separator and its redirect; None when shlex cannot split it."""
    text = _strip_heredocs(str(command or ""))
    # §4UL round 3: a backslash-newline joins two lines, and a newline right after `&&` / `||` / `|` continues the
    # command — only a bare newline ends one
    text = re.sub(r"\\\r?\n", " ", text)
    text = re.sub(r"(&&|\|\||\|)[ \t]*\r?\n", r"\1 ", text)
    try:
        lex = shlex.shlex(text.replace("\n", " ; "), posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        raw = list(lex)
    except ValueError:
        return None
    toks: List[str] = []
    for t in raw:
        if (t and set(t) <= set("();<>|&") and t not in _SEPARATORS and t not in _REDIRECTS
                and t not in _INPUT_REDIRECTS and not (t.startswith(">") and set(t) <= set(">|&"))):
            i = min((t.find(c) for c in "<>" if c in t), default=-1)
            if i > 0:
                toks += [t[:i], t[i:]]
                continue
        toks.append(t)
    return toks


def strip_command_prefixes(cmd: List[str]) -> List[str]:
    """``VAR=x``, ``sudo -u x``, ``env -i``, ``timeout 30``, ``nice -n 5`` … dropped from a simple command: what
    is left starts with the program that really runs (§4UL re-review m3)."""
    cmd = list(cmd)
    while cmd:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", cmd[0]):
            cmd = cmd[1:]                                # VAR=x cmd …
            continue
        head = os.path.basename(cmd[0])
        if head not in _SHELL_PREFIX_FLAGS:
            break
        takes = _SHELL_PREFIX_FLAGS[head]
        cmd = cmd[1:]
        while cmd and cmd[0].startswith("-") and cmd[0] != "-":
            flag = cmd[0]
            cmd = cmd[1:]
            if flag in takes and cmd:
                cmd = cmd[1:]                            # the flag's value (`sudo -u www`)
            if flag == "--":
                break
        if head == "timeout" and cmd and re.fullmatch(r"[0-9.]+[smhd]?", cmd[0]):
            cmd = cmd[1:]                                # the duration
    return cmd


def workspace_relative(path: str) -> str:
    """``/workspace/projects/x/a`` and ``~/projects/x/a`` as the workspace-relative ``projects/x/a`` the project
    hook names (§4UL re-review m3); any other path unchanged."""
    p = str(path or "")
    for pre in ("/workspace/", "~/"):
        if p.startswith(pre):
            return p[len(pre):]
    return p


def _opt_values(args: List[str], short: str, long: str = "") -> List[str]:
    """The values of ``-o FILE`` / ``-oFILE`` / ``-sSo FILE`` / ``--output FILE`` / ``--output=FILE`` in ``args``
    (a short option's letter ends a cluster: the value is the rest of the cluster, else the next argument)."""
    out = []
    letter = short[1:] if short else ""
    for j, a in enumerate(args):
        if long and a == long:
            out += args[j + 1:j + 2]
        elif long and a.startswith(long + "="):
            out.append(a.split("=", 1)[1])
        elif letter and len(a) > 1 and a[0] == "-" and a[1] != "-" and letter in a[1:]:
            k = a.index(letter, 1)
            out += args[j + 1:j + 2] if k == len(a) - 1 else [a[k + 1:]]
    return [v for v in out if v and v != "-"]


def _shell_write_targets(command: str) -> List[str]:
    """§4UL review M6: the paths a shell command WRITES — redirect targets, `tee` files, the destination of
    `cp`/`mv`/`install`/`rsync`/`git clone`/`ln`, `sed -i`/`touch`/`mkdir` operands, `unzip -d` / `tar -C` dirs,
    and (§4UL re-review m3) `curl -o`, `wget -O`, `dd of=`, `cp/mv -t DIR`, through a `sudo -u x` / `timeout 9`
    prefix — never a path it only reads (`python projects/x/app.py 2>&1 | tail`, `cp projects/x/a /tmp/b`,
    `grep 'x >= 1' projects/x/app.py`). A command shlex cannot split names nothing (advisory). A path relative
    to an earlier `cd` is returned as written (not resolved against the cd)."""
    toks = _shell_tokens(command)
    if toks is None:
        return []
    simple: List[List[str]] = [[]]
    targets: List[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _SEPARATORS:
            simple.append([])
        elif t in _REDIRECTS or (t.startswith(">") and set(t) <= set(">|&")):
            if i + 1 < len(toks) and toks[i + 1] not in _SEPARATORS:
                tgt = toks[i + 1]
                if not tgt.isdigit() and tgt != "-":
                    targets.append(tgt)
                i += 1
        elif t in _INPUT_REDIRECTS:
            i += 1                                       # an input (or a here-doc word): read, not written
        elif t.isdigit() and i + 1 < len(toks) and toks[i + 1][:1] in "<>":
            pass                                         # the fd number of `2>…`
        else:
            simple[-1].append(t)
        i += 1
    for cmd in simple:
        cmd = strip_command_prefixes(cmd)
        if not cmd:
            continue
        head = os.path.basename(cmd[0])
        args = [a for a in cmd[1:]]
        ops = [a for a in args if not a.startswith("-")]
        if head == "tee":
            targets += ops
        elif head in ("cp", "mv", "install", "ln") and _opt_values(args, "-t", "--target-directory"):
            dest = _opt_values(args, "-t", "--target-directory")[0]
            srcs = [a for a in ops if a != dest]
            targets += [dest.rstrip("/") + "/" + os.path.basename(src.rstrip("/")) for src in srcs]
        elif head in ("cp", "mv", "install", "rsync", "ln") and len(ops) >= 2:
            dest = ops[-1]
            if _PROJECT_DIR_RE.fullmatch(workspace_relative(dest)) or dest.endswith("/") or len(ops) > 2:
                targets += [dest.rstrip("/") + "/" + os.path.basename(src.rstrip("/")) for src in ops[:-1]]
            else:
                targets.append(dest)
        elif head == "git" and len(ops) >= 3 and ops[0] == "clone":
            targets.append(ops[-1])
        elif head == "sed" and any(a == "-i" or a.startswith("-i") or a.startswith("--in-place") for a in args):
            targets += ops[1:] if not any(a in ("-e", "-f") for a in args) else ops
        elif head in ("touch", "mkdir", "truncate"):
            targets += ops
        elif head == "unzip" and "-d" in args:
            j = args.index("-d")
            targets += args[j + 1:j + 2]
        elif head == "tar":
            for flag in ("-C", "--directory"):
                if flag in args:
                    j = args.index(flag)
                    targets += args[j + 1:j + 2]
            if args and re.fullmatch(r"-?[A-Za-z]*c[A-Za-z]*f", args[0]) and len(args) > 1:
                targets.append(args[1])                  # tar -czf OUT …: the archive it writes
        elif head == "curl":
            targets += _opt_values(args, "-o", "--output")
        elif head == "wget":
            targets += _opt_values(args, "-O", "--output-document")
        elif head == "dd":
            targets += [a[3:] for a in args if a.startswith("of=") and len(a) > 3]
    return targets


def _note_execute_project_writes(kwargs, result) -> str:
    """§4UL: an `execute` run from the workspace ROOT (no project pinned) that
    wrote into ``projects/<id>/…`` is that project's work — each such path
    the command names after a write construct, and an inline script saved
    there, go to `tools.projects.note_project_write` (via the file_system
    hook, which finds the request's context). Returns the hook's notes for
    the result. Advisory bookkeeping: a failed run, a pinned run or any
    error notes nothing."""
    try:
        _wpid = _project_id_from_workdir(kwargs.get("container_workdir"))
        if kwargs.get("container_workdir") and not _wpid:
            return ""           # pinned somewhere that is not a project
        st = getattr(getattr(result, "status", None), "value", None)
        if st is not None:
            if st not in ("ok", "partial"):
                return ""
        else:
            from .tool_failure import exec_exit_code
            _code = exec_exit_code(str(result or ""))
            if _code is not None and _code != 0:
                return ""
        sandbox_dir = kwargs.get("sandbox_dir")
        if sandbox_dir is None and not _wpid:
            return ""
        paths = []
        command = str(kwargs.get("command") or kwargs.get("cmd") or "")
        fname = str(kwargs.get("filename") or kwargs.get("file") or "")
        if _wpid:
            # §4UM round 3: a PINNED run — its relative write targets (and an inline script) are the project's
            # files, reported like the file tool's (the change log lists them even when the folder cannot be
            # compared); only paths that exist after the run (a target after a `cd` may be elsewhere)
            from ..core.request_scope import current_scope
            _sc = current_scope()
            _root = getattr(getattr(_sc, "context", None), "sandbox_dir", None) if _sc is not None else None
            if not _root:
                return ""
            _pdir = Path(_root).resolve() / "projects" / _wpid
            cands = list(_shell_write_targets(command)) if command else []
            if fname and (kwargs.get("content") or kwargs.get("code")):
                cands.append(fname)
            for tgt in cands:
                w = workspace_relative(str(tgt))
                if w.startswith(f"projects/{_wpid}/"):
                    full = Path(_root).resolve() / w
                elif w.startswith(("/", "~", "$")) or ".." in Path(w).parts:
                    continue
                else:
                    full = _pdir / (w[2:] if w.startswith("./") else w)
                if full.is_file():
                    paths.append(str(full))
            sandbox_dir = _root
        else:
            if command:                     # §4UL review M6: the paths it WRITES, never ones it reads
                for tgt in _shell_write_targets(command):
                    m = _PROJECT_PATH_RE.fullmatch(workspace_relative(tgt))
                    if m:
                        paths.append(m.group(1))
            if fname and (kwargs.get("content") or kwargs.get("code")) and _PROJECT_PATH_RE.fullmatch(fname):
                paths.append(_PROJECT_PATH_RE.fullmatch(fname).group(1))
        if not paths:
            return ""
        from .file_system import note_project_write_hook
        notes = [note_project_write_hook(sandbox_dir, p.rstrip("/")) for p in list(dict.fromkeys(paths))[:20]]
        return "\n".join(n for n in notes if n)
    except Exception:  # noqa: BLE001 — bookkeeping never breaks a run
        logging.getLogger("GhostAgent").debug("execute project-write note skipped", exc_info=True)
    return ""


_PROJECT_ID_IN_RE = re.compile(r"(?:^|[\s'\"=:/(])(?:\./|/workspace/)?projects/([0-9a-f]{12})(?=/|\s|$|['\";|&)])")


def _prepare_execute_project_writes(kwargs) -> None:
    """§4UM review M3: BEFORE a command runs, snapshot each project it can change — the project a pinned run works
    in (``container_workdir``), and every ``projects/<id>`` a root run names — so the request's end diffs the
    folder: a heredoc, `sed -i`, `rm`, `mv`, a build's output, whatever wrote it. Advisory; never raises."""
    try:
        pids = []
        _wpid = _project_id_from_workdir(kwargs.get("container_workdir"))
        if _wpid:
            pids.append(_wpid)
        command = str(kwargs.get("command") or kwargs.get("cmd") or "")
        pids += [m.group(1) for m in _PROJECT_ID_IN_RE.finditer(command)][:10]
        fname = str(kwargs.get("filename") or kwargs.get("file") or "")
        m = _PROJECT_ID_IN_RE.search(" " + fname)
        if m:
            pids.append(m.group(1))
        if not pids:
            return
        from ..core.request_scope import current_scope
        sc = current_scope()
        ctx = getattr(sc, "context", None) if sc is not None else None
        if ctx is None:
            return
        # round 4: real write evidence — the command names write targets, or an inline script is saved; a
        # `pytest` / `ls` / `git status` diffs the folder, but alone never makes an entry that names no file
        evidence = bool(command and _shell_write_targets(command)) or bool(
            fname and (kwargs.get("content") or kwargs.get("code")))
        if not evidence and command:            # round 5: rm, a mutating git, a formatter, patch, inline writes
            from ..core.project_turns import shell_check_events
            evidence = "w" in shell_check_events(command)
        from .projects import prepare_project_write
        for pid in list(dict.fromkeys(pids)):
            prepare_project_write(ctx, project_id=pid, evidence=evidence)
    except Exception:  # noqa: BLE001 — bookkeeping never breaks a run
        logging.getLogger("GhostAgent").debug("execute project-write prepare skipped", exc_info=True)


async def _finished_job_logs_named(kwargs) -> frozenset:
    """§4UO r3: the job logs the command names whose job had already finished before it ran."""
    try:
        command = str(kwargs.get("command") or "")
        if ".jobs/job-" not in command:
            return frozenset()
        from .runs import finished_job_logs
        return await asyncio.to_thread(finished_job_logs, kwargs.get("sandbox_manager"), command)
    except Exception:  # noqa: BLE001
        return frozenset()


async def _note_job_log_printed(kwargs, result, finished_before=frozenset()) -> None:
    """§4UO: a command that printed a finished job's log (`tail`/`cat .jobs/<id>.log`) delivered it — its wake
    turn must not report it again (`tools.runs.mark_job_log_output`). Off the loop; never raises."""
    try:
        command = str(kwargs.get("command") or "")
        if ".jobs/job-" not in command:
            return
        from .runs import mark_job_log_output
        jid = await asyncio.to_thread(mark_job_log_output, kwargs.get("sandbox_manager"), command, str(result or ""),
                                      finished_before)
        if jid:
            pretty_log("Job Delivered", f"{jid}: its log's tail was printed after it finished — no wake turn",
                       icon=Icons.JOB_PROMOTE)
    except Exception:  # noqa: BLE001 — bookkeeping never breaks a run
        logging.getLogger("GhostAgent").debug("job log print not noted", exc_info=True)


_tool_execute_impl = tool_execute


async def _net_snapshot(kwargs):
    """§4UY N6: the sandbox's outbound-connection counters (`SandboxManager.net_connections`), or None."""
    probe = getattr(kwargs.get("sandbox_manager"), "net_connections", None)
    if not callable(probe):
        return None
    try:
        snap = await asyncio.to_thread(probe)
    except Exception:  # noqa: BLE001
        return None
    return snap if isinstance(snap, tuple) and len(snap) == 2 else None


async def _note_network_use(kwargs, before, run=None) -> None:
    """§4UY N6: did THIS run open a network connection? Measured: the counters moved between before and
    after. r4: the run is marked by the old name rule OR the measured connection (`run_reached_out`), and
    reads outside content when it names a ledgered file too (`execute_reads_outside`). A marked run's files —
    those it created in its working directory and the write targets it names (`note_run_files`) — are
    recorded, never when another sandbox call overlapped it (the counters are container-wide); a run
    promoted to a detached job hands its record to the job (`note_job_started`, landed by
    `delegate._land_sandbox_row`). Never raises."""
    try:
        run = run or {}
        from ..utils import provenance as _prov
        used = None
        after = await _net_snapshot(kwargs) if before is not None else None
        if after is not None and after[0] >= before[0]:   # read, and the rules were not reloaded mid-run
            used = after[0] > before[0]
        # r5: overlapped counters may be another call's — a "yes" marks THIS request only, and is never stored
        # for the command text (a later run of the same command is not marked by it) nor handed to a job
        _shared_yes = bool(used and run.get("overlap"))
        if used is not None and not _shared_yes:
            _prov.note_measured_run(kwargs, used=used, complete=bool(before[1] and after[1]))
        if run.get("job"):
            _prov.note_job_started(run["job"], kwargs, run.get("root"), run.get("before"),
                                   None if _shared_yes else used, run.get("t0", 0.0), run.get("ws", ""))
        # r5: a cp/mv/rsync of a ledgered file or folder keeps its rows — first (the source of a `mv` is gone
        # afterwards), and whatever the counters say
        await asyncio.to_thread(_prov.carry_run_moves, str(kwargs.get("command") or ""), run.get("root"),
                                run.get("ws", ""))
        # r7: extraction is a copy — a ledgered archive's unpacked files are its bytes (`unzip -q dl` names
        # the archive only through the tool's own `.zip` default, so this is read before the naming verdict)
        _ex = _prov.ledgered_extract(str(kwargs.get("command") or ""), run.get("root"), run.get("ws", ""))
        if not (_ex or _prov.execute_reads_outside(kwargs)
                or (_shared_yes and _prov.run_reached_out(kwargs, used))):
            return
        _prov.note_untrusted_content("execute (network)")
        # §4UY N6 r7 (operator): only a run that itself reached out has its files recorded; one that only
        # NAMES ledgered content marks the request (its copies/moves were carried above) — no spread
        if not (_shared_yes or _prov.run_reached_out(kwargs, _prov.measured_run(kwargs))):
            if _ex and not run.get("overlap"):
                n = await asyncio.to_thread(_prov.note_run_files, run.get("root"), run.get("before"), kwargs,
                                            run.get("t0", 0.0), run.get("ws", ""), _ex)
                pretty_log("Outside Content", f"the run extracts a downloaded archive — {n} file(s) it created or "
                           f"changed are outside content ({_ex})", icon=Icons.SHIELD, level="INFO")
                return
            pretty_log("Outside Content", "the run names outside content — its output is outside content; it did "
                       "not reach out, so no file it wrote is recorded", icon=Icons.SHIELD, level="INFO")
            return
        if run.get("overlap"):
            pretty_log("Outside Content", "the run reads outside content — another sandbox call overlapped "
                       "it, so no file is recorded", icon=Icons.SHIELD, level="INFO")
            return
        n = await asyncio.to_thread(_prov.note_run_files, run.get("root"), run.get("before"), kwargs,
                                    run.get("t0", 0.0), run.get("ws", ""), "execute run that read outside content")
        pretty_log("Outside Content", f"the run reached out — its output and {n} file(s) it created "
                   f"or wrote are outside content", icon=Icons.SHIELD, level="INFO")
    except Exception:  # noqa: BLE001 — bookkeeping never breaks a run
        logging.getLogger("GhostAgent").debug("network-use note skipped", exc_info=True)


#: §4UY N6 r3: the run in progress — `_record_promoted_command` writes the job id here when it detaches
_RUN_STATE: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar("ghost_execute_run", default=None)


def _running_sandbox_jobs(sm) -> frozenset:
    """§4UY N6 r3: ids of the sandbox's detached jobs marked running (their traffic shares the counters);
    {"?"} when the registry cannot be read (treated as an overlap)."""
    try:
        if sm is None or not getattr(sm, "host_workspace", None):
            return frozenset()
        from ..sandbox.jobs import STATE_RUNNING, get_job_supervisor
        sup = get_job_supervisor(sm)
        if sup is None:
            return frozenset()
        return frozenset(str(e.get("id")) for e in sup.list_entries()
                         if isinstance(e, dict) and e.get("state") == STATE_RUNNING)
    except Exception:  # noqa: BLE001
        return frozenset({"?"})


async def _tool_execute_noting(*args, **kwargs):
    _prepare_execute_project_writes(kwargs)
    _done_before = await _finished_job_logs_named(kwargs)        # §4UO r3: BEFORE the command runs
    from ..utils import provenance as _prov_n6
    _sm = kwargs.get("sandbox_manager")
    _run = {"root": _prov_n6.run_root(kwargs.get("sandbox_dir"), kwargs.get("container_workdir")), "job": None,
            "ws": _prov_n6.workspace_root(kwargs.get("sandbox_dir"))}
    _tok = _prov_n6.sandbox_call_begin("execute")                 # §4UY N6 r3: overlap
    try:
        _jobs0 = await asyncio.to_thread(_running_sandbox_jobs, _sm)
        _run["before"] = await asyncio.to_thread(_prov_n6.snapshot_names, _run["root"])
        _net0 = await _net_snapshot(kwargs)                       # §4UY N6: every run is measured
        _run["t0"] = time.time()
        _cv = _RUN_STATE.set(_run)
        try:
            res = await _tool_execute_impl(*args, **kwargs)
        finally:
            _RUN_STATE.reset(_cv)
    finally:
        _overlap = _prov_n6.sandbox_call_end(_tok)
    _jobs1 = await asyncio.to_thread(_running_sandbox_jobs, _sm)
    _run["overlap"] = bool(_overlap or _jobs0 or (_jobs1 - {str(_run.get("job"))}))
    await _note_network_use(kwargs, _net0, _run)
    await _note_job_log_printed(kwargs, res, _done_before)
    _note = _note_execute_project_writes(kwargs, res)
    if _note:
        from .outcome import append_note
        res = append_note(res, "\n" + _note)
    return res


import functools as _functools  # noqa: E402
_functools.update_wrapper(_tool_execute_noting, _tool_execute_impl)
tool_execute = _tool_execute_noting
