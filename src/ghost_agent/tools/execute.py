import asyncio
import os
import shlex
import sys
import re
import logging
import uuid
import base64
import datetime
import ast
import json
from pathlib import Path
from typing import List
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import sanitize_code
from .file_system import _get_safe_path


# Commands for which exit code 1 means "no matches found" — a successful
# query with an empty result — rather than a failure. grep & friends signal
# genuine errors (bad regex, unreadable file) with exit 2 + stderr output.
_EXIT1_MEANS_NO_MATCH = {"grep", "egrep", "fgrep", "zgrep", "rg", "pgrep"}

# Execution budget for sandboxed runs (seconds). The sandbox layer wraps
# every run in `timeout -k 5s <budget>s`, so hitting the budget surfaces
# as exit 124 (or 137/143 when the -k SIGKILL / a SIGTERM landed).
_EXEC_TIMEOUT_S = 600
_TIMEOUT_KILL_CODES = (124, 137, 143)

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


async def tool_execute(filename: str = None, content: str = None, sandbox_dir: Path = None, sandbox_manager=None, scrapbook=None, args: list = None, memory_dir: Path = None, stateful: bool = False, command: str = None, workspace_model=None, container_workdir: str = None, **kwargs):
    # When a project is active, run from /workspace/projects/<id> so files
    # written via file_system (also scoped) read back. Passed ONLY when set,
    # so sandbox managers without a `workdir` param keep working unchanged.
    _workdir_kw = {"workdir": container_workdir} if container_workdir else {}
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
        return _AGENT_PORT_PROBE_MSG

    # --- 🛡️ HIJACK LAYER: CODE SANITIZATION ---
    
    # Helper for consistent error reporting. Emits the REAL exit code —
    # hardcoding 1 made a 124 timeout kill indistinguishable from a
    # program failure, so the model re-ran identical >=10-min commands.
    # The annotation rides AFTER the digits: downstream parsers match
    # `EXIT CODE:\s*(\d+)` and must keep working.
    def _format_error(msg, hint=None, exit_code=1):
        _code_note = ""
        if exit_code in _TIMEOUT_KILL_CODES:
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
        return out

    if command:
        if not sandbox_manager: return _format_error("Error: Sandbox manager not initialized.")

        # `command` takes precedence and returns below; a filename/content
        # supplied in the SAME call would be silently dropped (never written
        # or run). Warn so the model doesn't assume its file was created.
        if filename or content:
            pretty_log(
                "Execute Ambiguous Args",
                f"Both `command` and file/content given; running `command` and "
                f"IGNORING filename={filename!r}. Issue a separate call to write/run a file.",
                level="WARNING", icon=Icons.WARN,
            )

        # Auto-strip `cd /<nonexistent> && ` prefixes. Qwen 3.6 has a very
        # strong prior for starting shell commands with `cd /sandbox`,
        # `cd /home/user`, `cd /root`, etc. — none of those paths exist in
        # our container (the shell starts in /workspace). Four rounds of
        # prompt-side warnings (top-of-state `⚠️` banner, SPECIALIST rule,
        # explicit ✗ negative examples) still produced occasional leaks.
        # Cleaner to fix at the tool level: detect the pattern, strip it,
        # log a warning so the operator sees it, and run the rest from the
        # real CWD. /workspace is always the correct destination, so this
        # auto-fix can never be wrong.
        _cd_strip_pattern = re.compile(
            r'^\s*cd\s+/(?:sandbox|home(?:/\w+)?|root|app|tmp|usr/src|opt)'
            r'(?:/[^\s&;]*)?\s*&&\s*',
            re.IGNORECASE,
        )
        _stripped = _cd_strip_pattern.sub('', command, count=1)
        if _stripped != command:
            pretty_log(
                "CWD Auto-fix",
                f"Stripped invalid `cd` prefix; running from /workspace. "
                f"Original: {command[:80]}",
                level="WARNING", icon=Icons.SHIELD,
            )
            command = _stripped

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
                    _path = f"/tmp/_ghost_inline_{uuid.uuid4().hex[:8]}.{_ext}"
                    _b64 = base64.b64encode(_bash_body.encode("utf-8")).decode("ascii")
                    _prefix = command[:_inline_py_match.start()]
                    _sep = _inline_py_match.group("sep")
                    # `cd proj && python3 /tmp/x.py` — cd (if any) is preserved
                    # so the script's relative paths resolve like the inline
                    # form did; the file is written to an absolute /tmp path so
                    # the write itself is cwd-independent.
                    #
                    # For Python, `python -c "<body>"` puts the CWD on sys.path
                    # ('' as sys.path[0]); running the converted file from /tmp
                    # puts /tmp there instead, so a body that imports a module
                    # sitting in the working directory breaks with
                    # ModuleNotFoundError (observed live, 2026-07 chess trace:
                    # `from chess_engine import Board` failed only BECAUSE of
                    # this conversion). Prepend the runtime CWD to PYTHONPATH
                    # to keep the inline form's import semantics.
                    if _ext == "py":
                        _invoke = ('PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" '
                                   f"{_interp} {_path}")
                    else:
                        _invoke = f"{_interp} {_path}"
                    _run = f"{_prefix}{_sep} {_invoke}".strip()
                    command = (
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

                    return _format_error(
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
                        f"{_skill_hint}"
                    )

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
            return _format_error(
                f"SYSTEM BLOCK: shell command rejected by pre-execution "
                f"validator: {_shell_reason}. The command was not run. Re-emit a "
                f"safer form, or use file_system + execute if you need to "
                f"run a script."
            )

        # Execute the command securely using the sandbox manager's built-in execute wrapper.
        # 600s (not 300s): real project work — training a small model, running a
        # full test suite, installing a package, a multi-stage build — routinely
        # needs more than five minutes, and a premature kill burned turns
        # re-running from scratch. The hard kill (-k 5s) still guarantees the
        # process can't hang the container indefinitely.
        import time as _time
        _t0 = _time.time()
        cmd_str = f"bash -c {shlex.quote(command)}"
        # spill_large_output: same small-view + full-log-to-file policy as the
        # script path (the bash branch previously had NO tool-level trim, so a
        # noisy direct command dumped its whole 256 KB into context).
        output, exit_code = await asyncio.to_thread(
            sandbox_manager.execute, cmd_str, timeout=_EXEC_TIMEOUT_S,
            spill_large_output=True, **_workdir_kw)
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
        if exit_code != 0 and exit_code not in _TIMEOUT_KILL_CODES and not _workdir_kw and _looks_like_file_not_found(output):
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
                    level="WARNING", icon=Icons.SHIELD,
                )
                _fh_out, _fh_code = await asyncio.to_thread(
                    sandbox_manager.execute, cmd_str, timeout=_EXEC_TIMEOUT_S,
                    spill_large_output=True, workdir=_proj_wd)
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

        if exit_code != 0 and exit_code not in _TIMEOUT_KILL_CODES and _workdir_kw and _looks_like_file_not_found(output):
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
                    level="WARNING", icon=Icons.SHIELD,
                )
                # spill_large_output on the retry too — dropping it meant a
                # noisy retried command dumped its full output into context,
                # the exact flood the primary call's spill mode prevents.
                _re_out, _re_code = await asyncio.to_thread(
                    sandbox_manager.execute,
                    f"bash -c {shlex.quote(_remapped)}", timeout=_EXEC_TIMEOUT_S,
                    spill_large_output=True, **_workdir_kw)
                if _re_code == 0 or not _looks_like_file_not_found(_re_out):
                    output, exit_code = _re_out, _re_code
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
        if exit_code == 1 and _no_output:
            _tail_seg = re.split(r"&&|\|\||;|\|", command)[-1].strip()
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

        if exit_code != 0:
            return _format_error(
                output or f"Process failed (Exit {exit_code}) with no output.",
                exit_code=exit_code,
            ) + _host_proc_note

        return (f"--- COMMAND RESULT ---\nEXIT CODE: {exit_code}\n"
                f"STDOUT/STDERR:\n{output}{_host_proc_note}")

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

    is_new_code = True
    _probe_note = ""
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
                level="WARNING", icon=Icons.SHIELD,
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
    # `file_system` (very common module name), `scratchpad`, `recall`, `replan`
    # — were REMOVED: a strong model importing a local `file_system.py` helper
    # was getting its valid code hard-blocked. `browser` is KEPT: it's a
    # prominent native tool the model does hallucinate importing, and a dedicated
    # guard test pins it (the Pyodide-`browser` false positive is rare in this
    # sandbox). The rest are unambiguous tool names.
    if ext == "py":
        forbidden_modules = [
            "knowledge_base", "system_utility", "manage_tasks",
            "postgres_admin", "web_search", "fact_check", "deep_research",
            "vision_analysis", "delegate_to_swarm",
            "learn_skill", "update_profile", "dream_mode", "browser",
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
        out_chk, check_code = await asyncio.to_thread(sandbox_manager.execute, f"test -f {conn_file}")
        
        if check_code == 0:
            # File exists, check if process is actually alive
            out_pg, pgrep_code = await asyncio.to_thread(sandbox_manager.execute, "pgrep -f ipykernel_launcher")
            if pgrep_code != 0:
                check_code = 1 # Force reboot
                
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
                _container.exec_run(f"python3 -m ipykernel_launcher -f {conn_file}", **exec_kwargs)
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

while True:
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
        if time.time() - start_time > 295:
            sys.stdout.write("\\n[SYSTEM ERROR: Kernel Timeout. Execution exceeded 5 minutes]\\n")
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
        output, exit_code = await asyncio.to_thread(
            sandbox_manager.execute, cmd, spill_large_output=True, **_workdir_kw)
        _dt = _time.time() - _t0

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

        if exit_code != 0:
             return _format_error(output, hint=diagnostic_info,
                                  exit_code=exit_code) + _probe_note

        return (f"--- EXECUTION RESULT ---\nEXIT CODE: {exit_code}\n"
                f"STDOUT/STDERR:\n{output}{_probe_note}")
    except Exception as e:
        return _format_error(f"Error: {e}")
    finally:
        # Cleanup paths — log on failure but don't raise. These leaks would
        # accumulate over time without diagnostic visibility.
        _logger = logging.getLogger("GhostAgent")
        if is_ephemeral:
            try:
                await asyncio.to_thread(host_path.unlink, missing_ok=True)
            except Exception as ce:
                _logger.debug(f"execute cleanup (ephemeral): {ce}")
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
