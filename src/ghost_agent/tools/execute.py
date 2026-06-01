import asyncio
import os
import shlex
import sys
import re
import logging
import uuid
import datetime
import ast
import json
from pathlib import Path
from typing import List
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import sanitize_code
from .file_system import _get_safe_path


def _looks_like_file_not_found(out) -> bool:
    """Heuristic: did a command fail because the target file wasn't where it
    looked? Used to trigger the project-scoped → root cwd retry. Matches the
    common interpreter/shell messages (python, node, bash, cat, ...)."""
    if not isinstance(out, str):
        return False
    o = out.lower()
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

    # --- 🛡️ HIJACK LAYER: CODE SANITIZATION ---
    
    # Helper for consistent error reporting
    def _format_error(msg, hint=None):
        out = f"--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n{msg}"
        if hint:
            out += f"\n\n--- 💡 DIAGNOSTIC HINT ---\n{hint}\n------------------------"
        return out

    if command:
        if not sandbox_manager: return _format_error("Error: Sandbox manager not initialized.")

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

        # Tool-level enforcement of the WRITE-THEN-EXECUTE rule. Four
        # rounds of prompt-side warnings still saw the model emit
        # `python3 -c "from X import Y; print(Y('... \"GET /foo\" ...'))"`
        # shapes where nested quotes corrupt the f-string via bash escape.
        # Reject any inline `python -c`, `python3 -c`, or `bash -c` body
        # that's substantive enough to deserve a proper file. Threshold:
        # >= 120 chars of inline body OR contains more than one `;` OR
        # contains an import statement — any of those trigger a REDIRECT
        # error telling the model to use file_system write + execute.
        _inline_py_match = re.match(
            r'^\s*(?:python3?|bash)\s+-c\s+([\'"])(.*)\1\s*(?:\|.*)?$',
            command, re.DOTALL,
        )
        if _inline_py_match:
            _body = _inline_py_match.group(2)
            _body_compact = _body.strip()
            _too_long = len(_body_compact) >= 120
            _too_many_stmts = _body_compact.count(";") >= 2
            _has_import = bool(re.search(r'\b(?:from|import)\s+\w', _body_compact))
            if _too_long or _too_many_stmts or _has_import:
                reason = []
                if _too_long:
                    reason.append(f"body is {len(_body_compact)} chars (>= 120)")
                if _too_many_stmts:
                    reason.append(f"{_body_compact.count(';')} semicolons (multi-statement)")
                if _has_import:
                    reason.append("contains an import statement")
                reason_str = "; ".join(reason)
                pretty_log(
                    "Inline Script Blocked",
                    f"Rejected inline `-c` body ({reason_str})",
                    level="WARNING", icon=Icons.SHIELD,
                )

                # Heuristic: does the blocked body look like a failed
                # attempt to call an acquired skill? Patterns seen in
                # the 2026-04-24 EA incident:
                #
                #   from greece_top_news import greece_top_news; ...
                #   python3 acquired_skills/foo.py ...
                #
                # Acquired skills are TOP-LEVEL callable tools — the
                # LLM should just invoke them by name. Append a
                # targeted hint so the retry goes there directly
                # instead of falling back to file_system(write) +
                # execute, which would be a correct but still-wrong
                # second-best path for this class of request.
                _skill_hint = ""
                _skill_pattern = re.search(
                    r'from\s+([a-zA-Z_][\w]*)\s+import\s+\1\b',
                    _body_compact,
                )
                _subprocess_pattern = re.search(
                    r'acquired_skills/([a-zA-Z_][\w]*)\.py',
                    _body_compact,
                )
                _candidate_name = None
                if _skill_pattern:
                    _candidate_name = _skill_pattern.group(1)
                elif _subprocess_pattern:
                    _candidate_name = _subprocess_pattern.group(1)
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
                    f"SYSTEM BLOCK: Inline `python -c '...'` / `bash -c '...'` "
                    f"scripts are restricted. Trigger: {reason_str}. Bash "
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

        # Execute the command securely using the sandbox manager's built-in execute wrapper
        import time as _time
        _t0 = _time.time()
        cmd_str = f"bash -c {shlex.quote(command)}"
        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd_str, timeout=300, **_workdir_kw)
        # Root fallback for project-scoped commands. When a project is active
        # the command runs from /workspace/projects/<id>, but the model may
        # reference a file that lives at the sandbox ROOT — e.g. one it wrote
        # in the SAME turn it switched into the project, before the switch
        # took effect (observed: file_system write emitted just before the
        # `switch` tool call → file at root, then `python3 chart.py` from the
        # scoped cwd fails). If the scoped run failed with a file-not-found
        # signature, retry once from the root. Safe: "can't open file" means
        # nothing executed, so there are no partial side effects to repeat.
        if exit_code != 0 and _workdir_kw and _looks_like_file_not_found(output):
            output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd_str, timeout=300)
        _dt = _time.time() - _t0

        # Workspace command-outcome capture: record significant runs so
        # the user can later ask "what commands did I run yesterday?" or
        # "did that pipeline ever finish?". Significance gate: failed
        # commands always, or successful runs that took >5s. Skip noise.
        if workspace_model is not None and getattr(workspace_model, "enabled", False):
            try:
                if exit_code != 0 or _dt >= 5.0:
                    workspace_model.record_command_outcome(
                        command=command, exit_code=int(exit_code),
                        duration_seconds=float(_dt),
                        note=("failed" if exit_code != 0 else "long-running"),
                    )
            except Exception:  # noqa: BLE001
                pass

        if exit_code != 0:
            return _format_error(output or f"Process failed (Exit {exit_code}) with no output.")

        return f"--- COMMAND RESULT ---\nEXIT CODE: {exit_code}\nSTDOUT/STDERR:\n{output}"

    is_ephemeral = False
    if content and not filename and not command:
        import uuid
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
    if not content:
        if not host_path.exists():
            return _format_error(f"SYSTEM ERROR: File '{filename}' does not exist. You must provide 'content' to create it.")
        content = await asyncio.to_thread(host_path.read_text)
        is_new_code = False

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
    if ext == "py":
        forbidden_modules = [
            "knowledge_base", "system_utility", "file_system", "manage_tasks",
            "postgres_admin", "web_search", "fact_check", "deep_research",
            "vision_analysis", "delegate_to_swarm", "recall", "scratchpad",
            "learn_skill", "update_profile", "dream_mode", "replan", "browser"
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
    
    # Extract rel_path early for wrapper filename generation
    if not filename: return _format_error("Error: filename is required.")
    rel_path = str(filename).lstrip("/")
    if rel_path.startswith("sandbox/"):
        rel_path = rel_path[8:]

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
                
            sandbox_manager.container.exec_run(f"python3 -m ipykernel_launcher -f {conn_file}", **exec_kwargs)
            
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
                     import json
                     args = json.loads(args)
                 except (json.JSONDecodeError, ValueError):
                     # Not valid JSON — treat the whole string as a single argument.
                     args = [args]
             if not isinstance(args, list):
                 args = [args]
             cmd += " " + " ".join(shlex.quote(str(a)) for a in args)

        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd, **_workdir_kw)

        # --- Output truncation (head + tail) ---
        # The sandbox layer may already cap output at its own limit, but we
        # apply a defensive head+tail trim here too. Keeping the TAIL is
        # critical: Python tracebacks always print the actual exception
        # type at the very end, so a head-only trim would hide the most
        # useful diagnostic information from the LLM.
        # Cap: 512 KB total, weighted toward the tail (100 KB head + 400 KB tail).
        try:
            if isinstance(output, str):
                MAX_TOTAL = 512 * 1024
                HEAD_BYTES = 100 * 1024
                TAIL_BYTES = 400 * 1024
                # `len(str)` here is a chars-as-proxy-for-bytes heuristic.
                # For mostly-ASCII script output it's accurate; for output
                # with heavy multibyte content we slightly over-trim, which
                # is the safer failure mode.
                if len(output) > MAX_TOTAL:
                    head = output[:HEAD_BYTES]
                    tail = output[-TAIL_BYTES:]
                    dropped = len(output) - HEAD_BYTES - TAIL_BYTES
                    output = (
                        f"{head}\n\n"
                        f"[... {dropped} bytes truncated by execute.py — "
                        f"showing first {HEAD_BYTES // 1024} KB and last "
                        f"{TAIL_BYTES // 1024} KB of {len(output) // 1024} KB total ...]\n\n"
                        f"{tail}"
                    )
        except Exception as _trim_err:
            # Truncation is best-effort — never let it crash the tool path.
            logging.getLogger("GhostAgent").debug(
                f"execute output truncation failed: {_trim_err}"
            )

        diagnostic_info = ""
        if exit_code != 0:
            if stateful and ext == "py":
                # Hide the temporary execution filename from the output
                output = output.replace(exec_rel_path, rel_path)
            
            tb_match = re.findall(r'File "([^"]+)", line (\d+),', output)
            if tb_match:
                # Prioritize matches from the actual script or workspace, ignore deep library traces
                script_matches = [m for m in tb_match if filename in m[0] or rel_path in m[0] or "/workspace/" in m[0] or m[0].startswith("./")]
                
                if script_matches:
                    _, last_error_line = script_matches[-1]
                else:
                    _, last_error_line = tb_match[-1]
                    
                try:
                    line_num = int(last_error_line)
                    if stateful and ext == "py" and (rel_path in output or filename in output):
                         line_num = max(1, line_num - wrapper_line_offset)
                    
                    lines = clean_content.splitlines()
                    start_l = max(0, line_num - 3)
                    end_l = min(len(lines), line_num + 2)
                    snippet = "\n".join([f"{i+1}: {l}" for i, l in enumerate(lines) if start_l <= i < end_l])
                    diagnostic_info = f"Error detected at Line {line_num}:\n{snippet}\n\nSUGGESTION: Review the snippet above line {line_num}."
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
        # wall time. Non-fatal — never block a real return.
        if workspace_model is not None and getattr(workspace_model, "enabled", False):
            try:
                workspace_model.record_command_outcome(
                    command=f"{runner} {filename}" if runner else str(filename),
                    exit_code=int(exit_code),
                    duration_seconds=0.0,
                    note=("failed" if exit_code != 0 else "ran"),
                )
            except Exception:  # noqa: BLE001
                pass

        if exit_code != 0:
             return _format_error(output, hint=diagnostic_info)

        return f"--- EXECUTION RESULT ---\nEXIT CODE: {exit_code}\nSTDOUT/STDERR:\n{output}"
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
