#!/usr/bin/env python3
r"""§4UV battery — live functional run fixes (2026-11-18): foreground servers, status echo, capability walls, rm
/tmp rule, suite heads + data heredocs, local data fetch, self-loading overwrite, service cap message.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uv.py [-jN] [all|rerun M1 M2 …]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = "src/ghost_agent/tools/execute.py"; VA = "src/ghost_agent/tools/validators.py"; EL = "src/ghost_agent/core/env_limits.py"
AG = "src/ghost_agent/core/agent.py"; CC = "src/ghost_agent/core/code_claims.py"; PC = "src/ghost_agent/core/page_check.py"
SS = "src/ghost_agent/sandbox/services.py"
MUT = [
    ("CONTROL no-op", EX, r'''def _foreground_server_block(command: str):''', r'''def _foreground_server_block(command: str):  # no-op'''),
    ("CONTROL known-bad", EX, r'''        if _FG_SERVER_RE.search(seg):
            return _FG_SERVER_BLOCK_MSG''', r'''        if False:
            return _FG_SERVER_BLOCK_MSG'''),
    # F1 foreground servers
    ("F1a call-site", EX, r'''_db = _daemonized_server_block(command) or _foreground_server_block(command)''', r'''_db = _daemonized_server_block(command)'''),
    ("F1b kill-any", EX, r'''    if not cmd or (re.search(r"\bkill\s+%", cmd) and re.search(r"[^&]&(?!&)", cmd)):''', r'''    if not cmd or re.search(r"\bkill\s+%", cmd):'''),
    ("F1c heredoc-kept", EX, r'''    cmd = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n[ \t]*\2[ \t]*(?=\n|$)", "<<HEREDOC", cmd, flags=re.S)''', r'''    pass'''),
    ("F1d sh-c-skip", EX, r'''        if inner and _foreground_server_block(inner):''', r'''        if False:'''),
    ("F1e quotes-kept", EX, r'''    cmd = _QUOTED_RE.sub("''", cmd)''', r'''    cmd = cmd'''),
    ("F1f bg-blocked", EX, r'''        if not seg or seg.endswith("&"):''', r'''        if not seg:'''),
    ("F1g pipe-tail", EX, r'''        seg = seg.split("|", 1)[0].strip()''', r'''        seg = seg.split("|", 1)[-1].strip()'''),
    # F1h cd-prefix: EQUIVALENT — removed (see journal §4UV battery)
    ("F1i flask-h-help", EX, r'''        _usage = r"\s(?:--help|--version|-V)(?:\s|$)" if re.search(r"\bflask\s+run\b", seg)''', r'''        _usage = r"\s(?:--help|-h|--version|-V)(?:\s|$)" if re.search(r"\bflask\s+run\b", seg)'''),
    # F1j timeout-blocked: EQUIVALENT — removed (see journal §4UV battery)
    ("F1k sudo-u", EX, r'''    r"^(?:sudo\s+(?:-[A-Za-z]\s+\S+\s+|-\S+\s+)*|env\s+''', r'''    r"^(?:sudo\s+|env\s+'''),
    ("F1l npm-C", EX, r'''(?:(?:--prefix|-C)\s+\S+\s+)?''', r'''(?:--prefix\s+\S+\s+)?'''),
    ("F1m vite-optimize", EX, r'''(?!\s*(?:build|optimize))"''', r'''(?!\s*build)"'''),
    ("F1n end-word", EX, r'''_END = r"(?=\s|$|[;&|)])"''', r'''_END = r"\b"'''),
    ("F1o http-server", EX, r'''(?:http\.server|SimpleHTTPServer|flask\s+run|uvicorn|gunicorn)" + _END +''', r'''(?:SimpleHTTPServer|flask\s+run|uvicorn|gunicorn)" + _END +'''),
    # F2 status echo
    ("F2a always-ends", EX, r'''        if _RUNNER_HEAD_RE.match(head):
            return False
    return True''', r'''        if False:
            return False
    return True'''),
    ("F2b any-member-ends", EX, r'''        head = toks[0] if toks[0].startswith("./") else toks[0].rsplit("/", 1)[-1]
        if _RUNNER_HEAD_RE.match(head):''', r'''        head = toks[0] if toks[0].startswith("./") else toks[0].rsplit("/", 1)[-1]
        if head:'''),
    ("F2c brace-var", EX, r'''    i = max(cmd.rfind("$?"), cmd.rfind("${?}"))''', r'''    i = cmd.rfind("$?")'''),
    ("F2d wrappers", EX, r'''        while toks and (re.fullmatch(r"[A-Za-z_]\w*=.*", toks[0]) or toks[0] in _RUNNER_WRAPPERS
                        or toks[0] in ("do", "then", "else")):''', r'''        while toks and toks[0] in ("do", "then", "else"):'''),
    ("F2e timeout-flags", EX, r'''                while toks and toks[0].startswith("-"):''', r'''                while False:'''),
    ("F2f heredoc-closers", EX, r'''        if not toks or toks[0].startswith("#") or (len(toks) == 1 and toks[0] in closers):''', r'''        if not toks or toks[0].startswith("#"):'''),
    ("F2g quote-scan", EX, r'''        elif c in ";\n" or cmd.startswith("&&", j) or cmd.startswith("||", j):''', r'''        elif c in "\n":'''),
    ("F2h gate-off", EX, r'''    if ("$?" in str(command or "") or "${?}" in str(command or "")) and _status_echo_ends_the_run(str(command or "")):''', r'''    if ("$?" in str(command or "") or "${?}" in str(command or "")):'''),
    # F8 self-loading overwrite
    ("F8a call-site", EX, r'''    if _clobber:''', r'''    if False:'''),
    ("F8b loader-any-quote", EX, r'''    loads = (re.search(r"(?:spec_from_file_location|run_path|SourceFileLoader|exec\s*\(\s*open|load_source)"
                       r"\s*\([^)\n]*" + _q, new) or imports''', r'''    loads = (re.search(_q, new) or imports'''),
    ("F8c no-imports", EX, r'''                       r"\s*\([^)\n]*" + _q, new) or imports''', r'''                       r"\s*\([^)\n]*" + _q, new)'''),
    ("F8d test-exempt-all", EX, r'''    if re.fullmatch(r"test_\w+\.py|\w+_test\.py|\w+\.(?:test|spec)\.js", base) and not imports:''', r'''    if re.fullmatch(r"test_\w+\.py|\w+_test\.py|\w+\.(?:test|spec)\.js", base):'''),
    ("F8e same-content", EX, r'''    if not old.strip() or old.strip() == new.strip():''', r'''    if not old.strip():'''),
    # F8f missing-file: EQUIVALENT — read_text of a missing file raises OSError → ""
    ("F8g require-js", EX, r'''             or (base.endswith(".js") and re.search(''', r'''             or (False and re.search('''),
    ("F8h dot-prefix", EX, r'''    while rel.startswith("./"):
        rel = rel[2:]''', r'''    rel = rel.lstrip("./")'''),
    # F4 rm rule
    ("F4a newline", VA, r'''        r"\brm\b(?=[^|;&\n]*\s(?:-\w*r|--recursive))(?=[^|;&\n]*\s(?:-\w*f|--force))[^|;&\n]*(?<![<>])\s+"''', r'''        r"\brm\b(?=[^|;&]*\s(?:-\w*r|--recursive))(?=[^|;&]*\s(?:-\w*f|--force))[^|;&]*(?<![<>])\s+"'''),
    ("F4b redirect", VA, r'''[^|;&\n]*(?<![<>])\s+"''', r'''[^|;&\n]*\s+"'''),
    ("F4c dotdot", VA, r'''        r"(?:/(?!tmp/(?![\w\-./]*\.\.)[\w\-][\w\-./]*(?:\s|$))''', r'''        r"(?:/(?!tmp/[\w\-][\w\-./]*(?:\s|$))'''),
    ("F4d quoted-dotdot", VA, r'''r"|['\"]\s*(?:/(?!tmp/(?![\w\-./ ]*\.\.)''', r'''r"|['\"]\s*(?:/(?!tmp/'''),
    ("F4e message", VA, r'''            if pat is _SHELL_DENY[0]:''', r'''            if False:'''),
    # F3 capability note
    ("F3a exit0", EL, r'''    if not rid or exit_code in (0, None) or not (privilege_refused(output) or''', r'''    if not rid or not (privilege_refused(output) or'''),
    ("F3b no-rid", EL, r'''    if not rid or exit_code in (0, None) or not (privilege_refused(output) or
                                      any(''', r'''    if exit_code in (0, None) or not (privilege_refused(output) or
                                      any('''),
    ("F3c quotes-kept", EL, r'''    text = "\n".join(re.sub(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"", "''", ln) for ln in _command_lines(command))''', r'''    text = "\n".join(_command_lines(command))'''),
    ("F3d script-any", EL, r'''    if not what and _RUNS_A_SCRIPT_RE.search(text):''', r'''    if not what:'''),
    ("F3e once", EL, r'''        if key in row:
            return ""''', r'''        if False:
            return ""'''),
    ("F3f mount-pipe", EL, r'''(?:mount\s+(?:-\S+\s+)*[^\s|;&<>]+\s+[^\s|;&<>]+|''', r'''(?:mount\s+(?:-\S+\s+)*\S+\s+\S+|'''),
    ("F3g tc-show", EL, r'''tc\s+qdisc\s+(?:add|del|change|replace)|''', r'''tc\s+qdisc|'''),
    ("F3h ip-abbrev", EL, r'''(?:l|link|a|addr|address|r|route|ru|rule|n|neigh|tuntap)\s+''', r'''(?:link|addr|address|route|rule|neigh|tuntap)\s+'''),
    ("F3i wrappers", EL, r'''       r"(?:[A-Za-z_]\w*=\S*\s+|env\s+(?:-\S+\s+)*(?:\S+=\S*\s+)*|timeout\s+\S+\s+|nice\s+(?:-n\s*\S+\s+)?)*"''', r'''       r"''"'''),
    ("F3j abs-path", EL, r'''       r"(?:/[\w./-]*/)?")''', r'''       r"")'''),
    ("F3k syntax-check", EL, r'''sh\s+(?!-\w*[nc]\b)''', r'''sh\s+'''),
    ("F3l wiring", AG, r'''                            if _cn:
                                from ..tools.outcome import with_text as _with_text''', r'''                            if False:
                                from ..tools.outcome import with_text as _with_text'''),
    # F6 suite heads / data heredocs
    ("F6a heads", AG, r'''(?:(?-i:(?:do|then|else|elif|time|if|while|until)[ \t]+)|![ \t]+|\{[ \t]+|''', r'''(?:'''),
    ("F6b heads-ci", AG, r'''(?-i:(?:do|then|else|elif|time|if|while|until)[ \t]+)''', r'''(?:(?:do|then|else|elif|time|if|while|until)[ \t]+)'''),
    ("F6c heredoc-kept", AG, r'''    text = _NONSHELL_HEREDOC_RE.sub(_heredoc, blob)''', r'''    text = blob'''),
    ("F6d shell-heredoc-dropped", AG, r'''        return m.group(0) if _SHELL_HEREDOC_HEAD_RE.search(m.group("head")) else m.group("head")''', r'''        return m.group("head")'''),
    ("F6e quotes-kept", AG, r'''    return bool(_SUITE_CMD_RE.search(_suite_text(str(blob or ""))))''', r'''    return bool(_SUITE_CMD_RE.search(str(blob or "")))'''),
    ("F6f py-content", CC, r'''    return keys, bool(is_suite_command(cmd) or (_sh and is_suite_command(content))), checked - keys''', r'''    return keys, bool(is_suite_command(cmd) or is_suite_command(content)), checked - keys'''),
    # F7 local fetch
    ("F7a caller", AG, r'''                if extra:
                    note = _pc.note_for(res, wrote_html, local_fetch=True) + extra''', r'''                if False:
                    note = _pc.note_for(res, wrote_html, local_fetch=True) + extra'''),
    ("F7b comment-skip", PC, r'''            if re.search(r"(?<![:\\])//|/\*|<!--", lead) or lead.lstrip().startswith("*"):''', r'''            if False:'''),
    ("F7c strings-are-comments", PC, r'''            lead = re.sub(r"'[^'\n]*'|\"[^\"\n]*\"|`[^`\n]*`", "''", lead)''', r'''            pass'''),
    ("F7d abs-url", PC, r'''(?!https?:|//|/api/|\$\{)''', r'''(?!//|/api/|\$\{)'''),
    ("F7e conditional", PC, r'''        if local_fetch:''', r'''        if False:'''),
    # cap message
    ("S1 member-names", SS, r'''                    if _member():
                        _names = ["…"]''', r'''                    if False:
                        _names = ["…"]'''),
    ("S2 names-off", SS, r'''already running ({', '.join(_names)})''', r'''already running'''),
]
FORCE = {"F7d abs-url": ["test_battery_local_fetch_ignores_absolute_urls"],
         "F3c quotes-kept": ["test_battery_capability_note_needs_a_failure_and_ignores_quoted_ops"]}
PINS = ["tests/test_ft_live_fixes.py", "tests/test_execute_self_reported_exit.py", "tests/test_4og_dev_loop.py",
        "tests/test_4oe_dev_loop.py", "tests/test_sandbox_services.py", "tests/test_audit_fixes_20260725.py",
        "tests/test_recent_fixes.py"]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        if label in FORCE:      # module-level regex: coverage cannot see which test exercises it
            ids = sorted(set(ids or []) | {f"tests/test_ft_live_fixes.py::{t}" for t in FORCE[label]})
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(PINS), f"fallback {len(PINS)} files"
    print(f"{len(PINS)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
