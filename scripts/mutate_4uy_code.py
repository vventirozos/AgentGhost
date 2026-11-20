#!/usr/bin/env python3
r"""§4UY battery — the code subsystem's §4UX-2 fixes (2026-11-18): C1 the leading-`cd` rewrite, C6/C21 one
overwrite/delete rule + kept versions + the project-wipe refusal, C7 smoke-test forms, C8 the deny list reads
code, C9/C10 one exit-status rule, C11 a poll of a running job, C12 the description, C13 stale servers, C14
server shapes, C15 command+content, C16 the syntax note, C17 the head envelope, C19 regex cost; plus the
C2–C4/C20 run-ledger mutants and the C18 spec-executor removal mutants; review r1 (C9 status file + runner rule,
C7 kill-after-start, C8 ANSI-C/braces/written-then-run, C13 owner+idle, the version store, the wipe rule). (C5 is a deletion: its pin is
tests/test_futility_breaker_trio.py::test_the_write_run_cycle_draws_no_futility_steer.)
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uy_code.py [-jN] [all|rerun M1 M2 …]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = "src/ghost_agent/tools/execute.py"; VA = "src/ghost_agent/tools/validators.py"
SA = "src/ghost_agent/tools/shell_analysis.py"; FSY = "src/ghost_agent/tools/file_system.py"
FV = "src/ghost_agent/tools/file_versions.py"; OUT = "src/ghost_agent/tools/outcome.py"
AG = "src/ghost_agent/core/agent.py"; SVC = "src/ghost_agent/sandbox/services.py"
REG = "src/ghost_agent/tools/registry.py"; CCL = "src/ghost_agent/core/code_claims.py"
FSO = "src/ghost_agent/tools/fs_ops.py"; OH = "src/ghost_agent/distill/outcome_heuristics.py"
RUNS = "src/ghost_agent/tools/runs.py"; MAIN = "src/ghost_agent/main.py"; DEL = "src/ghost_agent/tools/delegate.py"
MUT_CODE = [
    ("CONTROL known-bad", EX, r'''    """Does `cmd` end EVERY server it backgrounds, after starting it (see `_SMOKE_FORMS`)?"""
    # review B2-M3''', r'''    """Does `cmd` end EVERY server it backgrounds, after starting it (see `_SMOKE_FORMS`)?"""
    return True
    # review B2-M3'''),
    # C1 — the rewrite comes back
    ("C1 cd-strip-back", EX, r'''        # §4UY C1: the leading-`cd` rewrite is gone.''',
     r'''        command = re.sub(r"^\s*cd\s+/(?:opt|sandbox|app|root|home|usr/src)\S*\s*&&\s*", "", command)
        # §4UY C1: the leading-`cd` rewrite is gone.'''),
    # C15
    ("C15 content-dropped", EX, r'''        if content:
            pretty_log("Execute Ambiguous Args"''', r'''        if False:
            pretty_log("Execute Ambiguous Args"'''),
    # C8 — the deny list reads every byte again / r1: ANSI-C, braces, written-then-run
    ("C8a code-view-is-raw", VA, r'''    from .shell_analysis import _strip_heredocs
    text, _bodies = _strip_heredocs(str(cmd or ""))''', r'''    return str(cmd or "")
    from .shell_analysis import _strip_heredocs
    text, _bodies = _strip_heredocs(str(cmd or ""))'''),
    ("C8c redirect-is-target", VA, r'''            r = re.match(r"^\d*(?:[<>]{1,2}|&>>?|>\|)&?(\S*)$", a)''', r'''            r = None'''),
    ("C8d ansi-c-unread", VA, r'''        s = _ansi_c_unquote(s)                     # §4UY r1''', r'''        s = s                     # §4UY r1'''),
    ("C8e ansi-c-in-quotes", VA, r'''        if c == "$" and s[i + 1:i + 2] == "'":
            j, buf = i + 2, []''', r'''        if c == "$" and s[i + 1:i + 2] == "'" and False:
            j, buf = i + 2, []'''),
    ("C8f ansi-c-no-escapes", VA, r'''                    if e in _ANSI_C_ESC:''', r'''                    if False:'''),
    ("C8g ansi-c-no-hex", VA, r'''                        buf.append(chr(int(m.group(1), 16) if m.group(1) else int(m.group(2), 8)))''',
     r'''                        buf.append(m.group(0))'''),
    ("C8h braces-unread", VA, r'''    if "{" in t and "," in t:
        return any(''', r'''    if False:
        return any('''),
    ("C8i brace-empty-alt-dropped", VA, r'''    for alt in m.group(1).split(","):''', r'''    for alt in [a for a in m.group(1).split(",") if a]:'''),
    # C9/C10 — one exit-status rule (r1: status file, runners only, no trap when the command reads statuses)
    ("C9a no-list-failure", EX, r'''    if not failures:
        return 0, ""
    members = _list_members(command)''', r'''    return 0, ""
    members = _list_members(command)'''),
    ("C9b no-status-file", EX, r'''printf "%s\\n" "${__g_f[*]}") > "$__g_sf"; }''', r'''printf "%s\\n" "${__g_f[*]}") > /dev/null; }'''),
    ("C9c non-runner-counts", EX, r'''        cands = [(st, c) for st, c in cands if _runs_code(st)]''', r'''        cands = list(cands)'''),
    ("C9d trap-for-own-status", EX, r'''    traps = "" if _reads_own_status(command) else _status_traps(''', r'''    traps = "" if False else _status_traps('''),
    ("C9e last-stage-named", EX, r'''                 else [(stages[i], c) for i, c in enumerate(stage_codes) if c] if len(stage_codes) == len(stages)''',
     r'''                 else [(stages[i], c) for i, c in enumerate(stage_codes) if c] if False'''),
    ("C9f timeout-counts", EX, r'''        if _TIMEOUT_LED_RE.match(stage) and scode in (124, 137, 143):''', r'''        if False:'''),
    # the lstat S_ISREG check and O_NOFOLLOW both refuse a link (the flag closes the swap race): one mutant drops both
    ("C9g file-link-followed", EX, r'''        if not _stat.S_ISREG(os.lstat(p).st_mode):
            return []
        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))''',
     r'''        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))'''),
    ("C9h dir-link-followed", EX, r'''        if not _stat.S_ISDIR(os.lstat(p.parent).st_mode):
            return []''', r'''        if False:
            return []'''),
    ("C9i job-file-deleted", EX, r'''def status_failures_of_job(command_text, host_root, *, delete: bool = False) -> list:''',
     r'''def status_failures_of_job(command_text, host_root, *, delete: bool = True) -> list:'''),
    ("C9j no-age-out", EX, r'''                    and e.stat(follow_symlinks=False).st_mtime < cut:''', r'''                    and False:'''),
    ("C9k legacy-marker-kept", EX, r'''        text = _FAILED_LINE_RE.sub("\n", text)''', r'''        text = text'''),
    ("C9l run-logs-unstripped", RUNS, r'''            text = _take_status(str(text or ""))[0]      # §4UY C9''', r'''            text = str(text or "")      # §4UY C9'''),
    ("C9m collect-unread", RUNS, r'''            _fails = await asyncio.to_thread(status_failures_of_job, (_entry or {}).get("command"),''',
     r'''            _fails = await asyncio.to_thread(status_failures_of_job, None,'''),
    ("C9n wake-unread", MAIN, r'''            _fails = await asyncio.to_thread(status_failures_of_job, entry.get("command"),''',
     r'''            _fails = await asyncio.to_thread(status_failures_of_job, None,'''),
    ("C9o landing-unread", DEL, r'''        _fails = status_failures_of_job(entry.get("command"),''', r'''        _fails = status_failures_of_job(None,'''),
    ("C9p fg-unread", EX, r'''        return await asyncio.to_thread(status_failures_of_job, wrapped,''',
     r'''        return await asyncio.to_thread(status_failures_of_job, None,'''),
    ("C9q set-e-gets-trap", EX, r'''    r"|" + _CMD_POS + r"set\s+(?:-[a-zA-Z]*e|[-+]o\s+(?:errexit|pipefail)|\+e)"''', r'''    r"|xx"'''),
    ("C9r echo-set-e-no-trap", EX, r'''_CMD_POS = r"(?:^|[;&|(){]|\b(?:then|do|else)\b)\s*"''', r'''_CMD_POS = r"(?:^|[;&|(){\s]|\b(?:then|do|else)\b)\s*"'''),
    ("C9s note-keeps-keyword", EX, r'''        found = (scode, " ".join(re.sub(r"^\s*(?:(?:do|then|else|elif|if|while|until|\{|\(|!)\s+)+", "", stage).split()))''',
     r'''        found = (scode, " ".join(stage.split()))'''),
    # review r2
    ("C9t fifo-guards-gone", EX, r'''        if not _stat.S_ISREG(os.lstat(p).st_mode):
            return []
        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))''',
     r'''        fd = os.open(str(p), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))'''),
    ("C9u read-on-the-loop", EX, r'''        return await asyncio.to_thread(status_failures_of_job, wrapped,
                                       getattr(sandbox_manager, "host_workspace", None), delete=True)''',
     r'''        return status_failures_of_job(wrapped, getattr(sandbox_manager, "host_workspace", None), delete=True)'''),
    ("C9v no-errtrace", EX, r'''    return (f"set -E; __g_f=(); ''', r'''    return (f"__g_f=(); '''),
    ("C9w function-member-named", EX, r'''        cands = ([(t, code)] if len(stages) == 1''', r'''        cands = ([(stages[0], code)] if len(stages) == 1'''),
    ("C9x installer-is-a-run", EX, r'''    toks = str(text or "").split()
    i = next((k for k, w in enumerate(toks)''', r'''    return True
    toks = str(text or "").split()
    i = next((k for k, w in enumerate(toks)'''),
    ("C9y make-C-clean-is-a-run", EX, r'''(?:-[Cf]\s+\S+\s+|-\S+\s+)*(?:clean''', r'''(?:-\S+\s+)*(?:clean'''),
    ("C9z pip-module-is-a-run", EX, r'''-m\s+(?:pip|venv|virtualenv|ensurepip|http\.server|json\.tool|site|pydoc)\b"''',
     r'''-m\s+(?:venv|virtualenv|ensurepip|http\.server|json\.tool|site|pydoc)\b"'''),
    ("C7m stop-signal-ends", EX, r'''        if _NON_ENDING_SIGNAL_RE.match(args):
            continue''', r'''        if False:
            continue'''),
    ("C7n kill-zero-ends", EX, r'''    r"^\s*(?:-(?:0|l|L|''', r'''    r"^\s*(?:-(?:l|L|'''),
    ("C7o newest-start-off-by-one", EX, r'''        return _bisect.bisect_right(ends, at) - 1''', r'''        return _bisect.bisect_left(ends, at) - 2'''),
    # review r3 — the one heredoc reader, header targets in either order, $? kept, sourcing skips, function heads
    ("C8l delim-word-only", SA, r'''(?:(['\"])([\w.-]+)\1|''', r'''(?:(['\"])(\w+)\1|'''),
    ("C8m strip-skips-wrong", SA, r'''            i = docs[k][3]''', r'''            i = i + 1'''),
    ("C9aa source-keeps-trap", EX, r'''    r"PIPESTATUS|" + _CMD_POS + r"trap\s|" + _CMD_POS + r"(?:source|\.)\s+(?!" + _ACTIVATE_SCRIPT + r")\S"''', r'''    r"PIPESTATUS|" + _CMD_POS + r"trap\s"'''),
    # review r4 — data proof over the shipped deny list; heredoc operator; exact-stage `$?` forgive
    ("C8r live-deny-dropped", VA, r'''    for text in dict.fromkeys((raw, s)):
        hit = _live_deny_hit(text)''', r'''    for text in ():
        hit = _live_deny_hit(text)'''),
    ("C8s raw-text-skipped", VA, r'''    for text in dict.fromkeys((raw, s)):''', r'''    for text in dict.fromkeys((s,)):'''),
    ("C8t partial-span-exempt", VA, r'''            if not any(a <= m.start() and m.end() <= b for a, b in spans):''', r'''            if not any(a <= m.start() < b for a, b in spans):'''),
    ("C8u inert-any", VA, r'''    if heads and all(_inert(cmd) for cmd in heads):''', r'''    if heads and any(_inert(cmd) for cmd in heads):'''),
    ("C8v subst-not-bail", VA, r'''        if c == "`" or text[i:i + 2] in ("$(", "<(", ">(", "(("):
            return None''', r'''        if c == "`":
            return None'''),
    ("C8w continuation-splits", VA, r'''            if text[i + 1:i + 2] == "\n":               # a continuation joins the lines: no word boundary
                i += 2
                continue''', r'''            if text[i + 1:i + 2] == "\n":               # a continuation joins the lines: no word boundary
                i += 1
                continue'''),
    ("C8x ansi-escape-ignored", VA, r'''                j += 2 if (c == "$" and text[j] == "\\") else 1''', r'''                j += 1'''),
    ("C8y dq-subst-allowed", VA, r'''                if text[j] == "`" or text[j:j + 2] == "$(":
                    return None''', r'''                if False:
                    return None'''),
    ("C8z delim-trailing-space", VA, r'''                        if (line.lstrip("\t") if dash else line) == delim:''', r'''                        if (line.lstrip("\t") if dash else line).strip() == delim:'''),
    ("C8aa computed-delim-ok", VA, r'''                        return None                     # `<<EOF$x`, `<<'a'b`: a delimiter bash builds otherwise''', r'''                        pass'''),
    ("C8ab assignment-not-head", VA, r'''            heads.append([w])                           # an assignment `X=…` is a head too: never inert''', r'''            heads.append([w]) if "=" not in w else None'''),
    ("C8ac herestring-is-heredoc", SA, r'''_HEREDOC_OP_RE = re.compile(r"(?<!<)<<(?!<)''', r'''_HEREDOC_OP_RE = re.compile(r"<<'''),
    ("C8ad arith-is-heredoc", SA, r'''        if pre.count("((") > pre.count("))"):
            continue ''', r'''        if False:
            continue '''),
    ("C8ae fed-heredoc-is-data", SA, r'''        head = fed.group(1).lower() if fed else''', r'''        head = "cat" if fed else'''),
    ("C9ai substring-first", EX, r'''        for exact, rng in ((True, range(pos, len(norm))), (True, range(0, len(norm))),''', r'''        for exact, rng in ((False, range(pos, len(norm))), (True, range(0, len(norm))),'''),
    # review r5 — expanded heredoc bodies; evaluating options of inert heads
    ("C8af unquoted-body-data", VA, r'''        spans += [(a, b) for a, b, quoted in bodies if quoted or not re.search(r"[$`]", text[a:b])]''', r'''        spans += [(a, b) for a, b, quoted in bodies]'''),
    ("C8ag backslash-delim-unquoted", VA, r'''                    quoted = q in ("'", '"', "\\")''', r'''                    quoted = q in ("'", '"')'''),
    ("C8ah backtick-body-data", VA, r'''not re.search(r"[$`]", text[a:b])''', r'''not re.search(r"[$]", text[a:b])'''),
    ("C8ai options-ignored", VA, r'''    return not any(w.lstrip("'\"").startswith(bad) for w in words[1:]) if bad else True''', r'''    return True'''),
    ("C8aj test-R-ok", VA, r'''"test": ("-v", "-R"), "[": ("-v", "-R")}''', r'''"test": ("-v",), "[": ("-v",)}'''),
    ("C8ak sort-back", VA, r'''"test", "[", "sleep", "date",''', r'''"test", "[", "sleep", "date", "sort",'''),
    ("C8al args-not-collected", VA, r'''            heads[-1].append(w)                         # the command's arguments (its options are read)''', r'''            pass'''),
    # final seam review — a download keeps the version of its LANDED file; §4SD blocking reads
    ("S1 keep-not-at-landing", FSY, r'''                    await _keep_final(target_path, filename)            # the landed name, before the bytes''', r''''''),
    ("S2 hook-url-name", FSY, r'''                _dt = _landed.group(1) if _landed else _download_target(path, kwargs)''', r'''                _dt = _download_target(path, kwargs)'''),
    ("S3 status-read-nonblocking", EX, r'''        if not _stat.S_ISREG(_block_if_regular(fd).st_mode):''', r'''        if not _stat.S_ISREG(os.fstat(fd).st_mode):'''),
    ("S4 host-page-nonblocking", CCL, r'''        st = _block_if_regular(fd)''', r'''        st = os.fstat(fd)'''),
    # live probe 11-19 — a job kill ends a server only when the job IS its process
    ("J1 dollar-bang-any-job", EX, r'''                covered[n] = covered[n] or own[n]''', r'''                covered[n] = True'''),
    ("J2 var-kill-any-job", EX, r'''                    covered[held] = covered[held] or own[held]''', r'''                    covered[held] = True'''),
    ("J3 jobspec-any-job", EX, r'''                    covered[idx] = covered[idx] or spec[idx]''', r'''                    covered[idx] = True'''),
    ("J4 list-is-a-process", EX, r'''        return "own" if parts[-1].startswith("exec ") and _is_one_program(parts[-1][5:], fns) else ""''', r'''        return "own" if _is_one_program(re.sub(r"^exec\s+", "", parts[-1]), fns) else ""'''),
    ("J5 multi-without-exec", EX, r'''        return "own" if last.startswith("exec ") and _is_one_program(last[5:], fns) else ""
    p = _plain(j)''', r'''        return "own" if _is_one_program(re.sub(r"^exec\s+", "", last), fns) else ""
    p = _plain(j)'''),
    ("J25 single-group-needs-exec", EX, r'''        if g.group(1) == "(" and len(parts) == 1:''', r'''        if False:'''),
    ("J6 pipeline-not-spec", EX, r'''    spec = [k in ("own", "pipe") for k in kinds]''', r'''    spec = own'''),
    ("J12 server-not-last-ok", EX, r'''        if not parts or any(_SERVERISH_RE.search(x) or x.startswith("trap") for x in parts[:-1]):''', r'''        if not parts:'''),
    ("J14 function-is-own", EX, r'''    return bool(head) and head not in _UNPROVEN_HEADS and head not in fns''', r'''    return bool(head) and head not in _UNPROVEN_HEADS'''),
    ("J15 eval-payload-unscanned", EX, r'''\w*\}?)\s+(?:-\S+\s+)*-[A-Za-z]*c|\beval)"''', r'''\w*\}?)\s+(?:-\S+\s+)*-[A-Za-z]*c)"'''),
    ("J17 heredoc-body-read", EX, r'''    text = _HEREDOC_ANY_RE.sub(lambda m: m.group(1), str(cmd or ""))''', r'''    text = str(cmd or "")'''),
    ("J19 nested-shell-is-own", EX, r'''_UNPROVEN_HEADS = frozenset("bash sh zsh dash ksh fish eval source''', r'''_UNPROVEN_HEADS = frozenset("ksh fish source'''),
    ("J20 outer-kill-of-subshell-job", EX, r'''            if m.start() > closes[n]:''', r'''            if False:'''),
    ("J21 pipe-amp-is-start", EX, r'''    parts = re.split(r"(\n|;|&&|\|\||(?<![&>|])&(?![&>]))", text)''', r'''    parts = re.split(r"(\n|;|&&|\|\||(?<![&>])&(?![&>]))", text)'''),
    ("J23 group-redirects-refused", EX, r'''_GROUP_RE = re.compile(r"^([({])(.*)([)}])((?:''', r'''_GROUP_RE = re.compile(r"^([({])(.*)([)}])(?!\s*\S)((?:'''),
    ("J26 head-path-kept", EX, r'''        base = h.rsplit("/", 1)[-1]''', r'''        base = h'''),
    ("J27 wrappers-not-skipped", EX, r'''        if base in ("exec", "env", "nice", "nohup"):''', r'''        if base in ("exec", "nohup"):'''),
    ("J28 quoted-head-ok", EX, r'''        if re.search(r"[\\'\"$`]", h) or h == "v":''', r'''        if h == "v":'''),
    ("J30 pipe-any-stage", EX, r'''        if any(_SERVERISH_RE.search(x) for x in stages[1:]):''', r'''        if False:'''),
    ("J31 jobspec-across-subshell", EX, r'''                if 0 <= idx <= n and m.start() < closes[idx]:''', r'''                if 0 <= idx <= n:'''),
    ("J32 jobs-p-across-subshell", EX, r'''(spec[k] and m.start() < closes[k])''', r'''spec[k]'''),
    ("J33 quoted-shell-payload-unseen", EX, r'''    r"(?:(?:(?<![\w-])['\"\\]?(?:/[\w./-]*/)?(?:ba|z|da)?sh['\"]?|\$\{?[A-Za-z_]\w*\}?)\s+(?:-\S+\s+)*-[A-Za-z]*c|\beval)"''', r'''    r"(?:(?:\b(?:ba|z|da)?sh)\s+(?:-\S+\s+)*-[A-Za-z]*c|\beval)"'''),
    ("J35 nested-subst-once", EX, r'''    while prev != cmd:
        prev, cmd = cmd, _SUBST_RE.sub("v", cmd)''', r'''    if prev != cmd:
        prev, cmd = cmd, _SUBST_RE.sub("v", cmd)'''),
    ("J36 flag-cluster-unread", EX, r'''(?:-\S+\s+)*-[A-Za-z]*c|\beval)"''', r'''(?:-\S+\s+)*-c|\beval)"'''),
    ("J37 payload-job-credited", EX, r'''    kinds = ["" if _SH_C_PAYLOAD_RE.search(j) else _job_kind(j, fns) for j in jobs]''', r'''    kinds = [_job_kind(j, fns) for j in jobs]'''),
    ("J38 resolver-quadratic", EX, r'''    w, i, n = words, 0, len(words)''', r'''    w, i, n = list(words), 0, len(words)
    for _q in range(n):
        list(words)'''),
    ("J7 job-text-stops-at-andand", EX, r'''        elif depth == 0 and (c == ";" or (c == "&" and view[i - 1:i] != "&" and view[i + 1:i + 2] != "&"''', r'''        elif depth == 0 and (c == ";" or (c == "&"'''),
    ("J9 quotes-are-shape", EX, r'''    view = _shape_view(text)                          # quotes / comments / continuations read as shape only''', r'''    view = text'''),
    ("J10 newline-after-andand-ends", EX, r'''            if not re.search(r"(?:&&|\|\||\|)\s*$", view[max(0, i - 80):i]):''', r'''            if True:'''),
    ("J11 continuation-kept", EX, r'''                out[i] = out[i + 1] = " "
            i += 2''', r'''                pass
            i += 2'''),
    ("J16 case-paren-nests", EX, r'''            if depth == 0:
                out[k] = ";"''', r'''            if False:
                out[k] = ";"'''),
    ("J8 jobs-p-any-job", EX, r'''                covered[:n + 1] = [c or (spec[k] and m.start() < closes[k]) for k, c in enumerate(covered[:n + 1])]''', r'''                covered[:n + 1] = [True] * (n + 1)'''),
    ("C9ag activate-skips-trap", EX, r'''r"(?:source|\.)\s+(?!" + _ACTIVATE_SCRIPT + r")\S"''', r'''r"(?:source|\.)\s+\S"'''),
    ("C9ah activate-prefix-matches", EX, r'''                    r"(?=$|[\s;&|)])")''', r'''                    r"")'''),
    ("C9ab dollarq-skips-trap", EX, r'''    r"PIPESTATUS|" + _CMD_POS + r"trap\s|"''', r'''    r"PIPESTATUS|\$\?|" + _CMD_POS + r"trap\s|"'''),
    ("C9ac probe-not-forgiven", EX, r'''            if any(re.search(r"\$\?|\$\{\?\}", st) for st in nxt):
                continue''', r'''            if False:
                continue'''),
    ("C9ad any-later-read-forgives", EX, r'''            nxt = members[idx + 1] if idx + 1 < len(members) else []''', r'''            nxt = [st for m in members[idx + 1:] for st in m]'''),
    ("C9ae function-head-not-skipped", EX, r'''        if re.fullmatch(r"[\w.:-]+\(\)\{?", t) or t == "()" or (t == "{" and len(toks) > 1):''', r'''        if t == "()":'''),
    ("C9af function-keyword-kept", EX, r'''        if t == "function" and len(toks) > 1:
            toks = toks[2:]''', r'''        if t == "function" and len(toks) > 1:
            toks = toks[1:]'''),
    # C7 — smoke-test forms (r1: after the start, naming it, every server)
    ("C7a only-kill-percent", EX, r'''    pid_vars = {}                                     # VAR=$! → (the start it holds, where it was set)''',
     r'''    return bool(re.search(r"\bkill\s+%", text))
    pid_vars = {}                                     # VAR=$! → (the start it holds, where it was set)'''),
    ("C7b no-forms-in-message", EX, r'''_DAEMON_BLOCK_MSG += _SMOKE_FORMS + "."''', r'''_DAEMON_BLOCK_MSG += "."'''),
    ("C7c kill-before-start-ok", EX, r'''        if n < 0:
            continue                                  # a kill before any start ends nothing it started''',
     r'''        if n < 0:
            n = 0'''),
    ("C7d any-kill-ok", EX, r'''            if re.search(r"\$!|\$\{!\}", args):''', r'''            if True:'''),
    ("C7e one-server-enough", EX, r'''    return all(covered[i] for i in need)''', r'''    return any(covered[i] for i in need)'''),
    ("C7f pkill-any-name", EX, r'''                if any(nm and nm in starts[idx][2] for nm in names):''', r'''                if names:'''),
    ("C7g percent-n-is-last", EX, r'''                idx = int(g) - 1 if g.isdigit() else (n - 1 if g == "-" else n)''', r'''                idx = n'''),
    ("C7h pid-var-any-start", EX, r'''                if 0 <= held and set_at < at and m.start() < closes[held]:''', r'''                if 0 <= held and m.start() < closes[held]:'''),
    ("C7i trap-runs-in-place", EX, r'''        at = len(text) if any(a <= m.start() < b for a, b in traps) else m.start()''', r'''        at = m.start()'''),
    ("C7j sleep-is-a-server", EX, r'''    need = [i for i, (_s, _e, seg) in enumerate(starts) if _SERVERISH_RE.search(seg)] or range(len(starts))''',
     r'''    need = range(len(starts))'''),
    ("C7k no-timeout-bound", EX, r'''    covered = [bool(_TIMEOUT_BOUNDED_RE.match(seg)) for _s, _e, seg in starts]''', r'''    covered = [False for _s, _e, seg in starts]'''),
    ("C7l jobs-p-ignored", EX, r'''            if re.search(r"\$\(\s*jobs\s+-p", args):''', r'''            if False:'''),
    # C14
    ("C14a compound-head-kept", EX, r'''        seg = re.sub(r"^(?:(?:then|do|else|elif|if|while|until|\{|!)\s+)+", "", seg).strip()''',
     r'''        seg = seg.strip()'''),
    ("C14b jupyter-list-server", EX, r'''r"(?!\s+(?:list|stop|password|--version|--help|-h)\b)" +''', r'''r"" +'''),
    ("C14c gunicorn-daemon-missed", EX, r'''                     or re.search(r"\bgunicorn\b[^;&|\n]*\s(?:--daemon|-D)(?=\s|$)", text)))   # §4UY C14''',
     r'''                     ))'''),
    # C11 — a poll of a running job is a "same result" repeat again
    ("C11 poll-is-a-repeat", AG, r'''                            and not _res_unresolved):''', r'''                            and True):'''),
    # C12 — the description
    ("C12 tmp-missing-again", REG, r'''/sandbox, /root, /app do not exist (a `cd` there fails); /tmp exists and is scratch.''',
     r'''/sandbox, /root, /app, /tmp do not exist.'''),
    # C13 (r1: owner's own turn, idle time)
    ("C13a no-reclaim", SVC, r'''        if not stale:
            return ""
        _age, k2 = max(stale)''', r'''        return ""
        _age, k2 = max(stale)'''),
    ("C13b project-reclaimed", SVC, r'''            if k2 == keep_key or e2.get("project_id"):''', r'''            if k2 == keep_key:'''),
    ("C13c any-origin-reclaims", SVC, r'''            if _scope is None or _scope.origin != ORIGIN_OWNER:
                return ""''', r'''            if False:
                return ""'''),
    ("C13d no-scope-reclaims", SVC, r'''            if _scope is None or _scope.origin != ORIGIN_OWNER:''', r'''            if _scope is not None and _scope.origin != ORIGIN_OWNER:'''),
    ("C13e start-time-not-idle", SVC, r'''            age = now - self._last_use(k2, e2)''', r'''            age = now - float(e2.get("started_at") or now)'''),
    ("C13f log-not-a-use", SVC, r'''            if _stat.S_ISREG(st.st_mode):
                times.append(st.st_mtime)''', r'''            if False:
                times.append(st.st_mtime)'''),
    ("C13g look-not-a-use", SVC, r'''                    reg[key]["used_at"] = time.time()''', r'''                    pass'''),
    ("C13h stop-unreported", SVC, r'''                if self._stale_note:            # §4UY C13: say which stale server made room''',
     r'''                if False:            # §4UY C13: say which stale server made room'''),
    # C16
    ("C16 browser-for-all", FSY, r'''    _web = str(filename).lower().rsplit("?", 1)[0].endswith((".js", ".mjs", ".cjs", ".html", ".htm"))''',
     r'''    _web = True'''),
    # C17
    ("C17 banner-anywhere", OUT, r'''_ENVELOPE_HEAD_LINES = 2''', r'''_ENVELOPE_HEAD_LINES = 10 ** 6'''),
    # C19
    ("C19 code-name-unanchored", CCL, r'''    r"(?<![\w.\-/])[\w.\-/]+\.''', r'''    r"[\w.\-/]+\.'''),
]

# ── C6/C21 + r1 items 2/7/8 (the version store, cost, the wipe rule) ──
CONTROL_VERSIONS = ("CONTROL no-op", FV, r'''def versions_note(kept: List[str], skipped: List[str]) -> str:''',
                    r'''def versions_note(kept: List[str], skipped: List[str]) -> str:  # no-op''')

MUT_VERSIONS = [
    # ── C6 / C21 (updated) ──
    ("C6a fs-keeps-nothing", FSY, r'''        kept, skipped = keep_previous([path], _outer_root(sandbox_dir))''',
     r'''        kept, skipped = [], []'''),
    ("C6b fs-no-note", FSY, r'''    if not unseen:
        return ("\n" + versions_note([], skipped)) if skipped else ""''', r'''    if True:
        return ("\n" + versions_note([], skipped)) if skipped else ""'''),
    ("C6c shell-keeps-nothing", EX, r'''        if not removed and not written:
            return 0''', r'''        if True:
            return 0'''),
    ("C6d script-not-kept", EX, r'''        if not is_ephemeral and isinstance(host_path, Path) and host_path.is_file():   # §4UY C6/C21''',
     r'''        if False:'''),
    ("C6e project-wipe-passes", SA, r'''                if (cwd.kind == "project" and cwd.depth == 0 and n in (".", "*", "./*")''',
     r'''                if (False and n in (".", "*", "./*")'''),
    ("C6f up-wipe-passes", SA, r'''                if (cwd.kind == "project" and _ups == cwd.depth and _parts[_ups:] == ["*"]''',
     r'''                if (False and _parts[_ups:] == ["*"]'''),
    ("C6g project-root-target-passes", SA, r'''    if re.fullmatch(r"(?:/workspace/)?projects/[0-9a-f]{12}(?:/\*|/\.)?", a):''', r'''    if False:'''),
    ("C6h build-dir-copied", EX, r'''                if hp is not None and hp.is_file():
                    written.append(hp)''', r'''                if hp is not None:
                    written.append(hp)'''),
    ("C21a no-prune", FV, r'''    for p in olds[:-KEEP_PER_FILE]:''', r'''    for p in olds[:0]:'''),
    ("C21b no-size-cap", FV, r'''        if size > MAX_FILE_BYTES:''', r'''        if False:'''),
    # ── item 2: the store outside the sandbox, symlink-safe ──
    ("V1 store-in-workspace", FV, r'''    return store_root() / key''', r'''    return Path(root) / ".ghost_versions"'''),
    ("V2 store-link-followed", FV, r'''                if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                    return None
            except FileNotFoundError:
                os.mkdir(cur, 0o700)''', r'''                if not stat.S_ISDIR(os.stat(cur).st_mode):
                    return None
            except FileNotFoundError:
                os.mkdir(cur, 0o700)'''),
    ("V3 source-link-followed", FV, r'''    if stat.S_ISLNK(st.st_mode):
        os.symlink(os.readlink(src), dest)''', r'''    if False:
        os.symlink(os.readlink(src), dest)'''),
    ("V4 source-parent-unresolved", FV, r'''        parent = os.path.realpath(path.parent)''', r'''        parent = os.path.abspath(path.parent)'''),
    ("V5 dest-not-excl", FV, r'''    flags_w = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)''',
     r'''    flags_w = os.O_WRONLY | os.O_CREAT | os.O_TRUNC'''),
    ("V6 read-store-unchecked", FV, r'''        want = os.path.join(os.path.realpath(base), *below.parts)
        return want if os.path.realpath(store) == want else None''',
     r'''        want = os.path.join(os.path.realpath(base), *below.parts)
        return os.path.realpath(store)'''),
    ("V9 restore-op-unrouted", FSY, r'''    if operation in ("versions", "restore"):
        return await tool_file_versions(''', r'''    if operation in ("versions-x",):
        return await tool_file_versions('''),
    ("V10 restore-not-advertised", REG, r'''"delete", "versions", "restore", "quarantine_list"''', r'''"delete", "quarantine_list"'''),
    ("V11 restore-keeps-nothing", FV, r'''    kept, _sk = keep_previous([target], root) if os.path.lexists(target) else ([], [])''',
     r'''    kept, _sk = [], []'''),
    # ── item 7: cost ──
    ("V12 walk-not-stopped", FV, r'''                        _one(Path(d) / n)
                        if spent:
                            break
                    if spent:
                        break''', r'''                        _one(Path(d) / n)'''),
    ("V13 no-kept-n-of-m", FV, r'''    if spent:
        skipped.append(f"kept {len(kept)} of''', r'''    if False:
        skipped.append(f"kept {len(kept)} of'''),
    ("V14 generated-dirs-walked", FV, r'''                    dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and x not in _SKIP_PARTS''',
     r'''                    dirs[:] = sorted(x for x in dirs if x not in _SKIP_PARTS'''),
    ("V15 named-generated-dir-walked", FV, r'''                if p.name in SKIP_DIRS:
                    continue''', r'''                if False:
                    continue'''),
    ("V16 no-age-expiry", FV, r'''        if now - t > MAX_AGE_S:''', r'''        if False:'''),
    ("V17 no-total-cap", FV, r'''        if total <= MAX_STORE_BYTES:
            break''', r'''        if True:
            break'''),
    ("V18 cap-drops-newest", FV, r'''    for t, size, p in sorted(keep):''', r'''    for t, size, p in sorted(keep, reverse=True):'''),
    ("V19 byte-budget-ignored", FV, r'''        if budget_files <= 0 or size > budget_bytes:''', r'''        if budget_files <= 0:'''),
    # ── item 8: the wipe rule ──
    ("W1 narrowers-ignored", SA, r'''        if any(e in _FIND_NARROWERS and not (k and expr[k - 1] in ("-not", "!"))''',
     r'''        if False and any(e in _FIND_NARROWERS and not (k and expr[k - 1] in ("-not", "!"))'''),
    ("W2 negated-narrower-narrows", SA, r'''        if any(e in _FIND_NARROWERS and not (k and expr[k - 1] in ("-not", "!"))''',
     r'''        if any(e in _FIND_NARROWERS'''),
    ("W3 type-narrows", SA, r'''_FIND_NARROWERS = frozenset({"-empty",''', r'''_FIND_NARROWERS = frozenset({"-type", "-empty",'''),
    ("W4 git-clean-top-passes", SA, r'''                             or (head == "git" and _git_clean_wipes(args)))):''', r'''                             or False)):'''),
    ("W5 git-clean-fd-wipes", SA, r'''    return ("d" in flags) and ("x" in flags or "X" in flags)''', r'''    return "d" in flags'''),
    # ── r2 M2: no narrower exempts a whole-folder removal ──
    ("M2a user-narrows", SA, r'''                             "-samefile", "-used"})''', r'''                             "-samefile", "-used", "-user", "-readable"})'''),
    ("M2b trees-ignored", SA, r'''               and (e == "-empty" or not _find_removes_trees(expr))''',
     r'''               and (e == "-empty" or True)'''),
    ("M2c empty-not-exempt", SA, r'''               and (e == "-empty" or not _find_removes_trees(expr))''',
     r'''               and (not _find_removes_trees(expr))'''),
    ("M2d exec-rm-r-not-tree", SA, r'''            if _is_recursive([a for a in tail if a.startswith("-")]):
                return True''', r'''            if False:
                return True'''),
    ("M2e maxdepth-delete-not-tree", SA, r'''        if any(e == "-maxdepth" and k + 1 < len(expr) and expr[k + 1] == "1" for k, e in enumerate(expr)):
            return True''', r'''        if False:
            return True'''),
    ("M2f type-d-delete-not-tree", SA, r'''        if any(e == "-type" and k + 1 < len(expr) and "d" in expr[k + 1].split(",")''',
     r'''        if False and any(e == "-type" and k + 1 < len(expr) and "d" in expr[k + 1].split(",")'''),
    # ── r2 m4: a narrowed find keeps versions of what it matches ──
    ("m4a narrowed-find-not-kept", EX, r'''            if head == "find" and not dt and not via_xargs:''', r'''            if False:'''),
    ("m4b filter-ignored", FV, r'''                        if not whole and not match(os.path.join(drel, n), False):''',
     r'''                        if False:'''),
    ("m4c matched-dir-not-whole", FV, r'''                        taken += [os.path.join(d, x) for x in dirs if match(os.path.join(drel, x), True)]''',
     r'''                        taken += []'''),
    ("m4g one-file-unfiltered", FV, r'''            elif os.path.lexists(p) and (match is None or match("", False)):''',
     r'''            elif os.path.lexists(p):'''),
    ("m4h find-start-written-whole", EX, r'''                if t in dt or t in _fstarts:''', r'''                if t in dt:'''),
    # ── r3: find's grammar (`-o`, `!`, `( )`, `,`) and `find … | xargs rm` ──
    ("r3a or-as-and", SA, r'''        return lambda *x: (lambda u, v: True if (u is True or v is True) else''',
     r'''        return lambda *x: (lambda u, v: True if (u is True and v is True) else'''),
    ("r3b not-not-inverted", SA, r'''        return lambda *x: (lambda u: None if u is None else not u)(a(*x))''',
     r'''        return lambda *x: (lambda u: None if u is None else u)(a(*x))'''),
    ("r3c and-false-ignored", SA, r'''        return lambda *x: (lambda u, v: False if (u is False or v is False) else''',
     r'''        return lambda *x: (lambda u, v: False if (u is False and v is False) else'''),
    ("r3d paren-ignored", SA, r'''        if t == "(":
            v, i = p_list(i + 1)''', r'''        if False:
            v, i = p_list(i + 1)'''),
    ("r3e comma-ignored", SA, r'''        while i < n and toks[i] == ",":''', r'''        while False:'''),
    ("r3f exec-words-read", SA, r'''            while j < n and toks[j] not in (";", "+"):''', r'''            while False:'''),
    ("r3g noarg-takes-arg", SA, r'''        if t in _FIND_NOARG or not t.startswith("-"):''', r'''        if not t.startswith("-"):'''),
    ("r3h unreadable-narrowed", SA, r'''        return f if i >= n else (lambda *_x: None)''', r'''        return f'''),
    ("r3i name-test-wrong", SA, r'''            if e == "-name":
                return fnmatch.fnmatchcase(name, pat)''', r'''            if e == "-name":
                return fnmatch.fnmatchcase(printed, pat)'''),
    ("r3j xargs-not-versioned", EX, r'''            elif via_xargs and head in ("rm", "unlink", "shred", "srm", "mv") and _prev_find is not None:''',
     r'''            elif False:'''),
    ("r3k removes-flag-ignored", SA, r'''    if not removes and not _find_removes(expr):''', r'''    if not _find_removes(expr):'''),
    ("r3l prev-find-kept", EX, r'''            _prev_find = args if (head == "find" and not via_xargs) else None''',
     r'''            _prev_find = args if (head == "find" and not via_xargs) else _prev_find'''),
    # ── r4: only what Python reads exactly as find does narrows ──
    ("r4a regex-dialect-trusted", SA, r'''            if _rt not in ("posix-extended", "egrep") or "\\" in pat or "[[:" in pat:''',
     r'''            if "\\" in pat or "[[:" in pat:'''),
    ("r4b regex-backslash-trusted", SA, r'''            if _rt not in ("posix-extended", "egrep") or "\\" in pat or "[[:" in pat:''',
     r'''            if _rt not in ("posix-extended", "egrep") or "[[:" in pat:'''),
    ("r4g posix-class-trusted", SA, r'''            if _rt not in ("posix-extended", "egrep") or "\\" in pat or "[[:" in pat:''',
     r'''            if _rt not in ("posix-extended", "egrep") or "\\" in pat:'''),
    ("r4c glob-escape-trusted", SA, r'''        elif "\\" in pat or "[^" in pat:''', r'''        elif "[^" in pat:'''),
    ("r4d caret-class-trusted", SA, r'''        elif "\\" in pat or "[^" in pat:''', r'''        elif "\\" in pat:'''),
    ("r4e start-printing-one-way", SA, r'''            printed = {start + rel if start.endswith("/") else start + "/" + rel,''',
     r'''            printed = {start.rstrip("/") + "/" + rel,'''),
    ("r4f regex-takes-all-narrows", SA, r'''                     and not (e in ("-regex", "-iregex") and _regex_takes_all(expr[k + 1]))]''',
     r'''                     ]'''),
    # ── r2 m1: one mutating-op list ──
    ("m1a restore-not-mutating", FSO, r'''"copy", "restore",''', r'''"copy",'''),
    ("m1b agent-cache-private-list", AG, r'''.strip().lower() in _FS_MUTATING_OPS)   # §4UY r2 m1: the one list''',
     r'''.strip().lower() in ["write", "edit", "replace", "download", "delete", "move", "rename", "unzip", "git_clone", "copy"])'''),
    ("m1c heuristics-private-list", OH, r'''from ..tools.fs_ops import FS_MUTATING_OPS as _FS_MUTATING_OPS  # noqa: E402 — §4UY r2 m1: the ONE list''',
     r'''_FS_MUTATING_OPS = frozenset({"write", "replace", "edit", "delete"})'''),
]

# ── C2/C3/C4/C20 (the run ledger) ──
CC = "src/ghost_agent/core/code_claims.py"
MUT_CLAIMS = [
    ('CL2 no-pass-subject', CC, '    r"|\\b\\d+\\s+(?:tests?\\s+)?passed\\b"', '    r"|\\bxx\\d+\\s+(?:tests?\\s+)?passed\\b"'),
    ('CL3 failure-word-ignored', CC, '        if _FAILURE_WORD_RE.search(_clause + " " + m.group(0) + " " + _after):\n            continue', '        if False:\n            continue'),
    ('CL4 negation-ignored', CC, '        if _NEGATION_RE.search(_clause) or _NEGATION_RE.search(m.group(0)):', '        if False:'),
    ('CL5 negated-failure-counts', CC, '    r"(?<!\\bno\\s)(?<!\\b0\\s)(?<!zero\\s)(?<!without\\s)(?<!any\\s)"', '    r""'),
    ('CL6 working-directory-claim', CC, '    r"|directory|dir|tree|copy|set|memory|hours|day)\\b)"', '    r")\\b)"'),
    ('CK1 suffix-everywhere', CC, '    if cwd is None and not str(name).startswith(("/", "~/")):\n        k = _norm(name, "")', '    if True:\n        k = _norm(name, "")'),
    ('CK2 basename-keys', CC, '    p = _PROJ_PREFIX_RE.sub("", p)\n    return p.strip("/").casefold()', '    p = _PROJ_PREFIX_RE.sub("", p)\n    return p.strip("/").rsplit("/", 1)[-1].casefold()'),
    ('CK3 cd-ignored', CC, '        if lead and lead[0] in ("cd", "pushd"):\n            cwd = _cd_target(lead, cwd)', '        if lead and lead[0] in ("cd", "pushd"):\n            pass'),
    ('CK4 project-prefix-kept', CC, '_PROJ_PREFIX_RE = re.compile(r"^projects/[0-9a-f]{12}(?:/|$)", re.I)', '_PROJ_PREFIX_RE = re.compile(r"^NEVER/[0-9a-f]{12}(?:/|$)", re.I)'),
    ('CK5 write-key-no-cd', CC, '                wk = _written_key(str(args.get("command") or ""), p)', '                wk = _norm(p, "")'),
    ('CN1 compile-is-run', CC, '    if _compile_only(prog, rest):\n        return "check"', '    if False:\n        return "check"'),
    ('CN2 help-is-run', CC, '    if any(r in _HELP_FLAGS for r in rest):\n        return "read"', '    if False:\n        return "read"'),
    ('CN3 mask-ignored', CC, '            if not (idx < len(seps) and _masked(seps, idx)):', '            if True:'),
    ('CN4 mask-on-and', CC, '        if s == "||":\n            return True', '        if s in ("||", "&&"):\n            return True'),
    ('CN6 content-names-credit', CC, '        else:\n            _credit_code_text(cr, content, base)\n    return cr, checked - cr["exact"]', '        else:\n            _credit_code_text(cr, content, base)\n            for nm in _CODE_NAME_RE.findall(content):\n                _add_path(cr, nm, base)\n    return cr, checked - cr["exact"]'),
    ('CN7 written-heredoc-is-commands', CC, '        if im:\n            fed.append((im.group("interp") or im.group("interp2"), m.group("body")))\n        return m.group("head")', '        if im:\n            fed.append((im.group("interp") or im.group("interp2"), m.group("body")))\n            return m.group("head")\n        return m.group(0)'),
    ('CN8 sh-content-as-names', CC, '            sub, sub_checked = _command_credits(content, base, scripts, 1)\n            _merge(cr, sub)', '            sub, sub_checked = _new_credits(), set()\n            for nm in _CODE_NAME_RE.findall(content):\n                _add_path(sub, nm, base)\n            _merge(cr, sub)'),
    ('CR1 run-tool-ignored', CC, '            _credit(led, _command_credits(cmd, "", texts)[0], i, ok)\n            continue', '            continue'),
    ('CR2 job-start-is-result', CC, '            if not cmd or where != "workspace" or (act != "collect" and life == "job"):', '            if not cmd or where != "workspace":'),
    ('CR3 run-failed-is-pass', CC, '    if st == "failed":\n        return False\n    if st == "ok":', '    if st == "failed":\n        return True\n    if st == "ok":'),
    ('CR4 lab-is-workspace', CC, '            if act == "start" and args.get("command") and where == "workspace":', '            if act == "start" and args.get("command"):'),
    ('CR5 no-dir-runners', CC, '    if prog in _DIR_RUNNERS and (sub in _DIR_RUNNERS[prog] or (prog == "make" and False)):', '    if False:'),
    ('CR7 binary-ignored', CC, '        _add_built(cr, head, cwd, scripts)                # `./app` a build of this request made', '        pass'),
    ('CR8 script-text-ignored', CC, '        if text is not None and k != _BUILDS_KEY:\n            _merge(cr, _command_credits(text, cwd, {p: t for p, t in scripts.items() if p != k}, depth + 1)[0])', '        if False:\n            pass'),
    ('CR9 module-paths-flat', CC, '    return [f"{p}.py", f"{p}/__main__.py", f"{p}/__init__.py"] if p else []', '    return [f"{p}.py"] if p else []'),
    ('CR10 pipe-fed-heredoc-ignored', CC, '                                r"|<<.*\\|\\s*(?P<interp2>python[\\d.]*|node|nodejs|deno|bash|sh|zsh|dash)(?:\\s+-\\S*)*\\s*-?\\s*$")', '                                r"|<<.*\\|\\s*(?P<interp2>NEVERMATCH)$")'),
    ('CR11 npm-build-is-run', CC, '        if not (prog in ("npm", "yarn", "pnpm", "bun") and script.lower() in _NPM_CHECK_SCRIPTS):', '        if True:'),
    ('CR12 cargo-build-is-run', CC, '        return subs is None or sub.lower() in subs', '        return subs is None'),
    ('CP1 every-script', CC, '                    if k == want or (not exact and k.endswith("/" + want)):', '                    if True:'),
    ('CP2 other-host-ok', CC, '            if (u.hostname or "").lower() not in _LOCAL_HOSTS:\n                continue\n            p = u.path or ""', '            p = u.path or ""'),
    ('CP3 edit-text-stale', CC, '                _apply_edit_text(texts, args)', '                pass'),
    ('CP4 served-exact', CC, '                    want, exact = _norm(src.lstrip("/"), ""), False', '                    want, exact = _norm(src.lstrip("/"), ""), True'),
    ('R2d builds-not-shared', CC, '    texts: dict = {_BUILDS_KEY: {}, _ROOTS_KEY: _host_roots(tools_run)}', '    texts: dict = {_ROOTS_KEY: _host_roots(tools_run)}'),
    ('R2e no-pythonpath', CC, '    for pp in pypath:\n        for part in str(pp or "").split(":"):', '    for pp in ():\n        for part in str(pp or "").split(":"):'),
    ('R2f no-src-base', CC, '    bases.append(_norm("src", cwd))\n    return list(dict.fromkeys(bases))', '    return list(dict.fromkeys(bases))'),
    ('R2g export-ignored', CC, '            pypath += [w.split("=", 1)[1] for w in lead[1:] if w.startswith("PYTHONPATH=")]', '            pass'),
    ('R2h suite-root-is-cwd', CC, '    while parts and re.match(r"^(?:tests?|specs?|__tests__)$", parts[-1]):\n        parts.pop()\n    return "/".join(parts)', '    return "/".join(parts)'),
    ('R2i nothing-is-claim', CC, '        if _NOTHING_AFTER_RE.search(_after):\n            continue', '        if False:\n            continue'),
    ('R2k colon-standalone', CC, '(?=\\s*(?:[.!]|$)|\\s+" + _CODE_OBJECT + r")"', '(?=\\s*(?:[.!:]|$)|\\s+" + _CODE_OBJECT + r")"'),
    ('R2l pythonpath-inline-ignored', CC, '    pypath = list(pypath) + [w.split("=", 1)[1] for w in ws if w.startswith("PYTHONPATH=")]', '    pypath = list(pypath)'),
    ('Q301 host-page-ignored (was R3a)', CC, '        if not pages:                                    # r3 (review M3): a page this request did not write whole\n            pages = _host_pages(page, led, texts, served)', '        if not pages:\n            pages = []'),
    ('Q302 no-scope-project (was R3b)', CC, '        if sc is not None and sc.project_id:\n            ids.append(str(sc.project_id))', '        if sc is not None and sc.project_id:\n            pass'),
    ('Q303 no-named-project (was R3c)', CC, '            for m in _PROJECT_REF_RE.findall(str(_args_of(t))):\n                if m not in ids:', '            for m in ():\n                if m not in ids:'),
    ('Q304 served-root-only (was R3d)', CC, '    if served:\n        for k in led:\n            parts = k.split("/")[:-1]', '    if False:\n        for k in led:\n            parts = k.split("/")[:-1]'),
    ('Q305 fs-slash-absolute (was R3g)', CC, '    if p.startswith("/") and not re.match(r"^/(?:workspace|sandbox)(?:/|$)", p):\n        p = p.lstrip("/")', '    if False:\n        p = p.lstrip("/")'),
    ('Q306 fs-host-path-kept (was R3h)', CC, '    p = re.sub(r"^/(?:[^/]+/)*?sandbox/", "/workspace/", p) if not p.startswith(("/workspace", "/sandbox")) else p', '    p = p'),
    ('Q307 fs-write-key-plain (was R3i)', CC, '                led[_fs_norm(p)] = _new_rec(p, i, "file_system")', '                led[_norm(p, "")] = _new_rec(p, i, "file_system")'),
    ('Q308 binary-ext-blocked (was R3j)', CC, '_BINARY_EXT_RE = re.compile(r"\\.(?:out|bin|exe|elf|run|x86_64|aarch64)$", re.I)', '_BINARY_EXT_RE = re.compile(r"NEVER$", re.I)'),
    ('Q309 built-not-merged (was R3l)', CC, '    for d, ix in other["built"].items():\n        into["built"][d] = max(into["built"].get(d, -1), ix)', '    for d, ix in ():\n        into["built"][d] = max(into["built"].get(d, -1), ix)'),
    ('Q310 java-class-ignored (was R3m)', CC, '            _add_path(cr, cls.replace(".", "/") + ".java", cwd)', '            pass'),
    ('Q311 java-jar-is-class (was R3n)', CC, '                    and rest[rest.index(r) - 1] not in ("-cp", "-classpath", "--class-path", "-jar")), "")', '                    ), "")'),
    ('Q312 importlib-ignored (was R3o)', CC, '    r"""(?:spec_from_file_location|SourceFileLoader|SourcelessFileLoader)\\(\\s*[^,()]+,\\s*|"""', '    r"""(?:NEVERspec)\\(\\s*[^,()]+,\\s*|"""'),
    ('Q313 tools-are-runs (was R3p)', CC, '    if _not_a_runner(prog, rest):\n        return "read"', '    if False:\n        return "read"'),
    ('Q314 pip-module-runs (was R3q)', CC, '        return mod in _PY_TOOL_MODULES or mod.split(".")[0] in ("pip", "venv", "ensurepip")', '        return False'),
    ('Q315 clean-runs (was R3r)', CC, '    if prog in ("make", "cargo", "go", "mvn", "gradle", "dotnet", "ninja") and sub in _CLEAN_SUBS:', '    if False:'),
    ('Q316 npx-tool-runs (was R3s)', CC, '    if prog in ("npx", "pnpx", "bunx") and sub.split("@")[0] in _NPX_TOOLS:', '    if False:'),
    ('Q317 spawn-ignored (was R3w)', CC, '        if depth < 3:\n            _credit_spawned(cr, w, cwd, scripts)', '        if False:\n            _credit_spawned(cr, w, cwd, scripts)'),
    ('Q318 code-texts-not-kept (was R3z)', CC, '            if p.lower().endswith((".sh", ".bash") + WEB_EXTS) or _is_code(p):', '            if p.lower().endswith((".sh", ".bash") + WEB_EXTS):'),
    ('Q319 rejected-is-failure (was R3A)', CC, '    if st == "rejected" or _NOT_RUN_RE.search(text):\n        return None', '    if st == "rejected" or _NOT_RUN_RE.search(text):\n        return False'),
    ('Q320 system-files-tracked (was R3B)', CC, '                if not _SYSTEM_FILE_RE.match(wk):', '                if True:'),
    ('Q321 system-re-broad (was R3C)', CC, '_SYSTEM_FILE_RE = re.compile(r"^/(?:etc|usr|proc|sys|boot|dev|lib|lib32|lib64|sbin|bin|var/lib|root/\\.)/")', '_SYSTEM_FILE_RE = re.compile(r"^/")'),
    ('Q322 partial-eval-failure (was R3D)', CC, '    if _own_script_throws_only(text):\n        return True', '    if False:\n        return True'),
    ('Q323 partial-any-error-ok (was R3E)', CC, '        a[1] == "evaluate" and "timed out" not in a[2] for a in errs)', '        True for a in errs)'),
    ('Q324 append-replaces (was R3G)', CC, '                if args.get("append") in (True, "true", "True", 1):     # r3: a part appended to the text so far', '                if False:'),
    ('Q325 pre4uy-forms-lost (was R3H)', CC, '    r"(?:tested|verified|confirmed|validated|functional|working|passing)\\b(?!\\s+(?:directory|dir|tree|copy)\\b)"', '    r"(?:NEVERtested)\\b(?!\\s+(?:directory|dir|tree|copy)\\b)"'),
    ('Q326 fully-functional-lost (was R3I)', CC, '    r"|\\b(?:fully|successfully|thoroughly)\\s+(?:functional|working)\\b(?!\\s+(?:directory|dir|tree|copy)\\b)"', '    r"|\\bNEVERfully\\b"'),
    ('Q327 partial-eval-timeout-ok (was R3F)', CC, '        a[1] == "evaluate" and "timed out" not in a[2] for a in errs)', '        a[1] == "evaluate" for a in errs)'),
    ('R3W bare-working-claims', CC, '    r"|\\b(?:works|runs\\s+(?:fine|correctly|cleanly|without\\s+(?:any\\s+)?errors?)"', '    r"|\\b(?:works|working|runs\\s+(?:fine|correctly|cleanly|without\\s+(?:any\\s+)?errors?)"'),
    ('R3X predicate-working-missed', CC, '    r"(?:(?:now|all|fully|properly|correctly|still)\\s+)?working"', '    r"(?:(?:now|all|fully|properly|correctly|still)\\s+)?workingXX"'),
    ('Q402 spawn-not-run', CC, '        if n and _norm(n, cwd) != k:\n            _add_path(cr, n, cwd)', '        if False:\n            _add_path(cr, n, cwd)'),
    ('Q405 compiled-ext-includes-py', CC, '_COMPILED_EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".rs", ".go", ".java", ".kt", ".kts",', '_COMPILED_EXTS = (".py", ".sh", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".rs", ".go", ".java", ".kt", ".kts",'),
    ('Q406 compiled-ext-no-c', CC, '_COMPILED_EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".rs", ".go", ".java", ".kt", ".kts",', '_COMPILED_EXTS = (".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".rs", ".kt", ".kts",'),
    ('Q407 page-escapes-root', CC, '        if not f.parent.resolve().is_relative_to(root.resolve()):\n            return None', '        if False:\n            return None'),
    ('Q408 host-follows-links', CC, '        fd = os.open(str(f), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))', '        fd = os.open(str(f), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))'),
    ('Q409 host-reads-fifo', CC, '        if not _stat.S_ISREG(st.st_mode) or st.st_size > _HOST_PAGE_MAX:', '        if st.st_size > _HOST_PAGE_MAX:'),
    ('Q410 validated-is-claim', CC, '    r"(?:verified|confirmed)\\b(?=\\s*(?:[.!]|$)|\\s+" + _CODE_OBJECT + r")"', '    r"(?:verified|confirmed|validated)\\b(?=\\s*(?:[.!]|$)|\\s+" + _CODE_OBJECT + r")"'),
    ('Q411 no-verify-stems', CC, '    r"(?:verified|confirmed)\\b(?=\\s*(?:[.!]|$)|\\s+" + _CODE_OBJECT + r")"', '    r"(?:NEVERverified)\\b(?=\\s*(?:[.!]|$)|\\s+" + _CODE_OBJECT + r")"'),
    ('Q412 verify-any-object', CC, '    r"(?:verified|confirmed)\\b(?=\\s*(?:[.!]|$)|\\s+" + _CODE_OBJECT + r")"', '    r"(?:verified|confirmed)\\b"'),
    # r5 — conservative credit (review round 4)
    ("CN5 suite-everywhere", CC, r"""        else:
            _add_dir(cr, _suite_root(cwd), _suite_exts(seg))""", r"""        else:
            _add_dir(cr, "", _suite_exts(seg))"""),
    ("CR6 node-dot-ignored", CC, r"""            _add_dir(cr, None if cwd is None else _norm(tgt, cwd), _JS_EXTS)  # `node .` / `node server/`""", r"""            pass"""),
    ("Q501 dirs-any-language", CC, r"""    if not web and any((d == "" or key == d or key.startswith(d + "/")) and key.endswith(ex)""", r"""    if not web and any((d == "" or key == d or key.startswith(d + "/"))"""),
    ("Q503 suite-lang-unread", CC, r"""    for rx, exts in _SUITE_LANGS:
        if rx.search(low):
            return exts
    return ()""", r"""    return _PY_EXTS + _JS_EXTS + (".sh",)"""),
    ("Q504 make-has-language", CC, r"""                "gradle": (".java", ".kt", ".scala")}               # make / cmake: nothing by itself""", r"""                "gradle": (".java", ".kt", ".scala"), "make": (".c", ".sh", ".py")}"""),
    ("Q505 runner-any-lang", CC, r"""                _add_dir(cr, cwd, _RUNNER_EXTS.get(prog, ()))""", r"""                _add_dir(cr, cwd, _PY_EXTS + _JS_EXTS + (".rs", ".go", ".c"))"""),
    ("Q506 pkg-dir-any-lang", CC, r"""                    _add_dir(cr, None if base is None else _norm(mod.replace(".", "/"), base), _PY_EXTS)""", r"""                    _add_dir(cr, None if base is None else _norm(mod.replace(".", "/"), base), _PY_EXTS + (".sh",))"""),
    ("Q507 node-dir-any-lang", CC, r"""            _add_dir(cr, None if cwd is None else _norm(tgt, cwd), _JS_EXTS)  # `node .` / `node server/`""", r"""            _add_dir(cr, None if cwd is None else _norm(tgt, cwd), _JS_EXTS + (".py",))"""),
    ("Q508 suite-js-is-py", CC, r"""    (re.compile(r"(?:^|\s)(?:py\.?test|pytest|unittest|nose2?|tox|nox|ward)(?:\s|$)"), _PY_EXTS),""", r"""    (re.compile(r"(?:^|\s)(?:py\.?test|pytest|unittest|nose2?|tox|nox|ward)(?:\s|$)"), _PY_EXTS + (".js", ".sh")),"""),
    ("Q509 go-test-no-lang", CC, r"""    (re.compile(r"(?:^|\s)go\s+test(?:\s|$)"), (".go",)),""", r"""    (re.compile(r"(?:^|\s)go\s+test(?:\s|$)"), (".go", ".py")),"""),
    ("Q510 js-suite-lost", CC, r'''    (re.compile(r"(?:^|\s)(?:jest|vitest|mocha|ava|tap|karma|jasmine|playwright|cypress|npm|npx|yarn|pnpm|bun|node"''', r'''    (re.compile(r"(?:^|\s)(?:NEVERjest"'''),
    # r5 — the script a run runs
    ("Q511 args-are-runs", CC, r"""        return [r]
    return []""", r"""        return [x for x in rest if not x.startswith("-")]
    return []"""),
    ("Q512 script-not-credited", CC, r"""            continue                                     # (substitutions / loop variables below)
        _add_path(cr, w, cwd)""", r"""            continue
        pass"""),
    ("Q513 value-opts-ignored", CC, r"""            skip = r in _INTERP_VALUE_OPTS""", r"""            skip = False"""),
    ("Q514 inline-flags-ignored", CC, r"""        if r in _INTERP_NO_SCRIPT:
            return []""", r"""        if False:
            return []"""),
    ("Q515 head-code-ignored", CC, r"""    if _is_code(head):
        return [head]""", r"""    if False:
        return [head]"""),
    ("Q516 uv-run-ignored", CC, r"""    if prog in ("uv", "poetry", "pipenv", "pdm", "hatch", "rye") and rest[:1] == ["run"]:""", r"""    if False:"""),
    ("Q517 npx-tsx-ignored", CC, r"""    if prog in ("npx", "pnpx", "bunx", "pnpm", "yarn") and rest and rest[0].lower() in ("tsx", "ts-node", "node"):""", r"""    if False:"""),
    ("Q518 go-run-files-ignored", CC, r"""        return [r for r in rest[1:] if r.endswith(".go")]""", r"""        return []"""),
    ("Q519 deno-run-ignored", CC, r"""    if prog in ("deno", "bun") and rest[:1] == ["run"]:
        rest = rest[1:]""", r"""    if False:
        rest = rest[1:]"""),
    ("Q520 func-def-ignored", CC, r"""    elif ws and re.fullmatch(r"[\w.-]+\(\)", ws[0]):     # `t() { python3 app.py; }`""", r"""    elif False:"""),
    ("Q521 function-kw-ignored", CC, r"""    if ws[:1] == ["function"]:                           # `function t { python3 app.py; }`""", r"""    if False:"""),
    ("Q522 subst-any-body", CC, r"""            if re.match(r"^(?:find|ls|realpath|readlink|which|echo|printf)\b", body):""", r"""            if True:"""),
    ("Q523 subst-ignored", CC, r"""                for n in _CODE_NAME_RE.findall(body):
                    _add_path(cr, n, cwd)""", r"""                for n in ():
                    _add_path(cr, n, cwd)"""),
    ("Q524 loop-any-position", CC, r"""            if w in (f"${var}", f"${{{var}}}", f'"${var}"', f'"${{{var}}}"'):""", r"""            if f"${var}" in seg:"""),
    ("Q525 loop-ignored", CC, r"""                for n in _CODE_NAME_RE.findall(items):
                    _add_path(cr, n, cwd)""", r"""                for n in ():
                    _add_path(cr, n, cwd)"""),
    # r5 — strings / comments / data lists
    ("Q526 spans-ignored", CC, r"""    return [m for m in rx.finditer(code) if not _in_spans(spans, m.start())]""", r"""    return list(rx.finditer(code))"""),
    ("Q527 imports-in-strings", CC, r"""    for m in _live_matches(_IMPORT_ANY_RE, code):""", r"""    for m in _IMPORT_ANY_RE.finditer(str(code or "")):"""),
    ("Q528 refs-in-comments", CC, r"""    for m in _live_matches(_EXEC_REF_RE, code):""", r"""    for m in _EXEC_REF_RE.finditer(str(code or "")):"""),
    ("Q529 no-comment-span", CC, r"""    r"|`(?:\\.|[^`\\])*`|#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/")""", r"""    r"|`(?:\\.|[^`\\])*`")"""),
    ("Q530 list-literal-anywhere", CC, r'''    r"""|\(\s*\[\s*(?:['"](?:python[\d.]*|node|bash|sh|ruby|perl|php)['"]|sys\.executable)\s*,\s*['"]([\w.\-/]+)['"]"""''', r'''    r"""|(?:['"](?:python[\d.]*|node|bash|sh|ruby|perl|php)['"]|sys\.executable)\s*,\s*['"]([\w.\-/]+)['"]"""'''),
    ("Q531 spawn-any-ref", CC, r"""        n = m.group(2) or m.group(3)                     # the interpreter-plus-path alternatives only""", r"""        n = m.group(1) or m.group(2) or m.group(3)"""),
    # r5 — a binary credits the sources its build named
    ("Q532 build-not-noted", CC, r"""        if out and srcs:
            scripts[_BUILDS_KEY][out] = (int(scripts.get(_INDEX_KEY, 0)), frozenset(srcs))""", r"""        if False:
            scripts[_BUILDS_KEY][out] = (int(scripts.get(_INDEX_KEY, 0)), frozenset(srcs))"""),
    ("Q533 built-ignores-order", CC, r"""    if not web and any(write_idx <= ix and _built_source(key, src) for src, ix in cr["built"].items()):""", r"""    if not web and any(_built_source(key, src) for src, ix in cr["built"].items()):"""),
    ("Q534 built-any-source", CC, r"""        return key.endswith(".go") and folder == src[4:]
    return key == src""", r"""        return key.endswith(".go") and folder == src[4:]
    return True"""),
    ("Q535 go-folder-recursive", CC, r"""        return key.endswith(".go") and folder == src[4:]""", r"""        return key.endswith(".go") and (folder + "/").startswith(src[4:])"""),
    ("Q536 go-folder-any-ext", CC, r"""        return key.endswith(".go") and folder == src[4:]""", r"""        return folder == src[4:]"""),
    ("Q537 no-default-aout", CC, r'''            out = "a.out"''', r'''            out = None'''),
    ("Q538 dash-o-ignored", CC, r"""        if w == "-o":
            out = rest[j + 1] if j + 1 < len(rest) else None""", r"""        if w == "-o":
            out = None"""),
    ("Q541 go-files-ignored", CC, r"""            if w.endswith(".go"):
                srcs.append(_norm(w, cwd))""", r"""            if w.endswith(".go"):
                pass"""),
    ("Q542 rustc-no-output", CC, r"""        elif prog in ("rustc", "ghc", "swiftc", "nim", "zig"):""", r"""        elif False:"""),
    ("Q543 built-binary-key", CC, r"""    b = (scripts.get(_BUILDS_KEY) or {}).get(_norm(head, cwd))""", r"""    b = next(iter((scripts.get(_BUILDS_KEY) or {}).values()), None)"""),
    ("Q544 go-build-word-source", CC, r"""            if w == "build":
                continue""", r"""            if w == "build":
                srcs.append("\x00go:")"""),
    ("Q545 go-run-dir-credits", CC, r'''                    target = None                        # `go run main.go`: the named files only (above)''', r'''                    target = ""'''),
]
PINS_CLAIMS = [
    "tests/test_4uy_code_claims.py", "tests/test_4ph_coding_turn_review.py", "tests/test_4uu_coding.py",
    "tests/test_ft_live_fixes.py", "tests/test_4pl_failure_census.py", "tests/test_tool_interrupt.py",
    "tests/test_4kw_review_round6.py", "tests/test_agent_image_review_fixes.py", "tests/test_queued_messages.py",
    "tests/test_exec_exit_code_anchored_r41.py", "tests/test_outcome_consumers_r7.py",
    "tests/test_verifier_auto_repair.py",
]

# ── C18 (the spec executor removal) ──
PA = "src/ghost_agent/core/project_advancer.py"
CEX = "src/ghost_agent/core/coding_executor.py"
MUT_C18 = [
    ("CONTROL no-op (C18)", CEX, r'''async def build_coding_task(context, description: str, **_ignored) -> CodingResult:''',
     r'''async def build_coding_task(context, description: str, **_ignored) -> CodingResult:  # no-op'''),
    ("CONTROL known-bad (C18)", PA,
     r'''            logger.warning("task turn unavailable for %s — the task goes back to the queue", nxt.id)''',
     r'''            raise RuntimeError("known-bad")'''),
    ("C18a no-reopen-guard", PA,
     r'''        if classification == "coding" and getattr(coding_executor, "_ghost_real_executor", False) is True:''',
     r'''        if False:'''),
    ("C18b research-reopened-too", PA,
     r'''        if classification == "coding" and getattr(coding_executor, "_ghost_real_executor", False) is True:''',
     r'''        if getattr(coding_executor, "_ghost_real_executor", False) is True:'''),
    ("C18c claim-not-released", PA,
     r'''                store.release_claim(nxt.id, failure_reason="the task turn could not run")''',
     r'''                pass'''),
    ("C18d reopen-charged-as-work", PA,
     r'''f"only as an agent turn (needs the model, the project store and task turns on)",
                                 worked=False)''',
     r'''f"only as an agent turn (needs the model, the project store and task turns on)")'''),
    ("C18e batch-reads-it-as-budget", PA,
     r'''                                                 or res.summary.startswith(TASK_TURN_UNAVAILABLE_PREFIX)):''',
     r'''                                                 or False):'''),
    ("C18f marker-claims-a-build", CEX,
     r'''    return CodingResult(False, SPEC_EXECUTOR_REMOVED)''',
     r'''    return CodingResult(True, SPEC_EXECUTOR_REMOVED, files=["x.py"], verified=True)'''),
    ("C18g marker-unmarked", CEX,
     r'''_mark_real_executor(build_coding_task)''',
     r'''pass'''),
]
PINS_C18 = [
    "tests/test_4uy_code_spec.py", "tests/test_4ul_autopilot_turns.py", "tests/test_4ul_projects.py",
    "tests/test_4ul_review_fixes.py", "tests/test_4ul_review2.py", "tests/test_advance_pacing.py",
    "tests/test_4sr_gate_words.py", "tests/test_projects_api.py", "tests/test_project_advancer.py",
    "tests/test_advancer_fail_closed.py", "tests/test_4sd_executor_generate.py", "tests/test_4pg_autopilot_review.py",
    "tests/test_4qa_project_task_integrity.py", "tests/test_4mq_long_horizon.py", "tests/test_4su_phase5.py",
    "tests/test_4sx_d4_autopilot_labs.py", "tests/test_4fg_coding_loop.py", "tests/test_4ud_head_saves.py",
]

MUT = [CONTROL_VERSIONS] + MUT_CODE + MUT_VERSIONS + MUT_CLAIMS + MUT_C18
#: module-level regexes / constants: coverage cannot map import-time lines
FORCE = {
    "J36 flag-cluster-unread": ["tests/test_4uy_code_r2.py"],
    "J37 payload-job-credited": ["tests/test_4uy_code_r2.py"],
    "J38 resolver-quadratic": ["tests/test_4uy_code_r2.py"],
    "J26 head-path-kept": ["tests/test_4uy_code_r2.py"],
    "J27 wrappers-not-skipped": ["tests/test_4uy_code_r2.py"],
    "J28 quoted-head-ok": ["tests/test_4uy_code_r2.py"],
    "J30 pipe-any-stage": ["tests/test_4uy_code_r2.py"],
    "J31 jobspec-across-subshell": ["tests/test_4uy_code_r2.py"],
    "J32 jobs-p-across-subshell": ["tests/test_4uy_code_r2.py"],
    "J33 quoted-shell-payload-unseen": ["tests/test_4uy_code_r2.py"],
    "J35 nested-subst-once": ["tests/test_4uy_code_r2.py"],
    "J12 server-not-last-ok": ["tests/test_4uy_code_r2.py"],
    "J15 eval-payload-unscanned": ["tests/test_4uy_code_r2.py"],
    "J17 heredoc-body-read": ["tests/test_4uy_code_r2.py"],
    "J19 nested-shell-is-own": ["tests/test_4uy_code_r2.py"],
    "J20 outer-kill-of-subshell-job": ["tests/test_4uy_code_r2.py"],
    "J21 pipe-amp-is-start": ["tests/test_4uy_code_r2.py"],
    "J23 group-redirects-refused": ["tests/test_4uy_code_r2.py"],
    "J5 multi-without-exec": ["tests/test_4uy_code_r2.py"],
    "J25 single-group-needs-exec": ["tests/test_4uy_code_r2.py"],
    "J9 quotes-are-shape": ["tests/test_4uy_code_r2.py"],
    "J10 newline-after-andand-ends": ["tests/test_4uy_code_r2.py"],
    "J11 continuation-kept": ["tests/test_4uy_code_r2.py"],
    "J14 function-is-own": ["tests/test_4uy_code_r2.py"],
    "J16 case-paren-nests": ["tests/test_4uy_code_r2.py"],
    "J1 dollar-bang-any-job": ["tests/test_4uy_code_r2.py"],
    "J2 var-kill-any-job": ["tests/test_4uy_code_r2.py"],
    "J3 jobspec-any-job": ["tests/test_4uy_code_r2.py"],
    "J4 list-is-a-process": ["tests/test_4uy_code_r2.py"],
    "J6 pipeline-not-spec": ["tests/test_4uy_code_r2.py"],
    "J7 job-text-stops-at-andand": ["tests/test_4uy_code_r2.py"],
    "J8 jobs-p-any-job": ["tests/test_4uy_code_r2.py"],
    "C8a code-view-is-raw": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8c redirect-is-target": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8d ansi-c-unread": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8e ansi-c-in-quotes": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8f ansi-c-no-escapes": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8g ansi-c-no-hex": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8h braces-unread": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8i brace-empty-alt-dropped": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C9s note-keeps-keyword": ["tests/test_4uy_code_status.py"],
    "m1b agent-cache-private-list": ["tests/test_sandbox_cache_invalidation.py"],
    "C9b no-status-file": ["tests/test_4uy_code_status.py"],
    "C9q set-e-gets-trap": ["tests/test_4uy_code_status.py"],
    "C9r echo-set-e-no-trap": ["tests/test_4uy_code_status.py"],
    "C9l run-logs-unstripped": ["tests/test_4uy_code_status.py"],
    "C9m collect-unread": ["tests/test_4uy_code_status.py"],
    "C9n wake-unread": ["tests/test_4uy_code_status.py"],
    "C9o landing-unread": ["tests/test_4uy_code_status.py"],
    "C9p fg-unread": ["tests/test_4uy_code_status.py", "tests/test_execute_self_reported_exit.py",
                      "tests/test_4ph_coding_turn_review.py"],
    "C14b jupyter-list-server": ["tests/test_4uy_code.py"],
    "C14c gunicorn-daemon-missed": ["tests/test_4uy_code.py"],
    "C7b no-forms-in-message": ["tests/test_4uy_code.py", "tests/test_4uy_code_r2.py"],
    "C12 tmp-missing-again": ["tests/test_4uy_code.py"],
    "C17 banner-anywhere": ["tests/test_4uy_code.py", "tests/test_outcome_consumers_r3.py"],
    "C19 code-name-unanchored": ["tests/test_4uy_code.py"],
    "W3 type-narrows": ["tests/test_4uy_code.py", "tests/test_4uy_code_versions.py"],
    "M2a user-narrows": ["tests/test_4uy_code_versions.py"],
    "m1a restore-not-mutating": ["tests/test_sandbox_cache_invalidation.py"],
    "m1c heuristics-private-list": ["tests/test_sandbox_cache_invalidation.py"],
    "C9y make-C-clean-is-a-run": ["tests/test_4uy_code_status.py"],
    "C9z pip-module-is-a-run": ["tests/test_4uy_code_status.py"],
    "C7n kill-zero-ends": ["tests/test_4uy_code_r2.py"],
    "C8l delim-word-only": ["tests/test_4uy_code_r2.py"],
    "C8m strip-skips-wrong": ["tests/test_4uy_code_r2.py"],
    "C9aa source-keeps-trap": ["tests/test_4uy_code_status.py"],
    "C8r live-deny-dropped": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8s raw-text-skipped": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8t partial-span-exempt": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8u inert-any": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8v subst-not-bail": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8w continuation-splits": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8x ansi-escape-ignored": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8y dq-subst-allowed": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8z delim-trailing-space": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8aa computed-delim-ok": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8ab assignment-not-head": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8ac herestring-is-heredoc": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8ad arith-is-heredoc": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C8ae fed-heredoc-is-data": ["tests/test_4uy_code_r4.py", "tests/test_4uy_code_r2.py", "tests/test_4uy_code.py"],
    "C9ai substring-first": ["tests/test_4uy_code_status.py"],
    "C8af unquoted-body-data": ["tests/test_4uy_code_r4.py"],
    "C8ag backslash-delim-unquoted": ["tests/test_4uy_code_r4.py"],
    "C8ah backtick-body-data": ["tests/test_4uy_code_r4.py"],
    "C8ai options-ignored": ["tests/test_4uy_code_r4.py"],
    "C8aj test-R-ok": ["tests/test_4uy_code_r4.py"],
    "C8ak sort-back": ["tests/test_4uy_code_r4.py"],
    "C8al args-not-collected": ["tests/test_4uy_code_r4.py"],
    "S1 keep-not-at-landing": ["tests/test_4uy_code_versions.py"],
    "S2 hook-url-name": ["tests/test_4uy_code_versions.py"],
    "S3 status-read-nonblocking": ["tests/test_4uy_code_status.py", "tests/test_4sd_transient_write_retry.py"],
    "S4 host-page-nonblocking": ["tests/test_4uy_code_claims.py", "tests/test_4sd_transient_write_retry.py"],
    "C9ag activate-skips-trap": ["tests/test_4uy_code_status.py"],
    "C9ah activate-prefix-matches": ["tests/test_4uy_code_status.py"],
    "C9ab dollarq-skips-trap": ["tests/test_4uy_code_status.py", "tests/test_ft_live_fixes.py"],
    "C9ac probe-not-forgiven": ["tests/test_4uy_code_status.py", "tests/test_ft_live_fixes.py"],
    "C9ad any-later-read-forgives": ["tests/test_4uy_code_status.py"],
    "C9ae function-head-not-skipped": ["tests/test_4uy_code_r2.py"],
    "C9af function-keyword-kept": ["tests/test_4uy_code_r2.py"],
}
PINS_CODE = [
    "tests/test_4uy_code.py", "tests/test_4uy_code_status.py", "tests/test_4uy_code_r2.py",
    "tests/test_4uy_code_versions.py", "tests/test_4ll_file_handling.py", "tests/test_sandbox_cache_invalidation.py", "tests/test_4lo_execute.py", "tests/test_execute_self_reported_exit.py", "tests/test_ft_live_fixes.py",
    "tests/test_4ph_coding_turn_review.py", "tests/test_4sx_d5_review.py", "tests/test_futility_breaker_trio.py",
    "tests/test_4og_dev_loop.py", "tests/test_4oe_dev_loop.py", "tests/test_breaker_second_tier.py",
    "tests/test_breaker_closed_loop_stays_closed.py", "tests/test_same_error_survives_writes.py",
    "tests/test_4kw_review_round4.py", "tests/test_4kw_review_round5.py", "tests/test_4kw_review_round6.py",
    "tests/test_4tj_saves_back_and_unread_delete.py", "tests/test_sandbox_services.py",
    "tests/test_outcome_consumers_r3.py", "tests/test_outcome_consumers_r7.py",
    "tests/test_exec_exit_code_anchored_r41.py", "tests/test_tool_outcome_contract.py",
    "tests/test_4sx_r11_review.py", "tests/test_execute.py", "tests/test_execute_security.py",
    "tests/test_4pe_files_projects_review.py", "tests/test_4um_review.py", "tests/test_edit_ladder_ledger.py",
]
PINS = sorted(set(PINS_CODE + PINS_CLAIMS + PINS_C18))


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
        if label in FORCE:
            ids = sorted(set(ids or []) | set(FORCE[label]))
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
