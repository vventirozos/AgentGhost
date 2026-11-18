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


#: formatters / fixers and the flag that makes them rewrite the files they name
_FIXERS = {"black": None, "isort": None, "autopep8": "-i", "yapf": "-i", "prettier": "--write", "eslint": "--fix",
           "ruff": "--fix", "gofmt": "-w", "rustfmt": None, "clang-format": "-i", "standard": "--fix"}


#: each formatter's own check-only flags (`-l` is black/isort's line length, eslint's `-c` its config)
_FIXER_CHECK_FLAGS = {"black": {"--check", "--diff"}, "isort": {"--check-only", "--check", "-c", "--diff"},
                      "ruff": {"--check", "--diff"}, "rustfmt": {"--check"},
                      "prettier": {"--check", "-c", "--list-different", "-l"}, "yapf": {"--diff", "-d"},
                      "autopep8": {"--diff", "-d"}, "clang-format": {"--dry-run", "-n"}, "gofmt": {"-l", "-d"}}


def _inplace_written(cmd: str) -> list:
    """§4UU: code files a command rewrites IN PLACE — `sed -i`, `perl -pi`, `cp`/`mv`/`install` onto a code
    file, `git checkout`/`restore` of named files, formatters with their write flag (`prettier --write`,
    `ruff format`, `black`). Each is a new version of the file: it was not run yet."""
    out = []
    for words in _shell_segments(cmd):
        words = _strip_shell_lead(list(words))
        while words and (re.match(r"^[A-Za-z_]\w*=", words[0]) or words[0] in ("sudo", "time", "nohup", "command", "npx", "-y", "--yes")):
            words = words[1:]
        if not words:
            continue
        prog = words[0].rsplit("/", 1)[-1]
        args = words[1:]
        files = [a for a in args if not a.startswith("-") and _is_code(a)]
        if prog in _FIXER_CHECK_FLAGS and any(a in _FIXER_CHECK_FLAGS[prog] for a in args) \
                and not (_FIXERS.get(prog) and _FIXERS[prog] in args):
            continue                                    # a formatter that only CHECKS rewrites nothing
        if prog in ("sed", "gsed") and any(re.match(r"^-[nErsuz]*i", a) or a.startswith("--in-place") for a in args):
            out.extend(files)
        elif prog == "perl" and any(re.match(r"^-[pnlae0]*i", a) for a in args):
            out.extend(files)
        elif prog in ("cp", "install") and len(files) >= 1 and args and _is_code(args[-1]):
            out.append(args[-1])                        # (`mv a.py b.py` renames: the content is unchanged)
        elif prog == "git" and args[:1] and args[0] in ("checkout", "restore", "stash", "reset", "mv"):
            out.extend(files)
        elif prog in _FIXERS:
            flag = _FIXERS[prog]
            if prog == "ruff" and args[:1] == ["format"]:
                flag = None
            if flag is None or flag in args:
                out.extend(files)
    return out


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
    out.extend(p for p in _inplace_written(cmd) if p not in out)
    out.extend(p for p in _inline_written(cmd + "\n" + str(args.get("content") or "")) if p not in out)
    fn = str(args.get("filename") or "")
    if fn and args.get("content") and _is_code(fn):
        out.append(fn)
    return out


# ── §4UU-3: what counts as RUNNING a file ─────────────────────────────
# 37 of 38 owner coding turns ended "done / tested / live"; about 10 of the 15
# that "ran something" after the last write ran only `node --check`, grep, cat
# or curl — each of which NAMED the file and so counted as its run. A command
# segment that only READS a file, or only checks its SYNTAX / lint, is not a
# run of it: only a segment that executes it (or its tests) is.

#: programs that only read / list / fetch — naming a file is not running it
_READ_PROGRAMS = frozenset({
    "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "ag", "ack", "sed", "awk", "gawk",
    "wc", "ls", "stat", "file", "diff", "cmp", "md5sum", "sha1sum", "sha256sum", "shasum", "tree", "find",
    "du", "echo", "printf", "curl", "wget", "nl", "od", "xxd", "hexdump", "strings", "sort", "uniq", "cut",
    "tr", "jq", "touch", "cp", "mv", "rm", "mkdir", "chmod", "ln", "git", "test", "[", "true", "cd", "export",
    "realpath", "readlink", "basename", "dirname", "column", "fold", "iconv", "base64", "tar", "zip", "unzip",
    "gzip", "gunzip", "open", "xdg-open", "code", "vim", "vi", "nano",
    # §4UU review: no-ops that "pass" without running anything
    "exit", ":", "false", "sleep", "pwd", "which", "type", "wait", "<",
})
#: arguments that only ask a program for its version / help — it runs nothing the request wrote
_INFO_ARGS = frozenset({"-v", "-V", "--version", "version", "-h", "--help", "help"})
#: interpreters that run a script fed on stdin (`cat app.py | python3`)
_STDIN_INTERP_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|node|nodejs|bash|sh|zsh|ruby|perl|php|deno)$")
#: linters / type checkers / formatters — a pass is a CHECK, not a run
_LINT_PROGRAMS = frozenset({
    "tsc", "eslint", "pylint", "flake8", "pyflakes", "mypy", "pyright", "ruff", "black", "isort", "shellcheck",
    "prettier", "htmlhint", "stylelint", "jshint", "jslint", "standard", "biome", "html-validate", "tidy",
})
#: `python -m <these>` is a check, not a run
_LINT_MODULES = frozenset({"py_compile", "compileall", "pyflakes", "flake8", "pylint", "mypy", "ruff", "black",
                           "isort", "pyright", "tabnanny", "pycodestyle"})
#: `<interpreter> <flag>` = syntax only
_SYNTAX_FLAGS = {
    "node": ("--check", "-c"), "nodejs": ("--check", "-c"), "bash": ("-n",), "sh": ("-n",), "zsh": ("-n",),
    "dash": ("-n",), "php": ("-l",), "ruby": ("-c",), "perl": ("-c",),
}
_SEG_SPLIT_RE = re.compile(r"\|\||&&|[;|&\n]")
_SEG_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEG_PREFIX = frozenset({"time", "nohup", "exec", "command", "env", "nice", "stdbuf", "timeout", "sudo",
                         "xvfb-run", "xargs", "watch"})


#: a command substitution / process substitution / backtick — its text is a command, quoted or not
_SUBST_RE = re.compile(r"\$\(((?:[^()]|\([^()]*\))*)\)|[<>]\(((?:[^()]|\([^()]*\))*)\)|`([^`]*)`")


def _command_lines(cmd: str) -> list:
    """The command's lines (backslash continuations joined). Heredoc bodies stay: a body fed to an interpreter
    is the program (`timeout 30 python3 - <<EOF`); a body written to a file is read as commands too — the
    pre-§4UU behaviour, a known false credit (see the reference page's residuals)."""
    lines = str(cmd or "").replace("\\\n", " ").split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        i += 1
        m = _INLINE_CODE_OPEN_RE.search(line)
        if m:
            q = m.group(1)
            # a `python3 -c "` / `node -e '` body that spans lines: join it until its closing quote
            while line[m.end():].count(q) % 2 == 0 and i < len(lines) and q not in line[m.end():]:
                line += "\n" + lines[i]
                i += 1
        out.append(line)
    return out


#: an interpreter's inline-code flag whose quoted body may span lines
_INLINE_CODE_OPEN_RE = re.compile(r"(?:^|[\s;&|(])(?:python[\d.]*|node|nodejs|ruby|perl|php|deno)\s+(?:-\w+\s+)*-[ce]\s+(['\"])")
#: Python / Node writing a file from inline code (`open('app.py', 'w')`, `Path('a.js').write_text`, `fs.writeFileSync`)
_INLINE_WRITE_RE = re.compile(
    r"""open\(\s*(?:r|f|rb)?['"]([^'"\n]+)['"]\s*,\s*(?:mode\s*=\s*)?['"][wax]\+?b?['"]"""
    r"""|Path\(\s*['"]([^'"\n]+)['"]\s*\)\.write_(?:text|bytes)\("""
    r"""|fs\.(?:writeFileSync|writeFile|appendFileSync)\(\s*['"`]([^'"`\n]+)['"`]""")


def _inline_written(text: str) -> list:
    """Code files inline Python / Node code writes (a heredoc body, `-c` code, an execute script)."""
    out = []
    for m in _INLINE_WRITE_RE.finditer(str(text or "")):
        p = next(g for g in m.groups() if g)
        if _is_code(p) and p not in out:
            out.append(p)
    return out


#: Playwright / puppeteer navigation inside a script: `page.goto("file:///…/index.html")`
_GOTO_RE = re.compile(r"""\.goto\(\s*f?['"`]([^'"`\s]+)['"`]""")


def _shell_segments(cmd: str, subst_bodies: Optional[list] = None) -> list:
    """The simple commands of a shell text, as word lists: one per line (a `\\`-continued line joins the next),
    each line split on `&&`, `||`, `;`, `|`, `&` — quote-aware when the line tokenises, by the plain operators
    when it does not (an unbalanced quote that spans lines). Every `$(…)`, `<(…)` and backtick — quoted or
    not — is a command of its own."""
    import shlex
    out = []
    pending = list(_command_lines(cmd))
    while pending:
        line = pending.pop(0)
        if not line.strip():
            continue
        if line.count("`") % 2 or line.count("$(") > line.count(")"):
            line = line.replace("$(", " ; ").replace("`", " ; ")     # unbalanced: still a command boundary
        subs = []
        def _take(m):
            body = next(g for g in m.groups() if g is not None)
            subs.append(body)
            if subst_bodies is not None:
                subst_bodies.append(body)
                return f"_SUBST{len(subst_bodies) - 1}_"
            return "_SUBST_"
        line = _SUBST_RE.sub(_take, line) if len(line) <= 20000 else line
        pending[:0] = subs
        # a redirection is not a separator (`2>&1`, `&>log`)
        line = _REDIR_AMP_RE.sub(" ", line)
        try:
            lex = shlex.shlex(line, posix=True, punctuation_chars=";&|")
            lex.whitespace_split = True
            lex.commenters = ""
            cur = []
            for tok in lex:
                if tok and set(tok) <= set(";&|"):
                    if cur:
                        out.append(cur)
                    cur = []
                else:
                    cur.append(tok)
            if cur:
                out.append(cur)
        except ValueError:
            out.extend(s.split() for s in _SEG_SPLIT_RE.split(line) if s.strip())
    return out


_REDIR_AMP_RE = re.compile(r"\d*>&\d*-?|&>>?")
#: each prefix's options that take a value (`sudo -u www`, `nice -n 5`, `timeout -s KILL 5`, `xargs -I {}`)
_PREFIX_VALUE_OPTS = {
    "sudo": {"-u", "-g", "-C", "-p", "-h", "-U", "-D", "-r", "-t"}, "nice": {"-n"}, "timeout": {"-s", "-k"},
    "xargs": {"-I", "-n", "-P", "-L", "-d", "-s", "-E", "-a"}, "env": {"-u", "-C", "-S"}, "watch": {"-n"},
    "stdbuf": {"-i", "-o", "-e"},
}

#: shell keywords / grouping that only lead into the real command (`if`, `then`, `! grep`, `{ …`, `( …`)
_SHELL_LEAD = frozenset({"if", "then", "elif", "else", "do", "while", "until", "!", "{", "(", "((", "time"})
#: shell keywords / tests that run nothing themselves
_SHELL_NOOP = frozenset({"fi", "done", "esac", "}", ")", "))", "[[", "]]", "for", "case", "in", "select",
                         "read", "local", "declare", "set", "unset", "shift", "return", "break", "continue"})


def _strip_shell_lead(words) -> list:
    """The words after shell keywords / grouping that only lead into the command (`if`, `do`, `!`, `{`, `(cat`)
    and a `case … in pat)` head."""
    words = list(words or [])
    if words and words[0] == "case" and "in" in words:
        words = words[words.index("in") + 1:]
    while words and (words[0] in _SHELL_LEAD or (words[0][:1] in "({" and len(words[0]) > 1)
                     or (words[0].endswith(")") and "(" not in words[0])):     # a `case` pattern: `start)`
        if words[0] in _SHELL_LEAD or words[0].endswith(")"):
            words = words[1:]
        else:
            words = [words[0][1:]] + words[1:]
    return words


def segment_kind(words: list) -> str:
    """"run" | "check" (syntax / lint / type check) | "read" for one simple command's words."""
    words = _strip_shell_lead(words)
    if words and words[0] in _SHELL_NOOP:
        return "read"
    i = 0
    while i < len(words) and (_SEG_ASSIGN_RE.match(words[i]) or words[i].rsplit("/", 1)[-1] in _SEG_PREFIX):
        prog = words[i].rsplit("/", 1)[-1]
        i += 1
        if prog == "command" and i < len(words) and words[i] in ("-v", "-V"):
            return "read"                                # `command -v node` looks a program up
        # the prefix's own options (`sudo -u www`, `env -i`, `nice -n 5`, `timeout -s KILL 5`, `xargs -n1`)
        while i < len(words) and (words[i].startswith("-") or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", words[i])):
            i += 2 if (words[i] in _PREFIX_VALUE_OPTS.get(prog, ()) and i + 1 < len(words)) else 1
    if i >= len(words):
        return "read"
    prog = words[i].rsplit("/", 1)[-1].lower()
    rest = words[i + 1:]
    if prog == "git" and rest[:2] == ["bisect", "run"]:
        return segment_kind(rest[2:])                    # `git bisect run python3 app.py` runs app.py
    if prog in ("bash", "sh", "zsh", "dash") and rest[:1] == ["-c"] and len(rest) > 1:
        kinds = [segment_kind(w) for w in _shell_segments(rest[1])]       # `bash -c 'node --check a.js'`
        return "run" if "run" in kinds else ("check" if "check" in kinds else "read")
    if prog in _READ_PROGRAMS:
        return "read"
    if rest and all(r in _INFO_ARGS for r in rest):
        return "read"
    if prog in _LINT_PROGRAMS:
        return "check"
    if prog in ("npx", "pnpm", "yarn") and rest and rest[0].rsplit("/", 1)[-1].lower() in _LINT_PROGRAMS:
        return "check"
    lead = []                                            # the interpreter's OWN flags: those before the script
    for r in rest:
        if not r.startswith("-"):
            break
        lead.append(r)
    if prog in _SYNTAX_FLAGS and any(r in _SYNTAX_FLAGS[prog] for r in lead):
        return "check"
    if re.match(r"python(?:\d+(?:\.\d+)*)?$", prog):
        if "-m" in rest:
            mod = rest[rest.index("-m") + 1] if rest.index("-m") + 1 < len(rest) else ""
            if mod.split(".")[0] in _LINT_MODULES:
                return "check"
        if rest[:1] == ["-c"] and len(rest) > 1 and re.search(r"\b(?:ast\.parse|py_compile)|(?<![\w.])compile\(", rest[1]):
            return "check"
    if re.match(r"(?:gcc|g\+\+|clang\+?\+?|cc)$", prog) and "-fsyntax-only" in rest:
        return "check"
    return "run"


def _bare_interpreter(words: list) -> bool:
    """An interpreter with no script argument (it runs what arrives on stdin)."""
    for i, w in enumerate(words):
        if _SEG_ASSIGN_RE.match(w) or w.rsplit("/", 1)[-1] in _SEG_PREFIX:
            continue
        if not _STDIN_INTERP_RE.match(w.rsplit("/", 1)[-1].lower()):
            return False
        return all(r.startswith("-") for r in words[i + 1:] if r != "-")
    return False


#: inline code that RUNS another file: exec(open(x)), runpy, require/import of a path, a subprocess interpreter
_EXEC_REF_RE = re.compile(
    r"""(?:exec\(\s*(?:open|Path)\(|run_path\(|run_module\(|require\(|import\(|importScripts\(|execfile\(|"""
    r"""source\s+)\s*f?['"`]?([\w.\-/]+)"""
    r"""|(?:['"](?:python[\d.]*|node|bash|sh|ruby|perl|php)['"]|sys\.executable)\s*,\s*['"]([\w.\-/]+)['"]"""
    r"""|\b(?:system|run|call|check_call|check_output|Popen|popen|getoutput|getstatusoutput|exec|execSync|"""
    r"""spawn|spawnSync|execFileSync)\(\s*['"`](?:python[\d.]*|node|bash|sh|ruby|perl|php|deno)\s+(?:-\S+\s+)*"""
    r"""([\w.\-/]+\.\w+)""")


def _exec_refs(code: str) -> set:
    out = set()
    for groups in _EXEC_REF_RE.findall(str(code or "")):
        n = next((g for g in groups if g), "")
        if n:
            out.add(_key(n if "." in os.path.basename(n) else n + ".js"))
            out.add(_key(n))
    return out


def _inline_code(words: list) -> Optional[str]:
    """The inline code of `python3 -c "…"` / `node -e '…'` (the words after the prefixes), else None."""
    ws = [w for w in _strip_shell_lead(list(words)) if not _SEG_ASSIGN_RE.match(w)]
    while ws and ws[0].rsplit("/", 1)[-1] in _SEG_PREFIX:
        ws = ws[1:]
        while ws and (ws[0].startswith("-") or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", ws[0])):
            ws = ws[1:]
    if not ws or not re.match(r"(?:python[\d.]*|node|nodejs|ruby|perl|php|deno)$", ws[0].rsplit("/", 1)[-1]):
        return None
    for j, w in enumerate(ws[1:], 1):
        if w in ("-c", "-e", "--eval") and j + 1 < len(ws):
            return ws[j + 1]
        if not w.startswith("-"):
            return None
    return None


def _names_in(blob: str, *, imports: bool, modules: bool) -> set:
    keys = set()
    for m in _CODE_NAME_RE.findall(blob):
        keys.add(_key(m))
    if imports:
        for a, b in _IMPORT_ANY_RE.findall(blob):
            mod = (a or b).split(".")[0]
            if mod:
                keys.add(f"{mod}.py".casefold())
    if modules:
        for a, b in _MODULE_RUN_RE.findall(blob):
            mod = (a or b)
            if mod and mod.lower() not in _NOT_A_MODULE:
                keys.add(f"{mod.split('.')[-1]}.py".casefold())
    return keys


def _exec_run_keys(tool: dict, args: dict) -> tuple:
    """(keys the run exercised, is_suite, keys only checked). Keys are basenames (casefold) of the code files the
    RUNNING segments of the command, the filename / args and the script text name, plus `x.py` for an
    `import x`. §4UU-3: a segment that only reads (`cat`, `grep`, `curl` …) or only checks syntax / lint
    (`node --check`, `python -m py_compile`, `bash -n`, `tsc` …) names files without running them — those go to
    the third set, never to the first."""
    from .agent import is_suite_command
    # a heredoc / redirect TARGET is a write, not a run: `cat > app.py <<EOF`
    # names app.py without running it (the same command may run it later:
    # `…EOF\npython app.py`)
    cmd = _SHELL_WRITE_RE.sub(" ", str(args.get("command") or ""))
    fn = str(args.get("filename") or "")
    content = str(args.get("content") or "")[:6000]
    extra = " ".join(str(a) for a in (args.get("args") or []) if a is not None)
    keys, checked = set(), set()
    prev = ""
    loop_lists = {}                                       # `for t in a.py b.py; do python3 $t; done`
    bodies: list = []
    for words in _shell_segments(cmd, bodies):
        if len(words) > 2 and words[0] == "for" and words[2] == "in":
            loop_lists[words[1]] = " ".join(words[3:])
        kind = segment_kind(words)
        seg = " ".join(words)
        if kind == "run":
            code = _inline_code(words)
            if code is None:
                keys |= _names_in(seg, imports=True, modules=True)
            else:                                        # `python3 -c "…"`: what the code imports or executes —
                keys |= _names_in(code, imports=True, modules=False) & (     # a file it only OPENS is data
                    _exec_refs(code) | {f"{m.split('.')[0]}.py".casefold()
                                        for a, b in _IMPORT_ANY_RE.findall(code) for m in [a or b] if m})
            for w in words[1:]:                          # `python3 $(find . -name app.py)` runs what it names
                m = re.fullmatch(r"_SUBST(\d+)_", w)
                if m and int(m.group(1)) < len(bodies):
                    keys |= _names_in(bodies[int(m.group(1))], imports=False, modules=False)
            for var, items in loop_lists.items():
                if f"${var}" in seg or f"${{{var}}}" in seg:
                    keys |= _names_in(items, imports=False, modules=False)
            if prev and _bare_interpreter(words):
                keys |= _names_in(prev, imports=False, modules=False)    # `cat app.py | python3`
        elif kind == "check":
            checked |= _names_in(seg, imports=False, modules=False)
        prev = seg
    # `execute(filename=x, content=…)` RUNS x; its script text and args are what that run touches
    for blob in (fn, extra, content):
        keys |= _names_in(blob, imports=blob is content, modules=False)
    # a file the inline code WRITES (`open('app.py','w')`) is not run by naming it — only a segment that
    # runs it as a script (`python3 app.py`) is
    wrote = {_key(p) for p in _inline_written(cmd + "\n" + content)}
    if wrote:
        # what the inline code RUNS after writing it (`subprocess.run(['python3','gen.py'])`, `import helper`)
        code_text = cmd + "\n" + content
        direct = _exec_refs(code_text) | {f"{(a or b).split('.')[0]}.py".casefold()
                                          for a, b in _IMPORT_ANY_RE.findall(code_text) if (a or b)}
        for words in _shell_segments(cmd):
            if segment_kind(words) == "run":
                ws = [w for w in _strip_shell_lead(words) if not _SEG_ASSIGN_RE.match(w)]
                direct |= {_key(w) for w in ws[1:3] if _is_code(w)}
        if fn:
            direct.add(_key(fn))
        keys -= wrote - direct
    # final review 11-18: a script's text is a suite run only when it is a SHELL script (a .py whose code says
    # `if pytest:` runs no tests)
    _sh = fn.lower().endswith(".sh") or (not fn and content.lstrip().startswith("#!") and "sh" in content[:40])
    return keys, bool(is_suite_command(cmd) or (_sh and is_suite_command(content))), checked - keys


def _browser_keys(tool: dict, args: dict) -> set:
    """Web pages a browser / vision call named (url, path, target)."""
    keys = set()
    vals = [args.get(k) for k in ("url", "path", "target", "file", "page", "image", "out_path")]
    acts = args.get("actions")
    if isinstance(acts, list):                         # `interact actions=[{action: goto, url: …}]`
        vals += [a.get("url") for a in acts if isinstance(a, dict)
                 and str(a.get("action") or "").lower() in ("goto", "navigate", "open")]
    for v in vals:
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


_SCRIPT_EXTS = (".js", ".mjs", ".cjs")
#: folders a served page's own front-end scripts live in (a localhost page loads these, not migrations / tools)
_FRONTEND_DIRS = frozenset({"js", "static", "public", "assets", "client", "frontend", "www", "web", "apps",
                            "components", "pages", "ui", "app"})


def _load_was_clean(tool: dict) -> bool:
    """A browser load that threw nothing and whose own scripts all loaded (the result text says so)."""
    text = str(tool.get("content") or "")
    if "UNCAUGHT JS EXCEPTIONS" in text:
        return False
    try:
        from .agent import _local_load_failures
        return not _local_load_failures(text)
    except Exception:  # noqa: BLE001
        return True


def _browser_urls(args: dict) -> list:
    vals = [args.get(k) for k in ("url", "path", "target", "file", "page")]
    acts = args.get("actions")
    if isinstance(acts, list):
        vals += [a.get("url") for a in acts if isinstance(a, dict)
                 and str(a.get("action") or "").lower() in ("goto", "navigate", "open")]
    return [v for v in vals if isinstance(v, str) and v]


def _rel(path: str) -> str:
    """A workspace-relative, slash-separated form of a page / script path or file:// URL path."""
    p = str(path or "").replace("\\", "/")
    p = re.sub(r"^(?:file://)", "", p)
    m = re.search(r"/(?:workspace|sandbox)/(.*)$", p)
    if m:
        p = m.group(1)
    return p.lstrip("./").lstrip("/").casefold()


def _credit_served_scripts(led: dict, urls: list, i: int) -> None:
    """A page that loaded CLEAN ran the scripts it loads: written .js files under the page's folder (a file://
    page, or a written page the url names), and — for a page served from this machine (localhost / 127.0.0.1)
    — the written scripts in a front-end folder (`js/`, `static/`, `public/` …) or next to a written page.
    Only a CLEAN load is credited (the caller checks)."""
    from urllib.parse import urlparse
    dirs, served = set(), False
    for u in urls:
        u = u.split("?", 1)[0].split("#", 1)[0]
        if re.match(r"(?i)^https?://", u):
            host = (urlparse(u).hostname or "").lower()
            if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                served = True
            continue
        path = urlparse(u).path if u.lower().startswith("file://") else u
        if not _is_web(path):
            continue
        d = _rel(path).rsplit("/", 1)[0] if "/" in _rel(path) else ""
        dirs.add(d)
        for rec in led.values():                          # the written page's own folder, as it was written
            r, u = _rel(rec["path"]), _rel(path)
            if _is_web(rec["path"]) and (r == u or r.endswith("/" + u) or u.endswith("/" + r)):
                dirs.add(r.rsplit("/", 1)[0] if "/" in r else "")
    for rec in led.values():
        if not str(rec["path"]).casefold().endswith(_SCRIPT_EXTS) or rec["write_idx"] > i:
            continue
        r = _rel(rec["path"])
        if served:
            parts = r.split("/")[:-1]
            page_dirs = {_rel(x["path"]).rsplit("/", 1)[0] if "/" in _rel(x["path"]) else ""
                         for x in led.values() if _is_web(x["path"])}
            if not (set(parts) & _FRONTEND_DIRS or (r.rsplit("/", 1)[0] if "/" in r else "") in page_dirs):
                continue
            rec["run_idx"], rec["run_ok"] = i, True
            continue
        if any(d == "" or r.startswith(d + "/") or d.endswith("/" + r.rsplit("/", 1)[0]) for d in dirs):
            rec["run_idx"], rec["run_ok"] = i, True


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

def _new_rec(path, i, via) -> dict:
    return {"path": path, "write_idx": i, "run_idx": None, "run_ok": None, "via": via, "checked": False}


def _apply_page_checks(led: dict, row, i: int) -> None:
    """§4UU-1: the system's headless page check after a web write (its note on the write's own row) RUNS the
    page and every script it loaded — a clean load as a pass, a page that throws as a failure."""
    try:
        from .page_check import checks_in
    except Exception:  # noqa: BLE001
        return
    for ok, keys in checks_in(str(row.get("content") or "")):
        for k, rec in led.items():
            if k in keys:
                rec["run_idx"], rec["run_ok"] = i, ok


_CHECKLIST_ROW_RE = re.compile(r"^(✓|✗) #(\d+) .*? — check: (.+?)$", re.M)


def _checklist_runs(tool: dict, args: dict) -> list:
    """§4UU-2: a `checklist check` RAN each chosen item's check — [(ok, keys, is_suite), …] by the same rule as
    an execute (a shell check) or the page probe (`page:<file>`). `view` / `set` re-print old states: no run."""
    if str(args.get("action") or args.get("operation") or "").strip().lower() != "check":
        return []
    want = set()
    ids = args.get("ids")
    for x in (ids if isinstance(ids, list) else re.findall(r"\d+", str(ids or ""))):
        try:
            want.add(int(x))
        except (TypeError, ValueError):
            continue
    out = []
    for mark, num, check in _CHECKLIST_ROW_RE.findall(str(tool.get("content") or "")):
        if want and int(num) not in want:
            continue
        check = check.strip()
        if check.endswith("(not run in this call)"):
            continue                                     # printed with its old state: nothing ran it now
        if check.lower().startswith("file:"):
            continue
        if check.lower().startswith("page:"):
            out.append((mark == "✓", {_key(check[5:].strip())}, False))
            continue
        keys, suite, _checked = _exec_run_keys({}, {"command": check})
        out.append((mark == "✓", keys, suite))
    return out


def code_evidence(tools_run: Optional[list]) -> dict:
    """key → {path, write_idx, run_idx, run_ok, via, checked}. `run_idx` is the last
    run AFTER the last write (None when nothing ran the final version);
    `run_ok` is that run's verdict (None = detached / unknown); `checked` —
    a syntax / lint check named the final version (it is still not a run).
    A file_system row that carries a §4UU-1 page check counts as that run,
    right after its own write."""
    led: dict = {}
    for i, t in enumerate(tools_run or []):
        if not isinstance(t, dict) or t.get("_synthetic"):
            continue
        n = _name(t)
        args = _args_of(t)
        if n in ("file_system", "filesystem", "file"):
            if str(args.get("operation") or "").lower() == "delete" and \
                    str(t.get("content") or "").lstrip().upper().startswith("SUCCESS"):
                led.pop(_key(args.get("path") or ""), None)      # a deleted file is not "never run"
                continue
            written = _fs_written(t)
            for p in written:
                led[_key(p)] = _new_rec(p, i, "file_system")
            if written:
                _apply_page_checks(led, t, i)
            continue
        if n == "execute":
            ok = _exec_ok(t)
            if ok:
                for words in _shell_segments(str(args.get("command") or "")):
                    ws = _strip_shell_lead(list(words))
                    if ws and ws[0] == "rm":                     # `rm fix2.py` — gone, not "never run"
                        for w in ws[1:]:
                            if not w.startswith("-"):
                                led.pop(_key(w), None)
            for p in _exec_written(t, args):
                led[_key(p)] = _new_rec(p, i, "execute")
            if ok is None:
                continue
            keys, suite, checked = _exec_run_keys(t, args)
            # every record here was written at an index <= i (a later write
            # REPLACES the record, which is how order is kept)
            for k, rec in led.items():
                if k in keys or (suite and not _is_web(rec["path"])):
                    rec["run_idx"], rec["run_ok"] = i, ok
                elif k in checked and rec["run_idx"] is None:
                    rec["checked"] = True
            # a Playwright script that loads a page (`page.goto("file:///…/index.html")`) loads its scripts too
            gotos = _GOTO_RE.findall(str(args.get("command") or "") + "\n" + str(args.get("content") or ""))
            if gotos and ok:
                _credit_served_scripts(led, gotos, i)
            # `execute(filename=x, content=…)` writes AND runs x in one call
            fn = str(args.get("filename") or "")
            if fn and args.get("content") and _key(fn) in led and led[_key(fn)]["write_idx"] == i:
                led[_key(fn)]["run_idx"], led[_key(fn)]["run_ok"] = i, ok
            continue
        if n == "checklist":
            for ok, keys, suite in _checklist_runs(t, args):
                for k, rec in led.items():
                    if k in keys or (suite and not _is_web(rec["path"])):
                        rec["run_idx"], rec["run_ok"] = i, ok
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
            if ok and n == "browser" and _load_was_clean(t):
                _credit_served_scripts(led, _browser_urls(args), i)
    return led


def unbacked_code_writes(tools_run: Optional[list]) -> tuple:
    """(never_run, failed_last): code files written this request whose final
    version nothing ran / loaded, and those whose last run after the final
    write FAILED. Paths in first-write order."""
    never, failed, _checked = _unbacked(tools_run)
    return never, failed


def _unbacked(tools_run) -> tuple:
    led = code_evidence(tools_run)
    never, failed, checked = [], [], []
    for rec in sorted(led.values(), key=lambda r: r["write_idx"]):
        if rec["run_idx"] is None:
            never.append(rec["path"])
            if rec.get("checked"):
                checked.append(rec["path"])
        elif rec["run_ok"] is False:
            failed.append(rec["path"])
    return never, failed, checked


def _names(paths: list, limit: int = 3) -> str:
    shown = [f"`{os.path.basename(str(p))}`" for p in paths[:limit]]
    more = len(paths) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


UNBACKED_CLAIM_REASON = "unbacked code claim: the reply says it works/was tested but the file was never run (or its last run failed)"


def unbacked_claim(reply: str, tools_run: Optional[list]) -> tuple:
    """(claim, never_run, failed_last) when `reply` affirms the code works /
    was tested AND a code file written this request has no backing run —
    ("", [], []) otherwise."""
    claim, never, failed, _checked = _unbacked_claim(reply, tools_run)
    return claim, never, failed


def _unbacked_claim(reply, tools_run) -> tuple:
    claim = claims_code_works(reply)
    if not claim:
        return "", [], [], []
    never, failed, checked = _unbacked(tools_run)
    if not never and not failed:
        return "", [], [], []
    return claim, never, failed, checked


def _never_run_phrase(never: list, checked: list) -> str:
    """"`a.js` was written this turn and never run" — and, when a syntax / lint check was all that named
    it (§4UU-3), says so: a check is not a run."""
    if checked and set(checked) == set(never):
        return (f"{_names(never)} was written this turn and only syntax/lint-checked, never run "
                f"(a syntax check or a read is not a run)")
    return f"{_names(never)} was written this turn and never run"


def unbacked_claim_note(reply: str, tools_run: Optional[list],
                        forbids_running: bool = False) -> tuple:
    """(note, failed) for a reply whose works/tested claim has no backing run
    — ("", False) when the claim is backed or absent. ONE wording for both
    delivery paths. A request that said "don't run it" is said, not failed
    (§4KW) — but the claim is still named as unbacked."""
    claim, never, failed, checked = _unbacked_claim(reply, tools_run)
    if not claim:
        return "", False
    if failed:
        what = (f"the last run of {_names(failed)} after it was written FAILED"
                + (f", and {_names(never)} was never run" if never else ""))
        verdict = "Treat it as BROKEN until a run passes."
    else:
        what = _never_run_phrase(never, checked)
        verdict = "Treat it as UNTESTED — run it before relying on it."
    if forbids_running:
        return (f"\n\n---\n*Not run, as you asked — but note the reply's \"{claim}\" is not "
                f"backed by a run this turn: {what}.*"), False
    return (f"\n\n---\n**⚠ Unverified claim:** this reply says \"{claim}\", but {what}. "
            f"{verdict}"), True


def unbacked_claim_directive(reply: str, tools_run: Optional[list]) -> str:
    """The in-loop repair directive for an unbacked claim, or ""."""
    claim, never, failed, checked = _unbacked_claim(reply, tools_run)
    if not claim:
        return ""
    if failed:
        what = f"the last run of {_names(failed)} after you wrote it FAILED"
    elif checked and set(checked) == set(never):
        what = (f"{_names(never)} was written this turn and only SYNTAX/LINT-CHECKED — NEVER RUN "
                f"(a syntax check, grep, cat or curl is not a run)")
    else:
        what = f"{_names(never)} was written this turn and NEVER RUN"
    return (f"SYSTEM ALERT — your answer says \"{claim}\", but {what}. "
            f"Actually RUN it now: execute the file or its tests (a web app: its "
            f"`tests/check_page.py` click check, or browser interact that clicks a control) and read the result. If it fails, FIX it or say so plainly — do not "
            f"claim it works. THEN give your final answer, naming what you ran.")
