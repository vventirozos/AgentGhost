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
#: §4UY C19: a name starts where a path token starts (the lookbehind) — unanchored, findall retried the scan at
#: every character of a long `./././…` run (quadratic: 0.56 s at 2 KB)
_CODE_NAME_RE = re.compile(
    r"(?<![\w.\-/])[\w.\-/]+\.(?:py|sh|bash|js|mjs|cjs|ts|go|rs|rb|php|pl|c|cpp|java|html?)\b", re.I)
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

#: §4UY C4 — ONE principle instead of a phrase list: a clause that ASSERTS verification or a working state is a
#: claim. The stems: tested / verified / confirmed / validated (with live- / smoke- / end-to-end … prefixes), works /
#: working / runs fine, a test / check / suite / "it" that passes / is green / succeeds, a run that went
#: "successfully" / "cleanly" / "without errors". The phrase list matched 26 of 108 coding replies: "All 12 tests
#: pass", "verified", "confirmed it's working", "live-tested" were missed (probe-75a957e4: the last execute was a
#: 600 s kill and the reply said "verified working"). Broader is safe here: the arm speaks only when a code file
#: written this request has no backing run.
#: r2 (review): the claim is about RUNNING / TESTING / WORKING code — "tested" is that by itself; verified /
#: confirmed / validated only when what was verified is code or its run ("verified it", "confirmed the app
#: works", "verified working", "validated against the sample", a bare "Verified." ending the sentence) — never
#: "I verified the config file was saved", "Confirmed: the file is at app.py", "validated the JSON schema"
_CODE_OBJECT = (r"(?:that\s+)?(?:it|this|everything|all\b|the\s+(?:[\w-]+\s+){0,2}?(?:app|application|script|code"
                r"|server|page|site|fix|change|patch|build|tests?|suite|function|endpoint|api|cli|tool|game|program"
                r"|module|feature|component|bot|parser|service|scraper|ui|form|binary|backend|frontend|output)s?\b"
                r"|working|works|end[- ]to[- ]end|in\s+the\s+browser|against\b|by\s+running)")
_CLAIM_RE = re.compile(
    r"\b(?:(?:live|smoke|unit|manually|fully|successfully|thoroughly|end[- ]to[- ]end|e2e)[- ]?)?tested\b"
    r"|\b(?:(?:live|smoke|manually|fully|successfully|thoroughly|end[- ]to[- ]end|e2e)[- ]?)?"
    r"(?:verified|confirmed)\b(?=\s*(?:[.!]|$)|\s+" + _CODE_OBJECT + r")"
    # "working" is a state only as a predicate ("it's working", "is now working", "working fine") — a bare
    # "working" / "I'm working on it" is progress narration (§4UY r3: a pre-tool "working" note became a claim)
    r"|\b(?:works|runs\s+(?:fine|correctly|cleanly|without\s+(?:any\s+)?errors?)"
    r"|(?:(?:is|are|was|were|been|be|it|everything|all|now)\s+|[’'](?:s|re)\s+)"
    r"(?:(?:now|all|fully|properly|correctly|still)\s+)?working"
    r"|working\s+(?:fine|correctly|properly|perfectly|as\s+(?:expected|intended)|again|now))\b"
    r"(?!\s+(?:by|like|as\s+follows|because|via|through|with|on|in|at|when|once|if|only|after|before|until"
    r"|directory|dir|tree|copy|set|memory|hours|day)\b)"
    r"|\b(?:tests?|checks?|suites?|specs?|assertions?|it|everything|all|build)\b[^.;:!?\n]{0,30}?"
    r"\b(?:pass|passes|passed|passing|green|succeeds?|succeeded)\b"
    r"|\b\d+\s+(?:tests?\s+)?passed\b"
    # r3 (replay 598156cf / 945b9726, the pre-§4UY detector's forms): "everything is verified", "it is fully
    # functional", "the app has been tested"
    r"|\b(?:it|this|everything|all|the\s+(?:[\w-]+\s+){0,2}?(?:app|application|script|code|server|page|site|fix"
    r"|change|patch|build|tests?|suite|function|endpoint|api|cli|tool|game|program|module|feature|component|bot"
    r"|parser|service|scraper|ui|form|binary|backend|frontend|dashboard)s?)\s+"
    r"(?:is|are|was|were|has\s+been|have\s+been)\s+(?:now\s+|all\s+)?"
    r"(?:(?:fully|successfully|thoroughly|end[- ]to[- ]end)\s+)?"
    r"(?:tested|verified|confirmed|validated|functional|working|passing)\b(?!\s+(?:directory|dir|tree|copy)\b)"
    r"|\b(?:fully|successfully|thoroughly)\s+(?:functional|working)\b(?!\s+(?:directory|dir|tree|copy)\b)"
    r"|\b(?:ran|run|executed)\b[^.;:!?\n]{0,25}?\b(?:successfully|cleanly|fine|without\s+(?:any\s+)?errors?"
    r"|with\s+no\s+errors?)\b"
    r"|\bδουλε[υύ]ει\b|\bτο\s+(?:δοκ[ιί]μασα|τ[εέ]σταρα|[εέ]τρεξα)\b|\bπερν[αά](?:ει|νε)\s+(?:[οό]λα\s+)?τα\s+τεστ\b",
    re.I | re.M)
#: r2 (review): "I've tested nothing yet" — a claim whose object is nothing / none is a denial
_NOTHING_AFTER_RE = re.compile(r"^\s*(?:nothing|none|no\s+\w+)\b", re.I)
#: a claim governed by a negation / a conditional / a future is not a claim
_NEGATION_RE = re.compile(
    r"\b(?:not|n[’']t|never|cannot|can[’']t|unable|without|untested|unverified|no|none|nothing|nobody|"
    r"should|would|could|might|may|will|if|once|until|unless|after|when|before|assuming|"
    r"haven[’']t|hasn[’']t|didn[’']t|wasn[’']t|isn[’']t|aren[’']t|couldn[’']t|how|"
    r"δεν|μην|χωρις|χωρίς|αν|θα|πως|πώς)\b|\bun(?=tested|verified)", re.I)
#: a clause that reports a failure is honest reporting ("I ran the tests and 2 of 5 failed"), not a claim —
#: unless the failure word is itself negated ("no errors", "0 failed")
_FAILURE_WORD_RE = re.compile(
    r"(?<!\bno\s)(?<!\b0\s)(?<!zero\s)(?<!without\s)(?<!any\s)"
    r"\b(?:fail(?:s|ed|ing|ure|ures)?|errors?|broke|broken|crash(?:es|ed)?|exceptions?|traceback)\b", re.I)
_CLAUSE_END_RE = re.compile(r"[.;:!?\n]")


def claims_code_works(text: str) -> str:
    """The first affirmative works/tested/verified claim in `text`, or ""."""
    s = str(text or "")
    for m in _CLAIM_RE.finditer(s):
        before = s[max(0, m.start() - 45):m.start()]
        # a negation / conditional in the same clause governs the claim
        _clause = _CLAUSE_END_RE.split(before)[-1]
        if _NEGATION_RE.search(_clause) or _NEGATION_RE.search(m.group(0)):
            continue
        _end = _CLAUSE_END_RE.search(s, m.end())
        _after = s[m.end():_end.start() if _end else len(s)]
        if _NOTHING_AFTER_RE.search(_after):
            continue
        if _FAILURE_WORD_RE.search(_clause + " " + m.group(0) + " " + _after):
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


def _is_code(path) -> bool:
    return str(path or "").casefold().endswith(CODE_EXTS)


def _is_web(path) -> bool:
    return str(path or "").casefold().endswith(WEB_EXTS)


#: a command the pre-execution validator / a wall refused: nothing ran
_NOT_RUN_RE = re.compile(r"^\s*SYSTEM BLOCK: shell command rejected by pre-execution validator\b|"
                         r"\bThe command was not run\b", re.M)


#: the operating system's own folders: a file written there (`/etc/profile.d/x.sh`) configures the machine
_SYSTEM_FILE_RE = re.compile(r"^/(?:etc|usr|proc|sys|boot|dev|lib|lib32|lib64|sbin|bin|var/lib|root/\.)/")


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
    if st == "rejected" or _NOT_RUN_RE.search(text):
        return None                      # r3 (live 2b1e6eb4): a command refused before it ran is no run at all
    if st in ("failed", "partial"):
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
#: §4UY C3: a help / version flag anywhere on the run — the program prints its usage and exits
_HELP_FLAGS = frozenset({"-h", "--help", "--version", "-V", "--usage"})
#: compilers: a compile builds the program, it does not run it (running the built binary is the run)
_COMPILERS_RE = re.compile(r"^(?:gcc|g\+\+|clang\+?\+?|cc|c\+\+|rustc|javac|kotlinc|scalac|swiftc|ghc|nim|zig|go"
                           r"|cargo|make|cmake|mvn|gradle|dotnet|ninja|tsc|esbuild|webpack|vite|rollup|parcel)$")
#: their subcommands / targets that only build (or only check)
_BUILD_SUBS = {"cargo": {"build", "check", "clippy", "fmt", "doc"}, "go": {"build", "vet", "fmt", "install", "mod"},
               "make": {"", "all", "build", "install", "clean", "compile"}, "mvn": {"compile", "package", "install",
               "verify", "clean"}, "gradle": {"build", "assemble", "compilejava", "clean"},
               "dotnet": {"build", "restore", "publish"}, "zig": {"build-exe", "build"},
               "vite": {"build"}, "esbuild": None, "webpack": None, "rollup": None, "parcel": {"build"},
               "cmake": None, "ninja": None, "tsc": None}


#: r3 (review m2): `python -m <these>` installs, makes an env or serves files — it runs none of the request's code
_PY_TOOL_MODULES = frozenset({"pip", "venv", "virtualenv", "ensurepip", "http.server", "http", "json.tool",
                              "site", "pydoc", "zipfile", "tarfile", "webbrowser", "pipx", "uv"})
#: `make <these>` / `cargo <these>` clean up; `npx <these>` free a port or serve a folder
_CLEAN_SUBS = frozenset({"clean", "distclean", "mrproper", "clobber"})
_NPX_TOOLS = frozenset({"kill-port", "http-server", "serve", "live-server", "fkill", "fkill-cli", "rimraf"})


def _not_a_runner(prog: str, rest: list) -> bool:
    """A program call that runs none of the request's code, read by its module / subcommand / tool."""
    sub = next((r for r in rest if not r.startswith("-") and "=" not in r), "").lower()
    if re.match(r"python(?:\d+(?:\.\d+)*)?$", prog) and "-m" in rest:
        j = rest.index("-m") + 1
        mod = rest[j].lower() if j < len(rest) else ""
        return mod in _PY_TOOL_MODULES or mod.split(".")[0] in ("pip", "venv", "ensurepip")
    if prog in ("make", "cargo", "go", "mvn", "gradle", "dotnet", "ninja") and sub in _CLEAN_SUBS:
        return True
    if prog in ("npx", "pnpx", "bunx") and sub.split("@")[0] in _NPX_TOOLS:
        return True
    return False


def _compile_only(prog: str, rest: list) -> bool:
    """A compile / build that runs nothing the request wrote (`gcc -o app app.c`, `cargo build`, `go build`,
    bare `make`, `javac`)."""
    if not _COMPILERS_RE.match(prog):
        return False
    sub = next((r for r in rest if not r.startswith("-") and "=" not in r), "")
    if prog in _BUILD_SUBS:
        subs = _BUILD_SUBS[prog]
        return subs is None or sub.lower() in subs
    return True                                          # gcc / javac / rustc …: a compiler always only compiles


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


def _shell_segments(cmd: str, subst_bodies: Optional[list] = None, seps: Optional[list] = None) -> list:
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
            cur, got, got_seps = [], [], []
            for tok in lex:
                if tok and set(tok) <= set(";&|"):
                    if cur:
                        got.append(cur)
                        got_seps.append(tok)
                    cur = []
                else:
                    cur.append(tok)
            if cur:
                got.append(cur)
                got_seps.append(";")
            out.extend(got)
            if seps is not None:
                seps.extend(got_seps)          # §4UY C3: the operator after each segment (`||` masks its status)
        except ValueError:
            parts = re.split(r"(\|\||&&|[;|&\n])", line)
            for k in range(0, len(parts), 2):
                if parts[k].strip():
                    out.append(parts[k].split())
                    if seps is not None:
                        seps.append(parts[k + 1] if k + 1 < len(parts) else ";")
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
    if any(r in _HELP_FLAGS for r in rest):
        return "read"                                    # §4UY C3: `python3 app.py --help` exercises nothing
    if _not_a_runner(prog, rest):
        return "read"                                    # r3: `python -m pip`, `make clean`, `npx kill-port`
    if _compile_only(prog, rest):
        return "check"                                   # §4UY C3: a build compiles, it does not run
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
    # r3 (review M5): importlib loading a file by path
    r"""(?:spec_from_file_location|SourceFileLoader|SourcelessFileLoader)\(\s*[^,()]+,\s*|"""
    r"""source\s+)\s*f?['"`]?([\w.\-/]+)"""
    # r5 (review): an [interpreter, path] list counts only as a CALL's argument (`run(["python3", "x.py"])`) —
    # `FILES = ['python3', 'x.py']` is data
    r"""|\(\s*\[\s*(?:['"](?:python[\d.]*|node|bash|sh|ruby|perl|php)['"]|sys\.executable)\s*,\s*['"]([\w.\-/]+)['"]"""
    r"""|\b(?:system|run|call|check_call|check_output|Popen|popen|getoutput|getstatusoutput|exec|execSync|"""
    r"""spawn|spawnSync|execFileSync)\(\s*['"`](?:python[\d.]*|node|bash|sh|ruby|perl|php|deno)\s+(?:-\S+\s+)*"""
    r"""([\w.\-/]+\.\w+)""")


#: r5 (review): string literals and comments of inline code (Python / JS / shell) — an `import app` inside
#: `print('import app')` or a `// require('./util')` comment runs nothing
_STR_OR_COMMENT_RE = re.compile(
    r"'''[\s\S]*?'''"  r'|"""[\s\S]*?"""'
    r"|'(?:\\.|[^'\\\n])*'"  r'|"(?:\\.|[^"\\\n])*"'
    r"|`(?:\\.|[^`\\])*`|#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/")


def _code_spans(code: str) -> list:
    """[(start, end)] of the string literals and comments of `code`, left to right."""
    return [m.span() for m in _STR_OR_COMMENT_RE.finditer(code)]


def _in_spans(spans: list, pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def _live_matches(rx, code: str):
    """`rx` matches of `code` that START in code — not inside a string literal or a comment (the argument string
    of `require('./a')` starts after the call, which is code)."""
    code = str(code or "")
    spans = _code_spans(code)
    return [m for m in rx.finditer(code) if not _in_spans(spans, m.start())]


def _exec_refs(code: str) -> set:
    """The file NAMES (as written) inline code runs: exec(open(x)), runpy, require('./a'), a subprocess interpreter."""
    out = set()
    for m in _live_matches(_EXEC_REF_RE, code):
        groups = m.groups()
        n = next((g for g in groups if g), "")
        if n:
            out.add(n)
            if "." not in os.path.basename(n):
                out.add(n + ".js")                       # `require('./util')`
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


# ── §4UY C2/C3: what a run CREDITS, keyed by PATH ─────────────────────
# The ledger keyed every file by its basename, so `node backend/index.js` backed `frontend/index.js`, a suite run in
# another folder backed every file, and a script whose TEXT merely named a file ran it. A run now credits:
#   * the files it names, resolved against the command's working directory (`cd DIR &&` is followed; an
#     unknowable `cd $X` falls back to a suffix match);
#   * a DIRECTORY when the runner exercises the code under it — a test suite, `cargo run/test`, `go run/test .`,
#     `make run/test`, `node .`, `npm start/test/run X`, `python -m pkg`, a built binary `./app`;
#   * what a shell script the request wrote RUNS (its text, by the same rule), and what inline code imports or
#     executes — never a name that merely appears in it.
# A run whose status is MASKED by `||` (`pytest || true`) proves nothing and credits nothing; a compile, a
# `--help` / `--version` and a syntax check are checks, not runs (`segment_kind`).

_PROJ_PREFIX_RE = re.compile(r"^projects/[0-9a-f]{12}(?:/|$)", re.I)


def _norm(path, cwd: Optional[str] = "") -> str:
    """A path as the ledger keys it: workspace-relative, the active project's `projects/<id>/` prefix dropped,
    `./` and `..` resolved, case-folded. `cwd` (already normalised) anchors a relative path; an absolute path
    outside the workspace stays absolute."""
    import posixpath
    p = str(path or "").strip().strip("'\"").replace("\\", "/")
    p = re.sub(r"(?i)^file://", "", p).split("?", 1)[0].split("#", 1)[0]
    if p.startswith("~/"):
        p = p[2:]
    m = re.match(r"^/(?:workspace|sandbox)(?:/|$)(.*)$", p)
    if m:
        p = m.group(1)
    elif p.startswith("/"):
        return posixpath.normpath(p).casefold()
    elif cwd:
        p = posixpath.join(cwd, p)
    p = posixpath.normpath(p) if p else ""
    p = "" if p == "." else p
    p = _PROJ_PREFIX_RE.sub("", p)
    return p.strip("/").casefold()


def _fs_norm(path) -> str:
    """r3 (review M4): a `file_system` path keyed the way file_system resolves it — a leading `/` (other than
    /workspace, or the host's own sandbox path) is relative to the workspace: `/webos/index.html` IS
    webos/index.html, `/etc/profile.d/x.sh` is etc/profile.d/x.sh inside it."""
    p = str(path or "").strip().replace("\\", "/")
    p = re.sub(r"^/(?:[^/]+/)*?sandbox/", "/workspace/", p) if not p.startswith(("/workspace", "/sandbox")) else p
    if p.startswith("/") and not re.match(r"^/(?:workspace|sandbox)(?:/|$)", p):
        p = p.lstrip("/")
    return _norm(p, "")


def _new_credits() -> dict:
    return {"exact": set(), "suffix": set(), "dirs": set(), "suite": False, "built": {}}


def _merge(into: dict, other: dict) -> None:
    for k in ("exact", "suffix", "dirs"):
        into[k] |= other[k]
    for d, ix in other["built"].items():
        into["built"][d] = max(into["built"].get(d, -1), ix)
    into["suite"] = into["suite"] or other["suite"]


def _add_path(cr: dict, name: str, cwd: Optional[str]) -> None:
    """Credit `name` (as written in the command) run from `cwd` (None = unknown)."""
    if not name or "$" in name or "*" in name:
        return
    if cwd is None and not str(name).startswith(("/", "~/")):
        k = _norm(name, "")
        if k and not k.startswith(".."):
            cr["suffix"].add(k)
        return
    k = _norm(name, cwd)
    if k:
        cr["exact"].add(k)


#: r5 (review): a suite / project runner runs code of ITS language only
_PY_EXTS = (".py",)
_JS_EXTS = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".mts", ".cts")
_RUNNER_EXTS = {"cargo": (".rs",), "go": (".go",), "npm": _JS_EXTS, "yarn": _JS_EXTS, "pnpm": _JS_EXTS,
                "bun": _JS_EXTS, "dotnet": (".cs", ".fs"), "mvn": (".java", ".kt", ".scala"),
                "gradle": (".java", ".kt", ".scala")}               # make / cmake: nothing by itself
_SUITE_LANGS = (
    (re.compile(r"(?:^|\s)(?:py\.?test|pytest|unittest|nose2?|tox|nox|ward)(?:\s|$)"), _PY_EXTS),
    (re.compile(r"(?:^|\s)(?:jest|vitest|mocha|ava|tap|karma|jasmine|playwright|cypress|npm|npx|yarn|pnpm|bun|node"
                r"|deno)(?:\s|$)"), _JS_EXTS),
    (re.compile(r"(?:^|\s)go\s+test(?:\s|$)"), (".go",)),
    (re.compile(r"(?:^|\s)cargo\s+(?:test|nextest|bench)(?:\s|$)"), (".rs",)),
    (re.compile(r"(?:^|\s)(?:rspec|rake|minitest)(?:\s|$)"), (".rb",)),
    (re.compile(r"(?:^|\s)(?:phpunit|pest)(?:\s|$)"), (".php",)),
    (re.compile(r"(?:^|\s)(?:mvn|gradle|gradlew)(?:\s|$)"), (".java", ".kt", ".scala")),
    (re.compile(r"(?:^|\s)dotnet\s+test(?:\s|$)"), (".cs", ".fs")),
)


def _suite_exts(seg: str) -> tuple:
    """The source extensions a suite command runs — () when its language is unknown (`make test`)."""
    low = " " + str(seg or "").lower() + " "
    for rx, exts in _SUITE_LANGS:
        if rx.search(low):
            return exts
    return ()


def _add_dir(cr: dict, d: Optional[str], exts: tuple = ()) -> None:
    if d is None:
        return                                           # an unknowable folder: credits nothing
    cr["dirs"].add((d, tuple(exts)))                     # (no language: `endswith(())` credits nothing)


#: r4 (review): the sources a compiler turns into a binary — a built `./app` ran THESE, never a .py / .js / .sh
_COMPILED_EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".rs", ".go", ".java", ".kt", ".kts",
                  ".swift", ".m", ".mm", ".cs", ".fs", ".zig", ".nim", ".d", ".f90", ".f", ".hs", ".ml", ".scala",
                  ".s", ".asm")


def _built_source(key: str, src: str) -> bool:
    """`key` is a source a build named: the file itself, or a .go file directly in a named Go package folder."""
    if src.startswith("\x00go:"):
        folder = key.rsplit("/", 1)[0] if "/" in key else ""
        return key.endswith(".go") and folder == src[4:]
    return key == src


def _credited(key: str, cr: dict, *, web: bool, write_idx: int = -1) -> bool:
    if not web and any(write_idx <= ix and _built_source(key, src) for src, ix in cr["built"].items()):
        return True
    if key in cr["exact"]:
        return True
    # a suffix credit is a path whose root is unknown (a served URL, a run after `cd $X`): either may be deeper
    if any(key == s or key.endswith("/" + s) or s.endswith("/" + key) for s in cr["suffix"]):
        return True
    if not web and any((d == "" or key == d or key.startswith(d + "/")) and key.endswith(ex)
                       for d, ex in cr["dirs"]):
        return True
    return False


def _module_paths(mod: str) -> list:
    """`a.b` → the files `python -m a.b` / `import a.b` loads: a/b.py, a/b/__main__.py, a/b/__init__.py."""
    p = str(mod or "").strip(".").replace(".", "/")
    return [f"{p}.py", f"{p}/__main__.py", f"{p}/__init__.py"] if p else []


def _module_bases(cwd: Optional[str], pypath=()) -> list:
    """r2 (review): where `import x` / `-m x` finds x — the folder, every PYTHONPATH entry, and its `src/`
    (`PYTHONPATH=src python -m mypkg.cli` runs src/mypkg/cli.py)."""
    if cwd is None:
        return [None]
    bases = [cwd]
    for pp in pypath:
        for part in str(pp or "").split(":"):
            if part and "$" not in part:
                bases.append(_norm(part, cwd))
    bases.append(_norm("src", cwd))
    return list(dict.fromkeys(bases))


def _add_module(cr: dict, mod: str, cwd: Optional[str], pypath=(), *, with_top: bool = False) -> None:
    for base in _module_bases(cwd, pypath):
        for p in _module_paths(mod) + (_module_paths(mod.split(".")[0]) if with_top else []):
            _add_path(cr, p, base)


#: a program that starts another process (a test harness running the script it tests)
def _credit_spawned(cr: dict, name: str, cwd: Optional[str], scripts: dict) -> None:
    """r4 (review: lean conservative): a written program run here credits ONLY what one of its own process calls
    runs as interpreter + path in that same call (`subprocess.run(["python3", "x.py"])`, `os.system("node x.js")`)
    — resolved against the run's folder. A script name it merely quotes (a tar list, a deploy list, a path built
    with os.path.join) is not credited: when unsure, no credit."""
    if cwd is None:
        return
    k = _norm(name, cwd)
    text = scripts.get(k)
    if not isinstance(text, str) or k.startswith("\x00"):
        return
    for m in _live_matches(_EXEC_REF_RE, text):
        n = m.group(2) or m.group(3)                     # the interpreter-plus-path alternatives only
        if n and _norm(n, cwd) != k:
            _add_path(cr, n, cwd)


def _credit_code_text(cr: dict, code: str, base: Optional[str], pypath=()) -> None:
    """Inline code (a `-c` body, a Python heredoc, an `execute(content=…)` script): what it IMPORTS or EXECUTES,
    resolved against `base` — a file it only names or opens is data."""
    for m in _live_matches(_IMPORT_ANY_RE, code):
        mod = m.group(1) or m.group(2)
        if mod and not mod.startswith("."):
            _add_module(cr, mod, base, pypath, with_top=True)
    for n in _exec_refs(code):
        _add_path(cr, n, base)


#: project runners — the code under the folder they run in is exercised (like a suite's)
_DIR_RUNNERS = {
    "cargo": {"run", "test", "bench"}, "go": {"run", "test"}, "make": {"run", "test", "check", "start", "serve", "dev"},
    "npm": {"start", "test", "t", "run", "run-script", "exec"}, "yarn": {"start", "test", "run", "dev", "serve"},
    "pnpm": {"start", "test", "run", "dev", "serve"}, "bun": {"run", "test", "start"},
    "dotnet": {"run", "test"}, "mvn": {"test", "exec:java", "spring-boot:run"}, "gradle": {"run", "test"},
}
#: `npm run <these>` builds / lints — no run of the code
_NPM_CHECK_SCRIPTS = frozenset({"build", "lint", "typecheck", "type-check", "format", "fmt", "check-types", "tsc"})
_SCRIPT_RUNNERS = frozenset({"bash", "sh", "zsh", "dash", "source", "."})


def _suite_root(cwd: Optional[str]) -> Optional[str]:
    """r2 (review): a suite run from its tests folder tests the project's code — `cd tests && pytest` exercises
    the package above it, as it did before §4UY."""
    if cwd is None:
        return None
    parts = cwd.split("/") if cwd else []
    while parts and re.match(r"^(?:tests?|specs?|__tests__)$", parts[-1]):
        parts.pop()
    return "/".join(parts)


#: r3 (review M5): a built binary's own extensions (`./a.out`, `./x.bin`, `./app.exe`)
_BINARY_EXT_RE = re.compile(r"\.(?:out|bin|exe|elf|run|x86_64|aarch64)$", re.I)
#: in `scripts` (the request's written script / page texts): the binaries a build made this request —
#: output key → (call index, the source keys its command NAMED)
_BUILDS_KEY = "\x00builds"
#: in `scripts`: the index of the tool call being read (a build's position in the request)
_INDEX_KEY = "\x00index"


def _add_built(cr: dict, head: str, cwd: Optional[str], scripts: dict) -> None:
    """r5 (review): a built binary's run credits the sources its BUILD COMMAND NAMED (`gcc -o app main.c` →
    main.c, not a tool.c beside it), written before that build. A build that names none (`make`, `cargo build`,
    `cmake --build`) credits nothing — when unsure, no credit."""
    if cwd is None:
        return
    b = (scripts.get(_BUILDS_KEY) or {}).get(_norm(head, cwd))
    if b:
        ix, srcs = b
        for src in srcs:
            cr["built"][src] = max(cr["built"].get(src, -1), ix)


#: interpreters: the first non-option argument is the script they run
_INTERPRETER_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|pypy\d*|node|nodejs|deno|bun|ruby|perl|php|bash|sh|zsh|dash"
                             r"|ksh|source|\.|ts-node|tsx|rscript|lua|tclsh)$", re.I)
#: their options that take a value (`python3 -W ignore app.py`, `node -r dotenv/config app.js`)
_INTERP_VALUE_OPTS = frozenset({"-W", "-X", "-Q", "-r", "--require", "--import", "--loader", "--experimental-loader",
                                "--input-type", "--env-file", "--config", "-I", "--inspect-port", "--max-old-space-size"})
#: options after which no script follows (inline code / a module: read elsewhere)
_INTERP_NO_SCRIPT = frozenset({"-c", "-e", "-m", "--eval", "-p", "--print", "-"})


def _script_words(ws: list) -> list:
    """r5 (review): the words of one segment (prefixes stripped) that it RUNS as a script — the head when it is a
    code file (`./run.sh`), the first non-option argument of an interpreter, the files of `go run`, through
    `uv/poetry run` and `npx tsx`. Every other file the segment names is an argument, not a run."""
    ws = list(ws)
    if ws[:1] == ["function"]:                           # `function t { python3 app.py; }`
        ws = ws[2:]
    elif ws and re.fullmatch(r"[\w.-]+\(\)", ws[0]):     # `t() { python3 app.py; }`
        ws = ws[1:]
    ws = [w.strip("(){}") for w in ws]
    ws = [w for w in ws if w]
    if not ws:
        return []
    head, rest = ws[0], ws[1:]
    prog = head.rsplit("/", 1)[-1].lower()
    if _is_code(head):
        return [head]
    if prog in ("uv", "poetry", "pipenv", "pdm", "hatch", "rye") and rest[:1] == ["run"]:
        return _script_words(rest[1:])
    if prog in ("npx", "pnpx", "bunx", "pnpm", "yarn") and rest and rest[0].lower() in ("tsx", "ts-node", "node"):
        return _script_words(rest)
    if prog == "go" and rest[:1] == ["run"]:
        return [r for r in rest[1:] if r.endswith(".go")]
    if prog in ("deno", "bun") and rest[:1] == ["run"]:
        rest = rest[1:]
    if not _INTERPRETER_RE.match(prog):
        return []
    skip = False
    for r in rest:
        if skip:
            skip = False
            continue
        if r in _INTERP_NO_SCRIPT:
            return []
        if r.startswith("-"):
            skip = r in _INTERP_VALUE_OPTS
            continue
        return [r]
    return []


def _segment_credits(words: list, cwd: Optional[str], cr: dict, *, prev: str, loop_lists: dict, bodies: list,
                     scripts: dict, depth: int, pypath=()) -> None:
    """Credit what ONE running segment exercises (see the block comment above)."""
    ws = [w for w in _strip_shell_lead(list(words))]
    pypath = list(pypath) + [w.split("=", 1)[1] for w in ws if w.startswith("PYTHONPATH=")]
    i = 0
    while i < len(ws) and (_SEG_ASSIGN_RE.match(ws[i]) or ws[i].rsplit("/", 1)[-1] in _SEG_PREFIX):
        prog0 = ws[i].rsplit("/", 1)[-1]
        i += 1
        while i < len(ws) and (ws[i].startswith("-") or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", ws[i])):
            i += 2 if (ws[i] in _PREFIX_VALUE_OPTS.get(prog0, ()) and i + 1 < len(ws)) else 1
    ws = ws[i:]
    if not ws:
        return
    head, rest = ws[0], ws[1:]
    prog = head.rsplit("/", 1)[-1].lower()
    seg = " ".join(words)
    code = _inline_code(words)
    if code is not None:
        _credit_code_text(cr, code, cwd, pypath)
        return
    if prog in ("bash", "sh", "zsh", "dash") and rest[:1] == ["-c"] and len(rest) > 1:
        _merge(cr, _command_credits(rest[1], cwd, scripts, depth + 1)[0])
        return
    if prog == "git" and rest[:2] == ["bisect", "run"]:
        _segment_credits(rest[2:], cwd, cr, prev=prev, loop_lists=loop_lists, bodies=bodies, scripts=scripts,
                         depth=depth)
        return
    from .agent import is_suite_command
    if is_suite_command(seg):
        cr["suite"] = True
        paths = [a for a in rest if not a.startswith("-") and ("/" in a or _is_code(a))]
        outside = [_norm(a, cwd or "") for a in paths if cwd is not None]
        outside = [p for p in outside if cwd and not (p == cwd or p.startswith(cwd + "/"))
                   or (not cwd and p.startswith(".."))]
        if paths and outside and len(outside) == len(paths):
            for p in outside:                            # `pytest ../backend/tests`: that folder's code
                parts = p.split("/")
                while parts and (re.match(r"^(?:tests?|specs?|__tests__)$", parts[-1]) or "." in parts[-1]):
                    parts.pop()
                _add_dir(cr, "/".join(parts), _suite_exts(seg))
        else:
            _add_dir(cr, _suite_root(cwd), _suite_exts(seg))
    # r5 (review): what the segment RUNS — its head when that is a code file, or the script an interpreter is
    # given; the other files it names are arguments (`python3 lint.py main.py` runs lint.py, not main.py)
    for w in _script_words(ws):
        if re.fullmatch(r"_SUBST(\d+)_", w) or w.startswith("$"):
            continue                                     # (substitutions / loop variables below)
        _add_path(cr, w, cwd)
        if depth < 3:
            _credit_spawned(cr, w, cwd, scripts)
    for a, b in _MODULE_RUN_RE.findall(seg):
        mod = a or b
        if mod and mod.lower() not in _NOT_A_MODULE:
            _add_module(cr, mod, cwd, pypath)
            if a:                                        # `python -m pkg` runs the package
                for base in _module_bases(cwd, pypath):
                    _add_dir(cr, None if base is None else _norm(mod.replace(".", "/"), base), _PY_EXTS)
    # a project runner exercises its folder
    sub = next((r for r in rest if not r.startswith("-")), "")
    if prog in _DIR_RUNNERS and (sub in _DIR_RUNNERS[prog] or (prog == "make" and False)):
        script = rest[rest.index(sub) + 1] if sub in ("run", "run-script") and rest.index(sub) + 1 < len(rest) else ""
        if not (prog in ("npm", "yarn", "pnpm", "bun") and script.lower() in _NPM_CHECK_SCRIPTS):
            target = ""
            if prog == "go" and sub == "run":
                target = next((r for r in rest[rest.index(sub) + 1:] if not r.startswith("-")), "")
                if target.endswith(".go"):
                    target = None                        # `go run main.go`: the named files only (above)
                else:
                    target = target.rstrip("/").rstrip(".").rstrip("/") if target not in (".", "./...") else ""
                    _add_dir(cr, None if cwd is None else _norm(target, cwd), _RUNNER_EXTS.get(prog, ()))
                    target = None
            if target is not None:
                _add_dir(cr, cwd, _RUNNER_EXTS.get(prog, ()))
    elif prog == "java" and rest:
        cls = next((r for r in rest if not r.startswith("-") and not r.endswith((".jar", ".java"))
                    and rest[rest.index(r) - 1] not in ("-cp", "-classpath", "--class-path", "-jar")), "")
        if cls and re.fullmatch(r"[\w.$]+", cls):          # r3 (review M5): `java Main` runs Main.java
            _add_path(cr, cls.replace(".", "/") + ".java", cwd)
    elif prog in ("node", "nodejs", "deno", "bun") and rest:
        tgt = next((r for r in rest if not r.startswith("-")), "")
        if tgt and not _is_code(tgt) and not re.search(r"\.\w+$", tgt.rstrip("/")) or tgt in (".", "./"):
            _add_dir(cr, None if cwd is None else _norm(tgt, cwd), _JS_EXTS)  # `node .` / `node server/`
    elif ("/" in head and not _is_code(head) and (not re.search(r"\.\w{1,5}$", head) or _BINARY_EXT_RE.search(head))
          and prog not in _SCRIPT_RUNNERS):
        _add_built(cr, head, cwd, scripts)                # `./app` a build of this request made
    # a shell script the request wrote: what ITS text runs
    script_name = ""
    if prog in _SCRIPT_RUNNERS and rest:
        script_name = next((r for r in rest if not r.startswith("-")), "")
    elif head.endswith((".sh", ".bash")):
        script_name = head
    if script_name and depth < 3:
        k = _norm(script_name, cwd or "")
        text = scripts.get(k)
        if text is None and cwd is None:
            text = next((t for p, t in scripts.items() if p == k or p.endswith("/" + k)), None)
        if text is not None and k != _BUILDS_KEY:
            _merge(cr, _command_credits(text, cwd, {p: t for p, t in scripts.items() if p != k}, depth + 1)[0])
    for w in _script_words(ws):
        m = re.fullmatch(r"_SUBST(\d+)_", w)            # `python3 $(find . -name app.py)`: a LISTER's output is
        if m and int(m.group(1)) < len(bodies):          # the script; `$(cat other.py)` is an argument's text
            body = bodies[int(m.group(1))].strip()
            if re.match(r"^(?:find|ls|realpath|readlink|which|echo|printf)\b", body):
                for n in _CODE_NAME_RE.findall(body):
                    _add_path(cr, n, cwd)
        for var, items in loop_lists.items():            # `for t in a.py b.py; do python3 $t; done`
            if w in (f"${var}", f"${{{var}}}", f'"${var}"', f'"${{{var}}}"'):
                for n in _CODE_NAME_RE.findall(items):
                    _add_path(cr, n, cwd)
    if prev and _bare_interpreter(words):                # `cat app.py | python3`
        for n in _CODE_NAME_RE.findall(prev):
            _add_path(cr, n, cwd)


_HEREDOC_SPLIT_RE = re.compile(r"^(?P<head>[^\n]*<<-?[ \t]*(['\"]?)(?P<tag>[A-Za-z_][\w.-]*)\2[^\n]*)\n"
                               r"(?P<body>.*?)\n?^[ \t]*(?P=tag)[ \t]*$", re.S | re.M)
_HEREDOC_INTERP_RE = re.compile(r"(?:^|[\s;&|(])(?P<interp>python[\d.]*|node|nodejs|deno|bash|sh|zsh|dash)"
                                r"(?:\s+-\S*)*\s*(?:-\s*)?<<"
                                # `cat <<EOF | python3` — the body piped into an interpreter
                                r"|<<.*\|\s*(?P<interp2>python[\d.]*|node|nodejs|deno|bash|sh|zsh|dash)(?:\s+-\S*)*\s*-?\s*$")


def _split_heredocs(cmd: str) -> tuple:
    """(the command with every heredoc BODY removed, [(interpreter, body)] for bodies fed to an interpreter).
    A body written to a file (`cat > test.py <<EOF … import app … EOF`) is data — reading it as commands credited
    every file it named (§4UU residual)."""
    fed = []

    def _sub(m):
        im = _HEREDOC_INTERP_RE.search(m.group("head"))
        if im:
            fed.append((im.group("interp") or im.group("interp2"), m.group("body")))
        return m.group("head")
    return _HEREDOC_SPLIT_RE.sub(_sub, str(cmd or "")), fed


def _masked(seps: list, i: int) -> bool:
    """Segment i's status is masked: its `&&` / `|` chain reaches an `||` before the list ends."""
    j = i
    while j < len(seps):
        s = seps[j]
        if s == "||":
            return True
        if s not in ("&&", "|", "|&"):
            return False
        j += 1
    return False


def _cd_target(words: list, cwd: Optional[str]) -> Optional[str]:
    ws = _strip_shell_lead(list(words))
    tgt = next((w for w in ws[1:] if not w.startswith("-")), "")
    if not tgt or tgt in ("~", "/workspace"):
        return ""
    if "$" in tgt or "`" in tgt or tgt == "-" or "*" in tgt:
        return None
    if cwd is None and not tgt.startswith(("/", "~/")):
        return None
    return _norm(tgt, cwd or "")


def _command_credits(command: str, cwd: Optional[str] = "", scripts: Optional[dict] = None,
                     depth: int = 0) -> tuple:
    """(credits, checked keys) for a shell command run from `cwd`. A masked run (`… || true`) credits nothing."""
    scripts = scripts or {}
    cmd, fed = _split_heredocs(_SHELL_WRITE_RE.sub(" ", str(command or "")))
    cr, checked = _new_credits(), set()
    seps: list = []
    bodies: list = []
    segs = _shell_segments(cmd, bodies, seps)
    loop_lists: dict = {}
    prev = ""
    pypath: list = []
    for idx, words in enumerate(segs):
        if len(words) > 2 and words[0] == "for" and words[2] == "in":
            loop_lists[words[1]] = " ".join(words[3:])
        lead = _strip_shell_lead(list(words))
        if lead and lead[0] in ("cd", "pushd"):
            cwd = _cd_target(lead, cwd)
            prev = " ".join(words)
            continue
        if lead and lead[0] == "export":                 # `export PYTHONPATH=src; python -m pkg`
            pypath += [w.split("=", 1)[1] for w in lead[1:] if w.startswith("PYTHONPATH=")]
        kind = segment_kind(words)
        if kind == "run":
            if not (idx < len(seps) and _masked(seps, idx)):
                _segment_credits(words, cwd, cr, prev=prev, loop_lists=loop_lists, bodies=bodies, scripts=scripts,
                                 depth=depth, pypath=pypath)
        elif kind == "check":
            _note_build(words, cwd, scripts)
            for w in words:
                if _CODE_NAME_RE.fullmatch(w):
                    checked.add(_norm(w, cwd or ""))
        prev = " ".join(words)
    for interp, body in fed:                              # `python3 - <<EOF` runs the body
        if re.match(r"(?:ba|z|da)?sh$", interp):
            sub, sub_checked = _command_credits(body, cwd, scripts, depth + 1)
            _merge(cr, sub)
            checked |= sub_checked
        else:
            _credit_code_text(cr, body, cwd, pypath)
    return cr, checked


def _note_build(words: list, cwd: Optional[str], scripts: dict) -> None:
    """A compile / build ran in `cwd` (into `scripts[_BUILDS_KEY]`, shared across the request's calls): a binary
    under that folder is now one this request built."""
    if cwd is None or _BUILDS_KEY not in scripts:
        return
    ws = [w for w in _strip_shell_lead(list(words)) if not _SEG_ASSIGN_RE.match(w)]
    if not ws:
        return
    prog = ws[0].rsplit("/", 1)[-1].lower()
    if _compile_only(prog, ws[1:]):
        out, srcs = _build_output(prog, ws[1:], cwd)
        if out and srcs:
            scripts[_BUILDS_KEY][out] = (int(scripts.get(_INDEX_KEY, 0)), frozenset(srcs))


def _build_output(prog: str, rest: list, cwd: str) -> tuple:
    """(the binary's key, the source keys the command names) of one compile — (None, ()) when either is not
    named. gcc / clang with no -o write a.out; rustc / ghc / swiftc name it after the first source; `go build`
    names package folders (`.`, `./cmd/x`) or .go files."""
    out, srcs, skip = None, [], False
    for j, w in enumerate(rest):
        if skip:
            skip = False
            continue
        if w == "-o":
            out = rest[j + 1] if j + 1 < len(rest) else None
            skip = True
            continue
        if w.startswith("-o") and len(w) > 2 and prog != "go":
            out = w[2:]
            continue
        if w.startswith("-") or "*" in w or "$" in w or "=" in w:
            continue
        if prog == "go":
            if w == "build":
                continue
            if w.endswith(".go"):
                srcs.append(_norm(w, cwd))
            elif w in (".", "./") or w.startswith(("./", "../")):    # (`./...` names no one folder)
                srcs.append("\x00go:" + _norm(w, cwd))
        elif w.lower().endswith(_COMPILED_EXTS):
            srcs.append(_norm(w, cwd))
    if out is None and srcs:
        if re.match(r"^(?:gcc|g\+\+|clang\+?\+?|cc|c\+\+)$", prog):
            out = "a.out"
        elif prog in ("rustc", "ghc", "swiftc", "nim", "zig"):
            out = srcs[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            return _norm(out, cwd), srcs
    return (_norm(out, cwd) if out else None), srcs


def _exec_credits(tool: dict, args: dict, scripts: Optional[dict] = None) -> tuple:
    """(credits, checked keys) of one `execute` call: its command, and an `execute(filename=, content=)` script —
    the file itself, plus what a SHELL script's text runs or what code IMPORTS / EXECUTES (§4UY C3: a script
    whose text merely names a file did not run it)."""
    scripts = dict(scripts or {})
    cr, checked = _command_credits(str(args.get("command") or ""), "", scripts)
    fn = str(args.get("filename") or "")
    content = str(args.get("content") or "")[:20000]
    if fn:
        _add_path(cr, fn, "")
    if content:
        base = _norm(fn, "").rsplit("/", 1)[0] if "/" in _norm(fn, "") else ""
        if fn.lower().endswith((".sh", ".bash")) or (not fn and content.lstrip().startswith("#!")
                                                      and "sh" in content[:40]):
            sub, sub_checked = _command_credits(content, base, scripts, 1)
            _merge(cr, sub)
            checked |= sub_checked
        else:
            _credit_code_text(cr, content, base)
    return cr, checked - cr["exact"]


def _exec_run_keys(tool: dict, args: dict) -> tuple:
    """(exact path keys the run exercised, is_suite, keys only checked) — the §4UU-3 view of `_exec_credits`."""
    cr, checked = _exec_credits(tool, args)
    return cr["exact"] | cr["suffix"], bool(cr["suite"]), checked


def _browser_keys(tool: dict, args: dict) -> dict:
    """The pages a browser / vision call loaded, as credits: a file:// or workspace path exactly, a page served
    from THIS machine (localhost) by suffix — its server root is not known. A page on another host is not ours."""
    cr = _new_credits()
    vals = [args.get(k) for k in ("url", "path", "target", "file", "page", "image", "out_path")]
    acts = args.get("actions")
    if isinstance(acts, list):                         # `interact actions=[{action: goto, url: …}]`
        vals += [a.get("url") for a in acts if isinstance(a, dict)
                 and str(a.get("action") or "").lower() in ("goto", "navigate", "open")]
    for v in vals:
        if not isinstance(v, str) or not v:
            continue
        v = v.split("?", 1)[0].split("#", 1)[0]
        if re.match(r"(?i)^https?://", v):
            from urllib.parse import urlparse
            u = urlparse(v)
            if (u.hostname or "").lower() not in _LOCAL_HOSTS:
                continue
            p = u.path or ""
            if p in ("", "/") or p.endswith("/"):        # a served ROOT names the index page
                p = p + "index.html"
            if _is_web(p):
                cr["suffix"].add(_norm(p.lstrip("/"), ""))
            continue
        if _is_web(v):
            cr["exact"].add(_norm(v, ""))
    return cr


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
_SCRIPT_EXTS = (".js", ".mjs", ".cjs")
#: `<script src="js/app.js">` — the scripts a page LOADS (§4UY C20)
_SCRIPT_SRC_RE = re.compile(r"""<script\b[^>]*?\bsrc\s*=\s*["']([^"'>]+)["']""", re.I)


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


#: in `scripts`: the host folders the request's workspace paths live in (the project folders it named, the root)
_ROOTS_KEY = "\x00roots"
_PROJECT_REF_RE = re.compile(r"projects/([0-9a-f]{12})(?=/|\b)")
_HOST_PAGE_MAX = 1 << 20


def _host_roots(tools_run: Optional[list]) -> list:
    """r3 (review M3): the host folders a workspace key may live in — the request's project folder (its scope,
    and any `projects/<id>/` its calls named), then the workspace root. Nothing when there is no workspace."""
    import os
    from pathlib import Path
    home = os.environ.get("GHOST_HOME") or str(Path.home() / "ghost_llamacpp")
    base = Path(home) / "sandbox"
    ids: list = []
    try:
        from .request_scope import current_scope
        sc = current_scope()
        if sc is not None and sc.project_id:
            ids.append(str(sc.project_id))
    except Exception:  # noqa: BLE001
        pass
    for t in tools_run or []:
        if isinstance(t, dict):
            for m in _PROJECT_REF_RE.findall(str(_args_of(t))):
                if m not in ids:
                    ids.append(m)
    return [base / "projects" / x for x in ids] + [base]


def _host_text(root, key: str) -> Optional[str]:
    """The text of workspace file `key` under host folder `root` — None when absent, too big, or the key
    leaves the folder."""
    import stat as _stat
    try:
        f = root / key
        if not f.parent.resolve().is_relative_to(root.resolve()):
            return None
        # r4 (review): the workspace is the model's — a FIFO planted at the path would block the open forever, a
        # link would lead out of it: open the final name O_NOFOLLOW|O_NONBLOCK, read only a REGULAR file
        # (execute.read_status_file's rule)
        fd = os.open(str(f), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except Exception:  # noqa: BLE001
        return None
    try:
        from ..tools.file_system import _block_if_regular  # §4SD: a regular file reads BLOCKING (virtiofs EAGAIN)
        st = _block_if_regular(fd)
        if not _stat.S_ISREG(st.st_mode) or st.st_size > _HOST_PAGE_MAX:
            return None
        return os.read(fd, _HOST_PAGE_MAX).decode("utf-8", errors="replace")
    except OSError:
        return None
    finally:
        os.close(fd)


def _host_pages(page: str, led: dict, texts: dict, served: bool) -> list:
    """r3 (review M3): a page loaded clean whose text this request did not write (an edited or untouched
    index.html): its text as it is in the workspace now. A file page is its own key; a served page's root is
    unknown, so it is looked for at the workspace root and under the folders of the code the request wrote."""
    if not page:
        return []                                        # (a key outside the root is refused by _host_text)
    cands = [page]
    if served:
        for k in led:
            parts = k.split("/")[:-1]
            for j in range(len(parts), 0, -1):
                c = "/".join(parts[:j]) + "/" + page
                if c not in cands:
                    cands.append(c)
    for root in texts.get(_ROOTS_KEY) or ():
        for c in cands:
            txt = _host_text(root, c)
            if txt is not None:
                texts[c] = txt
                return [c]
    return []


def _credit_served_scripts(led: dict, texts: dict, urls: list, i: int) -> None:
    """§4UY C20: a page that loaded CLEAN ran the scripts IT loads — the `<script src>` of that page's text as the
    request wrote it, resolved against the page's folder (a served page: by suffix, its root is unknown). A
    written script the page does not load was not run by the load; a page whose text is unknown credits none.
    Only a CLEAN load is credited (the caller checks)."""
    from urllib.parse import urlparse
    import posixpath
    for u in urls:
        u = str(u or "").split("?", 1)[0].split("#", 1)[0]
        served = False
        if re.match(r"(?i)^https?://", u):
            pu = urlparse(u)
            if (pu.hostname or "").lower() not in _LOCAL_HOSTS:
                continue
            served = True
            path = pu.path or "/"
            if path.endswith("/"):
                path += "index.html"
            page = _norm(path.lstrip("/"), "")
            pages = [k for k in texts if k == page or k.endswith("/" + page)]
        else:
            page = _norm(u, "")
            pages = [k for k in texts if k == page]
        if not pages:                                    # r3 (review M3): a page this request did not write whole
            pages = _host_pages(page, led, texts, served)
        for pk in pages:
            pdir = pk.rsplit("/", 1)[0] if "/" in pk else ""
            for src in _SCRIPT_SRC_RE.findall(str(texts.get(pk) or "")):
                src = src.split("?", 1)[0].split("#", 1)[0]
                if re.match(r"(?i)^(?:https?:)?//|^data:", src):
                    continue
                if src.startswith("/"):
                    want, exact = _norm(src.lstrip("/"), ""), False
                else:
                    want, exact = _norm(posixpath.join(pdir, src), ""), not served
                for k, rec in led.items():
                    if rec["write_idx"] > i or not k.endswith(_SCRIPT_EXTS):
                        continue
                    if k == want or (not exact and k.endswith("/" + want)):
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
    if _own_script_throws_only(text):
        return True
    return not text.lstrip().lower().startswith(("error", "[failure banner]"))


_ACTION_LINE_RE = re.compile(r"^\s*\[\d+\] (OK|ERR) (\w+)(.*)$", re.M)


def _own_script_throws_only(text: str) -> bool:
    """r3 (live fb512751): a stored interact booked PARTIAL only because the model's own `evaluate` threw — the
    browser tool's §4RJ rule (browser.py) now reports that load OK: the page loaded and the other actions ran.
    A timed-out evaluate stays a failure."""
    if not re.search(r"^STATUS:[ \t]*PARTIAL\b", text, re.M):
        return False
    acts = _ACTION_LINE_RE.findall(text)
    errs = [a for a in acts if a[0] == "ERR"]
    return bool(errs) and any(a[0] == "OK" for a in acts) and all(
        a[1] == "evaluate" and "timed out" not in a[2] for a in errs)


def _run_tool_ok(tool: dict) -> Optional[bool]:
    """§4UY C2: the `run` tool's verdict — a server that STARTED (registered, alive) and a collected job that
    exited 0 pass; a failed start / a non-zero exit fail; a started job (no result yet), a refusal or an unknown
    exit is no run."""
    c = tool.get("content")
    st = str(getattr(getattr(c, "status", None), "value", "") or "")
    text = str(c or "")
    if st == "rejected" or st == "unresolved":
        return None
    if st == "failed":
        return False
    if st == "ok":
        return True
    from ..tools.outcome import envelope_exit_code
    code = envelope_exit_code(text)                      # the head envelope only (a log's quoted banner is data)
    if code is not None:
        return code == 0
    if text.lstrip().startswith("RUN ERROR"):
        return False
    if re.match(r"\s*\[run \S+ started — server", text):
        return True
    return None


# ── the ledger ───────────────────────────────────────────────────────

def _new_rec(path, i, via) -> dict:
    return {"path": path, "write_idx": i, "run_idx": None, "run_ok": None, "via": via, "checked": False}


def _apply_page_checks(led: dict, row, i: int) -> None:
    """§4UU-1: the system's headless page check after a web write (its note on the write's own row) RUNS the
    page and every script it loaded — a clean load as a pass, a page that throws as a failure. The check names
    the files it ran by name."""
    try:
        from .page_check import checks_in
    except Exception:  # noqa: BLE001
        return
    for ok, keys in checks_in(str(row.get("content") or "")):
        for k, rec in led.items():
            if k in keys or k.rsplit("/", 1)[-1] in keys:
                rec["run_idx"], rec["run_ok"] = i, ok


_CHECKLIST_ROW_RE = re.compile(r"^(✓|✗) #(\d+) .*? — check: (.+?)$", re.M)


def _checklist_runs(tool: dict, args: dict, scripts: Optional[dict] = None) -> list:
    """§4UU-2: a `checklist check` RAN each chosen item's check — [(ok, credits), …] by the same rule as an
    execute (a shell check) or the page probe (`page:<file>`). `view` / `set` re-print old states: no run."""
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
            cr = _new_credits()
            cr["exact"].add(_norm(check[5:].strip(), ""))
            out.append((mark == "✓", cr))
            continue
        out.append((mark == "✓", _command_credits(check, "", scripts)[0]))
    return out


def _texts_written(n: str, args: dict, tool: dict) -> dict:
    """{key: text} of the shell scripts and web pages this call wrote whole (a file_system write, an execute
    heredoc / `execute(filename=, content=)`) — the text a later run of the script or load of the page reads."""
    out = {}
    if n in ("file_system", "filesystem", "file"):
        if str(args.get("operation") or "").lower() == "write" and args.get("content") is not None:
            p = str(args.get("path") or args.get("filename") or "")
            if p.lower().endswith((".sh", ".bash") + WEB_EXTS) or _is_code(p):
                out[_fs_norm(p)] = str(args.get("content") or "")
        return out
    for m in _HEREDOC_WRITE_RE.finditer(str(args.get("command") or "")):
        tgt = m.group("t1") or m.group("t2") or ""
        if tgt.lower().endswith((".sh", ".bash") + WEB_EXTS):
            out[_norm(tgt, "")] = m.group("body")
    fn = str(args.get("filename") or "")
    if fn and args.get("content") and (fn.lower().endswith((".sh", ".bash")) or _is_code(fn)):
        out[_norm(fn, "")] = str(args.get("content"))
    return out


#: `cat > x.sh <<'EOF' … EOF` / `cat <<EOF > x.sh … EOF` — a file written from a heredoc body
_HEREDOC_WRITE_RE = re.compile(
    r"(?:>\s*['\"]?(?P<t1>[\w.\-/]+)['\"]?[^\n]*?<<-?\s*(['\"]?)(?P<tag1>\w+)\2"
    r"|<<-?\s*(['\"]?)(?P<tag2>\w+)\4[^\n]*?>\s*['\"]?(?P<t2>[\w.\-/]+))[^\n]*\n"
    r"(?P<body>.*?)\n[ \t]*(?:(?P=tag1)|(?P=tag2))[ \t]*(?:\n|$)", re.S)


def _apply_edit_text(texts: dict, args: dict) -> None:
    """A file_system edit / append of a page or script whose text is known: keep the text current."""
    op = str(args.get("operation") or "").lower()
    k = _fs_norm(str(args.get("path") or ""))
    if k not in texts:
        return
    if op == "edit" and args.get("old_string") is not None:
        old, new = str(args.get("old_string")), str(args.get("new_string") or "")
        texts[k] = texts[k].replace(old, new, 1) if old in texts[k] else texts[k]
    elif op == "append":
        texts[k] = texts[k] + str(args.get("content") or "")


def _written_key(command: str, path: str) -> str:
    """The ledger key of a file a command wrote: resolved against the `cd` in force where the command names it
    (`cd backend && cat > app.py <<EOF` writes backend/app.py)."""
    pos = command.find(path)
    cwd: Optional[str] = ""
    if pos > 0:
        for words in _shell_segments(command[:pos]):
            ws = _strip_shell_lead(list(words))
            if ws and ws[0] in ("cd", "pushd"):
                cwd = _cd_target(ws, cwd)
    if cwd is None:
        return _norm(path, "")
    return _norm(path, cwd)


def _credit(led: dict, cr: dict, i: int, ok: Optional[bool]) -> None:
    for k, rec in led.items():
        if _credited(k, cr, web=_is_web(rec["path"]), write_idx=rec["write_idx"]):
            rec["run_idx"], rec["run_ok"] = i, ok


def code_evidence(tools_run: Optional[list]) -> dict:
    """key → {path, write_idx, run_idx, run_ok, via, checked}. Keys are normalised workspace paths (§4UY C3).
    `run_idx` is the last run AFTER the last write (None when nothing ran the final version); `run_ok` is that
    run's verdict (None = detached / unknown); `checked` — a syntax / lint check named the final version (it is
    still not a run). A file_system row that carries a §4UU-1 page check counts as that run, right after its own
    write."""
    led: dict = {}
    texts: dict = {_BUILDS_KEY: {}, _ROOTS_KEY: _host_roots(tools_run)}                    # key → text of a written shell script / web page
    run_cmds: dict = {}                                   # `run` name → command (a job's collect reads it)
    for i, t in enumerate(tools_run or []):
        if not isinstance(t, dict) or t.get("_synthetic"):
            continue
        n = _name(t)
        args = _args_of(t)
        texts[_INDEX_KEY] = i
        if n in ("file_system", "filesystem", "file"):
            if str(args.get("operation") or "").lower() == "delete" and \
                    str(t.get("content") or "").lstrip().upper().startswith("SUCCESS"):
                led.pop(_fs_norm(args.get("path") or ""), None)      # a deleted file is not "never run"
                continue
            written = _fs_written(t)
            for p in written:
                led[_fs_norm(p)] = _new_rec(p, i, "file_system")
            if written:
                new = _texts_written(n, args, t)
                if args.get("append") in (True, "true", "True", 1):     # r3: a part appended to the text so far
                    new = {k: texts[k] + v if isinstance(texts.get(k), str) else None for k, v in new.items()}
                    for k, v in new.items():
                        if v is None:
                            texts.pop(k, None)               # appended to a text we never saw: unknown
                        else:
                            texts[k] = v
                else:
                    texts.update(new)
                _apply_edit_text(texts, args)
                _apply_page_checks(led, t, i)
            continue
        if n == "execute":
            ok = _exec_ok(t)
            if ok:
                cwd: Optional[str] = ""
                for words in _shell_segments(str(args.get("command") or "")):
                    ws = _strip_shell_lead(list(words))
                    if ws and ws[0] in ("cd", "pushd"):
                        cwd = _cd_target(ws, cwd)
                    elif ws and ws[0] == "rm":                   # `rm fix2.py` — gone, not "never run"
                        for w in ws[1:]:
                            if not w.startswith("-"):
                                led.pop(_norm(w, cwd or ""), None)
            for p in _exec_written(t, args):
                wk = _written_key(str(args.get("command") or ""), p)
                if not _SYSTEM_FILE_RE.match(wk):            # r3 (live 6aee2a43): machine config, not the program
                    led[wk] = _new_rec(p, i, "execute")
            if ok is not None:
                texts.update(_texts_written(n, args, t))
            if ok is None:
                continue
            cr, checked = _exec_credits(t, args, texts)
            # every record here was written at an index <= i (a later write
            # REPLACES the record, which is how order is kept)
            for k, rec in led.items():
                if _credited(k, cr, web=_is_web(rec["path"]), write_idx=rec["write_idx"]):
                    rec["run_idx"], rec["run_ok"] = i, ok
                elif k in checked and rec["run_idx"] is None:
                    rec["checked"] = True
            # a Playwright script that loads a page (`page.goto("file:///…/index.html")`) loads its scripts too
            gotos = _GOTO_RE.findall(str(args.get("command") or "") + "\n" + str(args.get("content") or ""))
            if gotos and ok:
                for g in gotos:                                  # the page itself, then the scripts it loads
                    _credit(led, _browser_keys({}, {"url": g}), i, ok)
                _credit_served_scripts(led, texts, gotos, i)
            # `execute(filename=x, content=…)` writes AND runs x in one call
            fn = str(args.get("filename") or "")
            if fn and args.get("content") and _norm(fn, "") in led and led[_norm(fn, "")]["write_idx"] == i:
                led[_norm(fn, "")]["run_idx"], led[_norm(fn, "")]["run_ok"] = i, ok
            continue
        if n == "run":
            # §4UY C2: the foreground-server block sends servers here — a started server and a collected job
            # are runs of what their command runs (in the project folder)
            act = str(args.get("action") or "").strip().lower()
            name = str(args.get("name") or "").strip()
            where = str(args.get("where") or "workspace").strip().lower()
            if act == "start" and args.get("command") and where == "workspace":
                run_cmds[name] = (str(args.get("command")), str(args.get("lifetime") or "").strip().lower())
            cmd, life = run_cmds.get(name, ("", "")) if act in ("start", "restart", "collect") else ("", "")
            if not cmd or where != "workspace" or (act != "collect" and life == "job"):
                continue                                         # a started job is not a result yet
            ok = _run_tool_ok(t)
            if ok is None:
                continue
            _credit(led, _command_credits(cmd, "", texts)[0], i, ok)
            continue
        if n == "checklist":
            for ok, cr in _checklist_runs(t, args, texts):
                _credit(led, cr, i, ok)
            continue
        if n in ("browser", "vision_analysis"):
            cr = _browser_keys(t, args)
            if not (cr["exact"] or cr["suffix"]):
                continue
            ok = _browser_ok(t)
            if ok is None:
                continue
            _credit(led, cr, i, ok)
            if ok and n == "browser" and _load_was_clean(t):
                _credit_served_scripts(led, texts, _browser_urls(args), i)
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
