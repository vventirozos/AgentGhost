"""Pre-execution validators for shell — roadmap phase 1.4. (§4SX E: the SQL validator went with
`postgres_admin`; SQL runs on a lab machine now, through `lab exec`.)

These run BEFORE a candidate command/statement is dispatched to its
executor. The goal is to catch shape errors (unclosed quotes, missing
semicolons in DDL, obviously-destructive forms) at a phase where the
agent's prompt history can still be enriched with a diagnostic, rather
than after a tool call has corrupted host state.

Contract for every validator:
    ``validate_X(text: str) -> (ok: bool, reason: str)``

Validators are deliberately CONSERVATIVE — they false-negative
(reject safe statements that happen to look risky) rather than
false-positive. A rejection is a hint to the planner, not a final
verdict; the agent can re-emit with a clarification or ask the user.

Validators never raise. A validator that itself crashes returns
``(True, "validator-error: <type>")`` so the bug doesn't break a
production turn. The bug surfaces in logs at debug level.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import Tuple

logger = logging.getLogger("GhostAgent")


# ──────────────────────────────────────────────────────────────────────
# Shell validator
# ──────────────────────────────────────────────────────────────────────

# §4UY C8: the deny list reads the COMMANDS of a shell text, never its data. It used to run its regexes over
# the whole text — a heredoc body written to a file (a Dockerfile's `rm -rf /var/cache/x`), quoted text
# (`echo '… rm -rf / …' >> NOTES.md`) and comments were refused as if they ran. Now:
#   * `rm` is judged per simple command (`tools/shell_analysis._shell_segments` — the one shell reader the
#     execute guards use: quotes are words, comments and data heredoc bodies are set aside, a heredoc fed to
#     a shell, `bash -c '…'`, `eval`, `$( … )` and `os.system("…")` are commands of their own);
#   * the other forms (dd/mkfs/shred to a device, a fork bomb, a download piped into an interpreter) are
#     matched over the CODE VIEW (`_code_view`): comments and data heredoc bodies removed, quoted strings
#     blanked except a shell's `-c` / `eval` payload, which is code.

#: an rm target that may not be removed recursively: an absolute path (a named subfolder of /tmp — a glob
#: with a literal prefix included, `/tmp/build_*` — is scratch; /tmp itself, `/tmp/*` and `..` stay
#: blocked; absolute /workspace/… stays DENIED: top-level workspace folders hold the owner's data, §4KW-r4),
#: the home folder in any spelling
_TMP_SCRATCH_RE = re.compile(r"^/tmp/[\w\-][\w\-./*?\[\] ]*$")


_ANSI_C_ESC = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f",
               "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}


def _ansi_c_unquote(s: str) -> str:
    """`s` with bash's ANSI-C `$'…'` quoting decoded into a plain single-quoted word and a locale `$"…"` read as
    `"…"` (§4UY r1: `bash -c $'rm -rf /etc'` and `rm -rf $'/'` read as the words `$rm` / `$/` and passed). Only
    outside quotes, an escaped `\$'` stays as written."""
    out, i, q = [], 0, None
    while i < len(s):
        c = s[i]
        if q:
            out.append(c)
            if c == "\\" and q == '"' and i + 1 < len(s):
                out.append(s[i + 1])
                i += 2
                continue
            if c == q:
                q = None
            i += 1
            continue
        if c == "\\" and i + 1 < len(s):
            out.append(s[i:i + 2])
            i += 2
            continue
        if c == "$" and s[i + 1:i + 2] == '"':
            i += 1
            continue
        if c == "$" and s[i + 1:i + 2] == "'":
            j, buf = i + 2, []
            while j < len(s) and s[j] != "'":
                if s[j] == "\\" and j + 1 < len(s):
                    e = s[j + 1]
                    m = re.match(r"x([0-9a-fA-F]{1,2})|([0-7]{1,3})", s[j + 1:j + 4])
                    if e in _ANSI_C_ESC:
                        buf.append(_ANSI_C_ESC[e])
                        j += 2
                    elif m:
                        buf.append(chr(int(m.group(1), 16) if m.group(1) else int(m.group(2), 8)))
                        j += 1 + len(m.group(0))
                    else:
                        buf.append(s[j:j + 2])
                        j += 2
                    continue
                buf.append(s[j])
                j += 1
            out.append("'" + "".join(buf).replace("'", "'\\''") + "'")
            i = j + 1
            continue
        if c in "'\"":
            q = c
        out.append(c)
        i += 1
    return "".join(out)


def _brace_words(t: str, cap: int = 64) -> list:
    """The words a brace expansion makes (`{/,}` → `/`, ``; `/{etc,usr}` → `/etc`, `/usr`), capped."""
    m = re.search(r"\{([^{}]*,[^{}]*)\}", t)
    if not m:
        return [t]
    out = []
    for alt in m.group(1).split(","):
        out.extend(_brace_words(t[:m.start()] + alt + t[m.end():], cap))
        if len(out) >= cap:
            break
    return out[:cap]


def _dangerous_rm_target(t: str) -> bool:
    t = str(t or "").strip()
    if not t:
        return False
    if "{" in t and "," in t:
        return any(_dangerous_rm_target(w) for w in _brace_words(t) if w.strip() and w != t)
    if t == "~" or t.startswith("~/") or re.match(r"^\$\{?HOME\}?(?:/|$)", t):
        return True
    if t.startswith("/"):
        return not (_TMP_SCRATCH_RE.match(t) and ".." not in t.split("/"))
    return False


def _rm_recursive_force(args) -> bool:
    opts = [a for a in args if a.startswith("-") and a != "-"]
    rec = any(a == "--recursive" or (not a.startswith("--") and re.search(r"[rR]", a[1:])) for a in opts)
    force = any(a == "--force" or (not a.startswith("--") and "f" in a[1:]) for a in opts)
    return rec and force


#: removal APIs inside interpreter code that shell_analysis reports as `rm -rf <arg>` — a Python/Node call,
#: not a shell `rm`; the old text rule never read them either (`rmtree` is not the word `rm`)
_API_REMOVAL_RAW_RE = re.compile(r"^\s*(?:fs\.(?:promises\.)?rm\b|rmtree|rmSync|rmdirSync|unlinkSync|removedirs"
                                 r"|remove|unlink|rmdir)\s*\(")


def _rm_denied(cmd: str) -> bool:
    from .shell_analysis import _shell_segments
    for head, args, raw, _via_xargs in _shell_segments(cmd):
        if head != "rm" or _API_REMOVAL_RAW_RE.match(str(raw or "")):
            continue
        if not _rm_recursive_force(args):
            continue
        targets, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            r = re.match(r"^\d*(?:[<>]{1,2}|&>>?|>\|)&?(\S*)$", a)
            if r:                                   # a redirection: `> /dev/null`, `2>&1`, `>log` — not a target
                skip = not r.group(1)
                continue
            if not a.startswith("-") or a == "-":
                targets.append(a)
        if any(_dangerous_rm_target(t) for t in targets):
            return True
    return False


def _strip_comments(s: str) -> str:
    """`s` without its shell comments — a `#` that starts a word OUTSIDE quotes runs to the end of the line
    (`a#b` is a word; `ls # don't` is `ls`)."""
    out, q, i = [], None, 0
    while i < len(s):
        c = s[i]
        if q:
            out.append(c)
            if c == "\\" and q == '"' and i + 1 < len(s):
                out.append(s[i + 1])
                i += 2
                continue
            if c == q:
                q = None
            i += 1
            continue
        if c == "\\" and i + 1 < len(s):
            out.append(s[i:i + 2])
            i += 2
            continue
        if c in "'\"":
            q = c
        elif c == "#" and (i == 0 or s[i - 1] in " \t\n;|&("):
            nl = s.find("\n", i)
            i = len(s) if nl < 0 else nl
            continue
        out.append(c)
        i += 1
    return "".join(out)


_SHELL_PAYLOAD_RE = re.compile(r"(?:\b(?:ba|z|da|k)?sh\s+-[A-Za-z]*c[A-Za-z]*\s+(?:--\s+)?|\beval\s+)"
                               r"(?:'([^']*)'|\"((?:[^\"\\]|\\.)*)\")")


def _code_view(cmd: str) -> str:
    """The shell text as code only: comments and data heredoc bodies removed (a body a shell reads is kept as
    code), every quoted string blanked except a shell's `-c` / `eval` payload (inlined: it is code)."""
    from .shell_analysis import _strip_heredocs
    text, _bodies = _strip_heredocs(str(cmd or ""))
    # (a body fed to a shell needs no inlining here: the shipped list reads the raw text, bodies included — r4)
    text = _strip_comments(text)
    payloads = []

    def _keep(m):
        payloads.append(m.group(1) if m.group(1) is not None else m.group(2))
        return " ; "
    text = _SHELL_PAYLOAD_RE.sub(_keep, text)
    text = re.sub(r"'[^'\n]*'|\"(?:[^\"\\\n]|\\.)*\"", "''", text)
    return "\n".join([text] + [_code_view(p) for p in payloads])


# Forms refused wherever they appear in the CODE VIEW (`rm` is judged per command, see `_rm_denied`).
_SHELL_DENY: tuple = (
    # dd to a raw device
    re.compile(r"\bdd\b[^|;&\n]*of=/dev/(?:sd|nvme|hd)"),
    # mkfs / fdisk / shred against a device
    re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b\s+/dev/"),
    re.compile(r"\bshred\b\s+/dev/"),
    # Fork bomb
    re.compile(r":\(\)\s*\{\s*:\|\s*:&\s*\}\s*;\s*:"),
    # Chmod 777 on root / system dirs
    re.compile(r"\bchmod\b\s+(?:-R\s+)?(?:0?777|a\+rwx)\b\s+/(?:bin|etc|usr|sys|root)\b"),
    # Download piped straight into an interpreter — common malware shape.
    # SHELLS stay fully blocked (curl | sh, | bash -s, …). Scripting
    # interpreters block only when the pipe feeds them AS THE PROGRAM —
    # bare (`| python3`) or explicit stdin (`| python3 -`). With `-m mod`,
    # `-c code`, or a script file the piped bytes are DATA, not code:
    # `curl … | python3 -m json.tool` is a legitimate verification pattern
    # the old blanket rule false-positived on (2026-07-25 audit: the block
    # made the agent substitute a WEAKER check instead of pretty-printing
    # its POST result).
    re.compile(
        r"\b(?:curl|wget|fetch)\b[^|\n]+\|\s*(?:sudo\s+)?"
        r"(?:sh|bash|zsh|dash|ksh)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:curl|wget|fetch)\b[^|\n]+\|\s*(?:sudo\s+)?"
        r"(?:python[0-9.]*|perl|ruby|node|php)\s*(?:-\s*)?(?:\||;|&|$)",
        re.IGNORECASE | re.M,
    ),
)


_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][\w.-]*)\1[^\n]*\n(.*?)^\t*\2[ \t]*$",
                         re.S | re.M)


def _strip_heredoc_bodies(cmd: str) -> str:
    """The command with every heredoc BODY removed (kept: the `<<EOF` line and
    its terminator). A body is data, not shell — one apostrophe in a file
    written with `cat > f <<'EOF'` failed the quote check (§4LO: four valid
    commands, `bash -n` clean, were refused)."""
    return _HEREDOC_RE.sub(lambda m: m.group(0)[:m.start(3) - m.start(0)] + m.group(2), cmd)


# ── §4UY r4: never weaker than the shipped deny list, by construction ────────────────────────────────────────
# The deny list as it shipped before §4UY runs over the WHOLE raw text (and over its ANSI-C-decoded form). A
# match is let through only when it lies entirely inside a region PROVEN to be data (`_data_spans`):
#   (a) a heredoc body, (b) a quoted string — both only when EVERY command of the text is a program that never
#       runs its input or arguments (`_INERT_HEADS`: cat, tee, echo, printf, grep …; no assignment, no
#       `$(…)`, backtick, `<(…)`, `>(…)` or `((…))` anywhere), so nothing can feed them to a shell;
#   (c) a comment.
# Anything the lexer cannot read the way bash does (an unterminated quote, a computed heredoc delimiter, a
# substitution) proves nothing: the match is refused as before.
_LIVE_DENY: tuple = (
    re.compile(
        r"\brm\b(?=[^|;&\n]*\s(?:-\w*r|--recursive))(?=[^|;&\n]*\s(?:-\w*f|--force))[^|;&\n]*(?<![<>])\s+"
        r"(?:/(?!tmp/(?![\w\-./]*\.\.)[\w\-][\w\-./]*(?:\s|$))(?:\s|$|\*|\w[^\s'\"]*)|~(?:\s|$|/)|\$\{?HOME\}?"
        r"|['\"]\s*(?:/(?!tmp/(?![\w\-./ ]*\.\.)[\w\-][\w\-./ ]*['\"](?:\s|$))[^'\"]*|~[^'\"]*|\$\{?HOME\}?)['\"])",
        re.IGNORECASE,
    ),
    re.compile(r"\bdd\b[^|;&]*of=/dev/(?:sd|nvme|hd)"),
    re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b\s+/dev/"),
    re.compile(r"\bshred\b\s+/dev/"),
    re.compile(r":\(\)\s*\{\s*:\|\s*:&\s*\}\s*;\s*:"),
    re.compile(r"\bchmod\b\s+(?:-R\s+)?(?:0?777|a\+rwx)\b\s+/(?:bin|etc|usr|sys|root)\b"),
    re.compile(r"\b(?:curl|wget|fetch)\b[^|]+\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh)\b", re.IGNORECASE),
    re.compile(r"\b(?:curl|wget|fetch)\b[^|]+\|\s*(?:sudo\s+)?(?:python[0-9.]*|perl|ruby|node|php)\s*(?:-\s*)?"
               r"(?:\||;|&|$)", re.IGNORECASE),
)

#: programs that never run their stdin or arguments as code — the only commands a data proof allows
#: Each was checked for an option that runs a program or evaluates an expression (review r5):
#:   cat, tee, echo, ls, mkdir, cd, pwd, true, false, :, touch, sleep, uniq, cut, nl, tac, rev, basename,
#:   dirname, realpath, readlink, chmod, df — none (options select output / modes / references only);
#:   wc, du — `--files0-from` READS a file of names, runs nothing; head, tail — `--pid` only watches a pid;
#:   grep, egrep, fgrep — `-f` reads patterns, `-P` is a regex dialect, nothing runs; stat — `--printf` is a
#:   format; file — `-m` reads magic files; diff, cmp — GNU diff has no program option (sdiff/diff3 do: not
#:   listed); md5sum, sha1sum, sha256sum — read only; date — `-f` reads dates, `-s` sets the clock (no code);
#:   printf — `-v NAME` assigns, and an array subscript in NAME is ARITHMETIC-evaluated (`a[$(…)]` runs):
#:   inert only without `-v` (`_NOT_INERT_OPT`); test, [ — `-v` / `-R` evaluate a subscript the same way: inert
#:   only without them. REMOVED: sort (`--compress-program=sh` runs a program).
_INERT_HEADS = frozenset({
    "cat", "tee", "echo", "printf", "ls", "mkdir", "cd", "pwd", "true", "false", ":", "wc", "head", "tail",
    "grep", "egrep", "fgrep", "stat", "file", "diff", "cmp", "touch", "test", "[", "sleep", "date",
    "uniq", "cut", "md5sum", "sha1sum", "sha256sum", "du", "df", "chmod", "basename", "dirname", "realpath",
    "readlink", "nl", "tac", "rev",
})
#: options that make an otherwise inert head evaluate code (any spelling that starts with them)
_NOT_INERT_OPT = {"printf": ("-v",), "test": ("-v", "-R"), "[": ("-v", "-R")}


def _inert(words) -> bool:
    """A command (its words) is inert: an inert head with none of that head's evaluating options."""
    if not words or words[0] not in _INERT_HEADS:
        return False
    bad = _NOT_INERT_OPT.get(words[0], ())
    return not any(w.lstrip("'\"").startswith(bad) for w in words[1:]) if bad else True
_LEX_META = set(" \t\n;&|()<>")


def _lex_shell(text: str):
    """bash's lexical view of `text`: (comment spans, quoted spans, heredoc bodies (start, end, quoted delimiter),
    commands as word lists), or None
    when a construct makes the view uncertain (substitution, arithmetic, a computed or unterminated piece)."""
    n, i = len(text), 0
    comments, quotes, bodies, heads, pending = [], [], [], [], []
    word_at, cmd_pos, redir = None, True, False

    def _end_word(end):
        nonlocal word_at, cmd_pos, redir
        if word_at is None:
            return
        w = text[word_at:end]
        word_at = None
        if redir:
            redir = False
        elif cmd_pos:
            heads.append([w])                           # an assignment `X=…` is a head too: never inert
            cmd_pos = False
        elif heads:
            heads[-1].append(w)                         # the command's arguments (its options are read)

    while i < n:
        c = text[i]
        if c == "\\":
            if text[i + 1:i + 2] == "\n":               # a continuation joins the lines: no word boundary
                i += 2
                continue
            if word_at is None:
                word_at = i
            i += 2
            continue
        if c == "'" or (c == "$" and text[i + 1:i + 2] == "'"):
            if word_at is None:
                word_at = i
            j = i + (2 if c == "$" else 1)
            while j < n and text[j] != "'":
                j += 2 if (c == "$" and text[j] == "\\") else 1
            if j >= n:
                return None
            quotes.append((i, j + 1))
            i = j + 1
            continue
        if c == '"':
            if word_at is None:
                word_at = i
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "`" or text[j:j + 2] == "$(":
                    return None
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                return None
            quotes.append((i, j + 1))
            i = j + 1
            continue
        if c == "`" or text[i:i + 2] in ("$(", "<(", ">(", "(("):
            return None
        if c == "#" and word_at is None:
            j = text.find("\n", i)
            j = n if j < 0 else j
            comments.append((i, j))
            i = j
            continue
        if c in _LEX_META:
            _end_word(i)
            if c == "\n":
                cmd_pos, redir = True, False
                k = i + 1
                for delim, dash, quoted in pending:     # bodies start after the line that opened them
                    start, found = k, False
                    while k < n:
                        nl = text.find("\n", k)
                        nl = n if nl < 0 else nl
                        line = text[k:nl]
                        if (line.lstrip("\t") if dash else line) == delim:
                            bodies.append((start, k, quoted))
                            k, found = nl + 1, True
                            break
                        k = nl + 1
                    if not found:
                        bodies.append((start, n, quoted))
                        k = n
                pending = []
                i = k
                continue
            if c in "<>":
                if text[i:i + 3] == "<<<":
                    redir, i = True, i + 3
                    continue
                if text[i:i + 2] == "<<":
                    j = i + 2
                    dash = text[j:j + 1] == "-"
                    j += dash
                    while j < n and text[j] in " \t":
                        j += 1
                    q = text[j:j + 1]
                    quoted = q in ("'", '"', "\\")              # a quoted delimiter: the body is not expanded
                    if q in ("'", '"'):
                        e = text.find(q, j + 1)
                        delim = text[j + 1:e] if e > 0 else ""
                        j = e + 1
                        if e < 0 or not re.fullmatch(r"[\w.-]+", delim):
                            return None
                    else:
                        m = re.match(r"\\?([A-Za-z_][\w.-]*)", text[j:])
                        if not m:
                            return None
                        delim, j = m.group(1), j + m.end()
                    if j < n and text[j] not in _LEX_META:
                        return None                     # `<<EOF$x`, `<<'a'b`: a delimiter bash builds otherwise
                    pending.append((delim, dash, quoted))
                    i = j
                    continue
                redir = True
                i += 1
                while i < n and text[i] in "<>&|":
                    i += 1
                continue
            if c in ";&|()":
                cmd_pos, redir = True, False
            i += 1
            continue
        if word_at is None:
            word_at = i
        i += 1
    _end_word(n)
    if pending:
        return None
    return comments, quotes, bodies, heads


def _data_spans(text: str) -> list:
    """The regions of `text` proven to be data (see above)."""
    lx = _lex_shell(text)
    if lx is None:
        return []
    comments, quotes, bodies, heads = lx
    spans = list(comments)
    if heads and all(_inert(cmd) for cmd in heads):
        spans += quotes
        # review r5: an UNQUOTED delimiter's body is expanded — `$(…)`, `${…}`, a backtick run — so it is data
        # only without any `$` or backtick
        spans += [(a, b) for a, b, quoted in bodies if quoted or not re.search(r"[$`]", text[a:b])]
    return spans


def _live_deny_hit(text: str):
    """The first shipped deny pattern with a match NOT inside a proven-data region of `text`, or None."""
    spans = None
    for pat in _LIVE_DENY:
        pos = 0
        while pos <= len(text):
            m = pat.search(text, pos)
            if not m:
                break
            if spans is None:
                spans = _data_spans(text)
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                return pat
            pos = m.start() + 1
    return None


_RM_DENY_REASON = ("deny-listed pattern: `rm -r -f` of an absolute or home path. Delete your own scratch "
                   "with a path RELATIVE to the workspace (`rm -rf tb_src`) or under /tmp (`rm -rf /tmp/x`)")


def validate_shell(cmd: str) -> Tuple[bool, str]:
    """Validate a shell command's shape and reject obviously-destructive
    forms. Returns ``(ok, reason)``.

    ``ok=True`` means "shape looks plausible, not on the deny list"
    — NOT "this command is safe in your environment". The host-level
    sandbox is still the authoritative safety boundary.
    """
    if not cmd or not cmd.strip():
        return False, "empty command"
    s = raw = cmd.strip()
    try:
        s = _ansi_c_unquote(s)                     # §4UY r1: `$'…'` is a quoted word — read as one, everywhere
    except Exception:  # noqa: BLE001
        pass
    # §4UY r4: the shipped deny list over the whole text, raw and decoded — only proven data is let through
    for text in dict.fromkeys((raw, s)):
        hit = _live_deny_hit(text)
        if hit is _LIVE_DENY[0]:
            return False, _RM_DENY_REASON
        if hit is not None:
            return False, f"deny-listed pattern: {hit.pattern[:60]}"
    # Shape check: must shlex-parse. Unclosed quotes are the most common
    # LLM emission bug ("echo 'hello world"). Heredoc bodies and comments are
    # not shell quoting (§4UY C8: `ls # don't` was refused).
    try:
        tokens = shlex.split(_strip_comments(_strip_heredoc_bodies(s)), posix=True)
    except ValueError as e:
        return False, f"shell syntax: {e}"
    if not tokens and not _strip_comments(s).strip():
        return False, "empty after parsing"
    try:
        if _rm_denied(s):
            return False, _RM_DENY_REASON          # ft live: the raw regex told the model nothing it could act on
        view = _code_view(s)
    except Exception as e:  # noqa: BLE001 — a crashing reader falls back to the whole text (fail closed)
        logger.debug("shell validator reader failed: %s", e)
        view = s
    for pat in _SHELL_DENY:
        if pat.search(view):
            return False, f"deny-listed pattern: {pat.pattern[:60]}"
    return True, ""


__all__ = ["validate_shell"]
