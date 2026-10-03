"""Shell command analysis for the sandbox guards (§4KW fourth/fifth review).

A command line is split OUTSIDE quotes into simple commands; comments,
heredoc bodies (data — unless a shell reads them) and huge quoted strings
are set aside; wrappers are stripped (`VAR=1`, `sudo -u x`, `timeout 5`,
`nice -n 5`, `exec`, `xargs -0`, `busybox`, `doas`, `do`/`then`, subshell
parentheses); a nested shell's `-c` payload (`-c`, `-lc`, `-ec`, `-c --`),
`eval`, `echo … | sh`, `$( … )` substitutions and interpreter one-liners
(`python -c "os.system(…)"`, `node -e "rmSync(…)"`) are analysed as commands
of their own; variables assigned in the command are substituted. Each simple
command's OWN verb is read — not a regex over the whole text (which took
`grep -rn 'rm -rf'` and `--rm` for removals and missed `bash -c "rm -rf
projects"`). Used by the execute guards (bulk removal, released workspaces,
the re-run heal) and by the lesson destruction screen.

This is the soft layer: the hard guarantee for a RELEASED workspace is the
user-immutable flag its release sets (memory/projects.py).
"""
import re

#: a quoted string longer than this is DATA (a base64 blob, a file body):
#: replaced before tokenising — `shlex` is quadratic on one long token
#: (fifth review: 2 MB took 22 s, on the event loop)
_MAX_QUOTED = 4096
_INTERPRETERS = {"python", "python3", "node", "nodejs", "perl", "ruby", "php", "deno", "bun"}


def _strip_heredocs(cmd: str):
    """(command without heredoc bodies, [(reader head, body), …])."""
    lines = str(cmd or "").split("\n")
    out, bodies, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = re.search(r"<<-?\s*(['\"]?)([A-Za-z_][\w-]*)\1", line)
        i += 1
        if not m:
            continue
        delim, body = m.group(2), []
        while i < len(lines) and lines[i].strip() != delim:
            body.append(lines[i])
            i += 1
        i += 1                                          # the delimiter line
        head = (line[:m.start()].strip().split() or [""])[0].lower().rsplit("/", 1)[-1]
        bodies.append((head, "\n".join(body)))
    return "\n".join(out), bodies


def _shrink_quotes(cmd: str) -> str:
    return re.sub(r"'[^']{%d,}'|\"(?:[^\"\\\\]|\\\\.){%d,}\"" % (_MAX_QUOTED, _MAX_QUOTED), "'<data>'", cmd)


def _split_shell(cmd: str) -> list:
    """Split a command line on `;`, `&&`, `||`, `|`, `&`, newlines and
    subshell parentheses OUTSIDE quotes. `2>&1`, `&>` and `>|` are
    redirects, not separators; a `#` that starts a word starts a comment.
    Returns the segments' texts; `_split_shell_sep` also gives the separator
    BEFORE each one."""
    return [t for t, _sep in _split_shell_sep(cmd)]


def _split_shell_sep(cmd: str) -> list:
    out, cur, q, i, sep = [], [], None, 0, ""
    s = str(cmd or "")
    depth = 0                                            # inside `$( … )`

    def _flush(next_sep):
        nonlocal cur, sep
        text = "".join(cur)
        if text.strip():
            out.append((text, sep))
        cur, sep = [], next_sep
    while i < len(s):
        c = s[i]
        if q:
            cur.append(c)
            if c == "\\" and q == '"' and i + 1 < len(s):
                cur.append(s[i + 1]); i += 2; continue
            if c == q:
                q = None
            i += 1
            continue
        if c in "'\"":
            q = c; cur.append(c); i += 1; continue
        if c == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2]); i += 2; continue
        if s.startswith("$(", i):
            depth += 1; cur.append("$("); i += 2; continue
        if depth:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            cur.append(c); i += 1
            continue
        if c == "#" and (not cur or cur[-1] in " \t"):
            while i < len(s) and s[i] != "\n":
                i += 1
            continue
        if c == "|" and cur and cur[-1] == ">":
            cur.append(c); i += 1; continue             # `>|` clobber redirect
        if c in "()":
            _flush(c); i += 1; continue
        if c in ";\n" or c == "|" or (c == "&" and not (i and s[i - 1] in "<>") and s[i + 1:i + 2] != ">"):
            two = s[i:i + 2]
            _flush(two if two in ("&&", "||") else c)
            i += 2 if two in ("&&", "||") else 1
            continue
        cur.append(c); i += 1
    _flush("")
    return out


#: wrappers that run the next word as the command, and the options of each
#: that take a value
_WRAPPERS = {"sudo": {"-u", "-g", "-C", "-h", "-p", "-U"}, "doas": {"-u", "-C"}, "env": {"-u", "-C", "-S"},
             "nohup": set(), "time": set(), "command": set(), "exec": set(), "builtin": set(), "nice": {"-n"},
             "ionice": {"-c", "-n", "-p"}, "stdbuf": {"-i", "-o", "-e"}, "timeout": {"-s", "-k", "--signal", "--kill-after"},
             "xargs": {"-I", "-i", "-n", "-P", "-L", "-l", "-d", "-E", "-e", "-s", "-a", "--max-args", "--max-procs",
                       "--delimiter", "--arg-file", "--replace", "--max-lines"},
             "busybox": set(), "do": set(), "then": set(), "else": set(), "{": set(), "!": set(), "}": set()}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "ash"}
_UNKNOWN = "<unknown>"


def _subst_vars(tok: str, env: dict) -> str:
    def _rep(m):
        name = m.group(1) or m.group(3)
        if name in env:
            return env[name]
        if m.group(2) is not None:
            return m.group(2)                            # ${NAME:-default}
        return m.group(0)
    return re.sub(r"\$\{(\w+)(?::?-([^}]*))?\}|\$(\w+)", _rep, tok)


def _resolve_substitutions(seg: str, extra: list, depth: int) -> str:
    """`$( … )` / backticks: their commands are analysed too; a substitution
    that only prints its arguments (`echo X`, `printf X`, `ls -d X`) is
    replaced by them, any other by an unknown marker."""
    def _one(inner: str) -> str:
        extra.extend(_shell_segments(inner, depth + 1))
        m = re.match(r"\s*(?:echo|printf|ls(?:\s+-\w+)*)\s+(.+?)\s*$", inner, re.S)
        return m.group(1) if m else _UNKNOWN
    out, i = [], 0
    while i < len(seg):
        if seg.startswith("$(", i) and not seg.startswith("$((", i):
            d, j = 1, i + 2
            while j < len(seg) and d:
                d += {"(": 1, ")": -1}.get(seg[j], 0)
                j += 1
            out.append(_one(seg[i + 2:j - 1]))
            i = j
            continue
        if seg[i] == "`":
            j = seg.find("`", i + 1)
            if j > i:
                out.append(_one(seg[i + 1:j]))
                i = j + 1
                continue
        out.append(seg[i])
        i += 1
    return "".join(out)


def _shell_segments(cmd: str, _depth: int = 0) -> list:
    """Each simple command as ``(head, args, raw, via_xargs)`` — see the
    module docstring for what is stripped and expanded."""
    import shlex
    if _depth > 3:
        # nesting past the limit: fail CLOSED when it could remove anything
        if re.search(r"\b(?:rm|rmdir|unlink|shred|truncate|mv|find|rsync|rmtree|rmSync|git|tar|zip)\b", str(cmd or "")):
            return [("rm", ["-rf", _UNKNOWN], str(cmd)[:200], False)]
        return []
    text, bodies = _strip_heredocs(_shrink_quotes(str(cmd or "")))
    out = []
    # code quoted as text (a lesson's fix, an inline snippet): the
    # parenthesis split below would cut `shutil.rmtree('projects')` apart
    _bare = re.sub(r"(?:python3?|node|nodejs|perl|ruby|php)\s+-[ceEr]\s+(?:'[^']*'|\"(?:[^\"\\]|\\.)*\")", " ", text)
    if re.search(r"\b(?:rmtree|rmSync|rmdirSync|os\.system|subprocess\.|execSync|fs\.rm)\b", _bare):
        out.extend(_interpreter_segments(_bare, _depth))
    env: dict = {}
    prev = None                                          # (head, args) of the segment before
    for seg, sep in _split_shell_sep(text):
        extra: list = []
        if "$(" in seg or "`" in seg:
            seg = _resolve_substitutions(seg, extra, _depth)
        out.extend(extra)
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        via_xargs = False
        assigns = {}
        while toks:
            am = re.match(r"^([A-Za-z_]\w*)=(.*)$", toks[0], re.S)
            if am:
                assigns[am.group(1)] = _subst_vars(am.group(2), env)
                toks = toks[1:]
                continue
            w = toks[0].lower().rsplit("/", 1)[-1]
            if w not in _WRAPPERS or (w == "time" and len(toks) == 1):
                break
            via_xargs = via_xargs or w == "xargs"
            valued = _WRAPPERS[w]
            toks = toks[1:]
            while toks and (toks[0].startswith("-") or (w == "env" and "=" in toks[0])):
                opt = toks[0]
                toks = toks[1:]
                if opt in valued and toks:
                    toks = toks[1:]
            if w == "timeout" and toks and re.fullmatch(r"[\d.]+[smhd]?", toks[0]):
                toks = toks[1:]
            if w == "nice" and toks and re.fullmatch(r"-?\d+", toks[0]):
                toks = toks[1:]
        if not toks:
            env.update(assigns)                          # `D=projects; rm -rf $D`
            continue
        local_env = dict(env, **assigns)
        toks = [_subst_vars(t, local_env) for t in toks]
        head = toks[0].lower().rsplit("/", 1)[-1]
        args = toks[1:]
        if head in _SHELLS:
            # `-c`, `-lc`, `-ec`, `-c --`: the payload is the first plain word after
            k = next((j for j, a in enumerate(args) if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", a)), None)
            if k is not None:
                rest = [a for a in args[k + 1:] if a != "--"]
                if rest:
                    out.extend(_shell_segments(rest[0], _depth + 1))
                    prev = (head, args)
                    continue
            if sep == "|" and prev and prev[0] in ("echo", "printf") and not [a for a in args if not a.startswith("-")]:
                out.extend(_shell_segments(" ".join(a for a in prev[1] if not a.startswith("-")), _depth + 1))
        if head == "eval" and args:
            out.extend(_shell_segments(" ".join(args), _depth + 1))
            prev = (head, args)
            continue
        if head in _INTERPRETERS:
            code = next((args[j + 1] for j, a in enumerate(args)
                         if a in ("-c", "-e", "--eval", "-E", "-r") and j + 1 < len(args)), "")
            out.extend(_interpreter_segments(code, _depth))

        out.append((head, args, seg, via_xargs))
        prev = (head, args)
    for reader, body in bodies:
        if reader in _SHELLS:
            out.extend(_shell_segments(body, _depth + 1))
        elif reader in _INTERPRETERS:
            out.extend(_interpreter_segments(body, _depth))
    return out


_LIT = r"""(?:'([^']*)'|"([^"]*)"|`([^`]*)`)"""


def _interpreter_segments(code: str, depth: int) -> list:
    """Shell commands inside interpreter code: strings handed to os.system /
    subprocess / child_process, and file-removal APIs (rmtree, rmSync,
    fs.rm, os.remove …) as `rm -rf` of their literal argument — or of an
    unknown target when the argument is computed (`os.listdir('.')`)."""
    if not code or depth > 3:
        return []
    out = []
    for m in re.finditer(r"(?:system|popen|execSync|exec|spawnSync|check_output|check_call|call|run|Popen)\(\s*"
                         + _LIT, code):
        out.extend(_shell_segments(next(g for g in m.groups() if g is not None), depth + 1))
    for m in re.finditer(r"(?:run|call|check_call|check_output|Popen)\(\s*\[([^\]]*)\]", code):
        words = re.findall(_LIT, m.group(1))
        out.extend(_shell_segments(" ".join(next(g for g in w if g is not None) for w in words), depth + 1))
    for m in re.finditer(r"\b(?:rmtree|rmSync|rmdirSync|unlinkSync|removedirs|remove|unlink|rmdir)\s*\(\s*(" + _LIT + r")?",
                         code):
        arg = next((g for g in m.groups()[1:] if g is not None), None)
        if re.search(r"\bfs\.rm\b|rmtree|rmSync|removedirs", m.group(0)) or arg is not None:
            out.append(("rm", ["-rf", arg if arg is not None else _UNKNOWN], m.group(0), False))
    for m in re.finditer(r"\bfs\.(?:promises\.)?rm\(\s*(" + _LIT + r")?", code):
        arg = next((g for g in m.groups()[1:] if g is not None), None)
        out.append(("rm", ["-rf", arg if arg is not None else _UNKNOWN], m.group(0), False))
    return out


def _norm_shell_arg(arg: str) -> str:
    """A shell argument as the path it names: quotes stripped, `$PWD`,
    `${PWD}`, `$(pwd)` and backtick-pwd read as `.`, `//`, `/./` and `x/..`
    collapsed, lower-cased (APFS)."""
    import posixpath
    a = str(arg).strip().strip("'\"")
    a = re.sub(r"\$\{?PWD\}?|\$\(\s*pwd\s*\)|`\s*pwd\s*`", ".", a, flags=re.IGNORECASE)
    a = re.sub(r"/{2,}", "/", a)
    if a and a not in (".", "..") and not a.startswith("..") and ".." in a.split("/"):
        a = posixpath.normpath(a)                       # `projects/<id>/..` is `projects`
    a = re.sub(r"/(?:\./)+", "/", a)
    if a.startswith("./") and len(a) > 2:
        a = a[2:]
    return a.rstrip("/").lower() or "/"


def _expand_braces(arg: str) -> list:
    m = re.match(r"^(.*?)\{([^{}]*,[^{}]*)\}(.*)$", arg)
    if not m:
        return [arg]
    return [x for part in m.group(2).split(",") for x in _expand_braces(m.group(1) + part + m.group(3))]


_GLOB_CHARS = "*?["


def _names_id(path: str, rid: str) -> bool:
    """`rid` is a whole path component of `path` (fifth review: a substring
    match refused `rm /tmp/<id>-backup.tgz`)."""
    return bool(re.search(r"(?:^|/)" + re.escape(rid) + r"(?:/|$)", path))


def _bulk_target(arg: str, released_ids) -> bool:
    """Does this (normalised) argument name the projects folder, a glob over
    it, the workspace or its root, `..`, or a released project?"""
    import fnmatch
    a = _norm_shell_arg(arg)
    if a in ("/", "/workspace", "/workspace/*", "workspace", "workspace/*", "/*",
             "projects", "/workspace/projects", "projects/*", "/workspace/projects/*", "projects/."):
        return True
    if a == ".." or a.startswith(("../", "/workspace/..")):
        return True
    m = re.match(r"^(?:/workspace/)?projects/(.+)$", a)
    if m and any(ch in m.group(1) for ch in _GLOB_CHARS):
        return True                                   # a glob over the project ids
    # a glob at the workspace root that matches the projects folder (`proj*`)
    base = a[len("/workspace/"):] if a.startswith("/workspace/") else a
    if "/" not in base and any(ch in base for ch in _GLOB_CHARS) and fnmatch.fnmatch("projects", base):
        return True
    return any(_names_id(a, rid) for rid in released_ids)


_RM_HEADS = {"rm", "rmdir", "unlink", "shred", "truncate", "srm"}
_FIND_FILTERS = ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-wholename", "-iwholename")
_FIND_ACTIONS = ("-delete", "-exec", "-execdir", "-ok", "-okdir")


def _is_recursive(opts) -> bool:
    return any(re.match(r"^-\w*[rR]|^--recursive", o) for o in opts)


def _find_removes(expr: list) -> bool:
    return "-delete" in expr or any(
        e in ("-exec", "-execdir", "-ok", "-okdir") and k + 1 < len(expr)
        and expr[k + 1].lower().rsplit("/", 1)[-1] in _RM_HEADS | {"mv"}
        for k, e in enumerate(expr))


def _destructive_targets(head: str, args: list, via_xargs: bool):
    """The paths a simple command REMOVES or moves away, or None when it
    removes nothing. An unknown target is ``"<unknown>"``."""
    opts = [a for a in args if a.startswith("-")]
    plain = [a for a in args if not a.startswith("-")]
    if head in _RM_HEADS:
        if head == "truncate":
            plain = [a for i, a in enumerate(args) if not a.startswith("-")
                     and not (i and args[i - 1] in ("-s", "--size", "-r", "--reference"))]
        if via_xargs:
            # targets come from stdin; a recursive one is unbounded
            return [_UNKNOWN] if _is_recursive(opts) else []
        if _UNKNOWN in plain and not _is_recursive(opts):
            plain = [a for a in plain if a != _UNKNOWN]
        return plain
    if head == "mv":
        if via_xargs:
            return []
        if "-t" in args:
            k = args.index("-t")
            dest = args[k + 1] if k + 1 < len(args) else None
            return [a for a in plain if a != dest]
        if any(o.startswith("--target-directory") for o in opts):
            return plain
        return plain[:-1]
    if head == "rsync" and any(o.startswith("--delete") or o == "--remove-source-files" for o in opts):
        return plain[-1:] if "--remove-source-files" not in opts else plain
    if head == "zip" and any(re.match(r"^-\w*m", o) for o in opts):
        return plain[1:]                                 # `zip -m` moves its inputs into the archive
    if head == "find":
        expr_at = next((i for i, a in enumerate(args) if a.startswith("-") or a in ("(", "!")), len(args))
        expr = args[expr_at:]
        if not _find_removes(expr):
            return None
        starts = args[:expr_at] or ["."]
        # a name/path filter that is not a bare wildcard narrows it — only a
        # POSITIVE filter placed BEFORE the action (fifth review: `-not
        # -name x`, `! -name x` and `-delete -name x` counted as narrowing)
        act = next((k for k, e in enumerate(expr) if e in _FIND_ACTIONS), len(expr))
        narrowing = [expr[k + 1] for k, e in enumerate(expr[:act]) if k + 1 < act and e in _FIND_FILTERS
                     and not (k and expr[k - 1] in ("-not", "!"))]
        if narrowing and any(p.strip("'\"") not in ("*", "*.*", ".*", "") for p in narrowing):
            return []
        # `-maxdepth 1` sweeps the start folder's contents: judged as the
        # folder itself, the same verdict (fifth review: the old `<entry>`
        # marker made `find projects -maxdepth 1 -exec rm -rf {} +` pass)
        return starts
    if head == "git":
        cdir = args[args.index("-C") + 1] if "-C" in args and args.index("-C") + 1 < len(args) else None
        sub = [a for a in plain if a != cdir]
        if sub[:1] == ["clean"] and any(re.match(r"^-\w*f", o) or o == "--force" for o in opts):
            return [cdir or "."]
        return None
    if head == "tar" and "--remove-files" in opts:
        return [a for a in plain if not a.endswith((".tar", ".tgz", ".gz", ".bz2", ".xz", ".zst"))]
    return None


class _Cwd:
    """Where a command runs: ``kind`` ∈ root | bulk (the projects folder) |
    project | released | elsewhere | unknown; ``pid`` the project; ``depth``
    how far below the project's own folder."""
    __slots__ = ("kind", "pid", "depth")

    def __init__(self, workdir: str, rel):
        m = re.search(r"/projects/([0-9a-f]{12})((?:/[^/]+)*)/?$", str(workdir or ""), re.IGNORECASE)
        if m:
            self.pid = m.group(1).lower()
            self.kind = "released" if self.pid in rel else "project"
            self.depth = len([x for x in m.group(2).split("/") if x])
        else:
            self.kind, self.pid, self.depth = "root", None, 0

    def cd(self, args, head, rel):
        tgt = next((a for a in args if not a.startswith("-") or a == "-"), None)
        if tgt is None or tgt in ("-", "~") or tgt.startswith("~") or head == "popd" or "$" in tgt or _UNKNOWN in tgt:
            self.kind, self.pid, self.depth = "unknown", None, 0
            return
        n = _norm_shell_arg(tgt)
        if n.startswith("/"):
            if n == "/workspace" or n.startswith("/workspace/"):
                self.kind, self.pid, self.depth = "root", None, 0
                n = n[len("/workspace"):].lstrip("/")
            else:
                self.kind, self.pid, self.depth = ("root" if n == "/" else "elsewhere"), None, 0
                return
        for part in [x for x in n.split("/") if x not in ("", ".")]:
            self._step(part, rel)

    def _step(self, part, rel):
        if part == "..":
            if self.kind in ("project", "released"):
                if self.depth:
                    self.depth -= 1
                else:
                    self.kind, self.pid = "bulk", None
            elif self.kind == "bulk":
                self.kind = "root"
            else:
                self.kind = "unknown"
        elif self.kind == "root":
            self.kind = "bulk" if part == "projects" else "elsewhere"
        elif self.kind == "bulk":
            if re.fullmatch(r"[0-9a-f]{12}", part):
                self.pid, self.depth = part, 0
                self.kind = "released" if part in rel else "project"
            else:
                self.kind = "elsewhere"
        elif self.kind in ("project", "released"):
            self.depth += 1

    def resolves_inside_project(self, n: str) -> bool:
        """A relative path that stays strictly inside the current project's
        folder (`../build` from `<project>/src` does; `..` from its root
        does not)."""
        if self.kind not in ("project", "released"):
            return False
        parts = [x for x in n.split("/") if x not in ("", ".")]
        ups = 0
        while ups < len(parts) and parts[ups] == "..":
            ups += 1
        rest = parts[ups:]
        return ups < self.depth or (ups == self.depth and bool(rest)) if ups else True


def _bulk_destructive(command: str, released_ids, workdir: str = "") -> bool:
    """A command that removes or moves files IN BULK — the projects folder, a
    glob over it, the workspace, `.`/`*` where the directory is the workspace
    root or the projects folder, or a released project — anywhere in the
    command (including nested shells, `eval`, pipes into a shell, `$( … )`
    and interpreter one-liners). The current directory is tracked through
    `cd`/`pushd`, including depth inside a project: `.`/`*` are refused at
    the workspace root or a bulk/unknown location, allowed in a project's own
    folder or any other directory. A real name filter narrows a find."""
    import fnmatch
    rel = [str(r).lower() for r in released_ids or [] if r]
    cwd = _Cwd(workdir, rel)
    loops = {}
    for head, args, _raw, via_xargs in _shell_segments(command):
        if head == "for" and len(args) >= 2 and args[1] == "in":
            loops[args[0]] = args[2:]
            continue
        if head in ("cd", "pushd", "popd"):
            cwd.cd(args, head, rel)
            continue
        tg = _destructive_targets(head, args, via_xargs)
        if not tg:
            continue
        expanded = []
        for t in tg:
            v = re.fullmatch(r"\$\{?(\w+)\}?(.*)", t)
            if v and v.group(1) in loops:
                expanded += [x + v.group(2) for x in loops[v.group(1)]]
            else:
                expanded += _expand_braces(t)
        for t in expanded:
            n = _norm_shell_arg(t)
            # an unresolved variable or substitution standing for the whole
            # target (`rm -rf $D`, `"$P"/*`) is unknown
            unknown = t == _UNKNOWN or bool(re.fullmatch(r"\$\{?\w+\}?(?:/\*?)?|<unknown>(?:/\*?)?", n))
            if unknown:
                if cwd.kind in ("root", "bulk", "unknown", "released") or n.endswith("*"):
                    return True
                continue
            relative = not n.startswith(("/", "~"))
            if cwd.kind == "released" and relative and cwd.resolves_inside_project(n):
                return True
            local = n in (".", "*", "./*") or ("/" not in n and any(c in n for c in _GLOB_CHARS))
            if cwd.kind == "bulk" and relative and (local or re.fullmatch(r"[0-9a-f]{12}", n) or n == ".."
                                                    or any(_names_id(n, r) for r in rel)):
                return True
            if local:
                if cwd.kind in ("root", "unknown") and (n in (".", "*", "./*") or fnmatch.fnmatch("projects", n)):
                    return True
                continue
            if relative and n.startswith("..") and cwd.resolves_inside_project(n):
                continue                                 # `cd src && rm -rf ../build` stays in the project
            if cwd.kind in ("root", "bulk", "unknown") or not relative or n.startswith(".."):
                if _bulk_target(t, rel):
                    return True
            elif any(_names_id(n, r) for r in rel):
                return True
    return False


_GIT_WRITES = {"checkout", "reset", "clean", "apply", "pull", "merge", "rm", "mv", "restore", "stash",
               "switch", "rebase", "cherry-pick", "revert", "am", "init", "clone", "commit", "add"}
#: tools that write into their working directory (dependencies, builds)
_CWD_WRITERS = {"npm", "pnpm", "yarn", "pip", "pip3", "make", "cargo", "go", "poetry", "uv", "bundle", "composer"}


def _tar_archive(args: list):
    """The archive a `tar` invocation names (`-f X`, `-czf X`, `--file=X`)."""
    for i, a in enumerate(args):
        if a.startswith("--file="):
            return a.split("=", 1)[1]
        if a == "--file" and i + 1 < len(args):
            return args[i + 1]
        if (i == 0 or a.startswith("-")) and not a.startswith("--") and "f" in a.lstrip("-") and i + 1 < len(args):
            return args[i + 1]
    return None


def _written_paths(head: str, args: list, raw: str) -> list:
    """The paths one simple command WRITES or removes (`.` = its directory).
    Sources of a copy are read, not written (fourth review); `find -exec
    grep`, `tar -czf /tmp/x.tgz .`, a `>` inside `$(( ))` or a heredoc body
    are no writes (fifth review)."""
    unq = re.sub(r"'[^']*'|\"[^\"]*\"|\$\(\([^)]*\)\)", "''", raw)
    out = [m.group(1) for m in re.finditer(r"(?:\d?>>?\|?|&>>?)\s*(?!&)([^\s;|&<>()]+)", unq)
           if m.group(1) != "/dev/null"]
    plain = [a for a in args if not a.startswith("-")]
    if head in ("cp", "install", "ln", "rsync", "scp"):
        if "-t" in args and args.index("-t") + 1 < len(args):
            out.append(args[args.index("-t") + 1])
        else:
            out += plain[-1:]
        if head == "rsync" and "--remove-source-files" in args:
            out += plain[:-1]
    elif head == "mv" or head in _RM_HEADS or head in ("touch", "mkdir", "tee", "patch"):
        out += [a for a in plain if a != _UNKNOWN]
    elif head in ("chmod", "chown", "chgrp"):
        out += plain[1:]
    elif head in ("sed", "perl") and any(a.startswith("-i") or a.startswith("--in-place") for a in args):
        out += plain[1:]
    elif head == "dd":
        out += [a[3:] for a in args if a.startswith("of=")]
    elif head == "unzip":
        out.append(args[args.index("-d") + 1] if "-d" in args and args.index("-d") + 1 < len(args) else ".")
    elif head == "zip" and plain:
        out.append(plain[0])                             # the archive
        if any(re.match(r"^-\w*m", a) for a in args if a.startswith("-")):
            out += plain[1:]
    elif head == "tar" and args:
        flags = args[0].lstrip("-") if not args[0].startswith("--") else ""
        archive = _tar_archive(args)
        if "x" in flags or "--extract" in args or "--get" in args:
            out.append(args[args.index("-C") + 1] if "-C" in args and args.index("-C") + 1 < len(args) else ".")
        elif "c" in flags or "--create" in args:
            if archive:
                out.append(archive)
            if "--remove-files" in args:
                out += [a for a in plain if a != archive]
    elif head == "git":
        cdir = args[args.index("-C") + 1] if "-C" in args and args.index("-C") + 1 < len(args) else None
        sub = [a for a in plain if a != cdir]
        if sub and sub[0] in _GIT_WRITES:
            out.append(cdir or ".")
    elif head == "find":
        expr_at = next((i for i, a in enumerate(args) if a.startswith("-") or a in ("(", "!")), len(args))
        expr = args[expr_at:]
        writes = _find_removes(expr) or any(
            e in ("-exec", "-execdir") and k + 2 < len(expr) and expr[k + 1] in ("sed", "perl")
            and expr[k + 2].startswith("-i") for k, e in enumerate(expr))
        if writes:
            out += args[:expr_at] or ["."]
    elif head in _CWD_WRITERS:
        out.append(".")
    return out


def _released_written_ids(command: str, ids, workdir: str = "") -> set:
    """Project ids among ``ids`` whose workspace a command WRITES into: a
    written path naming `projects/<id>`, or a relative path while the
    current directory (the working directory, then each `cd`, tracked like
    the bulk guard does) is inside that project's workspace."""
    rel = [str(i).lower() for i in ids or []]
    cwd = _Cwd(workdir, rel)
    hit = set()
    for head, args, raw, _x in _shell_segments(command):
        if head in ("cd", "pushd", "popd"):
            cwd.cd(args, head, rel)
            continue
        for w in _written_paths(head, args, raw):
            pm = re.search(r"projects/+(?:\./)*([0-9a-fA-F]{12})(?:/|$)", w)
            if pm and pm.group(1).lower() in rel:
                hit.add(pm.group(1).lower())
            elif cwd.kind == "released" and not w.startswith(("/", "~")) and cwd.resolves_inside_project(_norm_shell_arg(w)):
                hit.add(cwd.pid)
            elif cwd.kind == "bulk" and not w.startswith(("/", "~")):
                first = _norm_shell_arg(w).split("/")[0]
                if first in rel:
                    hit.add(first)
    return hit
