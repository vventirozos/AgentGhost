"""§4UL (decision 1A) — a project task runs as a FULL main-loop agent turn.

Until §4UL an autopilot or owner-requested batch built each coding task with
`coding_executor.build_coding_task` (the one-call "spec" executor, or the
agentic leaf with four tools), and a research task with one web search. The
owner's own chat turns did the same work with every tool, the project
briefing and the honest-close gates of `manage_projects task_update` — and
did it better. So a batch now runs each task as ONE fresh agent request:

* its own request id ``sub-ptask-<task>-<hex>``: a ``sub-`` id is background
  for `request_kind`, internal for `is_internal_request` and NOT the user for
  memory's deletion walls (`tools/memory._not_the_user`), so no turn of a batch
  can forget or reset the owner's memory;
* the project PINNED (the briefing reaches the prompt), workspace writes
  confined to it, owner memory read-only (`coding_loop.build_leaf_context`);
* authority from the batch's origin: the owner's own run carries ``owner``
  (a kept lab may be deployed — the owner asked for the work); an unattended
  run carries ``unattended`` (kept machines refused by the lab tool and the
  host). A task that needs a kept machine then WAITS for the owner's run (a
  hold, never a question);
* idle autopilot starts a turn only while no user request is active
  (``foreground_requests == 0``) and its every model call yields to one
  (`coding_loop._BackgroundOnlyLLM`);
* the turn may not steer the project itself: autoadvance / autopilot /
  switch / create / delete / … are refused inside it (one task, this task).

The turn closes its task with ``manage_projects task_update`` (DONE with its
deliverables, FAILED with a reason). The advancer reads the task's status
afterwards. The old executor path stays as the fallback, used only when this
one cannot be built (logged).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import time
from typing import Any, Dict, Optional

from ..utils.helpers import env_positive
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

TASK_TURN_RID_PREFIX = "sub-ptask-"
TASK_TURN_LABEL = "ptask"
#: a task turn's own cap; the unattended step's cap and the owner request's time left bound it further
TASK_TURN_TIMEOUT_S = env_positive("GHOST_TASK_TURN_TIMEOUT_S", 1800.0)
TASK_TURN_MAX_TURNS = int(env_positive("GHOST_TASK_TURN_MAX_TURNS", 40.0))
#: below this much time a turn is not started (the task goes back to READY, out of time)
TASK_TURN_MIN_S = 120.0
#: what a task's turn leaves to the owner's own turns: steering the project as a whole
REFUSED_PROJECT_ACTIONS = frozenset({
    "autoadvance", "autopilot", "switch", "create", "delete", "archive", "resume", "exit", "budget",
    "re" + "plan",        # (one module owns the bare "replan" constant: test_4tb_replan_removed)
    "create_version", "release", "unrelease", "promote_from_context", "task_delete",
    # §4UL review M4: the project's record, its constraints, its cross-project links and its workspace hygiene
    # are the owner's turns' business too
    "update", "constraint_retire", "clone", "cleanup", "set_dependency",
})
#: §4UL review M4: the task-level actions a turn may aim only at ITS task (or a subtask under it)
_OWN_TASK_ACTIONS = frozenset({"task_update"})
#: §4UL re-review m1: the actions that ADD tasks may aim only at the turn's task (a subtask under it; `task_add`
#: may also add a top-level follow-up) — another task's parts and the project's plan are the owner's turns
_OWN_PARENT_ACTIONS = {"task_add": "parent_id", "task_decompose": "task_id"}
#: what a turn may do on ANOTHER project: read it
_READ_ACTIONS = frozenset({"get", "status", "list", "task_list", "task_next", "artifact_list", "event_log",
                           "ledger", "describe_file", "file_history", "search", "research_list",
                           "verify_release"})
#: §4UL re-review M1: what a turn may do on ANOTHER project — the pure reads (`ledger` / `describe_file` only in
#: their read forms; `research` writes a brief into that project's workspace, so it is not a read)
_WRITE_FORM_ARGS = {"ledger": ("ledger",), "describe_file": ("description",)}
#: the lab / run tools' refusal of a kept (production) machine to unattended work
KEPT_REFUSAL_MARKS = ("not available to unattended work", "unattended work reads a production machine")
_CLOSED = frozenset({"DONE", "FAILED", "NEEDS_USER", "PAUSED", "BLOCKED"})
#: §4UL round 5 M1: a status the turn sets AFTER a refused DONE that ends that DONE's claim (the task is not left
#: waiting on a check any more); a non-terminal one (IN_PROGRESS, PENDING) keeps the hold
_DROPS_HOLD = frozenset({"FAILED", "BLOCKED", "NEEDS_USER", "PAUSED", "CANCELLED"})
#: the turn's OWN close — what makes the task's end status the turn's (round 5 MINOR)
_OWN_CLOSE = frozenset({"DONE", "FAILED"})


def _norm_status(value) -> str:
    """A task_update status the way the handler reads it (`WAITING` → NEEDS_USER, `RUNNING` → IN_PROGRESS)."""
    v = str(value or "").strip().upper()
    return {"WAITING": "NEEDS_USER", "RUNNING": "IN_PROGRESS"}.get(v, v)


def enabled() -> bool:
    """GHOST_PROJECT_TASK_TURNS (default ON); the test suite pins it OFF so
    the executor tests keep the path they were written for."""
    return str(os.environ.get("GHOST_PROJECT_TASK_TURNS", "1")).strip().lower() not in ("0", "false", "no", "off")


def mark_real_executor(fn, *, idle: bool = False):
    """Mark ``fn`` as a production build executor (the advancer's wired one):
    only then does a task run as a turn — a test's stub executor keeps its own
    path. ``idle``: the idle loop's (its turns wait for no user request)."""
    try:
        fn._ghost_real_executor = True
        fn._ghost_idle_executor = bool(idle)
    except Exception:  # noqa: BLE001
        pass
    return fn


def wants_task_turn(context, coding_executor) -> bool:
    """Run this task as a turn: switched on, a real executor wired (the
    caller is production, not a unit stub), a model and a store present."""
    return bool(enabled() and coding_executor is not None
                and getattr(coding_executor, "_ghost_real_executor", False) is True
                and getattr(context, "llm_client", None) is not None
                and getattr(context, "project_store", None) is not None)


def _origin_for(owner_requested: bool) -> str:
    from .request_scope import ORIGIN_OWNER, ORIGIN_PROBE, ORIGIN_UNATTENDED
    try:
        from ..utils.logging import request_kind
        if request_kind() == "probe":
            return ORIGIN_PROBE
    except Exception:  # noqa: BLE001
        pass
    return ORIGIN_OWNER if owner_requested else ORIGIN_UNATTENDED


def _foreground_busy(context) -> bool:
    llm = getattr(context, "llm_client", None)
    try:
        return int(getattr(llm, "foreground_requests", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


# ── §4UL re-review M1 (round 3: simplified) — the CHECK an unattended DONE needs, and the WRITE that voids it ───
# The gate guards an honest model's OMISSION (it built and never ran anything), never an adversary: a check is
# any allow-listed program that RAN after the last write — its exit status, `|| true`, `; echo $?`, `| tail`
# and loop / if bodies do not matter. Looking (`ls`, `cat`, `echo`) and installing are not checks.
_PY_RE = re.compile(r"(?:python|pypy)[0-9.]*")
#: interpreters: any run (a script, inline code, stdin) — but `perl -pi` / `ruby -i` edit files in place
_INTERPRETERS = frozenset({"node", "nodejs", "deno", "ruby", "php", "perl", "lua", "julia", "Rscript", "tsx",
                           "ts-node", "java", "elixir", "swift", "kotlin", "groovy", "scala", "dotnet-script"})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
#: test runners, build tools, compilers, database clients, linters and type checkers: any run
_RUNNERS = frozenset({"pytest", "py.test", "tox", "nox", "make", "gmake", "ninja", "just", "jest", "vitest",
                      "mocha", "phpunit", "rspec", "gradle", "gradlew", "mvn", "javac", "tsc", "gcc", "g++",
                      "cc", "c++", "clang", "clang++", "rustc", "kotlinc", "ctest", "psql", "sqlite3", "mysql",
                      "mariadb", "mongosh", "redis-cli",
                      "mypy", "ruff", "flake8", "pylint", "pyflakes", "pycodestyle", "pyright", "bandit",
                      "eslint", "stylelint", "htmlhint", "shellcheck", "hadolint", "yamllint", "jsonlint",
                      "xmllint", "markdownlint", "golangci-lint", "staticcheck"})
#: `<tool> <sub>`: the subcommands that run, build or check (`npm test`, `go test`, `cargo run`, `dotnet test`)
_SUBCOMMAND_RUNS = {
    "npm": {"test", "t", "tst", "run", "run-script", "rum", "urn", "start", "restart"},
    "go": {"test", "run", "build", "vet"},
    "cargo": {"test", "run", "build", "check", "bench", "clippy"},
    "dotnet": {"test", "run", "build"},
    "deno": {"test", "run", "task", "eval", "check", "lint"},
    "bun": {"test", "run"},
    "cmake": {"--build"},
    "swift": {"test", "run", "build"},
    "playwright": {"test"},
}
#: `yarn` / `pnpm`: everything but these runs a script (`yarn test`, `pnpm build`)
_PKG_NOT_RUNS = frozenset({"install", "i", "add", "remove", "rm", "uninstall", "un", "ci", "init", "create",
                           "link", "unlink", "publish", "login", "logout", "config", "set", "get", "cache",
                           "audit", "outdated", "list", "ls", "info", "view", "why", "upgrade", "up", "update",
                           "import", "pack", "store", "env", "help", "version", "setup", "prune"})
#: runs the rest as a command: `uv run pytest`, `npx eslint .`, `npm exec jest`, `docker exec web pytest`
_RUN_WRAPPERS = frozenset({"uv", "poetry", "pipenv", "pdm", "hatch", "rye", "conda", "mamba", "micromamba"})
_NPX = frozenset({"npx", "pnpx", "bunx"})
_PY_NOT_CHECK_MODULES = frozenset({"pip", "pip3", "venv", "virtualenv", "ensurepip", "http.server",
                                   "SimpleHTTPServer", "pydoc", "site", "this", "playwright"})
_INFO_ARGS = frozenset({"--version", "-V", "-v", "-version", "version", "--help", "-h", "help", "-help"})
_HTTP_HEADS = frozenset({"curl", "wget", "http", "https", "xh"})
_LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1", "[::1]", "host.docker.internal"})
#: formatters rewrite files (a WRITE) — unless asked only to check
_FORMATTERS = frozenset({"black", "isort", "autopep8", "yapf", "prettier", "gofmt", "rustfmt", "clang-format",
                         "autoflake"})
_FORMAT_CHECK_FLAGS = frozenset({"--check", "--diff", "--check-only", "-l", "--list-different", "-d"})
#: commands that change files without a write target `_shell_write_targets` can name
_MUTATING_HEADS = frozenset({"rm", "rmdir", "unlink", "patch", "shred"})
_GIT_MUTATING = frozenset({"apply", "checkout", "restore", "reset", "stash", "pull", "merge", "am",
                           "cherry-pick", "revert", "rebase", "switch", "clean", "mv", "rm"})
#: shell words that open or close a compound command: the command after them is the one that runs
_SHELL_KEYWORDS = frozenset({"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}",
                             "esac"})
_NO_WRITE_TARGETS = ("/dev/",)
#: prose: writing it after a check does not void the check (§4QB-c precedent)
_PROSE_EXT = (".md", ".markdown", ".txt", ".rst", ".adoc")
#: inline code that writes files (a generator, a fix script) — a write, not a check
_CODE_WRITES_RE = re.compile(
    r"\.write_(?:text|bytes)\s*\(|\bopen\s*\([^)]*?,\s*(?:mode\s*=\s*)?['\"][rbt+]*[wax]|"
    r"\.to_(?:csv|json|excel|parquet|html|sql)\s*\(|\bjson\.dump\s*\(|\bshutil\.(?:copy\w*|move|rmtree)\s*\(|"
    r"\bos\.(?:remove|unlink|rename|replace|makedirs|mkdir)\s*\(|\.unlink\s*\(|\.savefig\s*\(|"
    r"\b(?:writeFileSync|appendFileSync|writeFile|appendFile|mkdirSync|rmSync|unlinkSync)\s*\(|"
    r"\bfile_put_contents\s*\(|\bFile\.write\s*\(")


def is_prose_path(path) -> bool:
    return str(path or "").strip().lower().endswith(_PROSE_EXT)


def code_writes_files(code) -> bool:
    """Inline code (``python -c``, an ``execute`` content script, a here-doc fed to python / node) that writes
    files — a generator or a fix script: a WRITE, not the check of what it wrote."""
    return bool(_CODE_WRITES_RE.search(str(code or "")))


def _local_url(arg: str) -> bool:
    """``arg`` addresses a service on this machine (``http://localhost:8000/x``, ``127.0.0.1:5000``, a container's
    single-label name with a scheme: ``http://web:80``)."""
    m = re.match(r"(?i)(https?://)?(\[[^\]]+\]|[^/:@?#\s]+)(:\d+)?(?:[/?#]|$)", arg or "")
    if not m:
        return False
    host = m.group(2).lower()
    if host in _LOCAL_HOSTS or host.startswith("127."):
        return True
    return bool(m.group(1)) and "." not in host            # a scheme and a single-label (container) name


def _local_script(head: str) -> bool:
    """``./run.sh``, ``bin/test``, ``/workspace/projects/x/run.sh``: a program file of the workspace."""
    if head.startswith(("./", "../", "/workspace/", "~/")):
        return True
    return "/" in head and not head.startswith("/")


def _heredoc_bodies(command: str) -> list:
    """The here-document bodies of ``command`` (what a shell or an interpreter fed ``<<EOF`` runs)."""
    from ..tools.execute import _HEREDOC_RE
    out, rest = [], str(command or "")
    while True:
        m = _HEREDOC_RE.search(rest)
        if not m:
            return out
        nl = rest.find("\n", m.end())
        if nl < 0:
            return out
        body = rest[nl + 1:]
        end = re.search(rf"(?m)^\s*{re.escape(m.group(2))}\s*$", body)
        out.append(body[:end.start()] if end else body)
        rest = body[end.end():] if end else ""


def _substitutions(word: str) -> list:
    """The commands inside ``$(…)`` and backticks of one shell word (a quoted ``"$(pytest -q)"`` stays one word)."""
    out, i, s = [], 0, str(word or "")
    while i < len(s):
        if s.startswith("$(", i) and not s.startswith("$((", i):
            depth, j = 1, i + 2
            while j < len(s) and depth:
                depth += {"(": 1, ")": -1}.get(s[j], 0)
                j += 1
            out.append(s[i + 2:j - 1] if not depth else s[i + 2:])
            i = j
        elif s[i] == "`":
            j = s.find("`", i + 1)
            out.append(s[i + 1:j] if j > 0 else s[i + 1:])
            i = len(s) if j < 0 else j + 1
        else:
            i += 1
    return out


def _inline_code(head: str, args: list, command: str) -> Optional[str]:
    """The inline code an interpreter runs (``-c`` / ``-e`` / ``--eval`` / ``-p``, or its stdin here-doc), None for
    a script file."""
    for j, a in enumerate(args):
        if a in ("-c", "-e", "--eval", "-p", "--print", "-E") and j + 1 < len(args):
            return args[j + 1]
        if a.startswith("-c") and len(a) > 2 and _PY_RE.fullmatch(head):
            return a[2:]
        if not a.startswith("-"):
            return None                                  # a script file
        if a == "-":
            break
    return "\n".join(_heredoc_bodies(command))           # bare `python`, `python -`, `node <<EOF`


def _runs_program(words: list, command: str, depth: int) -> list:
    """The events of ONE simple command (prefixes and shell keywords already dropped): ``"c"`` where it runs a
    program, ``"w"`` where it rewrites files (a formatter, ``perl -pi``, inline code that writes), a nested
    command's events for ``bash -c …`` / ``uv run …`` / ``npx …`` / a shell fed a here-doc, else nothing."""
    if not words:
        return []
    head = os.path.basename(words[0])
    args = words[1:]
    if args and all(a in _INFO_ARGS for a in args):
        return []                                        # `python --version`, `npm help`
    if _PY_RE.fullmatch(head):
        for j, a in enumerate(args):
            if a == "-m" or (a.startswith("-m") and len(a) > 2):
                mod = args[j + 1] if a == "-m" and j + 1 < len(args) else a[2:]
                if mod in _PY_NOT_CHECK_MODULES:
                    return []
                return _runs_program([mod] + args[j + 2 if a == "-m" else j + 1:], command, depth) \
                    if mod in _FORMATTERS or mod == "ruff" else ["c"]
            if not a.startswith("-"):
                break
        code = _inline_code(head, args, command)
        return ["w"] if code is not None and code_writes_files(code) else ["c"]
    if head in _SHELLS:
        for j, a in enumerate(args):
            if a == "-c" or (a.startswith("-") and not a.startswith("--") and a.endswith("c")):
                return _events(args[j + 1], depth + 1) if j + 1 < len(args) else []
            if not a.startswith("-"):
                return ["c"]                              # bash run.sh
        evs = []                                         # a shell reading its script from a here-doc
        for body in _heredoc_bodies(command):
            evs += _events(body, depth + 1)
        return evs
    if head in ("perl", "ruby") and any(re.fullmatch(r"-[a-zA-Z]*i\S*", a) for a in args):
        return ["w"]                                     # `perl -pi -e …`, `ruby -i.bak …`: in-place edit
    if head in _INTERPRETERS:
        code = _inline_code(head, args, command)
        return ["w"] if code is not None and code_writes_files(code) else ["c"]
    if head in _FORMATTERS or (head == "ruff" and args[:1] == ["format"]) \
            or (head in ("cargo", "go", "dotnet") and args[:1] in (["fmt"], ["format"])):
        if any(a in _FORMAT_CHECK_FLAGS for a in args):
            return ["c"]                                 # `black --check`, `ruff format --check`
        if head in ("prettier", "gofmt") and not any(a in ("--write", "-w") for a in args):
            return ["c"]                                 # prints the formatted text, writes nothing
        return ["w"]
    if head in _RUNNERS:
        return ["c", "w"] if any(a in ("--fix", "--write") or a.startswith("--fix=") for a in args) else ["c"]
    if head in _RUN_WRAPPERS:
        return _simple_events(args[1:], command, depth + 1) if args[:1] == ["run"] else []
    if head in _NPX or (head == "npm" and args[:1] in (["exec"], ["x"])) \
            or (head in ("yarn", "pnpm") and args[:1] in (["dlx"], ["exec"])):
        rest = [a for a in (args if head in _NPX else args[1:]) if a not in ("-y", "--yes", "--")]
        evs = _simple_events(rest, command, depth + 1)
        if evs or not rest or "install" in rest or os.path.basename(rest[0]).startswith("create-"):
            return evs
        return ["c"]                                     # another program it fetched and ran
    if head in ("yarn", "pnpm"):
        return ["c"] if (not args or args[0] not in _PKG_NOT_RUNS) and not (args and args[0].startswith("-")) \
            else []
    if head in _SUBCOMMAND_RUNS:
        return ["c"] if args and args[0] in _SUBCOMMAND_RUNS[head] else []
    if head in ("docker", "podman") and args and args[0] == "exec":
        rest = args[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("-e", "--env", "-u", "--user", "-w", "--workdir") else rest[1:]
        return _simple_events(rest[1:], command, depth + 1) if len(rest) > 1 else []
    if head in _HTTP_HEADS:
        return ["c"] if any(_local_url(a) for a in args if not a.startswith("-")) else []
    if _local_script(words[0]):
        return ["c"]
    return []


def _strip_keywords(words: list) -> list:
    """``if pytest``, ``do npm test``, ``! grep``: the command after the compound-command words; a ``for`` /
    ``case`` / ``select`` header runs nothing itself."""
    w = list(words)
    while w and w[0] in _SHELL_KEYWORDS:
        w = w[1:]
    if w and w[0] in ("for", "case", "select", "function", "in"):
        return []
    return w


def _simple_events(words: list, command: str, depth: int) -> list:
    from ..tools.execute import strip_command_prefixes
    if depth > 4:
        return []
    w = _strip_keywords(strip_command_prefixes(_strip_keywords(words)))
    return _runs_program(w, command, depth) or []


def _mutates(words: list) -> bool:
    from ..tools.execute import strip_command_prefixes
    w = _strip_keywords(strip_command_prefixes(_strip_keywords(words)))
    if not w:
        return False
    head = os.path.basename(w[0])
    ops = [a for a in w[1:] if not a.startswith("-")]
    if head in _MUTATING_HEADS:
        return not ops or not all(is_prose_path(o) for o in ops)
    return head == "git" and bool(ops) and ops[0] in _GIT_MUTATING


def _events(command: str, depth: int = 0) -> list:
    """§4UL re-review M1: the ordered events of a shell command — ``"c"`` for every simple command that runs an
    allow-listed program (in a pipeline, after ``;`` / ``||``, inside a ``for`` / ``if`` body, ``bash -c``,
    ``$(…)``), ``"w"`` where it writes a file that is not prose (a redirect, ``tee``, ``cp``, ``sed -i``, ``rm``,
    a formatter, inline code that writes). A program's own output saved to a file (``pytest > out.txt``,
    ``… | tee run.log``) is its log, not a write. A command that cannot be split has no events."""
    from ..tools.execute import (_INPUT_REDIRECTS, _REDIRECTS, _SEPARATORS, _shell_tokens,
                                 _shell_write_targets, strip_command_prefixes)
    if depth > 4:
        return []
    toks = _shell_tokens(command)
    if toks is None:
        return []
    segs: list = []                                      # [words, raw, separator after it]
    words, raw = [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _SEPARATORS:
            segs.append([words, raw, t])
            words, raw = [], []
        elif t in _REDIRECTS or (t.startswith(">") and set(t) <= set(">|&")) or t in _INPUT_REDIRECTS:
            raw.append(t)
            if i + 1 < len(toks) and toks[i + 1] not in _SEPARATORS:
                raw.append(toks[i + 1])
                i += 1
        elif t.isdigit() and i + 1 < len(toks) and toks[i + 1][:1] in "<>":
            raw.append(t)
        else:
            words.append(t)
            raw.append(t)
        i += 1
    segs.append([words, raw, ""])
    evs: list = []
    prev_ran, prev_sep = False, ""
    for w, r, sep in segs:
        for word in w:                                   # `echo "$(pytest -q)"`, `x=\`make\``
            for inner in _substitutions(word):
                evs += _events(inner, depth + 1)
        prog = _simple_events(w, command, depth) if w else []
        if prog:
            evs += prog                                  # a program's redirect is its log
        elif w or r:
            ww = _strip_keywords(strip_command_prefixes(_strip_keywords(w)))
            head = os.path.basename(ww[0]) if ww else ""
            text = " ".join(t if (set(t) <= set("<>|&") or t.isdigit()) else shlex.quote(t) for t in r)
            tgts = [t for t in _shell_write_targets(text)
                    if not t.startswith(_NO_WRITE_TARGETS) and not is_prose_path(t)]
            if head == "mkdir" or (head == "tee" and prev_ran and prev_sep in ("|", "|&")):
                tgts = []                                # a directory is not content; a program's tee is its log
            if tgts or _mutates(w):
                evs.append("w")
        if w:
            prev_ran, prev_sep = bool(prog), sep
        elif sep:
            prev_sep = sep
    return evs


def shell_check_events(command: str) -> list:
    """Public face of `_events` for the turn's watch and its tests."""
    try:
        return _events(command)
    except Exception:  # noqa: BLE001 — never breaks a call: no events
        logger.debug("check events skipped", exc_info=True)
        return []


def _ran(res) -> bool:
    """The call RAN (its program started, the page answered): not refused before it ran. A failing exit status
    still ran — an honest model reads the failure; the gate only asks that something ran."""
    if getattr(getattr(res, "status", None), "value", None) == "rejected":
        return False
    return not str(res or "").lstrip().lower().startswith(("error", "system instruction", "lab error"))


def _write_landed(res) -> bool:
    """A write call changed something: not refused before it ran. A PARTIAL write (kept, syntax flagged) and a
    command that failed after it wrote both count."""
    if getattr(res, "world_changed", None) is False:
        return False
    st = getattr(getattr(res, "status", None), "value", None)
    if st is not None:
        return st != "rejected" and not (st in ("failed", "unresolved") and _refused_text(res))
    return not _refused_text(res)


def _refused_text(res) -> bool:
    return str(res or "").lstrip().lower().startswith(("error", "system instruction"))


def _canon(value) -> str:
    """An id the way `manage_projects` resolves it (`memory.projects._canon_id`)."""
    from ..memory.projects import _canon_id
    return _canon_id(str(value) if value is not None else "")


def _first_id(value) -> str:
    """One id from a value the model may send as ``"id"``, ``'["id"]'`` or ``["id"]``."""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    if isinstance(value, str) and value.strip().startswith("["):
        from ..tools.projects import _coerce_str_list
        return (_coerce_str_list(value) or [""])[0]
    return str(value or "")


def _task_ids(kwargs: Dict[str, Any]) -> list:
    """Every task id a ``task_update`` call aims at, read as the handler reads them: ``task_id`` (a JSON-list
    string there is expanded) and ``task_ids`` (a list, a JSON-list string or a bare / comma string)."""
    from ..tools.projects import _coerce_str_list
    out = []
    tid = kwargs.get("task_id")
    if isinstance(tid, str) and tid.strip().startswith("["):
        out += _coerce_str_list(tid) or []
    elif isinstance(tid, (list, tuple)):
        out += [str(x) for x in tid]
    elif tid:
        out.append(str(tid))
    out += _coerce_str_list(kwargs.get("task_ids")) or []
    return list(dict.fromkeys(c for c in (_canon(x) for x in out) if c))


def _applied_statuses(res, asked: list, status: str) -> Dict[str, str]:
    """Rounds 7–8: what a ``task_update`` really wrote — each ``updated`` entry with the status it ENDED at (a DONE
    the close gates turned into FAILED is the turn's own FAILED close). An asked id missing from ``updated`` (a
    gate refused it before writing) books nothing. A result that is not the handler's JSON books the ids asked
    for with the status asked for."""
    import json
    try:
        payload = json.loads(str(res or ""))
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("updated"), list):
        return {i: status for i in asked}
    out: Dict[str, str] = {}
    for u in payload["updated"]:
        if isinstance(u, dict) and _canon(u.get("id")) in asked:
            out[_canon(u.get("id"))] = _norm_status(u.get("status")) or status
    return out


def deferral_marks() -> tuple:
    """Round 8: how a deferral raised inside the turn's model call reads once `agent.py`'s catch-all has turned it
    into a reply ("… with the LLM (BackgroundDeferred) … [ATTEMPT_ABORTED_UPSTREAM]") — derived from the classes
    themselves (and every subclass), so a rename cannot drift. Round 9: the whole phrase agent.py writes, so a model
    reply that merely quotes "(BackgroundDeferred)" is not one (`is_deferral_reply` also needs the abort marker)."""
    from .llm import BackgroundDeferred
    seen, stack = [], [BackgroundDeferred]
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.append(c)
            stack.extend(c.__subclasses__())
    return tuple(f"with the LLM ({c.__name__})" for c in seen)


def is_deferral_reply(reply) -> bool:
    """Round 9: agent.py's catch-all reply for a deferral raised in the model call — its phrase AND its marker."""
    from .agent import UPSTREAM_ABORT_MARKER
    text = str(reply or "")
    return UPSTREAM_ABORT_MARKER in text and any(m in text for m in deferral_marks())


def _project_rel(path, project_id: str) -> str:
    """A path the turn wrote, relative to its project's folder ("" for one outside it)."""
    from ..tools.execute import workspace_relative
    p = workspace_relative(str(path or "").strip()).replace("\\", "/")
    pre = f"projects/{project_id}/"
    if project_id and p.startswith(pre):
        p = p[len(pre):]
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith(("/", "~")) or ".." in p.split("/") or p.startswith("projects/"):
        return ""
    return p


class _TurnWatch:
    """What the turn did that the advancer reads afterwards — and (§4UL review M2/M4/M7) the turn's scope: its
    project and task, and whether it runs unattended (then a DONE waits for a check run after the last write)."""

    def __init__(self, project_id: str = "", task_id: str = "", *, unattended: bool = False,
                 classification: str = "", description: str = "", store=None, kind: str = ""):
        self.kept_refused = ""
        self.last_write = -1.0
        self.last_check = -1.0
        self.n = 0
        self.refused_actions: list = []
        self.project_id = str(project_id or "")
        self.task_id = str(task_id or "")
        self.unattended = bool(unattended)
        self.classification = str(classification or "")
        self.description = str(description or "")
        self.kind = str(kind or "")
        self.store = store
        #: §4UL round 3 (D): the DONEs refused unattended, by the task each one TARGETED (canonical id) →
        #: {kind, why, result, kwargs, at}. Not committed; each still-open one is held for the owner after the turn,
        #: on every exit path (round 4: never replayed — the turn is told to close again after its check)
        self.held: Dict[str, Dict[str, Any]] = {}
        #: round 4 m6: the tasks the turn itself set a status on (a task_update that went through) — any other
        #: closed status at the turn's end was set by someone else during it (the owner's console)
        self.turn_set: Dict[str, str] = {}
        #: files the turn wrote (project-relative), kept with a hold so the owner's run re-checks, not rebuilds
        self.files: Dict[str, None] = {}

    def note(self, name: str, kwargs: Dict[str, Any], res) -> None:
        """Book one call (§4UL round 3). A CHECK is any allow-listed program that RAN (`shell_check_events`; its
        exit status does not matter), an `execute` of a file or of inline code that does not write files, a
        `lab exec` of an allow-listed command, `lab http` / `lab screenshot`, `run collect` / `run logs`, or a
        browser call (not ``close``). A WRITE is any file_system operation that is not a read and landed, or an
        `execute` that writes a file (redirect, `tee`, `cp`, `sed -i`, `rm`, a formatter, inline code that
        writes). Prose (.md / .txt / .rst …) is never a write that voids a check. Order inside one command
        counts: ``echo x > a.py && python a.py`` checks after its write."""
        from ..tools.execute import _shell_write_targets
        self.n += 1
        text = str(res or "")
        if name in ("lab", "run") and any(m in text for m in KEPT_REFUSAL_MARKS):
            self.kept_refused = text[:300]
        op = str(kwargs.get("operation") or kwargs.get("action") or "").strip().lower()
        if name == "manage_projects" and op == "task_update":
            st = _norm_status(kwargs.get("status"))
            if st and _write_landed(res):
                for i, st in _applied_statuses(res, _task_ids(kwargs) or [_canon(self.task_id)], st).items():
                    self.turn_set[i] = st
                    if st in _DROPS_HOLD:
                        self.held.pop(i, None)  # rounds 4–5: FAILED / BLOCKED / … after a refused DONE stands
            return
        evs: list = []
        wrote: list = []
        if name == "file_system":
            from ..tools.file_system import QUARANTINE_OPS, _READ_ONLY_OPS
            if op and op not in _READ_ONLY_OPS and op not in QUARANTINE_OPS:
                tgt = (kwargs.get("destination") if op in ("move", "rename", "copy") else None) \
                    or kwargs.get("path") or kwargs.get("filename") or kwargs.get("file") or ""
                wrote = [tgt]
                evs = [] if is_prose_path(tgt) else ["w"]
        elif name == "execute":
            cmd = str(kwargs.get("command") or kwargs.get("cmd") or "")
            if cmd:
                evs = shell_check_events(cmd)
                wrote = _shell_write_targets(cmd) if "w" in evs else []
            else:
                fname = kwargs.get("filename") or kwargs.get("file") or kwargs.get("script") or kwargs.get("name")
                code = (kwargs.get("content") or kwargs.get("code") or kwargs.get("script_content")
                        or kwargs.get("text"))
                if fname and code:
                    wrote = [fname]
                    evs = [] if is_prose_path(fname) else ["w"]
                if fname or code:
                    evs.append("w" if code and code_writes_files(code) else "c")
        elif name == "lab":
            if op == "exec":
                # what a lab command writes is on the lab, not in the workspace the DONE delivers
                evs = [e for e in shell_check_events(str(kwargs.get("cmd") or kwargs.get("command") or ""))
                       if e == "c"]
            elif op in ("http", "screenshot"):
                evs = ["c"]
        elif name == "run" and op in ("collect", "logs"):
            evs = ["c"]
        elif name == "browser" and op != "close":
            evs = ["c"]
        landed = _write_landed(res) if (wrote or "w" in evs) else False
        ran = _ran(res) if "c" in evs else False
        for k, e in enumerate(evs):
            pos = self.n + (k + 1) / 1000.0
            if e == "w" and landed:
                self.last_write = pos
            elif e == "c" and ran:
                self.last_check = pos
        if landed:
            for p in wrote:
                rel = _project_rel(p, self.project_id)
                if rel:
                    self.files[rel] = None

    @property
    def verified(self) -> bool:
        """A program ran after the last write."""
        return self.last_check > self.last_write

    def _own(self, tid: str) -> bool:
        """``tid`` is the turn's task or a subtask under it — ids compared the way the handler resolves them
        (§4UL re-review M2: stripped, lowercased)."""
        tid, mine = _canon(tid), _canon(self.task_id)
        if not tid or not mine:
            return False
        if tid == mine:
            return True
        try:
            seen, cur = set(), tid
            while cur and cur not in seen and len(seen) < 32:
                seen.add(cur)
                row = (self.store.get_task(cur) or {}) if self.store is not None else {}
                cur = _canon(row.get("parent_id"))
                if cur == mine:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def scope_refusal(self, act: str, kwargs: Dict[str, Any]):
        """A refusal for a manage_projects call outside this turn's task, else None (§4UL review M4; re-review
        M2: ids read the way the handler reads them — case, a JSON-list string, a bare string for ``task_ids``;
        re-review m1: another project only READ, tasks added only under this turn's task)."""
        if act in REFUSED_PROJECT_ACTIONS:
            return _refusal(act)
        if not self.project_id:
            return None
        pid = _canon(kwargs.get("project_id"))
        if pid and pid != _canon(self.project_id):
            writes = [k for k in _WRITE_FORM_ARGS.get(act, ()) if str(kwargs.get(k) or "").strip()]
            if act not in _READ_ACTIONS or writes:
                return _refusal(f"{act} on another project")
            return None
        if act in _OWN_TASK_ACTIONS:
            others = [i for i in _task_ids(kwargs) if not self._own(i)]
            if others:
                return _refusal(f"{act} on task(s) {', '.join(o[:12] for o in others[:3])} (not this turn's)")
        if act in _OWN_PARENT_ACTIONS:
            key = _OWN_PARENT_ACTIONS[act]
            parent = _canon(_first_id(kwargs.get(key)))
            top = not parent or parent == _canon(self.project_id)
            # a follow-up task at the top level stays allowed (review M4); a decompose there would lay down a
            # new project plan, and a parent elsewhere aims at another task
            if (top and act == "task_decompose") or (not top and not self._own(parent)):
                from ..tools.outcome import ToolOutcome
                return ToolOutcome.rejected(
                    f"Error: `{act}` here may only add work UNDER this turn's task — pass {key}={self.task_id} "
                    f"(another task's parts and the project's plan are the owner's turns' business). Nothing was "
                    f"changed.",
                    world_changed=False, reason_code="task_turn_scope")
        return None

    def close_verdict(self, tid: str):
        """`_unattended_close` (status, note) for an unattended DONE of task ``tid``."""
        from .project_advancer import _unattended_close
        desc, unverified = self._close_rule(tid)
        return _unattended_close(False, desc, unverified_build=unverified)

    def _close_rule(self, tid: str):
        """(description, unverified_build) for `_unattended_close` on task ``tid``: a check that ran clears the
        word hold too; research and a declared ``write`` (prose) task with no check keep only the word hold
        (§4LP; round 3 E)."""
        desc, kind = self.description, self.kind
        if _canon(tid) != _canon(self.task_id) and self.store is not None:
            try:
                row = self.store.get_task(tid) or {}
                desc = str(row.get("description") or "")
                from .experiment_gate import planned_kind
                kind = planned_kind(self.store, self.project_id, _canon(tid), str(row.get("kind") or ""))
            except Exception:  # noqa: BLE001
                pass
        if self.verified:
            return desc, False
        if self.classification == "research" or kind in ("research", "write"):
            return desc, None
        return desc, True

    def done_gate(self, act: str, kwargs: Dict[str, Any]):
        """§4UL review M2/M7: unattended, a DONE is taken only after a check that RAN after the last write
        (`_unattended_close`); otherwise nothing is closed (no roll-up, no clean-up sweep, no constraint
        retirement), the refusal is recorded against the task(s) it TARGETED (round 3 D) and the turn is told
        what counts. None = let the call through."""
        if not self.unattended or act != "task_update":
            return None
        if _norm_status(kwargs.get("status")) != "DONE":
            return None
        from .project_advancer import _hold_kind_of
        from .planning import TaskStatus
        ids = _task_ids(kwargs) or [_canon(self.task_id)]
        verdicts = {i: self.close_verdict(i) for i in ids}
        if all(st == TaskStatus.DONE for st, _ in verdicts.values()):
            for i in ids:
                self.held.pop(i, None)          # checked since: the DONE goes through the close gates
            return None
        for i in ids:                           # round 4 m2: each id by its OWN verdict
            st, _w = verdicts[i]
            if st != TaskStatus.DONE:
                self.held[i] = {"kind": _hold_kind_of(_w), "why": _w,
                                "result": str(kwargs.get("result") or "")[:600], "kwargs": dict(kwargs),
                                "at": float(self.n)}
        from ..tools.outcome import ToolOutcome
        return ToolOutcome.rejected(
            "Not closed yet: this run is unattended and nothing was run since your last edit. Run the program or its tests (execute), open the "
            "page (browser), or check the lab (lab http / lab exec), then close DONE again. If it cannot be "
            "checked here, end the turn without closing: the task then waits for the owner's look with what you "
            "did.", world_changed=False, reason_code="task_turn_unverified_done")

    def checked_since(self, held: Dict[str, Any]) -> bool:
        """A check ran after this refused DONE and after the last write (but the turn never closed it again)."""
        return self.verified and self.last_check > float(held.get("at") or 0.0)


def _refusal(action: str):
    from ..tools.outcome import ToolOutcome
    return ToolOutcome.rejected(
        f"Error: `{action}` is not part of this task's turn — this turn does ONE task. Do the task and close it "
        f"with manage_projects action=task_update (DONE with deliverables, or FAILED with the reason). Nothing "
        f"was changed.", world_changed=False, reason_code="task_turn_scope")


def _wrap_tools(tools: Dict[str, Any], watch: _TurnWatch) -> Dict[str, Any]:
    import functools
    out = dict(tools or {})
    for name, fn in list(out.items()):
        if getattr(fn, "_ghost_task_turn_wrapped", False) or not callable(fn):
            continue

        def _make(_name, _fn):
            @functools.wraps(_fn)                       # name, doc, signature and the `_ghost_*` marks
            async def _wrapped(**kwargs):
                if _name == "manage_projects":
                    act = str(kwargs.get("action") or "").strip().lower()
                    _ref = watch.scope_refusal(act, kwargs) or watch.done_gate(act, kwargs)
                    if _ref is not None:
                        watch.refused_actions.append(act)
                        return _ref
                res = await _fn(**kwargs)
                try:
                    watch.note(_name, kwargs, res)
                except Exception:  # noqa: BLE001 — bookkeeping never breaks a call
                    logger.debug("task turn watch skipped", exc_info=True)
                return res
            _wrapped._ghost_task_turn_wrapped = True
            return _wrapped
        out[name] = _make(name, fn)
    return out


def _prepare(watch: _TurnWatch):
    def _apply(agent) -> None:
        agent.available_tools = _wrap_tools(agent.available_tools, watch)
        _orig = agent._rebuild_available_tools

        def _rebuild():
            # a dispatch-miss rebuild heals the dict from the registry — the
            # scope wrapper must come back with it
            _orig()
            agent.available_tools = _wrap_tools(agent.available_tools, watch)
            return agent.available_tools
        agent._rebuild_available_tools = _rebuild
    return _apply


def task_turn_prompt(proj: Dict[str, Any], nxt, classification: str, origin: str) -> str:
    from .task_deliverable import declared_deliverable
    title = str(proj.get("title") or proj.get("id") or "")
    goal = str(proj.get("goal") or "").strip()
    kind = str(getattr(nxt, "kind", "") or "") or classification
    deliv = declared_deliverable(getattr(nxt, "description", "") or "")
    lines = [
        f"PROJECT TASK — project {title!r} ({proj.get('id')}), task {nxt.id} (kind: {kind}).",
        f"Project goal: {goal[:600]}" if goal else "",
        "",
        f"TASK: {str(nxt.description or '').strip()}",
        f"Declared deliverable: {deliv}" if deliv else "",
        "",
        "Do this ONE task now, with your tools, in the project workspace — the way you would in the owner's own "
        "turn: read what exists, build, RUN what you built and fix what the output shows. Nobody is present: do "
        "not ask questions; when something is unclear, choose the reasonable reading and say which in the result.",
        f"When the task is done, close it: manage_projects action=task_update task_id={nxt.id} status=DONE "
        "result=\"<what you did and how you checked it>\" deliverables=[<the files>]. If it cannot be done, close "
        f"it with status=FAILED and the reason. Work on this task only — the next task is not yours.",
    ]
    if origin != "owner":
        lines.append("This run is unattended: kept (production) machines are the owner's. If the task needs one "
                     "(a deploy to a kept lab), do everything else it needs, then end WITHOUT closing it — it "
                     "waits for the owner's run. A DONE is taken only after you ran something after your last edit "
                     "(the program, its tests, the page in the browser, lab http) — run it before you close.")
    return "\n".join(x for x in lines if x is not None).replace("\n\n\n", "\n\n").strip()


def project_done_hook_delivers(rid) -> bool:
    """§4UL review M3: does the project-done hook (`main._on_project_done`) deliver a stored "notify me when it
    is done" promise for a project that closed under request ``rid``? Outside any request (the idle loop) and in
    idle autopilot's task turn (``sub-ptask-…`` with no request around it) yes; inside a request no — that
    request's finalize delivers it (with its own did-the-model-already-notify check)."""
    r = str(rid or "")
    if r in ("", "SYSTEM", "None"):
        return True
    if not r.startswith(TASK_TURN_RID_PREFIX):
        return False
    try:
        from ..memory.projects import request_root
        return request_root(r) == r
    except Exception:  # noqa: BLE001
        return False


def _outer_request_id() -> str:
    """The request a task turn runs inside (the owner's, a scheduled one, a probe), "" for idle autopilot."""
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ""
    return "" if rid in ("", "SYSTEM") else rid


def _time_for_turn() -> float:
    """The turn's time: its cap, bounded by the request the batch runs in and
    by the unattended step's deadline (`leaf_deadline_context`)."""
    from .coding_executor import _request_left_s, leaf_deadline_context
    t = float(TASK_TURN_TIMEOUT_S)
    left = _request_left_s()
    if left is not None:
        t = min(t, float(left) - 60.0)          # the batch's own report still has to be written
    outer = float(leaf_deadline_context.get() or 0.0)
    if outer > 0:
        t = min(t, outer - time.monotonic())
    return t


async def run_task_turn_for_advance(context, store, project_id: str, nxt, classification: str,
                                    owner_requested: bool, *, idle: bool = False,
                                    started_at: Optional[float] = None):
    """Run task ``nxt`` (claimed IN_PROGRESS by the advancer) as one agent
    turn. Returns an `AdvanceResult`, or None when the turn could not be
    built at all (the caller falls back to the old executor and logs it)."""
    from .coding_loop import run_leaf_turn
    from .planning import ProjectPlan, TaskStatus
    from .project_advancer import (AdvanceResult, MODEL_SERVER_FAILED_PREFIX, OUT_OF_TIME_PREFIX, _HELD_MARK,
                                   _increment_budget, _work_log_step)
    from .project_safety import record_runtime
    t0 = started_at or time.time()
    tid = nxt.id
    origin = _origin_for(owner_requested)
    unattended = origin != "owner"

    def _release(why: str) -> str:
        """Give the claim back (PENDING, or the owner's hold it was claimed from); the status written."""
        try:
            return store.release_claim(tid, failure_reason=why) or ""
        except Exception:  # noqa: BLE001
            logger.debug("task turn: claim not released", exc_info=True)
            return ""

    if idle and _foreground_busy(context):
        _release("deferred for the owner's request")
        return AdvanceResult(True, tid, "deferred",
                             f"deferred: a user request is active — '{tid}' reopened for a quieter moment",
                             worked=False)      # round 6: nothing ran — never charged by the fallback
    budget_s = _time_for_turn()
    if budget_s < TASK_TURN_MIN_S:
        _release("out of time for a task turn")
        return AdvanceResult(True, tid, classification,
                             f"{OUT_OF_TIME_PREFIX}: {int(max(0.0, budget_s))} s left, a task turn needs "
                             f"{int(TASK_TURN_MIN_S)} s", worked=False)
    try:
        proj = store.get_project(project_id) or {"id": project_id}
        prompt = task_turn_prompt(proj, nxt, classification, origin)
        from .experiment_gate import planned_kind
        watch = _TurnWatch(project_id, tid, unattended=unattended, classification=classification,
                           description=str(nxt.description or ""), store=store,
                           kind=planned_kind(store, project_id, tid, str(getattr(nxt, "kind", "") or "")))
        rid = f"{TASK_TURN_RID_PREFIX}{str(tid)[:12]}-{os.urandom(3).hex()}"
        # §4UL review M5: the turn runs inside this request — its DONE sweeps wait for that request's end, and a
        # write the request makes after the turn closed the last task is the same work (`memory.projects`)
        outer = _outer_request_id()
        if outer:
            from ..memory.projects import set_request_parent
            set_request_parent(rid, outer)
    except Exception as e:  # noqa: BLE001 — nothing ran: the old path takes the task
        logger.warning("task turn not built for %s (%s: %s) — the build executor runs it", tid,
                       type(e).__name__, e)
        return None
    pretty_log("Task Turn", f"{origin} · {str(tid)[:8]} · {classification} · "
               f"{str(nxt.description or '')[:70]}", icon=Icons.BRAIN_PLAN)
    def _held_files(hid: str, h: Dict[str, Any], description: str) -> list:
        """Round 4 m8: what a hold keeps for the owner's re-check — the task's OWN DONE deliverables, and files the
        turn wrote under the path the task declared; never every file the turn touched."""
        from ..tools.projects import _coerce_str_list
        from .task_deliverable import declared_deliverable
        out = [r for r in (_project_rel(d, project_id)
                           for d in (_coerce_str_list(h["kwargs"].get("deliverables")) or [])) if r]
        decl = _project_rel(declared_deliverable(description) or "", project_id).rstrip("/")
        if decl:
            out += [f for f in watch.files if f == decl or f.startswith(decl + "/")]
        return list(dict.fromkeys(out))[:40]

    def _settle_holds():
        """§4UL review M2/M7, rounds 3–4: each DONE refused unattended and never closed again is held on the task it
        TARGETED — on EVERY exit path (a normal end, a timeout, the owner's request taking the slot, a crash, a
        cancel) that task waits for the owner's look with what the turn did; it is never rolled up DONE, and never
        replayed. Only a task still open as the turn left it is held: the turn's own task still IN_PROGRESS, a
        subtask it made still PENDING / READY — one the owner moved meanwhile keeps the owner's status. A task the
        turn split is never parked (a parked parent would hold every subtask). A check that ran after the refusal
        holds with an honest reason and no re-check kind, so the owner's run re-judges it. Synchronous: a cancel
        cannot cut it short; a store error on one task is logged and the others are still held. Returns the
        AdvanceResult when the turn's OWN task was held, else None."""
        mine = _canon(tid)
        held_mine = None
        if not watch.held:
            return None
        try:
            plan = ProjectPlan(store, project_id)
        except Exception:  # noqa: BLE001
            logger.warning("task turn: plan not read — refused DONEs not held", exc_info=True)
            return None
        for hid, h in list(watch.held.items()):
            try:
                _row = store.get_task(hid) or {}
                _st = str(_row.get("status") or "").upper()
                _open = ("PENDING", "READY", "IN_PROGRESS")
                if hid == mine and _st != "IN_PROGRESS" and not (
                        _st in ("PENDING", "READY") and watch.turn_set.get(hid) in ("PENDING", "READY")):
                    continue                    # closed, or moved by the owner during the turn (round 6: the turn
                if _st not in _open:            # setting its own task back to PENDING / READY keeps the hold; round
                                                # 7: an IN_PROGRESS it set earlier never covers an owner's reset)
                    continue
                node = plan.tree.nodes.get(hid)
                if node is not None and list(getattr(node, "children", None) or []):
                    continue                    # split: never park a parent
                if watch.checked_since(h):
                    _kind = ""
                    _why = f"a check ran after the refused DONE but the turn never closed it again — {_HELD_MARK}"
                else:
                    _kind = h.get("kind") or "unverified_build"
                    _why = h.get("why") or f"built but no verification command ran — {_HELD_MARK}"
                desc = str(_row.get("description") or "")
                files = _held_files(hid, h, desc)
                plan.update_status(hid, TaskStatus.NEEDS_USER,
                                   result=f"{(h.get('result') or '(no result text)')[:300]} — {_why}",
                                   wait=("owner_turn", _kind or "unverified_build", _why))
                store.log_event(project_id, hid, "autoadvance_needs_user",
                                {"description": desc, "reason": _why, "hold": _kind, "files": files})
                for rel in files:               # kept with the hold: the owner's run re-checks, not rebuilds
                    try:
                        store.register_file_artifact(hid, rel, description=(h.get("result") or "")[:180])
                    except Exception:  # noqa: BLE001
                        logger.debug("held file not registered", exc_info=True)
                _work_log_step(store, project_id, node if node is not None else nxt, outcome="held",
                               files=files, note=f"ran as an agent turn; {_why}")
                try:
                    store.log_event(project_id, hid, "task_turn",
                                    {"origin": origin, "status": "held", "hold": _kind, "calls": watch.n,
                                     "request": rid, "seconds": round(max(0.0, time.time() - t0), 1)})
                except Exception:  # noqa: BLE001
                    pass
                if hid == mine:
                    held_mine = AdvanceResult(True, tid, "needs_user", _why)
            except Exception:  # noqa: BLE001
                logger.warning("task turn: unverified DONE not held for %s", hid, exc_info=True)
        watch.held.clear()
        return held_mine

    def _release_subtasks() -> None:
        """Round 5 M2: a subtask the turn claimed (IN_PROGRESS) and neither closed nor held goes back (its claim
        released: PENDING, or the hold it was claimed from) — never left IN_PROGRESS for the boot reaper, which
        kept the parent open and autopilot idle. Round 6: only a subtask THIS turn set IN_PROGRESS — one another
        run claimed meanwhile (the owner's batch building it) is that run's. The turn's own task follows its own
        exit rules."""
        try:
            nodes = ProjectPlan(store, project_id).tree.nodes
        except Exception:  # noqa: BLE001
            logger.debug("task turn: subtasks not read", exc_info=True)
            return
        stack, seen = list(getattr(nodes.get(tid), "children", None) or []), set()
        while stack:
            cid = stack.pop()
            if cid in seen or cid not in nodes:
                continue
            seen.add(cid)
            stack.extend(nodes[cid].children or [])
            if watch.turn_set.get(_canon(cid)) != "IN_PROGRESS":
                continue
            try:
                if str((store.get_task(cid) or {}).get("status") or "").upper() == "IN_PROGRESS":
                    store.release_claim(cid, failure_reason="its task's turn ended without closing it")
            except Exception:  # noqa: BLE001
                logger.warning("task turn: subtask %s not released", cid, exc_info=True)

    def _flush_idle() -> None:
        """Round 4 (belt and braces for M2): idle autopilot's turn has no request around it — its deferred DONE
        sweeps (and the project-done hook) run now, under the turn's root request. Idempotent."""
        if outer:
            return
        try:
            from ..memory.projects import flush_deferred_sweeps, request_root
            flush_deferred_sweeps(request_root(rid))
        except Exception:  # noqa: BLE001
            logger.debug("task turn: idle sweep flush skipped", exc_info=True)

    async def _finish(reply, exit_kind, exc):
        elapsed = max(0.0, time.time() - t0)
        _h = _settle_holds()                    # round 5: the holds first — a charge that fails cannot lose them
        if exit_kind in ("", "timeout") or watch.n > 0:
            # round 4 m5: every exit where the turn ran charges the budget and the runtime — a deferral, a crash or
            # an abort after work too (only a deferral before any call did no work). The DAILY allowance is booked
            # once, by the advancer, from `AdvanceResult.worked` (round 5 M3)
            try:
                _increment_budget(store, project_id)
            except Exception:  # noqa: BLE001
                logger.warning("task turn: step not counted for %s", project_id, exc_info=True)
            try:
                record_runtime(store, project_id, seconds=elapsed, tool_calls=watch.n)
            except Exception:  # noqa: BLE001
                logger.warning("task turn: runtime not recorded for %s", project_id, exc_info=True)
        if _h is not None:
            return _h
        row = store.get_task(tid) or {}
        status = str(row.get("status") or "").upper()
        if exit_kind and status not in _CLOSED:
            # the turn ended early with its task still open: it goes back for another turn
            if exit_kind == "timeout":
                _release("task turn ran out of time")
                return AdvanceResult(True, tid, classification,
                                     f"{OUT_OF_TIME_PREFIX}: the task's turn passed its {int(budget_s)} s")
            if exit_kind == "deferred":
                _release("deferred for the owner's request")
                return AdvanceResult(True, tid, "deferred",
                                     f"deferred: the owner's request took the slot — '{tid}' reopened")
            if exit_kind == "aborted":
                # round 9: say what happened — the owner's request only when one really holds the slot
                _why = ("deferred for the owner's request" if _foreground_busy(context)
                        else "deferred (the main slot was busy)")
                _release(_why)
                return AdvanceResult(True, tid, "deferred", f"{_why}: the task's turn was cut short — '{tid}' reopened")
            if exit_kind == "upstream":
                _back = _release("the model server failed during the task's turn")
                return AdvanceResult(True, tid, "blocked",       # round 8: a claimed owner hold goes back to it
                                     f"{MODEL_SERVER_FAILED_PREFIX} on '{tid}' — "
                                     + ("task back on hold (owner)" if _back == "NEEDS_USER" else "task reopened"))
            _release(f"task turn crashed: {type(exc).__name__}")
            return AdvanceResult(True, tid, "blocked",
                                 f"coding executor crashed (task turn) on '{tid}' ({type(exc).__name__}: {exc}) — "
                                 "task left open")
        # round 3 G: a turn that closed its task and THEN timed out / crashed / lost the slot is reported by that
        # close, not as "out of time". (The post-turn DONE net is gone, round 5: the turn's own DONE reaches the
        # store only through `done_gate`, which already judged it, so the net could never fire.)
        head = re.sub(r"\s+", " ", str(reply or "")).strip()[:300]
        plan = ProjectPlan(store, project_id)

        def _event(kind: str, extra=None) -> None:
            try:
                store.log_event(project_id, tid, "task_turn",
                                {"origin": origin, "status": kind, "seconds": round(elapsed, 1), "calls": watch.n,
                                 "request": rid, **({"exit": exit_kind} if exit_kind else {}), **(extra or {})})
            except Exception:  # noqa: BLE001
                logger.debug("task_turn event skipped", exc_info=True)

        own_close = {i for i, st in watch.turn_set.items() if st in _OWN_CLOSE}
        _desc, _stack = set(), list(getattr(plan.tree.nodes.get(tid), "children", None) or [])
        while _stack:                           # round 7: the whole subtree — a grandchild's close rolls up too
            _k = _stack.pop()
            if _k in _desc:
                continue
            _desc.add(_k)
            _stack.extend(getattr(plan.tree.nodes.get(_k), "children", None) or [])
        rolled_up = bool(status in _OWN_CLOSE and any(_canon(k) in own_close for k in _desc))
        if status in _CLOSED and not (_canon(tid) in own_close or rolled_up):
            # round 4 m6: the turn never set this (directly, or by closing its subtasks: the roll-up) — the owner
            # (or a reaper) did, during the turn
            _event(status.lower(), {"by": "other"})
            return AdvanceResult(True, tid, classification, f"task is {status} (set during its turn)")
        if status in _CLOSED:
            summary = str(row.get("result_summary") or row.get("failure_reason") or head)[:400]
            _work_log_step(store, project_id, nxt, outcome=status.lower(),
                           note=f"ran as an agent turn ({origin}, {int(elapsed)} s, {watch.n} calls"
                                + (f", then {exit_kind}" if exit_kind else "") + ")")
            _event(status.lower())
            try:
                from .project_advancer import _note_misquotes_if_done
                await _note_misquotes_if_done(store, plan, project_id, nxt)
            except Exception:  # noqa: BLE001
                logger.debug("misquote note skipped", exc_info=True)
            return AdvanceResult(True, tid, classification, summary or status.lower())

        node = plan.tree.nodes.get(tid)
        if node is not None and list(getattr(node, "children", None) or []):
            # the turn split the task: its subtasks are the work now
            _release("split into subtasks by its turn")
            _work_log_step(store, project_id, nxt, outcome="split",
                           note="ran as an agent turn; split into subtasks")
            _event("split")
            return AdvanceResult(True, tid, classification,
                                 f"split into {len(node.children)} subtask(s) by its turn")
        if unattended and watch.kept_refused:
            _why = (f"needs a kept (production) machine — the owner's run deploys it; {_HELD_MARK}")
            plan.update_status(tid, TaskStatus.NEEDS_USER, result=f"{head[:300]} — {_why}",
                               wait=("owner_turn", "kept_lab", _why))
            store.log_event(project_id, tid, "autoadvance_needs_user",
                            {"description": nxt.description, "reason": _why, "hold": "kept_lab"})
            _work_log_step(store, project_id, nxt, outcome="held", note="ran as an agent turn; " + _why)
            _event("held", {"hold": "kept_lab"})
            return AdvanceResult(True, tid, "needs_user", _why)
        if status == "IN_PROGRESS":
            _why = f"the task's turn ended without closing it: {head or '(no reply)'}"
            plan.update_status(tid, TaskStatus.FAILED, failure_reason=_why[:500])
            store.log_event(project_id, tid, "autoadvance_failed", {"reason": _why[:500], "via": "task_turn"})
            _work_log_step(store, project_id, nxt, outcome="failed", note=_why[:300])
            _event("unclosed")
            return AdvanceResult(True, tid, classification, _why[:400])
        # someone else moved it (the owner's console, a reaper): the turn did its part
        _event(status.lower() or "unknown", {"by": "other"})
        return AdvanceResult(True, tid, classification, f"task is {status or 'gone'} (set during its turn)")

    reply = ""
    exit_kind = ""
    exc: Optional[BaseException] = None
    try:
        reply = await run_leaf_turn(
            # review m1: the background lane (yield to every foreground request) only for idle autopilot — a
            # scheduled or probe batch IS a request, and its own turn waited on itself there
            context, leaf_id=f"ptask-{str(tid)[:12]}", prompt=prompt, is_background=(unattended and not outer),
            max_turns=TASK_TURN_MAX_TURNS, timeout_s=budget_s, project_id=project_id,
            allowed_tools=None, origin=origin, request_id=rid, label=TASK_TURN_LABEL,
            leaf_end=unattended, prepare_agent=_prepare(watch))
    except asyncio.CancelledError:
        # §4UL re-review m2 / round 4 m3: a cancelled turn (shutdown, the batch's request cancelled) leaves each
        # refused DONE held — the task waits for the owner, it is not left claimed as if nothing asked
        try:
            _settle_holds()
        finally:
            _release_subtasks()
            _flush_idle()
        raise
    except asyncio.TimeoutError:
        exit_kind = "timeout"
    except Exception as e:  # noqa: BLE001
        from .llm import BackgroundDeferred
        exit_kind, exc = ("deferred" if isinstance(e, BackgroundDeferred) else "crash"), e
        if exit_kind == "crash":
            logger.warning("task turn crashed on %s: %s: %s", tid, type(e).__name__, e)
    if not exit_kind and not outer and is_deferral_reply(reply):
        # round 8: a BackgroundDeferred / LaneDeferred raised in the leaf's model call reaches here as agent.py's
        # catch-all reply — a deferral even with foreground_requests == 0 (an outside call on the main slot, the
        # slot kept, the owner's request ending before the unwind), never a model-server failure that rests
        # autopilot for the day. Inside an outer request the turn's calls are not background: not a deferral
        exit_kind = "aborted"
    if not exit_kind and "[ATTEMPT_ABORTED_UPSTREAM]" in str(reply or ""):
        # round 6: a deferral only when the owner's request really took the slot — an abort from a model-server
        # error is an upstream failure the advancer scores (no progress → rests), never a free retry every tick.
        # Round 7: inside an outer request (a scheduled or probe batch counts itself in foreground_requests) the
        # turn's model calls are not background — an owner's request never aborts them, so it is the server.
        # The owner's own run too (round 7): reopened, never FAILED over a server error
        exit_kind = "aborted" if (not outer and _foreground_busy(context)) else "upstream"
    try:
        res = await _finish(reply, exit_kind, exc)
    except asyncio.CancelledError:
        _settle_holds()                         # round 4 m3: cancelled while finishing — the holds still land
        raise
    except Exception as e:  # noqa: BLE001 — round 5: a store error in the bookkeeping never leaves the task claimed
        logger.warning("task turn: its outcome was not recorded for %s (%s: %s)", tid, type(e).__name__, e)
        _settle_holds()
        _release(f"task turn's outcome not recorded: {type(e).__name__}")
        res = AdvanceResult(True, tid, "blocked", f"the task's turn ended but its outcome was not recorded "
                                                  f"({type(e).__name__}: {e}) — task left open")
    finally:
        _release_subtasks()
        _flush_idle()
    # round 5 M3: the advancer books the daily allowance from this, once — any call ran = the turn worked
    res.worked = bool(watch.n > 0 or exit_kind in ("", "timeout"))
    return res
