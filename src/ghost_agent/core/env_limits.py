"""§4UU-4 — what the sandbox forbids, remembered for the rest of the request.

Live (req 09c9b24e): `su postgres -c …` and `setpriv --reuid …` failed with
"Operation not permitted" at steps 12 and 13, and the same `su` came back at
step 41. The pre-flight guard only blocks a BYTE-IDENTICAL re-issue, and every
retry had different arguments — but the sandbox's root has a reduced capability set,
so no argument makes a privilege switch work there.

The rule, deterministic:

* An `execute` that exits non-zero with a privilege refusal in its output
  ("Operation not permitted", "must be run as root", "not in the sudoers", …)
  records each PRIVILEGE program its command ran (`su`, `sudo`, `setpriv`, …)
  for this request.
* A later `execute` whose command runs one of those programs again is not run:
  the model gets the earlier error and the ways that do work (run as the current
  user, a purpose-built tool, a lab machine — labs give root).

Only programs in `PRIV_PROGRAMS` are ever recorded: a plain program that failed
with EPERM on one path may succeed on another, so it is never blocked. The memo
is per request and bounded. Pure functions + one in-memory dict; no I/O.
"""
from __future__ import annotations

import re
import shlex
import threading
from collections import OrderedDict
from typing import Dict, List

#: programs whose only job is to change identity / privilege — no argument makes them work for an unprivileged user
#: (`mount`, `sysctl`, `unshare` are NOT here: listing mounts, reading a sysctl and `unshare -r` work unprivileged)
PRIV_PROGRAMS = frozenset({
    "su", "sudo", "doas", "setpriv", "runuser", "gosu", "su-exec", "chroot", "nsenter", "capsh",
    "setcap", "umount", "modprobe", "insmod", "iptables", "ip6tables", "nft",
    "setuidgid", "newgrp", "sg", "pkexec",
})

#: the output of a privilege refusal (not a missing file, not a typo)
_EPERM_RE = re.compile(
    r"operation not permitted|\bEPERM\b|must be (?:run as )?root|you must be root|only root can|"
    r"not in the sudoers|is not allowed to run sudo|sudo: (?:a terminal|a password) is required|"
    r"authentication failure|permission denied \(you must be root\)|setres[ug]id failed|"
    r"setgroups failed|cannot set (?:user|group) id|may not be used by non-root|"
    r"requires? (?:root|superuser) privileges?|are you root\??",
    re.IGNORECASE)

#: words that only prefix the real program
_PREFIX_WORDS = frozenset({"time", "nohup", "exec", "command", "builtin", "env", "nice", "ionice", "stdbuf",
                           "timeout", "xargs"})
#: words that NAME a program without running it (`command -v sudo`, `which su`, `type sudo`)
#: shell keywords / grouping that only lead into the real command (`if …; then sudo x`, `{ sudo x; }`, `! sudo x`)
_SHELL_LEAD = frozenset({"if", "then", "elif", "else", "do", "while", "until", "!", "{", "}", "(", ")", "time",
                         "fi", "done"})
_LOOKUP_WORDS = frozenset({"which", "type", "hash", "whereis", "man", "help", "apropos"})
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: a heredoc opener — not a here-string (`<<<`) and not an arithmetic shift (`$((1<<x))`)
_HEREDOC_RE = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([A-Za-z_][\w.-]*)\1")
_SEP_TOKENS = frozenset({";", "&", "|", "&&", "||", ";;", "(", ")", "$(", "|&"})

_MAX_REQUESTS = 256
_memo: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
_lock = threading.Lock()


def _segment_program(words: List[str]) -> str:
    """The program one shell segment runs ("" when none): leading `X=y` assignments and prefix words
    (`time`, `env`, `timeout 30`, `nice -n 5` …) are skipped; a lookup (`command -v x`, `which x`) runs nothing.
    The basename is returned."""
    i = 0
    while i < len(words):
        w = words[i]
        if _ASSIGN_RE.match(w):
            i += 1
            continue
        if w in _SHELL_LEAD:
            i += 1
            continue
        base = w.rsplit("/", 1)[-1]
        # a firewall front-end's variants are the same privilege (`iptables-legacy`, `ip6tables-nft-restore`)
        base = re.sub(r"^((?:ip6?|eb|arp)tables)(?:-(?:legacy|nft))?(?:-(?:restore|save))?$", r"\1", base)
        if base == "command" and i + 1 < len(words) and words[i + 1] in ("-v", "-V"):
            return ""
        if base in _LOOKUP_WORDS:
            return base
        if base == "capsh" and words[i + 1:] and all(w.startswith(("--print", "--decode", "--has-", "--supports",
                                                                        "--help", "--current"))
                                                     for w in words[i + 1:]):
            return "capsh-query"                      # reads the capability sets; never a privilege switch
        if base in _PREFIX_WORDS:
            i += 1
            # skip the prefix word's own options / numeric argument (`timeout 30`, `nice -n 5`, `env -i`)
            while i < len(words) and (words[i].startswith("-") or re.fullmatch(r"\d+[smhd]?", words[i])
                                      or _ASSIGN_RE.match(words[i])):
                if words[i] in ("-n", "-s", "-k", "-u") and i + 1 < len(words):
                    i += 1
                i += 1
            continue
        return base
    return ""


#: shells: a heredoc fed to one is a SCRIPT (`bash <<EOF … sudo … EOF`), its body runs as commands
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})


def _words(text: str) -> List[str]:
    try:
        return shlex.split(text, comments=False, posix=True)
    except ValueError:
        return text.split()


def _feeds_a_shell(line: str, start: int, end: int) -> bool:
    """Whether the heredoc opened at line[start:end] is a shell's stdin: the command it belongs to is a
    shell (`bash <<EOF`, `sudo -u x sh -s <<EOF`), or its output is piped into one (`cat <<EOF | bash`)."""
    head = re.split(r"&&|\|\||;|\|", line[:start])[-1]
    words = _words(head)
    while words and _segment_program(words) in ("sudo", "doas") and len(words) > 1:
        words = words[1:]                           # `sudo -u pg bash <<EOF` — the shell under the privilege word
        while words and words[0].startswith("-"):
            opt = words[0]
            words = words[1:]
            if re.match(r"^-[A-Za-z]*[ugCpUDrt]$", opt) and not opt.startswith("--") and words:
                words = words[1:]                   # the option's value
    if words and _segment_program(words) in _SHELLS:
        # a shell READS its stdin as commands only without a script argument (`bash script.sh <<EOF` is data)
        rest = words[words.index(next(w for w in words if w.rsplit("/", 1)[-1] in _SHELLS)) + 1:]
        if "-s" in rest:
            return True                             # `bash -s arg <<EOF`: the args are $1…, stdin is the script
        plain, skip = [], False
        for w in rest:
            if skip:
                skip = False
            elif re.match(r"^\d*[<>]{1,2}&?$", w):
                skip = True                         # `bash > out.log <<EOF`: the redirect target is no script
            elif not w.startswith("-") and not re.match(r"^\d*[<>]", w):
                plain.append(w)
        return not plain
    for seg in line[end:].split("|")[1:]:
        if _segment_program(_words(seg)) in _SHELLS:
            return True
    return False


def _command_lines(command) -> List[str]:
    """The command's lines with heredoc BODIES removed (their text is data — `cat > run.sh <<EOF`) unless
    the heredoc feeds a shell (`bash <<EOF`, `cat <<EOF | sh`): then the body is the commands that run.
    Backslash continuations are joined."""
    out: List[str] = []
    lines = str(command or "").replace("\\\n", " ").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        opens = [m for m in _HEREDOC_RE.finditer(line)
                 if line[:m.start()].count("((") <= line[:m.start()].count("))")]   # not `$(( 1 << x ))`
        i += 1
        for m in opens:
            keep = _feeds_a_shell(line, m.start(), m.end())
            while i < len(lines) and lines[i].strip() != m.group(2):
                if keep:
                    out.append(lines[i])
                i += 1
            i += 1                                  # the delimiter line itself
    return out


def programs_in(command) -> List[str]:
    """Every program the shell `command` runs, in order — one per segment of `&&`, `||`, `;`, `|`, `&`, a
    newline or `$(`. Quoted text and heredoc bodies are data: `echo "no sudo here"` runs `echo` only."""
    out: List[str] = []
    for line in _command_lines(command):
        try:
            lex = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
            lex.whitespace_split = True
            lex.commenters = "#"
            toks = list(lex)
        except ValueError:
            toks = re.split(r"\s+|(?=[;&|()])|(?<=[;&|()])", line)
            toks = [t for t in toks if t]
        seg: List[str] = []
        for t in toks + [";"]:
            if t in _SEP_TOKENS or (t and set(t) <= set(";&|()")):
                if seg:
                    prog = _segment_program(seg)
                    if prog:
                        out.append(prog)
                seg = []
            elif t != "$":
                seg.append(t)
    return out


def privilege_refused(output) -> bool:
    """Does this program output read as a privilege refusal?"""
    return bool(_EPERM_RE.search(str(output or "")))


def _first_refusal_line(output) -> str:
    for ln in str(output or "").splitlines():
        if _EPERM_RE.search(ln):
            return ln.strip()[:200]
    return ""


def note_result(req_id, command, output, exit_code) -> List[str]:
    """Record the privilege programs of a FAILED command whose output is a privilege refusal. Returns the
    programs recorded now (empty when nothing was)."""
    rid = str(req_id or "")
    if not rid or exit_code in (0, None) or not privilege_refused(output):
        return []
    ran = [p for p in programs_in(command) if p in PRIV_PROGRAMS]
    if not ran:
        return []
    line = _first_refusal_line(output)
    # blame only the program the refusal names (`su: Authentication failure`, `sudo: … not in the sudoers`);
    # when it names none, only a command with ONE privilege program can be blamed
    def _named(p):
        return re.search(rf"(?:^|[\s/:(]){re.escape(p)}(?:[:\s]|$)", str(line or output or ""), re.M)
    named = [p for p in dict.fromkeys(ran) if _named(p)]
    # the refusal names another program of the command (`kill: … Operation not permitted`): not the privilege one's
    other = [p for p in dict.fromkeys(programs_in(command)) if p not in PRIV_PROGRAMS and _named(p)]
    progs = named or (ran[:1] if len(set(ran)) == 1 and not other else [])
    if not progs:
        return []
    with _lock:
        row = _memo.get(rid)
        if row is None:
            row = {}
            _memo[rid] = row
            while len(_memo) > _MAX_REQUESTS:
                _memo.popitem(last=False)
        else:
            _memo.move_to_end(rid)
        for p in progs:
            row.setdefault(p, line)
    return progs


def refusal(req_id, command) -> str:
    """The block message for a command that runs a privilege program this request already saw refused, else ""."""
    rid = str(req_id or "")
    if not rid:
        return ""
    with _lock:
        row = dict(_memo.get(rid) or {})
    if not row:
        return ""
    for p in programs_in(command):
        if p in row:
            err = row[p]
            return (f"SYSTEM BLOCK — not run: `{p}` already failed in this request"
                    + (f" with \"{err}\"" if err else "")
                    + ". The sandbox's root has a reduced capability set (no network admin, system time, mounts "
                      "or modules): no arguments, variant (`iptables-legacy`) or user switch adds one, so do not "
                      "retry it. Run the program as the current user (a database server: "
                      "start it as the current user with its own data directory); if you have the `lab` tool, "
                      "a lab machine gives root (`lab create`, then `lab exec`).")
    return ""


#: operations no user of this sandbox can do — its root lacks the capability (ft live 11-18, twice: `date -s` refused,
#: then the model built a faketime / LD_PRELOAD shim so `date` printed 2030 and replied "Done — and it works")
#: a program position: line/segment start, or after sudo / then / do (never a word inside an argument)
_AT = (r"(?:^|[;&|(!{]\s*|\bsudo\s+(?:-\S+\s+)*|\bthen\s+|\bdo\s+|^\s+)"
       r"(?:[A-Za-z_]\w*=\S*\s+|env\s+(?:-\S+\s+)*(?:\S+=\S*\s+)*|timeout\s+\S+\s+|nice\s+(?:-n\s*\S+\s+)?)*"
       r"(?:/[\w./-]*/)?")
_CAPABILITY_OPS = (
    ("the system clock", re.compile(_AT + r"(?:date\s+(?:[^;&|]*\s)?(?:-s|--set)\b|date\s+\d{8,12}(?:\.\d\d)?(?:\s|$)|hwclock\s+(?:[^;&|]*\s)?"
                                    r"(?:-w|--systohc|--set)|timedatectl\s+set-)", re.I | re.M)),
    ("network settings", re.compile(_AT + r"(?:ip\s+(?:-\S+\s+)*(?:l|link|a|addr|address|r|route|ru|rule|n|neigh|tuntap)\s+"
                                    r"(?:set|add|del|change|replace|flush)|ifconfig\s+\S+\s+(?:mtu|up\b|down\b|inet\b|"
                                    r"\d|hw\b|netmask)|iptables\s|ip6tables\s|nft\s|tc\s+qdisc\s+(?:add|del|change|replace)|ethtool\s+-s|brctl\s)",
                                    re.I | re.M)),
    ("mounts", re.compile(_AT + r"(?:mount\s+(?:-\S+\s+)*[^\s|;&<>]+\s+[^\s|;&<>]+|umount\s|swapon|swapoff|losetup)", re.I | re.M)),
    ("kernel modules and settings", re.compile(_AT + r"(?:modprobe|insmod|rmmod|sysctl\s+(?:-\S+\s+)*(?:-w\s|-p\b|\S+=))",
                                               re.I | re.M)),
)
#: the command runs a shell script — only then may the refusal LINE name the op (a pytest run printing such a line
#: is a test's output, not this command's operation)
_RUNS_A_SCRIPT_RE = re.compile(r"(?:^|[;&|(]\s*|\bsudo\s+)(?:(?:ba|z|da)?sh\s+(?!-\w*[nc]\b)(?:-\w+\s+)*[\w./-]+|"
                               r"\.?/[\w./-]+\.sh\b)",
                               re.M)
#: refusal lines that name the operation itself (the command only named the script that ran it)
_CAPABILITY_OUTPUT = (
    ("the system clock", re.compile(r"date: cannot set date|settimeofday.*(?:not permitted|EPERM)|"
                                    r"clock_settime.*(?:not permitted|EPERM)|hwclock:.*(?:not permitted|EPERM)", re.I)),
    ("network settings", re.compile(r"RTNETLINK answers: Operation not permitted|SIOCS\w+: Operation not permitted|"
                                    r"iptables.*Permission denied \(you must be root\)", re.I)),
    ("mounts", re.compile(r"^(?:mount|umount|swapon): .*(?:permission denied|not permitted)", re.I | re.M)),
    ("kernel modules and settings", re.compile(r"sysctl: setting key .*(?:not permitted|Read-only|denied)|modprobe: ERROR.*not permitted",
                                               re.I)),
)
_CAP_NOTE = (
    "\n[SANDBOX LIMIT] Changing {what} is impossible in this sandbox: its root has a reduced capability set, so no "
    "command, flag or user switch can do it. Do NOT simulate it (a fake clock, faketime / LD_PRELOAD shims, a wrapper "
    "that only makes the output look right) and present that as done — tell the user plainly it cannot be done here "
    "and what would do it (a machine where that privilege is really granted).")


def capability_note(req_id, command, output, exit_code) -> str:
    """The once-per-request note for a FAILED command whose refusal is a missing capability on an operation no
    sandbox user can do ("" otherwise)."""
    rid = str(req_id or "")
    if not rid or exit_code in (0, None) or not (privilege_refused(output) or
                                      any(rx.search(str(output or "")) for _w, rx in _CAPABILITY_OUTPUT)):
        return ""
    text = "\n".join(re.sub(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"", "''", ln) for ln in _command_lines(command))
    what = next((w for w, rx in _CAPABILITY_OPS if rx.search(text)), "")   # quoted words above are data, not ops
    if not what and _RUNS_A_SCRIPT_RE.search(text):  # the op ran inside a script (`bash settime.sh`): its refusal line
        out = str(output or "")
        what = next((w for w, rx in _CAPABILITY_OUTPUT if rx.search(out)), "")
    if not what:
        return ""
    with _lock:
        row = _memo.get(rid)
        if row is None:
            row = {}
            _memo[rid] = row
            while len(_memo) > _MAX_REQUESTS:
                _memo.popitem(last=False)
        else:
            _memo.move_to_end(rid)
        key = "\0cap:" + what
        if key in row:
            return ""
        row[key] = ""
    return _CAP_NOTE.format(what=what)


def forget(req_id) -> None:
    """Drop a request's memo (end of request)."""
    with _lock:
        _memo.pop(str(req_id or ""), None)


def _reset_for_tests() -> None:
    with _lock:
        _memo.clear()
