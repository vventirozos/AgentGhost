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

# Patterns that we flat-out refuse to dispatch. These are the
# canonical "you almost certainly do not want this" forms — exotic
# delete-the-whole-disk variants. Anchored at word boundaries so a
# benign substring ("description") doesn't trip the deny-list.
_SHELL_DENY: tuple = (
    # rm with BOTH -r and -f flags (any order/spelling, combined or split)
    # targeting a DANGEROUS path: absolute (/...), root glob (/*), home (~ /
    # $HOME), or a quoted form of those. Relative deletes (rm -rf ./build)
    # are intentionally NOT blocked. The two lookaheads require an `r` flag
    # and an `f` flag in the rm invocation before the target.
    re.compile(
        # long options too (fresh review: `rm --recursive --force /…` passed)
        r"\brm\b(?=[^|;&]*\s(?:-\w*r|--recursive))(?=[^|;&]*\s(?:-\w*f|--force))[^|;&]*\s+"
        r"(?:/(?:\s|$|\*|\w[^\s'\"]*)|~(?:\s|$|/)|\$\{?HOME\}?"
        r"|['\"]\s*(?:/[^'\"]*|~[^'\"]*|\$\{?HOME\}?)['\"])",
        re.IGNORECASE,
    ),
    # dd to a raw device
    re.compile(r"\bdd\b[^|;&]*of=/dev/(?:sd|nvme|hd)"),
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
        r"\b(?:curl|wget|fetch)\b[^|]+\|\s*(?:sudo\s+)?"
        r"(?:sh|bash|zsh|dash|ksh)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:curl|wget|fetch)\b[^|]+\|\s*(?:sudo\s+)?"
        r"(?:python[0-9.]*|perl|ruby|node|php)\s*(?:-\s*)?(?:\||;|&|$)",
        re.IGNORECASE,
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


def validate_shell(cmd: str) -> Tuple[bool, str]:
    """Validate a shell command's shape and reject obviously-destructive
    forms. Returns ``(ok, reason)``.

    ``ok=True`` means "shape looks plausible, not on the deny list"
    — NOT "this command is safe in your environment". The host-level
    sandbox is still the authoritative safety boundary.
    """
    if not cmd or not cmd.strip():
        return False, "empty command"
    s = cmd.strip()
    # Shape check: must shlex-parse. Unclosed quotes are the most common
    # LLM emission bug ("echo 'hello world").
    try:
        tokens = shlex.split(_strip_heredoc_bodies(s), posix=True)
    except ValueError as e:
        return False, f"shell syntax: {e}"
    if not tokens:
        return False, "empty after parsing"
    # Deny list
    for pat in _SHELL_DENY:
        if pat.search(s):
            return False, f"deny-listed pattern: {pat.pattern[:60]}"
    return True, ""


__all__ = ["validate_shell"]
