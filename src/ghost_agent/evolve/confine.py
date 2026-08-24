"""Confine a graded child so it can only write where the parent says.

⚠ WHY THIS EXISTS. Stages 1-3 of the cascade run CANDIDATE CODE. Five
review rounds found the same shape five times: a fix closed a channel
(the exit code, automatic bytecode, the digest's scope, an outbox that
could be appended to) while leaving the MEDIUM — a process with the
parent's own filesystem privileges — untouched, and the next round found
another route to the same place. The routes measured, each end to end:

* a stage-1 pin run wrote the canonical `scripts/evolve_bench_child.py`
  and stage 1 still returned `passed=True`;
* a candidate appended 2,000 losing rows to the operator's bench ledger,
  moving its own stage-2 floor from 0.93 to 0.04;
* a stage-2 child planted both stage-3 arms' result files, giving
  `PROMOTABLE: True, p=0.00049` from two children that wrote nothing;
* when appending was refused, a detached grandchild renamed a forged
  file over the incumbent arm's outbox instead (`os.replace`), giving
  `p=0.0` with every pre-run check intact;
* a `.pyc` planted in a watched tree, imported ahead of the source the
  digest hashes.

Every one of those is a WRITE the parent never intended and cannot see
until afterwards. So stop auditing writes and stop allowing them.

**What this closes.** On macOS, `sandbox-exec` applies a kernel-enforced
policy that a process cannot lift, and that its children — including a
`start_new_session=True` grandchild — inherit. Measured under this
module's own policy: writing outside the allow-list, `os.replace` over a
sibling arm, and a detached grandchild's write all fail with
`PermissionError`, and the grandchild fails *after* its parent exited.

**What it does NOT close, and never can.** The child still produces its
own result. It can lie about ITS OWN arm — its stdout, its exit code,
its own results file — because those are what it is for. What it can no
longer do is touch the harness that judges it, the incumbent's answers,
the operator's ledger, or anything belonging to another stage. That is
the difference between "a candidate may misreport itself" and "a
candidate may rewrite the exam".

⚠ **AND IT IS PLATFORM-SPECIFIC.** There is no seatbelt off macOS. This
module does not pretend otherwise: `confine()` returns the command
UNCHANGED with `confined=False`, and every caller records that in the
stage's detail. A cascade that silently ran unconfined while its
docstring said otherwise would be worse than one that never had this.
"""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

#: Set to "0" to run the children unconfined. It exists for diagnosing a
#: policy that refuses an honest candidate — NOT as a normal mode, and
#: the stage records which way it ran either way.
ENV_FLAG = "GHOST_EVOLVE_CONFINE"

_SEATBELT = "/usr/bin/sandbox-exec"


def available() -> Tuple[bool, str]:
    """(can we confine, why not)."""
    if str(os.environ.get(ENV_FLAG, "1")).strip().lower() in ("0", "false",
                                                              "no"):
        return False, f"{ENV_FLAG} is off"
    if platform.system() != "Darwin":
        return False, f"no seatbelt on {platform.system()}"
    if not (Path(_SEATBELT).exists() and shutil.which(_SEATBELT)):
        return False, "sandbox-exec is not present"
    return True, ""


def _both_spellings(p: Path) -> List[str]:
    """`/tmp` is a symlink to `/private/tmp` on macOS and the policy is
    matched against the RESOLVED path — but a caller may hand us either,
    so allow both. Getting this wrong denies a write the parent meant to
    permit, which reads as a hostile candidate."""
    raw = str(Path(p))
    out = [raw]
    try:
        # ⚠ THE RESOLVED FORM IS THE ONE THAT MATTERS, and it is enough.
        # `/tmp` is a symlink to `/private/tmp`, and the kernel matches
        # the policy against the path a write RESOLVES to — so allowing
        # `/private/tmp/x` also permits a child writing `/tmp/x`. Two
        # extra string branches for the `/tmp` <-> `/private/tmp`
        # spellings were carried here until a mutation showed deleting
        # them changed nothing: `Path("/tmp/x").resolve()` already
        # returns `/private/tmp/x`. Dead guard code reads as defence.
        real = str(Path(p).resolve())
        if real != raw:
            out.append(real)
    except OSError:
        pass
    return sorted(set(out))


def policy(writable: Iterable[Path], *, allow_network: bool) -> str:
    """A seatbelt policy: read anything, write only where told.

    Reads stay open on purpose. The property being protected is
    INTEGRITY — the candidate must not be able to change what judges it
    — not confidentiality, and a read-tight policy would refuse the pin
    suite for reasons that have nothing to do with the candidate.
    """
    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    if not allow_network:
        lines.append("(deny network*)")
    for w in writable:
        for spelling in _both_spellings(w):
            lines.append(f'(allow file-write* (subpath "{spelling}"))')
    # A process that cannot write /dev/null cannot run pytest.
    lines.append('(allow file-write-data (literal "/dev/null") '
                 '(literal "/dev/stdout") (literal "/dev/stderr") '
                 '(literal "/dev/urandom") (literal "/dev/random"))')
    lines.append("(allow file-write* (subpath \"/dev/fd\"))")
    return "\n".join(lines) + "\n"


def confine(cmd: List[str], *, writable: Iterable[Path],
            allow_network: bool, policy_dir: Path) -> Tuple[List[str], bool,
                                                            str]:
    """Wrap `cmd` so it may only write under `writable`.

    Returns `(cmd, confined, why_not)`. ⚠ The policy file itself must sit
    somewhere the child CANNOT write, or the confinement is advisory:
    `policy_dir` is the parent's, never the candidate's or an arm's.
    """
    ok, why = available()
    if not ok:
        return list(cmd), False, why
    try:
        policy_dir = Path(policy_dir)
        policy_dir.mkdir(parents=True, exist_ok=True)
        pf = policy_dir / "confine.sb"
        pf.write_text(policy(writable, allow_network=allow_network))
    except OSError as exc:
        return list(cmd), False, f"could not write the policy: {exc}"
    return [_SEATBELT, "-f", str(pf), *cmd], True, ""


__all__ = ["available", "confine", "policy", "ENV_FLAG"]
