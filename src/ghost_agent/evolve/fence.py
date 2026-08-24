"""What a candidate may touch, and what scores it — the write fence.

THE FAILURE MODE THIS EXISTS FOR is not "the mutator writes bad code";
that is what the evaluator is for. It is **the agent modifying the thing
that measures it**. DGM's agent learned to fabricate test logs; the AI
Scientist raised its own timeout rather than finishing in it. Neither was
caught by a better mutation — both were caught (eventually) by someone
noticing the evaluator had moved.

So the fence is two separate mechanisms, and neither is sufficient alone:

* **A path ALLOW-LIST** (`is_mutable`) — a candidate diff that touches
  anything outside it is rejected at stage 0, before it is built. An
  allow-list rather than a deny-list because the repo grows: a module
  added next month must be un-mutable until someone decides otherwise.
* **A harness CHECKSUM** (`harness_digest`) — taken before AND after
  every evaluation stage. The allow-list can only reject what it can see
  in a diff; a candidate that reaches the filesystem some other way
  (a test fixture that writes, a conftest import with a side effect)
  moves the checksum. A mismatch aborts the generation and alarms.

⚠ These two guard DIFFERENT things and the second is the one that
matters. A path check is a claim about the diff; a checksum is a fact
about the files.

⚠ STATE OF THIS FILE, 2026-08-22. `is_mutable` /
`check_diff_scope` / `resolves_to_immutable` ARE wired — the
mutator calls them on every candidate. `harness_digest` /
`compare_harness` are BUILT AND UNCALLED: they exist for the
evaluation cascade (E2), which does not exist yet, so nothing
takes a digest, nothing compares one, nothing aborts and
nothing alarms. Said here in the present tense once, which is
how a reader ends up believing the load-bearing half is live.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger("GhostAgent")

#: Repo-relative path PREFIXES a candidate may modify. Deliberately
#: narrow for v1: tool implementations and prompt constants are where a
#: scaffold improvement plausibly lives, and both are heavily covered by
#: existing tests.
MUTABLE_PREFIXES: Tuple[str, ...] = (
    "src/ghost_agent/tools/",
    "src/ghost_agent/core/prompts.py",
    "src/ghost_agent/reflection/prompts.py",
    "src/ghost_agent/reflection/postmortem_prompts.py",
    "src/ghost_agent/optim/signatures.py",
)

#: Paths that are NEVER mutable even if a prefix above would admit them.
#: Every entry is a thing that either scores a candidate, contains the
#: fence itself, or is load-bearing for containment.
IMMUTABLE_EXACT: Tuple[str, ...] = (
    #: ⚠ `src/__init__.py` DOES NOT EXIST TODAY, AND THAT IS THE POINT.
    #: `src/` is a PEP-420 namespace package; CREATING this file makes
    #: it a regular package whose body runs on every
    #: `import src.ghost_agent.*` — the production import shape. It was
    #: refused only by the catch-all ("not on the mutable allow-list")
    #: and `resolves_to_immutable` returned "" for it, while the entry
    #: one level down (`tools/__init__.py`) is listed here by name for
    #: exactly this hazard.
    "src/__init__.py",
    "src/ghost_agent/tools/registry.py",   # the tool surface itself
    "src/ghost_agent/tools/validators.py",  # argument validation
    # A 0-byte package init inside the mutable prefix is an import-time
    # code-execution seam: anything added here runs in EVERY process that
    # imports any tool, including a future evaluator. A candidate must
    # not be able to put code somewhere the harness executes.
    "src/ghost_agent/tools/__init__.py",
)

#: Trees the candidate must not touch under any circumstances. The first
#: four are the evaluator; the rest are containment and identity.
IMMUTABLE_PREFIXES: Tuple[str, ...] = (
    "tests/",
    "src/ghost_agent/eval/",
    "scripts/",
    "src/ghost_agent/evolve/",
    "src/ghost_agent/core/agent.py",
    "src/ghost_agent/core/isolation.py",
    "src/ghost_agent/core/replay_engine.py",
    "src/ghost_agent/core/imagination.py",
    "src/ghost_agent/core/admissibility.py",
    "src/ghost_agent/core/experiments.py",
    "src/ghost_agent/memory/",
    "src/ghost_agent/selfhood/",
    "src/ghost_agent/sandbox/",
    "src/ghost_agent/main.py",
    # (No repo-root `conftest.py` exists — the real one is
    # `tests/conftest.py`, covered by the `tests` tree. Kept as
    # forward-cover: the day someone adds one, it is watched from the
    # first commit rather than from the first incident.)
    "conftest.py",
    "pytest.ini",
)

#: What the harness digest covers. If a candidate can change any of
#: this, it can change its own score.
#: ⚠ THE DIGEST MUST COVER EVERYTHING THE FENCE CALLS IMMUTABLE. It did
#: not: 13 of 18 immutable entries were unwatched, including
#: `core/dream.py` — the module the stage-3 INCUMBENT arm runs. The
#: cleanest attack was never to inflate the candidate but to DEFLATE the
#: baseline: write to the canonical `core/` during a pin run and stage
#: 3's "incumbent" is no longer the incumbent. The fence declared those
#: paths immutable for the DIFF; the digest was supposed to be what
#: catches out-of-band writes, and for those paths it did not exist.
HARNESS_TREES: Tuple[str, ...] = (
    "tests",
    "src/ghost_agent/eval",
    "scripts",
    "src/ghost_agent/evolve",
    "pytest.ini",
    "conftest.py",
    # ⚠ THE WHOLE PACKAGE, NOT THE FENCE'S IMMUTABLE LIST. Watching the
    # paths the fence calls immutable answers the wrong question: what
    # matters is what the INCUMBENT ARM EXECUTES. Measured: of the 66
    # modules `import ghost_agent.core.dream` loads, 42 were unwatched
    # — `ghost_agent/__init__.py` (which runs in every process that
    # imports anything), all of `utils/`, `distill/`, `reflection/`, and
    # 22 tool modules. Those files were on NEITHER list: no diff may
    # touch them, and an out-of-band write to them was invisible. That
    # is precisely the gap this file's own docstring describes — "a path
    # check is a claim about the diff; a checksum is a fact about the
    # files" — reopened by scoping the checksum to the path check.
    #
    # The mutable tree is watched too. A candidate's own diff moves it,
    # which the cascade compares against the diff it approved; a write
    # that arrives any other way has no such alibi.
    "src/ghost_agent",
)


def _norm(path: str) -> str:
    """Repo-relative POSIX form, with traversal collapsed.

    A diff header can carry `a/src/../../etc/passwd` or a Windows
    separator; normalising BEFORE the prefix test is the difference
    between a fence and a substring check."""
    p = str(path or "").strip().replace("\\", "/")
    for pref in ("a/", "b/", "./"):
        while p.startswith(pref):
            p = p[len(pref):]
    p = os.path.normpath(p).replace("\\", "/")
    return p


def _case_fold(path: str) -> str:
    """Lower-cased form, for comparisons that must survive a
    case-insensitive filesystem.

    ⚠ MEASURED TWICE, NOT HYPOTHETICAL, AND `.lower()` WAS NOT ENOUGH.

    Round 1 of this bug: the volume is case-insensitive APFS, so
    `Registry.py` and `registry.py` are the same file — and an exact,
    case-SENSITIVE deny-list admitted `Registry.py`, which cleared
    `is_mutable`, cleared `check_diff_scope`, cleared `archive.admit`,
    and let `patch` write the real `tools/registry.py`: the first entry
    on the immutable list, "the tool surface itself".

    Round 2, after `.lower()` was applied: the filesystem's equivalence
    is UNICODE FOLD, not ASCII lower. `regiſtry.py` (U+017F LATIN SMALL
    LETTER LONG S) opens the existing `registry.py` — and `.lower()`
    leaves `ſ` alone while `.casefold()` maps it to `s`. Verified end to
    end in a scratch repo: `is_mutable` True, `check_diff_scope` clear,
    `validate_diff` clear, `archive.admit` ADMITTED, and the patched
    snapshot contained a modified `registry.py`. Same for `validatorſ.py`.

    So: `.casefold()`, which for ASCII paths is identical to `.lower()`
    and therefore does not narrow the allow-list. And because a string
    transform can only ever APPROXIMATE the filesystem's own equivalence
    relation, `resolves_to_immutable()` asks the filesystem directly —
    the string test is the cheap half, not the whole answer.
    """
    return str(path or "").casefold()


def resolves_to_immutable(path: str, repo_root) -> str:
    """Why `path` names a file the fence forbids, or "" — asked of the
    FILESYSTEM rather than of a string transform.

    A case-fold is a guess about the volume's equivalence relation. On a
    case-insensitive, unicode-folding filesystem the only thing that
    knows which file a name opens is the filesystem. When the target
    exists, compare REAL paths; anything that resolves onto an immutable
    file is refused however it was spelled. Falls back silently when the
    path does not exist yet (a new file cannot alias an existing one).
    """
    try:
        root = Path(repo_root).resolve()
        rel = _norm(path)
        if not rel or rel.startswith("..") or Path(path).is_absolute():
            # ⚠ FAIL CLOSED. This returned "" — no objection — for
            # `../../../etc/passwd`. Unreachable today because
            # `validate_diff` runs the string fence first, but
            # `materialize` never calls `is_mutable`, and the comment
            # beside the call reads as if this check subsumes it.
            return f"{path} escapes the repo"
        target = (root / rel)
        if not target.exists():
            return ""
        # `os.path.samefile` compares device+inode — the filesystem's own
        # answer. `Path.resolve()` is NOT enough: on macOS it preserves
        # the spelling it was given, so `regiſtry.py` resolves to
        # `…/regiſtry.py` and compares unequal to `…/registry.py` while
        # opening the same bytes.
        for entry in tuple(IMMUTABLE_EXACT) + tuple(
                p for p in IMMUTABLE_PREFIXES if not p.endswith("/")):
            other = (root / entry)
            if not other.exists():
                continue
            try:
                if os.path.samefile(str(target), str(other)):
                    return (f"{path} IS {entry} on this filesystem, which is "
                            f"immutable — the volume folds them onto one "
                            f"file")
            except OSError:
                continue
        for prefix in IMMUTABLE_PREFIXES:
            if not prefix.endswith("/"):
                continue
            base = (root / prefix.rstrip("/"))
            if not base.exists():
                continue
            for existing in base.rglob("*"):
                if not existing.is_file():
                    continue
                try:
                    if os.path.samefile(str(target), str(existing)):
                        return (f"{path} IS {existing.relative_to(root)}, "
                                f"inside the immutable tree {prefix}")
                except OSError:
                    continue
    except Exception:              # noqa: BLE001
        # A resolution failure is not permission to write.
        return f"{path} could not be resolved against the repo"
    return ""


def is_mutable(path: str) -> Tuple[bool, str]:
    """May a candidate modify ``path``? Returns (allowed, why-not).

    Fails CLOSED on anything it cannot classify: an absolute path, a path
    that escapes the repo, an empty string. "I could not tell" must never
    read as "allowed" for a write fence.
    """
    raw = str(path or "").strip()
    if not raw:
        return False, "empty path"
    if raw.startswith("/") or raw.startswith("~"):
        return False, "absolute path"
    p = _norm(raw)
    if not p or p.startswith("..") or p == ".":
        return False, "path escapes the repo"
    # CASE-FOLDED on both sides. The filesystem this runs on is
    # case-insensitive, so an exact deny-list that is case-sensitive is
    # not a deny-list — see `_case_fold`.
    low = _case_fold(p)
    if (low.startswith(tuple(_case_fold(x) for x in IMMUTABLE_PREFIXES))
            or low in {_case_fold(x) for x in IMMUTABLE_EXACT}):
        return False, f"{p} is immutable (it scores or contains the candidate)"
    if not low.startswith(tuple(_case_fold(x) for x in MUTABLE_PREFIXES)):
        return False, f"{p} is not on the mutable allow-list"
    return True, ""


def check_diff_scope(paths: Iterable[str]) -> Tuple[bool, List[str]]:
    """(allowed, rejections) for every path a candidate diff touches.

    An EMPTY path set is rejected: a diff that touches nothing is not a
    candidate, and treating it as allowed is how a no-op proposal reaches
    an operator carrying a claim of improvement."""
    rejects = []
    seen = 0
    for path in paths or ():
        seen += 1
        ok, why = is_mutable(path)
        if not ok:
            rejects.append(why)
    if not seen:
        return False, ["the diff touches no files"]
    return (not rejects), rejects


def harness_digest(repo_root: Path = None,
                   trees: Iterable[str] = None) -> Dict[str, str]:
    """sha256 over every file in the harness trees, keyed by path.

    A DICT rather than a single hash on purpose: when it moves, the
    operator needs to know WHICH file moved. A single digest says only
    that something did, which is the least actionable form of a true
    alarm.

    Missing trees are skipped silently — a fresh checkout without
    `scripts/` is not tampering. What is NOT silent is a file that
    existed at the first digest and does not at the second; the compare
    below reports that as a removal.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    out: Dict[str, str] = {}
    for tree in (trees or HARNESS_TREES):
        base = root / tree
        if base.is_file():
            files = [base]
        elif base.is_dir():
            # ⚠ BYTECODE IS WATCHED TOO. Excluding `__pycache__`/`.pyc`
            # left the one artefact Python imports AHEAD OF SOURCE
            # invisible to the check whose whole job is to notice the
            # harness changing. An unchecked-hash `.pyc` is loaded
            # without consulting the `.py` at all, so a planted one
            # replaces a module the digest still reports as pristine.
            # The cascade's own children never write bytecode
            # (PYTHONDONTWRITEBYTECODE), and the before/after window is
            # minutes, so a `.pyc` appearing inside it is exactly the
            # event this exists to catch — including the benign version
            # (an operator running pytest concurrently), which is also a
            # harness moving under a run and worth aborting on.
            # ⚠ `is_file()` FOLLOWS SYMLINKS AND SO MISSES THEM. A
            # DANGLING symlink is not a file, and `rglob` does not
            # descend into a symlinked DIRECTORY — so both could be
            # ADDED inside a watched, immutable tree and stay
            # permanently invisible. That is worse than the
            # modify-use-restore window this docstring concedes: it is a
            # permanent addition the checksum can never see. Combined
            # with a runner check that only inspects spelling, it yields
            # an executable outside the canonical tree that the digest
            # never notices. Enumerate entries, not files.
            files = sorted(p for p in base.rglob("*")
                           if p.is_symlink() or p.is_file())
        else:
            continue
        for f in files:
            try:
                if f.is_symlink():
                    # Hash the LINK, not its target: the target may sit
                    # outside every watched tree, and re-pointing the
                    # link is exactly the move to catch.
                    h = "SYMLINK:" + hashlib.sha256(
                        os.readlink(f).encode("utf-8")).hexdigest()
                else:
                    h = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError as exc:
                # Unreadable is NOT skippable: a file the digest cannot
                # read is a file the digest cannot protect.
                h = f"UNREADABLE:{type(exc).__name__}"
            out[str(f.relative_to(root)).replace("\\", "/")] = h
    return out


def compare_harness(before: Dict[str, str],
                    after: Dict[str, str]) -> List[str]:
    """Files that changed, appeared or vanished between two digests.

    Additions count. A test file a candidate ADDS is a test file that
    runs, and "it only added things" is exactly the shape of a candidate
    writing its own passing test."""
    changed = []
    for path in sorted(set(before) | set(after)):
        b, a = before.get(path), after.get(path)
        if b == a:
            continue
        # ⚠ BYTECODE IS NAMED, NOT EXCUSED. An earlier version dropped
        # `.pyc` from the digest entirely so it would not "flap" — which
        # made the check decorative, since a `.pyc` is imported AHEAD OF
        # the source being hashed. The alarm-fatigue concern behind that
        # was real, so the answer is a distinguishable label: an
        # operator can see at a glance whether SOURCE moved or only
        # bytecode did, and both still abort the generation, because
        # both mean the harness changed under a run in progress.
        kind = "BYTECODE " if (path.endswith(".pyc")
                               or "__pycache__" in path) else ""
        if b is None:
            changed.append(f"{kind}ADDED {path}")
        elif a is None:
            changed.append(f"{kind}REMOVED {path}")
        else:
            changed.append(f"{kind}MODIFIED {path}")
    return changed


__all__ = [
    "MUTABLE_PREFIXES", "IMMUTABLE_PREFIXES", "IMMUTABLE_EXACT",
    "HARNESS_TREES", "is_mutable", "check_diff_scope",
    "harness_digest", "compare_harness",
]
