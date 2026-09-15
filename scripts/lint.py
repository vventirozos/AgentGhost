#!/usr/bin/env python3
"""The ONE lint runner: `scripts/ci.sh` and `tests/test_lint_gate.py` both
call this, so the gate cannot be spelled differently in two places (§4GJ).

WHY THIS EXISTS. The tree had no linter. A four-minute pylint pass over
`src` found two undefined names in "this must not die" exception handlers
(`logger` in core/learning_health.py, `pretty_log` in sandbox/services.py —
both fixed in §4GI), a name read in an `except` handler that the `try` may
never have bound (`_wrote`, tools/file_system.py), a `dspy` fallback passing
a keyword the installed version removed, and two invisible zero-width
characters in source. None of those needs mutation testing to find; they
need a linter that runs.

THE RULE, in two halves:

* **zero-tolerance symbols** — the error classes that are at zero after
  §4GJ. Any finding fails the gate outright, baseline or not. These are the
  checks that found real defects here and produce no false positives in
  this tree.
* **the fingerprint ratchet** — every other error-class finding must appear
  in `tests/lint_baseline.json`, keyed by (symbol, path, normalised
  message) with a count. A NEW fingerprint fails; a HIGHER count for a
  known fingerprint fails; and a baseline that has drifted ABOVE reality
  fails too (§4GJ round 4 — see `ALLOWED_SLACK` below). Line numbers are
  deliberately excluded: they move on every edit, and a baseline that
  churns is a baseline nobody regenerates honestly.

A count-only baseline was rejected: it lets one defect be swapped for
another (fix one, add one, total unchanged) — the shape `pin-must-fail-
somewhere` warns about. The fingerprint set cannot absorb a swap.

THE RATCHET ONLY RATCHETS IF SLACK EXPIRES (§4GJ round 4). The first
version PRINTED stale entries and excluded them from the verdict:
"staleness is advisory". Measured in a scratch repo: baseline a
`no-member`, fix it (rc=0, "stale, please regenerate downward"), then
re-introduce the SAME defect — rc=0. A fixed finding could come back for
free, and 167 fingerprints of standing tolerance never expired. The gate
now fails when the baseline claims more findings than the tree has, so
fixing something ends in `--write-baseline` — the whole point: the
tolerance shrinks with the tree.

⚠ AND THE CEILING ON THAT SLACK IS A CONSTANT IN THIS FILE, NOT A KEY IN
THE BASELINE (§4GJ round 5). Round 4's first version read
`baseline["allowed_slack"]` — the gate asking the file it polices how much
tolerance it was allowed — and `--write-baseline` copied the number
forward, so it was sticky as well as self-granted. Reproduced in a scratch
repo: baseline a finding, fix it, write `"allowed_slack": 9999` into the
baseline, re-introduce the identical finding -> `OK`, rc=0. The pytest half
was bounded by the same self-declared number (`slack_total <=
verdict["allowed_slack"]`), so neither consumer could fail. Round 4 removed
"tolerance that never expires" and replaced it with "tolerance that expires
unless the baseline says otherwise". The committed baseline never carried
the key, so the hole was unexploited — and unguarded. A baseline that
declares a NON-ZERO ceiling now fails the gate by itself, because the only
reason to write that key is to disarm the arm above it.

⚠ AND ROUND 5 ONLY MOVED THE OFF SWITCH (§4GK round 6). `allowed_slack` is
genuinely closed — but the `--seed` guard's memory, `tracked_symbols`, is a
key in the very file the arm polices, and unlike `entries` it was never
checked for shrinkage. Measured in a scratch repo: bootstrap one `no-member`,
fix it, regenerate downward, `--seed no-member` -> rc 1; hand-edit
`"tracked_symbols": []` and the same command -> "seeding 2 new
fingerprint(s)", rc 0, gate OK. Two brand-new findings absorbed into a family
already ratcheted to zero, one hand-edit away, and no record left behind. So
the FLOOR of that history is `TRACKED_FLOOR`, a constant here; a list that
contradicts the file's own entries or seed records fails the gate; and
`--seed` now demands `--seed-reason` and stamps it, the way the pin ratchet's
`--migrate "reason"` has since round 5.

⚠ AND ROUND 6 ONLY MOVED THE FLOOR FOR THE EIGHT FAMILIES IT NAMED (§4GK
round 7). Everything seeded after it is remembered by `tracked_symbols` and
`seeded` alone — both keys in the policed file — and the `history_gaps`
cross-check reads its evidence out of those same two keys, so deleting BOTH
leaves nothing to contradict. Measured: seed a family, fix it, regenerate
downward, drop the two keys, seed it again -> rc 0 and `ok: True`. So the
PERMISSION to seed left the baseline entirely: `SEED_GRANTS` in this file,
normally empty, and a grant that has been spent FAILS the gate until it is
deleted. An upward write costs a code change going in and a code change
coming out.

The remaining baseline entries are pylint inference limits (Optional
narrowing across statements, `__slots__` assigned in `__new__`,
`**kwargs` forwarding, dataclass `__dataclass_fields__`) plus a handful of
real findings owned by other files at the time of writing — each is listed
in the baseline with a `note`, so the next reader sees which is which.

Usage:
    python scripts/lint.py                 # human summary, exit 1 on failure
    python scripts/lint.py --json          # machine summary for the test
    python scripts/lint.py --write-baseline  # regenerate (DOWNWARD only)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "ghost_agent"
#: `GHOST_LINT_BASELINE` overrides the path so a test can drive the writer
#: against a scratch baseline instead of the committed one (§4GJ).
BASELINE = Path(os.environ.get("GHOST_LINT_BASELINE")
                or REPO / "tests" / "lint_baseline.json")

#: How much standing tolerance the baseline may hold for findings the tree
#: no longer has. ZERO, and it lives HERE rather than in the baseline: a
#: gate that reads its own ceiling out of the file it polices has an off
#: switch, and `--write-baseline` made that switch sticky (see the module
#: docstring for the measurement). Raising this is a code change a reviewer
#: sees; raising a JSON key was not.
ALLOWED_SLACK = 0

#: Symbol families this tree has tracked, AS A CONSTANT IN THIS FILE.
#:
#: ⚠ THE OFF SWITCH MOVED, IT DID NOT CLOSE (§4GK round 6). Round 5 took the
#: slack ceiling out of the baseline for exactly the right reason — "a gate
#: that reads its own ceiling out of the file it polices has an off switch" —
#: and then gave the `--seed` guard a memory (`tracked_symbols`) that lives in
#: that same file, and, unlike `entries`, is never checked for shrinkage.
#: Measured against a scratch baseline: bootstrap one `no-member`, fix it,
#: regenerate downward, `--seed no-member` -> rc 1 (the guard works); then
#: hand-edit `"tracked_symbols": []` and `--seed no-member` -> "seeding 2 new
#: fingerprint(s)", rc 0, gate OK. Two brand-new findings absorbed into a
#: family already ratcheted to zero, one hand-edit away, leaving no record in
#: the file. So the floor of the history lives HERE, where widening it is a
#: code change a reviewer sees, and the baseline's own list can only ADD to it.
#:
#: The eight below are the families the committed baseline carries entries
#: for; the zero-tolerance set joins them because a family held at zero is the
#: one it would be worth laundering a finding into.
TRACKED_FLOOR = frozenset({
    "missing-kwoa", "no-member", "no-self-argument", "not-callable",
    "unsubscriptable-object", "unsupported-membership-test",
    "unused-import", "unused-variable",
})

#: The families `--seed` may widen coverage for RIGHT NOW, with the reason.
#: Normally EMPTY, and that is the point.
#:
#: ⚠ THE FLOOR CANNOT BE MADE OF THE THING THE BASELINE CAN DELETE (§4GK
#: round 7). Round 6 moved the seed guard's memory behind `TRACKED_FLOOR`
#: — for the eight families that were in it. Everything seeded AFTER round 6
#: is remembered only by `tracked_symbols` and `seeded`, both keys in the
#: file the gate polices, and `history_gaps` derives its evidence from those
#: same two keys: delete BOTH and the contradiction it looks for is gone.
#: Measured against a scratch baseline: seed `consider-using-with`, fix the
#: finding, regenerate downward, `--seed consider-using-with` -> rc 1;
#: then drop the two keys and the same command -> "seeding 1 new
#: fingerprint(s)", rc 0, `evaluate()["ok"] is True` — character for
#: character the round-6 measurement, one family later.
#:
#: So the permission does not live in the baseline at all. `--seed` is
#: refused unless the family is granted HERE, which makes every upward write
#: a code change a reviewer sees — and a grant EXPIRES: once the baseline's
#: own records show it was spent, the gate fails until the line is deleted,
#: so a grant left behind is loud instead of being a standing off switch.
#: Deleting baseline keys now buys nothing: it cannot grant a seed, and the
#: grant it would need is not in the file it can edit.
SEED_GRANTS: dict = {}

#: Error symbols that are at ZERO in this tree and must stay there. A
#: finding under one of these fails the gate even if a baseline entry
#: exists — these are the checks that caught real defects in §4GJ.
#:
#: ⚠ MEMBERSHIP IS EARNED, NOT ASPIRED TO. A symbol belongs here only when
#: its current count is zero; listing one that still fires would leave the
#: gate red and teach the next person to skip it. While a symbol still has
#: live findings it stays OUT of this set and the fingerprint ratchet holds
#: it at exactly its known instances, so a DIFFERENT undefined name cannot
#: slip in under the same count — then promote it the moment the last one
#: is fixed. (This paragraph used to name `undefined-variable` and
#: `bad-str-strip-call` as "the two that BELONG here and are not"; both were
#: promoted twenty lines below in the same change, so the warning had been
#: contradicting its own list since §4GJ. Say the RULE, not the roster: a
#: roster in prose goes stale the first time someone acts on it.)
ZERO_TOLERANCE = frozenset({
    "used-before-assignment",          # `_wrote`: the except handler raised NameError
    "invalid-character-zero-width-space",
    "invalid-character-backspace",
    "invalid-character-carriage-return",
    "invalid-character-esc",
    "invalid-character-nul",
    "invalid-character-sub",
    "unexpected-keyword-arg",          # dspy MIPROv2 `num_trials`
    "no-value-for-parameter",
    "too-many-function-args",
    "function-redefined",
    "no-name-in-module",
    "import-error",
    "unpacking-non-sequence",
    "return-in-init",
    "nonexistent-operator",
    "await-outside-async",
    "not-an-iterable",
    "not-context-manager",
    "raising-non-exception",
    "catching-non-exception",
    "bad-except-order",
    "duplicate-argument-name",
    "invalid-unary-operand-type",
    "unhashable-member",
    "dict-iter-missing-items",
    # Promoted 2026-09-13 (§4GJ) once their last live finding was fixed:
    "undefined-variable",              # memory/temporal.py: `Tuple` never imported —
                                       # `get_type_hints` raised NameError (latent, real)
    "bad-str-strip-call",              # tools/memory.py: `"/" + os.sep` is "//" on POSIX,
                                       # a duplicate char in a CHARACTER-SET argument
    "possibly-used-before-assignment", # the two docker.py reads are flag-guarded and
                                       # carry a narrow disable with the reason
})


def _pylint_path() -> str | None:
    """The venv's pylint, or None. The caller must SKIP loudly on None —
    an instrument that cannot run must never read as green (§R R6)."""
    candidates = [
        Path(sys.executable).parent / "pylint",
        REPO.parent / ".agent.venv" / "bin" / "pylint",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def lint_targets(target: Path) -> list[str]:
    """The files to hand pylint for `target`.

    ⚠ NEVER the bare directory (§4GJ round 4). Handed a directory, pylint
    walks PACKAGES, so a source subdirectory without an `__init__.py` is
    silently not linted: measured on a copy of the tree, an undefined name
    in `src/ghost_agent/nopkg/c.py` gave 0 findings and rc=0 while the
    identical defect inside a package reddened. `--recursive=y` does NOT fix
    it when the target is itself a package — also measured: same 0 findings,
    because pylint takes the package as the unit and only descends into
    subpackages. Enumerating the files is the only form that cannot skip
    one, and it is exactly equivalent on today's tree (183 findings, same
    fingerprints, all three ways — verified before the swap).

    Latent today, since every directory under `src` carries an
    `__init__.py`. Latent is how a gate is found to have been off.
    """
    if target.is_file():
        return [str(target)]
    files = [str(p) for p in sorted(target.rglob("*.py"))
             if "__pycache__" not in p.parts]
    if not files:
        raise RuntimeError(f"no Python files under {target} — nothing linted")
    return files


def run_pylint(target: Path | None = None, jobs: int = 8) -> list[dict]:
    """Error-class findings over `target` (default: the package), as a list
    of raw pylint message dicts. Raises RuntimeError if pylint is absent."""
    exe = _pylint_path()
    if exe is None:
        raise RuntimeError("pylint is not installed in this interpreter's venv")
    proc = subprocess.run(
        # The error class is the HARD gate. `unused-import`/`unused-variable`
        # ride along so the ratchet can bound them too (§4GJ follow-up): they
        # are not errors, they are the drift that hides one — 138 of them at
        # the baseline, and a real undefined name is easy to miss in that
        # noise. They can only shrink.
        [exe, "-j", str(jobs), "--disable=all", "--enable=E,W0611,W0612",
         "--output-format=json2", "--score=n",
         *lint_targets(target or SRC)],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}"},
        timeout=600,
    )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"pylint produced no output (rc={proc.returncode}): "
                           f"{proc.stderr[-400:]}")
    data = json.loads(out)
    return data["messages"] if isinstance(data, dict) else data


def fingerprint(msg: dict) -> str:
    """(symbol, path, normalised message) — NO line number.

    Line numbers move on every unrelated edit; a baseline keyed on them
    would be regenerated blind, which is how a ratchet stops ratcheting.
    The message text is kept because it names the offending symbol
    (`Undefined variable 'Tuple'`), which is what makes two findings in one
    file distinguishable."""
    path = str(Path(msg["path"]).as_posix())
    return f"{msg['symbol']}|{path}|{msg['message'].strip()}"


def current_counts(messages: list[dict]) -> dict[str, int]:
    return dict(Counter(fingerprint(m) for m in messages))


def load_baseline() -> dict:
    """The baseline, or an empty one when the file does not exist yet.

    A CORRUPT file raises RuntimeError rather than letting `JSONDecodeError`
    out (§4GJ round 4): the caller turns that into exit 2, "the gate could
    not run", which `ci.sh` fails the build on. Before this, a truncated
    baseline gave a raw traceback and exit 1 — indistinguishable from "new
    findings", so the one arm written for an unusable instrument never saw
    it. An instrument that cannot run must SAY it cannot run (§R R6)."""
    if not BASELINE.exists():
        return {"entries": {}, "notes": {}}
    try:
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"the lint baseline {BASELINE} is CORRUPT ({exc}). The ratchet "
            "has no memory without it — restore it from git or regenerate "
            "deliberately with --write-baseline.") from exc
    # ⚠ THE BYTES PARSED IS NOT THE SHAPE LOADED (§4GK round 6). Round 4
    # guarded the decode and round 5 guarded the `allowed_slack` KEY — and
    # left the CONTAINER unguarded: a baseline that is valid JSON but not an
    # object (`[]`, a bare string, `null`) reached `evaluate` and died there
    # with `AttributeError: 'list' object has no attribute 'get'` — a
    # traceback reading as rc=1, which `evaluate`'s own comment calls "the
    # one verdict it is not". Empty and truncated files already answered
    # `LINT UNAVAILABLE`/rc=2 correctly; this is the same fact arriving in a
    # different shape, and it must reach the same arm.
    if not isinstance(data, dict):
        raise RuntimeError(
            f"the lint baseline {BASELINE} is not a JSON OBJECT (got "
            f"{type(data).__name__}). The ratchet cannot read its memory out "
            "of it — restore it, or regenerate with --write-baseline.")
    if not isinstance(data.get("entries", {}), dict):
        raise RuntimeError(
            f"the lint baseline {BASELINE} has a non-object \"entries\" (got "
            f"{type(data['entries']).__name__}). Every fingerprint count "
            "lives there — restore it, or regenerate with --write-baseline.")
    # ⚠ AND THE CHECK STOPPED AT THE CONTAINER (§4GK round 7). Round 6
    # rejected a non-object baseline and a non-object `entries` — and every
    # value INSIDE them still reached the arithmetic raw. Measured:
    # `{"entries": {"sym|p|m": "3"}}` -> `TypeError: '>' not supported
    # between instances of 'str' and 'int'` and **exit 1**, which
    # `scripts/ci.sh:102` reports as "error-class lint findings": a dirty
    # tree, when the truth is that the gate DID NOT RUN. `null` and `[1]`
    # values do the same, and `{"tracked_symbols": 5}` dies in `set(5)`
    # inside `tracked_symbols`. Same fact as a truncated file, same arm:
    # rc=2, LINT UNAVAILABLE. An instrument that cannot run must SAY it
    # cannot run (§R R6) — and the one verdict it must never produce is the
    # one that reads as a finding.
    for fp, count in (data.get("entries") or {}).items():
        if not isinstance(fp, str) or isinstance(count, bool) \
                or not isinstance(count, int) or count < 0:
            raise RuntimeError(
                f"the lint baseline {BASELINE} has a CORRUPT entry "
                f"{fp!r}: {count!r} — every fingerprint maps to a "
                "non-negative integer count. The ratchet compares against "
                "these; it cannot compare against that. Restore the file, or "
                "regenerate with --write-baseline.")
    for key in ("tracked_symbols", "seeded"):
        value = data.get(key)
        if value is not None and not isinstance(value, list):
            raise RuntimeError(
                f"the lint baseline {BASELINE} has a non-list \"{key}\" (got "
                f"{type(value).__name__}). That is the seed guard's memory — "
                "restore it, or delete the key, which falls back to the "
                "floor in scripts/lint.py.")
    if any(not isinstance(s, str) for s in data.get("tracked_symbols") or ()):
        raise RuntimeError(
            f"the lint baseline {BASELINE} has a non-string in "
            "\"tracked_symbols\". Symbol families are pylint symbol names.")
    for rec in data.get("seeded") or ():
        if not isinstance(rec, dict) or not isinstance(
                rec.get("symbols") or [], list):
            raise RuntimeError(
                f"the lint baseline {BASELINE} has a CORRUPT \"seeded\" "
                f"record {rec!r} — each is an object with a list of "
                "symbols. It is the only audit trail an upward write leaves.")
    if not isinstance(data.get("notes", {}), dict):
        raise RuntimeError(
            f"the lint baseline {BASELINE} has a non-object \"notes\" (got "
            f"{type(data['notes']).__name__}). Notes are keyed by "
            "fingerprint — restore it, or regenerate with --write-baseline.")
    return data


def seeded_symbols(baseline: dict) -> set:
    """Every family a `--seed` write has ever widened coverage for.

    Seeding is the one write that lets NEW fingerprints into the baseline, so
    it stamps a record — the same discipline the pin-quality ratchet's
    `--migrate "reason"` has had since §4GJ round 5, and the thing `--seed`
    was doing without (round 6: "two brand-new findings absorbed, leaving NO
    record in the file")."""
    out = set()
    for rec in baseline.get("seeded") or ():
        if isinstance(rec, dict):
            out |= {str(s) for s in rec.get("symbols") or ()}
    return out


def tracked_symbols(baseline: dict) -> set:
    """Every symbol family this tree has EVER tracked.

    ⚠ NOT the same question as "has an entry right now" (§4GJ round 4). The
    first version of the `--seed` guard asked the snapshot: drive a family to
    zero, regenerate downward (its entries vanish), and `--seed <that
    symbol>` became legal again — absorbing arbitrary NEW findings in a
    family that had already been ratcheted to nothing. `--seed` is a
    one-time COVERAGE widening for a family nobody has measured yet, so the
    question it must ask is about history.

    ⚠ AND HISTORY THAT LIVES ONLY IN THE POLICED FILE IS AN OFF SWITCH
    (§4GK round 6, see `TRACKED_FLOOR`). `tracked_symbols` in the baseline
    only ever grows THROUGH THE WRITER; a text editor is not the writer. So
    the code-side floor is unioned in first, and the baseline's list, the
    live entries and the stamped seed records can only ADD to it."""
    return (set(TRACKED_FLOOR) | set(ZERO_TOLERANCE)
            | {fp.split("|", 1)[0] for fp in baseline.get("entries", {})}
            | set(baseline.get("tracked_symbols") or ())
            | seeded_symbols(baseline))


def history_gaps(baseline: dict) -> set:
    """Families the baseline's own records prove, but its `tracked_symbols`
    list omits — i.e. the list has been edited DOWN.

    A baseline with no `tracked_symbols` key at all is not a gap: the
    committed one predates the key (round 5 added it to the writer and the
    file has not been regenerated since), and `tracked_symbols` above falls
    back to the floor for it. A key that IS present is a claim about history,
    and a claim that contradicts the entries and seed records under it is the
    shape the round-6 measurement exploited.

    ⚠ AND IT CANNOT SEE THE EDIT THAT DELETES BOTH SIDES (§4GK round 7).
    The evidence comes from `entries` and `seeded` — two keys in the same
    file as the list it checks them against — so removing the key AND the
    seed records leaves nothing to contradict and this returns the empty
    set. That hole is closed in `SEED_GRANTS`, not here: an arm that derives
    its evidence from the file it polices can always be emptied, so the
    PERMISSION to seed had to leave the file instead."""
    declared = baseline.get("tracked_symbols")
    if declared is None:
        return set()
    proven = ({fp.split("|", 1)[0] for fp in baseline.get("entries", {})}
              | seeded_symbols(baseline))
    return proven - set(declared)


def spent_seed_grants(baseline: dict) -> set:
    """Grants in `SEED_GRANTS` the baseline's own records show already SPENT.

    A grant is a one-time permission to admit NEW fingerprints in a family
    nothing has measured. Once the write has landed, the family is tracked
    and the grant is a standing off switch for it — exactly what `--seed`'s
    refusal exists to prevent — so the gate FAILS until the line is deleted
    from this file. The expiry is the other half of "the permission lives in
    code": a code change to open it, a code change to leave it open."""
    proven = ({fp.split("|", 1)[0] for fp in baseline.get("entries", {})}
              | seeded_symbols(baseline)
              | set(baseline.get("tracked_symbols") or ()))
    return proven & set(SEED_GRANTS)


def evaluate(messages: list[dict], baseline: dict) -> dict:
    """The verdict: hard failures, ratchet failures, and expired slack."""
    entries = baseline.get("entries", {})
    counts = current_counts(messages)
    # The baseline does not get a vote on its own ceiling. A key is still
    # READ so it can be REPORTED — a silently ignored off switch teaches the
    # next reader that it works. ANY declaration counts, however it is
    # spelled: `int(x or 0)` would have let a string through as a traceback
    # (rc=1, "new findings" — the one verdict it is not), which is the same
    # shape as the corrupt baseline round 4 had to catch.
    declared = baseline.get("allowed_slack")
    self_granted = declared is not None and declared != 0

    zero_tol = [m for m in messages if m["symbol"] in ZERO_TOLERANCE]
    new_fps = {fp: n for fp, n in counts.items() if fp not in entries}
    grown = {fp: (n, entries[fp]) for fp, n in counts.items()
             if fp in entries and n > entries[fp]}
    stale = {fp: n for fp, n in entries.items() if fp not in counts}
    # SLACK = tolerance the tree no longer needs: an entry that no longer
    # fires at all, or one whose count has dropped. Either way it is a
    # standing permit to re-introduce the exact finding that was fixed, and
    # it is the reason the "ratchet" did not ratchet before round 4.
    slack = {fp: entries[fp] - counts.get(fp, 0) for fp in entries
             if entries[fp] > counts.get(fp, 0)}
    slack_total = sum(slack.values())

    return {
        "total": len(messages),
        "zero_tolerance": [
            {"symbol": m["symbol"], "path": m["path"], "line": m["line"],
             "message": m["message"]} for m in zero_tol],
        "new_fingerprints": new_fps,
        "grown_fingerprints": grown,
        "stale_baseline_entries": stale,
        "slack": slack,
        "slack_total": slack_total,
        "allowed_slack": ALLOWED_SLACK,
        "baseline_declared_slack": declared,
        "baseline_grants_slack": self_granted,
        # The seed guard's memory, checked for shrinkage the way `entries`
        # is (§4GK round 6). A `tracked_symbols` list that contradicts the
        # entries and seed records beneath it was edited by hand, and the
        # only thing that edit buys is the right to seed a family that was
        # already ratcheted to zero.
        "tracked_symbols_gap": sorted(history_gaps(baseline)),
        # A seed grant that has been SPENT (§4GK round 7). The arm above can
        # be emptied by deleting the two keys it reads; this one cannot,
        # because what it reads lives in this file. A grant left behind after
        # its write is a standing permission to seed that family again.
        "spent_seed_grants": sorted(spent_seed_grants(baseline)),
        "ok": (not zero_tol and not new_fps and not grown
               and slack_total <= ALLOWED_SLACK
               and not self_granted
               and not history_gaps(baseline)
               and not spent_seed_grants(baseline)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--seed", default="",
                    help="comma-separated symbols to seed into the baseline on "
                         "this write. ONLY legal for a symbol the baseline has "
                         "NEVER tracked (see `tracked_symbols`) — widening "
                         "COVERAGE is not the same as tolerating growth, and "
                         "everything ever tracked still ratchets downward. "
                         "Requires --seed-reason.")
    ap.add_argument("--seed-reason", default="",
                    help="why this family is being seeded. STAMPED into the "
                         "baseline, like the pin ratchet's --migrate: seeding "
                         "is the one write that admits new fingerprints, and "
                         "it used to leave no record at all.")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    try:
        messages = run_pylint(jobs=args.jobs)
        # Inside the same arm as a missing pylint: a baseline that cannot be
        # read is a gate that cannot run, not a tree that is dirty.
        baseline = load_baseline()
    except RuntimeError as exc:
        print(f"LINT UNAVAILABLE: {exc}", file=sys.stderr)
        return 2

    if args.write_baseline:
        bootstrap = not BASELINE.exists()
        old = baseline.get("entries", {})
        new = current_counts(messages)
        grew = {fp: (n, old[fp]) for fp, n in new.items()
                if fp in old and n > old[fp]}
        added = [fp for fp in new if fp not in old]
        # The very first write has nothing to ratchet against; every later
        # one must only shrink. (The refusal fired on the bootstrap run
        # while this was unconditional — the instrument checking itself.)
        if bootstrap:
            grew, added = {}, []
        # One-time COVERAGE widening: a symbol family the baseline has never
        # tracked can be seeded, because there is nothing to ratchet against
        # yet — the same reasoning as the bootstrap write. A symbol that IS
        # tracked cannot be seeded, so this can never launder growth in an
        # existing family (§4GJ; pinned in tests/test_lint_gate.py).
        seed = {sym.strip() for sym in args.seed.split(",") if sym.strip()}
        already = tracked_symbols(baseline)
        refused = seed & already
        if refused:
            print(f"REFUSING to seed already-tracked symbol(s): "
                  f"{sorted(refused)} — those ratchet downward.", file=sys.stderr)
            return 1
        # A hand-edited history is refused at the WRITE path too, or the
        # laundering round 6 measured would simply be re-blessed by the next
        # regenerate: the writer unions the list back together, so the edit
        # would leave no trace anywhere (§4GK round 6).
        gaps = history_gaps(baseline)
        if gaps:
            print(f"REFUSING to write against a baseline whose "
                  f"\"tracked_symbols\" was edited DOWN: {sorted(gaps)} still "
                  f"appear in its entries or seed records. Put them back (or "
                  f"delete the key, which falls back to the floor in "
                  f"scripts/lint.py) before regenerating.", file=sys.stderr)
            return 1
        seed_reason = args.seed_reason.strip()
        if seed and not seed_reason:
            # The asymmetry round 6 named: the pin ratchet forces
            # `--migrate "reason"` to be STAMPED for an upward write, and
            # `--seed` — the lint gate's only upward write — stamped nothing.
            print("--seed needs --seed-reason \"why this family has never "
                  "been measured\". It is stamped into the baseline, which is "
                  "the whole point of allowing new fingerprints in at all.",
                  file=sys.stderr)
            return 1
        # ⚠ AND "NEVER TRACKED" IS A CLAIM THE BASELINE MAKES ABOUT ITSELF
        # (§4GK round 7). Every memory the refusal above consults —
        # `entries`, `tracked_symbols`, `seeded` — is a key in the file this
        # write is about to overwrite, and the `history_gaps` cross-check
        # derives its evidence from two of them, so deleting BOTH leaves no
        # contradiction to find. Measured: seed a family, fix it, regenerate
        # downward, drop the two keys, seed it again -> rc 0, gate OK, no
        # record. The permission therefore lives in `SEED_GRANTS`, in THIS
        # file: a hand-edit of the baseline cannot write one.
        ungranted = seed - set(SEED_GRANTS)
        if ungranted:
            print(f"REFUSING to seed ungranted symbol(s): {sorted(ungranted)}. "
                  f"`--seed` is the only write that admits NEW fingerprints, "
                  f"so the permission is a CODE change: add the family to "
                  f"SEED_GRANTS in {Path(__file__).name} with the reason it "
                  f"has never been measured, seed it, then delete the grant "
                  f"(the gate fails while a spent grant is still there).",
                  file=sys.stderr)
            return 1
        seeded = []
        if seed:
            seeded = [fp for fp in added if fp.split("|", 1)[0] in seed]
            if seeded:
                print(f"seeding {len(seeded)} new fingerprint(s) for {sorted(seed)}")
            added = [fp for fp in added if fp not in set(seeded)]
        if grew or added:
            print("REFUSING to write a baseline that grew:", file=sys.stderr)
            for fp, (n, was) in grew.items():
                print(f"  {fp}  {was} -> {n}", file=sys.stderr)
            for fp in added:
                print(f"  NEW  {fp}", file=sys.stderr)
            print("\nThe baseline ratchets DOWNWARD only. Fix the finding.",
                  file=sys.stderr)
            return 1
        BASELINE.write_text(json.dumps(
            {"_comment": "Error-class findings tolerated at the time of writing. "
                         "Regenerate with scripts/lint.py --write-baseline, which "
                         "refuses to grow this file. Every entry is a pylint "
                         "inference limit or a known finding — see notes. The "
                         "slack ceiling is ALLOWED_SLACK in scripts/lint.py, "
                         "not a key here.",
             # NO `allowed_slack` (§4GJ round 5). The writer used to copy the
             # key forward, which made a hand-written off switch permanent;
             # regenerating now DROPS one instead of preserving it.
             "entries": new,
             # The seed guard's memory. Symbol families only ever ACCUMULATE
             # here, so driving one to zero does not make it seedable again —
             # and the FLOOR of this list is TRACKED_FLOOR in scripts/lint.py,
             # so editing this key down does not re-arm `--seed` either.
             "tracked_symbols": sorted(tracked_symbols(baseline)
                                       | {fp.split("|", 1)[0] for fp in new}),
             # Every coverage widening, on the record and accumulating. The
             # only write that admits NEW fingerprints has to say why it did
             # (§4GK round 6); `--seed` used to say nothing at all.
             "seeded": list(baseline.get("seeded") or ())
                       + ([{"symbols": sorted(seed), "reason": seed_reason,
                            "fingerprints": len(seeded)}] if seeded else []),
             "notes": baseline.get("notes", {})},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {len(new)} fingerprints, "
              f"{sum(new.values())} findings")
        return 0

    verdict = evaluate(messages, baseline)
    if args.json:
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1

    print(f"pylint error-class findings: {verdict['total']}")
    for m in verdict["zero_tolerance"]:
        print(f"  ZERO-TOLERANCE {m['symbol']} {m['path']}:{m['line']} "
              f"{m['message']}")
    for fp, n in verdict["new_fingerprints"].items():
        print(f"  NEW x{n}  {fp}")
    for fp, (n, was) in verdict["grown_fingerprints"].items():
        print(f"  GREW {was}->{n}  {fp}")
    live = current_counts(messages)
    for fp, n in verdict["slack"].items():
        print(f"  {'STALE' if fp not in live else 'SHRUNK'} (-{n}, tolerance "
              f"the tree no longer needs)  {fp}")
    if verdict["slack_total"] > verdict["allowed_slack"]:
        print(f"\n{verdict['slack_total']} findings of standing tolerance are "
              f"no longer used (allowed: {verdict['allowed_slack']}). Lock the "
              f"win in with `python scripts/lint.py --write-baseline`, or the "
              f"fixed findings can be re-introduced for free.")
    if verdict["tracked_symbols_gap"]:
        print(f"\n  HISTORY EDITED DOWN: {BASELINE} lists "
              f"\"tracked_symbols\" that omits "
              f"{verdict['tracked_symbols_gap']}, which its own entries or "
              f"seed records still name. The only thing that edit buys is "
              f"`--seed` for a family already ratcheted to zero. Put them "
              f"back, or delete the key.")
    if verdict["spent_seed_grants"]:
        print(f"\n  SPENT SEED GRANT: scripts/lint.py still grants "
              f"{verdict['spent_seed_grants']} to `--seed`, and {BASELINE} "
              f"already tracks {'them' if len(verdict['spent_seed_grants']) > 1 else 'it'}. "
              f"The grant is one write long — delete the line from "
              f"SEED_GRANTS, or it is a standing permission to absorb new "
              f"findings into a family that is already measured.")
    if verdict["baseline_grants_slack"]:
        print(f"\n  SELF-GRANTED SLACK: {BASELINE} declares "
              f"\"allowed_slack\": {verdict['baseline_declared_slack']}. The "
              f"ceiling is ALLOWED_SLACK in scripts/lint.py ({ALLOWED_SLACK}) "
              f"— the file under the gate does not set its own tolerance. "
              f"Delete the key (or regenerate, which drops it).")
    print("OK" if verdict["ok"] else "FAIL")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
