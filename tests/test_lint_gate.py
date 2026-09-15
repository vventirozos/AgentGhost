"""§4GJ — the lint gate, and the except-handler binding class.

WHY. The tree had no linter until §4GJ. A single pylint pass over `src`
found, with no reviewer involved: two undefined names inside "this report
must not die" exception handlers (§4GI), a name read in an `except` handler
that its `try` may never have bound (`_wrote`, tools/file_system.py), a
`dspy` fallback passing a keyword the installed version removed, and two
invisible zero-width characters in source. The gate below is what stops the
next one, and `scripts/lint.py` is its single implementation — `scripts/
ci.sh` calls the same runner, so the rule cannot drift between the two.

The two halves:

* `test_no_zero_tolerance_findings` — symbols at zero after §4GJ must stay
  at zero. The world where it fails: any commit reintroducing an undefined
  name, an unbound local read, or an invisible control character.
* `test_error_findings_do_not_grow` — every other error-class finding must
  be a KNOWN fingerprint (symbol, path, message; never a line number) with
  a count that may only shrink. The world where it fails: a new inference
  finding, a new file with the same defect, or one known defect swapped for
  another (a count-only baseline would absorb that swap silently).
* `test_the_baseline_does_not_outlive_the_findings` — and the tolerance
  must EXPIRE. Round 4 measured the gap: fix a baselined finding (rc=0,
  "stale, please regenerate downward") and re-introduce it — rc=0 again.
  A ratchet whose slack never expires is a list of permanent permissions.

`test_except_handler_reads_a_name_its_try_may_not_bind` is the R1
enumeration for the class the linter found: `_wrote` was one instance, and
the AST walk is what proves there is no second.
"""
from __future__ import annotations

import ast
import builtins
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "ghost_agent"
_BUILTIN_NAMES = frozenset(dir(builtins))

sys.path.insert(0, str(REPO / "scripts"))
import lint as lint_mod  # noqa: E402


# ── the gate ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def findings():
    """Error-class findings, once per module (the run costs ~50s).

    SKIPS LOUDLY when pylint is absent: an instrument that cannot run must
    never read as green (§R R6). A silent pass here would be worse than no
    gate, because the suite would claim the property holds.

    ⚠ THE ASYMMETRY WITH THE CANARY TESTS IS DELIBERATE (§4GJ round 4,
    reviewed and kept). A test that calls `run_pylint(target=...)` directly
    does NOT skip — it RAISES, and the file goes red. Making them consistent
    with this fixture was considered and REJECTED: with pylint uninstalled,
    every arm of this file would then report "skipped", and a suite full of
    skips is read as a suite that passed. (Round 4 wrote "the TWO tests"
    here; there were three — verified by driving the file with
    `_pylint_path` patched to None: 3 failed, 17 passed, 3 skipped. The
    decision was right and the roster was wrong the moment it was written,
    which is what `scripts/lint.py`'s own ZERO_TOLERANCE comment warns
    about: say the RULE, not the roster.) The three signals as they stand
    are each pointed at a different reader:

      * `ci.sh` — `scripts/lint.py` exits 2 and ci.sh FAILS THE BUILD on it,
        pinned by `test_the_ci_gate_treats_a_missing_linter_as_a_failure`.
        That is the authoritative answer, and it is a failure.
      * this fixture — skips, because a developer running the suite on a
        machine without pylint needs to be told which property is
        unverified, not handed a wall of red for an unrelated change.
      * the canaries — go red, so that "pylint vanished" cannot be mistaken
        for "the gate ran and found nothing" by anyone reading pytest alone.

    The one change that is NOT acceptable is making all three skippable.
    """
    if lint_mod._pylint_path() is None:
        pytest.skip("pylint is not installed in this venv — the lint gate "
                    "CANNOT RUN and is therefore NOT green; install pylint "
                    "or this property is unverified")
    try:
        return lint_mod.run_pylint()
    except RuntimeError as exc:            # pragma: no cover - environment
        pytest.skip(f"pylint could not run: {exc}")


def test_no_zero_tolerance_findings(findings):
    """Symbols that §4GJ drove to zero must stay at zero — baseline or not."""
    offenders = [
        f"{m['symbol']} {m['path']}:{m['line']} {m['message']}"
        for m in findings if m["symbol"] in lint_mod.ZERO_TOLERANCE
    ]
    assert offenders == [], (
        "error-class findings under a zero-tolerance symbol:\n  "
        + "\n  ".join(offenders)
        + "\n\nThese are not ratcheted: fix the code. If the finding is a "
          "pylint inference limit, silence it NARROWLY on the line with a "
          "reason, never by widening the config.")


def test_error_findings_do_not_grow(findings):
    """The fingerprint ratchet: no new finding, no grown count."""
    verdict = lint_mod.evaluate(findings, lint_mod.load_baseline())
    assert not verdict["new_fingerprints"], (
        "NEW error-class findings not in tests/lint_baseline.json:\n  "
        + "\n  ".join(f"x{n}  {fp}" for fp, n
                      in verdict["new_fingerprints"].items())
        + "\n\nFix the finding. The baseline ratchets DOWNWARD only — "
          "`python scripts/lint.py --write-baseline` refuses to grow it.")
    assert not verdict["grown_fingerprints"], (
        "error-class findings INCREASED for a known fingerprint:\n  "
        + "\n  ".join(f"{was} -> {n}  {fp}" for fp, (n, was)
                      in verdict["grown_fingerprints"].items()))


def test_the_baseline_does_not_outlive_the_findings(findings):
    """The third half (§4GJ round 4): tolerance the tree no longer needs
    must be handed back, or every fixed finding can be re-introduced for
    free. Pinned HERE as well as in `scripts/lint.py` because a gate that
    fails in ci.sh and passes in pytest teaches people to trust the wrong
    one (R5: the runner and its consumer must agree).

    ⚠ THE BOUND IS `lint_mod.ALLOWED_SLACK`, NOT `verdict["allowed_slack"]`
    (§4GJ round 5). Round 4 wrote the latter, which came from the baseline —
    so this assertion compared the file against a number the same file
    supplied, and `"allowed_slack": 9999` made BOTH halves of the gate
    unfailable at once. A test bounded by its subject's own claim is the
    `declared-vs-derived` shape: the ceiling has to come from somewhere the
    tree under test cannot write."""
    verdict = lint_mod.evaluate(findings, lint_mod.load_baseline())
    assert verdict["slack_total"] <= lint_mod.ALLOWED_SLACK, (
        f"tests/lint_baseline.json tolerates {verdict['slack_total']} "
        "findings the tree no longer has:\n  "
        + "\n  ".join(f"-{n}  {fp}" for fp, n in verdict["slack"].items())
        + "\n\nLock the win in: `python scripts/lint.py --write-baseline`.")
    assert not verdict["baseline_grants_slack"], (
        "tests/lint_baseline.json declares its own slack ceiling "
        f"({verdict['baseline_declared_slack']}). The ceiling is "
        "ALLOWED_SLACK in scripts/lint.py; a baseline that sets it is "
        "turning the arm above off.")


def test_the_baseline_is_annotated_and_every_note_still_has_a_finding():
    """A baseline of bare fingerprints rots into a list nobody reads, so
    every note must be classifiable from the file itself — and every note
    must still DESCRIBE something.

    ⚠ THE PREVIOUS VERSION OF THIS PIN KEPT DEAD DOCUMENTATION ALIVE. It
    demanded a note for `undefined-variable` and `bad-str-strip-call`
    specifically; both were fixed and PROMOTED to ZERO_TOLERANCE in §4GJ, so
    their entries left the baseline while their notes ("REAL, LATENT — NOT
    MINE TO FIX") stayed, and removing the stale notes — the correct
    housekeeping — would have reddened this file. A pin that punishes the
    cleanup is a pin that guarantees the rot.

    The rule is the general one and points the other way: a note names a
    tolerance, so a note whose fingerprint no longer fires is a claim about
    a finding that no longer exists. Fails in: today's tree before the three
    promoted-symbol notes were removed (measured: 3 stale of 11)."""
    baseline = lint_mod.load_baseline()
    assert baseline["entries"], "baseline is empty — was it ever generated?"
    notes = baseline.get("notes", {})
    assert notes, "baseline carries no notes: a reader cannot tell an " \
                  "inference limit from a real defect"
    orphans = []
    for key in notes:
        symbol, _, rest = key.partition("|")
        if not any(fp.startswith(symbol + "|") and rest in fp
                   for fp in baseline["entries"]):
            orphans.append(key)
    assert orphans == [], (
        "baseline notes describing findings that are no longer in the "
        f"baseline: {orphans}. Either the finding was fixed (delete the "
        "note — the promotion comment in scripts/lint.py is the record) or "
        "the note's key stopped matching its fingerprints.")


#: A stub lint runner: prints, drops a marker so the caller can prove it
#: RAN, and exits with the code the test is driving.
_STUB_LINT = """\
import pathlib, sys
pathlib.Path(__file__).parent.parent.joinpath("lint-ran").write_text("yes")
print("stub lint runner")
sys.exit({rc})
"""


def _ci_sandbox(tmp_path, rc: int) -> Path:
    """A scratch repo root holding the REAL `scripts/ci.sh` and a stub lint
    runner that exits `rc`.

    ci.sh resolves its own root from `$BASH_SOURCE`, so a copy under
    `tmp_path` executes every branch of the real script against a runner we
    control — no text assertion, and no full pytest suite (`--no-tests`).
    black still runs over the absent `src`/`interface`/`tests`; it is
    advisory in ci.sh and cannot set the status, so the lint verdict is the
    only thing that can.
    """
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "scripts" / "ci.sh", tmp_path / "scripts" / "ci.sh")
    (tmp_path / "scripts" / "lint.py").write_text(
        _STUB_LINT.format(rc=rc), encoding="utf-8")
    return tmp_path / "scripts" / "ci.sh"


def _run_ci(ci: Path):
    return subprocess.run(["bash", str(ci), "--no-tests"],
                          capture_output=True, text=True, timeout=180)


def test_the_gate_and_ci_use_the_same_runner(tmp_path):
    """One implementation. `ci.sh` calling pylint with its own flags is how
    the gate and CI come to disagree about what 'clean' means.

    ⚠ THIS USED TO GREP ci.sh (§4GJ round 5). It asserted that the string
    `scripts/lint.py` appeared and that no line looked like a direct pylint
    invocation — an R4 source-text assertion, which is satisfied by the
    string sitting in a comment, and which the pin-quality ratchet could not
    even see (ci.sh is not Python, so RECOGNITION skipped it). It now RUNS
    the script: the stub runner writes a marker when it is executed, so the
    claim "ci.sh's lint verdict comes from scripts/lint.py" is answered by
    the script's own behaviour.

    Fails in: a ci.sh that lints some other way (the marker never appears)
    or does not lint at all.
    """
    ci = _ci_sandbox(tmp_path, rc=0)

    out = _run_ci(ci)

    assert (tmp_path / "lint-ran").exists(), (
        "ci.sh did not execute scripts/lint.py — its lint verdict comes "
        f"from somewhere else.\n{out.stdout[-600:]}\n{out.stderr[-600:]}")
    assert out.returncode == 0, (out.stdout[-600:], out.stderr[-600:])
    assert "CI PASSED" in out.stdout


# ── the gate's four arms, on SYNTHETIC findings ─────────────────────────────
#
# ⚠ THE TESTS ABOVE CANNOT TELL A WORKING GATE FROM A DISABLED ONE. They run
# the real linter over the real `src`, and `src` is CLEAN: every arm reports
# "nothing wrong" whether it is implemented or stubbed to a constant. All
# four of `evaluate`'s arms SURVIVED mutation on that evidence —
# zero-tolerance emptied, new-fingerprint detection emptied, growth
# detection emptied, and the missing-linter path returning 0 — each with
# this file green. That is R4's "fixtures where the fixed and broken worlds
# agree", committed inside the gate built to enforce R4. The scratch-copy
# demonstration that shipped with the first version was a manual
# observation, and a manual observation is not a pin.
#
# These drive the pure functions with fabricated findings, so every arm has
# a world in which it FAILS. Each test carries its own inverse, so the
# passing and failing cases differ ONLY in the arm under test — a test that
# fails for two reasons at once cannot tell you which one is live. No pylint
# run: they are milliseconds.

_EXAMPLE = "src/ghost_agent/core/example.py"


def _msg(symbol, *, path=_EXAMPLE, message="Example finding", line=42):
    """One pylint message in the shape `evaluate` consumes."""
    return {"symbol": symbol, "path": path, "message": message, "line": line}


def _baseline(*pairs):
    return {"entries": dict(pairs), "notes": {}}


def test_zero_tolerance_arm_fails_even_when_the_baseline_tolerates_it():
    """Arm 1: a zero-tolerance symbol is NOT ratcheted. Present in the
    baseline at its exact count — the state that silences every other arm —
    it must still fail.

    World where it fails: `zero_tol = []`.
    """
    assert "used-before-assignment" in lint_mod.ZERO_TOLERANCE
    hit = _msg("used-before-assignment",
               message="Using variable '_wrote' before assignment")
    verdict = lint_mod.evaluate([hit], _baseline((lint_mod.fingerprint(hit), 1)))

    assert verdict["ok"] is False
    assert [m["symbol"] for m in verdict["zero_tolerance"]] == \
        ["used-before-assignment"]
    # The ratchet arms are deliberately silent here, so the failure can only
    # have come from the arm under test.
    assert not verdict["new_fingerprints"]
    assert not verdict["grown_fingerprints"]

    # INVERSE: identical shape, a symbol that IS ratcheted -> passes.
    assert "no-member" not in lint_mod.ZERO_TOLERANCE
    tolerated = _msg("no-member", message="Instance of 'X' has no 'y' member")
    ok = lint_mod.evaluate([tolerated],
                           _baseline((lint_mod.fingerprint(tolerated), 1)))
    assert ok["ok"] is True
    assert ok["zero_tolerance"] == []


def test_new_fingerprint_arm_fails_on_a_finding_the_baseline_never_saw():
    """Arm 2: an unknown fingerprint fails, a known one at the same count
    passes.

    World where it fails: `new_fps = {}`.
    """
    hit = _msg("no-member", message="Instance of 'Widget' has no 'colour' member")
    fp = lint_mod.fingerprint(hit)

    verdict = lint_mod.evaluate([hit], _baseline())
    assert verdict["ok"] is False
    assert verdict["new_fingerprints"] == {fp: 1}
    assert not verdict["grown_fingerprints"]
    assert verdict["zero_tolerance"] == []

    # INVERSE: the very same finding, already known -> passes.
    known = lint_mod.evaluate([hit], _baseline((fp, 1)))
    assert known["ok"] is True
    assert known["new_fingerprints"] == {}


def test_growth_arm_fails_when_a_known_fingerprint_multiplies():
    """Arm 3: the count may only shrink. Baseline+1 fails; baseline and
    baseline-1 pass.

    World where it fails: `grown = {}`.
    """
    first = _msg("not-callable", message="thing is not callable", line=10)
    second = _msg("not-callable", message="thing is not callable", line=99)
    fp = lint_mod.fingerprint(first)
    assert lint_mod.fingerprint(second) == fp, \
        "the fingerprint must ignore line numbers, or the ratchet churns"

    grew = lint_mod.evaluate([first, second], _baseline((fp, 1)))
    assert grew["ok"] is False
    assert grew["grown_fingerprints"] == {fp: (2, 1)}
    assert not grew["new_fingerprints"]
    assert grew["zero_tolerance"] == []

    # INVERSE 1: exactly at the baseline -> passes.
    level = lint_mod.evaluate([first, second], _baseline((fp, 2)))
    assert level["ok"] is True
    assert level["grown_fingerprints"] == {}

    # INVERSE 2: below it -> the GROWTH arm is silent, and the entry is not
    # stale (it still fires, just less). Whether an under-used tolerance is
    # a FAILURE is the slack arm's question, isolated in the test below —
    # this one must not answer it, or a red here would not say which arm
    # fired.
    shrunk = lint_mod.evaluate([first], _baseline((fp, 2)))
    assert shrunk["grown_fingerprints"] == {}
    assert shrunk["stale_baseline_entries"] == {}

    # Gone entirely -> reported as stale, and again not growth.
    gone = lint_mod.evaluate([], _baseline((fp, 2)))
    assert gone["grown_fingerprints"] == {}
    assert gone["stale_baseline_entries"] == {fp: 2}


def test_slack_arm_fails_when_the_baseline_outlives_its_findings():
    """Arm 5 (§4GJ round 4): tolerance that the tree no longer needs must
    EXPIRE, or the ratchet does not ratchet.

    The reviewer's measurement, in a scratch repo: baseline a `no-member`,
    fix it — rc=0, printed "(stale, please regenerate downward)" — then
    re-introduce the SAME defect: rc=0 again. The fixed finding came back
    for free, and the committed baseline carried 167 fingerprints of
    standing tolerance that could never expire. (The ceiling on that slack
    is `lint_mod.ALLOWED_SLACK`, a constant in the gate — round 4 read it
    out of the baseline instead, which is the arm below.)

    World where it fails: `ok` computed without `slack_total` — i.e. the
    pre-round-4 line, where `stale_baseline_entries` was printed and then
    dropped on the floor.
    """
    hit = _msg("no-member", message="Instance of 'Widget' has no 'colour' member")
    fp = lint_mod.fingerprint(hit)

    gone = lint_mod.evaluate([], _baseline((fp, 1)))
    assert gone["ok"] is False
    assert gone["slack"] == {fp: 1} and gone["slack_total"] == 1
    assert gone["stale_baseline_entries"] == {fp: 1}
    # the other three arms are silent, so the red can only be this one
    assert not gone["new_fingerprints"] and not gone["grown_fingerprints"]
    assert gone["zero_tolerance"] == []

    # A SHRUNK count is slack too: two tolerated, one left, one free slot to
    # re-introduce the identical finding into the identical file.
    shrunk = lint_mod.evaluate([hit], _baseline((fp, 2)))
    assert shrunk["ok"] is False
    assert shrunk["slack"] == {fp: 1}
    assert shrunk["stale_baseline_entries"] == {}      # it still fires

    # INVERSE: exactly at the baseline -> no slack, green.
    level = lint_mod.evaluate([hit], _baseline((fp, 1)))
    assert level["ok"] is True and level["slack_total"] == 0


def test_a_baseline_cannot_grant_itself_slack(tmp_path, monkeypatch, capsys):
    """Arm 6 (§4GJ round 5): the arm above read its ceiling out of the file
    it polices, and `--write-baseline` copied the number forward.

    Reproduced by the reviewer: baseline a finding, fix it, write
    `"allowed_slack": 9999` into the baseline, re-introduce the identical
    finding -> `OK`, rc=0 — with the pytest half bounded by the same
    self-declared number, so neither consumer could fail. Round 4 replaced
    "tolerance that never expires" with "tolerance that expires unless the
    baseline says otherwise".

    World where it fails: `allowed_slack = int(baseline.get("allowed_slack",
    0) or 0)` — under which every assertion below flips.
    """
    hit = _msg("no-member", message="Instance of 'Widget' has no 'colour' member")
    fp = lint_mod.fingerprint(hit)

    # the exact laundering shape: one entry of standing tolerance the tree no
    # longer needs, plus a key that says "that's allowed".
    bought = {"entries": {fp: 1}, "allowed_slack": 9999}
    verdict = lint_mod.evaluate([], bought)
    assert verdict["ok"] is False, \
        "the baseline bought its own slack and the gate agreed"
    assert verdict["allowed_slack"] == lint_mod.ALLOWED_SLACK == 0
    assert verdict["baseline_declared_slack"] == 9999
    assert verdict["baseline_grants_slack"] is True
    # however it is spelled — a string ceiling used to escape as a
    # ValueError traceback, i.e. rc=1, "new findings", the one verdict it is
    # not (the corrupt-baseline shape from round 4).
    assert lint_mod.evaluate([], {"entries": {}, "allowed_slack": "9999"})[
        "ok"] is False
    # the other arms stay silent, so the red can only be this one
    assert not verdict["new_fingerprints"] and not verdict["grown_fingerprints"]
    assert verdict["zero_tolerance"] == []

    # ...and the key is fatal even with NO slack to excuse: the only reason
    # to write it is to disarm the arm, so it is reported, never ignored.
    clean = {"entries": {fp: 1}, "allowed_slack": 9999}
    assert lint_mod.evaluate([hit], clean)["ok"] is False
    assert lint_mod.evaluate([hit], {"entries": {fp: 1}})["ok"] is True

    # R5: end to end through `main`, including the sticky half — the writer
    # must not carry a hand-written ceiling into the file it regenerates.
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"entries": {fp: 1}, "allowed_slack": 9999,
                                "notes": {}}), encoding="utf-8")
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [])
    monkeypatch.setattr(sys, "argv", ["lint.py"])
    assert lint_mod.main() == 1
    assert "SELF-GRANTED SLACK" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["lint.py", "--write-baseline"])
    assert lint_mod.main() == 0
    assert "allowed_slack" not in json.loads(base.read_text()), \
        "regenerating preserved the off switch"


def test_a_fixed_finding_cannot_be_re_introduced_for_free(tmp_path, monkeypatch,
                                                          capsys):
    """R5, the consumer half of the slack arm: the reviewer's whole scratch
    -repo sequence, replayed through `main()` with a fabricated pylint run so
    it costs milliseconds instead of 50s.

    Deliberately NOT single-armed — it is the story, not an arm: green ->
    fix -> red (slack) -> regenerate -> re-introduce -> red (new). Step 4 is
    the one that used to be rc=0.
    """
    hit = _msg("no-member", message="Instance of 'Widget' has no 'colour' member")
    fp = lint_mod.fingerprint(hit)
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"entries": {fp: 1}, "notes": {}}),
                    encoding="utf-8")
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    monkeypatch.setattr(sys, "argv", ["lint.py"])

    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [hit])
    assert lint_mod.main() == 0, "the tolerated finding must be green"

    # the finding is FIXED, and nobody regenerated: the gate says so now
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [])
    assert lint_mod.main() == 1
    assert "no longer used" in capsys.readouterr().out

    # regenerate DOWNWARD, then re-introduce the identical defect
    monkeypatch.setattr(sys, "argv", ["lint.py", "--write-baseline"])
    assert lint_mod.main() == 0
    assert json.loads(base.read_text())["entries"] == {}
    monkeypatch.setattr(sys, "argv", ["lint.py"])
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [hit])
    assert lint_mod.main() == 1, \
        "a fixed finding came back and the gate stayed green"


def test_a_corrupt_baseline_says_the_gate_could_not_run(tmp_path, monkeypatch,
                                                        capsys):
    """§4GJ round 4: a truncated baseline used to escape as a raw
    `JSONDecodeError` traceback and exit 1 — which `ci.sh` reads as "new
    findings", the one verdict it is NOT. The "could not run" arm (rc=2,
    which ci.sh turns into a build failure with its own message) exists for
    exactly this and never saw it.

    World where it fails: `load_baseline` calling `json.loads` bare.
    """
    base = tmp_path / "b.json"
    base.write_text('{"entries": {"no-member|src/x.py|Thing has no y": 1',
                    encoding="utf-8")          # truncated mid-write
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [])
    monkeypatch.setattr(sys, "argv", ["lint.py"])

    rc = lint_mod.main()

    assert rc == 2, "a gate whose baseline is unreadable must not exit 1"
    err = capsys.readouterr().err
    assert "LINT UNAVAILABLE" in err and "CORRUPT" in err


def test_a_directory_without_an_init_is_still_linted(tmp_path):
    """§4GJ round 4: handed a DIRECTORY, pylint walks packages only, so a
    source subdirectory without `__init__.py` was silently not linted —
    measured: an undefined name in `src/ghost_agent/nopkg/c.py` gave 0
    findings and rc=0, while the identical defect inside a package reddened.
    Latent in this tree (every directory has an `__init__.py`) and therefore
    exactly the kind of hole that is discovered by the commit that opens it.

    World where it fails: passing the directory itself. Note that
    `--recursive=y` is NOT the fix and this pin proves it — with the target
    a package, pylint still takes the package as its unit and reports the
    same 0 findings. Only enumerating the files covers every file.
    """
    pkg = tmp_path / "pkg"
    (pkg / "nopkg").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "nopkg" / "c.py").write_text(
        "def g():\n    return _definitely_undefined_4gj_r4\n", encoding="utf-8")
    (pkg / "inpkg.py").write_text(
        "def h():\n    return _definitely_undefined_in_package\n",
        encoding="utf-8")

    # the enumeration itself: no file under the tree may be missing from it
    assert sorted(Path(p).name for p in lint_mod.lint_targets(pkg)) == \
        ["__init__.py", "c.py", "inpkg.py"]

    findings = lint_mod.run_pylint(target=pkg, jobs=1)
    paths = {Path(m["path"]).name for m in findings
             if m["symbol"] == "undefined-variable"}

    # the in-package file is the control: it reddens either way, so a red
    # here can only be the namespace-package half
    assert "inpkg.py" in paths, findings
    assert "c.py" in paths, (
        "a subdirectory without __init__.py was not linted — pylint walks "
        f"packages, so the files must be enumerated. Saw: {sorted(paths)}")


def test_seeding_is_refused_for_a_symbol_the_baseline_has_EVER_tracked():
    """§4GJ round 4: `already = {symbols with an entry right now}` asked a
    SNAPSHOT. Drive a family to zero, regenerate downward (its entries
    vanish), and `--seed <that symbol>` was legal again — a one-time
    coverage widening re-armed for a family that had already been ratcheted
    to nothing, free to absorb arbitrary new findings.

    World where it fails: a guard that reads only `entries`.
    """
    # ⚠ NOT a family in `TRACKED_FLOOR` (§4GK round 6): the floor would
    # answer for it and these two arms would pass in a world where the
    # baseline's own memory is ignored entirely.
    assert "consider-using-with" not in lint_mod.tracked_symbols({"entries": {}})
    live = {"entries": {"consider-using-with|src/x.py|Consider using with": 1}}
    assert "consider-using-with" in lint_mod.tracked_symbols(live)

    zeroed = {"entries": {}, "tracked_symbols": ["consider-using-with"]}
    assert "consider-using-with" in lint_mod.tracked_symbols(zeroed), \
        "the history forgot a family that was driven to zero"
    assert "never-seen-symbol" not in lint_mod.tracked_symbols(zeroed)
    # §4GK round 6: and with the baseline's own memory gone as well. The
    # first assertion here used to be an EQUALITY (`== {"no-member"}`),
    # which is the same claim as "the file is the only memory" — exactly the
    # property the round-6 measurement exploited with one hand-edit.
    # `TRACKED_FLOOR`, in scripts/lint.py, is the floor now.
    assert "no-member" in lint_mod.tracked_symbols({"entries": {}}), \
        "deleting the key from the baseline re-armed --seed for the family"


def test_the_writer_records_the_history_and_main_honours_it(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """R5 for the guard above: the history only exists if the WRITER emits
    it, and only bites if `main` reads it. Driven end to end with a
    fabricated pylint run — the pre-round-4 pair (no `tracked_symbols`, a
    snapshot guard) lets the second seed through with rc=0."""
    hit = _msg("no-member", message="Instance of 'Widget' has no 'colour' member")
    base = tmp_path / "b.json"
    base.write_text(json.dumps(
        {"entries": {lint_mod.fingerprint(hit): 1}, "notes": {}}),
        encoding="utf-8")
    monkeypatch.setattr(lint_mod, "BASELINE", base)

    # the family is fixed and the baseline regenerated downward to empty
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [])
    monkeypatch.setattr(sys, "argv", ["lint.py", "--write-baseline"])
    assert lint_mod.main() == 0
    written = json.loads(base.read_text())
    assert written["entries"] == {}
    assert "no-member" in written["tracked_symbols"], \
        "the writer dropped the family's history the moment it hit zero"

    # ...and seeding it is STILL refused
    other = _msg("no-member", path="src/ghost_agent/core/other.py",
                 message="Instance of 'Q' has no 'z' member")
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [other])
    monkeypatch.setattr(sys, "argv",
                        ["lint.py", "--write-baseline", "--seed", "no-member"])
    assert lint_mod.main() == 1
    assert "REFUSING to seed already-tracked symbol" in capsys.readouterr().err
    assert json.loads(base.read_text())["entries"] == {}, \
        "the refused seed still wrote the finding into the baseline"


def test_a_missing_linter_exits_two_and_never_reads_as_green(monkeypatch, capsys):
    """Arm 4: an instrument that cannot run must not report success (R6).

    World where it fails: the `LINT UNAVAILABLE` branch returning 0 — which
    would make `ci.sh` treat "pylint is not installed" as a clean tree.
    """
    monkeypatch.setattr(lint_mod, "_pylint_path", lambda: None)
    monkeypatch.setattr(sys, "argv", ["lint.py"])

    rc = lint_mod.main()

    assert rc == 2, "a gate that could not run must not exit 0"
    assert "LINT UNAVAILABLE" in capsys.readouterr().err


def test_the_ci_gate_treats_a_missing_linter_as_a_failure(tmp_path):
    """The consumer half of arm 4 (R5): `scripts/lint.py` returning 2 only
    helps if `ci.sh` acts on it. Pinned as the pair, because a runner that
    reports correctly into a script that ignores it is still a silent pass.

    ⚠ THIS USED TO GREP ci.sh TOO (§4GJ round 5): `"rc -eq 2" in ci` plus
    `"status=1"` inside a 400-character window after it. That passes on a
    ci.sh where the branch is commented out and the window happens to reach
    the next `status=1`, and it says nothing about what the script DOES. It
    now drives the real script with a runner stubbed to each exit code, so
    the three verdicts are distinguished by observation.

    Fails in: a ci.sh whose `rc -eq 2` branch does not set the status —
    verified on a mutated copy, which returns 0 here.
    """
    could_not_run = _run_ci(_ci_sandbox(tmp_path / "two", rc=2))
    assert could_not_run.returncode != 0, (
        "ci.sh treated 'pylint could not run' as a clean tree:\n"
        + could_not_run.stdout[-600:])
    assert "DID NOT RUN" in could_not_run.stderr
    assert "CI FAILED" in could_not_run.stderr

    # findings (rc=1) must fail too — a different branch, same verdict
    findings = _run_ci(_ci_sandbox(tmp_path / "one", rc=1))
    assert findings.returncode != 0, findings.stdout[-600:]
    assert "error-class lint findings" in findings.stderr

    # INVERSE: the identical sandbox with a runner that exits 0 passes, so
    # the reds above can only be the lint exit code.
    clean = _run_ci(_ci_sandbox(tmp_path / "zero", rc=0))
    assert clean.returncode == 0, (clean.stdout[-600:], clean.stderr[-600:])


# ── R1: the class the linter found ──────────────────────────────────────────

def _binding_counts(nodes, owner=None) -> "Counter":
    """name -> how many binding SITES the given scope-local nodes contain.

    Counts rather than a set so the caller can ask "is this name bound
    anywhere OUTSIDE that subtree?" by comparing two counters — which keeps
    the whole walk linear.

    ⚠ BOTH counters must be built from the SAME scope rule. The first
    version counted the try body with `ast.walk` (descending into nested
    defs) and the function with `_own_scope` (not descending), so a name
    imported four times inside nested closures out-counted the function's
    own bindings and every such handler was flagged. One helper now, fed
    scope-local node lists by the caller."""
    bound = Counter()
    for n in nodes:
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound[n.id] += 1
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound[n.name] += 1
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound[(a.asname or a.name).split(".")[0]] += 1
        elif isinstance(n, _SCOPE_NODES) and n is not owner:
            bound[getattr(n, "name", "<lambda>")] += 1
    return bound


def _names_read_in(node) -> set:
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_scope(node, *, root=True):
    """Yield the nodes belonging to `node`'s OWN scope, never descending
    into a nested def/class/lambda.

    Without this the module-level walk re-judged every nested function's
    `try` with the module's (empty) parameter list, and flagged a name that
    is simply a parameter of the enclosing function."""
    if not root and isinstance(node, _SCOPE_NODES):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _own_scope(child, root=False)


def unbound_except_reads(tree: ast.AST) -> list:
    """(function, name, lineno) for every `except` handler that READS a name
    which is bound ONLY inside its own `try` body.

    That is the `_wrote` shape: the handler assumes the try reached the
    assignment, so an exception raised by an EARLIER statement makes the
    handler itself raise NameError — the error path destroyed by the error
    it was written to report.

    Precise about the false positives that matter: a name also bound
    anywhere else in the enclosing function (before the try, in another
    branch, as a parameter, as a comprehension target) is fine, and so is a
    module-level or builtin name.
    """
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        params = set()
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = fn.args
            for arg in (list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs)
                        + ([a.vararg] if a.vararg else [])
                        + ([a.kwarg] if a.kwarg else [])):
                params.add(arg.arg)
        # ONE walk for the whole function (`handle_chat` is 4,400 lines with
        # hundreds of try blocks — re-walking it per try was quadratic and
        # did not finish), and only over THIS scope.
        own = list(_own_scope(fn))
        fn_bound = _binding_counts(own, owner=fn)
        for tryn in own:
            if not isinstance(tryn, ast.Try):
                continue
            body_nodes = [n for stmt in tryn.body for n in _own_scope(stmt)]
            body_bound = _binding_counts(body_nodes, owner=fn)
            for handler in tryn.handlers:
                read = _names_read_in(handler)
                for name in sorted(read & set(body_bound)):
                    if name in params or name.startswith("__"):
                        continue
                    if name in _BUILTIN_NAMES:
                        continue
                    # Bound anywhere else in this function? Then the handler
                    # always has a value and this is not the `_wrote` shape.
                    if fn_bound[name] > body_bound[name]:
                        continue
                    hits.append((getattr(fn, "name", "<module>"), name,
                                 handler.lineno))
    return hits


def _src_files():
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in str(p))


def test_except_handler_reads_a_name_its_try_may_not_bind():
    """R1 enumeration for the `_wrote` class, over the whole package.

    The world where it fails: any handler reading a name the try binds
    late, so the first statement of the try raising turns the handler into
    a NameError. `_wrote` was exactly that until §4GJ pre-bound it.
    """
    offenders = []
    for path in _src_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn, name, lineno in unbound_except_reads(tree):
            offenders.append(f"{path.relative_to(REPO)}:{lineno} "
                             f"in {fn}(): except reads {name!r}, bound only "
                             f"inside the try body")
    assert offenders == [], (
        "exception handlers reading a name their `try` may not have bound:\n  "
        + "\n  ".join(offenders)
        + "\n\nBind the name before the `try` (see tools/file_system.py "
          "`_wrote`): an exception from an earlier statement otherwise "
          "makes the handler raise NameError instead of reporting.")


def test_the_except_binding_enumeration_fires():
    """R7-2: the checker must be shown to RED on the defect it exists for —
    here, the pre-§4GJ shape of `_wrote` reconstructed exactly."""
    broken = ast.parse(
        "def f(p):\n"
        "    try:\n"
        "        path = resolve(p)\n"
        "        wrote = True\n"
        "        write(path)\n"
        "    except Exception as e:\n"
        "        if wrote:\n"
        "            return 'landed'\n"
        "        return 'untouched'\n")
    assert [h[1] for h in unbound_except_reads(broken)] == ["wrote"]

    fixed = ast.parse(
        "def f(p):\n"
        "    wrote = False\n"
        "    try:\n"
        "        path = resolve(p)\n"
        "        wrote = True\n"
        "        write(path)\n"
        "    except Exception as e:\n"
        "        if wrote:\n"
        "            return 'landed'\n"
        "        return 'untouched'\n")
    assert unbound_except_reads(fixed) == []


def test_the_enumeration_does_not_flag_the_ordinary_shapes():
    """Both worlds must not agree (R4): a checker that flags nothing real
    passes vacuously, one that flags everything gets disabled."""
    ok_param = ast.parse(
        "def f(x):\n"
        "    try:\n"
        "        x = compute()\n"
        "    except Exception:\n"
        "        return x\n")
    assert unbound_except_reads(ok_param) == [], "a parameter is always bound"

    ok_before = ast.parse(
        "def f():\n"
        "    y = 1\n"
        "    try:\n"
        "        y = compute()\n"
        "    except Exception:\n"
        "        return y\n")
    assert unbound_except_reads(ok_before) == []

    ok_unrelated = ast.parse(
        "def f():\n"
        "    try:\n"
        "        z = compute()\n"
        "    except Exception as e:\n"
        "        return str(e)\n")
    assert unbound_except_reads(ok_unrelated) == []


# ── the one runtime defect the linter found, pinned behaviourally ───────────

@pytest.mark.asyncio
async def test_auto_promote_reports_a_path_escape_instead_of_raising(tmp_path):
    """`tools/file_system.py` `_wrote` — the instance behind the R1
    enumeration above, pinned where a user would feel it.

    The replace→write auto-promote binds `_wrote` inside its `try`, but the
    two statements before it can raise: `_get_safe_path` on a path escape,
    and the markdown extractor on malformed input. The handler reads
    `_wrote` to decide whether the file was touched, so before §4GJ a path
    escape raised `NameError: local variable '_wrote' referenced before
    assignment` — the error path destroyed by the error it exists to
    report. Verified against the pre-fix source: it raises; this asserts it
    reports.
    """
    from ghost_agent.tools.file_system import (
        _looks_like_complete_python_module, tool_replace_text)

    module = ("import os\nimport sys\n\n\n"
              "def alpha():\n    return os.getcwd()\n\n\n"
              "def beta():\n    return sys.version\n\n\n"
              "class Gamma:\n    pass\n")
    # Fixture guard (R4): if this stops looking like a module the test would
    # pass vacuously on the early-return branch instead of the promote path.
    assert _looks_like_complete_python_module(module)

    out = await tool_replace_text("../../escape.py", module, None, tmp_path)

    text = str(out)
    assert "auto-promote" in text and "Security Error" in text, text
    assert "NameError" not in text
    assert not (tmp_path.parent.parent / "escape.py").exists()


# ── §4GJ follow-up: the seed path widens COVERAGE, never tolerates growth ───
#
# `test_seeding_is_refused_for_an_already_tracked_symbol` lived here until
# §4GJ round 5. It shelled out to `scripts/lint.py --write-baseline --seed
# no-member` against a scratch baseline and asserted rc=1 — but it spent a
# FULL REAL PYLINT RUN (~50s, and the run is what `--write-baseline` does
# first) to reach a guard that fires before any of it matters, and its own
# comment admitted it could not tell which world it was in: "either it
# refused (the env override is honoured) or the real baseline also tracks
# no-member". A test that passes in both worlds is not evidence for either
# (R4). The millisecond twins below and above it — `test_seeding_is_refused_
# for_a_symbol_the_baseline_has_EVER_tracked` (the rule) and
# `test_the_writer_records_the_history_and_main_honours_it` (the writer and
# `main`, driven end to end with a fabricated pylint run) — pin the same
# guard, unambiguously and without the run.


def test_the_run_actually_reports_the_unused_families(tmp_path):
    """The §4GJ widening, pinned on the RUN rather than on the committed
    baseline. ~138 unused-import/unused-variable findings were outside every
    gate — the noise a real undefined name hides in — and are ratcheted now.

    ⚠ THE OBVIOUS ASSERTION IS VACUOUS. Checking that
    `tests/lint_baseline.json` CONTAINS unused-import entries stays true
    after the families are dropped from the enable list, because the
    baseline is a committed file: the coverage can be deleted and the
    assertion still passes. This drives `run_pylint` over a scratch module
    with one unused import and one unused local and asserts the findings
    come BACK — the only form that fails when the enable list narrows.

    World where it fails: `--enable=E` (the pre-widening run).
    Cost: ~0.5s, one file, `-j 1`.
    """
    # ⚠ the local must NOT be named `unused_*`: pylint's default
    # `dummy-variables-rgx` exempts that prefix as an intentional discard,
    # so the obvious name made this test pass for the wrong reason.
    canary = tmp_path / "canary.py"
    canary.write_text(
        "import os\n"
        "\n"
        "\n"
        "def g():\n"
        "    y = 5\n"
        "    return 0\n",
        encoding="utf-8")

    symbols = {m["symbol"] for m in lint_mod.run_pylint(target=canary, jobs=1)}

    assert "unused-import" in symbols, (
        "the runner no longer reports unused-import — the enable list has "
        f"narrowed and ~114 findings left the ratchet. Saw: {sorted(symbols)}")
    assert "unused-variable" in symbols, (
        "the runner no longer reports unused-variable — the enable list has "
        f"narrowed and ~23 findings left the ratchet. Saw: {sorted(symbols)}")


def test_the_error_class_is_still_the_hard_half_of_the_run(tmp_path):
    """The widening must not have traded the error class away: the same
    scratch route proves `E` is still enabled, so a real undefined name is
    still seen. Inverse of the test above — together they pin the whole
    enable list by behaviour.

    World where it fails: an enable list of `W0611,W0612` alone.
    """
    canary = tmp_path / "canary_err.py"
    canary.write_text("def g():\n    return _definitely_undefined_4gj\n",
                      encoding="utf-8")

    symbols = {m["symbol"] for m in lint_mod.run_pylint(target=canary, jobs=1)}

    assert "undefined-variable" in symbols, (
        "the runner no longer reports undefined-variable — the error class "
        f"has left the run. Saw: {sorted(symbols)}")


# ── §4GK round 6: the off switch moved from `allowed_slack` to the history ──
#
# Round 5 closed `allowed_slack` for the right reason — a gate must not read
# its own ceiling out of the file it polices — and left the `--seed` guard's
# memory, `tracked_symbols`, in that same file, unchecked for shrinkage. The
# reviewer's measurement, replayed below: bootstrap one `no-member`, fix it,
# regenerate downward, `--seed no-member` -> rc 1; hand-edit
# `"tracked_symbols": []`, same command -> "seeding 2 new fingerprint(s)",
# rc 0, gate OK.


def test_seeding_stays_refused_after_the_history_key_is_edited_away(
        tmp_path, monkeypatch, capsys):
    """The measurement itself, end to end through `main` with a fabricated
    pylint run — hand-edit included.

    The floor of the history is `TRACKED_FLOOR` in `scripts/lint.py`, so
    deleting the family from the JSON buys nothing: widening the gate's
    coverage is a code change a reviewer sees, which is exactly the rule
    round 5 wrote for `ALLOWED_SLACK` and did not apply here.

    World where it fails: `tracked_symbols()` reading only the baseline
    (rc=0, "seeding 2 new fingerprint(s)", and two brand-new findings land
    in a family that was already at zero).
    """
    base = tmp_path / "b.json"
    base.write_text(json.dumps(
        {"entries": {}, "tracked_symbols": ["no-member"], "notes": {}}),
        encoding="utf-8")
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    fresh = [_msg("no-member", path="src/ghost_agent/core/a.py",
                  message="Instance of 'A' has no 'x' member"),
             _msg("no-member", path="src/ghost_agent/core/b.py",
                  message="Instance of 'B' has no 'y' member")]
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: fresh)

    # the hand-edit: the family's only written memory, deleted
    base.write_text(json.dumps(
        {"entries": {}, "tracked_symbols": [], "notes": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["lint.py", "--write-baseline", "--seed", "no-member",
                         "--seed-reason", "laundering"])

    assert lint_mod.main() == 1
    err = capsys.readouterr().err
    assert "REFUSING to seed already-tracked symbol" in err, err
    assert json.loads(base.read_text())["entries"] == {}, \
        "the refused seed wrote the findings into the baseline anyway"

    # INVERSE: a family NOTHING has ever tracked is still seedable — the
    # guard must widen coverage, not forbid it. (§4GK round 7 moved the
    # PERMISSION into `SEED_GRANTS`, so the inverse now needs the grant the
    # operator would write; the refusal above is unchanged by it.)
    monkeypatch.setattr(lint_mod, "SEED_GRANTS",
                        {"consider-using-with": "never measured in this tree"})
    never = [_msg("consider-using-with", path="src/ghost_agent/core/a.py",
                  message="Consider using with")]
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: never)
    monkeypatch.setattr(sys, "argv",
                        ["lint.py", "--write-baseline", "--seed",
                         "consider-using-with", "--seed-reason",
                         "never measured in this tree"])
    assert lint_mod.main() == 0
    assert len(json.loads(base.read_text())["entries"]) == 1


def test_a_seed_leaves_a_stamped_record_behind(tmp_path, monkeypatch, capsys):
    """The asymmetry with the sibling gate: the pin ratchet forces
    `--migrate "reason"` to be STAMPED for an upward write, and `--seed` —
    the lint gate's ONLY upward write — stamped nothing. "Two brand-new
    findings absorbed into a family already ratcheted to zero, leaving NO
    record in the file."

    World where it fails: a `--seed` that needs no reason and writes no
    `seeded` record (the write succeeds and the file looks like an ordinary
    downward regenerate).
    """
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"entries": {}, "notes": {}}), encoding="utf-8")
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    # the code-side permission §4GK round 7 requires; the stamped RECORD this
    # test is about is a separate demand and is still the baseline's job
    monkeypatch.setattr(lint_mod, "SEED_GRANTS",
                        {"consider-using-with": "new family, never measured"})
    hit = _msg("consider-using-with", path="src/ghost_agent/core/a.py",
               message="Consider using with")
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [hit])

    monkeypatch.setattr(sys, "argv", ["lint.py", "--write-baseline",
                                      "--seed", "consider-using-with"])
    assert lint_mod.main() == 1, "a seed with no reason was accepted"
    assert "--seed needs --seed-reason" in capsys.readouterr().err
    assert json.loads(base.read_text())["entries"] == {}

    monkeypatch.setattr(sys, "argv",
                        ["lint.py", "--write-baseline", "--seed",
                         "consider-using-with", "--seed-reason",
                         "new family, never measured here"])
    assert lint_mod.main() == 0
    written = json.loads(base.read_text())
    assert written["seeded"] == [{"symbols": ["consider-using-with"],
                                  "reason": "new family, never measured here",
                                  "fingerprints": 1}], written.get("seeded")
    # and the record is itself history: the family cannot be seeded twice
    assert "consider-using-with" in lint_mod.tracked_symbols(
        {"seeded": written["seeded"]})


def test_a_history_list_edited_below_its_own_entries_fails_the_gate(tmp_path):
    """`entries` is checked for shrinkage on every run; `tracked_symbols`
    was not checked at all. A list that contradicts the entries or seed
    records beneath it was edited by hand, and the only thing that edit buys
    is `--seed` for a family already ratcheted to zero — so it is a verdict,
    not a curiosity.

    World where it fails: an `evaluate` that never looks at the key
    (`ok: True`, gate green, the history quietly gone).
    """
    hit = _msg("no-member", message="Instance of 'X' has no 'y' member")
    fp = lint_mod.fingerprint(hit)
    edited = {"entries": {fp: 1}, "tracked_symbols": [], "notes": {}}

    verdict = lint_mod.evaluate([hit], edited)

    assert verdict["ok"] is False
    assert verdict["tracked_symbols_gap"] == ["no-member"], verdict
    # the other arms are silent, so the failure can only be this one
    assert not verdict["new_fingerprints"] and not verdict["grown_fingerprints"]
    assert verdict["slack_total"] == 0

    # INVERSE 1: the same file with the key CONSISTENT -> green.
    consistent = {"entries": {fp: 1}, "tracked_symbols": ["no-member"],
                  "notes": {}}
    assert lint_mod.evaluate([hit], consistent)["ok"] is True
    # INVERSE 2: no key at all is the COMMITTED baseline's shape — it
    # predates round 5's writer and must not be read as a deleted history.
    assert lint_mod.evaluate([hit], {"entries": {fp: 1}})["ok"] is True


def test_a_baseline_that_is_valid_json_but_not_an_object_is_UNAVAILABLE(
        tmp_path, monkeypatch, capsys):
    """§4GK round 6. Round 4 guarded the DECODE and round 5 guarded the
    `allowed_slack` KEY; the CONTAINER was unguarded, so `[]` parsed fine,
    reached `evaluate`, and died with `AttributeError: 'list' object has no
    attribute 'get'` — a traceback that reads as rc=1, "new findings", which
    is the one verdict it is not. An empty or truncated file already
    answered rc=2 correctly; this is the same fact in a different shape.

    World where it fails: `return json.loads(...)` with no isinstance check
    (rc=1 and a traceback, indistinguishable from a dirty tree).
    """
    base = tmp_path / "b.json"
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [])
    monkeypatch.setattr(sys, "argv", ["lint.py"])

    for body in ("[]", '"a string"', "null", "3"):
        base.write_text(body, encoding="utf-8")
        with pytest.raises(RuntimeError) as exc:
            lint_mod.load_baseline()
        assert "not a JSON OBJECT" in str(exc.value), body
        assert lint_mod.main() == 2, body
        assert "LINT UNAVAILABLE" in capsys.readouterr().err

    # a non-object `entries` is the same fact one level down
    base.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        lint_mod.load_baseline()
    assert "non-object" in str(exc.value)
    assert lint_mod.main() == 2

    # INVERSE: a real object still loads, and the gate still votes.
    base.write_text(json.dumps({"entries": {}, "notes": {}}), encoding="utf-8")
    assert lint_mod.load_baseline() == {"entries": {}, "notes": {}}
    assert lint_mod.main() == 0


# ── §4GK round 7: the floor cannot be made of what the baseline can delete ──
#
# Round 6 put the seed guard's memory behind `TRACKED_FLOOR` — for the eight
# families that were in it. Everything seeded AFTER it is remembered only by
# `tracked_symbols` and `seeded`, and `history_gaps` derives its evidence
# from those same two keys, so deleting BOTH leaves no contradiction to find.


def test_a_seed_survives_no_hand_edit_because_the_permission_is_in_the_code(
        tmp_path, monkeypatch, capsys):
    """The round-6 measurement, replayed one family later.

    A family seeded after round 6 lives in `tracked_symbols` and `seeded`
    and NOWHERE else: `TRACKED_FLOOR` names the eight the tree already had.
    Delete both keys and round 6's two arms both fall silent — the seed
    guard has no memory of it, and `history_gaps` reads its evidence out of
    the very keys that were deleted. Measured on the round-6 code: rc **0**,
    "seeding 1 new fingerprint(s)", the findings land in a family that was
    already ratcheted to zero, `evaluate()["ok"] is True`, and no record is
    left anywhere.

    The permission therefore is not in the baseline at all. `SEED_GRANTS`
    lives in `scripts/lint.py`, a hand-edit of the JSON cannot write one,
    and a grant that has been SPENT fails the gate until it is deleted.

    World where it fails: a `--seed` gated only on the baseline's own
    history (the hand-edit below is accepted, rc=0).
    """
    base = tmp_path / "b.json"
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    monkeypatch.setattr(lint_mod, "SEED_GRANTS", {})
    fresh = [_msg("consider-using-with", path="src/ghost_agent/core/a.py",
                  message="Consider using with")]
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: fresh)
    monkeypatch.setattr(sys, "argv",
                        ["lint.py", "--write-baseline", "--seed",
                         "consider-using-with", "--seed-reason", "laundering"])

    # the hand-edit: the family's ENTIRE written memory, deleted — the
    # history key AND the seed record that would contradict it
    base.write_text(json.dumps({"entries": {}, "notes": {}}), encoding="utf-8")

    assert lint_mod.main() == 1
    assert "REFUSING to seed ungranted symbol" in capsys.readouterr().err
    assert json.loads(base.read_text())["entries"] == {}, \
        "the refused seed wrote the findings into the baseline anyway"

    # INVERSE 1: the grant is what widens coverage, and it still does —
    # this must not become "seeding is impossible".
    monkeypatch.setattr(lint_mod, "SEED_GRANTS",
                        {"consider-using-with": "never measured here"})
    assert lint_mod.main() == 0
    written = json.loads(base.read_text())
    assert len(written["entries"]) == 1, written
    assert written["seeded"][0]["symbols"] == ["consider-using-with"]

    # INVERSE 2: and the grant EXPIRES. It is one write long; left in the
    # file it is a standing permission to absorb new findings into a family
    # that is now measured, so the gate is red until the line is deleted.
    spent = lint_mod.evaluate(fresh, lint_mod.load_baseline())
    assert spent["spent_seed_grants"] == ["consider-using-with"], spent
    assert spent["ok"] is False
    # the other arms are silent, so the failure can only be this one
    assert not spent["new_fingerprints"] and not spent["grown_fingerprints"]
    assert spent["slack_total"] == 0 and not spent["tracked_symbols_gap"]

    monkeypatch.setattr(lint_mod, "SEED_GRANTS", {})
    assert lint_mod.evaluate(fresh, lint_mod.load_baseline())["ok"] is True


def test_a_baseline_whose_VALUES_are_corrupt_says_the_gate_could_not_run(
        tmp_path, monkeypatch, capsys):
    """Round 6 guarded the CONTAINER and stopped there: a baseline that is a
    JSON object with garbage inside it reached the arithmetic raw.

    Measured on the round-6 code: `{"entries": {"sym|path|msg": "3"}}` gives
    `TypeError: '>' not supported between instances of 'str' and 'int'` and
    **exit 1** — which `scripts/ci.sh:102` reports as "error-class lint
    findings", i.e. a dirty tree, when the truth is that the gate did not
    run. `null` and `[1]` counts do the same, and `{"tracked_symbols": 5}`
    dies in `set(5)`. Round 6's own comment names that verdict as the one it
    must never produce.

    World where it fails: an `isinstance(data, dict)` check that never looks
    inside (rc=1 and a traceback, indistinguishable from new findings).
    """
    base = tmp_path / "b.json"
    monkeypatch.setattr(lint_mod, "BASELINE", base)
    monkeypatch.setattr(lint_mod, "run_pylint", lambda **kw: [])
    monkeypatch.setattr(sys, "argv", ["lint.py"])

    corrupt = [
        '{"entries": {"no-member|a.py|m": "3"}}',      # a count as a string
        '{"entries": {"no-member|a.py|m": null}}',     # ...or absent
        '{"entries": {"no-member|a.py|m": [1]}}',      # ...or a list
        '{"entries": {"no-member|a.py|m": -1}}',       # ...or negative
        '{"entries": {}, "tracked_symbols": 5}',       # the seed memory
        '{"entries": {}, "tracked_symbols": [5]}',
        '{"entries": {}, "seeded": 3}',                # the audit trail
        '{"entries": {}, "seeded": [{"symbols": 5}]}',
        '{"entries": {}, "notes": []}',                # the annotations
    ]
    for body in corrupt:
        base.write_text(body, encoding="utf-8")
        with pytest.raises(RuntimeError):
            lint_mod.load_baseline()
        assert lint_mod.main() == 2, body
        assert "LINT UNAVAILABLE" in capsys.readouterr().err, body

    # INVERSE: the shapes the writer really emits still load and still vote.
    base.write_text(json.dumps(
        {"entries": {"no-member|a.py|m": 1}, "tracked_symbols": ["no-member"],
         "seeded": [{"symbols": ["no-member"], "reason": "r",
                     "fingerprints": 1}],
         "notes": {}}), encoding="utf-8")
    monkeypatch.setattr(lint_mod, "run_pylint",
                        lambda **kw: [_msg("no-member", path="a.py", message="m")])
    assert lint_mod.main() == 0
