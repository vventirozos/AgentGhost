"""§4GJ — the source-text pin RATCHET.

R4 hard-rejects **source-text assertions** (`assert "text" in src`,
`src.index(...)`, a regex over the source, a character-window slice): a
mutant that deletes the code and leaves the string in a comment survives
them, and they redden on any refactor that does not change behaviour. The
journal blames exactly this class for 73 phantom failures in one session.

But `inspect.getsource` also appears for the OPPOSITE reason. R1 *requires*
class-level enumerations, and the only way to walk a class of call sites is
to read the module's source and `ast.parse` it — §4GG/§4GH/§4GI all ship
such enumerations. A blanket ban on `getsource` would forbid the protocol's
own mechanism.

So this file does not ban a name. It classifies every source-obtaining
expression in the suite by **where its value ends up**:

* **PARSED** — the value reaches `ast.parse` / `compile`, directly or
  through local rebinding and string pass-throughs (`.lstrip()`,
  `textwrap.dedent`, `.replace()` — the last is how an enumeration is shown
  to FIRE). These are legal: they break on a rename, not on a behaviour
  change, and they close a class.
* **TEXTUAL** — anything else. Membership, `.index`, `.split`, a slice, a
  regex, an f-string... and **every use this analysis cannot prove is a
  parse**. Fail-closed by construction: an unclassifiable use is a text pin.

The ratchet then enforces one property: **the textual count can only go
down**, in total and per file. A new test may not add one, and a rewrite
may not hide growth in one file behind a shrink in another.

The unit is the textual **use**, not the call site (§4GJ round 4): a
hoisted `_SRC = getsource(M)` read by twenty assertions is twenty text
pins, exactly as the twenty inline reads it replaces are. Counting sites
let a pure refactor bank the difference and spend it again.

HONEST LIMITS of the analysis (all of them make it over-report, never
under-report, which is the safe direction):

* it is **per-scope**. A source string handed to a helper function, stored
  on an object, or closed over is not followed — it counts as TEXTUAL.
* **one binding hop.** `a = getsource(M); b = a; ast.parse(b)` is not
  followed past `a`; `b`'s parse does not rescue it.
* a source expression with BOTH a parse and a textual use is TEXTUAL, and
  ALL of its uses count as text pins. That is the fail-closed rule, and it
  is also the honest one: the textual half is still a text pin, and a
  parse sitting next to it does not launder the rest.
* it sees `__file__`-derived reads, `getsource`/`getsourcelines` (aliases
  included) and `linecache.getlines`/`inspect.findsource` on a source path.
  A test that reaches a module's source by some other route is invisible.

THE NON-PYTHON HOLE IS CLOSED (§4GO, 2026-09-14). `_looks_like_python_source`
gates the whole detector, and until now it recognised only `.py`, `src`,
`ghost_agent` and `mod.__file__` — so a text pin against `interface/static/
app.js`, `bin/start-ghost-agent.sh` or a `scripts/*.py` helper was invisible
however textual it was. §4GJ round 5 measured the hole and deliberately left
it open, because admitting the pins is a MEASUREMENT change that needs an
upward `--migrate` on a quiet tree by someone who can check the files it
touches. That was done here: the three code directories are now recognised,
which admits **45 textual uses across 14 files** — `interface` 41, `scripts`
2, `bin` 2 — every one of them a real read of code under test (the web
client's `app.js` and `index.html`, the launcher's exec line, a probe
script). Baseline migrated 1312 → 1357.

What stays OUT, on purpose: `docs/*.html`. A documentation link-checker is
not a pin on the code under test, and that is the reason `_PY_SOURCE_RE`
exists at all. Non-Python code EXTENSIONS as a blanket rule (the 129-across-30
figure round 5 measured) would drag those in with them.

A textual pin is not automatically worthless — `test_env_timeout_constants`
greps `src` for a bare `float(os.environ...)` and caught a real import-time
defect this session. The ratchet does not demand conversion; it freezes the
count so the ratio can only improve.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
BASELINE_PATH = TESTS_DIR / "pin_quality_baseline.json"

#: How many pins the baseline may claim that the suite no longer has. ZERO,
#: and it lives HERE, not in the baseline (§4GJ round 5). `scripts/lint.py`
#: grew the same defect in round 4 and it was measured there: a gate that
#: reads its own ceiling out of the file it polices has an off switch, and
#: one `"allowed_slack": 9999` disarms both the script and the pytest half
#: at once. This file's committed baseline carries the key at 0, which is
#: harmless and is dropped on the next write; a NON-ZERO one fails the gate.
ALLOWED_SLACK = 0

#: Calls that consume SOURCE TEXT and produce a tree. Reaching one of these
#: is what makes a source read an R1 enumeration rather than an R4 text pin.
_PARSE_SINKS = {"ast.parse", "parse", "compile"}

#: String operations that pass the source through unchanged for parsing
#: purposes. `.replace` is here because "show the enumeration fires" edits
#: the source and re-parses it (§R R7-2).
_PASS_THROUGH = {"lstrip", "strip", "rstrip", "dedent", "replace", "encode",
                 "decode", "join", "format"}

#: Attribute calls that read a file's bytes/text.
_READ_CALLS = {"read_text", "read_bytes", "read"}

#: Functions that return a file's TEXT given its PATH. `inspect.getsource`
#: is matched by its own name because it takes a module or a function;
#: these take the path, so it is the ARGUMENT that identifies them.
#:
#: ⚠ `linecache.getlines(mod.__file__)` WAS INVISIBLE (§4GJ round 5). The
#: matcher knew `getsource`, `getsourcelines` and `.read_text()`, so
#: `''.join(linecache.getlines(mod.__file__))` followed by a membership test
#: was classified as nothing at all — confirmed GREEN against the real
#: baseline. Same fail-OPEN direction as the aliased `getsource` round 4
#: closed: a source read this analysis does not RECOGNISE never reaches the
#: fail-closed rule about its uses. Costs 0 on the committed suite (measured)
#: — the route is simply shut.
_PATH_TEXT_FUNCS = {"getlines", "findsource"}

#: A path expression that plausibly names PYTHON SOURCE. Without this the
#: detector also claimed `docs/*.html` link-checkers and other file-content
#: tests, which are not pins on the code under test at all.
#: ⚠ The three code DIRECTORIES are matched as path segments, not as bare
#: words anywhere in the expression, and they were measured one at a time
#: before being admitted (§4GO): a fixture path that merely contains a `bin`
#: directory must not read as source. The existing fixture-root analysis
#: already excludes tmp_path-rooted reads, which is what keeps
#: `(dst / "bin" / "run.sh").read_text()` out of the count.
_PY_SOURCE_RE = __import__("re").compile(
    r"\.py\b|ghost_agent|\bsrc\b|\w\.__file__"
    r"|\binterface\b|\bscripts\b|\bbin\b")


def _dotted(node: ast.AST) -> str:
    """`ast.parse` for an Attribute chain, `parse` for a bare Name, "" else."""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return ""


def _looks_like_python_source(expr_text: str) -> bool:
    """True when a path expression names Python source: a module's
    `__file__` (`mod.__file__`, not a bare test-local `__file__`), anything
    under `src`/`ghost_agent`, or an explicit `.py`."""
    return bool(_PY_SOURCE_RE.search(expr_text or ""))


#: pytest temp-dir fixtures. A file a test CREATES under one of these is a
#: fixture, never the code under test — even when it is named `*.py`.
_FIXTURE_SEEDS = {"tmp_path", "tmpdir", "tmp_path_factory", "tmpdir_factory"}


def _fixture_names(tree: ast.AST) -> set:
    """Names of `@pytest.fixture` functions defined in this file. A test that
    takes one as a parameter is handed whatever it yields — very often a
    `tmp_path`-rooted workspace — so reads through it are fixture I/O. Without
    this, a fixture called `sandbox` yielding `tmp_path / "sb"` made every
    `(sandbox / "app.js").read_text()` look like a source pin."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if "fixture" in _dotted(dec):
                out.add(node.name)
                break
    return out


def _fixture_roots(tree: ast.AST) -> set:
    """Names that carry a pytest temp directory, transitively. Reads of
    these are fixture I/O, not source-text pins."""
    roots = set(_FIXTURE_SEEDS) | _fixture_names(tree)
    for _ in range(3):
        for node in ast.walk(tree):
            targets, value = [], None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                targets, value = [node.target], node.iter
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                targets, value = [node.optional_vars], node.context_expr
            if value is None:
                continue
            if not any(isinstance(n, ast.Name) and n.id in roots
                       for n in ast.walk(value)):
                continue
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        roots.add(sub.id)
    return roots


#: The source-reading functions of `inspect`, by their own names.
_SOURCE_FUNCS = {"getsource", "getsourcelines"}


def _source_aliases(tree: ast.AST) -> set:
    """Local names bound to `inspect.getsource`/`getsourcelines`.

    ⚠ AN ALIASED getsource WAS INVISIBLE (§4GJ round 4). The matcher below
    compares the CALLED NAME against `{"getsource", "getsourcelines"}`, so
    `from inspect import getsource as gs` followed by
    `assert "x" in gs(M)` was classified as nothing at all — confirmed GREEN
    against the real baseline. That is the fail-OPEN direction, in the one
    place this file's docstring promises fail-closed ("every use this
    analysis cannot prove is a parse" counts as textual). An import is an
    AST node; read it rather than trusting the spelling at the call."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "inspect":
            for a in node.names:
                if a.name in _SOURCE_FUNCS:
                    out.add(a.asname or a.name)
    return out


def _is_source_call(node: ast.AST, source_names: set,
                    fixture_names: set = frozenset(),
                    alias_names: set = frozenset()) -> bool:
    """True when `node` is a call that yields a MODULE/FUNCTION's own source.

    Two shapes, both AST-matched (never a text grep, so a `__file__` inside
    a string literal — `test_4ec_evaluator_pins` is full of them as
    fixtures — is correctly ignored):
      * `inspect.getsource(X)` / `getsourcelines(X)`, under its own name or
        any local alias `_source_aliases` found;
      * a `.read_text()`/`.read()` whose receiver mentions `__file__`, or
        whose receiver is a name derived from a source root in this scope;
      * a `_PATH_TEXT_FUNCS` call (`linecache.getlines`, `inspect.findsource`)
        whose first ARGUMENT names Python source — the shape that reads a
        module's text without ever mentioning `getsource`.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name in _SOURCE_FUNCS or name in alias_names:
        return True
    if name in _READ_CALLS and isinstance(func, ast.Attribute):
        return _names_source(func.value, source_names, fixture_names)
    if name in _PATH_TEXT_FUNCS and node.args:
        return _names_source(node.args[0], source_names, fixture_names)
    return False


def _names_source(expr: ast.AST, source_names: set, fixture_names: set) -> bool:
    """True when `expr` is a path expression naming Python source — and not
    a file the test itself created under a pytest temp dir."""
    if any(isinstance(n, ast.Name) and n.id in fixture_names
           for n in ast.walk(expr)):
        return False                # a file the test itself created
    rendered = _dotted(expr)
    if _looks_like_python_source(rendered):
        return True
    root = rendered.split(".")[0].split("[")[0].split("(")[0]
    return root in source_names


def _source_roots(tree: ast.AST) -> set:
    """Names bound to something derived from `__file__` (a source root, or a
    path iterated out of one). Deliberately generous: a name we wrongly
    consider a source root only ever makes us classify MORE uses."""
    roots = set()
    for _ in range(2):          # two passes: roots can be defined from roots
        for node in ast.walk(tree):
            targets, value = [], None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                targets, value = [node.target], node.iter
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                targets, value = [node.optional_vars], node.context_expr
            if value is None:
                continue
            rendered = _dotted(value)
            # "derived" is an AST question, not a substring one: does the
            # value actually READ a known source-root name? (A string test
            # made every name containing "src" a root and the count blew up.)
            derived = any(isinstance(n, ast.Name) and n.id in roots
                          for n in ast.walk(value))
            if not _looks_like_python_source(rendered) and not derived:
                continue
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        roots.add(sub.id)
    return roots


def _parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _reaches_parse(node: ast.AST, parents: dict) -> bool:
    """Walk UP from an expression through string pass-throughs; True when it
    lands as an argument to a parse sink. Anything else is textual."""
    cur = node
    for _ in range(12):
        parent = parents.get(cur)
        if parent is None:
            return False
        # `src.lstrip()` — a METHOD call on the source. The receiver check
        # must compare against `cur`, so the Attribute and its Call are
        # consumed in one step (advancing through the Attribute alone loses
        # the identity that proves `cur` is the receiver).
        if isinstance(parent, ast.Attribute) and parent.value is cur:
            grand = parents.get(parent)
            if isinstance(grand, ast.Call) and grand.func is parent \
                    and parent.attr in _PASS_THROUGH:
                cur = grand
                continue
            return False
        if isinstance(parent, ast.Call):
            fname = _dotted(parent.func)
            short = fname.split(".")[-1]
            if fname in _PARSE_SINKS or short in _PARSE_SINKS:
                # only as an ARGUMENT — `src.parse()` is not ast.parse(src)
                return any(cur is a for a in parent.args)
            if short in _PASS_THROUGH and any(cur is a for a in parent.args):
                cur = parent
                continue
            return False
        return False
    return False


def _enclosing_scope(node: ast.AST, parents: dict) -> ast.AST:
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module,
                            ast.ClassDef, ast.Lambda)):
            return cur
    return cur


def _scope_chain(node: ast.AST, parents: dict) -> list:
    """Every scope enclosing `node`, INNERMOST FIRST, ending at the Module.

    `_enclosing_scope` answers "which scope is this in"; allocating a use to
    exactly one binding site needs the whole chain, because the site that
    owns a load is the one in the innermost scope that binds the name.
    """
    chain, cur = [], node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module,
                            ast.ClassDef, ast.Lambda)):
            chain.append(cur)
    return chain


def _allocate_uses(sites: list, parents: dict, tree: ast.AST) -> dict:
    """site -> the textual uses it OWNS. Every load belongs to exactly one.

    ⚠ THE USES WERE COUNTED ONCE PER SITE, NOT ONCE (§4GJ round 5). Round 4
    moved the unit from the call site to the textual USE — correctly — but
    read the uses as "every Load of the bound name in the enclosing scope",
    per site. Two reads that rebind ONE name in ONE scope therefore each
    claimed BOTH loads. Measured on the round-4 classifier:

        src = getsource(M); assert 'a' in src
        src = getsource(N); assert 'b' in src     -> textual 4, for 2 pins

    and deleting one of the two assertions dropped it to 2 — a two-pin
    refund for removing one assertion, bankable against the per-file
    ratchet. That is the refactor-banks-pins defect round 4 set out to
    close, reappearing inside the fix (the same shape held across scopes: a
    module-level `_S` and a function-local `_S` both claimed the module
    load). A sweep of the committed suite found no instance today, so 1312
    is not inflated — latent, and spendable by the first person to write one.

    Allocation rule, flow-insensitively: the owner of a load is the site
    binding that name in the INNERMOST scope containing the load; among
    those, the LAST site at or before the load; failing that, the first
    after it (a load before every binding is still someone's use, and
    dropping it would under-report, which is the unsafe direction).
    """
    order = sorted(sites, key=lambda n: (n.lineno, n.col_offset))
    rank = {id(n): i for i, n in enumerate(order)}
    binds = {}                     # site -> (scope, bound name)
    for node in sites:
        parent = parents.get(node)
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1 \
                and isinstance(parent.targets[0], ast.Name):
            binds[id(node)] = (_enclosing_scope(node, parents),
                               parent.targets[0].id)
    claims = {id(n): [] for n in sites}
    if not binds:
        return claims
    for load in ast.walk(tree):
        if not (isinstance(load, ast.Name) and isinstance(load.ctx, ast.Load)):
            continue
        chain = _scope_chain(load, parents)
        lpos = (load.lineno, load.col_offset)
        best = None
        for node in order:
            entry = binds.get(id(node))
            if entry is None or entry[1] != load.id or entry[0] not in chain:
                continue
            pos = (node.lineno, node.col_offset)
            before = pos <= lpos
            key = (chain.index(entry[0]), 0 if before else 1,
                   -rank[id(node)] if before else rank[id(node)])
            if best is None or key < best[0]:
                best = (key, node)
        if best is not None:
            claims[id(best[1])].append(load)
    return claims


def classify_file(path: Path) -> dict:
    """`{"textual": n, "parsed": n, "sites": {(line, col): (verdict, uses)}}`
    for one test file.

    ⚠ THE KEY IS (line, col), NOT THE LINE (§4GJ round 4). `verdicts[node.
    lineno] = ...` collapsed two source-obtaining calls that START on the
    same line into one verdict — and the suite really contains them:
    `test_sandbox_chromium_gate.py` line 108 is
    `getsource(A) + getsource(\\n    B)`. Confirmed end to end against the
    real baseline: a new text pin on its own line goes RED, and the SAME pin
    appended to an existing pin's line stays GREEN. Three pins were hiding
    behind that collapse (633 counted vs 636 present).

    ⚠ THE UNIT IS THE TEXTUAL USE, NOT THE CALL SITE (§4GJ round 4). A
    module-level `_SRC = inspect.getsource(M)` plus twenty
    `assert "n" in _SRC` counted ONE, where the same twenty assertions
    written inline counted twenty. So a pure refactor — hoist the read, keep
    every assertion — banked a nineteen-pin "improvement" with no
    behavioural change, and then re-admitted nineteen assertions for free.
    Counting the USES makes the number invariant under that refactor, which
    is the only property a ratchet actually needs. Recounting the committed
    suite this way moved the total from 633 to 1312: those 679 were always
    source-text assertions; the old unit could not see them.

    The fail-closed rule is unchanged and now applies to all of a site's
    uses: if any use is textual the site is textual, and every use it owns
    counts as a text pin.

    ⚠ AND A USE BELONGS TO ONE SITE (§4GJ round 5). Round 4 read the uses
    per site as "every Load of the bound name in the enclosing scope", so
    two reads rebinding one name in one scope each claimed both loads —
    2 assertions counted 4, and deleting one refunded 2. `_allocate_uses`
    hands each load to exactly one site; see its docstring.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
        return {"textual": 0, "parsed": 0, "sites": {}}
    parents = _parent_map(tree)
    roots = _source_roots(tree)
    fixtures = _fixture_roots(tree)
    roots -= fixtures
    aliases = _source_aliases(tree)
    sites = [n for n in ast.walk(tree)
             if _is_source_call(n, roots, fixtures, aliases)]
    claims = _allocate_uses(sites, parents, tree)
    verdicts = {}
    for node in sites:
        # A site whose bound name is never read is still a source read, and
        # still counts one: fail-closed, and it keeps the unit invariant
        # when a use moves between two sites rather than disappearing.
        uses = claims[id(node)] or [node]
        textual = not uses or not all(_reaches_parse(u, parents) for u in uses)
        verdicts[(node.lineno, node.col_offset)] = (
            "textual" if textual else "parsed", len(uses))
    return {
        "textual": sum(n for v, n in verdicts.values() if v == "textual"),
        "parsed": sum(n for v, n in verdicts.values() if v == "parsed"),
        "sites": verdicts,
    }


def measure_suite(tests_dir: Path = TESTS_DIR) -> dict:
    """Per-file textual counts across the suite (files with zero omitted).

    ⚠ EVERY `.py` UNDER `tests/`, NOT JUST `test_*.py` (§4GJ round 4). The
    glob was `test_*.py`, so a text pin added to `conftest.py` and a
    getsource helper added to `helpers.py` were both outside the ratchet —
    both confirmed GREEN against the real baseline. A rule that names the
    files it applies to is a rule with a documented bypass; the unit is the
    suite. Measured at the widening: those files hold 0 source reads today,
    so the baseline did not move because of this — the route is simply
    closed. Keys are paths relative to `tests/`, which is the file name for
    everything at the top level, so the baseline keys did not change.
    """
    out = {}
    for path in sorted(tests_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        res = classify_file(path)
        if res["textual"]:
            out[path.relative_to(tests_dir).as_posix()] = res["textual"]
    return out


def growth_report(current: dict, per_file: dict) -> list:
    """Files whose textual-pin count EXCEEDS their baseline, as
    `(name, now, was)`. A file absent from the baseline has a baseline of
    zero, so a brand-new test file with a text pin is growth.

    Extracted so the gate and its own tests run the SAME code — pinning a
    re-implementation of this rule is the R4 "tests that rebuild the code
    under test" reject, and a mutant that emptied the gate's comprehension
    survived while the local copy in the test still passed."""
    return sorted(
        (name, count, per_file.get(name, 0))
        for name, count in current.items()
        if count > per_file.get(name, 0)
    )


def staleness_slack(current: dict, baseline: dict) -> int:
    """How many pins the baseline still CLAIMS that the suite no longer has.

    Positive slack is permission to re-add text pins for free, so the gate
    caps it. Extracted for the same reason as `growth_report`: with a
    freshly regenerated baseline the slack is 0, so a test that only ran the
    live numbers agreed in both worlds and could not fail (R4's
    "fixtures where the fixed and broken worlds agree")."""
    return int(baseline["total_textual"]) - sum(current.values())


def slack_verdict(current: dict, baseline: dict) -> dict:
    """The slack RULE: how much tolerance the baseline claims and no longer
    needs, against the ceiling — which is `ALLOWED_SLACK`, in this module.

    Extracted for the same reason as `growth_report` and `staleness_slack`
    (the gate and its tests must run one implementation), and pinned
    separately because of what it replaced: the gate compared the slack
    against `baseline["allowed_slack"]`, i.e. asked the file under the gate
    how much it was allowed. `scripts/lint.py` shipped the identical shape in
    round 4 and it was measured there — one `"allowed_slack": 9999` disarms
    the script and the pytest half at once, and the writer copied the key
    forward so it stuck. A baseline that declares a non-zero ceiling is now
    a failure in itself: there is no other reason to write the key.
    """
    slack = staleness_slack(current, baseline)
    # ANY declaration counts, however it is spelled: `int(x or 0)` turns a
    # string ceiling into a ValueError, which reads as a broken gate rather
    # than a disarmed one.
    declared = baseline.get("allowed_slack")
    granted = declared is not None and declared != 0
    return {"slack": slack, "ceiling": ALLOWED_SLACK, "declared": declared,
            "grants_slack": granted,
            "ok": slack <= ALLOWED_SLACK and not granted}


#: The four fields a `_migration` record carried before §4GJ round 5.
_LEGACY_MIGRATION_FIELDS = frozenset(
    {"reason", "previous_total", "previous_files", "files_that_grew"})
#: ...and the six round 5 added, because the four above cannot DECIDE the
#: claim the prose makes ("no file lost, no file decreased"): 250 files
#: before and 250 after is equally consistent with 250 unchanged and with one
#: lost plus one added.
_MIGRATION_COUNT_FIELDS = frozenset(
    {"new_total", "new_files", "files_that_shrank", "files_lost",
     "files_added", "bootstrap"})
#: The ONE record that predates the format, identified by what it SAYS and
#: not by the shape it happens to have — see `migration_gaps`. These are the
#: numbers the §4GJ round-4 unit change wrote; a regenerate carries them
#: forward verbatim, so they do not move unless someone re-migrates.
_GRANDFATHERED_MIGRATION = {"previous_total": 633, "previous_files": 250,
                            "files_that_grew": 178}

#: ...AND THE BASELINE THAT RECORD ACTUALLY DESCRIBES (§4GK round 7).
#:
#: ⚠ THE CONTENT KEY WAS FORGEABLE FROM DATA PUBLISHED IN THE REPO. All three
#: numbers above are IN the committed baseline (and two of them are spelled
#: out again in its own `reason` prose), `reason` is unconstrained, and
#: nothing tied the record to the file it sits in — so the grandfather clause
#: could be satisfied by anyone who had read the baseline. The laundering
#: loop, measured end to end: grow a file from 2 to 5 text pins (the gate
#: goes correctly red), hand-rewrite `per_file`/`total_textual` to the new
#: numbers, keep the `_migration` block VERBATIM -> 36 passed, the growth
#: invisible, and `_write_baseline`'s `elif old.get("_migration")` copies the
#: same block forward for ever, so the exemption never expires.
#:
#: The tie is a digest of what the record's own claim is ABOUT: the
#: `total_textual` and `per_file` that migration produced. It lives HERE, in
#: code, for the same reason `ALLOWED_SLACK` does — a hand-edit of the
#: baseline cannot move a number in this file. Any edit to the baseline, of
#: any kind, breaks the tie, and the exemption is GONE: the honest answer
#: then is a real `--migrate "reason"` write, which stamps all ten counts.
_GRANDFATHERED_BASELINE = \
    "d3a358c25474f1109b55cf3532dff93eb748eb2572dd80d4d72ab4b781eacfe8"

#: The digest of the baseline THIS TREE carries, in CODE (§4GO, 2026-09-14).
#:
#: ⚠ WITHOUT THIS, THE MIGRATION WOULD HAVE WEAKENED THE RATCHET. The
#: laundering loop — hand-edit `per_file`/`total_textual` upward, keep the
#: `_migration` block, watch the gate go green — was caught only by
#: `_GRANDFATHERED_BASELINE`'s tie to the legacy record. A COMPLETE record
#: carries its counts, so `migration_gaps` passes it on field presence alone,
#: and a complete record is deliberately carried forward across ordinary
#: downward writes (its `new_total` goes stale by design). So completeness
#: cannot be the tie, and the moment the committed record stopped being the
#: legacy one, nothing was tying provenance to the numbers any more. That was
#: never visible before because the committed record HAPPENED to be the
#: grandfathered one; it is a property of the design, not of this migration.
#:
#: The tie generalises: the digest lives here, in code, where a hand-edit of
#: the JSON cannot move it — the same reason `ALLOWED_SLACK` and the
#: grandfather digest live here. An honest re-write updates this constant in
#: the same change; `--write` prints the new digest for exactly that.
_COMMITTED_BASELINE_DIGEST = \
    "f37a2ac91e08165ebb17e8e0af90373fc0afa17842ae9143935dc81e88aa4779"


def baseline_digest(baseline: dict) -> str:
    """The identity of the numbers a `_migration` record claims to describe.

    Only the RESULT of a migration — the totals the gate reads — so an
    unrelated key (a `_comment` rewording) does not expire the exemption,
    and no change to a single per-file count can survive it."""
    payload = {"total_textual": baseline.get("total_textual"),
               "per_file": baseline.get("per_file") or {}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def migration_gaps(record: dict, baseline: dict = None) -> set:
    """Fields a `_migration` record needs to support its own claim.

    ⚠ THE ONE RECORD THE REPO CARRIES PREDATES THE FORMAT (§4GK round 6).
    Round 5 added the six counts to the WRITER; the committed baseline was
    not re-migrated (there is no git here, and re-migrating a ratchet's
    memory to make a test pass is the laundering the ratchet exists to
    prevent), so it still carries only the old four — and
    `_write_baseline`'s `elif old.get("_migration")` copies it forward
    verbatim on every regenerate. `test_the_baseline_can_say_where_it_came_
    from` asserted only that `reason` was truthy, so NOTHING required a
    record to carry the counts and the next upward write could have been as
    undecidable as this one.

    So: THE one record is grandfathered by its CONTENT — its four fields and
    the three numbers it carries — and everything else must carry all ten.
    Grandfathering the four-field SHAPE instead was tried and rejected in the
    same change: a writer regressed to the round-4 record would then be
    grandfathered too, and the rule would license the very format it is
    supposed to be finishing the adoption of (measured — the shape rule let a
    freshly written four-field record through). A partially-updated record is
    not grandfathered either: that is how a format gets half-adopted and
    stays that way.

    ⚠ AND CONTENT PUBLISHED IN THE REPO IS NOT A CREDENTIAL (§4GK round 7).
    The three numbers are printed in the baseline itself, `reason` is
    unconstrained, and nothing tied the record to the FILE IT DESCRIBES — so
    the clause was satisfiable by anyone who could read the repo, and the
    laundering loop in `_GRANDFATHERED_BASELINE` above is what that buys.
    `baseline` is the file carrying the record, and the exemption holds only
    while that file is still the one the record is about. Called WITHOUT a
    baseline — a record inspected on its own — nothing is grandfathered,
    because there is then nothing to tie it to.
    """
    if (set(record) == _LEGACY_MIGRATION_FIELDS
            and all(record.get(k) == v
                    for k, v in _GRANDFATHERED_MIGRATION.items())
            and baseline is not None
            and baseline_digest(baseline) == _GRANDFATHERED_BASELINE):
        return set()
    return (_LEGACY_MIGRATION_FIELDS | _MIGRATION_COUNT_FIELDS) - set(record)


def _load_baseline(path: Path = None) -> dict:
    path = BASELINE_PATH if path is None else path
    if not path.exists():
        raise AssertionError(
            f"the pin-quality baseline {path.name} is MISSING. It is "
            "the ratchet's only memory of how many source-text pins the suite "
            "had; without it the gate cannot fail and every new text pin is "
            "invisible. Restore it from git, or regenerate it deliberately "
            "with `python tests/test_pin_quality_ratchet.py --write`.")
    return json.loads(path.read_text(encoding="utf-8"))


# ── the ratchet ──────────────────────────────────────────────────────────────

def test_source_text_pins_never_grow():
    """The gate. Total textual pins <= baseline, and no file exceeds its own
    baseline — so a rewrite cannot hide growth behind another file's shrink.

    Fails in: any tree where a new (or edited) test asserts on source TEXT."""
    baseline = _load_baseline()
    current = measure_suite()

    grew = growth_report(current, baseline["per_file"])
    assert not grew, (
        "source-text pins GREW — R4 hard-rejects them (a mutant that deletes "
        "the code and leaves the string in a comment survives a text pin):\n  "
        + "\n  ".join(f"{n}: {now} now vs {was} at baseline (+{now - was})"
                      for n, now, was in grew)
        + "\n\nEither make the pin behavioural (drive the code, assert on the "
          "result), or make it a real R1 enumeration (`ast.parse` the source "
          "and walk the tree — that is classified PARSED and is allowed). "
          "If you genuinely removed pins elsewhere and want to re-baseline "
          "DOWNWARD, run `python tests/test_pin_quality_ratchet.py --write`.")

    # NO separate total assertion. Every file in `current` is at or below its
    # own baseline (a file absent from the baseline has a baseline of zero),
    # so the total is bounded by construction — the check was unfalsifiable,
    # its mutant survived the battery, and R2 says an equivalent mutant means
    # dead code. The total still lives in the baseline file, where
    # `test_the_baseline_is_not_stale_upward` uses it for the slack rule.


def test_the_baseline_is_not_stale_upward():
    """A baseline that drifts above reality lets pins creep back for free.
    When the suite improves, the baseline must be regenerated downward.

    Fails in: a tree where pins were removed and the baseline kept its old
    (higher) numbers — the slack would silently permit re-adding them."""
    baseline = _load_baseline()
    current = measure_suite()
    # the GATE's rule, not a re-implementation of it (`slack_verdict`)
    verdict = slack_verdict(current, baseline)
    assert not verdict["grants_slack"], (
        f"{BASELINE_PATH.name} declares its own slack ceiling "
        f"({verdict['declared']}). The ceiling is ALLOWED_SLACK in this "
        "module; the file under the gate does not set its own.")
    assert verdict["ok"], (
        f"the suite now has {verdict['slack']} FEWER source-text pins than "
        f"the baseline records (allowed drift: {verdict['ceiling']}). Lock "
        "the win in: `python tests/test_pin_quality_ratchet.py --write`.")


def test_the_baseline_can_say_where_it_came_from():
    """Provenance, the tripwire the writer's bootstrap path lacked (§4GJ
    round 5).

    `scripts/lint.py`'s bootstrap reset is caught because a bootstrap write
    emits `"notes": {}` and a pin demands notes be truthy. This ratchet had
    no equivalent: delete `pin_quality_baseline.json`, write any numbers you
    like (the upward refusal was skipped when there was no old file to
    compare against), and the gate then read the inflated file and went
    green — with no git, laundering was the only available "recovery".

    `_write_baseline` now refuses a write with no predecessor unless a
    reason is recorded, so every honest baseline either descends from this
    one (which carries its `_migration` forward) or was bootstrapped with a
    stamped reason.

    Fails in: a baseline whose predecessor was deleted rather than
    ratcheted — it arrives with no provenance at all."""
    baseline = _load_baseline()
    record = baseline.get("_migration") or {}
    assert record.get("reason"), (
        f"{BASELINE_PATH.name} carries no provenance. Every upward or "
        "from-scratch write records its reason; a baseline with none was "
        "not written by `python tests/test_pin_quality_ratchet.py --write`.")
    # ⚠ A REASON IS NOT EVIDENCE (§4GK round 6). The committed record's prose
    # asserts "no file lost, no file decreased", which its four fields cannot
    # decide — and this assertion, reading `reason` alone, required no more
    # of the NEXT one either. `migration_gaps` grandfathers exactly the old
    # four-field shape and demands the counts of everything else, so the
    # record this file carries is honestly marked as predating the format
    # instead of being re-migrated to look compliant.
    #
    # ⚠ AND THE RECORD IS CHECKED AGAINST THE FILE IT SITS IN (§4GK round 7).
    # Grandfathering on the record ALONE was forgeable from the repo's own
    # contents — see `_GRANDFATHERED_BASELINE`. Passing `baseline` is what
    # ties the one exempt record to the one baseline it describes, and what
    # makes the exemption expire the moment those numbers move.
    assert not migration_gaps(record, baseline), (
        f"{BASELINE_PATH.name}'s _migration record is missing "
        f"{sorted(migration_gaps(record, baseline))} — without those counts "
        f"its own claim about what moved is undecidable, and the "
        f"grandfathered record is grandfathered only for the baseline it "
        f"describes (this file's numbers have moved since). Re-run the write "
        f"with `--migrate \"reason\"`, which stamps them.")
    assert baseline["total_textual"] == sum(baseline["per_file"].values()), \
        "the recorded total does not match the per-file numbers under it"


def test_a_migration_record_written_from_now_on_must_carry_the_counts():
    """§4GK round 6. Round 5 added `files_that_shrank`/`files_lost`/
    `files_added`/`new_total`/`new_files`/`bootstrap` to the writer because
    the old four could not decide "purely additive" — and then nothing
    downstream required them. `test_the_baseline_can_say_where_it_came_from`
    asserted `reason` was truthy, so a record with a sentence and no numbers
    was as acceptable as a complete one; the committed baseline is exactly
    that record, carried forward verbatim by `elif old.get("_migration")`.

    The rule: THE committed record is grandfathered by its CONTENT AND by
    the baseline it describes (§4GK round 7) — it predates the format and
    must not be re-migrated to look compliant — and everything else carries
    all ten.

    ⚠ THE FIELD NAMES ARE WRITTEN OUT HERE ON PURPOSE (§4GK round 7). Every
    assertion in this test used to compare against `_MIGRATION_COUNT_FIELDS`
    itself, so emptying that constant — deleting the entire round-5/6 rule in
    one line — left the whole test GREEN: `migration_gaps(...) == set()` on
    both sides. A pin stated in the subject's own vocabulary cannot fail when
    the subject is deleted.

    Fails in: a check that only asks for `reason` (every case below is
    accepted, including the empty record), and in any tree where
    `_MIGRATION_COUNT_FIELDS` has been emptied.
    """
    _COUNTS = {"new_total", "new_files", "files_that_shrank", "files_lost",
               "files_added", "bootstrap"}
    _LEGACY = {"reason", "previous_total", "previous_files", "files_that_grew"}
    # a reason and nothing else — the shape the old assertion accepted
    assert migration_gaps({"reason": "because"}) >= _COUNTS
    assert migration_gaps({}) == _COUNTS | _LEGACY, "an empty record decides nothing"

    # ⚠ AND THE SHAPE ALONE IS NOT THE GRANDFATHER: a record with the old
    # four fields and DIFFERENT numbers is a new record in an old format.
    # (Measured while writing this: with a shape-keyed rule, a writer
    # regressed to the round-4 record passed this test.)
    assert migration_gaps(dict(committed_shape_other_numbers := {
        "reason": "some later write", "previous_total": 1, "previous_files": 1,
        "files_that_grew": 0})) == _COUNTS, committed_shape_other_numbers

    # ⚠ THE COMMITTED RECORD IS NO LONGER THE GRANDFATHERED ONE (§4GO,
    # 2026-09-14): the §4GJ-round-5 measurement hole was closed and the
    # baseline was migrated 1312 → 1357, which is precisely the act the
    # grandfather clause was written to expire. It must now carry all ten
    # counts like any other record.
    baseline = _load_baseline()
    committed = baseline["_migration"]
    assert not migration_gaps(committed, baseline), sorted(
        migration_gaps(committed, baseline))
    assert set(committed) == _COUNTS | _LEGACY, sorted(committed)
    # ⚠ NOT `new_total == total_textual` (§4GT). A complete record is
    # provenance for the LINEAGE and is carried forward verbatim across
    # ordinary DOWNWARD writes — INVERSE 2 below pins exactly that — so its
    # counts describe the write that stamped it, not necessarily the file it
    # now sits in. The identity between record and file is
    # `_COMMITTED_BASELINE_DIGEST`'s job, in code where a hand-edit cannot
    # reach it. What must hold here is that the counts are present and
    # internally coherent.
    assert isinstance(committed["new_total"], int)
    assert committed["new_total"] >= baseline["total_textual"], (
        "the committed total is ABOVE the record's — that is an upward move, "
        "which needs its own --migrate")
    assert committed["new_files"] >= 1

    # The grandfather MECHANISM stays pinned — it still decides any tree
    # that carries the legacy record — but synthetically, since the file it
    # used as a fixture has moved on.
    legacy = dict(_GRANDFATHERED_MIGRATION, reason="§4GJ round 5, no counts")
    assert set(legacy) == _LEGACY, sorted(legacy)
    assert migration_gaps(legacy) == _COUNTS, "grandfathered with no tie"
    assert migration_gaps(legacy, baseline) == _COUNTS, (
        "the legacy record vouches for a baseline it does not describe")

    # ⚠ PARTIAL ADOPTION IS NOT GRANDFATHERED: the old four plus ONE new
    # field is a record written under the new format, and it must be
    # complete. (A subset rule would have let the format stay half-adopted
    # for ever, which is how the round-5 fields came to be unrequired.)
    half = {k: v for k, v in committed.items() if k != "files_lost"}
    assert migration_gaps(half, baseline) == {"files_lost"}

    # and a record the WRITER produces satisfies it, by construction
    path = Path(tempfile.mkdtemp()) / "b.json"
    payload = _write_baseline({"a.py": 1}, path, migrate="first write here")
    assert not migration_gaps(payload["_migration"], payload), payload["_migration"]
    written = json.loads(path.read_text())
    assert not migration_gaps(written["_migration"], written)
    assert set(written["_migration"]) == _COUNTS | _LEGACY, written["_migration"]


def test_the_committed_baseline_is_the_one_this_module_vouches_for():
    """The successor to the grandfathered note, which said of itself: *"Fails
    in: a tree where someone re-migrated the baseline to add the counts — at
    which point delete this test, because the claim is then decidable."* That
    happened (§4GO): the non-Python measurement hole was closed and the
    baseline migrated 1312 → 1357, so the legacy record is gone and its claim
    is decidable by `migration_gaps` like every other.

    What must NOT go with it is the tie between the provenance record and the
    numbers it describes. A complete record passes on field presence alone —
    and is carried forward verbatim across ordinary downward writes, so its
    `new_total` goes stale by design — which means completeness cannot be the
    tie. The digest therefore lives in CODE, where a hand-edit of the JSON
    cannot move it: grow a file's pins, hand-write the new numbers into the
    baseline, keep the record, and this fails.

    World where it fails: any tree whose baseline was edited without the
    writer, and any honest re-write that did not update the constant.
    """
    baseline = _load_baseline()
    assert baseline_digest(baseline) == _COMMITTED_BASELINE_DIGEST, (
        "the committed baseline's numbers are not the ones this module "
        "vouches for. If you re-wrote it honestly, paste the digest "
        "`--write` printed into _COMMITTED_BASELINE_DIGEST in the same "
        "change; if you did not, the file was hand-edited.")
    # …and the tie is LIVE, not a tautology: any edit to the numbers breaks it
    laundered = dict(baseline, total_textual=baseline["total_textual"] + 3)
    assert baseline_digest(laundered) != _COMMITTED_BASELINE_DIGEST
    per = dict(baseline["per_file"])
    per[sorted(per)[0]] += 1
    assert baseline_digest(dict(baseline, per_file=per)) != _COMMITTED_BASELINE_DIGEST


# ── the classifier itself ────────────────────────────────────────────────────

_PARSED_SAMPLES = [
    "import ast, inspect\ndef t():\n    ast.parse(inspect.getsource(M))\n",
    "import ast, inspect\ndef t():\n    tree = ast.parse(inspect.getsource(M).lstrip())\n",
    "import ast, inspect\ndef t():\n    src = inspect.getsource(M)\n    ast.parse(src)\n",
    "import ast, inspect\ndef t():\n    src = inspect.getsource(M)\n"
    "    ast.parse(src.replace('a', 'b'))\n",
    "import ast\nfrom pathlib import Path\ndef t():\n"
    "    ast.parse(Path(M.__file__).read_text())\n",
    "import ast, inspect, textwrap\ndef t():\n"
    "    ast.parse(textwrap.dedent(inspect.getsource(M.f)))\n",
]

_TEXTUAL_SAMPLES = [
    "import inspect\ndef t():\n    assert 'x' in inspect.getsource(M)\n",
    "import inspect\ndef t():\n    src = inspect.getsource(M)\n    assert 'x' in src\n",
    "import inspect\ndef t():\n    src = inspect.getsource(M)\n    i = src.index('x')\n",
    "import inspect\ndef t():\n    src = inspect.getsource(M)\n    w = src[10:90]\n",
    "import inspect, re\ndef t():\n    src = inspect.getsource(M)\n    re.search('x', src)\n",
    "from pathlib import Path\ndef t():\n"
    "    src = Path(M.__file__).read_text()\n    assert 'x' in src\n",
    # handed to a helper — unfollowable, so textual
    "import inspect\ndef t():\n    helper(inspect.getsource(M))\n",
    # a parse that is not ast.parse
    "import inspect\ndef t():\n    src = inspect.getsource(M)\n    src.parse()\n",
]


@pytest.mark.parametrize("code", _PARSED_SAMPLES)
def test_enumerations_are_classified_parsed(tmp_path, code):
    """R1 enumerations must stay legal — if these read as textual the gate
    would forbid the protocol's own mechanism.

    Fails in: a classifier that ignores the parse sink or the pass-throughs."""
    f = tmp_path / "test_sample.py"
    f.write_text(code, encoding="utf-8")
    res = classify_file(f)
    assert (res["parsed"], res["textual"]) == (1, 0), (code, res)


@pytest.mark.parametrize("code", _TEXTUAL_SAMPLES)
def test_text_pins_are_classified_textual(tmp_path, code):
    """Fails in: a classifier that calls an unprovable use 'parsed' — the
    fail-open direction, which would let text pins in for free."""
    f = tmp_path / "test_sample.py"
    f.write_text(code, encoding="utf-8")
    res = classify_file(f)
    assert (res["parsed"], res["textual"]) == (0, 1), (code, res)


def test_a_string_literal_mentioning_dunder_file_is_not_a_source_read(tmp_path):
    """`test_4ec_evaluator_pins` carries `__file__` inside string FIXTURES for
    a taint analyser. A grep-based detector counts those; an AST one must not.

    Fails in: any implementation that greps the file instead of walking it."""
    f = tmp_path / "test_sample.py"
    f.write_text("CASES = ['def t():\\n    return open(__file__).read()\\n']\n"
                 "def t():\n    assert CASES\n", encoding="utf-8")
    assert classify_file(f) == {"textual": 0, "parsed": 0, "sites": {}}


def test_a_brand_new_file_with_a_text_pin_would_fail_the_gate(tmp_path):
    """The gate's real job, driven end to end against a fake suite: a file
    that is not in the baseline at all may not introduce a text pin.

    Fails in: a ratchet that only checks the total, or only checks files it
    already knows."""
    (tmp_path / "test_new.py").write_text(
        "import inspect\ndef test_x():\n"
        "    assert 'needle' in inspect.getsource(M)\n", encoding="utf-8")
    current = measure_suite(tmp_path)
    assert current == {"test_new.py": 1}
    # the GATE's own rule, not a re-implementation of it
    assert growth_report(current, {}) == [("test_new.py", 1, 0)]
    # ...and an existing file that gains one is growth too
    assert growth_report({"test_old.py": 4}, {"test_old.py": 3}) == \
        [("test_old.py", 4, 3)]
    # while shrinking, or holding, is not
    assert growth_report({"test_old.py": 2}, {"test_old.py": 3}) == []
    assert growth_report({"test_old.py": 3}, {"test_old.py": 3}) == []


def test_a_missing_baseline_is_loud(tmp_path):
    """The baseline file IS the ratchet's memory: if it can go missing
    quietly the gate passes forever and every new text pin is invisible.

    Fails in: a tree where `_load_baseline` invents a permissive default
    instead of refusing."""
    with pytest.raises(AssertionError, match="MISSING"):
        _load_baseline(tmp_path / "not_here.json")


def test_the_slack_rule_fires_on_a_stale_baseline():
    """Driven with a baseline that over-claims, because the live one is
    freshly regenerated and reads 0 in both worlds.

    Fails in: a tree where the slack rule stops subtracting, or the gate
    stops comparing it to `allowed_slack`."""
    stale = {"total_textual": 40}
    assert staleness_slack({"a.py": 10, "b.py": 5}, stale) == 25
    assert staleness_slack({"a.py": 40}, stale) == 0
    # and the comparison the gate makes
    assert slack_verdict({"a.py": 10}, stale)["ok"] is False
    assert slack_verdict({"a.py": 40}, stale)["ok"] is True


def test_a_baseline_cannot_grant_itself_slack():
    """§4GJ round 5: the gate compared the slack against
    `baseline["allowed_slack"]` — it asked the file under the gate how much
    tolerance it was allowed. `scripts/lint.py` shipped the identical shape
    in round 4 and the reviewer measured it there: baseline a finding, fix
    it, write `"allowed_slack": 9999`, re-introduce the identical finding ->
    `OK`, rc=0, with the pytest half bounded by the same self-declared
    number so neither consumer could fail.

    Fails in: `slack <= baseline["allowed_slack"]` — under which the first
    case below is `ok`.
    """
    stale = {"total_textual": 40, "allowed_slack": 9999}
    bought = slack_verdict({"a.py": 10}, stale)
    assert bought["ok"] is False, "the baseline bought its own slack"
    assert bought["slack"] == 30 and bought["ceiling"] == ALLOWED_SLACK == 0
    assert bought["declared"] == 9999 and bought["grants_slack"] is True
    # however it is spelled
    assert slack_verdict({"a.py": 40},
                         {"total_textual": 40, "allowed_slack": "9999"})["ok"] \
        is False

    # the key is fatal even with NO slack to excuse: the only reason to
    # write it is to disarm the arm.
    assert slack_verdict({"a.py": 40}, stale)["ok"] is False
    assert slack_verdict({"a.py": 40}, {"total_textual": 40})["ok"] is True


def test_this_file_holds_itself_to_its_own_rule():
    """The ratchet reads test sources and parses them — by its own
    classification that is PARSED, not a text pin. If this file ever needed
    a textual read it would have to appear in its own baseline.

    Fails in: a tree where the ratchet started grepping source itself."""
    res = classify_file(Path(__file__))
    assert res["textual"] == 0, res["sites"]


# ── §4GJ round 4: four routes that walked past the ratchet ──────────────

def test_two_source_reads_on_one_line_are_two_pins(tmp_path):
    """The verdict map was keyed by LINE, so two source-obtaining calls that
    start on the same line collapsed into one. Confirmed end to end against
    the real baseline: a new text pin on its own line goes RED, and the same
    pin appended to an existing pin's line stays GREEN — a free text pin for
    anyone who writes it on a line that already has one. The suite really
    contains the shape (`test_sandbox_chromium_gate.py:108` is
    `getsource(A) + getsource(B)`), and three pins were hiding behind it.

    Fails in: `verdicts[node.lineno] = ...`.
    """
    f = tmp_path / "test_sample.py"
    f.write_text("import inspect\n"
                 "def t():\n"
                 "    assert 'a' in inspect.getsource(M) and 'b' in inspect.getsource(N)\n",
                 encoding="utf-8")
    res = classify_file(f)
    assert (res["textual"], res["parsed"]) == (2, 0), res

    # and the mixed case on one line: one parsed, one textual, both seen
    g = tmp_path / "test_mixed.py"
    g.write_text("import ast, inspect\n"
                 "def t():\n"
                 "    ast.parse(inspect.getsource(M)); assert 'b' in inspect.getsource(N)\n",
                 encoding="utf-8")
    assert (classify_file(g)["textual"], classify_file(g)["parsed"]) == (1, 1)


def test_an_aliased_getsource_is_not_invisible(tmp_path):
    """`from inspect import getsource as gs` then `assert "x" in gs(M)` was
    classified as nothing at all — confirmed GREEN against the real
    baseline. That is the fail-OPEN direction in the file whose docstring
    promises "every use this analysis cannot prove is a parse" counts as
    textual.

    Fails in: a matcher that only compares the called name against
    `getsource`/`getsourcelines`.
    """
    f = tmp_path / "test_sample.py"
    f.write_text("from inspect import getsource as gs\n"
                 "def t():\n    assert 'x' in gs(M)\n", encoding="utf-8")
    assert (classify_file(f)["textual"], classify_file(f)["parsed"]) == (1, 0)

    # the alias used as an R1 enumeration is still legal, so the rule is
    # about VISIBILITY, not about the spelling
    g = tmp_path / "test_alias_parse.py"
    g.write_text("import ast\nfrom inspect import getsource as gs\n"
                 "def t():\n    ast.parse(gs(M))\n", encoding="utf-8")
    assert (classify_file(g)["textual"], classify_file(g)["parsed"]) == (0, 1)

    # an unrelated local named `gs` is not a source read
    h = tmp_path / "test_not_alias.py"
    h.write_text("def t():\n    assert 'x' in gs(M)\n", encoding="utf-8")
    assert classify_file(h)["textual"] == 0


def test_a_text_pin_outside_a_test_file_is_still_measured(tmp_path):
    """`measure_suite` globbed `test_*.py`, so `conftest.py` and
    `helpers.py` were outside the ratchet — both confirmed GREEN against the
    real baseline. Test code is test code wherever it is imported from, and
    a rule that names the files it applies to documents its own bypass.

    Fails in: `tests_dir.glob("test_*.py")`.
    """
    (tmp_path / "conftest.py").write_text(
        "import inspect\ndef pytest_configure():\n"
        "    assert 'needle' in inspect.getsource(M)\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text(
        "import inspect\ndef assert_contains(m, s):\n"
        "    assert s in inspect.getsource(m)\n", encoding="utf-8")
    (tmp_path / "test_real.py").write_text(
        "import inspect\ndef test_x():\n"
        "    assert 'x' in inspect.getsource(M)\n", encoding="utf-8")

    current = measure_suite(tmp_path)

    assert current == {"conftest.py": 1, "helpers.py": 1, "test_real.py": 1}
    # ...and all three are growth against a baseline that has never seen them
    assert [n for n, _, _ in growth_report(current, {})] == \
        ["conftest.py", "helpers.py", "test_real.py"]


def test_hoisting_a_source_read_banks_no_improvement(tmp_path):
    """The unit. Twenty inline `getsource` reads and one hoisted read used
    by twenty assertions are the same twenty source-text assertions, but the
    site-counting version scored them 20 and 1 — so a pure refactor banked a
    nineteen-pin "improvement" with no behavioural change, and could then
    re-admit nineteen assertions for free.

    Fails in: a classifier that counts sites (the hoisted file reads 1).
    """
    inline = "import inspect\ndef t():\n" + "".join(
        f"    assert 'n{i}' in inspect.getsource(M)\n" for i in range(20))
    hoisted = ("import inspect\n_SRC = inspect.getsource(M)\ndef t():\n"
               + "".join(f"    assert 'n{i}' in _SRC\n" for i in range(20)))
    (tmp_path / "test_inline.py").write_text(inline, encoding="utf-8")
    (tmp_path / "test_hoisted.py").write_text(hoisted, encoding="utf-8")

    current = measure_suite(tmp_path)

    assert current["test_inline.py"] == 20
    assert current["test_hoisted.py"] == 20, (
        "the hoisted read counts once — a refactor that changes no "
        "behaviour just bought 19 free text pins")

    # ...and re-admitting assertions to the hoisted file is still growth
    more = ("import inspect\n_SRC = inspect.getsource(M)\ndef t():\n"
            + "".join(f"    assert 'n{i}' in _SRC\n" for i in range(21)))
    (tmp_path / "test_hoisted.py").write_text(more, encoding="utf-8")
    assert growth_report(measure_suite(tmp_path), current) == \
        [("test_hoisted.py", 21, 20)]


def test_a_site_with_one_textual_use_is_textual_for_all_of_them(tmp_path):
    """Fail-closed, under the use-counting unit: a parse standing next to a
    text assertion does not launder the rest of the uses.

    Fails in: a classifier that counts the parsed uses as parsed and only
    the membership test as textual — which would let a text pin be diluted
    by adding `ast.parse(src)` beside it."""
    f = tmp_path / "test_sample.py"
    f.write_text("import ast, inspect\ndef t():\n"
                 "    src = inspect.getsource(M)\n"
                 "    ast.parse(src)\n"
                 "    assert 'x' in src\n", encoding="utf-8")
    assert (classify_file(f)["textual"], classify_file(f)["parsed"]) == (2, 0)


def test_the_writer_refuses_to_ratchet_upward(tmp_path):
    """The writer wrote `measure_suite()` unconditionally while the file it
    emits says "regenerate DOWNWARD only" and the lint gate's equivalent
    refuses in so many words. Confirmed on a copy of the real tree: add a
    text pin, run the writer, 633 -> 635, rc=0. A ratchet whose writer will
    write anything has a reset button that is shorter to press than the fix
    is to make.

    Fails in: `_write_baseline` with no growth check.
    """
    path = tmp_path / "pin_quality_baseline.json"
    # the FIRST write has no predecessor, which is its own refusal now
    # (§4GJ round 5) — it costs a recorded reason, and then behaves.
    _write_baseline({"test_a.py": 3, "test_b.py": 2}, path,
                    migrate="first write in this scratch fixture")
    assert json.loads(path.read_text())["total_textual"] == 5

    with pytest.raises(SystemExit, match="REFUSING"):
        _write_baseline({"test_a.py": 4, "test_b.py": 2}, path)
    with pytest.raises(SystemExit, match="REFUSING"):
        _write_baseline({"test_a.py": 3, "test_b.py": 2, "test_new.py": 1}, path)
    assert json.loads(path.read_text())["total_textual"] == 5, \
        "the refused write still landed"

    # DOWNWARD is the point of the thing, so it must still work
    _write_baseline({"test_a.py": 1}, path)
    assert json.loads(path.read_text())["per_file"] == {"test_a.py": 1}

    # ...and an upward write is possible ONLY with a recorded reason
    payload = _write_baseline({"test_a.py": 9}, path,
                              migrate="the unit changed, not the suite")
    assert payload["_migration"]["reason"] == "the unit changed, not the suite"
    assert payload["_migration"]["previous_total"] == 1
    assert json.loads(path.read_text())["_migration"]["files_that_grew"] == 1


# ── §4GJ round 5: four routes that walked past ROUND 4's fixes ──────────


def test_two_reads_rebinding_one_name_do_not_each_claim_both_uses(tmp_path):
    """Round 4 moved the unit from the call SITE to the textual USE, and the
    new unit double-counted: `uses` was every Load of the bound name in the
    enclosing scope, read once PER SITE, so two reads rebinding one name in
    one scope each claimed both loads.

    Measured on the round-4 classifier: the file below scored 4 for its 2
    assertions, and deleting ONE assertion scored 2 — a two-pin refund for
    removing one pin, bankable against the per-file ratchet. That is the
    refactor-banks-pins defect round 4 existed to close, reappearing inside
    the fix.

    Fails in: `uses = [n for n in ast.walk(scope) if ... n.id == bound]`
    (4 and 2 below, instead of 2 and 1).
    """
    two = tmp_path / "test_two.py"
    two.write_text("import inspect\n"
                   "def t():\n"
                   "    src = inspect.getsource(M)\n"
                   "    assert 'a' in src\n"
                   "    src = inspect.getsource(N)\n"
                   "    assert 'b' in src\n", encoding="utf-8")
    assert classify_file(two)["textual"] == 2, \
        "two assertions, two text pins — the loads were counted twice"

    # ...and the refund is gone: dropping one assertion drops the count by
    # ONE. (The now-unused read still counts one: a source read with no
    # followable use is fail-closed, and that keeps the unit invariant when
    # a use moves between two sites instead of disappearing.)
    one = tmp_path / "test_one.py"
    one.write_text("import inspect\n"
                   "def t():\n"
                   "    src = inspect.getsource(M)\n"
                   "    src = inspect.getsource(N)\n"
                   "    assert 'b' in src\n", encoding="utf-8")
    assert classify_file(one)["textual"] == 2

    # the same shape ACROSS scopes: a module-level bind and a function-local
    # one of the same name both claimed the module-level load
    cross = tmp_path / "test_cross.py"
    cross.write_text("import inspect\n"
                     "_S = inspect.getsource(M)\n"
                     "def t():\n"
                     "    assert 'a' in _S\n"
                     "def u():\n"
                     "    _S = inspect.getsource(N)\n"
                     "    assert 'b' in _S\n", encoding="utf-8")
    assert classify_file(cross)["textual"] == 2

    # INVERSE (the round-4 property this must not break): ONE read with
    # twenty assertions is still twenty, so hoisting still banks nothing.
    hoisted = tmp_path / "test_hoisted.py"
    hoisted.write_text("import inspect\n_SRC = inspect.getsource(M)\ndef t():\n"
                       + "".join(f"    assert 'n{i}' in _SRC\n" for i in range(20)),
                       encoding="utf-8")
    assert classify_file(hoisted)["textual"] == 20


def test_a_source_read_that_never_says_getsource_is_still_seen(tmp_path):
    """RECOGNITION is the fail-OPEN half (§4GJ round 5). The classifier's
    docstring promises that every use it cannot prove is a parse counts as
    textual — but a source read it does not RECOGNISE never reaches that
    rule at all. `''.join(linecache.getlines(mod.__file__))` reads a
    module's text without naming `getsource` or `.read_text()`, and was
    classified as nothing whatsoever; confirmed GREEN against the real
    baseline.

    Fails in: a matcher that knows only `_SOURCE_FUNCS` and `_READ_CALLS`.
    """
    f = tmp_path / "test_lc.py"
    f.write_text("import linecache\n"
                 "def t(m):\n"
                 "    src = ''.join(linecache.getlines(m.__file__))\n"
                 "    assert 'needle' in src\n", encoding="utf-8")
    assert (classify_file(f)["textual"], classify_file(f)["parsed"]) == (1, 0)

    # the same route used as an R1 enumeration is still legal: this is about
    # VISIBILITY, not about which function spelled the read
    g = tmp_path / "test_lc_parse.py"
    g.write_text("import ast, linecache\n"
                 "def t(m):\n"
                 "    ast.parse(''.join(linecache.getlines(m.__file__)))\n",
                 encoding="utf-8")
    assert (classify_file(g)["textual"], classify_file(g)["parsed"]) == (0, 1)

    # ...and a file the test CREATED is fixture I/O, not a pin on the code
    h = tmp_path / "test_lc_fixture.py"
    h.write_text("import linecache\n"
                 "def t(tmp_path):\n"
                 "    assert 'x' in ''.join("
                 "linecache.getlines(str(tmp_path / 'a.py')))\n",
                 encoding="utf-8")
    assert classify_file(h)["textual"] == 0


def test_the_writer_refuses_to_write_over_a_MISSING_baseline(tmp_path):
    """The upward refusal was skipped when the file was ABSENT — `grew =
    growth_report(...) if old else []` — so deleting the baseline disarmed
    the guard the baseline exists to hold.

    Reproduced: remove the file, write numbers above the committed total ->
    no refusal, no `_migration` stamp, nothing downstream asserting either,
    and the gate then read the inflated file and passed. `scripts/lint.py`
    has the same bootstrap reset but is CAUGHT, because a bootstrap write
    emits `"notes": {}` and a pin demands notes be truthy. With no git in
    this tree, delete-and-rewrite was also the only available "recovery",
    which is exactly why it has to cost a recorded reason.

    Fails in: `if old else []` — under which the first call below writes
    9999 pins and returns normally.
    """
    path = tmp_path / "pin_quality_baseline.json"
    assert not path.exists()

    with pytest.raises(SystemExit, match="MISSING"):
        _write_baseline({"test_a.py": 9999}, path)
    assert not path.exists(), "the refused write still landed"

    # ...and the escape hatch is the same one an upward write uses: a reason
    # that goes into the file, where a reader can challenge it.
    payload = _write_baseline({"test_a.py": 9999}, path,
                              migrate="genuine first write")
    assert payload["_migration"]["bootstrap"] is True
    assert payload["_migration"]["previous_total"] is None
    assert json.loads(path.read_text())["_migration"]["reason"] == \
        "genuine first write"

    # an EMPTY baseline is the same hole with the file still in place, so it
    # is refused too rather than treated as "nothing to ratchet against"
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSING"):
        _write_baseline({"test_a.py": 9999}, empty)


def test_the_migration_record_can_tell_additive_from_churn(tmp_path):
    """The `_migration` block could not support its own claim (§4GJ round
    5). It recorded `previous_total`, `previous_files` and `files_that_grew`
    — from which "purely additive" is not decidable: 250 files before and
    250 after is equally consistent with 250 unchanged and with one file
    lost plus one added, and a file whose count DROPPED inside an upward
    migration left no trace at all.

    The 633 -> 1312 unit change WAS additive (independently reconstructed: 0
    shrank, 0 lost, 0 added, 178 grew) — but the record it wrote could not
    say so, and the next reader had to take the prose on trust.

    Fails in: the round-4 record — every assertion below is a KeyError, and
    the two migrations are indistinguishable in it.
    """
    path = tmp_path / "b.json"
    _write_baseline({"a.py": 5, "b.py": 5}, path, migrate="bootstrap")

    additive = _write_baseline({"a.py": 9, "b.py": 5}, path,
                               migrate="the unit changed, not the suite")
    assert additive["_migration"]["files_lost"] == 0
    assert additive["_migration"]["files_added"] == 0
    assert additive["_migration"]["files_that_shrank"] == 0
    assert additive["_migration"]["files_that_grew"] == 1
    assert (additive["_migration"]["previous_total"],
            additive["_migration"]["new_total"]) == (10, 14)

    # CHURN with the same headline numbers: one file gone, one arrived, the
    # total unchanged. The round-4 record wrote the same three fields for
    # both; these four separate them.
    churn = _write_baseline({"a.py": 9, "c.py": 5}, path, migrate="churn")
    assert (churn["_migration"]["previous_total"],
            churn["_migration"]["new_total"]) == (14, 14)
    assert churn["_migration"]["previous_files"] == \
        churn["_migration"]["new_files"] == 2
    assert churn["_migration"]["files_lost"] == 1
    assert churn["_migration"]["files_added"] == 1

    # and a file that SHRANK inside an upward write is now on the record
    shrank = _write_baseline({"a.py": 1, "c.py": 9}, path, migrate="mixed")
    assert shrank["_migration"]["files_that_shrank"] == 1
    assert shrank["_migration"]["files_that_grew"] == 1


def test_migrate_without_a_reason_says_so_instead_of_crashing():
    """`--write --migrate` with `--migrate` last raised IndexError — a
    traceback instead of the one sentence the operator needed, on the
    command that overrides the ratchet. And `--migrate --something` took the
    next FLAG as the reason, stamping `"reason": "--something"` into the
    committed file: an audit trail that records nothing still reads as an
    audit trail.

    Fails in: `sys.argv[sys.argv.index("--migrate") + 1]` (IndexError, not
    SystemExit, on the first case; a silent `"--force"` on the second).
    """
    with pytest.raises(SystemExit, match="needs a REASON"):
        _reason_from_argv(["--write", "--migrate"])
    with pytest.raises(SystemExit, match="needs a REASON"):
        _reason_from_argv(["--write", "--migrate", "--force"])
    with pytest.raises(SystemExit, match="needs a REASON"):
        _reason_from_argv(["--write", "--migrate", "   "])

    assert _reason_from_argv(["--write", "--migrate", "the unit moved"]) == \
        "the unit moved"
    assert _reason_from_argv(["--write"]) == ""


def _write_baseline(current: dict = None, path: Path = None,
                    *, migrate: str = "") -> dict:
    """Regenerate the baseline. DOWNWARD ONLY.

    ⚠ THE WRITER HAD NO ENFORCEMENT (§4GJ round 4) — it wrote
    `measure_suite()` unconditionally, while the file it emits says
    "regenerate DOWNWARD only" and `scripts/lint.py --write-baseline`
    refuses in so many words. Confirmed on a copy of the real tree: add a
    text pin, run the writer, baseline 633 -> 635, rc=0. A ratchet whose
    writer will write anything is a ratchet with a reset button, and the
    reset is one command shorter than the fix.

    `migrate` is the deliberate exception, and it is not free: it stamps the
    reason and the numbers it overwrote into the committed file, so an
    upward write is a thing a reader can see and challenge. (This is the
    `--seed` shape from the lint gate: widening what is MEASURED is a real
    need, and the answer to it is an audit trail, not silence.)

    ⚠ AND THE REFUSAL WAS SKIPPED WHEN THE FILE WAS ABSENT (§4GJ round 5).
    `grew = growth_report(...) if old else []` — so deleting the baseline
    disarmed the guard that the baseline exists to hold. Reproduced: remove
    `pin_quality_baseline.json`, write numbers above the committed total,
    and there was no refusal, no `_migration` stamp, and nothing downstream
    asserting either; the gate then read the inflated file and passed. With
    no git in this tree, "delete and rewrite" was also the ONLY available
    recovery, which is precisely why it must cost a recorded reason. An
    absent predecessor is now the loudest case, not the quietest: `grew` is
    computed unconditionally (against an empty baseline every file has
    grown), and a missing file is refused by name.
    """
    path = BASELINE_PATH if path is None else path
    current = measure_suite() if current is None else current
    existed = path.exists()
    old = json.loads(path.read_text(encoding="utf-8")) if existed else {}
    grew = growth_report(current, old.get("per_file", {}))
    if not existed and not migrate:
        raise SystemExit(
            f"REFUSING to write {path.name} from scratch: the file is "
            "MISSING, so there is nothing to ratchet against and this write "
            f"would set the tolerance to whatever the suite happens to hold "
            f"right now ({sum(current.values())} text pins across "
            f"{len(current)} files).\n\nThe ratchet's memory does not "
            "regenerate — restore the file, or, if this really is a first "
            "write, say so on the record with "
            "`--migrate \"why there is no baseline\"`, which stamps the "
            "reason into the file it creates.")
    if grew and not migrate:
        raise SystemExit(
            "REFUSING to write a pin-quality baseline that GREW:\n  "
            + "\n  ".join(f"{n}: {now} now vs {was} at baseline (+{now - was})"
                          for n, now, was in grew)
            + "\n\nThe baseline ratchets DOWNWARD only — that is the whole "
              "mechanism. Remove the text pin, or (if the MEASUREMENT "
              "changed, not the suite) re-run with "
              "`--migrate \"why the count moved\"`, which records the reason "
              "in the baseline for the next reader.")
    payload = {
        "_comment": (
            "§4GJ pin-quality ratchet. Source-TEXT pins (R4 hard-reject) per "
            "test file; regenerate DOWNWARD only, with "
            "`python tests/test_pin_quality_ratchet.py --write`. The slack "
            "ceiling is ALLOWED_SLACK in the module, not a key here."),
        "total_textual": sum(current.values()),
        "per_file": current,
    }
    if migrate:
        # ⚠ THE RECORD MUST BE ABLE TO SUPPORT ITS OWN CLAIM (§4GJ round 5).
        # It used to carry `previous_total`, `previous_files` and
        # `files_that_grew` — from which "purely additive" is NOT decidable:
        # 250 files before and 250 after is equally consistent with 250
        # unchanged and with one file lost plus one added, and a file whose
        # count DROPPED inside an upward migration left no trace at all. The
        # 633 -> 1312 unit change really was additive (independently
        # reconstructed: 0 shrank, 0 lost, 0 added, 178 grew), but the block
        # it wrote could not say so — the next reader had to take the prose
        # on trust. These four counts decide it: additive means lost,
        # added and shrank are all zero.
        old_files = old.get("per_file", {}) or {}
        payload["_migration"] = {
            "reason": migrate,
            "previous_total": old.get("total_textual"),
            "previous_files": len(old_files),
            "new_total": payload["total_textual"],
            "new_files": len(current),
            "files_that_grew": len(grew),
            "files_that_shrank": sum(1 for n, was in old_files.items()
                                     if n in current and current[n] < was),
            "files_lost": sum(1 for n in old_files if n not in current),
            "files_added": sum(1 for n in current if n not in old_files),
            "bootstrap": not existed,
        }
    elif old.get("_migration"):
        # ⚠ AND THE CARRY-FORWARD IS WHAT MADE THE EXEMPTION PERMANENT (§4GK
        # round 7). This line copied the grandfathered four-field record into
        # every later baseline verbatim, so a record that predates the format
        # — and describes numbers that have since moved — went on vouching for
        # files it knows nothing about. A record that still DECIDES its claim
        # for the file it is about to sit in is carried; one that does not
        # has expired, and the write stops here rather than laundering it.
        carried = old["_migration"]
        if migration_gaps(carried, payload):
            raise SystemExit(
                f"REFUSING to copy {path.name}'s _migration record into a "
                f"baseline it no longer describes. It is the grandfathered "
                f"pre-format record (four fields, no counts), exempt only "
                f"for the exact numbers it was written for — and this write "
                f"changes them ({old.get('total_textual')} -> "
                f"{payload['total_textual']} text pins across "
                f"{len(current)} files).\n\nThat exemption has now expired. "
                f"Re-run with `--migrate \"why the count moved\"`, which "
                f"stamps all ten counts and makes the new record decide its "
                f"own claim.")
        payload["_migration"] = carried
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return payload


def _reason_from_argv(argv: list) -> str:
    """The `--migrate REASON` argument, or "".

    ⚠ `sys.argv[sys.argv.index("--migrate") + 1]` RAISED IndexError when
    `--migrate` was the last token (§4GJ round 5) — a traceback instead of
    the one sentence the operator needed, on the command that overrides the
    ratchet. And `--write --migrate --something` silently took the next FLAG
    as the reason, stamping `"reason": "--something"` into the committed
    file, which is worse than the crash: an audit trail that records
    nothing is still an audit trail as far as every reader is concerned.
    """
    if "--migrate" not in argv:
        return ""
    i = argv.index("--migrate")
    reason = argv[i + 1] if i + 1 < len(argv) else ""
    if not reason.strip() or reason.startswith("--"):
        raise SystemExit(
            "--migrate needs a REASON: --migrate \"why the count moved\". It "
            "is stamped into the baseline, which is the whole point of "
            "allowing an upward write at all.")
    return reason


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    import sys

    if "--write" in sys.argv:
        reason = _reason_from_argv(sys.argv)
        p = _write_baseline(migrate=reason)
        print(f"wrote {BASELINE_PATH.name}: {p['total_textual']} textual pins "
              f"across {len(p['per_file'])} files")
        print("update _COMMITTED_BASELINE_DIGEST in this file to:\n"
              f"    {baseline_digest(p)}")
    else:
        cur = measure_suite()
        for name, count in sorted(cur.items(), key=lambda kv: -kv[1]):
            print(f"{count:4d}  {name}")
        print(f"total textual: {sum(cur.values())} across {len(cur)} files")


# ── §4GK round 7: the grandfather clause was forgeable, and never expired ───


def _synthetic_grandfathered(monkeypatch):
    """A baseline carrying the LEGACY record, with the in-code digest patched
    to match it.

    ⚠ THE FIXTURE MOVED OFF THE COMMITTED FILE (§4GO, 2026-09-14). These two
    tests used the real baseline as their grandfathered specimen, which was
    free while the committed record WAS the legacy one. The migration that
    closed the non-Python measurement hole replaced it — exactly what the
    clause was designed to do — so the mechanism is now pinned against a
    specimen of its own. It still decides any tree that carries the legacy
    record; it just no longer borrows this repo's numbers to do it.
    """
    base = {
        "_comment": "synthetic",
        "allowed_slack": 0,
        "per_file": {"a.py": 2, "b.py": 3},
        "total_textual": 5,
        "_migration": dict(_GRANDFATHERED_MIGRATION,
                           reason="no file lost, no file decreased"),
    }
    monkeypatch.setattr(sys.modules[__name__], "_GRANDFATHERED_BASELINE",
                        baseline_digest(base), raising=False)
    return base


def test_the_grandfathered_record_cannot_vouch_for_a_REWRITTEN_baseline(
        tmp_path, monkeypatch):
    """THE LAUNDERING LOOP, replayed end to end.

    Measured on the round-6 code: grow a test file from 2 to 5 text pins (the
    gate goes correctly red), hand-rewrite `per_file`/`total_textual` in the
    baseline to the new numbers, keep the `_migration` block VERBATIM — 36
    passed. The provenance check was satisfied by three integers that are
    printed in the baseline itself and a `reason` string that is not
    constrained at all, so the record vouched for numbers it had never seen.

    The record is now tied to the numbers it describes. Any edit to them —
    honest or not — expires the exemption, and the honest answer is a
    `--migrate` write that stamps all ten counts.

    World where it fails: a grandfather clause keyed on the record alone
    (the rewritten baseline below passes provenance).
    """
    real_baseline = BASELINE_PATH          # the name is rebound below
    committed = _synthetic_grandfathered(monkeypatch)
    record = committed["_migration"]
    assert not migration_gaps(record, committed), (
        "the specimen is not grandfathered — this test proves nothing")

    laundered = dict(committed)
    laundered["per_file"] = dict(committed["per_file"])
    victim = sorted(laundered["per_file"])[0]
    laundered["per_file"][victim] += 3          # the growth being hidden
    laundered["total_textual"] = sum(laundered["per_file"].values())

    assert migration_gaps(record, laundered), (
        "the pre-format record still vouches for a baseline it does not "
        "describe — the numbers under it were rewritten")

    # and the GATE says so, not just the helper: drive the real provenance
    # test against the rewritten file.
    path = tmp_path / "pin_quality_baseline.json"
    path.write_text(json.dumps(laundered, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    monkeypatch.setitem(globals(), "BASELINE_PATH", path)
    with pytest.raises(AssertionError) as exc:
        test_the_baseline_can_say_where_it_came_from()
    assert "_migration record is missing" in str(exc.value)

    # INVERSE: the REAL committed file, untouched, still passes — this must
    # not become "no baseline can ever have provenance". (It passes by
    # carrying all ten counts now, not by the grandfather clause.)
    monkeypatch.setitem(globals(), "BASELINE_PATH", real_baseline)
    monkeypatch.undo()
    test_the_baseline_can_say_where_it_came_from()


def test_the_writer_will_not_carry_an_EXPIRED_record_forward(tmp_path,
                                                            monkeypatch):
    """`elif old.get("_migration"): payload["_migration"] = old["_migration"]`
    copied the pre-format record into every later baseline verbatim, so the
    exemption outlived the file it was written for — for ever, by design of
    that one line. A record that no longer decides its claim for the file it
    is about to sit in has expired, and the write says so instead of
    laundering it.

    World where it fails: an unconditional carry-forward (the write succeeds
    and the next baseline claims provenance it does not have).
    """
    committed = _synthetic_grandfathered(monkeypatch)
    path = tmp_path / "b.json"
    path.write_text(json.dumps(committed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")

    # a strictly DOWNWARD write: one file's pins removed, nothing grown
    shrunk = dict(committed["per_file"])
    del shrunk[sorted(shrunk)[0]]

    with pytest.raises(SystemExit) as exc:
        _write_baseline(shrunk, path)
    assert "no longer describes" in str(exc.value)
    assert "--migrate" in str(exc.value)
    assert json.loads(path.read_text()) == committed, \
        "the refused write changed the file anyway"

    # INVERSE 1: with the reason on the record, the write lands and the new
    # record carries all ten counts — the exemption is replaced, not removed.
    payload = _write_baseline(shrunk, path, migrate="dropped a file's pins")
    assert not migration_gaps(payload["_migration"], payload)
    assert payload["_migration"]["files_lost"] == 1

    # INVERSE 2: a COMPLETE record is carried forward by an ordinary write,
    # exactly as before — only the grandfathered one expires.
    smaller = dict(shrunk)
    smaller[sorted(smaller)[0]] = 0
    again = _write_baseline(smaller, path)
    assert again["_migration"] == payload["_migration"]
