"""Every decision in the dispatch block is keyed on the RESOLVED tool name.

`_canonicalise_tool_name` used to run ~330 lines after the guards that
classify a call, so each guard silently asked about a different tool than the
one that would execute. `_TOOL_ALIAS_TABLE` maps knowledgebase / knowledge-base
/ kb -> knowledge_base and filesystem / fs -> file_system, and its own comment
cites "knowledgebase" as an observed Qwen 3.5 hallucination, so these are names
the model really emits.

One consequence per consumer, all measured:

  * the memory-wipe flag stayed False while the wipe ran, so the turn's
    smart_memory / post_mortem / episode write re-learned the content the user
    had just deleted;
  * `is_mutating` classified two identical `knowledgebase` ingests as
    read-safe, and the second was dedup-collapsed — a real ingest dropped;
  * `is_idempotent_setter` stopped blocking repeated `kb` profile writes;
  * `is_sandbox_mutation` left the workspace cache stale after an `fs` write;
  * the empty-`content` write block was bypassed by `fs` outright;
  * the dedup hash keyed on the raw string, so the same call under two
    spellings never matched itself.

The name is resolved ONCE now, at the top of the block, and the dispatch reuses
that value — so the name every guard was keyed on is provably the one that
runs. These are AST pins: the property is "no guard in this block reads the raw
name", which is about the code, not about one input.
"""

import ast
import inspect

import pytest

from ghost_agent.core.agent import GhostAgent


@pytest.fixture(scope="module")
def block():
    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    return ast.parse(src.lstrip())


def _assignments(tree, name):
    return [ast.dump(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == name]


def test_the_name_is_resolved_once(block):
    resolved = [ast.dump(n.value) for n in ast.walk(block)
                if isinstance(n, ast.Assign)
                and "_canonicalise_tool_name" in ast.unparse(n.value)]
    assert len(resolved) == 1, (
        f"the tool name is resolved {len(resolved)} times in this block; two "
        f"resolutions of one decision drift by construction — that is how the "
        f"0.85-vs-0.70 difflib split happened"
    )
    assert "_canonicalise_tool_name" in resolved[0], (
        "it must be the DISPATCHER'S matcher, not a second implementation"
    )


@pytest.mark.parametrize("var", [
    "is_mutating", "is_idempotent_setter", "is_sandbox_mutation", "a_hash",
])
def test_the_classifiers_read_the_resolved_name(block, var):
    bodies = _assignments(block, var)
    assert bodies, f"{var} was renamed or moved"
    joined = " ".join(bodies)
    assert "'_cname'" in joined, f"{var} no longer reads the resolved name"
    # ABSENCE, not presence: these are multi-clause expressions, and checking
    # only that `_cname` appears somewhere left a mutant alive that reverted
    # ONE clause to `fname` while the others kept the assertion true.
    assert "'fname'" not in joined, (
        f"{var} still reads the RAW tool name in at least one clause"
    )
    # ONE source for the tool name. Checking only for the absence of the
    # token `fname` was evaded by spelling the same raw read differently:
    # `tool["function"]["name"] in [...]` restored the defect and walked
    # past every pin here. A subscript in these expressions is a second way
    # to ask the same question.
    assert "'tool'" not in joined, (
        f"{var} reads the tool name out of the raw call dict; the resolved "
        f"name is the only source these guards may consult"
    )


def test_no_guard_in_the_block_reads_the_raw_name_at_all(block):
    """Any pre-rebind use of the raw name to decide something about a TOOL.

    ⚠ The first version looked only for `ast.Compare` whose `left` is
    `Name('fname')`, against the names in `TOOL_DEFINITIONS`. Three
    single-edit reverts walked past it, each restoring a measured defect:
    `fname.startswith("file_system")` (not a Compare),
    `"learn_skill" == fname` (operand order), and
    `fname == "image_generation"` — because `image_generation` and
    `vision_analysis` are dispatchable but absent from the static
    `TOOL_DEFINITIONS`. Scan for the VARIABLE, not for one syntax.
    """
    from ghost_agent.tools.registry import TOOL_DEFINITIONS

    tools = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    # Dispatchable-but-unadvertised names the static list omits.
    tools |= {"image_generation", "vision_analysis"}

    rebind = [n.lineno for n in ast.walk(block)
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == "fname"
              and isinstance(n.value, ast.Name)]
    assert len(rebind) == 1, (
        "expected exactly one `fname = <resolved>` rebind; without it there "
        "is no point after which reading the raw name is safe"
    )
    cutoff = rebind[0]

    offenders = []
    for node in ast.walk(block):
        if getattr(node, "lineno", 0) > cutoff or getattr(node, "lineno", 0) == 0:
            continue
        # Any expression mentioning `fname` in the same breath as a tool name.
        if not isinstance(node, (ast.Compare, ast.Call)):
            continue
        src = ast.unparse(node)
        if "fname" not in src:
            continue
        named = {t for t in tools if f"'{t}'" in src or f'"{t}"' in src}
        if named:
            offenders.append(f"{src[:90]} -> {sorted(named)}")
    assert not offenders, (
        f"these run BEFORE the rebind and decide something about a tool from "
        f"the RAW name: {offenders}. A model emitting an alias gets a "
        f"different answer here than from the dispatch that follows."
    )


def test_the_dispatch_reuses_the_resolution(block):
    """If the dispatch resolved the name again it could, in principle,
    resolve it differently — and every guard above would have been keyed on
    the other answer."""
    calls = [n for n in ast.walk(block)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_canonicalise_tool_name"]
    assert len(calls) == 1, (
        f"{len(calls)} canonicalisation calls in this block; there must be "
        f"exactly one, whose result everything downstream reuses"
    )


# ------------------- the wiring, not just the helpers it is supposed to call

def test_the_disabled_tools_gate_is_deliberately_raw(block):
    """This pin used to assert the OPPOSITE — that the gate also checked the
    resolved name. A review showed that branch is unreachable by
    construction: `_canonicalise_tool_name` only ever returns a member of
    `available_tools`, and every containment site (subagent, isolation,
    dream, the rebuild) removes disabled names from that map, so
    `available ∩ disabled = ∅`. In the only configuration where it could
    have fired it would have blocked the wrong tool. An alias of a disabled
    tool now fails to resolve and falls through to the unknown-tool error.
    """
    gates = [ast.dump(n) for n in ast.walk(block)
             if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.In)
             and "disabled_tools" in ast.dump(n)]
    # ⚠ ASSERT THE LOOP FOUND SOMETHING FIRST. The assertions used to live
    # inside the `for`, so deleting the gate outright executed ZERO of them
    # and the test passed — a guard that never runs, pinned by a test that
    # never runs.
    assert gates, "the disabled-tools gate is gone entirely"
    for src in gates:
        assert "'fname'" in src, "the gate stopped reading the raw name"
        assert "'_cname'" not in src, (
            "the resolved name cannot be in disabled_tools — "
            "canonicalisation only returns available tools, and disabled "
            "tools are removed from that map"
        )


def test_the_invocation_error_handler_uses_the_describer(block):
    """`describe_invocation_error` being correct is worth nothing if the
    handler still formats its own string — and a unit test of the helper
    cannot tell the difference."""
    calls = [n for n in ast.walk(block)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "describe_invocation_error"]
    # There are THREE sites — the binding-time except, the awaited-result
    # formatter, and the strike-ledger record. `assert calls:` passed with
    # two of the three reverted, including the one this test names.
    assert len(calls) >= 3, (
        f"only {len(calls)} describe_invocation_error call site(s); an "
        f"exception reaching the model by any other route still reads as "
        f"'(Did you forget a required argument?)' — or, on the awaited "
        f"path, as a clean success"
    )


def test_the_mixed_turn_preview_budget_is_conditional(block):
    """This pin exists because the first version of it recomputed the budget
    in the test — `300 if is_argument_error(err) else 140` — and asserted
    against its own arithmetic. Both a mutant that cut the budget back to a
    flat 140 and one that widened everything to 300 passed it. Read the
    budget out of the CODE."""
    slices = []
    for node in ast.walk(block):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) \
                and isinstance(node.slice.upper, ast.IfExp):
            slices.append(node.slice.upper)
    conditional = [
        s for s in slices
        if isinstance(s.test, ast.Call) and isinstance(s.test.func, ast.Name)
        and s.test.func.id == "is_argument_error"
        # ...ON THE RESULT. `is_argument_error(fname)` and
        # `is_argument_error(str_res[:100])` both truncated every argument
        # error to 140 — the exact symptom the widening was for — and both
        # passed a pin that only checked the function's name.
        and s.test.args and ast.unparse(s.test.args[0]) == "str_res"
    ]
    assert conditional, (
        "the failure preview is truncated at a flat width; an argument "
        "error's worked call — the whole recovery path — sits at the END of "
        "the message and is cut out of any turn that mixed a success with a "
        "failure"
    )
    widths = {(getattr(s.body, "value", None), getattr(s.orelse, "value", None))
              for s in conditional}
    assert widths == {(300, 140)}, (
        f"unexpected preview budgets {widths}: argument errors need the room, "
        f"everything else must NOT get it for free"
    )


def _budget_for_argument_errors() -> int:
    """The argument-error preview width, READ FROM THE SOURCE so callers
    cannot assert against their own copy of the number."""
    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    for node in ast.walk(ast.parse(src.lstrip())):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) \
                and isinstance(node.slice.upper, ast.IfExp):
            test = node.slice.upper.test
            if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) \
                    and test.func.id == "is_argument_error":
                return node.slice.upper.body.value
    raise AssertionError("no conditional preview budget found")


def test_the_two_sandbox_op_lists_stay_identical(block):
    """`is_mutating` and `is_sandbox_mutation` both list the `file_system`
    operations that change the workspace, and the comment on one says to
    keep them in sync with the other. They drifted: `copy` was added to one
    in the F2b fix and not the other, so a copy created a file while
    `cached_sandbox_state` stayed valid and every later turn in the request
    listed a workspace without it. Pinned as an identity rather than as two
    lists to read."""
    ops = {}
    for node in ast.walk(block):
        if not (isinstance(node, ast.Compare)
                and isinstance(node.ops[0], ast.In)
                and isinstance(node.comparators[0], ast.List)):
            continue
        values = {e.value for e in node.comparators[0].elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        if "write" in values and "replace" in values:
            ops.setdefault("seen", []).append(values)
    # ...and each list must actually be guarded by a `file_system` test.
    # Comparing the two SETS to each other and to `expected` still passed
    # when one list was re-keyed onto `_cname == "execute"`, which would mean
    # no file_system write ever invalidates the sandbox cache.
    fs_guarded = 0
    for node in ast.walk(block):
        if not isinstance(node, ast.BoolOp):
            continue
        src = ast.unparse(node)
        if "'file_system'" in src and "'write'" in src and "'replace'" in src:
            fs_guarded += 1
    assert fs_guarded >= 2, (
        f"only {fs_guarded} of the two mutating-op lists is guarded by a "
        f"file_system test; the other decides on some different tool"
    )
    seen = ops.get("seen") or []
    assert len(seen) >= 2, (
        f"expected the file_system op list in both is_mutating and "
        f"is_sandbox_mutation; found {len(seen)}"
    )
    assert all(v == seen[0] for v in seen), (
        f"the file_system operation lists have drifted: "
        f"{[sorted(v ^ seen[0]) for v in seen if v != seen[0]]}. A workspace "
        f"mutation classified by one and not the other leaves the sandbox "
        f"cache stale."
    )
    # CONTENT, not just equality — dropping an op from BOTH lists kept them
    # equal and left `delete` no longer invalidating the sandbox cache.
    expected = {"write", "replace", "download", "delete", "move", "rename",
                "unzip", "git_clone", "copy"}
    assert seen[0] == expected, (
        f"the file_system mutating-op set changed: "
        f"{sorted(seen[0] ^ expected)}. Every one of these writes to the "
        f"workspace; an op missing here leaves the sandbox cache valid "
        f"after a real change and makes the call dedup-collapse eligible."
    )


def test_a_real_tool_name_outranks_an_alias():
    """`_TOOL_ALIAS_TABLE` used to be consulted before the exact match, and
    twelve of its keys (`fs`, `kb`, `search`, `system`, `vision`, …) are
    legal acquired-skill names — the shadow checks reject only EXACT
    built-in names. A skill actually called `fs` dispatched as itself while
    every name-keyed guard was told it was `file_system`; now that the
    guards act on that answer, one of them blocks it outright."""
    available = ["file_system", "web_search", "knowledge_base", "fs"]
    assert GhostAgent._canonicalise_tool_name("fs", available) == "fs"
    # ...and with no such skill, the alias still means the built-in.
    assert GhostAgent._canonicalise_tool_name(
        "fs", ["file_system", "web_search"]) == "file_system"
    # EVERY alias must reach its target. The lookup key has every
    # non-alphanumeric stripped, so a table key written with a separator
    # (`profile_update`) could never match and was dead from the day it was
    # added; the table is normalised at lookup now.
    for alias, target in GhostAgent._TOOL_ALIAS_TABLE.items():
        assert GhostAgent._canonicalise_tool_name(alias, [target]) == target, (
            f"the alias {alias!r} never resolves to {target!r}"
        )


def test_the_name_is_resolved_against_a_refreshed_map(block):
    """`available_tools` is a cached dict on a lifespan singleton, so a
    skill created mid-session is missing until a rebuild — and resolving
    against the stale map turned a brand-new `update_prices` into a difflib
    guess of `update_profile`, which every guard then acted on. The refresh
    must happen on the same condition, BEFORE the resolution."""
    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    lines = src.splitlines()
    rebuild = next(i for i, l in enumerate(lines)
                   if "_rebuild_available_tools()" in l)
    resolve = next(i for i, l in enumerate(lines)
                   if "_canonicalise_tool_name" in l and "=" in l)
    assert rebuild < resolve, (
        "the tool map is refreshed after the name is resolved, so the guards "
        "and the dispatch can see different tool lists"
    )


def test_every_action_comparison_in_is_mutating_is_healed(block):
    """`is_mutating` healed the `knowledge_base` action with
    `.strip().lower()` and matched `manage_composed_skills` raw on the very
    next line, so `action="Define"` mutated the skill registry and scored
    read-safe. Pinned as a property of the whole expression rather than of
    one tool, because that is how the two lines drifted."""
    raw = []
    for node in ast.walk(block):
        if not (isinstance(node, ast.Compare)
                and isinstance(node.ops[0], ast.In)):
            continue
        left = node.left
        # A bare `t_args.get("action")` on the left of an `in` — no
        # `.strip().lower()` between the read and the comparison.
        if isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute) \
                and left.func.attr == "get" and left.args \
                and isinstance(left.args[0], ast.Constant) \
                and left.args[0].value == "action":
            raw.append(ast.unparse(node)[:80])
    assert not raw, (
        f"these action comparisons read the raw value: {raw}. A model that "
        f"capitalises or pads the action gets a different classification "
        f"than the dispatcher, which strips and lowercases it."
    )


def test_the_forget_schema_matches_what_the_sweep_does(block):
    """The schema said "the sweep covers sandbox files" with no
    qualification while the runtime deletes only exact/stem matches and
    lists the rest — so the schema promised a deletion the tool refuses,
    and contradicted the report's own instruction to re-issue with an exact
    name."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS

    kb = next(t["function"] for t in TOOL_DEFINITIONS
              if t["function"]["name"] == "knowledge_base")
    desc = kb["parameters"]["properties"]["target"]["description"]
    assert "EXACTLY" in desc, (
        "the schema must say disk deletion is exact-match only; a model told "
        "'the sweep covers sandbox files' expects the partial matches to go"
    )
    assert "relative to the sandbox" in desc, (
        "the report tells the caller to re-issue with a name it prints as a "
        "relative path; the schema has to say that is accepted"
    )
