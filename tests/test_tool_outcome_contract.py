"""Every non-OK tool result is either a `ToolOutcome` or on a shrinking list.

WHY THIS TEST IS SHAPED LIKE THIS
=================================
An audit (§4DO) AST-parsed every ``return``/``raise`` in `ghost_agent/tools/`
— 1,026 result heads — evaluated every "did this fail?" predicate in the tree
over each, and cross-checked against 4,391 recorded live tool calls. It found
**ten** vocabularies where the code claimed one, and measured the cost: 63
live refusals credited with changing the world, 15 turns reporting a non-zero
exit as SUCCEEDED under an authoritative banner, 219 calls where the turn loop
and the trajectory corpus disagreed about whether the call failed.

Five review rounds had answered by adding prefixes, and **each addition
created the next round's split**. So the pin is at the PRODUCER, not at any
predicate: it walks the same AST the audit did and fails when a new
failure-shaped `return` appears that is neither a `ToolOutcome` nor listed
below. That is the failure mode every previous round's pin missed — they all
guarded a reader, and the next defect arrived through a writer.

The allowlist may only ever SHRINK. Adding to it is the thing this test
exists to make deliberate.
"""

import ast
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "tools"

#: Heads a STRING CANNOT EXPRESS. This is the line the pin guards, and it is
#: the line the audit drew: a plain `Error:` return classifies correctly
#: through the one predicate, but nothing in a string can say "this REFUSED
#: and touched nothing" or "this half-landed" — and those are precisely the
#: two distinctions whose absence caused the measured harm (63 refusals
#: credited with changing the world; PARTIAL writes recorded as applied so
#: the model's retry was refused).
#:
#: So: refusal- and partial-shaped returns MUST be ToolOutcome, everywhere,
#: with no allowlist. Plain failures may remain strings.
_MUST_BE_OUTCOME = (
    "SYSTEM INSTRUCTION", "REJECTED", "SYSTEM BLOCK", "PARTIAL:",
    # ⚠ Added after a reviewer found `execute`'s egress refusal — whose own
    # text says "command NOT executed" — sitting outside every head this
    # scanned, so "zero bare refusals package-wide" was true and VACUOUS.
    # A refusal is defined by what it did (nothing), not by its first word.
    "SANDBOX EGRESS BLOCKED", "REPLACE REJECTED",
)


def _leading_literal(node) -> str | None:
    """The literal text a return expression STARTS with, however it is built.

    ⚠ The first version handled one level: `Constant`, a `JoinedStr` whose
    first part is a Constant, and a `BinOp` whose `.left` is a Constant. A
    reviewer walked it through nine shapes and it missed five — including
    two LIVE bare refusals it was reporting the package clean of:
    `"SYSTEM INSTRUCTION: …" + filename + "…"` (a nested BinOp, so `.left`
    is another BinOp) and `f"REJECTED: …" + (…)` (`.left` is a JoinedStr).
    Descend to the leftmost leaf instead of testing one layer.
    """
    seen = 0
    while node is not None and seen < 20:
        seen += 1
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.JoinedStr):
            node = node.values[0] if node.values else None
            continue
        if isinstance(node, ast.BinOp):
            node = node.left
            continue
        if isinstance(node, ast.FormattedValue):
            # an f-string that STARTS with a placeholder — the text is not
            # knowable here; the variable-prefix case is checked separately
            return None
        return None
    return None


def _failure_returns(path: Path):
    """Every bare refusal-shaped result a module can hand back.

    Covers `return "<literal>"`, f-strings, concatenations — and
    module-level CONSTANTS assigned a refusal-shaped string, because
    `execute.py` returned one by name (`return _AGENT_PORT_PROBE_MSG`) and
    a scan of return literals alone could not see it.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover
        return []
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        head = _leading_literal(node.value)
        if head and head.lstrip().startswith(_MUST_BE_OUTCOME):
            hits.append((node.lineno, head[:70]))

    # Module-level constants holding a refusal, and whether they are ever
    # returned bare.
    const_heads = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            v = node.value
            head = None
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                head = v.value
            elif isinstance(v, ast.JoinedStr) and v.values \
                    and isinstance(v.values[0], ast.Constant):
                head = v.values[0].value
            elif isinstance(v, ast.BinOp) and isinstance(v.left, ast.Constant) \
                    and isinstance(v.left.value, str):
                head = v.left.value
            if isinstance(head, str) and head.lstrip().startswith(_MUST_BE_OUTCOME):
                const_heads[node.targets[0].id] = head
    # `raise`, which the audit counted and this did not inspect at all.
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            args = exc.args if isinstance(exc, ast.Call) else []
            for a in args:
                h = _leading_literal(a)
                if h and h.lstrip().startswith(_MUST_BE_OUTCOME):
                    hits.append((node.lineno, h[:70]))

    # ...returned by name from a PUBLIC function. A private helper handing a
    # marker back to its caller is plumbing — the caller is where the value
    # becomes a tool result, and that is where the wrap belongs.
    #
    # ⚠ STATED LIMIT: a refusal that leaves a public entry point already
    # wrapped in another call (`return _format_error(msg)`) is invisible
    # here. That shape is why `execute`'s daemon block went unmigrated; it
    # was found by reading, not by this scan. The scan catches the common
    # cases and is not a proof.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name.startswith("_"):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) \
                    and node.value.id in const_heads:
                hits.append((node.lineno, const_heads[node.value.id][:70]))
    return hits


PKG = TOOLS.parent


def _modules():
    """Every module that can PRODUCE a tool result.

    Was `tools/*.py` only, which is a scope hole rather than a rule: a
    refusal minted anywhere the dispatch loop can receive it has the same
    cost. Anything defining a `tool_*` entry point counts, wherever it
    lives.
    """
    mods = {p for p in TOOLS.glob("*.py") if p.name != "__init__.py"}
    for p in PKG.rglob("*.py"):
        if p in mods or p.name == "__init__.py":
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\ndef tool_" in src or "\nasync def tool_" in src:
            mods.add(p)
    return sorted(mods)


def test_the_scanner_can_actually_see_a_bare_refusal(tmp_path):
    """POSITIVE CONTROL for the pin above.

    The pin is green because its result set is EMPTY, which is exactly the
    state a broken scanner produces. A reviewer disabled each arm of
    `_failure_returns` in turn — the return-literal arm, `_leading_literal`,
    and the `_MUST_BE_OUTCOME` heads — and the whole suite stayed green
    through all three. A pin that cannot fail is documentation.

    Every head in `_MUST_BE_OUTCOME` is checked, in each shape the scanner
    claims to descend into: a plain literal, an f-string, a concatenation
    and a module constant.
    """
    for head in _MUST_BE_OUTCOME:
        mod = tmp_path / "fixture_tool.py"
        mod.write_text(
            "CONST = %r\n"
            "\n"
            "def tool_plain():\n"
            "    return %r\n"
            "\n"
            "def tool_fstring(x):\n"
            "    return f%r\n"
            "\n"
            "def tool_concat():\n"
            "    return %r + 'tail'\n"
            "\n"
            "def tool_const():\n"
            "    return CONST\n"
            % (head + " const", head + " plain", head + " fstr {x}",
               head + " concat"),
            encoding="utf-8")
        hits = _failure_returns(mod)
        assert len(hits) == 4, (
            f"the producer scanner found {len(hits)}/4 bare {head!r} "
            f"returns — it cannot see the thing it exists to catch: "
            f"{hits}"
        )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_returns_a_bare_refusal(path):
    """No allowlist. A refusal or a partial that leaves this package as a
    plain string is a distinction the turn loop cannot recover — and the
    audit measured what that costs.

    This is a PRODUCER-side pin on purpose. Five review rounds guarded
    readers, and each round's defect arrived through a writer instead: the
    next `return "SYSTEM INSTRUCTION: …"` someone adds fails here, at the
    line they are writing.
    """
    hits = _failure_returns(path)
    assert not hits, (
        f"{path.name} returns bare refusal/partial strings at "
        f"{[ln for ln, _ in hits]}:\n"
        + "\n".join(f"  line {ln}: {head!r}" for ln, head in hits)
        + "\n\nUse ToolOutcome.rejected(...) — a refusal touched NOTHING, "
          "and saying so is the only way the world-changed credit, the "
          "idempotency record and the strike decay can get it right — or "
          "ToolOutcome.partial(...) for a write that half-landed."
    )


def test_the_whole_package_is_clean():
    """The count, as a fact rather than a claim. It was 12 when the
    migration landed and may only go to 0."""
    remaining = {p.name: _failure_returns(p) for p in _modules()}
    remaining = {k: v for k, v in remaining.items() if v}
    assert not remaining, f"bare refusals remain: { {k: len(v) for k, v in remaining.items()} }"


class TestTheContractItself:
    def test_a_refusal_changed_nothing(self):
        from ghost_agent.tools.outcome import ToolOutcome

        o = ToolOutcome.rejected("REPLACE REJECTED (byte-identical)",
                                 reason_code="byte_identical")
        assert o.is_failure and o.is_rejection
        assert o.changed_the_world is False
        assert o.may_record_as_applied is False

    def test_a_partial_write_is_not_applied(self):
        from ghost_agent.tools.outcome import ToolOutcome

        o = ToolOutcome.partial("PARTIAL: memory write had failures")
        assert o.is_failure and not o.is_rejection
        # it DID touch the world — that is what makes it partial
        assert o.changed_the_world is True
        assert o.may_record_as_applied is False

    def test_a_non_refusal_string_classifies_against_a_FROZEN_table(self):
        """⚠ Twice now this test has been a tautology.

        v1 asserted `o.is_failure == (result_is_failure(t) or
        result_is_rejection(t))` — `coerce`'s own definition restated. v2
        asserted `o.is_failure == result_is_failure(t)`, which is that same
        definition with the refusal arm excluded by the precondition it
        also asserts. A reviewer searched 200,092 strings meeting v2's
        precondition and found ZERO that could falsify it, then confirmed by
        mutation: `result_is_failure := return False` passed it, narrowing
        `Error\\b` back to `Error:` passed it, and deleting `CRITICAL
        ERROR:` passed it.

        The fix is to stop comparing the rule against itself. Every entry
        below is a head an actual producer in this tree emits, with the
        answer written out by hand. A predicate change that alters any of
        them has to come here and say so.
        """
        from ghost_agent.tools.outcome import ToolOutcome
        from ghost_agent.tools.tool_failure import result_is_failure

        # (text, is a failure?) — hand-written, NOT computed.
        FROZEN = [
            ("Error: not found", True),
            ("Error storing memory: boom", True),
            ("  Error: leading space", True),
            ("ERROR unhandled", True),
            ("SYSTEM ERROR: The 'target' parameter is MANDATORY.", True),
            ("CRITICAL ERROR: the research tool returned nothing", True),
            ("Critical Tool Error: gone", True),
            ("SUCCESS: stored", False),
            ("SUCCESS: Wrote 120 chars to 'a.py'.", False),
            ("", False),
            ("Report: Memory disabled.", False),
            ("Contents of a.py:\n  raise Exception('x')", False),
            ("--- EXECUTION RESULT ---\nEXIT CODE: 0", False),
            ("Errors were avoided", False),
            ("SYSTEM: Found 10 highly relevant memories.", False),
            ("--- app.py CONTENTS ---\nimport os", False),
        ]
        for text, want in FROZEN:
            assert result_is_failure(text) is want, (
                f"result_is_failure({text!r}) changed: "
                f"{result_is_failure(text)} != {want}"
            )
            assert ToolOutcome.coerce(text).is_failure is want, (
                f"coerce({text!r}).is_failure changed"
            )

    def test_a_refusal_string_is_now_a_failure_and_that_is_deliberate(self):
        """The ONE intended behaviour change, stated as a change.

        Before, a `SYSTEM INSTRUCTION: …` refusal from an unmigrated tool
        scored zero strikes — which is how 63 live refusals were booked as
        successes and credited with changing the world. They are failures
        now. The cost, measured by a reviewer: four identical malformed
        calls reach the strike cap and the loop breaker instead of looping
        silently. That is the trade, and it is the point.
        """
        from ghost_agent.tools.outcome import ToolOutcome
        from ghost_agent.tools.tool_failure import result_is_failure

        for text in ("SYSTEM INSTRUCTION: The 'path' is missing",
                     "REJECTED: syntax error",
                     "SYSTEM BLOCK: project is RELEASED",
                     "PARTIAL: 1/2 dispatched"):
            assert not result_is_failure(text), (
                f"{text!r} would have been a failure before too — it does "
                f"not demonstrate the change"
            )
            o = ToolOutcome.coerce(text)
            assert o.is_failure is True
            # ⚠ NOT `is (o.status.value == "partial")` — that recomputes the
            # expectation from the very field a mutant would change, so
            # swapping `coerce`'s rejection check ahead of its `PARTIAL:`
            # check passed (`False is False`). The expected status is
            # written out by hand instead.
            want = "partial" if text.lstrip().startswith("PARTIAL:") \
                else "rejected"
            assert o.status.value == want, (
                f"{text!r}: classified {o.status.value}, expected {want} — "
                "`coerce` must test PARTIAL before REJECTED, because the "
                "rejection regex covers both heads"
            )
            assert o.changed_the_world is (want == "partial"), (
                f"{text!r}: a refusal touched nothing; a partial did"
            )

    def test_an_outcome_is_still_a_string_to_everyone_else(self):
        """~1,000 call sites treat a tool result as text. None of them may
        need to know a tool has been migrated."""
        from ghost_agent.tools.outcome import ToolOutcome

        o = ToolOutcome.rejected("SYSTEM INSTRUCTION: forgot 'replace_with'")
        assert o == "SYSTEM INSTRUCTION: forgot 'replace_with'"
        assert o.lower().startswith("system instruction")
        assert "replace_with" in o
        assert len(o) == len(str(o))
        assert o.splitlines() == [str(o)]
        assert f"{o}" == str(o)


class TestItReallyIsAString:
    """The proxy version had a hole per operation, found one reviewer at a
    time: `isinstance(x, str)` False (an existing test asserts it on a tool
    result), `json.dumps` raised, `"".join` raised, `+` raised,
    `re.search` raised, `os.path.join` raised, and mutating `.text` changed
    the object's hash while it sat in a container. ~1,000 call sites in this
    tree treat a tool result as a string; being one is the only version of
    "behaviour-identical" that holds."""

    def _o(self):
        from ghost_agent.tools.outcome import ToolOutcome
        return ToolOutcome.rejected("SYSTEM INSTRUCTION: forgot 'replace_with'",
                                    reason_code="missing_replace_with")

    def test_it_is_a_str(self):
        assert isinstance(self._o(), str)

    def test_it_serialises(self):
        import json
        assert "forgot" in json.dumps({"r": self._o()})

    def test_it_joins_concatenates_and_formats(self):
        o = self._o()
        assert "|".join(["a", o]).startswith("a|SYSTEM")
        assert ("pre " + o).startswith("pre SYSTEM")
        assert (o + " post").endswith(" post")
        assert f"{o:.6}" == "SYSTEM"
        assert "%s" % o == str(o)

    def test_regex_and_path_apis_accept_it(self):
        import os
        import re
        o = self._o()
        assert re.search(r"forgot", o)
        assert re.sub(r"SYSTEM", "X", o).startswith("X")
        assert os.path.join("/a", "b") in os.path.join("/a", "b")

    def test_it_survives_copy_and_pickle_with_its_status(self):
        import copy
        import pickle

        from ghost_agent.tools.outcome import ToolOutcome
        o = self._o()
        for clone in (copy.copy(o), copy.deepcopy(o), pickle.loads(pickle.dumps(o))):
            assert clone.status is o.status
            assert clone.reason_code == o.reason_code
            assert clone.changed_the_world is False
            assert clone.declared is o.declared

        # A DERIVED outcome is the only one that can catch `__reduce__`
        # dropping `declared`: `_rebuild_outcome` defaults it True, which is
        # exactly what a declared fixture asserts anyway.
        derived = ToolOutcome.coerce("xxd: not found\nEXIT CODE: 127")
        assert derived.declared is False
        for clone in (copy.copy(derived), copy.deepcopy(derived),
                      pickle.loads(pickle.dumps(derived))):
            assert clone.declared is False, (
                "a round-trip re-labelled a DERIVED outcome as a producer's "
                "declaration — that lets it settle the corpus label and the "
                "sniffer becomes unreachable"
            )

    def test_a_refusal_states_world_changed_rather_than_inferring_it(self):
        """`rejected()` sets `world_changed=False` EXPLICITLY. Deleting that
        left every assertion green, because REJECTED is also in
        `_NO_WORLD_CHANGE` so the fallback returns False anyway — two
        mechanisms, one of them untested. The raw field is the one a caller
        can read directly."""
        from ghost_agent.tools.outcome import ToolOutcome

        r = ToolOutcome.rejected("SYSTEM BLOCK: nothing ran")
        assert r.world_changed is False, (
            "the raw field is None — `changed_the_world` only agrees by "
            "accident of the status table"
        )
        # and an explicit override still wins over the table
        assert ToolOutcome.failed("Error: ENOSPC mid-write",
                                  world_changed=True).world_changed is True

    def test_the_status_survives_an_envelope(self):
        """`execute` builds its refusals then wraps them in an
        `--- EXECUTION RESULT ---` envelope. Interpolating into an f-string
        produced a plain str, so the status never reached the loop and three
        migrated sites were inert — surviving only because the envelope
        happens to carry `EXIT CODE: 1`."""
        from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

        import ast
        import inspect

        import ghost_agent.tools.execute as E

        inner = ToolOutcome.rejected("SYSTEM BLOCK: project is RELEASED",
                                     reason_code="project_released")

        # EXTRACT AND RUN the real `_format_error`. It is a closure inside
        # `tool_execute_command`, and hand-building the wrapper here only
        # asserted that `ToolOutcome.__new__` stores what it is given — so
        # deleting execute.py's entire carry-through left this green.
        _src = inspect.getsource(E.tool_execute)
        _fn = next(
            (n for n in ast.walk(ast.parse(_src.lstrip()))
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "_format_error"), None)
        assert _fn is not None, "_format_error moved — re-point this pin"
        # the module's own globals, so the closure's module-level
        # references (`_TIMEOUT_KILL_CODES`, …) resolve as they do live
        _ns = dict(vars(E))
        _ns["ToolOutcome"] = ToolOutcome
        exec(compile(ast.Module(body=[_fn], type_ignores=[]),
                     "<extracted>", "exec"), _ns)
        wrapped = _ns["_format_error"](inner)
        assert wrapped.status is OutcomeStatus.REJECTED
        assert wrapped.changed_the_world is False
        # ...and a bare f-string would have lost it, which is the bug
        assert ToolOutcome.coerce(f"--- EXECUTION RESULT ---\n{inner}").is_rejection is False


class TestTheStatusSurvivesTheMessage:
    """The status used to die the moment the result became a tool message.

    `safe_res` is a truncated plain string, so every downstream reader
    re-sniffed the prose — and measured on five live refusal shapes, the
    loop called them failures while the trajectory corpus, the verifier's
    high-stakes escalation and skills graduation all called them successes.
    That disagreement IS the 63-occurrence class §4DO exists to close; the
    dispatch rewrite only reached half of it.
    """

    def test_the_api_payload_is_byte_identical(self):
        """The status rides INSIDE the content value, so nothing is added to
        the message. An extra dict key would be sent to the LLM API."""
        import json

        from ghost_agent.tools.outcome import ToolOutcome

        text = "SYSTEM INSTRUCTION: forgot 'replace_with'"
        plain = {"role": "tool", "tool_call_id": "c0", "name": "file_system",
                 "content": text}
        carried = dict(plain, content=ToolOutcome.rejected(text))

        assert json.dumps(carried) == json.dumps(plain)
        assert sorted(carried) == sorted(plain)
        assert carried["content"] == text

    def test_the_corpus_prefers_the_status_over_the_sniffer(self):
        """`_reconstruct_tool_calls` set `ToolCall.error` only from
        `_looks_like_tool_error`, which does not match `SYSTEM INSTRUCTION`.
        It reads the recorded status first now, and keeps the sniffer as the
        fallback for historical rows and unmigrated tools."""
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import ToolOutcome

        text = "SYSTEM INSTRUCTION: The 'path' is missing for 'replace'."
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "file_system", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "file_system",
             "content": ToolOutcome.rejected(text)},
        ]
        calls = GhostAgent._reconstruct_tool_calls(msgs)
        assert calls and calls[0].error, (
            "a refusal the loop scored as a failure reached the corpus as a "
            "success — the two halves disagree about the same call"
        )

    def test_an_unmigrated_string_still_uses_the_sniffer(self):
        """The fallback must stay: historical rows and unmigrated tools
        carry no status at all."""
        from ghost_agent.core.agent import GhostAgent

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "execute", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "execute",
             "content": "Error: something broke"},
        ]
        calls = GhostAgent._reconstruct_tool_calls(msgs)
        assert calls and calls[0].error

    def test_a_DERIVED_status_adds_to_the_sniffer_and_never_replaces_it(self):
        """⚠ Read the history before changing this.

        v1 asserted that an OK status SUPPRESSES the sniffer, using a benign
        example. That rule silently killed exit-127 detection: every result
        is wrapped, so the status is never absent on the live path,
        `_looks_like_tool_error` became dead code, and with it the exit-code
        and traceback rules. Measured on the 4,391-call corpus: +61
        refusals for **-198 `execute` failures**.

        v2 (this) draws the line where it belongs — at DECLARED vs DERIVED.
        A derived `ok` is `coerce`'s guess from the same prose the sniffer
        reads, so it must never overrule the sniffer; that is what this test
        pins, and it is the arm that protects exit-127. A DECLARED `ok` is a
        producer's own answer and does settle it — see
        `test_a_DECLARED_success_settles_it`, and the guard that makes that
        safe: `execute` never declares.
        """
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import ToolOutcome

        def _calls(content):
            msgs = [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c0", "type": "function",
                     "function": {"name": "execute", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c0", "name": "execute",
                 "content": content},
            ]
            return GhostAgent._reconstruct_tool_calls(msgs)

        # the sniffer's evidence must survive a DERIVED ok status — this
        # is the live `execute` path, which returns a bare string
        crashed = ToolOutcome.coerce(
            "stdout:\nxxd: command not found\nEXIT CODE: 127")
        assert crashed.declared is False
        assert _calls(crashed)[0].error, (
            "a command that exited 127 reached the corpus as a SUCCESS — the "
            "status overruled the exit code"
        )
        tb = ToolOutcome.coerce(
            "Traceback (most recent call last):\n  ValueError")
        assert _calls(tb)[0].error

        # ...and the status adds what the sniffer has no rule for
        refusal = ToolOutcome.rejected(
            "SYSTEM INSTRUCTION: The 'path' is missing for 'replace'.")
        assert _calls(refusal)[0].error

        # a genuine success stays a success
        assert not _calls(ToolOutcome.coerce("Wrote 42 bytes."))[0].error

    def test_a_DECLARED_success_settles_it(self):
        """A producer's own "this succeeded" beats the sniffer's guess.

        `manage_projects` returns the project ledger, and a stored
        `autoadvance_failed` event quotes the failing tool's traceback and
        `EXIT CODE: 1` verbatim. The sniffer cannot tell that quote from a
        crash of its own — the producer can, and now says so. Measured: it
        removes the last 5 `manage_projects` rows from the loop-vs-corpus
        disagreement set (219 pre-refactor → 79 → 30).
        """
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import ToolOutcome

        ledger = ('{"project": {"id": "f36f", "events": [{"type": '
                  '"autoadvance_failed", "payload": {"reason": "Traceback '
                  '(most recent call last): ... EXIT CODE: 1"}}]}}')
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "manage_projects", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "manage_projects",
             "content": ToolOutcome.ok(ledger)},
        ]
        assert not GhostAgent._reconstruct_tool_calls(msgs)[0].error, (
            "a successful project READ is labelled a failure because its "
            "JSON body quotes someone else's crash"
        )

    def test_execute_never_declares_which_is_what_makes_that_safe(self):
        """THE GUARD on the rule above.

        Letting a declared `ok` settle the label is only safe because the
        one producer whose body legitimately carries its own failure
        envelope — the shell — never declares. Its results are bare strings,
        so `coerce` marks them DERIVED and the sniffer keeps full authority.
        If `execute` ever starts declaring success, the -198 regression is
        back, and this fails first.
        """
        import ast
        import inspect

        import ghost_agent.tools.execute as E
        from ghost_agent.tools.outcome import ToolOutcome

        assert ToolOutcome.coerce("--- EXECUTION RESULT ---\n"
                                  "EXIT CODE: 1").declared is False

        src = inspect.getsource(E)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            u = ast.unparse(node.func)
            assert not u.endswith("ToolOutcome.ok"), (
                f"execute.py declares a SUCCESS at line {node.lineno} — that "
                "re-opens the -198 regression: a shell result carrying its "
                "own non-zero EXIT CODE banner would settle as a success"
            )


class TestTheShellOverlayReachesEveryCell:
    """The exit-code / crash-marker overlay was applied to some readers and
    not others, and three MAJOR findings lived in that asymmetry."""

    def test_a_crashing_shell_is_not_SUCCEEDED_in_the_banner(self):
        """`op_outcomes.ok` read the exit-code BANNER while the strike
        branch read the crash MARKERS, so a traceback-crash `execute` with
        no banner was listed under SUCCEEDED in the MULTI-STEP OUTCOME line
        — the one message that tells the model "this live outcome is
        AUTHORITATIVE"."""
        from ghost_agent.tools.outcome import ToolOutcome

        # `coerce`, not `ok(...)`: `execute` returns a bare string, so this
        # is the shape the live loop actually builds. The distinction is
        # load-bearing now — see the declared-producer case below.
        banner = ToolOutcome.coerce("xxd: not found\nEXIT CODE: 127")
        markers = ToolOutcome.coerce(
            "Traceback (most recent call last):\nValueError")
        assert banner.exit_code_failed and banner.shell_failed

        # A producer that DECLARES ok cannot talk a shell out of its own
        # exit code: `shell_failed` is asked only of the shell, and there
        # the banner is the evidence, not the claim.
        assert ToolOutcome.ok("xxd: not found\nEXIT CODE: 127").shell_failed, (
            "a declared status overruled a non-zero exit banner on a SHELL "
            "result — the 226-successes fix inverted"
        )
        assert not markers.exit_code_failed, "no banner in this shape"
        assert markers.shell_failed, (
            "a crashed command with no exit banner must still read as a "
            "shell failure for the tools that ask the shell question"
        )

    def test_a_non_shell_tool_is_not_judged_by_prose(self):
        """...and the marker fallback must never reach a tool that merely
        QUOTES code: 226 live successes were booked as incompetence when it
        did — a file read of a script containing `except ValueError:`."""
        from ghost_agent.tools.outcome import ToolOutcome

        read = ToolOutcome.ok("Contents of a.py:\n    except ValueError:\n")
        assert read.shell_failed, "shell_failed is unanchored by design"
        assert not read.exit_code_failed, (
            "the predicate every non-shell reader uses must ignore prose"
        )

    def test_an_exit_code_failure_is_named_not_just_counted(self):
        """`preview` stayed gated on `_res_is_error` alone, so an exit-127
        command was reported to the model as the bare word "failed" — and
        in a batch with two failures the banner is the only place each one
        is named."""
        from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes

        out = summarize_multi_op_outcomes([
            {"tool": "recall", "ok": True, "preview": None},
            {"tool": "execute", "ok": False,
             "preview": "xxd: command not found\nEXIT CODE: 127"},
        ])
        assert "command not found" in out, (
            "the failure was named to the model as the literal word 'failed'"
        )


class TestUnresolvedIsNotAVerdict:
    def test_unresolved_is_not_a_failure(self):
        """"No verdict yet" was being given one: a strike, a guard record
        whose premise is "re-running this unchanged will fail the same way",
        a competence failure and a foresight ok=False — for a detached job
        still running.

        ⚠ This test used to assert `any("UNRESOLVED" in b)` across the
        `_res_is_error` assignments. There are TWO, and the second (the
        work-log rebind, 600 lines later) had no clause — so the exemption
        was undone and the pin stayed green through it. `all`, not `any`.
        """
        import ast
        import inspect

        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

        o = ToolOutcome.unresolved("[sandbox job 3 running in background]")
        assert o.status is OutcomeStatus.UNRESOLVED
        assert o.may_record_as_applied is False
        assert o.changed_the_world is False
        # The predicates that decide a strike must not answer for it either.
        # `shell_failed` returned True on its FIRST line for any non-OK
        # status, and `exit_code_val = 1 if shell_failed` then handed an
        # in-flight run a strike anyway.
        assert o.shell_failed is False
        assert o.exit_code_failed is False
        running = ToolOutcome.unresolved(
            "--- output so far ---\nTraceback (most recent call last):")
        assert running.shell_failed is False, (
            "the `output so far` of a job that is STILL RUNNING was read as "
            "its verdict"
        )

        # EXECUTE the loop's verdict rather than reading it for a token.
        # There is exactly ONE definition now — `_loop_expr` asserts that,
        # because a second one 660 lines away is how seven consumers of a
        # single call came to disagree.
        from tests.test_outcome_consumers_r4 import _loop_expr
        verdict = _loop_expr("_res_is_error")
        assert verdict(o) is False
        assert verdict(running) is False, (
            "the loop gives an unfinished call a failure verdict, and reads "
            "the `output so far` of a job that is STILL RUNNING to do it")
        assert verdict(ToolOutcome.rejected("SYSTEM BLOCK: x")) is True
        assert verdict(ToolOutcome.coerce("SUCCESS: done")) is False

    def test_a_quoted_traceback_does_not_fail_a_declared_success(self):
        """The `Traceback` arm is an UNANCHORED whole-body substring — the
        same shape as the marker fallback that booked 226 successes as
        incompetence. It survived the banner fix one arm over: 2 live
        `manage_projects` reads were still booked failed because the project
        ledger quotes a stored crash. A producer that DECLARED its status
        has already answered."""
        import ast
        import inspect

        from ghost_agent.core.agent import GhostAgent

        from ghost_agent.tools.outcome import ToolOutcome
        from tests.test_outcome_consumers_r4 import _loop_expr

        verdict = _loop_expr("_res_is_error")
        ledger = ('{"project": {"events": [{"type": "autoadvance_failed", '
                  '"payload": {"reason": "Traceback (most recent call last)"'
                  "}}]}}")
        # The rule is now stronger than "a DECLARED success survives it":
        # there is NO unanchored whole-body traceback arm in the verdict at
        # all. It lived in the old wide rebind, where it fed only the work
        # log; collapsing the three verdicts into one promoted it to the
        # STRIKE LEDGER, and a `file_system` read of a log quoting a
        # traceback drew a strike, failed the competence profile and
        # injected an AUTO-DIAGNOSTIC. 4.2% of readable files in the live
        # sandbox contain the word.
        assert verdict(ToolOutcome.ok(ledger)) is False
        assert verdict(ToolOutcome.coerce(ledger)) is False, (
            "the unanchored traceback substring is back in the verdict — "
            "that is the 226-successes-as-incompetence family")
        assert verdict(ToolOutcome.coerce(
            "--- a.py CONTENTS ---\nTraceback (most recent call last):"),
            fname="file_system") is False
        # ...and the SHELL keeps the prose fallback, via `shell_failed`
        assert verdict(ToolOutcome.coerce(
            "Traceback (most recent call last):"), fname="execute") is True
        # ...and the competence profile must read the SAME verdict, not a
        # private copy of it
        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        tf = [ast.unparse(n.value) for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and getattr(n.targets[0], "id", None) == "_tool_failed"]
        assert tf == ["_res_is_error"], (
            f"the competence profile keeps its own reading of failure: {tf}")

    def test_unresolved_has_a_real_producer(self):
        """A status no producer can mint is decoration: it makes the loop
        look more careful than it is. `execute`'s promoted-job result IS
        this case — a command detached at its budget, still running, whose
        text is deliberately success-shaped so the downstream verify gates
        that key on `EXIT CODE: 0` keep working."""
        from ghost_agent.tools.execute import _promoted_result
        from ghost_agent.tools.outcome import OutcomeStatus

        r = _promoted_result({"id": "job-1"}, 90, "half a line of stdout")
        assert r.status is OutcomeStatus.UNRESOLVED
        # ...and the TEXT is unchanged, which is what keeps every
        # banner-reading consumer downstream working.
        assert "EXIT CODE: 0" in r
        assert "NOT finished" in r


class TestARefusalDoesNotRewriteThePlan:
    """Four identical refusals rewrote the whole task tree because the model
    had forgotten one argument. Two rounds of fixes made it worse before
    this one: round 1 exempted refusals from the pre-flight guard, round 2
    exempted them from the same-failure breaker, and with both brakes off
    the System-3 pivot fired every other turn for ever."""

    def _live_pivot_arithmetic(self):
        """Lift the trigger, the decay and the cap out of the LIVE source.

        Executed, not asserted-about: a text pin over this arithmetic stays
        green while the loop it describes never terminates.
        """
        import ast
        import inspect

        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        trigger = decay = cap = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None) == "sys3_trigger"):
                trigger = ast.unparse(node.value)
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None)
                    == "execution_failure_count"):
                u = ast.unparse(node.value)
                if "- 2" in u:
                    decay = u
            if isinstance(node, ast.If):
                u = ast.unparse(node.test)
                if "execution_failure_count >= 6" in u:
                    cap = u
        # ...and the CONDITION the decay sits under. Hardcoding
        # `if pivot_num == 1` in the simulation instead of reading it was
        # this pin's own vacuity: it modelled the fixed loop no matter what
        # the source said, and survived a mutant that removed the guard.
        parent = {}
        for n in ast.walk(tree):
            for c in ast.iter_child_nodes(n):
                parent[c] = n
        decay_node = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None)
                    == "execution_failure_count"
                    and "- 2" in ast.unparse(node.value)):
                decay_node = node
        guards = []
        cur = decay_node
        while cur is not None and cur in parent:
            cur = parent[cur]
            if isinstance(cur, ast.If) and "pivot_num" in ast.unparse(cur.test):
                guards.append(ast.unparse(cur.test))
        # ...and the INCREMENT. Hardcoding `fired += 1` in the simulation
        # was this pin's second vacuity: replacing the counter with a bare
        # `True` (which is what it used to be) passed 101/101 here and
        # 316/317 across nine suites, while the real loop fired 55 pivots in
        # 60 turns and never reached the cap. `True == 1` satisfies the
        # second trigger arm for ever.
        bump = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None)
                    == "_request_sys3_fired_once"):
                bump = ast.unparse(node.value)
        assert bump is not None, "the pivot counter update moved"
        assert trigger and decay and cap, (trigger, decay, cap)
        return trigger, decay, cap, (guards[0] if guards else None), bump

    def test_the_pivot_cannot_outrun_the_strike_cap(self):
        """The trigger fires at 4 (first) or 5 (second) and the pivot then
        subtracts 2 — so after the second pivot the counter falls to 3,
        climbs back to 5 and fires again, for ever. Executed against the
        live arithmetic: 11 pivots in 25 all-failing turns, the task tree
        rewritten every other turn, `execution_failure_count >= 6` NEVER
        reached and no final response forced."""
        trigger, decay, cap, decay_guard, bump = self._live_pivot_arithmetic()

        efc = tfc = pivots = fired = 0
        capped_at = None
        # The pre-dispatch arms each `+= 1` inside the per-call loop, so a
        # turn can step the counter by more than one. Drive the simulation
        # with a STEPPING sequence, not +1 every turn: with equality
        # triggers, [1,1,1,2,...] walks straight past both.
        steps = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1] * 4
        for turn in range(1, 41):
            efc += steps[turn - 1]
            ns = {"execution_failure_count": efc,
                  "_request_sys3_fired_once": fired,
                  "transient_failure_count": tfc}
            if eval(trigger, {}, ns):
                pivots += 1
                ns["pivot_num"] = 2 if fired else 1
                # the loop's OWN counter update, not the test's idea of it
                fired = eval(bump, {}, ns)
                ns["_request_sys3_fired_once"] = fired
                if decay_guard is None or eval(decay_guard, {}, ns):
                    efc = eval(decay, {}, ns)
                tfc = 0
            if eval(cap, {}, {"execution_failure_count": efc,
                              "transient_failure_count": tfc,
                              "total_fail": efc + tfc}):
                capped_at = turn
                break

        # ...and the pivot must be bounded even when the pivot ITSELF fails.
        # `_run_system_3_pivot` returns {} on any exception (a 120 s LLM
        # timeout counts), so if the counter only advanced on success the
        # `>=` trigger re-armed every turn: 3 pivot calls instead of 1, on
        # exactly the path where the pivot is broken.
        efc2 = fired2 = pivots2 = 0
        for turn in range(1, 41):
            efc2 += steps[turn - 1]
            ns2 = {"execution_failure_count": efc2,
                   "_request_sys3_fired_once": fired2,
                   "transient_failure_count": 0}
            if eval(trigger, {}, ns2):
                pivots2 += 1
                ns2["pivot_num"] = 2 if fired2 else 1
                fired2 = eval(bump, {}, ns2)   # NO tree_update: nothing else runs
            if eval(cap, {}, {"execution_failure_count": efc2,
                              "transient_failure_count": 0,
                              "total_fail": efc2}):
                break
        assert pivots2 <= 2, (
            f"a FAILING System-3 pivot was called {pivots2} times in one "
            "request — the counter only advances when the pivot succeeds, so "
            "the trigger re-arms on exactly the path where it is broken"
        )

        assert pivots >= 1, (
            "the pivot never fires at all: the counter STEPS (a turn can add "
            "more than one strike), so equality triggers walk past it"
        )
        assert pivots <= 2, (
            f"the System-3 pivot fired {pivots} times in one request — it "
            "re-arms itself because the decay puts the counter back below "
            "its own trigger"
        )
        # The decay guard is no longer what BOUNDS the pivots — the pivot
        # COUNT does that — but it still decides how fast an all-failing
        # request reaches the cap: 7 turns with it, 8 without. Without this
        # bound the guard is unpinned and reads as defence while doing
        # nothing measurable.
        assert capped_at is not None and capped_at <= 7, (
            f"an all-failing request took {capped_at} turns to reach the "
            "6-strike cap; pivot #2 is decaying the counter it should leave "
            "alone"
        )
        assert capped_at is not None, (
            "an all-failing request never reaches the 6-strike cap: the "
            "pivot decays the counter faster than it climbs, so the loop "
            "rewrites the task tree for ever instead of answering"
        )

    def test_a_repeated_refusal_still_reaches_the_repeat_detector(self):
        """`note_failure` IS the brake — it counts identical failures and
        freezes the success-decay so the cap can fire. Round 2 skipped the
        call for refusals, which removed the last brake they had."""
        import ast
        import inspect

        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())
        parent = {}
        for n in ast.walk(tree):
            for c in ast.iter_child_nodes(n):
                parent[c] = n

        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and ast.unparse(n.func).endswith("strikes.note_failure")]
        assert calls, "`strikes.note_failure` is gone from the dispatch loop"

        # Every one of them must be UNCONDITIONAL with respect to whether
        # the failure was a refusal. Asking "does the rejection branch
        # mention note_failure" was vacuous: the bypass branch assigns the
        # tuple by hand and never names the function, so the check passed
        # on exactly the code it was written to catch.
        for call in calls:
            cur = call
            while cur in parent:
                cur = parent[cur]
                if isinstance(cur, ast.If):
                    t = ast.unparse(cur.test)
                    assert ("is_rejection" not in t
                            and "failure_was_rejection" not in t), (
                        "`strikes.note_failure` sits under a refusal test "
                        f"again ({t!r}) — that is the repeat detector AND "
                        "the decay freeze, and with it bypassed the "
                        "System-3 pivot re-arms for ever"
                    )

    def test_the_refusal_steer_says_correct_it_not_abandon_it(self):
        """The generic steer reads "STOP repeating it — retrying will not
        change the result", against a message whose entire content is what
        to change."""
        import inspect

        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        assert "re-issue the same tool" in src.lower(), (
            "no corrective steer for a repeated refusal"
        )
        # and it must be selected by the flag captured at the FAILING call
        assert "failure_was_rejection" in src

    def test_the_refusal_flag_is_captured_at_the_failing_call(self):
        """The strike block runs AFTER `for i, result in enumerate(results)`,
        so `_outcome` there is the batch's LAST result. Reading it made the
        refusal branch last-result-wins: a refusal followed by a successful
        call took the hard-failure path, and a hard failure followed by an
        unrelated refusal was exempted from the breaker AND the decay
        freeze. Same two calls, opposite behaviour, decided by batch order.
        """
        import ast
        import inspect

        from ghost_agent.core.agent import GhostAgent

        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        tree = ast.parse(src.lstrip())

        loops = [n for n in ast.walk(tree)
                 if isinstance(n, ast.For)
                 and "enumerate(results)" in ast.unparse(n.iter)]
        assert loops, "could not find the per-result loop"
        inside = []          # a LIST: the two sites unparse identically,
        for lp in loops:     # and a set collapsed them to one
            for n in ast.walk(lp):
                if (isinstance(n, ast.Assign) and len(n.targets) == 1
                        and getattr(n.targets[0], "id", None)
                        == "failure_was_rejection"):
                    inside.append(ast.unparse(n))
        assert inside, (
            "`failure_was_rejection` is never assigned inside the per-result "
            "loop — whatever the strike block reads is the LAST call's"
        )
        assert any("_outcome.is_rejection" in a for a in inside), (
            "the flag is never set from the failing result's own outcome"
        )
        # BOTH failing sites — the exit-code branch and the general one.
        # Requiring only "at least one" let a mutant delete the general
        # branch's assignment and live: the execute branch still had one.
        assert len([a for a in inside
                    if "_outcome.is_rejection" in a]) >= 2, (
            "only one of the two failure branches derives the refusal flag; "
            "the other leaves it stale from an earlier result in the batch: "
            + repr(sorted(inside)))
