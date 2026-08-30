"""Two guards the knowledge_base parameter fix depends on, pinned.

Both were shipped unpinned and survived a mutation run that killed 22 of 24
mutants — which is the whole reason they are here. A fix nothing can fail on
is documentation.

1. `call_runs_a_memory_wipe` — sets `forget_was_called`, which suppresses
   smart_memory / post_mortem / the episode write for the turn. Its previous
   form (`args.get("action") == "forget"` over a `strict=True` parse) missed
   wipes that then RAN, re-learning the content the user had just asked to
   delete, inside the same request.

2. `format_failure_context` on an ARGUMENT error — the missing-parameter
   message now names an accepted parameter and carries a worked call, so the
   recovery is to re-issue the call corrected. The FATAL branch used to
   append "PERMANENT ERROR — do NOT retry this tool call" to it, telling the
   model not to do the one thing that works. That advice was accidentally
   right while such errors named parameters the tool would reject; it is
   exactly wrong now.
"""

import asyncio
import json

import pytest

from ghost_agent.core.agent import (
    _FORGET_ACTION_RE,
    GhostAgent,
    call_runs_a_memory_wipe,
)
from ghost_agent.tools import memory as mem
from ghost_agent.tools.tool_failure import (
    FailureClass,
    classify_tool_failure,
    format_failure_context,
    should_retry,
)


# ------------------------------------------------ 1. the wipe predicate

@pytest.mark.parametrize("raw", [
    '{"action": "forget", "target": "atlas"}',
    '{"action": "Forget", "target": "atlas"}',          # models capitalise
    '{"action": " forget ", "target": "atlas"}',        # XML parser strips only CR/LF
    '{"action": "FORGET"}',
    '{"target": "atlas", "action": "forget"}',
])
def test_a_wipe_is_detected_however_the_model_spelled_it(raw):
    assert call_runs_a_memory_wipe("knowledge_base", raw) is True, (
        "the dispatcher lowercases and strips this action, so a wipe runs; "
        "missing it re-learns the forgotten content in the same turn"
    )


def test_a_wipe_is_detected_when_the_arguments_do_not_parse():
    """`json.loads` is strict about raw control characters. The same
    arguments parse fine at the dispatch site 100 lines later, so a parse
    failure here meant the flag was False while the wipe ran."""
    raw = '{"action": "forget", "note": "line1\x01line2"}'
    with pytest.raises(ValueError):
        json.loads(raw)  # strict=True — this is the failure being backstopped
    assert call_runs_a_memory_wipe("knowledge_base", raw) is True


# The predicate has two mechanisms — a healed JSON parse and a regex over the
# raw text — and for ordinary inputs EITHER ONE alone gives the right answer.
# A mutation run proved that: deleting either mechanism left every test above
# green, because none of them could tell the two apart. These two can. Keep
# them adjacent to any future change to either half.

def test_the_parse_carries_cases_the_regex_cannot_see():
    """JSON escapes and in-string whitespace: the raw text does not contain
    the literal token, so only the parse (and its strip/lower) finds it."""
    for raw in ('{"action": "for\\u0067et", "target": "atlas"}',
                '{"action": "\\tforget", "target": "atlas"}',
                '{"action": "FORG\\u0045T"}'):
        assert not _FORGET_ACTION_RE.search(raw), (
            f"{raw}: the regex CAN see this, so it does not separate the two "
            f"mechanisms — pick an input only the parse recovers"
        )
        assert call_runs_a_memory_wipe("knowledge_base", raw) is True, (
            f"{raw}: the dispatcher strips and lowercases this into a wipe"
        )


def test_the_regex_carries_cases_the_parse_cannot_see():
    """Truncated or trailing-garbage arguments — models emit both. The
    dispatch site recovers them; if this flag does not, the wipe runs
    unrecorded."""
    for raw in ('{"action": "forget", "target": "atlas"',      # unterminated
                '{"action": "forget"} trailing junk',
                '{"action": "forget", }'):                     # trailing comma
        with pytest.raises(ValueError):
            json.loads(raw)
        assert call_runs_a_memory_wipe("knowledge_base", raw) is True


@pytest.mark.parametrize("spelling", [
    "knowledge_base", "kb", "knowledgebase", "knowledge-base",
    "Knowledge_Base", "KNOWLEDGE_BASE", "knowlegebase",
    # Measured through the real batch dispatcher: every one of these routes
    # to knowledge_base, and every one of them left the flag False while the
    # wipe ran, back when this predicate used its own stricter matcher.
    "knowledge", "knowledge_db", "knowledgedb", "knowledge_bank",
    "knowledge_store", "knowledge_base_query", "know_base",
])
def test_a_wipe_is_detected_however_the_model_spelled_the_TOOL(spelling):
    """The first version healed the ACTION and left the TOOL NAME raw — and
    this predicate runs ~314 lines BEFORE `_canonicalise_tool_name`, whose
    alias table maps kb / knowledgebase / knowledge-base onto knowledge_base
    (its comment cites 'knowledgebase' as an observed Qwen 3.5
    hallucination). Measured through handle_chat with identical arguments,
    only the spelling varying: the wipe ran in all five, the flag was False
    in four, and each of those turns queued a smart_memory item carrying the
    request text and the wipe report back into the store just cleared."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    available = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    raw = '{"action": "forget", "target": "mortimer"}'
    assert call_runs_a_memory_wipe(spelling, raw, available=available) is True


def test_the_predicate_and_the_dispatcher_resolve_names_identically():
    """The general fix, of which the cutoff was the instance.

    This predicate ran a SECOND name matcher — the alias table plus difflib
    at cutoff 0.85 — while the dispatcher canonicalises at 0.70 against the
    whole tool list, 314 lines later. `knowledge` (ratio 0.818) and
    `know_base` therefore healed at dispatch and not here: the wipe RAN with
    the flag False. Two implementations of one decision drift by
    construction, so the predicate now asks the dispatcher's own question.
    This asserts they answer the same, over every advertised tool name and
    every mangling a model produces."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS

    available = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    args = '{"action": "forget", "target": "atlas"}'
    probes = list(available) + [
        "kb", "knowledgebase", "knowledge-base", "Knowledge_Base",
        "knowledge", "know_base", "knowlegebase", "KNOWLEDGE_BASE",
        "fs", "filesystem", "websearch", "vision", "nonsense_tool", "",
    ]
    # ⚠ The expectation is computed from what the DISPATCH would run, not
    # from the predicate's own resolver. An earlier version derived it with
    # `_canonicalise_tool_name` — the very call the predicate makes
    # internally — so it could only fail where the None-fallback diverged.
    wipe_names = {"knowledge_base", "forget"}
    for name in probes:
        dispatched = name if name in available else (
            GhostAgent._canonicalise_tool_name(name, available))
        expected = dispatched in wipe_names or name == "forget"
        assert call_runs_a_memory_wipe(name, args, available=available) is expected, (
            f"{name!r}: the dispatch would run {dispatched!r} but the wipe "
            f"predicate disagrees — the wipe would run unrecorded"
        )


def test_the_predicate_still_works_without_the_tool_list():
    """The `available` argument is optional; the alias-table fallback must
    still cover the spellings actually observed live."""
    args = '{"action": "forget", "target": "atlas"}'
    for name in ("knowledge_base", "kb", "knowledgebase", "knowledge-base"):
        assert call_runs_a_memory_wipe(name, args) is True
    for name in ("file_system", "execute", "recall"):
        assert call_runs_a_memory_wipe(name, args) is False


def test_the_regex_backstop_is_case_insensitive():
    """Models capitalise. The only path that can answer for arguments that
    do not parse is the regex, and nothing pinned its IGNORECASE flag —
    every other regex-path test used a lowercase action."""
    raw = '{"action": "Forget", "target": "atlas"'      # capitalised AND unterminated
    with pytest.raises(ValueError):
        json.loads(raw)
    assert call_runs_a_memory_wipe("knowledge_base", raw) is True
    assert call_runs_a_memory_wipe("knowledge_base",
                                   '{"action": "RESET_ALL"') is True


@pytest.mark.parametrize("other", ["file_system", "fs", "web_search",
                                   "execute", "update_profile", "recall"])
def test_tool_name_healing_does_not_swallow_other_tools(other):
    """Name healing must not turn every near-ish name into a wipe."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    available = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert call_runs_a_memory_wipe(other, '{"action": "forget"}') is False
    assert call_runs_a_memory_wipe(
        other, '{"action": "forget"}', available=available) is False


def test_reset_all_is_a_wipe_too():
    """`reset_all` deletes every id in the vector collection, resets the
    library index to [] and wipes the graph — a strictly LARGER wipe than
    forget — and returns a SUCCESS string, so nothing else stops the turn's
    smart_memory / post_mortem / episode write repopulating what it just
    emptied. A predicate named for wipes that answers False here is a wrong
    answer in the direction its own docstring calls unrecoverable."""
    for raw in ('{"action": "reset_all"}',
                '{"action": " Reset_All "}',
                '{"action": "reset_all"'):          # unparseable -> regex path
        assert call_runs_a_memory_wipe("knowledge_base", raw) is True
    assert call_runs_a_memory_wipe("knowledge_base", '{"action": "list_docs"}') is False


def test_each_wipe_mechanism_is_load_bearing_for_reset_all_too():
    """`reset_all` was added to `_WIPE_ACTIONS` AND to the regex, so deleting
    either left every existing test green — the same mutual-redundancy trap
    that hid the `forget` half. These separate them."""
    # Only the parse can recover an escaped action name.
    escaped = '{"action": "reset_a\\u006cl"}'
    assert not _FORGET_ACTION_RE.search(escaped)
    assert call_runs_a_memory_wipe("knowledge_base", escaped) is True
    # Only the regex can recover an action from arguments that do not parse.
    broken = '{"action": "reset_all", "confirm": true'
    with pytest.raises(ValueError):
        json.loads(broken)
    assert call_runs_a_memory_wipe("knowledge_base", broken) is True


def test_the_lenient_parse_is_load_bearing_on_its_own():
    """`strict=False` and the regex covered each other for every earlier
    input. This one needs the lenient parse: a raw control character (which
    strict json rejects) together with an escaped action the regex cannot
    see."""
    raw = '{"action": "for\\u0067et", "note": "a\x01b"}'
    assert not _FORGET_ACTION_RE.search(raw)
    import json as _json
    with pytest.raises(ValueError):
        _json.loads(raw)                    # strict=True rejects the \x01
    assert _json.loads(raw, strict=False)["action"] == "forget"
    assert call_runs_a_memory_wipe("knowledge_base", raw) is True


def test_the_turn_loop_consults_the_HEALED_tool_name():
    """Three decisions in one block are made from the tool name, and
    `_canonicalise_tool_name` runs ~150 lines below all of them. When they
    read `fname` raw:

      * the wipe flag stayed False while the wipe ran (tombstone
        resurrection), and
      * `is_mutating` / `is_idempotent_setter` classified two identical
        `knowledgebase` ingests as read-safe, so the second was
        dedup-collapsed and a REAL INGEST WAS DROPPED — the defect
        `is_mutating`'s own comment says was fixed, reached through the
        other half of the identity.

    Parsed, not grepped: every one of the three must consult the resolved
    name (or be handed the tool list so it can resolve it itself)."""
    import ast
    import inspect
    from ghost_agent.core.agent import GhostAgent

    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    tree = ast.parse(src.lstrip())

    assigns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            assigns.setdefault(node.targets[0].id, []).append(
                ast.dump(node.value))

    for var in ("is_mutating", "is_idempotent_setter"):
        assert var in assigns, f"{var} was renamed or moved"
        body = " ".join(assigns[var])
        assert "'_cname'" in body, f"{var} no longer reads the healed name"
        # ABSENCE, not presence. These are multi-clause boolean expressions:
        # checking only that `_cname` appears somewhere left a mutant alive
        # that reverted ONE clause to `fname` — the other clauses kept the
        # assertion true while the reverted branch misclassified.
        assert "'fname'" not in body, (
            f"{var} still reads the RAW tool name in at least one clause; a "
            f"model that emits 'kb' or 'knowledgebase' then gets a different "
            f"classification than the one the dispatcher acts on, and two "
            f"identical ingests collapse into one"
        )

    # ...and the wipe predicate is handed the list it needs to resolve with.
    wipe_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "call_runs_a_memory_wipe"
    ]
    assert wipe_calls, "the wipe predicate call site moved"
    for call in wipe_calls:
        kwargs = {k.arg: ast.dump(k.value) for k in call.keywords}
        assert "available" in kwargs, (
            "call_runs_a_memory_wipe was called without the tool list, so it "
            "falls back to the alias table and stops agreeing with the "
            "dispatcher on the names only difflib heals"
        )
        assert "None" not in kwargs["available"] or "available_tools" in kwargs["available"]


def test_the_classified_and_formatted_text_are_the_same_expression():
    """The failure class and the diagnostic the model reads must describe
    the same string. Structural, so it cannot drift: one local, two uses.
    Parsed rather than grepped."""
    import ast
    import inspect
    from ghost_agent.core.agent import GhostAgent

    tree = ast.parse(inspect.getsource(
        GhostAgent._dispatch_and_process_tool_batch).lstrip())
    args = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("classify_tool_failure",
                                     "format_failure_context") and node.args:
            args.setdefault(node.func.id, set()).add(
                ast.dump(node.args[0]))
    assert args.get("classify_tool_failure"), "classify call site moved"
    assert args.get("format_failure_context"), "format call site moved"
    assert args["classify_tool_failure"] == args["format_failure_context"], (
        "the class is derived from one string and the diagnostic rendered "
        "from another; a class earned by a token past the preview cut is "
        "then formatted against text that no longer contains it"
    )
    assert len(args["format_failure_context"]) == 1, (
        "more than one expression is formatted; they must share the local"
    )


def test_the_standalone_forget_tool_still_counts():
    assert call_runs_a_memory_wipe("forget", "{}") is True
    assert call_runs_a_memory_wipe("forget", "not json") is True


@pytest.mark.parametrize("fname,raw", [
    ("knowledge_base", '{"action": "query", "filename": "d"}'),
    ("knowledge_base", '{"action": "insert_fact", "fact": "f"}'),
    ("knowledge_base", "{}"),
    ("knowledge_base", "total gibberish"),
    ("file_system", '{"action": "forget"}'),      # different tool entirely
    ("execute", '{"command": "echo forget"}'),
])
def test_a_non_wipe_is_not_flagged(fname, raw):
    assert call_runs_a_memory_wipe(fname, raw) is False


def test_the_predicate_agrees_with_the_dispatcher_on_every_spelling():
    """The two must not drift: whatever the dispatcher treats as `forget`,
    this predicate must call a wipe. Executed against the real dispatcher,
    not asserted about it."""
    reached = []

    async def _forget(target, *a, **k):
        reached.append(target)
        return "OK"

    real = mem.tool_unified_forget
    mem.tool_unified_forget = _forget
    try:
        for spelling in ("forget", "Forget", " forget ", "FORGET",
                         "query", "list_docs"):
            reached.clear()
            asyncio.run(mem.tool_knowledge_base(action=spelling, target="atlas"))
            dispatcher_wiped = bool(reached)
            raw = json.dumps({"action": spelling, "target": "atlas"})
            assert call_runs_a_memory_wipe("knowledge_base", raw) is dispatcher_wiped, (
                f"{spelling!r}: dispatcher wiped={dispatcher_wiped} but the "
                f"turn loop's flag disagrees"
            )
    finally:
        mem.tool_unified_forget = real


# ------------------------------------- 2. what the model is told to do next

def _context_for(action):
    err = asyncio.run(mem.tool_knowledge_base(action=action))
    cls, _ = classify_tool_failure(err)
    return err, cls, format_failure_context(err, cls, "knowledge_base")


@pytest.mark.parametrize("action", ["forget", "insert_fact", "ingest_document"])
def test_an_argument_error_permits_the_corrected_reissue(action):
    err, cls, ctx = _context_for(action)
    assert cls is FailureClass.FATAL          # must not spend the retry budget
    assert should_retry(cls, 0) is False
    assert "do NOT retry this tool call" not in ctx, (
        "the message tells the model to pass the named argument; a blanket "
        "stop signal is then the only thing standing between it and the fix"
    )
    assert "re-issue the SAME tool" in ctx
    assert "Do NOT repeat it unchanged" in ctx   # still no identical retry
    assert err[:80] in ctx                       # the advice itself survives


def test_a_transient_failure_still_reads_as_a_retry():
    """The argument-error branch runs before the class dispatch, so it must
    exclude RETRYABLE — a message that is transient AND mentions a required
    parameter is still going to be retried by the system, and telling the
    model to re-issue it itself duplicates the call."""
    text = ("Connection timed out talking to the node (the 'target' "
            "parameter is MANDATORY once it is back)")
    cls, _ = classify_tool_failure(text)
    assert cls is FailureClass.RETRYABLE, "picked a text that isn't transient"
    ctx = format_failure_context(text, cls, "knowledge_base")
    assert "TRANSIENT ERROR (will retry)" in ctx
    assert "ARGUMENT ERROR" not in ctx


def test_a_genuine_permanent_error_still_says_stop():
    """The split must not swallow the real stop signal."""
    text = "PermissionError: [Errno 13] Permission denied: '/etc/shadow'"
    cls, _ = classify_tool_failure(text)
    assert cls is FailureClass.FATAL
    ctx = format_failure_context(text, cls, "execute")
    assert "do NOT retry this tool call" in ctx
    assert "re-issue the SAME tool" not in ctx


def test_an_argument_error_the_classifier_cannot_place_still_gets_it():
    """The split covers UNKNOWN, not just FATAL. Only `MANDATORY` is a FATAL
    pattern, so restricting the branch to FATAL left the one live message it
    was written for — `update_profile`'s "'key' is a required argument",
    where the knowledge_base redirect sends a model — with the bare
    "ERROR:" framing and no instruction to re-issue."""
    text = "Error: 'key' is a required argument for update_profile."
    cls, _ = classify_tool_failure(text)
    assert cls is FailureClass.UNKNOWN, "picked a text the classifier places"
    ctx = format_failure_context(text, cls, "update_profile")
    assert "re-issue the SAME tool" in ctx


def test_a_diagnostic_keeps_its_budget_and_framing():
    """DIAGNOSTIC exists to hand the model a full traceback to work from. A
    traceback that happens to say "required parameter" must not be traded
    for a 500-char argument nudge."""
    text = ("Traceback (most recent call last):\n"
            + "\n".join(f"  frame {i} ..." for i in range(200))
            + "\nTypeError: missing required parameter 'x'")
    ctx = format_failure_context(text, FailureClass.DIAGNOSTIC, "execute")
    assert "DIAGNOSTIC ERROR — analyze and fix" in ctx
    assert "ARGUMENT ERROR" not in ctx
    assert len(ctx) > 600, "the 2000-char diagnostic budget was cut to 500"


def test_a_transient_token_outranks_a_fatal_one():
    """`test_the_real_message_classifies_as_fatal` reasons from the scan
    ORDER — retryable patterns before fatal ones. Swapping the two loops
    left every test in this repo green while flipping real classifications
    (`503 Service Unavailable: invalid parameter` RETRYABLE -> FATAL), and
    it silently disarms the FATAL pin above. Pin the order by its effect."""
    both = "Connection timed out; also the 'target' parameter is MANDATORY"
    cls, _ = classify_tool_failure(both)
    assert cls is FailureClass.RETRYABLE, (
        "a message carrying both a transient and a fatal marker must read "
        "as transient — the fatal pin above is only meaningful because the "
        "real messages carry no transient marker"
    )


def test_the_unknown_action_error_gets_the_argument_treatment_too():
    err = asyncio.run(mem.tool_knowledge_base(action="delete"))
    cls, _ = classify_tool_failure(err)
    ctx = format_failure_context(err, cls, "knowledge_base")
    assert cls is FailureClass.FATAL
    assert "re-issue the SAME tool" in ctx, (
        "the error lists the valid actions — the model must be allowed to "
        "pick one"
    )


# ------------------------------- the invocation error (all tools, not just kb)

def test_a_duplicate_kwarg_error_says_REMOVE_not_add():
    """Model arguments are splatted unfiltered into a dispatch lambda that
    already supplies the tool's context, so a model passing one of those
    names gets `got multiple values for keyword argument 'model_name'` —
    and was then told "(Did you forget a required argument?)", i.e. to ADD
    an argument when the fix is to REMOVE one. Same class as the forget
    loop: advice that cannot work."""
    from ghost_agent.core.agent import describe_invocation_error

    exc = TypeError(
        "tool_knowledge_base() got multiple values for keyword argument "
        "'model_name'")
    msg = describe_invocation_error("knowledge_base", exc)
    # Check the PROSE, not the appended TypeError text — that already
    # contains 'model_name', so a message whose own wording said only "an
    # argument is MANDATORY to REMOVE" passed the naive assertion.
    prose = msg.split(": tool_knowledge_base()")[0]
    assert "'model_name'" in prose, (
        f"the message does not name the offending argument; the model has "
        f"to work out which one to drop: {prose!r}"
    )
    assert "REMOVE" in prose
    assert "Did you forget a required argument" not in msg

    cls, _ = classify_tool_failure(msg)
    assert cls is FailureClass.FATAL
    ctx = format_failure_context(msg, cls, "knowledge_base")
    assert "re-issue the SAME tool" in ctx


def test_an_unexpected_kwarg_error_points_at_the_schema():
    from ghost_agent.core.agent import describe_invocation_error

    msg = describe_invocation_error(
        "recall", TypeError("tool_recall() got an unexpected keyword "
                            "argument 'foo'"))
    assert "does not accept" in msg
    assert format_failure_context(
        msg, classify_tool_failure(msg)[0], "recall").count("re-issue") >= 1


def test_an_unrelated_invocation_failure_keeps_the_old_wording():
    """The rewrite must not swallow every invocation error into an argument
    diagnosis."""
    from ghost_agent.core.agent import describe_invocation_error

    msg = describe_invocation_error("execute", ValueError("kernel exploded"))
    assert "kernel exploded" in msg
    assert "REMOVE" not in msg


def test_an_argument_error_keeps_its_worked_call_in_a_mixed_turn():
    """`op_outcomes` previews a failure at 140 chars, and the missing-parameter
    messages carry their worked call at the END — so in any turn that mixed a
    success with a failure the model read "…filename to erase. Worked " and
    nothing more. Argument errors get enough budget to survive; everything
    else keeps the old one."""
    import asyncio

    from ghost_agent.tools import memory as _mem
    from ghost_agent.tools.tool_failure import is_argument_error

    err = asyncio.run(_mem.tool_knowledge_base(action="forget"))
    assert is_argument_error(err)
    assert not is_argument_error("Permission denied: /etc/shadow")

    # Read the budget out of the CODE, not out of this test. The first
    # version computed `300 if is_argument_error(err) else 140` here and
    # asserted against its own arithmetic — a mutant that cut the budget
    # back to a flat 140, and one that widened every preview to 300, both
    # passed it. The structural half lives in
    # tests/test_turn_loop_name_resolution.py; this half checks the message
    # actually fits.
    from tests.test_turn_loop_name_resolution import _budget_for_argument_errors

    # The WHOLE message must fit, not just the start of the worked call.
    # "Worked call:" begins at index 133 of a 209-char message, so any
    # budget >= 145 satisfied the naive form — including a mutant at 160,
    # which leaves the model reading "…Worked call: knowledge_ba".
    budget = _budget_for_argument_errors()
    assert len(err) <= budget, (
        f"the argument error is {len(err)} chars and the mixed-turn preview "
        f"budget is {budget}; the worked call at the end is cut off"
    )


def test_the_canonicaliser_normalises_a_name_that_is_not_exact():
    """The identity short-circuit added at the top shadowed the pin for the
    tier below it — every probe now returns before reaching the normalised
    exact-match. This case needs that tier: a name that differs from the
    real tool only in case, with no alias entry to fall back on."""
    assert GhostAgent._canonicalise_tool_name(
        "fs", ["file_system", "web_search", "FS"]) == "FS"
    assert GhostAgent._canonicalise_tool_name(
        "Web_Search", ["web_search"]) == "web_search"


def test_the_canonicaliser_never_returns_an_unavailable_tool():
    """The invariant the disabled-tools gate's whole rationale rests on —
    "canonicalisation only returns available tools, and containment strips
    disabled names from that map". Dropping the `mapped in available` check
    was invisible to every test."""
    for name in ("kb", "fs", "knowledgebase", "vision", "profile_update"):
        got = GhostAgent._canonicalise_tool_name(name, ["recall"])
        assert got in (None, "recall"), (
            f"{name!r} resolved to {got!r}, which is not an available tool — "
            f"a disabled or absent tool can be reached through an alias"
        )
