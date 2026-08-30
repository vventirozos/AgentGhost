"""A knowledge_base error may only name a parameter the tool accepts.

Live failure, 2026-08-28. The user asked the agent to forget a topic. The
model called ``knowledge_base(action='forget')`` — because the schema
advertised no parameter that reads as "the thing to forget"; the only legal
slot was ``filename``, whose description is about ``.mp4``/``.pdf`` ingestion
with "REQUIRED for forget (the topic name to forget)" tacked on the end. The
model's own reasoning in the log: *"The parameters are: action, fact,
question, ref, filename. None is obviously 'topic'."*

The dispatcher folded ``filename|fact|content|source|path|topic`` into a local
named ``target``, found none, and surfaced the INNER tool's guard verbatim:
"SYSTEM ERROR: The 'target' parameter is MANDATORY." But ``target`` was
neither in the schema nor in that alias list. The model complied exactly,
passed ``target='…'``, the dispatcher dropped it into ``**kwargs``, and the
identical error came back. Obeying the error could never work: every retry
was byte-identical until the strike budget ran out. ``insert_fact`` had the
same trap through a different name (``'text'``).

WHAT THESE PINS ASSERT, AND HOW THEY EARN IT
--------------------------------------------
The invariant is: *every identifier a knowledge_base error puts in front of
a model is one the dispatcher will accept.*

The first version of this file tested that by scraping quoted lowercase
tokens out of the message with a regex. Review killed it three ways, all
while 19/19 stayed green: an alias named ``docName`` was invisible to
``[a-z_]``; the worked example lived inside a free-text ``hint`` where a
parameter appears as bare ``name='value'``, so changing one hint's
``target=`` to ``subject=`` reproduced the whole live loop; and a quoted
lowercase *value* in a hint was scraped as if it were a parameter, failing
the test for a bug that did not exist. A lexical proxy for a semantic
property, exactly.

So nothing here scrapes. The message is now assembled entirely from
``_kb_tried_names(primary)``, and these tests RECOMPUTE the same values and
assert the rendered string contains them — then drive the obey-loop from that
tuple rather than from whatever the regex happened to catch. The one scan
that remains is a NEGATIVE one: no identifier may appear in the message that
is not the action or a tried name, which is what forbids a future editor from
smuggling a parameter back into the prose.

Nine of these fail against the pre-fix tree. The rest are back-compat and
structural guards that are expected to pass there.
"""

import ast
import inspect
import re

import pytest
from unittest.mock import MagicMock

import ghost_agent.tools.memory as mem
from ghost_agent.tools.memory import (
    _KB_ACTIONS,
    _KB_TARGET_ALIASES,
    _kb_target_or_error,
    _kb_tried_names,
)
from ghost_agent.tools.registry import TOOL_DEFINITIONS
from ghost_agent.tools.tool_failure import (
    FailureClass,
    classify_tool_failure,
    should_retry,
)


#: (action, the parameter the schema advertises as that action's subject).
SUBJECT_ACTIONS = [
    ("forget", "target"),
    ("insert_fact", "fact"),
    ("ingest_document", "filename"),
]

#: Any identifier-shaped token in a message: quoted ('x'), or the left side
#: of an assignment (x=). Deliberately broader than the names we expect —
#: its job is to catch a token we did NOT intend to be there.
_IDENTIFIERS = re.compile(r"'([^']{1,60})'|([A-Za-z_][A-Za-z0-9_\-]*)\s*=")


def _identifiers(message: str) -> set:
    return {a or b for a, b in _IDENTIFIERS.findall(message)}


def _is_marked_value(token: str) -> bool:
    """A placeholder, written so nobody can read it as a parameter name.
    The marker may be the whole token or a segment of it, so a value whose
    SHAPE matters (`'<ep:12>'`) can still show that shape."""
    return "<" in token and ">" in token


#: Tokens that may appear in a hint only if they are accepted names. English
#: prose does not usually contain underscores, hyphens, dots or leading
#: digits; parameter names usually do. A heuristic, and stated as one — it
#: catches `topic-name` / `fact_text` / `2target` / `kb.target`, and does not
#: catch a hint that merely mentions "the subject field" in plain words.
_NAME_SHAPED = re.compile(
    r"\b(?:\d[A-Za-z0-9_.\-]+|[A-Za-z0-9]+[_.\-][A-Za-z0-9_.\-]*[A-Za-z0-9])\b")


def _kb_schema():
    for entry in TOOL_DEFINITIONS:
        fn = entry.get("function", {})
        if fn.get("name") == "knowledge_base":
            return fn
    raise AssertionError("knowledge_base is not advertised at all")


@pytest.fixture
def subject_sink(monkeypatch):
    """Replace the three inner tools with recorders, so "the dispatcher
    accepted this parameter name" is directly observable instead of being
    inferred from the absence of an error. The full call is recorded, not
    just the subject, so a re-ordering of the positional context arguments
    at the three call sites is visible too."""
    seen = []

    async def _remember(text, *a, **k):
        seen.append(("insert_fact", text, a, k))
        return "OK-remember"

    async def _gain(filename, *a, **k):
        seen.append(("ingest_document", filename, a, k))
        return "OK-ingest"

    async def _forget(target, *a, **k):
        seen.append(("forget", target, a, k))
        return "OK-forget"

    monkeypatch.setattr(mem, "tool_remember", _remember)
    monkeypatch.setattr(mem, "tool_gain_knowledge", _gain)
    monkeypatch.setattr(mem, "tool_unified_forget", _forget)
    return seen


def _subject(sink_row):
    return (sink_row[0], sink_row[1])


# ------------------------------------------------- the class invariant

@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS)
async def test_every_name_the_error_offers_actually_works(
        action, primary, subject_sink):
    """The error must be OBEYABLE — driven from the tuple the message is
    built from, so an alias whose spelling the old regex could not see is
    covered like any other."""
    err = await mem.tool_knowledge_base(action=action)
    assert "MANDATORY" in err

    for name in _kb_tried_names(primary):
        out = await mem.tool_knowledge_base(action=action, **{name: "SUBJECT-VALUE"})
        assert "MANDATORY" not in out, (
            f"knowledge_base(action={action!r}) accepts {name!r} nowhere, yet "
            f"it is in the chain the error is generated from — obeying the "
            f"error would return the same error forever. This is the live "
            f"2026-08-28 forget loop."
        )
        assert _subject(subject_sink[-1]) == (action, "SUBJECT-VALUE"), (
            f"{name!r} was accepted without landing as the subject"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS)
async def test_the_message_is_the_recomputed_one(action, primary):
    """Identity, not shape: the required name, the worked call and the
    alternatives list are each recomputed here and must appear verbatim.
    A message that merely *looks* right — two of six alternatives, a
    hand-typed example — is not evidence."""
    err = await mem.tool_knowledge_base(action=action)

    assert f"The '{primary}' parameter is MANDATORY" in err
    assert f"knowledge_base(action='{action}', {primary}=" in err, (
        "the worked example must name the required parameter. It is "
        "generated from `primary` for exactly this reason: when each call "
        "site wrote its own example inside the hint, changing one to "
        "`subject=` reproduced the live loop with every test green."
    )
    # Exactly ONE parameter may be named. The message used to list the other
    # accepted aliases too, and `source`/`path` are as filename-shaped as the
    # `filename` that was excluded for being filename-shaped — a model
    # obeying "insert_fact also accepts 'source'" stores a filename as a
    # permanent fact and gets SUCCESS back. Guarding a proxy for "wrong type"
    # kept the harm; naming one parameter removes it.
    assert "also accepts" not in err
    named = {t for t in _identifiers(err)
             if not _is_marked_value(t)
             and t not in {action, "action", "knowledge_base"}}
    assert named == {primary}, (
        f"the {action} error names {sorted(named)}; it may name only the "
        f"required parameter"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS)
async def test_the_message_contains_no_identifier_it_did_not_try(
        action, primary):
    """The negative scan. Anything identifier-shaped in the message must be
    the action or a name the lookup consulted — this is what stops a
    parameter name being smuggled back into free-text prose."""
    err = await mem.tool_knowledge_base(action=action)
    allowed = {action, "action", "knowledge_base", *_kb_tried_names(primary)}
    # The example VALUE is quoted too; it is data, not a parameter, so it is
    # allowed only if it could not be mistaken for one.
    for token in _identifiers(err) - allowed:
        assert _is_marked_value(token), (
            f"{token!r} appears in the {action} error and is neither a name "
            f"the lookup tried nor a marked placeholder. An earlier version "
            f"of this check asked whether the token was `isidentifier()` — "
            f"which is a proxy, and a false one: the native tool path splats "
            f"arbitrary JSON keys into **kwargs, so 'topic-name' is a "
            f"parameter a model can pass and Python still calls it a "
            f"non-identifier. A hint reading \"or topic-name='atlas'\" "
            f"reproduced the live loop with every pin green. Values are "
            f"marked <like this>; everything else must be an accepted name."
        )


@pytest.mark.asyncio
async def test_the_reported_call_shape_works(subject_sink):
    """The exact call the live model made after reading the error."""
    out = await mem.tool_knowledge_base(action="forget", target="χυσογαργάρες")
    assert out == "OK-forget"
    assert _subject(subject_sink[-1]) == ("forget", "χυσογαργάρες")


# ------------------------------------- the invariant on EVERY error, unstubbed

@pytest.mark.asyncio
async def test_no_reachable_knowledge_base_error_names_an_unusable_parameter():
    """The inner tools keep their own guards, and those name THEIR OWN
    parameters — `tool_remember` says 'text', which the schema does not
    advertise and the dispatcher does not accept. They are unreachable
    through the dispatcher because it now validates the subject itself
    (present, a string, non-blank after stripping) before any inner tool
    sees it — which is a stronger guard than the two happening to share a
    truthiness, and is why the whitespace probes below stop at the
    dispatcher. This walks every error a caller can actually provoke, with
    NO inner tool stubbed, and applies the rule to all of them."""
    # Parameter names the dispatcher reads, kept separate from action names.
    # Folding the two together let a message advertise a PARAMETER called
    # 'forget' or 'reset_all' and pass the scan.
    accepted_params = {*_KB_TARGET_ALIASES, "action", "question", "query",
                       "q", "ref", "id"}
    mentionable = {*_KB_ACTIONS, "knowledge_base", "update_profile"}
    accepted = accepted_params | mentionable
    mem_sys = MagicMock()
    mem_sys.get_library = MagicMock(return_value=[])

    # Every advertised action, bare — derived, so an action added to the enum
    # is covered here without anyone remembering to extend a list. Plus the
    # off-schema and inner-guard cases a bare call cannot reach.
    probes = [{"action": a} for a in _KB_ACTIONS] + [
        {},
        {"action": "frobnicate"},
        {"action": "update_profile"},
        {"action": "update_profile", "key": "car", "value": "Tesla"},
        {"action": "query", "filename": "doc.txt"},
        # the document-not-found branch: reachable, and until now visited by
        # no pin at all
        {"action": "query", "filename": "missing.txt", "question": "q"},
        {"action": "forget", "target": ["a", "b"]},
        # Reachable inner guards: a subject that survives `if val:` but that
        # the inner tool then rejects.
        {"action": "forget", "target": "ab"},
        {"action": "forget", "target": "   "},
        {"action": "insert_fact", "fact": "   "},
        {"action": "expand", "ref": "nonsense:1"},
    ]
    for kwargs in probes:
        out = await mem.tool_knowledge_base(memory_system=mem_sys, **kwargs)
        out = str(out)
        if "SYSTEM ERROR" not in out and "Error" not in out:
            continue
        # A message may quote the caller's OWN input back (an unknown
        # action, a malformed ref); that is an echo, not advice.
        echoed = {str(v) for v in kwargs.values()}
        for token in _identifiers(out) - accepted - echoed:
            assert _is_marked_value(token) or " " in token, (
                f"{kwargs} produced an error naming {token!r}, which is "
                f"neither a parameter the dispatcher reads nor a marked "
                f"value. A caller that obeys it gets the argument "
                f"dropped:\n  {out}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS)
async def test_the_real_message_classifies_as_fatal(action, primary):
    """Classify what the tool ACTUALLY emits, not a synthetic string.
    `_RETRYABLE_PATTERNS` is scanned before `_FATAL_PATTERNS`, so one
    retry-flavoured word anywhere in the message ("timed out", "connection")
    silently re-routes a missing-argument error into the transient budget —
    and the loop comes back, paid for from a different purse."""
    err = await mem.tool_knowledge_base(action=action)
    cls, _match = classify_tool_failure(err)
    assert cls is FailureClass.FATAL, f"{action} error classified {cls}: {err!r}"
    assert should_retry(cls, 0) is False


@pytest.mark.asyncio
async def test_the_action_slot_gets_the_same_treatment():
    """Same bug class, action slot instead of parameter slot: "Unknown
    action 'delete'" named nothing to switch to, and delete/erase/remove are
    not in the alias map — so the next guess was another guess."""
    for probe in ({}, {"action": "delete"}):
        err = await mem.tool_knowledge_base(**probe)
        assert "MANDATORY" in err
        for a in _KB_ACTIONS:
            assert repr(a) in err, f"{probe} error omits the valid action {a!r}"
        cls, _ = classify_tool_failure(err)
        assert cls is FailureClass.FATAL


def test_the_schema_enum_is_the_actions_the_module_advertises():
    """One home. The registry enum and `_KB_ACTIONS` generate the same
    promise to the model — the error's valid-action list would otherwise
    drift from the schema's."""
    enum = _kb_schema()["parameters"]["properties"]["action"]["enum"]
    assert tuple(enum) == _KB_ACTIONS


# ------------------------------------------------- schema <-> dispatcher

def test_schema_advertises_target_for_forget():
    """Defect 2: the first call was unwinnable too — no advertised parameter
    read as "the thing to forget"."""
    props = _kb_schema()["parameters"]["properties"]
    assert "target" in props, (
        "action='forget' needs a parameter a model can find while holding "
        "the word 'topic'. Without it the model must guess, and the guess "
        "that reads best ('no field fits, call it bare') is the loop."
    )
    assert "forget" in props["target"]["description"]


def test_filename_no_longer_claims_to_be_the_forget_parameter():
    """`filename` still WORKS for forget (back-compat), but it must point at
    `target` rather than presenting itself as the way to name a topic."""
    desc = _kb_schema()["parameters"]["properties"]["filename"]["description"]
    assert "target" in desc


def test_every_advertised_param_is_read_by_some_branch():
    """The other direction: the schema must not advertise a name the handler
    would silently drop.

    The allowlist is DERIVED — every string literal the dispatcher passes to
    `kwargs.get(...)`, read out of its AST, plus the alias tuple (which the
    helper consults through a variable, so the AST cannot see it). An earlier
    version hardcoded the exceptions instead, which made the failure message
    tell the next developer to widen `_KB_TARGET_ALIASES` for a parameter
    that is not a subject at all — after which
    `knowledge_base(action='forget', limit=…)` would erase the limit.
    """
    tree = ast.parse(inspect.getsource(mem.tool_knowledge_base))
    read_literally = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "kwargs"
        and node.args and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    readable = read_literally | set(_KB_TARGET_ALIASES) | {"action"}
    advertised = set(_kb_schema()["parameters"]["properties"])
    unread = advertised - readable
    assert not unread, (
        f"Schema advertises {sorted(unread)}, which no branch of "
        f"tool_knowledge_base reads — a model that follows the schema gets "
        f"its argument dropped.\n"
        f"Fix it by making the OWNING BRANCH read it. Do NOT add a "
        f"non-subject parameter to _KB_TARGET_ALIASES: that would make it "
        f"resolvable as the thing to ingest or erase.\n"
        f"Note this check is a SHAPE check on `kwargs.get(\"literal\")` in "
        f"this one function. It cannot see `kwargs[...]`, a read inside a "
        f"helper, or a read whose result is discarded, and it cannot tell "
        f"WHICH branch reads a name — so it is a floor, not a proof. The "
        f"companion check is "
        f"test_every_parameter_an_error_demands_is_advertised."
    )


@pytest.mark.asyncio
async def test_the_error_keeps_the_verb_the_caller_reached_for():
    """§4AW made `transcribe` a first-class alias because a model holding
    that verb could not find this tool. Rendering the canonical name back at
    it — in the ONE worked call it is given — undoes that, and offers a .pdf
    example to an audio ask."""
    err = await mem.tool_knowledge_base(action="transcribe")
    assert "action='transcribe'" in err
    assert "ingest_document" not in err
    assert "mp4" in err, "a media verb deserves a media example"

    plain = await mem.tool_knowledge_base(action="ingest_document")
    assert "action='ingest_document'" in plain


@pytest.mark.asyncio
async def test_update_profile_redirects_without_naming_a_disabled_tool():
    """`update_profile` is in `disabled_tools` for subagents, self-play and
    dream. "Call update_profile instead" bounces the model between two
    errors with different signatures, so the same-failure loop breaker never
    fires and the strike budget drains. Name the actions THIS tool has."""
    err = await mem.tool_knowledge_base(action="update_profile", key="c", value="v")
    assert "MANDATORY" in err
    for a in _KB_ACTIONS:
        assert repr(a) in err
    cls, _ = classify_tool_failure(err)
    assert cls is FailureClass.FATAL


@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS)
async def test_unadvised_aliases_are_still_accepted(
        action, primary, subject_sink):
    """The aliases exist for old trajectories, not as advice. Dropping them
    from the message must not drop them from the lookup."""
    for other in set(_KB_TARGET_ALIASES) - {primary}:
        assert repr(other) not in await mem.tool_knowledge_base(action=action)
        out = await mem.tool_knowledge_base(action=action, **{other: "V"})
        assert "MANDATORY" not in out
        assert _subject(subject_sink[-1]) == (action, "V")


@pytest.mark.asyncio
async def test_the_worked_example_is_inert_if_copied_verbatim():
    """A model copies the example. `fact='The user lives in Athens'` wrote a
    plausible fabricated profile fact and returned SUCCESS;
    `target='project atlas'` ran a real four-store destructive sweep. The
    example has to be visibly a placeholder."""
    for action, _primary in SUBJECT_ACTIONS:
        err = await mem.tool_knowledge_base(action=action)
        example = re.search(r"Worked call: knowledge_base\([^)]*=('[^']*')\)",
                            err).group(1)
        assert example.startswith("'<") and example.endswith(">'"), (
            f"{action}: {example} reads as a real value; copied verbatim it "
            f"is a live write"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS)
async def test_schema_primary_name_reaches_the_inner_tool(
        action, primary, subject_sink):
    out = await mem.tool_knowledge_base(action=action, **{primary: "VALUE"})
    assert "MANDATORY" not in out
    assert _subject(subject_sink[-1]) == (action, "VALUE")


# ------------------------------------------------- back-compat

@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["content", "source", "path", "topic"])
async def test_legacy_aliases_still_resolve(alias, subject_sink):
    """The legacy aliases predate the schema rename and are still emitted by
    old trajectories; adding 'target' must not evict them."""
    out = await mem.tool_knowledge_base(action="forget", **{alias: "atlas"})
    assert out == "OK-forget"
    assert _subject(subject_sink[-1]) == ("forget", "atlas")


@pytest.mark.asyncio
async def test_appending_target_did_not_change_existing_precedence(subject_sink):
    """'target' was APPENDED to the alias chain, so a call that already
    resolved keeps resolving to the same value."""
    await mem.tool_knowledge_base(
        action="ingest_document", filename="doc.pdf", content="prose")
    assert _subject(subject_sink[-1]) == ("ingest_document", "doc.pdf")


@pytest.mark.asyncio
async def test_the_fallback_order_is_the_tuple_order(subject_sink):
    """Order is load-bearing — the generic resolution feeds query/expand and
    a multi-alias call resolves by position. Reversing the tuple must not be
    a silent no-op."""
    for i, alias in enumerate(_KB_TARGET_ALIASES):
        rest = {a: f"v-{a}" for a in _KB_TARGET_ALIASES[i:]}
        await mem.tool_knowledge_base(action="ingest_document", **rest)
        expected = next(a for a in _KB_TARGET_ALIASES if a in rest)
        assert _subject(subject_sink[-1]) == ("ingest_document", f"v-{expected}")


@pytest.mark.asyncio
async def test_each_action_prefers_its_own_advertised_name(subject_sink):
    """When both are present the action's OWN schema name wins — otherwise
    the fix would hand 'forget' a filename the caller meant as a document.
    This is a real widening on an irreversible action (`forget` sweeps disk,
    vector, profile and graph), so it is pinned deliberately rather than
    left to fall out of the alias order."""
    await mem.tool_knowledge_base(
        action="forget", target="the topic", filename="unrelated.pdf")
    assert _subject(subject_sink[-1]) == ("forget", "the topic")


@pytest.mark.asyncio
@pytest.mark.parametrize("padded,clean", [
    (" atlas ", "atlas"), ("\tatlas\n", "atlas"), ("  project atlas  ", "project atlas"),
])
async def test_the_subject_is_normalised_before_it_is_used(
        padded, clean, subject_sink):
    """STRIP, don't merely test. An earlier version computed `val.strip()`
    to decide the subject was present and passed the PADDED original on —
    and `tool_unified_forget` strips in only 3 of its 6 uses, so
    `target=' atlas '` skipped the disk and document sweeps while reporting
    every stage with a ✅ and being booked as a clean success. Padded values
    are the normal XML shape: that parser strips CR/LF, not spaces."""
    await mem.tool_knowledge_base(action="forget", target=padded)
    assert _subject(subject_sink[-1]) == ("forget", clean)

    await mem.tool_knowledge_base(action="insert_fact", fact=padded)
    assert _subject(subject_sink[-1]) == ("insert_fact", clean)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [["a", "b"], {"a": 1}, 12, True, 1.5])
async def test_a_non_string_subject_is_refused_not_half_applied(
        bad, subject_sink):
    """`target=['project atlas','mortimer']` is a plausible native-JSON
    shape for "forget X and Y". It used to sail through: the vector and
    profile sweeps raised and were swallowed, the graph sweep "succeeded"
    against the repr, NOTHING was deleted — and because the report never
    starts with an error prefix the turn was booked as a clean success with
    the bookkeeping suppressed. The schema says string."""
    out = await mem.tool_knowledge_base(action="forget", target=bad)
    assert "MANDATORY" in out and "must be a single string" in out
    assert subject_sink == [], f"{bad!r} reached the wipe"
    # obeyable: it names the parameter and carries a call that works
    assert "'target'" in out
    assert "knowledge_base(action='forget', target=" in out


@pytest.mark.asyncio
async def test_a_blank_primary_promotes_the_next_alias(subject_sink):
    """A blank primary falls through the chain rather than short-circuiting
    it. This is a widening on an irreversible action — pre-change the blank
    itself was passed on and bounced off the <3-character guard — so it is
    pinned deliberately."""
    await mem.tool_knowledge_base(
        action="forget", target="   ", filename="notes.txt")
    assert _subject(subject_sink[-1]) == ("forget", "notes.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  \n"])
async def test_a_blank_subject_is_rejected_not_stored(blank, subject_sink):
    """`if val:` passed a run of spaces, and the inner guard is `if not
    text:` — so `insert_fact(fact='   ')` stored a blank fact and reported
    SUCCESS. A subject made of whitespace is a missing subject."""
    out = await mem.tool_knowledge_base(action="insert_fact", fact=blank)
    assert "MANDATORY" in out, f"{blank!r} was accepted as a fact"
    assert subject_sink == [], f"{blank!r} reached the inner tool"

    out = await mem.tool_knowledge_base(action="forget", target=blank)
    assert "MANDATORY" in out
    assert subject_sink == []


@pytest.mark.asyncio
async def test_an_empty_subject_is_not_a_subject(subject_sink):
    """A blank string must fall through to the next alias, and a wholly
    blank call must produce the dispatcher's obeyable error rather than
    being forwarded as an empty topic (which the inner tool would then
    reject with a message naming its own parameter)."""
    await mem.tool_knowledge_base(action="forget", target="", filename="doc.txt")
    assert _subject(subject_sink[-1]) == ("forget", "doc.txt")

    out = await mem.tool_knowledge_base(action="forget", target="", topic="")
    assert "MANDATORY" in out
    assert _subject(subject_sink[-1]) == ("forget", "doc.txt")


# ---------------------------------- the untouched branches the change reached

@pytest.mark.asyncio
async def test_target_also_resolves_for_query_and_expand(monkeypatch):
    """`target` is appended LAST in the generic chain, so it can only fire
    for query/expand when every legacy name is empty — turning two calls
    that used to be dead ends into working ones. Pinned because the generic
    resolution was rewritten: setting it to None must not pass silently."""
    seen = {}

    async def _query(filename=None, question=None, **k):
        seen["query"] = (filename, question)
        return "OK-query"

    async def _expand(ref=None, **k):
        seen["expand"] = ref
        return "OK-expand"

    monkeypatch.setattr(mem, "tool_query_document", _query)
    monkeypatch.setattr(mem, "tool_expand_evidence", _expand)

    await mem.tool_knowledge_base(action="query", target="manual.pdf", question="q")
    assert seen["query"] == ("manual.pdf", "q")

    await mem.tool_knowledge_base(action="expand", target="ep:12")
    assert seen["expand"] == "ep:12"


# ------------------------------------------------- the helper's guarantee

def test_helper_can_only_name_parameters_it_tried():
    """Feed `_kb_target_or_error` a primary that is NOT in the alias tuple:
    the message must still only mention names the lookup covered. The
    property is structural — one `primary` argument feeds both halves."""
    value, err = _kb_target_or_error({}, "someaction", "weird_name",
                                     "some prose", "<an example value>")
    assert value is None
    tried = {"weird_name", *_KB_TARGET_ALIASES}
    named = {t for t in _identifiers(err)
             if t not in {"someaction", "action", "knowledge_base"}
             and not _is_marked_value(t)}
    assert named <= tried, f"message names untried params: {named - tried}"
    assert "weird_name" in named

    got, err2 = _kb_target_or_error(
        {"weird_name": "A", "filename": "B"}, "someaction", "weird_name",
        "prose", "<example>")
    assert (got, err2) == ("A", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("action,primary", SUBJECT_ACTIONS + [
    # `transcribe` renders a DIFFERENT hint and example from
    # `ingest_document` (the media branch), so it is a fourth hint. Missing
    # it left a `file-name='a b'` mutation alive in exactly the channel this
    # test exists to close.
    ("transcribe", "filename"),
])
async def test_no_hint_carries_a_name_shaped_token(action, primary):
    """The hint is free text, and free text was the channel that
    reproduced the live loop: `"...or topic-name='project atlas'"` renders a
    call shape a model copies and has dropped. Two things stop it — the
    dispatcher strips `=` and quotes from every hint, and this asserts no
    name-shaped token survives unless it is an accepted parameter.

    A heuristic, and only that: it catches `topic-name`, `fact_text`,
    `2target`, `kb.target` — the spellings an editor reaches for — and it
    does NOT catch a hint that mentions "the subject field" in plain prose.
    That residue is why the worked call is generated rather than written."""
    err = await mem.tool_knowledge_base(action=action)
    hint = err.split("—", 1)[1].split("Worked call:", 1)[0]
    allowed = {action, *_kb_tried_names(primary)}
    for token in _NAME_SHAPED.findall(hint):
        assert token in allowed, (
            f"the {action} hint contains {token!r}, which reads like a "
            f"parameter and is not one the lookup tries"
        )


@pytest.mark.asyncio
async def test_hints_cannot_render_a_call_shape():
    """Enforced at runtime, not only in review: `=` and quotes are stripped
    from every hint, so a hint added later that spells out a call cannot
    reach a model as something to copy."""
    # Typographic forms too: a curly apostrophe is what you get from pasting
    # prose out of a document, and `subject=‘project atlas’` renders a
    # perfectly copyable call that an ASCII-only class waves through.
    for hint in ("erase it, or pass subject='project atlas' instead",
                 "erase it, or pass subject＝‘project atlas’ instead",
                 "erase it, or pass `subject` instead",
                 'erase it, or pass subject="project atlas"'):
        _v, err = _kb_target_or_error({}, "forget", "target", hint, "<x>")
        body = err.split("—", 1)[1].split("Worked call:", 1)[0]
        for ch in "=＝'\"`‘’“”":
            assert ch not in body, (
                f"a hint rendered a callable fragment ({ch!r} survived): "
                f"{body!r}"
            )
    # ...and every `=` left in the message belongs to a generated call:
    # `action='forget'` in the prose, then `action=` and `target=` in the
    # worked call. Nothing the hint contributed can render as an assignment.
    assert err.count("=") == 3, err


@pytest.mark.asyncio
@pytest.mark.parametrize("action", list(_KB_ACTIONS) + ["frobnicate", ""])
async def test_every_error_carries_the_prefix_the_turn_loop_books_it_by(
        action):
    """`turn_has_failure` is set by `str_res.startswith(("Error:", "ERROR",
    "SYSTEM ERROR", "Critical Tool Error"))`, and the failure classifier and
    the ARGUMENT ERROR framing are reached ONLY from there. Reword the
    prefix and the error is booked as a SUCCESS: no strike, no class, no
    diagnostic — the whole recovery path hangs on this string."""
    mem_sys = MagicMock()
    mem_sys.get_library = MagicMock(return_value=[])
    out = str(await mem.tool_knowledge_base(action=action, memory_system=mem_sys))
    if "MANDATORY" not in out and "Unknown action" not in out:
        return                                  # this action succeeded
    assert out.startswith(("Error:", "ERROR", "SYSTEM ERROR",
                           "Critical Tool Error")), (
        f"action={action!r} produced an unbooked failure: {out[:120]!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", _KB_ACTIONS)
async def test_every_parameter_an_error_demands_is_advertised(action):
    """The other half of the original defect, generalised. `target` was
    demanded but not advertised; so is any parameter an error calls
    MANDATORY that the schema does not carry. Withdrawing `question` from
    the schema, for instance, leaves `query` demanding a name the model has
    never been shown."""
    mem_sys = MagicMock()
    mem_sys.get_library = MagicMock(return_value=[])
    out = str(await mem.tool_knowledge_base(action=action, memory_system=mem_sys))
    advertised = set(_kb_schema()["parameters"]["properties"])
    demanded = set(re.findall(r"'([a-z_]+)'(?= and | are | parameter)", out))
    demanded |= set(re.findall(r"The '([a-z_]+)' parameter is MANDATORY", out))
    for name in demanded - {"action"}:
        assert name in advertised, (
            f"action={action!r} demands {name!r}, which the schema does not "
            f"advertise — a model reading the schema cannot supply it"
        )


def test_the_required_name_is_always_tried_first():
    """`_kb_tried_names` is what makes the message's one named parameter a
    name the lookup consulted. Whatever the fallback chain becomes, the
    required name must lead it."""
    for primary in {*_KB_TARGET_ALIASES, "weird_name"}:
        tried = _kb_tried_names(primary)
        assert tried[0] == primary
        assert len(tried) == len(set(tried)), "a name consulted twice"
        # EXACT, not a superset. A name the lookup consults but the tuple
        # does not record is a parameter the dispatcher silently accepts and
        # nothing documents — `_KB_TARGET_ALIASES` is supposed to be the one
        # statement of what this tool takes.
        assert set(tried) == {primary} | set(_KB_TARGET_ALIASES), (
            f"tried names for {primary!r} are not the recorded set: "
            f"{sorted(set(tried) ^ ({primary} | set(_KB_TARGET_ALIASES)))}"
        )
