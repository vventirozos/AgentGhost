# src/ghost_agent/tools/composed_skills.py
"""Tool Composition and Macro Learning.

Compiled multi-step tool sequences that the agent has discovered through
repeated use. Unlike single acquired skills, composed skills are
*sequences* of tool calls with conditional branching — reusable
procedures the agent can execute as a single macro.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

# A composed-skill name is advertised to the LLM as a top-level tool name
# (see `to_tool_definitions`), so it must be a bare identifier — no spaces,
# dots, slashes, or punctuation that would break the function catalogue or
# let a macro masquerade as a path. Same shape-guard rationale as the
# acquired-skill name check in acquired_skills.py.
_SAFE_COMPOSED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _validate_composed_name(name: str) -> str:
    """Return `name` if it is a safe identifier; raise ValueError otherwise."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"composed-skill name must be a non-empty string, got {name!r}")
    if not _SAFE_COMPOSED_NAME_RE.match(name):
        raise ValueError(
            f"composed-skill name {name!r} rejected: must match "
            f"[A-Za-z_][A-Za-z0-9_]{{0,63}} (it becomes an LLM tool name, so no "
            f"spaces, dots, slashes, or punctuation)."
        )
    return name


# Per-step result cap for composed-skill execution. Each step's body is
# bounded so one chatty step can't blow the context budget when the macro's
# combined output is handed back to the LLM. The original cap was 1000 chars,
# which SILENTLY truncated a list-bearing step — e.g. the morning briefing's
# "latest 10 headlines" step (well over 1000 chars for 10 items) was cut down
# to ~2 headlines, which is exactly the "briefing only shows 2, not 10" bug.
# 4000 chars comfortably carries ~10 headlines while still bounding a runaway
# step; across a handful of steps the macro's total stays context-safe.
MAX_STEP_RESULT_CHARS = 4000

# `$var` / `${var}` templates (see SkillManager._resolve_args). _WHOLE_VAR_RE
# matches a param whose ENTIRE value is one reference (substitute as-is);
# _VAR_RE finds references embedded in surrounding text (interpolate).
_WHOLE_VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

# Cap a value BOUND via save_as. Larger than the display cap (a downstream
# step may legitimately need a big fetched body) but still bounded so a
# runaway step can't blow the next tool's args.
MAX_BOUND_VALUE_CHARS = 16000

_BIND_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

# Parallel-mode fan-out window. Most steps ultimately hit the single-slot
# local llama server (searches, LLM-backed tools), so an unbounded gather
# of 20 steps would stampede it; 4 keeps the pipeline busy without that.
_PARALLEL_STEP_CONCURRENCY = 4


def _validate_dataflow(steps, mode: str, branches=None):
    """Reject a macro whose `save_as`/`$var` wiring can't work at runtime.
    Returns an error string, or None when the data-flow is sound.

    Two authoring mistakes are caught here rather than silently resolving
    to "" at execution time:
      * a step referencing a name bound by a LATER step (or by itself) —
        bindings only flow forward;
      * any step-produced binding in a PARALLEL macro — steps start
        simultaneously, so no sibling can observe another's result.
    A name that is never produced by any step is fine: it's a runtime
    param the caller supplies (that's the pre-existing contract).

    ``branches`` (2026-07-14): each branch sequence gets the same checks.
    A branch step may freely reference MAIN-step bindings (at runtime the
    branch runs after the branching step, so main bindings bound up to that
    point are in scope) — only within-branch ordering is enforced.
    """
    def _check_seq(seq, label):
        produced_at = {}
        for i, st in enumerate(seq):
            if st.save_as:
                if st.save_as in produced_at:
                    return (f"Error: {label} {i + 1} re-binds 'save_as' name "
                            f"{st.save_as!r} already bound by {label} "
                            f"{produced_at[st.save_as] + 1} — use a distinct name.")
                produced_at[st.save_as] = i
        if not produced_at:
            return None
        if mode == "parallel":
            return ("Error: 'save_as' data-flow requires mode='sequential' — "
                    "parallel steps start simultaneously, so a step cannot "
                    "consume a sibling's result.")
        for i, st in enumerate(seq):
            if not isinstance(st.param_template, dict):
                continue
            for key, v in st.param_template.items():
                if not isinstance(v, str) or "$" not in v:
                    continue
                for mo in _VAR_RE.finditer(v):
                    nm = mo.group(1) or mo.group(2)
                    src = produced_at.get(nm)
                    if src is None:
                        continue  # runtime param or a main-step binding
                    if src == i:
                        return (f"Error: {label} {i + 1} param {key!r} references "
                                f"${nm}, which the SAME step produces — a step "
                                f"cannot consume its own output.")
                    if src > i:
                        return (f"Error: {label} {i + 1} param {key!r} references "
                                f"${nm}, but {label} {src + 1} produces it — "
                                f"bindings only flow forward. Reorder the steps.")
        return None

    err = _check_seq(steps, "step")
    if err:
        return err
    for bname, bsteps in (branches or {}).items():
        err = _check_seq(bsteps, f"branch {bname!r} step")
        if err:
            return err
    return None


def _cap_step_result(result_str: str, limit: int = MAX_STEP_RESULT_CHARS) -> str:
    """Bound a single step's result body, marking any truncation EXPLICITLY.

    Returns ``result_str`` unchanged when it fits within ``limit``. When it
    does not, truncates to ``limit`` chars and appends a visible marker noting
    how many chars were dropped — so the truncation is never silent. The model
    (and the verifier gate) can then SEE that content was cut and re-fetch the
    step standalone, instead of believing it received the whole list and
    delivering a short answer. That silent-drop-then-believe-it-was-complete
    path is what made the briefing ship 2 (and later 8) of 10 headlines.
    """
    if result_str is None:
        return ""
    if len(result_str) <= limit:
        return result_str
    dropped = len(result_str) - limit
    return (
        f"{result_str[:limit]}\n"
        f"…[truncated {dropped} chars — this step's full output exceeded the "
        f"{limit}-char per-step cap; re-run this step's tool standalone to get "
        f"the complete result]"
    )


def _step_result_ok(result_str: str) -> bool:
    """Classify a step's RESULT as success/failure.

    Tools in this codebase RETURN error strings (``"[error] …"``,
    ``"Error: …"``, ``"[SYSTEM ERROR] …"``, ``"SYSTEM BLOCK …"``) rather than
    raising, so a macro that only checks "did the executor raise" records an
    all-failed run as a success (inflating success_rate and telling the LLM the
    macro worked). Mirror the acquired-skill result gate: inspect the string.
    """
    # A migrated tool ANSWERS this. ADD-only: `ok` falls through to the
    # prose rules, so the exit-code banner keeps its authority.
    _st = getattr(result_str, "status", None)
    if _st is not None and str(getattr(_st, "value", _st)) != "ok":
        return False
    s = str(result_str or "").lstrip()
    if not s:
        return True  # empty output is not an error
    if "[SYSTEM ERROR]" in s or "SYSTEM BLOCK" in s or "Critical Tool Error" in s:
        return False
    # A step DETACHED at its budget (sandbox/jobs.py) reports exit 0 while
    # STILL RUNNING. Counting it a success inflates the macro's success_rate
    # — the number the model is shown when choosing a macro — on a step whose
    # outcome nobody knows yet.
    from ..sandbox.jobs import is_promoted_result
    if is_promoted_result(s):
        return False
    m = re.search(r"EXIT CODE:\s*(\d+)", s)
    if m:
        return m.group(1) == "0"
    # "SYSTEM INSTRUCTION:" and "REJECTED:" are file_system's hard-failure
    # prefixes (missing params, replace block not found, syntax-regression
    # rollback) — they used to count as SUCCESSES here, inflating macro
    # success_rate. Prefix-only checks: a SUCCESS message that merely
    # *contains* "SYSTEM INSTRUCTION" mid-text (e.g. a partial aider-block
    # report) still counts as ok.
    # `CRITICAL ERROR:` is search.py's hard-failure head (3 producers). It
    # was added to `_FAILURE_PREFIX_RE` and to none of the other prefix
    # banks, so a `deep_research` that never ran scored a step SUCCESS here.
    return not s.startswith(
        ("[error]", "Error", "ERROR", "SYSTEM ERROR", "CRITICAL ERROR",
         "Traceback", "SYSTEM INSTRUCTION", "REJECTED")
    )


# ── Auto-mint: a param SCHEMA, not baked literals (§4CS, 2026-08-23) ──
#
# An auto-mined macro used to get, per step, the MOST COMMON observed
# argument dict. That replays a stale one-off WRITE verbatim on every run
# (live: manage_projects(action='task_update', status='DONE',
# project_id=<old>, task_id=<old>, description='Simplified demo.py…')).
# The 2026-07-29 remedy was a tool DENY-LIST in core/dream.py that blanked
# the whole template for eleven tools — it removed the hazard and the
# artifact with it. A param-less macro advertises zero inputs, so every
# step runs with no args and it can never be activated. Measured
# 2026-08-23: 25 auto-mined macros, 0 invocations, all time.
#
# This mints a SCHEMA instead. Per (step, key) the observed values decide:
#
#   literal — the value is CONSTANT across every observation AND is drawn
#             from the tool's OWN DECLARED ``enum``. That is the whole
#             rule. An enum is a closed set the tool itself publishes, so
#             an enum member is a MODE — the macro's identity — and cannot
#             be a path, a command, an id, or free text.
#   dropped — a constant flag or blank (bool / None / "" / an empty
#             container) that the tool does not require. It carries no
#             identity, and dropping is strictly safer than freezing: an
#             omitted flag cannot pre-authorise anything.
#   slot    — EVERYTHING ELSE becomes ``$name``, which ``to_tool_definitions``
#             already advertises as a REQUIRED runtime param and
#             ``_resolve_args`` already substitutes at call time. No new
#             execution machinery: this fills a hole in an existing one.
#
# ⚠ THE FIRST VERSION OF THIS RULE ALSO FROZE ANY "payload-free" CONSTANT
# — bool, number, None, "" — on the reasoning that such a value cannot
# carry a path or a command. A reviewer ran it and found two holes, and
# both are the same mistake: the predicate named a SHAPE and the property
# needed was a ROLE.
#   * `postgres_admin.confirm=True` is a bool, and it is the DESTRUCTIVE-DDL
#     AUTHORISATION. The macro would have pre-consented to DROP/TRUNCATE
#     and asked the caller only for `$sql`.
#   * `manage_projects(task_id=42)` is an int — the exact stale-id replay
#     the deny-list existed for, reproduced through its replacement. It did
#     not fire live only because this project's ids happen to be hex
#     strings. `manage_services(port=5055)` WAS being frozen for real.
# So the licence is now enum-membership alone, which is the tool's own
# statement about its own closed sets rather than our guess about its
# values. The test is on the VALUE, not the tool NAME, so it still
# subsumes the deny-list — and now the claim it makes is true.

#: A macro needing more runtime inputs than this is not a macro, it is a
#: form — minting it produces an artifact no caller can fill.
MACRO_MAX_RUNTIME_SLOTS = 6

_SLOT_KEY_RE = re.compile(r"[^A-Za-z0-9_]")
_SLOT_HEAD_RE = re.compile(r"^[A-Za-z_]")

#: Built once from the tool registry. Never populated with a FAILURE —
#: see `_tool_schema_index`.
_TOOL_SCHEMA_INDEX: Optional[Dict[str, Dict[str, Any]]] = None


# ⚠ `_carries_no_identity` LIVED HERE and is GONE (review round 2). It named
# a value's SHAPE — bool / number / None / "" / empty container — and licensed
# first FREEZING and then DROPPING such a constant. Both were wrong, because
# the property needed was a ROLE, not a shape: `postgres_admin.confirm=True`
# is a bool AND the destructive-DDL authorisation, and `browser.stop_on_error`
# defaults to False so dropping an observed True inverts a fail-fast interlock
# to fail-open. Only the tool's own dispatch selector is frozen now, and
# everything else slots — so nothing needs this predicate. Deleted rather than
# left behind: a helper nobody calls reads as a guard that is still guarding.


def _tool_schema_index(force: bool = False):
    """``{tool: {"enums": {param: [values]}, "required": (params,)}}``, or
    ``None`` when the tool registry cannot be read.

    ``None`` is NOT an empty index. Without the registry we cannot tell a
    mode selector from a payload, and cannot tell whether a step's
    template covers its required params — so callers must treat ``None``
    as "mint nothing this pass". The alternative is the shape this project
    keeps finding, where a check that cannot run reports the favourable
    outcome.

    A tool that is simply ABSENT from a registry that loaded is a genuine
    absence (no enums, no known required params), which is a different
    answer from a registry that failed.
    """
    global _TOOL_SCHEMA_INDEX
    if _TOOL_SCHEMA_INDEX is not None and not force:
        return _TOOL_SCHEMA_INDEX
    try:
        # Lazy: registry.py imports THIS module at import time, so a
        # module-level import here is a cycle.
        from .registry import TOOL_DEFINITIONS, get_active_tool_definitions
    except Exception as exc:                                # pragma: no cover
        logger.warning("macro mint: tool registry unreadable (%s) — no "
                       "macro can be minted this pass", exc)
        return None
    # ⚠ REVIEW ROUND 1: the static list is NOT the tool set. `vision_analysis`
    # (unconditionally) and `image_generation` (with an image node) are
    # APPENDED in `get_active_tool_definitions`, so an index built from
    # `TOOL_DEFINITIONS` alone did not know `vision_analysis` — the tool in
    # the single highest-support mined sequence on the live corpus. Union
    # the two: the context-free call adds what it can, the static list
    # backstops anything that call drops for want of a context.
    entries = list(TOOL_DEFINITIONS or ())
    try:
        # `serve_tuned=False`: this builds a NAME index, and applying
        # the tuned descriptions stamps and prunes the caller's request
        # attribution over whatever set it assembled. §4DA round 16.
        entries += list(get_active_tool_definitions(
            None, serve_tuned=False) or ())
    except Exception as exc:                                # noqa: BLE001
        logger.debug("macro mint: active tool definitions unavailable (%s); "
                     "falling back to the static list", exc)
    idx: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        fn = (entry or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or {}
        props = schema.get("properties") or {}
        enums = {p: list(spec["enum"]) for p, spec in props.items()
                 if isinstance(spec, dict) and isinstance(spec.get("enum"), list)}
        required = tuple(schema.get("required") or ())
        # ⚠ REVIEW ROUND 2. "Enum-typed" is NOT the same as "a mode". A
        # tool's PRIMARY DISPATCH SELECTOR — the enum it requires, which
        # decides which operation runs — is the macro's identity. Its
        # OTHER enums are payload: `manage_projects.status` is enum-typed
        # and it is the VALUE BEING WRITTEN, so freezing it re-froze half
        # of the very artifact the retired deny-list existed for
        # (`task_update status=DONE` with stale ids), live, on 2 of 12
        # mintable sequences. Same for `.kind`, `.artifact_kind`,
        # `.dependency_type`, `browser.wait_until`,
        # `manage_composed_skills.mode` and `list_lessons.scope`.
        #
        # A tool that requires no enum but publishes exactly one named
        # `action`/`operation` still has a selector (`workspace`,
        # `introspect`); anything else does not.
        req_enums = [k for k in enums if k in required]
        if req_enums:
            selector = req_enums[0]
        else:
            named = [k for k in enums if k in ("action", "operation")]
            selector = named[0] if len(named) == 1 else None
        idx[str(name)] = {"enums": enums,
                          "selector": selector,
                          "params": tuple(props),
                          "required": required}
    _TOOL_SCHEMA_INDEX = idx
    return idx


def macro_step_inputs(param_template) -> List[str]:
    """Slot names one step's ``param_template`` references, first-seen order."""
    out: List[str] = []
    for v in (param_template or {}).values():
        if not isinstance(v, str):
            continue
        for mo in _VAR_RE.finditer(v):
            nm = mo.group(1) or mo.group(2)
            if nm and nm not in out:
                out.append(nm)
    return out


def macro_mode_key(tool_name: str, args) -> tuple:
    """The modes a single observed call FIXES, as a canonical key.

    A tool's ``enum``-declared params are its mode selectors — the closed
    sets it publishes (``file_system.operation``, ``manage_services.action``).
    Only values actually drawn from the declared enum count, so this agrees
    exactly with what ``mint_param_schema`` is willing to freeze as a
    literal.

    §4CS: the miner keys its windows on ``(tool, mode)`` rather than the
    tool name alone. Measured on the live corpus, name-only keying collapsed
    "read a file then START a service", "edit a file then RESTART a service"
    and "list a dir then start a service" into ONE signature whose operation
    and action therefore varied, so every mode became a runtime slot and the
    macro lost its identity: 2 of 106 windows were mintable. Keying on the
    mode separates them and 91 are — and they read as real workflows.
    """
    idx = _tool_schema_index()
    if idx is None:
        return ()
    enums = (idx.get(tool_name) or {}).get("enums") or {}
    args = args if isinstance(args, dict) else {}
    return tuple(sorted((k, str(args[k])) for k, allowed in enums.items()
                        if k in args and args[k] in allowed))


def macro_identity(steps) -> tuple:
    """Canonical identity of a macro: its tools AND the modes it FIXES.

    ``steps`` is an iterable of ``(tool_name, param_template)``. This is the
    ONE definition both sides of the re-propose guard use — the miner stamps
    it onto a proposal, and `Dreamer._propose_macros_sync` derives it from
    what is already on file. Deriving each side separately is how the guard
    used to compare a mined ``(tool, mode)`` window against a stored
    tool-name tuple and suppress every mode variant after the first.
    """
    idx = _tool_schema_index() or {}
    out = []
    for tool, template in steps:
        enums = (idx.get(tool) or {}).get("enums") or {}
        fixed = tuple(sorted(
            (k, str(v)) for k, v in (template or {}).items()
            if k in enums and not (isinstance(v, str) and v.startswith("$"))))
        out.append((str(tool), fixed))
    return tuple(out)


#: Tools that must never anchor an auto-proposed macro: meta / control-flow
#: tools, or one-off side-effecting tools that are not reusable as a bundled
#: step. A sequence made up entirely of these is dropped.
#:
#: ⚠ REVIEW ROUND 2 moved this here from `core/dream.py`. It gated the DREAM
#: producer only, so the skills_auto graduation producer — which reaches
#: `compile_from_pattern` by a different route — applied NO admission rule
#: at all. Over the live corpus that path offered
#: `manage_composed_skills{action:run, name:$name, params:$params} → …`:
#: a macro whose first step RUNS AN ARBITRARY COMPOSED SKILL by name, which
#: is exactly the meta-recursion this set exists to prevent (and the same
#: enum publishes `approve` and `delete`). It also offered
#: `notify_operator → notify_operator` and `web_search × 3`, all of which
#: the dream miner's same-tool rule refuses. Nothing shipped from them only
#: because an unrelated statistical threshold happened to bite first.
MACRO_IGNORE_TOOLS = frozenset({
    "replan", "abort_attempt", "flag_uncertainty", "manage_composed_skills",
    "create_skill", "manage_skills", "self_play", "self_play_loop",
    "stop_self_play", "dream_mode", "self_state", "introspect",
})


#: The stamps BOTH auto-mint producers write into a macro's
#: `trigger_description`, and that `core/liveness._is_loop_minted_macro`
#: reads back to decide whether a stored macro is the LOOP's output.
#:
#: ⚠ ONE definition, deliberately. The reader used to hold COPIES of the
#: producers' strings with nothing linking them, so rewording either
#: producer — a pure refactor — made the loop's own output invisible to
#: the yield surface AND made the row assert a fabricated provenance fact
#: ("2 hand-written macro(s) excluded" about two macros the loop had just
#: minted), with the suite green. Provenance is a semantic property, and a
#: detached copy of a string is a lexical proxy for it.
MACRO_MARK_MINED = "Auto-discovered recurring sequence"
MACRO_MARK_GRADUATED = "sequence graduated from"


def macro_sequence_admissible(tool_names) -> Optional[str]:
    """Why this tool sequence must not become a macro, or None.

    ONE definition, called by BOTH producers. Splitting it is how the
    graduation path ended up with no admission rule at all.
    """
    tools = [str(t) for t in (tool_names or ())]
    if not tools:
        return "empty tool sequence"
    if not all(tools):
        return "the sequence contains an unnamed tool call"
    if all(t in MACRO_IGNORE_TOOLS for t in tools):
        return ("every step is a meta / control-flow tool, which is not "
                "reusable as a bundled step")
    if any(t in MACRO_IGNORE_TOOLS for t in tools):
        return (f"step tool(s) {sorted(set(tools) & MACRO_IGNORE_TOOLS)} are "
                f"meta / control-flow tools — a macro that drives the macro "
                f"system, self-play, or the skill store is not a workflow")
    if len(set(tools)) == 1:
        return (f"every step is `{tools[0]}` — a single tool repeated is a "
                f"loop, not a composed skill")
    return None


def harvest_step_observations(trajectories, tool_sequence) -> List[List[dict]]:
    """Index-aligned argument sets for a name-only candidate sequence.

    `skills_auto`'s `SkillCandidate` carries only tool NAMES ("Arg-level
    consolidation is the consolidator's job" — skills_auto/extractor.py),
    which is why the graduation mint passed ``params: {}`` for every step
    and produced nine permanently unactivatable `auto_generic_*` macros.
    The arguments were never missing, only unread: they are on the same
    trajectories that phase already walked.

    Matches a trajectory only when its WHOLE named tool sequence equals
    ``tool_sequence`` — the candidate's identity is the whole chain, so a
    sub-window would pair position k with a different call. Only ``passed``
    trajectories count, matching the miner's own support rule.
    """
    seq = tuple(str(t) for t in (tool_sequence or ()))
    out: List[List[dict]] = [[] for _ in seq]
    if not seq:
        return out
    for traj in trajectories or ():
        if (getattr(traj, "outcome", "") or "") != "passed":
            continue
        named = [c for c in (getattr(traj, "tool_calls", None) or ())
                 if c is not None and (getattr(c, "name", "") or "").strip()]
        if tuple((c.name or "").strip() for c in named) != seq:
            continue
        for pos, call in enumerate(named):
            args = getattr(call, "arguments", None)
            out[pos].append(args if isinstance(args, dict) else {})
    return out


def mint_param_schema(tool_names, observations, *,
                      max_slots: int = MACRO_MAX_RUNTIME_SLOTS):
    """Build per-step ``param_template``s for a mined tool sequence.

    Parameters
    ----------
    tool_names
        Ordered tool names, one per step.
    observations
        One list per step, INDEX-ALIGNED across steps:
        ``observations[pos][k]`` is the argument dict the tool at ``pos``
        was called with on occurrence ``k``. The alignment is what lets
        two positions share one slot when they always carried the same
        value — "read then edit the SAME file" is one input, not two.

    Returns
    -------
    ``(templates, slots, reason)``. ``reason`` is ``None`` when the result
    is mintable; otherwise it names why and ``templates`` must not be used.
    """
    idx = _tool_schema_index()
    tool_names = list(tool_names or ())
    if idx is None:
        return [], [], "tool registry unreadable"
    n_steps = len(tool_names)
    if not n_steps:
        return [], [], "empty tool sequence"

    obs = [list(observations[i]) if i < len(observations) else []
           for i in range(n_steps)]
    # Occurrences are index-aligned across positions; a ragged sample set
    # would pair position 0's occurrence k with position 1's occurrence k
    # from a DIFFERENT window, so truncate to the common prefix.
    n_obs = min(len(o) for o in obs)
    obs = [[d if isinstance(d, dict) else {} for d in o[:n_obs]] for o in obs]

    literals: List[Dict[str, Any]] = [{} for _ in range(n_steps)]
    slot_groups: Dict[tuple, List[tuple]] = {}
    group_order: List[tuple] = []

    for pos, tool in enumerate(tool_names):
        spec = idx.get(tool) or {}
        enums = spec.get("enums") or {}
        selector = spec.get("selector")
        dicts = obs[pos]
        if not dicts:
            continue
        keys = set(dicts[0])
        for d in dicts[1:]:
            keys &= set(d)
        for key in sorted(keys):
            values = [d.get(key) for d in dicts]
            try:
                norm = tuple(json.dumps(v, sort_keys=True, default=str)
                             for v in values)
            except Exception:
                norm = tuple(repr(v) for v in values)
            v0 = values[0]
            constant = len(set(norm)) == 1
            if (constant and key == selector
                    and v0 in (enums.get(key) or ())):
                # THE MODE, and the only thing ever frozen: the value of
                # the tool's own primary dispatch selector, drawn from the
                # closed set the tool publishes.
                literals[pos][key] = v0
                continue
            # ⚠ REVIEW ROUND 2 REMOVED THE "DROP" BRANCH that used to sit
            # here for a constant flag or blank. Its comment claimed
            # "dropping is strictly the safe direction", and two of its
            # three examples (`port`, `limit`) were numbers this same fix
            # already routes to slots — it was defending a branch that had
            # been removed. Worse, the claim was FALSE for live booleans:
            # `browser.stop_on_error` defaults to False, so dropping an
            # observed True silently inverts a fail-fast interlock to
            # fail-open, and `browser.full_page` defaults True in the
            # other direction. Exactly one param in the whole registry
            # declares a schema `default`, so there was no sound basis for
            # deciding what an omission means. Everything that is not the
            # selector now becomes an explicit runtime slot: the caller
            # decides each time, and nothing is silently unset.
            gk = (key, norm)
            if gk not in slot_groups:
                slot_groups[gk] = []
                group_order.append(gk)
            slot_groups[gk].append((pos, key))

    slot_for: Dict[tuple, str] = {}
    used: set = set()
    for gk in group_order:
        base = _SLOT_KEY_RE.sub("_", str(gk[0]))[:40] or "arg"
        if not _SLOT_HEAD_RE.match(base):
            base = f"a_{base}"[:40]
        name, n = base, 1
        while name in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name)
        slot_for[gk] = name

    templates = [dict(literals[pos]) for pos in range(n_steps)]
    for gk, members in slot_groups.items():
        for pos, key in members:
            templates[pos][key] = f"${slot_for[gk]}"

    slots = sorted(used)
    if len(slots) > max_slots:
        return templates, slots, (
            f"needs {len(slots)} runtime inputs (max {max_slots})")
    for pos, tool in enumerate(tool_names):
        spec = idx.get(tool)
        if spec is None:
            # ⚠ REVIEW ROUND 1. This used to `continue`, on the reasoning
            # that an absent tool has "unknown requirements, nothing to
            # check". But `continue` skips ALL THREE refusal checks —
            # required-param coverage, the empty template, and the mode
            # identity — so a tool outside the registry was silently
            # exempt from every one of them. Live effect: `vision_analysis`
            # is appended by `get_active_tool_definitions`, not present in
            # the static `TOOL_DEFINITIONS`, so `browser:screenshot →
            # vision_analysis{$action,$target}` minted with step 2 fixing
            # no mode and asking the model to supply the tool's own action
            # — precisely the degenerate artifact the checks reject.
            # Registry membership was a PROXY for "we know this tool's
            # modes"; a check that cannot run must not report the
            # favourable outcome.
            return templates, slots, (
                f"step {pos + 1} ({tool}) is not in the tool registry, so "
                f"its required params and its modes cannot be checked")
        template = templates[pos]
        missing = [r for r in spec["required"] if r not in template]
        if missing:
            return templates, slots, (
                f"step {pos + 1} ({tool}) has neither a value nor a slot "
                f"for required param(s): {', '.join(missing)}")
        # ── The macro must have an IDENTITY ──────────────────────────
        # Coverage alone admits degenerate artifacts. Measured against the
        # live corpus 2026-08-23: of 47 sequences that passed the coverage
        # check, most minted as `file_system{$operation,$path} →
        # manage_services{$action}` — every mode selector left to the
        # caller. That is not a macro, it is a re-spelling of "call these
        # two tool types in this order", and the model will always reach
        # for the tools directly instead. Minting 47 of them into a
        # 50-entry registry with LRU-ish eviction would also push out the
        # macros that do mean something.
        if spec["params"] and not template:
            # The observations shared no key at all (e.g. some `execute`
            # calls carried `command`, others `filename`+`content`), so we
            # learned nothing about how this step is called.
            return templates, slots, (
                f"step {pos + 1} ({tool}) has an empty template: the "
                f"observed calls share no common argument")
        selector = spec.get("selector")
        if selector and not (selector in template
                             and not str(template[selector]).startswith("$")):
            return templates, slots, (
                f"step {pos + 1} ({tool}) fixes no mode: its dispatch "
                f"selector `{selector}` varied across observations, so the "
                f"macro has no identity")
        # ⚠ EVERY STEP MUST CARRY A RUNTIME INPUT. Review round 2, and it
        # is the CRITICAL of that round: a step with no slot is a call
        # FULLY DETERMINED AT MINT TIME, which is the literal definition
        # of the replay hazard the retired deny-list existed for, stated
        # at the step level. Confirmed mintable before this check:
        #
        #     file_system{operation:read, path:$path}
        #     knowledge_base{action:reset_all}     ← no runtime input
        #     manage_services{action:stop-all}     ← no runtime input
        #
        # `knowledge_base(action='reset_all')` deletes every id in the
        # vector store, truncates the library index and calls
        # `graph_memory.wipe_all()`, with no confirmation gate of any
        # kind; and a macro's steps are dispatched through
        # `build_step_executor`, which calls the tool function directly
        # and never reaches the turn loop's mutation classification. So a
        # zero-input mutating step has no guard above it at all.
        #
        # This is deliberately a STRUCTURAL rule and not a list of
        # dangerous verbs: every exemption in this project's deny-lists
        # became the next bypass (§4CI). The measured cost is one benign
        # read-only bundle (`introspect:summary → workspace:summary`),
        # which is the price of not having a list to leak.
        if not macro_step_inputs(template):
            return templates, slots, (
                f"step {pos + 1} ({tool}) takes NO runtime input: the call "
                f"would be fully determined at mint time, which is a replay")
    return templates, slots, None


@dataclass
class SkillStep:
    """A single step in a composed skill."""
    tool_name: str
    description: str
    param_template: Dict[str, str] = field(default_factory=dict)
    # If set, this step branches on a condition in the result
    branch_condition: str = ""  # e.g., "error" means branch if result contains error
    branch_target: str = ""     # Name of the alternative step sequence to follow
    optional: bool = False      # If True, failure doesn't abort the macro
    # DATA-FLOW (2026-07-11): bind this step's result to a name that LATER
    # steps can interpolate as `$name`. Without it a macro could only ever
    # substitute the macro's INITIAL params, so a step could never consume
    # the previous step's output — "fetch → transform → act on the fetched
    # value" was inexpressible and every real pipeline had to be driven
    # turn-by-turn by the main model.
    save_as: str = ""

    def to_dict(self) -> dict:
        d = {
            "tool_name": self.tool_name,
            "description": self.description,
            "param_template": self.param_template,
        }
        if self.branch_condition:
            d["branch_condition"] = self.branch_condition
            d["branch_target"] = self.branch_target
        if self.optional:
            d["optional"] = True
        if self.save_as:
            d["save_as"] = self.save_as
        return d


@dataclass
class ComposedSkill:
    """A reusable sequence of tool calls with conditional branching."""
    name: str
    trigger_description: str  # For semantic matching
    steps: List[SkillStep] = field(default_factory=list)
    branches: Dict[str, List[SkillStep]] = field(default_factory=dict)
    # "sequential" (default) runs steps in order with branching support;
    # "parallel" fans every step out concurrently and returns all results
    # — the right mode for an independent read-only bundle like a briefing.
    execution_mode: str = "sequential"
    # "active" macros are advertised to the LLM and dispatchable; "proposed"
    # macros are auto-discovered drafts (mined from the trajectory log by the
    # dream cycle) awaiting user approval — they are stored and listable but
    # deliberately NOT advertised or executable until approved via
    # manage_composed_skills(action="approve").
    status: str = "active"
    usage_count: int = 0
    success_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trigger_description": self.trigger_description,
            "steps": [s.to_dict() for s in self.steps],
            "branches": {k: [s.to_dict() for s in v] for k, v in self.branches.items()},
            "execution_mode": self.execution_mode,
            "status": self.status,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used,
            "created_at": self.created_at,
        }


class ComposedSkillRegistry:
    """Manages composed skills — discovery, storage, retrieval, and execution."""

    MAX_SKILLS = 50

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir
        self.skills: Dict[str, ComposedSkill] = {}
        self._save_lock = threading.Lock()
        if storage_dir:
            self._load()

    def _registry_path(self) -> Path:
        return self.storage_dir / "composed_skills.json" if self.storage_dir else Path("/dev/null")

    def _load(self):
        """Load composed skills from disk."""
        path = self._registry_path()
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load composed skills: %s", exc)
            return
        # Per-entry try so ONE malformed macro doesn't drop every macro
        # defined after it (single outer try aborted the whole load).
        for name, skill_data in (data or {}).items():
            try:
                # Validate on LOAD as well as on define: a legacy/hand-edited
                # entry with a dotted name (live registry still holds one)
                # would otherwise reach to_tool_definitions and emit an
                # invalid LLM function name for the whole catalogue.
                try:
                    _validate_composed_name(name)
                except ValueError as _nerr:
                    logger.warning("Quarantining composed skill with invalid "
                                   "name %r: %s", name, _nerr)
                    continue
                steps = [
                    SkillStep(**{k: v for k, v in s.items() if k in SkillStep.__dataclass_fields__})
                    for s in skill_data.get("steps", [])
                ]
                branches = {}
                for branch_name, branch_steps in skill_data.get("branches", {}).items():
                    branches[branch_name] = [
                        SkillStep(**{k: v for k, v in s.items() if k in SkillStep.__dataclass_fields__})
                        for s in branch_steps
                    ]
                self.skills[name] = ComposedSkill(
                    name=name,
                    trigger_description=skill_data.get("trigger_description", ""),
                    steps=steps,
                    branches=branches,
                    execution_mode=skill_data.get("execution_mode", "sequential"),
                    status=skill_data.get("status", "active"),
                    usage_count=skill_data.get("usage_count", 0),
                    success_count=skill_data.get("success_count", 0),
                    last_used=skill_data.get("last_used", 0),
                    created_at=skill_data.get("created_at", time.time()),
                )
            except Exception as exc:
                logger.warning("Skipping malformed composed skill %r: %s", name, exc)

    def save(self):
        """Persist composed skills to disk. Atomic (temp + os.replace) under a
        lock so a concurrent dream-cycle register / a macro's record_usage save
        can't interleave and truncate/corrupt the registry file."""
        if not self.storage_dir:
            return
        path = self._registry_path()
        try:
            # mkdir INSIDE the try: on a read-only/full volume it raised
            # through record_usage → execute(), discarding a macro run's real
            # results, contradicting this method's swallow-and-warn design.
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            data = {name: skill.to_dict() for name, skill in self.skills.items()}
            with self._save_lock:
                tmp = path.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to save composed skills: %s", exc)

    def register(self, skill: ComposedSkill) -> bool:
        """Register a new composed skill."""
        # Merge-don't-demote on re-register (belt-and-braces below the
        # compile_from_pattern no-op guard): an unconditional overwrite
        # reverted an approved macro to "proposed" and zeroed its stats.
        existing = self.skills.get(skill.name)
        if existing is not None:
            if existing.status == "active" and skill.status != "active":
                skill.status = "active"
            skill.usage_count = max(skill.usage_count, existing.usage_count)
            skill.success_count = max(skill.success_count, existing.success_count)
            skill.last_used = max(skill.last_used, existing.last_used)
        # Only evict when ADDING a genuinely new name — re-registering an
        # existing macro doesn't grow the count, so it must not evict a
        # bystander.
        if skill.name not in self.skills and len(self.skills) >= self.MAX_SKILLS:
            # Evict proposed (unapproved) drafts before any active macro,
            # then by lowest usage — so a flood of auto-proposals can never
            # push out a macro the user actually approved or uses.
            worst = min(
                self.skills.values(),
                key=lambda s: (s.status == "active", s.usage_count),
            )
            del self.skills[worst.name]
            logger.info(
                "Evicted composed skill '%s' (status=%s, usage=%d)",
                worst.name, worst.status, worst.usage_count,
            )

        self.skills[skill.name] = skill
        self.save()
        # DEBUG: re-registration fires on every rebuild (~119x identical lines
        # in the durable log); keep it greppable in debug, out of the signal.
        logger.debug("Registered composed skill: %s (%d steps)", skill.name, len(skill.steps))
        return True

    # NOTE: a keyword-overlap `find_matching(query)` used to live here — it
    # had zero callers in the runtime (macro discovery happens via the tool
    # definitions the LLM sees), so it was removed 2026-07-14 as dead code.

    def record_usage(self, skill_name: str, success: bool):
        """Record that a composed skill was used."""
        if skill_name in self.skills:
            skill = self.skills[skill_name]
            skill.usage_count += 1
            if success:
                skill.success_count += 1
            skill.last_used = time.time()
            self.save()

    def compile_from_pattern(self, pattern_name: str,
                             tool_sequence: List[Dict[str, Any]],
                             description: str,
                             *,
                             status: str = "proposed",
                             execution_mode: str = "sequential") -> ComposedSkill:
        """Compile a detected tool-call pattern into a ComposedSkill.

        Called by the dream cycle when it mines a recurring tool-call
        sequence from the trajectory log. Defaults to ``status="proposed"``
        — an auto-discovered draft that is stored and listable but NOT
        advertised to the LLM or dispatchable until the user approves it via
        ``manage_composed_skills(action="approve")``.
        """
        # Sanitize into a legal tool identifier. Minted skills-auto names
        # arrive DOTTED (auto.<cluster>.<head>.<sha6>) and used to flow
        # straight into the registry — on approval a dotted name entered
        # the LLM function catalogue, violating this module's own naming
        # contract (live: auto.generic.manage_services_manage_services.c73e69).
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", str(pattern_name or "skill"))
        if not re.match(r"^[A-Za-z_]", safe_name):
            safe_name = f"m_{safe_name}"
        safe_name = _validate_composed_name(safe_name[:64])

        # Re-minting an existing macro must be a no-op, not an overwrite:
        # the phase-2.6 mint used to re-register on every re-graduation of
        # the same candidate, demoting an operator-APPROVED macro back to
        # "proposed" (vanishing from the tool list) and wiping its stats.
        existing = self.skills.get(safe_name)
        if existing is not None:
            self._upgrade_paramless(existing, tool_sequence)
            return existing

        steps = []
        for i, entry in enumerate(tool_sequence):
            steps.append(SkillStep(
                tool_name=entry.get("tool", "unknown"),
                description=entry.get("description", f"Step {i+1}"),
                param_template=entry.get("params", {}) or {},
            ))
        skill = ComposedSkill(
            name=safe_name,
            trigger_description=description,
            steps=steps,
            execution_mode=execution_mode,
            status=status,
        )
        self.register(skill)
        return skill

    def _upgrade_paramless(self, existing: "ComposedSkill",
                           tool_sequence: List[Dict[str, Any]]) -> bool:
        """Give a still-PROPOSED macro the runtime slots it was minted without.

        §4CS. Re-minting an existing macro is a no-op BY DESIGN — it must
        never demote an operator-approved macro back to "proposed" and wipe
        its stats. The side effect was that the 25 macros minted before the
        param-schema mint existed could never acquire one: they were parked
        with empty templates, advertising zero inputs, permanently
        unactivatable.

        This is one MONOTONE upgrade step. A macro that is not active and
        carries ZERO ``$slot`` references adopts the incoming templates iff
        those carry at least one, and iff the step tools still match
        position-for-position. An ACTIVE macro is never touched; a macro
        that already has slots is never rewritten. So it is idempotent and
        it terminates.
        """
        if existing.status == "active":
            return False
        if any(macro_step_inputs(s.param_template) for s in existing.steps):
            return False
        entries = list(tool_sequence or ())
        if len(entries) != len(existing.steps):
            return False
        incoming = [(e or {}).get("params") or {} for e in entries]
        if not any(macro_step_inputs(p) for p in incoming):
            return False
        for step, entry in zip(existing.steps, entries):
            if step.tool_name != (entry or {}).get("tool"):
                return False
        for step, entry, params in zip(existing.steps, entries, incoming):
            step.param_template = dict(params)
            desc = (entry or {}).get("description")
            if desc:
                step.description = desc
        self.save()
        pretty_log(
            "Macro Schema",
            f"'{existing.name}' was minted param-less and could not be "
            f"activated; it now carries runtime slots "
            f"({', '.join('$' + s for s in sorted({n for st in existing.steps for n in macro_step_inputs(st.param_template)}))}). "
            f"Still status=proposed — approve it with "
            f"manage_composed_skills(action='approve').",
            icon=Icons.BRAIN_PLAN,
        )
        return True

    def to_tool_definitions(self) -> List[dict]:
        """Render each registered composed skill as an LLM-facing tool definition.

        Mirrors the shape used by the static TOOL_DEFINITIONS list in
        registry.py so callers can simply `.extend()` the agent's active
        tool list.
        """
        defs: List[dict] = []
        for name, skill in self.skills.items():
            # Proposed (auto-discovered, unapproved) macros are NOT shown to
            # the LLM — they await user approval first.
            if skill.status != "active":
                continue
            # Referenced names minus names BOUND by an earlier step's
            # `save_as`: an internally-produced value is not a runtime
            # param, so advertising it would ask the LLM to supply
            # something the pipeline computes for itself. Branch steps are
            # mined too (they run after the main steps, so main bindings
            # count as produced for them) — a runtime param used ONLY inside
            # a branch was previously never advertised.
            param_keys: set = set()

            def _mine(seq, produced_seed):
                produced = set(produced_seed)
                for step in seq:
                    if isinstance(step.param_template, dict):
                        for v in step.param_template.values():
                            if isinstance(v, str) and "$" in v:
                                for mo in _VAR_RE.finditer(v):
                                    nm = mo.group(1) or mo.group(2)
                                    if nm not in produced:
                                        param_keys.add(nm)
                    if step.save_as:
                        produced.add(step.save_as)
                return produced

            main_produced = _mine(skill.steps, set())
            for _bsteps in skill.branches.values():
                _mine(_bsteps, main_produced)
            param_keys -= main_produced
            properties = {
                k: {"type": "string", "description": f"Runtime value for ${k}."}
                for k in sorted(param_keys)
            }
            schema = {
                "type": "object",
                "properties": properties,
                # A macro's mined runtime params ARE its inputs — every one is a
                # `$var` the pipeline references but no step produces, so the
                # macro cannot run without them. Marking them required stops the
                # weak worker model from firing the macro with an empty arg set
                # (observed 2026-08-12: `url` advertised optional, the model
                # never supplied it and reached for the management tool instead).
                "required": sorted(param_keys),
            }
            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"[COMPOSED SKILL] {skill.trigger_description} "
                        f"To RUN it, CALL THIS TOOL DIRECTLY by name with its "
                        f"inputs — do NOT define/re-create it and do NOT plan a "
                        f"manage_composed_skills call to run it. "
                        f"({len(skill.steps)} steps; "
                        f"used {skill.usage_count}x with {skill.success_rate:.0%} success)"
                    ),
                    "parameters": schema,
                }
            })
        return defs

    @staticmethod
    def _resolve_args(step: "SkillStep", params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a step's `$variable` param templates.

        ``params`` is the live binding scope: the macro's initial runtime
        params PLUS any earlier step's result bound via its ``save_as``
        (see ``_execute_sequential``) — that union is what makes real
        pipelines expressible ("fetch → transform → act on the fetched
        value"). Two forms:

        * whole-value  — ``"$var"`` substitutes the binding as-is;
        * interpolated — ``"summarize: $var"`` / ``"${var}"`` substitutes
          into surrounding text (so a step can wrap a prior result in a
          prompt or a shell command).

        An UNresolved name becomes ``""`` (a missing value), never the
        literal ``"$var"`` — otherwise the tool receives ``location="$city"``.
        """
        resolved_args = {}
        for k, v in step.param_template.items():
            if not isinstance(v, str) or "$" not in v:
                resolved_args[k] = v
                continue
            m = _WHOLE_VAR_RE.fullmatch(v.strip())
            if m:
                # Whole-value: keep the binding's native type (a step could
                # bind a non-str via save_as in a future caller).
                resolved_args[k] = params.get(m.group(1), "")
            else:
                def _sub(mo):
                    # Two alternations (${var} | $var) — exactly one group hits.
                    name = mo.group(1) or mo.group(2)
                    return str(params.get(name, ""))
                resolved_args[k] = _VAR_RE.sub(_sub, v)
        return resolved_args

    async def execute(self, skill_name: str,
                      executor: Callable,
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a composed skill using the provided tool executor.

        Dispatches on the skill's ``execution_mode``: ``"parallel"`` fans
        all steps out concurrently (ideal for an independent read-only
        bundle like a morning briefing), ``"sequential"`` (default) runs
        them in order with conditional branching.

        Parameters
        ----------
        skill_name : name of the composed skill
        executor : async callable(tool_name, tool_args) -> result_str
        params : runtime parameters to fill templates

        Returns
        -------
        Dict with 'success', 'results', 'steps_completed', 'total_steps'
        and 'mode' keys.
        """
        if skill_name not in self.skills:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        skill = self.skills[skill_name]
        params = params or {}
        if skill.execution_mode == "parallel":
            return await self._execute_parallel(skill, executor, params)
        return await self._execute_sequential(skill, executor, params)

    async def _execute_sequential(self, skill: "ComposedSkill",
                                  executor: Callable,
                                  params: Dict[str, Any]) -> Dict[str, Any]:
        """Run steps in order, honouring conditional branches and optional steps.

        Maintains a live BINDING SCOPE (`scope`): the macro's initial params
        plus every completed step's result bound under its ``save_as`` name.
        Later steps interpolate those as ``$name`` — this is what makes a
        macro a real pipeline rather than a fixed list of independent calls
        (2026-07-11). A step's own args are resolved against the scope as it
        stands when that step runs, so bindings only ever flow FORWARD.
        """
        results = []
        success = True

        # Copy: never mutate the caller's params dict (the same dict can be
        # reused across executions of the same macro).
        scope = dict(params)
        bound: List[str] = []

        active_steps = list(skill.steps)
        step_idx = 0
        # Bound total step executions so a self-referential branch (hand-
        # authored/loaded branches JSON) can't loop forever issuing tool calls.
        _MAX_STEP_EXECUTIONS = 64
        _executions = 0

        while step_idx < len(active_steps):
            if _executions >= _MAX_STEP_EXECUTIONS:
                results.append({
                    "step": "(aborted)", "tool": "-",
                    "error": f"step-execution cap ({_MAX_STEP_EXECUTIONS}) hit — "
                             "possible branch loop.",
                    "success": False,
                })
                success = False
                break
            _executions += 1
            step = active_steps[step_idx]
            resolved_args = self._resolve_args(step, scope)

            try:
                result = await executor(step.tool_name, resolved_args)
                result_str = str(result)
                # Classify from the RESULT (tools return error strings, not
                # raises). Pass the OBJECT, not `result_str`: stringifying it
                # one line earlier discarded the status and made
                # `_step_result_ok`'s status arm unreachable from here —
                # browser FAILED, memory PARTIAL, swarm UNRESOLVED and
                # file_system PARTIAL all counted as successful steps and
                # inflated the macro success_rate the model is shown.
                step_ok = _step_result_ok(result)
                # Bind BEFORE the failure branch so an `optional` step that
                # failed still exposes its (error) output to later steps that
                # deliberately reference it — and a downstream step gets ""
                # rather than a stale binding from a previous execution.
                if step.save_as:
                    bound_val = result_str
                    if len(bound_val) > MAX_BOUND_VALUE_CHARS:
                        # NEVER truncate silently (same policy as the display
                        # cap): a downstream step consuming a cut-off body
                        # must be able to SEE it was cut, not act on partial
                        # data believing it is complete.
                        bound_val = (
                            bound_val[:MAX_BOUND_VALUE_CHARS]
                            + f"\n…[binding truncated: this step's output "
                              f"exceeded the {MAX_BOUND_VALUE_CHARS}-char "
                              f"save_as cap]"
                        )
                    scope[step.save_as] = bound_val
                    if step.save_as not in bound:
                        bound.append(step.save_as)
                results.append({
                    "step": step.description,
                    "tool": step.tool_name,
                    "result": _cap_step_result(result_str),
                    "success": step_ok,
                    **({"saved_as": step.save_as} if step.save_as else {}),
                })
                if not step_ok:
                    if not step.optional:
                        success = False
                        break
                    step_idx += 1
                    continue

                # Check branch condition (only on a genuinely successful step)
                if step.branch_condition and step.branch_condition.lower() in result_str.lower():
                    branch_steps = skill.branches.get(step.branch_target, [])
                    if branch_steps:
                        active_steps = branch_steps
                        step_idx = 0
                        continue
            except Exception as exc:
                results.append({
                    "step": step.description,
                    "tool": step.tool_name,
                    "error": str(exc),
                    "success": False,
                })
                if not step.optional:
                    success = False
                    break

            step_idx += 1

        self.record_usage(skill.name, success)
        out = {
            "success": success,
            "results": results,
            "steps_completed": len(results),
            "total_steps": len(skill.steps),
            "mode": "sequential",
        }
        if bound:
            out["bound"] = bound
        return out

    async def _execute_parallel(self, skill: "ComposedSkill",
                                executor: Callable,
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Fan every step out concurrently, then collect all results.

        Branching does NOT apply in parallel mode — there is no ordered
        result to test a `branch_condition` against, so branches are
        ignored. Every step runs even if a sibling fails; a non-optional
        step failing marks the whole macro failed but never aborts the
        fan-out (we want the briefing's other panels regardless).

        `save_as` DATA-FLOW likewise does not apply: steps start
        simultaneously, so no step can observe a sibling's result. Bindings
        are ignored here rather than silently resolving to "" — the
        validation path (`_validate_dataflow`) rejects a parallel macro that
        references a step-produced name, so this can't be authored by
        accident.

        Fan-out is BOUNDED by a semaphore: most steps land on the single-slot
        local llama box (searches, LLM-backed tools), and an unbounded
        20-step macro would stampede it. Steps beyond the window queue and
        start as slots free up — all of them still run.
        """
        sem = asyncio.Semaphore(_PARALLEL_STEP_CONCURRENCY)

        async def _run_step(step: "SkillStep") -> Dict[str, Any]:
            async with sem:
                return await self._run_parallel_step(step, executor, params)

        results = list(await asyncio.gather(*[_run_step(s) for s in skill.steps]))
        # A step is "tolerated" if it succeeded or was declared optional.
        success = all(r["success"] or r.get("optional") for r in results)

        self.record_usage(skill.name, success)
        return {
            "success": success,
            "results": results,
            "steps_completed": len(results),
            "total_steps": len(skill.steps),
            "mode": "parallel",
        }

    async def _run_parallel_step(self, step: "SkillStep",
                                 executor: Callable,
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Run ONE parallel-mode step and classify its result (tools return
        error strings, not raises)."""
        resolved_args = self._resolve_args(step, params)
        try:
            result = await executor(step.tool_name, resolved_args)
            result_str = str(result)
            return {
                "step": step.description,
                "tool": step.tool_name,
                "result": _cap_step_result(result_str),
                # the OBJECT, not `result_str` — stringifying it discards the
                # status and makes `_step_result_ok`'s status arm unreachable
                "success": _step_result_ok(result),
                "optional": step.optional,
            }
        except Exception as exc:
            return {
                "step": step.description,
                "tool": step.tool_name,
                "error": str(exc),
                "success": False,
                "optional": step.optional,
            }


def _registry_from_context(context) -> Optional[ComposedSkillRegistry]:
    """Build (or fetch a cached) ComposedSkillRegistry from a context object.

    The registry caches the loaded skills in-memory so we don't re-read
    the JSON file on every tool-list build. We attach the cached instance
    on the context under `_composed_skill_registry` (best-effort — if the
    context is a frozen MagicMock or otherwise rejects assignment we
    silently fall back to building a fresh registry).
    """
    if context is None:
        return None
    cached = getattr(context, "_composed_skill_registry", None)
    if isinstance(cached, ComposedSkillRegistry):
        return cached
    # Prefer memory_dir so macros persist across sandbox wipes (same
    # rationale as acquired skills); fall back to sandbox_dir for
    # early-init contexts that haven't wired memory_dir yet.
    base = getattr(context, "memory_dir", None) or getattr(context, "sandbox_dir", None)
    if base is None:
        return None
    try:
        storage_dir = Path(base) / "composed_skills"
    except Exception:
        return None
    reg = ComposedSkillRegistry(storage_dir=storage_dir)
    try:
        setattr(context, "_composed_skill_registry", reg)
    except Exception:
        # If the context refuses attribute assignment we just rebuild
        # next time — correctness is preserved, perf takes a small hit.
        pass
    return reg


def register_composed_skills(tool_definitions: list, context) -> int:
    """Mutate ``tool_definitions`` in-place to include composed-skill entries.

    Mirrors how acquired-skill definitions are appended in
    `registry.get_active_tool_definitions`. Returns the number of
    composed-skill entries added.

    Composed-skill names are skipped if they would shadow an existing
    tool entry — same shadowing policy as acquired skills.
    """
    if not isinstance(tool_definitions, list):
        return 0
    reg = _registry_from_context(context)
    if reg is None:
        return 0
    existing_names = {
        t.get("function", {}).get("name")
        for t in tool_definitions
        if isinstance(t, dict)
    }
    # Shadow against the FULL acquired-skill registry, not just the defs
    # present this turn: semantic routing filters acquired defs per query,
    # so on a turn where a same-named acquired def was routed OUT, the
    # composed def was advertised while dispatch (which registers ALL
    # acquired runners) executed the ACQUIRED skill — advertised schema ≠
    # executed tool.
    try:
        from .acquired_skills import AcquiredSkillManager
        _base = getattr(context, "memory_dir", None) or getattr(context, "sandbox_dir", None)
        if _base is not None:
            _mgr = AcquiredSkillManager.get_shared(
                _base, None, legacy_sandbox_dir=getattr(context, "sandbox_dir", None))
            existing_names |= set(_mgr.get_all_skills().keys())
    except Exception as e:
        logger.debug("composed-shadow acquired-registry read skipped: %s", e)
    added = 0
    for entry in reg.to_tool_definitions():
        name = entry.get("function", {}).get("name")
        if not name or name in existing_names:
            if name:
                logger.warning(
                    "Composed skill '%s' shadows an existing tool — skipping.",
                    name,
                )
            continue
        tool_definitions.append(entry)
        existing_names.add(name)
        added += 1
    if added:
        logger.info("Registered %d composed skill(s) into tool definitions.", added)
    return added


def _format_execution_result(skill_name: str, result: Dict[str, Any]) -> str:
    """Render an `execute()` result dict as a compact, LLM-readable string.

    This is what the agent's dispatch loop hands back as the tool result;
    the model then synthesises the briefing/answer from the per-step
    blocks. Each step's body is already bounded to ``MAX_STEP_RESULT_CHARS``
    by `execute()` (via `_cap_step_result`), so a chatty step can't blow the
    context budget — and any step that DID hit the cap carries an explicit
    truncation marker, so a list-bearing step is never silently shortened.
    """
    # Guard-style failures (unknown skill) carry an 'error' and no 'results'.
    if not result.get("success") and "error" in result and "results" not in result:
        return f"[composed skill '{skill_name}' error] {result.get('error')}"

    mode = result.get("mode", "sequential")
    header = (
        f"COMPOSED SKILL '{skill_name}' — "
        f"{result.get('steps_completed', 0)}/{result.get('total_steps', 0)} steps "
        f"({mode}), overall {'OK' if result.get('success') else 'PARTIAL/FAIL'}."
    )
    blocks = [header]
    for i, r in enumerate(result.get("results", []), 1):
        head = f"[{i}] {r.get('tool')} — {r.get('step')}"
        if r.get("success"):
            blocks.append(f"{head}:\n{r.get('result', '')}")
        else:
            opt = " (optional)" if r.get("optional") else ""
            # Fall back to the RESULT body: a step whose tool returned an
            # error STRING (the codebase norm) carries no "error" key, so the
            # model saw "FAILED — unknown error" with zero diagnostic and
            # could not recover or re-route.
            detail = r.get("error") or r.get("result") or "unknown error"
            blocks.append(f"{head}: FAILED{opt} — {detail}")
    return "\n\n".join(blocks)


def build_step_executor(tools_ref: Dict[str, Callable], composed_names) -> Callable:
    """Return an async ``(tool_name, args) -> result_str`` dispatcher that
    runs a single composed-skill STEP against the agent's live tool dict.

    ``tools_ref`` is the same dict ``get_available_tools`` builds — captured
    by reference, so it is fully populated by the time a step actually runs.
    ``composed_names`` is the set of composed-skill names; steps are
    FORBIDDEN from invoking another composed skill, which is what stops a
    macro from recursing into itself (or a cycle of macros) and blowing the
    stack.
    """
    composed = set(composed_names or ())

    async def _exec_step(tool_name: str, tool_args: Dict[str, Any]):
        if tool_name in composed:
            return (
                f"[blocked] '{tool_name}' is itself a composed skill; "
                f"composed skills cannot be nested as steps."
            )
        fn = tools_ref.get(tool_name)
        if fn is None:
            return f"[error] step tool '{tool_name}' is not available."
        return await fn(**(tool_args or {}))

    return _exec_step


def make_composed_skill_runner(skill_name: str, registry: "ComposedSkillRegistry",
                               tools_ref: Dict[str, Callable], composed_names) -> Callable:
    """Build the top-level tool runner for one composed skill.

    The returned coroutine is what ``get_available_tools`` registers under
    the macro's name. Calling it fans the macro's steps out through
    ``build_step_executor`` and returns a formatted, LLM-readable summary.
    """
    async def _run(**kwargs):
        skill = registry.skills.get(skill_name)
        nsteps = len(skill.steps) if skill else "?"
        mode = skill.execution_mode if skill else "?"
        pretty_log(
            "Composed Skill",
            f"Running macro '{skill_name}' ({nsteps} steps, {mode}).",
            icon=Icons.BRAIN_PLAN,
        )
        executor = build_step_executor(tools_ref, composed_names)
        result = await registry.execute(skill_name, executor, params=kwargs)
        return _format_execution_result(skill_name, result)

    return _run


def register_composed_skill_runners(tools: Dict[str, Callable], context) -> int:
    """Mutate the ``tools`` executor dict in-place to add a runner for each
    registered composed skill. Counterpart to ``register_composed_skills``
    (which adds the LLM-facing DEFINITIONS) — together they make a macro
    both visible to the model AND dispatchable.

    A macro name is skipped if it would shadow a built-in / acquired-skill
    runner already in ``tools`` (same shadow policy as the definition side),
    which keeps a macro from hijacking a real tool name.
    """
    if not isinstance(tools, dict):
        return 0
    reg = _registry_from_context(context)
    if reg is None or not reg.skills:
        return 0
    composed_names = set(reg.skills.keys())
    added = 0
    for name, skill in reg.skills.items():
        # Proposed (auto-discovered, unapproved) drafts are not dispatchable
        # until the user approves them.
        if skill.status != "active":
            continue
        if name in tools:
            logger.warning(
                "Composed skill '%s' shadows an existing tool runner — skipping.",
                name,
            )
            continue
        tools[name] = make_composed_skill_runner(name, reg, tools, composed_names)
        added += 1
    if added:
        logger.info("Wired %d composed-skill runner(s) into the tool dispatch.", added)
    return added


async def tool_manage_composed_skills(context=None, action: str = None,
                                      name: str = None, description: str = None,
                                      steps=None, mode: str = "parallel",
                                      known_tools=None, branches=None,
                                      params=None, **_extra):
    """Define / run / list / approve / delete composed skills — named macros
    that bundle several tool calls into ONE invocation.

    Actions
    -------
    define : register a new macro. ``steps`` is a list of
        ``{tool, description, params, optional, save_as, branch_condition,
        branch_target}`` objects. ``mode`` is ``"parallel"`` (default — fan
        out independent steps) or ``"sequential"`` (ordered; required for
        save_as data-flow and branching). ``branches`` (optional, sequential
        only) maps a branch name to its own step list; a step whose result
        contains its ``branch_condition`` substring jumps to the
        ``branch_target`` sequence. The macro becomes a top-level tool the
        agent invokes by ``name``. An existing name is NOT overwritten — the
        refusal points at ``run`` (active) or ``approve`` (proposed);
        replacement stays explicit: delete, then define.
    run    : execute an existing active macro by ``name``, passing its runtime
        inputs in ``params`` (e.g. params={'url': '…'}). This is the
        management-tool path for the same thing a direct ``name(...)`` call
        does — added because the worker model reaches for this tool instead of
        calling the macro by name (2026-08-12 postmortem). Dispatches through
        the SAME runner a direct call uses, so there is one execution path.
    list   : show all registered macros.
    approve: activate a proposed (auto-discovered) macro.
    delete : remove one by ``name``.
    """
    if not action:
        return "SYSTEM ERROR: 'action' is MANDATORY (define | run | list | approve | delete)."
    reg = _registry_from_context(context)
    if reg is None:
        return ("SYSTEM ERROR: composed-skill storage is unavailable "
                "(no sandbox/memory dir on the active context).")
    action = str(action).strip().lower()  # str() so a non-string can't raise

    if action == "list":
        if not reg.skills:
            return "No composed skills defined yet."
        active = [(n, sk) for n, sk in reg.skills.items() if sk.status == "active"]
        proposed = [(n, sk) for n, sk in reg.skills.items() if sk.status != "active"]
        out = ["Composed skills (macros):"]
        for n, sk in active:
            out.append(
                f"- {n} [{sk.execution_mode}] — {sk.trigger_description} "
                f"({len(sk.steps)} steps; used {sk.usage_count}x, "
                f"{sk.success_rate:.0%} ok)"
            )
        if not active:
            out.append("(none active)")
        if proposed:
            out.append("")
            out.append("Proposed (auto-discovered from your tool-use history — approve to activate):")
            for n, sk in proposed:
                seq = " → ".join(s.tool_name for s in sk.steps)
                out.append(f"- {n} [proposed, {sk.execution_mode}] — {sk.trigger_description} (steps: {seq})")
            out.append("")
            out.append(
                "Approve with manage_composed_skills(action='approve', name='<name>'); "
                "reject with action='delete'."
            )
        return "\n".join(out)

    if action == "run":
        if not name:
            return ("Error: 'name' is required for run — the macro to execute. "
                    "See the available macros with action='list'.")
        sk = reg.skills.get(name)
        if sk is None:
            return (f"Error: composed skill '{name}' not found. Check the name "
                    f"with action='list', or define it first with "
                    f"action='define'.")
        if sk.status != "active":
            return (f"Error: composed skill '{name}' is not active "
                    f"(status={sk.status}) — approve it first: "
                    f"manage_composed_skills(action='approve', name='{name}').")
        # Runtime inputs: the documented shape is params={...}, but a weak
        # model often passes them as bare top-level kwargs (url='…'), which
        # land in **_extra. Accept both; the explicit `params` object wins on
        # a key clash. `_extra` here is ALREADY free of the tool's control
        # kwargs (action/name/description/steps/mode/known_tools/branches are
        # named parameters), so it only carries genuine macro inputs.
        run_params: Dict[str, Any] = dict(_extra or {})
        if isinstance(params, dict):
            run_params.update(params)
        elif params is not None:
            return ("Error: 'params' must be an object mapping the macro's "
                    "runtime inputs to values, e.g. params={'url': '…'}.")
        # Dispatch through the SAME live runner a direct `name(...)` call would
        # use — no duplicate execution logic. Building the tool map here also
        # picks up dynamically-appended tools and acquired skills the macro's
        # steps may call. Lazy import: registry imports THIS module.
        try:
            from .registry import get_available_tools
            tools_map = get_available_tools(context)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the turn
            return (f"Error: could not assemble the tool set to run '{name}': "
                    f"{type(e).__name__}: {e}")
        runner = tools_map.get(name)
        if runner is None:
            return (f"Error: composed skill '{name}' is registered but not "
                    f"dispatchable in this build (it may shadow a built-in "
                    f"tool of the same name). Rename the macro or invoke the "
                    f"built-in directly.")
        pretty_log(
            "Macro Run",
            f"Invoking '{name}' via manage_composed_skills(action='run')"
            + (f" with {sorted(run_params)}" if run_params else ""),
            icon=Icons.BRAIN_PLAN,
        )
        return await runner(**run_params)

    if action == "approve":
        if not name:
            return "Error: 'name' is required for approve."
        if name not in reg.skills:
            return f"Error: composed skill '{name}' not found."
        sk = reg.skills[name]
        if sk.status == "active":
            return f"Composed skill '{name}' is already active."
        # Re-validate before activation: `define` validates, but a macro that
        # entered the registry another way (skills-auto mint, hand-edited
        # file) could otherwise be flipped active with a name that is illegal
        # as an LLM function name.
        try:
            _validate_composed_name(name)
        except ValueError as ve:
            return (f"Error: cannot approve '{name}' — {ve} Delete it and "
                    f"re-define it under a valid name.")
        sk.status = "active"
        reg.save()
        pretty_log("Macro Approved", f"Activated proposed macro: {name}", icon=Icons.OK)
        # Say what is actually true about the params: graduation-minted
        # macros carry EMPTY param templates (the miner knows tool order,
        # not per-step args), so the old blanket "mined from past calls"
        # claim was false on that path and the approved macro surprised the
        # operator by demanding args at run time.
        # ⚠ REVIEW ROUND 2: `_has_params` is now TRUE for every minted
        # macro (§4CS gives each step a schema), so the "mined from past
        # calls" branch fired for macros whose templates are `$slots`, not
        # mined values — and the branch that tells the operator inputs are
        # required at invocation became unreachable for exactly the macros
        # that require them. Three states, not two.
        _slots = sorted({n for st in sk.steps
                         for n in macro_step_inputs(
                             getattr(st, "param_template", None))})
        _has_params = any(getattr(st, "param_template", None) for st in sk.steps)
        _param_note = (
            f"It takes {len(_slots)} runtime input(s): "
            + ", ".join("$" + n for n in _slots)
            + ". Those are SLOTS, not values mined from past calls — supply "
              "them when you call the macro."
            if _slots else
            "Its step parameters are fixed values with no runtime inputs; "
            "delete + redefine if you want to adjust them."
            if _has_params else
            "NOTE: its step parameter templates are EMPTY (auto-graduated "
            "sequences carry tool order only) — each step's mandatory "
            "params must be supplied at invocation, or delete + redefine "
            "with concrete params."
        )
        return (
            f"Success: composed skill '{name}' approved and activated. It is now "
            f"a top-level tool — invoke it by name. ({_param_note})"
        )

    if action == "delete":
        if not name:
            return "Error: 'name' is required for delete."
        if name not in reg.skills:
            return f"Error: composed skill '{name}' not found."
        del reg.skills[name]
        reg.save()
        pretty_log("Macro Forgotten", f"Deleted composed skill: {name}", icon=Icons.MEM_WIPE)
        return f"Success: composed skill '{name}' deleted."

    if action == "define":
        if not name or not description or not steps:
            return ("SYSTEM ERROR: 'name', 'description', and 'steps' are "
                    "MANDATORY for define.")
        try:
            name = _validate_composed_name(name)
        except ValueError as ve:
            return f"Error: {ve}"
        # No silent overwrite: `register()` replaces the object, which also
        # resets usage/success stats — a name typo could clobber a tuned
        # macro. Replacement must be explicit: delete, then define. The
        # message leads with the EXECUTION path (2026-08-12 postmortem: a
        # run-intent model kept landing on `define`, and the old delete-first
        # wording steered it into deleting an active macro it only wanted to
        # invoke); delete is framed as the replace-only recovery, and a
        # not-yet-active duplicate points at `approve` instead of `run`.
        #
        # ORDER MATTERS (2026-08-12 fresh-eye review): this duplicate check
        # must run BEFORE the known_tools shadow check below. The production
        # call site derives known_tools from the fully-populated dispatch
        # table, which register_composed_skill_runners has already seeded
        # with every ACTIVE macro's own name — so with the shadow check
        # first, a duplicate define of an active macro was misdiagnosed as
        # a built-in collision ("choose a different name"), and this branch
        # was unreachable exactly in the postmortem scenario it exists for.
        # The registry's own macro is not "shadowing" a tool; it IS the tool.
        #
        # The execution guidance leads with action='run' rather than the
        # direct call: run re-verifies dispatchability itself (its
        # "registered but not dispatchable" branch), so the FIRST
        # recommendation stays truthful even in the rare active-but-shadowed
        # state (a same-named built-in/acquired tool appearing after the
        # macro was defined — the runner wiring lets the built-in win).
        # Volatile counters (steps/usage) sit at the message TAIL, outside
        # RecentFailureGuard's 80-char normalized prefix: for short macro
        # names a mid-message usage counter made two identical failures
        # normalize differently, so the repeat guard could never accumulate
        # a match.
        if name in reg.skills:
            existing = reg.skills[name]
            if existing.status == "active":
                how_to_use = (
                    f"To EXECUTE it, use manage_composed_skills("
                    f"action='run', name='{name}', params={{...}}), or call "
                    f"the tool '{name}' directly. Do NOT re-define it."
                )
                head = (f"Error: composed skill '{name}' already exists and "
                        f"is ACTIVE.")
            else:
                how_to_use = (
                    f"Activate it with manage_composed_skills("
                    f"action='approve', name='{name}'), then call it "
                    f"directly."
                )
                head = (f"Error: composed skill '{name}' already exists but "
                        f"is not active (status={existing.status}).")
            return (f"{head} {how_to_use} Only if you need to REPLACE its "
                    f"definition: delete it first (action='delete', "
                    f"name='{name}'), then define. "
                    f"({len(existing.steps)} steps, used "
                    f"{existing.usage_count}x)")
        # Reject a name that shadows a built-in / acquired tool: the runner
        # wiring skips such a macro (the built-in wins), so persisting it and
        # telling the model "it's now a TOP-LEVEL TOOL" is a lie. Reached
        # only for names that are NOT the registry's own (see order note
        # above).
        if known_tools and name in known_tools:
            return (f"Error: '{name}' is already a built-in/acquired tool; a "
                    "composed skill can't shadow it. Choose a different name.")
        if not isinstance(steps, list) or not steps:
            return "Error: 'steps' must be a non-empty list of step objects."
        mode = str(mode or "parallel").strip().lower()
        if mode not in ("parallel", "sequential"):
            return "Error: 'mode' must be 'parallel' or 'sequential'."

        unknown_tools: List[str] = []

        def _parse_steps(raw_list, label):
            """Parse a list of raw step objects → (steps, error)."""
            parsed: List[SkillStep] = []
            for i, raw in enumerate(raw_list):
                if not isinstance(raw, dict):
                    return None, f"Error: {label} {i + 1} must be an object, got {type(raw).__name__}."
                tool = (raw.get("tool") or raw.get("tool_name") or "").strip()
                if not tool:
                    return None, f"Error: {label} {i + 1} is missing 'tool'."
                if tool == name:
                    return None, (f"Error: {label} {i + 1} references the macro itself "
                                  f"('{name}') — composed skills cannot recurse.")
                if known_tools and tool not in known_tools:
                    unknown_tools.append(tool)
                params = raw.get("params") or raw.get("param_template") or {}
                if not isinstance(params, dict):
                    return None, f"Error: {label} {i + 1} 'params' must be an object."
                save_as = (raw.get("save_as") or raw.get("saveAs") or "").strip()
                if save_as and not _BIND_NAME_RE.fullmatch(save_as):
                    return None, (f"Error: {label} {i + 1} 'save_as' must be a plain "
                                  f"identifier (letters/digits/underscore), got "
                                  f"{save_as!r}.")
                b_cond = str(raw.get("branch_condition") or "").strip()
                b_target = str(raw.get("branch_target") or "").strip()
                if bool(b_cond) != bool(b_target):
                    _have = "branch_condition" if b_cond else "branch_target"
                    return None, (f"Error: {label} {i + 1} sets {_have} without its "
                                  f"counterpart — a branching step needs BOTH "
                                  f"branch_condition (substring to match in the "
                                  f"result) AND branch_target (a key in 'branches').")
                parsed.append(SkillStep(
                    tool_name=tool,
                    description=(raw.get("description") or f"Step {i + 1}"),
                    param_template=params,
                    optional=bool(raw.get("optional", False)),
                    save_as=save_as,
                    branch_condition=b_cond,
                    branch_target=b_target,
                ))
            return parsed, None

        skill_steps, err = _parse_steps(steps, "step")
        if err:
            return err

        # Branch sequences (2026-07-14): previously the executor honoured
        # branches but NOTHING could author them — the fields existed only
        # for hand-edited JSON. `branches` maps a name to its own step list;
        # a step whose result contains its branch_condition jumps there.
        skill_branches: Dict[str, List[SkillStep]] = {}
        if branches:
            if not isinstance(branches, dict):
                return ("Error: 'branches' must be an object mapping a branch "
                        "name to a list of step objects.")
            if mode != "sequential":
                return ("Error: branching requires mode='sequential' — parallel "
                        "steps have no ordered result to branch on.")
            for bname, braw in branches.items():
                bname_s = str(bname).strip()
                if not bname_s or not _BIND_NAME_RE.fullmatch(bname_s):
                    return (f"Error: branch name {bname!r} must be a plain "
                            f"identifier (letters/digits/underscore).")
                if not isinstance(braw, list) or not braw:
                    return (f"Error: branch '{bname_s}' must be a non-empty "
                            f"list of step objects.")
                bsteps, err = _parse_steps(braw, f"branch '{bname_s}' step")
                if err:
                    return err
                skill_branches[bname_s] = bsteps

        # Every branch_target must resolve to a defined branch — otherwise
        # the jump silently falls through at runtime and the "alternative
        # path" never runs.
        _all_steps = list(skill_steps) + [
            s for bs in skill_branches.values() for s in bs
        ]
        for st in _all_steps:
            if st.branch_target and st.branch_target not in skill_branches:
                return (f"Error: branch_target '{st.branch_target}' has no "
                        f"matching entry in 'branches' — pass "
                        f"branches={{'{st.branch_target}': [ …steps ]}}.")
        if mode == "parallel" and any(st.branch_condition for st in skill_steps):
            return ("Error: branching requires mode='sequential' — parallel "
                    "steps have no ordered result to branch on.")

        df_err = _validate_dataflow(skill_steps, mode, skill_branches)
        if df_err:
            return df_err

        skill = ComposedSkill(
            name=name,
            trigger_description=description,
            steps=skill_steps,
            branches=skill_branches,
            execution_mode=mode,
        )
        reg.register(skill)
        pretty_log(
            "Macro Defined",
            f"Composed skill '{name}' ({len(skill_steps)} steps, {mode}"
            + (f", {len(skill_branches)} branch(es)" if skill_branches else "")
            + ").",
            icon=Icons.MEM_SAVE,
        )
        msg = (
            f"Success: composed skill '{name}' defined with {len(skill_steps)} "
            f"steps ({mode} mode"
            + (f", branches: {', '.join(sorted(skill_branches))}" if skill_branches else "")
            + f"). It is now a TOP-LEVEL TOOL — invoke it by "
            f"name like any built-in; its steps run and the combined results "
            f"come back for you to synthesise."
        )
        if unknown_tools:
            msg += (
                f"\nWARNING: these step tools aren't recognised built-ins and "
                f"will error at run time unless they are acquired skills: "
                f"{', '.join(sorted(set(unknown_tools)))}."
            )
        return msg

    return f"Error: unknown action '{action}' (use define | run | list | approve | delete)."
