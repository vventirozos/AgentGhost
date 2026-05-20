# src/ghost_agent/core/agent.py

import asyncio
import datetime
import json
import logging
import random
import time
import uuid
import re
import sys
import gc

import ctypes
import platform
import httpx
from typing import List, Dict, Any, Optional
from pathlib import Path

from .prompts import SYSTEM_PROMPT, SPECIALIST_SYSTEM_PROMPT, SMART_MEMORY_PROMPT, PLANNING_SYSTEM_PROMPT, SYSTEM_3_GENERATION_PROMPT, SYSTEM_3_EVALUATOR_PROMPT, THINK_BUDGET_TIGHT, THINK_BUDGET_EXTENDED
from .planning import TaskTree, TaskStatus
from ..utils.logging import Icons, pretty_log, request_id_context, atomic_print
from ..utils import logging as _glog

# Sampling parameters for the current upstream model
# (HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive). The model
# publishes four canonical profiles in its card: two for thinking mode
# (the default) and two for non-thinking mode. We pin all four here
# verbatim so callers never have to remember the literal numbers.
#
# THINKING MODE — model is emitting `<think>...</think>` preamble. Used
# for everything the main agent does by default: user chat, self-play
# worker solving a challenge, debugging. The `<think>` budget is still
# regulated by the system prompt's THINK_BUDGET_* directives.
CODING_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 0,
}
GENERAL_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
}

# NON-THINKING MODE — model skips the `<think>` block entirely. Used
# for tasks where reasoning doesn't help and would just burn tokens:
# structured-XML emission (self-play challenge generation), one-shot
# JSON extraction, schema-constrained output. Callers opt in BOTH by
# selecting these sampling params AND by wiring `chat_template_kwargs=
# {"enable_thinking": False}` on the payload (see agent.py payload
# assembly) or appending `/no_think` to the user message (the Qwen
# portable soft-switch). Values are pinned from the model card exactly:
NON_THINKING_GENERAL_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
}
NON_THINKING_REASONING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "min_p": 0,
    "presence_penalty": 2.0,
}

# Adaptive sampling profiles for coding sub-tasks (#5)
_CODING_TASK_PROFILES = {
    "creative":  {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0, "presence_penalty": 0.3},
    "precise":   {"temperature": 0.3, "top_p": 0.90, "top_k": 10, "min_p": 0, "presence_penalty": 0},
    "balanced":  {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0},
}

_CREATIVE_KEYWORDS = {"design", "architect", "brainstorm", "creative", "alternative", "refactor", "naming", "generate", "invent"}
_PRECISE_KEYWORDS = {"sql", "query", "select", "insert", "update", "delete", "migrate", "schema", "exact", "precise", "security", "auth", "encrypt", "regex"}


def _classify_coding_task(query: str) -> str:
    """Classify a coding task for adaptive sampling. Returns 'creative', 'precise', or 'balanced'."""
    if not query:
        return "balanced"
    words = set(query.lower().split())
    creative_hits = len(words & _CREATIVE_KEYWORDS)
    precise_hits = len(words & _PRECISE_KEYWORDS)
    if creative_hits > precise_hits and creative_hits > 0:
        return "creative"
    if precise_hits > creative_hits and precise_hits > 0:
        return "precise"
    return "balanced"


# Queries matching any of these trigger the EXTENDED thinking budget.
# Intentionally narrow: the default stays TIGHT so conversational turns
# and simple tool calls don't pay the extra token cost. Extended is for
# genuine multi-step reasoning where the anti-paralysis cap hurts more
# than it helps (see prompts.py THINK_BUDGET_EXTENDED).
_EXTENDED_THINK_KEYWORDS = {
    "debug", "debugging", "traceback", "error",
    "optimize", "optimization", "tune", "tuning",
    "algorithm", "complexity", "proof", "prove",
    "refactor", "architecture", "architect",
    "explain", "why", "analyze", "analysis",
    "sql", "query", "explain analyze", "vacuum", "cte",
    "race condition", "deadlock", "concurrency",
}


# Tools whose output is state-change confirmation (not evidence of work
# the verifier can assess). When one of these is the last tool this
# turn, the verifier's "is the claim supported by evidence" question
# degenerates to "does 'I built a parser' follow from '{exited: …}'"
# — always REFUTED. Skip or look further back.
_BOOKKEEPING_TOOL_NAMES = frozenset({
    "manage_projects", "manage_tasks", "manage_skills",
    "scratchpad", "learn_skill", "update_profile",
    "replan", "abort_attempt", "dream_mode", "self_play",
    "self_play_loop", "stop_self_play", "list_lessons",
    "knowledge_base",  # insert_fact / forget return confirmations only
})


def _find_substantive_tool_for_verifier(tools_run: Optional[list]) -> Optional[dict]:
    """Return the most recent tool whose output carries verifiable
    evidence (execute output, file content, search results, etc.).

    Walks ``tools_run`` from the end and skips entries whose name is
    in ``_BOOKKEEPING_TOOL_NAMES`` — those produce state-change
    confirmations, not evidence, and feeding them to the verifier
    guarantees a REFUTED verdict (2026-04-19 trace blast radius:
    verifier marked an otherwise-successful run at 10% because the
    evidence slot held `{"exited": "..."}`). When every tool this
    turn was bookkeeping, returns ``None`` so the caller skips the
    verifier entirely — no evidence, no verdict.

    Also skips entries flagged ``_synthetic=True``. Those are
    error/nudge messages the agent loop synthesises when the model
    malforms a tool call (parse errors, invalid JSON args, unknown
    tool, idempotency blocks, etc.). They are NOT real tool output —
    feeding them to the verifier produces "Evidence is irrelevant to
    the claim" REFUTEDs that leak as a user-visible
    ``**Verifier note:**`` line at the end of an otherwise-clean
    conversational reply.
    """
    if not tools_run:
        return None
    for tool in reversed(tools_run):
        if not tool:
            continue
        if tool.get("_synthetic"):
            continue
        name = str(tool.get("name", "")).lower().strip()
        # Handle normalization aliases (agent.py has
        # "managetasks" → "manage_tasks" mapping elsewhere).
        collapsed = name.replace("-", "_").replace(" ", "_")
        if collapsed in _BOOKKEEPING_TOOL_NAMES:
            continue
        return tool
    return None


def _reconstruct_executed_code(
    messages: Optional[list],
    tool_msg: Optional[dict],
) -> str:
    """Walk `messages` backwards from the tool result to its matching
    assistant tool-call and return the code/command that was actually
    submitted to the tool.

    Previously the verifier gate at `handle_chat` called
    `verify_code_output(code=tool_name, ...)` — i.e. it passed the
    literal string "execute" as the code to audit. With nothing to
    reason about, the verifier hallucinated reasons the output didn't
    match ("appears to be a directory listing", "missing expected
    field", etc.) and REFUTED correct turns at high confidence. The
    fix: recover the actual code from the tool-call arguments via the
    `tool_call_id`, cap at 4000 chars, hand it to the verifier.

    Returns "" when the code can't be reconstructed — caller then
    falls back to `verify_claim` (no code slot, purely "does the
    evidence support the claim?" shape).
    """
    if not tool_msg or not messages:
        return ""
    tool_id = tool_msg.get("tool_call_id")
    if not tool_id:
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict) or tc.get("id") != tool_id:
                continue
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments")
            args: dict = {}
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except Exception:
                    # Not valid JSON — surface the raw string so the
                    # verifier has at least something to audit.
                    return args_raw[:4000]
            elif isinstance(args_raw, dict):
                args = args_raw
            # Common execute-tool arg shapes, in order of preference.
            for key in ("content", "code", "script_content", "text", "command", "cmd"):
                v = args.get(key)
                if isinstance(v, str) and v.strip():
                    return v[:4000]
            return ""
    return ""


def classify_thinking_budget(query: str,
                             has_coding_intent: bool = False,
                             is_meta_task: bool = False,
                             in_active_project: bool = False) -> str:
    """Pick the <think> block guidance for this turn.

    Returns either ``"tight"`` (the 5-sentence anti-paralysis cap used
    historically for everything) or ``"extended"`` (up to ~15 sentences
    for genuine multi-step reasoning). See ``prompts.THINK_BUDGET_*``
    for the prompt text each returns.

    Extended only fires when the query hints at multi-step reasoning
    AND the overall intent isn't a meta task (title/caption/rename —
    those are fundamentally single-shot).

    When ``in_active_project`` is True, we bias HARD toward tight.
    During long-running project work most turns are bookkeeping
    (mark task done, write next file, run a check) driven by the
    task tree rather than the user's last message. Extended thinking
    on those turns costs 15-25s and produces prose the task tree
    already encodes — the 2026-04-19 trace spent 8+ turns doing
    this. We still escalate to extended when the user's latest
    message is visibly a new complex ask (≥3 extended keywords) so
    genuine "now debug this tricky bug" follow-ups still get room.
    """
    if is_meta_task:
        return "tight"
    if not query:
        return "tight"
    lowered = query.lower()
    hits = sum(1 for kw in _EXTENDED_THINK_KEYWORDS if kw in lowered)
    if in_active_project:
        # Much higher bar inside a project: only escalate when the
        # user's current turn is unambiguously a new complex ask.
        return "extended" if hits >= 3 else "tight"
    if hits >= 2:
        return "extended"
    # Coding intent with a single strong keyword still warrants extended.
    if has_coding_intent and hits >= 1:
        return "extended"
    return "tight"


def render_think_budget_guidance(budget: str) -> str:
    """Return the guidance string to splice into QWEN_TOOL_PROMPT."""
    if budget == "selfplay":
        # Imported lazily so the module-level import block stays narrow;
        # SELFPLAY is only reachable when an explicit override is set.
        from .prompts import THINK_BUDGET_SELFPLAY
        return THINK_BUDGET_SELFPLAY
    if budget == "extended":
        return THINK_BUDGET_EXTENDED
    return THINK_BUDGET_TIGHT


def get_sampling_params(is_tool_turn: bool, query: str = "", is_coding: bool = False) -> dict:
    """Return the sampling profile for the current turn.

    Policy (matches the Qwen 3.6 35B-A3 model card):
    * Conversational turn (greetings, chit-chat, no tool expected) →
      GENERAL_SAMPLING_PARAMS: temperature=1.0, presence_penalty=1.5.
    * Any tool-using turn (coding OR a non-coding tool like
      update_profile / web_search / manage_tasks) → CODING_SAMPLING_PARAMS
      by default: temperature=0.6, presence_penalty=0.
    * Coding tool turns additionally get sub-classified
      (creative / precise / balanced) so algorithm design gets slightly
      higher temp and SQL / security / regex get lower temp.

    The old signature took a single ``is_coding`` flag and therefore
    left ALL non-coding tool turns on the temperature=1.0 conversational
    profile — that produced over-eager duplicate setter calls (e.g. a
    second `update_profile` on the same facts one turn after the first
    succeeded). Routing every tool turn through the precise profile
    removes that variance without costing conversational warmth.
    """
    if not is_tool_turn:
        return dict(GENERAL_SAMPLING_PARAMS)
    if is_coding:
        profile = _classify_coding_task(query)
        return dict(_CODING_TASK_PROFILES.get(profile, CODING_SAMPLING_PARAMS))
    # Non-coding tool turn — use the base precise profile (temp=0.6).
    return dict(CODING_SAMPLING_PARAMS)


# Streaming guards. When the upstream model enters a self-repeating thinking
# loop (re-deriving the same paragraph hundreds of times), we kill the stream
# instead of burning context indefinitely.
MAX_THINKING_CHARS = 32000          # initial cap on a single turn's <think> block
MAX_THINKING_CHARS_EXTENDED = 64000 # extended cap when model is making progress

# Upstream `max_tokens` cap for a single tool-using turn. Without this, the
# payload sends no max_tokens at all and the server default (often 1024 or
# 2048) silently truncates the assistant output mid-`<tool_call>`. The
# truncated XML then fails to parse and the agent burns 6 turns of its
# strike budget rebuilding the same almost-closed block. 8192 is generous
# for a thought + large code payload but small enough to fit comfortably
# inside a 65k-token context window with room for tool results.
#
# Raised from 8192 to 16384 for real workers that need room for a full
# solution script + reasoning preamble. The downside of a high cap
# ("16k tokens of `<tool_call>` spam instead of 8k") is bounded by the
# `_detect_tool_call_loop` streaming probe above, which aborts the
# stream after ~10 unclosed `<tool_call>` openings (tens of tokens of
# collapse, not tens of thousands). The cap and the detector are a
# pair: raise the cap without wiring the detector and a degenerate run
# turns 16k tokens of garbage; wire the detector and the cap is only
# reached by healthy verbose generations.
DEFAULT_TOOL_TURN_MAX_TOKENS = 16384


def _distill_terminal_tool_summary(tool_name: str, raw: str) -> str:
    """Turn a terminal tool's return string into a short user-facing
    summary. `tool_self_play` / `tool_dream_mode` return a blob that
    mixes three things: user-relevant status, internal telemetry
    (``CURIOSITY: ...``), and a ``SYSTEM INSTRUCTION:`` trailer meant
    for the summary-turn LLM. Dumping the whole blob into the chat
    window leaks the directive and makes the reply noisy; we parse
    out the key facts and format a 1-3 line summary instead.

    Expected shape of `raw` (from `dream.py:synthetic_self_play`):

        Synthetic Self-Play cycle completed. Final Status: ...
        SELF-PLAY POST-MORTEM REPORT:
        Challenge: ...
        Status: ...
        Cluster: <cluster>  Compression delta: ...
        Skill gate: ...
        [or the learning branch:
            Learned task: ...
            Mistake: ...
            Solution: ...]

        CURIOSITY: cluster=... compression_delta=...

        SYSTEM INSTRUCTION: ...  <-- for the LLM, not the user

    When the shape is unrecognisable (e.g. the tool hit an error
    path), fall back to the first ~400 chars after stripping the
    known trailers.
    """
    if not raw:
        return ""

    # Strip trailers meant for other consumers.
    cleaned = re.sub(
        r"\n*SYSTEM INSTRUCTION:.*$",
        "", raw, flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^CURIOSITY:.*?$", "", cleaned, flags=re.MULTILINE
    ).strip()
    cleaned = re.sub(
        r"\n*SYSTEM:\s*[A-Z_ ]+?(DONE|FINISHED|STAND BY)\.?\s*$",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()

    status_m = re.search(r"^Status:\s*(.+)$", cleaned, re.MULTILINE)
    cluster_m = re.search(r"^Cluster:\s*(\S+)", cleaned, re.MULTILINE)
    gate_m = re.search(r"^Skill gate:\s*(.+)$", cleaned, re.MULTILINE)
    learned_m = re.search(r"^Learned task:\s*(.+)$", cleaned, re.MULTILINE)

    status = status_m.group(1).strip() if status_m else ""
    cluster = cluster_m.group(1).strip() if cluster_m else ""
    gate = gate_m.group(1).strip() if gate_m else ""
    learned = learned_m.group(1).strip() if learned_m else ""

    lines = []
    if cluster and status:
        # Em-dash separator reads better than nested parens; status
        # already contains its own parenthetical like
        # "SUCCESS (in 1 attempts)".
        lines.append(f"Challenge: {cluster} — {status}.")
    elif status:
        lines.append(f"Result: {status}.")
    if learned:
        lines.append(f"Lesson learned: \"{learned}\".")
    elif gate:
        lines.append(f"Skill gate: {gate}.")

    if lines:
        return "\n".join(lines)

    # Unrecognised shape — show the cleaned raw, capped.
    return cleaned[:400] + ("..." if len(cleaned) > 400 else "")


def _tool_call_truncated(content: str) -> bool:
    """Return True when `content` opens a <tool_call>/<function> block but
    never closes it.

    This catches the upstream-truncation case where the server's max_tokens
    cap severed the assistant's output mid-tool-call. We use it both (a) as
    a diagnostic signal fed back to the model ("your output was cut off,
    shorten or chunk") and (b) as a hint to the parser that it should
    emit whatever partial `<parameter>` body it DID capture rather than
    giving up with `system_parse_error`.
    """
    if not content:
        return False
    opens = len(re.findall(r'<tool_call\b', content, re.IGNORECASE))
    closes = len(re.findall(r'</tool_call\b', content, re.IGNORECASE))
    if opens > closes:
        return True
    fn_opens = len(re.findall(r'<function\b[^>]*>', content, re.IGNORECASE))
    fn_closes = len(re.findall(r'</function\b', content, re.IGNORECASE))
    if fn_opens > fn_closes:
        return True
    return False
THINKING_LOOP_PROBE_EVERY = 300     # run the repetition probe every N chars
THINKING_LOOP_WINDOW = 150          # length of the n-gram we look for
THINKING_LOOP_THRESHOLD = 2         # window appearing >= N times = loop
# Prior settings (500 / 200 / 3) meant the probe needed ~600 chars of
# actual repetition before it fired. The 2026-04-19 trace 0B showed the
# model emitting ~10000 chars of "I'll write X. Then Y. Then Z." before
# the detector aborted — 60+ seconds of wasted decode time. Dropping
# the threshold to 2×150 fires at ~300 chars (5-10s), which is an
# acceptable false-positive risk given how distinctive enumeration
# loops are (the 150-char tail is usually "I'll write X. Then Y.").

# Tool-call generation-collapse detector. In the wild we've seen Qwen
# emit 8000+ consecutive `<tool_call>` tokens with zero `</tool_call>` /
# `<function>` / `<parameter>`, burning 300+ seconds of decoder time
# before hitting max_tokens. The n-gram thinking-loop detector catches
# this eventually, but only after ~600 chars of repetition; this
# specialised probe fails fast after ~10 unclosed openings.
TOOL_CALL_LOOP_THRESHOLD = 10       # unclosed `<tool_call>` openings = collapse
TOOL_CALL_LOOP_PROBE_EVERY = 200    # run the probe every N chars of new content


def _detect_thinking_loop(buf: str) -> bool:
    """True if the tail of `buf` repeats itself enough to be a loop.

    Checks two n-gram sizes (the tight 200-char window for fast repeats,
    plus a 400-char window as a backstop for slightly-paraphrased runs
    where each paragraph is just long enough to dodge the 200-char probe)."""
    if len(buf) < THINKING_LOOP_WINDOW * THINKING_LOOP_THRESHOLD:
        return False
    tail = buf[-THINKING_LOOP_WINDOW:]
    if buf.count(tail) >= THINKING_LOOP_THRESHOLD:
        return True
    wide_window = THINKING_LOOP_WINDOW * 2
    if len(buf) >= wide_window * THINKING_LOOP_THRESHOLD:
        wide_tail = buf[-wide_window:]
        if buf.count(wide_tail) >= THINKING_LOOP_THRESHOLD:
            return True
    return False


def _detect_tool_call_loop(buf: str) -> bool:
    """True if the content buffer has accumulated too many unclosed
    `<tool_call>` openings — a decoder-collapse signature where the
    model is stuck emitting opening tags but never closing them.

    The healthy case is N opens + N closes (≥0 complete tool calls) or a
    single open waiting for its close. Anything where opens -
    closes > THRESHOLD is a run of openings with no progress, and we
    should kill the stream rather than let it run to max_tokens."""
    if not buf:
        return False
    opens = len(re.findall(r'<tool_call\b', buf, re.IGNORECASE))
    closes = len(re.findall(r'</tool_call\b', buf, re.IGNORECASE))
    return (opens - closes) > TOOL_CALL_LOOP_THRESHOLD


def _render_assistant_with_tool_calls(content: str, tool_calls: list) -> str:
    """Render an assistant message + its native ``tool_calls`` list as the
    inline XML string the upstream Qwen-style API expects to see in history.

    The previous implementation re-checked ``"<tool_call>" in ast_content``
    on every iteration of the render loop. After the FIRST call was
    appended, the substring became present and silently dropped calls
    2..N — producing the production bug where 4 parallel ``update_profile``
    calls were re-derived on the next turn (and blocked by the idempotency
    guard). Snapshot the inline-check ONCE before the loop.
    """
    ast_content = content or ""
    if not tool_calls:
        return ast_content.strip()
    already_inline = "<tool_call>" in ast_content or "<tool_call" in ast_content
    if already_inline:
        return ast_content.strip()
    for tc in tool_calls:
        tc_func = tc.get("function", {})
        tc_args = tc_func.get("arguments", "{}")
        if isinstance(tc_args, str):
            try:
                tc_args_dict = json.loads(tc_args)
            except Exception:
                tc_args_dict = {}
        else:
            tc_args_dict = tc_args
        xml_call = f'\n<tool_call>\n<function name="{tc_func.get("name", "")}">\n'
        for k, v in tc_args_dict.items():
            if isinstance(v, (dict, list, bool, type(None))):
                v_rendered = json.dumps(v, ensure_ascii=False)
            else:
                v_rendered = str(v)
            xml_call += f'<parameter name="{k}">\n{v_rendered}\n</parameter>\n'
        xml_call += "</function>\n</tool_call>\n"
        ast_content += xml_call
    return ast_content.strip()
from ..utils.token_counter import estimate_tokens
from ..tools.registry import get_available_tools, TOOL_DEFINITIONS, get_active_tool_definitions
from ..tools.tasks import tool_list_tasks
from ..memory.skills import SkillMemory

logger = logging.getLogger("GhostAgent")


# ============================================================================
# ARCHITECTURAL OPTIMISATION #7: TOOL-SCHEMA CACHE
# ----------------------------------------------------------------------------
# `_json_to_xml_schema` used to be defined and executed inside the turn loop
# (one fresh string-build per turn even when the tool list was identical).
# These module-level helpers hoist the conversion out and memoize it on a
# stable signature of the (frozen) function definitions.
# ============================================================================
def _freeze_funcs(funcs: list) -> tuple:
    """Convert a list of mutable tool function dicts into a hashable tuple
    of immutable views suitable for `lru_cache` keys."""
    out = []
    for f in funcs:
        params = f.get("parameters") or {}
        properties = params.get("properties") or {}
        required = tuple(sorted(params.get("required") or []))
        prop_items = []
        for p_name, p_data in properties.items():
            enum_val = p_data.get("enum")
            prop_items.append((
                p_name,
                p_data.get("type", "string"),
                p_data.get("description", ""),
                tuple(enum_val) if isinstance(enum_val, list) else (),
            ))
        out.append((
            f.get("name", ""),
            f.get("description", ""),
            tuple(prop_items),
            required,
        ))
    return tuple(out)


def _xml_schema_key(tool_defs: list) -> str:
    """Stable cache key from a list of {function: {...}} dicts."""
    names = sorted(t.get("function", {}).get("name", "") for t in tool_defs)
    return "|".join(names)


import functools as _ft


@_ft.lru_cache(maxsize=128)
def _json_to_xml_schema_cached(frozen_funcs: tuple) -> str:
    """Memoized XML-schema serialiser. Keyed on the frozen tuple produced
    by `_freeze_funcs`. The output is deterministic per tool list, so a
    cache hit means zero string concatenation per turn.

    Uses REAL newlines (``\\n``) between XML elements, not the
    two-character escape sequence ``\\\\n`` that the previous version
    accidentally shipped. Literal ``\\n`` strings inflate the tokenizer
    output (one extra token per occurrence) and add zero information —
    with 21 tools and ~10 newlines each, that was ~210 wasted tokens
    per coding turn on top of whatever gain cache_prompt delivers.
    """
    parts = []
    for name, description, prop_items, required in frozen_funcs:
        parts.append(f'<tool_def>\n<function name="{name}">\n')
        if description:
            parts.append(f'<description>{description}</description>\n')
        if prop_items:
            for p_name, p_type, p_desc, p_enum in prop_items:
                is_req = "true" if p_name in required else "false"
                parts.append(f'<parameter name="{p_name}" type="{p_type}" required="{is_req}">\n')
                if p_desc:
                    parts.append(f'<description>{p_desc}</description>\n')
                if p_enum:
                    parts.append(f'<enum>{", ".join(p_enum)}</enum>\n')
                parts.append('</parameter>\n')
        parts.append('</function>\n</tool_def>\n')
    return "".join(parts).strip()

def extract_json_from_text(text: str) -> dict:
    """Safely extracts JSON from LLM outputs, ignoring conversational filler and markdown blocks.

    Returns ``{}`` on every failure mode (missing JSON, malformed JSON, etc.)
    — the empty-dict contract is load-bearing for the dozens of call sites
    that do `result = extract_json_from_text(...).get("score", 0)` style
    access.

    Distinguishing "no JSON present" from "JSON was malformed" is useful for
    debugging the planner / smart memory / post-mortem flows. We log a
    WARNING (not DEBUG) when the input clearly *contained* something that
    looked like JSON but could not be parsed, so the operator notices
    silent extraction failures in production logs.
    """
    import re, json, ast
    # Qwen Syntax Healing: Fix {"name"="tool"...} or {"name"= "tool"...} hallucinations.
    # Also heal `"key"=` → `"key":` for ANY field, not just `name` (the previous
    # version only fixed the `name` field).
    text = re.sub(r'(?<=[{,\s])"([\w_]+)"\s*=\s*', r'"\1": ', text)

    def _parse(t):
        try:
            return json.loads(t, strict=False)
        except json.JSONDecodeError:
            try:
                # AST Fallback for models that output Python dicts instead of strict JSON
                # Only replace standalone JSON keywords, not substrings inside strings
                pt = re.sub(r'\btrue\b', 'True', t)
                pt = re.sub(r'\bfalse\b', 'False', pt)
                pt = re.sub(r'\bnull\b', 'None', pt)
                res = ast.literal_eval(pt)
                if isinstance(res, dict): return res
            except Exception as e:
                logger.debug(f"JSON AST fallback failed: {type(e).__name__}")
            return {}

    looked_like_json = False
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            looked_like_json = True
            p = _parse(match.group(1))
            if p:
                return p

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            looked_like_json = True
            p = _parse(text[start:end+1])
            if p:
                return p

        result = _parse(text)
        if not result and looked_like_json:
            # We saw braces but couldn't parse them — that's an extraction
            # failure worth surfacing, not a "no JSON here" non-event.
            preview = text[:200].replace("\n", " ")
            logger.warning(f"extract_json_from_text: malformed JSON-like content failed to parse. Preview: {preview}")
        return result
    except Exception as e:
        logger.warning(f"extract_json_from_text raised {type(e).__name__}: {e}")
        return {}

class GhostContext:
    def __init__(self, args, sandbox_dir, memory_dir, tor_proxy):
        self.args = args
        self.sandbox_dir = sandbox_dir
        self.memory_dir = memory_dir
        self.tor_proxy = tor_proxy
        self.llm_client = None
        self.memory_system = None
        self.profile_memory = None
        self.graph_memory = None
        self.skill_memory = None
        self.scratchpad = None
        self.sandbox_manager = None
        self.scheduler = None
        self.memory_bus = None
        self.biological_task = None
        self.last_activity_time = datetime.datetime.now()
        self.cached_sandbox_state = None
        # Long-term project state. `project_store` is lazily initialized by
        # main.py (or tests) so GhostContext itself stays cheap to
        # construct; `current_project_id` tracks which project the agent
        # is currently scoped to (None == free chat / one-shot mode).
        self.project_store = None
        self.current_project_id = None
        # Stage-1 self-improvement wiring. Populated by main.py during
        # lifespan when the corresponding features are enabled; left as
        # None otherwise so deployments without the trajectory pipeline
        # can run unchanged. The biological watchdog's phase 2.5
        # (reflection) reads `reflector` + `trajectory_collector` and is
        # a no-op when either is None.
        self.trajectory_collector = None
        self.reflector = None
        self.complexity_dispatcher = None
        # Most-recent user message for the current turn. Stashed by
        # handle_chat right after it extracts it from the request body so
        # tools can inspect user intent without re-parsing the message
        # list. Used by `tools.memory.tool_self_play*` to refuse calls the
        # LLM hallucinated mid-session — the watchdog's biological
        # self-play path does NOT go through the tool layer, so clearing
        # this field is harmless for internal firing. Empty string when
        # no user turn is in flight (e.g. background tasks).
        self.last_user_content = ""

class GhostAgent:
    def __init__(self, context: GhostContext):
        self.context = context
        self.disabled_tools = set()
        self.available_tools = get_available_tools(context)
        self.agent_semaphore = asyncio.Semaphore(10)
        self.memory_semaphore = asyncio.Semaphore(1)

    def release_unused_ram(self):
        try:
            gc.collect()
            if platform.system() == "Linux":
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except: pass
        except: pass

    def clear_session(self):
        if hasattr(self.context, 'scratchpad') and self.context.scratchpad:
            self.context.scratchpad.clear()
        self.release_unused_ram()
        return True

    def _prepare_planning_context(self, tools_run_this_turn: List[Dict[str, Any]]) -> str:
        if not tools_run_this_turn:
            return "None (Start of Task)"

        char_limit = max(4000, int(self.context.args.max_context * 3.5 * 0.1))

        outputs = []
        for t in tools_run_this_turn:
            # Synthetic agent-loop error entries (parse-error nudges,
            # unknown-tool blocks, idempotency stops, etc.) are NOT
            # real tool output — feeding their `SYSTEM ERROR:` /
            # `Error: Invalid JSON arguments` strings to the planner
            # makes it plan against fabricated evidence and often
            # repeat the same broken action.
            if t.get("_synthetic"):
                continue
            content = str(t.get("content", ""))
            if len(content) > char_limit:
                # Keep top char_limit so the Planner actually sees the search matches
                content = content[:char_limit] + "\n\n... [TRUNCATED: Tool output too long. Showing top results only.]"
            outputs.append(f"Tool [{t.get('name', 'unknown')}]: {content}")

        if not outputs:
            return "None (Start of Task)"
        return "\n\n".join(outputs)

    @staticmethod
    def _opening_word_set(text: str, max_chars: int = 300, min_word_len: int = 4) -> set:
        """Fingerprint the opening of a reasoning block into a lower-case
        word set for Jaccard similarity. Short/common words are dropped
        so cross-turn similarity reflects topic overlap, not filler.

        Used by the cross-turn repetition guard — see the call sites
        inside `handle_chat`. Keep the signature stable: the test
        harness in `test_cross_turn_repetition.py` builds reference
        fingerprints the same way.
        """
        if not text:
            return set()
        snippet = text[:max_chars]
        return {
            w for w in re.findall(r"[a-zA-Z_]+", snippet.lower())
            if len(w) >= min_word_len
        }

    def _get_recent_transcript(self, messages: List[Dict[str, Any]]) -> str:
        msg_limit = max(40, int(self.context.args.max_context / 500))
        char_limit = max(500, int(self.context.args.max_context * 3.5 * 0.02))

        recent_transcript = ""
        transcript_msgs = [m for m in messages if m.get("role") in ["user", "assistant", "tool"]][-msg_limit:]
        for m in transcript_msgs:
            content_val = m.get('content') or ""
            if isinstance(content_val, list):
                text_parts = []
                for item in content_val:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            text_parts.append("[Image attached and passed to vision node]")
                content_val = "\n".join(text_parts)
            content_str = str(content_val)

            role = m['role'].upper()
            if role == "TOOL":
                role = f"TOOL ({m.get('name', 'unknown')})"
            recent_transcript += f"{role}: {content_str[:char_limit]}\n"
        return recent_transcript

    def process_rolling_window(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        if not messages: return []
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if len(system_msgs) > 1:
            merged_content = "\n\n".join([str(m.get("content", "")) for m in system_msgs])
            system_msgs = [{"role": "system", "content": merged_content}]
        raw_history = [m for m in messages if m.get("role") != "system"]

        current_tokens = sum(estimate_tokens(str(m.get("content", ""))) for m in system_msgs)
        final_history = []

        def _msg_tokens(msg):
            content = msg.get("content", "")
            if isinstance(content, list):
                text_only = " ".join([str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"])
                return estimate_tokens(text_only)
            return estimate_tokens(str(content))

        # Pure sliding window from newest to oldest.
        # We NEVER mutate historical strings, we just drop the oldest ones if we run out of space.
        # We also preserve tool-call / tool-result pairs to avoid orphaned messages.
        i = len(raw_history) - 1
        while i >= 0:
            msg = raw_history[i]
            msg_tok = _msg_tokens(msg)

            # If this is a "tool" role message, find its paired assistant message
            # (the one with tool_calls that triggered it) and keep them together.
            if msg.get("role") == "tool" and i > 0:
                # Collect consecutive tool messages that belong to the same assistant turn
                group = [msg]
                group_tokens = msg_tok
                j = i - 1
                while j >= 0 and raw_history[j].get("role") == "tool":
                    group.insert(0, raw_history[j])
                    group_tokens += _msg_tokens(raw_history[j])
                    j -= 1
                # The message at j should be the paired assistant with tool_calls
                if j >= 0 and raw_history[j].get("role") == "assistant":
                    group.insert(0, raw_history[j])
                    group_tokens += _msg_tokens(raw_history[j])
                    j -= 1

                if current_tokens + group_tokens > max_tokens:
                    break
                final_history = group + final_history
                current_tokens += group_tokens
                i = j
                continue

            if current_tokens + msg_tok > max_tokens:
                break
            final_history.insert(0, msg)
            current_tokens += msg_tok
            i -= 1

        return system_msgs + final_history

    async def _prune_context(self, messages: List[Dict[str, Any]], max_tokens: int = 12000, model: str = "test-model") -> List[Dict[str, Any]]:
        current_tokens = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                text_only = " ".join([str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"])
                current_tokens += estimate_tokens(text_only)
            else:
                current_tokens += estimate_tokens(str(content))
        if current_tokens < max_tokens:
            return messages

        pretty_log("Context Optimization", f"Context hit {current_tokens} tokens (> {max_tokens}). Running summarization pass...", icon=Icons.CUT)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        # If we have very few messages but still hit the token limit,
        # just truncate without summarizing as it's likely a huge prompt
        if len(non_system_msgs) <= 5:
            truncated = non_system_msgs[-3:]
            # Scrub images from the truncated fallback
            scrubbed_truncated = []
            for msg in truncated:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "image_url":
                            new_content.append({"type": "text", "text": "[Image attached and passed to vision node]"})
                        else:
                            new_content.append(block)
                    scrubbed_truncated.append({**msg, "content": new_content})
                else:
                    scrubbed_truncated.append(msg)
            return system_msgs + scrubbed_truncated

        # Keep recent context (last 3 turns = 6 messages + goal)
        original_goal = non_system_msgs[0]
        recent_context = non_system_msgs[-6:] # Keep last ~3 turns intact

        # SEMANTIC CONTEXT ANCHORING: extract key findings from middle
        # turns before they get summarized away. These anchors survive
        # context pruning so the agent retains its earlier insights.
        # Anchors are: (1) tool results with error messages or key data,
        # (2) assistant messages with explicit findings/conclusions,
        # (3) the most recent successful tool result (legacy behavior).
        anchor_messages = []
        anchored_originals = set()  # Track which original messages were anchored by id()
        _anchor_keywords = {
            "error:", "traceback", "found:", "result:", "conclusion:",
            "root cause", "the issue is", "the problem is", "discovered",
            "key finding", "important:", "note:", "exit code:",
        }
        for m in non_system_msgs[1:-6]:
            content = str(m.get("content", "")).lower()[:500]
            is_tool = m.get("role") == "tool"
            is_assistant_finding = m.get("role") == "assistant" and any(kw in content for kw in _anchor_keywords)
            is_tool_error = is_tool and any(kw in content for kw in {"error:", "traceback", "exit code: 1", "failed"})
            is_tool_data = is_tool and len(str(m.get("content", ""))) > 100

            if is_assistant_finding or is_tool_error:
                # Compact the anchor to save tokens
                anchor_text = str(m.get("content", ""))[:800]
                anchor_messages.append({
                    "role": m["role"],
                    "content": f"[ANCHORED] {anchor_text}",
                    **({"name": m["name"]} if "name" in m else {})
                })
                anchored_originals.add(id(m))

        # Legacy: always keep the most recent tool result from middle
        recent_tool_anchor = None
        for m in reversed(non_system_msgs[1:-6]):
            if m.get("role") == "tool" and id(m) not in anchored_originals:
                recent_tool_anchor = m
                break

        # Cap anchors to prevent them from dominating the context
        anchor_messages = anchor_messages[:4]

        middle_messages = [
            m for m in non_system_msgs[1:-6]
            if m is not recent_tool_anchor and id(m) not in anchored_originals
        ]

        if not middle_messages:
            all_anchors = anchor_messages + ([recent_tool_anchor] if recent_tool_anchor else [])
            return system_msgs + [original_goal] + all_anchors + recent_context

        # Condense the middle messages using a fast LLM worker
        condense_prompt = "The following is the middle segment of a long conversational transcript between an AI Agent and a User. Summarize the key actions taken, facts learned, and the current state of progress. Be concise. DO NOT write code. ONLY output the summary.\n\nTRANSCRIPT:\n"
        for m in middle_messages:
            content_val = m.get('content') or ""
            if isinstance(content_val, list):
                text_parts = []
                for item in content_val:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            text_parts.append("[Image attached and passed to vision node]")
                content_val = "\n".join(text_parts)
            content_str = str(content_val)
            condense_prompt += f"{m.get('role').upper()}:\n{content_str[:4000]}\n---\n"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": condense_prompt}],
            "temperature": 0.0,
            "max_tokens": 800
        }

        summary = "[SYSTEM: PREVIOUS TURNS SUMMARIZED]\n\n"
        try:
            summary_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            summary += str(summary_data["choices"][0]["message"].get("content") or "No summary generated.")

            from ..utils.helpers import get_utc_timestamp
            if self.context.memory_system and "Summarization unavailable" not in summary:
                episode_text = f"EPISODIC ARCHIVE (Past Conversation Summary):\n{summary}"
                asyncio.create_task(asyncio.to_thread(
                    self.context.memory_system.add,
                    episode_text,
                    {"type": "episode", "timestamp": get_utc_timestamp()}
                ))
        except Exception as e:
            logger.warning(f"Context summarization failed: {e}")
            summary += f"(Summarization unavailable due to error, dropping old turns: {e})"

        # Insert the summary BEFORE the recent context (so the model reads
        # the fresh stuff last), re-attach anchored findings and the tool
        # result we hoisted out of the middle.
        all_anchors = anchor_messages + ([recent_tool_anchor] if recent_tool_anchor else [])
        return system_msgs + [original_goal, {"role": "assistant", "content": summary}] + all_anchors + recent_context

    # Common LLM hallucination patterns → canonical tool names.
    _TOOL_ALIAS_TABLE = {
        "filesystem": "file_system",
        "file-system": "file_system",
        "fs": "file_system",
        "files": "file_system",
        "update-profile": "update_profile",
        "updateprofile": "update_profile",
        "profile_update": "update_profile",
        "knowledgebase": "knowledge_base",
        "knowledge-base": "knowledge_base",
        "kb": "knowledge_base",
        "websearch": "web_search",
        "web-search": "web_search",
        "search": "web_search",
        "deepresearch": "deep_research",
        "deep-research": "deep_research",
        "factcheck": "fact_check",
        "fact-check": "fact_check",
        "vision": "vision_analysis",
        "vision-analysis": "vision_analysis",
        "imagegeneration": "image_generation",
        "image-generation": "image_generation",
        "imagegen": "image_generation",
        "createskill": "create_skill",
        "create-skill": "create_skill",
        "manageskills": "manage_skills",
        "manage-skills": "manage_skills",
        "managetasks": "manage_tasks",
        "manage-tasks": "manage_tasks",
        "learnskill": "learn_skill",
        "learn-skill": "learn_skill",
        "systemutility": "system_utility",
        "system-utility": "system_utility",
        "system": "system_utility",
        "postgresadmin": "postgres_admin",
        "postgres-admin": "postgres_admin",
        "postgres": "postgres_admin",
        "delegatetoswarm": "delegate_to_swarm",
        "delegate-to-swarm": "delegate_to_swarm",
        "delegate": "delegate_to_swarm",
        "selfplay": "self_play",
        "self-play": "self_play",
        "selfplayloop": "self_play_loop",
        "self-play-loop": "self_play_loop",
        "continuousselfplay": "self_play_loop",
        "stopselfplay": "stop_self_play",
        "stop-self-play": "stop_self_play",
        "listlessons": "list_lessons",
        "list-lessons": "list_lessons",
        "whathaveyoulearned": "list_lessons",
        "lessons": "list_lessons",
        "dreammode": "dream_mode",
        "dream-mode": "dream_mode",
    }

    @classmethod
    def _canonicalise_tool_name(cls, name: str, available: List[str]) -> Optional[str]:
        """Map a hallucinated tool name back to a real one.

        Strategy: lowercase + strip non-alphanumeric → exact alias table →
        difflib close match against `available` (cutoff 0.7). Returns None
        if nothing plausibly matches; the caller will surface a hard error
        so the model retries with a different shape."""
        if not name:
            return None
        normalised = re.sub(r'[^a-z0-9]', '', str(name).lower())
        if normalised in cls._TOOL_ALIAS_TABLE:
            mapped = cls._TOOL_ALIAS_TABLE[normalised]
            if mapped in available:
                return mapped
        # Build normalised → canonical map of the actually-available tools
        # so the difflib fallback can match without dashes/case noise.
        norm_to_real = {re.sub(r'[^a-z0-9]', '', t.lower()): t for t in available}
        if normalised in norm_to_real:
            return norm_to_real[normalised]
        import difflib as _difflib
        close = _difflib.get_close_matches(normalised, list(norm_to_real.keys()), n=1, cutoff=0.7)
        if close:
            return norm_to_real[close[0]]
        return None

    def _get_memory_bus(self):
        """Return the cognitive MemoryBus, lazily building one from this
        context's memory subsystems if `context.memory_bus` was never wired
        (the production lifespan sets it; tests typically don't)."""
        from .bus import MemoryBus
        bus = getattr(self.context, 'memory_bus', None)
        if isinstance(bus, MemoryBus):
            return bus
        return MemoryBus(
            vector_memory=getattr(self.context, 'memory_system', None),
            graph_memory=getattr(self.context, 'graph_memory', None),
            skill_memory=getattr(self.context, 'skill_memory', None),
            profile_memory=getattr(self.context, 'profile_memory', None),
            episodic_memory=getattr(self.context, 'episodic_memory', None),
        )

    async def biological_watchdog(self):
        """Native asyncio daemon for biological background hooks.

        Replaces the previous APScheduler-driven `idle_dream_watchdog`. Runs
        as a long-lived task launched from the FastAPI lifespan; cancellation
        is propagated cleanly on shutdown. Timing thresholds for Journal
        consolidation, REM dreaming, and synthetic self-play are unchanged.
        """
        logger.info("Biological watchdog daemon started")
        try:
            while True:
                await asyncio.sleep(60)
                try:
                    await self._biological_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Biological watchdog tick failed: {e}")
        except asyncio.CancelledError:
            logger.info("Biological watchdog daemon cancelled")
            raise

    # Per-phase cooldowns (in seconds) so the watchdog can't fire two REM
    # cycles or two self-play sessions back-to-back when the user remains AFK.
    _DREAM_COOLDOWN = 1800        # 30 min between dreams
    _REFLECTION_COOLDOWN = 2400   # 40 min between reflections
    _SKILLS_AUTO_COOLDOWN = 7200  # 2 hours between skill auto-extractions
    _PRM_TRAIN_COOLDOWN = 10800   # 3 hours between PRM retrain passes
    _NARRATIVE_COOLDOWN = 3600    # 60 min between selfhood-narrative consolidations
    _SELFPLAY_COOLDOWN = 3600     # 60 min between self-plays
    # Belt-and-braces guard for phase 1. The journal-empty self-disarm
    # already prevents same-batch refire, but a journal write that
    # raises mid-loop (or a misbehaving consumer that fails to drain)
    # would otherwise re-fire every tick. The cooldown caps refire
    # rate at the same shape as the other five phases.
    _JOURNAL_COOLDOWN = 60        # 60 s between journal-process passes

    async def _biological_tick(self):
        """One pass of the biological hook state machine. Extracted from the
        loop for direct unit testing."""
        ctx = self.context
        if not getattr(ctx, 'memory_system', None):
            return

        # HARD LOCK: never interrupt an active LLM generation
        if getattr(getattr(ctx, 'llm_client', None), 'foreground_tasks', 0) > 0:
            return

        # Lazily install per-phase cooldown anchors on the agent instance.
        if not hasattr(self, '_last_journal_at'):
            self._last_journal_at = datetime.datetime.min
        if not hasattr(self, '_last_dream_at'):
            self._last_dream_at = datetime.datetime.min
        if not hasattr(self, '_last_reflection_at'):
            self._last_reflection_at = datetime.datetime.min
        if not hasattr(self, '_last_skills_auto_at'):
            self._last_skills_auto_at = datetime.datetime.min
        if not hasattr(self, '_last_prm_train_at'):
            self._last_prm_train_at = datetime.datetime.min
        if not hasattr(self, '_last_narrative_at'):
            self._last_narrative_at = datetime.datetime.min
        if not hasattr(self, '_last_selfplay_at'):
            self._last_selfplay_at = datetime.datetime.min
        # Adaptive self-play cooldown (curiosity-driven). Starts at the
        # static baseline and is rewritten after each run based on the
        # FrontierTracker's last compression delta.
        if not hasattr(self, '_current_selfplay_cooldown'):
            self._current_selfplay_cooldown = self._SELFPLAY_COOLDOWN

        idle_secs = (datetime.datetime.now() - ctx.last_activity_time).total_seconds()

        # Phase 1: Process Short-Term Journal (>120s idle)
        if idle_secs > 120 and getattr(ctx, 'journal', None) is not None:
            since_last_journal = (datetime.datetime.now() - self._last_journal_at).total_seconds()
            if since_last_journal >= self._JOURNAL_COOLDOWN:
                has_items = False
                try:
                    with ctx.journal._lock:
                        has_items = len(json.loads(ctx.journal.file_path.read_text())) > 0
                except Exception:
                    pass
                if has_items:
                    # Anchor BEFORE the await so an exception inside
                    # `process_journal_queue` doesn't leave
                    # `_last_journal_at` at its prior value, which would
                    # cause the watchdog to re-fire the failing
                    # processor on every tick. Mirrors the pattern in
                    # phases 2 / 2.5 / 2.6 / 2.7 / 3.
                    self._last_journal_at = datetime.datetime.now()
                    try:
                        await self.process_journal_queue()
                    finally:
                        self._last_journal_at = datetime.datetime.now()
                    # Do NOT reset last_activity_time here: that clock
                    # tracks *user* idleness and is what phases 2/3 gate
                    # on. Resetting it on internal work starves the later
                    # phases (phase 3 needs idle_secs > 3600, unreachable
                    # if phase 1 keeps firing every few minutes).
                    # Per-phase cooldowns already prevent immediate refire
                    # of this phase.
                    return

        # Phase 2: Deep REM Dream (10–60 min idle)
        if 600 < idle_secs <= 3600:
            since_last_dream = (datetime.datetime.now() - self._last_dream_at).total_seconds()
            if since_last_dream >= self._DREAM_COOLDOWN:
                res = await asyncio.to_thread(
                    ctx.memory_system.collection.get,
                    where={"type": "auto"},
                    limit=5
                )
                if res and len(res.get('ids', [])) >= 3:
                    if random.random() < 0.5:
                        from .dream import Dreamer
                        pretty_log("Biological Hook",
                                   "Agent is idle. Entering spontaneous REM cycle...",
                                   icon=Icons.BRAIN_THINK)
                        dreamer = Dreamer(ctx)
                        # Anchor the cooldown BEFORE the await so an exception
                        # mid-dream doesn't leave `_last_dream_at` at its prior
                        # value, which would cause the watchdog to re-fire the
                        # failing dream on every tick until the idle window
                        # naturally expires. Mirrors the fix already in place
                        # for synthetic self-play below.
                        self._last_dream_at = datetime.datetime.now()
                        try:
                            await dreamer.dream(model_name=getattr(ctx.args, 'model', 'default'))
                        finally:
                            # Do NOT reset ctx.last_activity_time here —
                            # see the comment in phase 1. Resetting it
                            # made phase 3 (self-play at >60 min idle)
                            # unreachable, because phase 2 fires every
                            # ~30 min and kept the idle clock pinned low.
                            # The `_last_dream_at` cooldown alone is
                            # sufficient to prevent phase 2 refire.
                            self._last_dream_at = datetime.datetime.now()

        # Phase 2.5: Reflection on recent failures (15-60 min idle).
        # Fires AFTER the dream window opens so dream has priority on
        # fresh idle — reflection operates on stable trajectory state,
        # not freshly-journalled material. Gated by:
        #   (a) its own cooldown (`_REFLECTION_COOLDOWN`), mirroring
        #       the dream/self-play anchor pattern so an exception can't
        #       leave the anchor un-advanced and cause re-fire every
        #       tick;
        #   (b) availability of a trajectory source + reflector on the
        #       context — the phase is a no-op when the distill
        #       pipeline isn't wired, which lets us ship the watchdog
        #       code without requiring the trajectory collector to be
        #       present in every deployment.
        # Does NOT reset `ctx.last_activity_time` — same rule as dream.
        if 900 < idle_secs <= 3600:
            since_last_reflection = (datetime.datetime.now() - self._last_reflection_at).total_seconds()
            if since_last_reflection >= self._REFLECTION_COOLDOWN:
                reflector = getattr(ctx, 'reflector', None)
                traj_collector = getattr(ctx, 'trajectory_collector', None)
                if reflector is not None and traj_collector is not None:
                    pretty_log(
                        "Biological Hook",
                        "Agent is idle. Entering reflection cycle on recent failures...",
                        icon=Icons.BRAIN_THINK,
                    )
                    # Anchor BEFORE the await so a crash mid-reflection
                    # still advances the cooldown — same defensive
                    # pattern the dream phase uses above.
                    self._last_reflection_at = datetime.datetime.now()
                    try:
                        already = getattr(ctx, '_reflected_trajectory_ids', None)
                        if already is None:
                            already = set()
                            ctx._reflected_trajectory_ids = already
                        # Prefer the composite sink when main.py wires
                        # one — it writes the reflection to JSONL AND
                        # to SkillMemory, closing the "failure → lesson
                        # retrieved on next similar turn" loop. Falls
                        # back to the plain collector if the composite
                        # isn't present.
                        _sink = getattr(ctx, 'reflection_sink', None) or traj_collector.append
                        report = await reflector.run(
                            failed_source=lambda: traj_collector.iter_trajectories(),
                            sink=_sink,
                            already_reflected=already,
                        )
                        pretty_log(
                            "Biological Hook",
                            f"Reflection complete: {report.summary()}",
                            icon=Icons.BRAIN_THINK,
                        )
                        # Reflection → self-play curriculum (proposal
                        # item #7): record which clusters the reflected
                        # failures belong to, so phase-3 self-play can
                        # drill the topics the agent fails at in REAL
                        # user turns — not just its self-play weaknesses.
                        _ft = getattr(ctx, 'frontier_tracker', None)
                        if _ft is not None:
                            try:
                                from ..memory.frontier import classify_cluster
                                for _o in report.outcomes:
                                    if not getattr(_o, 'ok', False):
                                        continue
                                    _rt = getattr(_o, 'reflected_trajectory', None)
                                    if _rt is None:
                                        continue
                                    _cl = (getattr(_rt, 'cluster', '') or '').strip()
                                    if not _cl:
                                        _cl = classify_cluster(
                                            getattr(_rt, 'user_request', '') or ''
                                        )
                                    _ft.note_reflection_failure(
                                        _cl, diagnosis=getattr(_o, 'diagnosis', ''),
                                    )
                            except Exception as _e:
                                logger.debug(
                                    "reflection→frontier wiring skipped: %s", _e,
                                )
                    except Exception as e:
                        logger.warning(f"Reflection phase failed: {e}")
                    finally:
                        self._last_reflection_at = datetime.datetime.now()

        # Phase 2.6: Skill auto-extraction (every ~2 hours during idle).
        # Pure data-level pass — no LLM call, no network, CPU-only —
        # so it's safe to run opportunistically whenever the idle
        # window covers it. Gated on `trajectory_collector` presence
        # (nothing to extract from without it) and on its own cooldown
        # anchor so a long AFK stretch doesn't produce N redundant
        # extraction passes back-to-back.
        if 900 < idle_secs <= 3600:
            since_last_skills = (datetime.datetime.now() - self._last_skills_auto_at).total_seconds()
            if since_last_skills >= self._SKILLS_AUTO_COOLDOWN:
                traj_collector = getattr(ctx, 'trajectory_collector', None)
                if traj_collector is not None:
                    self._last_skills_auto_at = datetime.datetime.now()
                    try:
                        from ..skills_auto import (
                            extract_candidates, consolidate, verify_candidate,
                        )
                        trajs = list(traj_collector.iter_trajectories())
                        if trajs:
                            candidates, report = extract_candidates(trajs, min_support=2)
                            if candidates:
                                consolidated, _ = consolidate(candidates)
                                # Graduation (proposal item #9): verify each
                                # consolidated candidate and persist the ones
                                # that clear the bar into the durable
                                # GraduatedSkillStore. Previously this phase
                                # discarded `consolidated` — pure overhead.
                                # The verify_fn is a robustness gate: a
                                # sequence only graduates if it is well-
                                # supported (>=3 validated runs) and high-
                                # confidence — so one-off coincidences don't
                                # become "proven approaches".
                                _graduated = 0
                                _store = getattr(ctx, 'auto_skill_store', None)
                                if _store is not None:
                                    def _verify_fn(c):
                                        return (getattr(c, 'support', 0) >= 3
                                                and getattr(c, 'confidence', 0.0) >= 0.5)
                                    for _cand in consolidated:
                                        try:
                                            _vr = verify_candidate(_cand, _verify_fn)
                                            if _vr.passed and _vr.action == "keep":
                                                _store.graduate(
                                                    _cand,
                                                    confidence=_vr.updated_confidence,
                                                )
                                                _graduated += 1
                                        except Exception as _ve:
                                            logger.debug(
                                                "skill graduation skipped for "
                                                "%s: %s",
                                                getattr(_cand, 'name', '?'), _ve,
                                            )
                                pretty_log(
                                    "Skills Auto",
                                    f"extracted {report.n_candidates_emitted} → "
                                    f"consolidated {len(consolidated)} → "
                                    f"graduated {_graduated} "
                                    f"(from {report.n_trajectories_seen} trajectories)",
                                    icon=Icons.BRAIN_PLAN,
                                )
                    except Exception as e:
                        logger.warning(f"Skills auto-extraction failed: {e}")
                    finally:
                        self._last_skills_auto_at = datetime.datetime.now()

        # Phase 2.7: PRM retrain on accumulated trajectories
        # (every ~3 hours during idle, configurable via
        # --prm-train-cooldown). Pure CPU pass — reads the same
        # trajectory log skills_auto reads, derives MC-discounted
        # per-step values, fits a logistic regression on hand-crafted
        # features, hot-swaps the freshly-trained model into the
        # live ``ctx.prm_scorer``. The MCTS reasoner reads scores via
        # ``ctx.prm_scorer.score`` so the swap is picked up on the
        # very next plan it scores — no agent restart required.
        #
        # Gated on:
        #   (a) ``--prm-train-cooldown`` (or the static default) so a
        #       long AFK stretch produces at most one retrain per
        #       cooldown — same anchor pattern reflection / skills_auto
        #       use, advanced BEFORE the await so an exception can't
        #       leave the cooldown un-reset.
        #   (b) presence of both a trajectory collector and a scorer
        #       on the context. When ``--prm-model`` is unset the
        #       scorer is still attached (fail-safe pass-through), so
        #       the phase still runs and can produce a first-ever
        #       checkpoint at the path stored on
        #       ``ctx._prm_checkpoint_path`` (when set) or under the
        #       default GHOST_HOME location.
        # Does NOT reset ``ctx.last_activity_time`` — same rule as
        # phases 1 / 2 / 2.5 / 2.6: that clock is the user's, not
        # the agent's.
        if 900 < idle_secs <= 3600:
            # Resolve cooldown defensively: ``ctx.args`` is a MagicMock
            # in many tests, and ``getattr(MagicMock(), 'x', None)``
            # returns a fresh MagicMock instead of None — comparing
            # that against a number raises TypeError. Type-gate the
            # override before using it.
            cooldown_override = getattr(
                getattr(ctx, 'args', None), 'prm_train_cooldown', None,
            )
            if not isinstance(cooldown_override, (int, float)):
                cooldown_override = None
            cooldown = float(cooldown_override) if cooldown_override else float(self._PRM_TRAIN_COOLDOWN)
            since_last_prm = (datetime.datetime.now() - self._last_prm_train_at).total_seconds()
            if since_last_prm >= cooldown:
                traj_collector = getattr(ctx, 'trajectory_collector', None)
                prm_scorer = getattr(ctx, 'prm_scorer', None)
                # Tight isinstance check: ``ctx`` may be a MagicMock in
                # tests, in which case ``getattr`` returns another mock
                # (truthy) for any unset attribute. Without the type
                # gate, phase 2.7 would refire on every test that
                # MagicMock-mocks the context, calling
                # ``iter_trajectories`` an extra time and inflating
                # any call-count assertion in unrelated phases.
                from ..prm.scorer import PRMScorer as _PRMScorer
                if (
                    traj_collector is not None
                    and isinstance(prm_scorer, _PRMScorer)
                ):
                    self._last_prm_train_at = datetime.datetime.now()
                    try:
                        from ..prm import PRMTrainer
                        save_path = getattr(ctx, '_prm_checkpoint_path', None)
                        if save_path is None:
                            base_mem = getattr(ctx, 'memory_dir', None)
                            if base_mem is not None:
                                save_path = base_mem.parent / "prm" / "checkpoint.json"
                        trainer = PRMTrainer()
                        report = trainer.run(
                            trajectories=traj_collector.iter_trajectories(),
                            save_path=save_path,
                        )
                        if report.fit_succeeded and trainer.model is not None:
                            prm_scorer.set_model(trainer.model)
                            # Plug into MCTS if it's attached but not yet
                            # using the PRM (first-ever fit case).
                            mcts = getattr(ctx, 'mcts_reasoner', None)
                            mcts_note = ""
                            if mcts is not None and getattr(mcts, 'prm_scorer', None) is None:
                                mcts.prm_scorer = prm_scorer
                                mcts_note = " · MCTS now scoring via PRM"
                            elif mcts is not None and getattr(mcts, 'prm_scorer', None) is prm_scorer:
                                mcts_note = " · MCTS weights refreshed"
                            pretty_log(
                                "PRM Retrain",
                                f"value model refit on idle: {report.summary()}{mcts_note}",
                                icon=Icons.BRAIN_PLAN,
                            )
                        else:
                            logger.debug(
                                "PRM idle retrain skipped: %s",
                                report.bail_reason or "unknown",
                            )
                    except Exception as e:
                        logger.warning(f"PRM retrain phase failed: {e}")
                    finally:
                        self._last_prm_train_at = datetime.datetime.now()

        # Phase 2.8: Selfhood narrative consolidation (15-60 min idle).
        # Re-generates the agent's first-person running diary from the
        # recent autobiographical experiences + self-state thread. The
        # output is what the wake-up prefix reads on each new turn, so
        # this is how a long-running session evolves its sense of
        # "what I've been up to" — the load-bearing component for
        # proposal item #5.
        # Gated on:
        #   (a) own cooldown (configurable via --self-narrative-cooldown
        #       or the static default `_NARRATIVE_COOLDOWN`), advanced
        #       BEFORE the await so a mid-consolidation crash can't
        #       leave the anchor un-reset and refire every tick — same
        #       defensive pattern as phases 2 / 2.5 / 2.6 / 2.7;
        #   (b) presence of a `self_model` on the context (no-op when
        #       --no-memory / --no-self-model — the lifespan still
        #       attaches a disabled facade so the call site doesn't
        #       branch on availability, and the inner method short-
        #       circuits on `enabled=False`).
        # Does NOT reset `ctx.last_activity_time` — same rule as every
        # other internal phase: that clock is the user's, not the agent's.
        if 900 < idle_secs <= 3600:
            cooldown_override = getattr(
                getattr(ctx, 'args', None), 'self_narrative_cooldown', None,
            )
            if not isinstance(cooldown_override, (int, float)):
                cooldown_override = None
            cooldown = float(cooldown_override) if cooldown_override else float(self._NARRATIVE_COOLDOWN)
            since_last_narrative = (datetime.datetime.now() - self._last_narrative_at).total_seconds()
            if since_last_narrative >= cooldown:
                self_model = getattr(ctx, 'self_model', None)
                if self_model is not None and getattr(self_model, 'enabled', False):
                    self._last_narrative_at = datetime.datetime.now()
                    # Meta-cognitive narrative (proposal item #4): fold the
                    # learning phases' output into the diary. Recent
                    # mistakes from SkillMemory are the convergence point —
                    # both dream-extracted heuristics and reflection-phase
                    # failure patterns land there — so the regenerated
                    # narrative reflects what the agent has LEARNED about
                    # itself, not merely what it did.
                    meta_insights = ""
                    try:
                        _sm_narr = getattr(ctx, 'skill_memory', None)
                        if _sm_narr is not None:
                            _recent_fail = _sm_narr.get_recent_failures(limit=5)
                            if _recent_fail and "No recent failures" not in _recent_fail \
                                    and "Failed to load" not in _recent_fail:
                                meta_insights = _recent_fail
                    except Exception as e:
                        logger.debug(f"meta-insight gather skipped: {e}")
                    try:
                        text = await self_model.consolidate_narrative(
                            meta_insights=meta_insights,
                        )
                        if text:
                            preview = text.replace("\n", " ")[:120]
                            pretty_log(
                                "Selfhood",
                                f"narrative regenerated ({len(text)} chars): {preview}…",
                                icon=Icons.BRAIN_THINK,
                            )
                    except Exception as e:
                        logger.warning(f"Narrative consolidation phase failed: {e}")
                    finally:
                        self._last_narrative_at = datetime.datetime.now()

        # Phase 3: Synthetic Self-Play (>60 min idle)
        if idle_secs > 3600:
            since_last_selfplay = (datetime.datetime.now() - self._last_selfplay_at).total_seconds()
            if since_last_selfplay >= self._current_selfplay_cooldown and random.random() < 0.2:
                from .dream import Dreamer
                pretty_log("Biological Hook",
                           "Agent is deeply idle. Initiating Synthetic Self-Play...",
                           icon=Icons.TOOL_CODE)
                dreamer = Dreamer(ctx)
                # C3: anchor `_last_selfplay_at` in a try/finally so a
                # synthetic_self_play exception doesn't leave the
                # cooldown un-reset. Previously, an exception skipped
                # the assignment and the biological hook would refire
                # the failing self-play on every tick until the 60-min
                # window elapsed on its own. Advance the anchor BEFORE
                # the await so even a ctrl-C / timeout mid-flight gets
                # caught by the cooldown next tick.
                self._last_selfplay_at = datetime.datetime.now()
                try:
                    await dreamer.synthetic_self_play(
                        model_name=getattr(ctx.args, 'model', 'default'),
                        is_background=True
                    )
                finally:
                    ctx.last_activity_time = datetime.datetime.now()
                    self._last_selfplay_at = datetime.datetime.now()
                    # Adapt the next cooldown from the FrontierTracker:
                    # shorter when the last run made compression progress
                    # (learning streak), longer when it was wasted.
                    tracker = getattr(ctx, 'frontier_tracker', None)
                    if tracker is not None:
                        try:
                            self._current_selfplay_cooldown = tracker.adaptive_cooldown(base=self._SELFPLAY_COOLDOWN)
                        except Exception as e:
                            logger.warning(f"Adaptive cooldown lookup failed: {e}")
                            self._current_selfplay_cooldown = self._SELFPLAY_COOLDOWN

    async def process_journal_queue(self, *, respect_idle: bool = True):
        if not hasattr(self.context, 'journal'): return

        items = await asyncio.to_thread(self.context.journal.pop_all)
        if not items: return

        pretty_log("Hippocampus", f"Waking up to process {len(items)} buffered memories...", icon=Icons.BRAIN_THINK)

        processed = 0
        for i, item in enumerate(items):
            if respect_idle:
                idle_secs = (datetime.datetime.now() - self.context.last_activity_time).total_seconds()
                if idle_secs < 30:
                    pretty_log("Hippocampus", f"User returned! Suspending memory processing. ({len(items)-i} items left)", icon=Icons.STOP)
                    await asyncio.to_thread(self.context.journal.push_front, items[i:])
                    break

            try:
                if item["type"] == "smart_memory":
                    await self.run_smart_memory_task(item["data"]["text"], item["data"]["model"], self.context.args.smart_memory)
                elif item["type"] == "post_mortem":
                    await self._execute_post_mortem(item["data"]["user"], item["data"]["tools"], item["data"]["ai"], item["data"]["model"])
                processed += 1
            except Exception as e:
                import logging
                logging.getLogger("GhostAgent").error(f"Journal processing error: {e}")
            await asyncio.sleep(0.5)

        if processed > 0:
            pretty_log("Hippocampus", f"Successfully consolidated {processed} memories.", icon=Icons.OK)

    async def run_smart_memory_task(self, interaction_context: str, model_name: str, selectivity: float):
        if not self.context.memory_system: return

        # --- ⚡ FAST-ABORT HEURISTIC (ZERO LLM COMPUTE) ---
        import re
        # We only care if the user said something memorable. Extract user lines.
        user_lines = [line.split("USER:", 1)[-1] for line in interaction_context.splitlines() if line.startswith("USER:")]
        user_text = " ".join(user_lines).lower() if user_lines else interaction_context.lower()

        # Fast exit if no identity/preference keywords are present
        if not re.search(r'\b(i|me|my|mine|prefer|always|never|remember|project|build|work|name|live|use|hate|love|want|need)\b', user_text):
            return


        async with self.memory_semaphore:
            interaction_context = interaction_context.encode('utf-8', 'replace').decode('utf-8').replace("\r", "")
            # Strict control character scrubbing for C++ JSON parsers
            interaction_context = "".join(ch for ch in interaction_context if ord(ch) >= 32 or ch in "\n\t")
            ic_lower = interaction_context.lower()
            summary_triggers = ["summarize", "summary", "recall", "tell me about", "what is", "recap", "forget", "list documents"]
            is_requesting_summary = any(w in ic_lower for w in summary_triggers)

            if is_requesting_summary and len(interaction_context) > 1500:
                return

            final_prompt = SMART_MEMORY_PROMPT + f"\n\n### EPISODE LOG:\n{interaction_context}"
            try:
                payload = {"model": model_name, "messages": [{"role": "user", "content": final_prompt}], "stream": False, "temperature": 0.1, "max_tokens": 1024}
                data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
                content = data["choices"][0]["message"]["content"]
                result_json = extract_json_from_text(content)
                score, fact, profile_up = float(result_json.get("score", 0.0)), result_json.get("fact", ""), result_json.get("profile_update", None)

                # --- UNCONDITIONAL KNOWLEDGE GRAPH INGESTION ---
                graph_triplets = result_json.get("graph_triplets", [])
                if getattr(self.context, 'graph_memory', None) and graph_triplets:
                    added = await asyncio.to_thread(self.context.graph_memory.add_triplets, graph_triplets)
                    if added and added > 0:
                        pretty_log("Graph Updated", f"Mapped {added} topological edges", icon=Icons.MEM_SAVE)

                if fact is None: fact = ""
                fact_lc = fact.lower()
                is_personal = any(w in fact_lc for w in ["user", "me", "my ", " i ", "identity", "preference", "like"])
                is_technical = any(w in fact_lc for w in ["file", "path", "code", "error", "script", "project", "repo", "build", "library", "version"])

                # Adaptive threshold override: if a self-tuning
                # AdaptiveThreshold is wired, its learned bar beats the
                # hardcoded --smart-memory CLI value. We take the higher
                # of the two so users can still tighten via CLI without
                # the adaptive side silently relaxing the gate. The
                # ``isinstance`` guard is deliberate — it keeps MagicMock
                # contexts in the test suite (where `ctx.adaptive_threshold`
                # is a generic mock) on the old CLI-only path, so existing
                # tests don't pick up a ghost 1.0 threshold from
                # ``float(MagicMock())``.
                from ..memory.adaptive_threshold import AdaptiveThreshold as _ATCls
                effective_threshold = selectivity
                at = getattr(self.context, 'adaptive_threshold', None)
                at_is_real = isinstance(at, _ATCls)
                if at_is_real:
                    try:
                        learned = float(at.get_threshold())
                        if learned > 0:
                            effective_threshold = max(selectivity, learned) if selectivity > 0 else learned
                    except Exception:
                        pass

                # Record the (score, was_useful) observation so the
                # threshold self-tunes. "Was useful" is approximated by
                # "did it clear the content gates below" — a fact that
                # passed all filters goes in as useful=True; one that
                # got filtered out is useful=False. This gives the
                # window a signal even without a user-feedback loop.
                passed_basic_gate = (
                    score >= effective_threshold
                    and bool(fact)
                    and 5 <= len(fact) <= 200
                    and "none" not in fact_lc
                )
                if at_is_real:
                    try:
                        at.record(score=score, was_useful=passed_basic_gate)
                    except Exception:
                        pass

                if score >= effective_threshold and fact and len(fact) <= 200 and len(fact) >= 5 and "none" not in fact_lc:
                    if score >= 0.9 and not (is_personal or is_technical):
                        pretty_log("Auto Memory Skip", f"Discarded generic knowledge: {fact}", icon=Icons.STOP)
                        return
                    memory_type = "identity" if (score >= 0.9 and profile_up) else "auto"

                    # --- CONTRADICTION ENGINE (LLM-Driven Belief Revision) ---
                    try:
                        candidates = await asyncio.to_thread(self.context.memory_system.search_advanced, fact, limit=3)
                        ids_to_delete = []
                        old_facts = []

                        if candidates:
                            for c in candidates:
                                if c.get('score', 1.0) < 0.6: # Broad threshold to catch potential semantic collisions
                                    old_facts.append({"id": c['id'], "text": c['text']})

                        if old_facts:
                            eval_prompt = f"NEW FACT:\n{fact}\n\nOLD FACTS:\n" + "\n".join([f"ID: {f['id']} | TEXT: {f['text']}" for f in old_facts]) + "\n\nAnalyze if the NEW FACT contradicts, updates, or supersedes any OLD FACTS. Return ONLY a JSON object with a list of 'ids' to delete. If they safely coexist (e.g. they refer to different topics/projects), return an empty list.\n\nExample: {{\"ids\": [\"ID:123\"]}}"
                            eval_payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Belief Revision Engine. Output JSON."}, {"role": "user", "content": eval_prompt}], "temperature": 0.0, "max_tokens": 1024}
                            eval_data = await self.context.llm_client.chat_completion(eval_payload, use_worker=True, is_background=True)
                            eval_res = extract_json_from_text(eval_data["choices"][0]["message"]["content"])

                            raw_ids = eval_res.get("ids", [])
                            ids_to_delete = [str(i).replace("ID: ", "").replace("ID:", "").strip() for i in raw_ids]

                        if ids_to_delete:
                            await asyncio.to_thread(self.context.memory_system.collection.delete, ids=ids_to_delete)
                            pretty_log("Belief Revision", f"Erased {len(ids_to_delete)} outdated/contradicting memories.", icon=Icons.CUT)
                            # Log the contradiction for explainability
                            contradiction_log = getattr(self.context, 'contradiction_log', None)
                            if contradiction_log is not None:
                                try:
                                    await asyncio.to_thread(
                                        contradiction_log.record,
                                        fact, old_facts, ids_to_delete,
                                        reason="LLM-driven belief revision"
                                    )
                                except Exception as cl_err:
                                    logger.debug(f"Contradiction log write failed: {cl_err}")

                    except Exception as ce:
                        logger.error(f"Contradiction Engine error: {ce}")

                    # Save the new fact (bypassing the old simplistic smart_update math check, since we just logically validated it)
                    from ..utils.helpers import get_utc_timestamp
                    await asyncio.to_thread(self.context.memory_system.add, fact, {"timestamp": get_utc_timestamp(), "type": memory_type})
                    pretty_log("Auto Memory Store", f"[{score:.2f}] {fact}", icon=Icons.MEM_SAVE)

                    if memory_type == "identity" and self.context.profile_memory:
                        await asyncio.to_thread(
                            self.context.profile_memory.update,
                            profile_up.get("category", "notes"),
                            profile_up.get("key", "info"),
                            profile_up.get("value", fact)
                        )


            except Exception as e: logger.error(f"Smart memory task failed: {e}")

    async def _execute_post_mortem(self, last_user_content: str, tools_run: list, final_ai_content: str, model: str):
        try:
            history_summary = f"User: {last_user_content}\n"
            for t_msg in tools_run[-5:]:
                history_summary += f"Tool {t_msg.get('name', 'unknown')}: {str(t_msg.get('content', ''))[:200]}\n"

            # Aggressively strip lone surrogates and raw control characters for C++ backends
            def _clean_for_cpp(text: str) -> str:
                if not isinstance(text, str): return str(text)
                text = text.encode('utf-8', 'replace').decode('utf-8')
                return "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")

            last_user_content = _clean_for_cpp(last_user_content)
            final_ai_content = _clean_for_cpp(final_ai_content)
            history_summary = _clean_for_cpp(history_summary)

            learn_prompt = f"### TASK POST-MORTEM\nReview this interaction. The agent either struggled and succeeded, OR failed completely. Identify the core technical error, hallucination, or bad strategy. Extract a concrete rule to fix or avoid this in the future.\n\nHISTORY:\n{history_summary}\n\nFINAL AI: {final_ai_content[:500]}\n\nReturn ONLY a JSON object with 'task', 'mistake', and 'solution' (what to do instead next time/the anti-pattern to avoid). If no unique technical lesson is found, return null."

            payload = {"model": model, "messages": [{"role": "system", "content": "You are a Meta-Cognitive Analyst. Output JSON."}, {"role": "user", "content": learn_prompt}], "temperature": 0.1, "max_tokens": 1024}
            l_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            l_content = str(l_data["choices"][0]["message"].get("content") or "")
            if l_content and "null" not in l_content.lower():
                l_json = extract_json_from_text(l_content)
                if all(k in l_json for k in ["task", "mistake", "solution"]):
                    if getattr(self.context, 'skill_memory', None):
                        await asyncio.to_thread(
                            self.context.skill_memory.learn_lesson,
                            l_json["task"], l_json["mistake"], l_json["solution"],
                            memory_system=self.context.memory_system
                        )
                    pretty_log("Auto-Learning", "New lesson captured automatically", icon=Icons.IDEA)
        except Exception as e:
            logger.error(f"Post-mortem failed: {e}")

    # =================================================================
    # ARCHITECTURAL OPTIMISATION #4: REQUEST-SCOPED CACHE
    # -----------------------------------------------------------------
    # `_RequestState` is a per-request memoizer for values that the turn
    # loop used to recompute every iteration even though they are
    # constant for the lifetime of one request:
    #   * profile context string         (was: thread-call per turn)
    #   * active tool definitions        (was: re-filtered per turn)
    #   * tool XML schema                 (was: re-serialised per turn)
    #   * skill playbook (per-query)     (was: thread-call per turn)
    #   * sandbox state                  (was: re-listed per turn)
    #
    # Mutating tools call `state.invalidate(...)` to drop the relevant
    # entry; the next consumer recomputes it lazily.
    # =================================================================
    class _RequestState:
        __slots__ = (
            "_agent_ref", "_profile_str", "_profile_loaded",
            "_active_tool_defs_cache", "_xml_schema_cache",
            "_skill_playbook_cache", "_sandbox_state",
        )

        def __init__(self, agent_ref):
            self._agent_ref = agent_ref
            self._profile_str = None
            self._profile_loaded = False
            # Cache by (intent_query) so a query-shift between turns still
            # benefits when the same query repeats.
            self._active_tool_defs_cache: Dict[str, list] = {}
            self._xml_schema_cache: Dict[str, str] = {}
            self._skill_playbook_cache: Dict[str, str] = {}
            self._sandbox_state = None

        async def get_profile_str(self) -> str:
            if not self._profile_loaded:
                pm = getattr(self._agent_ref.context, "profile_memory", None)
                if pm is not None:
                    try:
                        raw = await asyncio.to_thread(pm.get_context_string)
                        self._profile_str = (raw or "").replace("\r", "")
                    except Exception:
                        self._profile_str = ""
                else:
                    self._profile_str = ""
                self._profile_loaded = True
            return self._profile_str or ""

        def get_active_tool_defs(self, query: str) -> list:
            key = query or ""
            cached = self._active_tool_defs_cache.get(key)
            if cached is not None:
                return cached
            from ..tools.registry import get_active_tool_definitions
            defs = get_active_tool_definitions(self._agent_ref.context, query=key) or []
            self._active_tool_defs_cache[key] = defs
            return defs

        def get_xml_schema(self, tool_defs: list) -> str:
            # Hash the function-name set — schema is a pure function of it.
            key = _xml_schema_key(tool_defs)
            cached = self._xml_schema_cache.get(key)
            if cached is not None:
                return cached
            funcs_only = [t["function"] for t in tool_defs]
            xml = _json_to_xml_schema_cached(_freeze_funcs(funcs_only))
            self._xml_schema_cache[key] = xml
            return xml

        async def get_skill_playbook(self, query: str) -> str:
            sm = getattr(self._agent_ref.context, "skill_memory", None)
            if sm is None:
                return ""
            key = query or ""
            cached = self._skill_playbook_cache.get(key)
            if cached is not None:
                return cached
            try:
                playbook = await asyncio.to_thread(
                    sm.get_playbook_context,
                    query=key,
                    memory_system=getattr(self._agent_ref.context, "memory_system", None),
                )
            except Exception:
                playbook = ""
            playbook = playbook or ""
            self._skill_playbook_cache[key] = playbook
            return playbook

        async def get_sandbox_state(self):
            if self._sandbox_state is not None:
                return self._sandbox_state
            try:
                from ..tools.file_system import tool_list_files
                self._sandbox_state = await tool_list_files(
                    sandbox_dir=self._agent_ref.context.sandbox_dir,
                    memory_system=getattr(self._agent_ref.context, "memory_system", None),
                )
            except Exception:
                self._sandbox_state = ""
            return self._sandbox_state

        def invalidate_sandbox(self):
            self._sandbox_state = None

        def invalidate_skill_playbook(self):
            self._skill_playbook_cache.clear()

    # Allowlist of phrases that the trivial fast path is safe to intercept.
    # Anything not on this list (even a 1-word "test") falls through to the
    # full turn loop. The list is intentionally tight on **action verbs**
    # but generous on **pure-greeting filler words** (there/friend/buddy/
    # everyone/ghost/you/etc.) so common conversational openings like
    # "hello there", "hi friend", "hey ghost" don't get pushed into the
    # heavy turn loop and inherit the technical persona.
    _STRICT_GREETING_TOKENS = frozenset({
        # Greeting tokens
        "hi", "hello", "hey", "yo", "sup", "howdy", "hola", "heya", "hiya",
        # Acknowledgements
        "thanks", "thank", "ty", "thx", "tysm",
        "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome", "neat",
        "bye", "goodbye", "cya", "gn", "gm", "morning", "evening", "afternoon",
        "lol", "lmao", "haha", "hehe", "yep", "yup", "yeah", "nope", "nah",
        # Polite filler / conversational neutralisers — these alone are NOT
        # actionable, so any 2-word combo of allowlist tokens is safe.
        "there", "friend", "buddy", "everyone", "all", "you", "ghost",
        "agent", "bot", "again", "back", "today", "tonight", "now",
        "much", "lot", "very", "so", "the", "a", "u", "ya",
        # Emoji
        "👍", "❤️", "🙏", "😊", "🙂",
    })
    _STRICT_GREETING_PHRASES = (
        "good morning", "good afternoon", "good evening", "good night",
        "thank you", "thanks a lot", "thanks so much", "thank you so much",
        "see you", "see ya", "talk later", "got it", "sounds good",
        "no worries", "all good", "fair enough",
    )

    @classmethod
    def _is_strict_trivial_chat(cls, lc: str) -> bool:
        """Return True only when `lc` is unambiguously a greeting / ack.

        We deliberately reject anything with a question mark, a '/' command,
        a tool-name keyword, file extensions, or any token outside the small
        allowlist. The cost of missing a real greeting is one extra ~2 s
        request; the cost of intercepting a tool-bearing request is a
        broken response."""
        if not lc:
            return False
        lc = lc.strip()
        if not lc:
            return False
        # Hard rejects — any of these means the user wants real work done.
        for char in ("?", "/", "<", ">", "{", "}", "@"):
            if char in lc:
                return False
        if any(ext in lc for ext in (".py", ".js", ".html", ".css", ".sh", ".md", ".json")):
            return False
        # Strip punctuation for token check.
        import string as _string
        normalised = lc.translate(str.maketrans("", "", _string.punctuation))
        normalised = normalised.strip()
        if not normalised:
            return False
        for phrase in cls._STRICT_GREETING_PHRASES:
            if phrase in normalised:
                return True
        tokens = normalised.split()
        if not tokens:
            return False
        # All tokens must be in the allowlist.
        return all(tok in cls._STRICT_GREETING_TOKENS for tok in tokens)

    async def _route_query_expansion(self,
                                     prev_ai_snippet: str,
                                     user_intent: str,
                                     legacy_fallback: str) -> str:
        """ARCHITECTURAL OPTIMISATION #2 — route query expansion through the
        small worker model when one is available.

        Returns the expanded query (a single line of text). Always falls
        back to `legacy_fallback` when the router can't help, so the
        legacy `Context: ... | User intent: ...` shape is preserved when
        no worker pool is configured."""
        client = getattr(self.context, "llm_client", None)
        if client is None or not hasattr(client, "route"):
            return legacy_fallback
        if not getattr(client, "worker_clients", None):
            return legacy_fallback

        try:
            from .llm import RoutingTask
        except Exception:
            return legacy_fallback

        prompt = (
            "Rewrite the user's short follow-up question into a single "
            "self-contained search query that captures the implied subject. "
            "Output ONLY the rewritten query, nothing else.\n\n"
            f"PRIOR ASSISTANT REPLY:\n{prev_ai_snippet}\n\n"
            f"USER FOLLOW-UP:\n{user_intent}\n\n"
            "REWRITTEN QUERY:"
        )
        payload = {
            "model": getattr(self.context.args, "model", "default"),
            "messages": [
                {"role": "system", "content": "You are a query rewriter. Output one line."},
                {"role": "user", "content": prompt},
            ],
        }
        result = await client.route(
            task=RoutingTask.EXPAND_QUERY,
            payload=payload,
            max_tokens=64,
            temperature=0.0,
            fallback=legacy_fallback,
        )
        if not isinstance(result, str) or not result.strip():
            return legacy_fallback
        cleaned = result.strip().splitlines()[0].strip()
        return cleaned or legacy_fallback

    async def _handle_trivial_chat(self,
                                   last_user_content: str,
                                   messages: List[Dict[str, Any]],
                                   model: str,
                                   stream_response: bool,
                                   req_id: str):
        """Fast path for `is_trivial_greeting` requests.

        Returns the standard ``(content, created_time, req_id)`` tuple on
        success, or ``None`` if the fast path declines (which makes the
        caller fall through to the full turn loop). We decline if anything
        about the request looks unsafe to short-circuit (no LLM client,
        unparsable history, LLM call failed, etc.).

        IMPORTANT: this method ALWAYS returns ``content`` as a plain string
        even for streaming requests. The route layer's ``stream_openai``
        helper handles SSE-wrapping the string for streaming clients
        (`api/routes.py:163`). Returning a custom async generator here
        risked closure / iterator-lifecycle bugs and is no longer used.
        The trade-off: streaming clients see the response after the model
        finishes (~300ms total) rather than token-by-token, but the entire
        response is short by definition (256 tokens max), so the perceived
        difference is negligible — and the path is now bug-free."""
        if self.context.llm_client is None:
            return None

        created_time = int(datetime.datetime.now().timestamp())

        # A pared-down system prompt: just persona + profile snapshot.
        # NO tool schemas, NO dynamic state, NO skill playbook. The model
        # is being asked to respond conversationally; tools cannot help.
        try:
            profile_context = await asyncio.to_thread(
                self.context.profile_memory.get_context_string
            ) if self.context.profile_memory else ""
        except Exception:
            profile_context = ""
        profile_context = (profile_context or "").replace("\r", "")

        profile_block = ""
        if profile_context:
            profile_block = "### USER PROFILE ###\n" + profile_context + "\n"
        lite_system = (
            "You are Ghost, a concise conversational AI. The user has sent a "
            "short greeting or trivial message. Reply warmly, in one or two "
            "sentences. Do NOT call any tools. Do NOT mention internal state. "
            "Do NOT preface with filler.\n\n"
            + profile_block
        )

        # Keep only the last few user/assistant turns so the model has a tiny
        # bit of conversational context. Drop tool messages and system slots.
        history_tail = [
            m for m in messages[-12:]
            if m.get("role") in ("user", "assistant")
        ]

        req_messages = [{"role": "system", "content": lite_system}]
        for m in history_tail:
            content_val = m.get("content", "")
            if isinstance(content_val, list):
                content_val = " ".join(
                    block.get("text", "") for block in content_val
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            req_messages.append({"role": m["role"], "content": str(content_val)})

        if not req_messages or req_messages[-1].get("role") != "user":
            req_messages.append({"role": "user", "content": last_user_content})

        payload = {
            "model": model,
            "messages": req_messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 256,
            "stream": False,
        }

        pretty_log(
            "Trivial Fast Path",
            f"Bypassing turn loop ({'stream' if stream_response else 'sync'})",
            icon=Icons.OK,
        )

        # Tag the heartbeat both before and after the LLM call so the
        # biological watchdog never spuriously fires mid-bypass.
        self.context.last_activity_time = datetime.datetime.now()

        try:
            data = await self.context.llm_client.chat_completion(payload)
        except Exception as e:
            logger.error(f"Trivial fast path LLM call failed: {e}")
            return None  # fall through to full path

        try:
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as e:
            logger.error(f"Trivial fast path response parse failed: {e}")
            return None

        # If the model returned absolutely nothing, fall through so the
        # full path can take a second crack at it.
        if not content.strip():
            logger.warning("Trivial fast path returned empty content; falling through to full path")
            return None

        # Some models leak <think>...</think> blocks even on small prompts —
        # strip them so the user sees only the visible reply.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        self.context.last_activity_time = datetime.datetime.now()
        pretty_log("Trivial Fast Path", f"Resolved in {len(content)} chars", icon=Icons.OK)

        return content, created_time, req_id

    async def handle_chat(self, body: Dict[str, Any], background_tasks, request_id: Optional[str] = None):
        req_id = request_id or str(uuid.uuid4())[:8]
        token = request_id_context.set(req_id)
        self.context.last_activity_time = datetime.datetime.now()

        # Continuous self-play interrupt: if a `self_play_loop` task is
        # active, any new user message implicitly pauses it. The loop
        # checks `selfplay_loop_stop` between cycles and between the
        # adaptive cool-off waits, so flipping the event here is enough.
        # We don't await the task — the cycle may still be mid-flight and
        # we don't want to block the user's turn on it.
        try:
            loop_task = getattr(self.context, "selfplay_loop_task", None)
            stop_event = getattr(self.context, "selfplay_loop_stop", None)
            if loop_task is not None and not loop_task.done() and stop_event is not None:
                stop_event.set()
                pretty_log(
                    "Self-Play Loop",
                    "User message received — signalling loop to stop after current cycle.",
                    icon=Icons.STOP,
                )
        except Exception:
            pass

        try:
            async with self.agent_semaphore:
                char_budget = int(self.context.args.max_context * 3.5)
                pretty_log("Request Initialized", special_marker="BEGIN")
                messages, model, stream_response = body.get("messages", []), body.get("model", "qwen-3.6-35b-a3"), body.get("stream", False)

                # Pre-allocate the trajectory id for THIS turn. Several
                # in-turn paths (Perfection-Protocol's lesson save, the
                # post-turn user-correction retraction on the *next*
                # turn) need a stable id BEFORE `_record_turn_trajectory`
                # runs at end-of-turn. Threading the same id through to
                # the eventual `Trajectory(id=...)` keeps lesson
                # provenance (`source_trajectory_id`) and the persisted
                # trajectory in sync — the prerequisite for
                # `SkillMemory.retract_lessons_from_trajectory` to
                # actually find the lessons it needs to scrub.
                current_trajectory_id = uuid.uuid4().hex

                if len(messages) > 500:
                    messages = [m for m in messages if m.get("role") == "system"] + messages[-500:]
                for m in messages:
                    if isinstance(m.get("content"), str): m["content"] = m["content"].replace("\r", "")

                last_user_content_raw = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                if isinstance(last_user_content_raw, list):
                    last_user_content = " ".join([i.get("text", "") for i in last_user_content_raw if isinstance(i, dict) and i.get("type") == "text"])
                else:
                    last_user_content = str(last_user_content_raw)
                lc = last_user_content.lower()
                # Expose the current turn's user text to tools via the
                # context. Tools that need to validate user intent
                # (e.g. `tool_self_play`'s LLM-hallucination guard) read
                # `context.last_user_content`. Set BEFORE any tool
                # dispatch path can run in this turn.
                self.context.last_user_content = last_user_content

                # Stage-1 self-improvement: user-correction promotion.
                # If `last_user_content` looks like a correction of the
                # immediately-prior assistant turn, promote that
                # trajectory to FAILED via the corrections sidecar AND
                # schedule single-trajectory reflection. The biological
                # watchdog phase 2.5 is the 15-60 min idle backstop;
                # this hook closes the loop in real-time interactive
                # chat where the user rarely goes idle long enough for
                # the backstop to trigger. Non-fatal: a failure here
                # must never break the user turn.
                try:
                    self._maybe_promote_prior_turn_via_user_correction(
                        messages, last_user_content
                    )
                except Exception as e:
                    logger.debug(
                        "user-correction promotion skipped: %s: %s",
                        type(e).__name__, e,
                    )

                # Stage-1 self-improvement: consult the complexity
                # router (if wired) and stash the decision on body so
                # downstream dispatchers can pick a cheap / full path.
                # The dispatcher is FAIL-SAFE: if the router isn't
                # trained, or the prediction is low-confidence, it
                # returns `escalated=True` with the full pool list,
                # so this hook can never downgrade capability — only
                # reduce cost when the router is both confident AND
                # recognises an easy request. Non-fatal on error.
                try:
                    dispatcher = getattr(self.context, 'complexity_dispatcher', None)
                    if dispatcher is not None and last_user_content:
                        prev_ai_for_router = next(
                            (m.get("content", "") for m in reversed(messages[:-1])
                             if m.get("role") == "assistant" and isinstance(m.get("content"), str)),
                            "",
                        )
                        decision = dispatcher.route(
                            last_user_content,
                            prior_turn_text=str(prev_ai_for_router or "")[:1000],
                        )
                        body["_router_decision"] = {
                            "label": decision.label,
                            "confidence": decision.confidence,
                            "allowed_pools": list(decision.allowed_pools),
                            "escalated": decision.escalated,
                            "reason": decision.reason,
                        }
                except Exception as e:
                    logger.debug(f"complexity router consultation skipped: {e}")

                coding_keywords = [r"\bpython\b", r"\bbash\b", r"\bsh\b", r"\bscript\b", r"\bcode\b", r"\bdef\b", r"\bimport\b", r"\bhtml\b", r"\bcss\b", r"\bjs\b", r"\bjavascript\b", r"\btypescript\b", r"\breact\b", r"\bweb\b", r"\bfrontend\b"]
                coding_actions = [r"\bwrite\b", r"\brun\b", r"\bexecute\b", r"\bdebug\b", r"\bfix\b", r"\bcreate\b", r"\bgenerate\b", r"\bcount\b", r"\bcalculate\b", r"\banalyze\b", r"\bscrape\b", r"\bplot\b", r"\bgraph\b", r"\bbuild\b", r"\bdevelop\b"]
                has_coding_intent = False

                if any(re.search(k, lc) for k in coding_keywords):
                    if any(re.search(a, lc) for a in coding_actions):
                        has_coding_intent = True
                if any(ext in lc for ext in [".py", ".js", ".html", ".css", ".ts", ".tsx", ".jsx", ".sh"]) or re.search(r'\bscript\b', lc):
                    has_coding_intent = True

                dba_keywords = [r"\bsql\b", r"\bpostgres\b", r"\bpostgresql\b", r"\bpsql\b", r"\bdatabase\b", r"\bpg_stat\b", r"\bexplain analyze\b", r"\bquery\b", r"\bcte\b", r"\brdbms\b", r"\bdba\b", r"\bschema\b", r"\bvacuum\b", r"\bmvcc\b"]
                if any(re.search(k, lc) for k in dba_keywords):
                    has_coding_intent = True

                meta_keywords = [r"\btitle\b", r"\bname this\b", r"\brename\b", r"\bsummary\b", r"\bsummarize\b", r"\bcaption\b", r"\bdescribe\b"]
                is_meta_task = any(re.search(k, lc) for k in meta_keywords)
                if re.match(r'^[\d\s\+\-\*\/\(\)\=\?]+$', lc):
                    has_coding_intent = False

                # ARCHITECTURAL OPTIMISATION #4: Request-scoped lazy cache.
                # Profile / playbook / tool defs / XML schema all become
                # one-and-done per request instead of per-turn.
                request_state = GhostAgent._RequestState(self)
                profile_context = await request_state.get_profile_str()

                working_memory_context = ""



                base_prompt = SYSTEM_PROMPT.replace("{{PROFILE}}", profile_context)

                # Selfhood wake-up prefix (recognition layer, proposal item #4).
                # Splices the agent's own past — autobiographical
                # experiences, open questions, mood, the running diary —
                # into the system prompt as first-person continuity
                # material. The block is bounded by SELFHOOD:BEGIN /
                # SELFHOOD:END markers so evaluators can strip it. No-op
                # when --no-memory / --no-self-model / no prior state.
                # Non-fatal: a failure here must never break a user turn.
                #
                # Strict isinstance check on the SelfModel (NOT just
                # truthiness) — `ctx` is a MagicMock in many tests and
                # `getattr(MagicMock(), 'self_model', None)` returns a
                # fresh MagicMock that is truthy AND whose
                # `build_wakeup_prefix()` returns yet another truthy
                # MagicMock. Without the type gate, base_prompt would
                # become a MagicMock-arithmetic object and downstream
                # `messages[0]["content"]` would be unverifiable
                # garbage. Same defensive pattern as the PRM phase.
                try:
                    from ..selfhood import SelfModel as _SelfModel
                    self_model = getattr(self.context, 'self_model', None)
                    if isinstance(self_model, _SelfModel) and getattr(self_model, 'enabled', False):
                        # Pass the current request as `query` so the
                        # wake-up prefix surfaces RELEVANT past experiences
                        # (recall keyed to what's being asked now), not
                        # just the most recent N. Proposal item #2.
                        wakeup_prefix = self_model.build_wakeup_prefix(
                            recent_experiences_n=3,
                            query=last_user_content,
                        )
                        if isinstance(wakeup_prefix, str) and wakeup_prefix:
                            base_prompt = wakeup_prefix + "\n" + base_prompt
                except Exception as e:
                    logger.debug(f"selfhood wake-up prefix skipped: {type(e).__name__}: {e}")

                # Uncertainty context injection (proposal item #6).
                # Surface the agent's RECURRING blind-spots — unknowns it
                # has flagged turn after turn — into the system prompt so
                # it reasons WITH its own durable uncertainty in view,
                # instead of the tracker only producing an after-the-fact
                # footer. Non-fatal: must never break a user turn.
                try:
                    _utracker = getattr(self.context, 'uncertainty_tracker', None)
                    if _utracker is not None:
                        _uctx = _utracker.persisted_context()
                        # isinstance(str) guard: under MagicMock test
                        # contexts `persisted_context()` returns a mock,
                        # and `base_prompt + mock` would silently corrupt
                        # the system prompt. Same defensive pattern the
                        # selfhood wake-up path uses.
                        if isinstance(_uctx, str) and _uctx:
                            base_prompt = base_prompt + "\n\n" + _uctx
                except Exception as e:
                    logger.debug(f"uncertainty context injection skipped: {e}")

                # Graduated-skill injection (proposal item #9). Surface
                # auto-acquired, verification-passed tool sequences that
                # match this request as "proven approaches", so a skill
                # the agent mined from its own validated runs actually
                # gets reused instead of sitting unread on disk. Non-fatal.
                try:
                    _askstore = getattr(self.context, 'auto_skill_store', None)
                    if _askstore is not None and last_user_content:
                        _skblock = _askstore.format_for_prompt(
                            query=last_user_content, limit=3,
                        )
                        # isinstance(str) guard — see the uncertainty
                        # injection above: a MagicMock context must not
                        # corrupt base_prompt.
                        if isinstance(_skblock, str) and _skblock:
                            base_prompt = base_prompt + "\n\n" + _skblock
                except Exception as e:
                    logger.debug(f"graduated-skill injection skipped: {e}")

                # Deep-reason MCTS lookahead (proposal item #8). For
                # requests the complexity router judges HARD (or had to
                # escalate), run the MCTS action search and inject its
                # top-ranked next-action as a planning hint. This is the
                # wiring that makes `select_best_action` load-bearing —
                # previously the reasoner was constructed under
                # --deep-reason but never actually called in the turn
                # loop. Gated on: mcts_reasoner present (it is None
                # unless --deep-reason) AND a hard/escalated router
                # verdict, so the extra LLM round-trips land only where
                # the depth pays off. Bounded by a wall-clock timeout so
                # a slow search can never pin the turn. Non-fatal.
                try:
                    _mcts = getattr(self.context, 'mcts_reasoner', None)
                    _rd = body.get("_router_decision") or {}
                    _is_hard = (_rd.get("label") == "hard") or bool(_rd.get("escalated"))
                    if _mcts is not None and _is_hard and last_user_content:
                        from ..tools.registry import TOOL_DEFINITIONS as _MCTS_TD
                        _tool_names = [
                            t["function"]["name"] for t in _MCTS_TD
                            if t.get("function", {}).get("name")
                        ]
                        _winner = await asyncio.wait_for(
                            _mcts.select_best_action(
                                task=last_user_content,
                                plan_state="(turn start — no actions taken yet)",
                                available_tools=_tool_names,
                                context=working_memory_context or "",
                            ),
                            timeout=75.0,
                        )
                        _w_desc = getattr(_winner, 'description', '') if _winner else ''
                        if _winner is not None and isinstance(_w_desc, str) and _w_desc:
                            _hint = (
                                "### DEEP-REASON LOOKAHEAD\n"
                                "An MCTS search over candidate approaches ranked "
                                "this next step highest:\n"
                                f"  - {_w_desc}"
                            )
                            _w_tool = getattr(_winner, 'tool_name', '')
                            _w_risk = getattr(_winner, 'risk_notes', '')
                            if isinstance(_w_tool, str) and _w_tool:
                                _hint += f" (suggested tool: {_w_tool})"
                            if isinstance(_w_risk, str) and _w_risk:
                                _hint += f"\n  Watch out for: {_w_risk}"
                            _hint += (
                                "\nTreat this as a strong hint, not a mandate — "
                                "deviate if the task clearly needs another approach."
                            )
                            base_prompt = base_prompt + "\n\n" + _hint
                            pretty_log(
                                "Deep Reason",
                                f"MCTS lookahead → {_w_desc[:80]}",
                                icon=Icons.MCTS_TREE,
                            )
                except asyncio.TimeoutError:
                    logger.debug("MCTS lookahead timed out; proceeding without hint")
                except Exception as e:
                    logger.debug(f"MCTS lookahead skipped: {type(e).__name__}: {e}")

                # Dynamic "Perfect It" Protocol Injection
                if getattr(self.context.args, 'perfect_it', False):
                    # Inject as item 5 before Tool Orchestration
                    last_tool_content = "None"
                    if locals().get('tools_run_this_turn') and len(tools_run_this_turn) > 0:
                        last_tool_content = str(tools_run_this_turn[-1].get('content', ''))[:15000]

                    base_prompt = base_prompt.replace(
                        "### TOOL ORCHESTRATION",
                        f'5. THE "PERFECT IT" PROTOCOL: Upon successfully completing a complex technical task, analyze the result (Last Tool Output: {last_tool_content}) and proactively suggest one concrete way to optimize it.\n\n### TOOL ORCHESTRATION'
                    )
                base_prompt += working_memory_context

                # Tool schemas are now generated dynamically inside the turn loop
                # base_prompt is kept clean here

                active_persona = ""
                if has_coding_intent and not is_meta_task:
                    pretty_log("Mode Switch", "Ghost Specialist Activated", icon=Icons.MODE_GHOST)
                    active_persona = f"{SPECIALIST_SYSTEM_PROMPT.replace('{{PROFILE}}', profile_context)}\n\n"

                # base_prompt += active_persona  <-- RELOCATED to user message for cache efficacy

                found_system = False
                for m in messages:
                    if m.get("role") == "system": m["content"] = base_prompt; found_system = True; break
                if not found_system: messages.insert(0, {"role": "system", "content": base_prompt})

                is_fact_check = "fact-check" in lc or "verify" in lc

                tool_action_verbs = [
                    "search", "download", "run", "execute", "schedule", "read", "fetch",
                    "calculate", "count", "summarize", "find", "open", "check", "test",
                    "delete", "remove", "rename", "move", "copy", "scrape", "ingest",
                    "create", "draw", "make", "generate", "picture", "image", "paint",
                    "play", "train", "practice"
                ]
                has_action_verb = any(v in lc for v in tool_action_verbs)

                is_conversational = not has_coding_intent and not is_meta_task and not has_action_verb

                # OPTIMIZATION: Detect trivial greetings to bypass heavy processing
                is_trivial_greeting = (
                    is_conversational and
                    len(lc.split()) <= 5 and
                    "remember" not in lc and
                    "previous" not in lc
                )

                # ============================================================
                # ARCHITECTURAL OPTIMISATION #3: TRIVIAL-REQUEST FAST PATH
                # ------------------------------------------------------------
                # A "hi", "thanks", "ok cool" used to traverse the entire
                # turn loop: prune_context + tool schema gen + skill recall +
                # planner gate + KV-cache rebuild, even though none of it can
                # affect the response. This bypass collapses ~2.5s of work
                # into ~300ms by skipping straight to one tool-less LLM call.
                #
                # We use a STRICT greeting detector (match against an
                # explicit allowlist) — `is_trivial_greeting` alone is too
                # broad: it fires on any 5-word conversational message and
                # would intercept legitimate tool-bearing requests.
                # ============================================================
                if is_trivial_greeting and last_user_content and self._is_strict_trivial_chat(lc):
                    fast_result = await self._handle_trivial_chat(
                        last_user_content=last_user_content,
                        messages=messages,
                        model=model,
                        stream_response=stream_response,
                        req_id=req_id,
                    )
                    if fast_result is not None:
                        return fast_result

                should_fetch_memory = (not is_fact_check and not is_trivial_greeting)

                # --- COGNITIVE EVENT BUS HYDRATION ---
                # The previous sequential vector + graph blocks have collapsed
                # into a single MemoryBus.hydrate_context() call which fans
                # out to Vector / Graph / Skill stores in parallel and fuses
                # their rankings via Reciprocal Rank Fusion.
                fetched_context = ""
                if last_user_content and should_fetch_memory:
                    search_query = last_user_content
                    # Contextual query expansion for short follow-ups so the
                    # bus hydrates against the resolved intent, not the pronoun.
                    # ARCHITECTURAL OPTIMISATION #2: when a worker pool is
                    # available, route query expansion through a small model
                    # instead of dumb string concat. Falls back to the legacy
                    # concat shape if the router declines (no worker pool /
                    # error), so the test suite that asserts on the literal
                    # `Context: ... | User intent: ...` shape stays green.
                    if len(search_query.split()) < 10 and len(messages) >= 2:
                        prev_ai = next((m.get("content", "") for m in reversed(messages[:-1]) if m.get("role") == "assistant"), "")
                        if isinstance(prev_ai, str) and prev_ai:
                            legacy_expansion = f"Context: {prev_ai[:200].strip()} | User intent: {search_query}"
                            search_query = await self._route_query_expansion(
                                prev_ai_snippet=prev_ai[:200].strip(),
                                user_intent=search_query,
                                legacy_fallback=legacy_expansion,
                            )

                    bus = self._get_memory_bus()
                    if bus is not None:
                        try:
                            fetched_context = await bus.hydrate_context(search_query)
                            if fetched_context:
                                fetched_context = fetched_context.replace("\r", "")
                                pretty_log("Memory Bus", f"Hydrated context for: {search_query}", icon=Icons.BRAIN_CTX)
                        except Exception as e:
                            logger.error(f"MemoryBus hydration failed: {e}")

                fetched_playbook = ""  # Now dynamically populated inside the loop

                # ============================================================
                # ARCHITECTURAL OPTIMISATION #1: KV-CACHE-STABLE SYSTEM PROMPT
                # ------------------------------------------------------------
                # The system slot used to be rebuilt per turn (persona +
                # skill_instruction + tool schemas) which changed bytes from
                # turn to turn and invalidated upstream KV-cache prefixes.
                # We now lock the system slot to persona + skill_instruction
                # only — tool schemas move into the per-turn user-message
                # header (which already changes every turn anyway because of
                # the timestamp inside transient_injection).
                #
                # Net effect: on a multi-turn request, the system prefix is
                # byte-identical across turns and the upstream inference
                # server gets free prefix-cache hits.
                # ============================================================
                stable_skill_instruction = "\n\nYou have Natural-born tools and Acquired Skills. If you lack a tool for a complex repetitive task, use `create_skill` to program it permanently. It will be available on your next turn.\n"
                stable_system_prompt = (base_prompt + stable_skill_instruction).replace("\r", "")
                for m in messages:
                    if m.get("role") == "system":
                        m["content"] = stable_system_prompt
                        break

                messages = self.process_rolling_window(messages, self.context.args.max_context)

                final_ai_content, created_time = "", int(datetime.datetime.now().timestamp())
                force_stop, seen_tools, tool_usage, last_was_failure = False, set(), {}, False
                # Per-request idempotency ledger. Tools that are pure setters
                # (update_profile, learn_skill, knowledge_base.insert_fact /
                # forget) get refused on repeat-with-identical-args within the
                # same request. Without this guard, the model can lose track
                # of "I already did X" and loop the same call 9 turns in a
                # row — observed in production logs.
                executed_idempotent: set = set()
                # Request-scoped sandbox state cache (replaces the leaky
                # process-global `context.cached_sandbox_state`).
                request_sandbox_state = None
                raw_tools_called = set()
                execution_failure_count = 0
                transient_failure_count = 0   # Separate budget for retryable errors
                # Counts CONSECUTIVE `system_parse_error` events across turns
                # within this request. Reset on any successful parse. After
                # threshold (≥2) we pivot the recovery prompt to suggest
                # alternative tool-call shapes — the default "use XML"
                # message is provably not breaking the model out of the loop
                # (see selfplay session, attempt 2: 5 identical failures).
                consecutive_parse_errors = 0
                tools_run_this_turn = []
                forget_was_called = False
                thought_content = ""
                was_complex_task = False

                task_tree = TaskTree()
                current_plan_json = {}
                _request_sys3_fired_once = False
                _request_sys3_prev_justification = ""
                force_final_response = False

                # NB: the `[EPHEMERAL_TERMINAL_DIRECTIVE]` sweep that used
                # to run here was retired alongside the directive itself —
                # terminal tools now populate `final_ai_content` directly
                # from the tool result and exit the turn loop, so no
                # ephemeral messages are ever appended to the history in
                # the first place.

                # Counts thinking-cap / runaway-thinking events across
                # the turn loop. First event injects a SYSTEM ALERT and
                # retries; second event in the same attempt force-stops
                # the whole request. Without this, a reasoning-model
                # paralysis can burn through the full strike budget by
                # hitting the cap every second turn (see 09:07 log:
                # Turn 6 and Turn 9 both capped at 12 000 chars, Turn 10
                # resumed the same spiral).
                thinking_cap_events = 0

                # Cross-turn paragraph-repetition state. Each turn's
                # opening thought (first 200 chars of <think>) is
                # fingerprinted into a word set; Jaccard similarity
                # against the prior turn detects when the model is
                # re-entering the same derivation loop across turns,
                # not just within a single stream. The existing
                # `_detect_thinking_loop` only sees one stream at a
                # time, so it missed the 09:07 failure where the
                # solver restarted the same impossibility proof on
                # every new turn.
                prev_turn_opening_words: set = set()
                cross_turn_repeat_hits = 0

                # Self-play can cap a single attempt's turn count via
                # `max_turns_override` on the GhostAgent instance, so a
                # runaway simulation can't silently chew through 40 turns.
                effective_max_turns = getattr(self, "max_turns_override", None) or 40
                for turn in range(effective_max_turns):
                    self.context.last_activity_time = datetime.datetime.now() # Heartbeat

                    # Differentiated strike budgets: structural failures
                    # (logic errors, assertion failures) get 6 strikes, while
                    # transient failures (timeouts, connection errors) get a
                    # separate budget of 4 strikes (they auto-resolve). The
                    # combined cap is 8 total failures of any kind.
                    total_failures = execution_failure_count + transient_failure_count
                    if execution_failure_count >= 6 or total_failures >= 8:
                        pretty_log("Loop Breaker", f"Strike cap hit (structural={execution_failure_count}, transient={transient_failure_count}). Aborting turn loop.", icon=Icons.STOP)
                        messages.append({"role": "user", "content": "SYSTEM ALERT: You have failed too many times. The task cannot be completed."})
                        if not final_ai_content:
                            final_ai_content = "I hit a hard limit after repeated failures and could not complete this task. Please rephrase or break it into smaller steps."
                        break

                    turn_is_conversational = is_conversational and turn == 0

                    if turn > 2: was_complex_task = True
                    if force_stop: break

                    # --- CHECKPOINT & RESUME ---
                    # At turn milestones, save a structured progress summary
                    # to the scratchpad so earlier context can be pruned
                    # without losing key progress state.
                    if turn in (15, 30) and getattr(self.context, 'scratchpad', None):
                        try:
                            checkpoint_items = []
                            for m in messages[-10:]:
                                role = m.get("role", "?")
                                content = str(m.get("content", ""))[:200]
                                if role in ("assistant", "tool") and content.strip():
                                    checkpoint_items.append(f"{role}: {content}")
                            if checkpoint_items:
                                checkpoint_summary = f"[Turn {turn} checkpoint] " + " | ".join(checkpoint_items[-5:])
                                self.context.scratchpad.set(
                                    f"_checkpoint_t{turn}",
                                    checkpoint_summary[:1000]
                                )
                        except Exception:
                            pass

                    scratch_data = self.context.scratchpad.list_all() if getattr(self.context, 'scratchpad', None) else "None."
                    if has_coding_intent:
                        # Per-request sandbox state cache. The previous
                        # process-global cache (`context.cached_sandbox_state`)
                        # leaked across concurrent requests and could serve
                        # one request's stale view to another. Local var =
                        # request-scoped lifetime by construction.
                        if 'request_sandbox_state' not in locals() or request_sandbox_state is None:
                            from ..tools.file_system import tool_list_files
                            params = {
                                "sandbox_dir": self.context.sandbox_dir,
                                "memory_system": self.context.memory_system
                            }
                            request_sandbox_state = await tool_list_files(**params)
                        sandbox_state = request_sandbox_state
                    else:
                        sandbox_state = "N/A"

                    # Use System 2 Planner based on context arguments (Mock-safe check)
                    use_plan = getattr(self.context.args, 'use_planning', False) == True
                    # Router-gated reasoning depth (proposal item #10):
                    # when the complexity router is CONFIDENT the request
                    # is easy (and never had to escalate), skip the
                    # strategic planner. An easy request does not need a
                    # multi-task decomposition, and the planner LLM
                    # round-trip is the single most expensive step of the
                    # turn — this is what makes the router decision
                    # genuinely load-bearing on cost, not just advisory.
                    if use_plan:
                        _rd_plan = body.get("_router_decision") or {}
                        if (_rd_plan.get("label") == "easy"
                                and not _rd_plan.get("escalated")
                                and float(_rd_plan.get("confidence") or 0.0) >= 0.75):
                            use_plan = False
                            pretty_log(
                                "Reasoning Loop",
                                "Router: confident-easy request — skipping "
                                "strategic planner",
                                icon=Icons.BRAIN_ROUTE,
                            )
                    if use_plan and not turn_is_conversational:
                        pretty_log("Reasoning Loop", f"Turn {turn+1} Strategic Analysis...", icon=Icons.BRAIN_PLAN)

                        last_tool_output = self._prepare_planning_context(tools_run_this_turn[-2:])
                        recent_transcript = self._get_recent_transcript(messages)

                        tool_hints = {
                            "system_utility": "weather, health",
                            "execute": "python, bash",
                            "postgres_admin": "sql"
                        }
                        available_tools_list = ", ".join([
                            f"{t['function']['name']} ({tool_hints.get(t['function']['name'], 'native tool')})"
                            for t in get_active_tool_definitions(self.context)
                        ])
                        state_limit = max(1500, int(char_budget * 0.05))
                        safe_scratch = str(scratch_data)
                        if len(safe_scratch) > state_limit: safe_scratch = safe_scratch[:state_limit] + "\n...[TRUNCATED]"
                        safe_sandbox = str(sandbox_state)
                        if len(safe_sandbox) > state_limit: safe_sandbox = safe_sandbox[:state_limit] + "\n...[TRUNCATED]"

                        # Pre-fetch the skill playbook for the planner so
                        # it can reshape the PLAN (not just the per-
                        # action tool call) around prior lessons. The
                        # execution step below also calls
                        # `request_state.get_skill_playbook` with the
                        # same query — both hits are served from the
                        # per-query cache so we pay the cost once.
                        planner_playbook = ""
                        if self.context.skill_memory:
                            try:
                                planner_playbook = await request_state.get_skill_playbook(
                                    last_user_content or ""
                                )
                            except Exception:
                                planner_playbook = ""
                        planner_playbook_block = (
                            f"\n### RELEVANT PRIOR LESSONS\n{planner_playbook}\n"
                            if planner_playbook else ""
                        )

                        planner_transient = f"""
### CURRENT SITUATION
SCRAPBOOK:
{safe_scratch}
SANDBOX STATE:
{safe_sandbox if has_coding_intent else 'N/A'}

User Request: {last_user_content}
Last Tool Output: {last_tool_output}
{planner_playbook_block}
### AVAILABLE NATIVE TOOLS
[{available_tools_list}]
CRITICAL INSTRUCTION: If an action requires a tool, explicitly name the native JSON tool you intend to use. DO NOT plan to write Python scripts for tasks that have a dedicated native tool. If the user is just asking a question or requesting a code/SQL explanation, set "next_action_id" to "none" and do NOT plan to use a tool.

### TEMPORAL ANCHOR (READ CAREFULLY)
You are currently at TURN {turn+1}. Trust your CURRENT PLAN JSON to know what is already DONE. NEVER revert a 'DONE' task back to 'PENDING'.

### CURRENT PLAN (JSON)
{json.dumps(current_plan_json, indent=2) if current_plan_json else "No plan yet."}
"""
                        planner_messages = [
                            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                            {"role": "user", "content": f"### RECENT CONVERSATION:\n{recent_transcript}\n\n{planner_transient.strip()}"}
                        ]

                        planning_payload = {
                            "model": model,
                            "messages": planner_messages,
                            "temperature": 0.0,
                            "top_p": 0.1,
                            "max_tokens": 4096,
                            "response_format": {"type": "json_object"}
                        }

                        try:
                            p_data = await self.context.llm_client.chat_completion(planning_payload, use_swarm=True)
                            plan_content = p_data["choices"][0]["message"].get("content", "")
                            plan_json = extract_json_from_text(plan_content)

                            thought_content = plan_json.get("thought", "No thought provided.")
                            tree_update = plan_json.get("tree_update", {})
                            next_action_id = plan_json.get("next_action_id", "")
                            required_tool = plan_json.get("required_tool", "all")

                            if tree_update:
                                task_tree.load_from_json(tree_update)
                                current_plan_json = task_tree.to_json()

                            tree_render = task_tree.render()

                            # Planning content is no longer injected into history messages

                            pretty_log("INTERNAL MONOLOGUE", icon=Icons.BRAIN_THINK, special_marker="SECTION_START")
                            pretty_log("Planner Monologue", thought_content, icon=Icons.BRAIN_PLAN)
                            pretty_log("INTERNAL MONOLOGUE", icon=Icons.BRAIN_THINK, special_marker="SECTION_END")
                            pretty_log("Reasoning Loop", f"Plan Updated. Focus: {next_action_id}", icon=Icons.OK)

                            if task_tree.root_id and task_tree.nodes[task_tree.root_id].status == TaskStatus.DONE and turn > 0:
                                pretty_log("Finalizing", "Agent signaled completion", icon=Icons.OK)
                                force_stop = True
                        except Exception as e:
                            logger.error(f"Planning step failed: {e}")
                            if not any("### ACTIVE STRATEGY" in m.get("content", "") for m in messages):
                                messages.append({"role": "user", "content": "### ACTIVE STRATEGY: Proceed directly to using a tool. Do NOT provide any conversational response this turn, only output a tool_calls array!"})

                    # Dynamic state no longer mutated via re.sub

                    # Proactive Context Pruning before request
                    messages = await self._prune_context(messages, max_tokens=self.context.args.max_context, model=model)

                    # Dynamic Context Cache Tool Injection (Context Bloat Fix)
                    # ARCHITECTURAL OPTIMISATION #4 + #7: cached lookups via
                    # the request-scoped state. Repeated turns with the same
                    # search_query and tool list now hit the LRU + per-request
                    # caches instead of re-filtering and re-serialising.
                    search_query = thought_content if (use_plan and locals().get('thought_content')) else last_user_content
                    all_tools = request_state.get_active_tool_defs(search_query or "")
                    if hasattr(self, 'disabled_tools') and self.disabled_tools:
                        all_tools = [t for t in all_tools if t["function"]["name"] not in self.disabled_tools]

                    from .prompts import QWEN_TOOL_PROMPT

                    # ARCHITECTURAL OPTIMISATION #1: NO MORE per-turn system
                    # slot mutation. The system message was already locked at
                    # request start to `stable_system_prompt`. Tool schemas
                    # now ride in the user-message header (built below as
                    # `tool_header_block`) where the dynamic state already
                    # lives, so the system prefix stays byte-stable across
                    # all turns of this request.
                    # Per-task thinking budget: pick the <think> guidance
                    # string based on the user's query intent so debugging /
                    # algorithm / SQL-optimization tasks get room to derive
                    # step-by-step while chit-chat still fires the tight
                    # anti-paralysis cap.
                    # Explicit per-agent override wins over the keyword
                    # classifier. Self-play sets this to "selfplay" on
                    # `temp_agent` so synthetic exercises skip the
                    # EXTENDED tier that every coding+keyword challenge
                    # otherwise lands in — EXTENDED gives the solver
                    # license to draft full Python inside <think> and
                    # recompute outputs by hand, both of which waste
                    # minutes on a bounded simulation.
                    budget_override = getattr(self, "thinking_budget_override", None)
                    if budget_override in {"tight", "extended", "selfplay"}:
                        think_budget = budget_override
                    else:
                        think_budget = classify_thinking_budget(
                            last_user_content,
                            has_coding_intent=has_coding_intent,
                            is_meta_task=is_meta_task,
                            in_active_project=bool(
                                getattr(self.context, "current_project_id", None)
                            ),
                        )

                    # =====================================================
                    # CONTEXT-COMPACTION OPTIMISATIONS #1 + #2
                    # -----------------------------------------------------
                    # Decide whether to ship the XML tool schema this turn
                    # BEFORE serialising it, so we never pay the ~7.4K-
                    # token cost when the model can't / won't tool-call.
                    #
                    # #1 — Skip schema on final-generation turns. When the
                    # planner has set `force_final_response=True` or the
                    # required_tool is "none", the model is being asked
                    # to answer in plain text and tool_calls are dropped
                    # downstream anyway (see the force_final_response /
                    # is_final_generation guard around `tool_calls` in
                    # `apply_chat_outcome_heuristics` consumers). Shipping
                    # the schema in that case is wasted bytes and pollutes
                    # the model's attention with an option it cannot use.
                    #
                    # #2 — Don't double-ship under --native-tools. When
                    # native_tools is on, schemas are advertised through
                    # the OpenAI-style `payload["tools"]` channel below,
                    # so re-emitting the same definitions in the prompt
                    # XML is pure duplication. The XML format scaffolding
                    # (parsing rules, parallel-call guidance, CDATA hint)
                    # is preserved so the agent's XML parser still works
                    # as a fallback for models that emit the legacy shape.
                    # =====================================================
                    # Earliest-correct evaluation of `is_final_generation`.
                    # The canonical assignment further below stays for
                    # downstream consumers; this hoisted copy uses the
                    # same predicate so the two stay in sync. We also
                    # mirror the dynamic_state block's "next_action_id ==
                    # none → force_final_response" rule here so that the
                    # schema-skip kicks in even when the planner only
                    # signals via next_action_id (the dynamic_state line
                    # that sets force_final_response runs AFTER us).
                    _early_required_tool = locals().get("required_tool", "all")
                    _early_next_action_id = locals().get("next_action_id", "")
                    _is_final_generation_for_schema = (
                        force_final_response
                        or str(_early_required_tool).lower() == "none"
                        or (
                            use_plan and not turn_is_conversational
                            and bool(locals().get("thought_content"))
                            and str(_early_next_action_id).strip().lower() == "none"
                        )
                    )
                    _native_tools_active = bool(
                        getattr(self.context.args, "native_tools", False)
                    )

                    if _is_final_generation_for_schema:
                        # Slim header: drop the entire tool block. The
                        # model is being asked to answer the user, not
                        # to call a tool. Keep the think-budget guidance
                        # so reasoning depth stays controlled.
                        tool_header_block = (
                            f"# Final-generation turn\n\n"
                            f"You are answering the user directly this turn. "
                            f"DO NOT emit any <tool_call> blocks. Reply in "
                            f"plain prose only.\n\n"
                            f"ADAPT YOUR THINKING DEPTH TO THE TASK:\n"
                            f"{render_think_budget_guidance(think_budget)}\n"
                        ).replace("\r", "")
                        minified_schemas = ""  # for downstream visibility
                    elif _native_tools_active:
                        # Tool definitions arrive via the native API
                        # channel below; substitute a compact pointer
                        # for `{tool_schemas}` so the prompt scaffolding
                        # remains intact without restating ~7K tokens.
                        # Keep one-line tool-name list so the model can
                        # see what's available even if its native-tool
                        # fence is filtered downstream.
                        _tool_names = ", ".join(
                            t["function"]["name"] for t in all_tools
                        ) or "(none)"
                        _native_pointer = (
                            f"(Tool schemas are advertised via the native "
                            f"`tool_calls` API on this request. Available "
                            f"tools: {_tool_names}.)"
                        )
                        tool_header_block = (
                            QWEN_TOOL_PROMPT
                            .replace('{tool_schemas}', _native_pointer)
                            .replace('{think_budget_guidance}', render_think_budget_guidance(think_budget))
                            .replace("\r", "")
                        )
                        minified_schemas = _native_pointer  # for diagnostics
                    else:
                        # Legacy XML-only path: full schema in the prompt.
                        minified_schemas = request_state.get_xml_schema(all_tools)
                        tool_header_block = (
                            QWEN_TOOL_PROMPT
                            .replace('{tool_schemas}', minified_schemas)
                            .replace('{think_budget_guidance}', render_think_budget_guidance(think_budget))
                            .replace("\r", "")
                        )
                    # --- INTENT-DRIVEN SKILL RECALL ---
                    # ARCHITECTURAL OPTIMISATION #4: the playbook lookup is
                    # cached per (skill_query) inside `request_state`, so a
                    # multi-turn request with a stable query pays the cost
                    # exactly once.
                    fetched_playbook = ""
                    if self.context.skill_memory:
                        skill_query = last_user_content
                        if use_plan and not turn_is_conversational and locals().get("required_tool", "none") not in ["none", "all"]:
                            skill_query = f"Tool: {required_tool} - Context: {thought_content}"
                        playbook = await request_state.get_skill_playbook(skill_query or "")
                        if playbook:
                            fetched_playbook = f"### SKILL PLAYBOOK:\n{playbook}\n\n"

                    # Top-of-state CWD pin (coding turns only). The rule
                    # previously sat between SCRAPBOOK and SANDBOX STATE,
                    # buried ~400 chars into the block — the model still
                    # fell through to `cd /home/user` / `cd /sandbox`
                    # hallucinations. Moving it to the absolute top with
                    # loud formatting makes it the FIRST thing read every
                    # coding turn, after which CURRENT TIME / SCRAPBOOK /
                    # project briefing / SANDBOX STATE follow normally.
                    dynamic_state = ""
                    if has_coding_intent:
                        dynamic_state += (
                            "⚠️  SHELL CWD IS /workspace — DO NOT USE `cd` ANYWHERE  ⚠️\n"
                            "The shell session ALREADY starts in /workspace. "
                            "Paths like /sandbox, /home/user, /root, /app, /tmp "
                            "do NOT exist in this container and `cd` to any of "
                            "them will fail with 'No such file or directory' "
                            "and BURN A STRIKE. Run commands directly with "
                            "relative paths:\n"
                            "  ✓  python3 test_foo.py\n"
                            "  ✓  python -m unittest tests.test_parser\n"
                            "  ✓  ls subdir/ ; cat a/b.py\n"
                            "  ✗  cd /home/user && python3 …   (FAILS)\n"
                            "  ✗  cd /sandbox && …              (FAILS)\n\n"
                        )
                    dynamic_state += f"### DYNAMIC SYSTEM STATE\nCURRENT TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Day: {datetime.datetime.now().strftime('%A')})\n\nSCRAPBOOK:\n{scratch_data}\n\n"
                    try:
                        from .prompts import build_project_briefing
                        _proj_store = getattr(self.context, "project_store", None)
                        _proj_id = getattr(self.context, "current_project_id", None)
                        if _proj_store is not None and _proj_id:
                            _briefing = build_project_briefing(_proj_store, _proj_id)
                            if _briefing:
                                dynamic_state += _briefing + "\n"
                    except Exception:
                        logger.debug("project briefing skipped", exc_info=True)
                    if has_coding_intent:
                        dynamic_state += f"CURRENT SANDBOX STATE:\n{sandbox_state}\n\n"
                    if use_plan and not turn_is_conversational and 'thought_content' in locals() and thought_content:
                        dynamic_state += f"ACTIVE STRATEGY & PLAN:\nTHOUGHT: {thought_content}\nPLAN:\n{task_tree.render()}\nFOCUS TASK: {next_action_id}\n"

                        if str(next_action_id).strip().lower() == "none":
                            dynamic_state += "CRITICAL INSTRUCTION: DO NOT USE TOOLS this turn. Answer the user directly using insights from your THOUGHT.\n"
                            force_final_response = True
                        else:
                            dynamic_state += "CRITICAL INSTRUCTION: Execute the tool(s) required for the FOCUS TASK. You MAY emit MULTIPLE <tool_call> blocks in parallel within this turn when they all serve the same FOCUS TASK (e.g. writing several project files, batching knowledge_base inserts). DO NOT HALLUCINATE TOOL OUTPUTS.\n"

                    # -----------------------------------------------------------------
                    # QWEN-AGENT METHODOLOGY: Bypass Native Tools & Use String Prompts
                    # -----------------------------------------------------------------
                    target_tool = locals().get("required_tool", "all")
                    # With the planner disabled, default to final generation (stream directly).
                    # If the model returns tool_calls, the turn loop below will handle them.
                    is_final_generation = force_final_response or target_tool.lower() == "none"

                    # Translate messages to bypass strict API validation and emulate Qwen-Agent
                    req_messages = []
                    for m in messages:
                        if m.get("role") == "tool":
                            # Translate tool results to a user message wrapped in <tool_response>
                            req_messages.append({
                                "role": "user",
                                "content": f"<tool_response name=\"{m.get('name', 'unknown')}\">\n{m.get('content')}\n</tool_response>"
                            })
                        elif m.get("role") == "assistant":
                            req_messages.append({
                                "role": "assistant",
                                "content": _render_assistant_with_tool_calls(
                                    m.get("content"),
                                    m.get("tool_calls") or [],
                                ),
                            })
                        elif m.get("role") == "user":
                            content_val = m.get("content", "")
                            has_vision_node = bool(getattr(self.context.llm_client, 'vision_clients', None))
                            if isinstance(content_val, list):
                                if has_vision_node:
                                    text_parts = []
                                    for item in content_val:
                                        if isinstance(item, dict):
                                            if item.get("type") == "text":
                                                text_parts.append(item.get("text", ""))
                                            elif item.get("type") == "image_url":
                                                try:
                                                    import base64 as __base64
                                                    img_url = item.get("image_url", {}).get("url", "")
                                                    if img_url.startswith("data:"):
                                                        header, encoded = img_url.split(",", 1)
                                                        img_data = __base64.b64decode(encoded)
                                                        filename = f"vision_{uuid.uuid4().hex[:8]}.jpg"
                                                        tmp_path = self.context.sandbox_dir / filename
                                                        with open(tmp_path, "wb") as f:
                                                            f.write(img_data)
                                                        text_parts.append(f"[Image attached. SAVED LOCALLY to '{filename}'. You MUST use the `vision_analysis` tool with target='{filename}' to see it!]")
                                                    else:
                                                        text_parts.append(f"[Image attached at URL: {img_url}. You MUST use the `vision_analysis` tool with target='{img_url}' to see it!]")
                                                except Exception as e:
                                                    text_parts.append(f"[Image attached but failed to parse: {e}]")
                                    content_val = "\n".join(text_parts)
                                    req_messages.append({"role": "user", "content": content_val})
                                else:
                                    # PRESERVE NATIVE VISION: Let Qwen 3.5 process the image locally
                                    req_messages.append({"role": "user", "content": content_val})
                            else:
                                req_messages.append({"role": "user", "content": content_val})
                        else:
                            req_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

                    # Bundle ALL dynamic context
                    # ARCHITECTURAL OPTIMISATION #1: tool schemas now ride
                    # at the head of the transient injection (was: in the
                    # system slot). The system slot itself stays byte-stable
                    # across turns to maximise upstream KV-cache hits.
                    transient_injection = f"{tool_header_block}\n\n{active_persona}{fetched_playbook}{fetched_context}{dynamic_state.strip()}"

                    # Prepend transient state to the LAST message to preserve KV Cache and prevent burying user input
                    if req_messages and req_messages[-1]["role"] == "user":
                        original_msg = req_messages[-1]["content"]
                        req_messages[-1]["content"] = f"<system_state_update>\n{transient_injection}\n(CRITICAL: This is internal system state. Do NOT acknowledge or comment on this block in your thoughts. Focus entirely on the user instruction.)\n</system_state_update>\n\n[USER INSTRUCTION]\n{original_msg}"
                    else:
                        req_messages.append({"role": "user", "content": f"<system_state_update>\n{transient_injection}\n</system_state_update>"})

                    # Precise sampling for any tool-using turn; warm/creative
                    # sampling only for conversational turns (greetings,
                    # chit-chat). The coding sub-profile (creative/precise/
                    # balanced) still applies when the task IS coding — that
                    # routing lives inside `get_sampling_params`.
                    is_tool_turn = not turn_is_conversational
                    sampling_params = get_sampling_params(
                        is_tool_turn,
                        query=last_user_content if 'last_user_content' in locals() else "",
                        is_coding=has_coding_intent and not is_meta_task,
                    )
                    # Explicit max_tokens guard. Without this the payload
                    # relies on the upstream's default cap (often 1024 or
                    # 2048), which silently truncates the assistant's
                    # `<tool_call>` when thought+content grow past the
                    # cap — producing system_parse_error storms (the
                    # model emits 6 nearly-identical broken calls in a row
                    # because it can't tell why the prior attempt failed).
                    # Defensive int-cast: in tests the args namespace can
                    # be a MagicMock where getattr returns a Mock (truthy);
                    # fall back to the default rather than raising.
                    _cfg_max = getattr(self.context.args, 'tool_turn_max_tokens', 0)
                    try:
                        tool_turn_max_tokens = int(_cfg_max) if _cfg_max else DEFAULT_TOOL_TURN_MAX_TOKENS
                    except (TypeError, ValueError):
                        tool_turn_max_tokens = DEFAULT_TOOL_TURN_MAX_TOKENS
                    payload = {
                        "model": model,
                        "messages": req_messages,
                        "stream": False,
                        **sampling_params,
                        "frequency_penalty": 0.00,
                        "max_tokens": tool_turn_max_tokens,
                        # Stop sequences removed for Qwen 3.6 35B-A3. The
                        # QWEN_TOOL_PROMPT explicitly invites parallel
                        # execution ("PARALLEL EXECUTION: You may execute
                        # MULTIPLE tools in a single turn"), so the old
                        # `</tool_call>\n<tool_call>` stop would silently
                        # truncate legitimate second calls. The decoder-
                        # collapse shape `<tool_call>\n<tool_call>` is still
                        # covered by `_detect_tool_call_loop` (streaming
                        # detector, aborts after ~10 unclosed openings) and
                        # `TOOL_CALL_LOOP_THRESHOLD`, so removing the stop
                        # doesn't regress protection against the pathological
                        # 8135-opening trace.
                    }
                    # XML tool-call parsing is the primary path. When
                    # ``--native-tools`` is on AND the downstream model
                    # advertises OpenAI-style tool_calls support, we
                    # attach the native schema so the server can surface
                    # tool_calls via `message.tool_calls`. The XML
                    # schema in the prompt is suppressed in that case
                    # (see CONTEXT-COMPACTION OPTIMISATION #2 above) to
                    # avoid double-shipping ~7K tokens of definitions on
                    # every turn.
                    #
                    # On final-generation turns (force_final_response or
                    # required_tool=='none') we ALSO suppress the native
                    # schema: the model is being told to answer in plain
                    # text, and `force_final_response` already drops any
                    # tool_calls the model attempts. Sending tools on
                    # those turns is wasted bytes AND tempts the model
                    # to call something instead of answering.
                    if (
                        getattr(self.context.args, "native_tools", False)
                        and not is_final_generation
                    ):
                        try:
                            payload["tools"] = all_tools
                            payload["tool_choice"] = "auto"
                            # Qwen 3.6 + vLLM default to single-tool-per-reply
                            # when `tools` is attached, silently defeating the
                            # "emit multiple tool_calls in one turn" prompt
                            # guidance. Opt in explicitly. Servers that don't
                            # recognise the flag ignore it harmlessly.
                            payload["parallel_tool_calls"] = True
                        except Exception:
                            pass

                    pretty_log("LLM Request", f"Turn {turn+1} | Temp {sampling_params['temperature']:.2f}", icon=Icons.LLM_ASK)

                    if is_final_generation and stream_response:
                        payload["stream"] = True
                        # Capture outer variables to prevent NameError when finally block deletes them
                        stream_messages_snapshot = list(messages[-10:])
                        stream_tools_snapshot = list(tools_run_this_turn)
                        stream_thought = thought_content
                        stream_model = model

                        # NEW: Capture accumulated intermediate text (like image tags from previous turns)
                        stream_prefix = final_ai_content.strip() + "\n\n" if final_ai_content.strip() else ""

                        async def stream_wrapper():
                            full_content = ""
                            loop_detected = False

                            # NEW: Flush intermediate text to the UI as the first stream chunk
                            if stream_prefix:
                                start_chunk = {
                                    "id": f"chatcmpl-{req_id}", "object": "chat.completion.chunk", "created": created_time,
                                    "model": stream_model, "choices": [{"index": 0, "delta": {"content": stream_prefix}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(start_chunk)}\n\n".encode('utf-8')
                                full_content += stream_prefix

                            # On a final-generation turn (force_final_response
                            # or target_tool='none') the model is expected to
                            # emit plain TEXT — any <tool_call>, <function>
                            # or <tool_response> in the stream is a bug the
                            # user should never see. The non-streaming path
                            # already strips these (the widened end-of-
                            # handle_chat scrub), but the streaming `yield
                            # chunk` below ships raw upstream bytes straight
                            # to the client and nothing downstream can scrub
                            # them. Trace: the user reported seeing the
                            # literal `<tool_call><function name="self_play">
                            # </function></tool_call>` as the assistant's
                            # reply on a second `run self play` — that was
                            # this exact path. Scrubbed streaming mirrors the
                            # non-streaming scrub: accumulate full_content,
                            # compute the scrubbed view, and emit only the
                            # new SCRUBBED portion as a synthetic SSE chunk.
                            # Non-final-generation streams (still in a tool
                            # loop) keep the old raw-yield behaviour because
                            # legitimate tool_call XML there has to reach the
                            # parser downstream.
                            _stream_scrub_active = bool(is_final_generation)
                            # When the scrub is active we must also buffer
                            # upstream's `data: [DONE]\n\n` sentinel until
                            # AFTER the post-loop fallback check. SSE
                            # clients close the connection the moment they
                            # see [DONE], so emitting [DONE] first and the
                            # fallback second means the user never sees
                            # the fallback. Observed trace: the
                            # `Scrub consumed entire response` pretty_log
                            # fired but the client still rendered empty
                            # because [DONE] had already reached it. Hold
                            # [DONE] and emit it once at the very end.
                            _held_done_chunk = None
                            # Use `\Z` (absolute end of string) instead of
                            # `$` as the missing-close-tag alternative. In
                            # Python's non-MULTILINE mode, `$` matches BOTH
                            # at end-of-string AND just before a trailing
                            # newline — so `<tool_call>\n` matches as just
                            # `<tool_call>`, and the `\n` escapes the scrub.
                            # That's exactly the bug that left the user
                            # seeing an empty reply on a second `self play`:
                            # the model's entire response was
                            # `<tool_call>...\n</tool_call>`, the scrub
                            # dropped everything except the trailing `\n`,
                            # 1 char of whitespace reached the client, and
                            # the empty-output fallback didn't fire because
                            # `_scrubbed_emitted_len=1 > len(stream_prefix)=0`.
                            # `\Z` closes that loophole without affecting any
                            # closed-tag match.
                            _stream_scrub_pattern = re.compile(
                                r'<(tool_call|tool|function|tool_response)\b[^>]*>.*?'
                                r'(?:</\1\b[^>]*>|\Z)',
                                flags=re.DOTALL | re.IGNORECASE,
                            )
                            _scrubbed_emitted_len = len(full_content) if _stream_scrub_active else 0

                            async for chunk in self.context.llm_client.stream_chat_completion(payload, use_coding=has_coding_intent):
                                if loop_detected: break
                                self.context.last_activity_time = datetime.datetime.now() # Heartbeat

                                # Decode FIRST so we can decide whether to
                                # yield the raw chunk or substitute a
                                # scrubbed synthetic chunk. Non-data chunks
                                # (e.g. `data: [DONE]`) always pass through
                                # unchanged.
                                _is_content_chunk = False
                                _new_text = ""
                                try:
                                    chunk_str = chunk.decode("utf-8")
                                    if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                                        chunk_data = json.loads(chunk_str[6:])
                                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                            delta = chunk_data["choices"][0].get("delta", {})
                                            if "content" in delta and delta["content"] is not None:
                                                _is_content_chunk = True
                                                _new_text = delta["content"]
                                                full_content += _new_text
                                except Exception as e:
                                    logger.debug(f"Stream chunk decode error: {type(e).__name__}")

                                if _stream_scrub_active and _is_content_chunk:
                                    # Emit only the NEW portion of the
                                    # scrubbed view. If the model is
                                    # currently mid-<tool_call>, the
                                    # non-greedy pattern won't match until
                                    # the close tag arrives — so `scrubbed`
                                    # stops growing during a tool_call body
                                    # and resumes once the close lands,
                                    # with the whole block removed. Net
                                    # effect: the client sees clean text
                                    # and never sees the XML flicker past.
                                    _scrubbed_view = _stream_scrub_pattern.sub('', full_content)
                                    _to_emit = _scrubbed_view[_scrubbed_emitted_len:]
                                    if _to_emit:
                                        _synthetic = {
                                            "id": f"chatcmpl-{req_id}",
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": stream_model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": _to_emit},
                                                "finish_reason": None,
                                            }],
                                        }
                                        yield f"data: {json.dumps(_synthetic)}\n\n".encode('utf-8')
                                        _scrubbed_emitted_len = len(_scrubbed_view)
                                    # Skip the raw chunk — we already
                                    # emitted the scrubbed equivalent.
                                elif _stream_scrub_active and not _is_content_chunk:
                                    # Non-content chunk on the scrub path.
                                    # The [DONE] sentinel must be held back
                                    # so the fallback (emitted AFTER this
                                    # loop) reaches the client before the
                                    # SSE stream closes. Every other non-
                                    # content chunk passes through.
                                    try:
                                        _chunk_str = chunk.decode("utf-8")
                                    except Exception:
                                        _chunk_str = ""
                                    if _chunk_str.strip() == "data: [DONE]":
                                        _held_done_chunk = chunk
                                    else:
                                        yield chunk
                                else:
                                    # Non-scrub path (mid-tool-loop stream,
                                    # or non-content chunk): pass raw upstream
                                    # bytes through as before.
                                    yield chunk

                                # Cognitive Watchdog — tightened to reduce false positives
                                # on legitimate repetitive output (lists, tables, error
                                # traces). Now requires a longer window (400 chars) and
                                # a longer repeating motif (60 chars) seen 5+ times before
                                # severing the stream.
                                if _is_content_chunk and ("\n" in _new_text or "?" in _new_text):
                                    tail = full_content[-400:]
                                    if len(tail) == 400:
                                        last_60 = tail[-60:]
                                        if last_60.strip() and tail.count(last_60) >= 5:
                                            pretty_log("Cognitive Watchdog", "Infinite <think> loop detected. Severing stream.", level="WARNING", icon=Icons.STOP)
                                            loop_detected = True
                                            break_text = "\n</think>\n<tool_call>\n<function name=\"replan\">\n<parameter name=\"reason\">SYSTEM OVERRIDE: My internal monologue got stuck in an infinite loop. I am forcing a strategy reset.</parameter>\n</function>\n</tool_call>"
                                            break_chunk = {
                                                "id": f"chatcmpl-{req_id}", "object": "chat.completion.chunk", "created": created_time,
                                                "model": stream_model, "choices": [{"index": 0, "delta": {"content": break_text}, "finish_reason": None}]
                                            }
                                            yield f"data: {json.dumps(break_chunk)}\n\n".encode('utf-8')
                                            full_content += break_text

                            # --- SCRUBBED-STREAM EMPTY-OUTPUT FALLBACK ---
                            # If the scrub was active and swallowed EVERY
                            # content chunk (i.e. the whole upstream response
                            # was <tool_call>/<function> XML), the client has
                            # received zero visible text beyond the prefix.
                            # Before this fallback, the user saw a blank reply
                            # — which is what happened on the user's third
                            # `self play` invocation: the planner routed it as
                            # text-only (target_tool='none'), the model emitted
                            # a pure tool_call, the scrub stripped it, and the
                            # SSE stream closed with nothing in it. Emit a
                            # synthetic fallback so the user sees SOMETHING
                            # actionable instead of an empty bubble.
                            if (
                                _stream_scrub_active
                                and full_content.strip()
                                and len(full_content.strip()) > len(stream_prefix.strip())
                                and _scrubbed_emitted_len <= len(stream_prefix)
                            ):
                                _intended = ""
                                _fn_m = re.search(
                                    r'<function(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_]+)',
                                    full_content,
                                    re.IGNORECASE,
                                )
                                if _fn_m:
                                    _intended = _fn_m.group(1)

                                # The "cycle already completed" branch that
                                # used to live here was retired once terminal
                                # tools (self_play, dream_mode) started
                                # bypassing the summary LLM turn entirely —
                                # see the DIRECT-FROM-TOOL SUMMARY block in
                                # the tool-execution path. That path now
                                # writes directly to `final_ai_content` and
                                # flips `force_stop`, so we never reach this
                                # stream-scrub fallback when a terminal tool
                                # ran in the same handle_chat. This
                                # single-branch fallback handles only the
                                # remaining case: the planner routed the
                                # turn as text-only but the model still
                                # emitted a tool_call.
                                _fallback_text = (
                                    "I prepared a tool call but this turn was routed as "
                                    "text-only, so it wasn't executed."
                                    + (f" (Intended tool: `{_intended}`.)" if _intended else "")
                                    + " Please rephrase your request — for example, "
                                    "`run {_intended}` — or try again.".replace("{_intended}", _intended or "the command")
                                )
                                _fallback_chunk = {
                                    "id": f"chatcmpl-{req_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": stream_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": _fallback_text},
                                        "finish_reason": None,
                                    }],
                                }
                                yield f"data: {json.dumps(_fallback_chunk)}\n\n".encode('utf-8')
                                pretty_log(
                                    "Stream Scrub Fallback",
                                    f"Scrub consumed entire response (intended={_intended or 'unknown'}); emitted fallback.",
                                    level="WARNING", icon=Icons.WARN,
                                )

                            # Release any held [DONE] sentinel now that
                            # (a) all content chunks have been emitted and
                            # (b) the fallback has had its chance to fire.
                            # Without this, the SSE stream would never
                            # terminate cleanly on the scrub path when no
                            # [DONE] is yielded elsewhere.
                            if _held_done_chunk is not None:
                                yield _held_done_chunk

                            if self.context.args.smart_memory > 0.0 and last_user_content and not forget_was_called and not last_was_failure:
                                micro_msgs = []
                                for m in [msg for msg in stream_messages_snapshot if msg.get("role") in ["user", "assistant"]][-4:]:
                                    role = m.get("role", "user").upper()
                                    clean_content = re.sub(r'```.*?```', '', str(m.get("content", "")), flags=re.DOTALL)
                                    micro_msgs.append(f"{role}: {clean_content[:500].strip()}")
                                clean_ai = re.sub(r'```.*?```', '', full_content, flags=re.DOTALL)
                                recent_arc = "\n".join(micro_msgs) + f"\nAI: {clean_ai[:500].strip()}"
                                if getattr(self.context, 'journal', None):

                                    await asyncio.to_thread(self.context.journal.append, 'smart_memory', {'text': recent_arc, 'model': stream_model})

                            # --- EXTRACT & LOG INTERNAL THINKING (STREAM) ---
                            think_matches = re.findall(r'<think>(.*?)(?:</think>|$)', full_content, flags=re.DOTALL | re.IGNORECASE)
                            for think_text in think_matches:
                                clean_think = think_text.strip()
                                if clean_think:
                                    ui_think = clean_think.replace('\n', ' | ')
                                    logger.info(f"PLANNER MONOLOGUE: {ui_think}")

                                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                                    print(f"[INFO ] 💭 {timestamp} - [{req_id}] {'='*15} AGENT INTERNAL THINKING {'='*15}", flush=True)
                                    for line in clean_think.split('\n'):
                                        if line.strip():
                                            print(f"[INFO ] 💭 {timestamp} - [{req_id}] {line.strip()}", flush=True)
                                    print(f"[INFO ] 💭 {timestamp} - [{req_id}] {'='*55}", flush=True)

                            if was_complex_task or execution_failure_count > 0:
                                if not force_stop or "READY TO FINALIZE" in stream_thought.upper():
                                    # Gated on `--smart-memory > 0.0` to honour
                                    # the contract in CLAUDE.md ("Memory writes
                                    # are gated on --smart-memory / --no-memory").
                                    # Without this, a `--smart-memory 0.0` run
                                    # still queued post_mortem entries → phase 1
                                    # consumer → SkillMemory.learn_lesson, leaking
                                    # auto-extracted lessons into the playbook on
                                    # every complex/failing turn.
                                    if getattr(self.context, 'journal', None) and self.context.args.smart_memory > 0.0:

                                        await asyncio.to_thread(self.context.journal.append, 'post_mortem', {'user': last_user_content, 'tools': stream_tools_snapshot, 'ai': full_content, 'model': stream_model})

                            # Retrieval feedback loop: credit lessons that
                            # were surfaced during this turn whenever the
                            # turn finished without an execution failure.
                            # `credit_recent_retrievals` is idempotent per
                            # retrieval (via `last_credited_at`) and only
                            # touches lessons whose `last_retrieved_at`
                            # falls inside the window, so calling it on
                            # turns that didn't surface any lesson is a
                            # no-op. The previous gate
                            # (`was_complex_task or execution_failure_count
                            # > 0`) excluded simple successful tool turns
                            # — exactly the population where lessons are
                            # most likely to be helping — so the feedback
                            # signal was biased toward complex tasks and
                            # the pruner drifted accordingly.
                            sm = getattr(self.context, 'skill_memory', None)
                            if sm is not None and execution_failure_count == 0:
                                try:
                                    if hasattr(sm, 'credit_recent_retrievals'):
                                        await asyncio.to_thread(sm.credit_recent_retrievals, 300)
                                except Exception:
                                    pass

                        return stream_wrapper(), created_time, req_id

                    # Ensure msg is always defined in this scope
                    msg = {"role": "assistant", "content": "", "tool_calls": []}
                    thinking_loop_detected = False
                    try:
                        payload["stream"] = True
                        full_content = ""
                        reasoning_content = ""

                        # Thinking metrics: surfaced as a single summary line
                        # after the stream completes. We no longer print empty
                        # `=== THINKING ===` frames; in verbose mode the live
                        # tokens still echo to stdout.
                        thinking_started = time.monotonic()
                        thinking_token_count = 0
                        thinking_line_buf = ""
                        next_loop_probe = THINKING_LOOP_PROBE_EVERY

                        # Flush-size budget for streaming thought blocks.
                        # Reasoning models (Qwen3+) emit thinking as many
                        # short newline-separated bullets; the old policy
                        # of "flush on every \n" produced 40+ log events
                        # per turn. Accumulate into ~paragraph-sized
                        # chunks and flush only on a paragraph break
                        # (blank line) or when the buffer crosses the
                        # size budget. The final `_flush_thinking` at
                        # stream end emits whatever remains.
                        _THINK_FLUSH_CHARS = 400

                        def _emit_thinking(text: str):
                            nonlocal thinking_line_buf, thinking_token_count
                            if not text:
                                return
                            thinking_token_count += 1
                            if not _glog.VERBOSE_MODE:
                                return  # Silent — summary line is emitted post-stream.
                            thinking_line_buf += text
                            while True:
                                # Prefer paragraph boundary (blank line)
                                # as a flush point — it maps to a
                                # natural thought break. Blocks are
                                # emitted with raw newlines preserved so
                                # multi-line reasoning stays readable in
                                # the log viewer; the prior " | " join
                                # made streamed bullets / code
                                # unreadable.
                                para_idx = thinking_line_buf.find("\n\n")
                                if para_idx >= 0:
                                    block = thinking_line_buf[:para_idx].strip()
                                    thinking_line_buf = thinking_line_buf[para_idx + 2:]
                                    if block:
                                        pretty_log("thinking", block, icon=Icons.BRAIN_THINK, level="DEBUG")
                                    continue
                                # No paragraph boundary yet — flush only
                                # when the buffer exceeds the budget,
                                # and cut at the last `\n` so we don't
                                # split mid-sentence.
                                if len(thinking_line_buf) >= _THINK_FLUSH_CHARS:
                                    last_nl = thinking_line_buf.rfind("\n")
                                    if last_nl <= 0:
                                        # No newline inside the buffer —
                                        # single runaway token stream.
                                        # Flush the whole thing; the next
                                        # chunk starts fresh.
                                        block = thinking_line_buf.strip()
                                        thinking_line_buf = ""
                                    else:
                                        block = thinking_line_buf[:last_nl].strip()
                                        thinking_line_buf = thinking_line_buf[last_nl + 1:]
                                    if block:
                                        pretty_log("thinking", block, icon=Icons.BRAIN_THINK, level="DEBUG")
                                    continue
                                break

                        def _flush_thinking():
                            nonlocal thinking_line_buf
                            if thinking_line_buf and _glog.VERBOSE_MODE:
                                if thinking_line_buf.strip():
                                    pretty_log("thinking", thinking_line_buf.strip(), icon=Icons.BRAIN_THINK, level="DEBUG")
                                thinking_line_buf = ""

                        stop_printing = False

                        async for chunk in self.context.llm_client.stream_chat_completion(payload, use_coding=has_coding_intent):
                            self.context.last_activity_time = datetime.datetime.now() # Heartbeat to prevent Hippocampus from waking up
                            try:
                                chunk_str = chunk.decode("utf-8")
                                if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                                    chunk_data = json.loads(chunk_str[6:])
                                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                        delta = chunk_data["choices"][0].get("delta", {})

                                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                                            r_token = delta["reasoning_content"]
                                            reasoning_content += r_token
                                            if not stop_printing:
                                                if "</think" in reasoning_content.lower() or "<tool_call" in reasoning_content.lower():
                                                    stop_printing = True
                                                if not stop_printing:
                                                    if r_token.strip().lower() in ["<", "</", "think", ">", "think>", "<think"]:
                                                        pass  # Cosmetic: skip printing fragmented XML tags
                                                    else:
                                                        clean_token = r_token.replace("<think>\n", "").replace("<think>", "")
                                                        _emit_thinking(clean_token)

                                        if "content" in delta and delta["content"] is not None:
                                            text_chunk = delta["content"]
                                            full_content += text_chunk
                                            if not stop_printing:
                                                if "</think" in full_content.lower() or "<tool_call" in full_content.lower():
                                                    stop_printing = True
                                                if not stop_printing and not reasoning_content:
                                                    if text_chunk.strip().lower() in ["<", "</", "think", ">", "think>", "<function", "<parameter"]:
                                                        pass  # Cosmetic: skip printing fragmented XML tags
                                                    else:
                                                        clean_token = text_chunk.replace("<think>\n", "").replace("<think>", "")
                                                        _emit_thinking(clean_token)

                                            # Tool-call generation-collapse detector.
                                            # Specialised fail-fast probe for the
                                            # `<tool_call>`-spam shape (see the
                                            # 8135-openings-in-97k-chars production
                                            # trace). Fires after ~10 unclosed opens
                                            # — typically within 1-3 seconds of the
                                            # decoder entering the loop, versus the
                                            # 300+ seconds it used to take to hit
                                            # max_tokens. The generic n-gram
                                            # thinking-loop detector above eventually
                                            # catches this too, but only after
                                            # ~600 chars of repetition.
                                            if _detect_tool_call_loop(full_content):
                                                thinking_loop_detected = True
                                                _opens = len(re.findall(r'<tool_call\b', full_content, re.IGNORECASE))
                                                _closes = len(re.findall(r'</tool_call\b', full_content, re.IGNORECASE))
                                                pretty_log(
                                                    "Tool-Call Loop",
                                                    f"Decoder collapse: {_opens} <tool_call> opens vs {_closes} closes "
                                                    f"at {len(full_content)} chars. Aborting stream.",
                                                    level="WARNING", icon=Icons.STOP,
                                                )
                                                break

                                        # --- Streaming sanity guards ---
                                        # Two failure modes to catch: (a) the model
                                        # produces an unbounded amount of thinking
                                        # without ever closing </think>, (b) it
                                        # falls into a self-repeating paragraph
                                        # loop. Both manifest as a runaway buffer
                                        # with no tool call.
                                        guard_buf = reasoning_content if reasoning_content else full_content
                                        # Self-play can install a tighter cap via
                                        # `max_thinking_chars_override` on the
                                        # GhostAgent instance (we know the
                                        # simulation is bounded and can't afford
                                        # 32k chars of wasted introspection).
                                        # Progressive thinking budget: start at 32K, but
                                        # if the model is producing diverse content (no
                                        # n-gram repetition at the initial cap), extend
                                        # to 64K. This lets complex algorithmic reasoning
                                        # and multi-step debugging breathe while still
                                        # killing genuine loops.
                                        override_cap = getattr(self, "max_thinking_chars_override", None)
                                        base_cap = override_cap or MAX_THINKING_CHARS
                                        extended_cap = override_cap or MAX_THINKING_CHARS_EXTENDED

                                        if len(guard_buf) > base_cap and len(guard_buf) <= extended_cap:
                                            # At the initial cap boundary: check if content
                                            # is still diverse (not looping). If so, extend.
                                            if _detect_thinking_loop(guard_buf):
                                                thinking_loop_detected = True
                                                pretty_log("Thinking Loop", f"Detected repetition at {len(guard_buf)} chars. Aborting.", level="WARNING", icon=Icons.STOP)
                                                break
                                            # Content is diverse → allow extension silently
                                        elif len(guard_buf) > extended_cap:
                                            thinking_loop_detected = True
                                            pretty_log("Thinking Cap", f"Stream exceeded extended cap ({extended_cap} chars). Aborting turn.", level="WARNING", icon=Icons.STOP)
                                            break

                                        if len(guard_buf) >= next_loop_probe:
                                            next_loop_probe = len(guard_buf) + THINKING_LOOP_PROBE_EVERY
                                            if _detect_thinking_loop(guard_buf):
                                                thinking_loop_detected = True
                                                pretty_log("Thinking Loop", "Detected n-gram repetition in stream. Aborting turn.", level="WARNING", icon=Icons.STOP)
                                                break

                                        if "tool_calls" in delta and delta["tool_calls"]:
                                            if not msg.get("tool_calls"):
                                                msg["tool_calls"] = []
                                            for tc_chunk in delta["tool_calls"]:
                                                idx = tc_chunk.get("index", 0)
                                                while len(msg["tool_calls"]) <= idx:
                                                    msg["tool_calls"].append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                                                if tc_chunk.get("id"):
                                                    msg["tool_calls"][idx]["id"] = tc_chunk["id"]
                                                if tc_chunk.get("function"):
                                                    fn_chunk = tc_chunk["function"]
                                                    if fn_chunk.get("name"):
                                                        msg["tool_calls"][idx]["function"]["name"] += fn_chunk["name"]
                                                    if fn_chunk.get("arguments"):
                                                        msg["tool_calls"][idx]["function"]["arguments"] += fn_chunk["arguments"]
                            except Exception as e:
                                logger.debug(f"XML Tool parse text stream chunk error: {type(e).__name__}")

                        _flush_thinking()
                        thinking_duration = time.monotonic() - thinking_started
                        if thinking_token_count > 0 or reasoning_content or full_content:
                            reasoning_chars = len(reasoning_content)
                            content_chars = len(full_content)
                            # Show reasoning and content sizes separately. The
                            # previous `{thinking_token_count} tokens · {chars}
                            # chars` form conflated the reasoning-channel token
                            # count (e.g. 143) with TOTAL chars across both
                            # channels (e.g. 48821), producing a misleading
                            # 341-chars-per-token ratio in the log that looked
                            # like a degenerate generation when in fact it was
                            # just a long `execute` tool_call body.
                            pretty_log(
                                "thought",
                                f"reasoning: {thinking_token_count} tokens / {reasoning_chars} chars "
                                f"| content: {content_chars} chars "
                                f"| {thinking_duration:.1f}s",
                                icon=Icons.BRAIN_SUM,
                            )

                        merged_content = full_content
                        if reasoning_content:
                            merged_content = f"<think>\n{reasoning_content}\n</think>\n" + full_content

                        # --- CROSS-TURN REPETITION GUARD (STREAMING) ---
                        # Intra-stream loop detection only sees one turn.
                        # This compares the first 300 chars of this
                        # turn's reasoning to the prior turn's; two
                        # consecutive hits at Jaccard ≥ 0.7 means the
                        # solver is re-entering the same derivation
                        # across turns and no retry will unstick it.
                        # Must run BEFORE the <think> strip below,
                        # otherwise the opening is already gone.
                        _stream_first_think_match = re.search(
                            r'<think>(.*?)(?:</think>|$)',
                            merged_content, flags=re.DOTALL | re.IGNORECASE,
                        )
                        _stream_first_think = (
                            _stream_first_think_match.group(1).strip()
                            if _stream_first_think_match else reasoning_content[:300]
                        )
                        _stream_opening_words = self._opening_word_set(_stream_first_think)
                        if len(_stream_opening_words) >= 8 and prev_turn_opening_words:
                            _inter = _stream_opening_words & prev_turn_opening_words
                            _uni = _stream_opening_words | prev_turn_opening_words
                            _jac = len(_inter) / len(_uni) if _uni else 0.0
                            if _jac >= 0.7:
                                cross_turn_repeat_hits += 1
                                pretty_log(
                                    "Cross-Turn Repetition",
                                    f"Turn opening overlaps prior turn by {_jac:.0%} (hit {cross_turn_repeat_hits}/2).",
                                    level="WARNING", icon=Icons.WARN,
                                )
                                if cross_turn_repeat_hits >= 2:
                                    pretty_log(
                                        "Loop Breaker",
                                        "Cross-turn repetition loop — aborting attempt.",
                                        level="WARNING", icon=Icons.STOP,
                                    )
                                    final_ai_content = (
                                        "[ATTEMPT_ABORTED_CROSS_TURN_LOOP] The solver opened "
                                        "three consecutive turns with near-identical reasoning "
                                        f"(Jaccard {_jac:.0%}). Further retries would repeat "
                                        "the same derivation. Stopping."
                                    )
                                    force_stop = True
                                    prev_turn_opening_words = _stream_opening_words
                                    break
                            else:
                                cross_turn_repeat_hits = 0
                        prev_turn_opening_words = _stream_opening_words

                        # CRITICAL FIX: Strip <think> blocks from permanent history to prevent cognitive looping
                        clean_msg_content = re.sub(r'<think>.*?(?:</think>|(?=<(?:tool_call|function))|$)', '', merged_content, flags=re.DOTALL | re.IGNORECASE).strip()
                        msg["content"] = clean_msg_content

                        if thinking_loop_detected:
                            # Discard the runaway thinking entirely so it can't
                            # poison the next turn, and inject a hard reset
                            # message instead of letting the empty assistant
                            # turn fall through normal parsing.
                            msg["content"] = ""
                            msg["tool_calls"] = []
                            execution_failure_count += 1
                            thinking_cap_events += 1
                            messages.append({"role": "assistant", "content": "[Internal thinking aborted: runaway loop detected.]"})
                            # Escalation: on the SECOND cap/loop event in
                            # the same attempt, stop retrying. The solver
                            # is stuck in a self-consistent but unwinnable
                            # derivation (e.g. "split('\\n') on '' can't
                            # return []") — no amount of "stop re-deriving"
                            # reminders will unstick it. Force-stop and let
                            # the caller surface the failure. First event
                            # still gets the old retry path so normal
                            # one-off over-thinking is recoverable.
                            if thinking_cap_events >= 2:
                                pretty_log(
                                    "Loop Breaker",
                                    f"Thinking cap hit {thinking_cap_events}x in one attempt — aborting.",
                                    level="WARNING", icon=Icons.STOP,
                                )
                                final_ai_content = (
                                    "[ATTEMPT_ABORTED_THINKING_LOOP] The solver hit the thinking "
                                    f"cap {thinking_cap_events} times in this attempt without "
                                    "producing a tool call. Further retries would re-enter the "
                                    "same derivation. Stopping."
                                )
                                force_stop = True
                                break
                            messages.append({"role": "user", "content": "SYSTEM ALERT: Your previous turn entered a self-repeating thinking loop and was killed. STOP re-deriving the same paragraph. If a self-generated test assertion disagrees with your function's output, the TEST is likely wrong — re-read the spec and fix the assertion before changing the function. If you have ALREADY proven the task cannot be solved as specified (e.g. the validator has a structural bug), call `abort_attempt` now with a specific reason. Otherwise output a <tool_call> immediately. Do not write a long <think> block."})
                            if execution_failure_count >= 6:
                                pretty_log("Loop Breaker", "Forcing final response after thinking loops", icon=Icons.STOP)
                                force_final_response = True
                            continue
                    except (httpx.ConnectError, httpx.ConnectTimeout):
                        final_ai_content = "CRITICAL: The upstream LLM server is unreachable. It may have crashed due to memory pressure or is currently restarting. Please wait a moment and try again."
                        pretty_log("System Fault", "Upstream server unreachable", level="ERROR", icon=Icons.FAIL)
                        force_stop = True
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 400 and "context" in e.response.text.lower():
                            pretty_log("Context Overflow", "Emergency pruning triggered...", icon=Icons.WARN)
                            # Emergency Prune: Keep System + Last User + 1 Last Tool Result (Truncated)
                            system_msgs = [m for m in req_messages if m.get("role") == "system"]
                            last_user = next((m for m in reversed(req_messages) if m.get("role") == "user"), None)

                            recovery_msgs = list(system_msgs)
                            if last_user:
                                safe_user = last_user.copy()
                                if isinstance(safe_user.get("content"), str) and len(safe_user["content"]) > 10000:
                                    safe_user["content"] = safe_user["content"][:10000] + "\n... [EMERGENCY TRUNCATION] ..."
                                recovery_msgs.append(safe_user)

                            # If the last thing was a tool output that caused the overflow, keep it but heavily truncated.
                            # Walk back to the last *real* tool entry — a synthetic
                            # agent-loop error (parse-error nudge, etc.) is not real
                            # prior tool output and pretending it is leads the recovery
                            # retry to plan against fabricated evidence. Also strip
                            # internal-tracking keys (`_synthetic`) before forwarding
                            # upstream so the LLM payload stays clean OpenAI-shape.
                            real_tool = _find_substantive_tool_for_verifier(
                                tools_run_this_turn
                            )
                            if real_tool is not None:
                                last_tool = {
                                    k: v for k, v in real_tool.items()
                                    if not k.startswith("_")
                                }
                                last_tool["content"] = str(last_tool.get("content", ""))[:1000] + "\n... [EMERGENCY TRUNCATION] ..."
                                recovery_msgs.append(last_tool)

                            recovery_msgs.append({"role": "user", "content": "SYSTEM ALERT: The conversation history was truncated to fit within context limits. Continue task. Assume previous context has been handled."})

                            # RETRY ONCE with pruned context
                            try:
                                payload["messages"] = recovery_msgs
                                messages = recovery_msgs
                                data = await self.context.llm_client.chat_completion(payload, use_coding=has_coding_intent)
                                if "choices" in data and len(data["choices"]) > 0:
                                    msg = data["choices"][0]["message"]
                            except Exception as retry_e:
                                final_ai_content = f"CRITICAL: Context overflow recovery failed: {str(retry_e)}"
                                force_stop = True
                                break
                        else:
                            final_ai_content = f"CRITICAL: Upstream error {e.response.status_code}: {e.response.text}"
                            pretty_log("System Fault", f"HTTP {e.response.status_code}", level="ERROR", icon=Icons.FAIL)
                            force_stop = True
                            break
                    except Exception as e:
                        final_ai_content = f"CRITICAL: An unexpected error occurred while communicating with the LLM: {str(e)}"
                        pretty_log("System Fault", str(e), level="ERROR", icon=Icons.FAIL)
                        force_stop = True
                        break

                    content = msg.get("content") or ""

                    # Merge upstream reasoning_content if present (some models return it as a separate field)
                    if msg.get("reasoning_content"):
                        content = f"<think>\n{msg.get('reasoning_content')}\n</think>\n" + content

                    tool_calls = []
                    ui_content = content

                    # --- EXTRACT & LOG INTERNAL THINKING ---
                    think_matches = re.findall(r'<think>(.*?)(?:</think>|$)', content, flags=re.DOTALL | re.IGNORECASE)
                    for think_text in think_matches:
                        clean_think = think_text.strip()
                        if clean_think:
                            ui_think = clean_think.replace('\n', ' | ')

                            # 1. Trigger UI Planner Monologue Box
                            logger.info(f"PLANNER MONOLOGUE: {ui_think}")

                            # 2. Print multiline thinking safely to the terminal (REMOVED - now handled via streaming)
                            # Timestamp and iterative printing to the console has been moved to the streaming block above.

                    # ---------------------------------------------------------
                    #   ROBUST XML TOOL PARSER
                    # ---------------------------------------------------------
                    # Isolate actual output from <think> blocks FIRST
                    parse_target = re.sub(r'<think>.*?(?:</think>|(?=<(?:tool_call|function))|$)', '', content, flags=re.DOTALL | re.IGNORECASE)

                    # --- XML TAG NORMALIZATION ---
                    # Heal prompt-induced hallucinations (<tool> instead of <tool_call>)
                    parse_target = re.sub(r'<tool\b[^>]*>', '<tool_call>', parse_target, flags=re.IGNORECASE)
                    parse_target = re.sub(r'</tool>', '</tool_call>', parse_target, flags=re.IGNORECASE)

                    # Heal bare <function> tags missing the <tool_call> wrapper
                    if '<function' in parse_target and '<tool_call' not in parse_target:
                        parse_target = parse_target.replace('<function', '<tool_call>\n<function')

                    # Heal sloppy attribute syntax from newer Qwen variants
                    # (3.6+). They drop the `name` attribute and/or pad `=`
                    # with whitespace, producing shapes the downstream
                    # `<function(?:\s+name=|=)` regex never matched:
                    #   <function = "x">         → <function name="x">
                    #   <function name = "x">    → <function name="x">
                    #   <function_name = "x">    → <function name="x">
                    #   <function_name="x">      → <function name="x">
                    #   <parameter = "x">        → <parameter name="x">
                    #   <parameter name = "x">   → <parameter name="x">
                    # Runs AFTER the bare-<function> heal so the inserted
                    # wrapper doesn't get rewritten.
                    parse_target = re.sub(
                        r'<(function|parameter)(?:_name)?\s*=\s*(["\'])([^"\']+)\2',
                        r'<\1 name=\2\3\2',
                        parse_target, flags=re.IGNORECASE,
                    )
                    parse_target = re.sub(
                        r'<(function|parameter)\s+name\s*=\s*',
                        r'<\1 name=',
                        parse_target, flags=re.IGNORECASE,
                    )
                    # -----------------------------

                    has_tool_tag = re.search(r'<(?:tool_call|function)\b', parse_target, re.IGNORECASE) is not None

                    # Per-turn diagnosis of the parse reason. Populated when
                    # the parser fails so the recovery-message branch can
                    # surface a SPECIFIC cause (truncated output, no
                    # function tag, etc.) instead of the generic "your XML
                    # was invalid" — which sends the model guessing.
                    parse_failure_reason = ""

                    if has_tool_tag:
                        pretty_log("Agent Parser", "Extracting XML tool call...", icon=Icons.TOOL_CODE)

                        # --- UPSTREAM TRUNCATION DETECTOR ---
                        # If the assistant output opens a <tool_call> or
                        # <function> but never closes it, the upstream max
                        # tokens cap severed the stream mid-XML. We cannot
                        # recover a clean tool_call from a dangling header,
                        # but we CAN tell the model exactly what happened on
                        # the next turn — previously it got a generic "your
                        # XML is broken" and guessed at CDATA / heredoc /
                        # base64 while the actual problem was "your output
                        # was cut off, shorten it."
                        if _tool_call_truncated(parse_target):
                            parse_failure_reason = "truncated"
                            # Detailed truncation diagnostics. Previously we
                            # logged only the last 600 chars, which was not
                            # enough to diagnose e.g. "the model emitted a
                            # massive <parameter name='content'> Python body
                            # that hit the upstream token cap" vs. "the model
                            # opened a second <tool_call> and ran out of room
                            # in the second one's <function> tag." We now
                            # surface the total length, tool_call/function
                            # open/close counts, the head of the response, and
                            # the tail — enough signal to distinguish all the
                            # common truncation shapes from the log alone.
                            _tc_opens = len(re.findall(r'<tool_call\b', parse_target, re.IGNORECASE))
                            _tc_closes = len(re.findall(r'</tool_call\b', parse_target, re.IGNORECASE))
                            _fn_opens = len(re.findall(r'<function\b[^>]*>', parse_target, re.IGNORECASE))
                            _fn_closes = len(re.findall(r'</function\b', parse_target, re.IGNORECASE))
                            _param_opens = len(re.findall(r'<parameter\b[^>]*>', parse_target, re.IGNORECASE))
                            _param_closes = len(re.findall(r'</parameter\b', parse_target, re.IGNORECASE))
                            _tail_sample = parse_target[-600:].replace("\n", "\\n")
                            _head_sample = parse_target[:300].replace("\n", "\\n")
                            logger.warning(
                                "Parse target truncated. len=%d chars. "
                                "tool_call=%d/%d function=%d/%d parameter=%d/%d (open/close). "
                                "HEAD: %s... TAIL: ...%s",
                                len(parse_target),
                                _tc_opens, _tc_closes,
                                _fn_opens, _fn_closes,
                                _param_opens, _param_closes,
                                _head_sample, _tail_sample,
                            )
                            # Also surface the key numbers in the pretty stream
                            # so they show up in the live log the user watches,
                            # not just the file-based logger output.
                            pretty_log(
                                "Upstream Truncation",
                                f"len={len(parse_target)} chars | "
                                f"tool_call open/close = {_tc_opens}/{_tc_closes} | "
                                f"function = {_fn_opens}/{_fn_closes} | "
                                f"parameter = {_param_opens}/{_param_closes}",
                                level="WARNING", icon=Icons.WARN,
                            )

                        # Split by <tool_call> to handle missing closing tags, spaces, and markdown injections
                        # Heal <tool_call name="execute"> hallucinations before they are destroyed by re.split
                        parse_target = re.sub(r'<tool_call\s+(?:name|function)=["\']([^"\']+)["\'][^>]*>', r'<tool_call>\n<function name="\1">', parse_target, flags=re.IGNORECASE)
                        blocks = re.split(r'<tool_call.*?>', parse_target, flags=re.IGNORECASE)
                        # Cap parse attempts per response. A degenerate generation
                        # (Qwen has been observed looping on a truncated tool_call
                        # shape, producing thousands of malformed blocks in one
                        # reply — see the 4056-strike trace) would otherwise emit
                        # one system_parse_error per block and drain the strike
                        # counter into the thousands before the turn-level cap
                        # check runs. Five is plenty to recover a real call.
                        _MAX_TOOL_CALL_BLOCKS = 5
                        if len(blocks) > _MAX_TOOL_CALL_BLOCKS + 1:
                            logger.warning(
                                "Degenerate response: %d <tool_call> openings; truncating to %d to prevent parser flood.",
                                len(blocks) - 1, _MAX_TOOL_CALL_BLOCKS,
                            )
                            blocks = blocks[: _MAX_TOOL_CALL_BLOCKS + 1]
                        # Dedupe: when upstream cuts the response off mid
                        # tool_call, the split produces N fragments that all
                        # fail for the SAME root cause (truncation). Emitting
                        # one system_parse_error per fragment multiplies a
                        # single failure into N strikes on the execution
                        # counter — the model gets penalized N times for one
                        # mistake, and the truncation-specific recovery hint
                        # fires N times in the log. Track whether we've
                        # already emitted a truncation-reason error in this
                        # response and skip subsequent ones; non-truncation
                        # failures (no_function_tag, malformed) still emit
                        # normally because they're genuinely distinct.
                        emitted_truncation_error = False
                        for block in blocks[1:]:
                            # Strip out anything after the closing tag if it exists
                            block_content = re.split(r'</tool_call.*?>', block, flags=re.IGNORECASE)[0]

                            try:
                                # Fallback: if it's pure JSON wrapped in <tool_call> without <function>
                                t_data = None
                                if '<function' not in block_content.lower():
                                    try:
                                        t_data = extract_json_from_text(block_content)
                                        if t_data and "name" in t_data:
                                            func_match = None  # skip XML path, use t_data directly
                                        else:
                                            t_data = None
                                    except Exception:
                                        t_data = None

                                if t_data is None:
                                    func_match = re.search(r'<function(?:\s+name=|=)(.*?)>', block_content, re.IGNORECASE)
                                else:
                                    func_match = None
                                if func_match:
                                    func_name_raw = func_match.group(1).strip()
                                    func_name = func_name_raw.strip('"').strip("'").split()[0].strip('"').strip("'")
                                    args_val = {}

                                    # Extract stray attributes from the <function> tag itself
                                    attr_matches = re.finditer(r'([a-zA-Z0-9_-]+)=["\'](.*?)["\']', func_match.group(0))
                                    for a in attr_matches:
                                        if a.group(1).lower() not in ['name', 'function', 'tool_call']:
                                            args_val[a.group(1)] = a.group(2)

                                    # Format 0a: CDATA envelope — `<parameter name="x"><![CDATA[ANYTHING]]></parameter>`
                                    # The model can wrap content with `<![CDATA[...]]>` to embed
                                    # literal `</parameter>`, `<`, `>`, JSON, code, etc. without
                                    # the regex-based parser truncating early. Match these FIRST
                                    # and remove from the working block so subsequent regexes
                                    # don't double-parse the inner body. Strict — only fires when
                                    # the model actually emitted `<![CDATA[...]]>`, so it can never
                                    # corrupt non-CDATA tool calls.
                                    cdata_pattern = re.compile(
                                        r'<parameter(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_-]+)["\']?\s*>\s*<!\[CDATA\[(.*?)\]\]>\s*</parameter>',
                                        re.DOTALL | re.IGNORECASE,
                                    )
                                    cdata_hits = list(cdata_pattern.finditer(block_content))
                                    for cm in cdata_hits:
                                        args_val[cm.group(1).strip()] = cm.group(2)

                                    # Format 1: <parameter name="x">y</parameter> (Handles missing closing tags)
                                    # The lookahead also breaks on a stray *opening* `<function`
                                    # or `<tool_call` — when the model emits two tool calls in one
                                    # turn and uses a non-`</parameter>` close tag (e.g. `</path>`,
                                    # `</file_path>`) on the first call's last param, the lazy
                                    # `.*?` would otherwise consume past the first call's
                                    # `</function>` / `</tool_call>` pair (when block-splitting
                                    # already missed them) and latch onto the *second* call's
                                    # `</parameter>`, so a single param ends up holding both
                                    # tool calls' tag-soup as its value. The opening-tag stops
                                    # are belt-and-braces against block-split misses.
                                    param_matches = list(re.finditer(r'<parameter(?:\s+name=|=)([^>]+)>(.*?)(?=</parameter>|<parameter(?:\s+name=|=)|<function\b|</function>|<tool_call\b|</tool_call>|$)', block_content, re.DOTALL | re.IGNORECASE))
                                    for p in param_matches:
                                        p_name = p.group(1).split()[0].strip().strip('"').strip("'") # grab first token to avoid attribute bleed
                                        p_val = p.group(2).strip('\r\n')
                                        if p_name not in args_val:
                                            args_val[p_name] = p_val

                                    # Format 2: <parameter name="x" value="y" /> (Handles inner quotes)
                                    alt_matches = list(re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s+value=(["\'])(.*?)\2\s*(?:/|>.*?</parameter>)', block_content, re.DOTALL | re.IGNORECASE))
                                    for p in alt_matches:
                                        args_val[p.group(1)] = p.group(3)

                                    # Format 3: Bare tags <action>check_health</action> (Handles missing closing tags)
                                    bare_tags = list(re.finditer(r'<([a-zA-Z0-9_-]+)>(.*?)(?=</\1>|<[a-zA-Z0-9_-]+>|<[a-zA-Z0-9_-]+ |</function>|</tool_call>|$)', block_content, re.DOTALL | re.IGNORECASE))
                                    for b in bare_tags:
                                        b_name = b.group(1).lower()
                                        if b_name not in ['function', 'tool_call', 'parameter', 'arguments', 'parameters', 'kwargs']: # ignore structural tags
                                            if b_name not in args_val: # don't overwrite explicit parameter tags
                                                args_val[b_name] = b.group(2).strip('\r\n')

                                    # Format 4: Attribute tags <parameter action="check_health" /> (Handles inner quotes)
                                    attr_tags = list(re.finditer(r'<parameter\s+([a-zA-Z0-9_-]+)=(["\'])(.*?)\2\s*(?:/|>.*?</parameter>)', block_content, re.DOTALL | re.IGNORECASE))
                                    for a in attr_tags:
                                        if a.group(1).lower() != 'name' and a.group(1).lower() not in args_val:
                                            args_val[a.group(1)] = a.group(3)

                                    # Format 5: Direct attribute tags <action="check_health"> (Handles inner quotes)
                                    direct_attr = list(re.finditer(r'<([a-zA-Z0-9_-]+)=(["\'])(.*?)\2\s*(?:/|>.*?</\1>|>)', block_content, re.DOTALL | re.IGNORECASE))
                                    for d in direct_attr:
                                        if d.group(1).lower() not in ['function', 'tool_call', 'parameter', 'arguments', 'parameters', 'kwargs'] and d.group(1).lower() not in args_val:
                                            args_val[d.group(1)] = d.group(3)

                                    # Format 5b: Bounds-aware REPAIR pass — Format 1's non-greedy
                                    # regex truncates param bodies at the FIRST `</parameter>` it
                                    # finds, so any `</parameter>` literal inside a docstring or
                                    # string example silently mangles the content. This repair
                                    # walks `<parameter>` openings in order and uses the LAST
                                    # `</parameter>` before the next opening (or `</function>`)
                                    # as the body terminator. It runs AFTER Formats 1-5 and only
                                    # REPLACES a value when the repaired body is strictly longer
                                    # — so clean cases (Formats 1-5 extracted correctly) are never
                                    # perturbed, and CDATA-wrapped / self-closing / value= shapes
                                    # are untouched because their bodies are already populated.
                                    try:
                                        open_pattern = re.compile(
                                            r'<parameter(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_-]+)["\']?[^>]*>',
                                            re.IGNORECASE,
                                        )
                                        close_re = re.compile(r'</parameter>', re.IGNORECASE)
                                        # Mirror Format 1's boundary set: a stray `<function`
                                        # or `<tool_call` opening (block-split miss) terminates
                                        # the body just like a closing tag, so we never repair
                                        # a value past the start of the next tool call.
                                        end_func_re = re.compile(r'<function\b|</function>|<tool_call\b|</tool_call>', re.IGNORECASE)
                                        openings = list(open_pattern.finditer(block_content))
                                        # The FIRST `<function>` opening is the legitimate one
                                        # for this block — searching the boundary regex from
                                        # position 0 would latch onto it and pin `end_pos = 0`,
                                        # killing every parameter extraction. Start the search
                                        # AFTER the first `<function>` opening (if present), so
                                        # the next `<function>` / `<tool_call>` opening — which
                                        # only happens when block-split missed a boundary —
                                        # correctly terminates the function body.
                                        first_func = re.search(r'<function\b', block_content, re.IGNORECASE)
                                        search_start = first_func.end() if first_func else 0
                                        end_func = end_func_re.search(block_content, search_start)
                                        end_pos = end_func.start() if end_func else len(block_content)
                                        for i, op in enumerate(openings):
                                            p_name = op.group(1).strip().strip('"').strip("'")
                                            # Skip if the opening is inside a CDATA-wrapped body we
                                            # already consumed — would double-parse the inner text.
                                            in_cdata = any(cm.start() <= op.start() < cm.end() for cm in cdata_hits)
                                            if in_cdata:
                                                continue
                                            # Self-closing `<parameter .../>` has no body — skip.
                                            if block_content[op.end() - 2:op.end()] == "/>":
                                                continue
                                            body_start = op.end()
                                            next_open = openings[i + 1].start() if i + 1 < len(openings) else end_pos
                                            boundary = min(next_open, end_pos)
                                            candidates = [m for m in close_re.finditer(block_content, body_start, boundary)]
                                            if not candidates:
                                                # No `</parameter>` in range — Format 1 handled this
                                                # with its `|</function>|$` fallback. Don't second-
                                                # guess it; leave existing value alone.
                                                continue
                                            body_end = candidates[-1].start()
                                            if body_end <= body_start:
                                                continue
                                            repaired = block_content[body_start:body_end].strip("\r\n")
                                            existing = args_val.get(p_name)
                                            # Only REPLACE when the repair is strictly longer AND
                                            # the existing value looks truncated (Format 1 stopped
                                            # at a literal `</parameter>` inside the body).
                                            if isinstance(existing, str) and len(repaired) > len(existing):
                                                args_val[p_name] = repaired
                                            elif existing is None:
                                                args_val[p_name] = repaired
                                    except Exception:
                                        # Repair pass is best-effort. Never let it crash the
                                        # outer parse path — Formats 1-5 already produced an
                                        # args_val and we still have the remaining fallbacks.
                                        pass

                                    if not args_val:
                                        # Format 6: Try JSON inside the block if XML parsing failed
                                        try:
                                            possible_json = extract_json_from_text(block_content)
                                            if possible_json and isinstance(possible_json, dict):
                                                args_val = possible_json.get("arguments", possible_json)
                                        except Exception:
                                            pass

                                        # Format 7: Single-argument pure text fallback
                                        if not args_val and func_name in ["image_generation", "vision_analysis", "deep_research", "deep_think"]:
                                                    # Extract text between <function> and </function>
                                                    raw_content_match = re.search(r'<function[^>]*>(.*?)</function>', block_content, re.DOTALL | re.IGNORECASE)
                                                    if raw_content_match:
                                                        raw_text = raw_content_match.group(1).strip('\r\n')
                                                        if raw_text and not raw_text.startswith('<') and not raw_text.startswith('{'):
                                                            # Assign to the primary argument based on function name
                                                            if func_name == "image_generation":
                                                                args_val["prompt"] = raw_text
                                                            elif func_name == "vision_analysis":
                                                                args_val["target"] = raw_text
                                                                args_val["action"] = "describe_picture"
                                                            elif func_name == "deep_research" or func_name == "deep_think":
                                                                args_val["query"] = raw_text

                                    t_data = {"name": func_name, "arguments": args_val}
                                else:
                                    t_data = extract_json_from_text(block_content)

                                # Heal missing "name" if the JSON root key is the tool name
                                if t_data and "name" not in t_data and isinstance(t_data, dict):
                                    keys = list(t_data.keys())
                                    available_tool_names = list(self.available_tools.keys()) if hasattr(self, 'available_tools') else []
                                    if len(keys) == 1 and keys[0] in available_tool_names:
                                        t_data = {"name": keys[0], "arguments": t_data[keys[0]]}

                                # Fix for models that output OpenAI nested JSON inside XML
                                if t_data and "function" in t_data and isinstance(t_data["function"], dict) and "name" in t_data["function"]:
                                    t_data = t_data["function"]

                                if t_data and "name" in t_data:
                                    # Fix for models that stringify the arguments dict
                                    args_val = t_data.get("arguments", {})
                                    if isinstance(args_val, str):
                                        try: args_val = json.loads(args_val, strict=False)
                                        except: args_val = {}

                                    # Un-nest hallucinatory wrappers like <arguments> or <parameters>
                                    if isinstance(args_val, dict) and len(args_val) == 1:
                                        k = list(args_val.keys())[0].lower()
                                        if k in ["arguments", "parameters", "kwargs", "args"]:
                                            inner = args_val[list(args_val.keys())[0]]
                                            if isinstance(inner, str):
                                                try:
                                                    parsed = extract_json_from_text(inner)
                                                    if parsed and isinstance(parsed, dict):
                                                        args_val = parsed
                                                except: pass
                                            elif isinstance(inner, dict):
                                                args_val = inner

                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": t_data.get("name"),
                                            "arguments": json.dumps(args_val)
                                        }
                                    })
                                else:
                                    # Fallback 1: Raw text to JSON didn't work. Check if the LLM hallucinated <tool_name>...</tool_name> tags instead of <function name="...">
                                    first_tag_match = re.search(r'<([a-zA-Z0-9_-]+)[^>]*>', block_content.strip(), re.IGNORECASE)
                                    available_tool_names = list(self.available_tools.keys()) if hasattr(self, 'available_tools') else []

                                    if first_tag_match and first_tag_match.group(1).lower() not in ['tool_call', 'function']:
                                        func_fallback = first_tag_match.group(1)

                                        # Only accept if it's a known tool OR if we don't have available_tools (just blindly try)
                                        if not available_tool_names or func_fallback in available_tool_names:
                                            args_fallback = {}

                                            inner_content = re.sub(f'^\\s*<{func_fallback}[^>]*>', '', block_content, flags=re.IGNORECASE)
                                            inner_content = re.sub(f'</{func_fallback}>\\s*$', '', inner_content, flags=re.IGNORECASE).strip()

                                            bare_tags = list(re.finditer(r'<([a-zA-Z0-9_-]+)>(.*?)</\1>', inner_content, re.DOTALL | re.IGNORECASE))
                                            for b in bare_tags:
                                                if b.group(1).lower() not in [func_fallback.lower(), 'tool_call']:
                                                    args_fallback[b.group(1)] = b.group(2).strip('\r\n')

                                            if not args_fallback and func_fallback in ["image_generation", "vision_analysis", "deep_research", "deep_think"]:
                                                if inner_content and not inner_content.startswith('<') and not inner_content.startswith('{'):
                                                    if func_fallback == "image_generation": args_fallback["prompt"] = inner_content
                                                    elif func_fallback == "vision_analysis":
                                                        args_fallback["target"] = inner_content
                                                        args_fallback["action"] = "describe_picture"
                                                    elif func_fallback in ["deep_research", "deep_think"]: args_fallback["query"] = inner_content

                                            t_data = {"name": func_fallback, "arguments": args_fallback}

                                    if t_data and "name" in t_data:
                                        # The fallback successfully mapped a tool!
                                        args_val = t_data.get("arguments", {})
                                        tool_calls.append({
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": t_data["name"],
                                                "arguments": json.dumps(args_val) if isinstance(args_val, dict) else (args_val if isinstance(args_val, str) else json.dumps(args_val))
                                            }
                                        })
                                    else:
                                        # Extreme Regex Fallback for unescaped broken JSON (e.g. execute and file_system tools)
                                        name_match = re.search(r'<(?:name|tool_name)>(.*?)</(?:name|tool_name)>', block_content, re.IGNORECASE)
                                        if not name_match:
                                            name_match = re.search(r'(?:"|\')name(?:"|\')\s*:\s*(?:"|\')(.*?)(?:"|\')', block_content)
                                        tool_name_fallback = name_match.group(1).strip() if name_match else None

                                        if not tool_name_fallback and 'filename' in block_content:
                                            tool_name_fallback = 'execute'

                                        if tool_name_fallback == 'execute' and 'filename' in block_content and 'content' in block_content:
                                            f_match = re.search(r'(?:"|\')filename(?:"|\')\s*:\s*(?:"|\')(.*?)(?:"|\')', block_content)
                                            c_match = re.search(r'(?:"|\')content(?:"|\')\s*:\s*(?:"|\')(.*)', block_content, re.DOTALL)

                                            if f_match and c_match:
                                                raw_content = c_match.group(1).strip('\r\n')
                                                if raw_content.endswith('}'): raw_content = raw_content[:-1].strip('\r\n')
                                                if raw_content.endswith('"""'): raw_content = raw_content[:-3].strip('\r\n')
                                                elif raw_content.endswith('"') or raw_content.endswith("'"): raw_content = raw_content[:-1].strip('\r\n')

                                                tool_calls.append({
                                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "execute",
                                                        "arguments": json.dumps({"filename": f_match.group(1), "content": raw_content})
                                                    }
                                                })
                                                continue

                                        elif tool_name_fallback == 'file_system' and 'path' in block_content:
                                            op_match = re.search(r'(?:"|\')operation(?:"|\')\s*:\s*(?:"|\')(.*?)(?:"|\')', block_content)
                                            p_match = re.search(r'(?:"|\')path(?:"|\')\s*:\s*(?:"|\')(.*?)(?:"|\')', block_content)
                                            c_match = re.search(r'(?:"|\')content(?:"|\')\s*:\s*(?:"|\')(.*?)(?=(?:,\s*(?:"|\')replace_with(?:"|\')\s*:|$))', block_content, re.DOTALL)
                                            rw_match = re.search(r'(?:"|\')replace_with(?:"|\')\s*:\s*(?:"|\')(.*)', block_content, re.DOTALL)

                                            if op_match and p_match:
                                                args_dict = {
                                                    "operation": op_match.group(1),
                                                    "path": p_match.group(1)
                                                }
                                                if c_match:
                                                    raw_content = c_match.group(1).strip('\r\n')
                                                    if raw_content.endswith('}'): raw_content = raw_content[:-1].strip('\r\n')
                                                    if raw_content.endswith(','): raw_content = raw_content[:-1].strip('\r\n')
                                                    if raw_content.endswith('"""'): raw_content = raw_content[:-3].strip('\r\n')
                                                    elif raw_content.endswith('"') or raw_content.endswith("'"): raw_content = raw_content[:-1].strip('\r\n')
                                                    args_dict["content"] = raw_content

                                                if rw_match:
                                                    raw_rw = rw_match.group(1).strip('\r\n')
                                                    if raw_rw.endswith('}'): raw_rw = raw_rw[:-1].strip('\r\n')
                                                    if raw_rw.endswith(','): raw_rw = raw_rw[:-1].strip('\r\n')
                                                    if raw_rw.endswith('"""'): raw_rw = raw_rw[:-3].strip('\r\n')
                                                    elif raw_rw.endswith('"') or raw_rw.endswith("'"): raw_rw = raw_rw[:-1].strip('\r\n')
                                                    args_dict["replace_with"] = raw_rw

                                                tool_calls.append({
                                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "file_system",
                                                        "arguments": json.dumps(args_dict)
                                                    }
                                                })
                                                continue

                                        elif tool_name_fallback in ['vision_analysis', 'image_generation', 'deep_research', 'deep_think']:
                                            target_match = re.search(r'<(target|prompt|query)>(.*?)</\1>', block_content, re.DOTALL | re.IGNORECASE)
                                            if not target_match:
                                                target_match = re.search(r'(?:"|\')(?:target|prompt|query)(?:"|\')\s*:\s*(?:"|\')(.*?)(?:"|\')', block_content, re.DOTALL)

                                            if target_match:
                                                extracted_val = target_match.group(2) if len(target_match.groups()) > 1 else target_match.group(1)
                                                arg_key = "target" if tool_name_fallback == "vision_analysis" else ("prompt" if tool_name_fallback == "image_generation" else "query")
                                                args_dict = {arg_key: extracted_val.strip()}
                                                if tool_name_fallback == "vision_analysis":
                                                    args_dict["action"] = "describe_picture"

                                                tool_calls.append({
                                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_name_fallback,
                                                        "arguments": json.dumps(args_dict)
                                                    }
                                                })
                                                continue

                                        # Absolute failure, all syntax mangled. Log the first 4 KB
                                        # of the block so we can see WHY the parser rejected it
                                        # — selfplay sessions used to log only "tool syntax error"
                                        # with no diagnostic signal, making this regression class
                                        # impossible to debug from traces alone.
                                        try:
                                            logger.warning(
                                                "Parser emitted system_parse_error. "
                                                "Block content (truncated to 4 KB): %s",
                                                block_content[:4096].replace("\n", "\\n"),
                                            )
                                        except Exception:
                                            pass

                                        # Stamp a specific failure reason so the
                                        # recovery-message branch can produce an
                                        # actionable hint ("no <function> tag
                                        # found — did you forget the XML header?").
                                        if not parse_failure_reason:
                                            if '<function' not in block_content.lower():
                                                parse_failure_reason = "no_function_tag"
                                            elif re.search(r'<function[^>]*$', block_content, re.IGNORECASE):
                                                parse_failure_reason = "truncated"
                                            else:
                                                parse_failure_reason = "malformed"
                                        # Dedupe: fragments produced by a
                                        # single upstream truncation all share
                                        # `parse_failure_reason == "truncated"`.
                                        # Emit one synthetic error for the
                                        # whole response, skip the rest.
                                        if parse_failure_reason == "truncated" and emitted_truncation_error:
                                            continue
                                        if parse_failure_reason == "truncated":
                                            emitted_truncation_error = True
                                        tool_calls.append({
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": "system_parse_error",
                                                "arguments": "{}"
                                            }
                                        })
                            except Exception as e:
                                logger.debug(f"XML execution metadata parsing error: {type(e).__name__}")

                        # Scrub from the human-facing UI string. Three shapes
                        # can leak into the reply: `<tool_call>...</tool_call>`,
                        # `<tool>...</tool>`, and a bare `<function
                        # name="...">...</function>` (emitted without the outer
                        # wrapper — has_tool_tag still True via the `<function`
                        # branch, but the earlier regex only caught `<tool_call`
                        # / `<tool>` and left the bare function shape in the
                        # user-facing text). Backreference `\1` forces the close
                        # tag to match the same type as the open — otherwise a
                        # nested `</function>` inside `<tool_call>...</tool_call>`
                        # would terminate the outer match early and leave an
                        # orphan `</tool_call>` behind.
                        ui_content = re.sub(
                            # `\Z` (absolute EOS) instead of `$` — see the
                            # stream-wrapper pattern for the full rationale.
                            # The short version: `$` in non-MULTILINE mode
                            # matches before a trailing `\n`, letting the
                            # newline after `<tool_call>` escape the scrub.
                            r'<(tool_call|tool|function)\b[^>]*>.*?(?:</\1\b[^>]*>|\Z)',
                            '',
                            ui_content,
                            flags=re.DOTALL | re.IGNORECASE,
                        ).strip()
                    else:
                        # Fallback: honour native tool_calls if the model didn't use XML format
                        tool_calls = list(msg.get("tool_calls") or [])

                        if not tool_calls and parse_target.strip().startswith('{'):
                            # Fallback: check if it outputted raw JSON instead of XML
                            try:
                                possible_json = extract_json_from_text(parse_target)
                                if possible_json and isinstance(possible_json, dict) and "name" in possible_json and "arguments" in possible_json:
                                    args_val = possible_json.get("arguments", {})
                                    if isinstance(args_val, str):
                                        try: args_val = json.loads(args_val)
                                        except: pass
                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": possible_json["name"],
                                            "arguments": json.dumps(args_val) if isinstance(args_val, dict) else (args_val if isinstance(args_val, str) else json.dumps(args_val))
                                        }
                                    })
                                    pretty_log("Agent Parser", "Recovered raw JSON tool call.", icon=Icons.TOOL_CODE)
                            except Exception:
                                pass

                    ui_content = re.sub(r'<think>.*?(?:</think>|(?=<(?:tool_call|function))|$)', '', ui_content, flags=re.DOTALL | re.IGNORECASE).strip()

                    # --- HALLUCINATION & LEAK SCRUBBERS ---
                    if ui_content:
                        # 1. Hard Truncation for System Prompt Bleed
                        for bleed_marker in ["# Tools", "<tools>", "CRITICAL INSTRUCTION:", "You may call one or more functions", '{"type": "function"']:
                            if bleed_marker in ui_content:
                                ui_content = ui_content.split(bleed_marker)[0]

                        # 2. Regex scrubbers for XML and Execution Artifacts
                        ui_content = re.sub(r'<tool_response>.*?(?:</tool_response>|$)', '', ui_content, flags=re.DOTALL | re.IGNORECASE)
                        ui_content = re.sub(r'--- EXECUTION RESULT ---.*?(?:------------------------|$)', '', ui_content, flags=re.DOTALL)

                        # 3. Task Tree Regurgitation Scrubbers
                        # Old pattern was `^\s*(?: )\s*\[.*?\].*?\n?` which
                        # stripped any indented `[label]` prefix — that
                        # mangled legitimate indented markdown links
                        # ('  [docs](https://x)' → '(https://x)'). Tightened
                        # to require a task-shape token inside the
                        # brackets (a `task_NN` id or one of the status
                        # keywords) so markdown links survive.
                        ui_content = re.sub(r'(?m)^\s*\[(?:task_\d+|IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\b[^\]]*\].*?\n?', '', ui_content)
                        ui_content = re.sub(r'(?m)^.*?\((?:IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\)\s*\n?', '', ui_content)
                        ui_content = re.sub(r'(?m)^\s*(?:\[)?task_\d+(?:\])?\s*\n?', '', ui_content)
                        ui_content = re.sub(r'(?m)^\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\s*', '', ui_content)

                        ui_content = ui_content.strip()

                    # CRITICAL: Preserve the raw XML tags in the assistant's internal message context so it remembers!
                    # BUT STRIP <think> blocks to prevent cognitive looping!
                    clean_content_for_history = re.sub(r'<think>.*?(?:</think>|(?=<(?:tool_call|function))|$)', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                    msg["content"] = clean_content_for_history
                    msg["tool_calls"] = tool_calls

                    # Defense-in-depth for terminal tools and other cases
                    # where an earlier step set `force_final_response`. The
                    # turn was promised to be text-only, but the model can
                    # still emit a `<tool_call>` — especially after a
                    # terminal tool like `self_play` whose result text
                    # ("DO NOT call ... again") is buried inside a
                    # `<tool_response>` block and loses to the strong
                    # system-prompt directive above it. Drop those
                    # hallucinated tool_calls so the loop converges.
                    if force_final_response and tool_calls:
                        dropped = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                        logger.warning(
                            "Dropping %d tool_call(s) — force_final_response is set (names=%s)",
                            len(tool_calls), dropped,
                        )
                        tool_calls = []
                        msg["tool_calls"] = []

                    # Reasoning-channel divergence guard. Some models (Qwen-class
                    # reasoning variants in particular) emit `reasoning_content`
                    # explicitly disclaiming tool use ("I can answer directly
                    # without using any tools") AND still emit a structured
                    # tool_call in the same response. Trust the reasoning: drop
                    # the contradicting tool_calls and re-run the turn in
                    # final-generation mode so the model produces prose. This
                    # avoids the wasted strike on a tool call the model itself
                    # said wasn't needed (e.g. spurious `knowledge_base` saves
                    # of prose the user asked us to compose).
                    elif tool_calls and not force_final_response:
                        rc = locals().get("reasoning_content", "") or (msg.get("reasoning_content") or "")
                        if rc:
                            _NO_TOOL_DISCLAIM_PATTERNS = (
                                r"\bwithout\s+(?:needing\s+)?(?:any\s+|using\s+|calling\s+)?tools?\b",
                                r"\bno\s+tools?\s+(?:are\s+)?(?:needed|required|necessary)\b",
                                r"\bdon'?t\s+need\s+(?:any\s+|to\s+(?:use|call)\s+)?tools?\b",
                                r"\banswer\s+(?:this\s+)?directly\s+from\s+(?:my\s+)?knowledge\b",
                                r"\bI\s+can\s+answer\s+(?:this\s+)?directly\b",
                            )
                            if any(re.search(p, rc, re.IGNORECASE) for p in _NO_TOOL_DISCLAIM_PATTERNS):
                                dropped = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                                logger.warning(
                                    "Dropping %d tool_call(s) — reasoning channel disclaimed tools (names=%s, reasoning_head=%r)",
                                    len(tool_calls), dropped, rc[:200],
                                )
                                tool_calls = []
                                msg["tool_calls"] = []
                                force_final_response = True
                                # Re-run the turn in final-generation mode rather
                                # than emitting the bad-turn message — its
                                # think-stripped content is empty, so falling
                                # through would surface nothing useful.
                                continue

                    # Telemetry for un-caught divergences. When BOTH channels
                    # emit and neither drop fired, the regex set above missed
                    # the phrasing — log a sample at debug so the pattern list
                    # can be extended as new model phrasings appear.
                    if tool_calls:
                        _rc_for_log = locals().get("reasoning_content", "") or (msg.get("reasoning_content") or "")
                        if _rc_for_log and len(_rc_for_log) > 50:
                            logger.debug(
                                "Dual-channel emit: reasoning_content (%d chars) + tool_calls (%d, names=%s); reasoning_head=%r",
                                len(_rc_for_log), len(tool_calls),
                                [tc.get("function", {}).get("name", "?") for tc in tool_calls],
                                _rc_for_log[:120],
                            )

                    if not tool_calls:
                        clean_ui = ui_content.strip("` \n\r")
                        has_img_markdown = bool(re.search(r'!\[.*?\]\(.*?\)', clean_ui))
                        has_valid_image_tool = any(t in raw_tools_called for t in ["image_generation", "execute", "file_system"])
                        has_run_tools = len(tools_run_this_turn) > 0
                        # Catch Stalled Image Mentions
                        if has_img_markdown and not has_valid_image_tool:
                            is_valid_final = "```" in clean_ui or bool(re.search(r'\b(SUCCESS|DONE|COMPLETE|ERROR)\b', clean_ui.upper()))

                            # Validate image links to see if they are preexisting (not hallucinated)
                            if has_img_markdown and not is_valid_final:
                                img_links = re.findall(r'!\[.*?\]\((.*?)\)', clean_ui)
                                history_text = str([m.get("content", "") for m in messages[-4:]])
                                all_links_valid = len(img_links) > 0 and all(link in history_text for link in img_links)
                                if all_links_valid:
                                    is_valid_final = True

                            if not is_valid_final:
                                pretty_log("Agent Parser", "Caught image markdown without tool call.", level="WARNING", icon=Icons.WARN)
                                messages.append(msg)
                                messages.append({"role": "user", "content": "SYSTEM ALERT: You attempted to display an image using markdown `![]()` but you forgot to actually generate it! You MUST output the XML `<tool_call>` for `image_generation` NOW. DO NOT output the markdown tag until the tool successfully returns the filename."})
                                execution_failure_count += 1
                                continue

                        # Catch Conversational Filler promising a tool call.
                        #
                        # Old version did a raw substring match on the tool
                        # name, which false-positived on any casual reply
                        # that used the word naturally — `execute` is a
                        # common English verb, `forget` is a common English
                        # verb, and tool phrases like `file system` /
                        # `knowledge base` / `deep think` come up all the
                        # time in philosophical / meta conversations. The
                        # false positive cascaded into a full "SYSTEM ALERT:
                        # output the XML!" injection that trapped the model
                        # in an execute-tool-call loop even when the user
                        # was just chatting (see 23:09 log: user asked
                        # about consciousness, reply mentioned "structured
                        # execution via tools", guard fired, Turn 2 tried
                        # to call execute and truncated).
                        #
                        # New rule: fire only when BOTH
                        #   (a) the tool name matches with word boundaries
                        #       (so `executed` / `executive` / `execution`
                        #       don't match `execute`), AND
                        #   (b) an explicit intent marker ("I'll", "let me",
                        #       "running", "calling", "using") sits within
                        #       ~12 words of the tool name — casual mention
                        #       ("execute is a common English verb") does
                        #       not have this pattern, a real tool-promise
                        #       ("Let me execute that now") does.
                        if clean_ui and not force_final_response and not is_final_generation:
                            tool_names = list(self.available_tools.keys()) if hasattr(self, 'available_tools') else []
                            clean_ui_lower = clean_ui.lower()
                            # Intent markers the model uses when it actually
                            # means to run a tool. Kept narrow on purpose —
                            # over-broad markers (e.g. "will", "can") would
                            # reintroduce false positives on prose.
                            _intent_pattern = re.compile(
                                r"\b(?:i['’]?ll|i\s+will|i\s+am\s+going\s+to|"
                                r"let\s+me|let's|gonna|"
                                r"now\s+(?:i['’]?m\s+)?(?:running|calling|using|executing|"
                                r"invoking|firing)|"
                                r"running|calling|invoking|firing\s+off|executing\s+(?:the|a))\b",
                                re.IGNORECASE,
                            )
                            has_intent = bool(_intent_pattern.search(clean_ui_lower))

                            mentioned_tools = []
                            if has_intent:
                                for t in tool_names:
                                    # Word-boundary match on both the raw
                                    # `tool_name` and the space-separated
                                    # form `tool name` so `file_system` is
                                    # caught in either shape without
                                    # matching `file` alone.
                                    pat_underscore = rf"\b{re.escape(t)}\b"
                                    pat_spaced = rf"\b{re.escape(t.replace('_', ' '))}\b"
                                    if re.search(pat_underscore, clean_ui_lower) or (
                                        "_" in t and re.search(pat_spaced, clean_ui_lower)
                                    ):
                                        mentioned_tools.append(t)

                            if mentioned_tools and not has_run_tools:
                                is_valid_final = "```" in clean_ui or bool(re.search(r'\b(SUCCESS|DONE|COMPLETE|ERROR)\b', clean_ui.upper()))
                                if not is_valid_final and len(clean_ui.split()) < 100:
                                    pretty_log("Agent Parser", f"Caught conversational filler without XML ({mentioned_tools[0]}).", level="WARNING", icon=Icons.WARN)
                                    messages.append(msg)
                                    messages.append({"role": "user", "content": f"SYSTEM ALERT: You provided conversational text mentioning the tool `{mentioned_tools[0]}`, but you DID NOT output the actual XML `<tool_call>` block! Do not narrate your actions. Output the XML `<tool_call>` immediately."})
                                    execution_failure_count += 1
                                    continue

                        # Conversational fallback removed for smarter models.
                        if not clean_ui and not force_final_response and not is_final_generation:
                            pretty_log("Agent Parser", "Model stalled after thinking. Forcing retry.", level="WARNING", icon=Icons.WARN)
                            messages.append(msg)
                            messages.append({"role": "user", "content": "SYSTEM ALERT: You generated a thought process but stopped abruptly without outputting a valid XML <tool_call> or a response to the user. DO NOT STOP. If you need to use a tool, output the required XML <tool_call> block. If the task is fully complete, provide your final response to the user."})
                            execution_failure_count += 1
                            continue

                        user_request_context = last_user_content.lower()
                        has_meta_intent = any(kw in user_request_context for kw in ["learn", "skill", "profile", "lesson", "playbook", "memorize"])
                        meta_tools_called = any(t in raw_tools_called for t in ["learn_skill", "update_profile", "create_skill", "manage_skills"])
                        # Read-only SURFACE tools discharge the meta-task
                        # nudge: if the user asks "what have you learned
                        # today" the right answer is `list_lessons`, not a
                        # bogus `learn_skill` write. Before this exemption
                        # the nudge kept firing for 3-5 extra turns after
                        # `list_lessons` returned, and the model eventually
                        # caved and wrote a deduplicated no-op skill just
                        # to silence it (production trace 15:17, request
                        # 5C: 6 turns / 73s for one read-only query).
                        read_only_meta_tools_called = any(
                            t in raw_tools_called
                            for t in ["list_lessons", "recall", "manage_skills"]
                        )
                        if read_only_meta_tools_called:
                            meta_tools_called = True

                        meta_tools_available = any(t in self.available_tools for t in ["learn_skill", "update_profile", "create_skill", "manage_skills"])
                        # Self-play (and any isolated simulation) sets
                        # `suppress_meta_task_nudges` to skip this check —
                        # the nudge is a production-mode feature that
                        # pushes the agent to record skills/profile updates
                        # after a real user task, and it has no place
                        # inside a throwaway simulation where all memory
                        # writes are blocked anyway.
                        suppress_nudge = getattr(self, "suppress_meta_task_nudges", False)
                        if not suppress_nudge and has_meta_intent and meta_tools_available and not meta_tools_called and turn < 4:
                            pretty_log("Checklist Nudge", "Enforcing meta-task compliance", icon=Icons.SHIELD)
                            messages.append({"role": "user", "content": "CRITICAL: You have not fulfilled the learning/profile instructions in the user's request. You MUST call 'learn_skill' or 'update_profile' now before finishing."})
                            continue

                        if ui_content:
                            ui_content = ui_content.replace("\r", "")
                            if final_ai_content and not final_ai_content.endswith("\n\n"):
                                final_ai_content += "\n\n"
                            final_ai_content += ui_content

                        if self.context.args.smart_memory > 0.0 and last_user_content and not forget_was_called and not last_was_failure:
                            micro_msgs = []
                            for m in [msg for msg in messages if msg.get("role") in ["user", "assistant"]][-4:]:
                                role = m.get("role", "user").upper()
                                clean_content = re.sub(r'```.*?```', '', str(m.get("content", "")), flags=re.DOTALL)
                                micro_msgs.append(f"{role}: {clean_content[:500].strip()}")
                            clean_ai = re.sub(r'```.*?```', '', final_ai_content, flags=re.DOTALL)
                            recent_arc = "\n".join(micro_msgs) + f"\nAI: {clean_ai[:500].strip()}"
                            if getattr(self.context, 'journal', None):
                                await asyncio.to_thread(self.context.journal.append, 'smart_memory', {'text': recent_arc, 'model': model})
                        break

                    # Capture the end-of-prior-iteration boundary so we
                    # can rollback this iteration's preamble flush if
                    # every tool_call below ends up synthetic (parse
                    # error, invalid JSON args, unknown/disabled tool,
                    # idempotency block, empty-write block). Without the
                    # rollback, a confused iteration's preamble
                    # ("Thank you for the kind words!") concatenates to
                    # the NEXT iteration's real response and the user
                    # sees a Frankenstein reply.
                    # Trace: 2026-05-01 dialog log turn 28.
                    _pre_flush_final_len = len(final_ai_content)

                    if ui_content:
                        ui_content = ui_content.replace("\r", "")
                        if final_ai_content and not final_ai_content.endswith("\n\n"):
                            final_ai_content += "\n\n"
                        final_ai_content += ui_content

                    messages.append(msg)
                    last_was_failure = False

                    if tool_calls:
                        import html
                        def unescape_xml_values(val):
                            if isinstance(val, str):
                                return html.unescape(val).replace('\\"', '"').replace("\\'", "'")
                            elif isinstance(val, dict):
                                return {k: unescape_xml_values(v) for k, v in val.items()}
                            elif isinstance(val, list):
                                return [unescape_xml_values(v) for v in val]
                            return val

                        for tc in tool_calls:
                            try:
                                args_dict = json.loads(tc["function"]["arguments"], strict=False)
                                tc["function"]["arguments"] = json.dumps(unescape_xml_values(args_dict))
                            except: pass

                    tool_tasks, tool_call_metadata = [], []
                    for _tc_idx, tool in enumerate(tool_calls):
                        # Strike cap inside the per-tool loop. The outer cap
                        # at the top of the turn loop only runs at turn
                        # boundaries, so without this a single response
                        # carrying many system_parse_error entries drains
                        # the whole list before the cap fires next turn.
                        if execution_failure_count >= 6:
                            logger.warning(
                                "Strike cap hit mid-loop (execution_failure_count=%d); skipping remaining %d tool_call(s).",
                                execution_failure_count, len(tool_calls) - _tc_idx,
                            )
                            break
                        fname = tool["function"]["name"]
                        raw_tools_called.add(fname)
                        tool_usage[fname] = tool_usage.get(fname, 0) + 1



                        if fname == "forget":
                            forget_was_called = True
                        elif fname == "knowledge_base":
                            try:
                                args = json.loads(tool["function"]["arguments"])
                                if args.get("action") == "forget":
                                    forget_was_called = True
                            except: pass

                        if hasattr(self, 'disabled_tools') and fname in self.disabled_tools:
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"SYSTEM ERROR: Tool '{fname}' is explicitly disabled in this context."}
                            messages.append(err_msg)
                            tools_run_this_turn.append({**err_msg, "_synthetic": True})
                            execution_failure_count += 1
                            last_was_failure = True
                            continue

                        if fname == "system_parse_error":
                            consecutive_parse_errors += 1
                            pretty_log(
                                "Tool Syntax Error",
                                f"Failed to parse tool call (consecutive={consecutive_parse_errors}, reason={parse_failure_reason or 'unknown'})",
                                level="WARNING", icon=Icons.WARN,
                            )

                            # Reason-specific recovery hints. The prior
                            # implementation repeated the same generic "use
                            # XML" reminder on every failure, which trapped
                            # the model in a loop because it didn't say
                            # WHAT broke. Match the hint to the detected
                            # cause so the model can actually fix it.
                            if parse_failure_reason == "truncated":
                                # Most common failure in the self-play
                                # trace: upstream max_tokens cap severed
                                # the <tool_call> mid-content. The model
                                # has no visibility into the cap; tell it.
                                err_msg_content = (
                                    "SYSTEM ERROR: Your previous output was CUT OFF before the "
                                    "`<tool_call>` finished. The upstream server truncated your "
                                    "response mid-tag, so no closing `</parameter></function></tool_call>` "
                                    "ever arrived. This is NOT an XML-syntax problem — it is a "
                                    "length problem.\n\n"
                                    "FIX: shorten your output on the next turn. Pick ONE:\n"
                                    "  1. Write a SHORTER Python script (under 80 lines). Split "
                                    "work into multiple tool calls if needed.\n"
                                    "  2. Use `file_system` `operation=\"write\"` directly "
                                    "instead of cramming code into an `execute` `content` param.\n"
                                    "  3. Use `file_system` `operation=\"replace\"` to edit one "
                                    "chunk of an existing file rather than rewriting the whole thing.\n\n"
                                    "Stop trying CDATA / heredoc / base64 — those do not fix "
                                    "truncation."
                                )
                            elif parse_failure_reason == "no_function_tag":
                                err_msg_content = (
                                    "SYSTEM ERROR: Your `<tool_call>` block was present but "
                                    "contained no `<function name=\"...\">` tag. Output the "
                                    "tool call with the exact shape:\n"
                                    "<tool_call>\n  <function name=\"the_tool_name\">\n"
                                    "    <parameter name=\"arg1\">value1</parameter>\n"
                                    "  </function>\n</tool_call>"
                                )
                            elif consecutive_parse_errors >= 2:
                                # Switched-strategy hint only after the
                                # second failure, and only for genuinely
                                # malformed (not truncated) output.
                                err_msg_content = (
                                    f"SYSTEM ESCAPE HATCH (parse failed {consecutive_parse_errors}x): "
                                    "STOP repeating the same shape — it does not parse.\n\n"
                                    "Pick ONE alternative:\n"
                                    "(A) CDATA envelope for content with literal `</parameter>` / `<` / `>` / JSON:\n"
                                    "    `<parameter name=\"content\"><![CDATA[...]]></parameter>`\n"
                                    "(B) `file_system` `operation=\"write\"` (native) instead of `execute`.\n"
                                    "(C) `file_system` `operation=\"replace\"` for small edits.\n\n"
                                    "Output ONE complete tool_call. Do not ask for clarification."
                                )
                            else:
                                err_msg_content = (
                                    "SYSTEM ERROR: Your `<tool_call>` did not parse. Use strict XML:\n"
                                    "<tool_call>\n  <function name=\"the_tool_name\">\n"
                                    "    <parameter name=\"arg1\">value1</parameter>\n"
                                    "  </function>\n</tool_call>\n\n"
                                    "If a parameter body contains literal `</parameter>` / `<` / `>` / JSON, "
                                    "wrap it in `<![CDATA[ ... ]]>`."
                                )

                            err_msg = {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "name": "system",
                                "content": err_msg_content,
                            }
                            messages.append(err_msg)
                            tools_run_this_turn.append({**err_msg, "_synthetic": True})
                            execution_failure_count += 1
                            last_was_failure = True
                            continue

                        try:
                            t_args = json.loads(tool["function"]["arguments"], strict=False)

                            is_sandbox_mutation = fname in ["execute", "image_generation"] or \
                                                  (fname == "file_system" and t_args.get("operation") in ["write", "replace", "download", "delete", "move", "rename", "unzip", "git_clone"])

                            if is_sandbox_mutation:
                                # Invalidate both the legacy global and the
                                # per-request local cache so the next turn
                                # re-lists the workspace.
                                self.context.cached_sandbox_state = None
                                request_sandbox_state = None
                                request_state.invalidate_sandbox()
                            # Skill writes invalidate the playbook cache too.
                            if fname == "learn_skill":
                                request_state.invalidate_skill_playbook()

                            a_hash = f"{fname}:{json.dumps(t_args, sort_keys=True)}"
                        except Exception as e:
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"Error: Invalid JSON arguments - {str(e)}"}
                            messages.append(err_msg)
                            tools_run_this_turn.append({**err_msg, "_synthetic": True})
                            execution_failure_count += 1
                            last_was_failure = True
                            continue

                        is_mutating = fname in ["execute", "manage_tasks", "update_profile", "learn_skill", "vision_analysis"] or \
                                      (fname == "file_system" and t_args.get("operation") in ["write", "replace", "download", "delete", "move", "rename"]) or \
                                      (fname == "knowledge_base" and t_args.get("action") in ["ingest_document", "forget", "reset_all", "insert_fact"])

                        # --- IDEMPOTENCY GUARD (production loop fix) ---
                        # Pure setters with no value in repetition. Re-issuing
                        # them with identical args is always a model loop, not
                        # a legitimate retry. `execute` and `file_system.write`
                        # are intentionally NOT in this set: rerunning a script
                        # after an external fix is legitimate.
                        is_idempotent_setter = (
                            fname in ("update_profile", "learn_skill")
                            or (fname == "knowledge_base"
                                and t_args.get("action") in ("insert_fact", "forget"))
                        )
                        if is_idempotent_setter and a_hash in executed_idempotent:
                            pretty_log(
                                "Idempotency Guard",
                                f"Blocked duplicate {fname} call (args already applied)",
                                icon=Icons.STOP,
                            )
                            err_msg = {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "name": fname,
                                "content": (
                                    f"SYSTEM IDEMPOTENCY: '{fname}' was already executed earlier in this "
                                    f"request with these exact arguments. The intended state is already "
                                    f"applied. DO NOT call it again — proceed to the next step or finalize "
                                    f"your response to the user."
                                ),
                            }
                            messages.append(err_msg)
                            tools_run_this_turn.append({**err_msg, "_synthetic": True})
                            continue
                        if is_idempotent_setter:
                            executed_idempotent.add(a_hash)

                        seen_tools.add(a_hash)

                        if fname == "file_system":
                            op = t_args.get("operation")
                            if op == "write":
                                content_val = t_args.get("content")
                                if not content_val or not str(content_val).strip():
                                    # Allow empty writes for filenames that
                                    # legitimately have zero-byte bodies by
                                    # convention; block every other empty
                                    # write because it usually signals a
                                    # truncated/hallucinated tool call.
                                    # Allowlist (match on the filename, not
                                    # just the full path, so any subdir
                                    # variant is covered):
                                    #   __init__.py   — Python package marker
                                    #   py.typed      — PEP 561 inline-typing marker
                                    #   .gitkeep      — preserve empty directories in git
                                    #   .nojekyll     — GitHub Pages marker
                                    #   .gitignore    — sometimes written empty as placeholder
                                    #   conftest.py   — rare but legit empty pytest root marker
                                    _p_raw = str(t_args.get("path", ""))
                                    _basename = _p_raw.rsplit("/", 1)[-1]
                                    _ALLOW_EMPTY = {
                                        "__init__.py", "py.typed", ".gitkeep",
                                        ".nojekyll", ".gitignore", "conftest.py",
                                    }
                                    _is_allowed_empty = _basename in _ALLOW_EMPTY
                                    if not _is_allowed_empty:
                                        # Surface the path in the log so the
                                        # operator can tell a true hallucination
                                        # (`temp.py`, `output.txt`) from a
                                        # legitimate empty-file convention we
                                        # haven't whitelisted yet.
                                        pretty_log(
                                            "Local Guard",
                                            f"Blocked file_system write with empty content "
                                            f"(path={_p_raw!r}, basename={_basename!r})",
                                            icon=Icons.STOP,
                                        )
                                        err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"SYSTEM BLOCK: You invoked file_system operation='write' on path={_p_raw!r} but provided an empty or missing 'content' argument. This is completely useless and causes context bloat. Review your task and provide the ACTUAL FULL CONTENT when writing a file. (If you genuinely meant to create an empty file, only these basenames are allowed empty: __init__.py, py.typed, .gitkeep, .nojekyll, .gitignore, conftest.py.) The operation was aborted before execution."}
                                        messages.append(err_msg)
                                        tools_run_this_turn.append({**err_msg, "_synthetic": True})
                                        execution_failure_count += 1
                                        last_was_failure = True
                                        continue
                                    else:
                                        # Normalise to an empty string so the
                                        # downstream write path doesn't
                                        # NoneType-crash; this is the actual
                                        # intended file body.
                                        pretty_log(
                                            "Local Guard",
                                            f"Allowed empty {_basename} write "
                                            f"(path={_p_raw!r}) — conventional zero-byte file.",
                                            icon=Icons.OK,
                                        )
                                        t_args["content"] = ""

                                target_path = str(t_args.get("path", "")).lower()

                        if fname not in self.available_tools:
                            self.available_tools = get_available_tools(self.context)

                        # --- TOOL NAME CANONICALIZATION ---
                        # Qwen 3.5 (and other models) hallucinates tool name
                        # variants like "filesystem", "update-profile",
                        # "knowledgebase". Without canonicalization the
                        # dispatcher silently 404s and the model wastes a
                        # whole turn before retrying.
                        if fname not in self.available_tools:
                            canonical = self._canonicalise_tool_name(fname, list(self.available_tools.keys()))
                            if canonical:
                                pretty_log("Tool Alias", f"{fname} → {canonical}", icon=Icons.RETRY)
                                fname = canonical

                        if fname in self.available_tools:
                            try:
                                tool_tasks.append(self.available_tools[fname](**t_args))
                                tool_call_metadata.append((fname, tool["id"], a_hash, is_mutating))
                            except Exception as e:
                                pretty_log("Tool Invocation Error", str(e), level="WARNING", icon=Icons.WARN)
                                err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"Error invoking tool '{fname}' (Did you forget a required argument?): {str(e)}"}
                                messages.append(err_msg)
                                tools_run_this_turn.append({**err_msg, "_synthetic": True})
                                last_was_failure = True
                        else:
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"Error: Unknown tool '{fname}'"}
                            messages.append(err_msg)
                            tools_run_this_turn.append({**err_msg, "_synthetic": True})
                            execution_failure_count += 1

                    # If every tool_call this iteration was rejected
                    # synthetically, the model's preamble text was a
                    # stale deflection — drop it so it does not bleed
                    # into the next iteration's real response. We keep
                    # `messages.append(msg)` and the synthetic tool
                    # entries intact so the LLM still sees its own
                    # confused attempt + the corrective system message
                    # on the next iteration.
                    if tool_calls and not tool_tasks:
                        if len(final_ai_content) > _pre_flush_final_len:
                            final_ai_content = final_ai_content[:_pre_flush_final_len]

                    if tool_tasks:
                        # CRITICAL FIX: Run mutating tools (like file writes) BEFORE execution tools to prevent race conditions
                        results = [None] * len(tool_tasks)

                        # Phase 1: Mutations
                        mutation_coros = []
                        for i, (task, meta) in enumerate(zip(tool_tasks, tool_call_metadata)):
                            if meta[0] == "file_system":
                                mutation_coros.append((i, task))

                        if mutation_coros:
                            mut_results = await asyncio.gather(*(c[1] for c in mutation_coros), return_exceptions=True)
                            for (i, _), res in zip(mutation_coros, mut_results):
                                results[i] = res

                        # Phase 2: Executions
                        exec_coros = []
                        for i, (task, meta) in enumerate(zip(tool_tasks, tool_call_metadata)):
                            if meta[0] != "file_system":
                                exec_coros.append((i, task))

                        if exec_coros:
                            exec_results = await asyncio.gather(*(c[1] for c in exec_coros), return_exceptions=True)
                            for (i, _), res in zip(exec_coros, exec_results):
                                results[i] = res

                        turn_has_failure = False
                        last_error_res = ""
                        last_error_preview = "Unknown Error"

                        # We reached the parallel-execution path, which means
                        # at least one tool call parsed cleanly this turn.
                        # Drop the consecutive-parse-error streak so a single
                        # earlier failure can't latch the pivot prompt.
                        consecutive_parse_errors = 0

                        for i, result in enumerate(results):
                            fname, tool_id, a_hash, is_mutating = tool_call_metadata[i]
                            str_res = str(result).replace("\r", "") if not isinstance(result, Exception) else f"Error: {str(result)}"

                            shield_limit = max(16000, int(char_budget * 0.1))
                            if len(str_res) > shield_limit and fname not in ["file_system", "recall", "deep_research", "web_search", "knowledge_base", "postgres_admin"]:
                                payload = {
                                    "model": model,
                                    "messages": [{"role": "user", "content": f"The user asked: '{last_user_content}'. Summarize this tool output. If it contains facts relevant to the user, extract them. If it is a script error, state the root cause. Output: {str_res[:15000]}"}],
                                    "temperature": 0.0,
                                    "max_tokens": 300
                                }
                                try:
                                    pretty_log("Context Shield", f"Offloading {len(str_res)} chars from {fname} to Edge Worker...", icon=Icons.SHIELD)
                                    summary_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
                                    summary_content = summary_data["choices"][0]["message"].get("content", "").strip()
                                    if summary_content:
                                        str_res = f"[EDGE CONDENSED]: {summary_content}"
                                except Exception as e:
                                    logger.debug(f"Final generation XML parse failure: {type(e).__name__}")

                            trunc_limit = max(80000, int(char_budget * 0.4))
                            half = trunc_limit // 2
                            safe_res = str_res[:half] + "\n...[TRUNCATED]...\n" + str_res[-half:] if len(str_res) > trunc_limit else str_res

                            # Make sure exit-code / error markers are NEVER
                            # buried in the truncation. The model needs to
                            # see "EXIT CODE: 1" or "Error:" reliably to
                            # decide whether to retry or pivot.
                            if fname == "execute":
                                exit_match_full = re.search(r"EXIT CODE:\s*(\d+)", str_res)
                                if exit_match_full and "EXIT CODE:" not in safe_res:
                                    safe_res = f"[FAILURE BANNER] EXIT CODE: {exit_match_full.group(1)}\n" + safe_res
                            if (str_res.lstrip().startswith("Error")
                                    or "Traceback" in str_res
                                    or "SYSTEM ERROR" in str_res):
                                first_err_line = next(
                                    (ln for ln in str_res.splitlines()
                                     if ln.strip().startswith(("Error", "SYSTEM ERROR"))
                                     or "Traceback" in ln),
                                    "",
                                )
                                if first_err_line and first_err_line[:120] not in safe_res[:300]:
                                    safe_res = f"[FAILURE BANNER] {first_err_line[:200]}\n" + safe_res

                            # Per-tool fallback hint injection — when a tool
                            # call fails with a known error pattern, append a
                            # concrete remediation hint so the LLM has an
                            # actionable next step instead of blindly retrying.
                            if (str_res.lstrip().startswith("Error")
                                    or "Traceback" in str_res
                                    or "SYSTEM ERROR" in str_res
                                    or "EXIT CODE: 1" in str_res
                                    or "EXIT CODE: 2" in str_res):
                                try:
                                    from ..tools.tool_failure import get_fallback_hint
                                    hint = get_fallback_hint(fname, str_res)
                                    if hint:
                                        safe_res = safe_res + f"\n\n[FALLBACK HINT for {fname}] {hint}"
                                except Exception:
                                    pass

                            tool_msg = {"role": "tool", "tool_call_id": tool_id, "name": fname, "content": safe_res}
                            messages.append(tool_msg)
                            tools_run_this_turn.append(tool_msg)

                            # Escape-hatch tool: the solver has declared
                            # the current task structurally unwinnable.
                            # Stop the turn loop immediately and promote
                            # the sentinel into `final_ai_content` so the
                            # outer caller (e.g. dream.synthetic_self_play)
                            # can detect "CHALLENGE_ABORTED_BY_SOLVER"
                            # and skip remaining retry attempts. Firing
                            # on the tool RESULT (not the call) means the
                            # solver must have successfully run
                            # `abort_attempt` — hallucinated or malformed
                            # calls can't short-circuit the loop.
                            if fname == "abort_attempt" and "CHALLENGE_ABORTED_BY_SOLVER" in str_res:
                                pretty_log(
                                    "Attempt Aborted",
                                    "Solver declared task unwinnable — exiting turn loop",
                                    level="WARNING", icon=Icons.STOP,
                                )
                                final_ai_content = str_res
                                force_stop = True
                                break

                            if fname == "execute":
                                code_match = re.search(r"EXIT CODE:\s*(\d+)", str_res)
                                if code_match:
                                    exit_code_val = int(code_match.group(1))
                                else:
                                    if "Error" in str_res or "Exception" in str_res or "Traceback" in str_res:
                                        exit_code_val = 1
                                    else:
                                        exit_code_val = 0

                                if exit_code_val != 0:
                                    turn_has_failure = True
                                    last_error_res = str_res
                                    if "STDOUT/STDERR:" in str_res:
                                        last_error_preview = str_res.split("STDOUT/STDERR:")[1].strip().replace("\n", " ")
                                    elif "SYSTEM ERROR:" in str_res:
                                        last_error_preview = str_res.split("SYSTEM ERROR:")[1].strip().split("\n")[0]
                                    else:
                                        last_error_preview = str_res[:60].replace("\n", " ")
                                else:
                                    if is_mutating: seen_tools.clear()
                                    pretty_log("Execution Ok", "Script completed with exit code 0", icon=Icons.OK)

                                    # --- SIMULATION SHORT-CIRCUIT ---
                                    # In self-play (read-only skill memory
                                    # sentinel), when the solver has just
                                    # executed `solution.py` with exit=0 and
                                    # produced non-empty stdout, the inner
                                    # agent is done — the outer validator
                                    # will re-run solution.py directly and
                                    # ignore any further agent reasoning.
                                    # Without this gate, every successful
                                    # cycle burns one extra 15–25s "thinking"
                                    # turn to emit a "task complete" summary
                                    # that nobody reads (log-eval: turns 3→4
                                    # of every cycle were pure dead time).
                                    _is_sim = getattr(
                                        getattr(self.context, "skill_memory", None),
                                        "is_read_only",
                                        False,
                                    ) is True
                                    if _is_sim:
                                        # Pull the command text from the
                                        # assistant message that emitted this
                                        # tool_call — that's where the model
                                        # embedded `python3 solution.py` (or
                                        # `python3 -u solution.py`, or a
                                        # `bash -c "...solution.py..."`).
                                        _last_asst = ""
                                        for _m in reversed(messages):
                                            if _m.get("role") == "assistant":
                                                _tc = _m.get("tool_calls") or []
                                                try:
                                                    _last_asst = json.dumps(_tc, default=str)
                                                except Exception:
                                                    _last_asst = str(_tc)
                                                _last_asst += str(_m.get("content") or "")
                                                break
                                        _ran_solution = "solution.py" in _last_asst
                                        # Non-trivial stdout: strip the
                                        # EXECUTION RESULT header and check
                                        # the body has real content.
                                        _body = str_res.split("STDOUT/STDERR:", 1)[-1].strip()
                                        if _ran_solution and len(_body) > 0:
                                            final_ai_content = (
                                                "solution.py executed successfully (exit 0)."
                                            )
                                            force_stop = True
                                            pretty_log(
                                                "Self-Play Short-Circuit",
                                                "Skipped confirmation turn — solution.py ran clean (sim mode).",
                                                icon=Icons.STOP,
                                            )
                                            break  # exit the enumerate(results) loop

                            elif str_res.startswith("Error:") or str_res.startswith("ERROR") or str_res.startswith("SYSTEM ERROR") or str_res.startswith("Critical Tool Error"):
                                turn_has_failure = True
                                last_error_res = str_res
                                last_error_preview = str_res.replace("Error:", "").strip()
                                pretty_log("Tool Warning", f"{fname} -> {last_error_preview[:100]}", icon=Icons.WARN)

                            elif fname in ["manage_tasks", "learn_skill", "update_profile"] and "SUCCESS" in str_res.upper():
                                if is_mutating: seen_tools.clear()
                                pass
                            elif fname == "image_generation" and "SUCCESS" in str_res.upper():
                                pass
                            else:
                                if is_mutating: seen_tools.clear()

                        if turn_has_failure:
                            # Classify the failure to route to the right budget
                            from ..tools.tool_failure import classify_tool_failure, FailureClass, format_failure_context
                            failure_class, failure_match = classify_tool_failure(last_error_res or last_error_preview)

                            if failure_class == FailureClass.RETRYABLE:
                                transient_failure_count += 1
                                pretty_log("Transient Fail", f"Transient strike {transient_failure_count}/4 ({failure_match}) -> {last_error_preview[:100]}", icon=Icons.WARN)
                                diagnostic_msg = format_failure_context(last_error_preview, failure_class)
                            else:
                                execution_failure_count += 1
                                pretty_log("Execution Fail", f"Strike {execution_failure_count}/6 ({failure_class.value}) -> {last_error_preview[:150]}", icon=Icons.FAIL)
                                diagnostic_msg = format_failure_context(last_error_preview, failure_class)

                            last_was_failure = True

                            # Check for tool fallback suggestions
                            from ..tools.fallback_chains import get_fallback_hint
                            fallback_hint = ""
                            if fname:
                                hint = get_fallback_hint(fname, last_error_res or last_error_preview)
                                if hint:
                                    fallback_hint = f"\n\n{hint}"

                            from ..tools.file_system import tool_list_files
                            sandbox_state = await tool_list_files(self.context.sandbox_dir, self.context.memory_system)
                            messages.append({"role": "user", "content": f"AUTO-DIAGNOSTIC: {diagnostic_msg}{fallback_hint}\n\n{sandbox_state}"})

                            total_fail = execution_failure_count + transient_failure_count
                            # System 3 Crisis Pivot — fires at structural strike 4
                            # OR total strike 6. Can fire a SECOND time at strike 5
                            # with results of the first pivot as extra context.
                            sys3_trigger = (
                                (execution_failure_count == 4 and not _request_sys3_fired_once)
                                or (execution_failure_count == 5 and _request_sys3_fired_once)
                            )
                            if sys3_trigger:
                                pivot_num = 2 if _request_sys3_fired_once else 1
                                pretty_log(f"System 3 Crisis Intervention #{pivot_num}", "Engaging meta-cognitive pivot...", icon=Icons.BRAIN_THINK)

                                # On second pivot, include first pivot's justification
                                extra_context = ""
                                if pivot_num == 2:
                                    prev_just = _request_sys3_prev_justification
                                    extra_context = f"\n\n### PREVIOUS PIVOT (failed):\n{prev_just[:500]}"

                                sys3_result = await self._run_system_3_pivot(
                                    task_context=last_user_content,
                                    error_context=(last_error_res or '') + extra_context,
                                    sandbox_state=str(sandbox_state),
                                    model=model
                                )
                                if sys3_result.get("tree_update"):
                                    task_tree.load_from_json(sys3_result["tree_update"])
                                    current_plan_json = task_tree.to_json()
                                    execution_failure_count = max(0, execution_failure_count - 2)
                                    transient_failure_count = 0
                                    last_was_failure = False
                                    _request_sys3_fired_once = True
                                    _request_sys3_prev_justification = sys3_result.get('justification', '')
                                    messages.append({"role": "user", "content": f"SYSTEM 3 PIVOT #{pivot_num}: The previous approach failed. The strategy has been entirely rewritten. Justification: {sys3_result.get('justification')}. Follow the new plan."})
                                    continue

                            if execution_failure_count >= 6 or total_fail >= 8:
                                pretty_log("Loop Breaker", "Forcing final response", icon=Icons.STOP)
                                messages.append({"role": "user", "content": "SYSTEM ALERT: You have failed too many times. The task cannot be completed. Provide a final response explaining the situation."})
                                force_final_response = True
                        else:
                            # Only reset transient failures on success; structural
                            # failures require consecutive successes to decay.
                            transient_failure_count = 0
                            if execution_failure_count > 0:
                                execution_failure_count = max(0, execution_failure_count - 1)

                            # Terminal tools: `self_play` and `dream_mode`
                            # are self-contained cycles. Once they return,
                            # the expected next turn is a conversational
                            # summary to the user — NEVER another call.
                            # Without this gate, the main LLM (seeing the
                            # system prompt's "Call this tool EVERY TIME
                            # the user requests it" alongside the original
                            # user intent still in history) re-fires the
                            # same tool 2–3x per user ask and burns minutes
                            # per extra run. Setting force_final_response
                            # routes the next turn through the text-only
                            # streaming path; the tool_call suppressor
                            # above catches any stragglers in non-stream
                            # mode. The directive message is belt-and-
                            # suspenders for the model's benefit.
                            terminal_names = {"self_play", "dream_mode"}
                            just_ran_terminal = any(
                                t.get("name") in terminal_names
                                for t in tools_run_this_turn[-len(results):]
                            )
                            if just_ran_terminal and not force_final_response:
                                # --- DIRECT-FROM-TOOL SUMMARY ---
                                # We used to set force_final_response=True
                                # here and let the LLM summarise the tool
                                # result on a follow-up turn. That worked on
                                # the first invocation but deterministically
                                # failed from the second invocation onward:
                                # the conversation window now contained a
                                # repeated `user → <tool_call name="self_play">
                                # → tool result` pattern, and the model's
                                # attention attractor beat the "text-only"
                                # directive — it emitted yet another
                                # <tool_call> for the same tool, which the
                                # stream scrub stripped, which left the user
                                # with a generic fallback message instead of
                                # a real summary.
                                #
                                # The fix is to stop running the summary LLM
                                # turn at all. self_play / dream_mode already
                                # return a structured, user-readable string;
                                # we format it deterministically here and
                                # assign it straight to `final_ai_content`,
                                # then flip `force_stop` so the turn loop
                                # exits before any further LLM call runs.
                                # No LLM summary = no pattern priming = no
                                # tool_call leak = no scrub = no fallback.
                                _terminal_result = ""
                                _terminal_name = ""
                                for _t in reversed(tools_run_this_turn[-len(results):]):
                                    if _t.get("name") in terminal_names:
                                        _terminal_result = str(_t.get("content", "")).strip()
                                        _terminal_name = _t.get("name")
                                        break
                                # Distill the raw tool output into a short
                                # user-facing summary. The raw blob mixes
                                # user-relevant status with internal
                                # telemetry (`CURIOSITY: ...`) and an LLM-
                                # facing `SYSTEM INSTRUCTION:` trailer —
                                # the helper strips those and extracts
                                # the 1-3 key facts (cluster, status,
                                # skill-gate or learned lesson).
                                _summary_body = _distill_terminal_tool_summary(
                                    _terminal_name, _terminal_result
                                )
                                _prefix = {
                                    "self_play": "Self-play complete.",
                                    "dream_mode": "Dream cycle complete.",
                                }.get(_terminal_name, f"`{_terminal_name}` complete.")
                                final_ai_content = (
                                    f"{_prefix}\n\n{_summary_body}"
                                    if _summary_body else _prefix
                                )
                                force_stop = True
                                pretty_log(
                                    "Terminal Tool",
                                    f"Bypassed summary LLM — direct result displayed ({_terminal_name})",
                                    icon=Icons.STOP,
                                )
                                break  # exit the enumerate(results) loop

                # --- FINAL OUTPUT SCRUBBER ---
                # Apply scrubbers FIRST so we don't accidentally scrub our own manual fallback injections
                bleed_markers = [
                    "# Tools", "<tools>", "CRITICAL INSTRUCTION:", "You may call one or more functions",
                    '{"type": "function"', "SPECIALIST SUBSYSTEM ACTIVATED", "ENGINEERING STANDARDS",
                    "DYNAMIC SYSTEM STATE", "[SYSTEM STATE UPDATE]"
                ]
                for bleed_marker in bleed_markers:
                    if bleed_marker in final_ai_content:
                        final_ai_content = final_ai_content.split(bleed_marker)[0]

                # Last-resort scrub on final_ai_content. Must match the
                # widened mid-flow ui_content scrub (agent.py:~3149) shape
                # for shape — that path runs under `if has_tool_tag:` so
                # anything that bypasses it (perfect-it follow-up LLM call,
                # a fallback branch that writes straight to final_ai_content,
                # a turn where has_tool_tag was False but content still had
                # a bare <function>) reaches the user raw unless we also
                # strip it here. Three widenings vs. the old pattern:
                #   1. `function` added to the alternation — catches bare
                #      `<function name="...">...</function>` blocks the
                #      model sometimes emits without an outer <tool_call>
                #      wrapper (the exact shape the user reported as
                #      leaking verbatim after a successful self_play run).
                #   2. Backreference `\1` — the close tag must match the
                #      open tag type, so a nested `</function>` inside
                #      `<tool_call>...</tool_call>` can't terminate the
                #      outer match early and leave an orphan close tag.
                #   3. `[^>]*>` in the open tag tolerates attributes
                #      (`<tool_call name="...">`) without being fooled by
                #      a stray `>` in the body.
                final_ai_content = re.sub(
                    # `\Z` (absolute EOS) instead of `$` so trailing
                    # newlines after a tool_call don't escape the scrub.
                    r'<(tool_call|tool|function)\b[^>]*>.*?(?:</\1\b[^>]*>|\Z)',
                    '',
                    final_ai_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                final_ai_content = re.sub(r'<tool_response.*?>.*?(?:</tool_response.*?>|\Z)', '', final_ai_content, flags=re.DOTALL | re.IGNORECASE)
                final_ai_content = re.sub(r'--- EXECUTION RESULT ---.*?(?:------------------------|$)', '', final_ai_content, flags=re.DOTALL)
                final_ai_content = re.sub(r'(?m)^\s*(?:🔄|🟢|⏳|✅|❌|🛑|➖)\s*\[.*?\].*?\n?', '', final_ai_content)
                final_ai_content = re.sub(r'(?m)^.*?\((?:IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\)\s*\n?', '', final_ai_content)
                final_ai_content = re.sub(r'(?m)^\s*(?:\[)?task_\d+(?:\])?\s*\n?', '', final_ai_content)
                final_ai_content = re.sub(r'(?m)^\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\s*', '', final_ai_content)
                # Strip leaked Qwen Generative-Reward-Model JSON. These keys
                # are not produced anywhere in Ghost — verified via grep,
                # zero hits across the codebase. They are an upstream
                # training artifact: meta-prompts that ask the model to
                # rate or score itself pull it into evaluator mode and it
                # emits the GRM schema instead of an answer. Conservative
                # match requires both c_relevance_to_query AND
                # c_correctness_of_content, so a legitimate response that
                # happens to mention one key in isolation is not clobbered.
                final_ai_content = re.sub(
                    r'\{\{?\s*"c_relevance_to_query"\s*:\s*\d+\s*,\s*"c_correctness_of_content"\s*:.*?\}\}?',
                    '',
                    final_ai_content,
                    flags=re.DOTALL,
                )
                final_ai_content = final_ai_content.strip()

                # --- THE "PERFECT IT" PROTOCOL INJECTION ---
                # Only trigger proactive optimization for heavy engineering/research tasks
                heavy_tools_used = any(t.get('name') in ['execute', 'deep_research'] for t in tools_run_this_turn)

                # Skip the whole Perfect-It block during self-play: the
                # isolated context's `ReadOnlySkillMemory.learn_lesson`
                # is a no-op, so the write at the bottom lands in
                # /dev/null — but before we realised that, every single
                # self-play cycle was burning ~15s on a follow-up LLM
                # call to generate an "optimisation strategy" that was
                # silently discarded, and the misleading
                # `"Saved optimization strategy to playbook"` log line
                # was firing on the inner sub-agent's request ID
                # (production trace 16:36, request C6). The marker
                # lives on the ReadOnlySkillMemory class set up by
                # dream.py's isolation code. We check `is True`
                # explicitly (not `bool(...)`) so a MagicMock used in
                # production-agent unit tests doesn't accidentally
                # trigger the guard — MagicMock.is_read_only returns
                # another MagicMock (truthy), but not the sentinel.
                is_simulation = getattr(getattr(self.context, "skill_memory", None), "is_read_only", False) is True

                if not is_simulation and tools_run_this_turn and heavy_tools_used and execution_failure_count == 0 and not last_was_failure and (not final_ai_content or len(final_ai_content) < 50):
                    pretty_log("Perfect It Protocol", "Generating proactive optimization...", icon=Icons.IDEA)
                    perfect_it_prompt = f"Task completed successfully. Final tool output:\n\n{tools_run_this_turn[-1]['content']}\n\n<system_directive>First, succinctly present the tool output/result to the user. Then, based on your Perfection Protocol, analyze the result and proactively suggest one concrete way to optimize, scale, secure, or automate this work further. RESPOND IN PLAIN TEXT ONLY. DO NOT USE TOOLS.</system_directive>"
                    messages.append({"role": "user", "content": perfect_it_prompt})

                    p_req_messages = []
                    for m in messages:
                        if m.get("role") == "tool":
                            p_req_messages.append({"role": "user", "content": f"<tool_response name=\"{m.get('name', 'unknown')}\">\n{m.get('content')}\n</tool_response>"})
                        elif m.get("role") == "assistant":
                            p_req_messages.append({"role": "assistant", "content": m.get("content", "")})
                        else:
                            content_val = m.get("content", "")
                            if isinstance(content_val, list):
                                text_parts = []
                                for item in content_val:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            text_parts.append(item.get("text", ""))
                                        elif item.get("type") == "image_url":
                                            if bool(getattr(self.context.llm_client, 'vision_clients', None)):
                                                text_parts.append("[Image attached and passed to vision node]")
                                            else:
                                                text_parts.append(item) # Keep image dict
                                content_val = "\n".join(text_parts) if all(isinstance(x, str) for x in text_parts) else text_parts
                            p_req_messages.append({"role": m.get("role", "user"), "content": content_val})

                    payload["messages"] = p_req_messages

                    # 🔴 CRITICAL FIX: Physically remove tools from payload so it cannot hallucinate a tool call
                    if "tools" in payload: del payload["tools"]
                    if "tool_choice" in payload: del payload["tool_choice"]
                    payload["stream"] = False  # Prevent SSE streaming leak from the main loop

                    try:
                        perfection_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
                        p_msg = perfection_data["choices"][0]["message"].get("content", "")
                        p_msg = re.sub(r'<tool_call>.*?</tool_call>', '', p_msg, flags=re.DOTALL | re.IGNORECASE).strip()

                        # 1. User Display (Conditional on Flag)
                        if getattr(self.context.args, 'perfect_it', False):
                            if final_ai_content:
                                final_ai_content += "\n\n" + p_msg
                            else:
                                final_ai_content = p_msg

                        # 2. Internal Learning (Always)
                        if p_msg and getattr(self.context, 'skill_memory', None):
                            # Tag the lesson with this turn's trajectory
                            # id. If the user corrects on the next turn,
                            # `_maybe_promote_prior_turn_via_user_correction`
                            # promotes this trajectory to FAILED and
                            # calls `retract_lessons_from_trajectory`,
                            # which scrubs this exact entry from both
                            # the JSON playbook and the vector store.
                            # Without provenance, the rubber-stamped
                            # opt-prot lesson would survive the
                            # correction and poison future retrieval.
                            await asyncio.to_thread(
                                self.context.skill_memory.learn_lesson,
                                task=f"Optimization Analysis: {last_user_content[:50]}...",
                                mistake="Sub-optimal pattern identified via Perfection Protocol",
                                solution=p_msg,
                                memory_system=self.context.memory_system,
                                source_trajectory_id=current_trajectory_id,
                                source="perfection_protocol",
                            )
                            pretty_log("Internal Learning", "Saved optimization strategy to playbook.", icon=Icons.MEM_SAVE)

                    except Exception as e:
                        # Only report failure to user if they expected to see it
                        if getattr(self.context.args, 'perfect_it', False) and not final_ai_content:
                            final_ai_content = "Task finished successfully, but optimization generation failed."

                if tools_run_this_turn and not final_ai_content:
                    # Walk back to the last *real* tool output. Reading
                    # `tools_run_this_turn[-1]` blindly would wrap a
                    # synthetic agent-loop error ("SYSTEM ERROR: Your
                    # <tool_call> did not parse...") in a "Process
                    # finished successfully" banner — silent error to
                    # false-success leak.
                    fallback_tool = _find_substantive_tool_for_verifier(
                        tools_run_this_turn
                    )
                    if fallback_tool is not None:
                        last_out = str(fallback_tool.get('content', ''))

                        if "![Image]" in last_out:
                            final_ai_content = last_out.strip()
                        else:
                            # Extract just the pure STDOUT so the UI fallback is clean
                            if "STDOUT/STDERR:" in last_out:
                                last_out = last_out.split("STDOUT/STDERR:")[1].strip()
                                if "DIAGNOSTIC HINT" in last_out:
                                    last_out = last_out.split("DIAGNOSTIC HINT")[0].strip().strip("-").strip()

                            preview = (last_out[:2000] + '\n...[Truncated]') if len(last_out) > 2000 else last_out
                            final_ai_content = f"Process finished successfully.\n\n### Final Output:\n```text\n{preview}\n```"

                if not final_ai_content:
                    final_ai_content = "Task executed successfully."

                # --- VERIFIER GATE (reflection before finalising) ---
                # Run the claim/output verifier on the final answer when the
                # turn produced tool output — catches silent-error answers
                # ("Process finished successfully" when the tool actually
                # failed), wrong-unit answers, and empty results. The module
                # was orphaned (initialised but never called) before this
                # wiring. Skipped for pure-conversational turns (no tools
                # run) and when no verifier is installed.
                #
                # Failure mode on REFUTED: we append an auditor note to the
                # reply rather than silently regenerating. This keeps the
                # user in the loop (they can see what the verifier disagreed
                # with) without doubling latency by rerunning the whole turn.
                # Selfhood outcome backfill (proposal item #3): the verifier
                # verdict computed below is the first reliable signal of
                # whether this turn actually succeeded. Captured here and
                # applied AFTER `_record_turn_trajectory` writes the
                # autobiographical record (which is born `outcome="unknown"`
                # on the hot path). `(outcome, failure_reason)` or None.
                verifier_backfill: tuple | None = None
                try:
                    verifier = getattr(self.context, "verifier", None)
                    # Gate: any tool-using turn is worth verifying. The old
                    # `was_complex_task` constraint (turn > 2) silently
                    # skipped the common 1–2 turn tool path, so the verifier
                    # almost never fired. `tools_run_this_turn` alone is
                    # sufficient — trivial greetings already have no tools.
                    last_tool = _find_substantive_tool_for_verifier(
                        tools_run_this_turn
                    )
                    # Use the STRICT trivial-chat check (allowlist of
                    # actual greetings like "hi", "thanks") rather than
                    # the loose `is_trivial_greeting` flag. The loose
                    # flag fires on any 5-word conversational message
                    # without "remember"/"previous" — including
                    # correction-shaped prompts ("thanks but wrong",
                    # "no try again", "ok do it again") which are
                    # exactly the highest-leverage place to verify.
                    # The fast-path at the top of handle_chat already
                    # uses `_is_strict_trivial_chat` to decide whether
                    # to bypass the full turn loop; mirror that gate
                    # here so any prompt that DID run the full loop
                    # (and produced tool output) gets verified.
                    if (
                        verifier is not None
                        and verifier.llm_client is not None
                        and last_tool is not None
                        and final_ai_content
                        and not self._is_strict_trivial_chat(lc)
                    ):
                        tool_output = str(last_tool.get("content", ""))[:4000]
                        tool_name = str(last_tool.get("name", ""))[:80]
                        if "execute" in tool_name.lower() or "postgres" in tool_name.lower():
                            # Recover the code that was actually executed.
                            # Previously this slot held `tool_name` ("execute"),
                            # which the verifier couldn't audit — it
                            # hallucinated reasons the output didn't
                            # match. `_reconstruct_executed_code` walks
                            # messages → assistant → tool_calls[tool_id]
                            # and returns the content/code/command arg.
                            code_text = _reconstruct_executed_code(messages, last_tool)
                            if code_text:
                                v_result = await verifier.verify_code_output(
                                    code=code_text,
                                    output=tool_output,
                                    intent=last_user_content or "",
                                    # Pass the agent's user-facing reply so
                                    # the verifier can audit whether the
                                    # RESPONSE matches the user's request,
                                    # not just whether the tool output
                                    # matches the agent's printed claim.
                                    # Catches the "user asked for code,
                                    # agent gave a number" failure shape.
                                    response=final_ai_content or "",
                                )
                            else:
                                # Couldn't recover the submitted code —
                                # fall back to claim-shape verification,
                                # which doesn't need a code slot.
                                v_result = await verifier.verify_claim(
                                    claim=final_ai_content[:2000],
                                    evidence=tool_output,
                                    context=(last_user_content or "")[:1000],
                                )
                        else:
                            v_result = await verifier.verify_claim(
                                claim=final_ai_content[:2000],
                                evidence=tool_output,
                                context=(last_user_content or "")[:1000],
                            )
                        from .verifier import VerifyVerdict
                        # Record the verdict for the selfhood backfill that
                        # runs after the autobiographical record is written.
                        if v_result and v_result.confidence >= 0.7:
                            if v_result.verdict == VerifyVerdict.CONFIRMED:
                                verifier_backfill = ("passed", "")
                            elif v_result.verdict == VerifyVerdict.REFUTED:
                                _vr_reason = (
                                    "; ".join(v_result.issues[:2])
                                    if v_result.issues else v_result.reasoning
                                )
                                verifier_backfill = ("failed", _vr_reason or "")
                        if v_result and v_result.verdict == VerifyVerdict.REFUTED and v_result.confidence >= 0.7:
                            issues_str = "; ".join(v_result.issues[:3]) if v_result.issues else v_result.reasoning
                            note = f"\n\n---\n**Verifier note:** {issues_str}"
                            if note[:60] not in final_ai_content:
                                final_ai_content = f"{final_ai_content}{note}"
                            pretty_log(
                                "Verifier",
                                f"REFUTED ({v_result.confidence:.0%}): {issues_str[:120]}",
                                icon=Icons.BRAIN_THINK,
                            )
                            # Verifier-driven retraction: the
                            # Perfection-Protocol's `learn_lesson`
                            # runs BEFORE this gate, so by the time
                            # we get here a poisoned lesson tagged
                            # with `current_trajectory_id` may
                            # already be on disk. The verifier just
                            # said the response was REFUTED with
                            # high confidence — scrub anything that
                            # turn produced before the user even
                            # sees the response, so the next user
                            # query can't retrieve a lesson born
                            # from a turn we just disagreed with.
                            # Belt-and-braces with the user-
                            # correction retraction path: the user
                            # may never go on to correct, and we
                            # shouldn't depend on them noticing.
                            try:
                                _sm = getattr(self.context, "skill_memory", None)
                                if _sm is not None and current_trajectory_id:
                                    _sm.retract_lessons_from_trajectory(
                                        current_trajectory_id,
                                        memory_system=getattr(
                                            self.context, "memory_system", None
                                        ),
                                    )
                            except Exception as _e:
                                logger.debug(
                                    "verifier-driven retraction skipped: %s: %s",
                                    type(_e).__name__, _e,
                                )
                        else:
                            pretty_log(
                                "Verifier",
                                f"{v_result.verdict.value} ({v_result.confidence:.0%})" if v_result else "skipped",
                                icon=Icons.OK,
                            )
                except Exception as e:
                    logger.debug(f"Verifier gate skipped: {e}")

                # Plan postcondition gate (proposal item #10). If the
                # strategic planner produced a plan whose ROOT task
                # declared postconditions, hold the final response to
                # them — the plan stops being internal bookkeeping and
                # becomes a success contract on the answer itself. A
                # response that misses a declared postcondition gets a
                # visible note rather than passing silently. Non-fatal.
                try:
                    _plan_json = locals().get('current_plan_json')
                    if _plan_json and final_ai_content:
                        from .planning import TaskTree as _PlanTree
                        _ptree = _PlanTree()
                        _ptree.load_from_json(_plan_json)
                        _unsat = _ptree.root_postconditions_unsatisfied(
                            final_ai_content
                        )
                        if _unsat:
                            _pnote = (
                                "**Plan check:** this response may not yet "
                                "satisfy: " + "; ".join(_unsat[:3])
                            )
                            if _pnote[:48] not in final_ai_content:
                                final_ai_content = (
                                    f"{final_ai_content}\n\n---\n{_pnote}"
                                )
                            pretty_log(
                                "Plan Gate",
                                f"{len(_unsat)} unsatisfied postcondition(s)",
                                icon=Icons.BRAIN_PLAN,
                            )
                except Exception as e:
                    logger.debug(f"plan postcondition gate skipped: {e}")

                # --- AUTOMATED POST-MORTEM (AUTO-LEARNING) ---
                if was_complex_task or execution_failure_count > 0:
                    is_complete_failure = (execution_failure_count >= 3)
                    is_valid_success = (not force_stop or "READY TO FINALIZE" in thought_content.upper())

                    if is_valid_success or is_complete_failure:
                        # Gated on `--smart-memory > 0.0` to honour the
                        # contract in CLAUDE.md ("Memory writes are gated
                        # on --smart-memory / --no-memory"). The streaming
                        # producer has the same gate; both must agree.
                        if getattr(self.context, 'journal', None) and self.context.args.smart_memory > 0.0:

                            await asyncio.to_thread(self.context.journal.append, 'post_mortem', {'user': last_user_content, 'tools': list(tools_run_this_turn), 'ai': final_ai_content, 'model': model})

                # Retrieval feedback: a clean, non-failing turn credits
                # any lesson surfaced in the last ~5min. Hoisted out of
                # the `was_complex_task` gate above — simple successful
                # turns are exactly where retrieved lessons are most
                # likely to be paying off, and excluding them biased
                # the utility signal toward complex tasks only.
                # `credit_recent_retrievals` is idempotent and no-ops
                # when nothing was retrieved, so running it on every
                # clean-exit turn is safe.
                sm = getattr(self.context, 'skill_memory', None)
                if sm is not None and execution_failure_count == 0:
                    try:
                        if hasattr(sm, 'credit_recent_retrievals'):
                            await asyncio.to_thread(sm.credit_recent_retrievals, 300)
                    except Exception:
                        pass

                # Surface critical unknowns/assumptions tracked during this
                # turn. The UncertaintyTracker is a shared per-process
                # instance — appending the risk summary lets the user see
                # what the agent is uncertain about WITHOUT the LLM having
                # to remember to mention it. Skipped silently when the
                # tracker isn't wired or has no risks.
                try:
                    tracker = getattr(self.context, 'uncertainty_tracker', None)
                    if tracker is not None:
                        # Auto-populate from the agent's own output: any
                        # explicit first-person hedge ("I'm assuming…",
                        # "I couldn't verify…") becomes a tracked, persisted
                        # assumption — so the tracker is load-bearing even
                        # when the LLM never calls flag_uncertainty.
                        try:
                            for _hedge in tracker.scan_text_for_uncertainty(
                                final_ai_content or ""
                            ):
                                tracker.flag_assumption(
                                    _hedge, confidence=0.4,
                                    basis="auto-detected hedge in response",
                                )
                        except Exception:
                            pass
                        # Metacognitive gate (proposal item #6): if a
                        # critical unknown still needs the user, surface
                        # the clarifying question at the TOP of the reply
                        # — a real gate on the response, not a footer.
                        try:
                            question = tracker.should_ask_user()
                        except Exception:
                            question = None
                        if (question and final_ai_content
                                and question[:40] not in final_ai_content):
                            final_ai_content = (
                                f"**{question}**\n\n"
                                f"(Answering with my current understanding below "
                                f"— correct me if that clarification changes "
                                f"things.)\n\n---\n\n{final_ai_content}"
                            )
                        risk = tracker.get_risk_summary()
                        if risk and final_ai_content and risk[:60] not in final_ai_content:
                            final_ai_content = f"{final_ai_content}\n\n---\n{risk}"
                        # Reset in-memory turn state; the durable persisted
                        # log is untouched so recurring blind-spots survive.
                        tracker.reset()
                except Exception as e:
                    logger.debug(f"Uncertainty surfacing skipped: {e}")

                # Sync the locally-evolved `messages` list back to the caller's
                # `body`. `messages` was rebound (by `process_rolling_window`,
                # `_prune_context`, and recovery paths) into a new list object
                # partway through the run, so subsequent tool-call / tool-
                # result appends only landed on the local copy. Callers that
                # inspect `body["messages"]` after `handle_chat` (the Slack
                # bot, the web UI, and several regression tests) expect to
                # see the full trace, including tool calls and results.
                try:
                    body["messages"] = messages
                except Exception:
                    pass

                # Stage-1 self-improvement: append the turn's trajectory
                # to the distill log. No-op when the collector isn't
                # wired. Deliberately non-fatal — trajectory logging
                # must never break a user turn.
                try:
                    self._record_turn_trajectory(
                        messages=messages,
                        final_content=final_ai_content,
                        req_id=req_id,
                        model=model,
                        trajectory_id=current_trajectory_id,
                    )
                except Exception as e:
                    # Debug-level: a turn-logging failure must never be
                    # noisy in production. When diagnosing, bump to
                    # warning temporarily.
                    logger.debug(f"trajectory logging skipped: {type(e).__name__}: {e}")

                # Selfhood outcome backfill (proposal item #3): the
                # autobiographical record was just written `outcome=
                # "unknown"` by `_record_turn_trajectory`. If the verifier
                # produced a high-confidence verdict, propagate it into
                # the agent's self-memory so its recall of its own past
                # is verdict-aware. Non-fatal — backfill is secondary.
                if verifier_backfill is not None and current_trajectory_id:
                    try:
                        from ..selfhood import SelfModel as _SelfModelBF
                        _sm_bf = getattr(self.context, 'self_model', None)
                        if isinstance(_sm_bf, _SelfModelBF) and getattr(_sm_bf, 'enabled', False):
                            _bf_outcome, _bf_reason = verifier_backfill
                            _sm_bf.record_outcome(
                                current_trajectory_id, _bf_outcome,
                                failure_reason=_bf_reason,
                            )
                    except Exception as e:
                        logger.debug(
                            "selfhood outcome backfill skipped: %s: %s",
                            type(e).__name__, e,
                        )

                return final_ai_content, created_time, req_id

        finally:
            if 'messages' in locals(): del messages
            if 'tools_run_this_turn' in locals(): del tools_run_this_turn
            if 'sandbox_state' in locals(): del sandbox_state
            if 'data' in locals(): del data

            pretty_log("Request Finished", special_marker="END")
            request_id_context.reset(token)

    @staticmethod
    def _response_fingerprint(text: str) -> str:
        """Stable short fingerprint of an assistant response, used as
        the lookup key for correction-detection. First 500 chars of
        the response, lowercased and whitespace-collapsed, then
        md5-hashed. Empty input → empty string (a never-matched key).

        We hash the prefix, not the full response, because Slack /
        the web UI sometimes append small footers (timestamps,
        emoji status) to the assistant message that survive the
        round-trip back into the next request's `messages`. Matching
        on a stable prefix tolerates that without false collisions
        for distinct responses (md5 keeps collision risk negligible
        at this scale)."""
        if not isinstance(text, str) or not text:
            return ""
        import hashlib
        norm = re.sub(r"\s+", " ", text[:500]).strip().lower()
        if not norm:
            return ""
        return hashlib.md5(norm.encode("utf-8")).hexdigest()

    def _stash_trajectory_for_correction_lookup(self, traj) -> None:
        """Cache ``traj`` keyed by its ``final_response`` fingerprint
        so the next user turn can locate it via ``messages[-2]``.

        Bounded LRU on ``self.context._recent_trajectories_for_correction``
        (max 32 entries) — enough headroom for a few concurrent
        conversations on a single Ghost process without unbounded
        growth. The eviction path is intentionally O(1): we use an
        OrderedDict and ``popitem(last=False)`` to drop the oldest
        entry when the cap is reached."""
        if not getattr(traj, "final_response", "") or not traj.id:
            return
        from collections import OrderedDict
        cache = getattr(self.context, "_recent_trajectories_for_correction", None)
        if cache is None:
            cache = OrderedDict()
            self.context._recent_trajectories_for_correction = cache
        fp = self._response_fingerprint(traj.final_response)
        if not fp:
            return
        if fp in cache:
            cache.pop(fp)
        cache[fp] = traj
        while len(cache) > 32:
            cache.popitem(last=False)

    def _maybe_promote_prior_turn_via_user_correction(
        self,
        messages,
        current_user_text: str,
    ) -> None:
        """Stage-1 self-improvement: promote the immediately-prior
        assistant trajectory to FAILED when the current user message
        looks like a correction.

        Why this exists: the biological watchdog's reflection phase
        only fires after 15-60 minutes of idle, which means a
        back-to-back interactive session can never close the
        learning loop on its own. Treating the user's next message
        as the FAILED label and reflecting immediately is what makes
        the loop usable in real chat.

        Conditions (all must hold; on any miss this is a no-op):
          - ``ctx.trajectory_collector`` is wired
          - we have a stashed trajectory whose response fingerprint
            matches the previous assistant message in ``messages``
          - ``classify_user_correction`` returns ``is_correction=True``

        Side effects when promotion fires:
          1. The trajectory's in-memory ``outcome`` /
             ``failure_reason`` are mutated (so the immediate
             ``reflect_one`` call below sees the corrected state).
          2. A correction record is appended to the collector's
             sidecar (durable; future ``iter_trajectories`` walks
             will yield the corrected outcome).
          3. If a Reflector is wired, ``reflect_one`` is scheduled
             via ``asyncio.create_task`` — fire-and-forget. The
             user's turn doesn't block on it; the lesson lands when
             the LLM critique returns, typically before the user
             types their next message.

        Never raises — wrapped in try/except by the caller, but the
        helper itself swallows individual sub-errors so a failed
        sidecar write doesn't prevent the in-memory promotion etc."""
        ctx = self.context
        collector = getattr(ctx, "trajectory_collector", None)
        if collector is None:
            return

        cache = getattr(ctx, "_recent_trajectories_for_correction", None)
        if not cache:
            return

        # Walk messages[:-1] (everything before the current user turn)
        # in reverse to find the most recent assistant + user contents.
        prev_assistant = ""
        prev_user = ""
        msgs = list(messages or [])
        scope = msgs[:-1] if msgs else []
        for m in reversed(scope):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    i.get("text", "") for i in content
                    if isinstance(i, dict) and i.get("type") == "text"
                )
            content = str(content or "")
            if role == "assistant" and not prev_assistant:
                prev_assistant = content
            elif role == "user" and not prev_user:
                prev_user = content
            if prev_assistant and prev_user:
                break

        if not prev_assistant:
            return

        fp = self._response_fingerprint(prev_assistant)
        if not fp:
            return
        traj = cache.get(fp)
        if traj is None:
            return

        try:
            from ..distill.user_correction import classify_user_correction
            from ..distill.schema import Outcome
        except Exception:
            return

        verdict = classify_user_correction(
            prev_user_request=prev_user,
            prev_assistant_response=prev_assistant,
            current_user_text=current_user_text or "",
        )
        if not verdict.is_correction:
            return

        # Persist the correction. The sidecar is durable; the in-
        # memory mutation lets the immediate reflect_one() call see
        # the corrected outcome + failure_reason without re-reading
        # disk.
        if traj.outcome != Outcome.FAILED.value:
            traj.outcome = Outcome.FAILED.value
            traj.failure_reason = verdict.reason or "user-correction"
            try:
                collector.update_outcome(
                    traj.id,
                    Outcome.FAILED.value,
                    reason=verdict.reason or "user-correction",
                    source="user_correction",
                )
            except Exception as e:
                logger.debug(
                    "user-correction sidecar write failed: %s: %s",
                    type(e).__name__, e,
                )
            try:
                pretty_log(
                    "Trajectory Promoted",
                    f"prior turn marked FAILED via user-correction "
                    f"(confidence={verdict.confidence:.2f}, "
                    f"signals={','.join(verdict.signals) or 'none'})",
                    icon=Icons.BRAIN_THINK,
                )
            except Exception:
                pass

        # One promotion per stashed trajectory: drop the entry so a
        # follow-up user message that's also "correction-shaped"
        # doesn't re-fire on the same prior turn.
        cache.pop(fp, None)

        # Scrub any lessons the just-failed trajectory produced
        # before the user could push back. The dominant case is the
        # Perfection-Protocol writing an "Optimization Analysis"
        # lesson at end-of-turn from a turn the user is now
        # correcting — without retraction, that poisoned lesson
        # would survive in the playbook and surface on future
        # similar queries. Best-effort: a failed retraction must
        # not block the rest of the promotion path. The reflection
        # task scheduled below will write a CORRECT lesson tagged
        # with the reflection trajectory's id, so the playbook ends
        # up with the right entry rather than both.
        skill_memory = getattr(ctx, "skill_memory", None)
        vector_memory = getattr(ctx, "memory_system", None)
        if skill_memory is not None and traj.id:
            try:
                skill_memory.retract_lessons_from_trajectory(
                    traj.id,
                    memory_system=vector_memory,
                )
            except Exception as e:
                logger.debug(
                    "lesson retraction skipped: %s: %s",
                    type(e).__name__, e,
                )

        # Schedule single-trajectory reflection. Fire-and-forget; the
        # current user turn must not block on the LLM critique.
        reflector = getattr(ctx, "reflector", None)
        if reflector is None:
            return
        sink = getattr(ctx, "reflection_sink", None) or collector.append
        already = getattr(ctx, "_reflected_trajectory_ids", None)
        if already is None:
            already = set()
            ctx._reflected_trajectory_ids = already

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            task = loop.create_task(
                reflector.reflect_one(
                    traj,
                    sink=sink,
                    already_reflected=already,
                )
            )
        except Exception as e:
            logger.debug(
                "post-turn reflect_one schedule failed: %s: %s",
                type(e).__name__, e,
            )
            return

        pending = getattr(ctx, "_pending_reflection_tasks", None)
        if pending is None:
            pending = set()
            ctx._pending_reflection_tasks = pending
        pending.add(task)
        task.add_done_callback(pending.discard)

        # Observability: reflection runs as a fire-and-forget task —
        # without a done-callback, a critique timeout or unparseable
        # response is silent. Log the outcome at INFO so a tail of
        # the agent log makes the post-turn reflection path visible
        # without pulling --debug. Errors stay non-fatal: the task's
        # failure mode is "we don't get a lesson this turn", not
        # "the user turn breaks".
        traj_id_for_log = traj.id

        def _log_post_turn_reflection_result(t):
            try:
                if t.cancelled():
                    pretty_log(
                        "Post-Turn Reflection",
                        f"cancelled (traj={traj_id_for_log[:8]})",
                        icon=Icons.WARN,
                    )
                    return
                exc = t.exception()
                if exc is not None:
                    pretty_log(
                        "Post-Turn Reflection",
                        f"failed (traj={traj_id_for_log[:8]}): "
                        f"{type(exc).__name__}: {exc}",
                        icon=Icons.WARN,
                        level="WARNING",
                    )
                    return
                outcome = t.result()
                if outcome is None:
                    return
                if getattr(outcome, "ok", False):
                    pretty_log(
                        "Post-Turn Reflection",
                        f"ok (traj={traj_id_for_log[:8]}): "
                        f"diagnosis={outcome.diagnosis[:80]!r}",
                        icon=Icons.BRAIN_THINK,
                    )
                else:
                    pretty_log(
                        "Post-Turn Reflection",
                        f"no lesson (traj={traj_id_for_log[:8]}): "
                        f"{outcome.error or 'unknown'}",
                        icon=Icons.WARN,
                    )
            except Exception:
                # The done-callback itself must never raise — that
                # would propagate into the event loop's exception
                # handler and surface as noise.
                pass

        task.add_done_callback(_log_post_turn_reflection_result)

    def _record_turn_trajectory(
        self,
        *,
        messages,
        final_content,
        req_id: str,
        model: str,
        trajectory_id: str = "",
    ) -> None:
        """Build and persist a Trajectory for the turn that just finished.

        No-op when `ctx.trajectory_collector` isn't wired. Walks the
        final `messages` list to reconstruct the tool-call sequence
        (each assistant message with `tool_calls` is paired with the
        matching `role=tool` responses that follow it in list order).

        All free-text fields go through the collector's redactor before
        they hit disk; this method only assembles the Trajectory.
        """
        collector = getattr(self.context, "trajectory_collector", None)
        if collector is None:
            return

        # Lazy import so modules that don't wire distill stay clean.
        from ..distill.schema import Trajectory, ToolCall, Outcome

        msgs = list(messages or [])
        system_prompt = ""
        user_request = ""
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    i.get("text", "") for i in content
                    if isinstance(i, dict) and i.get("type") == "text"
                )
            if role == "system" and not system_prompt:
                system_prompt = str(content or "")
            elif role == "user":
                user_request = str(content or "")

        # Reconstruct tool call pairs. Walk the messages left-to-right;
        # every assistant with tool_calls gets paired with the tool
        # responses that follow it (by tool_call_id when present).
        tool_calls = []
        pending_calls = {}  # id -> (name, arguments)
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "assistant":
                for tc in (m.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or tc.get("name") or "").strip()
                    args_raw = fn.get("arguments")
                    args: dict = {}
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            args = {"_raw": args_raw[:500]}
                    elif isinstance(args_raw, dict):
                        args = args_raw
                    pending_calls[tc.get("id") or name] = (name, args, ToolCall(name=name, arguments=args))
                    tool_calls.append(pending_calls[tc.get("id") or name][2])
            elif role == "tool":
                tc_id = m.get("tool_call_id") or m.get("name")
                entry = pending_calls.get(tc_id)
                if entry is not None:
                    _n, _a, obj = entry
                    obj.result = str(m.get("content") or "")[:4000]

        # Final response content. Non-string (streaming generator) is
        # logged as empty — the stream path gets richer tool-call data
        # but the final text lives in the SSE frames, not here.
        final_response = final_content if isinstance(final_content, str) else ""

        traj_kwargs = dict(
            session_id=req_id or "",
            task_kind="user_request",
            cluster=None,
            tier=None,
            model=str(model or ""),
            system_prompt=system_prompt[:8000],
            user_request=user_request[:8000],
            tool_calls=tool_calls,
            n_steps=sum(
                1 for m in msgs
                if isinstance(m, dict) and m.get("role") == "assistant"
            ),
            outcome=Outcome.UNKNOWN.value,  # user turns have no validator
            final_response=final_response[:16000],
        )
        # Use the pre-allocated id from `handle_chat` when present so
        # in-turn writers (Perfection-Protocol's lesson save) and the
        # eventual on-disk record share one stable id. Falls back to
        # the Trajectory dataclass's uuid factory when the caller
        # didn't pre-allocate (legacy callers, isolated unit tests).
        if trajectory_id:
            traj_kwargs["id"] = trajectory_id
        traj = Trajectory(**traj_kwargs)
        # Stage-1 self-improvement: chat trajectories ship with
        # outcome=UNKNOWN (no validator on free-form chat), which
        # excludes them from the Reflector's input set — only FAILED
        # trajectories get reflected. The result is that interactive-
        # session failures (selector thrashing, repeated tool errors,
        # runtime-aborted attempts) never produced lessons.
        # ``apply_chat_outcome_heuristics`` looks at the just-recorded
        # turn's shape and promotes UNKNOWN → FAILED when the signals
        # are unambiguous (runtime abort markers, selector thrashing,
        # repeated identical tool errors, aborted browser sequences).
        # The classifier is conservative — calibrated against the
        # 2026-04-26 webOS incident — so a normal exploratory turn is
        # NOT promoted. See distill/outcome_heuristics.py for the
        # signal list and thresholds. Failure here must not break the
        # turn: if classification raises, log and continue with the
        # original outcome.
        try:
            from ..distill.outcome_heuristics import apply_chat_outcome_heuristics
            if apply_chat_outcome_heuristics(traj):
                logger.debug(
                    "trajectory promoted UNKNOWN→FAILED: %s",
                    traj.failure_reason,
                )
        except Exception as e:
            logger.debug(
                "outcome heuristics skipped: %s: %s",
                type(e).__name__, e,
            )
        collector.append(traj)
        # Stage-1 self-improvement: cache the just-recorded trajectory
        # keyed by its response fingerprint so the NEXT user message's
        # correction-classifier hook can locate it via `messages[-2]`.
        # Non-fatal — the cache is an optimization for the post-turn
        # reflection path, not a primary persistence layer.
        try:
            self._stash_trajectory_for_correction_lookup(traj)
        except Exception as e:
            logger.debug(
                "trajectory stash for correction lookup skipped: %s: %s",
                type(e).__name__, e,
            )

        # Selfhood capture (proposal item #1 + #2): write a first-person
        # experiential record sharing the trajectory id. Distinct from
        # the trajectory log (structured tool trace, ML-training shape);
        # this is the agent's own first-person diary entry, tagged
        # subject="self" so the recognition layer treats it as "mine".
        # Non-fatal — selfhood capture is secondary to the user turn.
        # Same MagicMock-resilient isinstance check as the wake-up
        # prefix path.
        try:
            from ..selfhood import SelfModel as _SelfModel
            self_model = getattr(self.context, 'self_model', None)
            if isinstance(self_model, _SelfModel) and getattr(self_model, 'enabled', False):
                self_model.capture_turn(
                    trajectory_id=traj.id,
                    user_request=traj.user_request,
                    tool_names=[tc.name for tc in traj.tool_calls],
                    outcome=traj.outcome,
                    final_response=traj.final_response,
                    failure_reason=traj.failure_reason,
                    cluster=traj.cluster,
                )
        except Exception as e:
            logger.debug(
                "selfhood capture skipped: %s: %s",
                type(e).__name__, e,
            )

    async def _run_system_3_pivot(self, task_context: str, error_context: str, sandbox_state: str, model: str) -> dict:
        """System 3 Crisis Pivot: generate 3 alternative strategies and pick the safest one."""
        try:
            # Deep-reason prelude: when --deep-reason is on, generate and
            # narrow root-cause hypotheses BEFORE drafting strategies. Feeding
            # the pruned hypothesis list into the strategy generator produces
            # more targeted strategies instead of generic "try harder" ones.
            # Gated through ``context.hypothesis_tester`` so it's a no-op when
            # the flag isn't set (module stays in "gated" state).
            hypotheses_hint = ""
            tester = getattr(self.context, "hypothesis_tester", None)
            if tester is not None:
                try:
                    hypotheses = await tester.generate_hypotheses(
                        problem=task_context,
                        context=sandbox_state,
                        error_output=error_context,
                    )
                    if hypotheses:
                        top = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
                        hypotheses_hint = "### CANDIDATE ROOT CAUSES (from parallel hypothesis gen):\n" + "\n".join(
                            f"- [conf {h.confidence:.0%}] {h.description}" for h in top
                        ) + "\n\n"
                        pretty_log(
                            "Deep Reason",
                            f"Generated {len(hypotheses)} hypotheses; top-3 informing strategy gen",
                            icon=Icons.BRAIN_THINK,
                        )
                except Exception as e:
                    logger.debug(f"Hypothesis prelude skipped: {e}")

            # Step 1: Generate 3 distinct strategies
            pretty_log("System 3 Generator", "Analysing failure and generating alternative strategies...", icon=Icons.BRAIN_THINK)
            gen_user_msg = (
                f"### TASK CONTEXT:\n{task_context}\n\n"
                f"{hypotheses_hint}"
                f"### ERROR CONTEXT (what failed and why):\n{error_context[:3000]}\n\n"
                f"### CURRENT SANDBOX STATE:\n{sandbox_state[:1500]}"
            )
            gen_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_3_GENERATION_PROMPT},
                    {"role": "user", "content": gen_user_msg}
                ],
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            }
            use_swarm = bool(getattr(self.context.llm_client, 'swarm_clients', None))
            gen_data = await self.context.llm_client.chat_completion(gen_payload, use_swarm=use_swarm)
            gen_content = gen_data["choices"][0]["message"].get("content", "")
            strategies_json = extract_json_from_text(gen_content)
            strategies = strategies_json.get("strategies", [])
            if not strategies:
                logger.warning("System 3 Generator returned no strategies.")
                return {}

            # Step 2: Evaluate and pick the safest strategy
            pretty_log("System 3 Evaluator", "Selecting safest recovery path...", icon=Icons.BRAIN_THINK)
            eval_user_msg = (
                f"### PROPOSED STRATEGIES:\n{gen_content}\n\n"
                f"### CURRENT SANDBOX STATE:\n{sandbox_state[:1500]}"
            )
            eval_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_3_EVALUATOR_PROMPT},
                    {"role": "user", "content": eval_user_msg}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            eval_data = await self.context.llm_client.chat_completion(eval_payload, use_swarm=use_swarm)
            eval_content = str(eval_data["choices"][0]["message"].get("content") or "")
            result = extract_json_from_text(eval_content)
            pretty_log("System 3 Complete", f"Winning strategy: {result.get('winning_id', '?')} — {result.get('justification', '')[:120]}", icon=Icons.BRAIN_THINK)
            return result
        except Exception as e:
            logger.error(f"System 3 pivot failed: {e}")
            return {}

