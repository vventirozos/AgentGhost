# src/ghost_agent/core/agent.py

import asyncio
import datetime
import hashlib
import json
import logging
import os
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
from dataclasses import dataclass
from pathlib import Path

from .prompts import SYSTEM_PROMPT, SPECIALIST_SYSTEM_PROMPT, SMART_MEMORY_PROMPT, PLANNING_SYSTEM_PROMPT, SYSTEM_3_GENERATION_PROMPT, SYSTEM_3_EVALUATOR_PROMPT, THINK_BUDGET_TIGHT, THINK_BUDGET_EXTENDED
from .planning import TaskTree, TaskStatus
# Shared "what did this call operate on" helper — same definition the
# offline post-mortem signature uses, so the in-run no-progress
# loop-breaker and the post-mortem read-loop detector agree on what
# "the same target" means.
from ..reflection.postmortem import primary_target_from_args
from .triggers import RecentFailureGuard
from ..utils.logging import Icons, pretty_log, request_id_context, atomic_print
from ..utils import logging as _glog
from ..utils.constraints import extract_constraints, render_constraint_block

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
# FACTUAL / DETERMINISTIC answer turn — a non-tool turn whose query has a single
# correct answer (math, counting, lookup, "what is X"). The conversational
# profile (temp=1.0, presence_penalty=1.5) is actively harmful here: high temp
# injects answer variance and presence_penalty=1.5 *discourages re-using the
# exact tokens that ARE the answer* (the number / name). Near-greedy with no
# presence penalty samples toward correctness. See `_is_factual_query`.
FACTUAL_SAMPLING_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 0,
}

# ---------------------------------------------------------------------------
# Cognitive-layer toggles (2026-06 redesign — see PROJECT_JOURNAL.md §3).
# A paired ablation showed the full cognitive stack did NOT beat a stripped
# baseline on a hard graded suite (78% vs 80%, p=1.0) at ~1.8x latency. The
# offending pieces were "advisory, not load-bearing" injections and ungrounded
# search. These flags DEFAULT-OFF the pieces that cost latency without changing
# outcomes; the grounded paths (post-failure hypothesis testing, verifier-judged
# best-of-N, signature-keyed lesson recall) are wired live below.
#
#  * MCTS turn-start hint: an MCTS "next action" injected as a prompt block
#    ending "a strong hint, not a mandate — deviate if needed" → the model
#    regenerates freely and ignores it, while the search costs ~4 LLM round
#    trips (the 390s blowups). Ungrounded (scores the model's self-prediction
#    of un-executed actions). OFF until it has an execution-grounded value fn.
_MCTS_TURNSTART_ENABLED = False
#  * Selfhood wake-up prefix: first-person narrative/mood prose spliced into the
#    system prompt every turn. Injects no facts/tools/constraints — cosmetic
#    voice that only adds tokens/latency. OFF on the request path.
_SELFHOOD_PREFIX_ENABLED = False
#  * Grounded hypothesis testing: on the post-failure replan path, don't just
#    LIST candidate root causes — run each hypothesis's minimal test in the
#    sandbox and keep only those consistent with the evidence. This gives
#    deep-reason a value function grounded in real execution (the one mechanism
#    that can convert a failure into a success). ON by default (no-op without
#    --deep-reason, which is what wires context.hypothesis_tester).
_HYPOTHESIS_GROUNDING_ENABLED = True
#  * Metacog dual-solver arbiter: on a low-confidence shell/sql turn it samples
#    TWO paraphrased completions, computes their cosine divergence, then throws
#    both candidates away and dispatches the ORIGINAL args — paying 2x inference
#    for a number that gates a near-unreachable `ask_user` pause. Net-negative
#    (the dominant latency source; zero answer changes). OFF. The intended
#    grounded replacement is a verifier-judged best-of-N that SUBSTITUTES the
#    winning answer (see PROJECT_JOURNAL.md §3, deferred #4) — a finalize-path
#    feature, tracked separately.
_METACOG_ARBITER_ENABLED = False

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

# Adaptive sampling profiles for coding sub-tasks. The classifier + routing
# stay wired (so per-sub-task tuning is a one-line change away), but all three
# profiles are pinned to the model-card CODING values (temp=0.6, top_p=0.95,
# top_k=20) — the recommended sampling for this model. The earlier spread
# (creative temp=0.8/top_k=40, precise temp=0.3/top_p=0.90/top_k=10) drifted
# coding turns off the model-card values; collapsing them to the "balanced"
# baseline keeps every coding turn on-spec while preserving the mechanism.
_CODING_TASK_PROFILES = {
    "creative":  {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0},
    "precise":   {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0},
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
# Defect-reopen churn brake (2026-07-18, Rick Dangerous). A DONE project
# can be reopened by a user bug report (add_task's DONE→ACTIVE semantic),
# and with --autoadvance-idle ON a reopened project immediately becomes
# advanceable — so a report/refute/reopen/advance/rollup cycle can grind
# turns indefinitely on a finished project. Cap reopens per rolling
# window; past the cap the report is surfaced LOUDLY for the operator
# instead of silently re-entering the grinder.
_DEFECT_REOPEN_CAP = 2
_DEFECT_REOPEN_WINDOW_S = 86_400.0

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


# Newest-heavy evidence budget splits for _collect_verifier_evidence,
# indexed by (item count - 1). A single-tool turn keeps the FULL budget —
# behaviourally identical to the old single-output evidence path.
_EVIDENCE_BUDGET_WEIGHTS = ([1.0], [0.65, 0.35], [0.5, 0.3, 0.2])


# A URL this long is feed/tracking plumbing (UTM params, slug echoes), not
# evidence — the claim under audit never cites it, and on RSS-shaped output
# it can be ~70% of the payload. Trimmed to a stub so the host+path still
# attribute the item while the chars go to verifiable content instead.
_EVIDENCE_LONG_URL_RE = re.compile(r"https?://\S{72,}")

# Appended to every auto-repair directive. The directive rides a user-role
# message and the refuted draft is discarded, so the REAL user never saw
# what is being corrected — without this, the model answers the alert
# conversationally and the acknowledgement ("You're right — I was
# embellishing…") leaks checker-facing dialogue into the only reply the
# user gets (req 4dab5067, 2026-07-17).
_REPAIR_STANDALONE_SUFFIX = (
    " IMPORTANT: the user never saw the draft this alert refers to and "
    "cannot see this alert. Write the corrected answer as a clean, "
    "standalone reply to the user's original request — do NOT acknowledge "
    "this alert, do NOT apologise or say what you fixed, and do NOT "
    "mention any verifier, review, or correction."
)


def _squeeze_evidence_noise(text: str) -> str:
    """Collapse zero-entailment-value bulk (long tracking URLs) in a tool
    output before it competes for the verifier's evidence budget."""
    return _EVIDENCE_LONG_URL_RE.sub(lambda m: m.group(0)[:64] + "…", text)


def _collect_verifier_evidence(tools_run: Optional[list],
                               max_items: int = 3,
                               budget: int = 4000) -> str:
    """Build the EVIDENCE string for the verifier's ``verify_claim`` gate
    from the last ``max_items`` substantive tool outputs — not just the
    single most recent one.

    Why: the final answer of a multi-source research turn synthesises
    MANY tool outputs, but the gate used to judge it against only the
    LAST substantive output (4000 chars). Whichever tool happened to run
    last decided the verdict: on the 2026-07-16 souvlaki turn (req
    738c/73) the last fetch was a 403 while the answer came from two
    EARLIER successful loads — the verifier truthfully reported "evidence
    does not support the claim" about the sliver it was shown and REFUTED
    a correct answer.

    Selection mirrors ``_find_substantive_tool_for_verifier`` exactly
    (synthetic and bookkeeping entries skipped), so "no evidence here"
    keeps meaning the same thing to both. Output is chronological (the
    judge reads the turn the way it happened), each item prefixed with
    ``[tool_name]`` so the prompt can attribute failures to a single
    tool, and budgets are newest-heavy (the final answer usually leans
    most on the latest evidence) with unused slack redistributed to
    still-truncated items — a tiny newest output must not starve a
    large older one. Long tracking URLs are trimmed first; they carry
    no entailment value. Total length is capped at ``budget``
    INCLUDING labels/separators, so ``verify_claim``'s own ``[:4000]``
    guard never truncates the newest item away. Returns ``""`` when no
    substantive tool exists — callers fall back / skip exactly as the
    single-tool path does.
    """
    picked: list = []
    for tool in reversed(tools_run or []):
        if not tool:
            continue
        if tool.get("_synthetic"):
            continue
        name = str(tool.get("name", "")).lower().strip()
        collapsed = name.replace("-", "_").replace(" ", "_")
        if collapsed in _BOOKKEEPING_TOOL_NAMES:
            continue
        picked.append(tool)
        if len(picked) >= max_items:
            break
    if not picked:
        return ""
    picked.reverse()  # chronological: oldest → newest
    weights = _EVIDENCE_BUDGET_WEIGHTS[len(picked) - 1]
    labels = [f"[{str(t.get('name', 'tool'))[:80]}] " for t in picked]
    bodies = [_squeeze_evidence_noise(str(t.get("content", "")))
              for t in picked]
    # Pass 1 — weighted caps. weights are newest-first; picked is
    # oldest-first — walk them from opposite ends so the NEWEST output
    # gets the biggest slice. Each cap covers its label and the joining
    # separator, so the assembled string can never exceed `budget`.
    caps = [max(1, int(budget * w) - len(lbl) - 2)
            for lbl, w in zip(labels, reversed(weights))]
    granted = [min(len(b), c) for b, c in zip(bodies, caps)]
    # Pass 2 — redistribute slack. A weighted slice its item doesn't fill
    # is dead budget under one-pass allocation: on the 2026-07-17
    # naftemporiki turn (req 4dab5067) a 106-char weather report — the
    # NEWEST tool — held 65% of the budget while the 4KB headlines feed
    # was cut mid-item #4, and the verifier then REFUTED the answer's
    # items #5–#10 as "not in the evidence". Unused chars go to
    # still-truncated items, newest first.
    spare = sum(c - g for c, g in zip(caps, granted))
    for i in reversed(range(len(picked))):
        if spare <= 0:
            break
        need = len(bodies[i]) - granted[i]
        if need > 0:
            take = min(need, spare)
            granted[i] += take
            spare -= take
    parts = [lbl + b[:g] for lbl, b, g in zip(labels, bodies, granted)]
    return "\n\n".join(parts)


# Upper bound on how long the hydration-usefulness judge defers to an
# in-flight verifier verdict at finalize (see _judge_hydration_safe).
# Sized above the verify worker budget (45s) + its direct-fallback retry,
# and well under the judge's own 600s stash-staleness guard.
_HYDRATION_JUDGE_STAGGER_S = 90.0


# Mutation-verb markers in a file_system SUCCESS message. A file READ
# returns the file's CONTENT (no SUCCESS:/verb), so it is correctly NOT
# flagged — only write/replace confirmations are.
_FILE_MUTATION_MARKERS = (
    "wrote", "written", "replaced", "replace applied",
    "auto-promoted", "search/replace", "overwrote", "overwritten",
)


def _is_unverified_mutation(tool: Optional[dict]) -> bool:
    """True when ``tool`` is a SUCCESSFUL file write/replace — i.e. the
    turn's final substantive action mutated a file but the turn ended
    without ever running or rendering it.

    The verifier gate uses this to refuse a clean "success" finish on an
    untested change: the req_C0 failure was a 33-minute build that
    "finished" immediately after a `file_system` write that was never
    executed or screenshotted, yet reported C=0.96. Conservative — fires
    only for the ``file_system`` tool AND only on a write/replace SUCCESS
    confirmation, so a file READ (which also carries evidence) is not
    mistaken for a mutation."""
    if not tool or not isinstance(tool, dict):
        return False
    name = str(tool.get("name", "")).lower().replace("-", "_").replace(" ", "_")
    if name not in ("file_system", "filesystem", "file"):
        return False
    content = str(tool.get("content", "")).lower()
    if "success" not in content:
        return False
    return any(marker in content for marker in _FILE_MUTATION_MARKERS)


# Quoted filenames with a web extension inside a file_system SUCCESS
# message ("SUCCESS: Wrote 8214 chars to 'index.html'. ...").
_WEB_ARTIFACT_RE = re.compile(r"'([^']+\.(?:html?|js|mjs|cjs))'")


def _web_artifacts_written(tools_run: Optional[list]) -> list:
    """Filenames of web files (html/js) successfully written this turn.

    Feeds the verifier's execution check: a turn that WROTE web files must
    have its entry page actually loaded before a CONFIRMED is credible.
    Parses the recorded tool SUCCESS messages because turn records carry
    only (name, content), not the original call arguments."""
    out: list = []
    for t in tools_run or []:
        if not isinstance(t, dict) or t.get("_synthetic"):
            continue
        if str(t.get("name", "")).lower().replace("-", "_") not in (
                "file_system", "filesystem", "file"):
            continue
        content = str(t.get("content", ""))
        if not content.startswith("SUCCESS"):
            continue
        for m in _WEB_ARTIFACT_RE.findall(content):
            if m not in out:
                out.append(m)
    return out


# ── Deliverable-file ground-truth check ──────────────────────────────
# The #1 most-retrieved real lesson (ret=55) is the agent "prematurely
# declared task completion ... without showing the actual content" — it
# claims a file deliverable that is missing or empty in the sandbox. A
# text/vision verifier can't catch this (the claim reads fine); re-reading
# the file is hard ground truth. This is the general, non-web form of the
# web-exec override's "execute, don't trust" philosophy.
_DELIVERABLE_EXT = (
    r"md|markdown|txt|text|csv|tsv|json|jsonl|ya?ml|py|js|ts|sh|bash|sql|"
    r"html?|xml|ini|toml|cfg|conf|log|pdf|xlsx?|docx?|pptx?|png|jpe?g|svg|"
    r"tex|rs|go|java|c|cpp|rb|php|ipynb"
)
# A COMPLETION verb (past/perfect — the agent DID produce it, not a present-
# tense description of what a script does) followed within a short window by
# a filename. Anchoring on completion verbs keeps input files the agent only
# READ out of the candidate set and avoids "the script writes to X" (present
# tense) false positives.
_CLAIMED_FILE_RE = re.compile(
    r"\b(?:saved|wrote|written|created|generated|produced|exported|stored|"
    r"placed|dumped)\b"
    r"[^\n`'\"]{0,60}?"
    r"[`'\"]?([A-Za-z0-9][A-Za-z0-9._/\-]*\.(?:" + _DELIVERABLE_EXT + r"))\b",
    re.IGNORECASE,
)
_SYS_PATH_PREFIXES = ("/usr", "/etc", "/bin", "/sbin", "/lib", "/opt",
                      "/var", "/System", "/Library", "/private", "/dev",
                      "/proc", "/Users", "/home", "/root", "/tmp")


def _claimed_deliverable_files(text) -> list:
    """Filenames the answer presents as a CREATED/SAVED deliverable."""
    if not text:
        return []
    out: list = []
    for m in _CLAIMED_FILE_RE.findall(str(text)):
        name = m.strip().strip("`'\"")
        if (not name or "://" in name
                or name.startswith(_SYS_PATH_PREFIXES)):
            continue
        if name not in out:
            out.append(name)
    return out[:8]


# ── Visual verification evidence ─────────────────────────────────────
# A text-only verifier cannot tell whether a reported VISUAL symptom
# ("the blocks look transparent", "the layout is broken") was actually
# fixed — the truth lives in the rendered pixels, not in the agent's
# self-claim. These helpers gate + gather the before/after screenshots so
# the verifier can look at the artifact and override an over-optimistic
# text CONFIRMED. See the visual-verify block in handle_chat.
_VISUAL_INTENT_RE = re.compile(
    r"\b(screenshot|screen[\s-]?grab|\bscreen\b|render(?:s|ed|ing)?|"
    r"transparen\w*|see[\s-]?through|invisible|blank|black\s+screen|"
    r"\bui\b|layout|css|displayed?|visual\w*|pixel\w*|graphic\w*|"
    r"looks?\s+(?:wrong|off|broken|weird|transparent)|appears?)\b",
    re.IGNORECASE,
)
_IMAGE_TOKEN_RE = re.compile(
    r"[\w./\\-]+\.(?:png|jpe?g|webp|gif|bmp)", re.IGNORECASE,
)


def _is_visual_intent(text: Optional[str]) -> bool:
    """Cheap (no-LLM) gate: does the user's request describe a VISUAL
    symptom or reference an image? Keeps the extra vision call off the
    ~99% of turns that aren't about how something looks."""
    if not text:
        return False
    if _IMAGE_TOKEN_RE.search(text):
        return True
    return bool(_VISUAL_INTENT_RE.search(text))


# ── Interaction-claim verification gap ───────────────────────────────
# A pointer/keyboard behavior ("dragging a window", "clicking the button")
# can only be confirmed by EXERCISING it. The WEB-EXEC probe proves the
# page loads without exceptions — nothing more — yet text entailment over
# a load-clean result happily CONFIRMED a still-broken drag fix at 100%
# (reqs AF/43, 2026-07-17). These helpers gate a confidence cap: the
# symptom is interaction-shaped AND no browser click/interact succeeded
# this turn → CONFIRMED is capped below the ≥0.7 consumption gates.
_INTERACTION_INTENT_RE = re.compile(
    r"\b(?:drag(?:ging|ged)?|resiz(?:e|ing|ed)|mov(?:e|ing|ed)|"
    r"scroll(?:ing)?|hover(?:ing)?|click(?:ing|ed)?|double[- ]click|"
    r"drop(?:ping|ped)?|typ(?:e|ing)|key(?:board|press)|swip(?:e|ing))\b"
    r".{0,60}?\b(?:window|element|button|icon|menu|item|panel|tab|card|"
    r"widget|handle|dialog|modal|slider|div|box)s?\b"
    r"|\b(?:window|element|button|icon|menu|item|panel|tab|card|widget|"
    r"handle|dialog|modal|slider)s?\b.{0,50}?"
    r"\b(?:drag|resiz|mov|click|scroll|hover|drop|swip)\w*\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_interaction_intent(text: Optional[str]) -> bool:
    """Cheap (no-LLM) gate: the request describes a pointer/keyboard
    behavior on a UI element ("moving a window doesn't work")."""
    return bool(text and _INTERACTION_INTENT_RE.search(str(text)))


def _has_interaction_evidence(tools_run: Optional[list]) -> bool:
    """True when a browser click/interact op SUCCEEDED this turn — the
    only tool evidence that can support an interaction-behavior claim.
    Detection reads the tool result header (``STATUS: OK`` + ``OP: …``)
    because tools_run entries carry name+content, not arguments."""
    for tool in (tools_run or []):
        if not isinstance(tool, dict) or tool.get("_synthetic"):
            continue
        if str(tool.get("name", "")).lower() != "browser":
            continue
        content = str(tool.get("content", ""))
        if "STATUS: OK" not in content:
            continue
        if re.search(r"^OP: (?:click|interact)\b", content, re.MULTILINE):
            return True
    return False


_BUG_REPORT_RE = re.compile(
    r"(nothing happens|doesn'?t work|does not work|not working|"
    r"isn'?t working|stopped working|won'?t (load|start|open|run)|"
    r"when i (click|press|open|run|load|type)\b.{0,60}"
    r"(nothing|fails?|error|crash|blank|stuck)|"
    r"\b(button|link|page|screen|app|game)\b.{0,40}"
    r"\b(broken|dead|stuck|frozen|blank)\b|"
    r"\bcrash(es|ed)?\b)",
    re.IGNORECASE | re.DOTALL,
)


def _estimate_messages_tokens(messages) -> int:
    """Rough token estimate over a message list (text parts only) — the
    shared basis for the occupancy-aware read budget and the context-
    pressure steers. Never raises."""
    total = 0
    try:
        for m in messages or []:
            c = m.get("content", "")
            if isinstance(c, str):
                total += estimate_tokens(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += estimate_tokens(str(part.get("text", "")))
    except Exception:
        return total
    return total


def _render_cwd_pin(project_id: Optional[str]) -> str:
    """The top-of-dynamic-state CWD pin for coding turns.

    Project-aware since 2026-07-18: the pin used to say "SHELL CWD IS
    /workspace" as static text even while a project was bound and the
    shell actually started in /workspace/projects/<id> — so the model
    trusted the pin over the (quieter) workspace note and ran
    `cd /workspace && git clone …`, planting a repo and the feasibility
    report at the sandbox ROOT where the project's manifest, briefing
    and cleanup never see them (observed live, request 6f14407f)."""
    if project_id:
        proj = f"/workspace/projects/{project_id}"
        return (
            f"⚠️  SHELL CWD IS {proj} — YOUR PROJECT WORKSPACE — DO NOT `cd` OUT OF IT  ⚠️\n"
            "The shell session ALREADY starts inside the project directory. "
            "Relative paths land INSIDE the project — exactly where its "
            "files belong. Anything written to /workspace/ outside "
            f"projects/{project_id}/ is OUTSIDE the project: invisible to "
            "its deliverables manifest, its briefing, and its cleanup. "
            "Paths like /sandbox, /home/user, /root, /app do NOT exist.\n"
            "  ✓  git clone <url> repo-src ; ls repo-src/\n"
            "  ✓  python3 parser.py ; cat output/report.md\n"
            f"  ✗  cd /workspace && …            (ESCAPES the project!)\n"
            f"  ✗  /workspace/some-dir/file.md   (outside the project)\n\n"
        )
    return (
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


# `cd /workspace` as a token (not followed by /projects/), or an explicit
# /workspace/<seg> reference whose first segment is not projects/ — the two
# shapes by which a project-scoped shell escapes to the sandbox root.
_OFFPROJECT_CD_RE = re.compile(
    r"cd\s+/workspace(?:(?=[\s\"'&;|)])|/(?!projects/)|$)"
)
_OFFPROJECT_PATH_RE = re.compile(
    r"/workspace/(?!projects/)[\w.-]+"
)


def _offproject_target(fname, ptarget, a_hash, project_id) -> Optional[str]:
    """Return the offending path when a SUCCESSFUL tool call operated at the
    sandbox root while a project is bound, else None.

    The scoping machinery re-roots RELATIVE paths into projects/<id>/, and
    the /workspace→project heal fires only on file-not-found — so a call
    that succeeds with an explicit root-absolute path escapes every guard
    silently (observed live, request 6f14407f: `cd /workspace && git clone
    … prince-persia-repo` planted the repo and then the feasibility report
    at the root; the project dir stayed empty). This detector powers the
    once-per-request corrective steer in the dispatch pipeline."""
    if not project_id:
        return None
    if fname == "file_system":
        t = str(ptarget or "")
        if t.startswith("/workspace/") and not t.startswith(
                f"/workspace/projects/{project_id}"):
            if not t.startswith("/workspace/projects/"):
                return t
        return None
    if fname == "execute":
        blob = str(a_hash or "")
        m = _OFFPROJECT_CD_RE.search(blob)
        if m:
            return "cd /workspace (escapes the project scope)"
        m = _OFFPROJECT_PATH_RE.search(blob)
        if m:
            return m.group(0)
    return None


def _is_bug_report_intent(text: Optional[str]) -> bool:
    """Cheap (no-LLM) gate: the user is REPORTING a defect in something
    that already exists ("when I click X nothing happens"). These turns
    have a known best first move — reproduce and observe — and a known
    catastrophic first move — re-reading the source and hypothesizing
    (req 70: six wrong "I found it!" theories and a killed thinking loop
    before the first observation, which then named the bug instantly)."""
    if not text:
        return False
    return bool(_BUG_REPORT_RE.search(text))


def _extract_image_tokens(blob: Optional[str]) -> List[str]:
    """All image-file-looking tokens in a text blob, in order."""
    if not blob:
        return []
    return _IMAGE_TOKEN_RE.findall(str(blob))


def _resolve_image_path(token: str, sandbox_dir: Any) -> Optional[str]:
    """Resolve an image reference (which may be a container path like
    ``/sandbox/x.png`` or ``/workspace/projects/<id>/y.png``, or a bare
    filename) to an existing host path under ``sandbox_dir``. Mirrors the
    container→host + project-root fallbacks used by tools/vision.py.
    Returns ``None`` if nothing matching exists."""
    if not token or sandbox_dir is None:
        return None
    t = str(token).strip().strip("'\"")
    for pre in ("/api/download/", "/sandbox/", "/workspace/"):
        if t.startswith(pre):
            t = t[len(pre):]
    candidates: List[Path] = []
    try:
        p = Path(t)
        if p.is_absolute():
            candidates.append(p)
    except Exception:
        pass
    try:
        from ..tools.file_system import _get_safe_path
        candidates.append(_get_safe_path(sandbox_dir, t))
        if Path(sandbox_dir).parent.name == "projects":
            candidates.append(_get_safe_path(Path(sandbox_dir).parent.parent, t))
    except Exception:
        candidates.append(Path(sandbox_dir) / t)
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return str(c)
        except Exception:
            continue
    # Last resort: match the basename anywhere under the sandbox (newest wins).
    try:
        base = Path(t).name
        roots = [Path(sandbox_dir)]
        if Path(sandbox_dir).parent.name == "projects":
            roots.append(Path(sandbox_dir).parent.parent)
        matches: List[Path] = []
        for root in roots:
            matches.extend(root.rglob(base))
        matches = [m for m in matches if m.is_file()]
        if matches:
            matches.sort(key=lambda m: m.stat().st_mtime, reverse=True)
            return str(matches[0])
    except Exception:
        pass
    return None


def _select_visual_evidence(messages: list, last_user_content: str,
                            sandbox_dir: Any) -> tuple:
    """Pick (before_image, after_image) host paths for visual verification.

    * before = an image the user referenced in their request (the symptom
      screenshot), if any — taken from ``last_user_content`` directly.
    * after  = a screenshot the agent produced this turn (a DIFFERENT image
      than the user's own). If the only image in play is the user's own,
      ``after`` is None — the agent never re-rendered, so there is no
      post-fix evidence and we must not judge.

    Scans ALL assistant/tool-role messages rather than "everything after the
    last user message": the agent loop injects synthetic user-role messages
    mid-turn (planning nudges, "### ACTIVE STRATEGY", AUTO-DIAGNOSTIC), so the
    last user message is often NOT the human's request and the screenshot can
    sit before it. The user's own before-image lives in a user-role message,
    so restricting to assistant/tool roles keeps before/after distinct.
    Among candidates the NEWEST file on disk wins — this turn's screenshot was
    just written, so it beats any stale image from an earlier turn.

    Returns ``(before|None, after|None)``."""
    before_img = None
    for tok in _extract_image_tokens(last_user_content):
        before_img = _resolve_image_path(tok, sandbox_dir)
        if before_img:
            break

    candidates: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("assistant", "tool"):
            continue
        blob_parts: List[str] = []
        c = msg.get("content")
        if isinstance(c, str):
            blob_parts.append(c)
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                try:
                    blob_parts.append(json.dumps(tc))
                except Exception:
                    blob_parts.append(str(tc))
        for tok in _extract_image_tokens(" ".join(blob_parts)):
            resolved = _resolve_image_path(tok, sandbox_dir)
            if resolved and resolved != before_img and resolved not in candidates:
                candidates.append(resolved)

    after_img = None
    if candidates:
        try:
            after_img = max(candidates, key=lambda p: Path(p).stat().st_mtime)
        except Exception:
            after_img = candidates[-1]
    return before_img, after_img


# Per-request loop-detection accounting lives in core.strikes. These names
# are re-exported here because existing tests import them from
# ghost_agent.core.agent; the StrikeLedger (also in core.strikes) bundles
# the request-scoped state the turn loop used to track as bare locals.
from .strikes import (  # noqa: E402
    StrikeLedger,
    note_repeated_failure as _note_repeated_failure,
    note_repeated_action as _note_repeated_action,
    action_result_fingerprint as _action_result_fingerprint,
    is_readwrite_loop_exempt as _is_readwrite_loop_exempt,
)


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


# Social / open-ended openers that should KEEP the warm conversational profile
# even though they ask a question (no single correct answer).
_CHITCHAT_MARKERS = (
    "how are you", "who are you", "what's up", "whats up", "tell me about yourself",
    "tell me a", "write a poem", "write a story", "what do you think",
    "your opinion", "how do you feel", "let's chat", "lets chat", "good morning",
    "good evening", "thank you", "thanks",
)
# Strong signals that the query expects ONE deterministic answer.
_FACTUAL_MARKERS = (
    "how many", "how much", "how long", "what is the", "what's the", "what are the",
    "compute", "calculate", "evaluate", "solve", "sum of", "product of", "count",
    "number of", "value of", "result of", "how do i", "what year", "what day",
    "which", "convert", "factorial", "prime", "divisible", "average", "median",
    "percentage", "square root", "list the", "define ",
)


def _is_factual_query(query: str) -> bool:
    """Heuristic: does this non-tool query have a single correct/deterministic
    answer (math, counting, lookup) rather than open-ended chit-chat?

    Conservative toward FACTUAL: a genuine question routes to near-greedy
    sampling; only clearly social/creative openers stay on the warm profile.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(m in q for m in _CHITCHAT_MARKERS):
        return False
    if any(m in q for m in _FACTUAL_MARKERS):
        return True
    # a digit present alongside a question usually means a computed answer
    if any(ch.isdigit() for ch in q) and ("?" in q or q.split()[0] in (
            "what", "how", "which", "when", "calculate", "compute", "find")):
        return True
    # short interrogative that isn't a social opener → treat as factual
    if q.endswith("?") and len(q.split()) <= 24 and q.split()[0] in (
            "what", "which", "when", "who", "where", "is", "are", "does", "do",
            "can", "how"):
        return True
    return False


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
        # A non-tool turn is NOT automatically chit-chat. If the query has a
        # single deterministic answer, sample near-greedy with no presence
        # penalty — temp=1.0/pp=1.5 samples AWAY from the correct tokens.
        if _is_factual_query(query):
            return dict(FACTUAL_SAMPLING_PARAMS)
        return dict(GENERAL_SAMPLING_PARAMS)
    if is_coding:
        profile = _classify_coding_task(query)
        return dict(_CODING_TASK_PROFILES.get(profile, CODING_SAMPLING_PARAMS))
    # Non-coding tool turn — use the base precise profile (temp=0.6).
    return dict(CODING_SAMPLING_PARAMS)


# Streaming guards. When the upstream model enters a self-repeating thinking
# loop (re-deriving the same paragraph hundreds of times), we kill the stream
# instead of burning context indefinitely.
MAX_THINKING_CHARS = 32000          # where loop-CHECKING begins (not a kill)
# 200K (was 64K): the hard length-abort is a true backstop, not a guillotine.
# Between 32K and 200K the stream is only aborted when `_detect_thinking_loop`
# actually fires — so a strong model genuinely working a hard algorithm/proof/
# debug problem (which can legitimately exceed ~16K thinking tokens) is allowed
# to finish its derivation, while a real repetition loop is still caught at the
# 32K check and the 200K ceiling stops any pathological runaway.
MAX_THINKING_CHARS_EXTENDED = 200000

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

# When the upstream stops a *text* answer at its token cap
# (`finish_reason == "length"`), the partial reply is shipped mid-sentence
# and the verifier correctly REFUTES it (truncated output / explicit
# question left unanswered). Rather than surface a cut-off final answer, we
# transparently continue the generation from where it stopped, up to this
# many times, then accept whatever we have. Bounded so a model that emits
# "length" on every continuation (mis-reported finish_reason, runaway) can't
# loop forever — each continuation is a full upstream round-trip.
MAX_TRUNCATION_CONTINUATIONS = 3


def _manage_projects_closed_a_task(tool_name: str, result_text: str) -> bool:
    """True iff a ``manage_projects`` call actually closed a task to DONE.

    Used to enforce one-task-per-turn in the interactive loop: a single
    "start task 1" / "proceed" advances EXACTLY one task and then stops, so the
    agent doesn't grind the whole project tree in one request (observed live).

    The signal is the store read-back the tool returns — ``{"updated":[{"id":…,
    "status":"DONE"},…]}`` — NOT the requested status, so a DONE-gated/held
    update (status stays non-DONE pending verification) correctly does NOT
    trip the gate. Defensive: any parse problem returns False.
    """
    if tool_name != "manage_projects" or not result_text:
        return False
    # Cheap pre-check before paying for a JSON parse.
    if '"status": "DONE"' not in result_text:
        return False
    try:
        payload = json.loads(result_text)
    except Exception:
        return False
    updated = payload.get("updated") if isinstance(payload, dict) else None
    if not isinstance(updated, list):
        return False
    return any(isinstance(u, dict) and str(u.get("status", "")).upper() == "DONE"
               for u in updated)


def _scrub_host_sandbox_paths(text: str, sandbox_root) -> str:
    """Rewrite any HOST-absolute sandbox path leaking into the model's context
    to its container-visible ``/workspace`` form.

    A recalled memory, a scratchpad note, or a workspace narrative written in a
    PRIOR session can carry an absolute host path like
    ``/Users/x/Data/AI/Data/sandbox/projects/<id>/...``. The model runs shell
    commands INSIDE the container, where the sandbox root is bind-mounted at
    ``/workspace`` and that host path does NOT exist — so a ``cd`` to it ENOENTs
    and burns a strike (observed live 3× in one session: the model kept doing
    ``cd /Users/.../sandbox/projects/<id>/PetAI``). The container form is valid
    for both the shell and the file tools (``_get_safe_path`` accepts
    ``/workspace/...``). Best-effort; returns the text unchanged on any problem.
    """
    if not text or not sandbox_root:
        return text
    roots = []
    try:
        roots.append(str(Path(sandbox_root)))
        rp = str(Path(sandbox_root).resolve())
        if rp not in roots:
            roots.append(rp)
    except Exception:
        roots = [str(sandbox_root)]
    for r in roots:
        if r and r not in ("/", "") and r in text:
            text = text.replace(r, "/workspace")
    return text


def _scrub_fallback_message(intended: str, task_closed: bool) -> str:
    """User-facing text when a text-only turn's tool_call was fully scrubbed.

    Two causes, two messages:
      * ``task_closed`` — the one-task-per-turn gate finalized the turn after a
        project task closed; the dropped tool_call was the model trying to start
        the NEXT task. Say so plainly — telling the user to "rephrase" a request
        that already succeeded is misleading.
      * otherwise — a genuine planner/model routing mismatch (the turn was
        routed text-only but the model still wanted a tool). Keep the generic
        prepared-a-tool-call guidance.
    """
    if task_closed:
        return (
            "✅ Task complete. I've stopped here so you stay in control — say "
            "*proceed* / *next* to continue with the next task, *do the rest* to "
            "run the remaining tasks, or give new direction."
        )
    tool = intended or "the command"
    msg = "I prepared a tool call but this turn was routed as text-only, so it wasn't executed."
    if intended:
        msg += f" (Intended tool: `{intended}`.)"
    msg += f" Please rephrase your request — for example, `run {tool}` — or try again."
    return msg


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
    # dream_mode's footer is two clauses joined by a period
    # ("SESSION FINISHED. STAND BY."). The single-keyword pattern above
    # can't span it — the period isn't in its [A-Z_ ] class — so strip the
    # combined form explicitly. (Pre-existing: the model-driven dream bypass
    # leaked this footer too; cleaned up alongside the deterministic path.)
    cleaned = re.sub(
        r"\n*SYSTEM:\s*SESSION\s+FINISHED\.?\s*STAND BY\.?\s*$",
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


# Phrases that, when they DOMINATE a short user message, mean "run ONE
# self-play cycle now". Deliberately tighter than tools.memory's
# `_SELF_PLAY_INTENT_PHRASES` (which also matches loop/continuous asks):
# the deterministic dispatch below only ever forces the single-cycle
# `self_play` tool, never the background `self_play_loop`.
_SELF_PLAY_COMMAND_PHRASES = (
    "self play",
    "self-play",
    "selfplay",
    "practice cycle",
    "practice round",
    "practice session",
    "training cycle",
)
# If any of these appear, the user wants the CONTINUOUS loop (or some other
# tool) — defer to the model instead of forcing a single cycle. "again" is
# NOT here: "self play again" is still a single cycle.
_SELF_PLAY_LOOP_MARKERS = (
    "loop",
    "keep ",
    "until stopped",
    "until i stop",
    "continuous",
    "continuously",
    "again and again",
    "over and over",
    "nonstop",
    "non-stop",
)


def _is_single_self_play_command(text: str) -> bool:
    """True when `text` is *predominantly* an explicit single-cycle
    self-play command (e.g. "self play", "self play again", "run self-play").

    This gates a deterministic, pre-LLM dispatch of the `self_play` tool.
    Without it, a bare "self play" runs the tool only on the FIRST ask and
    is silently replayed thereafter: the terminal-tool bypass persists the
    cycle's summary as a PLAIN-TEXT assistant turn (no tool_call), and the
    memory bus re-hydrates it as ``Context: Self-play complete…`` — so on
    the next identical ask the model's in-context example says the right
    response to "self play" is to reprint that text. It imitates the text
    and never re-fires the cycle (observed: identical 126-char reply,
    ``tool=-`` in the metacog trace).

    Conservative by construction: requires a command phrase, rejects
    loop/continuous markers (those route to `self_play_loop`, the model's
    call), and caps length so a long project message that merely *mentions*
    self-play in passing is left to the model — only a short, command-
    dominant message is force-dispatched.
    """
    if not text:
        return False
    lc = " ".join(str(text).lower().split())
    if not any(p in lc for p in _SELF_PLAY_COMMAND_PHRASES):
        return False
    if any(m in lc for m in _SELF_PLAY_LOOP_MARKERS):
        return False
    # Command-dominant: a short message that IS the command, not a paragraph
    # that happens to contain it. 6 words covers "do another self play cycle
    # please" while rejecting prose.
    return len(lc.split()) <= 6


# Phrases that, when they DOMINATE a short user message, mean "run a memory-
# consolidation (dream) cycle now". `dream_mode` is the other terminal tool
# subject to the same replay bug as `self_play` (its summary is persisted as
# a plain-text turn and re-hydrated). Deliberately multi-word / unambiguous:
# bare "sleep"/"rest"/"dream" are avoided because they substring-match prose
# ("interesting" contains "rest", "daydream" contains "dream").
_DREAM_COMMAND_PHRASES = (
    "dream mode",
    "go to sleep",
    "time to sleep",
    "get some sleep",
    "consolidate memories",
    "consolidate your memories",
    "consolidate memory",
    "consolidate your memory",
    "memory consolidation",
)


def _is_dream_command(text: str) -> bool:
    """True when `text` is *predominantly* an explicit memory-consolidation
    (dream) command (e.g. "dream mode", "go to sleep", "consolidate
    memories"). Same deterministic-dispatch rationale as
    `_is_single_self_play_command`. `dream_mode` has no continuous-loop
    variant, so there are no loop markers to exclude — just a command phrase
    and a length cap so a prose mention is left to the model."""
    if not text:
        return False
    lc = " ".join(str(text).lower().split())
    if not any(p in lc for p in _DREAM_COMMAND_PHRASES):
        return False
    return len(lc.split()) <= 6


def _explicit_terminal_command(text: str):
    """Return the terminal tool name (``"self_play"`` | ``"dream_mode"``) the
    message is an unambiguous, command-dominant request for, or ``None``.

    Single source of truth for the pre-LLM deterministic dispatch in
    ``handle_chat``: both terminal tools share the replay bug (their summary
    is persisted as a plain-text turn with no tool_call and re-hydrated by
    the memory bus), so both are force-dispatched the same way."""
    if _is_single_self_play_command(text):
        return "self_play"
    if _is_dream_command(text):
        return "dream_mode"
    return None


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
# Streaming sanity guards live in core/stream_guards.py (the guard-module seam,
# IMPROVEMENTS.md #5) — new stream guards land THERE, not inline here. Re-export
# the names so existing references + tests in this module keep working.
from .stream_guards import (  # noqa: E402
    THINKING_LOOP_PROBE_EVERY, THINKING_LOOP_WINDOW, THINKING_LOOP_THRESHOLD,
    TOOL_CALL_LOOP_THRESHOLD, TOOL_CALL_LOOP_PROBE_EVERY,
    _STREAM_STOP_MARKERS,
    _detect_thinking_loop, _tail_has_stop_marker, _detect_tool_call_loop,
)


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
# NATIVE TOOL-CALL DE-CORRUPTION
# ----------------------------------------------------------------------------
# With ``--native-tools`` on (default for Qwen 3.6+), the upstream server
# parses the model's tool-call XML itself and hands us structured
# ``message.tool_calls`` — bypassing our own robust XML parser. Some servers
# mis-handle a MULTI-tool reply (especially the equals-format
# ``<function=name>`` / ``<parameter=name>`` shape): they close the first
# ``<parameter>`` correctly but then leak the ENTIRE serialized XML of every
# FOLLOWING tool call into the first argument's string value, e.g.
#
#   {"action": "summary</parameter>\n</function>\n</tool_call>\n
#              <tool_call>\n<function=list_lessons>\n<parameter=scope>\nall",
#    "limit": 10}
#
# The value the model actually intended is ``"summary"``; everything from the
# first framing token on is another tool call. Left uncorrected, ``action``
# fails validation ('action' must be one of [...]) and the whole turn is
# wasted — the exact "introspect keeps erroring on a valid action" symptom.
#
# We detect the leak by its unambiguous CLOSE-then-CONTINUATION signature (a
# ``</parameter>`` / ``</function>`` / ``</tool_call>`` close FOLLOWED by a NEW
# ``<tool_call>`` / ``<function=...>`` opening OR a sibling ``<parameter=...>``)
# then truncate the value at the first framing token and reconstruct intent:
#  - a leaked sibling ``<parameter=...>`` folds back into the SAME call's args
#    (e.g. manage_tasks: ``action="stop</parameter><parameter=task_identifier>…"``
#    → ``{action: stop, task_identifier: …}``);
#  - a leaked ``<function=...>`` naming a KNOWN tool is recovered as a separate
#    call; and when the whole value is framing (no clean prefix) the phantom
#    primary is dropped and the first recovered call promoted.
# Content that merely *mentions* framing text (code, docs) has no ordered
# close-then-continuation pair — or names no real tool and carries no sibling
# param — so it is never mangled.
# ============================================================================
_TC_NEXT_CALL_RE = re.compile(r'<tool_call\b|<function\s*=|<function\s+name\s*=', re.IGNORECASE)
# Leak-continuation tokens: a NEW tool call (function/tool_call) OR a sibling
# `<parameter …>` of the SAME call. The upstream native parser can merge either
# shape into the first argument's value — e.g. manage_tasks got
# `{"action": "stop</parameter>\n<parameter=task_identifier>\ntask_123"}` where
# the second PARAMETER leaked into the first, not a whole new call.
_TC_NEXT_LEAK_RE = re.compile(
    r'<tool_call\b|<function\s*=|<function\s+name\s*=|<parameter\s*=|<parameter\s+name\s*=',
    re.IGNORECASE)
_TC_CLOSE_RE = re.compile(r'</parameter\s*>|</function\s*>|</tool_call\s*>', re.IGNORECASE)
_TC_FIRST_FRAME_RE = re.compile(
    r'</parameter\s*>|</function\s*>|</tool_call\s*>|<tool_call\b|<function\s*=|<function\s+name\s*=',
    re.IGNORECASE,
)
_TC_FUNC_OPEN_RE = re.compile(
    r'<function(?:\s+name\s*=\s*["\']?|\s*=\s*["\']?)\s*([a-zA-Z0-9_]+)', re.IGNORECASE)
_TC_PARAM_RE = re.compile(
    r'<parameter(?:\s+name\s*=\s*["\']?|\s*=\s*["\']?)\s*([a-zA-Z0-9_]+)["\']?\s*>'
    r'(.*?)(?=</parameter>|<parameter[\s=]|<function[\s=]|</function>|</tool_call>|<tool_call\b|$)',
    re.DOTALL | re.IGNORECASE,
)


def _value_has_leaked_framing(value) -> bool:
    """True when a string argument value carries the CLOSE-then-LEAK signature
    of an upstream-merged reply: a ``</parameter>``/``</function>``/``</tool_call>``
    close followed by a new tool call OR a sibling ``<parameter …>`` (see block
    comment). Content that merely mentions framing text lacks the ordered pair."""
    if not isinstance(value, str):
        return False
    close = _TC_CLOSE_RE.search(value)
    if not close:
        return False
    return _TC_NEXT_LEAK_RE.search(value, close.end()) is not None


def _recover_calls_from_tail(tail: str, available_names) -> list:
    """Recover ``[(name, args_dict), ...]`` for each ``<function ...>`` opening
    in the leaked ``tail``, keeping only functions in ``available_names``."""
    calls = []
    opens = list(_TC_FUNC_OPEN_RE.finditer(tail))
    for i, fm in enumerate(opens):
        fname = fm.group(1)
        if available_names and fname not in available_names:
            continue
        seg_start = fm.end()
        seg_end = opens[i + 1].start() if i + 1 < len(opens) else len(tail)
        segment = tail[seg_start:seg_end]
        args = {}
        for pm in _TC_PARAM_RE.finditer(segment):
            args[pm.group(1)] = pm.group(2).strip().strip('"').strip("'")
        calls.append((fname, args))
    return calls


def _repair_one_native_tool_call(tc: dict, available_names):
    """Return ``(primary_tc, [recovered_tcs])``. When no leak is present (or
    the leaked tail names no known tool), returns ``(tc, [])`` unchanged."""
    fn = tc.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, str):
        try:
            args = json.loads(raw, strict=False)
        except Exception:
            return tc, []
    elif isinstance(raw, dict):
        args = dict(raw)
    else:
        return tc, []
    if not isinstance(args, dict):
        return tc, []

    def _mk(name, a):
        return {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(a)}}

    for k, v in list(args.items()):
        if not (isinstance(v, str) and _value_has_leaked_framing(v)):
            continue
        m = _TC_FIRST_FRAME_RE.search(v)
        if not m:
            continue
        tail = v[m.start():]
        # Split the tail at the first NEW-CALL opening. Everything before it is
        # sibling <parameter …> of THIS call that leaked into the value; from
        # the opening on are separate merged tool calls.
        call_open = _TC_NEXT_CALL_RE.search(tail)
        pre = tail[:call_open.start()] if call_open else tail
        post = tail[call_open.start():] if call_open else ""
        sibling = {}
        for pm in _TC_PARAM_RE.finditer(pre):
            sibling[pm.group(1)] = pm.group(2).strip().strip('"').strip("'")
        recovered = _recover_calls_from_tail(post, available_names) if post else []
        # Positively identify a leak: we must have recovered a known call OR
        # extracted at least one sibling parameter. Otherwise it's content we
        # can't attribute to the parser — leave it untouched.
        if not recovered and not sibling:
            continue
        recovered_tcs = [_mk(name, a) for name, a in recovered]
        if m.start() > 0:
            # A clean prefix survives before the framing: truncate the primary's
            # value to it, fold in any leaked sibling params (without clobbering
            # an existing key), and append the recovered separate calls.
            new_args = dict(args)
            new_args[k] = v[:m.start()].rstrip().strip('"').strip("'")
            for sk, sv in sibling.items():
                new_args.setdefault(sk, sv)
            primary = {**tc, "function": {**fn, "arguments": json.dumps(new_args)}}
            return primary, recovered_tcs
        # m.start() == 0: the value is ENTIRELY framing (no clean prefix). The
        # native parser dumped the whole merged reply into this one argument.
        if recovered_tcs:
            # The primary call is a phantom whose real content IS the first
            # recovered call — promote it rather than dispatch raw tag-soup
            # (that struck as an invalid-arg failure — seen live on file_system).
            return recovered_tcs[0], recovered_tcs[1:]
        # Only leaked sibling params, no recovered call, no clean prefix: attach
        # them to the primary (dropping the tag-soup value for this key).
        new_args = {kk: vv for kk, vv in args.items() if kk != k}
        for sk, sv in sibling.items():
            new_args.setdefault(sk, sv)
        return {**tc, "function": {**fn, "arguments": json.dumps(new_args)}}, []
    return tc, []


def _repair_native_tool_calls(tool_calls: list, available_names=None):
    """Repair upstream native ``tool_calls`` corrupted by the server's own
    multi-call XML parser. Returns ``(new_tool_calls, repaired: bool)``.
    Clean calls pass through byte-for-byte."""
    out, repaired = [], False
    for tc in (tool_calls or []):
        primary, extras = _repair_one_native_tool_call(tc, available_names)
        out.append(primary)
        out.extend(extras)
        # A repair happened if we recovered extra calls OR rewrote the primary
        # (the pure-framing "phantom" case promotes a recovered call in place,
        # with no extras).
        if extras or primary is not tc:
            repaired = True
    return out, repaired


# ============================================================================
# <think> STRIPPING
# ----------------------------------------------------------------------------
# A single lazy regex `<think>.*?(?:</think>|(?=<tool_call|function)|$)` was
# used to strip reasoning blocks before tool-call parsing. The `(?=<tool_call>)`
# alternation exists for the pathological case where the model opens `<think>`
# and runs straight into a REAL tool call without ever emitting `</think>`.
# But with a lazy `.*?`, that lookahead also fires on a `<tool_call>` merely
# MENTIONED inside the reasoning — e.g. the model quoting the guidance
# "emit exactly one `<tool_call>` per turn". It cut the block at the quoted
# mention, leaving `<tool_call>` per turn.\n\n</think>\n<tool_call>…` as the
# parse target, which the tool parser then mis-read as a malformed call →
# `system_parse_error`, a wasted turn, and an execution strike (seen recurring
# in the live log on coding tasks). The fix PREFERS a real `</think>`: closed
# blocks are removed whole (a quoted `<tool_call>` inside them can no longer
# truncate anything), and only a genuinely UNCLOSED `<think>` is stripped up
# to the first real tool-call opening (wrapped `<tool_call><function>` or a
# named `<function …>`), never a bare quoted mention.
# ============================================================================
_THINK_CLOSED_RE = re.compile(r'<think\b[^>]*>.*?</think\s*>', re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(
    r'<think\b[^>]*>.*?(?=<tool_call\b[^>]*>\s*<function\b|<function\s+name\b|<function\s*=)'
    r'|<think\b[^>]*>.*$',
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_blocks(text: str) -> str:
    """Remove `<think>…</think>` reasoning, preferring the real close tag so a
    quoted `<tool_call>` mention inside the reasoning can't truncate the block
    (see block comment). Handles an unclosed `<think>` by stripping to the
    first real tool-call opening or EOS. Returns the input unchanged when there
    is no think block (cheap fast-path)."""
    if not isinstance(text, str) or '<think' not in text.lower():
        return text
    text = _THINK_CLOSED_RE.sub('', text)
    if '<think' in text.lower():
        text = _THINK_UNCLOSED_RE.sub('', text)
    return text


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

def _is_think_tag_fragment(token: str, accumulated_with_token: str) -> bool:
    """Display-stream filter: is this streamed token a fragment of a
    literal ``<think>`` / ``</think>`` tag (arriving split across
    chunks), rather than prose?

    The old check dropped ANY token whose stripped form was "think" or
    ">" — which deleted the literal word "think" from every rendered
    thought ("Let me think about…" rendered as "Let me about…", observed
    consistently in production logs; the model emits " think" as its own
    token). A bare "think"/"think>"/">" token is only a tag fragment
    when the stream immediately before it ends with the matching tag
    opener, so the check is now contextual: `accumulated_with_token` is
    the stream INCLUDING this token (both call sites append before
    filtering).

    This filter is cosmetic-only — it gates what `_emit_thinking` shows
    the operator; the accumulated content is never modified.
    """
    t = token.strip().lower()
    if t in ("<", "</", "<think", "<think>", "</think", "</think>"):
        return True
    if token.lower() != t:
        # Surrounding whitespace → can't be part of a contiguous tag.
        return False
    before = accumulated_with_token[: len(accumulated_with_token) - len(token)]
    if t in ("think", "think>"):
        return before.endswith(("<", "</"))
    if t == ">":
        return before.lower().endswith(("<think", "</think"))
    return False


def _repair_truncated_json(t: str) -> dict:
    """Best-effort parse of a JSON object cut off mid-generation.

    Strategy per attempt: close an unterminated string, drop a trailing
    comma, complete a dangling ``"key":`` with ``null``, append the
    missing closers, and parse. On failure, chop back to the previous
    comma/opener and retry (bounded). Returns ``{}`` when nothing
    salvageable remains — same contract as ``extract_json_from_text``.
    """
    import json
    cur = (t or "").strip()
    for _ in range(8):
        if not cur or cur[0] != '{':
            return {}
        stack = []
        in_str = False
        esc = False
        for ch in cur:
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in '{[':
                    stack.append(ch)
                elif ch in '}]':
                    if stack:
                        stack.pop()
        cand = cur + ('"' if in_str else '')
        cand = re.sub(r'[\s,]+$', '', cand)
        if cand.endswith(':'):
            cand += ' null'
        cand += ''.join('}' if op == '{' else ']' for op in reversed(stack))
        try:
            res = json.loads(cand, strict=False)
            if isinstance(res, dict):
                return res
        except json.JSONDecodeError:
            pass
        cut = max(cur.rfind(','), cur.rfind('{'), cur.rfind('['))
        if cut <= 0:
            return {}
        cur = cur[:cut]
    return {}


def extract_json_from_text(text: str, repair_truncated: bool = False) -> dict:
    """Safely extracts JSON from LLM outputs, ignoring conversational filler and markdown blocks.

    Returns ``{}`` on every failure mode (missing JSON, malformed JSON, etc.)
    — the empty-dict contract is load-bearing for the dozens of call sites
    that do `result = extract_json_from_text(...).get("score", 0)` style
    access.

    ``repair_truncated=True`` additionally salvages objects cut off at a
    max_tokens cap (see ``_repair_truncated_json``). It is OPT-IN and must
    only be set on background extraction paths (memory consolidation,
    graph triplets, judges) where a partial result beats none. NEVER set
    it when parsing tool-call ARGUMENTS: repairing a truncated
    ``{"path": …, "content": …`` would execute the tool with half its
    payload instead of triggering the parse-error retry loop.

    Distinguishing "no JSON present" from "JSON was malformed" is useful for
    debugging the planner / smart memory / post-mortem flows. We log a
    WARNING (not DEBUG) when the input clearly *contained* something that
    looked like JSON but could not be parsed, so the operator notices
    silent extraction failures in production logs.
    """
    import re, json, ast
    # Contract: return {} on EVERY failure mode. A non-str input (e.g. a
    # backend that returned `content: null`) must not raise out of the
    # re.sub below, which runs before the try/except guard.
    if not isinstance(text, str):
        return {}
    # Qwen Syntax Healing: Fix {"name"="tool"...} or {"name"= "tool"...} hallucinations.
    # Also heal `"key"=` → `"key":` for ANY field, not just `name` (the previous
    # version only fixed the `name` field).
    text = re.sub(r'(?<=[{,\s])"([\w_]+)"\s*=\s*', r'"\1": ', text)

    # Match a quoted string literal OR a bare JSON keyword. Strings are the
    # first alternative, so a `true`/`false`/`null` INSIDE a string is
    # consumed as part of the string match and never rewritten — the old
    # `\btrue\b` sweep corrupted keyword-like words inside value strings
    # despite the comment claiming otherwise.
    _JSON_KW_RE = re.compile(
        r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?P<kw>true|false|null)\b'''
    )
    _KW_MAP = {"true": "True", "false": "False", "null": "None"}

    def _kw_sub(m):
        kw = m.group("kw")
        return _KW_MAP[kw] if kw is not None else m.group(0)

    def _parse(t):
        try:
            return json.loads(t, strict=False)
        except json.JSONDecodeError:
            try:
                # AST Fallback for models that output Python dicts instead of
                # strict JSON. Only the standalone keywords are rewritten;
                # occurrences inside string literals are left intact.
                pt = _JSON_KW_RE.sub(_kw_sub, t)
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
        if start != -1:
            # An opening brace alone counts: output truncated at max_tokens
            # often has NO closing brace, and the old `end != -1` gate
            # skipped exactly those.
            looked_like_json = True
            if end > start:
                p = _parse(text[start:end + 1])
                if p:
                    return p

        result = _parse(text)
        if not result and repair_truncated and start != -1:
            # TRUNCATION REPAIR (opt-in): background consolidation calls
            # can hit their max_tokens cap mid-object. Salvage every
            # complete key/value pair instead of dropping the extraction.
            repaired = _repair_truncated_json(text[start:])
            if repaired:
                logger.info("extract_json_from_text: salvaged truncated JSON "
                            f"({len(repaired)} top-level keys)")
                return repaired
        if not result and looked_like_json:
            # We saw braces but couldn't parse them — that's an extraction
            # failure worth surfacing, not a "no JSON here" non-event.
            preview = text[:200].replace("\n", " ")
            logger.warning(f"extract_json_from_text: malformed JSON-like content failed to parse. Preview: {preview}")
        return result
    except Exception as e:
        logger.warning(f"extract_json_from_text raised {type(e).__name__}: {e}")
        return {}


_CODING_KEYWORDS = [r"\bpython\b", r"\bbash\b", r"\bsh\b", r"\bscript\b", r"\bcode\b", r"\bdef\b", r"\bimport\b", r"\bhtml\b", r"\bcss\b", r"\bjs\b", r"\bjavascript\b", r"\btypescript\b", r"\breact\b", r"\bweb\b", r"\bfrontend\b"]
_CODING_ACTIONS = [r"\bwrite\b", r"\brun\b", r"\bexecute\b", r"\bdebug\b", r"\bfix\b", r"\bcreate\b", r"\bgenerate\b", r"\bcount\b", r"\bcalculate\b", r"\banalyze\b", r"\bscrape\b", r"\bplot\b", r"\bgraph\b", r"\bbuild\b", r"\bdevelop\b"]
_DBA_KEYWORDS = [r"\bsql\b", r"\bpostgres\b", r"\bpostgresql\b", r"\bpsql\b", r"\bdatabase\b", r"\bpg_stat\b", r"\bexplain analyze\b", r"\bquery\b", r"\bcte\b", r"\brdbms\b", r"\bdba\b", r"\bschema\b", r"\bvacuum\b", r"\bmvcc\b"]
_META_KEYWORDS = [r"\btitle\b", r"\bname this\b", r"\brename\b", r"\bsummary\b", r"\bsummarize\b", r"\bcaption\b", r"\bdescribe\b"]
# Pushback markers for the follow-up inheritance rule below. Deliberately
# narrow: questions ("why did you…") and acknowledgements must NOT match.
_CORRECTION_MARKERS = [
    r"\bwrong\b", r"\bincorrect\b", r"\bnot optimal\b", r"\bmistake\b",
    r"\bbroken\b", r"\bdoesn'?t work\b", r"\bnot working\b",
    r"\byou forgot\b", r"\byou(?:'re| are) not using\b",
]


def detect_coding_intent(lc: str, messages: Optional[List[Dict]] = None) -> tuple:
    """Classify the current user turn: ``(has_coding_intent, is_meta_task)``.

    `lc` is the lowercased last user message. `messages` (the full chat
    history) powers the FOLLOW-UP INHERITANCE rule: a correction of a
    prior coding/SQL answer ("that is wrong — the insert belongs after
    the swap") usually carries no coding keyword of its own, so it used
    to fall through to the conversational sampling profile (temp 1.0)
    and silently drop the specialist persona mid-task. If the user is
    pushing back and the previous assistant turn shipped a fenced code
    block, the turn stays in coding mode.
    """
    has_coding_intent = False
    if any(re.search(k, lc) for k in _CODING_KEYWORDS):
        if any(re.search(a, lc) for a in _CODING_ACTIONS):
            has_coding_intent = True
    if any(ext in lc for ext in [".py", ".js", ".html", ".css", ".ts", ".tsx", ".jsx", ".sh"]) or re.search(r'\bscript\b', lc):
        has_coding_intent = True
    if any(re.search(k, lc) for k in _DBA_KEYWORDS):
        has_coding_intent = True

    is_meta_task = any(re.search(k, lc) for k in _META_KEYWORDS)
    if re.match(r'^[\d\s\+\-\*\/\(\)\=\?]+$', lc):
        has_coding_intent = False

    if not has_coding_intent and not is_meta_task and messages:
        if any(re.search(k, lc) for k in _CORRECTION_MARKERS):
            prev_assistant = next(
                (m.get("content", "") for m in reversed(messages[:-1])
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                "",
            )
            if isinstance(prev_assistant, str) and "```" in prev_assistant:
                has_coding_intent = True

    return has_coding_intent, is_meta_task


async def _timed_tool_coro(coro, sink: list, idx: int):
    """Await `coro`, writing its wall-clock duration into ``sink[idx]``.

    Lets the parallel tool-dispatch path capture a per-tool duration (feeding
    the metacog runtime-budget anomaly window) without changing how results
    are gathered. Never lets the timing bookkeeping affect the result.
    """
    import time as _t
    _t0 = _t.monotonic()
    try:
        return await coro
    finally:
        try:
            sink[idx] = _t.monotonic() - _t0
        except Exception:
            pass


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
        # Workspace continuity facade — world-model counterpart to
        # selfhood. Populated by main.py during lifespan; left as None
        # otherwise so callers that touch ``context.workspace_model``
        # before initialisation see a clean None rather than AttributeError.
        self.workspace_model = None
        # Most-recent user message for the current turn. Stashed by
        # handle_chat right after it extracts it from the request body so
        # tools can inspect user intent without re-parsing the message
        # list. Used by `tools.memory.tool_self_play*` to refuse calls the
        # LLM hallucinated mid-session — the watchdog's biological
        # self-play path does NOT go through the tool layer, so clearing
        # this field is harmless for internal firing. Empty string when
        # no user turn is in flight (e.g. background tasks).
        self.last_user_content = ""
        # Metacog uplift bundle. None when ``--enable-metacog`` is off
        # (the default), keeping legacy behaviour unchanged. main.py's
        # lifespan populates this just after llm_client is wired so the
        # arbiter's runner/embedder have a live upstream to talk to.
        self.metacog = None
        # Most-recent composite confidence reading (core.confidence).
        # Set per tool-turn under --enable-metacog; read at turn end by
        # the calibration spine to pair confidence with outcome.
        self.last_confidence = None
        self.last_entropy_reading = None
        # Calibration spine (core.calibration). Populated by main.py
        # lifespan; pairs confidence with realized outcome, measures
        # Brier/ECE, and re-fits τ/weights/λ in idle phase 2.7c.
        self.calibration_tracker = None
        # Holds the turn's last confidence reading until the turn's
        # outcome is known (then recorded + cleared). Per-turn, not
        # cross-turn — set at the confidence-compute site.
        self._calib_pending = None

# Async-critic deferred-correction bounds (see _record_late_verdict /
# _consume_pending_corrections). The agent is a process-wide singleton, so
# these keep a late REFUTED verdict from one conversation leaking into an
# unrelated one and from accumulating without bound:
#   * MAX — hard cap on queued corrections (newest win) so a busy multi-
#     conversation process can't pile up a runaway banner chain.
#   * TTL — a correction that its OWN conversation never returns to consume is
#     dropped after this many seconds instead of lingering forever.
_CORRECTION_MAX = 3
_CORRECTION_TTL = 900.0  # seconds

# Promised-notification finish-line guard (2026-07-13, req 11fe11d8): the
# user asked "notify me in slack when you're done", the model PLANNED the
# notify_operator call in its reasoning, then emitted the final response
# without ever calling the tool — the one explicit delivery requirement in
# the request was silently dropped (and the verifier confirmed the turn:
# the deliverable itself was fine). These patterns detect an explicit
# notify-me ask in the USER text so the turn loop can steer once toward
# notify_operator before letting the final response ship. Kept narrow on
# purpose — a false fire injects a bogus SYSTEM ALERT into casual chat.
_NOTIFY_INTENT_RE = re.compile(
    # Direct ask: "notify me", "ping me", "alert me".
    r"\b(?:notify|ping|alert)\s+me\b"
    # Communication verb bound to a Slack destination within one clause:
    # "let me know / tell me / report back / update me / message me /
    # dm me … in|on|via|over slack". A bare "in slack" or "slack message"
    # deliberately does NOT match — questions ABOUT Slack ("how do I
    # format a slack message?") must not arm the guard.
    r"|\b(?:let\s+me\s+know|tell\s+me|report\s+back|update\s+me|"
    r"message\s+me|dm\s+me|write\s+me)\b[^.!?\n]{0,40}?"
    r"\b(?:in|on|via|over|through)\s+slack\b"
    # "send me a notification / dm / slack message".
    r"|\bsend\s+me\s+a\s+(?:notification|dm|slack\s+(?:message|dm|notification))\b",
    re.IGNORECASE,
)
_NOTIFY_NEGATION_RE = re.compile(
    r"\b(?:don'?t|do\s+not|no\s+need\s+to|without|stop)\s+"
    r"(?:notify(?:ing)?|ping(?:ing)?|alert(?:ing)?|messag(?:e|ing))",
    re.IGNORECASE,
)


def _user_asked_for_notification(user_text) -> bool:
    """True when the user's request explicitly asks for an out-of-band
    notification ("notify me in slack when you're done") and doesn't
    negate it ("don't notify me"). Long pasted documents are truncated
    before matching so an incidental 'slack' deep inside one can't arm
    the guard."""
    t = str(user_text or "")[:4000]
    if not t:
        return False
    if _NOTIFY_NEGATION_RE.search(t):
        return False
    return bool(_NOTIFY_INTENT_RE.search(t))


# Trailing-promise detection (2026-07-14). A final reply whose LAST sentence
# promises imminent action ("…Let me fix it.") means the model narrated an
# action instead of doing it — the turn ends and nothing runs afterwards
# (observed live: a mid-repair turn shipped exactly that, the user believed
# the fix was applied). "let me know…" is conversational and excluded.
_ACTION_PROMISE_RE = re.compile(
    r"\b(?:let\s+me\s+(?!know\b)|i['’]ll\s+|i\s+will\s+"
    r"|i\s+am\s+going\s+to\s+|now\s+i['’]m\s+going\s+to\s+|gonna\s+)",
    re.IGNORECASE,
)


def _ends_with_action_promise(text) -> str:
    """Return the final sentence of ``text`` when it promises imminent
    action, else "". Only the LAST sentence counts — promises earlier in a
    reply are usually followed by the action's actual result."""
    t = str(text or "").strip()
    if not t:
        return ""
    last_sentence = re.split(r"(?<=[.!?])\s+|\n+", t)[-1].strip()
    if not last_sentence or len(last_sentence) > 120:
        return ""
    if _ACTION_PROMISE_RE.search(last_sentence):
        return last_sentence
    return ""


#: Tools whose dropped-at-the-finish-line call means user-visible work was
#: silently skipped (vs. terminal-tool re-calls, which are dropped by
#: design). Used for the honesty note when force_final_response eats a call.
_MUTATING_TOOLS_FOR_DROP_NOTE = frozenset({
    "file_system", "execute", "manage_services", "manage_projects", "database",
})


def _dropped_mutation_note(dropped_names) -> str:
    """Return the not-applied honesty note when ``dropped_names`` contains a
    mutating tool, else "". Appended to the final reply so a fix the turn
    closure swallowed is never presented as done."""
    muts = sorted({str(n) for n in (dropped_names or [])
                   if str(n) in _MUTATING_TOOLS_FOR_DROP_NOTE})
    if not muts:
        return ""
    return (
        "\n\n⚠ Note: this turn was finalized before my pending "
        f"{', '.join(muts)} action(s) could run — any change described "
        "above as about to happen has NOT been applied yet. Ask me to "
        "continue to apply it."
    )


@dataclass
class TurnState:
    """Turn-loop state crossing the `_dispatch_and_process_tool_batch`
    boundary (#5 decomposition, step 2 — designed from an AST capture
    analysis of handle_chat; see PROJECT_JOURNAL.md §4A #5).

    Two kinds of fields:
    * MUTATED_FIELDS — scalars the pipeline rebinds; the method repacks
      them in a `finally` so handle_chat sees every update even when a
      tool path raises. Several are cross-ITERATION state that never
      leaves the region in one pass (e.g. `_request_sys3_fired_once`,
      the once-per-request SYSTEM-3 latch): they live here because the
      method frame dies per call, while handle_chat's frame used to
      carry them between turns.
    * the rest — read-only values and mutable containers (mutated
      in place through the reference; never rebound).
    """
    _constraint_steer_pending: Any
    _proj_task_closed_this_req: Any
    _request_sys3_fired_once: Any
    _request_sys3_prev_justification: Any
    consecutive_parse_errors: Any
    current_plan_json: Any
    execution_failure_count: Any
    final_ai_content: Any
    fname: Any
    force_final_response: Any
    force_stop: Any
    forget_was_called: Any
    last_was_failure: Any
    preflight_blocks_this_request: Any
    request_sandbox_state: Any
    transient_failure_count: Any
    tool_calls: Any
    msg: Any
    ui_content: Any
    parse_failure_reason: Any
    model: Any
    last_user_content: Any
    char_budget: Any
    strikes: Any
    task_tree: Any
    _user_batch_intent: Any
    _request_constraints: Any
    repeated_action_steered: Any
    messages: Any
    seen_tools: Any
    executed_idempotent: Any
    raw_tools_called: Any
    tool_usage: Any
    tools_run_this_turn: Any
    request_state: Any

    MUTATED_FIELDS = ('_constraint_steer_pending', '_proj_task_closed_this_req', '_request_sys3_fired_once', '_request_sys3_prev_justification', 'consecutive_parse_errors', 'current_plan_json', 'execution_failure_count', 'final_ai_content', 'fname', 'force_final_response', 'force_stop', 'forget_was_called', 'last_was_failure', 'preflight_blocks_this_request', 'request_sandbox_state', 'transient_failure_count')


@dataclass
class FinalizeState:
    """Read-only inputs to `_finalize_and_return` (#5 decomposition,
    step 3). Unlike `TurnState` there are NO mutated fields: the
    finalization chain is the tail of handle_chat — nothing after it
    reads these locals, and its own `return` is the method return.
    """
    body: Any
    created_time: Any
    current_trajectory_id: Any
    execution_failure_count: Any
    final_ai_content: Any
    force_stop: Any
    forget_was_called: Any
    last_user_content: Any
    last_was_failure: Any
    lc: Any
    messages: Any
    model: Any
    payload: Any
    req_id: Any
    thought_content: Any
    tools_run_this_turn: Any
    was_complex_task: Any
    _stable_conv_fp: Any
    _verdict_is_fresh: Any
    _verifier_verdict_cache: Any


class GhostAgent:
    def _rebuild_available_tools(self):
        """Rebuild the dispatch dict from the registry after a lookup miss
        (models hallucinate tool-name variants). CONTAINMENT: a restricted
        sub-agent sets ``context._subagent_allowed_tools``; the rebuild MUST
        re-narrow to that allowlist, otherwise a dispatch miss heals the dict
        back to the full registry and a delegate reaches
        delegate/jobs/manage_* despite the allowlist (2026-07-14 bug hunt)."""
        rebuilt = get_available_tools(self.context)
        _allow = getattr(self.context, "_subagent_allowed_tools", None)
        if _allow is not None:
            rebuilt = {k: v for k, v in rebuilt.items() if k in _allow}
        self.available_tools = rebuilt
        return self.available_tools

    def __init__(self, context: GhostContext):
        self.context = context
        self.disabled_tools = set()
        # Corrections queued by a previous turn's async verdict (GHOST_CRITIC_ASYNC),
        # surfaced at the top of the next turn. See _record_late_verdict /
        # _consume_pending_corrections.
        self._pending_corrections = []
        self._correction_active_this_turn = False
        self._active_correction = ""
        self.available_tools = get_available_tools(context)
        # Turns are SERIALIZED (was Semaphore(10)). Per-turn state lives on the
        # singleton context — `last_user_content` (read mid-turn by intent
        # gates in tools/projects.py + tools/memory.py) and `current_project_id`
        # (drives project-scoped sandbox writes) — so two concurrent turns
        # clobber each other's scope. The concrete hazard: an APScheduler cron
        # job fires `handle_chat` mid-user-turn and switches the active project,
        # landing the user's uploads/writes in the wrong sandbox with
        # plausible-looking logs. There is ONE upstream llama slot, so
        # concurrent turns mostly just interleave waits anyway — serializing
        # costs almost nothing and closes the whole state-corruption class.
        # (If real per-turn concurrency is ever wanted, move those two fields
        # to contextvars instead of raising this back up.)
        self.agent_semaphore = asyncio.Semaphore(1)
        self.memory_semaphore = asyncio.Semaphore(1)
        # Live pre-flight repeat-failure guard (feature 1A). Always present
        # and fed; only consulted as a hard pre-dispatch block when the flag
        # is on (default on). Persists across turns on the long-lived agent
        # instance, with a bounded window so stale failures age out.
        self._failure_guard = RecentFailureGuard()
        self._preflight_guard_enabled = bool(
            getattr(getattr(context, "args", None), "enable_preflight_guard", True)
        )
        # B3 idle-loop ablation knobs (IMPROVEMENTS.md #4). Default to
        # production timings (scale 1.0, probabilistic). The trackb3 harness
        # sets a large scale (e.g. 60) to compress hours-long idle windows into
        # minutes and --bio-deterministic to fire the phases every epoch.
        _args = getattr(context, "args", None)
        _bts = getattr(_args, "bio_time_scale", 1.0)
        # Only a REAL positive number scales; a MagicMock args (tests) or a
        # missing attr falls back to production timing 1.0.
        self._bio_time_scale = _bts if (isinstance(_bts, (int, float))
                                        and not isinstance(_bts, bool)
                                        and _bts > 0) else 1.0
        # Only a real bool True enables — a MagicMock attr must NOT (else every
        # test agent fires the idle phases deterministically).
        self._bio_deterministic = getattr(_args, "bio_deterministic", False) is True

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

    def _get_context_manager(self):
        """Lazily construct the progressive ContextManager (L0-L4 compression).
        Wired in front of _prune_context so deterministic compression runs
        before the LLM summarization prune (IMPROVEMENTS.md #27a). Uses the
        real token estimator so its ratio matches the rest of the loop."""
        cm = getattr(self, "_context_manager", None)
        if cm is None:
            from .context_manager import ContextManager
            max_tokens = int(getattr(self.context.args, "max_context", 65536) or 65536)

            def _estimator(msgs):
                total = 0
                for m in msgs:
                    c = m.get("content", "")
                    if isinstance(c, str):
                        total += estimate_tokens(c)
                    elif isinstance(c, list):
                        for part in c:
                            if isinstance(part, dict) and part.get("type") == "text":
                                total += estimate_tokens(str(part.get("text", "")))
                return total

            cm = ContextManager(max_tokens=max_tokens, token_estimator=_estimator)
            self._context_manager = cm
        return cm

    @staticmethod
    def _cap_oversized_tail(msgs: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Post-assembly budget enforcement for ``_prune_context``.

        The summarization pass keeps the last ~6 messages VERBATIM — so a
        tail holding several parallel whole-file reads rode through a
        "successful" prune untouched, and the post-prune request still hit
        the upstream at 333k tokens against a 262k n_ctx (HTTP 400 → failed
        recovery → dead turn; 2026-07-18 xrick feasibility session). This
        pass truncates the LARGEST non-system contents (head+tail kept,
        marker inserted) until the estimate fits inside ~92% of the budget,
        leaving headroom for the dynamic state appended later. Mutates the
        message dicts in place — shrinking the durable history is the point."""
        target = int(max_tokens * 0.92)

        def _tok(m):
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(str(b.get("text", "")) for b in c
                             if isinstance(b, dict) and b.get("type") == "text")
            return estimate_tokens(str(c))

        guard = 0
        while sum(_tok(m) for m in msgs) > target and guard < 64:
            guard += 1
            cand = None
            for m in msgs:
                if m.get("role") == "system":
                    continue
                c = m.get("content", "")
                if isinstance(c, str) and len(c) > 4000 and (
                        cand is None or len(c) > len(cand.get("content") or "")):
                    cand = m
            if cand is None:
                break
            c = cand["content"]
            keep = max(1500, len(c) // 3)
            half = keep // 2
            cand["content"] = (
                c[:half]
                + f"\n[... {len(c) - keep:,} chars dropped by context budget enforcement — "
                  "re-read the specific region with start_line/end_line if needed ...]\n"
                + c[-half:]
            )
        return msgs

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
            return self._cap_oversized_tail(
                system_msgs + [original_goal] + all_anchors + recent_context,
                max_tokens)

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
            # FOREGROUND, not background: this summarizer is awaited INLINE
            # by the user's own turn (handle_chat calls _prune_context when
            # history overflows). Marking it is_background made it park in
            # _wait_for_foreground_clear against its OWN request — a
            # deterministic stall up to the 600s ceiling before the turn
            # could continue. use_worker still offloads it when a pool exists.
            summary_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=False, task_label="compaction")
            summary += str(summary_data["choices"][0]["message"].get("content") or "No summary generated.")

            from ..utils.helpers import get_utc_timestamp
            if self.context.memory_system and "Summarization unavailable" not in summary:
                episode_text = f"EPISODIC ARCHIVE (Past Conversation Summary):\n{summary}"
                # Hold a reference to this fire-and-forget archive write:
                # asyncio keeps only a weak ref to bare tasks, so an
                # un-stored create_task can be GC'd mid-flight (the write
                # silently never lands). Tracking it in a context-level set
                # also lets the lifespan shutdown drain it instead of
                # destroying it mid-await.
                _bg = getattr(self.context, "_pending_background_tasks", None)
                if _bg is None:
                    _bg = set()
                    self.context._pending_background_tasks = _bg
                _arch_task = asyncio.create_task(asyncio.to_thread(
                    self.context.memory_system.add,
                    episode_text,
                    {"type": "episode", "timestamp": get_utc_timestamp()}
                ))
                _bg.add(_arch_task)
                _arch_task.add_done_callback(_bg.discard)
        except Exception as e:
            logger.warning(f"Context summarization failed: {e}")
            summary += f"(Summarization unavailable due to error, dropping old turns: {e})"

        # Insert the summary BEFORE the recent context (so the model reads
        # the fresh stuff last), re-attach anchored findings and the tool
        # result we hoisted out of the middle.
        all_anchors = anchor_messages + ([recent_tool_anchor] if recent_tool_anchor else [])
        return self._cap_oversized_tail(
            system_msgs + [original_goal, {"role": "assistant", "content": summary}]
            + all_anchors + recent_context,
            max_tokens)

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
                # RSS self-defense runs BEFORE the tick and independent of its
                # memory_system guard: the known 270MB→2GB growth on a ~94%-RAM
                # box can OOM-kill the shared llama-server, so it must fire even
                # in a degraded boot. Only acts when quiescent.
                try:
                    self._rss_watchdog_check()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"RSS watchdog check failed: {e}")
                try:
                    await self._biological_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Biological watchdog tick failed: {e}")
        except asyncio.CancelledError:
            logger.info("Biological watchdog daemon cancelled")
            raise

    def _current_rss_mb(self):
        """Process RSS in MB, or None if psutil is unavailable / errors.
        Cheap enough to call once per 60s tick."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    def _rss_watchdog_check(self):
        """When the process RSS crosses GHOST_MAX_RSS_MB and no foreground work
        is in flight, log one WARNING and restart in place via os.execv —
        preserving argv + the exported GHOST_HOME, no external supervisor
        needed. Disabled when GHOST_MAX_RSS_MB<=0 (the default 0 = off, so
        this is opt-in and cannot surprise-restart an unconfigured deployment).
        A shutdown drain hook isn't available here, so we only fire when
        quiescent — mid-turn RSS is never interrupted."""
        import os
        try:
            limit_mb = float(os.environ.get("GHOST_MAX_RSS_MB", "0") or "0")
        except (TypeError, ValueError):
            limit_mb = 0.0
        if limit_mb <= 0:
            return  # opt-in; off by default

        rss = self._current_rss_mb()
        if rss is None or rss < limit_mb:
            return

        # Only restart when genuinely idle — never mid-request.
        _lc = getattr(self.context, "llm_client", None)
        if (getattr(_lc, "foreground_tasks", 0) > 0
                or getattr(_lc, "foreground_requests", 0) > 0):
            self._rss_over_since = getattr(self, "_rss_over_since", None) or "pending"
            logger.warning(
                "RSS %.0f MB over limit %.0f MB but foreground work is active; "
                "deferring controlled restart until idle.", rss, limit_mb,
            )
            return

        pretty_log(
            "RSS Watchdog",
            f"Process RSS {rss:.0f} MB exceeded GHOST_MAX_RSS_MB={limit_mb:.0f} "
            f"while idle — restarting in place (os.execv) to reclaim memory.",
            level="WARNING", icon=Icons.WARN,
        )
        try:
            import sys
            # Best-effort: stop the sandbox container cleanly so the restart
            # doesn't orphan it. Other resources are reclaimed by the exec.
            sm = getattr(self.context, "sandbox_manager", None)
            if sm is not None:
                try:
                    sm.close(remove=False)
                except Exception:
                    pass
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable, "-m", "ghost_agent.main", *sys.argv[1:]])
        except Exception as e:
            logger.error(f"RSS-watchdog restart failed (continuing): {e}")

    # Per-phase cooldowns (in seconds) so the watchdog can't fire two REM
    # cycles or two self-play sessions back-to-back when the user remains AFK.
    _DREAM_COOLDOWN = 1800        # 30 min between dreams
    _REFLECTION_COOLDOWN = 2400   # 40 min between reflections
    _POSTMORTEM_COOLDOWN = 10800  # 3 h between whole-transcript post-mortems (phase 2.5c)
    _SKILLS_AUTO_COOLDOWN = 7200  # 2 hours between skill auto-extractions
    _PRM_TRAIN_COOLDOWN = 10800   # 3 hours between PRM retrain passes
    _NARRATIVE_COOLDOWN = 3600    # 60 min between selfhood-narrative consolidations
    _WORKSPACE_NARRATIVE_COOLDOWN = 3600  # 60 min between workspace-narrative consolidations (phase 2.9)
    _STALE_QUESTIONS_COOLDOWN = 7200  # 2 h between stale open-question surfacings (phase 2.8b)
    _ROUTER_TRAIN_COOLDOWN = 10800   # 3 h between router-classifier retrains (phase 2.7b)
    _CALIB_REFIT_COOLDOWN = 3600  # 60 min between calibration refits (phase 2.7c)
    _WORKSPACE_TIDY_COOLDOWN = 21600  # 6 h between recurring workspace tidy passes (phase 2.7d)
    _AUTOADVANCE_COOLDOWN = 1800  # 30 min between autonomous project-advance ticks (phase 2.95)
    _SELFPLAY_COOLDOWN = 3600     # 60 min between self-plays
    # Belt-and-braces guard for phase 1. The journal-empty self-disarm
    # already prevents same-batch refire, but a journal write that
    # raises mid-loop (or a misbehaving consumer that fails to drain)
    # would otherwise re-fire every tick. The cooldown caps refire
    # rate at the same shape as the other five phases.
    _JOURNAL_COOLDOWN = 60        # 60 s between journal-process passes

    def _bio_scaled(self, seconds: float) -> float:
        """Scale an idle-window / cooldown threshold by ``--bio-time-scale``
        (default 1.0 = production timings). A scale of 60 turns a 1-hour idle
        window into 1 minute so the B3 ablation harness can exercise the pure-
        idle learning loops (dream/self-play, reflection critique, skills-auto
        graduation) in accelerated epochs. Applied to EVERY inline window bound
        and to the phase cooldowns via `_bio_cooldown`."""
        scale = getattr(self, "_bio_time_scale", 1.0) or 1.0
        return seconds / scale

    def _bio_cooldown(self, seconds: float) -> float:
        """Same scaling for a phase cooldown."""
        return self._bio_scaled(seconds)

    def _bio_roll(self, p: float) -> bool:
        """Probability gate for the idle phases. Returns True deterministically
        under ``--bio-deterministic`` (so the B3 control/treatment arms fire the
        same phases every accelerated epoch instead of sampling), else
        ``random.random() < p`` as before."""
        if getattr(self, "_bio_deterministic", False):
            return True
        return random.random() < p

    def _record_autonomous_activity(self, phase, summary,
                                    severity: str = "info", **meta) -> None:
        """Best-effort sink of an idle-phase outcome into the
        autonomous-activity ledger (core.autonomous_activity) so it reaches
        the operator via the next-turn digest and, for severity="notify",
        the outbound push transports. Fail-safe by contract: activity
        logging must never break a phase."""
        try:
            log = getattr(self.context, "activity_log", None)
            if log is not None:
                log.record(phase, summary, severity=severity, **meta)
        except Exception as e:  # noqa: BLE001
            logger.debug("autonomous-activity record skipped: %s", e)

    async def _biological_tick(self):
        """One pass of the biological hook state machine. Extracted from the
        loop for direct unit testing."""
        ctx = self.context
        if not getattr(ctx, 'memory_system', None):
            return

        # HARD LOCK: never interrupt an active LLM generation OR a user
        # request that is merely paused between its LLM calls.
        # `foreground_tasks` counts only an in-flight LLM call and momentarily
        # drops to 0 in the gaps of a live turn; `foreground_requests` spans
        # the whole request lifecycle (see llm.py and
        # `_wait_for_foreground_clear`, which waits on BOTH). Gating on tasks
        # alone let a tick start in that gap — and the synchronous CPU phases
        # below (skills-auto extract, PRM/router train) would then stall the
        # event loop mid-request.
        _lc = getattr(ctx, 'llm_client', None)
        if (getattr(_lc, 'foreground_tasks', 0) > 0
                or getattr(_lc, 'foreground_requests', 0) > 0):
            return

        # Lazily install per-phase cooldown anchors on the agent instance.
        if not hasattr(self, '_last_journal_at'):
            self._last_journal_at = datetime.datetime.min
        if not hasattr(self, '_last_dream_at'):
            self._last_dream_at = datetime.datetime.min
        if not hasattr(self, '_last_reflection_at'):
            self._last_reflection_at = datetime.datetime.min
        if not hasattr(self, '_last_postmortem_at'):
            self._last_postmortem_at = datetime.datetime.min
        if not hasattr(self, '_last_skills_auto_at'):
            self._last_skills_auto_at = datetime.datetime.min
        if not hasattr(self, '_last_prm_train_at'):
            self._last_prm_train_at = datetime.datetime.min
        if not hasattr(self, '_last_narrative_at'):
            self._last_narrative_at = datetime.datetime.min
        if not hasattr(self, '_last_workspace_narrative_at'):
            self._last_workspace_narrative_at = datetime.datetime.min
        if not hasattr(self, '_last_stale_questions_at'):
            self._last_stale_questions_at = datetime.datetime.min
        if not hasattr(self, '_last_router_train_at'):
            self._last_router_train_at = datetime.datetime.min
        # Corpus fingerprints from the last COMPLETED PRM / router refit.
        # When the trajectory corpus hasn't changed since, the refit would
        # reproduce the identical model — skip the pass entirely (the
        # 2026-07-17 overnight run refit both 3× on an unchanged corpus).
        if not hasattr(self, '_prm_corpus_fp'):
            self._prm_corpus_fp = None
        if not hasattr(self, '_router_corpus_fp'):
            self._router_corpus_fp = None
        if not hasattr(self, '_reflection_corpus_fp'):
            self._reflection_corpus_fp = None
        if not hasattr(self, '_last_workspace_tidy_at'):
            self._last_workspace_tidy_at = datetime.datetime.min
        if not hasattr(self, '_last_calib_refit_at'):
            self._last_calib_refit_at = datetime.datetime.min
        if not hasattr(self, '_last_autoadvance_at'):
            self._last_autoadvance_at = datetime.datetime.min
        if not hasattr(self, '_last_selfplay_at'):
            self._last_selfplay_at = datetime.datetime.min
        # Adaptive self-play cooldown (curiosity-driven). Starts at the
        # static baseline and is rewritten after each run based on the
        # FrontierTracker's last compression delta.
        if not hasattr(self, '_current_selfplay_cooldown'):
            self._current_selfplay_cooldown = self._SELFPLAY_COOLDOWN

        idle_secs = (datetime.datetime.now() - ctx.last_activity_time).total_seconds()

        # Phase 1: Process Short-Term Journal (>120s idle)
        if idle_secs > self._bio_scaled(120) and getattr(ctx, 'journal', None) is not None:
            since_last_journal = (datetime.datetime.now() - self._last_journal_at).total_seconds()
            if since_last_journal >= self._JOURNAL_COOLDOWN:
                has_items = False
                try:
                    # Route through the guarded load() (not a raw read_text +
                    # json.loads): on a corrupt journal, load() sidecars the
                    # bytes and returns [] rather than letting `except: pass`
                    # silently skip consolidation forever.
                    has_items = len(ctx.journal.load()) > 0
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
        if self._bio_scaled(600) < idle_secs <= self._bio_scaled(3600):
            since_last_dream = (datetime.datetime.now() - self._last_dream_at).total_seconds()
            if since_last_dream >= self._DREAM_COOLDOWN:
                try:
                    res = await asyncio.to_thread(
                        ctx.memory_system.collection.get,
                        where={"type": "auto"},
                        limit=5
                    )
                except Exception as _cgx:
                    # A ChromaDB error / locked store must only skip the dream
                    # eligibility check, not abort the whole tick (which would
                    # starve phases 2.5–3 for this idle session).
                    logger.debug("dream eligibility get() failed: %s", _cgx)
                    res = None
                _dream_eligible = bool(res and len(res.get('ids', [])) >= 3)
                # isinstance gate: only consult the trajectory fallback when
                # the auto-pool check ran against a REAL store (dict result) —
                # a chroma error (res=None) shouldn't silently reroute to
                # trajectories.
                if not _dream_eligible and isinstance(res, dict):
                    # Trajectory fallback (2026-07-09): the auto-memory pool
                    # is organically unsatisfiable (0 across 12 instrumented
                    # arm-runs — journal §6); dream() itself falls back to
                    # trajectory digests, so the ELIGIBILITY gate must match
                    # or the fallback is dead code. The probe is a cheap
                    # line-count (no JSON parse) — this runs every tick.
                    try:
                        from .dream import trajectory_seed_available as _tsa
                        _dream_eligible = await asyncio.to_thread(_tsa, ctx)
                    except Exception as _tfx:
                        logger.debug("dream trajectory eligibility failed: %s", _tfx)
                if not _dream_eligible and isinstance(res, dict):
                    # Self-play fallback (2026-07-19): dream() also seeds
                    # from frontier-tracker outcome digests — same
                    # gate-must-match reasoning as the trajectory probe.
                    try:
                        from .dream import selfplay_dream_fragments as _spf
                        _sp_ids, _ = await asyncio.to_thread(_spf, ctx)
                        _dream_eligible = len(_sp_ids) >= 3
                    except Exception as _spx:
                        logger.debug("dream self-play eligibility failed: %s", _spx)
                if _dream_eligible:
                    if self._bio_roll(0.5):
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
                            self._record_autonomous_activity(
                                "dream",
                                "REM cycle ran (memory consolidation / heuristic harvest)")
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
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            since_last_reflection = (datetime.datetime.now() - self._last_reflection_at).total_seconds()
            if since_last_reflection >= self._REFLECTION_COOLDOWN:
                reflector = getattr(ctx, 'reflector', None)
                traj_collector = getattr(ctx, 'trajectory_collector', None)
                # Skip-if-unchanged gate (2026-07-18): when the trajectory
                # corpus is byte-identical to the previous tick's AND that
                # tick reflected nothing new, this tick will deterministically
                # walk the same failures into the same dedup set again
                # (overnight log: eight consecutive "reflected 0/60,
                # dup-skipped 60" cycles). One stat pass replaces a full
                # JSONL parse of every day file. Any corpus change — or a
                # tick that DID reflect something — re-arms the phase.
                _refl_corpus_fp = None
                if reflector is not None and traj_collector is not None:
                    try:
                        _refl_fp_fn = getattr(traj_collector, 'corpus_fingerprint', None)
                        if callable(_refl_fp_fn):
                            _refl_corpus_fp = await asyncio.to_thread(_refl_fp_fn)
                    except Exception:
                        _refl_corpus_fp = None
                    if (_refl_corpus_fp is not None
                            and _refl_corpus_fp == getattr(self, '_reflection_corpus_fp', None)):
                        self._last_reflection_at = datetime.datetime.now()
                        logger.debug(
                            "Reflection tick skipped: corpus unchanged since "
                            "an all-duplicate pass (%s)", _refl_corpus_fp,
                        )
                        reflector = None  # falls through the phase gate below
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
                        # Loaded from disk once so restarts don't re-reflect
                        # the oldest failures (the loop progresses through the
                        # backlog instead of redoing work every boot).
                        already = self._get_reflected_ids()
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
                        # Persist the advanced dedup set so the progress
                        # survives the next restart.
                        self._persist_reflected_ids()
                        # Arm the skip gate ONLY after a do-nothing pass
                        # (everything dup-skipped, no errors): the next
                        # tick can then bail on the fingerprint alone.
                        # Errors stay re-armed — they may be transient.
                        if (report.reflected_ok == 0
                                and report.reflected_errors == 0):
                            self._reflection_corpus_fp = _refl_corpus_fp
                        else:
                            self._reflection_corpus_fp = None
                        pretty_log(
                            "Biological Hook",
                            f"Reflection complete: {report.summary()}",
                            icon=Icons.BRAIN_THINK,
                        )
                        if getattr(report, 'outcomes', None):
                            self._record_autonomous_activity(
                                "reflection", report.summary())
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

        # Phase 2.5c: Whole-transcript post-mortem (opt-in --postmortem).
        # Where phase 2.5 turns one failed turn into a behavioural
        # lesson, this reads the FULL tool-call sequence of the worst
        # recent failures and files a classified, durable defect report
        # (behavioural / configuration / code_defect) — the autonomous
        # form of the manual "evaluate the last N bad runs" pass. Runs at
        # a longer cadence than reflection (it's heavier — whole
        # transcripts, optional patch-proposal calls) and only when the
        # engine was wired in main.py (gated behind --postmortem). Same
        # anchor-before-await discipline as every other phase so a crash
        # can't pin it re-firing. Never resets last_activity_time; never
        # applies anything — output is a review queue.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            engine = getattr(ctx, 'postmortem_engine', None)
            traj_collector = getattr(ctx, 'trajectory_collector', None)
            if engine is not None and traj_collector is not None:
                # Read the configurable cooldown only once we know the
                # engine is wired — keeps this phase a true no-op (no
                # arg access) on deployments without --postmortem, and
                # the float() coercion tolerates a mocked args object.
                try:
                    pm_cooldown = float(getattr(
                        getattr(ctx, 'args', None),
                        'postmortem_cooldown', self._POSTMORTEM_COOLDOWN,
                    ))
                except (TypeError, ValueError):
                    pm_cooldown = float(self._POSTMORTEM_COOLDOWN)
                since_last_pm = (datetime.datetime.now() - self._last_postmortem_at).total_seconds()
                if since_last_pm >= pm_cooldown:
                    pretty_log(
                        "Biological Hook",
                        "Agent is idle. Running post-mortem on worst recent failures...",
                        icon=Icons.BRAIN_THINK,
                    )
                    self._last_postmortem_at = datetime.datetime.now()
                    try:
                        pm_report = await engine.run(
                            source=lambda: traj_collector.iter_trajectories(),
                        )
                        pretty_log(
                            "Biological Hook",
                            pm_report.summary(),
                            icon=Icons.BRAIN_THINK,
                        )
                        if getattr(pm_report, 'queued', 0):
                            self._record_autonomous_activity(
                                "postmortem",
                                f"{pm_report.queued} defect report(s) filed "
                                f"for review: {pm_report.summary()}")
                    except Exception as e:
                        logger.warning(f"Post-mortem phase failed: {e}")
                    finally:
                        self._last_postmortem_at = datetime.datetime.now()

        # Phase 2.6: Skill auto-extraction (every ~2 hours during idle).
        # Pure data-level pass — no LLM call, no network, CPU-only —
        # so it's safe to run opportunistically whenever the idle
        # window covers it. Gated on `trajectory_collector` presence
        # (nothing to extract from without it) and on its own cooldown
        # anchor so a long AFK stretch doesn't produce N redundant
        # extraction passes back-to-back.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            since_last_skills = (datetime.datetime.now() - self._last_skills_auto_at).total_seconds()
            if since_last_skills >= self._SKILLS_AUTO_COOLDOWN:
                traj_collector = getattr(ctx, 'trajectory_collector', None)
                if traj_collector is not None:
                    self._last_skills_auto_at = datetime.datetime.now()
                    try:
                        from ..skills_auto import (
                            extract_candidates, consolidate, verify_candidate,
                        )
                        # Offload the O(corpus) sync work to a thread: on a
                        # large trajectory log the iterate + extract + consolidate
                        # pipeline stalls the event loop for seconds otherwise
                        # (matching the to_thread pattern phases 2.7c/2.95 use).
                        trajs = await asyncio.to_thread(
                            lambda: list(traj_collector.iter_trajectories())
                        )
                        if trajs:
                            candidates, report = await asyncio.to_thread(
                                extract_candidates, trajs, min_support=2
                            )
                            if candidates:
                                consolidated, _ = await asyncio.to_thread(
                                    consolidate, candidates
                                )
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
                                                _persisted = _store.graduate(
                                                    _cand,
                                                    confidence=_vr.updated_confidence,
                                                )
                                                if _persisted is None:
                                                    # Overflow-trimmed by the store
                                                    # (lowest confidence) — not really
                                                    # graduated; skip count + macro mint.
                                                    continue
                                                _graduated += 1
                                                # Also mint the proven sequence
                                                # as a composed-skill MACRO so a
                                                # graduated skill becomes a
                                                # dispatchable unit, not just
                                                # prose (redesign #8). Status
                                                # "proposed": auto-activation
                                                # needs per-step runtime arg
                                                # templates the name-only
                                                # candidate doesn't carry yet, so
                                                # we surface it for activation
                                                # rather than risk a param-less
                                                # active tool. Best-effort.
                                                try:
                                                    from ..tools.composed_skills import _registry_from_context
                                                    _reg = _registry_from_context(ctx)
                                                    _seq = getattr(_cand, 'tool_sequence', ()) or ()
                                                    if _reg is not None and _seq:
                                                        _steps = [
                                                            {"tool": _t, "description": "", "params": {}}
                                                            for _t in _seq
                                                        ]
                                                        _reg.compile_from_pattern(
                                                            getattr(_cand, 'name', 'skill') or 'skill',
                                                            _steps,
                                                            f"Proven {len(_seq)}-step sequence graduated "
                                                            f"from {getattr(_cand, 'support', 0)} successful runs",
                                                            status="proposed",
                                                            execution_mode="sequential",
                                                        )
                                                except Exception as _ce:
                                                    logger.debug(
                                                        "composed-macro mint skipped: %s", _ce)
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
                                if _graduated:
                                    self._record_autonomous_activity(
                                        "skills_auto",
                                        f"graduated {_graduated} proven skill(s) "
                                        f"from {report.n_trajectories_seen} "
                                        f"trajectories")
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
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
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
                    # Skip-if-unchanged gate: stat-level corpus fingerprint.
                    # Same-fingerprint ⇒ identical training data ⇒ the refit
                    # is a guaranteed no-op — don't burn the pass or log a
                    # "refit" line that implies fresh signal.
                    _corpus_fp = None
                    try:
                        _fp_fn = getattr(traj_collector, 'corpus_fingerprint', None)
                        if callable(_fp_fn):
                            _corpus_fp = await asyncio.to_thread(_fp_fn)
                    except Exception:
                        _corpus_fp = None
                    if _corpus_fp is not None and _corpus_fp == self._prm_corpus_fp:
                        pretty_log(
                            "PRM Retrain",
                            "skipped — trajectory corpus unchanged since last refit",
                            icon=Icons.SKIP,
                        )
                    else:
                        try:
                            from ..prm import PRMTrainer
                            save_path = getattr(ctx, '_prm_checkpoint_path', None)
                            if save_path is None:
                                base_mem = getattr(ctx, 'memory_dir', None)
                                if base_mem is not None:
                                    save_path = base_mem.parent / "prm" / "checkpoint.json"
                            trainer = PRMTrainer()
                            report = await asyncio.to_thread(
                                trainer.run,
                                trajectories=traj_collector.iter_trajectories(),
                                save_path=save_path,
                            )
                            # Record the fingerprint once the pass ran to
                            # completion (fit OR clean bail) — both are
                            # deterministic in the corpus, so re-running on
                            # the same bytes is pointless either way.
                            self._prm_corpus_fp = _corpus_fp
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
                                self._record_autonomous_activity(
                                    "prm_train",
                                    f"value model refit: {report.summary()}"
                                    f"{mcts_note}")
                            else:
                                logger.debug(
                                    "PRM idle retrain skipped: %s",
                                    report.bail_reason or "unknown",
                                )
                        except Exception as e:
                            logger.warning(f"PRM retrain phase failed: {e}")
                        finally:
                            self._last_prm_train_at = datetime.datetime.now()

        # Phase 2.7b: Router classifier retrain on accumulated trajectories.
        # Mirrors the PRM phase. The router ships UNTRAINED (escalates every
        # request); this trains the ComplexityClassifier from the trajectory log
        # and hot-swaps it into the live dispatcher so cheap turns stop waking
        # the full swarm. CPU-only, same idle window + cooldown discipline.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            _rt_cd_override = getattr(getattr(ctx, 'args', None), 'router_train_cooldown', None)
            if not isinstance(_rt_cd_override, (int, float)):
                _rt_cd_override = None
            _rt_cooldown = float(_rt_cd_override) if _rt_cd_override else float(self._ROUTER_TRAIN_COOLDOWN)
            since_last_router = (datetime.datetime.now() - self._last_router_train_at).total_seconds()
            if since_last_router >= _rt_cooldown:
                traj_collector = getattr(ctx, 'trajectory_collector', None)
                from ..distill.collector import TrajectoryCollector as _TrajColCls
                from ..router import ComplexityDispatcher as _ComplexityDispatcher
                dispatcher = getattr(ctx, 'complexity_dispatcher', None)
                if (isinstance(traj_collector, _TrajColCls)
                        and isinstance(dispatcher, _ComplexityDispatcher)):
                    self._last_router_train_at = datetime.datetime.now()
                    # Skip-if-unchanged gate — same rationale as the PRM
                    # phase above: identical corpus ⇒ identical classifier.
                    _rt_corpus_fp = None
                    try:
                        _rt_fp_fn = getattr(traj_collector, 'corpus_fingerprint', None)
                        if callable(_rt_fp_fn):
                            _rt_corpus_fp = await asyncio.to_thread(_rt_fp_fn)
                    except Exception:
                        _rt_corpus_fp = None
                    if (_rt_corpus_fp is not None
                            and _rt_corpus_fp == self._router_corpus_fp):
                        pretty_log(
                            "Router Retrain",
                            "skipped — trajectory corpus unchanged since last refit",
                            icon=Icons.SKIP,
                        )
                    else:
                        try:
                            from ..router import RouterTrainer
                            save_path = getattr(ctx, '_router_checkpoint_path', None)
                            if save_path is None:
                                base_mem = getattr(ctx, 'memory_dir', None)
                                if base_mem is not None:
                                    save_path = base_mem.parent / "router" / "checkpoint.json"
                            trainer = RouterTrainer()
                            report = await asyncio.to_thread(
                                trainer.run,
                                trajectories=traj_collector.iter_trajectories(),
                                save_path=save_path,
                            )
                            self._router_corpus_fp = _rt_corpus_fp
                            _new_clf = trainer.classifier
                            if (report.fit_succeeded and _new_clf is not None
                                    and _new_clf.is_finite()):
                                # Hot-swap: the dispatcher re-reads .classifier /
                                # .disabled on every route() call, so this takes
                                # effect immediately (no restart). The is_finite()
                                # gate is defence-in-depth — fit() already raises on
                                # divergence, but a NaN classifier must NEVER reach
                                # the live router (it would return NaN confidences).
                                dispatcher.classifier = _new_clf
                                dispatcher.disabled = False
                                pretty_log(
                                    "Router Retrain",
                                    f"classifier refit on idle: {report.summary()} · "
                                    f"router now routing (was escalate-all)",
                                    icon=Icons.BRAIN_PLAN,
                                )
                                self._record_autonomous_activity(
                                    "router_train",
                                    f"complexity router refit: {report.summary()}")
                            elif _new_clf is not None and not _new_clf.is_finite():
                                logger.warning(
                                    "Router idle retrain produced a non-finite model "
                                    "— NOT hot-swapping; router stays escalate-all."
                                )
                            else:
                                logger.debug("Router idle retrain skipped: %s", report.bail_reason or "unknown")
                        except Exception as e:
                            logger.warning(f"Router retrain phase failed: {e}")
                        finally:
                            self._last_router_train_at = datetime.datetime.now()

        # Phase 2.7c: Calibration refit (roadmap phase 2.5). Re-fits the
        # confidence threshold τ + entropy/competence weights + the
        # verbalised-uncertainty penalty λ on the accumulated
        # (confidence, outcome) log, minimising Brier, and hot-swaps the
        # result into the live CompositeConfidence — so the agent's
        # "I'm 80% sure" becomes empirically calibrated without a
        # restart. CPU-only, same idle window + cooldown-anchor
        # discipline as the PRM/router phases (anchor advanced BEFORE the
        # work AND in finally so a mid-fit crash can't refire every tick).
        # No-op unless --enable-metacog produced confidence readings AND
        # the bail floors (≥40 samples, both outcome classes) are met.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            _calib_cd_override = getattr(
                getattr(ctx, 'args', None), 'calib_refit_cooldown', None,
            )
            if not isinstance(_calib_cd_override, (int, float)):
                _calib_cd_override = None
            _calib_cooldown = (
                float(_calib_cd_override) if _calib_cd_override
                else float(self._CALIB_REFIT_COOLDOWN)
            )
            since_last_calib = (
                datetime.datetime.now() - self._last_calib_refit_at
            ).total_seconds()
            tracker = getattr(ctx, 'calibration_tracker', None)
            from .calibration import CalibrationTracker as _CalibTrackerCls
            if (since_last_calib >= _calib_cooldown
                    and isinstance(tracker, _CalibTrackerCls)):
                self._last_calib_refit_at = datetime.datetime.now()
                try:
                    params = await asyncio.to_thread(tracker.fit)
                    if params is not None:
                        from .metacog_log import (
                            emit as _mc_emit, Subsystem as _mc_ss,
                        )
                        mc = getattr(ctx, 'metacog', None)
                        if mc is not None and getattr(mc, 'confidence', None) is not None:
                            mc.confidence.apply_fitted(params)
                        _mc_emit(
                            _mc_ss.CALIB, refit="ok",
                            threshold=params.threshold,
                            w_entropy=params.w_entropy,
                            lam=params.lambda_uncertainty,
                            brier=params.brier, n=params.n_samples,
                        )
                        self._record_autonomous_activity(
                            "calibration",
                            f"confidence recalibrated "
                            f"(τ={params.threshold:.2f}, "
                            f"Brier={params.brier:.3f}, "
                            f"n={params.n_samples})")
                    else:
                        logger.debug("calibration refit produced no fit (thin/single-class)")
                except Exception as e:
                    logger.warning(f"Calibration refit phase failed: {e}")
                finally:
                    self._last_calib_refit_at = datetime.datetime.now()

        # Phase 2.7d: Recurring workspace tidy (2026-07-18). The DONE
        # sweep fires once, on the transition — but verification and
        # post-completion debugging keep producing screenshots and
        # scaffolding AFTER it (live case: six unswept screenshots on
        # the DONE game project by the next morning), which the
        # operator was deleting by hand. This phase walks every project
        # workspace and removes categorical debris + unregistered,
        # unreferenced media older than TIDY_MIN_AGE_HOURS. Narrow by
        # design: never touches source files or the keep-set — see
        # workspace_cleanup.tidy_project_workspace.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            since_last_tidy = (datetime.datetime.now()
                               - self._last_workspace_tidy_at).total_seconds()
            if since_last_tidy >= self._WORKSPACE_TIDY_COOLDOWN:
                _tidy_store = getattr(ctx, 'project_store', None)
                if _tidy_store is not None:
                    self._last_workspace_tidy_at = datetime.datetime.now()
                    try:
                        from .workspace_cleanup import tidy_project_workspace
                        _tidy_deleted = 0
                        _tidy_freed = 0
                        for _tp in await asyncio.to_thread(_tidy_store.list_projects):
                            _ts = await asyncio.to_thread(
                                tidy_project_workspace, _tidy_store, _tp["id"])
                            _tidy_deleted += len(_ts.get("deleted") or [])
                            _tidy_freed += int(_ts.get("freed_bytes") or 0)
                        if _tidy_deleted:
                            self._record_autonomous_activity(
                                "workspace_tidy",
                                f"removed {_tidy_deleted} debris file(s) "
                                f"({_tidy_freed:,} bytes) across project "
                                "workspaces")
                    except Exception as e:
                        logger.warning(f"Workspace tidy phase failed: {e}")
                    finally:
                        self._last_workspace_tidy_at = datetime.datetime.now()

        # Phase 2.95: Autonomous project advancement (opt-in
        # --autoadvance-idle). The project autoadvancer existed but was
        # only ever reachable via the manage_projects tool or the HTTP
        # /advance route — NO idle phase ever called it, so "the
        # self-advancing research loop" never actually advanced on its
        # own. This phase picks one ACTIVE project and runs a single
        # advance_once tick on it, on the SAME hard rails the tool uses
        # (per-project step/runtime/tool-call budgets, human gates,
        # contradiction routing) plus a code generator so coding tasks
        # produce real code rather than the old no-op stub. Strictly
        # opt-in and one project / one tick per cooldown, so a runaway
        # is impossible. Anchor-before-await + finally, same as the
        # other phases.
        if (self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600)
                and getattr(getattr(ctx, 'args', None), 'autoadvance_idle', False) is True):
            since_last_aa = (datetime.datetime.now() - self._last_autoadvance_at).total_seconds()
            store = getattr(ctx, 'project_store', None)
            if since_last_aa >= self._AUTOADVANCE_COOLDOWN and store is not None:
                self._last_autoadvance_at = datetime.datetime.now()
                try:
                    actives = await asyncio.to_thread(store.list_projects, "ACTIVE")
                    if actives:
                        # Round-robin fairness: pick the project advanced
                        # LEAST recently (never-advanced projects, ts=0, go
                        # first). `list_projects` orders by `updated_at DESC`,
                        # but advancing a project bumps its updated_at, so the
                        # old `actives[0]` always re-selected the project just
                        # advanced and every other ACTIVE project starved.
                        target = min(
                            actives,
                            key=lambda p: float(
                                (p.get("metadata") or {}).get("last_autoadvance_ts", 0) or 0
                            ),
                        )
                        pid = target.get("id")
                        from .project_advancer import (
                            advance_once as _advance_once,
                            pinned_project_context as _pin_ctx,
                        )

                        async def _aa_tool_runner(name, targs):
                            # Build tools from a context whose
                            # current_project_id is PINNED to the target
                            # project: idle ticks have no conversation, so
                            # the process-global id is parked (None) and
                            # unpinned file writes land at the sandbox ROOT
                            # — invisible to interactive sessions on the
                            # project (observed live 2026-07-08, TinyAI).
                            try:
                                from ..tools.registry import get_available_tools
                                tmap = get_available_tools(_pin_ctx(ctx, pid))
                            except Exception:
                                tmap = None
                            handler = (tmap or {}).get(name)
                            if not handler:
                                return f"ERROR: tool {name} unavailable"
                            return await handler(**targs)

                        async def _aa_classify(description):
                            # LLM router for autonomous tasks. The keyword
                            # bucket classifier (project_advancer.classify_task)
                            # mislabels anything without an exact keyword —
                            # "Design the schema" has none, so it defaulted to
                            # research/web_search. Ask the model for the bucket
                            # and fall back to the heuristic on any failure.
                            from .project_advancer import classify_task as _kw
                            try:
                                # One-word bucket classification: the cheapest
                                # possible LLM call, with a keyword fallback
                                # right below if it fails. Offloaded to the
                                # worker pool (2026-07-11) so it never occupies
                                # the single 35B slot; falls back to the main
                                # node when no worker node is configured.
                                _r = await ctx.llm_client.chat_completion({
                                    "model": getattr(ctx.args, 'model', 'default'),
                                    "messages": [{"role": "user", "content": (
                                        "Classify this task into EXACTLY one word: "
                                        "'coding' (writes/runs code), 'research' "
                                        "(reads/searches/summarizes), or 'needs_user' "
                                        "(requires a human decision/approval/publish). "
                                        "Output ONLY the one word.\n\nTASK: "
                                        + str(description)[:500])}],
                                    "temperature": 0.0, "max_tokens": 8, "stream": False,
                                }, use_worker=True, is_background=True, task_label="classifier")
                                out = ((_r or {}).get("choices", [{}])[0]
                                       .get("message", {}).get("content", "") or "").strip().lower()
                                for label in ("needs_user", "coding", "research"):
                                    if label in out:
                                        return label
                            except Exception as _ce:
                                logger.debug(f"autoadvance classify failed: {_ce}")
                            return _kw(description)

                        async def _aa_code_gen(description):
                            _r = await ctx.llm_client.chat_completion({
                                "model": getattr(ctx.args, 'model', 'default'),
                                "messages": [{"role": "user", "content": (
                                    "Write a SINGLE shell command (you may invoke "
                                    "python3 -c). Output ONLY the command — no "
                                    "explanation, no markdown fences.\n\nTASK: "
                                    + str(description)[:500])}],
                                "temperature": 0.2, "max_tokens": 1024, "stream": False,
                            }, is_background=True)
                            out = ((_r or {}).get("choices", [{}])[0]
                                   .get("message", {}).get("content", "") or "").strip()
                            if out.startswith("```"):
                                out = out.strip("`")
                                if "\n" in out:
                                    out = out.split("\n", 1)[1]
                            return out.strip()

                        from .coding_executor import build_coding_task as _bct

                        async def _aa_build(*a, **kw):
                            # IDLE build — defer its spec-generation LLM call to
                            # the background lane so a user who starts typing
                            # mid-build isn't stuck behind a 8K-token spec call.
                            kw.setdefault("is_background", True)
                            return await _bct(*a, **kw)

                        # Pin the EVENT stamp too: the pinned context above
                        # only scopes sandbox writes (context.current_project_id);
                        # record_command_outcome et al. stamp from the shared
                        # workspace model, which still holds the LAST chat
                        # turn's project during an idle tick. Without this,
                        # every idle build's command outcomes land on the
                        # wrong project's activity view.
                        from ..workspace import pinned_event_project as _pin_evt
                        with _pin_evt(pid):
                            result = await _advance_once(
                                ctx, pid, tool_runner=_aa_tool_runner,
                                llm_classifier=_aa_classify,
                                code_generator=_aa_code_gen,
                                coding_executor=_aa_build,
                            )
                        pretty_log(
                            "Autoadvance",
                            f"project={str(pid)[:8]} task={str(result.task_id)[:8]} "
                            f"{result.classification}: {result.summary[:80]}",
                            icon=Icons.BRAIN_PLAN,
                        )
                except Exception as e:
                    logger.warning(f"Autoadvance phase failed: {e}")
                finally:
                    self._last_autoadvance_at = datetime.datetime.now()

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
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
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

        # Phase 2.8b: Surface STALE open questions (previously unwired). Every
        # few hours, pull self-questions the agent has carried unresolved past
        # `max_age_days` and log them so they can be re-engaged instead of
        # silently accreting. Same idle window + cooldown discipline.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            since_last_sq = (datetime.datetime.now() - self._last_stale_questions_at).total_seconds()
            if since_last_sq >= self._STALE_QUESTIONS_COOLDOWN:
                _sm = getattr(ctx, 'self_model', None)
                if _sm is not None and getattr(_sm, 'enabled', False):
                    self._last_stale_questions_at = datetime.datetime.now()
                    try:
                        stale = _sm.stale_open_questions(max_age_days=3.0)
                        if stale:
                            preview = "; ".join(getattr(q, "text", str(q))[:60] for q in stale[:3])
                            pretty_log(
                                "Selfhood",
                                f"{len(stale)} stale open question(s) carried >3d — re-engage: {preview}",
                                icon=Icons.SELF_STATE,
                            )
                            self._record_autonomous_activity(
                                "open_questions",
                                f"{len(stale)} open question(s) carried "
                                f">3 days: {preview}")
                    except Exception as e:
                        logger.debug(f"stale-question surfacing failed: {e}")
                    finally:
                        self._last_stale_questions_at = datetime.datetime.now()

        # Phase 2.9: Workspace Narrative Consolidation (15-60 min idle).
        # Mirrors phase 2.8 but for the world-model. Re-renders the
        # "running summary of the workspace" so the next session's
        # wake-up prefix can splice it in without a re-run. Same idle
        # window + cooldown discipline as the selfhood narrative.
        if self._bio_scaled(900) < idle_secs <= self._bio_scaled(3600):
            ws_cooldown_override = getattr(
                getattr(ctx, 'args', None), 'workspace_narrative_cooldown', None,
            )
            if not isinstance(ws_cooldown_override, (int, float)):
                ws_cooldown_override = None
            ws_cooldown = (
                float(ws_cooldown_override) if ws_cooldown_override
                else float(self._WORKSPACE_NARRATIVE_COOLDOWN)
            )
            since_last_ws = (
                datetime.datetime.now() - self._last_workspace_narrative_at
            ).total_seconds()
            if since_last_ws >= ws_cooldown:
                try:
                    from ..workspace import WorkspaceModel as _WorkspaceModel
                    ws = getattr(ctx, 'workspace_model', None)
                    if isinstance(ws, _WorkspaceModel) and getattr(ws, 'enabled', False):
                        self._last_workspace_narrative_at = datetime.datetime.now()
                        text = await ws.consolidate_narrative()
                        if text:
                            preview = text.replace("\n", " ")[:120]
                            pretty_log(
                                "Workspace",
                                f"narrative regenerated ({len(text)} chars): {preview}…",
                                icon=Icons.BRAIN_THINK,
                            )
                except Exception as e:
                    logger.warning(f"Workspace narrative phase failed: {e}")
                finally:
                    self._last_workspace_narrative_at = datetime.datetime.now()

        # Phase 3: Synthetic Self-Play (>60 min idle)
        if idle_secs > self._bio_scaled(3600):
            since_last_selfplay = (datetime.datetime.now() - self._last_selfplay_at).total_seconds()
            if since_last_selfplay >= self._current_selfplay_cooldown and self._bio_roll(0.2):
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
                    # Counterfactual phase 1 (2026-07-17): ~1 idle slot in 4
                    # replays PAST challenges against the current lessons
                    # instead of generating fresh ones — the measurement leg
                    # of the post-mortem→lesson loop. Only when a backlog
                    # exists; otherwise the slot stays a normal self-play.
                    _ran_cf = False
                    if self._bio_roll(0.25):
                        try:
                            from .counterfactual import (
                                load_replay_candidates, run_counterfactual_batch)
                            if load_replay_candidates(1):
                                _cf = await run_counterfactual_batch(
                                    dreamer, ctx)
                                if _cf.get("replayed"):
                                    _ran_cf = True
                                    self._record_autonomous_activity(
                                        "self_play",
                                        f"counterfactual replay: "
                                        f"{_cf['replayed']} challenge(s) — "
                                        f"{_cf['generalized']} generalized, "
                                        f"{_cf['regressions']} regression(s), "
                                        f"{_cf['stable']} stable")
                        except Exception as _cfe:
                            logger.debug("counterfactual slot skipped: %s", _cfe)
                    if not _ran_cf:
                        await dreamer.synthetic_self_play(
                            model_name=getattr(ctx.args, 'model', 'default'),
                            is_background=True
                        )
                        self._record_autonomous_activity(
                            "self_play",
                            "synthetic self-play session ran (new lessons land "
                            "in the skills playbook)")
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
        from ..memory.journal import (
            JOURNAL_MAX_RETRIES as _JRL_MAX,
            RetryableConsolidationError as _RetryableConsolidation,
        )
        if not hasattr(self.context, 'journal'): return

        items = await asyncio.to_thread(self.context.journal.pop_all)
        if not items: return

        pretty_log("Hippocampus", f"Waking up to process {len(items)} buffered memories...", icon=Icons.BRAIN_THINK)

        processed = 0
        requeue = []  # items that failed upstream-transiently → back to the journal
        for i, item in enumerate(items):
            if respect_idle:
                idle_secs = (datetime.datetime.now() - self.context.last_activity_time).total_seconds()
                if idle_secs < 30:
                    pretty_log("Hippocampus", f"User returned! Suspending memory processing. ({len(items)-i} items left)", icon=Icons.STOP)
                    await asyncio.to_thread(self.context.journal.push_front, requeue + items[i:])
                    requeue = []
                    break

            try:
                if item["type"] == "smart_memory":
                    await self.run_smart_memory_task(item["data"]["text"], item["data"]["model"], self.context.args.smart_memory)
                elif item["type"] == "post_mortem":
                    await self._execute_post_mortem(item["data"]["user"], item["data"]["tools"], item["data"]["ai"], item["data"]["model"])
                processed += 1
            except _RetryableConsolidation as e:
                # The item was already popped, so dropping it here is
                # PERMANENT — the fact never reaches memory and nothing
                # downstream notices. Re-queue with a bounded retry count;
                # the next drain runs after the journal cooldown, by which
                # time a busy llama has usually recovered.
                retries = int(item.get("retries", 0) or 0)
                if retries < _JRL_MAX:
                    item["retries"] = retries + 1
                    requeue.append(item)
                    pretty_log(
                        "Hippocampus",
                        f"Upstream transient — re-queued {item.get('type')} item "
                        f"(attempt {retries + 1}/{_JRL_MAX}): {e}",
                        level="WARNING", icon=Icons.RETRY,
                    )
                else:
                    pretty_log(
                        "Hippocampus",
                        f"Dropping {item.get('type')} item after {_JRL_MAX} "
                        f"failed re-queues: {e}",
                        level="WARNING", icon=Icons.WARN,
                    )
            except Exception as e:
                import logging
                logging.getLogger("GhostAgent").error(f"Journal processing error: {e}")
            await asyncio.sleep(0.5)

        if requeue:
            await asyncio.to_thread(self.context.journal.push_front, requeue)
        if processed > 0:
            pretty_log("Hippocampus", f"Successfully consolidated {processed} memories.", icon=Icons.OK)

    async def run_smart_memory_task(self, interaction_context: str, model_name: str, selectivity: float):
        from ..memory.journal import RetryableConsolidationError as _RetryableConsolidation
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
                # max_tokens 1024 truncated real consolidations mid-JSON
                # (observed: output died at `"profile_update":` and the
                # extractor logged a malformed-JSON warning). 3072 leaves
                # room for score+fact+profile_update+graph_triplets; the
                # json_object response_format (already used by the graph
                # extractor) keeps the model from padding with prose.
                payload = {"model": model_name, "messages": [{"role": "user", "content": final_prompt}], "stream": False, "temperature": 0.1, "max_tokens": 3072, "response_format": {"type": "json_object"}}
                try:
                    data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True, task_label="memory extract")
                except Exception as _ue:
                    # The in-client retries (worker failover + one 2s retry
                    # on 5xx) are exhausted and NOTHING has been stored yet,
                    # so the whole task is safe to re-run later. Signal the
                    # drain loop to re-queue the journal item — the old
                    # log-and-swallow path dropped the consolidation
                    # permanently (the item was already popped).
                    from ..memory.journal import is_upstream_transient as _iut
                    if _iut(_ue):
                        raise _RetryableConsolidation(f"{type(_ue).__name__}: {_ue}") from _ue
                    raise
                content = data["choices"][0]["message"]["content"]
                # repair_truncated: a cap hit here used to drop the whole
                # consolidation; salvaging the complete pairs keeps the fact.
                result_json = extract_json_from_text(content, repair_truncated=True)
                score, fact, profile_up = float(result_json.get("score", 0.0)), result_json.get("fact", ""), result_json.get("profile_update", None)
                # The LLM is asked for profile_update as an object, but it
                # sometimes returns a bare string (or other scalar). Anything
                # that isn't a dict can't drive the profile-update path below,
                # so normalize it to None — otherwise profile_up.get(...) later
                # raises "'str' object has no attribute 'get'".
                if not isinstance(profile_up, dict):
                    profile_up = None

                # --- UNCONDITIONAL KNOWLEDGE GRAPH INGESTION ---
                from ..utils.helpers import is_removal_or_negation_text, is_removal_triplet
                graph_triplets = result_json.get("graph_triplets", [])
                # Drop removal / past-ownership triplets (e.g. user
                # PREVIOUSLY_OWNED iguana). Ingesting these re-creates the
                # tombstone the user just asked to forget; a removal must
                # delete edges, never add one.
                if graph_triplets:
                    kept_triplets = [t for t in graph_triplets if not is_removal_triplet(t)]
                    dropped = len(graph_triplets) - len(kept_triplets)
                    if dropped:
                        pretty_log("Graph Tombstone Skip", f"Dropped {dropped} removal/past-ownership triplet(s)", icon=Icons.STOP)
                    graph_triplets = kept_triplets
                if getattr(self.context, 'graph_memory', None) and graph_triplets:
                    added = await asyncio.to_thread(self.context.graph_memory.add_triplets, graph_triplets)
                    if added and added > 0:
                        pretty_log("Graph Updated", f"Mapped {added} topological edges", icon=Icons.MEM_SAVE)

                if fact is None: fact = ""
                # A removal / non-ownership fact ("user previously had an
                # iguana that was removed") must NOT be stored: consolidating
                # it manufactures a self-perpetuating tombstone that survives
                # every `forget`. Bail before the fact is embedded. (The
                # graph triplets above were already filtered.)
                if is_removal_or_negation_text(fact):
                    pretty_log("Auto Memory Skip", f"Discarded removal/negation tombstone: {fact}", icon=Icons.STOP)
                    return
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
                            eval_data = await self.context.llm_client.chat_completion(eval_payload, use_worker=True, is_background=True, task_label="self-eval")
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


            except _RetryableConsolidation:
                raise  # handled by process_journal_queue (bounded re-queue)
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
            l_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True, task_label="postmortem")
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
            "_stable_tool_query",
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
            # #7: the acquired-skill semantic router filters the advertised
            # tool set by query. Feeding it the PER-TURN query (which becomes
            # the planner's thought mid-request) byte-changes the tool header
            # every turn → invalidates the upstream prompt-prefix KV cache →
            # a full re-prefill each turn. Pin the routing query to the FIRST
            # substantive query of the request so the advertised set stays
            # byte-stable across the request's turns (the lever #6's KV pin
            # needs). Topic relevance is preserved: the first user message
            # sets the request's topic.
            self._stable_tool_query = None

        def stable_tool_query(self, query: str) -> str:
            if self._stable_tool_query is None and query and query.strip():
                self._stable_tool_query = query
            return self._stable_tool_query or (query or "")

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
                from ..tools.file_system import tool_list_files, project_scoped_sandbox
                # Scope the ambient listing to the active project's dir so the
                # model is shown the SAME working set its file_system/execute
                # operate on (else it sees a root listing and fumbles paths).
                self._sandbox_state = await tool_list_files(
                    sandbox_dir=project_scoped_sandbox(self._agent_ref.context)[0],
                    memory_system=getattr(self._agent_ref.context, "memory_system", None),
                )
            except Exception:
                self._sandbox_state = ""
            return self._sandbox_state

        def invalidate_sandbox(self):
            self._sandbox_state = None

        def invalidate_skill_playbook(self):
            self._skill_playbook_cache.clear()

        def invalidate_tool_defs(self):
            """Drop the cached tool schema list so the next LLM call
            re-enumerates acquired skills from disk. Called after a
            tool that mutates the skill registry (create_skill,
            manage_skills(action=delete)) runs, so the freshly-created
            skill appears in the very next turn-iteration's tool list
            rather than being stranded until the next user message."""
            self._active_tool_defs_cache.clear()
            self._xml_schema_cache.clear()

    async def warm_up_main_prefix(self) -> None:
        """Prefill the MAIN node's prompt cache with the byte-stable request
        head at boot, so the first user request doesn't pay it (2026-07-14).

        The first request of a session prefills ~30k+ tokens at the measured
        ~450 tok/s (≈70s wall): the rendered head is
        ``[system slot][native tool schemas]`` — SYSTEM_PROMPT + profile is
        ~14 KB and the tools JSON ~63 KB, and BOTH are byte-stable across
        conversations (all volatile continuity was already moved to the tail
        injection for exactly this reason; the chat template renders `tools`
        immediately after the system text). Upstream prefix caching
        (``cache_prompt`` + ``--cache-ram``) reuses everything up to the
        first differing byte, so one background ``max_tokens=1`` request at
        startup with the SAME head moves that prefill off the user's
        critical path — the first real request then only pays its unique
        tail (hydrated memory, dynamic state, the query itself).

        Byte-exactness is load-bearing: the head here is built by the SAME
        code paths a live request uses (``_RequestState.get_profile_str`` +
        ``SYSTEM_PROMPT`` splice + ``get_active_tool_defs``), so it cannot
        drift from real request bytes. The acquired-skill tail of the tool
        list is query-routed and may differ per conversation — that only
        shortens the match, never breaks it (schemas render in list order,
        static tools first).

        Main-slot etiquette: ``is_background=True`` with no off-main flags
        targets the main node but waits for any live foreground request to
        clear first, so a user who beat the warmup to the first request is
        never queued behind it. Best-effort: any failure is logged at debug
        and never escapes. Sibling of ``LLMClient.warm_up_workers`` (which
        covers the off-main nodes); wired in main.py at lifespan start,
        opt-out via GHOST_MAIN_PREFIX_WARMUP=0.
        """
        try:
            llm = getattr(self.context, "llm_client", None)
            if llm is None:
                return
            request_state = self._RequestState(self)
            profile_str = await request_state.get_profile_str()
            base_prompt = SYSTEM_PROMPT.replace("{{PROFILE}}", profile_str)
            # Mirror the one config-driven system-slot splice a live request
            # applies (handle_chat's perfect_it block) — byte-exactness.
            if getattr(self.context.args, 'perfect_it', False) is True:
                base_prompt = base_prompt.replace(
                    "### TOOL ORCHESTRATION",
                    '5. THE "PERFECT IT" PROTOCOL: Upon successfully completing a complex technical task, analyze the result (the most recent <tool_response> output in this conversation) and proactively suggest one concrete way to optimize it.\n\n### TOOL ORCHESTRATION'
                )

            payload = {
                "model": getattr(self.context.args, "model", "default"),
                "messages": [
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": "ok"},
                ],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 1,
            }
            if getattr(self.context.args, "native_tools", False) is True:
                # Same builder + same neutral-query routing a live request's
                # first turn resolves to; the payload mirrors the live shape
                # (tool_choice/parallel flags don't affect the rendered
                # prefix, but identical payloads leave nothing to reason
                # about).
                all_tools = request_state.get_active_tool_defs("")
                if getattr(self, "disabled_tools", None):
                    all_tools = [t for t in all_tools
                                 if t["function"]["name"] not in self.disabled_tools]
                if all_tools:
                    payload["tools"] = all_tools
                    payload["tool_choice"] = "auto"
                    payload["parallel_tool_calls"] = True

            _est_tokens = (len(base_prompt)
                           + len(json.dumps(payload.get("tools", [])))) // 4
            pretty_log(
                "Main Prefix Warmup",
                f"prefilling ~{_est_tokens} tokens of byte-stable request head "
                f"(system slot + tool schemas) into the main node's cache",
                icon=Icons.BOOT_AWAKE,
            )
            # Generous timeout: this IS the ~70s prefill we're absorbing.
            await llm.chat_completion(
                payload, is_background=True, timeout=240.0,
                task_label="main-prefix-warmup",
            )
            pretty_log(
                "Main Prefix Warmup",
                "done — first user request now pays only its unique tail",
                icon=Icons.OK,
            )
        except Exception as e:  # noqa: BLE001 — warmup is best-effort
            logger.debug("main-prefix warmup skipped: %s: %s",
                         type(e).__name__, e)

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

    # A token already counts as a "concrete subject" when it is an
    # id-like blob — a long hex string (project/task ids like
    # `516217d294cc`) or any alphanumeric token of length >= 6 that
    # carries at least one digit. Pure prose words never match.
    _ID_LIKE_RE = re.compile(r"^[0-9a-f]{6,}$", re.IGNORECASE)
    _MIXED_ID_RE = re.compile(r"^(?=.*\d)[a-z0-9_\-]{6,}$", re.IGNORECASE)

    @classmethod
    def _has_concrete_reference(cls, text: str) -> bool:
        """True when a short user message already names a concrete subject,
        so prepending the previous assistant reply as `Context:` can only
        contaminate memory retrieval rather than resolve an anaphor.

        The contextual-query-expansion at the call site exists to resolve
        pronoun follow-ups ("run it then", "why?") against the prior reply.
        But an imperative that already carries its own subject — backticked
        names, quoted strings, or id-like blobs (`delete 516217d294cc`) — is
        self-contained. Prepending a stale reply there is what made a
        partial-failure `delete` re-answer the *previous* turn's question.
        Returns True → skip the prepend and search on the raw message.
        """
        if not text or not isinstance(text, str):
            return False
        # Explicit quoting is an unambiguous "I am naming a specific thing".
        # (Bare apostrophes are excluded — they are usually contractions
        # like "what's it doing?", which are genuinely anaphoric. A
        # single-quoted id is still caught by the id-token scan below, which
        # strips surrounding quotes before matching.)
        if "`" in text or '"' in text:
            return True
        for raw in text.split():
            tok = raw.strip("`\"'.,;:!?()[]{}<>")
            if not tok:
                continue
            if cls._ID_LIKE_RE.match(tok) or cls._MIXED_ID_RE.match(tok):
                return True
        return False

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

        # Disable thinking for the trivial path. This is a Qwen-style
        # reasoning model: left to its own devices it spends the whole
        # `max_tokens` budget on a `<think>` block and returns an EMPTY
        # `content` field — exactly the bug that made every greeting fall
        # through to the full turn loop. Chain-of-thought cannot improve a
        # "hi"/"thanks" reply, so we suppress it via BOTH switches the model
        # honours: the `enable_thinking` chat-template flag AND the portable
        # `/no_think` soft-switch appended to the user message (see the
        # NON_THINKING_* notes at the top of this module).
        last_msg = req_messages[-1]
        if not str(last_msg.get("content", "")).rstrip().endswith("/no_think"):
            last_msg["content"] = f"{last_msg.get('content', '')} /no_think".strip()

        payload = {
            "model": model,
            "messages": req_messages,
            "max_tokens": 256,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            **NON_THINKING_GENERAL_PARAMS,
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
            msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
            content = msg.get("content") or ""
            # Reasoning models return their chain-of-thought in a separate
            # `reasoning_content` field. Even with thinking disabled, a model
            # can occasionally route the whole reply there and leave `content`
            # empty. Mirror the full path (agent.py ~4106): fold reasoning in
            # so the <think>-stripper below can recover any text after it.
            reasoning = msg.get("reasoning_content")
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n{content}"
        except Exception as e:
            logger.error(f"Trivial fast path response parse failed: {e}")
            return None

        # Strip <think>...</think> preamble (closed or not) so the user sees
        # only the visible reply. The unclosed variant matters: if the model
        # was cut off mid-thought, everything after the last </think> — or
        # nothing — is the real answer.
        content = re.sub(
            r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        # If, after stripping, the model gave us nothing usable, fall through
        # so the full path can take a second crack at it.
        if not content:
            logger.warning("Trivial fast path returned empty content; falling through to full path")
            return None

        self.context.last_activity_time = datetime.datetime.now()
        pretty_log("Trivial Fast Path", f"Resolved in {len(content)} chars", icon=Icons.OK)

        return content, created_time, req_id

    # Max bounded VERIFIER-GATE AUTO-REPAIR rounds per request. When the
    # verifier REFUTES the final answer (or it finalised on an unverified
    # mutation), the agent gets up to this many extra in-loop attempts to
    # diagnose and fix the issue before the answer ships. Kept at 1: a
    # single grounded re-attempt fixes the common case (a wrong/untested
    # claim) without doubling latency on every hard task. Bounded
    # independently of the turn budget so a repair can't extend a runaway.
    _MAX_VERIFIER_REPAIRS = 1

    # A basename-fallback match must have been written within this window to
    # count — otherwise WEB-EXEC would "certify" a build by loading a STALE
    # index.html from an unrelated prior project (observed live: the turn's
    # deliverable never landed on disk, and the fallback found a 25-min-old
    # file from another project and reported it clean — a false confirm).
    _WEB_EXEC_FRESH_WINDOW_S = 900.0

    # When the turn WROTE web artifacts but the execution probe could not
    # run (no loadable entry page, navigation failed, probe crashed), a
    # CONFIRMED verdict rests on text/vision alone — the artifact was never
    # executed. Cap its confidence below the 0.7 consumption threshold so
    # an unexecuted build can't be recorded as a verified pass (observed
    # live 2026-06-20: "WEB-EXEC check skipped" followed by CONFIRMED 100%
    # on vision alone, on a page that threw on load).
    _WEB_EXEC_SKIP_CONF_CAP = 0.6

    def _locate_entry_page(self, candidates: list, sbx: Path, root: Path):
        """Resolve a web entry-page name to an existing host file, or None.

        Searches the scoped sandbox AND the root mount by the EXACT named
        path first (trusted — the agent named that file and it exists), then
        falls back to a freshest-wins basename search under the root. The
        fallback exists because a project-REUSE turn can leave the deliverable
        in ``<root>/projects/<id>/index.html`` while ``project_scoped_sandbox``
        reads as un-scoped (the binding gap that made WEB-EXEC silently skip
        on the live single-file build). It is gated on recency so a stale
        unrelated artifact can never stand in for a deliverable that didn't
        actually land this turn — a missing deliverable stays inconclusive.
        """
        from ..tools.file_system import _get_safe_path
        for c in candidates:
            for base in (sbx, root):
                try:
                    p = _get_safe_path(base, c)
                    if p.exists() and p.is_file():
                        return p
                except Exception:
                    continue
        matches: list = []
        for c in candidates:
            try:
                matches.extend(m for m in root.rglob(Path(c).name) if m.is_file())
            except Exception:
                continue
        if matches:
            try:
                matches.sort(key=lambda m: m.stat().st_mtime, reverse=True)
            except Exception:
                pass
            cutoff = time.time() - self._WEB_EXEC_FRESH_WINDOW_S
            for m in matches:
                try:
                    if m.stat().st_mtime >= cutoff:
                        return m
                except Exception:
                    continue
        return None

    async def _execute_web_artifact(self, written: list):
        """Headless-load the entry page for web files written this turn.

        Returns ``(page_rel, error_block)`` — ``error_block`` is "" when the
        page loaded with no uncaught JS exception — or ``None`` when the
        check cannot run (no entry page on disk, no browser tool, or the
        navigation itself failed; a failed probe must stay inconclusive,
        never read as "clean"). JS-only edits load the conventional
        ``index.html`` next to the edited file.
        """
        browser = (getattr(self, "available_tools", None) or {}).get("browser")
        if browser is None:
            return None
        from ..tools.file_system import project_scoped_sandbox, _to_container_path
        sbx = Path(project_scoped_sandbox(self.context)[0])
        # The bind mount sits at the sandbox ROOT even when file ops are
        # scoped to <root>/projects/<id>; un-scope so we can find (and build a
        # container URL for) files written under either layout.
        root = sbx.parent.parent if sbx.parent.name == "projects" else sbx
        candidates = [f for f in written if f.lower().endswith((".html", ".htm"))]
        if not candidates:
            candidates = [str(Path(f).parent / "index.html") for f in written]
        host = self._locate_entry_page(candidates, sbx, root)
        if host is None:
            return None
        try:
            page_rel = host.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            page_rel = host.name
        # ABSOLUTE container URL (triple-slash). A relative `file://index.html`
        # is parsed as a HOST with an empty path and never loads — that was the
        # WEB-EXEC reliability bug: the probe "skipped" on every build because
        # the URL pointed nowhere, so a page that throws on load still got a
        # text-only CONFIRMED.
        url = "file://" + _to_container_path(sbx, host)
        res = await browser(
            operation="navigate",
            url=url,
            wait_until="domcontentloaded",
        )
        text = str(res)
        if text.lstrip().startswith("Error") or "[BROWSER_ERR]" in text:
            return None
        marker = "UNCAUGHT JS EXCEPTIONS"
        if marker not in text:
            return page_rel, ""
        return page_rel, text[text.index(marker):][:1200]

    def _project_constraints_for(self, pid, limit: int = 5) -> List[str]:
        """Explicit user constraints stored on project ``pid``'s record
        (captured at create/correction time by tools.projects). Empty list
        when the project is missing or carries none. Never raises — every
        caller treats constraints as best-effort context."""
        try:
            store = getattr(self.context, "project_store", None)
            if store is None or not pid:
                return []
            proj = store.get_project(pid) or {}
            cons = [str(c) for c in
                    ((proj.get("metadata") or {}).get("constraints") or [])]
            return cons[:limit]
        except Exception:
            return []

    def _active_project_constraints(self, limit: int = 5) -> List[str]:
        """Stored constraints of the ACTIVE (conversation-bound) project."""
        return self._project_constraints_for(
            getattr(self.context, "current_project_id", None), limit)

    def _active_constraint_note(self, limit: int = 5) -> str:
        """Explicit user constraints stored on the active project, rendered
        as a short prefix for the verifier's request view. Empty string when
        no project is active or the project carries no constraints."""
        cons = self._active_project_constraints(limit)
        if not cons:
            return ""
        return ("ACTIVE PROJECT CONSTRAINTS (user-mandated, MUST hold): "
                + " | ".join(cons) + " || USER REQUEST: ")

    def _merge_project_constraints(self, request_constraints, user_text=""):
        """Merge stored project constraints into a request's constraint set.

        Returns ``(constraints, rendered_block, steer_pending)`` — the same
        triple the request-start extraction produces, so the call site
        overwrites all three locals in place. Rationale (2026-07-05 chess
        session): the request that wrote the forbidden engine carried the
        message "proceed." — ``extract_constraints()`` sees only the CURRENT
        message, so the post-write steer never armed even though the binding
        clauses ("not a coded AI, YOU play") sat on the project record from
        an earlier message. The verifier already replays stored constraints
        via ``_active_constraint_note``, but that whole path is dead under
        ``--no-verifier``; this merge makes the dynamic-state block and the
        post-write steer see them regardless.

        Two sources, both best-effort: the conversation-BOUND project, and
        any project referenced BY PATH in the message text (``projects/<id>/
        …``). The second covers the fresh-conversation escape (2026-07-05
        request C5): a traceback pasted into a new chat named the project
        on every line, but the new conversation had no binding, so zero
        constraints replayed and the mock-server misdiagnosis ran
        unchecked."""
        merged = list(request_constraints or [])
        seen = {c.lower() for c in merged}
        added = 0
        pools = [self._active_project_constraints()]
        referenced = list(dict.fromkeys(re.findall(
            r"\bprojects/([0-9a-f]{6,32})\b", (user_text or "").lower())))
        for pid in referenced[:3]:
            pools.append(self._project_constraints_for(pid))
        for pool in pools:
            for c in pool:
                if c.lower() not in seen:
                    merged.append(c)
                    seen.add(c.lower())
                    added += 1
        if added:
            pretty_log(
                "Constraints",
                f"replayed {added} stored project constraint(s) into the "
                f"request ({len(merged)} active)",
                icon=Icons.CONSTRAINT,
            )
        block = render_constraint_block(
            merged, header="EXPLICIT USER CONSTRAINTS (CURRENT REQUEST)")
        return merged, block, bool(merged)

    async def _compute_verifier_verdict(
        self, *, tools_run_this_turn, messages, final_ai_content,
        last_user_content, lc,
    ):
        """Pure verifier verdict computation — NO side effects.

        Extracted from the post-loop verifier gate so the same verdict can
        drive (a) the in-loop AUTO-REPAIR decision at finalisation and
        (b) the post-loop gate's auditor-note / backfill / lesson-retraction
        consumption, computed exactly once per final answer (the gate
        reuses the cached in-loop result). Returns
        ``(v_result_or_None, last_tool_or_None)``. Side effects (note,
        retraction, backfill, trajectory) stay at the post-loop call site so
        they run once, on the truly-final (possibly repaired) answer.
        """
        from .verifier import VerifyVerdict
        verifier = getattr(self.context, "verifier", None)
        last_tool = _find_substantive_tool_for_verifier(tools_run_this_turn)
        # --no-verifier ablation off-switch (covers BOTH the gated post-loop
        # gate and the direct in-loop auto-repair call): no verdict is computed
        # at all, so nothing is scrubbed/backfilled/repaired.
        if getattr(getattr(self, "context", None), "args", None) is not None and \
                getattr(self.context.args, "no_verifier", False) is True:
            return None, last_tool
        # Flat early-return (not an `if (verifier is not None and ...)`
        # block) so the post-loop gate stays the single match for the
        # gate-strictness source check — and so a missing verifier /
        # tool / trivial chat skips the LLM call cleanly.
        if (verifier is None
                or getattr(verifier, "llm_client", None) is None
                or last_tool is None
                or not final_ai_content
                or self._is_strict_trivial_chat(lc)):
            return None, last_tool
        # Replay the active project's explicit user constraints into the
        # verifier's view of the request. The current message is often just
        # "proceed" — the binding clauses ("don't come up with some random
        # AI") live on the project record from an EARLIER message, and the
        # verifier prompt treats constraint satisfaction as its
        # highest-priority check, so they must ride along here. Prepended,
        # not appended: the call sites truncate context to 1000 chars and a
        # tail-note would be the first thing cut.
        constraint_note = self._active_constraint_note()
        request_view = constraint_note + (last_user_content or "")
        v_result = None
        tool_output = str(last_tool.get("content", ""))[:4000]
        tool_name = str(last_tool.get("name", ""))[:80]
        # Claim-shaped verdicts judge the answer against the last FEW
        # substantive tool outputs, not just the final one — a synthesis
        # answer's support is spread across the turn, and judging it
        # against whichever tool ran last REFUTED correct answers on a
        # trailing 403 (req 738c/73). Code-shaped verdicts keep the
        # single-output view: they audit one specific run.
        claim_evidence = _collect_verifier_evidence(
            tools_run_this_turn) or tool_output
        if "execute" in tool_name.lower() or "postgres" in tool_name.lower():
            code_text = _reconstruct_executed_code(messages, last_tool)
            if code_text:
                v_result = await verifier.verify_code_output(
                    code=code_text,
                    output=tool_output,
                    intent=request_view,
                    response=final_ai_content or "",
                )
            else:
                v_result = await verifier.verify_claim(
                    claim=final_ai_content[:2000],
                    evidence=claim_evidence,
                    context=request_view[:1000],
                )
        else:
            v_result = await verifier.verify_claim(
                claim=final_ai_content[:2000],
                evidence=claim_evidence,
                context=request_view[:1000],
            )
        # Visual ground-truth override (unchanged from the inline gate).
        try:
            if _is_visual_intent(last_user_content):
                from ..tools.file_system import project_scoped_sandbox
                _sbx = project_scoped_sandbox(self.context)[0]
                _before_img, _after_img = _select_visual_evidence(
                    messages, last_user_content or "", _sbx,
                )
                if _after_img:
                    _vv = await verifier.verify_visual(
                        symptom=last_user_content or "",
                        claim=final_ai_content or "",
                        after_image=_after_img,
                        before_image=_before_img,
                    )
                    if _vv is not None:
                        if _vv.confidence >= 0.7 and _vv.verdict == VerifyVerdict.REFUTED:
                            v_result = _vv
                        elif _vv.confidence >= 0.7 and v_result is None:
                            v_result = _vv
                        pretty_log(
                            "Verifier",
                            f"VISUAL {_vv.verdict.value} "
                            f"({_vv.confidence:.0%}): "
                            f"{(_vv.reasoning or '')[:120]}",
                            icon=Icons.BRAIN_THINK,
                        )
                    else:
                        pretty_log(
                            "Verifier",
                            "VISUAL check skipped (vision returned no verdict)",
                            icon=Icons.BRAIN_THINK,
                        )
                else:
                    pretty_log(
                        "Verifier",
                        "VISUAL check skipped (no rendered after-image "
                        f"this turn; before={'yes' if _before_img else 'no'})",
                        icon=Icons.BRAIN_THINK,
                    )
        except Exception as _vv_exc:
            pretty_log(
                "Verifier",
                f"VISUAL check error: {type(_vv_exc).__name__}: {_vv_exc}",
                icon=Icons.WARN, level="WARNING",
            )
        # Web-artifact ground-truth override — execute, don't trust. The
        # text verifier CONFIRMED (95%) a build whose data.js had a parse
        # error: every claim/evidence pair read fine, but the page threw on
        # load and the user found out by clicking a dead button. When this
        # turn WROTE web files, load the entry page headless; an uncaught
        # exception refutes the answer regardless of how plausible the
        # claim text is. Runs last so it outranks the visual check —
        # execution is harder ground truth than pixels.
        _wx_inconclusive = False
        try:
            written = _web_artifacts_written(tools_run_this_turn)
            if written:
                check = await self._execute_web_artifact(written)
                if check is None:
                    _wx_inconclusive = True
                    pretty_log(
                        "Verifier",
                        "WEB-EXEC check skipped (no loadable entry page "
                        "or navigation failed)",
                        icon=Icons.BRAIN_THINK,
                    )
                else:
                    page_rel, err_block = check
                    if err_block:
                        from .verifier import VerifyResult
                        v_result = VerifyResult(
                            verdict=VerifyVerdict.REFUTED,
                            confidence=0.95,
                            reasoning=(
                                f"Execution check: loading '{page_rel}' raised "
                                f"uncaught JS exceptions — the artifact does not "
                                f"run, regardless of the claim text.\n{err_block}"
                            ),
                            issues=[f"uncaught JS exception(s) on {page_rel}"],
                        )
                        pretty_log(
                            "Verifier",
                            f"WEB-EXEC REFUTED: '{page_rel}' throws on load",
                            icon=Icons.BRAIN_THINK,
                        )
                    else:
                        pretty_log(
                            "Verifier",
                            f"WEB-EXEC clean: '{page_rel}' loaded with no "
                            f"uncaught exceptions",
                            icon=Icons.BRAIN_THINK,
                        )
        except Exception as _wx_exc:
            _wx_inconclusive = True
            pretty_log(
                "Verifier",
                f"WEB-EXEC check error: {type(_wx_exc).__name__}: {_wx_exc}",
                icon=Icons.WARN, level="WARNING",
            )
        # Fail-safe, not fail-open: web artifacts landed this turn but the
        # execution probe was inconclusive — a CONFIRMED from the text/vision
        # passes above certifies a build that never ran. Keep the verdict
        # (there is no evidence it's broken) but cap the confidence below
        # every ≥0.7 consumption gate (backfill "passed", auditor note,
        # calibration), so an unexecuted build is recorded as unverified.
        if (_wx_inconclusive and v_result is not None
                and v_result.verdict == VerifyVerdict.CONFIRMED
                and v_result.confidence > self._WEB_EXEC_SKIP_CONF_CAP):
            v_result.confidence = self._WEB_EXEC_SKIP_CONF_CAP
            v_result.reasoning = (
                (v_result.reasoning or "")
                + " [WEB-EXEC inconclusive: web artifacts were written this "
                  "turn but the entry page could not be loaded, so this "
                  "CONFIRMED is not execution-backed; confidence capped.]"
            ).strip()
            pretty_log(
                "Verifier",
                "WEB-EXEC inconclusive → CONFIRMED capped at "
                f"{self._WEB_EXEC_SKIP_CONF_CAP:.0%} (artifact never executed)",
                icon=Icons.BRAIN_THINK, level="WARNING",
            )
        # File-artifact ground-truth override (2026-07-16) — the general form
        # of the web-exec check for NON-web deliverables. If the answer claims
        # a file was produced but it is missing/empty in the sandbox, that's
        # the #1 premature-completion lesson made grounded (a real file read,
        # not the text verifier's plausibility). A REFUTED here is authoritative
        # and feeds the same auto-repair loop, so the agent gets a bounded
        # attempt to actually produce the deliverable.
        try:
            _claimed = _claimed_deliverable_files(final_ai_content)
            if _claimed:
                from ..tools.file_system import project_scoped_sandbox
                _host_dir = project_scoped_sandbox(self.context)[0]
                _fa = self._verify_file_artifacts(_claimed, _host_dir)
                if _fa is not None:
                    v_result = _fa
                    pretty_log(
                        "Verifier",
                        f"FILE-ARTIFACT REFUTED: {(_fa.reasoning or '')[:130]}",
                        icon=Icons.BRAIN_THINK, level="WARNING",
                    )
                else:
                    pretty_log(
                        "Verifier",
                        f"FILE-ARTIFACT clean: {len(_claimed)} claimed "
                        f"deliverable(s) present + non-empty",
                        icon=Icons.BRAIN_THINK,
                    )
        except Exception as _fa_exc:
            pretty_log(
                "Verifier",
                f"FILE-ARTIFACT check error: {type(_fa_exc).__name__}: {_fa_exc}",
                icon=Icons.WARN, level="WARNING",
            )
        # Interaction-claim cap (2026-07-17, reqs AF/43): the user reported a
        # POINTER/KEYBOARD behavior defect ("moving a window doesn't work"),
        # and a load-clean WEB-EXEC probe plus text entailment produced
        # CONFIRMED (100%) for a fix that was, in fact, still broken — a
        # drag/click behavior claim cannot be supported by evidence that
        # never exercised the interaction. Same philosophy as the
        # web-exec-inconclusive cap above: keep the verdict, cap the
        # confidence below every ≥0.7 consumption gate unless a browser
        # click/interact op actually SUCCEEDED this turn.
        try:
            if (v_result is not None
                    and v_result.verdict == VerifyVerdict.CONFIRMED
                    and v_result.confidence > self._WEB_EXEC_SKIP_CONF_CAP
                    and _is_interaction_intent(last_user_content)
                    and not _has_interaction_evidence(tools_run_this_turn)):
                v_result.confidence = self._WEB_EXEC_SKIP_CONF_CAP
                v_result.reasoning = (
                    (v_result.reasoning or "")
                    + " [INTERACTION untested: the request is about a "
                      "pointer/keyboard behavior, but no browser "
                      "click/interact succeeded this turn — a load-clean "
                      "page cannot confirm it; confidence capped.]"
                ).strip()
                pretty_log(
                    "Verifier",
                    "INTERACTION untested → CONFIRMED capped at "
                    f"{self._WEB_EXEC_SKIP_CONF_CAP:.0%} "
                    "(behavior never exercised)",
                    icon=Icons.BRAIN_THINK, level="WARNING",
                )
        except Exception as _ic_exc:
            pretty_log(
                "Verifier",
                f"interaction-cap check error: {type(_ic_exc).__name__}: {_ic_exc}",
                icon=Icons.WARN, level="WARNING",
            )
        return v_result, last_tool

    @staticmethod
    def _verify_file_artifacts(claimed, host_dir):
        """Grounded check: re-read the claimed deliverable files under the
        sandbox host dir. Refute if any is MISSING or EMPTY — hard ground
        truth the text verifier can't see. Returns a REFUTED VerifyResult or
        None (all present + non-empty, or nothing checkable)."""
        from pathlib import Path as _P
        from .verifier import VerifyResult, VerifyVerdict
        if not claimed or not host_dir:
            return None
        root = _P(host_dir)
        if not root.exists():
            return None
        missing, empty = [], []
        for name in claimed:
            rel = name
            for pfx in ("/workspace/", "workspace/", "./"):
                if rel.startswith(pfx):
                    rel = rel[len(pfx):]
            rel = rel.lstrip("/")
            found = None
            try:
                cand = root / rel
                if cand.is_file():
                    found = cand
                else:
                    base = _P(rel).name
                    for p in root.rglob(base):  # path mismatch → basename search
                        if p.is_file():
                            found = p
                            break
            except Exception:
                continue
            if found is None:
                missing.append(name)
            else:
                try:
                    if found.stat().st_size == 0:
                        empty.append(name)
                except OSError:
                    pass
        if not missing and not empty:
            return None
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if empty:
            parts.append("empty: " + ", ".join(empty))
        return VerifyResult(
            verdict=VerifyVerdict.REFUTED,
            confidence=0.9,
            reasoning=("File-artifact check: the answer claims deliverable "
                       "file(s) that are not actually present/non-empty in the "
                       "sandbox (" + "; ".join(parts) + "). The content was "
                       "declared done but not produced."),
            issues=[f"claimed-but-{'missing' if missing else 'empty'} deliverable"],
        )

    @staticmethod
    def _compose_injection(req_messages, stable_injection, dynamic_state, pin):
        """Place the per-turn stable + volatile context into ``req_messages``.

        Returns the (mutated) list. The per-turn injection has two parts:
        a large STABLE block (tool schemas + persona + playbook + hydrated
        memory + continuity — byte-identical across the turns of one
        request) and a small VOLATILE ``dynamic_state`` (timestamp /
        sandbox state / plan focus — changes every turn).

        ``pin=False`` (legacy): the whole injection rides the LAST message.
        That message is a *different* one each turn (turn 1: the query;
        turn 2: the tool result), so the stable block re-prefills every
        turn — only the system slot caches.

        ``pin=True``: the stable block is pinned to the FIRST user message
        — a position that never moves as the loop appends assistant/tool
        turns — so the upstream KV-cache reuses it on turns 2+. Only the
        volatile block rides the (moving) last message. On turn 1 the two
        coincide, so the volatile block is appended AFTER the instruction,
        keeping the ``[stable + instruction]`` prefix byte-identical on
        later turns. Native tool defs still travel in the payload ``tools``
        field, so tool-calling does not depend on the text block's position.
        """
        if not pin:
            transient_injection = f"{stable_injection}\n\n{dynamic_state.strip()}"
            if req_messages and req_messages[-1]["role"] == "user":
                original_msg = req_messages[-1]["content"]
                req_messages[-1]["content"] = (
                    f"<system_state_update>\n{transient_injection}\n(CRITICAL: This is "
                    "internal system state. Do NOT acknowledge or comment on this block in "
                    "your thoughts. Focus entirely on the user instruction.)\n"
                    f"</system_state_update>\n\n[USER INSTRUCTION]\n{original_msg}"
                )
            else:
                req_messages.append({
                    "role": "user",
                    "content": f"<system_state_update>\n{transient_injection}\n</system_state_update>",
                })
            return req_messages

        stable_block = (
            f"<session_context>\n{stable_injection}\n"
            "(CRITICAL: internal system state — do NOT acknowledge or comment on this "
            "block; focus entirely on the user instruction.)\n</session_context>"
        )
        volatile_block = (
            f"<system_state_update>\n{dynamic_state.strip()}\n"
            "(CRITICAL: This is internal system state. Do NOT acknowledge or comment on "
            "this block in your thoughts. Focus entirely on the user instruction.)\n"
            "</system_state_update>"
        )
        first_user_idx = next(
            (i for i, m in enumerate(req_messages) if m.get("role") == "user"), None,
        )
        if first_user_idx is None:
            ins = 1 if req_messages else 0
            req_messages.insert(ins, {"role": "user", "content": stable_block})
            first_user_idx = ins
        else:
            req_messages[first_user_idx]["content"] = (
                f"{stable_block}\n\n[USER INSTRUCTION]\n{req_messages[first_user_idx]['content']}"
            )
        last_idx = len(req_messages) - 1
        if last_idx == first_user_idx:
            # Turn 1 (the query is the only message). Do NOT fold volatile into
            # the pinned message — that would make turn 1's first message differ
            # from the clean [stable + instruction + query] that every later turn
            # presents, so turn 2 couldn't reuse the cached prefix. Emit volatile
            # as its own trailing message instead, keeping the pinned message
            # byte-identical from turn 1 onward.
            req_messages.append({"role": "user", "content": volatile_block})
        elif req_messages[last_idx]["role"] == "user":
            req_messages[last_idx]["content"] = (
                f"{volatile_block}\n\n{req_messages[last_idx]['content']}"
            )
        else:
            req_messages.append({"role": "user", "content": volatile_block})
        return req_messages

    def _critic_async_enabled(self) -> bool:
        """When ``GHOST_CRITIC_ASYNC=1`` the verifier never sits on the
        critical path: the in-loop repair-before-ship is skipped and the
        verdict runs purely AFTER the response ships — it logs, scrubs any
        poisoned lesson the turn wrote, and (on a high-confidence REFUTED)
        queues a correction surfaced at the top of the NEXT turn. Trades
        before-ship repair of the rare wrong answer for not paying the
        slow, off-host verdict latency on every turn.
        """
        return os.getenv("GHOST_CRITIC_ASYNC", "0").strip().lower() not in ("0", "false", "no")

    def _critic_gate_timeout(self) -> float:
        """Resolve the post-response verifier-gate budget, in seconds.

        Returns ``inf`` (block until the verdict lands — the historical
        behaviour) when no dedicated critic pool is configured, so legacy
        deployments are untouched. When ``--critic-nodes`` IS set, the
        verdict runs on a slower off-host model; default to ``0`` (pure
        async — ship the response immediately, never wait on the critic)
        so the 9B never serialises behind the answer. ``GHOST_CRITIC_GATE_TIMEOUT``
        overrides either way: set a positive value to wait that long for
        an inline verdict before releasing, ``0`` for pure async.
        """
        # Async mode forces pure-async regardless of any other setting.
        if self._critic_async_enabled():
            return 0.0
        raw = os.getenv("GHOST_CRITIC_GATE_TIMEOUT")
        if raw is not None and raw.strip() != "":
            try:
                return max(0.0, float(raw))
            except ValueError:
                pretty_log(
                    "Verifier",
                    f"GHOST_CRITIC_GATE_TIMEOUT={raw!r} is not a number — ignoring",
                    icon=Icons.WARN, level="WARNING",
                )
        verifier = getattr(self.context, "verifier", None)
        llm = getattr(verifier, "llm_client", None) if verifier else None
        return 0.0 if getattr(llm, "critic_clients", None) else float("inf")

    def _critic_repair_await_budget(self) -> float:
        """Seconds to wait at loop-exit for the async critic verdict so a
        REFUTED answer can be REPAIRED IN-LOOP rather than shipping with only a
        next-turn note (IMPROVEMENTS.md #18).

        Only meaningful in async-critic mode, where the verdict runs on the
        OFF-HOST second model — so this wait costs the MAIN inference slot
        nothing. Default 25s; ``GHOST_CRITIC_REPAIR_BUDGET`` overrides, 0
        disables (pure defer, the pre-2026-07-07 async behaviour)."""
        if not self._critic_async_enabled():
            return 0.0
        raw = os.getenv("GHOST_CRITIC_REPAIR_BUDGET")
        if raw is not None and raw.strip() != "":
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        return 25.0

    async def _compute_verifier_verdict_gated(
        self, *, tools_run_this_turn, messages, final_ai_content,
        last_user_content, lc, trajectory_id, conv_fp=None,
    ):
        """Non-blocking front door to ``_compute_verifier_verdict``.

        The verdict LLM call runs on the dedicated critic pool (the
        spare-box judge model when ``--critic-nodes`` is set), which is
        slower than the foreground model. To keep it from serialising
        behind the user's response, the verdict runs as a background task
        and the gate waits at most ``_critic_gate_timeout()`` seconds:

        * verdict lands in time → drives the response exactly as the old
          inline gate did (note / backfill / lesson-retraction);
        * verdict misses the budget → the response ships unverified (the
          existing ``v_result is None`` path) and the verdict finishes in
          the background, still scrubbing any poisoned lesson THIS turn
          produced (the safety-critical side effect) via a done-callback.

        With no critic pool the timeout is ``inf`` and this degrades to a
        plain ``await`` — byte-for-byte the legacy behaviour.
        """
        last_tool = _find_substantive_tool_for_verifier(tools_run_this_turn)

        # --no-verifier ablation off-switch: skip the post-response verifier
        # entirely (no verdict computed, no lesson scrubbing, no backfill). The
        # response ships exactly as the "verdict missed the budget" path already
        # handles (v_result is None). Lets the ablation harness measure whether
        # the late verifier's cost earns its (cross-session-only) keep.
        if getattr(getattr(self, "context", None), "args", None) is not None and \
                getattr(self.context.args, "no_verifier", False) is True:
            return None, last_tool

        gate = self._critic_gate_timeout()

        # Fast path / legacy: block for the verdict (no critic pool, or an
        # operator-set infinite budget). No task-spawn overhead.
        if gate == float("inf"):
            return await self._compute_verifier_verdict(
                tools_run_this_turn=tools_run_this_turn,
                messages=messages,
                final_ai_content=final_ai_content,
                last_user_content=last_user_content,
                lc=lc,
            )

        task = _glog.spawn_task(self._compute_verifier_verdict(
            tools_run_this_turn=tools_run_this_turn,
            messages=messages,
            final_ai_content=final_ai_content,
            last_user_content=last_user_content,
            lc=lc,
        ))

        if gate > 0:
            # asyncio.wait does NOT cancel on timeout — the task keeps
            # running in the background when it overruns the budget.
            done, _pending = await asyncio.wait({task}, timeout=gate)
            if task in done:
                try:
                    return task.result()
                except Exception as exc:
                    logger.debug("gated verifier verdict failed: %s", exc)
                    return None, last_tool
            pretty_log(
                "Verifier",
                f"critic still running after {gate:.0f}s — releasing response "
                "unverified; verdict will finish in the background",
                icon=Icons.WARN, level="WARNING",
            )

        # Pure async (gate == 0) or budget overrun: hand the still-running
        # task to the late handler and release the response now. Capture the
        # conversation fingerprint NOW (while its messages are in scope) so a
        # late correction can only surface back in the SAME conversation.
        self._attach_late_verdict_handler(
            task, trajectory_id,
            conv_fp if conv_fp is not None
            else self._conversation_fingerprint(messages),
        )
        return None, last_tool

    def _conversation_fingerprint(self, messages) -> str:
        """A stable-per-conversation, distinct-across-conversations tag.

        Derived from the FIRST user message, which the client resends on
        every turn of the same conversation (so it's constant across that
        conversation's turns) yet differs between conversations. Used to
        keep a deferred correction from one conversation from surfacing in
        an unrelated one on the shared singleton agent. Returns ``""`` when
        it can't be computed — treated downstream as "don't surface",
        i.e. fail safe (drop) rather than risk cross-posting.
        """
        try:
            first_user = next(
                (m.get("content") for m in messages
                 if isinstance(m, dict) and m.get("role") == "user"),
                "",
            )
            if isinstance(first_user, list):
                first_user = " ".join(
                    i.get("text", "") for i in first_user
                    if isinstance(i, dict) and i.get("type") == "text"
                )
            import hashlib
            return hashlib.sha1(
                str(first_user)[:2000].encode("utf-8", "ignore")
            ).hexdigest()[:16]
        except Exception:
            return ""

    def _attach_late_verdict_handler(self, task, trajectory_id, conv_fp="",
                                     force_correction=False):
        """Apply the safety-critical side effects of a verdict that lands
        AFTER its response already shipped.

        The response can't be amended at this point, so the auditor note
        and confidence backfill are forgone (logged for the operator).
        The one effect that must still happen is lesson retraction: a
        high-confidence REFUTED means any lesson THIS turn wrote is
        suspect, and the next user query must not retrieve it. Retraction
        is keyed by ``trajectory_id`` and idempotent, so running it a beat
        late is safe. ``conv_fp`` tags any queued correction with the
        conversation it belongs to, so it only surfaces back there.

        ``force_correction``: queue the next-turn correction banner on a
        high-confidence REFUTED even outside async-critic mode. Set by
        the STREAMING gate, whose replies never carry an inline Verifier
        note — without it, a refuted streamed answer under the default
        (non-async) config would be scrubbed/backfilled but the user
        would never hear about it.
        """
        def _on_done(t):
            # Release the finalize-burst stagger (see
            # _judge_hydration_safe) as soon as the verdict lands.
            if getattr(self, "_deferred_verdict_task", None) is t:
                self._deferred_verdict_task = None
            try:
                v_result, _lt = t.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                # This used to be a bare silent return — which hid the fact
                # that the async verdict NEVER completed in production (the
                # live log had 5 "verdict deferred" lines and 0 "LATE …"
                # lines): every safety side effect downstream (lesson scrub,
                # correction banner, corpus backfill) was dead without a
                # trace. A dying verdict task must be loud.
                pretty_log(
                    "Verifier",
                    f"async verdict task died: {type(exc).__name__}: "
                    f"{str(exc)[:200]}",
                    icon=Icons.WARN, level="WARNING",
                )
                return
            self._record_late_verdict(v_result, trajectory_id, conv_fp,
                                      last_tool=_lt,
                                      force_correction=force_correction)

        # Published so _judge_hydration_safe can serialise the finalize
        # burst behind the in-flight verdict (both used to hit the single
        # worker node in the same second — the loser blew the route
        # timeout on every substantive finalize, 2026-07-16 log).
        self._deferred_verdict_task = task
        task.add_done_callback(_on_done)

    def _backfill_trajectory_outcome(self, trajectory_id, outcome, reason=""):
        """Fold a late verifier verdict into the trajectory corpus.

        In async-critic mode (production) the verdict lands after
        ``_record_turn_trajectory`` already wrote ``outcome=UNKNOWN``, so
        the corpus that feeds the Reflector / PRM / skills-auto never saw
        what the verifier concluded — the sync path folds the same signal
        in at write time via ``resolve_turn_outcome``, so chat turns could
        only ever become PASSED when the verdict happened to land inline.
        Net effect before this: skills-auto graduation had ZERO passed-
        with-tools input in production (2058 UNKNOWN / 0 eligible on
        2026-07-05) and a late REFUTED never became a Reflector/PRM
        negative. Writes the corrections sidecar (overlay-on-read, audit-
        safe) and mutates the in-process correction cache so next-turn
        logic sees the current state.

        Direction guards mirror ``resolve_turn_outcome``'s priorities:
        FAILED always lands (a refute is never upgraded away by ordering);
        PASSED lands only when the recorded outcome is still UNKNOWN — a
        shape-heuristic or structural FAILED must never be upgraded, so a
        cache miss (can't prove UNKNOWN) skips conservatively. A later
        user-correction still wins either way: the sidecar is last-write-
        wins per id and the user's message arrives after this. Never
        raises — corpus backfill must not break the done-callback.
        """
        try:
            from ..distill.schema import Outcome as _Outcome
            collector = getattr(self.context, "trajectory_collector", None)
            if collector is None or not trajectory_id:
                return
            cached = None
            cache = getattr(
                self.context, "_recent_trajectories_for_correction", None)
            for t in (cache or {}).values():
                if getattr(t, "id", None) == trajectory_id:
                    cached = t
                    break
            if outcome == _Outcome.PASSED.value:
                if cached is None or cached.outcome != _Outcome.UNKNOWN.value:
                    return
            if cached is not None:
                cached.outcome = outcome
                if reason and outcome == _Outcome.FAILED.value \
                        and not (cached.failure_reason or ""):
                    cached.failure_reason = reason
            _glog.spawn_task(asyncio.to_thread(
                collector.update_outcome,
                trajectory_id, outcome,
                reason=reason, source="verifier_late",
            ))
            pretty_log(
                "Verifier",
                f"late verdict backfilled into the corpus: trajectory "
                f"{trajectory_id[:8]} → {outcome}",
                icon=Icons.VERIFIER_LAB,
            )
        except Exception as e:
            logger.debug(
                "late-verdict outcome backfill skipped: %s: %s",
                type(e).__name__, e,
            )

    def _record_late_verdict(self, v_result, trajectory_id, conv_fp="",
                             last_tool=None, force_correction=False):
        """Apply the side effects of a verdict that lands AFTER its
        response shipped. Extracted from the done-callback so it is
        unit-testable. On a high-confidence REFUTED: log it, scrub any
        poisoned lesson the turn wrote, and — in async mode — queue a
        correction to surface on the next turn. Either way, fold the
        high-confidence verdict into the trajectory corpus (the async
        counterpart of ``resolve_turn_outcome`` at record time).
        """
        if not v_result:
            # Differentiate WHY the verdict is empty (2026-07-08 operator
            # audit: one ambiguous line fired on every bookkeeping-only and
            # sim turn, drowning the case it exists for — a genuinely dead
            # verifier path).
            if last_tool is None:
                # No substantive evidence tool this turn — skipped by
                # design (no evidence, no verdict). Routine; log quietly.
                pretty_log(
                    "Verifier",
                    "no verdict — turn carried no verifiable evidence "
                    "(bookkeeping-only tools); skipped by design",
                    icon=Icons.BRAIN_THINK,
                )
            elif getattr(getattr(self.context, "verifier", None),
                         "llm_client", None) is None:
                # Sim/ablation contexts run without an attached verifier.
                pretty_log(
                    "Verifier",
                    "no verdict — verifier not attached in this context "
                    "(sim/ablation); skipped by design",
                    icon=Icons.BRAIN_THINK,
                )
            else:
                # Evidence existed AND a verifier is attached, yet no
                # verdict came back — trivial-chat skip or a genuinely
                # dead verifier path. THIS is the case worth watching.
                pretty_log(
                    "Verifier",
                    "LATE verdict was EMPTY despite verifiable evidence — "
                    "trivial-chat skip or a verifier error (investigate "
                    "if frequent)",
                    level="WARNING", icon=Icons.WARN,
                )
            return
        try:
            from .verifier import VerifyVerdict
        except Exception:
            return
        if v_result.confidence >= 0.7:
            if v_result.verdict == VerifyVerdict.CONFIRMED:
                self._backfill_trajectory_outcome(trajectory_id, "passed")
            elif v_result.verdict == VerifyVerdict.REFUTED:
                _bf_reason = (
                    "; ".join(v_result.issues[:2])
                    if v_result.issues else (v_result.reasoning or "")
                )
                self._backfill_trajectory_outcome(
                    trajectory_id, "failed",
                    reason=f"verifier refuted (late): {_bf_reason}"[:300],
                )
        if (v_result.verdict == VerifyVerdict.REFUTED
                and v_result.confidence >= 0.7):
            issues_str = (
                "; ".join(v_result.issues[:3])
                if v_result.issues else v_result.reasoning
            ) or "the previous answer was not supported by the evidence"
            pretty_log(
                "Verifier",
                f"LATE REFUTED ({v_result.confidence:.0%}): "
                f"{issues_str[:120]} — response already sent; "
                "scrubbing this turn's lessons",
                icon=Icons.WARN, level="WARNING",
            )
            _sm = getattr(self.context, "skill_memory", None)
            if _sm is not None and trajectory_id:
                _glog.spawn_task(asyncio.to_thread(
                    _sm.retract_lessons_from_trajectory,
                    trajectory_id,
                    memory_system=getattr(
                        self.context, "memory_system", None
                    ),
                ))
            # Async mode: stash a correction to surface on the next turn OF
            # THE SAME CONVERSATION, since the response is already out and
            # can't be repaired. Tagged with the conversation fingerprint and
            # a monotonic timestamp (for scoping + TTL), and the queue is
            # capped so a busy multi-conversation process can't accumulate an
            # unbounded banner chain.
            if self._critic_async_enabled() or force_correction:
                if not isinstance(getattr(self, "_pending_corrections", None), list):
                    self._pending_corrections = []
                _corr_note = issues_str[:300]
                # Dedup: an identical note already queued for this
                # conversation must not stack. Observed churn shape
                # (2026-07-18, Rick Dangerous): a repeated refute on the
                # same conversation queued the same banner again, each
                # surfaced banner led the next turn, and the model kept
                # re-doing "corrective" work — banners fed the very loop
                # the verdicts were complaining about.
                if any(c.get("note") == _corr_note
                       and c.get("conv") == (conv_fp or "")
                       for c in self._pending_corrections):
                    pretty_log(
                        "Verifier",
                        "identical correction already queued for this "
                        "conversation — not stacking",
                        icon=Icons.BRAIN_THINK,
                    )
                else:
                    self._pending_corrections.append({
                        "note": _corr_note,
                        "conv": conv_fp or "",
                        "ts": time.monotonic(),
                    })
                    if len(self._pending_corrections) > _CORRECTION_MAX:
                        self._pending_corrections = self._pending_corrections[-_CORRECTION_MAX:]
                    pretty_log(
                        "Verifier",
                        "queued a correction to surface on the next message of this conversation",
                        icon=Icons.IDEA,
                    )
        else:
            pretty_log(
                "Verifier",
                f"LATE {v_result.verdict.value} "
                f"({v_result.confidence:.0%}) — informational; "
                "response already sent",
                icon=Icons.VERIFIER_LAB,
            )

    def _consume_pending_corrections(self, messages, conv_fp=None):
        """Stage any correction queued by a previous turn's async verdict so
        it is DETERMINISTICALLY prepended to this turn's reply.

        Earlier this injected a system note asking the model to open with the
        correction — but the model reliably ignored it (it judged an unrelated
        correction irrelevant to the new question, and the trivial-chat fast
        path dropped it entirely). So the banner is now prepended to the
        response text directly (see ``_take_active_correction``), and a turn
        flag skips the trivial fast path (which returns via its own route and
        would bypass the prepend). Returns ``messages`` unchanged.
        """
        corrections = getattr(self, "_pending_corrections", None)
        self._active_correction = ""
        if not corrections:
            self._correction_active_this_turn = False
            return messages

        now = time.monotonic()
        current_fp = (
            conv_fp if conv_fp is not None
            else self._conversation_fingerprint(messages)
        )
        surface = []   # notes belonging to THIS conversation → prepend now
        kept = []      # other conversations' corrections → still fresh, hold
        for c in corrections:
            # Back-compat: a bare string is an untagged/legacy correction —
            # surface it unconditionally (only reachable via direct injection;
            # production always enqueues a tagged dict).
            if isinstance(c, str):
                surface.append(c)
                continue
            if not isinstance(c, dict):
                continue
            if (now - c.get("ts", now)) > _CORRECTION_TTL:
                continue  # expired → drop (its conversation never came back)
            conv = c.get("conv", "")
            # A dict correction with an EMPTY conv can't be safely targeted;
            # per _conversation_fingerprint's own contract (empty → fail-safe
            # DROP rather than risk cross-posting) it must NOT wildcard-surface
            # into an unrelated conversation. Require a real match on both
            # sides. (Bare-string legacy corrections still surface
            # unconditionally — handled by the isinstance(c, str) path above.)
            if current_fp and conv and conv == current_fp:
                surface.append(c.get("note", ""))
            else:
                kept.append(c)  # a different conversation — leave it queued

        # Hold non-matching (still-fresh) corrections for their own
        # conversation, bounded by the cap; drop everything else.
        self._pending_corrections = kept[-_CORRECTION_MAX:]
        self._correction_active_this_turn = bool(surface)
        if not surface:
            return messages

        note = "; ".join(n for n in surface if n)
        self._active_correction = (
            f"⚠️ **Correction to my previous answer:** {note}\n\n---\n\n"
        )
        pretty_log(
            "Verifier",
            f"surfacing {len(surface)} deferred correction(s) from a prior turn "
            "of this conversation",
            icon=Icons.IDEA,
        )
        return messages

    def _take_active_correction(self) -> str:
        """Return the staged correction banner and clear it (one-shot, so it
        prepends to exactly one reply). Empty string when none is staged."""
        banner = getattr(self, "_active_correction", "") or ""
        self._active_correction = ""
        return banner

    def _parse_assistant_tool_calls(self, content, msg):
        """Parse the assistant's raw response into tool calls (extracted
        verbatim from handle_chat, 2026-07-08 — step 1 of the hot-path
        decomposition; behavior-identical by construction).

        Handles every emission shape the upstream produces: the robust
        XML parser with its normalization heals (bare <function>, sloppy
        attribute syntax, CDATA bodies, unclosed parameters), the
        truncation detector, degenerate-flood cap, native ``tool_calls``
        with corruption repair, and the raw-JSON fallback.

        Returns ``(tool_calls, ui_content, parse_failure_reason)``:
        ``ui_content`` is the response with tool XML scrubbed (think
        blocks NOT yet stripped — the caller owns that plus the leak
        scrubbers); ``parse_failure_reason`` is ""/"truncated"/
        "no_function_tag" for the recovery-message branch.
        """
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
        parse_target = _strip_think_blocks(content)

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
                            except Exception:
                                # Never silently dispatch with empty args — the
                                # tool's "missing argument" error reads like the
                                # MODEL's mistake and can trap it in a retry loop.
                                pretty_log(
                                    "Agent Parser",
                                    f"Unparseable JSON arguments for tool "
                                    f"'{t_data.get('name')}' — dispatching empty "
                                    f"args (raw head: {args_val[:120]!r})",
                                    level="WARNING", icon=Icons.WARN,
                                )
                                args_val = {}

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

            # De-corrupt native tool_calls the upstream server
            # mangled by merging a multi-tool reply into one
            # call's arguments (see _repair_native_tool_calls).
            # This is the fix for "introspect keeps erroring on a
            # valid action" — the server leaked the following
            # tool call's XML into the `action` value.
            if tool_calls:
                _avail = (list(self.available_tools.keys())
                          if hasattr(self, 'available_tools') else None)
                # Snapshot the raw pre-repair calls: when a repair fires,
                # log them. Without this the repair was undiagnosable from
                # traces (the AF/43 path=query duplication took a session
                # of guesswork to attribute) — same lesson the XML
                # parse-error path learned on 2026-07-05.
                try:
                    _raw_tc_snapshot = json.dumps(
                        tool_calls, ensure_ascii=False, default=str)[:4096]
                except Exception:
                    _raw_tc_snapshot = str(tool_calls)[:4096]
                tool_calls, _repaired = _repair_native_tool_calls(tool_calls, _avail)
                if _repaired:
                    pretty_log(
                        "Agent Parser",
                        "Repaired corrupt native tool_call(s): upstream merged a "
                        "multi-tool reply into one call's arguments — recovered the "
                        "intended value and split the leaked calls.",
                        level="WARNING", icon=Icons.WARN,
                    )
                    logger.warning(
                        "native tool_call repair fired; raw pre-repair "
                        "calls (truncated to 4 KB): %s",
                        _raw_tc_snapshot.replace("\n", "\\n"),
                    )

            if not tool_calls and parse_target.strip().startswith('{'):
                # Fallback: check if it outputted raw JSON instead of XML
                try:
                    possible_json = extract_json_from_text(parse_target)
                    if possible_json and isinstance(possible_json, dict) and "name" in possible_json and "arguments" in possible_json:
                        args_val = possible_json.get("arguments", {})
                        if isinstance(args_val, str):
                            try: args_val = json.loads(args_val)
                            except Exception:
                                pretty_log(
                                    "Agent Parser",
                                    f"Raw-JSON tool call for "
                                    f"'{possible_json.get('name')}' has non-JSON "
                                    f"arguments — passing the raw string through",
                                    level="WARNING", icon=Icons.WARN,
                                )
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

        return tool_calls, ui_content, parse_failure_reason

    async def _dispatch_and_process_tool_batch(self, ts: "TurnState") -> bool:
        """The tool guard / dispatch / result pipeline — one whole turn's
        tool batch: preamble-rollback bookkeeping, per-tool pre-flight
        guards (strike cap, disabled/unknown tool, idempotency dedupe,
        recent-failure guard, participant/egress checks), parallel
        dispatch, result pairing, failure classification/steering and
        the SYSTEM-3 pivot. Extracted VERBATIM from handle_chat (#5
        step 2) against `TurnState`; the only rewrites are the two
        turn-loop control-flow statements (the region was the loop-body
        tail): returns True to BREAK the turn loop, False to continue
        with the next iteration. State repack happens in `finally` so a
        raising tool path leaves ts as up-to-date as the old inline
        code left handle_chat's locals.
        """
        _constraint_steer_pending = ts._constraint_steer_pending
        _proj_task_closed_this_req = ts._proj_task_closed_this_req
        _request_sys3_fired_once = ts._request_sys3_fired_once
        _request_sys3_prev_justification = ts._request_sys3_prev_justification
        consecutive_parse_errors = ts.consecutive_parse_errors
        current_plan_json = ts.current_plan_json
        execution_failure_count = ts.execution_failure_count
        final_ai_content = ts.final_ai_content
        fname = ts.fname
        force_final_response = ts.force_final_response
        force_stop = ts.force_stop
        forget_was_called = ts.forget_was_called
        last_was_failure = ts.last_was_failure
        preflight_blocks_this_request = ts.preflight_blocks_this_request
        request_sandbox_state = ts.request_sandbox_state
        transient_failure_count = ts.transient_failure_count
        tool_calls = ts.tool_calls
        msg = ts.msg
        ui_content = ts.ui_content
        parse_failure_reason = ts.parse_failure_reason
        model = ts.model
        last_user_content = ts.last_user_content
        char_budget = ts.char_budget
        strikes = ts.strikes
        task_tree = ts.task_tree
        _user_batch_intent = ts._user_batch_intent
        _request_constraints = ts._request_constraints
        repeated_action_steered = ts.repeated_action_steered
        messages = ts.messages
        seen_tools = ts.seen_tools
        executed_idempotent = ts.executed_idempotent
        raw_tools_called = ts.raw_tools_called
        tool_usage = ts.tool_usage
        tools_run_this_turn = ts.tools_run_this_turn
        request_state = ts.request_state
        try:
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
                    except Exception:
                        pretty_log(
                            "Agent Parser",
                            f"Arguments for tool "
                            f"'{tc.get('function', {}).get('name')}' are not valid "
                            f"JSON at the unescape step — leaving them raw",
                            level="WARNING", icon=Icons.WARN,
                        )

            # Reset the per-BATCH read budget before dispatching this
            # assistant message's tool calls. Parallel whole-file reads
            # were overflowing the window: each cleared the per-file cap,
            # but together they didn't fit (observed: two 170+ KB JSONs →
            # 136 K tokens vs a 131 K window → HTTP 400). The budget caps
            # cumulative raw-read bytes for THIS batch; file_system reads
            # charge against it via the registry closure.
            try:
                from ..tools.file_system import ReadBudget, read_byte_budget
                _mc = int(getattr(self.context.args, "max_context", 8192) or 8192)
                _cap = read_byte_budget(_mc)
                # Occupancy-aware shrink (2026-07-18): the per-turn cap alone
                # let a 60-file feasibility session balloon to 398k tokens —
                # every batch cleared its OWN budget while the conversation
                # grew without bound (xrick session: 2 compactions, then an
                # upstream 400). Raw reads this turn may only fill what
                # remains below ~80% of the window; at high occupancy the
                # budget hits zero and every whole-file read is refused with
                # the summarize-first steer (ranged reads stay exempt).
                try:
                    _occ = _estimate_messages_tokens(messages)
                    _headroom = int(max(0, 0.80 * _mc - _occ) * 3.5)
                    _cap = min(_cap, _headroom)
                except Exception:
                    pass
                if getattr(self.context, "_ctx_pressure_lockdown", False):
                    # Second overflow this request → no more whole-file reads.
                    _cap = 0
                self.context._read_budget = ReadBudget(_cap)
            except Exception:
                self.context._read_budget = None

            tool_tasks, tool_call_metadata = [], []
            tool_durations = []  # parallel to tool_tasks; filled by the timing shim (metacog anomaly window)
            # Idempotent setters dispatched in THIS batch. The guard
            # checks it alongside `executed_idempotent` so two
            # identical calls in one response still dedupe, while
            # the durable hash only commits at result time on
            # SUCCESS (a failed call must not block its own
            # corrected retry on the next iteration).
            pending_idempotent = set()
            # Batch-local collapse of byte-identical READ-ONLY calls: a
            # single response can carry a runaway burst of duplicates
            # (2026-07-17 22:14 request: one batch held 144 identical
            # file_system reads of the same path — the no-progress
            # breaker aborted correctly, but only after the sandbox
            # executed every one of them). Duplicates execute ONCE; the
            # other tool_call_ids receive a copy of that result, so the
            # model still gets one reply per call and the no-progress
            # ledger still counts every repeat (the breaker's view is
            # unchanged). Mutating calls are never collapsed — repeating
            # a write can be intentional.
            batch_seen_reads: dict = {}   # a_hash -> index of first dispatch
            batch_dup_of: dict = {}       # dup index -> source index
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
                    # Acquired-skill registry mutations
                    # (create_skill / manage_skills delete) change
                    # the schema list the LLM sees. Without this
                    # the new skill is unreachable until the next
                    # user message — observed loop where the model
                    # spent 11 min narrating "now invoking X" but
                    # never emitting a real tool_call because the
                    # cached schema didn't contain X yet.
                    if fname == "create_skill" or (
                        fname == "manage_skills"
                        and (t_args.get("action") or "").lower() == "delete"
                    ) or (
                        fname == "manage_composed_skills"
                        and (t_args.get("action") or "").lower() in ("define", "delete", "approve")
                    ):
                        request_state.invalidate_tool_defs()

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
                              (fname == "knowledge_base" and t_args.get("action") in ["ingest_document", "forget", "reset_all", "insert_fact"]) or \
                              (fname == "manage_composed_skills" and t_args.get("action") in ["define", "approve", "delete"])

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
                if is_idempotent_setter and (
                        a_hash in executed_idempotent
                        or a_hash in pending_idempotent):
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
                # NB: `executed_idempotent.add(a_hash)` is deliberately
                # deferred until AFTER the tool is actually dispatched
                # (see the dispatch site) — recording it here marked the
                # call "done" even when a downstream gate (metacog
                # ask_user) aborted the dispatch, which then blocked the
                # user's legitimate re-issue as a false duplicate.

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

                    # PARTICIPANT-MODE ENGINE GUARD (deterministic).
                    # Steers ask the model to reconsider; this guard
                    # refuses. When the request carries a
                    # participant constraint ("YOU play against
                    # me"), a write/replace whose body embeds
                    # move-selection logic is rejected BEFORE
                    # dispatch — the 2026-07-05 chess session
                    # shipped random.choice(legal_moves) past both
                    # the system prompt and the post-write steer.
                    if op in ("write", "replace"):
                        from ..utils.constraints import (
                            participant_write_violation,
                        )
                        _pv_msg = participant_write_violation(
                            _request_constraints, t_args)
                        if _pv_msg:
                            _pv_path = str(t_args.get("path")
                                           or t_args.get("filename")
                                           or "?")
                            pretty_log(
                                "Local Guard",
                                f"Blocked file_system {op} embedding "
                                f"move-selection logic "
                                f"(path={_pv_path!r}) — participant "
                                f"constraint active",
                                icon=Icons.STOP,
                            )
                            err_msg = {"role": "tool",
                                       "tool_call_id": tool["id"],
                                       "name": fname,
                                       "content": _pv_msg}
                            messages.append(err_msg)
                            tools_run_this_turn.append(
                                {**err_msg, "_synthetic": True})
                            execution_failure_count += 1
                            last_was_failure = True
                            continue

                        target_path = str(t_args.get("path", "")).lower()

                if fname not in self.available_tools:
                    self._rebuild_available_tools()

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
                    # Metacog mid-turn arbiter gate (roadmap phase 3,
                    # auto-route). Fires only when:
                    #   * the bundle is enabled and arbiter wired,
                    #   * the tool is in a mutating-host domain
                    #     (shell/sql — the irrecoverable kind),
                    #   * the last composite confidence reading
                    #     was below threshold,
                    #   * the per-request arbitration cap (1) is
                    #     unspent.
                    # All five gates live on the bundle; this site
                    # just awaits the decision and acts on it.
                    _mc_gate = getattr(self.context, "metacog", None)
                    _gate_decision = None
                    if (_METACOG_ARBITER_ENABLED and _mc_gate is not None
                            and getattr(_mc_gate, "enabled", False)):
                        try:
                            _gate_decision = await _mc_gate.arbitrate_tool_calls(
                                messages=messages, tool_name=fname,
                            )
                        except Exception as _gex:
                            logger.debug(
                                "metacog arbiter gate failed: %s", _gex,
                            )
                    if _gate_decision is not None:
                        # Severity policy: `ask_user` is an
                        # operational pause — surface as WARN
                        # so it doesn't get lost in the INFO
                        # stream. `execute` / `validate` are
                        # routine.
                        from .metacog_log import (
                            emit as _mc_emit,
                            Subsystem as _mc_ss,
                            LEVEL_INFO, LEVEL_WARN,
                        )
                        _arb_lvl = (
                            LEVEL_WARN
                            if _gate_decision.action == "ask_user"
                            else LEVEL_INFO
                        )
                        _mc_emit(
                            _mc_ss.ARBITER,
                            level=_arb_lvl,
                            tool=fname,
                            action=_gate_decision.action,
                            sim=_gate_decision.similarity,
                            candidates=len(_gate_decision.candidates),
                            # Full reason — no 80-char truncation;
                            # the helper auto-quotes the spaces.
                            reason=_gate_decision.reason,
                        )
                        if _gate_decision.action == "ask_user":
                            # Hard block: skip the dispatch and
                            # replace it with a synthetic tool
                            # result that surfaces the divergence
                            # to the model. The model's next turn
                            # will produce a clarification request
                            # to the user instead of charging ahead.
                            _diag = (
                                f"SYSTEM PAUSE — metacog arbiter detected "
                                f"ambiguous intent for a {fname} action. "
                                f"Two candidate plans diverged "
                                f"(sim={_gate_decision.similarity:.2f}) "
                                f"and the rule-based validator could not "
                                f"pick a clear winner. Reason: "
                                f"{_gate_decision.reason}. Ask the user "
                                f"to clarify what they want done before "
                                f"re-emitting any {fname} call."
                            )
                            err_msg = {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "name": fname,
                                "content": _diag,
                            }
                            messages.append(err_msg)
                            tools_run_this_turn.append(
                                {**err_msg, "_synthetic": True},
                            )
                            last_was_failure = True
                            continue

                    # ── Pre-flight repeat-failure guard (feature 1A) ──
                    # Before dispatch, ask whether this exact action
                    # (tool + primary target) already failed the same
                    # way in the recent window. If so, skip the call and
                    # hand the model the prior error instead of burning
                    # another turn re-running a known failure — the live
                    # counterpart to the offline post-mortem repeated-
                    # error fingerprint. Idempotent setters are exempt
                    # (the idempotency guard above already covers them,
                    # and they carry no error to repeat).
                    if self._preflight_guard_enabled and not is_idempotent_setter:
                        _pf_target = primary_target_from_args(t_args)
                        _pf_op = str(t_args.get("operation")
                                     or t_args.get("action") or "")
                        _pf_err = self._failure_guard.would_repeat(
                            fname, _pf_target, _pf_op)
                        if _pf_err:
                            pretty_log(
                                "Pre-Flight Guard",
                                f"Blocked repeat {fname}"
                                + (f" on {_pf_target}" if _pf_target else "")
                                + f" — already failed: {_pf_err}",
                                level="WARNING", icon=Icons.STOP,
                            )
                            preflight_blocks_this_request += 1
                            _diag = (
                                f"SYSTEM BLOCK — pre-flight guard: this exact "
                                f"'{fname}' call"
                                + (f" (operation='{_pf_op}')" if _pf_op else "")
                                + (f" on target '{_pf_target}'" if _pf_target else "")
                                + f" already failed recently with: \"{_pf_err}\". "
                                f"Re-running it UNCHANGED will fail the same way. "
                                f"Legal ways forward: use a DIFFERENT operation of "
                                f"the same tool (e.g. operation='write' with the "
                                f"full file instead of 'replace'), a different "
                                f"tool, fix the underlying cause first, or ask "
                                f"the user."
                            )
                            if preflight_blocks_this_request >= 2:
                                # Guard-block budget (2026-07-08): a
                                # model re-issuing known-identical
                                # failures twice is boxed in — force a
                                # final reply instead of letting it
                                # spin to the turn cap (observed live:
                                # 3 blocked turns x ~80 s of full-file
                                # generation each).
                                force_final_response = True
                                _diag += (
                                    " FINAL: you have now been blocked "
                                    f"{preflight_blocks_this_request} times on known-"
                                    "identical failures. STOP calling tools. Write "
                                    "your final reply: state what you tried, quote "
                                    "the exact error, and ask the user for the one "
                                    "thing you need to proceed."
                                )
                            err_msg = {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "name": fname,
                                "content": _diag,
                            }
                            messages.append(err_msg)
                            tools_run_this_turn.append(
                                {**err_msg, "_synthetic": True},
                            )
                            last_was_failure = True
                            continue
                    try:
                        # Wrap in a timing shim so each tool's wall-clock
                        # duration lands in tool_durations[idx] (parallel to
                        # results/metadata) — feeds the metacog runtime-budget
                        # anomaly window.
                        _dup_src_idx = None if is_mutating else batch_seen_reads.get(a_hash)
                        if _dup_src_idx is not None:
                            # Duplicate read-only call in the SAME batch —
                            # register a placeholder task; the result is
                            # copied from the first instance after the
                            # gather phases. The coroutine is deliberately
                            # never created (an unawaited coroutine warns).
                            batch_dup_of[len(tool_tasks)] = _dup_src_idx
                            tool_tasks.append(None)
                            tool_durations.append(None)
                        else:
                            _coro = self.available_tools[fname](**t_args)
                            tool_tasks.append(_timed_tool_coro(_coro, tool_durations, len(tool_tasks)))
                            tool_durations.append(None)
                            if not is_mutating:
                                batch_seen_reads[a_hash] = len(tool_tasks) - 1
                        tool_call_metadata.append((fname, tool["id"], a_hash, is_mutating, primary_target_from_args(t_args), is_idempotent_setter, str(t_args.get("operation") or t_args.get("action") or "")))
                        # The DURABLE idempotency hash is recorded at
                        # the RESULT-processing site, and only when
                        # the call actually SUCCEEDED. Recording it
                        # here (at dispatch) marked a call "applied"
                        # even when the tool returned an Error —
                        # observed live 2026-07-05: update_profile
                        # rejected a missing-value call, the model's
                        # corrected retry was then blocked as a
                        # "duplicate (args already applied)", and the
                        # turn finalised on a false "Done — removed".
                        # The batch-local pending set below still
                        # dedupes identical calls within THIS
                        # response.
                        if is_idempotent_setter:
                            pending_idempotent.add(a_hash)
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
                    if task is None:
                        continue  # batch-dedup placeholder, filled below
                    if meta[0] == "file_system":
                        mutation_coros.append((i, task))

                if mutation_coros:
                    mut_results = await asyncio.gather(*(c[1] for c in mutation_coros), return_exceptions=True)
                    for (i, _), res in zip(mutation_coros, mut_results):
                        results[i] = res

                # Phase 2: Executions
                exec_coros = []
                for i, (task, meta) in enumerate(zip(tool_tasks, tool_call_metadata)):
                    if task is None:
                        continue  # batch-dedup placeholder, filled below
                    if meta[0] != "file_system":
                        exec_coros.append((i, task))

                if exec_coros:
                    exec_results = await asyncio.gather(*(c[1] for c in exec_coros), return_exceptions=True)
                    for (i, _), res in zip(exec_coros, exec_results):
                        results[i] = res

                # Phase 3: fan the first instance's result out to its
                # batch-local duplicates (read-only calls collapsed at
                # dispatch — see batch_seen_reads above).
                if batch_dup_of:
                    for i, src in batch_dup_of.items():
                        results[i] = results[src]
                    pretty_log(
                        "Batch Dedup",
                        f"Collapsed {len(batch_dup_of)} duplicate read-only "
                        f"call(s) in one batch — executed once, result shared.",
                        icon=Icons.SKIP,
                    )

                turn_has_failure = False
                last_error_res = ""
                last_error_preview = "Unknown Error"

                # Per-call outcomes for this turn. A multi-id command
                # ("delete A and B") is N separate tool calls; tracking
                # each one lets the failure path emit an explicit
                # "X succeeded, Y failed" summary instead of collapsing
                # a partial failure into a single generic strike.
                op_outcomes = []

                # Tracks the worst no-progress signature seen this
                # turn (the same SUCCEEDING action+target+result
                # repeating), handled once after the results loop.
                _noprogress_trip = None

                # We reached the parallel-execution path, which means
                # at least one tool call parsed cleanly this turn.
                # Drop the consecutive-parse-error streak so a single
                # earlier failure can't latch the pivot prompt.
                consecutive_parse_errors = 0
                for i, result in enumerate(results):
                    fname, tool_id, a_hash, is_mutating, ptarget, _is_idem_setter, ptool_op = tool_call_metadata[i]
                    str_res = str(result).replace("\r", "") if not isinstance(result, Exception) else f"Error: {str(result)}"

                    # Record this call's outcome uniformly (before the
                    # branch chain below) so a partial failure can be
                    # summarised as "N ok / M failed" rather than
                    # last-write-wins on last_error_*.
                    _res_is_error = str_res.startswith((
                        "Error:", "ERROR", "SYSTEM ERROR", "Critical Tool Error"))

                    # Idempotency hash lands only on SUCCESS: a failed
                    # setter was NOT applied, so an identical corrected
                    # retry must be allowed through the guard.
                    if _is_idem_setter and not _res_is_error:
                        executed_idempotent.add(a_hash)
                    op_outcomes.append({
                        "tool": fname,
                        "ok": not _res_is_error,
                        "preview": (str_res.replace("Error:", "").strip()[:140]
                                    if _res_is_error else None),
                    })

                    # Feed the pre-flight repeat-failure guard (feature
                    # 1A): remember FAILED calls keyed by (tool, primary
                    # target) — ``ptarget`` was computed at dispatch time
                    # from the same args — so an identical re-issue on a
                    # later iteration is intercepted before dispatch.
                    if _res_is_error:
                        try:
                            self._failure_guard.record(fname, ptarget, str_res, ptool_op)
                        except Exception:
                            pass

                    # One-task-per-turn gate: a manage_projects call that
                    # actually closed a task to DONE ends the interactive
                    # turn, so a single "start task 1" advances exactly
                    # one task and stops. Skipped when the user asked for
                    # a batch ("do the next 3" / "finish the project").
                    if (not _proj_task_closed_this_req and not _user_batch_intent
                            and _manage_projects_closed_a_task(fname, str_res)):
                        _proj_task_closed_this_req = True
                        force_final_response = True
                        logger.info(
                            "one-task-per-turn: project task closed DONE "
                            "— forcing turn to wrap up and wait for the user")

                    # One-shot constraint check after the FIRST
                    # successful artifact write of a constrained
                    # request. The chess incident showed the
                    # constraint is dropped DURING generation (a
                    # 183s monolith started seconds after the model
                    # read "don't come up with some random AI"), so
                    # the steer lands right after the write — the
                    # earliest moment the model can still re-open
                    # the file and fix a violation cheaply.
                    if (_constraint_steer_pending and not _res_is_error
                            and fname == "file_system" and is_mutating):
                        _constraint_steer_pending = False
                        # Participant-role constraints get the
                        # architecture directive appended: the
                        # 2026-07-04 chess session showed the generic
                        # reminder alone is rationalised away ("the
                        # evaluation function IS me") — the steer must
                        # name the only two designs that satisfy the
                        # constraint, including the /api/game/move
                        # endpoint the model keeps forgetting exists.
                        from ..utils.constraints import (
                            PARTICIPANT_STEER,
                            has_participant_constraint,
                        )
                        _steer_txt = (
                            "SYSTEM ALERT (constraint check): you "
                            "just wrote an artifact, and this "
                            "request carries EXPLICIT USER "
                            "CONSTRAINTS:\n"
                            + "\n".join(f"- {c}" for c in
                                        _request_constraints)
                            + "\nBefore replying or marking "
                            "anything done: re-read what you "
                            "wrote and verify NONE of these are "
                            "violated. A coded stand-in for a "
                            "role the user assigned to YOU "
                            "(e.g. an embedded AI opponent when "
                            "the user said YOU will play) is a "
                            "violation — fix the artifact NOW "
                            "if so, then continue."
                        )
                        if has_participant_constraint(
                                _request_constraints):
                            _steer_txt += "\n\n" + PARTICIPANT_STEER
                        messages.append({
                            "role": "user",
                            "content": _steer_txt,
                        })
                        pretty_log(
                            "Constraint Check",
                            f"steer injected after first write "
                            f"({len(_request_constraints)} active "
                            f"constraint(s))",
                            icon=Icons.CONSTRAINT,
                        )

                    # Metacog per-tool outcome (roadmap phase 2.3):
                    # record THIS result's tool keyed on its OWN
                    # success — not (as before) a single post-loop call
                    # using the last tool's name + the turn-wide failure
                    # flag, which mis-attributed competence whenever a
                    # turn dispatched multiple tools in parallel.
                    _mc = getattr(self.context, "metacog", None)
                    if _mc is not None and getattr(_mc, "enabled", False) and fname:
                        _lstr = str_res.lstrip()
                        # Any NON-ZERO exit code is a failure, not just
                        # 1/2. The old substring check recorded codes
                        # 3-9 and multi-digit (127, 130, ...) as
                        # competence SUCCESS, poisoning the per-domain
                        # profile. Reuse the same regex the execute
                        # result-classification path below already uses.
                        _mc_exit = re.search(r"EXIT CODE:\s*(\d+)", str_res)
                        _tool_failed = isinstance(result, Exception) or (
                            _lstr.startswith(("Error", "ERROR", "SYSTEM ERROR", "Critical Tool Error"))
                            or "Traceback" in str_res
                            or (_mc_exit is not None and int(_mc_exit.group(1)) != 0)
                        )
                        _dur = tool_durations[i] if i < len(tool_durations) else None
                        _bus = getattr(_mc, "bus", None)
                        # Trigger publishes are best-effort and MUST NOT
                        # block the competence/budget record below — hence a
                        # separate try. is_anomalous runs BEFORE record_outcome
                        # so the current sample isn't compared against itself.
                        try:
                            # LoopDetected: observe the tool key; on a
                            # repeat-streak trip publish a loop event (the
                            # ReplanBridge can revise the active task), reset.
                            _rep = getattr(_mc, "repetition", None)
                            if _rep is not None and _bus is not None and hasattr(_rep, "observe"):
                                _streak = _rep.observe(fname)
                                if _rep.tripped():
                                    from .triggers import loop_event as _loop_event
                                    await _bus.publish(_loop_event(
                                        f"tool '{fname}' repeated {_streak}x in a row",
                                        key=fname, count=_streak,
                                    ))
                                    _rep.reset()
                            # ExecutionAnomaly: this duration vs the learned p95×budget.
                            _rb = getattr(_mc, "runtime_budget", None)
                            if (_rb is not None and _bus is not None and _dur is not None
                                    and hasattr(_rb, "is_anomalous") and _rb.is_anomalous(fname, _dur)):
                                from .triggers import anomaly_event as _anom_event
                                await _bus.publish(_anom_event(
                                    f"tool '{fname}' ran {_dur:.1f}s (>budget)",
                                    tool_name=fname, duration_s=_dur,
                                    budget_s=(_rb.budget(fname) or 0.0),
                                ))
                        except Exception as _trexc:
                            logger.debug("metacog trigger publish failed: %s", _trexc)
                        try:
                            # Feed the budget window + competence profile.
                            _mc.record_outcome(fname, success=not _tool_failed, duration_s=_dur)
                        except Exception as _mcexc:
                            logger.debug("metacog outcome hook failed: %s", _mcexc)

                    shield_limit = max(16000, int(char_budget * 0.1))
                    if len(str_res) > shield_limit and fname not in ["file_system", "recall", "deep_research", "web_search", "knowledge_base", "postgres_admin"]:
                        # Use a DISTINCT variable, not `payload`: the
                        # outer `payload` is reused later (e.g. the
                        # Perfect-It optimization call), and rebinding
                        # it here to this 300-token summarizer silently
                        # capped that later generation's max_tokens.
                        shield_payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": f"The user asked: '{last_user_content}'. Summarize this tool output. If it contains facts relevant to the user, extract them. If it is a script error, state the root cause. Output: {str_res[:15000]}"}],
                            "temperature": 0.0,
                            "max_tokens": 300
                        }
                        try:
                            pretty_log("Context Shield", f"Offloading {len(str_res)} chars from {fname} to Edge Worker...", icon=Icons.SHIELD)
                            # FOREGROUND: awaited inline mid-turn. With
                            # is_background=True this parked against its
                            # own request in _wait_for_foreground_clear
                            # (600s self-stall) on every oversized tool
                            # output. use_worker still prefers a pool.
                            summary_data = await self.context.llm_client.chat_completion(shield_payload, use_worker=True, is_background=False, task_label="shield")
                            summary_content = summary_data["choices"][0]["message"].get("content", "").strip()
                            if summary_content:
                                str_res = f"[EDGE CONDENSED]: {summary_content}"
                        except Exception as e:
                            logger.debug(f"Context-Shield edge summarisation failed: {type(e).__name__}: {e}")

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
                    _hint_exit = re.search(r"EXIT CODE:\s*(\d+)", str_res)
                    if (str_res.lstrip().startswith("Error")
                            or "Traceback" in str_res
                            or "SYSTEM ERROR" in str_res
                            or (_hint_exit is not None and int(_hint_exit.group(1)) != 0)):
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

                    # No-progress (ungrounded-verification) loop
                    # detection — ONLY on SUCCESSFUL, NON-MUTATING
                    # calls. Errors are already handled by the strike
                    # path + `_note_repeated_failure`; mutations
                    # (file writes) are exempt so legitimate iterative
                    # editing of one file is never mistaken for a loop.
                    # What's left — repeated reads of the same file,
                    # repeated browser interaction with the same
                    # selector, repeated screenshots returning the same
                    # view — is exactly the "succeeds but learns
                    # nothing" thrash from the Browser-OS run.
                    _res_is_error = isinstance(result, Exception) or str_res.lstrip().startswith(
                        ("Error", "ERROR", "SYSTEM ERROR", "Critical Tool Error")
                    ) or "Traceback" in str_res
                    # World-changed reset: a SUCCESSFUL file mutation
                    # invalidates every no-progress observation counted so
                    # far — re-observing after an edit is verification of
                    # the change, not thrash, even when the observation
                    # comes back byte-identical (a landing page can be
                    # unchanged while the fix targets a deeper state).
                    # Scoped to file_system mutations on purpose: `execute`
                    # probes stay exempt, so a probe loop that keeps
                    # returning identical results still trips the breaker
                    # (the steer message explicitly tells the model to
                    # probe — resetting on probes would nullify the trip).
                    # See StrikeLedger.note_world_changed for the observed
                    # failure this closes.
                    if (fname == "file_system" and is_mutating
                            and not _res_is_error):
                        strikes.note_world_changed()
                        repeated_action_steered.clear()
                        _noprogress_trip = None
                    # Work-log accumulation (2026-07-18): while a project is
                    # bound, record which files this request mutated and
                    # which work tools ran successfully. Read once by the
                    # finalize chain to write the project's automatic
                    # per-request work_log event — the write-back that
                    # finally makes interactive (incl. post-DONE debugging)
                    # work visible to future turns.
                    if not _res_is_error and fname and getattr(
                            self.context, "current_project_id", None):
                        try:
                            if fname == "file_system" and is_mutating and ptarget:
                                getattr(self.context, "_project_work_files",
                                        set()).add(str(ptarget))
                            if (fname in ("execute", "browser",
                                          "vision_analysis")
                                    or (fname == "file_system" and is_mutating)):
                                _wt = getattr(self.context,
                                              "_project_work_tools", None)
                                if isinstance(_wt, dict):
                                    _wt[fname] = _wt.get(fname, 0) + 1
                            # Command heads for the work log (2026-07-18):
                            # execute-created files (git clone, script
                            # outputs) are invisible to the file accumulator,
                            # so a failed turn's work-log said nothing about
                            # the clone that HAD succeeded — and the retry
                            # re-cloned into the existing dir (strike). The
                            # command text itself is the record.
                            if fname == "execute":
                                _wc = getattr(self.context,
                                              "_project_work_cmds", None)
                                if isinstance(_wc, list) and len(_wc) < 5:
                                    _cargs = json.loads(a_hash.split(":", 1)[1])
                                    _cmd = " ".join(str(
                                        _cargs.get("command")
                                        or _cargs.get("filename") or "").split())[:90]
                                    if _cmd:
                                        _wc.append(_cmd)
                        except Exception:
                            pass
                        # Off-project escape steer (2026-07-18): a SUCCESSFUL
                        # call that operated at the sandbox root while a
                        # project is bound gets one corrective steer per
                        # request. The remap heal only fires on errors, so a
                        # clean `cd /workspace && git clone` (request
                        # 6f14407f) used to escape every guard: repo +
                        # feasibility report at the root, project dir empty.
                        try:
                            _pid_now = self.context.current_project_id
                            if not getattr(self.context,
                                           "_offproject_steer_done", False):
                                _off = _offproject_target(
                                    fname, ptarget, a_hash, _pid_now)
                                if _off:
                                    self.context._offproject_steer_done = True
                                    pretty_log(
                                        "Project Scope",
                                        f"'{fname}' touched {_off} — outside "
                                        f"projects/{_pid_now}/ — steering "
                                        "relocation",
                                        level="WARNING", icon=Icons.WARN,
                                    )
                                    messages.append({"role": "user", "content": (
                                        f"SYSTEM ALERT (project scope): your last "
                                        f"'{fname}' call touched {_off}, which is "
                                        f"OUTSIDE the active project's workspace "
                                        f"(/workspace/projects/{_pid_now}/). Project "
                                        "files MUST live inside the project directory "
                                        "— anything at the sandbox root is invisible "
                                        "to the project's manifest, briefing, and "
                                        "cleanup. If you just created files there, "
                                        "MOVE them into the project now (e.g. "
                                        f"`mv /workspace/<name> /workspace/projects/"
                                        f"{_pid_now}/<name>`), then continue with "
                                        "RELATIVE paths — the shell already starts "
                                        "inside the project directory."
                                    )})
                        except Exception:
                            pass
                    # Edit-run futility breaker (2026-07-18). Every existing
                    # breaker watches FAILURES or identical observations —
                    # all-success thrash at the GOAL level was invisible: the
                    # xrick session rewrote extract_data.py 5x and reran it
                    # 5x (each run exit 0, counts still wrong) for 33 minutes
                    # until the thinking-loop guard killed the request. On
                    # the 3rd rewrite + 2nd rerun of the same code file, one
                    # strategy-shift steer: record confirmed facts NOW,
                    # verify a minimal slice, switch approach class. Tracked
                    # for ALL requests, project-bound or not.
                    try:
                        _si = getattr(self.context, "_script_iter", None)
                        if isinstance(_si, dict) and fname and not _res_is_error:
                            _CODE_EXTS = (".py", ".sh", ".js", ".mjs", ".ts")
                            if fname == "file_system" and is_mutating and ptarget:
                                _bn = str(ptarget).replace("\\", "/").rsplit("/", 1)[-1].lower()
                                if _bn.endswith(_CODE_EXTS):
                                    _rec = _si.setdefault(_bn, {"writes": 0, "runs": 0})
                                    _rec["writes"] += 1
                            elif fname == "execute":
                                _blob = str(a_hash).lower()
                                for _bn, _rec in _si.items():
                                    if _bn in _blob:
                                        _rec["runs"] += 1
                            if not getattr(self.context, "_futility_steer_done", False):
                                for _bn, _rec in _si.items():
                                    if _rec["writes"] >= 3 and _rec["runs"] >= 2:
                                        self.context._futility_steer_done = True
                                        pretty_log(
                                            "Futility Breaker",
                                            f"'{_bn}' rewritten {_rec['writes']}x and "
                                            f"rerun {_rec['runs']}x this request without "
                                            "a declared result — steering strategy shift",
                                            level="WARNING", icon=Icons.WARN,
                                        )
                                        messages.append({"role": "user", "content": (
                                            f"SYSTEM ALERT (futility breaker): you have "
                                            f"rewritten '{_bn}' {_rec['writes']} times and "
                                            f"rerun it {_rec['runs']} times this request "
                                            "and the goal is still not met. STOP patching "
                                            "the same approach. Do these IN ORDER:\n"
                                            "1. RECORD every fact you have CONFIRMED about "
                                            "the data/format so far (exact structures, "
                                            "counts, edge cases) in the project ledger "
                                            "(manage_projects action=ledger) or a notes "
                                            "file NOW — discoveries must survive this "
                                            "request.\n"
                                            "2. SHRINK the problem: make the script "
                                            "process the SINGLE smallest unit (one "
                                            "record/sprite/tile), print its parsed output "
                                            "next to the raw source lines, and verify that "
                                            "one unit end-to-end before scaling up.\n"
                                            "3. If the approach CLASS keeps failing (e.g. "
                                            "regex over nested structures), switch class — "
                                            "a tokenizer / brace-counting walk / an "
                                            "existing parser — instead of tuning the old "
                                            "one.\n"
                                            "If the goal metric has not moved after ONE "
                                            "more iteration, stop and report exactly what "
                                            "is known and what is blocked."
                                        )})
                                        break
                    except Exception:
                        pass

                    if fname and not is_mutating and not _res_is_error:
                        # threshold=2 (was 3): the SECOND identical
                        # read is already zero-information — nudge
                        # immediately, abort on the third (chess
                        # session 2026-07-08: 5 identical re-reads of
                        # index.html while theorizing about a URL one
                        # probe would have settled).
                        _asig, _acnt, _atrip = strikes.note_action(
                            fname, ptarget,
                            _action_result_fingerprint(str_res),
                            threshold=2,
                        )
                        if _atrip and (_noprogress_trip is None or _acnt > _noprogress_trip[1]):
                            _noprogress_trip = (_asig, _acnt, fname, ptarget)

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

                # (Metacog per-tool outcomes are now recorded inside
                # the enumerate(results) loop above, keyed per result.)

                # No-progress loop breaker. When the same SUCCEEDING
                # action on the same target returned the same result
                # >=3x this request, the agent is in an ungrounded
                # verification loop (it keeps re-observing instead of
                # trusting evidence it already has). FIRST trip: force
                # a grounded final answer (drop tools next turn via
                # force_final_response) + tell it to trust the
                # authoritative state and report how to verify if it
                # cannot. If it somehow keeps looping to >=5, hard-stop.
                # Independent of turn_has_failure — this path is for
                # all-success thrash, which the strike machinery below
                # never sees.
                if _noprogress_trip is not None and not force_stop and not force_final_response:
                    _asig, _acnt, _afname, _atarget = _noprogress_trip
                    _tgt_desc = f" on '{_atarget}'" if _atarget else ""
                    if _acnt >= 3:
                        pretty_log(
                            "Loop Breaker",
                            f"No-progress loop: '{_afname}'{_tgt_desc} repeated {_acnt}x "
                            "with no change — aborting turn loop.",
                            level="WARNING", icon=Icons.STOP,
                        )
                        if not final_ai_content:
                            final_ai_content = (
                                f"[ATTEMPT_ABORTED_NO_PROGRESS] I repeated the same "
                                f"'{_afname}' action{_tgt_desc} {_acnt} times and got the "
                                "same result each time, so I stopped instead of looping. "
                                "Any changes I made so far are in place. To move forward I "
                                "need one piece of real evidence I could not get from here: "
                                "the exact error text or failing URL from your side (e.g. "
                                "browser devtools), or the output of re-running the failing "
                                "step — send me that and I'll fix the actual cause instead "
                                "of guessing."
                            )
                        force_stop = True
                    elif _asig not in repeated_action_steered:
                        repeated_action_steered.add(_asig)
                        # Read/write tools (manage_composed_skills,
                        # file_system, ...) get a SOFTER steer: the
                        # loop is the agent re-READING to orient
                        # itself, and the action it was asked to do is
                        # a WRITE through the SAME tool. Forcing a
                        # text-only final turn here would bar that
                        # pending mutation forever — the agent would
                        # "finish" having silently done nothing (the
                        # reconfigure-a-composed-skill bug: looped on
                        # action="list", force-finalised, never reached
                        # action="define"). So we keep tools available
                        # and tell it to perform the WRITE now instead
                        # of reading again. The >=5 hard stop above is
                        # the backstop if it keeps thrashing.
                        _readwrite_loop = _is_readwrite_loop_exempt(_afname)
                        pretty_log(
                            "Loop Breaker",
                            f"No-progress: '{_afname}'{_tgt_desc} repeated {_acnt}x with no "
                            "new info — "
                            + ("steering to the write action (tools kept)."
                               if _readwrite_loop
                               else "forcing a grounded conclusion."),
                            level="WARNING", icon=Icons.WARN,
                        )
                        if _readwrite_loop:
                            messages.append({"role": "user", "content": (
                                f"SYSTEM ALERT: You have run '{_afname}'{_tgt_desc} {_acnt} "
                                "times and gotten the SAME result — re-reading produces NO new "
                                "information. You already have the current state; it is "
                                "AUTHORITATIVE. Do ONE of these NOW instead:\n"
                                "1. GATHER NEW EVIDENCE: probe the thing you are theorizing "
                                "about — `execute` a curl/run of the exact URL/command/code in "
                                "question and read the actual response, instead of predicting "
                                "it from the source.\n"
                                f"2. APPLY THE CHANGE: if the task needs a change, call "
                                f"'{_afname}' ONCE with the mutating action "
                                "(write/update/create) and report what you changed.\n"
                                "3. ASK THE USER: if the failure is on THEIR side (their "
                                "browser, their process, their network), ask for the exact "
                                "error text/URL from their screen — one question beats "
                                "guessing.\n"
                                "Do NOT issue another read/list of this tool."
                            )})
                        else:
                            force_final_response = True
                            messages.append({"role": "user", "content": (
                                f"SYSTEM ALERT: You have run '{_afname}'{_tgt_desc} {_acnt} times "
                                "and gotten the SAME result with no change — re-observing "
                                "produces NO new information. The evidence you already have is "
                                "AUTHORITATIVE. Write your FINAL answer now: if you confirmed "
                                "the change via state/file inspection, report success and how "
                                "you confirmed it. If you are still UNSURE why something fails, "
                                "do not guess — say plainly what you verified, what you could "
                                "not verify from here, and ask the user for the ONE piece of "
                                "evidence that would settle it (the exact error text, the "
                                "failing URL from their devtools, or the output of a command "
                                "you give them). Do NOT call this tool again."
                            )})

                if turn_has_failure:
                    # Any failure (transient or structural) breaks the
                    # consecutive-clean-success streak that unfreezes decay.
                    strikes.reset_clean_streak()
                    # Classify the failure to route to the right budget
                    from ..tools.tool_failure import classify_tool_failure, FailureClass, format_failure_context, summarize_multi_op_outcomes
                    failure_class, failure_match = classify_tool_failure(last_error_res or last_error_preview)
                    # Capture the failure head for harness-dimension
                    # attribution at finalize time (bounded: 6 per turn).
                    _ftexts = getattr(self.context, "_turn_failure_texts", None)
                    if isinstance(_ftexts, list) and len(_ftexts) < 6:
                        _ftexts.append(
                            f"{fname} [{failure_class.value}]: "
                            f"{(last_error_res or last_error_preview)[:300]}")
                    # Partial-failure summary (empty unless this turn
                    # mixed successes and failures across >=2 calls).
                    multi_op_summary = summarize_multi_op_outcomes(op_outcomes)

                    if failure_class == FailureClass.RETRYABLE:
                        transient_failure_count += 1
                        pretty_log("Transient Fail", f"Transient strike {transient_failure_count}/4 ({failure_match}) -> {last_error_preview[:100]}", icon=Icons.WARN)
                        diagnostic_msg = format_failure_context(last_error_preview, failure_class)
                    else:
                        execution_failure_count += 1
                        pretty_log("Execution Fail", f"Strike {execution_failure_count}/6 ({failure_class.value}) -> {last_error_preview[:150]}", icon=Icons.FAIL)
                        diagnostic_msg = format_failure_context(last_error_preview, failure_class)
                        # Detect the SAME structural failure recurring.
                        # At ≥3 repeats: freeze the success-decay (so the
                        # cap can finally fire on this oscillating loop)
                        # and tell the model ONCE to stop retrying — the
                        # live tool result is authoritative over any stale
                        # context/system-state hint that says otherwise.
                        _sig, _cnt, _persist, _first_warn = strikes.note_failure(
                            fname, last_error_preview
                        )
                        if _first_warn:
                            pretty_log(
                                "Loop Breaker",
                                f"Same failure ×{_cnt} "
                                f"({fname}) — freezing strike decay & redirecting.",
                                level="WARNING", icon=Icons.STOP,
                            )
                            messages.append({"role": "user", "content": (
                                f"SYSTEM ALERT: This exact action has now failed "
                                f"{_cnt} times with the SAME error: "
                                f"{str(last_error_preview)[:160]}. STOP repeating it — "
                                "retrying will not change the result. The live tool "
                                "result and the current sandbox listing are AUTHORITATIVE "
                                "over any prior context, memory, workspace narrative, or "
                                "DYNAMIC SYSTEM STATE hint that suggested otherwise. Pick a "
                                "DIFFERENT action now: if a file is missing, CREATE it with "
                                "file_system(operation='write', …) or choose an existing "
                                "file from the listing; if an approach is wrong, change it."
                            )})

                    last_was_failure = True
                    # strikes.note_failure already reset the clean-success
                    # streak; this assignment is implicit in the ledger.

                    # Check for tool fallback suggestions
                    from ..tools.fallback_chains import get_fallback_hint
                    fallback_hint = ""
                    if fname:
                        hint = get_fallback_hint(fname, last_error_res or last_error_preview)
                        if hint:
                            fallback_hint = f"\n\n{hint}"

                    from ..tools.file_system import tool_list_files, project_scoped_sandbox
                    sandbox_state = await tool_list_files(project_scoped_sandbox(self.context)[0], self.context.memory_system)
                    # Re-anchor on the LIVE request. This diagnostic floods
                    # several KB of user-role text (failure context + full
                    # sandbox listing) into a long turn, and the model has
                    # been observed re-anchoring on STALE framing right
                    # after it (req 43, 2026-07-17: a manage_services miss
                    # → this flood → three turns re-running the PREVIOUS
                    # request's "resume the project" flow before
                    # recovering). Restating the current request pins it.
                    _anchor = (
                        "\n\nREMINDER — the CURRENT user request you are "
                        f"working on: \"{str(last_user_content)[:300]}\". "
                        "Continue that task from where it stands; do NOT "
                        "restart earlier requests' flows."
                    ) if last_user_content else ""
                    messages.append({"role": "user", "content": f"AUTO-DIAGNOSTIC: {multi_op_summary}{diagnostic_msg}{fallback_hint}\n\n{sandbox_state}{_anchor}"})

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
                            return False  # was `continue` — the region is the loop-body tail

                    if execution_failure_count >= 6 or total_fail >= 8:
                        pretty_log("Loop Breaker", "Forcing final response", icon=Icons.STOP)
                        messages.append({"role": "user", "content": "SYSTEM ALERT: You have failed too many times. The task cannot be completed. Provide a final response explaining the situation."})
                        force_final_response = True
                else:
                    # Only reset transient failures on success; structural
                    # failures require consecutive successes to decay.
                    transient_failure_count = 0
                    # Record a clean success. The ledger freezes decay once
                    # a SAME-failure loop is detected (otherwise an
                    # interleaved success — e.g. the auto sandbox-listing
                    # after a failed read — cancels the strike and the cap
                    # never fires). The freeze is NOT permanent: 3
                    # consecutive clean successes mean a genuine pivot, so
                    # note_clean_success unfreezes it. Signature counts are
                    # kept, so the same failure re-freezes on recurrence.
                    if strikes.note_clean_success():
                        pretty_log(
                            "Loop Breaker",
                            "Strike decay unfrozen after 3 consecutive clean successes.",
                            icon=Icons.OK,
                        )
                    if execution_failure_count > 0 and not strikes.decay_frozen:
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
                        return True  # was `break` — exit the turn loop
            return False  # loop-body tail: fall through to the next turn
        finally:
            ts._constraint_steer_pending = _constraint_steer_pending
            ts._proj_task_closed_this_req = _proj_task_closed_this_req
            ts._request_sys3_fired_once = _request_sys3_fired_once
            ts._request_sys3_prev_justification = _request_sys3_prev_justification
            ts.consecutive_parse_errors = consecutive_parse_errors
            ts.current_plan_json = current_plan_json
            ts.execution_failure_count = execution_failure_count
            ts.final_ai_content = final_ai_content
            ts.fname = fname
            ts.force_final_response = force_final_response
            ts.force_stop = force_stop
            ts.forget_was_called = forget_was_called
            ts.last_was_failure = last_was_failure
            ts.preflight_blocks_this_request = preflight_blocks_this_request
            ts.request_sandbox_state = request_sandbox_state
            ts.transient_failure_count = transient_failure_count

    def _note_defect_on_done_project(self, lc: str) -> bool:
        """Record a bug report against a DONE project as a defect task.

        Called from the repro-first nudge path (the request already
        classified as a bug report). Before 2026-07-18 such a report
        changed nothing in the store — the briefing kept saying "DONE,
        no open tasks" through an entire evening of fix attempts, and
        the open defect (game canvas renders black) lived nowhere.
        Adding the task both reopens the project (add_task's
        DONE→ACTIVE semantic, 2026-07-11) and puts the pending work in
        OPEN TASKS on every subsequent turn. Deduped against existing
        open defect tasks so a repeated "still broken" message doesn't
        stack duplicates. Returns True iff a task was recorded.
        Never raises."""
        try:
            pid = getattr(self.context, "current_project_id", None)
            store = getattr(self.context, "project_store", None)
            if not pid or store is None:
                return False
            proj = store.get_project(pid)
            if not proj or str(proj.get("status", "")).upper() != "DONE":
                return False
            open_defects = [
                t for t in store.list_tasks(pid)
                if str(t.get("status", "")).upper() in (
                    "PENDING", "READY", "IN_PROGRESS", "PAUSED", "NEEDS_USER")
                and str(t.get("description", "")).startswith("FIX (defect):")
            ]
            if open_defects:
                return False
            # Churn brake: at most _DEFECT_REOPEN_CAP reopens per rolling
            # window per project. The prune + count + append runs in ONE
            # atomic metadata update (cross-process safe). Past the cap,
            # surface the report loudly instead of reopening — the
            # operator decides whether the project really needs another
            # round, not the grinder.
            _now_ts = time.time()
            _reopen_ok = {"ok": False}

            def _reopen_gate(meta):
                raw = meta.get("defect_reopens") or []
                recent = [float(t) for t in raw
                          if _now_ts - float(t) < _DEFECT_REOPEN_WINDOW_S]
                if len(recent) < _DEFECT_REOPEN_CAP:
                    recent.append(_now_ts)
                    _reopen_ok["ok"] = True
                meta["defect_reopens"] = recent
                return meta

            store._atomic_metadata_update(pid, _reopen_gate)
            if not _reopen_ok["ok"]:
                pretty_log(
                    "Project Scope",
                    f"defect report on DONE project '{pid}' NOT reopened — "
                    f"reopen cap hit ({_DEFECT_REOPEN_CAP} per "
                    f"{int(_DEFECT_REOPEN_WINDOW_S / 3600)}h). The project "
                    "has churned through report→reopen→advance cycles; "
                    "review it manually or raise the cap.",
                    icon=Icons.WARN, level="WARNING",
                )
                return False
            desc = "FIX (defect): " + " ".join((lc or "").split())[:180]
            store.add_task(pid, desc)
            pretty_log(
                "Project Scope",
                f"Defect report on DONE project '{pid}' — reopened with a "
                "defect task so the pending work is on the books",
                icon=Icons.BRAIN_PLAN,
            )
            return True
        except Exception as exc:
            logger.debug(f"defect-task recording skipped: {exc}")
            return False

    async def _finalize_and_return(self, fs: "FinalizeState"):
        """The post-turn-loop finalization chain — output scrubbers,
        deferred Perfect-It, the final verifier gate + calibration
        write, competence/skill credit, workspace/selfhood episode
        recording, correction stash, and the response return.
        Extracted VERBATIM from handle_chat (#5 step 3) — zero
        control-flow rewrites: the region ends in handle_chat's own
        `return final_ai_content, created_time, req_id`, which is now
        simply this method's return; the call site returns it. Runs
        inside the agent semaphore (the call site sits in the same
        `async with`). Raising here propagates to the same outer
        try/finally it always reached — there are no mutated locals
        to repack (nothing executes after the region).
        """
        body = fs.body
        created_time = fs.created_time
        current_trajectory_id = fs.current_trajectory_id
        execution_failure_count = fs.execution_failure_count
        final_ai_content = fs.final_ai_content
        force_stop = fs.force_stop
        forget_was_called = fs.forget_was_called
        last_user_content = fs.last_user_content
        last_was_failure = fs.last_was_failure
        lc = fs.lc
        messages = fs.messages
        model = fs.model
        payload = fs.payload
        req_id = fs.req_id
        thought_content = fs.thought_content
        tools_run_this_turn = fs.tools_run_this_turn
        was_complex_task = fs.was_complex_task
        _stable_conv_fp = fs._stable_conv_fp
        _verdict_is_fresh = fs._verdict_is_fresh
        _verifier_verdict_cache = fs._verifier_verdict_cache
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

        # Collapse consecutive duplicate paragraphs. When the
        # model emits its final answer as a preamble alongside
        # a tool_call AND then again after the tool returns, the
        # accumulator stores the same paragraph twice in a row
        # ("Your name is X.\n\nYour name is X."). Users see a
        # stutter. Cheap fix: split on blank-line boundaries and
        # drop a paragraph that exactly matches its predecessor
        # after whitespace normalisation. Non-adjacent repeats
        # are left alone — they're usually intentional (e.g.
        # quoting the user's question and then answering it).
        if "\n\n" in final_ai_content:
            parts = final_ai_content.split("\n\n")
            deduped: list[str] = []
            prev_key = None
            for p in parts:
                key = re.sub(r"\s+", " ", p).strip()
                if key and key == prev_key:
                    continue
                deduped.append(p)
                prev_key = key
            final_ai_content = "\n\n".join(deduped)

        # Multi-turn reply smoothing (2026-07-17, operator-picked option):
        # a turn that ran several tool batches accumulates each
        # iteration's working narration ("Let me fix both:", "Now add the
        # resize logic:") and often a pre-verification summary that the
        # post-verification one restates. Drop those two shapes from the
        # DELIVERED reply — the live stream already showed them as
        # progress. Gated on ≥2 real tool runs so conversational and
        # single-tool answers are never touched; runs BEFORE the verifier
        # gate so the verdict judges the text the user actually receives.
        if sum(1 for t in tools_run_this_turn
               if t and not t.get("_synthetic")) >= 2:
            try:
                from .reply_smoothing import smooth_reply
                _smoothed = smooth_reply(final_ai_content)
                if _smoothed != final_ai_content:
                    pretty_log(
                        "Reply Smoothing",
                        f"trimmed working narration: "
                        f"{len(final_ai_content)} → {len(_smoothed)} chars",
                        icon=Icons.BRAIN_SUM,
                    )
                    final_ai_content = _smoothed
            except Exception as _sm_exc:
                logger.debug("reply smoothing skipped: %s", _sm_exc)

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
            # Whether the optimization is part of the user-facing
            # response. When the flag is OFF the generation is pure
            # internal learning — so it runs as a tracked background
            # task instead of blocking the reply (observed: a 24s
            # task delivered at +271s because the response waited on
            # this call, which also let the watchdog cross its 120s
            # idle threshold and pile hippocampus consolidation onto
            # the same single-slot upstream).
            _pp_show_to_user = getattr(self.context.args, 'perfect_it', False) is True
            pretty_log(
                "Perfect It Protocol",
                "Generating an optimization suggestion to append to the reply..."
                if _pp_show_to_user else
                "Generating an optimization suggestion in the background "
                "(internal learning — not shown to the user)...",
                icon=Icons.IDEA,
            )
            # Heartbeat: end-of-turn post-processing is outside the
            # turn loop's heartbeats; without this, a long inline
            # generation makes the biological watchdog think the
            # system is idle MID-REQUEST and wake the hippocampus.
            self.context.last_activity_time = datetime.datetime.now()
            perfect_it_prompt = f"Task completed successfully. Final tool output:\n\n{tools_run_this_turn[-1]['content']}\n\n<system_directive>First, succinctly present the tool output/result to the user. Then, based on your Perfection Protocol, analyze the result and proactively suggest one concrete way to optimize, scale, secure, or automate this work further. RESPOND IN PLAIN TEXT ONLY. DO NOT USE TOOLS.</system_directive>"

            p_req_messages = []
            # Snapshot — deliberately NOT messages.append(): the
            # synthetic directive must not leak into the trajectory
            # record or any later consumer of `messages`.
            for m in messages + [{"role": "user", "content": perfect_it_prompt}]:
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

            # Build a payload COPY: the original is left untouched
            # for anything downstream, and the deferred task must
            # not share mutable state with the live request.
            # 🔴 Physically remove tools so it cannot hallucinate a tool call
            p_payload = {k: v for k, v in payload.items() if k not in ("tools", "tool_choice")}
            p_payload["messages"] = p_req_messages
            p_payload["stream"] = False  # Prevent SSE streaming leak from the main loop
            _pp_lesson_label = f"Optimization Analysis: {last_user_content[:50]}..."

            if _pp_show_to_user:
                # --perfect-it: the optimization IS part of the
                # reply, so it must block the response path —
                # foreground=True, or the call self-stalls waiting
                # for its own request to clear.
                try:
                    p_msg = await self._perfect_it_generate_and_learn(
                        p_payload, _pp_lesson_label, current_trajectory_id,
                        foreground=True,
                    )
                    if p_msg:
                        if final_ai_content:
                            final_ai_content += "\n\n" + p_msg
                        else:
                            final_ai_content = p_msg
                except Exception:
                    # Only report failure to user if they expected to see it
                    if not final_ai_content:
                        final_ai_content = "Task finished successfully, but optimization generation failed."
            else:
                # Internal-learning only: fire-and-forget, tracked in
                # the context-level set (strong ref — bare tasks can
                # be GC'd — and drained by the lifespan shutdown).
                # The background queue in llm.py yields to any
                # foreground call, so the verifier below still gets
                # upstream priority.
                async def _deferred_perfect_it(
                    _payload=p_payload,
                    _label=_pp_lesson_label,
                    _tid=current_trajectory_id,
                ):
                    try:
                        await self._perfect_it_generate_and_learn(_payload, _label, _tid)
                    except Exception as _e:
                        logger.debug(
                            "Deferred Perfect-It skipped: %s: %s",
                            type(_e).__name__, _e,
                        )
                _bg = getattr(self.context, "_pending_background_tasks", None)
                if _bg is None:
                    _bg = set()
                    self.context._pending_background_tasks = _bg
                _pp_task = asyncio.create_task(_deferred_perfect_it())
                _bg.add(_pp_task)
                _pp_task.add_done_callback(_bg.discard)

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
            # Heartbeat: the verifier completion can run for a minute
            # on a cold prefill; without refreshing the activity
            # clock here the biological watchdog (idle > 120s) wakes
            # the hippocampus MID-REQUEST and its consolidation LLM
            # calls compete with this one on the same upstream.
            self.context.last_activity_time = datetime.datetime.now()
            verifier = getattr(self.context, "verifier", None)
            # Gate: any tool-using turn is worth verifying. The old
            # `was_complex_task` constraint (turn > 2) silently
            # skipped the common 1–2 turn tool path, so the verifier
            # almost never fired. `tools_run_this_turn` alone is
            # sufficient — trivial greetings already have no tools.
            # Reuse the verdict the in-loop AUTO-REPAIR finalisation
            # already computed for THIS final answer (one verifier
            # pass per clean success — same cost as before the gate
            # was split). Recompute only when the loop exited without
            # a fresh in-loop verdict (error / abort / terminal path,
            # or repair budget spent on a non-5851 exit).
            if _verdict_is_fresh and _verifier_verdict_cache is not None:
                v_result, last_tool = _verifier_verdict_cache
            else:
                # Non-blocking gate: with a dedicated --critic-nodes
                # pool the (slower) verdict runs in the background and
                # the response ships without waiting on it; without
                # one this awaits inline exactly as before. `messages`
                # is snapshotted because the background verdict task
                # iterates it while the turn keeps mutating the live
                # list.
                v_result, last_tool = await self._compute_verifier_verdict_gated(
                    tools_run_this_turn=tools_run_this_turn,
                    messages=list(messages),
                    final_ai_content=final_ai_content,
                    last_user_content=last_user_content,
                    lc=lc,
                    trajectory_id=current_trajectory_id,
                    conv_fp=_stable_conv_fp,
                )
            from .verifier import VerifyVerdict
            # Consumption guard mirrors the original gate exactly: only
            # annotate / backfill / retract when the verifier was
            # actually applicable (installed, a substantive tool ran,
            # not strict trivial chat). `_compute_verifier_verdict`
            # returns `last_tool` even when it produces no verdict, so
            # the unverified-mutation branch below still fires.
            if (
                verifier is not None
                and verifier.llm_client is not None
                and last_tool is not None
                and final_ai_content
                and not self._is_strict_trivial_chat(lc)
            ):
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
                            # to_thread: retraction takes a file lock,
                            # rewrites the playbook and does a sync
                            # Chroma delete — run it off the event
                            # loop like every other SkillMemory call
                            # on this path.
                            await asyncio.to_thread(
                                _sm.retract_lessons_from_trajectory,
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
                    if v_result:
                        pretty_log(
                            "Verifier",
                            f"{v_result.verdict.value} ({v_result.confidence:.0%})",
                            icon=Icons.VERIFIER_LAB,
                        )
                    else:
                        # v_result is None → the verifier pipeline FAILED
                        # (e.g. LLM error); it did NOT pass. Don't show ✅.
                        #
                        # Unverified-mutation guard: when the gate is
                        # skipped AND the turn's final substantive action
                        # was a successful file write/replace, the agent
                        # is finalising on an UNTESTED change — the exact
                        # req_C0 failure (33-min build that "finished"
                        # right after a write that was never run, yet
                        # reported C=0.96). Treat that as a failed outcome
                        # so confidence drops below threshold and the turn
                        # is recorded as unverified rather than success.
                        if _is_unverified_mutation(last_tool):
                            verifier_backfill = (
                                "failed",
                                "unverified mutation — finalised on an "
                                "untested file write/replace; the change "
                                "was never run or screenshotted",
                            )
                            note = (
                                "\n\n---\n**⚠ Unverified:** the final action "
                                "was a file write that was never executed or "
                                "rendered, so I cannot confirm it works. Treat "
                                "this as INCOMPLETE — run/preview it before "
                                "relying on it."
                            )
                            if note[:40] not in final_ai_content:
                                final_ai_content = f"{final_ai_content}{note}"
                            pretty_log(
                                "Verifier",
                                "finalised on an UNVERIFIED file write (never "
                                "run/rendered) — flagged INCOMPLETE "
                                "(outcome=failed)",
                                icon=Icons.WARN, level="WARNING",
                            )
                        elif getattr(getattr(self.context, "args", None),
                                     "no_verifier", False) is True:
                            # Ablated, not deferred. This branch used to
                            # fall into the async "deferred" message
                            # below, which claimed a verdict was running
                            # that was never spawned — the live log
                            # showed days of "verdict deferred" while
                            # --no-verifier (an ablation leftover in the
                            # launcher) had the whole subsystem off, and
                            # nothing ever contradicted it.
                            pretty_log(
                                "Verifier",
                                "no verdict — verifier is ABLATED "
                                "(--no-verifier); nothing will land late",
                                icon=Icons.WARN, level="WARNING",
                            )
                        elif self._critic_async_enabled():
                            # Async mode: the verdict was deliberately
                            # deferred, not missing — it's running on the
                            # critic node now and will land as LATE …
                            pretty_log(
                                "Verifier",
                                "verdict deferred — verifying asynchronously "
                                "after the reply (off the critical path)",
                                icon=Icons.VERIFIER_LAB,
                            )
                        else:
                            pretty_log(
                                "Verifier",
                                "no verdict produced — gate skipped",
                                icon=Icons.WARN, level="WARNING",
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
                if getattr(self.context, 'journal', None) and self.context.args.smart_memory > 0.0 and not forget_was_called:

                    await self._journal_append_safe('post_mortem', {'user': last_user_content, 'tools': list(tools_run_this_turn), 'ai': final_ai_content, 'model': model})
                    await self._record_episode_safe(last_user_content, list(tools_run_this_turn), final_ai_content)
        # Post-turn hydration usefulness judge (fire-and-forget, worker-
        # hosted): scores which injected memories the reply actually used.
        self._judge_hydration_safe(final_ai_content, turn_id=str(req_id or ""))

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

        # Calibration spine (roadmap phase 2.5): pair THIS turn's
        # last composite-confidence reading with the realized
        # outcome (clean turn → 1.0; any structural tool failure
        # → 0.0). This is the loop that finally makes "the agent
        # is 80% confident" mean the turn succeeds ~80% of the
        # time — the JSONL it appends feeds the idle Brier/ECE
        # refit (phase 2.7c). Runs on BOTH clean and failed turns
        # (outside the credit gate above) so both outcome classes
        # are represented. `_calib_pending` is set only when a
        # reading was computed this turn, so we never record a
        # stale cross-turn pair.
        try:
            _ct = getattr(self.context, "calibration_tracker", None)
            _pending = getattr(self.context, "_calib_pending", None)
            # Finalization fallback (the load-bearing path in practice):
            # most turns never enter the streaming
            # is_final_generation+stream_response branch where the
            # entropy/confidence path lives, so _pending is usually None
            # here. Compute the reading NOW — logprob-optional: neutral
            # entropy term, competence + verbalised uncertainty drive it
            # — so calibration records on EVERY full-loop metacog turn
            # (and the arbiter bundle gets a reading via record_confidence).
            if _pending is None and _ct is not None:
                try:
                    _mc = getattr(self.context, "metacog", None)
                    if (_mc is not None and getattr(_mc, "enabled", False)
                            and getattr(_mc, "confidence", None) is not None
                            and getattr(_mc, "competence", None) is not None):
                        _last_tool = ""
                        for _t in reversed(tools_run_this_turn or []):
                            if isinstance(_t, dict) and _t.get("name"):
                                _last_tool = _t["name"]
                                break
                        from .metacog import _domain_for_tool
                        _dom = _domain_for_tool(_last_tool or "")
                        _p = _mc.competence.estimate(_dom, _last_tool or None)
                        _n = _mc.competence.observations(_dom, _last_tool or None)
                        _upress = 0.0
                        try:
                            _ut = getattr(self.context, "uncertainty_tracker", None)
                            if _ut is not None:
                                _upress = _ut.pressure()
                        except Exception:
                            _upress = 0.0
                        # Objective outcome penalty: a REFUTED verdict or
                        # an unverified mutation (both surface as
                        # verifier_backfill[0]=="failed") is ground truth
                        # that THIS answer is wrong/unconfirmed. Without
                        # it the reading is ≈ competence, so a historically
                        # strong domain reports high confidence on a build
                        # the verifier just rejected (the req_44/C0
                        # "below=no on broken work" failure). 0.8 reliably
                        # pulls a 0.92–0.96 competence reading below the
                        # 0.89 threshold.
                        _outcome_penalty = (
                            0.8 if (verifier_backfill
                                    and verifier_backfill[0] == "failed")
                            else 0.0
                        )
                        _pending = _mc.confidence.score(
                            normalised_entropy=0.5,
                            competence_p_success=_p,
                            n_observations=_n,
                            uncertainty_pressure=_upress,
                            outcome_penalty=_outcome_penalty,
                        )
                        self.context.last_confidence = _pending
                        try:
                            _mc.record_confidence(_pending)
                            _mc.count(confidence_total=True,
                                      confidence_below=_pending.below_threshold)
                        except Exception:
                            pass
                        from .metacog_log import (
                            emit as _mc_emit, Subsystem as _mc_ss,
                            LEVEL_INFO, LEVEL_DEBUG,
                        )
                        _mc_emit(
                            _mc_ss.CONF,
                            level=(LEVEL_INFO if _pending.below_threshold else LEVEL_DEBUG),
                            below=_pending.below_threshold, C=_pending.composite,
                            entropy=_pending.entropy_component,
                            competence=_pending.competence_component,
                            n=_n, domain=_dom, tool=_last_tool or None,
                            src="finalize", threshold=_pending.threshold,
                        )
                except Exception as _cfx:
                    logger.debug("finalize confidence compute failed: %s", _cfx)
            if _ct is not None and _pending is not None:
                # Outcome from the signals available THIS turn: a
                # structural tool failure OR a verifier REFUTED verdict
                # (≥0.7 → verifier_backfill[0]=="failed") is a negative.
                # Without a real negative source, free-form chat turns
                # are almost all "clean" → single-class → the fit bails;
                # the verifier verdict is what gives calibration its
                # "confidently wrong" examples.
                _verifier_failed = bool(
                    verifier_backfill and verifier_backfill[0] == "failed"
                )
                _calib_outcome = (
                    0.0 if (execution_failure_count > 0 or _verifier_failed)
                    else 1.0
                )
                await asyncio.to_thread(
                    _ct.record,
                    composite=_pending.composite,
                    entropy_component=_pending.entropy_component,
                    competence_component=_pending.competence_component,
                    uncertainty_pressure=getattr(_pending, "uncertainty_pressure", 0.0),
                    outcome=_calib_outcome,
                )
                # Stash the components keyed by this response's
                # fingerprint so a NEXT-turn user-correction can record
                # a (C, 0.0) negative for this turn — the strongest
                # "confidently wrong" calibration signal (the user is
                # the cheapest supervisor for free-form chat).
                try:
                    from collections import OrderedDict as _OD
                    _cc = getattr(self.context, "_recent_calib_for_correction", None)
                    if _cc is None:
                        _cc = _OD()
                        self.context._recent_calib_for_correction = _cc
                    _cc[self._response_fingerprint(final_ai_content or "")] = {
                        "composite": _pending.composite,
                        "entropy_component": _pending.entropy_component,
                        "competence_component": _pending.competence_component,
                        "uncertainty_pressure": getattr(_pending, "uncertainty_pressure", 0.0),
                    }
                    while len(_cc) > 32:
                        _cc.popitem(last=False)
                except Exception:
                    pass
                self.context._calib_pending = None
        except Exception as _calx:
            logger.debug("calibration record failed: %s", _calx)

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
                # Resolve unknowns the agent answered for ITSELF this
                # turn: when an info-gathering tool ran successfully and
                # an unknown's resolution pointed at that path, mark it
                # resolved so it drops out of the clarifying-question gate
                # and the risk footer below. (The resolve-side of the
                # uncertainty lifecycle was previously never called, so
                # self-answered unknowns kept re-surfacing to the user.)
                try:
                    _info_tools = {"web_search", "deep_research", "recall",
                                   "fact_check", "browser", "file_system", "knowledge_base"}
                    _ran_info = any(
                        isinstance(t, dict) and t.get("name") in _info_tools
                        and not str(t.get("content", "")).lstrip().startswith(("Error", "ERROR", "SYSTEM ERROR"))
                        for t in (tools_run_this_turn or [])
                    )
                    if _ran_info:
                        for _u in list(tracker.unknowns):
                            if getattr(_u, "resolved", False):
                                continue
                            _resn = (getattr(_u, "resolution", "") or "").lower()
                            if any(k in _resn for k in ("search", "web", "read", "file", "look", "fetch", "recall", "research")):
                                tracker.resolve_unknown(_u, "resolved via tool output this turn")
                except Exception:
                    pass
                # Verify-side of the lifecycle (previously unwired): if the
                # turn completed cleanly (a response, no error/failure
                # markers), treat the assumptions the agent proceeded on as
                # borne out (was_correct=True). Conservative — only confirms
                # on a clean turn, never marks them wrong.
                try:
                    _turn_clean = bool(final_ai_content) and "error" not in final_ai_content[:80].lower()
                    if _turn_clean:
                        for _a in list(tracker.assumptions):
                            if not getattr(_a, "verified", False):
                                tracker.verify_assumption(_a, True)
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

        # Value-alignment gate (opt-in --principle-gate). If the
        # agent has authored operating principles (selfhood/values),
        # an independent LLM check flags a response that contradicts
        # one and appends a brief self-note — turning the principles
        # from prompt decoration into an actual behavioural check.
        # Default off (adds one LLM call to final turns); never
        # blocks, only annotates.
        try:
            if (getattr(getattr(self.context, "args", None), "principle_gate", False) is True
                    and final_ai_content):
                _sm = getattr(self.context, "self_model", None)
                if _sm is not None and getattr(_sm, "enabled", False) and _sm.principles():
                    async def _pg_critique(p):
                        _r = await self.context.llm_client.chat_completion({
                            "model": self.context.args.model,
                            "messages": [{"role": "user", "content": p}],
                            "temperature": 0.0, "max_tokens": 512, "stream": False,
                        })
                        return ((_r or {}).get("choices", [{}])[0]
                                .get("message", {}).get("content", "") or "")
                    _aligned, _note = await _sm.evaluate_response_alignment(
                        final_ai_content, critique_fn=_pg_critique,
                    )
                    if not _aligned:
                        final_ai_content = (
                            f"{final_ai_content}\n\n---\n"
                            f"**Self-check (principle):** {_note}"
                        )
                        pretty_log("Principle Gate",
                                   f"response flagged: {_note[:80]}",
                                   icon=Icons.SHIELD)
        except Exception as e:
            logger.debug(f"principle gate skipped: {e}")

        # Autonomous-progress digest — closes the user-facing half of
        # phase 2.95. If projects were advanced in the background (by
        # the idle autoadvance phase, the tool, or the HTTP route)
        # since the user last saw a digest, surface a concise "while
        # you were away" header on this turn, flagging the items that
        # now need their input. Watermark-gated on the monotonic
        # project-event id so each batch shows exactly once; the first
        # run baselines silently (no historical backlog dump).
        # Header-prepended so needs-user items lead the reply.
        try:
            from .autonomous_activity import is_internal_request as _is_internal_req
            _ps = getattr(self.context, "project_store", None)
            # Internal turns (cron jobs / delegated sub-agents firing
            # handle_chat with a "sched-"/"job-"/"sub-" req_id) must not
            # consume the digest watermark — that would silently eat the
            # operator's next "while you were away" report.
            if (_ps is not None and final_ai_content
                    and not _is_internal_req(fs.req_id)):
                from pathlib import Path as _Path
                from .project_digest import (
                    summarize_since, render_digest,
                    load_watermark, save_watermark,
                )
                _wm_path = (_Path(str(self.context.memory_dir)).parent
                            / "projects_digest.json")
                _wm = load_watermark(_wm_path)
                if _wm is None:
                    # First run: baseline to the current high-water mark.
                    _base = await asyncio.to_thread(summarize_since, _ps, 0)
                    save_watermark(_wm_path, _base.new_event_id)
                else:
                    _dg = await asyncio.to_thread(summarize_since, _ps, _wm)
                    if _dg.has_content:
                        _digest = render_digest(_dg)
                        if _digest and _digest[:40] not in final_ai_content:
                            final_ai_content = f"{_digest}\n\n---\n\n{final_ai_content}"
                            pretty_log(
                                "Autoadvance Digest",
                                f"{_dg.advanced} advanced, "
                                f"{len(_dg.needs_user)} need-user, "
                                f"{_dg.projects_touched} project(s)",
                                icon=Icons.BRAIN_PLAN,
                            )
                    if _dg.new_event_id > _wm:
                        save_watermark(_wm_path, _dg.new_event_id)
        except Exception as _dgx:
            logger.debug(f"autoadvance digest skipped: {_dgx}")

        # Background-activity digest — the ALL-PHASE companion of the
        # project digest above (2026-07-11). Idle-phase outcomes (dream /
        # reflection / post-mortem / skills graduation / PRM / router /
        # calibration / self-play) and scheduled-turn conclusions recorded
        # in core.autonomous_activity surface here, byte-offset
        # watermarked so each batch shows exactly once; the first run
        # baselines silently. Project-phase records are excluded from the
        # render (the project digest above already covers them). Same
        # internal-request gate as the project digest.
        try:
            from .autonomous_activity import (
                get_activity_log as _get_alog,
                is_internal_request as _is_internal_req2,
                render_activity_digest as _render_adg,
                load_offset as _act_load, save_offset as _act_save,
                SEVERITY_NOTIFY as _SEV_NOTIFY,
            )
            _alog = _get_alog(self.context)
            if (_alog is not None and final_ai_content
                    and not _is_internal_req2(fs.req_id)):
                from pathlib import Path as _Path
                _act_wm_path = (_Path(str(self.context.memory_dir)).parent
                                / "activity_digest.json")
                _act_wm = _act_load(_act_wm_path)
                if _act_wm is None:
                    # First run: baseline to EOF, show nothing.
                    _act_save(_act_wm_path, _alog.current_offset())
                else:
                    _recs, _new_off = _alog.read_since(_act_wm)
                    # current_req_id: don't echo records THIS turn wrote
                    # (e.g. its own notify_operator call) back at the
                    # operator as "while you were away".
                    # notify-only (operator decision 2026-07-17): routine
                    # maintenance (dream/PRM/router/calibration/self-play)
                    # is info-severity and reads as noise in chat — it
                    # stays in the ledger, reachable on demand via
                    # `introspect action='activity'`. The watermark still
                    # advances over info records below, so they are
                    # "seen" without ever rendering.
                    _adg = _render_adg(_recs,
                                       current_req_id=str(fs.req_id or ""),
                                       severities=(_SEV_NOTIFY,))
                    if _adg and _adg[:40] not in final_ai_content:
                        final_ai_content = (
                            f"{_adg}\n\n---\n\n{final_ai_content}")
                        pretty_log(
                            "Activity Digest",
                            f"{len(_recs)} background record(s) surfaced",
                            icon=Icons.ACTIVITY,
                        )
                    if _new_off > _act_wm:
                        _act_save(_act_wm_path, _new_off)
        except Exception as _adx:
            logger.debug(f"activity digest skipped: {_adx}")

        # Chat→project promotion suggestion (advisory; previously unwired).
        # When a free-chat session accumulates enough turns / sandbox
        # work, gently suggest promoting it to a tracked project — ONCE
        # (suppressed via a scratchpad flag so we don't nag). Only the
        # explicit manage_projects(promote_from_context) tool actually
        # creates a project; this is just a one-line footer offer.
        try:
            if getattr(self.context, "current_project_id", None) is None and final_ai_content:
                _sp = getattr(self.context, "scratchpad", None)
                _already = False
                try:
                    _already = bool(_sp and _sp.get("_promotion_suggested"))
                except Exception:
                    _already = False
                if not _already:
                    from .project_safety import should_suggest_promotion as _ssp
                    _uturns = [
                        m.get("content") for m in messages
                        if isinstance(m, dict) and m.get("role") == "user"
                        and isinstance(m.get("content"), str)
                    ]
                    _aturns = [
                        m.get("content") for m in messages
                        if isinstance(m, dict) and m.get("role") == "assistant"
                        and isinstance(m.get("content"), str)
                    ]
                    # Sandbox writes accumulate ACROSS the session (the AND
                    # rule needs ≥3 total, and a chat rarely writes 3 files in
                    # one turn). Persist a running counter in the scratchpad.
                    _writes_this_turn = sum(
                        1 for t in (tools_run_this_turn or [])
                        if isinstance(t, dict) and t.get("name") == "file_system"
                    )
                    _writes = _writes_this_turn
                    try:
                        if _sp is not None:
                            _prev = int(_sp.get("_session_sandbox_writes") or 0)
                            _writes = _prev + _writes_this_turn
                            if _writes_this_turn:
                                _sp.set("_session_sandbox_writes", str(_writes))
                    except Exception:
                        _writes = _writes_this_turn
                    # If the user is administering the project system
                    # itself this turn (list / delete / switch / ...),
                    # don't nudge them to create one — they obviously
                    # know about projects (reported: nudge fired right
                    # after a `delete project`).
                    _managing = any(
                        isinstance(t, dict) and t.get("name") == "manage_projects"
                        for t in (tools_run_this_turn or [])
                    )
                    _sugg = _ssp(
                        user_turns=_uturns, assistant_turns=_aturns,
                        sandbox_writes=_writes, plan_node_count=0,
                        already_in_project=False,
                        managing_projects=_managing,
                    )
                    if getattr(_sugg, "should_suggest", False):
                        final_ai_content = (
                            f"{final_ai_content}\n\n---\n"
                            f"💡 This looks like ongoing work ({_sugg.reason}). "
                            f"Want me to promote it to a tracked project "
                            f"(“{_sugg.suggested_title}”)? Just say the word."
                        )
                        try:
                            if _sp:
                                _sp.set("_promotion_suggested", "1")
                        except Exception:
                            pass
        except Exception as _pexc:
            logger.debug(f"promotion suggestion skipped: {_pexc}")

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

        # Turn→project write-back (2026-07-18). While a project was
        # bound, any request that did real work (mutated files or ran
        # work tools) leaves ONE bounded work_log event on the project —
        # request head, files touched, tool counts, outcome (verifier-
        # aware), and the head of the final answer. This is the record
        # the store previously never got for interactive turns: agent.py
        # wrote nothing itself, so all post-completion debugging was
        # invisible to future turns (2026-07-17 session: 7 requests of
        # game debugging, zero store events after 21:41). Non-fatal.
        try:
            _wl_pid = getattr(self.context, "current_project_id", None)
            _wl_store = getattr(self.context, "project_store", None)
            _wl_files = getattr(self.context, "_project_work_files", None) or set()
            _wl_tools = getattr(self.context, "_project_work_tools", None) or {}
            if _wl_pid and _wl_store is not None and (_wl_files or _wl_tools):
                if verifier_backfill is not None:
                    _wl_outcome = f"verifier:{verifier_backfill[0]}"
                elif execution_failure_count > 0:
                    _wl_outcome = "had_failures"
                else:
                    _wl_outcome = "completed"
                # Harness-dimension attribution (2026-07-19): classify the
                # turn's captured failure heads so the work_log names the
                # layer that failed (a prior for debugging and the group
                # key for dream-side distillation — not a verdict).
                _wl_dim = ""
                if _wl_outcome != "completed":
                    try:
                        from .failure_dimension import (
                            classify_failure_dimension, failure_dim_enabled)
                        if failure_dim_enabled():
                            _parts = list(getattr(
                                self.context, "_turn_failure_texts",
                                None) or [])[-3:]
                            if (verifier_backfill is not None
                                    and verifier_backfill[0] == "failed"):
                                _parts.insert(
                                    0, f"verifier: {verifier_backfill[1]}")
                            if _parts:
                                _wl_dim, _wl_dim_sig = \
                                    classify_failure_dimension("\n".join(_parts))
                                if _wl_dim == "unknown":
                                    _wl_dim = ""
                                else:
                                    logger.debug(
                                        "work_log dimension=%s (signal: %r)",
                                        _wl_dim, str(_wl_dim_sig)[:60])
                    except Exception:
                        _wl_dim = ""
                await asyncio.to_thread(
                    _wl_store.add_work_log, _wl_pid,
                    request=last_user_content or "",
                    files=list(_wl_files),
                    tools=dict(_wl_tools),
                    commands=list(getattr(self.context,
                                          "_project_work_cmds", None) or []),
                    outcome=_wl_outcome,
                    note=final_ai_content or "",
                    failure_dimension=_wl_dim,
                )
                # Consumed — a queued follow-up in the same process must
                # not re-attribute this request's work.
                try:
                    self.context._project_work_files = set()
                    self.context._project_work_tools = {}
                    self.context._turn_failure_texts = []
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"project work-log write skipped: {type(e).__name__}: {e}")

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
                user_request=last_user_content,
                # Consolidate the corpus outcome with the same signals
                # calibration + selfhood use (was heuristics-only here).
                verifier=(verifier_backfill[0] if verifier_backfill else None),
                execution_failed=(execution_failure_count > 0),
            )
        except Exception as e:
            # Debug-level: a turn-logging failure must never be
            # noisy in production. When diagnosing, bump to
            # warning temporarily.
            logger.debug(f"trajectory logging skipped: {type(e).__name__}: {e}")

        # Selfhood reference-count (rest of H12): bump the reference
        # counter for the prior experiences the agent actually echoed
        # this turn (wake-up prefix it was shown vs. the response it
        # produced). Previously note_referenced_experiences had no
        # caller, so the "which memories did I reach for" signal was
        # always empty. Non-fatal.
        try:
            _sm = getattr(self.context, "self_model", None)
            if _sm is not None and getattr(_sm, "enabled", False):
                _wp = locals().get("wakeup_prefix", "") or ""
                if _wp and final_ai_content:
                    await asyncio.to_thread(
                        _sm.note_referenced_experiences,
                        prefix_text=_wp, response_text=final_ai_content,
                    )
        except Exception as _refexc:
            logger.debug(f"selfhood reference-count skipped: {_refexc}")

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

        # Deterministically prepend any deferred async-verdict
        # correction staged at turn start (GHOST_CRITIC_ASYNC).
        final_ai_content = self._take_active_correction() + (final_ai_content or "")
        return final_ai_content, created_time, req_id

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

        # Register this turn BEFORE acquiring the semaphore (2026-07-11):
        # turns are globally serialized, so a request QUEUED behind a runaway
        # turn must be cancellable too — otherwise the only way out of a
        # wedged turn is a restart. See core/turns.py.
        from .turns import get_turn_registry, TurnCancelled
        _turn_reg = get_turn_registry(self)
        _active_turn = _turn_reg.register(
            req_id,
            preview=str((body.get("messages") or [{}])[-1].get("content") or "")
            if isinstance(body.get("messages"), list) else "",
            session_id=str(body.get("session_id") or ""),
        )
        # register() may uniquify the key on a client-req-id collision — adopt
        # the effective id so mark_running / is_cancelled / unregister all
        # address THIS turn's entry, not the colliding one.
        req_id = _active_turn.req_id

        # A STREAMED turn hands the caller a generator and returns; the actual
        # upstream final-generation streams while the caller drains it, AFTER
        # this method's finally has already run. If the finally unregistered
        # the turn there, the streaming tail was invisible to /api/turns and
        # uncancellable (2026-07-15). When the streaming path takes ownership
        # it sets this flag so the finally DEFERS the unregister to the stream
        # wrapper's own finally (which runs when the drain completes). The
        # semaphore is NOT similarly deferred on purpose — see the wrapper.
        _stream_owns_unregister = False

        try:
            async with self.agent_semaphore:
                _turn_reg.mark_running(req_id)
                # A cancel that landed while we were queued: stop before doing
                # any work (the flag is set; the task may not have been killed
                # if it was already at the head of the queue).
                if _turn_reg.is_cancelled(req_id):
                    raise TurnCancelled(req_id, _active_turn.reason)
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

                # Stable per-conversation tag, computed ONCE from the client's
                # full (un-pruned) history. Both the async-correction consume
                # below and the late-verdict record deep in the loop use THIS
                # value, so they agree on the same opener. The record side used
                # to recompute the fingerprint from the token-pruned loop
                # `messages`, which diverged from the un-pruned consume side in
                # long sessions → queued corrections never matched and were
                # silently dropped.
                _stable_conv_fp = self._conversation_fingerprint(messages)

                # Async-critic deferred correction: if a prior turn's
                # post-response verdict refuted that answer, surface the
                # correction at the top of THIS turn (no-op otherwise).
                messages = self._consume_pending_corrections(messages, conv_fp=_stable_conv_fp)

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
                # Extract the request's explicit constraints ONCE, up front.
                # `_request_constraint_block` is re-rendered into the dynamic
                # state every turn of this request, and the first successful
                # file write triggers a one-shot constraint-check steer
                # (`_constraint_steer_pending`). Deterministic (no LLM);
                # empty for the vast majority of messages.
                _request_constraints = extract_constraints(last_user_content)
                _request_constraint_block = render_constraint_block(
                    _request_constraints,
                    header="EXPLICIT USER CONSTRAINTS (CURRENT REQUEST)",
                )
                _constraint_steer_pending = bool(_request_constraints)
                # Discard any confidence reading left over from a PREVIOUS
                # streaming turn: its stream wrapper sets `_calib_pending`
                # while the client drains the SSE stream, i.e. AFTER
                # handle_chat returned, so the turn-end pairing below never
                # consumed it. Without this reset the next turn would pair
                # that stale reading with ITS OWN outcome, systematically
                # mispairing the calibration JSONL.
                self.context._calib_pending = None

                # CONVERSATION-SCOPED PROJECT BINDING. `current_project_id`
                # is process-global; reconcile it against THIS request's
                # conversation before anything touches the sandbox (the
                # ambient tree listing, file_system, execute). Without
                # this, a project activated in one chat captured every
                # other conversation's file writes into its workspace
                # (observed: migration.sql written into an unrelated
                # project, then four turns burned explaining/undoing it).
                try:
                    from ..tools.projects import (
                        conversation_fingerprint, reconcile_conversation,
                    )
                    reconcile_conversation(
                        self.context, conversation_fingerprint(messages)
                    )
                except Exception as e:
                    logger.debug(f"project conversation reconcile skipped: {e}")
                # Work-log accumulators for THIS request (read by the
                # finalize chain's project write-back). Reset AFTER the
                # reconcile so they always describe work done under the
                # project binding that survived it. Process-global is safe:
                # turns are serialized by the agent semaphore.
                self.context._project_work_files = set()
                self.context._project_work_tools = {}
                self.context._project_work_cmds = []
                # Failure-text capture for harness-dimension attribution
                # (core/failure_dimension.py). Filled in the dispatch loop
                # next to classify_tool_failure; classified and consumed by
                # the finalize chain's work-log write-back. Not project-
                # gated: future lesson producers read it too.
                self.context._turn_failure_texts = []
                self.context._offproject_steer_done = False
                # Edit-run futility tracking: basename -> {writes, runs} for
                # code files this request. See the futility breaker in the
                # dispatch pipeline.
                self.context._script_iter = {}
                self.context._futility_steer_done = False
                # Context-pressure lockdown: set after the SECOND overflow in
                # one request (read budget drops to zero for its remainder).
                self.context._ctx_pressure_lockdown = False
                # Snapshot the project that was active when THIS user message
                # arrived — the delete-eligibility gate in tools.projects
                # only honours a bare "delete it" against this project. A
                # project the agent creates mid-request was never seen by
                # the user and therefore can't be what "it" refers to
                # (observed live: one "delete it and make something else"
                # cascaded into six hard deletes, five of them of projects
                # created seconds earlier in the same request).
                self.context.request_start_project_id = getattr(
                    self.context, "current_project_id", None)

                # Replay the bound project's stored constraints into THIS
                # request's constraint set (arming the post-write steer and
                # the dynamic-state block). Placed AFTER the conversation⇄
                # project reconciliation above so the binding is
                # trustworthy. See _merge_project_constraints for why the
                # message-only extraction at request start is not enough.
                (_request_constraints, _request_constraint_block,
                 _constraint_steer_pending) = self._merge_project_constraints(
                    _request_constraints, last_user_content)

                # Repro-first nudge for bug reports. Injected BEFORE the
                # first LLM call so the very first action is an
                # observation, not a theory. Suppressed in simulations
                # (same flag as the meta-task nudge).
                if (_is_bug_report_intent(lc)
                        and not getattr(self, "suppress_meta_task_nudges", False)):
                    pretty_log(
                        "Repro-First Nudge",
                        "Bug-report intent detected — steering first action "
                        "to reproduce/observe",
                        icon=Icons.SHIELD,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "SYSTEM HINT (repro-first): The message above reports "
                            "that something already built is not working. Do NOT "
                            "start by re-reading source files and hypothesizing "
                            "about causes. Your FIRST tool call must reproduce and "
                            "observe the failure: load the page in the browser (or "
                            "execute the code) and read the captured errors — "
                            "uncaught exceptions and console messages name the "
                            "exact file:line:col. Diagnose strictly from observed "
                            "output, and never apply a fix you cannot causally tie "
                            "to an observed error message."
                        ),
                    })
                    # Defect report against a DONE project (2026-07-18):
                    # record the pending work IN the project — see
                    # _note_defect_on_done_project.
                    self._note_defect_on_done_project(lc)

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

                # Extracted to module level (testable) + follow-up
                # inheritance: corrections of a prior fenced-code answer
                # keep coding mode/sampling instead of falling back to the
                # temp-1.0 conversational profile.
                has_coding_intent, is_meta_task = detect_coding_intent(lc, messages)

                # ARCHITECTURAL OPTIMISATION #4: Request-scoped lazy cache.
                # Profile / playbook / tool defs / XML schema all become
                # one-and-done per request instead of per-turn.
                request_state = GhostAgent._RequestState(self)
                profile_context = await request_state.get_profile_str()

                # Metacog: reset per-request arbitration counter so the
                # MAX_ARBITRATIONS_PER_REQUEST cap is enforced per user
                # turn, not per agent lifetime. Safe no-op when bundle
                # is absent.
                _mc_reset = getattr(self.context, "metacog", None)
                if _mc_reset is not None:
                    try:
                        _mc_reset.reset_arbitration_counter()
                    except Exception:
                        pass

                working_memory_context = ""

                # KV-CACHE FIX: per-turn / query-keyed blocks (selfhood
                # wake-up, workspace continuity, uncertainty, graduated
                # skills, MCTS hint) used to be prepended/appended to the
                # system slot, which changed its bytes every turn and
                # defeated ARCHITECTURAL OPTIMISATION #1 (byte-stable system
                # prefix → upstream prefix-cache hits). They now ride in the
                # per-turn TAIL injection instead (see `continuity_blocks`
                # → transient_injection below), so the system slot is once
                # again byte-identical across turns. The model still sees all
                # of this content — just at the end of the prompt, next to
                # the live system state, rather than the start.
                continuity_blocks = []

                base_prompt = SYSTEM_PROMPT.replace("{{PROFILE}}", profile_context)

                # Selfhood wake-up prefix (recognition layer, proposal item #4).
                # Splices the agent's own past — autobiographical
                # experiences, open questions, mood, the running diary —
                # in as first-person continuity material (collected into
                # `continuity_blocks` and emitted in the per-turn tail
                # injection to keep the system slot KV-cache-stable). The
                # block is bounded by SELFHOOD:BEGIN / SELFHOOD:END markers
                # so evaluators can strip it. No-op when
                # --no-memory / --no-self-model / no prior state.
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
                    if (_SELFHOOD_PREFIX_ENABLED
                            and isinstance(self_model, _SelfModel)
                            and getattr(self_model, 'enabled', False)):
                        # Pass the current request as `query` so the
                        # wake-up prefix surfaces RELEVANT past experiences
                        # (recall keyed to what's being asked now), not
                        # just the most recent N. Proposal item #2.
                        wakeup_prefix = self_model.build_wakeup_prefix(
                            recent_experiences_n=3,
                            query=last_user_content,
                        )
                        if isinstance(wakeup_prefix, str) and wakeup_prefix:
                            continuity_blocks.append(wakeup_prefix)
                except Exception as e:
                    logger.debug(f"selfhood wake-up prefix skipped: {type(e).__name__}: {e}")

                # Workspace continuity prefix (world-model counterpart
                # to the selfhood block just collected above). Reads
                # workspace state, recent activity, file diffs since
                # last scan; also rides in the per-turn tail injection.
                # Same defensive isinstance gate against MagicMock
                # contexts. Non-fatal: a failure must never break a turn.
                try:
                    from ..workspace import WorkspaceModel as _WorkspaceModel
                    workspace_model = getattr(self.context, 'workspace_model', None)
                    if isinstance(workspace_model, _WorkspaceModel) and getattr(workspace_model, 'enabled', False):
                        # Keep the model's active-project pointer in sync so
                        # both recorded events and the wake-up prefix are
                        # scoped to THIS project — a prior project's research
                        # artifacts / narrative must not bleed into a new one.
                        _active_pid = getattr(self.context, 'current_project_id', None) or ""
                        try:
                            workspace_model.current_project_id = _active_pid
                            # Also bind the TASK-LOCAL event stamp: record_*
                            # calls later in this turn read the ContextVar
                            # first, so a concurrent writer (idle autoadvance
                            # tick, self-play temp agent) mutating the shared
                            # attribute mid-turn can't mis-stamp this turn's
                            # events with another context's project.
                            from ..workspace import set_event_project as _set_evt_pid
                            _set_evt_pid(_active_pid)
                        except Exception:
                            pass
                        # Gate the workspace wake-up prefix on an ACTIVE project:
                        # with no project context the prose adds tokens but no
                        # task signal. The useful behaviour-shaping bits (file
                        # parse-warnings, re-fetch nudge) only exist inside real
                        # project work, which always carries an active pid.
                        if _active_pid:
                            ws_prefix = workspace_model.build_wakeup_prefix(active_project_id=_active_pid)
                            if isinstance(ws_prefix, str) and ws_prefix:
                                continuity_blocks.append(ws_prefix)
                except Exception as e:
                    logger.debug(f"workspace wake-up prefix skipped: {type(e).__name__}: {e}")

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
                            continuity_blocks.append(_uctx)
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
                            continuity_blocks.append(_skblock)
                except Exception as e:
                    logger.debug(f"graduated-skill injection skipped: {e}")

                # Deep-reason MCTS lookahead (proposal item #8). For
                # requests the complexity router GENUINELY classified as
                # hard, run the MCTS action search and inject its top
                # next-action as a planning hint. This is the wiring that
                # makes `select_best_action` load-bearing — previously
                # the reasoner was constructed under --deep-reason but
                # never called in the turn loop.
                #
                # Gating is deliberately strict — the search costs ~4 LLM
                # round-trips, so it must NEVER fire on a cheap request:
                #   * mcts_reasoner present (None unless --deep-reason);
                #   * a high-confidence "hard" verdict — `escalated` is
                #     excluded on purpose: an untrained / low-confidence
                #     router escalates EVERY request (label defaults to
                #     "hard", escalated=True), and treating that as hard
                #     fired the search on trivial greetings (25 s for a
                #     "hello"). With no trained router MCTS simply stays
                #     off — the correct fail-safe;
                #   * not a trivial-chat turn.
                # Bounded by a wall-clock timeout; non-fatal.
                try:
                    _mcts = getattr(self.context, 'mcts_reasoner', None)
                    _rd = body.get("_router_decision") or {}
                    _is_hard = (_rd.get("label") == "hard"
                                and not _rd.get("escalated"))
                    if (_MCTS_TURNSTART_ENABLED and _mcts is not None and _is_hard
                            and last_user_content
                            and not self._is_strict_trivial_chat(lc)):
                        from ..tools.registry import TOOL_DEFINITIONS as _MCTS_TD
                        _tool_names = [
                            t["function"]["name"] for t in _MCTS_TD
                            if t.get("function", {}).get("name")
                        ]
                        # Build the PRM prefix-state so the trained PRM
                        # fast path engages (was dead: prm_state was never
                        # constructed, so even a trained PRM never scored a
                        # live candidate — MCTS always paid 3-4 worker-LLM
                        # simulation round-trips). This is turn start, so
                        # steps/failures are 0; pending_count/plan_depth are
                        # pinned to the SAME neutral constants (1/1) the PRM
                        # was trained against (prm.labels._build_state_for_step
                        # / frontier_selection._seed_state_action) to avoid
                        # train/serve skew. When no trained PRM is loaded the
                        # MCTS gate falls back to LLM simulation automatically.
                        _prm_state = None
                        try:
                            from ..prm.features import PlanState as _PRMPlanState
                            _prm_state = _PRMPlanState(
                                user_request=last_user_content,
                                steps_so_far=0,
                                failures_so_far=0,
                                pending_count=1,
                                plan_depth=1,
                                tools_used_this_turn=(),
                                tools_failed_this_turn=(),
                            )
                        except Exception:
                            _prm_state = None
                        _winner = await asyncio.wait_for(
                            _mcts.select_best_action(
                                task=last_user_content,
                                plan_state="(turn start — no actions taken yet)",
                                available_tools=_tool_names,
                                context=working_memory_context or "",
                                prm_state=_prm_state,
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
                            continuity_blocks.append(_hint)
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
                    # Inject as item 5 before Tool Orchestration. This runs at
                    # request start — no tool has executed yet, so we point the
                    # model at the tool_response blocks it will see in-context
                    # rather than embedding output here (the old inline
                    # placeholder was always the literal string "None").
                    base_prompt = base_prompt.replace(
                        "### TOOL ORCHESTRATION",
                        '5. THE "PERFECT IT" PROTOCOL: Upon successfully completing a complex technical task, analyze the result (the most recent <tool_response> output in this conversation) and proactively suggest one concrete way to optimize it.\n\n### TOOL ORCHESTRATION'
                    )
                base_prompt += working_memory_context

                # Join the per-turn continuity/hint blocks collected above.
                # These ride in the tail injection (transient_injection) so
                # the system slot stays byte-stable for KV-cache reuse.
                continuity_text = "\n\n".join(b for b in continuity_blocks if b).strip()

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
                if (is_trivial_greeting and last_user_content
                        and self._is_strict_trivial_chat(lc)
                        and not getattr(self, "_correction_active_this_turn", False)):
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
                    # Only resolve anaphors. A short message that already
                    # names its own subject (ids, quotes, backticks — e.g.
                    # "delete `516217d294cc`") is self-contained; prepending
                    # the previous reply there contaminates retrieval and was
                    # the root cause of partial-failure turns re-answering the
                    # PRIOR question instead of the current command.
                    if (len(search_query.split()) < 10 and len(messages) >= 2
                            and not self._has_concrete_reference(search_query)):
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
                            # Pass llm_client so RAG-Fusion's query-decomposition
                            # runs. Budget reduced 12k→4k: the review found up to
                            # 12k chars of mostly-irrelevant memory was crowding
                            # the actual task. With fused-score ordering + per-tier
                            # relevance gating now governing inclusion (see bus.py),
                            # a tighter budget surfaces the relevant items and drops
                            # the noise instead of padding to a fixed section quota.
                            # INTERNAL requests (sub-/sched-/job- — machine
                            # sub-calls like chess moves or scheduled jobs)
                            # skip the LLM-assisted RAG-Fusion decomposition:
                            # their prompts are self-contained and machine-
                            # generated, and the DECOMPOSE_QUERY worker round-
                            # trip sat on the critical path (observed live
                            # 2026-07-12: 8s ReadTimeout at +0.00s of every
                            # chess move while nova chewed the previous move's
                            # background extracts). Plain vector recall still
                            # runs — llm_client=None is the bus's no-LLM path.
                            from .autonomous_activity import (
                                is_internal_request as _is_int_req_h)
                            fetched_context = await bus.hydrate_context(
                                search_query,
                                llm_client=(None if _is_int_req_h(req_id)
                                            else getattr(self.context,
                                                         "llm_client", None)),
                                context_budget=4000,
                                # Stamp the stash so only THIS turn's judge
                                # consumes it, and keep the active session's
                                # own stored history out of the PAST
                                # CONVERSATIONS tier.
                                turn_id=str(req_id or ""),
                                exclude_session_id=str(
                                    body.get("session_id") or ""),
                            )
                            if fetched_context:
                                fetched_context = fetched_context.replace("\r", "")
                                pretty_log("Memory Bus", f"Hydrated context for: {search_query}", icon=Icons.BRAIN_CTX)
                        except Exception as e:
                            logger.error(f"MemoryBus hydration failed: {e}")

                # Surface any past belief revision relevant to this turn's
                # query (feature 1C). The contradiction engine logs every
                # supersede via ContradictionLog.record (agent.py belief
                # revision + project_advancer), but explain_belief_change had
                # no live caller — so the agent could never actually say "I
                # previously thought X, updated to Y." Inject the explanation
                # alongside the hydrated memory, query-scoped to the user's
                # message, so it can. Best-effort.
                if last_user_content and should_fetch_memory:
                    _clog = getattr(self.context, "contradiction_log", None)
                    if _clog is not None:
                        try:
                            _belief = await asyncio.to_thread(
                                _clog.explain_belief_change, last_user_content,
                            )
                            if _belief:
                                fetched_context = (
                                    f"{fetched_context}\n\n{_belief}".strip()
                                    if fetched_context else _belief
                                )
                                pretty_log(
                                    "Belief Revision",
                                    "Surfaced a relevant past belief change",
                                    icon=Icons.BRAIN_CTX,
                                )
                        except Exception as e:
                            logger.debug("explain_belief_change surfacing failed: %s", e)

                fetched_playbook = ""  # Now dynamically populated inside the loop

                # ============================================================
                # ARCHITECTURAL OPTIMISATION #1: KV-CACHE-STABLE SYSTEM PROMPT
                # ------------------------------------------------------------
                # The system slot used to be rebuilt per turn (persona +
                # skill_instruction + tool schemas) which changed bytes from
                # turn to turn and invalidated upstream KV-cache prefixes.
                # We lock the system slot to SYSTEM_PROMPT + {{PROFILE}} +
                # skill_instruction only — tool schemas, live system state,
                # AND the per-turn continuity/hint blocks (selfhood, workspace,
                # uncertainty, graduated-skill, MCTS — collected above into
                # `continuity_blocks`) all ride in the per-turn user-message
                # header (transient_injection), which changes every turn anyway.
                #
                # This guarantee had silently regressed: the selfhood/workspace
                # wake-up prefixes were being PREPENDED to the system slot and
                # the uncertainty/skill/MCTS blocks APPENDED to it, all keyed to
                # the live query — so the first bytes of the prompt differed
                # every turn and llama-server logged `n_past=1 … forcing full
                # prompt re-processing`, re-prefilling ~17k tokens (~20s) per
                # turn. Moving them to the tail restores the byte-identical
                # prefix; only {{PROFILE}} remains, and it changes rarely
                # (update_profile only).
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
                # One-shot latch for the promised-notification guard: the
                # steer fires at most once per request (a model that still
                # won't call notify_operator after the alert ships its
                # final response rather than looping).
                notify_steer_fired = False
                # One-shot latch for the trailing-promise guard: a final
                # reply whose LAST sentence promises imminent action ("Let
                # me fix it.") gets steered once to either act or state the
                # action was not done; never loops.
                pending_promise_steer_fired = False
                execution_failure_count = 0
                transient_failure_count = 0   # Separate budget for retryable errors
                # Repeated-IDENTICAL-failure tracking. The structural strike
                # count decays by 1 on any successful tool turn, so an agent
                # that oscillates "same failing read → list sandbox (succeeds)
                # → same failing read …" never trips the strike cap and burns
                # all 40 turns (observed: a new project whose stale workspace
                # narrative kept asserting a prior project's index.html existed
                # → the model re-read the missing path every other turn forever).
                # Keyed by (tool, normalised-error) signature; once a signature
                # recurs, decay is frozen so genuine accumulation resumes and
                # the loop-breaker can fire, and the model is told ONCE to stop.
                # All of this request-scoped state — the failure/action
                # signature dicts, the persistent-failure freeze flag + warned
                # set, and the consecutive-clean-success counter that unfreezes
                # the decay after a genuine pivot — lives in one StrikeLedger
                # (core.strikes) instead of five interacting locals.
                strikes = StrikeLedger()
                # Counts CONSECUTIVE `system_parse_error` events across turns
                # within this request. Reset on any successful parse. After
                # threshold (≥2) we pivot the recovery prompt to suggest
                # alternative tool-call shapes — the default "use XML"
                # message is provably not breaking the model out of the loop
                # (see selfplay session, attempt 2: 5 identical failures).
                consecutive_parse_errors = 0
                tools_run_this_turn = []
                # Last dispatched tool name. Must be initialised here: the
                # streaming metacog block references it from a closure, and on
                # turns where no tool ran an unbound `fname` raises NameError
                # — silently swallowed, killing the calibration sample.
                fname = ""
                forget_was_called = False
                thought_content = ""
                # Pre-bind: `payload` is (re)built each LLM iteration, but a
                # deterministic-dispatch exit can reach finalization without one;
                # FinalizeState construction must never hit an unbound name.
                payload = None
                was_complex_task = False

                task_tree = TaskTree()
                current_plan_json = {}
                _request_sys3_fired_once = False
                _request_sys3_prev_justification = ""
                force_final_response = False

                # --- Interactive "one task per turn" enforcement -------------
                # The project briefing re-advertises a live NEXT TASK on EVERY
                # loop iteration, which pulls a small model into grinding the
                # whole project tree on a single "start task 1" / "proceed"
                # (observed live: one go-ahead built 8 tasks). The HARD-RULE
                # prompt text alone can't stop it. So: once a project task is
                # actually closed DONE this request, suppress the NEXT TASK
                # pointer and force the turn to wrap up — UNLESS the user asked
                # for a batch ("do the next 3", "finish the project"), which the
                # prompt routes to manage_projects autoadvance instead.
                _proj_task_closed_this_req = False
                try:
                    from .project_advancer import classify_advance_intent as _cai
                    _user_batch_intent = (_cai(
                        getattr(self.context, "last_user_content", "") or ""
                    ).get("mode") in ("all", "n"))
                except Exception:
                    _user_batch_intent = False

                # --- VERIFIER-GATE AUTO-REPAIR state (bounded) ---
                # When the verifier REFUTES the final answer (or the turn
                # finalised on an unverified mutation), inject the critique
                # and re-run the turn loop up to `_MAX_VERIFIER_REPAIRS`
                # times so the agent can actually FIX the issue rather than
                # ship a noted-but-wrong answer. Re-entry reuses the existing
                # `for turn` loop (a `continue` at the normal-success
                # finalisation) — no outer loop. `repair_round` bounds it
                # independently of the turn budget. The verdict computed at
                # finalisation is cached + reused by the post-loop gate
                # (`_verdict_is_fresh`) so a clean success costs exactly one
                # verifier pass, same as before.
                repair_round = 0
                _verifier_verdict_cache = None
                _verdict_is_fresh = False
                _final_len_at_turn_start = 0

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

                # No-progress (ungrounded-verification) loop tracking — the
                # companion to `repeated_failure_sigs`. That catches the same
                # FAILING action looping; this catches the same SUCCEEDING
                # action looping with no new information (e.g. re-click the
                # icon → screenshot → "no change" → repeat). The error-keyed
                # strike counter never moves (nothing errors) and the
                # reasoning-similarity breaker misses it (prose phrasing
                # varies), so this is the only thing that sees an all-success
                # thrash. Keyed by (tool, target, result-fingerprint) via
                # `_note_repeated_action`; on the first trip we force a
                # grounded final answer, escalating to force_stop if it
                # somehow continues. The (tool, target, result) signature
                # dict lives on the StrikeLedger above; only the "already
                # steered once" set is loop-local.
                repeated_action_steered: set = set()
                preflight_blocks_this_request = 0
                # Context-pressure steers issued this request (governor,
                # 2026-07-18): first overflow → externalize-notes steer;
                # second → synthesize-now steer + whole-file-read lockdown.
                context_pressure_steers = 0

                # Self-play can cap a single attempt's turn count via
                # `max_turns_override` on the GhostAgent instance, so a
                # runaway simulation can't silently chew through 40 turns.
                effective_max_turns = getattr(self, "max_turns_override", None) or 40
                for turn in range(effective_max_turns):
                    # Cooperative cancellation boundary (2026-07-11). A user
                    # who cancels gets a clean stop with whatever work is
                    # already done, rather than a killed task — and the
                    # semaphore is released either way. A turn wedged INSIDE a
                    # long upstream call never reaches this line; that is what
                    # `hard=true` (task.cancel()) is for. See core/turns.py.
                    if _turn_reg.is_cancelled(req_id):
                        raise TurnCancelled(req_id, _active_turn.reason)
                    self.context.last_activity_time = datetime.datetime.now() # Heartbeat
                    # Per-turn auto-repair bookkeeping: the verdict cache is
                    # only "fresh" for the post-loop gate if THIS turn reached
                    # the normal-success finalisation; reset each turn so an
                    # error/abort exit can't reuse a stale verdict. Snapshot
                    # the answer length so a repair `continue` can discard
                    # exactly this turn's text contribution.
                    _verdict_is_fresh = False
                    _final_len_at_turn_start = len(final_ai_content or "")

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

                    # --- DETERMINISTIC TERMINAL-TOOL DISPATCH (turn 0) -------
                    # A bare "self play" / "self play again" (and likewise
                    # "dream mode" / "go to sleep") used to run the cycle only
                    # on the FIRST ask; every repeat replayed the previous
                    # turn's text summary instead of re-firing. Root cause: the
                    # direct-from-tool-summary bypass below persists the cycle
                    # result as a PLAIN-TEXT assistant turn with NO tool_call,
                    # and the memory bus re-hydrates it as "Context: Self-play
                    # complete…", so the model's only in-context example says
                    # the right response to "self play" is to reprint that text
                    # — which it does (identical 126-char reply, tool=- in
                    # metacog), never re-running.
                    #
                    # Fix: when the message is unambiguously a single-cycle
                    # terminal command (self_play or dream_mode), dispatch the
                    # tool deterministically here and bypass the LLM turn
                    # entirely — the model's indecision can no longer swallow
                    # the request. We also append a real assistant tool_call +
                    # tool result to history so the in-context example shows a
                    # TOOL invocation, not bare text; any future model-driven
                    # turn (longer phrasings that skip this fast path) then
                    # imitates the tool, not the summary. Breaking the
                    # `for turn` loop falls straight through to the post-loop
                    # metacog/calibration finalisation, so the reading still
                    # records (domain=memory, tool=<name>) exactly as the
                    # model-driven path did. The `selfplay` budget guard
                    # prevents any recursion from a self-play solver sub-agent
                    # (whose challenge text would not match the command gate
                    # anyway — belt and suspenders).
                    _det_tool = (
                        _explicit_terminal_command(last_user_content)
                        if (turn == 0 and not force_stop
                            and getattr(self, "thinking_budget_override", None) != "selfplay")
                        else None
                    )
                    if _det_tool is not None and _det_tool in self.available_tools:
                        pretty_log(
                            "Terminal Tool",
                            f"Explicit '{_det_tool}' command — dispatching deterministically (bypassing LLM turn).",
                            icon=Icons.STOP,
                        )
                        try:
                            _det_raw = str(await self.available_tools[_det_tool]() or "").strip()
                        except Exception as _det_exc:
                            logger.error(f"Deterministic {_det_tool} dispatch failed: {_det_exc}")
                            _det_raw = ""
                        # Record as a genuine tool_call + result so subsequent
                        # turns imitate the tool, not the text summary.
                        _det_call_id = f"det_{_det_tool}_{turn}"
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": _det_call_id,
                                "type": "function",
                                "function": {"name": _det_tool, "arguments": "{}"},
                            }],
                        })
                        _det_tool_msg = {
                            "role": "tool",
                            "tool_call_id": _det_call_id,
                            "name": _det_tool,
                            "content": _det_raw,
                        }
                        messages.append(_det_tool_msg)
                        tools_run_this_turn.append(dict(_det_tool_msg))
                        _det_body = _distill_terminal_tool_summary(_det_tool, _det_raw)
                        _det_prefix = {
                            "self_play": "Self-play complete.",
                            "dream_mode": "Dream cycle complete.",
                        }.get(_det_tool, f"`{_det_tool}` complete.")
                        final_ai_content = (
                            f"{_det_prefix}\n\n{_det_body}"
                            if _det_body else _det_prefix
                        )
                        force_stop = True
                        break

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
                            from ..tools.file_system import tool_list_files, project_scoped_sandbox
                            params = {
                                "sandbox_dir": project_scoped_sandbox(self.context)[0],
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
                        # Apply a GEPA-optimized planning instruction when one
                        # has been produced offline (run_gepa → system/optim/
                        # planning.decompose.json). Prepend it as an authoritative
                        # directive rather than replacing the structured prompt —
                        # the tuned text is a short instruction, the baseline is a
                        # full multi-section system prompt. Falls back to baseline
                        # when no tuned file exists. (Was write-only before.)
                        from ..optim.loader import tuned_instruction as _tuned_instruction
                        _tuned_plan = _tuned_instruction("planning.decompose", "")
                        # Also surface the tuned tool-selection instruction:
                        # the planner's next_action_id / required_tool IS the
                        # tool-selection decision, so this is its natural
                        # read-site. tool_selection.pick was previously
                        # tuned-but-never-read (GEPA wrote it, nothing loaded it).
                        _tuned_toolsel = _tuned_instruction("tool_selection.pick", "")
                        _tuned_prefix = "\n\n".join(
                            t for t in (_tuned_plan, _tuned_toolsel) if t
                        )
                        _planner_system = (
                            f"{_tuned_prefix}\n\n{PLANNING_SYSTEM_PROMPT}"
                            if _tuned_prefix else PLANNING_SYSTEM_PROMPT
                        )
                        planner_messages = [
                            {"role": "system", "content": _planner_system},
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

                    # Progressive DETERMINISTIC compression (L1-L3, no LLM)
                    # BEFORE the expensive summarization prune. This graceful
                    # degradation (compress old tool outputs → collapse verbose
                    # assistant turns → truncate old user msgs) reduces how
                    # often _prune_context has to fire its LLM summarization
                    # (which pays an upstream call). Capped at L3 so every
                    # message is preserved (tool-call/result pairing intact);
                    # _prune_context stays the L4 emergency for the extreme case.
                    try:
                        messages = self._get_context_manager().compress_if_needed(
                            messages, max_level=3)
                    except Exception as _cmx:
                        logger.debug("progressive compression skipped: %s", _cmx)

                    # Proactive Context Pruning before request
                    _pre_prune_tokens = _estimate_messages_tokens(messages)
                    messages = await self._prune_context(messages, max_tokens=self.context.args.max_context, model=model)
                    # Context-pressure governor (2026-07-18). A prune that
                    # actually fired means detail was just summarized away —
                    # without a steer the model keeps bulk-reading, re-reads
                    # what compaction destroyed, and spirals (xrick session:
                    # 60 file reads, 2 compactions, 25+ min, dead turn).
                    if _pre_prune_tokens > int(getattr(self.context.args, "max_context", 0) or 0):
                        context_pressure_steers += 1
                        if context_pressure_steers == 1:
                            messages.append({"role": "user", "content": (
                                "SYSTEM ALERT (context pressure): the conversation "
                                "just exceeded the context window and older detail was "
                                "summarized away. STOP bulk-gathering NOW. This turn: "
                                "(1) WRITE everything learned so far as compact notes "
                                "to a file in the project (e.g. 'analysis_notes.md' "
                                "via file_system) — notes on disk survive compaction, "
                                "context does not; (2) from now on consult those notes "
                                "instead of re-reading sources; (3) gather only "
                                "targeted evidence: operation='search', ranged reads "
                                "(start_line/end_line), or an 'execute' script that "
                                "prints a compact digest. If you already have enough, "
                                "produce the deliverable NOW."
                            )})
                        elif context_pressure_steers == 2:
                            self.context._ctx_pressure_lockdown = True
                            pretty_log(
                                "Context Governor",
                                "second overflow this request — whole-file reads "
                                "locked for the remainder; steering to synthesize",
                                level="WARNING", icon=Icons.CUT,
                            )
                            messages.append({"role": "user", "content": (
                                "SYSTEM ALERT (context pressure — SECOND overflow): "
                                "whole-file reads are now DISABLED for the rest of "
                                "this request. Produce the deliverable NOW from your "
                                "notes and what you already know. If something "
                                "specific is missing, fetch ONLY that via "
                                "operation='search' or a ranged read."
                            )})

                    # Dynamic Context Cache Tool Injection (Context Bloat Fix)
                    # ARCHITECTURAL OPTIMISATION #4 + #7: cached lookups via
                    # the request-scoped state. Repeated turns with the same
                    # search_query and tool list now hit the LRU + per-request
                    # caches instead of re-filtering and re-serialising.
                    search_query = thought_content if (use_plan and locals().get('thought_content')) else last_user_content
                    # #7: route acquired-skill selection off a REQUEST-STABLE
                    # query (pinned to the first substantive query) so the
                    # advertised tool set — and thus the tool header block —
                    # stays byte-identical across the request's turns, letting
                    # the upstream prompt-prefix KV cache hold instead of
                    # re-prefilling every turn.
                    all_tools = request_state.get_active_tool_defs(
                        request_state.stable_tool_query(search_query or ""))
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
                        # Project-aware since 2026-07-18: the static
                        # "/workspace" wording taught the model to `cd`
                        # OUT of its project scope (see _render_cwd_pin).
                        dynamic_state += _render_cwd_pin(
                            getattr(self.context, "current_project_id", None))
                    # Minute precision (not seconds): a second-precision
                    # timestamp changes on every turn of the same request,
                    # busting the upstream prefix KV-cache for everything that
                    # follows this block. The model never needs sub-minute
                    # resolution; minute precision keeps the block byte-stable
                    # across the turns of a single request so the cache holds.
                    dynamic_state += f"### DYNAMIC SYSTEM STATE\nCURRENT TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} (Day: {datetime.datetime.now().strftime('%A')})\n\nSCRAPBOOK:\n{scratch_data}\n\n"
                    try:
                        from .prompts import build_project_briefing
                        _proj_store = getattr(self.context, "project_store", None)
                        _proj_id = getattr(self.context, "current_project_id", None)
                        if _proj_store is not None and _proj_id:
                            _briefing = build_project_briefing(
                                _proj_store, _proj_id,
                                suppress_next_task=_proj_task_closed_this_req,
                                # Pass the graph so the briefing can surface
                                # RELATED WORK — other projects sharing tech
                                # (feature 3B).
                                graph_memory=getattr(self.context, "graph_memory", None))
                            if _briefing:
                                dynamic_state += _briefing + "\n"
                    except Exception:
                        logger.debug("project briefing skipped", exc_info=True)
                    # EXPLICIT CONSTRAINTS OF THE CURRENT REQUEST — surfaced
                    # BEFORE the model plans anything, project or not. The
                    # project briefing covers constraints stored on the
                    # project record; this block covers the message the user
                    # just sent (which is where a correction like "don't come
                    # up with some random AI, YOU will play against me"
                    # first appears). Deterministic extraction, no LLM call.
                    if _request_constraint_block:
                        dynamic_state += _request_constraint_block + "\n\n"
                    # Re-assert the wrap-up gate every iteration: once a project
                    # task was closed this request, the turn must converge to a
                    # final answer (no further tool calls / no next task).
                    if _proj_task_closed_this_req:
                        force_final_response = True
                    if has_coding_intent:
                        dynamic_state += f"CURRENT SANDBOX STATE:\n{sandbox_state}\n\n"
                    if use_plan and not turn_is_conversational and 'thought_content' in locals() and thought_content:
                        dynamic_state += f"ACTIVE STRATEGY & PLAN:\nTHOUGHT: {thought_content}\nPLAN:\n{task_tree.render()}\nFOCUS TASK: {next_action_id}\n"

                        if str(next_action_id).strip().lower() == "none":
                            dynamic_state += "CRITICAL INSTRUCTION: DO NOT USE TOOLS this turn. Answer the user directly using insights from your THOUGHT.\n"
                            force_final_response = True
                        else:
                            dynamic_state += "CRITICAL INSTRUCTION: Execute the tool(s) required for the FOCUS TASK. You MAY emit MULTIPLE <tool_call> blocks in parallel within this turn when they all serve the same FOCUS TASK (e.g. writing several project files, batching knowledge_base inserts). DO NOT HALLUCINATE TOOL OUTPUTS.\n"

                    # Rewrite any leaked HOST-absolute sandbox path (from a
                    # recalled memory / scratchpad / workspace narrative) to its
                    # container /workspace form, so the model never builds a
                    # `cd /Users/.../sandbox/...` that ENOENTs in the container.
                    dynamic_state = _scrub_host_sandbox_paths(
                        dynamic_state, getattr(self.context, "sandbox_dir", None))

                    # -----------------------------------------------------------------
                    # QWEN-AGENT METHODOLOGY: Bypass Native Tools & Use String Prompts
                    # -----------------------------------------------------------------
                    target_tool = locals().get("required_tool", "all")
                    # With the planner disabled, default to final generation (stream directly).
                    # If the model returns tool_calls, the turn loop below will handle them.
                    # str(... or "all"): planners routinely emit `"required_tool": null`,
                    # which arrives here as None — .lower() on it would 500 the request.
                    is_final_generation = force_final_response or str(target_tool or "all").lower() == "none"

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
                                                        # Save into the active project's dir so the scoped
                                                        # vision_analysis finds it by bare name and it shows
                                                        # in the scoped listing (vision also has a root
                                                        # fallback as a safety net).
                                                        from ..tools.file_system import project_scoped_sandbox
                                                        tmp_path = project_scoped_sandbox(self.context)[0] / filename
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
                    _continuity_tail = f"\n\n{continuity_text}" if continuity_text else ""
                    # Cache-aware ordering: STABLE blocks first (tool schemas,
                    # persona, playbook, hydrated memory, per-request continuity
                    # — all byte-identical across the turns of one request),
                    # VOLATILE `dynamic_state` LAST (timestamp / sandbox state /
                    # plan focus change every turn). The upstream prefix KV-cache
                    # holds up to the first differing byte, so pushing the only
                    # per-turn-varying block to the end maximises the cached
                    # region — turns 2+ re-prefill just dynamic_state + the user
                    # instruction instead of the whole (large) injection.
                    _stable_injection = f"{tool_header_block}\n\n{active_persona}{fetched_playbook}{fetched_context}{_continuity_tail}"
                    # Measurement hook: a stable hash across a request's turns
                    # means the prefix is cacheable; a changing hash points at a
                    # remaining buster. Cheap (one sha1 of already-built text).
                    try:
                        _sp_hash = hashlib.sha1(_stable_injection.encode("utf-8", "ignore")).hexdigest()[:8]
                        pretty_log(
                            "Prefill Cache",
                            f"stable-prefix h={_sp_hash} len={len(_stable_injection)} · "
                            f"volatile dyn_state len={len(dynamic_state)}",
                            icon=Icons.BRAIN_CTX,
                        )
                    except Exception:
                        pass

                    # GHOST_PIN_TOOL_SCHEMAS: pin the byte-stable block to a
                    # FIXED position so the upstream KV-cache reuses it across
                    # every turn of this request (see `_compose_injection`).
                    # ENABLED durably in PROD via the launcher export
                    # (bin/start-ghost-agent.sh) and validated holding — a
                    # per-turn "prefill cache · stable-prefix h=…" log line whose
                    # hash is stable within a conversation. The CODE default is
                    # deliberately left OFF: flipping it globally reorders prompt
                    # assembly for every non-prod launch and trips 8 integration
                    # tests that pin the unpinned message layout, so durability
                    # lives in the launcher (the real prod path), not here.
                    _pin_stable = os.getenv("GHOST_PIN_TOOL_SCHEMAS", "0").strip().lower() not in ("0", "false", "no")
                    req_messages = self._compose_injection(
                        req_messages, _stable_injection, dynamic_state, _pin_stable,
                    )

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

                    # Grammar-constrained tool calls (2026-07-17): attach a
                    # LAZY GBNF grammar built from the active tool schemas.
                    # Dormant through thinking/prose; from the moment the
                    # model opens a line with `<tool_call>` only valid
                    # framing / known tools / known parameter names /
                    # enum-legal values / numeric-legal integers can decode
                    # — the malformed-call strike class dies at the sampler.
                    # Validated against the live llama-server build before
                    # wiring (pattern-type triggers; /v1/chat/completions).
                    # Best-effort: `{}` on any failure, and servers without
                    # the fields ignore them. GHOST_TOOL_GRAMMAR=0 disables.
                    if not is_final_generation and all_tools:
                        try:
                            from .tool_grammar import grammar_payload_fields
                            payload.update(grammar_payload_fields(all_tools))
                        except Exception as _tg_exc:
                            logger.debug("tool grammar skipped: %s", _tg_exc)

                    pretty_log("LLM Request", f"Turn {turn+1} | Temp {sampling_params['temperature']:.2f}", icon=Icons.LLM_ASK)

                    if is_final_generation and stream_response:
                        payload["stream"] = True
                        # Metacog logprobs opt-in (roadmap phase 2.1). When
                        # the bundle is wired and the operator has not opted
                        # out via --metacog-disable-logprobs, ask upstream
                        # for token-level top_logprobs so the streaming
                        # consumer can compute rolling Shannon entropy.
                        # llama.cpp + vLLM both honour the OpenAI-style
                        # extension; servers that don't recognise it ignore
                        # the fields. Default top_k=5 matches the entropy
                        # tracker's K-normalisation.
                        _mc = getattr(self.context, "metacog", None)
                        if (_mc is not None and getattr(_mc, "enabled", False)
                                and getattr(_mc, "logprobs_enabled", False)):
                            payload.setdefault("logprobs", True)
                            payload.setdefault("top_logprobs", 5)
                        # Capture outer variables to prevent NameError when finally block deletes them
                        stream_messages_snapshot = list(messages[-10:])
                        stream_tools_snapshot = list(tools_run_this_turn)
                        stream_thought = thought_content
                        stream_model = model
                        # Verifier-gate captures (2026-07-18). The verdict's
                        # code-reconstruction walks tool_call ids across the
                        # WHOLE message list, and the correction banner is
                        # keyed by the conversation fingerprint of the FIRST
                        # user message — messages[-10:] loses both on long
                        # turns, and the live `messages` is deleted by the
                        # outer finally before the stream drains. Shallow
                        # copy + eager fingerprint, taken while it's alive.
                        stream_verify_messages = list(messages)
                        stream_conv_fp = self._conversation_fingerprint(messages)

                        # NEW: Capture accumulated intermediate text (like image tags from previous turns)
                        # Prepend any deferred async-verdict correction (banner)
                        # so it leads the stream even when there's no other
                        # intermediate text to flush.
                        _corr_banner = self._take_active_correction()
                        _body_prefix = final_ai_content.strip() + "\n\n" if final_ai_content.strip() else ""
                        stream_prefix = _corr_banner + _body_prefix

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
                            # Once ANY '<' appears we must run the full scrub
                            # regex (a tag might be forming). Until then — the
                            # common case for a plain-text final answer — the
                            # sub is a guaranteed no-op, so we skip it and emit
                            # the raw new tokens. This turns the per-chunk
                            # O(len(answer)) re-sub into O(1) for tag-free
                            # answers (the majority of final generations).
                            _scrub_seen_lt = False

                            # Metacog entropy tracker (roadmap phase 2.2).
                            # Allocated once per stream; observes one top-K
                            # logprob vector per chunk that carries one.
                            # Reading is stashed on the agent context for
                            # the post-stream composite-confidence check.
                            _mc_for_stream = getattr(self.context, "metacog", None)
                            _entropy_tracker = None
                            # Create the tracker whenever metacog is enabled —
                            # NOT gated on logprobs_enabled. The post-stream
                            # confidence calc is logprob-OPTIONAL (phase 2.5):
                            # token entropy contributes when it flows, but
                            # competence + verbalised-uncertainty drive the
                            # score when the per-token logprob stream is sparse
                            # (speculative decoding / MTP) or disabled. Keeping
                            # the tracker present means that calc always runs
                            # and calibration collects a sample every turn.
                            if (_mc_for_stream is not None
                                    and getattr(_mc_for_stream, "enabled", False)):
                                try:
                                    from .entropy import EntropyTracker
                                    _entropy_tracker = EntropyTracker(window=32, top_k=5)
                                except Exception as _etx:
                                    logger.debug("entropy tracker init failed: %s", _etx)

                            async for chunk in self.context.llm_client.stream_chat_completion(payload, use_coding=has_coding_intent):
                                if loop_detected: break
                                # Cooperative cancel boundary (2026-07-15): the
                                # turn stays registered for the whole drain now,
                                # so /api/turn/cancel can flag it mid-stream —
                                # stop emitting on the next chunk. Finalization
                                # after the loop still runs on the partial text.
                                if _turn_reg.is_cancelled(req_id):
                                    break
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
                                            # Metacog: pipe top-logprobs into
                                            # the entropy tracker. Lives in
                                            # the same try/except as the
                                            # content decode so a malformed
                                            # logprobs payload never breaks
                                            # the stream — the tracker just
                                            # skips that chunk.
                                            if _entropy_tracker is not None:
                                                try:
                                                    from .entropy import extract_top_logprobs
                                                    _top_lps = extract_top_logprobs(chunk_data)
                                                    if _top_lps:
                                                        _entropy_tracker.observe(_top_lps)
                                                except Exception as _etxx:
                                                    logger.debug(
                                                        "entropy observe failed: %s",
                                                        _etxx,
                                                    )
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
                                    if not _scrub_seen_lt and "<" in full_content[_scrubbed_emitted_len:]:
                                        _scrub_seen_lt = True
                                    if _scrub_seen_lt:
                                        _scrubbed_view = _stream_scrub_pattern.sub('', full_content)
                                    else:
                                        # No '<' anywhere yet → the sub is a
                                        # no-op; the scrubbed view equals the
                                        # raw buffer. Skip the O(n) regex.
                                        _scrubbed_view = full_content
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
                                # When the turn was finalized BECAUSE a project
                                # task closed this turn (one-task-per-turn gate),
                                # the scrubbed tool_call was the model trying to
                                # barrel into the next task — say that honestly
                                # rather than telling the user to "rephrase" a
                                # request that already succeeded.
                                _fallback_text = _scrub_fallback_message(
                                    _intended, _proj_task_closed_this_req)
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
                                if _proj_task_closed_this_req:
                                    # Expected: the one-task gate finalized the
                                    # turn and dropped the model's attempt to
                                    # start the next task. Not a problem — log
                                    # at INFO so it doesn't read as an error.
                                    pretty_log(
                                        "One Task / Turn",
                                        f"Stopped after a task closed; dropped the model's "
                                        f"next-task {_intended or 'tool'} call — awaiting user go-ahead.",
                                        level="INFO", icon=Icons.BRAIN_PLAN,
                                    )
                                else:
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

                            # Metacog: compute this turn's composite confidence
                            # (logprob-OPTIONAL — phase 2.5). The entropy term
                            # is used when the window observed tokens; otherwise
                            # it is neutral (0.5) and competence + verbalised
                            # uncertainty drive the score. Runs on every metacog
                            # turn so calibration always records a sample, even
                            # on speculative-decoding / no-logprobs upstreams.
                            if _entropy_tracker is not None:
                                try:
                                    _reading = _entropy_tracker.reading()
                                    if _reading is not None:
                                        if _reading.n > 0:
                                            self.context.last_entropy_reading = _reading
                                        # Composite confidence (roadmap
                                        # phase 2.4). Fuse entropy with the
                                        # per-domain competence prior for
                                        # the most-recently-dispatched tool.
                                        # The result is logged and stashed
                                        # on the context so observability
                                        # and downstream gating share one
                                        # source of truth. Tool name comes
                                        # from `fname` if set this turn,
                                        # otherwise falls back to a global
                                        # roll-up via domain="other".
                                        _mc_conf = getattr(self.context, "metacog", None)
                                        if (_mc_conf is not None
                                                and getattr(_mc_conf, "confidence", None) is not None
                                                and getattr(_mc_conf, "competence", None) is not None):
                                            try:
                                                from .metacog import _domain_for_tool
                                                _dom = _domain_for_tool(fname or "")
                                                _p = _mc_conf.competence.estimate(_dom, fname or None)
                                                _n = _mc_conf.competence.observations(_dom, fname or None)
                                                # Verbalised-uncertainty pressure
                                                # (core.uncertainty) — fuses the
                                                # "agent said it was unsure" track
                                                # into the composite. No-op until
                                                # the calibration spine fits λ > 0.
                                                _upress = 0.0
                                                try:
                                                    _utk = getattr(self.context, "uncertainty_tracker", None)
                                                    if _utk is not None:
                                                        _upress = _utk.pressure()
                                                except Exception:
                                                    _upress = 0.0
                                                # Entropy term is neutral (0.5)
                                                # when the window is empty, so
                                                # competence + uncertainty drive
                                                # the score on logprob-starved
                                                # upstreams instead of suppressing
                                                # the whole reading.
                                                _norm = _reading.norm if _reading.n > 0 else 0.5
                                                _cr = _mc_conf.confidence.score(
                                                    normalised_entropy=_norm,
                                                    competence_p_success=_p,
                                                    n_observations=_n,
                                                    uncertainty_pressure=_upress,
                                                )
                                                self.context.last_confidence = _cr
                                                # Stash for the turn-end calibration
                                                # record (paired with the realized
                                                # outcome). Last reading of the turn
                                                # wins — that's the one we score
                                                # against the turn's outcome.
                                                self.context._calib_pending = _cr
                                                # Push the reading into the
                                                # bundle so the mid-turn
                                                # arbiter gate (consulted
                                                # at the next tool dispatch)
                                                # has authoritative state to
                                                # decide on. Without this
                                                # push the gate would never
                                                # fire — it reads from the
                                                # bundle, not the context.
                                                try:
                                                    _mc_conf.record_confidence(_cr)
                                                    _mc_conf.count(
                                                        confidence_total=True,
                                                        confidence_below=_cr.below_threshold,
                                                    )
                                                except Exception as _crx:
                                                    logger.debug(
                                                        "metacog record_confidence failed: %s",
                                                        _crx,
                                                    )
                                                # Log noise control: per-turn
                                                # confidence readings are
                                                # high-volume. We only want
                                                # INFO for the actionable
                                                # case (below threshold —
                                                # the arbiter is now armed
                                                # for the next mutating
                                                # dispatch). Above-threshold
                                                # readings drop to DEBUG so
                                                # monitoring greps stay
                                                # signal-rich.
                                                from .metacog_log import (
                                                    emit as _mc_emit,
                                                    Subsystem as _mc_ss,
                                                    LEVEL_INFO, LEVEL_DEBUG,
                                                )
                                                _mc_emit(
                                                    _mc_ss.CONF,
                                                    level=(LEVEL_INFO if _cr.below_threshold
                                                           else LEVEL_DEBUG),
                                                    below=_cr.below_threshold,
                                                    C=_cr.composite,
                                                    entropy=_cr.entropy_component,
                                                    competence=_cr.competence_component,
                                                    n=_n,
                                                    domain=_dom,
                                                    tool=fname or None,
                                                    ent_obs=_reading.n,
                                                    threshold=_cr.threshold,
                                                )
                                            except Exception as _cex:
                                                logger.debug(
                                                    "metacog confidence score failed: %s",
                                                    _cex,
                                                )
                                        logger.debug(
                                            "metacog entropy: norm=%.3f (n=%d)",
                                            _reading.norm, _reading.n,
                                        )
                                except Exception as _erx:
                                    logger.debug("entropy stash failed: %s", _erx)

                            # Internal (sub-/sched-) requests never feed smart
                            # memory: machine-generated prompts (chess FENs,
                            # job payloads) polluted retrieval AND their
                            # worker-side extract/scoring calls (max_tokens
                            # 3072) kept nova busy enough to time out the next
                            # request's routing calls (2026-07-12).
                            from .autonomous_activity import (
                                is_internal_request as _is_int_req_m1)
                            if self.context.args.smart_memory > 0.0 and last_user_content and not forget_was_called and not last_was_failure and not _is_int_req_m1(req_id):
                                micro_msgs = []
                                for m in [msg for msg in stream_messages_snapshot if msg.get("role") in ["user", "assistant"]][-4:]:
                                    role = m.get("role", "user").upper()
                                    clean_content = re.sub(r'```.*?```', '', str(m.get("content", "")), flags=re.DOTALL)
                                    micro_msgs.append(f"{role}: {clean_content[:500].strip()}")
                                clean_ai = re.sub(r'```.*?```', '', full_content, flags=re.DOTALL)
                                recent_arc = "\n".join(micro_msgs) + f"\nAI: {clean_ai[:500].strip()}"
                                if getattr(self.context, 'journal', None):

                                    await self._journal_append_safe('smart_memory', {'text': recent_arc, 'model': stream_model})

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

                                        await self._journal_append_safe('post_mortem', {'user': last_user_content, 'tools': stream_tools_snapshot, 'ai': full_content, 'model': stream_model})
                                    await self._record_episode_safe(last_user_content, stream_tools_snapshot, full_content)

                            # Post-turn hydration usefulness judge (fire-and-
                            # forget, worker-hosted) — every finalized turn,
                            # matching the non-streaming site.
                            self._judge_hydration_safe(
                                full_content, turn_id=str(req_id or ""))

                            # --- VERIFIER GATE (STREAM), 2026-07-18 ---
                            # The gated verdict was historically invoked only
                            # from _finalize_and_return — the NON-streaming
                            # finalization — so every streamed answer shipped
                            # unverified, with no log line of any kind (found
                            # when the operator asked why a turn had no
                            # verdict and the USR2 task dump showed none).
                            # The text is already out, so an inline verdict
                            # could never amend it: the verdict ALWAYS runs
                            # late here, regardless of GHOST_CRITIC_ASYNC —
                            # spawn the pure computation and hand it to the
                            # late handler (logs every outcome, scrubs
                            # poisoned lessons, backfills the trajectory).
                            # force_correction=True because a streamed reply
                            # never carries an inline Verifier note — a
                            # high-conf REFUTED queues a banner that the next
                            # stream already prepends via
                            # _take_active_correction. Skipped when the turn
                            # carried no verifiable evidence (toolless chat):
                            # spawning there adds only a no-op task and a
                            # noise line per message. The claim strips
                            # <think> blocks — they streamed to the client
                            # but are not part of the answer, and they would
                            # eat the verifier's 2000-char claim budget.
                            # Every branch is LOUD (log-only, INFO): the
                            # first live deploy produced total silence on
                            # streamed project turns and none of the debug
                            # channels are captured at INFO — a gate whose
                            # skip reasons are invisible cannot be
                            # distinguished from a gate that never ran
                            # (same lesson as _on_done's once-silent
                            # exception path).
                            try:
                                _sv_claim = re.sub(
                                    r'<think>.*?(?:</think>|$)', '',
                                    full_content,
                                    flags=re.DOTALL | re.IGNORECASE,
                                ).strip()
                                _sv_tool = _find_substantive_tool_for_verifier(
                                    stream_tools_snapshot)
                                if getattr(self.context.args,
                                           "no_verifier", False):
                                    pass  # ablation off-switch: silent
                                elif getattr(self.context, "verifier",
                                             None) is None:
                                    pretty_log(
                                        "Verifier",
                                        "stream gate: no verifier attached "
                                        "— skipped",
                                        icon=Icons.BRAIN_THINK)
                                elif not _sv_claim:
                                    pretty_log(
                                        "Verifier",
                                        "stream gate: empty claim after "
                                        "think-strip — skipped",
                                        icon=Icons.BRAIN_THINK)
                                elif _sv_tool is None:
                                    pretty_log(
                                        "Verifier",
                                        "stream gate: no substantive tool "
                                        f"in {len(stream_tools_snapshot)} "
                                        "record(s) — skipped "
                                        "(bookkeeping-only turn)",
                                        icon=Icons.BRAIN_THINK)
                                else:
                                    _sv_task = _glog.spawn_task(
                                        self._compute_verifier_verdict(
                                            tools_run_this_turn=stream_tools_snapshot,
                                            messages=stream_verify_messages,
                                            final_ai_content=_sv_claim,
                                            last_user_content=last_user_content,
                                            lc=lc,
                                        ))
                                    self._attach_late_verdict_handler(
                                        _sv_task, current_trajectory_id,
                                        stream_conv_fp,
                                        force_correction=True)
                                    pretty_log(
                                        "Verifier",
                                        "stream gate: verdict deferred — "
                                        "verifying asynchronously after "
                                        "the stream",
                                        icon=Icons.BRAIN_THINK)
                            except Exception as _svx:
                                pretty_log(
                                    "Verifier",
                                    "stream gate spawn failed: "
                                    f"{type(_svx).__name__}: "
                                    f"{str(_svx)[:160]}",
                                    icon=Icons.WARN, level="WARNING")

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

                        # Keep THIS turn registered — visible in /api/turns and
                        # cancellable — for the WHOLE streamed drain, then
                        # unregister when the client finishes reading. The outer
                        # finally used to unregister the instant handle_chat
                        # returned this generator (before a single token
                        # streamed), so the tail was invisible + uncancellable
                        # and could overlap the next turn (2026-07-15). The
                        # agent_semaphore is deliberately NOT held across the
                        # drain: stream_chat_completion already counts
                        # foreground_tasks for the whole stream (the single LLM
                        # slot isn't stolen), and holding the permit would couple
                        # turn serialization to CLIENT read speed — a stalled
                        # reader would then block every later turn.
                        _stream_owns_unregister = True

                        async def _stream_then_unregister(_gen):
                            try:
                                async for _chunk in _gen:
                                    yield _chunk
                            finally:
                                _turn_reg.unregister(req_id, _active_turn)

                        return (_stream_then_unregister(stream_wrapper()),
                                created_time, req_id)

                    # Ensure msg is always defined in this scope
                    msg = {"role": "assistant", "content": "", "tool_calls": []}
                    thinking_loop_detected = False
                    try:
                        payload["stream"] = True
                        full_content = ""
                        reasoning_content = ""
                        # Last non-null `finish_reason` seen on the stream.
                        # "length" means the upstream hit its token cap and
                        # the answer is truncated — handled after the loop.
                        stream_finish_reason = None

                        # Thinking metrics: surfaced as a single summary line
                        # after the stream completes. We no longer print empty
                        # `=== THINKING ===` frames; in verbose mode the live
                        # tokens still echo to stdout.
                        thinking_started = time.monotonic()
                        thinking_token_count = 0
                        thinking_line_buf = ""
                        next_loop_probe = THINKING_LOOP_PROBE_EVERY
                        # Cadence anchor for the tool-call-collapse probe so it
                        # doesn't run two full-buffer regex scans on EVERY
                        # content chunk (the TOOL_CALL_LOOP_PROBE_EVERY constant
                        # existed but was never consulted).
                        next_tool_probe = TOOL_CALL_LOOP_PROBE_EVERY

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
                            # NOT gated on VERBOSE_MODE (operator request
                            # 2026-07-08): thinking flows through the same
                            # pretty_log pipeline as every other line, so
                            # non-verbose mode shows it truncated to the
                            # standard LOG_TRUNCATE_LIMIT while verbose
                            # still gets the full blocks. The post-stream
                            # summary line is unchanged either way.
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
                                        pretty_log("thinking", block, icon=Icons.BRAIN_THINK, level="DEBUG", no_truncate=True)
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
                                        pretty_log("thinking", block, icon=Icons.BRAIN_THINK, level="DEBUG", no_truncate=True)
                                    continue
                                break

                        def _flush_thinking():
                            nonlocal thinking_line_buf
                            if thinking_line_buf:
                                if thinking_line_buf.strip():
                                    pretty_log("thinking", thinking_line_buf.strip(), icon=Icons.BRAIN_THINK, level="DEBUG", no_truncate=True)
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
                                        _fr = chunk_data["choices"][0].get("finish_reason")
                                        if _fr:
                                            stream_finish_reason = _fr

                                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                                            r_token = delta["reasoning_content"]
                                            reasoning_content += r_token
                                            if not stop_printing:
                                                if _tail_has_stop_marker(reasoning_content, r_token):
                                                    stop_printing = True
                                                if not stop_printing:
                                                    if _is_think_tag_fragment(r_token, reasoning_content):
                                                        pass  # Cosmetic: skip printing fragmented XML tags
                                                    else:
                                                        clean_token = r_token.replace("<think>\n", "").replace("<think>", "")
                                                        _emit_thinking(clean_token)

                                        if "content" in delta and delta["content"] is not None:
                                            text_chunk = delta["content"]
                                            full_content += text_chunk
                                            if not stop_printing:
                                                if _tail_has_stop_marker(full_content, text_chunk):
                                                    stop_printing = True
                                                if not stop_printing and not reasoning_content:
                                                    if (text_chunk.strip().lower() in ("<function", "<parameter")
                                                            or _is_think_tag_fragment(text_chunk, full_content)):
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
                                            if len(full_content) >= next_tool_probe:
                                                next_tool_probe = len(full_content) + TOOL_CALL_LOOP_PROBE_EVERY
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
                                        # base cap (MAX_THINKING_CHARS) is now implicit:
                                        # the periodic loop probe runs at all sizes, so a
                                        # loop is caught before base regardless; only the
                                        # extended hard cap needs an explicit length gate.
                                        extended_cap = override_cap or MAX_THINKING_CHARS_EXTENDED

                                        # Hard cap: past the extended budget, abort
                                        # regardless (a cheap length check).
                                        if len(guard_buf) > extended_cap:
                                            thinking_loop_detected = True
                                            pretty_log("Thinking Cap", f"Stream exceeded extended cap ({extended_cap} chars). Aborting turn.", level="WARNING", icon=Icons.STOP)
                                            break

                                        # The n-gram repetition detector is O(buffer)
                                        # (`buf.count(tail)` over up to 64K chars). The
                                        # old code ran it PER TOKEN once thinking passed
                                        # base_cap (32K) — the dominant CPU cost of a long
                                        # thinking stream — because that boundary branch
                                        # ignored the `next_loop_probe` cadence. Run it
                                        # ONCE per THINKING_LOOP_PROBE_EVERY chars at ALL
                                        # sizes: early loops (below 32K) are still caught
                                        # within 500 chars, and in the 32-64K window a
                                        # clean probe implicitly allows the extension —
                                        # the separate per-token boundary check is gone.
                                        if len(guard_buf) >= next_loop_probe:
                                            next_loop_probe = len(guard_buf) + THINKING_LOOP_PROBE_EVERY
                                            if _detect_thinking_loop(guard_buf):
                                                thinking_loop_detected = True
                                                pretty_log("Thinking Loop", f"Detected n-gram repetition at {len(guard_buf)} chars. Aborting turn.", level="WARNING", icon=Icons.STOP)
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

                        # --- TRUNCATED-ANSWER AUTO-CONTINUATION ---
                        # If the upstream stopped a *text* answer at its
                        # token cap (`finish_reason == "length"`), the partial
                        # reply would be shipped mid-sentence and the verifier
                        # correctly REFUTES it. Continue the generation from
                        # where it stopped — bounded by
                        # MAX_TRUNCATION_CONTINUATIONS — so the model can
                        # finish the thought and answer any explicit question.
                        # Skip when a tool call is in flight (those turns are
                        # handled by the parse/retry path, not user-facing
                        # prose) or when the model emitted no visible content.
                        _truncated_text_turn = (
                            stream_finish_reason == "length"
                            and bool(full_content.strip())
                            and not msg.get("tool_calls")
                            and "<tool_call" not in full_content.lower()
                            and "<function" not in full_content.lower()
                        )
                        _continue_tries = 0
                        while (
                            _truncated_text_turn
                            and _continue_tries < MAX_TRUNCATION_CONTINUATIONS
                        ):
                            _continue_tries += 1
                            pretty_log(
                                "Truncated Output",
                                "Upstream stopped at token cap mid-answer; "
                                f"continuing ({_continue_tries}/{MAX_TRUNCATION_CONTINUATIONS}).",
                                level="WARNING", icon=Icons.WARN,
                            )
                            cont_messages = list(req_messages) + [
                                {"role": "assistant", "content": full_content},
                                {"role": "user", "content": (
                                    "Your previous reply was cut off by a length "
                                    "limit. Continue it from exactly where it "
                                    "stopped — do NOT repeat anything you already "
                                    "wrote, do NOT restate the question, just emit "
                                    "the next characters and finish the answer."
                                )},
                            ]
                            cont_payload = {
                                **payload,
                                "messages": cont_messages,
                                "stream": False,
                            }
                            # Plain-text continuation: never invite a tool call.
                            cont_payload.pop("tools", None)
                            cont_payload.pop("tool_choice", None)
                            cont_payload.pop("parallel_tool_calls", None)
                            stream_finish_reason = None
                            try:
                                cont_result = await self.context.llm_client.chat_completion(cont_payload)
                                cont_choice = (cont_result or {}).get("choices", [{}])[0]
                                cont_text = (cont_choice.get("message", {}) or {}).get("content", "") or ""
                                stream_finish_reason = cont_choice.get("finish_reason")
                                # A continuation may re-open its own <think>
                                # prelude; strip it so only answer prose is
                                # appended to the user-facing content.
                                cont_text = re.sub(
                                    r'<think>.*?(?:</think>|$)', '',
                                    cont_text, flags=re.DOTALL | re.IGNORECASE,
                                )
                                if not cont_text.strip():
                                    break
                                # Bridge with a space only when the seam would
                                # otherwise weld two words together; mid-token
                                # cuts are continued without a gap.
                                if full_content and not full_content[-1].isspace() and not cont_text[:1].isspace():
                                    full_content += cont_text if cont_text[:1] in ",.;:!?)]}\"'" else " " + cont_text
                                else:
                                    full_content += cont_text
                            except Exception as exc:
                                logger.warning("Truncation continuation failed: %s", exc)
                                break
                            _truncated_text_turn = stream_finish_reason == "length"

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
                            # 0.85 (was 0.7): focused iterative work — refining
                            # the same function, debugging the same test — opens
                            # consecutive turns with naturally overlapping (~0.7)
                            # vocabulary. Only near-identical (~0.85+) openings
                            # indicate an actual restart-the-same-derivation loop.
                            if _jac >= 0.85:
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
                        clean_msg_content = _strip_think_blocks(merged_content).strip()
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
                            messages.append({"role": "user", "content": "SYSTEM ALERT: Your previous turn entered a self-repeating thinking loop and was killed. STOP re-deriving the same paragraph. Do NOT resume hypothesizing from memory — a killed loop means your mental model is missing a fact only OBSERVATION can supply. Your next output must be ONE grounding tool call: execute the code, load the page in the browser, or re-read the exact error/output you are reasoning about — then base the next step on what it returns. If a self-generated test assertion disagrees with your function's output, the TEST is likely wrong — re-read the spec and fix the assertion before changing the function. If you have ALREADY proven the task cannot be solved as specified (e.g. the validator has a structural bug), call `abort_attempt` now with a specific reason. Do not write a long <think> block."})
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
                                # Wrap as a <tool_response> user message — the
                                # same translation the main request path applies.
                                # A raw orphan role:"tool" message (no preceding
                                # assistant tool_calls) is rejected by strict
                                # chat templates, turning a recoverable overflow
                                # into a hard failure.
                                _rt_content = str(real_tool.get("content", ""))[:1000] + "\n... [EMERGENCY TRUNCATION] ..."
                                recovery_msgs.append({
                                    "role": "user",
                                    "content": (
                                        f"<tool_response name=\"{real_tool.get('name', 'unknown')}\">\n"
                                        f"{_rt_content}\n</tool_response>"
                                    ),
                                })

                            recovery_msgs.append({"role": "user", "content": "SYSTEM ALERT: The conversation history was truncated to fit within context limits. Continue task. Assume previous context has been handled."})

                            # RETRY ONCE with pruned context. `stream` MUST
                            # be off: the turn loop sets payload["stream"]=True
                            # every iteration, and chat_completion is the
                            # non-streaming API — reusing the flag made the
                            # upstream answer the recovery with SSE frames
                            # that parsed as "non-JSON body" and killed the
                            # turn (2026-07-18, xrick feasibility session).
                            try:
                                payload["messages"] = recovery_msgs
                                payload["stream"] = False
                                messages = recovery_msgs
                                data = await self.context.llm_client.chat_completion(payload, use_coding=has_coding_intent)
                                if "choices" in data and len(data["choices"]) > 0:
                                    msg = data["choices"][0]["message"]
                            except Exception as retry_e:
                                # Surface a calm, actionable message instead of a
                                # raw CRITICAL/traceback. The task state is intact;
                                # the inputs were just too large to read whole.
                                logger.error("Context overflow recovery failed: %s", retry_e)
                                final_ai_content = (
                                    "I hit my context limit while gathering data for this step — "
                                    "the inputs were too large to read all at once, and the automatic "
                                    "recovery didn't complete. Nothing is lost: the task and its files "
                                    "are preserved. Ask me to retry the step and I'll process the large "
                                    "files with a script (summarising them) instead of reading them whole."
                                )
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

                    tool_calls, ui_content, parse_failure_reason = self._parse_assistant_tool_calls(content, msg)

                    ui_content = _strip_think_blocks(ui_content).strip()

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
                    clean_content_for_history = _strip_think_blocks(content).strip()
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
                        # HONESTY NOTE ON DROPPED MUTATIONS (2026-07-14). When
                        # the dropped call would have CHANGED something
                        # (file_system replace at the finish line — observed
                        # live 2026-07-12, twice), silently eating it leaves
                        # the reply implying the action happened. Append an
                        # explicit not-applied note so the user (and the next
                        # turn's context) knows the work is still pending.
                        # Terminal-tool re-calls (self_play etc.) stay silent —
                        # dropping those is the point of this guard.
                        _drop_note = _dropped_mutation_note(dropped)
                        if _drop_note:
                            ui_content = (ui_content or "").rstrip() + _drop_note
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

                        # Catch a PROMISED NOTIFICATION dropped at the finish
                        # line (req 11fe11d8): user said "notify me in slack
                        # when you're done", the model planned the call in
                        # reasoning, then finalized without making it. Steer
                        # ONCE toward notify_operator; never fight a
                        # force-finalising loop-breaker.
                        if (clean_ui and not notify_steer_fired
                                and not force_final_response
                                and not is_final_generation
                                and not force_stop
                                and "notify_operator" not in raw_tools_called
                                and _user_asked_for_notification(last_user_content)):
                            notify_steer_fired = True
                            pretty_log(
                                "Notify Guard",
                                "Turn ending without the notification the user "
                                "explicitly asked for — steering to "
                                "notify_operator (once).",
                                level="WARNING", icon=Icons.WARN,
                            )
                            messages.append(msg)
                            messages.append({
                                "role": "user",
                                "content": (
                                    "SYSTEM ALERT: The user explicitly asked to "
                                    "be NOTIFIED when this task is done, but you "
                                    "are ending the turn without having called "
                                    "`notify_operator`. Call `notify_operator` "
                                    "NOW with one short line summarising the "
                                    "outcome, then give your final response."
                                ),
                            })
                            continue

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

                        # TRAILING-PROMISE GUARD (2026-07-14). The filler
                        # guard above only fires when a TOOL NAME is
                        # mentioned; a mid-repair turn that finalized with
                        # "…That's what's causing the error. Let me fix it."
                        # sailed through (observed live: the reply shipped, the
                        # fix never ran, and the user believed it had). Fire
                        # when the reply's LAST sentence promises imminent
                        # action after a working turn: steer ONCE to either DO
                        # the action now or state plainly that it was NOT
                        # done. `has_run_tools` keeps pure conversation exempt;
                        # "let me know…" is explicitly excluded.
                        if (clean_ui and not pending_promise_steer_fired
                                and not force_final_response
                                and not is_final_generation
                                and not force_stop
                                and has_run_tools):
                            _last_sentence = _ends_with_action_promise(clean_ui)
                            if _last_sentence:
                                pending_promise_steer_fired = True
                                pretty_log(
                                    "Pending-Promise Guard",
                                    f"Final reply ends promising an action "
                                    f"({_last_sentence[:80]!r}) — steering to "
                                    f"act-or-admit (once).",
                                    level="WARNING", icon=Icons.WARN,
                                )
                                messages.append(msg)
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "SYSTEM ALERT: Your reply ENDS by "
                                        f"promising an action ({_last_sentence[:120]!r}). "
                                        "The turn ends when you reply — nothing "
                                        "runs afterwards. Either output the "
                                        "tool_call(s) and DO it NOW, or rewrite "
                                        "your final sentence to state plainly "
                                        "that this was NOT done and what "
                                        "remains for the user to ask for."
                                    ),
                                })
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

                        # --- VERIFIER-GATE AUTO-REPAIR (in-loop re-entry) ---
                        # We're at the normal-success finalisation (model
                        # produced a final answer with no further tool calls).
                        # Verify it HERE so a high-confidence REFUTED verdict
                        # (or finalising on an unverified mutation) can trigger
                        # a bounded repair: inject the critique and `continue`
                        # the turn loop so the agent FIXES the issue instead of
                        # shipping a noted-but-wrong answer. The verdict is
                        # cached for the post-loop gate (no double LLM call on
                        # the clean path). Gated to clean first-pass successes
                        # (`execution_failure_count == 0`, `not force_stop`) so
                        # error/abort answers — which exit via other breaks —
                        # are never "repaired".
                        if (repair_round < self._MAX_VERIFIER_REPAIRS
                                and not force_stop
                                and execution_failure_count == 0):
                            # Async-critic mode skips the BLOCKING verdict (it
                            # stays deferred to the post-loop gate, off the
                            # critical path) — but NOT the unverified-mutation
                            # check, which is a pure predicate over this turn's
                            # tool records (no LLM call, no latency). The
                            # 2026-07-04 chess hunt showed why: with
                            # --critic-nodes on, this whole block was skipped,
                            # so SIX consecutive turns finalised on file writes
                            # that were never run — every crash (module
                            # shadowing, IndexError, NameError, hallucinated
                            # API) shipped to the user, who became the test
                            # harness. Now async mode still forces the bounded
                            # "actually RUN it" re-entry on untested writes.
                            #
                            # Defensive like the post-loop gate: a verifier
                            # error (or a misconfigured/non-async verifier in
                            # tests) must NEVER crash or block finalisation —
                            # on any failure we simply skip the repair and ship.
                            _do_repair = False
                            _directive = ""
                            _crit = ""
                            _refuted = False
                            _unverified = False
                            try:
                                from .verifier import VerifyVerdict as _VV
                                if self._critic_async_enabled():
                                    # The deterministic mutation guard fires
                                    # regardless (LLM-free): an untested write
                                    # always forces the "actually RUN it"
                                    # re-entry.
                                    _lt = _find_substantive_tool_for_verifier(
                                        tools_run_this_turn)
                                    _unverified = _is_unverified_mutation(_lt)
                                    # #18: bounded verdict await at loop-exit.
                                    # The critic runs on the OFF-HOST model, so
                                    # this wait costs the MAIN slot nothing — and
                                    # it lets a REFUTED answer be REPAIRED in-loop
                                    # instead of shipping with only a next-turn
                                    # note (the production gap under
                                    # GHOST_CRITIC_ASYNC=1). Only when the last
                                    # tool is substantive and the mutation guard
                                    # didn't already trip; on timeout we DEFER
                                    # exactly as before (verdict finishes in the
                                    # background, still scrubbing poisoned
                                    # lessons via its done-callback).
                                    _rbudget = self._critic_repair_await_budget()
                                    if (_rbudget > 0 and _lt is not None
                                            and not _unverified):
                                        try:
                                            _vtask = _glog.spawn_task(
                                                self._compute_verifier_verdict(
                                                    tools_run_this_turn=tools_run_this_turn,
                                                    messages=list(messages),
                                                    final_ai_content=final_ai_content,
                                                    last_user_content=last_user_content,
                                                    lc=lc,
                                                ))
                                            _vdone, _ = await asyncio.wait(
                                                {_vtask}, timeout=_rbudget)
                                            if _vtask in _vdone:
                                                _vr, _lt = _vtask.result()
                                                # Cache so the post-loop gate
                                                # reuses it (no double compute
                                                # on the common landed path).
                                                _verifier_verdict_cache = (_vr, _lt)
                                                _verdict_is_fresh = True
                                                _refuted = (
                                                    _vr is not None
                                                    and _vr.verdict == _VV.REFUTED
                                                    and _vr.confidence >= 0.7
                                                )
                                        except Exception as _await_exc:
                                            logger.debug(
                                                "async verdict await skipped: %s: %s",
                                                type(_await_exc).__name__, _await_exc,
                                            )
                                else:
                                    _vr, _lt = await self._compute_verifier_verdict(
                                        tools_run_this_turn=tools_run_this_turn,
                                        messages=messages,
                                        final_ai_content=final_ai_content,
                                        last_user_content=last_user_content,
                                        lc=lc,
                                    )
                                    _verifier_verdict_cache = (_vr, _lt)
                                    _verdict_is_fresh = True
                                    _refuted = (
                                        _vr is not None
                                        and _vr.verdict == _VV.REFUTED
                                        and _vr.confidence >= 0.7
                                    )
                                    # No verdict OR an unconvincing (<0.7)
                                    # CONFIRMED — e.g. one capped because the
                                    # WEB-EXEC probe couldn't run — is not
                                    # good enough to finalise on an untested
                                    # write: force the "actually RUN it"
                                    # re-entry, mirroring the async path's
                                    # pure-predicate behaviour.
                                    _unverified = (
                                        (_vr is None
                                         or (_vr.verdict == _VV.CONFIRMED
                                             and _vr.confidence < 0.7))
                                        and _is_unverified_mutation(_lt)
                                    )
                                if _refuted:
                                    _crit = (
                                        "; ".join(_vr.issues[:3]) if _vr.issues
                                        else (_vr.reasoning
                                              or "the answer was not supported by the evidence")
                                    )
                                    _directive = (
                                        "SYSTEM ALERT — the verifier REFUTED your previous "
                                        f"answer: {_crit}. Do NOT repeat the same claim. "
                                        "Diagnose the underlying problem and FIX it using tools "
                                        "(run / test / inspect the ACTUAL result), then give a "
                                        "corrected final answer grounded in that evidence."
                                    )
                                    _do_repair = True
                                elif _unverified:
                                    _crit = "unverified mutation (untested write)"
                                    _directive = (
                                        "SYSTEM ALERT — you finalised on an UNVERIFIED change: "
                                        "the last action was a file write/replace that was never "
                                        "executed or rendered, so it is unconfirmed. Actually RUN "
                                        "or preview it now (execute it, or screenshot the rendered "
                                        "result) and confirm it works, THEN give your final answer."
                                    )
                                    _do_repair = True
                            except Exception as _rep_exc:
                                logger.debug(
                                    "verifier auto-repair check skipped: %s: %s",
                                    type(_rep_exc).__name__, _rep_exc,
                                )
                                _do_repair = False
                            if _do_repair:
                                _directive += _REPAIR_STANDALONE_SUFFIX
                                messages.append(msg)
                                messages.append({"role": "user", "content": _directive})
                                repair_round += 1
                                force_final_response = False
                                # Discard exactly this turn's text contribution
                                # so the repaired answer replaces (not appends to)
                                # the refuted one.
                                final_ai_content = (final_ai_content or "")[:_final_len_at_turn_start]
                                _verdict_is_fresh = False
                                _verifier_verdict_cache = None
                                pretty_log(
                                    "Verifier Gate",
                                    f"{'REFUTED' if _refuted else 'UNVERIFIED'} → "
                                    f"auto-repair round {repair_round}/"
                                    f"{self._MAX_VERIFIER_REPAIRS}: {_crit[:100]}",
                                    icon=Icons.BRAIN_THINK, level="WARNING",
                                )
                                continue

                        # Internal requests never feed smart memory (same
                        # rationale as the streaming-path gate above).
                        from .autonomous_activity import (
                            is_internal_request as _is_int_req_m2)
                        if self.context.args.smart_memory > 0.0 and last_user_content and not forget_was_called and not last_was_failure and not _is_int_req_m2(req_id):
                            micro_msgs = []
                            for m in [msg for msg in messages if msg.get("role") in ["user", "assistant"]][-4:]:
                                role = m.get("role", "user").upper()
                                clean_content = re.sub(r'```.*?```', '', str(m.get("content", "")), flags=re.DOTALL)
                                micro_msgs.append(f"{role}: {clean_content[:500].strip()}")
                            clean_ai = re.sub(r'```.*?```', '', final_ai_content, flags=re.DOTALL)
                            recent_arc = "\n".join(micro_msgs) + f"\nAI: {clean_ai[:500].strip()}"
                            if getattr(self.context, 'journal', None):
                                await self._journal_append_safe('smart_memory', {'text': recent_arc, 'model': model})
                        break

                    # #5 step 2: the tool guard/dispatch/result pipeline lives in
                    # _dispatch_and_process_tool_batch (verbatim extraction against
                    # TurnState). The try/finally copy-back mirrors the method's own
                    # finally-repack: even if a tool path raises, this frame's locals
                    # match what the inline code would have left behind.
                    _ts = TurnState(
                        _constraint_steer_pending=_constraint_steer_pending,
                        _proj_task_closed_this_req=_proj_task_closed_this_req,
                        _request_sys3_fired_once=_request_sys3_fired_once,
                        _request_sys3_prev_justification=_request_sys3_prev_justification,
                        consecutive_parse_errors=consecutive_parse_errors,
                        current_plan_json=current_plan_json,
                        execution_failure_count=execution_failure_count,
                        final_ai_content=final_ai_content,
                        fname=fname,
                        force_final_response=force_final_response,
                        force_stop=force_stop,
                        forget_was_called=forget_was_called,
                        last_was_failure=last_was_failure,
                        preflight_blocks_this_request=preflight_blocks_this_request,
                        request_sandbox_state=request_sandbox_state,
                        transient_failure_count=transient_failure_count,
                        tool_calls=tool_calls,
                        msg=msg,
                        ui_content=ui_content,
                        parse_failure_reason=parse_failure_reason,
                        model=model,
                        last_user_content=last_user_content,
                        char_budget=char_budget,
                        strikes=strikes,
                        task_tree=task_tree,
                        _user_batch_intent=_user_batch_intent,
                        _request_constraints=_request_constraints,
                        repeated_action_steered=repeated_action_steered,
                        messages=messages,
                        seen_tools=seen_tools,
                        executed_idempotent=executed_idempotent,
                        raw_tools_called=raw_tools_called,
                        tool_usage=tool_usage,
                        tools_run_this_turn=tools_run_this_turn,
                        request_state=request_state,
                    )
                    try:
                        _dispatch_should_break = await self._dispatch_and_process_tool_batch(_ts)
                    finally:
                        _constraint_steer_pending = _ts._constraint_steer_pending
                        _proj_task_closed_this_req = _ts._proj_task_closed_this_req
                        _request_sys3_fired_once = _ts._request_sys3_fired_once
                        _request_sys3_prev_justification = _ts._request_sys3_prev_justification
                        consecutive_parse_errors = _ts.consecutive_parse_errors
                        current_plan_json = _ts.current_plan_json
                        execution_failure_count = _ts.execution_failure_count
                        final_ai_content = _ts.final_ai_content
                        fname = _ts.fname
                        force_final_response = _ts.force_final_response
                        force_stop = _ts.force_stop
                        forget_was_called = _ts.forget_was_called
                        last_was_failure = _ts.last_was_failure
                        preflight_blocks_this_request = _ts.preflight_blocks_this_request
                        request_sandbox_state = _ts.request_sandbox_state
                        transient_failure_count = _ts.transient_failure_count
                    if _dispatch_should_break:
                        break

                else:
                    # Natural exhaustion of the turn budget (2026-07-18): the
                    # loop ended WITHOUT a break, i.e. no deliberate finish —
                    # the request simply ran out of turns mid-work. Observed
                    # live (xrick request 5b9fcc8f): an n-gram thinking kill
                    # landed ON turn 40, its grounding retry had no next
                    # iteration, and the request finalized with trimmed
                    # working narration presented as the answer — which the
                    # verifier then late-refuted. Flag the reply as partial
                    # honestly and point the next request at the recorded
                    # state instead of letting it re-derive everything.
                    pretty_log(
                        "Turn Budget",
                        f"all {effective_max_turns} turns used without a "
                        "deliberate finish — flagging the reply as PARTIAL",
                        level="WARNING", icon=Icons.STOP,
                    )
                    final_ai_content = (
                        f"[TURN BUDGET EXHAUSTED] I used all {effective_max_turns} "
                        "reasoning turns without completing this task — what follows "
                        "is my working state, NOT a finished result. Ask me to "
                        "continue and I will resume from the recorded findings "
                        "(project work log / ledger / notes) instead of starting over.\n\n"
                        + (final_ai_content or "")
                    )

                # #5 step 3: the finalization chain lives in _finalize_and_return
                # (verbatim extraction; FinalizeState is read-only inputs — see the
                # method docstring). Its return IS the old inline return.
                return await self._finalize_and_return(FinalizeState(
                    body=body,
                    created_time=created_time,
                    current_trajectory_id=current_trajectory_id,
                    execution_failure_count=execution_failure_count,
                    final_ai_content=final_ai_content,
                    force_stop=force_stop,
                    forget_was_called=forget_was_called,
                    last_user_content=last_user_content,
                    last_was_failure=last_was_failure,
                    lc=lc,
                    messages=messages,
                    model=model,
                    payload=payload,
                    req_id=req_id,
                    thought_content=thought_content,
                    tools_run_this_turn=tools_run_this_turn,
                    was_complex_task=was_complex_task,
                    _stable_conv_fp=_stable_conv_fp,
                    _verdict_is_fresh=_verdict_is_fresh,
                    _verifier_verdict_cache=_verifier_verdict_cache,
                ))

        except TurnCancelled as _tc:
            # Clean stop, not a crash: return whatever the turn produced
            # before the cancel landed. Exiting through here unwinds the
            # `async with self.agent_semaphore`, so the global turn lock is
            # RELEASED — which is the whole point of cancellation (#22 made
            # one wedged turn block the UI, Slack, and the idle loops alike).
            pretty_log("Turn Cancelled",
                       f"{req_id}: {_tc.reason}", icon=Icons.STOP)
            _partial = ""
            try:
                _partial = str(locals().get("final_ai_content") or "")
            except Exception:  # noqa: BLE001
                _partial = ""
            _note = f"_(Turn cancelled: {_tc.reason}.)_"
            _content = (f"{_partial}\n\n{_note}" if _partial.strip()
                        else f"{_note} No output was produced before the "
                             f"cancellation.")
            return _content, int(datetime.datetime.now().timestamp()), req_id

        finally:
            # Identity-checked: only evict OUR entry (req_id may collide with
            # a concurrent turn's client-supplied id — see TurnRegistry).
            # A streamed turn defers this to its stream wrapper's finally so the
            # tail stays cancellable/visible for the whole drain (2026-07-15);
            # every non-streamed exit path still unregisters here.
            if not _stream_owns_unregister:
                _turn_reg.unregister(req_id, _active_turn)
            if 'messages' in locals(): del messages
            if 'tools_run_this_turn' in locals(): del tools_run_this_turn
            if 'sandbox_state' in locals(): del sandbox_state
            if 'data' in locals(): del data

            pretty_log("Request Finished", special_marker="END")
            request_id_context.reset(token)

    # Banners this agent DETERMINISTICALLY prepends to a reply, all sharing
    # this exact separator and all stacked in FRONT of the answer body:
    #   • async-verdict correction  — "⚠️ **Correction to my previous answer:** …\n\n---\n\n"  (_consume_pending_corrections)
    #   • clarifying-question lead-in — "…things.)\n\n---\n\n{body}"                            (finalize)
    #   • autonomous-progress digest  — "{digest}\n\n---\n\n{body}"                             (finalize)
    _BANNER_SEP = "\n\n---\n\n"
    _BANNER_MAX_BLOCK = 1500   # every real banner is far shorter; a longer
                               # leading block is genuine content before a rule.

    @staticmethod
    def _strip_leading_banners(text: str) -> str:
        """Peel any prepended banner block(s) off the front of a reply so the
        fingerprint keys off the ACTUAL answer body. Each banner is a short
        block terminated by ``_BANNER_SEP`` and stacked ahead of the body, so
        we drop leading separator-terminated blocks that are banner-sized
        (a large leading block before a markdown rule is real content and is
        left intact). This runs symmetrically at stash- and lookup-time: the
        banner-less body is invariant, so a next-turn user-correction still
        matches even though the returned message carried a banner the stashed
        response text did not (which otherwise shifted the hashed prefix and
        silently dropped the 'confidently wrong' calibration signal)."""
        if not isinstance(text, str) or not text:
            return text
        sep = GhostAgent._BANNER_SEP
        body = text
        for _ in range(5):   # correction + clarifying + project digest +
                             # activity digest can co-occur (+1 headroom)
            idx = body.find(sep)
            if 0 <= idx <= GhostAgent._BANNER_MAX_BLOCK:
                body = body[idx + len(sep):]
            else:
                break
        return body

    @staticmethod
    def _response_fingerprint(text: str) -> str:
        """Stable short fingerprint of an assistant response, used as
        the lookup key for correction-detection. Leading banners are
        peeled (see ``_strip_leading_banners``), then the first 500 chars
        of the remaining body are lowercased and whitespace-collapsed and
        md5-hashed. Empty input → empty string (a never-matched key).

        We hash the prefix, not the full response, because Slack /
        the web UI sometimes append small footers (timestamps,
        emoji status) to the assistant message that survive the
        round-trip back into the next request's `messages`. Matching
        on a stable prefix tolerates that without false collisions
        for distinct responses (md5 keeps collision risk negligible
        at this scale). Peeling the leading banners first makes the key
        equally tolerant of the correction/clarifying/digest blocks this
        agent prepends to a reply — the returned text carries them but the
        stashed response text does not."""
        if not isinstance(text, str) or not text:
            return ""
        import hashlib
        core = GhostAgent._strip_leading_banners(text)
        norm = re.sub(r"\s+", " ", core[:500]).strip().lower()
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

    def _run_prm_online_update(self, scorer, traj) -> None:
        """Apply a guarded online PRM step from one promoted-FAILED
        trajectory. Runs in a worker thread (CPU-only). The promoted
        trajectory is the new (negative) sample; a slice of recent
        trajectories is the holdout the update must not regress on. All
        best-effort — never raises into the scheduling task."""
        try:
            from ..prm.trainer import samples_to_xy
            new_X, new_y = samples_to_xy([traj])
            if not new_X:
                return
            holdout_X, holdout_y = [], []
            collector = getattr(self.context, "trajectory_collector", None)
            if collector is not None:
                try:
                    recent = list(collector.iter_trajectories())[-50:]
                    holdout_X, holdout_y = samples_to_xy(recent)
                except Exception:
                    holdout_X, holdout_y = [], []
            updated = scorer.online_update(
                new_X, new_y, holdout_X=holdout_X, holdout_y=holdout_y,
            )
            if updated:
                pretty_log(
                    "PRM Online",
                    f"nudged from user-correction "
                    f"(traj={(traj.id or '')[:8]}, {len(new_X)} step samples)",
                    icon=Icons.BRAIN_AIM,
                )
        except Exception as e:
            logger.debug(
                "PRM online update failed: %s: %s", type(e).__name__, e,
            )

    def _reflected_ids_path(self):
        md = getattr(self.context, "memory_dir", None)
        if md is None:
            return None
        # memory_dir must be a real path. If a test (or a misconfigured
        # context) leaves it a Mock, str(md) is a bogus name like
        # "<MagicMock name='mock.memory_dir' id=…>" and the persist mkdir
        # below would litter that directory into the CWD (the repo root,
        # under the test runner). An isinstance check against os.PathLike is
        # NOT enough — a MagicMock auto-implements __fspath__, so it passes
        # BOTH isinstance(_, os.PathLike) AND os.fspath(_). Restrict to the
        # concrete path types production actually uses (str / pathlib);
        # anything else means "no persistence dir" (2026-07-12).
        from pathlib import Path as _Path, PurePath as _PurePath
        if not isinstance(md, (str, _PurePath)):
            return None
        try:
            return _Path(str(md)) / "reflected_ids.json"
        except Exception:
            return None

    def _get_reflected_ids(self) -> set:
        """The set of already-reflected trajectory ids, loaded ONCE from disk.

        Persisting this across restarts is what lets the Reflector progress
        through the failure backlog instead of re-reflecting the oldest
        failures every boot (wasting idle-LLM calls and, under frequent
        restarts, never reaching recent failures)."""
        already = getattr(self.context, "_reflected_trajectory_ids", None)
        if already is not None:
            return already
        already = set()
        p = self._reflected_ids_path()
        if p is not None:
            try:
                if p.exists():
                    import json as _json
                    data = _json.loads(p.read_text())
                    if isinstance(data, list):
                        already = set(str(x) for x in data)
            except Exception as e:
                logger.debug("reflected-ids load failed: %s", e)
                already = set()
        self.context._reflected_trajectory_ids = already
        return already

    def _persist_reflected_ids(self, cap: int = 10000) -> None:
        """Atomically save the reflected-id set (bounded so it can't grow
        without limit). Best-effort — a persist failure never breaks a turn."""
        already = getattr(self.context, "_reflected_trajectory_ids", None)
        if already is None:
            return
        p = self._reflected_ids_path()
        if p is None:
            return
        try:
            ids = list(already)
            if len(ids) > cap:
                ids = ids[-cap:]
                self.context._reflected_trajectory_ids = set(ids)
            import json as _json
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(ids))
            tmp.replace(p)
        except Exception as e:
            logger.debug("reflected-ids persist failed: %s", e)

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
            # Propagate the same FAILED verdict into the autobio log.
            # Without this, the first-person diary stays stuck at
            # "without a verdict either way" while the trajectory log
            # already knows the user pushed back.
            try:
                from ..selfhood import SelfModel as _SelfModel
                self_model = getattr(ctx, 'self_model', None)
                if isinstance(self_model, _SelfModel) and getattr(self_model, 'enabled', False):
                    self_model.record_outcome(
                        traj.id,
                        Outcome.FAILED.value,
                        failure_reason=verdict.reason or "user-correction",
                    )
            except Exception as e:
                logger.debug(
                    "selfhood verdict backfill skipped: %s: %s",
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

        # Calibration negative (phase 2.5): a user-correction is the
        # strongest "confidently wrong" signal for free-form chat — the
        # user is the cheapest supervisor. If the corrected turn produced
        # a confidence reading (stashed by the turn-end calib block under
        # the same response fingerprint), record (C, 0.0) so the
        # calibration fit learns from overconfident misses that
        # entropy/competence alone tend to rate as clean.
        try:
            _cc = getattr(ctx, "_recent_calib_for_correction", None)
            _ct = getattr(ctx, "calibration_tracker", None)
            if _cc is not None and _ct is not None and fp in _cc:
                _comp = _cc.pop(fp)
                _ct.record(outcome=0.0, **_comp)
        except Exception as _cnx:
            logger.debug("calibration correction-negative skipped: %s", _cnx)

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
                # Retraction blocks (file lock + playbook rewrite + sync
                # Chroma delete). This helper runs ON the event loop, so
                # offload to a thread like the PRM online-update below;
                # inline only when no loop is running (sync/test callers).
                try:
                    asyncio.get_running_loop()
                    # spawn_bg: strong ref (won't be GC'd mid-write) + failure
                    # is logged, not swallowed. Previously a bare unstored
                    # create_task in violation of this file's own GC-safety
                    # rule (agent.py ~2025).
                    _glog.spawn_bg(asyncio.to_thread(
                        skill_memory.retract_lessons_from_trajectory,
                        traj.id,
                        memory_system=vector_memory,
                    ), name="lesson-retract")
                except RuntimeError:
                    skill_memory.retract_lessons_from_trajectory(
                        traj.id,
                        memory_system=vector_memory,
                    )
            except Exception as e:
                logger.debug(
                    "lesson retraction skipped: %s: %s",
                    type(e).__name__, e,
                )

        # Low-latency online PRM update (opt-in via --prm-online-update).
        # A user-correction-promoted FAILED trajectory is a high-signal
        # negative; nudging the PRM now closes the hours-long gap until
        # the next idle batch retrain (phase 2.7). Fire-and-forget in a
        # thread (CPU-only) so the user turn never blocks; the update is
        # guarded inside PRMScorer.online_update (clone + holdout BCE
        # check) so it can only refine, never destabilise, the model.
        try:
            if getattr(getattr(ctx, "args", None), "prm_online_update", False) is True:
                from ..prm.scorer import PRMScorer as _PRMScorer
                scorer = getattr(ctx, "prm_scorer", None)
                if isinstance(scorer, _PRMScorer) and scorer.has_model:
                    asyncio.get_running_loop()
                    # spawn_bg: was a bare unstored create_task — GC-unsafe and
                    # its failure vanished silently.
                    _glog.spawn_bg(asyncio.to_thread(
                        self._run_prm_online_update, scorer, traj,
                    ), name="prm-online-update")
        except Exception as e:
            logger.debug(
                "PRM online update schedule skipped: %s: %s",
                type(e).__name__, e,
            )

        # Schedule single-trajectory reflection. Fire-and-forget; the
        # current user turn must not block on the LLM critique.
        reflector = getattr(ctx, "reflector", None)
        if reflector is None:
            return
        sink = getattr(ctx, "reflection_sink", None) or collector.append
        already = self._get_reflected_ids()

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
            # Persist the dedup set once this fire-and-forget reflection
            # finishes mutating it, so a post-turn reflection isn't lost on a
            # restart before the next watchdog reflection.
            task.add_done_callback(lambda _t: self._persist_reflected_ids())
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

    async def _journal_append_safe(self, kind: str, payload: dict,
                                   timeout: float = 10.0) -> None:
        """Bounded, best-effort journal append.

        The journal write is a sync file op pushed to a thread; without a
        timeout a hung disk parks the event loop awaiting the thread until
        the biological watchdog's opaque "activity timeout" fires. Journal
        entries are an idle-time work queue, never load-bearing for the
        turn — so on timeout/error we log WHICH write was dropped and move
        on instead of stalling the request.
        """
        journal = getattr(self.context, "journal", None)
        if journal is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(journal.append, kind, payload),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pretty_log(
                "Journal Stalled",
                f"append('{kind}') exceeded {timeout:.0f}s — entry dropped "
                "(journal I/O hung?)",
                level="WARNING", icon=Icons.WARN,
            )
        except Exception as exc:
            logger.warning("journal append('%s') failed: %s: %s",
                           kind, type(exc).__name__, exc)

    def _judge_hydration_safe(self, ai_text, turn_id: str = "") -> None:
        """Fire-and-forget post-turn hydration usefulness judge.

        Consumes MemoryBus.last_hydration (set when this turn was hydrated)
        and spawns judge_hydration_usefulness on the worker, off the
        critical path — see core/bus.py for what it credits. No-op when
        the turn wasn't hydrated, and — via the ``turn_id`` stamp — when the
        stash belongs to a DIFFERENT turn (overlapping-turn race, or this
        turn skipped hydration while another's stash was live); consuming a
        foreign stash misattributed usefulness observations (found
        2026-07-15). Defers (bounded) to an in-flight deferred verifier
        verdict so the two never contend for the worker node at the same
        instant — see the stagger comment below. Never raises."""
        bus = getattr(self.context, "memory_bus", None)
        stash = getattr(bus, "last_hydration", None) if bus is not None else None
        if not stash:
            return
        if (turn_id and stash.get("turn_id")
                and stash["turn_id"] != str(turn_id)):
            return  # another turn's stash — its own judge will consume it
        judge = getattr(bus, "judge_hydration_usefulness", None)
        if not callable(judge):
            return
        try:
            from ..utils.logging import spawn_bg

            judge_coro = judge(
                str(ai_text or ""),
                getattr(self.context, "llm_client", None),
                getattr(getattr(self.context, "args", None), "model", "default") or "default",
                turn_id=str(turn_id or ""),
            )
            # Finalize-burst stagger (2026-07-16): this judge and the
            # deferred verifier verdict used to hit the single worker
            # node in the same second, and the loser blew the route
            # timeout (`Nova: ReadTimeout` on effectively every
            # substantive finalize). The verdict is the safety signal,
            # so the judge yields: wait — bounded — for the in-flight
            # verdict, then run. Costs nothing when no verdict is in
            # flight or it already landed; worst case the judge starts
            # _HYDRATION_JUDGE_STAGGER_S late, still far inside its own
            # 600s stash-staleness guard. asyncio.wait never cancels
            # the verdict task and shields this task from its errors.
            verdict_task = getattr(self, "_deferred_verdict_task", None)
            if verdict_task is not None and not verdict_task.done():
                async def _stagger_then_judge(t=verdict_task, jc=judge_coro):
                    try:
                        await asyncio.wait(
                            {t}, timeout=_HYDRATION_JUDGE_STAGGER_S)
                    except Exception:
                        pass
                    return await jc

                judge_coro = _stagger_then_judge()
            spawn_bg(judge_coro, name="hydration-judge")
        except Exception as e:
            logger.debug(f"hydration judge spawn skipped: {e}")

    async def _record_episode_safe(self, user_text, tools, ai_text) -> None:
        """Best-effort episodic-memory write for a completed significant turn.

        EpisodicMemory.record_episode previously had no caller, so the store
        was always empty and the MemoryBus's episodic RAG tier returned
        nothing. This populates it from the turn's tool sequence + outcome.
        Never raises — episodic logging is secondary to the turn.
        """
        em = getattr(self.context, "episodic_memory", None)
        if em is None:
            return
        try:
            actions = []
            for t in (tools or []):
                if not isinstance(t, dict):
                    continue
                content = str(t.get("content", ""))
                actions.append({
                    "tool": t.get("name", "unknown"),
                    "args": {},  # post_mortem entries keep result, not args
                    "result": content,
                    "success": not content.lstrip().startswith(
                        ("Error", "ERROR", "SYSTEM ERROR", "[SYSTEM ERROR]", "Traceback")
                    ),
                })
            ai_head = str(ai_text or "")[:80].lower()
            success = bool(ai_text) and "error" not in ai_head and "failed" not in ai_head
            await asyncio.to_thread(
                em.record_episode,
                trigger=str(user_text or "")[:500],
                context="",
                actions=actions,
                outcome=str(ai_text or "")[:1000],
                success=success,
                # Index the episode into the vector store so the MemoryBus's
                # episodic tier can recall it semantically, not just by
                # substring (feature 1C — previously a dormant ingestion gap).
                vector_memory=getattr(self.context, "memory_system", None),
            )
        except Exception:
            pass

    async def _perfect_it_generate_and_learn(
        self,
        p_payload: dict,
        lesson_label: str,
        trajectory_id: str,
        foreground: bool = False,
    ) -> str:
        """Run the Perfect-It completion and persist the lesson.

        Shared by the inline path (--perfect-it: the text joins the reply,
        so the caller awaits it on the response path — that caller MUST pass
        foreground=True, or the LLM call parks in _wait_for_foreground_clear
        against its own still-active request for up to 600s) and the
        deferred path (flag off: internal learning only, scheduled as a
        background task — default foreground=False yields to live users).
        Returns the generated optimization text ('' when the model produced
        nothing usable). Raises on LLM failure — each caller decides how to
        surface that.

        The lesson is tagged with this turn's trajectory id: if the user
        corrects on the next turn, `retract_lessons_from_trajectory` scrubs
        this exact entry from both the JSON playbook and the vector store.
        Without provenance, the rubber-stamped opt-prot lesson would survive
        the correction and poison future retrieval.
        """
        perfection_data = await self.context.llm_client.chat_completion(
            p_payload, use_worker=True, is_background=not foreground, task_label="perfect-it"
        )
        p_msg = perfection_data["choices"][0]["message"].get("content", "")
        p_msg = re.sub(r'<tool_call>.*?</tool_call>', '', p_msg, flags=re.DOTALL | re.IGNORECASE).strip()

        if p_msg and getattr(self.context, 'skill_memory', None):
            await asyncio.to_thread(
                self.context.skill_memory.learn_lesson,
                task=lesson_label,
                mistake="Sub-optimal pattern identified via Perfection Protocol",
                solution=p_msg,
                memory_system=self.context.memory_system,
                source_trajectory_id=trajectory_id,
                source="perfection_protocol",
            )
            pretty_log("Internal Learning", "Saved optimization strategy to playbook.", icon=Icons.MEM_SAVE)
        return p_msg

    def _record_turn_trajectory(
        self,
        *,
        messages,
        final_content,
        req_id: str,
        model: str,
        trajectory_id: str = "",
        user_request: str = "",
        verifier: Optional[str] = None,
        execution_failed: bool = False,
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
        # The caller passes the HUMAN's request explicitly. Re-deriving it
        # from the message list ("last user-role message wins") is wrong:
        # the turn loop injects synthetic user-role messages mid-turn
        # (SYSTEM ALERT / AUTO-DIAGNOSTIC / Perfect-It nudges), and on any
        # turn with a tool failure those would masquerade as the request —
        # poisoning reflection, frontier clustering, and PRM features.
        user_request = str(user_request or "")
        fallback_user_request = ""
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
                fallback_user_request = str(content or "")
        if not user_request:
            # Legacy fallback (last user-role message) — only correct when
            # the caller couldn't supply the real request.
            user_request = fallback_user_request

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
                    # Populate the STRUCTURED error flag on the chat path too
                    # (previously only self-play/batch set it, so the outcome
                    # heuristics had to regex-sniff result TEXT and missed
                    # atypical shapes — e.g. native-tools corruption). A short
                    # normalized signature is enough for the repeated-error
                    # counter; the full text stays in `result`.
                    try:
                        from ..distill.outcome_heuristics import (
                            _looks_like_tool_error, _normalize_tool_error,
                        )
                        if _looks_like_tool_error(obj.result):
                            obj.error = _normalize_tool_error(obj.result)
                    except Exception:
                        pass

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
        # Stamp the turn's wall-clock from the pretty-log request clock.
        # The writer used to leave duration_s at the schema default (0.0)
        # on every chat turn, so per-turn latency was invisible to the
        # corpus consumers (PRM features, reflection). None (request
        # already closed / sim context) keeps the default.
        try:
            _elapsed = _glog.request_elapsed_s(req_id or "")
            if _elapsed is not None:
                traj_kwargs["duration_s"] = round(_elapsed, 3)
        except Exception:
            pass
        # Hydrated-lesson attribution (counterfactual phase 1,
        # 2026-07-17): stamp WHICH playbook lessons were injected into
        # this turn's prompt (side-channel set by get_playbook_context —
        # turns are globally serialized). When a future counterfactual
        # flags a regression, this is the candidate set; without it,
        # attribution is unrecoverable after the fact.
        try:
            _sm_h = getattr(self.context, "skill_memory", None)
            _trigs = list(getattr(_sm_h, "last_playbook_triggers", []) or [])
            if _trigs:
                traj_kwargs["extra"] = {"hydrated_lessons": _trigs[:10]}
        except Exception:
            pass
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
        # Outcome consolidation (single source of truth): fold the verifier
        # verdict and structural-failure signal into the trajectory outcome so
        # the corpus that feeds the Reflector / PRM / skills matches what
        # calibration + selfhood already act on. Without this a verifier-caught
        # wrong answer stayed UNKNOWN here and never became a lesson.
        try:
            from ..distill.outcome_heuristics import resolve_turn_outcome
            _resolved = resolve_turn_outcome(
                current=traj.outcome, verifier=verifier,
                execution_failed=bool(execution_failed),
            )
            if _resolved != traj.outcome:
                if not traj.failure_reason and _resolved == Outcome.FAILED.value:
                    traj.failure_reason = (
                        "verifier refuted" if verifier == "failed" else "structural failure"
                    )
                traj.outcome = _resolved
        except Exception as e:
            logger.debug("outcome consolidation skipped: %s: %s", type(e).__name__, e)
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
                # Best-effort user handle: pull "root.name" from the
                # profile memory. Anonymous installations or missing
                # profile-memory leave it blank; the autobio writer
                # treats empty as "unknown user" and just omits it
                # from the rendered summary.
                user_handle = ""
                try:
                    pm = getattr(self.context, "profile_memory", None)
                    if pm is not None and hasattr(pm, "load"):
                        prof = pm.load() or {}
                        root = prof.get("root") if isinstance(prof, dict) else None
                        if isinstance(root, dict):
                            user_handle = str(root.get("name") or "")
                except Exception:
                    user_handle = ""
                self_model.capture_turn(
                    trajectory_id=traj.id,
                    user_request=traj.user_request,
                    tool_names=[tc.name for tc in traj.tool_calls],
                    outcome=traj.outcome,
                    final_response=traj.final_response,
                    failure_reason=traj.failure_reason,
                    cluster=traj.cluster,
                    user_handle=user_handle,
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
                    # GROUNDED ELIMINATION: instead of listing un-tested guesses,
                    # run each hypothesis's minimal test in the sandbox and keep
                    # only those consistent with the evidence. This is the change
                    # that turns deep-reason from an ignorable text hint into a
                    # value function grounded in execution.
                    tested = False
                    _sm = getattr(self.context, "sandbox_manager", None)
                    if (hypotheses and _HYPOTHESIS_GROUNDING_ENABLED
                            and _sm is not None):
                        async def _hyp_executor(tool_name, action_str):
                            try:
                                _out, _code = await asyncio.to_thread(
                                    _sm.execute, action_str, 30)
                                return f"[exit {_code}]\n{_out}"
                            except Exception as _ex:
                                return f"executor error: {_ex}"
                        try:
                            hypotheses = await asyncio.wait_for(
                                tester.test_hypotheses_parallel(
                                    hypotheses[:3], _hyp_executor),
                                timeout=120.0,
                            )
                            # LLM adjudication over the real test evidence.
                            await tester.evaluate_results(task_context, hypotheses)
                            tested = True
                        except Exception as _ex:
                            logger.debug(f"Hypothesis grounding skipped: {_ex}")
                    surviving = tester.get_surviving(hypotheses) if tested else []
                    ranked = surviving if surviving else hypotheses
                    if ranked:
                        top = sorted(ranked, key=lambda h: h.confidence, reverse=True)[:3]
                        _label = ("tested in sandbox — survivors" if tested
                                  else "candidate")
                        hypotheses_hint = (
                            f"### ROOT-CAUSE HYPOTHESES ({_label}):\n" + "\n".join(
                                f"- [conf {h.confidence:.0%}] {h.description}"
                                + (f"\n    evidence: {str(h.result)[:200]}"
                                   if tested and getattr(h, "result", "") else "")
                                for h in top
                            ) + "\n\n"
                        )
                        pretty_log(
                            "Deep Reason",
                            f"{'Tested' if tested else 'Generated'} {len(hypotheses)} "
                            f"hypotheses; {len(top)} informing strategy gen",
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
            # Bounded: the pivot is best-effort recovery INSIDE a stuck
            # request — on the default 1200s client budget a busy upstream
            # turned one pivot into a 20-minute hole (request 9c9b75aa:
            # +2886s → +4101s ReadTimeout, dead air). Fail-open beats that.
            gen_data = await self.context.llm_client.chat_completion(
                gen_payload, use_swarm=use_swarm, timeout=120.0)
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
            eval_data = await self.context.llm_client.chat_completion(
                eval_payload, use_swarm=use_swarm, timeout=120.0)
            eval_content = str(eval_data["choices"][0]["message"].get("content") or "")
            result = extract_json_from_text(eval_content)
            pretty_log("System 3 Complete", f"Winning strategy: {result.get('winning_id', '?')} — {result.get('justification', '')[:120]}", icon=Icons.BRAIN_THINK)
            return result
        except Exception as e:
            logger.error(f"System 3 pivot failed: {e}")
            return {}

