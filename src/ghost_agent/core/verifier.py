# src/ghost_agent/core/verifier.py
"""Reflective Self-Evaluation Module.

Provides a Verifier that challenges the agent's own conclusions using
a separate LLM call (ideally on a worker node or at a different temperature)
before returning results to the user.

Two capabilities:
1. verify_claim     — Check whether a stated conclusion is supported by evidence.
2. verify_code_output — Check whether code output actually answers the user's question.
"""

import asyncio
import base64
import inspect
import json
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

# Bounded wall-clock for a single critic-pool verdict call. The critic
# model is deliberately off-host and may be slow (e.g. a 9B on a spare
# box); this cap ensures an unreachable or stalled node falls through to
# the worker/direct fallback instead of blocking the turn. Override with
# GHOST_CRITIC_CALL_TIMEOUT (seconds).
try:
    _CRITIC_CALL_TIMEOUT = float(os.getenv("GHOST_CRITIC_CALL_TIMEOUT", "120") or 120)
except ValueError:
    _CRITIC_CALL_TIMEOUT = 120.0

# The verdict is a tiny JSON object — it does NOT need a reasoning model's
# <think> prelude, and that prelude is the dominant latency on an off-host
# 9B (observed ~37s of pure thinking before the JSON appeared). Disable it
# the way the rest of the codebase does for utility calls
# (project_research.py / dream.py): the `/no_think` soft-switch + the
# `enable_thinking=False` hard-switch, with a small token cap since there
# is no prelude to budget for. Override with GHOST_CRITIC_NO_THINK=0 to
# restore a thinking verdict, GHOST_CRITIC_MAX_TOKENS to tune the cap.
_CRITIC_NO_THINK = os.getenv("GHOST_CRITIC_NO_THINK", "1").strip().lower() not in ("0", "false", "no")
try:
    _CRITIC_MAX_TOKENS = int(os.getenv("GHOST_CRITIC_MAX_TOKENS", "512") or 512)
except ValueError:
    _CRITIC_MAX_TOKENS = 512

# Worker-route budget for a VERIFY verdict. A verify is a judged call over
# the whole turn's claim + evidence — NOT a sub-second routing chore — so it
# must not ride `route()`'s default `_ROUTE_TIMEOUT_S` (12s, sized for query
# expansion). Measured on the live worker (Gemma 4 E4B, 2026-07-16 log): an
# UNCONTENDED verdict takes 7–11s, one whisker under 12s; any contention on
# the node (the finalize burst fires verify + hydration-judge together)
# pushed it past the ceiling → `Nova: ReadTimeout` → the gate shipped a
# hallucinated answer unchecked (req 738c/35, the "Everest pizza" turn).
# 45s absorbs a contended verdict; the loop-exit repair budget (25s) and the
# late-verdict handler already tolerate a verdict that lands late, and a
# genuinely sick node still fails bounded. Override with
# GHOST_VERIFY_WORKER_TIMEOUT (seconds).
try:
    _VERIFY_WORKER_TIMEOUT_S = float(
        os.getenv("GHOST_VERIFY_WORKER_TIMEOUT", "45") or 45)
except ValueError:
    _VERIFY_WORKER_TIMEOUT_S = 45.0

# Hard wall-clock for the LAST-RESORT direct verdict call on the MAIN
# model (the final fallback in `_call_llm`, reached when the critic pool
# and the worker route are both absent or unusable). Without an explicit
# timeout that call rode the shared httpx client's 1200s default — a
# thinking-enabled 2048-token verdict could pin the single foreground
# inference slot for MINUTES in direct contention with a live user
# stream, and the verifier is reachable from BACKGROUND flows too
# (dream/self-play verify shares context.verifier). 90s is deliberately
# roomier than the worker's 45s budget because the main model may spend
# tokens thinking before the JSON, but it is BOUNDED: a stalled call now
# fails into "verdict skipped" (None) instead of starving the user.
# Override with GHOST_VERIFY_FALLBACK_TIMEOUT (seconds).
try:
    _VERIFY_FALLBACK_TIMEOUT_S = float(
        os.getenv("GHOST_VERIFY_FALLBACK_TIMEOUT", "90") or 90)
except ValueError:
    _VERIFY_FALLBACK_TIMEOUT_S = 90.0

# Hard cap per image fed to the visual verifier. The vision node rasterises
# and base64-encodes every image into the prompt; an oversized screenshot
# (or a hostile artifact) would blow the context / OOM the host. Mirrors the
# tools/vision.py MAX_VISION_BYTES guard.
_MAX_VISUAL_BYTES = 16 * 1024 * 1024

# Thinking suppression for the VISUAL verdict call. On a thinking vision
# model the entire max_tokens budget goes to the <think> prelude and the
# content field comes back EMPTY (finish_reason=length) — reproduced live
# 2026-07-31 against Qwen3.6+mmproj: 1024/1024 tokens of reasoning_content,
# 0 chars of content, so every VISUAL check logged "vision returned no
# verdict" and the whole ground-truth gate was silently inert. Same switch
# pair as the critic/stage paths. Override with GHOST_VISUAL_NO_THINK=0.
_VISUAL_NO_THINK = os.getenv("GHOST_VISUAL_NO_THINK", "1").strip().lower() not in ("0", "false", "no")
try:
    _VISUAL_MAX_TOKENS = int(os.getenv("GHOST_VISUAL_MAX_TOKENS", "2048") or 2048)
except ValueError:
    _VISUAL_MAX_TOKENS = 2048
if _VISUAL_MAX_TOKENS <= 0:
    # "0" is a truthy STRING, so it survives the `or 2048` guard — and a
    # zero/negative cap would silence every verdict, recreating the exact
    # dead-gate failure this flag exists to fix.
    _VISUAL_MAX_TOKENS = 2048


def _two_stage_enabled() -> bool:
    """Two-stage claim verification (forced identification → adjudication).

    A yes/no "is this acceptable?" probe is dominated by a default-No/
    default-Yes prior: the judge model often carries the signal that a
    specific fact is unsupported yet never surfaces it because nothing
    forced it to look at that fact ("Mechanisms of Introspective
    Awareness", arXiv:2603.21396 — detection-willingness gates suppress
    latent detection; forced identification bypasses the gate). Stage 1
    therefore FORCES the judge to name the reply's weakest fragments
    without ruling on them; stage 2 adjudicates each named suspect
    against the evidence under the strict rubric, which restores the
    false-positive control that forced enumeration alone would lose.

    Read per call (not at import) so the flag can be flipped without a
    restart — same idiom as llm_recording.recording_enabled(). Kill
    switch: GHOST_VERIFY_TWO_STAGE=0 restores the single-prompt path.
    """
    return os.getenv("GHOST_VERIFY_TWO_STAGE", "1").strip().lower() not in (
        "0", "false", "no")


def _escalate_refute_enabled() -> bool:
    """Re-adjudicate a CHEAP judge's REFUTED verdict on the main model
    before acting on it. Read per call so it can be flipped without a
    restart. Kill switch: GHOST_VERIFY_ESCALATE_REFUTE=0."""
    return os.getenv("GHOST_VERIFY_ESCALATE_REFUTE", "1").strip().lower() \
        not in ("0", "false", "no")


# Suspect hygiene caps: a runaway stage-1 response must not blow the
# stage-2 prompt (which re-embeds claim + evidence + suspects).
_MAX_SUSPECTS = 3
_MAX_SUSPECT_FIELD_CHARS = 300
_SUSPECT_CHECKS = ("alignment", "support", "constraint", "artifact")

# Output budget for each two-stage call. Measured on the live judge
# (Gemma 4 E4B on nova, 2026-07-18, ~15 tok/s): with the default 2048
# budget the model pretty-printed fenced JSON with essay-length reasons —
# 1217 completion tokens / 89s for one enumerate call, which would blow
# the 45s worker-route timeout and dump every verdict onto the foreground
# slot. The stage prompts demand minified single-line JSON with short
# fields; this cap is the hard backstop. A thinking judge that burns the
# whole budget on a <think> prelude parses empty → classic-prompt
# fallback, so the failure mode is a wasted call, never a wrong verdict.
try:
    _STAGE_MAX_TOKENS = int(
        os.getenv("GHOST_VERIFY_STAGE_MAX_TOKENS", "1024") or 1024)
except ValueError:
    _STAGE_MAX_TOKENS = 1024

# Thinking off for the two stage calls (same soft+hard switch as the
# critic path). Measured on the live judge (Gemma 4 E4B heretic, nova,
# 2026-07-18): the adjudicate prompt non-deterministically opened a
# <|channel>thought prelude — 600-1200 tokens / 30-70s for a 60-token
# verdict; with /no_think + enable_thinking=False it answered in ~4s,
# 6/6 valid JSON. Override with GHOST_VERIFY_STAGE_NO_THINK=0 to let a
# judge model think (expect to raise GHOST_VERIFY_STAGE_MAX_TOKENS and
# the worker timeout with it).
_STAGE_NO_THINK = os.getenv(
    "GHOST_VERIFY_STAGE_NO_THINK", "1").strip().lower() not in (
        "0", "false", "no")


class VerifyVerdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class VerifyResult:
    verdict: VerifyVerdict
    confidence: float  # 0.0 – 1.0
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)
    # True when a cheap judge's REFUTED verdict was overturned by the
    # main-model escalation (see verify_claim). Diagnostic only.
    escalated_overturn: bool = False
    # Two-stage path only: the forced-identification suspects that stage 2
    # adjudicated ([{"quote","check","reason"}, ...]). None on the classic
    # single-stage path so downstream dict shapes are unchanged there.
    suspects: Optional[List[Dict[str, str]]] = None
    # §4F Phase 3a: raw score-token expectation from the logit probe
    # (p(acceptable) ∈ [0,1]), None when the probe is off/unavailable.
    # Diagnostic — the blended value lands in `confidence`.
    probe_score: Optional[float] = None

    def passed(self) -> bool:
        return self.verdict == VerifyVerdict.CONFIRMED

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
        }
        if self.suspects is not None:
            d["suspects"] = self.suspects
        return d


# ── Prompts ──────────────────────────────────────────────────────────

_VERIFY_CLAIM_PROMPT = """You are a rigorous auditor. The agent ran a tool and gave the user a CLAIM as its final reply. Decide whether that reply is acceptable.

CLAIM (the agent's reply to the user):
{claim}

EVIDENCE (the tool output(s) the claim was built from — may contain the outputs of SEVERAL tools from the same turn, in chronological order, each prefixed with [tool_name]):
{evidence}

USER REQUEST (what the user actually asked for):
{context}

Check, in order:

1. **Request alignment (highest priority).** Does the CLAIM actually answer the USER REQUEST? If the user asked to do X (e.g. "stop self-play", "delete file foo", "list my notes") and the CLAIM is about something else (a weather report, an unrelated factoid, a different tool's output), this is REFUTED — even if the CLAIM is internally consistent with the EVIDENCE. A CLAIM that is true-but-off-topic is the wrong-question failure mode and must NOT be CONFIRMED.
   - If the USER REQUEST is empty or whitespace, skip this check and proceed to step 2.
2. **Evidence support.** Given that the CLAIM is on-topic, is it actually supported by the EVIDENCE? Flag silent errors (empty output, truncated results, wrong columns, "succeeded" claims when the tool actually failed).
   - Judge the CLAIM against ALL the tool outputs TOGETHER. One tool failing (403/timeout/empty) does NOT refute the parts of the CLAIM that are supported by OTHER tool outputs — refute on lack of support only when NO output supports the disputed part.
   - Specific facts in the CLAIM (names, dates, awards, rankings, prices) that appear in NO tool output are fabrications — REFUTED, no matter how plausible they sound.
   - But DERIVED facts are SUPPORTED — the evidence need not restate them word-for-word. Paraphrase; arithmetic, rounding and unit conversion (49152 bytes → "48 KB"); ordering and superlatives ("latest"/"largest" = the max of what the evidence lists); a classification the evidence itself marks ("19 is Beta" ⇒ the newest STABLE is 18.4); and counts over listed items are all supported. Only a fact with NO basis in any output is a fabrication.
   - You do NOT know today's date and cannot judge whether the evidence is CURRENT. "Not verifiable as the latest right now" / "that date is in the future" / "may be stale" are NEVER grounds for REFUTED — the tool output is a fresh snapshot from this turn.
   - SUBJECTIVE characterizations of data that IS in the evidence are supported, not fabrications: "warm and clear" summarizing 27°C / 0% cloud, "fast" for 12ms, "large" for 3.2GB. A qualitative gloss is REFUTED only when it CONTRADICTS the evidence (calling -5°C "warm"), never merely because the adjective itself does not appear in any tool output.
3. **Constraint satisfaction.** If the user's wording included explicit constraints on the form of the answer ("just the code", "in one sentence", "as JSON", "list only the names"), does the CLAIM satisfy them?

Bookkeeping is not a verdict: the state of any project/task ledger appearing in the EVIDENCE ("all tasks done", "project complete", "nothing left to do") is NEVER by itself grounds for REFUTED. If the USER REQUEST is an operational ask (restart/check/fix/show/run something) and the CLAIM reports doing exactly that with evidence support, it is on-topic and confirmable regardless of what the ledger says about completion. (Live failure this rule pins: user asked to restart a service; the agent restarted it; the judge refuted with "the project is already complete" — wrong.)

A verdict of CONFIRMED requires ALL THREE to hold. If alignment fails, return REFUTED regardless of how well the claim matches the evidence.

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "issues": ["list of specific problems, if any"]
}}"""

# Stage 1 of the two-stage claim path: forced identification. Deliberately
# does NOT ask for a verdict — asking "do you detect a problem?" lets a
# default-No prior swallow real signal; commanding "name the weakest parts"
# extracts it. False positives are expected and fine here: stage 2 exists
# to dismiss them.
_VERIFY_ENUMERATE_PROMPT = """You are auditing an agent's reply. Do NOT decide whether the reply is acceptable overall — that is a later pass. Your ONLY job is forced identification: name the fragments of the reply that are MOST LIKELY to be wrong. Every reply, even a perfect one, has weakest parts; you MUST name EXACTLY 3 of them, and at least one MUST be a specific checkable fact (a number, name, date, price, or event) quoted from the reply — cross-check every such fact against the EVIDENCE word by word before choosing.

CLAIM (the agent's reply to the user):
{claim}

EVIDENCE (the tool output(s) the claim was built from — may contain the outputs of SEVERAL tools from the same turn, in chronological order, each prefixed with [tool_name]):
{evidence}

USER REQUEST (what the user actually asked for):
{context}

For each suspect, quote the exact fragment of the CLAIM (or write "WHOLE REPLY" if the problem is the reply as a whole) and classify which check it might fail:
- "alignment" — the reply answers a different question than the USER REQUEST asked
- "support" — a specific fact (name, date, number, price, ranking, award) appears in NO tool output, or contradicts the tool outputs
- "constraint" — the reply violates an explicit format constraint stated in the USER REQUEST ("just the code", "in one sentence", "as JSON")
- "artifact" — the reply contains machine noise that should never reach a user (error text presented as content, diff/merge markers, template fragments, raw tool syntax)

Order the suspects most-suspicious first. Prefer specific factual fragments (names, numbers, dates) over vague ones.

Be terse: at most 3 suspects, each quote at most 15 words, each reason at most 20 words. Respond ONLY with a MINIFIED single-line JSON object — no code fences, no prose before or after, no extra keys. Your response MUST start with the character {{ and contain no newlines:
{{"suspects": [{{"quote": "exact fragment of the CLAIM", "check": "alignment|support|constraint|artifact", "reason": "why this fragment might fail that check"}}]}}"""

# Stage 2: adjudication. Re-applies the strict single-prompt rubric to each
# named suspect — this is where the false-positive control lives, so its
# dismissal rules must stay at least as strict as _VERIFY_CLAIM_PROMPT's.
_VERIFY_ADJUDICATE_PROMPT = """You are a rigorous auditor delivering a final verdict. The agent ran tool(s) and gave the user the CLAIM below as its final reply. A prior audit pass was FORCED to name the reply's weakest fragments — the SUSPECTS list below. Because naming was forced, suspects exist even for perfect replies: expect many, often all, of them to be false alarms.

CLAIM (the agent's reply to the user):
{claim}

EVIDENCE (the tool output(s) the claim was built from — may contain the outputs of SEVERAL tools from the same turn, in chronological order, each prefixed with [tool_name]):
{evidence}

USER REQUEST (what the user actually asked for):
{context}

SUSPECTS (from the forced identification pass, most-suspicious first):
{suspects}

For EACH suspect, decide against the EVIDENCE whether it is a REAL problem or a FALSE ALARM:
- "support" suspects are REAL only if the fact appears in NO tool output (fabrication) or directly contradicts one. Judge against ALL tool outputs TOGETHER: one tool failing (403/timeout/empty) does NOT make a fact wrong when ANOTHER output supports it.
- DERIVED facts are SUPPORTED — the evidence does NOT have to restate them word-for-word. Before calling a "support" suspect real, ask: can I reach it from the evidence by ordinary reasoning? If yes it is a FALSE ALARM. This covers: paraphrase; arithmetic, rounding and unit conversion (49152 bytes → "48 KB"; 3600s → "1 hour"); ordering and superlatives ("latest"/"newest"/"largest"/"highest" = the max of what the evidence lists); a classification the evidence itself marks ("19 is Beta" ⇒ the newest STABLE is 18.4); and counts or totals over listed items. Only a fact with NO basis in any output — an invented number, version, name or date — is a fabrication.
- You do NOT know today's date and cannot judge whether the evidence is CURRENT. "Not verifiable as the latest right now", "that date is in the future", or "the evidence may be stale" are NEVER grounds for REFUTED: the agent's tool output is by definition a fresh snapshot taken this turn. Judge the claim only against what the EVIDENCE says.
- SUBJECTIVE characterizations of data present in the evidence are FALSE ALARMS: "warm and clear" summarizing 27°C / 0% cloud, "fast" for 12ms, "large" for 3.2GB. A qualitative gloss is REAL only when it CONTRADICTS the evidence (calling -5°C "warm"), never merely because the adjective appears in no tool output. (Live failure this rule pins: a weather reply was refuted for "'warm and clear' not directly supported by the objective data" and had to be overturned on escalation.)
- "alignment" suspects are REAL only if the reply as a whole answers a different question than the USER REQUEST. If the USER REQUEST is empty or whitespace, alignment suspects are automatically FALSE ALARMS. A reply that answers the request and adds extra detail is NOT misaligned.
- "constraint" suspects are REAL only if the USER REQUEST explicitly states that constraint in its own wording.
- "artifact" suspects are REAL only if the quoted noise is actually present in the CLAIM text.
- Suspects that only cite project/task bookkeeping state ("the project is already complete", "all tasks are done", "nothing left to do") are FALSE ALARMS unless the USER REQUEST explicitly asked about completion state — a ledger's state never contradicts an operational reply (restart/check/fix/run) on its own.

The SUSPECTS list is a starting point, not a boundary: if you notice a REAL problem the suspects missed — a fact in the CLAIM that appears in no tool output or contradicts one, machine noise in the reply, a violated explicit constraint — count it as a real problem and name it in "issues".

Then give the overall verdict:
- Any REAL problem → "REFUTED"; list each real problem in "issues".
- Every suspect a FALSE ALARM and no other real problem found, and the reply answers the request with evidence support → "CONFIRMED" with empty "issues".
- You genuinely cannot tell (a load-bearing fact is unjudgeable because the evidence is too truncated or ambiguous) → "UNCERTAIN".
Do NOT refute the CLAIM for weaknesses of the EVIDENCE pipeline itself — tool output that is truncated or noisy but still consistent with the claim is grounds for UNCERTAIN at most, never REFUTED.

Be terse: each "why" and each issue at most 20 words, reasoning at most one short sentence. Fill "checks" FIRST — one entry per suspect, in order, deciding each against the EVIDENCE — before the verdict fields. Respond ONLY with a MINIFIED single-line JSON object — no code fences, no prose before or after, no extra keys. Your response MUST start with the character {{ and contain no newlines:
{{"checks": [{{"suspect": 1, "real": true, "why": "checked against which tool output, found what"}}], "extra_problems": ["REAL problems the suspects missed; empty if none"], "verdict": "CONFIRMED|REFUTED|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "one short sentence", "issues": ["each REAL problem; empty if none"]}}"""

# ── Logit-expectation score probe (§4F Phase 3a) ─────────────────────
# After a two-stage verdict parses, one tiny extra call asks the judge
# for a single acceptability digit 0-9 with top-logprobs; the EXPECTATION
# over the digit distribution is a continuous p(acceptable) that sharpens
# `confidence` (the self-reported value saturates near 1.0 — see the
# bench's mean-conf columns). Verdicts are NEVER changed by the probe;
# only confidence is blended, and any probe failure leaves the result
# exactly as it was. Default OFF until verify_bench shows a win
# (GHOST_VERIFY_LOGIT_EXPECT=1 to enable; read per call so the bench can
# A/B it via env).


def _logit_expect_enabled() -> bool:
    return os.getenv("GHOST_VERIFY_LOGIT_EXPECT", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _logit_expect_weight() -> float:
    """Blend weight for the probe reading (confidence = (1-w)*self +
    w*aligned). The first bench A/B (2026-07-30) showed w=0.5 drags nearly
    every verdict below the 0.7 actionable gate (actionable TPR 0.347 —
    would neuter the verifier); default is the lighter 0.25 pending the
    re-bench. Clamped to [0, 1]; read per call so benches can sweep it."""
    try:
        w = float(os.getenv("GHOST_VERIFY_LOGIT_EXPECT_WEIGHT", "0.25"))
    except ValueError:
        w = 0.25
    return max(0.0, min(1.0, w))


_VERIFY_SCORE_PROBE_PROMPT = """You are auditing an agent's reply. Rate how likely the CLAIM is an acceptable reply to the USER REQUEST given only the EVIDENCE.

CLAIM:
{claim}

EVIDENCE:
{evidence}

USER REQUEST:
{context}

Scale: 0 = certainly unacceptable (fabricated, contradicted, off-topic), 9 = certainly acceptable (supported and responsive).
Respond with ONLY one digit 0-9. No other text."""


def _digit_expectation(top_logprobs: List[Dict[str, Any]]) -> Optional[float]:
    """Expectation over digit tokens in one position's top-logprobs list
    ([{"token": str, "logprob": float}, ...]) → score in [0,1], or None
    when no digit mass is present."""
    import math

    num = 0.0
    den = 0.0
    for entry in top_logprobs or []:
        if not isinstance(entry, dict):
            continue
        tok = str(entry.get("token", "")).strip()
        lp = entry.get("logprob")
        if len(tok) == 1 and tok.isdigit() and lp is not None:
            p = math.exp(float(lp))
            num += int(tok) * p
            den += p
    if den <= 0.0:
        return None
    return max(0.0, min(1.0, (num / den) / 9.0))


# ── GEPA-tunable stage templates (§4F Phase 2) ───────────────────────
# The two-stage prompts above are optimizable text assets. Resolution
# order: _TEMPLATE_OVERRIDES (in-process hook used by the offline
# optimizer while evaluating candidates) → GEPA artifact on disk
# (optim.loader, which also feeds the learning-health activation
# counters) → the baseline constant. A tuned template is accepted ONLY
# if a probe-format with dummy values succeeds — this rejects candidates
# that lost a placeholder OR broke the {{ }} JSON-brace escaping, which
# would otherwise raise inside verify_claim at runtime (the fail-open
# parser class). Rejection falls back to the baseline and logs.

_TEMPLATE_PLACEHOLDERS: Dict[str, Tuple[str, ...]] = {
    "verifier.enumerate": ("claim", "evidence", "context"),
    "verifier.adjudicate": ("claim", "evidence", "context", "suspects"),
}

# Offline-optimizer hook: {"verifier.enumerate": "<template>", ...}.
# Never set on a live agent — candidates go through the loader artifact
# + restart path in production.
_TEMPLATE_OVERRIDES: Dict[str, str] = {}


def _validate_stage_template(name: str, template: str) -> bool:
    """True iff `template` format-probes cleanly with this stage's
    placeholders — catches missing/renamed placeholders, stray unescaped
    braces, and unknown fields in one check."""
    fields = _TEMPLATE_PLACEHOLDERS.get(name, ())
    try:
        template.format(**{f: "x" for f in fields})
        return all(("{%s}" % f) in template for f in fields)
    except Exception:
        return False


def _stage_template(name: str, baseline: str) -> str:
    """Resolve the live template for stage `name` (see block comment)."""
    override = _TEMPLATE_OVERRIDES.get(name)
    if override:
        if _validate_stage_template(name, override):
            return override
        logger.warning(
            "Verifier: override template %s failed placeholder probe — "
            "using baseline", name)
        return baseline
    try:
        from ..optim.loader import tuned_instruction
        tuned = tuned_instruction(name, "")
    except Exception:
        return baseline
    if not tuned:
        return baseline
    if not _validate_stage_template(name, tuned):
        logger.warning(
            "Verifier: tuned template %s failed placeholder probe — "
            "using baseline", name)
        return baseline
    return tuned


_VERIFY_CODE_PROMPT = """You are a code output auditor. Determine whether the agent's RESPONSE actually answers the user's INTENT — including any explicit constraints in the user's wording.

USER INTENT:
{intent}

CODE THE AGENT RAN:
{code}

TOOL OUTPUT:
{output}

AGENT'S RESPONSE TO THE USER:
{response}

Check, in order:

1. **Constraint satisfaction (highest priority).** Does the user's wording include explicit constraints on the form of the answer? Examples: "just give me the code", "in one sentence", "without using X", "list only the names", "as JSON". If yes, does the AGENT'S RESPONSE satisfy those constraints? If the user asked for code and the agent returned a number / prose / a result, that is a REFUTED — the agent answered a different question than the one asked, even if the tool output is internally consistent.
2. Does the response contain the information the user asked for?
3. Are the numbers/results plausible (no obvious off-by-one, wrong units, etc.)?
4. Are there silent errors (empty output, truncated results, wrong columns)?

Common failure shapes to flag:
- User asks for code/snippet/command → agent returns a result or summary instead of the snippet
- User asks for code AND the agent's RESPONSE does not contain a fenced code block — REFUTED regardless of what the tool output says. "The script ran correctly and prints 1 to 10" is NOT a substitute for the script itself; the user cannot paste a confirmation message into their editor. If `intent` contains verbs like give/show/write/draft + nouns like script/code/function/snippet/query/command, the response MUST include the source in a code fence.
  EXCEPTION — the code is the METHOD, not the deliverable: when the user's wording makes a RESULT the thing they want (e.g. "write a script to compute X and tell me the integer", "run code to find the value", "calculate/compute X", "what does this output"), and the RESPONSE states that result correctly, a missing code fence is NOT grounds for REFUTED. "write/run a script" there describes how to get the answer, not a demand to see the source. Only require the code fence when the code itself is the deliverable — the user asked to see/show/give the code with no result requested. When in doubt and the requested result is present and correct, prefer CONFIRMED over REFUTING on a missing fence alone.
- User asks "how do I X" → agent does X and reports the answer instead of explaining the method
- User asks for a specific format → agent ignores the format
- Tool output is a sandbox-internal artefact the user can't actually use

A verdict of CONFIRMED requires BOTH the tool output to be sound AND the response to match what the user asked for. If only the first holds, return REFUTED.

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "issues": ["list of specific problems, if any"]
}}"""

_VERIFY_VISUAL_PROMPT = """You are a meticulous UI auditor. The user reported a VISUAL problem; the agent then acted and gave a RESPONSE. Looking ONLY at the image(s), decide whether the agent's RESPONSE is HONEST about the current rendered state. You are catching FALSE claims of success — not grading whether the work is done.

USER SYMPTOM (the visual problem, in the user's words):
{symptom}

AGENT'S RESPONSE (its claim about the result):
{claim}

IMAGES PROVIDED (in order):
{images_desc}

Judge the agent's RESPONSE against the pixels:
- The RESPONSE accurately describes what is visible — whether it claims the problem is FIXED and the image confirms it, OR it honestly reports the problem is STILL PRESENT and the image confirms that → CONFIRMED. (An honest "it's still broken" is accurate and must NOT be refuted.)
- The RESPONSE MISREPRESENTS the pixels — most importantly, it claims the problem is fixed/resolved while the image still shows it broken; or it claims success while the screenshot is blank/black or stuck on a loading screen; or it claims it's broken when the image is actually fine → REFUTED.
- The image is blank, mid-load, or genuinely ambiguous so you cannot tell → UNCERTAIN.

A stuck loading/"Starting…" screen is NOT a fixed UI. Be conservative: only REFUTE when the image clearly contradicts the response.

START/MENU SCREEN = NOT RUNNING. If the image still shows a start menu, a "Click to Play"/"Start"/"Press to start" button, an instructions modal, or a loading screen, then the app has NOT started — it is showing its MENU, not its running state. A claim that the app/game "works", "renders correctly", is "fully functional", or "is playable" is REFUTED in that case: the agent graded the menu, not the app. The app must be shown actually running (the menu dismissed, the scene/gameplay visible) before any such claim can be CONFIRMED.

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence describing what you actually see vs. what the response claims",
  "issues": ["specific contradictions, if any"]
}}"""

# Claim packing (2026-08-01, req 56221fad post-mortem). The old blunt
# ``claim[:2000]`` cut a 5.7k-char reply mid-sentence: the judge reported
# "truncated at the end" as a defect of the ANSWER, and the confirmation
# lines living in the tail ("✅ Ledger updated …") were invisible — so
# "does not confirm the update" shipped as a 90% refute on a reply that
# confirmed it. Long claims are now packed head+tail around an explicit
# elision marker: openings carry constraint compliance, tails carry
# confirmations/conclusions; the middle is the part a judge can spare.
_CLAIM_LIMIT = 2000
_CLAIM_HEAD = 1200


def pack_claim(text: str, limit: int = _CLAIM_LIMIT,
               head: int = _CLAIM_HEAD) -> str:
    """Fit *text* into *limit* chars keeping its head AND tail, with an
    explicit "[… N chars omitted …]" marker at the seam so the judge knows
    the elision is the packer's, not the reply ending mid-sentence.
    Text already within the limit is returned unchanged (idempotent on
    its own output for any sane limit)."""
    text = text or ""
    if len(text) <= limit:
        return text

    def _marker(n: int) -> str:
        return (f"\n[… ~{n} chars of the reply omitted here — mid-answer "
                f"elision by the audit packer, NOT a truncated response …]\n")

    # Two passes: the true omitted count depends on the marker's own
    # length (digit width), so estimate, then recompute once.
    omitted = len(text) - limit
    for _ in range(2):
        marker = _marker(omitted)
        tail = limit - head - len(marker)
        omitted = len(text) - head - max(tail, 0)
    if tail <= 0:  # degenerate limit — fall back to a plain head cut
        return text[:limit]
    return text[:head] + marker + text[-tail:]


def _bounded_fallback_kwargs(llm_client: Any) -> Dict[str, Any]:
    """Kwargs for the last-resort direct verdict call on the MAIN model.

    Two guards, both applied only when the client's ``chat_completion``
    actually accepts the keyword (the verifier is duck-typed over stubs
    and wrappers whose signatures may be positional-only — passing an
    unknown kwarg there would TypeError into the fallback's broad except
    and silently skip the verdict):

    - ``timeout=_VERIFY_FALLBACK_TIMEOUT_S``: the call must be bounded.
      With no explicit timeout it inherited the shared httpx client's
      1200s default, so an exhausted worker path landed an unbounded
      thinking generation on the single main inference slot.
    - ``is_background=True`` — but ONLY when no user request is live
      (``foreground_requests <= 0``). In that state the verify was
      invoked from a background flow (dream/self-play/idle project
      advance) or a late async verdict, and must queue as background
      instead of inflating ``foreground_tasks`` and making other
      background work misread a live user. When a user request IS live
      we must NOT mark background: the verifier runs from INSIDE the
      user turn (the in-loop auto-repair verdict), and an is_background
      call would park on ``_wait_for_foreground_clear`` waiting for
      THIS request to finish — the same self-stall documented on the
      critic path in ``_call_llm``. A foreground-ambiguous case (user
      live, verify possibly background) therefore stays foreground and
      relies on the bounded timeout.
    """
    fn = getattr(llm_client, "chat_completion", None)
    if fn is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values())
    except (TypeError, ValueError):
        # Signature not introspectable — don't risk a TypeError that
        # would eat the verdict; behave exactly as before the guards.
        return {}

    def _accepts(name: str) -> bool:
        return has_var_kw or name in params

    kwargs: Dict[str, Any] = {}
    if _accepts("timeout"):
        kwargs["timeout"] = _VERIFY_FALLBACK_TIMEOUT_S
    if _accepts("is_background"):
        fg = getattr(llm_client, "foreground_requests", None)
        try:
            if fg is not None and int(fg) <= 0:
                kwargs["is_background"] = True
        except (TypeError, ValueError):
            # Non-numeric counter (mock / exotic wrapper) — assume a
            # user request may be live; stay foreground, never
            # self-park.
            pass
    return kwargs


class Verifier:
    """Self-evaluation module that uses LLM introspection to check the agent's
    own work before presenting it to the user."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def _call_llm(self, prompt: str, temperature: float = 0.1,
                        max_tokens: int = 2048,
                        json_only: bool = False,
                        force_main: bool = False) -> dict:
        """Make a verification LLM call, preferring worker nodes for cost.

        Default token budget is sized for thinking models (Qwen/DeepSeek-R1
        style) that emit a <think>...</think> prelude before the JSON — a
        512 cap was getting consumed entirely by the prelude on the default
        qwen-3.5-27b, so every verifier call came back empty. The two-stage
        claim path passes a tighter budget (_STAGE_MAX_TOKENS) because its
        prompts demand minified JSON and a verbose judge otherwise blows
        the worker-route timeout.
        """
        if not self.llm_client:
            return {}

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_only:
            # Single-line-JSON discipline for the two-stage calls, all
            # three measured necessary on the live judge (2026-07-18):
            # - no-think switch: kills the <|channel>thought prelude
            #   (600-1200 tokens of deliberation for a 60-token verdict);
            # - stop at first newline: a minified answer is one line, so
            #   any newline is either the natural end or a malformed
            #   (fenced/pretty/looping) answer — cut both instantly;
            # - the prompt's "MUST start with {" line keeps the fence
            #   from ever being the first token.
            # NOT response_format=json_object: grammar-constrained
            # sampling made this judge MORE verbose (41 -> 786+ tokens,
            # truncating at the cap). A malformed answer here fails fast
            # (~3 tokens) and falls back to the classic prompt.
            # The stop token travels WITH the no-think switch: with
            # GHOST_VERIFY_STAGE_NO_THINK=0 a thinking judge's reply
            # opens with a think prelude, so stop-at-newline would cut
            # it at the prelude's first line break — both stages parse
            # empty and every verdict silently rides the classic
            # fallback (plus a wasted stage-1 call each time).
            if _STAGE_NO_THINK:
                payload["messages"][0]["content"] = \
                    prompt + "\n\n/no_think"
                payload["chat_template_kwargs"] = {
                    "enable_thinking": False}
                payload["stop"] = ["\n"]

        # Dedicated critic pool takes precedence when configured
        # (--critic-nodes). It keeps the verdict off the foreground
        # inference slot AND off the worker pool, so a slow judge model
        # on a spare box never queues ahead of the fast routing/validation
        # chores the worker pool serves. Falls through to the worker route
        # / direct call below if the pool is absent, offline, or returns
        # an unparseable verdict.
        #
        # NOT is_background: the critic runs on its OWN node, so it never
        # contends for the main upstream's single inference slot — the
        # whole reason is_background exists (park behind the live user
        # request) does not apply. Worse, the verifier is invoked FROM
        # inside a user request (the in-loop auto-repair verdict), so an
        # is_background call would wait on `_wait_for_foreground_clear`
        # for THIS request to finish — a self-deadlock that hangs the turn
        # for the full 600s ceiling. A bounded timeout keeps a stalled or
        # unreachable critic node from blocking the turn: on timeout the
        # call raises and we fall through to the worker/direct path.
        # force_main: skip BOTH cheap pools and adjudicate on the main
        # model. Used by the refute-escalation (see verify_claim): the
        # cheap judge screens, the strong model confirms before a REFUTED
        # verdict is allowed to do damage.
        if getattr(self.llm_client, "critic_clients", None) and not force_main:
            # Build a critic-specific payload: thinking off + a small token
            # cap so the verdict is just the JSON, not a multi-second
            # <think> essay. Kept separate from `payload` so the worker /
            # direct fallbacks below still get the original (thinking)
            # request for whatever model backs them.
            if _CRITIC_NO_THINK:
                critic_payload = {
                    "messages": [
                        {"role": "user", "content": prompt + "\n\n/no_think"}
                    ],
                    "temperature": temperature,
                    "max_tokens": _CRITIC_MAX_TOKENS,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if json_only:
                    critic_payload["stop"] = ["\n"]
            else:
                critic_payload = payload
            try:
                result = await self.llm_client.chat_completion(
                    critic_payload, use_critic=True, timeout=_CRITIC_CALL_TIMEOUT,
                )
                text = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                parsed = self._parse_json(text)
                if parsed:
                    return parsed
            except Exception as exc:
                logger.debug("Verifier critic-pool call failed: %s", exc)

        # Try routing to worker pool first (cheaper, different perspective).
        # `LLMClient.route()` returns the extracted content string, NOT a
        # full chat-completion dict — the previous `isinstance(result, dict)`
        # check was always False, so the worker path was effectively dead
        # and every verify always fell through to the foreground model.
        route_fn = getattr(self.llm_client, "route", None) if not force_main else None
        if route_fn:
            try:
                result = await route_fn(
                    "VERIFY", payload, max_tokens=max_tokens,
                    temperature=temperature, fallback=None,
                    # Verify-sized budget — see _VERIFY_WORKER_TIMEOUT_S.
                    # route()'s 12s default killed contended verdicts.
                    timeout=_VERIFY_WORKER_TIMEOUT_S,
                )
            except Exception as exc:
                logger.debug("Verifier worker route failed: %s", exc)
                result = None
            if result:
                text = result if isinstance(result, str) else (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                parsed = self._parse_json(text)
                if parsed:
                    return parsed
                # Empty/unparseable worker response → fall through to
                # direct call rather than giving up.

        # Last-resort fallback: a direct call on the MAIN model. Bounded
        # and background-aware via _bounded_fallback_kwargs — previously
        # this was a foreground-marked call with NO timeout (1200s httpx
        # default), reachable from background flows, pinning the single
        # main inference slot against a live user stream. The payload
        # itself is deliberately UNTOUCHED here: the two-stage
        # (json_only) payloads already carry the /no_think + stop +
        # tight-cap discipline from the top of this method, while the
        # classic-prompt payload keeps its thinking-sized 2048 budget —
        # the main model is a thinking model and a starved budget came
        # back all-prelude/no-JSON (see the docstring above); the
        # timeout, not the token budget, is the containment.
        try:
            result = await self.llm_client.chat_completion(
                payload, **_bounded_fallback_kwargs(self.llm_client))
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Verifier LLM call failed: %s", exc)
            return {}

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Robustly extract a JSON object from LLM output."""
        if not text:
            return {}
        import re
        # Strip reasoning-model <think>...</think> preludes (closed OR
        # unclosed — budget exhaustion can leave the block open). The
        # greedy regex fallback below would otherwise match braces
        # INSIDE the thinking block instead of the real JSON verdict.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>[\s\S]*$", "", text).strip()
        if not text:
            return {}
        # Try direct parse. Callers do `.get(...)` on the result, so a
        # bare array/string reply must never escape this function — it
        # would raise AttributeError out of verify_claim, and the broad
        # debug-level except in agent.py silently skips the whole pass.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            if (isinstance(parsed, list) and len(parsed) == 1
                    and isinstance(parsed[0], dict)):
                # Salvage a dict the model needlessly wrapped in [].
                return parsed[0]
        except json.JSONDecodeError:
            pass
        # Non-dict top-level values fall through to the fragment walk
        # (its `{...}` candidates can only parse as dicts).
        # Walk every `{...}` block from the end — some models emit a
        # final JSON after prose; the last parseable one wins.
        for candidate in reversed(re.findall(r"\{[\s\S]*?\}", text) or []):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        # Last-resort greedy match (multi-line JSON with nested braces).
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _build_verify_result(self, data: dict) -> Optional[VerifyResult]:
        """Convert a parsed JSON dict into a VerifyResult.

        Returns ``None`` when the verifier LLM produced no usable output
        (worker unavailable, JSON unparseable, upstream error, or a
        parsed dict with no "verdict" key at all). Callers surface that
        as "skipped" rather than conflating it with a real
        low-confidence UNCERTAIN verdict — the two cases are logged
        identically as "UNCERTAIN (0%)" previously, which hid genuine
        failures of the verifier pipeline itself.
        """
        if not data:
            return None
        if "verdict" not in data:
            # A truncated reply's only balanced `{...}` is typically an
            # INNER fragment ({"suspect":2,"real":true,...}) — treating
            # it as a verdict fabricated UNCERTAIN@0.5 and, on the
            # two-stage path, suppressed the classic fallback.
            return None
        # A non-string verdict (null / list) or non-numeric confidence ("high",
        # null) would otherwise raise out of the verifier (callers don't wrap
        # this) — degrade to UNCERTAIN, and CLAMP confidence to [0,1] (the model
        # sometimes emits 95 meaning 95%).
        verdict_str = str(data.get("verdict") or "UNCERTAIN").upper()
        try:
            verdict = VerifyVerdict(verdict_str)
        except ValueError:
            verdict = VerifyVerdict.UNCERTAIN
        try:
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return VerifyResult(
            verdict=verdict,
            confidence=max(0.0, min(1.0, conf)),
            reasoning=data.get("reasoning", ""),
            issues=data.get("issues", []),
        )

    @staticmethod
    def _sanitize_suspects(raw: Any) -> List[Dict[str, str]]:
        """Coerce a stage-1 response's ``suspects`` into a bounded, typed
        list. Anything that isn't a dict with a usable quote/reason is
        dropped; unknown check labels degrade to "support" (the most
        evidence-anchored adjudication rule). Returns [] when nothing
        usable survives — the caller treats that as a stage failure."""
        out: List[Dict[str, str]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if not quote and not reason:
                continue
            check = str(item.get("check") or "").strip().lower()
            if check not in _SUSPECT_CHECKS:
                check = "support"
            out.append({
                "quote": quote[:_MAX_SUSPECT_FIELD_CHARS],
                "check": check,
                "reason": reason[:_MAX_SUSPECT_FIELD_CHARS],
            })
            if len(out) >= _MAX_SUSPECTS:
                break
        return out

    @staticmethod
    def _format_suspects_block(suspects: List[Dict[str, str]]) -> str:
        lines = []
        for i, s in enumerate(suspects, 1):
            lines.append(
                f'{i}. [{s["check"]}] "{s["quote"]}" — {s["reason"]}')
        return "\n".join(lines)

    async def _verdict_score_probe(self, claim: str, evidence: str,
                                   context: str) -> Optional[float]:
        """One bounded score-token call → p(acceptable) ∈ [0,1] via digit
        expectation (§4F Phase 3a). Never raises; None on any failure.
        Always rides a cheap pool (the probe is advisory — it must not
        cost a main-slot round-trip)."""
        if not self.llm_client:
            return None
        try:
            from .entropy import request_logprobs
            payload = {
                "messages": [{"role": "user", "content":
                              _VERIFY_SCORE_PROBE_PROMPT.format(
                                  claim=claim, evidence=evidence,
                                  context=context) + "\n\n/no_think"}],
                "temperature": 0.0,
                "max_tokens": 4,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_logprobs(payload, top_k=10)
            kwargs: Dict[str, Any] = {"timeout": 30.0}
            if getattr(self.llm_client, "critic_clients", None):
                kwargs["use_critic"] = True
            else:
                kwargs["use_worker"] = True
            res = await self.llm_client.chat_completion(payload, **kwargs)
            choice = ((res or {}).get("choices") or [{}])[0]
            content_lp = ((choice.get("logprobs") or {}).get("content")
                          or [])
            for entry in content_lp:
                tok = str((entry or {}).get("token", "")).strip()
                if tok and (tok.isdigit() or entry.get("top_logprobs")):
                    score = _digit_expectation(
                        entry.get("top_logprobs") or [])
                    if score is not None:
                        return score
            # No usable distribution — fall back to the emitted digit
            # (still a probe reading, just not an expectation).
            text = str(choice.get("message", {}).get("content", "") or "")
            for ch in text.strip()[:3]:
                if ch.isdigit():
                    return int(ch) / 9.0
        except Exception as exc:
            logger.debug("Verifier score probe failed: %s", exc)
        return None

    async def _verify_claim_two_stage(self, claim: str, evidence: str,
                                      context: str,
                                      force_main: bool = False
                                      ) -> Optional[VerifyResult]:
        """Forced identification (stage 1) → adjudication (stage 2).

        Returns ``None`` whenever either stage yields nothing usable, so
        ``verify_claim`` can fall back to the classic single-prompt path —
        the two-stage pipeline must never make the verifier LESS available
        than it was before.
        """
        enum_prompt = _stage_template(
            "verifier.enumerate", _VERIFY_ENUMERATE_PROMPT).format(
            claim=claim, evidence=evidence, context=context)
        stage1 = await self._call_llm(enum_prompt, temperature=0.1, force_main=force_main,
                                      max_tokens=_STAGE_MAX_TOKENS,
                                      json_only=True)
        suspects = self._sanitize_suspects((stage1 or {}).get("suspects"))
        if not suspects:
            # Parse failure OR an empty enumeration despite the forced-pick
            # instruction — either way there is nothing to adjudicate.
            logger.debug("Verifier two-stage: no usable suspects, "
                         "falling back to single-stage")
            return None

        adj_prompt = _stage_template(
            "verifier.adjudicate", _VERIFY_ADJUDICATE_PROMPT).format(
            claim=claim, evidence=evidence, context=context,
            suspects=self._format_suspects_block(suspects))
        stage2 = await self._call_llm(adj_prompt, temperature=0.1, force_main=force_main,
                                      max_tokens=_STAGE_MAX_TOKENS,
                                      json_only=True)
        result = self._build_verify_result(stage2)
        if result is None:
            logger.debug("Verifier two-stage: adjudication unparseable, "
                         "falling back to single-stage")
            return None
        result.suspects = suspects

        # §4F Phase 3a: sharpen confidence with the score-token probe.
        # The verdict itself is untouched; UNCERTAIN is excluded (the
        # acceptability scale doesn't map onto "cannot judge").
        if (_logit_expect_enabled()
                and result.verdict in (VerifyVerdict.CONFIRMED,
                                       VerifyVerdict.REFUTED)):
            probe = await self._verdict_score_probe(claim, evidence, context)
            if probe is not None:
                result.probe_score = round(probe, 3)
                aligned = (probe if result.verdict == VerifyVerdict.CONFIRMED
                           else 1.0 - probe)
                w = _logit_expect_weight()
                result.confidence = round(
                    (1.0 - w) * float(result.confidence) + w * aligned, 3)
        return result

    async def verify_claim(self, claim: str, evidence: str,
                           context: str = "") -> Optional[VerifyResult]:
        """Check whether *claim* is supported by *evidence*.

        Default path (GHOST_VERIFY_TWO_STAGE, on unless =0) is two LLM
        calls: forced identification of the reply's weakest fragments,
        then per-suspect adjudication against the evidence. Falls back to
        the classic single-prompt verdict when either stage fails, so the
        worst case matches the old behavior (plus one bounded extra call).

        REFUTE ESCALATION (2026-07-26): a REFUTED verdict from a CHEAP
        judge is re-adjudicated on the MAIN model before it is allowed to
        do damage. The cheap route is a small worker model (live: Gemma 4
        E4B) and it false-refutes DERIVED facts — measured on the real
        "latest PostgreSQL version" turn it refuted a correct answer at
        90%, and called 49152 bytes "not exactly 48 KB"; the main 35B got
        the same three cases right 3/3. A false REFUTE is expensive
        (scrubs the turn's lessons, files follow-up tasks, shows the user
        a correction, marks the trajectory failed → poisons the corpus
        every downstream learner trains on), while the escalation costs
        one main-model call on the RARE refute path only. Screen cheap,
        confirm expensive — the same gate discipline used elsewhere here.
        Kill switch: GHOST_VERIFY_ESCALATE_REFUTE=0.
        """
        # Head+tail packing, not a blunt cut — see pack_claim's rationale.
        claim_t = pack_claim(claim)
        evidence_t = evidence[:4000]
        context_t = context[:1000]
        result = None
        if _two_stage_enabled():
            result = await self._verify_claim_two_stage(
                claim_t, evidence_t, context_t)
        if result is None:
            prompt = _VERIFY_CLAIM_PROMPT.format(
                claim=claim_t,
                evidence=evidence_t,
                context=context_t,
            )
            data = await self._call_llm(prompt, temperature=0.1)
            result = self._build_verify_result(data)
        return await self._escalate_refute(result, claim_t, evidence_t, context_t)

    async def _escalate_refute(self, result: Optional[VerifyResult],
                               claim: str, evidence: str,
                               context: str) -> Optional[VerifyResult]:
        """Confirm a REFUTED verdict on the MAIN model before returning it.

        No-op unless the verdict is REFUTED, escalation is enabled, and a
        cheap route (critic pool / worker) was actually available to
        produce it — when the main model already IS the judge there is
        nothing to escalate to. On disagreement the main model's verdict
        wins (it is the stronger judge); on any error the original
        verdict stands, so escalation can only ever reduce false refutes,
        never make the gate less available."""
        if result is None or result.verdict != VerifyVerdict.REFUTED:
            return result
        if not _escalate_refute_enabled():
            return result
        client = self.llm_client
        cheap_route = bool(getattr(client, "critic_clients", None)) or bool(
            getattr(client, "worker_clients", None))
        if not cheap_route:
            return result  # main model already judged it
        try:
            strong = None
            if _two_stage_enabled():
                strong = await self._verify_claim_two_stage(
                    claim, evidence, context, force_main=True)
            if strong is None:
                prompt = _VERIFY_CLAIM_PROMPT.format(
                    claim=claim, evidence=evidence, context=context)
                data = await self._call_llm(prompt, temperature=0.1,
                                            force_main=True)
                strong = self._build_verify_result(data)
        except Exception as exc:
            logger.debug("Verifier refute-escalation failed (keeping "
                         "original verdict): %s", exc)
            return result
        if strong is None:
            return result
        if strong.verdict == VerifyVerdict.REFUTED:
            logger.info("Verifier escalation: main model CONFIRMED the "
                        "refute — verdict stands.")
            return strong
        logger.warning(
            "Verifier escalation OVERTURNED a cheap-judge refute: main "
            "model says %s. Original issues: %s",
            strong.verdict.value, "; ".join(result.issues or [])[:160])
        strong.escalated_overturn = True
        return strong

    async def verify_code_output(self, code: str, output: str,
                                 intent: str,
                                 *, response: str = "") -> Optional[VerifyResult]:
        """Check whether the agent's *response* actually answers
        *intent*, given the *code* it ran and the *output* it
        observed.

        ``response`` is the agent's user-facing reply. Defaults to
        empty for back-compat with older callers, but production
        callers should always pass it — without it, the verifier
        falls back to "does the output match the claim" auditing
        which can't catch wrong-question answers (user asks for
        code, agent gives a number; user asks for format X, agent
        replies in format Y). Those failure shapes are the dominant
        wrong-but-confidently-confirmed mode in practice.
        """
        prompt = _VERIFY_CODE_PROMPT.format(
            intent=intent[:1000],
            code=code[:4000],
            output=output[:4000],
            response=(response or "(response not provided to verifier)")[:4000],
        )
        data = await self._call_llm(prompt, temperature=0.1)
        return self._build_verify_result(data)

    async def _call_llm_vision(self, prompt: str, image_paths: List[str],
                               temperature: float = 0.1) -> dict:
        """Vision-enabled verification call. Loads each image off disk,
        base64-embeds it, and asks the vision-capable model for a verdict.

        Distinct from ``_call_llm`` in two ways: (1) it does NOT route to
        the text VERIFY worker pool (that pool isn't multimodal) — it goes
        straight to the main client's ``chat_completion(..., use_vision=True)``
        path, which the client routes to the vision node; (2) it carries
        images. Returns {} on any failure so the caller surfaces a skipped
        verdict rather than a false REFUTED."""
        if not self.llm_client or not image_paths:
            return {}

        if _VISUAL_NO_THINK:
            prompt = prompt + "\n\n/no_think"
        content_array: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        loaded = 0
        for pth in image_paths:
            if not pth:
                continue
            try:
                p = Path(pth)
                if not p.exists():
                    continue
                if p.stat().st_size > _MAX_VISUAL_BYTES:
                    logger.debug("visual verify: skipping oversized image %s", pth)
                    continue
                data_bytes = await asyncio.to_thread(p.read_bytes)
            except Exception as exc:
                logger.debug("visual verify: could not read %s: %s", pth, exc)
                continue
            mime, _ = mimetypes.guess_type(str(pth))
            mime = mime or "image/png"
            b64 = base64.b64encode(data_bytes).decode("utf-8")
            content_array.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
            loaded += 1

        if loaded == 0:  # nothing renderable to judge
            return {}

        payload = {
            "messages": [
                {"role": "system", "content": "You are a meticulous UI auditor. Judge only what is visible in the images."},
                {"role": "user", "content": content_array},
            ],
            "temperature": temperature,
            # _VISUAL_MAX_TOKENS, not a hardcoded 1024: with no-think off
            # (or a backend that ignores enable_thinking) the old 1024 cap
            # was consumed entirely by the <think> prelude and the verdict
            # never appeared — see _VISUAL_NO_THINK above.
            "max_tokens": _VISUAL_MAX_TOKENS,
            "stream": False,
        }
        if _VISUAL_NO_THINK:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            result = await self.llm_client.chat_completion(payload, use_vision=True)
            text = (
                (result or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Visual verifier call failed: %s", exc)
            return {}

    async def verify_visual(self, *, symptom: str, claim: str,
                            after_image: str,
                            before_image: Optional[str] = None
                            ) -> Optional[VerifyResult]:
        """Check whether a reported VISUAL symptom is still present in the
        rendered artifact, by looking at the actual pixels.

        ``after_image`` is the current rendered state (a screenshot taken
        AFTER the agent's change). ``before_image`` is the user's original
        screenshot showing the problem, if available — passing both lets the
        model do a before/after comparison, which is far more reliable than
        judging a fresh frame cold.

        Returns ``None`` when nothing could be rendered/loaded — the caller
        treats that as *skipped* and applies NO penalty, so the agent is
        never punished for infra it can't control (no browser, headless run).
        """
        if not after_image:
            return None
        if before_image:
            images = [before_image, after_image]
            images_desc = (
                "[1] the user's ORIGINAL screenshot showing the problem.\n"
                "[2] the CURRENT rendered state after the agent's change."
            )
        else:
            images = [after_image]
            images_desc = (
                "[1] the CURRENT rendered state after the agent's change "
                "(the user's original screenshot was not available)."
            )
        prompt = _VERIFY_VISUAL_PROMPT.format(
            symptom=(symptom or "")[:1000],
            claim=(claim or "")[:1500],
            images_desc=images_desc,
        )
        data = await self._call_llm_vision(prompt, images, temperature=0.1)
        return self._build_verify_result(data)
