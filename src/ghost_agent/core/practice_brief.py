"""§4MT: turn a real owner failure into a practice BRIEF — structured fields
only, never the owner's words.

The §4MS seed hint carried the raw request (up to 400 chars) plus the judge's
failure_reason, which quotes profile facts and Slack ids. A brief here is
built from what the CODE recorded about the turn: the tool families used,
each call's argument KEYS and status, and the code-written failure_reason
PREFIX — mapped to a failure SHAPE (agreement with hand labels: 79% on 75
FAILED owner turns, §4MT practicability lens). The shapes practised:

* ``research_grounding`` — a research answer must state only what the
  fetched pages say (stale/invented facts, results never opened);
* ``tool_output_fidelity`` — a reply must report a tool's output exactly;
* ``honest_failure`` — a step that failed must be reported as failed;
* ``code_data`` — a code / data task (the classic self-play drill).

Other shapes (protocol loops, tone, output format) abstain: there is no
practice format for them yet, and a seed that cannot be practised is not
used. ``leaks_source`` is the gate for any GENERATED challenge text: it must
not share a rare token with the source turn.
"""
from __future__ import annotations

import random
import re
from typing import Optional

WEB_TOOLS = frozenset({"web_search", "browser", "deep_research", "darkweb_search", "darkweb_research",
                       "news_headlines", "fact_check"})
STATE_TOOLS = frozenset({"manage_projects", "manage_services", "introspect", "system_utility", "list_lessons",
                         "self_state", "manage_skills", "knowledge_base", "recall"})
CODE_TOOLS = frozenset({"execute", "file_system", "git"})

SHAPES = {
    "research_grounding": {
        "skill": "Answer a research question using only what the fetched sources actually say.",
        "right": "Open the most authoritative result, prefer the newest dated statement, cite it, and say "
                 "plainly what the sources do not establish — never fill a gap from memory.",
        "grading": "final_response", "fixture": "search_pages",
    },
    "tool_output_fidelity": {
        "skill": "Report a tool's output to the user exactly as the tool returned it.",
        "right": "Every count, id, status and number in the reply matches the tool output; nothing is "
                 "added, dropped or renumbered; a value the tool did not give is said to be unknown.",
        "grading": "final_response", "fixture": "tool_result",
    },
    "honest_failure": {
        "skill": "Report a step that failed as failed, with its error.",
        "right": "Check each step's exit status; a non-zero exit or missing output is reported as a "
                 "failure — never as done.",
        "grading": "final_response", "fixture": "files",
    },
    "code_data": {
        "skill": "Write and run code that computes the requested result from the given files.",
        "right": "Read the inputs, compute the answer at runtime, and verify the output before finishing.",
        "grading": "artifact", "fixture": "files",
    },
}

#: neutral practice domains — the brief never names the owner's topic
_DOMAINS = {
    "research_grounding": ["software releases", "public transport timetables", "museum opening hours",
                           "library catalogue editions", "weather station records"],
    "tool_output_fidelity": ["a project list", "a service status table", "a backup report",
                             "an inventory listing", "a task queue"],
    "honest_failure": ["data export", "build and test run", "file conversion batch",
                       "backup and verify run"],
    "code_data": ["sales records", "sensor readings", "web server logs", "library loans", "train delays"],
}


def _tools(traj) -> list:
    return [str(getattr(tc, "name", "") or (tc.get("name") if isinstance(tc, dict) else "") or "")
            for tc in (getattr(traj, "tool_calls", None) or [])]


def failure_shape(traj, signal: str) -> Optional[str]:
    """The practisable shape of an owner failure, or None (abstain)."""
    reason = str(getattr(traj, "failure_reason", "") or "").lower()
    tools = set(_tools(traj))
    if reason.startswith(("runtime abort", "human negative")):
        return None                                   # an interruption / a reaction: no skill
    if reason.startswith(("tool '", "browser selector")):
        return None                                   # a protocol loop: no practice format yet
    if "claimed-but-missing" in reason:
        return "honest_failure"
    if "word_cap" in reason or "working narration" in reason:
        return None                                   # output format: not built
    if reason.startswith("structural failure"):
        return None if ("system block" in reason or "git:" in reason) else "honest_failure"
    if "too long" in reason or "usable output" in reason:
        return "code_data"
    # a verifier refute / uncertain verdict, a tool-error streak, or an
    # unmapped FAILED reason: the tool family decides
    if tools & WEB_TOOLS:
        return "research_grounding"
    if tools & STATE_TOOLS:
        return "tool_output_fidelity"
    if tools & CODE_TOOLS:
        return "honest_failure" if signal in ("refuted", "uncertain") else "code_data"
    return None


def build_brief(traj, signal: str, rng: Optional[random.Random] = None) -> Optional[dict]:
    """A structured practice brief for an owner failure, or None when its
    shape cannot be practised. Built from code-recorded fields only."""
    shape = failure_shape(traj, signal)
    if shape is None:
        return None
    spec = SHAPES[shape]
    rng = rng or random.Random()
    trace = []
    for tc in (getattr(traj, "tool_calls", None) or [])[:8]:
        args = getattr(tc, "arguments", None) if not isinstance(tc, dict) else tc.get("arguments")
        keys = sorted(args.keys())[:6] if isinstance(args, dict) else []
        failed = False
        try:
            from ..distill.outcome_heuristics import tool_call_failed
            failed = bool(tool_call_failed(tc))
        except Exception:  # noqa: BLE001
            pass
        name = str(getattr(tc, "name", "") or (tc.get("name") if isinstance(tc, dict) else ""))
        trace.append({"tool": name, "arg_keys": keys, "status": "error" if failed else "ok"})
    brief = {"shape": shape, "skill": spec["skill"], "right_behaviour": spec["right"],
             "failure_signal": signal, "grading": spec["grading"], "fixture_kind": spec["fixture"],
             "domain": rng.choice(_DOMAINS[shape]), "tool_trace": trace}
    brief["hint"] = render_hint(brief)
    return brief


def render_hint(brief: dict) -> str:
    """The generator-facing text of a brief: fields only."""
    tools = ", ".join(sorted({t["tool"] for t in brief.get("tool_trace") or [] if t.get("tool")})) or "none"
    return (f"PRACTICE A REAL WEAKNESS (shape: {brief['shape']}). Skill: {brief['skill']} "
            f"Right behaviour: {brief['right_behaviour']} The real turn used: {tools}. "
            f"Set the challenge in this neutral domain: {brief['domain']}. Use invented data only.")


#: a token rare enough that sharing it means copying: numbers of 3+ digits,
#: @mentions, URLs, file names, capitalised words that are not sentence-initial
_RARE_RE = re.compile(r"\d{3,}|@\w+|https?://\S+|\b[\w-]+\.(?:pdf|csv|txt|md|json|py|html?|png|jpe?g|docx?)\b"
                      r"|(?<=[a-z,;:] )[A-Z][a-zA-Z]{3,}")


def _token_hash(tok: str) -> str:
    import hashlib
    return hashlib.sha256(tok.lower().encode("utf-8")).hexdigest()[:16]


def source_token_hashes(traj) -> list:
    """Hashes of the source turn's rare tokens (request + failure reason) —
    what a seed carries so the generated challenge can be gated without the
    owner's text travelling with it."""
    src = " ".join([str(getattr(traj, "user_request", "") or ""), str(getattr(traj, "failure_reason", "") or "")])
    return sorted({_token_hash(m.group(0)) for m in _RARE_RE.finditer(src)})


def leaked_tokens(text: str, hashes) -> list:
    """Rare tokens of ``text`` whose hash is among ``hashes``. Non-empty =
    reject the generated challenge."""
    hs = set(hashes or ())
    if not hs:
        return []
    return sorted({m.group(0) for m in _RARE_RE.finditer(str(text or "")) if _token_hash(m.group(0)) in hs})


def leaks_source(text: str, traj) -> list:
    """Rare tokens a GENERATED challenge shares with the source owner turn
    (its request and failure reason). Non-empty = reject the challenge."""
    return leaked_tokens(text, source_token_hashes(traj))
