"""Evidence-sufficiency gate for the turn loop (§4FD, 2026-09-07).

Why. The largest real failure class in six weeks of live turns (~30 of 124
labelled failures, and most of the human 👎) is a confident answer produced
after every retrieval came back empty or unrelated: `recall` returned rows
about pg_stat when asked for a project codename and the agent invented a
name; `browser extract_text` returned ``LENGTH: 1`` and the agent answered
"around 4 hours"; web search returned nothing usable 54 times and the agent
defined a Greek dessert anyway. arXiv 2606.21409 measures the mechanism —
unreliable retrieval is WORSE than none (44.8 → 4.7 F1 on shuffled
context) — and this agent's model is perfectly capable of abstaining when it
is told the evidence is empty. Nothing told it.

What. ONE pure function reads the turn's tool results and answers "did any
call this turn return substantive evidence?". When every evidence-bearing
call so far came back empty, weak or errored, the loop injects a short steer
into the volatile state block: answer only from what was retrieved, make ONE
more targeted attempt, or say plainly what was not found. It never blocks a
call and never edits a result.

The empty shapes are the REAL strings the tools emit (pinned against them),
not guesses: this module is the single place they are recognised, so a tool
that changes its wording changes it here (`tests/test_evidence_gate.py`
carries the shapes). Rendered under an experiment arm (`evidence_gate`,
TRIGGER_KEYS `evidence_gate_fired`) so the live effect is measurable and the
steer text is stripped from nothing — it lives in the volatile block, not in
a tool result, so the corpus stays what the tool returned (§4K lesson).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Tools whose successful output IS evidence about the world (a retrieval,
#: an observation). Mutations and bookkeeping are neither evidence nor its
#: absence and are ignored by the gate.
EVIDENCE_TOOLS = frozenset({
    "web_search", "darkweb_search", "deep_research", "darkweb_research",
    "recall", "knowledge_base", "browser", "file_system", "execute",
    "fact_check", "news_headlines", "system_utility",
})

# Shapes, one per tool family. Each regex is the tool's own wording.
_BROWSER_LENGTH_RE = re.compile(r"^LENGTH:\s*(\d+)", re.M)
_BROWSER_HTTP_RE = re.compile(r"^HTTP_STATUS:\s*([45]\d\d)", re.M)
# anchored to the tool's OWN header lines: a snippet titled "No results found —
# Kibana" is a result, not an empty search (fresh-eye review §4IY)
_SEARCH_EMPTY_RE = re.compile(
    r"^\s*(?:ERROR:\s*)?(?:No search results found|no results found|SYSTEM OBSERVATION: Zero|"     # the live tool says "ERROR: No search results found."
    r"\(0 results\)|search returned 0 results)", re.I | re.M)
_RECALL_ZERO_RE = re.compile(r"Zero high-confidence memories found", re.I)
_RECALL_WEAK_RE = re.compile(r"best match:\s*LOW", re.I)
_FS_MISSING_RE = re.compile(r"not found\.|does not exist|No such file", re.I)
_EXECUTE_EXIT_RE = re.compile(r"EXIT CODE:\s*(\d+)", re.M)
_ERROR_HEAD_RE = re.compile(r"^\s*(?:SYSTEM )?ERROR\b|^\s*Error:|^\s*CRITICAL ERROR", re.I | re.M)

#: Below this many characters of page text an extract/navigate carries no
#: usable content (a cookie wall, an empty SPA shell, ``LENGTH: 1``).
BROWSER_MIN_TEXT = 40


@dataclass
class EvidenceAssessment:
    consulted: int = 0                  # evidence-bearing calls seen
    empty: List[str] = field(default_factory=list)   # one reason per empty call
    substantive: int = 0                # calls that returned usable evidence

    @property
    def sufficient(self) -> bool:
        """True unless the turn consulted evidence and got none of it."""
        return self.consulted == 0 or self.substantive > 0

    @property
    def fires(self) -> bool:
        return self.consulted > 0 and self.substantive == 0


#: `file_system` operations that OBSERVE the workspace. Anything else is a
#: mutation and is not consulted. An empty operation is treated as a read:
#: the tool rejects it, and the rejection then counts as an error (empty).
FS_READ_OPS = frozenset({"read", "read_chunked", "search", "find", "list_files",
                         "inspect", "read_files", ""})


def _fs_op_is_a_read(args: Dict[str, Any]) -> bool:
    return str((args or {}).get("operation") or "").strip().lower() in FS_READ_OPS


def row_call_facts(t: Dict[str, Any]) -> Tuple[Dict[str, Any], str, bool]:
    """(parsed call arguments, result text, is_error) for one recorded tool
    row — read from the row's REAL shape.

    The loop records a tool row as the API message it sends upstream:
    ``role`` / ``tool_call_id`` / ``name`` / ``content``, where ``content``
    is a ``ToolOutcome`` (a ``str`` subclass carrying ``status`` and the
    call's parsed ``call_args``). No ``arguments`` or ``error`` key ever
    exists on a production row; those spellings are accepted only for
    hand-built rows (tests, replays). The outcome's own status is the
    authority on failure — a REJECTED/FAILED result is an error even when
    its text carries no ``ERROR:`` head; PARTIAL and UNRESOLVED are judged
    by their text like an OK result.
    """
    content_obj = t.get("content")
    args = t.get("arguments") or t.get("args") or getattr(content_obj, "call_args", None) or {}
    if not isinstance(args, dict):
        args = {}
    content = "" if content_obj is None else str(content_obj)
    # FAILED and REJECTED are errors. PARTIAL and UNRESOLVED are NOT: a
    # `fact_check` PARTIAL carries the raw research results ("judge the claim
    # from the results below"), a promoted `execute` job is UNRESOLVED with
    # its `EXIT CODE: 0` output — both are evidence, and both were counted
    # as such by the text shapes before the status was consulted (R3 review
    # of the 2026-09-13 fix: `is_failure` is "not OK", which is wider).
    status = getattr(content_obj, "status", None)
    status_is_error = getattr(status, "value", None) in ("failed", "rejected")
    is_error = bool(t.get("error")) or bool(t.get("is_error")) or status_is_error
    return args, content, is_error


def _empty_reason(name: str, args: Dict[str, Any], content: str,
                  is_error: bool) -> Optional[str]:
    """The reason this call returned no usable evidence, or None if it did.
    Errors count as empty for every tool (an error is not evidence)."""
    text = content or ""
    if is_error:
        return f"{name}: error"
    # Tool-specific shapes first: a search tool's "ERROR: No search results
    # found" is an EMPTY result, and the more specific reason is the useful
    # one. The generic error prefix is the fallback at the end.
    if name in ("web_search", "darkweb_search", "deep_research",
                "darkweb_research", "fact_check", "news_headlines"):
        if _SEARCH_EMPTY_RE.search(text) or not text.strip():
            return f"{name}: no results"
    if name in ("recall", "knowledge_base"):
        if _RECALL_ZERO_RE.search(text) or not text.strip():
            return f"{name}: nothing found"
        if _RECALL_WEAK_RE.search(text):
            return f"{name}: best match LOW (unrelated)"
    if _ERROR_HEAD_RE.search(text[:200]):
        return f"{name}: error"
    if name == "browser":
        m = _BROWSER_HTTP_RE.search(text)
        if m:
            return f"browser: HTTP {m.group(1)}"
        m = _BROWSER_LENGTH_RE.search(text)
        if m and int(m.group(1)) < BROWSER_MIN_TEXT:
            return f"browser: page text length {m.group(1)}"
        return None
    if name in ("web_search", "darkweb_search", "deep_research",
                "darkweb_research", "fact_check", "news_headlines",
                "recall", "knowledge_base"):
        return None          # handled above; anything else is evidence
    if name == "file_system":
        op = str((args or {}).get("operation") or "").strip().lower()
        if _fs_op_is_a_read(args):
            if _FS_MISSING_RE.search(text[:300]) or text.strip() in ("", "[Empty]"):
                return f"file_system {op or 'read'}: nothing found"
        return None
    if name == "execute":
        m = _EXECUTE_EXIT_RE.search(text)
        if m and m.group(1) != "0":
            return f"execute: exit {m.group(1)}"
        return None
    if not text.strip():
        return f"{name}: empty"
    return None


def assess_turn_evidence(tools_run: Optional[Iterable[Dict[str, Any]]]
                         ) -> EvidenceAssessment:
    """Classify every evidence-bearing tool result recorded so far this
    turn. ``tools_run`` rows are the loop's REAL rows (``name``, ``content``
    = a ``ToolOutcome`` carrying ``status`` + ``call_args``) or hand-built
    ones with ``arguments``/``args`` and ``error`` keys — see
    ``row_call_facts``. ``_synthetic`` rows (a rejection the loop minted —
    not the tool's verdict) are ignored."""
    a = EvidenceAssessment()
    for t in tools_run or []:
        if not isinstance(t, dict) or t.get("_synthetic"):
            continue
        name = str(t.get("name") or "").strip().lower()
        if name not in EVIDENCE_TOOLS:
            continue
        args, content, is_error = row_call_facts(t)
        if name == "file_system" and not _fs_op_is_a_read(args):
            # A write/replace/delete is a MUTATION: neither evidence nor its
            # absence (module contract above). Before 2026-09-13 the loop's
            # rows carried no arguments at all, so every write fell into the
            # read branch, its "SUCCESS: Wrote …" counted as substantive, and
            # one scratch-file write silenced the gate for the whole turn.
            continue
        a.consulted += 1
        why = _empty_reason(name, args, content, is_error)
        if why is None:
            a.substantive += 1
        else:
            a.empty.append(why)
    return a


EVIDENCE_STEER_HEADER = "EVIDENCE CHECK (automated):"


def render_evidence_steer(a: EvidenceAssessment, *, max_reasons: int = 4) -> str:
    """The steer text for a turn whose evidence all came back empty, or ""."""
    if not a.fires:
        return ""
    reasons = "; ".join(a.empty[:max_reasons])
    more = f" (+{len(a.empty) - max_reasons} more)" if len(a.empty) > max_reasons else ""
    return (f"{EVIDENCE_STEER_HEADER} every retrieval this turn came back empty or "
            f"unrelated — {reasons}{more}. You have NO evidence for the question yet. "
            "Do NOT state facts, names, numbers or definitions you did not retrieve. "
            "Either make ONE more targeted attempt (a different query, tool or "
            "source), or tell the user plainly what could not be found and stop.")


def steer_for_turn(tools_run: Optional[Iterable[Dict[str, Any]]]
                   ) -> Tuple[str, EvidenceAssessment]:
    """Convenience for the loop: (steer_text_or_empty, assessment)."""
    a = assess_turn_evidence(tools_run)
    return render_evidence_steer(a), a
