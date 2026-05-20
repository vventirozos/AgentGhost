# src/ghost_agent/core/uncertainty.py
"""Metacognitive Monitoring — tracks what the agent knows and doesn't know.

The UncertaintyTracker maintains two lists during a reasoning session:
 - unknowns:    things the agent needs but doesn't have
 - assumptions: things the agent believes but hasn't verified

This lets the agent decide when to ask the user vs. when to proceed,
and attach a risk summary to final responses for transparency.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")


@dataclass
class Unknown:
    what: str           # What the agent doesn't know
    impact: int         # 1-5 scale, how much this affects correctness
    resolution: str     # How to resolve it (e.g., "ask user", "search web", "read file")
    resolved: bool = False
    resolved_value: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "what": self.what,
            "impact": self.impact,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "resolved_value": self.resolved_value,
        }


@dataclass
class Assumption:
    claim: str          # What the agent is assuming
    confidence: float   # 0.0 – 1.0
    basis: str          # Why the agent believes this
    verified: bool = False
    was_correct: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "confidence": self.confidence,
            "basis": self.basis,
            "verified": self.verified,
            "was_correct": self.was_correct,
        }


# Conservative first-person hedge markers. Used to auto-populate the
# tracker from the agent's own output, so uncertainty is load-bearing
# even when the LLM never calls the flag_uncertainty tool explicitly.
_HEDGE_RE = re.compile(
    r"\b(i(?:'m| am) (?:assuming|not sure|not certain|unsure)|i assume\b|"
    r"assuming that|i (?:can(?:no|')?t|could not|couldn't) verify|"
    r"i don'?t have access to|it'?s unclear|i'?m guessing|"
    r"without (?:more|further) (?:info|information|context))",
    re.IGNORECASE,
)


class UncertaintyTracker:
    """Tracks unknowns and assumptions during an agent reasoning session.

    The per-turn in-memory lists are cleared by ``reset()`` between
    turns. When a ``persist_path`` is supplied, every flag is also
    appended to a durable JSONL log — that is what makes recurring
    blind-spots (the same unknown flagged turn after turn) visible
    across sessions via ``recurring_unknowns()``."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.unknowns: List[Unknown] = []
        self.assumptions: List[Assumption] = []
        self.persist_path: Optional[Path] = Path(persist_path) if persist_path else None

    def _append_persist(self, record: dict) -> None:
        """Append one flag record to the durable log. Never raises —
        persistence is secondary to the reasoning turn."""
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("uncertainty persist failed: %s", e)

    def flag_unknown(self, what: str, impact: int = 3,
                     resolution: str = "ask user") -> Unknown:
        """Register something the agent doesn't know but needs."""
        impact = max(1, min(5, impact))
        unknown = Unknown(what=what, impact=impact, resolution=resolution)
        self.unknowns.append(unknown)
        logger.debug("Uncertainty: flagged unknown (impact=%d): %s", impact, what[:100])
        self._append_persist({
            "ts": time.time(), "kind": "unknown", "text": what,
            "impact": impact, "resolution": resolution,
        })
        return unknown

    def resolve_unknown(self, index_or_unknown, value: str) -> bool:
        """Mark an unknown as resolved.

        Accepts EITHER an int index into `self.unknowns` OR the
        `Unknown` object returned from `flag_unknown`. The latter is
        the ergonomic path:

            u = tracker.flag_unknown("timezone?")
            ...
            tracker.resolve_unknown(u, "UTC")

        Before the fix, only the int-index form worked — passing the
        returned object raised `TypeError: '<=' not supported between
        instances of 'int' and 'Unknown'`. Since `flag_unknown` is
        documented to RETURN an `Unknown`, holding the reference and
        passing it back is the natural call shape, so we accept it.

        Returns True if the item was found and marked resolved,
        False otherwise (out-of-range index, or object not in list).
        """
        target: Optional[Unknown] = None
        if isinstance(index_or_unknown, Unknown):
            # Identity check, not equality — multiple unknowns with
            # the same text are legitimately distinct entries.
            for u in self.unknowns:
                if u is index_or_unknown:
                    target = u
                    break
        elif isinstance(index_or_unknown, int):
            if 0 <= index_or_unknown < len(self.unknowns):
                target = self.unknowns[index_or_unknown]
        if target is None:
            return False
        target.resolved = True
        target.resolved_value = value
        return True

    def flag_assumption(self, claim: str, confidence: float = 0.5,
                        basis: str = "") -> Assumption:
        """Register an assumption the agent is making."""
        confidence = max(0.0, min(1.0, confidence))
        assumption = Assumption(claim=claim, confidence=confidence, basis=basis)
        self.assumptions.append(assumption)
        logger.debug("Uncertainty: flagged assumption (conf=%.2f): %s", confidence, claim[:100])
        self._append_persist({
            "ts": time.time(), "kind": "assumption", "text": claim,
            "confidence": confidence, "basis": basis,
        })
        return assumption

    def verify_assumption(self, index_or_assumption, was_correct: bool) -> bool:
        """Mark an assumption as verified.

        Mirrors `resolve_unknown`: accepts either an int index or the
        `Assumption` object returned from `flag_assumption`. See the
        sister method's docstring for the rationale — the API surfaces
        were asymmetric before the fix (`flag_*` returned objects but
        `verify_*`/`resolve_*` only accepted int indices), which made
        the natural "hold a reference, come back to it later" pattern
        raise a TypeError.
        """
        target: Optional[Assumption] = None
        if isinstance(index_or_assumption, Assumption):
            for a in self.assumptions:
                if a is index_or_assumption:
                    target = a
                    break
        elif isinstance(index_or_assumption, int):
            if 0 <= index_or_assumption < len(self.assumptions):
                target = self.assumptions[index_or_assumption]
        if target is None:
            return False
        target.verified = True
        target.was_correct = was_correct
        return True

    def get_critical_unknowns(self, min_impact: int = 4) -> List[Unknown]:
        """Return unresolved unknowns with impact >= min_impact."""
        return [
            u for u in self.unknowns
            if not u.resolved and u.impact >= min_impact
        ]

    def get_unverified_assumptions(self, max_confidence: float = 0.5) -> List[Assumption]:
        """Return unverified assumptions with confidence <= max_confidence."""
        return [
            a for a in self.assumptions
            if not a.verified and a.confidence <= max_confidence
        ]

    def should_ask_user(self) -> Optional[str]:
        """If critical unknowns exist that require user input, formulate a question.
        Returns None if no question is needed."""
        critical = [
            u for u in self.get_critical_unknowns(min_impact=4)
            if u.resolution == "ask user"
        ]
        if not critical:
            return None
        # Format the most impactful unknown as a question
        critical.sort(key=lambda u: u.impact, reverse=True)
        top = critical[0]
        return f"Before I proceed, I need to clarify: {top.what}"

    def get_risk_summary(self) -> str:
        """Generate a risk summary for inclusion in final responses."""
        lines = []
        unresolved = [u for u in self.unknowns if not u.resolved]
        risky_assumptions = [
            a for a in self.assumptions
            if not a.verified and a.confidence < 0.7
        ]

        if not unresolved and not risky_assumptions:
            return ""

        if unresolved:
            lines.append("**Things I'm not certain about:**")
            for u in unresolved[:3]:
                lines.append(f"- {u.what} (impact: {u.impact}/5)")

        if risky_assumptions:
            lines.append("**Assumptions I made:**")
            for a in risky_assumptions[:3]:
                lines.append(f"- {a.claim} (confidence: {a.confidence:.0%})")

        return "\n".join(lines)

    def recurring_unknowns(
        self, *, min_count: int = 2, lookback: int = 400,
    ) -> List[Tuple[str, int]]:
        """Unknowns flagged repeatedly across turns — the durable
        blind-spots. Returns ``[(text, count), ...]`` sorted count-desc.

        Reads the persisted log; empty when persistence is off."""
        if self.persist_path is None or not self.persist_path.exists():
            return []
        counts: Dict[str, int] = {}
        display: Dict[str, str] = {}
        try:
            lines = self.persist_path.read_text(
                encoding="utf-8").splitlines()[-lookback:]
        except OSError:
            return []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "unknown":
                continue
            text = str(rec.get("text") or "").strip()
            if not text:
                continue
            key = text.lower()
            counts[key] = counts.get(key, 0) + 1
            display[key] = text
        out = [(display[k], c) for k, c in counts.items() if c >= min_count]
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out

    def persisted_context(self, *, limit: int = 3) -> str:
        """Prompt block surfacing recurring blind-spots from prior turns,
        so the agent reasons with its own durable uncertainty in view."""
        recurring = self.recurring_unknowns()
        if not recurring:
            return ""
        parts = [
            "### RECURRING UNCERTAINTIES (unresolved across multiple past turns):"
        ]
        for text, count in recurring[:limit]:
            parts.append(
                f"  - {text} (flagged {count}× — resolve this if it is in scope)"
            )
        return "\n".join(parts)

    @staticmethod
    def scan_text_for_uncertainty(text: str, *, limit: int = 3) -> List[str]:
        """Best-effort extraction of explicit first-person hedge sentences
        from agent output. Lets the turn loop auto-populate the tracker
        without depending on the LLM remembering to flag uncertainty."""
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        hits: List[str] = []
        for s in sentences:
            s = s.strip()
            if s and _HEDGE_RE.search(s):
                hits.append(s[:200])
            if len(hits) >= limit:
                break
        return hits

    def reset(self):
        """Clear in-memory turn state for a new reasoning session.
        The durable persisted log (if any) is untouched."""
        self.unknowns.clear()
        self.assumptions.clear()

    def to_dict(self) -> Dict:
        return {
            "unknowns": [u.to_dict() for u in self.unknowns],
            "assumptions": [a.to_dict() for a in self.assumptions],
        }

    def get_context_for_prompt(self) -> str:
        """Format current uncertainty state for injection into LLM prompt."""
        critical = self.get_critical_unknowns(min_impact=3)
        risky = self.get_unverified_assumptions(max_confidence=0.6)
        if not critical and not risky:
            return ""
        parts = ["### CURRENT UNCERTAINTIES:"]
        if critical:
            parts.append("Unresolved unknowns:")
            for u in critical[:5]:
                parts.append(f"  - [impact {u.impact}] {u.what}")
        if risky:
            parts.append("Unverified assumptions:")
            for a in risky[:5]:
                parts.append(f"  - [conf {a.confidence:.0%}] {a.claim}")
        return "\n".join(parts)
