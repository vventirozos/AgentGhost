# src/ghost_agent/core/uncertainty.py
"""Metacognitive Monitoring — tracks what the agent knows and doesn't know.

The UncertaintyTracker maintains two lists during a reasoning session:
 - unknowns:    things the agent needs but doesn't have
 - assumptions: things the agent believes but hasn't verified

This lets the agent decide when to ask the user vs. when to proceed,
and attach a risk summary to final responses for transparency.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


class UncertaintyTracker:
    """Tracks unknowns and assumptions during an agent reasoning session."""

    def __init__(self):
        self.unknowns: List[Unknown] = []
        self.assumptions: List[Assumption] = []

    def flag_unknown(self, what: str, impact: int = 3,
                     resolution: str = "ask user") -> Unknown:
        """Register something the agent doesn't know but needs."""
        impact = max(1, min(5, impact))
        unknown = Unknown(what=what, impact=impact, resolution=resolution)
        self.unknowns.append(unknown)
        logger.debug("Uncertainty: flagged unknown (impact=%d): %s", impact, what[:100])
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

    def reset(self):
        """Clear all tracked state for a new reasoning session."""
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
