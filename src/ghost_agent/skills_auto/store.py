"""Graduated-skill store — the persistent home for auto-acquired skills.

The extractor/consolidator pipeline mined recurring validated tool
sequences, but phase 2.6 used to log the consolidated candidates and
then throw them away — extraction was pure overhead with no output.

This store closes that loop (proposal item #9). A consolidated
candidate that clears verification is *graduated*: persisted here as a
"proven approach". The turn loop reads the store back and surfaces the
relevant graduated skills into the system prompt, so a tool sequence
the agent discovered works gets reused on the next similar request.

Storage: a single JSON file ``auto_skills.json`` under the memory
directory. Bounded (lowest-confidence entries drop on overflow),
human-diffable, atomic write via ``.tmp`` + ``replace``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GhostAgent")

STORE_FILENAME = "auto_skills.json"
MAX_SKILLS = 60


def _tokens(text: str) -> set:
    """Word tokens longer than two characters — short filler words
    ("a", "of", "to") would otherwise create spurious keyword overlap."""
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


class GraduatedSkillStore:
    """Persistent registry of verified, auto-acquired tool-sequence skills."""

    def __init__(self, memory_dir: Path):
        self.path = Path(memory_dir) / STORE_FILENAME
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("auto-skill store read failed (%s); starting empty", e)
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception as e:
            logger.warning("auto-skill store write failed: %s", e)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def graduate(self, candidate, *, confidence: Optional[float] = None) -> dict:
        """Persist a verified skill candidate as a graduated skill.

        Idempotent on ``signature_hash`` — re-graduating an existing
        skill bumps its support / confidence / verification count rather
        than duplicating it. Returns the stored entry."""
        sig = getattr(candidate, "signature_hash", "") or getattr(candidate, "name", "")
        now = datetime.utcnow().isoformat() + "Z"
        conf = float(confidence if confidence is not None
                     else getattr(candidate, "confidence", 0.0))
        with self._lock:
            data = self._load()
            existing = data.get(sig)
            if existing:
                existing["support"] = max(
                    int(existing.get("support", 0)),
                    int(getattr(candidate, "support", 0)),
                )
                existing["confidence"] = round(max(existing.get("confidence", 0.0), conf), 4)
                existing["verifications"] = int(existing.get("verifications", 0)) + 1
                existing["last_verified_at"] = now
                entry = existing
            else:
                entry = {
                    "signature_hash": sig,
                    "name": getattr(candidate, "name", sig),
                    "cluster": getattr(candidate, "cluster", None),
                    "tool_sequence": list(getattr(candidate, "tool_sequence", ()) or ()),
                    "support": int(getattr(candidate, "support", 0)),
                    "confidence": round(conf, 4),
                    "trigger_examples": list(
                        getattr(candidate, "trigger_examples", []) or []
                    )[:3],
                    "exemplar_trajectory_id": getattr(
                        candidate, "exemplar_trajectory_id", ""),
                    "graduated_at": now,
                    "last_verified_at": now,
                    "verifications": 1,
                }
                data[sig] = entry
            # Bounded — drop the lowest-confidence skills on overflow.
            if len(data) > MAX_SKILLS:
                ordered = sorted(
                    data.items(),
                    key=lambda kv: kv[1].get("confidence", 0.0),
                    reverse=True,
                )
                data = dict(ordered[:MAX_SKILLS])
            self._save(data)
            # If the just-added skill was itself the lowest-confidence entry
            # and got evicted by the overflow trim, it was NOT persisted —
            # return None so the caller doesn't count/mint a macro for a skill
            # the store won't surface.
            return entry if sig in data else None

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def all_skills(self) -> List[dict]:
        """Every graduated skill, highest-confidence first."""
        with self._lock:
            data = self._load()
        return sorted(
            data.values(), key=lambda e: e.get("confidence", 0.0), reverse=True,
        )

    def count(self) -> int:
        with self._lock:
            return len(self._load())

    def relevant(self, query: str, *, limit: int = 3) -> List[dict]:
        """Graduated skills relevant to ``query`` — keyword overlap on
        trigger examples + cluster + tool names. Falls back to the
        highest-confidence skills when the query matches nothing."""
        skills = self.all_skills()
        if not skills:
            return []
        q = _tokens(query)
        if not q:
            return skills[:limit]
        scored = []
        for s in skills:
            hay = _tokens(
                " ".join(s.get("trigger_examples", []))
                + " " + str(s.get("cluster") or "")
                + " " + " ".join(s.get("tool_sequence", []))
            )
            overlap = len(q & hay)
            if overlap > 0:
                scored.append((overlap, s.get("confidence", 0.0), s))
        if not scored:
            return []
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [s for _, _, s in scored[:limit]]

    def format_for_prompt(self, *, query: Optional[str] = None, limit: int = 3) -> str:
        """A system-prompt block surfacing proven approaches. Empty when
        there is nothing relevant to surface."""
        skills = self.relevant(query, limit=limit) if query else self.all_skills()[:limit]
        if not skills:
            return ""
        lines = [
            "### PROVEN APPROACHES (auto-acquired from your own validated runs)"
        ]
        for s in skills:
            seq = " → ".join(s.get("tool_sequence", [])) or "(no tools)"
            trig = (s.get("trigger_examples") or [""])[0]
            trig = trig.strip().replace("\n", " ")[:90]
            cluster = s.get("cluster") or "general"
            line = f"  - [{cluster}] the sequence {seq} has worked {s.get('support', 0)}×"
            if trig:
                line += f' (e.g. for: "{trig}")'
            lines.append(line)
        lines.append(
            "Reuse a proven sequence when the current task matches — it is "
            "validated, not speculative."
        )
        return "\n".join(lines)
