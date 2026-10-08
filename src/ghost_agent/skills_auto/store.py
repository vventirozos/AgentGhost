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


#: words that say nothing about the TASK (§4LC: "the"/"you"/"what" matched
#: every English sentence, so past requests rode 40/40 owner turns)
_STOP = frozenset("""
the and for you your what how why who when where which with this that these those from into about
can could would should will please tell show give make get let want need like just also then than
are was were has have had does did not but all any some our out its it's i'm im me my mine yes no
hi hello hey thanks thank ok okay now new use using one two kai gia apo sto sti stin ston tou tis
ton tin ta na ti pos pou poio poia poios einai exo eimai mou mas sou sas den tha me se
""".split())

#: a past request older than this since it was last re-verified is not
#: surfaced (§4LC: an entry left only on overflow; one unverified since
#: 09-24 still rode every turn)
STALE_AFTER_DAYS = 14


def _tokens(text: str) -> set:
    """Content-word tokens of ``text``: case- and accent-folded, Greek
    transliterated (one script for both sides — §4LC: a Greek query had no
    `[a-z0-9]` token and got the top 3 regardless), stop-words dropped."""
    from ..memory.egress import fold
    folded = fold(text or "")[0]
    return {t for t in re.findall(r"[a-z0-9]+", folded) if len(t) > 2 and t not in _STOP}


def minable_requests(trajectories):
    """The trajectories past requests may be mined from: real requests only
    (`core.admissibility`, consumer "auto_skill_extraction"). §4LC: probe,
    reflection and bench turns were mined, and their verbatim text
    ("DISPATCH-OK-77") rode owner prompts as examples."""
    from ..core.admissibility import admitted_task_kinds
    kinds = set(admitted_task_kinds("skills_auto"))    # §4MI: the matrix's row (the old key was unregistered)
    for t in trajectories or []:
        if str(getattr(t, "task_kind", "") or "user_request") in kinds:
            yield t


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
        except json.JSONDecodeError as e:
            # §4M (Lens A MAJOR-2): returning {} left the corrupt bytes in
            # place for the NEXT graduate() to atomically overwrite — every
            # prior graduated skill unrecoverable. Preserve the corrupt file
            # under a timestamped sidecar first (same policy as the playbook,
            # profile and journal stores), then start empty.
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            backup = self.path.with_suffix(self.path.suffix + f".corrupt-{ts}")
            try:
                self.path.replace(backup)
                logger.warning(
                    "auto_skills.json was corrupt (%s); preserved as %s", e, backup)
            except OSError as rename_err:
                logger.error(
                    "auto_skills.json corrupt AND rename failed: %s", rename_err)
            return {}
        except OSError as e:
            logger.warning("auto-skill store read failed (%s); starting empty", e)
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        try:
            import os
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            # §4M (Lens A MINOR): fsync before the rename so power loss
            # can't promote a torn/empty file (same policy as journal).
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
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
        return self.graduate_changed(candidate, confidence=confidence)[0]

    def graduate_changed(self, candidate, *, confidence: Optional[float] = None):
        """§4MI: ``(entry, changed)`` — the idle phase counted every
        unchanged re-graduation (the §4LF no-new-evidence return) as a
        graduation and ledgered "graduated 17 proven skill(s)" 39 times
        after the last real change. Only a NEW entry or new evidence is
        ``changed``."""
        sig = getattr(candidate, "signature_hash", "") or getattr(candidate, "name", "")
        now = datetime.utcnow().isoformat() + "Z"
        conf = float(confidence if confidence is not None
                     else getattr(candidate, "confidence", 0.0))
        with self._lock:
            data = self._load()
            existing = data.get(sig)
            _sa_grew = False
            if existing:
                # §4LF: only NEW EVIDENCE is a verification. Every idle run
                # re-graduated every skill on the unchanged corpus (950
                # "verifications" on one), which also refreshed
                # `last_verified_at` and so disabled the 14-day staleness rule
                _new_support = int(getattr(candidate, "support", 0)) > int(existing.get("support", 0))
                _new_conf = round(conf, 4) != round(float(existing.get("confidence", 0.0) or 0.0), 4)
                if not (_new_support or _new_conf):
                    return existing, False
                _sa_grew = bool(_new_support)        # r2: a confidence-only change is a re-check, not a graduation
                existing["support"] = max(
                    int(existing.get("support", 0)),
                    int(getattr(candidate, "support", 0)),
                )
                # Track the verifier's CURRENT confidence, not a running
                # max: max() made downgrades impossible, so a skill's
                # stored confidence could only ever ratchet above its
                # observed pass-rate and the deprecation threshold became
                # unreachable.
                existing["confidence"] = round(conf, 4)
                existing["verifications"] = int(existing.get("verifications", 0)) + 1
                existing["last_verified_at"] = now
                # the examples come from the CURRENT evidence (§4LS m3): the
                # first graduation's were kept forever, so probe text mined
                # before §4LC stayed in owner prompts while real traffic
                # kept the entry fresh
                _ex = list(getattr(candidate, "trigger_examples", []) or [])
                if _ex:
                    existing["trigger_examples"] = _ex[:3]
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
            if sig not in data:
                return None, False
            return entry, (True if not existing else _sa_grew)

    #: Turn keys whose bookings are already on disk. Bounded, and keyed on
    #: the TURN rather than kept in a single slot: the lesson system's
    #: equivalent uses one `_retrieval_turn_key` that any interleaved
    #: retrieval resets, which a review found to be one of three routes by
    #: which it double-books. There is exactly one call site here and it is
    #: not inside a loop, so this is insurance rather than load-bearing —
    #: but a counter that can over-report is not an honest number.
    _TURN_DEDUP_MAX = 64

    def record_surfaced(self, hashes, *, turn_key: str = "") -> int:
        """Book a retrieval for each graduated skill surfaced this turn.

        §4CT — the write half of the gap. ONE load+save for the whole set
        (the lesson store paid ~20 synchronous ~100KB rewrites per turn
        before its bulk variant existed; this one starts bulk).

        ⚠ WHAT THIS NUMBER MEANS, and what it does not. It counts times a
        skill was INJECTED INTO THE PROMPT, which is the question the yield
        surface could not answer. It is NOT a helpfulness signal: nothing
        here says the model read it, used it, or benefited. The lessons
        store has a second pair of arms (`succeeded_retrievals` /
        `failed_retrievals`) fed from the turn's verified outcome; this
        store does not, and the note on the yield row says so rather than
        letting `invoked` be read as usefulness.

        Idempotent per ``turn_key``: re-booking the same (turn, skill) is a
        no-op. An EMPTY turn key cannot be deduped, so it books — that is
        the honest direction for a real turn that somehow carries no
        request id, and the caller gates simulated turns out before ever
        getting here.

        Returns the number of skills updated. Never raises: post-injection
        bookkeeping must not break a turn.
        """
        keys = [str(h) for h in (hashes or []) if h and str(h).strip()]
        if not keys:
            return 0
        with self._lock:
            tk = str(turn_key or "")
            if tk:
                booked = getattr(self, "_booked_by_turn", None)
                if booked is None:
                    from collections import OrderedDict
                    booked = OrderedDict()
                    self._booked_by_turn = booked
                seen = booked.setdefault(tk, set())
                booked.move_to_end(tk)
                while len(booked) > self._TURN_DEDUP_MAX:
                    booked.popitem(last=False)
                keys = [k for k in keys if k not in seen]
                if not keys:
                    return 0
                seen.update(keys)
            try:
                data = self._load()
                now = datetime.utcnow().isoformat() + "Z"
                updated = 0
                for k in keys:
                    entry = data.get(k)
                    if not isinstance(entry, dict):
                        continue
                    entry["retrievals"] = int(entry.get("retrievals") or 0) + 1
                    entry["last_retrieved_at"] = now
                    updated += 1
                if updated:
                    self._save(data)
                return updated
            except Exception as e:            # noqa: BLE001
                logger.debug("record_surfaced skipped (non-critical): %s", e)
                return 0

    def remove(self, signature_hash: str) -> bool:
        """Remove a graduated skill by signature hash. Returns True when an
        entry was actually deleted.

        This is the deprecation path the verifier's ``action="deprecate"``
        verdict always implied but never had: without it, a skill that
        failed re-verification could only ever leave the store via silent
        lowest-confidence eviction at the MAX_SKILLS cap."""
        if not signature_hash:
            return False
        with self._lock:
            data = self._load()
            if signature_hash not in data:
                return False
            del data[signature_hash]
            self._save(data)
        logger.info("auto-skill store: removed deprecated skill %s",
                    signature_hash)
        return True

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

    #: content words a query must share with a skill's examples/cluster
    MIN_OVERLAP = 2

    def relevant(self, query: str, *, limit: int = 3) -> List[dict]:
        """Graduated skills relevant to ``query``: at least
        ``MIN_OVERLAP`` content words shared with the trigger examples or
        cluster (tool names do not count — "file" matched every
        file_system skill), re-verified within ``STALE_AFTER_DAYS``.
        NOTHING when the query has no content word: the old fallback
        surfaced the top 3 on every Greek turn."""
        skills = [s for s in self.all_skills() if not self._stale(s)]
        if not skills:
            return []
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for s in skills:
            hay = _tokens(
                " ".join(s.get("trigger_examples", []))
                + " " + str(s.get("cluster") or "")
            )
            overlap = len(q & hay)
            if overlap >= self.MIN_OVERLAP:
                scored.append((overlap, s.get("confidence", 0.0), s))
        if not scored:
            return []
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [s for _, _, s in scored[:limit]]

    @staticmethod
    def _stale(entry: dict) -> bool:
        ts = str(entry.get("last_verified_at") or entry.get("graduated_at") or "")
        try:
            when = datetime.fromisoformat(ts.rstrip("Z"))
        except ValueError:
            return True
        return (datetime.utcnow() - when).total_seconds() > STALE_AFTER_DAYS * 86400

    def surfaced_for_prompt(self, *, query: Optional[str] = None,
                            limit: int = 3):
        """``(block, [signature_hash, ...])`` — the prompt block AND which
        skills it surfaced.

        §4CT. `format_for_prompt` returned only the text, so the ONE thing
        the caller needed in order to record that a skill had been used was
        thrown away at the moment it was known. That is why this loop read
        `unmeasured` on the yield surface: the skills ARE injected on every
        matching turn, and nothing anywhere counted it — an instrumentation
        gap, not a dead loop, and the two lead opposite places.

        `format_for_prompt` is now a thin wrapper over this, so there is one
        formatter and the hashes cannot drift from the block that listed
        them.
        """
        skills = (self.relevant(query, limit=limit) if query
                  else [e for e in self.all_skills() if not self._stale(e)][:limit])
        if not skills:
            return "", []
        hashes = [str(s.get("signature_hash") or "") for s in skills
                  if s.get("signature_hash")]
        return self._render(skills), hashes

    def format_for_prompt(self, *, query: Optional[str] = None, limit: int = 3) -> str:
        """A system-prompt block surfacing proven approaches. Empty when
        there is nothing relevant to surface."""
        return self.surfaced_for_prompt(query=query, limit=limit)[0]

    def _render(self, skills: List[dict]) -> str:
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
            "These sequences recurred in your past successful runs; reuse one only "
            "when the current task is genuinely the same kind of task."
        )
        return "\n".join(lines)
