"""Contradiction resolution log.

Records every belief revision performed by the smart-memory contradiction
engine so the agent can explain *why* it changed its mind:

    "I previously thought X, but updated to Y because you said Z."

Persisted as a JSON file alongside the other memory stores.
"""

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("GhostAgent")


class ContradictionLog:
    MAX_ENTRIES = 200

    def __init__(self, memory_dir: Path):
        self.file_path = memory_dir / "contradiction_log.json"
        self._lock = threading.RLock()
        # Set when the log file is PRESENT but unreadable — see _load().
        self._degraded = False
        if not self.file_path.exists():
            self._save([])
        else:
            # Probe once at construction so an unreadable / corrupt log is
            # detected (and quarantined) at boot rather than on the first
            # record() — clear() writes without reading, so a lazy check
            # left one path that could still overwrite a corrupt file
            # without preserving it.
            self._load()

    def _save(self, entries: list):
        if self._degraded:
            # The file exists and could not be read: record() would
            # otherwise write a ONE-entry list over the whole history.
            logger.warning(
                "Contradiction log write skipped: %s is unreadable and must "
                "not be overwritten.", self.file_path.name,
            )
            return
        tmp = self.file_path.with_suffix(".tmp")
        # fsync before rename — rename alone can publish a torn/empty file
        # on power loss (journal.py's rationale, applied to all siblings).
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(entries, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.file_path)

    def _quarantine_corrupt(self, why: str) -> list:
        """Corrupt log: preserve the raw file as a timestamped sidecar
        BEFORE the next _save() overwrites it, then start clean. Same
        policy as journal.py / profile.py."""
        try:
            sidecar = self.file_path.with_suffix(f".corrupt-{int(time.time())}.json")
            os.replace(self.file_path, sidecar)
            logger.warning(
                "contradiction_log.json was corrupt (%s); preserved to %s and "
                "started a fresh log.", why, sidecar.name,
            )
        except Exception:
            logger.warning(
                "contradiction_log.json was corrupt (%s) and could not be "
                "preserved; starting a fresh log.", why,
            )
        return []

    def _load(self) -> list:
        """Read the log.

        Three distinct outcomes — the old ``except Exception: return []``
        collapsed them into one and made an unreadable file behave exactly
        like an absent one, so the next record() atomically overwrote 200
        entries of belief-revision history with a single entry, silently:

        * absent        → [] (normal cold start);
        * corrupt       → sidecar the bytes, then [] (safe to overwrite:
                          the original is preserved);
        * unreadable    → [] **and fail closed** (``_degraded``), so no
                          write can destroy the intact file. Cleared
                          automatically as soon as a read succeeds.
        """
        try:
            content = self.file_path.read_text()
        except FileNotFoundError:
            self._degraded = False
            return []
        except UnicodeDecodeError:
            self._degraded = False
            return self._quarantine_corrupt("undecodable bytes")
        except OSError as exc:
            if not self._degraded:
                logger.error(
                    "contradiction_log.json is present but unreadable (%s: %s). "
                    "Belief-revision history is unavailable and writes are "
                    "BLOCKED so the file is not overwritten.",
                    type(exc).__name__, exc,
                )
            self._degraded = True
            return []
        if not content.strip():
            self._degraded = False
            return []
        try:
            data = json.loads(content)
            # Wrong-type (dict/scalar) would break record()/get_recent which
            # expect a list — treat as corrupt.
            if not isinstance(data, list):
                raise ValueError(
                    f"log is a {type(data).__name__}, expected list")
        except Exception as exc:
            self._degraded = False
            return self._quarantine_corrupt(f"{type(exc).__name__}: {exc}")
        self._degraded = False
        return data

    def record(self, new_fact: str, old_facts: list, deleted_ids: list, reason: str = ""):
        """Record a belief revision event.

        Parameters
        ----------
        new_fact : str
            The incoming fact that triggered the revision.
        old_facts : list[dict] | list[str]
            The old facts that were superseded. The contradiction engine
            passes dicts (each has 'id' and 'text'); the project-advancer
            path (core.project_safety.route_contradiction) passes plain
            strings with no ids to match.
        deleted_ids : list[str]
            IDs of the memories that were actually deleted.
        reason : str, optional
            Free-form reason from the contradiction engine.
        """
        if not isinstance(old_facts, (list, tuple)):
            old_facts = [old_facts] if old_facts else []
        deleted_ids = deleted_ids if isinstance(deleted_ids, list) else list(deleted_ids or [])
        superseded = []
        for f in old_facts:
            if isinstance(f, dict):
                if f.get("id") in deleted_ids or str(f.get("id", "")).replace("ID:", "").strip() in deleted_ids:
                    superseded.append({"id": f.get("id", "?"), "text": f.get("text", "?")})
            elif f is not None:
                # String-sourced fact: no id to match against deleted_ids,
                # so include it unconditionally — dropping it would leave
                # the revision unexplainable.
                superseded.append({"id": "", "text": str(f)})
        entry = {
            "timestamp": datetime.now().isoformat(),
            "new_fact": new_fact,
            "superseded": superseded,
            "deleted_ids": deleted_ids,
            "reason": reason,
        }
        with self._lock:
            entries = self._load()
            entries.insert(0, entry)
            if len(entries) > self.MAX_ENTRIES:
                entries = entries[:self.MAX_ENTRIES]
            self._save(entries)
            # Did the entry actually land? `_save` is deliberately a silent
            # no-op when the store is `_degraded` (unreadable/unwritable file),
            # which is the right FILE policy but was reported to the caller as
            # SUCCESS — `record()` returned None either way and still logged
            # "Belief revised" (§4R Lens-D). The caller had by then already
            # performed an IRREVERSIBLE vector delete, so the erased text
            # vanished with no record anywhere. Return a truthful bool so the
            # caller can refuse to delete when the revision cannot be recorded.
            _recorded = not getattr(self, "_degraded", False)
        if not _recorded:
            logger.warning(
                "Belief revision NOT recorded (contradiction log is degraded): "
                "'%s' — callers must not delete on this result.", new_fact[:60])
            return False
        # INFO (was debug → dropped in prod): the agent changing its mind by
        # deleting old facts is a first-class cognitive event and must be
        # reconstructable from the durable log — WITH the dropped facts named.
        _dropped = "; ".join(
            (s.get("text") or "")[:40] for s in (superseded or [])[:2])
        logger.info(
            "Belief revised: '%s' superseded %d old fact(s)%s",
            new_fact[:60], len(deleted_ids),
            f": {_dropped}" if _dropped else "",
        )
        return True

    def get_recent(self, limit: int = 10) -> list:
        """Return the most recent contradiction events."""
        with self._lock:
            entries = self._load()
        return entries[:limit]

    # Interrogative / filler tokens that would match every entry — the
    # overlap matcher below must key on CONTENT words only.
    _QUERY_STOPWORDS = frozenset({
        "what", "whats", "when", "where", "which", "whose", "who", "whom",
        "does", "did", "have", "has", "had", "that", "this", "these",
        "those", "with", "about", "know", "tell", "show", "give", "still",
        "then", "than", "them", "they", "there", "here", "will", "would",
        "could", "should", "your", "yours", "mine", "please", "anymore",
        "currently", "right",
        # Domain filler measured on the live ledger (§4R): these appear in
        # nearly every entry, so as HAYSTACK tokens they matched almost any
        # message. `user` alone was in 50/50 scanned entries.
        "user", "users", "project", "projects", "current", "system",
        "work", "works", "working", "thing", "things", "stuff", "need",
        "needs", "want", "wants", "using", "used",
    })

    # How many recent entries the matcher scans. Was a bare `[:50]`: with 96
    # live entries that left 46 — every genuinely useful revision, since the
    # newest 50 were dominated by one 30-hour burst of self-play churn —
    # structurally unreachable. Scanning the whole (capped) ledger costs a
    # few hundred string ops and removes the cliff; MAX_ENTRIES already bounds
    # the file.
    _SEARCH_WINDOW = MAX_ENTRIES

    # A match must cover at least this FRACTION of the query's content tokens.
    # An absolute floor alone is not enough: measured over 240 REAL user turns,
    # a floor of 2 fired on 61.3% of them because long messages clear any fixed
    # count trivially (16+ content tokens fired 98.1% of the time). The ratio
    # is what makes a long message need proportionally more evidence.
    _MIN_OVERLAP_RATIO = 0.34


    def explain_belief_change(self, query: str) -> str:
        """Search the log for contradictions related to a query.

        Matching is significant-token overlap, not whole-string
        containment: the production caller passes the ENTIRE user message,
        and `"what car do i drive now?" in "user drives a bmw"` is never
        True — the whole-substring form made this surface inert for any
        multi-word message (it shipped 2026-07-20 precisely because it had
        no live caller, and then never fired through the one it got).

        Returns a human-readable explanation if found, empty string
        otherwise."""
        if not query:
            return ""
        query_lower = query.lower()
        query_tokens = {
            t for t in re.findall(r"[a-z0-9]+", query_lower)
            if len(t) > 3 and t not in self._QUERY_STOPWORDS
        }
        with self._lock:
            entries = self._load()
        # SCORED matching, not first-N-that-touch (§4R Lens-A/C/D, all three
        # independently). The previous rule accepted an entry if ANY ONE query
        # token shared a prefix with ANY ONE haystack token, then emitted the
        # five most RECENT survivors. Measured against the live 96-entry
        # ledger that fired on 67 of 67 hydrating turns — 100% — and the five
        # lines it injected were the same recency-ordered block regardless of
        # the question. Over-correcting the 2026-07-20 inert version produced
        # the opposite failure, and the tests missed it because their fixtures
        # hold a single entry.
        window = entries[:self._SEARCH_WINDOW]
        # Pass 1 — tokenise every entry and count DOCUMENT FREQUENCY. Stopwords
        # are screened on BOTH sides now; applying them only to the query left
        # `user` (in 50/50 live entries), `project` (48/50) and `current`
        # (30/50) matching almost any message. A fixed stopword list can only
        # cover words we thought of, so df does the rest: a token carried by
        # most of the ledger is not evidence of anything.
        prepared = []
        for entry in window:
            new_fact = (entry.get("new_fact") or "").lower()
            old_texts = " ".join(
                s.get("text", "").lower() for s in entry.get("superseded", [])
            )
            hay = f"{new_fact} {old_texts}"
            hay_tokens = {
                t for t in re.findall(r"[a-z0-9]+", hay)
                if len(t) > 3 and t not in self._QUERY_STOPWORDS
            }
            prepared.append((entry, hay, hay_tokens))

        scored = []
        for entry, hay, hay_tokens in prepared:
            # Legacy exact containment still wins outright (single-word or
            # quoted queries) — a genuine substring hit is unambiguous.
            if query_lower in hay:
                scored.append((float("inf"), entry))
                continue
            if not query_tokens:
                continue
            matched_hay = []
            for qt in query_tokens:
                if qt in hay_tokens:
                    matched_hay.append(qt)
                    continue
                # BOUNDED prefix tolerance. Unbounded startswith() made
                # `code`↔`codename` and `dark`↔`darkweb` count as matches;
                # requiring near-equal length keeps real inflections
                # (`change`↔`changed`, `drive`↔`drives`) and drops the rest.
                hit = next((ht for ht in hay_tokens
                            if abs(len(qt) - len(ht)) <= 2
                            and (qt.startswith(ht) or ht.startswith(qt))), None)
                if hit is not None:
                    matched_hay.append(hit)
            if not matched_hay:
                continue
            # Evidence rule: TWO shared content tokens — or all of them when
            # the query has fewer, since a one-content-token question
            # ("what car do i drive now?" → {drive}) can offer no more.
            #
            # Each variant here was MEASURED against the live 96-entry ledger,
            # because the two obvious rules are both wrong:
            #   * accept ANY single shared token → 9/10 noise queries fire,
            #     even WITH the stopword + bounded-prefix fixes. This was the
            #     shipped behaviour: it fired on 67 of 67 live turns.
            #   * require two, flatly            → 0/10 noise, but it breaks
            #     genuine one-token questions and the recall test that the
            #     2026-07-26 un-inerting fix added.
            # The proportional rule lands at 2/10 noise with signal intact.
            #
            # A "…or one RARE token" escape hatch was tried and REMOVED: as a
            # 0.20 document-frequency ratio AND as an absolute df<=2 it
            # measured 7-10/10 noise, because in a narrow-domain ledger most
            # content tokens appear in only one or two entries anyway. It
            # existed only to satisfy a synthetic test of my own.
            #
            # KNOWN RESIDUAL: the 2/10 that still fire are single-content-token
            # queries ("what is my name"). Accepted — tightening further
            # re-breaks real recall. And no token matcher can bridge "what car
            # do i drive?" → "the user owns a BMW" (zero shared tokens); that
            # needs embeddings, not a longer stopword list.
            # Deduplicate: the exact branch appends the QUERY token and the
            # prefix branch the HAYSTACK token, so two query tokens collapsing
            # onto one haystack token used to count as two pieces of evidence.
            matched = set(matched_hay)
            ratio = len(matched) / len(query_tokens)
            if len(matched) < 2 or ratio < self._MIN_OVERLAP_RATIO:
                continue
            scored.append((ratio, entry))
        if not scored:
            return ""

        # Rank by RELEVANCE, then recency — the old code truncated in raw file
        # order, so the newest entries crowded out the best-matching ones. On
        # the live ledger that meant a burst of self-play "project codename"
        # churn monopolised every response.
        scored.sort(key=lambda p: (p[0], p[1].get("timestamp") or ""),
                    reverse=True)

        lines = ["## BELIEF REVISION HISTORY:"]
        for _score, m in scored[:5]:
            old_strs = "; ".join(
                s.get("text", "?")[:80] for s in m.get("superseded", [])
            )
            # `reason` is rendered now: it is present on 96/96 live entries and
            # the module docstring promises "…because you said Z", but Z was
            # dropped, leaving the agent unable to say WHY it changed its mind.
            _reason = (m.get("reason") or "").strip()
            lines.append(
                f"- [{m.get('timestamp', '?')}] Updated to: \"{m.get('new_fact', '?')[:160]}\""
                f" (superseded: \"{old_strs}\")"
                + (f" [reason: {_reason[:80]}]" if _reason else "")
            )
        return "\n".join(lines)

    def clear(self):
        """Wipe the contradiction log."""
        with self._lock:
            self._save([])
