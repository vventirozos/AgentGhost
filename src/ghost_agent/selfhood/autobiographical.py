"""Autobiographical memory — first-person experiential records.

Proposal item #1: a persistent log of "what it was like" to be the
agent on a given turn. Distinct from the distill ``Trajectory`` log
(which captures full structured tool traces for ML training); this
file is the agent's own first-person diary, summarised post-turn.

Each entry shares an id with the trajectory it summarises so the two
stay linked. The recognition layer can then cross-reference: a
retrieved experience can be traced back to the original tool trace
when needed, but the routine read path only needs the summary.

Storage: a single JSONL file under ``$GHOST_HOME/system/selfhood/
autobiographical.jsonl``. Append-only, line-buffered, sink failures
swallowed. Same shape as ``distill.collector`` so the agent's hot
path stays cheap.

Provenance & item #2 — continuity: every entry is tagged
``subject="self"``. Retrieval layers treat ``subject=="self"`` records
as "mine" rather than as external knowledge. Future-self reads its
own past in the first person.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .schema import Experience

logger = logging.getLogger("GhostSelfhood")


AUTOBIO_FILENAME = "autobiographical.jsonl"


class AutobiographicalMemory:
    """Append-only first-person experience log.

    The summary slot is the load-bearing field. A turn that does not
    produce one is not autobiographically significant and the writer
    refuses it (an experience with no narrative content is just a
    duplicate of the trajectory log)."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / AUTOBIO_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def append(self, exp: Experience) -> Optional[Path]:
        """Write ``exp`` to the log. Returns the path on success,
        None on disabled / failed / empty-summary.

        Never raises — autobiographical capture is secondary."""
        if not self.enabled:
            return None
        if not exp.summary or not exp.summary.strip():
            # An empty-summary experience would be a duplicate of the
            # trajectory log without adding any first-person content.
            # Refuse silently — callers that have no summary to write
            # should not write at all.
            return None
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(exp.to_jsonl())
                    f.write("\n")
                    f.flush()
            return self.path
        except Exception as e:
            logger.warning("autobiographical append failed: %s", e)
            return None

    def iter_experiences(self) -> Iterator[Experience]:
        """Stream experiences from disk, oldest first."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        yield Experience.from_dict(d)
                    except Exception:
                        continue
        except OSError as e:
            logger.warning("cannot read autobiographical log %s: %s", self.path, e)

    def recent(self, limit: int = 5) -> List[Experience]:
        """Return the most recent N experiences, newest last so the
        natural caller can do ``for e in mem.recent(3): print(e)`` and
        read in chronological order."""
        if limit <= 0:
            return []
        # Streaming the full file is fine — turns produce at most ~1
        # entry per minute under heavy use, so even a year of operation
        # is tens of thousands of lines.
        items: List[Experience] = []
        for exp in self.iter_experiences():
            items.append(exp)
        return items[-limit:]

    def search_my_past(self, query: str, limit: int = 5) -> List[Experience]:
        """Keyword-overlap search over my own past. Deliberately
        dependency-free — no embedder, no vector store, just lowercased
        token overlap on summary + user_first_words.

        Returned newest-first when scores tie; below the minimum overlap
        threshold (1 token) returns nothing.

        Why not vector search? The autobiographical log is small enough
        that exact-keyword overlap is competitive AND zero-dep; vector
        search lives on ``memory_system`` and any caller that wants
        semantic recall can route through there with ``subject:"self"``
        metadata."""
        if not query or limit <= 0:
            return []
        q_tokens = set(t for t in query.lower().split() if len(t) > 2)
        if not q_tokens:
            return []
        scored: List[tuple] = []
        for exp in self.iter_experiences():
            haystack = (exp.summary + " " + exp.user_first_words).lower()
            score = sum(1 for tok in q_tokens if tok in haystack)
            if score > 0:
                scored.append((score, exp))
        # Sort by score desc, then by timestamp desc (newest first).
        scored.sort(key=lambda s: (s[0], s[1].timestamp), reverse=True)
        return [exp for _, exp in scored[:limit]]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
        except OSError:
            return 0
        return n


def summarise_turn_first_person(
    *,
    user_request: str,
    tool_names: Iterable[str],
    outcome: str,
    final_response: str,
    failure_reason: str = "",
) -> str:
    """Template-based first-person summary of a turn.

    Cheap, deterministic, zero-LLM. Produces a one-sentence experience
    record from the raw turn data. The narrative phase later
    rewrites a richer summary across many of these.

    Kept template-based (not LLM) so the autobiographical write is on
    the hot path of every turn without paying a round-trip. The
    narrative consolidation phase is where LLM polish happens."""
    ur = (user_request or "").strip().replace("\n", " ")
    ur_short = ur[:120] + ("…" if len(ur) > 120 else "")
    tools = [t for t in tool_names if t]
    if tools:
        if len(tools) == 1:
            tool_phrase = f"reached for {tools[0]}"
        elif len(tools) == 2:
            tool_phrase = f"used {tools[0]} and {tools[1]}"
        else:
            tool_phrase = f"strung together {', '.join(tools[:-1])} and {tools[-1]}"
    else:
        tool_phrase = "reasoned through it without tools"

    out = (outcome or "unknown").lower()
    if out == "passed":
        outcome_phrase = "and the answer landed"
    elif out == "failed":
        reason = (failure_reason or "").strip().replace("\n", " ")[:120]
        if reason:
            outcome_phrase = f"and it didn't land: {reason}"
        else:
            outcome_phrase = "and it didn't land"
    else:
        outcome_phrase = "without a verdict either way"

    if ur_short:
        return f'I worked on "{ur_short}". I {tool_phrase} {outcome_phrase}.'
    return f"I {tool_phrase} {outcome_phrase}."
