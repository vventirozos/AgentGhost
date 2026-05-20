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
import math
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .schema import Experience

logger = logging.getLogger("GhostSelfhood")


AUTOBIO_FILENAME = "autobiographical.jsonl"


# Lightweight, zero-dependency topic clustering. The `cluster` field on
# Experience used to be inert (set to None on every record); deriving a
# coarse label at capture time lets the narrative / recall layers say
# "I've done this class of thing before" instead of treating every turn
# as unique. Order is irrelevant — the label with the most keyword hits
# wins, ties broken by dict order.
_CLUSTER_KEYWORDS = {
    "debugging": ("error", "fix", "debug", "crash", "exception", "traceback",
                  "didn't land", "failed"),
    "coding": ("code", "python", "function", "script", "compile", "syntax",
               "import", "refactor", "implement"),
    "data": ("sql", "query", "database", "postgres", "csv", "dataframe",
             "table", "schema"),
    "research": ("search", "research", "look up", "web", "wikipedia",
                 "fact check", "investigate"),
    "writing": ("write", "essay", "draft", "summary", "document", "article"),
    "memory": ("remember", "recall", "memory", "profile", "knowledge base"),
    "self_play": ("synthetic training", "self-play", "self play", "challenge",
                  "exercise"),
    "math": ("calculate", "compute", "equation", "arithmetic", "number"),
}


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens — shared by the recall scorer and clustering."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _derive_cluster(text: str) -> Optional[str]:
    """Best-effort coarse topic label for a turn. None when nothing matches."""
    low = (text or "").lower()
    if not low.strip():
        return None
    best: Optional[str] = None
    best_hits = 0
    for label, keywords in _CLUSTER_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in low)
        if hits > best_hits:
            best_hits, best = hits, label
    return best


def _outcome_phrase(outcome: str, failure_reason: str = "") -> str:
    """The trailing clause of a first-person turn summary. Extracted so the
    capture path and the post-hoc outcome backfill phrase verdicts identically."""
    out = (outcome or "unknown").lower()
    if out == "passed":
        return "and the answer landed"
    if out == "failed":
        reason = (failure_reason or "").strip().replace("\n", " ")[:120]
        return f"and it didn't land: {reason}" if reason else "and it didn't land"
    return "without a verdict either way"


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
        """Relevance-ranked search over my own past. Deliberately
        dependency-free — no embedder, no vector store — but scored with
        inverse-document-frequency weighting so a rare, distinctive query
        term (e.g. "trapdoor") dominates a common one (e.g. "the").

        That is the difference between "recent-N" recall and recall that
        actually surfaces the relevant memory: a salient experience from
        hundreds of turns ago becomes reachable as long as the query
        shares its distinctive vocabulary.

        Returned best-match first, newest-first on ties; below the minimum
        score (one matched token) returns nothing.

        Why not vector search? The autobiographical log is small and
        append-only; IDF-weighted overlap is competitive AND zero-dep.
        A caller wanting full embeddings can still route through
        ``memory_system`` with ``subject:"self"`` metadata."""
        if not query or limit <= 0:
            return []
        q_tokens = set(t for t in _tokenize(query) if len(t) > 2)
        if not q_tokens:
            return []

        experiences = list(self.iter_experiences())
        if not experiences:
            return []

        # Document frequency of each query token across the whole log.
        haystacks: List[str] = []
        df: Dict[str, int] = {t: 0 for t in q_tokens}
        for exp in experiences:
            hs = (exp.summary + " " + exp.user_first_words + " "
                  + (exp.cluster or "")).lower()
            haystacks.append(hs)
            for t in q_tokens:
                if t in hs:
                    df[t] += 1

        n = len(experiences)
        idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in q_tokens}

        scored: List[tuple] = []
        for exp, hs in zip(experiences, haystacks):
            score = sum(idf[t] for t in q_tokens if t in hs)
            if score > 0:
                scored.append((score, exp))
        scored.sort(key=lambda s: (s[0], s[1].timestamp), reverse=True)
        return [exp for _, exp in scored[:limit]]

    def update_outcome(
        self, trajectory_id: str, outcome: str, *, failure_reason: str = "",
    ) -> bool:
        """Backfill the verdict of an already-written experience.

        The capture path runs on the hot turn loop, before the verifier
        / reflection layers have decided whether the turn actually
        succeeded — so most records are first written ``outcome="unknown"``.
        This method rewrites the matching entry once that verdict exists,
        so the agent's memory of its own past stops being verdict-blind.

        Updates the most recent entry sharing ``trajectory_id``. Returns
        True when an entry was changed. Never raises — backfill is
        secondary to the user turn."""
        if not self.enabled or not trajectory_id or not self.path.exists():
            return False
        new_outcome = (outcome or "").strip().lower()
        if new_outcome not in ("passed", "failed"):
            return False
        try:
            with self._lock:
                lines = self.path.read_text(encoding="utf-8").splitlines()
                for i in range(len(lines) - 1, -1, -1):
                    s = lines[i].strip()
                    if not s:
                        continue
                    try:
                        d = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if d.get("trajectory_id") != trajectory_id:
                        continue
                    if (d.get("outcome") or "").strip().lower() == new_outcome:
                        return False  # already correct — nothing to do
                    d["outcome"] = new_outcome
                    # Patch the prose verdict clause too, so recall and the
                    # narrative read a coherent summary. Only the "unknown"
                    # clause is fixed text we can swap deterministically.
                    summary = d.get("summary") or ""
                    stale = "without a verdict either way"
                    if stale in summary:
                        d["summary"] = summary.replace(
                            stale, _outcome_phrase(new_outcome, failure_reason), 1,
                        )
                    lines[i] = json.dumps(d, ensure_ascii=False)
                    tmp = self.path.with_suffix(".jsonl.tmp")
                    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    tmp.replace(self.path)
                    return True
                return False
        except Exception as e:
            logger.warning("autobiographical update_outcome failed: %s", e)
            return False

    def cluster_counts(self) -> Dict[str, int]:
        """How many experiences fall into each derived cluster. Lets the
        narrative / stats layer report self-patterns ("I keep doing X")."""
        counts: Dict[str, int] = {}
        for exp in self.iter_experiences():
            c = (exp.cluster or "").strip()
            if c:
                counts[c] = counts.get(c, 0) + 1
        return counts

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

    outcome_phrase = _outcome_phrase(outcome, failure_reason)

    if ur_short:
        return f'I worked on "{ur_short}". I {tool_phrase} {outcome_phrase}.'
    return f"I {tool_phrase} {outcome_phrase}."
