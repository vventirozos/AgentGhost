"""Cognitive Event Bus.

Native, in-memory pub/sub between the agent and its memory subsystems.
Two responsibilities:

* `hydrate_context(query)` — fan-out reads to Vector / Graph / Skill memories
  in parallel, fuse the disparate result lists with Reciprocal Rank Fusion,
  and return a single token-capped Markdown block ready to splice into a
  prompt.
* `publish_fact(event_type, fact_data)` — fan-out writes from a tool/event
  source into Vector + Graph + Profile + Skill in parallel, so callers
  never touch the underlying stores directly.

No external broker (Redis/RabbitMQ/etc.) — pure asyncio.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from ..utils.helpers import get_utc_timestamp

logger = logging.getLogger("GhostAgent")

# Stopwords + token-extraction config used when mapping a free-form query
# to graph seed words. Mirrors the heuristic that previously lived inline in
# `handle_chat`, kept here so the bus owns the entire hydration pipeline.
_STOPWORDS = {
    "this", "that", "what", "with", "from", "your", "have", "they", "will",
    "would", "could", "should", "just", "about", "like", "when", "then",
    "there", "their", "them", "some", "make", "error", "file", "code",
    "test", "http", "https", "user", "agent", "ghost", "please", "know",
    "want", "need",
}


class MemoryBus:
    # Soft cap on the recent-fact dedup ledger. RRF / hydration is unaffected
    # — this only stops `publish_fact` from fanning out the same write twice
    # in a row (e.g. when the model loops the same `update_profile` call).
    _DEDUP_LRU_MAX = 256

    def __init__(self,
                 vector_memory: Any = None,
                 graph_memory: Any = None,
                 skill_memory: Any = None,
                 profile_memory: Any = None,
                 episodic_memory: Any = None,
                 intent_weights: Any = None):
        self.vector = vector_memory
        self.graph = graph_memory
        self.skill = skill_memory
        self.profile = profile_memory
        self.episodic = episodic_memory
        # Learned RRF intent→source weights (core.rrf_weights). None →
        # the hand-tuned class defaults are used (zero behaviour change);
        # main.py injects a fitted matrix here only when a weights.json
        # exists. Kept per-instance so the classmethod default stays
        # intact for direct callers / tests.
        self._intent_weights = intent_weights
        # LRU of recently-published fact signatures, used by publish_fact to
        # short-circuit duplicate writes from looping callers.
        self._publish_lru: "OrderedDict[str, bool]" = OrderedDict()

    @staticmethod
    def _fact_signature(event_type: str, fact_data: Dict[str, Any]) -> str:
        """Stable hash of the event type + fact payload, used for dedup."""
        try:
            blob = json.dumps(
                {"event_type": event_type, "fact": fact_data},
                sort_keys=True,
                default=str,
            )
        except Exception:
            blob = f"{event_type}:{repr(fact_data)}"
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def _seen_recently(self, sig: str) -> bool:
        if sig in self._publish_lru:
            self._publish_lru.move_to_end(sig)
            return True
        self._publish_lru[sig] = True
        while len(self._publish_lru) > self._DEDUP_LRU_MAX:
            self._publish_lru.popitem(last=False)
        return False

    # -------------------------------------------------------- intent weights

    # Per-intent source weights for weighted RRF. Instead of treating all
    # three sources equally, we classify the query and boost the source
    # most likely to have the answer.
    _INTENT_WEIGHTS = {
        "factual":    {"graph": 2.0, "vector": 1.0, "skill": 0.5, "episodic": 0.3},
        "procedural": {"graph": 0.5, "vector": 1.0, "skill": 2.0, "episodic": 1.5},
        "contextual": {"graph": 1.0, "vector": 1.5, "skill": 1.0, "episodic": 1.0},
    }

    # Keyword-based intent classifier (no LLM call needed)
    _FACTUAL_KEYWORDS = {
        "who", "what", "where", "when", "name", "born", "lives", "works",
        "age", "address", "married", "company", "owns", "drives",
    }
    _PROCEDURAL_KEYWORDS = {
        "how", "fix", "solve", "debug", "error", "mistake", "lesson",
        # "practice" (single token), NOT "best practice": intent is matched
        # by token-set intersection below, where a two-word phrase can never
        # appear as a single token — so "best practice" was dead.
        "avoid", "pattern", "practice", "should", "must", "never",
        "always", "steps", "procedure", "workflow",
    }

    @classmethod
    def _classify_query_intent(cls, query: str) -> str:
        """Cheap keyword classifier for query intent. Returns 'factual',
        'procedural', or 'contextual'."""
        if not query:
            return "contextual"
        words = set(query.lower().split())
        factual_hits = len(words & cls._FACTUAL_KEYWORDS)
        procedural_hits = len(words & cls._PROCEDURAL_KEYWORDS)
        if factual_hits > procedural_hits and factual_hits > 0:
            return "factual"
        if procedural_hits > factual_hits and procedural_hits > 0:
            return "procedural"
        return "contextual"

    # ============================================================ HYDRATION

    async def hydrate_context(self, query: str, *,
                              max_chars: int = 6000,
                              rrf_k: int = 60,
                              context_budget: int = 0,
                              llm_client: Any = None) -> str:
        """Concurrently query Vector / Graph / Skill / Episodic memories and
        fuse the results with Reciprocal Rank Fusion. Returns a Markdown block
        (or an empty string when nothing is available).

        When ``context_budget`` > 0, the hydration budget scales adaptively:
        simple queries stay at 6000 chars, complex queries expand up to the
        budget cap (max 12000 chars). This prevents complex research tasks
        from being starved of memory context.

        RAG-Fusion: When ``llm_client`` is provided, decomposes the query into
        sub-queries for broader retrieval coverage before fusing.
        """
        if not query or not str(query).strip():
            return ""

        # Adaptive budget: scale based on query complexity
        if context_budget > 0:
            query_words = len(query.split())
            if query_words > 30:
                max_chars = min(context_budget, 12000)
            elif query_words > 15:
                max_chars = min(context_budget, 9000)
            # else: keep default 6000

        # RAG-Fusion: decompose into sub-queries for broader coverage
        sub_queries = await self._decompose_query(query, llm_client)

        # Fan-out retrieval for each sub-query in parallel
        all_ranked_lists = []
        fetch_coros = []
        for sq in sub_queries:
            fetch_coros.append(self._fetch_all_tiers(sq))
        tier_results_per_query = await asyncio.gather(*fetch_coros)

        # Flatten all tier results into a single ranked list set
        combined_vector, combined_graph, combined_skill, combined_episodic = [], [], [], []
        for vector_items, graph_items, skill_items, episodic_items in tier_results_per_query:
            combined_vector.extend(vector_items)
            combined_graph.extend(graph_items)
            combined_skill.extend(skill_items)
            combined_episodic.extend(episodic_items)

        # Deduplicate by text content
        combined_vector = self._dedup_items(combined_vector)
        combined_graph = self._dedup_items(combined_graph)
        combined_skill = self._dedup_items(combined_skill)
        combined_episodic = self._dedup_items(combined_episodic)

        intent = self._classify_query_intent(query)
        fused = self._reciprocal_rank_fusion(
            [combined_vector, combined_graph, combined_skill, combined_episodic],
            k=rrf_k, intent=intent,
            weight_overrides=self._intent_weights,
        )
        return self._format_markdown(fused, max_chars=max_chars)

    async def _fetch_all_tiers(self, query: str):
        """Fetch from all memory tiers for a single query."""
        return await asyncio.gather(
            self._fetch_vector(query),
            self._fetch_graph(query),
            self._fetch_skill(query),
            self._fetch_episodic(query),
        )

    @staticmethod
    def _dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate items by text content."""
        seen = set()
        deduped = []
        for item in items:
            text = item.get("text", "")
            if text not in seen:
                seen.add(text)
                deduped.append(item)
        return deduped

    async def _decompose_query(self, query: str,
                               llm_client: Any = None) -> List[str]:
        """Decompose a query into sub-queries for broader retrieval.

        Uses the LLM worker pool for decomposition if available,
        otherwise falls back to simple keyword-based splitting.
        """
        # Always include the original query
        sub_queries = [query]

        # Skip decomposition for short/simple queries
        if len(query.split()) < 8:
            return sub_queries

        # Try LLM-based decomposition via worker pool
        if llm_client is not None:
            route_fn = getattr(llm_client, "route", None)
            if route_fn:
                try:
                    result = await route_fn(
                        "DECOMPOSE_QUERY",
                        {
                            "messages": [{
                                "role": "user",
                                "content": (
                                    f"Decompose this query into 2-3 distinct search sub-queries "
                                    f"that together would find all relevant information. "
                                    f"Return ONLY a JSON array of strings.\n\n"
                                    f"Query: {query[:500]}"
                                ),
                            }],
                        },
                        max_tokens=256,
                        temperature=0.1,
                        fallback=None,
                    )
                    if result:
                        # route() returns either a content string or a full
                        # response dict depending on the code path. Handle both.
                        if isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = (
                                result.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                        else:
                            text = ""
                        import re as _re
                        # Try to parse JSON array from response
                        match = _re.search(r'\[[\s\S]*\]', text)
                        if match:
                            parsed = json.loads(match.group())
                            if isinstance(parsed, list):
                                sub_queries.extend(str(s) for s in parsed[:3])
                                return sub_queries[:4]  # Original + up to 3 decomposed
                except Exception as exc:
                    logger.debug("Query decomposition failed: %s", exc)

        # Fallback: simple heuristic decomposition.
        # Only split on " and also " or sentences that clearly list
        # independent topics (e.g., "fix auth and update the schema").
        # Skip if the query contains internal markers like "Context:" or
        # "|" — those are expanded/compound queries already.
        ql = query.lower()
        if "|" not in ql and "context:" not in ql:
            if " and also " in ql or " as well as " in ql:
                splitter = " and also " if " and also " in ql else " as well as "
                parts = ql.split(splitter)
                for part in parts[:2]:
                    part = part.strip()
                    if len(part.split()) >= 4:
                        sub_queries.append(part)

        return sub_queries[:4]

    # ------------------------------------------------------------- fetchers

    async def _fetch_vector(self, query: str) -> List[Dict[str, Any]]:
        if not self.vector:
            return []
        try:
            mem_string = await asyncio.to_thread(self.vector.search, query)
        except Exception as e:
            logger.warning(f"MemoryBus vector fetch failed: {type(e).__name__}: {e}")
            return []
        if not mem_string or not isinstance(mem_string, str):
            return []
        chunks = [c.strip() for c in mem_string.split("\n---\n") if c.strip()]
        return [{"source": "vector", "text": c} for c in chunks]

    async def _fetch_graph(self, query: str) -> List[Dict[str, Any]]:
        if not self.graph:
            return []
        words = self._extract_query_terms(query)
        if not words:
            return []
        try:
            edges = await asyncio.to_thread(
                self.graph.get_neighborhood, words, 15
            )
        except Exception as e:
            logger.warning(f"MemoryBus graph fetch failed: {type(e).__name__}: {e}")
            return []
        if not isinstance(edges, list):
            return []
        return [{"source": "graph", "text": e} for e in edges if e]

    async def _fetch_skill(self, query: str) -> List[Dict[str, Any]]:
        if not self.skill:
            return []
        try:
            playbook = await asyncio.to_thread(
                self.skill.get_playbook_context,
                query=query,
                memory_system=self.vector,
            )
        except Exception as e:
            logger.warning(f"MemoryBus skill fetch failed: {type(e).__name__}: {e}")
            return []
        if not playbook or not isinstance(playbook, str):
            return []
        if playbook.strip() in {"", "No lessons learned yet."}:
            return []
        return [{"source": "skill", "text": playbook.strip()}]

    async def _fetch_episodic(self, query: str) -> List[Dict[str, Any]]:
        if not self.episodic:
            return []
        try:
            episodes = await asyncio.to_thread(
                self.episodic.search_similar, query, 5
            )
        except Exception as e:
            logger.warning(f"MemoryBus episodic fetch failed: {type(e).__name__}: {e}")
            return []
        if not episodes:
            return []
        try:
            formatted = await asyncio.to_thread(
                self.episodic.format_for_context, episodes, 1500
            )
        except Exception:
            formatted = ""
        if not formatted or not formatted.strip():
            return []
        return [{"source": "episodic", "text": formatted.strip()}]

    # ----------------------------------------------------------------- RRF

    @classmethod
    def _reciprocal_rank_fusion(cls, ranked_lists: List[List[Dict[str, Any]]],
                                k: int = 60,
                                intent: str = "contextual",
                                weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                                ) -> List[Tuple[Dict[str, Any], float]]:
        """Weighted RRF: score(d) = sum_r weight_r / (k + rank_r(d)).

        When ``intent`` is provided, source-specific weights scale each
        ranker's contribution so factual queries boost graph results and
        procedural queries boost skill results. ``weight_overrides`` (a
        learned matrix from :mod:`core.rrf_weights`) supersedes the
        hand-tuned ``_INTENT_WEIGHTS`` when supplied; ``None`` falls back
        to the defaults, so direct classmethod callers are unaffected.
        """
        wmap = weight_overrides or cls._INTENT_WEIGHTS
        weights = wmap.get(intent, wmap.get("contextual", cls._INTENT_WEIGHTS["contextual"]))
        # Map source names to their weights. Sources: vector, graph, skill
        _source_order = ["vector", "graph", "skill"]

        scores: Dict[Tuple[str, str], float] = {}
        index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for ranker_idx, ranker in enumerate(ranked_lists):
            if not ranker:
                continue
            # Infer source from first item, fallback to positional mapping
            source_name = ranker[0].get("source", _source_order[ranker_idx] if ranker_idx < len(_source_order) else "vector")
            w = weights.get(source_name, 1.0)
            for rank, item in enumerate(ranker):
                key = (item["source"], item["text"])
                scores[key] = scores.get(key, 0.0) + w / (k + rank + 1)
                index[key] = item
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(index[key], score) for key, score in ordered]

    # ----------------------------------------------------------- formatting

    # Per-section share of the overall hydration budget. Without this the
    # formatter assembled the full Markdown then truncated by char count,
    # so a heavy graph section could silently swallow the high-signal skill
    # playbook. Budgets sum to 1.0 — graph/vector/skill split.
    _SECTION_BUDGETS = {"graph": 0.25, "vector": 0.40, "skill": 0.20, "episodic": 0.15}

    @staticmethod
    def _format_markdown(fused: List[Tuple[Dict[str, Any], float]],
                         max_chars: int = 6000) -> str:
        if not fused:
            return ""

        # Group by source for human-readable sectioning, but preserve the
        # RRF ordering inside each section so the highest-fused items lead.
        order = ["graph", "vector", "skill", "episodic"]
        sections: Dict[str, List[Tuple[str, float]]] = {s: [] for s in order}
        for item, score in fused:
            src = item["source"]
            if src in sections:
                sections[src].append((item["text"], score))

        headers = {
            "graph": "### TOPOLOGICAL KNOWLEDGE GRAPH:",
            "vector": "### MEMORY CONTEXT:",
            "skill": "### SKILL PLAYBOOK:",
            "episodic": "### PAST EPISODES:",
        }

        lines: List[str] = []
        for src in order:
            if not sections[src]:
                continue
            lines.append(headers[src])
            section_budget = max(200, int(max_chars * MemoryBus._SECTION_BUDGETS.get(src, 0.33)))
            used = 0
            included = 0
            for text, _score in sections[src]:
                remaining = section_budget - used
                if remaining <= 0:
                    lines.append(f"[... +{len(sections[src]) - included} more {src} items truncated for budget]")
                    break
                if len(text) > remaining:
                    if included == 0:
                        # First (and possibly only) item is bigger than the
                        # whole section budget — truncate it in place rather
                        # than letting it crowd out other sections.
                        lines.append(text[:remaining].rstrip() + " [...]")
                        used += remaining
                        included += 1
                        if len(sections[src]) > 1:
                            lines.append(f"[... +{len(sections[src]) - included} more {src} items truncated for budget]")
                        break
                    else:
                        lines.append(f"[... +{len(sections[src]) - included} more {src} items truncated for budget]")
                        break
                lines.append(text)
                used += len(text) + 1
                included += 1
            lines.append("")

        out = "\n".join(lines).strip()
        if len(out) > max_chars:
            out = out[:max_chars].rstrip() + "\n\n[... TRUNCATED]"
        return out + "\n\n" if out else ""

    @staticmethod
    def _extract_query_terms(query: str) -> List[str]:
        # Hard cap the input BEFORE tokenising. A real production log
        # showed a 4 KB validator dump being passed through hydration —
        # tokenising the full string produced ~600 tokens, ran difflib
        # 25× against the graph node index, and burned CPU for nothing.
        clipped = str(query)[:1000]
        raw = [w.lower().strip('.,?!;"\'()[]{}') for w in clipped.split()]
        words = list({w for w in raw if len(w) > 3 and w not in _STOPWORDS})
        words.sort(key=len, reverse=True)
        words = words[:25]
        words.append("user")
        return words

    # ============================================================ PUBLISH

    async def publish_fact(self, event_type: str,
                           fact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fan-out an extracted fact into the appropriate memory subsystems.

        `fact_data` is a free-form dict; recognised keys (all optional):
            text:           plain-text fact for the Vector store
            metadata:       dict — vector metadata; defaults to {type=event_type}
            triplets:       List[{subject,predicate,object}] for the Graph store
            profile_update: {category, key, value} for ProfileMemory
            skill:          {task, mistake, solution} for SkillMemory

        Returns a small report dict mapping subsystem name to outcome
        (`"ok"`, `"skip"`, `"dedup"`, or an error message). Exceptions are
        swallowed so a single failing subsystem can't break a publish.
        """
        results: Dict[str, Any] = {}

        # --- DEDUP: refuse duplicate publishes within an LRU window. The
        # production loop bug fired the same update_profile event 9× in
        # one request, reinforcing the same graph edge weight 9× and
        # distorting RRF scores. With this LRU the bus refuses identical
        # repeats without touching the underlying stores.
        sig = self._fact_signature(event_type, fact_data)
        if self._seen_recently(sig):
            return {"vector": "dedup", "graph": "dedup", "profile": "dedup", "skill": "dedup"}

        async def _vector():
            if not self.vector:
                results["vector"] = "skip"
                return
            text = fact_data.get("text") or fact_data.get("fact")
            if not text:
                results["vector"] = "skip"
                return
            meta = fact_data.get("metadata") or {
                "timestamp": get_utc_timestamp(),
                "type": event_type,
            }
            try:
                await asyncio.to_thread(self.vector.add, text, meta)
                results["vector"] = "ok"
            except Exception as e:
                results["vector"] = f"error: {e}"

        async def _graph():
            triplets = fact_data.get("triplets") or []
            if not self.graph or not triplets:
                results["graph"] = "skip"
                return
            try:
                added = await asyncio.to_thread(self.graph.add_triplets, triplets)
                results["graph"] = f"ok ({added})"
            except Exception as e:
                results["graph"] = f"error: {e}"

        async def _profile():
            update = fact_data.get("profile_update")
            if not self.profile or not update:
                results["profile"] = "skip"
                return
            try:
                await asyncio.to_thread(
                    self.profile.update,
                    update.get("category", "notes"),
                    update.get("key", "info"),
                    update.get("value", ""),
                )
                results["profile"] = "ok"
            except Exception as e:
                results["profile"] = f"error: {e}"

        async def _skill():
            lesson = fact_data.get("skill")
            if not self.skill or not lesson:
                results["skill"] = "skip"
                return
            try:
                await asyncio.to_thread(
                    self.skill.learn_lesson,
                    lesson.get("task"),
                    lesson.get("mistake"),
                    lesson.get("solution"),
                    memory_system=self.vector,
                )
                results["skill"] = "ok"
            except Exception as e:
                results["skill"] = f"error: {e}"

        await asyncio.gather(
            _vector(), _graph(), _profile(), _skill(),
            return_exceptions=True,
        )
        return results
