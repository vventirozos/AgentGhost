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
import re
import time
from collections import OrderedDict
from pathlib import Path
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
                 session_store: Any = None,
                 intent_weights: Any = None,
                 usefulness_ledger_path: Any = None):
        self.vector = vector_memory
        self.graph = graph_memory
        self.skill = skill_memory
        self.profile = profile_memory
        self.episodic = episodic_memory
        # Raw-conversation tier (2026-07-14): stored sessions were durable
        # but unreachable by retrieval — replay-only. Fifth hydration source.
        self.sessions = session_store
        # Post-turn usefulness feedback (2026-07-14): hydrate_context stashes
        # this turn's surviving items here; judge_hydration_usefulness
        # consumes it after the reply is known and appends (intent, source,
        # used) observations to the ledger that dream's RRF refit reads.
        self.last_hydration: Optional[Dict[str, Any]] = None
        self.usefulness_ledger_path = (
            Path(usefulness_ledger_path) if usefulness_ledger_path else None)
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
        """Stable hash of the event type + fact payload, used for dedup.

        Excludes ``metadata.timestamp`` — a fresh microsecond value on every
        call that otherwise makes every publish's signature unique, so the
        dedup LRU never fired (the exact 9x-repeat loop it exists to stop
        published all 9 times). Dedup must key on the fact CONTENT, not the
        write time."""
        stable = fact_data
        if isinstance(fact_data, dict) and isinstance(fact_data.get("metadata"), dict):
            stable = dict(fact_data)
            stable["metadata"] = {
                k: v for k, v in fact_data["metadata"].items() if k != "timestamp"
            }
        try:
            blob = json.dumps(
                {"event_type": event_type, "fact": stable},
                sort_keys=True,
                default=str,
            )
        except Exception:
            blob = f"{event_type}:{repr(stable)}"
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
        "factual":    {"graph": 2.0, "vector": 1.0, "skill": 0.5, "episodic": 0.3, "session": 0.8},
        "procedural": {"graph": 0.5, "vector": 1.0, "skill": 2.0, "episodic": 1.5, "session": 0.5},
        "contextual": {"graph": 1.0, "vector": 1.5, "skill": 1.0, "episodic": 1.0, "session": 1.2},
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

        # Adaptive budget: scale UP for complex queries — but never BELOW the
        # 6000 default. Flooring at the default is what makes this adaptive:
        # the sole prod caller passes context_budget=4000, so `min(4000, 12000)`
        # gave complex queries 4000 chars while simple ones kept 6000 — the
        # exact inversion the docstring says it prevents ("complex research
        # tasks starved"). max(context_budget, 6000) fixes the direction.
        if context_budget > 0:
            query_words = len(query.split())
            if query_words > 30:
                max_chars = min(max(context_budget, 6000), 12000)
            elif query_words > 15:
                max_chars = min(max(context_budget, 6000), 9000)
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
        combined_vector, combined_graph, combined_skill, combined_episodic, combined_session = [], [], [], [], []
        for vector_items, graph_items, skill_items, episodic_items, session_items in tier_results_per_query:
            combined_vector.extend(vector_items)
            combined_graph.extend(graph_items)
            combined_skill.extend(skill_items)
            combined_episodic.extend(episodic_items)
            combined_session.extend(session_items)

        # Deduplicate by text content
        combined_vector = self._dedup_items(combined_vector)
        combined_graph = self._dedup_items(combined_graph)
        combined_skill = self._dedup_items(combined_skill)
        combined_episodic = self._dedup_items(combined_episodic)
        combined_session = self._dedup_items(combined_session)

        intent = self._classify_query_intent(query)
        fused = self._reciprocal_rank_fusion(
            [combined_vector, combined_graph, combined_skill, combined_episodic, combined_session],
            k=rrf_k, intent=intent,
            weight_overrides=self._intent_weights,
        )
        out, survivors = self._format_markdown_with_survivors(fused, max_chars=max_chars)
        # Reinforcement credit AFTER fusion: only memories/lessons that
        # actually entered the prompt earn spaced-repetition / retrieval
        # credit. The fetchers deliberately skip their internal bumps
        # (search_items / get_playbook_items), so without this call the
        # loop would be credit-free; with it, credit is one deduped write
        # per store per turn instead of ~20 rewrites for candidates the
        # model never saw.
        try:
            await asyncio.to_thread(self._credit_surfaced, survivors)
        except Exception as e:
            logger.debug(f"MemoryBus surfaced-credit failed (non-critical): {e}")
        # Stash this turn's injected items for the post-turn usefulness
        # judge. Timestamped so a turn that dies before finalization can't
        # leak stale survivors into the next turn's judgment.
        self.last_hydration = (
            {"intent": intent, "survivors": survivors, "ts": time.time()}
            if survivors else None
        )
        return out

    async def judge_hydration_usefulness(self, reply: str, llm_client: Any,
                                         model_name: str = "default",
                                         max_age_s: float = 600.0) -> int:
        """Post-turn usefulness judge — closes the retrieval feedback loop.

        ``_credit_surfaced`` credits items for ENTERING the prompt, which is
        circular: popular memories get more popular whether or not they help.
        This runs OFF the critical path (spawned after the reply is final)
        and asks the worker which injected snippets the reply actually drew
        on, then:

        - vector items  → ``bump_helpful`` (double spaced-repetition credit);
        - skill items   → ``record_helpful_retrieval`` (feeds hit_rate/utility
          and the prune ranking);
        - every survivor → an ``(intent, source, used)`` observation appended
          to the ledger consumed by the dream cycle's RRF-weight refit — so
          the learned fusion matrix tracks real usefulness, not surfacing.

        Consumes ``self.last_hydration``. Returns the number of items judged
        used; never raises.
        """
        state, self.last_hydration = self.last_hydration, None
        if (not state or not state.get("survivors") or not reply
                or llm_client is None):
            return 0
        if time.time() - float(state.get("ts", 0)) > max_age_s:
            return 0  # stale stash from a turn that never finalized
        survivors = state["survivors"][:12]
        intent = state.get("intent", "contextual")

        numbered = "\n".join(
            f"{i + 1}. {str(s.get('text', ''))[:150]}"
            for i, s in enumerate(survivors)
        )
        prompt = (
            "Below are memory snippets that were injected into an assistant's "
            "context, followed by the assistant's final reply. Decide which "
            "snippets the reply ACTUALLY DREW ON (facts restated, lessons "
            "applied, past events referenced). Merely being on-topic is NOT "
            "enough; when unsure, exclude it.\n\n"
            f"SNIPPETS:\n{numbered}\n\n"
            f"REPLY:\n{str(reply)[:2000]}\n\n"
            'Return ONLY JSON: {"used": [<snippet numbers>]}'
        )
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a precise evaluator. Output JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
        }
        try:
            data = await llm_client.chat_completion(
                payload, use_worker=True, is_background=True, timeout=60.0,
                task_label="hydration-judge",
            )
            content = data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            logger.debug(f"hydration usefulness judge failed (skipped): {e}")
            return 0

        used_idx: set = set()
        try:
            parsed = json.loads(re.search(r"\{[\s\S]*\}", content).group())
            for num in parsed.get("used", []) or []:
                idx = int(num) - 1
                if 0 <= idx < len(survivors):
                    used_idx.add(idx)
        except Exception:
            return 0  # unparseable verdict → no credit, no observations

        used_vec = [s.get("mem_id") for i, s in enumerate(survivors)
                    if i in used_idx and s.get("source") == "vector" and s.get("mem_id")]
        bump_h = getattr(self.vector, "bump_helpful", None) if self.vector else None
        if used_vec and callable(bump_h):
            try:
                await asyncio.to_thread(bump_h, used_vec)
            except Exception as e:
                logger.debug(f"vector bump_helpful failed: {e}")
        rec_h = getattr(self.skill, "record_helpful_retrieval", None) if self.skill else None
        if callable(rec_h):
            for i, s in enumerate(survivors):
                if i in used_idx and s.get("source") == "skill" and s.get("trigger"):
                    try:
                        await asyncio.to_thread(rec_h, s["trigger"])
                    except Exception as e:
                        logger.debug(f"skill record_helpful_retrieval failed: {e}")

        if self.usefulness_ledger_path is not None:
            lines = "".join(
                json.dumps({
                    "intent": intent,
                    "source": s.get("source", "vector"),
                    "success": (i in used_idx),
                    "ts": get_utc_timestamp(),
                }) + "\n"
                for i, s in enumerate(survivors)
            )

            def _append():
                self.usefulness_ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.usefulness_ledger_path, "a", encoding="utf-8") as f:
                    f.write(lines)

            try:
                await asyncio.to_thread(_append)
            except Exception as e:
                logger.debug(f"usefulness ledger append failed: {e}")
        return len(used_idx)

    def _credit_surfaced(self, survivors: List[Dict[str, Any]]) -> None:
        """One deduped retrieval-credit pass for the items injected this turn."""
        if not survivors:
            return
        vec_ids = [it.get("mem_id") for it in survivors
                   if it.get("source") == "vector" and it.get("mem_id")]
        bump = getattr(self.vector, "bump_retrievals", None) if self.vector else None
        if vec_ids and callable(bump):
            try:
                bump(vec_ids)
            except Exception as e:
                logger.debug(f"vector bump_retrievals failed: {e}")
        triggers = [it.get("trigger") for it in survivors
                    if it.get("source") == "skill" and it.get("trigger")]
        bulk = getattr(self.skill, "record_retrievals_bulk", None) if self.skill else None
        if triggers and callable(bulk):
            try:
                bulk(triggers)
            except Exception as e:
                logger.debug(f"skill record_retrievals_bulk failed: {e}")

    async def _fetch_all_tiers(self, query: str):
        """Fetch from all memory tiers for a single query."""
        return await asyncio.gather(
            self._fetch_vector(query),
            self._fetch_graph(query),
            self._fetch_skill(query),
            self._fetch_episodic(query),
            self._fetch_session(query),
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
        # Prefer the per-item API: it carries the real Chroma id per item
        # (so the bus can credit ONLY what survives fusion — see
        # _credit_surfaced) and skips the in-search retrieval-stat bump
        # that credited every candidate of every sub-query. String-search
        # stubs in tests fall back to the legacy path.
        search_items = getattr(self.vector, "search_items", None)
        if callable(search_items):
            try:
                items = await asyncio.to_thread(search_items, query)
                return [
                    {"source": "vector", "text": it.get("text", ""), "mem_id": it.get("id")}
                    for it in (items or [])
                    if isinstance(it, dict) and it.get("text")
                ]
            except Exception as e:
                logger.warning(f"MemoryBus vector fetch failed: {type(e).__name__}: {e}")
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
        # Prefer the per-item API: one RRF item per LESSON. The old
        # whole-playbook blob was a single item — always rank 1 regardless
        # of per-lesson relevance, immune to the per-item relevance floor
        # and _PER_SOURCE_CAP, and duplicated (near-identically) by every
        # sub-query without collapsing in exact-text dedup. Per-item also
        # defers retrieval credit to _credit_surfaced (post-fusion).
        get_items = getattr(self.skill, "get_playbook_items", None)
        if callable(get_items):
            try:
                items = await asyncio.to_thread(
                    get_items, query, self.vector,
                )
                return [
                    {"source": "skill", "text": it.get("text", ""), "trigger": it.get("trigger", "")}
                    for it in (items or [])
                    if isinstance(it, dict) and it.get("text")
                ]
            except Exception as e:
                logger.warning(f"MemoryBus skill fetch failed: {type(e).__name__}: {e}")
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
            # vector_memory closes the semantic-recall loop: episodes are
            # ingested into the vector store expressly so this tier can
            # recall them by MEANING; without the kwarg search_similar
            # always fell back to substring matching over the 100 most
            # recent episodes, leaving ~half the collection unreachable.
            episodes = await asyncio.to_thread(
                self.episodic.search_similar, query, 5,
                self.vector if self.vector else None,
            )
        except TypeError:
            # Test stubs with a 2-arg search_similar signature.
            try:
                episodes = await asyncio.to_thread(
                    self.episodic.search_similar, query, 5
                )
            except Exception as e:
                logger.warning(f"MemoryBus episodic fetch failed: {type(e).__name__}: {e}")
                return []
        except Exception as e:
            logger.warning(f"MemoryBus episodic fetch failed: {type(e).__name__}: {e}")
            return []
        if not episodes:
            return []
        # One RRF item per EPISODE (same rationale as _fetch_skill).
        fmt_one = getattr(self.episodic, "format_episode", None)
        if callable(fmt_one):
            items = []
            for ep in episodes:
                try:
                    text = fmt_one(ep)
                except Exception:
                    continue
                if isinstance(text, str) and text.strip():
                    items.append({"source": "episodic", "text": text.strip()})
            if items:
                return items
            # Stub episodic stores (tests) expose a format_episode that
            # yields non-strings — fall through to the blob renderer.
        try:
            formatted = await asyncio.to_thread(
                self.episodic.format_for_context, episodes, 1500
            )
        except Exception:
            formatted = ""
        if not formatted or not formatted.strip():
            return []
        return [{"source": "episodic", "text": formatted.strip()}]

    async def _fetch_session(self, query: str) -> List[Dict[str, Any]]:
        """Raw-conversation tier: keyword hits from stored sessions.

        The lowest-abstraction tier (NapMem-style raw layer): durable
        server-side conversations were previously replay-only. One RRF item
        per matching message, prefixed with the session title so the model
        can tell WHICH past conversation it came from."""
        if not self.sessions:
            return []
        search = getattr(self.sessions, "search_messages", None)
        if not callable(search):
            return []
        try:
            hits = await asyncio.to_thread(search, query, 5)
        except Exception as e:
            logger.warning(f"MemoryBus session fetch failed: {type(e).__name__}: {e}")
            return []
        items = []
        for h in hits or []:
            if not isinstance(h, dict) or not h.get("text"):
                continue
            title = str(h.get("title") or "untitled")[:60]
            items.append({
                "source": "session",
                "text": f"[{title}] {h.get('role', 'user')}: {h['text']}",
                "session_id": h.get("session_id"),
            })
        return items

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
        # Positional fallback for a source-less ranked list. hydrate_context
        # passes FIVE lists (vector, graph, skill, episodic, session) — keep
        # this order in sync or a source-less item gets the wrong weight.
        _source_order = ["vector", "graph", "skill", "episodic", "session"]

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

    # Cross-tier relevance floor. Applied to the fused RRF score AFTER it is
    # normalised to [0,1] against the top-ranked item, so the floor is
    # meaningful regardless of rrf_k / weight magnitudes. Items below the
    # floor are dropped before injection so zero/low-signal context is never
    # spliced into a prompt unconditionally. Default 0.0 keeps every scored
    # item (a positive RRF score is, by construction, a real signal); the
    # constant exists so it can be tuned UP to prune aggressively without
    # touching the formatter.
    _RELEVANCE_FLOOR = 0.0

    # Per-source cap on emitted items. The primary ordering / inclusion is by
    # fused score under one global char budget (below), but this cap stops a
    # single tier (e.g. a heavy graph ego-dump) from monopolising the budget
    # when its fused scores happen to cluster at the top.
    _PER_SOURCE_CAP = 6

    @staticmethod
    def _format_markdown(fused: List[Tuple[Dict[str, Any], float]],
                         max_chars: int = 6000) -> str:
        out, _survivors = MemoryBus._format_markdown_with_survivors(
            fused, max_chars=max_chars)
        return out

    @staticmethod
    def _format_markdown_with_survivors(
            fused: List[Tuple[Dict[str, Any], float]],
            max_chars: int = 6000) -> Tuple[str, List[Dict[str, Any]]]:
        if not fused:
            return "", []

        headers = {
            "graph": "### TOPOLOGICAL KNOWLEDGE GRAPH:",
            "vector": "### MEMORY CONTEXT:",
            "skill": "### SKILL PLAYBOOK (lessons from prior runs — follow to avoid repeats):",
            # Episodes are now emitted one-per-item, so the anti-confusion
            # framing that used to live inside the tier blob rides the
            # section header instead.
            "episodic": "### PAST EPISODES (prior sessions — reference only, NOT the current conversation):",
            "session": "### PAST CONVERSATIONS (stored sessions — reference only, NOT the current conversation):",
        }

        # Normalise fused scores to [0,1] against the top-ranked item so the
        # cross-tier relevance floor is comparable across queries, then keep
        # only items at/above the floor. This is the single gate that stops
        # no-signal context being injected unconditionally.
        top = max((s for _it, s in fused), default=0.0)
        norm = top if top > 0 else 1.0
        gated = [(item, score) for item, score in fused
                 if (score / norm) >= MemoryBus._RELEVANCE_FLOOR]
        if not gated:
            return "", []

        # Emit STRICTLY in descending fused-score order (interleaved across
        # sources) under ONE global char budget. The previous implementation
        # re-grouped by source with fixed per-tier budgets, which discarded
        # the fused ranking entirely (intent / learned weights became
        # cosmetic). Source headers are emitted lazily on first sight, so a
        # source's items still cluster under its header while overall
        # inclusion order follows the fused ranking.
        per_source_count: Dict[str, int] = {}
        emitted_headers: set = set()
        lines: List[str] = []
        survivors: List[Dict[str, Any]] = []
        used = 0
        for item, _score in gated:  # already sorted desc by RRF
            src = item.get("source", "vector")
            text = item.get("text", "")
            if not text:
                continue
            if per_source_count.get(src, 0) >= MemoryBus._PER_SOURCE_CAP:
                continue
            header = headers.get(src)
            header_cost = 0
            if header is not None and src not in emitted_headers:
                header_cost = len(header) + 1
            remaining = max_chars - used
            if remaining <= 0:
                break
            cost = len(text) + 1 + header_cost
            if cost > remaining:
                # Nothing emitted yet — truncate the single highest-ranked
                # item in place rather than returning empty context;
                # otherwise stop, since lower-ranked items can't fit either.
                if used == 0:
                    if header is not None and src not in emitted_headers:
                        lines.append(header)
                        emitted_headers.add(src)
                        used += len(header) + 1
                    remaining = max_chars - used
                    lines.append(text[:max(0, remaining)].rstrip() + " [...]")
                    survivors.append(item)
                    used = max_chars
                break
            if header is not None and src not in emitted_headers:
                lines.append(header)
                emitted_headers.add(src)
                used += len(header) + 1
            lines.append(text)
            survivors.append(item)
            used += len(text) + 1
            per_source_count[src] = per_source_count.get(src, 0) + 1

        out = "\n".join(lines).strip()
        if len(out) > max_chars:
            out = out[:max_chars].rstrip() + "\n\n[... TRUNCATED]"
        return (out + "\n\n" if out else ""), (survivors if out else [])

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
        if not words:
            # Trivial query produced no meaningful terms. Fall back to the
            # raw tokens as-is rather than seeding "user" — the old "user"
            # seed made graph retrieval return the user ego-graph on every
            # turn regardless of topic.
            return [w for w in raw if w][:25]
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

        # Guard a None/non-dict payload up front — otherwise the first fan-out
        # coroutine's fact_data.get(...) raises AttributeError, which
        # gather(return_exceptions=True) swallows, so the caller can't tell the
        # write was dropped.
        if not isinstance(fact_data, dict):
            return {"error": f"publish_fact requires a dict payload, got {type(fact_data).__name__}"}

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
