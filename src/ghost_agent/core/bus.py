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
from typing import Any, Dict, List, Optional, Tuple

from ..utils.helpers import get_utc_timestamp
from ..memory.experience import HEADER as _EXPERIENCE_HEADER
from ..memory.fact_ledger import HEADER as _HISTORY_HEADER

logger = logging.getLogger("GhostAgent")

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
                 experience_memory: Any = None,
                 fact_ledger: Any = None):
        self.vector = vector_memory
        self.graph = graph_memory
        self.skill = skill_memory
        self.profile = profile_memory
        self.episodic = episodic_memory
        # §4QL: human-graded past owner turns (memory/experience.py) — hydrates
        # only behind GHOST_EXPERIENCE_TIER / the `experience` probe arm
        self.experience = experience_memory
        # §4QM: the owner's fact ledger — "since when / what before / your
        # words" lines on a first-person question, behind GHOST_FACT_HISTORY
        self.ledger = fact_ledger
        # Raw-conversation tier (2026-07-14): stored sessions were durable
        # but unreachable by retrieval — replay-only. Fifth hydration source.
        self.sessions = session_store
        # hydrate_context stashes this turn's surviving items here (read by
        # the finalize path's foreign-stash check in core/agent.py).
        self.last_hydration: Optional[Dict[str, Any]] = None
        # §4MJ: one stash PER TURN — a single slot let an overlapping turn's
        # hydration take the stash before the first turn's judge (staggered
        # up to 90 s) read it; 96 of 318 real turns had another hydrating
        # turn start inside that window
        self._hydration_by_turn: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
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
        # …and on the REQUEST (§4KZ): the LRU outlived the request it was
        # built for (the 9x loop was one request), so a value changed BACK in
        # a later turn (Athens → Berlin → Athens) was dropped as a repeat and
        # still reported SUCCESS.
        try:
            from ..utils.logging import request_id_context as _rid
            _req = str(_rid.get() or "")
        except Exception:  # noqa: BLE001
            _req = ""
        try:
            blob = json.dumps(
                {"event_type": event_type, "fact": stable, "request": _req},
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

    # Per-intent source weights for weighted RRF: classify the query and
    # boost the source most likely to have the answer. §4NN: restricted to
    # the two sources that hydrate a prompt (graph = the owner's facts,
    # skill = lessons; §4NM retired the vector/episodic/session tiers), and
    # the ONLY weights — the learned matrix (core/rrf_weights.py,
    # system/rrf/weights.json) was removed. A source not listed fuses at 1.0.
    # The values ARE the last learned ones (system/rrf/weights.json, 10-11):
    # §4NN review — equal contextual weights tied graph and skill, and the tie
    # put owner facts before lessons on ~41% of default-intent prompts.
    _INTENT_WEIGHTS = {
        # §4QL/§4QM: experience and history fuse at an explicit 1.0 (review:
        # an unlisted source defaulted to 1.0 silently)
        "factual":    {"graph": 2.03, "skill": 0.5, "experience": 1.0, "history": 1.0},
        "procedural": {"graph": 0.464, "skill": 1.969, "experience": 1.0, "history": 1.0},
        "contextual": {"graph": 0.927, "skill": 0.994, "experience": 1.0, "history": 1.0},
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

    # Marker the agent's legacy anaphora expansion spliced in front of a short
    # follow-up (§4NQ: the agent no longer expands; kept as a fallback for a
    # direct caller that passes such a string): ``Context: <prev assistant reply> | User intent: <question>``.
    # Classifying that whole string let ~200 chars of the ASSISTANT's prose
    # (dense in classifier keywords) pick the tier weights instead of the
    # user's 3-word question — a follow-up "and the docs?" came out
    # `procedural` purely from the previous reply (found 2026-07-22). The
    # explicit ``raw_user_text`` argument is the real fix; this marker is the
    # fallback for call sites that don't pass it yet.
    _EXPANSION_MARKER = "| User intent:"

    @classmethod
    def _intent_source_text(cls, query: str, raw_user_text: str = "") -> str:
        """The text intent should be classified on: the caller's raw user
        message when given, else the tail of a legacy expanded query, else
        the query itself (today's behaviour)."""
        if raw_user_text and str(raw_user_text).strip():
            return str(raw_user_text)
        q = str(query or "")
        idx = q.rfind(cls._EXPANSION_MARKER)
        if idx >= 0:
            tail = q[idx + len(cls._EXPANSION_MARKER):].strip()
            if tail:
                return tail
        return q

    @classmethod
    def _classify_query_intent(cls, query: str) -> str:
        """Cheap keyword classifier for query intent. Returns 'factual',
        'procedural', or 'contextual'."""
        if not query:
            return "contextual"
        # §4MJ: real tokens ("what?" was not "what"), and "how" is
        # procedural only as "how to/do/can/should/would…": "how are you",
        # "how are things" (greetings) and "how old/many/much…" (facts) came
        # out procedural — 33 of 36 procedural turns were not how-to, and
        # the weights then put failed-greeting episodes above the graph
        # (r2: contractions stay one token — "what's up" is a greeting, not
        # a "what" question)
        toks = re.findall(r"[a-z]+(?:'[a-z]+)?", query.lower().replace("\u2019", "'"))
        words = set(toks)
        if "how" in words:
            nxt = {toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "how"}
            words.discard("how")
            if nxt & {"to", "do", "does", "did", "can", "could", "should", "would", "i", "we"}:
                words.add("steps")
            elif nxt & {"old", "many", "much", "long", "far", "big", "tall", "often"}:
                words.add("what")
        words.discard("should")             # "should I drive or walk?" is advice, not a procedure
        factual_hits = len(words & cls._FACTUAL_KEYWORDS)
        procedural_hits = len(words & cls._PROCEDURAL_KEYWORDS)
        if factual_hits > procedural_hits and factual_hits > 0:
            return "factual"
        if procedural_hits > factual_hits and procedural_hits > 0:
            return "procedural"
        return "contextual"

    # ============================================================ HYDRATION

    # Fused-score denominator for the hydration path. 60 is the web-scale RRF
    # default, sized for lists of hundreds; over the 5-15 item lists this bus
    # fuses it flattens intra-tier spread to ~19% (1/61 vs 1/75) while the
    # cross-tier weight spread runs up to 20x — so fused order collapsed to
    # `tier_weight x consensus` and the vector store's own distance ranking was
    # discarded. k=10 restores a ~2.3x spread across a 15-item list, which is
    # the right order of magnitude against the weights (found 2026-07-22).
    # `_reciprocal_rank_fusion` keeps k=60 as ITS default for back-compat with
    # direct classmethod callers.
    _RRF_K = 10

    async def hydrate_context(self, query: str, *,
                              max_chars: int = 6000,
                              rrf_k: int = 0,
                              context_budget: int = 0,
                              turn_id: str = "",
                              raw_user_text: str = "",
                              exclude_session_id: str = "") -> str:
        """Concurrently query the hydrating tiers — the owner's facts (graph)
        and the lessons (skill); §4NM retired the vector, episodic and session
        tiers from prompts — and fuse the results with Reciprocal Rank Fusion.
        Returns a Markdown block (or an empty string when nothing is
        available). ``exclude_session_id`` is accepted for callers and unused
        since the session tier went.

        When ``context_budget`` > 0 it is the BASE budget and is honoured
        as-is for a simple query; a medium query gets 2x it (capped 9000) and
        a complex one 3x (capped 12000), so the documented scaling is actually
        reachable from the production caller's 4000.

        ``raw_user_text`` is the user's own message, before any anaphora
        expansion the caller spliced into ``query``. Intent is classified on
        it when supplied (see ``_intent_source_text``); omitting it keeps the
        legacy behaviour of classifying whatever ``query`` holds.

        §4NQ: no LLM decomposition. A request that lists two topics with
        "and also" / "as well as" is still split by keyword (free).
        """
        if not query or not str(query).strip():
            return ""

        # ⚠ Turn-scoped reset (§4L R2 NEW-3): `last_bus_triggers` was set
        # only on turns WITH skill survivors and never cleared, so a turn
        # without any inherited the previous turn's list into its
        # trajectory attribution stamp. Reset here — the start of the
        # turn's hydration — with the turn key that gates every consumer.
        try:
            if self.skill is not None:
                self.skill.last_bus_triggers = []
                from ..utils.logging import request_id_context
                self.skill._bus_delivered_turn_key = str(
                    request_id_context.get() or "")
        except Exception:
            pass

        # Adaptive budget: the caller's budget is the BASE and scales UP with
        # query complexity. The old form was `min(max(context_budget, 6000),
        # cap)`, which with the sole prod caller's context_budget=4000
        # evaluated to 6000 in ALL THREE branches — the adaptive path was dead
        # code, the docstring's 12000 unreachable, and the operator's
        # deliberate 4000 silently overridden (found 2026-07-22).
        if context_budget > 0:
            # §4MJ: the USER's words (the expanded "Context: … | User
            # intent:" string made every short follow-up "complex", 2-3×
            # budget — the §4LE class, fixed there for decomposition only)
            query_words = len(str(raw_user_text or query).split())
            if query_words > 30:
                max_chars = min(context_budget * 3, 12000)
            elif query_words > 15:
                max_chars = min(context_budget * 2, 9000)
            else:
                max_chars = context_budget

        # §4NQ: the Nova decomposition (RAG-Fusion) was removed — on 118 real
        # owner requests it changed the block on 20, mostly by adding
        # off-topic lessons, for 1-4 s before the first token. Only the free
        # keyword split remains.
        sub_queries = self._split_query(query)
        # §4FB (2026-09-06): the fan-out queries were invisible — one INFO
        # line per hydration so a lesson admitted by a sub-query is
        # attributable.
        if len(sub_queries) > 1:
            logger.info("memory bus sub-queries (%d): %s", len(sub_queries) - 1,
                        " | ".join(str(q)[:80] for q in sub_queries[1:]))

        # Fan-out retrieval for each sub-query in parallel. Request-scoped
        # lessons are judged against the USER's request, not the sub-query
        # (§4KW): `raw_user_text` when the caller gave it, else the query.
        _scope_request = str(raw_user_text or query or "")
        fetch_coros = []
        for _i, sq in enumerate(sub_queries):
            # the experience and history tiers are judged on the user's whole
            # request, so they are fetched ONCE (review: N identical embeds)
            fetch_coros.append(
                self._fetch_all_tiers(sq, scope_request=_scope_request, extras=(_i == 0)))
        tier_results_per_query = await asyncio.gather(*fetch_coros)

        # One ranked list PER (sub-query, tier). RRF's keyed accumulation
        # both dedups (same (source, text) key sums, not duplicates) and
        # rewards cross-sub-query consensus — an item found by all sub-
        # queries outranks one found by only the first. The previous
        # concatenate-then-dedup kept only the FIRST occurrence, nullifying
        # the consensus boost that is the point of RAG-Fusion, and ranked a
        # later sub-query's best hit below the first sub-query's worst
        # (found 2026-07-15).
        ranked_lists = [tier_list
                        for tier_lists in tier_results_per_query
                        for tier_list in tier_lists
                        if tier_list]

        intent = self._classify_query_intent(
            self._intent_source_text(query, raw_user_text))
        fused = self._reciprocal_rank_fusion(
            ranked_lists,
            k=(rrf_k if rrf_k > 0 else self._RRF_K), intent=intent,
            normalize_consensus=True,
        )
        out, survivors = self._format_markdown_with_survivors(fused, max_chars=max_chars)
        self._log_hydration(ranked_lists, survivors, intent, max_chars)
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
        # leak stale survivors into the next turn's judgment, and stamped
        # with the turn id so an overlapping turn's judge (or a turn that
        # skipped hydration) can't consume another turn's stash and
        # misattribute usefulness observations (found 2026-07-15).
        if survivors:
            self.last_hydration = {
                "intent": intent, "survivors": survivors, "ts": time.time(),
                "turn_id": str(turn_id or ""),
            }
            if turn_id:
                self._hydration_by_turn[str(turn_id)] = self.last_hydration
                while len(self._hydration_by_turn) > 32:
                    self._hydration_by_turn.popitem(last=False)
        else:
            # Survivor-less hydration: clear only OUR OWN (or an unstamped)
            # stash. Blindly writing None here destroyed an overlapping
            # turn's stash before its judge consumed it — the turn_id stamp
            # guarded consumption but not clobbering.
            prev = self.last_hydration
            if not (isinstance(prev, dict) and prev.get("turn_id")
                    and str(prev.get("turn_id")) != str(turn_id or "")):
                self.last_hydration = None
        return out

    # Short tags for the per-hydration instrumentation line, in fetch order.
    _TIER_TAGS = (("vector", "v"), ("graph", "g"), ("skill", "s"),
                  ("episodic", "e"), ("session", "sess"), ("experience", "x"), ("history", "h"))
    #: §4NM: tiers that no longer hydrate a prompt (recall still searches them)
    _RETIRED_TIERS = frozenset({"vector", "episodic", "session"})

    def _log_hydration(self, ranked_lists: List[List[Dict[str, Any]]],
                       survivors: List[Dict[str, Any]], intent: str,
                       max_chars: int) -> None:
        """One line per hydration: candidates and survivors per tier.

        There was NO signal when a tier went permanently empty — the two
        quiet-return paths and an unwired (``None``) store all returned ``[]``
        indistinguishably, and the live log carried zero fetch-failure lines,
        which is equally consistent with "all healthy" and "a tier has been
        dead for weeks". A dead tier now reads as ``x=0`` and an unwired one
        as ``x=-``. INFO so it lands in the file log without touching the
        WARNING+ console/pretty stream."""
        try:
            stores = {"vector": self.vector, "graph": self.graph,
                      "skill": self.skill, "episodic": self.episodic,
                      "session": self.sessions, "experience": self.experience,
                      "history": self.ledger}
            cands: Dict[str, set] = {}
            for tier_list in ranked_lists:
                for it in tier_list or ():
                    src = it.get("source") if isinstance(it, dict) else None
                    if src:
                        cands.setdefault(src, set()).add(it.get("text", ""))
            surv: Dict[str, int] = {}
            for it in survivors or ():
                src = it.get("source") or "?"
                surv[src] = surv.get(src, 0) + 1
            # §4NM: the retired tiers still read "-" (the line's shape is
            # what the operator reads); graph carries only the owner's facts
            got = " ".join(
                f"{tag}=" + ("-" if stores.get(name) is None
                             or name in self._RETIRED_TIERS
                             else str(len(cands.get(name) or ())))
                for name, tag in self._TIER_TAGS)
            kept = " ".join(f"{tag}{surv.get(name, 0)}"
                            for name, tag in self._TIER_TAGS)
            logger.info(
                "Hydration tiers: %s -> %d survivors (%s) intent=%s budget=%d",
                got, len(survivors or ()), kept, intent, max_chars)
        except Exception as e:  # instrumentation must never break hydration
            logger.debug(f"hydration instrumentation failed: {e}")

    def _credit_surfaced(self, survivors: List[Dict[str, Any]]) -> None:
        """One deduped retrieval-credit pass for the items injected this turn."""
        if not survivors:
            return
        # (§4NM: only lessons earn credit here — the episodic and vector tiers,
        # whose items carried episode / Chroma ids, no longer hydrate.)
        triggers = [it.get("trigger") for it in survivors
                    if it.get("source") == "skill" and it.get("trigger")]
        bulk = getattr(self.skill, "record_retrievals_bulk", None) if self.skill else None
        if triggers and callable(bulk):
            try:
                bulk(triggers)
                # Attribution stamp for the trajectory record (§4L
                # Lens-D MINOR-2: `hydrated_lessons` recorded only the
                # playbook surface — bus-tier lessons were invisible).
                # §4N MAJOR-1: stamp the REAL store, not the wrapper. On
                # self-play `self.skill` is dream's ReadOnlySkillMemory
                # whose `get_playbook_context` DELEGATES to `real_sm` and
                # reads the dedup keys off THAT object — so a stamp on the
                # wrapper was invisible to the read and the §4L delivery
                # dedup failed open on 89/89 self-play turns (the
                # highest-volume prompt path). The stamp is turn-key gated
                # at every reader, so routing it to the real store cannot
                # leak the sim's lessons into a later user turn.
                try:
                    _target = (getattr(self.skill, "real_sm", None)
                               or getattr(self.skill, "_real", None)
                               or self.skill)
                    _target.last_bus_triggers = list(triggers)
                    from ..utils.logging import request_id_context
                    _target._bus_delivered_turn_key = str(
                        request_id_context.get() or "")
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"skill record_retrievals_bulk failed: {e}")

    async def _fetch_all_tiers(self, query: str, scope_request: str = "", extras: bool = True):
        """Fetch from the hydrating memory tiers for a single query.
        ``extras``: the request-level tiers (experience, history) too — only
        the first sub-query asks for them."""
        # §4MJ r2: NO per-tier timeout. One was tried and reverted: a timed-
        # out tier abandons its to_thread work item, the thread stays blocked
        # on the store lock, and under a held lock 8 turns filled the
        # 14-worker default executor — every to_thread in the process then
        # stalled. Unbounded, only the turn waiting on the lock waits.
        # §4NM: two tiers hydrate a prompt — the owner's facts (graph) and the
        # lessons (skill). The vector, episodic and session tiers were off in
        # production (§4MJ / §4NC / §4NG: "off in prompts, keep recall") and
        # their fetchers were deleted; `recall` still searches all of them.
        async def _none():
            return []
        return await asyncio.gather(
            self._fetch_graph(query),
            self._fetch_skill(query, scope_request=scope_request),
            self._fetch_experience(scope_request or query) if extras else _none(),
            self._fetch_history(scope_request or query) if extras else _none(),
        )

    async def _fetch_history(self, request: str) -> List[Dict[str, Any]]:
        """§4QM: on a first-person question about the owner, the fact
        ledger's lines for the words of the question (when a fact was
        stated / replaced / removed, and the owner's own sentence). Off
        unless `GHOST_FACT_HISTORY=1` or the probe arm `fact_history`."""
        led = self.ledger
        if led is None or not self._asks_about_owner(str(request or "")):
            return []
        try:
            from ..memory.fact_ledger import fact_history_enabled
            if not fact_history_enabled():
                return []
            # §4LB: on-demand fields (the street address, descriptions) never ride
            # an every-turn prompt, and an address quoted in the owner's words of
            # another field is scrubbed the way every outbound text is
            excl, scrub, on_demand = set(), None, None
            prof = self.profile
            if prof is not None:
                try:
                    excl = set(prof.on_demand_fields() or ())
                    # the profile's WHOLE on-demand rule, not only the policy set
                    # (review r1: `*_description`, `home_address` rendered)
                    on_demand = lambda c, k, v, _p=prof, _f=excl: bool(_p.is_on_demand(c, k, _f, value=v))   # noqa: E731
                    pats, repl = prof.egress_scrubber()
                    if pats:
                        from ..memory.egress import scrub_text as _scrub
                        scrub = lambda ln, _p=pats, _r=repl: _scrub(ln, _p, _r)   # noqa: E731
                except Exception as e:  # noqa: BLE001
                    # fail CLOSED: without the scrubber no line leaves (review r1)
                    logger.warning(f"MemoryBus history: profile rules unavailable, no lines ({type(e).__name__}: {e})")
                    return []
            lines = await asyncio.to_thread(led.lines_for_question, str(request or ""), 8, excl, scrub, on_demand)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MemoryBus history fetch failed: {type(e).__name__}: {e}")
            return []
        return [{"source": "history", "text": ln} for ln in (lines or []) if ln]

    async def _fetch_experience(self, request: str) -> List[Dict[str, Any]]:
        """§4QL: how past OWNER turns like this one went — only turns a human
        graded (gold/census grade, the owner's thumb), at most one RIGHT case
        (its steps and sources) and one bad case (the grader's reason). Off
        unless `GHOST_EXPERIENCE_TIER=1` or the probe arm `experience`; judged
        on the USER's own request, never a sub-query."""
        store = self.experience
        if store is None:
            return []
        try:
            from ..memory.experience import experience_tier_enabled
            if not experience_tier_enabled():
                return []
            lines = await asyncio.to_thread(store.lines_for, str(request or ""))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MemoryBus experience fetch failed: {type(e).__name__}: {e}")
            return []
        return [{"source": "experience", "text": ln} for ln in (lines or []) if ln]
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

    @staticmethod
    def _split_query(query: str) -> List[str]:
        """The query plus, for a request that lists independent topics with
        "and also" / "as well as", its parts (each of 4+ words)."""
        sub_queries = [query]
        # Only split on " and also " / " as well as ", which clearly list
        # independent topics. (§4NQ: the "|" / "Context:" skip guarded the
        # agent's expanded string, which no longer exists — a real message
        # with a shell pipe is split like any other.)
        ql = query.lower()
        if " and also " in ql or " as well as " in ql:
            splitter = " and also " if " and also " in ql else " as well as "
            parts = ql.split(splitter)
            for part in parts[:2]:
                part = part.strip()
                if len(part.split()) >= 4:
                    sub_queries.append(part)

        return sub_queries[:4]

    # ------------------------------------------------------------- fetchers

    async def _owner_facts(self, query: str) -> List[str]:
        """The owner's facts asked for by their kind (§4KZ) — kept when the
        topical graph tier is off (§4NC r1: the measurement was on topical
        edges; "where do I live?" must still find the answer)."""
        _own = getattr(self.graph, "owner_facts_matching", None)
        if not (callable(_own) and self._asks_about_owner(query)):
            return []
        try:
            owned = await asyncio.to_thread(_own, query, 8)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"MemoryBus owner-fact fetch failed: {e}")
            return []
        return [e for e in owned if isinstance(e, str)] if isinstance(owned, list) else []

    async def _fetch_graph(self, query: str) -> List[Dict[str, Any]]:
        """§4NC/§4NM: the graph tier hydrates only the OWNER's facts asked for
        by their kind ("where do I live?"); the topical edges were 20%
        relevant / 16% harmful on 86 real owner requests and stay reachable
        through `recall` alone."""
        if not self.graph:
            return []
        return [{"source": "graph", "text": e} for e in await self._owner_facts(query) if e]

    # KB review 10-30 r2: NOT a bare "με" — in Greek it is mostly "with" ("ένα
    # σπίτι με κήπο" put the owner's address into a house-price question); the
    # object pronoun counts only before the verbs that take it about the owner
    _FIRST_PERSON = re.compile(r"\b(?:i|i'm|im|my|me|mine|myself|μου|μας|εγώ|είμαι|έχω|μένα|εμένα)\b"
                               r"|\bμε\s+(?:λένε|λενε|λέγανε|ξέρεις|ξερεις|θυμάσαι|θυμασαι|φωνάζουν)\b",
                               re.IGNORECASE)
    # KB review 10-30: Greek drops the pronoun ("πού μένω;", "πότε γεννήθηκα;") —
    # a first-person singular verb (-ω/-ώ, -ομαι/-ούμαι, past -ηκα, -σα) is the
    # owner too. r3 (measured on a 58-sentence probe list, then stopped — see
    # the review): not -σσα (θάλασσα, γλώσσα), and a CAPITALISED word after the
    # first is a name (Λάρισα, Φρόσω, Μαριγώ), never a verb; the words below end
    # like one.
    _GREEK_FIRST_PERSON_VERB = re.compile(
        r"(?<!\w)[α-ωάέήίόύώϊϋΐΰ]{2,}(?:ω|ώ|ομαι|ούμαι|ηκα|ήκα|(?<!σ)(?:ησα|ισα|ασα|εσα|(?<!ο)υσα|ωσα))(?!\w)")
    _NOT_A_VERB = frozenset({"εδώ", "κάτω", "πάνω", "έξω", "πίσω", "μέσω", "γύρω", "παρακάτω", "παραπάνω",
                             "επάνω", "αποκάτω", "αποπάνω", "ολόγυρα", "μπρος", "ηχώ", "κλειώ", "σαπφώ",
                             "λητώ", "ιώ", "αργώ", "καλυψώ", "πειθώ", "οκτώ", "οχτώ", "λίγο", "ίσα"})
    _QUESTION = re.compile(r"\?|^\s*(?:what|whats|what's|where|who|whom|which|when|how|do|does|did|am|is|"
                           r"are|have|has|tell|remind|τι|πού|που|ποι\w*|πόσ\w*|ποσ\w*|πότε|ποτε|πώς|πως|"
                           r"θυμάσαι|θυμασαι|πες|πείτε|πειτε|θύμισέ|θύμισε|θυμισε|ξέρεις|ξερεις)\b",
                           re.IGNORECASE)
    # the Greek question mark is ";" (U+037E, or a plain semicolon) — a question
    # only in Greek text; an English line may end in ";" ("my wife is Anna;")
    _GREEK_QMARK = re.compile(r"[;;]\s*$")
    _GREEK_LETTER = re.compile(r"[α-ωά-ώΑ-Ω]")

    @classmethod
    def _asks_about_owner(cls, query: str) -> bool:
        q = str(query or "")
        if not (cls._QUESTION.search(q)
                or (cls._GREEK_QMARK.search(q) and cls._GREEK_LETTER.search(q))):
            return False
        if cls._FIRST_PERSON.search(q):
            return True
        # the FIRST word may be capitalised and still a verb ("Μένω πού;")
        _q = re.sub(r"^\s*\w", lambda mm: mm.group(0).lower(), q, count=1)
        return any(m.group(0).lower() not in cls._NOT_A_VERB
                   for m in cls._GREEK_FIRST_PERSON_VERB.finditer(_q))

    async def _fetch_skill(self, query: str, scope_request: str = "") -> List[Dict[str, Any]]:
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
                    **({"scope_request": scope_request} if scope_request else {}),
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

    # ----------------------------------------------------------------- RRF

    @classmethod
    def _reciprocal_rank_fusion(cls, ranked_lists: List[List[Dict[str, Any]]],
                                k: int = 60,
                                intent: str = "contextual",
                                weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                                normalize_consensus: bool = False,
                                ) -> List[Tuple[Dict[str, Any], float]]:
        """Weighted RRF: score(d) = sum_r weight_r / (k + rank_r(d)).

        When ``intent`` is provided, source-specific weights scale each
        ranker's contribution so factual queries boost graph results and
        procedural queries boost skill results. ``weight_overrides`` (an
        intent→source matrix) supersedes ``_INTENT_WEIGHTS`` when supplied;
        production passes none (§4NN).

        ``normalize_consensus`` divides each source's accumulated scores by
        that source's REDUNDANCY across the sub-query lists (emissions ÷
        distinct items). RAG-Fusion's cross-sub-query consensus bonus is
        only meaningful for a tier that CAN return different items per
        sub-query: graph is keyword-seeded, so its variants return
        near-identical edges and every edge banked ~4x the reciprocal rank,
        while vector — by design — returns different items per sub-query and
        banked 1x. Simulated on live shapes, top-8 for `contextual` came out
        6 graph + 3 skill with the first vector item at fused position 9,
        and _PER_SOURCE_CAP then handed graph six slots at the head of the
        budget (found 2026-07-22). Dividing by redundancy makes each tier's
        maximum attainable score w/(k+1) regardless of how stable it is,
        while leaving INTRA-tier ordering (and therefore the genuine
        consensus reward) untouched. Off by default: direct classmethod
        callers keep the exact legacy arithmetic; ``hydrate_context``
        turns it on.
        """
        wmap = weight_overrides or cls._INTENT_WEIGHTS
        weights = wmap.get(intent, wmap.get("contextual", cls._INTENT_WEIGHTS["contextual"]))

        scores: Dict[Tuple[str, str], float] = {}
        index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        emissions: Dict[str, int] = {}
        distinct: Dict[str, set] = {}
        for ranker in ranked_lists:
            if not ranker:
                continue
            # `source` is REQUIRED. The old positional fallback keyed off the
            # ranked-list index, which stopped being meaningful once
            # hydrate_context began passing one list per (sub-query, tier) —
            # and it was self-inconsistent anyway: it read `.get("source")`
            # for the weight but `item["source"]` for the key, so a
            # source-less item raised KeyError out of hydrate_context and
            # cost the whole turn its context (found 2026-07-22).
            head = ranker[0] if isinstance(ranker[0], dict) else {}
            source_name = head.get("source")
            if not source_name:
                logger.debug("RRF: dropping ranked list with no source")
                continue
            w = weights.get(source_name, 1.0)
            for rank, item in enumerate(ranker):
                if not isinstance(item, dict):
                    continue
                src = item.get("source")
                if not src:
                    continue
                key = (src, item.get("text", ""))
                scores[key] = scores.get(key, 0.0) + w / (k + rank + 1)
                index[key] = item
                emissions[src] = emissions.get(src, 0) + 1
                distinct.setdefault(src, set()).add(key)

        if normalize_consensus:
            redundancy = {
                src: (emissions.get(src, 0) / len(keys)) if keys else 1.0
                for src, keys in distinct.items()
            }
            for key in scores:
                factor = redundancy.get(key[0], 1.0)
                if factor > 1.0:
                    scores[key] /= factor

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(index[key], score) for key, score in ordered]

    # ----------------------------------------------------------- formatting

    # Cross-tier relevance floor on the NORMALISED fused RRF score. Kept 0.0:
    # measured 2026-07-15 that this CANNOT reject an off-topic query — RRF
    # scores are a function of (rank, tier weight, intent) and discard the
    # underlying embedding distance, so normalising against the top item makes
    # the best (least-bad) match 1.0 whether the query is on- or off-topic
    # (an off-topic query's normalised scores are actually FLATTER). The real
    # off-topic gate was the vector tier's `_VECTOR_MATCH_FLOOR`, deleted with
    # that tier (§4NM). This constant survives as a relative-pruning knob
    # (tune UP to trim the low-relative tail).
    _RELEVANCE_FLOOR = 0.0

    # Per-source cap on emitted items. The primary ordering / inclusion is by
    # fused score under one global char budget (below), but this cap stops a
    # single tier (e.g. a heavy graph ego-dump) from monopolising the budget
    # when its fused scores happen to cluster at the top.
    _PER_SOURCE_CAP = 6

    # Below this many chars left, no item (let alone a header + item) is worth
    # emitting — the emission loop stops scanning rather than skipping items
    # one by one to the end of a long fused list.
    _MIN_EMITTABLE_CHARS = 8

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
            "experience": _EXPERIENCE_HEADER,
            "history": _HISTORY_HEADER,
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
        # ⚠ SELECTION follows the fused ranking; PRESENTATION groups by
        # source (§4L Lens-D MAJOR-3). Items used to append to one flat
        # list with lazily-emitted headers, so when sources interleaved
        # an item landed visually under the WRONG header — confirmed
        # live: three SKILL lessons rendered inside "### MEMORY
        # CONTEXT:" and were framed as memories instead of follow-these
        # rules. Budget accounting and inclusion order are unchanged;
        # only the final assembly groups each source's items under its
        # own header, sources in first-acceptance order.
        per_source_count: Dict[str, int] = {}
        emitted_headers: set = set()
        groups: Dict[str, List[str]] = {}
        source_order: List[str] = []

        def _emit(src: str, line: str) -> None:
            if src not in groups:
                groups[src] = []
                source_order.append(src)
            groups[src].append(line)

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
            if remaining <= MemoryBus._MIN_EMITTABLE_CHARS:
                break  # nothing meaningful can fit any more
            cost = len(text) + 1 + header_cost
            if cost > remaining:
                # Nothing emitted yet — truncate the single highest-ranked
                # item in place rather than returning empty context.
                if used == 0:
                    if header is not None and src not in emitted_headers:
                        emitted_headers.add(src)
                        used += len(header) + 1
                    # §4N B-MINOR-3: DO NOT grant the whole budget to one
                    # oversized head item and break — that silently evicted
                    # the entire SKILL PLAYBOOK + graph block (the 2026-07-22
                    # silent-vanish class, resurrected for the HEAD item;
                    # vector items carry no per-item cap upstream). Cap the
                    # head grant at half the budget and KEEP FILLING so the
                    # other tiers still land. If the head item is genuinely
                    # the only thing, the loop just ends naturally.
                    head_cap = max(MemoryBus._MIN_EMITTABLE_CHARS,
                                   (max_chars - used) // 2)
                    marker = " [...]"
                    body = text[:max(0, head_cap - len(marker))].rstrip()
                    _emit(src, body + marker)
                    survivors.append(item)
                    used += len(body) + len(marker) + 1
                    per_source_count[src] = per_source_count.get(src, 0) + 1
                    continue
                # Otherwise SKIP this item and keep filling. Items are ordered
                # by fused score, NOT by length, so the old `break` (comment:
                # "lower-ranked items can't fit either") was simply false:
                # reproduced 2026-07-22, a 900-char vector item at rank 3
                # discarded a 23-char SKILL LESSON and a 6-char graph edge
                # that both fit trivially in the remaining 277 chars, and the
                # entire SKILL PLAYBOOK and TOPOLOGICAL KNOWLEDGE GRAPH
                # sections vanished from the prompt — headers included,
                # silently, because they are emitted lazily on first sight.
                continue
            if header is not None and src not in emitted_headers:
                emitted_headers.add(src)
                used += len(header) + 1
            _emit(src, text)
            survivors.append(item)
            used += len(text) + 1
            per_source_count[src] = per_source_count.get(src, 0) + 1

        lines: List[str] = []
        for src in source_order:
            header = headers.get(src)
            if header is not None and src in emitted_headers:
                lines.append(header)
            lines.extend(groups[src])
        out = "\n".join(lines).strip()
        if len(out) > max_chars:
            out = out[:max_chars].rstrip() + "\n\n[... TRUNCATED]"
        return (out + "\n\n" if out else ""), (survivors if out else [])

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
                _type = str(meta.get("type") or "")
                smart = getattr(self.vector, "smart_update", None)
                if _type == "identity" and callable(smart):
                    # Identity facts must REPLACE their predecessor, not
                    # stack: the plain add() here left smart_update (and its
                    # whole subject/attribute conflict guard) with no
                    # production caller once the bus was always wired — every
                    # profile change minted a NEW row and the stale value
                    # kept being served alongside it (live: duplicate
                    # home_lab_worker_node facts, 5 coexisting paraphrases of
                    # one activity fact).
                    await asyncio.to_thread(smart, text, "identity")
                else:
                    _r = await asyncio.to_thread(self.vector.add, text, meta)
                    # ⚠ A REFUSED WRITE IS NOT AN "ok" (§4GK round 6). `add()`
                    # returned None on refusal AND on success, so a write the
                    # store declined — a duplicate-id collision that would have
                    # RECLASSIFIED an existing row's type — was reported to the
                    # caller as landed. Measured: an identity fact colliding
                    # with an existing `auto` row answered "SUCCESS: Profile
                    # updated" while the fact never reached the identity tier
                    # and the surviving row stayed eviction-eligible.
                    _refusals = tuple(getattr(self.vector, "ADD_REFUSALS", None) or ())
                    if _refusals and _r in _refusals:
                        # §4GK round 6: `_r` is already the store's own
                        # "refused: …" sentence, so prefixing produced
                        # "refused: refused: …" for the user — the doubled
                        # word is the tell that nobody read the producer and
                        # its classifier together.
                        results["vector"] = str(_r)
                        return results
                results["vector"] = "ok"
            except Exception as e:
                results["vector"] = f"error: {e}"

        async def _graph():
            triplets = fact_data.get("triplets") or []
            if not self.graph or not triplets:
                results["graph"] = "skip"
                return
            try:
                # §4M (Lens C MINOR): tombstone parity — the consolidation
                # path filters removal-shaped triplets ("user REMOVED x")
                # before they mint graph edges, but this fan-out (and thus
                # tool_remember's background extraction) had no such
                # filter, so a negation sentence became a positive edge.
                try:
                    from ..utils.helpers import is_removal_triplet
                    kept = [t for t in triplets if not is_removal_triplet(t)]
                except Exception:
                    kept = triplets
                if not kept:
                    results["graph"] = "skip (removal-shaped)"
                    return
                added = await asyncio.to_thread(self.graph.add_triplets, kept)
                results["graph"] = f"ok ({added})"
            except Exception as e:
                results["graph"] = f"error: {e}"

        async def _profile():
            update = fact_data.get("profile_update")
            if not self.profile or not update:
                results["profile"] = "skip"
                return
            # §4M (Lens C): `update.get` on a non-dict raised OUTSIDE the
            # try — gather(return_exceptions=True) swallowed it, the
            # report simply LACKED the "profile" key, and the tool
            # reported SUCCESS on a total drop. Malformed payloads must
            # surface as an error the failure sniffer can see.
            if not isinstance(update, dict):
                results["profile"] = (
                    f"error: malformed profile_update "
                    f"({type(update).__name__}, expected dict)")
                return
            # A well-formed profile write names BOTH category and key —
            # the old defaults ("notes"/"info") minted junk `<cat>.info` /
            # notes-drawer keys from malformed updates (same guard as the
            # smart-memory consolidation side, 2026-07-26).
            if not (update.get("category") and update.get("key")):
                results["profile"] = "skip"
                return
            try:
                _msg = await asyncio.to_thread(
                    self.profile.update,
                    update.get("category"),
                    update.get("key"),
                    update.get("value", ""),
                )
                # a refusal is RETURNED, not raised (read-degraded store)
                results["profile"] = (f"error: {_msg}" if isinstance(_msg, str)
                                      and _msg.lower().startswith("error") else "ok")
            except Exception as e:
                results["profile"] = f"error: {e}"

        async def _skill():
            lesson = fact_data.get("skill")
            if not self.skill or not lesson:
                results["skill"] = "skip"
                return
            try:
                # the model's own lesson (learn_skill): tagged, and a GENERAL
                # rule must not restate the request in hand (producers review:
                # "Count lines in a specific file path provided by the user.")
                from ..memory.lesson_scope import current_request as _cur_req
                _written = await asyncio.to_thread(
                    self.skill.learn_lesson,
                    lesson.get("task"),
                    lesson.get("mistake"),
                    lesson.get("solution"),
                    memory_system=self.vector,
                    source="learn_skill",
                    generality_context=str(_cur_req.get() or ""),
                )
                # `learn_lesson` returns "written"/"reinforced" on a real
                # write and None on every drop path (quality gate, dedup that
                # neither wrote nor bumped, internal exception it swallows
                # itself) — "so callers can tell a real write from a silent
                # drop", per its own docstring. Discarding it made this leg
                # report ok whenever nothing escaped, which made the caller's
                # failure branch unreachable.
                results["skill"] = ("ok" if _written
                                    else "error: lesson was not written "
                                         "(dropped by the playbook)")
                if _written:
                    # where it applies, per call (r8 review)
                    results["skill_scope"] = str(getattr(_written, "scope", "") or "general")
            except Exception as e:
                results["skill"] = f"error: {e}"

        await asyncio.gather(
            _vector(), _graph(), _profile(), _skill(),
            return_exceptions=True,
        )
        # §4M (Lens C MAJOR-3): per-target failures were report-only — the
        # PARTIAL state went to the model's tool result and was discarded
        # after the turn, leaving no durable record that a fan-out write
        # half-landed. No rollback exists (by design: best-effort targets),
        # so the log line IS the repair breadcrumb.
        # §4GK round 6: a REFUSED leg is a partial write too. This keyed on
        # "error" only, so the store declining a write — the one outcome that
        # needs the breadcrumb most, because nothing rolls back — produced no
        # warning at all. Same vocabulary the tool-side classifier uses.
        _errs = {k: v for k, v in results.items()
                 if isinstance(v, str) and v.startswith(("error", "refused"))}
        if _errs:
            logger.warning(
                "publish_fact partial write (%s): %s — other targets "
                "landed; no rollback", event_type, _errs)
        return results
