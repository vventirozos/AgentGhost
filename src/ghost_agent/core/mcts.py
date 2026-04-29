# src/ghost_agent/core/mcts.py
"""Monte Carlo Tree Search Reasoning.

Instead of single-path depth-first execution, this module generates
multiple candidate action paths, simulates their outcomes using
lightweight worker-node LLM calls, and selects the best path before
committing to real execution.

Flow:
  1. Generate N candidate next-actions (expansion)
  2. Simulate each path's outcome (cheap worker LLM call — no sandbox)
  3. Score each path (progress likelihood, cost, risk)
  4. Select and return the winning action for real execution

The tree is cached so that if the chosen path fails, the agent can
instantly backtrack to the next-best candidate.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # avoid a runtime cycle: prm imports nothing from core.
    from ..prm.features import PlanState
    from ..prm.scorer import PRMScorer

logger = logging.getLogger("GhostAgent")


@dataclass
class ActionCandidate:
    """A candidate next-action the agent could take."""
    description: str
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    simulated_outcome: str = ""
    score: float = 0.0  # 0.0 – 1.0 composite score
    risk_notes: str = ""
    selected: bool = False

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "tool_name": self.tool_name,
            "score": self.score,
            "risk_notes": self.risk_notes,
            "selected": self.selected,
        }


@dataclass
class MCTSNode:
    """A node in the search tree representing a state + action."""
    action: ActionCandidate
    depth: int = 0
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.visits if self.visits > 0 else 0.0


# ── Prompts ──────────────────────────────────────────────────────────

_EXPAND_PROMPT = """Given the current TASK, PLAN STATE, and available TOOLS, generate {n} distinct candidate next-actions. Each should represent a meaningfully different approach.

TASK:
{task}

CURRENT PLAN STATE:
{plan_state}

AVAILABLE TOOLS: {tools}

RECENT CONTEXT:
{context}

Return ONLY a JSON object:
{{
  "candidates": [
    {{
      "description": "What this action does and why",
      "tool_name": "which tool to use",
      "tool_args_summary": "key arguments",
      "expected_outcome": "what we expect to happen",
      "risk": "what could go wrong"
    }}
  ]
}}"""

_SIMULATE_PROMPT = """Predict the outcome of this ACTION in the current CONTEXT. Be realistic about what could go wrong.

ACTION:
{action}

CONTEXT:
{context}

Rate this action:
- progress: 0.0-1.0 (how much closer to the goal does this get us?)
- cost: 0.0-1.0 (0=free, 1=very expensive in tokens/time)
- risk: 0.0-1.0 (0=safe, 1=likely to fail or cause problems)

Return ONLY a JSON object:
{{
  "predicted_outcome": "one sentence",
  "progress": 0.0,
  "cost": 0.0,
  "risk": 0.0,
  "reasoning": "why these scores"
}}"""


class MCTSReasoner:
    """MCTS-style reasoning for action selection.

    Two scoring modes:

      * **LLM simulation** (default) — each candidate is rated by a
        worker-LLM call that predicts progress/cost/risk. Slow but
        general; the original behaviour.
      * **PRM scoring** (opt-in) — when a ``prm_scorer`` is wired AND
        the caller passes ``prm_state`` (a ``PlanState`` from the
        ``prm.features`` module), candidates are scored by the trained
        Process Reward Model in microseconds, no LLM round-trip. The
        scorer is fail-safe: if it has no trained model loaded, it
        returns a neutral 0.5 for every candidate, which would make
        all candidates tie and effectively defeat the purpose of
        MCTS — so callers should only pass ``prm_state`` when
        ``prm_scorer.has_model`` is true.

    Callers that don't pass ``prm_state`` get the legacy LLM-simulation
    path even when ``prm_scorer`` is attached. That preserves the
    existing public contract for callers that haven't been updated.
    """

    def __init__(self, llm_client: Any = None, max_candidates: int = 3,
                 max_depth: int = 2,
                 prm_scorer: Optional["PRMScorer"] = None):
        self.llm_client = llm_client
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        self.prm_scorer = prm_scorer
        # Cache of unexplored alternatives for backtracking
        self._backtrack_stack: List[List[ActionCandidate]] = []

    async def select_best_action(
        self,
        task: str,
        plan_state: str,
        available_tools: List[str],
        context: str = "",
        *,
        prm_state: Optional["PlanState"] = None,
    ) -> Optional[ActionCandidate]:
        """Generate candidates, score them, return the best action.

        When ``prm_state`` is provided AND a PRM scorer with a trained
        model is attached, candidates are scored by the PRM. Otherwise
        the legacy LLM-simulation path is used. Returns None if no
        candidates could be generated.
        """
        # Step 1: Expand — generate candidate actions
        candidates = await self._expand(task, plan_state, available_tools, context)
        if not candidates:
            return None

        # Step 2: Score — PRM fast path or LLM simulation
        if (
            prm_state is not None
            and self.prm_scorer is not None
            and self.prm_scorer.has_model
        ):
            scored = self._score_with_prm(candidates, prm_state)
            scoring_mode = "prm"
        else:
            scored = await self._simulate_parallel(candidates, context)
            scoring_mode = "sim"

        # Step 3: Select — pick the best, cache alternatives for backtracking
        scored.sort(key=lambda c: c.score, reverse=True)
        winner = scored[0]
        winner.selected = True

        # Cache the runners-up for backtracking
        alternatives = [c for c in scored[1:] if c.score > 0.2]
        if alternatives:
            self._backtrack_stack.append(alternatives)

        logger.info(
            "MCTS[%s]: selected '%s' (score=%.2f) over %d alternatives",
            scoring_mode, winner.description[:60], winner.score, len(alternatives),
        )
        return winner

    async def backtrack(self) -> Optional[ActionCandidate]:
        """Pop the next-best alternative from the backtrack stack.

        Called when the selected action fails, to try the next candidate
        without re-generating.
        """
        while self._backtrack_stack:
            alternatives = self._backtrack_stack[-1]
            if alternatives:
                candidate = alternatives.pop(0)
                candidate.selected = True
                logger.info(
                    "MCTS: backtracking to '%s' (score=%.2f)",
                    candidate.description[:60], candidate.score,
                )
                return candidate
            self._backtrack_stack.pop()
        return None

    def has_alternatives(self) -> bool:
        """Check if there are cached alternatives to backtrack to."""
        return any(alts for alts in self._backtrack_stack)

    def clear(self):
        """Reset the search tree."""
        self._backtrack_stack.clear()

    async def _expand(self, task: str, plan_state: str,
                      tools: List[str], context: str) -> List[ActionCandidate]:
        """Generate candidate next-actions using LLM."""
        if not self.llm_client:
            return []

        prompt = _EXPAND_PROMPT.format(
            n=self.max_candidates,
            task=task[:2000],
            plan_state=plan_state[:1000],
            tools=", ".join(tools[:20]),
            context=context[:2000],
        )

        try:
            result = await self.llm_client.chat_completion({
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1024,
                "stream": False,
            })
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            data = self._parse_json(text)
        except Exception as exc:
            logger.warning("MCTS expand failed: %s", exc)
            return []

        candidates = []
        for c in data.get("candidates", []):
            candidates.append(ActionCandidate(
                description=c.get("description", ""),
                tool_name=c.get("tool_name", ""),
                risk_notes=c.get("risk", ""),
            ))
        return candidates[:self.max_candidates]

    def _score_with_prm(
        self,
        candidates: List[ActionCandidate],
        prm_state: "PlanState",
    ) -> List[ActionCandidate]:
        """Score candidates using the attached PRM. Pure function: each
        ``ActionCandidate`` gets its ``score`` field populated and a
        short ``risk_notes`` string explaining the source.

        Falls back gracefully when the scorer raises — the candidate
        keeps a neutral 0.5 and the failure is logged at debug level.
        Plan-selection should never crash because the scorer hiccupped.

        Defensive clamp: even though the canonical ``PRMScorer`` returns
        scores already clamped to [0, 1], MCTS doesn't enforce that
        callers pass in a ``PRMScorer`` instance — any object with a
        ``score`` method is accepted (duck typing). A custom scorer
        returning NaN, -3.0, or 42.0 would otherwise propagate straight
        into the ranking and break the sort. Clamp here so MCTS is
        robust against any scorer implementation.
        """
        # Local import keeps `core` independent of `prm` at import
        # time — only the runtime path that USES PRM pays the import.
        from ..prm.features import ActionFeatures

        scorer = self.prm_scorer
        for candidate in candidates:
            try:
                action = ActionFeatures(
                    description=candidate.description,
                    tool_name=candidate.tool_name,
                    tool_args=candidate.tool_args or {},
                )
                raw = scorer.score(prm_state, action)
                candidate.score = _clamp_unit_score(raw)
                candidate.simulated_outcome = (
                    f"PRM: p(success)={candidate.score:.2f}"
                )
            except Exception as exc:
                logger.debug(
                    "MCTS PRM scoring failed for '%s': %s",
                    candidate.description[:40], exc,
                )
                candidate.score = 0.5
        return candidates

    async def _simulate_parallel(self, candidates: List[ActionCandidate],
                                 context: str) -> List[ActionCandidate]:
        """Simulate outcomes for all candidates in parallel."""
        if not self.llm_client:
            return candidates

        async def _sim(candidate: ActionCandidate) -> ActionCandidate:
            prompt = _SIMULATE_PROMPT.format(
                action=f"{candidate.description} (tool: {candidate.tool_name})",
                context=context[:2000],
            )
            try:
                # Use worker nodes if available (cheaper)
                route_fn = getattr(self.llm_client, "route", None)
                if route_fn:
                    result = await route_fn(
                        "MCTS_SIMULATE", {
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        max_tokens=256, temperature=0.2, fallback=None,
                    )
                else:
                    result = None

                if not result:
                    result = await self.llm_client.chat_completion({
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 256,
                        "stream": False,
                    })

                text = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                data = self._parse_json(text)
                progress = float(data.get("progress", 0.5))
                cost = float(data.get("cost", 0.5))
                risk = float(data.get("risk", 0.5))
                # Composite score: maximize progress, minimize cost and risk
                candidate.score = progress * 0.6 + (1 - cost) * 0.15 + (1 - risk) * 0.25
                candidate.simulated_outcome = data.get("predicted_outcome", "")
            except Exception as exc:
                logger.debug("MCTS simulate failed for '%s': %s",
                             candidate.description[:40], exc)
                candidate.score = 0.3  # Neutral default
            return candidate

        await asyncio.gather(*[_sim(c) for c in candidates])
        return candidates

    @staticmethod
    def _parse_json(text: str) -> dict:
        return _parse_json_static(text)


def _clamp_unit_score(x: Any) -> float:
    """Clamp arbitrary scorer output into a finite float in [0, 1].

    NaN / non-finite values become the neutral 0.5; out-of-range
    finite values are clipped. Mirrors ``prm.scorer._clamp_unit`` so
    MCTS is robust against any custom scorer implementation, not just
    the canonical ``PRMScorer``.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    import math as _m
    if not _m.isfinite(v):
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _parse_json_static(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
