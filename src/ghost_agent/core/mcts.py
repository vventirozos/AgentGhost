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
from typing import Any, Dict, List, Optional

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
    """MCTS-style reasoning for action selection."""

    def __init__(self, llm_client: Any = None, max_candidates: int = 3,
                 max_depth: int = 2):
        self.llm_client = llm_client
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        # Cache of unexplored alternatives for backtracking
        self._backtrack_stack: List[List[ActionCandidate]] = []

    async def select_best_action(
        self,
        task: str,
        plan_state: str,
        available_tools: List[str],
        context: str = "",
    ) -> Optional[ActionCandidate]:
        """Generate candidates, simulate outcomes, return the best action.

        Returns None if no candidates could be generated.
        """
        # Step 1: Expand — generate candidate actions
        candidates = await self._expand(task, plan_state, available_tools, context)
        if not candidates:
            return None

        # Step 2: Simulate — predict outcomes in parallel
        scored = await self._simulate_parallel(candidates, context)

        # Step 3: Select — pick the best, cache alternatives for backtracking
        scored.sort(key=lambda c: c.score, reverse=True)
        winner = scored[0]
        winner.selected = True

        # Cache the runners-up for backtracking
        alternatives = [c for c in scored[1:] if c.score > 0.2]
        if alternatives:
            self._backtrack_stack.append(alternatives)

        logger.info(
            "MCTS: selected '%s' (score=%.2f) over %d alternatives",
            winner.description[:60], winner.score, len(alternatives),
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
