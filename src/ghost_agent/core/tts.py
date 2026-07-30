"""Trajectory-level test-time scaling (§4F Phase 3b) — adaptive
comparative best-of-N for final answers.

Fires only on the verifier's WOBBLE BAND — verdicts that neither confirm
convincingly nor refute actionably (UNCERTAIN, or REFUTED below the 0.7
action threshold). Hard-REFUTED turns keep the existing auto-repair path;
the two regeneration mechanisms never interact by construction.

Mechanism (the 2026 consensus shape, arXiv:2604.16529 / Agent S3):
  1. generate K alternative standalone final answers SEQUENTIALLY on the
     main model (single-slot box — parallel would double KV RAM);
  2. reduce each candidate to a compact excerpt;
  3. ONE list-wise comparative judge call (cheap pool — worker/critic),
     never per-candidate independent scores;
  4. substitute the winner; any failure anywhere keeps the original.

Everything is env-gated OFF by default (GHOST_TTS_ADAPTIVE_BON=1 to
enable) per the §3 doctrine: no unproven layer rides a live turn without
a measured win. Read per call so a bench can A/B via env.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

# Extra candidates generated on top of the original answer.
_BON_K_DEFAULT = 2
# Per-candidate excerpt budget for the judge prompt: keeps the list-wise
# call bounded even when candidates are long.
_EXCERPT_CHARS = 1400


def adaptive_bon_enabled() -> bool:
    return os.getenv("GHOST_TTS_ADAPTIVE_BON", "0").strip().lower() in (
        "1", "true", "yes", "on")


def bon_k() -> int:
    try:
        k = int(os.getenv("GHOST_TTS_BON_K", str(_BON_K_DEFAULT)))
    except ValueError:
        k = _BON_K_DEFAULT
    return max(1, min(4, k))


def wobble_band(verify_result: Any) -> bool:
    """True iff the verdict is in the band where extra compute pays:
    UNCERTAIN, or REFUTED that stayed below the 0.7 action threshold
    (so the repair path did NOT take it). None (no verdict) is NOT a
    trigger — absence of signal is not a signal."""
    if verify_result is None:
        return False
    verdict = str(getattr(verify_result, "verdict", "") or "")
    conf = float(getattr(verify_result, "confidence", 0.0) or 0.0)
    if verdict.endswith("UNCERTAIN"):
        return True
    return verdict.endswith("REFUTED") and conf < 0.7


def candidate_excerpt(text: str, budget: int = _EXCERPT_CHARS) -> str:
    """Deterministic compact view of one candidate: head + tail, code
    fences collapsed. No LLM call — the judge sees the substance of each
    answer, not a paraphrase of it."""
    t = re.sub(r"```.*?```", "[code block]", text or "", flags=re.DOTALL)
    t = t.strip()
    if len(t) <= budget:
        return t
    head = t[: budget * 2 // 3].rstrip()
    tail = t[-(budget // 3):].lstrip()
    return f"{head}\n[... truncated ...]\n{tail}"


def judge_payload(prompt: str) -> Dict[str, Any]:
    """Chat payload for the list-wise judge call. Lives HERE, not in
    agent.py: judges are verifier-class calls that must not think (same
    discipline as the verifier stages), while agent.py is guarded to
    never carry the disable-thinking switch outside its trivial
    fast-path (test_self_play_redesign guard) — the main agent's own
    reasoning turns keep thinking on, including BoN candidate
    generation."""
    return {
        "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stop": ["\n"],
    }


_JUDGE_PROMPT = """You are comparing {n} candidate replies to the same user request. Exactly ONE will be sent to the user. Judge them AGAINST EACH OTHER — cite concrete differences, do not score them independently.

USER REQUEST:
{request}

{candidates}

Pick the reply that is the most correct, most complete for the request, and free of unsupported claims. Penalize invented specifics, ignored constraints, and machine noise. Respond ONLY with a MINIFIED single-line JSON object, starting with {{ and no other text:
{{"winner": <candidate number 1-{n}>, "why": "one short sentence citing a concrete difference"}}"""


def build_judge_prompt(user_request: str, candidates: List[str]) -> str:
    blocks = []
    for i, c in enumerate(candidates, 1):
        blocks.append(f"CANDIDATE {i}:\n{candidate_excerpt(c)}")
    return _JUDGE_PROMPT.format(
        n=len(candidates),
        request=(user_request or "").strip()[:1500],
        candidates="\n\n".join(blocks))


def parse_judge_verdict(text: str, n: int) -> Optional[int]:
    """Winner as 0-based index, or None when unparseable/out-of-range.
    Never guesses: an unusable verdict must keep the original answer."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        w = int(data.get("winner"))
    except Exception:
        return None
    if 1 <= w <= n:
        return w - 1
    return None


async def adaptive_bon(
    generate_fn: Callable[[int], Awaitable[Optional[str]]],
    judge_fn: Callable[[str], Awaitable[str]],
    user_request: str,
    original: str,
    *,
    k: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run the best-of-N pass. Returns (winner_text, meta).

    `generate_fn(i)` produces the i-th alternative (1-based) or None on
    failure; failures shrink the pool instead of aborting. `judge_fn`
    takes the rendered judge prompt and returns raw judge text. The
    ORIGINAL answer is always candidate 1 — ties and failures resolve
    to it."""
    meta: Dict[str, Any] = {"fired": True, "substituted": False,
                            "candidates": 1, "winner": 0, "why": ""}
    candidates = [original]
    for i in range(1, (k if k is not None else bon_k()) + 1):
        try:
            alt = await generate_fn(i)
        except Exception as exc:
            logger.debug("BoN candidate %d failed: %s", i, exc)
            alt = None
        if alt and alt.strip() and alt.strip() != original.strip():
            candidates.append(alt.strip())
    meta["candidates"] = len(candidates)
    if len(candidates) < 2:
        meta["why"] = "no distinct alternatives generated"
        return original, meta

    try:
        raw = await judge_fn(build_judge_prompt(user_request, candidates))
    except Exception as exc:
        logger.debug("BoN judge call failed: %s", exc)
        meta["why"] = "judge call failed"
        return original, meta

    idx = parse_judge_verdict(raw or "", len(candidates))
    if idx is None:
        meta["why"] = "judge verdict unparseable"
        return original, meta
    meta["winner"] = idx
    try:
        parsed = json.loads(re.search(r"\{.*\}", raw, flags=re.DOTALL).group(0))
        meta["why"] = str(parsed.get("why", ""))[:200]
    except Exception:
        pass
    if idx == 0:
        return original, meta
    meta["substituted"] = True
    return candidates[idx], meta
