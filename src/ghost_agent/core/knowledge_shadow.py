"""Shadow correctness check for answers that ran NO tool (§4MF).

About 46% of real owner turns run no tool, so the verifier — which judges a
claim against TOOL evidence — has nothing to check and skips them. That is
where the §4ME review found the most dangerous answer: a database-migration
plan with the steps in an unsafe order and a function and two columns that
do not exist, "confirmed" by nobody.

This module asks the critic node a narrower question than the verifier: not
"is it supported by evidence" (there is none) but "is anything in it WRONG"
— a false statement, a named function / column / command / API that does not
exist, steps in an unsafe order, a contradiction with what the user supplied.

SHADOW ONLY. It writes ``system/verifier/knowledge_shadow.jsonl`` and nothing
else: no reply, no label, no lesson reads it. Its precision is measured
against later human signals before anything acts on it — the verifier's own
history (§4KK: 42 of 104 refutes were false) is why. Off-main: the critic or
worker node, never the user's slot. ``GHOST_KNOWLEDGE_SHADOW=0`` turns it off.
"""
from __future__ import annotations

from ..utils.json_store import open_append  # torn-tail-safe JSONL appends (§4MF)
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("GhostAgent")

SHADOW_REL = "system/verifier/knowledge_shadow.jsonl"
#: an answer shorter than this carries nothing worth a critic call
MIN_ANSWER_CHARS = 400
#: at most one judgement per this many seconds — the critic shares the Nova
#: host with the worker that routes the NEXT turn (§4MF review)
MIN_INTERVAL_S = 300.0
#: one judgement at a time — a burst of short turns never queues a backlog
_RUNNING = {"n": 0, "last": 0.0}

_PROMPT = """\
You are checking an answer for ERRORS. No tools were used to produce it, so \
judge it from your own knowledge — but flag only what you are confident is \
WRONG, never mere style, length or completeness.

Look for:
  - a factual statement that is false;
  - a named function, command, flag, API, setting, table column or file that \
does not exist (or has a different name);
  - steps in an order that would break or lose data;
  - a contradiction with what the user themselves supplied in the request.

Facts, names, plans or code the USER supplied are given — never an error.
If you are not confident something is wrong, do not list it.

REQUEST:
{request}

ANSWER:
{answer}

Reply with JSON only:
{{"verdict": "ok" | "errors" | "unsure", "issues": [{{"quote": "<short exact fragment of the answer>", "why": "<one sentence>"}}]}}"""


def enabled() -> bool:
    return os.environ.get("GHOST_KNOWLEDGE_SHADOW", "1").strip().lower() not in ("0", "false", "no", "off")


def eligible(request: str, answer: str) -> bool:
    """A substantive answer to a substantive request."""
    a, r = str(answer or "").strip(), str(request or "").strip()
    return enabled() and len(a) >= MIN_ANSWER_CHARS and len(re.findall(r"\w+", r)) >= 4


def _parse(text: str) -> Optional[dict]:
    t = str(text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(v, dict) or v.get("verdict") not in ("ok", "errors", "unsure"):
        return None
    issues = [i for i in (v.get("issues") or []) if isinstance(i, dict)]
    return {"verdict": v["verdict"], "issues": issues[:8]}


def record(row: dict, home: Optional[Path] = None) -> bool:
    try:
        base = Path(home) if home else Path(os.environ.get("GHOST_HOME", Path.home() / "Data" / "AI" / "Data"))
        p = base / SHADOW_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        with open_append(p) as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("knowledge shadow record failed: %s", e)
        return False


async def check_and_record(request: str, answer: str, llm_client: Any, *,
                           trajectory_id: str = "", req_id: str = "",
                           home: Optional[Path] = None) -> Optional[dict]:
    """Judge one no-tool answer and append the shadow row. Never raises;
    None when skipped (off, ineligible, busy, or no judgement came back)."""
    if llm_client is None or not eligible(request, answer) or _RUNNING["n"]:
        return None
    if time.time() - _RUNNING["last"] < MIN_INTERVAL_S:
        return None
    _RUNNING["n"] += 1
    _RUNNING["last"] = time.time()
    try:
        payload = {"messages": [{"role": "user", "content": _PROMPT.format(
                       request=str(request)[:4000], answer=str(answer)[:8000])}],
                   "temperature": 0.0, "max_tokens": 1200,
                   # the critic is a THINKING model: with thinking on it spent
                   # the whole budget reasoning and returned no content, so
                   # this check never produced a row (§4MF review)
                   "chat_template_kwargs": {"enable_thinking": False}}
        # off-main or not at all: a client that cannot take these keywords
        # is skipped, never retried bare on the user's slot
        res = await llm_client.chat_completion(
            payload, use_critic=True, is_background=True, off_main_only=True,
            timeout=120.0, task_label="knowledge-shadow")
        text = ((res or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")
        v = _parse(text)
        if v is None:
            # say WHY nothing was recorded — a silent None is how this check
            # would go inoperative unnoticed
            logger.info("knowledge shadow: no verdict parsed (%d chars returned) req=%s",
                        len(str(text or "")), req_id)
            return None
        row = {"ts": time.time(), "req_id": req_id, "trajectory_id": trajectory_id,
               "verdict": v["verdict"], "issues": v["issues"],
               "answer_chars": len(str(answer or ""))}
        record(row, home)
        return row
    except Exception as e:  # noqa: BLE001
        logger.debug("knowledge shadow skipped: %s", e)
        return None
    finally:
        _RUNNING["n"] -= 1
