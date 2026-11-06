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

SHADOW toward the owner. It writes ``system/verifier/knowledge_shadow.jsonl``;
no reply, label or lesson reads it. §4NH (operator: "yes, factual ones only"):
an "errors" verdict is ONE consumer's input — the failure replay's seed list
(read by the failure-replay picker until §4SZ removed it), which only picked a turn to REPLAY;
nothing is adopted without the owner. Its precision is measured
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
#: an answer shorter than this carries nothing worth a critic call…
MIN_ANSWER_CHARS = 400
#: …unless it states specific facts (§4NH: the census's invented ETF tickers
#: and PostgreSQL functions were in no-tool answers): a number with a decimal
#: or thousands mark, a version, a year, an all-caps ticker/acronym, a code
#: identifier (snake_case, call()), or a CamelCase name
MIN_FACTUAL_CHARS = 120
_SPECIFIC_RE = re.compile(r"\d+[.,]\d+|(?<!\w)(?:19|20)\d{2}(?!\w)|(?<![\w-])[A-Z]{3,5}(?![\w-])"
                          r"|[a-z]+_[a-z_]+|\w+\(\)|(?<!\w)[A-Z][a-z]+[A-Z]\w*")
#: at most one judgement per this many seconds — the critic shares the Nova
#: host with the worker that routes the NEXT turn (§4MF review); §4NH: 300 → 90
#: (it skipped the ETF follow-up 87 s after the first wrong answer)
MIN_INTERVAL_S = 90.0
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
    """§4OH (2026-10-13): OFF by default. The §4MF follow-up measured it on its
    7 live verdicts: 3 "errors", all false (valid `ALTER TABLE s.t SET SCHEMA`
    syntax called wrong, a correct age judged from a 2023 clock, a speculative
    tool-name doubt); both real knowledge errors in those answers (non-UCITS /
    mislabelled ETFs, PostgreSQL's SET SCHEMA privileges) passed as "ok" —
    precision 0/3, recall 0/2 — and an "errors" verdict seeds the failure
    replay, so it was feeding false failures into practice. `1` re-enables it
    for a new measurement (e.g. after a critic-model change)."""
    return os.environ.get("GHOST_KNOWLEDGE_SHADOW", "0").strip().lower() in ("1", "true", "yes", "on")


def eligible(request: str, answer: str) -> bool:
    """A substantive answer to a substantive request."""
    a, r = str(answer or "").strip(), str(request or "").strip()
    if not enabled() or len(re.findall(r"\w+", r)) < 4:
        return False
    return len(a) >= MIN_ANSWER_CHARS or (len(a) >= MIN_FACTUAL_CHARS
                                          and len(_SPECIFIC_RE.findall(a)) >= 2)


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
