"""Claude-driven conversational trainer for the Ghost Agent.

What this does
--------------
Drives the Ghost Agent's *interactive* learning loop using Claude as the
teacher/supervisor. For each task in a curriculum it:

  1. Asks the live agent the question (turn 1).
  2. Has Claude judge the answer against an authoritative reference.
  3. If wrong, sends a **correction** crafted to fire the agent's
     ``classify_user_correction`` predicate (BOTH the anchored
     correction-phrase signal AND the token-rephrase signal), which is
     what promotes the prior turn to FAILED and schedules a reflection
     that distils a lesson into ``SkillMemory``.
  4. After a short delay (so the fire-and-forget reflection lands), it
     re-asks the same question in a **fresh** conversation and has Claude
     score whether the agent retained the lesson.

Why the correction is built so carefully
----------------------------------------
The agent only learns from a correction when ``classify_user_correction``
(``src/ghost_agent/distill/user_correction.py``) returns
``is_correction=True``. That requires:

  * Signal A — an *anchored* correction phrase ("No, that's not right…")
    at the very start of the message.
  * Signal B — Jaccard overlap >= 0.40 between the prior user question
    and the correction message (the correction must restate the ask).

A plain "you're wrong, the answer is X" misses Signal B and silently
fails to trigger any learning. This trainer therefore restates the
original question inside the correction and, when the real classifier is
importable, **verifies the verdict fires before sending** — repairing the
message if it doesn't.

Requirements
------------
  * The live Ghost Agent reachable at ``GHOST_URL`` (default :8000).
  * ``ANTHROPIC_API_KEY`` for Claude (the teacher brain).
  * The official ``anthropic`` SDK + ``pydantic`` + ``httpx``. The repo's
    shared ``.agent.venv`` pins ``anthropic`` 0.4.1 (pre-Messages-API,
    required by evalplus/tau_bench), so this script runs from a dedicated
    ``.trainer.venv`` instead — see Usage. The ``Ghost`` client uses
    ``httpx`` directly: it speaks the local agent's OpenAI/Ollama API,
    not the Anthropic API.

Usage
-----
    # One-time: create the isolated venv with a modern anthropic SDK
    python -m venv .trainer.venv
    .trainer.venv/bin/pip install anthropic pydantic 'httpx[socks]'

    # Let Claude invent a 6-task curriculum on a topic and train on it:
    ANTHROPIC_API_KEY=sk-... \\
    .trainer.venv/bin/python scripts/claude_trainer.py --topic "PostgreSQL internals" --tasks 6

    # Train from a curriculum file (JSON list of {question, reference}):
    .trainer.venv/bin/python scripts/claude_trainer.py --curriculum mytasks.json

    # Skip retention re-test (faster):
    .trainer.venv/bin/python scripts/claude_trainer.py --topic "Python asyncio" --no-retention

Env (all overridable by flags):
    GHOST_URL          default http://127.0.0.1:8000
    GHOST_API_KEY      sent as both X-Ghost-Key and Bearer (auth-tolerant)
    GHOST_MODEL        default "qwen-3.6-35b-a3"
    ANTHROPIC_API_KEY  required
    CLAUDE_MODEL       default "claude-opus-4-8"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import anthropic
import httpx
from pydantic import BaseModel, Field

# --- Optional: import the agent's REAL correction classifier so we can
#     verify that every correction we send will actually trigger learning.
#     If the package isn't importable (run outside the repo), we fall back
#     to a conservative built-in check that mirrors its two signals.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

try:  # pragma: no cover - import path varies by environment
    from ghost_agent.distill.user_correction import (  # type: ignore
        classify_user_correction as _classify,
    )
    _HAVE_REAL_CLASSIFIER = True
except Exception:  # pragma: no cover
    _classify = None
    _HAVE_REAL_CLASSIFIER = False


# ====================================================================
# Console formatting — distinct, aligned section markers so the run is
# legible when interleaved with the agent's own pretty_log stream.
# ====================================================================

def _hr(char: str = "─", n: int = 64) -> str:
    return char * n


def banner(text: str) -> None:
    print(f"\n{_hr('═')}\n  {text}\n{_hr('═')}")


def section(tag: str, text: str = "") -> None:
    print(f"\n▸ {tag}" + (f"  {text}" if text else ""))


def detail(text: str) -> None:
    for line in text.splitlines() or [""]:
        print(f"    {line}")


# ====================================================================
# Correction-message construction (the crux of triggering learning)
# ====================================================================

# Anchored correction opener — matches Signal A of the agent's
# classify_user_correction (the leading "no" + "that's not right").
_OPENER = "No, that's not right."


def build_correction_message(question: str, teaching: str) -> str:
    """Build a correction message engineered to fire BOTH learning
    signals: an anchored correction phrase (Signal A) plus a restatement
    of the original question (Signal B, token overlap >= 0.40).

    Trigger-first: the teaching answer adds tokens that *dilute* the
    question-overlap, so for short questions appending the full answer can
    drop Jaccard below 0.40 and silently kill the learning trigger. We
    therefore include the teaching only when overlap still passes; if it
    would break the trigger we degrade to opener+question (learning still
    fires — the prior turn is still promoted to FAILED and reflected on —
    the answer just isn't embedded in the trigger turn).

    Pure-ish: uses the imported classifier only to choose the richest
    variant that still triggers. See ``tests/test_claude_trainer_correction.py``.
    """
    question = (question or "").strip()
    teaching = (teaching or "").strip()
    # opener+question maximizes overlap (extra tokens ≈ just "right");
    # this is the trigger-guaranteed core for any non-trivial question.
    core = f"{_OPENER} You misunderstood. {question}"
    rich = f"{core} The correct answer is: {teaching}" if teaching else core
    # Prefer the richer (answer-bearing) variant when it still triggers.
    if teaching and correction_will_trigger(question, "", rich):
        return rich
    return core


def correction_will_trigger(question: str, agent_answer: str, correction: str) -> bool:
    """Return True if the agent's real classifier (or the fallback)
    judges ``correction`` to be a learning-triggering correction of the
    prior turn."""
    if _HAVE_REAL_CLASSIFIER and _classify is not None:
        verdict = _classify(
            prev_user_request=question,
            prev_assistant_response=agent_answer,
            current_user_text=correction,
        )
        return bool(verdict.is_correction)
    return _fallback_correction_check(question, correction)


_FALLBACK_PHRASE_RE = re.compile(
    r"^\s*(?:no\b|nope\b|wrong\b|that[' ]?s\s+not|actually\b|incorrect\b)",
    re.IGNORECASE,
)
_FALLBACK_TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_FALLBACK_STOP = frozenset(
    "a an the of to for in on at by from with is are was were be the "
    "what which who where when why how this that and or but if".split()
)


def _fallback_tokens(text: str) -> set:
    return {
        t.lower()
        for t in _FALLBACK_TOK_RE.findall(text or "")
        if t.lower() not in _FALLBACK_STOP
    }


def _fallback_correction_check(question: str, correction: str) -> bool:
    if not _FALLBACK_PHRASE_RE.search((correction or "").lstrip()[:240]):
        return False
    q, c = _fallback_tokens(question), _fallback_tokens(correction)
    if not q or not c:
        return False
    inter, union = len(q & c), len(q | c)
    return union > 0 and (inter / union) >= 0.40


# ====================================================================
# Claude (teacher) client — official Anthropic SDK
# ====================================================================
#
# Per the claude-api skill, a Python project calls Claude through the
# official `anthropic` SDK, not raw HTTP. We use messages.parse() with
# Pydantic schemas for the teacher's structured outputs (the SDK enforces
# the schema and returns a validated object — no hand-rolled JSON parsing)
# and adaptive thinking, since grading correctness and authoring a
# curriculum both benefit from reasoning. Model defaults to claude-opus-4-8.


class _CurriculumTask(BaseModel):
    question: str
    reference: str = Field("", description="Authoritative correct answer.")
    domain: str = Field("", description="Sub-area / topic tag.")


class _Curriculum(BaseModel):
    tasks: List[_CurriculumTask]


class _Judgment(BaseModel):
    score: float = Field(description="Correctness 0.0-1.0 vs the reference.")
    teaching: str = Field(
        "",
        description="Authoritative correct answer when score < 0.7, else empty.",
    )
    rationale: str = Field("", description="One-sentence justification.")


class Claude:
    """Teacher brain — wraps the Anthropic SDK with parse()-based
    structured output and adaptive thinking."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        # anthropic.Anthropic() resolves ANTHROPIC_API_KEY from the env;
        # only pass api_key when one was supplied explicitly.
        try:
            self._client = (
                anthropic.Anthropic(api_key=api_key)
                if api_key
                else anthropic.Anthropic()
            )
        except anthropic.AnthropicError as e:
            raise SystemExit(
                f"ERROR: could not initialize the Anthropic client: {e}\n"
                "Set ANTHROPIC_API_KEY (or pass --anthropic-key)."
            )
        self.model = model

    def parse(self, *, system: str, user: str, schema, max_tokens: int = 4096):
        """Run one structured-output turn and return a validated ``schema``
        instance. ``messages.parse`` enforces the JSON schema server-side
        and parses the result, so there's nothing to hand-decode."""
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return resp.parsed_output

    def close(self) -> None:
        self._client.close()


# ====================================================================
# Ghost Agent (student) client
# ====================================================================

class Ghost:
    def __init__(self, url: str, api_key: str, model: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Ghost-Key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            headers=headers, timeout=httpx.Timeout(timeout, connect=10.0)
        )

    def chat(self, messages: list[dict]) -> str:
        """Send an OpenAI/Ollama-shaped chat turn; return assistant text.

        ``messages`` is the full conversation so far — the agent reads the
        prior assistant + user turns to run its correction classifier."""
        resp = self._client.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        # Both OpenAI ("choices") and Ollama ("message") shapes are emitted.
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return data.get("message", {}).get("content", "") or ""

    def close(self) -> None:
        self._client.close()


# ====================================================================
# Curriculum
# ====================================================================

@dataclass
class Task:
    question: str
    reference: str = ""  # authoritative answer (Claude fills if absent)
    domain: str = ""


def load_curriculum(path: str) -> list[Task]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["tasks"] if isinstance(raw, dict) else raw
    out: list[Task] = []
    for it in items:
        if isinstance(it, str):
            out.append(Task(question=it))
        elif isinstance(it, dict):
            out.append(
                Task(
                    question=it.get("question") or it.get("q") or "",
                    reference=it.get("reference") or it.get("answer") or "",
                    domain=it.get("domain", ""),
                )
            )
    return [t for t in out if t.question.strip()]


_CURRICULUM_SYS = (
    "You are a rigorous curriculum designer building a training set to "
    "probe and teach a local AI agent. Produce questions that have a "
    "single, verifiable, factual correct answer (not opinion), span easy "
    "to hard, and are likely to expose mistakes. For each, give the "
    "authoritative correct answer concisely."
)


def generate_curriculum(claude: Claude, topic: str, n: int) -> list[Task]:
    user = (
        f"Topic: {topic}\nProduce exactly {n} tasks, ordered easy to hard. "
        "Each task needs a single verifiable correct answer in `reference` "
        "and a `domain` sub-area tag."
    )
    curr = claude.parse(
        system=_CURRICULUM_SYS, user=user, schema=_Curriculum, max_tokens=8192
    )
    return [
        Task(question=t.question, reference=t.reference, domain=t.domain)
        for t in curr.tasks
        if t.question.strip()
    ]


# ====================================================================
# Judging
# ====================================================================

_JUDGE_SYS = (
    "You are a strict grader. Compare the STUDENT answer to the "
    "REFERENCE answer for the QUESTION. Score correctness 0.0-1.0 "
    "(1.0 = fully correct & complete; 0.0 = wrong/missing). If the "
    "student is not essentially correct (score < 0.7), provide a concise, "
    "authoritative 'teaching' string stating the correct answer plainly "
    "so it can be taught back. Be terse."
)


@dataclass
class Judgment:
    score: float
    correct: bool
    teaching: str = ""
    rationale: str = ""


def judge(claude: Claude, question: str, reference: str, answer: str) -> Judgment:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE:\n{reference or '(none provided — use your own knowledge)'}\n\n"
        f"STUDENT:\n{answer}"
    )
    j = claude.parse(system=_JUDGE_SYS, user=user, schema=_Judgment, max_tokens=4096)
    score = float(j.score)
    return Judgment(
        score=score,
        correct=score >= 0.7,
        teaching=(j.teaching or reference),
        rationale=j.rationale,
    )


# ====================================================================
# Per-task training
# ====================================================================

@dataclass
class TaskResult:
    question: str
    domain: str = ""
    first_score: float = 0.0
    first_correct: bool = False
    corrected: bool = False
    correction_triggered: bool = False
    post_correction_score: Optional[float] = None
    retained_score: Optional[float] = None
    retained_improved: Optional[bool] = None
    notes: list[str] = field(default_factory=list)


def train_one(
    claude: Claude,
    ghost: Ghost,
    task: Task,
    *,
    do_retention: bool,
    retention_delay: float,
) -> TaskResult:
    res = TaskResult(question=task.question, domain=task.domain)
    convo: list[dict] = [{"role": "user", "content": task.question}]

    # --- Turn 1: ask the agent ---------------------------------------
    section("ASK", task.question[:100])
    answer = ghost.chat(convo)
    convo.append({"role": "assistant", "content": answer})
    detail(f"agent → {answer[:240]}")

    j = judge(claude, task.question, task.reference, answer)
    res.first_score, res.first_correct = j.score, j.correct
    section("JUDGE", f"score={j.score:.2f} {'✅ correct' if j.correct else '❌ wrong'}")
    detail(j.rationale)

    if j.correct:
        res.notes.append("first-try correct; nothing to teach")
        return res

    # --- Correction: craft a message that fires BOTH signals ---------
    # build_correction_message already picks the richest variant that
    # still triggers; this re-check just records the verdict for the
    # report (a very short question can still fail Signal B).
    correction = build_correction_message(task.question, j.teaching or task.reference)
    triggers = correction_will_trigger(task.question, answer, correction)
    res.correction_triggered = triggers
    res.corrected = True

    section(
        "CORRECT",
        f"learning-trigger={'YES ✅' if triggers else 'NO ⚠️ (lesson may not be recorded)'}",
    )
    detail(f"teach: {(j.teaching or task.reference)[:200]}")
    if not triggers:
        res.notes.append(
            "correction did NOT satisfy classify_user_correction — the "
            "agent likely will not record a lesson from this turn"
        )

    # Send the correction. The agent promotes the prior turn to FAILED
    # and schedules a fire-and-forget reflection that writes a lesson.
    convo.append({"role": "user", "content": correction})
    follow = ghost.chat(convo)
    convo.append({"role": "assistant", "content": follow})
    detail(f"agent (post-correction) → {follow[:200]}")
    pj = judge(claude, task.question, task.reference, follow)
    res.post_correction_score = pj.score
    section("RE-JUDGE", f"in-context score={pj.score:.2f}")

    # --- Retention: fresh conversation after reflection lands --------
    if do_retention:
        section("WAIT", f"{retention_delay:.0f}s for reflection to distil a lesson…")
        time.sleep(retention_delay)
        section("RETEST", "fresh conversation, same question")
        fresh = ghost.chat([{"role": "user", "content": task.question}])
        detail(f"agent (fresh) → {fresh[:240]}")
        rj = judge(claude, task.question, task.reference, fresh)
        res.retained_score = rj.score
        res.retained_improved = rj.score > res.first_score + 0.05
        section(
            "RETENTION",
            f"first={res.first_score:.2f} → retained={rj.score:.2f} "
            f"{'📈 improved' if res.retained_improved else '➖ no gain'}",
        )

    return res


# ====================================================================
# Reporting
# ====================================================================

def summarize(results: list[TaskResult]) -> dict:
    n = len(results)
    first_ok = sum(1 for r in results if r.first_correct)
    triggered = sum(1 for r in results if r.correction_triggered)
    needed_teaching = sum(1 for r in results if r.corrected)
    retained = [r for r in results if r.retained_score is not None]
    improved = sum(1 for r in retained if r.retained_improved)
    return {
        "tasks": n,
        "first_try_correct": first_ok,
        "needed_teaching": needed_teaching,
        "corrections_that_triggered_learning": triggered,
        "retention_tested": len(retained),
        "retention_improved": improved,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Claude-driven trainer for the Ghost Agent")
    ap.add_argument("--topic", help="Topic for Claude to generate a curriculum")
    ap.add_argument("--curriculum", help="Path to a JSON curriculum file")
    ap.add_argument("--tasks", type=int, default=6, help="How many tasks to generate")
    ap.add_argument("--ghost-url", default=os.environ.get("GHOST_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--ghost-key", default=os.environ.get("GHOST_API_KEY", ""))
    ap.add_argument("--ghost-model", default=os.environ.get("GHOST_MODEL", "qwen-3.6-35b-a3"))
    ap.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    ap.add_argument("--claude-model", default=os.environ.get("CLAUDE_MODEL", "claude-opus-4-8"))
    ap.add_argument("--no-retention", action="store_true", help="Skip the fresh-conversation retest")
    ap.add_argument("--retention-delay", type=float, default=20.0, help="Seconds to wait before retest")
    ap.add_argument("--report", default="", help="Write a JSON report to this path")
    args = ap.parse_args(argv)

    if not args.topic and not args.curriculum:
        ap.error("provide --topic (Claude generates tasks) or --curriculum <file>")

    banner("GHOST AGENT · CLAUDE TRAINER")
    print(f"  student : {args.ghost_url}  model={args.ghost_model}")
    print(f"  teacher : Claude {args.claude_model}")
    print(f"  classifier: {'real (imported)' if _HAVE_REAL_CLASSIFIER else 'fallback heuristic'}")

    # Pass api_key only when supplied; otherwise the SDK reads ANTHROPIC_API_KEY.
    claude = Claude(args.claude_model, api_key=args.anthropic_key or None)
    ghost = Ghost(args.ghost_url, args.ghost_key, args.ghost_model)

    try:
        # quick liveness check on the student
        try:
            ghost._client.get(f"{ghost.url}/api/version").raise_for_status()
        except Exception as e:
            print(f"\n⚠️  Could not reach the agent at {ghost.url}: {e}")
            print("    Start it first (see run-and-test-setup); aborting.")
            return 2

        if args.curriculum:
            tasks = load_curriculum(args.curriculum)
            section("CURRICULUM", f"{len(tasks)} task(s) from {args.curriculum}")
        else:
            section("CURRICULUM", f"asking Claude for {args.tasks} task(s) on “{args.topic}”")
            tasks = generate_curriculum(claude, args.topic, args.tasks)
            for i, t in enumerate(tasks, 1):
                detail(f"{i}. [{t.domain}] {t.question}")

        results: list[TaskResult] = []
        for i, t in enumerate(tasks, 1):
            banner(f"TASK {i}/{len(tasks)}")
            try:
                results.append(
                    train_one(
                        claude,
                        ghost,
                        t,
                        do_retention=not args.no_retention,
                        retention_delay=args.retention_delay,
                    )
                )
            except httpx.HTTPError as e:
                print(f"  ✖ task failed (HTTP): {e}")
                results.append(TaskResult(question=t.question, notes=[f"http error: {e}"]))

        banner("SUMMARY")
        summary = summarize(results)
        for k, v in summary.items():
            print(f"  {k:<38} {v}")

        if args.report:
            Path(args.report).write_text(
                json.dumps(
                    {"summary": summary, "results": [asdict(r) for r in results]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\n  report written → {args.report}")
        return 0
    finally:
        claude.close()
        ghost.close()


if __name__ == "__main__":
    raise SystemExit(main())
