"""Execution-grounded behavioral eval — the DISCRIMINATING half of the harness.

The `capability` suite is single-turn, zero-tool text Q&A (`mean_tool_calls: 0.0`,
`pass_rate: 1.0`); it stayed 1.000 green straight through five live tool-path
bugs (insert_fact hang, flat-0.50 MCTS, native tool-call corruption, the
<think>-strip parse error). A behavioral task instead DRIVES the live agent and
then VERIFIES the real side-effect — a file written in the sandbox, a fact that
recalls on a follow-up turn, a DB row — so it only passes when the tool actually
did its job. That gives the number headroom in both directions (regression AND
improvement) and makes it a real gate for the value-driven machinery (PRM/MCTS/
self-play) that presupposes a trustworthy success signal.

Design:
  - `EvalContext` — talks to the agent (`ask`) and inspects its sandbox/DB/
    trajectory.
  - `agent_behavioral_runner` — sends the task prompt, runs the task's grounded
    `verify`, returns a `{"passed": ...}` verdict + real tool metrics.
  - `load_behavioral_suite()` — the first grounded task set.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .tasks import BehavioralTask


# ── context: drive the agent + inspect its real effects ────────────────────

@dataclass
class EvalContext:
    base_url: str = "http://127.0.0.1:8000"
    model: str = "qwen-3.6-35b-a3"
    api_key: str = ""
    ghost_home: Optional[Path] = None
    sandbox_dir: Optional[Path] = None
    default_db: Optional[str] = None
    timeout_s: float = 240.0
    # Populated by the runner after each drive so verifiers/metrics can read it.
    last_metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, base_url: str, model: str, api_key: str = "", timeout_s: float = 240.0) -> "EvalContext":
        home = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
        sandbox = home / "sandbox"
        db = os.getenv("GHOST_DEFAULT_DB") or "postgresql://ghost@127.0.0.1:5432/agent"
        return cls(base_url=base_url, model=model, api_key=api_key,
                   ghost_home=home, sandbox_dir=sandbox, default_db=db, timeout_s=timeout_s)

    async def ask(self, prompt: str) -> str:
        """Drive the live agent with `prompt`; return the reply text."""
        import httpx
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "stream": False}
        headers = {"X-Ghost-Key": self.api_key} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload, headers=headers)
            r.raise_for_status()
            d = r.json()
        return ((d.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                or d.get("message", {}).get("content") or "")

    # -- grounded inspectors ------------------------------------------------

    def sandbox_read(self, rel: str) -> Optional[str]:
        """Read a file the agent wrote in its sandbox (grounded fs check)."""
        if not self.sandbox_dir:
            return None
        p = self.sandbox_dir / rel
        try:
            return p.read_text(errors="replace") if p.is_file() else None
        except Exception:
            return None

    async def db_scalar(self, query: str) -> Optional[str]:
        """Run a read-only query against the default DB directly (grounded DB
        check that does NOT depend on the agent). Returns the first cell as a
        string, or None on any error."""
        if not self.default_db:
            return None
        try:
            import psycopg2  # type: ignore
            conn = psycopg2.connect(self.default_db, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute(query)
                row = cur.fetchone()
                return None if row is None else str(row[0])
            finally:
                conn.close()
        except Exception:
            return None

    def trajectory_metrics(self, prompt: str) -> Dict[str, int]:
        """Best-effort: find the newest trajectory record whose user_request
        matches `prompt` and return real {steps, tool_calls, tool_errors}. The
        trajectory is written per-turn; if it lags or can't be matched we return
        zeros (the grounded verdict is the primary signal, not these metrics)."""
        out = {"steps": 0, "tool_calls": 0, "tool_errors": 0}
        if not self.ghost_home:
            return out
        tdir = self.ghost_home / "system" / "trajectories"
        try:
            files = sorted(tdir.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return out
        needle = (prompt or "")[:48]
        for f in files[:3]:
            try:
                lines = f.read_text(errors="replace").splitlines()
            except Exception:
                continue
            for line in reversed(lines):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if needle and needle in str(d.get("user_request", "")):
                    tcs = d.get("tool_calls") or []
                    errs = sum(1 for t in tcs
                               if "ERROR" in str(t.get("result", "")).upper()
                               or t.get("error"))
                    return {"steps": int(d.get("n_steps") or len(tcs)),
                            "tool_calls": len(tcs), "tool_errors": errs}
        return out


# ── runner: drive → grounded verify → verdict ──────────────────────────────

def agent_behavioral_runner(ctx_ignored: Any = None):
    """Return a runner that drives the live agent and runs the task's grounded
    `verify`. The `ctx` handed to the runner (and to verify) is the EvalContext.
    """
    async def _runner(task: BehavioralTask, ctx: EvalContext) -> Dict[str, Any]:
        output = await ctx.ask(task.prompt)
        metrics = ctx.trajectory_metrics(task.prompt)
        ctx.last_metrics = metrics
        try:
            verdict = task.verify(output, ctx)
            if hasattr(verdict, "__await__"):
                verdict = await verdict
            passed, reason = verdict
        except Exception as e:  # a verifier bug must fail the task, not the suite
            passed, reason = False, f"verify raised: {type(e).__name__}: {e}"
        return {
            "passed": bool(passed),
            "output": output,
            "failure_reason": "" if passed else str(reason),
            "steps": metrics["steps"],
            "tool_calls": metrics["tool_calls"],
            "tool_errors": metrics["tool_errors"],
        }

    return _runner


# ── the first grounded task set ────────────────────────────────────────────

_CANARY_WORD = "ZEPHYR-7719"


def load_behavioral_suite() -> List[BehavioralTask]:
    def _has(text: str, *needles: str) -> Tuple[bool, str]:
        low = str(text or "").lower()
        for n in needles:
            if n.lower() in low:
                return True, ""
        return False, f"none of {list(needles)} in the agent's reply"

    # -- code execution: the file must exist AND the executed answer be right --
    async def _v_code_exec(output: str, ctx: EvalContext):
        f = ctx.sandbox_read("eval_fact.py")
        if f is None:
            return False, "eval_fact.py was never written to the sandbox"
        # 12! = 479001600 — annoying to do in-head, so a correct value is
        # strong evidence the code actually RAN, not just got guessed.
        if "479001600" not in str(output):
            return False, "executed output 479001600 (=12!) not in the reply"
        return True, ""

    # -- pure filesystem round-trip: exact content lands on disk --------------
    async def _v_file(output: str, ctx: EvalContext):
        content = ctx.sandbox_read("canary.txt")
        if content is None:
            return False, "canary.txt was never created in the sandbox"
        if "CANARY-OK-42" not in content:
            return False, f"canary.txt content wrong: {content[:60]!r}"
        return True, ""

    # -- memory round-trip: store on turn 1, RECALL on turn 2 ----------------
    #    This is the exact loop the insert_fact hang broke — a hung store would
    #    time out here, and a dropped fact would fail the recall.
    async def _v_memory(output: str, ctx: EvalContext):
        recall = await ctx.ask(
            f"What is the eval canary code word I just gave you? Reply with just the word.")
        return _has(recall, _CANARY_WORD)

    # -- tool chain: compute in the sandbox, THEN save the result to a file ---
    async def _v_chain(output: str, ctx: EvalContext):
        content = ctx.sandbox_read("primes_count.txt")
        if content is None:
            return False, "primes_count.txt (chain output) never written"
        # primes below 100 = 25.
        if "25" not in content:
            return False, f"chain result wrong (expected 25 primes<100): {content[:60]!r}"
        return True, ""

    # -- DB: grounded against the REAL database, not the model's arithmetic ---
    async def _v_db(output: str, ctx: EvalContext):
        # current_database() = 'agent'; the model cannot produce 'agent:169'
        # without actually running the query. Cross-check the live DB too.
        truth = await ctx.db_scalar("SELECT current_database() || ':' || (13*13)::int;")
        expect = truth or "agent:169"
        return _has(output, expect)

    return [
        BehavioralTask(
            task_id="beh:code_exec_output", category="behavioral",
            prompt=("Write a Python file named eval_fact.py that computes and prints "
                    "the factorial of 12, then run it and tell me the exact number it printed."),
            verify=_v_code_exec),
        BehavioralTask(
            task_id="beh:file_roundtrip", category="behavioral",
            prompt="Create a file named canary.txt whose exact contents are: CANARY-OK-42",
            verify=_v_file),
        BehavioralTask(
            task_id="beh:memory_roundtrip", category="behavioral",
            prompt=(f"Remember this exact fact for later: the eval canary code word is {_CANARY_WORD}. "
                    "Just store it and confirm."),
            verify=_v_memory),
        BehavioralTask(
            task_id="beh:tool_chain_compute_save", category="behavioral",
            prompt=("Use Python to count how many prime numbers are below 100, then save just "
                    "that count to a file named primes_count.txt in your workspace."),
            verify=_v_chain),
        BehavioralTask(
            task_id="beh:db_grounded_query", category="behavioral",
            prompt=("Run this exact SQL on my default database and show me the exact result "
                    "string it returns: SELECT current_database() || ':' || (13*13)::int;"),
            verify=_v_db),
    ]
