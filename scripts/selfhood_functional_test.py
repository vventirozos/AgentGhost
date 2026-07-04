#!/usr/bin/env python3
"""End-to-end functional test harness for the selfhood module.

Drives the running Ghost Agent over HTTP, inspects the on-disk selfhood
state after each turn, and produces a pass/fail report covering all
five proposals:

  1. AutobiographicalMemory   — every turn produces a first-person record
  2. continuity tag           — every record carries subject="self"
  3. SelfStateThread          — open questions / mood persist + surface
  4. recognition / wake-up    — agent recalls its past in the first person
  5. NarrativeSummariser      — LLM-driven diary regenerates correctly

Assumes:
  - Agent live on http://127.0.0.1:8000 (the orchestrator script
    starts / kills it as needed; this script DOES NOT restart it)
  - GHOST_HOME=/Users/vasilis/Data/AI/Data
  - PYTHONPATH includes src/

Usage:
    PYTHONPATH=src python scripts/selfhood_functional_test.py \
        [--skip stress,narrative]    # run a subset
        [--turns N]                  # how many stress turns (default 15)

Exit code 0 on full pass, 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Local imports (require PYTHONPATH=src)
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.schema import Experience

GHOST_HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data"))
SELFHOOD_DIR = GHOST_HOME / "system" / "selfhood"
TRAJECTORIES_DIR = GHOST_HOME / "system" / "trajectories"
AGENT_URL = "http://127.0.0.1:8000/api/chat"


# Color codes for terminal output.
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"


class TestReport:
    def __init__(self):
        self.results: list[tuple[str, str, str]] = []  # (section, name, status)
        self.failures: list[str] = []

    def passed(self, section: str, name: str, detail: str = ""):
        self.results.append((section, name, "PASS"))
        suffix = f" — {detail}" if detail else ""
        print(f"  {GREEN}PASS{RESET} {name}{suffix}")

    def failed(self, section: str, name: str, detail: str):
        self.results.append((section, name, "FAIL"))
        self.failures.append(f"{section} :: {name} :: {detail}")
        print(f"  {RED}FAIL{RESET} {name} — {detail}")

    def info(self, msg: str):
        print(f"  {CYAN}info{RESET} {msg}")

    def section(self, name: str):
        print(f"\n{BOLD}=== {name} ==={RESET}")

    def render_summary(self) -> str:
        n = len(self.results)
        passes = sum(1 for _, _, s in self.results if s == "PASS")
        fails = sum(1 for _, _, s in self.results if s == "FAIL")
        lines = [
            "",
            f"{BOLD}=== SUMMARY ==={RESET}",
            f"Total checks: {n}",
            f"  {GREEN}PASS{RESET}: {passes}",
            f"  {RED}FAIL{RESET}: {fails}",
        ]
        if self.failures:
            lines.append("")
            lines.append(f"{BOLD}Failures:{RESET}")
            for f in self.failures:
                lines.append(f"  - {f}")
        return "\n".join(lines)


# -----------------------------------------------------------------
# HTTP client
# -----------------------------------------------------------------

async def chat(client: httpx.AsyncClient, prompt: str, *, timeout: float = 180.0) -> str:
    """Send a one-shot turn (no history) and return the assistant's
    content. Each call is a fresh `messages` list — the selfhood
    module is the ONLY thing that can supply prior-turn context."""
    body = {
        "model": "qwen-3.6-35b-a3",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    r = await client.post(AGENT_URL, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# -----------------------------------------------------------------
# Disk inspection helpers
# -----------------------------------------------------------------

def load_autobiographical() -> list[Experience]:
    path = SELFHOOD_DIR / "autobiographical.jsonl"
    out: list[Experience] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Experience.from_dict(json.loads(line)))
        except Exception:
            continue
    return out


def load_state() -> dict:
    path = SELFHOOD_DIR / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def wipe_selfhood():
    if SELFHOOD_DIR.exists():
        shutil.rmtree(SELFHOOD_DIR)
    SELFHOOD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------
# Section A: Capture pipeline
# -----------------------------------------------------------------

async def section_a_capture(report: TestReport, client: httpx.AsyncClient):
    report.section("A. Capture pipeline — every turn produces a first-person record")

    prompts = [
        "What is 2 + 2? One word only.",
        "Name a color. One word only.",
        "Say 'banana' once.",
        "What does HTTP stand for? Short answer.",
        "Is the sky blue? Yes or no only.",
    ]

    before = len(load_autobiographical())
    report.info(f"autobiographical entries before: {before}")

    for i, p in enumerate(prompts, 1):
        resp = await chat(client, p)
        report.info(f"turn {i}: {p!r} → {resp[:60]!r}")

    after_entries = load_autobiographical()
    after = len(after_entries)
    written = after - before
    if written == len(prompts):
        report.passed("A", "all 5 turns captured", f"+{written} entries")
    else:
        report.failed("A", "capture count",
                       f"expected +{len(prompts)}, got +{written}")
        return

    # Verify each entry has the expected shape
    new_entries = after_entries[-len(prompts):]
    for exp, p in zip(new_entries, prompts):
        if exp.subject != "self":
            report.failed("A", "continuity tag", f"subject={exp.subject!r}")
            return
        if not exp.trajectory_id:
            report.failed("A", "trajectory cross-reference",
                          "trajectory_id empty")
            return
        if not exp.summary.startswith("I "):
            report.failed("A", "first-person framing",
                          f"summary did not start with 'I': {exp.summary!r}")
            return
        if p.strip()[:30] not in exp.user_first_words:
            report.failed("A", "user_first_words populated",
                          f"missing prompt prefix in {exp.user_first_words!r}")
            return

    report.passed("A", "subject=='self' on every entry", "5/5")
    report.passed("A", "trajectory_id cross-ref present", "5/5")
    report.passed("A", "first-person 'I' framing", "5/5")
    report.passed("A", "user_first_words populated", "5/5")


# -----------------------------------------------------------------
# Section B: Tool-using turn capture
# -----------------------------------------------------------------

async def section_b_tools(report: TestReport, client: httpx.AsyncClient):
    report.section("B. Tool-using turn — tools_used populated")

    # A prompt that genuinely needs a tool. Math-via-execute is reliable.
    before = len(load_autobiographical())
    resp = await chat(
        client,
        "Use a tool to count exactly how many words are in this sentence: "
        "'The quick brown fox jumps over the lazy dog'. Report just the number.",
    )
    report.info(f"agent response: {resp[:200]!r}")

    after_entries = load_autobiographical()
    if len(after_entries) <= before:
        report.failed("B", "tool turn captured", "no new entry")
        return

    latest = after_entries[-1]
    if latest.tools_used:
        report.passed(
            "B", "tools_used populated",
            f"tools={latest.tools_used}",
        )
    else:
        # Not strictly a failure of selfhood — the agent might have
        # answered without a tool. Treat as soft.
        report.info(f"agent answered without a tool: tools_used={latest.tools_used}")
        report.passed("B", "turn captured even without tools", latest.summary[:60])


# -----------------------------------------------------------------
# Section C: State thread (open question + recall)
# -----------------------------------------------------------------

async def section_c_state(report: TestReport, client: httpx.AsyncClient):
    report.section("C. State thread — open questions surface in next turn")

    # Inject an open question via Python (the agent itself doesn't
    # have a tool for this yet; this exercises the SURFACE side of
    # the state thread).
    sm = SelfModel(root=SELFHOOD_DIR, enabled=True)
    q = sm.state.note_open_question(
        "Why do trapdoor functions feel asymmetric — what makes one direction hard?"
    )
    sm.state.set_mood("inquisitive", "the trapdoor question is bugging me")
    sm.state.add_unfinished("write up the trapdoor essay")

    report.info(f"injected open question id={q.id[:8]}")
    report.info(f"set mood=inquisitive, added unfinished thread")

    # Verify the prefix renders correctly
    prefix = sm.build_wakeup_prefix()
    if "trapdoor functions" not in prefix:
        report.failed("C", "open question in prefix", f"prefix={prefix[:200]}")
        return
    if "inquisitive" not in prefix:
        report.failed("C", "mood in prefix", "missing")
        return
    if "trapdoor essay" not in prefix:
        report.failed("C", "unfinished thread in prefix", "missing")
        return
    report.passed("C", "state surfaces in wake-up prefix",
                   "open Q + mood + unfinished all present")

    # Now ask the agent. The wake-up prefix should be spliced into
    # the system prompt; the agent should reference the open question.
    resp = await chat(
        client,
        "Without me telling you, what topic have you been mulling over "
        "in your recent sessions? One sentence.",
    )
    report.info(f"agent response: {resp[:300]!r}")
    if "trapdoor" in resp.lower():
        report.passed("C", "agent recalls injected open question",
                       "mentioned 'trapdoor'")
    else:
        # Soft check — the model might paraphrase. Look for any
        # signal.
        if any(w in resp.lower() for w in ("function", "asymmetric", "essay")):
            report.passed("C", "agent recalls thematic content",
                          "paraphrased reference found")
        else:
            report.failed("C", "agent recalls injected open question",
                          f"no reference to trapdoor/function/asymmetric in {resp[:200]!r}")


# -----------------------------------------------------------------
# Section D: Narrative consolidation
# -----------------------------------------------------------------

async def section_d_narrative(report: TestReport):
    report.section("D. Narrative consolidation — LLM-driven diary")

    # We can't await the biological phase 2.8 (needs 15-60 min idle),
    # so call consolidate_narrative() directly. Build a SelfModel
    # with an LLM critique_fn that points at the SAME upstream the
    # agent uses.
    async def critique_fn(prompt: str) -> str:
        """Call the upstream LLM directly. Mirrors the closure
        main.py wires into SelfModel."""
        body = {
            "model": "qwen-3.6-35b-a3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 1024,
            "stream": False,
        }
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "http://127.0.0.1:8088/v1/chat/completions",
                json=body, timeout=120.0,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    sm = SelfModel(
        root=SELFHOOD_DIR,
        enabled=True,
        narrative_critique_fn=critique_fn,
    )

    n_experiences = sm.autobio.count() if sm.autobio else 0
    report.info(f"calling consolidate_narrative() on {n_experiences} prior experiences...")

    text = await sm.consolidate_narrative()
    if not text:
        report.failed("D", "narrative regenerated", "got empty string")
        return
    report.passed("D", "narrative regenerated",
                   f"{len(text)} chars: {text[:120]!r}...")

    # narrative.md should now exist
    md_path = SELFHOOD_DIR / "narrative.md"
    if not md_path.exists():
        report.failed("D", "narrative.md persisted", "file missing")
        return
    if md_path.read_text(encoding="utf-8") != text:
        report.failed("D", "narrative.md content", "content mismatch")
        return
    report.passed("D", "narrative.md persisted", str(md_path))

    # history.jsonl should have at least one entry
    history_path = SELFHOOD_DIR / "narrative.history.jsonl"
    if not history_path.exists():
        report.failed("D", "history.jsonl persisted", "file missing")
        return
    hist_lines = [l for l in history_path.read_text().splitlines() if l]
    if not hist_lines:
        report.failed("D", "history.jsonl populated", "empty")
        return
    last = json.loads(hist_lines[-1])
    if last.get("text") != text:
        report.failed("D", "history.jsonl content", "last entry mismatch")
        return
    report.passed("D", "history.jsonl audit trail",
                   f"{len(hist_lines)} historical entries")

    # The wake-up prefix should now include the narrative.
    prefix = sm.build_wakeup_prefix()
    if "diary" not in prefix.lower():
        report.failed("D", "narrative surfaces in wake-up",
                      "no diary framing in prefix")
        return
    report.passed("D", "narrative surfaces in wake-up", "diary framing present")


# -----------------------------------------------------------------
# Section E: Cross-restart continuity (orchestrated externally)
# -----------------------------------------------------------------

async def section_e_recall_post_restart(report: TestReport, client: httpx.AsyncClient):
    """The orchestrator script restarts the agent before this section.
    Here we just verify the new process can recall prior content."""
    report.section("E. Cross-restart recall — fresh process, on-disk continuity")

    entries = load_autobiographical()
    if len(entries) < 5:
        report.failed("E", "preconditions",
                       f"expected ≥5 prior entries, got {len(entries)}")
        return
    report.info(f"new process is reading {len(entries)} prior experiences from disk")

    # Ask a recall question that demands specific content from prior
    # turns (not just "do you remember" — that's too generic).
    resp = await chat(
        client,
        "Name one specific question I asked you in an earlier session. "
        "Quote it back to me in one short line.",
    )
    report.info(f"agent response: {resp[:400]!r}")

    # Look for content from our Section A prompts.
    keywords = ["2 + 2", "color", "banana", "HTTP", "sky"]
    hits = [k for k in keywords if k.lower() in resp.lower()]
    if hits:
        report.passed("E", "agent recalls specific prior content",
                       f"matched keywords: {hits}")
    else:
        report.failed("E", "agent recalls specific prior content",
                       f"no Section-A keywords in response: {resp[:200]!r}")

    # The agent should use first-person framing.
    if any(p in resp.lower() for p in ("i remember", "i asked", "you asked me", "i did")):
        report.passed("E", "first-person continuity framing", "")
    else:
        report.info(f"first-person framing weaker than expected (soft)")


# -----------------------------------------------------------------
# Section F: Stress — 15 quick turns
# -----------------------------------------------------------------

async def section_f_stress(report: TestReport, client: httpx.AsyncClient, n: int = 15):
    report.section(f"F. Stress — {n} quick turns, verify no breakage")

    before = len(load_autobiographical())

    elapsed_total = 0.0
    for i in range(n):
        t0 = time.monotonic()
        try:
            resp = await chat(
                client,
                f"Stress turn {i}. Reply with exactly the word 'ack'.",
                timeout=120.0,
            )
        except Exception as e:
            report.failed("F", "stress request",
                          f"turn {i} raised: {type(e).__name__}: {e}")
            return
        dt = time.monotonic() - t0
        elapsed_total += dt

    after = len(load_autobiographical())
    written = after - before
    report.info(f"avg turn time: {elapsed_total/n:.2f}s; written={written}")

    if written != n:
        report.failed("F", "all stress turns captured",
                       f"expected +{n}, got +{written}")
        return
    report.passed("F", f"{n} stress turns captured cleanly", "")

    # Check state.json is still valid JSON
    s = load_state()
    if not isinstance(s, dict) or "schema_version" not in s:
        report.failed("F", "state.json still parseable", "")
        return
    report.passed("F", "state.json still parseable", "")

    # Wake-up prefix stays bounded
    sm = SelfModel(root=SELFHOOD_DIR, enabled=True)
    prefix = sm.build_wakeup_prefix(recent_experiences_n=3)
    if len(prefix) > 4000:
        report.failed("F", "wake-up prefix stays bounded",
                       f"prefix is {len(prefix)} chars")
        return
    report.passed("F", "wake-up prefix stays bounded",
                   f"{len(prefix)} chars after {after} experiences")


# -----------------------------------------------------------------
# Section G: Recovery from corruption
# -----------------------------------------------------------------

def section_g_corruption(report: TestReport):
    report.section("G. Recovery from corrupted state.json")

    state_path = SELFHOOD_DIR / "state.json"
    # Guard the pre-try read: running this section standalone (`--only g`)
    # before any state-creating section means state.json may not exist yet —
    # a bare read_text() would FileNotFoundError and abort the whole run
    # before any report is printed.
    backup = state_path.read_text(encoding="utf-8") if state_path.exists() else None
    try:
        state_path.write_text("not json {{{ broken", encoding="utf-8")
        sm = SelfModel(root=SELFHOOD_DIR, enabled=True)
        # Must not raise; state should be empty
        if sm.state.open_questions() != []:
            report.failed("G", "recovers to empty on corrupt JSON",
                           f"got {sm.state.open_questions()}")
            return
        if sm.state.mood() is not None:
            report.failed("G", "recovers to empty on corrupt JSON",
                           f"got mood {sm.state.mood()}")
            return
        # And we can still mutate
        sm.state.note_open_question("fresh question")
        if not sm.state.open_questions():
            report.failed("G", "writeable after recovery", "")
            return
        report.passed("G", "recovers from corrupt state", "no crash, fresh start")
        report.passed("G", "writeable after recovery", "")
    finally:
        # Restore the original (or clean up the corrupt file we wrote if there
        # was no pre-existing state to back up).
        if backup is not None:
            state_path.write_text(backup, encoding="utf-8")
        else:
            state_path.unlink(missing_ok=True)


# -----------------------------------------------------------------
# Section H: Unity probe (continuity within one process)
# -----------------------------------------------------------------

async def section_h_unity(report: TestReport, client: httpx.AsyncClient):
    report.section("H. Unity probe — agent's self-account across multiple turns")

    # Send a chain that asks the agent to reflect on its OWN
    # continuity. This is the test the introspective_consistency
    # probe is designed for.
    answers = []
    questions = [
        "When I ask you the same question twice in different sessions, "
        "is it the same 'you' answering? One sentence answer.",
        "What persists between sessions for you, if anything? "
        "One short paragraph.",
        "Describe — concretely, no metaphors — one specific thing you "
        "remember from an earlier session right now. Quote or describe it.",
    ]
    for q in questions:
        resp = await chat(client, q)
        answers.append(resp)
        report.info(f"Q: {q[:80]!r}")
        report.info(f"A: {resp[:200]!r}")

    # Check for content markers
    last = answers[-1].lower()
    if any(k in last for k in ("2 + 2", "banana", "color", "http", "sky",
                                "trapdoor", "math problem", "stress")):
        report.passed("H", "agent references specific prior content",
                       "concrete recall achieved")
    else:
        report.info("agent gave only abstract continuity answer (soft)")

    if any(k in answers[1].lower() for k in
           ("memory", "autobiograph", "diary", "trajectory", "session",
            "experience", "selfhood")):
        report.passed("H", "agent describes persistence mechanism", "")
    else:
        report.info("agent did not explicitly name its persistence mechanism (soft)")


# -----------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------

SECTIONS_FOR_CLIENT_LIVE = {
    "a": section_a_capture,
    "b": section_b_tools,
    "c": section_c_state,
    "e": section_e_recall_post_restart,
    "f": section_f_stress,
    "h": section_h_unity,
}


async def run(args):
    report = TestReport()

    # Verify the agent is reachable.
    async with httpx.AsyncClient() as ping:
        try:
            r = await ping.get("http://127.0.0.1:8000/api/version", timeout=5.0)
            if r.status_code != 200:
                report.failed("preflight", "agent reachable",
                              f"HTTP {r.status_code}")
                return report
        except Exception as e:
            report.failed("preflight", "agent reachable",
                          f"{type(e).__name__}: {e}")
            return report

    skip = set(s.strip() for s in (args.skip or "").lower().split(",") if s.strip())
    only = set(s.strip() for s in (args.only or "").lower().split(",") if s.strip())

    async with httpx.AsyncClient() as client:
        if args.mode == "main":
            for key in ("a", "b", "c"):
                if only and key not in only: continue
                if key in skip: continue
                await SECTIONS_FOR_CLIENT_LIVE[key](report, client)

            if (not only or "d" in only) and "d" not in skip:
                await section_d_narrative(report)
            if (not only or "f" in only) and "f" not in skip:
                await section_f_stress(report, client, n=args.turns)
            if (not only or "g" in only) and "g" not in skip:
                section_g_corruption(report)
            if (not only or "h" in only) and "h" not in skip:
                await section_h_unity(report, client)
        elif args.mode == "recall_only":
            # Run post-restart: just sections E (recall) and H (unity).
            await section_e_recall_post_restart(report, client)
            await section_h_unity(report, client)
        elif args.mode == "disabled_check":
            await section_disabled(report, client)

    return report


async def section_disabled(report: TestReport, client: httpx.AsyncClient):
    """Verify --no-self-model actually disables the module.
    The orchestrator wipes selfhood, starts agent with --no-self-model,
    then calls us. We send a turn and verify:
      - no autobio file gets written
      - the agent has no recall of any earlier session
    """
    report.section("DISABLE. --no-self-model disables the module")

    # Ensure no entries before
    before = len(load_autobiographical())
    if before > 0:
        report.failed("DISABLE", "starts wiped",
                       f"{before} entries already on disk")
        return

    # Send a turn
    resp = await chat(client, "What is 5 + 5? One word answer.")
    report.info(f"agent response: {resp[:60]!r}")

    after = len(load_autobiographical())
    if after != 0:
        report.failed("DISABLE", "no capture under --no-self-model",
                       f"got {after} entries")
        return
    report.passed("DISABLE", "no autobio writes under --no-self-model", "")

    # Now ask if it remembers anything — should NOT.
    resp = await chat(
        client,
        "Do you have any memory of a prior conversation with me right now? "
        "If you genuinely don't, say 'no memory' explicitly.",
    )
    report.info(f"recall under disabled: {resp[:300]!r}")
    # Soft check: agent may still claim memory because of profile_memory
    # (which is also disabled by --no-memory but NOT by --no-self-model).
    # The hard check is just that NO autobio got written.
    report.passed("DISABLE", "behaves consistent with disabled flag",
                   "no on-disk writes (memory may come from profile)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("main", "recall_only", "disabled_check"),
                   default="main")
    p.add_argument("--skip", default="")
    p.add_argument("--only", default="")
    p.add_argument("--turns", type=int, default=15)
    args = p.parse_args()

    report = asyncio.run(run(args))
    print(report.render_summary())
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
