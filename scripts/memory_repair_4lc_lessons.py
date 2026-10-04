"""§4LC one-off (2026-10-03, operator: "go with your recommendation"): the
lesson cleanup the three reviewers measured —
  A  27 of the 30 unverified rules the §4KW backfill (lesson_generalize_legacy_4kw)
     wrote as general lessons on 2026-10-02; KEPT: renaming a variable, answer a
     direct question first, switch tool after a repeated error
  B  private / third-party: the son's name and age, a Slack member's request
  C  advice to use a git tool that no longer exists
  D  taught by probe/bench traffic before the gates
  E  wrong one-off rules ("how much more likely" as an absolute difference; one
     web search per turn)
  F  self-play lessons recording no mistake ("None observed")
plus: the 33 request-scoped lessons' retrieval counters reset (inflated by probe
traffic), and the past-request store (auto_skills.json) moved aside so it is
re-mined from real requests only (`minable_requests`).
Deletions go through `SkillMemory.remove_by_trigger` (archived; twin removed).
Every trigger must exist and every KEPT one must survive, or nothing is
applied. Agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4lc_lessons.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
DELETE = {
    "A-backfill": [
        "Factual or domain-specific inquiries where precise details, regulations, or procedures are requested.",
        "Multi-step research or investigative requests where initial broad data collection has been completed and the next phase requires deep, targeted extraction or analysis from specific sources.",
        "When synthesizing answers from provided search results or external data snippets to address factual queries.",
        "When conducting research or planning for a task that includes a strict, specific constraint that significantly limits viable options.",
        "When a request requires navigating to multiple external websites to extract current pricing, specifications, or feature details.",
        "When a user requests an image generation task that requires preserving a specific person's likeness or identity from a provided reference image.",
        "When using a pre-existing or downloaded file as a reference or input for another tool or function that expects a specific, validated, or internally generated identifier.",
        "When generating visual content for a request that specifies a recognizable subject with a highly distinctive visual identifier or specific requested detail.",
        "When generating images that require highly specific physical details, precise states, or complex visual compositions.",
        "When generating an image using a reference photo for likeness, but the request asks for a highly specific, dynamic, or exaggerated pose, action, or emotion.",
        "When explaining a specific, unconventional, or niche physical process or technique.",
        "When a user asks for an explanation of a specific term, concept, or cultural practice in a particular language.",
        "When a request requires performing a calculation, transformation, or counting operation on data retrieved from an external source or system.",
        "When a user references a specific entity and asks for validation or opinion regarding a qualitative claim or reputation.",
        "Explaining idiomatic or colloquial expressions that contain personal pronouns, possessive markers, or non-literal components.",
        "When responding to queries containing obscure, misspelled, or potentially non-standard terms or entities.",
        "Requests that rely on implicit or context-dependent references requiring alignment with stored user data or external information.",
        "When an assistant produces a generated output based on a user's specific instructions and must confirm it meets those instructions.",
        "When a request asks for the current age, duration, or elapsed time based on a stored reference date and the present moment.",
        "When a prompt specifies a strict quantitative limit (such as a maximum word or character count) alongside a requirement for detailed or step-by-step content.",
        "When a user makes a multi-part request that combines a visual or file display task with a factual question about the implementation or tools used.",
        "When a user requests retrieval or interaction with an external service, API, or data resource that might be assumed to require authentication or restricted permissions.",
        "When researching or verifying information about historical figures, organizations, or groups, particularly when multiple similar entities or overlapping memberships exist.",
        "When generating a response or status update that relies on multiple internal data points, logs, or sections that must maintain consistency.",
        "When a request involves evaluating, comparing, or listing multiple options, institutions, or programs, or when a specific alternative is explicitly mentioned in the prompt.",
        "When querying internal memory, logs, or system context to answer questions about personal attributes or identity.",
        "When performing file operations on paths located within special or system-managed directories that may have restricted access or require specialized handling."
    ],
    "B-private": [
        "Fact check whether <@U56CVBHHQ> enjoys the company of young C programmers",
        "i am thinking signing thodoris for basketball, i was told that panerithraikos is a good team, what do you think ?"
    ],
    "C-dead-git-tool": [
        "When a file system operation targets a path within a known system or repository ",
        "When executing commands involving Git, verify the workspace is initialized as a ",
        "When a tool chain involves multiple sequential calls, ensure the prerequisite st"
    ],
    "D-probe-taught": [
        "Return a JSON object with keys name and year for the first person on the Moon.",
        "Build me a small tool in your sandbox: a Python script `logslow.py` that reads a ghost-agent log file given as argv[1], finds every line matching the request-finished marker 'request finished — +<N>s', and prints the 10 slowest request ids with their durations, slowest first, one per line as '<req_id> <seconds>s'. Requirements: handle a missing file with a clear error and exit code 2; ignore malfo",
        "In at most three words, explain step by step how Rayleigh scattering makes the sky blue at noon and why the same physics makes sunsets red and orange, including the role of path length through the atmosphere.",
        "Using the file system tool, count the lines in /Users/<user>/Data/AI/Agent/PROJECT_JOURNAL.md and reply with just the number."
    ],
    "E-wrong-rule": [
        "Interpreting 'how much more likely' in probability questions",
        "When performing sequential web searches, ensure each search is executed as a dis"
    ],
    "F-no-mistake": [
        "Producer-Consumer pattern with shared state aggregation",
        "Concurrent file scanning with early exit",
        "Concurrent producer/consumer with early exit",
        "Data cleaning and aggregation from CSV with multiple filtering criteria",
        "SQL aggregation with specific output formatting and sorting requirements",
        "Merging overlapping or adjacent time intervals from a file",
        "Data aggregation and filtering from multiple log files"
    ]
}
KEEP = [
    "When renaming a parameter or variable in a code file.",
    "When a user asks a direct, specific question, particularly a yes/no or factual query.",
    "When troubleshooting a task and a specific tool consistently fails with the same error."
]
COUNTERS = ("retrievals", "helpful_retrievals", "succeeded_retrievals", "failed_retrievals")


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _trig(row) -> str:
    return (row.get("trigger") or row.get("task") or "") if isinstance(row, dict) else ""


def check(playbook: list) -> list:
    """The triggers to delete, or SystemExit (nothing applied) when one is
    missing, a KEPT lesson is listed, or a kept one is not live."""
    live = {_trig(r).strip().lower() for r in playbook}
    keep = {k.strip().lower() for k in KEEP}
    todo = []
    for group, trigs in DELETE.items():
        for t in trigs:
            k = t.strip().lower()
            if k in keep:
                raise SystemExit(f"{group} lists a kept lesson {t[:60]!r} — nothing applied")
            if k not in live:
                raise SystemExit(f"{group}: {t[:60]!r} is not in the playbook — nothing applied")
            todo.append(t)
    for k in keep:
        if k not in live:
            raise SystemExit(f"kept lesson {k[:60]!r} is not live — nothing applied")
    return todo


def reset_counters(playbook: list) -> int:
    n = 0
    for r in playbook:
        if isinstance(r, dict) and r.get("scope") == "request":
            for c in COUNTERS:
                if r.get(c):
                    r[c] = 0
            n += 1
    return n


def main():
    raw = json.loads((MEM / "skills_playbook.json").read_text())
    playbook = raw if isinstance(raw, list) else (raw.get("playbook") or raw.get("lessons") or [])
    todo = check(playbook)
    print(f"delete {len(todo)} lessons (keep {len(KEEP)}), reset counters on "
          f"{sum(1 for r in playbook if isinstance(r, dict) and r.get('scope') == 'request')} request lessons, "
          f"move auto_skills.json aside{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4lc-lessons-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    sm = SkillMemory(MEM)
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in todo)
    with sm._get_lock():
        pb = sm._load_playbook()
        n = reset_counters(pb)
        sm._save_playbook_unlocked(pb)
    ask = MEM / "auto_skills.json"
    moved = ask.exists()
    if moved:
        ask.replace(MEM / f"auto_skills.json.pre-4lc-{stamp}")
    print(f"lessons removed {gone}/{len(todo)}; counters reset on {n}; past-request store moved aside: {moved}")


if __name__ == "__main__":
    main()
