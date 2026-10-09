"""§4KW one-off (2026-10-03): playbook repair after the review of the lesson
PRODUCERS. Rows named by EXACT trigger; any that does not resolve to exactly
one row aborts before anything is written.
  * RETRACT 10: seven self-play observations ("The solution correctly…",
    "None observed" mistakes — the new gate refuses them), one rule about the
    test harness, the dream rule minted from a self-play harness nudge ("When
    a final turn is reached…"), the distilled row carrying another pattern's
    fix.
  * REWRITE 4 dream rules (rewritten by r4/r5) into the "…, always …" form the
    write gate requires.
  * RE-KEY 29 dream/episode rows whose trigger no longer matches their fix (a
    merge replaced the fix): the trigger becomes the fix's own head, the dream
    convention (`task = trigger = rule[:80]`); a fresh twin each.
Run with the agent STOPPED; the memory dir is backed up first (without older
*.bak copies).
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_repair_4kw_r6.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
RETRACT = [
 "State reconciliation against historical logs",
 "Topological sort for critical path analysis in dependency graphs",
 "Stateful aggregation over sequential, delimited data",
 "Multi-file data aggregation using Python's csv module",
 "Multi-stage data aggregation and transformation across multiple CSV inputs",
 "Robust JSON parsing of semi-structured log files",
 "Stateful time-series sessionization",
 "Output formatting requirements in automated testing",
 "When a final turn is reached, always provide a direct answer instead of only too",
 "distilled(output_processing/python_general): Failure to validate input existence and data integrity before processing"
]
REWRITE = {
 "When a task requires a specific, constrained output (e.g., 'Reply with exactly X": {
  "solution": "When a task requires an exact reply (e.g., 'Reply with exactly X'), always make the final answer exactly that text, and call tools only when the request itself asks for a tool's result."
 },
 "When asked about a specific entity with a format or source constraint, follow both constraints": {
  "solution": "When answering questions about a specific entity, always follow both the requested format (e.g., one short sentence) and the requested source (e.g., only from your own records)."
 },
 "When building tools in the sandbox, ensure the source code is provided alongside": {
  "solution": "When building tools in the sandbox, always write the tool's source file before running the command that executes it."
 },
 "When searching for specific data, ensure the search query covers all required an": {
  "solution": "When searching for specific data, always make the queries cover every part the request names (for example both of two documents or both of two events) rather than focusing on a single aspect."
 }
}
REKEY = {
 "When responding to user queries, ensure the output language matches the input la": "When the user requests an answer in a language different from the input, ensure ",
 "When executing commands that reference local loopback addresses, confirm the san": "When executing system commands, verify the sandbox environment allows for the in",
 "When performing complex web searches that require navigation, implement a timeou": "When investigating a specific address, use a combination of web_search and brows",
 "When researching specific entities, cross-reference information from multiple so": "When investigating a specific person or organisation, cross-reference initial fi",
 "When searching for specific software versions, verify the requested version agai": "When performing web searches for specific software versions, ensure the search q",
 "When investigating external incidents, use both open-source and dark-web search ": "When investigating a specific incident or entity, use several tools (web_search,",
 "When analyzing data or generating reports, ensure the evidence gathered is suffi": "When analyzing data or making claims about specific incidents, ensure the eviden",
 "When a visualization tool fails to render data, verify that necessary IDs or sel": "When generating charts or visualizations, ensure the necessary data blocks or se",
 "When retrieving information from the knowledge base, specify the exact document ": "When querying knowledge bases, verify the exact file name before attempting to d",
 "When executing code, check for indentation errors, as they are a frequent cause ": "When executing code, ensure indentation is consistent across all lines to avoid ",
 "When a file is referenced for vision analysis, verify its existence using the fi": "When a vision tool provides a summary, use it as a baseline. If subsequent tool ",
 "When retrieving personal data, strictly adhere to constraints like 'from your re": "When answering questions about stored personal data, prioritize retrieving infor",
 "When performing web searches for specific items, check for zero results and cons": "When performing web searches requiring specific version or release data, verify ",
 "When executing a service management command, verify the service status before at": "When using the `manage_services` tool, confirm service binding and operation com",
 "When performing web scraping or data extraction, use specific CSS selectors (e.g": "When using the `browser` tool for content extraction, specify robust selectors (",
 "When a specific technical key is requested, perform multiple sequential web sear": "When performing sequential web searches, ensure each search is executed as a dis",
 "When executing code, check for and correct violations of assigned roles or stand": "When executing Python scripts, check for and address role violations (e.g., code",
 "When a tool call fails, attempt to use a different tool or re-execute the same t": "When using the browser tool, implement retry logic or check for tool-specific er",
 "When querying entity age, verify the calculation against stored birth dates befo": "When querying age, verify the stored birth date against the current date to calc",
 "When using `file_system` operations on files within a `.git` directory, use the ": "When a file system operation targets a path within a known system or repository ",
 "When initial file system operations fail, re-evaluate the execution sequence to ": "When performing file system operations, verify the success of the operation, esp",
 "When ingesting files, verify the file path or URL is correct, as the tool requir": "When using the file_system tool to read or check for files, verify the exact fil",
 "When performing sequential state management tasks, ensure all required actions (": "When a tool chain involves multiple sequential calls, ensure the prerequisite st",
 "When modifying files, ensure the remembered 'old_text' byte-matches the current ": "When attempting to modify or search for content in a file, verify that the remem",
 "When using the browser tool, check for and handle HTTP 403 (bot challenge) or ti": "When navigating to a URL using the browser tool, implement a retry mechanism or ",
 "When making targeted edits to a file, specify the exact function or line to modi": "When modifying a function in a file, specify the exact function name and the req",
 "When editing an existing image, use the exact filename provided in the previous ": "When referencing a previous image generation result, use the exact filename prov",
 "When performing file system operations involving replacement, always specify the": "When using the `file_system` tool to modify a file, always ensure the operation ",
 "When using file system tools, verify the exact file name before attempting opera": "When performing file system operations, verify the creation and content of the f"
}


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _key(x) -> str:
    return x.get("trigger") or x.get("task") or ""


def check(playbook: list) -> None:
    """Every named row resolves to exactly one row; every new trigger is new;
    every rewritten rule passes the write gate — or SystemExit."""
    from ghost_agent.memory.lesson_quality import is_actionable_lesson
    from ghost_agent.memory.skills import _normalize_trigger
    for group, names in (("retract", RETRACT), ("rewrite", list(REWRITE)), ("rekey", list(REKEY))):
        for t in names:
            n = sum(1 for x in playbook if _key(x) == t)
            if n != 1:
                raise SystemExit(f"{group}: {n} rows for {t[:70]!r} — nothing applied")
    taken = {_normalize_trigger(_key(x)) for x in playbook if _key(x) not in REKEY}
    for old, new in REKEY.items():
        if not new or _normalize_trigger(new) in taken:
            raise SystemExit(f"rekey: the new trigger for {old[:60]!r} collides — nothing applied")
        taken.add(_normalize_trigger(new))
    for old, new in REWRITE.items():
        if not is_actionable_lesson("none", new["solution"], old):
            raise SystemExit(f"rewrite still fails the write gate: {old[:60]!r}")


def main():
    playbook = json.loads((MEM / "skills_playbook.json").read_text(encoding="utf-8"))
    check(playbook)
    print(f"retract {len(RETRACT)}, rewrite {len(REWRITE)}, re-key {len(REKEY)}{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kw-r6-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.skills import SkillMemory, _delete_lesson_twin
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(MEM)
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in RETRACT)
    with sm._get_lock():
        pb = sm._load_playbook()
        for x in pb:
            k = _key(x)
            if k in REWRITE or k in REKEY:
                _delete_lesson_twin(vm, dict(x))
            if k in REWRITE:
                x["solution"] = x["correct_pattern"] = REWRITE[k]["solution"]
            if k in REKEY:
                x["trigger"] = x["task"] = REKEY[k]
        sm._save_playbook_unlocked(pb)
    healed = sm.heal_missing_twins(vm)
    print(f"retracted {gone}; rewritten {len(REWRITE)}; re-keyed {len(REKEY)}; twins written {healed}")


# §4MR: a store writer must be the ONLY writer — refuse beside the running
# agent (two chroma writers left the store segfaulting on open, §4MN)
if __name__ == "__main__" and "--apply" in __import__("sys").argv:
    import os as _os4mr, sys as _sys4mr
    from pathlib import Path as _P4mr
    _sys4mr.path.insert(0, str(_P4mr(__file__).resolve().parents[1] / "src"))
    from ghost_agent.memory.store_lock import assert_no_other_writer as _no_other_writer
    _no_other_writer(_P4mr(_os4mr.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory",
                     _P4mr(__file__).name)


if __name__ == "__main__":
    main()
