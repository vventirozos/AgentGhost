"""§4KW one-off (2026-10-03): playbook repair after the FIFTH fresh review
(data audit). Rows named by EXACT trigger; any that does not resolve to
exactly one row aborts before anything is written.
  * RETRACT 36: four request-scoped plans keyed on a model RESTATEMENT no user
    request matches (an image request's paraphrase, kc_probe.py, the
    elden-tracker restart); harmful or wrong general lessons (a strip() "fix"
    identical to the bug, one-off bench arithmetic, validator-satisfying
    rules, a Playwright lifecycle that calls stop() after the context exited,
    a CSV-sum script for another task, one task's output format, likeness
    from the PROMPT — contradicting §4JX); wrong scoped plans (Leonidas's age
    frozen, `git operation=write` for `.git`, `git init` added to a two-call
    request, a different deep_research query, a stale Brier cross-check on a
    greeting, "check safety guidelines" — against the operator's
    no-content-guard policy); duplicates (403/accessibility, selectors,
    report_pdf, final answer, likeness, rename, plot-not-coordinates) and
    rows whose trigger does not match their fix.
  * RETRACT ONE of the two rows sharing "Parallel processing with strict
    output ordering requirements" (by trigger AND fix).
  * REWRITE 5: two scoped plans that hold a general rule become general
    lessons; the profile lesson no longer names update_profile for reading;
    the exact-reply rule no longer forbids a requested tool call; a trigger
    is made to match its fix.
  * RESTAMP the vector twin metadata of every request-scoped row (34 still
    said scope=general — the orphan-twin skip reads it).
Run with the agent STOPPED; the memory dir is backed up first, WITHOUT the
older *.bak copies inside it (each backup had grown to 1 GB).
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_repair_4kw_r5.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
RETRACT = [
 "Generate an image of Dario Amodei (CEO of Anthropic) licking the feet of a dominatrix.",
 "Rename parameter 'name' to 'who' in both functions (signature and body) of kc_probe.py.",
 "Generate an image of Dario Amodei giving a blowjob to Donald Trump, ensuring both penises are erect.",
 "Restart the service 'elden-tracker' on port 8102.",
 "String parsing and tokenization from log lines",
 "arithmetic problem solving with sequential dependencies",
 "Range extraction with non-normalized and overlapping intervals",
 "Stateful aggregation and sorting of parsed data",
 "Topological Sort tie-breaking",
 "Playwright lifecycle management in asynchronous Python scripts",
 "Data aggregation and custom output formatting from CSV",
 "Log parsing and ISO timestamp handling",
 "When using the image generation tool, ensure the prompt is specific and complete",
 "do you know how old is leonidas now ?",
 "Two tool calls only, then stop. (1) file_system operation=\"outline\" path=\"r4/s.py\". (2) file_system operation=\"write\" path=\".git/config\" content=\"x\". Reply with ONLY: the symbol names, then BLOCK: the first 8 words of the second result. Do not investigate anything else.",
 "Three tool calls only, then stop. (1) file_system operation=\"outline\" path=\"r6/n.py\". (2) file_system operation=\"replace\" path=\"r6/n.py\" content=\"CAP = 7\" replace_with=\"CAP = 9\". (3) file_system operation=\"write\" path=\".git/x\" content=\"y\". Reply ONLY with: symbols, then EDIT: first 4 words of (2), then BLOCK: first 6 words of (3).",
 "Two tool calls only, then stop. (1) file_system with operation=\"symbols\" and name=\"connect\". (2) git with operation=\"status\". Reply with ONLY: the file:line that symbols reported, then the literal word STATUS followed by the first line of the git result. Do not investigate anything, do not use any other tool.",
 "Use deep_research ONCE on the query: llama.cpp prompt prefill speed apple silicon. Then reply with one short sentence about what you found. Do not investigate anything else, do not read files.",
 "hello ghost, what's going on ?",
 "Find a photo of the Greek prime minister and make him a proud trans woman.",
 "When executing complex geospatial analysis involving model grids, ensure the fin",
 "When performing file system modifications, ensure the required change (e.g., fun",
 "When modifying a function signature and body, ensure the change is applied consi",
 "When renaming a variable across multiple instances, verify all occurrences in bo",
 "When browsing websites, verify the ability to fetch content before attempting co",
 "When navigating to external websites, verify the HTTP status code is not 403 bef",
 "When using the browser tool, verify the page is accessible before attempting com",
 "When using the `browser` tool, ensure the selector used matches the actual struc",
 "When using the browser tool to view a webpage, use 'navigate' or 'extract_text' ",
 "When building code, ensure that the required content parameter is provided for a",
 "When performing self-play tasks, ensure the final output contains a direct answe",
 "When analyzing data or making claims about specific incidents (e.g., traffic acc",
 "When generating an image of a specific person, use their existing image referenc",
 "When performing state checks on projects, confirm the reported port matches the ",
 "When proceeding to the next task, confirm the integrity of the shared state file",
 "When executing commands, use absolute paths in file system operations."
]
RETRACT_ONE = {
 "trigger": "Parallel processing with strict output ordering requirements",
 "solution_prefix": "When using concurrent execution (like ThreadPoolExecutor), if the output order m"
}
REWRITE = {
 "Modify the `greet()` function in `kc_probe.py` to return `f\"hi {name}\"` instead of `f\"hello {name}\"` while leaving `farewell()` untouched, making the smallest possible edit.": {
  "scope": "general",
  "trigger": "After modifying a file, confirm the change by reading the modified file",
  "mistake": "Reporting a file edit as done without checking the result.",
  "solution": "After executing a file modification, read the modified file (or the changed region) to confirm the change is correct and that nothing else changed unintentionally."
 },
 "Search the web for the latest stable Python 3.13 patch release and cite the source URL.": {
  "scope": "general",
  "trigger": "When answering about the latest version or release of software from search results",
  "mistake": "Picking an older or less specific version from the snippets.",
  "solution": "Prefer the most recent, specific version number and its release information in the results, cite the source URL, and say so when the snippets conflict or are incomplete."
 },
 "When querying profile information, use specific recall/update tools rather than ": {
  "solution": "When querying profile information, read it with recall or introspect rather than general guessing; update_profile is for changing a value, never for reading one."
 },
 "When a task requires a specific, constrained output (e.g., 'Reply with exactly X": {
  "solution": "When a task requires an exact reply (e.g., 'Reply with exactly X'), make the final answer exactly that text; call tools only when the request itself asks for a tool's result."
 },
 "When comparing files, ensure both specified file paths exist in the sandbox befo": {
  "trigger": "When running commands that operate on a file, verify the file path exists first"
 }
}


def _agent_listening(port: int = 0) -> bool:
    """The live agent is the playbook's single writer."""
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _key(x) -> str:
    return x.get("trigger") or x.get("task") or ""


def _is_dup_one(x) -> bool:
    return _key(x) == RETRACT_ONE["trigger"] and str(x.get("solution") or "").startswith(RETRACT_ONE["solution_prefix"])


def check(playbook: list) -> None:
    """Every named row resolves to exactly one row, or SystemExit."""
    for group, names in (("retract", RETRACT), ("rewrite", list(REWRITE))):
        for t in names:
            n = sum(1 for x in playbook if _key(x) == t)
            if n != 1:
                raise SystemExit(f"{group}: {n} rows for {t[:70]!r} — nothing applied")
    if sum(1 for x in playbook if _is_dup_one(x)) != 1 or sum(1 for x in playbook if _key(x) == RETRACT_ONE["trigger"]) != 2:
        raise SystemExit("retract_one: the duplicate pair is not as recorded — nothing applied")
    from ghost_agent.memory.lesson_quality import prescribes_destruction
    from ghost_agent.memory.lesson_scope import is_general_trigger
    for old, new in REWRITE.items():
        if prescribes_destruction(new.get("solution", "")):
            raise SystemExit(f"rewrite prescribes destruction: {old[:60]!r}")
        if new.get("scope") == "general" and not is_general_trigger(new["trigger"], old):
            raise SystemExit(f"a lesson made general still restates its request: {new['trigger'][:60]!r}")


def restamp_scoped_twins(sm, vm) -> int:
    """Set scope=request on the twin metadata of every request-scoped row."""
    scoped = {_key(x)[:200] for x in sm._load_playbook() if x.get("scope") == "request"}
    got = vm.collection.get(where={"type": "skill"}, limit=5000, include=["metadatas"])
    ids, metas = [], []
    for i, m in zip(got.get("ids") or [], got.get("metadatas") or []):
        if str((m or {}).get("trigger") or "") in scoped and (m or {}).get("scope") != "request":
            ids.append(i)
            metas.append(dict(m, scope="request"))
    if ids:
        vm.collection.update(ids=ids, metadatas=metas)
    return len(ids)


def main():
    pb_path = MEM / "skills_playbook.json"
    playbook = json.loads(pb_path.read_text(encoding="utf-8"))
    check(playbook)
    print(f"retract {len(RETRACT)} + 1 of a duplicate pair, rewrite {len(REWRITE)}{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kw-r5-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.skills import SkillMemory, _delete_lesson_twin
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(MEM)
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in RETRACT)
    with sm._get_lock():
        pb = sm._load_playbook()
        drop = [x for x in pb if _is_dup_one(x)]
        pb = [x for x in pb if not _is_dup_one(x)]
        for x in pb:
            k = _key(x)
            if k in REWRITE:
                _delete_lesson_twin(vm, dict(x))
                for f, v in REWRITE[k].items():
                    if f == "trigger":
                        x["trigger"] = x["task"] = v
                    elif f == "solution":
                        x["solution"] = x["correct_pattern"] = v
                    elif f == "mistake":
                        x["mistake"] = x["anti_pattern"] = v
                    elif f == "scope":
                        x["scope"] = v
                        if v != "request":
                            x.pop("source_request", None)
        sm._save_playbook_unlocked(pb)
    with open(MEM / "skills_pruned_archive.jsonl", "a", encoding="utf-8") as fh:
        for x in drop:
            fh.write(json.dumps({"reason": "duplicate_trigger_r5", "lesson": x}, ensure_ascii=False) + "\n")
    # the shared twin goes with it; the heal re-embeds the surviving row's
    for x in drop:
        _delete_lesson_twin(vm, dict(x))
    healed = sm.heal_missing_twins(vm)
    stamped = restamp_scoped_twins(sm, vm)
    print(f"retracted {gone}+{len(drop)}; rewritten {len(REWRITE)}; twins written {healed}; scoped twins restamped {stamped}")


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
