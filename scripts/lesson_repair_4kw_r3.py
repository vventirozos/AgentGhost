"""§4KW one-off (2026-10-02 night): playbook repair after the third fresh review.

Every row is named by its EXACT trigger (read one by one, reasons in the
journal), never by index. Each must resolve to exactly one row, or nothing is
applied.
  * RETRACT (archived by `remove_by_trigger`, vector twin deleted): 13 generated
    general lessons that are harmful or overbroad (likeness from a description,
    follow a tool's suggested action, skip tool steps, dump created files,
    numbered-list format, unrequested init steps, install dependencies, resolve
    identifiers first, one side of a research contradiction, 4 near-duplicate
    "verify against evidence" rules); 14 junk rows (echoes of the reflection
    prompt, observations, a stored answer, bare "Post-Mortem Analysis" titles,
    the sandbox `delete path=/*` lesson, test-file dream rules).
  * SCOPE: 25 lessons whose trigger is one specific task (model-written
    restatements the request matcher could not find).
  * RESTORE from the archive: 3 lessons with real use evicted by the cap-trim.
  * REFRESH every vector twin whose text is not its row's current text (data
    audit: 111 of 299 twins carried an older fix — merges replaced the fix
    without re-embedding; `_refresh_twin` now does that going forward).
Run with the agent STOPPED. The whole memory dir is copied first (the chroma
segment files change too — data audit).
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_repair_4kw_r3.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
SPEC = {
 "retract": [
  "When a tool returns a status or error message indicating an existing state and explicitly recommends a specific action to resolve it.",
  "When a user makes a direct, specific request requiring a particular output format or concise response, and the system has multiple available tools or actions.",
  "When creating files and executing commands to verify or demonstrate them.",
  "Multi-step research or investigative tasks where the user requests a consolidated analysis or structured report based on gathered evidence.",
  "When a user requests highly specific, granular details about a precise entity, location, or niche subject that may not be widely documented online.",
  "When creating a visual representation of a specific real person using a reference image retrieved from a search.",
  "When a user asks for a procedure, command, or set of instructions and expects a structured, step-by-step response.",
  "When asked to list or report on multiple items and their specific attributes or statuses.",
  "When researching and recommending specific technical attributes or factual claims based on retrieved data.",
  "When summarizing findings from a tool-based search or research task in a brief statement.",
  "When a request references an entity using an identifier, code, or technical reference rather than a standard name.",
  "When running a command or tool that requires a specific environment setup or prerequisite state, especially when that state is unknown or likely unconfigured.",
  "When executing a task that relies on external tools or libraries in a constrained or sandboxed environment.",
  "lots of stuff in your sandbox, clean it up",
  "When creating new utility files like `kc_tool.py`, ensure the docstring meets th",
  "When making changes to kc_probe.py, verify the operation succeeded, as 'verifier",
  "Identify the core technical error, hallucination, or bad strategy in the agent's response to the user's prompt.",
  "Identify the core technical error, hallucination, or bad strategy in the agent's response to a user query about a specific murder case (Anna's murder) and provide a concrete rule to fix it.",
  "Analyze the interaction to identify the core technical error, hallucination, or bad strategy, and extract a concrete rule for future interactions.",
  "Review the interaction to identify a core technical error, hallucination, or bad strategy.",
  "The agent successfully retrieved and described an image and answered a specific technical question about its implementation.",
  "The agent successfully completed the multi-step instruction while adhering to the one-sentence constraint and avoiding the forbidden words ('file', 'delete', 'create').",
  "The agent successfully created, listed (implicitly, by the tool output), and deleted the three specified files, and then provided the required final confirmation.",
  "Optimization Analysis: Use a tool to count exactly how many words are in ...",
  "Post-Mortem Analysis",
  "Post-Mortem Analysis of Elden Ring Data Collection and File Writing",
  "The user requested an image edit to ensure the generated person retained the likeness of Kyriakos Mitsotakis, specifically instructing the agent to 'use his face' and 'don't generate a new one.' The agent successfully used the `image_generation` tool to create an edited image based on a reference, which was then confirmed by `vision_analysis` to have preserved his recognizable features."
 ],
 "scope": [
  "Verify the successful deployment and functionality of the 'war-sim' service after a configuration change.",
  "Generate a highly specific, photorealistic image of a real person (George Delaportas) in a specific costume (Evzonas) in a specific location (Syntagma Square) with specific atmospheric details.",
  "Execute a single tool call to `git` with `operation=\"status\"` and return only the first 15 words of the output.",
  "Generate an image of a space cat using the `image_generation` tool.",
  "Find the founding years of Naftemporiki, Kathimerini, To Vima, and Eleftherotypia, one search each, and summarize.",
  "Progress report on extracting data from 'uk_part_time_msc_international_relations_research.md'",
  "Identify all degrees/ranks held by the individual mentioned in the search results, specifically within the context of the 'Stoa' (Στοά) organization.",
  "Describe one specific thing remembered from an earlier session, concretely, with no metaphors.",
  "Identify the best investment platforms for a Greek citizen based on web search results.",
  "Find and plot grid points of the IFS model in the reduced Gaussian grid that fall within the area of Oxford, UK.",
  "Execute a sequence of shell commands (calculate, write to file, read from file, delete file) and return a specific JSON structure based on the results.",
  "Calculate the rate of major global crises over time and compare the last 20-25 years to older periods.",
  "Investigate if Tempi in Greece is a 'sacrificial ground' based on a number of 'freak accidents'.",
  "Count the number of chapters in the PostgreSQL 19 manual based on provided knowledge base snippets.",
  "Compare `cerebro_dev.schema.sql` and `cerebro.schema.sql` and provide a report.",
  "Task 3 (Weapon upgrades section) — building `weapon-upgrades.html`.",
  "Run youtube_transcribe on a YouTube URL and report download/transcription results",
  "Define a composed skill 'youtube_transcribe' with specific sequential steps and confirm its activation.",
  "Execute a sequence of shell commands (write to file, count lines, delete file, calculate result) and return only the final number.",
  "Identify BJJ schools near 'North Athens' using web search and browsing tools.",
  "Identify the name and monthly price of the cheapest shared-vCPU plan on https://www.hetzner.com/cloud/server.",
  "Execute a specific shell command and return the exit code verbatim.",
  "Find specific information (Open University MA International Relations Security modules) when initial direct searches and browser attempts fail due to access restrictions (403/404).",
  "Determine the required SD card type for the Nintendo Switch 2 based on provided search results.",
  "Determine if Xanax (alprazolam) can be prescribed to a 15-year-old and list side effects."
 ],
 "restore_prefixes": [
  "When performing dark web searches, verify the success of the search across all c",
  "When using the 'file_system' tool for downloads, ensure the operation is not blo",
  "When using the browser tool, check for and handle execution context destruction "
 ]
}


def _agent_listening(port: int = 0) -> bool:
    """The live agent is the playbook's single writer: an apply refuses to
    run while it listens (fourth review — this was convention only). The
    port is the agent's (8000; ``GHOST_AGENT_PORT`` overrides it)."""
    import os
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def main():
    from ghost_agent.memory.skills import SkillMemory
    pb_path = MEM / "skills_playbook.json"
    playbook = json.loads(pb_path.read_text(encoding="utf-8"))
    key = lambda x: (x.get("trigger") or x.get("task") or "")
    for group in ("retract", "scope"):
        for t in SPEC[group]:
            n = sum(1 for x in playbook if key(x) == t)
            if n != 1:
                raise SystemExit(f"{group}: {n} rows for {t[:70]!r} — nothing applied")
    archive = [json.loads(l) for l in open(MEM / "skills_pruned_archive.jsonl", encoding="utf-8") if l.strip()]
    restore = []
    for pfx in SPEC["restore_prefixes"]:
        hits = [r["lesson"] for r in archive if isinstance(r.get("lesson"), dict) and key(r["lesson"]).startswith(pfx)]
        if not hits:
            raise SystemExit(f"restore: not in the archive: {pfx!r}")
        les = hits[-1]
        if any(key(x) == key(les) for x in playbook):
            continue
        restore.append(les)
    print(f"retract {len(SPEC['retract'])}, scope {len(SPEC['scope'])}, restore {len(restore)}"
          f"{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kw-r3-{stamp}.bak")
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(MEM)
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in SPEC["retract"])
    with sm._get_lock():
        pb = sm._load_playbook()
        for x in pb:
            if key(x) in SPEC["scope"]:
                x["scope"] = "request"
                x["source_request"] = key(x)[:4000]
        pb.extend(restore)
        sm._save_playbook_unlocked(pb)
    stale = stale_twins(sm, vm)
    from ghost_agent.memory.skills import _delete_lesson_twin
    for les in stale:
        _delete_lesson_twin(vm, les)
    healed = sm.heal_missing_twins(vm)
    print(f"retracted {gone}; scoped {len(SPEC['scope'])}; restored {len(restore)}; stale twins {len(stale)}; "
          f"twins re-embedded {healed}")


def stale_twins(sm, vm) -> list:
    """Playbook lessons whose vector twin's text differs from the text the
    lesson embeds as now."""
    from ghost_agent.memory.skills import _normalize_lesson, _normalize_trigger, lesson_embedding_text
    got = vm.collection.get(where={"type": "skill"}, limit=5000, include=["metadatas", "documents"])
    docs = {}
    for meta, doc in zip(got.get("metadatas") or [], got.get("documents") or []):
        k = _normalize_trigger(str((meta or {}).get("trigger") or "")[:200])
        if k:
            docs[k] = doc or ""
    out = []
    for p in sm._load_playbook():
        les = _normalize_lesson(p)
        k = _normalize_trigger((les.get("trigger") or les.get("task") or "")[:200])
        if k in docs and docs[k] != lesson_embedding_text(les):
            out.append(les)
    return out


if __name__ == "__main__":
    main()
