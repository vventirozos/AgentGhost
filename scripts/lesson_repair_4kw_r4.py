"""§4KW one-off (2026-10-02 night): playbook repair after the FOURTH fresh
review (data audit). Rows named by EXACT trigger; any that does not resolve
to exactly one row aborts before anything is written.
  * RETRACT (archived by `remove_by_trigger`, twin deleted):
    - 25 general lessons that are harmful or junk: "report only the count" of
      a listing, self-play rules that satisfy a test VALIDATOR instead of the
      user, a leftover "print 'Let me…' first" rule, "confirm update_profile
      and DONE", init-the-environment / verbatim-no-checking / always-add-
      unit-tests rules, one-off bench arithmetic (Patchy and Trixie …), stale
      hard-coded introspect action lists, duplicates and mismatched rows;
    - 23 request-scoped lessons keyed on a MODEL restatement of a request no
      recorded trajectory can trace: the strict request match can never admit
      them (they were retired in place, holding slots).
  * RE-KEY 2 scoped lessons to the user's ACTUAL request (traced through
    their source trajectory).
  * REWRITE 14 general lessons that named one request's entities or files
    (Hetzner, Leonidas, Fousekis/Revolut, KYC passport, war scenarios,
    `.frag_copy`, IFS, kc_probe.py, farewell(), logslow.py, 'emp1', 'can you
    dig it?', 'delete it'): the example goes, the rule stays; a changed
    trigger gets a fresh twin.
  * ADD 3 general rules from retired plans (403/404 pivot, the geographic
    constraint in the first query, the most specific product source).
Run with the agent STOPPED; the whole memory dir is copied first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_repair_4kw_r4.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
RETRACT = [
 "When asked to list workspace contents, use the 'file_system' tool and then repor",
 "Fulfilling ambiguous user requests by adhering strictly to the required output format",
 "Fulfilling a generic request using file content as context",
 "Fulfilling specific output requirements based on input structure",
 "Fulfilling complex, multi-step user requests via a constrained execution environment",
 "When performing multi-step commands, ensure all required sequential steps are ex",
 "When executing commands that require a specific output prefix, include that pref",
 "When updating profile information, confirm the exact data structure (e.g., color",
 "When performing complex tasks, ensure the execution environment is properly init",
 "When performing actions that require external data (like weather or search), use",
 "When browsing external websites, verify the target URL is within the defined san",
 "When investigating external resources like building histories, verify the tool's",
 "File naming convention adherence",
 "File naming consistency and execution path",
 "When interacting with project management tools, cross-reference task counts agai",
 "When using the introspect tool, verify the 'action' parameter matches allowed va",
 "When using the introspect tool to retrieve open questions, specify the required ",
 "Arithmetic problem solving with sequential updates",
 "multi-step arithmetic problem solving",
 "interpreting ambiguous movement descriptions in word problems",
 "When building code, ensure that the implementation includes corresponding unit t",
 "When using the `browser` tool, avoid repeating the same selector multiple times ",
 "When implementing file persistence in Flask applications, ensure the `create_app",
 "Debugging a chart rendering issue where all bars displayed the same value.",
 "When investigating specific geographic areas like Oxford, ensure the final outpu",
 "Verify the successful deployment and functionality of the 'war-sim' service after a configuration change.",
 "Generate a highly specific, photorealistic image of a real person (George Delaportas) in a specific costume (Evzonas) in a specific location (Syntagma Square) with specific atmospheric details.",
 "Execute a single tool call to `git` with `operation=\"status\"` and return only the first 15 words of the output.",
 "Generate an image of a space cat using the `image_generation` tool.",
 "Find the founding years of Naftemporiki, Kathimerini, To Vima, and Eleftherotypia, one search each, and summarize.",
 "Progress report on extracting data from 'uk_part_time_msc_international_relations_research.md'",
 "Identify all degrees/ranks held by the individual mentioned in the search results, specifically within the context of the 'Stoa' (Στοά) organization.",
 "Describe one specific thing remembered from an earlier session, concretely, with no metaphors.",
 "Identify the best investment platforms for a Greek citizen based on web search results.",
 "Execute a sequence of shell commands (calculate, write to file, read from file, delete file) and return a specific JSON structure based on the results.",
 "Calculate the rate of major global crises over time and compare the last 20-25 years to older periods.",
 "Investigate if Tempi in Greece is a 'sacrificial ground' based on a number of 'freak accidents'.",
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
]
REKEY = {
 "Find and plot grid points of the IFS model in the reduced Gaussian grid that fall within the area of Oxford, UK.": "find which grid points of the IFS model in the reduced gaussian grid are around and over the area of Oxford in the UK? Can you plot them and show me the results?",
 "Count the number of chapters in the PostgreSQL 19 manual based on provided knowledge base snippets.": "how many chapters does postgresql 19 manual in your knowledgebase has ?"
}
REWRITE = {
 "When querying external services, use specific, direct URLs for known resources (": {
  "solution": "When querying external services, use specific, direct URLs for the provider's own resource pages (pricing, status, documentation) rather than relying solely on general search results."
 },
 "When asked about specific entities (e.g., Leonidas), prioritize answering from s": {
  "trigger": "When asked about a specific entity with a format or source constraint, follow both constraints",
  "solution": "When answering questions about a specific entity, adhere strictly to the requested format (e.g., one short sentence) and to the requested source (e.g., only from your own records)."
 },
 "When researching specific entities, cross-reference information from multiple so": {
  "solution": "When investigating a specific person or organisation, cross-reference initial findings with subsequent queries to ensure consistency and avoid relying on single, potentially incomplete data points."
 },
 "When investigating external incidents, use both open-source and dark-web search ": {
  "solution": "When investigating a specific incident or entity, use several tools (web_search, darkweb_search, browser) in sequence so that one blocked or empty source does not end the investigation."
 },
 "When searching for specific data, ensure the search query covers all required an": {
  "solution": "When searching for specific data, make the queries cover every part the request names (for example both of two documents or both of two events) rather than focusing on a single aspect."
 },
 "When executing complex simulations (like war scenarios), ensure all necessary se": {
  "trigger": "When executing a simulation that depends on running services, ensure every service is up first",
  "solution": "When executing a simulation that depends on services, ensure all necessary services are running and listening on their expected ports before proceeding."
 },
 "When using the `browser` tool, ensure the selector used matches the actual struc": {
  "solution": "When performing browser operations, inspect the page and use a selector that matches its actual structure instead of a generic element match, to avoid 'selector did not match any element' errors."
 },
 "When executing complex geospatial analysis involving model grids, ensure the fin": {
  "solution": "When a geospatial analysis request asks to plot or show results, ensure the final step produces the plot, not just a list of coordinates."
 },
 "When modifying a function signature and body, ensure the change is applied consi": {
  "solution": "When modifying a function signature or body, apply the change consistently everywhere it is used (for example renaming a parameter in the signature and the body of every affected function)."
 },
 "When performing file system modifications, ensure the required change (e.g., fun": {
  "solution": "When performing file modifications, state the required change (e.g., a function's return value) and its scope (what must stay untouched) explicitly before editing."
 },
 "When building tools in the sandbox, ensure the source code is provided alongside": {
  "solution": "When building tools in the sandbox, write the tool's source file before running the command that executes it."
 },
 "Interpret a single-token user input ('emp1') following a system block/tool execution context.": {
  "trigger": "Interpret a single-token user input that follows a system block or tool execution",
  "mistake": "Defaulting to a generic conversational fallback instead of interpreting the single token as a possible command, identifier or continuation of the previous context.",
  "solution": "When the user input is ambiguous (e.g., a single token) immediately following a system block or tool execution, first try to map it to known commands or context values before asking for clarification. If mapping fails, state the ambiguity and ask, referencing the previous context."
 },
 "Interpret ambiguous user input ('can you dig it?') and provide a helpful, actionable response.": {
  "trigger": "Interpret idiomatic or ambiguous user input before choosing a technical action",
  "mistake": "Reading an idiom literally as a request for a technical operation and answering with a list of tool options."
 },
 "Delete a project based on a vague user instruction ('delete it') when the system requires an explicit project identifier and the project is not currently active.": {
  "trigger": "Delete a project from a vague instruction when the tool requires an explicit project id and the project is not active",
  "mistake": "Repeating the same failing hard delete instead of acting on the alternative the tool's response offered (archiving)."
 }
}
NEW_GENERAL = [
 {
  "situation": "When browsing a page fails with access errors such as 403 or 404",
  "mistake": "Retrying the same landing page or aggregator links after repeated access blocks.",
  "rule": "Pivot to alternatives: a cached or archived copy (for example Archive.org), a more specific subdomain or department page, or a different source for the same information."
 },
 {
  "situation": "When a request carries a geographic constraint such as a district or area",
  "mistake": "Searching without the constraint and returning results for the wider region.",
  "rule": "Put the location constraint in the first search query, and verify each result's location against it before answering."
 },
 {
  "situation": "When sources describe both a specific product version and its general product line",
  "mistake": "Mixing general product-line information into claims about the specific version.",
  "rule": "Prefer the most specific, recent and authoritative source for that exact version; where the sources conflict or are incomplete, say so instead of stating a definite limitation."
 }
]


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


def check(playbook: list) -> None:
    """Every named row resolves to exactly one row, or SystemExit."""
    for group, names in (("retract", RETRACT), ("rekey", list(REKEY)), ("rewrite", list(REWRITE))):
        for t in names:
            n = sum(1 for x in playbook if _key(x) == t)
            if n != 1:
                raise SystemExit(f"{group}: {n} rows for {t[:70]!r} — nothing applied")
    from ghost_agent.memory.lesson_scope import is_general_text, is_general_trigger
    from ghost_agent.memory.lesson_quality import is_actionable_lesson, prescribes_destruction
    for c in NEW_GENERAL:
        if not (is_general_trigger(c["situation"], "") and is_actionable_lesson(c["mistake"], c["rule"], c["situation"])
                and not prescribes_destruction(c["rule"])):
            raise SystemExit(f"new rule fails a write gate: {c['situation'][:60]!r}")
    for old, new in REWRITE.items():
        if prescribes_destruction(new.get("solution", "")):
            raise SystemExit(f"rewrite prescribes destruction: {old[:60]!r}")


def main():
    pb_path = MEM / "skills_playbook.json"
    playbook = json.loads(pb_path.read_text(encoding="utf-8"))
    check(playbook)
    print(f"retract {len(RETRACT)}, re-key {len(REKEY)}, rewrite {len(REWRITE)}, add {len(NEW_GENERAL)}"
          f"{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kw-r4-{stamp}.bak")
    from ghost_agent.memory.skills import SkillMemory, _delete_lesson_twin
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(MEM)
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in RETRACT)
    with sm._get_lock():
        pb = sm._load_playbook()
        for x in pb:
            k = _key(x)
            if k in REKEY:
                x["source_request"] = REKEY[k][:4000]
            if k in REWRITE:
                _delete_lesson_twin(vm, dict(x))          # the old text's twin
                new = REWRITE[k]
                for f, v in new.items():
                    if f == "trigger":
                        x["trigger"] = x["task"] = v
                    elif f == "solution":
                        x["solution"] = x["correct_pattern"] = v
                    elif f == "mistake":
                        x["mistake"] = x["anti_pattern"] = v
        sm._save_playbook_unlocked(pb)
    added = 0
    for c in NEW_GENERAL:
        added += bool(sm.learn_lesson(c["situation"], c["mistake"], c["rule"], memory_system=vm,
                                      source="operator_repair", origin="auto"))
    healed = sm.heal_missing_twins(vm)
    print(f"retracted {gone}; re-keyed {len(REKEY)}; rewritten {len(REWRITE)}; added {added}; twins written {healed}")


if __name__ == "__main__":
    main()
