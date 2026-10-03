"""§4KY one-off (2026-10-03): remove the wrong "facts about the owner" the
operator confirmed — "delete a b c heart attack is wrong":
  A probe/test leftovers, B the agent's own project/tool state, C other people's
  or game facts (classified on a copy by the fresh-eye review, listed below
  verbatim), plus `user EXPERIENCED heart attack`. Group D (unclear) and E
  (life facts) are KEPT. Graph rows go through `delete_edge` (archived);
  profile fields through `delete(exact=True)`. A row no longer live is
  skipped. Run with the agent STOPPED; memory dir backed up first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4ky.py [--apply]
"""
import os, shutil, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
EDGES = [
[
"user",
"ACQUIRED",
"blasphemous blade"
],
[
"user",
"BUILDING",
"glassos webos"
],
[
"user",
"BUILDS",
"ai ecosystem"
],
[
"user",
"BUILDS",
"glassos"
],
[
"user",
"CALLS",
"manage_services"
],
[
"user",
"CONSTRAINED_BY",
"16gb unified memory"
],
[
"user",
"CREATED",
"glassos"
],
[
"user",
"CREATED",
"notes_4fv.txt"
],
[
"user",
"CREATED",
"youtube transcription"
],
[
"user",
"CREATES",
"mini ai v3"
],
[
"user",
"CREATES",
"pg_stat_statements_explain"
],
[
"user",
"EXECUTED",
"python3 -c \"import nosuchmodule_xyz\" 2>&1 | head -20"
],
[
"user",
"EXECUTED",
"wc -l"
],
[
"user",
"EXPECTED",
"exit code"
],
[
"user",
"GENERATED",
"image of kyriakos mitsotakis"
],
[
"user",
"HAS_DOCUMENTATION",
"agentghost"
],
[
"user",
"HAS_FILE",
"kc_crlf.ini"
],
[
"user",
"HAS_FILE",
"kc_probe.py"
],
[
"user",
"HAS_ITEM",
"blasphemous blade"
],
[
"user",
"HAS_LEVEL",
"100"
],
[
"user",
"HAS_PREFERENCE",
"verdigris211826"
],
[
"user",
"HAS_PROJECT",
"ai self awareness exploration"
],
[
"user",
"HAS_PROJECT",
"ai self-awareness"
],
[
"user",
"HAS_PROJECT",
"chess coach"
],
[
"user",
"HAS_PROJECT",
"chess coach v2"
],
[
"user",
"HAS_PROJECT",
"e4e240b630f6"
],
[
"user",
"HAS_PROJECT",
"elden ring blasphemous build tracker"
],
[
"user",
"HAS_PROJECT",
"jiu jitsu calendar"
],
[
"user",
"HAS_PROJECT",
"mini ai v3"
],
[
"user",
"HAS_PROJECT",
"procedural dungeon crawler"
],
[
"user",
"HAS_PROJECT",
"solar system orbit simulation"
],
[
"user",
"HAS_PROJECT",
"the consciousness lab"
],
[
"user",
"HAS_PROJECT",
"webos"
],
[
"user",
"HAS_PROJECT",
"youtube transcription"
],
[
"user",
"HAS_PROJECT_CODE",
"zephyrinebfjie"
],
[
"user",
"HAS_PROJECT_CODE",
"zephyrineceehh"
],
[
"user",
"HAS_PROJECT_CODE",
"zephyrinecffhb"
],
[
"user",
"HAS_PROJECT_CODE",
"zephyrinecggce"
],
[
"user",
"HAS_PROJECT_CODE",
"zephyrinedfjca"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrine"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrinebfjie"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrinecaicd"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrinecchhe"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrineciaci"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrinedaace"
],
[
"user",
"HAS_PROJECT_CODENAME",
"zephyrineehbjb"
],
[
"user",
"HAS_PROPERTY",
"car wash"
],
[
"user",
"HAS_SANDBOX",
"sandbox"
],
[
"user",
"HAS_SKILL",
"auto_workspace_introspect"
],
[
"user",
"HAS_SKILL",
"format_results_t"
],
[
"user",
"HAS_SKILL",
"format_results_to_csv"
],
[
"user",
"HAS_SKILL",
"generate_password"
],
[
"user",
"HAS_SKILL",
"log analysis"
],
[
"user",
"HAS_SKILL",
"news_headlines"
],
[
"user",
"HAS_TASK",
"newspaper founding year summary"
],
[
"user",
"HAS_TASK",
"research newspaper founding years"
],
[
"user",
"HAS_TEST_COLOUR",
"verdigris211826"
],
[
"user",
"INTERACTED_WITH",
"adonis georgiadis"
],
[
"user",
"ISSUED_ALERT",
"system constraint check"
],
[
"user",
"ISSUED_ALERT",
"unverified change"
],
[
"user",
"IS_WORKING_ON",
"pinball.html"
],
[
"user",
"MANAGES",
"ai self awareness exploration"
],
[
"user",
"MANAGES",
"chess coach v2"
],
[
"user",
"MANAGES",
"elden ring blazing bladed blade build tracker"
],
[
"user",
"MANAGES",
"elden-tracker"
],
[
"user",
"MANAGES",
"jiu jitsu calendar"
],
[
"user",
"MANAGES",
"manage_tasks"
],
[
"user",
"MANAGES",
"mini ai v3"
],
[
"user",
"MANAGES",
"project 114a46ac81f0"
],
[
"user",
"MANAGES",
"project list"
],
[
"user",
"MANAGES",
"report"
],
[
"user",
"MANAGES",
"task ba9c48f041c7"
],
[
"user",
"MANAGES",
"webos"
],
[
"user",
"MANAGES",
"youtube transcription"
],
[
"user",
"MANAGES",
"zqx probe project"
],
[
"user",
"MODIFIED",
"kc_probe.py"
],
[
"user",
"MODIFIES",
"kc_probe.py"
],
[
"user",
"MODIFIES",
"parameter name"
],
[
"user",
"OBSERVES",
"image generation failure"
],
[
"user",
"OBSERVES",
"system alert"
],
[
"user",
"OPERATES",
"nova"
],
[
"user",
"OPERATES_IN",
"sandbox"
],
[
"user",
"OPERATES_ON",
"darwin 25.6.0 (arm64)"
],
[
"user",
"PERFORMED",
"self-play challenge"
],
[
"user",
"RECEIVED",
"system alert"
],
[
"user",
"REFERENCES",
"adolf hitler"
],
[
"user",
"REFERENCES",
"agentghost"
],
[
"user",
"REFERENCES",
"zoi konstantopoulou"
],
[
"user",
"REQUESTED",
"auto_file_system_write_execute"
],
[
"user",
"REQUESTED",
"boiling point of water at sea level"
],
[
"user",
"REQUESTED",
"custom skill"
],
[
"user",
"REQUESTED",
"deployment check 2"
],
[
"user",
"REQUESTED",
"file creation and deletion sequence"
],
[
"user",
"REQUESTED",
"file kc_tool.py"
],
[
"user",
"REQUESTED",
"file manipulation sequence"
],
[
"user",
"REQUESTED",
"graduated skills"
],
[
"user",
"REQUESTED",
"introspect tool"
],
[
"user",
"REQUESTED",
"jobs tool"
],
[
"user",
"REQUESTED",
"lesson count"
],
[
"user",
"REQUESTED",
"manage_services"
],
[
"user",
"REQUESTED",
"primary additive colors"
],
[
"user",
"REQUESTED",
"primary colors"
],
[
"user",
"REQUESTED",
"sandbox_operation"
],
[
"user",
"REQUESTED",
"system_utility"
],
[
"user",
"REQUESTED",
"tool execution"
],
[
"user",
"REQUESTED",
"zoi to spread legs"
],
[
"user",
"REQUESTED_ACTION",
"outline kb_probe.py"
],
[
"user",
"REQUIRES",
"constraint_check"
],
[
"user",
"REQUIRES",
"deployment check 2"
],
[
"user",
"REQUIRES",
"file_system"
],
[
"user",
"REQUIRES",
"json_output"
],
[
"user",
"REQUIRES",
"quarterly_report_probe_4fr.pdf"
],
[
"user",
"REQUIRES_CHANGE",
"greet()"
],
[
"ai",
"RESPONDED_TO",
"user"
],
[
"user",
"RUNS",
"critic model"
],
[
"user",
"RUNS",
"image generation model"
],
[
"user",
"RUNS",
"image recognition model"
],
[
"user",
"RUNS",
"judge model"
],
[
"user",
"RUNS",
"main llm"
],
[
"user",
"RUNS",
"video transcription model"
],
[
"user",
"RUNS",
"voice recognition model"
],
[
"user",
"SEARCHES_FOR",
"φουσέκης facebook profile leak"
],
[
"user",
"SPECIFIED",
"sandbox"
],
[
"user",
"SPECIFIED_COMMAND",
"python3 macro_probe2.py"
],
[
"user",
"SPECIFIED_CONTENT",
"print(6*7)"
],
[
"user",
"SPECIFIED_PATH",
"/macro_probe2.py"
],
[
"user",
"TARGETS",
"pinball_verify_v5.png"
],
[
"user",
"TESTED",
"agent adherence"
],
[
"user",
"USES",
"ai stack"
],
[
"user",
"USES",
"eldenstore"
],
[
"user",
"USES",
"jetson nano orin"
],
[
"user",
"USES",
"mac mini"
],
[
"user",
"USES",
"manage_services"
],
[
"user",
"USES",
"manage_tasks"
],
[
"user",
"USES",
"self_state_tool"
],
[
"user",
"USES",
"utc_probe"
],
[
"user",
"USES",
"vision_analysis_tool"
],
[
"user",
"WANTS",
"kyriakos mitsotakis"
],
[
"user",
"WANTS_ACTION",
"self_state_tool"
],
[
"user",
"WANTS_TO_COLLECT",
"miner's bell bearings"
],
[
"user",
"WANTS_TO_COLLECT",
"somberstone miner's bell bearings"
],
[
"user",
"WANTS_TO_OPTIMIZE",
"stat allocation"
],
[
"user",
"WANTS_TO_UPGRADE",
"blasphemous blade"
],
[
"user",
"WANTS_TO_UPGRADE",
"weapon"
],
[
"user",
"WORKS_ON",
"bell bearings section"
],
[
"user",
"WORKS_ON",
"elden ring blazing bladed blade build tracker"
],
[
"user",
"WORKS_ON",
"project 95438a740ca0"
],
[
"user",
"WORKS_ON",
"revolut breach report"
],
[
"user",
"WORKS_ON",
"stats section"
],
[
"user",
"WORKS_ON",
"weapon upgrades section"
],
[
"user",
"WORKS_ON",
"webos"
],
[
"user",
"EXPERIENCED",
"heart attack"
]
]
PROFILE_FIELDS = [["root", "project_codename"], ["preferences", "last_introspection_learning"], ["preferences", "last_stats_check"], ["projects", "home_lab_worker_node"]]


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def check(live: set) -> list:
    """The listed edges that are still live. Every listed edge must have an
    owner (`user`) end — anything else is a spec error, nothing applied."""
    for s, p, o in EDGES:
        if "user" not in (s, o):
            raise SystemExit(f"spec: {s} {p} {o} has no owner end — nothing applied")
    return [e for e in EDGES if tuple(e) in live]


def main():
    with sqlite3.connect(f"file:{MEM / 'knowledge_graph.db'}?mode=ro", uri=True) as c:
        live = set(c.execute("SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL"))
    todo = check(live)
    print(f"remove {len(todo)}/{len(EDGES)} edges, {len(PROFILE_FIELDS)} profile fields{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4ky-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.graph import GraphMemory
    from ghost_agent.memory.profile import ProfileMemory
    g = GraphMemory(MEM)
    n = sum(g.delete_edge(*e) for e in todo)
    pm = ProfileMemory(MEM)
    f = [pm.delete(c, k, exact=True) for c, k in PROFILE_FIELDS]
    print(f"edges removed {n}; profile: {f}")


if __name__ == "__main__":
    main()
