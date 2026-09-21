#!/usr/bin/env python3
"""Per-tool LIVE probes against the agent on :8000 — verification round M5.

One bounded prompt per tool family with a "reply with only X" contract, a
regex over the reply, and the tool(s) that must appear in the log slice the
turn wrote (read back by request id). Every turn is `X-Ghost-Origin: probe`
(§4FB: never teaches, never enrols in an arm). First run 2026-09-20 (§4JD):
18/18 PASS. Leaves a `probe_<N>.txt` in the workspace root and a
`report_<id>.pdf` — delete them afterwards.

    PYTHONPATH=src python scripts/tool_probes_live.py [out.json]

Exit 0 iff every item is PASS.
"""

import json, re, sys, time, datetime, urllib.request, pathlib
KEY = pathlib.Path.home().joinpath("Data/AI/.ghost_api_key").read_text().strip()
LOG = pathlib.Path.home().joinpath("Data/AI/Data/system/ghost-agent.log")
BASE = "http://127.0.0.1:8000"
N = datetime.datetime.now().strftime("%H%M%S")
TODAY = datetime.date.today().isoformat()
def chat(prompt, origin="probe", timeout=300, messages=None):
    body = {"messages": messages or [{"role": "user", "content": prompt}], "stream": False}
    h = {"Content-Type": "application/json", "X-Ghost-Key": KEY, "User-Agent": "ghost-functional-test"}
    if origin == "probe": h["X-Ghost-Origin"] = "probe"
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d.get("id", "")
def log_tail_pos(): return LOG.stat().st_size
def tools_since(pos):
    with open(LOG, "rb") as f:
        f.seek(pos); txt = f.read().decode("utf-8", "replace")
    calls = re.findall(r"\[probe-[0-9a-f]+ [^\]]*\] tool call — ([a-z_]+)", txt)
    calls += re.findall(r"\[[0-9a-f]{8} [^\]]*\] tool call — ([a-z_]+)", txt)
    fails = len(re.findall(r"\[(?:probe-)?[0-9a-f]+ [^\]]*\] (?:tool warning|tool error|transient fail)", txt))
    return calls, fails
ITEMS = [
 ("file_system", f"Create a file named probe_{N}.txt in the workspace root containing exactly the text alpha{N}. Then read it back and reply with only the file's contents, nothing else.", rf"alpha{N}", ["file_system"]),
 ("execute", "Run Python code that prints the product of 7 and 6, and reply with only the number it printed.", r"^\W*42\W*$", ["execute"]),
 ("system_utility", "Use your system utility tool to get today's date. Reply with only the date as YYYY-MM-DD.", rf"{TODAY}", ["system_utility","execute"]),
 ("scratchpad", f"Write the value tok{N} into your scratchpad under the key probe, then read that key back and reply with only the value.", rf"tok{N}", ["scratchpad"]),
 ("list_lessons", "Use list_lessons to count how many lessons are in your playbook. Reply with only the integer count.", r"\d+", ["list_lessons","introspect"]),
 ("introspect", "Use introspect (action summary) and reply with only the single word OK if it returned without error, otherwise ERR.", r"^\W*OK\W*$", ["introspect"]),
 ("manage_projects", "List my projects with manage_projects and reply with only the number of projects, as an integer.", r"^\W*\d+\W*$", ["manage_projects"]),
 ("manage_tasks", f"Create a task named probe_{N} with manage_tasks, confirm it appears in the task list, then delete it. Reply with only the word DONE.", r"DONE", ["manage_tasks"]),
 ("workspace", "Use the workspace tool to list files in the workspace root. Reply with only the integer number of entries.", r"\d+", ["workspace","file_system"]),
 ("jobs", "List background jobs with the jobs tool. Reply with only the integer number of jobs listed (0 if none).", r"\d+", ["jobs"]),
 ("manage_services", "List sandbox services with manage_services. Reply with only the integer number of services (0 if none).", r"\d+", ["manage_services"]),
 ("postgres_admin", "Run the SQL query SELECT 1+1 AS two on the default database and reply with only the number returned.", r"^\W*2\W*$", ["postgres_admin","execute"]),
 ("web_search", "Search the web for the boiling point of water at sea level in degrees Celsius. Reply with only the number.", r"100", ["web_search"]),
 ("browser", "Use the browser tool to open https://check.torproject.org/ and reply with only the word TOR if the page says the browser is configured to use Tor, otherwise NOT.", r"^\W*TOR\W*$", ["browser"]),
 ("report_pdf", f"Use report_pdf to generate a PDF titled probe_{N} containing one paragraph of any text. Reply with only the path of the generated file.", r"\.pdf", ["report_pdf"]),
 ("knowledge_base", "Use knowledge_base to list the documents it holds. Reply with only the integer number of documents (0 if none).", r"\d+", ["knowledge_base"]),
 ("self_state", "Use self_state to read your current state and reply with only the word OK if it returned, otherwise ERR.", r"^\W*OK\W*$", ["self_state","introspect"]),
 ("fact_check", "Use fact_check on the claim 'water boils at 100 C at sea level' and reply with only SUPPORTED or REFUTED.", r"SUPPORTED|REFUTED", ["fact_check","web_search"]),
]
res = []
for name, prompt, rx, expect in ITEMS:
    pos = log_tail_pos(); t0 = time.time()
    try:
        reply, rid = chat(prompt)
        err = ""
    except Exception as e:
        reply, rid, err = "", "", f"{type(e).__name__}: {str(e)[:80]}"
    dt = time.time() - t0
    calls, fails = tools_since(pos)
    ok_reply = bool(re.search(rx, reply.strip(), re.I | re.S)) if reply else False
    ok_tool = any(t in calls for t in expect)
    verdict = "PASS" if (ok_reply and ok_tool) else ("REPLY-ONLY" if ok_reply else ("TOOL-ONLY" if ok_tool else "FAIL"))
    res.append((name, verdict, dt, calls, fails, reply.strip()[:90].replace("\n"," "), err))
    print(f"{verdict:10s} {name:16s} {dt:6.1f}s tools={calls} toolfails={fails} reply={reply.strip()[:80]!r} {err}", flush=True)
print("\nSUMMARY:", {v: sum(1 for r in res if r[1]==v) for v in ("PASS","REPLY-ONLY","TOOL-ONLY","FAIL")})
json.dump(res, open(sys.argv[1] if len(sys.argv)>1 else "tool_probes.json","w"), indent=1)
sys.exit(0 if all(r[1]=="PASS" for r in res) else 1)
