"""§4KW one-off (2026-10-02): take a GENERAL lesson from each legacy
request-scoped plan, the way reflection now does for new ones.

The legacy lessons keyed to one request were scoped by
`lesson_scope_migrate_4kw.py`; none had a transferable rule extracted. This
asks the main model, for each scoped lesson whose reflection plan was
CONFIRMED (`verified`) — or whose fix is already a rule (post-mortem and
journal lessons) — for the same GENERAL LESSON block the reflection prompt asks for
(SITUATION / MISTAKE / RULE, or NONE), and keeps a candidate only when it passes
every gate a new reflection rule passes: `is_general_trigger` (situation),
`is_general_text` (mistake), `prescribes_destruction` (rule) and the write
gate (`is_actionable_lesson`).

Two phases:
  --generate OUT.json   call the model (one call at a time, only while the live
                        agent has no foreground request); writes candidates.
  --apply IN.json       write the kept candidates as general reflection
                        lessons (`learn_lesson`, which dedups). Run with the
                        agent STOPPED (the vector store has one writer).
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_generalize_legacy_4kw.py --generate c.json
"""
import json, os, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
MEM = HOME / "system" / "memory"
MODEL_URL = os.environ.get("GHOST_MAIN_URL", "http://127.0.0.1:8088/v1/chat/completions")
AGENT_HEALTH = "http://127.0.0.1:8000/api/health"

PROMPT = """\
Below is a lesson an assistant recorded about ONE earlier request: the request, what went wrong, and the corrected \
plan for that request (the plan was checked and confirmed). Extract the part that would help with a DIFFERENT \
request.

Required output format:

GENERAL LESSON:
SITUATION: <the KIND of request or situation where the same mistake could happen again, in general terms — no \
names, places, numbers, paths, quotes or topics from this request>
MISTAKE: <the mistake to avoid in that kind of situation, in the same general terms>
RULE: <what to do instead, one or two sentences>

If nothing about it would help with a different request, write instead:
GENERAL LESSON: NONE

Request:
{request}

What went wrong:
{mistake}

Corrected plan:
{plan}
"""


def gate(cand: dict, request: str) -> str:
    """'' when the candidate passes every gate, else the first reason."""
    from ghost_agent.memory.lesson_scope import is_general_trigger, is_general_text
    from ghost_agent.memory.lesson_quality import is_actionable_lesson, prescribes_destruction
    if not is_general_trigger(cand["situation"], request):
        return "situation restates the request"
    if not is_general_text(cand.get("mistake", ""), request):
        return "mistake restates the request"
    if not is_general_text(cand["rule"], request):
        return "rule restates the request"
    if prescribes_destruction(cand["rule"]):
        return "rule prescribes destruction"
    if not is_actionable_lesson(cand.get("mistake", ""), cand["rule"], cand["situation"]):
        return "fails the write gate"
    return ""


def _idle() -> bool:
    try:
        key = Path("~/Data/AI/.ghost_api_key").expanduser().read_text().strip()
        req = urllib.request.Request(AGENT_HEALTH, headers={"X-Ghost-Key": key})
        return json.load(urllib.request.urlopen(req, timeout=5)).get("foreground_requests", 1) == 0
    except Exception:
        return True          # agent down (apply phase) or unreachable: nothing to yield to


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


def generate(out_path: str):
    from ghost_agent.reflection.prompts import parse_general_lesson
    lessons = json.loads((MEM / "skills_playbook.json").read_text(encoding="utf-8"))
    # resume: candidates already in OUT are kept and not asked again
    rows = json.loads(Path(out_path).read_text(encoding="utf-8")) if Path(out_path).exists() else []
    done = {r["source_trigger"] for r in rows}
    for l in lessons:
        if l.get("scope") != "request":
            continue
        # a reflection PLAN only when its check confirmed it; a post-mortem
        # or journal lesson's fix is already a rule (second review)
        if str(l.get("source") or "") == "reflection" and not l.get("verified"):
            continue
        if done and (l.get("trigger") or l.get("task") or "")[:400] in done:
            continue
        request = str(l.get("source_request") or l.get("trigger") or l.get("task") or "")
        prompt = PROMPT.format(request=request[:1200], mistake=str(l.get("mistake") or l.get("anti_pattern") or "")[:600],
                               plan=str(l.get("solution") or l.get("correct_pattern") or "")[:1500])
        while not _idle():
            time.sleep(10)
        body = json.dumps({"model": "qwen-3.6-35b-a3", "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0.3, "max_tokens": 4096, "stream": False}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            MODEL_URL, data=body, headers={"Content-Type": "application/json"}), timeout=600))
        text = r["choices"][0]["message"].get("content") or ""
        cand = parse_general_lesson(text)
        rec = {"source_trigger": (l.get("trigger") or l.get("task") or "")[:400],
               "source_trajectory_id": l.get("source_trajectory_id") or "", "candidate": cand,
               "rejected": ("no general lesson (NONE or unparsed)" if not cand else gate(cand, request))}
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        # after EVERY candidate: an interrupted run resumes from here
        Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"candidates: {len(rows)}; kept: {sum(1 for r in rows if not r['rejected'])}")


def apply(in_path: str):
    from ghost_agent.memory.skills import SkillMemory
    rows = json.loads(Path(in_path).read_text(encoding="utf-8"))
    keep = [r for r in rows if not r["rejected"] and r["candidate"]]
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    # the WHOLE memory dir: the chroma segment files change too (fourth review)
    import shutil
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kw-generalize-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))

    # the REAL vector store: learn_lesson writes the twin through
    # `memory_system.add`, and a collection-only stand-in made every twin
    # write fail after the JSON row was saved (first apply: 0 of 44 twins)
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(MEM)
    written = 0
    for r in keep:
        c = r["candidate"]
        out = sm.learn_lesson(c["situation"], c.get("mistake", ""), c["rule"], memory_system=vm,
                              source="reflection", source_trajectory_id=r["source_trajectory_id"],
                              origin="auto")
        print(f"  {out!s:>10}: {c['situation'][:90]}")
        written += bool(out)
    print(f"general lessons written or reinforced: {written} of {len(keep)}")
    print(f"missing vector twins re-embedded: {sm.heal_missing_twins(vm)}")


def heal():
    """Re-embed every playbook lesson whose vector twin is missing (agent
    stopped)."""
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    print(f"missing vector twins re-embedded: {SkillMemory(MEM).heal_missing_twins(vm)}")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate(sys.argv[sys.argv.index("--generate") + 1])
    elif "--heal-twins" in sys.argv:
        heal()
    elif "--apply" in sys.argv:
        apply(sys.argv[sys.argv.index("--apply") + 1])
    else:
        print(__doc__)
