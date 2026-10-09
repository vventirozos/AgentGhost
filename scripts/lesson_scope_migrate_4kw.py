"""§4KW one-off (2026-10-02, operator: "fix the lessons keyed to one specific
request"): tag the live playbook's REQUEST-KEYED lessons ``scope="request"``.

A lesson is request-keyed when its trigger IS a recorded user request
(normalised text, or the first 400 chars of one) — reflection, the post-mortem
engine and the journal post-mortem keyed corrected plans on the failed
request. Measured 2026-10-02: 40 such lessons, 927 injections on the recorded
traffic, 20 of them on the lesson's own request (or a rewording). Tagged, a
lesson is retrieved only for the same request (``memory.lesson_scope``).
Nothing is deleted or rewritten; ``source_request`` keeps the request text.
The vector copies are not touched: retrieval reads the scope from the
playbook row.

  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_scope_migrate_4kw.py [--apply]
Run --apply with the agent STOPPED: the agent is the playbook's single
writer (fourth review — "safe while running" was wrong: its own next save
overwrites this one). A backup is written next to the playbook. Dry run by
default.
"""
import glob, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ghost_agent.memory.lesson_scope import normalize_request, SCOPE_REQUEST  # noqa: E402

APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
SYS = HOME / "system"
STAMP = time.strftime("%Y%m%dT%H%M%S")


#: Request-keyed lessons whose trigger PARAPHRASES its request (no text rule
#: can find them; listed by the second fresh review, each read: the trigger
#: restates one request, the fix is its plan or a rule written for it).
PARAPHRASED = (
    "Search the web for the latest stable Python 3.13 patch release and cite the source URL.",
    "Rename parameter 'name' to 'who' in both functions (signature and body) of kc_probe.py.",
    "Modify the `greet()` function in `kc_probe.py` to return",
    "Restart the service 'elden-tracker' on port 8102.",
    "Generate an image of Dario Amodei (CEO of Anthropic) licking",
    "Generate an image of Dario Amodei giving",
)


def recorded_requests() -> dict:
    """normalised request -> the full request, over every recorded request
    (probes included: a probe's lesson is request-keyed too)."""
    out = {}
    for f in glob.glob(str(SYS / "trajectories" / "*" / "session-*.jsonl")):
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                t = json.loads(line)
            except Exception:
                continue
            if t.get("task_kind") == "reflection":
                continue
            r = str(t.get("user_request") or "")
            if r.strip():
                out[normalize_request(r)] = r
    return out


def match_request(trigger: str, reqs: dict):
    """The full recorded request a trigger was keyed on, else None: equal
    normalised text, the trigger a prefix of the request (a trigger cut at
    400 chars), or a listed paraphrase."""
    from ghost_agent.memory.lesson_scope import same_request
    t = trigger.strip()
    n = normalize_request(t)
    if not n:
        return None
    if n in reqs:
        return reqs[n]
    if len(n) >= 40:
        for nr, r in reqs.items():
            if nr.startswith(n):
                return r
    if any(t.startswith(p) for p in PARAPHRASED):
        return t
    for nr, r in reqs.items():
        if same_request(t, r):
            return r
    return None


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
    if APPLY and _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    reqs = recorded_requests()
    sm = SkillMemory(SYS / "memory")
    with sm._get_lock():
        playbook = sm._load_playbook()
        tagged, widened = [], 0
        for row in playbook:
            trig = (row.get("trigger") or row.get("task") or "").strip()
            hit = match_request(trig, reqs)
            if row.get("scope") == SCOPE_REQUEST:
                # widen a 400-char source_request to the full request
                if hit and len(str(row.get("source_request") or "")) < len(hit[:4000]):
                    widened += 1
                    if APPLY:
                        row["source_request"] = hit[:4000]
                continue
            if hit:
                tagged.append(trig)
                if APPLY:
                    row["scope"] = SCOPE_REQUEST
                    row["source_request"] = hit[:4000]
        for t in tagged:
            print("  request-scoped:", t[:100].replace("\n", " "))
        print(f"lessons tagged request-scoped: {len(tagged)} of {len(playbook)}; source_request widened: "
              f"{widened}{'' if APPLY else ' (dry run)'}")
        if APPLY and (tagged or widened):
            pb = sm.file_path
            backup = pb.with_name(pb.name + f".pre-4kw-scope-{STAMP}.bak")
            backup.write_bytes(pb.read_bytes())
            sm._save_playbook_unlocked(playbook)


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
