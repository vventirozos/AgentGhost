"""§4KW one-off (2026-10-02, operator: "do" — relabel the test rows).

Seventeen of the operator's live test requests were recorded as user traffic:
their senders set an `X-Request-ID` (probe4jr1, live4ks01, kf-live, …) without
the `X-Ghost-Origin: probe` header, and `is_probe_request_id` needs the
`probe-` prefix WITH a dash. Every one was enrolled in the experiments, fed
calibration and the memory-ranking ledger, and two taught a playbook lesson.

What a correctly-marked probe writes is the target, store by store:
1. Trajectories: `task_kind="probe"` (every reader admits only `user_request`).
   The reflection rows derived from them (`reflect:<id>`) too — reflection
   never reads a probe. The original kind is kept in `extra.relabelled_4kw`.
2. Calibration rows: removed. A probe never writes one (§4FB), and the
   readers count every origin but "bench" as real, so a relabel would not
   exclude them.
3. Memory-ranking observations (`rrf/observations.jsonl`, keyed `turn`):
   removed — a probe turn never credits the memories it surfaced (§4FB).
4. Playbook lessons sourced from these trajectories: retracted from the JSON
   playbook AND the vector store (`retract_lessons_from_trajectory`); and the
   dream lessons that restate a test prompt (`TEST_DERIVED_TRIGGERS`).
5. Diary rows (`selfhood/autobiographical.jsonl`, by trajectory_id): removed —
   written only for `turn_origin == "user"`.
NOT touched: the foresight ledger and the verifier's escalation/shadow logs —
probes write those by design.

Run ONLY with the agent stopped (the vector store has one writer):
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/probe_relabel_4kw.py [--apply]
Backups are written next to every file touched. Dry run by default.
"""
import glob, json, os, shutil, sys, time
from pathlib import Path

APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
SYS = HOME / "system"
MEM = SYS / "memory"
STAMP = time.strftime("%Y%m%dT%H%M%S")
TEST_IDS = frozenset({
    "probe4jr1", "probe4jr2", "probe4jr3", "probe4jr4", "probe4jr5", "probe4js1", "probe4jt",
    "live4ks01", "live4ks02", "live4ks03", "live4ks04", "live4kt01",
    "kf-live", "kg-live", "yt-4ke-live", "fb-ctrl-01", "fb-ctrl-02",
    # fresh review: older test requests with the same defect (request text is
    # a test in each: PONG/echo controls, a "Diagnostic request", a watch test
    # and the callback it fired, the Naftemporiki search probes)
    "probe-a3b3b65c", "probe-87956894", "probe-fb-probe-01", "slack-kd4probe", "sniffer-probe-1",
    "fsnote-demo-2", "imgtest01", "smoketest-stream-1", "smoketest-stream-2", "watchtest1",
    "sched-watch_cd61a16f33",
})
#: Dream lessons whose text restates a test prompt (fresh review). They carry
#: no `source_trajectory_id` (dream summarises a 40-trajectory window), so they
#: are named by trigger; removed with `remove_by_trigger` (archived first).
#: Lessons where the test turns are only part of the window are NOT here.
TEST_DERIVED_TRIGGERS = (
    "When performing web searches targeting a specific domain (e.g., site:blogs.lupyd",   # live4ks03/04
    "When executing shell commands, ensure the output is returned verbatim unless oth",   # live4kt01
    "When calling manage_services, ensure the output is pasted verbatim as the entire",   # live4ks01/02
)


def backup(p: Path):
    b = p.with_name(p.name + f".pre-4kw-{STAMP}.bak")
    shutil.copy2(p, b)
    return b


def _rewrite(path: Path, out_lines, size_before: int):
    """Atomic replace; refuses when the file grew since it was read (a live
    append would be lost)."""
    if path.stat().st_size != size_before:
        raise SystemExit(f"{path} changed while being rewritten — stop the agent and rerun")
    backup(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    os.replace(tmp, path)


def _is_test_session(sid: str) -> bool:
    sid = str(sid or "")
    return sid in TEST_IDS or (sid.startswith("reflect:") and sid[len("reflect:"):] in TEST_IDS)


def relabel_trajectories() -> set:
    """Returns the trajectory ids relabelled (or that would be)."""
    ids, n = set(), 0
    for f in sorted(glob.glob(str(SYS / "trajectories" / "*" / "session-*.jsonl"))):
        path = Path(f)
        size = path.stat().st_size
        out, changed = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                t = json.loads(line)
            except Exception:
                out.append(line); continue
            if _is_test_session(t.get("session_id")):
                ids.add(str(t.get("id") or ""))
            if _is_test_session(t.get("session_id")) and t.get("task_kind") != "probe":
                ex = t.get("extra") if isinstance(t.get("extra"), dict) else {}
                ex["relabelled_4kw"] = t.get("task_kind")
                t["extra"] = ex
                t["task_kind"] = "probe"
                changed += 1
                out.append(json.dumps(t, ensure_ascii=False))
            else:
                out.append(line)
        if changed:
            n += changed
            print(f"  {path.parent.name}/{path.name}: {changed} row(s)")
            if APPLY:
                _rewrite(path, out, size)
    print(f"trajectories relabelled probe: {n}{'' if APPLY else ' (dry run)'}")
    ids.discard("")
    return ids


def _drop_rows(path: Path, key: str, label: str):
    if not path.exists():
        print(f"{label}: no file")
        return
    size = path.stat().st_size
    keep, drop = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rid = json.loads(line).get(key)
        except Exception:
            keep.append(line); continue
        if isinstance(rid, str) and rid in TEST_IDS:
            drop += 1
        else:
            keep.append(line)
    print(f"{label}: {drop} row(s) removed{'' if APPLY else ' (dry run)'}")
    if drop and APPLY:
        _rewrite(path, keep, size)


def _drop_diary(traj_ids: set):
    path = SYS / "selfhood" / "autobiographical.jsonl"
    if not path.exists():
        print("diary: no file")
        return
    size = path.stat().st_size
    keep, drop = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            tid = json.loads(line).get("trajectory_id")
        except Exception:
            keep.append(line); continue
        if isinstance(tid, str) and tid in traj_ids:
            drop += 1
        else:
            keep.append(line)
    print(f"diary: {drop} row(s) removed{'' if APPLY else ' (dry run)'}")
    if drop and APPLY:
        _rewrite(path, keep, size)


def retract_lessons(traj_ids: set):
    pb = MEM / "skills_playbook.json"
    if not pb.exists():
        print("lessons: no playbook")
        return
    lessons = json.loads(pb.read_text(encoding="utf-8"))
    sources = sorted({x.get("source_trajectory_id") for x in lessons
                      if x.get("source_trajectory_id") in traj_ids})
    _trig = {t.strip().lower() for t in TEST_DERIVED_TRIGGERS}
    named = [x for x in lessons if (x.get("trigger") or x.get("task") or "").strip().lower() in _trig]
    for x in lessons:
        if x.get("source_trajectory_id") in traj_ids or x in named:
            print("  lesson:", (x.get("task") or "")[:80].replace("\n", " "))
    if not APPLY or not (sources or named):
        print(f"lessons to retract: {len(sources)} source(s) + {len(named)} named"
              f"{'' if APPLY else ' (dry run)'}")
        return
    backup(pb)
    shim = None
    if (MEM / "chroma.sqlite3").exists():
        shutil.copy2(MEM / "chroma.sqlite3", MEM / f"chroma.sqlite3.pre-4kw-{STAMP}.bak")
        import chromadb

        class _Shim:  # the retraction reads `.collection`
            collection = chromadb.PersistentClient(path=str(MEM)).get_collection("agent_memory")
        shim = _Shim()
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(MEM)
    total = sum(sm.retract_lessons_from_trajectory(s, memory_system=shim) for s in sources)
    total += sum(bool(sm.remove_by_trigger(x.get("trigger") or x.get("task") or "", memory_system=shim))
                 for x in named)
    print(f"lessons retracted: {total}")


if __name__ == "__main__":
    traj_ids = relabel_trajectories()
    _drop_rows(SYS / "calibration" / "calibration.jsonl", "req_id", "calibration")
    _drop_rows(SYS / "rrf" / "observations.jsonl", "turn", "rrf observations")
    _drop_diary(traj_ids)
    retract_lessons(traj_ids)
