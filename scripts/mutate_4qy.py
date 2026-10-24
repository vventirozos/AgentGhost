"""§4QY battery — the census splits knowledge failures A/B/C (2026-10-24). Reuses the §4QA harness.
    python3 scripts/mutate_4qy.py [all|check|rerun Y1 Y2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

FC = "src/ghost_agent/core/failure_census.py"
CLI = "scripts/failure_census.py"
m.PINS = ["tests/test_4qy_knowledge_split.py", "tests/test_4pm_observability.py", "tests/test_4pl_failure_census.py"]


def _tests():
    import glob, subprocess
    pat = r"failure_census\|ingest_gold\|draw_split\|GOLD_BRIEF"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", FC, "def split_decision(counts: Dict[str, int]) -> dict:",
     "# ctl\ndef split_decision(counts: Dict[str, int]) -> dict:"),
    ("CONTROL known-bad (must DIE): every turn is A", FC,
     "            if not _looked_up(t, earlier):", "            if True:"),
    ("Y1 ingest drops the kind", FC, '                     "kind": kind if kind in GOLD_KINDS else "",',
     '                     "kind": "",'),
    ("Y2 ingest keeps any kind", FC, '                     "kind": kind if kind in GOLD_KINDS else "",',
     '                     "kind": kind,'),
    ("Y3 the kind is not lower-cased", FC, '        kind = str(r.get("kind") or "").strip().lower()',
     '        kind = str(r.get("kind") or "").strip()'),
    ("Y4 the brief loses the kind paragraph", FC, "**knowledge** (facts about the world", "knowledge (facts about the world"),
    ("Y5 RIGHT turns are split too", FC,
     '    return [r for r in last.values() if r.get("kind") == "knowledge" and r.get("grade") in SPLIT_GRADES]',
     '    return [r for r in last.values() if r.get("kind") == "knowledge"]'),
    ("Y6 every kind is split", FC,
     '    return [r for r in last.values() if r.get("kind") == "knowledge" and r.get("grade") in SPLIT_GRADES]',
     '    return [r for r in last.values() if r.get("grade") in SPLIT_GRADES]'),
    ("Y7 a failed search counts as a lookup", FC,
     '    return tool_class(c) in ("ok", "partial")\n\n\ndef _looked_up', '    return True\n\n\ndef _looked_up'),
    ("Y8 documents are not evidence", FC, '_LOOKUP_TOOLS = _RESEARCH_TOOLS | {"knowledge_base", "file_system",',
     '_LOOKUP_TOOLS = _RESEARCH_TOOLS | {"file_system",'),
    ("Y9 any tool is a lookup", FC, "    if not isinstance(c, dict) or _tname(c) not in _LOOKUP_TOOLS:\n        return False",
     "    if not isinstance(c, dict):\n        return False"),
    ("Y9b transcripts are not lookups", FC, '"youtube_transcribe",\n', '\n'),
    ("Y9c an empty search is evidence", FC,
     "            return assess_turn_evidence([loop_shape(c)]).substantive > 0",
     "            return assess_turn_evidence([loop_shape(c)]).consulted > 0"),
    ("Y9d earlier turns are ignored for A", FC,
     "for u in (t, *earlier) for c in", "for u in (t,) for c in"),
    ("Y9e earlier turns left out of the packet", FC,
     '"tools": _split_tools(u, full)} for u in earlier],', '"tools": _split_tools(u, full)} for u in []],'),
    ("Y9f another channel counts as earlier", FC, "if u is not t and _channel(u) == ch\n", "if u is not t\n"),
    ("Y9g a later turn counts as earlier", FC, "and 0 <= now - _epoch(u.get(\"timestamp\")) <= HISTORY_WINDOW_S)",
     "and abs(now - _epoch(u.get(\"timestamp\"))) <= HISTORY_WINDOW_S)"),
    ("Y9h the sidecar is ignored", FC, '        side = full.get((str(t.get("id")), i))', '        side = None'),
    ("Y9i the sidecar is keyed off by one", FC, '        side = full.get((str(t.get("id")), i))',
     '        side = full.get((str(t.get("id")), i + 1))'),
    ("Y9j memory text is not capped", FC,
     "        cap = SPLIT_MEMORY_CHARS if _tname(c) in _SPLIT_CAPPED else SPLIT_TOOL_CHARS",
     "        cap = SPLIT_TOOL_CHARS"),
    ("Y9l documents are capped like owner facts", FC, '_SPLIT_CAPPED = _MEMORY_TOOLS - {"knowledge_base"}',
     '_SPLIT_CAPPED = _MEMORY_TOOLS'),
    ("Y9m a health check is world evidence", FC,
     '    if _tname(c) == "system_utility" and args.get("action") != "check_weather":\n        return False',
     '    if False:\n        return False'),
    ("Y9n a grader's corrected line counts twice", FC,
     '    last = {str(r.get("id")): r for r in _load_jsonl(grades)}\n    return [r for r in last.values()',
     '    last = list(_load_jsonl(grades))\n    return [r for r in last'),
    ("Y9k a row at the 4,000 cap with no full text is not marked cut", FC,
     '                            or (side is None and len(str(c.get("result") or "")) >= 4000)})', '                            })'),
    ("Y10 the split is redrawn", FC, "    if out.exists():\n        return {\"error\": f\"the {date} split",
     "    if False:\n        return {\"error\": f\"the {date} split"),
    ("Y11 a missing turn is booked as A", FC,
     "            if t is None:\n                missing.append(r[\"id\"])\n                continue",
     "            if t is None:\n                a.append(r[\"id\"])\n                continue"),
    ("Y12 the packet leaks the trajectory id", FC,
     '                "id": r["id"], "asked_on": str(t.get("timestamp") or "")[:10],',
     '                "id": r["id"], "tid": r["tid"], "asked_on": str(t.get("timestamp") or "")[:10],'),
    ("Y13 the grader's reason is left out", FC,
     '                "grader_reason": _redact_packet(str(r.get("reason") or ""))}',
     '                "grader_reason": ""}'),
    ("Y14 a long result is not marked cut", FC, '                     "cut": len(txt) > cap or bool(', '                     "cut": bool('),
    ("Y15 a long result is not cut", FC, '"returned": txt[:cap],', '"returned": txt,'),
    ("Y16 the brief is not written", FC, '        (stage / "brief.md").write_text(SPLIT_BRIEF, encoding="utf-8")\n', ""),
    ("Y17 the A floor is ignored", FC, "    if a_share >= SPLIT_A_FLOOR:\n        cands[\"A\"]",
     "    if True:\n        cands[\"A\"]"),
    ("Y18 the floor is strict", FC, "    if a_share >= SPLIT_A_FLOOR:", "    if a_share > SPLIT_A_FLOOR:"),
    ("Y19 ties go to C first", FC, '    order = ("B", "A", "C")', '    order = ("C", "A", "B")'),
    ("Y19b an A-C tie goes to C", FC, '    order = ("B", "A", "C")', '    order = ("B", "C", "A")'),
    ("Y20 DISAGREE leaves the denominator", FC,
     '    n = sum(counts.get(k, 0) for k in ("A", "B", "C", "OTHER", "DISAGREE"))',
     '    n = sum(counts.get(k, 0) for k in ("A", "B", "C", "OTHER"))'),
    ("Y20b unrated items join the denominator", FC,
     '    n = sum(counts.get(k, 0) for k in ("A", "B", "C", "OTHER", "DISAGREE"))',
     '    n = sum(counts.values())'),
    ("Y21 underpowered never flagged", FC, '"underpowered": n < SPLIT_MIN_N}', '"underpowered": False}'),
    ("Y22 a disagreement counts as the first reader's bucket", FC,
     '            counts[seen[0] if len(set(seen)) == 1 else "DISAGREE"] += 1', '            counts[seen[0]] += 1'),
    ("Y22b an item one of two readers skipped counts as agreed", FC,
     '        elif len(seen) < len(readers):\n            counts["PARTLY_READ"] += 1',
     '        elif False:\n            counts["PARTLY_READ"] += 1'),
    ("Y23 an A item a reader rated is counted", FC,
     '            if gid not in drawn or b not in SPLIT_BUCKETS:', '            if b not in SPLIT_BUCKETS:'),
    ("Y24 the bucket is not upper-cased", FC, '            b = str(r.get("bucket") or "").strip().upper()',
     '            b = str(r.get("bucket") or "").strip()'),
    ("Y25 an empty reader file is accepted", FC, "        if not got:\n            return {\"error\"",
     "        if False:\n            return {\"error\""),
    ("Y25b a conflicting duplicate keeps the last bucket", FC,
     "            elif got.get(gid, b) != b:", "            elif False:"),
    ("Y25c stale grades are accepted", FC,
     '    if _grades_digest(_split_rows(gdir / date / "grades.jsonl")) != meta.get("grades"):', "    if False:"),
    ("Y25f a changed reason is not stale", FC,
     '[str(r.get("id")), str(r.get("grade")), str(r.get("reason") or "")]', '[str(r.get("id")), str(r.get("grade"))]'),
    ("Y25g the same file twice is two readers", FC,
     "    if len({p.resolve() for p in paths}) < len(paths):", "    if False:"),
    ("Y25h identical files are not flagged", FC,
     '    identical = len(paths) > 1 and None not in blobs and len(set(blobs)) < len(blobs)', '    identical = False'),
    ("Y25d complete with one reader", FC, "missing == 0 and len(readers) > 1,", "missing == 0,"),
    ("Y25e complete with unread items", FC, '"complete": counts["UNREAD"] == counts["PARTLY_READ"] == missing == 0 and',
     '"complete": counts["PARTLY_READ"] == missing == 0 and'),
    ("Y26 the result is not saved", FC,
     '    (sd / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")\n', ""),
    ("Y27 the CLI ignores --draw-split", CLI, "    if ns.draw_split or ns.ingest_split:", "    if ns.ingest_split:"),
    ("Y28 the CLI split exit code is always 0", CLI,
     '               else FC.ingest_split(ns.ingest_split, date=ns.gold_date))\n        print(json.dumps(res, indent=1, ensure_ascii=False))\n        return 2 if res.get("error") else 0',
     '               else FC.ingest_split(ns.ingest_split, date=ns.gold_date))\n        print(json.dumps(res, indent=1, ensure_ascii=False))\n        return 0'),
    ("Y29 unread items are not shown", FC, '            counts["UNREAD"] += 1\n', "            pass\n"),
    ("Y30 the stage folder is left on failure", FC,
     "        shutil.rmtree(stage, ignore_errors=True)\n        raise\n    return {\"dir\": str(out)",
     "        raise\n    return {\"dir\": str(out)"),
    ("Y31 ingest does not count kinds", FC, '        kinds[x["kind"] or "none"] = kinds.get(x["kind"] or "none", 0) + 1', '        pass'),
]

if __name__ == "__main__":
    m.main()
