#!/usr/bin/env python3
"""§4PP battery — temporary video transcripts (memory/temp_documents.py, the
knowledge_base hooks, the store-hygiene sweep) and the Slack conversation
key kept out of the session store, 2026-10-16. Same harness as §4PM, on a
tree COPY (`BATTERY_ROOT`).

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4pp.py [all|rerun T1 …]
"""
import os, runpy
_H = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4pm.py"), run_name="harness")
TD = "src/ghost_agent/memory/temp_documents.py"
KB = "src/ghost_agent/tools/memory.py"
AG = "src/ghost_agent/core/agent.py"
RT = "src/ghost_agent/api/routes.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", TD, "def _is_video_transcript(memory_system, filename: str) -> bool:", "# ctl\ndef _is_video_transcript(memory_system, filename: str) -> bool:"),
    ("CONTROL known-bad (must DIE): nothing is ever temporary", TD, "def is_temporary(memory_system, filename: str) -> bool:\n", "def is_temporary(memory_system, filename: str) -> bool:\n    return False\n"),
    ("T1 expired never removed", TD, "                if exp and float(exp) <= now:", "                if False:"),
    ("T2 removed before expiry", TD, "                if exp and float(exp) <= now:", "                if exp:"),
    ("T3 kept re-marked", TD, "            if cur.get(\"kept_at\"):\n                return None", "            if False:\n                return None"),
    ("T4 old transcripts not marked", TD, "                if name not in data and _is_video_transcript(memory_system, name):", "                if False:"),
    ("T6 gone entries kept", TD, "                if name not in library and not e.get(\"kept_at\"):", "                if False:"),
    ("T7 corrupt registry read as empty", TD, "        logger.warning(\"temporary-document registry unreadable (%s); nothing will be removed\", e)\n        raise", "        return {}"),
    ("T8 failed delete drops the entry", TD, "                        logger.warning(\"temporary document %s not removed: %s\", name, ex)\n                        continue", "                        logger.warning(\"temporary document %s not removed: %s\", name, ex)"),
    ("T11 bad TTL accepted", TD, "        return v if v > 0 else DEFAULT_TTL_DAYS", "        return v"),
    ("T12 the ingest note not appended", KB, "        return str(res.message) + await asyncio.to_thread(_video_retention_note, memory_system, res, _had)", "        return res.message"),
    ("T13 keep ignores the owner's words", KB, "        if request_kind() == \"owner\" and TD.owner_asked_to_keep(user_message()):", "        if request_kind() == \"owner\":"),
    ("T14 probes can keep", KB, "        if request_kind() == \"owner\" and TD.owner_asked_to_keep(user_message()):", "        if TD.owner_asked_to_keep(user_message()):"),
    # T24 is EQUIVALENT: an empty forgiving read meets the empty-library guard (T25)
    # T15 is EQUIVALENT: mark() itself refuses a kept document (two guards)
    ("T15 kept doc re-marked by the tool", KB, "        if entry.get(\"kept_at\"):\n            return \"\"                                   # the owner kept it", "        if False:\n            return \"\"                                   # the owner kept it"),
    ("T16 read-only isolate writes", KB, "            if isinstance(memory_system, ReadOnlyVectorMemory):\n                return \"\"", "            if False:\n                return \"\""),
    ("T18 keep of an unknown doc", KB, "        if _name not in _lib:\n            return ToolOutcome.rejected", "        if False:\n            return ToolOutcome.rejected"),
    ("T19 list shows no TEMPORARY", KB, "                    facts.append(f\"TEMPORARY — removed on {_TD.expiry_note(_te['expires_at'])}\")", "                    pass"),
    ("T20 keep not advertised", KB, "    \"expand\", \"forget\", \"list_docs\", \"reset_all\", \"keep\",", "    \"expand\", \"forget\", \"list_docs\", \"reset_all\","),
    ("T21 idle job never sweeps", AG, "                        _tds = await asyncio.to_thread(_TD.sweep, _ms)", "                        _tds = {\"removed\": [], \"marked\": [], \"dropped\": []}"),
    ("T22 slack key reaches the store", RT, "    if is_conversation_key_only(_session_id):", "    if False:"),
    ("T23 every key skips the store", RT, "    return isinstance(session_id, str) and session_id.startswith(_CONVERSATION_KEY_PREFIXES)", "    return isinstance(session_id, str)"),
    ("T5 filename heuristics mark owner files", TD, "    return bool(str(rec.get(\"video_id\") or \"\").strip())", "    return bool(str(rec.get(\"video_id\") or \"\").strip()) or str(filename).startswith(\"yt-\")"),
    ("T24 the forgiving library read", TD, "    load = getattr(memory_system, \"_load_library\", None)", "    load = None"),
    ("T25 empty library drops entries", TD, "            if not library and data and not _library_is_authoritative(memory_system):", "            if False:"),
    ("T27 failed ingest renews", KB, "        if not getattr(res, \"ok\", False):\n            return \"\"", "        if False:\n            return \"\""),
    ("T9 negation ignored", TD, "    return bool(t) and any(rx.search(t) for rx in _KEEP_RES) and not _NEGATED_RE.search(t)", "    return bool(t) and any(rx.search(t) for rx in _KEEP_RES)"),
    ("T10 any keep word keeps at ingest", TD, "    return bool(t) and any(rx.search(t) for rx in _KEEP_RES) and not _NEGATED_RE.search(t)", "    return bool(t) and bool(_re.search(r\"\\bkeep\\b|\\bsave\\b\", t)) and not _NEGATED_RE.search(t)"),
    ("T17 keep open to probes", KB, "        if _rk_keep() != \"owner\":\n            return ToolOutcome.rejected(\n                f\"Not kept: only the owner", "        if False:\n            return ToolOutcome.rejected(\n                f\"Not kept: only the owner"),
    ("T28 forget keeps the record", "src/ghost_agent/memory/vector.py", "                _TD.forget_entry(self, filename)", "                pass"),
    ("T29 forget_entry drops nothing", TD, "            if data.pop(str(filename), None) is not None:\n                _save(memory_system, data)", "            if False:\n                _save(memory_system, data)"),
    ("T30 clear clears nothing", TD, "        with (_NoLock() if held else _lock(memory_system)):\n            _save(memory_system, {})", "        with (_NoLock() if held else _lock(memory_system)):\n            pass"),
    ("T31 authoritative empty library skipped", TD, "            if not library and data and not _library_is_authoritative(memory_system):", "            if not library and data:"),
    ("T32 the Greek store phrase", TD, "    _re.compile(r\"(?:\u03b1\u03c0\u03bf\u03b8[\u03b7\u03ae]\u03ba\u03b5\u03c5\u03c3|\u03c0\u03c1[\u03bf\u03cc]\u03c3\u03b8\u03b5\u03c3|\u03b2[\u03b1\u03ac]\u03bb|\u03ba\u03c1[\u03b1\u03ac]\u03c4)\\w*\\s+[^.;!\\n]{0,30}?\u03c3\u03c4\u03b7?\u03bd?\\s+\u03b2[\u03b1\u03ac]\u03c3\u03b7\\s+\u03b3\u03bd[\u03c9\u03ce]\u03c3\", _re.I),", "    _re.compile(r\"(?!x)x\", _re.I),"),
    ("T33 keep after outside content", KB, "        if _us_keep() and not _TD.owner_asked_to_keep(_um_keep()):", "        if False:"),
    ("T34 keep after outside content ignores the owner's words", KB, "        if _us_keep() and not _TD.owner_asked_to_keep(_um_keep()):", "        if _us_keep():"),
    ("T35 kept dropped on absence", TD, "                if name not in library and not e.get(\"kept_at\"):", "                if name not in library:"),
    ("T36 keep reads as outside content", "src/ghost_agent/utils/provenance.py", "                           \"keep\"})        # \u00a74PP: marks a document kept; reads nothing", "                           })"),
    ("T37 reconciler keeps the record", "src/ghost_agent/memory/vector.py", "                _TD.forget_entry(self, str(name))", "                pass"),
]
_H["MUT"][:] = MUT
_H["PINS"][:] = ["tests/test_4pp_temp_documents.py", "tests/test_sessions_and_cancel.py"]
g = _H["main"].__globals__
g["MUT"], g["PINS"] = MUT, _H["PINS"]
g["_FULL_PAT"] = r"temp_documents\|tool_knowledge_base\|is_conversation_key_only\|_make_app\|list_docs\|_biological_tick"
#: tests that read files OUTSIDE the tree (the repo's parent bin/ and update
#: scripts) — they fail in any copy, so they are not part of this battery
_LOCATION_BOUND = {"tests/test_4pi_youtube_review.py", "tests/test_4kh_youtube_canary.py", "tests/test_ghost_cli.py"}
_orig_grep = g["_grep"]
g["_grep"] = lambda pat: _orig_grep(pat) - _LOCATION_BOUND
if __name__ == "__main__":
    _H["main"]()
