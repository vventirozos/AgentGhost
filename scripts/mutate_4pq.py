#!/usr/bin/env python3
"""§4PQ battery — the idle-cycle lines say what each job is for and what it
did (core/agent.py idle_cycle_line + per-job notes, core/dream.py
maintenance_summary), 2026-10-16. Same harness as §4PM, on a tree COPY.

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4pq.py [all|rerun Q1 …]
"""
import os, runpy
_H = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4pm.py"), run_name="harness")
AG = "src/ghost_agent/core/agent.py"
DR = "src/ghost_agent/core/dream.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def idle_cycle_line(ran, notes, idle_min: float) -> str:", "# ctl\ndef idle_cycle_line(ran, notes, idle_min: float) -> str:"),
    ("CONTROL known-bad (must DIE): the old bare line", AG, "    return f\"idle {idle_min:.0f}m — ran \" + \" | \".join(parts)", "    return f\"ran {', '.join(ran)} (idle {idle_min:.0f}m)\""),
    ("Q1 purpose dropped", AG, "        text = base + (f\" ({purpose})\" if purpose else \"\")", "        text = base"),
    ("Q2 outcome dropped", AG, "        parts.append(text + (f\": {outcome}\" if outcome else \"\"))", "        parts.append(text)"),
    ("Q3 failure reads as its last outcome", AG, "                   if failed else note)", "                   if False else note)"),
    ("Q4 tidy outcome never set", AG, "                        _idle_notes[\"tidy\"] = (", "                        _unused = ("),
    ("Q6 hygiene note never set", AG, "                    _idle_notes[\"store-hygiene\"] = _store_hygiene_note(", "                    _unused = _store_hygiene_note("),
    ("Q7 failures alone read as nothing to fix", AG, "        return \"; \".join(bad) if bad else \"nothing to fix\"", "        return \"nothing to fix\""),
    ("Q10 failed backup reads quiet", DR, "    idle = not (d or f or snap or rec or graph_ran == \"failed\"", "    idle = not (d or f or rec or graph_ran == \"failed\""),
    ("Q11 graph not-due reads as cleaned", DR, "    elif graph_ran:\n        graph = (", "    elif True:\n        graph = ("),
    ("Q12 backup name dropped", DR, "        backup = \"store backup taken (\" + snap.replace(\"snapshot \", \"\", 1) + \")\"", "        backup = \"store backup taken\""),
    ("Q13 failed backup called taken", DR, "    elif \"FAILED\" in snap:", "    elif False:"),
    ("Q14 the pass still says counters", DR, "        _summary = maintenance_summary(", "        _summary = \"maintenance: \" + maintenance_summary("),
    ("Q16 cycle note from side_output", AG, "                                else \"nothing needed\" if _summary.startswith(\"nothing needed\")", "                                else \"nothing needed\" if not _wrote"),
    ("Q17 reconcile not passed", DR, "            self._maint_counts.get(\"reconcile\", \"\"))", "            \"\")"),
    ("Q18 reconcile not stamped", DR, "                self._maint_counts[\"reconcile\"] = _rec", "                pass"),
    ("Q19 reconcile ignored in nothing-needed", DR, "    idle = not (d or f or snap or rec or graph_ran == \"failed\"", "    idle = not (d or f or snap or graph_ran == \"failed\""),
    ("Q20 raised maintenance reads cleaned", DR, "                _graph_due = \"failed\"   # §4PQ r1: not \"cleaned\"", "                pass"),
    ("Q21 failed graph reads cleaned", DR, "    if graph_ran == \"failed\":\n        graph = \"memory graph cleanup FAILED\"\n    elif graph_ran:", "    if graph_ran:"),
    ("Q22 failed step hides its outcome", AG, "        outcome = ((\"had an error (see the log)\" + (f\"; then {note}\" if note else \"\"))", "        outcome = ((\"had an error (see the log)\" + (\"\" if note else \"\"))"),
    ("Q23 raising narrative not listed", AG, "                        _idle_ran.append(\"selfhood-narrative\")\n                        _idle_notes[\"selfhood-narrative\"] = \"FAILED (see the warning above)\"", "                        _idle_notes[\"selfhood-narrative\"] = \"FAILED (see the warning above)\""),
    ("Q24 error outcome reads did work", AG, "                                if isinstance(_outcome, dict) and _outcome.get(\"phase\") == \"error\"   # r2 m2", "                                if False"),
    ("Q25 skipped tidy counted as checked", AG, "                            if _tst.startswith(\"skipped\") and _tst != \"skipped: no project dir\":", "                            if False:"),
    ("Q26 no project dir counted as a fault", AG, "                            if _tst.startswith(\"skipped\") and _tst != \"skipped: no project dir\":", "                            if _tst.startswith(\"skipped\"):"),
]
_H["MUT"][:] = MUT
_H["PINS"][:] = ["tests/test_4pq_idle_lines.py", "tests/test_4nm_batch5.py", "tests/test_bio_r1_fixes.py",
                 "tests/test_idle_scheduler_logging.py", "tests/test_logging_stream_fixes.py"]
g = _H["main"].__globals__
g["MUT"], g["PINS"] = MUT, _H["PINS"]
g["_FULL_PAT"] = r"idle_cycle_line\|maintenance_summary\|_store_hygiene_note\|_biological_tick\|last_dream_outcome"
_LOCATION_BOUND = {"tests/test_4pi_youtube_review.py", "tests/test_4kh_youtube_canary.py", "tests/test_ghost_cli.py"}
_orig_grep = g["_grep"]
g["_grep"] = lambda pat: _orig_grep(pat) - _LOCATION_BOUND
if __name__ == "__main__":
    _H["main"]()
