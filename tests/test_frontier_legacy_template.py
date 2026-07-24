"""Regression: FrontierTracker.record_run must survive a LEGACY persisted
template that predates the per-template "runs" key.

`templates.setdefault(template_key, {...defaults...})` no-ops on an existing
dict, so an old on-disk template (written before "runs" was added to the
template schema) came back WITHOUT "runs"; the bare `tstats["runs"] += 1`
then raised KeyError('runs') and aborted record_run — surfaced live as a
recurring "Frontier record_run failed: 'runs'". (The *cluster* level was
already back-filled by _ensure_cluster; the template was the remaining gap.)
"""
from __future__ import annotations

import json

from ghost_agent.memory.frontier import FrontierTracker


def test_record_run_survives_legacy_template_without_runs_key(tmp_path):
    ft = FrontierTracker(tmp_path)
    p = tmp_path / "self_play_frontier.json"

    # Create a template the normal way.
    ft.record_run(cluster_key="python_general", challenge="c1", attempts_used=1,
                  passed=True, description_length=50, template_key="tmplA")

    # Corrupt it into a legacy shape: drop the "runs" key from the template.
    state = json.loads(p.read_text())
    state["clusters"]["python_general"]["templates"]["tmplA"].pop("runs", None)
    p.write_text(json.dumps(state))

    # Previously raised KeyError('runs'); must record cleanly now.
    ft.record_run(cluster_key="python_general", challenge="c2", attempts_used=1,
                  passed=False, description_length=60, template_key="tmplA")

    after = json.loads(p.read_text())
    assert after["clusters"]["python_general"]["templates"]["tmplA"]["runs"] >= 1
