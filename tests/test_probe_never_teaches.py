"""Diagnostics never teach — at the place every learner reads (§4FS, 2026-09-09).

§4FB gated the three OUTCOME-CREDIT sites on `turn_origin == "probe"`. The
three LESSON producers were never gated. Found in the playbook:

  * 3 lessons stamped `origin=probe` by the perfection protocol
    ("Optimization Analysis: Run exactly this in the sandbox: echo …");
  * 2 more distilled from probe trajectories by the dream seed and the
    post-mortem, stamped `auto` — "When executing shell commands, ensure
    the command is run verbatim…", a rule about the operator's diagnostic
    phrasing, hydrated into real shell tasks.

Every reader of `TrajectoryCollector.iter_trajectories()` (~40: dream
seeds, post-mortem, fixture mining, experiment stats, backtests) is a
learner or a report, so the gate lives there. The perfection protocol does
not read the collector and is gated at both its scheduling and write site.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.core import agent as agent_mod


def _root(tmp_path, rows):
    day = tmp_path / "2026-09-09"; day.mkdir(parents=True)
    with open(day / "session-abc.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return tmp_path


ROWS = [
    {"id": "u1", "task_kind": "user_request", "user_request": "download and ingest the manual"},
    {"id": "p1", "task_kind": "probe", "user_request": "Run exactly this and report the exit code"},
    {"id": "b1", "task_kind": "bench", "user_request": "bench row"},
    {"id": "s1", "task_kind": "self_play", "user_request": "self-play row"},
]


# --- the collector ---------------------------------------------------------

def test_probe_trajectories_are_skipped_by_default(tmp_path):
    """THE REGRESSION. World where it fails: the dream seed, the post-mortem
    and fixture mining all see the operator's diagnostics as data."""
    col = TrajectoryCollector(root=_root(tmp_path, ROWS), session_id="reader")
    ids = [t.id for t in col.iter_trajectories()]
    assert "p1" not in ids, ids
    assert ids == ["u1", "b1", "s1"], "only the probe row goes — bench and self-play are other consumers' business"


def test_a_caller_that_wants_probes_says_so(tmp_path):
    col = TrajectoryCollector(root=_root(tmp_path, ROWS), session_id="reader")
    ids = [t.id for t in col.iter_trajectories(include_probes=True)]
    assert ids == ["u1", "p1", "b1", "s1"]


def test_the_day_filter_composes_with_the_probe_gate(tmp_path):
    col = TrajectoryCollector(root=_root(tmp_path, ROWS), session_id="reader")
    assert [t.id for t in col.iter_trajectories(day="2026-09-09")] == ["u1", "b1", "s1"]
    assert [t.id for t in col.iter_trajectories(day="2026-01-01")] == []


def test_every_opt_in_states_its_reason():
    """Enumerated from the source: each `include_probes=True` must carry a
    comment naming why nearby — the default is the rule, an opt-in is an
    exception someone owns. 7 sites exist today (the framing-leak gauge, the
    ontology purity report ×3, the human-feedback join, the escalation audit,
    the collector's own counter); the count is pinned so a new one is a decision."""
    import re
    root = Path(__file__).resolve().parents[1]
    found, offenders = 0, []
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        if path.name == "collector.py":
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"include_probes\s*=\s*True", src):
            found += 1
            # the reason may sit above the call or trail it on the same line
            window = src[max(0, m.start() - 400):m.end() + 200].replace("include_probes", "")
            if "probe" not in window.lower():
                offenders.append(f"{path.relative_to(root)}:{src[:m.start()].count(chr(10)) + 1}")
    assert not offenders, offenders
    # 8: §4MW — the failure-replay loop looks its own probe replays up by id
    # 9: §4NE — the regression suite reads its own test runs back by id
    # §4NM: the three ontology-purity opt-ins left with optim/tool_ontology.py → 6
    # §4SZ: 8 and 9 above left with the replay loop and the regression suite → 4
    assert found == 4, f"{found} opt-ins — a new one is a decision, record it here"


# --- the perfection protocol ----------------------------------------------

class _LLM:
    async def chat_completion(self, payload, **kw):
        return {"choices": [{"message": {"content": "Always verify exit codes with a second command."}}]}


class _Skills:
    def __init__(self):
        self.lessons = []

    def learn_lesson(self, **kw):
        self.lessons.append(kw)


def _agent():
    a = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
    a.context = SimpleNamespace(llm_client=_LLM(), memory_system=None)
    return a


def test_perfect_it_returns_the_text_for_the_inline_path():
    a = _agent()
    with patch.object(agent_mod, "pretty_log", lambda *x, **k: None):
        out = asyncio.run(a._perfect_it_generate({"messages": []}))
    assert "verify exit codes" in out, "the text is returned for the inline path's own use"


# §4NM: the deferred (background) Perfect-It scheduling pins were removed — §4NC
# skipped that lesson-only phase and §4NM removed its restoring switch; only the
# direct-call gate above remains.


