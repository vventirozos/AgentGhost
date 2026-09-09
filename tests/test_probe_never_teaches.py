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
    assert found == 7, f"{found} opt-ins — a new one is a decision, record it here"


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
    a.context = SimpleNamespace(llm_client=_LLM(), skill_memory=_Skills(), memory_system=None)
    return a


def test_perfect_it_writes_nothing_for_a_probe_turn():
    a = _agent()
    with patch.object(agent_mod, "turn_origin", lambda ctx: "probe"), \
         patch.object(agent_mod, "pretty_log", lambda *x, **k: None):
        out = asyncio.run(a._perfect_it_generate_and_learn({"messages": []}, "Optimization Analysis: Run exactly this", "tid"))
    assert a.context.skill_memory.lessons == [], "a probe turn became a playbook lesson"
    assert "verify exit codes" in out, "the text is still returned for the inline path's own use"


def test_perfect_it_still_writes_for_a_user_turn():
    a = _agent()
    with patch.object(agent_mod, "turn_origin", lambda ctx: "user"), \
         patch.object(agent_mod, "pretty_log", lambda *x, **k: None):
        asyncio.run(a._perfect_it_generate_and_learn({"messages": []}, "Compare two schema dumps", "tid"))
    assert len(a.context.skill_memory.lessons) == 1
    assert a.context.skill_memory.lessons[0]["source"] == "perfection_protocol"


from unittest.mock import AsyncMock
from tests.test_finalize_stream_pins import make_fin_agent, _fs


def _scheduled(origin):
    """Drive the real finalize path with one tool run and see whether the
    deferred Perfect-It coroutine was scheduled."""
    a = make_fin_agent()
    a._perfect_it_generate_and_learn = AsyncMock(return_value="")
    a.context.args.perfect_it = False
    with patch.object(agent_mod, "turn_origin", lambda ctx: origin):
        asyncio.run(a._finalize_and_return(_fs(
            final_ai_content="Done.",
            tools_run_this_turn=[{"name": "execute", "content": "ok"}])))
        # the deferred task runs on the loop; give it a turn
        asyncio.run(asyncio.sleep(0))
    return a._perfect_it_generate_and_learn.await_count > 0 or a._perfect_it_generate_and_learn.call_count > 0


def test_perfect_it_is_not_even_scheduled_for_a_probe_turn():
    """The deferred path is gated BEFORE the worker call — a probe must not
    cost a background LLM call either. Executed through the real finalize
    path; a source pin here survived an arm-swap of the gate (review)."""
    assert _scheduled("probe") is False


def test_perfect_it_is_scheduled_for_a_user_turn():
    """…and the gate is not "gate everything"."""
    assert _scheduled("user") is True


def test_orphan_twins_are_reconciled_at_boot_too():
    """A pruned lesson's vector twin is rendered VERBATIM by retrieval when
    its playbook row is gone, and the idle-phase reconcile needs 15-60 min of
    idle plus a 2 h cooldown. The boot path runs the same reconcile (review,
    2026-09-09). Pinned at the source: `main()` is not drivable here; the
    reconcile itself is pinned by the skills-store tests."""
    import inspect
    from ghost_agent import main as main_mod
    src = inspect.getsource(main_mod.main)
    i = src.index("reconcile_vector_orphans(_ms_boot)")
    window = src[max(0, i - 1500):i]
    assert "context.skill_memory = SkillMemory(memory_dir)" in window, "the boot reconcile must follow the persistent skill store"
    assert "Thread(" in src[i:i + 1200], "the boot reconcile must run off the loop"
