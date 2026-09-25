"""A stored member trajectory never teaches, even from an idle phase —
2026-09-24 (§4KJ).

The request-time gates read the request context; the idle phases
(reflection, dream seeds, macro mining, distillation, post-mortem) run
WITHOUT it, so nine hours after the request-time gate blocked a channel
member's turn, the reflection cycle acquired a playbook skill from that same
trajectory. Every trajectory now carries the requester's role in
`extra["requester_role"]`, and one predicate
(`memory.skills.trajectory_may_teach` / `iter_teachable`) sits in every
trajectory-reading lesson producer. No request-id prefix is consulted.
"""
import ast
import re
from pathlib import Path

import pytest

from ghost_agent.distill.schema import Trajectory, Outcome
from ghost_agent.memory.skills import trajectory_may_teach

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "ghost_agent"


@pytest.mark.parametrize("role,sid,expect", [
    ("member", "slack-23a1fa85", False), ("member", "web-1", False), ("MEMBER", "x", False),
    ("owner", "slack-23a1fa85", True),       # the id's spelling means nothing
    ("", "slack-23a1fa85", True), ("", "abc12345", True), (None, "", True),
    ("", "probe-x", True),                   # task KINDS are admissibility's job, not this rule's
])
def test_predicate_table(role, sid, expect):
    extra = {"requester_role": role} if role is not None else {}
    assert trajectory_may_teach(Trajectory(session_id=sid, extra=extra)) is expect


def test_never_raises():
    assert trajectory_may_teach(None) is True
    assert trajectory_may_teach(object()) is True


def test_reflection_skips_a_failed_slack_trajectory():
    from ghost_agent.reflection.loop import Reflector
    r = Reflector.__new__(Reflector)
    slack = Trajectory(session_id="slack-23a1fa85", task_kind="user_request", extra={"requester_role": "member"},
                       outcome=Outcome.FAILED.value, user_request="x", final_response="y")
    own = Trajectory(session_id="abcd1234", task_kind="user_request",
                     outcome=Outcome.FAILED.value, user_request="x", final_response="y")
    assert r._is_reflectable(slack) is False
    assert r._is_reflectable(own) is True


def test_dream_seed_fragments_skip_slack():
    from types import SimpleNamespace
    from ghost_agent.core.dream import trajectory_dream_fragments
    trajs = [Trajectory(session_id="slack-1", user_request="member ask", final_response="r", extra={"requester_role": "member"}),
             Trajectory(session_id="own1", user_request="owner ask", final_response="r")]
    collector = SimpleNamespace(iter_trajectories=lambda: iter(trajs))
    ctx = SimpleNamespace(trajectory_collector=collector)
    ids, docs = trajectory_dream_fragments(ctx, limit=40)
    assert len(ids) == 1 and "owner ask" in " ".join(docs)


# The lesson producers that read the trajectory store. Anything that
# iterates trajectories to LEARN from them must filter through the
# predicate; the allowlist names the readers that measure or train models
# rather than mint lessons (a separate decision, noted in §4KJ).
TEACHING_SITES = {
    # reflection's source is the `_biological_tick` lambda below; its filter
    # lives in `Reflector._is_reflectable` (behavioural pin above)
    ("core/dream.py", "trajectory_dream_fragments"),
    ("core/replay_engine.py", "_iter_real"),        # dream replay REPLAYS turns into learning (R6)
    ("core/dream.py", "_propose_macros_sync"),
    ("core/agent.py", "_biological_tick"),          # post-mortem + skills_auto sources
    # §4KJ R9: foresight precedent reaches owner prompts; cluster counts steer self-play
    ("core/foresight.py", "_seed_from_trajectories"),
    ("core/dream.py", "synthetic_self_play"),
}
NON_TEACHING_ALLOWLIST = {
    # kwargs callers (R2 review): world model, feedback ledger, replay, experiments, health
    ("core/feedback.py", "find_trajectory_for_request"),
    ("core/experiments.py", "_summaries_from_trajectories"),
    ("core/learning_health.py", "_framing_leak_health"),
    ("main.py", "_lesson_sink"), ("main.py", "lifespan"),          # lifespan: router bootstrap (a model, not a lesson)
    ("tools/memory.py", "_maybe_retrain_prm"),
    ("tools/memory.py", "_maybe_retrain_router"),
    ("core/admissibility.py", "iter_bench_trajectories"), ("core/experiments.py", "announce_new_verdicts"),
    ("core/learning_health.py", "_experiment_health_lines"), ("core/agent.py", "_verify_fn"),
    ("core/agent.py", "_run_prm_online_update"), ("distill/collector.py", "_default_root"),
    ("distill/collector.py", "corpus_fingerprint"), ("distill/collector.py", "iter_trajectories"),
    ("distill/collector.py", "<module>"),                                   # a docstring example
    ("optim/tool_fixtures.py", "<module>"), ("optim/tool_fixtures.py", "_trajectory_index"),
    ("prm/trainer.py", "<module>"), ("prm/trainer.py", "summary"),
}


FILTER_NAMES = ("trajectory_may_teach", "iter_teachable")
WRAPPER_NAMES = {"iter_teachable", "_iter_teachable_pm", "_iter_teachable_sa", "_iter_teachable_refl"}


def _trainer_names(fn):
    """Names bound in `fn` to a `<Something>Trainer(...)` construction — the
    honest shape of the training exemption (a name alone is a convention)."""
    out = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id.endswith("Trainer")):
            out.update(t.id for t in n.targets if isinstance(t, ast.Name))
    return out


def _consumers():
    """(file, innermost enclosing def) → every read wrapped? for every read
    of `iter_trajectories` under src/ghost_agent: a `.iter_trajectories(...)`
    call (any arguments), a bare `.iter_trajectories` attribute (an alias),
    or `getattr(x, "iter_trajectories")`. A read is wrapped when an ancestor
    Call in the same expression is one of WRAPPER_NAMES (exact), or it is the
    `trajectories=` argument of `<trainer>.run(...)` / `asyncio.to_thread(
    <trainer>.run, …)` where `<trainer>` is bound to a `…Trainer(...)` in the
    same def (PRM / router checkpoints — models, not lessons)."""
    out = {}
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        par = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                par[ch] = node
        rel = str(path.relative_to(SRC))
        reads = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == "iter_trajectories":
                reads.append(n)
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
                  and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "iter_trajectories"):
                reads.append(n)
        for n in reads:
            fn = par.get(n)
            while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = par.get(fn)
            trainers = _trainer_names(fn) if fn is not None else set()
            node = par.get(n); wrapped = False; hops = 0
            while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.Call) and hops < 6:
                    fname = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                    if fname in WRAPPER_NAMES:
                        wrapped = True
                    if (fname == "run" and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id in trainers):
                        wrapped = True
                    if (fname == "to_thread" and node.args and isinstance(node.args[0], ast.Attribute)
                            and node.args[0].attr == "run" and isinstance(node.args[0].value, ast.Name)
                            and node.args[0].value.id in trainers):
                        wrapped = True
                node = par.get(node); hops += 1
            key = (rel, fn.name if fn is not None else "<module>")
            out[key] = out.get(key, True) and wrapped
            READS[key] = READS.get(key, 0) + 1
    return out


READS = {}


def test_iter_teachable_filters_and_tolerates_none():
    from ghost_agent.memory.skills import iter_teachable
    trajs = [Trajectory(session_id="m1", extra={"requester_role": "member"}), Trajectory(session_id="own"),
             Trajectory(session_id="m2", extra={"requester_role": "member"})]
    assert [t.session_id for t in iter_teachable(trajs)] == ["own"]
    assert list(iter_teachable(None)) == [] and list(iter_teachable(iter([]))) == []


def test_every_trajectory_consumer_is_classified_and_every_teacher_filters():
    found = _consumers()
    names = set(found)
    unknown = names - TEACHING_SITES - NON_TEACHING_ALLOWLIST
    assert not unknown, f"new trajectory consumer(s) must be classified: {sorted(unknown)}"
    missing = TEACHING_SITES - names
    assert not missing, f"teaching site(s) no longer read trajectories?: {sorted(missing)}"
    unfiltered = {k for k, ok in found.items() if k in TEACHING_SITES and not ok}
    assert not unfiltered, f"teaching site(s) with an unwrapped read: {sorted(unfiltered)}"


@pytest.mark.parametrize("body", [
    "return [t for t in c.iter_trajectories(day=None)]",                        # kwargs (R2 review)
    "it = c.iter_trajectories\n    return list(it())",                          # a local alias (R3 review)
    "return list(getattr(c, 'iter_trajectories')())",                            # getattr (R3 review)
    "return list(iter_teachable_not_really(c.iter_trajectories()))",             # a look-alike wrapper (R3 review)
    "trainer = miner\n    return trainer.run(trajectories=c.iter_trajectories())",   # a NAME called trainer, no Trainer() (R3 review)
])
def test_the_enumeration_sees_every_bypass_shape(tmp_path, monkeypatch, body):
    fake = tmp_path / "ghost_agent"; fake.mkdir()
    (fake / "newmod.py").write_text(f"def _new_teacher_bypass(c):\n    {body}\n", encoding="utf-8")
    monkeypatch.setattr("tests.test_trajectory_may_teach.SRC", fake)
    found = _consumers()
    assert ("newmod.py", "_new_teacher_bypass") in found
    assert found[("newmod.py", "_new_teacher_bypass")] is False


def test_a_real_trainer_read_is_exempt_by_shape(tmp_path, monkeypatch):
    fake = tmp_path / "ghost_agent"; fake.mkdir()
    (fake / "newmod.py").write_text(
        "def _retrain(c):\n    trainer = PRMTrainer()\n    return trainer.run(trajectories=c.iter_trajectories())\n",
        encoding="utf-8")
    monkeypatch.setattr("tests.test_trajectory_may_teach.SRC", fake)
    assert _consumers()[("newmod.py", "_retrain")] is True



def test_reflection_reads_the_role_not_the_id():
    """R4 pins review: the member row used a slack- id, so a prefix rule
    passed too. A member on a web id is skipped; an owner on a slack- id is
    reflected."""
    from ghost_agent.reflection.loop import Reflector
    r = Reflector.__new__(Reflector)
    member_web = Trajectory(session_id="web-1", task_kind="user_request", extra={"requester_role": "member"},
                            outcome=Outcome.FAILED.value, user_request="x", final_response="y")
    owner_slack = Trajectory(session_id="slack-x", task_kind="user_request", extra={"requester_role": "owner"},
                             outcome=Outcome.FAILED.value, user_request="x", final_response="y")
    assert r._is_reflectable(member_web) is False
    assert r._is_reflectable(owner_slack) is True



def test_an_allowlisted_reader_does_not_grow_new_reads():
    """R4 pins review: the allowlist exempts whole functions, so a NEW
    unwrapped read added inside an allowlisted function passed. Each
    allowlisted key's read count is pinned; a new read fails here until it
    is classified."""
    READS.clear()
    found = _consumers()
    counts = {k: READS[k] for k in found if k in NON_TEACHING_ALLOWLIST}
    assert counts == EXPECTED_ALLOWLISTED_READS, counts


EXPECTED_ALLOWLISTED_READS = {   # one read each, as of 2026-09-24 — a new read must be classified
    ("main.py", "lifespan"): 1, ("tools/memory.py", "_maybe_retrain_prm"): 1,
    ("tools/memory.py", "_maybe_retrain_router"): 1, 
    ("core/feedback.py", "find_trajectory_for_request"): 1, ("core/admissibility.py", "iter_bench_trajectories"): 1,
    ("core/experiments.py", "_summaries_from_trajectories"): 1, ("core/experiments.py", "announce_new_verdicts"): 1,
    ("core/learning_health.py", "_framing_leak_health"): 1, ("core/learning_health.py", "_experiment_health_lines"): 1,
    ("core/agent.py", "_run_prm_online_update"): 1, ("optim/tool_fixtures.py", "_trajectory_index"): 1,
}



def test_experiment_summaries_never_count_a_members_turn():
    """§4KJ R9: every A/B reader (verdicts, health lines, announcements) goes
    through `summarize_streaming`, so the filter lives there, once."""
    from ghost_agent.core.experiments import summarize_streaming
    from ghost_agent.distill.schema import Trajectory
    own = Trajectory(session_id="o", task_kind="user_request")
    mem = Trajectory(session_id="m", task_kind="user_request", extra={"requester_role": "member"})
    _all, _trig, cov_both = summarize_streaming([own, mem])
    _all, _trig, cov_own = summarize_streaming([own])
    assert cov_both == cov_own and cov_own.get("user_turns") == 1, (cov_both, cov_own)
