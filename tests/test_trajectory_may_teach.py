"""A stored member trajectory never teaches, even from an idle phase —
2026-09-24 (§4KJ).

The request-time gates read the request context; the idle phases
(reflection, dream seeds, macro mining, distillation, post-mortem) run
WITHOUT it, so nine hours after the request-time gate blocked a channel
member's turn, the reflection cycle acquired a playbook skill from that same
trajectory. Every trajectory now carries the requester's role in
`extra["requester_role"]`, and one predicate
(`memory.population.trajectory_may_teach` / `iter_teachable`) sits in every
trajectory-reading lesson producer. No request-id prefix is consulted.
§4NM: reflection, the dream seeds, macro mining, the post-mortem and the PRM
retrains were removed; the live readers are the owner-failure seed pickers.
"""
import ast
import re
from pathlib import Path

import pytest

from ghost_agent.distill.schema import Trajectory, Outcome
from ghost_agent.memory.population import trajectory_may_teach

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


# The lesson producers that read the trajectory store. Anything that
# iterates trajectories to LEARN from them must filter through the
# predicate; the allowlist names the readers that measure or train models
# rather than mint lessons (a separate decision, noted in §4KJ).
TEACHING_SITES = {
    # §4MT: the reaction judge labels seeds; withdrawal retires lessons on owner failures
    # §4MW: the failure-replay loop picks owner failures (iter_teachable)
    # §4NE: regression-test candidates are owner failures (iter_teachable)
    # §4QL: the human-graded experience index is built from owner turns (iter_teachable)
    ("memory/experience.py", "sync_experience_index"),
}
NON_TEACHING_ALLOWLIST = {
    # kwargs callers (R2 review): world model, feedback ledger, replay, health
    # (§4NN: the experiments readers left with core/experiments.py)
    ("core/feedback.py", "find_trajectory_for_request"),
    ("core/learning_health.py", "_framing_leak_health"),
    ("core/agent.py", "_verify_fn"),
    ("distill/collector.py", "_default_root"),
    ("distill/collector.py", "corpus_fingerprint"), ("distill/collector.py", "iter_trajectories"),
    ("distill/collector.py", "<module>"),                                   # a docstring example
    # §4MW: looks up its OWN probe replays by request id; teaches nothing
    # §4NE: reads its OWN test run (probe-rt-…) back by request id; teaches nothing
    # §4PL: the failure census — a REPORT over owner turns (its own owner
    # filter, members excluded); it writes no lesson, label or prompt
    ("core/failure_census.py", "_owner_turns"),     # §4PL/§4PM: the census report and the gold draw
}


WRAPPER_NAMES = {"iter_teachable"}


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
    from ghost_agent.memory.population import iter_teachable
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
    ("core/feedback.py", "find_trajectory_for_request"): 1,
    ("core/learning_health.py", "_framing_leak_health"): 1,
    ("core/failure_census.py", "_owner_turns"): 1,                   # §4PL/§4PM: a report, one read per day
}
