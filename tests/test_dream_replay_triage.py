"""§4CM D0 — the Dream replay corpus and its triage.

The triage decides which recorded turns can honestly be re-executed. It
is the phase's input filter, so both directions are load-bearing: letting
an irreproducible episode through produces a verdict about noise, and
over-filtering produces a phase with nothing to work on. Both failures
look like "a number" from outside, which is why the census reports the
per-reason funnel rather than a count.

Pinned here:
  * the allow-list is fail-CLOSED — an unknown tool (a runtime-registered
    acquired skill, anything added later) is not replayable;
  * each filter is independently necessary, with the reason it reports;
  * a spec is only generated when its PREMISE is satisfied (a withhold
    needs a lesson the turn really hydrated; a swap needs a step that
    really failed);
  * `spec_id` is the identity of the QUESTION, so a re-run is
    recognisable as a re-run;
  * the two admissibility rows say different things, and `dream_credit`
    is closed.
"""
import json
from types import SimpleNamespace

import pytest

from ghost_agent.core import replay_engine as RE


def _tc(name, **args):
    return SimpleNamespace(name=name, arguments=args, result="ok", error="")


def _traj(*, id="t1", kind="user_request", outcome="passed", steps=3,
          tools=("file_system", "execute"), request="fix the parser",
          extra=None, results=None):
    calls = [_tc(t) for t in tools]
    if results:
        for c, r in zip(calls, results):
            c.result = r
    return SimpleNamespace(
        id=id, task_kind=kind, outcome=outcome, n_steps=steps,
        tool_calls=calls, user_request=request, extra=extra or {})


# ------------------------------------------------------------------ #
# The allow-list is fail-closed                                      #
# ------------------------------------------------------------------ #

def test_a_replayable_episode_is_admitted():
    t = RE.triage(_traj())
    assert t.replayable is True and t.reason == ""


def test_an_unknown_tool_is_not_replayable():
    """An acquired skill is registered at RUNTIME under a name this
    module has never seen and runs arbitrary code. A denylist cannot name
    it; an allow-list does not have to."""
    t = RE.triage(_traj(tools=("file_system", "naftemporiki_headlines")))
    assert t.replayable is False and t.reason == RE.REJECT_TOOL


@pytest.mark.parametrize("tool", [
    "web_search", "browser", "deep_research", "vision_analysis",
    "image_generation", "darkweb_search",
])
def test_network_tools_are_not_replayable(tool):
    """Not (only) a safety call: a search returns different results than
    it did on the day, so both legs diverge for reasons that have nothing
    to do with the perturbation and the paired rule abstains. A verdict
    about noise is worse than no verdict."""
    assert RE.triage(_traj(tools=("file_system", tool))).reason == \
        RE.REJECT_TOOL


@pytest.mark.parametrize("tool", [
    "manage_projects", "postgres_admin", "notify_operator", "delegate",
])
def test_containment_forbidden_tools_are_also_excluded(tool):
    assert RE.triage(_traj(tools=(tool,))).reason == RE.REJECT_TOOL


def test_the_allow_list_and_the_containment_list_are_disjoint():
    """A tool that is replay-SAFE but containment-FORBIDDEN would be a
    contradiction: the triage would admit an episode the runner then
    refuses to let act."""
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    assert not (RE.REPLAYABLE_TOOLS & REPLAY_FORBIDDEN_TOOLS)


# ------------------------------------------------------------------ #
# Each filter, and the reason it gives                               #
# ------------------------------------------------------------------ #

def test_an_undecided_episode_is_rejected():
    """D1's validator self-test compares against the RECORDED outcome.
    An episode without one has nothing to self-test against, so admitting
    it means trusting a synthesised validator on its synthesis alone —
    the one thing D1 exists not to do."""
    t = RE.triage(_traj(outcome="unknown"))
    assert t.replayable is False and t.reason == RE.REJECT_UNDECIDED


def test_a_single_step_episode_is_rejected():
    assert RE.triage(_traj(steps=1)).reason == RE.REJECT_THIN
    assert RE.triage(_traj(steps=5, tools=())).reason == RE.REJECT_THIN


def test_a_self_play_trajectory_is_not_a_source():
    assert RE.triage(_traj(kind="self_play")).reason == RE.REJECT_KIND
    assert RE.triage(_traj(kind="reflection")).reason == RE.REJECT_KIND
    assert RE.triage(_traj(kind="bench")).replayable is True


def test_an_episode_with_no_request_text_is_rejected():
    assert RE.triage(_traj(request="  ")).reason == RE.REJECT_NO_REQUEST


def test_the_destructive_denylist_gates_the_request():
    t = RE.triage(_traj(request="drop table users and rebuild it"))
    assert t.replayable is False and t.reason == RE.REJECT_UNSAFE


def test_the_destructive_denylist_also_reads_the_commands():
    """The denylist matches TEXT, and the destructive thing an episode
    did lives in its commands — not necessarily in the request that
    prompted them. A benign-sounding ask that ran `rm -rf` is exactly the
    episode that must not be replayed."""
    traj = _traj(request="clean up the build")
    traj.tool_calls[1].arguments = {"command": "rm -rf /workspace/out"}
    assert RE.triage(traj).reason == RE.REJECT_UNSAFE


def test_triage_fails_closed_on_a_broken_record():
    class _Exploding:
        @property
        def task_kind(self):
            raise RuntimeError("corrupt")

    t = RE.triage(_Exploding())
    assert t.replayable is False


# ------------------------------------------------------------------ #
# Specs are only generated when their premise holds                  #
# ------------------------------------------------------------------ #

def test_a_withhold_spec_needs_a_lesson_that_was_really_hydrated():
    """"What if lesson X had been absent" is meaningless unless X was
    present. A spec whose premise is false produces a verdict about
    nothing, and a corpus of those is how a label source becomes a noise
    source."""
    bare = RE.build_specs(_traj())
    assert not [s for s in bare
                if s.perturbation == RE.PERTURB_LESSON_WITHHOLD]
    with_lessons = RE.build_specs(
        _traj(extra={"hydrated_lessons": ["parse before you index", "b"]}))
    holds = [s for s in with_lessons
             if s.perturbation == RE.PERTURB_LESSON_WITHHOLD]
    assert [s.target for s in holds] == ["parse before you index", "b"]


def test_a_deny_spec_needs_a_step_that_really_failed():
    """Denying a step that worked asks nothing. On a clean episode there
    is no swap spec, and that is correct — not a gap."""
    clean = RE.build_specs(_traj())
    assert not [s for s in clean if s.perturbation == RE.PERTURB_STEP_DENY]
    traj = _traj(tools=("file_system", "execute", "file_system"),
                 results=["ok", "Error: no such file", "ok"])
    denies = [s for s in RE.build_specs(traj)
              if s.perturbation == RE.PERTURB_STEP_DENY]
    assert len(denies) == 1 and denies[0].fork_step == 1
    assert denies[0].target == "execute"


def test_the_failing_step_uses_the_shared_corpus_rule():
    """Not a second definition of failure — `outcome_heuristics` is THE
    corpus rule, and a private one would drift from every other reader."""
    import inspect
    src = inspect.getsource(RE._first_failing_step)
    assert "tool_call_failed" in src
    # …and it actually fires on the shape that rule recognises.
    traj = _traj(tools=("execute",), steps=2,
                 results=["EXIT CODE: 1\nboom"])
    assert RE._first_failing_step(traj) == 0


def test_a_verify_toggle_spec_is_always_generated():
    specs = RE.build_specs(_traj())
    assert [s.perturbation for s in specs] == [RE.PERTURB_VERIFY_TOGGLE]


def test_no_specs_for_a_non_replayable_episode():
    assert RE.build_specs(_traj(outcome="unknown")) == []


def test_specs_are_capped_per_episode():
    many = _traj(extra={"hydrated_lessons": [f"L{i}" for i in range(20)]})
    assert len(RE.build_specs(many, max_per_episode=3)) <= 5


# ------------------------------------------------------------------ #
# spec_id is the identity of the QUESTION                            #
# ------------------------------------------------------------------ #

def test_the_same_question_gets_the_same_id():
    """A re-run has to be recognisable as a re-run: D4's stability check
    compares two runs of the SAME spec, and a per-row id would make every
    re-run look like new evidence."""
    a = RE.ReplaySpec(trajectory_id="t1", perturbation="lesson_withhold",
                      fork_step=0, target="x")
    b = RE.ReplaySpec(trajectory_id="t1", perturbation="lesson_withhold",
                      fork_step=0, target="x", seed=999,
                      user_request="different framing")
    assert a.spec_id == b.spec_id


def test_different_questions_get_different_ids():
    base = dict(trajectory_id="t1", perturbation="lesson_withhold",
                fork_step=0, target="x")
    ids = {
        RE.ReplaySpec(**base).spec_id,
        RE.ReplaySpec(**{**base, "target": "y"}).spec_id,
        RE.ReplaySpec(**{**base, "fork_step": 2}).spec_id,
        RE.ReplaySpec(**{**base, "perturbation": "verify_toggle"}).spec_id,
        RE.ReplaySpec(**{**base, "trajectory_id": "t2"}).spec_id,
    }
    assert len(ids) == 5


def test_seeds_are_deterministic_per_episode_and_perturbation():
    a = RE.build_specs(_traj(extra={"hydrated_lessons": ["x"]}))
    b = RE.build_specs(_traj(extra={"hydrated_lessons": ["x"]}))
    assert [s.seed for s in a] == [s.seed for s in b]
    assert len({s.seed for s in a}) == len(a)


# ------------------------------------------------------------------ #
# Persistence                                                        #
# ------------------------------------------------------------------ #

def test_specs_round_trip_through_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    specs = RE.build_specs(_traj(extra={"hydrated_lessons": ["x"]}))
    assert RE.write_specs(specs) == len(specs)
    rows = list(RE.iter_specs())
    assert len(rows) == len(specs)
    assert {r["spec_id"] for r in rows} == {s.spec_id for s in specs}


def test_a_planned_spec_is_not_an_answered_one(tmp_path, monkeypatch):
    """`known_spec_ids` reads the CREDITS ledger, not the spec plan. A
    spec that was planned and then skipped has not been answered, and
    treating it as answered burns it permanently — on a box where docker
    is down, every batch would consume specs and record nothing until the
    engine had quietly exhausted its own corpus."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    specs = RE.build_specs(_traj(extra={"hydrated_lessons": ["x"]}))
    RE.write_specs(specs)
    assert RE.known_spec_ids() == set(), \
        "a planned-but-unanswered spec was burned"
    RE.write_credits([{"spec_id": specs[0].spec_id,
                       "verdict": RE.VERDICT_ABSTAIN, "ts": "t"}])
    assert RE.known_spec_ids() == {specs[0].spec_id}


def test_writing_without_a_home_is_silent_not_fatal(monkeypatch):
    """An ad-hoc import (a script, a test) must not create files."""
    monkeypatch.setenv("GHOST_HOME", "")
    assert RE.write_specs(RE.build_specs(_traj())) == 0
    assert list(RE.iter_specs()) == []


def test_the_spec_ledger_rotates(tmp_path, monkeypatch):
    """A durable store is never truncated in place."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    d = tmp_path / "system" / "dream_replay"
    row = RE.ReplaySpec(trajectory_id="t0", perturbation="verify_toggle")
    RE.write_specs([row])
    row_bytes = (d / "specs.jsonl").stat().st_size
    # Size the cap so a generation holds ~3 rows: one generation is kept,
    # so 6 writes must leave 2 generations that BOTH still read.
    monkeypatch.setattr(RE, "_LEDGER_MAX_BYTES", row_bytes * 3)
    for i in range(1, 6):
        RE.write_specs([RE.ReplaySpec(trajectory_id=f"t{i}",
                                      perturbation="verify_toggle")])
    assert (d / "specs.jsonl.1").exists(), "the ledger never rotated"
    ids = [r["trajectory_id"] for r in RE.iter_specs()]
    assert len(ids) > 3, f"rotation lost a whole generation: {ids}"
    # oldest first — a time-ordered read is what makes the census honest
    assert ids == sorted(ids)


# ------------------------------------------------------------------ #
# The census — a count alone cannot be acted on                      #
# ------------------------------------------------------------------ #

def test_the_census_reports_the_funnel_not_just_a_count(monkeypatch):
    """"the corpus is thin" and "the filter is too tight" produce the
    same admitted count and need opposite responses."""
    trajs = [
        _traj(id="a"),                                    # admitted
        _traj(id="b", outcome="unknown"),                 # undecided
        _traj(id="c", tools=("web_search",)),             # tool
        _traj(id="d", steps=1),                           # thin
        _traj(id="e", kind="self_play"),                  # kind
    ]
    src = RE.EpisodeSource(include_bench=False)
    monkeypatch.setattr(src, "_iter_real", lambda: iter(trajs))
    census = src.census()
    assert census["seen"] == 5 and census["replayable"] == 1
    assert census["rejected"] == {
        RE.REJECT_UNDECIDED: 1, RE.REJECT_TOOL: 1,
        RE.REJECT_THIN: 1, RE.REJECT_KIND: 1,
    }
    assert census["by_outcome"] == {"passed": 1, "failed": 0}


def test_the_episode_limit_is_honoured(monkeypatch):
    trajs = [_traj(id=f"t{i}") for i in range(10)]
    src = RE.EpisodeSource(include_bench=False)
    monkeypatch.setattr(src, "_iter_real", lambda: iter(trajs))
    assert len(list(src.iter_episodes(limit=3))) == 3


# ------------------------------------------------------------------ #
# Admissibility                                                      #
# ------------------------------------------------------------------ #

def test_the_read_row_admits_bench_and_the_credit_row_does_not():
    """Two rows on purpose: one could not express "may read the corpus,
    may not yet believe the output", and a single permissive row is how
    an unvalidated label source opens."""
    from ghost_agent.core import admissibility as AD
    assert AD.admits_bench(RE.CONSUMER_READ) is True
    assert AD.admits_bench(RE.CONSUMER_CREDIT) is False
    assert AD.policy_for(RE.CONSUMER_CREDIT) == AD.POLICY_REAL_ONLY
    assert set(AD.admitted_task_kinds(RE.CONSUMER_READ)) == {
        "user_request", "bench"}


def test_an_unregistered_reader_gets_nothing():
    from ghost_agent.core import admissibility as AD
    assert list(AD.iter_bench_trajectories("dream_something_new")) == []


# ================================================================== #
# The two defects the first A/A census surfaced                      #
# ================================================================== #

class _TC:
    def __init__(self, name, result="ok", **args):
        self.name = name
        self.arguments = args
        self.result = result


class _Traj:
    def __init__(self, calls, request="do it", outcome="passed",
                 kind="user_request"):
        self.id = "t1"
        self.tool_calls = calls
        self.user_request = request
        self.outcome = outcome
        self.task_kind = kind
        self.n_steps = max(2, len(calls))
        self.final_response = "done"


# ---- fix 1: the trace must not carry paths the fork strips --------- #

@pytest.mark.parametrize("raw,want", [
    ("projects/7b62e5e533d1/cross_session_results.json",
     "cross_session_results.json"),
    ("python3 projects/9009ff89d8bb/run.py --flag", "python3 run.py --flag"),
    ("README.md", "README.md"),
    ("my_projects/x.py", "my_projects/x.py"),      # not a false match
    # ⚠ THE LENGTH FLOOR IS LOAD-BEARING. A real project id is 12 hex
    # (`7b62e5e533d1`); a short `projects/ab/` is far likelier to be a
    # user's own directory called "projects", and stripping it would
    # rewrite a path the replay does NOT relocate. Conservative: leave it.
    ("projects/ab/x.py", "projects/ab/x.py"),
    ("projects//x.py", "projects//x.py"),
    ("", ""),
    # ⚠ EVERY ONE OF THESE IS A FORM THE FIRST VERSION LEAKED, found by
    # sweeping the trace of all 42 replayable episodes rather than by
    # imagining what a path looks like. The mount point hides the `cd`
    # hop from a rule that does not strip roots first:
    ("cd /workspace/projects/6a7c76ab5a80 && python3 demo.py",
     "python3 demo.py"),
    ("/workspace/projects/7b62e5e533d1/y.json", "y.json"),
    # …a HOST path, which reaches the trace through a `list_files`:
    ("/Users/v/Data/AI/Data/sandbox/projects/10c7855c5a13", "."),
    # …a project or mount root with nothing under it names the fork's
    # WORKING DIRECTORY. It becomes `.`, not nothing: an empty `path=`
    # would tell the synthesiser less than the truth does.
    ("find /workspace -type f | wc -l", "find . -type f | wc -l"),
    ("path=/workspace", "path=."),
    # …and the result channel, which the arguments fix did not cover:
    ("SUCCESS: Wrote 5202 chars to 'projects/7b62e5e533d1/x.py'",
     "SUCCESS: Wrote 5202 chars to 'x.py'"),
    # Boundaries. Without the left lookbehind `my/workspace` became
    # `my.` — a rewrite of a path the replay does NOT relocate.
    # ⚠ THE OLD `my_projects/x.py` CASE WAS VACUOUS — it passed because
    # `x.py` has a dot, not because of any boundary, so adding or
    # removing a left lookbehind on the project rules changed nothing.
    # These three DO discriminate: without the boundary they became
    # `my_x.py`, `my.` and `sub_x`.
    ("my_projects/abcd/x.py", "my_projects/abcd/x.py"),
    ("myprojects/abcdef", "myprojects/abcdef"),
    ("sub_projects/data1/x", "sub_projects/data1/x"),
    # …and the sandbox rule had NO negative control at all. Without a
    # left boundary this became `notes.md`.
    ("docs/sandbox/notes.md", "docs/sandbox/notes.md"),
    ("my/workspace", "my/workspace"),
    ("my/workspace/x.py", "my/workspace/x.py"),
    ("/workspaces/foo", "/workspaces/foo"),
    ("a_workspace/x", "a_workspace/x"),
    ("cd /tmp && ls", "cd /tmp && ls"),
])
def test_recorded_paths_are_rewritten_to_where_the_REPLAY_puts_them(raw,
                                                                    want):
    """⚠ MEASURED DEFECT. `isolated_replay_context` nulls
    `current_project_id` on purpose, so the replay writes at the sandbox
    root — but the trace fed to the validator synthesiser still carried
    the RECORDED path, and the model wrote checks against
    `projects/<id>/…` that CANNOT pass. Verified by synthesising the real
    validators for two such episodes: both named the project path. 4 of
    12 non-reproducing episodes in the first A/A census were this."""
    assert RE._fork_relative(raw) == want


def test_roots_come_off_BEFORE_the_cd_hop():
    """⚠ THE FIRST FIX WAS COMPLETE AND STILL LEAKED, because the rules
    ran in the wrong order. `_PROJECT_CD_RE` cannot see `cd projects/…`
    while the value still reads `cd /workspace/projects/…`, so stripping
    the mount point AFTER the cd rule leaves the whole hop behind. This
    pins the order by asserting the one case that distinguishes it."""
    assert RE._fork_relative(
        "cd /workspace/projects/6a7c76ab5a80 && python3 demo.py"
    ) == "python3 demo.py"
    # …and the same value with the roots already gone, which BOTH
    # orderings handle — proving the case above is the discriminating one.
    assert RE._fork_relative(
        "cd projects/6a7c76ab5a80 && python3 demo.py") == "python3 demo.py"


def test_the_RESULT_channel_is_rewritten_too():
    """⚠ MEASURED. After the arguments were fixed, the trace still said
    `SUCCESS: Wrote 5202 chars to 'projects/<id>/x.py'` — the tool's own
    result echoed the path the arguments no longer did."""
    traj = _Traj([_TC("file_system", operation="write", path="x.py",
                      result="SUCCESS: Wrote 5202 chars to "
                             "'projects/7b62e5e533d1/x.py'")])
    trace = RE._trace_for_prompt(traj)
    assert "projects/7b62e5e533d1" not in trace
    assert "x.py" in trace


def test_the_trace_ACTUALLY_applies_the_rewrite():
    """The helper existing and the prompt using it are two facts."""
    traj = _Traj([_TC("file_system", operation="write",
                      path="projects/7b62e5e533d1/out.json",
                      content="see projects/7b62e5e533d1/out.json"),
                  _TC("execute", command="python3 projects/abc123de/run.py")])
    trace = RE._trace_for_prompt(traj)
    assert "path=out.json" in trace
    assert "command=python3 run.py" in trace
    # Content is normalised like every other value: a project path
    # inside a file's text is a location reference too, and a check
    # built from it would assert a path the fork does not have.
    assert "content=see out.json" in trace
    assert "projects/7b62e5e533d1" not in trace


def test_the_prompt_tells_the_model_not_to_re_add_a_prefix():
    assert "do not prepend" in RE._VALIDATOR_PROMPT
    assert "projects/<id>/" in RE._VALIDATOR_PROMPT


# ---- fix 2: a turn with no artifact is not filesystem-checkable ---- #

@pytest.mark.parametrize("calls,produces", [
    ([_TC("file_system", operation="write", path="a.py")], True),
    ([_TC("file_system", operation="replace", path="a.py")], True),
    ([_TC("execute", command="python3 -c \"open('x','w')\"")], True),
    ([_TC("report_pdf", filename="r.pdf")], True),
    ([_TC("file_system", operation="read", path="a.py")], False),
    ([_TC("file_system", operation="list_files", path=".")], False),
    ([_TC("workspace", action="summary")], False),
    ([_TC("introspect", action="experiments")], False),
    ([_TC("recall", query="x"), _TC("list_lessons")], False),
    ([], False),
])
def test_only_a_turn_that_LEFT_something_is_checkable(calls, produces):
    assert RE._produces_an_artifact(_Traj(calls)) is produces


def test_report_pdf_counts_as_producing():
    """⚠ The measured false reject. The first version of the rule counted
    only `execute` and file-writing `file_system` ops, and rejected
    "write me a pdf report about tasks 15-19" as having no checkable
    deliverable — when writing a PDF into the sandbox is the entire
    tool."""
    assert "report_pdf" in RE._PRODUCING_TOOLS
    assert RE._produces_an_artifact(_Traj([_TC("report_pdf",
                                               filename="r.pdf")])) is True


def test_a_conversational_turn_is_rejected_by_TRIAGE_not_by_a_wasted_run():
    """The validator prompt already has an exit-2 escape hatch and it
    fires — but not reliably: 7 episodes used it while 8 more got a
    filesystem check written for a task whose deliverable was a
    SENTENCE. Each of those cost a synthesis plus six full agent runs to
    discover."""
    traj = _Traj([_TC("workspace", action="summary"),
                  _TC("file_system", operation="list_files", path=".")],
                 request="Summarise the latest files in my sandbox.")
    tri = RE.triage(traj)
    assert tri.replayable is False
    assert tri.reason == RE.REJECT_NO_ARTIFACT


def test_a_turn_that_wrote_a_file_is_still_replayable():
    traj = _Traj([_TC("file_system", operation="write", path="out.py",
                      content="x = 1"),
                  _TC("execute", command="python3 out.py")])
    assert RE.triage(traj).replayable is True


def test_the_artifact_rule_runs_LAST_so_the_census_stays_readable():
    """It may only ever claim episodes that would otherwise have been
    admitted — otherwise the rejection histogram stops meaning what it
    says."""
    forbidden = _Traj([_TC("browser", action="goto"),
                       _TC("workspace", action="summary")])
    assert RE.triage(forbidden).reason == RE.REJECT_TOOL
    thin = _Traj([], request="hi")
    thin.n_steps = 0
    assert RE.triage(thin).reason == RE.REJECT_THIN
    undecided = _Traj([_TC("workspace", action="summary")],
                      outcome="unknown")
    assert RE.triage(undecided).reason == RE.REJECT_UNDECIDED


@pytest.mark.asyncio
async def test_ALL_THREE_prompt_slots_are_fork_relative():
    """⚠ THE FIRST FIX GUARDED ONE SLOT OF THREE. `_VALIDATOR_PROMPT` has
    `{request}`, `{trace}` and `{final}`; only the trace was rewritten.
    MEASURED across the 42 replayable episodes: trace leaked a project
    path in 0, `final_response` in 3, `user_request` in 1.

    The final reply is the worst channel — it is where the agent
    ANNOUNCES what it wrote, naming the recorded location in prose the
    synthesiser copies verbatim. Two of those files still exist in the
    live sandbox, which a seeded fork COPIES, so a check built on one
    passes on every leg with no agent run at all: a free `p̂ = 1.0` that
    measured nothing."""
    seen = {}

    class _LLM:
        async def chat_completion(self, payload, **kw):
            seen["prompt"] = payload["messages"][0]["content"]
            return {"choices": [{"message": {"content": ""}}]}

    traj = _Traj([_TC("file_system", operation="write", path="out.json")])
    traj.user_request = "look in /Users/v/Data/AI/Data/sandbox/projects/f36f04d446a6/"
    traj.final_response = ("The file `projects/7b62e5e533d1/message.md` "
                           "has been written")
    await RE.synthesize_validator(traj, _LLM())
    prompt = seen["prompt"]
    assert "projects/7b62e5e533d1" not in prompt, "the FINAL slot leaked"
    assert "projects/f36f04d446a6" not in prompt, "the REQUEST slot leaked"
    # …and the surrounding prose survives, so this is a rewrite and not
    # a blanket deletion of the slot.
    assert "has been written" in prompt
    assert "look in" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("returned,expect", [
    ("", "screen:empty"),
    ("import os\nos.system('rm -rf /')", "screen:uses os.system"),
    ("def f(:\n  pass", "screen:syntax error: invalid syntax"),
    ("x" * 9000, "screen:over"),
])
async def test_the_synthesis_REJECTION_REASON_is_reported(returned, expect):
    """⚠ THE FUNNEL'S BIGGEST LOSS WAS UNEXPLAINED. The A/A census
    measured 46 of 128 episodes dying at `no_admissible_validator` —
    more than every other rejection combined — and the reason went to a
    log line and was thrown away. "The model returned nothing" and "the
    screen rejected what it returned" are opposite problems with
    opposite fixes, and the only number anyone reads could not tell them
    apart."""
    class _LLM:
        async def chat_completion(self, payload, **kw):
            return {"choices": [{"message": {"content": returned}}]}

    out = {}
    v = await RE.synthesize_validator(_Traj([_TC("file_system",
                                                 operation="write",
                                                 path="x.py")]),
                                      _LLM(), out=out)
    assert v == ""
    assert out["reason"].startswith(expect), out["reason"]


@pytest.mark.asyncio
async def test_a_SUCCESSFUL_synthesis_reports_no_reason():
    """The other half of the identity — a reason that is always set is
    not evidence of anything."""
    class _LLM:
        async def chat_completion(self, payload, **kw):
            return {"choices": [{"message":
                                 {"content": "import sys\nsys.exit(1)"}}]}

    out = {}
    v = await RE.synthesize_validator(_Traj([_TC("file_system",
                                                 operation="write",
                                                 path="x.py")]),
                                      _LLM(), out=out)
    assert v == "import sys\nsys.exit(1)"
    assert out["reason"] == ""
    assert out["raw_chars"] > 0


@pytest.mark.asyncio
async def test_callers_that_do_not_ASK_for_a_reason_still_work():
    """`out` is optional; the engine's own call sites pass nothing."""
    class _LLM:
        async def chat_completion(self, payload, **kw):
            return {"choices": [{"message": {"content": ""}}]}
    assert await RE.synthesize_validator(_Traj([]), _LLM()) == ""
