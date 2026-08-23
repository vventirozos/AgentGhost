"""§4AT round-3 fixes across five subsystems (2026-08-11).

Each pin names the live defect it closes. All were found by the audit's
pre-registered questions and verified against the real code before fixing.
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import dream as DREAM              # noqa: E402
from ghost_agent.core import project_advancer as PA      # noqa: E402
from ghost_agent.core.coding_executor import (           # noqa: E402
    _replace_failure_kind,
)
from ghost_agent.reflection import postmortem as PM      # noqa: E402
from ghost_agent.tools import registry as REG            # noqa: E402


# ── F: the retry-steering mislabel (req 6e9efd6a's 20-minute grind) ──────

def test_a_syntax_rollback_is_not_reported_as_a_missing_anchor():
    """THE LIVE DEFECT. Every non-SUCCESS replace was labelled "anchor not
    found" and fed to the next attempt behind "YOUR PREVIOUS ATTEMPT FAILED:".
    A syntax-regression rollback therefore sent the model hunting for a better
    ANCHOR while the defect was its replacement's INDENTATION — 14 attempts,
    20 minutes, 5 of them rejected by the same guard."""
    out = ("REJECTED: that replace would introduce a syntax error and was NOT "
           "applied — 'x.py' is unchanged on disk: unexpected indent (line 238).")
    kind = _replace_failure_kind(out)
    assert "syntax" in kind.lower()
    assert "anchor not found" not in kind.lower()
    assert "replacement" in kind.lower(), "must point at the REPLACE block"


def test_each_distinct_rejection_gets_its_own_label():
    assert "ambiguous" in _replace_failure_kind(
        "SYSTEM INSTRUCTION: Multiple instances of this text block found").lower()
    # ⚠ THE REAL STRING, copied from file_system.py:1671. The first version
    # of this test invented "None of the blocks matched the file contents",
    # which appears nowhere in the tree — so it validated the classifier
    # against a string production never emits, and the genuine multi-block
    # failure fell through to the neutral branch unnoticed.
    assert _replace_failure_kind(
        "SYSTEM INSTRUCTION: None of the SEARCH/REPLACE blocks matched in "
        "'x.py'.\nCould not find block:\ndef foo()..."
    ) == "anchor not found"
    # …and a MISSING FILE is not a missing anchor (file_system.py:1362)
    assert "file" in _replace_failure_kind("Error: 'utils.py' not found.").lower()
    assert "anchor" not in _replace_failure_kind("Error: 'utils.py' not found.").lower()
    assert "marker" in _replace_failure_kind(
        "REJECTED: replacement contains SEARCH/REPLACE markers (====)").lower()
    assert "block" in _replace_failure_kind(
        "SYSTEM BLOCK: released project is immutable; this write was NOT applied.").lower()


def test_an_UNRECOGNISED_rejection_stays_neutral():
    """⚠ The whole defect was a confident wrong guess. An unknown shape must
    NOT fall back to the most common label — the caller appends the tool's raw
    text anyway, so saying less is strictly better than saying something
    false."""
    kind = _replace_failure_kind("some novel failure nobody has seen")
    assert "anchor not found" not in kind.lower()
    assert "syntax" not in kind.lower()


def test_the_call_site_actually_uses_the_classifier():
    """⚠ SEAM. The helper can be perfect while the caller still hardcodes the
    old string."""
    src = inspect.getsource(
        sys.modules["ghost_agent.core.coding_executor"])
    assert "_replace_failure_kind(out)" in src
    assert "did not apply (anchor not found)" not in src


# ── G: simulation must not author real selfhood ─────────────────────────

def _self_play_denylist():
    """The set self-play denies its temp agent.

    ⚠ Read from the MODULE CONSTANT now, not from an AST walk of an
    inline literal. The literal was extracted to
    `dream.SELF_PLAY_FORBIDDEN_TOOLS` on 2026-08-22 (§4CM) because
    `core/isolation.REPLAY_FORBIDDEN_TOOLS` claims to be a superset of
    it — and while it was a literal, that claim was both FALSE (missing
    `web_search`/`deep_research`, i.e. real host-process egress) and
    unfalsifiable, since there was nothing to import.

    This reads what the code actually uses; dream.py asserts the literal
    it documents and the constant it applies have not drifted.
    """
    from ghost_agent.core.dream import SELF_PLAY_FORBIDDEN_TOOLS
    return set(SELF_PLAY_FORBIDDEN_TOOLS)

def test_self_state_is_denied_to_self_play():
    """`isolated_context.self_model` is the PRODUCTION SelfModel, so a
    synthetic run calling `self_state(action='note_principle')` wrote the
    agent's real stated operating principles. The list's own stated rule is
    "any tool that writes to real, non-isolated state must be disabled here";
    `subagent.py` already denies selfhood authoring. Self-play is 165 of the
    last 249 requests on this box."""
    denied = _self_play_denylist()
    assert "self_state" in denied, "self_state is not denied to self-play"

    # ⚠ FIXING ONE INSTANCE LEFT FIVE (review round 4). `isolated_context`
    # nulls/wraps ten memory surfaces but never `project_store`, so these
    # bound the PRODUCTION store and stayed callable from a synthetic run.
    # `core/subagent.py` FORBIDDEN_TOOLS already named every one of them —
    # self-play had drifted from its sibling.
    from ghost_agent.core.subagent import FORBIDDEN_TOOLS
    for t in ("manage_projects", "manage_services", "delegate",
              "self_play_loop", "stop_self_play"):
        assert t in denied, f"{t} still callable from self-play"
    missing = {t for t in FORBIDDEN_TOOLS if t not in denied}
    assert not missing, (
        f"self-play allows tools its sibling subagent forbids: {sorted(missing)}")

    # …and the constant is actually the set the code applies. Extracting a
    # literal into a constant is only an improvement if the literal stops
    # being the thing that runs.
    src = inspect.getsource(DREAM.Dreamer.synthetic_self_play)
    assert "disabled_tools.update(SELF_PLAY_FORBIDDEN_TOOLS)" in src


# ── C: a string constraint must not shred into characters ───────────────

def test_project_advancer_uses_the_SHARED_constraint_normaliser():
    """Metadata is model-written JSON, so `{"constraints": "no external
    APIs"}` is legal — and iterating it raw yields 17 single-character
    "constraints" fed to the constraint gate. `_constraint_list` exists
    because this shredding already destroyed a record once (2026-08-01); this
    call site kept its own copy and so kept the bug."""
    src = inspect.getsource(PA)
    assert "_constraint_list(" in src, "shared normaliser not used"
    assert "[str(c) for c in\n                        ((proj.get" not in src

    from ghost_agent.memory.projects import _constraint_list
    assert _constraint_list("no external APIs") == ["no external APIs"]
    assert len(_constraint_list("no external APIs")) == 1, "shredded again"
    assert _constraint_list(["a", "b"]) == ["a", "b"]
    assert _constraint_list(None) == []


# ── H: postmortem's two re-enable landmines ─────────────────────────────

def test_postmortem_materialises_the_corpus_OFF_the_event_loop():
    """Measured 0.207 s over 1,573 trajectories / 22 MB and growing. The
    identical pattern ~100 lines away in agent.py was moved to a thread under
    §4Q; this site never got the fix because the gate has been shut for
    months, so nobody felt the stall."""
    src = inspect.getsource(PM.PostMortemEngine.run)
    assert "to_thread" in src, "corpus load still blocks the event loop"
    assert "trajectories = list(source())" not in src


def _failed_traj(ts, tid="t"):
    """A FAILED trajectory with a repeated tool error — severe enough to be
    selected, so selection turns purely on the date."""
    from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
    calls = [ToolCall(name="file_system", arguments={"path": "a.py"},
                      error="boom: could not read") for _ in range(6)]
    t = Trajectory(id=tid, task_kind="user_request",
                   outcome=Outcome.FAILED.value)
    t.tool_calls = calls
    if ts is not None:
        t.timestamp = ts
    return t


def test_an_OLD_failed_run_is_actually_dropped():
    """⚠ BEHAVIOURAL, not a substring match. The first version of this pin
    asserted `"max_age_days" in signature` and `"if _ts and _ts < _cutoff" in
    source` — both of which pass if the body is `... : pass`. Review was right
    that they could not fail if the feature broke. Every fixture in the
    existing postmortem suite defaults its timestamp to NOW, so the bound was
    a no-op across the whole suite."""
    import datetime as dt
    old = (dt.datetime.utcnow() - dt.timedelta(days=30)).isoformat() + "Z"
    new = dt.datetime.utcnow().isoformat() + "Z"
    picked = PM.select_failed_runs([_failed_traj(old, "old")], limit=5,
                                   max_age_days=7.0)
    assert picked == [], "a 30-day-old failed run was still selected"
    picked = PM.select_failed_runs([_failed_traj(new, "new")], limit=5,
                                   max_age_days=7.0)
    assert [t.id for t, _ in picked] == ["new"], "a fresh run must survive"


def test_an_UNDATED_run_is_KEPT_not_dropped():
    """⚠ A filter that fails CLOSED on missing data hides work rather than
    bounding it. Exercised, not grepped."""
    picked = PM.select_failed_runs([_failed_traj("", "undated")], limit=5,
                                   max_age_days=7.0)
    assert [t.id for t, _ in picked] == ["undated"], (
        "an undated run was dropped — unknown date must not shrink the pool")


def test_the_bound_can_be_DISABLED_and_then_old_runs_return():
    """Proves the filter is what excluded them, not severity or dedup."""
    import datetime as dt
    old = (dt.datetime.utcnow() - dt.timedelta(days=30)).isoformat() + "Z"
    picked = PM.select_failed_runs([_failed_traj(old, "old")], limit=5,
                                   max_age_days=0)
    assert [t.id for t, _ in picked] == ["old"]


# ── E: the live A/B must differ by exactly the treatment ────────────────

def test_fs_batch_arm_does_not_silently_relax_path_for_MUTATING_ops():
    """Dropping `path` from `required` also told the treatment arm that
    write/replace/delete need no path — unrelated to the batch macro under
    test, i.e. an unmeasured confound in a live experiment. JSON Schema cannot
    express "required unless operation == read" portably, so the constraint is
    stated in the description, which is the channel the model reads."""
    suffix = REG._FS_BATCH_DESC_SUFFIX.lower()
    assert "still required" in suffix or "still REQUIRED".lower() in suffix
    assert "batch" in suffix and "read" in suffix


# ── B: self-play attribution, re-landed after the data repair ───────────

def test_the_TEMPLATE_outranks_a_rotated_away_seed():
    """⚠ BOTH HALVES OF §4AT-B. (1) A deterministic template knows what it is;
    re-deriving the cluster by keyword-matching its RENDERED PROMPT misfiled
    51% of 80 real renders, with `python_general`/`regex_parse` swapped.
    (2) `pick_seed` can pick a cluster, find it saturated and rotate to a fresh
    template — `_cluster_key` is cleared on that path but the seed dict is
    never updated, so consulting the seed first returns the ABANDONED target
    while the template that actually ran sits right there.

    Landing this needed the live `bash.mastered` flag cleared first: correct
    attribution otherwise routed bash renders into a mastered cluster and
    silenced lesson extraction for ~12% of self-play cycles."""
    assert DREAM.resolve_cluster_key(
        {"cluster_key": "python_general"}, "web_automation", "x") == "web_automation"


def test_the_seed_still_wins_when_no_template_ran():
    assert DREAM.resolve_cluster_key({"cluster_key": "sql"}, "", "x") == "sql"


def test_the_text_classifier_is_still_the_last_resort():
    """LLM-generated and journal-mined challenges carry no template identity."""
    from ghost_agent.memory.frontier import classify_cluster
    got = DREAM.resolve_cluster_key({}, "", "write a SQL query grouping sales")
    assert got == classify_cluster("write a SQL query grouping sales") == "sql"


def test_the_journal_boilerplate_no_longer_FORCES_sql():
    """The wrapper prose contained the literal "SQL", and `classify_cluster`
    tests `\\bsql\\b` first — so the boilerplate, not the task, decided the
    cluster for every mined challenge (23 of 23 stored runs, including CSV,
    log and shell tasks). `sql` reached expert tier on work that was never SQL.

    ⚠ PARTIAL BY DESIGN, and stated so: the wrapper still contributes when the
    task itself carries no keyword. The complete fix is to classify the
    ORIGINAL request rather than the wrapped prompt."""
    from ghost_agent.memory.frontier import classify_cluster
    from ghost_agent.core import journal_challenges as JC
    # ⚠ Behavioural, not a source grep: my first version asserted the literal
    # was absent from the SOURCE and failed on my own explanatory comment —
    # a pin that fires on prose rather than on the property it guards.
    import ast as _ast
    prompt_literals = [n.value for n in _ast.walk(_ast.parse(inspect.getsource(JC)))
                       if isinstance(n, _ast.Constant) and isinstance(n.value, str)]
    assert not any("SQL/text" in v for v in prompt_literals), (
        "the sql-forcing token is back in a prompt string")
    boiler = ("A deterministic `input.csv` fixture has been written in your "
              "working directory — its shape matches what the original user "
              "task referenced (tabular, structured, log-style or plain text).")
    assert classify_cluster(boiler) != "sql"
    # a task with its own keyword now wins over the wrapper
    assert classify_cluster(boiler + " write a shell pipeline with awk") == "bash"


# ── F: a dropped edit must not read as "nothing to do" ──────────────────

@pytest.mark.asyncio
async def test_edits_for_an_UNSNAPSHOTTED_file_return_a_REASON_not_silence():
    """⚠ THE CRITICAL. `edits` are applied only when `path in snap`, and the
    prompt snapshot is hard-capped at 12 files. On a larger project an `edits`
    entry for an unsnapshotted file fell through to the `content` branch,
    `content` is None, and the function returned (None, None) — "skipped, not
    a failure": zero tool calls, no reason, nothing in `written`. A sibling
    entry that DID apply then made the whole spec look successful, so the leaf
    closed **DONE claiming work that never happened**.
    """
    from ghost_agent.core import coding_executor as CE

    calls = []

    async def _runner(tool, args):
        calls.append((tool, args))
        return "SUCCESS"

    written, reason = await CE._apply_file(
        _runner,
        {"path": "utils.js", "edits": [{"search": "a", "replace": "b"}]},
        {"index.html": "<html>"},               # utils.js NOT snapshotted
        touched=set(), fresh=set())
    assert written is None
    assert reason, "a dropped edit returned no reason — silently 'nothing to do'"
    assert "utils.js" in reason and "snapshot" in reason.lower()
    assert calls == [], "no tool call should have been attempted"


@pytest.mark.asyncio
async def test_a_rewrite_is_never_judged_against_a_TRUNCATED_snapshot():
    """⚠ THE SECOND CRITICAL. `_gather_project_files` keeps only a prefix once
    its 400 KB budget is nearly spent, and that truncation was SILENT — so a
    300 KB index.html snapshotted as 4 KB made a 20 KB full rewrite look like
    GROWTH (20000 > 4000×0.85). The non-regression guard waved through an
    overwrite that destroyed 280 KB of working code. For `.py` the prefix
    usually fails `ast.parse`, disabling the overwrite guard outright.

    A marked entry now means "unknown baseline" → re-read from disk. If the
    re-read fails, REFUSE: a refusal costs one retry, a wrong pass costs the
    file."""
    from ghost_agent.core import coding_executor as CE
    from ghost_agent.core.project_advancer import SNAPSHOT_TRUNCATED_MARK

    big = "x" * 300_000
    truncated_snap = big[:4000] + SNAPSHOT_TRUNCATED_MARK
    reads = []

    async def _runner(tool, args):
        if args.get("operation") in ("read", "read_chunked"):
            reads.append(args.get("path"))
            return "CONTENTS OF index.html:\n" + big
        return "SUCCESS"

    written, reason = await CE._apply_file(
        _runner,
        {"path": "index.html", "content": "<html>small rewrite</html>"},
        {"index.html": truncated_snap},
        touched=set(), fresh=set())
    assert written is None, "a 20KB rewrite of a 300KB file was allowed"
    assert reason, "no reason given for refusing the overwrite"
    assert reads, "the truncated snapshot was trusted instead of re-read"


def test_the_truncation_marker_has_exactly_ONE_definition():
    """Producer and consumer drifting apart is the defect class this file
    already carries scars for."""
    from ghost_agent.core.coding_executor import _snap_truncated_mark
    from ghost_agent.core.project_advancer import SNAPSHOT_TRUNCATED_MARK
    assert _snap_truncated_mark() == SNAPSHOT_TRUNCATED_MARK
    src = inspect.getsource(sys.modules["ghost_agent.core.coding_executor"])
    assert "SNAPSHOT TRUNCATED" not in src, (
        "the marker literal was copied instead of imported")


def test_the_PRODUCER_actually_marks_a_truncated_snapshot_entry():
    """⚠ BOTH SIDES OR NEITHER. Mutation-testing showed the consumer guard was
    pinned but the PRODUCER was not: removing the marker from
    `_gather_project_files` reddened nothing, so the guard would silently
    never fire again. A one-sided pin on a producer/consumer contract is the
    same drift it exists to prevent."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from ghost_agent.core.project_advancer import (
        _gather_project_files, SNAPSHOT_TRUNCATED_MARK)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "projects" / "pid"
        root.mkdir(parents=True)
        # Exceed the char budget so truncation is forced.
        (root / "big.html").write_text("<html>" + "y" * 600_000 + "</html>")
        (root / "small.py").write_text("print(1)\n")
        store = SimpleNamespace(sandbox_root=str(Path(d)))
        snap = _gather_project_files(store, "pid")
        assert snap, "snapshot came back empty — the probe is broken"
        marked = [k for k, v in snap.items() if SNAPSHOT_TRUNCATED_MARK in v]
        assert marked, (
            "a truncated entry carries no marker — the consumer's guard can "
            f"never fire. entries: { {k: len(v) for k, v in snap.items()} }")
        # …and a file that fits is NOT marked (no false positives).
        assert SNAPSHOT_TRUNCATED_MARK not in snap.get("small.py", "")


# ── C: a crash must not downgrade to a weaker closer ────────────────────

def test_a_coding_executor_CRASH_leaves_the_leaf_open():
    """⚠ THE THIRD MEMBER OF THE FAMILY. A transient exception inside
    `coding_executor` left `cres = None` and fell through to the generic
    single-shell-command path, which marks the leaf DONE on ANY output that is
    not `ERROR:`-prefixed and not a non-zero exit — verify gate, smoke gate,
    files-written check and constraint gate all bypassed. `echo`-shaped output
    closed a build task.

    Same shape as the two coding-executor CRITICALs fixed the same day: a task
    reaching DONE on evidence that no work happened. A crash is a reason to
    STOP, not to accept a lower standard of proof."""
    src = inspect.getsource(PA)
    i = src.find('logger.warning("coding_executor crashed')
    assert i > 0, "the crash handler moved — re-anchor this pin"
    after = src[i:i + 700]
    assert "AdvanceResult(" in after and '"blocked"' in after, (
        "a coding-executor crash still falls through to the weaker "
        "single-command path that can close the leaf DONE")
    # and it must NOT simply swallow and continue
    assert "return" in after.split("AdvanceResult(")[0] or "AdvanceResult(" in after


def test_a_frontier_run_records_WHICH_TEMPLATE_produced_it():
    """⚠ AUDITABILITY, not correctness. A run recorded only the cluster it was
    FILED under, never the template that produced it — so when attribution
    turned out to be 51% wrong, reconstructing what belonged where required
    re-rendering the whole template bank at four tiers and matching challenge
    hashes. The record has to say."""
    import tempfile
    from pathlib import Path
    from ghost_agent.memory.frontier import FrontierTracker

    with tempfile.TemporaryDirectory() as d:
        f = FrontierTracker(Path(d))          # takes a memory DIR
        f.record_run("bash", "count lines in logs", 1, True, 40,
                     template_key="bash")
        f.record_run("sql", "an LLM-generated challenge", 1, True, 40)
        import json, glob
        files = glob.glob(str(Path(d) / "*frontier*.json"))
        assert files, f"no frontier file written: {list(Path(d).iterdir())}"
        runs = json.load(open(files[0]))["runs"]
        assert len(runs) == 2
        assert runs[0].get("template_key") == "bash", (
            "the template that produced the run is not recorded")
        # …and an untemplated challenge records EMPTY, which is itself the
        # distinction the cluster field cannot carry.
        assert runs[1].get("template_key") == ""


def test_a_mined_challenge_uses_its_OWN_domain_not_the_wrapper():
    """⚠ THE COMPLETE FIX for journal→sql. `_guess_domains(cleaned)` is
    computed from the ORIGINAL user request, before any wrapper prose exists,
    and was already plumbed to the record site as `challenge_domains` —
    unused. Classifying the WRAPPED prompt instead filed 23 of 23 journal runs
    as `sql` (the boilerplate carried the literal and `\\bsql\\b` is tested
    first), so `sql` reached expert tier on work that was never SQL.

    Rewording the boilerplate stopped it DOMINATING; this never looks at the
    wrapper at all."""
    wrapped = ("A deterministic `input.log` fixture has been written in your "
               "working directory. Write a solution.py that... ### ORIGINAL "
               "USER REQUEST count the 5xx lines per IP in access.log")
    assert DREAM.resolve_cluster_key(
        {}, "", wrapped, challenge_domains=["regex_parse"]) == "regex_parse"
    # a bogus domain is ignored rather than minting a fake cluster
    assert DREAM.resolve_cluster_key(
        {}, "", "write a SQL query", challenge_domains=["not_a_cluster"]) == "sql"
    # …and the template still outranks it
    assert DREAM.resolve_cluster_key(
        {}, "bash", wrapped, challenge_domains=["regex_parse"]) == "bash"
