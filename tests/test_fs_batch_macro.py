"""§4F NEXT STEPS item 1 — the `file_system` BATCH macro.

Turn DEPTH is this project's strongest measured failure predictor (17.8% at
step 1 → 60.6% at step 12) and the risk governor cannot fire before step 6 by
construction, so removing loop steps attacks the failure RATE, not just
latency. This module covers the two mechanisms that remove them:

  * MULTI-PATH READ (`paths`) — several files, one call.
  * POST-EDIT VIEW — a successful `replace` hands back the changed lines as
    they now are on disk, so the trailing verify-read is unnecessary.

Both ship as the TREATMENT arm of the `fs_batch` experiment, never as a
default. The tests that matter most here are not the happy paths:

  * BACKWARD COMPATIBILITY — the control arm must be byte-identical to the
    pre-macro tool, because the recorded fixture corpus and every existing
    caller depend on it.
  * PARTIAL FAILURE vs OUTCOME LABELS — a batch where 1 of 5 paths failed
    must NOT read as a wholly failed tool call to ANY of the three live
    failure classifiers (`agent._res_is_error`,
    `distill.outcome_heuristics`, `composed_skills._step_result_ok`).
    Getting that wrong moves outcome labels in the corpus, which is a
    heavier defect than the macro is a win. The guard is the header LENGTH
    (per-path bodies legitimately contain "Error:"), so it is mutation-
    checked: the same body under a short header IS misclassified.
  * TOTAL FAILURE — with zero paths read the result must be Error-shaped.
    The mirror defect (a wholly failed call laundered into a success) is
    exactly what the 2026-08-04 shape rule exists to stop.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from types import SimpleNamespace

import pytest

from ghost_agent.tools.file_system import (
    ReadBudget,
    _BATCH_HEADER_MIN_CHARS,
    _BATCH_MAX_PATHS,
    parse_batch_paths,
    post_edit_view,
    resolve_batch_entry,
    tool_file_system,
    tool_read_files,
    tool_replace_text,
)


# ── the three live "did this tool call fail?" classifiers ─────────────

def _agent_says_failed(res: str) -> bool:
    """`core/agent.py` turn loop: `_res_is_error`. Drives the strike ledger,
    the pre-flight failure guard, edit-churn and `last_was_failure` (which
    becomes the trajectory's STRUCTURAL failure)."""
    return str(res).startswith(
        ("Error:", "ERROR", "SYSTEM ERROR", "Critical Tool Error"))


def _corpus_says_failed(res: str) -> bool:
    from ghost_agent.distill.outcome_heuristics import looks_like_tool_error
    return looks_like_tool_error(res)


def _macro_says_failed(res: str) -> bool:
    from ghost_agent.tools.composed_skills import _step_result_ok
    return not _step_result_ok(res)


def _all_classifiers(res: str):
    return (_agent_says_failed(res), _corpus_says_failed(res),
            _macro_says_failed(res))


# ── `paths` parsing ───────────────────────────────────────────────────

def test_parse_batch_paths_accepts_every_transport_shape():
    # A real list (native tool-call path / a future JSON transport).
    assert parse_batch_paths(["a.py", "b.py"]) == ["a.py", "b.py"]
    # The live agent speaks the XML dialect, where EVERY argument arrives as
    # a string — a schema-declared array reaches the tool as literal text.
    assert parse_batch_paths('["a.py", "b.py"]') == ["a.py", "b.py"]
    assert parse_batch_paths("a.py\nb.py") == ["a.py", "b.py"]
    assert parse_batch_paths("a.py, b.py") == ["a.py", "b.py"]
    assert parse_batch_paths("a.py") == ["a.py"]
    # Order-preserving dedupe: a repeated path must not be read twice.
    assert parse_batch_paths(["x", "y", "x"]) == ["x", "y"]
    assert parse_batch_paths(None) == [] and parse_batch_paths("") == []
    assert parse_batch_paths("   ") == []


def test_parse_batch_paths_refuses_to_invent_filenames_from_json_objects():
    """False-positive audit for a tolerant parser (this project's own rule).

    A bracketed payload that is NOT a list of strings is structured-but-wrong.
    Comma-splitting it would manufacture filenames out of JSON punctuation;
    the whole string is passed through instead, so the model gets ONE legible
    "not found" rather than five nonsense ones."""
    got = parse_batch_paths('[{"path": "a.py"}, {"path": "b.py"}]')
    assert got == ['[{"path": "a.py"}, {"path": "b.py"}]']
    # Nested lists are equally not a path list.
    assert parse_batch_paths('[["a.py"]]') == ['[["a.py"]]']


@pytest.mark.asyncio
async def test_transport_corrupted_paths_fail_legibly(tmp_path):
    """The live corpus carries 17 tool calls (of 3572) whose ARGUMENT VALUES
    absorbed the next parameter's markup — the dual-dialect corruption family
    (`<parameter=path>` leaking into an `operation` value). `paths` adds no
    new exposure to that (it is one more parameter, like `path`), but it must
    fail the SAME way `path` does: a legible miss, never a read of some other
    file the split invented."""
    (tmp_path / "index.html").write_text("REAL\n")
    corrupt = '["index.html</parameter>\n<parameter=chunk_size>', '32000"]'
    res = await tool_file_system(
        operation="read", sandbox_dir=tmp_path,
        paths="".join(corrupt), fs_batch_enabled=True)
    assert "REAL" not in res
    assert _all_classifiers(res) == (True, True, True)


def test_resolve_batch_entry_ranges(tmp_path):
    assert resolve_batch_entry("train.py:120-180", tmp_path) == ("train.py", 120, 180)
    assert resolve_batch_entry("train.py:120", tmp_path) == ("train.py", 120, None)
    assert resolve_batch_entry("train.py", tmp_path) == ("train.py", None, None)
    # A real file whose NAME ends that way wins over the range reading.
    (tmp_path / "weird:10-20").write_text("hi")
    assert resolve_batch_entry("weird:10-20", tmp_path) == ("weird:10-20", None, None)


# ── multi-path read ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_read_all_ok_returns_every_file(tmp_path):
    (tmp_path / "a.py").write_text("AAA\n")
    (tmp_path / "b.py").write_text("BBB\n")
    res = await tool_read_files(["a.py", "b.py"], tmp_path)
    assert "AAA" in res and "BBB" in res
    assert "[1/2] OK" in res and "[2/2] OK" in res
    assert _all_classifiers(res) == (False, False, False)


@pytest.mark.asyncio
async def test_batch_read_partial_failure_is_not_a_failed_call(tmp_path):
    """1 of 3 paths failing must not brand the call failed anywhere.

    This is the sharpest risk in the macro: the 2026-08-04 shape rule keys on
    "did every tool call fail", so a mislabelled partial batch moves outcome
    labels in the corpus."""
    (tmp_path / "a.py").write_text("AAA\n")
    (tmp_path / "b.py").write_text("BBB\n")
    res = await tool_read_files(["a.py", "missing.py", "b.py"], tmp_path)
    # The successes are complete and the failure is explained, in one result.
    assert "AAA" in res and "BBB" in res
    assert "NOT READ" in res and "missing.py" in res
    assert res.startswith("BATCH READ")
    assert _all_classifiers(res) == (False, False, False)


@pytest.mark.asyncio
async def test_partial_batch_header_is_the_guard_and_is_load_bearing(tmp_path):
    """MUTATION CHECK. The per-path bodies legitimately contain "Error:";
    what keeps the corpus labeller off them is that the envelope pushes them
    past its 120-char sniff window. Assert the invariant at the SAME
    granularity the classifier reads it, and show the raw body really would
    be misclassified without the envelope."""
    (tmp_path / "a.py").write_text("AAA\n")
    res = await tool_read_files(["a.py", "missing.py"], tmp_path)
    head = res.split("\n", 1)[0]
    assert len(head) >= _BATCH_HEADER_MIN_CHARS
    # This is `_looks_like_tool_error`'s actual window and marker set.
    window = res.strip()[:120].lower()
    for marker in ("error:", "[error]", "exception", "traceback", "failed:",
                   "syntax error", "operation failed"):
        assert marker not in window, f"{marker!r} leaked into the sniff window"
    assert not _corpus_says_failed(res)
    # Mutation: the SAME failing body, without the envelope in front of it,
    # IS classified as a failed call — so the envelope is load-bearing.
    raw_failure = res.split("— NOT READ\n", 1)[1]
    assert _corpus_says_failed(raw_failure)
    assert _corpus_says_failed("PARTIAL.\n" + raw_failure)


@pytest.mark.asyncio
async def test_batch_read_total_failure_is_error_shaped(tmp_path):
    """The mirror defect: every path failed, so the call DID fail and must be
    labelled that way by all three classifiers."""
    res = await tool_read_files(["nope1.py", "nope2.py"], tmp_path)
    assert res.startswith("Error:")
    assert _all_classifiers(res) == (True, True, True)


@pytest.mark.asyncio
async def test_batch_read_respects_the_existing_read_budget(tmp_path):
    """Bounding is DELEGATED to `ReadBudget`, not reinvented: the second file
    is refused by the same cumulative allowance a parallel pair of separate
    reads would hit, and the batch is then PARTIAL, not failed."""
    (tmp_path / "a.txt").write_text("x" * 400)
    (tmp_path / "b.txt").write_text("y" * 400)
    budget = ReadBudget(500)
    res = await tool_read_files(["a.txt", "b.txt"], tmp_path,
                                max_context=131072, read_budget=budget)
    assert "x" * 400 in res
    assert "would overflow the context window" in res
    assert budget.spent == 400
    assert _all_classifiers(res) == (False, False, False)


@pytest.mark.asyncio
async def test_batch_read_fan_out_cap_is_reported_never_silent(tmp_path):
    names = []
    for i in range(_BATCH_MAX_PATHS + 3):
        (tmp_path / f"f{i}.txt").write_text(f"body{i}\n")
        names.append(f"f{i}.txt")
    res = await tool_read_files(names, tmp_path)
    assert f"[{_BATCH_MAX_PATHS}/{_BATCH_MAX_PATHS}] OK" in res
    assert f"past the {_BATCH_MAX_PATHS}-path batch limit were NOT attempted" in res
    assert f"f{_BATCH_MAX_PATHS + 2}.txt" in res          # named, not just counted
    assert f"body{_BATCH_MAX_PATHS}" not in res           # and not read


@pytest.mark.asyncio
async def test_batch_read_inline_line_ranges(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(1, 51)))
    (tmp_path / "small.py").write_text("only\n")
    res = await tool_read_files(["big.py:3-5", "small.py"], tmp_path)
    assert "line3" in res and "line5" in res
    assert "line40" not in res
    assert "(lines 3-5)" in res
    assert "only" in res


@pytest.mark.asyncio
async def test_batch_read_empty_list_is_an_instruction_not_a_crash(tmp_path):
    res = await tool_read_files([], tmp_path)
    assert res.startswith("SYSTEM INSTRUCTION")


# ── dispatcher: arm gating + backward compatibility ───────────────────

@pytest.mark.asyncio
async def test_control_arm_ignores_paths_exactly_as_today(tmp_path):
    """BACKWARD COMPATIBILITY. Without the treatment flag the tool behaves
    exactly as it did before the macro existed: `paths` is an unknown kwarg
    swallowed by **kwargs, and a `path` read is byte-identical."""
    (tmp_path / "a.py").write_text("AAA\n")
    (tmp_path / "b.py").write_text("BBB\n")
    res = await tool_file_system(operation="read", path="a.py",
                                 sandbox_dir=tmp_path, paths=["a.py", "b.py"])
    assert "AAA" in res and "BBB" not in res
    assert "BATCH READ" not in res
    direct = await tool_file_system(operation="read", path="a.py",
                                    sandbox_dir=tmp_path)
    assert res == direct


@pytest.mark.asyncio
async def test_treatment_arm_batches(tmp_path):
    (tmp_path / "a.py").write_text("AAA\n")
    (tmp_path / "b.py").write_text("BBB\n")
    res = await tool_file_system(operation="read", path="a.py",
                                 sandbox_dir=tmp_path, paths=["b.py"],
                                 fs_batch_enabled=True)
    assert res.startswith("BATCH READ")
    assert "AAA" in res and "BBB" in res


@pytest.mark.asyncio
async def test_single_path_in_treatment_keeps_the_plain_read_shape(tmp_path):
    """One path is not a batch. Keeping the ordinary shape means a lone
    failure stays honestly Error-shaped instead of hiding inside an envelope
    that claims a partial success."""
    (tmp_path / "a.py").write_text("AAA\n")
    res = await tool_file_system(operation="read", sandbox_dir=tmp_path,
                                 paths=["a.py"], fs_batch_enabled=True)
    plain = await tool_file_system(operation="read", path="a.py",
                                   sandbox_dir=tmp_path)
    assert res == plain
    miss = await tool_file_system(operation="read", sandbox_dir=tmp_path,
                                  paths=["gone.py"], fs_batch_enabled=True)
    assert _all_classifiers(miss) == (True, True, True)


@pytest.mark.asyncio
async def test_plain_path_never_acquires_range_parsing_in_treatment(tmp_path):
    """FALSE-POSITIVE AUDIT for the range suffix (a tolerant parser).

    The inline `file:10-20` range is a `paths` feature. A real file whose
    name happens to contain ':<digits>' must not be silently retargeted just
    because the request is in the treatment arm — a plain `path` call has to
    behave identically in both arms."""
    d = tmp_path / "logs"
    d.mkdir()
    (d / "2026-08-04").write_text("WRONG FILE\n")
    ctrl = await tool_file_system(operation="read", path="logs/2026-08-04:12",
                                  sandbox_dir=tmp_path)
    treat = await tool_file_system(operation="read", path="logs/2026-08-04:12",
                                   sandbox_dir=tmp_path, fs_batch_enabled=True)
    assert ctrl == treat
    assert "WRONG FILE" not in treat


@pytest.mark.asyncio
async def test_treatment_single_entry_range_and_explicit_range_precedence(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(1, 51)))
    res = await tool_file_system(operation="read", sandbox_dir=tmp_path,
                                 paths=["big.py:3-4"], fs_batch_enabled=True)
    assert "line3" in res and "line4" in res and "line10" not in res
    # An explicit start_line/end_line wins over the inline suffix.
    res2 = await tool_file_system(operation="read", sandbox_dir=tmp_path,
                                  paths=["big.py:3-4"], start_line=10,
                                  end_line=11, fs_batch_enabled=True)
    assert "line10" in res2 and "line3\n" not in res2


@pytest.mark.asyncio
async def test_paths_never_batches_a_destructive_operation(tmp_path):
    """Deliberate scope limit: only `read` fans out. A batch delete/write is
    a data-loss shape with no measured demand behind it."""
    (tmp_path / "a.py").write_text("AAA\n")
    (tmp_path / "b.py").write_text("BBB\n")
    res = await tool_file_system(operation="delete", path="a.py",
                                 sandbox_dir=tmp_path, paths=["a.py", "b.py"],
                                 fs_batch_enabled=True)
    assert "BATCH" not in res
    assert (tmp_path / "b.py").exists()


# ── post-edit view (removes the trailing verify-read) ─────────────────

def test_post_edit_view_shows_the_new_lines_with_numbers():
    prev = "\n".join(f"line{i}" for i in range(1, 11))
    new = prev.replace("line5", "CHANGED")
    view = post_edit_view(prev, new)
    assert "POST-EDIT VIEW" in view
    assert "    5 | CHANGED" in view
    assert "    3 | line3" in view          # context above
    assert "line9" not in view              # bounded, not the whole file
    assert post_edit_view(prev, prev) == ""  # nothing changed → nothing said


def test_post_edit_view_is_bounded():
    prev = "\n".join(f"line{i}" for i in range(1, 400))
    new = "\n".join(f"CH{i}" for i in range(1, 400))
    view = post_edit_view(prev, new)
    assert len(view) < 2200


def test_post_edit_view_stays_linear_on_a_huge_file():
    """difflib is worst-case quadratic and the replace path around it is
    linear in file size; above the line cap the view falls back to a
    prefix/suffix scan rather than becoming the slowest thing in the call."""
    from ghost_agent.tools.file_system import _POST_EDIT_DIFF_MAX_LINES
    n = _POST_EDIT_DIFF_MAX_LINES * 3
    prev = "\n".join(f"line{i}" for i in range(1, n))
    new = prev.replace(f"line{n // 2}", "CHANGED")
    view = post_edit_view(prev, new)
    assert "CHANGED" in view and len(view) < 2000
    assert post_edit_view(prev, prev) == ""


def test_post_edit_view_handles_deletion_seams():
    prev = "a\nb\nc\nd\n"
    view = post_edit_view(prev, "a\nd\n")
    assert "POST-EDIT VIEW" in view and "| d" in view


@pytest.mark.asyncio
async def test_replace_post_edit_only_in_treatment(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\ny = 2\nz = 3\n")
    control = await tool_replace_text("m.py", "y = 2", "y = 22", tmp_path)
    assert control.startswith("SUCCESS")
    assert "POST-EDIT VIEW" not in control

    (tmp_path / "n.py").write_text("x = 1\ny = 2\nz = 3\n")
    treat = await tool_replace_text("n.py", "y = 2", "y = 22", tmp_path,
                                    post_edit=True)
    assert treat.startswith("SUCCESS")
    assert "POST-EDIT VIEW" in treat and "y = 22" in treat


@pytest.mark.asyncio
async def test_post_edit_view_covers_the_search_replace_block_form(tmp_path):
    (tmp_path / "m.py").write_text("alpha = 1\nbeta = 2\ngamma = 3\n")
    block = "<<<< SEARCH\nbeta = 2\n====\nbeta = 22\n>>>>"
    res = await tool_replace_text("m.py", block, None, tmp_path, post_edit=True)
    assert res.startswith("SUCCESS")
    assert "POST-EDIT VIEW" in res and "beta = 22" in res


@pytest.mark.asyncio
async def test_no_post_edit_view_when_the_edit_was_rolled_back(tmp_path):
    """A REJECTED edit left the file unchanged; showing a "post-edit view"
    would assert a change that did not happen."""
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    res = await tool_replace_text("m.py", "    return 1", "return 1", tmp_path,
                                  post_edit=True)
    assert res.startswith("REJECTED")
    assert "POST-EDIT VIEW" not in res
    assert (tmp_path / "m.py").read_text() == "def f():\n    return 1\n"


@pytest.mark.asyncio
async def test_replace_dispatch_threads_the_arm(tmp_path):
    (tmp_path / "m.py").write_text("a = 1\nb = 2\n")
    res = await tool_file_system(operation="replace", path="m.py",
                                 content="b = 2", replace_with="b = 3",
                                 sandbox_dir=tmp_path, fs_batch_enabled=True)
    assert "POST-EDIT VIEW" in res
    (tmp_path / "m2.py").write_text("a = 1\nb = 2\n")
    res2 = await tool_file_system(operation="replace", path="m2.py",
                                  content="b = 2", replace_with="b = 3",
                                  sandbox_dir=tmp_path)
    assert "POST-EDIT VIEW" not in res2


@pytest.mark.asyncio
async def test_partial_multi_block_replace_still_says_which_blocks_failed(tmp_path):
    """MULTI-EDIT ATOMICITY, documented rather than assumed. The existing
    multi-block form is BEST-EFFORT per block and ATOMIC per file: applied
    blocks are written in one `write_text`, and a block that did not match is
    named. The result must never read as a clean success."""
    (tmp_path / "m.py").write_text("alpha = 1\nbeta = 2\n")
    blocks = ("<<<< SEARCH\nalpha = 1\n====\nalpha = 11\n>>>>\n"
              "<<<< SEARCH\nnot_in_the_file = 9\n====\nzzz = 9\n>>>>")
    res = await tool_replace_text("m.py", blocks, None, tmp_path, post_edit=True)
    assert "Applied 1 SEARCH/REPLACE blocks" in res
    assert "blocks failed" in res
    assert (tmp_path / "m.py").read_text() == "alpha = 11\nbeta = 2\n"


# ── the experiment wiring ─────────────────────────────────────────────

def test_experiment_is_registered_with_both_keys():
    from ghost_agent.core import experiments as ex
    spec = {s.name: s for s in ex.DEFAULT_SPECS}["fs_batch"]
    assert spec.arms == (ex.CONTROL, ex.TREATMENT)
    # A powered triggered-only comparison needs the trigger key…
    assert ex.TRIGGER_KEYS["fs_batch"] == "fs_batch_fired"
    # …and the fixture corpus needs the context key, because the treatment
    # changes the ADVERTISED SCHEMA and GHOST_LLM_RECORD is capturing.
    assert "fs_batch_context" in ex.CONTEXT_MUTATING_KEYS
    # The name must survive the trajectory redactor (an arm stored as
    # "<REDACTED>" silently destroys the whole analysis).
    from ghost_agent.distill.redact import _is_sensitive_key
    assert not _is_sensitive_key("fs_batch")


def test_stamped_flags_drive_the_two_readers():
    from ghost_agent.core.experiments import context_was_mutated, trigger_fired
    treat = SimpleNamespace(extra={"experiments": {"fs_batch": "treatment"},
                                   "fs_batch_context": True,
                                   "fs_batch_fired": True})
    ctrl = SimpleNamespace(extra={"experiments": {"fs_batch": "control"},
                                  "fs_batch_context": False,
                                  "fs_batch_fired": False})
    assert context_was_mutated(treat) and not context_was_mutated(ctrl)
    # PRESENCE is the trigger (both arms), VALUE is compliance — so a control
    # turn whose trigger fired still counts in the triggered-only block.
    assert trigger_fired(treat, "fs_batch") and trigger_fired(ctrl, "fs_batch")
    assert not trigger_fired(SimpleNamespace(extra={}), "fs_batch")


def test_treatment_turns_are_excluded_from_the_phase_2b_fixture_corpus(tmp_path):
    """Ship rule 3, verified end to end: without this the optimizer would
    replay a tool block only half of production ever saw."""
    import json as _json
    from ghost_agent.optim.tool_fixtures import mine_fixtures

    traj_root = tmp_path / "trajectories" / "2026-08-05"
    traj_root.mkdir(parents=True)

    def _traj(sid, extra):
        return {"id": sid, "session_id": sid, "task_kind": "user_request",
                "outcome": "passed", "user_request": "read the files",
                "tool_calls": [], "extra": extra,
                "timestamp": "2026-08-05T12:00:00Z"}

    # The collector globs `session-*.jsonl`; any other name is simply not
    # walked (and the miner then reports `unjoined`, not an error).
    (traj_root / "session-a.jsonl").write_text(
        _json.dumps(_traj("req-ctrl",
                          {"experiments": {"fs_batch": "control"},
                           "fs_batch_context": False})) + "\n"
        + _json.dumps(_traj("req-treat",
                            {"experiments": {"fs_batch": "treatment"},
                             "fs_batch_context": True})) + "\n")

    def _record(req):
        return {"kind": "chat_completion_stream", "ts": "2026-08-05T12:00:00Z",
                "request_id": req, "session_id": "s1", "ordinal": 1,
                "payload": {"tools": [{"function": {"name": "file_system"}}],
                            "messages": []},
                "response": {"choices": [{"message": {"tool_calls": [
                    {"function": {"name": "file_system", "arguments": "{}"}}]}}]}}

    rec = tmp_path / "rec.jsonl"
    rec.write_text(_json.dumps(_record("req-ctrl")) + "\n"
                   + _json.dumps(_record("req-treat")) + "\n")
    fixtures, stats = mine_fixtures([rec], tmp_path / "trajectories",
                                    era_cutoff_local="2026-07-31T19:15")
    assert [f.request_id for f in fixtures] == ["req-ctrl"]
    assert stats["experiment_context_excluded"] == 1


# ── the advertised schema ─────────────────────────────────────────────

def _fs_def(tools):
    return next(t for t in tools
                if (t.get("function") or {}).get("name") == "file_system")


def test_treatment_schema_adds_paths_and_relaxes_required():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS, _apply_fs_batch_schema
    before = _fs_def(TOOL_DEFINITIONS)
    after = _fs_def(_apply_fs_batch_schema(list(TOOL_DEFINITIONS)))
    props = after["function"]["parameters"]["properties"]
    assert "paths" in props and props["paths"]["type"] == "array"
    # A pure `paths` read has no single path, so `path` cannot stay required.
    assert after["function"]["parameters"]["required"] == ["operation"]
    assert "'paths'" in after["function"]["description"]
    # Copy-on-write: the shared definition must be untouched, or a treatment
    # request leaks its schema into the next control request.
    assert "paths" not in before["function"]["parameters"]["properties"]
    assert before["function"]["parameters"]["required"] == ["operation", "path"]


def test_non_file_system_tools_are_passed_through_unchanged():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS, _apply_fs_batch_schema
    out = _apply_fs_batch_schema(list(TOOL_DEFINITIONS))
    for a, b in zip(TOOL_DEFINITIONS, out):
        if (a.get("function") or {}).get("name") != "file_system":
            assert a is b


def test_schema_gate_reads_the_arm_and_fails_closed():
    from ghost_agent.tools.registry import _fs_batch_active
    ctx = SimpleNamespace(_experiment_arms=("r1", {"fs_batch": "treatment"}))
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("r1")
    try:
        assert _fs_batch_active(ctx) is True
        ctx2 = SimpleNamespace(_experiment_arms=("r1", {"fs_batch": "control"}))
        assert _fs_batch_active(ctx2) is False
        # A request that is not the stashed one must NOT inherit its arm.
        request_id_context.set("other")
        assert _fs_batch_active(ctx) is False
    finally:
        request_id_context.reset(tok)
    # No context at all → control, never an exception on the request path.
    assert _fs_batch_active(None) is False


# ── the acceptance instrument (simulated corpus) ──────────────────────

def _c(name, **args):
    return SimpleNamespace(name=name, arguments=args, result="", error="")


def test_collapse_merges_consecutive_reads():
    from ghost_agent.optim.tool_ontology import collapse_fs_batch
    calls = [_c("file_system", operation="read", path="a.py"),
             _c("file_system", operation="read", path="b.py"),
             _c("file_system", operation="read", path="c.py"),
             _c("execute", command="pytest")]
    out = collapse_fs_batch(calls)
    assert [c.name for c in out] == ["file_system", "execute"]
    assert out[0].arguments["paths"] == ["a.py", "b.py", "c.py"]


def test_collapse_chunks_at_the_batch_cap():
    from ghost_agent.optim.tool_ontology import collapse_fs_batch
    calls = [_c("file_system", operation="read", path=f"f{i}.py")
             for i in range(25)]
    out = collapse_fs_batch(calls, max_batch=12)
    assert len(out) == 3                       # 12 + 12 + 1
    assert out[-1].arguments["path"] == "f24.py"


def test_collapse_drops_only_a_same_file_verify_read():
    from ghost_agent.optim.tool_ontology import collapse_fs_batch
    same = [_c("file_system", operation="replace", path="a.py"),
            _c("file_system", operation="read", path="a.py")]
    assert len(collapse_fs_batch(same)) == 1
    other = [_c("file_system", operation="replace", path="a.py"),
             _c("file_system", operation="read", path="b.py")]
    assert len(collapse_fs_batch(other)) == 2


def test_collapse_leaves_pagination_and_writes_alone():
    """Paging (`read_chunked` runs) is a DIFFERENT fix and is deliberately
    out of scope; claiming it here would overstate the macro."""
    from ghost_agent.optim.tool_ontology import collapse_fs_batch
    calls = [_c("file_system", operation="read_chunked", path="big.md", page=1),
             _c("file_system", operation="read_chunked", path="big.md", page=2),
             _c("file_system", operation="write", path="a.py"),
             _c("file_system", operation="write", path="b.py")]
    assert len(collapse_fs_batch(calls)) == 4


def test_simulated_corpus_is_mineable():
    from ghost_agent.optim.tool_ontology import mine_sequences, simulate_fs_batch
    trajs = [SimpleNamespace(
        id=f"t{i}", task_kind="user_request",
        tool_calls=[_c("file_system", operation="read", path="a.py"),
                    _c("file_system", operation="read", path="b.py"),
                    _c("execute", command="pytest")])
        for i in range(4)]
    before = {m.sequence: m for m in mine_sequences(trajs, min_support=3)}
    after = {m.sequence: m
             for m in mine_sequences(list(simulate_fs_batch(trajs)),
                                     min_support=3)}
    assert ("file_system", "file_system") in before
    assert ("file_system", "file_system") not in after
