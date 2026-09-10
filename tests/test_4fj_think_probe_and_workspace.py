"""§4FJ pins: (1) the coding executor's no-think regime has a DISARM path —
after GHOST_CODING_THINK_PROBE_AFTER no-think spec calls, or right after a
verify failure on a no-think spec, one call thinks again as a probe; a clean
probe re-enables thinking, an aborted probe extends the window; (2) the
`workspace` tool no longer advertises itself as a place to do work, and the
diet bench can measure that change against the old text as a fixed head.
"""
import inspect
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from ghost_agent.core import coding_executor as ce
from ghost_agent.core.coding_executor import _generate_build_spec, build_coding_task
from ghost_agent.tools import registry as R
from tests.test_coding_executor_think_policy_and_append_guards import (
    StreamingFakeLLM, _ceiling_chunks, _clean_chunks, _is_nothink)
from tests.test_coding_executor import FakeRunner, SPEC_OK as SPEC_WITH_VERIFY


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "2")
    monkeypatch.delenv("GHOST_CODING_THINK_PROBE_AFTER", raising=False)
    monkeypatch.delenv("GHOST_HOME", raising=False)


async def _spec(llm):
    spec, _ = await _generate_build_spec(llm, "m", "build the thing", "")
    return spec


async def _trip(llm):
    """Two aborted think phases → the tripped (no-think) regime."""
    await _spec(llm)
    await _spec(llm)
    st = ce._think_state(llm)
    assert llm.stream_calls == 2 and st["skipping"] == 1 and st["aborts"] == 2


@pytest.mark.asyncio
async def test_probe_after_k_nothink_calls_and_a_clean_probe_reenables_thinking():
    """World where it fails: the pre-§4FJ sticky regime — after the trip, no
    spec call ever streams a think phase again in this process."""
    llm = StreamingFakeLLM([_ceiling_chunks(), _ceiling_chunks(), _clean_chunks(), _clean_chunks()])
    await _trip(llm)
    for _ in range(3):
        assert (await _spec(llm)).get("files")
    # the window: three no-think calls, no streaming at all
    assert llm.stream_calls == 2 and llm.chat_calls == 5
    assert all(_is_nothink(p) for p in llm.chat_payloads[2:])
    assert ce._think_state(llm)["nothink_calls"] == 3
    # the 4th call is the probe: it streams, the think is clean → full reset
    assert (await _spec(llm)).get("files")
    assert llm.stream_calls == 3 and llm.chat_calls == 5
    st = ce._think_state(llm)
    assert (st["aborts"], st["skipping"], st["nothink_calls"], st["probe"]) == (0, 0, 0, 0)
    # thinking is back on for good: the next call streams too
    assert (await _spec(llm)).get("files")
    assert llm.stream_calls == 4


@pytest.mark.asyncio
async def test_an_aborted_probe_extends_the_window_by_another_k_calls():
    llm = StreamingFakeLLM()             # every think phase hits the ceiling
    await _trip(llm)
    for _ in range(3):
        await _spec(llm)
    await _spec(llm)                     # probe: streams (3rd stream), aborts, no-think retry
    st = ce._think_state(llm)
    assert llm.stream_calls == 3
    assert (st["skipping"], st["aborts"], st["nothink_calls"], st["probe"]) == (1, 3, 0, 0)
    for _ in range(3):
        await _spec(llm)                 # a fresh window of three no-think calls
    assert llm.stream_calls == 3
    await _spec(llm)                     # …then the next probe
    assert llm.stream_calls == 4


@pytest.mark.asyncio
async def test_probe_knob_zero_or_negative_is_the_legacy_sticky_regime(monkeypatch):
    for val in ("0", "-1"):
        monkeypatch.setenv("GHOST_CODING_THINK_PROBE_AFTER", val)
        llm = StreamingFakeLLM()
        await _trip(llm)
        for _ in range(10):
            await _spec(llm)
        assert llm.stream_calls == 2, val


@pytest.mark.asyncio
async def test_verify_failure_on_a_nothink_spec_probes_on_the_very_next_call():
    """At the REAL call site (build_coding_task → _run_verify → hook): a spec
    built no-think fails its verify → the next spec call thinks. World where
    it fails: the hook is missing at the verify site (the state stays at
    nothink_calls=1 and the next call is no-think again)."""
    llm = StreamingFakeLLM(retry_content=SPEC_WITH_VERIFY)
    await _trip(llm)
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n SyntaxError")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "build x", tool_runner=runner, max_attempts=1)
    assert not res.ok and "verify failed" in res.summary
    st = ce._think_state(llm)
    assert llm.stream_calls == 2                      # the build's spec call was no-think
    assert (st["skipped_think_last"], st["nothink_calls"]) == (1, 3)   # K = THINK_PROBE_AFTER_DEFAULT
    assert ce.THINK_PROBE_AFTER_DEFAULT == 3
    await _spec(llm)                                  # probes now, not after two more calls
    assert llm.stream_calls == 3


@pytest.mark.asyncio
async def test_verify_only_spec_failing_verify_also_probes():
    """The OTHER verify site: a spec with no files and a verify command
    (the "nothing to build, verify the existing deliverable" path). World
    where it fails: the hook sits at the written-files site only."""
    verify_only = json.dumps({"files": [], "verify": "python3 -m pytest -q", "summary": "verify only"})
    llm = StreamingFakeLLM(retry_content=verify_only)
    await _trip(llm)
    runner = FakeRunner(verify_out="--- EXECUTION RESULT ---\nEXIT CODE: 1\nFAILED tests")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "verify x", tool_runner=runner, max_attempts=1)
    assert not res.ok
    st = ce._think_state(llm)
    assert llm.stream_calls == 2
    assert (st["skipped_think_last"], st["nothink_calls"]) == (1, 3)


@pytest.mark.asyncio
async def test_passing_verify_must_not_arm_a_probe():
    """§4FK review C-2: hoisting the hook out of `if vfail:` makes every
    SUCCESSFUL no-think build pay a ~115 s aborted think on the next call."""
    llm = StreamingFakeLLM(retry_content=SPEC_WITH_VERIFY)
    await _trip(llm)
    runner = FakeRunner(verify_out="--- EXECUTION RESULT ---\nEXIT CODE: 0\nok")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "build x", tool_runner=runner, max_attempts=1)
    assert res.ok
    assert ce._think_state(llm)["nothink_calls"] == 1          # not armed
    await _spec(llm)
    assert llm.stream_calls == 2                                 # still no-think


@pytest.mark.asyncio
async def test_budget_exhausted_think_with_no_stream_abort_counts_as_aborted():
    """§4FK review M-1: reasoning under the ceiling, no content, no stream
    abort — the 2026-07-06 live shape. Scored as clean, the streak never
    trips and the whole policy is dead."""
    from tests.test_coding_executor_think_policy_and_append_guards import _diverse_chunks
    quiet = _diverse_chunks(3_000)                               # reasoning only, then [DONE]
    llm = StreamingFakeLLM([list(quiet), list(quiet)])
    await _spec(llm)
    await _spec(llm)
    st = ce._think_state(llm)
    assert (st["aborts"], st["skipping"]) == (2, 1)


@pytest.mark.asyncio
async def test_verify_failure_on_a_think_built_spec_does_not_arm_a_probe():
    """A verify failure after a CLEAN think phase is a code problem, not a
    policy problem: the counter is untouched."""
    llm = StreamingFakeLLM([_clean_chunks()], retry_content=SPEC_WITH_VERIFY)
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n SyntaxError")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "build x", tool_runner=runner, max_attempts=1)
    assert not res.ok
    st = ce._think_state(llm)
    assert (st["skipping"], st["skipped_think_last"], st["nothink_calls"]) == (0, 0, 0)


@pytest.mark.asyncio
async def test_verify_failure_right_after_an_aborted_probe_does_not_reprobe_immediately():
    """Tripped regime, window elapsed, the probe ABORTS and its no-think retry
    yields a spec that fails verify. Thinking was just tried and failed, so
    the hook must NOT arm another probe: the next call is no-think and the
    window runs its full K. World where it fails: the hook ignores
    `last_nothink` (or a think phase never clears it) and every verify
    failure after an aborted probe re-probes at once — the ~115 s abort paid
    on every leaf, which is the §4EI shape the policy exists to avoid."""
    llm = StreamingFakeLLM(retry_content=SPEC_WITH_VERIFY)   # every think aborts
    await _trip(llm)
    for _ in range(3):
        await _spec(llm)                                     # the window
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n SyntaxError")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "build x", tool_runner=runner, max_attempts=1)
    assert not res.ok and llm.stream_calls == 3               # the probe streamed and aborted
    st = ce._think_state(llm)
    assert (st["skipping"], st["skipped_think_last"], st["nothink_calls"]) == (1, 0, 0)
    for _ in range(3):
        await _spec(llm)                                     # a full window, no re-probe
    assert llm.stream_calls == 3
    await _spec(llm)
    assert llm.stream_calls == 4                             # …and then the scheduled probe


@pytest.mark.asyncio
async def test_empty_upstream_response_during_a_probe_is_inconclusive_not_a_clean_think():
    """§4FJ review MAJOR 1. An error frame / empty stream returns ("", "",
    None); scored as a clean think it wiped the streak and printed "the probe
    succeeded" for a zero-byte response. World where it fails: the streak
    resets on nothing and the next two spec calls each pay a ~115 s abort."""
    empty = [b"data: [DONE]"]
    llm = StreamingFakeLLM([_ceiling_chunks(), _ceiling_chunks(), empty])
    await _trip(llm)
    for _ in range(3):
        await _spec(llm)
    await _spec(llm)                                     # the probe streams… nothing
    st = ce._think_state(llm)
    assert llm.stream_calls == 3
    assert (st["aborts"], st["skipping"], st["probe"], st["skipped_think_last"]) == (2, 1, 0, 0)
    assert st["nothink_calls"] >= ce._think_probe_after()   # window unchanged → probes again
    await _spec(llm)                                     # next call is the probe again (ceiling)
    assert llm.stream_calls == 4


@pytest.mark.asyncio
async def test_a_raising_stream_leaves_no_stale_probe_flag():
    """§4FJ review M-1: if the stream raises, the bookkeeping still runs, so a
    later unrelated abort is not logged as a failed probe."""
    class Raising(StreamingFakeLLM):
        async def stream_chat_completion(self, payload, is_background=False):
            self.stream_calls += 1
            raise RuntimeError("socket gone")
            yield  # pragma: no cover
    llm = Raising()
    st = ce._think_state(llm)
    st.update({"aborts": 2, "skipping": 1, "nothink_calls": 3})
    with pytest.raises(RuntimeError):
        await _spec(llm)
    assert (st["probe"], st["skipped_think_last"]) == (0, 0)


@pytest.mark.asyncio
async def test_failing_leaf_pays_at_most_half_its_attempts_in_probes():
    """§4FJ review MAJOR 2, the bound at the REAL call site: a 4-attempt leaf
    whose every verify fails, entered in the tripped regime, streams exactly
    2 think phases (attempts 2 and 4) — the hook and the aborted-probe rule
    alternate. World where it fails: the hook re-arms after an aborted probe
    (4 probes, ~460 s of aborts) or never arms (0)."""
    llm = StreamingFakeLLM(retry_content=SPEC_WITH_VERIFY)
    await _trip(llm)
    await _spec(llm)                                     # window at 1
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n SyntaxError")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "build x", tool_runner=runner, max_attempts=4)
    assert not res.ok
    assert llm.stream_calls == 2 + 2
    assert llm.chat_calls == 2 + 1 + 4


@pytest.mark.asyncio
async def test_skip_after_zero_never_probes_even_after_the_window_fills(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "0")
    llm = StreamingFakeLLM([_clean_chunks()] * 8)
    for _ in range(8):
        await _spec(llm)
    assert llm.stream_calls == 0 and llm.chat_calls == 8


@pytest.mark.asyncio
async def test_pre_4fj_state_records_are_migrated_with_safe_defaults():
    """A two-key record from before this change (same process, weak or
    fallback store) starts a fresh window with no pending probe."""
    llm = StreamingFakeLLM()
    ce._THINK_STATE[llm] = {"aborts": 2, "skipping": 1}
    st = ce._think_state(llm)
    assert st == {"aborts": 2, "skipping": 1, "nothink_calls": 0, "probe": 0, "skipped_think_last": 0}
    await _spec(llm)                                     # tripped → no-think, no probe yet
    assert llm.stream_calls == 0 and st["nothink_calls"] == 1

    class Unhashable:                                    # the fallback store
        __hash__ = None
        async def chat_completion(self, payload, is_background=False, **_kw):
            return {"choices": [{"message": {"content": "{}", "reasoning_content": ""}}]}
    u = Unhashable()
    ce._THINK_STATE_FALLBACK[id(u)] = {"aborts": 2, "skipping": 1}
    assert ce._think_state(u)["nothink_calls"] == 0 and ce._think_disabled_now(u) is True


@pytest.mark.asyncio
async def test_hook_is_idempotent_within_an_armed_window(caplog):
    import logging
    llm = StreamingFakeLLM(retry_content=SPEC_WITH_VERIFY)
    await _trip(llm)
    caplog.set_level(logging.WARNING)
    runner = FakeRunner(verify_out="Traceback (most recent call last):\n SyntaxError")
    ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
    await build_coding_task(ctx, "build x", tool_runner=runner, max_attempts=1)
    n1 = caplog.text.count("verify failed on a no-think spec")
    ce._note_nothink_verify_failure(llm, "again")        # a second failure in the same window
    assert caplog.text.count("verify failed on a no-think spec") == n1 == 1
    assert "build x" in caplog.text                      # attributed to the leaf


def test_think_disabled_now_is_a_pure_predicate():
    """§4FJ review M-2: a telemetry reader may ask without arming a probe."""
    llm = StreamingFakeLLM()
    st = ce._think_state(llm)
    st.update({"aborts": 2, "skipping": 1, "nothink_calls": 3})
    assert ce._think_disabled_now(llm) is False
    assert st["probe"] == 0


# --- workspace description + the bench pair ----------------------------------

def _workspace_description():
    return next(t["function"]["description"] for t in R.TOOL_DEFINITIONS
                if t["function"]["name"] == "workspace")


def test_workspace_describes_a_ledger_and_names_the_tools_that_do_work():
    """World where it fails: the old text is back ("files", "show me what
    you've been doing in my project") and coding requests route to it."""
    d = _workspace_description()
    assert "ALREADY HAPPENED" in d and "NEVER call this to work on files" in d
    for tool in ("file_system", "execute", "browser", "manage_projects"):
        assert tool in d
    assert "in my project" not in d and "(files," not in d
    assert "'summary'" in d and "'search'" in d          # the actions survived


def _load_bench():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "tool_head_diet_bench.py"
    spec = importlib.util.spec_from_file_location("tool_head_diet_bench", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_legacy_head_differs_from_the_live_head_only_in_the_workspace_description():
    """A verification that cannot distinguish is not one: the fixed head must
    carry the OLD text and nothing else may differ."""
    mod = _load_bench()
    live, legacy = mod._head("full"), mod._head("full-legacy-workspace")
    assert len(live) == len(legacy)
    diffs = [(a["function"]["name"]) for a, b in zip(live, legacy) if a != b]
    assert diffs == ["workspace"]
    old = next(t["function"]["description"] for t in legacy if t["function"]["name"] == "workspace")
    assert old == mod.WORKSPACE_DESCRIPTION_BEFORE_4FJ and old != _workspace_description()
    assert "show me what you've been doing in my project" in old
    with pytest.raises(ValueError):
        mod._head("nope")


def test_bench_pair_scores_each_head_on_its_own_rule_and_writes_pair_keys(tmp_path, monkeypatch):
    """Executed --pair run with the network faked: the legacy head picks
    `workspace` for a coding request, the live head picks `file_system`;
    only `diet` ever earns the catalog credit."""
    mod = _load_bench()
    monkeypatch.setattr(mod, "GHOST_HOME", tmp_path)
    p = tmp_path / "system" / "optim"; p.mkdir(parents=True)
    rows = [{"fixture_id": f"f{i}", "user_request": f"the ball spawns inside wall {i}, fix it",
             "chosen_tools": [{"name": "file_system"}]} for i in range(3)]
    (p / "tool_choice_fixtures.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def fake_call(msgs, tools):
        ws = next(t["function"]["description"] for t in tools if t["function"]["name"] == "workspace")
        return "workspace" if ws == mod.WORKSPACE_DESCRIPTION_BEFORE_4FJ else "file_system"
    monkeypatch.setattr(mod, "_call", fake_call)
    out = tmp_path / "out"
    assert mod.main(["--limit", "0", "--out", str(out), "--pair", "full-legacy-workspace,full"]) == 0
    summary = json.loads(next(out.glob("*.summary.json")).read_text())
    assert summary["pair"] == ["full-legacy-workspace", "full"]
    assert (summary["acc_full-legacy-workspace"], summary["acc_full"]) == (0.0, 1.0)
    assert (summary["b_full-legacy-workspace_only"], summary["c_full_only"]) == (0, 3)
    row = json.loads(next(out.glob("*.jsonl")).read_text().splitlines()[0])
    assert row["arms"] == ["full-legacy-workspace", "full"]
    assert row["picked_full-legacy-workspace"] == "workspace" and row["ok_full"] is True
    # the ledger reader honours the pair: rows of another pair are invisible
    assert mod._load_ledger(next(out.glob("*.jsonl")), ("full", "diet"))[0] == set()
    # scoring rule: the catalog credit belongs to a head that advertises the
    # catalog AND does not advertise the truth — never to a core truth
    # §4FX: the diet is retired from production; the bench owns its copy.
    hidden = next(iter(sorted(set(t["function"]["name"] for t in R.TOOL_DEFINITIONS) - set(mod.CORE))))
    assert mod._ok("diet", "tool_catalog", hidden) is True
    assert mod._ok("diet", "tool_catalog", "file_system") is False        # core truth: no credit
    assert mod._ok("full", "tool_catalog", hidden) is False
    assert mod._ok("full-legacy-workspace", "tool_catalog", hidden) is False
    assert mod._ok("full", "web_search", "web_search") is True
    # §4FK M6: the credit follows the head ACTUALLY SENT, not TOOL_HEAD_CORE —
    # vision_analysis is in the core set but only the live builder appends it,
    # so the bench diet head never advertises it and deferring to the catalog
    # on it is the designed path
    assert "vision_analysis" in mod.CORE
    assert "vision_analysis" not in mod._names(mod._head("diet"))
    assert mod._ok("diet", "tool_catalog", "vision_analysis") is True
    # the wiring at the call site (§4FK M-3): a full,diet pair on a HIDDEN
    # truth where full picks wrong and diet defers to the catalog
    rows = [{"fixture_id": "h1", "user_request": "list the top news headlines",
             "chosen_tools": [{"name": hidden}]}]
    (p / "tool_choice_fixtures.jsonl").write_text(json.dumps(rows[0]) + "\n")

    def fake_call2(msgs, tools):
        return "tool_catalog" if any(t["function"]["name"] == "tool_catalog" for t in tools) else "recall"
    monkeypatch.setattr(mod, "_call", fake_call2)
    out2 = tmp_path / "out2"
    assert mod.main(["--limit", "0", "--out", str(out2), "--pair", "full,diet"]) == 0
    row = json.loads(next(out2.glob("*.jsonl")).read_text().splitlines()[0])
    assert (row["ok_full"], row["ok_diet"], row["truth_hidden"]) == (False, True, True)
    # a malformed pair is refused before any call
    assert mod.main(["--pair", "full,full", "--out", str(out)]) == 2
    assert mod.main(["--pair", "full,nope", "--out", str(out)]) == 2


# --- the description optimiser cannot shed the pinned sentences (§4FJ review M1)

def test_tuned_description_must_keep_the_baselines_never_and_only_sentences():
    """World where it fails: a promoted tool_description.workspace.json
    rewrites the text toward the reward, drops the NEVER sentence, passes
    the size cap, and the live head serves the confusion again while the
    registry pin stays green."""
    base = _workspace_description()
    pinned = R._pinned_sentences("workspace", base)
    assert pinned == R.TOOL_DESC_PINNED["workspace"], "a pin drifted out of the live baseline"
    assert any(s.startswith("NEVER call this") for s in pinned)
    assert R._validate_tool_description("workspace", base, base) is True
    shed = base.replace(next(s for s in pinned if s.startswith("NEVER")), "")
    assert R._validate_tool_description("workspace", base, shed) is False
    reordered = "Prefix. " + base
    assert R._validate_tool_description("workspace", base, reordered) is True
    # every pinned sentence of every tool must occur verbatim in its live baseline
    live = {t["function"]["name"]: t["function"]["description"] for t in R.TOOL_DEFINITIONS}
    for tool, pins in R.TOOL_DESC_PINNED.items():
        assert tool in live, tool
        for sent in pins:
            assert sent in live[tool], (tool, sent[:40])
    # unpinned tools keep the tuning contract the read-site invariants rely on
    assert R._pinned_sentences("web_search", live["web_search"]) == []
    assert R._validate_tool_description("web_search", live["web_search"], "Search anything.") is True


def test_wake_up_prefix_no_longer_frames_the_workspace_as_a_place_to_work():
    """§4FJ review C1: the prefix sits ABOVE the tool schemas on every turn."""
    from tests.test_workspace_recognition import (WorkspaceActivity, WorkspaceEvent,
                                                  build_workspace_prefix)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        act = WorkspaceActivity(Path(d))
        act.append(WorkspaceEvent(kind="note", summary="one tracked change"))
        text = build_workspace_prefix(activity=act, state=None)
    assert text, "an empty prefix cannot be checked"
    assert "WHAT'S OUTSIDE OF ME" not in text and "what I'm looking at" not in text
    assert "ACTIVITY LEDGER" in text and "not a place to do work" in text
    assert "### WORKSPACE STATE" in text          # the substring three older pins rely on
