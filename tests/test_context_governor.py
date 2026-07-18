"""Context-pressure governor + ceiling enforcement (2026-07-18).

Source incident (xrick feasibility session): ~60 whole-file reads across
turns — each batch cleared its own per-turn ReadBudget while the
conversation grew unboundedly — two compactions, then a "successful"
summarization prune whose kept tail still carried five parallel reads of
generated data files, a 333k-token send against a 262k n_ctx (HTTP 400),
and a recovery retry that reused the streaming payload and read SSE frames
as a "non-JSON body" → dead turn, 25+ minutes lost.

Covers:
  * GhostAgent._cap_oversized_tail (post-prune budget enforcement)
  * occupancy-aware ReadBudget (zero-capacity refusal in tool_read_file)
  * generated/data-shaped file sampling
  * work-log command heads (store + briefing render + dispatch capture)
  * wiring pins: stream-strip on recovery + llm retry, pressure steers,
    lockdown, browser commit-retry, command-not-found hint
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import inspect

import pytest

from ghost_agent.core.agent import GhostAgent, _estimate_messages_tokens
from ghost_agent.tools.file_system import ReadBudget, tool_read_file


# ── _cap_oversized_tail ──────────────────────────────────────────────

def test_cap_oversized_tail_truncates_largest_until_fit(monkeypatch):
    # Pin the estimator: when another test has loaded the real Qwen
    # tokenizer (module-global TOKEN_ENCODER), a 400k single-char run
    # BPE-merges to almost nothing and the fixture isn't over budget.
    import ghost_agent.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "estimate_tokens", lambda t: len(t) // 4)
    big = "x" * 400_000          # ≈100k tokens under len//4
    msgs = [
        {"role": "system", "content": "s" * 8000},
        {"role": "user", "content": "goal"},
        {"role": "tool", "name": "file_system", "content": big},
        {"role": "tool", "name": "file_system", "content": big},
    ]
    out = GhostAgent._cap_oversized_tail(msgs, max_tokens=50_000)
    assert sum(len(str(m.get("content", ""))) // 4 for m in out) <= 50_000
    # system content untouched
    assert out[0]["content"] == "s" * 8000
    # truncation marker present and head+tail retained
    t = out[2]["content"]
    assert "dropped by context budget enforcement" in t
    assert t.startswith("x") and t.endswith("x")


def test_cap_oversized_tail_noop_under_budget():
    msgs = [{"role": "user", "content": "hello"},
            {"role": "tool", "name": "t", "content": "y" * 2000}]
    before = [m["content"] for m in msgs]
    out = GhostAgent._cap_oversized_tail(msgs, max_tokens=100_000)
    assert [m["content"] for m in out] == before


# ── occupancy-aware read budget ──────────────────────────────────────

@pytest.mark.asyncio
async def test_zero_capacity_budget_refuses_first_read_with_pressure_message(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("print('hi')\n" * 2000)
    out = await tool_read_file("big.py", tmp_path, max_context=240000,
                               read_budget=ReadBudget(0))
    assert out.startswith("Error")
    assert "near the context ceiling" in out
    assert "analysis_notes" in out          # externalize-notes guidance
    assert "start_line" in out              # ranged reads stay available


@pytest.mark.asyncio
async def test_first_read_still_passes_with_normal_budget(tmp_path):
    f = tmp_path / "src.py"
    f.write_text("def f():\n    return 1\n" * 500)
    out = await tool_read_file("src.py", tmp_path, max_context=240000,
                               read_budget=ReadBudget(10_000_000))
    assert "CONTENTS" in out


@pytest.mark.asyncio
async def test_ranged_read_exempt_from_zero_budget(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line{i}" for i in range(500)))
    out = await tool_read_file("big.py", tmp_path, max_context=240000,
                               read_budget=ReadBudget(0),
                               start_line=10, end_line=12)
    assert "line10" in out or "line9" in out
    assert not out.startswith("Error")


# ── generated-file sampling ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_hex_data_table_gets_sampled_not_read(tmp_path):
    row = ", ".join("0x%02x" % (i % 256) for i in range(16))
    content = ("static const u8 dat[] = {\n"
               + (f"  {row},\n" * 4000)
               + "};\n")
    assert len(content) > 96 * 1024
    (tmp_path / "dat_tilesPC.c").write_text(content)
    out = await tool_read_file("dat_tilesPC.c", tmp_path, max_context=240000,
                               read_budget=ReadBudget(10_000_000))
    assert "SAMPLE ONLY" in out
    assert "machine-generated" in out
    assert len(out) < 6000


@pytest.mark.asyncio
async def test_large_normal_source_still_reads_whole(tmp_path):
    content = ("def handler_%d(x):\n    return x + %d\n\n" * 1
               ).join("")  # placeholder, built below
    content = "".join(f"def handler_{i}(x):\n    return x + {i}\n\n"
                      for i in range(4000))
    assert len(content) > 96 * 1024
    (tmp_path / "handlers.py").write_text(content)
    out = await tool_read_file("handlers.py", tmp_path, max_context=240000,
                               read_budget=ReadBudget(10_000_000))
    assert "SAMPLE ONLY" not in out
    assert "handler_3999" in out


# ── work-log command heads ───────────────────────────────────────────

def test_add_work_log_stores_bounded_commands(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    sb = tmp_path / "sb"; sb.mkdir()
    store = ProjectStore(tmp_path / "mem", sandbox_root=sb)
    pid = store.create_project("P", kind="CODING")
    store.add_work_log(pid, request="r",
                       commands=["git clone https://x.git repo " + "y" * 200,
                                 "", "ls", "a", "b", "c", "d"])
    p = store.recent_work_logs(pid)[0]["payload"]
    assert len(p["commands"]) <= 5
    assert all(len(c) <= 90 for c in p["commands"])
    assert p["commands"][0].startswith("git clone")


def test_briefing_work_log_falls_back_to_commands(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.core.prompts import build_project_briefing
    sb = tmp_path / "sb"; sb.mkdir()
    store = ProjectStore(tmp_path / "mem", sandbox_root=sb)
    pid = store.create_project("P", kind="CODING")
    store.add_work_log(pid, request="clone the repo", files=[],
                       commands=["git clone https://x.git xrick"],
                       outcome="completed", note="cloned ok")
    briefing = build_project_briefing(store, pid)
    assert "ran: git clone https://x.git xrick" in briefing


# ── wiring pins ──────────────────────────────────────────────────────

def test_recovery_and_llm_retry_strip_stream_flag():
    import ghost_agent.core.agent as agent_mod
    import ghost_agent.core.llm as llm_mod
    hc = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    _recov = hc[hc.index("Emergency pruning triggered"):]
    assert 'payload["stream"] = False' in _recov[:4000]
    lc = inspect.getsource(llm_mod)
    assert "SSE body on a non-streaming call" in lc


def test_pressure_steers_and_lockdown_wired():
    import ghost_agent.core.agent as agent_mod
    hc = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    assert "SYSTEM ALERT (context pressure)" in hc
    assert "SECOND overflow" in hc
    assert "_ctx_pressure_lockdown = True" in hc
    dp = inspect.getsource(agent_mod.GhostAgent._dispatch_and_process_tool_batch)
    assert "_ctx_pressure_lockdown" in dp     # lockdown zeroes the budget
    assert "0.80 * _mc" in dp                 # occupancy-aware shrink


def test_prune_returns_run_through_cap():
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod.GhostAgent._prune_context)
    assert src.count("_cap_oversized_tail") >= 2


def test_browser_commit_retry_and_file_hint():
    import ghost_agent.tools.browser as browser_mod
    src = inspect.getsource(browser_mod)
    assert "wait_until='commit'" in src or 'wait_until="commit"' in src
    from ghost_agent.tools.tool_failure import get_fallback_hint
    hint = get_fallback_hint("execute", "bash: line 1: file: command not found")
    assert hint and "od -An" in hint
