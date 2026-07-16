"""Regressions from the 2026-07-11 (evening) ghost-agent.log audit.

1. WORKER CALLS WERE THINKING. The worker pool runs a reasoning model with
   thinking ON by default. Measured on the live node for the exact
   query-expansion call `route()` makes:
       before : 7.0s, 128/128 tokens burned on hidden reasoning, content == ""
       after  : 0.5s, 5 tokens, correct answer
   So the offload added ~13.7s to the FRONT of every user request (worker call
   at +0.01s; memory bus didn't hydrate until +13.8s), returned nothing usable,
   fell back to the legacy path anyway, and periodically tripped the 15s
   timeout (`Nova: ReadTimeout`). A 14x latency regression that also didn't
   work.

2. "DONE SO FAR (5 of 31)" read as a PROGRESS FRACTION, not a truncation. With
   all 31 tasks actually DONE, the model saw "5 of 31", decided the system
   state was "out of sync with the actual task_list", and burned ~5 turns.

3. report_pdf named the files it SKIPPED but never what existed, so the model
   guessed filenames across THREE PDF regenerations + two sandbox listings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pathlib import Path

import pytest

from ghost_agent.core.llm import _disable_thinking, _ROUTE_TIMEOUT_S


# ══════════════════════════════════════════════════════════════════════
# 1. Worker-routed calls must not think
# ══════════════════════════════════════════════════════════════════════

class TestDisableThinking:
    def test_injects_enable_thinking_false(self):
        p = {"model": "x", "messages": []}
        _disable_thinking(p)
        assert p["chat_template_kwargs"]["enable_thinking"] is False

    def test_preserves_other_template_kwargs(self):
        p = {"chat_template_kwargs": {"preserve_thinking": True}}
        _disable_thinking(p)
        assert p["chat_template_kwargs"]["preserve_thinking"] is True
        assert p["chat_template_kwargs"]["enable_thinking"] is False

    def test_explicit_caller_preference_wins(self):
        # setdefault semantics: a caller that DELIBERATELY wants thinking keeps it.
        p = {"chat_template_kwargs": {"enable_thinking": True}}
        _disable_thinking(p)
        assert p["chat_template_kwargs"]["enable_thinking"] is True

    def test_non_dict_kwargs_replaced_safely(self):
        p = {"chat_template_kwargs": "garbage"}
        _disable_thinking(p)
        assert p["chat_template_kwargs"] == {"enable_thinking": False}

    def test_does_not_mutate_the_callers_nested_dict(self):
        original = {"enable_thinking": True, "k": 1}
        p = {"chat_template_kwargs": original}
        _disable_thinking(p)
        p["chat_template_kwargs"]["k"] = 999
        assert original["k"] == 1        # the caller's dict is untouched

    def test_applied_on_the_worker_dispatch_path(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "llm.py").read_text()
        # Must be applied to the node_payload COPY (so a fallback to the main
        # model keeps the caller's original payload intact). Anchor on the
        # pretty_log CALL, not the bare phrase — a comment elsewhere quotes it.
        worker = src.split('pretty_log("Worker Compute"', 1)[1][:500]
        assert "_disable_thinking(node_payload)" in worker
        assert "node_payload = payload.copy()" in worker

    def test_route_timeout_bounds_the_critical_path(self):
        # route() is awaited BEFORE memory-bus hydration and its fallback is a
        # free string concat — a sick worker must degrade fast, not stall the
        # user for 15s. (12s: absorbs a double-queued call on a small-`-np`
        # worker — see test_worker_warmup.py::TestRouteTimeoutSizing.)
        # Since 2026-07-16 callers with genuinely slow judged tasks (VERIFY)
        # may pass their own budget, but the DEFAULT must stay the routing
        # ceiling — see tests/test_verify_worker_timeout.py for the
        # behavioural pin of both halves.
        assert _ROUTE_TIMEOUT_S <= 12.0
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "llm.py").read_text()
        assert ("timeout=(timeout if timeout is not None "
                "else _ROUTE_TIMEOUT_S)") in src
        assert "timeout=15.0" not in src


# ══════════════════════════════════════════════════════════════════════
# 2. DONE SO FAR must not read as a progress fraction
# ══════════════════════════════════════════════════════════════════════

pytest.importorskip("ghost_agent.memory.projects")
from ghost_agent.memory.projects import ProjectStore          # noqa: E402
from ghost_agent.core.prompts import build_project_briefing  # noqa: E402


def _block(store, pid):
    return build_project_briefing(store, pid) or ""


class TestDoneSoFarLabel:
    def test_truncated_list_states_the_completed_count_first(self, tmp_path):
        store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
        pid = store.create_project("Meta")
        for i in range(31):
            t = store.add_task(pid, f"task {i}")
            store.update_task(t, status="DONE", result_summary=f"r{i}")
        block = _block(store, pid)
        # The old label "(5 of 31, most recent first)" parses as "5 done of 31".
        assert "31 task(s) complete" in block
        assert "showing the" in block
        assert "5 of 31" not in block

    def test_untruncated_list_has_no_confusing_fraction(self, tmp_path):
        store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
        pid = store.create_project("Small")
        for i in range(2):
            t = store.add_task(pid, f"task {i}")
            store.update_task(t, status="DONE")
        block = _block(store, pid)
        assert "2 task(s) complete" in block
        assert "showing the" not in block   # nothing was truncated

    def test_no_done_tasks_omits_the_section(self, tmp_path):
        store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
        pid = store.create_project("Fresh")
        store.add_task(pid, "pending one")
        # NB: the phrase "DONE SO FAR" also appears in the briefing's static
        # guidance text — assert on the rendered HEADER, not the bare phrase.
        assert "DONE SO FAR —" not in _block(store, pid)
        assert "task(s) complete" not in _block(store, pid)


# ══════════════════════════════════════════════════════════════════════
# 3. report_pdf must say what DOES exist when a path misses
# ══════════════════════════════════════════════════════════════════════

from ghost_agent.tools.report_pdf import _available_files_hint  # noqa: E402


class TestAvailableFilesHint:
    def test_lists_real_files_including_subdirs(self, tmp_path):
        (tmp_path / "research").mkdir()
        (tmp_path / "context_boundary.md").write_text("x")
        (tmp_path / "research" / "deep_dive_on_tools.md").write_text("y")
        hint = _available_files_hint(tmp_path)
        assert "context_boundary.md" in hint
        assert "research/deep_dive_on_tools.md" in hint
        assert "do not guess names" in hint

    def test_ignores_binaries_and_dotfiles(self, tmp_path):
        (tmp_path / "a.md").write_text("x")
        (tmp_path / "img.png").write_bytes(b"\x89PNG")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.md").write_text("s")
        hint = _available_files_hint(tmp_path)
        assert "a.md" in hint
        assert "img.png" not in hint
        assert "secret.md" not in hint

    def test_empty_workspace_yields_no_hint(self, tmp_path):
        assert _available_files_hint(tmp_path) == ""

    def test_missing_dir_never_raises(self, tmp_path):
        assert _available_files_hint(tmp_path / "nope") == ""

    def test_truncates_a_huge_listing(self, tmp_path):
        for i in range(80):
            (tmp_path / f"f{i:02d}.md").write_text("x")
        hint = _available_files_hint(tmp_path)
        assert "list truncated" in hint

    def test_wired_into_the_skipped_files_note(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "tools" / "report_pdf.py").read_text()
        assert "_available_files_hint(Path(sandbox_dir))" in src
