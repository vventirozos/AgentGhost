"""Dream-cycle wiring for failure distillation + the project dream pass
(2026-07-19).

`project_dream_pass` was written with a docstring promising a caller in
core/dream.py that never existed (journal: built-but-unwired). Both it
and the new `distill_failure_clusters` now run at the top of every REM
cycle, BEFORE the entropy and idempotency gates — their corpora
(playbook, work_logs, counterfactual results, project events) are
independent of the auto-memory pool, so a thin pool must not starve
them (same rationale as the episodic-consolidation pass).

These tests pin the position and the fail-open guards; behaviour of the
passes themselves is covered by test_failure_distill.py and
test_project_advancer.py.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import inspect

from unittest.mock import MagicMock

from ghost_agent.core.dream import Dreamer
from ghost_agent.core.failure_distill import distill_failure_clusters


class TestDreamWiring:
    def test_both_passes_present_and_before_gates(self):
        src = inspect.getsource(Dreamer.dream)
        assert "distill_failure_clusters" in src
        assert "project_dream_pass" in src
        # positioned ahead of the entropy gate (and therefore also the
        # idempotency gate, which follows it)
        gate = src.index("Not enough entropy")
        assert src.index("distill_failure_clusters") < gate
        assert src.index("project_dream_pass") < gate

    def test_early_return_messages_report_new_passes(self):
        src = inspect.getsource(Dreamer.dream)
        assert src.count("failure-pattern lesson") >= 2
        assert src.count("project digest") >= 2

    async def test_distill_guard_on_mock_context(self):
        # a MagicMock context (every existing dream test) must fall
        # through the _is_real guard without writing anything
        assert await distill_failure_clusters(MagicMock()) == 0

    def test_project_pass_is_magicmock_guarded(self):
        src = inspect.getsource(Dreamer.dream)
        guard = src.index('__module__.startswith("ghost_agent")')
        call = src.index("project_dream_pass, _pstore")  # the invocation
        assert guard < call
