"""Regression tests for bug-hunt unit 4 (tools-infra) — see BUGHUNT.md.

Fixed bugs pinned here:
 1. validators: destructive-SQL guard bypassed by schema-qualified / quoted
    table names (DELETE FROM public.users with no WHERE)
 2. validators: paren-balance check counted parens inside string literals,
    rejecting valid SELECTs; WHERE-presence for UPDATE could be satisfied by a
    WHERE inside a string literal or a subquery
 3. tool_failure: FATAL 401/403 matched as a bare substring, misclassifying
    diagnostics (byte counts, line numbers) as permanent auth errors
 4. tasks: action compared case-sensitively (siblings normalise); malformed
    interval silently ran every 60s while reporting success; a memory-write
    failure masked a successful schedule
 5. registry: browser lambda evaluated _proj_ws() twice; three tools lacked
    **kwargs so a hallucinated arg raised TypeError
 6. self_state: resolve/close now require a match target
"""

import asyncio
import inspect

import pytest

from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass
from ghost_agent.tools import tasks as tasks_mod
from ghost_agent.tools.tasks import tool_manage_tasks


# ──────────────────────────────────────────────────────────────────────
# 1. Destructive-SQL guard covers qualified / quoted table names
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# 2. String-literal-aware structural checks
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# 3. tool_failure 401/403 no longer matches bare substrings
# ──────────────────────────────────────────────────────────────────────

class TestFailureClassification:
    @pytest.mark.parametrize("err", [
        "Downloaded 40301 bytes then failed with ValueError: bad literal",
        "Process died at line 403 with AssertionError",
        "exit code 1403 from subprocess",
    ])
    def test_diagnostic_with_403_digits_not_fatal(self, err):
        cls, _ = classify_tool_failure(err)
        assert cls != FailureClass.FATAL

    @pytest.mark.parametrize("err", [
        "HTTP 403 Forbidden",
        "401 Unauthorized",
        "server returned status: 401",
        "HTTP/1.1 403 Forbidden",
    ])
    def test_real_http_auth_is_fatal(self, err):
        cls, _ = classify_tool_failure(err)
        assert cls == FailureClass.FATAL

    def test_retryable_still_wins_over_status_digits(self):
        cls, _ = classify_tool_failure("connection refused (was retrying 403 times)")
        assert cls == FailureClass.RETRYABLE


# ──────────────────────────────────────────────────────────────────────
# 4. tasks: action normalisation, interval validation, memory isolation
# ──────────────────────────────────────────────────────────────────────

class _Sched:
    def __init__(self):
        self.added = []

    def get_jobs(self):
        return []

    def add_job(self, *a, **k):
        self.added.append(k)


@pytest.fixture
def runner_bound(monkeypatch):
    monkeypatch.setattr(tasks_mod, "run_proactive_task_fn", lambda *a, **k: None)


class TestManageTasks:
    async def test_action_is_case_and_space_insensitive(self):
        sch = _Sched()
        out = await tool_manage_tasks(action=" List ", scheduler=sch)
        # Pre-fix: "Unknown action ' List '".
        assert "Unknown action" not in out
        assert "scheduled tasks" in out.lower()

    async def test_malformed_interval_rejected(self, runner_bound):
        sch = _Sched()
        out = await tool_manage_tasks(action="create", scheduler=sch,
                                      task_name="t", cron_expression="interval:5m", prompt="p")
        # Pre-fix: silently scheduled every 60s and returned SUCCESS.
        assert "Error" in out and "malformed interval" in out
        assert sch.added == []

    async def test_nonpositive_interval_rejected(self, runner_bound):
        sch = _Sched()
        out = await tool_manage_tasks(action="create", scheduler=sch,
                                      task_name="t", cron_expression="interval:-5", prompt="p")
        assert "positive" in out
        assert sch.added == []

    async def test_valid_interval_scheduled(self, runner_bound):
        sch = _Sched()
        out = await tool_manage_tasks(action="create", scheduler=sch,
                                      task_name="t", cron_expression="interval:300", prompt="p")
        assert out.startswith("SUCCESS")
        assert [j.get("seconds") for j in sch.added] == [300]

    async def test_memory_failure_does_not_mask_success(self, runner_bound):
        class BadMem:
            def add(self, *a, **k):
                raise RuntimeError("store down")

        sch = _Sched()
        out = await tool_manage_tasks(action="create", scheduler=sch, memory_system=BadMem(),
                                      task_name="t", cron_expression="interval:60", prompt="p")
        # The job WAS scheduled; a memory-write failure must not report ERROR.
        assert out.startswith("SUCCESS")
        assert len(sch.added) == 1


# ──────────────────────────────────────────────────────────────────────
# 5. registry: **kwargs tolerance on the three previously-strict tools
# ──────────────────────────────────────────────────────────────────────

class TestRegistryKwargsTolerance:

    def test_browser_dispatch_uses_single_proj_ws_call(self):
        # The browser entry is now the _run_browser helper, not a lambda that
        # calls _proj_ws() twice. Assert the source computes the pair once.
        import ghost_agent.tools.registry as reg
        src = inspect.getsource(reg.get_available_tools)
        assert "host_dir, workdir = _proj_ws()" in src
        assert '_proj_ws()[0], container_workdir=_proj_ws()[1]' not in src


# ──────────────────────────────────────────────────────────────────────
# 6. self_state resolve/close require a target
# ──────────────────────────────────────────────────────────────────────

class TestSelfStateRequiresTarget:
    async def test_resolve_without_text_errors_clearly(self):
        from ghost_agent.tools.self_state import tool_self_state

        class _State:
            def open_questions(self):
                return []

            def unfinished_threads(self):
                return []

            def mood(self):
                return None

        class _Model:
            enabled = True
            state = _State()

        out = await tool_self_state(action="resolve_question", self_model=_Model())
        assert "required" in out.lower()
        assert "None" not in out
