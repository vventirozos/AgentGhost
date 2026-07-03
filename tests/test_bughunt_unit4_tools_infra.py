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

from ghost_agent.tools.validators import validate_sql
from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass
from ghost_agent.tools import tasks as tasks_mod
from ghost_agent.tools.tasks import tool_manage_tasks


# ──────────────────────────────────────────────────────────────────────
# 1. Destructive-SQL guard covers qualified / quoted table names
# ──────────────────────────────────────────────────────────────────────

class TestSqlGuardQualifiedNames:
    @pytest.mark.parametrize("stmt", [
        "DELETE FROM public.users",
        'DELETE FROM "users"',
        'DELETE FROM "public"."users"',
        "DELETE FROM public.users;",
        "delete from analytics.events returning id",
    ])
    def test_unguarded_delete_qualified_blocked(self, stmt):
        ok, reason = validate_sql(stmt)
        assert ok is False
        assert "DELETE" in reason

    @pytest.mark.parametrize("stmt", [
        "UPDATE public.users SET active=false",
        'UPDATE "users" SET active=false',
    ])
    def test_unguarded_update_qualified_blocked(self, stmt):
        ok, reason = validate_sql(stmt)
        assert ok is False
        assert "UPDATE" in reason

    @pytest.mark.parametrize("stmt", [
        "DELETE FROM public.users WHERE id=1",
        "UPDATE public.users SET active=false WHERE id=1",
    ])
    def test_qualified_with_where_pass(self, stmt):
        ok, _ = validate_sql(stmt)
        assert ok is True


# ──────────────────────────────────────────────────────────────────────
# 2. String-literal-aware structural checks
# ──────────────────────────────────────────────────────────────────────

class TestSqlStringAwareChecks:
    @pytest.mark.parametrize("stmt", [
        "SELECT * FROM t WHERE note = 'a)'",
        "SELECT * FROM t WHERE note = 'open ( paren'",
    ])
    def test_paren_inside_string_not_flagged(self, stmt):
        # Pre-fix: rejected as "unbalanced parentheses".
        ok, reason = validate_sql(stmt)
        assert ok is True, reason

    def test_where_inside_string_does_not_satisfy_update_guard(self):
        # The WHERE lives only inside a string literal → still unguarded.
        ok, reason = validate_sql("UPDATE t SET note='no where here'")
        assert ok is False
        assert "UPDATE" in reason

    def test_where_inside_subquery_does_not_satisfy_update_guard(self):
        ok, reason = validate_sql("UPDATE t SET x=(SELECT a FROM b WHERE c=1)")
        assert ok is False
        assert "UPDATE" in reason

    def test_real_outer_where_with_subquery_passes(self):
        ok, _ = validate_sql("UPDATE t SET x=1 WHERE id IN (SELECT id FROM b WHERE c=1)")
        assert ok is True

    def test_genuinely_unbalanced_parens_still_flagged(self):
        ok, reason = validate_sql("SELECT (a + b FROM t")
        assert ok is False
        assert "paren" in reason.lower()


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
    def test_three_tools_accept_var_keyword(self):
        from ghost_agent.tools.acquired_skills import tool_manage_skills
        from ghost_agent.tools.memory import tool_self_play_loop, tool_list_lessons
        for fn in (tool_manage_skills, tool_self_play_loop, tool_list_lessons):
            kinds = [p.kind for p in inspect.signature(fn).parameters.values()]
            assert inspect.Parameter.VAR_KEYWORD in kinds, fn.__name__

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
