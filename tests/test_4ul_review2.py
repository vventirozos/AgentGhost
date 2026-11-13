"""§4UL projects RE-review — the fixes, each pinned by behaviour (round 3: the gate SIMPLIFIED — it guards an
honest model's omission, never an adversary):

M1  an unattended DONE needs a CHECK that RAN after the last WRITE: any allow-listed program (tests, a build, the
    program, a linter, the local server, `lab http`, the browser) anywhere in the line — exit status, `|| true`,
    `| tail`, loop / if bodies do not matter. A here-doc body, a look, an install is no check. A WRITE is a
    mutating file_system operation or an `execute` that writes (redirect, `sed -i`, `perl -pi`, `rm`, a formatter,
    inline code that writes files); prose (.md / .txt) never voids a check. Order inside one command counts.
M2  the turn's scope reads ids the way the handler does: case-mangled, a JSON-list string, a bare `task_ids`.
m1  another project is only READ (no research, no describe_file / ledger writes); task_add / task_decompose aim
    only at the turn's task.
m2  a refused DONE is held on the task it TARGETED (a split parent is never parked) on every exit path, cancel too.
round 3 E/F/G  a declared `write` task needs no run; the refusal says what counts; a close before a timeout is
    reported as the close; a hold keeps the task's own files.
round 4  no replay: a check after the refusal with no new close is held honestly; a later FAILED / the owner's
    status stands; each id by its own verdict; an idle turn's last DONE fires the project-done hook.
round 5  only a terminal status ends a hold; subtasks left claimed go back; the day is charged exactly once (from
    `AdvanceResult.worked`); holds land before the charges; a failed tail releases the task.
round 6  only subtasks THIS turn claimed are released; steps that ran nothing say so; a model-server abort is scored
    (rest, not spin) unless the owner took the slot; a task set back to PENDING keeps its hold; the owner's close
    is not the turn's.
m3  `_shell_write_targets` names curl -o, wget -O, dd of=, ln, cp -t, a `sudo -u x` prefix, `)>` / `;>`, and
    /workspace/… paths.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ghost_agent.core import project_turns as PT
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.utils.logging import request_id_context
from tests.test_4ul_autopilot_turns import _FakeTools, _advance, _ctx, _executor, _fake_turn, _project

OK = ToolOutcome.ok("EXIT CODE: 0\nSTDOUT/STDERR:\n")


@pytest.fixture
def turns_on(monkeypatch):
    monkeypatch.setenv("GHOST_PROJECT_TASK_TURNS", "1")


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _after_write(name, kwargs, res=OK):
    w = PT._TurnWatch()
    w.note("file_system", {"operation": "write"}, ToolOutcome.ok("w"))
    w.note(name, kwargs, res)
    return w


# ── M1: what is a check ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "env python a.py", "python -m pytest -q", "ls && pytest", "cd x && npm test",
    "python -m pytest 2>&1 | tail -20", "pytest || exit 1", "sudo -u www python app.py", "timeout 60 node s.js",
    "./run_tests.sh", "bash -c 'pytest -q'", "uv run pytest", "go test ./...", "cargo test", "make",
    "(cd app && pytest)", "python - <<'EOF'\nimport app\nEOF", "curl -s http://localhost:8000/",
    "python app.py & sleep 2; curl 127.0.0.1:8000", "pytest > out.txt", "python app.py 2>&1 | tee run.log",
    "npm run build && npm test", "docker exec -it web pytest", "psql -c 'select 1'",
    # round 3 A: what an honest model writes — none of these was a check before
    "pytest || true", "pytest; echo $?", "pytest && echo PASS || echo FAIL", "pytest &",
    "for f in tests/*.py; do python $f; done", "if pytest -q; then echo ok; fi", 'echo "$(pytest -q)"',
    "x=$(python app.py)", "mypy src", "ruff check .", "python -m mypy app", "eslint .", "npx eslint .",
    "shellcheck run.sh", "htmlhint index.html", "tsc --noEmit", "node --check app.js", "php -l index.php",
    "black --check app.py",
    # round 3 B: continuation lines
    "pytest \\\n  -q tests", "ls &&\npytest", "cat log |\npython parse.py",
])
def test_a_run_that_checks_the_build_counts(cmd):
    assert _after_write("execute", {"command": cmd}).verified, cmd


@pytest.mark.parametrize("cmd", [
    "cat > notes.md <<'EOF'\npytest\nnpm test\nEOF",          # a here-doc body is data, not commands
    "ls -la", "cat a.py | head", "echo pytest", "true", "pip install x", "python --version", "npm install",
    "python -m pip install -r requirements.txt", "python -m http.server 8000", "cp a.py b.py", "chmod +x run.sh",
    "curl https://example.com", "git status", "mkdir -p build", "bash -c 'ls'", "playwright install chromium",
    "npx playwright install", "python -m playwright install",
])
def test_a_command_that_checks_nothing_does_not_count(cmd):
    assert not _after_write("execute", {"command": cmd}).verified, cmd


def test_a_failed_run_still_ran():
    fail = ToolOutcome("EXIT CODE: 1\nSTDOUT/STDERR:\nboom", status=OutcomeStatus.FAILED)
    assert _after_write("execute", {"command": "pytest"}, fail).verified
    refused = ToolOutcome.rejected("Error: the sandbox refused it", world_changed=False)
    assert not _after_write("execute", {"command": "pytest"}, refused).verified


@pytest.mark.parametrize("cmd", [
    "sed -i 's/a/b/' a.py", "echo x > a.py", "rm a.py", "git checkout -- a.py", "patch -p1 < fix.diff",
    "cat > a.py <<'EOF'\nprint(1)\nEOF", "cp /tmp/x.py a.py", "curl -o a.html https://example.com/a.html",
    "bash <<'EOF'\ncat > a.py <<'X'\nprint(1)\nX\nEOF", "perl -pi -e 's/a/b/' a.py", "ruby -i -pe 'x' a.rb",
    "black app.py", "ruff format .", "prettier --write src", "python -m black .",
    "python -c \"open('a.py','w').write('x')\"",
    "python - <<'EOF'\nfrom pathlib import Path\nPath('a.py').write_text('x')\nEOF",
])
def test_an_execute_that_writes_voids_the_earlier_check(cmd):
    w = PT._TurnWatch()
    w.note("execute", {"command": "pytest -q"}, OK)
    assert w.verified
    w.note("execute", {"command": cmd}, OK)
    assert not w.verified, cmd


def test_inline_code_that_writes_files_is_a_write_not_a_check():
    w = _after_write("execute", {"content": "open('index.html', 'w').write('<p>x</p>')"})
    assert not w.verified
    assert _after_write("execute", {"content": "import app; print(app.total())"}).verified


@pytest.mark.parametrize("call", [
    ("file_system", {"operation": "write", "path": "NOTES.md"}),
    ("file_system", {"operation": "edit", "path": "docs/usage.rst"}),
    ("execute", {"command": "echo done >> CHANGELOG.txt"}),
    ("execute", {"command": "cat > README.md <<'EOF'\nhow to run\nEOF"}),
])
def test_prose_written_after_a_check_does_not_void_it(call):
    w = PT._TurnWatch()
    w.note("execute", {"command": "pytest"}, OK)
    w.note(call[0], call[1], OK)
    assert w.verified, call


def test_inside_one_command_the_order_counts():
    assert PT.shell_check_events("echo x > a.py && python a.py") == ["w", "c"]
    assert PT.shell_check_events("pytest && sed -i s/a/b/ a.py") == ["c", "w"]
    assert PT.shell_check_events("eslint --fix .") == ["c", "w"]
    assert _after_write("execute", {"command": "echo x > a.py && python a.py"}).verified
    assert not _after_write("execute", {"command": "python a.py && echo x > a.py"}).verified


@pytest.mark.parametrize("op", ["write", "edit", "replace", "append", "append_file", "write_append", "copy",
                                "download", "move", "rename", "delete", "remove", "rm"])
def test_every_mutating_file_system_operation_is_a_write(op):
    w = PT._TurnWatch()
    w.note("execute", {"command": "pytest"}, OK)
    w.note("file_system", {"operation": op, "path": "app.py"}, ToolOutcome.ok("done"))
    assert not w.verified, op


def test_a_read_a_refused_write_or_a_failed_write_is_not_a_write_and_a_partial_one_is():
    w = PT._TurnWatch()
    w.note("execute", {"command": "pytest"}, OK)
    for op in ("read", "search", "list_files", "outline", "quarantine_list"):
        w.note("file_system", {"operation": op}, ToolOutcome.ok("x"))
    w.note("file_system", {"operation": "write"}, ToolOutcome.rejected("Error: refused", world_changed=False))
    w.note("file_system", {"operation": "edit"}, "Error: old_string not found")
    assert w.verified
    w.note("file_system", {"operation": "edit"}, ToolOutcome("kept; syntax error", status=OutcomeStatus.PARTIAL))
    assert not w.verified


def test_execute_of_a_file_or_inline_code_and_the_other_tools():
    w = _after_write("execute", {"filename": "a.py"})
    assert w.verified                                              # ran the existing file
    w = PT._TurnWatch()
    w.note("execute", {"command": "pytest"}, OK)
    w.note("execute", {"filename": "a.py", "content": "print(1)"}, OK)
    assert w.verified and w.last_write > 0                         # saved, then ran: the run is after its write
    w.note("execute", {"filename": "a.py", "content": "x"}, ToolOutcome.rejected("refused", world_changed=False))
    assert w.verified                                              # refused before it ran: no write
    assert not _after_write("lab", {"action": "exec", "cmd": "ls /srv"}).verified
    assert _after_write("lab", {"action": "exec", "cmd": "curl -s localhost:8080/health"}).verified
    w = PT._TurnWatch()
    w.note("lab", {"action": "exec", "cmd": "pytest"}, OK)
    w.note("lab", {"action": "exec", "cmd": "echo x > /srv/app/cfg"}, OK)   # the lab's file, not the workspace
    assert w.verified
    assert _after_write("lab", {"action": "http", "path": "/"}, ToolOutcome("HTTP 500", status=OutcomeStatus.FAILED)
                        ).verified                                 # any response
    assert _after_write("lab", {"action": "screenshot"}, ToolOutcome.ok("png")).verified
    assert not _after_write("lab", {"action": "http"}, ToolOutcome.rejected("LAB ERROR (not_allowed): no")).verified
    assert _after_write("run", {"action": "collect"}).verified
    assert _after_write("run", {"action": "logs"}, ToolOutcome.ok("log lines")).verified
    assert not _after_write("run", {"action": "list"}).verified
    assert _after_write("browser", {"operation": "screenshot", "url": "http://localhost:8080"},
                        ToolOutcome.ok("png")).verified
    assert not _after_write("browser", {"operation": "close"}, ToolOutcome.ok("closed")).verified


async def test_the_reviewers_heredoc_bypass_no_longer_closes_a_task(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")
    seen = []

    async def go(agent, kw):
        t = agent.available_tools
        await t["file_system"](operation="write", path="a.py", content="x")
        await t["execute"](command="cat > notes.txt <<'EOF'\npython -m pytest\nEOF")
        kw["_out"] = await t["manage_projects"](action="task_update", task_id=tid, status="DONE", result="ok")
        return "done"
    _fake_turn(monkeypatch, go, seen)
    await _advance(_ctx(store), pid, owner=False)
    out = seen[0]["_out"]
    assert out.reason_code == "task_turn_unverified_done"
    assert "lab http" in str(out) and "execute" in str(out)        # round 3 F: it says what counts
    assert store.get_task(tid)["status"] == "NEEDS_USER"


# ── M2 + m1: the turn's scope ───────────────────────────────────────────────

async def test_ids_are_read_the_way_the_handler_reads_them(store):
    pid, tid = _project(store)
    other = store.add_task(pid, "Another task")
    sub = store.add_task(pid, "A part of it", parent_id=tid)
    calls = []
    agent = _FakeTools(calls)
    PT._prepare(PT._TurnWatch(pid, tid, store=store))(agent)
    mp = agent.available_tools["manage_projects"]
    for kw in ({"task_id": tid.upper()}, {"task_id": f" {sub.upper()} "}, {"task_ids": tid},
               {"task_ids": f'["{tid.upper()}", "{sub}"]'}, {"task_id": f'["{tid}"]'},
               {"task_id": tid, "project_id": pid.upper()}):
        assert (await mp(action="task_update", status="FAILED", **kw)).status is OutcomeStatus.OK, kw
    for kw in ({"task_ids": other}, {"task_ids": f"{tid},{other}"}, {"task_id": f'["{tid}", "{other}"]'},
               {"task_id": other.upper()}):
        assert (await mp(action="task_update", status="DONE", **kw)).reason_code == "task_turn_scope", kw


async def test_another_project_is_only_read_and_tasks_are_added_only_under_this_one(store):
    pid, tid = _project(store)
    other = store.add_task(pid, "Another task")
    sub = store.add_task(pid, "A part of it", parent_id=tid)
    calls = []
    agent = _FakeTools(calls)
    PT._prepare(PT._TurnWatch(pid, tid, store=store))(agent)
    mp = agent.available_tools["manage_projects"]
    refused = ({"action": "research", "project_id": "elsewhere", "topic": "x"},
               {"action": "describe_file", "project_id": "elsewhere", "file_path": "a", "description": "d"},
               {"action": "ledger", "project_id": "elsewhere", "ledger": "a fact"},
               {"action": "config", "project_id": "elsewhere", "config_key": "k", "config_value": "v"},
               {"action": "task_add", "description": "x", "parent_id": other},
               {"action": "task_add", "description": "x", "parent_id": other.upper()},
               {"action": "task_decompose", "subtasks": ["a", "b"]},
               {"action": "task_decompose", "subtasks": ["a", "b"], "task_id": pid},
               {"action": "task_decompose", "subtasks": ["a", "b"], "task_id": other})
    for kw in refused:
        assert (await mp(**kw)).reason_code == "task_turn_scope", kw
    assert calls == []
    allowed = ({"action": "describe_file", "project_id": "elsewhere"},
               {"action": "ledger", "project_id": "ELSEWHERE"},
               {"action": "research_list", "project_id": "elsewhere"},
               {"action": "research", "topic": "x"},                       # its own project
               {"action": "task_add", "description": "a part", "parent_id": tid.upper()},
               {"action": "task_add", "description": "a part of a part", "parent_id": sub},
               {"action": "task_add", "description": "a follow-up"},
               {"action": "task_decompose", "subtasks": ["a", "b"], "task_id": tid})
    for kw in allowed:
        assert (await mp(**kw)).status is OutcomeStatus.OK, kw
    assert len(calls) == len(allowed)


# ── m2 + round 3 D/E/G: holds, on the task the DONE targeted ───────────────

def _turn(monkeypatch, store, script, seen):
    """`_fake_turn` with a manage_projects that really writes the status a task_update carries (as the handler
    reads it: WAITING → NEEDS_USER, RUNNING → IN_PROGRESS)."""
    from ghost_agent.core import coding_loop as cl

    async def fake(context, **kw):
        seen.append(kw)
        calls = []
        agent = _FakeTools(calls)

        async def mp(**k):
            calls.append(("manage_projects", k))
            st = PT._norm_status(k.get("status"))
            if k.get("action") == "task_update" and st:
                for t in PT._task_ids(k):
                    store.update_task(t, status=st, result_summary=k.get("result") or "")
            return ToolOutcome.ok("updated")
        agent.base["manage_projects"] = mp
        agent.available_tools = dict(agent.base)
        kw["prepare_agent"](agent)
        kw["_calls"] = calls
        return await script(agent, kw)
    monkeypatch.setattr(cl, "run_leaf_turn", fake)


async def _write_then_done(agent, kw, task_id, path="a.py"):
    t = agent.available_tools
    await t["file_system"](operation="write", path=path, content="x")
    kw["_out"] = await t["manage_projects"](action="task_update", task_id=task_id, status="DONE",
                                            result="built it", deliverables=[path])
    return kw["_out"]


async def test_a_refused_done_holds_the_child_it_targeted_never_the_split_parent(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build the app → app.py")
    seen, kids = [], {}

    async def go(agent, kw):
        kids["c1"] = store.add_task(pid, "Build part one → one.py", parent_id=tid)
        kids["c2"] = store.add_task(pid, "Build part two → two.py", parent_id=tid)
        m = dict(store.get_project(pid).get("metadata") or {})
        m["independent_tasks"] = [kids["c2"]]
        store.update_project(pid, metadata=m)
        await _write_then_done(agent, kw, kids["c1"], "one.py")
        return "split it and built part one"
    _turn(monkeypatch, store, go, seen)
    res = await _advance(_ctx(store), pid, owner=False)
    assert seen[0]["_out"].reason_code == "task_turn_unverified_done"
    c1 = store.get_task(kids["c1"])
    assert c1["status"] == "NEEDS_USER" and c1["wait_code"] == "unverified_build"
    assert store.get_task(tid)["status"] != "NEEDS_USER" and res.classification != "needs_user"
    from ghost_agent.core.planning import ProjectPlan
    nxt = ProjectPlan(store, pid).next_ready_leaf()
    assert nxt is not None and nxt.id == kids["c2"]                # the sibling can still run


async def test_an_unchecked_done_on_its_own_task_is_held_with_its_files(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")
    seen = []

    async def go(agent, kw):
        await agent.available_tools["file_system"](operation="write", path="scratch/tmp.py", content="x")
        await _write_then_done(agent, kw, tid)
        return "built a.py"
    _turn(monkeypatch, store, go, seen)
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "unverified_build" and res.classification == \
        "needs_user"
    ev = store.latest_task_event(pid, tid, ("autoadvance_needs_user",))
    assert ev["payload"]["files"] == ["a.py"]                      # the owner's run re-checks them
    assert [a.get("payload") for a in store.list_artifacts(task_id=tid) if a.get("kind") == "file"] == ["a.py"]


async def test_an_unchecked_done_is_held_when_the_turn_is_cancelled(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        raise asyncio.CancelledError()
    _turn(monkeypatch, store, go, [])
    with pytest.raises(asyncio.CancelledError):
        await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "unverified_build"


async def test_a_check_after_the_refusal_without_a_new_close_is_held_honestly(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")
    seen = []

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        await agent.available_tools["execute"](command="python a.py")    # ran it, never closed again
        return "ran it"
    _turn(monkeypatch, store, go, seen)
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and res.classification == "needs_user"   # round 4: never replayed
    assert "never closed it again" in res.summary
    ev = store.latest_task_event(pid, tid, ("autoadvance_needs_user",))
    assert ev["payload"]["hold"] == ""                             # the owner's run re-judges it (no file re-check)
    closes = [k for n, k in seen[0]["_calls"] if n == "manage_projects" and k.get("status") == "DONE"]
    assert closes == []


async def test_a_failed_close_after_a_refused_done_stays_failed(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        await agent.available_tools["execute"](
            command="python a.py", _res=ToolOutcome("EXIT CODE: 1\nTraceback boom", status=OutcomeStatus.FAILED))
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="FAILED",
                                                       result="tests fail, cannot fix")
        return "failed it"
    _turn(monkeypatch, store, go, [])
    res = await _advance(_ctx(store), pid, owner=False)
    assert store.get_task(tid)["status"] == "FAILED" and res.classification != "needs_user"


async def test_a_status_the_owner_set_during_the_turn_is_kept_and_reported_as_theirs(turns_on, monkeypatch,
                                                                                     store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        store.update_task(tid, status="PAUSED")                  # the owner's console, mid-turn
        return "built it"
    _turn(monkeypatch, store, go, [])
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)                                       # PAUSED is stored as the owner's wait
    assert row["status"] == "NEEDS_USER" and row["wait_on"] == "owner" and row["wait_code"] != "unverified_build"
    assert res.summary == "task is NEEDS_USER (set during its turn)"


async def test_a_multi_id_refusal_holds_each_task_by_its_own_verdict(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build app → app.py")
    kids = {}

    async def go(agent, kw):
        kids["b"] = store.add_task(pid, "Build b → b.py", parent_id=tid)
        kids["w"] = store.add_task(pid, "Write the notes → NOTES.md", parent_id=tid, kind="write")
        t = agent.available_tools
        await t["file_system"](operation="write", path="b.py", content="x")
        kw["_out"] = await t["manage_projects"](action="task_update", task_ids=[kids["b"], kids["w"]],
                                                status="DONE", result="both")
        return "done"
    seen = []
    _turn(monkeypatch, store, go, seen)
    await _advance(_ctx(store), pid, owner=False)
    assert seen[0]["_out"].reason_code == "task_turn_unverified_done"
    assert store.get_task(kids["b"])["status"] == "NEEDS_USER"
    assert store.get_task(kids["w"])["status"] == "PENDING"        # its own verdict passed: not held as unverified


async def _unattended_step(store, pid):
    """One step through autopilot's own wrapper (which books the daily allowance)."""
    from ghost_agent.core.project_advancer import advance_unattended
    from tests.test_4ul_autopilot_turns import _executor
    m = dict(store.get_project(pid).get("metadata") or {})
    m["autopilot"] = True
    store.update_project(pid, metadata=m)
    return await advance_unattended(_ctx(store), pid, tool_runner=lambda *a, **k: None, coding_executor=_executor())


def _steps_today(store, pid):
    return int((store.get_project(pid).get("metadata") or {}).get("unattended_steps_today") or 0)


async def test_a_turn_that_worked_and_then_lost_the_slot_spends_the_day_once(turns_on, monkeypatch, store):
    from ghost_agent.core.llm import BackgroundDeferred
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await agent.available_tools["file_system"](operation="write", path="a.py", content="x")
        raise BackgroundDeferred("owner")
    _turn(monkeypatch, store, go, [])
    res = await _unattended_step(store, pid)
    assert res.classification == "deferred" and _steps_today(store, pid) == 1


async def test_a_deferral_before_any_call_spends_nothing(turns_on, monkeypatch, store):
    from ghost_agent.core.llm import BackgroundDeferred
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        raise BackgroundDeferred("owner")
    _turn(monkeypatch, store, go, [])
    await _unattended_step(store, pid)
    assert _steps_today(store, pid) == 0


@pytest.mark.parametrize("then", ["deferred", "aborted"])
async def test_a_close_or_a_hold_then_a_lost_slot_is_charged_exactly_once(turns_on, monkeypatch, store, then):
    from ghost_agent.core.llm import BackgroundDeferred
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        t = agent.available_tools
        await t["file_system"](operation="write", path="a.py", content="x")
        if then == "deferred":                                     # DONE (checked), then the owner takes the slot
            await t["execute"](command="python a.py")
            await t["manage_projects"](action="task_update", task_id=tid, status="DONE", result="built and ran")
            raise BackgroundDeferred("owner")
        await t["manage_projects"](action="task_update", task_id=tid, status="DONE", result="built")   # held
        return "[ATTEMPT_ABORTED_UPSTREAM] owner"
    _turn(monkeypatch, store, go, [])
    await _unattended_step(store, pid)
    assert store.get_task(tid)["status"] == ("DONE" if then == "deferred" else "NEEDS_USER")
    assert _steps_today(store, pid) == 1


@pytest.mark.parametrize("later", ["IN_PROGRESS", "RUNNING", "PENDING"])
async def test_a_non_terminal_update_after_a_refused_done_keeps_the_hold(turns_on, monkeypatch, store, later):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status=later,
                                                       result="needs a check by the owner")
        store.update_task(tid, status="IN_PROGRESS")               # (the fake handler wrote nothing for these)
        return "left for the owner"
    _turn(monkeypatch, store, go, [])
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "unverified_build" and res.classification == \
        "needs_user"


async def test_a_refused_subtask_set_running_again_is_still_held(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")
    kids = {}

    async def go(agent, kw):
        kids["b"] = store.add_task(pid, "Build b → b.py", parent_id=tid)
        await _write_then_done(agent, kw, kids["b"], "b.py")
        await agent.available_tools["manage_projects"](action="task_update", task_id=kids["b"], status="RUNNING")
        store.update_task(kids["b"], status="IN_PROGRESS")
        return "left"
    _turn(monkeypatch, store, go, [])
    await _advance(_ctx(store), pid, owner=False)
    assert store.get_task(kids["b"])["status"] == "NEEDS_USER"


async def test_a_waiting_update_after_a_refused_done_ends_the_hold(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="WAITING",
                                                       result="needs the vendor key")
        store.update_task(tid, status="NEEDS_USER", wait_on="owner", wait_code="asked")
        return "waiting"
    _turn(monkeypatch, store, go, [])
    await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "asked"        # the turn's own wait, not a hold


@pytest.mark.parametrize("ending", ["normal", "cancel"])
async def test_a_subtask_the_turn_claimed_and_left_goes_back(turns_on, monkeypatch, store, ending):
    pid, tid = _project(store, "Build the app → app.py")
    kids = {}

    async def go(agent, kw):
        kids["a"] = store.add_task(pid, "Build a → a.py", parent_id=tid)
        await agent.available_tools["manage_projects"](action="task_update", task_id=kids["a"],
                                                       status="IN_PROGRESS")   # the turn started on it
        if ending == "cancel":
            raise asyncio.CancelledError()
        return "ran out of ideas"
    _turn(monkeypatch, store, go, [])
    if ending == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await _advance(_ctx(store), pid, owner=False)
    else:
        await _advance(_ctx(store), pid, owner=False)
    assert store.get_task(kids["a"])["status"] == "PENDING"


async def test_a_subtask_another_run_claimed_is_left_to_that_run(turns_on, monkeypatch, store):
    from tests.test_4ul_autopilot_turns import _fake_turn
    pid, tid = _project(store)
    box = {}

    async def script(agent, kw):
        await agent.available_tools["manage_projects"](action="task_add", parent_id=tid,
                                                       description="sub part → sub.py")
        box["sid"] = store.add_task(pid, "sub part → sub.py", parent_id=tid)
        store.update_task(box["sid"], status="IN_PROGRESS")        # the owner's batch claims it meanwhile
        return "working on it"
    _fake_turn(monkeypatch, script, [])
    await _advance(_ctx(store), pid, owner=True)
    assert store.get_task(box["sid"])["status"] == "IN_PROGRESS"   # never released under the other run


async def test_steps_that_ran_nothing_are_marked_as_no_work(turns_on, monkeypatch, store):
    pid, tid = _project(store)
    monkeypatch.setattr(PT, "_time_for_turn", lambda: 10.0)
    res = await _advance(_ctx(store), pid, owner=False)
    assert res.worked is False and "out of time" in res.summary
    monkeypatch.setattr(PT, "_time_for_turn", lambda: 900.0)
    store.update_task(tid, status="PENDING")
    res = await PT.run_task_turn_for_advance(_ctx(store, fg=1), store, pid,
                                             SimpleNamespace(id=tid, description="Build", kind=""), "coding",
                                             False, idle=True)
    assert res.classification == "deferred" and res.worked is False


async def test_a_model_server_abort_is_not_a_free_retry(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        return "[ATTEMPT_ABORTED_UPSTREAM] the model server returned 503"
    _turn(monkeypatch, store, go, [])
    for _ in range(3):
        res = await _unattended_step(store, pid)
    assert res.classification == "blocked" and "model server failed" in res.summary
    md = store.get_project(pid)["metadata"]
    assert int(md.get("no_progress_streak") or 0) >= 3 or md.get("autopilot_rest")   # rests, never spins
    assert _steps_today(store, pid) == 0                            # nothing ran: no day spent

    pid2, tid2 = _project(store, "Build b → b.py")                 # the owner's request did take the slot
    from ghost_agent.core.project_advancer import advance_once
    res = await advance_once(_ctx(store, fg=1), pid2, owner_requested=False, tool_runner=lambda *a, **k: None,
                             coding_executor=__import__("tests.test_4ul_autopilot_turns",
                                                        fromlist=["_executor"])._executor())
    assert res.classification == "deferred"


async def test_a_turn_setting_its_task_back_to_pending_after_a_refused_done_keeps_the_hold(turns_on,
                                                                                         monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="PENDING")
        return "left it"
    _turn(monkeypatch, store, go, [])
    res = await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_code"] == "unverified_build" and res.classification == \
        "needs_user"


async def test_a_parent_closed_by_the_owner_is_not_the_turns_close(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build the app → app.py")

    async def go(agent, kw):
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="IN_PROGRESS")
        store.update_task(tid, status="FAILED", failure_reason="the owner gave up on it")   # the console
        return "was working"
    _turn(monkeypatch, store, go, [])
    res = await _advance(_ctx(store), pid, owner=False)
    assert res.summary == "task is FAILED (set during its turn)"


async def test_holds_land_before_the_charges_and_a_failed_tail_releases_the_task(turns_on, monkeypatch, store):
    from ghost_agent.core import project_advancer as PA
    pid, tid = _project(store, "Build a → a.py")

    def boom(*a, **k):
        raise RuntimeError("db locked")
    monkeypatch.setattr(PA, "_increment_budget", boom)

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid)
        return "built"
    _turn(monkeypatch, store, go, [])
    await _advance(_ctx(store), pid, owner=False)
    assert store.get_task(tid)["status"] == "NEEDS_USER"           # the hold landed first

    pid2, tid2 = _project(store, "Build b → b.py")
    from ghost_agent.core.planning import ProjectPlan
    _orig = ProjectPlan.update_status

    def fail_on_failed(self, task_id, status, *a, **k):
        if str(getattr(status, "value", status)).upper() == "FAILED":
            raise RuntimeError("db locked")
        return _orig(self, task_id, status, *a, **k)
    monkeypatch.setattr(ProjectPlan, "update_status", fail_on_failed)

    async def go2(agent, kw):
        return "never closed it"
    _turn(monkeypatch, store, go2, [])
    res = await _advance(_ctx(store), pid2, owner=False)
    assert store.get_task(tid2)["status"] == "PENDING" and "not recorded" in res.summary


async def test_the_last_task_closed_in_an_idle_turn_fires_the_project_done_hook(turns_on, monkeypatch, store):
    from ghost_agent.core import coding_loop as cl
    from ghost_agent.core.planning import ProjectPlan, TaskStatus
    from ghost_agent.utils.logging import request_id_context
    pid, tid = _project(store, "Build a → a.py")
    fired = []
    store.on_project_done = lambda p: fired.append(p)

    async def fake(context, **kw):
        agent = _FakeTools([])

        async def mp(**k):
            if k.get("action") == "task_update" and str(k.get("status")).upper() == "DONE":
                ProjectPlan(store, pid).update_status(tid, TaskStatus.DONE, result="built and ran a.py")
            return ToolOutcome.ok("updated")
        agent.base["manage_projects"] = mp
        agent.available_tools = dict(agent.base)
        kw["prepare_agent"](agent)
        t = agent.available_tools
        tok = request_id_context.set(kw["request_id"])
        try:                                                       # no handle_chat around it: nothing flushes
            await t["file_system"](operation="write", path="a.py", content="x")
            await t["execute"](command="python a.py")
            await t["manage_projects"](action="task_update", task_id=tid, status="DONE", result="b")
        finally:
            request_id_context.reset(tok)
        return "done"
    monkeypatch.setattr(cl, "run_leaf_turn", fake)
    await _advance(_ctx(store), pid, owner=False)
    assert store.get_task(tid)["status"] == "DONE" and fired == [pid]


async def test_a_close_before_a_timeout_is_reported_as_the_close(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        t = agent.available_tools
        await t["file_system"](operation="write", path="a.py", content="x")
        await t["execute"](command="python a.py")
        kw["_out"] = await t["manage_projects"](action="task_update", task_id=tid, status="DONE",
                                                result="built and ran a.py")
        raise asyncio.TimeoutError()
    seen = []
    _turn(monkeypatch, store, go, seen)
    res = await _advance(_ctx(store), pid, owner=False)
    assert seen[0]["_out"].status is OutcomeStatus.OK
    assert store.get_task(tid)["status"] == "DONE"
    assert "out of time" not in res.summary and "built and ran" in res.summary


async def test_a_declared_write_task_closes_without_a_run(turns_on, monkeypatch, store):
    pid = store.create_project("Docs", kind="CODING", goal="docs")
    tid = store.add_task(pid, "Write the user guide → GUIDE.md", kind="write")
    seen = []

    async def go(agent, kw):
        await _write_then_done(agent, kw, tid, "GUIDE.md")
        return "wrote it"
    _turn(monkeypatch, store, go, seen)
    await _advance(_ctx(store), pid, owner=False)
    assert seen[0]["_out"].status is OutcomeStatus.OK
    assert store.get_task(tid)["status"] == "DONE"


def test_the_wrapped_tools_keep_their_marks_and_names():
    async def manage_projects(**kw):
        return ToolOutcome.ok("x")
    manage_projects._ghost_real_executor = True
    out = PT._wrap_tools({"manage_projects": manage_projects}, PT._TurnWatch())
    w = out["manage_projects"]
    assert w.__name__ == "manage_projects" and w._ghost_real_executor is True and w._ghost_task_turn_wrapped


# ── m3: write targets ───────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd, want", [
    ("curl -sSo projects/abc/page.html http://localhost:8000", ["projects/abc/page.html"]),
    ("curl --output=projects/abc/p2 http://x", ["projects/abc/p2"]),
    ("wget -O projects/abc/z http://x", ["projects/abc/z"]),
    ("dd if=/dev/zero of=projects/abc/blob bs=1 count=1", ["projects/abc/blob"]),
    ("ln -s ../shared projects/abc/link", ["projects/abc/link"]),
    ("sudo -u www cp x.py projects/abc/y.py", ["projects/abc/y.py"]),
    ("timeout 9 cp x.py projects/abc/q.py", ["projects/abc/q.py"]),
    ("(echo a)>projects/abc/p", ["projects/abc/p"]),
    ("echo a;>projects/abc/r", ["projects/abc/r"]),
    ("cp -t projects/abc/dir x y", ["projects/abc/dir/x", "projects/abc/dir/y"]),
    ("echo hi > /workspace/projects/abc/w", ["projects/abc/w"]),
    ("echo hi > ~/projects/abc/h", ["projects/abc/h"]),
])
def test_more_write_forms_name_what_they_wrote(cmd, want, monkeypatch, tmp_path):
    import ghost_agent.tools.projects as P
    from ghost_agent.core.request_scope import RequestScope, install, reset
    from ghost_agent.tools.execute import _note_execute_project_writes
    noted = []
    monkeypatch.setattr(P, "note_project_write", lambda ctx, rel: noted.append(str(rel)), raising=False)
    tok = install(RequestScope(context=SimpleNamespace(sandbox_dir=tmp_path)))
    try:
        _note_execute_project_writes({"command": cmd, "sandbox_dir": tmp_path}, "EXIT CODE: 0\nSTDOUT/STDERR:\n")
    finally:
        reset(tok)
    assert [n.split(str(tmp_path) + "/")[-1] for n in noted] == want, cmd


@pytest.mark.parametrize("cmd", ["wget -O- http://x", "curl -s http://x | tail", "dd if=projects/abc/a of=/dev/null",
                                 "sudo -u projects/abc/x ls"])
def test_reads_through_the_new_forms_are_not_writes(cmd):
    from ghost_agent.tools.execute import _PROJECT_PATH_RE, _shell_write_targets, workspace_relative
    assert not [t for t in _shell_write_targets(cmd) if _PROJECT_PATH_RE.fullmatch(workspace_relative(t))], cmd


# ── round 7 ──────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rid,owner", [("sched-4ul-r7", False), ("probe-4ul-r7", True)])
async def test_an_upstream_abort_inside_a_scheduled_or_probe_batch_stops_it_once(turns_on, monkeypatch, store,
                                                                                 rid, owner):
    """A scheduled batch / probe counts itself in foreground_requests — its abort is the model server, and the
    batch stops on the first one instead of re-picking the task for its whole count."""
    from ghost_agent.core.project_advancer import advance_many
    from ghost_agent.tools.projects import _ADVANCE_STOP_BLURB
    pid, tid = _project(store, "Build a → a.py")
    store.add_task(pid, "Build b → b.py")
    store.add_task(pid, "Build c → c.py")
    m = dict(store.get_project(pid).get("metadata") or {})
    m["autopilot"] = True
    store.update_project(pid, metadata=m)
    seen = []

    async def go(agent, kw):
        return "[ATTEMPT_ABORTED_UPSTREAM] the model server returned 503"
    _turn(monkeypatch, store, go, seen)
    tok = request_id_context.set(rid)
    try:
        batch = await advance_many(_ctx(store, fg=1), pid, max_tasks=3, owner_requested=owner, stop_on_fail=False,
                                   tool_runner=lambda *a, **k: None, coding_executor=_executor())
    finally:
        request_id_context.reset(tok)
    assert len(seen) == 1 and batch.stop_reason == "model_server_failed"
    assert [a["classification"] for a in batch.advanced] == ["blocked"]
    assert batch.advanced[0]["summary"].startswith("the model server failed during the task's turn")
    assert store.get_task(tid)["status"] in ("PENDING", "READY")          # reopened, not failed
    assert "model server failed" in _ADVANCE_STOP_BLURB["model_server_failed"]


async def test_an_in_progress_the_turn_set_never_covers_the_owners_reset_to_pending(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await agent.available_tools["manage_projects"](action="task_update", task_id=tid, status="IN_PROGRESS")
        await _write_then_done(agent, kw, tid)                  # refused: nothing ran after the write
        store.update_task(tid, status="PENDING")                # the owner's console resets it
        return "was working"
    _turn(monkeypatch, store, go, [])
    await _advance(_ctx(store), pid, owner=False)
    row = store.get_task(tid)
    assert row["status"] == "PENDING" and not row.get("wait_code")          # the owner's status stands


async def test_a_grandchild_close_that_rolls_the_task_up_is_the_turns_close(turns_on, monkeypatch, store):
    pid, tid = _project(store, "Build the app → app.py")

    async def go(agent, kw):                                    # the turn splits its task two levels deep
        kid = store.add_task(pid, "Build the core → core.py", parent_id=tid)
        grand = store.add_task(pid, "Build the parser → parser.py", parent_id=kid)
        await agent.available_tools["manage_projects"](action="task_update", task_id=grand, status="DONE",
                                                       result="built it")
        store.update_task(kid, status="DONE")                   # the handler's roll-up, played by the store
        store.update_task(tid, status="DONE")
        return "built it"
    _turn(monkeypatch, store, go, [])
    res = await _advance(_ctx(store), pid, owner=True)
    assert "set during its turn" not in res.summary


def test_a_multi_id_update_books_only_the_ids_the_handler_applied():
    import json
    w = PT._TurnWatch("p", "a", unattended=True)
    out = ToolOutcome.ok(json.dumps({"updated": [{"id": "a", "status": "DONE"}, {"id": "b", "status": "FAILED"}],
                                     "count": 2, "not_closed": [{"task_id": "b", "status": "FAILED"}]}))
    w.turn_set["c"] = "IN_PROGRESS"
    w.note("manage_projects", {"action": "task_update", "task_ids": ["a", "b", "c"], "status": "DONE"}, out)
    # round 8: b ENDED FAILED (a close gate turned the DONE down) — the turn's own FAILED close; c was never
    # written (refused before the write), so what the turn set on it before stands
    assert w.turn_set == {"a": "DONE", "b": "FAILED", "c": "IN_PROGRESS"}
    w2 = PT._TurnWatch("p", "a", unattended=True)
    w2.note("manage_projects", {"action": "task_update", "task_ids": ["a", "b"], "status": "FAILED"},
            ToolOutcome.ok("updated"))
    assert w2.turn_set == {"a": "FAILED", "b": "FAILED"}        # not the handler's JSON: the ids asked for


async def test_an_upstream_abort_in_the_owners_run_all_reopens_the_task_and_stops(turns_on, monkeypatch, store):
    from ghost_agent.core.project_advancer import advance_many
    pid, tid = _project(store, "Build a → a.py")
    store.add_task(pid, "Build b → b.py")
    seen = []

    async def go(agent, kw):
        return "[ATTEMPT_ABORTED_UPSTREAM] the model server returned 503"
    _turn(monkeypatch, store, go, seen)
    tok = request_id_context.set("req-owner-r7")
    try:
        batch = await advance_many(_ctx(store, fg=1), pid, max_tasks=None, owner_requested=True,
                                   stop_on_fail=False, tool_runner=lambda *a, **k: None, coding_executor=_executor())
    finally:
        request_id_context.reset(tok)
    assert len(seen) == 1 and batch.stop_reason == "model_server_failed"
    assert store.get_task(tid)["status"] in ("PENDING", "READY")
    assert all(t["status"] != "FAILED" for t in store.list_tasks(pid))


# ── round 8 ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _catch_all_reply(exc_name):
    """The reply agent.py's catch-all writes for an exception raised in the model call."""
    from ghost_agent.core.agent import UPSTREAM_ABORT_MARKER
    return (f"CRITICAL: An unexpected error occurred while communicating with the LLM ({exc_name}). "
            f"Please try again in a moment. " + UPSTREAM_ABORT_MARKER)


@pytest.mark.parametrize("cls_name", ["BackgroundDeferred", "LaneDeferred"])
async def test_a_deferral_raised_in_the_model_call_is_a_deferral_even_with_no_foreground(turns_on, monkeypatch,
                                                                                         store, cls_name):
    from ghost_agent.core import llm
    assert f"with the LLM ({getattr(llm, cls_name).__name__})" in PT.deferral_marks()   # derived from the classes
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        await agent.available_tools["file_system"](operation="write", path="a.py", content="x")
        return _catch_all_reply(cls_name)
    _turn(monkeypatch, store, go, [])
    res = await _unattended_step(store, pid)                    # idle autopilot, foreground_requests == 0
    md = store.get_project(pid)["metadata"]
    assert res.classification == "deferred" and "model server failed" not in res.summary
    assert not int(md.get("no_progress_streak") or 0) and not md.get("autopilot_rest")
    assert store.get_task(tid)["status"] in ("PENDING", "READY")
    assert "the main slot was busy" in res.summary and "owner's request" not in res.summary   # round 9: no owner here


async def test_a_reply_merely_quoting_a_deferral_class_is_not_a_deferral(turns_on, monkeypatch, store):
    """Round 9: only agent.py's whole phrase AND its marker — a model reply naming the class is the model's text."""
    from ghost_agent.core.agent import UPSTREAM_ABORT_MARKER
    assert not PT.is_deferral_reply("the log said (BackgroundDeferred) once")
    assert not PT.is_deferral_reply("it failed (BackgroundDeferred) " + UPSTREAM_ABORT_MARKER)
    assert not PT.is_deferral_reply("communicating with the LLM (BackgroundDeferred)")
    assert PT.is_deferral_reply(_catch_all_reply("BackgroundDeferred"))
    pid, tid = _project(store, "Build a → a.py")

    async def go(agent, kw):
        return "I saw (BackgroundDeferred) in the server log " + UPSTREAM_ABORT_MARKER
    _turn(monkeypatch, store, go, [])
    res = await _unattended_step(store, pid)
    assert res.classification == "blocked" and "model server failed" in res.summary


async def test_a_deferral_mark_inside_an_outer_request_is_the_model_server(turns_on, monkeypatch, store):
    from ghost_agent.core.project_advancer import advance_many
    pid, tid = _project(store, "Build a → a.py")
    m = dict(store.get_project(pid).get("metadata") or {})
    m["autopilot"] = True
    store.update_project(pid, metadata=m)

    async def go(agent, kw):
        return _catch_all_reply("BackgroundDeferred")
    _turn(monkeypatch, store, go, [])
    tok = request_id_context.set("sched-4ul-r8")
    try:
        batch = await advance_many(_ctx(store, fg=0), pid, max_tasks=2, owner_requested=False, stop_on_fail=False,
                                   tool_runner=lambda *a, **k: None, coding_executor=_executor())
    finally:
        request_id_context.reset(tok)
    assert batch.stop_reason == "model_server_failed"


async def test_a_claimed_owner_hold_goes_back_on_hold_when_the_model_server_fails(turns_on, monkeypatch, store):
    from ghost_agent.core.project_advancer import advance_many
    pid, tid = _project(store, "Build a → a.py")
    store.update_task(tid, status="NEEDS_USER", wait_on="owner_turn", wait_code="unverified_build",
                      wait_why="built but not checked")
    seen = []

    async def go(agent, kw):
        return _catch_all_reply("ReadTimeout")                  # no call ran
    _turn(monkeypatch, store, go, seen)
    tok = request_id_context.set("req-owner-r8")
    try:
        batch = await advance_many(_ctx(store, fg=1), pid, max_tasks=None, owner_requested=True,
                                   stop_on_fail=False, tool_runner=lambda *a, **k: None, coding_executor=_executor())
    finally:
        request_id_context.reset(tok)
    assert len(seen) == 1 and batch.stop_reason == "model_server_failed"
    row = store.get_task(tid)
    assert row["status"] == "NEEDS_USER" and row["wait_on"] == "owner_turn"
    assert batch.advanced[0]["summary"].endswith("task back on hold (owner)")
    assert tid in batch.held and batch.ran == 0
