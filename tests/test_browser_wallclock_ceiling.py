"""§4GI (2026-09-13): no browser call can hold the shared profile lock past
one wall-clock ceiling, whatever the model passes.

Pre-fix: `timeout_ms` was model-supplied and unclamped, `interact`
multiplied it by the action count, and the sandbox exec got
`max(60, ms//1000 + 30)` — `timeout_ms=3_600_000` produced a 3,630 s exec
under `_BROWSER_PROFILE_LOCK`, queuing every other conversation's browser
call behind it. Now ONE ceiling (`GHOST_BROWSER_WALLCLOCK_S`, default 600,
floor 60) bounds the per-op runner budget, the interact total, and the exec
timeout, at the single function every exec goes through — pinned by an AST
enumeration of the exec sites.
"""
import ast
import inspect
import json
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools import browser as B


def _stub(output=None):
    stub = MagicMock()
    stub.timeouts = []
    stub.cmds = []
    payload = output or {"status": 200, "url": "file:///workspace/x.html",
                         "title": "X", "text": "hello world " * 20}

    def _execute(cmd, timeout=300, **kwargs):
        stub.timeouts.append(timeout)
        stub.cmds.append(cmd)
        return (f"[BROWSER_OK] {json.dumps(payload)}\n", 0)
    stub.execute = _execute
    return stub


def _runner_timeout_ms(stub):
    """The `timeout_ms` the runner was handed (from the exec command)."""
    cmd = stub.cmds[-1]
    raw = cmd.split(" ", 3)[-1]
    import shlex
    return json.loads(shlex.split(raw)[0])["timeout_ms"]


async def test_a_one_hour_timeout_is_clamped_to_the_ceiling(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub()
    await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                         timeout_ms=3_600_000, sandbox_dir=tmp_path, sandbox_manager=stub)
    assert stub.timeouts and max(stub.timeouts) <= 600          # pre-fix: 3630
    assert _runner_timeout_ms(stub) <= (600 - 30) * 1000


async def test_sixty_interact_actions_do_not_multiply_past_the_ceiling(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub({"status": 200, "url": "file:///workspace/x.html", "title": "X",
                  "results": []})
    actions = [{"action": "goto", "url": "file:///workspace/x.html"}] + \
              [{"action": "extract_text", "selector": "body"} for _ in range(59)]
    await B.tool_browser(operation="interact", actions=actions, timeout_ms=30000,
                         sandbox_dir=tmp_path, sandbox_manager=stub)
    assert stub.timeouts and max(stub.timeouts) <= 600          # pre-fix: 1830
    assert _runner_timeout_ms(stub) <= (600 - 30) * 1000        # pre-fix: 30000 (per op) but exec 1830 s


async def test_a_normal_call_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub()
    await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                         timeout_ms=30000, sandbox_dir=tmp_path, sandbox_manager=stub)
    assert stub.timeouts == [60]
    assert _runner_timeout_ms(stub) == 30000


async def test_the_ceiling_is_env_overridable_with_a_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_BROWSER_WALLCLOCK_S", "120")
    stub = _stub()
    await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                         timeout_ms=3_600_000, sandbox_dir=tmp_path, sandbox_manager=stub)
    assert max(stub.timeouts) <= 120
    monkeypatch.setenv("GHOST_BROWSER_WALLCLOCK_S", "5")
    assert B._wallclock_ceiling_s() == 60
    monkeypatch.setenv("GHOST_BROWSER_WALLCLOCK_S", "nope")
    assert B._wallclock_ceiling_s() == 600


@pytest.mark.parametrize("ms,expect_s", [
    (1_000, 60), (30_000, 60), (300_000, 330), (570_000, 600), (3_600_000, 600),
])
def test_bounded_subprocess_timeout_table(ms, expect_s, monkeypatch):
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    assert B._bounded_subprocess_timeout(ms) == expect_s


def test_runner_budget_always_expires_before_the_exec_kill(monkeypatch):
    """The runner's own timeout must be under the exec timeout by the slack,
    so a hung page yields a runner-level error, not a sandbox kill."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    for ms in (1_000, 30_000, 600_000, 3_600_000):
        runner = B._clamp_runner_timeout_ms(ms)
        assert runner // 1000 + B._SUBPROCESS_SLACK_S <= B._bounded_subprocess_timeout(runner) + 0
        assert runner // 1000 < B._bounded_subprocess_timeout(runner)


# ── R1 enumeration: every exec of the runner goes through the clamp ─────────

def test_every_sandbox_exec_in_tool_browser_uses_the_bounded_timeout():
    src = inspect.getsource(B)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "tool_browser")
    execs = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "asyncio.to_thread"
             and n.args and ast.unparse(n.args[0]) == "sandbox_manager.execute"]
    assert len(execs) >= 3, "the exec sites moved — re-point this enumeration"
    # §4GI round 3: it is no longer enough for each exec to be individually
    # bounded — they must share ONE deadline, so the timeout each is given
    # has to come from `_exec_timeout_for(_deadline, ...)`. Passing the raw
    # `subprocess_timeout` is exactly the defect (3 x 600 s for one call).
    # Names bound from the deadline helper count too (the retries bind it
    # first so they can SKIP the attempt when nothing is left) — one hop,
    # resolved here rather than accepted on faith.
    deadline_names = {
        t.id
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
        and "_exec_timeout_for(_deadline" in ast.unparse(n.value)
    }
    for call in execs:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        t = kw.get("timeout", "")
        ok = ("_exec_timeout_for(_deadline" in t
              or any(name == t or t.startswith(name) for name in deadline_names))
        assert ok, ast.unparse(call)
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "subprocess_timeout" for t in n.targets)]
    assert len(assigns) == 1
    assert ast.unparse(assigns[0].value).startswith("_bounded_subprocess_timeout(")
    dl = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == "_deadline" for t in n.targets)]
    assert len(dl) == 1 and ast.unparse(dl[0].value) == "_call_deadline()"


def test_the_enumeration_fires_on_a_raw_timeout():
    src = inspect.getsource(B)
    # Re-pointed in §4GK round 4: the primary site no longer spells the
    # helper inline — the `or 1` that inverted the "do not issue" sentinel was
    # removed and the value is now bound to `_first_t` first.
    broken = src.replace("                timeout=_first_t,\n",
                         "                timeout=3600,\n", 1)
    assert broken != src, "re-point this instrument check at the real call site"
    tree = ast.parse(broken)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "tool_browser")
    execs = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "asyncio.to_thread"
             and n.args and ast.unparse(n.args[0]) == "sandbox_manager.execute"]
    deadline_names = {
        t.id
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
        and "_exec_timeout_for(_deadline" in ast.unparse(n.value)
    }
    raw = []
    for c in execs:
        t = {k.arg: ast.unparse(k.value) for k in c.keywords}.get("timeout", "")
        if not ("_exec_timeout_for(_deadline" in t
                or any(name == t or t.startswith(name) for name in deadline_names)):
            raw.append(c)
    assert len(raw) == 1, [ast.unparse(c) for c in raw]


# ── §4GI round 3: ONE deadline, not three independent ceilings ──────────────
#
# Round 2 found that §4GI clamped each exec but shared nothing between them.
# The two defects below were invisible to every pin above, because those feed
# the CLAMPED per-op value and never the production `effective_timeout_ms`.

def _interact_actions(n):
    return ([{"action": "goto", "url": "file:///workspace/x.html"}]
            + [{"action": "extract_text", "selector": "body"} for _ in range(n - 1)])


async def test_the_runner_total_fits_inside_the_exec_kill(tmp_path, monkeypatch):
    """The mid-flow SIGTERM. 30 actions x the 30 s per-action default is
    900 s of runner against a 600 s exec cap: `timeout -k 5s` killed the
    sequence, exit 124, no `[BROWSER_OK]`, and the model could not tell
    which actions had landed. The per-action budget is now divided so the
    runner's worst case fits the window it was given.

    Fails in the pre-fix world, where the runner got the full 30 s per
    action regardless of how many there were."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub({"status": 200, "url": "file:///workspace/x.html",
                  "title": "X", "results": []})
    n = 30
    await B.tool_browser(operation="interact", actions=_interact_actions(n),
                         timeout_ms=30000, sandbox_dir=tmp_path,
                         sandbox_manager=stub)
    per_action_ms = _runner_timeout_ms(stub)
    exec_s = stub.timeouts[-1]
    assert n * per_action_ms <= exec_s * 1000, (n, per_action_ms, exec_s)
    assert per_action_ms < 30000            # pre-fix: exactly 30000


async def test_one_action_still_gets_the_whole_per_op_budget(tmp_path, monkeypatch):
    """Control — the two worlds must differ only when the count forces it.
    A single-action interact is not starved by the division."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub({"status": 200, "url": "file:///workspace/x.html",
                  "title": "X", "results": []})
    await B.tool_browser(operation="interact", actions=_interact_actions(1),
                         timeout_ms=30000, sandbox_dir=tmp_path,
                         sandbox_manager=stub)
    assert _runner_timeout_ms(stub) == 30000


async def test_an_action_is_never_starved_below_the_floor(tmp_path, monkeypatch):
    """A pathological action count divides the budget, but not to zero —
    below `_MIN_ACTION_MS` an action cannot even open a page."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub({"status": 200, "url": "file:///workspace/x.html",
                  "title": "X", "results": []})
    await B.tool_browser(operation="interact", actions=_interact_actions(500),
                         timeout_ms=30000, sandbox_dir=tmp_path,
                         sandbox_manager=stub)
    assert _runner_timeout_ms(stub) == B._MIN_ACTION_MS


async def test_retries_share_the_call_deadline_instead_of_restarting_it(tmp_path, monkeypatch):
    """The 1800 s call. Both retry sites re-used the FULL
    `subprocess_timeout`, so one `navigate` could issue 600 + 600 + 600.
    Every exec now gets only what is LEFT of the one deadline.

    Fails in the pre-fix world, where the three timeouts were identical."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub()
    # a TargetClosedError forces the launch-race retry
    calls = {"n": 0}

    def _execute(cmd, timeout=300, **kwargs):
        stub.timeouts.append(timeout)
        stub.cmds.append(cmd)
        calls["n"] += 1
        if calls["n"] == 1:
            return ("[BROWSER_ERR] TargetClosedError: browser has been closed\n", 1)
        return (f"[BROWSER_OK] {json.dumps({'status': 200, 'url': 'u', 'title': 'T', 'text': 'ok'})}\n", 0)
    stub.execute = _execute

    # burn most of the budget between the attempts, so "remaining" is visible
    real_deadline = B._call_deadline

    def _short_deadline(now=None):
        return real_deadline(now) - (B._wallclock_ceiling_s() - 30)
    monkeypatch.setattr(B, "_call_deadline", _short_deadline)

    await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                         timeout_ms=30000, sandbox_dir=tmp_path,
                         sandbox_manager=stub)
    assert len(stub.timeouts) == 2, stub.timeouts
    # the retry may not exceed what is left of the call's own deadline; 30 is
    # below the per-op budget (60), which is what a restart would use
    assert stub.timeouts[1] <= 30, stub.timeouts
    assert sum(stub.timeouts) <= B._wallclock_ceiling_s(), stub.timeouts


async def test_no_exec_at_all_is_issued_when_the_budget_is_already_spent(tmp_path, monkeypatch):
    """§4GK round 4. `_exec_timeout_for` returns 0 to mean "do NOT issue
    this exec"; both retry sites honour it, but the PRIMARY site wrote
    `... or 1`, turning the sentinel into a ONE-SECOND exec that cannot
    launch Chromium. The model got "runner exit 124" — which reads as "the
    site timed out" — plus a failure strike, and paid a doomed browser
    launch for it. The deadline starts before the agent-wide profile lock is
    taken, so arriving here with a spent budget is the ordinary queued case.

    Fails in any tree that issues an exec with no budget left."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub()

    def _execute(cmd, timeout=300, **kwargs):
        stub.timeouts.append(timeout)
        stub.cmds.append(cmd)
        return ("[BROWSER_ERR] TargetClosedError: browser has been closed\n", 1)
    stub.execute = _execute
    monkeypatch.setattr(B, "_call_deadline", lambda now=None: 0.0)  # already spent

    out = await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                               timeout_ms=30000, sandbox_dir=tmp_path,
                               sandbox_manager=stub)
    assert stub.timeouts == [], stub.timeouts       # pre-fix: [1], then [1, 1]
    # and it says what actually happened, not "the page timed out"
    assert "budget" in str(out).lower(), out
    assert "124" not in str(out), out


def test_exec_timeout_for_table(monkeypatch):
    """The helper itself, at its edges."""
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    now = 1000.0
    assert B._exec_timeout_for(now + 600, 600, now=now) == 600      # full budget
    assert B._exec_timeout_for(now + 100, 600, now=now) == 100      # capped by remaining
    assert B._exec_timeout_for(now + 600, 45, now=now) == 45        # capped by its own want
    assert B._exec_timeout_for(now + 5, 600, now=now) == 0          # too little: skip
    assert B._exec_timeout_for(now - 10, 600, now=now) == 0         # already past


def test_runner_action_timeout_divides_and_floors():
    assert B._runner_action_timeout_ms(570_000, 1) == 570_000
    assert B._runner_action_timeout_ms(570_000, 30) == 19_000
    assert B._runner_action_timeout_ms(570_000, 10_000) == B._MIN_ACTION_MS
    assert B._runner_action_timeout_ms(570_000, 0) == 570_000       # n<1 guarded


async def test_the_commit_milestone_retry_also_shares_the_deadline(tmp_path, monkeypatch):
    """The SECOND retry site. The existing pin above drives the launch-race
    retry (TargetClosedError); the §4GJ battery survived a mutant that gave
    the commit-milestone retry the full budget again, because nothing
    exercised that branch. A navigate that TIMES OUT over Tor retries once
    with `wait_until='commit'` — and that exec must also get only what is
    left of the one call deadline.

    Fails in the pre-fix world, where the retry restarted from
    `subprocess_timeout`.
    """
    monkeypatch.delenv("GHOST_BROWSER_WALLCLOCK_S", raising=False)
    stub = _stub()
    calls = {"n": 0}

    def _execute(cmd, timeout=300, **kwargs):
        stub.timeouts.append(timeout)
        stub.cmds.append(cmd)
        calls["n"] += 1
        if calls["n"] == 1:
            return ("[BROWSER_ERR] navigate failed: Timeout 30000ms exceeded\n", 1)
        return (f"[BROWSER_OK] {json.dumps({'status': 200, 'url': 'u', 'title': 'T', 'text': 'ok'})}\n", 0)
    stub.execute = _execute

    real_deadline = B._call_deadline

    # Leave ~30 s: LESS than the per-op `subprocess_timeout` (60 s for a
    # 30 s timeout_ms) and more than `_MIN_EXEC_S`, so a retry that restarts
    # from the full budget is visibly different from one that takes what is
    # left. An earlier version of this pin left 120 s — above the per-op
    # value — so both worlds passed it and the mutant survived.
    def _short_deadline(now=None):
        return real_deadline(now) - (B._wallclock_ceiling_s() - 30)
    monkeypatch.setattr(B, "_call_deadline", _short_deadline)

    await B.tool_browser(operation="navigate", url="https://example.org/x",
                         timeout_ms=30000, sandbox_dir=tmp_path,
                         sandbox_manager=stub,
                         tor_proxy="socks5://127.0.0.1:9050")
    assert len(stub.timeouts) == 2, stub.timeouts
    assert any("commit" in c for c in stub.cmds), stub.cmds     # the right retry ran
    # 30 is BELOW the per-op `subprocess_timeout` (60), so a retry that
    # restarts from the full budget fails here. (Both execs read the same
    # remaining budget in this fixture — "strictly less than the first" is
    # NOT the distinguishing property, and asserting it reddened the fix.)
    assert stub.timeouts[1] <= 30, stub.timeouts                # pre-fix: the full 60
    assert sum(stub.timeouts) <= B._wallclock_ceiling_s(), stub.timeouts
