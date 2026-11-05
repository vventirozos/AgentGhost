"""The READY banner must be the LAST line of boot.

The main-prefix warmup is spawned in the background (it prefills ~24k tokens
and must not hold up the socket), so it was the one boot step that logged
AFTER "system ready" — which made the banner read as a lie in the operator's
stream, since the stream is what the operator watches boot in.

The banner is awaited BEFORE the `yield` in `lifespan`, and the server starts
serving at the `yield`: the waits here are bounded because they hold the socket
too (§4SY review — this docstring used to say only the log line waits).
"""

import asyncio
import time

import pytest

import ghost_agent.main as main_mod


@pytest.mark.asyncio
async def test_ready_is_announced_after_the_warmup(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))

    async def _warm():
        await asyncio.sleep(0.05)
        seen.append("Prefix Warmup Done")

    await main_mod._announce_ready_when_warm(asyncio.ensure_future(_warm()))
    assert seen == ["Prefix Warmup Done", "System Ready"], (
        f"ready must come last, got {seen}")


@pytest.mark.asyncio
async def test_ready_is_immediate_when_there_is_no_warmup(monkeypatch):
    """A mocked client or GHOST_MAIN_PREFIX_WARMUP=0 spawns no task."""
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))
    t0 = time.monotonic()
    await main_mod._announce_ready_when_warm(None)
    assert seen == ["System Ready"]
    assert time.monotonic() - t0 < 0.5, "must not wait on nothing"


@pytest.mark.asyncio
async def test_a_wedged_warmup_cannot_suppress_the_ready_line(monkeypatch):
    """A log line must never be able to hide. A hung upstream would otherwise
    mean NO ready banner for a server that is already serving."""
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))
    hung = asyncio.ensure_future(asyncio.sleep(30))
    try:
        t0 = time.monotonic()
        await main_mod._announce_ready_when_warm(hung, timeout=0.2)
        assert "System Ready" in seen
        assert "Prefix Warmup Slow" in seen, "the operator is told why"
        assert time.monotonic() - t0 < 2.0
        # …and the wait is SHIELDED: timing out must not cancel the warmup.
        assert not hung.done(), "the warmup must survive the announce timeout"
    finally:
        hung.cancel()


@pytest.mark.asyncio
async def test_a_failed_warmup_still_announces_ready(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))

    async def _boom():
        raise RuntimeError("upstream down")

    task = asyncio.ensure_future(_boom())
    await main_mod._announce_ready_when_warm(task)
    assert seen == ["System Ready"]


@pytest.mark.asyncio
async def test_an_already_finished_warmup_does_not_stall(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))

    async def _done():
        return None

    task = asyncio.ensure_future(_done())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    t0 = time.monotonic()
    await main_mod._announce_ready_when_warm(task)
    assert seen == ["System Ready"]
    assert time.monotonic() - t0 < 0.5


def test_the_lifespan_announces_through_the_helper():
    """The banner must not be re-inlined by a later edit — that is exactly
    how it ended up printing before the warmup in the first place."""
    import inspect
    src = inspect.getsource(main_mod)
    assert "_announce_ready_when_warm(_warmup_task)" in src
    assert src.count('pretty_log("System Ready"') == 1, (
        "one announce site only — the helper's")


def test_the_ready_banner_waits_five_minutes_for_the_warmup():
    """§4NX (operator, dense model): fails where the banner gave up at 120 s
    while a dense model was still prefilling (or the warmup request's own
    timeout cut it shorter than the wait)."""
    import inspect
    import ghost_agent.main as main_mod
    assert main_mod.PREFIX_WARMUP_WAIT_S == 300.0
    assert inspect.signature(main_mod._announce_ready_when_warm).parameters["timeout"].default == 300.0


# ── §4SY: the labs and runs lines are boot lines too (operator 11-05: labs was logged after the banner) ──

def _boot_sets(monkeypatch):
    from ghost_agent.tools import labs, runs
    lab_tasks, run_tasks = set(), set()
    monkeypatch.setattr(labs, "_BOOT_TASKS", lab_tasks)
    monkeypatch.setattr(runs, "_BOOT_TASKS", run_tasks)
    return lab_tasks, run_tasks


def _log(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log", lambda title, content="", **kw: seen.append(title))
    return seen


@pytest.mark.asyncio
async def test_ready_is_announced_after_the_labs_line_started_the_real_way(monkeypatch):
    """Through `start_labs_boot_line` itself: the banner finds the task where the starter keeps it."""
    from ghost_agent.tools import labs
    seen = _log(monkeypatch)
    _boot_sets(monkeypatch)

    async def _labs_line(context):
        await asyncio.sleep(0.15)
        seen.append("Labs")
    monkeypatch.setattr(labs, "labs_boot_line", _labs_line)
    labs.start_labs_boot_line(object())

    async def _warm():
        await asyncio.sleep(0.02)
        seen.append("Prefix Warmup Done")
    await main_mod._announce_ready_when_warm(asyncio.ensure_future(_warm()))
    assert seen == ["Prefix Warmup Done", "Labs", "System Ready"], f"ready must come last, got {seen}"


@pytest.mark.asyncio
async def test_ready_is_announced_after_the_runs_line_started_the_real_way(monkeypatch):
    from ghost_agent.tools import runs
    seen = _log(monkeypatch)
    _boot_sets(monkeypatch)

    async def _runs_line(context):
        await asyncio.sleep(0.15)
        seen.append("Runs")
    monkeypatch.setattr(runs, "runs_boot_line", _runs_line)
    runs.start_runs_boot_line(object())
    await main_mod._announce_ready_when_warm(None)
    assert seen == ["Runs", "System Ready"]


@pytest.mark.asyncio
async def test_a_boot_line_that_never_comes_cannot_suppress_the_ready_line_and_is_not_cancelled(monkeypatch):
    seen = _log(monkeypatch)
    lab_tasks, _ = _boot_sets(monkeypatch)
    assert main_mod.BOOT_LINES_WAIT_S == 12.0
    monkeypatch.setattr(main_mod, "BOOT_LINES_WAIT_S", 0.2)
    hung = asyncio.ensure_future(asyncio.sleep(30))
    lab_tasks.add(hung)
    try:
        t0 = time.monotonic()
        await main_mod._announce_ready_when_warm(None)
        assert seen == ["System Ready"] and 0.15 < time.monotonic() - t0 < 2.0
        await asyncio.sleep(0)           # a cancel requested by the wait would land on this turn
        await asyncio.sleep(0)
        assert not hung.done() and not hung.cancelled(), "the boot line's task must survive the wait"
    finally:
        hung.cancel()


@pytest.mark.asyncio
async def test_a_failed_boot_line_still_announces_ready(monkeypatch):
    seen = _log(monkeypatch)
    lab_tasks, _ = _boot_sets(monkeypatch)

    async def _boom():
        raise RuntimeError("labs line died")
    t = asyncio.ensure_future(_boom())
    lab_tasks.add(t)
    await main_mod._announce_ready_when_warm(None)
    assert seen == ["System Ready"] and t.done()
    t.exception()                        # retrieved: no "never retrieved" warning


@pytest.mark.asyncio
async def test_ready_does_not_wait_when_no_boot_line_is_pending(monkeypatch):
    seen = _log(monkeypatch)
    _boot_sets(monkeypatch)
    t0 = time.monotonic()
    await main_mod._announce_ready_when_warm(None)
    assert seen == ["System Ready"] and time.monotonic() - t0 < 0.5


@pytest.mark.asyncio
async def test_a_cancelled_startup_is_not_swallowed_by_the_wait(monkeypatch):
    seen = _log(monkeypatch)
    lab_tasks, _ = _boot_sets(monkeypatch)
    hung = asyncio.ensure_future(asyncio.sleep(30))
    lab_tasks.add(hung)
    try:
        announce = asyncio.ensure_future(main_mod._announce_ready_when_warm(None))
        await asyncio.sleep(0.05)
        announce.cancel()
        with pytest.raises(asyncio.CancelledError):
            await announce
        assert seen == [], "a cancelled startup must not announce ready"
    finally:
        hung.cancel()


def test_the_banner_is_the_last_statement_before_serving():
    """The source-staleness audit can log a line; it ran AFTER the banner."""
    import ast
    import inspect
    import textwrap
    fn = next(n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(main_mod))))
              if isinstance(n, ast.AsyncFunctionDef)
              and any(isinstance(x, ast.Yield) for x in ast.walk(n))
              and any(isinstance(x, ast.Name) and x.id == "_announce_ready_when_warm" for x in ast.walk(n)))
    def line_of(pred):
        hits = [x.lineno for x in ast.walk(fn) if pred(x)]
        assert len(hits) == 1, hits
        return hits[0]
    announce = line_of(lambda x: isinstance(x, ast.Await) and isinstance(x.value, ast.Call)
                       and getattr(x.value.func, "id", "") == "_announce_ready_when_warm")
    audit = line_of(lambda x: isinstance(x, ast.Call) and getattr(x.func, "id", "") == "audit_source_newer_than_process")
    serve = min(x.lineno for x in ast.walk(fn) if isinstance(x, ast.Yield))
    assert audit < announce < serve
    # nothing but the `try:` that holds the yield stands between the banner and serving
    between = [x for x in fn.body if announce < x.lineno < serve]
    assert [type(x).__name__ for x in between] == ["Try"] and between[0].lineno < serve
