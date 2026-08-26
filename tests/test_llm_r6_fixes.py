"""§4BV R6 — pins for the guards R5 shipped without one.

R6's mutation audit found 21 survivors across round 5's own tests. These are
the replacements for the ones that guard live behaviour. The recurring shape:
a test that asserts a NAME is present rather than driving what it does.
"""

import ast
import asyncio
import inspect
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.llm import LLMClient

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


# --------------------------------------------------------------- callers

def _call_kwargs(rel_path, task_label):
    """The kwargs of the chat_completion call carrying `task_label`."""
    tree = ast.parse((_SRC / rel_path).read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        kw = {k.arg: k.value for k in n.keywords if k.arg}
        lbl = kw.get("task_label")
        if isinstance(lbl, ast.Constant) and lbl.value == task_label:
            out.append(kw)
    return out


@pytest.mark.parametrize("rel_path,label", [
    ("tools/search.py", "web summary"),
    ("tools/darkweb_search.py", "web summary"),
    ("core/build_gates.py", "constraint gate"),
])
def test_every_caller_under_an_outer_deadline_states_a_real_budget(
        rel_path, label):
    """⚠ ONE TEST, SIX MUTANTS. R6 found that dropping `slot_wait`
    entirely at build_gates, passing `slot_wait=None` at search and
    darkweb, and flipping `off_main_only` to False all survived — the last
    one because the assertion guarding it was `"off_main_only=is_background"
    in src`, satisfied by the identical string sitting in a COMMENT.

    `slot_wait=None` matters because it is byte-for-byte the regression
    these call sites exist to prevent: `_do_chat_completion` skips the cap
    entirely on None, so the call silently inherits the 90s operator
    ceiling and blows the caller's own outer deadline."""
    calls = _call_kwargs(rel_path, label)
    assert calls, f"{rel_path}: the {label!r} call is gone"
    for kw in calls:
        for name in ("timeout", "slot_wait", "total_budget"):
            assert name in kw, (
                f"{rel_path}: the {label!r} call does not state {name}. It "
                f"runs under an outer deadline of its own; without a stated "
                f"total it inherits the 90s ceiling and the outer "
                f"`wait_for` cancels the whole operation.")
            assert not (isinstance(kw[name], ast.Constant)
                        and kw[name].value is None), (
                f"{rel_path}: {name}=None is exactly the unbounded case")


def _fn_named(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n.name == name:
            return n
    return None


@pytest.mark.parametrize("rel_path,outer_const", [
    ("tools/search.py", "PER_URL_TIMEOUT"),
    ("tools/darkweb_search.py", "_ONION_URL_CEILING_S"),
])
def test_the_per_url_distill_budget_derives_from_the_clock(
        rel_path, outer_const):
    """⚠ STATING A BUDGET IS NOT THE SAME AS STATING THE RIGHT ONE.

    The sibling test above only checks that the three kwargs are present and
    not literal `None` — so it passed while `search.py` handed the distill a
    flat `_WEB_SUMMARY_TIMEOUT_S` (45.0) under a `PER_URL_TIMEOUT` of 55.0
    that ALSO covers the semaphore wait and up to two 22s fetch attempts.
    45 is not the remainder of 55; it is merely smaller. Measured: the URL
    was lost at 55.00s with the worker node completely FREE (R7 lens A), and
    two mutants that hardcode an absurd budget survived (R7 lens C).

    The property is a RELATIONSHIP: the budget passed must be computed from
    a deadline that is itself computed from the clock — so it shrinks as the
    fetch consumes time.

    ⚠ UPDATED 2026-08-25: the ceiling moved one frame out, to `_bounded`, and
    this pin follows the mechanism rather than the name. `process_url` used to
    open its own `PER_URL_TIMEOUT` window on entry — but every `_bounded`
    coroutine is created by one `asyncio.gather`, so all eight windows opened
    SIMULTANEOUSLY while `Semaphore(3)` let only three run, and URLs 4-8
    reached the distiller with seconds left (the 4th was handed 6s for a 27s
    job and posted it anyway — req 08766aa1). The ceiling is now applied
    after the semaphore is acquired and passed in as `budget_s`, so both
    halves are pinned here: `_bounded` derives the budget from the outer
    ceiling AND the phase deadline, and `process_url` turns that into a
    clock-based deadline."""
    tree = ast.parse((_SRC / rel_path).read_text(encoding="utf-8"))
    fn = _fn_named(tree, "process_url")
    assert fn is not None, f"{rel_path}: process_url is gone"

    # the deadline must come from the monotonic clock plus the outer ceiling
    deadlines = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_url_deadline"
                         for t in n.targets)]
    assert deadlines, (
        f"{rel_path}: no per-URL deadline — the distill budget cannot know "
        f"how much of the outer window the fetch already spent")
    names = {n.id for d in deadlines for n in ast.walk(d)
             if isinstance(n, ast.Name)}
    attrs = {n.attr for d in deadlines for n in ast.walk(d)
             if isinstance(n, ast.Attribute)}
    assert "monotonic" in attrs, (
        f"{rel_path}: the per-URL deadline is not read from the clock")
    # The window must be the one HANDED IN, not a constant re-opened here —
    # a constant would restore the simultaneous-clock defect.
    assert "budget_s" in names, (
        f"{rel_path}: the per-URL deadline is not derived from the budget "
        f"issued after the semaphore was acquired")
    assert {a.arg for a in fn.args.args} >= {"budget_s"}, (
        f"{rel_path}: process_url does not accept a budget")

    # ...and the ISSUER must clip that budget by both the per-URL ceiling and
    # the whole-phase deadline, after acquiring the semaphore.
    issuer = _fn_named(tree, "_bounded")
    assert issuer is not None, f"{rel_path}: _bounded is gone"
    issued = [n for n in ast.walk(issuer)
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "budget"
                      for t in n.targets)]
    assert issued, f"{rel_path}: _bounded issues no budget"
    inames = {n.id for b in issued for n in ast.walk(b)
              if isinstance(n, ast.Name)}
    iattrs = {n.attr for b in issued for n in ast.walk(b)
              if isinstance(n, ast.Attribute)}
    assert outer_const in inames, (
        f"{rel_path}: the issued budget ignores {outer_const}, the ceiling "
        f"the outer wait_for actually applies")
    assert "_phase_deadline" in inames, (
        f"{rel_path}: the issued budget ignores the whole-phase deadline, so "
        f"a late URL can be started with work that cannot finish")
    assert "monotonic" in iattrs, (
        f"{rel_path}: the issued budget is not read from the clock")
    assert any(isinstance(n, ast.AsyncWith) for n in ast.walk(issuer)), (
        f"{rel_path}: _bounded no longer acquires the semaphore, so the "
        f"budget is issued before the URL actually starts")

    # ...and the budget handed to the model must derive from that deadline
    budgets = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_summary_budget"
                       for t in n.targets)]
    assert budgets, f"{rel_path}: the distill budget is not computed"
    bnames = {n.id for b in budgets for n in ast.walk(b)
              if isinstance(n, ast.Name)}
    assert "_url_deadline" in bnames, (
        f"{rel_path}: the distill budget ignores the per-URL deadline — it "
        f"is a constant again, and a slow fetch will lose the URL")

    calls = _call_kwargs(rel_path, "web summary")
    assert calls
    for kw in calls:
        for name in ("timeout", "total_budget"):
            v = kw[name]
            assert isinstance(v, ast.Name) and v.id == "_summary_budget", (
                f"{rel_path}: {name}={ast.unparse(v)} is not the clock-"
                f"derived budget")
        # ⚠ SLOT_WAIT IS NO LONGER THE WHOLE BUDGET, DELIBERATELY (2026-08-25).
        # Passing timeout == slot_wait == total_budget let a call sit in the
        # per-node permit queue for nearly the entire budget and then POST a
        # request sized for ALL of it — the plan's arithmetic measured against
        # time already spent. That is req 08766aa1's "6s for a 27s job"
        # relocated from the tool's semaphore to `_node_slot`, and it still
        # ends as a ReadTimeout charged to the node. The permit wait is now a
        # bounded allowance, and the distill is sized for the budget MINUS it.
        sw = kw["slot_wait"]
        assert isinstance(sw, ast.Name) and sw.id == "_QUEUE_ALLOWANCE_S", (
            f"{rel_path}: slot_wait={ast.unparse(sw)} — queueing must be "
            f"bounded by an allowance the plan reserved, not by the whole "
            f"budget")

    # ...and the plan must actually subtract that allowance, or the reserve
    # is a comment rather than a constraint.
    plans = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "plan_distill"]
    assert plans, f"{rel_path}: the distill is not sized by a plan"
    pnames = {n.id for c in plans for n in ast.walk(c)
              if isinstance(n, ast.Name)}
    assert {"_summary_budget", "_QUEUE_ALLOWANCE_S"} <= pnames, (
        f"{rel_path}: plan_distill is not sized against the clock-derived "
        f"budget minus the queue allowance")


def test_the_background_gate_still_refuses_to_dogpile_main():
    """The `off_main_only` half, parsed rather than grepped."""
    calls = _call_kwargs("core/build_gates.py", "constraint gate")
    assert calls
    v = calls[0].get("off_main_only")
    assert v is not None, "the background gate no longer avoids the main slot"
    assert not (isinstance(v, ast.Constant) and v.value is False), (
        "off_main_only=False — a background build will dogpile the single "
        "main inference slot")


# ------------------------------------------------------------- the gate

def _node(url="http://N", slots=1):
    cl = MagicMock()
    r = MagicMock()
    r.json = lambda: {"choices": [{"message": {"content": "node"}}]}
    r.raise_for_status = lambda: None
    r.status_code, r.text = 200, "{}"
    cl.post = AsyncMock(return_value=r)
    props = MagicMock()
    props.json = lambda: {"total_slots": slots}
    props.raise_for_status = lambda: None
    cl.get = AsyncMock(return_value=props)
    return {"url": url, "model": "m", "client": cl, "name": "N"}


def test_growing_the_capacity_actually_adds_usable_permits():
    """⚠ R5's grow test checked object identity and the bookkeeping dict,
    never the permits. Deleting `sem.release()` left it green — the gate
    reported capacity 8 while still admitting 3 (R6 lens C, B4)."""
    c = LLMClient("http://main.invalid:8088")
    node = _node("http://G")
    node["client"].get = AsyncMock(side_effect=RuntimeError("down"))

    live = {"now": 0, "peak": 0}

    async def _work():
        async with c._node_slot(node, wait_timeout=10):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.15)
            live["now"] -= 1

    async def _drive():
        async with c._node_slot(node, wait_timeout=5):     # build at default
            pass
        node["client"] = _node("http://G", slots=8)["client"]
        c._node_cap_retry_at.clear()
        async with c._node_slot(node, wait_timeout=5):     # learn 8
            pass
        await asyncio.gather(*[_work() for _ in range(8)])

    asyncio.run(_drive())
    assert c._node_slot_built_cap["http://G"] == 8
    assert live["peak"] == 8, (
        f"the gate says capacity 8 but only admitted {live['peak']} at once "
        f"— the extra permits were never released, so a node that grew "
        f"stays throttled at its old guess forever")


def test_the_gate_is_ENTERED_not_merely_wrapped():
    """⚠ R5's driven gate test wrapped `_node_slot` in a spy and asserted a
    POST happened while `held` was True. That cannot distinguish holding a
    permit from BYPASSING the gate: with `wait_timeout=None` the real
    `_node_slot` yields without acquiring, `held` is still True, and the
    test passes (R6 lens C, P1). Count the permits instead."""
    c = LLMClient("http://main.invalid:8088")
    node = _node("http://W", slots=1)
    c.worker_clients = [node]
    c._worker_index = 0

    async def _main(*a, **kw):
        r = MagicMock()
        r.json = lambda: {"choices": [{"message": {"content": "MAIN"}}]}
        r.raise_for_status = lambda: None
        r.status_code, r.text = 200, "{}"
        return r

    c.http_client = MagicMock()
    c.http_client.post = AsyncMock(side_effect=_main)

    async def _drive():
        await c._node_capacity(node)
        # Occupy the node's only permit. A gated dispatch MUST NOT get
        # through; an un-gated one will POST straight past us.
        async with c._node_slot(node, wait_timeout=5):
            return await c._do_chat_completion(
                {"model": "m", "messages": []}, use_worker=True,
                timeout=5, slot_wait=6, total_budget=6)

    res = asyncio.run(_drive())
    assert res["choices"][0]["message"]["content"] == "MAIN", (
        "a dispatch reached a node whose only permit was held — the "
        "concurrency gate is being bypassed, not merely wrapped")
    assert node["client"].post.await_count == 0


def test_each_node_gets_its_own_lock_behaviourally():
    """R5 asserted `hasattr(c, "_node_slot_locks")`. Routing every node back
    through the single global lock leaves that True (R6 lens C, m2)."""
    c = LLMClient("http://main.invalid:8088")
    slow = _node("http://SLOW")

    async def _hang(*a, **kw):
        await asyncio.sleep(0.8)
        raise RuntimeError("no /props")

    slow["client"].get = AsyncMock(side_effect=_hang)
    fast = _node("http://FAST", slots=4)

    async def _touch(n):
        import time as _t
        t0 = _t.monotonic()
        async with c._node_slot(n, wait_timeout=8):
            pass
        return _t.monotonic() - t0

    async def _both():
        return await asyncio.gather(_touch(slow), _touch(fast))

    slow_dt, fast_dt = asyncio.run(_both())
    assert slow_dt >= 0.7
    assert fast_dt < 0.4, (
        f"a healthy node waited {fast_dt:.2f}s because an unrelated node was "
        f"being probed — the capacity lock is global again")


def test_image_generation_reuses_one_deadline_across_retries():
    """⚠ R5 pinned this with `"_img_slot_wait_now()" in src`, which both real
    forms of the defect keep: returning the full budget from the helper, and
    recomputing the deadline INSIDE the retry loop (R6 lens C, K1/K2b).

    Parsed, not grepped — a comment cannot satisfy an AST shape. Two
    properties: the deadline is established once, before the retry loop; and
    the per-attempt wait is derived from it rather than from the budget."""
    tree = ast.parse(inspect.getsource(LLMClient.generate_image).lstrip())
    fn = tree.body[0]

    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
    in_loop = {id(n) for lp in loops for n in ast.walk(lp)}

    deadline_assigns = [n for n in ast.walk(fn)
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name)
                                and t.id == "_img_deadline"
                                for t in n.targets)]
    assert deadline_assigns, "the image retry loop has no shared deadline"
    assert all(id(n) not in in_loop for n in deadline_assigns), (
        "`_img_deadline` is assigned inside the retry loop, so every attempt "
        "resets it — at the live default that is ~273s of queueing for one "
        "image instead of one 90s budget")

    helper = [n for n in ast.walk(fn)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_img_slot_wait_now"]
    assert helper, "the per-attempt wait helper is gone"
    names = {n.id for n in ast.walk(helper[0]) if isinstance(n, ast.Name)}
    assert "_img_deadline" in names, (
        "the per-attempt wait does not read the shared deadline — it is "
        "handing every retry a fresh full budget")


@pytest.mark.parametrize("exc_name", ["ConnectTimeout", "PoolTimeout",
                                      "ConnectError", "ReadError"])
def test_the_streaming_path_retries_transport_faults_and_closes_the_stream(
        exc_name):
    """⚠ LIVE, on the path every user turn flows through. The streaming
    retry tuple was the only one of three that omitted the timeout family —
    `_do_chat_completion` retries ConnectTimeout/PoolTimeout with a written
    rationale, `get_embeddings` catches the whole `TimeoutException` family.
    Here they fell through to the generic `except Exception`, which re-raises
    WITHOUT emitting `data: [DONE]`: one attempt instead of two, and a
    truncated SSE stream the client cannot terminate cleanly (R6 lens B)."""
    import httpx as _httpx

    c = LLMClient("http://main.invalid:8088")
    c.http_client = MagicMock()
    c.http_client.build_request = MagicMock(return_value=MagicMock())
    c.http_client.send = AsyncMock(
        side_effect=getattr(_httpx, exc_name)("boom"))

    chunks = []

    async def _go():
        async for ch in c._do_stream_chat_completion(
                {"model": "m", "messages": []}):
            chunks.append(ch)

    try:
        asyncio.run(_go())
    except Exception:
        pass

    assert c.http_client.send.await_count == 2, (
        f"{exc_name} was fatal on the first attempt "
        f"({c.http_client.send.await_count} attempt(s)) — every other "
        f"transport fault on this client is retried once")
    assert any("[DONE]" in str(ch) for ch in chunks), (
        f"{exc_name} ended the stream without a [DONE] sentinel — the "
        f"client is left waiting on a connection that will never speak")
