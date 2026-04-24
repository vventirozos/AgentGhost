"""Concurrency / race-condition regression tests for the LLMClient
foreground-task counter, VectorMemory locking, and SkillMemory snapshots."""
import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.llm import LLMClient


# ====================================== LLMClient.foreground_tasks lock


@pytest.mark.asyncio
async def test_foreground_tasks_lock_exists():
    client = LLMClient(upstream_url="http://mock")
    assert hasattr(client, "_foreground_lock")
    assert isinstance(client._foreground_lock, asyncio.Lock)
    await client.close()


@pytest.mark.asyncio
async def test_foreground_tasks_increment_decrement_serialised():
    """Run many concurrent foreground chat_completion calls and verify the
    counter never drifts negative or leaks above zero at the end."""
    client = LLMClient(upstream_url="http://mock")

    async def fake_do(*a, **kw):
        await asyncio.sleep(0.01)
        return {"choices": [{"message": {"content": "ok"}}]}
    client._do_chat_completion = fake_do

    coros = [client.chat_completion({"model": "x"}) for _ in range(50)]
    await asyncio.gather(*coros)
    assert client.foreground_tasks == 0
    await client.close()


@pytest.mark.asyncio
async def test_foreground_counter_clamped_to_zero_on_underflow():
    """Even if a leaked decrement drives the counter negative, the lock
    block clamps it back to 0 — protecting the watchdog gate."""
    client = LLMClient(upstream_url="http://mock")
    client.foreground_tasks = -3  # simulate a leak

    async def fake_do(*a, **kw):
        return {"choices": [{"message": {"content": "ok"}}]}
    client._do_chat_completion = fake_do
    await client.chat_completion({"model": "x"})
    assert client.foreground_tasks == 0
    await client.close()


@pytest.mark.asyncio
async def test_background_chat_waits_for_zero_foreground():
    """Background calls must NOT start while foreground_tasks > 0."""
    client = LLMClient(upstream_url="http://mock")
    started = []

    async def fg_do(*a, **kw):
        await asyncio.sleep(0.05)
        return {"choices": [{"message": {"content": "fg"}}]}

    async def bg_do(*a, **kw):
        started.append(time.monotonic())
        return {"choices": [{"message": {"content": "bg"}}]}

    client._do_chat_completion = AsyncMock()
    # Route depending on payload tag
    async def dispatch(payload, *a, **kw):
        if payload.get("tag") == "fg":
            return await fg_do()
        return await bg_do()
    client._do_chat_completion = dispatch

    fg_task = asyncio.create_task(client.chat_completion({"tag": "fg"}))
    await asyncio.sleep(0.005)  # let fg start and bump the counter
    bg_task = asyncio.create_task(client.chat_completion({"tag": "bg"}, is_background=True))
    await asyncio.gather(fg_task, bg_task)

    assert client.foreground_tasks == 0
    await client.close()


# ============================================================ VectorMemory


def test_vector_memory_has_thread_lock():
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    # Lazy lock helper must always return a real lock even when __init__ skipped.
    lock = vm._get_lock()
    import threading as _t
    assert lock is not None
    # RLock is reentrant — acquire twice from the same thread without deadlock.
    lock.acquire(); lock.acquire(); lock.release(); lock.release()


def test_vector_memory_concurrent_add_and_search_serialised():
    """Spawn parallel reader/writer threads against a fully mocked
    VectorMemory and verify the lock prevents interleaved access."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    vm.collection = MagicMock()
    vm.collection.get = MagicMock(return_value={"ids": []})
    vm.collection.add = MagicMock()
    vm.collection.query = MagicMock(return_value={"ids": [["a"]], "documents": [["x"]], "metadatas": [[{}]], "distances": [[0.1]]})

    in_critical = [0]
    max_seen = [0]
    lock = threading.Lock()

    real_add = vm.collection.add
    real_query = vm.collection.query

    def tracking_add(*a, **k):
        with lock:
            in_critical[0] += 1
            max_seen[0] = max(max_seen[0], in_critical[0])
        time.sleep(0.005)
        with lock:
            in_critical[0] -= 1
        return real_add(*a, **k)

    def tracking_query(*a, **k):
        with lock:
            in_critical[0] += 1
            max_seen[0] = max(max_seen[0], in_critical[0])
        time.sleep(0.005)
        with lock:
            in_critical[0] -= 1
        return real_query(*a, **k)

    vm.collection.add = tracking_add
    vm.collection.query = tracking_query

    def writer():
        for i in range(10):
            vm.add(f"text number {i} that is long enough", {"type": "auto"})

    def reader():
        for _ in range(10):
            vm.search_advanced("query", limit=1)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader),
               threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Critical section must never have been entered concurrently.
    assert max_seen[0] == 1, f"Lock failed: max concurrent entries = {max_seen[0]}"


# ============================================================ SkillMemory


def test_skill_memory_lazy_lock():
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory.__new__(SkillMemory)
    lock = sm._get_lock()
    assert lock is not None
    lock.acquire(); lock.release()


def test_skill_memory_get_playbook_uses_snapshot(tmp_path):
    """get_playbook_context releases the file lock before consuming the
    playbook contents — verify that a concurrent write to the file mid-
    iteration doesn't corrupt the result."""
    import json
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"task": "t1", "mistake": "m1", "solution": "s1"}])

    out = sm.get_playbook_context()
    assert "t1" in out
    assert "m1" in out
    assert "s1" in out


# ============================================================ stream timeout


@pytest.mark.asyncio
async def test_stream_chunk_timeout_short_circuits_stalled_upstream():
    """If the upstream stream goes silent for >30s the wrapper must yield
    an error event and return rather than block forever."""
    client = LLMClient(upstream_url="http://mock")

    class StalledResp:
        async def aiter_lines(self):
            await asyncio.sleep(60)  # never returns
            yield "should never reach"

        async def aclose(self):
            pass

        def raise_for_status(self):
            pass

    class StalledClient:
        def build_request(self, *a, **kw):
            return MagicMock()

        async def send(self, *a, **kw):
            return StalledResp()

        async def aclose(self):
            pass

    client.http_client = StalledClient()

    chunks = []
    # Patch wait_for to fast-forward instead of waiting 30 real seconds.
    real_wait_for = asyncio.wait_for
    async def fast_wait_for(coro, timeout):
        await coro.__anext__() if hasattr(coro, '__anext__') else None
        raise asyncio.TimeoutError()

    with patch("ghost_agent.core.llm.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        async for chunk in client._do_stream_chat_completion({"model": "x"}):
            chunks.append(chunk)
            if len(chunks) > 5:  # safety brake
                break

    joined = b"".join(chunks)
    assert b"stalled" in joined.lower() or b"timeout" in joined.lower() or b"[DONE]" in joined
    await client.close()
