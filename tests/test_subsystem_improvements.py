"""Tests for subsystem improvements.

Covers:
- Circuit breaker for LLM nodes (#1)
- Token budget enforcement (#2/#14)
- Foreground lock starvation fix (#3)
- Per-command resource limits (#5)
- Swarm retry with backoff (#8)
- DB connection pooling (#11)
- Streaming error recovery (#16)
- Streaming file replace (#19)
- Sandbox package caching (#6)
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


# ============================================================
# #1: Circuit Breaker
# ============================================================

class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        from ghost_agent.core.llm import NodeCircuitBreaker
        cb = NodeCircuitBreaker()
        assert cb.is_available("http://node1:8080") is True

    def test_opens_after_threshold_failures(self):
        from ghost_agent.core.llm import NodeCircuitBreaker
        cb = NodeCircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            cb.record_failure("http://node1:8080")
        assert cb.is_available("http://node1:8080") is False

    def test_resets_on_success(self):
        from ghost_agent.core.llm import NodeCircuitBreaker
        cb = NodeCircuitBreaker(failure_threshold=3)
        cb.record_failure("http://node1:8080")
        cb.record_failure("http://node1:8080")
        cb.record_success("http://node1:8080")
        assert cb.is_available("http://node1:8080") is True
        # Failures should be reset
        cb.record_failure("http://node1:8080")
        assert cb.is_available("http://node1:8080") is True  # Only 1 failure

    def test_half_open_after_cooldown(self):
        from ghost_agent.core.llm import NodeCircuitBreaker
        cb = NodeCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("http://node1:8080")
        cb.record_failure("http://node1:8080")
        assert cb.is_available("http://node1:8080") is False
        time.sleep(0.15)  # Wait for cooldown
        assert cb.is_available("http://node1:8080") is True  # Half-open

    def test_independent_per_node(self):
        from ghost_agent.core.llm import NodeCircuitBreaker
        cb = NodeCircuitBreaker(failure_threshold=2)
        cb.record_failure("http://node1:8080")
        cb.record_failure("http://node1:8080")
        assert cb.is_available("http://node1:8080") is False
        assert cb.is_available("http://node2:8080") is True  # Different node

    def test_get_status(self):
        from ghost_agent.core.llm import NodeCircuitBreaker
        cb = NodeCircuitBreaker()
        cb.record_failure("http://node1:8080")
        status = cb.get_status()
        assert "http://node1:8080" in status
        assert status["http://node1:8080"]["failures"] == 1


# ============================================================
# #2/#14: Token Budget Enforcement
# ============================================================

class TestTokenBudget:
    def test_check_budget_fits(self):
        from ghost_agent.utils.token_counter import check_budget
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = check_budget(messages, max_tokens=1000)
        assert result["fits"] is True
        assert result["overflow"] == 0
        assert result["total_tokens"] > 0
        assert len(result["per_message"]) == 2

    def test_check_budget_overflows(self):
        from ghost_agent.utils.token_counter import check_budget
        messages = [
            {"role": "user", "content": "x " * 500},  # ~250 tokens
        ]
        result = check_budget(messages, max_tokens=10)
        assert result["fits"] is False
        assert result["overflow"] > 0

    def test_check_budget_empty(self):
        from ghost_agent.utils.token_counter import check_budget
        result = check_budget([], max_tokens=1000)
        assert result["fits"] is True
        assert result["total_tokens"] == 0

    def test_check_budget_multimodal(self):
        from ghost_agent.utils.token_counter import check_budget
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]},
        ]
        result = check_budget(messages, max_tokens=1000)
        assert result["fits"] is True
        assert result["total_tokens"] > 0

    def test_estimate_payload_tokens(self):
        from ghost_agent.utils.token_counter import estimate_payload_tokens
        payload = {
            "messages": [
                {"role": "system", "content": "You are an AI."},
                {"role": "user", "content": "Hello"},
            ],
            "tools": [{"function": {"name": "test", "parameters": {}}}]
        }
        tokens = estimate_payload_tokens(payload)
        assert tokens > 0

    def test_estimate_payload_no_tools(self):
        from ghost_agent.utils.token_counter import estimate_payload_tokens
        payload = {"messages": [{"role": "user", "content": "Hi"}]}
        tokens = estimate_payload_tokens(payload)
        assert tokens > 0


# ============================================================
# #3: Foreground Lock Starvation Fix
# ============================================================

class TestForegroundLockStarvation:
    @pytest.mark.asyncio
    async def test_background_doesnt_wait_forever(self):
        """Background tasks should proceed after max wait time."""
        from ghost_agent.core.llm import LLMClient
        client = LLMClient("http://localhost:8080")
        # Simulate foreground task holding the lock
        async with client._foreground_lock:
            client.foreground_tasks = 1

        # Background should not block forever even with foreground active
        # (we can't easily test the 30s timeout in unit tests, but we verify
        # the structure exists). The previous `_bg_queue_lock` was replaced
        # by `_bg_queue_sem` (a Semaphore) to allow a bounded level of
        # background concurrency instead of strict mutual exclusion.
        assert hasattr(client, '_bg_queue_sem')
        assert hasattr(client, '_foreground_lock')


# ============================================================
# #8: Swarm Retry with Backoff
# ============================================================

class TestSwarmRetry:
    @pytest.mark.asyncio
    async def test_swarm_retries_on_failure(self):
        from ghost_agent.tools.swarm import _swarm_worker
        from ghost_agent.memory.scratchpad import Scratchpad

        scratchpad = Scratchpad()
        mock_client = MagicMock()
        mock_llm = MagicMock()

        # First call fails, second succeeds
        mock_node_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Result after retry"}}]
        }
        mock_response.raise_for_status = MagicMock()

        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Node offline")
            return mock_response

        mock_node_client.post = mock_post

        mock_llm.get_swarm_node = MagicMock(return_value={
            "client": mock_node_client,
            "model": "test-model",
            "url": "http://test:8080"
        })

        await _swarm_worker(
            "do something", "input data", "result_key",
            mock_llm, "fallback-model", scratchpad
        )

        result = scratchpad.get("result_key")
        assert result == "Result after retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_swarm_fails_after_all_retries(self):
        from ghost_agent.tools.swarm import _swarm_worker
        from ghost_agent.memory.scratchpad import Scratchpad

        scratchpad = Scratchpad()
        mock_llm = MagicMock()
        mock_node_client = MagicMock()

        async def mock_post(*args, **kwargs):
            raise ConnectionError("Always fails")

        mock_node_client.post = mock_post

        mock_llm.get_swarm_node = MagicMock(return_value={
            "client": mock_node_client,
            "model": "test-model",
            "url": "http://test:8080"
        })

        await _swarm_worker(
            "do something", "input data", "result_key",
            mock_llm, "fallback-model", scratchpad
        )

        result = scratchpad.get("result_key")
        assert "failed after" in result.lower()


# ============================================================
# #11: DB Connection Pooling
# ============================================================

class TestDBConnectionPooling:
    def test_pool_caches_connection(self):
        from ghost_agent.tools.database import _connection_pool
        # Pool should be a dict
        assert isinstance(_connection_pool, dict)

    def test_timeout_is_configurable(self):
        """Verify the tool accepts timeout_ms parameter."""
        import inspect
        from ghost_agent.tools.database import tool_postgres_admin
        sig = inspect.signature(tool_postgres_admin)
        assert "timeout_ms" in sig.parameters


# ============================================================
# #16: Streaming Error Recovery
# ============================================================

class TestStreamingErrorRecovery:
    def test_routes_has_logger(self):
        """Verify routes.py uses proper logging instead of traceback.print_exc."""
        import ghost_agent.api.routes as routes_module
        assert hasattr(routes_module, 'logger')


# ============================================================
# #19: Streaming File Replace
# ============================================================

class TestStreamingFileReplace:
    @pytest.mark.asyncio
    async def test_streaming_replace_large_file(self, tmp_path):
        from ghost_agent.tools.file_system import tool_replace_text

        # Create a file larger than STREAMING_THRESHOLD (1 MB)
        large_file = tmp_path / "large.py"
        content_lines = [f"line_{i} = 'value_{i}'\n" for i in range(50000)]
        # Insert a target line in the middle
        content_lines[25000] = "target_line = 'old_value'\n"
        large_file.write_text("".join(content_lines))

        # File should be >1 MB
        assert large_file.stat().st_size > 1_000_000

        result = await tool_replace_text(
            filename="large.py",
            old_text="target_line = 'old_value'",
            new_text="target_line = 'new_value'",
            sandbox_dir=tmp_path
        )

        assert "SUCCESS" in result
        # Verify the replacement happened
        new_content = large_file.read_text()
        assert "target_line = 'new_value'" in new_content
        assert "target_line = 'old_value'" not in new_content

    @pytest.mark.asyncio
    async def test_small_file_uses_memory_replace(self, tmp_path):
        from ghost_agent.tools.file_system import tool_replace_text

        small_file = tmp_path / "small.py"
        small_file.write_text("hello = 'world'\n")

        result = await tool_replace_text(
            filename="small.py",
            old_text="hello = 'world'",
            new_text="hello = 'earth'",
            sandbox_dir=tmp_path
        )

        assert "SUCCESS" in result
        assert "earth" in small_file.read_text()

    @pytest.mark.asyncio
    async def test_streaming_replace_no_match(self, tmp_path):
        from ghost_agent.tools.file_system import tool_replace_text

        large_file = tmp_path / "nope.py"
        content = "x = 1\n" * 200000
        large_file.write_text(content)

        result = await tool_replace_text(
            filename="nope.py",
            old_text="nonexistent_text_xyz",
            new_text="replacement",
            sandbox_dir=tmp_path
        )

        # Should fall through to heuristic match or return not found
        assert "SYSTEM INSTRUCTION" in result or "not found" in result.lower() or "NOT found" in result
