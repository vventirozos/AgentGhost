"""Tests for the Progressive Context Compression module."""

import pytest

from ghost_agent.core.context_manager import ContextManager


@pytest.fixture
def cm():
    return ContextManager(max_tokens=1000)


def _make_messages(n_tool_msgs=5, tool_content_len=500):
    """Build a realistic message list with system + user + assistant + tool msgs."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this CSV file."},
    ]
    for i in range(n_tool_msgs):
        msgs.append({
            "role": "assistant",
            "content": f"Let me analyze step {i}." + "x" * 200,
        })
        msgs.append({
            "role": "tool",
            "name": f"tool_{i}",
            "content": f"Line 1: header\nLine 2: data\n" + f"data line\n" * 50 + f"Total: {i * 10}",
        })
    msgs.append({"role": "user", "content": "What did you find?"})
    return msgs


class TestContextManager:
    def test_no_compression_under_threshold(self):
        cm = ContextManager(max_tokens=100000)
        msgs = _make_messages(n_tool_msgs=2)
        result = cm.compress_if_needed(msgs)
        assert result == msgs
        assert cm.compression_level == 0

    def test_l1_compression_summarizes_tool_outputs(self):
        cm = ContextManager(max_tokens=200)  # Force compression
        msgs = _make_messages(n_tool_msgs=10, tool_content_len=1000)
        result = cm.compress_if_needed(msgs)
        # Should have fewer total chars than original
        original_chars = sum(len(m.get("content", "")) for m in msgs)
        compressed_chars = sum(len(m.get("content", "")) for m in result)
        assert compressed_chars <= original_chars
        # System messages should survive
        assert any(m["role"] == "system" for m in result)

    def test_emergency_prune_l4(self):
        cm = ContextManager(max_tokens=50)  # Extreme constraint
        msgs = _make_messages(n_tool_msgs=20, tool_content_len=2000)
        result = cm.compress_if_needed(msgs)
        # Should keep system + last user + maybe last tool
        assert any(m["role"] == "system" for m in result)
        assert any(m["role"] == "user" for m in result)
        assert len(result) <= 4

    def test_recent_turns_preserved(self):
        cm = ContextManager(max_tokens=300)
        msgs = _make_messages(n_tool_msgs=8)
        result = cm.compress_if_needed(msgs)
        # Last few messages should be intact
        last_user = [m for m in result if m["role"] == "user"]
        assert len(last_user) >= 1

    def test_compression_level_property(self, cm):
        assert cm.compression_level == 0
        msgs = _make_messages(n_tool_msgs=20, tool_content_len=2000)
        cm.compress_if_needed(msgs)
        assert cm.compression_level > 0

    def test_get_stats(self, cm):
        stats = cm.get_stats()
        assert "compression_level" in stats
        assert "max_tokens" in stats

    def test_empty_messages(self, cm):
        result = cm.compress_if_needed([])
        assert result == []

    def test_only_system_message(self, cm):
        msgs = [{"role": "system", "content": "system prompt"}]
        result = cm.compress_if_needed(msgs)
        assert result == msgs

    def test_truncate_with_summary(self):
        text = "A" * 2000
        result = ContextManager._truncate_with_summary(text, max_len=800)
        assert len(result) < len(text)
        assert "compressed" in result

    def test_truncate_short_text_unchanged(self):
        text = "short text"
        assert ContextManager._truncate_with_summary(text) == text

    def test_summarize_tool_output_small_preserved(self, cm):
        msg = {"role": "tool", "name": "test", "content": "small output"}
        result = cm._summarize_tool_output(msg)
        assert result["content"] == "small output"

    def test_summarize_tool_output_large_compressed(self, cm):
        lines = ["Line header"] + [f"data row {i}" for i in range(50)] + ["error: something failed", "Total: 42"]
        msg = {"role": "tool", "name": "test", "content": "\n".join(lines)}
        result = cm._summarize_tool_output(msg)
        assert "compressed" in result["content"] or len(result["content"]) < len(msg["content"])

    def test_emergency_prune_truncates_large_tool_output(self):
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "tool", "name": "t", "content": "x" * 5000},
        ]
        result = ContextManager._emergency_prune(msgs)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        if tool_msgs:
            assert len(tool_msgs[0]["content"]) <= 1100  # 1000 + truncation marker

    def test_custom_token_estimator(self):
        def always_huge(msgs):
            return 999999

        cm = ContextManager(max_tokens=1000, token_estimator=always_huge)
        msgs = _make_messages(n_tool_msgs=3)
        result = cm.compress_if_needed(msgs)
        assert cm.compression_level == 4  # Emergency

    def test_list_content_estimation(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
        estimate = ContextManager._default_estimate(msgs)
        assert estimate > 0
