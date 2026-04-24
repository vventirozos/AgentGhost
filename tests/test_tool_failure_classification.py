"""Tests for tool failure classification (#9).

Verifies that:
- classify_tool_failure correctly categorizes error types
- Retry logic respects max retries and failure class
- format_failure_context produces appropriate messages per class
"""

import pytest
from ghost_agent.tools.tool_failure import (
    FailureClass,
    classify_tool_failure,
    should_retry,
    get_retry_delay,
    format_failure_context,
    MAX_RETRIES,
)


class TestClassifyToolFailure:
    @pytest.mark.parametrize("error_text,expected_class", [
        ("Connection timed out after 30s", FailureClass.RETRYABLE),
        ("timeout waiting for response", FailureClass.RETRYABLE),
        ("Rate limit exceeded, try again later", FailureClass.RETRYABLE),
        ("Too many requests (429)", FailureClass.RETRYABLE),
        ("Connection reset by peer", FailureClass.RETRYABLE),
        ("ECONNREFUSED", FailureClass.RETRYABLE),
        ("sandbox busy, container starting", FailureClass.RETRYABLE),
        ("503 Service Unavailable", FailureClass.RETRYABLE),
        ("temporarily unavailable", FailureClass.RETRYABLE),
    ])
    def test_retryable_errors(self, error_text, expected_class):
        fc, _ = classify_tool_failure(error_text)
        assert fc == expected_class

    @pytest.mark.parametrize("error_text,expected_class", [
        ("Permission denied: /etc/shadow", FailureClass.FATAL),
        ("Access denied for user", FailureClass.FATAL),
        ("tool 'invalid_tool' not found", FailureClass.FATAL),
        ("SYSTEM ERROR: 'name' is MANDATORY", FailureClass.FATAL),
        ("Invalid argument: expected string", FailureClass.FATAL),
        ("Authentication failed: bad token", FailureClass.FATAL),
        ("401 Unauthorized", FailureClass.FATAL),
    ])
    def test_fatal_errors(self, error_text, expected_class):
        fc, _ = classify_tool_failure(error_text)
        assert fc == expected_class

    @pytest.mark.parametrize("error_text,expected_class", [
        ("AssertionError: expected 5 got 3", FailureClass.DIAGNOSTIC),
        ("RuntimeError: division by zero", FailureClass.DIAGNOSTIC),
        ("SyntaxError: invalid syntax line 5", FailureClass.DIAGNOSTIC),
        ("TypeError: unsupported operand type", FailureClass.DIAGNOSTIC),
        ("Traceback (most recent call last):\n  File...", FailureClass.DIAGNOSTIC),
        ("EXIT CODE: 1", FailureClass.DIAGNOSTIC),
        ("FileNotFoundError: No such file 'data.csv'", FailureClass.DIAGNOSTIC),
        ("ImportError: No module named 'pandas'", FailureClass.DIAGNOSTIC),
    ])
    def test_diagnostic_errors(self, error_text, expected_class):
        fc, _ = classify_tool_failure(error_text)
        assert fc == expected_class

    def test_unknown_error(self):
        fc, _ = classify_tool_failure("Something went wrong but no patterns match")
        assert fc == FailureClass.UNKNOWN

    def test_empty_error(self):
        fc, _ = classify_tool_failure("")
        assert fc == FailureClass.UNKNOWN

    def test_none_error(self):
        fc, _ = classify_tool_failure(None)
        assert fc == FailureClass.UNKNOWN


class TestShouldRetry:
    def test_retryable_under_max(self):
        assert should_retry(FailureClass.RETRYABLE, 0) is True
        assert should_retry(FailureClass.RETRYABLE, 1) is True
        assert should_retry(FailureClass.RETRYABLE, 2) is True

    def test_retryable_at_max(self):
        assert should_retry(FailureClass.RETRYABLE, MAX_RETRIES) is False

    def test_fatal_never_retries(self):
        assert should_retry(FailureClass.FATAL, 0) is False

    def test_diagnostic_never_retries(self):
        assert should_retry(FailureClass.DIAGNOSTIC, 0) is False

    def test_unknown_never_retries(self):
        assert should_retry(FailureClass.UNKNOWN, 0) is False


class TestGetRetryDelay:
    def test_exponential_backoff(self):
        d0 = get_retry_delay(0, base_delay=1.0)
        d1 = get_retry_delay(1, base_delay=1.0)
        d2 = get_retry_delay(2, base_delay=1.0)
        # Each subsequent delay should be roughly double (with jitter)
        assert d1 > d0
        assert d2 > d1

    def test_respects_max_delay(self):
        d = get_retry_delay(100, base_delay=1.0, max_delay=30.0)
        # Should not exceed max_delay + 25% jitter
        assert d <= 30.0 * 1.25

    def test_positive_delays(self):
        for attempt in range(5):
            d = get_retry_delay(attempt)
            assert d > 0


class TestFormatFailureContext:
    def test_retryable_format(self):
        msg = format_failure_context("timeout", FailureClass.RETRYABLE, "execute")
        assert "TRANSIENT" in msg
        assert "execute" in msg

    def test_fatal_format(self):
        msg = format_failure_context("permission denied", FailureClass.FATAL, "file_system")
        assert "PERMANENT" in msg
        assert "do NOT retry" in msg

    def test_diagnostic_format(self):
        msg = format_failure_context("TypeError: bad args", FailureClass.DIAGNOSTIC)
        assert "DIAGNOSTIC" in msg
        assert "analyze and fix" in msg

    def test_unknown_format(self):
        msg = format_failure_context("weird error", FailureClass.UNKNOWN)
        assert "ERROR" in msg

    def test_truncates_long_errors(self):
        long_error = "x" * 10000
        msg = format_failure_context(long_error, FailureClass.DIAGNOSTIC)
        assert len(msg) < 3000
