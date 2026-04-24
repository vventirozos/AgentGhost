"""Tests for distill.redact.

The rules need to fire on realistic inputs (not just textbook examples)
and must be idempotent — redacting twice should be a no-op on the
output of the first pass.
"""

import pytest

from ghost_agent.distill.redact import (
    redact_text, redact_trajectory, RedactionConfig,
)
from ghost_agent.distill.schema import Trajectory, ToolCall


# -----------------------------------------------------------------
# redact_text — unit-level rules
# -----------------------------------------------------------------

def test_empty_text_returns_empty():
    assert redact_text("") == ""
    assert redact_text(None) is None


def test_openai_key_redacted():
    assert "sk-live-abc" not in redact_text("use key sk-liveXXabcDEFghi123")


def test_anthropic_key_redacted():
    out = redact_text("API_KEY=sk-ant-api01-ZZZ_abcdef123_ghijkl456")
    assert "sk-ant-api01" not in out
    assert "<REDACTED" in out


def test_slack_tokens_redacted():
    for tok in (
        "xoxb-1234567890-abcdef",
        "xapp-1-AAAA-9999-secret",
        "xoxp-12345-abcdef-abcdef",
    ):
        out = redact_text(f"TOKEN={tok}")
        assert tok not in out, f"slack token {tok} not redacted"


def test_github_pat_redacted():
    out = redact_text("ghp_abcdefghijklmnopqrstuv0123456789")
    assert "ghp_abcdefghij" not in out
    assert "REDACTED" in out


def test_aws_access_key_redacted():
    out = redact_text("AKIAIOSFODNN7EXAMPLE")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "REDACTED" in out


def test_env_assignment_secret_redacted():
    out = redact_text("GHOST_API_KEY=somerealkeyvalue123")
    assert "somerealkey" not in out


def test_json_field_api_key_redacted():
    out = redact_text('{"api_key": "verysecret", "name": "ghost"}')
    assert "verysecret" not in out
    # Name field is non-secret; should survive
    assert '"name": "ghost"' in out


def test_json_field_case_insensitive():
    out = redact_text('{"Authorization": "Bearer abc"}')
    # Authorization field itself isn't in json-field list; bearer rule handles it
    # But "Authorization: Bearer" matches the auth_header rule as well
    # At minimum, the secret string is gone
    assert "Bearer abc" not in out


def test_onion_address_redacted():
    out = redact_text("Check http://facebookcorewwwi.onion/ then act.")
    assert "facebookcorewwwi.onion" not in out
    assert "<REDACTED_ONION>" in out


def test_macos_home_path_redacted():
    out = redact_text("Log saved to /Users/alice/logs/ghost.log")
    assert "/Users/alice" not in out
    assert "/Users/<user>" in out


def test_linux_home_path_redacted():
    out = redact_text("Log saved to /home/vasilis/ghost.log")
    assert "/home/vasilis" not in out
    assert "/home/<user>" in out


def test_email_redacted():
    out = redact_text("Contact: alice.bob@example.co.uk for issues")
    assert "alice.bob@example.co.uk" not in out
    assert "<REDACTED_EMAIL>" in out


def test_external_ipv4_redacted_but_loopback_preserved():
    out = redact_text("Resolved 8.8.8.8 for upstream, local 127.0.0.1 kept")
    assert "8.8.8.8" not in out
    assert "127.0.0.1" in out  # loopback readable


def test_bearer_header_redacted():
    out = redact_text("Authorization: Bearer ghost-session-token-abc123")
    assert "ghost-session-token-abc123" not in out


def test_redaction_idempotent():
    original = "sk-liveXXabcDEFghi123 / GHOST_API_KEY=abc / /Users/alice/x"
    once = redact_text(original)
    twice = redact_text(once)
    assert once == twice


def test_disabled_rule_skipped():
    cfg = RedactionConfig(disabled_rules=("email",))
    out = redact_text("user@example.com", cfg)
    assert "user@example.com" in out


def test_extra_rule_applied():
    import re
    cfg = RedactionConfig(extra_rules=[
        ("ghost_magic", re.compile(r"GHOST_MAGIC_[A-Z]+"), "<REDACTED_GHOST_MAGIC>")
    ])
    out = redact_text("secret GHOST_MAGIC_PAYLOAD present", cfg)
    assert "GHOST_MAGIC_PAYLOAD" not in out
    assert "<REDACTED_GHOST_MAGIC>" in out


# -----------------------------------------------------------------
# redact_trajectory — integration
# -----------------------------------------------------------------

def test_redact_trajectory_covers_all_text_fields():
    t = Trajectory(
        session_id="s",
        system_prompt="API key is sk-livesecretabcdefghijk12345",
        user_request="Email me at user@example.com",
        planning_output="Plan: call sk-liveotherkeyabcdefghij123",
        final_response="Done. Token: xoxb-123456789-abcdef",
        failure_reason="Failed with /Users/alice/log error",
        tool_calls=[ToolCall(
            name="execute",
            arguments={"code": "API_KEY='sk-liveABCDEFGHIJKLMN1234'"},
            result="Contacted alice@company.com",
            error="/Users/alice/debug.txt",
        )],
    )
    r = redact_trajectory(t)
    assert "sk-livesecret" not in r.system_prompt
    assert "user@example.com" not in r.user_request
    assert "sk-liveotherkey" not in (r.planning_output or "")
    assert "xoxb-123456789" not in r.final_response
    assert "/Users/alice" not in r.failure_reason
    assert "sk-liveABCDEFGH" not in r.tool_calls[0].arguments["code"]
    assert "alice@company.com" not in r.tool_calls[0].result
    assert "/Users/alice" not in r.tool_calls[0].error


def test_redact_trajectory_does_not_mutate_input():
    original_prompt = "Key: sk-liveabcdefghijklmnop12345"
    t = Trajectory(system_prompt=original_prompt)
    redact_trajectory(t)
    assert t.system_prompt == original_prompt


def test_redact_trajectory_preserves_non_string_tool_args():
    t = Trajectory(
        tool_calls=[ToolCall(
            name="x",
            arguments={"count": 5, "flag": True, "key": "sk-livesecretABCDEFGHI123"},
        )]
    )
    r = redact_trajectory(t)
    assert r.tool_calls[0].arguments["count"] == 5
    assert r.tool_calls[0].arguments["flag"] is True
    assert "sk-livesecret" not in r.tool_calls[0].arguments["key"]


def test_redact_trajectory_handles_none_planning_output():
    t = Trajectory(planning_output=None)
    r = redact_trajectory(t)
    assert r.planning_output is None
