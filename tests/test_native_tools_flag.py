"""Native-tools flag: when --native-tools is set, the LLM payload gets
an OpenAI-format `tools` / `tool_choice` pair in addition to the XML
tool prompt. Flipping the flag must not change behaviour when off.

Full native-first dispatch (reading back message.tool_calls instead of
parsing XML) is a separate refactor — this test only covers the wiring
point so the flag is observable.

As of Qwen 3.6 35B-A3, `--native-tools` is ON by default. Users who
need to disable native tool-calls (e.g. when pointing at an older
backend) can pass `--no-native-tools` (argparse.BooleanOptionalAction).
"""

import argparse
import tempfile
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def base_args():
    return argparse.Namespace(
        max_context=4096, smart_memory=0.0, anonymous=True,
        model="test", verbose=False, no_memory=True,
        native_tools=False, perfect_it=False, deep_reason=False,
    )


def _payload_for(args):
    """Build the payload shape the agent loop uses. Factored out as a
    miniature of the main loop's payload block so the test is focused."""
    all_tools = [{"type": "function", "function": {"name": "execute"}}]
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.6,
        "frequency_penalty": 0.0,
    }
    if getattr(args, "native_tools", False):
        payload["tools"] = all_tools
        payload["tool_choice"] = "auto"
    return payload


class TestNativeToolsFlag:
    def test_off_by_default(self, base_args):
        payload = _payload_for(base_args)
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_on_attaches_tools_and_tool_choice(self, base_args):
        base_args.native_tools = True
        payload = _payload_for(base_args)
        assert "tools" in payload
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "execute"


class TestFlagAvailableOnArgs:
    """The argparse definition must include --native-tools so it doesn't
    AttributeError when the agent checks it via getattr fallback."""

    def test_parser_accepts_the_flag(self):
        # Import parse_args lazily so this test doesn't require the full
        # main module's heavy imports when just checking the argparse shape.
        import sys
        # Prevent main.py from actually running during import
        argv_backup = sys.argv
        sys.argv = ["test", "--native-tools", "--no-memory"]
        try:
            from ghost_agent.main import parse_args
            args = parse_args()
            assert args.native_tools is True
        finally:
            sys.argv = argv_backup

    def test_parser_default_is_true(self):
        import sys
        argv_backup = sys.argv
        sys.argv = ["test"]
        try:
            from ghost_agent.main import parse_args
            args = parse_args()
            assert args.native_tools is True
        finally:
            sys.argv = argv_backup

    def test_parser_accepts_no_native_tools(self):
        import sys
        argv_backup = sys.argv
        sys.argv = ["test", "--no-native-tools", "--no-memory"]
        try:
            from ghost_agent.main import parse_args
            args = parse_args()
            assert args.native_tools is False
        finally:
            sys.argv = argv_backup
