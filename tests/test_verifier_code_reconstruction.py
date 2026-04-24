"""Tests for the verifier gate's executed-code reconstruction.

Before the fix, `handle_chat`'s verifier gate called
    verifier.verify_code_output(code=tool_name, ...)

which passed the literal string "execute" as the "code" to audit.
With nothing to reason about, the verifier LLM hallucinated plausible-
sounding reasons the output didn't match the intent — emitting
REFUTED at 85%+ confidence on CORRECT turns. A real observed example:

    user:  "Write probe.py: browser='Chrome'; print(f'Using {browser}')"
    agent: <correctly ran it, output 'Using Chrome'>
    verifier: "Missing expected output 'Using Chrome'; Output appears to
               be a directory listing rather than script execution result"

The user then saw a "Verifier note: …" banner contradicting their
perfectly-correct answer. These tests pin the helper
`_reconstruct_executed_code` so the bug can't silently regress.
"""

from ghost_agent.core.agent import _reconstruct_executed_code


# ------------------------------------------------------------------
# Happy path: recover content / code / command args
# ------------------------------------------------------------------

def test_recovers_content_from_assistant_tool_call():
    tool_msg = {"role": "tool", "tool_call_id": "call_1", "name": "execute",
                "content": "EXIT 0\nHello"}
    messages = [
        {"role": "user", "content": "write code"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "execute",
                "arguments": '{"filename": "h.py", "content": "print(\\"Hello\\")"}',
            },
        }]},
        tool_msg,
    ]
    code = _reconstruct_executed_code(messages, tool_msg)
    assert code == 'print("Hello")'


def test_recovers_code_argument_alias():
    """Some tool-call schemas use `code` instead of `content`."""
    tool_msg = {"role": "tool", "tool_call_id": "c1"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "c1",
            "function": {
                "name": "execute",
                "arguments": '{"code": "print(1+1)"}',
            },
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == "print(1+1)"


def test_recovers_shell_command_argument():
    """When the agent issues a shell command (not a python file), the
    `command` arg should be recovered so the verifier audits the shell
    invocation — not the string "execute"."""
    tool_msg = {"role": "tool", "tool_call_id": "sh1"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "sh1",
            "function": {
                "name": "execute",
                "arguments": '{"command": "ls -la /tmp"}',
            },
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == "ls -la /tmp"


def test_prefers_content_over_command_when_both_present():
    """If both `content` and `command` somehow coexist, content (the
    actual code) takes precedence — shell commands are a fallback for
    no-file invocations."""
    tool_msg = {"role": "tool", "tool_call_id": "b1"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "b1",
            "function": {
                "name": "execute",
                "arguments": '{"content": "print(42)", "command": "echo fallback"}',
            },
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == "print(42)"


def test_accepts_dict_arguments():
    """Some normalizers hand us `arguments` already parsed as a dict
    instead of a JSON string. Must work either way."""
    tool_msg = {"role": "tool", "tool_call_id": "d1"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "d1",
            "function": {
                "name": "execute",
                "arguments": {"content": "print('dict form')"},
            },
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == "print('dict form')"


def test_malformed_json_arguments_fall_back_to_raw_string():
    """If arguments isn't valid JSON, we still hand the raw string to
    the verifier rather than an empty slot. Partial audit beats no
    audit."""
    tool_msg = {"role": "tool", "tool_call_id": "m1"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "m1",
            "function": {
                "name": "execute",
                "arguments": '{"content": "print(1)"  # not valid json',
            },
        }]},
        tool_msg,
    ]
    out = _reconstruct_executed_code(messages, tool_msg)
    assert "print(1)" in out


# ------------------------------------------------------------------
# Graceful degradation
# ------------------------------------------------------------------

def test_returns_empty_when_no_matching_id():
    tool_msg = {"role": "tool", "tool_call_id": "nope"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "other_id",
            "function": {"name": "execute",
                         "arguments": '{"content": "irrelevant"}'},
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == ""


def test_returns_empty_when_tool_msg_has_no_call_id():
    tool_msg = {"role": "tool", "name": "execute"}  # missing tool_call_id
    messages = [{"role": "assistant", "tool_calls": [{"id": "x"}]}, tool_msg]
    assert _reconstruct_executed_code(messages, tool_msg) == ""


def test_returns_empty_when_messages_empty():
    assert _reconstruct_executed_code([], {"tool_call_id": "x"}) == ""
    assert _reconstruct_executed_code(None, {"tool_call_id": "x"}) == ""


def test_returns_empty_when_tool_msg_none():
    assert _reconstruct_executed_code([{"role": "user"}], None) == ""


def test_caps_oversize_content():
    """Verifier can't take megabyte code blobs. Cap at 4000 chars."""
    big = "x = 1\n" * 10000
    tool_msg = {"role": "tool", "tool_call_id": "big"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "big",
            "function": {"name": "execute",
                         "arguments": '{"content": ' + __import__("json").dumps(big) + '}'},
        }]},
        tool_msg,
    ]
    out = _reconstruct_executed_code(messages, tool_msg)
    assert len(out) <= 4000


def test_walks_back_past_unrelated_messages():
    """The tool_call id match happens on the assistant BEFORE the tool
    msg; system/user messages in between are ignored."""
    tool_msg = {"role": "tool", "tool_call_id": "walk1"}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "tool_calls": [{
            "id": "walk1",
            "function": {"name": "execute",
                         "arguments": '{"content": "print(\\"far back\\")"}'},
        }]},
        {"role": "tool", "tool_call_id": "other_tool", "content": "unrelated"},
        {"role": "user", "content": "intermediate message"},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == 'print("far back")'


def test_picks_most_recent_matching_assistant_when_id_collides():
    """Unlikely in practice (tool_call_ids are UUIDs) but defensive:
    walking REVERSED means the most recent matching assistant wins."""
    tool_msg = {"role": "tool", "tool_call_id": "c1"}
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "c1",
            "function": {"name": "execute",
                         "arguments": '{"content": "old"}'},
        }]},
        {"role": "assistant", "tool_calls": [{
            "id": "c1",
            "function": {"name": "execute",
                         "arguments": '{"content": "new"}'},
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == "new"


def test_non_dict_messages_skipped_gracefully():
    tool_msg = {"role": "tool", "tool_call_id": "c1"}
    messages = [
        "not a dict",
        None,
        {"role": "assistant", "tool_calls": [{
            "id": "c1",
            "function": {"name": "execute",
                         "arguments": '{"content": "survived"}'},
        }]},
        tool_msg,
    ]
    assert _reconstruct_executed_code(messages, tool_msg) == "survived"
