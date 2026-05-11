"""Regression: post_mortem journal writes must respect ``--smart-memory``.

CLAUDE.md contract: "Memory writes are gated on ``--smart-memory`` /
``--no-memory``. New write paths must respect them or memories leak
unconditionally into the vector store."

The post_mortem path was unconditional. With ``--smart-memory 0.0``
(the documented "smart memory off" mode), the producer still queued
``post_mortem`` items. The journal consumer (phase 1) dispatched them
to ``_execute_post_mortem``, which calls
``skill_memory.learn_lesson`` — leaking auto-extracted lessons into
the playbook on every complex/failing turn. ``--no-memory`` did
shut it off via ``skill_memory=None``, but ``--smart-memory 0.0``
(the looser knob) leaked.

Fix: gate both producer sites (streaming and non-streaming) on
``self.context.args.smart_memory > 0.0`` to mirror the existing
``smart_memory`` journal-write gate.
"""
from pathlib import Path
import pytest


SRC = Path("src/ghost_agent/core/agent.py")


def test_post_mortem_producer_gated_in_streaming_path():
    """The streaming-path post_mortem producer must check
    smart_memory > 0.0 before appending to the journal."""
    src = SRC.read_text()
    # The streaming path uses `stream_tools_snapshot` and `full_content`.
    # Find the line that appends to journal in that vicinity.
    streaming_marker = "'post_mortem', {'user': last_user_content, 'tools': stream_tools_snapshot"
    idx = src.find(streaming_marker)
    assert idx != -1, "Streaming post_mortem producer not found"
    # Walk back ~6 lines and confirm a smart_memory > 0.0 gate is present.
    preceding = src[max(0, idx - 600):idx]
    assert "smart_memory > 0.0" in preceding, (
        "Streaming-path post_mortem journal append must be gated on "
        f"smart_memory > 0.0; preceding context was:\n{preceding!r}"
    )


def test_post_mortem_producer_gated_in_nonstreaming_path():
    """The non-streaming-path post_mortem producer must check
    smart_memory > 0.0 before appending to the journal."""
    src = SRC.read_text()
    # The non-streaming path uses `list(tools_run_this_turn)` and `final_ai_content`.
    nonstreaming_marker = "'post_mortem', {'user': last_user_content, 'tools': list(tools_run_this_turn)"
    idx = src.find(nonstreaming_marker)
    assert idx != -1, "Non-streaming post_mortem producer not found"
    preceding = src[max(0, idx - 600):idx]
    assert "smart_memory > 0.0" in preceding, (
        "Non-streaming-path post_mortem journal append must be gated on "
        f"smart_memory > 0.0; preceding context was:\n{preceding!r}"
    )


def test_post_mortem_gate_matches_smart_memory_branch():
    """Both producers (smart_memory branch + post_mortem branch) must
    use the SAME gate expression, so the user's mental model is
    "one knob controls all auto-memory writes". The smart_memory
    branch uses `self.context.args.smart_memory > 0.0` — the
    post_mortem branch must too."""
    src = SRC.read_text()
    # Count occurrences of the canonical gate. There should be at
    # least 4: two for smart_memory journal appends (streaming +
    # non-streaming) and two for post_mortem (streaming +
    # non-streaming).
    occurrences = src.count("self.context.args.smart_memory > 0.0")
    assert occurrences >= 4, (
        f"Expected at least 4 `smart_memory > 0.0` gate sites "
        f"(2 smart_memory + 2 post_mortem); found {occurrences}"
    )
