"""2026-07-22 LLM-stack review — core/agent.py stream + background fixes.

These live deep inside the ~1000-line handle_chat turn loop, so they are
pinned by source inspection (the repo's established pattern for turn-loop
internals — behavioral coverage rides the full-suite integration run):

- mid-stream fail-open: an upstream abort frame (`data: {"error": ...}`, no
  "choices") was silently dropped and the partial reply finalized as complete
  and fed to the verifier/memory. Now detected and folded into the truncation
  path so the turn is treated as cut off (continuation attempt).
- smart-memory extract + post-mortem carry a bounded timeout (was 1200s
  default → a wedged worker pinned the journal drain ~20 min).
- post-mortem transient failure re-queues instead of being dropped (the
  2026-07-09 requeue fix had covered only run_smart_memory_task).
- the idle `_aa_code_gen` twin uses max_tokens=4096 (parity with
  default_code_generator; 1024 truncated inline `python3 -c` programs).
"""
import inspect
import re

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import GhostAgent


def _handle_chat_src():
    return inspect.getsource(GhostAgent.handle_chat)


class TestMidStreamFailOpen:
    def test_error_frame_is_detected_in_the_turn_loop_drain(self):
        src = _handle_chat_src()
        # The turn-loop stream consumer must notice an error frame (no
        # "choices") and record it — not silently drop it.
        assert "stream_errored" in src
        assert '"error" in chunk_data and "choices" not in chunk_data' in src

    def test_errored_stream_folds_into_truncation_handling(self):
        src = _handle_chat_src()
        # An aborted stream must trigger the same truncation/continuation path
        # as a length-cap, so the partial isn't finalized as complete.
        assert re.search(
            r'stream_finish_reason == "length" or stream_errored', src)


class TestBackgroundCallTimeouts:
    def test_smart_memory_extract_is_bounded(self):
        src = inspect.getsource(GhostAgent.run_smart_memory_task)
        # The extract call carries a bounded timeout + its label.
        m = re.search(r'chat_completion\(payload, use_worker=True[^)]*\)', src)
        assert m is not None
        assert "timeout=90.0" in m.group(0)
        assert 'task_label="memory extract"' in m.group(0)

    def test_post_mortem_is_bounded_and_requeues(self):
        src = inspect.getsource(GhostAgent._execute_post_mortem)
        assert "timeout=90.0" in src
        # Transient upstream failure must raise the retryable signal (so the
        # drain re-queues) rather than being swallowed by the outer except.
        assert "_RetryableConsolidation" in src
        assert "is_upstream_transient" in src


class TestIdleCodeGenParity:
    def test_aa_code_gen_uses_4096(self):
        # The idle twin must match default_code_generator's cap.
        src = inspect.getsource(GhostAgent)
        # Find the _aa_code_gen closure body.
        idx = src.index("async def _aa_code_gen")
        window = src[idx: idx + 1400]
        assert '"max_tokens": 4096' in window
        assert '"max_tokens": 1024' not in window
