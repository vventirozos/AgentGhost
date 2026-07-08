"""Fixes for the guard-box incident (2026-07-08, chess request 04).

Two guards composed to leave the model ZERO legal paths to write app.py:
the pre-flight guard keyed on (tool, target) blocked the correct
replace→write recovery as a "repeat", and the egress guard blocked the
heredoc fallback because the FILE CONTENT legitimately mentions the
agent's API URL. Neither block advanced any loop budget, so the model
spun ~6 minutes (3 × ~80 s of full-file generation, each discarded).
"""
import inspect

import pytest

from ghost_agent.core.triggers import RecentFailureGuard
from ghost_agent.tools.execute import (
    _AGENT_PORT_PROBE_RE,
    _command_probes_agent_port,
)


class TestOperationAwareGuardKey:
    def test_different_operation_is_not_a_repeat(self):
        # The chess trap: replace failed twice; write must stay legal.
        g = RecentFailureGuard(repeat_threshold=2)
        err = "Error: You must specify the exact 'content' to be replaced."
        g.record("file_system", "app.py", err, "replace")
        g.record("file_system", "app.py", err, "replace")
        assert g.would_repeat("file_system", "app.py", "replace") is not None
        assert g.would_repeat("file_system", "app.py", "write") is None

    def test_same_operation_same_error_still_blocked(self):
        g = RecentFailureGuard(repeat_threshold=2)
        g.record("web_search", "foo", "ERROR: zero results", "")
        g.record("web_search", "foo", "ERROR: zero results", "")
        assert g.would_repeat("web_search", "foo", "") is not None

    def test_different_error_on_same_op_not_blocked(self):
        g = RecentFailureGuard(repeat_threshold=2)
        g.record("file_system", "app.py", "Error: A", "replace")
        g.record("file_system", "app.py", "Error: B", "replace")
        assert g.would_repeat("file_system", "app.py", "replace") is None


class TestBlockBudgetWiring:
    def test_dispatch_passes_operation_and_budget_forces_final(self):
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        # would_repeat receives the operation discriminator.
        idx = src.index("would_repeat(")
        assert "_pf_op" in src[idx:idx + 120]
        # record() feeds the op back.
        assert "record(fname, ptarget, str_res, ptool_op)" in src
        # Two blocks in one request force a final reply.
        assert "preflight_blocks_this_request >= 2" in src
        bidx = src.index("preflight_blocks_this_request >= 2")
        assert "force_final_response = True" in src[bidx:bidx + 800]
        # The block message names a LEGAL alternative.
        assert "operation='write' with the" in src


class TestEgressGuardContext:
    def test_heredoc_file_write_with_agent_url_is_allowed(self):
        # The exact chess-session false positive: writing app.py whose
        # content calls the Ghost API.
        cmd = (
            "cat > app.py <<'EOF'\n"
            "import urllib.request\n"
            "GHOST_API = 'http://127.0.0.1:8000/api/chat'\n"
            "req = urllib.request.Request(GHOST_API)\n"
            "EOF"
        )
        assert not _command_probes_agent_port(cmd)

    def test_direct_curl_probe_still_blocked(self):
        assert _command_probes_agent_port("curl -s http://127.0.0.1:8000/api/health")
        assert _command_probes_agent_port("wget -qO- localhost:8088/v1/models")

    def test_inline_python_client_probe_still_blocked(self):
        cmd = ("python3 -c 'import urllib.request; "
               "urllib.request.urlopen(\"http://127.0.0.1:8000\")'")
        assert _command_probes_agent_port(cmd)

    def test_plain_text_mention_without_client_is_allowed(self):
        # echo/sed on text that mentions the URL proves nothing.
        assert not _command_probes_agent_port(
            "echo 'API at http://127.0.0.1:8000' > notes.txt")

    def test_probe_after_heredoc_still_blocked(self):
        # Sneaking a probe AFTER a heredoc must not be shadowed by the strip.
        cmd = (
            "cat > f.txt <<'EOF'\nhello\nEOF\n"
            "curl http://127.0.0.1:8000/api/health"
        )
        assert _command_probes_agent_port(cmd)

    def test_port_regex_unchanged_for_content_path(self):
        # Inline content keeps the strict any-match rule — the caller uses
        # _AGENT_PORT_PROBE_RE directly on `content`.
        assert _AGENT_PORT_PROBE_RE.search("x = 'http://127.0.0.1:8000'")


class TestReplaceErrorSteer:
    @pytest.mark.asyncio
    async def test_contentless_replace_names_the_write_escape(self, tmp_path):
        from ghost_agent.tools.file_system import tool_replace_text
        out = await tool_replace_text("app.py", "", "new text", tmp_path)
        assert "operation='write'" in out
        assert "do NOT retry 'replace'" in out
