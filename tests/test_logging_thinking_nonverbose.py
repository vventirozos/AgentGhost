"""Thinking lines in non-verbose mode (operator request 2026-07-08).

Previously `_emit_thinking` / `_flush_thinking` (the streaming closures in
handle_chat) returned early unless VERBOSE_MODE was set, so a non-verbose
launch showed every log line EXCEPT the model's thinking. The operator
wants the stream format untouched — thinking must simply flow through the
same pretty_log pipeline as everything else, which in non-verbose mode
means the standard LOG_TRUNCATE_LIMIT (60-char) truncation.
"""
import inspect
import re

from ghost_agent.utils import logging as glog


class TestEmitClosuresNotVerboseGated:
    """The emit/flush closures live inside handle_chat's streaming block;
    pin by source inspection that neither is gated on VERBOSE_MODE."""

    def _closure_src(self, name: str) -> str:
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        m = re.search(
            rf"def {name}\(.*?\n(?=\s{{24}}def |\s{{24}}stop_printing)",
            src, re.DOTALL)
        assert m, f"{name} not found"
        return m.group(0)

    def test_emit_thinking_not_gated(self):
        body = self._closure_src("_emit_thinking")
        # The old gate was `if not _glog.VERBOSE_MODE: return`. A comment
        # may legitimately mention VERBOSE_MODE — assert the CODE gate.
        assert "_glog.VERBOSE_MODE" not in body
        assert "pretty_log(\"thinking\"" in body or "pretty_log('thinking'" in body

    def test_flush_thinking_not_gated(self):
        body = self._closure_src("_flush_thinking")
        assert "_glog.VERBOSE_MODE" not in body


class TestPrettyLogPipelineForThinking:
    def test_nonverbose_thinking_is_never_truncated(self, capsys, monkeypatch):
        # Operator contract: thinking arrives IN FULL in every mode. The
        # call sites pass no_truncate=True, which exempts the line from
        # the 60-char budget while keeping the standard format (column
        # wrapping, redaction, newline flattening).
        monkeypatch.setattr(glog, "VERBOSE_MODE", False)
        monkeypatch.setattr(glog, "LOG_TRUNCATE_LIMIT", 60)
        long_block = ("The user wants me to research PostgreSQL and I should "
                      "start by checking the roadmap and then the mailing "
                      "lists for proposed patches and features END_MARKER")
        glog.pretty_log("thinking", long_block, icon=glog.Icons.BRAIN_THINK,
                        level="DEBUG", no_truncate=True)
        out = capsys.readouterr().out
        assert "thinking" in out
        assert "END_MARKER" in out  # full block survived non-verbose mode

    def test_other_lines_still_truncated_in_nonverbose(self, capsys, monkeypatch):
        # The exemption is per-call — everything else keeps the budget.
        monkeypatch.setattr(glog, "VERBOSE_MODE", False)
        monkeypatch.setattr(glog, "LOG_TRUNCATE_LIMIT", 60)
        glog.pretty_log("some tool", "y" * 300 + " END_MARKER")
        out = capsys.readouterr().out
        assert "END_MARKER" not in out
        assert "…" in out

    def test_verbose_thinking_line_full_length(self, capsys, monkeypatch):
        monkeypatch.setattr(glog, "VERBOSE_MODE", True)
        monkeypatch.setattr(glog, "LOG_TRUNCATE_LIMIT", 1000000)
        long_block = "x" * 300 + " END_MARKER"
        glog.pretty_log("thinking", long_block, icon=glog.Icons.BRAIN_THINK,
                        level="DEBUG", no_truncate=True)
        out = capsys.readouterr().out
        assert "END_MARKER" in out

    def test_call_sites_pass_no_truncate(self):
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        # All three thinking emissions must carry the exemption.
        assert src.count('pretty_log("thinking"') == 3
        assert src.count("no_truncate=True") >= 3
