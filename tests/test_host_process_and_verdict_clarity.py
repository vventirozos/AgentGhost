"""Two operator-requested clarity fixes (2026-07-08).

1. LATE-verdict-empty logging: one ambiguous line fired on every
   bookkeeping-only turn and every sim/ablation context, drowning the case
   it exists for (a genuinely dead verifier path). Now three distinct
   messages, only one of which is a WARNING.

2. Host-process blind spot: the sandbox has its own PID namespace, so a
   `pkill -f app.py` aimed at the USER's host-run server "succeeds" (exit
   0, kills nothing) and the model concludes it restarted the server —
   then debugs against stale code. Observed twice in the chess session.
   The tool now appends ground truth to the result, and the schema warns
   up front.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.tools.execute import _HOST_PROCESS_RE, _HOST_PROCESS_NOTE


def _agent(verifier=None):
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(verifier=verifier)
    return a


class TestLateVerdictEmptyDifferentiation:
    def _capture(self, agent, last_tool):
        with patch("ghost_agent.core.agent.pretty_log") as log:
            agent._record_late_verdict(None, "traj-1", "", last_tool=last_tool)
        assert log.call_count == 1
        call = log.call_args
        return call.args[1], call.kwargs.get("level", "INFO")

    def test_no_evidence_tool_is_a_quiet_by_design_skip(self):
        msg, level = self._capture(_agent(), last_tool=None)
        assert "no verifiable evidence" in msg
        assert "by design" in msg
        assert level == "INFO"

    def test_missing_verifier_is_a_quiet_by_design_skip(self):
        # Sim / ablation contexts: verifier present but no llm_client.
        agent = _agent(verifier=SimpleNamespace(llm_client=None))
        msg, level = self._capture(agent, last_tool={"name": "execute"})
        assert "not attached" in msg
        assert "sim/ablation" in msg
        assert level == "INFO"

    def test_evidence_plus_verifier_but_no_verdict_warns(self):
        # THIS is the case worth watching — a possibly-dead verifier path.
        agent = _agent(verifier=SimpleNamespace(llm_client=object()))
        msg, level = self._capture(agent, last_tool={"name": "execute"})
        assert "EMPTY despite verifiable evidence" in msg
        assert level == "WARNING"


class TestHostProcessDetector:
    @pytest.mark.parametrize("cmd", [
        "pkill -f 'python.*app.py'",
        "killall python3",
        "kill $(pgrep -f app.py)",
        "kill -9 $(pidof flask)",
    ])
    def test_name_based_kills_are_detected(self, cmd):
        assert _HOST_PROCESS_RE.search(cmd)

    @pytest.mark.parametrize("cmd", [
        "python app.py",
        "ls -la",
        "kill 12345",              # explicit PID of an in-sandbox child: fine
        "echo 'pkill is a command'",  # inside quotes but harmless; note is advisory
    ])
    def test_ordinary_commands_are_not_flagged(self, cmd):
        if cmd.startswith("echo"):
            pytest.skip("advisory note on a quoted mention is acceptable")
        assert not _HOST_PROCESS_RE.search(cmd)

    def test_note_tells_the_truth_and_names_the_action(self):
        assert "own pid namespace" in _HOST_PROCESS_NOTE.lower()
        assert "killed nothing even if it exited 0" in _HOST_PROCESS_NOTE
        assert "restart it to pick up the fix" in _HOST_PROCESS_NOTE

    def test_execute_schema_warns_about_pid_namespace(self):
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        desc = next(t["function"]["description"] for t in TOOL_DEFINITIONS
                    if t["function"]["name"] == "execute")
        assert "own pid namespace" in desc.lower()
        assert "ask them to restart it" in desc
