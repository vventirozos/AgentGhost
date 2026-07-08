"""Tests for the post-build gates (2026-07-08, chess-session post-mortem).

The chess build passed every mechanical check while (a) violating the
user's core constraint (a heuristic engine where "Ghost plays directly"
was demanded) and (b) shipping three crash-on-first-touch bugs no endpoint
smoke would have missed. constraint_gate audits the artifact against the
stored constraints; smoke_gate py_compiles written .py files and sweeps
Flask routes via test_client. Both fail OPEN on infrastructure errors.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.build_gates import (
    constraint_gate,
    files_from_specs,
    smoke_gate,
)


def _ctx(reply: str):
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value={
        "choices": [{"message": {"content": reply}}]}))
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


CONSTRAINTS = ["Ghost plays directly at inference time — never a coded "
               "chess engine or random move picker"]
ENGINE_FILE = {"app.py": "def get_ghost_move(board):\n    import random\n"
                         "    return random.choice(list(board.legal_moves))\n"}


class TestConstraintGate:
    @pytest.mark.asyncio
    async def test_confirmed_violation_blocks_with_evidence(self):
        reply = json.dumps({"violates": True,
                            "constraint": CONSTRAINTS[0],
                            "evidence": "app.py: random.choice(legal_moves)"})
        ok, reason = await constraint_gate(_ctx(reply), CONSTRAINTS, ENGINE_FILE)
        assert not ok
        assert "CONSTRAINT VIOLATION" in reason
        assert "random.choice" in reason

    @pytest.mark.asyncio
    async def test_clean_verdict_passes(self):
        reply = json.dumps({"violates": False, "constraint": "", "evidence": ""})
        ok, reason = await constraint_gate(_ctx(reply), CONSTRAINTS, ENGINE_FILE)
        assert ok and reason == ""

    @pytest.mark.asyncio
    async def test_fails_open_on_llm_error(self):
        llm = SimpleNamespace(chat_completion=AsyncMock(side_effect=RuntimeError("down")))
        ctx = SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))
        ok, _ = await constraint_gate(ctx, CONSTRAINTS, ENGINE_FILE)
        assert ok

    @pytest.mark.asyncio
    async def test_fails_open_on_unparseable_reply(self):
        ok, _ = await constraint_gate(_ctx("I think it looks fine?"),
                                      CONSTRAINTS, ENGINE_FILE)
        assert ok

    @pytest.mark.asyncio
    async def test_no_constraints_or_files_is_a_pass_without_llm(self):
        ctx = _ctx("unused")
        assert (await constraint_gate(ctx, [], ENGINE_FILE))[0]
        assert (await constraint_gate(ctx, CONSTRAINTS, {}))[0]
        ctx.llm_client.chat_completion.assert_not_called()

    def test_files_from_specs_collects_content_append_edits(self):
        specs = [
            {"path": "a.py", "content": "print(1)"},
            {"path": "b.html", "append": "<script>x</script>"},
            {"path": "c.py", "edits": [{"replace": "x", "replace_with": "y = 2"}]},
            {"path": "", "content": "ignored"},
            "not-a-dict",
        ]
        files = files_from_specs(specs)
        assert files["a.py"] == "print(1)"
        assert "<script>" in files["b.html"]
        assert "y = 2" in files["c.py"]
        assert len(files) == 3


class TestSmokeGate:
    @pytest.mark.asyncio
    async def test_failures_become_retry_feedback(self):
        async def runner(name, args):
            assert name == "execute"
            assert args["filename"] == ".smoke_gate.py"
            return ("--- EXECUTION RESULT ---\nSMOKE_RESULT "
                    + json.dumps({"failures": [
                        "py_compile app.py: invalid syntax",
                        "GET /api/state -> 500"]}))
        reason = await smoke_gate(runner, ["app.py"])
        assert reason is not None
        assert "SMOKE GATE FAILED" in reason
        assert "invalid syntax" in reason and "500" in reason

    @pytest.mark.asyncio
    async def test_clean_run_passes(self):
        async def runner(name, args):
            return "SMOKE_RESULT " + json.dumps({"failures": []})
        assert await smoke_gate(runner, ["app.py"]) is None

    @pytest.mark.asyncio
    async def test_no_python_files_skips_entirely(self):
        called = []
        async def runner(name, args):
            called.append(name)
            return ""
        assert await smoke_gate(runner, ["index.html", "style.css"]) is None
        assert called == []

    @pytest.mark.asyncio
    async def test_fails_open_when_marker_missing_or_runner_errors(self):
        async def mangled(name, args):
            return "sandbox exploded before the script ran"
        assert await smoke_gate(mangled, ["app.py"]) is None

        async def raiser(name, args):
            raise RuntimeError("sandbox down")
        assert await smoke_gate(raiser, ["app.py"]) is None

    def test_smoke_script_catches_real_bugs_end_to_end(self, tmp_path):
        # Run the ACTUAL generated script against a Flask app with the two
        # chess-session bug classes: a 500-ing GET route and a module-level
        # crash would both surface. Uses the real flask in the test env.
        import subprocess, sys, textwrap
        from ghost_agent.core.build_gates import _SMOKE_TEMPLATE
        app_py = tmp_path / "app.py"
        app_py.write_text(textwrap.dedent("""
            from flask import Flask
            app = Flask(__name__)

            @app.route('/ok')
            def ok():
                return 'fine'

            @app.route('/boom')
            def boom():
                return undefined_name  # NameError -> 500
        """))
        script = _SMOKE_TEMPLATE.format(written=["app.py"])
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, cwd=tmp_path)
        assert proc.returncode == 1
        payload = json.loads(proc.stdout.split("SMOKE_RESULT ", 1)[1])
        assert any("/boom" in f and "500" in f for f in payload["failures"])
        assert not any("/ok" in f for f in payload["failures"])
