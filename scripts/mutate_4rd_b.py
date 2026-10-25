"""§4RD battery (b) — service launch race, Stop order, cut-reply session text
(2026-10-25). Reuses the §4QA harness.   python3 scripts/mutate_4rd_b.py [all|check|rerun R1 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

SV = "src/ghost_agent/sandbox/services.py"
SR = "interface/server.py"
RT = "src/ghost_agent/api/routes.py"
m.PINS = ["tests/test_4rd_service_launch.py", "tests/test_4rd_interface_stop.py",
          "tests/test_sandbox_services.py", "tests/test_interface_fix_first_2026_09_11.py"]


def _tests():
    return list(m.PINS)


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", SV, "def _launch_redirect_failed(out: str) -> bool:",
     "# ctl\ndef _launch_redirect_failed(out: str) -> bool:"),
    ("CONTROL known-bad (must DIE): a failed redirect is never seen", SV,
     '    return "cannot create" in str(out or "")', "    return False"),
    ("V1 the stale files go back to a host unlink", SV,
     '                f"rm -f {CONTAINER_SERVICES_DIR}/{stem}.log {CONTAINER_SERVICES_DIR}/{stem}.pid; "\n', ""),
    ("V2 no retry", SV,
     "                time.sleep(0.5)\n                out, code = self._exec(f\"sh -c {shlex.quote(inner)}\", timeout=30)\n                if code == 0 and _launch_redirect_failed(out):",
     "                if True:"),
    ("V3 the give-up keeps the registry row", SV,
     "                    reg.pop(key, None)\n                    self._save(reg)\n                    # nothing ran: a rejection",
     "                    # nothing ran: a rejection"),
    ("V4 the give-up is a failure (arms the re-issue guard)", SV,
     "                    return ToolOutcome.rejected(\n                        f\"Error: '{name}' was not started",
     "                    return ToolOutcome.failed(\n                        f\"Error: '{name}' was not started"),
    ("I1 the relay is cut before the agent's cancel", SR,
     "        verdict = await _cancel_agent_turn(rid) if rid else None\n        bg = task.get(\"background_task\")\n",
     "        bg = task.get(\"background_task\")\n        if bg:\n            bg.cancel()\n        verdict = await _cancel_agent_turn(rid) if rid else None\n"),
    ("I2 no grace for the stream to end", SR,
     "                await asyncio.wait_for(asyncio.shield(bg), STOP_DRAIN_GRACE_S)\n", "                pass\n"),
    ("I3 the grace waits on a refused cancel", SR,
     "        if (verdict and verdict.get(\"cancelled\") and isinstance(bg, asyncio.Future)",
     "        if (isinstance(bg, asyncio.Future)"),
    ("I4 a finished relay is cancelled anyway", SR,
     "        if bg and not (isinstance(bg, asyncio.Future) and bg.done()):\n            bg.cancel()",
     "        if bg:\n            bg.cancel()"),
    ("S1 a cut reply is not kept", RT,
     '                        if not _stream_ended and "".join(_acc + _pre).strip():   # r1: text, not empty deltas',
     "                        if False:"),
    ("S2 the cut is not marked", RT,
     '                                    + "\\n\\n" + STREAM_CUT_NOTE, utf16_len(_cut_pre)), name="session-cut")',
     '                                    , utf16_len(_cut_pre)), name="session-cut")'),
    ("S3 a finished stream is stored twice", RT,
     "                        _stream_ended = True\n", ""),
]

if __name__ == "__main__":
    m.main()
