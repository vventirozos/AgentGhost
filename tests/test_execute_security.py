
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.tools.execute import tool_execute

@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    "../../../tmp/evil.py",      # deep relative traversal
    "../../etc/passwd",
    "/etc/passwd",               # absolute outside path
    "../outside.py",
])
async def test_tool_execute_blocks_paths_outside_the_sandbox(payload):
    """§4BW: replaces a VACUOUS predecessor. The old
    `test_tool_execute_path_traversal_vulnerability` fired a traversal
    payload and then did `if 'outside'...: pass / elif 'EXIT CODE'...: pass`
    with no assertion and no else — it passed whether the code was safe,
    vulnerable, or threw. A security-named test that certified nothing
    (§4BW test-instrument findings). This one asserts the write is refused
    and the sandbox manager is never asked to execute anything outside."""
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    sandbox_manager.execute = AsyncMock(return_value=("output", 0))

    result = await tool_execute(payload, "print('hacked')",
                                sandbox_dir, sandbox_manager)

    # THE security property, asserted against the code's ACTUAL refusal
    # messages (verified live, not guessed): a `.py` traversal is stopped by
    # the containment guard ("Security Error"), a non-`.py` outside path by
    # the extension guard ("SYSTEM ERROR"); both refuse with EXIT CODE: 1.
    # Asserting the refusal — not "execute wasn't called with the literal
    # path" — is what catches a DISABLED containment guard: with it off,
    # `../outside.py` runs (the mock returns EXIT CODE 0 + output) instead of
    # refusing. The first version of this test scanned exec args and SURVIVED
    # that mutation while its older sibling killed it; this does not.
    assert "EXIT CODE: 1" in result, (
        f"{payload!r} did not refuse (EXIT CODE 1): {result[:160]!r}")
    assert ("Security Error" in result or "SYSTEM ERROR" in result), (
        f"{payload!r} produced no refusal message — a guard was bypassed: "
        f"{result[:160]!r}")


@pytest.mark.asyncio
async def test_tool_execute_prevents_traversal():
    """Explicitly test that traversal attempts are blocked."""
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    
    filename = "../outside.py"
    content = "print('fail')"
    
    result = await tool_execute(filename, content, sandbox_dir, sandbox_manager)
    
    # We expect the fix to trigger a specific error message
    assert "Security Error" in result or "outside sandbox" in result
