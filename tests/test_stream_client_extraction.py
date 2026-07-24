"""Guards for #5 step 4a — the client-facing SSE branch extracted from
handle_chat into GhostAgent._stream_final_generation(ss: StreamState).

The behavioral equivalence of the streamed output is already covered by
test_streaming_tail_cancellable / test_streaming_scrub_behavioral /
test_agent (they drive handle_chat's streaming path end-to-end, which now
delegates here). These tests pin the EXTRACTION CONTRACT itself, so a future
edit can't silently reintroduce the capture bug the extraction was careful to
avoid: the stream_wrapper generator runs AFTER handle_chat returns, so every
frame local it closes over must arrive via StreamState — a missed one is a
mid-stream NameError that unit tests of a happy path may not hit.
"""
import inspect
import symtable

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import GhostAgent, StreamState


def test_method_has_no_uncaptured_free_variables():
    """The load-bearing invariant: _stream_final_generation must close over
    NOTHING from an enclosing scope — every captured handle_chat local is
    unpacked from ss at the top, so symtable reports zero free names. If this
    fails, a capture was dropped and the stream would NameError after
    handle_chat returns."""
    src = inspect.getsource(agent_mod)
    st = symtable.symtable(src, "agent.py", "exec")

    def find(scope, name):
        for c in scope.get_children():
            if c.get_name() == name:
                return c
            r = find(c, name)
            if r:
                return r
        return None

    meth = find(st, "_stream_final_generation")
    assert meth is not None, "extracted method not found"
    frees = sorted(meth.get_frees())
    assert not frees, f"uncaptured free vars (would NameError mid-stream): {frees}"


def test_stream_state_fields_match_the_methods_unpack():
    """StreamState's field set is the capture contract; the method must unpack
    exactly those (a drift either leaves a field unused or reads an un-provided
    one)."""
    fields = {f.name for f in StreamState.__dataclass_fields__.values()}
    body = inspect.getsource(GhostAgent._stream_final_generation)
    for f in fields:
        assert f"{f} = ss.{f}" in body, f"StreamState.{f} is not unpacked in the method"


def test_handle_chat_delegates_the_streaming_branch():
    """The streaming branch is no longer inline: handle_chat builds StreamState
    and returns the method call, and the streamer body lives in the method."""
    hc = inspect.getsource(GhostAgent.handle_chat)
    assert "return self._stream_final_generation(ss)" in hc
    assert "async def stream_wrapper" not in hc          # moved out
    assert "async def stream_wrapper" in inspect.getsource(
        GhostAgent._stream_final_generation)


def test_method_is_sync_returning_the_response_tuple():
    """It only DEFINES the async generators and returns the tuple (no top-level
    await), so it is a plain def — handle_chat returns the tuple directly."""
    assert not inspect.iscoroutinefunction(GhostAgent._stream_final_generation)
