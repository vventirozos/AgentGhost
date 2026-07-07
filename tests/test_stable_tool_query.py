"""Request-stable acquired-skill routing query (IMPROVEMENTS.md #7).

The acquired-skill semantic router filters the ADVERTISED tool set by query.
Feeding it the per-turn query (which becomes the planner's thought mid-request)
byte-changes the tool header every turn → invalidates the upstream prompt-prefix
KV cache. Pinning the routing query to the request's first substantive query
keeps the advertised set byte-stable across the request's turns — the lever the
KV pin (#6) needs.
"""
from ghost_agent.core.agent import GhostAgent


def _state():
    ag = GhostAgent.__new__(GhostAgent)
    return GhostAgent._RequestState(ag)


def test_first_query_pins_the_routing_query():
    st = _state()
    assert st.stable_tool_query("build me a chess UI") == "build me a chess UI"
    # A later, different per-turn query does NOT change the pinned value.
    assert st.stable_tool_query("Thought: I should validate the FEN") == "build me a chess UI"
    assert st.stable_tool_query("another shift") == "build me a chess UI"


def test_empty_first_query_does_not_pin():
    st = _state()
    # An empty/whitespace first query must not LOCK IN emptiness — nothing is
    # pinned yet, so the next substantive query pins instead.
    assert st.stable_tool_query("") == ""
    assert st._stable_tool_query is None       # not pinned
    st.stable_tool_query("   ")
    assert st._stable_tool_query is None       # whitespace didn't pin either
    assert st.stable_tool_query("real query") == "real query"
    assert st.stable_tool_query("later") == "real query"


def test_stable_across_many_turns():
    st = _state()
    first = st.stable_tool_query("analyze the sales csv")
    for turn_query in ["Thought: load pandas", "Thought: groupby region", "final answer"]:
        assert st.stable_tool_query(turn_query) == first


def test_agent_uses_stable_query_for_tool_defs():
    """Guard: handle_chat must route tool-def selection through
    stable_tool_query, not the raw per-turn search_query."""
    import inspect
    src = inspect.getsource(GhostAgent.handle_chat)
    assert "request_state.stable_tool_query(" in src
    # The old direct raw-query call must be gone.
    assert "get_active_tool_defs(search_query or \"\")" not in src
