"""GhostAgent._compose_injection — per-turn context placement.

The pinned mode exists to make the large STABLE block (tool schemas +
persona + memory) cacheable across the turns of one request by keeping it
at a fixed position, while the small VOLATILE dynamic_state rides the
moving last message. The cache property under test: the portion of the
first user message UP TO the stable block is byte-identical on turn 1 and
turn 2, so the upstream KV-cache can reuse it.
"""

from ghost_agent.core.agent import GhostAgent

STABLE = "TOOLS+PERSONA+MEMORY" * 50  # large, stable within a request
DYN_T1 = "### DYNAMIC SYSTEM STATE\nCURRENT TIME: 2026-06-30 10:00\nplan: A"
DYN_T2 = "### DYNAMIC SYSTEM STATE\nCURRENT TIME: 2026-06-30 10:01\nplan: B"


def _turn1():
    # Single user message (the query) — the loop's first turn.
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "compute 15!"},
    ]


def _turn2():
    # After turn 1 the loop appended the assistant tool-call + tool result;
    # the tool result is the new last message.
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "compute 15!"},
        {"role": "assistant", "content": "<tool_call>run</tool_call>"},
        {"role": "user", "content": "TOOL RESULT: 1307674368000"},
    ]


def test_legacy_puts_whole_injection_on_last_message():
    msgs = GhostAgent._compose_injection(_turn2(), STABLE, DYN_T2, pin=False)
    # Stable block rides the LAST message (the tool result), nothing pinned early.
    assert STABLE in msgs[-1]["content"]
    assert "compute 15!" == msgs[1]["content"]  # original query untouched
    assert STABLE not in msgs[1]["content"]


def test_pinned_places_stable_first_volatile_last_turn2():
    msgs = GhostAgent._compose_injection(_turn2(), STABLE, DYN_T2, pin=True)
    # Stable pinned to the first user message...
    assert STABLE in msgs[1]["content"]
    assert "compute 15!" in msgs[1]["content"]
    # ...volatile on the last message, NOT the first.
    assert DYN_T2 in msgs[-1]["content"]
    assert DYN_T2 not in msgs[1]["content"]
    assert STABLE not in msgs[-1]["content"]


def test_pinned_turn1_keeps_pinned_message_clean():
    """Turn 1: the pinned first message holds stable + query but NOT volatile —
    volatile rides its own trailing message so the pinned message is byte-
    identical to every later turn's (enabling turn-2 reuse)."""
    msgs = GhostAgent._compose_injection(_turn1(), STABLE, DYN_T1, pin=True)
    assert STABLE in msgs[1]["content"]
    assert "compute 15!" in msgs[1]["content"]
    assert DYN_T1 not in msgs[1]["content"]          # volatile NOT folded in
    assert DYN_T1 in msgs[-1]["content"]             # it's the trailing message
    assert msgs[-1] is not msgs[1]


def test_pinned_first_message_identical_across_turns():
    """The cache win: the entire pinned first user message must be byte-
    identical on turn 1 and turn 2, so the upstream reuses it on turn 2 — the
    regression that previously made turn 2 re-prefill the full context."""
    m1 = GhostAgent._compose_injection(_turn1(), STABLE, DYN_T1, pin=True)
    m2 = GhostAgent._compose_injection(_turn2(), STABLE, DYN_T2, pin=True)
    # System message identical AND the first user message fully identical,
    # despite different dynamic_state each turn.
    assert m1[0] == m2[0]
    assert m1[1]["content"] == m2[1]["content"]
    assert DYN_T1 != DYN_T2  # volatile genuinely differs per turn


def test_pinned_handles_no_user_message():
    msgs = GhostAgent._compose_injection(
        [{"role": "system", "content": "SYS"}], STABLE, DYN_T1, pin=True,
    )
    # A stable-block user message is inserted right after system.
    assert any(STABLE in m["content"] for m in msgs if m["role"] == "user")
