"""Native-path tool header — the tool_call-corruption root-cause fix
(2026-07-31, journal §6 "native tool_call corruption ROOT-CAUSED").

Controlled ablation against the live upstream (llama.cpp b10180, template
froggeric-v21.3) proved the corruption trigger was the agent itself: the
LEGACY XML tool prompt (QWEN_TOOL_PROMPT — attribute-dialect examples, XML
RULES, CDATA hatch, "output the <tool_call> XML block IMMEDIATELY") was
spliced into the system slot on the NATIVE path too. With it, the model
emits hybrid XML (agent attribute-dialect × template equals-dialect) and
the upstream's incremental parser swallows every stacked call after the
first parameter into that argument value — 8/8 corrupt with the block,
13/13 clean without. 18 repair fires in one 2.5h log window.

The fix splits the header: QWEN_TOOL_PROMPT_NATIVE keeps think-discipline +
the parallel invitation and drops EVERY XML format instruction (the chat
template owns the call format on the native path). The legacy XML path
keeps the full prompt — there the XML rules ARE the format contract.
"""

import inspect

from ghost_agent.core.prompts import (
    QWEN_TOOL_PROMPT,
    QWEN_TOOL_PROMPT_NATIVE,
    SYSTEM_PROMPT,
)

# Every marker of the XML call dialect. Any ONE of these back in the native
# header re-arms the corruption trigger.
_XML_DIALECT_MARKERS = (
    '<function name=',
    '<parameter name=',
    '<tool_call>',
    '</tool_call>',
    'CDATA',
    'XML RULES',
    'XML block',
    'XML tags:',   # the <tools></tools> schema-wrapper phrasing
)


def test_native_header_has_no_xml_dialect():
    for marker in _XML_DIALECT_MARKERS:
        assert marker not in QWEN_TOOL_PROMPT_NATIVE, (
            f"XML dialect marker {marker!r} found in QWEN_TOOL_PROMPT_NATIVE "
            f"— this is the exact corruption trigger the 2026-07-31 ablation "
            f"proved (hybrid-XML emission → merged multi-tool calls)."
        )


def test_native_header_keeps_the_load_bearing_parts():
    # Splice placeholders — agent.py replaces both.
    assert "{tool_schemas}" in QWEN_TOOL_PROMPT_NATIVE
    assert "{think_budget_guidance}" in QWEN_TOOL_PROMPT_NATIVE
    # Think discipline survives (these rules are format-independent).
    assert "ANTI-PARALYSIS" in QWEN_TOOL_PROMPT_NATIVE
    assert "DO NOT draft or write Python" in QWEN_TOOL_PROMPT_NATIVE
    # Parallel tool use stays invited — the fix must not un-teach
    # parallelism, only the broken dialect.
    assert "PARALLEL EXECUTION" in QWEN_TOOL_PROMPT_NATIVE
    assert "MULTIPLE tools in a single turn" in QWEN_TOOL_PROMPT_NATIVE


def test_legacy_header_keeps_its_xml_contract():
    """The legacy (native-tools-off) path still needs the full XML format
    instruction — that dialect IS its parser contract."""
    assert '<function name="function_name">' in QWEN_TOOL_PROMPT
    assert "CDATA" in QWEN_TOOL_PROMPT


def test_agent_splices_native_variant_on_native_path():
    """Source pin: the native branch must use QWEN_TOOL_PROMPT_NATIVE and
    the legacy branch QWEN_TOOL_PROMPT. A refactor that re-unifies them
    silently reintroduces the corruption."""
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod)
    assert "QWEN_TOOL_PROMPT_NATIVE\n" in src.replace(" ", "\n") or \
        "QWEN_TOOL_PROMPT_NATIVE" in src
    # The native pointer rides in the NATIVE variant's splice…
    native_idx = src.index("QWEN_TOOL_PROMPT_NATIVE\n                            .replace('{tool_schemas}', _native_pointer)")
    # …and the legacy branch still splices the full XML prompt.
    legacy_idx = src.index("QWEN_TOOL_PROMPT\n                            .replace('{tool_schemas}', minified_schemas)")
    assert native_idx > 0 and legacy_idx > 0


def test_system_prompt_tool_instruction_is_path_neutral():
    """The CRITICAL INSTRUCTION section must not order XML output
    unconditionally — on the native path that instruction was part of the
    mixed messaging behind the hybrid-dialect emission."""
    assert "you MUST use the exact tool calling format instructed using XML tags" \
        not in SYSTEM_PROMPT
    assert "native tool_calls API when tool schemas are advertised natively" \
        in SYSTEM_PROMPT


# ══════════════════════════════════════════════════════════════════════
# Repair fire = tripwire, not routine noise (2026-07-31)
# ══════════════════════════════════════════════════════════════════════

def test_repair_fire_is_flagged_as_unexpected():
    """The root cause is fixed and a 6-probe battery (incl. explicit
    parallel-call demands) produced zero fires, so a fire now means a
    NOVEL corruption shape. For months this line read as background
    noise and was scrolled past — it must now say what it means and
    point at the raw snapshot that is the only record of the new shape."""
    import inspect
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod)
    assert "UNEXPECTED since" in src
    assert "NEW corruption shape" in src


def test_repair_fire_is_recorded_in_the_activity_ledger():
    """The live stream scrolls away; the rate must stay answerable after
    the fact via introspect action='activity'. INFO severity — one
    repaired call is not worth interrupting the operator."""
    import inspect
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod)
    assert '"native_tool_repair"' in src
    # Recorded, but deliberately NOT notify-severity (chat stays clean).
    idx = src.index('"native_tool_repair"')
    region = src[idx:idx + 600]
    assert "SEVERITY_NOTIFY" not in region
    assert "raw_head" in region
