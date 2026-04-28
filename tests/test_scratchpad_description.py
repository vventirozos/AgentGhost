"""Scratchpad tool description must be the obvious choice for
'set a key/value' prompts.

Bug from production T3 trace: user said "Use the scratchpad to set a
key 'eval_run_id' with value 'qa-batch-2026-04-28'." The model emitted
`file_system.write(path='scratchpad/eval_run_id', ...)` AND
`scratchpad(action='get', key='eval_run_id')` in parallel — the get
ran before any set existed (because no set was ever issued via
scratchpad), strike fired, and the agent rationalized the failure as
"expected behavior, system is working correctly."

The description was too terse ("Read, write, or clear short-term
persistent notes to your SCRAPBOOK") and didn't claim the surface for
key/value prompts. Models reach for `file_system.write` to a tagged
path because writing-a-named-thing is what `write` does in their
training corpus.

Fix: the description must (a) name the verbs the user uses ("set a
key", "save a variable"), (b) explicitly forbid using file_system.write
to persist named values, and (c) frame this as the FIRST CHOICE.
"""

from ghost_agent.tools.registry import TOOL_DEFINITIONS


def _scratchpad_def():
    for t in TOOL_DEFINITIONS:
        if t.get("function", {}).get("name") == "scratchpad":
            return t["function"]
    raise AssertionError("scratchpad not registered")


def test_description_claims_set_a_key_phrasing():
    """Description must include the literal user phrasings models map
    onto — without these, they reach for file_system.write."""
    desc = _scratchpad_def()["description"]
    # The two most common verbs the user actually says:
    assert "set a key" in desc, (
        "Description must explicitly mention 'set a key' so the model "
        "associates that exact phrase with this tool, not file_system."
    )
    assert "save a variable" in desc or "save a value" in desc


def test_description_forbids_file_system_write_for_kv():
    """The negative is what stopped the T3 misroute. Without it,
    file_system.write looks like a valid 'persist a tagged value'
    option to the model."""
    desc = _scratchpad_def()["description"]
    assert "Do NOT use file_system.write" in desc, (
        "Description must explicitly negate the file_system.write "
        "antipattern that caused the production T3 misroute."
    )


def test_description_marks_first_choice():
    """Salience matters when multiple tools could plausibly match. The
    description should mark scratchpad as the FIRST CHOICE for kv
    storage so it wins on tie."""
    desc = _scratchpad_def()["description"]
    assert "FIRST CHOICE" in desc, (
        "Description must mark scratchpad as the FIRST CHOICE for "
        "key/value prompts so it wins over file_system.write."
    )


def test_action_enum_unchanged():
    """The bug was in description salience, not action shape. Pin the
    action enum so the rename doesn't accidentally drop any."""
    props = _scratchpad_def()["parameters"]["properties"]
    assert set(props["action"]["enum"]) == {"set", "get", "list", "clear"}


def test_value_param_description_does_not_say_just_content():
    """Old description said 'The content to save' — the word 'content'
    overloads the knowledge_base param name and confused the model on
    cross-tool reads. New description should be tool-agnostic."""
    props = _scratchpad_def()["parameters"]["properties"]
    val_desc = props["value"]["description"]
    # The exact old wording was 'The content to save'. Either the
    # phrase is gone OR it's qualified with the new key/value framing.
    assert "key" in val_desc.lower() or "data" in val_desc.lower() or "value" in val_desc.lower()
