"""§4HZ — the live prompt must SAY how to show a plot, not only what not to do.

THE LIVE FAILURE (reqs 0e6cf008 + 30419cf0, 2026-09-17). "Plot them and
show me" → the reply named `/workspace/ifs_grid_oxford.png` in a code span.
"Can you show me the PNG" → a `vision_analysis` caption, a `browser`
navigate to `file://`, then **`/api/download/…`** in bold code. The model's
own thinking quoted the rule it was following: *"I can't type a markdown
image tag myself per instructions"*. The live `SYSTEM_PROMPT` carried only
the prohibition (never hallucinate the tag); the affirmative "show it with
`![Image](/api/download/f.png)`" lived in `SYSTEM_PROMPT_COMPILED`, which
`_select_system_prompt` serves only to a diagnostic probe. The web client
renders that tag (authed blob swap); a code span renders as text.

World where each pin fails: the affirmative sentence is dropped from the
live prompt, the live selector stops serving SYSTEM_PROMPT, or the client
stops rendering `/api/download/` images inline.
"""
import os
import re

import pytest

from ghost_agent.core.prompts import (SPECIALIST_SYSTEM_PROMPT, SYSTEM_PROMPT,
                                      SYSTEM_PROMPT_COMPILED)

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _rule(prompt: str) -> str:
    m = re.search(r"- DISPLAYING IMAGES:.*", prompt)
    assert m, "the DISPLAYING IMAGES rule is gone from the live prompt"
    return m.group(0)


def test_live_prompt_tells_the_model_to_emit_the_image_tag():
    rule = _rule(SYSTEM_PROMPT)
    assert "![Image](/api/download/" in rule
    assert re.search(r"\bSHOW it\b", rule)
    assert "ONLY way the user sees the picture" in rule


def test_live_prompt_names_the_non_displays_the_model_reached_for():
    rule = _rule(SYSTEM_PROMPT)
    for phrase in ("bare path", "code span", "`vision_analysis` caption", "`browser` navigate"):
        assert phrase in rule, phrase
    assert "displays nothing" in rule


def test_prohibition_is_kept_beside_the_instruction():
    """The affirmative half must not have replaced the guard against a
    hallucinated tag — both sentences, one rule."""
    rule = _rule(SYSTEM_PROMPT)
    assert "NEVER hallucinate or type out a markdown image tag" in rule
    assert rule.index("NEVER hallucinate") < rule.index("SHOW it")


def test_the_rule_reads_the_tool_output_as_the_trigger():
    rule = _rule(SYSTEM_PROMPT)
    assert "Saved plot ->" in rule and "file_system write" in rule


def test_specialist_prompt_keeps_its_own_affirmative_line():
    """Where the instruction lived before: the SPECIALIST prompt (coding
    subsystem turns) — which a main conversational turn never sees. It must
    keep saying it; the main prompt now says it too."""
    assert "![Image](/api/download/" in SPECIALIST_SYSTEM_PROMPT
    assert "PLOTTING & IMAGES" in SPECIALIST_SYSTEM_PROMPT


def test_compiled_variant_is_not_where_the_rule_was_hiding():
    """Recorded fact, so the journal's account stays checkable: the compiled
    (probe-only) prompt never carried the image tag; the specialist one did."""
    assert "![Image](/api/download/" not in SYSTEM_PROMPT_COMPILED


def test_web_client_renders_api_download_images_inline():
    """Cross-surface (R5): the prompt promises what the client does. If the
    authed-blob image path leaves app.js, the promise is false and this
    fails before a user asks 'show me' again."""
    p = os.path.join(_ROOT, "interface", "static", "app.js")
    if not os.path.exists(p):
        pytest.skip("web client not present in this checkout")
    src = open(p, encoding="utf-8").read()
    assert "_toAuthedBlobUrl" in src
    assert "/api/download/" in src
    assert re.search(r"<img src=", src)
