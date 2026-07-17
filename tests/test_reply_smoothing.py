"""Finalize reply smoothing (2026-07-17).

Regression target: the WebOS minesweeper turn — a multi-tool fix whose
delivered reply was the concatenation of every loop iteration's visible
text: four working-narration paragraphs ("Let me fix both:", "Now add
the resize logic in openWindow:", …) and the summary stated twice (once
before the verify/restart step, once after). The operator picked the
finalize-scrub option: the live stream keeps showing narration as
progress; the DELIVERED reply drops connective narration and superseded
summary groups. Conservative by construction — fences atomic, only two
shapes removable, fail-open.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.reply_smoothing import smooth_reply


WEBOS_REPLY = """Let me read the current WebOS code to understand both issues.

I see both issues clearly:

1. **Minesweeper creates new grids each turn** — render() creates a new `<div class="app-container">` and appends it every time, stacking old grids on top.
2. **No window resize** — CSS has min-size but no resize handle or JS resize logic.

Let me fix both:

Good, resize handle CSS is in. Now I need to add the resize handle element and logic to openWindow, and fix the minesweeper's render issue.

Now add the resize logic in openWindow:

Now the critical fix — the Minesweeper render() function creates a new app-container div and appends it every time, stacking old grids on top. Let me fix it to clear and reuse the container:

Both issues are now fixed:

1. **Minesweeper** — The render() function now reuses the same grid DOM element instead of creating new `<div class="app-container">` elements on every click. Cells are recycled and only their content/classes are updated.
2. **Window resizing** — Added a resize handle (bottom-right corner with diagonal pattern) and full resize logic. You can now drag the corner to resize any window freely.

Let me verify the fixes are live:

Both issues are fixed and the service is restarted:

1. **Minesweeper** — The render() function was creating a new `<div class="app-container">` on every click, stacking old grids on top of new ones. Now it reuses the same DOM element — cells are recycled and only their content/classes are updated on each render.
2. **Window resizing** — Added a resize handle (bottom-right corner with diagonal pattern) and full resize logic. Any window can now be resized by dragging that handle.

The WebOS is back up on port 8100 with all 5 apps. Give it a try!"""


class TestWebOSRegression:
    def test_narration_paragraphs_dropped(self):
        out = smooth_reply(WEBOS_REPLY)
        assert "Let me read the current WebOS code" not in out
        assert "Let me fix both:" not in out
        assert "Good, resize handle CSS is in" not in out
        assert "Now add the resize logic in openWindow:" not in out
        assert "Now the critical fix" not in out
        assert "Let me verify the fixes are live:" not in out

    def test_pre_verification_summary_superseded_by_final(self):
        out = smooth_reply(WEBOS_REPLY)
        # The first summary (written before the verify/restart step) is
        # restated by the final one — only the final survives.
        assert "Both issues are now fixed:" not in out
        assert "Both issues are fixed and the service is restarted:" in out
        assert out.count("**Minesweeper**") == 1
        assert out.count("**Window resizing**") == 1

    def test_final_answer_content_intact(self):
        out = smooth_reply(WEBOS_REPLY)
        assert "was creating a new" in out           # final summary body
        assert "resized by dragging that handle" in out
        assert "back up on port 8100 with all 5 apps" in out
        # The final summary must come out ahead of the closing line.
        assert out.index("service is restarted") < out.index("port 8100")

    def test_diagnosis_group_is_kept_conservatively(self):
        # The "I see both issues clearly:" diagnosis overlaps the final
        # summary only partially — below the supersede threshold — and
        # is deliberately kept (it carries the problem statement).
        out = smooth_reply(WEBOS_REPLY)
        assert "I see both issues clearly:" in out


class TestConservativeInvariants:
    def test_short_and_single_paragraph_untouched(self):
        assert smooth_reply("Done — 42.") == "Done — 42."
        assert smooth_reply("") == ""
        one = "Now the answer: 42."          # connective but single block
        assert smooth_reply(one) == one

    def test_final_block_never_dropped_even_if_connective(self):
        text = "The fix is applied.\n\nNow you can drag the corner to resize."
        assert "drag the corner" in smooth_reply(text)

    def test_fenced_code_never_dropped(self):
        text = ("Let me show the diff:\n\n"
                "```python\nprint('hi')\n\nprint('bye')\n```\n\n"
                "Applied and tested.")
        out = smooth_reply(text)
        assert "print('hi')" in out
        assert "print('bye')" in out
        # Blank line inside the fence must not split it.
        assert "```python\nprint('hi')\n\nprint('bye')\n```" in out

    def test_all_narration_fails_open(self):
        text = "Let me check this.\n\nNow, running it."
        # Everything matches the narration shape except the protected
        # final block; the reply must never come out empty.
        out = smooth_reply(text)
        assert out.strip()

    def test_distinct_lists_not_treated_as_duplicates(self):
        text = ("Results by host:\n\n"
                "1. ghost — 12ms latency, 0 loss\n"
                "2. eva — 15ms latency, 0 loss\n\n"
                "Results by service:\n\n"
                "1. postgres — healthy, 42 connections\n"
                "2. uvicorn — healthy, 8 workers")
        out = smooth_reply(text)
        assert "ghost — 12ms" in out
        assert "postgres — healthy" in out

    def test_idempotent(self):
        once = smooth_reply(WEBOS_REPLY)
        assert smooth_reply(once) == once


class TestWiring:
    def test_finalize_gates_on_multi_tool_turns(self):
        """Smoothing must ride _finalize_and_return behind the ≥2
        substantive-tools gate — conversational and single-tool replies
        are never rewritten."""
        src = (Path(__file__).resolve().parents[1]
               / "src" / "ghost_agent" / "core" / "agent.py").read_text()
        assert "from .reply_smoothing import smooth_reply" in src
        idx = src.find("from .reply_smoothing import smooth_reply")
        window = src[idx - 600:idx]
        assert ">= 2" in window and "_synthetic" in window

    def test_prompt_forbids_repeated_summaries(self):
        from ghost_agent.core.prompts import SYSTEM_PROMPT
        lowered = SYSTEM_PROMPT.lower()
        assert "never restate a summary" in lowered
        assert "one final summary" in lowered
