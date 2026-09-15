"""§4GL (2026-09-14, request cf45e352): the gate that failed a turn for
producing the report it was asked for, and the OCR path no research turn
reached for.

Request cf45e352 was an eight-minute OSINT investigation. It produced a
10.9 KB forensic report and a 4.4 KB standalone answer that correctly
REFUSED to attribute the incident — the right answer, since the domain is
not public. It was recorded as `failed` at confidence 0.15.

Two independent defects, both confirmed from that request's own log.
"""
import pytest

from ghost_agent.core import prompts as prompts_mod
from ghost_agent.core.agent import _is_unverified_mutation
from ghost_agent.tools import registry as registry_mod


def _row(msg: str) -> dict:
    return {"name": "file_system", "content": msg}


# ── 1. the gate fires only for something that can BE run ────────────────────

@pytest.mark.parametrize("msg,fires,why", [
    ("SUCCESS: Wrote 10812 chars to 'revolut_report.md'. "
     "Script-side path (from sandbox cwd): 'revolut_report.md'.", False,
     "the cf45e352 case: a prose report has nothing to run or render"),
    ("SUCCESS: Wrote 90 chars to 'notes.txt'.", False, "prose"),
    ("SUCCESS: Wrote 900 chars to 'findings.json'.", False, "structured data"),
    ("SUCCESS: Exact match found and replaced in 'README.md'.", False,
     "a replace into prose is still prose"),
    ("SUCCESS: Wrote 4200 chars to 'build.py'.", True,
     "req_C0: the defect the gate was written for — untested code"),
    ("SUCCESS: Wrote 25000 chars to 'webos/index.html'.", True,
     "markup can be rendered, so it must be"),
    ("SUCCESS: Exact match found and replaced in 'app.py'.", True, "code"),
    ("SUCCESS: Wrote 300 chars to 'Makefile'.", True,
     "extensionless: unknown stays guarded, the conservative default"),
])
def test_the_gate_asks_what_was_written_not_merely_that_something_was(msg, fires, why):
    """The gate filtered on the TOOL NAME alone, so every successful write
    demanded the file be "run or rendered". A prose deliverable can never
    satisfy that, so writing the report a request ASKED FOR failed the turn —
    and the turn was annotated telling the operator to "run/preview it before
    relying on it".

    The cost was not cosmetic: six lessons were filed `present-on-FAILURE`
    (the outcome-gated loop prunes on that) and the self-model moved to
    "stuck: 3 of my last 5 verdict-bearing turns failed". It had fired six
    times before this fix.

    Fails in any tree where the gate cannot tell a report from a program."""
    assert _is_unverified_mutation(_row(msg)) is fires, why


def test_a_file_read_is_still_not_a_mutation():
    """Control, unchanged: reads carry evidence and were never the subject."""
    assert _is_unverified_mutation(_row("SUCCESS: Read 100 chars from 'x.py'.")) is False
    assert _is_unverified_mutation(None) is False
    assert _is_unverified_mutation({"name": "web_search", "content": "success: wrote"}) is False


def test_a_mixed_write_still_demands_verification():
    """One runnable artifact among inert ones keeps the guard: the exemption
    is "nothing here can be run", not "something here cannot"."""
    both = ("SUCCESS: Wrote 100 chars to 'report.md'."
            "\n"
            "SUCCESS: Wrote 4200 chars to 'build.py'.")
    assert _is_unverified_mutation(_row(both)) is True


def test_an_unparseable_confirmation_keeps_the_guard():
    """If the message shape changes and no path can be read, the gate must
    stay ON — the reworded-message failure mode this file's neighbours warn
    about must not silently disarm it."""
    assert _is_unverified_mutation(
        _row("SUCCESS: wrote the thing (some future wording)")) is True


# ── 2. the OCR path is discoverable from the research seat ─────────────────

def test_the_vision_tool_advertises_reading_text_out_of_a_found_image():
    """The capability was complete — `tool_vision_analysis` takes a REMOTE
    URL, fetches it over Tor behind the SSRF guard, and `extract_text_picture`
    OCRs it. What was missing was any hint that it applies to research: the
    description and the system prompt both framed vision as a way to check
    the agent's OWN generated images and OWN UI. So req cf45e352 navigated to
    the ZachXBT Telegram post, ran extract_text, got the page markup, and
    reported the sender domain as unpublished — with the one screenshot that
    would have shown it never read.

    Fails in any tree where a researcher reading the tool list cannot tell
    that a screenshot is readable."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    ctx = MagicMock()
    ctx.args = SimpleNamespace(anonymous=True)
    defs = registry_mod.get_active_tool_definitions(ctx)
    vision = [d for d in defs
              if d.get("function", {}).get("name") == "vision_analysis"]
    assert vision, "vision_analysis is not registered"
    blob = str(vision[0]).lower()
    # ⚠ ASSERT WHAT IS NEW, NOT WHAT WAS ALREADY THERE. The first version of
    # this pin checked for "screenshot", "url", "extract_text_picture" and
    # "image" — every one of which the PRE-FIX description already contained
    # (the action enum names the OCR mode, and the UI-checking guidance talks
    # about screenshots). It passed with the research paragraph deleted. Two
    # facts distinguish the two worlds, and both are the point:
    #   • a REMOTE url is fetched for you, so there is no download step;
    #   • the text tools return NOTHING from inside an image, which is the
    #     trap that made a found screenshot unreadable in practice.
    assert "fetched over tor" in blob, (
        "the description does not say a remote image is fetched for you, so "
        "the model still has to guess that a URL works")
    assert "nothing from inside an image" in blob, (
        "the description does not say what extract_text CANNOT do — the "
        "reason a screenshot silently reads as 'no such detail exists'")


def test_the_system_prompt_tells_the_research_path_to_OCR_a_screenshot():
    """The prompt mentioned vision only under "CHECKING YOUR OWN UI/APP
    SCREENSHOTS". A researcher following it would never reach for OCR."""
    # Read the CONSTANT, not the module source: `inspect.getsource` here
    # would be a source-text pin, which is the class this project ratchets
    # downward — and the ratchet caught the first version of this file.
    low = prompts_mod.SYSTEM_PROMPT.lower()
    assert "extract_text_picture" in low
    # the rule has to be stated as a RESEARCH instrument, not a UI check
    i = low.index("extract_text_picture")
    for j in (low.rfind("screenshot", 0, i), low.find("screenshot", i)):
        assert j != -1
    assert "not in the page text" in low or "page text" in low, (
        "the prompt does not warn against reporting 'unavailable' when only "
        "the page TEXT was checked")
