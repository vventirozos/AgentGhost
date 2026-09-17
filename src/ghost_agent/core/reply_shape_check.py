"""Mechanical reply-SHAPE refutations (§4FN, 2026-09-08).

The late verifier judges CLAIMS against EVIDENCE. It has no opinion about
whether the reply is an answer at all — and on the 121 human-labelled turns
of 2026-08-13 → 09-08 every one of its seven false PASSes carried an empty
reason: the reply was not an answer and there was nothing to refute.

What the one shipped check actually detects (review §4FN M5): every dump-
shaped reply in the live corpus is the agent's OWN finalize fallback — when
tools ran and the model emitted no final text, `_finalize_and_return`
ships `"<head>\\n\\n### Final Output:\\n```text\\n<tool preview>"`. The model
is not pasting a result; the fallback is. The heads live HERE, in one place,
and the finalize site imports them, so the banner cannot be reworded without
this check following (two copies of one literal is how a check goes dark).
The tools' own framings ("--- EXECUTION RESULT ---", "--- COMMAND RESULT
---", "[sandbox job N finished — EXIT CODE: k]") are matched too, for the
rarer case where the model does paste.

Measured on the corpus: 0 of 103 human-approved replies open this way,
1 of 18 human-rejected, 4 of 149 verifier-refuted, 1 of 237 verifier-passed.

Deliberately NOT here: a narration check that fires on a reply CONTAINING
narration ("Let me run…", "I now have…"). Measured on the corpus it would
refute 4 human-APPROVED replies for every 1 human-rejected (long task
replies narrate and the operator accepts them) — see journal §4FN.

What IS here since §4GH (2026-09-13): the reply that is NOTHING BUT
narration after tools ran (`reply_smoothing.narration_only`) — request
e57ad0cf shipped five stacked "Let me now dig into…" paragraphs and no
finding, the cheap judge refuted it, and the escalation to the main model
overturned that to CONFIRMED 0.85. A reply with no claim has nothing to
escalate about; it is refuted here, mechanically, before any judge. Measured
the same way: 0 of 115 human-approved and 0 of 601 verifier-passed replies
match. Gated on at least one real tool (a tool-free "You're welcome! Let me
know…" is an answer) and not on image turns (the image is the answer).

Arithmetic, not a judge: no model call, nothing to be argued out of.
"""
from __future__ import annotations

import re
from typing import List

#: The finalize fallback's three heads — the ONE home of these literals.
FALLBACK_HEADS = {
    "running": ("The command is STILL RUNNING in the background "
                "(it outran its execution budget and was detached, "
                "not killed); its result is not in yet."),
    "failed": ("The last command FAILED — the output below "
               "is its error result, not a success."),
    "success": "Process finished successfully.",
    # §4GH: the forced-final fallback — an honest non-answer, and refuted as
    # one. Lives here so the shape check follows any rewording.
    "no_answer": ("I ran out of this turn's budget before writing an answer, "
                  "so here is the last evidence I gathered instead of a "
                  "summary I did not write."),
    # §4HW: the stream-scrub last resort (the forced-final retry produced
    # nothing after a scrub ate the whole reply). Lives here so the shape
    # check follows any rewording, like `no_answer`.
    "text_only": ("I prepared a tool call but this turn was routed as "
                  "text-only, so it wasn't executed."),
}
FALLBACK_OUTPUT_MARKER = "### Final Output:"

#: Tool framings a model might paste verbatim as the reply.
_TOOL_FRAMINGS = (
    r"--- EXECUTION RESULT ---",
    r"--- COMMAND RESULT ---",
    r"EXIT CODE:\s*-?\d+",
    r"\[sandbox job \d+ finished",
)

_DUMP_HEAD_RE = re.compile(
    r"\A\s*(?:```\w*\s*)?(?:"
    + "|".join([re.escape(h) for k, h in FALLBACK_HEADS.items() if k not in ("no_answer", "text_only")]
               + [re.escape(FALLBACK_OUTPUT_MARKER)] + list(_TOOL_FRAMINGS))
    + r")",
    re.IGNORECASE)
#: The §4GH forced-final fallback is refuted by ITS OWN arm (below), not as a
#: raw dump: a raw-dump refute is repairable ("re-send the same answer in
#: the right form"), a fallback is the honest end state of a turn that
#: could not answer twice — nothing to repair, no judge to consult.
_NO_ANSWER_HEAD_RE = re.compile(
    r"\A\s*(?:" + re.escape(FALLBACK_HEADS["no_answer"]) + "|"
    + re.escape(FALLBACK_HEADS["text_only"]) + ")")

#: The user asked for the raw thing: not a non-answer, an answer (review
#: §4FN minor 8). No live instance yet; the exemption exists so one cannot
#: be refuted at 0.9 when it appears.
_RAW_REQUEST_RE = re.compile(
    r"\b(?:raw|verbatim|exact|full|unmodified|complete)\s+(?:tool\s+)?(?:output|result|log|stdout)\b"
    r"|\bas[- ]is\b|\bdon'?t\s+(?:summari[sz]e|interpret)\b",
    re.IGNORECASE)


def refute_no_answer_fallback(reply: str) -> List[str]:
    """One issue when ``reply`` is the §4GH forced-final fallback."""
    if not _NO_ANSWER_HEAD_RE.match(reply or ""):
        return []
    return ["the reply is the forced-final fallback — the turn ran out of "
            "budget twice without writing an answer; it carries the last "
            "evidence, not a finding"]


def refute_narration_only(reply: str, *, n_real_tools: int,
                          tool_names=()) -> List[str]:
    """One issue when ``reply`` is nothing but working narration after the
    turn ran tools (§4GH). Empty list otherwise — never a pass."""
    from .reply_smoothing import narration_only, strip_system_notes
    if int(n_real_tools or 0) < 1:
        return []
    if any(str(n).strip().lower() == "image_generation" for n in (tool_names or ())):
        return []
    if not narration_only(strip_system_notes(reply or "")):
        return []
    return ["the reply is working narration only ('Let me…' / 'I'll…') — it "
            "announces work and reports no finding, so there is no claim to "
            "verify; the turn ended before an answer was written"]


def refute_raw_tool_dump(reply: str, request: str = "") -> List[str]:
    """One issue when ``reply`` opens as the finalize fallback or a tool's
    own framing — unless ``request`` asked for the raw output — else []."""
    text = str(reply or "")
    m = _DUMP_HEAD_RE.match(text)
    if not m:
        return []
    if request and _RAW_REQUEST_RE.search(str(request)):
        return []
    head = m.group(0).strip().strip("`").strip()
    return [f"the reply is raw tool output pasted as the answer (it opens with "
            f"{head[:40]!r}); the request was not answered"]


# ── §4HY (2026-09-16, req 1cc63597) — a source "actually read" that was never opened
#: The request asked for a source the agent READ (opened/visited), not one it
#: found in a snippet.
_READ_ASK_RE = re.compile(
    r"\b(?:actually|really|you)\s+(?:read|opened|visited|fetched|loaded)\b"
    r"|\bsources?\s+(?:you|that you)\s+(?:read|opened|visited)\b"
    r"|\b(?:read|open|visit)\s+(?:the\s+)?(?:official\s+)?(?:page|site|source|announcement|url)\b",
    re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")
#: Tools that LOAD a page (a search returns snippets, not the page).
PAGE_LOADING_TOOLS = frozenset({"browser", "deep_research", "darkweb_research"})


def refute_unread_source(reply: str, request: str, tool_names=()) -> List[str]:
    """One issue when the request asked for a source the agent actually
    read, the reply cites a URL as its source, and no page-loading tool ran
    this turn — the URL came from a search snippet (the echo the judge then
    confirms). Empty list otherwise — never a pass."""
    req = str(request or "")
    if not _READ_ASK_RE.search(req):
        return []
    names = {str(n or "").strip().lower() for n in (tool_names or ())}
    if names & PAGE_LOADING_TOOLS:
        return []
    m = _URL_RE.search(str(reply or ""))
    if not m:
        return []
    return [f"the request asked for a source the agent actually read, and the reply "
            f"cites {m.group(0)[:80]} as its source, but no page was opened this turn "
            f"(only search results were seen) — open the page and quote it, or say "
            f"plainly that the source was not read"]

