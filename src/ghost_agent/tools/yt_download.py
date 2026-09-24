"""The code-owned ``youtube_transcribe`` macro definition.

History. Until §4KE (2026-09-24) this module smuggled a base64-encoded bash
script (``yt_tor_download.sh``: yt-dlp + exit-node rotation) into the macro's
``execute`` step. It could not work by construction: the sandbox promotes
anything still running at 90 s to a background job, the sequential runner
books that as a failed step, and step 2 (transcribe) never ran — live usage
3, successes 0. It also wrote every video to one constant filename (so the
knowledge base's name-keyed dedup answered every later video from the first),
ran ``pip install`` and ``curl … | sudo sh`` at run time inside the sandbox,
and its quoted-heredoc URL defence could be ended by a line break in the URL.

Now the whole route lives in-process — :mod:`ghost_agent.memory.youtube_ingest`
behind ``knowledge_base(action='transcribe', filename='<url>')`` — and this
macro is a ONE-STEP alias kept so the name the system prompt teaches
(``youtube_transcribe(url=…)``) keeps working. The stored copy in the
composed-skill registry is reconciled from :data:`CODE_OWNED_MACROS` at load
(``ComposedSkillRegistry._reconcile_code_owned``), so a definition change
ships with the code instead of needing the agent stopped for a sync script.
"""

from __future__ import annotations

# The macro's single runtime input, substituted by the composed-skill
# resolver at call time.
URL_VAR = "$url"


def build_youtube_transcribe_definition() -> dict:
    """The full, current ``youtube_transcribe`` macro definition: one step,
    the knowledge-base transcribe action on the URL (Tor, PO-token helper,
    captions first, audio otherwise — all inside that tool)."""
    return {
        "name": "youtube_transcribe",
        "description": (
            "Transcribe a YouTube video into the knowledge base in one call and "
            "return the opening of its transcript: the link is fetched over Tor "
            "(captions when the video has them, otherwise the audio is transcribed "
            "on the private audio node), indexed with timestamps under "
            "'yt-<video id>…', and readable in order with "
            "knowledge_base(action='transcript'). Input: url."
        ),
        "mode": "sequential",
        "steps": [
            {
                "tool": "knowledge_base",
                "description": "Fetch the video over Tor and transcribe it into the knowledge base",
                "params": {"action": "transcribe", "filename": URL_VAR},
            },
        ],
    }


#: Hand-written macros whose definition is CODE, keyed by name. The registry
#: overwrites a stored copy's steps/description with these at load, keeping
#: the stored usage counters.
CODE_OWNED_MACROS = {
    "youtube_transcribe": build_youtube_transcribe_definition,
}
