"""Stylometric egress scrubbing.

Tor anonymises the *network* layer. It does nothing about the *content*
layer: the agent's prose — its politeness habits, first-person framing,
punctuation idiosyncrasies, verbosity — is a stable author fingerprint
that egresses verbatim to every search engine and fetched site. For an
autonomous agent issuing many requests, that stylometric fingerprint is
a larger deanonymisation surface than the IP.

This module normalises outbound query text into a neutral canonical form
*before* it leaves the box:

* :func:`scrub_query` — pure-Python, deterministic, always-safe. Strips
  the assistant-directed request frame ("can you find me…", "please",
  first-person "I'm looking for…"), lowercases (search is
  case-insensitive; casing is a style signal), drops trailing sentence
  punctuation, and removes politeness tokens — while **preserving every
  content keyword and in-word technical symbol** (``c#``, ``c++``,
  ``.net``, ``asyncio.run`` survive intact). Cheap enough to run on
  every search.

* :func:`neutralize_query` — optional LLM re-author into a neutral
  keyword query via Ghost's own upstream (no third party). Falls back to
  :func:`scrub_query` on any failure or when no client is supplied, so
  it is always at least as safe as the lexical path.

Tied to ``--anonymous`` (the same flag that mandates Tor for search):
content anonymisation and network anonymisation travel together.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("GhostAgent")


# Assistant-directed request frames + first-person/politeness openers that
# fingerprint an individual author but carry no search intent. "how to" /
# "what is" are deliberately NOT here — those are universal search idioms,
# not personal style. Bare "search" / "lookup" are deliberately NOT here
# either: they lead real content phrases ("search algorithms in python",
# "lookup table sql") and stripping them violates the never-drop-a-content-
# keyword contract; only the unambiguous framed forms ("search for",
# "look up") are stripped. Longest-first matching is enforced at use site.
_REQUEST_PREFIXES = (
    "can you please tell me about", "can you please tell me", "can you tell me about",
    "can you tell me", "can you please", "could you please", "can you", "could you",
    "would you please", "would you", "i want to know about", "i want to know",
    "i want to", "i need to know", "i need to", "i need", "i'd like to know",
    "i'd like to", "i would like to know", "i would like to", "i'm looking for",
    "im looking for", "i am looking for", "looking for", "search for",
    "find me", "look up", "tell me about", "tell me", "show me",
    "give me", "help me", "please", "kindly",
)

# Standalone politeness tokens removed anywhere in the query. Gratitude
# tokens ("thank", "thanks", "thx") are NOT here — leading/interior they
# are usually content ("thank you card template", "thanks in spanish");
# they are only politeness when trailing, handled by the regex below.
_POLITE_TOKENS = {"please", "kindly", "pls", "plz"}

# Trailing gratitude run: "… thanks", "… thank you", "… thx thanks".
_TRAILING_GRATITUDE_RE = re.compile(
    r"(?:\s+(?:thank\s+you|thanks|thank|thx))+\s*$", re.IGNORECASE
)


def scrub_query(text: Optional[str]) -> str:
    """Normalise a query into a neutral, fingerprint-reduced keyword form.

    Deterministic and conservative: never drops a content keyword or an
    in-word technical symbol. Returns the original (stripped) text if
    scrubbing would empty it.
    """
    if not text or not str(text).strip():
        return "" if text is None else str(text)
    original = str(text).strip()
    s = original

    # Strip leading request/politeness frames REPEATEDLY (longest match
    # first), so stacked openers like "please can you find me …" peel off
    # fully — and so the function is idempotent. Bounded against any
    # pathological cycle.
    _prefixes = sorted(_REQUEST_PREFIXES, key=len, reverse=True)
    for _ in range(8):
        low = s.lower()
        for pref in _prefixes:
            if low.startswith(pref + " "):
                s = s[len(pref):].strip()
                break
        else:
            break

    # Casing is a style signal and search is case-insensitive.
    s = s.lower()
    # Drop trailing sentence punctuation runs (keep interior c#, c++, .net).
    s = re.sub(r"[?!.…]+\s*$", "", s).strip()
    # Remove standalone politeness tokens (strip surrounding light punct).
    toks = [t for t in s.split() if t.strip(",.;:!?") not in _POLITE_TOKENS]
    s = re.sub(r"\s+", " ", " ".join(toks)).strip()
    # Trailing gratitude is politeness; leading/interior gratitude is
    # content ("thank you card template") and must survive.
    s = _TRAILING_GRATITUDE_RE.sub("", s).strip()

    return s or original.lower()


_NEUTRALIZE_PROMPT = (
    "Rewrite the following request as a concise, neutral web-search query. "
    "Output ONLY the query — no preamble, no quotes, no explanation. Remove "
    "first-person phrasing, politeness, and any stylistic or identifying "
    "wording; keep only the essential search keywords and any technical "
    "terms/symbols verbatim. Keep it under 12 words.\n\nREQUEST: {q}\n\nQUERY:"
)


async def neutralize_query(
    text: Optional[str],
    *,
    llm_client=None,
    model: str = "",
    max_tokens: int = 64,
) -> str:
    """LLM re-author of a query into neutral keyword form, with a
    deterministic fallback to :func:`scrub_query`.

    Used on the deep-research path (already LLM-heavy, latency-tolerant).
    Never raises; any failure degrades to the lexical scrub so the egress
    text is always at least fingerprint-reduced.
    """
    lexical = scrub_query(text)
    if not text or llm_client is None:
        return lexical
    try:
        payload = {
            "model": model or "default",
            "messages": [{"role": "user", "content": _NEUTRALIZE_PROMPT.format(q=str(text)[:600])}],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        res = await llm_client.chat_completion(payload)
        out = (
            (res or {}).get("choices", [{}])[0]
            .get("message", {}).get("content", "") or ""
        ).strip()
        # The model occasionally wraps the query in quotes or a prefix;
        # strip a leading "query:" label and surrounding quotes.
        out = re.sub(r"^\s*query\s*:\s*", "", out, flags=re.IGNORECASE).strip()
        out = out.strip("\"'`").strip()
        # Take the first line only — guards against an explanatory tail.
        out = out.splitlines()[0].strip() if out else ""
        if not out or len(out) > 300:
            return lexical
        # Final lexical pass so the LLM's own phrasing is also normalised.
        return scrub_query(out)
    except Exception as exc:
        logger.debug("neutralize_query fell back to lexical scrub: %s", exc)
        return lexical


__all__ = ["scrub_query", "neutralize_query"]
