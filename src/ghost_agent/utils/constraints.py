"""Heuristic extraction of explicit user constraints from a request.

The chess-game incident (2026-07-02) showed the failure shape this module
guards against: the user wrote "don't come up with some random AI for
this, it's gonna be a turn by turn game where YOU will play against me",
and the agent acknowledged the clause in its reasoning, then rationalised
it away and shipped a minimax engine anyway. Reinforced skill priors and
a stale task decomposition out-shouted a one-off negation in the current
message.

The counterweight is deterministic, not model-based: pull the clauses the
user explicitly marked as binding (negations, "YOU will ..." role
assertions, ALL-CAPS emphasis) out of the free text so they can be

* stored on the project/task record (``manage_projects``),
* re-rendered into the prompt every turn (project briefing + dynamic
  state), and
* replayed at verification time (verifier context, post-write steer).

No LLM call. False positives are cheap (an extra clause is re-shown to
the model); false negatives are what we are trying to eliminate.
"""

import re
from typing import List

# Clause-level markers of an explicit prohibition / exclusion.
_NEGATION_RE = re.compile(
    r"\b(don'?t|do\s+not|never|must\s+not|should\s+not|shouldn'?t|"
    r"no\s+need\s+(?:for|to)|without|avoid|not\s+(?:a|an|some|the|going|gonna)|"
    r"instead\s+of|rather\s+than|none\s+of|stop\s+(?:using|doing)|"
    r"skip\s+the)\b",
    re.IGNORECASE,
)

# Role assertions binding the AGENT itself into the deliverable ("YOU will
# play against me"). These are constraints on architecture, not phrasing:
# the agent must be a runtime participant, not the author of a stand-in.
_PARTICIPANT_RE = re.compile(
    r"\b(you(?:'ll| will| should| are going to| play| act)\b|"
    r"against\s+me|with\s+me\b|between\s+us\b)",
    re.IGNORECASE,
)

# ALL-CAPS tokens are user emphasis — except ubiquitous technical
# acronyms, which are capitalised in normal prose.
_CAPS_ACRONYMS = {
    "AI", "API", "CLI", "CSS", "CSV", "FEN", "GET", "GPU", "HTML", "HTTP",
    "HTTPS", "IDE", "JS", "JSON", "LLM", "MCP", "OK", "OS", "PDF", "PNG",
    "POST", "RAM", "SQL", "SSH", "TODO", "UI", "URL", "USB", "UX", "XML",
}
_CAPS_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")

_MAX_CLAUSE_CHARS = 160


def _has_caps_emphasis(clause: str) -> bool:
    return any(
        tok not in _CAPS_ACRONYMS for tok in _CAPS_TOKEN_RE.findall(clause)
    )


def _clauses(text: str) -> List[str]:
    """Split into sentence- then comma-level clauses.

    Real requests pack several independent requirements into one long
    sentence ("build X, don't do Y, it's gonna be Z"); sentence-level
    capture would return the whole message and dilute the signal.
    """
    out: List[str] = []
    for sentence in re.split(r"[.!?\n;]+", text or ""):
        for clause in re.split(r",\s+", sentence):
            clause = clause.strip()
            if clause:
                out.append(clause)
    return out


def extract_constraints(text: str, max_items: int = 6) -> List[str]:
    """Return the explicit-constraint clauses of a user message.

    A clause qualifies when it contains a negation marker, a
    participant-role assertion about the agent, or ALL-CAPS emphasis.
    Clauses are returned in message order, de-duplicated, trimmed to
    ``_MAX_CLAUSE_CHARS``, at most ``max_items`` of them.
    """
    found: List[str] = []
    seen = set()
    for clause in _clauses(text):
        if len(clause) < 8:
            continue
        if not (
            _NEGATION_RE.search(clause)
            or _PARTICIPANT_RE.search(clause)
            or _has_caps_emphasis(clause)
        ):
            continue
        cleaned = re.sub(r"\s+", " ", clause)[:_MAX_CLAUSE_CHARS]
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(cleaned)
        if len(found) >= max_items:
            break
    return found


def render_constraint_block(constraints: List[str],
                            header: str = "EXPLICIT USER CONSTRAINTS") -> str:
    """Render constraints as the prompt block shared by briefing/steers."""
    if not constraints:
        return ""
    lines = [f"### {header} (MUST HOLD — verify before finishing)"]
    for c in constraints:
        lines.append(f"- {c}")
    return "\n".join(lines)


# Participant-mode DETECTION over already-captured constraints (broader than
# `_PARTICIPANT_RE`, which decides clause capture): the chess sessions showed
# the binding clauses arrive as "with YOU", "play chess with you", "Ghost
# plays directly", "not a generated chess AI" — none of which the capture
# regex's verb forms cover. False positives are cheap (an extra steer
# paragraph); a false negative ships another coded stand-in.
_PARTICIPANT_MODE_RE = re.compile(
    r"\b(?:against|with|vs\.?)\s+you\b"          # "play chess with you"
    r"|\byou\b.{0,32}?\bplay(?:s|ing)?\b"        # "you ... will play"
    r"|\bplay(?:s|ing)?\b.{0,32}?\byou\b"        # "playing against you"
    r"|\bagainst\s+me\b|\bbetween\s+us\b"
    r"|\bplays?\s+directly\b"
    r"|\bnot\s+a\s+generated\b",
    re.IGNORECASE | re.DOTALL,
)


def has_participant_constraint(constraints: List[str]) -> bool:
    """True when any captured constraint assigns a PLAYER role to the agent
    itself ("with YOU", "you will play against me", "Ghost plays directly",
    "not a generated chess AI") — i.e. the deliverable must make the agent a
    runtime participant, not embed a coded stand-in."""
    return any(
        _PARTICIPANT_MODE_RE.search(c or "") for c in (constraints or [])
    )


# The architecture directive injected (post-write steer + coding-executor
# spec prompt) whenever a participant constraint is active. Deterministic
# counterweight to the observed rationalisation "the evaluation function IS
# me" (2026-07-04 chess session): names the only two designs that satisfy
# the constraint and the endpoint that already implements "the agent picks
# the move at inference time".
PARTICIPANT_STEER = (
    "PARTICIPANT-MODE ARCHITECTURE (binding): the user assigned a PLAYER "
    "role to YOU. No code you write can BE you — an embedded move picker "
    "(random.choice, minimax, evaluation loops, piece-square tables) "
    "violates this no matter what it is named. Exactly two valid designs: "
    "(A) conversational — authoritative game state lives in a workspace "
    "file; each chat turn you validate+apply the user's move with a short "
    "script, then YOU choose your reply move by reasoning and apply it; or "
    "(B) live client — the program the user runs on THEIR machine POSTs "
    "each position to your API: POST http://127.0.0.1:8000/api/game/move "
    "with JSON {\"fen\": ..., \"user_move\": ..., \"history\": [...]}; the "
    "response's 'move' is chosen by YOU at inference time (no engine "
    "fallback exists on that endpoint). The user's machine reaches that "
    "endpoint fine — ONLY your code sandbox cannot, so never test the call "
    "from the sandbox and never mock it; verify the script offline and ask "
    "the user to run it live. If what you just wrote embeds move-selection "
    "logic for your side, rewrite it to design (B) NOW."
)
