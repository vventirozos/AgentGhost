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
from typing import List, Optional

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

# §4N MAJOR-2: the SELF-play lesson-verify replay (and any user who pastes
# ``###``-structured text) prepends the rendered SKILL PLAYBOOK into the user
# message. Its label lines — ``### SKILL PLAYBOOK:``, ``## RELEVANT LESSONS
# LEARNED``, ``TRIGGER (·):``, ``DOMAINS:``, ``ANTI-PATTERN:``,
# ``CORRECT-PATTERN:``, legacy ``SITUATION:``/``THE FIX:`` — qualify as
# constraints (ALL-CAPS emphasis / negation words) and filled all 6 MUST-HOLD
# slots, rendering ``- ANTI-PATTERN: use a relative path`` as an INVERTED
# instruction and crowding out the real request. These are system furniture,
# never a user constraint — recognised by their leading label and dropped
# before qualification. Tolerant of list markers / numbering / indentation.
# §4N R2 MAJOR-3: the original filter was case-INSENSITIVE, matched bare
# label words with no colon, and dropped ANY ``#``-led clause — so it
# destroyed REAL user constraints: ``Domain names must not contain
# underscores`` (DOMAIN), ``### IMPORTANT: don't touch prod`` (#-led),
# ``#1 rule: never force-push``. The playbook renders its labels ALL-CAPS
# WITH A COLON (``TRIGGER (·):``, ``DOMAINS:``, ``ANTI-PATTERN:`` …), so a
# CASE-SENSITIVE, colon-anchored match keeps the A-MAJOR-2 win without the
# false positives. Markdown-header furniture is restricted to the SPECIFIC
# injected section headers, not every ``#`` line.
_FURNITURE_LABEL_RE = re.compile(
    r"^[\s\-\*•>0-9.)]*"
    r"(?:TRIGGER\s*\(|DOMAINS?:|ANTI[\s\-]?PATTERN:|CORRECT[\s\-]?PATTERN:"
    r"|SITUATION:|PREVIOUS\s+MISTAKE:|THE\s+FIX:|CODE(?:\s+EXAMPLE)?\s*:"
    r"|FREQUENCY:|CONFIDENCE:)"
)  # case-SENSITIVE by design — real prose is mixed-case
# §4N R3 MINOR-2: case-SENSITIVE — the injected section headers are
# ALL-CAPS, but IGNORECASE dropped real user clauses that merely open with
# the same words in prose case ("## Past conversations should not be
# quoted", "### Memory Context: NEVER store my address"). Also covers the
# two ALL-CAPS headers R3 found uncovered (PAST EPISODES, RECENT LESSONS).
_FURNITURE_HEADER_RE = re.compile(
    r"^\s*#{1,6}\s*(?:SKILL\s+PLAYBOOK|RELEVANT\s+LESSONS|RECENT\s+LESSONS"
    r"|MEMORY\s+CONTEXT|PAST\s+CONVERSATIONS|PAST\s+EPISODES"
    r"|TOPOLOGICAL\s+KNOWLEDGE)"
)
# Back-compat alias for any external reader of the old name.
_FURNITURE_RE = _FURNITURE_LABEL_RE


def _is_furniture(clause: str) -> bool:
    """True for an injected system-context label line — an ALL-CAPS
    playbook label with a colon, or one of the specific injected section
    headers. Case-sensitive on labels so real mixed-case prose survives."""
    return bool(_FURNITURE_LABEL_RE.match(clause)
                or _FURNITURE_HEADER_RE.match(clause))


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
    # §4N MINOR-2: split sentences on .!? only at a real boundary (followed
    # by whitespace or end), plus \n and ; — a bare `.` inside a token
    # (``sniffer_probe.txt``, ``3.5``, ``/data/input.csv``) used to split a
    # filename across two constraint bullets on the operator's own requests.
    for sentence in re.split(r"(?:[.!?]+(?=\s|$)|[\n;]+)", text or ""):
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
        if _is_furniture(clause):
            continue  # §4N MAJOR-2: injected playbook/context label line
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


# ---------------------------------------------------------------------------
# START-WITH constraint enforcement on the ASSEMBLED reply
# ---------------------------------------------------------------------------
# Req 56221fad (2026-08-01): the model's FINAL turn opened with the mandated
# phrase, but the delivered reply is a multi-turn accumulation — an earlier
# turn's analysis block was prepended, burying the opener at char 2253. The
# verifier judged the assembled text and refuted "fails constraint to start
# with …" even though the model complied at the message level. Per-turn
# steers cannot fix an assembly-order problem, so the reply itself is
# repaired deterministically at finalize: if a start-with constraint is
# active and some later paragraph-boundary segment opens with the phrase,
# everything before that segment is pre-answer narration — drop it.

# DELIBERATELY NARROW (review catch 2026-08-01): a bare "start with X" is
# usually an ORDERING instruction ("Fix the bugs. START with the parser"),
# and this module's cheap-false-positive contract does not hold here — a
# misparse doesn't just re-render into a prompt, it DELETES delivered
# reply text via the hoist. A format mandate must therefore be explicit:
# a colon ("Start with: X"), a quoted phrase ("start with 'X'"), or a
# reply-shaped noun ("begin your reply with X").
_SW_PREFIX = r"^(?:(?:must|always|please)\s+)?(?:start|begin|open)s?\s+"
_SW_NOUNS = (r"(?:reply|replies|response|responses|answer|answers|message|"
             r"messages|deliverable|deliverables|file|files|document|"
             r"documents)")
_START_WITH_RES = (
    re.compile(_SW_PREFIX + r"(?:your|the|each|every|all)\s+" + _SW_NOUNS
               + r"\s+with\s*:?\s*(.{3,120})$", re.IGNORECASE),
    re.compile(_SW_PREFIX + r"with\s*:\s*(.{3,120})$", re.IGNORECASE),
    re.compile(_SW_PREFIX + r"with\s+['\"“”`](.{3,120}?)['\"“”`]\s*$",
               re.IGNORECASE),
)

# Cosmetic head noise that may legally precede the mandated phrase:
# horizontal rules, heading hashes, blockquote marks, bold/italic fences,
# stray quotes/backticks. Stripped iteratively before comparing.
_HEAD_NOISE_RE = re.compile(r"^(?:-{3,}\s*|#{1,6}\s+|>\s+|\*+|_+|['\"“”`]+)")

# A hoist that keeps less than this fraction of the reply is treated as
# suspicious (the phrase-led segment is a stub, not the answer) — fail open.
_START_WITH_MIN_KEEP = 0.4


def parse_start_with_phrase(constraints: List[str]) -> Optional[str]:
    """The mandated opening phrase, when any active constraint is an
    EXPLICIT reply-format mandate — "Start with: X", "start with 'X'",
    or "begin your reply with X". Bare ordering instructions
    ("start with the parser") deliberately return ``None``."""
    for c in constraints or []:
        text = str(c or "").strip()
        m = next((mm for rx in _START_WITH_RES
                  if (mm := rx.match(text))), None)
        if not m:
            continue
        phrase = m.group(1).strip().strip("'\"“”`").strip()
        if phrase.endswith((".", "!", "?")) and len(phrase) > 4:
            phrase = phrase[:-1].rstrip()
        if len(phrase) >= 3:
            return phrase
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _head_matches(segment: str, phrase: str) -> bool:
    """True when *segment* opens with *phrase* after cosmetic head noise."""
    s = (segment or "").lstrip()
    for _ in range(6):
        m = _HEAD_NOISE_RE.match(s)
        if not m:
            break
        s = s[m.end():].lstrip()
    return _norm(s).startswith(_norm(phrase))


def reply_satisfies_start_with(reply: str, phrase: str) -> bool:
    return _head_matches(reply or "", phrase)


def enforce_start_with(reply: str, constraints: List[str]):
    """Hoist the phrase-led segment of a multi-turn assembled reply to the
    head when a start-with constraint demands it.

    Returns ``(reply, dropped_chars)`` — unchanged input and 0 when no
    start-with constraint is active, the reply already complies, no
    paragraph segment opens with the phrase, the candidate cut would land
    inside a code fence, or the surviving tail is under
    ``_START_WITH_MIN_KEEP`` of the reply (fail open: deliver as-is and
    let the verifier report it)."""
    phrase = parse_start_with_phrase(constraints)
    if not phrase or not reply:
        return reply, 0
    if reply_satisfies_start_with(reply, phrase):
        return reply, 0
    segments = reply.split("\n\n")
    for i in range(1, len(segments)):
        if not _head_matches(segments[i], phrase):
            continue
        prefix = "\n\n".join(segments[:i])
        # Never cut inside an unclosed ``` fence — an odd fence count in
        # the dropped prefix means the "segment head" is mid-code.
        if prefix.count("```") % 2 == 1:
            return reply, 0
        tail = "\n\n".join(segments[i:])
        if len(tail) < _START_WITH_MIN_KEEP * len(reply):
            return reply, 0
        return tail, len(reply) - len(tail)
    return reply, 0


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


# ---------------------------------------------------------------------------
# Deterministic engine-pattern detection (participant-mode WRITE GUARD)
# ---------------------------------------------------------------------------
# The 2026-07-05 chess session shipped `random.choice(legal_moves)` straight
# through BOTH the PARTICIPANT MODE system-prompt rule and (in a later
# request) the post-write steer — steers ask the model to reconsider; only a
# guard refuses. These patterns feed `participant_write_violation`, the
# pre-dispatch check on file_system write/replace. Intentionally narrow:
# a false negative is still caught by the steer/verifier layers, while a
# false positive blocks a legitimate write outright.

# Unambiguous engine machinery — flagged wherever it appears.
_ENGINE_GLOBAL_PATTERNS = [
    (re.compile(r"\bminimax\b", re.IGNORECASE), "minimax search"),
    (re.compile(r"\bnegamax\b", re.IGNORECASE), "negamax search"),
    (re.compile(r"\balpha[\s_-]?beta\b", re.IGNORECASE),
     "alpha-beta pruning"),
    (re.compile(r"\bpiece[\s_-]?square\b", re.IGNORECASE),
     "piece-square tables"),
    (re.compile(r"\bstockfish\b", re.IGNORECASE),
     "external engine (stockfish)"),
]
# Randomness is only a move picker when the same content is recognisably
# move-generation code; a coin toss in an unrelated script must not trip
# the guard.
_ENGINE_RANDOM_RE = re.compile(
    r"\brandom\.(?:choice|choices|sample|shuffle|randint|randrange)\s*\("
    r"|\bMath\.random\s*\("
)
_GAME_MOVE_CONTEXT_RE = re.compile(
    r"legal_moves|legalMoves|import\s+chess|from\s+chess\s+import"
    r"|chess\.js|generate_moves|movegen",
    re.IGNORECASE,
)


def find_engine_patterns(content: str) -> List[str]:
    """Human-readable labels of embedded move-selection logic in *content*.

    Deterministic, no LLM. Empty list means clean."""
    text = content or ""
    hits = [label for rx, label in _ENGINE_GLOBAL_PATTERNS
            if rx.search(text)]
    if _ENGINE_RANDOM_RE.search(text) and _GAME_MOVE_CONTEXT_RE.search(text):
        hits.append("randomised move picker (random.* over game moves)")
    return hits


def participant_write_violation(constraints: List[str],
                                tool_args: dict) -> Optional[str]:
    """Pre-dispatch guard for ``file_system`` write/replace calls.

    Returns the SYSTEM BLOCK message when a participant constraint is
    active and the write body embeds move-selection logic; ``None``
    otherwise. Pure function (agent.py calls it before dispatching the
    tool) so the whole guard is unit-testable without an agent loop.
    Scans ``content`` AND ``replace_with``: the replace→write
    auto-promotion means either argument can become the full file body.
    """
    if not constraints or not has_participant_constraint(list(constraints)):
        return None
    args = tool_args or {}
    op = str(args.get("operation") or "")
    if op not in ("write", "replace"):
        return None
    body = "\n".join(str(args.get(k) or "")
                     for k in ("content", "replace_with"))
    if not body.strip():
        return None
    hits = find_engine_patterns(body)
    if not hits:
        return None
    path = str(args.get("path") or args.get("filename") or "?")
    return (
        f"SYSTEM BLOCK (participant-mode engine guard): the {op} to "
        f"'{path}' was ABORTED before execution because its content embeds "
        f"move-selection logic — {'; '.join(hits)} — while the user's "
        f"explicit constraint assigns the PLAYER role to YOU. No code you "
        f"write can BE you. " + PARTICIPANT_STEER +
        " Rewrite the artifact with the embedded move picker removed, then "
        "write it again."
    )
