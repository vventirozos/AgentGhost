"""Streaming sanity guards — pure functions over the generation buffer.

This is the first module of the guard SEAM (IMPROVEMENTS.md #5). `core/agent.py`
grew to 11k+ lines because every hardening session appended another inline guard
to the streaming turn loop, each parametrised by yet another module constant at
the top of the file. These guards are pure functions of the buffer (no agent
state), so they belong in their own testable module — new stream guards should
land HERE and be referenced from the loop, not inlined into `handle_chat`.

`agent.py` re-exports these names for backward compatibility, so existing
references (and tests) keep working unchanged.
"""
import re

# --- thinking-loop (n-gram repetition) detector -----------------------------
THINKING_LOOP_PROBE_EVERY = 500     # run the repetition probe every N chars
THINKING_LOOP_WINDOW = 200          # length of the n-gram we look for
THINKING_LOOP_THRESHOLD = 3         # window appearing >= N times = loop
# Conservative (500/200/3): needs ~600 chars of genuine repetition before firing.
# The aggressive (300/150/2) was tuned to kill a weak model's enumeration loops
# fast at an "acceptable false-positive risk"; a strong model rarely produces
# those, so that cost (aborting legitimate reasoning that restates a constraint
# / loop invariant / repeated code pattern) now outweighs the benefit. The
# tool-call-collapse probe + the 200K char ceiling remain as fast backstops.

# --- paragraph-repeat detector (thinking channel only) ----------------------
# The n-gram probe needs the EXACT current 200-char tail to have occurred 3×,
# so a paraphrase loop that interleaves varied text between verbatim blocks
# evades it for a long time: the live 2026-07-25 planning loop (req f59a793d)
# repeated "Let me write this now."-style paragraphs dozens of times and ran
# to 19,586 chars (~3 minutes) before a probe finally landed inside a
# repeated block. Whole-line repetition is a much earlier signature of that
# shape: the same non-trivial LINE recurring several times in prose reasoning
# is a loop long before an exact 200-char window repeats.
#
# THINKING CHANNEL ONLY — callers must pass reasoning text, never content:
# generated code/data legitimately repeats identical lines (table rows,
# closing tags, fixture entries); prose reasoning does not.
#
# CONSERVATIVE by hard-won necessity (2026-07-25 second deploy): the first
# cut (min-line 30, threshold 4, no floor) fired on EVERY coding-leaf spec
# generation of the first live autoadvance run — one abort at just 3,019
# chars of reasoning — because legitimate spec planning restates the task
# description and field lists several times. That made the guard a thinking
# suppressor (every leaf built no-think). Current calibration: lines under
# 48 chars ("Let me check the file.") never count, six verbatim occurrences
# required, and buffers under 6K chars are never judged at all — a real
# runaway (dozens of repeats over tens of KB) still dies far earlier than
# the n-gram/extended-cap backstops, but multi-restatement planning passes.
PARAGRAPH_LOOP_MIN_LINE = 48       # ignore shorter lines (common phrases)
PARAGRAPH_LOOP_THRESHOLD = 6       # same line appearing >= N times = loop
PARAGRAPH_LOOP_SCAN_LINES = 3      # newest completed long lines checked/probe
PARAGRAPH_LOOP_MIN_BUF = 6_000     # never judge a buffer smaller than this


def _detect_paragraph_loop(reasoning_buf: str) -> bool:
    """True when one of the newest completed non-trivial LINEs of the
    thinking stream has already appeared ``PARAGRAPH_LOOP_THRESHOLD``
    times.

    Scans backwards from the tail for the ``PARAGRAPH_LOOP_SCAN_LINES``
    most recent lines of at least ``PARAGRAPH_LOOP_MIN_LINE`` chars
    (skipping the final, possibly still-streaming fragment) and counts
    each one's occurrences over the whole buffer — a few O(buffer)
    substring counts, same cost class as ``_detect_thinking_loop``, and
    intended to run on the same probe cadence. Checking a small window
    (not just the single newest line) keeps a novel interleaved line
    from masking the repeats around it."""
    if not reasoning_buf or len(reasoning_buf) < PARAGRAPH_LOOP_MIN_BUF:
        return False
    lines = reasoning_buf.splitlines()
    if len(lines) < 2:
        return False
    # § finalize/stream R1 B-4: count STANDALONE line repeats, not substring
    # occurrences. `buf.count(line)` also counted the line EMBEDDED in
    # otherwise-diverse prose — a debugging monologue quoting the same
    # ≥48-char error/traceback line six times (routine) fired the guard with
    # only two verbatim standalone repeats. A loop repeats the line AS a
    # line; a quote embeds it mid-sentence. One pass over the already-split
    # lines, same cost class as before.
    _stripped = [l.strip() for l in lines[:-1]]
    _counts: dict = {}
    for _s in _stripped:
        if len(_s) >= PARAGRAPH_LOOP_MIN_LINE:
            _counts[_s] = _counts.get(_s, 0) + 1
    # The last element may be a mid-line streaming fragment — check only
    # COMPLETED lines (everything before it), newest first.
    checked = 0
    for line in reversed(_stripped):
        if len(line) < PARAGRAPH_LOOP_MIN_LINE:
            continue
        if _counts.get(line, 0) >= PARAGRAPH_LOOP_THRESHOLD:
            return True
        checked += 1
        if checked >= PARAGRAPH_LOOP_SCAN_LINES:
            break
    return False


# --- sentence-frame run detector (thinking channel only) --------------------
# §4KV (req slack-124c85b8, 2026-10-01): a loop that repeats one sentence
# FRAME with one word changing — "I will check for any genius he has been
# praised as. I will check for any genius he has been admired as. …" — never
# repeats a 200-char window and never repeats a line, so both probes above
# stayed silent for 15,564 chars / 47.6 s (about 290 sentences) until the
# model happened to emit the same word six times. The same request's second
# loop ("I will stop. I will exit. I will finish. …") ran 4,014 chars.
#
# The signature is the run: N newest completed PROSE sentences in a row that
# open with the same two words. Measured on the main loop's reasoning in the
# detailed log (5,775 healthy turns of 300+ chars, 49 killed ones), at EVERY
# sentence end, not only on the probe's 500-char cadence (whose phase drifts
# in production): the longest such run in any healthy turn is 5; the two
# slack-12 loops die between about 1,600 and 2,150 chars at every phase
# (instead of 15,564 and 4,014).
#
# What does NOT count, by construction — each of these ENDS a run:
#   * a line that ends without sentence punctuation (a data row, a code
#     line, a heading): enumerating forty log rows is work, not a loop;
#   * a sentence that does not open with a letter (a bullet of any glyph, a
#     quoted line, a numbered item): twenty "- Test that …" bullets are a plan;
#   * a sentence whose DIGITS differ from the run's: "For i = 3, total
#     becomes 6." then "For i = 4, …" — a hand simulation or a case table
#     varies an INDEX, the loops this probe exists for vary a WORD (review,
#     §4KV: twenty iterations of a hand simulation fired the first cut; the
#     corpus's longest healthy run, 12, was one). Digits that stay the SAME
#     through the run ("… the 2024 laureate has been praised as.") are part
#     of the frame and count (second review: refusing every digit let the
#     slack-12 loop escape with one year added to it);
#   * a sentence shorter than the opener.
# A script written without spaces yields no two-word opener and the probe
# abstains. NOT caught, left to the other probes and the hard cap: sentences
# that end in `."`, `.)` or `.**`; a frame with an abbreviation in it ("Dr. "
# reads as a sentence end); two frames alternating; one stray sentence every
# nineteen; sentences averaging over 300 chars (twenty do not fit the window). A density rule that catches the last two fired on 3 healthy turns.
#
# Known cost: reasoning that legitimately writes twenty sentences in a row
# with one opener and no moving number is killed — a drafted anaphora, "If
# the input is …" over twenty cases, "The server on port 8080 …" twenty
# times. None in the corpus (longest: 5, "Let me …" planning).
#
# THINKING CHANNEL ONLY, MAIN LOOP ONLY — same reasons as the paragraph
# probe above; it has not been measured on coding-leaf spec thinking. (The
# corpus is the log's copy of the reasoning, which flattens line breaks
# inside a block: real streams have MORE line structure, and every
# line-structure rule here can only end a run.)
SENTENCE_RUN_THRESHOLD = 20        # consecutive same-opener sentences = loop
SENTENCE_RUN_OPENER_WORDS = 2      # the opener is the first N words
SENTENCE_RUN_TAIL = 6_000          # only the newest N chars are examined

# Where a piece ends: a sentence terminator followed by whitespace, or a line
# break. Pieces are cut BETWEEN boundaries in one linear pass. Two patterns
# were quadratic and are pinned out: one that matched whole sentences
# ("[^.!?\n]*[.!?]+(?=\s)", 200 ms a probe on a punctuation-free data
# dump) and a greedy terminator run ("[.!?]+(?=\s)", 145 ms on a window of
# punctuation with no whitespace).
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s)|\n")
_DIGITS_RE = re.compile(r"\d+")


def _sentence_opener_run(reasoning_buf: str) -> int:
    """Length of the run of newest COMPLETED prose sentences that open with
    the same ``SENTENCE_RUN_OPENER_WORDS`` words (case-folded). The trailing,
    possibly still-streaming fragment is never counted."""
    tail = reasoning_buf[-SENTENCE_RUN_TAIL:]
    pieces, start = [], 0
    for m in _SENTENCE_END_RE.finditer(tail):
        pieces.append(tail[start:m.end()])
        start = m.end()
    # (The window's first piece may be cut mid-sentence. It is read last, and
    # by then it can only end a run or lengthen one that already fills the
    # whole window — nothing to correct.)
    run, key = 0, None
    for piece in reversed(pieces):
        s = piece.strip()
        if not s:
            continue                   # a blank line is not a sentence
        if s[-1] not in ".!?":
            break                      # a row / code line / heading: not prose
        if not s[0].isalpha():
            break                      # a list item, a quoted line
        words = s.split()
        if len(words) < SENTENCE_RUN_OPENER_WORDS:
            break                      # a fragment
        # The frame: its opener AND its digits. An index that moves is not
        # the same frame; a constant year or version number is.
        frame = (" ".join(words[:SENTENCE_RUN_OPENER_WORDS]).lower(),
                 tuple(_DIGITS_RE.findall(s)))
        if key is None:
            key = frame
        elif frame != key:
            break
        run += 1
    return run


def _detect_sentence_run_loop(reasoning_buf: str) -> bool:
    """True when the thinking stream's newest ``SENTENCE_RUN_THRESHOLD``
    completed sentences all open with the same two words — a frame loop.
    O(tail) per probe; runs on the n-gram probe's cadence."""
    if not reasoning_buf:
        return False
    return _sentence_opener_run(reasoning_buf) >= SENTENCE_RUN_THRESHOLD


# --- tool-call generation-collapse detector ---------------------------------
# Qwen has been observed emitting 8000+ consecutive `<tool_call>` tokens with
# zero `</tool_call>` / `<function>` / `<parameter>`, burning 300+ s of decoder
# time before hitting max_tokens. The n-gram detector catches this eventually
# (after ~600 chars); this specialised probe fails fast after ~10 unclosed opens.
TOOL_CALL_LOOP_THRESHOLD = 10       # unclosed `<tool_call>` openings = collapse
TOOL_CALL_LOOP_PROBE_EVERY = 200    # run the probe every N chars of new content

# --- stream stop markers ----------------------------------------------------
_STREAM_STOP_MARKERS = ("</think", "<tool_call")


def _detect_thinking_loop(buf: str) -> bool:
    """True if the tail of `buf` repeats itself enough to be a loop.

    Single tight-window n-gram check. The old 400-char "wide backstop" was
    PROVABLY DEAD (§ finalize/stream R1 C-G2): every occurrence of the
    400-char tail contains an occurrence of the 200-char tail, so
    count(wide) >= threshold implies count(tight) >= threshold — the tight
    check above it had always already returned True. Removed so mutation
    sweeps stop reading the inert branch as an unpinned guard."""
    if len(buf) < THINKING_LOOP_WINDOW * THINKING_LOOP_THRESHOLD:
        return False
    tail = buf[-THINKING_LOOP_WINDOW:]
    return buf.count(tail) >= THINKING_LOOP_THRESHOLD


def _tail_has_stop_marker(buf: str, new_token: str) -> bool:
    """True if ``</think`` or ``<tool_call`` appears in the RECENT tail of buf.

    The streaming display latches ``stop_printing`` the first time either
    marker appears, then never checks again — but the old check lowercased the
    ENTIRE accumulated buffer on every chunk (O(n) per token → O(n²) over a
    long thinking stream, plus GB of transient string allocation on the event
    loop). A marker can straddle at most one chunk boundary, so scanning a tail
    of ``len(new_token) + 16`` chars is sufficient and O(1) per token.

    QUOTED-MENTION GUARD (2026-07-29 log audit): reasoning that QUOTES the
    rule text — ``the constraint says "Emit EXACTLY ONE `<tool_call>`
    block"`` — used to latch the mute mid-sentence, silently dropping every
    later thinking token of the turn from the log (the visible symptom was
    thinking lines cut right before a backtick). A marker immediately
    preceded by a backtick or quote character is a MENTION, not a stream
    transition; skip it. A real transition later in the same stream still
    latches — this check runs on every chunk."""
    window = len(new_token) + 16
    # +1 char of look-behind so the quote check works for a marker sitting
    # at the exact front of the scan window.
    tail = buf[-(window + 1):].lower()
    for m in _STREAM_STOP_MARKERS:
        idx = tail.find(m)
        while idx != -1:
            # A marker at the very front of the window (idx == 0) has an
            # UNKNOWN predecessor; `prev` is "" there, which is not a quote,
            # so the check below returns True — the conservative verdict.
            # (§4EC F13: an explicit idx-0 arm saying the same was redundant.)
            prev = tail[idx - 1] if idx > 0 else ""
            if prev not in ("`", '"', "'"):
                return True
            idx = tail.find(m, idx + 1)
    return False


def _detect_tool_call_loop(buf: str) -> bool:
    """True if the content buffer has accumulated too many unclosed
    `<tool_call>` openings — a decoder-collapse signature where the
    model is stuck emitting opening tags but never closing them.

    The healthy case is N opens + N closes (≥0 complete tool calls) or a
    single open waiting for its close. Anything where opens -
    closes > THRESHOLD is a run of openings with no progress, and we
    should kill the stream rather than let it run to max_tokens."""
    if not buf:
        return False
    opens = len(re.findall(r'<tool_call\b', buf, re.IGNORECASE))
    closes = len(re.findall(r'</tool_call\b', buf, re.IGNORECASE))
    return (opens - closes) > TOOL_CALL_LOOP_THRESHOLD


# --- native tool-call flood detector ----------------------------------------
# Sibling of `_detect_tool_call_loop` for the OTHER tool-call channel.
#
# That probe reads the CONTENT buffer and counts unclosed `<tool_call>` tags,
# so it is structurally blind whenever the upstream emits tool calls NATIVELY
# (OpenAI-shape `delta.tool_calls`) — which is this agent's default
# (`--native-tools`, main.py). In that mode a degenerate turn puts nothing in
# either guarded buffer: `agent.handle_chat` computes
# `guard_buf = reasoning_content if reasoning_content else full_content`, and
# a native flood leaves BOTH frozen (measured: 264 chars of reasoning, 0 chars
# of content) while the call list grows to the token cap. Every stream guard
# therefore saw a healthy stream and none of them fired.
#
# Measured consequence — three production floods, one signature:
#   2026-08-24 11:43  req 87e45af8   960 calls   294.8s of decode
#   2026-08-31 09:55  req 97b2dc8e   817 calls   251.2s
#   2026-08-31 13:12  req bench-ee   629 calls   237.7s
# each a self-play turn whose reasoning stopped mid-quote of the literal rule
# text ("emit EXACTLY ONE `") and then repeated one `execute` call until
# max_tokens. In the 629 case every duplicate was DISPATCHED — `execute` is
# blanket-mutating, so the batch dedup (read-safe allowlist) never collapses
# it — spawning 629 real `python3` processes and 629 tool results into the
# context window.
#
# TWO conditions, because a flood has two shapes:
#   * a run of byte-identical COMPLETED calls — the observed decoder collapse.
#     Fires in ~13 calls (~2-3s of decode) instead of ~5 minutes.
#   * a hard ceiling on the batch — catches a flood whose arguments vary
#     (e.g. the 144-identical-path `file_system` burst noted at the batch
#     dedup site, had the paths differed).
# Both are far above anything legitimate: the largest healthy batch in 27 days
# of production log is FOUR calls, and `delegate` caps its own fan-out at 4.
TOOL_CALL_BATCH_CEILING = 32     # calls in ONE assistant message = flood
NATIVE_TOOL_CALL_REPEAT = 12     # byte-identical COMPLETED calls in a row


def _native_call_identity(tc) -> tuple:
    """(name, arguments) of one accumulated native tool-call entry.

    Byte-level, deliberately: two calls that differ anywhere in their
    arguments are different work, and only EXACT repetition is evidence of a
    decoder collapse."""
    if not isinstance(tc, dict):
        return ("", "")
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return ("", "")
    return (str(fn.get("name") or ""), str(fn.get("arguments") or ""))


def _detect_native_tool_call_flood(tool_calls) -> bool:
    """True when the natively-streamed tool-call list has collapsed.

    ``tool_calls`` is the list being accumulated in place by the streaming
    loop, so the LAST entry is still receiving argument fragments and can
    look identical to its predecessor by mere prefix. Only COMPLETED
    entries (everything before the last) are compared — the cost of one
    extra call before firing, in exchange for never killing a live turn on
    a half-streamed argument string.

    O(NATIVE_TOOL_CALL_REPEAT) per call: the backwards scan stops at the
    first difference or at the threshold, so it is safe to run on every
    tool-call delta chunk."""
    if not tool_calls:
        return False
    if len(tool_calls) > TOOL_CALL_BATCH_CEILING:
        return True
    completed = tool_calls[:-1]
    if len(completed) < NATIVE_TOOL_CALL_REPEAT:
        return False
    run = 1
    ident = _native_call_identity(completed[-1])
    for tc in reversed(completed[:-1]):
        if _native_call_identity(tc) != ident:
            return False
        run += 1
        if run >= NATIVE_TOOL_CALL_REPEAT:
            return True
    return False
