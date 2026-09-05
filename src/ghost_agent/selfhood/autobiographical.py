"""Autobiographical memory — first-person experiential records.

Proposal item #1: a persistent log of "what it was like" to be the
agent on a given turn. Distinct from the distill ``Trajectory`` log
(which captures full structured tool traces for ML training); this
file is the agent's own first-person diary, summarised post-turn.

Each entry shares an id with the trajectory it summarises so the two
stay linked. The recognition layer can then cross-reference: a
retrieved experience can be traced back to the original tool trace
when needed, but the routine read path only needs the summary.

Storage: a single JSONL file under ``$GHOST_HOME/system/selfhood/
autobiographical.jsonl``. Append-only, line-buffered, sink failures
swallowed. Same shape as ``distill.collector`` so the agent's hot
path stays cheap.

Provenance & item #2 — continuity: every entry is tagged
``subject="self"``. Retrieval layers treat ``subject=="self"`` records
as "mine" rather than as external knowledge. Future-self reads its
own past in the first person.
"""

from __future__ import annotations

import json
import logging
import math
import datetime
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .schema import Experience, _utcnow_iso

logger = logging.getLogger("GhostSelfhood")


AUTOBIO_FILENAME = "autobiographical.jsonl"

# Bounded growth (mirrors workspace/activity.py). Without a cap the log grew
# monotonically (877 KB / 1.7k lines at ~6 weeks) while every turn did 3-4
# full-file O(n) parses + one O(n) rewrite — quadratic over the agent's
# lifetime. Once the file passes the byte cap on append we keep the newest N
# entries and roll the dropped ones into a single summary record so the
# narrative layer still knows history existed.
_COMPACT_MAX_BYTES = 2 * 1024 * 1024
_COMPACT_KEEP_LINES = 2000


# Lightweight, zero-dependency topic clustering. The `cluster` field on
# Experience used to be inert (set to None on every record); deriving a
# coarse label at capture time lets the narrative / recall layers say
# "I've done this class of thing before" instead of treating every turn
# as unique. Order is irrelevant — the label with the most keyword hits
# wins, ties broken by dict order.
_CLUSTER_KEYWORDS = {
    "debugging": ("error", "fix", "debug", "crash", "exception", "traceback",
                  "didn't land", "failed"),
    "coding": ("code", "python", "function", "script", "compile", "syntax",
               "import", "refactor", "implement"),
    "data": ("sql", "query", "database", "postgres", "csv", "dataframe",
             "table", "schema"),
    "research": ("search", "research", "look up", "web", "wikipedia",
                 "fact check", "investigate"),
    "writing": ("write", "essay", "draft", "summary", "document", "article"),
    "memory": ("remember", "recall", "memory", "profile", "knowledge base"),
    "self_play": ("synthetic training", "self-play", "self play", "challenge",
                  "exercise"),
    "math": ("calculate", "compute", "equation", "arithmetic", "number"),
    # Introspection / philosophy bin. Previously a turn like "are you
    # self-aware?" would land in cluster=None because no engineering
    # keyword matched. The selfhood module's own subject matter
    # deserves its own bin so the narrative can say "I keep returning
    # to questions about my own attention" instead of treating each
    # such turn as unique.
    "meta": ("consciousness", "self-aware", "self aware", "attention",
             "phenomenology", "introspect", "introspection", "identity",
             "subjective", "qualia", "mood", "feel", "experience",
             "what it's like", "first-person", "self-model", "selfhood"),
}


# Templates whose user prompt is system-generated and effectively
# identical across many turns. Capturing each one as its own
# autobiographical record floods the diary with near-duplicates and
# starves the narrative regeneration of variety. We roll them up
# instead: only the first occurrence inside a short window writes a
# full record; subsequent ones bump a counter on the rollup.
_TEMPLATE_PROMPT_MARKERS = (
    "### SYNTHETIC TRAINING EXERCISE",
    "SYSTEM JUDGE REJECTION",
    "AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR",
    "SYSTEM ALERT: Your previous turn entered a self-repeating",
)


def _template_marker_for(text: str) -> Optional[str]:
    """Return the marker that matches the start of ``text``, or None.

    Match is on the leading prefix because system templates always
    place their banner at the head of the user message."""
    if not text:
        return None
    head = text.lstrip()[:120]
    for marker in _TEMPLATE_PROMPT_MARKERS:
        if head.startswith(marker):
            return marker
    return None


# PII redaction. We only run this against ``user_first_words`` (the
# short prompt prefix stored on every record) because the rest of the
# Experience is agent-generated prose. Patterns are deliberately
# conservative — false positives in the diary cost more than a missed
# leak does.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# A phone match must carry phone STRUCTURE — a leading `+`, parenthesised
# area code, or internal separators. The old core (`\d{3}[ -]?\d{4}` with
# everything else optional) matched any bare 7-10 digit integer, mangling
# numeric literals (row counts, ids) quoted in prompt prefixes. Kept in
# sync with distill.redact's `phone` rule.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\+\d{1,3}[ -]?(?:\(?\d{2,4}\)?[ -]?)?\d{3}[ -]?\d{4}"
    r"|\(\d{2,4}\)[ -]?\d{3}[ -]?\d{4}"
    r"|(?:\d{1,3}[ -])?\d{2,4}[ -]\d{3}[ -]?\d{4}"
    r"|\d{3}[ -]\d{4}"
    r")(?!\d)"
)
# `sk`/`pk` must be followed by a `-`/`_` separator (sk-live…, sk_test…)
# — as a bare word-prefix they ate ordinary words ("skateboarding" →
# "[REDACTED_KEY] yesterday").
_API_KEY_RE = re.compile(
    r"\b(?:(?:sk|pk)[-_]|ghp|github_pat|AKIA|AIza|xoxb|xoxp)[A-Za-z0-9_-]{8,}\b",
    re.IGNORECASE,
)
# Assignment-shaped leaks (`api_key=abcd1234efgh`, `password: hunter2xyz`)
# — the common real shape the prefix rule can't see because the separator
# splits the token. The key name is kept so the diary line stays readable.
_KV_SECRET_RE = re.compile(
    r"\b(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|token|"
    r"password|passwd|secret)\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+]{8,}[\"']?",
    re.IGNORECASE,
)
# Letter-excluding boundaries: digit runs inside longer alphanumeric
# tokens (32-hex experience/trajectory ids) are identifiers, not cards —
# redacting them corrupts the ids the diary joins on.
_CREDIT_CARD_RE = re.compile(r"(?<![0-9A-Za-z])(?:\d[ -]?){13,19}(?![0-9A-Za-z])")


def _luhn_ok(digits: str) -> bool:
    """True when `digits` passes the Luhn checksum (every real PAN does)."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_cc_if_luhn(m) -> str:
    """Redact a 13-19 digit run only when it Luhn-validates — bigint ids
    and epoch-millis literals quoted in prompts otherwise get eaten."""
    return "[REDACTED_CC]" if _luhn_ok(re.sub(r"\D", "", m.group(0))) else m.group(0)


_ROLLUP_PHRASES = {
    "### SYNTHETIC TRAINING EXERCISE": "synthetic training exercises",
    "SYSTEM JUDGE REJECTION": "judge-rejected attempts",
    "AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR": "auto-diagnostic error fixups",
    "SYSTEM ALERT: Your previous turn entered a self-repeating":
        "self-repeating-thinking-loop alerts",
}


def _rollup_summary(marker: str, count: int) -> str:
    label = _ROLLUP_PHRASES.get(marker, "system-template turns")
    if count <= 1:
        return f"I worked on one of my recurring {label}."
    return f"I worked on {count} of my recurring {label} in a row."


def redact_pii(text: str) -> str:
    """Best-effort PII scrub: emails, phone numbers, API keys, credit
    cards. Returns the input unchanged when nothing matches. Designed
    to be cheap enough to call on every turn's user_first_words."""
    if not text:
        return text
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = _API_KEY_RE.sub("[REDACTED_KEY]", out)
    out = _KV_SECRET_RE.sub(lambda m: f"{m.group(1)}=[REDACTED_KEY]", out)
    out = _CREDIT_CARD_RE.sub(_redact_cc_if_luhn, out)
    out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
    return out


#: `recent()` over-scan when a filter is active (see its docstring).
_RECENT_FILTER_OVERSCAN = 8
_RECENT_FILTER_MIN_TAIL = 64


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens — shared by the recall scorer and clustering."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


#: Length cap for `answer_gist`. 160 chars is one or two sentences — enough
#: to carry a fact ("born 12 March 2026", "the table is 41 GB with toast")
#: without turning the diary into a transcript. Sized against the 1.1 MB
#: live log: ~1,800 rows × 160 B adds under 300 KB before compaction.
ANSWER_GIST_MAX_CHARS = 160
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def answer_gist(final_response: str, *,
                max_chars: int = ANSWER_GIST_MAX_CHARS) -> str:
    """The "what I said" half of an experience: a short, single-line,
    redacted excerpt of the agent's own final reply.

    Reasoning blocks are dropped first (a gist of the think-trace is not an
    answer), whitespace is collapsed, and PII is redacted BEFORE the clip —
    clipping first can cut a secret in half and leave the pattern
    unmatchable (the same rule `_outcome_phrase` documents for failure
    reasons). Empty in → empty out; callers store "" rather than a
    placeholder so a missing gist never reads as an answer."""
    text = _THINK_BLOCK_RE.sub(" ", str(final_response or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = redact_pii(text)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _derive_cluster(text: str) -> Optional[str]:
    """Best-effort coarse topic label for a turn. None when nothing matches."""
    low = (text or "").lower()
    if not low.strip():
        return None
    best: Optional[str] = None
    best_hits = 0
    for label, keywords in _CLUSTER_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in low)
        if hits > best_hits:
            best_hits, best = hits, label
    return best


def _outcome_phrase(outcome: str, failure_reason: str = "") -> str:
    """The trailing clause of a first-person turn summary. Extracted so the
    capture path and the post-hoc outcome backfill phrase verdicts identically.

    The reason is REDACTED here (2026-08-21 review of queue #7), which is the
    one place every writer passes through — capture, the inline backfill, the
    late verdict, a human label, and the reconciler script. It had been the
    diary's redaction hole: ``capture_turn`` scrubs the quoted user_request
    at the boundary, but the failure reason went in raw, and the reasons are
    not the agent's own prose — they are verifier ``issues``/``reasoning``
    (which quote tool output) and, on the user-correction path, text derived
    from the USER'S OWN correcting message. The trajectory corpus's sibling
    writer has always redacted its reason for exactly this ("redaction runs
    on every write is the package contract"); the diary is the surface that
    gets read back INTO prompts, so it needed it more, not less. Redact
    BEFORE the 120-char clip — clipping first can cut a secret in half and
    leave the pattern unmatchable."""
    out = (outcome or "unknown").lower()
    if out == "passed":
        return "and the answer landed"
    if out == "failed":
        reason = redact_pii(
            (failure_reason or "").strip().replace("\n", " "))[:120]
        return f"and it didn't land: {reason}" if reason else "and it didn't land"
    return "without a verdict either way"


# The three verdict clauses `_outcome_phrase` can produce. A backfill has to
# find whichever one the summary ALREADY carries: patching only the "unknown"
# clause (all this did until 2026-08-21) leaves a row that was captured
# FAILED and later upgraded reading "…and it didn't land: <reason>." under
# `outcome="passed"` — the prose contradicting the field, on the surface the
# narrative layer and recall actually read. Three such rows exist on the live
# box from July. The corpus refuses the same incoherence at both ends
# ("`passed` + a failure reason is an incoherent record"); the diary should
# not be looser about its own prose.
_VERDICT_CLAUSE_STARTS = (
    "without a verdict either way",
    "and the answer landed",
    "and it didn't land",
)

#: The unknown-outcome clause, by name — the ONE authority for the phrase
#: the read surfaces strip (`strip_no_verdict_clause`). It stays in the
#: STORED summary on purpose: the late-verdict backfill
#: (`_swap_verdict_clause`) finds and replaces it there, and 39% of live
#: rows are waiting for exactly that. Only rendering drops it.
NO_VERDICT_CLAUSE = _VERDICT_CLAUSE_STARTS[0]


def strip_no_verdict_clause(summary: str) -> str:
    """Render-side twin of `_swap_verdict_clause`: drop a TRAILING
    "without a verdict either way" (a reader gains nothing from it — the
    missing `[outcome]` tag already says the turn was never judged).

    Only the tail is eligible: the clause must run to the end of the
    string (modulo the final period). A decoy inside the quoted user
    request ("…without a verdict either way, what do you think?") is
    therefore left alone — the truncation hazard `_swap_verdict_clause`
    documents cannot arise here, because a non-tail match is not a
    match. A summary without the clause is returned unchanged."""
    s = summary or ""
    i = s.rfind(NO_VERDICT_CLAUSE)
    if i < 0:
        return s
    if s[i:].rstrip().rstrip(".").strip() != NO_VERDICT_CLAUSE:
        return s
    head = s[:i].rstrip()
    if s.rstrip().endswith(".") and not head.endswith("."):
        head += "."
    return head


def _swap_verdict_clause(summary: str, new_clause: str) -> str:
    """Replace the summary's existing verdict clause with ``new_clause``.

    The clause is always LAST in the template
    (``summarise_turn_first_person``), so it runs to the end of the string —
    which is what makes this safe: a failure reason may itself contain a
    period ("cannot open file.txt"), and cutting at the first period after
    the clause start would splice the record. Returns the summary unchanged
    when it carries no clause at all (rollup records).

    ⚠ Matched from the RIGHT, and that is load-bearing. The summary QUOTES
    the user's request, so a request containing one of these phrases
    ("you claimed X and the answer landed but it did not") plants a decoy
    earlier in the string — and because this rule cuts to the END, a
    leftmost match does not merely mis-phrase the verdict, it TRUNCATES the
    record mid-quote and drops the tool phrase entirely. Taking the last
    match keeps the decoy in the quote where it belongs. (The pre-2026-08-21
    code substituted the clause in place, so it could mis-phrase but never
    truncate — this cut-to-end rule is what made leftmost matching
    dangerous, i.e. the hazard arrived with the fix.) Residual, documented:
    a failure REASON that itself ends in one of these phrases loses its
    tail — degraded prose, never a truncated quote, and the reason is being
    replaced anyway."""
    best = -1
    for start in _VERDICT_CLAUSE_STARTS:
        i = summary.rfind(start)
        if i > best:
            best = i
    if best < 0:
        return summary
    tail = "." if summary.rstrip().endswith(".") else ""
    return summary[:best] + new_clause + tail


class AutobiographicalMemory:
    """Append-only first-person experience log.

    The summary slot is the load-bearing field. A turn that does not
    produce one is not autobiographically significant and the writer
    refuses it (an experience with no narrative content is just a
    duplicate of the trajectory log)."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / AUTOBIO_FILENAME
        self.refcount_path = self.root / "reference_counts.json"
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        # IDF / search cache. (mtime, size) → (experiences, haystacks,
        # idf_table). Lazily populated on first search; invalidated when
        # the file changes on disk between calls.
        self._search_cache: Dict[tuple, tuple] = {}
        # In-memory reference-count overlay so the hot path doesn't
        # round-trip through disk on every prefix-utility increment.
        self._ref_counts: Optional[Dict[str, int]] = None

    def append(self, exp: Experience) -> Optional[Path]:
        """Write ``exp`` to the log. Returns the path on success,
        None on disabled / failed / empty-summary.

        Never raises — autobiographical capture is secondary."""
        if not self.enabled:
            return None
        if not exp.summary or not exp.summary.strip():
            # An empty-summary experience would be a duplicate of the
            # trajectory log without adding any first-person content.
            # Refuse silently — callers that have no summary to write
            # should not write at all.
            return None
        # Template-prompt rollup: SYNTHETIC TRAINING / JUDGE REJECTION /
        # AUTO-DIAGNOSTIC turns share a banner and recur dozens of times
        # in a row. Bumping a counter on a single rollup record beats
        # writing 100 near-identical entries — the narrative phase can
        # still say "I did 100 synthetic exercises this week" without
        # the diary being drowned in template noise.
        marker = _template_marker_for(exp.user_first_words)
        if marker is not None:
            if self._bump_template_rollup(marker, exp):
                return self.path
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(exp.to_jsonl())
                    f.write("\n")
                    f.flush()
                self._maybe_compact_locked()
            return self.path
        except Exception as e:
            logger.warning("autobiographical append failed: %s", e)
            return None

    def _maybe_compact_locked(self) -> None:
        """Rewrite the log keeping the newest ``_COMPACT_KEEP_LINES`` entries
        once it passes the byte cap, rolling the dropped entries into ONE
        summary record at the head so the narrative layer still knows earlier
        history existed. Caller holds ``self._lock``. Best-effort — a failed
        compaction must never fail the append that triggered it."""
        try:
            if self.path.stat().st_size <= _COMPACT_MAX_BYTES:
                return
            from collections import deque
            with self.path.open("r", encoding="utf-8") as f:
                tail = deque(f, maxlen=_COMPACT_KEEP_LINES)
            # Count what we're dropping so the summary is honest.
            with self.path.open("r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            dropped = max(0, total - len(tail))
            tmp = self.path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                if dropped > 0:
                    summary_exp = Experience(
                        summary=(f"[Consolidated] {dropped} earlier autobiographical "
                                 f"entries were compacted out of the diary to bound "
                                 f"its size; the newest {len(tail)} are retained."),
                        subject="self",
                    )
                    f.write(summary_exp.to_jsonl())
                    f.write("\n")
                f.writelines(tail)
            tmp.replace(self.path)
            # The on-disk file changed → drop the IDF/search cache so the next
            # search rebuilds against the compacted contents.
            self._search_cache = {}
            logger.info(
                "autobiographical log compacted to newest %d entries (dropped %d)",
                len(tail), dropped,
            )
        except Exception as e:
            logger.warning("autobiographical compaction failed: %s", e)

    # ------------------------------------------------------------------
    # Template rollup helpers
    # ------------------------------------------------------------------

    def _bump_template_rollup(self, marker: str, fresh: Experience) -> bool:
        """If the tail of the log is a rollup record for ``marker``,
        bump its count + update its tail timestamp in place. Returns
        True when an existing rollup was extended (and the caller
        should NOT also append the fresh record).

        We restrict the scan to the last few lines: a rollup window
        spanning hours and arbitrary other turns would defeat the
        point. The natural caller is the agent's hot loop, where these
        templates fire in dense bursts."""
        # Hold the lock across the ENTIRE read → walk → write → replace
        # (and the fresh-append fallback). Previously the read was locked
        # but released before tmp.replace, so a concurrent append (hot
        # loop) or watchdog write landing between our read and our replace
        # was silently clobbered by the stale snapshot.
        with self._lock:
            # Skip the tail scan entirely on a missing file — we'll fall
            # straight through to opening a fresh rollup record below.
            if not self.path.exists():
                lines: List[str] = []
            else:
                try:
                    lines = self.path.read_text(encoding="utf-8").splitlines()
                except OSError as e:
                    logger.warning("autobiographical rollup read failed: %s", e)
                    lines = []
            try:
                # Walk back through the last 5 lines looking for a rollup
                # we already opened for this marker. Skip plain whitespace;
                # bail on the first non-rollup, non-matching entry so
                # unrelated turns can't accidentally extend the window.
                for i in range(len(lines) - 1, max(-1, len(lines) - 6), -1):
                    s = lines[i].strip()
                    if not s:
                        continue
                    try:
                        d = json.loads(s)
                    except json.JSONDecodeError:
                        break
                    if d.get("template_marker") == marker:
                        d["template_count"] = int(d.get("template_count") or 1) + 1
                        d["timestamp"] = fresh.timestamp
                        # Refresh tools_used to the union — the rollup
                        # should reflect what got reached for across the
                        # burst, not just the first turn's tools.
                        existing_tools = list(d.get("tools_used") or [])
                        for t in fresh.tools_used:
                            if t not in existing_tools:
                                existing_tools.append(t)
                        d["tools_used"] = existing_tools[:10]
                        # Re-render the summary with the new count.
                        d["summary"] = _rollup_summary(marker, d["template_count"])
                        lines[i] = json.dumps(d, ensure_ascii=False)
                        tmp = self.path.with_suffix(".jsonl.tmp")
                        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        tmp.replace(self.path)
                        return True
                    break
            except Exception as e:
                logger.warning("autobiographical rollup bump failed: %s", e)
            # No matching tail entry — open a fresh rollup with count=1.
            fresh_dict = fresh.to_dict()
            fresh_dict["template_marker"] = marker
            fresh_dict["template_count"] = 1
            fresh_dict["summary"] = _rollup_summary(marker, 1)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(fresh_dict, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
            except Exception as e:
                logger.warning("autobiographical rollup open failed: %s", e)
                return False
            return True

    # ------------------------------------------------------------------
    # Session-boundary marker (#9)
    # ------------------------------------------------------------------

    def mark_session_boot(self, *, prior_session_at: str = "") -> Optional[Path]:
        """Write a synthetic boot-event record so the narrative can say
        "after a few hours off, I started a new session and…" instead of
        having to infer session boundaries from timestamps.

        No-ops when the most recent record is already a boot marker
        with a timestamp within the same minute — protects against
        crash-restart loops emitting a flurry of boot events."""
        if not self.enabled:
            return None
        try:
            recent_list = self.recent(limit=1)
            if recent_list and getattr(recent_list[0], "outcome", "") == "boot":
                last_ts = recent_list[0].timestamp or ""
                # Drop the seconds slice — same-minute restarts are noise.
                if last_ts[:16] == _utcnow_iso()[:16]:
                    return None
        except Exception:
            pass
        ago = ""
        if prior_session_at:
            ago = f" (last active {prior_session_at})"
        exp = Experience(
            summary=f"Session resumed{ago}.",
            outcome="boot",
            cluster="meta",
            user_first_words="(session boot)",
        )
        return self._raw_append(exp)

    def _raw_append(self, exp: Experience) -> Optional[Path]:
        """Bypass the template-rollup / empty-summary checks. Internal
        helper used by boot markers, which legitimately have a fixed
        synthetic summary and must not be conflated with the template
        rollup."""
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(exp.to_jsonl())
                    f.write("\n")
                    f.flush()
            return self.path
        except Exception as e:
            logger.warning("autobiographical raw append failed: %s", e)
            return None

    def iter_experiences(self) -> Iterator[Experience]:
        """Stream experiences from disk, oldest first."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        yield Experience.from_dict(d)
                    except Exception:
                        continue
        except OSError as e:
            logger.warning("cannot read autobiographical log %s: %s", self.path, e)

    def recent(self, limit: int = 5, *, include_boots: bool = True,
               hours: Optional[float] = None) -> List[Experience]:
        """Return the most recent N experiences, newest last so the
        natural caller can do ``for e in mem.recent(3): print(e)`` and
        read in chronological order.

        Reads only the tail of the file (a bounded deque over the lines)
        instead of materialising the whole log — `recent()` is on the
        per-turn hot path (wake-up prefix, reference-note), and a full
        O(n) parse per call was quadratic over the log's monotonic growth.

        ``include_boots=False`` drops session-boot markers (2026-09-05):
        they are 25% of the live log and one of five "recent experiences"
        rendered by introspect was "Session resumed". The default stays
        True because the narrative summariser reads boots on purpose —
        they structure its "after a few hours off…" diary voice.
        ``hours`` keeps only rows younger than the window; a row whose
        timestamp does not parse is KEPT (unknown age is not old age).
        With either filter the tail over-scan is ``limit × 8`` lines
        (floor 64) — a boot-heavy tail can therefore return fewer than
        ``limit`` rows rather than reading the whole file."""
        if limit <= 0:
            return []
        if not self.path.exists():
            return []
        from collections import deque
        filtering = (not include_boots) or (hours is not None)
        scan = (max(limit * _RECENT_FILTER_OVERSCAN, _RECENT_FILTER_MIN_TAIL)
                if filtering else limit)
        try:
            with self.path.open("r", encoding="utf-8") as f:
                tail_lines = deque(f, maxlen=scan)
        except OSError as e:
            logger.warning("cannot read autobiographical log %s: %s", self.path, e)
            return []
        cutoff = None
        if hours is not None:
            cutoff = (datetime.datetime.now(datetime.timezone.utc)
                      - datetime.timedelta(hours=float(hours)))
        items: List[Experience] = []
        for line in tail_lines:
            line = line.strip()
            if not line:
                continue
            try:
                exp = Experience.from_dict(json.loads(line))
            except Exception:
                continue
            if not include_boots and exp.outcome == "boot":
                continue
            if cutoff is not None:
                from .mood import parse_iso_utc
                ts = parse_iso_utc(exp.timestamp)
                if ts is not None and ts < cutoff:
                    continue
            items.append(exp)
        return items[-limit:] if filtering else items

    def recent_verdicts(
        self,
        limit: int = 5,
        *,
        scan_limit: int = 200,
        max_age_days: Optional[float] = None,
        now=None,
    ) -> List[str]:
        """Outcomes ("passed"/"failed") of the newest VERDICT-BEARING
        experiences, oldest → newest. Scans back through the newest
        ``scan_limit`` raw lines — NOT ``limit`` raw lines — because the
        raw tail is dominated by boot markers and unknown-verdict turns
        (live mix ≈ 19% verdict-bearing, and a crash-restart burst can
        stack 8+ boot markers in a row): a raw-tail read of N records
        yields ~0 verdicts and starves every streak-derived mood
        (review R2). ``max_age_days`` drops verdicts whose record
        timestamp is older (or unparseable) — without it, weeks-old
        verdicts reached this deep can mint a "fresh" streak mood
        (review R3). Bounded: the deque holds at most ``scan_limit``
        lines and the file itself is compaction-capped. Never raises."""
        if limit <= 0 or not self.path.exists():
            return []
        from collections import deque
        try:
            with self.path.open("r", encoding="utf-8") as f:
                tail_lines = deque(f, maxlen=max(int(scan_limit), limit))
        except OSError as e:
            logger.warning(
                "cannot read autobiographical log %s: %s", self.path, e)
            return []
        cutoff_secs = (None if max_age_days is None
                       else max(0.0, float(max_age_days)) * 86400.0)
        # Function-level import: mood.py is a leaf module (stdlib only),
        # so no cycle — but keep it lazy to match the file's style.
        from .mood import age_seconds as _age
        verdicts: List[str] = []
        for line in tail_lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("outcome") not in ("passed", "failed"):
                continue
            if cutoff_secs is not None:
                age = _age(str(d.get("timestamp") or ""), now=now)
                if age is None or age > cutoff_secs:
                    continue
            verdicts.append(d["outcome"])
        return verdicts[-limit:]

    def get_by_ids(self, ids: Iterable[str]) -> List[Experience]:
        """Fetch experiences by id, in log order; unknown ids are silently
        skipped. Served from the cached search index so repeated calls
        don't re-parse the log. Never raises."""
        wanted = {i for i in (ids or ()) if i}
        if not wanted:
            return []
        try:
            experiences, _, _ = self._search_index()
        except Exception:
            return []
        return [e for e in experiences if e.id in wanted]

    def search_my_past(self, query: str, limit: int = 5) -> List[Experience]:
        """Relevance-ranked search over my own past. Deliberately
        dependency-free — no embedder, no vector store — but scored with
        inverse-document-frequency weighting so a rare, distinctive query
        term (e.g. "trapdoor") dominates a common one (e.g. "the").

        That is the difference between "recent-N" recall and recall that
        actually surfaces the relevant memory: a salient experience from
        hundreds of turns ago becomes reachable as long as the query
        shares its distinctive vocabulary.

        Returned best-match first, newest-first on ties; below the minimum
        score (one matched token) returns nothing.

        Why not vector search? The autobiographical log is small and
        append-only; IDF-weighted overlap is competitive AND zero-dep.
        A caller wanting full embeddings can still route through
        ``memory_system`` with ``subject:"self"`` metadata."""
        if not query or limit <= 0:
            return []
        q_tokens = set(t for t in _tokenize(query) if len(t) > 2)
        if not q_tokens:
            return []

        experiences, haystacks, doc_freq = self._search_index()
        if not experiences:
            return []

        n = len(experiences)
        # Build idf from cached document frequencies — most callers
        # share the same vocabulary, so the cache cost is amortised
        # across many queries. Tokens not in cached df are scored 0,
        # which is the correct answer (no document contains them).
        idf = {
            t: math.log((n + 1) / (doc_freq.get(t, 0) + 1)) + 1.0
            for t in q_tokens
        }

        # Reference-count prior: memories the agent has actually reached
        # for before (their summary echoed in a past reply — tracked in
        # reference_counts.json) are more useful than ones that merely
        # match vocabulary. Previously this signal was WRITTEN but never
        # READ — a dead loop. Folding `(1 + log1p(refs))` into the score
        # makes "memories that paid off" rank higher, the relevance prior
        # the design always intended. Cheap: ref counts are an in-memory
        # dict loaded once.
        scored: List[tuple] = []
        for exp, hs_tokens in zip(experiences, haystacks):
            # A boot marker is not an experience: "session"/"boot" match
            # 600 identical rows on the live log and bury real recall.
            if exp.outcome == "boot":
                continue
            score = sum(idf[t] for t in q_tokens if t in hs_tokens)
            if score > 0:
                refs = self.reference_count(exp.id)
                score *= (1.0 + math.log1p(refs))
                scored.append((score, exp))
        scored.sort(key=lambda s: (s[0], s[1].timestamp), reverse=True)
        return [exp for _, exp in scored[:limit]]

    def _search_index(self):
        """Return cached (experiences, haystacks, df_table). The table
        maps every token observed in the log to its document
        frequency; queries grab their token's df from here instead of
        rescanning the file. Cache key is the file's (mtime, size) so
        a fresh append on disk auto-invalidates the cache."""
        if not self.path.exists():
            return [], [], {}
        try:
            st = self.path.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            return [], [], {}
        cached = self._search_cache.get(key)
        if cached is not None:
            return cached
        experiences = list(self.iter_experiences())
        haystacks: List[set] = []
        df: Dict[str, int] = {}
        for exp in experiences:
            # The gist is in the haystack so recall reaches a memory by
            # what was ANSWERED, not only by how the question was phrased.
            hs = (exp.summary + " " + exp.user_first_words + " "
                  + (exp.cluster or "") + " "
                  + (getattr(exp, "answer_gist", "") or "")).lower()
            # Store the TOKEN SET (not the raw string). Scoring tests
            # membership token-wise to match how df is built — a query
            # token must EQUAL a document token, not merely appear as a
            # substring of one. The old `t in hs` substring test ranked
            # "art" into "smart"/"part" and, worse, gave an unseen query
            # token max idf whenever it happened to be a substring.
            toks = {t for t in _tokenize(hs) if len(t) > 2}
            haystacks.append(toks)
            # Every distinct token contributes +1 to its df.
            for t in toks:
                df[t] = df.get(t, 0) + 1
        # Bound the cache so a long-running agent doesn't hold every
        # historical (mtime, size) snapshot — only the latest is useful.
        self._search_cache = {key: (experiences, haystacks, df)}
        return experiences, haystacks, df

    def update_outcome(
        self, trajectory_id: str, outcome: str, *, failure_reason: str = "",
    ) -> bool:
        """Backfill the verdict of an already-written experience.

        The capture path runs on the hot turn loop, before the verifier
        / reflection layers have decided whether the turn actually
        succeeded — so most records are first written ``outcome="unknown"``.
        This method rewrites the matching entry once that verdict exists,
        so the agent's memory of its own past stops being verdict-blind.

        Updates the most recent entry sharing ``trajectory_id``. Returns
        True when an entry was changed. Never raises — backfill is
        secondary to the user turn."""
        if not self.enabled or not trajectory_id or not self.path.exists():
            return False
        new_outcome = (outcome or "").strip().lower()
        if new_outcome not in ("passed", "failed"):
            return False
        try:
            with self._lock:
                lines = self.path.read_text(encoding="utf-8").splitlines()
                for i in range(len(lines) - 1, -1, -1):
                    s = lines[i].strip()
                    if not s:
                        continue
                    try:
                        d = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if d.get("trajectory_id") != trajectory_id:
                        continue
                    if (d.get("outcome") or "").strip().lower() == new_outcome:
                        return False  # already correct — nothing to do
                    d["outcome"] = new_outcome
                    # Patch the prose verdict clause too, so recall and the
                    # narrative read a coherent summary — whichever clause
                    # the record already carries, not just the "unknown" one
                    # (see `_swap_verdict_clause`: an upgrade off a captured
                    # FAILED used to leave the prose contradicting the field).
                    summary = d.get("summary") or ""
                    patched = _swap_verdict_clause(
                        summary, _outcome_phrase(new_outcome, failure_reason))
                    if patched != summary:
                        d["summary"] = patched
                    lines[i] = json.dumps(d, ensure_ascii=False)
                    tmp = self.path.with_suffix(".jsonl.tmp")
                    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    tmp.replace(self.path)
                    return True
                return False
        except Exception as e:
            logger.warning("autobiographical update_outcome failed: %s", e)
            return False

    def cluster_counts(self) -> Dict[str, int]:
        """How many experiences fall into each derived cluster. Lets the
        narrative / stats layer report self-patterns ("I keep doing X").

        Served from the (mtime, size)-cached search index instead of a
        per-call full parse — ``stats()`` calls this on the introspect
        summary path, which previously cost a whole-log scan per call."""
        counts: Dict[str, int] = {}
        try:
            experiences, _, _ = self._search_index()
        except Exception:  # noqa: BLE001 — stats must never raise
            return counts
        for exp in experiences:
            # Session-boot markers are bookkeeping, not experiences. On the
            # live store they were 445 of 2001 entries (22%), and 84% of
            # the top cluster — the agent's #1 reported "topic" about
            # itself was restart noise rendered as introspection. They
            # stay in the store (real chronology, `recent` may show them);
            # they just don't count as things the agent DID.
            if (exp.outcome or "").strip().lower() == "boot":
                continue
            c = (exp.cluster or "").strip()
            if c:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def boot_count(self) -> int:
        """Session-boot markers on file — reported SEPARATELY from
        experiences (see `cluster_counts` for why they are excluded)."""
        try:
            experiences, _, _ = self._search_index()
        except Exception:  # noqa: BLE001
            return 0
        return sum(1 for e in experiences
                   if (e.outcome or "").strip().lower() == "boot")

    # ------------------------------------------------------------------
    # Reference-count tracker (#13)
    # ------------------------------------------------------------------

    def _load_ref_counts(self) -> Dict[str, int]:
        if self._ref_counts is not None:
            return self._ref_counts
        if not self.refcount_path.exists():
            self._ref_counts = {}
            return self._ref_counts
        try:
            d = json.loads(self.refcount_path.read_text(encoding="utf-8"))
            self._ref_counts = {str(k): int(v) for k, v in d.items()
                                if isinstance(v, (int, float))}
        except Exception:
            self._ref_counts = {}
        return self._ref_counts

    def _persist_ref_counts(self) -> None:
        if self._ref_counts is None:
            return
        try:
            self.refcount_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.refcount_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._ref_counts, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.refcount_path)
        except Exception as e:
            logger.warning("ref-count persist failed: %s", e)

    def record_reference(self, experience_id: str) -> int:
        """Bump the reference counter for ``experience_id``. Returns
        the new count (0 when disabled / empty id). Persistence is
        write-through so the count survives a process restart even if
        no further turns happen."""
        if not self.enabled or not experience_id:
            return 0
        # Hold the lock across load → bump → persist so concurrent callers
        # (post-turn hot path + biological watchdog) can't lose an
        # increment, and so _persist_ref_counts never json.dumps a dict
        # another thread is mutating ("dict changed size during iteration").
        with self._lock:
            counts = self._load_ref_counts()
            counts[experience_id] = counts.get(experience_id, 0) + 1
            new_count = counts[experience_id]
            self._persist_ref_counts()
        return new_count

    def reference_count(self, experience_id: str) -> int:
        if not self.enabled or not experience_id:
            return 0
        with self._lock:
            return self._load_ref_counts().get(experience_id, 0)

    def ref_count_summary(self) -> Dict[str, int]:
        """Snapshot of the full reference-count table. Used by stats()
        and by the falsification probes to see which memories the
        agent has actually reached for."""
        if not self.enabled:
            return {}
        with self._lock:
            return dict(self._load_ref_counts())

    def count(self) -> int:
        """Number of parseable experiences on file — served from the
        cached search index (was a per-call full line scan). Corrupt
        lines no longer count; every caller reports this number as
        "experiences", which is what the index holds."""
        try:
            experiences, _, _ = self._search_index()
        except Exception:  # noqa: BLE001
            return 0
        return sum(1 for e in experiences
                   if (e.outcome or "").strip().lower() != "boot")


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "you", "your",
    "into", "have", "been", "are", "was", "were", "but", "not", "any",
    "all", "out", "its", "they", "them", "their", "what", "when", "where",
    "who", "how", "why", "did", "does", "doing", "i", "me", "my",
    "of", "to", "a", "an", "in", "is", "on", "it",
})


def _trigrams(text: str):
    toks = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2]
    return [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]


def detect_referenced_experiences(
    *,
    prefix_text: str,
    response_text: str,
    experiences,
) -> List[str]:
    """Return ids of experiences whose summary shares a non-trivial
    trigram with ``response_text``. Trigrams are computed after
    stopword filtering so generic three-word sequences ("I worked on")
    don't trigger false matches.

    Pure function — no state mutation. The caller decides whether to
    feed the result to ``AutobiographicalMemory.record_reference``."""
    if not response_text or not experiences:
        return []
    response_trigrams = set(_trigrams(response_text))
    if not response_trigrams:
        return []
    # Only credit experiences that were actually SHOWN in the wake-up prefix
    # (the documented contract: "which experiences FROM THE PREFIX were
    # echoed"). Without this gate the function ignored `prefix_text` entirely
    # and bumped reference_count for any recent experience sharing a trigram
    # with the response — e.g. one the model never saw but that overlaps the
    # echoed-back user request — polluting the recall relevance prior.
    prefix_trigrams = set(_trigrams(prefix_text or ""))
    matched: List[str] = []
    for exp in experiences:
        exp_text = (getattr(exp, "summary", "") or "") + " " + (
            getattr(exp, "user_first_words", "") or "")
        exp_trigrams = list(_trigrams(exp_text))
        if not any(tg in prefix_trigrams for tg in exp_trigrams):
            continue  # not in the prefix → not a real reference
        if any(tg in response_trigrams for tg in exp_trigrams):
            matched.append(exp.id)
    return matched


def summarise_turn_first_person(
    *,
    user_request: str,
    tool_names: Iterable[str],
    outcome: str,
    final_response: str,
    failure_reason: str = "",
) -> str:
    """Template-based first-person summary of a turn.

    Cheap, deterministic, zero-LLM. Produces a one-sentence experience
    record from the raw turn data. The narrative phase later
    rewrites a richer summary across many of these.

    Kept template-based (not LLM) so the autobiographical write is on
    the hot path of every turn without paying a round-trip. The
    narrative consolidation phase is where LLM polish happens."""
    ur = (user_request or "").strip().replace("\n", " ")
    ur_short = ur[:120] + ("…" if len(ur) > 120 else "")
    tools = [t for t in tool_names if t]
    if tools:
        if len(tools) == 1:
            tool_phrase = f"reached for {tools[0]}"
        elif len(tools) == 2:
            tool_phrase = f"used {tools[0]} and {tools[1]}"
        else:
            tool_phrase = f"strung together {', '.join(tools[:-1])} and {tools[-1]}"
    else:
        tool_phrase = "reasoned through it without tools"

    outcome_phrase = _outcome_phrase(outcome, failure_reason)

    if ur_short:
        return f'I worked on "{ur_short}". I {tool_phrase} {outcome_phrase}.'
    return f"I {tool_phrase} {outcome_phrase}."
