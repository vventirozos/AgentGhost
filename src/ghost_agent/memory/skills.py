import json
import logging
import math
import re
import threading
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


# ---------------------------------------------------------------------------
# Structured lesson schema (additive; backward-compatible with legacy
# `{task, mistake, solution}` entries already on disk).
#
# A lesson now carries a structured trigger + correct_pattern (ideally a
# runnable code snippet), domains for filtering retrieval, and utility
# counters that let the retrieval feedback loop prune low-value lessons.
# Legacy fields (task/mistake/solution/frequency/graduated/timestamp) are
# still written so older readers keep working.
# ---------------------------------------------------------------------------

LESSON_SCHEMA_VERSION = 2

# Default retrieval distance threshold. Tightened from the old 0.65 —
# that threshold pulled in lessons with only tangential keyword overlap
# and polluted the SKILL PLAYBOOK injection. 0.45 still captures true
# semantic neighbours while dropping weak-overlap noise.
DEFAULT_RETRIEVAL_DISTANCE = 0.45

# Hard cap on playbook size. When a new lesson is written and the cap
# is exceeded, we drop by utility (verified pinned, lowest-utility
# unverified first) rather than plain FIFO — the old FIFO rule could
# evict a verified high-utility lesson purely because it was old.
PLAYBOOK_MAX = 50

# Fields that signal a lesson is using the new schema. If any of these
# keys is present we treat the entry as structured; absence means the
# entry is a legacy record and we fall back to rendering from
# task/mistake/solution.
_STRUCTURED_KEYS = ("trigger", "correct_pattern", "domains", "confidence")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _extract_code_block(text: str) -> str:
    """Pull the first fenced code block (```lang\n...\n```) out of `text`.

    Used to normalise the `correct_pattern` field: the extraction LLM may
    return the pattern wrapped in fences or as raw prose. We keep code
    examples separately so retrieval can show them verbatim.
    """
    if not text:
        return ""
    m = re.search(r"```(?:\w+)?\n?(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Heuristic: a pattern that starts with `def ` / `import ` / `for `
    # and spans multiple lines is probably code.
    stripped = text.strip()
    first = stripped.split("\n", 1)[0]
    code_starts = ("def ", "import ", "from ", "with ", "for ", "while ", "class ")
    if "\n" in stripped and any(first.startswith(c) for c in code_starts):
        return stripped
    return ""


def _ensure_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if isinstance(x, (str, int, float))]
    if isinstance(v, str):
        parts = re.split(r"[,\s]+", v.strip())
        return [p for p in parts if p]
    return []


# Tokens that carry no discriminative signal for trigger matching. Dropping
# them lets paraphrased triggers ("How do I parse JSON?" vs "parse JSON")
# collapse to the same normalized key so frequency actually accumulates.
_TRIGGER_STOPWORDS = frozenset({
    "a", "an", "the", "to", "of", "for", "and", "or", "in", "on", "at",
    "is", "are", "be", "do", "does", "did", "how", "what", "why", "when",
    "i", "my", "me", "we", "you", "it", "its", "with", "that", "this",
    "can", "should", "would", "could", "please", "need", "want", "get",
})


def _normalize_trigger(s: str, *, sort_tokens: bool = True) -> str:
    """Normalize a trigger string into a stable match key.

    Lowercase, strip punctuation, collapse whitespace, drop stopwords, and
    (by default) sort the remaining significant tokens so that semantically
    equivalent / paraphrased triggers map to the SAME key. This is what makes
    the frequency counter actually climb: exact-string matching almost never
    collided on natural-language triggers, so nothing ever graduated.

    `sort_tokens=False` preserves word order — used when we want the token
    SET (for overlap relevance) rather than a canonical ordered key.

    Backward-compatible: storage is untouched; this only affects how the
    match KEY is computed.
    """
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)          # punctuation -> space
    tokens = [t for t in s.split() if t]    # collapse whitespace
    significant = [t for t in tokens if t not in _TRIGGER_STOPWORDS]
    if not significant:
        # All-stopword trigger (rare) — fall back to the raw tokens so we
        # don't collapse every such trigger into the empty key.
        significant = tokens
    if sort_tokens:
        significant = sorted(significant)
    return " ".join(significant)


def _trigger_token_set(s: str) -> set:
    """Set of significant (stopword-stripped) tokens for overlap scoring."""
    return set(_normalize_trigger(s, sort_tokens=False).split())


def _lesson_is_structured(lesson: dict) -> bool:
    return any(k in lesson for k in _STRUCTURED_KEYS)


def _normalize_lesson(lesson: dict) -> dict:
    """Fill in missing schema fields with defaults without dropping
    existing ones. Safe to call on both legacy and new-style lessons."""
    if not isinstance(lesson, dict):
        return {}
    out = dict(lesson)
    out.setdefault("timestamp", _now_iso())
    out.setdefault("task", out.get("trigger") or "")
    out.setdefault("mistake", out.get("anti_pattern") or "")
    out.setdefault("solution", out.get("correct_pattern") or "")
    out.setdefault("frequency", 1)
    out.setdefault("graduated", False)

    # New fields
    out.setdefault("schema_version", LESSON_SCHEMA_VERSION)
    out.setdefault("trigger", out.get("task", ""))
    out.setdefault("anti_pattern", out.get("mistake", ""))
    out.setdefault("correct_pattern", out.get("solution", ""))
    out.setdefault("code_example", _extract_code_block(out.get("correct_pattern", "")))
    out["domains"] = _ensure_list(out.get("domains"))
    try:
        out["confidence"] = float(out.get("confidence", 0.5))
    except Exception:
        out["confidence"] = 0.5
    out.setdefault("source_challenge_hash", "")
    out.setdefault("verified", False)
    out.setdefault("verification_attempted", False)
    out["retrievals"] = int(out.get("retrievals") or 0)
    out["helpful_retrievals"] = int(out.get("helpful_retrievals") or 0)
    out.setdefault("last_retrieved_at", "")
    out.setdefault("source", "")
    # Trajectory id of the turn that produced this lesson. Used by
    # `retract_lessons_from_trajectory` to scrub poisoned lessons
    # when the source trajectory is later promoted to FAILED (e.g.
    # the user-correction path catches an opt-prot lesson that was
    # written before the user could push back). Empty string for
    # legacy lessons / lessons with no clear single source.
    out.setdefault("source_trajectory_id", "")
    return out


def build_lesson(
    *,
    task: str = "",
    trigger: str = "",
    anti_pattern: str = "",
    correct_pattern: str = "",
    domains=None,
    confidence: float = 0.5,
    source_challenge_hash: str = "",
    verified: bool = False,
    source: str = "",
    source_trajectory_id: str = "",
) -> dict:
    """Construct a canonical structured lesson. Callers that only have
    legacy `task/mistake/solution` can pass those as trigger/
    anti_pattern/correct_pattern — the normalizer fills the rest.

    `source` is free-form provenance metadata (e.g. ``self_play``,
    ``post_mortem``). It is deliberately NOT mixed into the trigger
    string — the trigger must stay close to the user's query language
    so semantic retrieval and BM25 re-ranking keep working. Provenance
    is surfaced at render time from this separate field.

    ``source_trajectory_id`` is the trajectory whose turn produced
    this lesson. Production writers (Perfection-Protocol,
    reflection sink) thread the current turn's trajectory id
    through here so the user-correction path can retract poisoned
    lessons — see ``SkillMemory.retract_lessons_from_trajectory``.
    """
    lesson = {
        "timestamp": _now_iso(),
        "task": task or trigger or "",
        "mistake": anti_pattern or "",
        "solution": correct_pattern or "",
        "frequency": 1,
        "graduated": False,
        "schema_version": LESSON_SCHEMA_VERSION,
        "trigger": trigger or task or "",
        "anti_pattern": anti_pattern or "",
        "correct_pattern": correct_pattern or "",
        "code_example": _extract_code_block(correct_pattern or ""),
        "domains": _ensure_list(domains),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "source_challenge_hash": source_challenge_hash or "",
        "verified": bool(verified),
        "verification_attempted": False,
        "retrievals": 0,
        "helpful_retrievals": 0,
        "last_retrieved_at": "",
        "source": source or "",
        "source_trajectory_id": source_trajectory_id or "",
    }
    return lesson


def render_lesson_for_prompt(lesson: dict) -> str:
    """Single-lesson rendering used by `get_playbook_context`. Structured
    lessons get the new trigger/pattern/fix layout; legacy lessons fall
    back to the SITUATION/MISTAKE/FIX prose format."""
    if not isinstance(lesson, dict):
        return ""
    if _lesson_is_structured(lesson):
        trig = lesson.get("trigger") or lesson.get("task") or ""
        anti = lesson.get("anti_pattern") or lesson.get("mistake") or ""
        fix = lesson.get("correct_pattern") or lesson.get("solution") or ""
        code = lesson.get("code_example") or ""
        domains = lesson.get("domains") or []
        verified = "✓" if lesson.get("verified") else "·"
        parts = [f"TRIGGER ({verified}): {trig}"]
        if domains:
            parts.append(f"DOMAINS: {', '.join(domains)}")
        if anti:
            parts.append(f"ANTI-PATTERN: {anti}")
        if fix:
            parts.append(f"CORRECT-PATTERN: {fix}")
        if code and code not in fix:
            parts.append(f"CODE:\n```python\n{code}\n```")
        return "\n   ".join(parts)
    # Legacy path
    task = lesson.get("task", "") or ""
    mistake = lesson.get("mistake", "") or ""
    solution = lesson.get("solution", "") or ""
    return (
        f"SITUATION: {task}\n"
        f"   PREVIOUS MISTAKE: {mistake}\n"
        f"   THE FIX: {solution}"
    )


def lesson_embedding_text(lesson: dict) -> str:
    """Canonical text used when embedding a lesson into the vector
    store. Kept stable across the legacy and new schemas so dedup and
    retrieval keep working on older entries."""
    task = lesson.get("trigger") or lesson.get("task") or ""
    mistake = lesson.get("anti_pattern") or lesson.get("mistake") or ""
    solution = lesson.get("correct_pattern") or lesson.get("solution") or ""
    return f"SITUATION: {task}\nMISTAKE: {mistake}\nSOLUTION: {solution}"


def _bm25_like_score(query: str, trigger: str) -> float:
    """Tiny keyword-overlap re-ranker. Returns a score in [0, 1] — the
    fraction of query tokens that appear in the lesson trigger.

    Not true BM25 (no IDF corpus), but enough to break ties when two
    lessons have similar vector distance. The retrieval pipeline uses
    this to prefer lessons whose trigger literally mentions words from
    the user's current task.
    """
    if not query or not trigger:
        return 0.0
    q_tokens = {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", query.lower()) if len(t) > 2}
    t_tokens = {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", trigger.lower()) if len(t) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def compute_lesson_utility(lesson: dict) -> float:
    """Utility score used for ranking + prune decisions.

    Combines:
      - helpful_retrievals / retrievals  (empirical utility)
      - confidence                        (initial trust)
      - verified                          (verification-grounded boost)
      - frequency                         (diminishing log)
      - age penalty for unused lessons    (pre-empts stale entries)

    Score is in [0, ~2]. Unused lessons with 0 retrievals get a neutral
    score anchored by confidence (so freshly-written but unverified
    lessons aren't pruned before they have had a chance to be useful).
    """
    r = int(lesson.get("retrievals") or 0)
    h = int(lesson.get("helpful_retrievals") or 0)
    conf = float(lesson.get("confidence") or 0.5)
    verified = bool(lesson.get("verified"))
    freq = int(lesson.get("frequency") or 1)
    # Empirical hit-rate — smoothed by a +1/+2 prior so a single
    # retrieval doesn't jump to 100% or crash to 0%.
    hit_rate = (h + 1) / (r + 2)
    score = conf * 0.5 + hit_rate * 0.8 + (0.3 if verified else 0.0)
    score += min(math.log1p(freq) * 0.1, 0.3)
    # Stale penalty: if a lesson has been retrieved many times but never
    # helped, demote it harder than a never-retrieved one.
    if r >= 5 and hit_rate < 0.35:
        score *= 0.5
    return round(score, 4)


def _trim_playbook_by_utility(playbook: list, max_entries: int) -> list:
    """Cap `playbook` to `max_entries` entries while preserving the
    highest-utility lessons.

    Rules:
      * The head of `playbook` (index 0) is always kept. `learn_lesson`
        prepends the freshly-written lesson, and new lessons have not
        yet had a chance to be retrieved — their utility score is
        intentionally low, but evicting them immediately would defeat
        the purpose of writing them in the first place.
      * Verified lessons are pinned — they cleared the `_verify_lesson_
        helpful` gate in `dream.py`, which is a much stronger signal
        than retrieval statistics.
      * The remaining slots are filled with the highest-utility
        unverified lessons. Utility is computed via
        `compute_lesson_utility`, which already blends confidence,
        hit-rate, frequency and verification status.
      * If verified lessons alone exceed the cap (ignoring the head),
        drop the lowest-utility verified ones so the total fits.
    """
    if max_entries <= 0 or not playbook:
        return []
    if len(playbook) <= max_entries:
        return list(playbook)

    head = playbook[0]
    rest = list(playbook[1:])
    verified = [p for p in rest if _normalize_lesson(p).get("verified")]
    unverified = [p for p in rest if not _normalize_lesson(p).get("verified")]
    unverified.sort(
        key=lambda p: compute_lesson_utility(_normalize_lesson(p)),
        reverse=True,
    )

    kept = [head] + verified
    slots_left = max_entries - len(kept)
    if slots_left > 0:
        kept.extend(unverified[:slots_left])

    if len(kept) > max_entries:
        # Overflow from too many verified lessons — keep the head and
        # then the highest-utility rest (verified first by tie-break
        # via `compute_lesson_utility`'s +0.3 verified bonus).
        head_keep = kept[0]
        tail = sorted(
            kept[1:],
            key=lambda p: compute_lesson_utility(_normalize_lesson(p)),
            reverse=True,
        )
        kept = [head_keep] + tail[: max_entries - 1]

    return kept


def _delete_lesson_twin(memory_system, lesson) -> None:
    """Best-effort delete of a lesson's embedded vector twin.

    When a lesson is trimmed/pruned/removed from the JSON playbook, its
    ChromaDB twin was previously orphaned — accumulating stale docs that
    `get_playbook_context` would still surface. Keyed by the exact metadata
    written at add() time (skills.py learn_lesson): trigger[:200]+type (the
    precise per-lesson key), falling back to source_trajectory_id. Never
    raises — the JSON playbook is canonical; the vector scrub is advisory.
    """
    if memory_system is None or not isinstance(lesson, dict):
        return
    try:
        coll = getattr(memory_system, "collection", None)
        if coll is None or not hasattr(coll, "delete"):
            return
        trig = (lesson.get("trigger") or lesson.get("task") or "")[:200]
        if trig:
            # Chroma requires multi-key filters to be wrapped in $and; a flat
            # two-key dict raises ValueError (silently caught below).
            coll.delete(where={"$and": [{"type": "skill"}, {"trigger": trig}]})
            return
        src = lesson.get("source_trajectory_id")
        if isinstance(src, str) and src:
            coll.delete(where={"source_trajectory_id": src})
    except Exception as e:
        logger.warning("skill twin delete failed: %s", e)


class SkillMemory:
    def __init__(self, memory_dir: Path):
        self.file_path = memory_dir / "skills_playbook.json"
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self.save_playbook([])

    def _get_lock(self):
        """Lazy-init wrapper for tests that bypass __init__ via monkeypatch."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock
        return lock

    def _save_playbook_unlocked(self, playbook):
        """Atomic write — caller MUST already hold self._get_lock()."""
        temp_path = self.file_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(playbook, indent=2))
        os.replace(temp_path, self.file_path)

    def save_playbook(self, playbook):
        # Helper for atomic save
        with self._get_lock():
            self._save_playbook_unlocked(playbook)

    def _load_playbook(self) -> list:
        """Load the JSON playbook, with safe handling for missing /
        corrupt files.

        - File doesn't exist: return ``[]`` (first run).
        - File is empty: return ``[]`` (first run, half-initialised).
        - File contains a non-list JSON value: return ``[]`` (defensive).
        - File is corrupt JSON: rename to
          ``skills_playbook.json.corrupt-<ts>`` and return ``[]``. The
          rename preserves the bad bytes for human recovery; returning
          ``[]`` lets the caller proceed with a fresh playbook on the
          next save. Without the rename, the next ``learn_lesson`` call
          would silently overwrite the corrupt file with a single-entry
          playbook — every prior lesson lost without a trace.
        - OSError (disk full, permission, etc.): re-raise. Callers
          treat the playbook as canonical; pretending it is empty when
          the disk is sick lies to the next ``learn_lesson`` call,
          which then "atomically saves" a 1-entry playbook on top of
          a perfectly-readable file the OS just refused to read for
          us — same data loss as the corrupt-file case but harder to
          spot.
        """
        try:
            content = self.file_path.read_text()
        except FileNotFoundError:
            return []
        except OSError:
            raise
        # Tolerate a non-string return from `read_text` — pre-existing
        # tests mock `file_path` with a `MagicMock` and exercise the
        # "no playbook on disk" branch by relying on the empty fallback.
        # Returning `[]` for non-str content keeps that contract.
        if not isinstance(content, str):
            return []
        if not content:
            return []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Preserve the corrupt bytes under a timestamped sidecar
            # so a human can recover, then fall through to empty.
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            backup = self.file_path.with_suffix(
                self.file_path.suffix + f".corrupt-{ts}"
            )
            try:
                os.replace(str(self.file_path), str(backup))
                logger.warning(
                    "skills_playbook.json was corrupt; renamed to %s",
                    backup,
                )
            except OSError as rename_err:
                # If we can't even rename, log loudly — but still
                # return [] so the agent doesn't crash. The next
                # save will overwrite the corrupt file (same as
                # the old behaviour, but at least we logged).
                logger.error(
                    "skills_playbook.json corrupt AND rename failed: %s",
                    rename_err,
                )
            return []
        return data if isinstance(data, list) else []

    def _find_duplicate_lesson(self, task: str, mistake: str, solution: str, memory_system=None) -> dict:
        """Check if a semantically similar lesson already exists.

        Returns the matching lesson dict (with 'index' key) if found, else None.
        Uses vector search first (fast, semantic), then falls back to JSON
        playbook exact-task matching if no vector store is available."""
        lesson_text = f"SITUATION: {task}\nMISTAKE: {mistake}\nSOLUTION: {solution}"

        if memory_system and hasattr(memory_system, 'collection') and memory_system.collection:
            try:
                results = memory_system.collection.query(
                    query_texts=[lesson_text],
                    n_results=3,
                    where={"type": "skill"}
                )
                if results['distances'] and results['distances'][0]:
                    for i, dist in enumerate(results['distances'][0]):
                        if dist < 0.15:
                            return {
                                "id": results['ids'][0][i],
                                "text": results['documents'][0][i],
                                "distance": dist,
                                "source": "vector",
                            }
            except Exception as e:
                logger.debug(f"Vector dedup check failed (non-critical): {e}")

        with self._get_lock():
            playbook = self._load_playbook()
        # Normalized (fuzzy) match key instead of exact lowercased string —
        # paraphrased triggers ("parse JSON" / "How do I parse JSON?") now
        # collide so the frequency counter accumulates and graduation can fire.
        task_key = _normalize_trigger(task or "")
        for idx, p in enumerate(playbook):
            existing_key = _normalize_trigger(p.get("task") or p.get("trigger") or "")
            if existing_key and existing_key == task_key:
                return {"index": idx, "lesson": p, "source": "json"}
        return None

    def learn_lesson(
        self,
        task: str,
        mistake: str,
        solution: str,
        memory_system=None,
        *,
        trigger: str = "",
        anti_pattern: str = "",
        correct_pattern: str = "",
        domains=None,
        confidence: float = 0.5,
        source_challenge_hash: str = "",
        verified: bool = False,
        source: str = "",
        source_trajectory_id: str = "",
    ):
        """Write a lesson to the playbook. Accepts both legacy positional
        args (task/mistake/solution) and the new structured kwargs.

        When structured kwargs are provided they take precedence. The
        canonical on-disk entry contains BOTH representations so older
        readers (pattern-matching on `task/mistake/solution`) keep
        working unchanged.
        """
        try:
            effective_trigger = trigger or task or ""
            effective_anti = anti_pattern or mistake or ""
            effective_correct = correct_pattern or solution or ""

            # Dedup uses legacy fields for compatibility with older
            # playbooks & vector entries written before the redesign.
            duplicate = self._find_duplicate_lesson(
                effective_trigger, effective_anti, effective_correct, memory_system
            )
            if duplicate:
                if duplicate.get("source") == "json":
                    with self._get_lock():
                        playbook = self._load_playbook()
                        # Re-locate the duplicate by key under the lock — the
                        # index from _find_duplicate_lesson came from an older
                        # snapshot, and a concurrent learn_lesson prepends
                        # entries, shifting every index. Trusting the stale
                        # index could merge into an unrelated lesson.
                        key = _normalize_trigger(effective_trigger or "")
                        idx = next(
                            (i for i, p in enumerate(playbook)
                             if _normalize_trigger(p.get("task") or p.get("trigger") or "") == key),
                            None,
                        )
                        if idx is not None:
                            existing = _normalize_lesson(playbook[idx])
                            existing["frequency"] = int(existing.get("frequency") or 1) + 1
                            # Prefer the new solution if it's richer or
                            # if the caller marked it verified.
                            if len(effective_correct) > len(existing.get("solution") or ""):
                                existing["solution"] = effective_correct
                                existing["correct_pattern"] = effective_correct
                                existing["code_example"] = _extract_code_block(effective_correct)
                            if domains:
                                merged = sorted(set(existing.get("domains", [])) | set(_ensure_list(domains)))
                                existing["domains"] = merged
                            if verified and not existing.get("verified"):
                                existing["verified"] = True
                                existing["confidence"] = max(
                                    float(existing.get("confidence") or 0.5),
                                    min(1.0, float(confidence or 0.5) + 0.2),
                                )
                            existing["timestamp"] = _now_iso()
                            playbook[idx] = existing
                            self._save_playbook_unlocked(playbook)
                            pretty_log(
                                "SKILL REINFORCED",
                                f"Merged duplicate lesson: {effective_trigger[:30]}... (freq={existing['frequency']})",
                                icon=Icons.MEM_REINFORCE,
                            )
                            return
                else:
                    # Vector-dedup path. Previously this just returned, so a
                    # near-exact re-learn (which ALWAYS hits the vector store
                    # first) never bumped `frequency` — and graduation needs
                    # frequency>=5, so lessons never graduated. Bump the
                    # matching playbook entry (and upgrade verified/confidence)
                    # just like the JSON-dedup branch does.
                    with self._get_lock():
                        playbook = self._load_playbook()
                        key = _normalize_trigger(effective_trigger or "")
                        idx = next(
                            (i for i, p in enumerate(playbook)
                             if _normalize_trigger(p.get("task") or p.get("trigger") or "") == key),
                            None,
                        )
                        if idx is not None:
                            existing = _normalize_lesson(playbook[idx])
                            existing["frequency"] = int(existing.get("frequency") or 1) + 1
                            if len(effective_correct) > len(existing.get("solution") or ""):
                                existing["solution"] = effective_correct
                                existing["correct_pattern"] = effective_correct
                                existing["code_example"] = _extract_code_block(effective_correct)
                            if verified and not existing.get("verified"):
                                existing["verified"] = True
                                existing["confidence"] = max(
                                    float(existing.get("confidence") or 0.5),
                                    min(1.0, float(confidence or 0.5) + 0.2),
                                )
                            existing["timestamp"] = _now_iso()
                            playbook[idx] = existing
                            self._save_playbook_unlocked(playbook)
                            pretty_log(
                                "SKILL REINFORCED",
                                f"Vector-dedup: bumped lesson freq={existing['frequency']}: {effective_trigger[:30]}...",
                                icon=Icons.MEM_REINFORCE,
                            )
                        else:
                            pretty_log(
                                "SKILL DEDUP",
                                f"Skipped near-duplicate (dist={duplicate.get('distance', 0):.3f}, "
                                f"no JSON twin to bump): {effective_trigger[:30]}...",
                                icon=Icons.SKIP,
                            )
                    return

            new_lesson = build_lesson(
                task=task,
                trigger=effective_trigger,
                anti_pattern=effective_anti,
                correct_pattern=effective_correct,
                domains=domains,
                confidence=confidence,
                source_challenge_hash=source_challenge_hash,
                verified=verified,
                source=source,
                source_trajectory_id=source_trajectory_id,
            )

            with self._get_lock():
                before = [new_lesson] + self._load_playbook()
                playbook = _trim_playbook_by_utility(before, PLAYBOOK_MAX)
                self._save_playbook_unlocked(playbook)

            # Delete the vector twins of any lessons the trim dropped, so the
            # capped JSON playbook and the vector store don't drift apart
            # (orphans waste retrieval slots). Identity-keyed against `before`.
            if memory_system:
                kept_ids = {id(l) for l in playbook}
                for dropped in before:
                    if id(dropped) not in kept_ids and dropped is not new_lesson:
                        _delete_lesson_twin(memory_system, dropped)

            if memory_system:
                text = lesson_embedding_text(new_lesson)
                meta = {
                    "type": "skill",
                    "timestamp": new_lesson["timestamp"],
                    "trigger": new_lesson.get("trigger", "")[:200],
                    "domains": ",".join(new_lesson.get("domains", []))[:200],
                    "verified": bool(new_lesson.get("verified")),
                    # Persist provenance on the vector copy so
                    # `retract_lessons_from_trajectory` can scrub
                    # both stores at once via collection.delete with
                    # a `where` filter on this metadata key.
                    "source_trajectory_id": new_lesson.get("source_trajectory_id", "") or "",
                }
                memory_system.add(text, meta)

            pretty_log(
                "SKILL ACQUIRED",
                f"Lesson learned: {effective_trigger[:30]}...",
                icon=Icons.SKILL_GRADUATE,
            )
        except Exception as e:
            logger.error(f"Failed to save skill: {e}")

    # ---- Retrieval + feedback -----------------------------------------

    def _update_lesson_fields(self, match_predicate, mutator):
        """Atomically update the first lesson matching `match_predicate`.

        The mutator receives the normalized lesson and should mutate
        it in place. Returns True if a lesson was updated, False otherwise.
        """
        with self._get_lock():
            playbook = self._load_playbook()
            for idx, raw in enumerate(playbook):
                if match_predicate(raw):
                    lesson = _normalize_lesson(raw)
                    mutator(lesson)
                    playbook[idx] = lesson
                    self._save_playbook_unlocked(playbook)
                    return True
            return False

    def retract_lessons_from_trajectory(
        self,
        trajectory_id: str,
        memory_system=None,
    ) -> int:
        """Remove every lesson whose ``source_trajectory_id`` matches.

        Called by the user-correction promotion path: when a turn is
        promoted to FAILED via the user's next message, any lesson
        the just-finished turn produced (typically via the
        Perfection-Protocol's eager ``learn_lesson`` write) was
        sourced from a now-discredited turn and must be scrubbed
        before retrieval can surface it on a future user query.

        Scrubs both:
          1. The JSON playbook (atomic write under the lock).
          2. The vector store (when ``memory_system`` is wired) via
             ``collection.delete(where={"source_trajectory_id": ...})``.
             ChromaDB ignores unknown metadata keys gracefully — a
             store missing this key is a no-op delete.

        Returns the number of lessons removed from the JSON playbook.
        Idempotent: a second call with the same id returns 0.
        Empty / non-string ids return 0 without touching disk
        (defensive against legacy lessons whose
        ``source_trajectory_id`` was never set).
        """
        if not isinstance(trajectory_id, str) or not trajectory_id:
            return 0
        removed = 0
        json_failed = False
        try:
            with self._get_lock():
                playbook = self._load_playbook()
                kept = []
                for entry in playbook:
                    src = (
                        entry.get("source_trajectory_id")
                        if isinstance(entry, dict) else None
                    )
                    if isinstance(src, str) and src == trajectory_id:
                        removed += 1
                        continue
                    kept.append(entry)
                if removed:
                    self._save_playbook_unlocked(kept)
        except Exception as e:
            logger.warning(
                "retract_lessons_from_trajectory JSON pass failed: %s", e
            )
            json_failed = True

        # Run the vector scrub whenever the JSON pass SUCCEEDED, even
        # when ``removed == 0``. Previously gated on ``removed > 0``,
        # which meant a stale vector entry whose JSON twin was already
        # gone (drift from a prior partial state) would never get
        # cleaned up by a retraction call. The drift entry costs a
        # retrieval slot until the next REM prune.
        #
        # When JSON pass FAILED we deliberately DO NOT run the vector
        # scrub: the on-disk JSON is unchanged (`_save_playbook_unlocked`
        # is atomic via os.replace, so a save raise leaves the original
        # file intact), so JSON still has the lesson. Scrubbing only
        # the vector side would create the opposite drift.
        if memory_system is not None and not json_failed:
            try:
                coll = getattr(memory_system, "collection", None)
                if coll is not None and hasattr(coll, "delete"):
                    coll.delete(where={"source_trajectory_id": trajectory_id})
            except Exception as e:
                # Vector scrub is best-effort — the JSON playbook is
                # canonical, and a stale vector entry whose JSON twin
                # is gone will be filtered out by the playbook-snapshot
                # lookup in get_playbook_context (vector docs without a
                # JSON match render the raw embedded doc, which still
                # costs a slot but is recoverable on a later rebuild).
                logger.warning(
                    "retract_lessons_from_trajectory vector pass failed: %s", e
                )

        if removed:
            try:
                pretty_log(
                    "Skill Retracted",
                    f"removed {removed} lesson(s) sourced from "
                    f"trajectory {trajectory_id[:8]}",
                    icon=Icons.MEM_WIPE,
                )
            except Exception:
                pass
        return removed

    def record_retrieval(self, trigger: str):
        """Mark that a lesson with `trigger` was surfaced to the agent.

        Feedback loop hook: the caller increments this when a lesson is
        injected into the SKILL PLAYBOOK context block. Later,
        `record_helpful_retrieval` decides post-hoc whether the task
        succeeded. Also emits a debug log keyed by
        `source_challenge_hash` / `source` so the "is this self-play
        lesson ever actually used?" question has data to answer.
        """
        if not trigger:
            return False

        _hit_info = {"hash": "", "source": ""}

        def _match(raw):
            t = (raw.get("trigger") or raw.get("task") or "").strip().lower()
            return t == trigger.strip().lower()

        def _mut(lesson):
            lesson["retrievals"] = int(lesson.get("retrievals") or 0) + 1
            lesson["last_retrieved_at"] = _now_iso()
            _hit_info["hash"] = str(lesson.get("source_challenge_hash") or "")
            _hit_info["source"] = str(lesson.get("source") or "")

        updated = self._update_lesson_fields(_match, _mut)
        if updated:
            logger.debug(
                "lesson_retrieval trigger=%r source=%s hash=%s",
                trigger[:80], _hit_info["source"] or "?", _hit_info["hash"][:12] or "?",
            )
        return updated

    def record_helpful_retrieval(self, trigger: str):
        """Mark that the most-recent retrieval of `trigger` preceded a
        successful outcome. Increments helpful_retrievals and lightly
        bumps confidence."""
        if not trigger:
            return False

        def _match(raw):
            t = (raw.get("trigger") or raw.get("task") or "").strip().lower()
            return t == trigger.strip().lower()

        def _mut(lesson):
            lesson["helpful_retrievals"] = int(lesson.get("helpful_retrievals") or 0) + 1
            lesson["confidence"] = min(1.0, float(lesson.get("confidence") or 0.5) + 0.05)

        return self._update_lesson_fields(_match, _mut)

    def credit_recent_retrievals(self, window_seconds: int = 300, *,
                                 query: str = "",
                                 top_triggers=None,
                                 min_token_overlap: int = 2) -> int:
        """Increment `helpful_retrievals` on lessons retrieved in the last
        `window_seconds` that are ACTUALLY RELEVANT to the succeeding query.

        Designed to be called right after a turn / post-mortem that completed
        successfully. The old behaviour credited EVERY recently-retrieved
        lesson, which made `helpful_retrievals` ≈ `retrievals` and turned the
        utility ranking into noise. Crediting is now discriminative:

          * If `query` is given, a recent lesson is credited only when its
            trigger/pattern shares at least `min_token_overlap` significant
            (stopword-stripped) tokens with the query, OR
          * its normalized trigger is in `top_triggers` — the set of triggers
            that were the top-ranked retrievals for this query.

        Backward-compatible: when NEITHER `query` NOR `top_triggers` is
        supplied (legacy callers), we fall back to the original "credit every
        recent retrieval" behaviour so existing call sites keep working.

        Returns the number of lessons credited. Idempotent per window:
        we stamp `last_credited_at` so a second success inside the same
        window does not double-count.
        """
        if window_seconds <= 0:
            return 0
        cutoff = datetime.now() - timedelta(seconds=window_seconds)

        # Relevance inputs. `discriminate` stays False for legacy callers
        # (no query / no top_triggers) → original credit-everything behaviour.
        query_tokens = _trigger_token_set(query) if query else set()
        top_keys = {_normalize_trigger(t) for t in (top_triggers or []) if t}
        discriminate = bool(query_tokens) or bool(top_keys)

        def _is_relevant(lesson) -> bool:
            if not discriminate:
                return True
            trig = lesson.get("trigger") or lesson.get("task") or ""
            if top_keys and _normalize_trigger(trig) in top_keys:
                return True
            if query_tokens:
                lesson_tokens = _trigger_token_set(trig)
                # widen with anti/correct-pattern tokens so a lesson whose
                # trigger is terse but whose pattern clearly addresses the
                # query still counts.
                lesson_tokens |= _trigger_token_set(lesson.get("correct_pattern") or "")
                if len(query_tokens & lesson_tokens) >= max(1, min_token_overlap):
                    return True
            return False

        credited = 0
        with self._get_lock():
            playbook = self._load_playbook()
            mutated = False
            for idx, raw in enumerate(playbook):
                lesson = _normalize_lesson(raw)
                last = lesson.get("last_retrieved_at") or ""
                if not last:
                    continue
                try:
                    last_dt = datetime.fromisoformat(last)
                except Exception:
                    continue
                if last_dt < cutoff:
                    continue
                # Discriminative gate: skip recently-retrieved lessons that
                # are not actually relevant to the succeeding query.
                if not _is_relevant(lesson):
                    continue
                # Idempotency guard: only credit once per retrieval.
                last_credit = lesson.get("last_credited_at") or ""
                if last_credit:
                    try:
                        if datetime.fromisoformat(last_credit) >= last_dt:
                            continue
                    except Exception:
                        pass
                lesson["helpful_retrievals"] = int(lesson.get("helpful_retrievals") or 0) + 1
                lesson["confidence"] = min(1.0, float(lesson.get("confidence") or 0.5) + 0.05)
                lesson["last_credited_at"] = _now_iso()
                playbook[idx] = lesson
                mutated = True
                credited += 1
            if mutated:
                self._save_playbook_unlocked(playbook)
        return credited

    def prune_low_utility(self, min_retrievals: int = 5, max_drop_fraction: float = 0.25,
                          memory_system=None) -> int:
        """Drop lessons whose utility score is in the bottom quartile
        **and** that have been retrieved at least `min_retrievals` times
        (we need real data before punishing a lesson).

        Verified lessons are always retained regardless of score. Never
        drops more than `max_drop_fraction` of the playbook in one pass.

        When `memory_system` is given, each pruned lesson's embedded vector
        twin is also deleted so the stores don't drift (orphan vectors).
        """
        removed = 0
        _pruned_lessons = []
        with self._get_lock():
            playbook = self._load_playbook()
            if len(playbook) < 10:
                return 0
            scored = []
            for raw in playbook:
                lesson = _normalize_lesson(raw)
                scored.append((compute_lesson_utility(lesson), lesson))
            scored.sort(key=lambda kv: kv[0])

            # Candidates: below-median, retrieved enough times, not verified.
            sorted_scores = [s for s, _ in scored]
            cutoff = sorted_scores[max(0, len(sorted_scores) // 4)]
            cap = max(1, int(len(playbook) * max_drop_fraction))
            survivors = []
            pruned = []
            for score, lesson in scored:
                if (
                    score <= cutoff
                    and int(lesson.get("retrievals") or 0) >= min_retrievals
                    and not lesson.get("verified")
                    and len(pruned) < cap
                ):
                    pruned.append(lesson)
                    continue
                survivors.append(lesson)
            if not pruned:
                return 0
            # Preserve original ordering (most-recent first) among survivors.
            survivors.sort(
                key=lambda l: l.get("timestamp", ""), reverse=True
            )
            self._save_playbook_unlocked(survivors)
            removed = len(pruned)
            _pruned_lessons = list(pruned)
        # Vector-twin scrub AFTER a successful save (canonical store first).
        if removed and memory_system is not None:
            for lesson in _pruned_lessons:
                _delete_lesson_twin(memory_system, lesson)
        if removed:
            pretty_log(
                "SKILL PRUNE",
                f"Dropped {removed} low-utility lesson(s).",
                icon=Icons.MEM_WIPE,
            )
        return removed

    def get_playbook_context(
        self,
        query: str = None,
        memory_system=None,
        *,
        distance_threshold: float = DEFAULT_RETRIEVAL_DISTANCE,
        limit: int = 5,
        record_retrievals: bool = True,
    ) -> str:
        """Render the top lessons relevant to `query` for prompt injection.

        Uses vector search first (if memory_system provided), tightens
        the distance threshold, and re-ranks the candidates with a BM25-
        lite overlap score against the lesson trigger. Increments
        retrieval counters on every lesson actually surfaced so the
        feedback loop can tell which lessons the agent sees."""
        with self._get_lock():
            playbook_snapshot = self._load_playbook()

        # Build a {trigger → index} map for quick counter bumping.
        def _trigger_of(lesson):
            return (lesson.get("trigger") or lesson.get("task") or "").strip().lower()

        vector_attempted = False
        try:
            if memory_system and query:
                vector_attempted = True
                try:
                    results = memory_system.collection.query(
                        query_texts=[query],
                        n_results=max(limit * 2, 10),
                        where={"type": "skill"},
                    )
                except Exception as e:
                    logger.debug(f"Vector retrieval failed, falling back to JSON: {e}")
                    results = None
                    vector_attempted = False

                candidates = []  # (combined_score, distance, doc, meta)
                if results and results.get("documents") and results["documents"][0]:
                    docs = results["documents"][0]
                    dists = results["distances"][0]
                    metas = (results.get("metadatas") or [[]])[0] if results.get("metadatas") else [{}] * len(docs)
                    for doc, dist, meta in zip(docs, dists, metas or [{}] * len(docs)):
                        if dist >= distance_threshold:
                            continue
                        trigger = (meta or {}).get("trigger", "") or _extract_trigger_from_doc(doc)
                        bm25 = _bm25_like_score(query, trigger or doc)
                        # Lower distance is better; higher bm25 is better.
                        combined = (1.0 - dist) + bm25 * 0.4
                        candidates.append((combined, dist, doc, meta or {}, trigger))
                if candidates:
                    candidates.sort(key=lambda t: -t[0])
                    chosen = candidates[:limit]
                    lines = ["## RELEVANT LESSONS LEARNED (Follow these to avoid repeats):"]
                    for i, (_, _dist, doc, meta, trig) in enumerate(chosen):
                        # Prefer the structured rendering from the on-disk
                        # copy (has verified flag, code example) — fall
                        # back to the raw embedded doc if no playbook
                        # match was found.
                        lesson_entry = _find_playbook_entry_by_trigger(playbook_snapshot, trig)
                        if lesson_entry:
                            lines.append(f"{i+1}. {render_lesson_for_prompt(lesson_entry)}")
                        else:
                            lines.append(f"{i+1}. {doc}")
                        if record_retrievals and trig:
                            try:
                                self.record_retrieval(trig)
                            except Exception:
                                pass
                    return "\n".join(lines)
                # Vector search was attempted and came back clean (or
                # nothing above threshold). Preserve the legacy
                # contract: don't fall back to recency in that case,
                # just return empty. Recency fallback applies only
                # when the vector path wasn't available at all.
                if vector_attempted:
                    return ""
        except Exception as e:
            logger.debug(f"Playbook retrieval path failed: {e}")

        if not playbook_snapshot:
            return "No lessons learned yet."

        # BM25 fallback: when the caller has a query but the vector
        # path wasn't attempted (memory_system is None — common when
        # VectorMemory init failed or is disabled), do keyword-overlap
        # filtering on the playbook's triggers BEFORE falling back to
        # recency. Without this, a query like "what's the capital of
        # France?" would surface a Python-syntax lesson just because
        # it happened to be the most recent. The recency fallback is
        # only appropriate when there's NO query — i.e. the caller
        # wants a generic "any recent lesson" injection for the system
        # prompt, not a per-turn retrieval.
        if query and str(query).strip():
            scored: List[Tuple[float, dict]] = []
            for p in playbook_snapshot:
                trig = _trigger_of(p)
                score = _bm25_like_score(query, trig or "")
                if score > 0:
                    scored.append((score, p))
            if scored:
                scored.sort(key=lambda t: -t[0])
                lines = ["## RELEVANT LESSONS LEARNED (Follow these to avoid repeats):"]
                for i, (_, p) in enumerate(scored[:limit]):
                    lines.append(f"{i+1}. {render_lesson_for_prompt(p)}")
                    if record_retrievals:
                        trig = _trigger_of(p)
                        if trig:
                            try:
                                self.record_retrieval(trig)
                            except Exception:
                                pass
                return "\n".join(lines)
            # Had a query, zero BM25 hits → no lesson is relevant to
            # this turn. Return empty instead of dumping recency.
            return ""

        # No query supplied → recency fallback (system-prompt injection style).
        lines = ["## RECENT LESSONS LEARNED (Follow these to avoid repeats):"]
        for i, p in enumerate(playbook_snapshot[:limit]):
            lines.append(f"{i+1}. {render_lesson_for_prompt(p)}")
            if record_retrievals:
                trig = _trigger_of(p)
                if trig:
                    try:
                        self.record_retrieval(trig)
                    except Exception:
                        pass
        return "\n".join(lines)

    def get_recent_failures(self, limit: int = 5) -> str:
        """Fetch the most recent mistakes and their tasks to generate targeted self-play scenarios."""
        try:
            with self._get_lock():
                playbook = self._load_playbook()

            if not playbook:
                return "No recent failures recorded."

            recent_lessons = playbook[:limit]
            context = "## RECENT MISTAKES:\n"
            for p in recent_lessons:
                task = p.get("task") or p.get("trigger") or ""
                mistake = p.get("mistake") or p.get("anti_pattern") or ""
                context += f"- TASK: {task}\n  ERROR/MISTAKE: {mistake}\n\n"
            return context.strip()
        except Exception:
            return "Failed to load recent failures."

    def list_lessons(
        self,
        *,
        scope: str = "all",
        source: str = "",
        limit: int = 20,
    ) -> list:
        """Return lessons filtered by time window + source, most-recent first.

        `scope` ∈ {"today", "week", "all"} — boundary is **local** wall-clock
        (today starts at local midnight; week = last 7 days from `now`).
        `source` filters by the provenance string on the lesson (e.g.
        "self_play") — empty matches any source. Both legacy and structured
        lessons are returned; entries are normalized before filtering so
        timestamp / source / domains are always present.
        """
        cutoff = None
        now_local = datetime.now()
        s = (scope or "all").strip().lower()
        if s == "today":
            cutoff = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        elif s in ("week", "7d"):
            cutoff = now_local - timedelta(days=7)
        # any other value (including "all") leaves cutoff=None

        src_filter = (source or "").strip().lower()

        with self._get_lock():
            snapshot = self._load_playbook()

        filtered = []
        for raw in snapshot:
            lesson = _normalize_lesson(raw)
            ts = lesson.get("timestamp") or ""
            dt = None
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    dt = None
            if cutoff is not None:
                if dt is None or dt < cutoff:
                    continue
            if src_filter:
                if (lesson.get("source") or "").strip().lower() != src_filter:
                    continue
            filtered.append((dt or datetime.min, lesson))

        filtered.sort(key=lambda kv: kv[0], reverse=True)
        return [l for _, l in filtered[: max(0, int(limit))]]

    def find_by_trigger(self, trigger: str) -> dict:
        """Return the first lesson with an exact-match trigger (case
        insensitive). Used by the verification-grounded lesson flow."""
        if not trigger:
            return None
        target = trigger.strip().lower()
        with self._get_lock():
            playbook = self._load_playbook()
        for raw in playbook:
            t = (raw.get("trigger") or raw.get("task") or "").strip().lower()
            if t == target:
                return _normalize_lesson(raw)
        return None

    def mark_verified(self, trigger: str, verified: bool = True):
        """Flip the verified flag on a lesson, nudging confidence."""

        def _match(raw):
            t = (raw.get("trigger") or raw.get("task") or "").strip().lower()
            return t == (trigger or "").strip().lower()

        def _mut(lesson):
            lesson["verified"] = bool(verified)
            lesson["verification_attempted"] = True
            if verified:
                lesson["confidence"] = min(1.0, float(lesson.get("confidence") or 0.5) + 0.2)

        return self._update_lesson_fields(_match, _mut)

    def remove_by_trigger(self, trigger: str, memory_system=None) -> bool:
        """Delete the first lesson with a matching trigger. Returns True
        if one was removed. Used when verification proves a lesson
        unhelpful / actively harmful. When `memory_system` is given, the
        lesson's embedded vector twin is deleted too (no orphan)."""
        if not trigger:
            return False
        target = trigger.strip().lower()
        removed_lesson = None
        with self._get_lock():
            playbook = self._load_playbook()
            for idx, raw in enumerate(playbook):
                t = (raw.get("trigger") or raw.get("task") or "").strip().lower()
                if t == target:
                    removed_lesson = playbook[idx]
                    del playbook[idx]
                    self._save_playbook_unlocked(playbook)
                    break
        if removed_lesson is not None:
            if memory_system is not None:
                _delete_lesson_twin(memory_system, removed_lesson)
            return True
        return False


# ---------------------------------------------------------------------------
# Module-level helpers reused by Retrieval feedback and by the Dreamer to
# correlate embedded docs back to playbook entries.
# ---------------------------------------------------------------------------

def _find_playbook_entry_by_trigger(playbook: list, trigger: str):
    if not trigger:
        return None
    target = trigger.strip().lower()
    for entry in playbook or []:
        t = (entry.get("trigger") or entry.get("task") or "").strip().lower()
        if t == target:
            return entry
    return None


def _extract_trigger_from_doc(doc: str) -> str:
    """Pull the SITUATION line out of a legacy embedded lesson."""
    if not doc:
        return ""
    m = re.search(r"SITUATION:\s*(.+)", doc)
    return m.group(1).strip() if m else ""
