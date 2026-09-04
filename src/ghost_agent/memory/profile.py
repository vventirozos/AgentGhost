import json
import logging
import threading
import os
from pathlib import Path
from typing import Any, Dict
from ..utils.logging import pretty_log, Icons
from .temporal import anchor as _anchor, derive as _derive
from .temporal import has_anchor as _has_anchor, signature as _signature

# Keys whose value is inherently SINGLE-valued, so a second write is a
# CORRECTION and must REPLACE rather than merge.
#
# The merge branch below exists for facts that genuinely coexist
# ("python" AND "rust" are both interests). Applying it to a singular
# noun turned every correction into an accumulation:
# ``update("relationships","wife","Anna")`` then ``(…,"wife","Maria")``
# produced ``- wife: Anna, Maria`` — injected into the system prompt every
# turn, for ever, with nothing anywhere able to reconcile it. Note that
# `wife`/`husband`/`son`/`daughter`/`car` are keys the canonicalisation
# table in update() explicitly CREATES, so they were guaranteed to hit
# the merge branch.
#
# Rule of thumb for extending this: SINGULAR key ⇒ singleton. Genuinely
# multi-valued facts belong under a PLURAL key (`pets`, `children`,
# `languages`, `topics`), which keeps merge semantics untouched.
_SINGLETON_KEYS = {
    # identity
    "name", "role", "email", "timezone", "age",
    "birthday", "pronouns", "title", "location",
    # relationships (singular — the canonicalisation table creates these)
    "wife", "husband", "spouse", "partner",
    "mother", "father", "mom", "dad",
    "son", "daughter", "child",
    # possessions / residence (singular)
    "car", "vehicle", "home", "house", "address", "phone",
    # work
    "employer", "company", "job", "occupation", "nationality",
}

# Upper bound on how many values a MERGING (multi-valued) key may hold.
# Nothing capped these before: every merged key grew without limit and is
# rendered inline by get_context_string() into every system prompt, so an
# unbounded key is a slow context-pressure leak. Oldest values are dropped
# first (the newest statement is the most likely to be current).
_MAX_VALUES_PER_KEY = 8

# `notes.info` is the DEFAULT sink for a malformed profile_update: the
# callers (core.agent's smart-memory path and core.bus) do
# ``profile_up.get("category", "notes"), profile_up.get("key", "info")``,
# so ANY dict the model emits without those keys lands here — with the
# whole extracted fact as the value. It is not a singleton, so it appended
# for ever with no cap, no TTL and no dedup beyond exact-string match.
# Keep it as a tiny ring of short values: it is a junk drawer, not memory
# (real facts also go to the vector store on that same path).
_SINK_KEYS = {("notes", "info")}
_SINK_MAX_VALUES = 3
_SINK_MAX_VALUE_CHARS = 200

# ── Layer 3: per-value provenance (`as_of`) ─────────────────────────────
#
# Anchoring (see memory/temporal.py) fixes decaying facts that have a
# DERIVABLE invariant — an age has a birth date behind it. Plenty of facts
# decay with no invariant to derive: an employer, a location, "currently
# learning X". For those the only truthful thing the store can say is WHEN
# it learned them, and until now it could not say even that: a value was a
# bare string, so the profile had no idea when anything was written.
#
# That gap was not hypothetical. The corpus-repair script could not ask
# this store when `relationships.sons` was recorded and had to infer the
# date from the contradiction log — a different store — which is why it
# carries a --said-at flag and a refuse-to-guess rule.
#
# ON-DISK SHAPE. A value is either a bare string/list (LEGACY: as_of
# unknown) or a stamped object:
#
#     "company":   {"v": "EvolMonkey",   "as_of": "2026-01-15T09:00:00Z"}
#     "languages": [{"v": "python", "as_of": …}, {"v": "rust", "as_of": …}]
#
# Stamps are PER VALUE, not per key, because a merging key accumulates its
# items at different times — one stamp for the key would be a lie about
# every item but the last.
#
# A legacy bare value stays unstamped. It is NOT back-filled with "now":
# that would fabricate provenance, which is worse than admitting none.
# `scripts/repair_temporal_anchors.py --targets profile-stamps` recovers
# real dates from the vector store's derived facts, with the evidence
# printed for review.
#
# READER CONTRACT. `load()` returns the LEGACY shape — values unwrapped to
# plain strings/lists — so every existing reader is untouched (all twelve
# production call sites, enumerated from the AST, go through `load()`).
# Code that wants provenance uses `load_raw()` or `as_of()`.
_VALUE_KEY = "v"
_AS_OF_KEY = "as_of"

# Keys whose value does not meaningfully decay. Stamping them is pure
# prompt noise: a name or a birth date is not more doubtful for being
# recorded a year ago. Note `sons`/`children` are durable here because
# their AGES are handled by anchoring, which self-updates.
_DURABLE_KEYS = {
    "name", "birthday", "birthdate", "born", "pronouns", "nationality",
    "wife", "husband", "spouse", "partner", "wife_name", "husband_name",
    "mother", "father", "mom", "dad",
    "son", "sons", "daughter", "daughters", "child", "children",
}

# A perishable value older than this is rendered with its date, so the
# model can weigh it instead of asserting it. Below the threshold the line
# stays clean — a fact learned last week needs no caveat.
_STALE_AFTER_DAYS = 90
# Past this, say so outright.
_VERY_STALE_AFTER_DAYS = 365


def _is_stamped(item: Any) -> bool:
    return isinstance(item, dict) and _VALUE_KEY in item


def unwrap(item: Any) -> Any:
    """A stored item → its plain value. Identity on legacy values."""
    if _is_stamped(item):
        return item[_VALUE_KEY]
    if isinstance(item, list):
        return [unwrap(i) for i in item]
    return item


def stamp_of(item: Any) -> Any:
    """A stored item → its ISO `as_of`, or None when unknown."""
    if _is_stamped(item):
        got = item.get(_AS_OF_KEY)
        return got if isinstance(got, str) and got.strip() else None
    return None


def _wrap(value: Any, as_of: str) -> Any:
    """A plain value + a date → the stamped item to persist.

    ``as_of`` is required. An earlier revision returned the bare value for
    a falsy date, but a mutation run showed that branch was unreachable —
    update() always resolves a date and stamp() now rejects a falsy one —
    so it was dead code no pin could distinguish from a live guard. The
    "never fabricate provenance" rule is enforced where it is actually
    reachable: legacy values are simply left alone, and the backfill
    reports a value it cannot date rather than stamping it with today.
    """
    return {_VALUE_KEY: value, _AS_OF_KEY: as_of}


class ProfileMemory:
    def __init__(self, path: Path):
        self.file_path = path / "user_profile.json"
        self._lock = threading.RLock()
        # Fail-closed flag (same discipline as contradiction_log /
        # adaptive_threshold / competence): a transient READ failure must
        # block writes until a read succeeds, or the next save() overwrites
        # the intact on-disk identity with the default skeleton.
        self._degraded = False
        if not self.file_path.exists():
            self.save({"root": {"name": "User"}, "relationships": {}, "interests": {}, "assets": {}})

    def load_raw(self) -> Dict[str, Any]:
        _default = {"root": {"name": "User"}, "relationships": {}, "interests": {}, "assets": {}}
        with self._lock:
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                # A valid-JSON-but-wrong-TYPE file (e.g. a list or a scalar)
                # would break every caller (data[cat] = {}). Treat it as corrupt.
                if not isinstance(data, dict):
                    raise ValueError(f"profile is a {type(data).__name__}, expected object")
                self._degraded = False
                return data
            except FileNotFoundError:
                return dict(_default)
            except OSError as e:
                # Transient disk sickness (EIO/EACCES/…), NOT corruption:
                # the on-disk profile is probably intact, so do not shunt
                # it to a sidecar — serve the default in-memory and refuse
                # writes until a read succeeds. The old path treated this
                # like corruption: intact profile sidecar'd away, identity
                # reverted, and if the replace also failed the next save()
                # destroyed the real file.
                self._degraded = True
                pretty_log("Profile Read Failed",
                           f"{type(e).__name__}: {e}; serving default identity, "
                           "writes DISABLED until a read succeeds",
                           icon=Icons.USER_ID, level="WARNING")
                return dict(_default)
            except Exception as e:
                # A corrupt profile would otherwise silently revert the user's
                # identity to the default — and the next save() would OVERWRITE
                # the real file, destroying the facts (and any forensic copy).
                # Preserve the bad file as a timestamped sidecar first (same
                # discipline as journal.py / frontier.py).
                try:
                    import time as _time
                    sidecar = self.file_path.with_suffix(f".corrupt-{int(_time.time())}.json")
                    if self.file_path.exists():
                        os.replace(self.file_path, sidecar)
                        pretty_log("Profile Corrupt",
                                   f"{type(e).__name__}: {e}; preserved at {sidecar.name}, "
                                   "reverting to default identity",
                                   icon=Icons.USER_ID, level="WARNING")
                    else:
                        pretty_log("Profile Corrupt", f"{type(e).__name__}: {e}",
                                   icon=Icons.USER_ID, level="WARNING")
                except Exception:
                    pretty_log("Profile Corrupt",
                               f"{type(e).__name__}: {e}; could not preserve, reverting to default",
                               icon=Icons.USER_ID, level="WARNING")
                return dict(_default)

    def load(self) -> Dict[str, Any]:
        """The profile with every value UNWRAPPED to its plain form.

        This is the contract every existing reader already has (all twelve
        production call sites go through here), so adding provenance to the
        on-disk shape changes nothing for them. Code that wants the stamps
        calls :meth:`load_raw` or :meth:`as_of`.
        """
        data = self.load_raw()
        return {cat: ({k: unwrap(v) for k, v in fields.items()}
                      if isinstance(fields, dict) else fields)
                for cat, fields in data.items()}

    def as_of(self, category: str, key: str):
        """When this store learned ``category.key`` — ISO string, or None
        when unknown (a legacy value, written before provenance existed).

        For a multi-valued key this is the NEWEST stamp present: the answer
        to "when did I last learn something here". Returning None rather
        than a guess is deliberate — a fabricated provenance is worse than
        an admitted gap, and the corpus-repair script depends on being able
        to tell the difference.
        """
        with self._lock:
            data = self.load_raw()
            cat, k = self.canonicalize(category, key)
            fields = data.get(cat)
            if not isinstance(fields, dict) or k not in fields:
                return None
            item = fields[k]
            stamps = [stamp_of(i) for i in item] if isinstance(item, list) \
                else [stamp_of(item)]
            stamps = [t for t in stamps if t]
            return max(stamps) if stamps else None

    def _preserve_stamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Re-attach the on-disk stamp to any value handed to save() in
        plain form that is UNCHANGED.

        ``load()`` unwraps, so a caller doing ``save(load())`` would
        otherwise wipe every stamp in the file — a footgun that no
        convention can close, since the two shapes are indistinguishable
        once unwrapped. A CHANGED value keeps whatever the caller gave
        (update() is the path that stamps); only identical values get
        their provenance back. Best-effort: a read failure here must never
        block the write.
        """
        try:
            current = self.load_raw()
        except Exception:
            return data
        for cat, fields in data.items():
            old_fields = current.get(cat)
            if not isinstance(fields, dict) or not isinstance(old_fields, dict):
                continue
            for k, v in list(fields.items()):
                prev = old_fields.get(k)
                if prev is None or _is_stamped(v):
                    continue
                if isinstance(prev, list) and isinstance(v, list):
                    fields[k] = [p if _is_stamped(p) and unwrap(p) == n else n
                                 for p, n in zip(prev, v)] + v[len(prev):] \
                        if len(v) >= len(prev) else v
                elif _is_stamped(prev) and unwrap(prev) == v:
                    fields[k] = prev
        return data

    def save(self, data: Dict[str, Any]):
        with self._lock:
            data = self._preserve_stamps(data)
            if self._degraded:
                pretty_log("Profile Write Blocked",
                           "profile store is read-degraded; refusing to "
                           "overwrite the on-disk identity",
                           icon=Icons.USER_ID, level="WARNING")
                return
            temp_path = self.file_path.with_suffix('.tmp')
            # fsync before rename: the rename alone can publish a torn/empty
            # file on power loss (see journal.py's identical rationale).
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.file_path)

    @staticmethod
    def _bounded(values: list, cap: int, cat: str, key: str) -> list:
        """Trim a merged value list to ``cap``, dropping the OLDEST first.

        Returns a scalar-free list; a one-element result is still a list so
        the caller's list/scalar handling elsewhere (prune_value collapses
        singletons) stays the single place that decides that."""
        if cap <= 0 or len(values) <= cap:
            return values
        dropped = len(values) - cap
        trimmed = values[-cap:]
        if (cat, key) in _SINK_KEYS:
            # The junk-drawer ring rotating at its designed cap is routine
            # by-design behaviour, not a leak signal — forensics-level only,
            # never the operator stream (flagged as noise 2026-07-25).
            logging.getLogger(__name__).debug(
                "profile sink %s.%s rotated: dropped %d oldest value(s)",
                cat, key, dropped)
        else:
            # A real multi-valued key overflowing ITS cap is the
            # context-pressure-leak signal the warning exists for.
            pretty_log(
                "Profile Capped",
                f"{cat}.{key} hit the {cap}-value cap; dropped {dropped} oldest "
                f"value(s) to keep the system prompt bounded",
                icon=Icons.USER_ID, level="WARNING",
            )
        return trimmed

    # Canonicalisation table: the model often passes synonyms. Normalised
    # once at write time so retrieval is deterministic. Module-visible via
    # `canonicalize()` because SIBLING stores mint from the same field —
    # §4M (Lens C MAJOR-4): the graph edge was built from the RAW key
    # (`HAS_VEHICLE`) while the profile stored the canonical one
    # (`assets.car`), so the two stores permanently disagreed on 11/15
    # live pairs. Every caller composing a triplet/vector text for a
    # profile field must canonicalise FIRST.
    _CANONICAL_FIELD_MAP = {
        "wife": ("relationships", "wife"),
        "husband": ("relationships", "husband"),
        "son": ("relationships", "son"),
        "daughter": ("relationships", "daughter"),
        "car": ("assets", "car"),
        "vehicle": ("assets", "car"),
        "science": ("interests", "science"),
        "interest": ("interests", "general"),
    }

    @staticmethod
    def _same_fact(a: str, b: str) -> bool:
        """True when two values state the SAME fact.

        Exact equality, as before — EXCEPT when either side carries a
        temporal anchor, where the comparison goes anchor-blind. Without
        that exemption, anchoring would convert a harmless restatement
        into an accumulation: "Leonidas is 4 months old" and, six weeks
        later, "…is 5 months old" anchor to different dates, so the
        exact-string dedup below would keep BOTH and inject two
        contradictory birth dates into every system prompt. Matching on
        the signature lets the newer statement REFINE the anchor in place.

        Deliberately NOT a general case/whitespace-insensitive compare:
        values without an anchor keep byte-exact semantics so this change
        cannot alter dedup for the rest of the store.
        """
        if a == b:
            return True
        if _has_anchor(a) or _has_anchor(b):
            return _signature(a) == _signature(b)
        return False

    @classmethod
    def canonicalize(cls, category: str, key: str):
        """(category, key) → the canonical (category, key) this store will
        actually file the value under. Idempotent; safe on unmapped keys."""
        cat = str(category or "").strip().lower()
        k = str(key or "").strip().lower()
        if k in cls._CANONICAL_FIELD_MAP:
            return cls._CANONICAL_FIELD_MAP[k]
        return cat, k

    def update(self, category: str, key: str, value: Any, as_of=None):
        """Write ``value`` under ``category.key``, stamped with when it was
        learned.

        ``as_of`` defaults to now — the caller is telling us this fact NOW.
        It is explicit so the backfill script can attribute a recovered
        date, and so tests can write a value as of a past day. Operates on
        the RAW shape: going through load() would unwrap every other value
        in the file and the save() below would strip their provenance.
        """
        from ..utils.helpers import get_utc_timestamp
        with self._lock:
            as_of = as_of or get_utc_timestamp()
            data = self.load_raw()
            cat = str(category).strip().lower()
            k = str(key).strip().lower()
            # TEMPORAL ANCHORING at the store boundary. A stated age is a
            # measurement, not a fact — "Leonidas is 4 months old" is only
            # true on the day it is said, and this store renders bare,
            # UNSTAMPED `- key: value` lines into every system prompt, so a
            # snapshot reads as present tense for ever (observed live: a
            # fact stated 2026-07-07 still recalled verbatim 2 months
            # later). Rewrite it to the invariant behind it — a birth date
            # — and let get_context_string() derive the current value at
            # render time. Done HERE rather than per-caller because every
            # writer funnels through update(): the update_profile tool
            # (both its bus and legacy paths), the memory bus's _profile
            # leg, and smart-memory consolidation. anchor() is idempotent,
            # so a caller that already anchored (to attach the right
            # said_at, or to keep sibling stores in step) is unaffected.
            v = _anchor(str(value).strip())

            original_cat, original_key = cat, k
            cat, target_key = self.canonicalize(cat, k)

            # Ensure category exists as a dictionary
            if cat not in data or not isinstance(data[cat], dict):
                data[cat] = {}

            # Singleton keys: identity-style facts where the user has
            # exactly one value. Merging here produced absurd results
            # like `name: ["User", "Vasilis"]` (the seeded default kept
            # alongside the user-supplied name). For these keys we
            # always REPLACE; for everything else the merge behavior
            # below applies. (Module-level table — see _SINGLETON_KEYS.)
            if target_key in _SINGLETON_KEYS:
                data[cat][target_key] = _wrap(v, as_of)
                self.save(data)
                if (cat, target_key) != (original_cat, original_key):
                    return (f"Synchronized: {cat}.{target_key} = {v}  "
                            f"[normalised from {original_cat}.{original_key}]")
                return f"Synchronized: {cat}.{target_key} = {v}"

            # MERGE semantics (was overwrite). Profile facts often coexist
            # ("python" AND "rust" are both interests; the user owns BOTH a
            # car and a bike). Overwriting silently dropped prior facts. We
            # now keep both values as a deduped, order-preserved list. If
            # the new value matches what's already stored, no-op.
            #
            # BOUNDED since 2026-07-22: the accumulation is capped (oldest
            # first out) so a merging key can no longer grow without limit
            # into every system prompt. `notes.info` — the sink a malformed
            # profile_update defaults into — gets a much tighter ring.
            is_sink = (cat, target_key) in _SINK_KEYS
            cap = _SINK_MAX_VALUES if is_sink else _MAX_VALUES_PER_KEY
            if is_sink and len(v) > _SINK_MAX_VALUE_CHARS:
                v = v[:_SINK_MAX_VALUE_CHARS].rstrip() + "…"

            existing = data[cat].get(target_key)
            if existing is None or unwrap(existing) == "":
                data[cat][target_key] = _wrap(v, as_of)
            elif isinstance(existing, list):
                # Compare on the UNWRAPPED text; a stamp is provenance, not
                # part of the fact, so it must not affect dedup.
                items = [str(unwrap(x)).strip() for x in existing]
                # _same_fact, not ==: a restatement that only moves the
                # anchor date REFINES the stored value in place instead of
                # appending a second, contradictory birth date.
                hit = next((i for i, it in enumerate(items)
                            if self._same_fact(it, v)), None)
                if hit is None:
                    data[cat][target_key] = self._bounded(
                        list(existing) + [_wrap(v, as_of)], cap, cat, target_key)
                else:
                    # Refined in place, or restated verbatim. EITHER WAY the
                    # item carries the new date: the user saying a fact
                    # again is evidence it is still true, and "last
                    # confirmed" is exactly what the staleness marker is
                    # reporting. The scalar branch below does the same; an
                    # exact-duplicate no-op here would have left one path
                    # refreshing and the other not.
                    merged = list(existing)
                    merged[hit] = _wrap(v, as_of)
                    data[cat][target_key] = merged
            else:
                # Scalar existing value
                existing_str = str(unwrap(existing)).strip()
                if self._same_fact(existing_str, v):
                    # Same fact: keep the NEWER phrasing, which for an
                    # anchored value is the refined anchor. Byte-identical
                    # values fall through here too, as before — and either
                    # way the restatement refreshes the stamp, because the
                    # user has just told us it is still true.
                    data[cat][target_key] = _wrap(v, as_of)
                else:
                    # Promote to list, dedup, preserve order. The EXISTING
                    # item is carried over as-is so its own provenance
                    # survives the promotion — only the new value is
                    # stamped with now.
                    merged = [existing, _wrap(v, as_of)]
                    seen = set()
                    deduped = []
                    for item in merged:
                        plain = str(unwrap(item)).strip()
                        if plain not in seen:
                            seen.add(plain)
                            deduped.append(item)
                    data[cat][target_key] = self._bounded(deduped, cap,
                                                          cat, target_key)
            self.save(data)

            if (cat, target_key) != (original_cat, original_key):
                return f"Synchronized: {cat}.{target_key} = {v}  [normalised from {original_cat}.{original_key}]"
            return f"Synchronized: {cat}.{target_key} = {v}"

    def stamp(self, category: str, key: str, as_of: str) -> int:
        """Attach ``as_of`` to the UNSTAMPED items under ``category.key``.

        Surgical on purpose: the backfill recovers provenance for legacy
        values, and routing that through update() would re-anchor,
        canonicalise and cap them — a migration that can silently rewrite
        the values it was only supposed to date. This cannot change a
        value, and it never overwrites a stamp that already exists.

        Returns how many items it stamped.
        """
        if not as_of or not str(as_of).strip():
            # The caller could not date this value. Recording "unknown" as
            # a stamp would be worse than leaving it unstamped, which is
            # already how the store says "I don't know when".
            return 0
        with self._lock:
            data = self.load_raw()
            cat, k = self.canonicalize(category, key)
            fields = data.get(cat)
            if not isinstance(fields, dict) or k not in fields:
                return 0
            item = fields[k]
            n = 0
            if isinstance(item, list):
                out = []
                for i in item:
                    if _is_stamped(i):
                        out.append(i)
                    else:
                        out.append(_wrap(i, as_of))
                        n += 1
                fields[k] = out
            elif not _is_stamped(item):
                fields[k] = _wrap(item, as_of)
                n = 1
            if n:
                self.save(data)
            return n

    def delete(self, category: str, key: str) -> str:
        with self._lock:
            # RAW: load() unwraps, and the save() below would then strip
            # provenance from every OTHER key in the file.
            data = self.load_raw()
            cat = str(category).strip().lower()
            k = str(key).strip().lower()

            # Apply the same canonicalization mapping used by update()
            # §4M R2 MINOR-5: one canonical map (see _CANONICAL_FIELD_MAP)
            # — this was an inline verbatim copy; the next map entry would
            # have diverged write/delete/prune silently.
            cat, k = self.canonicalize(cat, k)

            if cat in data and k in data[cat]:
                del data[cat][k]
                # Clean up empty categories
                if not data[cat]:
                    del data[cat]
                self.save(data)
                return f"Removed from Profile: {cat}.{k}"

            return f"Profile key not found: {cat}.{k}"

    def prune_value(self, category: str, key: str, target: str) -> str:
        """Remove every list item (or a matching scalar) under
        ``category.key`` that *mentions* ``target``, persisting the result.

        This is the value-level counterpart to :meth:`delete` (which can
        only remove a whole key). It exists because pets / interests / assets
        are stored as VALUES inside a list — e.g.
        ``assets.pets = ["Hanzo the dog", "Mortimer the iguana (removed)"]`` —
        so ``forget('mortimer')`` previously had no way to reach them and the
        stale entry kept being injected into the system prompt every turn.

        Matching is token/word-boundary aware: ``forget('age')`` will NOT
        strip a value of ``"language"``, but ``forget('mortimer')`` DOES match
        ``"Mortimer the iguana (removed)"`` (even the soft-delete tombstone).
        Multi-word targets fall back to substring (tokens can't span spaces).

        Deletes the key when nothing survives, and the category when it
        becomes empty. Returns a human-readable report line.
        """
        import re
        with self._lock:
            # RAW, for the same reason as delete(); _mentions() unwraps.
            data = self.load_raw()
            cat = str(category).strip().lower()
            k = str(key).strip().lower()

            # Same canonicalisation table as update()/delete() so the value
            # sweep lands on the row the writer actually created.
            # §4M R2 MINOR-5: shared canonical map, same as delete().
            cat, k = self.canonicalize(cat, k)

            if cat not in data or k not in data[cat]:
                return f"Profile key not found: {cat}.{k}"

            target_lc = str(target).strip().lower()
            if not target_lc:
                return "Profile: empty target, nothing pruned."

            def _mentions(val) -> bool:
                v = str(unwrap(val)).lower()
                if " " in target_lc:
                    return target_lc in v
                return target_lc in re.split(r"[^a-z0-9]+", v)

            existing = data[cat][k]

            if isinstance(existing, list):
                removed = [it for it in existing if _mentions(it)]
                if not removed:
                    return f"No matching value under {cat}.{k}"
                kept = [it for it in existing if not _mentions(it)]
                if kept:
                    # Collapse a singleton list back to a scalar for tidiness
                    # (mirrors how update() promotes scalar→list only when >1).
                    data[cat][k] = kept if len(kept) > 1 else kept[0]
                    self.save(data)
                    return (f"Pruned {len(removed)} value(s) from {cat}.{k}: "
                            f"{[unwrap(r) for r in removed]}")
                # Nothing survived → drop the key (and category if now empty).
                del data[cat][k]
                if not data[cat]:
                    del data[cat]
                self.save(data)
                return f"Removed {cat}.{k} (all values matched '{target_lc}')"

            # Scalar value.
            if _mentions(existing):
                del data[cat][k]
                if not data[cat]:
                    del data[cat]
                self.save(data)
                return f"Removed {cat}.{k}"
            return f"No matching value under {cat}.{k}"

    def get_context_string(self) -> str:
        """Render the profile for the ``{{PROFILE}}`` system-prompt slot.

        Temporal anchors are DERIVED here, not stored: a value held as
        ``Leonidas (born ~2026-02-20)`` is rendered as
        ``Leonidas (born ~2026-02-20 → ~6 months old)``. The store keeps
        constants; the prompt gets the value that is true today. This is
        the whole point of the split — the model is never asked to
        subtract a timestamp it will not notice, because the line it reads
        already carries the answer.

        The gloss is a ROUNDED age, never a date or a day count, so this
        block still changes only when the derived age changes (about
        monthly for an infant, yearly for a child). ``{{PROFILE}}`` sits
        inside the byte-stable system prefix that core/agent.py keeps
        identical across turns for upstream prefix-cache hits; rendering
        "today" here would have invalidated that prefix daily.
        """
        # RAW: the stamps are what the staleness marker is rendered from.
        data = self.load_raw()
        lines = []
        for key, val in data.items():
            if not val: continue
            label = key.replace("_", " ").capitalize()
            if isinstance(val, dict):
                lines.append(f"## {label}:")
                for sub_k, sub_v in val.items():
                    # Sub-values may now be lists (multi-value merge); flatten
                    # them inline so the LLM sees "language: python, rust"
                    # rather than a Python repr like "['python', 'rust']".
                    if isinstance(sub_v, list):
                        sub_v_str = ", ".join(self._render_item(sub_k, i)
                                              for i in sub_v)
                    else:
                        sub_v_str = self._render_item(sub_k, sub_v)
                    lines.append(f"- {sub_k}: {sub_v_str}")
            elif isinstance(val, list):
                lines.append(f"## {label}: "
                             + ", ".join(self._render_item(key, i) for i in val))
            else:
                lines.append(f"{label}: {self._render_item(key, val)}")
        return "\n".join(lines)

    @classmethod
    def _render_item(cls, key: str, item) -> str:
        """One stored item → its prompt text: temporal anchors derived, and
        a staleness marker when the value is old enough for its age to
        matter.

        Two different mechanisms, deliberately. A fact with a DERIVABLE
        invariant (an age behind a birth date) is recomputed and needs no
        caveat — it is simply correct today. A fact with no invariant (an
        employer, a location) cannot be recomputed, so the only honest move
        is to say when it was learned and let the model weigh it.

        Quiet by default: a DURABLE key is never marked (a name is not more
        doubtful for being a year old), and a perishable value younger than
        the threshold renders exactly as before. An unstamped legacy value
        is never marked either — absent provenance is not evidence of age.
        """
        text = _derive(str(unwrap(item)))
        marker = cls._staleness_marker(key, stamp_of(item))
        return f"{text}{marker}"

    @staticmethod
    def _staleness_marker(key: str, as_of) -> str:
        """`` (as of 2026-01-15)`` / `` (as of …, may be stale)`` / ``""``."""
        if not as_of or str(key).strip().lower() in _DURABLE_KEYS:
            return ""
        try:
            import datetime
            from ..utils.helpers import parse_utc_timestamp
            then = parse_utc_timestamp(as_of)
            if then.tzinfo is not None:
                then = then.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            days = (now - then).days
        except Exception:
            # A malformed stamp must not break the system prompt.
            return ""
        if days < _STALE_AFTER_DAYS:
            return ""
        day = as_of[:10]
        if days >= _VERY_STALE_AFTER_DAYS:
            return f" (as of {day}, may be stale)"
        return f" (as of {day})"