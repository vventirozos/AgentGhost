import json
import threading
import os
from pathlib import Path
from typing import Any, Dict
from ..utils.logging import pretty_log, Icons

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


class ProfileMemory:
    def __init__(self, path: Path):
        self.file_path = path / "user_profile.json"
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self.save({"root": {"name": "User"}, "relationships": {}, "interests": {}, "assets": {}})

    def load(self) -> Dict[str, Any]:
        _default = {"root": {"name": "User"}, "relationships": {}, "interests": {}, "assets": {}}
        with self._lock:
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                # A valid-JSON-but-wrong-TYPE file (e.g. a list or a scalar)
                # would break every caller (data[cat] = {}). Treat it as corrupt.
                if not isinstance(data, dict):
                    raise ValueError(f"profile is a {type(data).__name__}, expected object")
                return data
            except FileNotFoundError:
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

    def save(self, data: Dict[str, Any]):
        with self._lock:
            temp_path = self.file_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(data, indent=2))
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
        pretty_log(
            "Profile Capped",
            f"{cat}.{key} hit the {cap}-value cap; dropped {dropped} oldest "
            f"value(s) to keep the system prompt bounded",
            icon=Icons.USER_ID, level="WARNING",
        )
        return trimmed

    def update(self, category: str, key: str, value: Any):
        with self._lock:
            data = self.load()
            cat = str(category).strip().lower()
            k = str(key).strip().lower()
            v = str(value).strip()

            # Canonicalisation table: the model often passes synonyms. We
            # normalise them once at write time so retrieval is deterministic.
            # IMPORTANT: the return value now SURFACES the rewrite so the
            # caller (and downstream graph publish) can stay in sync. The
            # previous silent rewrite caused confusion when the graph
            # published `vehicle` while the profile stored it as `car`.
            mapping = {
                "wife": ("relationships", "wife"),
                "husband": ("relationships", "husband"),
                "son": ("relationships", "son"),
                "daughter": ("relationships", "daughter"),
                "car": ("assets", "car"),
                "vehicle": ("assets", "car"),
                "science": ("interests", "science"),
                "interest": ("interests", "general")
            }

            original_cat, original_key = cat, k
            if k in mapping:
                cat, target_key = mapping[k]
            else:
                target_key = k

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
                data[cat][target_key] = v
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
            if existing is None or existing == "":
                data[cat][target_key] = v
            elif isinstance(existing, list):
                if v not in [str(x).strip() for x in existing]:
                    data[cat][target_key] = self._bounded(existing + [v], cap,
                                                          cat, target_key)
                # else: duplicate — no-op
            else:
                # Scalar existing value
                existing_str = str(existing).strip()
                if existing_str == v:
                    # No-op: identical fact
                    pass
                else:
                    # Promote to list, dedup, preserve order
                    merged = [existing_str, v]
                    seen = set()
                    deduped = []
                    for item in merged:
                        if item not in seen:
                            seen.add(item)
                            deduped.append(item)
                    data[cat][target_key] = self._bounded(deduped, cap,
                                                          cat, target_key)
            self.save(data)

            if (cat, target_key) != (original_cat, original_key):
                return f"Synchronized: {cat}.{target_key} = {v}  [normalised from {original_cat}.{original_key}]"
            return f"Synchronized: {cat}.{target_key} = {v}"

    def delete(self, category: str, key: str) -> str:
        with self._lock:
            data = self.load()
            cat = str(category).strip().lower()
            k = str(key).strip().lower()

            # Apply the same canonicalization mapping used by update()
            mapping = {
                "wife": ("relationships", "wife"),
                "husband": ("relationships", "husband"),
                "son": ("relationships", "son"),
                "daughter": ("relationships", "daughter"),
                "car": ("assets", "car"),
                "vehicle": ("assets", "car"),
                "science": ("interests", "science"),
                "interest": ("interests", "general")
            }
            if k in mapping:
                cat, k = mapping[k]

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
            data = self.load()
            cat = str(category).strip().lower()
            k = str(key).strip().lower()

            # Same canonicalisation table as update()/delete() so the value
            # sweep lands on the row the writer actually created.
            mapping = {
                "wife": ("relationships", "wife"),
                "husband": ("relationships", "husband"),
                "son": ("relationships", "son"),
                "daughter": ("relationships", "daughter"),
                "car": ("assets", "car"),
                "vehicle": ("assets", "car"),
                "science": ("interests", "science"),
                "interest": ("interests", "general"),
            }
            if k in mapping:
                cat, k = mapping[k]

            if cat not in data or k not in data[cat]:
                return f"Profile key not found: {cat}.{k}"

            target_lc = str(target).strip().lower()
            if not target_lc:
                return "Profile: empty target, nothing pruned."

            def _mentions(val) -> bool:
                v = str(val).lower()
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
                    return f"Pruned {len(removed)} value(s) from {cat}.{k}: {removed}"
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
        # Load is thread-safe now
        data = self.load()
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
                        sub_v_str = ", ".join(str(i) for i in sub_v)
                    else:
                        sub_v_str = str(sub_v)
                    lines.append(f"- {sub_k}: {sub_v_str}")
            elif isinstance(val, list):
                lines.append(f"## {label}: " + ", ".join([str(i) for i in val]))
            else:
                lines.append(f"{label}: {val}")
        return "\n".join(lines)