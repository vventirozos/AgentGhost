import json
import threading
import os
from pathlib import Path
from typing import Any, Dict
from ..utils.logging import pretty_log

class ProfileMemory:
    def __init__(self, path: Path):
        self.file_path = path / "user_profile.json"
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self.save({"root": {"name": "User"}, "relationships": {}, "interests": {}, "assets": {}})

    def load(self) -> Dict[str, Any]:
        with self._lock:
            try: 
                return json.loads(self.file_path.read_text())
            except: 
                return {"root": {"name": "User"}, "relationships": {}, "interests": {}, "assets": {}}

    def save(self, data: Dict[str, Any]):
        with self._lock:
            temp_path = self.file_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(data, indent=2))
            os.replace(temp_path, self.file_path)

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
            # below applies.
            _SINGLETON_KEYS = {
                "name", "role", "email", "timezone", "age",
                "birthday", "pronouns", "title", "location",
            }
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
            existing = data[cat].get(target_key)
            if existing is None or existing == "":
                data[cat][target_key] = v
            elif isinstance(existing, list):
                if v not in [str(x).strip() for x in existing]:
                    data[cat][target_key] = existing + [v]
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
                    data[cat][target_key] = deduped
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