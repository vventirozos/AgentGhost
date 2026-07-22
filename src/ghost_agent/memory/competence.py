"""Per-domain capability profile — roadmap phase 2.3.

The doc's Level-1 (Cross-Domain Transfer) gap is that the agent has
no internal record of *which* operational domains it is good at. A
calibrated confidence score that ignores this prior treats every
domain the same — wrong, because the same model is materially better
at shell scripting than it is at PostgreSQL DDL.

``CompetenceProfile`` is a thin Beta-prior estimator keyed by
``(domain, tool)``. After each tool outcome the caller records
success/failure; ``estimate(domain, tool=None)`` returns the
posterior mean p(success) which feeds the composite-confidence
calculation in ``core.confidence``.

Storage: a single JSON file under the agent's memory directory.
We deliberately do not reuse the SQLite project store here — the
competence profile is per-agent (not per-project) and changes too
often for transactional storage to be worth it. The atomic-rename
write keeps the file consistent under concurrent flushes.

Priors: ``alpha=1, beta=1`` (uniform Beta(1,1)). After 100 samples
the prior contributes <2 % of the posterior mass, so it's only
load-bearing during the cold-start window — which is exactly when
we want it (a fresh agent should not radiate overconfidence in any
domain).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")


# Canonical domain taxonomy. Callers should pass these strings (not
# raw tool names) so the profile aggregates cleanly across tools that
# operate in the same domain. Unknown domain → "other" bucket.
KNOWN_DOMAINS: Tuple[str, ...] = (
    "shell", "sql", "fetch", "code", "fs", "memory", "vision", "other",
)


@dataclass
class _Cell:
    """One (domain, tool) bucket. ``alpha-1`` = successes, ``beta-1`` =
    failures, so ``mean = alpha/(alpha+beta)``.

    ``samples`` counts RECORDED OUTCOMES, independently of their Beta
    weight. Deriving the count from the mass — ``int((alpha-1)+(beta-1))``
    — truncates: ten records at ``weight=0.05`` add 0.5 of mass and floor
    to n=0, so the cell stayed invisible to the ``n >= 1`` gate in
    ``estimate()`` and reported ``n=0`` in the prompt for ever. The
    derived value is still used as the fallback when loading a legacy
    file that predates this field.
    """

    alpha: float = 1.0
    beta: float = 1.0
    samples: int = 0

    @property
    def n(self) -> int:
        if self.samples:
            return int(self.samples)
        # Legacy cell (no persisted sample count): fall back to the mass.
        return max(0, int((self.alpha - 1.0) + (self.beta - 1.0)))

    @property
    def mean(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0.0:
            return 0.5
        return self.alpha / denom

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        self.samples += 1


class CompetenceProfile:
    """In-memory + JSON-backed capability map.

    Reads are lock-free fast-path (a single dict lookup); writes acquire
    an RLock and flush to disk under that lock. The flush is
    atomic-rename so a crash mid-write never leaves a half-written file.
    """

    FILE_NAME = "competence_profile.json"

    def __init__(self, memory_dir: Path, flush_interval: float = 0.0):
        """``flush_interval`` > 0 debounces the disk write: ``record()``
        marks the profile dirty and flushes at most once per that many
        seconds (plus whenever ``_FLUSH_EVERY`` records have piled up, on
        ``reset()``, and on an explicit ``flush()``).

        The DEFAULT is 0.0 = write-through, i.e. the historical behaviour,
        because the profile's durability contract is "an outcome recorded
        is an outcome persisted" and the documented deploy procedure is a
        plain ``kill`` (no atexit, no signal handler) — a debounce would
        silently drop the tail. Metacog, which calls ``record()`` once per
        tool outcome, can opt in when the write volume actually matters.
        """
        self.file_path = Path(memory_dir) / self.FILE_NAME
        self._lock = threading.RLock()
        # cells keyed by (domain, tool). domain "*" is the per-domain
        # aggregate; tool "*" inside a non-"*" domain is the per-domain
        # roll-up. Concrete (domain, tool) cells are leaves.
        self._cells: Dict[Tuple[str, str], _Cell] = {}
        # Set when the profile file is PRESENT but unreadable — see _load().
        self._degraded = False
        self._flush_interval = max(0.0, float(flush_interval or 0.0))
        self._dirty = 0
        self._last_flush = time.monotonic()
        self._load()

    # Force a flush after this many debounced records regardless of time.
    _FLUSH_EVERY = 10

    # ----------------------------------------------------------- public

    def record(self, domain: str, tool: str, success: bool, *,
               weight: float = 1.0) -> None:
        """Record one outcome. ``tool`` may be empty when the caller
        only knows the domain; both the leaf and the per-domain
        roll-up cells are updated either way so estimates stay coherent.
        """
        d = self._canonical_domain(domain)
        t = (tool or "*").strip().lower() or "*"
        with self._lock:
            self._cell(d, t).update(success, weight=weight)
            if t != "*":
                self._cell(d, "*").update(success, weight=weight)
            self._cell("*", "*").update(success, weight=weight)
            self._dirty += 1
            if self._should_flush():
                self.flush()

    def _should_flush(self) -> bool:
        if self._flush_interval <= 0.0:
            return True
        if self._dirty >= self._FLUSH_EVERY:
            return True
        return (time.monotonic() - self._last_flush) >= self._flush_interval

    def flush(self) -> None:
        """Persist any debounced records. No-op when already clean."""
        with self._lock:
            if not self._dirty:
                return
            self._save()
            self._dirty = 0
            self._last_flush = time.monotonic()

    def estimate(self, domain: str, tool: Optional[str] = None) -> float:
        """Return the posterior mean p(success) for ``(domain, tool)``.

        Tool fallback: when the leaf cell has no observations, falls
        back to the per-domain roll-up. When THAT has no observations
        either, falls back to the global roll-up — never returns a
        prior-only point estimate without falling back.
        """
        d = self._canonical_domain(domain)
        t = (tool or "*").strip().lower() or "*"
        with self._lock:
            cell = self._cells.get((d, t))
            if cell is not None and cell.n >= 1:
                return cell.mean
            if t != "*":
                roll = self._cells.get((d, "*"))
                if roll is not None and roll.n >= 1:
                    return roll.mean
            glob = self._cells.get(("*", "*"))
            if glob is not None and glob.n >= 1:
                return glob.mean
            return 0.5  # neutral prior

    def observations(self, domain: str, tool: Optional[str] = None) -> int:
        """Number of recorded outcomes for ``(domain, tool)``."""
        d = self._canonical_domain(domain)
        t = (tool or "*").strip().lower() or "*"
        cell = self._cells.get((d, t))
        return 0 if cell is None else cell.n

    def by_domain(self) -> Dict[str, Tuple[float, int]]:
        """Per-domain roll-up: ``{domain: (p_success, n)}``."""
        out: Dict[str, Tuple[float, int]] = {}
        with self._lock:
            for (d, t), cell in self._cells.items():
                if t != "*" or d == "*":
                    continue
                out[d] = (cell.mean, cell.n)
        return out

    def reset(self) -> None:
        with self._lock:
            self._cells.clear()
            self._dirty = 0
            self._save()
            self._last_flush = time.monotonic()

    def get_context_string(self) -> str:
        """Prompt-renderable summary, used by the planner to know which
        domains the agent's track record supports."""
        roll = self.by_domain()
        if not roll:
            return ""
        lines = ["### Competence (per-domain p(success), n):"]
        for d in sorted(roll, key=lambda k: roll[k][0]):
            mean, n = roll[d]
            lines.append(f"  - {d}: {mean:.0%} (n={n})")
        return "\n".join(lines)

    # ---------------------------------------------------------- helpers

    def _cell(self, d: str, t: str) -> _Cell:
        key = (d, t)
        cell = self._cells.get(key)
        if cell is None:
            cell = _Cell()
            self._cells[key] = cell
        return cell

    @staticmethod
    def _canonical_domain(domain: str) -> str:
        d = (domain or "").strip().lower()
        if d in KNOWN_DOMAINS:
            return d
        # Common synonyms / convenience aliases — keep small so
        # downstream consumers can pattern-match without surprises.
        if d in ("bash", "zsh", "terminal", "command"):
            return "shell"
        if d in ("postgres", "postgresql", "mysql", "sqlite", "database"):
            return "sql"
        if d in ("http", "web", "url", "browse"):
            return "fetch"
        if d in ("python", "javascript", "ts", "rust", "go"):
            return "code"
        if d in ("filesystem", "file", "directory"):
            return "fs"
        return "other"

    # ------------------------------------------------------- persistence

    def _quarantine_corrupt(self, why: str) -> None:
        """Corrupt profile: preserve the raw file as a timestamped sidecar
        BEFORE the next _save() overwrites it, then start empty. Same
        policy as journal.py / profile.py."""
        try:
            sidecar = self.file_path.with_suffix(f".corrupt-{int(time.time())}.json")
            os.replace(self.file_path, sidecar)
            logger.warning(
                "competence_profile.json was corrupt (%s); preserved to %s and "
                "started a fresh profile.", why, sidecar.name,
            )
        except Exception:
            logger.warning(
                "competence_profile.json was corrupt (%s) and could not be "
                "preserved; starting a fresh profile.", why,
            )

    def _load(self) -> None:
        """Populate ``_cells`` from disk.

        Three outcomes, deliberately distinguished. The old
        ``except Exception: logger.debug(...); return`` treated an
        UNREADABLE file exactly like an ABSENT one, so the very next
        ``record()`` atomically overwrote a profile with hundreds of
        observations with a single-tool file — total history loss, zero
        sidecars, and the only trace was a debug line.
        """
        try:
            content = self.file_path.read_text()
        except FileNotFoundError:
            return
        except UnicodeDecodeError:
            self._quarantine_corrupt("undecodable bytes")
            return
        except OSError as exc:
            # Present but unreadable → fail CLOSED: keep serving the
            # neutral prior in memory, but never write over the file.
            self._degraded = True
            logger.error(
                "competence_profile.json is present but unreadable (%s: %s). "
                "Serving neutral priors and REFUSING to overwrite the on-disk "
                "profile until it can be read.",
                type(exc).__name__, exc,
            )
            return
        if not content.strip():
            return
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(
                    f"profile is a {type(data).__name__}, expected object")
        except Exception as exc:
            self._quarantine_corrupt(f"{type(exc).__name__}: {exc}")
            return
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            try:
                d, t = k.split("|", 1)
            except ValueError:
                continue
            try:
                alpha = float(v.get("alpha", 1.0))
                beta = float(v.get("beta", 1.0))
                samples = int(v.get("n", 0) or 0)
            except (TypeError, ValueError):
                continue
            self._cells[(d, t)] = _Cell(alpha=alpha, beta=beta, samples=samples)

    def _save(self) -> None:
        if self._degraded:
            # See _load(): the file exists but could not be read.
            logger.warning(
                "Competence profile save skipped: %s is unreadable and must "
                "not be overwritten.", self.file_path.name,
            )
            return
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                f"{d}|{t}": {"alpha": cell.alpha, "beta": cell.beta,
                             "n": cell.n}
                for (d, t), cell in self._cells.items()
            }
            tmp = self.file_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.file_path)
        except Exception as exc:
            logger.warning("CompetenceProfile save failed: %s", exc)


__all__ = ["CompetenceProfile", "KNOWN_DOMAINS"]
