import hashlib
import json
import logging
import os
import re
import random
import threading
from pathlib import Path
from datetime import datetime

# Cross-process advisory lock for the frontier JSON. Slack + web + CLI can
# run in parallel and update self_play_frontier.json concurrently; the
# thread-lock alone only serialises intra-process writes. `fcntl` is
# POSIX-only — on platforms without it we degrade gracefully to the
# threading lock (rare for this deployment; all supported hosts are Linux
# or macOS).
try:
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows fallback
    _HAS_FCNTL = False

logger = logging.getLogger("GhostAgent")


CLUSTER_KEYWORDS = [
    ("sql", [r"\bselect\b", r"\bjoin\b", r"\bwindow function", r"\bcte\b", r"\bgroup by\b", r"\bsql\b"]),
    ("bash", [r"\bbash\b", r"\bshell script", r"\bawk\b", r"\bsed\b", r"\bgrep\b", r"\bxargs\b"]),
    ("data_analysis", [r"\bpandas\b", r"\bdataframe\b", r"\bcsv\b", r"\bnumpy\b", r"\banalysis\b", r"\bstatistic"]),
    ("algo", [r"\balgorithm", r"\bgraph\b", r"\btree\b", r"\bdynamic programming", r"\brecurs", r"\bcomplexity\b"]),
    ("concurrency", [r"\basync", r"\bthread", r"\bconcurren", r"\bprocess pool"]),
    ("regex_parse", [r"\bregex\b", r"\bparse\b", r"\bparser\b", r"\btoken"]),
]


def classify_cluster(text: str) -> str:
    """Cheap keyword classifier. Returns a stable cluster key for a challenge."""
    if not text:
        return "python_general"
    lowered = text.lower()
    for key, patterns in CLUSTER_KEYWORDS:
        for pat in patterns:
            if re.search(pat, lowered):
                return key
    return "python_general"


DIFFICULTY_TIERS = ["basic", "intermediate", "advanced", "expert"]

# Tier-specific hints injected into the challenge generation prompt
DIFFICULTY_HINTS = {
    "basic": "Generate a BASIC challenge: simple input parsing, one file, straightforward logic. No edge cases.",
    "intermediate": "Generate an INTERMEDIATE challenge: multiple files or data sources, some error handling needed, moderate complexity.",
    "advanced": "Generate an ADVANCED challenge: complex data transformations, multiple edge cases, performance considerations, multi-step reasoning.",
    "expert": "Generate an EXPERT challenge: requires sophisticated algorithms, concurrent/async patterns, or multi-system integration. Push the agent's limits.",
}

# Mastered challenges required per tier to unlock next
TIER_UNLOCK_THRESHOLD = 3


class FrontierTracker:
    """Tracks self-play compression progress per cluster. Drives curiosity:
    which clusters are the learning frontier, how quickly descriptions are
    shrinking, and when a cluster is mastered or stuck.

    Now includes difficulty tier scaffolding: the agent starts with basic
    challenges and unlocks harder tiers as it masters each level."""

    MAX_RUNS = 200
    # Mastery requires a run of first-try wins long enough that random noise
    # is unlikely to produce it. The previous threshold of 3 let a brand-new
    # cluster self-master after 3 first-try wins with delta=0 (which is the
    # guaranteed delta on the first run) — wildly over-confident. 5 runs
    # with at least one positive compression delta is a much stronger
    # signal: both length & streak count, not just streak.
    MASTERED_STREAK = 5
    BRITTLE_WINDOW = 3
    # Saturation: last N runs in the cluster are all first-try-passes with
    # effectively zero compression gain. The cluster is "done" for the
    # current template bank — continuing to target it burns cycles on
    # material the agent already aces. `pick_seed` skips saturated
    # clusters and falls back to exploration so the loop actually
    # exercises new material. 3 is tight enough to react fast, loose
    # enough not to misfire on a single fluke run.
    SATURATION_WINDOW = 3
    SATURATION_DELTA_EPSILON = 0.001
    # Recent-challenge dedup window. A repeated prompt inflates mastery
    # counters if we record it as a fresh data point, so we refuse to
    # record runs whose challenge hash matches one of the last N.
    DEDUP_WINDOW = 20

    def __init__(self, memory_dir: Path):
        self.file_path = Path(memory_dir) / "self_play_frontier.json"
        self._lock = threading.RLock()
        # `_lock_path` is a sibling file used purely for cross-process
        # `fcntl.flock`. Keeping it separate from the data file means the
        # atomic `os.replace` in `_save` can't race with lock acquisition.
        self._lock_path = self.file_path.with_suffix(".lock")
        if not self.file_path.exists():
            self._save({"runs": [], "clusters": {}})

    def _crossproc_lock(self):
        """Context manager that acquires an fcntl advisory lock across
        processes. No-op fallback on platforms without fcntl."""
        tracker = self

        class _Ctx:
            def __enter__(self_inner):
                if not _HAS_FCNTL:
                    self_inner._fh = None
                    return
                # Open in append mode so we never truncate. The lock is on
                # the fd, not the file contents.
                self_inner._fh = open(tracker._lock_path, "a+")
                try:
                    fcntl.flock(self_inner._fh.fileno(), fcntl.LOCK_EX)
                except OSError:
                    # Lock acquisition failed — degrade to in-process lock.
                    self_inner._fh.close()
                    self_inner._fh = None

            def __exit__(self_inner, exc_type, exc, tb):
                fh = getattr(self_inner, "_fh", None)
                if fh is None:
                    return
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

        return _Ctx()

    def _load(self) -> dict:
        try:
            return json.loads(self.file_path.read_text())
        except Exception:
            return {"runs": [], "clusters": {}}

    def _save(self, state: dict):
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.file_path)

    @staticmethod
    def _hash_challenge(challenge: str) -> str:
        """Stable short hash of a challenge prompt for dedup checks."""
        if not challenge:
            return ""
        return hashlib.sha1(challenge.strip().encode("utf-8")).hexdigest()[:16]

    def get_cluster_stats(self, cluster_key: str) -> dict:
        with self._lock:
            return self._load().get("clusters", {}).get(cluster_key, {})

    @classmethod
    def _cluster_is_saturated(cls, stats: dict) -> bool:
        """True when the last `SATURATION_WINDOW` runs in this cluster are
        all first-try passes with near-zero compression delta — i.e. the
        agent is solving every template we throw at it instantly, and the
        cluster is producing no new learning signal.

        Skipped clusters fall back to the exploration / cold-start path,
        which picks a random template from a DIFFERENT cluster so the
        loop stops burning cycles on saturated material.
        """
        recent = stats.get("recent_outcomes", [])[-cls.SATURATION_WINDOW:]
        if len(recent) < cls.SATURATION_WINDOW:
            return False
        for r in recent:
            if not r.get("passed"):
                return False
            if int(r.get("attempts_used", 1)) != 1:
                return False
            if float(r.get("delta", 0.0)) > cls.SATURATION_DELTA_EPSILON:
                return False
        return True

    def list_saturated_clusters(self) -> list:
        """Return cluster keys whose recent outcomes indicate saturation.
        Exposed so the scheduler / loop log can explain rotation."""
        with self._lock:
            state = self._load()
        return [
            k for k, s in state.get("clusters", {}).items()
            if self._cluster_is_saturated(s)
        ]

    def _get_brittle_clusters_scored(self, limit: int = 3) -> list:
        """Return the top `limit` brittle clusters as `(score, key, stats)`
        tuples. Exposed separately so `pick_seed` can do weighted sampling
        by score; `get_brittle_clusters` wraps it into the historical
        `(key, stats)` shape expected by callers."""
        with self._lock:
            state = self._load()
        out = []
        for key, stats in state.get("clusters", {}).items():
            if stats.get("mastered"):
                continue
            # Saturated clusters stop producing learning signal — skip
            # them so `pick_seed` rotates to something less boring.
            if self._cluster_is_saturated(stats):
                continue
            recent = stats.get("recent_outcomes", [])[-self.BRITTLE_WINDOW:]
            if len(recent) < 1:
                continue
            failures = sum(1 for r in recent if not r.get("passed"))
            hard_wins = sum(
                1 for r in recent if r.get("passed") and r.get("attempts_used", 1) >= 3
            )
            soft_wins = sum(
                1 for r in recent if r.get("passed") and r.get("attempts_used", 1) == 2
            )
            # Failure is the strongest signal (weight 2), burning all 3
            # retries is medium (weight 2), a 2-attempt struggle is a soft
            # signal (weight 1). This means after a single struggled-then-
            # won run the cluster already registers as mildly brittle, so
            # the next self-play session targets it instead of cold-starting.
            score = failures * 2 + hard_wins * 2 + soft_wins * 1
            if score > 0:
                out.append((score, key, stats))
        out.sort(key=lambda x: -x[0])
        return out[:limit]

    def get_brittle_clusters(self, limit: int = 3) -> list:
        """Clusters where the agent keeps failing, burning all 3 retries, or
        even struggling through 2 retries to succeed. A struggled-then-won
        run is a weaker but still real signal that the cluster is weak."""
        return [(k, s) for _, k, s in self._get_brittle_clusters_scored(limit)]

    def get_difficulty_tier(self, cluster_key: str) -> str:
        """Determine the appropriate difficulty tier for a cluster.

        Monotonic: once a tier is unlocked for a cluster, it stays unlocked.
        The previous version derived the tier from the rolling 10-outcome
        buffer and silently downgraded when old first-try wins scrolled
        out — a spurious regression purely from noise. Now we store
        `unlocked_tier_index` on the cluster and only ever move it upward.
        """
        with self._lock:
            state = self._load()
        cluster = state.get("clusters", {}).get(cluster_key, {})
        if not cluster:
            return DIFFICULTY_TIERS[0]  # New cluster → basic

        stored_idx = cluster.get("unlocked_tier_index")
        if isinstance(stored_idx, int):
            return DIFFICULTY_TIERS[max(0, min(stored_idx, len(DIFFICULTY_TIERS) - 1))]

        # Migration path for clusters written before `unlocked_tier_index`
        # existed: derive once from the cumulative first-try-win counter.
        total_wins = int(cluster.get("total_first_try_wins", 0))
        if total_wins == 0:
            recent = cluster.get("recent_outcomes", [])
            total_wins = sum(
                1 for r in recent
                if r.get("passed") and r.get("attempts_used", 1) == 1
            )
        tier_index = min(total_wins // TIER_UNLOCK_THRESHOLD, len(DIFFICULTY_TIERS) - 1)
        return DIFFICULTY_TIERS[tier_index]

    def get_difficulty_hint(self, cluster_key: str) -> str:
        """Return the difficulty hint for the given cluster's current tier."""
        tier = self.get_difficulty_tier(cluster_key)
        return DIFFICULTY_HINTS.get(tier, DIFFICULTY_HINTS["basic"])

    def pick_seed(self, random_explore_prob: float = 0.2) -> dict:
        """Pick a frontier cluster to target the next challenge at.

        Returns a dict: {mode, cluster_key, hint} where mode is 'frontier',
        'exploration', or 'cold_start'. Callers inject hint into the prompt.

        Among brittle clusters, we sample weighted-proportionally to
        brittleness score rather than always picking the top. Always
        picking brittle[0] meant 80% of self-play runs re-targeted the
        SAME cluster and never diversified to the secondary frontiers.
        """
        if random.random() < random_explore_prob:
            return {"mode": "exploration", "cluster_key": None, "hint": ""}

        scored = self._get_brittle_clusters_scored(limit=3)
        if not scored:
            # If any cluster exists but they're ALL saturated, hint to
            # the caller so it can prefer `pick_random_template` over
            # re-hitting the same cluster template. This is how the loop
            # escapes the "6 concurrency cycles in a row, all pass, zero
            # learning" failure mode.
            saturated = self.list_saturated_clusters()
            if saturated:
                return {
                    "mode": "exploration",
                    "cluster_key": None,
                    "saturated_clusters": saturated,
                    "hint": (
                        "All tracked clusters are saturated "
                        f"({', '.join(saturated)}) — explore a fresh "
                        "challenge shape instead of re-targeting a "
                        "mastered cluster."
                    ),
                }
            return {"mode": "cold_start", "cluster_key": None, "hint": ""}

        total = sum(max(1, s) for s, _, _ in scored)
        r = random.random() * total
        acc = 0.0
        cluster_key, stats = scored[0][1], scored[0][2]
        for score, key, st in scored:
            acc += max(1, score)
            if r <= acc:
                cluster_key, stats = key, st
                break
        last = stats.get("recent_outcomes", [])[-1] if stats.get("recent_outcomes") else {}
        last_mistake = last.get("mistake", "")
        difficulty_hint = self.get_difficulty_hint(cluster_key)
        tier = self.get_difficulty_tier(cluster_key)
        hint_lines = [
            f"FRONTIER TARGET: the agent is currently weak at the '{cluster_key}' cluster.",
            f"Recent runs: {stats.get('runs', 0)} total, best description length {stats.get('best_length', 'n/a')} chars.",
            f"DIFFICULTY TIER: {tier.upper()} — {difficulty_hint}",
        ]
        if last_mistake:
            hint_lines.append(f"Last observed mistake: {last_mistake[:200]}")
        hint_lines.append(
            "Generate a challenge that exercises this cluster in a way that would force the agent "
            "to generalize beyond its last attempt. Do NOT repeat an identical prior challenge."
        )
        return {
            "mode": "frontier",
            "cluster_key": cluster_key,
            "difficulty_tier": tier,
            "hint": "\n".join(hint_lines),
        }

    def record_run(
        self,
        cluster_key: str,
        challenge: str,
        attempts_used: int,
        passed: bool,
        description_length: int,
        mistake: str = "",
    ) -> dict:
        """Append a run, update cluster stats, return a result dict with
        compression_delta, is_new_cluster, and mastered flags.

        Cross-process safe via fcntl advisory lock on a sibling file; the
        threading.RLock still guards intra-process concurrency.
        """
        challenge_hash = self._hash_challenge(challenge or "")
        with self._crossproc_lock():
            with self._lock:
                state = self._load()
                clusters = state.setdefault("clusters", {})
                runs = state.setdefault("runs", [])

                cluster = clusters.setdefault(
                    cluster_key,
                    {
                        "runs": 0,
                        "best_length": None,
                        "last_length": None,
                        "last_compression": 0.0,
                        "mastered": False,
                        "recent_outcomes": [],
                        "recent_hashes": [],
                        "total_first_try_wins": 0,
                        "unlocked_tier_index": 0,
                    },
                )
                is_new_cluster = cluster["runs"] == 0

                # M7 dedup: if we've seen this exact challenge recently,
                # don't record it. Returning a no-op-shaped result keeps
                # callers stable without polluting mastery counters.
                recent_hashes = cluster.get("recent_hashes", [])
                if challenge_hash and challenge_hash in recent_hashes:
                    return {
                        "compression_delta": 0.0,
                        "is_new_cluster": False,
                        "mastered": bool(cluster.get("mastered")),
                        "best_length": cluster.get("best_length"),
                        "runs": cluster["runs"],
                        "duplicate": True,
                    }

                if passed:
                    prev_best = cluster.get("best_length")
                    if prev_best is None or prev_best <= 0:
                        compression_delta = 0.0
                    else:
                        compression_delta = (prev_best - description_length) / prev_best
                    if prev_best is None or description_length < prev_best:
                        cluster["best_length"] = description_length
                else:
                    compression_delta = -1.0

                cluster["runs"] += 1
                cluster["last_length"] = description_length
                cluster["last_compression"] = compression_delta
                cluster["last_cluster_run_at"] = datetime.now().isoformat()
                outcome = {
                    "timestamp": cluster["last_cluster_run_at"],
                    "passed": passed,
                    "attempts_used": attempts_used,
                    "length": description_length,
                    "delta": compression_delta,
                    "mistake": mistake[:300] if mistake else "",
                }
                cluster["recent_outcomes"] = (cluster.get("recent_outcomes", []) + [outcome])[-10:]
                if challenge_hash:
                    cluster["recent_hashes"] = (
                        recent_hashes + [challenge_hash]
                    )[-self.DEDUP_WINDOW:]

                # C7 mastery: require a longer streak AND require that
                # the streak shows real compression progress somewhere
                # (at least one delta > 0.05). A brand-new cluster with
                # delta=0 on every run can no longer self-master in 3
                # trivial first-try wins.
                recent = cluster["recent_outcomes"][-self.MASTERED_STREAK:]
                if len(recent) >= self.MASTERED_STREAK:
                    all_first_try = all(
                        r["passed"] and r["attempts_used"] == 1 for r in recent
                    )
                    any_real_progress = any(r["delta"] > 0.05 for r in recent)
                    cluster["mastered"] = bool(all_first_try and any_real_progress)
                else:
                    cluster["mastered"] = False

                # S2 monotonic tier: keep a cumulative first-try-win
                # counter and a stored tier index that only moves up.
                if passed and attempts_used == 1:
                    cluster["total_first_try_wins"] = int(
                        cluster.get("total_first_try_wins", 0)
                    ) + 1
                computed_tier = min(
                    int(cluster.get("total_first_try_wins", 0)) // TIER_UNLOCK_THRESHOLD,
                    len(DIFFICULTY_TIERS) - 1,
                )
                current_tier = int(cluster.get("unlocked_tier_index", 0))
                cluster["unlocked_tier_index"] = max(current_tier, computed_tier)

                runs.append(
                    {
                        "timestamp": outcome["timestamp"],
                        "cluster_key": cluster_key,
                        "challenge": (challenge or "")[:300],
                        "attempts_used": attempts_used,
                        "passed": passed,
                        "length": description_length,
                        "delta": compression_delta,
                    }
                )
                if len(runs) > self.MAX_RUNS:
                    state["runs"] = runs[-self.MAX_RUNS:]

                self._save(state)

                return {
                    "compression_delta": compression_delta,
                    "is_new_cluster": is_new_cluster,
                    "mastered": cluster["mastered"],
                    "best_length": cluster["best_length"],
                    "runs": cluster["runs"],
                    "duplicate": False,
                }

    def adaptive_cooldown(
        self,
        base: int = 3600,
        floor: int = 600,
        ceiling: int = 7200,
        cluster_key: str = "",
    ) -> int:
        """Shorter cooldown when the last run made compression progress;
        longer when it was wasted. Returns seconds.

        When `cluster_key` is provided, the cooldown is computed from the
        most recent run IN THAT CLUSTER rather than the global tail.
        Without this, cross-cluster churn poisons the signal — e.g., a
        learning streak in cluster A gets the long "wasted" cooldown
        simply because cluster B had the most recent failed run.
        """
        with self._lock:
            state = self._load()
        runs = state.get("runs", [])
        if not runs:
            return base
        if cluster_key:
            cluster_runs = [r for r in runs if r.get("cluster_key") == cluster_key]
            last = cluster_runs[-1] if cluster_runs else runs[-1]
        else:
            last = runs[-1]
        delta = last.get("delta", 0.0)
        passed = last.get("passed", False)
        if passed and delta > 0.05:
            return max(floor, base // 2)
        if passed and last.get("attempts_used", 1) == 1 and delta == 0.0:
            return max(floor, int(base * 0.75))
        if not passed:
            return min(ceiling, base * 2)
        return base
