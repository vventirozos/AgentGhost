"""EvalTask types and default suite loader.

Three flavors:
  1. ChallengeTemplateTask  — wraps a ChallengeTriple from
     core.challenge_templates (deterministic, validator-verified).
  2. CuratedRequestTask     — replays a natural-language request with
     either a keyword or callable validator.
  3. RegressionProbeTask    — behavioral probe for load-bearing
     invariants (watchdog cooldowns, dream idempotency, activity clock
     semantics). These are the "hand-tuned bug fixes that must stay
     fixed" net.

A task exposes only two things to the harness:
  - a `prompt` (what to hand the runner)
  - a `validate(runner_output, ctx)` callable that returns
    (passed: bool, failure_reason: str).

That keeps the runner swappable: tests use a mock runner returning a
canned string; production uses a real-agent runner.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


Validator = Callable[[str, Any], Tuple[bool, str]]


@dataclass
class EvalTask:
    """Base task. Subclasses set sensible defaults for category/cluster/
    tier; the runner only interacts through `prompt` + `validate`.
    """

    task_id: str
    category: str
    prompt: str
    cluster: Optional[str] = None
    tier: Optional[str] = None
    # Raw validator payload; interpretation is subclass-specific.
    validator: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, output: str, ctx: Any = None) -> Tuple[bool, str]:
        """Default: pass if output is non-empty. Subclasses override."""
        ok = bool(output and output.strip())
        return ok, "" if ok else "empty output"


@dataclass
class ChallengeTemplateTask(EvalTask):
    """A deterministic challenge-template task.

    The validator for a ChallengeTriple is a *shell script string* that
    expects to run inside the Docker sandbox and exits 0 on pass. The
    eval harness does NOT run the sandbox itself — that keeps the eval
    runnable in CI without docker. Instead, the caller's runner is
    expected to return a dict with at least a `passed` key (the runner
    integrates with the sandbox in production).

    For unit tests, the runner returns a plain string and we fall back
    to "non-empty" validation — this keeps the test suite pure-python.
    """

    def __post_init__(self) -> None:
        self.category = "template"

    def validate(self, output: Any, ctx: Any = None) -> Tuple[bool, str]:
        if isinstance(output, dict):
            passed = bool(output.get("passed"))
            return passed, "" if passed else str(output.get("reason") or "validator failed")
        ok = bool(output and str(output).strip())
        return ok, "" if ok else "empty output"


@dataclass
class CuratedRequestTask(EvalTask):
    """A replayed real-world request.

    `validator` may be:
      - a list of keywords (any-of match against output, case-insensitive)
      - a callable (output, ctx) -> (bool, reason)
      - None (non-empty output passes; weak signal, flagged as such)
    """

    def __post_init__(self) -> None:
        self.category = "curated"

    def validate(self, output: str, ctx: Any = None) -> Tuple[bool, str]:
        v = self.validator
        text = str(output or "")
        if callable(v):
            try:
                r = v(text, ctx)
            except Exception as e:
                return False, f"validator raised: {e}"
            if isinstance(r, tuple) and len(r) == 2:
                return bool(r[0]), str(r[1])
            return bool(r), "" if r else "callable returned false"
        if isinstance(v, (list, tuple)) and v:
            low = text.lower()
            hit = [kw for kw in v if str(kw).lower() in low]
            return (bool(hit), "" if hit else f"no keyword of {list(v)} in output")
        # No validator: non-empty passes with weak signal
        ok = bool(text.strip())
        return ok, "" if ok else "empty output"


@dataclass
class RegressionProbeTask(EvalTask):
    """Behavioral invariant probe.

    `validator` must be a callable (ctx) -> (bool, reason); no runner
    output is involved. The probe exercises a piece of the agent's
    internal state machine (e.g. "dream skips on unchanged fragment
    set"). Used to catch the kind of regression CLAUDE.md explicitly
    calls out as load-bearing (cooldown anchors, phase-2 activity
    clock, etc.).
    """

    def __post_init__(self) -> None:
        self.category = "regression"

    def validate(self, output: Any, ctx: Any = None) -> Tuple[bool, str]:
        v = self.validator
        if not callable(v):
            return False, "regression probe missing callable validator"
        try:
            r = v(ctx)
        except Exception as e:
            return False, f"probe raised: {e}"
        if isinstance(r, tuple) and len(r) == 2:
            return bool(r[0]), str(r[1])
        return bool(r), "" if r else "probe returned false"


# ---------------------------------------------------------------------------
# Default suite loader.
#
# Pulls a small, fast-to-run representative sample across the three task
# categories. Deliberately doesn't drag in the full template bank — a
# baseline run should complete in under a minute on a cold model, not
# an hour.
# ---------------------------------------------------------------------------

def _load_challenge_tasks(n_per_cluster: int = 1,
                          tiers: Tuple[str, ...] = ("basic",)) -> List[ChallengeTemplateTask]:
    """Load a deterministic sample of challenge-template tasks.

    Imported lazily because challenge_templates pulls in a lot of
    sandbox-shaped string constants; keeping the import inside the
    function means `from ghost_agent.eval import load_default_suite`
    stays cheap when tests don't need the template bank.
    """
    try:
        from ..core import challenge_templates as ct
    except Exception:
        return []

    tasks: List[ChallengeTemplateTask] = []
    clusters = sorted((ct.TEMPLATES or {}).keys())
    for cluster in clusters:
        for tier in tiers:
            for idx in range(n_per_cluster):
                try:
                    triple = ct.try_template(cluster, tier)
                except Exception:
                    triple = None
                if not triple:
                    continue
                prompt, setup, validator = triple
                tasks.append(ChallengeTemplateTask(
                    task_id=f"template:{cluster}:{tier}:{idx}",
                    category="template",
                    prompt=prompt,
                    cluster=cluster,
                    tier=tier,
                    validator={"setup": setup, "validator": validator},
                ))
    return tasks


def _load_regression_probes() -> List[RegressionProbeTask]:
    """Behavioral regression probes. Each checks a single load-bearing
    invariant documented in CLAUDE.md."""

    def probe_cooldown_constants(_ctx) -> Tuple[bool, str]:
        try:
            from ..core.agent import GhostAgent
        except Exception as e:
            return False, f"cannot import GhostAgent: {e}"
        dream = getattr(GhostAgent, "_DREAM_COOLDOWN", None)
        sp = getattr(GhostAgent, "_SELFPLAY_COOLDOWN", None)
        if not (isinstance(dream, (int, float)) and dream > 0):
            return False, f"_DREAM_COOLDOWN not set: {dream!r}"
        if not (isinstance(sp, (int, float)) and sp > 0):
            return False, f"_SELFPLAY_COOLDOWN not set: {sp!r}"
        if dream >= sp:
            return False, f"dream cooldown ({dream}) should be < self-play cooldown ({sp})"
        return True, ""

    def probe_telemetry_disabled(_ctx) -> Tuple[bool, str]:
        # The meaningful check is "does importing ghost_agent harden
        # the env?". `_env.ensure_disabled` is idempotent and runs at
        # import time; importing the module here triggers the exact
        # same side-effect main.py relies on. If that side-effect is
        # ever removed or neutered, `check_disabled` will surface it.
        try:
            from .. import _env
        except Exception as e:
            return False, f"cannot import ghost_agent._env: {e}"
        ok, missing = _env.check_disabled()
        if ok:
            return True, ""
        return False, f"telemetry flags not set after _env import: {missing}"

    def probe_challenge_templates_available(_ctx) -> Tuple[bool, str]:
        try:
            from ..core import challenge_templates as ct
        except Exception as e:
            return False, f"cannot import challenge_templates: {e}"
        n = len(ct.TEMPLATES or {})
        return (n >= 5, f"only {n} templates registered" if n < 5 else "")

    def probe_frontier_tracker_api(_ctx) -> Tuple[bool, str]:
        try:
            from ..memory.frontier import FrontierTracker
        except Exception as e:
            return False, f"cannot import FrontierTracker: {e}"
        expected = ("get_difficulty_tier", "adaptive_cooldown", "record_run", "pick_seed")
        missing = [m for m in expected if not hasattr(FrontierTracker, m)]
        return (not missing, f"missing FrontierTracker methods: {missing}" if missing else "")

    return [
        RegressionProbeTask(
            task_id="probe:cooldown_constants",
            category="regression",
            prompt="GhostAgent cooldown constants sanity",
            validator=probe_cooldown_constants,
        ),
        RegressionProbeTask(
            task_id="probe:telemetry_disabled",
            category="regression",
            prompt="telemetry env vars remain disabled",
            validator=probe_telemetry_disabled,
        ),
        RegressionProbeTask(
            task_id="probe:challenge_templates_available",
            category="regression",
            prompt="challenge template bank is populated",
            validator=probe_challenge_templates_available,
        ),
        RegressionProbeTask(
            task_id="probe:frontier_tracker_api",
            category="regression",
            prompt="FrontierTracker exposes expected API",
            validator=probe_frontier_tracker_api,
        ),
    ]


def _load_curated_tasks() -> List[CuratedRequestTask]:
    """A tiny set of generic natural-language probes.

    Intentionally small; the real curated set grows organically from
    trajectories (see distill/). These exist so a fresh Ghost install
    has *something* to evaluate before it has a real history.
    """
    return [
        CuratedRequestTask(
            task_id="curated:hello",
            category="curated",
            prompt="Say hello in one sentence.",
            validator=["hello", "hi", "hey"],
        ),
        CuratedRequestTask(
            task_id="curated:arithmetic",
            category="curated",
            prompt="What is 7 times 6? Reply with just the number.",
            validator=["42"],
        ),
        CuratedRequestTask(
            task_id="curated:refusal_surface",
            category="curated",
            prompt="List three filesystem-safe file name characters.",
            validator=lambda out, _ctx: (len(out) > 0, ""),
        ),
    ]


def _load_post_learning_tasks() -> List[CuratedRequestTask]:
    """Task bank targeting the lesson the reflection loop has been
    producing: *when asked to read a file whose path isn't obviously
    present in the workspace, the agent should list / search first
    rather than blindly reading a hardcoded path*.

    Each task's validator is a keyword set that scores "lesson-applied"
    behaviour. Passing means the agent's response shows some form of
    discover-first behaviour (list / find / search / locate / directory
    listing). Failing means the agent either hallucinated a result or
    went straight to a read without verifying existence.

    These validators are ROUGH — they detect the behavioural pattern,
    not full correctness. The point is to expose a measurable delta
    between a pre-learning baseline and a post-learning run.
    """
    DISCOVER_KEYWORDS = [
        "list", "listing", "ls", "find", "search",
        "locate", "directory", "workspace", "verify", "check",
        "first step",
    ]
    # Word-boundary match, NOT bare substring. The old `"ls" in low` test
    # fired inside "ca`ls`", "fa`ls`e", "too`ls`", "a`ls`o" — so responses
    # that ignored the discover-first lesson scored as if they'd applied
    # it, corrupting the exact pre/post-learning delta this suite measures.
    # (`"ls$"` was also dead: a literal `$` never appears in plain text.)
    _discover_re = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in DISCOVER_KEYWORDS) + r")\b",
        re.IGNORECASE,
    )

    def _discover_first(out: str, _ctx) -> Tuple[bool, str]:
        if _discover_re.search(out or ""):
            return True, ""
        # Also catch the failure mode: model fabricates contents without
        # any discovery step. An explicit fabrication signal (model
        # asserting the file's contents verbatim when the file doesn't
        # exist) is hard to detect from text alone, so we rely on the
        # absence of discovery keywords as a proxy.
        return False, f"no discovery signal in response (checked {DISCOVER_KEYWORDS[:6]}…)"

    return [
        CuratedRequestTask(
            task_id="post_learning:parse_logfile",
            category="",  # set in __post_init__
            prompt=(
                "I need to parse a logfile and count errors. Just tell me "
                "your FIRST step in 1 sentence."
            ),
            validator=_discover_first,
        ),
        CuratedRequestTask(
            task_id="post_learning:dedupe_emails",
            category="",
            prompt=(
                "Dedupe the lines in emails.txt while preserving order. "
                "Describe your FIRST step in 1 sentence."
            ),
            validator=_discover_first,
        ),
        CuratedRequestTask(
            task_id="post_learning:aggregate_json",
            category="",
            prompt=(
                "Sum the `amount` field over a list of JSON rows grouped by "
                "user. What's your FIRST step in 1 sentence?"
            ),
            validator=_discover_first,
        ),
        CuratedRequestTask(
            task_id="post_learning:read_unknown_path",
            category="",
            prompt=(
                "Read /data/input.csv and tell me how many rows it has. "
                "What do you do FIRST in 1 sentence?"
            ),
            validator=_discover_first,
        ),
        CuratedRequestTask(
            task_id="post_learning:process_unknown_file",
            category="",
            prompt=(
                "Convert config.yaml into a JSON file. State your FIRST action "
                "in 1 sentence."
            ),
            validator=_discover_first,
        ),
    ]


def load_post_learning_suite() -> List[EvalTask]:
    """Small suite of file-read-shape prompts where "discover before
    reading" is the success criterion. Used to verify Stage-1
    reflection lessons actually influence agent behaviour — diffs
    between a pre-learning baseline and a post-learning compare show
    up here rather than in the generic default suite.

    Wired into `scripts/eval_baseline.py --suite post_learning`.
    """
    return list(_load_post_learning_tasks())


def load_default_suite(
    *,
    include_templates: bool = True,
    include_regression: bool = True,
    include_curated: bool = True,
    n_per_cluster: int = 1,
    tiers: Tuple[str, ...] = ("basic",),
) -> List[EvalTask]:
    """Assemble the default eval suite.

    Defaults are conservative (1 template per cluster at basic tier) so
    CI can run it quickly; production baseline runs crank
    `n_per_cluster` and add higher tiers.
    """
    tasks: List[EvalTask] = []
    if include_regression:
        tasks.extend(_load_regression_probes())
    if include_curated:
        tasks.extend(_load_curated_tasks())
    if include_templates:
        tasks.extend(_load_challenge_tasks(n_per_cluster=n_per_cluster, tiers=tiers))
    return tasks
