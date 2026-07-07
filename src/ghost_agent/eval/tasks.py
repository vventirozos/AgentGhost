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
            # The ONLY valid pass signal for a template task is the sandbox
            # verdict. A dict WITHOUT a `passed` key means the runner never
            # executed the shell-script validator (e.g. the http runner) — it
            # must NOT fall through to the "non-empty text" convenience and
            # silently score PASS. Treat a missing verdict as unverified/fail.
            if "passed" not in output:
                return False, "template not validated: runner returned no 'passed' verdict"
            passed = bool(output.get("passed"))
            return passed, "" if passed else str(output.get("reason") or "validator failed")
        ok = bool(output and str(output).strip())
        return ok, "" if ok else "empty output"


@dataclass
class BehavioralTask(EvalTask):
    """An EXECUTION-GROUNDED task: drive the live agent, then verify the REAL
    side-effect (a file written in the sandbox, a fact that recalls, a DB row),
    not the response text.

    This is the DISCRIMINATING half of the harness. The trivial `capability`
    suite is single-turn, zero-tool text Q&A (it scored 1.000 straight through
    five live tool-path bugs). A BehavioralTask instead exercises the tool
    surface and only passes when the real effect is observed.

    `verify` is `async (output_text, ctx) -> (passed, reason)` and lives on the
    task (self-describing); the behavioral runner invokes it and folds the
    verdict into a `{"passed": ...}` dict — the same verdict contract templates
    use. Run under a NON-behavioral runner (stub/http), a BehavioralTask has no
    verdict and correctly scores FAIL ("unverified"), never a false green.
    """

    verify: Optional[Callable[[str, Any], Any]] = None

    def __post_init__(self) -> None:
        self.category = "behavioral"

    def validate(self, output: Any, ctx: Any = None) -> Tuple[bool, str]:
        if isinstance(output, dict):
            if "passed" not in output:
                return False, "behavioral task not verified: runner returned no verdict"
            passed = bool(output.get("passed"))
            return passed, "" if passed else str(
                output.get("failure_reason") or output.get("reason") or "verification failed")
        # A plain string means it ran under a runner that did NOT execute the
        # grounded verify (stub/http) — treat as unverified, never a soft pass.
        return False, "behavioral task requires the behavioral runner (no execution verdict)"


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

    # ── Learning-loop input-integrity invariants ──────────────────────
    # These guard the data that FEEDS the self-improvement loops (outcome
    # labelling, trajectory corpus, PRM training). A regression here is
    # silent — the loops keep running on corrupted signal — so it belongs
    # in the always-on gate, not just pytest.

    def probe_outcome_heuristic_exit_codes(_ctx) -> Tuple[bool, str]:
        try:
            from ..distill.outcome_heuristics import _looks_like_tool_error
        except Exception as e:
            return False, f"cannot import outcome_heuristics: {e}"
        # A non-zero exit code (any, not just 1/2) must read as a failure,
        # else a genuinely-stuck run trains the loops as a success.
        if not _looks_like_tool_error("stdout...\nEXIT CODE: 127\n"):
            return False, "non-zero exit code 127 not detected as tool error"
        if _looks_like_tool_error("EXIT CODE: 0\nall good"):
            return False, "exit code 0 wrongly detected as tool error"
        return True, ""

    def probe_trajectory_schema_drift(_ctx) -> Tuple[bool, str]:
        try:
            from ..distill.schema import Trajectory
        except Exception as e:
            return False, f"cannot import Trajectory: {e}"
        # A record from a NEWER schema (an added field) must still load, or
        # a version skew silently drops EVERY such record from the corpus.
        try:
            t = Trajectory.from_dict({
                "user_request": "hi", "outcome": "passed",
                "a_future_field": 1, "tool_calls": [{"name": "x", "future": 2}],
            })
        except Exception as e:
            return False, f"drift record rejected: {e}"
        if t.user_request != "hi" or t.outcome != "passed":
            return False, "drift record loaded but lost known fields"
        return True, ""

    def probe_prm_skips_junk_outcomes(_ctx) -> Tuple[bool, str]:
        try:
            from ..prm.labels import derive_step_labels
            from ..distill.schema import Trajectory, ToolCall
        except Exception as e:
            return False, f"cannot import prm.labels: {e}"
        junk = Trajectory(user_request="q", outcome="error",
                          tool_calls=[ToolCall(name="a"), ToolCall(name="b")])
        if derive_step_labels(junk) != []:
            return False, "junk outcome 'error' produced PRM labels (should skip)"
        return True, ""

    def probe_browser_ssrf_guard_wired(_ctx) -> Tuple[bool, str]:
        try:
            from ..tools import browser
            src = browser._runner_script()
        except Exception as e:
            return False, f"cannot load browser runner: {e}"
        # Tolerant of the guard's kwargs (sandbox_root=, anonymous=) added when
        # the file://-escape + DNS-rebind residuals were closed — just assert
        # the interceptor is still installed on the context.
        if "_install_ssrf_guard(ctx" not in src:
            return False, "browser runner missing the SSRF request interceptor"
        return True, ""

    def probe_redact_conn_uri(_ctx) -> Tuple[bool, str]:
        try:
            from ..distill.redact import redact_text
        except Exception as e:
            return False, f"cannot import redact: {e}"
        out = redact_text("mongodb://admin:aB/cD3f@host:27017/db")
        if "aB/cD3f" in out:
            return False, "conn-uri password with '/' leaked past redaction"
        return True, ""

    def probe_outcome_consolidation(_ctx) -> Tuple[bool, str]:
        # The trajectory corpus outcome must fold in the SAME signals
        # calibration + selfhood use (verifier verdict + structural failure),
        # so a verifier-caught wrong answer becomes a lesson / PRM negative
        # instead of silently staying UNKNOWN.
        try:
            from ..distill.outcome_heuristics import resolve_turn_outcome
            from ..distill.schema import Outcome
        except Exception as e:
            return False, f"cannot import resolve_turn_outcome: {e}"
        F, P, U = Outcome.FAILED.value, Outcome.PASSED.value, Outcome.UNKNOWN.value
        checks = [
            (resolve_turn_outcome(current=U, verifier="failed") == F, "verifier-refuted → FAILED"),
            (resolve_turn_outcome(current=U, execution_failed=True) == F, "structural → FAILED"),
            (resolve_turn_outcome(current=F, verifier="passed") == F, "FAILED not upgraded away"),
            (resolve_turn_outcome(current=U, verifier="passed") == P, "verifier-passed → PASSED"),
        ]
        bad = [msg for ok, msg in checks if not ok]
        return (not bad, "outcome consolidation wrong: " + "; ".join(bad) if bad else "")

    def probe_native_toolcall_repair(_ctx) -> Tuple[bool, str]:
        # With --native-tools on, some upstream servers merge a multi-tool
        # reply into ONE call's arguments, leaking the following call's XML
        # into the first argument's string value — so a valid `action` like
        # 'summary' arrives as 'summary</parameter>...<function=...>' and
        # fails validation. _repair_native_tool_calls must recover the
        # intended value and split the leaked call back out.
        try:
            import json
            from ..core.agent import _repair_native_tool_calls
            from ..tools.introspect import _VALID_ACTIONS
        except Exception as e:
            return False, f"cannot import repair helper: {e}"
        corrupt = ("summary</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
                   "<function=list_lessons>\n<parameter=scope>\nall")
        tc = {"id": "c", "type": "function", "function": {
            "name": "introspect",
            "arguments": json.dumps({"action": corrupt, "limit": 10})}}
        out, repaired = _repair_native_tool_calls([tc], ["introspect", "list_lessons"])
        if not repaired:
            return False, "leak not detected — corrupt native tool_call left uncorrected"
        action = json.loads(out[0]["function"]["arguments"]).get("action")
        if action not in _VALID_ACTIONS:
            return False, f"recovered action {action!r} still invalid"
        names = [t["function"]["name"] for t in out]
        if "list_lessons" not in names:
            return False, "leaked second tool call was dropped, not recovered"
        # A clean call must pass through byte-for-byte.
        clean = {"id": "d", "type": "function", "function": {
            "name": "introspect", "arguments": json.dumps({"action": "summary"})}}
        out2, rep2 = _repair_native_tool_calls([clean], ["introspect"])
        if rep2 or out2 != [clean]:
            return False, "clean tool_call was mutated by the repair"
        return True, ""

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
        RegressionProbeTask(
            task_id="probe:outcome_heuristic_exit_codes",
            category="regression",
            prompt="outcome labelling flags any non-zero exit code",
            validator=probe_outcome_heuristic_exit_codes,
        ),
        RegressionProbeTask(
            task_id="probe:trajectory_schema_drift",
            category="regression",
            prompt="trajectory corpus tolerates a newer-schema field",
            validator=probe_trajectory_schema_drift,
        ),
        RegressionProbeTask(
            task_id="probe:prm_skips_junk_outcomes",
            category="regression",
            prompt="PRM skips non-PASSED/FAILED outcomes (no false negatives)",
            validator=probe_prm_skips_junk_outcomes,
        ),
        RegressionProbeTask(
            task_id="probe:browser_ssrf_guard_wired",
            category="regression",
            prompt="browser runner installs the SSRF request interceptor",
            validator=probe_browser_ssrf_guard_wired,
        ),
        RegressionProbeTask(
            task_id="probe:redact_conn_uri",
            category="regression",
            prompt="redaction catches connection-URI passwords with special chars",
            validator=probe_redact_conn_uri,
        ),
        RegressionProbeTask(
            task_id="probe:outcome_consolidation",
            category="regression",
            prompt="trajectory outcome folds in verifier + structural signals",
            validator=probe_outcome_consolidation,
        ),
        RegressionProbeTask(
            task_id="probe:native_toolcall_repair",
            category="regression",
            prompt="corrupt native tool_call args are repaired to a valid call",
            validator=probe_native_toolcall_repair,
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
            # Word-boundary + a fair greeting set: a personalized "Good
            # morning, <name>!" is a correct greeting, so the old literal
            # ["hello","hi","hey"] substring list was validator noise (and
            # "hi" as a substring also false-matched "this"/"which").
            prompt="Greet the user in one friendly sentence.",
            validator=_word_match(
                "hello", "hi", "hey", "greetings", "welcome", "howdy",
                "morning", "afternoon", "evening",
            ),
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


def _word_match(*answers: str) -> Validator:
    """Validator: pass if ANY answer appears as a whole word (case-insensitive).
    Word-boundary so "7" doesn't match inside "1972" and "no" doesn't match
    inside "another"."""
    pats = [re.compile(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", re.IGNORECASE)
            for a in answers]

    def _v(out: str, _ctx) -> Tuple[bool, str]:
        text = str(out or "")
        if any(p.search(text) for p in pats):
            return True, ""
        return False, f"expected one of {list(answers)} as a word in the reply"
    return _v


def _json_field_is(key: str, value: Any) -> Validator:
    """Validator: the reply contains a JSON object with key==value."""
    def _v(out: str, _ctx) -> Tuple[bool, str]:
        import json as _json
        text = str(out or "")
        # Extract the first {...} block (models wrap JSON in prose/fences).
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return False, "no JSON object in reply"
        try:
            obj = _json.loads(text[start:end + 1])
        except Exception:
            return False, "reply's JSON did not parse"
        if not isinstance(obj, dict) or obj.get(key) != value:
            return False, f"expected JSON {key}={value!r}, got {obj.get(key)!r}"
        return True, ""
    return _v


def _load_capability_tasks() -> List[CuratedRequestTask]:
    """Deterministic, text-validated CAPABILITY tasks across the agent's real
    job categories: factual recall, multi-step arithmetic, code tracing,
    instruction/format following, and structured output. A competent agent
    passes ~all; a broken one fails. These validate the agent's TEXT reply,
    so they run against a live agent WITHOUT the Docker sandbox — giving a
    reproducible capability number for the baseline gate.

    Kept deliberately unambiguous (single correct token / parseable JSON) so
    the metric reflects capability, not validator noise."""
    return [
        CuratedRequestTask(
            task_id="cap:factual_capital",
            category="capability",
            prompt="What is the capital of France? Answer in one word.",
            validator=_word_match("paris"),
        ),
        CuratedRequestTask(
            task_id="cap:arithmetic_multistep",
            category="capability",
            prompt="Compute (17 * 3) + 5. Reply with just the number.",
            validator=_word_match("56"),
        ),
        CuratedRequestTask(
            task_id="cap:code_trace_len",
            category="capability",
            prompt="What does this Python print: print(len('hello'))? Reply with just the number.",
            validator=_word_match("5"),
        ),
        CuratedRequestTask(
            task_id="cap:count_week",
            category="capability",
            prompt="How many days are in a week? Reply with just the number.",
            validator=_word_match("7", "seven"),
        ),
        CuratedRequestTask(
            task_id="cap:primality",
            category="capability",
            prompt="Is 17 a prime number? Answer yes or no.",
            validator=_word_match("yes"),
        ),
        CuratedRequestTask(
            task_id="cap:instruction_one_word",
            category="capability",
            prompt="Answer in exactly one word: what color is a clear daytime sky?",
            validator=_word_match("blue"),
        ),
        CuratedRequestTask(
            task_id="cap:format_json",
            category="capability",
            prompt='Output a JSON object with a single key "answer" whose value is the integer 42. Output only the JSON.',
            validator=_json_field_is("answer", 42),
        ),
        CuratedRequestTask(
            task_id="cap:sequence_next",
            category="capability",
            prompt="What is the next number in the sequence 2, 4, 8, 16, ...? Reply with just the number.",
            validator=_word_match("32"),
        ),
    ]


def load_capability_suite() -> List[EvalTask]:
    """The http-scorable CAPABILITY baseline: regression invariants + capability
    + curated tasks, but NO challenge templates.

    Template (coding) tasks need the Docker sandbox to run their shell-script
    validator and return a ``passed`` verdict; the plain http runner can't, so
    they always score unverified→fail and drag the pass-rate down without
    measuring anything. Freeze THIS suite for a clean, trustworthy capability
    number against a live agent; measuring coding capability needs a
    sandbox-verdict runner (see docs/self_improvement.md)."""
    return load_default_suite(include_templates=False)


def load_offline_suite() -> List[EvalTask]:
    """The AGENT-FREE invariant gate: regression probes only.

    Runs entirely in-process (no live agent, no Docker, no model) — so it is
    the fast CI gate and the self-audit the agent can run on itself. Every
    probe is a load-bearing invariant (cooldown ordering, telemetry-off,
    learning-loop input integrity, the browser SSRF guard). Expectation is
    100% pass; any failure is a real regression.
    """
    return list(_load_regression_probes())


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
    include_capability: bool = True,
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
    if include_capability:
        tasks.extend(_load_capability_tasks())
    if include_curated:
        tasks.extend(_load_curated_tasks())
    if include_templates:
        tasks.extend(_load_challenge_tasks(n_per_cluster=n_per_cluster, tiers=tiers))
    return tasks
