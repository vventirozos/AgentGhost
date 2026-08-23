"""Dream (§4CM) — counterfactual replay of recorded episodes.

Part 1 (D0): the corpus and the replayability triage. Nothing here runs
anything; it decides WHICH recorded turns can honestly be re-executed and
what perturbations are worth asking about, and it writes the specs.

**The rule the whole phase rests on** (the Replay Gap result, COLM'26):
always re-execute FORWARD from the fork. Never splice a changed decision
into a recorded suffix — 61-94% of post-fork actions get rewritten, so a
spliced evaluation scores a world that never happens. Every perturbation
below is therefore "change one thing at step k, then let the agent run to
the end on its own".

Three filters, and the order matters because each one is cheap relative
to the next:

1. **Decisive.** ``outcome in {passed, failed}`` after the corrections
   overlay. An episode with no recorded verdict has nothing for D1's
   synthesised validator to self-test against, so it cannot be admitted
   without trusting the validator on its synthesis alone — which is the
   thing D1 exists not to do. Measured 2026-08-22: this alone is 46% of
   live turns.
2. **Safe.** ``journal_challenges._is_unsafe_challenge`` — the same
   destructive denylist that gates mined challenges. A replay runs REAL
   tools; the mined-challenge path has already paid for getting this
   wrong once.
3. **Replay-safe tools, as an ALLOW-LIST.** An episode is replayable only
   if every tool it used is on :data:`REPLAYABLE_TOOLS`. Deliberately not
   the complement of ``isolation.REPLAY_FORBIDDEN_TOOLS``: an unknown
   tool (a runtime-registered acquired skill, a macro, anything added
   later) must read as NOT replayable. Guard the thing, not a proxy.

**Why network tools are excluded, and it is not (only) safety.** A replay
of an episode that searched the web is not reproducible: the search
returns different results than it did on the day, so control and
perturbed legs both diverge for reasons that have nothing to do with the
perturbation, and the paired consistency rule abstains. Non-reproducible
episodes produce verdicts about noise. This is a REPRODUCIBILITY filter
that happens to also be a safety one.

**What that costs, measured by running the triage itself (2026-08-22):**

    REAL corpus (46 days of user turns)   1,799 records → 67 replayable
    BENCH corpus (bank solves)              118 records → 77 replayable
                                                        ─────────────
                                                        144 episodes

    …yielding 303 specs: 131 lesson_withhold (real only — bench
    trajectories carry no hydrated lessons), 144 verify_toggle,
    28 step_deny.

Real episodes accrue at ~1.5/day. That is the honest ceiling on this
phase's input, and it is an order of magnitude below what IDE.md's
throughput target assumed: at `DEFAULT_BATCH=3` a night produces at most
**3 credit rows**, so §6's "≥30 verdicts/night" is 10× out of reach and
draining the backlog is ~100 nights at one batch, ~17 days at the 6
batches/day the 4-hour cooldown allows. The multiplier is
perturbations-per-episode, not episodes.

⚠ The bench half is why `EpisodeSource` defaults `include_bench=True`:
bench items carry EXECUTABLE validators already, which is exactly what
D1 has to synthesise for a real episode. They are the cheapest seeded
ground truth available and D4 should use them — it currently does not,
which is recorded as remaining work rather than papered over.

**A DEVIATION from IDE.md, recorded because it is deliberate.** The plan
says to reuse ``_is_transferable_challenge`` for triage. That predicate
asks "can this request be honestly re-posed as *write a solution.py over
the fixture in your working directory*" — it exists for the mined-
challenge harness, which rewrites the task. A replay does not rewrite
anything; it re-runs the recorded turn. Applying a filter built for a
different harness would be a lexical proxy for a property nobody
measured. The safety half (``_is_unsafe_challenge``) IS reused, because
that one is about what the tools do, which is identical in both.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import re
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .isolation import (
    IsolationUnavailable, isolated_replay_context,
)

logger = logging.getLogger("GhostAgent")

#: Consumer key for READING the corpus. Registered BENCH_FEATURE in
#: `core/admissibility.py`: bench episodes carry their own executable
#: validators, which is exactly what D1 has to synthesise for real ones,
#: so they are the cheapest source of seeded ground truth.
CONSUMER_READ = "dream_replay"

#: Consumer key for anything that later READS a replay verdict and acts
#: on it. Separate from the read row on purpose — D4 has to pass before
#: any consumer opens, and a single row could not express "may read the
#: corpus, may not yet believe the output".
CONSUMER_CREDIT = "dream_credit"

#: Tools a replayed episode may use. ALLOW-LIST: an episode using
#: anything not named here is not replayable. Everything here is
#: fork-scoped (the sandbox and workspace are copies), read-only, or
#: computes without leaving the process.
#:
#: ⚠ Not the complement of `isolation.REPLAY_FORBIDDEN_TOOLS`. That is a
#: containment list; this is a reproducibility list, and the two differ
#: on exactly the tools that are safe to run but do not return the same
#: thing twice (`web_search`, `browser`, `vision_analysis`, and every
#: runtime-registered acquired skill).
REPLAYABLE_TOOLS = frozenset({
    "file_system", "execute", "workspace", "workspace_track",
    "recall", "list_lessons", "introspect", "scratchpad",
    "postmortem", "flag_uncertainty", "report_pdf",
})

#: Perturbation kinds v1. Each asks a question execution can answer.
PERTURB_LESSON_WITHHOLD = "lesson_withhold"
#: ⚠ NOT GENERATED IN v1. `build_specs` never emits one, because there
#: is no honest source for WHICH lesson to inject — picking one would put
#: an ungrounded generator inside a measurement whose value is that it is
#: execution-grounded, which is the same objection that turned `tool_swap`
#: into `step_deny`. The perturbation is IMPLEMENTED (a caller can build
#: the spec by hand, and D4 uses it as a seeded positive); it is not
#: mined. Advertising a kind nothing produces is how a capability list
#: stops describing the system.
PERTURB_LESSON_INJECT = "lesson_inject"
#: ⚠ IDE.md calls this ``tool_swap(step_call -> alternative_call)``.
#: Nothing in a recording says what the alternative SHOULD be, so
#: something would have to invent one — and an invented alternative puts
#: an ungrounded generator inside a measurement whose entire value is
#: that it is execution-grounded. Denying the recorded call asks the same
#: counterfactual ("did that step matter, or was there another route?")
#: with nothing invented, so the name says that instead.
PERTURB_STEP_DENY = "step_deny"
PERTURB_VERIFY_TOGGLE = "verify_toggle"

#: Remove a set of tools for the WHOLE leg (`target` is comma-separated).
#: ⚠ NOT MINED, like `lesson_inject`: nothing in a recording says which
#: tools an episode could not have done without, and guessing would put
#: an ungrounded generator inside an execution-grounded measurement.
#:
#: It exists because it is the only perturbation that can be made
#: MECHANICALLY GUARANTEED to matter, which is what a seeded positive
#: needs. Given a task whose validator runs `python3 solution.py`,
#: ablating both `file_system` and `execute` leaves no path to create
#: that file — so the perturbed arm MUST fail, whatever the model does.
#: `scripts/dream_replay_validate.py` builds its positives this way, and
#: its matched nulls travel the identical code path (ablate tools the
#: task never needed), which is the property the old seeded pair lacked.
PERTURB_TOOL_ABLATE = "tool_ablate"

PERTURB_KINDS = (PERTURB_LESSON_WITHHOLD, PERTURB_LESSON_INJECT,
                 PERTURB_STEP_DENY, PERTURB_VERIFY_TOGGLE,
                 PERTURB_TOOL_ABLATE)

#: Why an episode was rejected. These are counted and reported: a triage
#: that says only "77 of 1,607" cannot tell a corpus that is thin from
#: one that is being over-filtered, and those lead opposite places.
REJECT_KIND = "not_a_user_turn"
REJECT_UNDECIDED = "no_recorded_outcome"
REJECT_THIN = "under_two_steps_or_no_tool_calls"
REJECT_UNSAFE = "destructive_denylist"
REJECT_TOOL = "used_a_non_replayable_tool"
REJECT_NO_REQUEST = "no_user_request_text"
REJECT_NO_ARTIFACT = "no_filesystem_checkable_deliverable"

#: Tools that leave something on disk regardless of their arguments.
#: `execute` because a script can write files the trace never names, and
#: `report_pdf` because writing a PDF into the sandbox is the entire
#: tool — it was missing from the first version of this rule and the
#: measurement caught it: "write me a pdf report about tasks 15-19" was
#: being rejected as having no checkable deliverable. `workspace` is
#: documented "read-only workspace introspection" and `scratchpad` is
#: in-memory in an isolated context, so neither produces anything a
#: filesystem check could see.
_PRODUCING_TOOLS = frozenset({"execute", "report_pdf"})

#: `file_system` operations that can leave something on disk for a
#: validator to inspect. Everything else (read, list_files, search,
#: inspect) observes without producing.
_PRODUCING_FS_OPS = frozenset({"write", "replace", "append", "create",
                               "insert", "mkdir", "move", "copy",
                               "download", "extract"})

_STATE_DIRNAME = "dream_replay"
_SPECS_FILENAME = "specs.jsonl"
_LEDGER_MAX_BYTES = 8_000_000
_WRITE_LOCK = threading.Lock()


def _state_dir(home: str = None) -> Optional[Path]:
    base = (home if home is not None else os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return None
    return Path(base) / "system" / _STATE_DIRNAME


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

@dataclass
class Triage:
    """The verdict on one recorded turn, and why."""
    replayable: bool
    reason: str = ""
    tools: Tuple[str, ...] = ()
    n_steps: int = 0
    outcome: str = ""


def _tool_names(traj) -> Tuple[str, ...]:
    out = []
    for tc in (getattr(traj, "tool_calls", None) or []):
        name = str(getattr(tc, "name", "") or "").strip()
        if name:
            out.append(name)
    return tuple(out)


def _produces_an_artifact(traj) -> bool:
    """Did this turn leave anything on disk for a check to inspect?

    ⚠ MEASURED. The validator prompt already has an escape hatch — rule
    5, "print WHY on stderr and exit 2 if the task cannot be checked from
    the filesystem" — and it fires, but not reliably: in the first A/A
    census 7 episodes exited 2 while 8 more got a filesystem check
    written for a task whose whole deliverable was a SENTENCE
    ("Summarise the latest files in my sandbox", "what patterns do you
    notice in yourself?", "tell me about one mistake you learned from").
    Those checks cannot pass, so the episode reads as a recorded success
    that does not reproduce — and each one costs a validator synthesis
    plus six full agent runs to discover.

    The trace already knows. A turn that never wrote a file and never
    ran anything is not filesystem-checkable, and asking the model to
    notice that is asking it to volunteer against the shape of the task
    it was given.

    DELIBERATELY CONSERVATIVE: any `execute` or `report_pdf` call counts
    as producing, because a script can write files the trace does not
    name. That keeps 2 of the 8 measured cases in the corpus rather than
    risk dropping a real one — a false reject costs an episode
    permanently, a false accept costs one night's slot. The first version
    of this list omitted `report_pdf` and the measurement caught it
    immediately: "write me a pdf report about tasks 15-19" is exactly a
    filesystem-checkable deliverable.
    """
    for tc in (getattr(traj, "tool_calls", None) or []):
        name = str(getattr(tc, "name", "") or "").lower()
        if name in _PRODUCING_TOOLS:
            return True
        args = getattr(tc, "arguments", None)
        if not isinstance(args, dict):
            continue
        op = str(args.get("operation") or args.get("action") or "").lower()
        if name == "file_system" and op in _PRODUCING_FS_OPS:
            return True
    return False


def triage(traj) -> Triage:
    """Can this recorded turn be honestly re-executed? Never raises: a
    trajectory that cannot even be inspected is not replayable, which is
    the safe answer."""
    try:
        kind = str(getattr(traj, "task_kind", "") or "")
        outcome = str(getattr(traj, "outcome", "") or "").lower()
        tools = _tool_names(traj)
        n_steps = int(getattr(traj, "n_steps", 0) or len(tools))
        base = dict(tools=tools, n_steps=n_steps, outcome=outcome)

        if kind not in ("user_request", "bench"):
            return Triage(False, REJECT_KIND, **base)
        if outcome not in ("passed", "failed"):
            return Triage(False, REJECT_UNDECIDED, **base)
        if n_steps < 2 or not tools:
            return Triage(False, REJECT_THIN, **base)
        request = str(getattr(traj, "user_request", "") or "").strip()
        if not request:
            return Triage(False, REJECT_NO_REQUEST, **base)
        try:
            from .journal_challenges import _is_unsafe_challenge
            if _is_unsafe_challenge(request):
                return Triage(False, REJECT_UNSAFE, **base)
            # The denylist matches on TEXT, and the destructive thing an
            # episode did lives in its commands, not necessarily in the
            # request that prompted them.
            for tc in (getattr(traj, "tool_calls", None) or []):
                args = getattr(tc, "arguments", None)
                if not isinstance(args, dict):
                    continue
                for key in ("command", "cmd", "sql", "query"):
                    val = args.get(key)
                    if isinstance(val, str) and _is_unsafe_challenge(val):
                        return Triage(False, REJECT_UNSAFE, **base)
        except ImportError:  # pragma: no cover - defensive
            return Triage(False, REJECT_UNSAFE, **base)
        unknown = [t for t in tools if t not in REPLAYABLE_TOOLS]
        if unknown:
            return Triage(False, REJECT_TOOL, **base)
        # LAST, so the census delta is readable: this reason only ever
        # claims episodes that would otherwise have been admitted.
        if not _produces_an_artifact(traj):
            return Triage(False, REJECT_NO_ARTIFACT, **base)
        return Triage(True, "", **base)
    except Exception as exc:  # noqa: BLE001
        logger.debug("replay triage failed closed (%s)", exc)
        return Triage(False, "triage_error")


# ---------------------------------------------------------------------------
# Episode source
# ---------------------------------------------------------------------------

class EpisodeSource:
    """Replayable episodes, newest day first, with a triage census.

    The census is not decoration. A count of admitted episodes says
    nothing on its own — "the corpus is thin" and "the filter is too
    tight" produce the same number and need opposite responses. Every
    call fills :attr:`rejected` with the per-reason histogram.
    """

    def __init__(self, *, consumer: str = CONSUMER_READ, args=None,
                 include_bench: bool = True):
        self.consumer = consumer
        self.args = args
        self.include_bench = include_bench
        self.rejected: Dict[str, int] = {}
        self.seen = 0

    def _note(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def _iter_real(self) -> Iterator:
        try:
            from ..distill.collector import TrajectoryCollector
            collector = TrajectoryCollector()
            if not collector.root.exists():
                return
            days = sorted((p.name for p in collector.root.iterdir()
                           if p.is_dir()), reverse=True)
            for day in days:
                yield from collector.iter_trajectories(day=day)
        except Exception as exc:  # noqa: BLE001
            logger.debug("replay: real corpus unavailable (%s)", exc)

    def _iter_bench(self) -> Iterator:
        if not self.include_bench:
            return
        try:
            from .admissibility import iter_bench_trajectories
            yield from iter_bench_trajectories(self.consumer, self.args)
        except Exception as exc:  # noqa: BLE001
            logger.debug("replay: bench corpus unavailable (%s)", exc)

    def iter_episodes(self, limit: int = None) -> Iterator[Tuple[Any, Triage]]:
        """Yield ``(trajectory, triage)`` for REPLAYABLE episodes only.
        Rejections are counted, not yielded."""
        n = 0
        for traj in self._iter_real():
            self.seen += 1
            t = triage(traj)
            if not t.replayable:
                self._note(t.reason)
                continue
            yield traj, t
            n += 1
            if limit is not None and n >= limit:
                return
        for traj in self._iter_bench():
            self.seen += 1
            t = triage(traj)
            if not t.replayable:
                self._note(t.reason)
                continue
            yield traj, t
            n += 1
            if limit is not None and n >= limit:
                return

    def census(self, limit: int = None) -> Dict[str, Any]:
        """Walk the corpus and report the funnel. Cheap (no execution),
        and the number the phase's throughput claim rests on."""
        admitted = [t for _, t in self.iter_episodes(limit=limit)]
        return {
            "seen": self.seen,
            "replayable": len(admitted),
            "rejected": dict(sorted(self.rejected.items(),
                                    key=lambda kv: -kv[1])),
            "steps_total": sum(t.n_steps for t in admitted),
            "by_outcome": {
                o: sum(1 for t in admitted if t.outcome == o)
                for o in ("passed", "failed")
            },
        }


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class ReplaySpec:
    """One question, in a form execution can answer.

    ``fork_step`` is where the perturbation applies; the run then FREE-RUNS
    to the end (never splices). ``seed`` makes a spec reproducible enough
    to re-run for the stability check D4 requires — it does not make the
    model deterministic, which is why the verdict rule is paired legs, not
    a single comparison.
    """
    trajectory_id: str
    perturbation: str
    fork_step: int = 0
    target: str = ""           # the trigger / tool name the perturbation acts on
    seed: int = 0
    origin: str = "user_request"
    #: For `lesson_inject`: the lesson dict to add. Empty for every kind
    #: the miner generates.
    inject: Dict[str, Any] = field(default_factory=dict)
    #: For challenge-backed seeds: a script run in the fork BEFORE the
    #: agent starts, so the task has its fixture. Empty for a replay of a
    #: recorded turn, which forks the live workspace instead.
    setup_script: str = ""
    user_request: str = ""
    recorded_outcome: str = ""
    n_steps: int = 0
    created: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")

    @property
    def spec_id(self) -> str:
        """Stable identity of the QUESTION, not of the row. Two specs
        asking the same thing about the same episode collide on purpose —
        that is what makes a re-run recognisable as a re-run (the D4
        stability check) instead of as new evidence."""
        # `recorded_outcome` is IN the identity: 406 correction rows exist
        # in the trajectory sidecar, and an episode later corrected
        # passed→failed is a DIFFERENT question. Without it the dedup
        # blocks re-asking and the credit row on disk keeps an outcome
        # that is now wrong.
        basis = (f"{self.trajectory_id}\x00{self.perturbation}"
                 f"\x00{self.fork_step}\x00{self.target}"
                 f"\x00{self.recorded_outcome}")
        return hashlib.sha1(basis.encode("utf-8", "replace")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spec_id"] = self.spec_id
        return d


def _hydrated_triggers(traj) -> List[str]:
    """Lesson triggers the recorded turn actually had in its prompt.
    This is the ONLY honest source for a withhold perturbation: asking
    "what if lesson X had been absent" is meaningless unless X was
    present."""
    try:
        extra = getattr(traj, "extra", None) or {}
        raw = extra.get("hydrated_lessons")
        if isinstance(raw, str):
            raw = [raw]
        return [str(x) for x in (raw or []) if str(x).strip()]
    except Exception:  # noqa: BLE001
        return []


def build_specs(traj, tri: Triage = None, *,
                max_per_episode: int = 4) -> List[ReplaySpec]:
    """The perturbations worth asking about THIS episode.

    Only perturbations whose premise the episode actually satisfies are
    generated: a withhold needs a lesson that was really hydrated, a tool
    swap needs a step with an alternative worth trying. A spec whose
    premise is false produces a verdict about nothing, and a corpus of
    those is how a label source becomes a noise source.
    """
    tri = tri or triage(traj)
    if not tri.replayable:
        return []
    tid = str(getattr(traj, "id", "") or "")
    request = str(getattr(traj, "user_request", "") or "")
    common = dict(trajectory_id=tid, user_request=request[:2000],
                  recorded_outcome=tri.outcome, n_steps=tri.n_steps,
                  origin=str(getattr(traj, "task_kind", "") or ""))
    specs: List[ReplaySpec] = []

    # 1. lesson_withhold — "did that lesson matter?" Only for lessons the
    #    turn genuinely hydrated.
    for trigger in _hydrated_triggers(traj)[:max_per_episode]:
        specs.append(ReplaySpec(perturbation=PERTURB_LESSON_WITHHOLD,
                                fork_step=0, target=trigger,
                                seed=_seed_of(tid, trigger), **common))

    # 2. verify_toggle — "did the verifier's gate change the outcome?"
    #    One per episode; the premise is always satisfied.
    specs.append(ReplaySpec(perturbation=PERTURB_VERIFY_TOGGLE,
                            fork_step=0, target="verifier",
                            seed=_seed_of(tid, "verify"), **common))

    # 3. step_deny — "did that step matter, or was there another route?"
    #    Marked at the first step whose recorded result looks like a
    #    failure. On a passing episode there is no such step and no spec,
    #    which is correct: denying a step that worked asks nothing.
    deny_at = _first_failing_step(traj)
    if deny_at is not None:
        specs.append(ReplaySpec(perturbation=PERTURB_STEP_DENY,
                                fork_step=deny_at,
                                target=tri.tools[deny_at]
                                if deny_at < len(tri.tools) else "",
                                seed=_seed_of(tid, f"deny{deny_at}"),
                                **common))
    return specs[:max_per_episode + 2]


def _seed_of(tid: str, salt: str) -> int:
    h = hashlib.sha1(f"{tid}\x00{salt}".encode("utf-8", "replace")).digest()
    return int.from_bytes(h[:4], "big")


def _first_failing_step(traj) -> Optional[int]:
    """Index of the first recorded tool call that failed, using THE
    corpus failure rule — not a second definition of failure."""
    try:
        from ..distill.outcome_heuristics import tool_call_failed
    except Exception:  # noqa: BLE001
        return None
    for i, tc in enumerate(getattr(traj, "tool_calls", None) or []):
        try:
            if tool_call_failed(tc):
                return i
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_specs(specs: List[ReplaySpec], home: str = None) -> int:
    """Append specs to ``system/dream_replay/specs.jsonl``. Size-rotated
    at 8 MB following the foresight ledger pattern; a durable store is
    never truncated in place. Returns the number written (0 when there is
    no GHOST_HOME — an ad-hoc import must stay silent)."""
    d = _state_dir(home)
    if d is None or not specs:
        return 0
    path = d / _SPECS_FILENAME
    lines = [json.dumps(s.to_dict(), ensure_ascii=False) for s in specs]
    blob = "\n".join(lines) + "\n"
    with _WRITE_LOCK:
        try:
            d.mkdir(parents=True, exist_ok=True)
            try:
                if (path.stat().st_size + len(blob.encode("utf-8"))
                        > _LEDGER_MAX_BYTES):
                    os.replace(str(path), str(path) + ".1")
            except FileNotFoundError:
                pass
            with path.open("a", encoding="utf-8") as f:
                f.write(blob)
                f.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("replay: could not write specs (%s)", exc)
            return 0
    return len(specs)


def iter_specs(home: str = None) -> Iterator[Dict[str, Any]]:
    d = _state_dir(home)
    if d is None:
        return
    path = d / _SPECS_FILENAME
    for p in (Path(str(path) + ".1"), path):
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
        except OSError:
            continue


def known_spec_ids(home: str = None) -> set:
    """Specs that have been ANSWERED — i.e. that produced a credit row.

    ⚠ Deliberately keyed on `credits.jsonl`, not on `specs.jsonl`. A spec
    that was planned and then skipped has not been answered, and treating
    it as answered burns it permanently: on a box where docker is down
    every batch would consume three specs and record nothing, and the
    engine would quietly exhaust its own corpus.

    A nightly batch that re-asks a question it HAS answered spends its
    budget re-measuring; D4's stability check is the one place a
    deliberate re-run belongs.
    """
    return ({str(r.get("spec_id") or "") for r in iter_credits(home)
             if r.get("spec_id")})




# ===========================================================================
# D1 — validator synthesis, with a self-test
# ===========================================================================

#: A synthesised validator that cannot fail is not a check. The self-test
#: below is its mutation test: run the UNPERTURBED control leg and require
#: the validator to reproduce the episode's RECORDED outcome. Disagreement
#: means either the world drifted or the validator is wrong, and both mean
#: "not replayable" — never "assume the validator".
VALIDATOR_FILENAME = ".validator.py"
VALIDATOR_MAX_CHARS = 6000
VALIDATOR_TIMEOUT_S = 30

_VALIDATOR_PROMPT = """You are writing an EXECUTABLE CHECK for a task that was already attempted.

THE TASK the agent was given:
{request}

WHAT IT DID (tool calls, in order):
{trace}

ITS FINAL REPLY:
{final}

Write a Python 3 script that decides, by INSPECTING THE WORKING DIRECTORY, whether the task was accomplished. Rules:

1. Exit 0 if the task WAS accomplished; exit 1 if it was not. No other exit codes.
2. Inspect files under the current working directory. Standard library ONLY.
   Every path above is ALREADY relative to that directory — do not prepend
   a `projects/<id>/` or any other prefix.
3. NO network, NO subprocess, NO installing anything, NO writing or deleting files.
4. Check the OUTCOME, not the method: a different-but-correct approach must still pass.
5. If the task cannot be checked from the filesystem, print WHY on stderr and exit 2.
6. Under 60 lines. Print one short line on stdout saying what you checked.

Return ONLY the Python source, no markdown fence, no commentary."""


#: A project-scoped path as the RECORDING saw it. The live agent runs
#: with `current_project_id` set, so `file_system` and `execute` are
#: scoped into `<sandbox>/projects/<id>/` — 45 of 93 sampled file calls
#: carry such a path.
_PROJECT_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])projects/[A-Za-z0-9_-]{4,}/"
)
#: The sandbox mount point as it appears in tool OUTPUT.
_WORKSPACE_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9_.-])/workspace/")
#: `cd projects/<id> && …` — the project directory as a DESTINATION
#: rather than a path prefix. Stripping just the directory would leave a
#: bare `cd && python3 x.py`, so the whole hop goes: in the fork the
#: working directory already IS what that `cd` was reaching for.
_PROJECT_CD_RE = re.compile(r"cd\s+projects/[A-Za-z0-9_-]{4,}\s*&&\s*")
#: The sandbox root as it appears in a HOST path — a `list_files`
#: argument reads `/Users/<user>/Data/AI/Data/sandbox/projects/<id>`.
_SANDBOX_ROOT_RE = re.compile(r"(?<![A-Za-z0-9_.-])/[^\s'\"]*/sandbox/")
#: A project directory with nothing under it: `…/projects/<id>` at a
#: boundary. ⚠ The right-hand lookahead is UNREACHABLE as the rules are
#: ordered — `_PROJECT_PREFIX_RE` runs first and consumes every
#: `projects/<id>/…` form, so the only case the lookahead excludes never
#: arrives. Mutation-tested: deleting it survives the suite, and a fuzz
#: over 219,558 strings found NO input where it changes the result. It
#: stays as a guard for a future reordering, not as a live rule; do not
#: write a test for it, since no test could distinguish the two. That names the project ROOT, which in the fork is the
#: working directory, so it becomes `.` rather than vanishing — an empty
#: `path=` would tell the model nothing at all.
_PROJECT_DIR_RE = re.compile(
    r"(?<![A-Za-z0-9_-])projects/[A-Za-z0-9_-]{4,}(?![A-Za-z0-9_/-])"
)
#: …and the same for the mount point with nothing under it, which is how
#: a search root reads: `find /workspace -type f`, `path=/workspace`.
_WORKSPACE_DIR_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/workspace(?![A-Za-z0-9_/-])"
)


def _fork_relative(value: str) -> str:
    """A recorded path, rewritten to where the REPLAY will put it.

    ⚠ MEASURED DEFECT, not a precaution. `isolated_replay_context` nulls
    `current_project_id` deliberately — its own comment says why: the
    scoping would send the agent's writes into `projects/<id>/` "while
    the setup and validator scripts write at the root". But the trace fed
    to the validator synthesiser still carried the RECORDED paths, so the
    model wrote checks like

        result_path = "projects/7b62e5e533d1/cross_session_results.json"

    against a fork that puts the file at the root. Those checks cannot
    pass — not "usually fail", CANNOT — and the episode then reads as a
    recorded success that does not reproduce. Verified by synthesising
    the real validators for two such episodes: both named the project
    path. 4 of 12 non-reproducing episodes in the first A/A census were
    this and nothing else.
    """
    # ⚠ ORDER MATTERS, and getting it wrong is how the first version
    # still leaked. The real command is
    # `cd /workspace/projects/<id> && python3 demo.py`: the cd rule
    # cannot see it until the mount point is gone, so roots come off
    # FIRST, then the `cd` hop, then the prefix, then a bare directory.
    out = _SANDBOX_ROOT_RE.sub("", str(value or ""))
    out = _WORKSPACE_PREFIX_RE.sub("", out)
    out = _PROJECT_CD_RE.sub("", out)
    out = _PROJECT_PREFIX_RE.sub("", out)
    # A project directory named with no path under it means "the project
    # root", which in the fork is the working directory.
    out = _PROJECT_DIR_RE.sub(".", out)
    return _WORKSPACE_DIR_RE.sub(".", out)


def _trace_for_prompt(traj, max_calls: int = 12) -> str:
    lines = []
    for i, tc in enumerate(getattr(traj, "tool_calls", None) or []):
        if i >= max_calls:
            lines.append(f"  … {len(traj.tool_calls) - max_calls} more")
            break
        args = getattr(tc, "arguments", None)
        args = args if isinstance(args, dict) else {}
        bits = []
        for k in ("operation", "action", "path", "command", "content"):
            v = args.get(k)
            if isinstance(v, str) and v:
                # EVERY value, `content` included. Sparing content was the
                # last leak: a script the agent wrote held the literal
                # `path = "projects/<id>/model.py"`, and a check built from
                # that asserts a location the fork does not have. The
                # preview is 120 chars of historical text the replay does
                # not reproduce byte-for-byte, so a distorted preview costs
                # less than a poisoned assertion — and one rule for all
                # four keys leaves no exemption to become the next bypass.
                bits.append(f"{k}={_fork_relative(v)[:120]}")
        # ⚠ THE RESULT LEAKED THE PATH THE ARGUMENTS NO LONGER DID.
        # Rewriting only `path`/`command` left the tool's OUTPUT saying
        # `Wrote 5202 chars to 'projects/<id>/x.py'` and
        # `-rw-r--r-- … /workspace/projects/<id>/x.json`, which is
        # exactly what the model copies into the check. Found by running
        # the fix end to end against the two real episodes instead of
        # trusting the unit test.
        res = _fork_relative(str(getattr(tc, "result", "") or ""))[:160]
        lines.append(f"  {i}. {getattr(tc, 'name', '?')}({'; '.join(bits)})"
                     f" -> {res}")
    return "\n".join(lines) or "  (no tool calls recorded)"


_FORBIDDEN_VALIDATOR_IMPORTS = (
    "subprocess", "socket", "requests", "urllib", "httpx", "curl_cffi",
    "shutil.rmtree", "os.remove", "os.unlink", "os.rmdir",
    # The first list named `subprocess` and not `os.system` — half a
    # class, which reads as a closed door and is not one. The sandbox is
    # the real containment (the docstring says so); a screen that names
    # some of a class and not the rest is worse than one that names none,
    # because it looks complete.
    "os.system", "os.popen", "os.execv", "os.spawn", "pty.spawn",
    "eval(", "exec(", "__import__", "compile(",
    "path.unlink", "os.rename", "os.replace", "os.truncate",
)


def validator_is_admissible(src: str) -> Tuple[bool, str]:
    """Static screen on a synthesised validator.

    A check that shells out, reaches the network, or deletes things is not
    a check — it is a second agent with no supervision. This is a cheap
    pre-filter; the sandbox (``network=none``, a forked workspace) is the
    real containment.
    """
    text = str(src or "")
    if not text.strip():
        return False, "empty"
    if len(text) > VALIDATOR_MAX_CHARS:
        return False, f"over {VALIDATOR_MAX_CHARS} chars"
    low = text.lower()
    for bad in _FORBIDDEN_VALIDATOR_IMPORTS:
        if bad in low:
            return False, f"uses {bad}"
    try:
        compile(text, "<validator>", "exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"
    return True, ""


async def synthesize_validator(traj, llm_client, *, model: str = "",
                               out: dict = None) -> str:
    """Ask the main model for an executable check for this episode.

    Returns "" when nothing admissible came back — an episode with no
    validator is simply not replayable, which is a better outcome than a
    validator nobody screened.

    ⚠ `out` EXISTS BECAUSE THIS IS THE FUNNEL'S BIGGEST LOSS AND NOBODY
    COULD SEE WHY. The A/A census measured 46 of 128 episodes dying here
    — more than every other rejection combined — and the reason was
    written to a log line and thrown away, so "the model returned
    nothing" and "the screen rejected what it returned" were
    indistinguishable at the only place the count is reported. Pass a
    dict and it comes back carrying `reason` and `raw_chars`; callers
    that do not care are unaffected.
    """
    def _note(reason: str, raw: str = "") -> str:
        if out is not None:
            out["reason"] = reason
            out["raw_chars"] = len(raw or "")
        return ""

    if llm_client is None:
        return _note("no_llm_client")
    # ⚠ ALL THREE SLOTS, NOT ONE. The first fix rewrote the trace and
    # left `request` and `final` interpolated raw — and MEASURED on the
    # 42 replayable episodes, the trace leaked a project path in 0 while
    # `final_response` leaked in 3 and `user_request` in 1. The final
    # reply is the worst channel of the three: it is where the agent
    # ANNOUNCES what it wrote ("The file `projects/<id>/message.md` has
    # been written"), so it names the deliverable's recorded location in
    # prose the synthesiser copies verbatim. Two of those files still
    # exist in the live sandbox, which the seeded fork COPIES, so a
    # check built on them passes with no agent run at all.
    # Cut before rewriting: the scan is linear in the value's length and
    # these slots are unbounded, while the rewrite is byte-identical on
    # every real value once the slice is well past the cap.
    prompt = _VALIDATOR_PROMPT.format(
        request=_fork_relative(str(getattr(traj, "user_request", ""))[:2000]),
        trace=_trace_for_prompt(traj),
        final=_fork_relative(str(getattr(traj, "final_response", ""))[:1200]),
    )
    try:
        payload = {
            "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
            "temperature": 0.1, "max_tokens": 1200, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if model:
            payload["model"] = model
        result = await llm_client.chat_completion(payload, is_background=True)
        text = (result.get("choices", [{}])[0].get("message", {})
                .get("content", "") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay: validator synthesis failed (%s)", exc)
        return _note(f"call_failed:{type(exc).__name__}")
    raw = text
    text = _strip_fence(text)
    ok, why = validator_is_admissible(text)
    if not ok:
        logger.info("replay: validator rejected (%s)", why)
        return _note(f"screen:{why}", raw)
    if out is not None:
        out["reason"] = ""
        out["raw_chars"] = len(raw or "")
    return text


def _strip_fence(text: str) -> str:
    t = str(text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if "```" in t:
            t = t.rsplit("```", 1)[0]
    return t.strip()


# ===========================================================================
# D2 — the replay executor
# ===========================================================================
#
# ⚠ A DEVIATION from IDE.md, recorded because it is deliberate. The plan
# says to run replays "via the existing `Dreamer.synthetic_self_play`
# machinery". That method is ~3,200 lines specialised for a different
# job — generating a challenge, retrying a solver, minting lessons,
# bench oracle verdicts — and every perturbation would have to be
# threaded through it as a new branch. Instead the executor is built
# directly on the §4CL S1 substrate (`isolated_replay_context` +
# `IsolatedRun.build_agent`), which is the part of self-play worth
# reusing and is now separately tested. Self-play is not touched.
#
# THE RULE THE WHOLE PHASE RESTS ON: re-execute FORWARD from the fork.
# A perturbation changes ONE thing and the agent then runs to the end on
# its own. Nothing is spliced into a recorded suffix.

CREDITS_FILENAME = "credits.jsonl"

VERDICT_MATTERED_POS = "mattered_pos"   # perturbation broke a passing run
VERDICT_MATTERED_NEG = "mattered_neg"   # perturbation fixed a failing run
VERDICT_NO_EFFECT = "no_effect"
VERDICT_ABSTAIN = "abstain"

#: Legs per arm, and the number is arithmetic rather than taste.
#:
#: For a genuinely NULL perturbation on a task with per-run pass
#: probability p (q = 1-p), the unanimity rule decides only when both
#: arms are internally consistent, and mislabels noise as an effect at::
#:
#:     false-`mattered` | decided  =  2·pⁿ·qⁿ / (pⁿ + qⁿ)²
#:
#:     p     n=2      n=3
#:     0.50  0.500    0.500
#:     0.70  0.262    0.135
#:     0.80  0.111    0.030
#:     0.90  0.024    0.003
#:
#: At n=2 the pre-registered specificity bar of 0.90 is UNREACHABLE
#: unless the replayed tasks are near-deterministic (p ≥ 0.815). n=3
#: brings p ≥ 0.75 inside the bar at the cost of 6 legs per spec instead
#: of 4 — which is the right trade for a corpus of 303 specs whose whole
#: value is that its labels are trustworthy.
#:
#: ⚠ It does not make the floor vanish. Nothing here measures p, so
#: `pass_rate` is reported on every credit row: a consumer that reads a
#: `mattered_*` without looking at it is reading a number whose error bar
#: it does not know.
DEFAULT_N_PAIRS = 3
#: Wall-clock for one leg. The outer deadline is shared across all legs
#: of a spec, and a BATCH deadline is shared across specs
#: (budget-is-a-deadline, not a duration — three durations that each look
#: reasonable are not a bounded night).
DEFAULT_LEG_TIMEOUT_S = 300.0
#: ⚠ ONE COMMAND MUST NOT BE ABLE TO EAT A WHOLE LEG. The clamp below
#: used to hand a single sandbox command everything left of the leg, and
#: a leg's budget is meant to cover MANY turns. MEASURED: a replayed
#: episode ran `python3 -m http.server 8080` — a foreground server that
#: never returns — and was handed `timeout -k 5s 569s` out of a 600 s
#: leg. The leg died ungradable, and the episode with it.
#:
#: The execute tool's own guard does not catch this and should not: it
#: blocks servers that DETACH (`… &` / setsid) and says so — "foreground
#: and short-lived commands are unaffected" — which is right for a live
#: turn, where a blocking server is the operator's problem. In a replay
#: leg nobody is watching, so the bound has to come from the budget.
#:
#: 120 s is generous against the measured distribution (legs run a p50 of
#: 26 s and a p90 of 127 s for ALL their commands and LLM turns together)
#: and leaves a timed-out command with most of the leg still to recover
#: in — which is what the original turn had.
REPLAY_MAX_CMD_S = 120.0
DEFAULT_SPEC_TIMEOUT_S = 1500.0
#: The whole batch. Its caller in `agent.py` wraps `run_batch` in an
#: hour, and the first version's own constants could not fit inside it:
#: 3 specs x 1500 s of fresh per-spec deadline, plus a self-test and a
#: negative-control leg per trajectory, is up to 5,400 s of `handle_chat`
#: alone. The bound would fire every time and the warning would blame the
#: spec deadline, which had held perfectly. The batch deadline is now
#: DERIVED and asserted against the caller's bound below.
DEFAULT_BATCH_TIMEOUT_S = 3300.0
#: Turn cap inside a replayed episode. The recorded episodes average 6.9
#: steps (median 3, max 37); 20 leaves headroom without letting a
#: replay spin to a real turn's cap.
DEFAULT_MAX_TURNS = 20


@dataclass
class ReplayLeg:
    """One execution of one arm."""
    arm: str                      # "control" | "perturbed"
    passed: Optional[bool] = None  # None = the leg did not produce a verdict
    reason: str = ""
    steps: int = 0
    duration_s: float = 0.0
    validator_exit: Optional[int] = None
    validator_output: str = ""
    #: Did the perturbation this leg was supposed to apply ACTUALLY
    #: apply? Always True on a control leg (there is nothing to apply).
    #:
    #: ⚠ This is the single most load-bearing field on the object. Every
    #: way a perturbation can silently fail — a withheld lesson that is
    #: no longer in the store, a step-deny whose index the free-running
    #: replay never reaches, a wrapper destroyed by a dispatch-miss
    #: rebuild — produces a perturbed arm BYTE-IDENTICAL to the control
    #: arm, and therefore a confident `no_effect` about a counterfactual
    #: that never happened. An unapplied perturbation is ungradable, not
    #: a null result.
    applied: bool = True


def _leg_is_gradable(leg: "ReplayLeg") -> bool:
    return leg.passed is not None and leg.applied


def _network_failure(text: str) -> bool:
    """A leg that failed because the CONTAINER has no network failed for a
    reason that has nothing to do with the perturbation under test. The
    triage excludes network-using episodes, so this is the belt: a
    workspace script that reaches out anyway must not be graded."""
    t = str(text or "").lower()
    return any(m in t for m in (
        "temporary failure in name resolution", "name or service not known",
        "network is unreachable", "could not resolve host",
        "connection refused", "no route to host", "errno -3", "errno 101",
    ))


# ── Perturbation, applied to the DATA ──────────────────────────────────

def apply_lesson_perturbation(memory_dir: Path, real_skill_memory, *,
                              withhold: str = "", inject: dict = None):
    """Build a fork-local playbook with the perturbation baked in, and
    return a REAL ``SkillMemory`` over it.

    The perturbation is applied to the STORE, not to the retrieval code
    path. That matters: filtering a rendered lesson string, or
    intercepting one of the two surfaces (``get_playbook_context`` for
    the volatile block, ``get_playbook_items`` for the bus tier), would
    be a lexical proxy for "this lesson was not available" and would miss
    whichever surface the interception did not cover. A store that simply
    does not contain the lesson is the property itself, and every ranking,
    dedup and quarantine rule downstream keeps working unchanged.

    The store is WRITABLE and lives in the fork, so a replay's own lesson
    writes land there and are discarded with the workspace.
    """
    from ..memory.skills import SkillMemory

    memory_dir.mkdir(parents=True, exist_ok=True)
    items = []
    try:
        items = list(real_skill_memory._load_playbook() or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay: could not read the playbook (%s)", exc)
    want = (withhold or "").strip().lower()
    if want:
        items = [it for it in items
                 if str(it.get("trigger", "")).strip().lower() != want]
    if inject:
        items = list(items) + [dict(inject)]
    sm = SkillMemory(memory_dir)
    sm.save_playbook(items)
    # ⚠ THE SIMULATION MARKER. `skill_memory.is_read_only` is the ONE
    # derivation the whole repo uses for "this turn is not real traffic"
    # — `agent.turn_origin`, the foresight predict hook, the metacog
    # competence write, and about nine other gates all read it. A plain
    # SkillMemory does not have it, so swapping one in here silently
    # re-armed two PRODUCTION corpora on every lesson-perturbation leg:
    # the per-domain competence prior the confidence composite reads on
    # real turns, and the foresight ledger + index. 59% of specs are
    # lesson perturbations, so that was the majority of every night.
    #
    # The store is genuinely writable (it is fork-local and thrown away);
    # the marker is about the TURN's population, not about this object's
    # mutability.
    sm.is_read_only = True
    return sm


class _WithholdingCollection:
    """A Chroma collection façade that drops one lesson from every query.

    ⚠ THE REASON THIS EXISTS. Withholding a lesson from the fork-local
    JSON playbook does NOT withhold it. Hydration is VECTOR-FIRST
    (`memory/skills.py::_playbook_items_and_branch`): it queries Chroma
    for `type="skill"` documents, and only then looks the trigger up in
    the JSON to pick a nicer rendering — falling back to the EMBEDDED
    DOCUMENT when the lookup misses. So a JSON-only withhold produced a
    perturbed arm in which the lesson was still in the prompt, merely
    rendered differently, and 59% of this engine's specs are that kind.

    The store has two surfaces and a perturbation has to cover both.
    """

    def __init__(self, inner, trigger: str):
        self._inner = inner
        self._want = (trigger or "").strip().lower()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _matches(self, meta, doc) -> bool:
        try:
            trig = str((meta or {}).get("trigger", "") or "").strip().lower()
            if trig and trig == self._want:
                return True
            # The vector path derives a trigger from the DOCUMENT when the
            # metadata has none — same function, so this is the property
            # rather than a substring proxy for it.
            from ..memory.skills import _extract_trigger_from_doc
            derived = str(_extract_trigger_from_doc(doc) or "").strip().lower()
            return bool(derived and derived == self._want)
        except Exception:  # noqa: BLE001
            return False

    def query(self, *args, **kwargs):
        res = self._inner.query(*args, **kwargs)
        if not self._want or not isinstance(res, dict):
            return res
        try:
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0] or [{}] * len(docs)
            keep = [i for i, (d, m) in enumerate(zip(docs, metas))
                    if not self._matches(m, d)]
            if len(keep) == len(docs):
                return res
            out = dict(res)
            # Every parallel list is filtered by the SAME index set —
            # dropping one and not another silently misaligns
            # documents/distances/metadatas, which is worse than not
            # filtering at all.
            for key in ("documents", "distances", "metadatas", "ids",
                        "embeddings", "uris", "data"):
                col = res.get(key)
                if isinstance(col, list) and col and isinstance(col[0], list):
                    out[key] = [[col[0][i] for i in keep
                                 if i < len(col[0])]]
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("replay: vector withhold failed (%s) — refusing "
                           "to run a leg whose perturbation did not apply",
                           exc)
            raise


def withholding_memory(real_memory_system, trigger: str):
    """A read-only vector façade that also drops ``trigger`` from every
    similarity query. Returns the plain read-only façade when there is
    nothing to withhold."""
    from .isolation import ReadOnlyVectorMemory

    ro = ReadOnlyVectorMemory(real_memory_system)
    if not (trigger or "").strip():
        return ro
    object.__setattr__(ro, "collection",
                       _WithholdingCollection(ro.collection, trigger))
    return ro


def _withheld_was_present(real_skill_memory, trigger: str) -> bool:
    """A withhold whose lesson is not in the store asks nothing. The spec
    builder only proposes triggers the episode really hydrated, but the
    store moves between the recording and the replay — a lesson can be
    pruned or quarantined in between."""
    want = (trigger or "").strip().lower()
    if not want:
        return False
    try:
        return any(str(it.get("trigger", "")).strip().lower() == want
                   for it in (real_skill_memory._load_playbook() or []))
    except Exception:  # noqa: BLE001
        return False


# ── One leg ────────────────────────────────────────────────────────────

_REPLAY_FRAMING = (
    "### REPLAY\n"
    "Do the following, working in your sandbox workspace. Use tools; do "
    "not answer from memory. Stop as soon as the task is done.\n\n"
)


#: Sentinel: "take the live sandbox". `None` means "start empty", which
#: is a legitimate mode (D4's negative control uses it) and was ALSO the
#: accidental default at every call site — so every leg replayed a turn
#: about files that were not there.
USE_LIVE_WORKSPACE = "__live__"


def _cmd_budget(leg_budget: float, elapsed: float) -> int:
    """Seconds a SINGLE sandbox command may run inside a leg.

    Two bounds, and the tighter one wins: what is LEFT of the leg (a
    command outliving its leg blocks an executor thread the leg's
    cancellation cannot reach), and `REPLAY_MAX_CMD_S` (a leg's budget
    is for many turns, so no one command may spend it all).

    The 30 s floor is deliberate and is NOT a third bound: when the leg
    is nearly over the command still gets a usable slice, because
    handing it one second produces a guaranteed-useless timeout rather
    than a shorter one.
    """
    remaining = leg_budget - elapsed - 30
    return int(max(30.0, min(REPLAY_MAX_CMD_S, remaining)))


async def run_leg(context, spec: dict, *, arm: str, validator: str,
                  source_workspace=USE_LIVE_WORKSPACE, deadline: float = None,
                  leg_timeout_s: float = DEFAULT_LEG_TIMEOUT_S,
                  max_turns: int = DEFAULT_MAX_TURNS) -> ReplayLeg:
    """Re-execute one episode once, under one arm, and grade it.

    ``deadline`` is a monotonic instant shared by every leg of a spec —
    a budget is a deadline, not a duration, and per-leg durations that
    each look reasonable add up to a night.
    """
    import asyncio
    import time as _time

    leg = ReplayLeg(arm=arm)
    started = _time.monotonic()
    if deadline is not None and started >= deadline:
        leg.reason = "spec deadline passed before this leg started"
        return leg

    budget = leg_timeout_s
    if deadline is not None:
        budget = min(budget, max(1.0, deadline - started))

    perturb = str(spec.get("perturbation") or "")
    target = str(spec.get("target") or "")
    perturbed = (arm == "perturbed")
    # ⚠ The recorded turn ran against a POPULATED sandbox. Replaying it
    # into an empty tempdir means every task whose success is defined by
    # files that already existed fails in the control arm — which the
    # self-test then reads as "the validator disagrees with the
    # recording" and discards the episode. The corpus does not shrink to
    # the reproducible subset; it shrinks to the subset where a
    # wrong-for-the-right-reason validator agrees with a
    # wrong-for-the-wrong-reason leg.
    if source_workspace == USE_LIVE_WORKSPACE:
        source_workspace = getattr(context, "sandbox_dir", None)

    try:
        async with isolated_replay_context(
                context, network="none", label="dream-replay",
                source_workspace=source_workspace) as run:
            if not run.fork_complete:
                leg.reason = f"fork incomplete: {run.fork_reason}"
                return leg

            iso = run.context
            # ── perturbations, applied before the agent is built
            if perturb in (PERTURB_LESSON_WITHHOLD, PERTURB_LESSON_INJECT):
                _withhold = (target if perturbed and
                             perturb == PERTURB_LESSON_WITHHOLD else "")
                if _withhold and not _withheld_was_present(
                        getattr(context, "skill_memory", None), _withhold):
                    # Measured 2026-08-22: 88.5% of withhold specs target
                    # a lesson that has since been pruned or quarantined.
                    # Control and perturbed stores would be identical and
                    # the verdict would be pure sampling noise wearing a
                    # `no_effect` label.
                    leg.applied = False
                    leg.reason = (f"the withheld lesson {_withhold!r} is no "
                                  f"longer in the store — nothing to remove")
                    return leg
                iso.skill_memory = apply_lesson_perturbation(
                    Path(iso.memory_dir),
                    getattr(context, "skill_memory", None),
                    withhold=_withhold,
                    inject=(dict(spec.get("inject") or {})
                            if perturbed and
                            perturb == PERTURB_LESSON_INJECT else None),
                )
                # …and the OTHER surface. Hydration is vector-first; the
                # JSON is only consulted for rendering.
                iso.memory_system = withholding_memory(
                    getattr(context, "memory_system", None), _withhold)
            # ⚠ THE CONTROL ARM MUST BE THE RECORDED CONDITION. The
            # isolation recipe nulls `verifier` (it carries its own
            # llm_client reference), but the live agent constructs one
            # UNCONDITIONALLY at boot and the recorded episodes all ran
            # with it — it is priority 1 and 3 in `resolve_turn_outcome`,
            # i.e. the top signal that produced the very
            # `recorded_outcome` the self-test compares against. A
            # verifier-less control is not "the turn as recorded", it is
            # a third condition, and every verdict would answer a
            # question about a world that never existed.
            #
            # So: control ALWAYS has a verifier, and `verify_toggle`
            # REMOVES it on the perturbed arm — which is what "toggle"
            # says and the opposite of what the first version did.
            _want_verifier = not (perturb == PERTURB_VERIFY_TOGGLE
                                  and perturbed)
            if _want_verifier:
                try:
                    from .verifier import Verifier as _V
                    iso.verifier = _V(llm_client=iso.llm_client)
                except Exception as exc:  # noqa: BLE001
                    leg.reason = f"verifier unavailable: {exc}"
                    return leg

            # ⚠ Clamp the per-command budget to what is LEFT of the leg,
            # at the SANDBOX. `tools/execute.py` passes a module constant
            # (600 s) — twice a leg's whole budget — via
            # `asyncio.to_thread`, and cancelling the leg's coroutine
            # cannot stop that thread. The leg would return, the container
            # would be force-removed and the workspace rmtree'd while a
            # process was still writing into it, leaving a blocked
            # executor thread (shared with every live-turn `to_thread`)
            # for up to 660 s. Clamping at the sandbox works regardless of
            # which caller passes what.
            try:
                if run.sandbox is not None:
                    run.sandbox.max_exec_timeout = _cmd_budget(
                        budget, _time.monotonic() - started)
            except Exception:  # noqa: BLE001
                pass

            # ── A challenge-backed seed brings its own fixture. Run it
            # BEFORE the agent so the task has something to work on; a
            # setup that fails makes the leg ungradable rather than a
            # failure charged to the agent.
            _setup = str(spec.get("setup_script") or "")
            if _setup and run.sandbox is not None:
                spath = run.workspace / ".setup.py"
                await asyncio.to_thread(spath.write_text, _setup)
                s_out, s_code = await asyncio.to_thread(
                    run.sandbox.execute, "python3 .setup.py", 60)
                if s_code != 0:
                    leg.reason = (f"setup script failed (exit {s_code}): "
                                  f"{str(s_out)[:200]}")
                    return leg

            _ablated = ()
            if perturb == PERTURB_TOOL_ABLATE and perturbed:
                _ablated = tuple(t.strip() for t in target.split(",")
                                 if t.strip())
                if not _ablated:
                    # An empty target ablates NOTHING, so the perturbed
                    # arm is byte-identical to control and the verdict is
                    # a confident `no_effect` about a counterfactual that
                    # never happened. Every other kind has an
                    # applied-guard for its premise-false case
                    # (`_withheld_was_present`, `_deny_state["fired"]`);
                    # this one had a guard only for the PARTIAL case.
                    leg.applied = False
                    leg.reason = ("tool_ablate spec carries no target — "
                                  "nothing to ablate")
                    return leg

            agent = run.build_agent(extra_forbidden=_ablated)
            _missing = set(_ablated) - set(run.ablated_tools)
            if _ablated and _missing:
                # A tool the agent never had cannot be ablated, and a leg
                # whose perturbation removed only SOME of its targets is
                # the control arm wearing a label — the survivor is
                # exactly the capability the ablation was meant to take.
                leg.applied = False
                leg.reason = (f"ablation incomplete — {sorted(_missing)} "
                              f"of {sorted(_ablated)} were not available")
                return leg
            agent.max_turns_override = max_turns
            _deny_state = None
            if perturb == PERTURB_STEP_DENY and perturbed:
                _deny_state = _install_step_deny(
                    agent, str(spec.get("target") or ""))

            body = {"model": str(spec.get("model") or "replay"),
                    "messages": [{"role": "user", "content":
                                  _REPLAY_FRAMING
                                  + str(spec.get("user_request") or "")}]}
            req_id = f"replay-{spec.get('spec_id', '')[:10]}-{arm}"
            try:
                await asyncio.wait_for(
                    agent.handle_chat(body, background_tasks=None,
                                      request_id=req_id),
                    timeout=budget)
            except asyncio.TimeoutError:
                leg.reason = f"leg exceeded {budget:.0f}s"
                leg.duration_s = _time.monotonic() - started
                return leg
            leg.steps = sum(1 for m in body.get("messages", [])
                            if isinstance(m, dict) and m.get("role") == "tool")
            if _deny_state is not None and not _deny_state.get("fired"):
                # The replay free-runs from a different starting state, so
                # it need not emit the recorded call at all. When it does
                # not, the perturbed arm IS the control arm.
                leg.applied = False
                leg.reason = (f"the step-deny never fired — the replay "
                              f"never called {spec.get('target') or '?'}")
                return leg

            # ── grade it
            vpath = run.workspace / VALIDATOR_FILENAME
            await asyncio.to_thread(vpath.write_text, validator)
            out, code = await asyncio.to_thread(
                run.sandbox.execute,
                f"python3 {VALIDATOR_FILENAME}", VALIDATOR_TIMEOUT_S)
            leg.validator_exit = int(code)
            _full = str(out or "")
            # ⚠ Classify on the FULL output, store the truncation. A
            # chatty validator's network error lives past 1,000 chars and
            # was being graded as a genuine failure.
            leg.validator_output = _full[:1000]
            if "[SANDBOX INFRA ERROR" in _full:
                # docker.execute never raises; it RETURNS this banner. It
                # became passed=False and was charged to the agent —
                # the §4AO label-noise class, and the exact mistake
                # self-play already paid for.
                leg.reason = "sandbox infra fault during validation"
            elif _network_failure(_full):
                leg.reason = "validator needed a network the fork does not have"
            elif code in (0, 1):
                leg.passed = (code == 0)
            else:
                # Exit 2 is the validator's own "I cannot check this from
                # the filesystem"; anything else is a crash. Neither is a
                # verdict about the agent.
                leg.reason = f"validator inconclusive (exit {code})"
    except IsolationUnavailable as exc:
        leg.reason = f"isolation unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        leg.reason = f"{type(exc).__name__}: {exc}"
    leg.duration_s = _time.monotonic() - started
    return leg


#: Marker the step-deny perturbation returns instead of executing a call.
#: Recognised as SYNTHETIC by `foresight.is_synthetic_result` for the same
#: reason the pre-flight steer's is: the call never ran, so it carries no
#: transition information and its label would be inverted.
STEP_DENY_MARKER = "SYSTEM PREFLIGHT — replay:"


def _install_step_deny(agent, tool_name: str) -> dict:
    """Deny the FIRST call to ``tool_name``, then free-run. Returns the
    state dict so the caller can check whether it ever fired.

    ⚠ A DEVIATION from IDE.md, and the reason is the whole point of the
    phase. The plan calls this ``tool_swap(step_call -> alternative_call)``
    — but nothing in a recording says what the alternative SHOULD be, so
    something would have to invent one, and an invented alternative puts
    an ungrounded generator inside a measurement whose entire value is
    that it is execution-grounded. Denying the recorded call asks the same
    counterfactual with nothing invented.

    ⚠ CONTENT, NOT POSITION. The first version denied the call at the
    recorded index. The replay free-runs from a different starting state,
    so it does not emit the recorded sequence: recorded
    `[file_system, file_system, execute]` against replayed
    `[file_system, execute, …]` means index 2 is not the execute call,
    nothing is ever denied, and the perturbed arm is byte-identical to
    the control arm — a confident `no_effect` about a counterfactual that
    never happened.

    ⚠ AND IT MUST SURVIVE THE REBUILD. `GhostAgent._rebuild_available_tools`
    fires on any tool-name miss (its own docstring says models hallucinate
    variants routinely) and REPLACES the dispatch dict from the registry,
    which would drop these closures silently. So the denial is ALSO
    recorded on the context and re-applied from there.
    """
    state = {"fired": False, "tool": str(tool_name or "")}

    def _wrap(name, fn):
        async def _inner(**kwargs):
            if state["tool"] and name == state["tool"] and not state["fired"]:
                state["fired"] = True
                return (f"{STEP_DENY_MARKER} '{name}' is unavailable for "
                        f"this task. Accomplish it another way, or say why "
                        f"you cannot.")
            return await fn(**kwargs)
        return _inner

    def _apply(a):
        a.available_tools = {k: _wrap(k, v)
                             for k, v in (a.available_tools or {}).items()}

    _apply(agent)
    # Re-arm after a dispatch-miss rebuild. `_rebuild_available_tools`
    # returns the fresh dict; wrapping the bound method is the only hook
    # that runs on every rebuild without editing agent.py.
    _orig_rebuild = getattr(agent, "_rebuild_available_tools", None)
    if callable(_orig_rebuild):
        def _rebuild_and_rearm():
            out = _orig_rebuild()
            _apply(agent)
            return agent.available_tools
        agent._rebuild_available_tools = _rebuild_and_rearm
    return state


async def run_validator_only(context, validator: str, *,
                             leg_timeout_s: float = DEFAULT_LEG_TIMEOUT_S
                             ) -> ReplayLeg:
    """Run the validator against an EMPTY fork with NO agent run.

    The negative control for a synthesised check. Its only job is to
    answer "does this validator discriminate at all", and the answer for
    `sys.exit(0)` is no.
    """
    import asyncio

    leg = ReplayLeg(arm="negative_control")
    try:
        async with isolated_replay_context(
                context, network="none", label="dream-negctl",
                source_workspace=None) as run:
            vpath = run.workspace / VALIDATOR_FILENAME
            await asyncio.to_thread(vpath.write_text, validator)
            out, code = await asyncio.to_thread(
                run.sandbox.execute,
                f"python3 {VALIDATOR_FILENAME}", VALIDATOR_TIMEOUT_S)
            leg.validator_exit = int(code)
            leg.validator_output = str(out or "")[:1000]
            if "[SANDBOX INFRA ERROR" in leg.validator_output:
                leg.reason = "sandbox infra fault during the negative control"
            elif code in (0, 1):
                leg.passed = (code == 0)
            else:
                leg.reason = f"validator inconclusive (exit {code})"
    except IsolationUnavailable as exc:
        leg.reason = f"isolation unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        leg.reason = f"{type(exc).__name__}: {exc}"
    return leg


# ── The paired verdict ─────────────────────────────────────────────────

def decide_verdict(control: List[ReplayLeg],
                   perturbed: List[ReplayLeg]) -> Tuple[str, str]:
    """(verdict, why) from the paired legs.

    Keep-first / abstain-on-tie, applied strictly:

      * any leg that produced no verdict (timeout, infra fault, an
        inconclusive validator) ⇒ ABSTAIN. A missing leg is not a null
        result, and the corpus is far too small to absorb guesses;
      * legs WITHIN an arm that disagree ⇒ ABSTAIN. That is the
        stochasticity the pairs exist to detect, and calling it an effect
        is how a label source becomes a noise source (§4BE);
      * arms agree ⇒ no effect;
      * arms differ consistently ⇒ the perturbation mattered, signed by
        which direction.
    """
    if not control or not perturbed:
        return VERDICT_ABSTAIN, "an arm ran no legs"
    ungradable = [l for l in list(control) + list(perturbed)
                  if not _leg_is_gradable(l)]
    if ungradable:
        return VERDICT_ABSTAIN, ("ungradable leg(s): "
                                 + "; ".join(sorted({l.reason for l
                                                     in ungradable}))[:200])
    c = {l.passed for l in control}
    p = {l.passed for l in perturbed}
    if len(c) > 1 or len(p) > 1:
        return VERDICT_ABSTAIN, ("legs within an arm disagreed — the run is "
                                 "stochastic at this task, so a difference "
                                 "between arms cannot be attributed")
    c_pass, p_pass = c.pop(), p.pop()
    if c_pass == p_pass:
        return VERDICT_NO_EFFECT, ""
    return ((VERDICT_MATTERED_POS if c_pass else VERDICT_MATTERED_NEG), "")


#: Which perturbations REMOVE something (so "control passed, perturbed
#: failed" means the removed thing HELPED) and which ADD something (so
#: the same verdict means the added thing HURT). Without this a consumer
#: summing `mattered_pos` across kinds is adding opposite facts.
_REMOVES = (PERTURB_LESSON_WITHHOLD, PERTURB_STEP_DENY,
            PERTURB_VERIFY_TOGGLE, PERTURB_TOOL_ABLATE)
#: The other half of the partition. Both are enumerated so an
#: unclassified kind returns "" instead of silently inheriting a branch.
_ADDS = (PERTURB_LESSON_INJECT,)


def _pass_rate(legs) -> Optional[float]:
    """Observed pass rate over gradable legs, or None."""
    vals = [1.0 if l.passed else 0.0 for l in legs if l.passed is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _noise_floor(pass_rate: Optional[float], n: int) -> Optional[float]:
    """P(a NULL perturbation reads as `mattered_*` | the rule decided).

    The paired unanimity rule cannot tell a stochastic flip from an
    effect; it can only make the flip rarer. This is how rare, at the
    observed pass rate — the number that says whether a `mattered_*` on
    this task is worth anything."""
    if pass_rate is None:
        return None
    p = max(0.0, min(1.0, float(pass_rate)))
    q = 1.0 - p
    pn, qn = p ** n, q ** n
    denom = (pn + qn) ** 2
    if denom <= 0:
        return None
    return round(2.0 * pn * qn / denom, 4)


def _verdict_sign(perturbation: str, verdict: str) -> str:
    """"helped" | "hurt" | "" — what the verdict says about the THING the
    perturbation acted on, normalised across kinds."""
    if verdict not in (VERDICT_MATTERED_POS, VERDICT_MATTERED_NEG):
        return ""
    # ⚠ FAIL CLOSED. `perturbation in _REMOVES` sent everything unlisted
    # down the ADDITIVE branch, so an unregistered string — a typo, a
    # different case, an empty value, a kind someone adds and forgets to
    # classify — reported the exact OPPOSITE sign with full confidence,
    # and `credit_stats` sums `by_sign` across kinds. An unclassified
    # perturbation has no sign; saying so is the only honest answer.
    if perturbation in _REMOVES:
        removes = True
    elif perturbation in _ADDS:
        removes = False
    else:
        return ""
    broke_it = (verdict == VERDICT_MATTERED_POS)   # control passed, pert failed
    return "helped" if (removes == broke_it) else "hurt"


async def run_spec(context, spec: dict, *, validator: str,
                   source_workspace=USE_LIVE_WORKSPACE,
                   n_pairs: int = DEFAULT_N_PAIRS,
                   spec_timeout_s: float = DEFAULT_SPEC_TIMEOUT_S,
                   leg_timeout_s: float = DEFAULT_LEG_TIMEOUT_S,
                   batch_deadline: float = None,
                   write: bool = True) -> Dict[str, Any]:
    """Run one spec's paired legs and record the verdict.

    Control legs first: if the control arm is already ungradable there is
    nothing to compare against, and running the perturbed arm anyway
    would spend the night's budget producing abstains.
    """
    import time as _time

    # The spec's own deadline, CLAMPED to the batch's. A fresh per-spec
    # budget is what let three specs outrun the hour their caller allows.
    deadline = _time.monotonic() + float(spec_timeout_s)
    if batch_deadline is not None:
        deadline = min(deadline, float(batch_deadline))
    control: List[ReplayLeg] = []
    for _ in range(max(1, int(n_pairs))):
        control.append(await run_leg(
            context, spec, arm="control", validator=validator,
            source_workspace=source_workspace, deadline=deadline,
            leg_timeout_s=leg_timeout_s))
    perturbed: List[ReplayLeg] = []
    if all(_leg_is_gradable(l) for l in control):
        for _ in range(max(1, int(n_pairs))):
            perturbed.append(await run_leg(
                context, spec, arm="perturbed", validator=validator,
                source_workspace=source_workspace, deadline=deadline,
                leg_timeout_s=leg_timeout_s))

    verdict, why = decide_verdict(control, perturbed)
    rec = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "spec_id": str(spec.get("spec_id") or ""),
        "trajectory_id": str(spec.get("trajectory_id") or ""),
        "perturbation": str(spec.get("perturbation") or ""),
        "target": str(spec.get("target") or ""),
        "fork_step": int(spec.get("fork_step") or 0),
        "recorded_outcome": str(spec.get("recorded_outcome") or ""),
        "n_pairs": int(n_pairs),
        "control_pass": [l.passed for l in control],
        "pert_pass": [l.passed for l in perturbed],
        # Whether the perturbation actually applied on every perturbed
        # leg. False ⇒ the arms were identical and the verdict is about
        # nothing; `decide_verdict` already abstains, this is the record.
        "applied": all(l.applied for l in perturbed) if perturbed else False,
        # The observed per-run pass rate across BOTH arms. The paired
        # rule's false-`mattered` floor is a function of exactly this
        # (2·pⁿ·qⁿ/(pⁿ+qⁿ)²), and nothing else measures it — so a
        # consumer reading a `mattered_*` without it is reading a number
        # whose error bar it does not know.
        "pass_rate": _pass_rate(control + perturbed),
        "noise_floor": _noise_floor(_pass_rate(control + perturbed),
                                    int(n_pairs)),
        "verdict": verdict,
        # The SIGN of `mattered_pos` depends on the perturbation, and a
        # consumer that aggregates across kinds without it gets nonsense:
        # for `lesson_withhold`, `step_deny`, `verify_toggle` and
        # `tool_ablate` — all of which REMOVE something — pos = "the thing
        # helped"; for `lesson_inject`, which ADDS, pos = "it hurt".
        "sign": _verdict_sign(str(spec.get("perturbation") or ""), verdict),
        "why": why,
        "validator_hash": hashlib.sha1(
            (validator or "").encode("utf-8", "replace")).hexdigest()[:16],
        "duration_s": round(sum(l.duration_s for l in control + perturbed), 1),
        "steps": [l.steps for l in control + perturbed],
    }
    if write:
        write_credits([rec])
    return rec


def write_credits(records: List[Dict[str, Any]], home: str = None) -> int:
    d = _state_dir(home)
    if d is None or not records:
        return 0
    path = d / CREDITS_FILENAME
    blob = "\n".join(json.dumps(r, ensure_ascii=False)
                     for r in records) + "\n"
    with _WRITE_LOCK:
        try:
            d.mkdir(parents=True, exist_ok=True)
            try:
                if (path.stat().st_size + len(blob.encode("utf-8"))
                        > _LEDGER_MAX_BYTES):
                    os.replace(str(path), str(path) + ".1")
            except FileNotFoundError:
                pass
            with path.open("a", encoding="utf-8") as f:
                f.write(blob)
                f.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("replay: could not write credits (%s)", exc)
            return 0
    return len(records)


def iter_credits(home: str = None) -> Iterator[Dict[str, Any]]:
    d = _state_dir(home)
    if d is None:
        return
    path = d / CREDITS_FILENAME
    for p in (Path(str(path) + ".1"), path):
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
        except OSError:
            continue


def credit_stats(home: str = None) -> Dict[str, Any]:
    """The learning-health surface. Reports the ABSTAIN rate as
    prominently as the verdicts — a replay engine whose output is mostly
    abstains is not producing labels, and an aggregate that hides that
    reads as throughput."""
    rows = list(iter_credits(home))
    if not rows:
        return {"present": False,
                "reason": "no replay verdicts yet — D3's idle phase writes "
                          "them; until then Dream is a producer of nothing"}
    counts: Dict[str, int] = {}
    # ⚠ BY PERTURBATION KIND. `mattered_pos` means "the thing helped" for
    # a perturbation that REMOVES and "the thing hurt" for one that ADDS —
    # summing them across kinds adds opposite facts. The normalised
    # `sign` field is what a consumer should read.
    by_kind: Dict[str, Dict[str, int]] = {}
    by_sign: Dict[str, int] = {}
    unapplied = 0
    noisy = 0
    for r in rows:
        v = str(r.get("verdict") or "?")
        counts[v] = counts.get(v, 0) + 1
        k = str(r.get("perturbation") or "?")
        by_kind.setdefault(k, {})[v] = by_kind.setdefault(k, {}).get(v, 0) + 1
        sign = str(r.get("sign") or "")
        if sign:
            by_sign[sign] = by_sign.get(sign, 0) + 1
        if r.get("applied") is False:
            unapplied += 1
        nf = r.get("noise_floor")
        if (v in (VERDICT_MATTERED_POS, VERDICT_MATTERED_NEG)
                and isinstance(nf, (int, float)) and nf > 0.10):
            noisy += 1
    decisive = sum(counts.get(k, 0) for k in
                   (VERDICT_MATTERED_POS, VERDICT_MATTERED_NEG,
                    VERDICT_NO_EFFECT))
    return {
        "present": True,
        "verdicts": len(rows),
        "by_verdict": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "by_kind": by_kind,
        "by_sign": by_sign,
        "decisive": decisive,
        "abstain_rate": round(counts.get(VERDICT_ABSTAIN, 0) / len(rows), 3),
        # Perturbations that did not apply. These already abstain, but the
        # COUNT is the health signal: a high number means the specs are
        # stale (their lessons pruned, their steps never re-emitted), not
        # that the world is null.
        "unapplied": unapplied,
        # `mattered_*` verdicts whose own noise floor exceeds the
        # pre-registered 0.10 specificity bar — i.e. verdicts a consumer
        # must not act on, sitting inside the same aggregate as the ones
        # it can.
        "mattered_above_noise_floor": (
            counts.get(VERDICT_MATTERED_POS, 0)
            + counts.get(VERDICT_MATTERED_NEG, 0) - noisy),
        "mattered_below_noise_floor": noisy,
        "specs": len({r.get("spec_id") for r in rows}),
        "last_ts": max((str(r.get("ts") or "") for r in rows), default=""),
    }


__all__ = [
    "CONSUMER_READ", "CONSUMER_CREDIT", "REPLAYABLE_TOOLS",
    "PERTURB_KINDS", "PERTURB_LESSON_WITHHOLD", "PERTURB_LESSON_INJECT",
    "PERTURB_STEP_DENY", "PERTURB_VERIFY_TOGGLE", "PERTURB_TOOL_ABLATE",
    "REJECT_KIND", "REJECT_UNDECIDED", "REJECT_THIN", "REJECT_UNSAFE",
    "REJECT_TOOL", "REJECT_NO_REQUEST",
    "Triage", "triage", "EpisodeSource", "ReplaySpec", "build_specs",
    "REJECT_NO_ARTIFACT",
    "write_specs", "iter_specs", "known_spec_ids",
    # D1
    "VALIDATOR_FILENAME", "VALIDATOR_MAX_CHARS", "VALIDATOR_TIMEOUT_S",
    "validator_is_admissible", "synthesize_validator",
    # D2
    "ReplayLeg", "STEP_DENY_MARKER", "CREDITS_FILENAME",
    "VERDICT_MATTERED_POS", "VERDICT_MATTERED_NEG", "VERDICT_NO_EFFECT",
    "VERDICT_ABSTAIN", "DEFAULT_N_PAIRS", "DEFAULT_LEG_TIMEOUT_S",
    "DEFAULT_SPEC_TIMEOUT_S", "DEFAULT_BATCH_TIMEOUT_S",
    "DEFAULT_MAX_TURNS",
    "apply_lesson_perturbation", "withholding_memory",
    "run_leg", "run_validator_only", "decide_verdict", "run_spec",
    "_verdict_sign", "USE_LIVE_WORKSPACE",
    "write_credits", "iter_credits", "credit_stats",
    # D3
    "DEFAULT_BATCH", "preflight", "plan_batch", "run_batch",
    "batch_summary",
]


# ===========================================================================
# D3 — the nightly batch
# ===========================================================================

#: Specs per night. Deliberately small: a spec is 2*n_pairs full solve
#: loops, and the corpus is 67 episodes / ~222 specs — a batch that
#: drains it in a week would spend a week of idle inference re-measuring
#: a fixed set. Raise it only after D4 says the verdicts are worth
#: having.
DEFAULT_BATCH = 3


def _enabled() -> bool:
    return os.getenv("GHOST_DREAM_REPLAY", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _batch_size() -> int:
    try:
        raw = os.getenv("GHOST_DREAM_REPLAY_BATCH", "").strip()
        return max(1, int(raw)) if raw else DEFAULT_BATCH
    except (TypeError, ValueError):
        return DEFAULT_BATCH


#: §4U preflight floors. A replay batch spawns a fresh container per leg
#: (mem_limit 4g) on a 38.7 GB box where llama-server holds ~22.3 GB
#: wired, so what matters is whether there is room for one more.
MIN_FREE_MB = 1500.0
#: The journal's operational note is "abort a run if swap_free < 250MB".
#:
#: ⚠ THAT RULE ONLY MEANS SOMETHING WHEN SWAP EXISTS. macOS allocates
#: swap dynamically: on a box that has never needed it, `swapusage total`
#: is 0.00M and therefore `free` is 0 too — which the first version of
#: this gate read as maximally starved and refused. Measured on this box
#: 2026-08-22 with 17.1 GB RAM available and llama-server up: `total =
#: 0.00M, used = 0.00M, free = 0.00M`, i.e. the HEALTHIEST possible
#: state, and the preflight called it "only 0 MB swap free".
#:
#: Same shape as §4BR: a floor applied to the wrong statistic. Swap
#: headroom is a signal about a box already under pressure; with no swap
#: allocated the question is answered by available RAM alone.
MIN_SWAP_FREE_MB = 250.0
#: Free space on the temp filesystem. A batch is 15 forks and nothing
#: bounds what a replayed turn writes into one.
MIN_DISK_FREE_MB = 2048.0
#: Seconds the daemon probe waits. Short on purpose: the answer to "is
#: docker answering" is either immediate or no.
DOCKER_PING_TIMEOUT_S = 5


def preflight() -> Tuple[bool, str]:
    """The §4U gate, in-process, for a batch that will run unattended.

    `scripts/preflight_longrun.py` is an operator CLI for a launch a human
    is about to type; a nightly batch has no human, so the checks it can
    actually make itself are the resource ones — and those are the ones
    that turned an overnight run into a swap-thrashed box before.

    Says what it checked and what it could not. A preflight that cannot
    read a precondition must report that, not pass.
    """
    notes = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        free_mb = vm.available / (1024 * 1024)
        swap_mb = sw.free / (1024 * 1024)
        if free_mb < MIN_FREE_MB:
            return False, (f"only {free_mb:.0f} MB available (floor "
                           f"{MIN_FREE_MB:.0f})")
        swap_total_mb = sw.total / (1024 * 1024)
        if swap_total_mb > 0 and swap_mb < MIN_SWAP_FREE_MB:
            return False, (f"only {swap_mb:.0f} MB swap free of "
                           f"{swap_total_mb:.0f} MB allocated (floor "
                           f"{MIN_SWAP_FREE_MB:.0f}) — the box is already "
                           f"swapping and a batch spawns a container per leg")
        notes.append(
            f"{free_mb:.0f} MB free"
            + (f", {swap_mb:.0f}/{swap_total_mb:.0f} MB swap"
               if swap_total_mb > 0 else ", no swap allocated"))
    except ImportError:
        # ⚠ NOT a pass. The docstring's own rule: a preflight that cannot
        # read a precondition must REPORT that, not clear the launch.
        return False, ("memory pressure UNCHECKED (no psutil) — refusing "
                       "to run a container-per-leg batch blind")
    except Exception as exc:  # noqa: BLE001
        return False, f"could not read memory pressure ({exc})"

    # Disk: the batch's whole footprint is temp directories, and nothing
    # bounds what a replayed turn writes into one.
    try:
        import shutil as _sh
        import tempfile as _tf
        free_disk_mb = _sh.disk_usage(_tf.gettempdir()).free / (1024 * 1024)
        if free_disk_mb < MIN_DISK_FREE_MB:
            return False, (f"only {free_disk_mb:.0f} MB free on the temp "
                           f"filesystem (floor {MIN_DISK_FREE_MB:.0f})")
        notes.append(f"{free_disk_mb:.0f} MB temp disk")
    except Exception as exc:  # noqa: BLE001
        return False, f"could not read temp-disk space ({exc})"

    # ⚠ The DAEMON, not the package. `find_spec("docker")` passes with
    # OrbStack stopped — which is the actual failure on this box, and it
    # is the one that made every leg raise while the batch burned its
    # specs anyway.
    try:
        import docker as _docker
        # ⚠ BOUNDED. `run_batch` calls this between trajectories, and the
        # default client retries a dead socket for tens of seconds — a
        # preflight that costs 30 s per call is itself a resource problem
        # on a job whose whole point is to stand down cheaply.
        client = _docker.from_env(timeout=DOCKER_PING_TIMEOUT_S)
        try:
            client.ping()
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        return False, "no docker package — a replay leg needs a sandbox"
    except Exception as exc:  # noqa: BLE001
        return False, (f"the docker daemon is not answering "
                       f"({type(exc).__name__}) — every leg would raise")
    notes.append("docker answering")

    # ⚠ THE BASE IMAGE. A container created with `network=none` cannot
    # provision itself: `apt-get update` has no interfaces. So a routine
    # bump of the provisioning marker (v1→v5 already happened) turns
    # every leg into a raise, the backoff arms, and the batch burns its
    # specs — with nothing anywhere naming the image. This is the one
    # precondition an unattended job can neither detect nor repair once
    # it has started.
    try:
        import docker as _docker2
        cl = _docker2.from_env(timeout=DOCKER_PING_TIMEOUT_S)
        try:
            cl.images.get("ghost-agent-base:latest")
        finally:
            try:
                cl.close()
            except Exception:  # noqa: BLE001
                pass
        notes.append("base image present")
    except Exception as exc:  # noqa: BLE001
        return False, (f"the provisioned base image is missing "
                       f"({type(exc).__name__}) — a network=none container "
                       f"cannot build one, so every leg would raise")
    return True, "; ".join(notes)


async def plan_batch(context, *, limit: int = None,
                     write: bool = True) -> List[Dict[str, Any]]:
    """Pick tonight's specs: replayable episodes whose questions have not
    been asked yet.

    Deduped against `known_spec_ids` — a batch that re-asks a question it
    has already answered spends the night re-measuring, and D4's
    stability check is the ONE place a deliberate re-run belongs.
    """
    limit = _batch_size() if limit is None else int(limit)
    seen = known_spec_ids()
    source = EpisodeSource(args=getattr(context, "args", None))
    picked: List[Dict[str, Any]] = []
    for traj, tri in source.iter_episodes():
        for spec in build_specs(traj, tri):
            if spec.spec_id in seen:
                continue
            picked.append(spec.to_dict())
            seen.add(spec.spec_id)
            if len(picked) >= limit:
                break
        if len(picked) >= limit:
            break
    # ⚠ DELIBERATELY NOT WRITTEN HERE. The first version appended every
    # picked spec to the durable ledger at PLAN time, and `known_spec_ids`
    # reads that ledger — so a spec that was then skipped (docker down, no
    # admissible validator, a control leg that could not be graded) was
    # marked "already asked" FOREVER, with no row anywhere saying why. A
    # week of a wedged daemon would consume ~21 specs of a 222-spec corpus
    # invisibly, and `credit_stats` would report a HEALTHIER abstain rate
    # the more specs were lost that way.
    #
    # A spec is recorded when it produced a verdict — including an
    # abstain. `write` is kept for callers that want the plan on disk.
    if write and picked:
        write_specs([ReplaySpec(**{k: v for k, v in d.items()
                                   if k != "spec_id"}) for d in picked])
    return picked


async def run_batch(context, *, limit: int = None,
                    n_pairs: int = DEFAULT_N_PAIRS,
                    spec_timeout_s: float = DEFAULT_SPEC_TIMEOUT_S,
                    batch_timeout_s: float = DEFAULT_BATCH_TIMEOUT_S
                    ) -> Dict[str, Any]:
    """One night of replay. Returns a summary for the activity ledger.

    Every spec pays for its own validator: synthesis, the static screen,
    and then the SELF-TEST — the control leg must reproduce the episode's
    recorded outcome. An episode whose control leg disagrees is marked
    non-replayable and skipped, because at that point either the world
    drifted or the validator is wrong, and neither can be told apart from
    the outcome the perturbed leg would produce.
    """
    import time as _time

    out = {"planned": 0, "validated": 0, "verdicts": [], "skipped": {},
           "stopped_early": ""}
    if not _enabled():
        out["skipped"]["disabled"] = 1
        return out
    batch_deadline = _time.monotonic() + float(batch_timeout_s)

    def _skip(reason: str) -> None:
        out["skipped"][reason] = out["skipped"].get(reason, 0) + 1

    specs = await plan_batch(context, limit=limit, write=False)
    out["planned"] = len(specs)
    if not specs:
        return out

    # One trajectory can supply several specs; synthesise (and self-test)
    # its validator ONCE.
    by_traj: Dict[str, List[Dict[str, Any]]] = {}
    for spec in specs:
        by_traj.setdefault(str(spec.get("trajectory_id") or ""),
                           []).append(spec)

    trajs = {}
    src = EpisodeSource(args=getattr(context, "args", None))
    for traj, _tri in src.iter_episodes():
        tid = str(getattr(traj, "id", "") or "")
        if tid in by_traj:
            trajs[tid] = traj

    llm = getattr(context, "llm_client", None)
    for tid, tspecs in by_traj.items():
        # ⚠ Re-checked between trajectories, not only at entry. Without
        # this the master kill switch cannot stop an in-flight batch, and
        # the only way to end up to an hour of unattended container work
        # is to kill the process.
        if not _enabled():
            out["stopped_early"] = "disabled mid-batch"
            break
        if _time.monotonic() >= batch_deadline:
            out["stopped_early"] = "batch deadline reached"
            break
        # Resources can change under a batch that spawns a container per
        # leg; a single reading at entry is not a guarantee for an hour.
        _ok, _why = preflight()
        if not _ok:
            out["stopped_early"] = f"preflight stood down mid-batch: {_why}"
            break
        traj = trajs.get(tid)
        if traj is None:
            _skip("episode_vanished")
            continue
        validator = await synthesize_validator(traj, llm)
        if not validator:
            _skip("no_admissible_validator")
            continue
        probe = dict(tspecs[0])
        probe["perturbation"] = PERTURB_VERIFY_TOGGLE   # inert on control
        recorded = str(probe.get("recorded_outcome") or "").lower()

        # ── THE NEGATIVE CONTROL, first, because it is the cheap one and
        # it is the check the positive half cannot make.
        #
        # `import sys; sys.exit(0)` passes the static screen (non-empty,
        # under the cap, no forbidden substring, compiles) and then passes
        # the agree-with-the-recording test with probability 1.0 on every
        # `passed` episode — 47 of the 67 real ones. It would report
        # `no_effect` for every perturbation of that episode forever,
        # because both arms exit 0 by construction. Nothing in the
        # agreement test can see that, because agreement is exactly what a
        # constant validator gives you.
        #
        # So: run the validator against an EMPTY fork with NO agent run.
        # A check that discriminates must fail there. A constant-pass
        # validator cannot, and a constant-fail one cannot pass the
        # positive half below.
        neg = await run_validator_only(context, validator,
                                       leg_timeout_s=DEFAULT_LEG_TIMEOUT_S)
        if neg.validator_exit == 2:
            # ⚠ NOT the same as a vacuous validator, and the first live
            # smoke lumped them together. Exit 2 is the validator's own
            # reserved "I cannot check this from the filesystem" — an
            # honest statement about the EPISODE, not a defect in the
            # check. Measured 2026-08-22: 2 of 16 sampled episodes, both
            # conversational turns whose deliverable was a reply rather
            # than an artifact.
            #
            # It belongs upstream, in D0's triage: an episode with no
            # filesystem-checkable deliverable is not replayable and
            # should never reach a validator call. Recorded here as its
            # own reason so the census can show how much of the corpus
            # that is before anyone builds the triage rule.
            _skip("episode_not_filesystem_checkable")
            continue
        if neg.passed is None:
            _skip("negative_control_ungradable")
            continue
        if neg.passed is True:
            _skip("validator_passes_an_empty_workspace")
            logger.info(
                "replay: episode %s rejected — its validator passes an "
                "EMPTY workspace, so it checks nothing", tid[:12])
            continue

        # ── THE POSITIVE HALF. A validator that cannot fail is not a
        # check, and its own synthesis is not evidence that it works.
        control = await run_leg(context, probe, arm="control",
                                validator=validator,
                                leg_timeout_s=DEFAULT_LEG_TIMEOUT_S)
        if not _leg_is_gradable(control):
            _skip(f"control_ungradable:{control.reason[:60]}")
            continue
        if control.passed != (recorded == "passed"):
            _skip("validator_disagreed_with_the_recording")
            logger.info(
                "replay: episode %s is not replayable — the control leg "
                "%s but the recording says %s", tid[:12],
                "passed" if control.passed else "failed", recorded)
            continue
        out["validated"] += 1
        for spec in tspecs:
            if _time.monotonic() >= batch_deadline:
                out["stopped_early"] = "batch deadline reached"
                break
            if not _enabled():
                out["stopped_early"] = "disabled mid-batch"
                break
            rec = await run_spec(context, spec, validator=validator,
                                 n_pairs=n_pairs,
                                 spec_timeout_s=spec_timeout_s,
                                 batch_deadline=batch_deadline)
            out["verdicts"].append(rec)
        if out["stopped_early"]:
            break

    # ── Sweep OUR OWN forks. The boot sweep spares any fork whose owner
    # PID is alive — which is this process — so a fork leaked at 03:00
    # was never reclaimed while the agent ran. Nothing bounds what a
    # replayed turn writes into one, and the batch's whole footprint is
    # temp directories.
    try:
        from .isolation import sweep_own_forks
        _swept = await asyncio.to_thread(sweep_own_forks)
        if _swept:
            out["swept_forks"] = len(_swept)
            logger.info("replay: reclaimed %d leaked fork(s)", len(_swept))
    except Exception as exc:  # noqa: BLE001
        logger.debug("replay: fork sweep skipped (%s)", exc)
    return out


def batch_summary(out: Dict[str, Any]) -> str:
    """One operator line. Names the ABSTAINS and the SKIPS, because a
    night that produced three abstains and a night that produced three
    verdicts are different nights and a count of "3" hides which."""
    v = out.get("verdicts") or []
    counts: Dict[str, int] = {}
    for r in v:
        k = str(r.get("verdict") or "?")
        counts[k] = counts.get(k, 0) + 1
    bits = [f"{n}x {k}" for k, n in sorted(counts.items(),
                                           key=lambda kv: -kv[1])]
    skipped = out.get("skipped") or {}
    skip_bits = [f"{n}x {k}" for k, n in sorted(skipped.items(),
                                                key=lambda kv: -kv[1])]
    stopped = str(out.get("stopped_early") or "")
    return (f"dream replay: {len(v)} verdict(s) from "
            f"{out.get('validated', 0)}/{out.get('planned', 0)} "
            f"episode(s)"
            + (f" — {', '.join(bits)}" if bits else "")
            + (f"; skipped {', '.join(skip_bits)}" if skip_bits else "")
            + (f"; STOPPED EARLY: {stopped}" if stopped else ""))
