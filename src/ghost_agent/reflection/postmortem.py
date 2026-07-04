"""Post-mortem engine — whole-transcript failure analysis → durable defect reports.

Where the Reflector (``reflection/loop.py``) sees only a failed turn's
final outcome + ``failure_reason`` and produces a *behavioural* revised
plan, the post-mortem engine reads the WHOLE tool-call sequence and asks
a different question: *what about the agent itself made this fail?* —
the question a human maintainer asks when triaging a bad run.

That question has three kinds of answer, and the engine classifies into
exactly those three:

  * **behavioural** — the agent chose badly; a future lesson fixes it.
    Routed straight into the existing lesson pipeline (SkillMemory),
    same channel the Reflector uses, so it gets retrieved on the next
    similar request.
  * **configuration** — a flag / threshold let the failure mode through
    (e.g. a decay rate that let an oscillation evade the loop cap).
    Queued as a proposed config change for the operator.
  * **code_defect** — a tool or the control loop is broken (e.g. the
    browser tool never surfaced ``pageerror``; ``replace`` failed
    silently on whitespace drift). Queued WITH an optional
    LLM-generated reproducing test + unified diff — the exact artifact
    a maintainer reviews. Nothing is ever applied automatically.

The grounding that makes the queue trustworthy is the *structural
signature* (``compute_signature``): a pure, LLM-free fingerprint of the
transcript's pathology — repeated identical tool errors, two-tool
oscillation, same-file read loops, dominant failing tool. It both scores
which runs are worth a post-mortem (selection is deterministic, no model
call wasted on a healthy run) and gives the LLM concrete evidence to
diagnose rather than guess.

Safety mirrors the Reflector: never raises into the watchdog; never
mutates input trajectories; respects a per-call timeout. The LLM is
injected as ``analyze_fn`` (and an optional ``patch_fn`` for the
code-defect diff) so tests pass deterministic stubs and production wraps
Ghost's LLMClient.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence, Set, Tuple, Union

from ..distill.schema import Trajectory, ToolCall, Outcome
from .postmortem_prompts import (
    build_postmortem_prompt,
    parse_postmortem_output,
    build_patch_prompt,
)

logger = logging.getLogger("GhostPostMortem")


# Primary argument names tried, in order, when deciding what a tool call
# "operated on" — used for same-target read/write loop detection. A
# browser re-clicking one selector, a reader re-reading one path, and a
# fetch re-pulling one url are all the same pathology: repeating an
# action against an unchanged target.
_PRIMARY_ARG_KEYS = ("path", "file_path", "file", "filename", "url", "selector", "query", "target")

# Categories the engine emits. Anything the LLM returns that isn't one of
# these is coerced to "behavioural" (the safe default — it routes to a
# lesson, which is additive and never touches code or config).
CATEGORY_BEHAVIOURAL = "behavioural"
CATEGORY_CONFIGURATION = "configuration"
CATEGORY_CODE_DEFECT = "code_defect"
_VALID_CATEGORIES = frozenset(
    {CATEGORY_BEHAVIOURAL, CATEGORY_CONFIGURATION, CATEGORY_CODE_DEFECT}
)


def primary_target_from_args(args) -> str:
    """The value of the first recognised primary-arg key in ``args``, or
    "" when none is present (or ``args`` isn't a dict). Lower-cased +
    stripped so trivially different spellings of the same target collapse
    together.

    Shared by the offline post-mortem signature (``_primary_target``) and
    the in-run no-progress loop-breaker in ``core/agent.py`` so both agree
    on what "the same target" means — a single definition of the thing a
    tool call operated on."""
    if not isinstance(args, dict):
        return ""
    for k in _PRIMARY_ARG_KEYS:
        if k in args and args[k] is not None:
            return str(args[k]).strip().lower()[:200]
    return ""


def _primary_target(tc: ToolCall) -> str:
    """Primary target of a ToolCall (offline path) — delegates to
    ``primary_target_from_args`` on the call's arguments dict."""
    return primary_target_from_args(getattr(tc, "arguments", None) or {})


def _error_key(tc: ToolCall) -> str:
    """A normalised prefix of a tool call's error, or "" when it
    succeeded. Truncated so two errors that differ only in a trailing
    variable (a path, an offset) count as the same recurring failure."""
    err = (getattr(tc, "error", "") or "").strip()
    if not err:
        return ""
    return err[:80].lower()


@dataclass
class TranscriptSignature:
    """Pure, LLM-free fingerprint of a transcript's failure shape.

    Every field is derived deterministically from the tool sequence;
    ``severity`` is a 0..1 blend used both to rank runs for analysis and
    to stamp the resulting defect. ``hash`` is stable across runs that
    share the same pathology so the queue can dedup — the engine never
    spends an LLM call on a defect it has already filed.
    """

    n_steps: int = 0
    duration_s: float = 0.0
    distinct_tools: int = 0
    dominant_tool: str = ""
    dominant_tool_share: float = 0.0
    repeated_error_count: int = 0
    repeated_error_tool: str = ""
    repeated_error_text: str = ""
    oscillation_count: int = 0
    oscillation_pair: str = ""
    read_loop_count: int = 0
    read_loop_tool: str = ""
    read_loop_target: str = ""
    severity: float = 0.0
    hash: str = ""

    def summary(self) -> str:
        """One-paragraph human-readable evidence block for the prompt and
        the stored defect — the grounded facts, no interpretation."""
        parts = [
            f"{self.n_steps} tool calls over {self.duration_s:.0f}s",
            f"{self.distinct_tools} distinct tools",
        ]
        if self.dominant_tool:
            parts.append(
                f"dominated by '{self.dominant_tool}' ({self.dominant_tool_share:.0%} of calls)"
            )
        if self.repeated_error_count >= 2:
            parts.append(
                f"the SAME error from '{self.repeated_error_tool}' recurred "
                f"{self.repeated_error_count}x: \"{self.repeated_error_text}\""
            )
        if self.oscillation_count >= 2:
            parts.append(
                f"two-tool oscillation {self.oscillation_pair} repeated "
                f"{self.oscillation_count}x (thrashing between two actions)"
            )
        if self.read_loop_count >= 2:
            parts.append(
                f"'{self.read_loop_tool}' hit the SAME target "
                f"'{self.read_loop_target}' {self.read_loop_count}x "
                f"(re-acting on an unchanged target)"
            )
        return "; ".join(parts) + "."


def _norm(value: float, scale: float) -> float:
    """Saturating normaliser: value/scale clamped to [0, 1]. Used to fold
    each raw pathology count into a bounded severity contribution."""
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(scale)))


def compute_signature(traj: Trajectory) -> TranscriptSignature:
    """Derive the structural failure fingerprint of ``traj``.

    Pure function — no LLM, no I/O, deterministic. This is the trust
    anchor of the whole engine: the operator reviewing a defect can see
    the concrete pathology (e.g. "the same not-found error recurred 11x")
    independent of whatever the LLM later wrote about it.
    """
    calls: List[ToolCall] = [c for c in (getattr(traj, "tool_calls", None) or []) if c is not None]
    sig = TranscriptSignature(
        n_steps=len(calls),
        duration_s=float(getattr(traj, "duration_s", 0.0) or 0.0),
    )
    if not calls:
        sig.hash = _hash_signature(sig)
        return sig

    names = [(getattr(c, "name", "") or "unknown").strip() for c in calls]

    # --- dominant tool ---
    counts: dict = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    sig.distinct_tools = len(counts)
    dom_tool, dom_n = max(counts.items(), key=lambda kv: kv[1])
    sig.dominant_tool = dom_tool
    sig.dominant_tool_share = dom_n / len(calls)

    # --- repeated identical (tool, error) ---
    err_counts: dict = {}
    for nm, c in zip(names, calls):
        ek = _error_key(c)
        if not ek:
            continue
        key = (nm, ek)
        err_counts[key] = err_counts.get(key, 0) + 1
    if err_counts:
        (etool, etext), ecount = max(err_counts.items(), key=lambda kv: kv[1])
        if ecount >= 2:
            sig.repeated_error_count = ecount
            sig.repeated_error_tool = etool
            sig.repeated_error_text = etext

    # --- two-tool oscillation (A,B,A,B,...) ---
    # Count, for every adjacent A!=B pair, how far an alternating A/B
    # pattern extends; keep the longest. Length L means L positions
    # follow the pattern, i.e. ~L//2 full A->B->A cycles.
    best_osc = 0
    best_pair = ""
    i = 0
    n_names = len(names)
    while i < n_names - 1:
        a, b = names[i], names[i + 1]
        if a != b:
            run = 2
            j = i + 2
            while j < n_names and names[j] == names[j - 2]:
                run += 1
                j += 1
            if run > best_osc:
                best_osc = run
                best_pair = f"{a}<->{b}"
            i = j - 1 if run > 2 else i + 1
        else:
            i += 1
    cycles = best_osc // 2
    if cycles >= 2:
        sig.oscillation_count = cycles
        sig.oscillation_pair = best_pair

    # --- same-target read/act loop ---
    target_counts: dict = {}
    for nm, c in zip(names, calls):
        tgt = _primary_target(c)
        if not tgt:
            continue
        key = (nm, tgt)
        target_counts[key] = target_counts.get(key, 0) + 1
    if target_counts:
        (rtool, rtgt), rcount = max(target_counts.items(), key=lambda kv: kv[1])
        if rcount >= 2:
            sig.read_loop_count = rcount
            sig.read_loop_tool = rtool
            sig.read_loop_target = rtgt

    # --- severity blend ---
    # Each term saturates so no single pathology dominates; the weights
    # reflect how strongly each predicts a real, fixable agent defect
    # (recurring identical errors and oscillation are the clearest tells,
    # which is exactly what the manual June post-mortems keyed on).
    sig.severity = max(0.0, min(1.0,
        0.30 * _norm(sig.repeated_error_count, 5)
        + 0.25 * _norm(sig.oscillation_count, 4)
        + 0.20 * _norm(sig.read_loop_count, 5)
        + 0.15 * _norm(sig.n_steps, 30)
        + 0.10 * _norm(sig.duration_s, 1200)
    ))
    sig.hash = _hash_signature(sig)
    return sig


def _hash_signature(sig: TranscriptSignature) -> str:
    """Stable dedup key for a pathology.

    Built from the *categorical* shape (which tool, which error, which
    bucketed magnitude) — NOT raw counts — so two runs that fail the
    same way (11x vs 13x of the same error) collapse to one defect
    instead of re-filing on every idle window. Bucketing magnitudes
    keeps "got slightly worse" from looking like a new defect."""
    def bucket(n: float) -> str:
        n = float(n)
        if n <= 0:
            return "0"
        if n < 3:
            return "lo"
        if n < 8:
            return "mid"
        return "hi"

    basis = "|".join([
        sig.repeated_error_tool,
        sig.repeated_error_text[:40],
        bucket(sig.repeated_error_count),
        sig.oscillation_pair,
        bucket(sig.oscillation_count),
        sig.read_loop_tool,
        sig.read_loop_target[:40],
        bucket(sig.read_loop_count),
        sig.dominant_tool if not sig.repeated_error_tool else "",
    ])
    return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]


@dataclass
class DefectReport:
    """A durable, operator-reviewable finding from one post-mortem.

    Persisted as one JSONL line in the ``DefectQueue``. The category
    decides which fields carry payload: ``lesson`` for behavioural,
    ``config_change`` for configuration, ``proposed_test`` /
    ``proposed_patch`` for code_defect. ``status`` tracks the lifecycle
    (pending → routed/applied/dismissed); nothing here is ever applied
    by the engine itself.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    signature_hash: str = ""
    source_trajectory_ids: List[str] = field(default_factory=list)
    category: str = CATEGORY_BEHAVIOURAL
    title: str = ""
    severity: float = 0.0
    root_cause: str = ""
    evidence: str = ""           # the pure structural-signature summary
    # category-specific payloads
    lesson: Optional[dict] = None        # behavioural: kwargs for learn_lesson
    config_change: str = ""              # configuration: proposed flag/threshold
    code_fix: str = ""                   # code_defect: which file/function + what
    proposed_test: str = ""              # code_defect: reproducing test (text, unapplied)
    proposed_patch: str = ""             # code_defect: unified diff (text, unapplied)
    status: str = "pending"              # pending | routed | applied | dismissed

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "DefectReport":
        d = dict(d)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class DefectQueue:
    """Append-only JSONL store of defect reports under ``root``.

    Modelled on ``TrajectoryCollector``: thread-safe append, never
    raises on write (a failed queue write must not break the watchdog),
    and a sidecar for status mutations so the original finding stays
    immutable for audit. Dedup is by ``signature_hash`` — ``add`` is a
    no-op for a pathology already on file.
    """

    FILENAME = "defects.jsonl"
    STATUS_FILENAME = "defect_status.jsonl"

    def __init__(self, root, *, enabled: bool = True):
        from pathlib import Path
        import threading
        self.root = Path(root)
        self.enabled = enabled
        self._lock = threading.Lock()

    def _path(self):
        return self.root / self.FILENAME

    def _status_path(self):
        return self.root / self.STATUS_FILENAME

    def known_signatures(self) -> Set[str]:
        """Signature hashes already filed (any status). Used to skip
        re-analysing a pathology the engine has already seen."""
        out: Set[str] = set()
        for rep in self._iter_raw():
            if rep.signature_hash:
                out.add(rep.signature_hash)
        return out

    def has_signature(self, signature_hash: str) -> bool:
        if not signature_hash:
            return False
        return signature_hash in self.known_signatures()

    def add(self, report: DefectReport) -> bool:
        """Append ``report`` unless its signature is already filed.
        Returns True iff written. Never raises."""
        if not self.enabled:
            return False
        try:
            if report.signature_hash and self.has_signature(report.signature_hash):
                return False
            with self._lock:
                self._path().parent.mkdir(parents=True, exist_ok=True)
                with self._path().open("a", encoding="utf-8") as f:
                    f.write(report.to_jsonl())
                    f.write("\n")
                    f.flush()
            return True
        except Exception as e:
            logger.warning("defect queue append failed: %s", e)
            return False

    def update_status(self, defect_id: str, status: str, note: str = "") -> bool:
        """Record a status mutation in the sidecar (original line stays
        immutable). Never raises."""
        if not self.enabled or not defect_id:
            return False
        try:
            import json
            rec = {
                "id": str(defect_id),
                "status": str(status or ""),
                "note": str(note or "")[:500],
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with self._lock:
                self._status_path().parent.mkdir(parents=True, exist_ok=True)
                with self._status_path().open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
            return True
        except Exception as e:
            logger.warning("defect status append failed: %s", e)
            return False

    def _load_status(self) -> dict:
        path = self._status_path()
        if not path.exists():
            return {}
        out: dict = {}
        try:
            import json
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    did = rec.get("id")
                    if did:
                        out[did] = rec
        except OSError as e:
            logger.warning("cannot read defect status sidecar: %s", e)
        return out

    def _iter_raw(self) -> Iterable[DefectReport]:
        path = self._path()
        if not path.exists():
            return
        try:
            import json
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield DefectReport.from_dict(json.loads(line))
                    except Exception:
                        continue
        except OSError as e:
            logger.warning("cannot read defect queue: %s", e)

    def all(self) -> List[DefectReport]:
        """Every report, with the latest sidecar status overlaid."""
        status = self._load_status()
        out: List[DefectReport] = []
        for rep in self._iter_raw():
            srec = status.get(rep.id)
            if srec and srec.get("status"):
                rep.status = srec["status"]
            out.append(rep)
        return out

    def pending(self) -> List[DefectReport]:
        """Reports still awaiting operator action (status 'pending'),
        most severe first."""
        return sorted(
            [r for r in self.all() if r.status == "pending"],
            key=lambda r: r.severity,
            reverse=True,
        )


@dataclass
class PostMortemRunReport:
    """Aggregate of one post-mortem tick, for a one-line watchdog log."""

    selected: int = 0
    analysed_ok: int = 0
    analysed_errors: int = 0
    behavioural: int = 0
    configuration: int = 0
    code_defect: int = 0
    queued: int = 0
    skipped_duplicate: int = 0
    reports: List[DefectReport] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"post-mortem: {self.analysed_ok}/{self.selected} analysed "
            f"({self.behavioural} behav, {self.configuration} config, "
            f"{self.code_defect} code) · {self.queued} queued, "
            f"{self.skipped_duplicate} dup, {self.analysed_errors} err"
        )


AnalyzeCallable = Callable[[str], Union[str, Awaitable[str]]]
PatchCallable = Callable[[str], Union[str, Awaitable[str]]]
LessonSink = Callable[..., Any]


def select_failed_runs(
    trajectories: Iterable[Trajectory],
    *,
    limit: int = 2,
    min_severity: float = 0.4,
    exclude_signatures: Optional[Set[str]] = None,
    include_unknown: bool = False,
) -> List[Tuple[Trajectory, TranscriptSignature]]:
    """Pick the worst FAILED runs worth a post-mortem.

    Deterministic and LLM-free: compute each candidate's structural
    signature, drop anything below ``min_severity`` or already filed
    (``exclude_signatures``), and return the top ``limit`` by severity.
    Selecting before any model call is the point — a healthy run never
    costs a post-mortem.

    By default only ``FAILED`` trajectories are considered. ``UNKNOWN``
    turns (ordinary chat with no verifier) are admitted only when
    ``include_unknown`` is set AND they clear the severity bar — a
    conservative default, since UNKNOWN is the common case.
    """
    exclude = exclude_signatures or set()
    scored: List[Tuple[Trajectory, TranscriptSignature]] = []
    seen_this_run: Set[str] = set()
    for traj in trajectories:
        outcome = getattr(traj, "outcome", "")
        if outcome == Outcome.FAILED.value:
            pass
        elif include_unknown and outcome == Outcome.UNKNOWN.value:
            pass
        else:
            continue
        sig = compute_signature(traj)
        if sig.severity < min_severity:
            continue
        if sig.hash in exclude or sig.hash in seen_this_run:
            continue
        seen_this_run.add(sig.hash)
        scored.append((traj, sig))
    scored.sort(key=lambda ts: ts[1].severity, reverse=True)
    return scored[:limit]


class PostMortemEngine:
    """Drives the post-mortem phase: select → analyse → classify → route.

    Usage::

        engine = PostMortemEngine(analyze_fn=my_llm, queue=DefectQueue(root))
        report = await engine.run(source=collector.iter_trajectories)

    ``analyze_fn`` returns the classification block (see
    ``postmortem_prompts``). ``patch_fn`` (optional) is a *coding* model
    that, for a code_defect, returns a reproducing test + unified diff —
    stored as a proposal only, never applied. ``lesson_sink`` (optional)
    is called with ``learn_lesson``-shaped kwargs for behavioural
    findings, reusing the existing failure→lesson channel.
    """

    def __init__(
        self,
        analyze_fn: AnalyzeCallable,
        *,
        queue: DefectQueue,
        lesson_sink: Optional[LessonSink] = None,
        patch_fn: Optional[PatchCallable] = None,
        per_call_timeout_s: float = 90.0,
        patch_timeout_s: float = 120.0,
        max_runs: int = 2,
        min_severity: float = 0.4,
        include_unknown: bool = False,
        model: str = "",
    ):
        self.analyze_fn = analyze_fn
        self.queue = queue
        self.lesson_sink = lesson_sink
        self.patch_fn = patch_fn
        self.per_call_timeout_s = float(per_call_timeout_s)
        self.patch_timeout_s = float(patch_timeout_s)
        self.max_runs = int(max_runs)
        self.min_severity = float(min_severity)
        self.include_unknown = bool(include_unknown)
        self.model = model

    async def run(
        self,
        *,
        source: Union[Iterable[Trajectory], Callable[[], Iterable[Trajectory]]],
    ) -> PostMortemRunReport:
        """One post-mortem tick. Never raises."""
        report = PostMortemRunReport()
        try:
            if callable(source) and not hasattr(source, "__iter__"):
                trajectories = list(source())
            else:
                trajectories = list(source)
        except Exception as e:
            logger.warning("post-mortem source iteration failed: %s", e)
            return report

        known = set()
        try:
            known = self.queue.known_signatures()
        except Exception:
            pass

        selected = select_failed_runs(
            trajectories,
            limit=self.max_runs,
            min_severity=self.min_severity,
            exclude_signatures=known,
            include_unknown=self.include_unknown,
        )
        report.selected = len(selected)

        for traj, sig in selected:
            try:
                rep = await self._analyse_one(traj, sig)
            except Exception as e:
                logger.warning("post-mortem analyse failed: %s", e)
                report.analysed_errors += 1
                continue
            if rep is None:
                report.analysed_errors += 1
                continue
            report.analysed_ok += 1
            report.reports.append(rep)
            if rep.category == CATEGORY_BEHAVIOURAL:
                report.behavioural += 1
            elif rep.category == CATEGORY_CONFIGURATION:
                report.configuration += 1
            elif rep.category == CATEGORY_CODE_DEFECT:
                report.code_defect += 1

            # Route behavioural findings into the existing lesson channel
            # immediately (additive, safe). They're ALSO queued (status
            # 'routed') so the operator has a unified ledger.
            if rep.category == CATEGORY_BEHAVIOURAL and self.lesson_sink is not None and rep.lesson:
                try:
                    self.lesson_sink(**rep.lesson)
                    rep.status = "routed"
                except Exception as e:
                    logger.warning("post-mortem lesson sink failed: %s", e)

            if self.queue.add(rep):
                report.queued += 1
            else:
                report.skipped_duplicate += 1

        return report

    async def _call(self, fn, prompt: str, timeout: float) -> str:
        call = fn(prompt)
        if inspect.isawaitable(call):
            return await asyncio.wait_for(call, timeout=timeout)
        return str(call)

    async def _analyse_one(
        self, traj: Trajectory, sig: TranscriptSignature
    ) -> Optional[DefectReport]:
        prompt = build_postmortem_prompt(traj, sig)
        try:
            text = await self._call(self.analyze_fn, prompt, self.per_call_timeout_s)
        except asyncio.TimeoutError:
            logger.debug("post-mortem analyse timed out")
            return None
        except Exception as e:
            logger.debug("post-mortem analyse_fn raised: %s", e)
            return None

        parsed = parse_postmortem_output(text or "")
        if not parsed.get("root_cause") and not parsed.get("category"):
            return None

        category = parsed.get("category") or CATEGORY_BEHAVIOURAL
        if category not in _VALID_CATEGORIES:
            category = CATEGORY_BEHAVIOURAL

        rep = DefectReport(
            signature_hash=sig.hash,
            source_trajectory_ids=[traj.id],
            category=category,
            title=(parsed.get("title") or "")[:160],
            severity=round(sig.severity, 3),
            root_cause=(parsed.get("root_cause") or "")[:1200],
            evidence=sig.summary(),
        )

        if category == CATEGORY_BEHAVIOURAL:
            _solution = (parsed.get("lesson") or parsed.get("root_cause") or "")[:1200]
            # A terse classifier reply can carry CATEGORY + TITLE but no ROOT
            # CAUSE and no LESSON → empty solution. Don't route a blank lesson
            # into the playbook (learn_lesson doesn't reject an empty solution).
            if _solution.strip():
                rep.lesson = {
                    "task": (getattr(traj, "user_request", "") or "")[:400],
                    "mistake": (parsed.get("root_cause") or "")[:400],
                    "solution": _solution,
                    "source": "postmortem",
                    "source_trajectory_id": traj.id,
                }
        elif category == CATEGORY_CONFIGURATION:
            rep.config_change = (parsed.get("config_change") or "")[:1000]
        elif category == CATEGORY_CODE_DEFECT:
            rep.code_fix = (parsed.get("code_fix") or "")[:1000]
            if self.patch_fn is not None:
                await self._attach_patch(rep, traj, sig)

        return rep

    async def _attach_patch(
        self, rep: DefectReport, traj: Trajectory, sig: TranscriptSignature
    ) -> None:
        """Ask the coding model for a reproducing test + unified diff and
        store them on the report as a PROPOSAL. Never applied here — the
        artifact exists for the operator to review and apply by hand."""
        prompt = build_patch_prompt(traj, sig, rep.root_cause, rep.code_fix)
        try:
            text = await self._call(self.patch_fn, prompt, self.patch_timeout_s)
        except asyncio.TimeoutError:
            logger.debug("post-mortem patch_fn timed out")
            return
        except Exception as e:
            logger.debug("post-mortem patch_fn raised: %s", e)
            return
        test_block, patch_block = _split_patch_output(text or "")
        rep.proposed_test = test_block[:6000]
        rep.proposed_patch = patch_block[:8000]


def _split_patch_output(text: str) -> Tuple[str, str]:
    """Split the coding model's reply into (test, diff).

    Looks for ``REPRODUCING TEST`` and ``PATCH`` / ``DIFF`` section
    markers; falls back to fenced code blocks, then to putting
    everything in the patch slot. Lenient by design — same posture as
    the reflection parser."""
    if not text:
        return "", ""
    import re
    lower = text.lower()

    def _find(markers):
        # Word-boundary match so "patch" doesn't hit "dispatch"/"patchwork",
        # "diff" doesn't hit "difference", and "test:" doesn't hit "latest:".
        for marker in markers:
            pat = r"\b" + re.escape(marker)
            if marker[-1].isalnum():
                pat += r"\b"
            m = re.search(pat, lower)
            if m:
                return m.start()
        return -1

    test_pos = _find(("reproducing test", "failing test", "test:"))
    patch_pos = -1
    for marker in ("patch", "diff", "unified diff"):
        pat = r"\b" + re.escape(marker)
        if marker[-1].isalnum():
            pat += r"\b"
        m = re.search(pat, lower)
        if m and m.start() != test_pos:
            patch_pos = m.start()
            break
    if test_pos != -1 and patch_pos != -1 and patch_pos > test_pos:
        return text[test_pos:patch_pos].strip(), text[patch_pos:].strip()
    # Fallback: two fenced blocks → first is test, second is diff.
    blocks = re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) >= 2:
        return blocks[0].strip(), blocks[1].strip()
    if len(blocks) == 1:
        return "", blocks[0].strip()
    return "", text.strip()
