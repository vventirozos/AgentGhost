"""§4F Phase 2b — tool-choice fixture miner.

Mines tool-CHOICE fixtures for tool-description optimization from the
GHOST_LLM_RECORD day-files (`$GHOST_HOME/system/llm_recordings/*.jsonl`),
joined to trajectory ground truth. The contract is the audited 2026-08-01
one (PROJECT_JOURNAL §4F Phase 2b):

- **Era filter**: only records with ``ts >= 2026-07-31T19:15`` LOCAL time.
  The QWEN_TOOL_PROMPT_NATIVE split (~18:54) changed the system prompt in
  every native-path context and the honest-failure rule (~19:14) changed
  outcome semantics — earlier records mix eras and would teach against a
  prompt that no longer exists. Record ``ts`` is UTC ("...Z"); the cutoff
  converts through the local timezone.
- **Choice records**: reassembled stream records
  (``kind=chat_completion_stream``, plus non-stream ``chat_completion``)
  whose payload advertises ``tools`` and whose response carries STRUCTURED
  ``message.tool_calls`` — never content-parsed. Records carrying the
  ``SYSTEM`` request sentinel (idle loops, background work) are excluded:
  they have no per-request trajectory to join, and the first background
  path that writes one would silently label the whole SYSTEM bucket.
- **Ground truth**: trajectory outcome labels via
  ``TrajectoryCollector.iter_trajectories()`` (corrections-sidecar overlay
  applied), joined on recording ``request_id`` == ``Trajectory.session_id``.
  Never raw exit-code heuristics. The overlay is load-bearing, measured
  2026-08-04 on the live store: 214 of 1488 trajectories (14.4%) read a
  different outcome after it, and for 212 of those the on-disk line says
  ``unknown`` — i.e. the overlay is the ONLY source of a label. Reading the
  raw JSONL instead would discard them as unlabeled.
- **Polarity**: clean PASSED = positive; FAILED (verifier-refuted or
  shape) = negative; honest-failure turns (PASSED with a failed tool call
  — the verifier outranked the structural failure) are EXCLUDED: the
  choice signal is ambiguous, never labeled bad per the rule, not claimed
  good either. UNKNOWN / unjoined records are excluded. Since 2026-08-04 a
  small class moved from EXCLUDED to negative rather than changing meaning
  here: a turn where EVERY tool call failed and the reply never said so is
  now labelled FAILED upstream (`resolve_turn_outcome` rule 2b), and a
  shape-FAILED is a negative by the line above — which is right, since the
  tool CHOICE is what failed. 7 of 1491 live records.
- **Result pairing**: a choice's tool result rides the NEXT record (same
  recorder ``session_id``, ``ordinal + 1``, same ``request_id``) as
  USER-role messages. Two live gotchas are handled: the volatile
  system-state injection is PREPENDED to that same user message, so
  extraction slices from the ``<tool_response`` marker (never the message
  head); and same-request background calls (e.g. the context-shield
  summarizer) can occupy ``ordinal + 1`` — a candidate that isn't
  choice-kind or has no marker yields no preview rather than a mispair.
- **Split**: public/private tier by ``request_id`` hash
  (``holdout_tier("toolfx:<request_id>")``) so every fixture from one turn
  shares a tier — membership can never migrate as the corpus grows.
- **Experiment isolation** (added with `core.experiments`): turns where a
  live A/B treatment MUTATED the prompt context are excluded by default
  (``exclude_mutated_context``). The optimizer replays recorded payloads
  verbatim, so a steered turn's later payloads carry the steer text — mixing
  them in would tune descriptions against a context only half of production
  sees, and would make the fixture corpus quietly non-stationary while the
  experiment runs. Same class of hazard as the era filter above, which is
  why it is handled the same way: excluded, counted, never silent.

Memory: the mine is ONE streaming pass. Only the light fixtures and a
pending-pair index are retained — full records (each ~100 KB with the
tools block) are dropped as soon as they are classified, so multi-GB
recording backlogs mine in bounded memory. Fixtures are deliberately
LIGHT: the full recorded payload stays in the day-file; each fixture
carries a ``source`` pointer (file, session_id, ordinal) so the GEPA eval
adapter can rehydrate the exact request and swap candidate descriptions
in. (Adapter note: `RequestState._active_tool_defs_cache` and the XML
schema cache key on tool NAMES, not description bytes — an offline
adapter must build a fresh RequestState per candidate.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ..distill.collector import TrajectoryCollector
from ..distill.outcome_heuristics import tool_call_failed
from ..distill.schema import Outcome
from .trainset import holdout_tier

logger = logging.getLogger("GhostAgent")

# The pinned era boundary (LOCAL time — see module docstring).
DEFAULT_ERA_CUTOFF_LOCAL = "2026-07-31T19:15"

# Kinds that can carry a structured tool choice. The main loop streams, so
# chat_completion_stream dominates; plain chat_completion is accepted for
# completeness (a non-stream record without structured tool_calls is
# skipped by the tools/tool_calls requirement either way).
_CHOICE_KINDS = ("chat_completion_stream", "chat_completion")

# Tool results are rendered into user-role messages as <tool_response ...>
# blocks (native dialect). Everything before the marker is injected
# system state, never tool output.
_TOOL_RESPONSE_MARKER = "<tool_response"


@dataclass
class ToolChoiceFixture:
    """One tool-choice decision with ground-truth polarity."""

    fixture_id: str
    request_id: str
    ts: str                       # record timestamp (UTC ISO, trailing Z)
    user_request: str             # from the joined trajectory (canonical)
    chosen_tools: List[Dict[str, str]] = field(default_factory=list)
    advertised_tools: List[str] = field(default_factory=list)
    label: float = 0.0            # 1.0 clean PASSED, 0.0 FAILED
    outcome: str = ""             # joined trajectory outcome (post-overlay)
    failure_reason: str = ""
    tool_results: List[str] = field(default_factory=list)  # previews
    tier: str = "public"          # public | private (request_id-hash split)
    source: Dict[str, Any] = field(default_factory=dict)   # {file, session_id, ordinal}
    # §4BF 1c: population of the joined trajectory ("user_request" or
    # "bench") — the explicit origin feature the admissibility matrix
    # requires on every bench-admitted example.
    origin: str = "user_request"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def era_cutoff_utc(local_iso: str = DEFAULT_ERA_CUTOFF_LOCAL) -> datetime:
    """Interpret a naive ISO string as LOCAL wall-clock time and return the
    equivalent aware UTC datetime (record ``ts`` fields are UTC)."""
    naive = datetime.fromisoformat(local_iso)
    return naive.astimezone().astimezone(timezone.utc)


def _record_ts_utc(rec: Dict[str, Any]) -> Optional[datetime]:
    raw = rec.get("ts")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _response_message(rec: Dict[str, Any]) -> Dict[str, Any]:
    response = rec.get("response")
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def _structured_tool_calls(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = _response_message(rec).get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    return [c for c in calls if isinstance(c, dict)
            and (c.get("function") or {}).get("name")]


def is_choice_record(rec: Dict[str, Any]) -> bool:
    """A record that captured an actual tool CHOICE: an advertised tool set
    in the payload plus a structured tool_calls response."""
    if rec.get("kind") not in _CHOICE_KINDS:
        return False
    payload = rec.get("payload")
    if not isinstance(payload, dict) or not payload.get("tools"):
        return False
    return bool(_structured_tool_calls(rec))


def iter_recording_records(
    paths: Iterable[Path],
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Tolerantly iterate JSONL day-files; corrupt lines are skipped."""
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        yield path, rec
        except OSError as e:
            logger.warning("tool-fixture miner: cannot read %s: %s", path, e)


def _result_previews_from_messages(messages: List[Any],
                                   max_chars: int) -> List[str]:
    """Extract tool-result previews from a follow-up record's messages.

    Only user-role messages AFTER the last assistant message are
    considered, and only from the ``<tool_response`` marker onward — the
    volatile ``<system_state_update>`` block is PREPENDED to the same
    message on the live path and must never leak into fixtures. A message
    without the marker (e.g. a same-request summarizer prompt) yields
    nothing.
    """
    if not isinstance(messages, list):
        return []
    last_assistant = -1
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_assistant = i
    previews: List[str] = []
    for m in messages[last_assistant + 1:]:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        idx = content.find(_TOOL_RESPONSE_MARKER)
        if idx == -1:
            continue
        previews.append(content[idx:idx + max_chars])
    return previews


def _label_for(traj) -> Optional[float]:
    """Fixture polarity per the pinned contract; None = exclude."""
    outcome = getattr(traj, "outcome", "")
    if outcome == Outcome.PASSED.value:
        if any(tool_call_failed(tc) for tc in getattr(traj, "tool_calls", [])):
            # Honest failure: a tool broke, the agent said so, the verifier
            # passed the turn. The tool CHOICE is neither confirmed good nor
            # labeled bad — ambiguous, so it never enters the fixture set.
            return None
        return 1.0
    if outcome == Outcome.FAILED.value:
        return 0.0
    return None


# Set when `core.experiments` could not be imported, so the caller can report
# that the exclusion was UNAVAILABLE rather than that nothing was excluded.
_FILTER_UNAVAILABLE = {"hit": False}
# Set when the filter was importable but RAISED on a trajectory. Swallowing
# that silently is the same defect as swallowing the ImportError: the turn is
# then included as if it had been checked and found clean.
_FILTER_ERRORS = {"n": 0}


def _experiment_filter_available() -> bool:
    """Can the steered-context exclusion run at all, in THIS environment?

    Probed EAGERLY, once per mine, not lazily on the first trajectory that
    reaches the filter. The lazy version only learned the answer if some
    record survived every earlier drop — so a corpus that is entirely
    pre-era, unjoined, or empty reported `experiment_filter_unavailable 0`
    alongside `experiment_context_excluded 0`, which is exactly the
    "indistinguishable from a clean corpus" outcome this counter exists to
    prevent. Measured: the current live mine drops 219 of 490 post-era choice
    records as `unjoined`, so "no record reaches the filter" is a live shape,
    not a hypothetical.
    """
    try:
        from ..core.experiments import context_was_mutated  # noqa: F401
        return True
    except ImportError:
        return False


def _context_was_mutated(traj) -> bool:
    """Did a live A/B treatment alter this turn's prompt context?

    Lazy import so the miner keeps working if `core.experiments` is absent (a
    trimmed deployment, an older checkout replaying a STAMPED archive).

    An ImportError is NOT the same as "nothing was stamped": replaying a
    stamped archive with the module missing would drop the exclusion and print
    `experiment_context_excluded 0`, indistinguishable from a clean corpus —
    which violates this file's own "no silent caps" contract. It is recorded
    and reported instead (see `_experiment_filter_available`, which is what
    actually reports it; this branch is the belt to that braces).
    """
    try:
        from ..core.experiments import context_was_mutated
    except ImportError:
        _FILTER_UNAVAILABLE["hit"] = True
        return False
    try:
        return context_was_mutated(traj)
    except Exception as e:  # noqa: BLE001
        _FILTER_ERRORS["n"] += 1
        logger.debug("tool-fixture miner: context_was_mutated raised (%s: %s)",
                     type(e).__name__, e)
        return False


def _trajectory_index(trajectory_root: Path) -> Dict[str, Any]:
    """request_id (== Trajectory.session_id) -> trajectory, overlay applied.
    On duplicate session_ids the latest timestamp wins."""
    collector = TrajectoryCollector(root=trajectory_root, session_id="reader")
    index: Dict[str, Any] = {}
    for traj in collector.iter_trajectories():
        sid = getattr(traj, "session_id", "") or ""
        if not sid:
            continue
        prev = index.get(sid)
        if prev is None or getattr(traj, "timestamp", "") >= getattr(prev, "timestamp", ""):
            index[sid] = traj
    return index


def _bench_trajectory_index() -> Dict[str, Any]:
    """request_id (== extra["req_id"]) -> BENCH trajectory (§4BF 1c).

    Bench sessions are per-BANK (``session_id="bench-<bank>"``), so the
    production join key (session_id) is useless there — every bench
    record instead carries the solve turn's req_id in ``extra``, which
    is exactly what the recorder stamps on its records. Keyed separately
    from the production index (real joins keep their pre-1c semantics;
    a bench join happens only when the production index missed). Empty
    when no banks exist or the matrix stops admitting this consumer.
    """
    from ..core.admissibility import iter_bench_trajectories
    index: Dict[str, Any] = {}
    for traj in iter_bench_trajectories("gepa_tool_fixtures"):
        extra = getattr(traj, "extra", None) or {}
        rid = str(extra.get("req_id") or "") if isinstance(extra, dict) else ""
        if not rid:
            continue
        prev = index.get(rid)
        if prev is None or getattr(traj, "timestamp", "") >= getattr(prev, "timestamp", ""):
            index[rid] = traj
    return index


def mine_fixtures(
    recording_paths: Iterable[Path],
    trajectory_root: Path,
    *,
    era_cutoff_local: str = DEFAULT_ERA_CUTOFF_LOCAL,
    private_pct: int = 30,
    max_result_chars: int = 1500,
    exclude_mutated_context: bool = True,
) -> Tuple[List[ToolChoiceFixture], Dict[str, int]]:
    """Mine labeled tool-choice fixtures in one streaming pass.

    Returns (fixtures, stats). ``stats`` reports every drop reason —
    nothing is silently truncated: scanned / bad_ts / pre_era / malformed /
    no_choice / no_request_context / unjoined / honest_failure_excluded /
    unlabeled_outcome / experiment_context_excluded /
    experiment_filter_unavailable / experiment_filter_errors / positive /
    negative / paired_results.

    ``experiment_context_excluded == 0`` only means "nothing was excluded"
    when ``experiment_filter_unavailable == 0`` as well; the latter is
    resolved eagerly, before the scan, so it is a property of the
    environment rather than of whether any record survived the earlier
    filters.
    """
    _FILTER_UNAVAILABLE["hit"] = False
    _FILTER_ERRORS["n"] = 0
    cutoff = era_cutoff_utc(era_cutoff_local)
    index = _trajectory_index(trajectory_root)
    try:
        bench_index = _bench_trajectory_index()
    except Exception as e:  # noqa: BLE001 — bench is additive, never fatal
        logger.debug("tool-fixture miner: bench index skipped (%s)", e)
        bench_index = {}

    stats = {
        "scanned": 0, "bad_ts": 0, "pre_era": 0, "malformed": 0,
        "no_choice": 0, "no_request_context": 0, "unjoined": 0,
        "honest_failure_excluded": 0, "unlabeled_outcome": 0,
        "experiment_context_excluded": 0,
        # 1 when the steered-context exclusion could not run AT ALL — either
        # `--include-experiment-context` disabled it, or core.experiments is
        # unimportable here. Probed eagerly (see
        # `_experiment_filter_available`), so a zero in
        # `experiment_context_excluded` means "none excluded" only while this
        # is also 0; otherwise it means "unknown".
        "experiment_filter_unavailable": 0,
        # The filter was importable but RAISED on this many trajectories;
        # each of those was INCLUDED without ever being checked.
        "experiment_filter_errors": 0,
        "positive": 0, "negative": 0, "paired_results": 0,
        # §4BF 1c: fixtures whose trajectory join came from the BENCH index.
        "bench_joined": 0,
    }
    if not exclude_mutated_context:
        stats["experiment_filter_unavailable"] = 1
    elif not _experiment_filter_available():
        stats["experiment_filter_unavailable"] = 1
        logger.warning("tool-fixture miner: core.experiments unavailable — "
                       "steered-context exclusion will NOT run")

    fixtures: List[ToolChoiceFixture] = []
    # (session_id, ordinal) -> index of the fixture awaiting that record as
    # its result carrier. Recorder ordinals are per-process and append-only,
    # and day-files are iterated in date order, so the carrier always
    # arrives AFTER its choice. Only light fixtures are retained — never
    # full records.
    pending: Dict[Tuple[str, int], int] = {}

    for path, rec in iter_recording_records(recording_paths):
        stats["scanned"] += 1
        try:
            ts = _record_ts_utc(rec)
            if ts is None:
                stats["bad_ts"] += 1
                continue
            if ts < cutoff:
                stats["pre_era"] += 1
                continue

            sid = rec.get("session_id") or ""
            ordinal = rec.get("ordinal")
            key = ((sid, ordinal)
                   if sid and isinstance(ordinal, int) else None)

            # Complete a pending pair first: this record may be the result
            # carrier for the previous choice. A wrong-kind or wrong-request
            # occupant (same-request background call) consumes the slot but
            # contributes nothing — a lost pair, never a mispair.
            if key is not None and key in pending:
                fi = pending.pop(key)
                if (rec.get("kind") in _CHOICE_KINDS
                        and rec.get("request_id") == fixtures[fi].request_id):
                    payload = rec.get("payload")
                    messages = (payload.get("messages")
                                if isinstance(payload, dict) else []) or []
                    previews = _result_previews_from_messages(
                        messages, max_result_chars)
                    if previews:
                        fixtures[fi].tool_results = previews
                        stats["paired_results"] += 1

            if not is_choice_record(rec):
                stats["no_choice"] += 1
                continue
            request_id = rec.get("request_id") or ""
            if request_id in ("", "SYSTEM"):
                stats["no_request_context"] += 1
                continue
            # Production join first (pre-1c semantics untouched); the bench
            # index answers only what production missed, so a real turn can
            # never be shadowed by a bench row.
            traj = index.get(request_id)
            if traj is None:
                traj = bench_index.get(request_id)
                if traj is not None:
                    # Counted apart (R1 review): bench joins folding
                    # silently into positive/negative made the mix
                    # invisible to the miner's own accounting.
                    stats["bench_joined"] = stats.get("bench_joined", 0) + 1
            if traj is None:
                stats["unjoined"] += 1
                continue
            if exclude_mutated_context and _context_was_mutated(traj):
                # A live experiment's treatment rewrote this turn's context.
                # Replaying it would optimize against a prompt only one arm
                # ever sees. Counted, never silent.
                stats["experiment_context_excluded"] += 1
                continue
            label = _label_for(traj)
            if label is None:
                if getattr(traj, "outcome", "") == Outcome.PASSED.value:
                    stats["honest_failure_excluded"] += 1
                else:
                    stats["unlabeled_outcome"] += 1
                continue

            payload = rec.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            advertised = [
                (t.get("function") or {}).get("name", "")
                for t in payload.get("tools") or []
                if isinstance(t, dict)
            ]
            chosen = [
                {"name": (c.get("function") or {}).get("name", ""),
                 "arguments": (c.get("function") or {}).get("arguments", "") or ""}
                for c in _structured_tool_calls(rec)
            ]

            fixtures.append(ToolChoiceFixture(
                fixture_id=f"toolfx-{request_id}-{sid}-{ordinal}",
                request_id=request_id,
                ts=rec.get("ts") or "",
                user_request=getattr(traj, "user_request", "") or "",
                chosen_tools=chosen,
                advertised_tools=[n for n in advertised if n],
                label=label,
                outcome=getattr(traj, "outcome", ""),
                failure_reason=getattr(traj, "failure_reason", "") or "",
                tool_results=[],
                tier=holdout_tier(f"toolfx:{request_id}",
                                  private_pct=private_pct),
                source={"file": str(path), "session_id": sid,
                        "ordinal": ordinal},
                origin=(str(getattr(traj, "task_kind", "") or "")
                        or "user_request"),
            ))
            stats["positive" if label >= 0.5 else "negative"] += 1
            if key is not None:
                pending[(sid, ordinal + 1)] = len(fixtures) - 1
        except Exception as e:
            stats["malformed"] += 1
            logger.debug("tool-fixture miner: record skipped (%s: %s)",
                         type(e).__name__, e)

    if _FILTER_UNAVAILABLE["hit"]:
        stats["experiment_filter_unavailable"] = 1
        logger.warning("tool-fixture miner: core.experiments unavailable — "
                       "steered-context exclusion did NOT run")
    if _FILTER_ERRORS["n"]:
        stats["experiment_filter_errors"] = _FILTER_ERRORS["n"]
        logger.warning("tool-fixture miner: the steered-context filter raised "
                       "on %d trajectories — those were INCLUDED unchecked",
                       _FILTER_ERRORS["n"])
    return fixtures, stats


def write_fixtures_jsonl(fixtures: List[ToolChoiceFixture], path: Path) -> int:
    """Write fixtures as JSONL (atomic replace). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for fx in fixtures:
            fh.write(json.dumps(fx.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return len(fixtures)


def load_fixtures_jsonl(path: Path) -> List[ToolChoiceFixture]:
    """Per-line tolerant loader for a ``write_fixtures_jsonl`` file: torn,
    non-JSON, or field-incomplete lines are skipped, never fatal."""
    out: List[ToolChoiceFixture] = []
    known = set(ToolChoiceFixture.__dataclass_fields__)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                out.append(ToolChoiceFixture(
                    **{k: v for k, v in data.items() if k in known}))
            except (json.JSONDecodeError, TypeError):
                continue
    return out
