"""§4CN E1 — propose one scaffold mutation, from evidence rather than vibes.

The loop picks a parent from the archive, picks a TARGET the evidence
actually indicts, asks the main model for a small unified diff, and
materialises a candidate snapshot. It proposes; it never applies.

WHY "EVIDENCE, NOT VIBES" IS A MECHANISM AND NOT A SLOGAN. A mutator with
no evidence floor asks a model to improve a file it has no complaint
about, and the model — being a model — will produce a plausible diff
anyway. That diff is then evaluated by a cascade whose stage-3 power is
in single digits per night, so a stream of ungrounded candidates is a
stream of coin flips dressed as evolution. So a target with fewer than
:data:`MIN_EVIDENCE_ITEMS` recorded failures is NOT ELIGIBLE, and a run
with no eligible target produces nothing and says so.

THE THREE SOURCES, and the honesty rule they carry.

* **foresight** — the per-tool transition ledger. Real observed failures
  with their error heads, already redacted at write time.
* **postmortem** — `defects.jsonl`, filed by the postmortem engine.
* **dream_credit** — replay verdicts touching that tool.

Exactly one of those has data on this box today, and that is the point of
:class:`Evidence` carrying ``present`` and ``reason`` rather than just a
list. A brief built from one source is not the same artefact as a brief
built from three, and a mutator that silently degrades to whichever
source happens to be populated is inventing the difference. Every brief
names the sources that were empty and why, and every ledger row records
which sources contributed.

⚠ `dream_credit` is ADMISSIBILITY-GATED. `core/admissibility.py` has it
`real_only`, and §4CM D4 — the sensitivity/specificity gate on the replay
engine — has not returned PASS. The source is wired and OFF: it reports
its own gate instead of quietly contributing labels nobody has validated.
Flipping `GHOST_EVOLVE_DREAM_CREDIT=1` before D4 passes is how an
unvalidated label source opens, which is the §4AO/§4BE failure this
project has now paid for twice.

WHAT MAY BE MUTATED is `evolve/fence.py`'s allow-list, and the mapping
from a tool NAME to the file that implements it is resolved by walking
the dispatch entry's closure — executed fact — rather than guessed from
the name. `registry.py` is immutable and every dispatch entry mentions
it, so a name-based guess would map every tool to an immutable file and
the mutator would silently propose nothing forever.
"""
from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import logging
import os
import re
import stat
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import archive as A
from . import fence as F

logger = logging.getLogger("GhostAgent")

#: A target needs at least this many recorded failures before a model is
#: asked to change it. Below it there is nothing to work from and the
#: brief would be a request to invent a problem.
MIN_EVIDENCE_ITEMS = 3

#: How much of the foresight ledger to read. Bounded, like every other
#: reader of that file.
FORESIGHT_TAIL_BYTES = 2_000_000

#: Distinct error heads carried into the brief per target. Enough to show
#: a pattern, few enough that the brief stays a brief.
MAX_ERRORS_IN_BRIEF = 6
MAX_BRIEF_BYTES = 8000

#: Bounded reads. Every evidence source is a file the agent itself can
#: grow, and one of them lives next to a directory the model writes to.
MAX_SOURCE_BYTES = 2_000_000
MAX_SOURCE_ROWS = 5000

_DEFECTS_FILE = "defects.jsonl"

#: Attempts per run. A rejected diff gets ONE retry with the rejection
#: reason fed back, because "≤2 files" is a rule a model can comply with
#: once told. More than that is paying for the same mistake.
MAX_ATTEMPTS = 2

SOURCE_FORESIGHT = "foresight"
SOURCE_POSTMORTEM = "postmortem"
SOURCE_CREDIT = "dream_credit"

LEDGER_FILE = "mutations.jsonl"
WORK_DIRNAME = "work"

#: Outcomes recorded per run. A night that produced nothing must say
#: WHICH nothing: an archive with no eligible parent, a corpus with no
#: indicted target, a model that would not comply, and a snapshot that
#: would not apply are four different problems with one symptom.
OUT_PROPOSED = "proposed"
OUT_NO_EVIDENCE = "no_evidence"
OUT_NO_MODEL = "no_model_output"
OUT_REJECTED = "rejected"
OUT_DUPLICATE = "duplicate"
OUT_MATERIALIZE_FAILED = "materialize_failed"
OUT_DISABLED = "disabled"


def _enabled() -> bool:
    return os.getenv("GHOST_EVOLVE", "").lower() in ("1", "true", "yes", "on")


def _credit_source_enabled() -> bool:
    return os.getenv("GHOST_EVOLVE_DREAM_CREDIT", "").lower() in (
        "1", "true", "yes", "on")


def _state_dir(home: str = None) -> Optional[Path]:
    # `home if home is not None` — matching `archive._archive_dir`. With
    # `or`, an explicit `home=""` fell back to the env and sent the
    # ledger to `$GHOST_HOME` while the archive resolved to None: two
    # halves of one run writing to different places.
    base = (home if home is not None
            else os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return None
    return Path(base) / "system" / "evolve"


# ---------------------------------------------------------------------------
# Which file implements which tool
# ---------------------------------------------------------------------------

def _module_to_path(module_name: str) -> str:
    """`ghost_agent.tools.file_system` → `src/ghost_agent/tools/file_system.py`."""
    if not module_name.startswith("ghost_agent."):
        return ""
    return "src/" + module_name.replace(".", "/") + ".py"


def implementation_paths(fn) -> List[str]:
    """Repo-relative MUTABLE files a dispatch entry actually calls into.

    Walks the entry's closure cells and its code object's global names for
    callables defined in this package. `inspect.getsourcefile` alone is
    useless here: every dispatch entry is a closure defined inside
    `registry.get_available_tools`, so it reports `registry.py` — which is
    immutable — for all 41 tools.

    Fails CLOSED: anything unresolvable, or resolving only to immutable
    files, yields an empty list and the tool is simply not a target.
    """
    out: List[str] = []
    try:
        fn = inspect.unwrap(fn)
    except Exception:  # noqa: BLE001
        return out
    mods = set()
    for cell in (getattr(fn, "__closure__", None) or ()):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        mod = getattr(val, "__module__", "")
        if callable(val) and str(mod).startswith("ghost_agent."):
            mods.add(mod)
    code = getattr(fn, "__code__", None)
    globs = getattr(fn, "__globals__", None) or {}
    if code is not None:
        for name in set(code.co_names) | set(code.co_freevars or ()):
            val = globs.get(name)
            mod = getattr(val, "__module__", "")
            if callable(val) and str(mod).startswith("ghost_agent."):
                mods.add(mod)
    for mod in sorted(mods):
        path = _module_to_path(mod)
        if path and F.is_mutable(path)[0]:
            out.append(path)
    return out


def tool_target_map(tools: Dict[str, Any]) -> Dict[str, List[str]]:
    """{tool name: [mutable implementation paths]}, empty entries dropped."""
    out: Dict[str, List[str]] = {}
    for name, fn in (tools or {}).items():
        paths = implementation_paths(fn)
        if paths:
            out[str(name)] = paths
    return out


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """One source's contribution, INCLUDING its absence.

    ``present`` is whether the source could be read at all; ``reason``
    says why not. A source that is missing, gated or empty must be
    distinguishable in the record from one that was read and had nothing
    to say about this target — they look identical in a count.
    """
    source: str
    present: bool = False
    reason: str = ""
    #: {tool: {"fails": int, "total": int, "errors": [str]}}
    by_tool: Dict[str, Dict[str, Any]] = field(default_factory=dict)

def _blank(source: str, reason: str) -> Evidence:
    return Evidence(source=source, present=False, reason=reason)


#: REFUSAL SHAPES the authoritative list does not cover, because that
#: list answers a different question — it enumerates results the DISPATCH
#: PIPELINE mints for calls that never executed, while these are emitted
#: by a tool that RAN and refused. A first version enumerated two exact
#: messages and a fresh-eye review found seven emitters it missed,
#: including the SSRF guard and the destructive-operation guard, both of
#: which live in `tools/file_system.py` — inside the mutable fence.
#:
#: So this is deliberately SHAPES, not messages, and deliberately
#: OVER-INCLUSIVE. A false exclusion costs a rank position; a false
#: inclusion costs a security guard. Measured against the live 751-row
#: ledger: 9 of 73 failure rows excluded, and all 9 are genuine refusals
#: — no real failure is lost at this width.
#:
#: ⚠ THIS IS RANKING HYGIENE. THERE IS NO LEXICAL SECURITY GATE — three
#: rounds of review killed three attempts at one, and `guard_flags`
#: REJECTS NOTHING; it surfaces removed refusal-shaped lines to the
#: operator. This comment previously named `_weakens_a_guard` as "the
#: boundary" and survived that function's deletion by a round, telling a
#: reader a gate existed that did not.
_EXTRA_REFUSAL_MARKERS: Tuple[str, ...] = (
    # "SYSTEM INSTRUCTION: replace rejected — …" (tools/file_system.py,
    # the identical-args corruption guard). NOTE `"system block"` is NOT
    # here: `_guard_markers` already borrows `"SYSTEM BLOCK"` from
    # foresight and lowercases it, so listing it again was a duplicate
    # that no test could distinguish from a live entry.
    "system instruction:",
    # "REJECTED: that replace would introduce a syntax error",
    # "REJECTED: that replace would have written …",
    # "Block REJECTED: its SEARCH text is …" (tools/file_system.py).
    "rejected",
    # "Security Error: refusing to run a destructive operation",
    # "Reading '…' is refused — the conversation is near the context
    # ceiling", "the 'replace' operation refuses files larger than",
    # "vision refuses files > N MB" (file_system.py, vision.py).
    "refus",
    "security error",
    # "Error: download redirect blocked (SSRF)" (file_system.py).
    "ssrf",
    "deny-listed",
    # NOTE: `"not allowed"` was here and is gone. It matched nothing in
    # the live 751-row ledger and nothing in the tools, while being the
    # phrase most likely to swallow a genuine HTTP 403/405 — an
    # exclusion with all of the cost and none of the coverage.
)


def _guard_markers() -> Tuple[str, ...]:
    """The scaffold's refusal markers: the authoritative pipeline list
    plus the shapes above.

    The first half is BORROWED from `foresight._SYNTHETIC_RESULT_PREFIXES`
    rather than copied — a second list of "what a refusal looks like"
    drifts from the first, and the first is maintained by the subsystem
    that has to get it right.
    """
    from ..core.foresight import _SYNTHETIC_RESULT_PREFIXES
    return tuple(p.strip().lower() for p in _SYNTHETIC_RESULT_PREFIXES
                 if p.strip()) + _EXTRA_REFUSAL_MARKERS


def is_guard_refusal(text: str) -> bool:
    """True when a failure is the scaffold REFUSING — a guard firing, an
    idempotency dedupe, a pre-flight block, a rejected argument.

    ⚠ WHY THIS EXISTS, and it is the sharpest correctness point in this
    module. The evidence for `execute` on this box includes
    ``SYSTEM BLOCK: shell command rejected by pre-execution validator:
    deny-listed pattern``. That is the security guard WORKING. Feeding it
    to a model under the instruction "propose the smallest change that
    would prevent this failure" is a request to weaken the deny-list —
    and `execute.py` is inside the mutable fence while `validators.py` is
    not, so the fence would not stop the resulting diff. A guard that
    fires is not a defect; it is the only evidence the guard is alive.

    Matching is CONTAINMENT, not `startswith` like
    `foresight.is_synthetic_result`, because the ledger stores the tool's
    whole result envelope (``--- execution result --- exit code: 1
    stdout/stderr: system block: …``) and lowercases it, so a prefix test
    finds none of them. Verified against the live ledger: prefix 0,
    containment several.

    ⚠ KNOWN LIMIT. A tool that refuses in prose nobody enumerated still
    reaches the brief. FOUR things stand behind this filter, none of them
    lexical except the first: `guard_flags` surfaces removed
    refusal-shaped lines to the operator; the brief states that a guard
    firing is not a defect and that an exit code must never be hidden;
    the write fence keeps the evaluator and the tool registry out of
    reach (asked of the FILESYSTEM, not of a case-fold); and promotion is
    operator-applied. A blacklist over text nobody controls is the
    weakest of the four, which is why it flags rather than judges.
    """
    low = str(text or "").lower()
    if not low:
        return False
    return any(m in low for m in _guard_markers())


def foresight_evidence(home: str = None,
                       tail_bytes: int = FORESIGHT_TAIL_BYTES) -> Evidence:
    """Per-tool failures from the foresight transition ledger."""
    base = (home or os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return _blank(SOURCE_FORESIGHT, "no GHOST_HOME")
    path = Path(base) / "system" / "foresight" / "predictions.jsonl"
    if not path.exists():
        return _blank(SOURCE_FORESIGHT,
                      "no ledger yet — it appears on the first resolved "
                      "prediction after a boot")
    try:
        _guard_markers()
    except Exception as exc:       # noqa: BLE001
        # No filter ⇒ no evidence. Unfiltered evidence is worse than
        # none: it is the only path by which a guard becomes a target.
        return _blank(SOURCE_FORESIGHT,
                      f"refusal markers unreadable ({exc}) — refusing to "
                      f"build a brief from unfiltered failures")
    ev = Evidence(source=SOURCE_FORESIGHT, present=True)
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - int(tail_bytes)))
            raw = fh.read()
        lines = raw.decode("utf-8", "replace").splitlines()
        if size > tail_bytes and lines:
            lines = lines[1:]      # the first line may be a partial record
    except Exception as exc:       # noqa: BLE001
        return _blank(SOURCE_FORESIGHT, f"unreadable ({exc})")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:          # noqa: BLE001
            continue
        tool = str(rec.get("tool") or "")
        if not tool:
            continue
        cell = ev.by_tool.setdefault(tool, {"fails": 0, "total": 0,
                                            "errors": []})
        cell["total"] += 1
        if rec.get("ok", True):
            continue
        err = str(rec.get("err") or "").strip()
        if is_guard_refusal(err):
            # The scaffold refused. Counting it as evidence AGAINST the
            # file that refused is how a mutator ends up asking for a
            # guard to be removed.
            cell["guard_refusals"] = cell.get("guard_refusals", 0) + 1
            continue
        cell["fails"] += 1
        if err and err not in cell["errors"]:
            cell["errors"].append(err[:300])
    if not ev.by_tool:
        ev.reason = "ledger read, no rows"
    return ev


def postmortem_evidence(home: str = None) -> Evidence:
    """Defect reports filed by the postmortem engine."""
    base = (home or os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return _blank(SOURCE_POSTMORTEM, "no GHOST_HOME")
    # ⚠ ONE EXACT PATH, never `rglob`. `$GHOST_HOME` contains
    # `sandbox/`, which is the model's own writable workspace — five
    # project directories on this box that `file_system` and `execute`
    # write to freely. An `rglob("defects.jsonl")` over 14,759 files
    # accepts anything the agent chose to name that, and every field of
    # it lands verbatim in a brief that asks a model to rewrite a tool.
    # That is a prompt-injection channel from the sandbox into the
    # self-modification loop, and the fence does not close it: the fence
    # bounds WHICH file gets rewritten, not WHY.
    path = Path(base) / "system" / "postmortem" / _DEFECTS_FILE
    if not path.is_file():
        return _blank(SOURCE_POSTMORTEM,
                      f"no {path.name} at the canonical root — the "
                      f"postmortem engine is behind --postmortem and has "
                      f"not run on this box")
    ev = Evidence(source=SOURCE_POSTMORTEM, present=True)
    seen_sigs = set()
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - MAX_SOURCE_BYTES))
            raw = fh.read()
        lines = raw.decode("utf-8", "replace").splitlines()
        if size > MAX_SOURCE_BYTES and lines:
            lines = lines[1:]
    except Exception as exc:       # noqa: BLE001
        return _blank(SOURCE_POSTMORTEM, f"unreadable ({exc})")
    for line in lines[-MAX_SOURCE_ROWS:]:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:          # noqa: BLE001
            continue
        tool = str(rec.get("tool") or rec.get("tool_name") or "")
        if not tool:
            continue
        # One defect per signature. The engine re-files a recurring
        # pathology, and counting each filing separately would let one
        # repeated defect clear an evidence floor by itself.
        sig = str(rec.get("signature_hash") or rec.get("signature") or "")
        if sig:
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
        head = str(rec.get("summary") or rec.get("title")
                   or rec.get("signature") or "").strip()
        if is_guard_refusal(head):
            cell = ev.by_tool.setdefault(tool, {"fails": 0, "total": 0,
                                                "errors": []})
            cell["guard_refusals"] = cell.get("guard_refusals", 0) + 1
            cell["total"] += 1
            continue
        cell = ev.by_tool.setdefault(tool, {"fails": 0, "total": 0,
                                            "errors": []})
        cell["fails"] += 1
        cell["total"] += 1
        if head and head not in cell["errors"]:
            cell["errors"].append(head[:300])
    if not ev.by_tool:
        ev.reason = "defects.jsonl read, no rows naming a tool"
    return ev


def credit_evidence(home: str = None) -> Evidence:
    """Replay verdicts touching a tool — ADMISSIBILITY-GATED.

    Returns an absent source with its gate as the reason unless BOTH the
    operator flag is on and the admissibility row still says what it says
    now. There is no path here that reads a verdict without saying so.
    """
    try:
        from ..core import admissibility as ADM
        policy = ADM.ADMISSIBILITY.get("dream_credit")
        real_only = (policy == ADM.POLICY_REAL_ONLY)
    except Exception:              # noqa: BLE001
        policy, real_only = "unknown", True
    if not _credit_source_enabled():
        return _blank(SOURCE_CREDIT,
                      f"OFF by default: `dream_credit` is {policy} and §4CM "
                      f"D4 has not returned PASS. Set "
                      f"GHOST_EVOLVE_DREAM_CREDIT=1 only after it does")
    if real_only:
        # The flag alone must not open it. The admissibility row is the
        # authority, and it is a reviewable file.
        return _blank(SOURCE_CREDIT,
                      "GHOST_EVOLVE_DREAM_CREDIT is set but the "
                      "admissibility row for `dream_credit` is still "
                      "real_only — the row is the authority, not the flag")
    base = (home or os.getenv("GHOST_HOME", "")).strip()
    ev = Evidence(source=SOURCE_CREDIT, present=True)
    try:
        from ..core.replay_engine import iter_credits
        rows = list(iter_credits(base))
    except Exception as exc:       # noqa: BLE001
        return _blank(SOURCE_CREDIT, f"credits unreadable ({exc})")
    for rec in rows[-MAX_SOURCE_ROWS:]:
        tool = str(rec.get("target") or "")
        if not tool:
            continue
        cell = ev.by_tool.setdefault(tool, {"fails": 0, "total": 0,
                                            "errors": []})
        cell["total"] += 1
        why = str(rec.get("why") or "")
        if is_guard_refusal(why):
            cell["guard_refusals"] = cell.get("guard_refusals", 0) + 1
            continue
        if str(rec.get("verdict") or "").startswith("mattered"):
            cell["fails"] += 1
            if why and why not in cell["errors"]:
                cell["errors"].append(why[:300])
    if not ev.by_tool:
        ev.reason = "credits read, no rows"
    return ev


def gather_evidence(home: str = None) -> List[Evidence]:
    return [foresight_evidence(home), postmortem_evidence(home),
            credit_evidence(home)]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

@dataclass
class Target:
    tool: str
    paths: List[str]
    fails: int = 0
    total: int = 0
    errors: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    #: Failures EXCLUDED because they were the scaffold refusing. Carried
    #: so the brief can say the number rather than quietly shrink.
    guard_refusals: int = 0
    #: Per-source {source: {"fails": n, "total": n}}. Kept because the
    #: sources OVERLAP — the postmortem engine files defects derived from
    #: the same failed runs the foresight ledger records, with no shared
    #: key to join on — so a summed count silently halves the evidence
    #: floor, and a summed rate is arithmetic over two different
    #: denominators.
    by_source: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def dominant_source(self) -> str:
        if not self.by_source:
            return ""
        return max(self.by_source.items(),
                   key=lambda kv: kv[1].get("fails", 0))[0]

    @property
    def independent_fails(self) -> int:
        """The most any SINGLE source saw. The eligibility floor is
        measured against this, not against the sum."""
        return max((c.get("fails", 0) for c in self.by_source.values()),
                   default=0)

    @property
    def fail_rate(self) -> Optional[float]:
        """Computed within the dominant source only."""
        cell = self.by_source.get(self.dominant_source) or {}
        total = cell.get("total") or 0
        return round(cell.get("fails", 0) / total, 3) if total else None


def rank_targets(evidences: List[Evidence],
                 targets: Dict[str, List[str]]) -> List[Target]:
    """Eligible targets, most-indicted first.

    Ranked by absolute FAILURE COUNT, not failure rate: one failure in one
    call is a 100% rate and no evidence at all, and a ranking that puts it
    above fifteen failures in a hundred-and-fifty is a ranking that
    chases noise. The rate is carried for the brief; it does not order.

    ⚠ ELIGIBILITY IS PER-SOURCE. The floor is measured against the most
    any SINGLE source saw, never the sum, because the sources overlap
    with no key to join on: the postmortem engine files defects derived
    from the same failed runs the foresight ledger records, so two real
    failures seen twice would clear a floor of three.
    """
    merged: Dict[str, Target] = {}
    for ev in evidences:
        if not ev.present:
            continue
        for tool, cell in ev.by_tool.items():
            paths = targets.get(tool)
            if not paths:
                continue           # not mutable, or unresolvable → not a target
            tgt = merged.setdefault(tool, Target(tool=tool, paths=list(paths)))
            tgt.fails += int(cell.get("fails") or 0)
            tgt.total += int(cell.get("total") or 0)
            tgt.guard_refusals += int(cell.get("guard_refusals") or 0)
            tgt.by_source[ev.source] = {
                "fails": int(cell.get("fails") or 0),
                "total": int(cell.get("total") or 0)}
            for err in (cell.get("errors") or []):
                if err not in tgt.errors:
                    tgt.errors.append(err)
            if int(cell.get("fails") or 0) and ev.source not in tgt.sources:
                tgt.sources.append(ev.source)
    out = [t for t in merged.values()
           if t.independent_fails >= MIN_EVIDENCE_ITEMS]
    out.sort(key=lambda t: (-t.independent_fails, -t.fails, t.tool))
    return out


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------

#: ⚠ THE BRIEF MUST CONTAIN THE FILE. Measured: without it the model was
#: asked for a unified diff — exact context lines, exact line numbers —
#: against a file it had never seen, and did the only thing it could:
#: invented a plausible `class ExecuteTool(BaseTool)` for a module that
#: has no classes at all. `patch --dry-run` refused all four hunks and
#: `materialize` failed closed, so nothing downstream was corrupted, but
#: E1 could not produce an applicable diff except by luck.
#:
#: A file bigger than this is REFUSED rather than briefed blind: asking
#: for exact-context hunks against an unseen file is a request that
#: cannot succeed, and spending a model call on it is the same error in
#: a cheaper form.
MAX_BRIEF_SOURCE_BYTES = 120_000


def _numbered_source(path: Path) -> Tuple[str, str]:
    """(rendered source, refusal reason). Exactly one is non-empty."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return "", f"could not read {path.name}: {type(exc).__name__}"
    n = len(raw.encode("utf-8", "replace"))
    if n > MAX_BRIEF_SOURCE_BYTES:
        return "", (f"{path.name} is {n:,} bytes, over the "
                    f"{MAX_BRIEF_SOURCE_BYTES:,}-byte brief budget — a diff "
                    f"cannot be written against a file the model cannot see")
    width = len(str(raw.count("\n") + 1))
    body = "\n".join(f"{i:>{width}}| {line}"
                      for i, line in enumerate(raw.splitlines(), 1))
    return body, ""


_BRIEF_TEMPLATE = """You are improving ONE file of a running agent's own scaffold.

TARGET FILE: {path}
TOOL IT IMPLEMENTS: {tool}

THE FILE, AS IT IS ON DISK RIGHT NOW. Line numbers are shown for your
reference and are NOT part of the file — do not include them in the diff,
and count context lines from the text after the `|`.

<<<BEGIN {path}>>>
{source}
<<<END {path}>>>

THE EVIDENCE AGAINST IT — {fails} recorded failures out of {total} calls\
{rate_note}, from: {sources}.
These are real error heads from the agent's own transition ledger:

{errors}
{guard_note}
⚠ READ THE EVIDENCE HONESTLY. Some of these are the COMMAND or ARGUMENT the
tool was given failing, not the tool failing. Never make the tool hide,
swallow or rewrite a non-zero exit code, and never widen or remove a
safety check to make a failure go away — a guard that fires is the guard
working. Preventing an avoidable precondition failure, or turning an
unhelpful error into an actionable one, IS in scope.

SOURCES THAT CONTRIBUTED NOTHING, and why (do not treat their silence as
agreement):
{empty}

YOUR TASK. Propose the smallest change to {path} that would prevent or
correctly handle one of the failures above. Do not refactor. Do not add
features. Do not rename anything a caller depends on.

HARD CONSTRAINTS — a diff that breaks any of them is rejected mechanically,
without being read:
  * unified diff format ONLY, `--- a/<path>` / `+++ b/<path>` headers;
  * **{path} and nothing else** — a diff touching any other file is
    rejected without being read, even a file on the allow-list;
  * at most {max_lines} changed lines;
  * no changes to tests, evaluation harnesses, or scripts under any
    circumstances — those score your work.

Output the diff and nothing else. No prose, no fences, no explanation.
"""


def _format_empty(evidences: List[Evidence]) -> str:
    rows = [f"  * {ev.source}: {ev.reason or 'no rows'}"
            for ev in evidences if not ev.present]
    return "\n".join(rows) if rows else "  * (none — all three had data)"


def build_brief(target: Target, evidences: List[Evidence],
                repo_root: Path = None) -> str:
    """⚠ `repo_root` IS NOT OPTIONAL IN SPIRIT. The target's path is
    repo-relative, so resolving it against the process CWD makes the
    brief depend on where the interpreter was started — and under
    launchd the daemon's CWD is not the repo. Same default as
    `materialize`, so the two agree on what "the repo" means.
    """
    errors = target.errors[:MAX_ERRORS_IN_BRIEF]
    body = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(errors)) or \
        "  (failures recorded with no error text)"
    rate = target.fail_rate
    guard_note = ""
    if target.guard_refusals:
        guard_note = (
            f"\n({target.guard_refusals} further failure(s) were EXCLUDED "
            f"from this brief because they were the scaffold's own guards "
            f"refusing — those are not defects and are not yours to "
            f"remove.)\n")
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    source, refusal = _numbered_source(root / target.paths[0])
    if refusal:
        # ⚠ REFUSE, DO NOT BRIEF BLIND. A caller that gets "" skips the
        # target; one that got a brief without the file would spend a
        # model call on a request that cannot succeed. Said out loud,
        # because a silently skipped target looks like a target with no
        # evidence.
        logger.info("evolve: no brief for %s — %s", target.tool, refusal)
        return ""
    brief = _BRIEF_TEMPLATE.format(
        source=source,
        path=target.paths[0], tool=target.tool, fails=target.fails,
        total=target.total, guard_note=guard_note,
        rate_note=(f" ({rate:.1%} of calls in {target.dominant_source})"
                   if rate is not None else ""),
        sources=", ".join(target.sources) or "unknown",
        errors=body, empty=_format_empty(evidences),
        max_lines=A.MAX_CHANGED_LINES)
    # ⚠ THE CAP BOUNDS THE PROSE, NOT THE FILE. `MAX_BRIEF_BYTES` exists
    # so runaway error text cannot blow the prompt; spending it on the
    # embedded source truncated the HARD CONSTRAINTS off the end — the
    # brief kept the evidence and dropped the rules the diff is judged
    # by. The file has its own budget (`MAX_BRIEF_SOURCE_BYTES`), so the
    # two are capped separately and the total is bounded by their sum.
    head, _, tail = brief.partition(source) if source else (brief, "", "")
    prose = len(head) + len(tail)
    if prose > MAX_BRIEF_BYTES:
        tail = tail[:max(0, MAX_BRIEF_BYTES - len(head))]
    return head + source + tail


# ---------------------------------------------------------------------------
# The proposal
# ---------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    """Unwrap a fenced reply, wherever the fence starts.

    The first version only unwrapped a reply that STARTS with a fence, so
    a model that prefaced its diff with one sentence left the markdown in
    `candidate.diff` and `patch` rejected the whole thing — a compliant
    proposal lost to a preamble.
    """
    t = str(text or "").strip()
    if "```" in t:
        chunks = t.split("```")
        if len(chunks) >= 3:
            body = chunks[1]
            # drop an info string like ```diff
            if "\n" in body:
                first, rest = body.split("\n", 1)
                if first.strip() and " " not in first.strip():
                    body = rest
            return body.strip()
        t = chunks[-1]
    return t.strip()


async def propose_diff(llm_client, brief: str, *, model: str = "",
                       rejection: str = "") -> str:
    """Ask the main model for one unified diff. "" when nothing usable."""
    if llm_client is None:
        return ""
    prompt = brief
    if rejection:
        prompt += (f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: {rejection}\n"
                   f"Fix exactly that and output the diff again.")
    try:
        payload = {
            "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
            "temperature": 0.2, "max_tokens": 1800, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if model:
            payload["model"] = model
        result = await llm_client.chat_completion(payload, is_background=True)
        text = (result.get("choices", [{}])[0].get("message", {})
                .get("content", "") or "")
    except Exception as exc:       # noqa: BLE001
        logger.warning("evolve: diff proposal failed (%s)", exc)
        return ""
    return _strip_fence(text)


_DIFF_HEADER_RE = re.compile(r"^--- ", re.M)
#: A header `patch -p1` can strip a component from. A model that emits
#: `--- src/ghost_agent/tools/file_system.py` (no `a/`) writes a diff the
#: fence ACCEPTS — `fence._norm` strips nothing and the path is on the
#: allow-list — and that `patch -p1` then cannot apply, because it looks
#: for `ghost_agent/tools/file_system.py`. Each one costs both attempts
#: and a 12 MB snapshot, and reads in the ledger as a bad diff rather
#: than a header convention.
_DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+) +(\S+)", re.M)

#: Refusal-shaped source lines — a CHEAP PRE-CHECK on removed lines.
#:
#: ⚠ THE FIRST VERSION OF THIS REGEX WAS FICTION, and the measurement is
#: worth keeping. It was written alongside its own test fixtures, from a
#: vocabulary nobody checked against the tree: `_is_allowed`, `_DENY_LIST`,
#: `_ALLOW_LIST`, `SecurityError`, `validate_args` — **none of which
#: exist anywhere in `src/ghost_agent/tools/`**. Measured against seven
#: REAL guard lines pulled out of the tools, 4 of 4 invented fixtures
#: matched and **5 of 7 real lines missed**, and 12 of its 20
#: alternatives matched nothing in the repo at all. A guard written from
#: an imagined vocabulary, tested against the same imagination:
#: `harness-grades-own-homework` and `guard-a-proxy-not-the-thing` in one
#: artifact.
#:
#: The alternatives below are the ones measured to appear on real
#: enforcement lines IN THE MUTABLE FENCE — the only place a removed
#: line can come from. ⚠ The previous measurement used the whole package
#: as its denominator, which INCLUDED THIS MODULE, so `is_dangerous`
#: "matched" nothing but the regex reading its own source, and five more
#: alternatives survived only on files a candidate can never edit
#: (`validators.py`, where the real deny-list lives, is IMMUTABLE — its
#: enforcement CALL SITES in `execute.py` are what a candidate can
#: reach). Every measurement here has to be scoped to what the mechanism
#: can actually see.
#:
#: It feeds `guard_flags`, which REPORTS and does not judge — see that
#: docstring for the three gates this replaced and why none of them
#: worked, and for what actually bounds this loop.
_GUARD_LINE_RE = re.compile(
    # measured on tools/*.py — refusal MESSAGES the tools emit
    r"SYSTEM\s+BLOCK|SECURITY\s+ERROR|REJECTED|\brefus|"
    # …and the identifiers those guards are actually built from
    r"SSRF|_DENY_|deny.?list|allow.?list|validate|sanitiz|"
    r"is_relative_to|\b_roots\b|\.resolve\(\)|IMMUTABLE|"
    r"\bforbidden\b|\bdenied\b|\bblocked\b",
    re.I)


def guard_flags(diff: str) -> List[str]:
    """Removed lines that LOOK like safety machinery — **surfaced to the
    operator, never a rejection**.

    ⚠ THREE ROUNDS OF REVIEW KILLED THREE VERSIONS OF A LEXICAL GATE
    HERE, and the honest conclusion is that there should not be one.

    * v1 rejected a diff removing a "refusal-shaped" line. Measured
      against guard lines pulled out of the tools: **4 of 4 invented test
      fixtures matched, 5 of 7 real lines missed**, and 12 of 20
      alternatives matched nothing in the repo. Written from an imagined
      vocabulary, tested against the same imagination.
    * v2 counted refusal-emitting lines before and after the patch and
      rejected any decrease. Its marginal coverage contained **zero**
      guards; 8 of 15 markers appeared nowhere a candidate can edit; and
      the count went **UP** when a real refusal was deleted alongside two
      added comments. Its false positives were terminal — outside the
      retry loop, blacklisting the diff forever.
    * v3 widened the vocabulary until it reached the real lines — and
      **33% of its 382 matches across the mutable fence are pure
      comments**. Six verbatim, entirely ordinary lines from the tree are
      rejected by it: a task-status set `{"DONE","FAILED","BLOCKED"}`, a
      cache key `str(Path(base_dir).resolve())`, a health string
      `"Internet: Disconnected or Blocked"`, a prompt legend, a comment
      beside a timeout, and `_invalidate_form_token(...)` (matched on
      "VALIDATE"). At those rates a 30-line hunk trips it 31–73% of the
      time depending on the file, and with `MAX_ATTEMPTS = 2` one hit
      costs half the night.

    A lexical test cannot decide a semantic property. Each round made the
    recall better and the precision worse, and none of them could see the
    attacks that matter — an early `return` above a guard, an `or True`
    in a condition, a changed constant, a guard moved into dead code.

    **So it flags and does not judge.** The list rides on the proposal —
    the archive node, the ledger row, the operator's activity line and
    the health report — where whoever applies the patch can read it. Its
    honest reach, measured over a corpus selected by names the pattern
    does not contain, is **11%** of enforcement-adjacent lines in the
    fence; every larger figure quoted for it in earlier rounds was
    inflated by selectors that were themselves alternatives of the regex
    being measured. WHAT ACTUALLY BOUNDS THIS LOOP: the write fence (an
    allow-list, case-folded, with the evaluator outside it), the
    target-file constraint, the diff-shape caps, the throwaway snapshot,
    and the fact that promotion is operator-applied.
    """
    out: List[str] = []
    for line in str(diff or "").splitlines():
        if line.startswith("-") and not line.startswith("---"):
            body = line[1:]
            if _GUARD_LINE_RE.search(body):
                out.append(body.strip()[:140])
    return out


def validate_diff(diff: str, arch: "A.Archive" = None,
                  target_paths=None) -> Tuple[bool, str]:
    """Every mechanical rejection, cheapest first. Never reads intent."""
    # NOTE: no CRLF normalisation here. It was added defensively and
    # measured inert — `str.splitlines()` already splits on `\r\n`, and
    # `archive.normalized_diff_hash` strips each line, so a 1,168-input
    # LF/CRLF differential found zero verdict divergences with it
    # removed. The one in `materialize` is real: `patch` fails every
    # hunk on CRLF, and that one IS pinned.
    text = str(diff or "").strip()
    if not text:
        return False, "empty"
    if not _DIFF_HEADER_RE.search(text):
        return False, "not a unified diff (no `---` header)"
    for name in _DIFF_FILE_RE.findall(text):
        if name == "/dev/null":
            continue
        head = name.split("/", 1)[0]
        if head not in ("a", "b"):
            # `fence._norm` strips `a/`/`b/`, so `--- src/…` normalises to
            # an ALLOWED path and passes the scope check — and then
            # `patch -p1` strips `src` and looks for `ghost_agent/…`,
            # which does not exist. Validating and then failing to apply
            # burns both attempts and a 12 MB snapshot, and reads in the
            # ledger as a bad diff rather than a header convention.
            return False, (f"header {name!r} is not `a/<path>` or "
                           f"`b/<path>` — `patch -p1` cannot strip it")
    paths = A.diff_touched_paths(text)
    if not paths:
        return False, "no touched paths could be parsed"
    ok, reason = A.check_diff_shape(text)
    if not ok:
        return False, reason
    ok, bad = F.check_diff_scope(paths)
    if not ok:
        return False, f"touches paths outside the fence: {', '.join(bad)}"
    if target_paths:
        # Through the fence's OWN normaliser: `diff_touched_paths`
        # returns raw header names (`a/…`, `b/…`), which is what
        # `check_diff_scope` normalises before testing. Comparing raw
        # names against repo-relative targets would reject every
        # correctly-formed diff.
        want = {F._norm(t) for t in target_paths}
        stray = sorted({F._norm(x) for x in paths} - want)
        if stray:
            # The brief names ONE file and hands it that file's error
            # heads. A diff that edits a different mutable file is inside
            # the fence and outside the question that was asked — and the
            # ledger row would record the target while the archive
            # recorded something else.
            return False, (f"edits {', '.join(stray)}, but the brief was "
                           f"about {', '.join(target_paths)}")
    if arch is not None:
        known = arch.known_diff_hashes()
        h = A.normalized_diff_hash(text)
        if h in known:
            # ~30% of evolved lines are resurrections (EvoTrace). A
            # duplicate is not a candidate, it is a re-run of one.
            return False, f"duplicate of {known[h]}"
    return True, ""


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

#: How many candidate snapshots to keep. Each is a full copy of `src/`
#: — 12 MB measured — and the phase can mint four a day, so without a
#: bound this is ~48 MB/day forever on the box the RSS watchdog exists
#: for. Every other disposable tree in this project has a sweeper
#: (`isolation.sweep_fork_workspaces`, `docker.sweep_orphaned_containers`);
#: this one had none, and the consumer that would consume them (E2) does
#: not exist yet.
MAX_KEPT_SNAPSHOTS = 8


def sweep_work_dirs(home: str = None, keep: int = MAX_KEPT_SNAPSHOTS) -> int:
    """Delete all but the `keep` newest candidate snapshots. Returns the
    number removed. Never raises — this runs before a nightly job.

    Also collects ORPHANS: a `wait_for` timeout in the idle phase cancels
    the coroutine but cannot cancel the `to_thread` already inside
    `materialize`, so a completed 12 MB snapshot can exist with no
    archive node, no ledger row and nothing referencing it.
    """
    base = _state_dir(home)
    if base is None:
        return 0
    root = base / WORK_DIRNAME
    if not root.is_dir():
        return 0
    removed = 0
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
    except Exception:              # noqa: BLE001
        return 0
    for stale in dirs[max(0, int(keep)):]:
        # ⚠ COUNT DELETIONS, NOT ATTEMPTS. `rmtree(ignore_errors=True)`
        # never raises, so incrementing beside it reported reclamation
        # for a symlinked or read-only directory that survived — and the
        # ledger would then show `swept_snapshots` climbing forever while
        # the disk never shrank.
        shutil.rmtree(stale, ignore_errors=True)
        if not stale.exists():
            removed += 1
    return removed


def work_dir(node_id: str, home: str = None) -> Optional[Path]:
    base = _state_dir(home)
    return None if base is None else base / WORK_DIRNAME / str(node_id)


def materialize(node_id: str, diff: str, *, home: str = None,
                repo_root: Path = None) -> Tuple[bool, str]:
    """Snapshot `src/` and apply the diff to the COPY.

    No git assumption: the repo is not a checkout on this box. `patch` is
    dry-run first, so a diff that does not apply costs a snapshot and not
    a half-patched tree. A tree that patched only partially is worse than
    no candidate at all — it is a candidate nobody can reproduce.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    dest = work_dir(node_id, home)
    if dest is None:
        return False, "no GHOST_HOME"
    patch_bin = shutil.which("patch")
    if not patch_bin:
        # Checked FIRST: a precondition of the whole operation should
        # refuse before a 12 MB snapshot, not after. Fail closed —
        # applying a diff by hand here would be a second implementation
        # of `patch` inside a security fence.
        return False, "no `patch` binary — refusing to hand-apply a diff"
    src = root / "src"
    if not src.is_dir():
        return False, f"no src/ under {root}"
    # ⚠ ASK THE FILESYSTEM, not just the path strings. A case-fold is a
    # GUESS about the volume's equivalence relation, and this one is
    # unicode-folding: `regiſtry.py` opens `registry.py`. `os.path.
    # samefile` compares device+inode, which is the only authority.
    for _rel in {F._norm(x) for x in A.diff_touched_paths(diff)}:
        if not _rel:
            continue
        _why = F.resolves_to_immutable(_rel, root)
        if _why:
            return False, _why
    try:
        dest.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            subprocess.run([rsync, "-a", "--delete",
                            "--exclude", "__pycache__",
                            f"{src}/", f"{dest / 'src'}/"],
                           check=True, capture_output=True, timeout=300)
        else:
            if (dest / "src").exists():
                shutil.rmtree(dest / "src")
            shutil.copytree(src, dest / "src",
                            ignore=shutil.ignore_patterns("__pycache__"))
    except Exception as exc:       # noqa: BLE001
        _discard(dest)
        return False, f"snapshot failed ({exc})"
    patch_file = dest / "candidate.diff"
    try:
        # ⚠ The trailing newline is load-bearing. Every model reply goes
        # through `_strip_fence`, whose `.strip()` removes it, and
        # `patch` then fails the hunk with "1 out of 1 hunks failed" —
        # so EVERY proposal would have died at materialisation with a
        # message that reads like a bad diff rather than a lost byte.
        # CRLF likewise: a model that emits Windows line endings writes a
        # diff that validates and then fails every hunk.
        patch_file.write_text(
            str(diff).replace("\r\n", "\n").rstrip("\n") + "\n")
        for dry in (True, False):
            # -F0: NO FUZZ. `patch` defaults to fuzz 2, so a hunk whose
            # context does not match can still land — a few lines away
            # from where the diff said, silently. A self-modification
            # fence cannot accept "applied approximately": the artefact
            # the operator reviews has to be the artefact that ran.
            # --no-backup-if-mismatch keeps stray .orig files out of the
            # snapshot the evaluator imports from.
            cmd = [patch_bin, "-p1", "-F0", "--batch", "--forward",
                   "--no-backup-if-mismatch", "-i", str(patch_file)]
            if dry:
                cmd.insert(1, "--dry-run")
            res = subprocess.run(cmd, cwd=str(dest), capture_output=True,
                                 text=True, timeout=120)
            if res.returncode != 0:
                _discard(dest)
                return False, (f"patch {'dry-run ' if dry else ''}failed "
                               f"(exit {res.returncode}): "
                               f"{(res.stdout + res.stderr).strip()[:300]}")

    except Exception as exc:       # noqa: BLE001
        _discard(dest)
        return False, f"patch failed ({exc})"
    # ⚠ CONTAINMENT AFTER THE FACT. The fence is a test on path STRINGS;
    # an escape can live in the filesystem instead. `MAX_FILES = 2` is
    # exactly enough for the classic two-section traversal: create
    # `src/ghost_agent/tools/link` as a symlink to `../../../..`, then
    # write through it into the canonical tree. This box's `patch`
    # (Apple 2.0-12u11) ignores `new file mode 120000` and writes a
    # regular file, so the vector is dead HERE — but `shutil.which` takes
    # whatever `PATH` offers, and GNU patch >= 2.7 honours git extended
    # headers. Verify where the bytes actually landed.
    escaped = containment_violation(dest, diff)
    if escaped:
        _discard(dest)
        return False, escaped
    moved = applied_where_it_said(dest, diff)
    if moved:
        _discard(dest)
        return False, moved
    return True, str(dest)


_HUNK_HEAD_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

_HUNK_FULL_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


#: A re-anchor never needs more than this, and the path it reads comes
#: from a line the MODEL wrote.
MAX_ANCHOR_SOURCE_BYTES = 4_000_000


def _read_anchor_source(root: Path, target: str):
    """The candidate file to re-anchor against, or None to leave the hunk.

    ⚠ THIS PATH IS MODEL-CONTROLLED AND UNVETTED AT THIS POINT.
    `repair_hunk_starts` runs BEFORE `validate_diff`, so the `+++` line
    has not yet been checked against the fence. Reading it naively:

    * `+++ b/../outside_secret.txt` read a file OUTSIDE the repo and
      re-anchored against it — `validate_diff` only objected afterwards;
    * `root / "/etc/hosts"` discards `root` entirely, because that is
      what `pathlib` does with an absolute right-hand side;
    * `+++ /dev/zero` reached **3.9 GB RSS in 2 seconds and never
      returned**, and a FIFO blocked indefinitely — and this runs
      SYNCHRONOUSLY on the event loop, unlike `materialize`, which is
      deliberately wrapped in `asyncio.to_thread` for exactly this
      hazard. One model-authored line wedges the process, and
      `/dev/null` is one character from `/dev/zero`.

    So: repo-relative only, no escapes, a regular file, and bounded.
    None of this replaces `validate_diff` — it makes the read that
    happens before it survivable.
    """
    rel = str(target or "")
    if not rel or Path(rel).is_absolute():
        return None
    try:
        base = Path(root).resolve()
        cand = (base / rel).resolve()
        if not str(cand).startswith(str(base) + os.sep):
            return None                       # escapes the repo
        st = cand.lstat()
        if not stat.S_ISREG(st.st_mode):
            return None                       # device, FIFO, socket (a symlink was
            # already FOLLOWED by resolve() above, so containment is what refuses it)
        if st.st_size > MAX_ANCHOR_SOURCE_BYTES:
            return None
        return cand.read_text(encoding="utf-8",
                              errors="replace").splitlines()
    except (OSError, ValueError, RuntimeError):
        return None


def repair_hunk_starts(diff: str, repo_root: Path = None) -> Tuple[str, int]:
    """Re-anchor each hunk's START LINE by finding its context in the file.

    ⚠ THE EDIT IS THE MODEL'S, THE COORDINATES ARE THE MACHINE'S.
    Measured: a semantically correct one-line addition to a real
    function declared line 468 when the context actually sits elsewhere.
    `patch` relocated it and `applied_where_it_said` refused — correctly,
    because the artefact an operator reviews must be the artefact that
    ran. Re-anchoring makes the diff HONEST before it is applied rather
    than letting `patch` fuzz it silently.

    ⚠ ONLY ON AN UNAMBIGUOUS MATCH. If the hunk's old-side block occurs
    zero times, or more than once, the header is left exactly as written
    and `patch` refuses it. Guessing between two candidate locations is
    how a repair becomes a relocation nobody authorised — and the
    containment checks still run afterwards regardless.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    lines = str(diff or "").splitlines()
    target, out, moved, i = None, [], 0, 0
    # ⚠ KEYED BY PATH, NOT RESET PER SECTION. A diff may carry two
    # `--- `/`+++ ` sections naming the SAME file — `check_diff_shape`
    # counts unique paths, so `files=1` and the shape is allowed — and
    # `patch` applies the second to the ALREADY-PATCHED file. Resetting
    # the offset on every `+++` made the second section's new-side
    # starts one short, `patch` still produced exactly correct bytes,
    # and `applied_where_it_said` then objected — discarding a perfectly
    # good candidate and recording it, by normalised hash, as one the
    # model may never propose again.
    offsets: Dict[str, int] = {}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("+++ "):
            rel = ln[4:].strip().split("\t")[0]
            target = rel[2:] if rel.startswith(("a/", "b/")) else rel
            offsets.setdefault(target, 0)
            out.append(ln); i += 1; continue
        m = _HUNK_FULL_RE.match(ln)
        if not m or not target:
            out.append(ln); i += 1; continue
        a_start, a_len, b_start, b_len, trailer = m.groups()
        body, j = [], i + 1
        while not _body_ends_here(lines, j):
            body.append(lines[j]); j += 1
        old_side = [x[1:] if x else "" for x in body
                    if x[:1] in (" ", "-") or x == ""]
        new_len = sum(1 for x in body if x[:1] in (" ", "+") or x == "")
        src = _read_anchor_source(root, target)
        if src is None:
            out.append(ln); out.extend(body); i = j; continue
        hits = [k for k in range(len(src) - len(old_side) + 1)
                if src[k:k + len(old_side)] == old_side] if old_side else []
        if len(hits) == 1:
            new_a = hits[0] + 1
            # ⚠ THE NEW-SIDE START IS NOT THE OLD-SIDE START. It is offset
            # by the net lines every EARLIER hunk in this file added.
            # Setting them equal silently rewrote `@@ -6,2 +7,3 @@` to
            # `@@ -6,2 +6,3 @@` on a diff that was already correct — and
            # `applied_where_it_said` then rejected the candidate with a
            # message blaming `patch`. Every multi-hunk proposal was a
            # self-inflicted permanent failure, reported as someone
            # else's fault.
            new_b = new_a + offsets.get(target, 0)
            head = (f"@@ -{new_a},{len(old_side)} "
                    f"+{new_b},{new_len} @@{trailer}")
            # ⚠ COUNT MOVES, NOT NORMALISATIONS. `@@ -1 +1,2 @@` is the
            # valid short spelling of `@@ -1,1 +1,2 @@`, and comparing
            # the STRINGS counted that rewrite as a relocation — so
            # `hunks_reanchored` over-reported and an operator reading
            # the ledger saw the model missing anchors it had hit. The
            # question is whether the START changed.
            if (new_a != int(a_start)) or (new_b != int(b_start)):
                moved += 1
            out.append(head)
        else:
            out.append(ln)
        # …and the offset advances on EVERY hunk, repaired or not.
        offsets[target] = offsets.get(target, 0) + new_len - len(old_side)
        out.extend(body); i = j
    text = "\n".join(out)
    if diff.endswith("\n"):
        text += "\n"
    return text, moved


def _is_file_header(lines: List[str], j: int) -> bool:
    """Is `lines[j]` a diff FILE HEADER rather than diff CONTENT?

    ⚠ `--- ` IS AMBIGUOUS. A deleted source line whose text is `-- x`
    becomes `--- x` in the diff — byte-identical in prefix to the `---
    a/path` header. Breaking the hunk-body scan on the prefix alone
    truncated the body, which (a) mis-counts the header, and (b) since
    the re-anchor's running `offset` is computed from body lengths,
    poisons the NEW-SIDE START OF EVERY LATER HUNK IN THE FILE — one
    ambiguous line silently relocating everything after it.

    A real header always comes in a pair: `--- a/…` immediately followed
    by `+++ b/…`. Content never does, because the `+++ ` line would have
    to be an added line reading `++ …` on exactly the next line.
    """
    ln = lines[j] if 0 <= j < len(lines) else ""
    if ln.startswith("--- "):
        return j + 1 < len(lines) and lines[j + 1].startswith("+++ ")
    if ln.startswith("+++ "):
        return j > 0 and lines[j - 1].startswith("--- ")
    return False


def _body_ends_here(lines: List[str], j: int) -> bool:
    """Should the hunk-body scan stop at `lines[j]`?

    A bare "" is accepted as a body line — models strip the trailing
    space off blank CONTEXT lines constantly — but a "" that SEPARATES
    two file sections is not part of anything, and swallowing it
    inflated both counts and made `patch` reject the result. Look ahead:
    a blank followed by a header or a new hunk is a separator.
    """
    if j >= len(lines):
        return True
    ln = lines[j]
    if _is_file_header(lines, j) or _HUNK_FULL_RE.match(ln):
        return True
    if ln == "":
        # ⚠ A BLANK BEFORE `@@` IS THE HUNK'S LAST CONTEXT LINE, NOT A
        # SEPARATOR. Within one file section hunks follow each other
        # directly, so nothing separates them; only a `--- `/`+++ ` pair
        # (a new file) or the end of the diff does. Treating `@@` as a
        # separator dropped that context line from the body — shrinking
        # both counts while still emitting the line — and `patch` then
        # saw a stray blank plus `@@` and started a headerless section.
        # MEASURED: it broke 26 of 299 diffs over real repo files that
        # applied cleanly BEFORE the repair, on the one model behaviour
        # this module's own comments say happens constantly (a blank
        # context line arriving as "" rather than " "). A repair that
        # breaks working diffs is worse than no repair, and the cost is
        # permanent: the failure is archived by normalised hash, so the
        # novelty filter blocks the model from ever re-proposing the
        # same correct edit.
        k = j + 1
        while k < len(lines) and lines[k] == "":
            k += 1
        return k >= len(lines) or _is_file_header(lines, k)
    return ln[:1] not in (" ", "+", "-", "\\")


def repair_hunk_counts(diff: str) -> Tuple[str, int]:
    """Recompute each hunk header's LINE COUNTS from its own body.

    ⚠ ARITHMETIC ONLY, AND IT IS THE MACHINE'S JOB. Measured: the model
    wrote a semantically correct three-line addition into a real
    function and headed it `@@ -466,6 +466,7 @@` — the body has 9 new
    lines, not 7 — and `patch` rejected the whole diff as malformed. The
    edit was right and the counting was wrong.

    This rewrites ONLY the two counts. Context and +/- lines are
    untouched, the start lines stay as the model claimed them, and a
    hunk whose header will not parse is left exactly as it is so `patch`
    still refuses it. Containment is unaffected: `patch` must still
    match the context, and `containment_violation` /
    `applied_where_it_said` still run afterwards.

    Returns (diff, hunks_repaired) — the count is reported, never
    silent, because a normalisation nobody sees is a normalisation that
    hides how sloppy the model actually was.
    """
    lines = str(diff or "").splitlines()
    out, repaired, i = [], 0, 0
    while i < len(lines):
        m = _HUNK_FULL_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        a_start, _a_len, b_start, _b_len, trailer = m.groups()
        body, j = [], i + 1
        while not _body_ends_here(lines, j):
            body.append(lines[j])
            j += 1
        old_n = sum(1 for x in body if x[:1] in (" ", "-") or x == "")
        new_n = sum(1 for x in body if x[:1] in (" ", "+") or x == "")
        head = f"@@ -{a_start},{old_n} +{b_start},{new_n} @@{trailer}"
        if head != lines[i]:
            repaired += 1
        out.append(head)
        out.extend(body)
        i = j
    text = "\n".join(out)
    if diff.endswith("\n"):
        text += "\n"
    return text, repaired


def applied_where_it_said(dest: Path, diff: str) -> str:
    """Why the patched file does not match the diff's own line numbers.

    ⚠ `-F0` disables FUZZ, not OFFSET. An exactly-matching hunk still
    relocates, and the line numbers an operator reviews then differ from
    where the change landed — in a self-modification fence, "applied
    approximately" is not acceptable.

    The obvious check — grep `patch`'s stdout for "offset" — is NOT a
    mechanism: measured on this box, Apple `patch 2.0-12u11` applies a
    hunk 25 lines away from its declared position and says nothing at
    all, exit 0. A check that never fires is worse than no check, so this
    reconstructs each hunk's post-image and compares it against the file
    at the line the hunk DECLARED.
    """
    files: dict = {}
    current = None
    deleting = None          # repo-relative path this section REMOVES
    last_old = None          # the `--- a/<path>` most recently seen
    hunk_start = None
    post: list = []

    def _flush() -> str:
        # ⚠ A DELETED FILE STILL HAS TO BE DELETED. `+++ /dev/null` set
        # `current = None`, so every hunk of a whole-file removal was
        # skipped and this returned "no objection". Measured: Apple
        # `patch` leaves a **0-byte file** rather than removing it, and
        # both containment checks stayed silent about a tree that did
        # not match the diff's own claim.
        if current is None and deleting:
            leftover = dest / deleting
            if leftover.exists():
                return (f"{deleting}: the diff deletes this file but it is "
                        f"still present after patching "
                        f"({leftover.stat().st_size} bytes) — the artefact "
                        f"an operator reviews must be the artefact that ran")
            return ""
        if current is None or hunk_start is None:
            return ""
        # ⚠ AN EMPTY POST-IMAGE CANNOT BE CHECKED, AND MUST NOT PASS.
        # A pure-deletion hunk with no context line leaves `post == []`,
        # so the comparison below was `[] == []` — true wherever the
        # hunk landed. Measured: a two-line deletion declared at line 1
        # was applied 400 lines away, `patch` exited 0, and this
        # returned "" while `materialize` archived the candidate and
        # offered it to an operator. The re-anchor cannot cover it
        # either: two matching blocks are ambiguous by design and left
        # alone. Refusing is the only honest answer — a model can always
        # emit context, and `difflib` always does.
        if not post:
            return (f"{current}: a hunk at line {hunk_start} has no context "
                    f"or added lines, so there is nothing to compare and "
                    f"nothing can show it landed where it said")
        try:
            body = (dest / current).read_text(errors="replace").splitlines()
        except Exception:          # noqa: BLE001
            return f"{current} is unreadable after patching"
        got = body[hunk_start - 1: hunk_start - 1 + len(post)]
        if got != post:
            return (f"{current}: hunk declared line {hunk_start} but the "
                    f"file does not match there — `patch` relocated it, "
                    f"and the artefact an operator reviews must be the "
                    f"artefact that ran")
        return ""

    all_lines = str(diff or "").replace("\r\n", "\n").splitlines()
    for idx, line in enumerate(all_lines):
        m = _HUNK_HEAD_RE.match(line)
        if m:
            bad = _flush()
            if bad:
                return bad
            hunk_start, post = int(m.group(1)), []
            continue
        # ⚠ AN ADDED LINE CAN SPELL A FILE HEADER. Source text `++ x`
        # becomes `+++ x` in the diff, and treating that as a header set
        # `current = None` — after which every remaining hunk was
        # skipped and this check returned "" for a diff it had stopped
        # reading. A header is only ever the second half of a
        # `--- `/`+++ ` pair.

        if line.startswith("+++ ") and not (
                idx and all_lines[idx - 1].startswith("--- ")):
            post.append(line[1:])
            continue
        if line.startswith("+++ "):
            bad = _flush()
            if bad:
                return bad
            name = line[4:].split("\t")[0].strip()
            current = F._norm(name) if name != "/dev/null" else None
            deleting = last_old if current is None else None
            files[current] = True
            hunk_start, post = None, []
            continue
        if line.startswith("--- "):
            old_name = line[4:].split("\t")[0].strip()
            last_old = (F._norm(old_name) if old_name != "/dev/null"
                        else None)
            continue
        if line.startswith("---") or line.startswith("diff "):
            continue
        if hunk_start is None:
            continue
        if line.startswith("+"):
            post.append(line[1:])
        elif line.startswith(" ") or line == "":
            post.append(line[1:] if line else "")
        # "-" lines are not in the post-image
    return _flush()


def containment_violation(dest: Path, diff: str) -> str:
    """Why the patched snapshot escaped its own directory, or "".

    ⚠ THE FENCE IS A TEST ON PATH STRINGS; an escape can live in the
    FILESYSTEM instead. `MAX_FILES = 2` is exactly enough for the classic
    two-section traversal: section 1 creates
    `src/ghost_agent/tools/link` as a symlink to `../../../..`, section 2
    writes through it into the canonical tree — and `is_mutable` approves
    both paths, because both are under an allowed prefix. This box's
    `patch` (Apple 2.0-12u11) ignores `new file mode 120000` and writes a
    regular file, so the vector is dead HERE — but `shutil.which("patch")`
    takes whatever `PATH` offers and GNU patch >= 2.7 honours git
    extended headers. Verify where the bytes actually landed.
    """
    try:
        root_real = dest.resolve()
    except Exception:              # noqa: BLE001
        return "the snapshot root could not be resolved"
    for rel in A.diff_touched_paths(diff):
        rel = F._norm(rel)
        if not rel:
            continue
        probe = dest
        for part in Path(rel).parts:
            probe = probe / part
            if probe.is_symlink():
                return (f"a symlink appeared at {probe.relative_to(dest)} — "
                        f"refusing a candidate that can write out of its "
                        f"own snapshot")
        try:
            if probe.exists() and not str(probe.resolve()).startswith(
                    str(root_real)):
                return f"{rel} resolved outside the snapshot"
        except Exception:          # noqa: BLE001
            return f"{rel} could not be resolved after patching"
    return ""


def _discard(dest: Path) -> None:
    """Remove a snapshot that will never be evaluated.

    A failed materialisation used to leave 12 MB behind on every attempt,
    forever, with nothing in the package referencing the work tree — and
    because the failing diff was never archived either, the same model
    re-proposed it the next night into a NEW directory."""
    try:
        shutil.rmtree(dest, ignore_errors=True)
    except Exception:              # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def write_mutation(rec: Dict[str, Any], home: str = None) -> bool:
    """One row per RUN, including the runs that produced nothing.

    Without this, "the mutator proposed nothing last night" and "the
    mutator never ran" are the same observation — the activation-counter
    lesson, applied to the loop that is supposed to be improving things.
    """
    base = _state_dir(home)
    if base is None:
        return False
    try:
        base.mkdir(parents=True, exist_ok=True)
        with (base / LEDGER_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:              # noqa: BLE001
        return False


def iter_mutations(home: str = None) -> List[Dict[str, Any]]:
    base = _state_dir(home)
    if base is None or not (base / LEDGER_FILE).exists():
        return []
    out = []
    for line in (base / LEDGER_FILE).read_text(
            errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:      # noqa: BLE001
                continue
    return out


def mutation_stats(home: str = None) -> Dict[str, Any]:
    rows = iter_mutations(home)
    by_outcome: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for r in rows:
        by_outcome[str(r.get("outcome"))] = \
            by_outcome.get(str(r.get("outcome")), 0) + 1
        if r.get("outcome") != OUT_PROPOSED and r.get("reason"):
            key = str(r["reason"])[:60]
            by_reason[key] = by_reason.get(key, 0) + 1
    flagged = sum(1 for r in rows if r.get("guard_flags"))
    return {"runs": len(rows), "by_outcome": by_outcome,
            "by_reason": by_reason,
            "proposed": by_outcome.get(OUT_PROPOSED, 0),
            # Proposals whose diff removed a refusal-shaped line. NOT a
            # rejection count — the operator decides. Surfaced because a
            # flag nobody reads is the same as no flag.
            "guard_flagged": flagged}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

async def run_mutation(context, *, rand: float = 0.5, home: str = None,
                       repo_root: Path = None, model: str = "",
                       write: bool = True) -> Dict[str, Any]:
    """One mutation attempt end to end. Always returns a record."""
    now = datetime.datetime.utcnow().isoformat() + "Z"
    rec: Dict[str, Any] = {
        "ts": now, "outcome": OUT_DISABLED, "reason": "", "parent": "",
        "target": "", "target_path": "", "attempts": 0, "node_id": "",
        "diff_hash": "", "files": 0, "lines": 0,
        "evidence": {}, "sources_present": [],
    }
    if not _enabled():
        # ⚠ SWEEP EVEN WHEN THE GATE IS SHUT. The sweeper used to sit
        # BELOW this return, i.e. gated behind the thing it cleans up, so
        # an enable→disable cycle stranded up to `MAX_KEPT_SNAPSHOTS + 1`
        # snapshots — ~110 MB — permanently.
        rec["reason"] = "GHOST_EVOLVE is off"
        _swept = await asyncio.to_thread(sweep_work_dirs, home)
        if _swept:
            rec["swept_snapshots"] = _swept
        if write:
            write_mutation(rec, home)
        return rec

    # Reclaim before spending: a snapshot is 12 MB and the consumer does
    # not exist yet.
    _swept = await asyncio.to_thread(sweep_work_dirs, home)
    if _swept:
        rec["swept_snapshots"] = _swept
    evidences = await asyncio.to_thread(gather_evidence, home)
    rec["sources_present"] = [e.source for e in evidences if e.present]
    rec["evidence"] = {
        e.source: ({"tools": len(e.by_tool),
                    "fails": sum(int(c.get("fails") or 0)
                                 for c in e.by_tool.values()),
                    "guard_refusals": sum(int(c.get("guard_refusals") or 0)
                                          for c in e.by_tool.values())}
                   if e.present else e.reason)
        for e in evidences}

    try:
        from ..tools.registry import get_available_tools
        tools = get_available_tools(context)
    except Exception as exc:       # noqa: BLE001
        rec["outcome"] = OUT_NO_EVIDENCE
        rec["reason"] = f"tool surface unreadable ({exc})"
        if write:
            write_mutation(rec, home)
        return rec

    ranked = rank_targets(evidences, tool_target_map(tools))
    if not ranked:
        rec["outcome"] = OUT_NO_EVIDENCE
        rec["reason"] = (f"no mutable target reached "
                         f"{MIN_EVIDENCE_ITEMS} recorded failures")
        if write:
            write_mutation(rec, home)
        return rec

    target = ranked[0]
    rec["target"] = target.tool
    rec["target_path"] = target.paths[0]
    arch = A.Archive(home)
    rec["parent"] = A.pick_parent(arch, rand)
    brief = build_brief(target, evidences, repo_root=repo_root)

    llm = getattr(context, "llm_client", None)
    diff, rejection = "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rec["attempts"] = attempt
        candidate = await propose_diff(llm, brief, model=model,
                                       rejection=rejection)
        # ⚠ FIX THE ARITHMETIC BEFORE JUDGING THE EDIT. Measured: a
        # semantically correct three-line addition to a real function
        # was headed `@@ -466,6 +466,7 @@` when its body held 9 new
        # lines, and `patch` rejected the whole diff as malformed. The
        # counting is the machine's job; rejecting a good edit over it
        # spends a generation on bookkeeping. Only the two counts move —
        # context and +/- lines are untouched, `patch` must still match,
        # and containment is still checked afterwards.
        if candidate:
            # Re-anchor FIRST (needs the old-side block as written), then
            # fix the counts on the re-anchored hunks.
            candidate, _moved = repair_hunk_starts(candidate, repo_root)
            if _moved:
                rec["hunks_reanchored"] = rec.get("hunks_reanchored", 0) + _moved
            candidate, _fixed = repair_hunk_counts(candidate)
            if _fixed:
                rec["hunks_repaired"] = rec.get("hunks_repaired", 0) + _fixed
        if not candidate:
            rec["outcome"] = OUT_NO_MODEL
            rec["reason"] = "the model returned nothing"
            continue
        ok, why = validate_diff(candidate, arch,
                                target_paths=target.paths)
        if ok:
            diff = candidate
            break
        rejection = why
        rec["outcome"] = (OUT_DUPLICATE if why.startswith("duplicate")
                          else OUT_REJECTED)
        rec["reason"] = why
    if not diff:
        if write:
            write_mutation(rec, home)
        return rec

    files, lines = A.diff_size(diff)
    node_id = A.new_id(diff, rec["parent"])
    # ⚠ OFF THE EVENT LOOP. `materialize` is an rsync of 12 MB plus two
    # `patch` invocations — three synchronous `subprocess.run` calls with
    # 300/120/120 s timeouts. Called directly from a coroutine they block
    # the whole ASGI process, and the `asyncio.wait_for` the idle phase
    # wraps this in cannot preempt a blocking call: it can only fire once
    # control comes back.
    ok, detail = await asyncio.to_thread(
        materialize, node_id, diff, home=home, repo_root=repo_root)
    rec.update(node_id=node_id, diff_hash=A.normalized_diff_hash(diff),
               files=files, lines=lines)
    if not ok:
        rec["outcome"] = OUT_MATERIALIZE_FAILED
        rec["reason"] = detail
        rec["guard_flags"] = guard_flags(diff)
        if write:
            # Archived as REJECTED so the novelty filter remembers it.
            # Without this the same diff is not a duplicate tomorrow: the
            # model re-proposes it, it re-fails, and a fresh snapshot is
            # created under a new id every night. `selection_weights`
            # excludes rejected nodes and their children, so this cannot
            # pollute parent sampling.
            arch.add(A.Node(id=node_id, parent=rec["parent"], diff=diff,
                            diff_hash=rec["diff_hash"],
                            target_files=A.diff_touched_paths(diff),
                            brief=brief, status=A.STATUS_REJECTED,
                            eval={"stage_materialize": {
                                "passed": False, "detail": detail[:400]},
                                  "guard_flags": rec.get("guard_flags") or []}))
            write_mutation(rec, home)
        return rec

    # The flags ride on the ledger row AND the archive node, so the
    # operator reviewing the proposal sees them without a join.
    rec["guard_flags"] = guard_flags(diff)
    node = A.Node(id=node_id, parent=rec["parent"], diff=diff,
                  diff_hash=rec["diff_hash"],
                  target_files=A.diff_touched_paths(diff), brief=brief,
                  eval={"guard_flags": rec["guard_flags"]})
    if write:
        arch.add(node)
    rec["outcome"] = OUT_PROPOSED
    rec["reason"] = ""
    rec["work_dir"] = detail
    if write:
        write_mutation(rec, home)
    return rec


__all__ = [
    "MIN_EVIDENCE_ITEMS", "MAX_ATTEMPTS", "MAX_ERRORS_IN_BRIEF",
    "MAX_KEPT_SNAPSHOTS", "sweep_work_dirs",
    "SOURCE_FORESIGHT", "SOURCE_POSTMORTEM", "SOURCE_CREDIT",
    "OUT_PROPOSED", "OUT_NO_EVIDENCE", "OUT_NO_MODEL", "OUT_REJECTED",
    "OUT_DUPLICATE", "OUT_MATERIALIZE_FAILED", "OUT_DISABLED",
    "Evidence", "Target", "implementation_paths", "tool_target_map",
    "foresight_evidence", "postmortem_evidence", "credit_evidence",
    "gather_evidence", "rank_targets", "build_brief", "propose_diff",
    "validate_diff", "guard_flags", "materialize", "work_dir",
    "write_mutation",
    "iter_mutations", "mutation_stats", "run_mutation",
]
