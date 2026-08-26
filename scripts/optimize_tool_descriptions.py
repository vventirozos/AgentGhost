#!/usr/bin/env python3
"""§4F Phase 2b — GEPA optimization of tool DESCRIPTIONS.

Metric = tool-choice fidelity on mined fixtures: for each recorded
tool-choice decision from a turn that PASSED (label 1.0), replay the
recorded request against the MAIN model with candidate descriptions
swapped into the tools array, and score 1.0 iff the model picks the same
tool production picked. Ground truth, split, and era rules live in the
miner (`scripts/mine_tool_fixtures.py` / `optim/tool_fixtures.py`).

Hygiene (§4F Phase 0): the optimizer sees only `tier == "public"`
fixtures; the ship-gate judges only `tier == "private"`; iterations clamp
to MAX_OPT_ITERATIONS. Components = the top-N tools by public-fixture
count. Ships per-tool artifacts (`tool_description.<name>.json`) consumed
by the registry read-site (`_apply_tuned_descriptions`, validator +
aggregate inflation guard); deploy = the §4DE epoch swap (live within ~a minute of promotion — no restart).

⚠ SUPPLY GATE — REAL POSITIVES, not fixtures and not all positives.
Negatives cannot score a tool-choice replay, and bench fixtures may TEACH
(public side) but may never GRADE, so they may never be the reason a run
starts either. Measured 2026-08-26 on a fresh mine of all 29 day-files:
575 fixtures / 409 positives / 121 REAL positives.

⚠⚠ AND THE PARAGRAPH THAT USED TO BE HERE WAS FALSE IN BOTH HALVES. It
said the miner "reports ready while this runner still refuses" and told the
operator to "read the miner's `Labels:` line, not its exit code". The miner
gained `--min-positives` (default 200) applying the SAME real-positive
predicate, so it refuses too — driven on the live corpus it prints
`⚠ REAL positives 121 < --min-positives 200 … the runner … will refuse to
start` and `Gates NOT met … parked at <pool>.notready`, exit 1. And the
remedy was backwards: `Labels:` prints 409, the bench-inflated count the
same paragraph condemns, while the EXIT CODE and the `⚠ REAL positives` /
`Private REAL positives:` lines are the ones that agree with this runner.
Read the exit code. `--force-supply` overrides (smoke only).

⚠ TWO PRE-FLIGHTS, ONE REQUIREMENT: the private tier must be able to
RESOLVE --min-delta (a 1/n step finer than the bar) and to REACH
`ab_eval.SHIP_ALPHA` (enough discordant replays to be significant at all).
Both are checked before the expensive part; the refusal states the combined
number and offers a margin that actually re-passes.

Usage — the live pool is written by the miner ONLY when its own gates pass;
otherwise the mine is parked at `<pool>.notready` and that is what you point
at. As of 2026-08-25 the live pool does not exist and the parked mine holds
192 rows / 66 positives, so this runner refuses at the supply gate:

    PYTHONPATH=src python scripts/optimize_tool_descriptions.py \
        --fixtures $GHOST_HOME/system/optim/tool_choice_fixtures.jsonl \
        --upstream-url http://127.0.0.1:8088 --max-iterations 6 \
        --run-dir <state dir>
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from ghost_agent.optim import ab_eval
from ghost_agent.optim import gate_contract  # noqa: E402
from ghost_agent.optim.run_gepa import MAX_OPT_ITERATIONS  # noqa: E402
from ghost_agent.tools import registry as registry_mod  # noqa: E402

DEFAULT_MIN_FIXTURES = 200


def _baseline_descriptions() -> Dict[str, str]:
    out = {}
    for t in registry_mod.TOOL_DEFINITIONS:
        f = t.get("function") or {}
        if f.get("name"):
            out[f["name"]] = f.get("description", "") or ""
    return out


def _aggregate_inflation(by_tool: Dict[str, str], baselines: Dict[str, str],
                         ghost_home: Path):
    """Inflation of the WHOLE tools block once `by_tool` is promoted.

    → (worst_case_total, already_on_disk).

    Summing only the components of THIS run reproduced the exact failure the
    gate exists to prevent, because incremental promotion — 6 components
    against 39 tools — is the normal case: measured, 4 previously-promoted,
    individually-valid artifacts already consuming 21,870 chars plus this
    run's +4,800 shipped with `aggregate_ok=True` while production applied 0
    descriptions. So every artifact on disk is counted, not just this run's.

    WORST CASE, deliberately, because the read-site's input is not a fixed
    list. `_apply_tuned_descriptions` sums over the `tools` it is handed,
    which is the per-request `_intent_filter`ed subset — so the same artifact
    set can apply on one request and be dropped on the next. Two consequences:

      * per-tool deltas are CLAMPED at 0. A shrunk description offsets the
        others only while both tools survive the filter; drop the shrinking
        one and inflation goes UP. Re-measured 2026-08-04 from the shipped
        fixture (`test_the_aggregate_gate_is_worst_case_over_request_subsets`):
        a set whose naive sum is 19,998 (pass, ceiling 20,000) reaches 20,320
        clamped and applies ZERO once `postgres_admin` is filtered out, which
        is the default config (no `--default-db`). An earlier revision of this
        docstring said 19,800 -> 20,122; those numbers came from no fixture in
        the tree and do not reproduce.
      * artifacts naming tools absent from `TOOL_DEFINITIONS` (runtime-only:
        vision, image-gen, acquired/composed skills) are counted at FULL
        length, since their baseline is not visible from here but the
        read-site will still charge them.

    A conservative gate over-rejects; a permissive one burns a whole
    optimization run and then silently disables every OTHER tuned description
    in production. Only one of those is recoverable by re-running.

    Text is `.strip()`ed to match `optim.loader.tuned_instruction`, which
    strips before production ever sees it — otherwise trailing whitespace
    counts here and nowhere else.
    """
    optim_dir = ghost_home / "system" / "optim"
    on_disk: Dict[str, str] = {}
    unknown: Dict[str, str] = {}
    try:
        for p in optim_dir.glob("tool_description.*.json"):
            name = p.name[len("tool_description."):-len(".json")]
            if not name:
                continue
            try:
                text = json.loads(p.read_text())["optimized_instruction"]
            except Exception:
                continue
            if not isinstance(text, str):
                continue
            text = text.strip()
            if name not in baselines:
                # A runtime-only tool. Unknown baseline -> charge it in full.
                unknown[name] = text
                continue
            if registry_mod._validate_tool_description(
                    name, baselines[name], text):
                on_disk[name] = text
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not scan promoted artifacts ({e}) — the "
              "aggregate gate is measuring THIS RUN ONLY and may pass a set "
              "that production drops", file=sys.stderr)

    if unknown:
        print(f"note: {len(unknown)} promoted artifact(s) name tools outside "
              f"TOOL_DEFINITIONS ({', '.join(sorted(unknown))}) — counted at "
              f"full length, since their baseline is not visible here",
              file=sys.stderr)

    def _infl(mapping):
        return sum(max(0, len(t) - len(baselines.get(n, "")))
                   for n, t in mapping.items())

    incumbent = _infl(on_disk) + sum(len(t) for t in unknown.values())
    effective = {**on_disk,          # this run supersedes its own artifacts
                 **{n: (t or "").strip() for n, t in by_tool.items()}}
    total = _infl(effective) + sum(len(t) for t in unknown.values())
    return total, incumbent


def _load_fixtures(path: Path) -> List[Dict[str, Any]]:
    # ⚠ THE DOCUMENTED PATH HAS NEVER EXISTED ON THIS MACHINE. The miner
    # writes the live pool only when ITS gates pass and otherwise parks the
    # mine at `<path>.notready` — which is the state the pool has been in
    # since 2026-08-04. Running the command in this module's own docstring
    # therefore raised a bare FileNotFoundError traceback out of
    # `path.read_text()`, with no hint that the mine exists one suffix away.
    if not path.exists():
        _parked = path.with_suffix(path.suffix + ".notready")
        _hint = (f"\n  A parked mine DOES exist at {_parked} — the miner "
                 f"writes there when its own gates fail. Point --fixtures "
                 f"at it to run against that mine (it is the same schema), "
                 f"or re-run scripts/mine_tool_fixtures.py."
                 if _parked.exists() else
                 f"\n  Run scripts/mine_tool_fixtures.py first; it writes "
                 f"this file when its gates pass and {_parked} when they "
                 f"do not.")
        # ⚠ SystemExit(<string>) exits 1 — "the gate rejected the
        # candidate" — and a missing pool measured nothing. Since the
        # pool has been parked at `.notready` since 2026-08-04, this was
        # the DEFAULT invocation's exit code (final verification pass,
        # finding 3). Message to stderr, code 2.
        print(f"no fixture pool at {path}.{_hint}", file=sys.stderr)
        raise SystemExit(2)
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _first_choice(fx: Dict[str, Any]) -> Optional[str]:
    # ⚠ A NON-DICT ELEMENT IS A TORN LINE, NOT A CRASH. `(c or {}).get`
    # raises `AttributeError` on a string or a number, and this is called
    # from the component ranking (`Counter(_first_choice(f) for f in pub)`)
    # — so one malformed row in a mined pool takes the whole run down with
    # a traceback instead of being skipped, which is the class
    # `_load_recorded_payload` was already hardened against.
    _c = fx.get("chosen_tools")
    if not isinstance(_c, list):
        return None
    for c in _c:
        if not isinstance(c, dict):
            continue
        n = c.get("name")
        if n:
            return n
    return None


def _load_recorded_payload(fx: Dict[str, Any],
                           recordings_dir: Path) -> Optional[Dict[str, Any]]:
    """Re-read the full recorded request payload via the fixture's light
    source pointer {file, ordinal} (fixtures deliberately don't embed the
    heavy messages/tools arrays)."""
    src = fx.get("source") or {}
    fname, ordinal = src.get("file"), src.get("ordinal")
    session = src.get("session_id")
    if not fname or ordinal is None:
        return None
    # ⚠ `recordings_dir / fname` IS `fname` WHEN fname IS ABSOLUTE, and
    # every fixture a real mine emits carries an ABSOLUTE `source.file`.
    # So `--recordings` was inert: pointing it at /tmp/DOES_NOT_EXIST
    # still reported 35/35 replayable. That is the one flag an operator
    # would reach for after moving the recordings directory — which is
    # the scenario the replayability pre-flight's own comment names.
    _p = Path(fname)
    _candidates = [recordings_dir / _p.name]
    if _p.is_absolute():
        _candidates.append(_p)
    for _path in _candidates:
        try:
            fh = open(_path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        break
    else:
        return None
    try:
        with fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                # ⚠ A JSON LINE NEED NOT BE AN OBJECT. `null` and `42`
                # parse fine and then `.get` raises — and the probe now
                # runs BEFORE the refusal branch, so a pool that would
                # have refused cleanly died with a traceback instead. A
                # torn append mid-line is the reachable case.
                if not isinstance(r, dict):
                    continue
                # Ordinals are PER-SESSION counters — matching on ordinal
                # alone grabs the first session's record and made 7/12
                # smoke fixtures "unreplayable" (2026-08-03).
                if (r.get("ordinal") == ordinal
                        and (session is None
                             or r.get("session_id") == session)):
                    p = r.get("payload")
                    return p if isinstance(p, dict) and p.get("tools") else None
    except OSError:
        return None
    return None


class ToolDescAdapter:
    """gepa.GEPAAdapter over recorded tool-choice replays.

    Sequential by design: every eval call rides the MAIN inference slot
    (descriptions target the main model), and prod serves live traffic on
    the same slot — politeness beats wall-clock here."""

    propose_new_texts = None  # gepa probes this optional hook directly

    def __init__(self, upstream_url: str, recordings_dir: Path,
                 baselines: Dict[str, str], timeout: float = 120.0):
        base = upstream_url.rstrip("/")
        self.url = (base if base.endswith("/v1") else base + "/v1") \
            + "/chat/completions"
        self.recordings_dir = recordings_dir
        self.baselines = baselines
        self.timeout = timeout

    def _call(self, payload: Dict[str, Any]) -> Optional[str]:
        """POST the replay; return the FIRST chosen tool name, or None when
        the model called no tool, or `TRANSPORT_FAILED` when it never
        answered.

        ⚠ THE SECOND ELEMENT IS THE POINT. This used to swallow every
        exception into a bare `None`, making "the upstream was down" and
        "the model called no tool" the same observation — scored 0.0 with
        `err=""`, invisible in the unreplayable count and in the artifact.
        The two arms run hours apart on the same shared slot (the
        candidate pass follows the whole GEPA run), so a restart during
        one of them manufactures discordant pairs in one direction.
        Measured before this fix: a 6-replay outage confined to the
        incumbent arm gave 0 incumbent wins, 6 candidate wins, p=0.0156,
        ships=True — on descriptions that were effectively identical.
        """
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except Exception:
            return TRANSPORT_FAILED
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        for tc in (msg.get("tool_calls") or []):
            n = ((tc.get("function") or {}).get("name"))
            if n:
                return n
        return None

    def _swap_descriptions(self, tools: List[Dict[str, Any]],
                           candidate: Dict[str, str]) -> Tuple[List[Dict[str, Any]], bool]:
        """→ (tools, all_swapped). ``all_swapped`` is False when any candidate
        description was refused by the validator.

        The refusal used to be SILENT: the recorded (incumbent) description
        was used instead, so an over-cap candidate scored exactly like the
        incumbent and GEPA got no gradient against length — it could carry
        the oversized text to the frontier and only discover at the ship gate
        that the whole run was invalid. That is the run-craft failure the
        journal records for the verifier round (37 wasted iterations).
        """
        out = []
        all_swapped = True
        for t in tools:
            f = t.get("function") or {}
            name = f.get("name")
            if name in candidate:
                if registry_mod._validate_tool_description(
                        name, self.baselines.get(name, ""), candidate[name]):
                    t = {**t, "function": {**f,
                                           "description": candidate[name].strip()}}
                else:
                    all_swapped = False
            out.append(t)
        return out, all_swapped

    def evaluate(self, batch: List[Dict[str, Any]],
                 candidate: Dict[str, str], capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        scores, outputs, trajectories = [], [], []
        for fx in batch:
            truth = _first_choice(fx)
            payload = _load_recorded_payload(fx, self.recordings_dir)
            if not truth or payload is None:
                # Unusable fixture — neutral-zero with explicit feedback so
                # reflection never "learns" from replay plumbing gaps.
                scores.append(0.0)
                outputs.append(None)
                trajectories.append({"fx": fx, "picked": None, "truth": truth,
                                     "score": 0.0, "err": "unreplayable"})
                continue
            swapped_tools, all_swapped = self._swap_descriptions(
                payload["tools"], candidate)
            if not all_swapped:
                # Score ZERO with loud feedback rather than silently grading
                # the incumbent. gepa's reflector composes from the parent's
                # traces, so the constraint has to be visible THERE.
                print("[cap-guard] candidate rejected by the per-tool "
                      "validator (length/shape) — scored 0.0")
                scores.append(0.0)
                outputs.append(None)
                trajectories.append({"fx": fx, "picked": None, "truth": truth,
                                     "score": 0.0,
                                     "err": "candidate over per-tool cap"})
                continue
            replay = {
                "messages": payload["messages"],
                "tools": swapped_tools,
                "temperature": 0.0,
                "max_tokens": min(int(payload.get("max_tokens") or 4096), 4096),
                "stream": False,
            }
            picked = self._call(replay)
            _down = picked is TRANSPORT_FAILED
            if _down:
                picked = None
            s = 1.0 if picked == truth else 0.0
            scores.append(s)
            outputs.append(picked)
            trajectories.append({"fx": fx, "picked": picked, "truth": truth,
                                 "score": s,
                                 "err": "transport" if _down else ""})
        return EvaluationBatch(
            outputs=outputs, scores=scores,
            trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch,
                                components_to_update):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for comp in components_to_update:
            records = []
            for traj in (eval_batch.trajectories or []):
                fx = traj["fx"]
                truth, picked = traj["truth"], traj["picked"]
                _err = traj.get("err") or ""
                # ⚠ EVERY PLUMBING ERR, NOT JUST THE ONE THIS LINE WAS
                # WRITTEN FOR. §4DA round 1 added a SECOND plumbing marker
                # (`err="transport"`, for a replay that never reached the
                # model) and left this consumer reading the old two-valued
                # world, so a llama-server restart mid-run fell into the
                # CANDIDATE-REJECTION branch below and taught the reflector
                # — on every affected fixture at once, since `evaluate`
                # runs the public trainset each iteration — that its
                # description "fails the production validator. Propose a
                # SHORTER one." Nothing of the sort happened: the
                # description was fine and the socket was down. The
                # premise of the round-1 fix is that these restarts happen
                # mid-run, so it created the state and then poisoned the
                # optimizer with it. `_TRANSPORT_ERRS` is the one list.
                if _err in _TRANSPORT_ERRS:
                    continue        # a plumbing gap teaches nothing
                if _err:
                    # A CANDIDATE-level rejection (over the per-tool cap) is
                    # feedback the reflector needs, not noise to drop — and
                    # it is the one error class that makes the dataset empty
                    # if skipped, since it fires on every fixture at once.
                    records.append({
                        "Inputs": {
                            "user_request": (fx.get("user_request") or "")[:400],
                            "advertised_tools":
                                ", ".join(fx.get("advertised_tools") or [])[:400],
                        },
                        "Generated Outputs": "(candidate REJECTED before "
                                             "replay)",
                        "Feedback": (
                            f"REJECTED, scored 0.0: {_err}. The proposed "
                            "description was never tried because it fails the "
                            "production validator. Propose a SHORTER one — "
                            "length is the problem, not the content."),
                    })
                    continue
                if traj["score"] >= 1.0:
                    fb = f"CORRECT: picked {truth}, matching production."
                elif picked is None:
                    fb = (f"NO TOOL CALLED — production called {truth} here "
                          "(and the turn PASSED). The description must make "
                          "this tool an obvious pick for this request.")
                else:
                    fb = (f"WRONG TOOL: picked {picked}; production picked "
                          f"{truth} and the turn PASSED. If this request "
                          f"is ambiguous, the {comp.split('.', 1)[-1]} "
                          "description should disambiguate it.")
                records.append({
                    "Inputs": {
                        "user_request": (fx.get("user_request") or "")[:400],
                        "advertised_tools":
                            ", ".join(fx.get("advertised_tools") or [])[:400],
                    },
                    "Generated Outputs": str(picked),
                    "Feedback": fb,
                })
            out[comp] = records
        return out


def _dump_confusion(trajectories: List[Dict[str, Any]], path: Path) -> int:
    """Persist per-fixture replay results for ontology analysis.

    Deliberately light: the tool names, the error flag, and enough request
    context to eyeball a pair. The full payload stays in the recordings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for t in trajectories or []:
            fx = t.get("fx") or {}
            fh.write(json.dumps({
                "fixture_id": fx.get("fixture_id", ""),
                "request_id": fx.get("request_id", ""),
                "tier": fx.get("tier", ""),
                "user_request": (fx.get("user_request") or "")[:300],
                "advertised_tools": fx.get("advertised_tools") or [],
                "truth": t.get("truth"),
                "picked": t.get("picked"),
                "score": t.get("score"),
                "err": t.get("err", ""),
            }, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _make_reflection_lm(url: str):
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"

    def _call(prompt: str) -> str:
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": 8192, "stream": False,
        }).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return ((data.get("choices") or [{}])[0]
                .get("message", {}).get("content", "") or "")

    return _call


class ShipDecision:
    """Why this candidate ships, or does not.

    An object rather than five locals in `main()` so the rule can be
    driven directly with real trajectory lists. Every §4CY/§4CZ round that
    went wrong went wrong in a harness that stubbed out the thing under
    test; there is nothing to stub here.

    ⚠ A PLAIN CLASS, NOT `@dataclass`, AND THAT IS NOT STYLE. `dataclasses`
    resolves annotations through `sys.modules[cls.__module__].__dict__`,
    and this file is loaded by `importlib.util.module_from_spec` +
    `exec_module` WITHOUT being registered in `sys.modules` — which is what
    every existing test that touches it does. Under `@dataclass` that
    lookup returns None and the module fails to import with
    `AttributeError: 'NoneType' object has no attribute '__dict__'`,
    breaking 3 tests and 14 collection errors in
    `test_audit_response_2026_08_04.py`. My change had silently imposed a
    new requirement on every consumer of this script; fixing the consumers
    would have been backwards, and the next loader would hit it again.
    """

    def __init__(self) -> None:
        self.ships = False
        self.p_value: Optional[float] = None
        self.incumbent_wins = 0
        self.candidate_wins = 0
        self.cleared_margin = False
        self.significant = False
        self.overridden = False
        self.unpaired = False
        self.transport_excluded = 0
        self.usable = 0
        self.outage_excluded = 0
        self.gap_excluded = 0
        self.underpowered = False
        #: The margin over USABLE pairs only. `delta` as computed by main()
        #: is over every row including transport failures, which score 0.0.
        self.paired_delta = 0.0
        self.paired_incumbent = 0.0
        self.paired_candidate = 0.0

    @property
    def discordant(self) -> int:
        return self.incumbent_wins + self.candidate_wins


#: Returned by `_call` when the replay never reached the model, as
#: distinct from `None` meaning "the model called no tool".
#:
#: ⚠ A SENTINEL RATHER THAN A `(value, ok)` TUPLE, AND THAT IS NOT STYLE.
#: Widening the return type broke four existing tests that stub `_call` —
#: `ValueError: too many values to unpack`. A sentinel keeps the contract
#: one value wide, so every stub that returns a tool name or None still
#: works, and only code that asks the new question needs to know. The same
#: lesson as §4DA's `@dataclass`: a change that silently imposes a new
#: requirement on every consumer is the wrong shape, however clean it
#: looks in isolation.
TRANSPORT_FAILED = object()


#: A replay that produced NO VERDICT — excluded from the comparison,
#: because the pair says nothing about the descriptions either way.
#: `_call` swallows every exception into None, so a transport failure is
#: otherwise indistinguishable from a legitimate no-tool answer; anything
#: the adapter DOES mark lands here.
_TRANSPORT_ERRS = ("unreplayable", "transport")

#: ⚠ AND THE TWO ARE NOT THE SAME KIND OF THING, WHICH ROUND 2 MISSED.
#: `unreplayable` means the fixture has NO RECORDED PAYLOAD — a permanent,
#: deterministic property of the pool, identical in both arms and on every
#: re-run. `transport` means the model was unreachable for this replay — a
#: transient, one-armed event. Round 2 fixed the PRINTED line for exactly
#: this confusion ("a corpus gap, stable across both arms" vs "an outage,
#: the thing that invalidates the pairing") and then armed its NEW
#: `underpowered` gate on the merged set. Measured: 60 rows, 12 with no
#: recorded payload, an honest 6-0 sweep on the 48 that can be replayed —
#: paired_delta +0.125, p=0.0156, BOTH bars cleared — refused with
#: "re-run when the upstream is stable". The upstream was never involved,
#: so re-running is a fixed point that costs `iterations * len(pub)`
#: main-model calls each time. Only an OUTAGE arms the power guard; a
#: corpus gap is the PRE-FLIGHT's business, and it now measures it.
_OUTAGE_ERRS = ("transport",)


def _transport_failed(traj: Dict[str, Any]) -> bool:
    """Did this replay fail to reach a verdict, as opposed to reaching a
    wrong one? Such a pair is not evidence about the descriptions."""
    return str(traj.get("err") or "") in _TRANSPORT_ERRS


def _outage(traj: Dict[str, Any]) -> bool:
    """Did this replay fail because the MODEL was unreachable, as opposed
    to because the fixture has nothing to replay?"""
    return str(traj.get("err") or "") in _OUTAGE_ERRS


def _replay_passed(traj: Dict[str, Any]) -> bool:
    """A replay counts as a pass at full fidelity, matching the adapter's
    own `traj["score"] >= 1.0` test."""
    try:
        return float(traj.get("score") or 0.0) >= 1.0
    except (TypeError, ValueError):
        return False


def _ship_decision(inc_traj: List[Dict[str, Any]],
                   cand_traj: List[Dict[str, Any]], *,
                   min_delta: float, valid: bool,
                   aggregate_ok: bool, min_usable: int = 0,
                   allow_insignificant: bool = False) -> ShipDecision:
    """The ship rule: margin AND significance, §4CY ported.

    ⚠ THIS USED TO BE `delta > min_delta` WITH NO SIGNIFICANCE TEST — the
    same rule `optim/ab_eval.py` carried until §4CY measured what it does:
    under the null a bare margin promotes 25-40% of the time, because the
    smallest swing clearing the bar is one or two flipped replays. §4CY
    fixed the GEPA gate and left this sibling and `optimize_verifier.py`
    untouched, so it landed in one of three ship gates. This is the second.

    McNemar is the right statistic HERE and the wrong one in §4CZ, and the
    difference is the whole point: both arms replay the SAME fixture list
    in order, so fixture i is a matched pair. §4CZ's live arms are
    different requests — unpaired — and need Fisher. Reaching for whichever
    test is to hand produces a number for the wrong question.
    """
    # ⚠ THE MARGIN IS DERIVED HERE, NOT PASSED IN. It used to be a
    # `delta` parameter, and a caller could hand it a number the
    # trajectories do not support — which is exactly what main() was
    # doing (a delta over ALL rows, transport failures scored 0.0 among
    # them) and exactly what the round-1 tests were doing (arms built for
    # one delta, a different delta passed beside them, both green). A
    # parameter that can contradict the evidence beside it will.
    d = ShipDecision()
    if len(inc_traj) != len(cand_traj) or not inc_traj:
        # ⚠ REFUSE RATHER THAN PAIR BY POSITION ACROSS DIFFERENT LENGTHS,
        # which would compare fixture i against fixture j and still yield
        # a p.
        d.unpaired = True
        return d
    # ⚠ A ONE-ARM OUTAGE MANUFACTURES CANDIDATE WINS. `_call` returns
    # None for EVERY exception — connection refused, read timeout, bad
    # JSON — indistinguishable from "the model called no tool", and the
    # row is then scored 0.0 with `err=""`, so it does not appear in the
    # unreplayable count and leaves no trace in the artifact. Measured: 60
    # fixtures, effectively identical descriptions, a 6-replay outage
    # confined to the INCUMBENT pass gave inc_wins=0, cand_wins=6,
    # p=0.0156, SHIPS=True. The direction is the dangerous one: the
    # incumbent arm runs first and the candidate arm runs after the whole
    # GEPA run, hours later on the same shared slot, so a llama-server
    # restart in between produces exactly that run.
    #
    # A pair where either arm failed to TRANSPORT is not evidence about
    # the descriptions, so it is excluded from the comparison entirely
    # rather than counted as a win for whichever arm stayed up.
    def _usable(i, c):
        return not (_transport_failed(i) or _transport_failed(c))

    _excluded = sum(1 for i, c in zip(inc_traj, cand_traj)
                    if not _usable(i, c))
    d.transport_excluded = _excluded
    d.outage_excluded = sum(1 for i, c in zip(inc_traj, cand_traj)
                            if _outage(i) or _outage(c))
    # ⚠ BY PREDICATE, NOT BY SUBTRACTION — the shape this entry condemned
    # for the printed err counts, still present here one round later. A
    # pair is a CORPUS GAP only when neither arm hit an outage; a pair
    # that is a gap in one arm and an outage in the other is an OUTAGE
    # pair, because that is the half that makes it re-runnable.
    d.gap_excluded = sum(1 for i, c in zip(inc_traj, cand_traj)
                         if not _usable(i, c)
                         and not (_outage(i) or _outage(c)))

    d.incumbent_wins = sum(1 for i, c in zip(inc_traj, cand_traj)
                           if _usable(i, c)
                           and _replay_passed(i) and not _replay_passed(c))
    d.candidate_wins = sum(1 for i, c in zip(inc_traj, cand_traj)
                           if _usable(i, c)
                           and _replay_passed(c) and not _replay_passed(i))
    d.p_value = ab_eval.mcnemar_p(d.incumbent_wins, d.candidate_wins,
                                  alternative="candidate")
    d.significant = (d.p_value is not None
                     and d.p_value <= ab_eval.SHIP_ALPHA)
    # ⚠ THE MARGIN MUST BE DE-CONTAMINATED TOO, NOT JUST THE STATISTIC.
    # Round 1 excluded transport failures from the PAIRING and left
    # `delta` — computed by main() over every row, where a transport
    # failure scores 0.0 — feeding the margin unchanged. Since
    # --allow-insignificant-ship ships on the margin ALONE, the exact
    # outage that round 1 recorded as closed still promoted through the
    # override: measured, 60 rows / 54 concordant passes / a 6-replay
    # outage confined to the incumbent arm gave delta +0.100, p=None,
    # ships=True, and an artifact recording ZERO discordant replays as
    # its evidence. The fix and the bug agreed on the default path, which
    # is the only path the round-1 check drove.
    _pairs = [(i, c) for i, c in zip(inc_traj, cand_traj) if _usable(i, c)]
    d.usable = len(_pairs)
    if _pairs:
        d.paired_incumbent = sum(
            1.0 for i, _c in _pairs if _replay_passed(i)) / len(_pairs)
        d.paired_candidate = sum(
            1.0 for _i, c in _pairs if _replay_passed(c)) / len(_pairs)
        d.paired_delta = d.paired_candidate - d.paired_incumbent
    # ⚠ AND AN OUTAGE THAT GUTS THE TIER MUST NOT SHIP AT ALL. With the
    # margin de-contaminated a big enough outage no longer FAKES a win,
    # but it still leaves a tier too coarse to have earned one: the
    # pre-flight refused to start below `min_usable` rows, and an outage
    # walks the run below that number after the pre-flight has passed.
    # Same requirement, applied to the evidence that actually survived.
    # ⚠ ARMED ON THE SHORTFALL, WHATEVER CAUSED IT — and round 4 armed it
    # on the CAUSE, which was wrong in both directions at once.
    #
    # Round 4's reasoning was: a corpus gap cannot be "re-run when the
    # upstream is stable" away, and the pre-flight now counts replayable
    # rows, so a tier big enough to start is big enough to finish. If that
    # premise holds, `usable < min_usable` without an outage is
    # UNREACHABLE and the extra clause changes nothing — swept every gap
    # size through the real `main()` and found no gap at which it altered
    # the outcome. And where the premise FAILS, the clause removes the
    # guard: the incumbent arm runs before `gepa.optimize` and the
    # candidate arm `iterations * len(pub)` = 318 main-model calls later,
    # so the recordings can move in between — which this script's own
    # docstring names as a one-step hazard. Driven with the recordings
    # pruned during that window: 5 usable pairs of the 60 the pre-flight
    # demanded 50 of, `delta=+1.000`, **ships=True, PROMOTED**, no
    # warning. Round 2's rule blocked exactly that.
    #
    # A guard disarmed on a property that makes the disarming pointless if
    # true, and dangerous if false, is the round-4 pattern. What round 4
    # was actually right about is the MESSAGE: "re-run when the upstream
    # is stable" is a no-op remedy for a corpus gap. So the guard blocks
    # on the shortfall and the message names the cause.
    d.underpowered = bool(min_usable and d.usable < min_usable)
    d.cleared_margin = bool(valid and aggregate_ok
                            and not d.underpowered
                            and d.paired_delta > min_delta)
    d.ships = d.cleared_margin and d.significant
    # ⚠ `not d.significant` IS LOAD-BEARING. Without it, running with
    # --allow-insignificant-ship and a genuinely significant winner stamps
    # `significance_overridden: true` on an honest promotion — a false
    # audit record in the field that exists so the call can be audited.
    if d.cleared_margin and not d.significant and allow_insignificant:
        d.ships = True
        d.overridden = True
    return d


def _significance_floor() -> int:
    """Delegates to `ab_eval.significance_floor` — one derivation shared by
    both GEPA runners and the miner, so the instrument cannot drift from
    the gate it reports."""
    return ab_eval.significance_floor()


def main() -> int:
    # ⚠ THE WHOLE DOCSTRING. With `__doc__.splitlines()[0]` the ⚠ SUPPLY
    # GATE warning and the usage block — the two things an operator needs
    # before running this — were unreachable from `--help`.
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", required=True)
    from ghost_agent.core.llm import DEFAULT_UPSTREAM_URL
    ap.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL,
                    help="MAIN model endpoint (descriptions target it); "
                         "default has ONE home: core.llm (§4DF CRIT-1)")
    ap.add_argument("--recordings", default="",
                    help="day-file dir (default $GHOST_HOME/system/llm_recordings)")
    ap.add_argument("--components", type=int, default=6,
                    help="top-N tools by public-fixture count to optimize")
    ap.add_argument("--max-iterations", type=int, default=6,
                    help=f"budget in full public evals (clamp {MAX_OPT_ITERATIONS})")
    ap.add_argument("--timeout", type=float, default=360.0,
                    help="per-replay deadline in seconds (default 360, "
                         "matching run_gepa.py and "
                         "recheck_gepa_incumbent.py). A replay that "
                         "exceeds it is a TRANSPORT failure: excluded "
                         "from both arms, never scored as a wrong pick.")
    ap.add_argument("--min-delta", type=float, default=0.02,
                    help="ship bar on private tool-choice fidelity "
                         "(default 0.02). Must be >=1e-6 and <1, and the "
                         "private tier must be able to resolve it — the "
                         "pre-flight refuses otherwise.")
    ap.add_argument(
        "--allow-seed-loss", action="store_true",
        help="Promote a candidate that beats the LIVE artifact while "
             "losing to the HAND-WRITTEN description. The default "
             "refuses: this gate seeds from the live artifact when one "
             "exists, so run N's arms are artifact-N-1 vs artifact-N and "
             "the hand-written text is in neither. Nothing else stops a "
             "chain of individually-winning promotions from drifting "
             "below where it started — `recheck_gepa_incumbent.py` exits "
             "3 for every tool_description signature, so there is no "
             "offline re-score. RECORDED in the artifact.")
    ap.add_argument(
        "--allow-insignificant-ship", action="store_true",
        help="Promote a candidate that clears --min-delta but whose "
             "discordant replays do not reach McNemar "
             "p<=ab_eval.SHIP_ALPHA. The default refuses: a bare margin "
             "promotes 25-40%% of the time under the null, because the "
             "smallest swing that clears it is one or two flipped "
             "replays. Use this when the margin is large and the fixture "
             "tier is simply too small to reach significance; it is "
             "RECORDED in the artifact so the call can be audited.")
    ap.add_argument("--min-fixtures", type=int, default=DEFAULT_MIN_FIXTURES,
                    help="minimum POSITIVE fixtures (the miner's flag of the "
                         "same name counts ALL fixtures — see the module "
                         "docstring)")
    ap.add_argument("--min-promotion-age-days", type=float, default=7.0,
                    help="refuse to run when the live artifact for any "
                         "component was promoted fewer than this many "
                         "days ago. Each run is a fresh draw at the "
                         "gate, so re-promoting before the last one can "
                         "be judged turns one decision into many — and "
                         "every re-promotion resets the live-check era. "
                         "0 disables.")
    ap.add_argument("--force-supply", action="store_true",
                    help="run below the supply gate. SMOKE RUNS ONLY — "
                         "requires --smoke, and is refused without it.")
    ap.add_argument("--smoke", action="store_true",
                    help="evaluate the incumbent on private fixtures and "
                         "exit — de-risks the replay path without an "
                         "optimization run")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--confusion-out", default="",
                    help="dump the INCUMBENT private-tier replay rows as JSONL "
                         "({truth, picked, err, ...}) for "
                         "scripts/tool_ontology_report.py. The 0.772 ceiling "
                         "check printed its misses and threw them away; the "
                         "structure of those misses is the evidence for "
                         "whether the toolbox needs re-carving or just "
                         "re-wording.")
    args = ap.parse_args()
    # ⚠ BEFORE ANY I/O (§4DF). The caller's proof the SCRIPT ran — the
    # same banner discipline as the judge and the miner, because this
    # gate's exit 2 is the same triply-overloaded code. ONE home.
    print(gate_contract.GATE_RUN_BANNER_OTD, flush=True)

    # ⚠ THE CONSTRAINT WAS PROSE. Both the module docstring and this
    # flag's help say "smoke runs only", nothing enforced it, and no test
    # pinned it — driven, `--force-supply --min-delta 0.029` on 121 real
    # positives PROMOTED 6 artifacts and exited 0. The one flag that
    # bypasses the gate this whole entry hardened carried its own
    # constraint only in a sentence.
    # ⚠ THE RE-DRAW GUARD THE SIBLING GATE HAS AND THIS ONE DID NOT.
    # `run_gepa.py` refuses when the live artifact is younger than
    # `--min-promotion-age-days`, because "each run is a fresh draw at
    # the gate, so re-promoting before the last one can be judged turns
    # one decision into many". This gate WRITES `promoted_utc` and never
    # read it — the computed-and-thrown-away shape, in the field §4DA
    # itself added.
    #
    # It bites harder here. A significance bar at alpha=0.05 promotes
    # under the null 5% PER RUN, and unlimited re-draws restore exactly
    # the failure §4CY's rule closed. Worse, `recheck_gepa_incumbent`
    # exits 3 for every `tool_description.*` signature — it cannot
    # re-score them — so `gepa_live_check` is the ONLY post-promotion
    # judge, and it needs MIN_PER_ARM per arm scoped to one sha. Every
    # re-promotion changes the text, changes the sha, and resets the era,
    # discarding every accrued turn as stale. At ~3.5 turns/day an
    # unguarded loop keeps the live check permanently INSUFFICIENT.
    def _too_young(sig_path):
        if args.min_promotion_age_days <= 0 or not sig_path.exists():
            return None
        try:
            _st = ((json.loads(sig_path.read_text()).get("gate") or {})
                   .get("promoted_utc") or "")
            if not _st:
                return None
            _age = (time.time()
                    - calendar.timegm(time.strptime(
                        _st, "%Y-%m-%dT%H:%M:%SZ"))) / 86400.0
        except Exception:  # noqa: BLE001 — an unreadable stamp must not
            return None    # block a run; it is "age unknown".
        # A stamp in the FUTURE is a clock, not a recent promotion —
        # `run_gepa`'s comment records an unbounded outage from exactly
        # that. Treated as age unknown.
        if _age < 0 or _age >= args.min_promotion_age_days:
            return None
        return _age

    if args.force_supply and not args.smoke:
        print("REFUSING TO RUN: --force-supply bypasses the supply gate "
              "and is for SMOKE RUNS ONLY, so it requires --smoke. A run "
              "that can PROMOTE must clear the gate: the gate tier is the "
              "only evidence a promotion ships on.", file=sys.stderr)
        return 2

    ghost_home = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    recordings_dir = Path(args.recordings) if args.recordings else \
        ghost_home / "system" / "llm_recordings"

    fixtures = _load_fixtures(Path(args.fixtures))
    positives = [f for f in fixtures
                 if f.get("label") == 1.0 and _first_choice(f)]
    # ⚠ THE SUPPLY GATE COUNTS REAL POSITIVES ONLY, and the reason is
    # written six lines below in the tier-split comment: "bench volume
    # alone could clear the supply/resolution gates for a run whose real
    # evidence is too thin". §4BF-1c fixed the TIER and left the GATE
    # counting bench — the hazard its own comment names. Measured on a
    # fresh mine of all 28 day-files: 403 positives, of which 121 real.
    # The gate passed on 282 bench fixtures. Bench may TEACH; it may
    # never GRADE, and it may never be the reason a run starts.
    _real_pos = [f for f in positives if f.get("origin") != "bench"]
    if len(_real_pos) < args.min_fixtures and not args.force_supply:
        sys.stdout.flush()
        print(f"supply gate: {len(_real_pos)} REAL positive fixtures < "
              f"{args.min_fixtures} ({len(positives)} counting bench, which "
              f"may teach but never grade) — wait for more recording days "
              f"(--force-supply to override)", file=sys.stderr)
        return 2

    pub = [f for f in positives if f.get("tier") == "public"]
    priv = [f for f in positives if f.get("tier") == "private"]
    # §4BF 1c (R1 review CRIT): the gate tier is REAL-ONLY. The fixture
    # tier hash ignores origin, so bench-joined fixtures landed in `priv`
    # — the incumbent/candidate comparison (the ONLY evidence a promotion
    # ships on) would be graded partly on bench solves, and bench volume
    # alone could clear the supply/resolution gates for a run whose real
    # evidence is too thin. Bench may TEACH (public side, equal-mass
    # capped against real public fixtures — bench tool-choice profiles are
    # near-uniform `execute`, and uncapped they would dominate the
    # optimizer's replay set); it may never GRADE.
    _bench_priv = [f for f in priv if f.get("origin") == "bench"]
    if _bench_priv:
        pub = pub + _bench_priv
        priv = [f for f in priv if f.get("origin") != "bench"]
        print(f"moved {len(_bench_priv)} bench fixtures out of the PRIVATE "
              f"gate tier (real-only gate) into public")
    _pub_real = [f for f in pub if f.get("origin") != "bench"]
    _pub_bench = [f for f in pub if f.get("origin") == "bench"]
    if len(_pub_bench) > len(_pub_real):
        _pub_bench = _pub_bench[-len(_pub_real):] if _pub_real else []
        pub = _pub_real + _pub_bench
        print(f"public tier equal-mass capped: {len(_pub_real)} real + "
              f"{len(_pub_bench)} bench fixtures")
    if not pub or not priv:
        print("degenerate public/private fixture split", file=sys.stderr)
        return 2

    # ── RESOLUTION CHECK, *BEFORE* the expensive part ─────────────────
    # Both sibling runners (run_gepa.py, optimize_verifier.py) refuse to run
    # when the private tier is too coarse to resolve their own --min-delta;
    # this one did not, and its private tier is the COARSEST of the three.
    #
    # The fixture tier is hashed per REQUEST, and one request emits many
    # fixtures, so the realised private share is not `--private-pct`:
    # measured 2026-08-04 on the real mine, 13 of 65 positives are private
    # (20%, against 30% requested) — a smallest step of 0.077 against a
    # --min-delta of 0.02. Under that arithmetic a single flipped replay
    # decides a run that costs `iterations * len(pub)` main-model calls.
    # `--smoke` is exempt: it evaluates the incumbent and ships nothing.
    # ⚠ A USABLE MARGIN, VALIDATED — ported from `run_gepa.py`'s margin pre-flight, whose
    # comment names each of these failures as already-fixed there. Measured
    # ⚠ NOT `and not args.smoke`. Both siblings validate unconditionally
    # (`run_gepa.py`, `recheck_gepa_incumbent.py`), and an unusable margin
    # is a broken invocation in a smoke run too — `math.ceil(1/x)` raises
    # the same OverflowError whichever mode it is in.
    # here before porting: `--min-delta 0` raised an uncaught
    # ZeroDivisionError out of main(); `1e-320` an uncaught OverflowError
    # from `math.ceil(1/x)`; `1.0` passed both pre-flights, paid for the
    # whole optimizer, then could never ship because `delta > 1.0` is
    # unsatisfiable; and a NEGATIVE margin made `delta > min_delta`
    # trivially true, so `--allow-insignificant-ship` shipped a candidate
    # measurably WORSE than the incumbent (20 incumbent wins, 0 candidate,
    # p=1.0, delta=-0.40, ships=True).
    if not 1e-6 <= args.min_delta < 1:
        sys.stdout.flush()
        print(f"REFUSING TO RUN: --min-delta {args.min_delta} is not a "
              f"usable margin. It must be >=1e-6 (a bar of 0 admits any "
              f"non-zero swing, and anything smaller cannot be resolved by "
              f"a fixture tier of any size) and <1 (no fidelity delta can "
              f"exceed 1.0, so nothing could ever ship).", file=sys.stderr)
        return 2

    # ⚠ RESOLUTION IS NOT POWER — the second half of the §4CY pre-flight.
    # A one-sided exact McNemar cannot reach SHIP_ALPHA with fewer than
    # `_significance_floor()` discordant replays, so a private tier below
    # that CANNOT ship whatever the candidate does, and the run would pay
    # for `iterations * len(pub)` main-model calls before refusing.
    # ⚠ SMOKE SKIPS THE VALIDATION, SO IT MUST SKIP THE ARITHMETIC THAT
    # DEPENDS ON IT — one exemption, applied to the whole block rather
    # than repeated per statement. Round 1 exempted `--smoke` from the
    # margin check (it ships nothing, so an unusable bar is harmless) and
    # left the division below unguarded: `--smoke --min-delta 0` raised
    # the very ZeroDivisionError that check was added to close. A guard
    # with an exemption has two paths and the exempt one was never driven.
    _need = 0
    if not args.smoke:
        _min_discordant = _significance_floor()
        _resolution_need = math.ceil(1.0 / args.min_delta)
        _need = max(_min_discordant, _resolution_need)
        # ⚠ COUNT THE ROWS THAT CAN ACTUALLY PRODUCE EVIDENCE. A fixture
        # whose recorded payload is gone replays in NEITHER arm, so it can
        # never be a discordant pair — counting it here promises a
        # resolution the tier cannot deliver, and the shortfall surfaced
        # only AFTER the optimizer had been paid for. Probing the whole
        # live pool costs ~20s of file reads against `iterations *
        # len(pub)` main-model calls. Measured 2026-08-03: 7 of 12 smoke
        # fixtures were unreplayable at once, and every fixture in the
        # live pool carries an ABSOLUTE `source.file`, so a pruned or
        # moved recordings dir makes the whole tier unreplayable in one
        # step.
        _replayable = [f for f in priv
                       if _load_recorded_payload(f, recordings_dir)
                       is not None]
        _gap = len(priv) - len(_replayable)
        if _gap:
            print(f"{_gap} of {len(priv)} private fixtures have no "
                  f"recorded payload and can never be replayed — the "
                  f"gate tier is effectively {len(_replayable)}")
        priv_effective = len(_replayable)
        if priv_effective < _need:
            # ⚠ ONE COMBINED REQUIREMENT. Reported separately, an operator
            # satisfies the weaker one, re-runs, and only then learns the real
            # number — `run_gepa.py`'s combined pre-flight combines them for exactly this reason,
            # and records that a single-cause message was FALSE in 81% of
            # refusals. Each reason is stated only when it actually applies.
            _why = []
            _below_floor = priv_effective < _min_discordant
            if _below_floor:
                _why.append(f"even a perfect sweep needs {_min_discordant} "
                            f"discordant replays to reach "
                            f"p<={ab_eval.SHIP_ALPHA}, so NO candidate could "
                            f"ship at any margin")
            if priv_effective < _resolution_need:
                # The step is a FRACTION beside its decimal: at a fixed
                # precision it can render identically to the bar it is being
                # compared against, and no precision closes that.
                _why.append(f"a smallest step of 1/{priv_effective} "
                            f"({1.0 / max(1, priv_effective):.6g}) cannot "
                            f"resolve "
                            f"--min-delta {args.min_delta}, so one flipped "
                            f"replay would decide the run")
            # The offered margin is rounded UP and verified AS PRINTED —
            # rendered down it re-triggers the identical refusal, a fixed
            # point rather than a remedy (§4CY).
            _offer = math.ceil(1000.0 / max(1, priv_effective)) / 1000.0
            assert (math.ceil(1.0 / float(f"{_offer:.3f}"))
                    <= max(1, priv_effective)), (
                f"offered --min-delta {_offer} still refuses at "
                f"n={priv_effective}")
            # ⚠ REAL-OVER-REAL. `positives` counts BENCH; `priv` is real-only
            # after the tier split above, so a bench-inflated numerator over a
            # real-only denominator over-states the operator's remaining work —
            # measured, 608 against the miner's 181 on the same mine, 3.4x.
            # `mine_tool_fixtures.py:186` records this exact bug as fixed on
            # its side ("dividing real-private by ALL positives (bench
            # included) stopped measuring the hash split"); the runner, which
            # claims to report the same number, still had it.
            _fix = (f"Collect at least {_need} PRIVATE positives (~"
                    f"{math.ceil(_need * len(_real_pos) / max(1, priv_effective))} "
                    f"REAL positives at today's realised private share)"
                    + ("." if _below_floor else
                       f", or raise --min-delta to at least {_offer:.3f} "
                       f"(which does NOT lower the {_min_discordant}-pair "
                       f"significance floor)."))
            sys.stdout.flush()
            print(f"REFUSING TO RUN: {priv_effective} REPLAYABLE private "
                  f"positive fixtures is not enough — {'; and '.join(_why)}. {_fix}", file=sys.stderr)
            return 2

    # ⚠ ONLY THE COMPONENTS THIS RUN COULD PROMOTE. Globbing every
    # `tool_description.*.json` refused a legitimate run because some
    # OTHER signature — one this run cannot even select, e.g. a tool with
    # a single public fixture — had been promoted an hour ago. The
    # sibling it was ported from is signature-scoped
    # (`run_gepa.py`'s `output_path.exists()`), and the remedy this
    # printed, `--min-promotion-age-days 0`, disables the guard for EVERY
    # component including the one being re-promoted: every exemption
    # became the next bypass.
    _optim_now = ghost_home / "system" / "optim"
    _young = {}
    if not args.smoke:
        _counts = Counter(_first_choice(f) for f in pub)
        _will_touch = {
            f"{registry_mod.TOOL_DESC_SIGNATURE_PREFIX}{n}"
            for n, _c in _counts.most_common(args.components)}
        for _p in sorted(_optim_now.glob(
                f"{registry_mod.TOOL_DESC_SIGNATURE_PREFIX}*.json")):
            if _p.stem not in _will_touch:
                continue
            _a = _too_young(_p)
            if _a is not None:
                _young[_p.stem] = _a
    if _young:
        sys.stdout.flush()
        print("REFUSING TO RUN: "
              + "; ".join(f"{k} was promoted {v:.1f} days ago"
                          for k, v in sorted(_young.items()))
              + f" and --min-promotion-age-days is "
              f"{args.min_promotion_age_days}. Each run is a fresh draw "
              f"at the gate, so re-promoting before the last one can be "
              f"judged turns one decision into many — and because this "
              f"gate's artifacts cannot be re-scored offline, every "
              f"re-promotion resets the live check's era and discards "
              f"the turns accrued against the current one. Wait, or pass "
              f"--min-promotion-age-days 0 to override deliberately.",
              file=sys.stderr)
        return 2

    baselines = _baseline_descriptions()
    counts = Counter(_first_choice(f) for f in pub)
    comp_tools = [n for n, _ in counts.most_common(args.components)
                  if n in baselines]
    seed_candidate = {}
    for n in comp_tools:
        # Seed from a live artifact when present+valid, else the registry
        # baseline — the gate compares against what production runs.
        text = baselines[n]
        try:
            live = json.loads((ghost_home / "system" / "optim" /
                               f"tool_description.{n}.json").read_text())[
                "optimized_instruction"]
            if registry_mod._validate_tool_description(n, baselines[n], live):
                text = live
        except Exception:
            pass
        seed_candidate[f"tool_description.{n}"] = text
    print(f"{len(pub)} public / {len(priv)} private positives; "
          f"components: {sorted(seed_candidate)}")

    # gepa candidates key by component name; the adapter swaps by TOOL name.
    def _by_tool(cand: Dict[str, str]) -> Dict[str, str]:
        return {k.split(".", 1)[1]: v for k, v in cand.items()}

    class _Adapter(ToolDescAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            return super().evaluate(batch, _by_tool(candidate),
                                    capture_traces)

    # ⚠ THE REPLAY DEADLINE IS A FLAG, LIKE ITS SIBLINGS'. A hard-coded
    # 120s while `recheck_gepa_incumbent.py` and `run_gepa.py` both use
    # 360 means a replay this gate calls a TRANSPORT FAILURE — excluded
    # from the statistic AND, in bulk, an abort — is one the instrument
    # that re-checks the same artifact scores normally. The two are
    # supposed to measure the same thing.
    adapter = _Adapter(args.upstream_url, recordings_dir, baselines,
                       timeout=args.timeout)

    inc_eval = adapter.evaluate(priv, seed_candidate, capture_traces=True)
    inc_acc = sum(t["score"] for t in inc_eval.trajectories) / len(priv)
    # ⚠ THE TWO STATES THE SENTINEL EXISTS TO SEPARATE, REPORTED
    # SEPARATELY. One count printed as "(N unreplayable)" said the same
    # thing about a fixture with no recorded payload (a corpus gap, stable
    # across both arms) and a replay that never reached the model (an
    # outage, which is exactly what invalidates the pairing).
    # ⚠ COUNTED BY CATEGORY, NOT BY SUBTRACTION. `n_err - n_down` labelled
    # EVERY other truthy err "unreplayable", and there is a third: a
    # candidate refused by the per-tool cap sets a descriptive `err`, so
    # the line read "(60 unreplayable, 0 transport-failed)" for a run
    # where nothing was unreplayable at all. The counts now come from the
    # same predicates the gate uses, so the printed number and
    # `transport_excluded` cannot diverge.
    n_err = sum(1 for t in inc_eval.trajectories if t.get("err"))
    n_down = sum(1 for t in inc_eval.trajectories if _outage(t))
    n_gap = sum(1 for t in inc_eval.trajectories
                if _transport_failed(t) and not _outage(t))
    n_other = n_err - n_down - n_gap
    print(f"INCUMBENT tool-choice fidelity on PRIVATE: {inc_acc:.3f} "
          f"({n_gap} unreplayable, {n_down} transport-failed"
          + (f", {n_other} other" if n_other else "") + ")")
    if args.confusion_out:
        _dump_confusion(inc_eval.trajectories, Path(args.confusion_out))
        print(f"wrote {len(inc_eval.trajectories)} replay rows to "
              f"{args.confusion_out}")
    # ⚠ THE ABORT MUST NOT BE SMOKE-ONLY. The identical `n_down` is
    # computed above for every run, and the non-smoke path fell through
    # to `gepa.optimize`: driven against the real corpus with a dead
    # upstream, the incumbent arm reported "0.000 (0 unreplayable, 35
    # transport-failed)" and the run then paid for **1032 rollouts**
    # before refusing. A guard with an exemption exempts the expensive
    # path — the "fixed point costing a full optimizer run each attempt"
    # this entry named two rounds earlier.
    # ONE abort for both paths — `--smoke` keeps its own wording because
    # de-risking the replay path is its whole job, and the expensive path
    # says what it is about to be spared.
    _lead = "SMOKE FAILED" if args.smoke else "REFUSING TO RUN"
    _cost = ("" if args.smoke else
             " The candidate arm would fail the same way and the run "
             "would refuse at the evidence bar — after paying for "
             "`iterations * len(pub)` main-model calls.")
    if n_down and n_down == len(priv):
        print(f"{_lead}: none of the {len(priv)} replays reached the "
              f"model ({n_down} transport failures).{_cost} Re-run when "
              f"the upstream is stable.", file=sys.stderr)
        return 2
    if n_gap and n_gap == len(priv):
        print(f"{_lead}: none of the {len(priv)} private fixtures has a "
              f"recorded payload, so nothing could be replayed. Re-mine, "
              f"or point --recordings at the directory the fixtures "
              f"reference.", file=sys.stderr)
        return 2
    if args.smoke:
        for t in inc_eval.trajectories[:8]:
            print(f"  {t['truth']:<18} -> {str(t['picked']):<18} "
                  f"score={t['score']} {t.get('err', '')}")
        # ⚠ AN EXIT CODE THAT MEANS SOMETHING. `--smoke`'s one job is to
        # de-risk the replay path, and `--force-supply` REQUIRES it, so it
        # is the sanctioned first step of the loop. Driven against the
        # real corpus with the upstream pointed at a dead port: 35 of 35
        # replays never reached the model and it exited **0**. The counts
        # distinguished the state on stdout; the code an operator or a
        # script branches on did not.
        # The two aborts above already fired for both paths.
        return 0

    iterations = min(args.max_iterations, MAX_OPT_ITERATIONS)
    import gepa
    # The cap must be stated in the PROPOSAL PROMPT, not only enforced by
    # zero-scoring. `make_reflective_dataset` skips every trajectory carrying
    # an `err`, so the `[cap-guard]` feedback was dropped alongside the
    # unreplayable-fixture feedback and the two were INDISTINGUISHABLE to the
    # reflector. Worse, the cap is a property of the CANDIDATE, so an
    # over-cap proposal errs on every fixture at once and the component's
    # reflective dataset comes back EMPTY — the reflector is asked to improve
    # an instruction while being told nothing whatsoever. That is the same
    # shape as optimize_verifier's run 3a (37 straight ~5KB proposals, all
    # silently zeroed), which was fixed the same way.
    _caps = {n: max(6000, 3 * len(baselines.get(n, "")))
             for n in comp_tools}
    reflection_template = None
    if _caps:
        from gepa.strategies.instruction_proposal import (
            InstructionProposalSignature,
        )
        _cap_min = min(_caps.values())
        # Both numbers come from the read-site, never from a literal here:
        # a prompt that advertises a stale ceiling teaches the reflector to
        # aim at a cap production does not enforce.
        _agg = getattr(registry_mod, "_TOOL_DESC_AGGREGATE_SLACK", 20_000)
        reflection_template = (
            InstructionProposalSignature.default_prompt_template
            + f"\n\nHARD CONSTRAINT: the new tool description MUST be UNDER "
              f"{_cap_min} characters (aim for ~{int(_cap_min * 0.5)}). It "
              "lives in the KV-PINNED tools block that prefixes EVERY "
              "request, so length is paid on every turn by every tool, not "
              "just this one. Descriptions over the cap score ZERO, and the "
              "whole promoted SET is dropped if the tools block inflates by "
              f"more than {_agg:,} characters in total. Sharpen the "
              "disambiguation against sibling tools; do not pad."
        )
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=pub,
        adapter=adapter,
        reflection_lm=_make_reflection_lm(args.upstream_url),
        max_metric_calls=iterations * len(pub),
        display_progress_bar=True,
        seed=0,
        run_dir=args.run_dir or None,
        reflection_prompt_template=reflection_template,
    )
    best = dict(result.best_candidate)

    cand_eval = adapter.evaluate(priv, best, capture_traces=True)
    cand_acc = sum(t["score"] for t in cand_eval.trajectories) / len(priv)
    delta = cand_acc - inc_acc
    # The two arms run hours apart on one shared slot, so the number that
    # matters is not either count but the ASYMMETRY between them — and the
    # incumbent's count was the only one ever printed.
    _cand_err = sum(1 for t in cand_eval.trajectories if t.get("err"))
    _cand_down = sum(1 for t in cand_eval.trajectories if _outage(t))
    _cand_gap = sum(1 for t in cand_eval.trajectories
                    if _transport_failed(t) and not _outage(t))
    _cand_other = _cand_err - _cand_down - _cand_gap
    print(f"CANDIDATE tool-choice fidelity on PRIVATE: {cand_acc:.3f} "
          f"({_cand_gap} unreplayable, {_cand_down} transport-failed"
          + (f", {_cand_other} other" if _cand_other else "")
          + f", against {n_gap}/{n_down} for the incumbent)")
    by_tool = _by_tool(best)
    valid = all(registry_mod._validate_tool_description(
        n, baselines.get(n, ""), t) for n, t in by_tool.items())
    # The READ-SITE additionally enforces an all-or-nothing aggregate ceiling
    # (`_TOOL_DESC_AGGREGATE_SLACK`) across the whole tools block. Checking
    # only the per-tool caps let a candidate set pass this gate, get
    # PROMOTED, and then be 100% inert in production: measured, 6 components
    # each individually valid summed to 38,248 chars of inflation against a
    # 20,000 ceiling -> 0 of 6 descriptions reached the model on next boot,
    # while the ship line said `valid=True ships=True`.
    _inflation, _incumbent_inflation = _aggregate_inflation(
        by_tool, baselines, ghost_home)
    _slack = getattr(registry_mod, "_TOOL_DESC_AGGREGATE_SLACK", 20_000)
    aggregate_ok = _inflation <= _slack
    if not aggregate_ok:
        print(f"AGGREGATE REJECT: with this candidate set promoted, the WHOLE "
              f"tools block inflates by {_inflation} chars against the "
              f"read-site ceiling of {_slack} (already-promoted artifacts "
              f"account for {_incumbent_inflation}) — every tuned description "
              "on disk, not just this run's, would be silently dropped in "
              "production. Not shipping.", file=sys.stderr)
    _dec = _ship_decision(
        list(getattr(inc_eval, "trajectories", []) or []),
        list(getattr(cand_eval, "trajectories", []) or []),
        min_delta=args.min_delta, valid=valid,
        aggregate_ok=aggregate_ok, min_usable=_need,
        allow_insignificant=args.allow_insignificant_ship)
    if _dec.unpaired:
        print("⚠ CANNOT PAIR THE ARMS: the two evaluations returned "
              "different numbers of trajectories, so no significance test "
              "is possible and nothing ships.", file=sys.stderr)
    if _dec.underpowered:
        # ⚠ THE REMEDY DEPENDS ON THE CAUSE, and one message for both was
        # what round 4 was right to object to. An outage is re-runnable; a
        # corpus gap that opened DURING the run is not — it means the
        # recordings moved under the two arms, and re-running reproduces
        # it.
        if _dec.outage_excluded and not _dec.gap_excluded:
            _why_short = ("re-run when the upstream is stable")
        elif _dec.gap_excluded and not _dec.outage_excluded:
            _why_short = (
                f"{_dec.gap_excluded} pairs lost their recorded payload "
                f"BETWEEN the pre-flight and now — the recordings moved "
                f"under the run. Re-mine or restore "
                f"{recordings_dir} and re-run; re-running against the "
                f"same state reproduces this")
        else:
            _why_short = (
                f"{_dec.outage_excluded} lost to a transport outage and "
                f"{_dec.gap_excluded} to a missing recorded payload — the "
                f"first is re-runnable, the second needs the recordings "
                f"restored first")
        print(f"⚠ EVIDENCE BELOW THE PRE-FLIGHT BAR: only {_dec.usable} of "
              f"{len(priv)} pairs reached a verdict in BOTH arms, under "
              f"the {_need} this run was allowed to start on. Nothing "
              f"ships — {_why_short} (--allow-insignificant-ship does NOT "
              f"override this; it waives significance, not evidence).",
              file=sys.stderr)
    if _dec.transport_excluded:
        print(f"   {_dec.transport_excluded} of {len(priv)} pairs excluded "
              f"({_dec.outage_excluded} transport outage, "
              f"{_dec.gap_excluded} no recorded payload); margin over the "
              f"{_dec.usable} usable pairs is "
              f"{_dec.paired_delta:+.3f} (incumbent "
              f"{_dec.paired_incumbent:.3f} candidate "
              f"{_dec.paired_candidate:.3f}), against {delta:+.3f} over all "
              f"rows. THE GATE USES THE PAIRED NUMBER.", file=sys.stderr)
    if _dec.overridden:
        print("   --allow-insignificant-ship given; treating the margin as "
              "sufficient despite the discordant replays.", file=sys.stderr)
    ships = _dec.ships
    _p, _inc_wins, _cand_wins = (_dec.p_value, _dec.incumbent_wins,
                                 _dec.candidate_wins)
    _cleared_margin, _significant = _dec.cleared_margin, _dec.significant
    _ship_override = _dec.overridden

    _p_str = "n/a (no discordant replays)" if _p is None else f"{_p:.4f}"
    # ⚠ THE LINE MUST STATE THE NUMBER THE GATE DECIDED ON. It printed the
    # RAW delta beside `ships=`, while `cleared_margin` reads
    # `paired_delta`. Driven with 6 candidate-arm transport failures on
    # rows the incumbent passed plus 3 honest wins, the rejection sentence
    # read "the candidate cleared the margin (delta -0.0500, bar 0.02)" —
    # a NEGATIVE delta against a positive bar, in the same sentence as
    # "cleared". With the override it then promoted while the A/B line
    # said `delta=-0.050 ships=True`.
    print(f"A/B (PRIVATE fixtures, n={len(priv)}, "
          f"{_dec.usable} usable pairs): "
          f"incumbent={_dec.paired_incumbent:.3f} "
          f"candidate={_dec.paired_candidate:.3f} "
          f"delta={_dec.paired_delta:+.3f} (bar "
          f"{args.min_delta}; raw over all rows "
          f"{inc_acc:.3f}/{cand_acc:.3f}, {delta:+.3f}) valid={valid} "
          f"aggregate_ok={aggregate_ok} (+{_inflation}/{_slack} chars) "
          f"McNemar p={_p_str} over {_dec.discordant} discordant "
          f"replays ({_cand_wins} candidate / {_inc_wins} incumbent) "
          f"ships={ships}")

    optim_dir = ghost_home / "system" / "optim"
    optim_dir.mkdir(parents=True, exist_ok=True)
    _promoted_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # ⚠ THE GATE IS SET-LEVEL AND THE ARTIFACTS ARE PER-COMPONENT.
    # ⚠ THE §4CW SEED VETO, PORTED. This gate SEEDS FROM THE LIVE
    # ARTIFACT when one exists, so run N's two arms are artifact-(N-1) and
    # artifact-N — the hand-written description is in NEITHER, and each
    # run only has to beat the previous winner. Driven, two consecutive
    # promotions into one home:
    #
    #   run1  baseline_instruction len 569  == registry baseline
    #   run2  baseline_instruction len 599  == run1's OPTIMIZED text
    #
    # `run_gepa.py`'s seed arm exists for exactly this and measured the
    # damage on its own signature (chain 0.393 vs hand-written 0.484).
    # Here the escape hatch is closed on both sides:
    # `recheck_gepa_incumbent.py` exits 3 for every `tool_description.*`
    # signature — it cannot re-score them — so `gepa_live_check` at ~3.5
    # turns/day is the only judge, and every re-promotion resets its era.
    #
    # Cost: ONE extra private-tier evaluation, and only when an artifact
    # is live AND the main gate already passed. A first promotion (seed ==
    # hand-written) and a rejected run pay nothing.
    _hand_written = {f"tool_description.{n}": baselines[n]
                     for n in comp_tools}
    _ratcheting = sorted(k for k, v in _hand_written.items()
                         if seed_candidate.get(k, "").strip() != v.strip())
    _seed_dec = None
    _seed_vetoed = False
    _seed_undecidable = False
    if ships and _ratcheting:
        print(f"SEED ARM: {len(_ratcheting)} component(s) seeded from a "
              f"LIVE artifact, so the A/B above did not include the "
              f"hand-written text. Scoring it — a candidate that beats "
              f"the incumbent while LOSING to the hand-written baseline "
              f"is a ratchet, not an improvement: "
              f"{', '.join(_ratcheting)}")
        _seed_eval = adapter.evaluate(priv, _hand_written,
                                      capture_traces=True)
        _seed_acc = (sum(t["score"] for t in _seed_eval.trajectories)
                     / max(1, len(priv)))
        # The arms are (candidate, hand-written): "ships" here means the
        # HAND-WRITTEN text measurably beats the candidate.
        _seed_dec = _ship_decision(
            list(getattr(cand_eval, "trajectories", []) or []),
            list(getattr(_seed_eval, "trajectories", []) or []),
            min_delta=args.min_delta, valid=True, aggregate_ok=True,
            min_usable=_need, allow_insignificant=False)
        _seed_p = ("n/a" if _seed_dec.p_value is None
                   else f"{_seed_dec.p_value:.4f}")
        # ⚠ THE DECISION HELPER'S SLOTS ARE (incumbent=CANDIDATE,
        # candidate=SEED) on this call — that inversion is what makes
        # `ships` mean "the hand-written text beats the candidate". So
        # `paired_incumbent` is the CANDIDATE's rate here, and the first
        # version of this line printed it under the label "hand-written":
        # one sentence whose halves contradicted each other (raw rates
        # right, paired rates swapped), and the record carried the same
        # swap — a promoted-because-it-lost artifact whose recorded rates
        # said it won. Lens C, finding B1.
        print(f"SEED ARM: hand-written={_seed_acc:.3f} "
              f"candidate={cand_acc:.3f}; over the "
              f"{_seed_dec.usable} usable pairs "
              f"hand-written={_seed_dec.paired_candidate:.3f} "
              f"candidate={_seed_dec.paired_incumbent:.3f} "
              f"(seed-minus-candidate {_seed_dec.paired_delta:+.3f}, bar "
              f"{args.min_delta}), McNemar p={_seed_p} over "
              f"{_seed_dec.discordant} discordant replays")
        # ⚠ THE SEED ARM GETS THE SAME EXCLUSION AND POWER TREATMENT AS
        # THE MAIN ONE, so an outage can neither SUPPRESS the veto (by
        # eating the pairs that would have fired it) nor MANUFACTURE one.
        # `run_gepa`'s round-2 note records both directions.
        if _seed_dec.underpowered:
            print(f"   ⚠ THE SEED ARM IS UNDERPOWERED: {_seed_dec.usable} "
                  f"of {len(priv)} pairs reached a verdict in both arms, "
                  f"under the {_need} this run started on "
                  f"({_seed_dec.outage_excluded} transport outage, "
                  f"{_seed_dec.gap_excluded} no recorded payload). The "
                  f"veto cannot fire on evidence this thin, so the run "
                  f"is refused rather than promoted on a suppressed "
                  f"check.", file=sys.stderr)
            ships = False
            # ⚠ NOT `_seed_vetoed = True`. Nothing was measured: the
            # record said the veto FIRED and the shared reader printed
            # "THE SEED ARM FIRED THE VETO" about a check that never ran.
            _seed_undecidable = True
        elif _seed_dec.ships:
            _seed_vetoed = True
            if args.allow_seed_loss:
                print("   --allow-seed-loss given; promoting a candidate "
                      "that loses to the hand-written baseline. RECORDED.",
                      file=sys.stderr)
            else:
                ships = False
                print(f"SEED VETO: the candidate beats the live artifact "
                      f"but LOSES to the hand-written description by "
                      f"{_seed_dec.paired_delta:+.3f} "
                      f"(p={_seed_p} <= {ab_eval.SHIP_ALPHA}). This is "
                      f"the ratchet §4CW closed for the sibling gate: "
                      f"each run beats the last one and the chain drifts "
                      f"below where it started. Retire the live "
                      f"artifact(s) first (scripts/gepa_live_check.py "
                      f"--revert), or override with --allow-seed-loss.",
                      file=sys.stderr)

    # `ships` is ONE decision from ONE A/B over the whole candidate set,
    # and this loop wrote N artifacts each stamped with that set-level
    # gate block. Driven with GEPA mutating exactly one component — which
    # is what a real proposal does — two descriptions BYTE-IDENTICAL to
    # the incumbent were promoted, each carrying `p_value: 0.003906,
    # candidate_wins: 8, gate_arm: "…[paired-v2]"`: a claim of a
    # measured, significant win belonging entirely to a third component.
    # `recheck_gepa_incumbent` then prints it back per-signature.
    #
    # Two things follow. An UNCHANGED component has nothing to promote,
    # so it is skipped — that is the byte-identical case, closed. And a
    # component that DID change still ships on the set's evidence, so
    # the artifact records the set it was judged with, instead of
    # implying the number is its own. Measuring per-component
    # contribution would need N more A/B passes and is not what this
    # gate does; saying so in the record is.
    _changed = {c: t for c, t in best.items()
                if t.strip() != seed_candidate.get(c, "").strip()}
    _unchanged = sorted(set(best) - set(_changed))
    if _unchanged:
        print(f"{len(_unchanged)} component(s) unchanged by the "
              f"optimizer — not promoted (nothing to promote, and a "
              f"byte-identical artifact stamped with the set's win is a "
              f"false audit record): {', '.join(_unchanged)}")
    # ⚠ THE VERDICT AND THE EXIT CODE MUST AGREE. `ships = False` here
    # dropped the run into the "A/B gate REJECTED — live descriptions
    # stand." message and exit 1 — the SAME code a genuine loss returns.
    # A caller could not tell "the candidate lost" from "the optimizer
    # returned the seed verbatim", which is a broken reflection LM or a
    # wasted run, not a verdict about the incumbent. Round 11 split
    # `recheck`'s codes for exactly this collision.
    # 0 = promoted, 1 = the gate rejected the candidate,
    # 2 = refused to run / aborted, 3 = no candidate was produced.
    # ⚠⚠ AND NOT `ships and not _changed`. Gating it on the A/B made the
    # code UNREACHABLE for the case it documents: replays run at
    # `temperature: 0.0`, so a byte-identical candidate produces
    # byte-identical requests in both arms, the paired delta is exactly
    # 0, `ships` is False — and the run exited 1, the same code as a
    # measured rejection, which is the collision this branch exists to
    # close. Driven end to end:
    #
    #   delta=+0.000 (bar 0.02) ... ships=False
    #   1 component(s) unchanged by the optimizer — not promoted
    #   A/B gate REJECTED — live descriptions stand.        rc=1
    #
    # The round-15 pin reached rc=3 only because the harness scores by
    # fixture INDEX and ignores the candidate text, awarding a
    # byte-identical candidate a 6-0 sweep — a corpus the pipeline cannot
    # produce (`harness-grades-own-homework`). "No candidate" is a
    # property of the OPTIMIZER'S OUTPUT, not of the gate's arithmetic
    # over it: with nothing proposed there is nothing to accept and
    # nothing to reject.
    _no_candidate = not _changed
    if _no_candidate:
        print("NO CANDIDATE: the optimizer returned the seed verbatim "
              "for every component — there is nothing to promote and "
              "nothing to reject. This is a wasted run (or a broken "
              "reflection LM), not a verdict about the incumbent.",
              file=sys.stderr)
        ships = False
    # ⚠ AND THE REJECT PATH HAS THE SAME DEFECT THE PROMOTE PATH JUST
    # LOST. `(_changed if ships else best)` wrote a `.candidate.rejected`
    # record for every component INCLUDING the untouched ones: driven
    # with one of three mutated, two records byte-identical to the
    # incumbent each carried the set's `p_value`/`candidate_wins` and a
    # `co_promoted` list that did not contain the file itself. An
    # untouched component has no candidate — there is nothing to reject
    # and nothing to record. `_changed` on BOTH paths.
    # ⚠ STAGE ALL, THEN SWAP ALL (§4DF round 1, MAJOR-4). The old loop
    # backed up + replaced PER COMPONENT, so an OSError on component N
    # (ENOSPC is the realistic one — the backup/staging writes) aborted
    # with components 1..N-1 already LIVE, exit 1, no REJECTED marker:
    # the launcher then notified "nothing was believed or acted on"
    # about a partial promotion the epoch swap was busy deploying, and
    # every record named a `co_promoted` set that never fully promoted.
    # Now every failure-prone write (backup copy, staging write, record
    # validation) happens BEFORE the first `os.replace`; the only
    # residual partial window is a crash between the bare renames.
    _to_swap = []
    for comp, text in _changed.items():
        payload = json.dumps({
            "signature_name": comp,
            # ⚠ TWO DIFFERENT THINGS HAVE BEEN CALLED `baseline_instruction`.
            # `run_gepa.py` writes the HAND-WRITTEN seed; this gate writes the
            # arm it actually compared against, which is the PREVIOUS
            # ARTIFACT once one exists — and `recheck_gepa_incumbent.py`
            # reads the key as "the hand-written baseline" ("Does the LIVE
            # artifact still beat the hand-written baseline?"). Both are
            # recorded, so the ratchet leaves a trace either way.
            "baseline_instruction": seed_candidate[comp],
            "hand_written_baseline": _hand_written.get(comp, ""),
            "seeded_from_live_artifact": comp in _ratcheting,
            "optimized_instruction": text,
            "optimizer": "GEPA-tool-desc",
            "iterations": iterations,
            "private_incumbent": round(inc_acc, 4),
            "private_candidate": round(cand_acc, 4),
            # ⚠ THE SAME SHAPE AND THE SAME KEY NAMES AS `run_gepa.py`,
            # deliberately. §4DA first stamped these seven fields FLAT and
            # named the count `discordant_replays`; §4CY writes them nested
            # under `gate` as `discordant_pairs`, and
            # `recheck_gepa_incumbent.py's `art["gate"]` read` reads `art["gate"]`. So the
            # audit trail §4DA added specifically so an override could be
            # re-examined was written in a shape the only reader of that
            # trail cannot open — the override warning could never print
            # for a tool-description artifact. Two gates, one vocabulary.
            # ⚠ PROMOTION-ONLY. `gate_arm` is the loader's proxy for "this
            # artifact has gate provenance" (the loader's `gate_arm` provenance check), and
            # `promoted_utc` says a promotion happened. Stamped on the
            # `.candidate.rejected` file too, a rejected candidate renamed
            # into place — plausible right next to the `.prev` restore
            # workflow — loads as a gated artifact instead of raising the
            # "predates the gate schema" warning. `run_gepa.py`'s gate stamp stamps
            # only on the promote path.
            **({"gate_arm": ("tool-choice fidelity A/B, private holdout "
                             f"[{ab_eval.GATE_METRIC_VERSION}]"
                             + (" (SIGNIFICANCE OVERRIDDEN)"
                                if _ship_override else ""))}
               if ships else
               {"gate_arm_candidate": "tool-choice fidelity A/B, private "
                                      "holdout (NOT PROMOTED)"}),
            "gate": {
                "metric": "tool_choice_fidelity>=1.0",
                "n_private": len(priv),
                "n_usable_pairs": _dec.usable,
                "transport_excluded": _dec.transport_excluded,
                # ⚠ THE PAIRED RATES ARE THE ONES THE GATE COMPARED, and
                # they were computed, printed to stderr and never
                # recorded — so an artifact promoted through an outage
                # could not reconstruct its own comparison. The raw pair
                # is kept beside them under `raw_*`, because that is what
                # the operator saw scroll past.
                "incumbent_pass_rate": round(_dec.paired_incumbent, 4),
                "candidate_pass_rate": round(_dec.paired_candidate, 4),
                # The RAW delta is what the operator saw printed; the
                # PAIRED one is what the gate actually decided on. Record
                # both — they differ exactly when transport failed, which
                # is when a promotion most needs re-examining.
                # ⚠ ONE NAME PER NUMBER. This block briefly carried
                # `delta` and `paired_delta` as byte-identical values, and
                # `raw_*_pass_rate` duplicating the top-level
                # `private_incumbent`/`private_candidate`: four names for
                # one delta, three for each rate. `delta` is the DECIDING
                # number, matching every sibling gate's use of the key;
                # `raw_delta` is the all-rows one. The raw RATES live at
                # top level as `private_incumbent`/`private_candidate`.
                "delta": round(_dec.paired_delta, 4),
                "raw_delta": round(delta, 4),
                "outage_excluded": _dec.outage_excluded,
                "corpus_gap_excluded": _dec.gap_excluded,
                # This gate replays RECORDINGS, so the two causes come
                # from distinct predicates and the split above is real.
                "exclusion_cause_distinguished": True,
                "min_delta": args.min_delta,
                # The evidence behind the delta, not just the delta — an
                # artifact whose record cannot answer "how many replays
                # actually moved?" cannot be re-checked later.
                "p_value": (None if _p is None else round(_p, 6)),
                "ship_alpha": ab_eval.SHIP_ALPHA,
                "discordant_pairs": _dec.discordant,
                "candidate_wins": _cand_wins,
                "incumbent_wins": _inc_wins,
                "significance_overridden": _ship_override,
                # The §4CW arm, recorded whether or not it fired — an
                # artifact whose record cannot say whether the
                # hand-written text was ever scored cannot be re-examined.
                # ⚠ BUILT, NOT HAND-WRITTEN. The first version of this
                # block invented `hand_written_pass_rate` and
                # `seed_loss_overridden` while `run_gepa.py` wrote
                # `seed_pass_rate` and `overridden` — and
                # `recheck_gepa_incumbent.py` reads `overridden`, so the
                # "THAT PROMOTION USED --allow-seed-loss" warning was
                # structurally unreachable for every artifact this gate
                # writes. That is the round-7 defect, reproduced by the
                # round-16 fix that ported the veto.
                # ⚠ THE SLOTS ARE INVERTED ON THIS CALL — see the SEED
                # ARM print. `paired_candidate` is the SEED's rate,
                # `paired_incumbent` the candidate's; the wins cross the
                # same way. The schema's identity check (delta ==
                # seed_rate - candidate_rate) refuses the swap at write
                # time now, which is how this one was caught.
                "seed_arm": (None if _seed_dec is None else
                             gate_contract.build_seed_arm(
                                 seed_pass_rate=_seed_dec.paired_candidate,
                                 candidate_pass_rate=(
                                     _seed_dec.paired_incumbent),
                                 seed_minus_candidate_delta=(
                                     _seed_dec.paired_delta),
                                 seed_minus_candidate_raw_delta=(
                                     _seed_acc - cand_acc),
                                 n_usable_pairs=_seed_dec.usable,
                                 transport_excluded=(
                                     _seed_dec.transport_excluded),
                                 seed_wins=_seed_dec.candidate_wins,
                                 candidate_wins=_seed_dec.incumbent_wins,
                                 p_value=_seed_dec.p_value,
                                 vetoed=_seed_vetoed,
                                 undecidable=_seed_undecidable,
                                 overridden=bool(
                                     _seed_vetoed
                                     and args.allow_seed_loss))),
                # ⚠ THE SET THIS COMPONENT WAS JUDGED WITH. The gate ran
                # ONE A/B over all of them, so the numbers above are the
                # set's, not this component's — an artifact that does not
                # say so implies a per-component measurement nobody made.
                # ⚠ AND THE NAME MUST BE TRUE ON THE PATH IT IS WRITTEN
                # ON. A `.candidate.rejected` record carrying
                # `co_promoted` names a set that was NOT promoted —
                # `recheck_gepa_incumbent.py` prints this field back to
                # an operator as "this win is the SET's". Only the ship
                # path promotes anything.
                **({"co_promoted": sorted(_changed)} if ships else
                   {"co_candidates": sorted(_changed)}),
                # ⚠ AND IT MUST NOT UNDER-CLAIM EITHER. With a SINGLE
                # changed component the A/B compared seed-set against
                # seed-set-with-this-one-change, so it measured exactly
                # this component's contribution — and the record said
                # "no per-component contribution was measured" anyway.
                # One changed component is the ordinary case (it is what
                # a real proposal does), so the false wording was the
                # common one, on a record an operator weighs before
                # retiring a live artifact.
                "gate_scope": (
                    ("solo — the A/B differed from the seed set in this "
                     "component alone, so the numbers above ARE this "
                     "component's contribution")
                    if len(_changed) == 1 else
                    ("set — one A/B over all co-promoted "
                     "components; no per-component "
                     "contribution was measured")),
                **({"promoted_utc": _promoted_utc} if ships else {}),
            },
        }, indent=2)
        gate_contract.validate_gate_record(
            json.loads(payload)["gate"],
            writer="scripts/optimize_tool_descriptions.py")
        live = optim_dir / f"{comp}.json"
        if ships:
            # ⚠ BACK UP THE INCUMBENT FIRST, AND ABORT IF THAT FAILS —
            # `run_gepa.py`'s `.prev` backup does, and its comment records why: a
            # candidate that clears the bar against a WEAK seed replaces a
            # stronger incumbent unrecoverably. A promotion made under
            # --allow-insignificant-ship is an operator judgement call, and
            # this was destroying the only thing that could undo it.
            try:
                if live.exists():
                    backup = live.with_suffix(live.suffix + ".prev")
                    try:
                        shutil.copy2(live, backup)
                        print(f"incumbent backed up to {backup}")
                    except OSError as e:
                        print(f"WARNING: could not back up {live} ({e}) "
                              f"— promotion of {comp} aborted",
                              file=sys.stderr)
                        raise
                # ⚠ STAGE + os.replace, NOT write_text. The comment above
                # cites `run_gepa.py`'s `.prev` backup for the backup; `run_gepa.py`'s `os.replace(staging_path, output_path)`
                # ALSO does `os.replace(staging_path, output_path)`, and round
                # 2 ported half the discipline. A torn `write_text` (crash,
                # ENOSPC, or a first read landing inside the truncate window)
                # leaves invalid JSON, and `loader.py` caches the failure as
                # `None` for the life of the PROCESS — so repairing the file
                # on disk does not bring the signature back, and the only
                # trace is a `logger.debug`. Re-promoting an already-live tool
                # is the normal case.
                _staged = live.with_suffix(live.suffix + ".staging")
                _staged.write_text(payload)
            except OSError:
                # An abort must leave NOTHING promoted: drop every
                # already-staged sibling so a re-run starts clean, then
                # let the abort propagate (exit 1, no verdict marker —
                # the launcher files it as an instrument failure, which
                # is now TRUE: no component went live).
                for _s, _l in _to_swap:
                    try:
                        _s.unlink()
                    except OSError:
                        pass
                raise
            _to_swap.append((_staged, live))
        else:
            (optim_dir / f"{comp}.json.candidate.rejected").write_text(payload)
    # Every failure-prone write is done; the renames are the cheap tail.
    for _staged, live in _to_swap:
        os.replace(_staged, live)
        print(f"{gate_contract.GATE_PROMOTED_MARKER_OTD}{live}")
    # ⚠ THE VERDICT LINE MUST MATCH THE EXIT IT CLAIMS (§4DF round 1,
    # MAJOR-2). The old order printed the REJECTED (or NO CANDIDATE)
    # marker and THEN decided the exit — so an underpowered run carried
    # a verdict marker on stdout beside exit 2, and §4DF had just made
    # those exact strings load-bearing for an autonomous caller. The
    # code is computed FIRST; each marker prints only on the exit that
    # claims it. Precedence: no-candidate outranks underpowered — "no
    # candidate" is a property of the OPTIMIZER'S OUTPUT, and two
    # byte-identical arms are GUARANTEED underpowered, so the old
    # 2-before-3 order could relabel every wasted run as thin data.
    # (An abort is still not a rejection: lens C, C4(iii).)
    # The if/elif GUARDS below mirror the return conditional's order
    # exactly — the conformance suite's fail-closed literal scan refuses
    # a Name in a return value, so the ordering is stated twice, one
    # statement apart, on purpose.
    if ships:
        pass  # the PROMOTED lines printed at the swap loop above
    elif _no_candidate:
        # The verdict line must not contradict the one above it: a run
        # with no candidate must not ALSO print a rejection.
        print(f"{gate_contract.GATE_NO_CANDIDATE_MARKER} — live "
              f"descriptions stand, and nothing was "
              "measured about them.")
    elif _dec.underpowered or _seed_undecidable:
        _causes = []
        if _dec.underpowered:
            _causes.append(f"underpowered ({_dec.usable} usable pairs "
                           f"of {len(priv)})")
        if _seed_undecidable:
            _causes.append("the seed arm was undecidable (outage)")
        print(f"A/B gate ABORTED: {' and '.join(_causes)} — nothing "
              f"decidable was measured; live descriptions stand "
              f"untouched. Mine more fixtures, or re-run when the "
              f"upstream is stable.")
    elif _cleared_margin and not _significant:
        print(f"{gate_contract.GATE_REJECTED_MARKER}: "
              f"the candidate cleared the margin "
              f"(paired delta {_dec.paired_delta:+.4f}, bar "
              f"{args.min_delta}) but the "
              f"discordant replays do not support it (McNemar "
              f"p={_p_str}, bar {ab_eval.SHIP_ALPHA}, {_cand_wins} "
              f"candidate / {_inc_wins} incumbent). This is an "
              f"UNDERPOWERED verdict on {_dec.usable} usable pairs "
              f"of {len(priv)}, not a measured "
              f"loss — mine more fixtures, or override deliberately "
              f"with --allow-insignificant-ship.")
    else:
        print(f"{gate_contract.GATE_REJECTED_MARKER} — live "
              f"descriptions stand.")
    return (0 if ships
            else 3 if _no_candidate
            else 2 if (_dec.underpowered or _seed_undecidable)
            else 1)


if __name__ == "__main__":
    raise SystemExit(main())
