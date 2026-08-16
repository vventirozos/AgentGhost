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
aggregate inflation guard); deploy = agent restart.

⚠ SUPPLY GATE — this is NOT the miner's gate, despite the shared flag name
and default. `scripts/mine_tool_fixtures.py --min-fixtures 200` counts ALL
fixtures; this counts POSITIVES only (negatives cannot score a tool-choice
replay). Measured 2026-08-04 on the real mine: 183 fixtures / 65 positives,
i.e. the miner reports "ready" and overwrites the live fixture pool at
~71 positives while this runner still refuses. Read the miner's `Labels:`
line, not its exit code. `--force-supply` overrides here (smoke runs only).

Usage:
    PYTHONPATH=src python scripts/optimize_tool_descriptions.py \
        --fixtures $GHOST_HOME/system/optim/tool_choice_fixtures.jsonl \
        --upstream-url http://127.0.0.1:8088 --max-iterations 6 \
        --run-dir <state dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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
    for c in fx.get("chosen_tools") or []:
        n = (c or {}).get("name")
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
    try:
        with open(recordings_dir / fname) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
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
        """POST the replay; return the FIRST chosen tool name (or None)."""
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except Exception:
            return None
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
            s = 1.0 if picked == truth else 0.0
            scores.append(s)
            outputs.append(picked)
            trajectories.append({"fx": fx, "picked": picked, "truth": truth,
                                 "score": s, "err": ""})
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
                if _err == "unreplayable":
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--upstream-url", default="http://127.0.0.1:8088",
                    help="MAIN model endpoint (descriptions target it)")
    ap.add_argument("--recordings", default="",
                    help="day-file dir (default $GHOST_HOME/system/llm_recordings)")
    ap.add_argument("--components", type=int, default=6,
                    help="top-N tools by public-fixture count to optimize")
    ap.add_argument("--max-iterations", type=int, default=6,
                    help=f"budget in full public evals (clamp {MAX_OPT_ITERATIONS})")
    ap.add_argument("--min-delta", type=float, default=0.02)
    ap.add_argument("--min-fixtures", type=int, default=DEFAULT_MIN_FIXTURES,
                    help="minimum POSITIVE fixtures (the miner's flag of the "
                         "same name counts ALL fixtures — see the module "
                         "docstring)")
    ap.add_argument("--force-supply", action="store_true",
                    help="run below the supply gate (smoke runs only)")
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

    ghost_home = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    recordings_dir = Path(args.recordings) if args.recordings else \
        ghost_home / "system" / "llm_recordings"

    fixtures = _load_fixtures(Path(args.fixtures))
    positives = [f for f in fixtures
                 if f.get("label") == 1.0 and _first_choice(f)]
    if len(positives) < args.min_fixtures and not args.force_supply:
        print(f"supply gate: {len(positives)} positive fixtures < "
              f"{args.min_fixtures} — wait for more recording days "
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
    _resolution = 1.0 / len(priv)
    if _resolution > args.min_delta and not args.smoke:
        _need = math.ceil(1.0 / args.min_delta)
        print(f"REFUSING TO RUN: the A/B gate cannot resolve its own "
              f"threshold. {len(priv)} private positive fixtures give a "
              f"smallest step of {_resolution:.3f}, coarser than --min-delta "
              f"{args.min_delta}. One flipped replay would decide the run. "
              f"Collect at least {_need} PRIVATE positives (~"
              f"{math.ceil(_need * len(positives) / max(1, len(priv)))} "
              f"positives at today's realised private share), or raise "
              f"--min-delta.", file=sys.stderr)
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

    adapter = _Adapter(args.upstream_url, recordings_dir, baselines)

    inc_eval = adapter.evaluate(priv, seed_candidate, capture_traces=True)
    inc_acc = sum(t["score"] for t in inc_eval.trajectories) / len(priv)
    n_err = sum(1 for t in inc_eval.trajectories if t.get("err"))
    print(f"INCUMBENT tool-choice fidelity on PRIVATE: {inc_acc:.3f} "
          f"({n_err} unreplayable)")
    if args.confusion_out:
        _dump_confusion(inc_eval.trajectories, Path(args.confusion_out))
        print(f"wrote {len(inc_eval.trajectories)} replay rows to "
              f"{args.confusion_out}")
    if args.smoke:
        for t in inc_eval.trajectories[:8]:
            print(f"  {t['truth']:<18} -> {str(t['picked']):<18} "
                  f"score={t['score']} {t.get('err', '')}")
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
    ships = valid and aggregate_ok and delta > args.min_delta
    print(f"A/B (PRIVATE fixtures, n={len(priv)}): incumbent={inc_acc:.3f} "
          f"candidate={cand_acc:.3f} delta={delta:+.3f} valid={valid} "
          f"aggregate_ok={aggregate_ok} (+{_inflation}/{_slack} chars) "
          f"ships={ships}")

    optim_dir = ghost_home / "system" / "optim"
    optim_dir.mkdir(parents=True, exist_ok=True)
    for comp, text in best.items():
        payload = json.dumps({
            "signature_name": comp,
            "baseline_instruction": seed_candidate[comp],
            "optimized_instruction": text,
            "optimizer": "GEPA-tool-desc",
            "iterations": iterations,
            "private_incumbent": round(inc_acc, 4),
            "private_candidate": round(cand_acc, 4),
        }, indent=2)
        live = optim_dir / f"{comp}.json"
        if ships:
            live.write_text(payload)
            print(f"PROMOTED {live}")
        else:
            (optim_dir / f"{comp}.json.candidate.rejected").write_text(payload)
    if not ships:
        print("A/B gate REJECTED — live descriptions stand.")
    return 0 if ships else 1


if __name__ == "__main__":
    raise SystemExit(main())
