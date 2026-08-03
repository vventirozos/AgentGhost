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
to MAX_OPT_ITERATIONS; the miner's supply gate is re-checked here
(--force-supply to override, e.g. for a smoke run). Components = the
top-N tools by public-fixture count. Ships per-tool artifacts
(`tool_description.<name>.json`) consumed by the registry read-site
(`_apply_tuned_descriptions`, validator + aggregate inflation guard);
deploy = agent restart.

Usage:
    PYTHONPATH=src python scripts/optimize_tool_descriptions.py \
        --fixtures $GHOST_HOME/system/optim/tool_choice_fixtures.jsonl \
        --upstream-url http://127.0.0.1:8088 --max-iterations 6 \
        --run-dir <state dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                           candidate: Dict[str, str]) -> List[Dict[str, Any]]:
        out = []
        for t in tools:
            f = t.get("function") or {}
            name = f.get("name")
            if name in candidate and registry_mod._validate_tool_description(
                    name, self.baselines.get(name, ""), candidate[name]):
                t = {**t, "function": {**f,
                                       "description": candidate[name].strip()}}
            out.append(t)
        return out

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
            replay = {
                "messages": payload["messages"],
                "tools": self._swap_descriptions(payload["tools"], candidate),
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
                if traj.get("err"):
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
    ap.add_argument("--min-fixtures", type=int, default=DEFAULT_MIN_FIXTURES)
    ap.add_argument("--force-supply", action="store_true",
                    help="run below the supply gate (smoke runs only)")
    ap.add_argument("--smoke", action="store_true",
                    help="evaluate the incumbent on private fixtures and "
                         "exit — de-risks the replay path without an "
                         "optimization run")
    ap.add_argument("--run-dir", default="")
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
    if not pub or not priv:
        print("degenerate public/private fixture split", file=sys.stderr)
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
    if args.smoke:
        for t in inc_eval.trajectories[:8]:
            print(f"  {t['truth']:<18} -> {str(t['picked']):<18} "
                  f"score={t['score']} {t.get('err', '')}")
        return 0

    iterations = min(args.max_iterations, MAX_OPT_ITERATIONS)
    import gepa
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=pub,
        adapter=adapter,
        reflection_lm=_make_reflection_lm(args.upstream_url),
        max_metric_calls=iterations * len(pub),
        display_progress_bar=True,
        seed=0,
        run_dir=args.run_dir or None,
    )
    best = dict(result.best_candidate)

    cand_eval = adapter.evaluate(priv, best, capture_traces=True)
    cand_acc = sum(t["score"] for t in cand_eval.trajectories) / len(priv)
    delta = cand_acc - inc_acc
    by_tool = _by_tool(best)
    valid = all(registry_mod._validate_tool_description(
        n, baselines.get(n, ""), t) for n, t in by_tool.items())
    ships = valid and delta > args.min_delta
    print(f"A/B (PRIVATE fixtures, n={len(priv)}): incumbent={inc_acc:.3f} "
          f"candidate={cand_acc:.3f} delta={delta:+.3f} valid={valid} "
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
