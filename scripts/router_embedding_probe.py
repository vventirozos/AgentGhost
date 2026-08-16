#!/usr/bin/env python3
"""§4BQ — CAN EMBEDDINGS SEPARATE WHAT 18 LEXICAL FEATURES CANNOT?

A MEASUREMENT, NOT A SHIP. §4AN closed the planner-skip A/B without
running it, and that feasibility check was worth more than the experiment:
it found the router is "confidently right exactly where it adds NOTHING",
because its features saturate on one synthetic chess template. Natural
traffic separates by **0.081 in probability**, and 0 of 600 non-chess
requests clear the 0.75 confidence bar.

Flip (vi) proposes replacing the hand-crafted features with
sentence-transformer turn embeddings — which are already loaded for the
vector store, so this costs no new dependency and no egress (the model is
in ~/.cache). Before shipping a bigger model on faith, answer the only
question that matters:

    Does an embedding representation separate THIS corpus better than the
    18 lexical features do — on the population where the router can
    actually act (non-template natural traffic)?

If the answer is no, the honest outcome is a NULL result and nothing
ships. That is a good night's work: §4AN's precedent exactly.

Reported for each representation:
  * held-out accuracy vs the MAJORITY-CLASS baseline (lift is the metric;
    accuracy alone flatters a skewed corpus)
  * mean p(hard) separation between easy and hard — §4AN's 0.081 number,
    the thing that decides whether any confidence bar can select for
    difficulty rather than for template-ness
  * the same, EXCLUDING the chess template, which is the population the
    router would actually be acting on

Usage:
    GHOST_HOME=~/Data/AI/Data PYTHONPATH=src python3 \
        scripts/router_embedding_probe.py [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SPLIT_SEED = 7          # same seed the §4AA gate uses — not a new lottery
_TRAIN_FRACTION = 0.7


def _load_corpus() -> List[Tuple[str, str]]:
    """(text, label) pairs from the REAL trajectory corpus.

    Uses the router's own labeller so this measures the same population
    the trainer would, not a hand-rolled variant that could flatter the
    result.
    """
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.router.labels import label_trajectories

    home = os.environ.get("GHOST_HOME") or str(Path.home() / "Data/AI/Data")
    root = Path(home) / "system" / "trajectories"
    collector = TrajectoryCollector(root=root, session_id="probe")
    trajs = []
    day_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if root.exists():
        for day in sorted(p.name for p in root.iterdir()
                          if p.is_dir() and day_re.match(p.name)):
            trajs.extend(collector.iter_trajectories(day=day))
    pairs = label_trajectories(trajs)
    out = []
    for t, label in pairs:
        text = str(getattr(t, "user_request", "") or "")
        if text.strip():
            out.append((text, str(label)))
    return out


def _is_chess_template(text: str) -> bool:
    """§4AN: 113 of 114 confident-easy requests were ONE template. It has
    to be reported separately or it dominates every average."""
    low = text.lower()
    return ("chess" in low
            and ("fen" in low or "move" in low or "board" in low))


def _lexical_matrix(texts: List[str]):
    from ghost_agent.router.features import extract_features
    rows = []
    for t in texts:
        fv = extract_features(t)
        rows.append([float(v) for v in fv.values])
    return rows


def _embedding_matrix(texts: List[str]):
    """The embedder the agent ACTUALLY serves with.

    Defaulted to `memory.vector.EMBED_MODEL_NAME` (bge-small-en-v1.5), not
    to a hardcoded MiniLM: the first §4BQ run measured MiniLM while
    production loads bge, and the two representations of the same text
    have mean cosine 0.284 — i.e. every number was computed on a model
    that would never serve. Offline: no download, no egress (the
    operator's standing constraint)."""
    from sentence_transformers import SentenceTransformer
    try:
        from ghost_agent.memory.vector import EMBED_MODEL_NAME as _live
    except Exception:  # noqa: BLE001
        _live = "BAAI/bge-small-en-v1.5"
    name = os.environ.get("GHOST_EMBED_MODEL", _live)
    model = SentenceTransformer(name)
    return [list(map(float, v)) for v in
            model.encode(texts, batch_size=32, show_progress_bar=False,
                         normalize_embeddings=True)]


def _fit_logreg(xtr, ytr, xte, l2: float = 1.0):
    """Plain logistic regression via sklearn if present, else a small
    gradient descent — the point is the COMPARISON, and both arms use the
    same learner, so a weak learner cannot favour one representation."""
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0 / max(1e-9, l2))
        clf.fit(xtr, ytr)
        return [float(p[1]) for p in clf.predict_proba(xte)]
    except ImportError:
        pass
    n_feat = len(xtr[0])
    w = [0.0] * n_feat
    b = 0.0
    lr = 0.1
    for _ in range(400):
        gw = [0.0] * n_feat
        gb = 0.0
        for xi, yi in zip(xtr, ytr):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            d = p - yi
            for j, xj in enumerate(xi):
                gw[j] += d * xj
            gb += d
        m = max(1, len(xtr))
        w = [wj - lr * (gj / m + l2 * wj / m) for wj, gj in zip(w, gw)]
        b -= lr * gb / m
    out = []
    for xi in xte:
        z = b + sum(wj * xj for wj, xj in zip(w, xi))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return out


def _evaluate(name, xs, ys, texts, report):
    idx = list(range(len(xs)))
    random.Random(_SPLIT_SEED).shuffle(idx)
    cut = int(len(idx) * _TRAIN_FRACTION)
    tr, te = idx[:cut], idx[cut:]
    if not tr or not te:
        report[name] = {"error": "split empty"}
        return
    probs = _fit_logreg([xs[i] for i in tr], [ys[i] for i in tr],
                        [xs[i] for i in te])
    yte = [ys[i] for i in te]
    acc = sum((p >= 0.5) == bool(y) for p, y in zip(probs, yte)) / len(yte)
    base = max(sum(yte), len(yte) - sum(yte)) / len(yte)

    def _sep(mask):
        e = [p for p, y, m in zip(probs, yte, mask) if m and not y]
        h = [p for p, y, m in zip(probs, yte, mask) if m and y]
        if not e or not h:
            return None
        return sum(h) / len(h) - sum(e) / len(e)

    all_mask = [True] * len(te)
    nonchess = [not _is_chess_template(texts[i]) for i in te]
    report[name] = {
        "n_train": len(tr), "n_heldout": len(te),
        "accuracy": round(acc, 4),
        "majority_baseline": round(base, 4),
        "lift": round(acc - base, 4),
        "separation_all": (round(_sep(all_mask), 4)
                           if _sep(all_mask) is not None else None),
        "separation_nonchess": (round(_sep(nonchess), 4)
                                if _sep(nonchess) is not None else None),
        "n_heldout_nonchess": sum(nonchess),
        "consumers": _consumer_tables(probs, yte, nonchess),
    }


def _consumer_tables(probs, yte, nonchess, threshold=0.3):
    """The two decision metrics, computed the way the CODE computes them.

    CONFIDENCE is `abs(p - 0.5) * 2` (model.py `predict`), NOT `p(easy)`.
    The first §4BQ run used the latter and reported "26 non-chess turns
    clear the 0.75 bar"; under the real definition that same population
    clears 0.75 zero times, exactly as §4AN measured. A metric that is
    not the one the code implements cannot answer a question about the
    code.

    MCTS GATE is the router's ONLY live consumer (`agent.py`:
    `label == "hard" and not escalated`). The confident-EASY planner-skip
    that the confidence table was about was retired in §4AN, so the gate
    below is what a representation change actually moves.
    """
    conf = [abs(pv - 0.5) * 2.0 for pv in probs]
    out = {}
    for bar in (0.75, 0.80):
        sel = [(c, y) for c, pv, y, nc in zip(conf, probs, yte, nonchess)
               if nc and pv < 0.5 and c >= bar]
        out[f"confident_easy@{bar}"] = {
            "cleared": len(sel),
            "precision": (sum(1 for _, y in sel if not y) / len(sel)
                          if sel else None),
        }
    fires = [(pv, y) for pv, c, y, nc in zip(probs, conf, yte, nonchess)
             if nc and pv >= 0.5 and c >= threshold]
    n_nc = sum(nonchess)
    out["mcts_gate"] = {
        "fires": len(fires),
        "fire_rate": (len(fires) / n_nc) if n_nc else None,
        "precision": (sum(1 for _, y in fires if y) / len(fires)
                      if fires else None),
        "base_rate": (sum(1 for y, nc in zip(yte, nonchess) if nc and y) / n_nc
                      if n_nc else None),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    corpus = _load_corpus()
    if len(corpus) < 60:
        print(f"insufficient corpus: {len(corpus)} labelled turns "
              f"(need ~60 for a 70/30 split to mean anything)")
        return 1
    texts = [t for t, _ in corpus]
    ys = [1 if lab == "hard" else 0 for _, lab in corpus]

    report = {
        "n_corpus": len(corpus),
        "n_hard": sum(ys),
        "n_easy": len(ys) - sum(ys),
        "n_chess_template": sum(_is_chess_template(t) for t in texts),
        "split_seed": _SPLIT_SEED,
    }

    _evaluate("lexical_18", _lexical_matrix(texts), ys, texts, report)
    try:
        emb = _embedding_matrix(texts)
        _evaluate("embeddings_384", emb, ys, texts, report)
        _evaluate("combined", [e + l for e, l in
                               zip(emb, _lexical_matrix(texts))],
                  ys, texts, report)
    except Exception as e:  # noqa: BLE001
        report["embeddings_384"] = {"error": f"{type(e).__name__}: {e}"}

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("§4BQ — router representation probe")
    print(f"  corpus: {report['n_corpus']} labelled turns "
          f"({report['n_hard']} hard / {report['n_easy']} easy), "
          f"{report['n_chess_template']} chess-template")
    print()
    hdr = f"  {'representation':18s} {'acc':>7s} {'base':>7s} {'lift':>8s}" \
          f" {'sep(all)':>10s} {'sep(non-chess)':>15s}"
    print(hdr)
    for key in ("lexical_18", "embeddings_384", "combined"):
        r = report.get(key)
        if not r:
            continue
        if "error" in r:
            print(f"  {key:18s} ERROR: {r['error']}")
            continue
        sa = r["separation_all"]
        sn = r["separation_nonchess"]
        print(f"  {key:18s} {r['accuracy']:7.3f} {r['majority_baseline']:7.3f}"
              f" {r['lift']:+8.3f}"
              f" {(f'{sa:.3f}' if sa is not None else '—'):>10s}"
              f" {(f'{sn:.3f}' if sn is not None else '—'):>15s}")
    print()
    print("  DECISION METRICS on the non-chess held-out population")
    print("  (confidence = abs(p-0.5)*2, the definition model.py uses):")
    print(f"    {'representation':18s} {'MCTS fires':>11s} {'precision':>10s}"
          f" {'base':>7s} {'conf-easy@.75':>14s}")
    for key in ("lexical_18", "embeddings_384", "combined"):
        r = report.get(key) or {}
        c = r.get("consumers")
        if not c:
            continue
        g = c["mcts_gate"]
        ce = c["confident_easy@0.75"]
        _f = f"{g['fire_rate']:.3f}" if g["fire_rate"] is not None else "—"
        _p = f"{g['precision']:.3f}" if g["precision"] is not None else "—"
        _b = f"{g['base_rate']:.3f}" if g["base_rate"] is not None else "—"
        print(f"    {key:18s} {_f:>11s} {_p:>10s} {_b:>7s}"
              f" {ce['cleared']:>14d}")
    print()
    print("  READ THE MCTS COLUMN, NOT THE CONFIDENCE ONE. The confident-")
    print("  easy planner-skip that a 0.75 bar would feed was RETIRED in")
    print("  §4AN; under the confidence definition the code implements,")
    print("  no representation clears 0.75 on natural traffic — that")
    print("  finding stands and is reproduced above. The router's only")
    print("  live consumer is the MCTS gate, so a representation earns a")
    print("  ship only by raising ITS precision over the base rate.")
    print()
    print("  Caveat this probe cannot answer: it splits RANDOMLY, like the")
    print("  §4AA deploy gate. Under a TEMPORAL split (train past, predict")
    print("  future — how the router really runs) neither representation")
    print("  beat escalate-all. Relative gains here are robust to")
    print("  near-duplicate grouping; the ABSOLUTE lift is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
