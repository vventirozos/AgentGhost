"""Router classifier trainer.

Mirrors ``prm/trainer.py``: turns the trajectory log into labeled training
data and fits the ``ComplexityClassifier`` so the router stops shipping
untrained (an untrained classifier escalates EVERY request to the full swarm,
which is exactly the cost the router exists to avoid).

Bail floors are essential here because ``ComplexityClassifier.fit`` *raises*
(rather than bailing gracefully) on too-few-samples or single-class data, and
``label_trajectories`` skews toward "hard" (every failed trajectory labels
hard). So we gate on a minimum labeled count AND require both classes present
before calling ``fit``.
"""

from __future__ import annotations

import itertools
import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

# §4AA held-out split. Fixed seed: a gate whose verdict wobbles between
# runs on the same corpus is not a gate.
_GATE_SPLIT_SEED = 7
_GATE_TRAIN_FRACTION = 0.7

# The corpus floor is DERIVED from the gate, not chosen separately. The gate
# needs `_GATE_MIN_HELDOUT` held-out samples to mean anything; at a 70/30
# split that implies this many labelled trajectories. Keeping these as two
# independent numbers is how you get a trainer that looks like it should run
# at 20 samples and silently never deploys — the same "mechanism that appears
# to work" defect this codebase keeps finding. One number, derived.
def _gate_min_trajectories() -> int:
    from .model import ComplexityClassifier
    import math
    return int(math.ceil(ComplexityClassifier._GATE_MIN_HELDOUT
                         / max(1e-9, 1.0 - _GATE_TRAIN_FRACTION)))

logger = logging.getLogger("GhostAgent")


# Startup bootstrap-train gate (mirrors the idle-retrain floors, but
# higher: a one-time boot train should only fire when there's enough
# real signal to beat the safe escalate-all pass-through). Below this many
# LABELED, multi-class samples we stay pass-through and wait for an idle
# retrain to produce a checkpoint later.
BOOTSTRAP_MIN_SAMPLES = 50  # superseded by the gate floor when lower
# Cap the raw trajectory stream read at boot so a giant log can't stall
# startup. Labeling drops the ambiguous middle, so we read generously
# more than BOOTSTRAP_MIN_SAMPLES before the cap bites.
BOOTSTRAP_MAX_TRAJECTORIES = 20000


@dataclass
class RouterTrainerReport:
    fit_succeeded: bool = False
    bail_reason: str = ""
    n_samples: int = 0
    easy: int = 0
    hard: int = 0
    # §4BF 1c: bench rows that augmented the TRAIN side (equal-mass
    # capped). Counted apart so the operator can see WHY a fit changed —
    # an invisible bench mass was how the R1 review's repro drove the
    # gate to permanent escalate-all without a trace in this summary.
    n_bench: int = 0
    # §4BQ flip (vi): which REPRESENTATION this fit actually used. Shown
    # in the summary because a silent fall back to lexical-only (embedder
    # not wired, wrong-width model) would otherwise be indistinguishable
    # from a successful flip in every log line the operator reads.
    uses_embeddings: bool = False

    def summary(self) -> str:
        if not self.fit_succeeded:
            return f"bailed: {self.bail_reason}"
        return (f"fit on {self.n_samples} samples "
                f"(easy={self.easy}, hard={self.hard})"
                + (f" +{self.n_bench} bench" if self.n_bench else "")
                + (" [lexical+embedding]" if self.uses_embeddings
                   else " [lexical only]"))


def _looks_path(save_path):
    """Ledger for THIS checkpoint, keyed by its filename.

    Keyed by directory, a second `--router-model experiment.json` in the
    same directory spent the production checkpoint's look budget — and
    `--no-memory` ablation runs resolve the REAL memory dir, so a control
    arm could consume it too. §4BQ protected the checkpoint from exactly
    that and left the ledger unprotected.
    """
    if save_path is None:
        return None
    p = Path(save_path)
    return p.parent / f"{p.stem}.gate_looks.json"


def _gate_fingerprint(embeddings_available: bool, *,
                      confidence_threshold=None,
                      deployed_state: str = "") -> str:
    """What this look would TEST, beyond corpus size.

    Only size re-opened a look, so a changed learner, alpha, split seed or
    feature schema could not be gated for 8+ days — and §4BF's bench
    flywheel could never trigger a re-gate at all. A look is cheap when
    the QUESTION has changed; the control exists to stop re-asking the
    SAME question of nearly the same data.
    """
    from .features import FEATURE_NAMES, EMBED_DIM
    from .model import ComplexityClassifier as _C
    _d = _C()          # learner defaults, so a retune re-opens a look
    return "|".join(str(x) for x in (
        len(FEATURE_NAMES) + (EMBED_DIM if embeddings_available else 0),
        _C._GATE_ALPHA, _C._GATE_MIN_HELDOUT, _C._GATE_MAX_FALSE_EASY,
        _C._GATE_MIN_DISCORDANT_FRACTION, _C._GATE_MIN_DISCORDANT_FLOOR,
        _GATE_SPLIT_SEED, _GATE_TRAIN_FRACTION,
        # ⚠ These three were named by this docstring as the motivating
        # cases and were NOT in the hash: the OPERATING POINT (a gate input
        # and a CLI flag — measured, a tau 0.30 -> 0.75 change was
        # blocked), the LEARNER, and BENCH MASS (a /api/bench/drain fires a
        # retrain that then silently bails, so §4BF's flywheel could never
        # trigger a re-gate).
        confidence_threshold, _d.learning_rate, _d.l2, _d.epochs,
        _d.tol, _d.random_state,
        # ...and WHAT IS DEPLOYED: none / servable / broken. "The model
        # that was standing here is now unusable" is a genuinely different
        # question from the one answered last time, even on an identical
        # corpus — so recovery re-opens a look through the ordinary
        # mechanism instead of a special case that had to be exempted from
        # the very budget it was bypassing.
        deployed_state,
    ))


def _evidence_ids(pairs) -> frozenset:
    """Identity of the EVIDENCE: (trajectory id, label) for every row.

    ⚠ Keyed on the id alone this was blind to half of what the gate
    tests. The gate scores `(X_test, y_test)`; flipping 877 of 1,707
    OUTCOMES — relabelling the corpus wholesale — left the id set
    identical and the look blocked. That is a live mechanism, not a
    hypothesis: `corrections.jsonl` already overlays 350 rows across 329
    trajectories and `derive_label` branches on `traj.outcome`, and
    `labels.py` documents a past wrong-tool-name bug that mislabelled 27
    rows. A labelling-policy fix could never re-open a look.

    Including the label means a small correction overlay (1.2% today,
    overlap 0.988) still does not re-open one, while a wholesale
    relabelling does — which is the intent.

    Bench rows never appear here: they join the TRAIN side only.
    """
    out = []
    for t, lab in pairs:
        key = str(getattr(t, "id", "") or getattr(t, "user_request", ""))
        out.append(f"{key}\u0000{lab}")
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _evidence_unchanged(new_ids: frozenset, prev_ids: frozenset) -> bool:
    """THE definition of "this is the same evidence" — used by BOTH the
    decision (`RouterTrainer.run`) and the record (`_record_look`).

    ⚠ These were two separate expressions and they disagreed in both
    directions. Seventh defect in this control, same duplicated-logic
    class as the six before it:

      * FAILED OPEN: `run` treated a strict subset as "same" while the
        record used the Jaccard bar alone, so a look granted on a smaller
        corpus RESET the config set and the config just asked became
        askable again. Measured through the real entry points
        (`bootstrap_router`'s 20,000-row cap vs the uncapped idle
        retrain) with a flapping embedder: **8 looks in 8 runs** on a
        corpus that never grew — unlimited looks out of two controls that
        each work alone.
      * FAILED CLOSED: above ~1,420 labelled rows the absolute slack is
        the tighter bar, so growth in [250, 0.176N) read as "moved" to
        the decision and "same" to the record. Measured at 1,491 → 1,748
        (+257 real rows): a healthy retrain BLOCKED while the message
        claimed the corpus was "100% the same".

    One function, one answer, no window to disagree in.
    """
    if not prev_ids or not new_ids:
        return False
    # A strict subset is never new evidence: `bootstrap_router` caps its
    # read at 20,000 rows oldest-first while the idle and self-play
    # retrains read uncapped, so the two windows would otherwise
    # alternate and each truncated read would score as a change.
    if new_ids <= prev_ids:
        return True
    from .model import ComplexityClassifier as _C
    return (_jaccard(new_ids, prev_ids) >= _C._GATE_MAX_HELDOUT_OVERLAP
            and len(new_ids - prev_ids) < _C._GATE_LOOK_ABSOLUTE_SLACK)


def _last_look(path) -> tuple:
    """(held_out_ids, configs_asked) from the ledger; empty when unknown."""
    try:
        raw = json.loads(Path(path).read_text())
        return (frozenset(raw.get("corpus_ids") or ()),
                {str(f) for f in (raw.get("configs") or ())})
    except Exception:  # noqa: BLE001
        return frozenset(), set()


def _record_look(path, corpus_ids: frozenset, config_fp: str,
                 prev_ids: frozenset = frozenset()) -> None:
    """Record a spent look. Never fails a fit — but says so LOUDLY.

    A silent failure disables the whole control: the ledger then reads
    empty forever and every look is waved through. Measured with an
    unwritable ledger, 10/10 looks were permitted with one DEBUG line as
    the only trace. Written atomically (tmp + replace) so a crash
    mid-write cannot truncate it into a free look.
    """
    if path is None:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _, prev_fps = _last_look(p)
        # Configs accumulate while the EVIDENCE is unchanged, and reset
        # when it moves. Keying that reset to corpus size instead meant a
        # single added turn cleared the set, so any alternating condition
        # bought a look every run.
        same = _evidence_unchanged(corpus_ids, prev_ids)
        fps = (prev_fps if same else set()) | ({config_fp} if config_fp else set())
        # UNIQUE tmp name. A shared one made concurrent trainers rename
        # it out from under each other: measured, 8 concurrent writes lost
        # 7 and fired 6 false "control is DISABLED" alarms on a perfectly
        # writable ledger — a false positive on this feature's own
        # tripwire.
        import os as _os, uuid as _uuid
        tmp = p.with_suffix(f"{p.suffix}.{_os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps({
            "corpus_ids": sorted(corpus_ids),
            "configs": sorted(fps),
        }))
        tmp.replace(p)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "router gate-look ledger UNWRITABLE (%s: %s) — the "
            "multiple-looks control is DISABLED until this is fixed; the "
            "gate will re-test on every corpus change", type(e).__name__, e)



def _gate_looks_enabled() -> bool:
    """Operator kill switch for the multiple-looks control.

    Exists because the control can DELAY a deploy, and a review found
    states where it delayed recovery for days with no affordance to clear
    it. A guard that can wedge a subsystem needs an off switch.
    """
    return os.environ.get("GHOST_ROUTER_GATE_LOOKS", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _deployed_model_is_recoverably_broken(path) -> bool:
    """A checkpoint EXISTS but this process cannot serve it.

    Corrupt JSON, a schema/identity rejection (a `GHOST_EMBED_MODEL`
    migration), a model that fails its own held-out gate, or one needing
    an embedder this process lacks. `main.py` documents the boot path as
    "retrain over the bad checkpoint".

    Consumed by `_gate_fingerprint` as `deployed_state`, so becoming
    unservable reads as a NEW QUESTION and re-opens exactly one look —
    rather than as an unconditional bypass, which two designs ago handed
    out a look on every other run whenever the embedder flapped. The
    bounded form is only safe because the config set now resets when the
    evidence moves; check that before weakening either.
    """
    try:
        p = Path(path)
        if not p.exists():
            return False               # nothing to recover; not this case
        from .model import ComplexityClassifier
        clf = ComplexityClassifier.load(p)
        if not clf.looks_sane():
            return True
        # ...and a checkpoint that LOADS but this process cannot SERVE is
        # equally "nothing deployed". `load()` compares the recorded
        # embedder NAME, not whether an embedder is actually available, so
        # a 402-dim checkpoint on a process with no embedder passed as
        # healthy while `main.py` refused to restore it — the router was
        # escalate-all and the control still blocked its replacement for
        # +250 turns (~8-12 days), with the bail claiming "the deployed
        # model stays". Reachable from `--no-memory`, an evicted model, a
        # vector-store init failure, or a batch-scale embed failure.
        from .embedding import probe_router_embedder
        if clf.uses_embeddings_ and not probe_router_embedder().available:
            return True
        return False
    except Exception:  # noqa: BLE001 — unloadable IS recoverably broken
        return True



def _checkpoint_uses_embeddings(path) -> bool:
    """Does the checkpoint ALREADY on disk carry the richer representation?

    Read from `feature_names` (authoritative — the same field the loader
    matches on), never from the diagnostic `uses_embeddings` flag. Any
    read failure returns False: an unreadable or absent file is not
    something worth protecting, and the caller's alternative is refusing
    to ever write one.
    """
    try:
        from .features import FEATURE_NAMES
        raw = json.loads(Path(path).read_text())
        if len(raw.get("feature_names") or ()) <= len(FEATURE_NAMES):
            return False
        # It must also LOAD. A 402-name checkpoint that the loader rejects
        # — e.g. one trained under a different GHOST_EMBED_MODEL, which is
        # a supported migration — is not worth protecting: protecting it
        # meant a degraded process could never overwrite it, so every boot
        # load-failed, retrained lexical in memory, and refused to write,
        # forever. That is precisely the never-converges failure this rule
        # was narrowed to avoid, resurrected for the unloadable case.
        from .model import ComplexityClassifier
        ComplexityClassifier.load(path)
        return True
    except Exception:  # noqa: BLE001
        return False


class RouterTrainer:
    """Label trajectories → extract features → fit ComplexityClassifier.

    On success ``self.classifier`` holds the fitted model (the hot-swap
    handle) and ``run()`` returns a report with ``fit_succeeded=True``.
    """

    def __init__(self, min_trajectories: int = 20, min_per_class: int = 1,
                 confidence_threshold: Optional[float] = None):
        # Require ≥ this many LABELED (non-ambiguous) trajectories before
        # training — below it the classifier would overfit noise.
        # The LIVE dispatcher threshold, so the gate scores the operating
        # point that actually ships. None => the dispatcher's own default.
        self.confidence_threshold = confidence_threshold
        self.min_trajectories = int(min_trajectories)
        self.min_per_class = int(min_per_class)
        self.classifier = None

    def run(self, trajectories: Iterable, save_path: Optional[Path] = None,
            bench_trajectories: Optional[Iterable] = None) -> RouterTrainerReport:
        """Train from real trajectories, optionally augmented with bench ones.

        ``bench_trajectories`` (§4BF Track 1c, admissibility: router =
        bench_feature): bench rows join the TRAIN side only, featurized
        with ``origin_bench=1.0`` and EQUAL-MASS capped (at most as many
        bench rows as real ones, newest first — bench labels are
        near-single-class "hard", and uncapped they outweighed a balanced
        real corpus ~10:1 in the R1 review's repro, driving the gate to
        permanent escalate-all). Every floor (min corpus, per-class, gate
        floor) and the §4AA held-out gate are computed on the REAL
        population alone — bench may teach the model, it may never be the
        evidence that deploys it, and it may never unlock a gate the real
        corpus is too thin to earn. Bench labeling runs AFTER the real
        floors so a thin-corpus bail never pays the bench read.
        """
        from .embedding import embed_texts, embeddings_enabled
        from .features import extract_features
        from .labels import label_trajectories, class_balance
        from .model import ComplexityClassifier

        report = RouterTrainerReport()
        try:
            pairs = label_trajectories(list(trajectories))
        except Exception as e:
            report.bail_reason = f"labeling failed: {e}"
            return report

        if len(pairs) < self.min_trajectories:
            report.bail_reason = (
                f"too few labeled trajectories ({len(pairs)} < {self.min_trajectories})"
            )
            return report

        y = [label for _, label in pairs]
        bal = class_balance(y)
        report.n_samples = int(bal.get("total", len(y)))
        report.easy = int(bal.get("easy", 0))
        report.hard = int(bal.get("hard", 0))
        if report.easy < self.min_per_class or report.hard < self.min_per_class:
            report.bail_reason = (
                f"single-class data (easy={report.easy}, hard={report.hard})"
            )
            return report

        # ⚠ ORDERING: this runs AFTER the too-few / single-class checks. Put
        # first, it shadowed them — a 10-sample single-class corpus reported
        # "needs >= 200 for the gate" instead of "single-class", replacing a
        # precise diagnosis with a vaguer one. The most specific reason wins.
        _floor = _gate_min_trajectories()
        if len(pairs) < _floor:
            report.bail_reason = (
                f"only {len(pairs)} labelled trajectories; the held-out "
                f"deploy gate needs >= {_floor} to be meaningful "
                f"(router stays escalate-all — safe, and it will train "
                f"itself once enough turns accumulate)")
            logger.info("router train: %s", report.bail_reason)
            return report

        # ── MULTIPLE-LOOKS CONTROL ──────────────────────────────────────
        # The gate is a significance test, and it is re-run on every corpus
        # change while its verdict is STICKY. Measured over 184 simulated
        # 200→400-turn growth histories with signal-free labels, re-gating
        # at every size deployed noise at least once in 10.3% of them —
        # not because any single test is miscalibrated (measured type-I is
        # 0.075%), but because ~200 looks compound.
        #
        # So: refuse to look again until the corpus is MATERIALLY larger
        # than when the deployed model was gated. Placed in the trainer so
        # all three retrain sites inherit it, like the persistence policy.
        # Which representation THIS fit will produce. PROBED, not assumed:
        # a registered-but-broken embedder makes the fit degrade to
        # lexical, and asking only whether an embedder is REGISTERED
        # predicted the wrong representation in both directions — the
        # comment here used to claim the opposite of what the code did.
        # `embedding.py` grew `probe_router_embedder` for precisely this
        # ("registration is not capability"); use it.
        from .embedding import probe_router_embedder as _emb_probe
        _embs_available = bool(_emb_probe().available)
        # The later of "when the deployed model was gated" and "when the
        # gate last LOOKED" — a rejected look still consumed a test.
        # ── MULTIPLE-LOOKS CONTROL ──────────────────────────────────────
        # ONE QUESTION: has the HELD-OUT EVIDENCE materially changed?
        #
        # ⚠ This replaces five successive designs, each of which was
        # measured INERT in the live configuration while its comment said
        # otherwise. They failed the same way every time: each keyed off a
        # PROXY for "new evidence" (deployed corpus size, a representation
        # flag, a config fingerprint, bench mass, deployability), and every
        # proxy turned out to have a channel that changes without the
        # evidence changing —
        #   * bench rows are TRAIN-only, so adding one re-opened a look
        #     while the held-out split was untouched: 101 of 201 looks;
        #   * a flapping embedder flipped two fingerprint fields per run:
        #     10/10 looks on a corpus growing 1 turn at a time;
        #   * an orphan-ledger bypass latched and gave 201/201.
        # Each proxy also needed an exemption for recovery, and each
        # exemption became the next bypass.
        #
        # The gate evaluates ONE thing: the held-out split. So that is what
        # is compared. Overlap above the threshold means the same test on
        # the same evidence — the compounding this exists to stop — no
        # matter what else changed. Bench cannot reach it (train-only), an
        # embedder flap cannot reach it, and a deleted checkpoint cannot
        # reach it. There is nothing left to alternate.
        _lp = _looks_path(save_path)
        _prev_ids, _prev_fps = _last_look(_lp)
        _ids = _evidence_ids(pairs)
        _overlap = _jaccard(_ids, _prev_ids)
        _bench_list = ([] if bench_trajectories is None
                       else list(bench_trajectories))
        # The learner/threshold config still matters — a retuned gate IS a
        # new question on the same evidence — but it is only consulted
        # while the evidence is unchanged, and the set of configs asked is
        # reset only when the evidence moves. Deliberately excludes bench
        # mass and deployability: neither changes what is being tested.
        # "The model standing here cannot be served" IS a different
        # question, so it belongs in the fingerprint — restoring recovery
        # from a corrupt / inverted / missing checkpoint, which the
        # rebuild dropped and left promised by a now-dead function's
        # docstring. It is safe HERE (it was not, two designs ago) because
        # the config set now resets only when the evidence moves: a
        # flapping embedder yields a bounded set of configs, not a look
        # per flip.
        # TWO values, and ABSENT counts as "ok". A successful look moves
        # absent -> present, and treating that as a new question hands out
        # a free look after every single deploy (measured again here as
        # "1 look bought with bench rows alone"). Only a checkpoint that
        # EXISTS and cannot be served is a different question.
        _ckpt_present = save_path is not None and Path(save_path).exists()
        _servable = not (_ckpt_present
                         and _deployed_model_is_recoverably_broken(save_path))
        _fp = _gate_fingerprint(
            _embs_available, confidence_threshold=self.confidence_threshold,
            deployed_state=("ok" if _servable else "broken"))
        from .model import ComplexityClassifier as _CC
        # ONE definition, shared with `_record_look` — see
        # `_evidence_unchanged`. Two separate expressions disagreed in
        # both directions: 8 looks in 8 runs on a corpus that never grew,
        # and a healthy retrain blocked on 257 genuinely new rows.
        _same_evidence = _evidence_unchanged(_ids, _prev_ids)
        _blocked = (bool(_prev_ids)
                    and _gate_looks_enabled()
                    and _same_evidence
                    and _fp in _prev_fps)
        if _blocked:
            # NOT `_servable`: that treats an ABSENT checkpoint as "ok",
            # which is correct for the fingerprint and wrong for the
            # operator. With nothing on disk the bail claimed "the
            # deployed model stays" at INFO — below the live pretty
            # stream's WARNING floor — and withheld the recovery hint in
            # precisely the state that needs it. Reachable today: the
            # first look's model can be REJECTED by the gate, spending a
            # look without ever writing a checkpoint.
            _anything_deployed = _ckpt_present and _servable
            report.bail_reason = (
                f"the labelled corpus is {_overlap:.0%} the same as the last "
                f"look and nothing about the gate changed — not re-running "
                f"the same test on the same evidence; "
                + ("the deployed model stays" if _anything_deployed else
                   "NOTHING IS DEPLOYED and the router stays escalate-all "
                   "(clear the look ledger or set GHOST_ROUTER_GATE_LOOKS=0 "
                   "to force a retrain)"))
            logger.log(logging.INFO if _anything_deployed else logging.WARNING,
                       "router train: %s", report.bail_reason)
            return report

        # Bench augmentation — AFTER every real-corpus floor above, so a
        # bail never reads the bench corpus, and equal-mass capped (newest
        # bench rows win; the collector iterates day files in date order).
        bench_pairs = []
        try:
            if _bench_list:
                bench_pairs = label_trajectories(_bench_list)
                # Equal mass vs the REAL TRAIN SIDE (70% of pairs), since
                # that is the population bench actually joins.
                _cap = max(1, int(len(pairs) * _GATE_TRAIN_FRACTION))
                bench_pairs = bench_pairs[-_cap:]
                report.n_bench = len(bench_pairs)
        except Exception as e:  # noqa: BLE001 — bench is additive, never fatal
            logger.debug("router train: bench labeling skipped: %s", e)
            bench_pairs = []

        # We are about to spend a look; record it BEFORE the fit so a
        # crash mid-fit cannot buy a free retest.
        _record_look(_lp, _ids, _fp, prev_ids=_prev_ids)

        try:
            texts = [str(getattr(t, "user_request", "") or "") for t, _ in pairs]
            bench_texts = [str(getattr(t, "user_request", "") or "")
                           for t, _ in bench_pairs]

            # §4BQ flip (vi). ALL-OR-NOTHING across BOTH populations: a
            # partial result would stack 402-wide real rows on 18-wide
            # bench rows into one design matrix. Embedding failure is
            # never fatal — it degrades to the pre-flip lexical fit,
            # which the same held-out gate still has to pass.
            # No kill-switch check here: `embed_texts` returns None when
            # GHOST_ROUTER_EMBED is off, so the switch has ONE definition
            # and cannot be honoured by the trainer while the serving path
            # ignores it (which is exactly what happened before the switch
            # was made to gate serving too).
            embs = bench_embs = None
            embs = embed_texts(texts)
            if embs is not None and bench_texts:
                bench_embs = embed_texts(bench_texts)
                if bench_embs is None:
                    logger.warning(
                        "router train: bench embedding failed — "
                        "falling back to a lexical-only fit")
                    embs = None
            report.uses_embeddings = embs is not None

            X = [extract_features(txt,
                                  embedding=(embs[i] if embs else None))
                 for i, txt in enumerate(texts)]
            X_bench = [extract_features(txt, origin_bench=True,
                                        embedding=(bench_embs[i]
                                                   if bench_embs else None))
                       for i, txt in enumerate(bench_texts)]
            y_bench = [label for _, label in bench_pairs]
            # §4AA HELD-OUT SPLIT. The gate below asks an EMPIRICAL question
            # ("does this beat doing nothing on data it never saw?"), which
            # requires data the fit never saw. Shuffled with a FIXED seed so
            # the same corpus always yields the same verdict — a gate whose
            # answer wobbles run to run is not a gate.
            #
            # The VALIDATED model is the one deployed: we fit on the train
            # split and ship that, rather than refitting on 100% afterwards.
            # Refitting would ship a model whose evidence describes a
            # *different* model — the same "measured X, shipped Y" gap this
            # project keeps finding elsewhere. The cost is the held-out
            # share of the data; with ~1350 labelled trajectories that is a
            # trade worth making for an honest gate.
            import random as _random
            order = list(range(len(y)))
            _random.Random(_GATE_SPLIT_SEED).shuffle(order)
            cut = int(len(order) * _GATE_TRAIN_FRACTION)
            tr_i, te_i = order[:cut], order[cut:]
            clf = ComplexityClassifier()
            # Bench rows augment the fit AFTER the real split is fixed, so
            # the held-out set is untouched real data and the same corpus
            # (with or without banks on disk) yields the same te_i.
            clf.fit([X[i] for i in tr_i] + X_bench,
                    [y[i] for i in tr_i] + y_bench)
            gate_ev = clf.evaluate([X[i] for i in te_i], [y[i] for i in te_i],
                                   confidence_threshold=self.confidence_threshold)
            gate_ev["train_n"] = len(tr_i)
            gate_ev["bench_train_n"] = len(y_bench)
            clf.gate_report_ = gate_ev
        except Exception as e:
            report.bail_reason = f"fit failed: {e}"
            return report

        # §4O R2 MAJOR-1: reject an INVERTED model AT THE SOURCE — before
        # saving and before it becomes self.classifier. A model with net-
        # negative technical/coding weights routes the hardest requests to
        # "easy" (the n_steps-counts-history bug produced exactly this).
        # Guarding only the load + hot-swap call sites left the BOOTSTRAP
        # install (main.py) and the SAVE unguarded, so a restart retrained
        # from the frozen inverted corpus, installed it ungated, and
        # re-poisoned the checkpoint every idle tick. Bailing here leaves
        # fit_succeeded=False → the idle retrain won't hot-swap and the
        # bootstrap returns (None, report) → router stays escalate-all
        # (planner runs), and no inverted model is ever persisted.
        _passed, _why = ComplexityClassifier.gate_verdict(clf.gate_report_)
        if not clf.is_finite() or not _passed:
            report.bail_reason = (
                f"model REJECTED by the held-out gate ({_why}) — not "
                f"installing or saving; router stays escalate-all")
            logger.warning("router train: %s", report.bail_reason)
            return report
        logger.info("router train: held-out gate PASSED — %s", _why)

        # §4BQ PERSISTENCE POLICY, enforced HERE rather than at the call
        # sites. Three places train the router (boot bootstrap, the idle
        # retrain, the self-play in-loop refit) and the first version of
        # this guard was applied to exactly one of them — the wrapper-split
        # defect class this project keeps rediscovering. A policy that must
        # hold for every trainer belongs inside the trainer.
        #
        # A DEGRADED fit — embeddings wanted but unobtainable (no vector
        # store under `--no-memory`, an evicted model, a batch-scale
        # failure) — must not overwrite the shared checkpoint with a
        # lexical model that a healthy process would then load. An
        # explicit GHOST_ROUTER_EMBED=0 is different: that is operator
        # intent, and its lexical model IS the one to keep.
        # Targeted at the ACTUAL harm: replacing a RICHER checkpoint with a
        # poorer one. A blanket "degraded runs never persist" also stopped a
        # box that legitimately has no vector store from ever writing a
        # checkpoint at all — it would retrain from scratch every boot
        # forever. (Caught by five existing tests, which were right.)
        _degraded = embeddings_enabled() and not report.uses_embeddings
        if save_path is not None and _degraded and _checkpoint_uses_embeddings(save_path):
            logger.warning(
                "router train: embeddings enabled but this fit is "
                "LEXICAL-ONLY (embedder unavailable at corpus scale?) and "
                "%s holds an EMBEDDING model — using this fit in memory and "
                "leaving the checkpoint untouched", save_path)
        elif save_path is not None:
            try:
                clf.save(save_path)
                # Stamp "a model HAS deployed here" only once it actually
                # has. It was recorded before the fit from "does a
                # checkpoint exist right now", so the first ever look wrote
                # False and the flag never became true — which left the
                # ledger blocking a retrain after the operator's documented
                # fix (delete the bad checkpoint) with no checkpoint in
                # existence.
            except Exception as e:
                logger.warning("router classifier save failed: %s", e)

        self.classifier = clf
        report.fit_succeeded = True
        return report


def bootstrap_router(
    trajectories: Iterable,
    *,
    save_path: Optional[Path] = None,
    min_samples: int = BOOTSTRAP_MIN_SAMPLES,
    max_trajectories: int = BOOTSTRAP_MAX_TRAJECTORIES,
    confidence_threshold: Optional[float] = None,
    bench_trajectories: Optional[Iterable] = None,
) -> Tuple[Optional["ComplexityClassifier"], RouterTrainerReport]:  # noqa: F821
    """One-time startup bootstrap-train from the trajectory log.

    Why this exists: the router ships UNTRAINED, and a trained model is
    otherwise only ever produced by an IDLE retrain (needs a long-lived idle
    process). A busy server or a benchmark never idles, so the dispatcher
    stays escalate-all forever. This trains ONCE at boot from the existing
    trajectory log so the router starts routing immediately when enough
    labeled, multi-class data already exists.

    Reuses RouterTrainer (same labeling + feature extraction + fit as the
    idle path); only difference is a higher min-sample floor and a cap on
    how much of the log we read, so a huge log can't stall startup.

    NEVER raises — any failure (malformed log, fit divergence, IO error) is
    logged and returns ``(None, report)`` so the caller falls back to the
    safe escalate-all pass-through. Returns ``(classifier, report)``; the
    classifier is None whenever training bailed.
    """
    report = RouterTrainerReport()
    try:
        # Cap the raw stream so a giant log doesn't stall boot. islice is
        # lazy, so we never materialise the whole log.
        capped = itertools.islice(trajectories, max(0, int(max_trajectories)))
        # Bench augmentation rides the same cap so a huge bench corpus
        # can't stall boot either (§4BF 1c; floors/gate stay real-only —
        # see RouterTrainer.run).
        bench_capped = (itertools.islice(bench_trajectories,
                                         max(0, int(max_trajectories)))
                        if bench_trajectories is not None else None)
        trainer = RouterTrainer(min_trajectories=int(min_samples), min_per_class=1,
                                confidence_threshold=confidence_threshold)
        report = trainer.run(trajectories=capped, save_path=save_path,
                             bench_trajectories=bench_capped)
        if report.fit_succeeded and trainer.classifier is not None:
            return trainer.classifier, report
        return None, report
    except Exception as e:  # never crash boot
        logger.warning("router bootstrap-train failed (staying pass-through): %s", e)
        report.fit_succeeded = False
        report.bail_reason = f"bootstrap exception: {e}"
        return None, report
