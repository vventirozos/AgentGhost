# src/ghost_agent/eval/verify_bench.py
"""Verifier fault-injection calibration bench.

Measures the verifier's REAL catch rate instead of assuming it: take
known-good (claim, evidence, context) triples, inject a known corruption
from a typed fault library, run each variant through
``Verifier.verify_claim`` against a live judge model, and score verdicts
against the expectation the fault class defines. Clean cases measure the
false-positive rate; corrupted cases measure the true-positive (catch)
rate; evidence-degradation cases measure robustness against punishing a
correct claim for pipeline noise (the 2026-07-17 evidence-packer failure
shape).

Methodology borrowed from "Mechanisms of Introspective Awareness"
(arXiv:2603.21396): inject a KNOWN anomaly, then report detection at a
controlled false-positive rate — a detector is only meaningful with both
numbers. "Actionable" rates use the same confidence gate (>= 0.7) as the
production repair/consumption sites in agent.py, so the bench reports
what the agent would actually have acted on.

Case sources:
  - a hand-authored seed set (scripts/verify_bench_cases.jsonl), so the
    bench runs before any recordings exist;
  - GHOST_LLM_RECORD day-files: every recorded VERIFY call embeds the
    rendered claim prompt, and ``extract_cases_from_recordings`` parses
    the (claim, evidence, context) triple back out of it using the live
    template's own literal segments — real production turns become bench
    cases for free.

TWO ARMS, and every report says which one it ran (`provenance.escalation`):
  - `raw_judge` — a single-endpoint `HttpChatClient`. `_escalate_refute`
    is a no-op there, so this measures the cheap judge STANDALONE;
  - `judge+escalation` — an `EscalatingChatClient` (cheap endpoint on the
    worker route + main endpoint for `force_main`), which is the pipeline
    production actually acts on. The difference is not cosmetic: on the
    live recorded corpus the main model overturned 42 of 50 (84%) of the
    cheap judge's refutes, so a raw-arm false-alarm rate is an UPPER
    BOUND on production's. `score_trials` therefore emits `fpr_raw_judge`
    or `fpr_escalated` — never a bare `fpr` that could be mixed.

Offline by design on the data side; the only things it talks to are the
judge endpoint (the system under test) and, on the escalated arm, the
main model it escalates to. See scripts/verify_bench.py for the CLI
runner.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..core.verifier import _VERIFY_CLAIM_PROMPT, Verifier

logger = logging.getLogger("GhostAgent")

# Mirrors the production consumption gates (agent.py in-loop AUTO-REPAIR
# and the post-loop verdict gate both act only on confidence >= 0.7). A
# REFUTED below this is visible in logs but changes nothing the user sees.
ACTIONABLE_CONF = 0.7

# ── The verdict pipeline a run measured ──────────────────────────────
#
# Escalation is TWO-DIRECTIONAL since 2026-08-04, and the directions are
# independent: `_escalate_refute` re-adjudicates a cheap REFUTED (main
# model wins, verdict can flip), `_escalate_confirm` re-adjudicates a
# HIGH-STAKES cheap CONFIRMED (verdict never flips — confidence is capped
# to 0.6, below every >=0.7 consumption gate). A run where only one
# direction fires is NOT production-equivalent, so the arm names all four
# states rather than collapsing to "escalated".
# ── Making a re-bench cheap: response cache + semantic code digest ───
#
# THE PROBLEM THIS SOLVES. A full live run costs hundreds of judge calls
# plus a main-model adjudication on every REFUTED. Paying that on every
# code change is untenable, so in practice the number goes stale and gets
# quoted anyway — which is how the 2026-08-04 baseline came to be compared
# against a run on a DIFFERENT route leg.
#
# Two mechanisms, aimed at two different questions:
#
#   "did MY change move the number?"  -> replay against cached responses.
#       The judge's outputs are held FIXED, so every difference is
#       attributable to the code. Zero LLM calls, seconds not hours.
#
#   "what is the number today?"       -> a fresh, uncached run.
#       Cached replay cannot answer this and must never be reported as if
#       it had: holding the judge fixed also hides the judge's own drift
#       and sampling variance. `cache_mode` is recorded in provenance so a
#       replayed report can never be mistaken for a measured one.
#
# The cache key covers everything the endpoint sees, so a changed prompt,
# model or endpoint MISSES by construction — you cannot accidentally
# replay a stale answer to a new question.
_CACHE_MODES = ("off", "write", "read", "strict")


class ResponseCache:
    """Content-addressed cache of judge/main-model responses.

    Modes:
      off    — passthrough; nothing read, nothing written (a fresh run).
      write  — live calls, every response recorded (seeds the cache).
      read   — cache hit reused, miss falls through to a live call and is
               recorded. This is the INCREMENTAL mode: after a prompt
               edit only the genuinely-changed prompts cost anything.
      strict — replay only; a miss is an ERROR. Use this to PROVE a run
               was fully offline, so its delta is 100% attributable to
               code rather than to the model having a different day.
    """

    def __init__(self, path: Optional[str | Path], mode: str = "off"):
        if mode not in _CACHE_MODES:
            raise ValueError(f"cache mode must be one of {_CACHE_MODES}")
        self.mode = mode
        self.path = Path(path) if path else None
        self.hits = 0
        self.misses = 0
        self.writes = 0
        # Strict misses are kept SEPARATELY and their requests are dumped.
        # A strict miss is swallowed upstream (the verifier treats the
        # raised KeyError as an unparseable judge reply and returns a null
        # verdict), so without this a broken replay looks like a run with a
        # few "skipped" trials instead of a replay that did not replay.
        self.strict_misses: List[Dict[str, Any]] = []
        self._hit_mtimes: List[float] = []
        if self.path and mode != "off":
            # ⚠ A REPLAY MODE MUST NOT CREATE ITS OWN CACHE (2026-08-10).
            #
            # `mkdir(exist_ok=True)` used to run for every non-off mode, so
            # pointing read/strict at a path that does not exist produced a
            # brand-new EMPTY cache and then missed on every lookup. That is
            # indistinguishable from a cache the code digest legitimately
            # invalidated: both report "0 hits". It cost an hour and a
            # published wrong conclusion ("the cache is stale, a full live
            # re-bench is unavoidable") — the run was reading an empty
            # directory created by this line, while the real 2389-entry
            # cache sat untouched. See scripts/verify_bench.py --cache-dir.
            #
            # MISSING and EMPTY are different failures and must stay
            # distinguishable: missing = you pointed somewhere wrong (fatal
            # for a replay); empty = nothing recorded yet (legitimate, and
            # what `write` mode is for).
            if mode in ("read", "strict") and not self.path.exists():
                raise FileNotFoundError(
                    f"cache directory does not exist: {self.path.resolve()}\n"
                    f"  --cache-mode {mode} REPLAYS a cache; it will not "
                    f"create one, because an empty cache misses on every\n"
                    f"  lookup and that is indistinguishable from a stale "
                    f"one. Check the path, or seed a new cache with\n"
                    f"  --cache-mode write.")
            self.path.mkdir(parents=True, exist_ok=True)

    # PER-PROCESS NONCES THAT CARRY NO MEANING FOR THE JUDGE.
    #
    # `agent._PACKER_NONCE` is a uuid4 minted at import and embedded in the
    # evidence-truncation marker ("…[PACKER CUT#53d14ea5: 237 of 309 chars
    # shown]"). It exists for a good reason — a bare substring test is
    # forgeable by evidence the agent itself read, so the marker must be
    # unguessable — and production must keep it.
    #
    # But it means every verify prompt containing a truncated digest is
    # UNIQUE PER RUN, so those prompts could never hit the cache. Measured
    # on the first replay: 16 of 69 calls missed, exactly one per trial,
    # and before the strict-miss diagnostics existed this was invisible —
    # the run simply looked like it had a few "skipped" trials.
    #
    # The nonce is opaque to the judge: two prompts differing only in those
    # 8 hex characters are the same question. So it is NORMALISED OUT of
    # the key while the real body is still stored verbatim. Narrow by
    # construction — it matches only this exact marker shape, so it cannot
    # collapse two genuinely different prompts.
    _NONCE_PATTERNS = (
        (re.compile(r"(\[PACKER CUT)#[0-9a-f]{8}(:)"), r"\1#<nonce>\2"),
    )

    @classmethod
    def _canonical(cls, obj: Any) -> Any:
        if isinstance(obj, str):
            for rx, repl in cls._NONCE_PATTERNS:
                obj = rx.sub(repl, obj)
            return obj
        if isinstance(obj, dict):
            return {k: cls._canonical(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._canonical(v) for v in obj]
        return obj

    @classmethod
    def key(cls, base_url: str, body: Dict[str, Any]) -> str:
        """Everything that can change the ANSWER goes into the key.

        `stream` is excluded (it changes transport, not content), the body
        is canonicalised with sorted keys so dict ordering cannot produce
        two keys for one request, and per-process nonces that the judge
        cannot read are normalised away.
        """
        k = cls._canonical({kk: vv for kk, vv in body.items()
                            if kk != "stream"})
        blob = json.dumps({"url": base_url.rstrip("/"), "body": k},
                          sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _file(self, key: str) -> Path:
        assert self.path is not None
        return self.path / key[:2] / f"{key}.json"

    def get(self, key: str, base_url: str = "",
            body: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if self.mode in ("off", "write") or not self.path:
            return None
        f = self._file(key)
        if not f.exists():
            self.misses += 1
            if self.mode == "strict":
                self.strict_misses.append(
                    {"key": key, "url": base_url, "request": body})
                raise KeyError(
                    f"response cache MISS in strict mode ({key[:12]}). The "
                    "run is not reproducible offline — the prompt, model or "
                    "endpoint differs from what was recorded. Re-run with "
                    "--cache-mode read to fill the gap with live calls.")
            return None
        try:
            data = json.loads(f.read_text())
        except Exception:  # noqa: BLE001 — a corrupt entry must not kill a run
            self.misses += 1
            return None
        # ⚠ Recorded HERE, at a real hit — not at f.exists(). A corrupt
        # entry exists on disk but is counted a MISS above, and an earlier
        # version appended its mtime anyway, so `replay_age_s` could
        # describe evidence the run never actually used and
        # len(_hit_mtimes) could exceed `hits`.
        try:
            self._hit_mtimes.append(f.stat().st_mtime)
        except OSError:
            pass
        self.hits += 1
        return data.get("response")

    def put(self, key: str, base_url: str, body: Dict[str, Any],
            response: Dict[str, Any]) -> None:
        if self.mode == "off" or not self.path:
            return
        f = self._file(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        # The request is stored ALONGSIDE the response, not just its hash:
        # without it a cache is an unauditable pile of answers and there is
        # no way to check later what question produced one.
        tmp.write_text(json.dumps(
            {"url": base_url.rstrip("/"), "request": body,
             "response": response}, ensure_ascii=False, default=str))
        os.replace(tmp, f)        # atomic: concurrent trials share this dir
        self.writes += 1

    def entry_count(self) -> int:
        """How many responses are actually ON DISK, right now.

        The one number that makes a misdirected --cache-dir obvious BEFORE a
        run burns an hour: "0 entries" next to a replay mode is the whole
        bug, visible in a single line. Counting is cheap next to the run it
        guards, and `.tmp` files are half-written puts, never entries.
        """
        if not self.path or not self.path.exists():
            return 0
        try:
            return sum(1 for p in self.path.rglob("*")
                       if p.is_file() and p.suffix != ".tmp")
        except OSError:
            return 0

    def stats(self) -> Dict[str, Any]:
        """⚠ A SNAPSHOT. Call it AFTER the run, never before.

        The first version was read while building the provenance block,
        which happens before a single trial executes — so a fully replayed
        run reported `hits: 0 … measures: "live judge"`. That is precisely
        the mislabel this field exists to prevent, produced by the field
        itself. `run_bench` now re-reads it once the arms are done.
        """
        total = self.hits + self.misses
        _now = time.time()
        return {
            "mode": self.mode,
            "path": str(self.path) if self.path else None,
            # Recorded in PROVENANCE so a past run can be re-read and the
            # "0 hits" question answered — empty cache, or stale cache?
            "entries": self.entry_count(),
            "hits": self.hits, "misses": self.misses, "writes": self.writes,
            "hit_rate": round(self.hits / total, 4) if total else None,
            # A strict miss is swallowed upstream into a null verdict, so it
            # must be counted where a reader will see it.
            "strict_misses": len(self.strict_misses),
            # How OLD the replayed evidence is. A resumed run (seconds to
            # hours) is a legitimate absolute measurement; a run padded from
            # a week-old cache is describing a judge that may have moved.
            "replay_age_s": ({
                "oldest": round(_now - min(self._hit_mtimes)),
                "newest": round(_now - max(self._hit_mtimes)),
            } if self._hit_mtimes else None),
            # The honest label. A report built on cache hits measures the
            # CODE against a frozen judge; it does not re-measure the judge.
            "measures": ("not yet run" if not total and not self.writes
                         else "code-vs-frozen-judge (replayed)"
                         if self.hits and not self.misses
                         else "live judge" if not self.hits
                         else "MIXED live/replayed — NOT a clean attribution"),
        }

    def dump_misses(self, path: str | Path) -> Optional[str]:
        """Write the requests that missed, so a broken replay is diagnosable
        rather than merely reported."""
        if not self.strict_misses:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.strict_misses, indent=1,
                                ensure_ascii=False, default=str))
        return str(p)


def _strip_docstrings(tree):
    """Drop docstrings so prose edits don't invalidate a benchmark."""
    import ast
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def semantic_code_digest(paths: Iterable[str | Path]) -> str:
    """Hash what the code DOES, ignoring how it reads.

    A raw file hash would mark a bench stale on every comment edit — and
    this codebase is heavily commented, so that hash would cry wolf until
    it was ignored, which is worse than not having it. Parsing to an AST
    with docstrings stripped and positions dropped means reformatting,
    comments and docstrings are invisible while any change to logic,
    constants or control flow is not.
    """
    import ast
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        try:
            tree = _strip_docstrings(ast.parse(Path(p).read_text()))
            h.update(Path(p).name.encode("utf-8"))
            h.update(ast.dump(tree, annotate_fields=False,
                              include_attributes=False).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            # LOUD: a digest that silently skips a file it could not read
            # would report "unchanged" for a file it never looked at.
            h.update(f"UNREADABLE:{p}:{type(exc).__name__}".encode("utf-8"))
    return h.hexdigest()[:16]


def build_case_pool(seed_path: str | Path,
                    mined_path: Optional[str | Path] = None,
                    no_mined: bool = False,
                    tier: str = "all") -> List["BenchCase"]:
    """THE case pool, in ONE place. Both the bench and the staleness oracle
    call this; neither may keep its own copy.

    ⚠ WHY THIS EXISTS (2026-08-09, found the hard way). The oracle had a
    hand-written replica of the bench's pool construction. It computed 35
    cases where the bench loaded 58, because the replica got two things
    wrong that are easy to miss and impossible to notice from the outside:

      * the tier filter applies to the MINED pool ONLY — seed cases are
        always included, since they are hand-authored rather than derived
        from turns the optimizer may have trained on;
      * mined cases are DEDUPED against the seed set by (claim, evidence),
        because a mined case can be a re-recording of a seed scenario.

    A fingerprint tool whose pool disagrees with the bench's reports
    permanent false drift on `cases_sha256` — which is precisely the
    wolf-crying the oracle was built to prevent. The test that was supposed
    to catch this only exercised the `--no-mined, tier=all` path, where the
    two implementations happen to agree.
    """
    cases = load_cases_jsonl(seed_path)
    if no_mined or not mined_path or not Path(mined_path).exists():
        return cases
    extra = load_cases_jsonl(mined_path)
    have = {(c.claim.strip(), c.evidence.strip()) for c in cases}
    extra = [c for c in extra
             if (c.claim.strip(), c.evidence.strip()) not in have]
    if tier != "all":
        from ..optim.trainset import holdout_tier
        extra = [c for c in extra
                 if holdout_tier(f"vbcase:{c.case_id}",
                                 private_pct=30) == tier]
    return cases + extra


def verify_path_sources() -> Dict[str, List[str]]:
    """The two source sets whose changes invalidate a bench number.

    Split deliberately: a change to the VERIFIER moves the number because
    the system got better or worse; a change to the BENCH moves it because
    the ruler changed. Collapsing them into one digest would make those
    indistinguishable, and only one of them is a result.
    """
    root = Path(__file__).resolve().parents[1]
    return {
        # ⚠ THE VERIFY PATH IS MORE THAN verifier.py (2026-08-09). This
        # listed `core/verifier.py` alone, and the oracle duly reported
        # "NO DRIFT" on a tree whose `core/objection.py` had been changed
        # TWICE that day — both changes shipped to production, one of them
        # verdict-affecting. A fingerprint blind to a file in the system
        # under test is worse than no fingerprint: it does not merely fail
        # to warn, it actively certifies a stale baseline as current.
        #
        # `objection.py` decides UPHOLD/DISMISS/UNRESOLVED before the
        # escalation runs, so it moves verdicts directly. It is imported
        # as `from . import objection as _objection`, which the
        # import-scan below would miss — hence listed explicitly rather
        # than inferred.
        "verifier": [str(root / "core" / "verifier.py"),
                     str(root / "core" / "objection.py"),
                     # §4BF flip (i) R1 M2 — same defect shape as
                     # objection.py above: the score-token probe's
                     # payload (logprobs field choice, top_k) comes from
                     # `entropy.request_logprobs`; an edit there changes
                     # the probe distribution of a probe-armed system
                     # while the fingerprint certified NO DRIFT.
                     str(root / "core" / "entropy.py")],
        "bench": [str(Path(__file__).resolve())],
    }


ARM_RAW = "raw_judge"
ARM_ESC_REFUTE = "judge+escalation(refute)"
ARM_ESC_CONFIRM = "judge+escalation(confirm)"
ARM_ESCALATED = "judge+escalation"           # BOTH directions live
_ARM_DIRECTIONS: Dict[str, Tuple[bool, bool]] = {
    ARM_RAW: (False, False),
    ARM_ESC_REFUTE: (True, False),
    ARM_ESC_CONFIRM: (False, True),
    ARM_ESCALATED: (True, True),
}


def arm_for(refute: bool, confirm: bool) -> str:
    """The arm label for a (refute, confirm) escalation state."""
    for name, dirs in _ARM_DIRECTIONS.items():
        if dirs == (bool(refute), bool(confirm)):
            return name
    raise ValueError(f"no arm for {(refute, confirm)!r}")  # unreachable


# Arm-QUALIFIED metric keys. There is deliberately no plain `fpr` key: a
# raw-judge false-alarm rate is an upper bound on production's, and the
# two were previously reported under one name, so a report comparison
# could silently mix them. A consumer that wants "the FPR" must now say
# which arm it means — and a KeyError is the point.
#
# Each metric is keyed on the direction that can ACTUALLY move it, not on
# the whole arm — false precision would block legitimate comparisons:
#   * FPR counts REFUTED verdicts on clean trials. `_escalate_confirm`
#     provably never emits a REFUTED (it returns the main model's
#     CONFIRMED, or caps the cheap one's confidence), so the confirm
#     direction cannot move FPR or TPR. FPR is keyed on REFUTE only.
#   * the false-CONFIRM rate counts corrupted trials CONFIRMED at
#     actionable confidence — exactly what the confirm direction's
#     0.6 cap removes. It is keyed on CONFIRM only.
FPR_KEY: Dict[str, str] = {
    arm: ("fpr_escalated" if refute else "fpr_raw_judge")
    for arm, (refute, _c) in _ARM_DIRECTIONS.items()
}
CONFIRM_KEY: Dict[str, str] = {
    arm: ("false_confirm_actionable_escalated" if confirm
          else "false_confirm_actionable_raw")
    for arm, (_r, confirm) in _ARM_DIRECTIONS.items()
}
# Measured 2026-08-04 over $GHOST_HOME/system/llm_recordings (day-files
# 2026-07-30..08-04): joining recorded verify prompts on the CLAIM and
# reading which model served each verdict (worker = Gemma 4 E4B,
# main = Qwen3.6-35B), 50 cheap-judge REFUTEs reached the main model and
# 42 were overturned to CONFIRMED. The durable log's own GhostAgent lines
# over a longer window read 80 overturned / 19 stood = 81% (a naive grep
# gives 89% because WARNING lines are mirrored to the GhostStream logger
# and the "verdict stands" line is INFO — count one logger only).
ESCALATION_OVERTURN_MEASURED = "42 of 50 (84%) on the recorded corpus"


# ── High stakes (the CONFIRM direction's trigger) ────────────────────

# Segment marker of the packed evidence digest: `_collect_verifier_
# evidence` emits `"\n\n".join(f"[{tool_name}] {content}")`, so this
# recovers the per-tool-output boundaries.
_EVIDENCE_SEGMENT_RE = re.compile(r"(?:\A|\n\n)(\[[^\]\n]{1,80}\] )")


def _evidence_segments(evidence: str) -> List[str]:
    """Split a packed evidence digest back into its per-tool outputs."""
    starts = [m.start(1) for m in _EVIDENCE_SEGMENT_RE.finditer(evidence or "")]
    if not starts:
        return [evidence or ""]
    return [evidence[a:b] for a, b in zip(starts, starts[1:] + [len(evidence)])]


def derive_high_stakes(evidence: str) -> bool:
    """Would production have marked this turn HIGH-STAKES?

    Production computes it as `_turn_had_tool_failure(tools_run_this_turn)`
    — `looks_like_tool_error` applied to EACH tool output's content. A
    bench case has no `tools_run`; what it has is the packed evidence
    digest the verifier was actually shown, which is those same outputs
    joined as `[tool] content`. So this splits the digest back into its
    segments and applies THE SAME sniffer per segment — not a second
    failure definition (a duplicated one is what made the corpus and the
    operator line disagree before), and not one blob check either:
    `looks_like_tool_error` only scans the first 120 chars for its text
    markers, so a blob check sees only the FIRST tool's head. Measured
    2026-08-04 on the 86-case mined pool: segmented 14 (16.3%) against
    blob-only 10 (11.6%) — 4 real failures the blob check missed.

    ⚠ HONEST BOUND: the digest is budget-truncated (newest-heavy, 4-way
    split), so a failed tool whose output was trimmed out entirely is
    invisible here. This is therefore a LOWER bound on production's flag,
    never an upper one. For reference, production measured 140 of 1488
    turns (9.4%) with a failed tool call across the whole corpus; the
    mined pool is a biased subset of that (production-REFUTED turns are
    excluded by construction), so the two rates are not expected to match
    and neither validates the other.
    """
    try:
        from ..distill.outcome_heuristics import looks_like_tool_error
    except Exception:  # noqa: BLE001 — never lose a trial to this
        logger.debug("verify_bench: outcome_heuristics unavailable; "
                     "high_stakes derivation off")
        return False
    return any(looks_like_tool_error(seg)
               for seg in _evidence_segments(evidence or ""))

# Literal segments of the live claim template, split around its
# placeholders. Parsing rendered prompts with these (instead of
# hand-copied header strings) keeps extraction correct automatically when
# the template text changes. The {{ }} format-escapes (the JSON schema at
# the template's tail) must be unescaped to match RENDERED text.
def _unescape(seg: str) -> str:
    return seg.replace("{{", "{").replace("}}", "}")


def _invertible_templates() -> List[Tuple[str, str, str, str]]:
    """Every verify template whose rendered form can be reversed into a case.

    Deliberately includes the two-stage ENUMERATE template, not just the
    classic fallback — see `parse_rendered_claim_prompt`. Adjudicate is
    excluded: it interpolates `{suspects}` between the fields, so a recovered
    "evidence" would carry stage-1 output.
    """
    from ..core import verifier as _v
    candidates = [_VERIFY_CLAIM_PROMPT, _v._VERIFY_ENUMERATE_PROMPT]
    # ...AND the LIVE (tuned) forms. Recorded prompts were rendered with
    # whatever template was deployed at the time, which since 2026-07-30 is a
    # GEPA artifact — not the baseline string in this source file. When the
    # deployed enumerate template DIVERGED from the baseline, inverting only
    # the baselines matched 0 of 580 verify-shaped records on the live corpus
    # even though the opening sentence was identical, because the two differ
    # further down. Resolving through `_stage_template` picks up the artifact
    # when $GHOST_HOME has one, so both eras invert.
    #
    # (As of 2026-08-04 `verifier.enumerate.json` is byte-identical to the
    # baseline again, so baseline-only inversion currently matches 336 of 580.
    # The 0-of-580 figure describes the DIVERGED state this guards against,
    # not today's.)
    try:
        for name, base in (("verifier.enumerate", _v._VERIFY_ENUMERATE_PROMPT),):
            live = _v._stage_template(name, base)
            if live and live != base:
                candidates.append(live)
    except Exception:  # noqa: BLE001 — baseline-only is a valid degraded mode
        pass
    out = []
    seen = set()
    for tpl in candidates:
        parts = _split_template(tpl)
        # Keyed on the WHOLE 4-tuple, not `parts[0]`. Keying on the pre-claim
        # prefix alone dropped precisely the shape described above — a tuned
        # artifact whose opening sentence is identical to the baseline but
        # which diverges after `{claim}` — so the deployed template lost the
        # tie-break to the baseline and its prompts stopped inverting. Dormant
        # only while `verifier.enumerate.json` happens to be byte-identical to
        # the baseline; the next promotion would re-arm the 0-of-580 failure.
        if parts is not None and parts not in seen:
            seen.add(parts)
            out.append(parts)
    return out


_INVERTIBLE_TEMPLATES: List[Tuple[str, str, str, str]] = []

_PRE_CLAIM, _rest = _VERIFY_CLAIM_PROMPT.split("{claim}")
_PRE_EVIDENCE, _rest2 = _rest.split("{evidence}")
_PRE_CONTEXT, _POST_CONTEXT = _rest2.split("{context}")
_PRE_CLAIM = _unescape(_PRE_CLAIM)
_PRE_EVIDENCE = _unescape(_PRE_EVIDENCE)
_PRE_CONTEXT = _unescape(_PRE_CONTEXT)
_POST_CONTEXT = _unescape(_POST_CONTEXT)


# ── Cases ────────────────────────────────────────────────────────────

@dataclass
class BenchCase:
    """One known-good (claim, evidence, context) triple.

    ``high_stakes`` is the CONFIRM-escalation trigger (a tool failed this
    turn). ``None`` — the default, and what every mined case carries —
    means DERIVE it per trial from that trial's own evidence, so a fault
    that turns the evidence into a tool failure (`silent_failure`) makes
    the trial high-stakes exactly as production would. An explicit
    True/False is an author pinning the case and overrides derivation for
    all of its trials; the hand-authored seed set uses that field when a
    fixture needs to assert its own stakes.
    """
    case_id: str
    claim: str
    evidence: str
    context: str
    source: str = "seed"       # "seed" | "recording"
    notes: str = ""
    high_stakes: Optional[bool] = None


@dataclass
class BenchTrial:
    """One verify call to make: a case, possibly corrupted by a fault.

    ``high_stakes`` is resolved at build time (see `build_trials`) and is
    passed straight through to `verify_claim`, which is the only way the
    CONFIRM-direction escalation can fire at all.
    """
    case_id: str
    fault: str                 # "clean" or a fault-library name
    expected: str              # "CONFIRMED" | "REFUTED" | "NOT_REFUTED"
    claim: str
    evidence: str
    context: str
    note: str = ""
    high_stakes: bool = False


@dataclass
class TrialResult:
    trial: BenchTrial
    verdict: Optional[str]     # None => verifier skipped / unparseable
    confidence: float = 0.0
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)
    suspects: Optional[List[Dict[str, str]]] = None
    elapsed_s: float = 0.0
    error: str = ""
    # Whether each escalation direction actually BIT on this trial, read
    # off the VerifyResult rather than inferred. Inference would be a
    # second copy of the escalation's own rules; these two flags are the
    # only direct evidence a run has that a direction fired at all, and
    # they are what makes "GHOST_VERIFY_ESCALATE_CONFIRM=0 restores the
    # old behaviour" a checkable statement instead of a claim.
    escalated_overturn: bool = False
    confirm_withheld: bool = False
    # Escalation discipline B (2026-08-06): tier routing downgraded a
    # gloss-shaped refute to UNCERTAIN without a main call.
    escalation_downgraded: bool = False
    truncation_guarded: bool = False
    escalation_replaced: bool = False
    # The only discipline that shipped ON (2026-08-07 review): without
    # these two the bench could not tell a mechanically-settled trial
    # from one where nothing fired.
    objection_dismissed: bool = False
    objection_upheld: bool = False
    # Pre-escalation snapshot (replay-scorer infra, 2026-08-07): the
    # cheap judge's verdict before the mechanical layer/escalation —
    # what turns a policy change into an offline replay.
    cheap_verdict: Optional[str] = None
    cheap_confidence: Optional[float] = None
    cheap_issues: Optional[List[str]] = None
    # §4BF flip (i): the logit-expectation probe's raw score, when the
    # probe fired. Without this the A/B can only see the BLENDED
    # confidence — a null result could not distinguish "probe
    # uninformative" from "probe informative but the blend washed it
    # out", which is exactly the 2026-07-30 w=0.5 ambiguity again.
    probe_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.trial.case_id,
            "fault": self.trial.fault,
            "expected": self.trial.expected,
            "high_stakes": self.trial.high_stakes,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
            "suspects": self.suspects,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "note": self.trial.note,
            "escalated_overturn": self.escalated_overturn,
            "confirm_withheld": self.confirm_withheld,
            "escalation_downgraded": self.escalation_downgraded,
            "truncation_guarded": self.truncation_guarded,
            "escalation_replaced": self.escalation_replaced,
            "objection_dismissed": self.objection_dismissed,
            "objection_upheld": self.objection_upheld,
            "cheap_verdict": self.cheap_verdict,
            "cheap_confidence": self.cheap_confidence,
            "cheap_issues": self.cheap_issues,
            "probe_score": self.probe_score,
        }


def load_cases_jsonl(path: str | Path) -> List[BenchCase]:
    """Load seed cases. Keys: id/case_id, claim, evidence, context,
    notes (optional). Lines that fail to parse or lack a claim+evidence
    are skipped with a warning — a bad line must not kill the bench."""
    cases: List[BenchCase] = []
    for lineno, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("verify_bench: %s:%d unparseable (%s)",
                           path, lineno, exc)
            continue
        claim = str(obj.get("claim") or "")
        evidence = str(obj.get("evidence") or "")
        if not claim.strip() or not evidence.strip():
            logger.warning("verify_bench: %s:%d missing claim/evidence",
                           path, lineno)
            continue
        # `high_stakes` is TRI-STATE: absent/null => derive per trial.
        # `bool(obj.get(...))` would collapse "absent" into an explicit
        # False and freeze every case at not-high-stakes, which is exactly
        # how the CONFIRM direction would stay dark in the bench while
        # looking configured.
        hs = obj.get("high_stakes")
        cases.append(BenchCase(
            case_id=str(obj.get("case_id") or obj.get("id")
                        or f"seed-{lineno}"),
            claim=claim,
            evidence=evidence,
            context=str(obj.get("context") or ""),
            source="seed",
            notes=str(obj.get("notes") or ""),
            high_stakes=(None if hs is None else bool(hs)),
        ))
    return cases


def _split_template(template: str) -> Optional[Tuple[str, str, str, str]]:
    """(pre_claim, pre_evidence, pre_context, post_context) for a template
    whose placeholders are claim/evidence/context in that order."""
    try:
        pre_claim, rest = template.split("{claim}", 1)
        pre_evidence, rest = rest.split("{evidence}", 1)
        pre_context, post = rest.split("{context}", 1)
    except ValueError:
        return None
    # `_unescape` is not optional: these templates carry `{{`/`}}` for the
    # literal JSON braces in their output spec, and `.format()` collapses them
    # to single braces. Matching the RAW segments against a RENDERED prompt
    # therefore fails ~800 chars into the tail — which is exactly why the
    # first version of this inverter matched 0 of 580 verify-shaped records
    # while its opening sentence matched perfectly.
    return (_unescape(pre_claim), _unescape(pre_evidence),
            _unescape(pre_context), _unescape(post))


def _invert(text: str, parts: Tuple[str, str, str, str]) -> Optional[Dict[str, str]]:
    pre_claim, pre_evidence, pre_context, post_context = parts
    if not text or not text.startswith(pre_claim):
        return None
    body = text[len(pre_claim):]
    i = body.find(pre_evidence)
    if i < 0:
        return None
    claim = body[:i]
    body = body[i + len(pre_evidence):]
    j = body.find(pre_context)
    if j < 0:
        return None
    evidence = body[:j]
    body = body[j + len(pre_context):]
    k = body.rfind(post_context) if post_context else len(body)
    if k < 0:
        return None
    return {"claim": claim, "evidence": evidence, "context": body[:k]}


def parse_rendered_claim_prompt(text: str) -> Optional[Dict[str, str]]:
    """Recover claim/evidence/context from ANY recorded verify prompt.

    ⚠ This used to invert ONLY `_VERIFY_CLAIM_PROMPT` — the CLASSIC path.
    Measured on 618 recorded verify calls: 281 enumerate, 244 adjudicate, 38
    code, **55 classic**. And classic is by construction the two-stage
    FALLBACK (`verifier.py` drops to it when stage 1 yields no usable
    suspects), so harvesting only it SELECTED CASES ON TWO-STAGE FAILURE —
    a bench built from the 9% of traffic where the real pipeline had already
    broken down. The enumerate template carries the same three fields, so it
    is inverted too, which is where the volume is.
    """
    if not _INVERTIBLE_TEMPLATES:
        _INVERTIBLE_TEMPLATES.extend(_invertible_templates())
    for parts in _INVERTIBLE_TEMPLATES:
        got = _invert(text, parts)
        if got is not None:
            return got
    return None


_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([A-Z_]+)"')


def _claim_extractors() -> List[Tuple[str, str]]:
    """(prefix, terminator) pairs that isolate `{claim}` in ANY verify prompt.

    Broader than `_invertible_templates`: adjudicate cannot yield a usable
    *evidence* field (it interpolates `{suspects}` between the fields), but
    its claim is recoverable, and adjudicate is where the verdicts are.
    """
    from ..core import verifier as _v
    tpls = [_VERIFY_CLAIM_PROMPT, _v._VERIFY_ENUMERATE_PROMPT]
    for name, base in (("verifier.enumerate", _v._VERIFY_ENUMERATE_PROMPT),
                       ("verifier.adjudicate",
                        getattr(_v, "_VERIFY_ADJUDICATE_PROMPT", ""))):
        if not base:
            continue
        tpls.append(base)
        try:
            live = _v._stage_template(name, base)
            if live and live != base:
                tpls.append(live)
        except Exception:  # noqa: BLE001
            pass
    out, seen = [], set()
    for tpl in tpls:
        if not tpl or "{claim}" not in tpl:
            continue
        pre, rest = tpl.split("{claim}", 1)
        m = re.search(r"\{[a-z_]+\}", rest)
        # FULL terminator segment, not a 120-char slice: `_invert` matches on
        # the whole thing, and a claim that happens to contain the truncated
        # prefix would make the two disagree about where the claim ends.
        term = _unescape(rest[:m.start()] if m else rest)
        # `{claim}` is not guaranteed to be the FIRST placeholder —
        # `_validate_stage_template` checks presence, not order. When it is
        # not, `pre` holds `{evidence}`/`{context}` literals that no rendered
        # text can start with, and `startswith` silently matched nothing at
        # all. Anchor on the literal run immediately before `{claim}` and
        # search for it instead of requiring a prefix.
        pre = _unescape(pre)
        m_pre = None
        for m_pre in re.finditer(r"\{[a-z_]+\}", pre):
            pass
        anchored = m_pre is not None
        if anchored:
            pre = pre[m_pre.end():]
        if not term.strip() or not pre.strip() or (pre, term) in seen:
            continue
        seen.add((pre, term))
        out.append((pre, term))
    # LONGEST prefix first. `_claim_of` returns on the first extractor that
    # matches, so a baseline whose pre-claim is a PREFIX of a tuned one would
    # shadow it and hand back the tuned template's extra preamble as part of
    # the claim — the same shadowing bug the 4-tuple dedupe above just fixed,
    # and here it cannot self-correct because all templates share a
    # terminator. A wrong claim key is silent: every case reverts to
    # `prod_verdict=unknown` and `exclude_refuted` stops excluding.
    out.sort(key=lambda pt: len(pt[0]), reverse=True)
    return out


_CLAIM_EXTRACTORS: List[Tuple[str, str]] = []


def _claim_of(text: str) -> str:
    """The `{claim}` a recorded verify prompt was rendered with ("" if none)."""
    if not _CLAIM_EXTRACTORS:
        _CLAIM_EXTRACTORS.extend(_claim_extractors())
    for pre, term in _CLAIM_EXTRACTORS:
        # `find`, not `startswith`: the anchor is the literal run immediately
        # before `{claim}`, which is at position 0 only when `{claim}` is the
        # template's first placeholder.
        start = text.find(pre)
        if start < 0:
            continue
        body = text[start + len(pre):]
        i = body.find(term)
        if i > 0:
            return body[:i].strip()
    return ""


def _claim_key(claim: str) -> str:
    return hashlib.sha256(" ".join(claim.split()).encode("utf-8")).hexdigest()


def _production_verdicts(paths: Iterable[str | Path]) -> Dict[str, str]:
    """claim-hash -> the verdict production actually reached FOR THAT CLAIM.

    Read from the ADJUDICATE/classic responses, which are the only records
    that carry one (an enumerate response is a suspects list). Used to keep
    claims the verifier REFUTED out of the clean-case pool.

    ⚠ Keyed on the CLAIM, not on `request_id`, and taking the LAST verdict
    rather than letting a REFUTED win. `request_id` is per-TURN — measured on
    the live corpus 2026-08-04, 348 distinct ids over 12,047 records with a
    single `"SYSTEM"` id holding 9,780 of them — so a request-level join is
    not "conservative but safe", it is wrong three ways. Of the 62 deduped
    cases the old join excluded:

      * 43 were `['REFUTED', 'CONFIRMED']` within one request, which is the
        `_escalate_refute` SIGNATURE (cheap judge false-refutes, main model
        overturns — all 43 multi-verdict requests are exactly this shape).
        Production did NOT refute those turns, and they are the *best* clean
        cases available because they survived two judges;
      * 16 were dropped on a verdict the CODE AUDITOR reached about a
        different claim in the same turn;
      * 3 fell in the shared `"SYSTEM"` bucket, poisoned by one REFUTED
        anywhere in 9,780 records.

    Net: 106 candidates → 44 kept under the old join, 86 under this one. The
    survivors were biased toward turns that never triggered a refutation
    anywhere — the same selection-bias class as the two-stage fallback bug
    that `parse_rendered_claim_prompt` exists to fix.

    ⚠ Known gap: a claim REFUTED in its own turn and CONFIRMED later in an
    UNRELATED turn would be admitted as clean. Zero instances on the live
    corpus today (the 42 conflicting claims are 41 same-request escalation
    pairs plus one cross-day claim that resolves REFUTED), and it is
    unguarded — day-files are read in `sorted()` i.e. chronological order,
    so "last" is genuinely last.
    """
    out: Dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = (rec.get("payload") or {}).get("messages") or []
            if not msgs:
                continue
            claim = _claim_of(str(msgs[0].get("content") or ""))
            if not claim:
                continue
            resp = rec.get("response")
            if not isinstance(resp, dict):
                continue
            choices = resp.get("choices") or [{}]
            first = choices[0] if isinstance(choices, list) and choices else {}
            content = ""
            if isinstance(first, dict):
                content = str((first.get("message") or {}).get("content") or "")
            m = _VERDICT_RE.search(content)
            if m:
                # LAST verdict wins: an escalation's final answer is the one
                # production acted on.
                out[_claim_key(claim)] = m.group(1)
    return out


def extract_cases_from_recordings(
        paths: Iterable[str | Path],
        *, exclude_refuted: bool = True) -> List[BenchCase]:
    """Mint bench cases out of GHOST_LLM_RECORD day-files.

    Cases are PRESUMED-GOOD by construction: every trial built from them
    treats the recovered claim as correct, so a `clean` trial expects
    CONFIRMED and every fault expects the corruption to be caught.

    ⚠ `exclude_refuted` is on by default and is load-bearing. A turn the
    verifier REFUTED in production is NOT a clean case, and admitting one
    enters a known-bad claim as `expected="CONFIRMED"` — which inflates the
    measured false-positive rate and steers the optimizer toward a MORE
    PERMISSIVE judge, i.e. the bench would actively reward the failure mode
    it exists to detect. Measured on the live corpus at the time this was
    added: 69 REFUTED verdicts among the recorded verify calls. The verdict
    is in the same day-file (`_production_verdicts`), joined on the CLAIM —
    it was simply never read. (The first version of this join used
    `request_id`, which is per-TURN and wrong; see `_production_verdicts`.)
    """
    # Computed unconditionally: it is the exclusion key when
    # `exclude_refuted` is on, and the `prod_verdict=` provenance note in
    # every case either way.
    verdicts = _production_verdicts(paths)
    dropped_refuted = 0
    seen: set = set()
    cases: List[BenchCase] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("verify_bench: cannot read %s: %s", p, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = (rec.get("payload") or {}).get("messages") or []
            if not msgs:
                continue
            parsed = parse_rendered_claim_prompt(
                str(msgs[0].get("content") or ""))
            if not parsed or not parsed["claim"].strip() \
                    or not parsed["evidence"].strip():
                continue
            if exclude_refuted and verdicts.get(
                    _claim_key(parsed["claim"])) == "REFUTED":
                dropped_refuted += 1
                continue
            digest = hashlib.sha256(
                (parsed["claim"] + "\x00" + parsed["evidence"] + "\x00"
                 + parsed["context"]).encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            # REDACT. `GHOST_LLM_RECORD` day-files are unredacted by design
            # (that is why the journal marks them local-disk-only), but a
            # bench case file is a durable artifact that gets copied into
            # ablation bundles and compared across weeks. Every other corpus
            # write in this codebase redacts; this one did not, because until
            # now it produced almost nothing.
            # Resolve the verdict BEFORE redaction: the claim hash must be
            # taken from the same text `_production_verdicts` hashed.
            # `verdicts` is empty under exclude_refuted=False, so read the
            # map directly rather than annotating all 106 cases
            # `prod_verdict=unknown` — that note is the only provenance a
            # mined case carries, and claiming "unknown" for a verdict that
            # is sitting in the same day-file is a lie about the data.
            _pv = verdicts.get(_claim_key(parsed["claim"]), "unknown")
            try:
                from ..distill.redact import redact_text, RedactionConfig
                _cfg = RedactionConfig()
                parsed = {k: redact_text(v, _cfg) for k, v in parsed.items()}
            except Exception:  # noqa: BLE001 — never lose a case to redaction
                pass
            cases.append(BenchCase(
                case_id=f"rec-{digest[:10]}",
                claim=parsed["claim"],
                evidence=parsed["evidence"],
                context=parsed["context"],
                source="recording",
                notes=(f"{rec.get('ts') or ''} "
                       f"prod_verdict={_pv}").strip(),
            ))
    if dropped_refuted:
        logger.info(
            "verify_bench: dropped %d recording-derived case(s) whose turn was "
            "REFUTED in production — they are not clean cases",
            dropped_refuted)
    return cases


# ── Fault library ────────────────────────────────────────────────────
#
# A fault takes (case, rng, pool) and returns a corrupted
# (claim, evidence, context, note) tuple, or None when it does not apply
# to this case (no shared number to swap, no donor claim, ...). Each
# fault carries the verdict it SHOULD produce:
#   "REFUTED"     — the corruption must be caught (TPR side)
#   "NOT_REFUTED" — the claim is still true; refuting it is a false
#                   alarm (robustness side). CONFIRMED and UNCERTAIN
#                   both count as correct here.

_NUM_RE = re.compile(r"\d[\d,.]*\d|\d")


def _mutate_number(tok: str) -> str:
    """Deterministically perturb a numeric token so it contradicts the
    evidence but keeps its shape (digit count, separators)."""
    digits = [c for c in tok if c.isdigit()]
    # Bump the last digit; if that wraps to the same token (single "9"),
    # bump the first too.
    out = list(tok)
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].isdigit():
            out[idx] = str((int(out[idx]) + 1) % 10)
            break
    mutated = "".join(out)
    if mutated == tok and digits:
        for idx in range(len(out)):
            if out[idx].isdigit():
                out[idx] = str((int(out[idx]) + 3) % 10)
                break
        mutated = "".join(out)
    return mutated


def fault_fact_swap(case: BenchCase, rng: random.Random,
                    pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Change a number that the claim shares with the evidence, so the
    claim now contradicts its own source."""
    claim_nums = _NUM_RE.findall(case.claim)
    shared = sorted({t for t in claim_nums
                     if len(t) >= 2 and t in case.evidence})
    if not shared:
        return None
    target = rng.choice(shared)
    mutated = _mutate_number(target)
    if mutated == target:
        return None
    new_claim = case.claim.replace(target, mutated, 1)
    return (new_claim, case.evidence, case.context,
            f"swapped {target!r} -> {mutated!r}")


def fault_fabrication(case: BenchCase, rng: random.Random,
                      pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Append a concrete, plausible-sounding fact that appears in no
    tool output — the rubric's canonical fabrication."""
    fabrications = [
        "The figure was independently confirmed by the Karlsen "
        "Institute's annual audit on 12 March 2019.",
        "It also won the Meridian Prize for this category in 2021.",
        "The lead maintainer, Dr. Elin Vasquez, verified the result "
        "earlier today.",
    ]
    usable = [f for f in fabrications
              if "Karlsen" not in case.evidence
              and f.split()[2] not in case.evidence]
    if not usable:
        return None
    fab = rng.choice(usable)
    return (case.claim.rstrip() + " " + fab, case.evidence, case.context,
            f"appended fabrication: {fab[:60]}")


def fault_wrong_topic(case: BenchCase, rng: random.Random,
                      pool: List[BenchCase], seed: int = 0
                      ) -> Optional[Tuple[str, str, str, str]]:
    """Answer a different question: swap in another case's claim wholesale.
    Needs a non-empty user request (the alignment check is skipped when
    the request is blank) and a donor case."""
    if not case.context.strip():
        return None
    donors = [c for c in pool
              if c.case_id != case.case_id and c.claim != case.claim]
    if not donors:
        return None
    # Donor chosen by RANK, not by `rng.choice(donors)`. An index into a list
    # whose length changes reshuffles on every pool growth; ranking each
    # candidate by hash(this_case, donor) keeps the pick stable across
    # growth. `rng` is unused on purpose: the choice must not depend on how
    # many draws earlier faults made.
    #
    # `seed` IS mixed in, so --seed varies this fault like every other one.
    # Without it the donor was a pure function of the case-id pair and
    # `--seed 0` vs `--seed 999` produced identical wrong_topic trials while
    # the report recorded `seed` as if it parameterised the whole trial set.
    #
    # Note this is stability under growth, not immunity: for a pool growing
    # 21 -> 65 the chance a new case outranks the incumbent donor is
    # m/(n+m) ~ 69%, and 16 of 21 wrong_topic trials (76%) did change. The
    # pool hash in `bench_provenance` is what makes that safe, by refusing
    # the comparison rather than by preventing the change.
    donor = min(donors, key=lambda c: hashlib.sha256(
        f"{seed}|{case.case_id}|{c.case_id}".encode("utf-8")).hexdigest())
    return (donor.claim, case.evidence, case.context,
            f"claim replaced with {donor.case_id}'s")


def fault_silent_failure(case: BenchCase, rng: random.Random,
                         pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """The tool actually failed but the claim still reports success:
    replace the evidence with an error body, keep the claim."""
    m = re.match(r"\[([^\]\n]{1,40})\]", case.evidence)
    tool = m.group(1) if m else "web_search"
    failures = [
        f"[{tool}] ERROR: HTTP 403 Forbidden — request blocked",
        f"[{tool}] (empty output)",
        f"[{tool}] Traceback (most recent call last):\n  ...\n"
        f"TimeoutError: request timed out after 30s",
    ]
    return (case.claim, rng.choice(failures), case.context,
            "evidence replaced with tool failure")


def fault_artifact_leak(case: BenchCase, rng: random.Random,
                        pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Inject machine noise into the reply — the historical
    SEARCH/REPLACE marker-leak shape (2026-07-14)."""
    noise = "\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
    mid = len(case.claim) // 2
    cut = case.claim.rfind(" ", 0, mid)
    cut = cut if cut > 0 else mid
    new_claim = case.claim[:cut] + noise + case.claim[cut:]
    return (new_claim, case.evidence, case.context,
            "diff markers injected into reply")


def fault_constraint_violation(case: BenchCase, rng: random.Random,
                               pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Add an explicit one-word format constraint to the request that the
    (unchanged, long) reply clearly violates."""
    if len(case.claim.split()) <= 8 or not case.context.strip():
        return None
    new_context = case.context.rstrip() + " Answer with a single word only."
    return (case.claim, case.evidence, new_context,
            "request now demands a single word")


def fault_evidence_truncation(case: BenchCase, rng: random.Random,
                              pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Degrade the EVIDENCE, not the claim: cut it hard mid-stream. The
    claim is still true — refuting it for pipeline noise is the
    evidence-packer failure shape (spurious REFUTE on truncated output,
    fixed 2026-07-17). Expected NOT_REFUTED."""
    # Emit the PRODUCTION marker, not a lookalike: with a bare
    # " …[truncated]" string the truncation guard was structurally DARK
    # in the very arm that motivates it — `evidence_was_truncated()`
    # returned False on every degraded trial, so no bundle could ever
    # show the mechanism working (fresh-eye MAJOR, 2026-08-06). Routing
    # through the real packer slicer also exercises the packer itself.
    from ..core.agent import _slice_evidence_body, evidence_was_truncated
    if len(case.evidence) < 200:
        return None
    # ⚠ VARIED cut depth (task #25, 2026-08-07): the fixed 0.35 keep
    # made severity ~0.65 on every trial — always above the guard's
    # 0.25 floor, so the "degraded-evidence robustness" axis was a
    # 10-trial test of the guard's regex and NEVER tested the judge on
    # lightly-cut evidence (where absence complaints are real evidence
    # and the guard correctly stands down). The per-case rng keeps each
    # case's depth stable for life (per-case seeding contract).
    _keep_frac = rng.choice((0.35, 0.6, 0.8, 0.92))
    keep = max(80, int(len(case.evidence) * _keep_frac))
    sliced = _slice_evidence_body(case.evidence, keep, case.claim)
    # ⚠ The slicer drops the marker when `keep` cannot fit it (its
    # fail-closed cap invariant, tightened 2026-08-07) — but an UNMARKED
    # cut is a different fault than this one claims to inject. Widen the
    # budget once to make room for the marker (120 covers the marker +
    # its 40-char floor for any body under ~100KB); a case whose cut
    # STILL goes unmarked is skipped rather than emitted mislabelled.
    if not evidence_was_truncated(sliced):
        keep = max(120, keep)
        sliced = _slice_evidence_body(case.evidence, keep, case.claim)
        if not evidence_was_truncated(sliced):
            return None
    return (case.claim, sliced,
            case.context, "evidence hard-truncated, claim still true")


# name -> (expected verdict, fault fn)
FAULTS: Dict[str, Tuple[str, Callable]] = {
    "fact_swap": ("REFUTED", fault_fact_swap),
    "fabrication": ("REFUTED", fault_fabrication),
    "wrong_topic": ("REFUTED", fault_wrong_topic),
    "silent_failure": ("REFUTED", fault_silent_failure),
    "artifact_leak": ("REFUTED", fault_artifact_leak),
    "constraint_violation": ("REFUTED", fault_constraint_violation),
    "evidence_truncation": ("NOT_REFUTED", fault_evidence_truncation),
}


def _takes_seed(fn) -> bool:
    """True when a fault function accepts the run seed as a 4th argument.

    Deliberately NOT `lru_cache`d: the argument is a callable, so an
    unhashable one would raise inside the cache wrapper BEFORE this
    function's own error handling, `maxsize=None` would pin every fault
    function for the process lifetime, and equal-but-distinct callables
    would collide onto one arity. Callers resolve arity once per
    `build_trials`, so there is nothing to cache.
    """
    try:
        return len(inspect.signature(fn).parameters) >= 4
    except (TypeError, ValueError):
        return False


def build_trials(cases: List[BenchCase],
                 fault_names: Optional[List[str]] = None,
                 seed: int = 0,
                 include_clean: bool = True) -> List[BenchTrial]:
    """Expand cases into the trial list: one clean trial per case plus one
    trial per applicable fault.

    ⚠ Seeded PER CASE, not once for the whole run. A single shared `rng` made
    every trial depend on how many cases preceded it, so ADDING a case
    silently rewrote earlier ones: measured, re-running the identical command
    after the pool grew from 13 to 21 cases changed the content of 19 of the
    original 97 trials (20%) — while the report still looked like the same
    benchmark. Since T0/T+7d comparisons are the whole point of this
    instrument, that made corpus growth indistinguishable from a real
    regression. Per-case seeding keeps a case's faults stable for life.

    (`wrong_topic` still draws its donor from the pool, so its content can
    change when pool MEMBERSHIP changes. That is why the report records a
    hash of the case set — see `run_bench` — so two runs are only ever
    compared when they were built from the same pool.)

    HIGH STAKES is resolved here, per trial: an explicit `case.high_stakes`
    pins every trial of that case, otherwise it is DERIVED from the
    trial's own (post-fault) evidence. Deriving after the fault is the
    point, not a side effect — `silent_failure` replaces the evidence with
    a tool error under an unchanged success claim, which IS the population
    `_escalate_confirm` exists for, and it is the only thing that makes the
    hand-authored seed set exercise the CONFIRM direction at all (measured
    2026-08-04: 0 of 21 seed cases derive high-stakes from their own clean
    evidence, but 70 of 107 `silent_failure` trials do).
    """
    names = list(fault_names) if fault_names else list(FAULTS)
    unknown = [n for n in names if n not in FAULTS]
    if unknown:
        raise ValueError(f"unknown fault(s): {unknown}")
    # Arity resolved ONCE per call, not per trial.
    _seed_arity = {n: _takes_seed(FAULTS[n][1]) for n in names}
    trials: List[BenchTrial] = []
    def _stakes(case: BenchCase, evidence: str) -> bool:
        return (derive_high_stakes(evidence) if case.high_stakes is None
                else bool(case.high_stakes))

    for case in cases:
        if include_clean:
            trials.append(BenchTrial(
                case_id=case.case_id, fault="clean", expected="CONFIRMED",
                claim=case.claim, evidence=case.evidence,
                context=case.context,
                high_stakes=_stakes(case, case.evidence)))
        for name in names:
            expected, fn = FAULTS[name]
            # Per (case, FAULT). Per-case alone was not enough: the faults
            # share one rng in sequence, so a fault whose draw count depends
            # on the pool (`wrong_topic` picks a donor) shifted the stream for
            # every fault after it — growing the pool still rewrote unrelated
            # trials. An independent stream per fault makes each trial's
            # content a pure function of (seed, case_id, fault).
            fault_seed = int(hashlib.sha256(
                f"{seed}:{case.case_id}:{name}".encode("utf-8")
            ).hexdigest()[:8], 16)
            rng = random.Random(fault_seed)
            # Explicit arity check, NOT try/except TypeError — the latter
            # would silently swallow a genuine TypeError raised INSIDE a
            # fault and retry it with fewer arguments.
            out = (fn(case, rng, cases, seed) if _seed_arity[name]
                   else fn(case, rng, cases))
            if out is None:
                continue
            claim, evidence, context, note = out
            trials.append(BenchTrial(
                case_id=case.case_id, fault=name, expected=expected,
                claim=claim, evidence=evidence, context=context,
                note=note, high_stakes=_stakes(case, evidence)))
    return trials


# ── Scoring ──────────────────────────────────────────────────────────

def _rate(num: int, denom: int) -> Optional[float]:
    return round(num / denom, 3) if denom else None


def score_trials(results: List[TrialResult],
                 arm: str = ARM_RAW) -> Dict[str, Any]:
    """Aggregate trial results into per-fault and overall metrics.

    Rates are computed over JUDGED trials (verdict produced); skips and
    errors are reported alongside, never silently folded into a rate.

    ``arm`` names the VERDICT PIPELINE these results came from (see
    `escalation_arm`) and is not cosmetic. Two headline numbers are
    emitted under ARM-QUALIFIED keys, each keyed on the escalation
    direction that can actually move it:

      * `fpr_raw_judge` / `fpr_escalated` — clean trials REFUTED. Only
        the REFUTE direction moves this (it overturns refutes, which
        lowers FPR — and lowers TPR whenever it overturns a genuine
        catch). `_escalate_confirm` never emits a REFUTED, so it cannot.
      * `false_confirm_actionable_raw` / `_escalated` — CORRUPTED trials
        that came back CONFIRMED, with the actionable (>=0.7) rate. This
        is the quantity the CONFIRM direction removes, by capping a
        withheld confirmation at 0.6. §4J item 2 measured 8% of corrupted
        trials CONFIRMED at >=0.7, each able to upgrade a structural
        FAILED into a fabricated PASSED — the bench reported that number
        nowhere at all before.

    There is no bare `fpr`, and no unqualified false-confirm key, so a
    stale reader gets a KeyError instead of a number from the wrong arm.
    """
    if arm not in FPR_KEY:
        raise ValueError(f"unknown arm {arm!r}; expected one of "
                         f"{sorted(FPR_KEY)}")
    by_fault: Dict[str, List[TrialResult]] = {}
    for r in results:
        by_fault.setdefault(r.trial.fault, []).append(r)

    per_fault: Dict[str, Any] = {}
    for fault, rs in sorted(by_fault.items()):
        expected = rs[0].trial.expected
        skipped = [r for r in rs if r.verdict is None]
        judged = [r for r in rs if r.verdict is not None]
        refuted = [r for r in judged if r.verdict == "REFUTED"]
        refuted_act = [r for r in refuted if r.confidence >= ACTIONABLE_CONF]
        confirmed = [r for r in judged if r.verdict == "CONFIRMED"]
        uncertain = [r for r in judged if r.verdict == "UNCERTAIN"]
        entry: Dict[str, Any] = {
            "expected": expected,
            "n": len(rs),
            "judged": len(judged),
            "skipped": len(skipped),
            "confirmed": len(confirmed),
            "refuted": len(refuted),
            "uncertain": len(uncertain),
            "mean_confidence": round(
                sum(r.confidence for r in judged) / len(judged), 3)
                if judged else None,
        }
        if expected == "REFUTED":
            entry["catch_rate"] = _rate(len(refuted), len(judged))
            entry["catch_rate_actionable"] = _rate(
                len(refuted_act), len(judged))
            # The MISS side, at the confidence the agent would act on.
            # A corrupted trial CONFIRMED at >=0.7 is a fabricated pass
            # that can upgrade a structural FAILED (§4J item 2) — the
            # exact population the CONFIRM escalation caps to 0.6.
            entry["false_confirm_rate"] = _rate(len(confirmed), len(judged))
            entry["false_confirm_rate_actionable"] = _rate(
                len([r for r in confirmed
                     if r.confidence >= ACTIONABLE_CONF]), len(judged))
        elif expected == "CONFIRMED":
            entry["false_alarm_rate"] = _rate(len(refuted), len(judged))
            entry["false_alarm_rate_actionable"] = _rate(
                len(refuted_act), len(judged))
            entry["confirm_rate"] = _rate(len(confirmed), len(judged))
        else:  # NOT_REFUTED
            entry["false_alarm_rate"] = _rate(len(refuted), len(judged))
            entry["false_alarm_rate_actionable"] = _rate(
                len(refuted_act), len(judged))
        per_fault[fault] = entry

    def _overall(expected_values: Tuple[str, ...],
                 positive: str) -> Dict[str, Any]:
        rs = [r for r in results if r.trial.expected in expected_values]
        judged = [r for r in rs if r.verdict is not None]
        hit = [r for r in judged if r.verdict == positive]
        hit_act = [r for r in hit if r.confidence >= ACTIONABLE_CONF]
        return {
            "n": len(rs), "judged": len(judged),
            "rate": _rate(len(hit), len(judged)),
            "rate_actionable": _rate(len(hit_act), len(judged)),
        }

    hs = [r for r in results if r.trial.high_stakes]
    return {
        "arm": arm,
        "per_fault": per_fault,
        # Did each direction actually BITE? Read off the verdicts, not
        # inferred. `high_stakes_trials == 0` means the run carries no
        # evidence about the CONFIRM direction whatever the arm says.
        #
        # `confirm_eligible` exists because `confirm_withheld == 0` is
        # otherwise ambiguous between "the main model agreed every time"
        # (evidence) and "no high-stakes trial ever reached a CONFIRMED,
        # so the direction was never invoked" (no evidence). Silence that
        # reads as success is the defect class this whole review keeps
        # turning up. Eligible = high-stakes AND ended CONFIRMED AND was
        # not already adjudicated by the refute escalation (which
        # `_escalate_confirm` skips via its `escalated_overturn` guard).
        "escalation_events": {
            "high_stakes_trials": len(hs),
            "refute_overturned": sum(1 for r in results
                                     if r.escalated_overturn),
            # Escalation discipline ship-gate metrics (2026-08-06): the
            # 2026-08-05 ad-hoc overturn attribution, formalized. A
            # RESCUE is an overturn on a trial that EXPECTED a
            # non-refute (the escalation repaired a false alarm); DAMAGE
            # is an overturn on a REFUTED-expected trial (the escalation
            # destroyed a correct catch — 23 of these vs 13 rescues on
            # the Selene pipeline run, the number that motivated the
            # rebuttal-burden contract). Same split for tier-routing
            # downgrades, which cost no main call.
            "overturn_rescues": sum(
                1 for r in results if r.escalated_overturn
                and r.trial.expected in ("CONFIRMED", "NOT_REFUTED")),
            "overturn_damage": sum(
                1 for r in results if r.escalated_overturn
                and r.trial.expected == "REFUTED"),
            # ⚠ EXCLUSIVE of the truncation guard (2026-08-07 review):
            # the guard sets BOTH flags, so one guard event used to
            # increment downgrades + downgrade_rescues + the two guard
            # counters at once — and with tier routing default-OFF,
            # every "downgrade" the report attributed to tier routing
            # was actually a guard event.
            "downgrades": sum(1 for r in results
                              if r.escalation_downgraded
                              and not r.truncation_guarded),
            # Split by MECHANISM: tier routing and the truncation guard
            # have different failure modes and must not be pooled.
            "truncation_guarded": sum(1 for r in results
                                      if r.truncation_guarded),
            "truncation_guard_rescues": sum(
                1 for r in results if r.truncation_guarded
                and r.trial.expected in ("CONFIRMED", "NOT_REFUTED")),
            "truncation_guard_damage": sum(
                1 for r in results if r.truncation_guarded
                and r.trial.expected == "REFUTED"),
            "downgrade_rescues": sum(
                1 for r in results if r.escalation_downgraded
                and not r.truncation_guarded
                and r.trial.expected in ("CONFIRMED", "NOT_REFUTED")),
            "downgrade_damage": sum(
                1 for r in results if r.escalation_downgraded
                and not r.truncation_guarded
                and r.trial.expected == "REFUTED"),
            # The shipped-ON mechanism, first-class: a dismiss on a
            # non-refute-expected trial repaired a false alarm; an
            # uphold on a REFUTED-expected trial protected a catch.
            # The inverse cells are the mechanism's own damage.
            "objection_dismissed": sum(1 for r in results
                                       if r.objection_dismissed),
            "objection_dismiss_rescues": sum(
                1 for r in results if r.objection_dismissed
                and r.trial.expected in ("CONFIRMED", "NOT_REFUTED")),
            "objection_dismiss_damage": sum(
                1 for r in results if r.objection_dismissed
                and r.trial.expected == "REFUTED"),
            "objection_upheld": sum(1 for r in results
                                    if r.objection_upheld),
            "objection_uphold_protects": sum(
                1 for r in results if r.objection_upheld
                and r.trial.expected == "REFUTED"),
            "objection_uphold_damage": sum(
                1 for r in results if r.objection_upheld
                and r.trial.expected in ("CONFIRMED", "NOT_REFUTED")),
            # A strong-UNCERTAIN replacement is neither overturn nor
            # downgrade: the refute is gone but nothing was confirmed.
            # Counted separately so neither pool absorbs it silently.
            "replaced_uncertain": sum(1 for r in results
                                      if r.escalation_replaced),
            "replaced_rescues": sum(
                1 for r in results if r.escalation_replaced
                and r.trial.expected in ("CONFIRMED", "NOT_REFUTED")),
            "replaced_damage": sum(
                1 for r in results if r.escalation_replaced
                and r.trial.expected == "REFUTED"),
            "confirm_eligible": sum(1 for r in hs
                                    if r.verdict == "CONFIRMED"
                                    and not r.escalated_overturn),
            "confirm_withheld": sum(1 for r in results
                                    if r.confirm_withheld),
        },
        "overall": {
            # Which pipeline produced every rate below. Repeated here
            # because `overall` is what gets pasted into comparisons.
            "arm": arm,
            # TPR: corrupted trials refuted.
            "tpr": _overall(("REFUTED",), "REFUTED"),
            # FPR: clean trials refuted — keyed on the REFUTE direction.
            FPR_KEY[arm]: _overall(("CONFIRMED",), "REFUTED"),
            # Corrupted trials CONFIRMED — keyed on the CONFIRM direction,
            # whose whole job is to strip these of actionable confidence.
            CONFIRM_KEY[arm]: _overall(("REFUTED",), "CONFIRMED"),
            # Robustness false alarms: degraded-evidence trials refuted.
            # REFUTE-direction sensitive; `arm` above is what qualifies it.
            "degraded_evidence_fp": _overall(("NOT_REFUTED",), "REFUTED"),
            # §4BF flip (i) V1 liveness, IN the report (R1 C1: without
            # an aggregate, a silently-dead probe minted a fake NULL —
            # per-trial probe_score existed only as buried JSON). Fired
            # = final result carries a probe reading; eligible = final
            # verdict CONFIRMED/REFUTED (the probe's own gate).
            "probe": (lambda _f, _e: {
                "eligible": len(_e),
                "fired": len(_f),
                "fire_rate": (round(len(_f) / len(_e), 3)
                              if _e else None),
                "mean_score": (round(sum(_f) / len(_f), 3)
                               if _f else None),
            })([r.probe_score for r in results
                if r.probe_score is not None
                and r.verdict in ("CONFIRMED", "REFUTED")],
               [r for r in results
                if r.verdict in ("CONFIRMED", "REFUTED")]),
        },
    }


def _keyed_entry(overall: Dict[str, Any], escalated_key: str,
                 raw_key: str, legacy_key: str = ""
                 ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(direction-state, metric block) for one arm-qualified metric.

    Returns "escalated" / "raw" / "unrecorded" — the state of the ONE
    direction that keys this metric, not the whole arm. Looked up by KEY
    NAME, not by iterating arms: two arms map to the same key by design
    (a confirm-only run's FPR is a raw-judge FPR), so an arm-first search
    would report whichever arm happened to be first in the dict.
    """
    if escalated_key in overall:
        return "escalated", overall[escalated_key]
    if raw_key in overall:
        return "raw", overall[raw_key]
    if legacy_key and legacy_key in overall:
        return "unrecorded", overall[legacy_key]
    return "unrecorded", None


def fpr_entry(overall: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(refute-direction state, false-alarm block) for a report's `overall`.

    Tolerates the pre-2026-08-04 shape, whose bare `fpr` key carries no
    arm at all — those reports are raw-judge numbers by construction (no
    bench client had an escalation route before this), but they are
    labelled `unrecorded` rather than back-dated, because "we know what
    it must have been" is how an unlabelled number becomes a comparison.
    """
    return _keyed_entry(overall, "fpr_escalated", "fpr_raw_judge", "fpr")


def false_confirm_entry(overall: Dict[str, Any]
                        ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(confirm-direction state, false-CONFIRM block) for `overall`.

    No legacy key: pre-2026-08-04 reports do not carry this metric at
    all, so there is nothing to be tolerant of and `unrecorded` here means
    genuinely absent.
    """
    return _keyed_entry(overall, "false_confirm_actionable_escalated",
                        "false_confirm_actionable_raw")


# ── Runner ───────────────────────────────────────────────────────────

class HttpChatClient:
    """Single-endpoint OpenAI-compatible chat client — the RAW-JUDGE arm.

    Duck-types the one surface ``Verifier`` needs when no critic pool /
    worker route is present: ``chat_completion(payload)``. It defines
    neither ``critic_clients`` nor ``worker_clients`` truthy and has no
    ``route``, so ``Verifier._call_llm`` goes straight to the direct path
    against the given endpoint.

    ⚠ That also means ``Verifier._escalate_refute`` RETURNS IMMEDIATELY
    (`cheap_route` is False — "the main model already judged it"), so a
    run on this client measures the judge STANDALONE. Production runs
    cheap-judge-screens / main-model-adjudicates, and the main model
    overturned 42 of 50 (84%) of the cheap judge's refutes on the live
    recorded corpus. Use `EscalatingChatClient` for the arm production
    actually runs; `escalation_arm()` labels whichever one is in play and
    `score_trials` refuses to emit an unqualified `fpr` because of it.
    """

    critic_clients: Any = None
    # Declared explicitly, not left to `getattr(..., None)`: this is the
    # attribute `_escalate_refute` reads to decide whether a cheap route
    # existed, so "the raw arm has no worker pool" is a stated contract
    # rather than an accident of which attributes were never set.
    worker_clients: Any = None

    def __init__(self, base_url: str, timeout: float = 90.0,
                 api_key: str = "", model: str = "",
                 cache: Optional[ResponseCache] = None):
        import httpx
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._model = model
        self._timeout = timeout
        self.cache = cache or ResponseCache(None, "off")
        # Public, for `bench_provenance`. The judge IS the system under test,
        # so "which endpoint/model produced these numbers" belongs in the
        # block that decides whether two reports are comparable — and a field
        # that is always "" is worse than an absent one, because it looks
        # like the question was answered.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout,
            headers=headers)

    async def _post_cached(self, client, base_url: str, body: dict,
                           **kw) -> dict:
        """Single choke point for every judge/main call in the bench.

        Both legs go through here so the cache cannot cover one route and
        silently miss the other — a half-cached run would report a hit
        rate that looked fine while still paying (and varying) on the leg
        that mattered.
        """
        key = ResponseCache.key(base_url, body)
        hit = self.cache.get(key, base_url, body)
        if hit is not None:
            return hit
        resp = await client.post("/v1/chat/completions", json=body, **kw)
        resp.raise_for_status()
        data = resp.json()
        self.cache.put(key, base_url, body, data)
        return data

    async def chat_completion(self, payload: dict, **_kw) -> dict:
        # `timeout=` arrives here from `_bounded_fallback_kwargs` and is
        # DELIBERATELY ignored on this client: the raw arm is a harness for
        # measuring one endpoint, so `--timeout` is the operator's patience
        # and must stay effective. (`EscalatingChatClient` honours it,
        # because that client's contract is to reproduce production, whose
        # main leg is bounded by GHOST_VERIFY_FALLBACK_TIMEOUT.) At the
        # default `--timeout 90` the two are the same number anyway.
        body = dict(payload)
        if self._model:
            body.setdefault("model", self._model)
        return await self._post_cached(self._client, self.base_url, body)

    async def aclose(self) -> None:
        await self._client.aclose()


class EscalatingChatClient(HttpChatClient):
    """Two-endpoint client that reproduces PRODUCTION's verify topology.

    Production (measured on the live process 2026-08-04: `--worker-nodes
    http://100.83.184.117:8088|Nova`, `--upstream-url http://127.0.0.1:8088`,
    no `--critic-nodes`) runs verify as TWO models:

      1. the CHEAP judge on the worker route — `Verifier._call_llm` calls
         `route("VERIFY", …)`, which returns the content STRING;
      2. on REFUTED, `_escalate_refute` re-adjudicates on the MAIN model
         via `force_main=True`, which skips both pools and lands on
         `chat_completion`. On disagreement the main model wins.

    So the mapping is exact and needs no new verifier code: `route()` →
    cheap endpoint, `chat_completion()` → main endpoint, and a truthy
    `worker_clients` so `_escalate_refute` sees a cheap route to have
    escalated FROM. Everything else (which template, which timeout,
    force_main semantics) is the verifier's own production code path.

    Failure semantics mirror `LLMClient.route`: it never raises — a failed
    cheap call returns `fallback`, and the verifier then falls through to
    the direct (main) call, exactly as in production.

    SCOPE: this mirrors the REFUTE-direction escalation. `verify_claim`
    also grew a `high_stakes=` CONFIRMED-direction escalation
    (`_escalate_confirm_enabled`, §4J item 2); `run_trials` never passes
    it, so the bench does not exercise that path. Whoever wires it into
    production consumers needs a matching bench knob, or the two arms
    will diverge again in the other direction.
    """

    def __init__(self, base_url: str, main_base_url: str,
                 timeout: float = 90.0, api_key: str = "", model: str = "",
                 main_model: str = "", main_api_key: str = "",
                 main_timeout: Optional[float] = None,
                 leg: str = "worker",
                 cache: Optional[ResponseCache] = None):
        super().__init__(base_url, timeout=timeout, api_key=api_key,
                         model=model, cache=cache)
        # ⚠ WHICH RUNG serves the cheap judge (2026-08-07 review). The
        # 2026-08-04 topology this class was written against ran verify
        # on the WORKER route (45s ceiling); the live process since
        # 2026-08-06 boots `--critic-nodes`, so production's cheap judge
        # rides the CRITIC branch (120s ceiling, one more fallback
        # rung). leg="critic" mirrors that: `critic_clients` truthy and
        # `chat_completion(use_critic=True)` served by the CHEAP
        # endpoint. The default stays "worker" for the unit fixtures;
        # the CLI harness passes the leg that matches the deployment and
        # provenance records it — an unrecorded leg made two reports
        # incomparable without either saying so.
        self.leg = leg if leg in ("worker", "critic") else "worker"
        if self.leg == "critic":
            self.critic_clients = [{"url": base_url.rstrip("/"),
                                    "model": model or "(unnamed)"}]
        import httpx
        headers = {}
        if main_api_key or api_key:
            headers["Authorization"] = f"Bearer {main_api_key or api_key}"
        self.main_base_url = main_base_url.rstrip("/")
        self.main_model = main_model
        self._main_timeout = (main_timeout if main_timeout is not None
                              else timeout)
        self._main_client = httpx.AsyncClient(
            base_url=self.main_base_url, timeout=self._main_timeout,
            headers=headers)
        # Truthy => `_escalate_refute` treats the verdict as cheap-judge
        # output worth re-adjudicating. Shaped like `LLMClient.worker_clients`
        # entries so anything that introspects it gets something real.
        self.worker_clients = [{"url": self.base_url,
                                "model": model or "(unnamed)"}]
        # ROUTE HEALTH. A failed cheap leg is NOT a neutral event here:
        # `route()` returns `fallback` (None), `_call_llm` falls through to
        # the direct call, and that call lands on the MAIN model — so the
        # trial still produces a verdict, but it is the STRONG judge's, not
        # the cheap judge's. Production does the same thing (this is the
        # faithful behaviour), yet a run that silently scores such trials as
        # cheap-judge verdicts is measuring a blend it never names. The live
        # log shows the failure mode is real: 9 `worker node failed — Nova:
        # ReadTimeout` events, 5 of them at exactly +12.0/12.1s, dead on
        # `_ROUTE_TIMEOUT_S`. Verify passes its own 45s budget rather than
        # that 12s ceiling, so the bench's exposure is smaller — but "smaller"
        # is an argument, and this is a measurement.
        self.route_calls = 0
        self.route_failures = 0
        self.route_timeouts = 0
        self.route_empty = 0
        # §4BF flip (i): the score-token probe's calls, counted apart —
        # a failed probe is an advisory miss (the trial's judge verdict
        # is untouched), not a judged-by-MAIN blend event.
        self.probe_calls = 0
        self.probe_failures = 0

    def route_health(self) -> Dict[str, Any]:
        """Cheap-leg call outcomes for the provenance block.

        `fell_through_to_main` is the number the reader actually needs: those
        trials were judged by the MAIN model, so they did not measure the
        cheap judge at all. Probe calls are reported apart and never count
        toward the blend warning (§4BF flip-i R1 M4).
        """
        fell = self.route_failures + self.route_empty
        out = {
            "leg": self.leg,
            "route_calls": self.route_calls,
            "route_failures": self.route_failures,
            "route_timeouts": self.route_timeouts,
            "route_empty_replies": self.route_empty,
            "fell_through_to_main": fell,
            "clean": fell == 0,
        }
        if self.probe_calls or self.probe_failures:
            out["probe_calls"] = self.probe_calls
            out["probe_failures"] = self.probe_failures
        return out

    async def route(self, task: str, payload: dict, max_tokens: int = 128,
                    temperature: float = 0.0, fallback: Any = None,
                    timeout: Optional[float] = None, **_kw) -> Any:
        """The CHEAP leg. Same contract as `LLMClient.route`: size the
        payload, return the extracted content string, and degrade to
        `fallback` (never raise) so the verifier's fall-through is the
        production one."""
        sized = dict(payload)
        sized.setdefault("temperature", temperature)
        sized.setdefault("max_tokens", max_tokens)
        sized["stream"] = False
        if self._model:
            sized.setdefault("model", self._model)
        self.route_calls += 1
        try:
            data = await self._post_cached(
                self._client, self.base_url, sized,
                timeout=(timeout if timeout is not None else self._timeout))
            content = ((data.get("choices") or [{}])[0]
                       .get("message", {}).get("content", ""))
        except Exception as exc:  # noqa: BLE001 — route() never raises
            self.route_failures += 1
            if "timeout" in type(exc).__name__.lower():
                self.route_timeouts += 1
            # WARNING, not debug: this is the event that quietly turns a
            # cheap-judge trial into a main-model one, and a diagnostic
            # nobody sees is how the escalation axis went unmeasured for
            # weeks in the first place.
            logger.warning(
                "verify_bench cheap leg FAILED (%s: %s) — this trial's "
                "verdict will come from the MAIN model, not the judge "
                "under test", type(exc).__name__, exc)
            return fallback
        if not content:
            self.route_empty += 1
            logger.warning(
                "verify_bench cheap leg returned an EMPTY reply — this "
                "trial falls through to the MAIN model")
            return fallback
        return content

    async def chat_completion(self, payload: dict, timeout: Any = None,
                              use_critic: bool = False,
                              use_worker: bool = False, **_kw) -> dict:
        """The MAIN leg — where `force_main=True` lands, i.e. the
        escalation's adjudicator (and the verifier's last-resort fallback
        when the cheap leg returns nothing parseable, same as prod).

        ``use_worker`` (§4BF flip-i R1 MAJ): the score-token probe sends
        ``use_worker=True`` when the client has no critic pool — on
        ``--leg worker`` this used to be swallowed into ``**_kw`` and
        the probe silently rode the MAIN endpoint while the report
        claimed a cheap-pool probe. Both cheap-leg flags now route to
        the cheap endpoint when they match the configured leg.

        PROBE-CALL HEALTH is counted SEPARATELY from route health
        (§4BF flip-i R1 M4): a probe request is identifiable by its
        logprobs fields (nothing else on the cheap leg requests them),
        and a failed probe is an advisory miss — the trial's judge
        verdict replayed fine. Folding probe failures into
        ``route_failures`` branded a valid arm "judged by the MAIN
        model — treat as a blend" while an actually-dead probe showed
        CLEAN route health: the two failure modes were labeled
        backwards.

        With ``leg="critic"``, a ``use_critic=True`` call (the
        verifier's critic-branch judge call) is served by the CHEAP
        endpoint under the verifier's own critic timeout. ⚠ On failure
        it does NOT raise (round-2 review): production's
        `LLMClient._do_chat_completion(use_critic=True)` swallows
        critic-pool exhaustion and answers the SAME critic payload on
        the MAIN upstream (`fell_back_from_node`), never touching the
        worker rung — an earlier version of this method raised, which
        sent the bench down a worker-retry chain with the classic
        2048-token thinking payload that production never runs, and
        double-counted one judge failure as two route failures.

        ``timeout`` is HONOURED, not swallowed. `_bounded_fallback_kwargs`
        passes the verifier's own ceiling here (`GHOST_VERIFY_FALLBACK_
        TIMEOUT`, 90s) whenever the client's `chat_completion` accepts the
        keyword — and a `**kw`-only signature counts as accepting it, so
        the value was arriving and being dropped. Production's LLMClient
        applies it, so an escalated bench run that used the CLI
        `--timeout` instead would be measuring a more (or less) patient
        adjudicator than production has. Both legs of this client are now
        bounded by the verifier exactly as they are live: 45s on the
        route leg, 90s here.
        """
        _is_probe = any(k in payload
                        for k in ("logprobs", "top_logprobs", "n_probs"))
        if (use_critic and self.leg == "critic") \
                or (use_worker and self.leg == "worker"):
            body = dict(payload)
            if self._model:
                body.setdefault("model", self._model)
            if _is_probe:
                self.probe_calls += 1
            else:
                self.route_calls += 1
            try:
                return await self._post_cached(
                    self._client, self.base_url, body,
                    timeout=(timeout if timeout is not None
                             else self._timeout))
            except Exception as exc:
                if _is_probe:
                    self.probe_failures += 1
                    # An advisory probe failing must NOT fall through to
                    # the MAIN model: the verifier treats an exception
                    # as "no probe reading" and keeps the unblended
                    # confidence — answering the probe on the strong
                    # model would both waste a main-slot call and blend
                    # a DIFFERENT model's distribution than registered.
                    raise
                self.route_failures += 1
                if "timeout" in type(exc).__name__.lower():
                    self.route_timeouts += 1
                logger.warning(
                    "verify_bench critic leg FAILED (%s: %s) — answering "
                    "the critic payload on the MAIN endpoint, as "
                    "production's LLMClient does (fell_back_from_node)",
                    type(exc).__name__, exc)
                main_body = dict(payload)
                if self.main_model:
                    main_body.setdefault("model", self.main_model)
                return await self._post_cached(
                    self._main_client, self.main_base_url, main_body)
        body = dict(payload)
        if self.main_model:
            body.setdefault("model", self.main_model)
        kw = {"timeout": timeout} if timeout is not None else {}
        return await self._post_cached(
            self._main_client, self.main_base_url, body, **kw)

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._main_client.aclose()


ARM_ENV = {"two_stage_on": "1", "two_stage_off": "0"}


def verify_claim_accepts_high_stakes(verifier) -> bool:
    """Does this verifier's ``verify_claim`` take ``high_stakes=``?

    The CONFIRM-direction escalation fires ONLY when the caller passes it,
    so a verifier (or a test double) without the parameter makes that
    direction structurally dead — the same shape as the missing worker
    route that made the REFUTE direction dead. Probed rather than assumed,
    and reported in `provenance.escalation`, so the report can never claim
    a direction the call site could not reach. Also keeps `run_trials`
    working against older stubs instead of turning every trial into a
    TypeError the per-trial handler would swallow as an ERROR row.
    """
    fn = getattr(verifier, "verify_claim", None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "high_stakes" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def run_trials(verifier: Verifier, trials: List[BenchTrial],
                     concurrency: int = 1,
                     on_result: Optional[Callable] = None
                     ) -> List[TrialResult]:
    """Run every trial through ``verifier.verify_claim``. Exceptions are
    captured per-trial (error + verdict None), never raised out.

    ``trial.high_stakes`` is forwarded as ``high_stakes=`` when the
    verifier accepts it — without that the CONFIRM-direction escalation
    can never fire, however the client is wired.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[TrialResult]] = [None] * len(trials)
    # Resolved ONCE per run, not per trial: the signature cannot change
    # mid-run and `inspect.signature` is not free.
    pass_stakes = verify_claim_accepts_high_stakes(verifier)

    async def _one(i: int, t: BenchTrial) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                kw = {"high_stakes": t.high_stakes} if pass_stakes else {}
                vr = await verifier.verify_claim(
                    t.claim, t.evidence, t.context, **kw)
                res = TrialResult(
                    trial=t,
                    verdict=vr.verdict.value if vr else None,
                    confidence=vr.confidence if vr else 0.0,
                    reasoning=vr.reasoning if vr else "",
                    issues=list(vr.issues) if vr else [],
                    suspects=vr.suspects if vr else None,
                    elapsed_s=time.monotonic() - t0,
                    escalated_overturn=bool(
                        getattr(vr, "escalated_overturn", False)),
                    escalation_replaced=bool(
                        getattr(vr, "escalation_replaced", False)),
                    objection_dismissed=bool(
                        getattr(vr, "objection_dismissed", False)),
                    objection_upheld=bool(
                        getattr(vr, "objection_upheld", False)),
                    cheap_verdict=getattr(vr, "cheap_verdict", None),
                    cheap_confidence=getattr(vr, "cheap_confidence", None),
                    cheap_issues=getattr(vr, "cheap_issues", None),
                    confirm_withheld=bool(
                        getattr(vr, "confirm_withheld", False)),
                    escalation_downgraded=bool(
                        getattr(vr, "escalation_downgraded", False)),
                    truncation_guarded=bool(
                        getattr(vr, "truncation_guarded", False)),
                    probe_score=getattr(vr, "probe_score", None),
                )
            except Exception as exc:
                res = TrialResult(
                    trial=t, verdict=None,
                    elapsed_s=time.monotonic() - t0, error=str(exc))
            results[i] = res
            if on_result:
                on_result(res)

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(trials)))
    return [r for r in results if r is not None]


def _judge_identity(verifier) -> Dict[str, str]:
    """Endpoint + model of the judge actually under test ({} when unknown).

    Walks to the client, because `Verifier` holds no endpoint of its own.
    Returns `{"unresolved": ...}` rather than empty strings when it cannot
    tell — a provenance block must never quietly assert "no model".
    """
    client = getattr(verifier, "llm_client", None)
    if client is None:
        return {"unresolved": "verifier has no llm_client"}
    for url_attr, model_attr in (("base_url", "model"),
                                 ("_base_url", "_model")):
        url = getattr(client, url_attr, "")
        model = getattr(client, model_attr, "")
        if url or model:
            return {"base_url": str(url or ""), "model": str(model or "")}
    inner = getattr(client, "_client", None)
    url = str(getattr(inner, "base_url", "") or "")
    model = str(getattr(client, "_model", "") or "")
    if url or model:
        return {"base_url": url, "model": model}
    return {"unresolved": type(client).__name__}


def escalation_arm(verifier,
                   trials: Optional[List[BenchTrial]] = None
                   ) -> Dict[str, Any]:
    """Which VERDICT PIPELINE this run measures — the report's load-bearing
    provenance field.

    Recomputes exactly the predicates `Verifier._escalate_refute` and
    `_escalate_confirm` apply (kill switch enabled AND a truthy
    `critic_clients`/`worker_clients` on the client; the confirm direction
    additionally needs `verify_claim` to accept `high_stakes=` and at
    least one high-stakes trial to exercise it), because the bench's job
    is to say what it measured, not what it was configured to measure.
    `tests/test_verify_bench_escalation_arm.py` fences these against the
    verifier by OBSERVING call sequences through it, so a change to either
    predicate fails a test instead of silently mislabelling a report.

    Why it matters (§4J items 1 and 2). Escalation is TWO-DIRECTIONAL and
    each direction was dark in the bench for its own reason:

      * REFUTE — with a single-endpoint client `_escalate_refute` is a
        no-op. Measured on `$GHOST_HOME/system/llm_recordings`
        (2026-07-30..08-04): of 50 recorded cheap-judge REFUTEs that
        reached the main model, 42 (84%) were overturned to CONFIRMED;
        independently the durable log's `GhostAgent` lines read 80
        overturned / 19 stood = 81%. A raw arm's false-alarm rate is an
        UPPER BOUND on production's.
      * CONFIRM — `run_trials` never passed `high_stakes=`, so
        `_escalate_confirm` could not fire at all even with the two-leg
        client installed. Re-measured independently over the same day-files
        (2026-08-04, and it reproduces exactly): 62 CONFIRMED / 68 REFUTED
        cheap verdicts = 130, of which **0 sit below the 0.7 actionable
        gate** — the distribution is 1.0 (79), 0.95 (32), 0.9 (19), i.e.
        the cheap judge never expresses uncertainty, that gate filters
        nothing, and every cheap CONFIRMED was consumed unconditionally.
        Capping a withheld confirmation to 0.6 is therefore the ONLY thing
        that can make the gate bind.

    ``trials`` is optional; pass it and the block also records how many
    trials could exercise the confirm direction. A run with zero
    high-stakes trials carries NO evidence about that direction, whatever
    the arm says, and the report says so out loud.
    """
    from ..core.verifier import (_escalate_confirm_enabled,
                                 _escalate_refute_enabled)

    client = getattr(verifier, "llm_client", None)
    cheap = None
    if getattr(client, "critic_clients", None):
        cheap = "critic"
    elif getattr(client, "worker_clients", None):
        cheap = "worker"

    ref_enabled = bool(_escalate_refute_enabled())
    con_enabled = bool(_escalate_confirm_enabled())
    stakes_ok = verify_claim_accepts_high_stakes(verifier)
    n_hs = (sum(1 for t in trials if t.high_stakes)
            if trials is not None else None)

    refute_live = bool(cheap) and ref_enabled
    confirm_live = bool(cheap) and con_enabled and stakes_ok
    arm = arm_for(refute_live, confirm_live)

    def _why(enabled: bool, switch: str, extra: str = "") -> str:
        why = []
        if not enabled:
            why.append(f"{switch} is off")
        if not cheap:
            why.append("the client exposes no cheap route "
                       "(critic_clients/worker_clients falsy), so the "
                       "escalation returns immediately")
        if extra:
            why.append(extra)
        return "; ".join(why)

    out: Dict[str, Any] = {
        "arm": arm,
        "cheap_route": cheap,
        "judge": _judge_identity(verifier),
        "directions": {
            "refute": {
                "live": refute_live,
                "enabled": ref_enabled,
                "effect": "main model re-adjudicates a cheap REFUTED and "
                          "its verdict wins — moves FPR and TPR",
                "why_not": "" if refute_live else _why(
                    ref_enabled, "GHOST_VERIFY_ESCALATE_REFUTE"),
            },
            "confirm": {
                "live": confirm_live,
                "enabled": con_enabled,
                "high_stakes_supported": stakes_ok,
                "high_stakes_trials": n_hs,
                "effect": "main model re-adjudicates a HIGH-STAKES cheap "
                          "CONFIRMED; the verdict never flips, confidence "
                          "is capped to 0.6 — moves the actionable "
                          "false-CONFIRM rate only",
                "why_not": "" if confirm_live else _why(
                    con_enabled, "GHOST_VERIFY_ESCALATE_CONFIRM",
                    "" if stakes_ok else
                    "verify_claim does not accept high_stakes=, so "
                    "_escalate_confirm can never be reached"),
            },
        },
        # Retained for readers written against the one-directional block.
        "escalate_refute_enabled": ref_enabled,
    }

    if arm == ARM_ESCALATED:
        measures = ("the verdict production acts on: cheap judge screens, "
                    "the main model re-adjudicates every REFUTED and every "
                    "high-stakes CONFIRMED")
    elif arm == ARM_ESC_REFUTE:
        measures = ("only HALF of production's escalation — refutes are "
                    "re-adjudicated, high-stakes CONFIRMEDs are not, so "
                    "the actionable false-CONFIRM rate here is an upper "
                    "bound on production's")
    elif arm == ARM_ESC_CONFIRM:
        measures = ("only HALF of production's escalation — high-stakes "
                    "CONFIRMEDs are re-adjudicated, refutes are not, so "
                    "the false-alarm rate here is an upper bound on "
                    "production's")
    else:
        measures = ("the CHEAP JUDGE STANDALONE. Production re-adjudicates "
                    "every REFUTED on the main model and overturned "
                    f"{ESCALATION_OVERTURN_MEASURED}, so false-alarm rates "
                    "here are an upper bound on production's")
    out["measures"] = measures

    if refute_live or confirm_live:
        out["main"] = {
            "base_url": str(getattr(client, "main_base_url", "") or ""),
            "model": str(getattr(client, "main_model", "") or ""),
        }
        if not out["main"]["base_url"]:
            # A client that claims a cheap route but exposes no main
            # endpoint escalates onto whatever `chat_completion` is — the
            # provenance must not imply it knows which model that was.
            out["main"] = {"unresolved": type(client).__name__}
        elif out["main"]["base_url"] == str(getattr(client, "base_url", "")):
            # Legitimate ablation ("does a second look help at all?"), but
            # it is NOT production's shape — production's strong leg is a
            # different, larger model.
            out["same_endpoint"] = True
    if arm != ARM_ESCALATED:
        # Kept under the old name so existing readers of `why_raw` still
        # get a reason string; it now covers "half-escalated" too.
        out["why_raw"] = "; ".join(
            f"{d}: {out['directions'][d]['why_not']}"
            for d in ("refute", "confirm")
            if out["directions"][d]["why_not"])
    if confirm_live and n_hs == 0:
        out["confirm_unexercised"] = (
            "the CONFIRM direction is live but NO trial is high-stakes, so "
            "this run measured nothing about it")
    health = getattr(client, "route_health", None)
    if callable(health):
        try:
            out["route_health"] = health()
        except Exception:  # noqa: BLE001 — provenance must not break a run
            pass
    return out


async def run_bench(cases: List[BenchCase],
                    verifier: Verifier,
                    arms: Optional[List[str]] = None,
                    fault_names: Optional[List[str]] = None,
                    seed: int = 0,
                    concurrency: int = 1,
                    on_result: Optional[Callable] = None,
                    logit_expect: Optional[str] = None) -> Dict[str, Any]:
    """Full bench: build trials once, run them under each arm
    (two-stage on / off, toggled via GHOST_VERIFY_TWO_STAGE, which the
    verifier reads per call), score, and return the report dict.

    ``logit_expect``: "on"/"off" pins GHOST_VERIFY_LOGIT_EXPECT for the
    whole run (restored afterwards); None inherits the shell, same as
    every other discipline flag. The provenance block is built first and
    verify_flags is RE-STAMPED with the pinned value (R4: an earlier
    docstring claimed the pin landed before provenance — the re-stamp
    comment at the pin site is the accurate account).
    """
    arms = arms or ["two_stage_on", "two_stage_off"]
    unknown = [a for a in arms if a not in ARM_ENV]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")
    trials = build_trials(cases, fault_names=fault_names, seed=seed)
    # Resolved ONCE, before any trial runs, and threaded into both the
    # provenance block and every metric key. `_escalate_refute_enabled()`
    # is read per verify call, so a mid-run flag flip would split the
    # results across two pipelines — pinning it here at least makes the
    # report state which one it claims to be, and the env-var arm loop
    # below never touches the escalation switch.
    esc = escalation_arm(verifier, trials)
    report: Dict[str, Any] = {
        "n_cases": len(cases),
        "n_trials": len(trials),
        "seed": seed,
        "actionable_conf": ACTIONABLE_CONF,
        # PROVENANCE — what this run actually measured.
        #
        # Without it, two runs were compared on nothing but a file PATH: the
        # T0 bundle records n_cases=13/n_trials=97 against a case file that
        # later held 21 cases, and template identity silently depends on
        # $GHOST_HOME (unset -> the BASELINE template is used instead of the
        # deployed artifact). A verifier verdict that "carries the §4F weight"
        # cannot rest on that. Compare two reports only when `provenance`
        # matches.
        # The judge lives on the CLIENT, not on the Verifier — reading it off
        # `verifier` produced `{"base_url": "", "model": ""}` for a fully
        # configured run, i.e. a provenance field that always claimed to
        # answer a question it never answered.
        "provenance": bench_provenance(
            cases, fault_names=fault_names,
            judge=_judge_identity(verifier), escalation=esc,
            cache=getattr(getattr(verifier, "llm_client", None),
                          "cache", None)),
        "arms": {},
    }
    prev = os.environ.get("GHOST_VERIFY_TWO_STAGE")
    # Same save/set/restore discipline as TWO_STAGE. Armed here — after
    # the provenance block above was built — would record the SHELL'S
    # value, not the run's; so re-stamp verify_flags after pinning.
    prev_le = os.environ.get("GHOST_VERIFY_LOGIT_EXPECT")
    if logit_expect not in (None, "on", "off"):
        raise ValueError(f"logit_expect must be on/off/None: {logit_expect!r}")
    try:
        if logit_expect is not None:
            os.environ["GHOST_VERIFY_LOGIT_EXPECT"] = (
                "1" if logit_expect == "on" else "0")
            report["provenance"]["verify_flags"]["GHOST_VERIFY_LOGIT_EXPECT"] \
                = os.environ["GHOST_VERIFY_LOGIT_EXPECT"]
        for arm in arms:
            os.environ["GHOST_VERIFY_TWO_STAGE"] = ARM_ENV[arm]
            results = await run_trials(
                verifier, trials, concurrency=concurrency,
                on_result=on_result)
            report["arms"][arm] = {
                "metrics": score_trials(results, arm=esc["arm"]),
                "trials": [r.to_dict() for r in results],
            }
    finally:
        if prev is None:
            os.environ.pop("GHOST_VERIFY_TWO_STAGE", None)
        else:
            os.environ["GHOST_VERIFY_TWO_STAGE"] = prev
        if logit_expect is not None:
            if prev_le is None:
                os.environ.pop("GHOST_VERIFY_LOGIT_EXPECT", None)
            else:
                os.environ["GHOST_VERIFY_LOGIT_EXPECT"] = prev_le
    # Route health is only meaningful AFTER the trials have run — the block
    # built above was for the arm, whose counters were all zero then.
    _post = escalation_arm(verifier, trials).get("route_health")
    if _post is not None:
        report["provenance"]["escalation"]["route_health"] = _post
        if not _post.get("clean"):
            logger.warning(
                "verify_bench: %d cheap-leg call(s) fell through to the MAIN "
                "model (%d timeouts) — those trials did not measure the "
                "judge under test",
                _post.get("fell_through_to_main"), _post.get("route_timeouts"))
    # Same reason route_health is re-read here: the provenance block is
    # built BEFORE any trial runs, so a snapshot taken there records the
    # state of a run that has not happened yet. A fully replayed report
    # said `hits: 0, measures: "live judge"` until this line existed.
    # ⚠ Kept OUTSIDE the `_post is not None` branch: the raw arm has no
    # route health, and nesting the cache refresh under it (as a first
    # edit did) both crashed that arm and would have left raw-arm reports
    # with a pre-run cache snapshot.
    _cache = getattr(getattr(verifier, "llm_client", None), "cache", None)
    if _cache is not None:
        report["provenance"]["cache"] = _cache.stats()
        if _cache.strict_misses:
            logger.warning(
                "verify_bench: %d STRICT CACHE MISSES — this run is NOT a "
                "clean replay. Each miss is swallowed into a null verdict, "
                "so the scores below are computed over fewer trials than "
                "they appear to be.", len(_cache.strict_misses))
    return report


def bench_provenance(cases: List[BenchCase],
                     fault_names: Optional[List[str]] = None,
                     judge: Optional[Dict[str, Any]] = None,
                     escalation: Optional[Dict[str, Any]] = None,
                     cache: Optional[ResponseCache] = None
                     ) -> Dict[str, Any]:
    """Everything that must match before two bench reports are comparable.

    ``escalation`` (from `escalation_arm`) is part of that list: two runs
    against the same pool, faults and judge are STILL not comparable if
    one measured the cheap judge standalone and the other measured
    judge+escalation. Absent, it records `unrecorded` rather than
    assuming — an unlabelled arm is the defect this field exists to stop.
    """
    case_digest = hashlib.sha256()
    # Sorted by the FULL identity, not by `case_id` alone. A duplicate
    # case_id (which a stale mined pool produced) made the sort order — and
    # therefore the digest — depend on input ordering, so the one field whose
    # entire job is deciding "are these two runs comparable?" disagreed with
    # itself on identical case sets.
    for c in sorted(cases, key=lambda x: (x.case_id, x.claim, x.evidence,
                                          x.context)):
        case_digest.update(
            (c.case_id + "\x00" + c.claim + "\x00" + c.evidence + "\x00"
             + c.context + "\x00").encode("utf-8"))
    _ids = [c.case_id for c in cases]
    prov: Dict[str, Any] = {
        "cases_sha256": case_digest.hexdigest()[:16],
        "n_cases": len(cases),
        "duplicate_case_ids": len(_ids) - len(set(_ids)),
        # The fault library IS part of the experiment: a report built from a
        # different fault set is not comparable, and the names appeared
        # nowhere in the report at all.
        "faults": sorted(fault_names) if fault_names else sorted(FAULTS),
        "faults_sha256": hashlib.sha256(
            "|".join(sorted(fault_names) if fault_names else sorted(FAULTS)
                     ).encode("utf-8")).hexdigest()[:16],
        # The judge is the system under test; it was recorded outside
        # `provenance`, so a provenance match could still be a different model.
        "judge": dict(judge or {}),
        # WHICH PIPELINE ran: raw cheap judge, or judge+escalation (what
        # production acts on). Two reports whose `escalation.arm` differs
        # are measuring two different systems.
        "escalation": dict(escalation) if escalation else {
            "arm": "unrecorded",
            "why_raw": "run_bench was not given an escalation_arm() block",
        },
        "ghost_home": os.environ.get("GHOST_HOME") or "<unset>",
        # ⚠ The discipline flags SELECT the pipeline (2026-08-07 review):
        # run_bench arms only GHOST_VERIFY_TWO_STAGE itself — everything
        # else is inherited from the operator's shell, so two reports
        # with byte-identical provenance could be two different systems.
        # "<unset>" is recorded distinctly from an explicit value: unset
        # means "whatever the code default was at that commit".
        "verify_flags": {
            k: os.environ.get(k, "<unset>")
            for k in ("GHOST_VERIFY_OBJECTION_CHECK",
                      "GHOST_VERIFY_TRUNCATION_GUARD",
                      "GHOST_VERIFY_TRUNCATION_MIN_SEVERITY",
                      "GHOST_VERIFY_OBJECTION_DISMISS",
                      "GHOST_VERIFY_OVERTURN_QUOTE",
                      "GHOST_VERIFY_TIER_ROUTING",
                      "GHOST_VERIFY_ESCALATE_REFUTE",
                      "GHOST_VERIFY_ESCALATE_CONFIRM",
                      "GHOST_VERIFY_TWO_STAGE",
                      # §4BF Track 2 flip (i): the logit-expectation probe
                      # was INVISIBLE to provenance — a probe-on vs
                      # probe-off pair produced byte-identical fingerprints,
                      # so verify_bench_status called a probe-flipped live
                      # system "unchanged" and the comparator demanded no
                      # --expect-differs declaration.
                      "GHOST_VERIFY_LOGIT_EXPECT",
                      # (LOGIT_EXPECT_WEIGHT was retired with the §4BL
                      # cap — a dead flag recorded as pipeline-selecting
                      # would read as phantom drift.)
                      # §4BK R2 — same defect class, found by the review
                      # while the flags were the very thing under test:
                      # the escalated-adjudicator gate and the §4BJ stop
                      # hold SELECT the MAIN-leg pipeline, and neither
                      # was recorded — the §4BK A/B's own config-equality
                      # check was trivially blind to them.
                      "GHOST_VERIFY_MAIN_TWO_STAGE",
                      "GHOST_VERIFY_MAIN_STAGE_STOP",
                      # §4BL: the CONFIRM-cap selectors, recorded from
                      # birth (the invisible-flag class, third strike).
                      "GHOST_VERIFY_PROBE_CAP_T",
                      "GHOST_VERIFY_PROBE_CONF_CAP")
        },
        # ⚠ THE GAP THIS CLOSES (2026-08-09): templates and flags were
        # recorded, but the CODE was not. A change to `_escalate_refute`,
        # the truncation guard or the objection parser moves every number
        # in the report while leaving provenance byte-identical — so two
        # genuinely different systems compared as if they were one. Split
        # in two on purpose: `verifier` changing means the SYSTEM changed
        # (a result); `bench` changing means the RULER changed (not a
        # result). One digest could not tell those apart.
        "code": {name: semantic_code_digest(paths)
                 for name, paths in verify_path_sources().items()},
        # Whether the judge actually answered, or was replayed from cache.
        # A replayed report measures CODE against a frozen judge and must
        # never be quoted as "the number today".
        "cache": (cache.stats() if cache is not None
                  else {"mode": "off", "measures": "live judge"}),
        "templates": {},
    }
    try:
        from ..core import verifier as _v
        for name, base in (
            ("verifier.enumerate", _v._VERIFY_ENUMERATE_PROMPT),
            ("verifier.adjudicate", _v._VERIFY_ADJUDICATE_PROMPT),
            # The CLASSIC template — the ONLY one the `two_stage_off` arm
            # uses. Half the report's arms ran on a template whose hash the
            # provenance block never recorded.
            ("verifier.claim", _VERIFY_CLAIM_PROMPT),
        ):
            live = _v._stage_template(name, base)
            prov["templates"][name] = {
                "sha256": hashlib.sha256(live.encode("utf-8")).hexdigest()[:16],
                "chars": len(live),
                "tuned": live != base,
            }
    except Exception as exc:  # noqa: BLE001 — provenance must not break a run
        prov["templates"] = {"error": f"{type(exc).__name__}: {exc}"}
    return prov


# Labels are keyed on the DIRECTION STATE returned by `fpr_entry` /
# `false_confirm_entry` ("escalated" / "raw" / "unrecorded"), not on the
# arm: two arms share one FPR key by design.
# No nested `**` in any of these: the caller already wraps the label in
# bold, and a nested pair renders as literal asterisks in the .md the
# journal gets pasted from.
_FPR_LABEL = {
    "escalated": "FPR (clean refuted — refute escalation LIVE, "
                 "production-equivalent)",
    "raw": "FPR-RAW (clean refuted by the CHEAP JUDGE ALONE — "
           "NOT a production FPR)",
    "unrecorded": "FPR (arm UNRECORDED — pre-2026-08-04 report, "
                  "not comparable to a labelled one)",
}
_CONFIRM_LABEL = {
    "escalated": "false-CONFIRM @≥0.7 (corrupted trials passed — confirm "
                 "escalation LIVE, production-equivalent)",
    "raw": "false-CONFIRM-RAW @≥0.7 (corrupted trials passed by the CHEAP "
           "JUDGE ALONE — NOT a production rate)",
    "unrecorded": "false-CONFIRM (not measured by this report)",
}


def render_report_md(report: Dict[str, Any]) -> str:
    """Human-readable summary: one table per arm plus the headline
    TPR/FPR line — paste-able into PROJECT_JOURNAL.md.

    The false-alarm headline is LABELLED WITH ITS ARM and, on the raw
    arm, refuses the name "FPR" outright: a cheap-judge false-alarm rate
    is an upper bound on production's, and printing the two under one
    name is how the §4J item-1 gap survived a written report.
    """
    esc = (report.get("provenance") or {}).get("escalation") or {}
    esc_arm = str(esc.get("arm") or "unrecorded")
    lines = [
        "# Verifier fault-injection bench",
        "",
        f"cases: {report['n_cases']} · trials/arm: {report['n_trials']}"
        f" · seed: {report['seed']}"
        f" · actionable conf ≥ {report['actionable_conf']}",
        "",
        f"**verdict pipeline measured: `{esc_arm}`**"
        + (f" — cheap route `{esc.get('cheap_route')}` "
           f"{(esc.get('judge') or {}).get('base_url', '?')}"
           f" → main {(esc.get('main') or {}).get('base_url', '?')}"
           if esc.get("main") else ""),
        "",
    ]
    _dirs = esc.get("directions") or {}
    if _dirs:
        lines += [
            "| escalation direction | live | moves |",
            "|---|---|---|",
        ] + [
            f"| {name} | {'YES' if d.get('live') else 'no'}"
            + (f" ({d['why_not']})" if not d.get("live")
               and d.get("why_not") else "")
            + f" | {d.get('effect', '')} |"
            for name, d in _dirs.items()
        ] + [""]
    if esc_arm != ARM_ESCALATED:
        lines += [
            f"> ⚠ this run measures {esc.get('measures') or 'an UNRECORDED arm'}"
            + (f" ({esc.get('why_raw')})" if esc.get("why_raw") else "")
            + ". Do not quote the numbers below as production rates, and "
              "do not compare them with a `judge+escalation` report.",
            "",
        ]
    if esc.get("confirm_unexercised"):
        lines += [f"> ⚠ {esc['confirm_unexercised']}.", ""]
    _rh = esc.get("route_health") or {}
    if _rh and not _rh.get("clean"):
        lines += [
            f"> ⚠ **{_rh.get('fell_through_to_main')} of "
            f"{_rh.get('route_calls')} cheap-leg calls FAILED** "
            f"({_rh.get('route_timeouts')} timeouts, "
            f"{_rh.get('route_empty_replies')} empty replies). Each one "
            f"fell through to the MAIN model, so those trials were judged "
            f"by the strong model rather than the judge under test — "
            f"production-faithful, but they are not cheap-judge "
            f"measurements. Treat the rates below as a blend until the "
            f"judge node is healthy.",
            "",
        ]
    elif _rh:
        lines += [f"cheap leg: {_rh.get('route_calls')} calls, 0 failures.",
                  ""]
    if esc.get("same_endpoint"):
        lines += [
            "> ⚠ the escalation's main leg is the SAME endpoint as the "
            "judge. That is a second-look ablation, not production's "
            "shape (production escalates a small judge to a larger model).",
            "",
        ]
    for arm, data in report["arms"].items():
        m = data["metrics"]
        o = m["overall"]
        fpr_state, fpr = fpr_entry(o)
        con_state, con = false_confirm_entry(o)
        ev = m.get("escalation_events") or {}
        lines.append(f"## {arm}")
        lines.append("")
        lines.append(
            f"**TPR (catch rate)**: {o['tpr']['rate']} raw / "
            f"{o['tpr']['rate_actionable']} actionable "
            f"({o['tpr']['judged']}/{o['tpr']['n']} judged) — "
            f"**{_FPR_LABEL.get(fpr_state, fpr_state)}**: "
            f"{(fpr or {}).get('rate')} raw / "
            f"{(fpr or {}).get('rate_actionable')} actionable — "
            f"**degraded-evidence FP**: "
            f"{o['degraded_evidence_fp']['rate']}")
        lines.append("")
        if con is not None:
            lines.append(
                f"**{_CONFIRM_LABEL.get(con_state, con_state)}**: "
                f"{con.get('rate_actionable')} "
                f"({con.get('rate')} at any confidence, "
                f"{con.get('judged')}/{con.get('n')} judged)")
            lines.append("")
        _pb = o.get("probe") or {}
        if _pb.get("fired"):
            lines.append(
                f"**score-token probe**: fired on {_pb['fired']}/"
                f"{_pb['eligible']} eligible trials "
                f"(rate {_pb['fire_rate']}, mean score "
                f"{_pb['mean_score']}) — the probe fires only inside "
                f"the two-stage path, so WHEN probe transport is clean "
                f"(provenance route_health.probe_failures == 0) a rate "
                f"far below 1.0 is measuring the pipeline's "
                f"classic-fallback share, not probe health (measured "
                f"48% on the standing pool, §4BF flip i run log); with "
                f"probe failures present, both causes are live.")
            lines.append("")
        if ev:
            lines.append(
                f"escalation events — high-stakes trials: "
                f"{ev.get('high_stakes_trials')} · refutes overturned: "
                f"{ev.get('refute_overturned')} · confirms withheld: "
                f"{ev.get('confirm_withheld')} of "
                f"{ev.get('confirm_eligible')} eligible")
            # The escalation-discipline ship-gate numbers (2026-08-06) —
            # the journal points decisions at these, so they must be in
            # the RENDERED report, not only the JSON.
            if any(k in ev for k in ("overturn_rescues", "downgrades")):
                lines.append(
                    f"discipline — overturn rescues: "
                    f"{ev.get('overturn_rescues', 0)} · overturn DAMAGE: "
                    f"{ev.get('overturn_damage', 0)} · downgrades: "
                    f"{ev.get('downgrades', 0)} "
                    f"(rescues {ev.get('downgrade_rescues', 0)} / damage "
                    f"{ev.get('downgrade_damage', 0)})")
            # ⚠ Every mechanism family renders, or it reads as "never
            # fired" (round-2 review): the objection check is the ONE
            # discipline that ships ON, and it was the one family
            # missing from the rendered report — the exact
            # invisible-mechanism shape this bench exists to prevent.
            if any(k in ev for k in ("objection_dismissed",
                                     "truncation_guarded",
                                     "replaced_uncertain")):
                lines.append(
                    f"mechanical — objection dismissed: "
                    f"{ev.get('objection_dismissed', 0)} "
                    f"(rescues {ev.get('objection_dismiss_rescues', 0)} / "
                    f"damage {ev.get('objection_dismiss_damage', 0)}) · "
                    f"objection upheld: {ev.get('objection_upheld', 0)} "
                    f"(protects {ev.get('objection_uphold_protects', 0)} / "
                    f"damage {ev.get('objection_uphold_damage', 0)}) · "
                    f"truncation guard: {ev.get('truncation_guarded', 0)} "
                    f"(rescues {ev.get('truncation_guard_rescues', 0)} / "
                    f"damage {ev.get('truncation_guard_damage', 0)}) · "
                    f"replaced-uncertain: {ev.get('replaced_uncertain', 0)} "
                    f"(rescues {ev.get('replaced_rescues', 0)} / damage "
                    f"{ev.get('replaced_damage', 0)})")
            lines.append("")
        lines.append("| fault | expected | n | judged | skipped | "
                     "confirmed | refuted | uncertain | rate | "
                     "actionable | mean conf |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for fault, e in m["per_fault"].items():
            rate = e.get("catch_rate", e.get("false_alarm_rate"))
            rate_act = e.get("catch_rate_actionable",
                             e.get("false_alarm_rate_actionable"))
            lines.append(
                f"| {fault} | {e['expected']} | {e['n']} | {e['judged']} "
                f"| {e['skipped']} | {e['confirmed']} | {e['refuted']} "
                f"| {e['uncertain']} | {rate} | {rate_act} "
                f"| {e['mean_confidence']} |")
        lines.append("")
    return "\n".join(lines)
