"""§4CU — rubric grading for the turns the verifier DECLINES.

THE POPULATION. Measured 2026-08-24 over `user_request` turns with the
corrections overlay applied: tool-using turns are 80.7% decided over the
last 14 days, chat turns 30.4%. The undecided remainder is dominated by
turns that ran no tool at all — nothing was executed, so there is nothing
for an evidence-packing verifier to check, and `_find_substantive_tool_
for_verifier` correctly returns None. §4CS item D established there are
ZERO lost verdicts in that bucket: every one is a mechanism working as
designed. The verifier is not failing on chat, it is DECLINING.

WHY A RUBRIC. Declining is right for a binary evidence check and wrong as
the last word, because "was this a good answer to THIS request" is a
multi-criteria judgement, not a claim to be refuted. That is exactly the
gap Rubrics-as-Rewards (arXiv:2507.17746, ICLR 2026) measures: structured
checklist rewards beat direct Likert LLM-judge scoring by up to 31% on
HealthBench, and — the part that matters for a local 35B judge — they
"yield better alignment for smaller judges and reduce performance variance
across judge scales".

⚠⚠ WHY THIS SHIPS IN SHADOW, AND WHY THAT IS NOT TIMIDITY.

A rubric IS a checklist injected into a judging prompt, and this project
has already been burned by exactly that shape. §4BE: the checklist nudge
asserted the user had given learning instructions; **59 of 59 arming turns
had none — 100% false positive — it fired 39 times, and the model reasoned
"there's no explicit learning instruction in their message… But the system
is requiring it", complied, and minted a lesson that vector-dedup
reinforced to freq=11.** A checklist the model feels obliged to satisfy
manufactures the evidence it asks for.

So the contract here is structural, not aspirational:

* **This module CANNOT write an outcome label.** It writes one file,
  `system/verifier/rubric_shadow.jsonl`, which no learning path reads. The
  only way its verdicts ever become labels is a future, separate change
  made against measured agreement with the human channel.
* **The rubric is synthesised from the REQUEST ALONE.** `build_rubric()`
  is not given the response, at all, as a parameter. A criterion written
  while looking at the answer is a criterion written to fit it — the
  "predict the label from the label" failure that made "any tool errored"
  the strongest and most useless confidence feature (AUC 0.229).
* **ABSTAIN is a real outcome, never a fabricated neutral.** "thanks!"
  has no checkable content and scores nothing. Recording 0.5 for it would
  put a zero-variance column into a corpus, which is precisely how
  `w_entropy` stayed pinned at 0 across 1200 samples.
* **It carries an EPOCH.** Change the prompt or the scale and old rows
  stop being comparable; `RUBRIC_EPOCH` is what lets the agreement
  reporter drop them instead of pooling eras (the Simpson's-paradox trap
  that forced a negative Platt slope on the calibration corpus).

Kill switch / flip:
  GHOST_RUBRIC_SHADOW=1   — enable (DEFAULT OFF). Nothing here becomes
                            autonomous by having been built.
  GHOST_RUBRIC_MAX_ITEMS  — criteria cap (default 7)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Bump on ANY change to the prompt, the scale, or the aggregation. Rows
#: from a different epoch are not comparable and the reporter drops them.
RUBRIC_EPOCH = "r1"

SHADOW_REL = "system/verifier/rubric_shadow.jsonl"

#: Fewer than this and the rubric is not a rubric. A one-item checklist
#: is a Likert score wearing a list, which is the baseline RaR beats — and
#: a ZERO-item one would score 1.0 vacuously, passing every response by
#: having asked nothing. That failure is silent and flattering, so the
#: floor is enforced as an ABSTAIN rather than a clamp.
MIN_CRITERIA = 3

#: Bounded so one turn cannot spend a judging budget, and because RaR's
#: own instance-specific rubrics sit in this range.
DEFAULT_MAX_CRITERIA = 7

PASS, FAIL, NA = "pass", "fail", "na"
ABSTAIN = "abstain"
GRADED = "graded"


def shadow_enabled() -> bool:
    """OFF unless explicitly switched on. Read at CALL time, not import
    time, so a flip does not require a restart to take effect in a
    long-running process — and so tests do not need module reloading,
    which rebinds run-wide (`reload-contaminates-the-session`)."""
    # ⚠ Lowercased. Case-sensitive first, and a test caught it: an
    # operator writing `GHOST_RUBRIC_SHADOW=TRUE` got a feature that was
    # silently OFF while the launcher line said it was on — the failure
    # mode where the flag and the behaviour disagree and nothing says so.
    return str(os.environ.get("GHOST_RUBRIC_SHADOW", "")).strip().lower() in (
        "1", "true", "yes", "on")


def _max_criteria() -> int:
    try:
        v = int(os.environ.get("GHOST_RUBRIC_MAX_ITEMS", DEFAULT_MAX_CRITERIA))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CRITERIA
    return max(MIN_CRITERIA, min(20, v))


@dataclass
class RubricVerdict:
    """One shadow judgement.

    `status` is ABSTAIN or GRADED — never a third thing, and never a
    score standing in for "could not tell". `score` is None on ABSTAIN,
    which is the invariant that keeps a fabricated neutral out of any
    corpus that later consumes this.
    """
    status: str = ABSTAIN
    score: Optional[float] = None
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    n_pass: int = 0
    n_fail: int = 0
    n_na: int = 0
    reason: str = ""
    epoch: str = RUBRIC_EPOCH

    def to_row(self, **extra) -> Dict[str, Any]:
        row = {
            # `is None`, not `or`: a caller passing ts=0.0 (epoch, and
            # what a zeroed clock produces) had it silently replaced by
            # the wall clock, which is a fabricated timestamp on a row
            # whose whole value is being joinable and orderable.
            "ts": (lambda t: time.time() if t is None else t)(
                extra.pop("ts", None)),
            "epoch": self.epoch,
            "status": self.status,
            "score": self.score,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_na": self.n_na,
            "reason": self.reason,
            "criteria": self.criteria,
        }
        row.update(extra)
        return row


# ══════════════════════════════════════════════════════════════════════
# 1. Rubric synthesis — FROM THE REQUEST ONLY
# ══════════════════════════════════════════════════════════════════════

_BUILD_PROMPT = """\
You are writing an evaluation rubric for a request that has ALREADY been \
answered by someone else. You will NOT see their answer.

Write {lo}-{hi} criteria that any good answer to this request must \
satisfy. Each criterion must be:
  - OBSERVABLE: decidable by reading the answer alone, with no outside \
lookup and no judgement of style or tone;
  - ATOMIC: one property, so it can be answered yes or no;
  - DERIVED FROM THE REQUEST, never assumed about the answer.

If the request is small talk, an acknowledgement, or carries no checkable \
content at all, return an EMPTY list. Returning criteria for such a \
request is worse than returning none.

REQUEST:
{request}

Reply with JSON only:
{{"criteria": [{{"id": "c1", "criterion": "..."}}]}}"""


_GRADE_PROMPT = """\
Grade the ANSWER against each criterion. Judge only what the answer \
actually says. An answer that does not address a criterion FAILS it; do \
not credit intent.

Use "na" ONLY when the criterion turned out not to apply to this request \
at all — not when the answer merely omitted it.

CRITERIA:
{criteria}

ANSWER:
{answer}

Reply with JSON only:
{{"grades": [{{"id": "c1", "verdict": "pass|fail|na", "why": "..."}}]}}"""


def _parse_json(text: str) -> Optional[dict]:
    """Tolerant JSON extraction — the model wraps objects in prose and
    fences often enough that a strict parse loses real verdicts."""
    if not text:
        return None
    t = str(text).strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:                                       # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except Exception:                                       # noqa: BLE001
        return None


def normalize_criteria(raw: Any, *, cap: Optional[int] = None
                       ) -> List[Dict[str, str]]:
    """Coerce a model's criteria list into the stored shape, dropping
    anything unusable.

    Ids are RE-ASSIGNED positionally rather than trusted. A model that
    emits two `c1`s (it does) would otherwise make the grade join
    ambiguous, and an ambiguous join resolved by first-match silently
    grades one criterion twice while dropping another — a wrong score
    that looks exactly like a right one.
    """
    cap = cap or _max_criteria()
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    seen = set()
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("criterion") or item.get("text") or "").strip()
        else:
            continue
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:          # a duplicated criterion double-weights it
            continue
        seen.add(key)
        out.append({"id": f"c{len(out) + 1}", "criterion": text})
        if len(out) >= cap:
            break
    return out


async def build_rubric(user_request: str, llm_client: Any, *,
                       call_kwargs: Optional[Dict[str, Any]] = None
                       ) -> List[Dict[str, str]]:
    """Instance-specific criteria for a request.

    ⚠ THE RESPONSE IS NOT A PARAMETER, and that is the load-bearing
    property of this whole module rather than an oversight. A rubric
    written with the answer in view is a rubric written to fit it. Callers
    cannot pass it in even by mistake; there is nowhere to put it.
    """
    if not str(user_request or "").strip():
        return []
    prompt = _BUILD_PROMPT.format(
        lo=MIN_CRITERIA, hi=_max_criteria(),
        request=str(user_request)[:4000])
    data = await _call(llm_client, prompt, call_kwargs)
    if not data:
        return []
    return normalize_criteria(data.get("criteria"))


#: ⚠ THE ROUTING KWARGS, COPIED FROM THE THREE PEER BACKGROUND JUDGES in
#: this codebase (memory extract, self-eval, postmortem), not invented.
#: The first version passed NO kwargs, so every rubric call took the
#: `is_background=False` leg: it incremented `foreground_tasks`, went to
#: the MAIN node, and carried `timeout=None`.
#:
#: Three consequences, all live once the flag flips: two full-priority
#: main-slot calls per declined chat turn competing with the NEXT user
#: turn (the call site's own comment says "a judging call must not add
#: latency"); every genuine background caller parked on
#: `_wait_for_foreground_clear` while they were in flight; and an
#: unbounded call holding `foreground_tasks` up if the upstream wedged.
#: Measured eligible volume: ~33% of all user turns.
_BG_KWARGS = {"use_worker": True, "is_background": True,
              "off_main_only": True, "timeout": 90.0,
              "task_label": "rubric-shadow"}


def _background_kwargs(llm_client: Any,
                       call_kwargs: Optional[Dict[str, Any]]
                       ) -> Dict[str, Any]:
    """`_BG_KWARGS`, filtered to what this client actually accepts.

    The verifier's `_bounded_fallback_kwargs` does the same introspection
    for the same reason: this module is duck-typed over stubs and
    wrappers, and passing an unknown keyword would TypeError into the
    broad except and silently skip the judgement — a feature that looks
    switched on and never runs.
    """
    import inspect
    out = dict(_BG_KWARGS)
    fn = getattr(llm_client, "chat_completion", None)
    if fn is not None:
        try:
            params = inspect.signature(fn).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in params.values()):
                out = {k: v for k, v in out.items() if k in params}
        except (TypeError, ValueError):
            out = {}
    out.update(call_kwargs or {})       # an explicit caller override wins
    return out


async def _call(llm_client: Any, prompt: str,
                call_kwargs: Optional[Dict[str, Any]]) -> Optional[dict]:
    if llm_client is None:
        return None
    payload = {"messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 800}
    try:
        result = await llm_client.chat_completion(
            payload, **_background_kwargs(llm_client, call_kwargs))
    except Exception as exc:                                # noqa: BLE001
        logger.debug("rubric call failed: %s", exc)
        return None
    try:
        text = (result.get("choices", [{}])[0]
                .get("message", {}).get("content", ""))
    except Exception:                                       # noqa: BLE001
        return None
    return _parse_json(text)


# ══════════════════════════════════════════════════════════════════════
# 2. Grading + aggregation
# ══════════════════════════════════════════════════════════════════════

def aggregate(criteria: List[Dict[str, str]],
              grades: Any) -> RubricVerdict:
    """Fold per-criterion verdicts into a graded score, or ABSTAIN.

    THE SCORE IS PASS / (PASS + FAIL). `na` items leave the denominator
    entirely rather than counting as passes — crediting an inapplicable
    criterion is how a judge inflates itself, and a rubric where every
    item went `na` has decided nothing and must ABSTAIN rather than
    score 1.0 over an empty denominator.
    """
    v = RubricVerdict()

    # `.get`, not `[]`: `aggregate` is an exported entry point and raised
    # KeyError on a criterion without an id. Safe on the production path
    # only because `normalize_criteria` reassigns them — an invariant one
    # caller away.
    # ⚠ DE-DUPED ON ID. `normalize_criteria` reassigns ids positionally,
    # so this is unreachable from the production path — but `aggregate`
    # is exported, and a duplicated id double-counted its verdict
    # (`[c1, c1, c2]` scored 2 passes from one grade). An exported entry
    # point that is safe only because of an invariant one caller away is
    # safe by luck.
    _seen_ids = set()
    _clean = []
    for c in criteria:
        if not (isinstance(c, dict) and c.get("id") and c.get("criterion")):
            continue
        if c["id"] in _seen_ids:
            continue
        _seen_ids.add(c["id"])
        _clean.append(c)
    criteria = _clean
    # ⚠ ONE floor check, not two. There used to be an identical check on
    # the RAW list above; the id/criterion filter added this one, and a
    # mutation showed the first could be deleted with the whole suite
    # green — it was subsumed, and dead guard code reads as defence
    # (§4CR removed a dead `/tmp` branch for the same reason). This is
    # the one that matters, because it counts what will actually be
    # graded.
    if len(criteria) < MIN_CRITERIA:
        v.reason = (f"{len(criteria)} usable criteria — under the "
                    f"{MIN_CRITERIA} floor. A checklist this short is a "
                    f"Likert score wearing a list, and an empty one passes "
                    f"everything by asking nothing")
        return v
    by_id = {c["id"]: c for c in criteria}
    seen: Dict[str, str] = {}
    if isinstance(grades, list):
        for g in grades:
            if not isinstance(g, dict):
                continue
            gid = str(g.get("id") or "").strip()
            verdict = str(g.get("verdict") or "").strip().lower()
            if gid not in by_id or verdict not in (PASS, FAIL, NA):
                continue
            if gid in seen:      # first grade wins; a re-grade is noise
                continue
            seen[gid] = verdict

    rows = []
    for c in criteria:
        # ⚠ AN UNGRADED CRITERION IS NOT A PASS. A truncated or malformed
        # response would otherwise silently shrink the denominator in the
        # flattering direction — the same "a check that cannot run reports
        # the favourable outcome" this project has a standing lesson for.
        # It is recorded as ungraded and forces an abstain below.
        verdict = seen.get(c["id"])
        rows.append({"id": c["id"], "criterion": c["criterion"],
                     "verdict": verdict})
    v.criteria = rows

    ungraded = sum(1 for r in rows if r["verdict"] is None)
    if ungraded:
        v.reason = (f"{ungraded} of {len(rows)} criteria came back "
                    f"ungraded — an ungraded criterion is not a pass, so "
                    f"the turn is abstained rather than scored on a "
                    f"denominator that quietly shrank")
        return v

    v.n_pass = sum(1 for r in rows if r["verdict"] == PASS)
    v.n_fail = sum(1 for r in rows if r["verdict"] == FAIL)
    v.n_na = sum(1 for r in rows if r["verdict"] == NA)
    denom = v.n_pass + v.n_fail
    if denom == 0:
        v.reason = ("every criterion came back not-applicable — the rubric "
                    "decided nothing, and 1.0 over an empty denominator "
                    "would be a fabricated pass")
        return v

    v.status = GRADED
    v.score = v.n_pass / denom
    v.reason = f"{v.n_pass}/{denom} criteria met" + (
        f" ({v.n_na} n/a)" if v.n_na else "")
    return v


async def grade_turn(user_request: str, response: str, llm_client: Any, *,
                     call_kwargs: Optional[Dict[str, Any]] = None
                     ) -> RubricVerdict:
    """Full shadow judgement for one declined turn.

    Two calls, deliberately: the rubric is built BEFORE the answer is in
    any context, so the judge cannot write criteria around it.
    """
    if not str(response or "").strip():
        v = RubricVerdict()
        v.reason = "no response text to grade"
        return v
    criteria = await build_rubric(user_request, llm_client,
                                  call_kwargs=call_kwargs)
    if len(criteria) < MIN_CRITERIA:
        v = RubricVerdict()
        v.reason = (f"no checkable rubric for this request "
                    f"({len(criteria)} criterion/criteria) — small talk "
                    f"and acknowledgements ABSTAIN by design")
        return v
    listing = "\n".join(f"{c['id']}: {c['criterion']}" for c in criteria)
    data = await _call(
        llm_client,
        _GRADE_PROMPT.format(criteria=listing, answer=str(response)[:8000]),
        call_kwargs)
    if not data:
        v = RubricVerdict()
        v.criteria = [{"id": c["id"], "criterion": c["criterion"],
                       "verdict": None} for c in criteria]
        v.reason = "the grading call returned nothing parseable"
        return v
    return aggregate(criteria, data.get("grades"))


# ══════════════════════════════════════════════════════════════════════
# 3. The shadow ledger — the ONLY thing this module writes
# ══════════════════════════════════════════════════════════════════════

def shadow_path(home: Optional[Path] = None) -> Path:
    base = Path(home) if home else Path(
        os.environ.get("GHOST_HOME") or Path.home() / "Data" / "AI" / "Data")
    return base / SHADOW_REL


def record_shadow(verdict: RubricVerdict, *, trajectory_id: str = "",
                  req_id: str = "", home: Optional[Path] = None,
                  ts: Optional[float] = None) -> bool:
    """Append one shadow row. Never raises; returns whether it landed.

    `trajectory_id` and `req_id` are the durable join keys — the same
    pair the corrections overlay and the calibration corpus use — because
    a shadow verdict that cannot be joined to a human label can never be
    evaluated, and an unevaluable instrument is the thing this whole
    exercise exists to stop building.
    """
    try:
        p = shadow_path(home)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = verdict.to_row(trajectory_id=trajectory_id, req_id=req_id, ts=ts)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:                                # noqa: BLE001
        logger.debug("rubric shadow write failed: %s", exc)
        return False


async def shadow_grade_and_record(user_request: str, response: str,
                                  llm_client: Any, *, trajectory_id: str = "",
                                  req_id: str = "",
                                  home: Optional[Path] = None,
                                  call_kwargs: Optional[Dict[str, Any]] = None
                                  ) -> Optional[RubricVerdict]:
    """The one entry point a caller should use. Returns None when the
    feature is off, so a caller cannot accidentally consume a verdict
    that was never produced."""
    if not shadow_enabled():
        return None
    try:
        v = await grade_turn(user_request, response, llm_client,
                             call_kwargs=call_kwargs)
    except Exception as exc:                                # noqa: BLE001
        logger.debug("rubric shadow grade failed: %s", exc)
        return None
    record_shadow(v, trajectory_id=trajectory_id, req_id=req_id, home=home)
    return v


# ══════════════════════════════════════════════════════════════════════
# 4. Agreement — the ONLY thing that could ever promote this
# ══════════════════════════════════════════════════════════════════════

#: Paired rows needed before an agreement rate is a number rather than an
#: anecdote. §4CE is the reason this is a hard gate and not advice: ten of
#: ten arm/metric pairs there reported a conclusion for a difference that
#: was arithmetically undetectable.
MIN_PAIRED = 30


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson interval — correct near 0 and 1, where a Wald interval runs
    off the end of the scale and manufactures confidence."""
    if n <= 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def agreement(shadow_rows, labels: Dict[str, str], *,
              threshold: float = 0.5) -> Dict[str, Any]:
    """How often does the shadow rubric agree with a HUMAN label?

    `labels` maps trajectory_id → "passed"/"failed" and must come from
    the human channel, never from the machine verifier: scoring a judge
    against another judge measures their shared blind spot, and this
    project has one instrument that was credited with beating a base rate
    on a delta whose CI straddled zero.

    Returns a verdict that REFUSES to conclude under `MIN_PAIRED`.
    """
    n = agree = 0
    epochs = set()
    skipped_abstain = skipped_unlabelled = skipped_epoch = 0
    # ⚠ THE PAIRED LABELS, NOT THE WHOLE LABEL STORE. The first version
    # computed the majority-class baseline over every human label on the
    # box while measuring `rate` over the joined subset only — two
    # different populations, compared. Three independent reviewers
    # demonstrated the same consequence: a judge emitting score=1.0 for
    # EVERY row (zero information content) reads "agrees 95%, whose LOWER
    # bound beats the 78% majority-class baseline" and is declared
    # USABLE, because ABSTAIN concentrates on ungradable turns and leaves
    # the scored subset more class-skewed than the label pool. It fails
    # the other way too: a genuinely discriminating judge is refused when
    # the unpaired labels skew the opposite direction.
    #
    # This is the §4BR wrong-statistic trap in the very function whose
    # comment cites §4BR. The baseline must describe the same rows the
    # rate does.
    paired_labels: List[str] = []
    for row in shadow_rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("epoch") != RUBRIC_EPOCH:
            skipped_epoch += 1
            epochs.add(row.get("epoch"))
            continue
        if row.get("status") != GRADED or row.get("score") is None:
            skipped_abstain += 1
            continue
        tid = str(row.get("trajectory_id") or "")
        human = labels.get(tid)
        if human not in ("passed", "failed"):
            skipped_unlabelled += 1
            continue
        try:
            _score = float(row["score"])
        except (TypeError, ValueError):
            # A row whose score is not a number is not a judgement. It
            # must not take the whole liveness row down with it.
            skipped_abstain += 1
            continue
        n += 1
        paired_labels.append(human)
        said_pass = _score >= threshold
        if said_pass == (human == "passed"):
            agree += 1

    out: Dict[str, Any] = {
        "n": n, "agree": agree,
        "rate": (agree / n) if n else None,
        "epoch": RUBRIC_EPOCH,
        "skipped_abstain": skipped_abstain,
        "skipped_unlabelled": skipped_unlabelled,
        "skipped_other_epoch": skipped_epoch,
    }
    lo, hi = _wilson(agree, n)
    out["ci95"] = [lo, hi]
    if n < MIN_PAIRED:
        # Present on EVERY return, so a caller can index them without a
        # branch — an optional key is a KeyError waiting for the one path
        # nobody exercised.
        out["base_rate"] = None
        out["base_rate_n"] = len(paired_labels)
        out["verdict"] = (
            f"NO VERDICT: {n} paired row(s) against a {MIN_PAIRED} floor. "
            f"The rate above is reported, not judged — at this denominator "
            f"it cannot distinguish a useful judge from a coin flip.")
        out["usable"] = None
        return out
    # A judge must beat the MAJORITY-CLASS baseline, not 50%. Chat turns
    # that get labelled at all are ~73% `passed` (16/22 measured), so
    # "always say pass" scores ~0.73 — the §4BR wrong-statistic trap, stated here so the
    # first reader of this number cannot fall into it.
    #
    # Computed over `paired_labels` — the labels of the rows that were
    # actually scored — NOT over the whole label store. See the note at
    # the top of the loop for what the wrong denominator did.
    base = max(sum(1 for v in paired_labels if v == "passed"),
               sum(1 for v in paired_labels if v == "failed"))
    base_rate = base / len(paired_labels) if paired_labels else None
    out["base_rate"] = base_rate
    out["base_rate_n"] = len(paired_labels)
    # A single-class paired set cannot be beaten (base_rate == 1.0), and
    # saying "does NOT clear the 100% baseline" invites the reader to
    # conclude the judge is bad when the truth is that the LABELS carry
    # no contrast. Named, because the two lead opposite places.
    if base_rate is not None and base_rate >= 1.0:
        out["usable"] = None
        out["verdict"] = (
            f"NO VERDICT: all {len(paired_labels)} paired label(s) are the "
            f"same class, so there is no contrast to beat — this is a "
            f"property of the LABELS, not of the judge")
        return out
    # ⚠ IS THE BAR EVEN REACHABLE AT THIS n? Round 3: with 30 paired
    # rows the largest possible Wilson LOWER bound is 0.8865 (a perfect
    # judge, k == n), so any paired base rate above that cannot be
    # cleared by ANY judge — and the function was issuing a definite
    # "does NOT clear the 90% baseline" about a judge that got every row
    # right. That is `verdict without power` (§4CE) in the function whose
    # comments cite §4BR. The live label pool is ~76% passed and this
    # function's own note says the graded subset is MORE class-skewed, so
    # the unreachable region is not hypothetical.
    ceiling = _wilson(n, n)[0] if n else None
    if (base_rate is not None and ceiling is not None
            and ceiling <= base_rate):
        out["usable"] = None
        out["verdict"] = (
            f"NO VERDICT: at n={n} the highest reachable 95% lower bound "
            f"is {ceiling:.3f}, which cannot exceed the {base_rate:.0%} "
            f"majority-class baseline — no judge could clear this bar on "
            f"this many rows. Needs more PAIRED rows, not a better judge")
        return out
    if base_rate is not None and lo is not None and lo > base_rate:
        out["usable"] = True
        out["verdict"] = (f"agrees {out['rate']:.0%} (95% CI "
                          f"[{lo:.2f}, {hi:.2f}]), whose LOWER bound beats "
                          f"the {base_rate:.0%} majority-class baseline")
    else:
        out["usable"] = False
        out["verdict"] = (
            f"agrees {out['rate']:.0%} (95% CI [{lo:.2f}, {hi:.2f}]) — does "
            f"NOT clear the {base_rate:.0%} majority-class baseline"
            if base_rate is not None else
            "no labels to form a baseline from")
    return out


def read_shadow(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every shadow row on disk. Malformed lines are skipped, not fatal."""
    rows: List[Dict[str, Any]] = []
    p = shadow_path(home)
    if not p.is_file():
        return rows
    try:
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except Exception:                               # noqa: BLE001
                continue
            if isinstance(v, dict):
                rows.append(v)
    except Exception as exc:                                # noqa: BLE001
        logger.debug("rubric shadow read failed: %s", exc)
    return rows


__all__ = [
    "RUBRIC_EPOCH", "MIN_CRITERIA", "MIN_PAIRED", "ABSTAIN", "GRADED",
    "PASS", "FAIL", "NA", "RubricVerdict", "shadow_enabled", "build_rubric",
    "normalize_criteria", "aggregate", "grade_turn", "record_shadow",
    "shadow_grade_and_record", "agreement", "read_shadow", "shadow_path",
]
