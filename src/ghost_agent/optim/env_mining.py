"""§4CV — mining the agent's OWN failures into verifiable-reward tasks.

WHY THIS EXISTS, in one paragraph. `optim/trainset.py` builds GEPA's
examples from PASSED trajectories and scores a candidate prompt by token
overlap against the recorded `final_response`. Two consequences follow,
and this module addresses both: the 191 FAILED turns in the corpus
contribute **nothing** (a failure has no gold answer to overlap against),
and the metric rewards *looking like* the recorded reply rather than being
right. An executable oracle fixes the second and unlocks the first — a
mined failure becomes an example whose target is "the checker exits 0",
which is a verifiable reward rather than a string-similarity proxy.

The shape is Envs-FORGE (arXiv:2608.14312) and SENTINEL (arXiv:2606.12908)
reduced to what this corpus can actually support. Measured before building
(this project's Rule 0):

    failed `user_request` turns, all time ......... 191
    ...with >=1 tool call ........................ 176
    ...containment-clean (real REPLAY_FORBIDDEN) .. 35
    ...that synthesize into a parseable triple .... ~45% of seeds
    ...that clear BOTH gates (measured live) ...... ~8% of synthesized
    => usable items, first pass .................... 1-2
    growth ......................................... well under 1/month

⚠ THE PROJECTED FUNNEL HAS NOW BEEN WRONG TWICE, AND BOTH CORRECTIONS
BELONG HERE RATHER THAN IN A QUIETER PLACE, because the funnel IS the
argument for the scope.

  * The first draft said 67 containment-clean seeds. That used a
    hand-written list of "live-world" tools; the REAL
    `REPLAY_FORBIDDEN_TOOLS` also denies `manage_services`,
    `system_utility`, `postgres_admin`, `knowledge_base` and more.
    Measured after wiring the real list: 1,813 trajectories -> **35**.
  * The second said "~12 items" by applying Envs-FORGE's ~34% acceptance
    to those 35. MEASURED over today's live runs after both gates were
    working: **12 synthesized, 1 accepted (~8%)** — a quarter of the
    published rate. The gates here are stricter than Envs-FORGE's
    because they add a DETERMINACY requirement that its pipeline does
    not have, and because a 35B synthesizing its own oracles produces a
    lot of indeterminate ones.

**So the honest expectation is ONE TO TWO usable items, not twelve.**
That is below GEPA's documented 20-100 range, which means this does not
yet supply the consumer it was scoped for. Recorded as a MEASURED
RESULT, not a setback to be worked around: the remedy is a better
synthesizer or a different seed population, and either is a decision
someone should make with this number in front of them rather than a
knob to quietly loosen. Loosening the gates would restore the count and
destroy the reason to want it — §4AO measured skill-prune deciding 52%
of its victims on noise, which is what a permissive oracle gate buys.

**Why GEPA was the intended consumer** (the reasoning, kept because the
scoping decision was made on it): a McNemar promotion gate needs 6
candidate-only wins with zero losses, IDENTICAL at n=24 and n=120 since
McNemar counts only discordant pairs (§4CN) — so a small bank cannot
clear one at any size, while GEPA's documented range starts at 20. That
argument assumed ~20 items. At 1-2 it no longer holds, and the paragraph
above is the operative one. A bank pointed at a consumer it cannot supply
is how a loop runs for six weeks producing nothing anyone reads (§4CS).

⚠ ALSO MEASURED, AND IT KILLED THE OBVIOUS DESIGN: **112 of 176 failing
turns carry no `failure_reason` at all**, and only 19 have a specific
diagnosable one (34 say "structural failure", 11 "verifier refuted").
⚠ A reviewer read this as a two-population error and re-derived "176 of
176 carry a reason"; that is wrong — `corrections.jsonl` has no
`failure_reason` field at all (only a `reason` NOTE), so the overlay
cannot supply one. Both numbers here are over the same 176 rows. SENTINEL's Controller mines RECURRING FAILURE MODES; this
corpus does not have them. Mining is therefore per-trajectory, not
per-mode, and no recurrence threshold is applied — one would reduce the 35
seeds to almost zero while looking principled.

────────────────────────────────────────────────────────────────────────
THREE GUARDS, each from a specific incident
────────────────────────────────────────────────────────────────────────

1. **The oracle is tested in BOTH directions, mechanically.** A validator
   that passes a correct answer proves nothing on its own — one that
   `sys.exit(0)`s unconditionally passes it too, and then writes FAILED on
   nothing and PASSED on everything forever. §4AO measured skill-prune
   deciding **52% of its victims on noise**; an oracle that cannot fail is
   how noise gets manufactured at scale. The negative controls are
   MECHANICAL (a sentinel string and the empty string), not model-authored
   — asking a model for a "wrong answer" risks it being accidentally
   right, and an unforgeable control is worth more than a plausible one.

2. **Containment is inherited, not re-invented.** Seeds are refused if
   their trajectory touched anything in `REPLAY_FORBIDDEN_TOOLS` — the
   same list `core/isolation.py` fails closed on. A failure that needed
   the live world is not environment-able, exactly as it is not
   replayable, and a second private notion of "safe" is how the two came
   to disagree in the first place (§4CL).

3. **`origin="bench"`, so the existing doctrine applies for free.**
   `trainset.real_only_gate` evicts bench examples from the PRIVATE
   ship-gate tier: **bench may TEACH, it may never GRADE** (§4BH). Mined
   items are synthetic and must never decide a promotion. Tagging them
   correctly is the whole mechanism; nothing else needs to know.

────────────────────────────────────────────────────────────────────────
STAGING, NOT PROMOTION
────────────────────────────────────────────────────────────────────────

`eval.banks.pick_next_item` walks **every** bank in
`system/bench/banks/`, and the biological watchdog calls it in
production. Writing a bank there would therefore ARM THE LIVE BENCH
FLYWHEEL with unvetted synthetic items as a side effect of running a
miner. This module writes to `system/optim/mined_envs/` instead, and
promotion into the bank directory is a separate, explicit operator act
(`scripts/mine_failure_envs.py --promote`). Nothing here becomes
autonomous by having been built.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("GhostOptim")

#: Bump when the synthesis prompt or the item schema changes. Items from a
#: different epoch are not comparable and the reader drops them — the same
#: rule the calibration corpus learned the hard way (pooling eras forced a
#: negative Platt slope and rejected every refit).
MINING_EPOCH = "e1"

STAGING_REL = "system/optim/mined_envs"

#: The synthesised subject file. `final_response` items are the ONLY ones
#: a GEPA metric can score, because GEPA optimises an instruction that
#: produces TEXT — there is no agent loop in a GEPA rollout to write a
#: `solution.py`. Artifact-graded items are still mineable (they are
#: useful to the bench flywheel) but `trainset_from_items` refuses them,
#: loudly, rather than scoring a prompt against a file it cannot produce.
GRADED_TEXT = "final_response"
GRADED_ARTIFACT = "artifact"

#: Mechanical negative controls. Unforgeable by construction: no correct
#: answer to any request is the empty string, and none is this sentinel.
#: A validator that passes either is vacuous.
_WRONG_SENTINEL = "NOT-AN-ANSWER-a2f19c4e-THIS-MUST-FAIL"
_NEGATIVE_CONTROLS = (_WRONG_SENTINEL, "")

#: Modules a validator must not touch. An oracle that reaches the network
#: is not deterministic, and a bench item whose verdict depends on the
#: internet manufactures label noise on every outage. Checked statically
#: BEFORE execution, because the execution itself would already have done
#: the damage.
_FORBIDDEN_VALIDATOR_IMPORTS = (
    "socket", "urllib", "requests", "httpx", "aiohttp", "ftplib",
    "telnetlib", "smtplib", "curl_cffi", "selenium", "playwright",
)

_VALIDATOR_TIMEOUT_S = 20.0

#: How many times the reference answer is scored before an oracle is
#: believed. `random` is not on the import denylist and round 1 scored it
#: ONCE, so a validator that coin-flips after a correct-answer check was
#: admitted in roughly half of runs.
#:
#: ⚠ AND THE RESIDUAL IS STATED RATHER THAN PAPERED OVER: no finite
#: repeat count can exclude an adversarial nondeterministic oracle. At
#: four runs a FAIR coin ships the reference leg with probability
#: 0.5**4 = 6.25% unconditionally (0.5**3 = 12.5% GIVEN the first run
#: already returned 0 — the earlier comment quoted the conditional
#: figure), and once the two negative controls are each repeated too a
#: uniformly coin-flipping oracle ships at ~0.4%. An oracle that flips
#: one time in a thousand is still effectively invisible here. What catches the rest is downstream — `solvability_probe` scores
#: the same checker four more times on varied answers, and a checker that
#: disagrees with itself scatters those results. This is a filter with a
#: known leak, not a proof.
_REFERENCE_REPEATS = 4


def _confinement_required() -> bool:
    """Refuse to run model-authored code without the kernel sandbox.

    DEFAULT OFF so the miner still works off macOS, where `sandbox-exec`
    does not exist — but the item records `confined=False` either way, so
    an operator can always tell which items were verified under what.
    """
    return str(os.environ.get("GHOST_MINE_REQUIRE_CONFINE", "")
               ).strip().lower() in ("1", "true", "yes", "on")


# ══════════════════════════════════════════════════════════════════════
# 1. Seeds
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Seed:
    """One failed turn, judged environment-able."""
    trajectory_id: str = ""
    user_request: str = ""
    final_response: str = ""
    failure_reason: str = ""
    tool_names: List[str] = field(default_factory=list)
    n_steps: int = 0


def _tool_names(traj: Any) -> List[str]:
    out = []
    for t in (getattr(traj, "tool_calls", None) or []):
        if isinstance(t, dict):
            n = str(t.get("name") or "").strip().lower()
        else:
            n = str(getattr(t, "name", "") or "").strip().lower()
        if n:
            out.append(n.replace("-", "_").replace(" ", "_"))
    return out


def _forbidden() -> frozenset:
    """The containment list, imported from the module that FAILS CLOSED on
    it. A copy here is what drifts (§4CL found the isolation lists drifted
    from subagent's twice)."""
    try:
        from ..core.isolation import REPLAY_FORBIDDEN_TOOLS
        return frozenset(REPLAY_FORBIDDEN_TOOLS)
    except Exception:                                       # noqa: BLE001
        # ⚠ FAIL CLOSED, not open. Being unable to read the containment
        # list is not permission to ignore it — an empty set here would
        # admit every seed the list exists to refuse.
        raise RuntimeError(
            "REPLAY_FORBIDDEN_TOOLS is unreadable; refusing to mine "
            "without a containment list rather than mining without one")


def mine_seeds(trajectories: Iterable[Any], *,
               limit: Optional[int] = None) -> List[Seed]:
    """Failed, tool-using, containment-clean real turns.

    NO recurrence threshold, deliberately — see the module docstring: 112
    of 176 failing turns carry no `failure_reason`, so a "recurring failure
    mode" filter would look principled and reduce 67 seeds to almost none.
    """
    forbidden = _forbidden()
    seeds: List[Seed] = []
    for t in trajectories or []:
        if str(getattr(t, "task_kind", "") or "") != "user_request":
            continue
        outcome = getattr(t, "outcome", "")
        outcome = str(getattr(outcome, "value", outcome) or "").lower()
        if outcome != "failed":
            continue
        names = _tool_names(t)
        if not names:
            # Nothing was executed, so there is no procedure to rebuild.
            continue
        if any(n in forbidden for n in names):
            continue
        req = str(getattr(t, "user_request", "") or "").strip()
        if not req:
            continue
        seeds.append(Seed(
            trajectory_id=str(getattr(t, "id", "") or ""),
            user_request=req,
            final_response=str(getattr(t, "final_response", "") or ""),
            failure_reason=str(getattr(t, "failure_reason", "") or ""),
            tool_names=names,
            n_steps=int(getattr(t, "n_steps", 0) or 0),
        ))
        if limit is not None and len(seeds) >= limit:
            break
    return seeds


# ══════════════════════════════════════════════════════════════════════
# 2. Synthesis
# ══════════════════════════════════════════════════════════════════════

_SYNTH_PROMPT = """\
You are building ONE self-contained evaluation task from a real request \
that an AI agent previously failed.

Produce a task that tests the same underlying ability, but that can be \
graded by RUNNING CODE with no network, no external services and no \
operator state.

⚠ THE SOLVER ANSWERS IN ONE REPLY. It has NO tools, NO shell and NO \
filesystem. Its whole reply IS the answer. So:
  - NEVER instruct the solver to create files, directories or scripts, \
and never mention `answer.txt` in the challenge — a task that says \
"write the total to answer.txt" is unanswerable and would be discarded.
  - The challenge must ask for the ANSWER ITSELF, and must say what form \
it takes (e.g. "reply with only the number").

Rules the task MUST satisfy:
  - SELF-CONTAINED: everything needed is in the question text.
  - DETERMINISTIC: exactly ONE correct answer. If a competent solver \
could reasonably give a DIFFERENT valid answer, the task is unusable — \
prefer counting, arithmetic, parsing and lookup over choices of \
strategy, style or preference.
  - The validator reads the solver's reply from the file `answer.txt` in \
the current directory (the harness puts it there) and exits 0 if correct, \
non-zero otherwise. Be tolerant of surrounding whitespace.
  - The validator must NOT import socket, urllib, requests, httpx or any \
network library.
  - The validator must actually CHECK the content of answer.txt. A \
validator that always exits 0 is worthless.
  - `reference_answer` must be the exact text of a CORRECT reply.

ORIGINAL REQUEST THE AGENT FAILED:
{request}

TOOLS IT USED: {tools}
{reason}
Reply with JSON only:
{{"challenge": "the self-contained task text, including how to give the \
answer",
  "validation_script": "python source; reads answer.txt; sys.exit(0) on \
correct",
  "reference_answer": "the exact text of a correct one-reply answer"}}"""


@dataclass
class MinedItem:
    """One candidate task. Becomes a bank row only after `oracle_is_sound`."""
    item_id: str = ""
    bank: str = "ghost_failures"
    cluster: str = "mined_failure"
    challenge: str = ""
    setup_script: str = "# mined item: no setup files required\n"
    validation_script: str = ""
    graded_on: str = GRADED_TEXT
    reference_answer: str = ""
    source_trajectory_id: str = ""
    epoch: str = MINING_EPOCH
    #: Was the oracle self-test run under the kernel sandbox? Recorded
    #: rather than assumed — see `_confined_cmd`.
    confined: bool = False

    def to_bank_row(self) -> Dict[str, str]:
        """The `eval.banks` schema, exactly. Reusing it is what lets every
        downstream piece — pick_next_item, record_result, admissibility —
        work with no changes at all."""
        return {
            "bank": self.bank,
            "item_id": self.item_id,
            "cluster": self.cluster,
            "challenge": self.challenge,
            "setup_script": self.setup_script,
            "validation_script": self.validation_script,
            "graded_on": self.graded_on,
            # Mining-specific provenance. `eval.banks` ignores unknown
            # keys, so carrying them costs nothing and an item that cannot
            # say where it came from cannot be audited or retracted.
            "source_trajectory_id": self.source_trajectory_id,
            "mining_epoch": self.epoch,
            "reference_answer": self.reference_answer,
            "verified_confined": self.confined,
        }


def _parse_json(text: str) -> Optional[dict]:
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


def _item_id(seed: Seed, challenge: str) -> str:
    h = hashlib.sha1(
        (seed.trajectory_id + "|" + challenge).encode("utf-8", "ignore"))
    return f"mined-{h.hexdigest()[:12]}"


async def synthesize(seed: Seed, llm_client: Any, *,
                     call_kwargs: Optional[Dict[str, Any]] = None
                     ) -> Optional[MinedItem]:
    """One candidate from one seed. None on any failure — a seed that does
    not yield a parseable triple is a reject, not an error."""
    if llm_client is None:
        return None
    reason = (f"WHY IT FAILED: {seed.failure_reason}\n"
              if seed.failure_reason else "")
    prompt = _SYNTH_PROMPT.format(
        request=seed.user_request[:3000],
        tools=", ".join(sorted(set(seed.tool_names))[:10]) or "(none)",
        reason=reason)
    payload = {"messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3, "max_tokens": 2000}
    try:
        res = await llm_client.chat_completion(payload, **(call_kwargs or {}))
        text = (res.get("choices", [{}])[0]
                .get("message", {}).get("content", ""))
    except Exception as exc:                                # noqa: BLE001
        logger.debug("env synthesis call failed: %s", exc)
        return None
    data = _parse_json(text)
    if not data:
        return None
    challenge = str(data.get("challenge") or "").strip()
    # ⚠ REFUSE AGENT-SHAPED CHALLENGES. The solver in a GEPA rollout has
    # no tools and no filesystem; a challenge that says "create a file"
    # or names `answer.txt` is unanswerable in one reply, and the probe
    # would score its prose against a checker expecting the bare value.
    # Measured on the first live run: EVERY synthesised challenge was
    # this shape, because the prompt told the model the validator reads
    # answer.txt and it dutifully told the solver to write it.
    _low = challenge.lower()
    if "answer.txt" in _low or re.search(
            r"\bcreate\s+(?:a\s+)?(?:new\s+)?(?:file|directory|folder|"
            r"script)\b|\bwrite\s+(?:it\s+|the\s+\w+\s+)?to\s+(?:a\s+)?"
            r"(?:file|['\"`]?\w+\.(?:txt|py|json|csv))", _low):
        logger.debug("refused agent-shaped challenge for %s",
                     seed.trajectory_id)
        return None
    # rstrip only — a Python source file wants its trailing
    # newline, and `.strip()` also ate the leading one on
    # scripts the model opened with a blank line.
    validator = str(data.get("validation_script") or "").rstrip()
    if validator:
        validator += "\n"
    reference = str(data.get("reference_answer") or "")
    if not (challenge and validator and reference.strip()):
        return None
    return MinedItem(
        item_id=_item_id(seed, challenge),
        challenge=challenge,
        validation_script=validator,
        reference_answer=reference,
        source_trajectory_id=seed.trajectory_id,
    )


# ══════════════════════════════════════════════════════════════════════
# 3. The oracle self-test — BOTH directions
# ══════════════════════════════════════════════════════════════════════

def validator_static_defects(validator: str) -> List[str]:
    """Refusals decidable without running anything.

    Static FIRST, because the execution is the damage: a validator that
    opens a socket has already reached the network by the time an
    execution-based check could notice.
    """
    out: List[str] = []
    src = str(validator or "")
    if not src.strip():
        return ["empty validator"]

    # ⚠ PARSED, NOT PATTERN-MATCHED. The first version used
    # `\bimport\s+{mod}\b`, which does not match `import sys, socket` —
    # the single most natural way to write the very thing being refused.
    # Four of five network modules sailed through. A lexical proxy for a
    # semantic property, in the guard whose whole job is refusing.
    #
    # Parsing also makes a SYNTAX ERROR a static refusal, which is where
    # it belongs: an unparseable validator exits non-zero on every input,
    # which is indistinguishable from a very strict oracle if it is only
    # ever observed through its exit code.
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"validator does not parse ({e.msg} at line {e.lineno}) — "
                f"it would exit non-zero on EVERY input, which reads as a "
                f"strict oracle rather than a broken one"]
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(str(a.name or "").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            imported.add(str(node.module or "").split(".")[0])
    for mod in _FORBIDDEN_VALIDATOR_IMPORTS:
        if mod in imported:
            out.append(f"validator imports {mod} — an oracle that reaches "
                       f"the network is not deterministic")
    # Dynamic imports evade the AST walk's name set, so the textual
    # backstop stays for exactly those two forms.
    # ⚠ Matched against the CODE, not the raw text. `# NOTE: do not use
    # importlib here` tripped the substring check and refused a sound
    # item — a false positive in an already yield-starved funnel.
    _code = "\n".join(
        ln.split("#", 1)[0] for ln in src.splitlines())
    for dyn in ("__import__", "importlib"):
        if dyn in _code:
            out.append(f"validator uses {dyn} — a dynamically imported "
                       f"module cannot be checked before it runs")
    if "answer.txt" not in src:
        out.append("validator never reads answer.txt — it cannot be "
                   "checking the answer")
    if not re.search(r"\bsys\.exit\b|\bexit\s*\(|\braise\b|\bassert\b", src):
        out.append("validator has no failure path — nothing in it can "
                   "signal a wrong answer")
    return out


def _confined_cmd(cmd: List[str],
                  workdir: Path) -> Tuple[List[str], bool, str]:
    """`cmd` wrapped so it may only write under `workdir`, and may not
    reach the network.

    ⚠ THIS IS THE ROUND-1 CRITICAL, AND TWO REVIEWERS FOUND IT
    INDEPENDENTLY WITH EXECUTED PROOFS. The first version ran
    model-authored Python with `subprocess.run(..., cwd=tmpdir)` and
    called a tuple of 11 module names its containment. `cwd` is not
    confinement. Thirteen of fifteen attempted bypasses returned CLEAN
    from `validator_static_defects` — `os.system("curl …")`,
    `subprocess.run(["/usr/bin/curl", …])`, `import http.client` (top-level
    name is "http"), `ctypes`, `xmlrpc.client`, `webbrowser`,
    `shutil.rmtree` of the live bank directory, and reading
    `~/.ghost_api_key`. One reviewer executed a validator that wrote
    OUTSIDE the temp dir and spawned a socket-holding grandchild while
    `oracle_is_sound` reported "accepts its reference, rejects both
    negative controls".

    A deny-list of module NAMES is a lexical proxy for "cannot touch the
    box". The AST rewrite moved that proxy; it did not remove it. §4CI's
    lesson is that every exemption in a deny-list becomes the next
    bypass, and §4CR's is that closing this class needs the child under
    the sandbox rather than beside it.

    `evolve/confine.py` already exists for exactly this — a
    kernel-enforced `sandbox-exec` policy a process cannot lift and that
    its children inherit, including a detached grandchild. The static
    checks stay, because refusing before execution is still cheaper than
    refusing after; they are now a filter, not the containment.

    `policy_dir` must be a directory the CHILD cannot write, or the
    confinement is advisory — so it is a sibling of the work dir, never
    the work dir itself.

    ⚠ WHAT THIS DOES **NOT** CLOSE, stated because the list above
    enumerates it and a reader would otherwise assume it did. The policy
    is an INTEGRITY boundary, not a confidentiality one: reads stay open
    (a read-tight policy would refuse the interpreter its own stdlib), so
    a validator CAN read `~/.ghost_api_key` — round 2 executed it. What
    it cannot do is make that value land anywhere: writes, network and
    every file destination outside the workdir are denied, and
    `_run_validator` discards stdout. The residual is `(allow default)`
    permitting mach IPC, through which a reviewer moved a secret to the
    macOS pasteboard. Closing that needs a deny-by-default policy, which
    would refuse the pin suite for reasons unrelated to the candidate —
    the same trade `evolve/confine.py` documents. The honest statement is
    "a validator may READ what this box can read; it may not change or
    transmit anything", and that is the property the gate relies on.
    """
    try:
        from ..evolve.confine import confine
        cmd2, confined, why = confine(
            cmd, writable=[workdir], allow_network=False,
            policy_dir=workdir.parent / "policy")
        return cmd2, confined, why
    except Exception as exc:                                # noqa: BLE001
        logger.debug("confine unavailable: %s", exc)
        return list(cmd), False, f"{type(exc).__name__}: {exc}"


#: `sandbox-exec` prefixes its own diagnostics; a validator that merely
#: exits non-zero prints nothing of the sort. Anchored to the start of a
#: line so a validator cannot forge it by printing the string.
_WRAPPER_ERR = re.compile(r"^sandbox-exec:", re.M)


def _wrapper_failed(proc) -> bool:
    """Did the SANDBOX fail to start the child, rather than the child
    failing? Only ever consulted on a non-zero exit, so a clean run is
    never second-guessed."""
    if getattr(proc, "returncode", 0) == 0:
        return False
    return bool(_WRAPPER_ERR.search(str(getattr(proc, "stderr", "") or "")))


#: Set by `_run_validator` on every call: whether the last validator ran
#: under the kernel sandbox. Read by `oracle_is_sound` so an item records
#: HOW it was verified — a run that silently went unconfined while
#: claiming otherwise would be worse than never having this (§4CR).
_LAST_RUN_CONFINED: Dict[str, Any] = {
    "confined": False, "checked": False, "why": ""}


def _run_validator(validator: str, answer: str,
                   timeout_s: float = _VALIDATOR_TIMEOUT_S) -> Optional[int]:
    """Exit code, or None when the validator could not be run at all.

    None and a non-zero exit are DIFFERENT and must not be conflated: a
    crashing validator is a rejected item, while treating "could not run"
    as "correctly failed" would let a broken oracle pass its own negative
    control — a harness that cannot run reporting the favourable outcome,
    which is this project's most-repeated lesson.
    """
    # ⚠ THE CHECK BELONGS HERE, NOT AT ONE CALLER. Round 2 found
    # `_confinement_required()` referenced only from `oracle_is_sound`,
    # so `oracle_score` and `solvability_probe` — which run the SAME
    # model-authored code, once per GEPA rollout — ignored it entirely.
    # Executed proof: with confinement off and REQUIRE_CONFINE on,
    # `oracle_score` ran a validator that wrote outside its workdir and
    # returned 1.0 with no refusal. The documented promise ("refuse to
    # run model-authored code without the kernel sandbox") failed open on
    # exactly the repeated-execution surface this module exists to
    # contain. Guarding one route instead of the medium is the §4CI
    # shape; this is the medium.
    # Cleared FIRST, every call. Stale-on-failure meant a run that
    # could not start still reported the PREVIOUS item's confinement.
    _LAST_RUN_CONFINED.update(
        {"confined": False, "checked": False, "why": "run did not start"})
    if _confinement_required():
        from ..evolve.confine import available as _sb_available
        if not _sb_available()[0]:
            _LAST_RUN_CONFINED["why"] = (
                "GHOST_MINE_REQUIRE_CONFINE is set and the kernel sandbox "
                "is unavailable")
            return None
    try:
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "w"
            work.mkdir()
            (work / "answer.txt").write_text(answer, encoding="utf-8")
            (work / ".validator.py").write_text(validator, encoding="utf-8")
            cmd, confined, why = _confined_cmd(
                ["python3", ".validator.py"], work)
            _LAST_RUN_CONFINED["confined"] = confined
            _LAST_RUN_CONFINED["checked"] = True
            _LAST_RUN_CONFINED["why"] = why
            p = subprocess.run(
                cmd, cwd=str(work),
                capture_output=True, text=True, timeout=timeout_s)
            # ⚠ A WRAPPER FAILURE IS NOT A VALIDATOR VERDICT. Round 2
            # made `_confined_cmd` the choke point for every caller,
            # which widened this: `sandbox-exec`'s OWN exit codes were
            # being returned as the validator's. Executed with a missing
            # policy file: rc=65 on a CORRECT answer, so `oracle_score`
            # returned 0.0 ("the candidate was WRONG") — the thing its
            # own docstring forbids — and in `solvability_probe` one such
            # hiccup flips a TRIVIAL item (4/4) to SEPARATES (3/4), i.e.
            # an infrastructure failure moving an item into the ACCEPTED
            # band. `sandbox-exec` announces itself on stderr; a
            # validator's own failure does not.
            if confined and _wrapper_failed(p):
                logger.debug("sandbox wrapper failed: %s",
                             (p.stderr or "")[:200])
                return None
            return p.returncode
    except Exception:                                       # noqa: BLE001
        return None


def oracle_is_sound(item: MinedItem, *,
                    timeout_s: float = _VALIDATOR_TIMEOUT_S
                    ) -> Tuple[bool, str]:
    """Does this oracle both ACCEPT the right answer and REJECT wrong ones?

    ⚠ THE NEGATIVE HALF IS THE POINT. Envs-FORGE verifies that the
    generated tests pass the oracle solution; that alone admits a
    validator which passes EVERYTHING, and such an item then reports
    PASSED forever and teaches nothing while looking like a working
    measurement. Both directions, or the item does not ship.

    The controls are MECHANICAL — a sentinel and the empty string — not
    model-authored: a model asked for a "wrong answer" can produce an
    accidentally-correct one, and the resulting rejection would be a real
    oracle failing a real answer, i.e. exactly backwards.
    """
    defects = validator_static_defects(item.validation_script)
    if defects:
        return False, "; ".join(defects)

    # The refusal is enforced at the EXECUTION CHOKE POINT (`_run_validator`,
    # so `oracle_score` and the probe inherit it) — but it is also checked
    # HERE, because the choke point can only answer `None` and "could not
    # be run (crash or timeout)" hides why. The guard belongs at the
    # medium; the REASON belongs where a reader will look for it.
    if _confinement_required():
        from ..evolve.confine import available as _sb_available
        _ok, _why = _sb_available()
        if not _ok:
            return False, (f"GHOST_MINE_REQUIRE_CONFINE is set and the "
                           f"kernel sandbox is unavailable ({_why}) — "
                           f"refusing to execute model-authored code "
                           f"unconfined")

    # ⚠ TWICE, and they must AGREE. `random` is not on the import
    # denylist and nothing re-ran the checker, so a validator that
    # coin-flips after a correct-answer check was admitted in roughly
    # half of runs (measured by a reviewer: 7 of 12). A nondeterministic
    # oracle writes a different verdict for the same answer on different
    # days, which is label noise with a plausible face.
    rc = _run_validator(item.validation_script, item.reference_answer,
                        timeout_s)
    if rc is None:
        return False, "validator could not be run (crash or timeout)"
    if rc != 0:
        return False, (f"validator REJECTS its own reference answer "
                       f"(exit {rc}) — the oracle and the task disagree")
    for _ in range(_REFERENCE_REPEATS - 1):
        rc2 = _run_validator(item.validation_script, item.reference_answer,
                             timeout_s)
        if rc2 is None:
            return False, "validator could not be re-run for the repeat check"
        if rc2 != rc:
            return False, (f"validator is NONDETERMINISTIC — the same "
                           f"reference answer scored {rc} then {rc2}. An "
                           f"oracle that disagrees with itself writes a "
                           f"different verdict for the same answer on "
                           f"different days")

    # ⚠ REPEATED TOO, AND THIS IS THE LEG THAT MATTERS MORE. Round 2
    # measured the asymmetry: after the reference-side repeat, a
    # validator that coin-flips on a CORRECT answer was admitted 4% of
    # the time — but one that coin-flips on WRONG answers was still
    # admitted 26%, unchanged. The second shape is the dangerous one: it
    # manufactures false PASSes, which is label noise pointed straight at
    # the learning corpus. Guarding only the flattering direction is how
    # a filter comes to look strict while leaking the harm.
    for ctrl in _NEGATIVE_CONTROLS:
        for _ in range(_REFERENCE_REPEATS):
            rc = _run_validator(item.validation_script, ctrl, timeout_s)
            if rc is None:
                return False, ("validator could not be run on a negative "
                               "control")
            if rc == 0:
                label = ("the empty string" if ctrl == ""
                         else "a sentinel non-answer")
                return False, (f"validator ACCEPTS {label} — it cannot "
                               f"fail, so every verdict it produces is "
                               f"label noise")
    # ⚠ HOW it was verified travels with the verdict. An unconfined run
    # is not a failure (off macOS there is no seatbelt), but a cascade
    # that silently ran unconfined while claiming otherwise would be
    # worse than one that never had this — §4CR's own rule, applied here.
    item.confined = bool(_LAST_RUN_CONFINED.get("confined"))
    if not item.confined and _confinement_required():
        # Belt and braces: the pre-flight above refuses when the sandbox
        # is UNAVAILABLE, and this catches the case where it was
        # available but a particular run still came back unconfined.
        return False, ("the validator ran WITHOUT the kernel sandbox and "
                       "GHOST_MINE_REQUIRE_CONFINE is set — refusing to "
                       "trust an item verified by unconfined execution")
    return True, ("accepts its reference, rejects both negative controls"
                  + ("" if item.confined else
                     f" ⚠ UNCONFINED ({_LAST_RUN_CONFINED.get('why') or 'unknown'})"))


# ══════════════════════════════════════════════════════════════════════
# 3b. The DETERMINACY probe — found by running the miner, not by reading
# ══════════════════════════════════════════════════════════════════════
#
# ⚠ THE FIRST LIVE RUN SHIPPED A BAD ITEM THROUGH A SOUND ORACLE, and
# that is the whole reason this stage exists.
#
# Mined item `mined-1399212a06bc` asked for a chess move and validated
# with `if data['move'] != 'e7e6': sys.exit(1)`. That oracle passes every
# check above it: it accepts its reference, and it rejects both the
# sentinel and the empty string, so it demonstrably discriminates. But
# `e7e6` is ONE OF MANY reasonable moves in that position — the task is
# INDETERMINATE, and the validator encodes a preference as a fact.
#
# `oracle_is_sound` answers "can this checker fail?". It cannot answer
# "is there one right answer?", and those are different questions. An
# item that is sound but indeterminate trains a prompt to reproduce an
# arbitrary choice, which is worse than no item: it looks like signal.
#
# The probe is Envs-FORGE's frontier-scoring stage, which the first draft
# of this module deliberately simplified away as "over-engineered at
# Ghost's volume". Running the miner showed it is not optional. Sample
# the challenge k times INDEPENDENTLY and score each with the item's own
# oracle:
#
#   passes == 0  → unreachable OR arbitrary. Either way a metric that can
#                  only reject, which §4F already shipped once (both arms
#                  at the noise floor, a gate that could only ever fail).
#   passes == k  → every candidate prompt scores 1.0, so the item cannot
#                  DISCRIMINATE between prompts. A constant column has
#                  zero variance and the optimizer can only lose by
#                  weighting it — the same invariant that kept
#                  `w_entropy` pinned at 0 across 1200 samples.
#   otherwise    → the item separates, which is the only useful state.

#: Independent attempts per candidate. Four rather than three so a single
#: unlucky sample cannot decide the band.
DEFAULT_PROBE_K = 4

#: Non-zero on purpose. At temperature 0 the k samples are near-identical
#: and the probe would measure the decoder, not the task.
_PROBE_TEMPERATURE = 0.7

#: ⚠ THE ROLLOUT REGIME, COPIED FROM THE A/B SHIP-GATE, NOT INVENTED.
#: `scripts/run_gepa.py`'s `_ab_runner` carries this comment: "at 1024
#: tokens with thinking on, the reasoning phase consumed the entire
#: budget, content came back EMPTY, and BOTH arms scored at the noise
#: floor — a gate that can only ever reject."
#:
#: The first version of this probe used `max_tokens: 1200` with thinking
#: left ON and rebuilt that defect exactly: every live probe reply was the
#: empty string, so 12 of 12 mined items were rejected as "arbitrary" by a
#: gate that could not have accepted anything. The numbers looked like a
#: finding about the items. They were a finding about the probe.
_PROBE_MAX_TOKENS = 8192
_PROBE_TEMPLATE_KWARGS = {"enable_thinking": False}

PROBE_ARBITRARY = "arbitrary_or_unreachable"
PROBE_TRIVIAL = "trivial"
PROBE_SEPARATES = "separates"
PROBE_UNRUN = "could_not_run"


@dataclass
class ProbeResult:
    verdict: str = PROBE_UNRUN
    #: Samples the CHECKER actually scored.
    passes: int = 0
    #: Samples that produced text and were handed to the checker.
    attempts: int = 0
    #: Of `attempts`, those the checker could not score (it crashed or
    #: timed out). Distinct from `no_output` — different causes, and
    #: sharing one counter double-subtracted them from the denominator.
    unscored: int = 0
    #: Calls that produced nothing to score at all (empty generation, or
    #: the call itself failed). Never entered `attempts`.
    no_output: int = 0
    why: str = ""

    @property
    def usable(self) -> bool:
        return self.verdict == PROBE_SEPARATES


async def solvability_probe(row: Dict[str, Any], llm_client: Any, *,
                            k: int = DEFAULT_PROBE_K,
                            call_kwargs: Optional[Dict[str, Any]] = None,
                            timeout_s: float = _VALIDATOR_TIMEOUT_S
                            ) -> ProbeResult:
    """Does this item have ONE reachable right answer, or a preferred one?

    Independent samples, scored by the item's own checker. Never raises;
    an unrunnable probe returns PROBE_UNRUN and the caller must treat
    that as "unknown", not as a pass — an item admitted because its probe
    crashed is an item admitted for no reason.
    """
    res = ProbeResult(attempts=0)
    answers: List[str] = []
    challenge = str((row or {}).get("challenge") or "").strip()
    if not challenge or llm_client is None:
        res.why = "no challenge text or no client — probe not run"
        return res
    for _ in range(max(1, int(k))):
        try:
            out = await llm_client.chat_completion({
                "messages": [{"role": "user", "content": challenge}],
                "temperature": _PROBE_TEMPERATURE,
                "max_tokens": _PROBE_MAX_TOKENS,
                "stream": False,
                "chat_template_kwargs": dict(_PROBE_TEMPLATE_KWARGS),
            }, **(call_kwargs or {}))
            text = (out.get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or "")
        except Exception as exc:                            # noqa: BLE001
            logger.debug("solvability probe call failed: %s", exc)
            res.no_output += 1
            continue
        # ⚠ AN EMPTY REPLY IS AN INFRASTRUCTURE FAILURE, NOT A WRONG
        # ANSWER, and the distinction is what makes this probe honest. A
        # starved or errored generation returns "" for every sample; if
        # that counted as an attempt that failed, the probe would report
        # "0 of 4 — arbitrary" about a task it never actually asked.
        if not text.strip():
            res.no_output += 1
            continue
        res.attempts += 1
        _ans = _extract_answer(text)
        answers.append(" ".join(_ans.lower().split()))
        score = oracle_score(row, _ans, timeout_s=timeout_s)
        if score is None:
            res.unscored += 1
        elif score >= 1.0:
            res.passes += 1
    # ⚠ THE DENOMINATOR IS SCORED SAMPLES, NOT LLM CALLS. Round 1
    # incremented `attempts` before calling `oracle_score`, so a sample
    # the CHECKER could not score still counted — and `unscored` was
    # computed and then never consulted by any verdict branch.
    #
    # Two reviewers demonstrated both consequences. A validator that
    # times out on every sample produced "0 of 4 independent attempts
    # satisfied the checker — the answer is unreachable or ARBITRARY", a
    # confident, plausible and wrong reason for condemning an item that
    # was never scored at all. And one unscorable sample flipped a
    # TRIVIAL item (4/4) to SEPARATES (3/4) — a harness failure moving an
    # item into the favourable band. `oracle_score`'s own docstring says
    # None must never be called zero; its first in-module caller did.
    scored = res.attempts - res.unscored
    if scored <= 0:
        res.why = (f"none of {res.attempts + res.no_output} probe sample(s) "
                   f"could be SCORED ({res.no_output} produced no output, "
                   f"{res.unscored} could not be scored by the checker) — "
                   f"the task was never actually asked and answered, so no "
                   f"verdict about it is available")
        return res
    excl = []
    if res.unscored:
        excl.append(f"{res.unscored} the checker could not score")
    if res.no_output:
        excl.append(f"{res.no_output} with no output")
    tail = f" ({'; '.join(excl)}, excluded)" if excl else ""
    if res.passes == 0:
        res.verdict = PROBE_ARBITRARY
        res.why = (f"0 of {scored} scored attempts satisfied the checker — "
                   f"the answer is unreachable or ARBITRARY (the oracle "
                   f"encodes a preference, not a fact), so this item can "
                   f"only ever reject{tail}")
    elif res.passes >= scored:
        res.verdict = PROBE_TRIVIAL
        res.why = (f"{res.passes} of {scored} scored attempts passed — every "
                   f"candidate prompt scores 1.0, so the item cannot "
                   f"DISCRIMINATE between prompts (a constant column){tail}")
    else:
        # ⚠ SELF-CONSISTENCY, AND THIS IS THE GATE'S REAL TEETH. Round 1's
        # band (0 < passes < k) cannot tell a HARD-BUT-DETERMINATE item
        # from an INDETERMINATE one whose preferred answer is sometimes
        # produced — two reviewers said so and one demonstrated it end to
        # end: a "favourite colour" item was ACCEPTED at 1/4 while a
        # determinate 6*7 item was REJECTED as trivial at 4/4. The live
        # staged item validated `data['move'] != 'c8g4'` — one of 23 legal
        # moves — the exact shape §3b was written to catch, sitting
        # accepted.
        #
        # The distinguishing signal is not the PASS RATE, it is whether
        # the independent samples AGREE WITH EACH OTHER. A determinate
        # task pulls samples toward one answer; an indeterminate one
        # scatters them, and the oracle's preferred value is merely one
        # of the scattered answers. So: the modal answer must recur.
        #
        # Costs nothing extra — the samples were already drawn — and it
        # errs toward REJECTION, which is the safe direction for a
        # trainset (a thin corpus beats a poisoned one).
        _modal = max((answers.count(a) for a in set(answers)), default=0)
        if len(answers) >= 2 and _modal < 2:
            res.verdict = PROBE_ARBITRARY
            res.why = (f"{res.passes} of {scored} scored attempts passed, but "
                       f"all {len(answers)} independent samples gave a "
                       f"DIFFERENT answer — the task has no single right "
                       f"answer, and the checker's preferred value is just "
                       f"one of them. An item like this trains a prompt to "
                       f"reproduce an arbitrary choice{tail}")
            return res
        res.verdict = PROBE_SEPARATES
        res.why = (f"{res.passes} of {scored} scored attempts passed; the "
                   f"modal answer recurred {_modal}x, so the task has a "
                   f"stable answer{tail}")
    return res


def _extract_answer(text: str) -> str:
    """What a candidate would have written to `answer.txt`.

    A model asked a question replies in prose around the answer. The
    challenge text tells it to give the answer directly, so the whole
    reply is the primary candidate — but a fenced block, when present, is
    what the reply is offering AS the answer, and scoring the prose
    around it would fail determinate items for formatting.
    """
    t = str(text or "").strip()
    # ⚠ NEITHER FIRST NOR LAST — BOTH FIXES WERE WRONG, and round 2
    # showed the second one did not even fix its own cited example:
    #   "```python\nprint(6*7)\n```\nSo the answer is 42."
    # has ONE fence, so last-fence still returned `print(6*7)` — the
    # working. And on the commoner "answer, then code" shape, last-fence
    # is strictly WORSE than first-fence.
    #
    # The mistake was picking a position at all. A fence containing CODE
    # is never the answer to these challenges, whose prompt says "reply
    # with only the value". So: prefer the last fence that does not look
    # like code, and fall back to the prose with fences REMOVED — which
    # is where "So the answer is 42" lives.
    fences = [m.group(1).strip()
              for m in re.finditer(r"```(?:\w+)?\s*(.+?)```", t, re.S)
              if m.group(1).strip()]
    for f in reversed(fences):
        if not _looks_like_code(f):
            return f
    stripped = re.sub(r"```(?:\w+)?\s*.+?```", " ", t, flags=re.S).strip()
    return stripped or (fences[-1] if fences else t)


#: ⚠ NO BARE `}` — a JSON answer like `{"move": "e7e6"}` ends with one,
#: and mined challenges legitimately ask for JSON. The hints are
#: STATEMENT shapes, which an answer never has.
#: ⚠ REWRITTEN AFTER ROUND 3, WHICH MEASURED BOTH FAILURE DIRECTIONS.
#: The previous version MISSED the working it exists to reject —
#: `a = 6\nb = 7\na * b`, `sys.exit(0)`, `console.log(6*7)` all read as
#: answers — and REJECTED real answers, because `;\s*$` caught
#: `SELECT name FROM users;`, `^\w+\s*=\s*\w+\(` caught `y = sin(x)`,
#: and `for \w+ in` caught the English sentence "Search for John in the
#: directory".
#:
#: The rewrite drops the shapes that overlapped with answers and keeps
#: only ones an ANSWER never has. It is still a heuristic and says so:
#: the fallback when every fence looks like code is the fence-stripped
#: prose, so a false "is code" costs one candidate rather than the item.
_CODE_HINT = re.compile(
    r"^\s*(?:import\s+\w|from\s+\w+\s+import\s|def\s+\w+\s*\(|"
    r"class\s+\w+[\s(:]|return\s|print\s*\(|console\.log\s*\(|"
    r"sys\.exit\s*\(|(?:let|const|var)\s+\w+\s*=|#\s|//\s)"
    r"|^\s*(?:for|while|if|elif|else|try|except|with)\b[^\n]*:\s*$"
    r"|^\s*\w+\s*=\s*\d+\s*$", re.M)


def _looks_like_code(s: str) -> bool:
    """A fence holding CODE is never the answer to a mined challenge —
    the prompt asks for the value itself.

    Deliberately conservative in the direction that costs least: a false
    'is code' drops one candidate and falls through, while a false 'is
    an answer' scores the working as the answer. Multi-line assignment
    blocks are caught by the `\w+ = <number>` line shape, which an
    answer does not have but a worked calculation does.
    """
    return bool(_CODE_HINT.search(str(s or "")))


# ══════════════════════════════════════════════════════════════════════
# 4. Orchestration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MineReport:
    seeds: int = 0
    synthesized: int = 0
    accepted: List[MinedItem] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    #: Candidates whose determinacy probe could not be RUN at all. These
    #: are not evidence about the items and must not read as gate
    #: strictness — see `summary`.
    unprobed: int = 0
    #: Did the DETERMINACY gate run? "accepted" means something weaker
    #: when it did not, and a reader cannot tell from the count alone.
    probed: bool = True

    @property
    def acceptance_rate(self) -> Optional[float]:
        return (len(self.accepted) / self.synthesized) if self.synthesized else None

    def summary(self) -> str:
        rate = ("n/a" if self.acceptance_rate is None
                else f"{self.acceptance_rate:.0%}")
        # ⚠ A HIGH ACCEPTANCE RATE IS A WARNING, NOT A WIN. Envs-FORGE
        # needed 291 attempts for 100 accepted items (~34%); near-100%
        # here means the oracle self-test is not biting, which is the
        # silent failure this whole module is built around.
        warn = ""
        if self.acceptance_rate is not None and self.acceptance_rate > 0.75:
            warn = ("  ⚠ acceptance is ABOVE 75% — Envs-FORGE measures ~34%, "
                    "so the oracle self-test is probably NOT BITING; "
                    "check that it still rejects an always-passing "
                    "validator before trusting these items")
        gates = ("oracle + determinacy" if self.probed
                 else "oracle ONLY — determinacy gate SKIPPED, so an item "
                      "here may encode a preference as a fact")
        # ⚠ A FAILURE TO MEASURE IS NOT A STRICT GATE. Round 1 rendered a
        # run where every probe call raised as "accepted 0 (0%)
        # [oracle + determinacy]" — nothing was measured, and 0%
        # acceptance reads as the gates biting hard. The most flattering
        # possible statement about gate strictness, produced by total
        # instrument failure.
        unrun = ""
        if self.unprobed:
            unrun = (f"  ⚠ {self.unprobed} candidate(s) COULD NOT BE PROBED "
                     f"(no verdict about them was possible — this is an "
                     f"instrument failure, NOT the gate being strict)")
            if self.unprobed >= max(1, self.synthesized):
                unrun += ("; NO determinacy verdict was obtained for ANY "
                          "candidate in this run (the oracle gate did run)")
        return (f"seeds {self.seeds} → synthesized {self.synthesized} → "
                f"accepted {len(self.accepted)} ({rate}) [{gates}]"
                f"{unrun}{warn}")


async def mine(seeds: Sequence[Seed], llm_client: Any, *,
               call_kwargs: Optional[Dict[str, Any]] = None,
               timeout_s: float = _VALIDATOR_TIMEOUT_S,
               probe: bool = True, probe_k: int = DEFAULT_PROBE_K,
               on_item=None) -> MineReport:
    """Seeds → verified items, through BOTH gates.

    `oracle_is_sound` answers "can this checker fail?"; `solvability_probe`
    answers "is there one right answer?". The live first run proved the
    second is not implied by the first — a chess-move item with a sound
    oracle encoded a preference as a fact and would have taught a prompt
    to reproduce an arbitrary choice.

    Every rejection is RECORDED with its reason; a miner that reports only
    its successes cannot be debugged, and the acceptance rate is the
    headline number for whether the gates are doing anything.

    `probe=False` skips the second gate (it costs `probe_k` LLM calls per
    surviving candidate). The report SAYS SO when it is skipped, because
    "accepted" means something weaker then and a reader cannot otherwise
    tell which gate an item cleared.
    """
    rep = MineReport(seeds=len(seeds), probed=bool(probe))
    for seed in seeds:
        item = await synthesize(seed, llm_client, call_kwargs=call_kwargs)
        if item is None:
            # The reason is deliberately vague because `synthesize`
            # returns None for two different things (unparseable output
            # and a refused agent-shaped challenge). Both are rejects;
            # conflating their COUNTS would deflate `synthesized` and
            # inflate the acceptance rate, so neither is counted as
            # synthesized and the string says so.
            rep.rejected.append((seed.trajectory_id,
                                 "no usable candidate (unparseable output, "
                                 "or a challenge refused as agent-shaped)"))
            continue
        rep.synthesized += 1
        ok, why = oracle_is_sound(item, timeout_s=timeout_s)
        if ok and probe:
            pr = await solvability_probe(item.to_bank_row(), llm_client,
                                         k=probe_k, call_kwargs=call_kwargs,
                                         timeout_s=timeout_s)
            if not pr.usable:
                # ⚠ PROBE_UNRUN lands here too, and that is deliberate: an
                # item admitted because its probe crashed is an item
                # admitted for no reason. Unknown is not permission — but
                # it is COUNTED separately, so the report cannot present
                # an instrument failure as a strict gate.
                if pr.verdict == PROBE_UNRUN:
                    rep.unprobed += 1
                ok, why = False, f"{pr.verdict}: {pr.why}"
            else:
                why = f"{why}; probe {pr.why}"
        if ok:
            rep.accepted.append(item)
        else:
            rep.rejected.append((item.item_id, why))
        if on_item is not None:
            try:
                on_item(item, ok, why)
            except Exception:                               # noqa: BLE001
                pass
    return rep


# ══════════════════════════════════════════════════════════════════════
# 5. Staging IO — deliberately NOT the bank directory
# ══════════════════════════════════════════════════════════════════════

#: `\Z`, not `$` — `$` also matches before a trailing newline, so
#: `"ghost_failures\n"` was ALLOWED (round 2). No traversal was possible,
#: but a boundary with a known hole is not a boundary.
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _check_name(name: str) -> str:
    """⚠ The staging/promotion separation was a PATH CONVENTION, not a
    check: `--name ../../bench/banks/live_bank` wrote straight into the
    live bank directory, arming `pick_next_item` without `--promote`. A
    boundary that only holds while nobody types the wrong thing is not a
    boundary."""
    n = str(name or "")
    if not _SAFE_NAME.fullmatch(n):
        raise ValueError(
            f"bank name {n!r} must match {_SAFE_NAME.pattern} — no path "
            f"separators; staging must not be able to reach the live bank "
            f"directory")
    return n


def staging_path(name: str = "ghost_failures",
                 home: Optional[str] = None) -> Path:
    from ..eval.banks import bench_dir
    base = Path(bench_dir(home)).parent          # $GHOST_HOME/system
    return base / "optim" / "mined_envs" / f"{_check_name(name)}.jsonl"


def write_staging(items: Sequence[MinedItem], name: str = "ghost_failures",
                  home: Optional[str] = None, *, append: bool = True) -> Path:
    """Stage items, ACCUMULATING by default.

    ⚠ Round 1 used `write_text`, so every run replaced the file. Item ids
    are `sha1(trajectory_id|challenge)` and synthesis runs at temperature
    0.3, so re-mining the same seed yields a DIFFERENT id — the bank
    churned completely on each run and any earlier `eval.banks` result
    history was orphaned. The module's own scope argument is "~12 items
    now, ~4/month accumulating toward GEPA's 20-100 range"; as written
    the corpus could never accumulate.

    De-duplicated on `item_id`, newest wins, so a re-run is idempotent
    for an unchanged item.
    """
    p = staging_path(name, home)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Dict[str, Any]] = {}
    if append:
        # ⚠ `_read_raw`, NOT `read_staging`. `read_staging` filters to the
        # CURRENT epoch, and this then rewrites the whole file — so the
        # first miner run after an epoch bump silently DELETED every
        # earlier row, plus any row with no epoch at all. `MINING_EPOCH`
        # is documented as a thing you bump; bumping it must supersede
        # the old corpus, not erase it. Superseded rows stay on disk and
        # `read_staging` keeps ignoring them.
        _kept = 0
        for old in _read_raw(name, home):
            key = str(old.get("item_id") or "")
            if key:
                rows[key] = old
                _kept += 1
        # ⚠ SAY WHAT IS BEING DROPPED. A rewrite that silently discards
        # id-less rows and unparseable lines is data loss with no trace,
        # in the function whose purpose is preservation. `_read_raw`
        # already skips unparseable lines, so the file's own line count
        # is the only honest denominator.
        try:
            _lines = sum(1 for ln in staging_path(name, home).read_text(
                errors="replace").splitlines() if ln.strip())
            if _lines > _kept:
                logger.warning(
                    "staging rewrite is dropping %d unusable row(s) of %d "
                    "(unparseable, or carrying no item_id)",
                    _lines - _kept, _lines)
        except Exception:                                   # noqa: BLE001
            pass
    for i in items:
        r = i.to_bank_row()
        key = str(r.get("item_id") or "")
        if not key:
            # An id-less row cannot be deduped or retracted; two of them
            # collapsed into one on disk and then vanished on the next
            # append.
            logger.debug("refusing to stage an item with no id")
            continue
        rows[key] = r
    # Atomic: a crash mid-write truncated the corpus this function exists
    # to protect.
    # ⚠ PER-PROCESS. One fixed `.tmp` shared by every writer makes the
    # "atomic" guarantee hold only for a single process — round 3 ran two
    # concurrent writers and one trial lost 400 items outright. Unique
    # per pid, so concurrent writers cannot tear each other's temp file.
    # (Last-writer-still-wins on the final `replace`; this is a single-
    # operator tool and a lock would be the wrong weight.)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows.values()), encoding="utf-8")
    tmp.replace(p)
    return p


def _read_raw(name: str = "ghost_failures",
              home: Optional[str] = None) -> List[Dict[str, Any]]:
    """EVERY staged row, any epoch. Only `write_staging` uses this — it
    must preserve what it does not serve."""
    p = staging_path(name, home)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
        except Exception:                                   # noqa: BLE001
            continue
        if isinstance(v, dict):
            rows.append(v)
    return rows


def read_staging(name: str = "ghost_failures",
                 home: Optional[str] = None) -> List[Dict[str, Any]]:
    """Staged rows from the CURRENT epoch. Rows from a superseded epoch
    are not comparable and are ignored — but they are not deleted."""
    return [v for v in _read_raw(name, home)
            if v.get("mining_epoch") == MINING_EPOCH]


def promote_to_bank(name: str = "ghost_failures",
                    home: Optional[str] = None) -> Optional[Path]:
    """Copy staged items into `system/bench/banks/` — an EXPLICIT operator
    act, never a side effect of mining.

    `eval.banks.pick_next_item` walks every bank in that directory and the
    biological watchdog calls it in production, so writing there arms the
    live flywheel. Mining and arming are separate decisions and the code
    keeps them separate.
    """
    rows = read_staging(name, home)
    if not rows:
        return None
    from ..eval.banks import write_bank
    return write_bank(rows, name, home)


# ══════════════════════════════════════════════════════════════════════
# 6. The GEPA bridge — the consumer this is scoped for
# ══════════════════════════════════════════════════════════════════════

#: A mined example can only be joined back to its checker if the
#: challenge text survives into the dspy gold — and `_to_dspy_examples`
#: copies ONLY the signature's declared inputs. `user_request` is the
#: field `trainset_from_items` writes the challenge into, so a signature
#: without it discards the challenge entirely: round 2 measured
#: `tool_selection.pick` golds coming through as
#: `{'step_description': '', 'tool_catalog': '', 'recent_tool_outcomes': ''}`
#: — all-empty examples that score 0.0 in BOTH arms and are pure noise.
#: Refusing is better than adding noise that looks like data.
_JOINABLE_INPUT = "user_request"


def signature_can_use_mined(sig: Any) -> bool:
    """Can mined examples reach this signature's metric at all?"""
    try:
        return _JOINABLE_INPUT in set(getattr(sig, "inputs", ()) or ())
    except Exception:                                       # noqa: BLE001
        return False


def trainset_from_items(items: Sequence[Dict[str, Any]],
                        signature_name: str,
                        outputs: Optional[Sequence[str]] = None) -> List[Any]:
    """Mined items → `TrainExample`s for one signature.

    ⚠ `origin="bench"` IS THE SAFETY MECHANISM, not bookkeeping.
    `trainset.real_only_gate` moves bench examples out of the PRIVATE
    ship-gate tier: bench may TEACH, it may never GRADE (§4BH). These
    items are synthetic and must never decide a promotion — tagging them
    correctly is the entire way that is enforced, and the equal-mass
    re-cap in the same function keeps them from swamping the public tier.

    ⚠ TEXT-GRADED ITEMS ONLY. GEPA optimises an instruction that produces
    TEXT; there is no agent loop in a GEPA rollout to write a
    `solution.py`, so an artifact-graded item would be scored against a
    file the candidate cannot produce and would fail uniformly — a metric
    that can only reject, which §4F already shipped once (both arms at the
    noise floor because the token budget starved the content phase).
    Artifact items are dropped here, and the caller is expected to say so.
    """
    from .trainset import TrainExample
    out = []
    for row in items or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("graded_on") or GRADED_ARTIFACT) != GRADED_TEXT:
            continue
        challenge = str(row.get("challenge") or "").strip()
        ref = str(row.get("reference_answer") or "")
        if not challenge or not ref.strip():
            continue
        out.append(TrainExample(
            signature_name=signature_name,
            inputs={"user_request": challenge,
                    "cluster": str(row.get("cluster") or ""),
                    "tier": ""},
            # The reference is carried so the existing overlap metric still
            # has a target when a caller does not use the oracle metric —
            # but `oracle_score` is the reason this module exists, and the
            # bank row travels with the example so the checker is always
            # reachable from the thing being scored.
            # ⚠ THE SIGNATURE'S OWN OUTPUT FIELDS. Round 1 stamped
            # `{"final_response": ref, "plan": ""}` — and `run_gepa`'s
            # `keyed` filter twelve lines later keeps only examples with
            # a truthy field named in `sig.outputs`. `plan` was empty and
            # `final_response` is not a signature output, so **100% of
            # mined examples were dropped**, measured on the live corpus:
            # 0 of 1 survived. The §4CV consumer added in round 1 had
            # exactly the defect it was added to remove — the third time
            # this loop has been built unwired.
            expected_output=({f: ref for f in outputs} if outputs
                             else {"final_response": ref, "plan": ref}),
            source_trajectory_id=str(row.get("item_id") or ""),
            weight=1.0,
            origin="bench",
        ))
    return out


def oracle_score(row: Dict[str, Any], produced_text: str, *,
                 timeout_s: float = _VALIDATOR_TIMEOUT_S) -> Optional[float]:
    """1.0 / 0.0 by RUNNING the item's checker against a candidate output.

    This is the point of the module: a verifiable reward in place of token
    overlap against a recorded reply.

    Returns **None**, not 0.0, when the checker could not be run. A metric
    that scores an infrastructure failure as "the candidate was wrong"
    punishes prompts for a broken box, and the resulting optimisation is
    against noise. The caller decides what to do with None — the honest
    options are "skip this example" or "abort the run", never "call it
    zero".
    """
    validator = str((row or {}).get("validation_script") or "")
    if not validator.strip():
        return None
    rc = _run_validator(validator, str(produced_text or ""), timeout_s)
    if rc is None:
        return None
    return 1.0 if rc == 0 else 0.0


__all__ = [
    "MINING_EPOCH", "GRADED_TEXT", "GRADED_ARTIFACT", "Seed", "MinedItem",
    "MineReport", "mine_seeds", "synthesize", "validator_static_defects",
    "oracle_is_sound", "mine", "staging_path", "write_staging",
    "solvability_probe", "ProbeResult", "DEFAULT_PROBE_K",
    "signature_can_use_mined",
    "PROBE_ARBITRARY", "PROBE_TRIVIAL", "PROBE_SEPARATES",
    "PROBE_UNRUN",
    "read_staging", "promote_to_bank", "trainset_from_items", "oracle_score",
]
