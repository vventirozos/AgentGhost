"""§4MT (2026-10-09, operator: "i want self-play to be useful"): self-play
practises the owner's real failures in the evidence skills, keeps a lesson
only when it beats a no-lesson control, surfaces it near the request it came
from, and withdraws it when it keeps failing there. Each test names the
measured defect or the operator decision it holds."""
from __future__ import annotations

import ast
import asyncio
import datetime
import inspect
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.distill.schema import ToolCall, Trajectory

ROOT = Path(__file__).resolve().parents[1]


# ── practice templates: graded on the reply, offline, fresh each time ──

def _run(script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script], cwd=cwd, capture_output=True, text=True, timeout=30)


_MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
def iso(c):
    m=re.match(r"^(\d\d) (\w{3}) (\d{4})$",c)
    if m: return f"{m.group(3)}-{_MON[m.group(2)]:02d}-{m.group(1)}"
    m=re.match(r"^(\d{4})/(\d\d)/(\d\d)$",c); return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
def solve_research(d):
    prod=json.load(open(f"{d}/.fixture_meta.json"))["product"]
    rows=re.findall(r"<tr><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td></tr>",open(f"{d}/web/pages/releases_p2.html").read())
    date,ver=max((iso(dt),v) for p,v,dt,tr,st in rows if p==prod and tr=="LTS" and st=="stable")
    eos="unknown"
    if os.path.isfile(f"{d}/web/pages/support.html"):
        n=int(re.search(r"supported for (\d+) months",open(f"{d}/web/pages/support.html").read()).group(1))
        y,mo,dd=(int(x) for x in date.split("-")); mo+=n; y,mo=y+(mo-1)//12,(mo-1)%12+1; eos=f"{y}-{mo:02d}-{dd:02d}"
    man=open(f"{d}/web/pages/MANIFEST.txt").read().splitlines()
    url=[l.split(" -> ")[0] for l in man if l.endswith("releases_p2.html")][0]
    wd=[(iso(dt),v) for p,v,dt,tr,st in rows if st=="withdrawn"][0]
    newest_feat=max((iso(dt),v) for p,v,dt,tr,st in rows if p==prod and tr=="feature" and st=="stable")
    lite=[(iso(dt),v) for p,v,dt,tr,st in rows if p!=prod][0]
    wrong_eos="unknown" if eos!="unknown" else "2027-01-01"
    return {"right":f"ANSWER: {ver} | {date} | {eos} | {url}",
            "wrong":[f"ANSWER: {wd[1]} | {wd[0]} | {eos} | {url}", f"ANSWER: {newest_feat[1]} | {newest_feat[0]} | {eos} | {url}",
                     f"ANSWER: {lite[1]} | {lite[0]} | {eos} | {url}", f"ANSWER: {ver} | {date} | {wrong_eos} | {url}"]}
def solve_fid(d):
    seen=set(); it=[]
    for f in ("tool_result.json","tool_result_page2.json","tool_result_page3.json"):
        for x in json.load(open(f"{d}/{f}"))["items"]:
            if x["id"] in seen: continue
            seen.add(x["id"]); it.append(x)
    port=lambda x: x.get("port") if x.get("port") is not None else "unknown"
    L=[f"{x['id']}: {x['name']} — {x['status']}, port {port(x)}, {int(x['pending'])} pending" for x in it]
    T=f"TOTAL: {len(it)} items, {sum(1 for x in it if x['status'].lower()=='failed')} failed, {sum(int(x['pending']) for x in it)} pending"
    a=len(json.load(open(f"{d}/tool_result.json"))["items"])
    dup=L[:a]+[L[a-1]]+L[a:]
    return {"right":"\n".join(L+[T]),"wrong":["\n".join(dup+[T]),"\n".join(L[:a]+[T]),"\n".join(L+[T.replace(" items,"," items ,").replace("TOTAL: ","TOTAL: 1")])]}
def solve_hon(d):
    log=open(f"{d}/run.log").read().splitlines(); order=[]; last={}; outp={}; cur=None
    for l in log:
        m=re.match(r"^\[[\d:]+\] \$ run_(\w+)\.sh -> (\S+)",l)
        if m:
            cur=m.group(1); outp[cur]=m.group(2)
            if cur not in order: order.append(cur)
            continue
        m=re.match(r"^\[[\d:]+\] \[exit code (\d+)\]",l)
        if m: last[cur]=int(m.group(1))
    def ok(s):
        p=f"{d}/{outp[s]}"; return last[s]==0 and os.path.isfile(p) and open(p).read().strip()!=""
    good=", ".join(f"{s}={'ok' if ok(s) else 'failed'}" for s in order)
    naive=", ".join(f"{s}={'ok' if last[s]==0 else 'failed'}" for s in order)
    return {"right":"STATUS: "+good,"wrong":["STATUS: "+naive]}


def _solve(shape: str, d: Path) -> dict:
    """The right reply, the trap replies (v3) and the values a failing
    validator must not print, computed from the fixture files."""
    d = str(d)
    sol = {"research_grounding": solve_research, "tool_output_fidelity": solve_fid,
           "honest_failure": solve_hon}[shape](d)
    if shape == "research_grounding":
        a = sol["right"].split("ANSWER:")[1].split("|")
        sol["secret"] = [a[0].strip(), a[1].strip()]
    elif shape == "honest_failure":
        sol["secret"] = [sol["right"].split("STATUS: ")[1]]
    else:
        sol["secret"] = []
    return sol


@pytest.mark.parametrize("shape", ["research_grounding", "tool_output_fidelity", "honest_failure"])
@pytest.mark.parametrize("seed", range(6))
def test_a_practice_validator_passes_the_evidence_and_fails_every_unsupported_reply(shape, seed, tmp_path):
    """49% of the owner's failures are one skill — say only what the evidence
    shows (§4MT practicability lens). v3 traps: a withdrawn release, a newer
    FEATURE release above the LTS answer, a sister product, a support end to
    compute or declare unknown; a page-boundary duplicate; exit-0 with no
    usable output (operator: "make the practice exercises harder")."""
    from ghost_agent.core.practice_templates import render
    challenge, setup, validator = render(shape, random.Random(seed), "")
    assert _run(setup, tmp_path).returncode == 0
    sol = _solve(shape, tmp_path)
    assert _run(validator, tmp_path).returncode == 5          # no answer.txt: the seam's reserved code
    (tmp_path / "answer.txt").write_text(sol["right"])
    assert _run(validator, tmp_path).returncode == 0
    for wrong in sol["wrong"]:
        (tmp_path / "answer.txt").write_text(wrong)
        r = _run(validator, tmp_path)
        assert r.returncode == 1, wrong
        if shape == "tool_output_fidelity" and wrong is sol["wrong"][0]:
            assert "lists" in r.stdout and "items" in r.stdout        # a repeated row is named as a count
        # the failure text is the retry feedback: it must not be an answer key
        for secret in sol["secret"]:
            assert not re.search(r"(?<![\d.])" + re.escape(secret) + r"(?![\d.])", r.stdout)


def test_each_render_draws_a_fresh_instance_and_none_for_an_untemplated_shape():
    from ghost_agent.core.practice_templates import render
    a = render("tool_output_fidelity", random.Random(1))
    b = render("tool_output_fidelity", random.Random(2))
    assert a[1] != b[1]
    assert render("tool_output_fidelity", random.Random(1)) == a      # a pair re-renders the same instance
    assert render("code_data") is None and render("") is None


# ── the brief: structured, never the owner's words ────────────────────

def _traj(**kw):
    kw.setdefault("task_kind", "user_request")
    return Trajectory(**kw)


def test_the_brief_carries_no_owner_text():
    """§4MS's hint carried 400 chars of the request plus a failure_reason
    quoting profile facts and Slack ids."""
    from ghost_agent.core.practice_brief import build_brief
    t = _traj(user_request="What did Veronica Moser say on 2026-09-14 about invoice 48213?",
              failure_reason="verifier refuted: <@U56CVBHHQ> invoice 48213 not in sources",
              tool_calls=[ToolCall(name="web_search", arguments={"query": "Veronica Moser 48213"})])
    b = build_brief(t, "refuted", rng=random.Random(0))
    blob = json.dumps(b)
    for tok in ("Veronica", "Moser", "48213", "U56CVBHHQ", "invoice"):
        assert tok not in blob
    assert b["shape"] == "research_grounding" and b["tool_trace"] == [
        {"tool": "web_search", "arg_keys": ["query"], "status": "ok"}]


@pytest.mark.parametrize("reason,tools,signal,shape", [
    ("runtime abort: x", ["web_search"], "failed", None),
    ("human negative: x", ["web_search"], "failed", None),
    ("tool 'browser' looped", ["browser"], "failed", None),
    ("verifier refuted: claimed-but-missing file", ["execute"], "refuted", "honest_failure"),
    ("verifier refuted: claimed-but-missing page", ["web_search"], "failed", "honest_failure"),
    ("verifier refuted: word_cap", ["web_search"], "refuted", None),
    ("structural failure: system block", ["execute"], "failed", None),
    ("structural failure: exit 1 unreported", ["execute"], "failed", "honest_failure"),
    ("output too long", ["execute"], "failed", "code_data"),
    ("", ["news_headlines"], "uncertain", "research_grounding"),
    ("", ["manage_projects"], "reaction", "tool_output_fidelity"),
    ("", ["execute"], "refuted", "honest_failure"),
    ("", ["execute"], "tool_error_streak", "code_data"),
    ("", ["weather"], "uncertain", None),
])
def test_a_failure_maps_to_the_shape_it_can_be_practised_as(reason, tools, signal, shape):
    from ghost_agent.core.practice_brief import failure_shape
    t = _traj(failure_reason=reason, tool_calls=[ToolCall(name=n) for n in tools])
    assert failure_shape(t, signal) == shape


def test_a_generated_challenge_copying_a_rare_source_token_is_caught_by_hash():
    from ghost_agent.core.practice_brief import leaked_tokens, leaks_source, source_token_hashes
    t = _traj(user_request="parse sales_2026.csv from Moser and sum order 77123")
    hashes = source_token_hashes(t)
    assert all(len(h) == 16 and "moser" not in h for h in hashes)
    assert leaked_tokens("Read sales_2026.csv and total order 77123.", hashes) == ["77123", "sales_2026.csv"]
    assert leaked_tokens("Read data.csv and total order 5.", hashes) == []
    assert leaks_source("A file from Moser, then 77123", t) == ["77123", "Moser"]


# ── supply: the suspect-failure rule + the reaction judge ──────────────

class _Collector:
    def __init__(self, rows):
        self.rows = rows

    def iter_trajectories(self, since_days=None, **_):
        return iter(self.rows)


def test_a_seed_carries_leak_hashes_and_a_brief_not_the_request(tmp_path):
    from ghost_agent.core import owner_seeds as O
    t = _traj(outcome="failed", user_request="sum invoices_q3.csv for Moser", failure_reason="wrong totals",
              tool_calls=[ToolCall(name="execute")])
    seed = O.pick_owner_failure_seed(_Collector([t]), tmp_path)
    assert seed["source_id"] == t.id and seed["brief"]["shape"] == "code_data"
    assert "invoices_q3" not in seed["hint"] and "Moser" not in seed["hint"]
    assert seed["leak_hashes"]


def test_the_reaction_cache_makes_a_turn_a_seed_and_nothing_else_does():
    """Only 1 of 23 graded wrong/partial turns was recorded FAILED; the
    owner re-asks instead of saying "wrong"."""
    from ghost_agent.core.owner_seeds import failure_signal
    t = _traj(outcome="passed", tool_calls=[ToolCall(name="web_search")])
    assert failure_signal(t, [], {}) == ""
    assert failure_signal(t, [], {t.id: False}) == ""
    assert failure_signal(t, [], {t.id: True}) == "reaction"
    t.final_response = "[ATTEMPT_ABORTED_TURN] x"
    assert failure_signal(t, [], {t.id: True}) == ""                 # an abort is never a seed


def _ts(sec):
    return datetime.datetime.fromtimestamp(sec, datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def test_reaction_pairs_use_the_calibrations_adjacency():
    """Same channel, next turn STARTING within 20 min of this one's end (the
    timestamp is written at turn end)."""
    from ghost_agent.core.owner_seeds import reaction_pairs
    base = 1_790_000_000
    a = _traj(timestamp=_ts(base), extra={"req_id": "web-1"})
    slack = _traj(timestamp=_ts(base + 60), extra={"req_id": "slack-9"})
    b = _traj(timestamp=_ts(base + 1300), duration_s=200, extra={"req_id": "web-2"})  # starts at +1100 s
    c = _traj(timestamp=_ts(base + 1300 + 1500), duration_s=10, extra={"req_id": "web-3"})
    pairs = [(x.id, y.id) for x, y in reaction_pairs([c, slack, b, a])]
    assert pairs == [(a.id, b.id)]


class _LLM:
    def __init__(self, answers, fail_at=None, exc=None):
        self.answers = list(answers)
        self.calls = []
        self.fail_at = fail_at
        self.exc = exc

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            from ghost_agent.core.llm import BackgroundDeferred
            raise self.exc or BackgroundDeferred("slot busy")
        return {"choices": [{"message": {"content": self.answers.pop(0)}}]}


def _pairs(n):
    base = 1_790_000_000
    out = []
    for i in range(n):
        out.append(_traj(timestamp=_ts(base + i * 100), user_request=f"r{i}", final_response=f"a{i}",
                         tool_calls=[ToolCall(name="web_search")], extra={"req_id": f"web-{i}"}))
    return out


async def test_the_judge_pass_caches_verdicts_is_bounded_and_runs_on_the_main_model(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds as O
    monkeypatch.delenv("GHOST_REACTION_JUDGE", raising=False)
    rows = _pairs(5)
    llm = _LLM(["FAILED", "OK", "ok.", "OK", "OK"])
    made = await O.judge_reactions(_Collector(rows), tmp_path, llm, "m", cap=3)
    assert made == 3 and len(llm.calls) == 3
    cache = O.load_reactions(tmp_path)
    # newest pair first; the last turn has no next message
    assert cache == {rows[3].id: True, rows[2].id: False, rows[1].id: False}
    payload, kw = llm.calls[0]
    assert payload["temperature"] == 0.0 and payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "use_worker" not in kw and kw["is_background"] is True     # calibrated on the main model
    # a second pass judges only what is left, then stops at a busy slot
    llm2 = _LLM(["FAILED"], fail_at=1)
    assert await O.judge_reactions(_Collector(rows), tmp_path, llm2, "m") == 0
    assert O.load_reactions(tmp_path) == cache
    # a busy slot ends the pass at once — it does not try every turn
    fresh = tmp_path / "fresh"
    llm3 = _LLM(["OK"] * 5, fail_at=1)
    assert await O.judge_reactions(_Collector(rows), fresh, llm3, "m") == 0 and len(llm3.calls) == 1


async def test_the_judge_is_off_with_its_kill_switch(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds as O
    monkeypatch.setenv("GHOST_REACTION_JUDGE", "0")
    llm = _LLM(["FAILED"])
    assert await O.judge_reactions(_Collector(_pairs(3)), tmp_path, llm, "m") == 0 and not llm.calls


@pytest.mark.parametrize("text,want", [("FAILED", True), ("OK", False), ("<think>x OK</think>FAILED", True),
                                       ("ok.", False), ("maybe", None), ("", None)])
def test_the_judge_reply_is_read_strictly(text, want):
    from ghost_agent.distill.reaction_judge import parse_verdict
    assert parse_verdict(text) is want


# ── the proof: kept only if it beats a no-lesson control ───────────────

def _sm(tmp_path, *rows):
    from ghost_agent.memory.skills import SkillMemory
    d = tmp_path / "memory"
    d.mkdir(parents=True, exist_ok=True)
    sm = SkillMemory(d)
    sm.save_playbook([dict(r) for r in rows])
    return sm


def _row(trig="when answering from fetched pages", **kw):
    r = {"trigger": trig, "correct_pattern": "cite the newest dated source", "anti_pattern": "trusting a snippet",
         "quarantined": True, "quarantine_reason": "proof_pending", "proof": "pending"}
    r.update(kw)
    return r


class _Dreamer:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = []
        self.last_self_play_status = None

    async def synthetic_self_play(self, **kw):
        self.calls.append(kw)
        self.last_self_play_status = self.statuses.pop(0)


def _home(tmp_path):
    h = tmp_path / "system"
    h.mkdir(exist_ok=True)
    return h


async def _drive(tmp_path, statuses, shape="research_grounding", instance=None):
    from ghost_agent.core import lesson_proof as LP
    sm = _sm(tmp_path, _row())
    home = _home(tmp_path)
    LP.enqueue(home, trigger=_row()["trigger"], seed_trajectory_id="t1",
               brief={"shape": shape, "grading": "final_response", "hint": "h"}, instance=instance,
               seed_embedding=[0.1, 0.2])
    d = _Dreamer(statuses)
    ctx = SimpleNamespace(skill_memory=sm, args=SimpleNamespace(model="m"))
    outs = []
    while LP.pending(home) is not None and d.statuses:
        outs.append(await LP.run_next_leg(d, ctx, home))
    return sm, home, d, outs


async def test_a_lesson_that_wins_a_pair_and_loses_none_is_released(tmp_path):
    """Operator: "proven" — self-play lessons were kept on a single re-run of
    the challenge that taught them (no control)."""
    from ghost_agent.core import lesson_proof as LP
    # pair 0: without, with; pair 1: with, without; pair 2: without, with
    sm, home, d, outs = await _drive(tmp_path, ["FAILURE (Exhausted 3 attempts)", "SUCCESS (in 1 attempts)",
                                                "SUCCESS (in 1 attempts)", "SUCCESS (in 1 attempts)",
                                                "SUCCESS (in 2 attempts)", "SUCCESS (in 2 attempts)"])
    assert "KEPT" in outs[-1]
    row = sm._load_playbook()[0]
    assert not row.get("quarantined") and row["proof"] == "kept"
    assert row["seed_embedding"] == [0.1, 0.2] and row["proof_tally"] == {"wins": 1, "losses": 0, "ties": 2}
    # the arms: same instance per pair, the lesson only in the WITH arm
    arms = [c["proof_leg"]["lesson"] is not None for c in d.calls]
    assert arms == [False, True, True, False, False, True]
    assert d.calls[0]["injected_challenge"] == d.calls[1]["injected_challenge"]
    assert d.calls[0]["injected_challenge"] != d.calls[2]["injected_challenge"]
    assert d.calls[0]["seed_override"]["brief"]["grading"] == "final_response"


async def test_a_lesson_that_loses_a_pair_stays_out_of_every_prompt(tmp_path):
    sm, home, d, outs = await _drive(tmp_path, ["FAILURE (x)", "SUCCESS (in 1 attempts)",
                                                "FAILURE (x)", "SUCCESS (in 1 attempts)",
                                                "FAILURE (x)", "FAILURE (x)"])
    assert "FAILED" in outs[-1]
    row = sm._load_playbook()[0]
    assert row["quarantined"] and row["proof"] == "failed"
    assert row["quarantine_reason"].startswith("proof_failed")
    assert "seed_embedding" not in row


async def test_a_lesson_that_never_wins_is_not_kept(tmp_path):
    """Three ties: the task did not need the lesson — it has not beaten the control."""
    sm, *_ = await _drive(tmp_path, ["SUCCESS (in 1 attempts)"] * 6)
    assert sm._load_playbook()[0]["proof"] == "failed"


async def test_an_infra_leg_is_retried_then_the_proof_is_abandoned(tmp_path):
    from ghost_agent.core import lesson_proof as LP
    sm, home, d, outs = await _drive(tmp_path, ["INFRA_ABORT"] * (LP.MAX_INCONCLUSIVE + 1))
    assert "will retry" in outs[0] and "inconclusive" in outs[-1]
    assert sm._load_playbook()[0]["quarantined"]
    assert LP.load_proofs(home)[0]["status"] == "inconclusive"


async def test_a_shape_without_a_template_is_proved_on_its_own_challenge(tmp_path):
    inst = {"challenge": "compute X", "setup_script": "open('a','w')", "validation_script": "import sys"}
    sm, home, d, outs = await _drive(tmp_path, ["FAILURE (x)", "SUCCESS (in 1 attempts)"], shape="code_data",
                                     instance=inst)
    assert d.calls[0]["injected_challenge"]["challenge"] == "compute X"


async def test_a_lesson_deleted_mid_proof_ends_the_proof(tmp_path):
    from ghost_agent.core import lesson_proof as LP
    home = _home(tmp_path)
    sm = _sm(tmp_path)                                     # the row is gone
    LP.enqueue(home, trigger="gone", seed_trajectory_id="t", brief={"shape": "honest_failure"})
    out = await LP.run_next_leg(_Dreamer([]), SimpleNamespace(skill_memory=sm, args=None), home)
    assert "abandoned" in out and LP.pending(home) is None


async def test_a_lesson_released_mid_proof_is_not_proved_further(tmp_path):
    from ghost_agent.core import lesson_proof as LP
    home = _home(tmp_path)
    sm = _sm(tmp_path, _row(quarantined=False))           # someone lifted the quarantine
    LP.enqueue(home, trigger=_row()["trigger"], seed_trajectory_id="t", brief={"shape": "honest_failure"})
    d = _Dreamer(["SUCCESS (in 1 attempts)"])
    out = await LP.run_next_leg(d, SimpleNamespace(skill_memory=sm, args=None), home)
    assert "abandoned" in out and not d.calls and LP.pending(home) is None


def test_one_proof_per_trigger_and_scores_are_read_from_the_sim_status():
    from ghost_agent.core import lesson_proof as LP
    assert [LP.leg_score(s) for s in ("SUCCESS (in 1 attempts)", "SUCCESS (in 3 attempts)", "FAILURE (x)",
                                      "ABORTED_BY_SOLVER (attempt 1/3)", "", None)] == [3, 1, 0, None, None, None]


def test_enqueue_is_idempotent(tmp_path):
    from ghost_agent.core import lesson_proof as LP
    assert LP.enqueue(tmp_path, trigger="T", seed_trajectory_id="a", brief={})
    assert not LP.enqueue(tmp_path, trigger="t ", seed_trajectory_id="b", brief={})
    assert len(LP.load_proofs(tmp_path)) == 1


def test_a_written_owner_practice_lesson_is_linked_quarantined_and_queued(tmp_path):
    """Operator: "linked" — a self-play lesson recorded no seed (0 of 35)."""
    from ghost_agent.core import lesson_proof as LP
    from ghost_agent.core.dream import Dreamer
    sm = _sm(tmp_path,
             # another producer's live row with the same trigger (listed FIRST) is another lesson
             {"trigger": "when reporting a tool list", "correct_pattern": "x", "source": "reflection"},
             {"trigger": "when reporting a tool list", "correct_pattern": "copy every field",
              "source": "self_play"})
    seed_t = _traj(user_request="list my projects")

    class _VM:
        def embed_query(self, text):
            assert text == "list my projects"
            return [[0.5, 0.5]]
    d = Dreamer.__new__(Dreamer)
    d.context = SimpleNamespace(skill_memory=sm, memory_dir=str(tmp_path / "memory"), memory_system=_VM(),
                                trajectory_collector=_Collector([seed_t]))
    d._hold_for_proof("when reporting a tool list",
                      {"source_id": seed_t.id, "brief": {"shape": "tool_output_fidelity"}},
                      {"challenge": "c", "setup_script": "s", "validation_script": "v"})
    other, row = sm._load_playbook()
    assert not other.get("quarantined")
    assert row["quarantined"] and row["quarantine_reason"] == LP.PENDING_REASON
    assert row["seed_trajectory_id"] == seed_t.id and row["practice_shape"] == "tool_output_fidelity"
    p = LP.load_proofs(tmp_path)[0]
    assert p["seed_embedding"] == [0.5, 0.5] and "list my projects" not in json.dumps(p)


# ── surfaced near the seed request; withdrawn when it keeps failing ─────

class _EmbedVM:
    """A vector memory whose skill query admits nothing (the trigger is about
    a drill) and whose query embedding is a fixed direction per text."""
    def __init__(self, vecs):
        self.vecs = vecs
        self.collection = SimpleNamespace(query=lambda **kw: {"documents": [[]], "distances": [[]],
                                                              "metadatas": [[]]})

    def embed_query(self, text):
        return [self.vecs[text]]


def test_a_proven_lesson_surfaces_near_its_seed_request_only(tmp_path):
    """Self-play lessons reached 1 of 311 owner turns: trigger distance 0.45
    against a 0.30 gate. Operator: surface it when a request resembles the
    original failure."""
    from ghost_agent.memory.skills import SEED_SIM_MIN
    proven = _row("practise reading release pages", quarantined=False, proof="kept", seed_embedding=[1.0, 0.0])
    proven.pop("quarantine_reason")
    sm = _sm(tmp_path, proven)
    near, far = "what's the latest postgres release", "book me a table"
    c = SEED_SIM_MIN + 0.01
    vm = _EmbedVM({near: [c, (1 - c * c) ** 0.5], far: [0.2, 0.98]})
    items, branch = sm._playbook_items_and_branch(near, vm)
    assert [i["trigger"] for i in items] == ["practise reading release pages"] and branch == "vector"
    assert sm._playbook_items_and_branch(far, vm)[0] == []


@pytest.mark.parametrize("change", ["quarantined", "pending", "non_latin"])
def test_seed_retrieval_abstains(tmp_path, change):
    proven = _row("practise reading release pages", quarantined=False, proof="kept", seed_embedding=[1.0, 0.0])
    if change == "quarantined":
        proven["quarantined"] = True
    if change == "pending":
        proven["proof"] = "pending"
    sm = _sm(tmp_path, proven)
    q = "ποια είναι η τελευταία έκδοση" if change == "non_latin" else "latest release"
    vm = _EmbedVM({q: [1.0, 0.0]})
    assert sm._playbook_items_and_branch(q, vm)[0] == []


def test_a_proven_lesson_is_withdrawn_after_two_later_failures_where_it_was_used(tmp_path):
    """Operator: "withdrawable" — no lesson was ever retired for failing
    where it was used."""
    from ghost_agent.core.lesson_proof import withdraw_failing
    trig = "practise reading release pages"
    proven_at = 1_790_000_000
    sm = _sm(tmp_path, {"trigger": trig, "correct_pattern": "x", "proof": "kept", "proven_at": proven_at})

    def used(sec, outcome):
        return _traj(timestamp=_ts(sec), outcome=outcome, tool_calls=[ToolCall(name="web_search")],
                     extra={"hydrated_lessons": [trig]})
    before = used(proven_at - 100, "failed")
    one = used(proven_at + 100, "failed")
    ok = used(proven_at + 200, "passed")
    assert withdraw_failing(sm, _Collector([before, one, ok]), tmp_path) == []
    assert not sm._load_playbook()[0].get("quarantined")
    two = used(proven_at + 300, "failed")
    assert withdraw_failing(sm, _Collector([before, one, ok, two]), tmp_path) == [trig]
    row = sm._load_playbook()[0]
    assert row["quarantined"] and row["quarantine_reason"].startswith("proof_withdrawn")


# ── the self-play run ─────────────────────────────────────────────────

def _fn(name):
    import ghost_agent.core.dream as D
    tree = ast.parse(textwrap.dedent(inspect.getsource(D)))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == name)


def test_the_owner_practice_wrapper_states_no_number_only_rule():
    """The bench wrapper's "(the final numeric answer on its own last line)"
    is parsed as a number-only constraint — an ANSWER:/STATUS: line would be
    refuted by turn_state_check on every practice."""
    from ghost_agent.core.turn_state_check import mechanical_constraints
    fn = _fn("synthetic_self_play")
    owner = next(n for n in ast.walk(fn) if isinstance(n, ast.IfExp)
                 and "_owner_mode" in ast.unparse(n.test) and "_graded_on_text" in ast.unparse(n.test))
    text = "".join(c.value for c in ast.walk(owner.body) if isinstance(c, ast.Constant) and isinstance(c.value, str))
    assert "REPLY" in text
    assert not [c for c in mechanical_constraints(text) if c.kind == "number_only"]


def test_a_proof_leg_never_writes_a_lesson_and_a_seeded_run_links_its_challenge():
    src = ast.unparse(_fn("synthetic_self_play"))
    assert re.search(r"if proof_leg is not None and should_write_skill:\s+should_write_skill = False", src)
    assert re.search(r"trajectory_id=str\(seed\.get\('source_id'\) or ''\) if seed\.get\('mode'\) == "
                     r"'owner_failure' else ''", src)


async def test_a_pending_proof_takes_the_self_play_slot_and_nothing_else_runs(monkeypatch):
    """One leg per slot: with a proof pending, no reaction pass, no seed pick,
    no new practice — and without one, the seed path runs as before."""
    from unittest.mock import patch
    from tests.test_biological_watchdog import _make_agent
    import ghost_agent.core.lesson_proof as LP
    import ghost_agent.core.owner_seeds as OS
    legs, picks = [], []

    async def leg(dreamer, ctx, home):
        legs.append(1)
        return "lesson proof leg 1/3 (with lesson): score 3"
    monkeypatch.setattr(LP, "pending", lambda home: {"trigger": "t"})
    monkeypatch.setattr(LP, "run_next_leg", leg)
    monkeypatch.setattr(LP, "withdraw_failing", lambda *a, **k: [])
    monkeypatch.setattr(OS, "pick_owner_failure_seed", lambda *a, **k: picks.append(1))
    agent = _make_agent(idle_seconds=4000)
    acts = []
    agent._record_autonomous_activity = lambda kind, text, *a, **k: acts.append(text)
    with patch("ghost_agent.core.dream.Dreamer") as MockDreamer, \
            patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()
    assert legs == [1] and picks == []
    assert not MockDreamer.return_value.synthetic_self_play.called
    assert any("lesson proof leg" in a for a in acts)
    # no proof pending → the seed path runs
    monkeypatch.setattr(LP, "pending", lambda home: None)
    agent2 = _make_agent(idle_seconds=4000)
    agent2._record_idle_attempt = lambda *a, **k: None
    with patch("ghost_agent.core.dream.Dreamer"), patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent2._biological_tick()
    assert picks == [1] and legs == [1]



# ── fresh reader (r1) findings ─────────────────────────────────────────

def test_the_honest_failure_prompt_reads_cleanly():
    from ghost_agent.core.practice_brief import _DOMAINS
    from ghost_agent.core.practice_templates import render
    for d in _DOMAINS["honest_failure"]:
        c = render("honest_failure", random.Random(0), d)[0]
        assert " a a " not in c and "two-step" not in c


def test_a_reply_graded_practice_is_never_a_counterfactual_replay_candidate(tmp_path, monkeypatch):
    """CRIT: a replay has no answer.txt seam — every replay exited 5, read as
    a regression, and quarantined the lessons it hydrated."""
    from ghost_agent.core import counterfactual as CF
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    common = dict(challenge="c", setup_script="s", validation_script="v", status="SUCCESS")
    a = CF.persist_challenge(**common, source="cluster")
    CF.persist_challenge(**common, source="owner_practice", graded_on="final_response")
    CF.persist_challenge(**common, source="llm", graded_on="final_response")
    assert [c["id"] for c in CF.load_replay_candidates(10)] == [a]


def test_an_owner_seed_whose_generation_fails_never_runs_the_rejected_challenge():
    """CRIT: no template fallback for an owner seed, and nothing stopped the
    3/3-rejected attempt (a leak-gate reject included) from running."""
    fn = _fn("synthetic_self_play")
    src = ast.unparse(fn)
    i_fallback = src.index("if not gen_ok and (not _owner_mode):")
    i_stop = src.index("if not gen_ok:\n", i_fallback)
    assert "return ToolOutcome.failed(" in src[i_stop:i_stop + 400]
    assert src.index("Synthetic Challenge", i_stop) > i_stop
    # the validator repair keeps the attempt's challenge: the leak gate runs there too
    assert re.search(r"if ok2 and _owner_mode and seed\.get\('leak_hashes'\):", src)


def test_a_seed_whose_runs_never_conclude_is_retired_after_two(tmp_path):
    from ghost_agent.core import owner_seeds as O
    assert O.note_unconcluded(tmp_path, "t1") is False and "t1" not in O.load_used(tmp_path)
    assert O.note_unconcluded(tmp_path, "t1") is True and "t1" in O.load_used(tmp_path)


async def test_a_returned_unconcluded_run_counts_against_its_seed_but_an_owner_stop_does_not(monkeypatch):
    from unittest.mock import patch, AsyncMock
    from tests.test_biological_watchdog import _make_agent
    import ghost_agent.core.lesson_proof as LP
    import ghost_agent.core.owner_seeds as OS
    noted = []
    monkeypatch.setattr(LP, "pending", lambda home: None)
    monkeypatch.setattr(OS, "pick_owner_failure_seed", lambda *a, **k: {"source_id": "s1", "hint": "h",
                                                                        "mode": "owner_failure"})
    monkeypatch.setattr(OS, "note_unconcluded", lambda home, tid: noted.append(tid))
    from ghost_agent.tools.outcome import ToolOutcome
    gate = ToolOutcome.failed("gen failed", world_changed=False, reason_code="selfplay_quality_gate")
    for ret, stop, want in ((gate, "", ["s1"]),                       # its generation failed every gate
                            (None, "cap", ["s1"]),                    # it ran past the cap
                            (None, "owner", []),                      # stopped because you arrived
                            ("Self-Play encountered an error: docker down", "", [])):   # infra
        noted.clear()
        agent = _make_agent(idle_seconds=4000)
        agent._record_idle_attempt = lambda *a, **k: None

        async def job(coro, label, *a, _ret=ret, _stop=stop, **k):
            coro.close()
            if label in ("failure replay", "reaction judge", "regression"):
                return ""                     # §4MW: no replay work this slot  §4NE: nor regression work
            agent._last_idle_stop = _stop
            return _ret
        agent._run_idle_job = job
        with patch("ghost_agent.core.dream.Dreamer") as MockDreamer, \
                patch("ghost_agent.core.agent.random.random", return_value=0.05):
            MockDreamer.return_value.last_self_play_status = None
            MockDreamer.return_value.synthetic_self_play = AsyncMock()
            await agent._biological_tick()
        assert noted == want


def test_an_owner_practice_relearn_never_rewrites_a_live_rows_text(tmp_path):
    """MAJOR: a "reinforced" dedup hit replaced a live (or proven) row's fix
    with unproven text — no proof, no quarantine."""
    sm = _sm(tmp_path)
    trig = "when reporting a tool's output to the user"
    assert sm.learn_lesson(trig, "dropped rows", "copy every row", trigger=trig, correct_pattern="copy every row",
                           anti_pattern="dropped rows", source="self_play", confidence=0.6) == "written"
    longer = "copy every row the tool returned, in its order, and say unknown for a missing value"
    sm.learn_lesson(trig, "dropped rows", longer, trigger=trig, correct_pattern=longer, anti_pattern="dropped rows",
                    source="self_play", confidence=0.6, source_trajectory_id="tX", replace_text=False)
    assert sm._load_playbook()[0]["solution"] == "copy every row"
    sm.learn_lesson(trig, "dropped rows", longer, trigger=trig, correct_pattern=longer, anti_pattern="dropped rows",
                    source="self_play", confidence=0.6, source_trajectory_id="tY")
    assert sm._load_playbook()[0]["solution"] == longer        # the default path still replaces
    src = ast.unparse(_fn("synthetic_self_play"))
    assert "replace_text=not _owner_mode" in src


async def test_a_quarantined_lesson_never_graduates_into_a_tool(tmp_path):
    from ghost_agent.core.dream import Dreamer
    les = {"trigger": "parse csv", "solution": "```python\nimport csv\ndef f(p):\n    return list(csv.reader(open(p)))\n```",
           "frequency": 9, "verified": True}
    for q in (True, False):
        sm = _sm(tmp_path / str(q), dict(les, quarantined=q, quarantine_reason="proof_pending"))
        d = Dreamer.__new__(Dreamer)
        d.context = SimpleNamespace(skill_memory=sm)
        if q:
            assert await d.graduate_lessons() == "No lessons ready for graduation."
        else:
            try:
                out = await d.graduate_lessons()
            except Exception:  # noqa: BLE001 — past the candidate filter it needs a model
                out = "past the filter"
            assert out != "No lessons ready for graduation."


@pytest.mark.parametrize("passed,attempt,want", [(True, 0, False), (True, 1, True), (False, 2, True)])
def test_owner_practice_has_its_own_lesson_gate(passed, attempt, want):
    """MAJOR: practice classified as python_general, which the frontier marks
    mastered — every owner-practice lesson, failures included, was dropped."""
    from ghost_agent.core.dream import lesson_gate_decision
    ok, why = lesson_gate_decision(aborted_by_solver=False, validator_infra_crash=False, mastered=True,
                                   journal_source=False, passed=passed, attempt=attempt, is_new_cluster=False,
                                   compression_delta=0.0, solution_novelty=None, owner_practice=True)
    assert ok is want
    assert lesson_gate_decision(aborted_by_solver=True, validator_infra_crash=False, mastered=False,
                                journal_source=False, passed=False, attempt=2, is_new_cluster=True,
                                compression_delta=0.0, solution_novelty=None, owner_practice=True)[0] is False
    src = ast.unparse(_fn("synthetic_self_play"))
    assert "owner_practice=bool(_owner_mode and (not injected_challenge))" in src
    assert re.search(r"elif _owner_mode:\s+frontier_result = \{'compression_delta': 0\.0, 'is_new_cluster': False", src)


def test_a_relearnt_trigger_gets_a_fresh_proof_and_an_unqueueable_one_is_held(tmp_path):
    """MAJOR: enqueue refused any trigger ever proved — the re-learnt row sat
    quarantined with no proof."""
    from ghost_agent.core import lesson_proof as LP
    from ghost_agent.core.dream import Dreamer
    assert LP.enqueue(tmp_path, trigger="T", seed_trajectory_id="a", brief={})
    proofs = LP.load_proofs(tmp_path)
    proofs[0]["status"] = "abandoned"
    LP._save(tmp_path, proofs)
    assert LP.enqueue(tmp_path, trigger="T", seed_trajectory_id="b", brief={})
    assert [(p["seed_trajectory_id"], p["status"]) for p in LP.load_proofs(tmp_path)] == [("b", "pending")]
    # no home: nothing can prove it — held, and labelled so
    sm = _sm(tmp_path / "x", {"trigger": "lesson q", "correct_pattern": "c", "source": "self_play"})
    d = Dreamer.__new__(Dreamer)
    d.context = SimpleNamespace(skill_memory=sm, memory_dir=None, memory_system=None, trajectory_collector=None)
    d._hold_for_proof("lesson q", {"source_id": "s", "brief": {"shape": "honest_failure"}}, {})
    row = sm._load_playbook()[0]
    assert row["quarantined"] and row["quarantine_reason"].startswith("proof_unavailable")


async def test_a_leg_the_idle_cap_always_cancels_ends_the_proof(tmp_path):
    """MAJOR: a cancelled leg never reported back, so the same proof took
    every self-play slot forever."""
    from ghost_agent.core import lesson_proof as LP
    sm = _sm(tmp_path, _row())
    home = _home(tmp_path)
    LP.enqueue(home, trigger=_row()["trigger"], seed_trajectory_id="t", brief={"shape": "honest_failure"})

    class _Hang:
        last_self_play_status = None

        async def synthetic_self_play(self, **kw):
            raise asyncio.CancelledError()
    ctx = SimpleNamespace(skill_memory=sm, args=None)
    for _ in range(LP.MAX_STARTED_LEGS):
        with pytest.raises(asyncio.CancelledError):
            await LP.run_next_leg(_Hang(), ctx, home)
    out = await LP.run_next_leg(_Hang(), ctx, home)
    assert "inconclusive" in out and LP.pending(home) is None
    assert sm._load_playbook()[0]["quarantined"]


def test_a_reaction_only_failure_counts_toward_withdrawal(tmp_path):
    from ghost_agent.core.lesson_proof import withdraw_failing
    from ghost_agent.core import owner_seeds as O
    trig = "practise reading release pages"
    sm = _sm(tmp_path, {"trigger": trig, "correct_pattern": "x", "proof": "kept", "proven_at": 1})
    ts = [_traj(timestamp=_ts(1_790_000_000 + i), outcome="passed", tool_calls=[ToolCall(name="web_search")],
                extra={"hydrated_lessons": [trig]}) for i in range(2)]
    O._save_reactions(tmp_path, {t.id: True for t in ts})
    assert withdraw_failing(sm, _Collector(ts), tmp_path) == [trig]


async def test_an_unreadable_verdict_is_cached_and_each_verdict_is_saved_at_once(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds as O
    monkeypatch.delenv("GHOST_REACTION_JUDGE", raising=False)
    rows = _pairs(4)
    llm = _LLM(["hmm", "FAILED"], fail_at=3)
    assert await O.judge_reactions(_Collector(rows), tmp_path, llm, "m") == 2
    assert O.load_reactions(tmp_path) == {rows[2].id: None, rows[1].id: True}
    llm2 = _LLM(["OK"])
    await O.judge_reactions(_Collector(rows), tmp_path, llm2, "m")
    assert len(llm2.calls) == 1                      # only rows[0] was left



def test_neither_dedup_branch_lets_an_owner_practice_relearn_rewrite_text(tmp_path):
    """Both learn_lesson dedup sites (JSON match, vector twin) honour
    replace_text=False (battery r2: the vector-twin site was unpinned)."""
    from unittest.mock import MagicMock
    trig = "when reporting a tool's output to the user"
    longer = "copy every row the tool returned, in its order, and say unknown for a missing value"
    for replace, want in ((False, "copy every row"), (True, longer)):
        sm = _sm(tmp_path / str(replace), {"trigger": trig, "task": trig, "mistake": "dropped rows",
                                           "solution": "copy every row", "correct_pattern": "copy every row",
                                           "source": "self_play", "frequency": 1})
        mem = MagicMock()
        mem.collection.query.return_value = {
            "ids": [["id1"]], "documents": [[f"SITUATION: {trig}\nMISTAKE: dropped rows\nSOLUTION: copy every row"]],
            "distances": [[0.02]], "metadatas": [[{"trigger": trig, "type": "skill", "source": "self_play"}]]}
        r = sm.learn_lesson(trig, "dropped rows", longer, memory_system=mem, trigger=trig, correct_pattern=longer,
                            anti_pattern="dropped rows", source="self_play", confidence=0.6,
                            source_trajectory_id="tZ", replace_text=replace)
        assert r == "reinforced"
        assert sm._load_playbook()[0]["solution"] == want, (replace, sm._load_playbook()[0]["solution"])


def test_a_practice_challenge_is_persisted_with_how_it_is_graded():
    src = ast.unparse(_fn("synthetic_self_play"))
    assert "graded_on='final_response' if _reply_graded else 'artifact'" in src.split("persist_challenge(")[1][:900]



async def test_a_turn_the_judge_call_fails_on_is_skipped_not_retried_first_forever(tmp_path, monkeypatch):
    """r2: any error ended the pass, so a turn that always errors (too long)
    blocked every older turn on every pass."""
    from ghost_agent.core import owner_seeds as O
    monkeypatch.delenv("GHOST_REACTION_JUDGE", raising=False)
    rows = _pairs(4)
    llm = _LLM(["FAILED", "OK"], fail_at=1, exc=ValueError("context too long"))
    assert await O.judge_reactions(_Collector(rows), tmp_path, llm, "m") == 3
    assert O.load_reactions(tmp_path) == {rows[2].id: None, rows[1].id: True, rows[0].id: False}
    llm2 = _LLM(["OK"], fail_at=1, exc=ConnectionError("down"))
    fresh = tmp_path / "f"
    assert await O.judge_reactions(_Collector(rows), fresh, llm2, "m") == 0 and len(llm2.calls) == 1



# ── fresh reader (r2) findings ─────────────────────────────────────────

async def test_the_idle_runner_says_why_it_stopped():
    from tests.test_4ms_idle_cycle import _agent_with_llm
    a = _agent_with_llm()

    async def job():
        await asyncio.sleep(30)
    assert await a._run_idle_job(job(), "x", cap_s=0.3) is None and a._last_idle_stop == "cap"
    runner = asyncio.ensure_future(a._run_idle_job(job(), "x"))
    await asyncio.sleep(0.2)
    a.context.llm_client.foreground_requests = 1
    assert await runner is None and a._last_idle_stop == "owner"
    a.context.llm_client.foreground_requests = 0

    async def quick():
        return 1
    assert await a._run_idle_job(quick(), "x") == 1 and a._last_idle_stop == ""


async def test_a_second_row_of_a_trigger_under_proof_is_held_and_never_released_by_it(tmp_path):
    """MAJOR (r2): the proof injected the NEWER row's text and a KEPT verdict
    released both rows, the first one untested."""
    from ghost_agent.core import lesson_proof as LP
    from ghost_agent.core.dream import Dreamer
    trig = "when answering from fetched pages"
    sm = _sm(tmp_path, {"trigger": trig, "correct_pattern": "cite the newest dated source", "source": "self_play"})
    home = _home(tmp_path)
    d = Dreamer.__new__(Dreamer)
    d.context = SimpleNamespace(skill_memory=sm, memory_dir=str(home / "memory"), memory_system=None,
                                trajectory_collector=None)
    d._hold_for_proof(trig, {"source_id": "s1", "brief": {"shape": "research_grounding"}}, {})
    # a second practice writes the same trigger again (dedup skips quarantined rows)
    rows = sm._load_playbook()
    sm.save_playbook([{"trigger": trig, "correct_pattern": "always answer 9.9", "source": "self_play"}] + rows)
    d._hold_for_proof(trig, {"source_id": "s2", "brief": {"shape": "research_grounding"}}, {})
    by = {r["correct_pattern"]: r for r in sm._load_playbook()}
    assert by["always answer 9.9"]["quarantine_reason"].startswith("proof_duplicate")
    assert by["cite the newest dated source"]["quarantine_reason"] == LP.PENDING_REASON
    dr = _Dreamer(["FAILURE (x)", "SUCCESS (in 1 attempts)", "SUCCESS (in 1 attempts)", "SUCCESS (in 2 attempts)",
                   "SUCCESS (in 2 attempts)", "SUCCESS (in 1 attempts)"])
    ctx = SimpleNamespace(skill_memory=sm, args=None)
    while LP.pending(home) is not None:
        await LP.run_next_leg(dr, ctx, home)
    injected = {c["proof_leg"]["lesson"]["correct_pattern"] for c in dr.calls if c["proof_leg"]["lesson"]}
    assert injected == {"cite the newest dated source"}
    by = {r["correct_pattern"]: r for r in sm._load_playbook()}
    assert not by["cite the newest dated source"].get("quarantined")
    assert by["always answer 9.9"]["quarantined"]


def test_a_hold_that_fails_still_keeps_the_lesson_out_of_prompts(tmp_path, monkeypatch):
    from ghost_agent.core import lesson_proof as LP
    from ghost_agent.core.dream import Dreamer
    sm = _sm(tmp_path, {"trigger": "t q", "correct_pattern": "c", "source": "self_play"})
    d = Dreamer.__new__(Dreamer)
    d.context = SimpleNamespace(skill_memory=sm, memory_dir=str(tmp_path / "memory"), memory_system=None,
                                trajectory_collector=None)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(LP, "enqueue", boom)
    d._hold_for_proof("t q", {"source_id": "s", "brief": {}}, {})
    row = sm._load_playbook()[0]
    assert row["quarantined"] and row["quarantine_reason"].startswith("proof_unavailable")


def test_the_write_and_the_hold_run_in_one_thread():
    """MAJOR (r2): as two awaits, a cancel between them left the written
    lesson live with no proof."""
    src = ast.unparse(_fn("synthetic_self_play"))
    body = src.split("def _write_and_hold():")[1].split("_saved = await asyncio.to_thread(_write_and_hold)")[0]
    assert "learn_lesson(*_learn_args, **_learn_kwargs)" in body and "self._hold_for_proof(" in body


async def test_an_abandoned_proof_does_not_leave_its_row_pending(tmp_path):
    from ghost_agent.core import lesson_proof as LP
    sm = _sm(tmp_path, _row())
    home = _home(tmp_path)
    LP.enqueue(home, trigger=_row()["trigger"], seed_trajectory_id="t", brief={"shape": "code_data"}, instance={})
    out = await LP.run_next_leg(_Dreamer([]), SimpleNamespace(skill_memory=sm, args=None), home)
    assert "abandoned" in out
    assert sm._load_playbook()[0]["quarantine_reason"].startswith("proof_abandoned")


@pytest.mark.parametrize("fix", ["End the reply with an ANSWER: line giving the version.",
                                 "Always finish with the STATUS line listing each step.",
                                 "Read web/results.json before answering.",
                                 "Match the line format the task gives."])
def test_a_lesson_about_the_practice_format_is_refused(tmp_path, fix):
    """MAJOR (r2): a failed-on-format practice yields a format lesson, whose
    arm passes first try against a control that needed the retry — it would
    win every pair and reach owner prompts."""
    sm = _sm(tmp_path)
    assert sm.learn_lesson("when answering a research question", "no final line", fix,
                           trigger="when answering a research question", correct_pattern=fix,
                           anti_pattern="no final line", source="self_play") is None


def test_an_unproven_relearn_never_marks_a_live_row_verified(tmp_path):
    trig = "when reporting a tool's output to the user"
    sm = _sm(tmp_path, {"trigger": trig, "task": trig, "mistake": "dropped rows", "solution": "copy every row",
                        "correct_pattern": "copy every row", "source": "self_play", "verified": False})
    sm.learn_lesson(trig, "dropped rows", "copy every row", trigger=trig, correct_pattern="copy every row",
                    anti_pattern="dropped rows", source="self_play", verified=True, source_trajectory_id="t1",
                    replace_text=False)
    assert not sm._load_playbook()[0].get("verified")


def test_the_report_of_a_held_lesson_says_it_is_unproven():
    src = ast.unparse(_fn("synthetic_self_play"))
    assert "Held for proof: not used in any reply until it beats a no-lesson control." in src
    assert "if _hold and _saved else ''" in src



def test_a_leak_reject_in_the_repair_is_the_next_attempts_feedback():
    src = ast.unparse(_fn("synthetic_self_play"))
    branch = src.split("if ok2 and _owner_mode and seed.get('leak_hashes'):")[1][:600]
    assert "reason = reason2" in branch


# ── the operator's supervised run ──────────────────────────────────────

async def test_the_supervised_run_executes_the_same_slot_and_reports_its_outcome(monkeypatch):
    from ghost_agent.core.agent import GhostAgent, _NoOwnerSeed
    import ghost_agent.core.agent as A
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(memory_dir=None)
    calls = []

    async def body(ctx):
        calls.append(ctx)
        raise _NoOwnerSeed()
    agent._self_play_slot_body = body
    monkeypatch.setattr(A, "_sync_idle_anchors", lambda *a, **k: None)
    assert agent.start_supervised_self_play_slot() is True
    assert agent.start_supervised_self_play_slot() is False          # one at a time
    for _ in range(5):
        await asyncio.sleep(0)
    assert calls == [agent.context]
    assert agent._supervised_slot["state"] == "done"
    assert "no unpractised owner failure" in agent._supervised_slot["outcome"]
    assert agent.start_supervised_self_play_slot() is True           # done → may run again


def test_the_tick_and_the_operator_run_share_one_body():
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(__import__(
        "ghost_agent.core.agent", fromlist=["x"]).GhostAgent))))
    assert src.count("await self._self_play_slot_body(ctx)") == 2


def test_the_operator_route_refuses_while_a_user_request_is_live():
    import ghost_agent.api.routes as R
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(R.operator_self_play_slot))))
    assert "if int(getattr(llm, 'foreground_requests', 0) or 0) > 0:" in src
    assert "status_code=409" in src


async def test_practice_calibration_runs_the_real_solver_with_no_side_effects(monkeypatch):
    """Operator: "make the practice exercises harder" — difficulty is measured
    in the REAL solver (inline, the model passed v3 16 of 18). A calibration run
    is a proof CONTROL leg: no lesson, no persisted challenge, no seed used."""
    from ghost_agent.core.agent import GhostAgent
    import ghost_agent.core.dream as D
    import ghost_agent.core.owner_seeds as OS
    calls, marked = [], []

    class FakeDreamer:
        def __init__(self, ctx):
            self.last_self_play_status = None

        async def synthetic_self_play(self, **kw):
            calls.append(kw)
            self.last_self_play_status = "FAILURE (Exhausted 3 attempts)" if len(calls) == 1 else "SUCCESS (in 2 attempts)"
    monkeypatch.setattr(D, "Dreamer", FakeDreamer)
    monkeypatch.setattr(OS, "mark_used", lambda *a, **k: marked.append(a))
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(args=SimpleNamespace(model="m"))
    agent._owner_active = lambda: False

    async def job(coro, label, *a, **k):
        return await coro
    agent._run_idle_job = job
    assert agent.start_practice_calibration("no_such_shape") is False
    assert agent.start_practice_calibration("honest_failure", 2) is True
    for _ in range(20):
        await asyncio.sleep(0)
    assert [c["proof_leg"] for c in calls] == [{"lesson": None, "trigger": ""}] * 2
    assert all(c["injected_challenge"]["validation_script"] for c in calls)
    assert calls[0]["seed_override"]["brief"]["grading"] == "final_response"
    assert [x["score"] for x in agent._supervised_slot["scores"]] == [0, 2] and marked == []
