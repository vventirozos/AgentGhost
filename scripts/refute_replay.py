#!/usr/bin/env python
"""Live replay of the turn-gate verdict on labelled, recorded turns.

Re-runs the agent's REAL verdict computation —
``GhostAgent._compute_verifier_verdict`` — on turns rebuilt from the
trajectory corpus, against the LIVE critic/main nodes, so an OLD and a NEW
copy of the verifier code can be compared on the same labelled set:

    # baseline snapshot
    PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python \\
        scripts/refute_replay.py \\
        --src /Users/vasilis/.claude/jobs/d97bd0f9/tmp/baseline/src \\
        --out /tmp/refute_replay.baseline.jsonl
    # current repo
    PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python \\
        scripts/refute_replay.py --out /tmp/refute_replay.new.jsonl

WHAT IS FAITHFUL
  * the verdict method itself, the Verifier, and the LLMClient wiring
    (upstream + ``--critic-nodes`` pools parsed from the launcher's exec line
    through ``ghost_agent.main.parse_args``, keyword-built like main.py);
  * every ``export GHOST_*=…`` literal in the launcher (critic think flag,
    MAIN_STAGE_STOP, …) is exported here BEFORE ghost_agent is imported;
  * ``tools_run_this_turn`` rows carry a ``ToolOutcome`` with the call's
    parsed args and a status (``coerce`` OR the recorded ``error``), the same
    shape the dispatch loop appends; ``messages`` carry the assistant
    ``tool_calls`` (OpenAI shape, JSON-string arguments) that
    ``_reconstruct_executed_code`` walks back to;
  * ``lc = last_user_content.lower()``.

WHAT IS NOT (be explicit when reading the numbers)
  * tool results are the trajectory's copy: foresight-note stripped and
    CLIPPED at 4000 chars by ``_reconstruct_tool_calls``;
  * no earlier conversation turns (``--history N`` approximates them from
    the same session file, which interleaves conversations — opt-in);
  * no project store (active-constraint note / ledger block are empty);
  * no browser tool, so WEB-EXEC is always inconclusive (0.6 confirm cap);
  * the §4BR/§4EC depth decision is taken from the recorded
    ``extra.verify_depth_deep`` (``--depth-live`` recomputes it on an empty
    context instead);
  * the sandbox is the LIVE one, read-only (files may have changed since the
    turn); ``--sandbox-dir`` points it elsewhere;
  * member turns are replayed as the owner (the member gate would skip them).

NO LIVE WRITES: GHOST_HOME is a temp dir (escalation ledger, claim-binding
shadow, verdict sidecar all land there and are read back per case); the
turn-facts ring is in-memory; no memory stores are constructed.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO / "src"
DEFAULT_LABELS = Path("/Users/vasilis/.claude/jobs/d97bd0f9/tmp/labels.json")
LIVE_HOME = Path("/Users/vasilis/Data/AI/Data")
TRAJ_ROOT = LIVE_HOME / "system" / "trajectories"
LAUNCHER = Path("/Users/vasilis/Data/AI/bin/start-ghost-agent.sh")
KEY_FILE = Path("/Users/vasilis/Data/AI/.ghost_api_key")

# Used only if the launcher cannot be read/parsed (kept in sync by hand with
# the 2026-09-25 exec line).
_FALLBACK_ARGV = [
    "--upstream-url", "http://127.0.0.1:8088",
    "--visual-nodes", "http://127.0.0.1:8088|Eva",
    "--worker-nodes", "http://100.83.184.117:8088|Nova",
    "--critic-nodes", "http://100.83.184.117:8088|Nova",
    "--deep-reason", "--smart-memory", "0.9", "--max-context", "240000",
    "--enable-metacog", "--use-planning", "--postmortem",
]
_FALLBACK_ENV = {
    "GHOST_PIN_TOOL_SCHEMAS": "1", "GHOST_LLM_RECORD": "0",
    "GHOST_CRITIC_ASYNC": "1", "GHOST_VERIFY_MAIN_STAGE_STOP": "1",
    "GHOST_CRITIC_NO_THINK": "0",
}
# Launcher flags that start servers / guards, not verdict behaviour.
_DROP_FLAGS_WITH_VALUE = {"--host", "--port", "--image-gen-nodes"}
_DROP_FLAGS = {"--mandatory-tor", "--autoadvance-idle"}


# ── pure parts (tested offline) ──────────────────────────────────────────
def parse_launcher(text: str) -> Tuple[Dict[str, str], List[str]]:
    """``(env, argv)`` from the launcher script: every uncommented
    ``export GHOST_X=literal`` (values with ``$`` are skipped — they are
    computed, e.g. the API key) and the argv of the ``exec … ghost_agent.main``
    line, continuation lines joined, ``"$@"`` and server-only flags dropped."""
    env: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*export\s+(GHOST_[A-Z0-9_]+)=(.*)$", line)
        if not m:
            continue
        val = m.group(2).strip()
        # computed values and secrets (the API key is read from its file, and
        # the launcher's missing-file branch exports an EMPTY one) are skipped
        if "$" in val or re.search(r"KEY|TOKEN|SECRET", m.group(1)):
            continue
        try:
            parts = shlex.split(val, comments=True)
        except ValueError:
            continue
        env[m.group(1)] = parts[0] if parts else ""
    lines = text.splitlines()
    argv: List[str] = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*exec\s+.*ghost_agent\.main", line) and "--help" not in line:
            buf = []
            j = i
            while j < len(lines):
                cur = lines[j].rstrip()
                cont = cur.endswith("\\")
                buf.append(cur[:-1] if cont else cur)
                if not cont:
                    break
                j += 1
            toks = shlex.split(" ".join(buf))
            k = next(n for n, t in enumerate(toks) if t.endswith("ghost_agent.main"))
            raw = toks[k + 1:]
            n = 0
            while n < len(raw):
                t = raw[n]
                if t in _DROP_FLAGS_WITH_VALUE:
                    n += 2
                    continue
                if t in _DROP_FLAGS or t == "$@":
                    n += 1
                    continue
                argv.append(t)
                n += 1
            break
    return env, argv


def parse_call_args(a: Any) -> dict:
    """A recorded call's arguments as a dict (dict, JSON text or a repr)."""
    if isinstance(a, dict):
        return a
    if isinstance(a, str) and a.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(a)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return {}


def build_turn(rec: dict, *,
               wrap: Optional[Callable[[str, str, dict, str], Any]] = None,
               history: Optional[List[dict]] = None,
               include_system: bool = True) -> dict:
    """Rebuild the verdict inputs from one trajectory record.

    ``wrap(result_text, name, args, error)`` turns a result into the tool
    row's content (the live harness returns a ``ToolOutcome``); default is
    the plain text. ``history`` = earlier records whose user_request /
    final_response become prior user/assistant messages."""
    last_user = str(rec.get("user_request") or "")
    final = str(rec.get("final_response") or "")
    messages: List[dict] = []
    sp = rec.get("system_prompt")
    if include_system and isinstance(sp, str) and sp.strip():
        messages.append({"role": "system", "content": sp})
    for h in history or []:
        hu, ha = str(h.get("user_request") or ""), str(h.get("final_response") or "")
        if hu:
            messages.append({"role": "user", "content": hu})
        if ha:
            messages.append({"role": "assistant", "content": ha})
    messages.append({"role": "user", "content": last_user})
    tools_run: List[dict] = []
    for i, tc in enumerate(rec.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("name") or "")
        args = parse_call_args(tc.get("arguments", tc.get("args")))
        tid = f"call_replay_{i}"
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": tid, "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args, ensure_ascii=False)}}],
        })
        text = "" if tc.get("result") is None else str(tc.get("result"))
        err = str(tc.get("error") or "")
        content = wrap(text, name, args, err) if wrap else text
        tool_msg = {"role": "tool", "tool_call_id": tid, "name": name, "content": content}
        messages.append(tool_msg)
        tools_run.append(tool_msg)
    messages.append({"role": "assistant", "content": final})
    return {"tools_run_this_turn": tools_run, "messages": messages,
            "final_ai_content": final, "last_user_content": last_user,
            "lc": last_user.lower()}


def select_labels(labels: List[dict], *, only: Optional[Iterable[str]] = None,
                  cases: Optional[Iterable[str]] = None,
                  limit: Optional[int] = None) -> List[dict]:
    only_s = {s.strip().upper() for s in (only or []) if s.strip()}
    case_s = set()
    for c in cases or []:
        c = c.strip()
        if c:
            case_s.add(c)
            case_s.add(c[:-5] if c.endswith(".json") else c + ".json")
    out = []
    for lab in labels:
        if only_s and str(lab.get("verdict", "")).upper() not in only_s:
            continue
        if case_s and not ({str(lab.get("case", "")), str(lab.get("trajectory_id", "")),
                            str(lab.get("req_id", ""))} & case_s):
            continue
        out.append(lab)
    return out[:limit] if limit else out


def index_trajectories(root: Path, wanted: Iterable[str],
                       with_history: bool = False) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    """``id -> record`` for every wanted id, plus (optionally) the records
    that precede each one in its session file."""
    want = set(wanted)
    found: Dict[str, dict] = {}
    before: Dict[str, List[dict]] = {}
    for f in sorted(root.glob("*/*.jsonl")):
        session: List[dict] = []
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                hit = any(w in line[:200] for w in want) if want else False
                if not hit and not with_history:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get("id")
                if rid in want and rid not in found:
                    found[rid] = rec
                    if with_history:
                        before[rid] = list(session)
                if with_history:
                    session.append({k: rec.get(k) for k in
                                    ("id", "timestamp", "task_kind", "user_request",
                                     "final_response", "extra")})
        if len(found) == len(want) and not with_history:
            break
    return found, before


def pick_history(prior: List[dict], rec: dict, n: int) -> List[dict]:
    """Up to ``n`` preceding records of the same task_kind and requester role
    (session files interleave conversations — this is an approximation)."""
    if n <= 0:
        return []
    role = str((rec.get("extra") or {}).get("requester_role") or "")
    kind = rec.get("task_kind")
    keep = [p for p in prior if p.get("task_kind") == kind
            and str((p.get("extra") or {}).get("requester_role") or "") == role]
    return keep[-n:]


def is_refuted(verdict: Optional[str]) -> bool:
    return str(verdict or "").upper() == "REFUTED"


def summarize(rows: List[dict]) -> str:
    def _grp(label):
        return [r for r in rows if r.get("label") == label]
    lines = []
    fr, tr = _grp("FALSE_REFUTE"), _grp("TRUE_REFUTE")
    fr_ok = [r for r in fr if not r.get("error")]
    tr_ok = [r for r in tr if not r.get("error")]
    fixed = sum(1 for r in fr_ok if not is_refuted(r.get("verdict")))
    kept = sum(1 for r in tr_ok if is_refuted(r.get("verdict")))
    lines.append(f"{'label':<14}{'n':>4}{'ran':>5}{'err':>5}  result")
    lines.append(f"{'FALSE_REFUTE':<14}{len(fr):>4}{len(fr_ok):>5}{len(fr) - len(fr_ok):>5}"
                 f"  no longer REFUTED: {fixed}/{len(fr_ok)}")
    lines.append(f"{'TRUE_REFUTE':<14}{len(tr):>4}{len(tr_ok):>5}{len(tr) - len(tr_ok):>5}"
                 f"  still REFUTED:     {kept}/{len(tr_ok)}")
    other = [r for r in rows if r.get("label") not in ("FALSE_REFUTE", "TRUE_REFUTE")]
    if other:
        o_ok = [r for r in other if not r.get("error")]
        lines.append(f"{'other':<14}{len(other):>4}{len(o_ok):>5}{len(other) - len(o_ok):>5}"
                     f"  REFUTED: {sum(1 for r in o_ok if is_refuted(r.get('verdict')))}/{len(o_ok)}")
    dist: Dict[str, int] = {}
    for r in rows:
        if not r.get("error"):
            k = f"{r.get('label')}->{r.get('verdict')}"
            dist[k] = dist.get(k, 0) + 1
    if dist:
        lines.append("transitions: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    errs = [r for r in rows if r.get("error")]
    lines.append(f"errors: {len(errs)}")
    for r in errs[:10]:
        lines.append(f"  {r.get('case')}: {str(r.get('error'))[:160]}")
    return "\n".join(lines)


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


# ── live wiring ──────────────────────────────────────────────────────────
def prepare_env(src: Path, home: Path, launcher_env: Dict[str, str]) -> None:
    for k, v in launcher_env.items():
        os.environ[k] = v
    os.environ["GHOST_HOME"] = str(home)
    os.environ.setdefault("GHOST_LLM_RECORD", "0")
    if not os.environ.get("GHOST_API_KEY"):
        try:
            os.environ["GHOST_API_KEY"] = KEY_FILE.read_text().strip()
        except OSError:
            pass
    # --src FIRST, and evict any ghost_agent already imported from elsewhere.
    # (PYTHONPATH=src would otherwise let the repo copy shadow a snapshot.)
    if src.resolve() != DEFAULT_SRC.resolve():
        sys.path[:] = [p for p in sys.path
                       if Path(p or ".").resolve() != DEFAULT_SRC.resolve()]
    sys.path.insert(0, str(src))
    for m in [m for m in sys.modules if m == "ghost_agent" or m.startswith("ghost_agent.")]:
        del sys.modules[m]


def seed_home(home: Path) -> None:
    """Temp GHOST_HOME. Active (``*.json``) optimizer artifacts are COPIED
    from the live store so tuned verifier templates apply as they do live."""
    optim_live = LIVE_HOME / "system" / "optim"
    dst = home / "system" / "optim"
    dst.mkdir(parents=True, exist_ok=True)
    for f in optim_live.glob("*.json"):
        try:
            shutil.copy2(f, dst / f.name)
        except OSError:
            pass
    (home / "sandbox").mkdir(parents=True, exist_ok=True)


def build_agent(argv: List[str], home: Path, sandbox_dir: Path):
    import ghost_agent.main as gmain
    from ghost_agent.core.agent import GhostAgent, GhostContext
    from ghost_agent.core.llm import LLMClient
    from ghost_agent.core.verifier import Verifier

    saved = sys.argv
    try:
        sys.argv = ["ghost_agent.main", *argv]
        args = gmain.parse_args()
    finally:
        sys.argv = saved
    mem = home / "system" / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    tor = os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    ctx = GhostContext(args, sandbox_dir, mem, tor)
    ctx.llm_client = LLMClient(
        args.upstream_url,
        tor_proxy=ctx.tor_proxy,
        swarm_nodes=args.swarm_nodes_parsed,
        worker_nodes=args.worker_nodes_parsed,
        visual_nodes=getattr(args, "visual_nodes_parsed", None),
        coding_nodes=getattr(args, "coding_nodes_parsed", None),
        image_gen_nodes=None,
        critic_nodes=getattr(args, "critic_nodes_parsed", None),
        node_api_key=args.api_key,
    )
    ctx.verifier = Verifier(llm_client=ctx.llm_client)
    # The sidecar writer runs for real — into the temp home — so the route /
    # escalation it records is read back exactly as the live one is written.
    ctx.trajectory_collector = SimpleNamespace(root=home / "system" / "trajectories")
    agent = GhostAgent(ctx)
    # No browser / sandbox execution from a replay.
    agent.available_tools = {}
    return agent, args


def make_wrap():
    from ghost_agent.tools.outcome import ToolOutcome

    def wrap(text: str, name: str, args: dict, err: str):
        out = ToolOutcome.coerce(text)
        try:
            if err and not out.is_failure:
                out = ToolOutcome.failed(text, reason_code="replay_recorded_error",
                                         declared=False)
            return ToolOutcome(str(out), status=out.status,
                               world_changed=out.world_changed,
                               reason_code=out.reason_code,
                               declared=out.declared, call_args=args or None)
        except TypeError:        # an older ToolOutcome without call_args
            return out
    return wrap


def _read_jsonl(p: Path) -> List[dict]:
    rows = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except OSError:
        pass
    return rows


async def run_case(agent, lab: dict, rec: Optional[dict], history: List[dict], *,
                   home: Path, timeout: float, depth_live: bool, wrap) -> dict:
    t0 = time.monotonic()
    tid = str(lab.get("trajectory_id") or "")
    extra = (rec or {}).get("extra") or {}
    row: Dict[str, Any] = {
        "case": lab.get("case"), "trajectory_id": tid,
        "req_id": lab.get("req_id") or extra.get("req_id"),
        "task_kind": (rec or {}).get("task_kind"),
        "requester_role": extra.get("requester_role") or "",
        "label": lab.get("verdict"), "mechanism": lab.get("mechanism"),
        "recorded_route": lab.get("route"), "recorded_escalation": lab.get("escalation"),
        "recorded_verdict": extra.get("verifier_verdict"),
        "verdict": None, "confidence": None, "route": None, "escalation": None,
        "override": None, "escalation_ledger": [], "issues": [],
        "cheap_verdict": None, "n_tools": 0, "seconds": None, "error": None,
    }
    if rec is None:
        row["error"] = "trajectory not found"
        row["seconds"] = 0.0
        return row
    try:
        turn = build_turn(rec, wrap=wrap, history=history)
        row["n_tools"] = len(turn["tools_run_this_turn"])
        captured: Dict[str, Any] = {}
        orig_rec = agent._record_verdict_instruments

        def _capture(v_result, **kw):
            captured["verify_route"] = kw.get("verify_route")
            return orig_rec(v_result, **kw)
        agent._record_verdict_instruments = _capture
        if not depth_live:
            deep = bool(extra.get("verify_depth_deep", False))
            agent._verify_depth_for_turn = lambda *_a, **_k: deep
        req_id = f"replay-{row['req_id'] or tid[:8]}"
        v, last_tool = await asyncio.wait_for(
            agent._compute_verifier_verdict(
                tools_run_this_turn=turn["tools_run_this_turn"],
                messages=turn["messages"],
                final_ai_content=turn["final_ai_content"],
                last_user_content=turn["last_user_content"],
                lc=turn["lc"], req_id=req_id, trajectory_id=tid,
                project_id=None),
            timeout=timeout)
        if v is not None:
            vv = getattr(v, "verdict", None)
            row["verdict"] = str(getattr(vv, "value", vv))
            row["confidence"] = round(float(getattr(v, "confidence", 0.0) or 0.0), 4)
            row["escalation"] = getattr(v, "escalation", None) or None
            row["override"] = getattr(v, "override", None) or None
            row["issues"] = [str(i)[:400] for i in (getattr(v, "issues", None) or [])[:2]]
            row["cheap_verdict"] = getattr(v, "cheap_verdict", None)
            for f in ("escalated_overturn", "escalation_replaced", "confirm_withheld",
                      "objection_upheld", "objection_dismissed", "binder_decided",
                      "truncation_guarded", "escalation_downgraded"):
                if getattr(v, f, False):
                    row.setdefault("flags", []).append(f)
        row["route"] = captured.get("verify_route")
        side = [r for f in sorted((home / "system" / "verdicts").glob("*.jsonl"))
                for r in _read_jsonl(f) if r.get("trajectory_id") == tid]
        if side:
            s = side[-1]
            row["route"] = s.get("route", row["route"])
            row["escalation"] = s.get("escalation", row["escalation"]) or row["escalation"]
            row["override"] = s.get("override", row["override"]) or row["override"]
        led = home / "system" / "verifier" / "escalations.jsonl"
        row["escalation_ledger"] = [
            {k: r.get(k) for k in ("kind", "route", "outcome", "cheap_verdict", "strong_verdict")}
            for r in _read_jsonl(led) if r.get("trajectory_id") == tid]
        row["last_tool"] = (last_tool or {}).get("name") if isinstance(last_tool, dict) else None
    except asyncio.TimeoutError:
        row["error"] = f"timeout after {timeout:.0f}s"
    except Exception as e:  # noqa: BLE001 — one case never kills the run
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-1500:]
    row["seconds"] = round(time.monotonic() - t0, 1)
    return {k: _jsonable(v) for k, v in row.items()}


async def amain(ns) -> int:
    src = Path(ns.src).resolve()
    labels_all = json.loads(Path(ns.labels).read_text())
    only = ns.only.split(",") if ns.only else None
    cases = ns.cases.split(",") if ns.cases else None
    labels = select_labels(labels_all, only=only, cases=cases, limit=ns.limit)
    if not labels:
        print("no cases selected", file=sys.stderr)
        return 2
    try:
        lenv, largv = parse_launcher(LAUNCHER.read_text())
        if not largv:
            raise ValueError("no exec line")
    except Exception as e:  # noqa: BLE001
        print(f"launcher parse failed ({e}); using the built-in fallback", file=sys.stderr)
        lenv, largv = dict(_FALLBACK_ENV), list(_FALLBACK_ARGV)
    home = Path(tempfile.mkdtemp(prefix="refute_replay_home_"))
    seed_home(home)
    prepare_env(src, home, lenv)
    sandbox = Path(ns.sandbox_dir) if ns.sandbox_dir else LIVE_HOME / "sandbox"
    print(f"src={src}\nGHOST_HOME(temp)={home}\nsandbox(read)={sandbox}\n"
          f"env={json.dumps(lenv)}\nargv={' '.join(largv)}", flush=True)

    recs, before = index_trajectories(Path(ns.trajectories),
                                      [str(l.get("trajectory_id")) for l in labels],
                                      with_history=ns.history > 0)
    agent, _args = build_agent(largv, home, sandbox)
    import ghost_agent
    print(f"ghost_agent imported from {Path(ghost_agent.__file__).parent}", flush=True)
    wrap = make_wrap()

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, ns.concurrency))
    rows: List[dict] = []
    lock = asyncio.Lock()

    async def one(i, lab):
        async with sem:
            rec = recs.get(str(lab.get("trajectory_id")))
            hist = pick_history(before.get(str(lab.get("trajectory_id")), []), rec or {}, ns.history)
            r = await run_case(agent, lab, rec, hist, home=home, timeout=ns.timeout,
                               depth_live=ns.depth_live, wrap=wrap)
            r["src"] = str(src)
            async with lock:
                rows.append(r)
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"[{len(rows)}/{len(labels)}] {r['case']} label={r['label']} "
                      f"rec={r['recorded_route']}/{r['recorded_escalation']} -> "
                      f"{r['verdict']} {r['confidence']} route={r['route']} "
                      f"esc={r['escalation']} ov={r['override']} {r['seconds']}s"
                      + (f" ERROR {r['error']}" if r["error"] else ""), flush=True)

    if out.exists() and not ns.append:
        out.unlink()
    try:
        await asyncio.gather(*(one(i, l) for i, l in enumerate(labels)))
    finally:
        try:
            await agent.context.llm_client.close()
        except Exception:  # noqa: BLE001
            pass
    print("\n" + summarize(rows), flush=True)
    if not ns.keep_home:
        shutil.rmtree(home, ignore_errors=True)
    else:
        print(f"temp home kept: {home}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", default=str(DEFAULT_SRC),
                   help="src dir whose ghost_agent is imported (default: repo src)")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", required=True, help="output JSONL (overwritten unless --append)")
    p.add_argument("--append", action="store_true")
    p.add_argument("--only", default="", help="comma list of labels, e.g. FALSE_REFUTE,TRUE_REFUTE")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cases", default="", help="comma list of case / trajectory_id / req_id")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=float, default=240.0, help="per-case seconds")
    p.add_argument("--trajectories", default=str(TRAJ_ROOT))
    p.add_argument("--sandbox-dir", default="",
                   help="sandbox root the file/visual checks read (default: live, read-only)")
    p.add_argument("--history", type=int, default=0,
                   help="prepend up to N earlier same-session turns (approximation)")
    p.add_argument("--depth-live", action="store_true",
                   help="recompute the depth decision instead of using the recorded one")
    p.add_argument("--keep-home", action="store_true", help="keep the temp GHOST_HOME")
    ns = p.parse_args(argv)
    return asyncio.run(amain(ns))


if __name__ == "__main__":
    sys.exit(main())
