"""§4RB option 1 measurement: does the effort sentence still steer the 27B when the template puts it at
the END of the system block (after the tools) instead of the start?  Pre-registered rule (2026-10-24):
  PASS iff (1) moved: median thinking tokens low < medium < xhigh;
           (2) per level, median(moved/official thinking) within [0.6, 1.6];
           (3) moved correct >= official correct - 1, per level (tasks T1-T6);
           (4) a level switch under the moved template re-processes <= 500 prompt tokens.
Runs on the LIVE llama server (one slot) only while the agent has no foreground request.
Resumable: results.jsonl rows are skipped when present.  python -I measure.py"""
import json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R

S = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.join(S, "..", "tpl.jinja")
MOVED = os.path.join(S, "tpl_moved.jinja")
OUT = os.path.join(S, "results.jsonl")
KEY = open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()
N_PREDICT = 6144

TASKS = [
    ("T1", "How many positive integers less than 1000 are divisible by 7 but not by 11? "
           "Put the final answer on the last line as ANSWER: <number>.", r"ANSWER:\s*130\b"),
    ("T2", "What day of the week was 1 March 2026? Put the final answer on the last line as ANSWER: <day>.",
     r"ANSWER:\s*\**Sunday"),
    ("T3", "How many times does the letter r appear in the phrase 'strawberry raspberry'? "
           "Put the final answer on the last line as ANSWER: <number>.", r"ANSWER:\s*6\b"),
    ("T4", "What does this Python print? Do not run it.\n\nx = [1, 2, 3, 4, 5]\ny = [v * v for v in x if v % 2]\n"
           "print(sum(y) - len(x))\n\nPut the final answer on the last line as ANSWER: <number>.", r"ANSWER:\s*30\b"),
    ("T5", "Alice is older than Bob. Carol is younger than Bob. Dave is older than Alice. Who is the second "
           "youngest? Put the final answer on the last line as ANSWER: <name>.", r"ANSWER:\s*\**Bob"),
    ("T6", "What is the newest llama.cpp release, and what changed in it?", r"<tool_call>"),
]
CONDITIONS = [("official", "medium"), ("official", "low"), ("official", "xhigh"), ("moved", "low"), ("moved", "xhigh")]


def _post(path, body, timeout=1800):
    req = urllib.request.Request(R.SERVER + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _idle():
    for _ in range(720):
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/health", headers={"X-Ghost-Key": KEY})
            if int(json.load(urllib.request.urlopen(req, timeout=5)).get("foreground_requests") or 0) == 0:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def _ntok(text):
    return len(_post("/tokenize", {"content": text}, timeout=60).get("tokens") or [])


def prompt(tpl, level, task_text):
    h = R.head()
    msgs = [{"role": "system", "content": h["system"]}, {"role": "user", "content": task_text}]
    return R.render(tpl, msgs, h["tools"], reasoning_effort=level)


def run_one(tpl_name, level, task):
    tid, text, rx = task
    p = prompt(OFFICIAL if tpl_name == "official" else MOVED, level, text)
    if not _idle():
        raise SystemExit("agent never idle")
    t0 = time.time()
    r = _post("/completion", {"prompt": p, "n_predict": N_PREDICT, "temperature": 0.6, "top_p": 0.95,
                              "top_k": 20, "seed": 42, "cache_prompt": True, "stream": False})
    out = r.get("content") or ""
    think, _, answer = out.partition("</think>")
    tm = r.get("timings") or {}
    return {"tpl": tpl_name, "level": level, "task": tid, "think_tokens": _ntok(think),
            "answer_tokens": _ntok(answer), "closed_think": "</think>" in out,
            "correct": bool(re.search(rx, answer if "</think>" in out else "")),
            "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
            "predicted_ms": tm.get("predicted_ms"), "wall_s": round(time.time() - t0, 1),
            "answer_tail": answer.strip()[-200:], "t": time.strftime("%H:%M:%S")}


def switch_cost():
    """(4): prompt tokens a level switch re-processes, each template, back-to-back 1-token requests."""
    rows = []
    for tpl_name in ("moved", "official"):
        tpl = OFFICIAL if tpl_name == "official" else MOVED
        seq = ["low", "xhigh", "low"]
        if not _idle():
            raise SystemExit("agent never idle")
        for lv in seq:
            p = prompt(tpl, lv, TASKS[0][1])
            r = _post("/completion", {"prompt": p, "n_predict": 1, "cache_prompt": True, "temperature": 0})
            tm = r.get("timings") or {}
            rows.append({"kind": "switch", "tpl": tpl_name, "level": lv, "prompt_n": tm.get("prompt_n"),
                         "prompt_ms": tm.get("prompt_ms"), "t": time.strftime("%H:%M:%S")})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            d = json.loads(line)
            done.add((d.get("kind", "run"), d.get("tpl"), d.get("level"), d.get("task")))
    with open(OUT, "a") as f:
        if ("switch", "moved", "low", None) not in done:
            for row in switch_cost():
                f.write(json.dumps(row) + "\n"); f.flush()
        for tpl_name, level in CONDITIONS:
            for task in TASKS:
                if ("run", tpl_name, level, task[0]) in done:
                    continue
                row = {"kind": "run", **run_one(tpl_name, level, task)}
                f.write(json.dumps(row) + "\n"); f.flush()
                print(json.dumps({k: row[k] for k in ("tpl", "level", "task", "think_tokens", "correct",
                                                       "closed_think", "prompt_n", "wall_s", "t")}), flush=True)


if __name__ == "__main__":
    main()
