"""§4RC: what the 27B's host RAM prompt cache (--cache-ram 4096) keeps when the one slot is taken.
A = the head + a ~3k-token conversation. After A, one of:
  rewarm      a head-only prompt (the agent's head re-warm),
  short       a short unrelated prompt (the look, the greeting fast path),
  short+rewarm the two in that order,
then A again. Reported: the tokens A re-processes the second time (0-ish = kept/restored).
Runs on the LIVE llama server only while the agent has no foreground request and the slot is idle.
python -I ramcache.py [reps]"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as M, render as R

LOG = "/Users/vasilis/Data/AI/Logs/llama-server.log"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ramcache.jsonl")


def chat(msgs, tools, max_tokens=1):
    return M._post("/v1/chat/completions", {"messages": msgs, "tools": tools, "max_tokens": max_tokens,
                                             "temperature": 0, "cache_prompt": True})


def n(r):
    return (r.get("timings") or {}).get("prompt_n")


def slot_idle():
    for _ in range(600):
        try:
            import urllib.request
            slots = json.load(urllib.request.urlopen(R.SERVER + "/slots", timeout=5))
            if not any(s.get("is_processing") for s in slots if isinstance(s, dict)):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


h = R.head()
conv = " ".join(f"Line {i}: the quarterly report needs the regional totals checked against the ledger."
                for i in range(160))
A = [{"role": "system", "content": h["system"]},
     {"role": "user", "content": "Please summarise this:\n" + conv}]
HEAD = [{"role": "system", "content": h["system"]}, {"role": "user", "content": "hi"}]
SHORT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say ok."}]
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for rep in range(reps):
    for case in ("rewarm", "short", "short+rewarm", "control"):
        M._idle(); slot_idle()
        off = os.path.getsize(LOG)
        a1 = chat(A, h["tools"])
        mid = []
        if case in ("short", "short+rewarm"):
            mid.append(n(chat(SHORT, None)))
        if case in ("rewarm", "short+rewarm"):
            mid.append(n(chat(HEAD, h["tools"])))
        a2 = chat(A, h["tools"])
        with open(LOG, "rb") as f:
            f.seek(off); txt = f.read().decode("utf-8", "replace")
        tasks = len(set(re.findall(r"task (\d+) \| processing task", txt)))
        row = {"case": case, "rep": rep, "A_first": n(a1), "middle": mid, "A_again": n(a2),
               "tasks": tasks, "clean": tasks == 2 + len(mid), "t": time.strftime("%H:%M:%S")}
        print(json.dumps(row), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
