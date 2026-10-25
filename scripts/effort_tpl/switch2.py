"""Clean switch cost: low then xhigh back-to-back (1-token requests); an attempt counts only when the
llama log shows no other task between the two. Up to 4 attempts per template; keep the clean ones."""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as M
LOG = "/Users/vasilis/Data/AI/Logs/llama-server.log"

def tasks_since(off):
    with open(LOG, "rb") as f:
        f.seek(off); txt = f.read().decode("utf-8", "replace")
    return sorted(set(int(x) for x in re.findall(r"launch_slot_: id  0 \| task (\d+)", txt)))

out = []
for tpl_name in ("moved", "official"):
    tpl = M.OFFICIAL if tpl_name == "official" else M.MOVED
    for attempt in range(4):
        M._idle()
        p_low, p_x = M.prompt(tpl, "low", M.TASKS[0][1]), M.prompt(tpl, "xhigh", M.TASKS[0][1])
        M._post("/completion", {"prompt": p_low, "n_predict": 1, "cache_prompt": True, "temperature": 0})
        off = os.path.getsize(LOG)
        r = M._post("/completion", {"prompt": p_x, "n_predict": 1, "cache_prompt": True, "temperature": 0})
        n = len(tasks_since(off))
        row = {"tpl": tpl_name, "attempt": attempt, "switch_prompt_n": r["timings"]["prompt_n"],
               "switch_prompt_ms": round(r["timings"]["prompt_ms"]), "tasks_in_window": n, "clean": n == 1}
        print(json.dumps(row), flush=True); out.append(row)
        if row["clean"]:
            break
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "switch2.json"), "w"))
