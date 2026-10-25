"""§4RB option 1, round 2: does the effort sentence still steer the 27B when moved to the END of the
system block?  Pre-registered 2026-10-24, before any round-2 run:
  PASS iff (a) moved: median think low < medium < xhigh, and moved xhigh/low ratio of medians
               >= 0.5 x the official's;
           (b) per level, median over task x seed of moved/official think tokens within [0.67, 1.5];
           (c) runaways (no </think> within the cap): moved <= official + 1, per level;
           (d) well-formed outputs (think closed AND a parseable <tool_call> or a non-empty answer):
               moved >= official - 1, per level;
           (e) a clean level switch under the moved template re-processes <= 1000 prompt tokens
               (switch2.json).
Medium renders identically under both templates (one shared condition)."""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2.jsonl")
CAP = 8192
SEEDS = (1, 2, 3)
TASKS = [
    ("H1", "Five houses in a row are painted red, green, blue, white and yellow, in some order. The green "
           "house is immediately left of the white one. The red house is in the middle. The blue house is at "
           "one end. The yellow house is not next to the blue one. Which colour is the leftmost house? "
           "Explain briefly."),
    ("H2", "A train leaves at 09:40 and travels 210 km at 84 km/h, stops 25 minutes, then travels 150 km at "
           "100 km/h. At what time does it arrive? Show the steps."),
    ("H3", "This function should return the length of the longest strictly increasing contiguous run in a "
           "list, but it is wrong for some inputs. Find the bug and give the fixed code.\n\n"
           "def longest_run(xs):\n    best = cur = 1\n    for i in range(1, len(xs)):\n        if xs[i] > xs[i-1]:\n"
           "            cur += 1\n        else:\n            best = max(best, cur)\n            cur = 1\n    return best\n"),
    ("H4", "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals "
           "given as (start, end) tuples, and state its time complexity."),
    ("H5", "I want a small web page that shows a countdown to New Year with days, hours, minutes and seconds. "
           "Build it in the workspace as countdown.html."),
    ("H6", "What day of the week was 1 March 2026? Put the final answer on the last line as ANSWER: <day>."),
]
CONDITIONS = [("official", "medium"), ("official", "low"), ("moved", "low"), ("official", "xhigh"), ("moved", "xhigh")]


def well_formed(out):
    if "</think>" not in out:
        return False
    ans = out.split("</think>", 1)[1]
    if "<tool_call>" in ans:
        return bool(re.search(r"<tool_call>\s*<function=[\w.-]+>.*?</function>\s*</tool_call>", ans, re.S))
    return bool(ans.strip())


def run_one(tpl_name, level, task, seed):
    tid, text = task
    p = M.prompt(M.OFFICIAL if tpl_name == "official" else M.MOVED, level, text)
    M._idle()
    t0 = time.time()
    r = M._post("/completion", {"prompt": p, "n_predict": CAP, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                                "seed": seed, "cache_prompt": True, "stream": False}, timeout=3600)
    out = r.get("content") or ""
    think = out.split("</think>", 1)[0]
    tm = r.get("timings") or {}
    return {"tpl": tpl_name, "level": level, "task": tid, "seed": seed, "think_tokens": M._ntok(think),
            "closed_think": "</think>" in out, "well_formed": well_formed(out), "tool_call": "<tool_call>" in out,
            "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n"), "wall_s": round(time.time() - t0, 1),
            "tail": out[-300:], "t": time.strftime("%H:%M:%S")}


def main():
    done = set()
    if os.path.exists(OUT):
        done = {(d["tpl"], d["level"], d["task"], d["seed"]) for d in map(json.loads, open(OUT))}
    with open(OUT, "a") as f:
        for tpl_name, level in CONDITIONS:
            for task in TASKS:
                for seed in SEEDS:
                    if (tpl_name, level, task[0], seed) in done:
                        continue
                    row = run_one(tpl_name, level, task, seed)
                    f.write(json.dumps(row) + "\n"); f.flush()
                    print(json.dumps({k: row[k] for k in ("tpl", "level", "task", "seed", "think_tokens",
                                                           "closed_think", "well_formed", "wall_s", "t")}), flush=True)


if __name__ == "__main__":
    main()
