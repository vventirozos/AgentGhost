"""Thinking-in-history cache check: after a tool-call turn, how many prompt tokens does the NEXT turn
re-process when the assistant turn is sent back (strip) without its thinking — what Ghost does now —
versus (keep) with reasoning_content?  Uses /v1/chat/completions on the live server (the server parses
and renders, as for the agent).  Clean only when the llama log shows no other task in the window.
python -I hist.py [reps]"""
import json, os, re, sys, time, urllib.request
sys.path.insert(0, "/private/tmp/claude-501/-Users-vasilis-Data-AI-Agent/b215a9f5-572f-4cfd-8d27-27d2d096231c/scratchpad/effort_tpl")
import measure as M, render as R
LOG = "/Users/vasilis/Data/AI/Logs/llama-server.log"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hist.jsonl")
ASKS = ["What is the newest llama.cpp release, and what changed in it?",
        "Search the web for today's weather forecast in Athens and summarise it.",
        "Find the current version of Python on python.org and tell me its release date."]

def tasks_since(off):
    with open(LOG, "rb") as f:
        f.seek(off); txt = f.read().decode("utf-8", "replace")
    return sorted(set(int(x) for x in re.findall(r"launch_slot_: id  0 \| task (\d+)", txt)))

def chat(msgs, tools, max_tokens, seed):
    return M._post("/v1/chat/completions", {"messages": msgs, "tools": tools, "max_tokens": max_tokens,
                   "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed, "cache_prompt": True})

h = R.head()
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for rep in range(reps):
    for ask in ASKS:
        for variant in (("strip", "keep") if rep % 2 == 0 else ("keep", "strip")):
            for attempt in range(3):
                M._idle()
                msgs = [{"role": "system", "content": h["system"]}, {"role": "user", "content": ask}]
                off = os.path.getsize(LOG)
                r1 = chat(msgs, h["tools"], 4096, 1000 + rep)
                m = r1["choices"][0]["message"]
                if not m.get("tool_calls"):
                    print(json.dumps({"skip": "no tool call", "ask": ask[:30]}), flush=True); break
                a = {"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]}
                if variant == "keep":
                    a["reasoning_content"] = m.get("reasoning_content") or ""
                tool = {"role": "tool", "tool_call_id": m["tool_calls"][0].get("id", ""),
                        "content": "Result: (no results found)"}
                r2 = chat(msgs + [a, tool], h["tools"], 1, 1)
                n = len(tasks_since(off))
                t1, t2 = r1.get("timings") or {}, r2.get("timings") or {}
                row = {"variant": variant, "rep": rep, "ask": ask[:30], "attempt": attempt,
                       "gen_tokens": t1.get("predicted_n"), "think_chars": len(m.get("reasoning_content") or ""),
                       "turn2_prompt_n": t2.get("prompt_n"), "turn2_prompt_ms": round(t2.get("prompt_ms") or 0),
                       "tasks_in_window": n, "clean": n == 2, "t": time.strftime("%H:%M:%S")}
                print(json.dumps(row), flush=True)
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if row["clean"]:
                    break
