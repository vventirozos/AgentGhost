"""What pass 1 used to delete, measured on the real corpus.

OLD = the 2026-07-17 rule (temporal leads in the beat class, no restatement
test). NEW = §4FV. For every recorded reply, the difference is the set of
paragraphs the old rule deleted and the new one keeps.
"""
import json, glob, re, sys, importlib
sys.path.insert(0, "/Users/vasilis/Data/AI/Agent/src")
from ghost_agent.core import reply_smoothing as rs

NEW_BEAT = rs._BEAT_RE
OLD_BEAT = re.compile(
    r"^(?:let me\b|let's\b|good[,.! ]|okay\b|ok[,.! ]|alright\b|"
    r"great[,.! ]|perfect[,.! ]|time to\b|now[, ]|next[, ]|first[, ]|then[, ]|"
    r"i'll\b|i will\b|i need to\b)", re.IGNORECASE)

def smooth(text, old):
    rs._BEAT_RE = OLD_BEAT if old else NEW_BEAT
    try:
        return rs.smooth_reply(text)
    finally:
        rs._BEAT_RE = NEW_BEAT

rows = 0; changed = 0; kept = []
beat_drops = 0
for f in sorted(glob.glob("/Users/vasilis/Data/AI/Data/system/trajectories/2026-*/*.jsonl")):
    for line in open(f, errors="replace"):
        try: r = json.loads(line)
        except Exception: continue
        t = r.get("final_response") or ""
        if not isinstance(t, str) or "\n\n" not in t: continue
        rows += 1
        o, n = smooth(t, True), smooth(t, False)
        if o != t:
            beat_drops += 1
        if o != n:
            changed += 1
            for para in t.split("\n\n"):
                if para not in o and para in n:
                    kept.append((r.get("id", "?")[:8], para.strip()))
print(f"replies scanned: {rows}")
print(f"  old pass1+pass2 changed the reply: {beat_drops}")
print(f"  OLD and NEW disagree:              {changed}")
print(f"  paragraphs the narrowing SAVES:    {len(kept)}\n")
for rid, p in kept[:25]:
    print(f"  [{rid}] {p[:150]}")

# --- second pass: coverage retained ---------------------------------------
new_changed = 0; temporal_seen = 0; temporal_dropped = 0
for f in sorted(glob.glob("/Users/vasilis/Data/AI/Data/system/trajectories/2026-*/*.jsonl")):
    for line in open(f, errors="replace"):
        try: r = json.loads(line)
        except Exception: continue
        t = r.get("final_response") or ""
        if not isinstance(t, str) or "\n\n" not in t: continue
        if smooth(t, False) != t:
            new_changed += 1
        blocks = rs._split_blocks(t)
        for i, b in enumerate(blocks[:-1]):
            st = b.strip()
            if rs._TEMPORAL_LEAD_RE.match(st) and not rs._BEAT_RE.match(st):
                temporal_seen += 1
                if rs._is_narration(b, blocks[i+1:]):
                    temporal_dropped += 1
print(f"  new pass1+pass2 changed the reply:  {new_changed}")
print(f"  bare temporal-lead paragraphs:      {temporal_seen}")
print(f"    of those, still dropped (restated): {temporal_dropped}")

