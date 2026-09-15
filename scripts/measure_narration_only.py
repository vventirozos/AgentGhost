"""Item 2 measurement: how many recorded replies would flip to "no answer"
under each candidate relaxation of `narration_only`, and which ones."""
import glob, json, os, re, sys
sys.path.insert(0, "/Users/vasilis/Data/AI/Agent/src")
os.environ.setdefault("GHOST_API_KEY", "x")
from ghost_agent.core import reply_smoothing as rs

ROOT = os.path.expanduser("~/Data/AI/Data/system/trajectories")

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "20*", "*.jsonl"))):
    for line in open(f, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        rep = (d.get("final_response") or "").strip()
        tc = d.get("tool_calls") or []
        n_tools = len(tc) if isinstance(tc, list) else 0
        if rep and n_tools >= 1:
            rows.append((d.get("id", "")[:8], n_tools, d.get("outcome"), rep))
print(f"corpus: {len(rows)} tool-bearing replies")

def flips(fn, label):
    out = []
    for rid, n, outcome, rep in rows:
        body = rs.strip_system_notes(rep)
        if rs.narration_only(body):
            continue                      # already caught today
        if fn(body):
            out.append((rid, n, outcome, rep))
    print(f"\n=== {label}: {len(out)} new 'no answer' verdicts "
          f"({100*len(out)/max(len(rows),1):.2f}% of the corpus)")
    return out

# R1 — the per-paragraph size bound only
def r1(text):
    old = rs._MAX_NARRATION_CHARS
    rs._MAX_NARRATION_CHARS = 500
    try:
        return rs.narration_only(text)
    finally:
        rs._MAX_NARRATION_CHARS = old

# R2 — drop bold/quotes from the content markers only
_R2_RE = re.compile(r"https?://|\d{2,}|`|^\s*[-*•]|^\s*\d+[.)]\s|\||!\[|\]\(", re.MULTILINE)
def r2(text):
    old = rs._NARRATION_CONTENT_RE
    rs._NARRATION_CONTENT_RE = _R2_RE
    try:
        return rs.narration_only(text)
    finally:
        rs._NARRATION_CONTENT_RE = old

# R3 — both
def r3(text):
    old_re, old_n = rs._NARRATION_CONTENT_RE, rs._MAX_NARRATION_CHARS
    rs._NARRATION_CONTENT_RE, rs._MAX_NARRATION_CHARS = _R2_RE, 500
    try:
        return rs.narration_only(text)
    finally:
        rs._NARRATION_CONTENT_RE, rs._MAX_NARRATION_CHARS = old_re, old_n

base = sum(1 for rid, n, o, rep in rows
           if rs.narration_only(rs.strip_system_notes(rep)))
print(f"caught TODAY: {base} ({100*base/max(len(rows),1):.2f}%)")

# R4 — the assessment-glue bound only (a long non-beat sentence counts as an answer)
def _glue(n):
    def f(text):
        old = rs._NARRATION_GLUE_MAX_CHARS
        rs._NARRATION_GLUE_MAX_CHARS = n
        try:
            return rs.narration_only(text)
        finally:
            rs._NARRATION_GLUE_MAX_CHARS = old
    return f

# R6 — glue + bold/quotes
def r6(text):
    old_re, old_g = rs._NARRATION_CONTENT_RE, rs._NARRATION_GLUE_MAX_CHARS
    rs._NARRATION_CONTENT_RE, rs._NARRATION_GLUE_MAX_CHARS = _R2_RE, 160
    try:
        return rs.narration_only(text)
    finally:
        rs._NARRATION_CONTENT_RE, rs._NARRATION_GLUE_MAX_CHARS = old_re, old_g

for fn, label in ((r1, "R1 size bound 300→500"),
                  (r2, "R2 bold/quotes are not content"),
                  (r3, "R3 both"),
                  (_glue(160), "R4 glue bound 140→160"),
                  (_glue(200), "R5 glue bound 140→200"),
                  (r6, "R6 glue 160 + bold/quotes")):
    got = flips(fn, label)
    for rid, n, outcome, rep in got[:12]:
        head = " / ".join(p.strip()[:70] for p in rep.split("\n\n")[:2])
        print(f"  [{rid}] tools={n} outcome={outcome} :: {head}")
