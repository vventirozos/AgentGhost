#!/usr/bin/env python3
"""§4GN mutation battery — the OCR route, the eaten write, and breadth.

Whole-file mutants against a COPY of the tree, each run against the WHOLE
test set so the harness cannot name its own killer. A no-op control must
SURVIVE and a known-bad control must be KILLED.

    python3 scripts/mutate_4gn.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
DW = os.path.join(ROOT, "src/ghost_agent/tools/darkweb_search.py")
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
RG = os.path.join(ROOT, "src/ghost_agent/tools/registry.py")
PR = os.path.join(ROOT, "src/ghost_agent/core/prompts.py")
FILES = [DW, AG, RG, PR]
TESTS = [
    "tests/test_4gn_route_dropped_write_and_breadth.py",
    "tests/test_4gl_unverified_write_and_ocr.py",
    "tests/test_darkweb_search.py",
    "tests/test_grounded_file_verify.py",
]

MUTANTS = [
    ("CONTROL no-op comment", DW,
     "async def tool_darkweb_search(",
     "# a no-op edit\nasync def tool_darkweb_search("),

    # ── the route ────────────────────────────────────────────────────────
    ("M1 the vision tool forgets the no-URL route", RG,
     ("NO DIRECT IMAGE URL?", "just as well as the image file."), ""),
    ("M2 the browser op stops naming the OCR follow-up", RG,
     "PNG with vision_analysis action=extract_text_picture.",
     "PNG."),
    ("M3 the prompt drops the two-step route", PR,
     ("WHEN YOU CANNOT GET A DIRECT IMAGE URL", "this two-step route exists."), ""),

    # ── the eaten write ──────────────────────────────────────────────────
    ("M4 the note goes back to disclaiming only the future", AG,
     'f"{\', \'.join(muts)} action(s) could run — nothing described above as "\n        f"written, saved or created was actually written{where}. Ask me to "',
     'f"{\', \'.join(muts)} action(s) could run — any change described "\n        f"above as about to happen has NOT been applied yet{where and \'\'}. Ask me to "'),
    ("M5 the note stops naming the file", AG,
     '    named = [str(x) for x in (paths or []) if str(x).strip()][:3]',
     '    named = []'),
    ("M6 the drop site stops reading the call's arguments", AG,
     "                        _drop_note = _dropped_mutation_note(\n                            dropped, _dropped_write_paths(tool_calls))",
     "                        _drop_note = _dropped_mutation_note(dropped)"),
    ("M7 the admission is no longer absence-grade", AG,
     "            if _dropped_write_admitted(_claim_src):",
     "            if False:"),
    ("M8 any reply with the note arms the absence leg", AG,
     '    return "file_system" in reply[i:i + 400]',
     "    return True"),

    # ── breadth ──────────────────────────────────────────────────────────
    ("M9 the extra phrasings are silently dropped", DW,
     "    cand = [query] + [q for q in list(extra_queries or []) if isinstance(q, str)]",
     "    cand = [query]"),
    ("M10 a duplicate phrasing eats a slot again", DW,
     "        if not q or k in seen:\n            continue",
     "        if not q:\n            continue"),
    ("M11 cross-query corroboration stops ranking", DW,
     '    ranked = sorted((merged[h] for h in order),\n                    key=lambda r: (-len(r["indexes"]), -len(r["queries"])))',
     "    ranked = list(merged[h] for h in order)"),
    ("M12 every phrasing rides ONE circuit", DW,
     "        _q, _p = (_apply_anonymous_scrub(q, base_proxy) if anonymous\n                  else (q, base_proxy))",
     "        _q, _p = (q, base_proxy)"),
    ("M13 one failing phrasing loses the whole call", DW,
     "        return_exceptions=True,\n    )\n    per_query, _skipped_all",
     "        return_exceptions=False,\n    )\n    per_query, _skipped_all"),

    ("KNOWN-BAD control: the dark-web tool returns nothing", DW,
     "    queries = _query_set(query, extra_queries)",
     "    return \"\"\n    queries = _query_set(query, extra_queries)"),
]


def main():
    saved = {f: f + ".pristine" for f in FILES}
    for f, p in saved.items():
        shutil.copy2(f, p)
    env = dict(os.environ, GHOST_API_KEY="x",
               PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    try:
        for n, (label, target, old, new) in enumerate(MUTANTS):
            for f, p in saved.items():          # restore every file first
                shutil.copy2(p, f)
            src = open(saved[target]).read()
            if isinstance(old, tuple):
                # SPAN mutant: delete start..end inclusive. A prose route
                # cannot be removed by swapping its lead-in — the first pass
                # of this battery did exactly that and two mutants survived
                # because the sentence they were meant to delete was still
                # there under a different opening.
                a, b = old
                if src.count(a) != 1 or src.count(b) != 1 or src.index(b) < src.index(a):
                    print(f"[{n:2d}] ANCHOR-MISS (span) {label}", flush=True)
                    continue
                mutated = src[:src.index(a)] + new + src[src.index(b) + len(b):]
            else:
                if src.count(old) != 1:
                    print(f"[{n:2d}] ANCHOR-MISS ({src.count(old)}x) {label}", flush=True)
                    continue
                mutated = src.replace(old, new, 1)
            open(target, "w").write(mutated)
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                 "-p", "no:randomly", "--timeout=300"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
            tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:80]}",
                  flush=True)
    finally:
        for f, p in saved.items():
            shutil.copy2(p, f)
            os.unlink(p)
    print("=== 4GN BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
