#!/usr/bin/env python3
r"""§4VD battery — the open items fixed on 2026-11-20:
(A) an edit's result names the file it was made from, and a retry edits that file again;
(B) the web chat proxy forwards the probe mark and its experiment headers, only on a probe;
(C) the smoother deletes a beat-opening paragraph only on evidence, and supersedes a summary only when
    the later one carries all its facts;
(D) a hyphenated letter compound is one distinctive search token.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4vd.py [-jN] [all|rerun M1 M2 …|check]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS = "interface/server.py"
RS = "src/ghost_agent/core/reply_smoothing.py"
SE = "src/ghost_agent/tools/search.py"
IG = "src/ghost_agent/tools/image_gen.py"
RG = "src/ghost_agent/tools/registry.py"
MUT = [
    ("CONTROL no-op", RS, r'''def _facts(text: str) -> set:''', r'''def _facts(text: str) -> set:  # no-op'''),
    ("CONTROL known-bad", RS, r'''    kept = [b for b, d in zip(blocks, drop) if not d]
    if not kept:
        return text''', r'''    kept = [b for b, d in zip(blocks, drop) if not d][1:]
    if not kept:
        return text'''),
    # B — the web probe mark
    ("B1 streamed-no-probe", IS, r'''                                   **_effort,
                                   **_probe}''', r'''                                   **_effort}'''),
    ("B2 plain-no-probe", IS, r'''                         **_effort,                                             # §4RB
                         **_probe},''', r'''                         **_effort},'''),
    ("B3 every-request-a-probe", IS, r'''    if str(headers.get("X-Ghost-Origin") or "").strip().lower() != "probe":
        return {}''', r'''    if False:
        return {}'''),
    ("B4 followers-dropped", IS, r'''    for name in _PROBE_FOLLOWERS:''', r'''    for name in ():'''),
    ("B5 case-sensitive-mark", IS, r'''.strip().lower() != "probe":''', r''' != "probe":'''),
    ("B6 one-follower-lost", IS, r'''                    "X-Ghost-Probe-Self-Image", "X-Ghost-Probe-Self-Image-Rule")''',
     r'''                    "X-Ghost-Probe-Self-Image")'''),
    ("B7 mark-not-sent", IS, r'''    out = {"X-Ghost-Origin": "probe"}''', r'''    out = {}'''),
    ("B8 probe-id-plain", IS, r'''                request_id = PROBE_REQUEST_PREFIX + request_id''', r'''                pass'''),
    ("B9 header-as-str", IS, r'''            out[name] = str(v).encode("latin-1", "replace")''', r'''            out[name] = str(v)'''),
    # C — the smoother
    ("C9 no-facts-check", RS, r'''and _facts(texts[a]) <= _facts(texts[b]):''', r'''and True:'''),
    ("C10 facts-reversed", RS, r'''and _facts(texts[a]) <= _facts(texts[b]):''', r'''and _facts(texts[b]) <= _facts(texts[a]):'''),
    ("C11 no-figures", RS, r'''_FACT_RE = re.compile(r"https?://\S+|`[^`\n]+`|\d+(?:[.,:]\d+)*")''',
     r'''_FACT_RE = re.compile(r"https?://\S+|`[^`\n]+`")'''),
    ("C12 no-code", RS, r'''_FACT_RE = re.compile(r"https?://\S+|`[^`\n]+`|\d+(?:[.,:]\d+)*")''',
     r'''_FACT_RE = re.compile(r"https?://\S+|\d+(?:[.,:]\d+)*")'''),
    ("C13 no-urls", RS, r'''_FACT_RE = re.compile(r"https?://\S+|`[^`\n]+`|\d+(?:[.,:]\d+)*")''',
     r'''_FACT_RE = re.compile(r"`[^`\n]+`|\d+(?:[.,:]\d+)*")'''),
    ("C18 facts-keep-punctuation", RS, r'''{m.group(0).rstrip(".,;:)") for m''', r'''{m.group(0) for m'''),
    ("C19 facts-case-folded", RS, r'''{m.group(0).rstrip(".,;:)") for m''', r'''{m.group(0).rstrip(".,;:)").lower() for m'''),
    # D — hyphenated compounds
    ("D1 no-join", SE, r'''    joined = _COMPOUND_RE.sub(lambda m: m.group(0).replace("-", ""), folded)''',
     r'''    joined = folded'''),
    ("D2 digit-compounds-joined", SE, r'''_COMPOUND_RE = re.compile(r"(?<![\w.-])[^\W\d_]+(?:-[^\W\d_]+)+(?![\w-])", re.UNICODE)''',
     r'''_COMPOUND_RE = re.compile(r"\w+(?:-\w+)+", re.UNICODE)'''),
    ("D3 no-lookbehind", SE, r'''_COMPOUND_RE = re.compile(r"(?<![\w.-])[^\W\d_]+(?:-[^\W\d_]+)+(?![\w-])", re.UNICODE)''',
     r'''_COMPOUND_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)+(?![\w-])", re.UNICODE)'''),
    ("D4 no-lookahead", SE, r'''_COMPOUND_RE = re.compile(r"(?<![\w.-])[^\W\d_]+(?:-[^\W\d_]+)+(?![\w-])", re.UNICODE)''',
     r'''_COMPOUND_RE = re.compile(r"(?<![\w.-])[^\W\d_]+(?:-[^\W\d_]+)+", re.UNICODE)'''),
    ("D5 compound-joined-out", SE, r"""            out.append(forms.get(tok, tok))""", r"""            out.append(tok)"""),
    ("D6 compound-parts-any", SE, r"""    return tok.replace("-", "") in flat or all(p in words for p in parts)""", r"""    return tok.replace("-", "") in flat or any(p in words for p in parts)"""),
    ("D7 compound-no-parts", SE, r"""    return tok.replace("-", "") in flat or all(p in words for p in parts)""", r"""    return tok.replace("-", "") in flat"""),
    ("D8 on-topic-plain-substr", SE, r"""    if any((compound_hit(tok, flat, all_words) if "-" in tok else tok in flat) for tok in dist):""", r"""    if any(tok in flat for tok in dist):"""),
    ("D9 wiki-no-compound", SE, r"""    if "-" in tok:                                         # §4VD-D: a hyphenated compound
        return compound_hit(tok, flat, words)""", r"""    if False:
        return compound_hit(tok, flat, words)"""),
    # A — the edit names its source
    ("A1 generic-note", IG, r'''               if len(ref_b64) == 1 and len(ref_paths) == 1 else''', r'''               if False else'''),
    ("A2 name-only", IG, r'''    for r in roots:
        try:
            rel = str(p.resolve().relative_to(r))''', r'''    for r in []:
        try:
            rel = str(p.resolve().relative_to(r))'''),
    ("A3 no-root-fallback", IG, r'''        roots = [root] + ([root.parent.parent.resolve()] if root.parent.name == "projects" else [])''',
     r'''        roots = [root]'''),
    ("A4 brackets-raw", IG, r'''    if _plain(rel) == rel:
        return rel''', r'''    if True:
        return rel'''),
    ("A4b brackets-plain", IG, r'''    if _plain(rel) == rel:
        return rel''', r'''    if True:
        return _plain(rel)'''),
    ("A5 schema-rule-gone", RG, r'''the retry edits the SAME file that edit was made from (the result names it), never the failed result; ''', r''''''),
]
FORCE: dict = {}
ONLY: dict = {}
PINS = [
    # A
    "tests/test_4vd_edit_retry_from_source.py", "tests/test_image_gen_model_swap.py", "tests/test_image_join_references.py",
    # B
    "tests/test_4vd_web_probe_header.py", "tests/test_4uy_progress.py", "tests/test_4rb_reasoning_effort.py",
    "tests/test_interface_chat_timeout.py", "tests/test_4rd_interface_stop.py", "tests/test_interface_proxy_auth.py",
    # C
    "tests/test_4vd_smoother_content.py", "tests/test_smoother_beat_before_delivery.py", "tests/test_reply_stale_announcement.py",
    "tests/test_smoother_status_then_opener.py", "tests/test_smoother_overbound_trailing_beat.py",
    "tests/test_reply_smoothing.py", "tests/test_forced_final_no_answer.py", "tests/test_4go_trailing_beat_narration.py",
    "tests/test_smoother_empty_handoff.py", "tests/test_4ji_request_echo_narration.py", "tests/test_reply_toolcall_scrub.py",
    "tests/test_reply_pass1_temporal.py", "tests/test_judged_text_is_delivered_text.py", "tests/test_stream_persisted_reply.py", "tests/test_4go_trailing_beat_narration.py",
    # D
    "tests/test_4vd_hyphen_compound.py", "tests/test_search_distinctive_relevance.py", "tests/test_search_relevance_floor.py",
    "tests/test_4un_web_search.py", "tests/test_4un_deep_research.py",
]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    if len({m[0] for m in MUT}) != len(MUT) or any(lab not in {m[0] for m in MUT} for lab in list(FORCE) + list(ONLY)):
        raise SystemExit("duplicate mutant label, or a FORCE entry names no mutant")
    missing = [p for p in PINS if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"test files not found: {missing}")
    if args[0] == "check":
        print(f"anchors ok: {len(MUT)} mutants in {len(srcs)} files; {len(PINS)} test files"); return
    import battery_cov as BC
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        if label in ONLY:
            return list(ONLY[label]), f"only {len(ONLY[label])} file(s)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | set(FORCE[label]))
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(PINS), f"fallback {len(PINS)} files"
    print(f"{len(PINS)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
