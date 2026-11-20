#!/usr/bin/env python3
r"""§4VB battery — three fixes from the 11-20 live functional run:
(1) an onion engine's page without its result-listing marker (Ahmia's "Help Us to Help You" notice) books no
result and ends as "answered, no match"; (2) every top-level task of a project is in its task tree
(`TaskTree.roots()`); (3) "<praise word> <work noun>" is thanks ("nice job", "τέλεια δουλειά").
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4vb.py [-jN] [all|rerun M1 M2 …|check]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DW = "src/ghost_agent/tools/darkweb_search.py"; PL = "src/ghost_agent/core/planning.py"
MO = "src/ghost_agent/selfhood/mood.py"
MUT = [
    ("CONTROL no-op", DW, r'''def _engine_onion_hosts(engines: List[Dict[str, str]]) -> set:''',
     r'''def _engine_onion_hosts(engines: List[Dict[str, str]]) -> set:  # no-op'''),
    ("CONTROL known-bad", DW, r'''                        _diag["outcome"] = OUTCOME_RESULTS
                        return parsed''', r'''                        _diag["outcome"] = OUTCOME_RESULTS
                        return []'''),
    # D — the darkweb notice page
    ("D1 clearnet-ahmia-no-marker", DW, r'''        "results_marker": "#totalResults",
    },
    {
        "name": "ahmia-onion",''', r'''    },
    {
        "name": "ahmia-onion",'''),
    ("D2 onion-ahmia-no-marker", DW, r'''        "result_selector": "li.result",
        "results_marker": "#totalResults",
    },
    {
        # Torch''', r'''        "result_selector": "li.result",
    },
    {
        # Torch'''),
    ("D3 override-drops-marker", DW, r'''            if e.get("results_marker"):
                entry["results_marker"] = str(e["results_marker"])
''', ""),
    ("D4 marker-check-inverted", DW, r'''_listing = soup.select_one(results_marker) is not None''',
     r'''_listing = soup.select_one(results_marker) is None'''),
    ("D5 bad-selector-fails-closed", DW, r'''            except Exception:  # noqa: BLE001 — a selector bs4 rejects: no marker check (read as before)
                _listing = True''', r'''            except Exception:  # noqa: BLE001 — a selector bs4 rejects: no marker check (read as before)
                _listing = False'''),
    ("D6 diag-not-set", DW, r'''                diag["no_listing"] = True''', r'''                pass'''),
    ("D7 notice-parsed-on", DW, r'''                rest.append("")
            return []
''', r'''                rest.append("")
'''),
    ("D8 caller-drops-marker", DW, r'''                        results_marker=str(engine.get("results_marker") or ""), diag=_pdiag)''',
     r'''                        diag=_pdiag)'''),
    ("D9 notice-charged-as-parser", DW, r'''                        _diag["outcome"] = OUTCOME_NO_MATCH
                        pretty_log("Darkweb Engine",
                                   f"{name}: answered, no match — served''', r'''                        _diag["outcome"] = OUTCOME_PARSER
                        pretty_log("Darkweb Engine",
                                   f"{name}: answered, no match — served'''),
    ("D10 notice-branch-gone", DW, r'''                    if _pdiag.get("no_listing"):''', r'''                    if False:'''),
    ("D11 notice-not-nohits", DW, r'''                        _diag["nohits"] = True
                        _diag["outcome"] = OUTCOME_NO_MATCH
                        pretty_log("Darkweb Engine",
                                   f"{name}: answered, no match — served''', r'''                        _diag["outcome"] = OUTCOME_NO_MATCH
                        pretty_log("Darkweb Engine",
                                   f"{name}: answered, no match — served'''),
    ("D12 notice-rest-not-filled", DW, r'''            if rest is not None:
                rest.append("")
            return []
''', r'''            return []
'''),
    # r1 — the marker judges only pages that have the engine's result rows
    ("D13 marker-judges-pages-without-rows", DW, r'''        _rows_without_marker = bool(containers) and not _listing''',
     r'''        _rows_without_marker = not _listing'''),
    ("D14 drifted-listing-dropped", DW, r'''        if len(results) > _NO_HITS_MAX_LINKS:
            if diag is not None:
                diag["marker_missing"] = True''', r'''        if False:
            if diag is not None:
                diag["marker_missing"] = True'''),
    ("D15 drift-not-flagged", DW, r'''                diag["marker_missing"] = True''', r'''                pass'''),
    ("D16 drift-not-warned", DW, r'''                    if _pdiag.get("marker_missing"):''', r'''                    if False:'''),
    ("D18 drift-threshold-inclusive", DW, r'''        if len(results) > _NO_HITS_MAX_LINKS:
            if diag is not None:''', r'''        if len(results) >= _NO_HITS_MAX_LINKS:
            if diag is not None:'''),
    ("D17 drift-threshold-off-by-one", DW, r'''        if len(results) > _NO_HITS_MAX_LINKS:
            if diag is not None:''', r'''        if len(results) > _NO_HITS_MAX_LINKS + 1:
            if diag is not None:'''),
    # P — every top-level task in the tree
    ("P1 orphan-hidden", PL, r'''        return [nid for nid, n in self.nodes.items() if not n.parent_id or n.parent_id not in self.nodes]''',
     r'''        return [nid for nid, n in self.nodes.items() if not n.parent_id]'''),
    ("P2 first-root-only", PL, r'''        return [nid for nid, n in self.nodes.items() if not n.parent_id or n.parent_id not in self.nodes]''',
     r'''        return [nid for nid, n in self.nodes.items() if not n.parent_id or n.parent_id not in self.nodes][:1]'''),
    ("P3 render-one-root", PL, r'''        for rid in roots:
            self._render_node(rid, 0, lines, visited)''', r'''        self._render_node(roots[0], 0, lines, visited)'''),
    ("P4 render-roots-reversed", PL, r'''        for rid in roots:
            self._render_node(rid, 0, lines, visited)''', r'''        for rid in reversed(roots):
            self._render_node(rid, 0, lines, visited)'''),
    ("P5 active-first-root-only", PL, r'''            for rid in self.roots():
                found = find_status(rid, wanted, visited)''', r'''            for rid in self.roots()[:1]:
                found = find_status(rid, wanted, visited)'''),
    ("P6 active-pending-before-running", PL,
     r'''        for wanted in ([TaskStatus.IN_PROGRESS], [TaskStatus.READY, TaskStatus.PENDING]):''',
     r'''        for wanted in ([TaskStatus.READY, TaskStatus.PENDING], [TaskStatus.IN_PROGRESS]):'''),
    ("P7 json-one-root-for-many", PL, r'''        if len(roots) == 1:
            return serialize(roots[0])''', r'''        if True:
            return serialize(roots[0])'''),
    ("P8 json-roots-truncated", PL, r'''        return {"roots": [serialize(rid, visited) for rid in roots]}''',
     r'''        return {"roots": [serialize(rid, visited) for rid in roots[:1]]}'''),
    ("P9 retro-guard-root-id", PL, r'''        if not self.nodes:                    # §4VB''', r'''        if not self.root_id:                    # §4VB'''),
    ("P10 active-guard-root-id", PL, r'''        if not self.nodes: return None''', r'''        if not self.root_id: return None'''),
    ("P11 render-empty-guard", PL, r'''        roots = self.roots()
        if not roots: return "No Plan."''', r'''        roots = self.roots()
        if not self.root_id: return "No Plan."'''),
    ("P12 roots-shape-not-read-back", PL, r'''        if isinstance(json_data, dict) and "id" not in json_data and isinstance(json_data.get("roots"), list):''',
     r'''        if False:'''),
    # M — praise
    ("M1 nouns-are-filler", MO, r'''        elif words[i] in PRAISE_FILLER:''', r'''        elif words[i] in PRAISE_FILLER or words[i] in PRAISE_NOUNS:'''),
    ("M2 noun-not-consumed", MO, r'''                i += 1                                  # §4VB: "nice job", "τέλεια δουλειά"''',
     r'''                pass'''),
    ("M3 noun-bound-off-by-one", MO, r'''            if (i < len(words) and words[i] in PRAISE_NOUNS''',
     r'''            if (i <= len(words) and words[i] in PRAISE_NOUNS'''),
    ("M4 no-job", MO, r'''PRAISE_NOUNS = frozenset({"job", "work", "stuff", "δουλεια"})''',
     r'''PRAISE_NOUNS = frozenset({"work", "stuff", "δουλεια"})'''),
    ("M5 no-work", MO, r'''PRAISE_NOUNS = frozenset({"job", "work", "stuff", "δουλεια"})''',
     r'''PRAISE_NOUNS = frozenset({"job", "stuff", "δουλεια"})'''),
    ("M6 no-stuff", MO, r'''PRAISE_NOUNS = frozenset({"job", "work", "stuff", "δουλεια"})''',
     r'''PRAISE_NOUNS = frozenset({"job", "work", "δουλεια"})'''),
    ("M7 no-greek-noun", MO, r'''PRAISE_NOUNS = frozenset({"job", "work", "stuff", "δουλεια"})''',
     r'''PRAISE_NOUNS = frozenset({"job", "work", "stuff"})'''),
    ("M8 no-feminine", MO, r'''                          "αψογη", "εξαιρετικη", "υπεροχη"})''', r'''                          })'''),
    ("M10 noun-across-punctuation", MO, r'''                    and _adjacent(i - 1)):''', r'''                    and True):'''),
    ("M13 pair-across-punctuation", MO, r'''        if tuple(words[i:i + 2]) in PRAISE_PAIRS and _adjacent(i):''',
     r'''        if tuple(words[i:i + 2]) in PRAISE_PAIRS:'''),
    ("M15 hyphen-is-a-gap", MO, r'''_PRAISE_JOINERS = " \t\r\n\v\f-_‐‑​"''', r'''_PRAISE_JOINERS = None'''),
    ("M16 punctuation-joins", MO, r'''_PRAISE_JOINERS = " \t\r\n\v\f-_‐‑​"''',
     r'''_PRAISE_JOINERS = " \t\r\n\v\f-_‐‑​,."'''),
    ("M14 adjacency-ignores-gap", MO, r'''        return j + 1 < len(toks) and not t[toks[j].end():toks[j + 1].start()].strip(_PRAISE_JOINERS)''',
     r'''        return j + 1 < len(toks)'''),
    ("M11 noun-after-thanks", MO, r'''words[i] in PRAISE_NOUNS and words[i - 1] not in PRAISE_THANKS''',
     r'''words[i] in PRAISE_NOUNS'''),
    ("M12 thanks-set-missing-thanks", MO,
     r'''PRAISE_THANKS = frozenset({"thanks", "thx", "ty", "cheers", "bravo", "ευχαριστω", "σ'ευχαριστω", "μπραβο"})''',
     r'''PRAISE_THANKS = frozenset({"thx", "ty", "cheers", "bravo", "ευχαριστω", "σ'ευχαριστω", "μπραβο"})'''),
    ("M9 no-file-address", MO, r'''                           "σημερα", "παλι", "φιλε"})''', r'''                           "σημερα", "παλι"})'''),
]
FORCE: dict = {}
ONLY: dict = {}
PINS = [
    # darkweb
    "tests/test_4vb_darkweb_notice.py", "tests/test_darkweb_search.py", "tests/test_4un_darkweb.py",
    "tests/test_4un_darkres.py", "tests/test_4gm_onion_fetch_async.py", "tests/test_4nk_darkweb_research.py",
    "tests/test_4pd_search_review.py", "tests/test_darkweb_no_hits_phrasings.py", "tests/test_darkweb_form_token.py",
    "tests/test_onion_dead_link_and_breaker.py", "tests/test_4uz_research_distill.py", "tests/test_4nv_log_lines.py",
    "tests/test_4uy_internet.py", "tests/test_research_failure_honesty.py",
    # planning / projects
    "tests/test_4vb_task_tree.py", "tests/test_project_plan.py", "tests/test_hierarchical_planning.py",
    "tests/test_hierarchical_planning_extended.py", "tests/test_planning_hierarchical.py", "tests/test_planning_cortex.py",
    "tests/test_planning_merge.py", "tests/test_planning_failure.py", "tests/test_plan_revision.py",
    "tests/test_tasktree_redescribed_id.py", "tests/test_project_mgmt_fixes.py", "tests/test_4su_phase2.py",
    "tests/test_4qa_dependency_fuzz.py", "tests/test_projects_round23_fixes.py",
    # mood
    "tests/test_4vb_praise.py", "tests/test_4tc_mood_engine.py", "tests/test_4tx_mood_review.py",
    "tests/test_4ub_mood_fixes.py", "tests/test_4tc_mood_hooks.py",
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
