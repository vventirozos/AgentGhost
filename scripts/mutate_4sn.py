#!/usr/bin/env python3
"""§4SN battery — memory consolidation + retrieval review (2026-10-31). Coverage-selected,
parallel (scripts/battery_cov.py); the real src/ is never mutated. No-op control must SURVIVE,
known-bad control must DIE.    python3 scripts/mutate_4sn.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = "src/ghost_agent/core/agent.py"; T = "src/ghost_agent/memory/attribution.py"
GR = "src/ghost_agent/memory/graph.py"; VE = "src/ghost_agent/memory/vector.py"
TM = "src/ghost_agent/tools/memory.py"; PR = "src/ghost_agent/memory/profile.py"
S = "scripts/restamp_owner_mirrors_4sn.py"
JR = "src/ghost_agent/memory/journal.py"
MUT = [
    ("CONTROL no-op", T, "def field_kind(category, key) -> str:", "def field_kind(category, key) -> str:  # no-op"),
    ("CONTROL known-bad", G, "        interaction_context = _latest_ep(interaction_context)\n", "        interaction_context = ''\n"),
    # C1 corrections
    ("C1a kind-gate-off", G, "                    if _want_kind is not None and _fkind(cat, k) != _want_kind:\n                        continue",
     "                    if False:\n                        continue"),
    ("T3 pkind-isa", T, '"IS_A": "HAS_PROFESSION", ', '"IS_A": "", '),
    ("T3b pkind-cues", T, "    if p in PREDICATE_CUES:\n        return p\n    return _PREDICATE_KINDS", "    if False:\n        return p\n    return _PREDICATE_KINDS"),
    ("T4 latest-first-user", T, '        if m and m.group(1).lower() == "user":\n            start = i',
     '        if m and m.group(1).lower() == "user" and start is None:\n            start = i'),
    ("T5 latest-one-line", T, "    end = next((j for j in range(start + 1, len(lines)) if roles[j]), len(lines))", "    end = start + 1"),
    ("T6 latest-foreign", T, '    return "" if not owner_lines(out) else out', "    return out"),
    # C2 drain
    ("C2a nodedown-not-raised", G, "                    if isinstance(_ue, _OffMainDown):", "                    if False:"),
    ("C2b nodedown-burns-retry", G, "            except _NodeDown as e:", "            except ZeroDivisionError as e:"),
    ("C2d failed-counted", G, '                if _outcome == "failed":', "                if False:"),
    ("C2e skipped-counted", G, '                if _outcome != "extracted":\n                    skipped += 1',
     "                if False:\n                    skipped += 1"),
    ("C2f failure-reads-extracted", G, '                return "failed"\n            return "extracted"', '                return "extracted"\n            return "extracted"'),
    # C3 negation
    ("C3a negation-returns", G, '                    fact = ""\n                if profile_up is not None and is_removal_or_negation_text(',
     '                    return "extracted"\n                if profile_up is not None and is_removal_or_negation_text('),
    ("C3b neg-profile-written", G, '                if profile_up is not None and is_removal_or_negation_text(str(profile_up.get("value") or "")):',
     "                if False:"),
    # C4 one home
    ("C4a held-off", G, "                    if _held:", "                    if False:"),
    ("C4b category-raw", G, '_cat_p = _extractor_category(profile_up.get("category"))', '_cat_p = profile_up.get("category")'),
    ("C4c rule-prefs", G, r'    (re.compile(r"\b(?:prefer\w*|tastes?|likes?|settings?)\b"), "preferences"),', r'    (re.compile(r"\b(?:preferences)\b"), "preferences"),'),
    ("C4d rule-unbounded", G, r'(?:assets?|vehicles?|cars?|propert(?:y|ies)|possessions?|belongings?)\b"), "assets"),', r'(?:assets?|vehicles?|car|propert(?:y|ies)|possessions?|belongings?)"), "assets"),'),
    ("C4e holding-list", G, "                for it in (cur if isinstance(cur, list) else [cur]):\n                    if _pf(str(it)).strip() == want:",
     "                for it in [cur]:\n                    if _pf(str(it)).strip() == want:"),
    # C5 / C7
    ("C5a latest-off", G, "        interaction_context = _latest_ep(interaction_context)\n", "        interaction_context = interaction_context\n"),
    ("C7a anchor-now", G, "                    fact = _anchor_temporal(fact, said_at=_said_dt)", "                    fact = _anchor_temporal(fact)"),
    ("C7b ts-now", G, '{"timestamp": _said_iso or get_utc_timestamp(), "type": "auto"}', '{"timestamp": get_utc_timestamp(), "type": "auto"}'),
    # R1 kind words
    ("R1a greek-off", GR, "_kinds.get(self._greek_kind_key(w, _kinds), set()) if extended else set())", "set() if extended else set())"),
    ("R1b greek-base", GR, "_kinds.get(self._greek_kind_key(w, _kinds), set()) if extended else set())", "_kinds.get(self._greek_kind_key(w, _kinds), set()) if True else set())"),
    ("R1e heart-base", GR, '        "heart", "καρδια", "pet", "pets",', '        "καρδια", "pet", "pets",'),
    ("R1f heart-key", GR, '        "heart": {"condition", "conditions",', '        "heartx": {"condition", "conditions",'),
    # R2 address mirror
    ("R2a value-off", TM, "            values = [v for v in values if profile_memory.is_on_demand(cat, k, value=v) is not True]",
     "            values = list(values)"),
    # R3 dates
    ("R3a select-timestamp", GR, "\"COALESCE(strftime('%Y-%m-%d', valid_from, 'unixepoch'), timestamp) FROM triplets \"", '"timestamp FROM triplets "'),
    ("R3b mirror-asof", GR, "raw=True, as_of=float(_when[v]))", "raw=True)"),
    ("R3c vector-asof", VE, "            _w = (as_of_by_value or {}).get(v.lower())", "            _w = None"),
    ("R3d mirrors-no-dates", TM, '                    _kw["as_of_by_value"] = _when', "                    pass"),
    ("R3e restamp-later-too", S, "                        if vf is None or epoch < float(vf) - 86400:",
     "                        if vf is None or abs(float(vf) - epoch) > 86400:"),
    ("R3f restamp-no-lock", S, '        assert_no_other_writer(md, "restamp_owner_mirrors_4sn.py")   # the agent must be stopped', "        pass"),
    # R4 / R5 / R6 / R8 / R9
    ("R4a titlecase", GR, '            return "User" if str(x).strip().lower() == "user" else str(x)',
     '            return "User" if str(x).strip().lower() == "user" else str(x).title()'),
    ("R5a dedup-raw", TM, '                            _key = _AS_OF_SUFFIX.sub("", str(e)).strip().lower()', "                            _key = str(e)"),
    ("R6a user-topic", TM, "his her\nuser users\n", "his her\n"),
    ("R8a age-off", PR, '                and "born" not in text.lower():', "                and False:"),
    ("R8b age-twice", PR, '                and "born" not in text.lower():', "                and True:"),
    ("R9a episode-filter-off", TM, '        if m_type == "episode" and not _episode_on_topic(str(query), str(text)):', "        if False:"),
    ("R9b episode-4-letters", TM, "if len(w) >= 3 and w not in _EPISODE_FILTER_STOP", "if len(w) >= 4 and w not in _EPISODE_FILTER_STOP"),
    # round-1 fixes in agent.py
    ("C5b owner-text-off", G, "            if _head and _head not in", "            if False and _head not in"),
    ("C5d fast-first-line", G, "        user_lines = _ol_fast(interaction_context)\n",
     "        user_lines = [l[5:] for l in interaction_context.splitlines() if l.startswith('USER:')]\n"),
    ("C5e drain-no-owner-text", G, '                        owner_text=str((item.get("data") or {}).get("user") or ""))', '                        owner_text="")'),
    # round 2 (re-anchored + new)
    ("C1b no-kind-deletes", G, "                if not _want_kind:\n                    # a kind no profile field holds: graph only.",
     "                if False:\n                    # a kind no profile field holds: graph only."),
    ("C1c removal-any-said", G, '                    if is_removal_triplet(_t) and _said_any == "stated":', "                    if is_removal_triplet(_t) and _said_any is not None:"),
    ("T7 fieldkind-none", T, '    return next(iter(kinds)) if len(kinds) == 1 else ""', '    return ""'),
    ("T7b spouse-name-off", T, '        if subj == {"spouse"} and kt <= ', "        if False and kt <= "),
    ("T7c subject-owner-kind", T, '        return ""\n    kinds = {pred', "        pass\n    kinds = {pred"),
    ("T7d ambiguous-first", T, 'if len(kinds) == 1 else ""', 'if kinds else ""'),
    ("C2c nodedown-first-stops", G, "                if _nd_row >= 2:", "                if _nd_row >= 1:"),
    ("C2g nodedown-never-stops", G, "                if _nd_row >= 2:", "                if _nd_row >= 99:"),
    ("C2k pause-drops-aside", G, "push_front, requeue + items[i:] + _nd_failed)", "push_front, requeue + items[i:])"),
    ("C2l keep-lost", G, "            await asyncio.to_thread(self.context.journal.push_front, requeue + _nd_rest + _nd_keep)",
     "            await asyncio.to_thread(self.context.journal.push_front, requeue + _nd_rest)"),
    ("R1c greek-stem-loose", GR, '            if (n >= max(3, len(k) - 1) and k[n:] in ("", "σ", "ς", "α", "η")', "            if (n >= 2 and True"),
    ("R1g greek-ending-any", GR, "and w[n:] in GraphMemory._GREEK_ENDINGS and n > len(best)):", "and len(w) - n <= 3 and n > len(best)):"),
    ("R1h greek-verb-cut", GR, 'k[n:] in ("", "σ", "ς", "α", "η")', "len(k[n:]) <= 1"),
    ("R9c episode-empty-hides", TM, "    if not q:\n        return True\n    t = {_episode_word", "    if not q:\n        return False\n    t = {_episode_word"),
    ("R9d plural-off", TM, '    return w[:-1] if len(w) > 3 and w.endswith("s") else w', "    return w"),
    ("R9e sigma-stop", TM, '_EPISODE_FILTER_STOP = frozenset(w.replace("ς", "σ") for w in _EPISODE_FILTER_STOP)', "_EPISODE_FILTER_STOP = frozenset(_EPISODE_FILTER_STOP)"),
    ("R9f time-words", TM, '"talk", "talked", "discuss", "discussed",', ""),
    ("C4f held-any-subject", G, "                if key is not None and not (_fs(cat, k) == _subj\n", "                if key is not None and not (True\n"),
    ("C4g held-no-stems", G, "                                            and ((_kind and _fk(cat, k) == _kind) or (_stems & _key_stems(k)))):",
     "                                            and ((_kind and _fk(cat, k) == _kind))):"),
    ("C4h root-suffix", G, '                r"(?: (?:info|information|details|data|profile|facts))?$"), "root"),', '                r"$"), "root"),'),
    ("C5c owner-text-unclosed", G, "            _ot = re.sub(r'```.*?(?:```|$)', '', str(owner_text), flags=re.DOTALL)",
     "            _ot = re.sub(r'```.*?```', '', str(owner_text), flags=re.DOTALL)"),
    ("C5f arc-list-content", G, "        clean_content = re.sub(r'```.*?```', '', _msg_text(m), flags=re.DOTALL)",
     "        clean_content = re.sub(r'```.*?```', '', str(m.get(\"content\", \"\")), flags=re.DOTALL)"),
    # round 3: the "stopped" rule removed; reasons split clauses; outage never counts
    ("T1 because-joins", T, "|\\bbecause\\b|\\bsince\\b|", "|"),
    ("T1b greek-reason-joins", T, '                 r"|(?<!\\w)(?:γιατ[ιί]|επειδ[ηή]|αφο[υύ])(?!\\w)")', '                 r"")'),
    ("C2h outage-counts", G, "(1 if processed and _nd_row < 2 else 0)", "(1 if processed else 0)"),
    ("C2i never-counts", G, "(1 if processed and _nd_row < 2 else 0)", "0"),
    ("C2j skip-resets-row", G, '                if _outcome == "extracted":\n                    _nd_row = 0', "                if True:\n                    _nd_row = 0"),
    ("JRa dedup-node-down", JR, '            _volatile = ("retries", "node_down")', '            _volatile = ("retries",)'),
]
PINS = ["tests/test_consolidation_review_1031.py", "tests/test_retrieval_review_1031.py"]


def _tests():
    pat = (r"run_smart_memory_task\|process_journal_queue\|owner_facts_matching\|sync_owner_mirrors\|tool_recall"
           r"\|latest_owner_episode\|owner_stopped\|_format_path\|get_context_string\|restamp_owner\|sync_owner_field"
           r"\|_apply_owner_corrections\|get_neighborhood")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in batch if originals[f].count(old) != 1]
    if bad:
        print("ANCHOR NOT UNIQUE/FOUND:", bad); sys.exit(2)
    cov = BC.build_map(srcs, full, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        # every mutant also runs the pin files: the coverage map missed a
        # script's own pin (R3e survived beside a pin that kills it)
        return (sorted(set(ids) | set(PINS)), f"cov {len(ids)} tests + pins") if ids else (PINS, "fallback pins")
    print(f"{len(full)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)


if __name__ == "__main__":
    main()
