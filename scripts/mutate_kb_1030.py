"""KB / graph / recall review battery (2026-10-30): recall bands, scope and
owner rows, the Greek owner gate, category-qualified mirrors, forget resync,
ingest (splitter, resolver, binary sniff, scanned PDFs, section crumbs, the
in-progress marker). Reuses the §4QA harness (in place, backup + restore +
byte compare, two-tier, controls).
    python3 scripts/mutate_kb_1030.py [all|check|rerun K1 …]
"""
import glob
import importlib.util
import os
import subprocess

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

TM = "src/ghost_agent/tools/memory.py"; BU = "src/ghost_agent/core/bus.py"
GR = "src/ghost_agent/memory/graph.py"; VE = "src/ghost_agent/memory/vector.py"
PR = "src/ghost_agent/tools/projects.py"; PI = "src/ghost_agent/memory/pdf_ingest.py"
HE = "src/ghost_agent/utils/helpers.py"; RO = "src/ghost_agent/memory/readonly.py"
PF = "src/ghost_agent/memory/profile.py"; DR = "src/ghost_agent/core/dream.py"
m.PINS = ["tests/test_kb_review_2026_10.py"]
_FAST = {RO: m.PINS + ["tests/test_readonly_memory.py"],
         VE: m.PINS + ["tests/test_memory_store_reconcile_4gj.py", "tests/test_4lt_kb.py"],
         DR: m.PINS + ["tests/test_memory_writes_review.py"]}


def _tests():
    pat = (r"tool_recall\|sync_owner\|_asks_about_owner\|ingest_document\|split_page_by_sections\|"
           r"semantic_split_text\|recursive_split_text\|delete_document_by_name\|resync_all\|mirror_name\|"
           r"_not_found_with_recall\|_CATEGORY_WORDS\|owner_facts_matching\|forget_execute\|ingest_pdf")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m._fast_tests = lambda f, pins: sorted(set(_FAST.get(f, pins)))
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", TM,
     "def recall_relevance(distance: float) -> str:", "# ctl\ndef recall_relevance(distance: float) -> str:"),
    ("CONTROL known-bad (must DIE): L2-era bands", TM,
     'RECALL_HIGH_D = float(os.environ.get("GHOST_RECALL_HIGH_D", "0.28") or 0.28)',
     'RECALL_HIGH_D = float(os.environ.get("GHOST_RECALL_HIGH_D", "0.8") or 0.8)'),
    # ── recall
    ("R1 the keep band never drops a row", TM, "        if score < RECALL_KEEP_D:", "        if True:"),
    ("R2 medium band as high", TM, '    return "MEDIUM" if distance < RECALL_MEDIUM_D else "LOW"',
     '    return "HIGH" if distance < RECALL_MEDIUM_D else "LOW"'),
    ("R3 recall searches documents", TM, '                                          where={"type": {"$nin": _types_out}}, record_retrievals=False)',
     '                                          record_retrievals=False)'),
    ("R4 recall bumps every hit", TM, "                                          where={\"type\": {\"$nin\": _types_out}}, record_retrievals=False)",
     "                                          where={\"type\": {\"$nin\": _types_out}}, record_retrievals=True)"),
    ("R5 the public refusal is gone", TM, "    if _public():\n        return ToolOutcome.rejected(_PUBLIC_RECALL_BLOCK",
     "    if False:\n        return ToolOutcome.rejected(_PUBLIC_RECALL_BLOCK"),
    ("R6 the member check fails open", TM,
     "    except Exception:  # noqa: BLE001 — KB review 10-30: unknown requester = refused, not the owner\n        return ToolOutcome.rejected(_MEMBER_BLOCK, reason_code=\"owner_data_blocked\")",
     "    except Exception:  # noqa: BLE001 — KB review 10-30: unknown requester = refused, not the owner\n        return None"),
    ("R7 stored text not neutralised", TM, '    return "\\n".join(("> " + ln) if _RECALL_FORGE_RE.match(ln) else ln',
     '    return "\\n".join(ln'),
    ("R8 owner rows by distance (field match off)", TM, "        hit = len(toks & implied)", "        hit = 1"),
    ("R9 Greek questions graded", TM, "    _ungradable = _mostly_non_latin(str(query))", "    _ungradable = False"),
    ("R10 Greek kind lookup without an owner question", TM,
     "                        _own = await asyncio.to_thread(_ofm, query, extended=_about_owner)",
     "                        _own = await asyncio.to_thread(_ofm, query, extended=True)"),
    ("R17 profile rows fill unrelated recalls", TM,
     "        if m_type == \"identity\" and not _about_owner and (_ungradable or score >= RECALL_MEDIUM_D):",
     "        if False:"),
    ("K6 base vocabulary is the full one", GR, "        _kinds = self._CATEGORY_WORDS if extended else self._CATEGORY_WORDS_BASE",
     "        _kinds = self._CATEGORY_WORDS"),
    ("R14 owner section subtracts identity rows", TM,
     "            r for r in (results or []) if isinstance(r, dict)]\n",
     "            r for r in (results or []) if isinstance(r, dict) and (r.get(\"metadata\") or {}).get(\"type\") != \"identity\"]\n"),
    ("R15 a role does not name the person", TM,
     "            if \"name\" in _kt and (_kt - {\"name\"}) & implied and isinstance(_v, str) and len(_v) <= 60:",
     "            if False:"),
    ("R16 look up is a looks question", TM, "how do i look(?!\\s+(?:up|for|into|at|through|it))|", "how do i look|"),
    ("R11 a description on any describe word", TM,
     "                if not (_named or (cat == \"root\" and _OWN_LOOKS_RE.search(str(query)))):",
     "                if False:"),
    ("R12 on-demand facts without the gate", TM, "            if _about_owner:\n                _od = await",
     "            if True:\n                _od = await"),
    ("R13 the graph section graded UNRELATED", TM,
     '        if _graph_section:\n            _head += " The owner\'s stored facts below are not graded by that match."',
     '        if False:\n            _head += " The owner\'s stored facts below are not graded by that match."'),
    # ── project fallback
    ("P1 fallback in a public channel", PR, "        if reply_is_public() or requester_is_member() or _mostly_non_latin(str(term)):",
     "        if requester_is_member() or _mostly_non_latin(str(term)):"),
    ("P2 fallback offers profile rows", PR, '{"type": {"$nin": list(_EXCL) + ["identity"]}}', '{"type": {"$nin": list(_EXCL)}}'),
    ("P3 fallback MEDIUM gate", PR, "                if score < RECALL_HIGH_D:", "                if score < 0.36:"),
    # ── the owner gate
    ("G1 bare με is first person", BU, "μου|μας|εγώ|είμαι|έχω|μένα|εμένα)", "μου|μας|εγώ|είμαι|έχω|με|μένα|εμένα)"),
    ("G2 no με λένε", BU, r'r"|\bμε\s+(?:λένε|', r'r"|\bXX\s+(?:λένε|'),
    ("G3 -σσα nouns are I", BU, "|(?<!σ)(?:ησα|ισα|ασα|εσα|(?<!ο)υσα|ωσα))(?!\\w)\")", "|(?:ησα|ισα|ασα|εσα|(?<!ο)υσα|ωσα))(?!\\w)\")"),
    ("G8 no past -σα", BU, "|(?<!σ)(?:ησα|ισα|ασα|εσα|(?<!ο)υσα|ωσα))(?!\\w)\")", ")(?!\\w)\")"),
    ("G9 capitalised names are verbs", BU, "        r\"(?<!\\w)[α-ωάέήίόύώϊϋΐΰ]{2,}", "        r\"(?i)(?<!\\w)[α-ωάέήίόύώϊϋΐΰ]{2,}"),
    ("G10 a capitalised first verb missed", BU, "        _q = re.sub(r\"^\\s*\\w\", lambda mm: mm.group(0).lower(), q, count=1)", "        _q = q"),
    ("G11 no Greek openers", BU, "|πες|πείτε|πειτε|θύμισέ|θύμισε|θυμισε|ξέρεις|ξερεις)", ")"),
    ("G12 οκτώ is a verb", BU, '"οκτώ", "οχτώ", ', ""),
    ("G4 adverbs are verbs", BU, "        return any(m.group(0).lower() not in cls._NOT_A_VERB\n", "        return any(True\n"),
    ("G5 English ; is a question", BU, "                or (cls._GREEK_QMARK.search(q) and cls._GREEK_LETTER.search(q))):",
     "                or cls._GREEK_QMARK.search(q)):"),
    ("G6 no Greek verb rule", BU, "        return any(m.group(0).lower() not in cls._NOT_A_VERB\n                   for m in cls._GREEK_FIRST_PERSON_VERB.finditer(_q))",
     "        return False"),
    ("G7 unaccented question words lost", BU, "πότε|ποτε|πώς", "πότε|πώς"),
    # ── graph
    ("K1 kind words not folded", GR, "        _folded.setdefault(_fold(_k), set()).update(_v)", "        _folded.setdefault(_k, set()).update(_v)"),
    ("K2 raw writes expire siblings", GR, "if (pn in self._FUNCTIONAL_PREDICATES and not raw", "if (pn in self._FUNCTIONAL_PREDICATES"),
    ("K3 graph mirror unqualified", GR, '        pred = "HAS_" + mirror_name(category, key).upper().replace(" ", "_")',
     '        pred = "HAS_" + str(key or "").upper().replace(" ", "_")'),
    ("K4 graph legacy never retired", GR, "        if retire_legacy and legacy != pred:\n            # KB review 10-30 r2",
     "        if False:\n            # KB review 10-30 r2"),
    ("K5 work does not cover company", GR, '"employer", "work", "company", "role"}', '"employer", "work"}'),
    # ── vector mirrors
    ("V1 vector mirror unqualified", VE, '        prefix = f"User {mirror_name(category, key)} is "', '        prefix = f"User {key} is "'),
    ("V2 vector legacy never retired", VE, "            stale += [i for i, d in docs if str(d).startswith(legacy)]", "            pass"),
    ("V3 mirror name ignores category", PF, '    return f"{cat}_{k}"\n', "    return k\n"),
    # ── mirrors orchestration
    ("M1 retire even when root has the key", TM, "        _retire = cat != _ROOT and k not in ((profile_memory.load() or {}).get(_ROOT) or {})",
     "        _retire = cat != _ROOT"),
    ("M2 resync on a missing profile", TM, "    if _fp is not None and not Path(str(_fp)).exists():\n        return 0, 0",
     "    if False:\n        return 0, 0"),
    ("M3 resync the placeholder name", TM, '        return str(cat) == "root" and str(k) == "name" and str(v or "").strip() in ("", "User")',
     "        return False"),
    ("M4 forget sweeps take mirror rows", TM, "def _profile_mirror_texts(profile_memory) -> set:",
     "def _profile_mirror_texts(profile_memory) -> set:\n    return set()\n\n\ndef _unused_mirror_texts(profile_memory) -> set:"),
    ("M5 maintenance never resyncs", DR, "resync_all_owner_mirrors(", "(lambda *a, **k: __import__('asyncio').sleep(0, (0, 0)))("),
    ("M6 façade lets resync through", RO, "    r\"sync|resync|rollback|stamp)_\")", "    r\"stamp)_\")"),
    # ── ingest
    ("I1 overlap not clamped", HE, "    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 4))",
     "    chunk_overlap = max(0, int(chunk_overlap))"),
    ("I2 long header text lost", HE, "        if len(header) > 120:\n", "        if False:\n"),
    ("I3 dotted names lose stem match", TM, '_has_ext = bool(re.fullmatch(r"\\.[A-Za-z][A-Za-z0-9]{0,4}", Path(filename).suffix))',
     "_has_ext = bool(Path(filename).suffix)"),
    ("I4 an ambiguous name picks the first", TM, "                if len(_matches) > 1:", "                if False:"),
    ("I5 binary not sniffed", TM, "                _why = _binary_content_reason(file_path)", "                _why = ''"),
    ("I6 a blank page is a scan", PI, "                if _img:\n                    st.empty_pages.append(page_i + 1)",
     "                if True:\n                    st.empty_pages.append(page_i + 1)"),
    ("I7 titles match mid-word", PI, '+ r")(?!\\w)",', '+ r")",'),
    ("I8 one crumb per page", PI, "            for crumb, piece_text in split_page_by_sections(text, _start, _here):",
     "            for crumb, piece_text in [(_start, text)]:"),
    ("I9 no marker on ingest", VE, "                self.mark_ingest_started(filename)\n", ""),
    ("I10 delete keeps the marker", VE, "            self.clear_ingest_marker(filename)\n            # §4PP", "            # §4PP"),
    ("I11 boot rollback off", VE, "            self._rollback_interrupted_ingests()\n", ""),
    ("I12 delete not type-scoped", VE, '                {"type": {"$in": list(self.DOCUMENT_ROW_TYPES)}}]})\n            self._update_library_index(filename, "remove")',
     '                ]})\n            self._update_library_index(filename, "remove")'),
    ("I14 dir names lose the stem match", TM, "target_stem = Path(filename).name.lower()", "target_stem = filename.lower()"),
    ("I15 a bodiless header is dropped", HE, "                elif current_header:\n                    # KB review 10-30 r3",
     "                elif False:\n                    # KB review 10-30 r3"),
    ("I13 pdf keeps the marker", PI, "            _done(filename)\n        return _stats", "            pass\n        return _stats"),
]

if __name__ == "__main__":
    m.main()
