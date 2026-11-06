"""§4QL battery — outcome-labelled experience memory (2026-10-21). Reuses the §4QA
harness (in place, backup + restore + byte compare, two-tier, controls).
    python3 scripts/mutate_4ql.py [all|check|rerun X1 X2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

X = "src/ghost_agent/memory/experience.py"; B = "src/ghost_agent/core/bus.py"; D = "src/ghost_agent/core/dream.py"
L = "src/ghost_agent/utils/logging.py"; MN = "src/ghost_agent/main.py"; AG = "src/ghost_agent/core/agent.py"
m.PINS = ["tests/test_4ql_experience.py", "tests/test_bus_rag_fusion.py", "tests/test_4nc_producers.py",
          "tests/test_bus_rrf_fixes.py", "tests/test_4mv_speed_arms.py", "tests/test_4oj_quote_bound.py"]


def _tests():
    import glob, subprocess
    pat = r"experience\|core.bus\|MemoryBus\|prompt_arm\|PROMPT_ARMS\|_store_maintenance\|hydrate_context"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", X, "def normalize_request(text: str) -> str:", "# ctl\ndef normalize_request(text: str) -> str:"),
    ("CONTROL known-bad (must DIE): the tier is on for every production turn", X,
     '    if os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "on", "yes"):\n        return True',
     "    return True"),
    # labels
    ("X1 a thumb overrides a gold grade", X,
     "        if prev is None or LABEL_RANK.get(source, 0) >= LABEL_RANK.get(prev[1], 0):", "        if True:"),
    ("X2 judge verdicts become labels", X, "            if src not in OWNER_THUMB_SOURCES:\n                continue",
     "            if False:\n                continue"),
    ("X23 the owner's correction is not a label", X,
     '                       "user_correction",   # the owner\'s own next-message correction (always a failure)\n', ""),
    ("X24 reason cut mid-clause", X,
     '    return (cut[:end + 1] if end >= REASON_CHARS // 2 else cut.rsplit(" ", 1)[0] + "…").strip()', "    return cut"),
    # who is indexed
    ("X3 a member's or a probe's turn is indexed", X,
     "                if not tid or tid in excluded or not is_owner_turn(t) or not delivered(t):",
     "                if not tid or tid in excluded or not delivered(t):"),
    ("X4 an undelivered reply is indexed", X,
     "                if not tid or tid in excluded or not is_owner_turn(t) or not delivered(t):",
     "                if not tid or tid in excluded or not is_owner_turn(t):"),
    ("X5 a go-ahead is indexed", X, "                if len(content_words(request)) < MIN_CONTENT_WORDS:",
     "                if len(content_words(request)) < 0:"),
    ("X6 the hold-out is ignored by sync", X,
     "                if not tid or tid in excluded or not is_owner_turn(t) or not delivered(t):",
     "                if not tid or not is_owner_turn(t) or not delivered(t):"),
    ("X7 a withdrawn label keeps its row", X, "                gone = [t for t in have if t in seen and t not in keep]", "                gone = []"),
    ("X19 old rows never recomputed", X, "                if have.get(tid) == _lab_stored and tid not in stale:",
     "                if have.get(tid) == _lab_stored:"),
    # similarity
    ("X8 the floor is ignored", X, "                if sim < SIMILARITY_FLOOR:\n                    continue",
     "                if False:\n                    continue"),
    ("X9 identical wording is not special", X, '            if r["request_norm"] == norm:\n                sim = 1.0',
     "            if False:\n                sim = 1.0"),
    ("X10 exclude_tid ignored", X, '        rows = [r for r in self._rows() if r["tid"] != str(exclude_tid or "")]',
     "        rows = list(self._rows())"),
    ("X20 rows cached past a file change", X,
     '            if self._cache is None or stamp != getattr(self, "_cache_stamp", None):', "            if self._cache is None:"),
    ("X25 a short request matches", X,
     "        if len(content_words(request)) < MIN_CONTENT_WORDS:\n            return []\n        rows =",
     "        if False:\n            return []\n        rows ="),
    # what is derived / rendered
    ("X11 a thumb-labelled answer is shown", X,
     '                if (lab[0] in GOOD_GRADES and lab[1] != "thumb" and not moving and kind in ANSWER_KINDS',
     "                if (lab[0] in GOOD_GRADES and not moving and kind in ANSWER_KINDS"),
    ("X12 a moving answer is shown", X,
     '                if (lab[0] in GOOD_GRADES and lab[1] != "thumb" and not moving and kind in ANSWER_KINDS',
     '                if (lab[0] in GOOD_GRADES and lab[1] != "thumb" and kind in ANSWER_KINDS'),
    ("X13 a status/tool turn's answer is shown", X,
     '                if (lab[0] in GOOD_GRADES and lab[1] != "thumb" and not moving and kind in ANSWER_KINDS',
     '                if (lab[0] in GOOD_GRADES and lab[1] != "thumb" and not moving'),
    ("X22 the time-bound rule is gone", X, "    if _TIME_BOUND_RE.search(t):\n        return True", "    pass"),
    ("X14 a failed page is a source", X,
     "            if bad:\n                failed.append(f\"{_host(url)} — {_error_head(c)}\".rstrip(\" —\"))\n            else:\n                sources.append(url)",
     "            sources.append(url)"),
    ("X15 page prose after the banner kept", X, '    head = _BANNER_TAIL_RE.sub("", head)', "    pass"),
    ("X16 any RIGHT case, not the closest", X,
     '        good = sorted([r for r in rows if r.get("grade") in GOOD_GRADES],\n                      key=lambda r: (-float(r.get("sim") or 0), -float(r.get("ts") or 0)))',
     '        good = [r for r in rows if r.get("grade") in GOOD_GRADES]'),
    ("X26 the WRONG case loses its reason", X,
     "                line += f\"{(': ' + why) if why else ''}. It did: {did}. Do not repeat that.\"",
     "                line += f\". It did: {did}. Do not repeat that.\""),
    # the gate
    ("X17 production ON without the switch", X,
     '    if os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "on", "yes"):\n        return True',
     '    if os.environ.get(ENV_FLAG, "1").strip().lower() in ("1", "true", "on", "yes"):\n        return True'),
    ("X18 the probe arm is not honoured", X, "        return prompt_arm() == ARM", '        return prompt_arm() == "no_lessons"'),
    ("X21 sync without a store raises", X,
     '    if store is None or not hasattr(store, "sync"):\n        return {}', "    pass"),
    # the bus
    ("B1 the fetcher ignores the gate", B, "            if not experience_tier_enabled():\n                return []",
     "            pass"),
    ("B2 the fan-out drops the tier", B, "            self._fetch_experience(scope_request or query) if extras else _none(),\n", ""),
    ("B3 judged on the sub-query, not the user's words", B, "            self._fetch_experience(scope_request or query) if extras else _none(),",
     "            self._fetch_experience(query) if extras else _none(),"),
    ("B4 no header for the block", B, '            "experience": _EXPERIENCE_HEADER,\n', ""),
    ("B5 the log line hides the tier", B, '                  ("episodic", "e"), ("session", "sess"), ("experience", "x"), ("history", "h"))',
     '                  ("episodic", "e"), ("session", "sess"), ("history", "h"))'),
    # the writers and the wiring
    ("D1 maintenance never syncs", D, "            _xs = await asyncio.to_thread(sync_experience_index, self.context)",
     "            _xs = {}"),
    ("L1 the arm is not accepted", L, 'PROMPT_ARMS: frozenset = frozenset({"no_lessons", "quote_bound", "experience", "fact_history"})',
     'PROMPT_ARMS: frozenset = frozenset({"no_lessons", "quote_bound", "fact_history"})'),
    ("M1 the lifespan bus has no store", MN, "        experience_memory=getattr(context, 'experience_memory', None),   # §4QL\n", ""),
    ("M2 boot never builds the store", MN, "        context.experience_memory = ExperienceMemory(_mdir, embed_fn=_embed_locked)",
     "        context.experience_memory = None"),
    ("A1 the lazy bus has no store", AG, "            experience_memory=getattr(self.context, 'experience_memory', None),   # §4QL\n", ""),
    # fresh-eye review r1 fixes (mutated FIRST next round, R3)
    ("R1 public-channel turns indexed", X, '                if str(ex.get("surface") or "") == "public":\n                    continue', "                pass"),
    ("R2 thumb beats census by lookup order", X, '                lab = best_label(labels.get(tid), labels.get("req:" + rid) if rid else None)',
     '                lab = labels.get(tid) or (labels.get("req:" + rid) if rid else None)'),
    ("R3 a build turn is knowledge", X,
     '    if all(n in _SEARCH_TOOLS or n in ("browser", "news_headlines", "fact_check") for n in names):',
     '    if any(n in _SEARCH_TOOLS or n in ("browser", "news_headlines", "fact_check") for n in names):'),
    ("R4 moving reason unmasked at render", X, "            if r.get(\"moving\"):\n                why = mask_figures(why)", "            pass"),
    ("R4b moving reason unmasked at sync", X, "                reason = mask_figures(lab[2]) if moving else lab[2]", "                reason = lab[2]"),
    ("R5 an unseen turn is removed", X, "                gone = [t for t in have if t in seen and t not in keep]",
     "                gone = [t for t in have if t not in keep]"),
    ("R5b archive not walked", X, "    return store.sync(iter_teachable(col.iter_trajectories(include_archive=True), consumer=\"experience\"), labels)",
     "    return store.sync(iter_teachable(col.iter_trajectories(since_days=180), consumer=\"experience\"), labels)"),
    ("R6 a follow-up's answer rendered", X, "                        and not followup and not is_system_command(request)):",
     "                        and not is_system_command(request)):"),
    ("R6b a system command's answer rendered", X, "                        and not followup and not is_system_command(request)):",
     "                        and not followup):"),
    ("R7 clockwork thumbs dropped", X, '                       "human_feedback:clockwork", "human_feedback:api",\n', ""),
    ("R8 the thumb's default note is a reason", X, "    if r.lower() in _EMPTY_REASONS:\n        return \"\"", "    pass"),
    ("R9 query strings kept in sources", X, "                url = _page_only(v.strip())[:200]", "                url = v.strip()[:200]"),
    ("R9b redaction placeholders are sources", X,
     '            if isinstance(v, str) and v.startswith(("http://", "https://")) and "<REDACTED" not in v:',
     '            if isinstance(v, str) and v.startswith(("http://", "https://")):'),
    ("R10 experience weight unlisted", B, '        "factual":    {"graph": 2.03, "experience": 1.0, "history": 1.0},',
     '        "factual":    {"graph": 2.03},'),
    ("R11 extras fetched per sub-query", B,
     "                self._fetch_all_tiers(sq, scope_request=_scope_request, extras=(_i == 0)))",
     "                self._fetch_all_tiers(sq, scope_request=_scope_request, extras=True))"),
    ("R13 concurrent insert aborts the sync", X,
     '                            "INSERT OR REPLACE INTO experience(tid, req_id, ts, request, request_norm, kind, grade, label_source,"',
     '                            "INSERT INTO experience(tid, req_id, ts, request, request_norm, kind, grade, label_source,"'),
]

if __name__ == "__main__":
    m.main()
