#!/usr/bin/env python3
"""§4SS battery — req b45ec99f (2026-11-01): the experiment hold reads the TASK's model subject, the DONE
sweep keeps documents and project-named files, the reply says "removed by the cleanup" for a swept file.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4ss.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = "src/ghost_agent/core/project_advancer.py"
W = "src/ghost_agent/core/workspace_cleanup.py"
AG = "src/ghost_agent/core/agent.py"
_RX_L = 'r"(?<![\\w.\\-])" + _re.escape(token) + r"(?![\\w\\-]|\\.\\w)",'
MUT = [
    ('CONTROL known-bad', A, '    if _CODE_FILE_RE.search(topic) and not _names_a_model(_CODE_FILE_RE.sub(" ", topic)):\n        return False', '    if True:\n        return False'),
    ('CONTROL no-op', W, 'def _is_document_deliverable(rel: str) -> bool:', 'def _is_document_deliverable(rel: str) -> bool:  # no-op'),
    ('A4 topic-ignored', A, '    if not (_names_a_model(goal) or _names_a_model(topic)', '    if not (_names_a_model(goal)'),
    ('A5 probe-not-a-model', A, '    return bool(_MODEL_GOAL_RE.search(text) or _PROBE_RE.search(text))', '    return bool(_MODEL_GOAL_RE.search(text))'),
    ('A6 probe-case', A, '_PROBE_RE = re.compile(r"\\bprobe_model\\b", re.IGNORECASE)', '_PROBE_RE = re.compile(r"\\bprobe_model\\b")'),
    ('A7 bare-model-again', A, '    return bool(_MODEL_GOAL_RE.search(text) or _PROBE_RE.search(text))', "    return bool(_MODEL_GOAL_RE.search(text) or _PROBE_RE.search(text) or re.search(r'\\bmodels?\\b', text, re.I))"),
    ('A8 caller-no-texts', A, '            and is_model_experiment(nxt.description, proj.get("goal"),\n                                    _project_task_texts(store, project_id))):', '            and is_model_experiment(nxt.description, proj.get("goal"))):'),
    ('A9 texts-raise', A, '    except Exception:  # noqa: BLE001\n        logger.debug("project task texts unreadable", exc_info=True)', '    except ZeroDivisionError:  # noqa: BLE001\n        logger.debug("project task texts unreadable", exc_info=True)'),
    ('B1 doc-recovery-off', W, '            if not (_is_source_like(rel) or _is_data_deliverable(rel)\n                    or _is_document_deliverable(rel)):', '            if not (_is_source_like(rel) or _is_data_deliverable(rel)):'),
    ('B2 no-pdf', W, '    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".epub",', '    ".docx", ".doc", ".odt", ".rtf", ".epub",'),
    ('B2b no-docx', W, '    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".epub",', '    ".pdf", ".doc", ".odt", ".rtf", ".epub",'),
    ('B2c no-slides', W, '    ".pptx", ".ppt", ".odp", ".key",', '    ".ppt", ".odp", ".key",'),
    ('B2d no-sheets', W, '    ".xlsx", ".xls", ".ods", ".pages", ".numbers",', '    ".xls", ".ods", ".pages", ".numbers",'),
    ('B2e doc-case', W, '    name = rel.split("/")[-1].lower()\n    dot = name.rfind(".")\n    return dot > 0 and name[dot:] in _DOCUMENT_DELIVERABLE_SUFFIXES', '    name = rel.split("/")[-1]\n    dot = name.rfind(".")\n    return dot > 0 and name[dot:] in _DOCUMENT_DELIVERABLE_SUFFIXES'),
    ('B3 sweep-named-off', W, '    named = _named_by_project(store, project_id, root) - set(keep)', '    named = set()'),
    ('B3b named-minus-keep', W, '    named = _named_by_project(store, project_id, root) - set(keep)', '    named = _named_by_project(store, project_id, root)'),
    ('B4 tidy-named-off', W, '                          and m.lower() not in _named_low]', '                          ]'),
    ('B5 no-dot-strip', W, '            segs = [g.rstrip(".") for g in m.group(0).split("/")]', '            segs = m.group(0).split("/")'),
    ('B5b run-no-dot', W, '_PATH_RUN_RE = re.compile(r"[\\w.\\-/]+")', '_PATH_RUN_RE = re.compile(r"[\\w\\-/]+")'),
    ('B5c run-no-dash', W, '_PATH_RUN_RE = re.compile(r"[\\w.\\-/]+")', '_PATH_RUN_RE = re.compile(r"[\\w./]+")'),
    ('B5d run-no-slash', W, '_PATH_RUN_RE = re.compile(r"[\\w.\\-/]+")', '_PATH_RUN_RE = re.compile(r"[\\w.\\-]+")'),
    ('B7 media-by-name', W, '            if name[dot:] in _MEDIA_SUFFIXES:\n                if low in paths:', '            if name[dot:] in _MEDIA_SUFFIXES and False:\n                if low in paths:'),
    ('B8 media-path-is-name', W, '                if low in paths:\n                    found.add(rel)', '                if name in names:\n                    found.add(rel)'),
    ('B9 no-ext-allowed', W, '            if dot <= 0:\n                continue                  # "data"', '            if False:\n                continue                  # "data"'),
    ('B11 nonmedia-off', W, '            elif name in names:\n                found.add(rel)', '            elif False:\n                found.add(rel)'),
    ('B14 no-goal', W, '        texts.append(str(proj.get("goal") or ""))', '        pass'),
    ('B15 no-descr', W, '            texts.append(str(t.get("description") or ""))', '            pass'),
    ('B16 no-result', W, '            texts.append(str(t.get("result_summary") or ""))', '            pass'),
    ('B17 tasks-raise', W, '    except Exception:\n        logger.debug("named-file scan: tasks unreadable", exc_info=True)', '    except ZeroDivisionError:\n        logger.debug("named-file scan: tasks unreadable", exc_info=True)'),
    ('B18 goal-raise', W, '    except Exception:\n        logger.debug("named-file scan: goal unreadable", exc_info=True)', '    except ZeroDivisionError:\n        logger.debug("named-file scan: goal unreadable", exc_info=True)'),
    ('C1 swept-branch-off', AG, '        if _link_was_swept(name, swept):\n            # §4SS', '        if False:\n            # §4SS'),
    ('C2 not-project', AG, '            if len(parts) < 3 or parts[0] != "projects":', '            if len(parts) < 3:'),
    ('C3 no-casefold-id', AG, '            key = parts[1].casefold()', '            key = parts[1]'),
    ('C4 no-casefold-rel', AG, '            if rels is not None and parts[2].casefold() in rels:', '            if rels is not None and parts[2] in rels:'),
    ('C5 hard-delete-is-sweep', AG, '            if rels is not None and parts[2].casefold() in rels:', '            if rels is None or parts[2].casefold() in rels:'),
    ('C6 any-file-of-project', AG, '            if rels is not None and parts[2].casefold() in rels:', '            if True:'),
    ('C7 no-unquote', AG, '            r = _uq(raw.split("?", 1)[0].split("#", 1)[0]).lstrip("/")', '            r = raw.split("?", 1)[0].split("#", 1)[0].lstrip("/")'),
    ('C8 no-punct-reading', AG, '        for raw in (name, name.split("|", 1)[0].rstrip(".,;:!?`*\'\\"")):\n            r = _uq(', '        for raw in (name,):\n            r = _uq('),
    ('C9 stale-marks-kept', AG, '                if (time.time() - float(v[0])) < _REMOVAL_MARK_FRESH_WINDOW_S}\n    except Exception:  # noqa: BLE001\n        return None\n\n\ndef _link_was_swept', '                if True}\n    except Exception:  # noqa: BLE001\n        return None\n\n\ndef _link_was_swept'),
    ('C10 finalize-no-swept', AG, '                final_ai_content, getattr(self.context, "sandbox_dir", None),\n                swept=_fresh_removal_marks(self.context))', '                final_ai_content, getattr(self.context, "sandbox_dir", None))'),
    ('C11 retry-no-swept', AG, '                    _ff_retry_text, getattr(self.context, "sandbox_dir", None),\n                    swept=_fresh_removal_marks(self.context))', '                    _ff_retry_text, getattr(self.context, "sandbox_dir", None))'),
    ('C12 stream-gone-empty', AG, '    gone = [n for n in missing if _link_was_swept(n, swept)]', '    gone = []'),
    ('C13 stream-never-all', AG, '    never = [n for n in missing if n not in gone]', '    never = list(missing)'),
    ('C14 stream-swept-none', AG, '                            _st_missing, _fresh_removal_marks(self.context))', '                            _st_missing, None)'),
    ('C15 verifier-marks-none', AG, '                _swept = _fresh_removal_marks(self.context)', '                _swept = None'),
    ('A1 code-beats-model', A, '    if _CODE_FILE_RE.search(topic) and not _names_a_model(_CODE_FILE_RE.sub(" ", topic)):', '    if _CODE_FILE_RE.search(topic):'),
    ('A1b code-beats-name', A, '    if _CODE_FILE_RE.search(topic) and not _names_a_model(_CODE_FILE_RE.sub(" ", topic)):', '    if _CODE_FILE_RE.search(topic) and not _PROBE_RE.search(topic):'),
    ('A2 code-check-off', A, '    if _CODE_FILE_RE.search(topic) and not _names_a_model(_CODE_FILE_RE.sub(" ", topic)):', '    if False:'),
    ('A3 project-texts-ignored', A, '            or any(_PROBE_RE.search(str(t or "")) for t in (project_texts or ()))):', '            ):'),
    ('A3b sibling-names-count', A, '            or any(_PROBE_RE.search(str(t or "")) for t in (project_texts or ()))):', '            or any(_names_a_model(t) for t in (project_texts or ()))):'),
    ('A10 no-gemma', A, '                r"gemma\\w*|deepseek\\w*")', '                r"deepseek\\w*")'),
    ('A10c no-deepseek', A, '                r"gemma\\w*|deepseek\\w*")', '                r"gemma\\w*")'),
    ('B6 full-path-only', W, '            for k in range(max(0, len(segs) - _PATH_CHAIN_MAX), len(segs)):\n                paths.add("/".join(segs[k:]))', '            paths.add("/".join(segs))'),
    ('B10 no-lower', W, '        low = _nfc(str(text or "").lower()).replace("\\\\", "/")', '        low = _nfc(str(text or "")).replace("\\\\", "/")'),
    ('B10b no-backslash', W, '        low = _nfc(str(text or "").lower()).replace("\\\\", "/")', '        low = _nfc(str(text or "").lower())'),
    ('B10c no-nfc-text', W, '        low = _nfc(str(text or "").lower()).replace("\\\\", "/")', '        low = str(text or "").lower().replace("\\\\", "/")'),
    ('B10d no-nfc-rel', W, '            low = _nfc(rel.lower())', '            low = rel.lower()'),
    ('B10e chains-unbounded', W, '            for k in range(max(0, len(segs) - _PATH_CHAIN_MAX), len(segs)):', '            for k in range(len(segs)):'),
    ('B10f chain-one', W, '_PATH_CHAIN_MAX = 32', '_PATH_CHAIN_MAX = 1'),
    ('B12 symlink-kept', W, '                if fpath.is_symlink():\n                    continue\n                rel = fpath.relative_to(root).as_posix()\n            except (OSError, ValueError):\n                continue\n            if _is_debris(rel):\n                continue\n            low = _nfc(', '                rel = fpath.relative_to(root).as_posix()\n            except (OSError, ValueError):\n                continue\n            if _is_debris(rel):\n                continue\n            low = _nfc('),
    ('B13 debris-named', W, '            if _is_debris(rel):\n                continue\n            low = _nfc(', '            low = _nfc('),
    ('A1c filename-is-model', A, '    if _CODE_FILE_RE.search(topic) and not _names_a_model(_CODE_FILE_RE.sub(" ", topic)):', '    if _CODE_FILE_RE.search(topic) and not _names_a_model(topic):'),
    ('B10g chain-eight', W, '_PATH_CHAIN_MAX = 32', '_PATH_CHAIN_MAX = 8'),
    ('B19 mark-cap-64', W, '_MARK_MAX_PATHS = 512', '_MARK_MAX_PATHS = 64'),
]
PINS = ["tests/test_4ss_b45ec99f_fixes.py"]


def _tests():
    pat = (r"workspace_cleanup\|is_model_experiment\|model_experiment\|_drop_missing_download_links\|"
           r"recent_workspace_removals\|swept_removals\|_link_was_swept\|_fresh_removal_marks\|tidy_project_workspace")
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
        return (sorted(set(ids) | set(PINS)), f"cov {len(ids)} tests + pins") if ids else (PINS, "fallback pins")
    print(f"{len(full)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
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
