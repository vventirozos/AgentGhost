#!/usr/bin/env python3
"""§4SY battery — the health check's internals report (tools/health_internals.py), the problems line and the
graph count in `tool_check_health`, the node reader split out of introspect, and the READY banner waiting for the
labs boot line. Coverage-selected tests in parallel tree copies (scripts/battery_cov.py); the real src/ is never
mutated. No-op control must SURVIVE, the known-bad control must DIE.
    python3 scripts/mutate_4sy.py [check | -j6 all | rerun H1 S2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HI = "src/ghost_agent/tools/health_internals.py"
SY = "src/ghost_agent/tools/system.py"
IN = "src/ghost_agent/tools/introspect.py"
LB = "src/ghost_agent/tools/labs.py"
MN = "src/ghost_agent/main.py"
MUT = [
    ("CONTROL no-op", HI, 'def _plural(n: int, one: str, many: str = "") -> str:', 'def _plural(n: int, one: str, many: str = "") -> str:  # no-op'),
    ("CONTROL known-bad", HI, "    return ([line for line, _p, _u in got], [p for _l, ps, _u in got for p in ps],", "    return ([], [p for _l, ps, _u in got for p in ps],"),
    ("H1 bool-is-a-count", HI, "    if type(value) is not int:           # bool and a MagicMock are not counts", "    if not isinstance(value, int):"),
    ("H2 unreadable-not-a-problem", HI, '            self.problems.append(f"{name} could not be read ({why})")\n', ""),
    ("H3 unreadable-not-on-line", HI, '            self.parts.append(f"{name} UNREADABLE ({why})")\n', ""),
    ("H4 failing-node-not-flagged", HI, '            if isinstance(st, dict) and st.get("state") in ("open", "half_open"):', "            if False:"),
    ("H4b half-open-not-flagged", HI, 'st.get("state") in ("open", "half_open"):', 'st.get("state") == "open":'),
    ("H4c every-node-flagged", HI, '            if isinstance(st, dict) and st.get("state") in ("open", "half_open"):', "            if isinstance(st, dict):"),
    ("H5 no-client-not-flagged", HI, '        out.problem("no model client is attached")\n', ""),
    ("H5b models-in-a-thread", HI, '    ("Models", _models, False),', '    ("Models", _models, True),'),
    ("H5c stores-on-the-loop", HI, '    ("Memory stores", _memory, True),', '    ("Memory stores", _memory, False),'),
    ("H6 no-memory-flag-ignored", HI, '    return getattr(getattr(context, "args", None), "no_memory", None) is True', "    return False"),
    ("H6b no-memory-truthy", HI, '    return getattr(getattr(context, "args", None), "no_memory", None) is True', '    return bool(getattr(getattr(context, "args", None), "no_memory", None))'),
    ("H7c unloaded-store-not-named", HI, '            self.parts.append(f"{name} NOT loaded")\n', ""),
    ("H8 damaged-file-not-flagged", HI, '            self.problems.append(f"the {name} file is damaged ({why}); it was left as it is")\n', ""),
    ("H8b damaged-file-read-anyway", HI, "        return bool(why)", "        return False"),
    ("H9 damaged-index-not-flagged", HI, '            out.problem(f"the document index file is damaged ({why}); it was left as it is")\n', ""),
    ("H10 no-embedder-not-flagged", HI, '        if getattr(ms, "embedding_fn", None) is None:', "        if False:"),
    ("H11 no-collection", HI, '        if col is None:\n            raise _Unreadable("no collection")\n', ""),
    ("H14 pending-episodes-dropped", HI, '(f" ({pending} not yet consolidated)" if type(pending) is int and pending else "")', '""'),
    ("H15 owner-rules-zero", HI, "_plural(len(_or.rules(system_dir)), 'owner rule')", "_plural(0, 'owner rule')"),
    ("H16 damaged-rules-file-not-named", HI, "_or.damaged(system_dir) or None", "None"),
    ("H17 ledger-wrong-dir", HI, "    system_dir = system_dir.parent if system_dir is not None else None", "    system_dir = system_dir if system_dir is not None else None"),
    ("H19 status-mix-order", HI, "sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))", "sorted(by.items())"),
    ("H20 task-store-flag-ignored", HI, '        if _tasks_mod._STORE_STATE.get("unreadable") is True:', "        if False:"),
    ("H21 empty-key-not-flagged", HI, '            out.problem("a labs host is set but its key file holds no key (labs cannot be used)")\n', ""),
    ("H22 inventory-created", HI, '        if mdir is None or not (mdir / "lab_inventory.db").is_file():', "        if mdir is None:"),
    ("H23 registry-created", HI, '    if mdir is not None and (mdir / "runs.db").is_file():', "    if mdir is not None:"),
    ("H24 destroyed-labs-counted", HI, "        live = inv.rows(live_only=True, strict=True)", "        live = inv.rows(strict=True)"),
    ("H24b labs-read-error-is-zero", HI, "        live = inv.rows(live_only=True, strict=True)", "        live = inv.rows(live_only=True)"),
    ("H24c unopened-inventory-is-zero", HI, '        if inv is None:                  # the file is there and did not open: not "0 live labs"\n            raise _Unreadable("the inventory did not open")\n        live = inv.rows(live_only=True, strict=True)', "        live = inv.rows(live_only=True, strict=True) if inv is not None else []"),
    ("H25 ended-runs-counted", HI, "            live = reg.rows(live_only=True, strict=True)", "            live = reg.rows(strict=True)"),
    ("H25b unopened-registry-is-zero", HI, '            if reg is None:\n                raise _Unreadable("the registry did not open")\n            live = reg.rows(live_only=True, strict=True)', "            live = reg.rows(live_only=True, strict=True) if reg is not None else []"),
    ("H26 mock-sandbox-used", HI, '    if not isinstance(sbx, DockerSandbox):\n        out.parts.append("sandbox not attached")', '    if sbx is None:\n        out.parts.append("sandbox not attached")'),
    ("H27 cut-off-not-flagged", HI, '        if getattr(sbx, "_cut_off", False) is True:', "        if False:"),
    ("H27a cut-off-without-container", HI, '        if getattr(sbx, "container", None) is None:\n            return "sandbox container not started (it starts on first use)"\n', ""),
    ("H27b unconfined-without-tor-only", HI, '            if getattr(getattr(context, "args", None), "mandatory_tor", None) is True:\n                out.problem("the sandbox could not be confined to Tor', '            if True:\n                out.problem("the sandbox could not be confined to Tor'),
    ("H27c blocked-reads-unconfined", HI, '{"enforced": "Tor-only egress enforced", "blocked": "direct egress blocked"}', '{"enforced": "Tor-only egress enforced", "blocked": "egress NOT confined"}'),
    ("H29 no-backup-not-flagged", HI, '                out.problem("no store backup exists")\n', ""),
    ("H30a overdue-later", HI, "BACKUP_OVERDUE_FACTOR = 2", "BACKUP_OVERDUE_FACTOR = 3"),
    ("H30b overdue-sooner", HI, "BACKUP_OVERDUE_FACTOR = 2", "BACKUP_OVERDUE_FACTOR = 1"),
    ("H30c age-from-mtime", HI, '        return datetime.datetime.strptime(snap.name[:16] + "+0000", "%Y%m%dT%H%M%SZ%z").timestamp()', "        return snap.stat().st_mtime"),
    ("H31 snapshots-switch-ignored", HI, '            if os.getenv("GHOST_SNAPSHOTS", "1").strip().lower() in ("0", "false", "no", "off"):', "            if False:"),
    ("H32 home-one-level-short", HI, "    home = mdir.parent.parent if mdir is not None else None          # $GHOST_HOME/system/memory", "    home = mdir.parent if mdir is not None else None"),
    ("H33 guard-assumed", HI, "        if egress_guard.is_installed() is True:", "        if True:"),
    ("H34 tor-only-flag-ignored", HI, '        if getattr(getattr(context, "args", None), "mandatory_tor", None) is not True:\n            return "Tor-only mode off"\n', ""),
    ("H35 no-section-timeout", HI, "                await wait_for(asyncio.to_thread(fn, context, out), SECTION_TIMEOUT_S)", "                await asyncio.to_thread(fn, context, out)"),
    ("H35d timeout-value", HI, "SECTION_TIMEOUT_S = 4.0", "SECTION_TIMEOUT_S = 40.0"),
    ("H36 sections-in-a-row", HI, "    got = await asyncio.gather(*(_one(label, fn, threaded) for label, fn, threaded in SECTIONS))", "    got = [await _one(label, fn, threaded) for label, fn, threaded in SECTIONS]"),
    ("H37 section-raise-escapes", HI, '        except Exception as e:  # noqa: BLE001\n            logger.warning("health: section %s failed: %s: %s", label, type(e).__name__, e)', '        except ZeroDivisionError as e:\n            logger.warning("health: section %s failed: %s: %s", label, type(e).__name__, e)'),
    ("H38 quarantine-size-dropped", HI, '                    + (f" ({size / 1048576:.1f} MB)" if recs else ""))', "                    )"),
    ("H40 real-dir-any-path", HI, "    return p if p.is_dir() else None", "    return p"),
    ("H41 unchecked-not-listed", HI, '        text += f". Could not be checked ({len(unchecked)}): " + " | ".join(unchecked)\n', "        pass\n"),
    ("H25c runs-read-error-is-zero", HI, "            live = reg.rows(live_only=True, strict=True)", "            live = reg.rows(live_only=True)"),
    ("H41b unchecked-reads-fine", HI, '    elif unchecked:\n        text = "Problems found: none in what was checked"', '    elif False:\n        text = "Problems found: none in what was checked"'),
    ("H7 unloaded-store-not-flagged", HI, '            self.problems.append(f"{name} is not loaded (it failed at boot)")\n', ""),
    ("H7b no-memory-excuses-every-store", HI, '            self.problems.append(f"{name} is not loaded (it failed at boot)")\n', '            if not _memory_off(context):\n                self.problems.append(f"{name} is not loaded (it failed at boot)")\n'),
    ("H8c wrong-type-accepted", HI, '    return None if isinstance(data, kind) else f"holds a JSON {type(data).__name__}, not a {kind.__name__}"', "    return None"),
    ("H8e empty-file-always-fine", HI, "    if empty_ok and not text.strip():", "    if not text.strip():"),
    ("H35b timeout-is-a-problem", HI, '            return (f"{label}: " + "; ".join(parts + [note]), problems, unchecked +', '            return (f"{label}: " + "; ".join(parts + [note]), problems + [note], unchecked +'),
    ("H35c timeout-drops-what-was-read", HI, "            parts, problems, unchecked = list(out.parts), list(out.problems), list(out.unchecked)", "            parts, problems, unchecked = [], [], list(out.unchecked)"),
    ("H42c registry-link-followed", HI, "        raw = read_bytes_nofollow(sup._registry_path, max_bytes=_REGISTRY_MAX_BYTES)", "        raw = Path(sup._registry_path).read_bytes()"),
    ("H43 node-unchecked-dropped", HI, '                out.unchecked.append(f"model node {name} (no live test was run; last known: {reading})")', "                pass"),
    ("H43b node-is-a-problem", HI, '                out.unchecked.append(f"model node {name} (no live test was run; last known: {reading})")', '                out.problem(f"model node {name} (no live test was run; last known: {reading})")'),
    ("H43c section-unchecked-dropped", HI, '        return (f"{label}: " + ("; ".join(out.parts) or "nothing to report"), list(out.problems),\n                list(out.unchecked))', '        return (f"{label}: " + ("; ".join(out.parts) or "nothing to report"), list(out.problems),\n                [])'),
    ("S4b slow-circuit-is-a-problem", SY, '                unchecked.append("the internet through Tor (not reached this time; the circuit may be slow)")', '                problems.append("the internet through Tor (not reached this time; the circuit may be slow)")'),
    ("S4c slow-circuit-unsaid", SY, '                unchecked.append("the internet through Tor (not reached this time; the circuit may be slow)")', "                pass"),
    ("H13 profile-counts-categories", HI, "sum(len(v) for v in data.values() if isinstance(v, dict))", "len(data)"),
    ("H28 running-jobs-inverted", HI, '        running = sum(1 for r in rows if r.get("state") == STATE_RUNNING)', '        running = sum(1 for r in rows if r.get("state") != STATE_RUNNING)'),
    ("H8d bad-json-accepted", HI, '    except ValueError:\n        return "not readable JSON"\n    except RecursionError:', "    except ValueError:\n        return None\n    except RecursionError:"),
    ("H8g deep-json-escapes", HI, '    except RecursionError:\n        return "nested too deeply to read"\n', ""),
    ("H8h undecodable-file-is-fine", HI, '    except UnicodeDecodeError:\n        return "not readable JSON"\n    except OSError as e:', "    except UnicodeDecodeError:\n        return None\n    except OSError as e:"),
    ("H8i unreadable-file-is-fine", HI, '    except OSError as e:\n        return f"cannot be read: {type(e).__name__}"\n    if empty_ok', "    except OSError as e:\n        return None\n    if empty_ok"),
    ("H8f empty-index-is-damaged", HI, '        why = _damaged(getattr(ms, "library_file", None), list, empty_ok=True, must_exist=True)', '        why = _damaged(getattr(ms, "library_file", None), list, must_exist=True)'),
    ("H9b damaged-index-read-anyway", HI, '        why = _damaged(getattr(ms, "library_file", None), list, empty_ok=True, must_exist=True)', "        why = None"),
    ("H9c index-any-json-type", HI, '        why = _damaged(getattr(ms, "library_file", None), list, empty_ok=True, must_exist=True)', '        why = _damaged(getattr(ms, "library_file", None), object, empty_ok=True, must_exist=True)'),
    ("H9d missing-index-is-fine", HI, '        why = _damaged(getattr(ms, "library_file", None), list, empty_ok=True, must_exist=True)', '        why = _damaged(getattr(ms, "library_file", None), list, empty_ok=True)'),
    ("H44 missing-file-is-fine", HI, "            return _MISSING if must_exist else None", "            return None"),
    ("H44b missing-file-always", HI, "            return _MISSING if must_exist else None", "            return _MISSING"),
    ("H44c missing-not-flagged", HI, '            self.problems.append(f"the {name} file is missing: a reader set a damaged copy aside, or it was "\n                                 "removed — the store runs without it until its next save")\n', ""),
    ("H44d profile-missing-is-fine", HI, '_damaged(getattr(pm, "file_path", None), dict, must_exist=True)', '_damaged(getattr(pm, "file_path", None), dict)'),
    ("H44e profile-any-json-type", HI, '_damaged(getattr(pm, "file_path", None), dict, must_exist=True)', '_damaged(getattr(pm, "file_path", None), object, must_exist=True)'),
    ("H42 job-registry-counted-when-damaged", HI, '        if out.damaged("sandbox job registry", why):\n            return None\n', ""),
    ("H42b service-registry-counted-when-damaged", HI, '        if out.damaged("service registry", why):\n            return None\n', ""),
    ("H42d missing-registry-is-damaged", HI, '        return (None if isinstance(e.__cause__, FileNotFoundError) else "a link, or it cannot be read"), []', '        return "a link, or it cannot be read", []'),
    ("H42e unreadable-registry-is-fine", HI, '        return (None if isinstance(e.__cause__, FileNotFoundError) else "a link, or it cannot be read"), []', "        return None, []"),
    ("H42f registry-any-json-type", HI, "    why = _shape(text, dict)", "    why = _shape(text, object)"),
    ("H42g registry-read-unbounded", HI, "        raw = read_bytes_nofollow(sup._registry_path, max_bytes=_REGISTRY_MAX_BYTES)", "        raw = read_bytes_nofollow(sup._registry_path)"),
    ("H42h registry-strict-decode", HI, '    text = raw.decode("utf-8", "replace")', '    text = raw.decode("utf-8")'),
    ("H42i supervisor-loader-called", HI, '        why, rows = _registry_rows(sup)\n        if out.damaged("sandbox job registry", why):', '        why, rows = None, sup.list_entries()\n        if out.damaged("sandbox job registry", why):'),
    ("H46 status-printed-raw", HI, ' if st.isascii() and len(st) <= 24 and all(c.isalpha() or c == "_" for c in st) else "?"', ' if True else "?"'),
    ("H47 backup-tag-printed", HI, 'said = f"last store backup {_ago(age)}"      # the folder', 'said = f"last store backup {_ago(age)} ({last.name.partition(chr(45))[2]})"  # the folder'),
    ("H48 ledger-hidden-under-no-memory", HI, '        if getattr(context, "fact_ledger", None) is not None:\n            out.store(context, "fact_ledger", "fact ledger",\n', '        if False:\n            out.store(context, "fact_ledger", "fact ledger",\n'),
    ("H9e missing-index-not-flagged", HI, '            out.problem("the document index file is missing (the store creates it when it starts, so it was "\n                        "removed): the catalogue is empty until documents are added again")\n', ""),
    ("H49 non-object-rows-counted", HI, "    return None, [r for r in json.loads(text).values() if isinstance(r, dict)]", "    return None, list(json.loads(text).values())"),
    ("H50 status-length-unbounded", HI, "st.isascii() and len(st) <= 24 and all(", "st.isascii() and all("),
    ("H50b status-non-ascii", HI, "st.isascii() and len(st) <= 24 and all(", "len(st) <= 24 and all("),
    ("H50c status-digits-allowed", HI, 'all(c.isalpha() or c == "_" for c in st)', 'all(c.isalnum() or c in "_ " for c in st)'),
    ("S1 problems-line-second", SY, "    health_status.insert(2, problems_line(problems, unchecked))", "    health_status.insert(1, problems_line(problems, unchecked))"),
    ("S1b problems-line-last", SY, "    health_status.insert(2, problems_line(problems, unchecked))", "    health_status.append(problems_line(problems, unchecked))"),
    ("S1c unchecked-dropped", SY, "    health_status.insert(2, problems_line(problems, unchecked))", "    health_status.insert(2, problems_line(problems))"),
    ("S3 docker-down-not-flagged", SY, '            problems.append("Docker is not running or not found")\n', ""),
    ("S3b docker-check-failed-not-flagged", SY, '        problems.append("Docker could not be checked")\n', ""),
    ("S4 internet-down-not-flagged", SY, '                    health_status.append("Internet: Disconnected or Blocked")\n                    problems.append("the internet was not reached")\n', '                    health_status.append("Internet: Disconnected or Blocked")\n'),
    ("S5 dead-scheduler-not-flagged", SY, '        if bio_status != "Running":', "        if False:"),
    ("S6 no-bus-not-flagged", SY, '        if bus_status != "Active":', "        if False:"),
    ("S7 expired-edges-counted", SY, 'conn.execute("SELECT count(*) FROM triplets WHERE valid_until IS NULL").fetchone()[0]', 'conn.execute("SELECT count(*) FROM triplets").fetchone()[0]'),
    ("S8 internals-lines-dropped", SY, "        health_status.extend(_lines)\n", ""),
    ("S9 internals-problems-dropped", SY, "        problems.extend(_found)\n", ""),
    ("S9b internals-unchecked-dropped", SY, "        unchecked.extend(_unchecked)\n", ""),
    ("S10 tor-failed-not-flagged", SY, '                    problems.append("the Tor check failed")\n', ""),
    ("S10b tor-exhausted-not-flagged", SY, '            problems.append("Tor did not answer (retries exhausted)")\n', ""),
    ("S11 not-anonymous-not-flagged", SY, '                            health_status.append("Tor: Connected but Not Anonymous (Check Config)")\n                            problems.append("Tor answers but the connection is NOT anonymous")\n                        break\n            except Exception as e:', '                            health_status.append("Tor: Connected but Not Anonymous (Check Config)")\n                        break\n            except Exception as e:'),
    ("I1 shared-server-not-merged", IN, '    return [("/".join(ns) + (" (one server)" if len(ns) > 1 else ""), status.get(u))', '    return [("/".join(ns), status.get(u))'),
    ("I2 node-status-lost", IN, '    return [("/".join(ns) + (" (one server)" if len(ns) > 1 else ""), status.get(u))', '    return [("/".join(ns) + (" (one server)" if len(ns) > 1 else ""), None)'),
    ("N1 banner-does-not-wait", MN, "        await _wait_boot_lines()\n", "        pass\n"),
    ("N2 boot-wait-unbounded", MN, "        await asyncio.wait(pending, timeout=BOOT_LINES_WAIT_S if timeout is None else timeout)", "        await asyncio.wait(pending, timeout=None)"),
    ("N3 runs-line-not-waited", MN, "    pending = [t for t in (*_labs._BOOT_TASKS, *_runs._BOOT_TASKS) if not t.done()]", "    pending = [t for t in (*_labs._BOOT_TASKS,) if not t.done()]"),
    ("N3b labs-line-not-waited", MN, "    pending = [t for t in (*_labs._BOOT_TASKS, *_runs._BOOT_TASKS) if not t.done()]", "    pending = [t for t in (*_runs._BOOT_TASKS,) if not t.done()]"),
    ("N4 wait-cancels-the-lines", MN, "        await asyncio.wait(pending, timeout=BOOT_LINES_WAIT_S if timeout is None else timeout)", "        await asyncio.wait(pending, timeout=BOOT_LINES_WAIT_S if timeout is None else timeout)\n        [t.cancel() for t in pending]"),
    ("N5 cancel-swallowed", MN, "        await _wait_boot_lines()\n    except asyncio.CancelledError:\n        raise\n", "        await _wait_boot_lines()\n    except asyncio.CancelledError:\n        pass\n"),
    ("N6 wait-bound-value", MN, "BOOT_LINES_WAIT_S = 12.0", "BOOT_LINES_WAIT_S = 120.0"),
    ("N7 audit-after-banner", MN, "    audit_source_newer_than_process()\n\n    # READY IS ANNOUNCED LAST", "    # READY IS ANNOUNCED LAST"),
    ("L1 labs-task-not-kept", LB, "    t = asyncio.get_running_loop().create_task(labs_boot_line(context))\n    _BOOT_TASKS.add(t)\n", "    t = asyncio.get_running_loop().create_task(labs_boot_line(context))\n"),
    ("L2 runs-task-not-kept", "src/ghost_agent/tools/runs.py", "    t = asyncio.get_running_loop().create_task(runs_boot_line(context))\n    _BOOT_TASKS.add(t)\n", "    t = asyncio.get_running_loop().create_task(runs_boot_line(context))\n"),
]
PINS = ["tests/test_4sy_health_internals.py", "tests/test_boot_ready_ordering.py"]


def _tests():
    pat = (r"health_internals\|tool_check_health\|check_health\|_overview_health\|_known_nodes\|"
           r"_announce_ready_when_warm\|wait_boot_line\|labs_boot_line")
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
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [(lab, originals[f].count(old)) for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if args[0] == "check":
        print("anchors:", "OK" if not bad else bad); sys.exit(1 if bad else 0)
    if bad:
        print("ANCHOR NOT UNIQUE/FOUND:", bad); sys.exit(2)
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
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
