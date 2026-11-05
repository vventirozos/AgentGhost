#!/usr/bin/env python3
"""Mutation battery for ghost-labs (§4SX, review r1 fixes K1–K11 + core pins).

Each mutant is an exact (file, old, new) replacement that undoes or weakens one fix. A fresh copy
of interface/ghost-labs is made OUTSIDE the repo per mutant (src is never touched), the unit suite
runs there, and the mutant is KILLED when any test fails. Survivors are printed; the score is the
convergence signal. Usage: python3 scripts/mutate_4sx_labs.py [name-substring]
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "interface", "ghost-labs")
K, L, S = "ghost_labs/kept.py", "ghost_labs/labs.py", "ghost_labs/service.py"

MUTANTS = [
    # K1
    # K2
    ("K2 switch no reverse", K, 'back = self._reverse(from_lab, lab_id, old_last, f"switch to {lab_id} failed")',
     'back = {"ok": True}'),
    ("K2 switch no live archive check", K, "        if not self._archive_for(from_lab):\n            raise LabError(\"not_ready\", \"the live version has no deploy archive",
     "        if False:\n            raise LabError(\"not_ready\", \"the live version has no deploy archive"),
    ("K2 rollback no archive precheck", K, '        if not self._archive_for(from_lab):\n            raise LabError("no_rollback"',
     '        if False:\n            raise LabError("no_rollback"'),
    ("K2 rollback no reverse", K, 'back = self._reverse(lab_id, from_lab, new_last, f"rollback to {from_lab} failed")',
     'back = {"ok": False}'),
    ("K2 rehearse error not recorded", K, '''                    self._save(lab_id, app)
                    raise
                self._save(lab_id, app)
            finally:''', '''                    raise
                self._save(lab_id, app)
            finally:'''),
    ("K2 rehearse state stuck", K, '''            finally:
                self.store.update_lab(lab_id, state="ready")
        twin = self._twin''', '''            finally:
                pass
        twin = self._twin'''),
    # K3
    ("K3 fresh second version allowed", K, '            if data != "shared":\n                raise LabError("bad_request", f"{name} already runs live',
     '            if False:\n                raise LabError("bad_request", f"{name} already runs live'),
    ("K3 other member allowed", K, '            if member != oapp.get("member"):', '            if False:'),
    ("K3 staged only with volume", K, "            staged = bool(other)\n", "            staged = bool(other) and self._volume_exists(data_volume_name(lineage, name, member))\n"),
    ("K3 live holder ignores mounted", K, '                if o.get("name") == app and o.get("mounted"):\n                    return l["lab_id"]\n        return None\n\n    def _volumes_for',
     '                if o.get("name") == app and o.get("mounted") and o.get("data") == "shared" and False:\n                    return l["lab_id"]\n        return None\n\n    def _volumes_for'),
    ("K3 lease guard off", K, '                if hr and hr["app_json"] and loads(hr["app_json"]).get("mounted"):',
     '                if False:'),
    # K4
    ("K4 start_all not isolated", K, '            except Exception:  # noqa: BLE001\n                log.exception("start of kept app in %s failed (%s)", row["lab_id"], reason)',
     '            except LabError:\n                pass'),
    # on_member_start re-check
    ("on_member_start no recheck", K, "            if not self._startable(app, member) or self._alive(lab_id, app):",
     "            if self._alive(lab_id, app):"),
    # restart is not a crash
    ("exited always crash", K, "        if self._restarted_since(lab_id, app):\n            self.mgr.apply_route",
     "        if False:\n            self.mgr.apply_route"),
    ("docker_ts drops fraction", K, '        frac = float("0." + f) if f.isdigit() else 0.0', "        frac = 0.0"),
    # K5
    ("K5 prune ignores pins", K, "            if float(r[\"ts\"]) in pinned:\n                continue", "            pass"),
    ("K5 archive names collide", K, 'p = d / f"{int(ts * 1000)}-{uuid.uuid4().hex[:8]}.tgz"', 'p = d / f"{int(ts * 1000)}.tgz"'),
    # K6
    ("K6 partial file kept", K, "        except Exception:\n            path.unlink(missing_ok=True)\n            raise\n        return {\"backup_id\": bid",
     "        except Exception:\n            raise\n        return {\"backup_id\": bid"),
    ("K6 no retry gating", K, "                        and now - float(app.get(\"last_backup_attempt_at\") or 0) > BACKUP_RETRY_S:",
     "                        and True:"),
    ("K6 attempt not saved", K, '                    app["last_backup_attempt_at"] = now\n', ""),
    # K7
    ("K7 marker always rewritten", K, "        if paths and sorted(marked) != sorted(paths):", "        if paths:"),
    # K8
    ("K8 workdir unchecked", K, "        if not valid_workdir(workdir):", "        if False:"),
    ("K8 /data allowed", K, '"/sbin", "/sys", "/usr", "/var", "/data", "/tmp")', '"/sbin", "/sys", "/usr", "/var", "/tmp")'),
    ("K8 dotdot allowed", K, '    if not WORKDIR_RE.match(wd) or any(p in (".", "..") for p in wd.split("/")):', "    if not WORKDIR_RE.match(wd):"),
    ("K8 venv ancestor allowed", K, '    if wd in WORKDIR_DENY_EXACT or wd.startswith("/opt/venv/"):', "    if False:"),
    # K9
    # K10
    ("K10 restore into unmounted", K, '''        if not app.get("mounted"):
            raise LabError("not_ready", "this version has no /data mounted (staged)", state=app.get("state"),''',
     '''        if False:
            raise LabError("not_ready", "this version has no /data mounted (staged)", state=app.get("state"),'''),
    ("K10 put_archive result ignored", K, "                if not self.mgr.container(lab_id, m).put_archive(\"/\", f):",
     "                if not self.mgr.container(lab_id, m).put_archive(\"/\", f) and False:"),
    # K11
    ("K11 read_raw size unchecked", L, '        if int(stat.get("size", 0)) > limit:', "        if False:"),
    ("K11 read_raw stream unbounded", L, "            if len(buf) > cap:", "            if False:"),
    ("K11 read_raw dirs allowed", L, "        if mode & (1 << 31) or mode & (1 << 27) or mode & (1 << 26) or mode & (1 << 25):   # dir, symlink, device, pipe\n            raise LabError(\"bad_request\", f\"{path} is not a regular file\", field=\"path\")\n        if int(stat.get(\"size\", 0)) > limit:",
     "        if False:\n            raise LabError(\"bad_request\", f\"{path} is not a regular file\", field=\"path\")\n        if int(stat.get(\"size\", 0)) > limit:"),
    ("deploy b64 lenient", K, 'data = base64.b64decode(str(body["archive_b64"]), validate=True)', 'data = base64.b64decode(str(body["archive_b64"]))'),
    # twin
    ("twin real secrets", K, '            tapp = {**app, "lineage": tlin,', '            tapp = {**app, "lineage": app["lineage"],'),
    ("twin keeps cron", K, '"data_seed": [], "cron": [],\n', '"data_seed": [],\n'),
    ("twin secrets left", K, '            self.store.x("DELETE FROM secrets WHERE lineage=?", (tlin,))\n', ""),
    # sidecar rule
    ("sidecar rule dropped while shared", K, '        if any(l["lab_id"] != lab_id and l["agent_api_url"] == url for l in self.store.labs()):\n            return',
     "        pass"),
    # core pins
    ("core preemption evicts when infeasible", L, "                auto = []        # evicting everything still wouldn't fit: evict nothing", "                pass"),
    ("core extend 0 allowed", L, '        want = as_int(body, "ttl_idle_s", -1, 1, 10 ** 9)', '        want = as_int(body, "ttl_idle_s", -1, 0, 10 ** 9)'),
    ("core extend not clamped", L, "        self.store.update_lab(lab_id, ttl_idle_s=min(want, cap))", "        self.store.update_lab(lab_id, ttl_idle_s=want)"),
    ("core reconcile drops kept", L, '            if lid not in seen and lab["lifetime"] == "kept":', '            if False:'),
    ("core creating stays", L, '            if lab["state"] == "creating":\n                # the build finished', "            if False:\n                # the build finished"),
    ("core destroy_owner stops", L, '            except LabError as e:\n                failed.append(', '            except KeyError as e:\n                failed.append('),
    ("core retire mounted", L, '                if loads(row["app_json"], {}).get("mounted"):', "                if False:"),
    # ---- anchors refreshed after review r2
    ("K2 rollback no snapshot precheck", K, '                if not r or not Path(r[0]["path"]).exists():\n                    raise LabError("unknown_backup"',
     '                if False:\n                    raise LabError("unknown_backup"'),
    ("K9 final only app member", K, '            [m] + ([x["name"] for x in self.store.members(lab_id) if x["name"] != m] if all_members else [])',
     "            [m]"),
    ("K9 drop_data partial", K, 'final = self._backup_locked(lab_id, "final", all_members=True, restart=False)',
     'final = self._backup_locked(lab_id, "final", restart=False)'),
    ("K10 no pre-restore snapshot", K, '        pre = self._backup_locked(lab_id, "pre_restore", only_member=m)["backup_id"] if snapshot else None',
     "        pre = None"),
    ("K10 failure not marked", K, '            self._save(lab_id, app)\n            if isinstance(e, LabError):\n                e.extra["pre_restore"] = pre',
     '            if isinstance(e, LabError):\n                e.extra["pre_restore"] = pre'),
    # ---- review r2
    ("r2#1 nested seeds allowed", K, '            if a != b and b.startswith(a + "/"):', "            if False:"),
    ("r2#1 empty parts allowed", K, '        if not SEED_RE.match(p) or any(x in ("", ".", "..") for x in parts) or p == ".ghost-seeded":',
     "        if not SEED_RE.match(p):"),
    ("r2#2 unmounted drops volumes", K, '                note = "this version had no data mounted; data volumes kept"',
     '                note = "x"\n                vols.append(data_volume_name(row["lineage"], app["name"], app["member"]))'),
    ("r2#3 restore not undone", K, "                for pre in reversed([p for p in pres if p]):", "                for pre in []:"),
    ("r2#3 pre id not carried", K, '                e.extra["pre_restore"] = pre\n', ""),
    ("r2#4 archive seeds dropped", K, '                    loads(r[0]["data_seed_json"], []) or []', "                    []"),
    ("r2#5 switch from fresh", K, '        if old.get("data") != "shared":\n            raise LabError("bad_request", "the live version uses data: fresh',
     '        if False:\n            raise LabError("bad_request", "the live version uses data: fresh'),
    ("r2#5 keep behind fresh", K, '            if oapp.get("data") != "shared":\n                raise LabError("bad_request", f"the live version in {other} keeps',
     '            if False:\n                raise LabError("bad_request", f"the live version in {other} keeps'),
    ("r2#6 reconcile keeps switching", L, '            if lab["state"] in ("switching", "rehearsing"):', "            if False:"),
    ("r2#7 cross-project name", K, '            if l["lifetime"] == "kept" and l["lineage"] != lineage and l["app_json"] \\',
     '            if False and l["lineage"] != lineage and l["app_json"] \\'),
    ("r2#8 keep unlocked", K, "        with self._app_locked(name):\n            return self._keep(lab_id, body)", "        return self._keep(lab_id, body)"),
    ("r2#8 app lock never blocks", K, "        if not lk.acquire(timeout=timeout):", "        if not lk.acquire(blocking=False) and False:"),
    ("r2#9 snapshot app member", K, '"pre_restore", only_member=m)', '"pre_restore")'),
    ("r2#11 member down ignored", K, "                if not running:", "                if False:"),
    ("r2#11 no restart after down", K, '                if st == "stopped" and app.get("last_error") == MEMBER_DOWN and not app.get("stopped_by_owner"):',
     "                if False:"),
    ("r2#12 final restarts app", K, 'final = self._backup_locked(lab_id, "final", all_members=True, restart=False)',
     'final = self._backup_locked(lab_id, "final", all_members=True)'),
    ("r2#12 restart flag ignored", K, "            if not hook and was_running and restart:", "            if not hook and was_running:"),
    ("r2#13 switch snapshot app only", K, 'snap = self._backup_locked(from_lab, "pre_switch", all_members=True)', 'snap = self._backup_locked(from_lab, "pre_switch")'),
    ("r2#13 snapshot order", K, "        return [i for i in ids if i != main] + ([main] if main in ids else [])", "        return ids"),
    ("core destroy _have_lock ignored", L, "        if not _have_lock:\n            lk.__enter__()", "        lk.__enter__()"),
    ("K2 switch state not reset", K, '            self._clear_marks(lab_id, from_lab)\n            self.store.update_lab(lab_id, state="ready")\n\n    def _archive_for',
     '            self._clear_marks(lab_id, from_lab)\n\n    def _archive_for'),
    ("exited counts the restart", K, '            self._start_app(lab_id, app, count=False)\n            self.mgr.actions.write(lab_id, "app_start_after_member_restart"',
     '            self._start_app(lab_id, app, count=True)\n            self.mgr.actions.write(lab_id, "app_start_after_member_restart"'),
    ("r2#6 marker not cleared", K, '                if a.pop("moving_to", None) is not None:\n                    self._save(lab, a)', "                pass"),
    ("r2#10 post-swap error silent", K, '            app.update(state="stopped" if app.get("mounted") else "staged",\n                       last_error=f"deploy failed after the swap: {_msg(e)[:300]}")\n', ""),
    # ---- review r3
    ("r3#1 reverse failure keeps marker", K, '            # "recover" onto the version that failed (r3 #1)\n            self._clear_marks(live_lab, other)',
     '            # "recover" onto the version that failed (r3 #1)\n            pass'),
    ("r3#1 switch finally keeps marker", K, '        finally:\n            self._clear_marks(lab_id, from_lab)\n            self.store.update_lab(lab_id, state="ready")\n\n    def _archive_for',
     '        finally:\n            self.store.update_lab(lab_id, state="ready")\n\n    def _archive_for'),
    ("r3#2 undo failure still reverses", K, "                if note:\n                    # /data is wiped or half-restored", "                if False:\n                    # /data is wiped or half-restored"),
    ("r3#4 staged failure becomes stopped", K, 'app.update(state="stopped" if app.get("mounted") else "staged",\n                       last_error=f"deploy failed after',
     'app.update(state="stopped",\n                       last_error=f"deploy failed after'),
    ("r3#6 final with hook keeps writing", K, "            if not restart:\n                self._stop_proc(lab_id, app)          # the final backup", "            if False:\n                self._stop_proc(lab_id, app)          # the final backup"),
    ("r3#7 tick overwrites diagnosis", K, '                    if st in ("up", "starting", "restarting"):\n                        app.update(state="stopped", last_error=MEMBER_DOWN)',
     '                    if True:\n                        app.update(state="stopped", last_error=MEMBER_DOWN)'),
    ("r3#9 staged installs cron", K, '            if app.get("mounted"):\n                self._install_cron(lab_id, app)       # a staged version', '            if True:\n                self._install_cron(lab_id, app)       # a staged version'),
    ("r3#12 adopted row KeyError", K, '        if not a.get("name"):\n            # an adopted row', '        if False:\n            # an adopted row'),
    ("K1 retire_at kept on new", K, '                self.store.update_lab(lab_id, retire_at=None)\n                self._clear_marks(from_lab)',
     '                self._clear_marks(from_lab)'),
    ("K2 switch raise not caught", K, '            except Exception as e:  # noqa: BLE001\n                res, err = {"ok": False, "error": _msg(e)}, _msg(e)\n            if res.get("ok"):\n                # the switch is done',
     '            except LabError:\n                raise\n            if res.get("ok"):\n                # the switch is done'),
    ("K7 marker not read back", K, '            known = sorted(marked or app.get("data_seed") or [])', '            known = sorted(app.get("data_seed") or [])'),
    ("r2#4 staged seed dropped", K, '            app["data_seed"] = check_seeds(seed)', "            pass"),
    # ---- seed shell, judged in the real image on disorder
    ("r2#1 link guard off", K, 'f"case \\"$d/\\" in \\"$w\\"/*) ;; *) echo \\"unsafe link parent $d\\" >&2; exit 9;; esac; "', 'f""', "shell"),
    ("K7 seed overwrites data", K, 'f"if [ ! -e {dp} ]; then if [ -e {ap} ]; then mv {ap} {dp}; else "\n            f"echo \'seed path is in neither the archive nor /data\' >&2; exit 8; fi; fi; "',
     'f"if [ -e {ap} ]; then rm -rf {dp}; mv {ap} {dp}; fi; "', "shell"),
    ("r3#10 guard after mkdir", K, 'f"case \\"$d/\\" in \\"$w\\"/*) ;; *) echo \\"unsafe link parent $d\\" >&2; exit 9;; esac; "\n            f"if [ -L {ap} ]; then echo \'seed path is a symlink in the archive\' >&2; exit 9; fi; "\n            f"mkdir -p $(dirname {ap}) $(dirname {dp}); "',
     'f"mkdir -p $(dirname {ap}) $(dirname {dp}); "\n            f"case \\"$d/\\" in \\"$w\\"/*) ;; *) echo \\"unsafe link parent $d\\" >&2; exit 9;; esac; "\n            f"if [ -L {ap} ]; then echo \'seed path is a symlink in the archive\' >&2; exit 9; fi; "', "shell"),
    ("r4#6 leaf symlink allowed", K, 'f"if [ -L {ap} ]; then echo \'seed path is a symlink in the archive\' >&2; exit 9; fi; "', 'f""', "shell"),
    ("r4#6 readlink -f parent", K, 'd=$(readlink -m $(dirname {ap}))', 'd=$(readlink -f $(dirname {ap}))', "shell"),
    ("r4#7 missing path silent", K, "echo 'seed path is in neither the archive nor /data' >&2; exit 8;", "true;", "shell"),
    # ---- review r4
    ("r4#1 mount takes the other's seeds", K, '        to_app.update(mounted=True, state="stopped", data_seed=to_app.get("data_seed") or list(seeds or []))',
     '        to_app.update(mounted=True, state="stopped", data_seed=sorted(set(to_app.get("data_seed") or []) | set(seeds or [])))'),
    ("r4#1 switch seed precheck off", K, '        if new.get("data_seed") and old.get("data_seed") and sorted(new["data_seed"]) != sorted(old["data_seed"]):',
     "        if False:"),
    ("r4#1 staged seed not fixed", K, "            if fixed and sorted(seed) != sorted(fixed):", "            if False:"),
    ("r4#2 only one side told", K, "            for lid in (lab_id, to):", "            for lid in (lab_id,):"),
    ("r4#2 no switch back when none live", K, '        if not old.get("mounted") and self._live_holder(rn["lineage"], new["name"], lab_id):',
     '        if not old.get("mounted"):'),
    ("r4#2 unmounted from_lab with a live one", K, '        if not old.get("mounted") and self._live_holder(rn["lineage"], new["name"], lab_id):\n            raise',
     '        if False:\n            raise'),
    ("r4#3 undo failure not held", K, 'o.update(state="stopped", stopped_by_owner=True, last_error=f"rollback failed', 'o.update(state="stopped", last_error=f"rollback failed'),
    ("r4#4 lease error escapes", K, "        except Exception as e:  # noqa: BLE001\n            log.exception(\"lease move to %s\", lab_id)",
     "        except KeyError as e:  # noqa: BLE001\n            log.exception(\"lease move to %s\", lab_id)"),
    ("r4#4 switched_from saved after lease", K, '''                new.update(switched_from=from_lab, switch_snapshot=snap["backup_id"],
                           switch_snapshots=[x["backup_id"] for x in snap["members"]])
                self._save(lab_id, new)
                self.store.update_lab(from_lab, retire_at=time.time() + 86400, state="stopped")
                self.store.update_lab(lab_id, retire_at=None)
                self._clear_marks(from_lab)
                lease_err = self._lease_or_note(lab_id, new, "switched")''', '''                lease_err = self._lease_or_note(lab_id, new, "switched")
                new.update(switched_from=from_lab, switch_snapshot=snap["backup_id"],
                           switch_snapshots=[x["backup_id"] for x in snap["members"]])
                self._save(lab_id, new)
                self.store.update_lab(from_lab, retire_at=time.time() + 86400, state="stopped")
                self.store.update_lab(lab_id, retire_at=None)
                self._clear_marks(from_lab)'''),
    # ---- review r5
    ("r5#1 recovery switch snapshots unmounted", K, '                if self._app(from_lab).get("mounted") else {"backup_id": None, "members": []}',
     '                if True else {"backup_id": None, "members": []}'),
    ("r5#2 hold saved after clearing", K, '''        app.update(state="stopped", stopped_by_owner=True,
                   last_error=f"restore of {backup_id} in progress (pre-restore snapshot {pre})")
        self._save(lab_id, app)
''', ""),
    ("r5#2 hold never released", K, '''        res: Dict[str, Any] = {"ok": None}
        app["last_error"] = None
''', '''        res: Dict[str, Any] = {"ok": None}
'''),
    ("r5#3 tick no ensure_leases", K, "        try:\n            self.ensure_leases()\n        except Exception:  # noqa: BLE001\n            log.exception(\"ensure_leases\")\n        if now - self._last_daily",
     "        if now - self._last_daily"),
    ("r5#4 recover leaves app running", K, "                    try:\n                        self._stop_proc(lid, a)\n                    except Exception:  # noqa: BLE001\n                        pass\n                    a.pop",
     "                    a.pop"),
    ("r5#5 recover keeps retire_at", K, "                    self.store.update_lab(lid, retire_at=None)\n", ""),
    ("r5#6 seed precheck off", K, "                if r[\"exit_code\"] != 0:\n                    raise LabError(\"bad_request\", f\"data_seed path {p!r} is in neither",
     "                if False:\n                    raise LabError(\"bad_request\", f\"data_seed path {p!r} is in neither"),
    # ---- review r6
    ("r6#1 pending ignored", K, 'r[0]["member"] == app["member"] and not app.get("lease_pending"):', 'r[0]["member"] == app["member"]:'),
    ("r6#1 pending not set", K, '            app["lease_pending"] = True          # ensure_leases retries it', '            pass          # ensure_leases retries it'),
    ("r6#1 pending never cleared", K, '                    a.pop("lease_pending", None)\n                    a.pop("lease_retry_at", None)', '                    a.pop("lease_retry_at", None)'),
    ("r6#4 no throttle", K, '            if time.time() < float(app.get("lease_retry_at") or 0):\n                continue', '            pass'),
    ("r6#2 restore diagnosis dropped", K, '                    keep = f" | before the restart: {prev[:300]}" if "restore" in prev else ""', '                    keep = ""'),
    ("r6#3 recovery switch reverses", K, '            if snap["backup_id"] is None:\n                # a recovery switch', '            if False:\n                # a recovery switch'),
    ("r5#3 ensure_leases skips moved rows", K, 'if r and r[0]["lab_id"] == lab_id and r[0]["member"] == app["member"] and not app.get("lease_pending"):',
     'if r and not app.get("lease_pending"):'),
    ("r5#4 recover leaves target mounted", K, '                    a.update(mounted=False, state="staged", last_error=msg + keep)', '                    a.update(last_error=msg + keep)'),
    ("r4#2 recovery forgets marker", K, '                    a.pop("moving_to", None)\n                    prev = a.get("last_error")', '                    prev = a.get("last_error")'),
    # ---- §4SX-C job group (judged by tests/integration/test_job_script.py on disorder)
    ("D repeat destroy hides leftovers", L, "            if left:\n                return dict(zero, leftovers=left)", "            if False:\n                return dict(zero, leftovers=left)"),
    ("D leftovers only within memory", L, "            left = self.leftovers(lab_id)\n            if left:", "            left = self.leftovers(lab_id) if self.store.destroyed(lab_id, self.cfg.destroyed_memory_s) else []\n            if left:"),
    ("C job no group TERM", L, "echo $r > {d}.rc.tmp; kill -TERM 0 2>/dev/null; sleep 2; ", "echo $r > {d}.rc.tmp; sleep 2; ", "shell", "tests/integration/test_job_script.py"),
    ("C job no KILL", L, "mv {d}.rc.tmp {d}.rc; kill -KILL 0\\'", "mv {d}.rc.tmp {d}.rc\\'", "shell", "tests/integration/test_job_script.py"),
    ("C job rc before cleanup", L, "echo $r > {d}.rc.tmp; kill", "echo $r > {d}.rc; echo $r > {d}.rc.tmp; kill", "shell", "tests/integration/test_job_script.py"),
    ("C job leader not TERM-proof", L, "trap \"\" TERM; echo $r", "echo $r", "shell", "tests/integration/test_job_script.py"),
    ("C job exit code lost", L, "2>&1; r=$?; ", "2>&1; r=0; ", "shell", "tests/integration/test_job_script.py"),
    ("C job not its own group", L, "f'setsid bash -c \\'c=", "f'bash -c \\'c=", "shell", "tests/integration/test_job_script.py"),
    ("C start_job ignores job_script", L, '        script = job_script(d, body.get("workdir"))', '        script = job_script(d, None)'),
    # agent §4SX D3/D4 host changes (run: python3 scripts/mutate_4sx_labs.py D34)
    ("D34 CONTROL no-op", K, "    def _live_retire(self, lab_id: str):", "    def _live_retire(self, lab_id: str):  # no-op"),
    ("D34 lineage join any kept", K, "                    and (l[\"owner\"] != row[\"owner\"] or not self.mgr.may_touch_kept(l, origin))):",
     "                    and False):"),
    ("D34 lineage join other project ok", K, "                    and (l[\"owner\"] != row[\"owner\"] or not self.mgr.may_touch_kept(l, origin))):",
     "                    and (not self.mgr.may_touch_kept(l, origin))):"),
    ("D34 live retire none", K, "        return time.time() + 86400 if r and str(r[\"origin\"] or \"\") == \"probe\" else None",
     "        return None"),
    ("D34 retire keeps secrets", K, "        if lineage and not others:\n            self.store.x(\"DELETE FROM secrets WHERE lineage=?\", (lineage,))",
     "        if False:\n            self.store.x(\"DELETE FROM secrets WHERE lineage=?\", (lineage,))"),
    ("D34 retire keeps lease", K, "            self.store.x(\"DELETE FROM ports WHERE lease_key=?\", (key,))\n            if self.cfg.ts_services:\n                try:\n                    self.mgr.ports.unserve_service(app[\"name\"])\n                except Exception:  # noqa: BLE001\n                    log.exception(\"unserve %s\", app[\"name\"])\n            try:\n                vols +=",
     "            if self.cfg.ts_services:\n                try:\n                    self.mgr.ports.unserve_service(app[\"name\"])\n                except Exception:  # noqa: BLE001\n                    log.exception(\"unserve %s\", app[\"name\"])\n            try:\n                vols +="),
    ("D34 retire fresh vol kept", K, "        vols = [data_volume_name(lineage, app[\"name\"], m, lab_id) for m in members] if app.get(\"name\") and fresh else []",
     "        vols = []"),
    ("D34 retire shared vol kept", K, "                vols += [v.name for v in self.docker.volumes.list(", "                vols += [] and [v.name for v in self.docker.volumes.list("),
    ("D34 retire vols any lineage", K, "                    f\"ghost.data.lineage={lineage}\", f\"ghost.data.app={app['name']}\"]}) if v.name not in vols]",
     "                    f\"ghost.data.app={app['name']}\"]}) if v.name not in vols]"),
    ("D34 retire no re-check", K, "        if ra is None or float(ra) > time.time():\n            return {\"lab_id\": lab_id, \"skipped\"",
     "        if False:\n            return {\"lab_id\": lab_id, \"skipped\""),
    ("D34 retire waits long", K, "        with self._app_locked(app[\"name\"], timeout=5.0):\n            return self._retire_probe_locked(lab_id)",
     "        with self._app_locked(app[\"name\"]):\n            return self._retire_probe_locked(lab_id)"),
    ("D34 reaper counts skipped", L, "                        if self.kept.retire_probe_lab(lid).get(\"skipped\"):\n                            continue",
     "                        self.kept.retire_probe_lab(lid)"),
    ("D34 reaper silent on refusal", L, "                        self._reap_warned.add((lid, e.code))\n                        log.warning(", "                        self._reap_warned.add((lid, e.code))\n                        log.debug("),
    ("D34 reaper warns every tick", L, "                    elif (lid, e.code) not in self._reap_warned:", "                    elif True:"),
    ("D34 health no floor", "ghost_labs/service.py", "                \"disk_floor_mb\": int(self.cfg.disk_floor_mb),", ""),
    ("D34 reaper plain destroy", L, "                    if reason == \"retired\" and str(lab[\"origin\"] or \"\") == \"probe\" and getattr(self, \"kept\", None):",
     "                    if False:"),
    ("D34 reaper stops on error", L, "                except Exception:  # noqa: BLE001 — agent §4SX D4 r2 #2: one lab never stops the tick\n                    log.exception(\"reap %s (%s)\", lid, reason)",
     "                except ZeroDivisionError:\n                    log.exception(\"reap %s (%s)\", lid, reason)"),
    ("D34 switch from_lab unchecked", "ghost_labs/api.py", "            if o != \"owner\" and fl and not mgr.may_touch_kept(mgr.get(fl), o):",
     "            if False:"),
    ("D34 rollback prev unchecked", "ghost_labs/api.py", "            if o != \"owner\" and prev and mgr.store.lab(prev) and not mgr.may_touch_kept(mgr.get(prev), o):",
     "            if False:"),
    ("D34 no app_mounted", L, "                **({\"app_mounted\": bool((loads(row[\"app_json\"], None) or {}).get(\"mounted\"))}",
     "                **({\"app_mounted\": False}"),
]


REMOTE = "disorder"     # mutants tagged "shell" are judged by tests/integration/test_seed_cmd.py there (real GNU shell)


def run(m, base):
    name, f, old, new = m[:4]
    shell = len(m) > 4 and m[4] == "shell"
    itest = m[5] if len(m) > 5 else "tests/integration/test_seed_cmd.py"
    work = tempfile.mkdtemp(prefix="mut4sx-", dir=base)
    tree = os.path.join(work, "gl")
    shutil.copytree(SRC, tree, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "venv"))
    p = os.path.join(tree, f)
    s = open(p).read()
    n = s.count(old)
    if n != 1:
        shutil.rmtree(work)
        return name, f"BAD-ANCHOR({n})"
    open(p, "w").write(s.replace(old, new))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/unit"],
                       cwd=tree, capture_output=True, text=True, timeout=300)
    killed = r.returncode != 0
    if not killed and shell:
        dst = f"/tmp/mut4sx-{os.getpid()}-{os.path.basename(work)}"   # one per mutant; sudo pytest leaves root files
        subprocess.run(["rsync", "-a", "--delete", tree + "/", f"{REMOTE}:{dst}/"], check=True, capture_output=True)
        r = subprocess.run(["ssh", REMOTE, f"cd {dst} && sudo /opt/ghost-labs/venv/bin/python -m pytest -q -x -p no:cacheprovider "
                            f"-m integration {itest}; rc=$?; sudo rm -rf {dst}; exit $rc"],
                           capture_output=True, text=True, timeout=600)
        killed = r.returncode != 0
    shutil.rmtree(work)
    return name, "KILLED" if killed else "SURVIVED"


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else ""
    base = tempfile.mkdtemp(prefix="battery4sx-")         # outside the repo
    ms = [m for m in MUTANTS if sel in m[0]]
    res = [run(m, base) for m in ms]
    shutil.rmtree(base, ignore_errors=True)
    for name, st in res:
        print(f"{st:14s} {name}")
    killed = sum(st == "KILLED" for _, st in res)
    print(f"\nscore {killed}/{len(res)}")
    return 0 if killed == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
