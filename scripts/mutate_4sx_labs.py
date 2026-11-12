#!/usr/bin/env python3
"""Mutation battery for ghost-labs (§4SX, review r1 fixes K1–K11 + core pins).

Each mutant is an exact (file, old, new) replacement that undoes or weakens one fix. A fresh copy
of interface/ghost-labs is made OUTSIDE the repo per mutant (src is never touched), the unit suite
runs there, and the mutant is KILLED when any test fails. Survivors are printed; the score is the
convergence signal. Usage: python3 scripts/mutate_4sx_labs.py [-jN] [--no-shell] [name-substring]

`-jN` runs N mutants at a time (each has its own tree copy). `--no-shell` SKIPS the mutants judged by the
integration test on the real labs host (tagged "shell") instead of ssh-ing there, and lists them. Every mutant is
py_compile'd first (NOCOMP is never a kill). Both controls are always run: the known-bad one must be KILLED and
the no-op must SURVIVE, or the run is void.
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
    # controls (protocol §R2): the known-bad one must DIE and the no-op ("D34 CONTROL no-op", below) must SURVIVE,
    # or the run is void
    ("CONTROL known-bad", L, '        return f"ghost-lab-{lab_id}-{member}"', '        return f"ghost-lab-{member}"'),
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
    ('K11 read_raw dirs allowed', L, '        if mode & (1 << 31) or mode & (1 << 27) or mode & (1 << 26) or mode & (1 << 25):   # dir, symlink, device, pipe\n            # `kind`', '        if False:   # dir, symlink, device, pipe\n            # `kind`'),
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
    ('core extend 0 allowed', L, '    want = as_int(body, "ttl_idle_s", default, 1, 10 ** 9)', '    want = as_int(body, "ttl_idle_s", default, 0, 10 ** 9)'),
    ('core extend not clamped', L, '    return min(want, int(cap)), want > int(cap)', '    return want, want > int(cap)'),
    ("core reconcile drops kept", L, '            if lid not in seen and lab["lifetime"] == "kept":', '            if False:'),
    ('core creating stays', L, '            if lab["state"] == "creating" or (lab["lifetime"] == "kept" and lab["state"] == "destroying"):', '            if False:'),
    ("core destroy_owner stops", L, '            except LabError as e:\n                failed.append(', '            except KeyError as e:\n                failed.append('),
    ('core retire mounted', L, '                if loads(row["app_json"], {}).get("mounted") and str(row.get("origin") or "") != "probe":', '                if False:'),
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
    ('r2#8 keep unlocked', K, '        with self._app_locked(name):\n            return self._keep(lab_id, body, origin)', '        return self._keep(lab_id, body, origin)'),
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
    ('K1 retire_at kept on new', K, '                self.store.update_lab(lab_id, retire_at=self._live_retire(lab_id))\n                self._clear_marks(from_lab)', '                self._clear_marks(from_lab)'),
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
    ('r4#4 switched_from saved after lease', K, '                new.update(switched_from=from_lab, switch_snapshot=snap["backup_id"],\n                           switch_snapshots=[x["backup_id"] for x in snap["members"]])\n                self._save(lab_id, new)\n                self.store.update_lab(from_lab, retire_at=time.time() + 86400, state="stopped")\n                self.store.update_lab(lab_id, retire_at=self._live_retire(lab_id))\n                self._clear_marks(from_lab)\n                lease_err = self._lease_or_note(lab_id, new, "switched")', '                lease_err = self._lease_or_note(lab_id, new, "switched")\n                new.update(switched_from=from_lab, switch_snapshot=snap["backup_id"],\n                           switch_snapshots=[x["backup_id"] for x in snap["members"]])\n                self._save(lab_id, new)\n                self.store.update_lab(from_lab, retire_at=time.time() + 86400, state="stopped")\n                self.store.update_lab(lab_id, retire_at=self._live_retire(lab_id))\n                self._clear_marks(from_lab)'),
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
    ('r5#5 recover keeps retire_at', K, '                    self.store.update_lab(lid, retire_at=self._live_retire(lid))\n', ''),
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
    ('D34 retire keeps secrets', K, '                self.store.x("DELETE FROM secrets WHERE lineage=? AND app=?", (lineage, app["name"]))', '                pass'),
    ('D34 retire keeps lease', K, '                self.store.x("DELETE FROM ports WHERE lease_key=?", (key,))\n                try:\n                    vols +=', '                try:\n                    vols +='),
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
    ('UD Q1 exec no wait for state change', L, '            if lk.acquire(timeout=LOCK_WAIT_S, over_exec=True):\n                lk.release()\n', ''),
    ('UD Q3 detach no guard', L, '            lk = self._acquire_guarded(lab_id, lambda row: self.guard_kept_mutation(row, origin, "exec"),\n                                       over_exec=True)', '            lk = self._acquire_guarded(lab_id, lambda row: None,\n                                       over_exec=True)'),
    ('UD Q5 detach keeps lock', L, '                return self.start_job(lab_id, member, cmd, body)\n            finally:\n                lk.release()', '                return self.start_job(lab_id, member, cmd, body)\n            finally:\n                pass'),
    ('UD G4 member op guard stale', L, '        lk = self._acquire_guarded(lab_id, lambda row: self.guard_kept_mutation(row, origin, f"member {action}"),', '        lk = self._acquire_guarded(lab_id, lambda row: None,'),
    ('UD G5 screenshot guard stale', L, '        lk = self._acquire_guarded(lab_id, lambda r: self.guard_kept_read(r, origin, "pages"), over_exec=False)', '        lk = self._acquire_guarded(lab_id, lambda r: None, over_exec=False)'),
    ('UD G6 exec guard stale', L, '        lk = self._acquire_guarded(lab_id, lambda row: self.guard_kept_mutation(row, origin, "exec"), over_exec=False)', '        lk = self._acquire_guarded(lab_id, lambda row: None, over_exec=False)'),
    ('UD J1 job read unguarded', L, '            self.guard_kept_read(row, reader, "job output")', '            pass'),
    ('UD J2 job start rows outlive lab', L, '            for t in ("jobs", "job_starts"):\n                self.store.x(f"DELETE FROM {t} WHERE job_id=?", (job_id,))\n            self.get(lab_id)', '            pass'),
    ('UD FA2 member fails the create', L, '            self.store.update_lab(lab_id, state="ready" if last["passed"] else "unready", readiness_json=dumps(last))', '            self.store.update_lab(lab_id, state="ready" if (last["passed"] and all(got.values())) else "unready", readiness_json=dumps(last))'),
    ('UD FA5 wait ends on template only', L, '            if last["passed"] and all(got.values()):\n                break', '            if last["passed"]:\n                break'),
    ('UD FA8 detail drops ready', L, '"ready_cmd": ready_cmd.get(m["name"], "")} if m["name"] in ready else {})})', '"ready_cmd": ready_cmd.get(m["name"], "")} if False else {})})'),
    ('UD FB1 cancel never seen', L, '            if cancel is not None and cancel.is_set():', '            if False:'),
    ('UD FB2 stale pid signalled', L, '            if _started_at(c) != started:\n                return None', '            if False:\n                return None'),
    ('UD FB6 anyone may cancel', L, '            return bool(tag) and r["origin"] == "autopilot" and r["tag"] == tag', '            return True'),
    ('UD FB7 empty tag matches', L, '            return bool(tag) and r["origin"] == "autopilot" and r["tag"] == tag', '            return r["origin"] == "autopilot" and r["tag"] == tag'),
    ('UD FB10 cancel no kept guard', L, '        self.guard_kept_mutation(row, origin, "cancel")', '        pass'),
    ('UD FB12 cancel silent refusal', L, '        if running and not mine:', '        if False:'),
    ('UD FC2 tree dotdot ok', L, '        if not parts or any(p == ".." for p in parts):', '        if not parts:'),
    ('UD FC3 tree links ok', L, '                    if not (ti.isfile() or ti.isdir()):\n                        raise LabError("bad_request", f"entry {ti.name!r} is a link or a device', '                    if False:\n                        raise LabError("bad_request", f"entry {ti.name!r} is a link or a device'),
    ('UD FC5 tree no unpacked bound', L, '                    if total > limit:', '                    if False:'),
    ('UD FC8 tree put kept guard', L, '        self.guard_kept_mutation(self.get(lab_id), origin, "put")\n        self.touch(lab_id, origin)\n        member, path = str(body.get("member") or ""), str(body.get("path") or "")\n        self.member(lab_id, member)\n        if not path.startswith("/") or ".."', '        self.get(lab_id)\n        self.touch(lab_id, origin)\n        member, path = str(body.get("member") or ""), str(body.get("path") or "")\n        self.member(lab_id, member)\n        if not path.startswith("/") or ".."'),
    ('UD FC11 tree get keeps links', L, '                if "\\\\" in rel or "\\x00" in rel or not (ti.isfile() or ti.isdir() or linked is not None):', '                if False:'),
    ('UD FC12 tree get no redact', L, '                red = linked if linked is not None else redact.apply_bytes(src.extractfile(ti).read(), rules)', '                red = linked if linked is not None else src.extractfile(ti).read()'),
    ('UD H1a destroy waits', L, 'lk = self._locked(lab_id, None, True)', 'lk = self._locked(lab_id, None, False)'),
    ('UD H1c exec holds the lock (old world)', L, '            while self._held or (self._execs and not over_exec):', '            while self._held or self._execs:'),
    ('UD H2 reaper idle', L, '                if self.exec_running(lid):', '                if False:'),
    ('UD H2b exec end touch', L, "        self.touch(lab_id, origin)          # idle time counts from the command's END", '        pass'),
    ('UD H3a no watchdog', L, '        dog.start()\n', '        pass\n'),
    ('UD H3c body loop unbounded', L, '                chunk = resp.read1(min(65536, want_bytes - len(buf)))', '                conn.sock.settimeout(15); hit.clear(); chunk = resp.read(want_bytes - len(buf))'),
    ('UD H3d screenshot fresh 60', L, '            res = c.wait(timeout=max(1, int(min(60.0, left))))', '            res = c.wait(timeout=60)'),
    ('UD H4a stale ignored', L, '            stale = bool(was and was[0]["container_started"] and was[0]["container_started"] != _started_at(c))', '            stale = False'),
    ('UD H4c signal though ended', L, '            if rc == "" and not stale:', '            if not stale:'),
    ('UD H5 pump re-sum', L, '        while len(self.chunks) > 1 and self.held - len(self.chunks[0]) >= self.keep:', '        while len(self.chunks) > 1 and sum(len(x) for x in self.chunks) - len(self.chunks[0]) >= self.keep:'),
    ('UD H5b pump total wrong', L, '        self.held += len(chunk)\n', '        self.held += 1\n'),
    ('UD H6 kept read guard', L, '        if row["lifetime"] == "kept" and not self.may_touch_kept(row, origin):\n            raise LabError("owner_only", f"lab {row[\'lab_id\']} is kept (production): unattended', '        if False:\n            raise LabError("owner_only", f"lab {row[\'lab_id\']} is kept (production): unattended'),
    ('UD H7 host drops shared data', K, '            sharing = self._other_versions(lab_id, row["lineage"], app["name"])', '            sharing = []'),
    ('UD H7b fresh kept', K, '            if app.get("mounted") and sharing and app["data"] != "fresh":', '            if app.get("mounted") and sharing:'),
    ('UD H8 logs unbounded', L, '        want = cap + redact.margin(pairs)          # fetched', '        want = cap = 10 ** 9          # fetched'),
    ('UD H9a create refuses above max', L, '    return min(want, int(cap)), want > int(cap)', '    return want, False'),
    ('UD H9b loopback never', L, '            return r.exit_code == 0\n        except Exception:  # noqa: BLE001\n            return False', '            return False\n        except Exception:  # noqa: BLE001\n            return False'),
    ('UD H11 job tail flag', L, '"output_truncated": bool(cut), "log_path": f"{d}.log",', '"output_truncated": False, "log_path": f"{d}.log",'),
    ('UD H12 loopback asked without refused', L, '            if r["refused"] and self._listens_on_loopback(lab_id, m["name"], port):', '            if self._listens_on_loopback(lab_id, m["name"], port):'),
    ('UD H14 clamp boundary', L, '    return min(want, int(cap)), want > int(cap)', '    return min(want, int(cap)), want >= int(cap)'),
    ('UD H17 job_starts not cleared', 'ghost_labs/store.py', 'for t in ("labs", "members", "jobs", "job_starts"):', 'for t in ("labs", "members", "jobs"):'),
    ('UD H22 job read: autopilot only', L, '            self.guard_kept_read(row, reader, "job output")', '            if reader == "autopilot":\n                self.guard_kept_read(row, reader, "job output")'),
    ('UD H31 max_expires_at for kept', L, '(float(row["created_at"]) + int(row["ttl_max_s"])) if row["lifetime"] != "kept" else None,', '(float(row["created_at"]) + int(row["ttl_max_s"])),'),
    ('UD H35 features not in health', S, '                "features": list(FEATURES)}', '                }'),
    ('UD H36 reader not passed', 'ghost_labs/api.py', '        return mgr.job(lab_id, job_id, tail_bytes, reader=_origin(x_origin))', '        return mgr.job(lab_id, job_id, tail_bytes)'),
    ('UD H42 post-check keeps rows', L, '            for t in ("jobs", "job_starts"):\n                self.store.x(f"DELETE FROM {t} WHERE job_id=?", (job_id,))\n            self.get(lab_id)', '            self.get(lab_id)'),
    ('UD G1 guard not re-read', L, '            guard(self.get(lab_id))', '            self.get(lab_id)'),
    ('UD G2 lock kept on refusal', L, '            guard(self.get(lab_id))\n        except BaseException:\n            lk.release()\n            raise', '            guard(self.get(lab_id))\n        except ZeroDivisionError:\n            raise'),
    ('UD G3 member lock not released', L, '            m = self.member(lab_id, member)\n        finally:\n            lk.release()', '            m = self.member(lab_id, member)\n        finally:\n            pass'),
    ('UD2 m7 member checks share the turn', L, '            for name, cmd in (want.items() if last["passed"] else ()):', '            for name, cmd in want.items():'),
    ('UD2 m7 a check with no time left runs', L, '                if got[name] or left < 1:', '                if got[name]:'),
    ('UD2 m7 unchecked starts as not ready', L, 'got: Dict[str, Optional[bool]] = {name: None for name in want}', 'got: Dict[str, Optional[bool]] = {name: False for name in want}'),
    ('UD2 m7 detail turns null into false', L, '(None if ready[m["name"]] is None else bool(ready[m["name"]]))', 'bool(ready[m["name"]])'),
    ('UD2 M4 path not normalised', L, '        path = posixpath.normpath(path)', '        path = path'),
    ('UD2 M4 root fetched', L, '        if path == "/" or path.startswith("//"):', '        if False:'),
    ('UD2 M4 any first component', L, '                if not parts or parts[0] != base or ".." in parts:', '                if not parts or ".." in parts:'),
    ('UD2 M4 dotdot entry', L, '                if not parts or parts[0] != base or ".." in parts:', '                if not parts or parts[0] != base:'),
    ('UD2 H21 skipped unbounded', L, '"skipped": skipped[:50], "skipped_total": len(skipped),', '"skipped": skipped, "skipped_total": len(skipped),'),
    ('UD2 H21 skipped total wrong', L, '"skipped": skipped[:50], "skipped_total": len(skipped),', '"skipped": skipped[:50], "skipped_total": len(skipped[:50]),'),
    ('UD2 H28 hardlink sent as a file', L, '                if "\\\\" in rel or "\\x00" in rel or not (ti.isfile() or ti.isdir() or linked is not None):', '                if "\\\\" in rel or "\\x00" in rel or not (ti.isfile() or ti.isdir() or ti.islnk()):'),
    ('UD2 m4 backslash name sent', L, '                if "\\\\" in rel or "\\x00" in rel or not (ti.isfile() or ti.isdir() or linked is not None):', '                if not (ti.isfile() or ti.isdir()):'),
    ('UD2 H26 mode bits kept', L, 'clean.size, clean.mode = len(red), (int(ti.mode) & 0o755) or 0o644', 'clean.size, clean.mode = len(red), int(ti.mode)'),
    ('UD2 m8 unbounded inflate', L, '        out = d.decompress(data, cap + 1)', '        out = d.decompress(data)'),
    ('UD2 m8 plain tar unbounded', L, '        if len(data) > cap:', '        if False:'),
    ('UD2 m8 put reads the archive unbounded', L, '            plain = tar_bytes_bounded(data, limit, slack=2048 * (TREE_MAX_ENTRIES + 2))\n            with tarfile.open(fileobj=io.BytesIO(plain), mode="r:") as src,', '            plain = data\n            with tarfile.open(fileobj=io.BytesIO(plain), mode="r:*") as src,'),
    ('UD2 m1 kill never looked at', L, '            return r.exit_code == 0\n        except Exception:  # noqa: BLE001 — a paused', '            return True\n        except Exception:  # noqa: BLE001 — a paused'),
    ('UD2 m1 failed kill reads as killed', L, 'nothing was confirmed\n            return False', 'nothing was confirmed\n            return True'),
    ('UD2 m1 always killed', L, '            killed = bool(sent) or not th.is_alive()', '            killed = True'),
    ('UD2 m1 mark dropped while it runs', L, '            if pump is not None and pump.is_alive():', '            if False:'),
    ('UD2 m1 cancel guesses killed', L, '"outcome": r["outcome"] if r["ended"].is_set() else ""} for r in mine]}', '"outcome": "killed"} for r in mine]}'),
    ('UD2 m1 cancel does not wait', L, '            r["ended"].wait(max(0.0, end - time.monotonic()))', '            pass'),
    ('UD2 H2 put backslash name', L, 'if raw.startswith("/") or "\\x00" in raw or "\\\\" in raw:', 'if raw.startswith("/") or "\\x00" in raw:'),
    ('UD2 H19 ready command unbounded', 'ghost_labs/templates.py', 'len(ready) > 500', 'len(ready) > 5000'),
    ('UD2 H24 cancel is not a use', L, '        self.touch(lab_id, origin)\n        rules = self.redactor()\n        for r in mine:', '        rules = self.redactor()\n        for r in mine:'),
    ('UD2 m1 second cancel does not retry', L, '                if self._kill_group(*r["kill"]):', '                if False:'),
    ('UD2 m1 retry forgets the mark', L, '                state["kill"] = (c, xid, started)', '                pass'),
    ('UD3 hard link never resolved', L, '                        linked = sent.get("/".join(tparts[1:]))', '                        linked = None'),
    ('UD3 hard link to another folder', L, '                    if tparts and tparts[0] == base:', '                    if tparts:'),
    ('UD3 hard link target not remembered', L, '                sent[rel] = red', '                pass'),
    ('UD3 link bytes re-read unredacted', L, '                red = linked if linked is not None else redact', '                red = redact'),
    ('UD3 etc template wins', 'ghost_labs/templates.py', '        if t["name"] in shipped_names:\n', '        if False:\n'),
    ('UD3 etc templates ignored', 'ghost_labs/templates.py', '            out[t["name"]] = t\n', '            pass\n'),
    ('UD3 shadow silent', 'ghost_labs/service.py', '            log.warning("template file in %s is NOT used', '            log.debug("template file in %s is NOT used'),
    ('UD3 server reads etc', 'ghost_labs/server.py', 'svc = Service(cfg, docker.from_env(timeout=300), SHIPPED_DIR, C.STATE_DIR,', 'svc = Service(cfg, docker.from_env(timeout=300), C.CONFIG_DIR / "templates", C.STATE_DIR,'),
    ('UD3 N1 pid file dropped when not killed', L, '+ ("" if killed is False else f" /run/ghost-exec/{xid}.pid")])', '+ f" /run/ghost-exec/{xid}.pid"])'),
    ('UD3 r4 pid never removed', L, '+ ("" if killed is False else f" /run/ghost-exec/{xid}.pid")])', '+ ""])'),
    ('UD3 r5 host stdin rm dropped', L, 'c.exec_run(["sh", "-c", f"rm -f /run/ghost-exec/{xid}.stdin"\n', 'c.exec_run(["sh", "-c", "true" + f""\n'),
    ('UD3 r5 links any', L, '                        links += linked is not None', '                        links += 1'),
    ('UD3 r4 stdin file left by the shell', L, "rm -f /run/ghost-exec/{xid}.stdin; '", "'"),
    ('UD3 r4 stdin left when exec cannot start', L, '            if stdin is not None:           # nothing started', '            if False:           # nothing started'),
    ('UD3 r3 built bytes unbounded', L, '                if total > built_max:', '                if False:'),
    ('UD3 r4 built bound 4x', L, '                if total > built_max:', '                if total > limit:'),
    ('UD3 r4 links blamed always', L, '+ (" (hard links count as copies)" if links else "")', '+ " (hard links count as copies)"'),
    ('UD3 r3 zero padding refused', L, 'd.unused_data.strip(b"\\0"):', 'd.unused_data:'),
    ('UD3 r3 broken etc file fatal', 'ghost_labs/templates.py', '        except Exception as e:  # noqa: BLE001 — any failure of ONE', '        except (LabError, ValueError) as e:  # noqa: BLE001 — any failure of ONE'),
    ('UD3 r4 etc clash called shipped', 'ghost_labs/templates.py', '        if t["name"] in shipped_names:', '        if t["name"] in out:'),
    ('UD3 N1 no pid file reads killed', L, '2>/dev/null) || exit 2;', '2>/dev/null) || exit 0;'),
    ('UD3 N1 pid file kept after the kill', L, '|| { rm -f $f.pid $f.stdin; "', '|| { "'),
    ('UD3 N3 host cut gzip accepted', L, '    if not d.eof or d.unused_data.strip(b"\\0"):', '    if False:'),
    ('UD3 N3 host second gzip ignored', L, '    if not d.eof or d.unused_data.strip(b"\\0"):', '    if not d.eof:'),
]


REMOTE = "disorder"     # mutants tagged "shell" are judged by tests/integration/test_seed_cmd.py there (real GNU shell)


NO_SHELL = False


def run(m, base):
    name, f, old, new = m[:4]
    shell = len(m) > 4 and m[4] == "shell"
    if shell and NO_SHELL:
        return name, "SKIPPED(shell)"
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
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if f.endswith(".py") and subprocess.run([sys.executable, "-m", "py_compile", p], capture_output=True,
                                            env=env).returncode != 0:
        shutil.rmtree(work)
        return name, "NOCOMP"
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/unit"],
                           cwd=tree, capture_output=True, text=True, timeout=300, env=env)
        killed = r.returncode != 0
        if r.returncode in (4, 5):
            shutil.rmtree(work)
            return name, f"HARNESS(pytest exit {r.returncode})"
    except subprocess.TimeoutExpired:
        killed = True                      # a mutant that wedges the suite is caught by it
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
    global NO_SHELL
    args = sys.argv[1:]
    jobs = 1
    for a in list(args):
        if a.startswith("-j"):
            jobs = int(a[2:]); args.remove(a)
        elif a == "--no-shell":
            NO_SHELL = True; args.remove(a)
    sel = args[0] if args else ""
    base = tempfile.mkdtemp(prefix="battery4sx-")         # outside the repo (TMPDIR decides where)
    ms = [m for m in MUTANTS if sel in m[0] or "CONTROL" in m[0]]
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(lambda m: run(m, base), ms))
    shutil.rmtree(base, ignore_errors=True)
    for name, st in res:
        print(f"{st:14s} {name}")
    by = dict(res)
    ctl_ok = by.get("CONTROL known-bad") == "KILLED" and by.get("D34 CONTROL no-op") == "SURVIVED"
    judged = [(n, st) for n, st in res if "CONTROL" not in n and not st.startswith("SKIPPED")]
    killed = sum(st == "KILLED" for _, st in judged)
    print(f"\ncontrols: known-bad {by.get('CONTROL known-bad')}, no-op {by.get('D34 CONTROL no-op')}"
          f" -> {'OK' if ctl_ok else 'VOID RUN'}")
    print(f"score {killed}/{len(judged)}  skipped(shell) {sum(st.startswith('SKIPPED') for _, st in res)}")
    return 0 if ctl_ok and killed == len(judged) else 1


if __name__ == "__main__":
    sys.exit(main())
