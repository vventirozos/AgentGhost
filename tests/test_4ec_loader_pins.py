"""§4EC — loader survivors from the §R re-verification of §4DE (2026-09-02).

Each pin names the mutant that survived `tests/test_gepa_epoch_swap.py` AND
the 45-file wide tier with every test green, and the world in which it fails.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R

REPO = Path(__file__).resolve().parents[1]


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "system" / "optim").mkdir(parents=True)
    monkeypatch.setenv("GHOST_HOME", str(home))
    monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
    L.clear_cache()
    return home


def _write(home, sig, text, gate="g"):
    d = home / "system" / "optim"
    staged = d / f"{sig}.json.staging"
    staged.write_text(json.dumps({"signature_name": sig,
                                  "optimized_instruction": text,
                                  "gate_arm": gate}))
    os.replace(staged, d / f"{sig}.json")


def test_a_bare_import_serves_its_first_read_without_clear_cache(tmp_path):
    """Mutant: `_CURRENT_EPOCH: Optional[_Epoch] = None` deleted at module
    level survived 2,038 tests — every test calls `clear_cache()` first, which
    re-creates the global. Production's first read happens in whatever process
    imported the module; nothing guarantees a `clear_cache()` before it. Drive
    the import→first-read path in a fresh interpreter."""
    home = tmp_path / "home"
    (home / "system" / "optim").mkdir(parents=True)
    _write(home, "planning.decompose", "TUNED-FIRST-READ")   # an artifact exists
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "GHOST_HOME": str(home)}
    code = ("from ghost_agent.optim import loader as L\n"
            "print(L.tuned_instruction('planning.decompose', 'BASE', "
            "context=None, req_id='rq'))\n"
            "print(L.maybe_advance_epoch())\n")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    # The FIRST read must serve the artifact (not the baseline), and the tick
    # must then report no change — both computed by a process that never
    # called clear_cache().
    assert r.stdout.splitlines()[:2] == ["TUNED-FIRST-READ", "None"], r.stdout


def test_provenance_warning_fires_once_per_cache_life_across_snapshots(
        tmp_path, monkeypatch, caplog):
    """Mutant: `_WARNED_PROVENANCE.add(_wkey)` deleted — the dedup set is
    never filled, so an UNGATED artifact re-warns on EVERY snapshot. The F4
    pin counts warnings across two cache LIVES (one snapshot each), where
    'never dedup' and 'dedup per life' agree. Two snapshots in ONE life: a
    second, unrelated promotion must not repeat the first artifact's
    warning."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "planning.decompose", "T", gate="UNGATED (--no-ab-gate)")
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        L.maybe_advance_epoch()                 # snapshot 1 → warns once
        _write(home, "verifier.judge", "U")     # unrelated promotion
        assert L.maybe_advance_epoch()          # snapshot 2, same cache life
    warns = [r for r in caplog.records
             if "UNGATED" in r.message and "planning.decompose" in r.message]
    assert len(warns) == 1, (
        f"{len(warns)} UNGATED warnings for one artifact within one cache "
        f"life — the warn-once set is not being filled")
    L.clear_cache()


def test_a_repinned_request_is_touched_to_the_LRU_tail(tmp_path, monkeypatch):
    """Mutant: `_PINNED.pop(req_id, None)` deleted on the orphaned-pin
    fallback. Reassigning an existing OrderedDict key keeps its POSITION, so
    the re-pinned (active!) request stays the LRU victim at the cap; the pop
    + insert is what makes the re-pin a touch. Plant the orphan at the head,
    re-pin it, flood past the cap: the untouched flood entry must go, the
    re-pinned request must stay."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "planning.decompose", "A")
    L.clear_cache()
    L.tuned_instruction("planning.decompose", "BASE", context=None, req_id="victim")
    for i in range(L._PINNED_MAX - 1):
        L.tuned_instruction("planning.decompose", "BASE", context=None,
                            req_id=f"flood{i}")
    assert next(iter(L._PINNED)) == "victim"      # premise: oldest position
    assert len(L._PINNED) == L._PINNED_MAX          # premise: AT the cap
    L._PINNED["victim"] = 999_999                 # orphan it in place
    L.tuned_instruction("planning.decompose", "BASE", context=None, req_id="victim")
    assert L._PINNED["victim"] == L.current_generation()
    for i in range(2):
        L.tuned_instruction("planning.decompose", "BASE", context=None,
                            req_id=f"late{i}")
    assert "victim" in L._PINNED, (
        "the re-pinned ACTIVE request was evicted first — the fallback "
        "re-pin did not count as a touch")
    assert "flood0" not in L._PINNED
    L.clear_cache()


def test_a_no_change_tick_keeps_the_generation(tmp_path, monkeypatch):
    """Mutant: `if stamp == _CURRENT_EPOCH.stamp: return None` → False. The
    tick would mint a NEW generation every 60 s with nothing changed —
    quiet in the summary (no shas differ), so the I7 pin is blind to it —
    churning `_EPOCHS`, the registry name-set rebuild and every unpinned
    reader's identity. Pin the generation, not the summary."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "planning.decompose", "A")
    L.maybe_advance_epoch()
    gen = L.current_generation()
    ep = L._CURRENT_EPOCH
    assert L.maybe_advance_epoch() is None
    assert L.maybe_advance_epoch() is None
    assert L.current_generation() == gen
    assert L._CURRENT_EPOCH is ep, "a no-change tick replaced the epoch object"
    L.clear_cache()


def test_forgetting_an_orphaned_pin_does_not_raise(tmp_path, monkeypatch):
    """Mutant: `_release_gen`'s `if ep is not None:` → True. A request pinned
    to a generation that `clear_cache()` has since dropped ends normally via
    `forget_request` — under the mutant that is an AttributeError on None
    at request end. Plant the orphan directly (clear_cache empties _PINNED)."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "planning.decompose", "A")
    L.clear_cache()
    L._PINNED["orphan"] = 999_999
    L.forget_request("orphan")          # must not raise
    assert "orphan" not in L._PINNED
    L.clear_cache()


def test_unknown_module_attributes_still_raise(tmp_path, monkeypatch):
    """Mutants: PEP 562 `__getattr__` with `if name == "_ARTIFACT_SHAS"` →
    True (every unknown name returns the shas) or the trailing
    `raise AttributeError` deleted (every unknown name is None). Either
    turns `hasattr(loader, anything)` into True for the whole module."""
    _home(tmp_path, monkeypatch)                # an isolated, cleared epoch
    with pytest.raises(AttributeError):
        getattr(L, "definitely_not_an_attribute_xyz")
    assert not hasattr(L, "definitely_not_an_attribute_xyz")
    assert isinstance(L._ARTIFACT_SHAS, dict)   # the legacy view still works
    L.clear_cache()


def test_unnoting_the_last_signature_drops_the_ring_slot(tmp_path, monkeypatch):
    """Mutant: `_SERVED_RING.pop(req_id, None)` deleted in `unnote_served`.
    An emptied slot must leave the bounded ring, or dead requests keep
    occupying ring positions until they evict a live one."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "planning.decompose", "A")
    L.clear_cache()
    _write(home, "verifier.judge", "B")
    L.clear_cache()
    L.tuned_instruction("planning.decompose", "BASE", context=None, req_id="rq-u")
    L.tuned_instruction("verifier.judge", "BASE", context=None, req_id="rq-u")
    assert "rq-u" in L._SERVED_RING
    L.unnote_served("rq-u", "planning.decompose")
    assert "rq-u" in L._SERVED_RING, "a slot with a signature left must stay"   # `if not slot` → True
    L.unnote_served("rq-u", "verifier.judge")
    assert "rq-u" not in L._SERVED_RING, "empty slot left in the served ring"
    L.clear_cache()


def test_a_file_vanishing_mid_glob_does_not_hide_the_others(tmp_path, monkeypatch):
    """Mutant: `_dir_stamp`'s OSError `continue` → `break`. With two files and
    the FIRST stat failing, `break` drops the second from the stamp — a
    phantom change on the next tick (the file 'appears'), i.e. a spurious
    epoch swap. The existing vanish pin used a single file."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "a.first", "A")
    _write(home, "z.second", "Z")
    real_stat = type(home).stat
    calls = {"n": 0, "failed": None}

    def _flaky_stat(self, *a, **k):
        # fail the FIRST .json the glob yields, whatever its name — glob order
        # is not name-sorted on this filesystem (R2 reviewer)
        if self.suffix == ".json" and calls["n"] == 0:
            calls["n"] += 1; calls["failed"] = self.name
            raise OSError("vanished mid-glob")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(type(home), "stat", _flaky_stat)
    stamp = L._dir_stamp()
    monkeypatch.setattr(type(home), "stat", real_stat)
    assert stamp is not None and calls["failed"] is not None
    other = {"a.first.json", "z.second.json"} - {calls["failed"]}
    assert [e[0] for e in stamp] == sorted(other), stamp   # the survivor is stamped


def test_the_stamp_is_name_sorted_whatever_the_glob_order(tmp_path, monkeypatch):
    """`_dir_stamp` sorts; the snapshot's 'a bad artifact before a good one'
    premise and the stamp equality across ticks both rest on it."""
    home = _home(tmp_path, monkeypatch)
    for sig in ("m.mid", "a.first", "z.last"):
        _write(home, sig, "x")
    assert [e[0] for e in L._dir_stamp()] == ["a.first.json", "m.mid.json", "z.last.json"]


def test_a_bad_artifact_before_a_good_one_does_not_stop_the_snapshot(tmp_path, monkeypatch):
    """Mutants: `_snapshot`'s three skip paths (`continue` → `break`). Sorted
    stamp order puts `a.bad` before `z.good`; every skip must SKIP, not stop.
    The existing bad-file pin had the good artifact sorting first."""
    home = _home(tmp_path, monkeypatch)
    d = home / "system" / "optim"
    (d / "a.bad.json").write_text(json.dumps({"signature_name": "a.bad",
                                              "optimized_instruction": "   "}))   # empty → skipped
    (d / "b.bad.json").write_text("{not json")                                    # unreadable → skipped
    _write(home, "z.good", "GOOD")
    L.clear_cache()
    assert L.tuned_instruction("z.good", "BASE", context=None, req_id="") == "GOOD"
    L.clear_cache()


def test_an_empty_but_present_epoch_held_unreadable_does_not_warn(
        tmp_path, monkeypatch, caplog):
    """Mutant: the HOLD warning's `shas and stamp is not None` → `or`. F5a
    pinned the no-dir boot (stamp None); an EMPTY dir (stamp `()`) that then
    becomes unreadable would warn under `or` — 'holding 0 artifacts' is not
    an emergency."""
    home = _home(tmp_path, monkeypatch)
    L.maybe_advance_epoch()                   # empty dir → stamp ()
    assert L._CURRENT_EPOCH.stamp == ()
    monkeypatch.setattr(L, "_dir_stamp", lambda: None)   # now unreadable
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        assert L.maybe_advance_epoch() is None
    assert not [r for r in caplog.records if "HOLDING" in r.message], (
        "held an EMPTY epoch and warned about a mass retirement")
    L.clear_cache()


def test_only_the_current_generation_survives_an_unpinned_swap(tmp_path, monkeypatch):
    """Mutant: `_drop_unpinned`'s `ep is not _CURRENT_EPOCH` → `is`. With no
    pins anywhere, two swaps must leave EXACTLY the current generation in
    `_EPOCHS` — the mutant deletes the current one and keeps every old one
    (a leak, and `_release_gen` can then never find the live epoch)."""
    home = _home(tmp_path, monkeypatch)
    _write(home, "planning.decompose", "A")
    L.maybe_advance_epoch()
    _write(home, "planning.decompose", "B")
    L.maybe_advance_epoch()
    _write(home, "planning.decompose", "C")
    L.maybe_advance_epoch()
    assert set(L._EPOCHS) == {L.current_generation()}, sorted(L._EPOCHS)
    assert L._EPOCHS[L.current_generation()] is L._CURRENT_EPOCH
    L.clear_cache()
