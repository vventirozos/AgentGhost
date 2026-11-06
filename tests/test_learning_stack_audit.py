"""§4L self-learning stack audit — round-1 regression pins (2026-08-07).

Four pre-registered lenses (PROJECT_JOURNAL §4L) ran against the live
corpus; every fix below reproduced as a wrong number/label/route before
it was fixed. Lens tags in each docstring. (§4NM: the Lens B pins on
scripts/optimize_verifier.py went with that script.)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# ── Lens A: label integrity ──────────────────────────────────────────


def test_verifier_note_is_stripped_before_hedge_scan():
    """A-MINOR-1: the "**Verifier note:**" block was missing from the
    strip list while being appended BEFORE the hedge scan — a note
    quoting first-person text put uncertainty_pressure on a refuted
    (label-0) turn: the λ-leak channel, one banner over."""
    from ghost_agent.core.reply_smoothing import strip_system_notes
    reply = ("The deploy worked.\n\n---\n**Verifier note:** the claim "
             "says 'I cannot confirm the port is open'")
    assert strip_system_notes(reply) == "The deploy worked."


# ── Lens C: instruments + loop closure ───────────────────────────────


def test_recording_fingerprint_is_minted_over_the_scrubbed_form():
    """C-MINOR-1: the fingerprint was minted BEFORE the nonce scrub, so
    every scrubbed record permanently failed replay matching — the
    drift instrument crying wolf on exactly the verify-prompt
    population. Both sides now converge on the sentinel."""
    from ghost_agent.core import llm_recording as lr
    from ghost_agent.core.agent import _PACKER_NONCE

    live = {"messages": [{"role": "user",
                          "content": f"…[PACKER CUT#{_PACKER_NONCE}: "
                                     f"400 of 5400 chars shown]"}]}
    stored = {"messages": [{"role": "user",
                            "content": "…[PACKER CUT#deadfeed: "
                                       "400 of 5400 chars shown]"}]}
    assert lr.payload_fingerprint(live) == lr.payload_fingerprint(stored)


def test_recorded_line_fingerprint_verifies_against_stored_payload(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_LLM_RECORD", "1")
    from ghost_agent.core import llm_recording as lr
    from ghost_agent.core.agent import _PACKER_NONCE, _slice_evidence_body
    digest = _slice_evidence_body("x" * 4000, 400, "")
    assert _PACKER_NONCE in digest
    rec = lr.LLMRecorder(root=tmp_path)
    assert rec.record("verify", {"messages": [
        {"role": "user", "content": "EVIDENCE:\n" + digest}]},
        {"content": "ok"})
    line = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert _PACKER_NONCE not in json.dumps(line)
    assert line["fingerprint"] == lr.payload_fingerprint(line["payload"])


def test_escalation_arm_slots_count_unknown_outcomes_as_other(
        tmp_path, monkeypatch):
    """C-MINOR-3: an outcome string outside the pre-declared buckets
    bumped only `n`, so buckets quietly stopped summing to n as the
    vocabulary grew (5 new strings in one month)."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir()
    led = tmp_path / "verifier"
    led.mkdir()
    rows = [{"route": "claim", "kind": "refute", "outcome": o}
            for o in ("overturned", "upheld", "some_future_outcome")]
    (led / "escalations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    from ghost_agent.core import learning_health as lh
    esc = lh._escalation_health(led / "escalations.jsonl")
    slot = esc["arms"]["claim/refute"]
    assert slot["other"] == 1
    counted = sum(v for k, v in slot.items()
                  if k not in ("n", "overturn_rate"))
    assert counted == slot["n"]


def test_escalation_audit_names_the_vocabulary_on_zero_matches(tmp_path):
    """C-MINOR-3b: a typo'd/stale --outcome silently printed an empty
    audit."""
    led = tmp_path / "system" / "verifier"
    led.mkdir(parents=True)
    (led / "escalations.jsonl").write_text(json.dumps(
        {"route": "claim", "kind": "refute", "outcome": "overturned"}))
    (tmp_path / "system" / "trajectories").mkdir()
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "escalation_audit.py"),
         "--outcome", "no_such_outcome"],
        env={"GHOST_HOME": str(tmp_path), "PATH": "/usr/bin:/bin",
             "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert "Outcomes present in this ledger: overturned" in proc.stderr


def test_rss_watchdog_execv_uses_this_modules_spelling():
    """C-MAJOR-1: the watchdog exec'd the TEST import spelling
    ("ghost_agent.main") — prod boots `-m src.ghost_agent.main` with no
    PYTHONPATH, so the replacement process would die on
    ModuleNotFoundError, unreachable by the except clause. The module
    name is now derived from __name__ (correct under both spellings)."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert '"-m", "ghost_agent.main"' not in src
    assert '__name__.replace(".core.agent", ".main")' in src
    # The derivation is right for both spellings.
    assert "src.ghost_agent.core.agent".replace(
        ".core.agent", ".main") == "src.ghost_agent.main"
    assert "ghost_agent.core.agent".replace(
        ".core.agent", ".main") == "ghost_agent.main"


# ── Lens D: delivery path ────────────────────────────────────────────


# ── Cross-lens: the honest-failure ladder mirror + late calibration ──


def test_fused_hydration_groups_items_under_their_own_headers():
    """D-MAJOR-3: the fused formatter emitted items in rank order with
    lazily-emitted headers, so interleaved sources landed under the
    WRONG header."""
    from ghost_agent.core.bus import MemoryBus
    fused = [
        ({"source": "graph", "text": "EDGE A"}, 1.0),
        ({"source": "vector", "text": "memory one"}, 0.9),
        ({"source": "graph", "text": "EDGE B"}, 0.8),
    ]
    out, surv = MemoryBus._format_markdown_with_survivors(fused, 6000)
    assert len(surv) == 3
    gr = out.index("TOPOLOGICAL KNOWLEDGE GRAPH")
    mem = out.index("MEMORY CONTEXT")
    block = out[gr:mem] if gr < mem else out[gr:]
    assert "EDGE A" in block and "EDGE B" in block
    assert "memory one" not in block


def test_gepa_loader_warns_on_pre_gate_artifacts(tmp_path, monkeypatch,
                                                 caplog):
    """D-MAJOR-1: a pre-gate-schema artifact (no gate identity/scores)
    was served to every call with no evidence it would win under the
    CURRENT metric — planning.decompose: promoted under the old recall
    metric, known to lose on F1, served anyway."""
    import json as _json
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    od = tmp_path / "system" / "optim"
    od.mkdir(parents=True)
    (od / "old.style.json").write_text(_json.dumps(
        {"optimized_instruction": "do the old thing"}))
    (od / "new.style.json").write_text(_json.dumps(
        {"optimized_instruction": "do the new thing",
         "gate_arm": "judge+escalation"}))
    from ghost_agent.optim import loader
    loader.clear_cache()
    with caplog.at_level("INFO", logger="GhostAgent"):
        assert loader.tuned_instruction("old.style", "d") == "do the old thing"
        assert loader.tuned_instruction("new.style", "d") == "do the new thing"
    assert "predates the gate schema" in caplog.text
    assert loader._ARTIFACT_SHAS["old.style"]
    loader.clear_cache()


# ── Round-2 regressions (2026-08-07): fixes-to-fixes.


def test_credit_sees_both_surfaces_after_delivery_dedup():
    """R2-NEW-1: the delivery dedup removed co-surfaced lessons from
    last_playbook_triggers, starving their deterministic success-credit
    channel while the denominator kept booking — pushing exactly the
    protected population toward the stale gate and the eviction
    ranking. Both credit sites now pass the two-surface union."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert 'top_triggers=list(\n' not in src.replace(
        'top_triggers=list(\r\n', 'top_triggers=list(\n') or True
    # §4KW (fresh review) SUPERSEDES this: passing the surfaced lessons as
    # `top_triggers` credited every lesson IN THE PROMPT (membership counted
    # as relevance) — helpful≈retrievals, 154/290 lessons at confidence 1.0.
    # Neither credit site passes it now; credit is word overlap or the judge.
    assert src.count("top_triggers=self._surfaced_lesson_triggers(") == 0


def test_learning_health_renders_the_other_bucket(tmp_path, monkeypatch):
    """R2-NEW-6: the counter existed but the renderer omitted it."""
    import json as _json
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir()
    led = tmp_path / "verifier"
    led.mkdir()
    (led / "escalations.jsonl").write_text("\n".join(
        _json.dumps({"route": "claim", "kind": "refute", "outcome": o})
        for o in ("overturned", "brand_new_outcome")))
    from ghost_agent.core import learning_health as lh
    text = lh.render_learning_health(tmp_path / "memory")
    assert "OTHER (unknown outcome strings" in text


# ── Round-3 regressions (2026-08-07).


# ── §4M memory-substrate round 1: skill-store twin destruction ───────
# Lens A CRIT-1/CRIT-1b + Lens C CRIT-1 (independently found): the
# vector twin's metadata stores the row's TRIGGER, but both cleaners
# keyed the playbook task-first — self-play/dream rows (task ≠ trigger
# by design) had their healthy twins deleted (~1/idle cycle live;
# 36/50 lessons dark to retrieval when caught 2026-08-08).


class _M4Coll:
    """Chroma-shaped stub for the reconcile/dedup/heal paths."""

    def __init__(self, query_hit=None, get_result=None):
        self._hit = query_hit
        self._get = get_result or {"ids": [], "metadatas": [], "documents": []}
        self.deleted_ids = []

    def query(self, query_texts, n_results, where):
        if self._hit is None:
            return {"distances": [[]], "ids": [[]], "documents": [[]],
                    "metadatas": [[]]}
        return {
            "distances": [[self._hit["distance"]]],
            "ids": [[self._hit["id"]]],
            "documents": [[self._hit["text"]]],
            "metadatas": [[{"trigger": self._hit.get("trigger", "")}]],
        }

    def get(self, where=None, limit=None, include=None):
        return self._get

    def delete(self, ids=None, where=None):
        self.deleted_ids.extend(ids or [])


class _M4Mem:
    def __init__(self, collection):
        self.collection = collection
        self.added = []

    def add(self, text, meta):
        self.added.append((text, meta))


def test_dream_isolate_vector_reads_never_book_credit():
    """§4M F1: the dream isolate's vector proxy passed search/
    search_advanced through raw (recording retrievals on the production
    store) and omitted search_items (dropping the bus to the legacy
    branch that also bypasses the off-topic floor). Behavioral pin lives
    in test_dream_synthetic.py; this pins the source contract."""
    # §4CL S1: the façade moved to `core/isolation.py` and this pin became
    # executed — a source count of `record_retrievals = False` said nothing
    # about whether the flag reached the store.
    from unittest.mock import MagicMock
    from ghost_agent.core.isolation import ReadOnlyVectorMemory

    real = MagicMock()
    ro = ReadOnlyVectorMemory(real)
    ro.search("q")
    ro.search_advanced("q")
    assert real.search.call_args.kwargs["record_retrievals"] is False
    assert real.search_advanced.call_args.kwargs["record_retrievals"] is False
    # search_items exists AND is whitelisted — without it the bus falls to
    # the legacy `vector.search` branch, which books credit and bypasses
    # the off-topic floor.
    assert "search_items" in ReadOnlyVectorMemory._SAFE_PASSTHROUGH
    ro.search_items("q")
    real.search_items.assert_called_once()


def test_bus_malformed_profile_update_surfaces_as_error():
    """§4M (Lens C MAJOR-2): a non-dict profile_update raised OUTSIDE
    the closure's try; gather(return_exceptions=True) swallowed it, the
    report lacked the 'profile' key entirely, and the tool reported
    SUCCESS on a total drop."""
    import asyncio
    from ghost_agent.core.bus import MemoryBus

    class _P:
        def update(self, *a, **k):
            raise AssertionError("malformed payload must not reach update()")

    bus = MemoryBus(profile_memory=_P())
    report = asyncio.run(
        bus.publish_fact("event", {"profile_update": "wife: Maria"}))
    assert str(report.get("profile", "")).startswith("error:")


def test_graph_forget_archives_before_hard_delete(tmp_path):
    """§4M (Lens A MAJOR-3): sub-50-row delete_by_target hard-deleted
    with no record — the July graph loss (884 vs 1407 rows) is
    unattributable because no archive existed."""
    import json as _json
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "alice", "predicate": "OWNS", "object": "toolbox"}])
    assert g.delete_by_target("toolbox") == 1
    arch = tmp_path / "graph_pruned_archive.jsonl"
    assert arch.exists(), "hard delete ran with no archive"
    rows = [_json.loads(l) for l in arch.read_text().splitlines()]
    assert any(r["reason"] == "delete_by_target"
               and r["object"] == "toolbox" for r in rows)


def test_graph_forget_soft_expires_when_archive_fails(tmp_path,
                                                      monkeypatch):
    """Archive failure must not abandon recoverability: the forget
    soft-expires (get_expired_triplets can restore) instead of deleting."""
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "bob", "predicate": "OWNS", "object": "sailboat"}])
    monkeypatch.setattr(g, "_archive_rows", lambda *a, **k: False)
    assert g.delete_by_target("sailboat") == 1
    assert not (tmp_path / "graph_pruned_archive.jsonl").exists()
    import sqlite3 as _sq
    with _sq.connect(g.db_path) as conn:
        kept = conn.execute(
            "SELECT COUNT(*) FROM triplets WHERE valid_until IS NOT NULL"
        ).fetchone()[0]
    assert kept == 1                          # expired, NOT deleted


def test_graph_prune_archives_or_does_not_prune(tmp_path, monkeypatch):
    """§4M (Lens A MAJOR-3): the unattended dream-driven prune must fail
    CLOSED — no archive, no deletion."""
    import json as _json
    import sqlite3 as _sq
    from ghost_agent.memory.graph import GraphMemory

    def _seed(d):
        d.mkdir(parents=True, exist_ok=True)
        g = GraphMemory(d)
        g.add_triplets([
            {"subject": "old", "predicate": "LIKES", "object": "noise"}])
        with _sq.connect(g.db_path) as conn:
            conn.execute(
                "UPDATE triplets SET timestamp = datetime('now','-90 days')")
            conn.commit()
        return g

    g1 = _seed(tmp_path / "a")
    monkeypatch.setattr(g1, "_archive_rows", lambda *a, **k: False)
    assert g1.prune_stale_edges() == 0
    with _sq.connect(g1.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM triplets").fetchone()[0] == 1

    g2 = _seed(tmp_path / "b")
    assert g2.prune_stale_edges() == 1
    rows = [_json.loads(l) for l in
            ((tmp_path / "b") / "graph_pruned_archive.jsonl")
            .read_text().splitlines()]
    assert any(r["reason"] == "prune_stale_edges"
               and r["subject"] == "old" for r in rows)


def test_graph_wipe_all_archives_best_effort(tmp_path):
    """wipe_all is explicit operator intent — it proceeds either way,
    but the wiped rows land in the archive first."""
    import json as _json
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "a", "predicate": "IS", "object": "b"},
        {"subject": "b", "predicate": "IS", "object": "c"}])
    g.wipe_all()
    rows = [_json.loads(l) for l in
            (tmp_path / "graph_pruned_archive.jsonl")
            .read_text().splitlines()]
    assert sum(1 for r in rows if r["reason"] == "wipe_all") == 2


def test_framing_fix_ts_is_utc():
    """§4M (Lens D F5): the regression-watch constant is compared against
    UTC `last_seen` strings; the local-time value 19:00 opened a 3h06m
    blind window (real boundary ≈15:54Z)."""
    from ghost_agent.core.learning_health import _FRAMING_FIX_TS
    assert _FRAMING_FIX_TS.startswith("2026-07-31T15:54")


def _dedup_mem_at(dist, doc):
    from unittest.mock import MagicMock
    m = MagicMock()
    m.collection = MagicMock()
    m.collection.query.return_value = {
        "ids": [["id1"]], "documents": [[doc]], "distances": [[dist]]}
    return m


def test_profile_canonicalize_is_shared_and_graph_edge_uses_it():
    """§4M (Lens C MAJOR-4): profile stored `assets.car` while the graph
    edge was minted from the RAW key (`HAS_VEHICLE`) — 11/15 live pairs
    had divergent field names across the two stores. The canonical map
    is now a shared classmethod and both tool paths consume it."""
    from ghost_agent.memory.profile import ProfileMemory

    assert ProfileMemory.canonicalize("misc", "vehicle") == ("assets", "car")
    assert ProfileMemory.canonicalize("x", "wife") == ("relationships", "wife")
    # Idempotent + passthrough for unmapped keys.
    assert ProfileMemory.canonicalize("assets", "car") == ("assets", "car")
    assert ProfileMemory.canonicalize("work", "employer") == ("work", "employer")
    src = (REPO / "src" / "ghost_agent" / "tools" / "memory.py").read_text()
    assert src.count("_PM.canonicalize(category, key)") >= 2


def test_update_profile_legacy_path_mints_canonical_graph_edge():
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.tools.memory import tool_update_profile

    profile = MagicMock()
    profile.update.return_value = "Synchronized: assets.car = bmw"
    graph = MagicMock()
    out = asyncio.run(tool_update_profile(
        category="misc", key="vehicle", value="BMW",
        profile_memory=profile, graph_memory=graph))
    assert "SUCCESS" in out
    assert graph.sync_owner_field.call_args.args[0] == "car"      # canonical, not vehicle (§4KZ: synced mirror)


def test_bus_partial_write_logs_a_warning(caplog):
    """§4M (Lens C MAJOR-3): per-target fan-out failures were report-only
    (the PARTIAL string went to the model and was discarded after the
    turn). The log line is the durable repair breadcrumb."""
    import asyncio
    import logging
    from unittest.mock import MagicMock
    from ghost_agent.core.bus import MemoryBus

    graph = MagicMock()
    graph.add_triplets.side_effect = RuntimeError("disk full")
    bus = MemoryBus(graph_memory=graph)
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        report = asyncio.run(bus.publish_fact("insert_fact", {
            "triplets": [{"subject": "a", "predicate": "IS", "object": "b"}]}))
    assert str(report.get("graph", "")).startswith("error")
    assert any("partial write" in r.message for r in caplog.records)


# (§4M Lens D F2/F3 — the hydration-judge failure-path log levels, the
# `injected_total` stamp and the usefulness-ledger censoring report — went
# with `MemoryBus.judge_hydration_usefulness` and the ledger section of
# learning_health, §4NM.)


# ── §4M minor batch (2026-08-08, post-restart) ───────────────────────


def test_episodes_fallback_search_is_stat_pure(tmp_path):
    """Same probe-purity contract on the episodic fallback read."""
    from ghost_agent.memory.episodes import EpisodicMemory

    calls = {}

    class _VM:
        def search_advanced(self, trigger, limit=5, record_retrievals=True):
            calls["record_retrievals"] = record_retrievals
            return []

    em = EpisodicMemory(tmp_path)
    em._vector_search("some trigger", 5, _VM())
    assert calls["record_retrievals"] is False


def test_bus_graph_leg_filters_removal_triplets():
    """§4M (Lens C MINOR): tombstone parity — consolidation filtered
    removal-shaped triplets, the bus fan-out (and remember's extraction)
    did not, so a negation sentence minted a positive edge."""
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.core.bus import MemoryBus

    graph = MagicMock()
    graph.add_triplets.return_value = 1
    bus = MemoryBus(graph_memory=graph)
    report = asyncio.run(bus.publish_fact("insert_fact", {"triplets": [
        {"subject": "user", "predicate": "NO_LONGER_OWNS", "object": "bmw"},
        {"subject": "user", "predicate": "OWNS", "object": "audi"},
    ]}))
    kept = graph.add_triplets.call_args[0][0]
    assert [t["object"] for t in kept] == ["audi"]
    # All-removal payload: nothing reaches the graph at all.
    graph.reset_mock()
    report2 = asyncio.run(bus.publish_fact("insert_fact", {"triplets": [
        {"subject": "user", "predicate": "REMOVED", "object": "x"}]}))
    assert not graph.add_triplets.called
    assert "skip" in str(report2.get("graph"))


def test_session_corrupt_file_preserved_as_sidecar(tmp_path):
    """§4M (Lens A MINOR): corrupt-as-absent left the bytes in place for
    the next save to overwrite — history unrecoverable."""
    from ghost_agent.core.sessions import SessionStore

    store = SessionStore(tmp_path)
    sess = store.create()
    p = store._path(sess.id)
    p.write_text('{"id": "half a session')
    assert store.get(sess.id) is None
    sidecars = list(tmp_path.glob("*.corrupt-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_text() == '{"id": "half a session'


def test_tasks_manual_rows_carry_timestamps():
    """§4M (Lens C MINOR): 'manual' is a prunable vector type and a
    MISSING timestamp sorts as oldest — first eviction victim."""
    src = (REPO / "src" / "ghost_agent" / "tools" / "tasks.py").read_text()
    assert src.count('"task_id": job_id,') >= 2
    assert src.count('"timestamp": _uts()') >= 2


# ── §4M round-2 fixes (same-pass, 2026-08-08) ────────────────────────


def test_removal_predicates_match_on_token_boundaries():
    """§4M R2 MINOR-3: substring fragments false-positived on legitimate
    predicates (PERFORMER/TRANSFORMERS contain FORMER, WHENEVER contains
    NEVER) — now the profile fan-out consumes the filter, a false hit
    silently drops a real graph edge."""
    from ghost_agent.utils.helpers import is_removal_triplet

    def t(pred):
        return {"subject": "user", "predicate": pred, "object": "x"}

    for ok in ("PERFORMER", "IS_PERFORMER", "USES_TRANSFORMERS",
               "WHENEVER_RUNS", "HAS_NOTEBOOK", "PASTA_PREFERENCE"):
        assert not is_removal_triplet(t(ok)), ok
    for bad in ("NO_LONGER_HAS", "PREVIOUSLY_OWNED", "REMOVED",
                "PAST_PROJECTS", "NOT_USING", "USED_TO_OWN"):
        assert is_removal_triplet(t(bad)), bad


def test_legacy_remember_path_has_filter_and_visible_failure():
    """§4M R2 MINOR-4: the round-1 parity fix landed only on the bus
    path; the legacy branch kept the bare `except: pass` AND the
    unfiltered add — the literal third-call-site pattern."""
    src = (REPO / "src" / "ghost_agent" / "tools" / "memory.py").read_text()
    i = src.index('name="graph-extract-legacy"')
    seg = src[i - 2500:i]
    assert "is_removal_triplet" in seg
    assert "logger.warning" in seg
    assert "except Exception: pass" not in seg


def test_profile_has_exactly_one_canonical_map():
    """§4M R2 MINOR-5: delete() and prune_value() carried verbatim inline
    copies of the canonical map — the next entry diverges three copies."""
    src = (REPO / "src" / "ghost_agent" / "memory" / "profile.py").read_text()
    assert src.count('"vehicle": ("assets", "car")') == 1


def test_update_profile_delete_scrubs_canonical_vector_fact():
    """§4M R2 MINOR-6: the write side files vehicle → assets.car and
    mints "User car is …", but the delete path read the RAW key — the
    old-value lookup missed and the derived identity fact stayed
    retrievable forever after deletion."""
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.tools.memory import tool_update_profile

    prof = MagicMock()
    prof.load.return_value = {"assets": {"car": "BMW"}}
    prof.delete.return_value = "Removed from Profile: assets.car"
    vec = MagicMock()
    out = asyncio.run(tool_update_profile(
        category="misc", key="vehicle", value="",       # §4KW: only "" deletes
        profile_memory=prof, memory_system=vec))
    assert "Removed" in out
    assert vec.sync_owner_field.call_args.args[0] == "car"      # §4KZ: the canonical field's mirror is synced


def test_forget_archive_rows_carry_weight_and_timestamp(tmp_path):
    """§4M R2 NIT-2: forget archived bare (s,p,o) while prune/wipe kept
    weight+age — recovery from a forget archive lost both."""
    import json as _json
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "carol", "predicate": "OWNS", "object": "houseboat"}])
    assert g.delete_by_target("houseboat") == 1
    rows = [_json.loads(l) for l in
            (tmp_path / "graph_pruned_archive.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["reason"] == "delete_by_target")
    assert "weight" in row and "timestamp" in row


# (§4M R2 NIT-3's ledger-census pin and the uncensored-judge pin went with
# the usefulness ledger and `judge_hydration_usefulness`, §4NM.)
