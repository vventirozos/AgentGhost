"""§4CT — the graduated-skill loop stops being UNMEASURED.

§4CS's yield surface named exactly one remaining instrumentation gap:
`skills.graduated` read `unmeasured` because the skills ARE injected into
the prompt on every matching turn and NOTHING anywhere counted it. That is
not a zero — the remedy for "no counter" is to wire a counter, and the
remedy for "measured zero" is to kill or fix the loop, and reading one as
the other kills a loop that may be working.

Three properties, and the third is the one that is easy to get wrong:

  1. the hashes reach the caller at the moment they are known
     (`format_for_prompt` threw them away, which is the whole gap);
  2. the booking is idempotent per turn — a counter that can over-report is
     not an honest number;
  3. only REAL traffic counts. Self-play, dream, subagent and bench turns
     inject the same block, and counting them is the §4K defect verbatim:
     28 "user turns" reported on a box whose true count was ZERO.
"""

import json
import types
from pathlib import Path

import pytest

from ghost_agent.core import liveness as L
from ghost_agent.core.agent import book_graduated_retrieval
from ghost_agent.core.liveness import (
    BARREN, EMPTY, UNMEASURED, YIELDING, _yield_status, yield_all,
)
from ghost_agent.skills_auto.store import GraduatedSkillStore


def _store(tmp_path, entries=None):
    entries = entries if entries is not None else {
        "h1": {"signature_hash": "h1", "name": "alpha", "cluster": "db",
               "tool_sequence": ["execute", "file_system"], "support": 5,
               "confidence": 0.9, "trigger_examples": ["slow query"],
               "verifications": 7},
        "h2": {"signature_hash": "h2", "name": "beta", "cluster": None,
               "tool_sequence": ["web_search"], "support": 3,
               "confidence": 0.8, "trigger_examples": ["look it up"],
               "verifications": 2},
    }
    (tmp_path / "auto_skills.json").write_text(json.dumps(entries))
    return GraduatedSkillStore(tmp_path)


def _ctx(origin="user"):
    """A context `turn_origin` reads as `origin`."""
    ro = origin != "user"
    c = types.SimpleNamespace(
        skill_memory=types.SimpleNamespace(is_read_only=ro))
    if origin not in ("user", "sim"):
        c.turn_origin_label = origin
    return c


def _raw(store):
    return json.loads(store.path.read_text())


# ── the hashes reach the caller ─────────────────────────────────────────
class TestSurfacedForPrompt:
    def test_it_returns_the_block_AND_the_hashes_it_listed(self, tmp_path):
        s = _store(tmp_path)
        block, hashes = s.surfaced_for_prompt(query="slow query", limit=3)
        assert block.startswith("### PROVEN APPROACHES")
        assert hashes == ["h1"]
        # The hashes must describe the block that was actually rendered —
        # one formatter, so they cannot drift.
        assert "alpha" not in block          # the block lists sequences...
        assert "execute → file_system" in block   # ...not names

    def test_format_for_prompt_is_a_WRAPPER_not_a_second_formatter(
            self, tmp_path):
        s = _store(tmp_path)
        block, _h = s.surfaced_for_prompt(query="slow query", limit=3)
        assert s.format_for_prompt(query="slow query", limit=3) == block

    def test_nothing_relevant_surfaces_nothing(self, tmp_path):
        s = _store(tmp_path)
        assert s.surfaced_for_prompt(query="zzz nothing matches") == ("", [])

    def test_an_empty_store_surfaces_nothing(self, tmp_path):
        assert _store(tmp_path, {}).surfaced_for_prompt(query="x") == ("", [])

    def test_a_row_without_a_hash_is_not_offered_as_one(self, tmp_path):
        s = _store(tmp_path, {"h1": {"name": "n", "trigger_examples": ["q"],
                                     "tool_sequence": ["execute"],
                                     "signature_hash": ""}})
        block, hashes = s.surfaced_for_prompt(query="q")
        assert block and hashes == [], "an unbookable row must not be booked"


# ── the counter ─────────────────────────────────────────────────────────
class TestRecordSurfaced:
    def test_it_books_retrievals_and_a_timestamp(self, tmp_path):
        s = _store(tmp_path)
        assert s.record_surfaced(["h1", "h2"], turn_key="t1") == 2
        raw = _raw(s)
        assert raw["h1"]["retrievals"] == 1 and raw["h2"]["retrievals"] == 1
        assert raw["h1"]["last_retrieved_at"].endswith("Z")

    def test_it_is_idempotent_within_ONE_turn(self, tmp_path):
        """A counter that can over-report is not an honest number, and the
        lesson store's equivalent has three known double-booking routes."""
        s = _store(tmp_path)
        assert s.record_surfaced(["h1"], turn_key="t1") == 1
        assert s.record_surfaced(["h1"], turn_key="t1") == 0
        assert _raw(s)["h1"]["retrievals"] == 1

    def test_a_NEW_turn_books_again(self, tmp_path):
        s = _store(tmp_path)
        s.record_surfaced(["h1"], turn_key="t1")
        s.record_surfaced(["h1"], turn_key="t2")
        assert _raw(s)["h1"]["retrievals"] == 2

    def test_dedup_is_PER_SKILL_not_per_turn(self, tmp_path):
        """Surfacing h1 then h1+h2 in one turn must book h2 exactly once —
        not skip the whole call because the turn was seen."""
        s = _store(tmp_path)
        s.record_surfaced(["h1"], turn_key="t1")
        assert s.record_surfaced(["h1", "h2"], turn_key="t1") == 1
        raw = _raw(s)
        assert raw["h1"]["retrievals"] == 1 and raw["h2"]["retrievals"] == 1

    def test_an_interleaved_turn_does_not_reset_the_other_turns_dedup(
            self, tmp_path):
        """The lesson store keeps ONE `_retrieval_turn_key` slot, which any
        interleaved retrieval resets — a confirmed double-booking route.
        Keyed per turn here, so interleaving is harmless."""
        s = _store(tmp_path)
        s.record_surfaced(["h1"], turn_key="t1")
        s.record_surfaced(["h1"], turn_key="t2")
        assert s.record_surfaced(["h1"], turn_key="t1") == 0
        assert _raw(s)["h1"]["retrievals"] == 2

    def test_an_empty_turn_key_still_books(self, tmp_path):
        """It cannot be deduped, and under-counting a real turn is the
        worse direction. Simulated turns never reach here at all."""
        s = _store(tmp_path)
        assert s.record_surfaced(["h1"], turn_key="") == 1
        assert s.record_surfaced(["h1"], turn_key="") == 1
        assert _raw(s)["h1"]["retrievals"] == 2

    def test_an_unknown_hash_creates_nothing(self, tmp_path):
        s = _store(tmp_path)
        assert s.record_surfaced(["ghost"], turn_key="t1") == 0
        assert "ghost" not in _raw(s)

    def test_empty_input_writes_nothing(self, tmp_path):
        s = _store(tmp_path)
        before = s.path.read_text()
        assert s.record_surfaced([], turn_key="t1") == 0
        assert s.record_surfaced(None, turn_key="t1") == 0
        assert s.path.read_text() == before

    def test_it_never_raises(self, tmp_path, monkeypatch):
        s = _store(tmp_path)
        monkeypatch.setattr(s, "_load",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert s.record_surfaced(["h1"], turn_key="t1") == 0

    def test_the_turn_dedup_map_is_BOUNDED(self, tmp_path):
        s = _store(tmp_path)
        for i in range(s._TURN_DEDUP_MAX * 3):
            s.record_surfaced(["h1"], turn_key=f"t{i}")
        assert len(s._booked_by_turn) <= s._TURN_DEDUP_MAX

    def test_it_does_not_disturb_the_graduation_fields(self, tmp_path):
        s = _store(tmp_path)
        before = _raw(s)["h1"]["verifications"]
        s.record_surfaced(["h1"], turn_key="t1")
        after = _raw(s)["h1"]
        assert after["verifications"] == before
        assert after["confidence"] == 0.9 and after["support"] == 5


# ── REAL traffic only ───────────────────────────────────────────────────
class TestOnlyRealTrafficCounts:
    @pytest.mark.parametrize("origin,expected", [
        ("user", 1), ("sim", 0), ("bench", 0), ("subagent", 0),
    ])
    def test_the_gate_is_turn_origin(self, tmp_path, origin, expected):
        """§4K: counting simulated turns reported 28 "user turns" on a box
        whose true count was ZERO, while every ledger with its own
        simulation gate was correctly silent — the denominator and the
        ledgers counting opposite populations."""
        s = _store(tmp_path)
        assert book_graduated_retrieval(_ctx(origin), s, ["h1"]) == expected
        assert int(_raw(s)["h1"].get("retrievals") or 0) == expected

    def test_it_uses_the_CANONICAL_predicate_not_a_private_one(self, tmp_path,
                                                               monkeypatch):
        """A second notion of "is this real traffic" that can drift from the
        gates' notion is how the two came to disagree in the first place."""
        import ghost_agent.core.agent as A
        seen = []
        real = A.turn_origin
        monkeypatch.setattr(A, "turn_origin",
                            lambda c: (seen.append(c), real(c))[1])
        ctx = _ctx("user")
        A.book_graduated_retrieval(ctx, _store(tmp_path), ["h1"])
        assert seen == [ctx], "the booking must consult turn_origin"

    def test_no_hashes_is_a_noop(self, tmp_path):
        assert book_graduated_retrieval(_ctx("user"), _store(tmp_path), []) == 0

    def test_a_store_without_the_method_is_a_noop(self, tmp_path):
        assert book_graduated_retrieval(
            _ctx("user"), types.SimpleNamespace(), ["h1"]) == 0

    def test_it_never_raises(self, tmp_path):
        class _Boom:
            def record_surfaced(self, *a, **k):
                raise RuntimeError("nope")
        assert book_graduated_retrieval(_ctx("user"), _Boom(), ["h1"]) == 0
        assert book_graduated_retrieval(None, _store(tmp_path), ["h1"]) == 0


# ── the yield row stops lying ───────────────────────────────────────────
class TestTheYieldRowIsNowMeasured:
    def _home(self, tmp_path, entries=None):
        md = tmp_path / "system" / "memory"
        md.mkdir(parents=True)
        _store(md, entries)
        return tmp_path

    def test_an_UNRETRIEVED_store_is_a_MEASURED_zero_not_unmeasured(
            self, tmp_path):
        home = self._home(tmp_path)
        res = L._yield_graduated_skills(home)
        assert res.invoked == 0, "the channel is observable now"
        assert _yield_status(res) == BARREN
        assert _yield_status(res) != UNMEASURED
        assert "2 of 2 graduated skill(s) have NEVER been surfaced" in res.note

    def test_a_retrieval_flips_the_row_to_yielding(self, tmp_path):
        home = self._home(tmp_path)
        s = GraduatedSkillStore(home / "system" / "memory")
        book_graduated_retrieval(_ctx("user"), s, ["h1"])
        res = L._yield_graduated_skills(home)
        assert res.invoked == 1
        assert res.activated == 1, "retrieved at least once, not 'all of them'"
        assert res.minted == 2
        assert res.last_invoked is not None
        assert _yield_status(res) == YIELDING
        assert "skills.graduated" in yield_all(home)["rows"][0]["name"] \
            or "skills.graduated" not in yield_all(home)["unmeasured"]

    def test_activated_is_a_STATISTIC_not_the_row_count(self, tmp_path):
        """It used to be `len(rows)` — "all of them are eligible" — which is
        true of every store and evidence of nothing."""
        home = self._home(tmp_path)
        s = GraduatedSkillStore(home / "system" / "memory")
        book_graduated_retrieval(_ctx("user"), s, ["h1"])
        res = L._yield_graduated_skills(home)
        assert res.activated == 1 and res.minted == 2
        assert res.activated != res.minted

    def test_verifications_are_STILL_not_read_as_usage(self, tmp_path):
        """A PRODUCER-side re-verification counter. Reading it as usage
        would turn a silent loop into a healthy-looking one — and both live
        rows carry 7 and 2."""
        home = self._home(tmp_path)
        res = L._yield_graduated_skills(home)
        assert res.invoked == 0
        assert res.invoked != 9

    def test_the_note_says_it_is_NOT_helpfulness(self, tmp_path):
        """`invoked` on this view is otherwise read as value delivered, and
        unlike lessons this store has no outcome arm."""
        home = self._home(tmp_path)
        s = GraduatedSkillStore(home / "system" / "memory")
        book_graduated_retrieval(_ctx("user"), s, ["h1"])
        note = L._yield_graduated_skills(home).note
        assert "NOT helpfulness" in note
        assert "no outcome arm" in note
        means = {p.name: p.invoked_means
                 for p in L.YIELD_PROBES}["skills.graduated"]
        assert "NOT helpfulness" in means

    def test_an_EMPTY_store_is_EMPTY_and_says_why(self, tmp_path):
        home = self._home(tmp_path, {})
        res = L._yield_graduated_skills(home)
        assert res.minted == 0
        assert _yield_status(res) == EMPTY
        # The dedicated branch exists for the MESSAGE: falling through the
        # general path renders "0 of 0 graduated skill(s) have NEVER been
        # surfaced", which reads as a finding about skills that do not
        # exist.
        assert res.note == "no skill has graduated yet"

    def test_a_missing_store_is_still_NO_SOURCE(self, tmp_path):
        (tmp_path / "system" / "memory").mkdir(parents=True)
        assert _yield_status(L._yield_graduated_skills(tmp_path)) == L.NO_SOURCE
