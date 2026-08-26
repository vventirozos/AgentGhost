"""§4DA round 3 — the stamp that lied, the key that collided, the torn write.

Round 2 wired the tool-description read site into §4CZ's attribution so a
promoted artifact could be judged by production turns and reverted. Round 3's
production-safety lens found that the artefact round 2 shipped — the stamp —
can assert "this artifact served this turn" for turns where nothing reached
the model, that the randomization key collides on 17 of the 70 live tool
names, and that the promotion write ports half of the `run_gepa` discipline
its own comment cites.

All three corrupt the causal claim `--revert` acts on, which is the one thing
the round was for.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

from ghost_agent.core import experiments as EXP
from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R


# ══════════════════════════════════════════════════════════════════════
# MAJOR-1 — the stamp fired at LOAD time; the refusal happens two layers up
# ══════════════════════════════════════════════════════════════════════
class TestTheStampMatchesWhatTheModelACTUALLY_saw:
    @staticmethod
    def _artifact(home, tool, text, *, gate=True):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        payload = {"signature_name": f"tool_description.{tool}",
                   "optimized_instruction": text}
        if gate:
            payload["gate_arm"] = "tool-choice fidelity A/B, private holdout"
        (d / f"tool_description.{tool}.json").write_text(json.dumps(payload))

    @staticmethod
    def _build(tools, req_id):
        from ghost_agent.utils.logging import request_id_context
        token = request_id_context.set(req_id)
        try:
            return R._apply_tuned_descriptions(tools, context=object())
        finally:
            request_id_context.reset(token)

    def _tools(self, names):
        out = []
        for n in names:
            t = next(t for t in R.TOOL_DEFINITIONS
                     if (t.get("function") or {}).get("name") == n)
            out.append({"type": "function",
                        "function": {"name": n,
                                     "description":
                                         t["function"]["description"],
                                     "parameters": {}}})
        return out

    @pytest.fixture(autouse=True)
    def _clean(self):
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        yield
        L.clear_cache()
        R._TUNED_DESC_NAMES = None

    def test_the_AGGREGATE_reject_leaves_no_served_stamp(self, tmp_path,
                                                         monkeypatch):
        """⚠ MEASURED: 8 individually-valid artifacts summing past the
        20,000 ceiling → 40 of 40 requests rendered baselines ONLY, while
        21 treatment turns carried the served-stamp and `gepa_live_check`
        returned a KEEP verdict comparing two arms whose prompts were
        BYTE-IDENTICAL. `activation_stats` already knew the truth
        (`applied: 0, rejected: 21`); only the stamp disagreed."""
        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:4]]
        for n in names:
            base = next(t for t in R.TOOL_DEFINITIONS
                        if t["function"]["name"] == n)["function"]["description"]
            self._artifact(home, n, base + " " + ("y" * 400))
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 100)
        R._TUNED_DESC_NAMES = None
        L.clear_cache()

        tools = self._tools(names)
        out = self._build(tools, "req-agg-1")
        for t_in, t_out in zip(tools, out):
            assert (t_out["function"]["description"]
                    == t_in["function"]["description"]), \
                "the ceiling did not fire — this test proves nothing"
        assert L.served_for_request("req-agg-1") == {}, (
            "a turn that saw only baselines was stamped as SERVED the "
            "artifact — live_check would compare identical prompts")

    def test_the_PER_TOOL_reject_leaves_no_served_stamp(self, tmp_path,
                                                        monkeypatch):
        """The other refusal point, which fires per tool rather than for
        the set."""
        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        n = R.TOOL_DEFINITIONS[0]["function"]["name"]
        # Far over any per-tool cap, so `_validate_tool_description` refuses.
        self._artifact(home, n, "z" * 50_000)
        R._TUNED_DESC_NAMES = None
        L.clear_cache()
        tools = self._tools([n])
        out = self._build(tools, "req-cap-1")
        assert (out[0]["function"]["description"]
                == tools[0]["function"]["description"]), \
            "the per-tool validator did not fire"
        assert L.served_for_request("req-cap-1") == {}

    def test_an_ACCEPTED_artifact_IS_stamped(self, tmp_path, monkeypatch):
        """⚠ THE ADMIT SIDE. Un-stamping everything closes the false
        positives and destroys the mechanism — `--revert` needs the stamp
        on the turns the artifact really served."""
        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        n = R.TOOL_DEFINITIONS[0]["function"]["name"]
        base = R.TOOL_DEFINITIONS[0]["function"]["description"]
        self._artifact(home, n, base + " Prefer it for current events.")
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 20_000)
        R._TUNED_DESC_NAMES = None
        L.clear_cache()
        tools = self._tools([n])
        out = self._build(tools, "req-ok-1")
        assert out[0]["function"]["description"] != \
            tools[0]["function"]["description"], "nothing was swapped"
        served = L.served_for_request("req-ok-1")
        assert f"tool_description.{n}" in served, (
            "a genuinely-served artifact lost its stamp — --revert has "
            "nothing to judge")

    def test_unnote_leaves_OTHER_signatures_alone(self):
        """One refused tool must not clear a sibling's stamp."""
        L._SERVED_RING.clear()
        L._note_served("r1", "tool_description.a", "sha", "treatment")
        L._note_served("r1", "tool_description.b", "sha", "treatment")
        L.unnote_served("r1", "tool_description.a")
        assert L.served_for_request("r1") == {
            "tool_description.b": {"sha": "sha", "arm": "treatment"}}
        L.unnote_served("r1", "tool_description.b")
        assert L.served_for_request("r1") == {}
        L._SERVED_RING.clear()

    def test_unnote_never_raises(self):
        for args in (("", "x"), ("nope", "x"), ("r", ""), (None, None)):
            L.unnote_served(*args)


# ══════════════════════════════════════════════════════════════════════
# MAJOR-2 — the randomization key collided on live names
# ══════════════════════════════════════════════════════════════════════
class TestTheExperimentNameIsUNIQUE:
    """⚠ `_NAME_RE` caps at 40 chars and `gepa_` + `tool_description_`
    spends 22, leaving 18 characters of tool name. The 39 static tools are
    clean; the read site also covers COMPOSED SKILLS, and over the 70 live
    names a bare truncation produced 7 collision groups covering 17 names.
    Colliding signatures are never independently randomized — control
    withholds BOTH and treatment serves BOTH — so `--revert` on one can
    retire an artifact whose measured loss belongs to the other."""

    COLLIDING = [
        "tool_description.auto_file_system_file_system_execute",
        "tool_description.auto_file_system_file_system_report_pdf",
        "tool_description.auto_file_system_manage_projects",
        "tool_description.auto_file_system_manage_services",
        "tool_description.auto_file_system_file_system_manage_services",
    ]

    def test_long_names_do_not_collide(self):
        seen = {}
        for n in self.COLLIDING:
            e = L.experiment_name(n)
            assert e not in seen, (
                f"{n} and {seen[e]} share the experiment name {e} — one "
                f"--revert would retire the other's artifact")
            seen[e] = n

    def test_every_name_still_matches_the_REAL_registry_regex(self):
        """A unique name the registry SKIPS is not an improvement — the
        first version of `experiment_name` returned dots and the whole
        randomized arm was inert."""
        for n in self.COLLIDING + ["planning.decompose",
                                   "tool_description.web_search"]:
            e = L.experiment_name(n)
            assert EXP._NAME_RE.match(e), f"{e!r} ({len(e)} chars)"
            assert len(e) <= 40

    def test_a_REAL_registry_round_trip(self, tmp_path, monkeypatch):
        """Driven through `load_registry`, which SILENTLY SKIPS a spec
        whose name fails the regex."""
        home = tmp_path / "home"
        (home / "system").mkdir(parents=True)
        names = [L.experiment_name(n) for n in self.COLLIDING]
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t",
            "experiments": [{"name": e, "arms": ["control", "treatment"],
                             "traffic": 1.0, "enabled": True}
                            for e in names]}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        EXP.reset_registry_cache()
        reg = EXP.load_registry(ghost_home=home)
        live = set(reg.names_for_scope(EXP.SCOPE_LIVE))
        for e in names:
            assert e in live, f"the registry rejected {e!r}"
        assert len(live & set(names)) == len(self.COLLIDING), \
            "two signatures collapsed into one experiment"
        EXP.reset_registry_cache()

    def test_short_names_are_UNCHANGED(self):
        """The three GEPA signatures already registered must keep the
        names an operator may already have in experiments.json."""
        assert L.experiment_name("planning.decompose") == \
            "gepa_planning_decompose"
        assert L.experiment_name("tool_selection.pick") == \
            "gepa_tool_selection_pick"
        assert L.experiment_name("tool_description.web_search") == \
            "gepa_tool_description_web_search"

    def test_the_suffix_is_derived_from_the_WHOLE_name(self):
        """Two names differing only past the truncation point must differ."""
        a = L.experiment_name("tool_description." + "x" * 40 + "aaa")
        b = L.experiment_name("tool_description." + "x" * 40 + "bbb")
        assert a != b


# ══════════════════════════════════════════════════════════════════════
# MAJOR-4 — "" from a REGISTERED experiment is control, not "serve it"
# ══════════════════════════════════════════════════════════════════════
class TestALostArmIsUNENROLLED_notControl:
    """⚠ ROUND 6 REVERSED THIS, AND THE REVERSAL IS THE LESSON.

    Round 3 read `arm_for`'s "Consumers MUST treat '' as the control
    path" and made this loader return "control" whenever the experiment
    was REGISTERED AND ENABLED. That is a PROXY for "was this turn
    enrolled", and the two differ exactly where it matters: `assign`
    returns "" for a unit outside `traffic`, `names_for_scope` ignores
    traffic entirely. Measured over 400 requests at `traffic: 0.2` — 308
    un-enrolled turns were served the BASELINE and stamped `control`, so
    `live_check` compared treatment 30/46 against a control arm inflated
    to 354 and returned **REVERT at p=0.0195**, where the real randomized
    46-vs-46 comparison is **KEEP at p=0.2485**. A false REVERT retiring a
    live artifact, with `unenrolled` reading 0 so CONFOUNDED could never
    fire. A ramped rollout also withheld the artifact from the un-enrolled
    majority, and `traffic: 0` disabled it for 100% of traffic.

    "" means NOT ENROLLED FOR THIS REQUEST — control is spelled
    "control". Outside the experiment the artifact serves everything (the
    pre-experiment status quo) and the turn is stamped `unenrolled`,
    which `live_check` buckets into NEITHER arm. That also covers the
    eviction case round 3 was reaching for: an evicted request drops out
    of the comparison instead of polluting one side of it."""

    @pytest.fixture
    def registered(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "tool-choice fidelity A/B, private holdout"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t",
            "experiments": [{"name": L.experiment_name(sig),
                             "arms": ["control", "treatment"],
                             "traffic": 1.0, "enabled": True}]}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        EXP.reset_registry_cache()
        L.clear_cache()
        yield sig
        EXP.reset_registry_cache()
        L.clear_cache()

    def test_an_EVICTED_arm_serves_the_artifact_and_is_stamped_unenrolled(
            self, registered):
        """A context that knows nothing about this request — what an
        evicted ring slot, and an un-enrolled turn under `traffic < 1`,
        both look like to `arm_for`."""
        out = L.tuned_instruction(registered, "BASELINE",
                                  context=object(), req_id="lost-req")
        assert out == "TUNED", (
            "a turn outside the experiment was WITHHELD the artifact — "
            "that is a silent behaviour change for every un-enrolled "
            "request, which under `traffic: 0` is all of them")

    def test_and_is_stamped_neither_control_nor_treatment(self,
                                                          registered):
        """The stamp is what `live_check` buckets on, and an un-enrolled
        turn must land in NEITHER arm or it inflates one of them."""
        L.tuned_instruction(registered, "BASELINE",
                            context=object(), req_id="lost-req-2")
        served = L.served_for_request("lost-req-2")
        arm = (served.get(registered) or {}).get("arm", "")
        assert arm == "unenrolled", f"stamped {arm!r}"

    def test_a_REAL_enrolment_still_randomizes(self, registered):
        """The admit side — the whole point of the plumbing."""
        from ghost_agent.core import experiments as _EXP
        arms = set()
        for i in range(40):
            ctx = type("C", (), {})()
            req = f"enrolled-{i}"
            _EXP.enroll_request(ctx, req)
            L.tuned_instruction(registered, "BASELINE",
                                context=ctx, req_id=req)
            got = (L.served_for_request(req) or {}).get(registered) or {}
            if got.get("arm"):
                arms.add(got["arm"])
        assert arms == {"control", "treatment"}, arms

    def test_with_NO_experiment_registered_the_artifact_still_serves(
            self, tmp_path, monkeypatch):
        """⚠ THE ADMIT SIDE, AND THE PRE-EXISTING BEHAVIOUR. Treating
        every "" as control would withhold the artifact from every turn on
        an agent with no experiments file — a silent global disable."""
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "x"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        EXP.reset_registry_cache()
        L.clear_cache()
        out = L.tuned_instruction(sig, "BASELINE", context=object(),
                                  req_id="unenrolled-req")
        assert out == "TUNED", (
            "with no registered experiment the artifact must serve "
            "everything — that is the un-randomized status quo")
        EXP.reset_registry_cache()
        L.clear_cache()


# ══════════════════════════════════════════════════════════════════════
# MAJOR-3 — the promotion write was not atomic
# ══════════════════════════════════════════════════════════════════════
class TestThePromotionWriteIsATOMIC:
    def test_it_stages_and_replaces(self, tmp_path, monkeypatch):
        """⚠ A TORN `write_text` LEAVES INVALID JSON, and `loader.py`
        caches the failure as `None` for the life of the PROCESS — so
        repairing the file on disk does not bring the signature back, and
        the only trace is a `logger.debug`. `run_gepa.py:799` does
        `os.replace`; round 2 ported the backup half of that discipline
        and not the atomic half."""
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as H)
        seen = {"replace": 0, "staged": []}
        real_replace = os.replace

        def _spy(src, dst):
            seen["replace"] += 1
            seen["staged"].append((str(src), str(dst)))
            return real_replace(src, dst)
        monkeypatch.setattr(os, "replace", _spy)
        rc, live, _r, _n = H()._run(tmp_path, monkeypatch, cand_wins=6)
        assert rc == 0 and live
        assert seen["replace"] >= 1, "the artifact was written in place"
        src, dst = seen["staged"][-1]
        assert src.endswith(".staging") and dst.endswith(".json")

    def test_no_staging_file_is_left_behind(self, tmp_path, monkeypatch):
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as H)
        rc, live, _r, _n = H()._run(tmp_path, monkeypatch, cand_wins=6)
        optim = tmp_path / "home" / "system" / "optim"
        assert not list(optim.glob("*.staging")), list(optim.iterdir())

    def test_a_reader_never_sees_a_PARTIAL_artifact(self, tmp_path,
                                                    monkeypatch):
        """The property `os.replace` buys, stated as behaviour: at every
        moment the live path either does not exist or is valid JSON."""
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as H)
        optim = tmp_path / "home" / "system" / "optim"
        real_replace = os.replace
        checked = {"n": 0}

        def _spy(src, dst):
            for p in optim.glob("tool_description.*.json"):
                json.loads(p.read_text())      # must not raise mid-write
                checked["n"] += 1
            return real_replace(src, dst)
        H()._run(tmp_path, monkeypatch, cand_wins=6)
        monkeypatch.setattr(os, "replace", _spy)
        rc, live, _r, _n = H()._run(tmp_path, monkeypatch, cand_wins=6)
        assert rc == 0
        assert checked["n"] >= 1, "the second promotion saw no live file"


class TestTheRetirementMessageIsRightForBOTH_readSites:
    def test_the_tool_description_paragraph_names_the_tool_block(self):
        src = Path("scripts/gepa_live_check.py").read_text()
        assert "every TOOL-BLOCK BUILD keeps using the retired" in src
        assert "every planner turn keeps using the retired" in src, \
            "the prepend case still applies to the planner signatures"
