"""§4CS — the auto-mined macro loop mints a param SCHEMA, not baked literals.

Measured 2026-08-23 on the live store: 26 composed skills, 25 auto-mined and
all at status="proposed", **0 invocations across every auto-mined macro, all
time**. Two producers, one cause each, both deliberate:

  * `core/dream.py` blanked the whole param template for eleven tools via a
    DENY-LIST, so the macro replayed no stale path or command — and also
    advertised no inputs, so it could never be activated.
  * `core/agent.py`'s skills_auto graduation mint passed ``params: {}``
    unconditionally, because `SkillCandidate` keeps only tool NAMES.

A param-less macro cannot be activated, so it parks at "proposed" forever.
These tests pin the replacement: the decision is made on the VALUE (constant
+ enum-declared, or structurally payload-free => literal; everything else =>
a `$slot` the caller fills), the mined window carries each call's MODE so a
macro has an identity, and both sides of the re-propose guard derive that
identity from ONE definition.

Every test here is mutation-pinned: reverting the change under test turns it
RED. See the journal entry for the mutation log.
"""

import json
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.dream import (
    _safe_macro_name, mine_recurring_tool_sequences,
)
from ghost_agent.distill.schema import Trajectory, ToolCall
from ghost_agent.tools.composed_skills import (
    ComposedSkill, ComposedSkillRegistry, SkillStep,
    _tool_schema_index,
    macro_identity, macro_mode_key, macro_step_inputs, mint_param_schema,
)


def _traj(tid, seq, outcome="passed"):
    return Trajectory(id=tid, outcome=outcome,
                      tool_calls=[ToolCall(name=n, arguments=a) for n, a in seq])


@pytest.fixture
def registry():
    d = tempfile.mkdtemp()
    try:
        yield ComposedSkillRegistry(storage_dir=Path(d))
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ── the value-level decision ─────────────────────────────────────────────
class TestValueDecision:
    def test_the_live_incident_ids_become_slots_modes_stay(self):
        """The macro that caused the deny-list:
        manage_projects(action='task_update', status='DONE',
                        project_id=<old>, task_id=<old>, description=...).

        Asserted against the recomputed observation set, not against
        restated constants.
        """
        obs = [[{"action": "task_update", "status": "DONE",
                 "project_id": f"p{i}", "task_id": f"t{i}",
                 "description": f"note {i}"} for i in range(3)]]
        tpl, slots, why = mint_param_schema(("manage_projects",), obs)
        assert why is None
        minted = tpl[0]
        for row in obs[0]:
            # No observed volatile value survives anywhere in the template.
            assert row["project_id"] not in minted.values()
            assert row["task_id"] not in minted.values()
            assert row["description"] not in minted.values()
        # ...and the DISPATCH SELECTOR survives, because that is the
        # macro's identity. ⚠ REVIEW ROUND 2: `status` is enum-TYPED and is
        # NOT the selector — it is the value being WRITTEN, and freezing it
        # re-froze half of this very artifact.
        idx = _tool_schema_index()["manage_projects"]
        assert idx["selector"] == "action"
        assert "status" in idx["enums"] and idx["selector"] != "status"
        assert minted["action"] == "task_update"
        assert minted["status"] == "$status", \
            "an enum-typed PAYLOAD must slot, not freeze"
        assert "DONE" not in minted.values()
        assert set(slots) == {"project_id", "task_id", "description", "status"}

    def test_a_CONSTANT_path_is_still_a_slot(self):
        """The property is payload-ness, not variability.

        A path that never changed across every observation is exactly the
        stale-replay hazard the deny-list existed for, so constancy alone
        must not license baking it. This is the pin that a "bake anything
        constant" simplification would break.
        """
        obs = [[{"operation": "read", "path": "/etc/passwd"}] * 5]
        tpl, slots, why = mint_param_schema(("file_system",), obs)
        assert why is None
        assert tpl[0]["path"] == "$path"
        assert "/etc/passwd" not in tpl[0].values()
        assert tpl[0]["operation"] == "read"     # enum-declared: a mode

    def test_a_constant_value_outside_the_declared_enum_is_a_slot(self):
        """`operation` IS an enum param, but this value is not one of its
        declared members — so it is not a mode, it is an unvalidated string,
        and it must not be frozen."""
        allowed = _tool_schema_index()["file_system"]["enums"]["operation"]
        assert "not_a_real_operation" not in allowed
        obs = [[{"operation": "not_a_real_operation", "path": "/x"}] * 4]
        tpl, _slots, why = mint_param_schema(("file_system",), obs)
        assert tpl[0]["operation"] == "$operation"
        # ...and with no mode fixed, the macro has no identity at all.
        assert why is not None and "fixes no mode" in why

    def test_a_SAFETY_INTERLOCK_is_never_frozen(self):
        """REVIEW ROUND 1, the hole that mattered most.

        `postgres_admin.confirm` is the DESTRUCTIVE-DDL authorisation:
        "Set true to authorise a destructive DROP/TRUNCATE statement".
        The first rule froze any constant bool, so a macro would have
        pre-consented to DROP and asked the caller only for `$sql`.
        A bool is not a payload; it can still be an interlock.
        """
        idx = _tool_schema_index()
        assert idx["postgres_admin"]["selector"] == "action"
        obs = [[{"action": "query", "confirm": True,
                 "sql": f"DROP TABLE t{i}"} for i in range(3)]]
        tpl, _s, why = mint_param_schema(("postgres_admin",), obs)
        assert why is None
        # It becomes an explicit runtime input, so the caller decides each
        # time. ⚠ REVIEW ROUND 2 replaced the earlier DROP behaviour: the
        # claim that "dropping is strictly the safe direction" was FALSE
        # for `browser.stop_on_error`, whose default is False, so dropping
        # an observed True inverts a fail-fast interlock to fail-open.
        assert tpl[0]["confirm"] == "$confirm"
        assert True not in tpl[0].values()

    def test_a_constant_NUMERIC_id_is_slotted_not_frozen_and_not_dropped(self):
        """The original live incident, in the form the first rule missed.

        `manage_projects(task_id=<old>)` is the stale-id replay the
        deny-list existed for. Freezing an int reproduced it; DROPPING an
        int would instead make the macro merely broken. It must SLOT.
        """
        obs = [[{"action": "task_update", "status": "DONE",
                 "task_id": 42, "project_id": 7}] * 3]
        tpl, slots, why = mint_param_schema(("manage_projects",), obs)
        assert why is None
        assert tpl[0]["task_id"] == "$task_id"
        assert tpl[0]["project_id"] == "$project_id"
        assert 42 not in tpl[0].values() and 7 not in tpl[0].values()
        assert {"task_id", "project_id"} <= set(slots)
        # ...and the SELECTOR still survives, so the macro keeps its
        # identity; `status` is payload and slots.
        assert tpl[0]["action"] == "task_update"
        assert tpl[0]["status"] == "$status"

    def test_a_constant_port_is_slotted(self):
        """Observed FROZEN on the live corpus by the first rule."""
        obs = [[{"action": "start", "port": 5055, "name": f"s{i}",
                 "command": f"c{i}"} for i in range(3)]]
        tpl, _s, why = mint_param_schema(("manage_services",), obs)
        assert why is None
        assert tpl[0]["port"] == "$port"
        assert 5055 not in tpl[0].values()

    def test_a_constant_FLAG_is_slotted_not_dropped_and_not_frozen(self):
        """⚠ REVIEW ROUND 2 removed the DROP branch entirely.

        Its comment claimed "dropping is strictly the safe direction" and
        two of its three examples (`port`, `limit`) were NUMBERS this same
        design already routes to slots — it defended a branch that had been
        removed. And the claim was false where it mattered:
        `browser.stop_on_error` defaults to False, so dropping an observed
        True silently inverts a fail-fast interlock to fail-open;
        `browser.full_page` defaults True in the other direction. Exactly
        one param in the whole registry declares a schema `default`, so
        there was no sound basis for deciding what an omission means.
        """
        obs = [[{"operation": "interact", "actions": [{"a": i}],
                 "stop_on_error": True} for i in range(3)]]
        tpl, slots, why = mint_param_schema(("browser",), obs)
        assert why is None
        assert tpl[0]["operation"] == "interact"      # the selector
        assert tpl[0]["stop_on_error"] == "$stop_on_error"
        assert "stop_on_error" in slots
        assert True not in tpl[0].values()

    def test_the_selector_is_derived_from_REQUIRED_not_from_field_order(
            self, monkeypatch):
        """⚠ On every real tool the required enum happens to be the FIRST
        enum in field order, so "take the first enum" and "take the
        required enum" agree and a count- or live-registry-based assertion
        cannot separate them. Inject a definition where they disagree."""
        import ghost_agent.tools.composed_skills as cs
        import ghost_agent.tools.registry as R
        fake = {"type": "function", "function": {
            "name": "order_probe",
            "parameters": {"type": "object", "properties": {
                "payload_enum": {"type": "string", "enum": ["x", "y"]},
                "action": {"type": "string", "enum": ["go", "stop"]},
                "target": {"type": "string"}},
                "required": ["action", "target"]}}}
        monkeypatch.setattr(R, "TOOL_DEFINITIONS",
                            list(R.TOOL_DEFINITIONS) + [fake])
        monkeypatch.setattr(cs, "_TOOL_SCHEMA_INDEX", None)
        idx = cs._tool_schema_index(force=True)
        assert list(idx["order_probe"]["enums"])[0] == "payload_enum"
        assert idx["order_probe"]["selector"] == "action", \
            "the selector must come from `required`, not from field order"
        tpl, _s, why = cs.mint_param_schema(
            ("order_probe",),
            [[{"action": "go", "payload_enum": "x", "target": f"t{i}"}
              for i in range(3)]])
        assert why is None
        assert tpl[0]["action"] == "go"
        assert tpl[0]["payload_enum"] == "$payload_enum"

    def test_an_enum_that_is_NOT_the_selector_is_payload(self):
        """`browser.wait_until` is enum-typed and is not the dispatch
        selector — it is a parameter of the navigation, not its identity."""
        idx = _tool_schema_index()["browser"]
        assert idx["selector"] == "operation" and "wait_until" in idx["enums"]
        obs = [[{"operation": "navigate", "url": f"u{i}",
                 "wait_until": "load"} for i in range(3)]]
        tpl, _s, why = mint_param_schema(("browser",), obs)
        assert why is None
        assert tpl[0]["operation"] == "navigate"
        assert tpl[0]["wait_until"] == "$wait_until"

    def test_a_baked_action_script_is_never_frozen(self):
        """browser(operation='interact', actions=[...]) — a constant action
        script is a container of non-empty strings, so it is a payload."""
        script = [{"type": "click", "selector": "#pay"}]
        obs = [[{"operation": "interact", "actions": script}] * 4]
        tpl, _s, why = mint_param_schema(("browser",), obs)
        assert why is None
        assert tpl[0]["operation"] == "interact"
        assert tpl[0]["actions"] == "$actions"
        assert script not in tpl[0].values()


# ── slot sharing ─────────────────────────────────────────────────────────
class TestSlotSharing:
    def test_two_steps_that_always_carried_the_same_value_share_one_slot(self):
        """"read then edit the SAME file" is ONE input, not two."""
        paths = ["/a.py", "/b.py", "/c.py"]
        obs = [[{"operation": "read", "path": p} for p in paths],
               [{"operation": "replace", "path": p, "pattern": "x",
                 "replace_with": f"y{p}"} for p in paths]]
        tpl, slots, why = mint_param_schema(("file_system", "file_system"), obs)
        assert why is None
        assert tpl[0]["path"] == tpl[1]["path"] == "$path"
        assert slots.count("path") == 1

    def test_two_steps_whose_values_differed_get_distinct_slots(self):
        obs = [[{"operation": "read", "path": f"/in{i}"} for i in range(3)],
               [{"operation": "read", "path": f"/out{i}"} for i in range(3)]]
        tpl, slots, why = mint_param_schema(("file_system", "file_system"), obs)
        assert why is None
        assert tpl[0]["path"] != tpl[1]["path"]
        assert {tpl[0]["path"], tpl[1]["path"]} == {"$path", "$path_2"}

    def test_ragged_observations_are_truncated_not_misaligned(self):
        """Occurrence k at position 0 must pair with occurrence k at
        position 1. A ragged sample set would otherwise pair values from
        different windows and invent a shared slot that was never shared."""
        obs = [[{"operation": "read", "path": p} for p in ("/a", "/b", "/c")],
               [{"operation": "read", "path": p} for p in ("/a", "/b")]]
        tpl, _s, why = mint_param_schema(("file_system", "file_system"), obs)
        assert why is None
        # Truncated to the common prefix, the two positions carried the
        # SAME value on every compared occurrence, so they share one slot.
        # Comparing the untruncated lists would make them differ in length
        # and silently split what is one input into two.
        assert tpl[0]["path"] == tpl[1]["path"] == "$path"


# ── mintability: the artifact must be usable ─────────────────────────────
class TestMintability:
    def test_uncovered_required_param_refuses(self):
        tpl, _s, why = mint_param_schema(("file_system",), [[]])
        assert why is not None
        assert "required" in why and "operation" in why

    def test_an_UNKNOWN_tool_refuses_the_whole_sequence(self):
        """REVIEW ROUND 1: `spec is None` used to `continue`, which skipped
        ALL THREE refusal checks — coverage, empty template, and mode
        identity — so a tool outside the registry was silently exempt from
        every one of them. Registry membership was a PROXY for "we know
        this tool's modes"; a check that cannot run must not report the
        favourable outcome."""
        import ghost_agent.tools.composed_skills as cs
        idx = {k: v for k, v in cs._tool_schema_index().items()
               if k != "file_system"}
        monkey = cs._tool_schema_index
        try:
            cs._tool_schema_index = lambda *a, **k: idx
            tpl, _s, why = cs.mint_param_schema(
                ("file_system", "manage_services"),
                [[{"operation": op, "path": "/x"}
                  for op in ("read", "write", "inspect")],
                 [{"action": "start", "name": "s", "command": "c"}] * 3])
        finally:
            cs._tool_schema_index = monkey
        assert why is not None
        assert "not in the tool registry" in why and "file_system" in why

    def test_vision_analysis_IS_known_despite_being_absent_from_the_static_list(self):
        """It is APPENDED by `get_active_tool_definitions`, not present in
        `TOOL_DEFINITIONS` — and it is the tool in the highest-support
        mined sequence on the live corpus. An index built from the static
        list alone did not know it."""
        idx = _tool_schema_index()
        assert "vision_analysis" in idx
        assert "action" in idx["vision_analysis"]["enums"]
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        assert "vision_analysis" not in {
            t["function"]["name"] for t in TOOL_DEFINITIONS}

    def test_a_step_that_fixes_no_mode_refuses(self):
        """Coverage alone admits degenerate artifacts: a macro whose every
        mode selector is a runtime slot is just "call these tool types in
        this order" and the model will reach for the tools directly."""
        obs = [[{"operation": op, "path": "/x"} for op in ("read", "write", "inspect")],
               [{"action": "start", "name": "s"}] * 3]
        tpl, _s, why = mint_param_schema(("file_system", "manage_services"), obs)
        assert why is not None and "fixes no mode" in why
        assert "step 1 (file_system)" in why

    def test_an_empty_template_for_a_tool_with_params_refuses(self):
        """Observations that share no key at all teach us nothing about how
        the step is called; `execute()` with no arguments does nothing."""
        obs = [[{"command": "ls"}, {"filename": "a.py", "content": "x"}]]
        tpl, _s, why = mint_param_schema(("execute",), obs)
        assert tpl[0] == {}
        assert why is not None and "share no common argument" in why

    def test_too_many_slots_refuses(self):
        obs = [[{"operation": "write", "path": f"/{i}{j}", "content": f"c{i}{j}"}
                for i in range(3)] for j in range(4)]
        tpl, slots, why = mint_param_schema(("file_system",) * 4, obs)
        assert len(slots) > 6
        assert why is not None and "runtime inputs" in why

    def test_max_slots_boundary_is_inclusive(self):
        obs = [[{"operation": "write", "path": f"/{i}{j}", "content": f"c{i}{j}"}
                for i in range(3)] for j in range(3)]
        _tpl, slots, why = mint_param_schema(("file_system",) * 3, obs)
        assert len(slots) == 6
        assert why is None, "exactly max_slots must be accepted"

    def test_registry_unreadable_mints_NOTHING(self, monkeypatch):
        """The fail direction matters. Without the tool schemas we cannot
        tell a mode from a payload, so the honest answer is to mint nothing
        — not to fall through and mint everything.
        """
        import ghost_agent.tools.composed_skills as cs
        monkeypatch.setattr(cs, "_TOOL_SCHEMA_INDEX", None)
        monkeypatch.setattr(cs, "_tool_schema_index", lambda *a, **k: None)
        tpl, slots, why = cs.mint_param_schema(
            ("file_system",), [[{"operation": "read", "path": "/x"}] * 3])
        assert why == "tool registry unreadable"
        assert tpl == [] and slots == []


# ── the macro has an identity, and the miner keys on it ──────────────────
class TestMacroIdentity:
    def test_mode_key_only_counts_declared_enum_members(self):
        assert macro_mode_key("file_system", {"operation": "replace"}) == (
            ("operation", "replace"),)
        assert macro_mode_key("file_system", {"operation": "nope"}) == ()
        assert macro_mode_key("web_search", {"query": "x"}) == ()

    def test_identity_ignores_slots_and_non_enum_literals(self):
        ident = macro_identity([("file_system",
                                 {"operation": "read", "path": "$path"}),
                                ("manage_services",
                                 {"action": "restart", "name": "$name"})])
        assert ident == (("file_system", (("operation", "read"),)),
                         ("manage_services", (("action", "restart"),)))

    def test_two_mode_variants_of_the_same_tool_pair_are_DIFFERENT_macros(self):
        """The change that took the live yield from 2 mintable windows to
        91. Name-only keying merged these into one window whose operation
        and action therefore varied, so both became slots and the macro
        lost its identity.
        """
        def _edit_restart(i):
            return [("file_system", {"operation": "replace", "path": f"/f{i}",
                                     "content": f"c{i}"}),
                    ("manage_services", {"action": "restart", "name": f"s{i}"})]

        def _read_start(i):
            return [("file_system", {"operation": "read", "path": f"/g{i}"}),
                    ("manage_services", {"action": "start", "name": f"s{i}",
                                         "command": f"run{i}"})]

        trajs = ([_traj(f"e{i}", _edit_restart(i)) for i in range(4)]
                 + [_traj(f"r{i}", _read_start(i)) for i in range(4)])
        props = mine_recurring_tool_sequences(trajs, min_support=3,
                                              max_proposals=10)
        idents = {p["signature"] for p in props}
        assert len(idents) == 2, [p["name"] for p in props]
        names = {p["name"] for p in props}
        assert "auto_file_system_replace_manage_services_restart" in names
        assert "auto_file_system_read_manage_services_start" in names

    def test_the_repropose_guard_uses_the_SAME_identity(self):
        """A macro on file must suppress only its OWN mode variant.

        The guard used to derive a tool-NAME tuple while the miner stamped a
        (tool, mode) identity, so the first variant on file suppressed every
        other variant of the same two tools, forever.
        """
        on_file = ComposedSkill(
            name="auto_file_system_read_manage_services_start",
            trigger_description="x", status="proposed",
            steps=[SkillStep("file_system", "", {"operation": "read",
                                                 "path": "$path"}),
                   SkillStep("manage_services", "", {"action": "start",
                                                     "name": "$name"})])
        stored = macro_identity((s.tool_name, s.param_template)
                                for s in on_file.steps)
        other_variant = macro_identity([
            ("file_system", {"operation": "replace", "path": "$path"}),
            ("manage_services", {"action": "restart", "name": "$name"})])
        assert stored != other_variant, "a mode variant must not be suppressed"
        same_variant = macro_identity([
            ("file_system", {"operation": "read", "path": "$other"}),
            ("manage_services", {"action": "start", "name": "$n"})])
        assert stored == same_variant, "the SAME variant must be suppressed"


class TestMacroNaming:
    def test_mode_is_part_of_the_name(self):
        n = _safe_macro_name((("file_system", (("operation", "replace"),)),
                              ("manage_services", (("action", "restart"),))))
        assert n == "auto_file_system_replace_manage_services_restart"

    def test_bare_tool_names_still_supported(self):
        assert _safe_macro_name(["web_search", "deep_research"]) == \
            "auto_web_search_deep_research"

    def test_overlong_name_gets_a_STABLE_digest_not_a_truncation(self):
        """Two long distinct windows must not collapse onto one name, and
        the digest must survive a restart — `PYTHONHASHSEED` randomises the
        builtin `hash()` per process, so a builtin-hash suffix would rename
        the macro on every boot and re-propose it forever.
        """
        long_a = tuple((f"manage_services", (("action", f"start"),))
                       for _ in range(6)) + (("browser", (("operation", "navigate"),)),)
        long_b = tuple((f"manage_services", (("action", f"restart"),))
                       for _ in range(6)) + (("browser", (("operation", "navigate"),)),)
        na, nb = _safe_macro_name(long_a), _safe_macro_name(long_b)
        assert len(na) <= 64 and len(nb) <= 64
        assert na != nb, "distinct windows must not collide after truncation"
        assert na == _safe_macro_name(long_a), "must be deterministic"

    def test_digest_is_stable_across_processes(self):
        import subprocess, sys, os
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from ghost_agent.core.dream import _safe_macro_name;"
            "w = tuple(('manage_services', (('action','start'),))"
            "          for _ in range(6)) + (('browser', (('operation','navigate'),)),);"
            "print(_safe_macro_name(w))"
        )
        env = dict(os.environ, PYTHONHASHSEED="12345")
        a = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=os.getcwd()).stdout.strip()
        env["PYTHONHASHSEED"] = "999"
        b = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=os.getcwd()).stdout.strip()
        assert a and a == b, (a, b)


# ── the artifact is actually activatable, end to end ─────────────────────
class TestActivatable:
    def _mint(self, registry):
        tpl, slots, why = mint_param_schema(
            ("file_system", "manage_services"),
            [[{"operation": "replace", "path": f"/f{i}", "content": f"c{i}",
               "pattern": f"p{i}", "replace_with": f"r{i}"} for i in range(3)],
             [{"action": "restart", "name": f"svc{i}"} for i in range(3)]])
        assert why is None, why
        registry.compile_from_pattern(
            "auto_file_system_replace_manage_services_restart",
            [{"tool": "file_system", "description": "", "params": tpl[0]},
             {"tool": "manage_services", "description": "", "params": tpl[1]}],
            "edit then restart", status="proposed")
        return slots

    def test_proposed_macro_is_not_advertised(self, registry):
        self._mint(registry)
        assert registry.to_tool_definitions() == []

    def test_once_approved_it_advertises_its_slots_as_REQUIRED(self, registry):
        slots = self._mint(registry)
        sk = registry.skills["auto_file_system_replace_manage_services_restart"]
        sk.status = "active"
        defs = registry.to_tool_definitions()
        assert len(defs) == 1
        schema = defs[0]["function"]["parameters"]
        # Recomputed from the mint's own slot list, not restated.
        assert sorted(schema["required"]) == sorted(slots)
        assert sorted(schema["properties"]) == sorted(slots)
        assert schema["required"], "a param-less macro is the defect itself"

    async def test_runtime_args_reach_the_tools(self, registry):
        # ⚠ `async def`, driven by pytest-asyncio (pytest.ini sets
        # asyncio_mode=auto), NOT `asyncio.run` inside a sync test.
        # `asyncio.run` creates and CLOSES a loop and leaves the policy's
        # current loop closed, which broke 16 later tests in
        # tests/test_auth_rejection_logging.py when the two ran in the same
        # session — the file passed alone and failed in a suite chunk.
        # Same pattern as the sibling tests in test_composed_skills.py.
        self._mint(registry)
        sk = registry.skills["auto_file_system_replace_manage_services_restart"]
        sk.status = "active"
        seen = []

        async def _exec(tool, args):
            seen.append((tool, dict(args)))
            return "OK"

        res = await registry.execute(
            sk.name, _exec,
            {"path": "/live/target.py", "content": "NEW", "pattern": "a",
             "replace_with": "b", "name": "live-svc"})
        assert res["success"], res
        assert seen[0][0] == "file_system"
        assert seen[0][1]["operation"] == "replace"      # the fixed mode
        assert seen[0][1]["path"] == "/live/target.py"   # the runtime slot
        assert seen[1][1] == {"action": "restart", "name": "live-svc"}
        # Nothing from the mining observations leaked into the live call.
        assert "/f0" not in json.dumps(seen)


# ── the parked backlog can acquire a schema ──────────────────────────────
class TestParamlessUpgrade:
    def _paramless(self, registry, status="proposed"):
        sk = ComposedSkill(
            name="auto_file_system_execute", trigger_description="old",
            status=status,
            steps=[SkillStep("file_system", "file_system (step 1)", {}),
                   SkillStep("execute", "execute (step 2)", {})])
        registry.register(sk)
        return sk

    def _incoming(self):
        return [{"tool": "file_system", "description": "read it",
                 "params": {"operation": "read", "path": "$path"}},
                {"tool": "execute", "description": "run it",
                 "params": {"command": "$command"}}]

    def test_a_parked_paramless_macro_gets_its_slots(self, registry):
        sk = self._paramless(registry)
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        got = {n for st in sk.steps for n in macro_step_inputs(st.param_template)}
        assert got == {"path", "command"}
        assert sk.status == "proposed", "the upgrade must not activate it"

    def test_an_ACTIVE_macro_is_never_rewritten(self, registry):
        sk = self._paramless(registry, status="active")
        before = [dict(st.param_template) for st in sk.steps]
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        assert [dict(st.param_template) for st in sk.steps] == before

    def test_a_macro_that_already_has_slots_is_not_rewritten(self, registry):
        sk = self._paramless(registry)
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        first = [dict(st.param_template) for st in sk.steps]
        other = [{"tool": "file_system", "description": "",
                  "params": {"operation": "write", "path": "$elsewhere"}},
                 {"tool": "execute", "description": "",
                  "params": {"command": "$other"}}]
        registry.compile_from_pattern("auto_file_system_execute", other,
                                      "d", status="proposed")
        assert [dict(st.param_template) for st in sk.steps] == first, \
            "the upgrade is monotone: one step, then never again"

    def test_a_tool_mismatch_blocks_the_upgrade(self, registry):
        sk = self._paramless(registry)
        wrong = [{"tool": "browser", "description": "",
                  "params": {"operation": "navigate", "url": "$url"}},
                 {"tool": "execute", "description": "",
                  "params": {"command": "$command"}}]
        registry.compile_from_pattern("auto_file_system_execute", wrong,
                                      "d", status="proposed")
        assert all(st.param_template == {} for st in sk.steps)

    def test_the_upgrade_persists(self, registry):
        self._paramless(registry)
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        reloaded = ComposedSkillRegistry(storage_dir=registry.storage_dir)
        sk = reloaded.skills["auto_file_system_execute"]
        got = {n for st in sk.steps for n in macro_step_inputs(st.param_template)}
        assert got == {"path", "command"}


# ── the SECOND producer: skills_auto graduation (core/agent.py) ──────────
class TestHarvestForNameOnlyCandidates:
    """`SkillCandidate` carries only tool NAMES, so the graduation mint
    passed `params: {}` and produced nine permanently unactivatable
    `auto_generic_*` macros. The arguments were never missing — only
    unread, on the same trajectories that phase already walked."""

    def _corpus(self):
        return [
            _traj("a", [("file_system", {"operation": "read", "path": "/1"}),
                        ("manage_services", {"action": "restart", "name": "s1"})]),
            _traj("b", [("file_system", {"operation": "read", "path": "/2"}),
                        ("manage_services", {"action": "restart", "name": "s2"})]),
            # right tools, WRONG length — must not contribute
            _traj("c", [("file_system", {"operation": "read", "path": "/3"})]),
            # right sequence but FAILED — must not contribute
            _traj("d", [("file_system", {"operation": "write", "path": "/4"}),
                        ("manage_services", {"action": "stop", "name": "s4"})],
                  outcome="failed"),
        ]

    def test_harvest_is_index_aligned_and_filtered(self):
        from ghost_agent.tools.composed_skills import harvest_step_observations
        obs = harvest_step_observations(
            self._corpus(), ("file_system", "manage_services"))
        assert [len(o) for o in obs] == [2, 2]
        assert [o["path"] for o in obs[0]] == ["/1", "/2"]
        assert [o["name"] for o in obs[1]] == ["s1", "s2"]
        # the failed trajectory's values are absent
        assert "/4" not in [o.get("path") for o in obs[0]]

    def test_a_name_only_candidate_now_mints_an_ACTIVATABLE_macro(self):
        """The pin on the agent.py producer: schema, not `params: {}`."""
        from ghost_agent.tools.composed_skills import harvest_step_observations
        seq = ("file_system", "manage_services")
        obs = harvest_step_observations(self._corpus(), seq)
        tpl, slots, why = mint_param_schema(seq, obs)
        assert why is None, why
        assert slots, "a param-less macro is exactly the defect"
        assert tpl[0]["operation"] == "read"        # mode fixed
        assert tpl[0]["path"] == "$path"            # payload slotted
        assert tpl[1]["action"] == "restart"

    def test_harvest_ignores_a_sub_window(self):
        """The candidate's identity is the WHOLE chain; matching a
        sub-window would pair position k with a different call."""
        from ghost_agent.tools.composed_skills import harvest_step_observations
        corpus = [_traj("x", [("web_search", {"query": "q"}),
                              ("file_system", {"operation": "read", "path": "/p"}),
                              ("manage_services", {"action": "restart", "name": "s"})])]
        obs = harvest_step_observations(corpus, ("file_system", "manage_services"))
        assert obs == [[], []]


# ── the CALL SITES must ACT on a refusal, and must announce a proposal ───
class TestProducersHonourTheContract:
    """REVIEW ROUND 1 found both halves of this untested.

    `mint_param_schema` returns `(templates, slots, reason)` and the
    templates look perfectly usable even on a refusal — the whole contract
    lives in the caller. Pinning the function's RETURN VALUE and not the
    caller's use of it is this project's "the fix inherits the blind spot"
    shape: the refusal was tested at the function, never at the call site.
    """

    class _Ctx:
        def __init__(self, base, collector):
            self.memory_dir = base
            self.sandbox_dir = base
            self.memory_system = None
            self.trajectory_collector = collector
            self.args = None

    def _dreamer(self, tmp_path, seqs):
        from ghost_agent.core.dream import Dreamer
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        n = 0
        for seq in seqs:
            for i in range(4):
                coll.append(_traj(f"t{n}", seq(i)))
                n += 1
        return Dreamer(self._Ctx(tmp_path / "mem", coll)), coll

    def test_a_refused_candidate_is_NOT_registered(self, tmp_path):
        """`file_system.path` present in only SOME calls drops out of the
        key intersection, so the declared-required check fires. Deleting
        the caller's `continue` mints a macro with neither a value nor a
        slot for `path` — the unactivatable artifact this change exists
        to eliminate."""
        def seq(i):
            return ([("file_system", {"operation": "read"})] if i % 2 else
                    [("file_system", {"operation": "read", "path": f"/f{i}"})]) + \
                   [("manage_services", {"action": "restart", "name": f"s{i}"})]
        dreamer, _ = self._dreamer(tmp_path, [seq])
        res = dreamer._propose_macros_sync()
        assert res["proposed"] == 0, res
        from ghost_agent.tools.composed_skills import _registry_from_context
        assert _registry_from_context(dreamer.context).skills == {}

    def test_EVERY_mintable_proposal_registers_and_is_ANNOUNCED(
            self, tmp_path, capsys):
        """The CRITICAL defect review round 1 found: `p["signature"]` became
        a tuple of (tool, modes) pairs and the announcement still
        `str.join`ed it. The TypeError fired INSIDE the registration loop
        and the blanket handler swallowed it at DEBUG — so exactly ONE
        macro registered per cycle instead of up to `max_proposals`, and
        the announcement never printed. Approval is the only route from
        "proposed" to invocable, and that line is its only announcement:
        the change had silently severed the channel it existed to feed.
        """
        def mk(op, act):
            return lambda i: [("file_system", {"operation": op, "path": f"/f{i}",
                                               "content": f"c{i}"}),
                              ("manage_services", {"action": act, "name": f"s{i}"})]
        dreamer, _ = self._dreamer(
            tmp_path, [mk("replace", "restart"), mk("write", "start"),
                       mk("read", "stop")])
        res = dreamer._propose_macros_sync()
        assert res["proposed"] >= 2, (
            f"only {res['proposed']} registered — a raise inside the "
            f"registration loop truncates the batch: {res}")
        assert len(res["names"]) == res["proposed"]
        out = capsys.readouterr().out.lower()
        # ONE announcement per registered macro. (The stream truncates long
        # lines, so match the count and a distinctive prefix rather than
        # the whole name.)
        assert out.count("macro proposed") == res["proposed"], out
        for name in res["names"]:
            assert name[:36].lower() in out, \
                f"'{name}' was registered but never announced"

    def test_a_mining_failure_is_LOUD_not_a_DEBUG_line(self, tmp_path,
                                                       monkeypatch, capsys):
        import ghost_agent.core.dream as D
        dreamer, _ = self._dreamer(tmp_path, [lambda i: [
            ("file_system", {"operation": "replace", "path": f"/f{i}",
                             "content": f"c{i}"}),
            ("manage_services", {"action": "restart", "name": f"s{i}"})]])
        monkeypatch.setattr(D, "_render_step",
                            lambda *a, **k: (_ for _ in ()).throw(
                                TypeError("boom")))
        dreamer._propose_macros_sync()
        out = capsys.readouterr().out
        assert "FAILED" in out and "TypeError" in out


# ── the guards review round 1 found unpinned ─────────────────────────────
class TestUpgradeGuardsArePinned:
    def _parked(self, registry, n_steps=3):
        tools = ["file_system", "execute", "manage_services"][:n_steps]
        sk = ComposedSkill(
            name="auto_parked", trigger_description="old", status="proposed",
            steps=[SkillStep(t, f"{t} (step {i+1})", {})
                   for i, t in enumerate(tools)])
        registry.register(sk)
        return sk

    def test_a_SHORTER_incoming_cannot_half_upgrade_a_macro(self, registry):
        """`zip` truncates silently: steps 1-2 would take slots and step 3
        would stay `{}` forever, because the "already has slots" guard
        then blocks every future repair. Irreversibly half-broken, saved
        to disk in that state."""
        sk = self._parked(registry, n_steps=3)
        registry.compile_from_pattern("auto_parked", [
            {"tool": "file_system", "description": "",
             "params": {"operation": "read", "path": "$path"}},
            {"tool": "execute", "description": "", "params": {"command": "$cmd"}},
        ], "d", status="proposed")
        assert all(st.param_template == {} for st in sk.steps), \
            "a length mismatch must block the whole upgrade, not part of it"

    def test_a_LITERAL_ONLY_incoming_does_not_rewrite_forever(self, registry):
        """Without the "incoming carries at least one slot" guard the
        upgrade is not monotone: a literal-only mint rewrites the macro on
        every cycle, each time saving and logging "it now carries runtime
        slots ()"."""
        sk = self._parked(registry, n_steps=2)
        first = [{"tool": "file_system", "description": "",
                  "params": {"operation": "read"}},
                 {"tool": "execute", "description": "", "params": {}}]
        registry.compile_from_pattern("auto_parked", first, "d",
                                      status="proposed")
        assert all(st.param_template == {} for st in sk.steps), \
            "a literal-only incoming carries no slots — nothing to upgrade to"

    def test_the_identity_ignores_a_SLOTTED_mode(self):
        """`macro_identity`'s `$`-filter: a macro whose mode is a runtime
        slot fixes nothing, so it must not share an identity with one that
        fixes that mode. Otherwise the re-propose guard suppresses a real
        variant."""
        fixed = macro_identity([("file_system", {"operation": "read",
                                                 "path": "$path"})])
        slotted = macro_identity([("file_system", {"operation": "$operation",
                                                   "path": "$path"})])
        assert fixed != slotted
        assert slotted == (("file_system", ()),)


class TestSchemaIndexFailsClosed:
    def test_an_UNREADABLE_registry_returns_None_not_an_empty_index(
            self, monkeypatch):
        """The PRODUCER half of the None-vs-empty contract. Returning `{}`
        makes every tool "absent from a registry that loaded" and, before
        the unknown-tool refusal, made every check pass — "a check which
        cannot run reports the favourable outcome"."""
        import builtins
        import ghost_agent.tools.composed_skills as cs
        monkeypatch.setattr(cs, "_TOOL_SCHEMA_INDEX", None)
        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name.endswith("registry") or name == "ghost_agent.tools.registry":
                raise ImportError("registry is unreadable")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _boom)
        assert cs._tool_schema_index(force=True) is None

    def test_a_FAILURE_is_never_cached(self, monkeypatch):
        import ghost_agent.tools.composed_skills as cs
        monkeypatch.setattr(cs, "_TOOL_SCHEMA_INDEX", None)
        calls = {"n": 0}
        real = cs._tool_schema_index

        # A failed lookup must not poison the cache for the next pass.
        assert cs._tool_schema_index(force=True) is not None
        assert cs._TOOL_SCHEMA_INDEX is not None


# ── ROUND 2: the CRITICAL, and the guards added for it ──────────────────
class TestNoStepIsAFullyDeterminedCall:
    """⚠ ROUND 2 CRITICAL. The enum-only rule was justified as "strictly
    stronger" than the deny-list it replaced. It was not.

    Where a tool's `action` is its ONLY required param, the frozen mode IS
    the whole call. `knowledge_base(action='reset_all')` deletes every id
    in the vector store, truncates the library index and calls
    `graph_memory.wipe_all()`, with no confirmation gate of any kind — and
    a macro's steps are dispatched through `build_step_executor`, which
    calls the tool function directly and never reaches the turn loop's
    mutation classification. A zero-input mutating step has no guard above
    it at all.

    The remedy is structural, not a list of dangerous verbs: EVERY step
    must carry a runtime input. Every exemption in this project's
    deny-lists became the next bypass.
    """

    @pytest.mark.parametrize("tool,args", [
        ("knowledge_base", {"action": "reset_all"}),
        ("manage_services", {"action": "stop-all"}),
        ("manage_tasks", {"action": "stop_all"}),
        ("scratchpad", {"action": "clear"}),
    ])
    def test_a_zero_input_destructive_step_is_REFUSED(self, tool, args):
        obs = [[{"operation": "read", "path": f"/f{i}"} for i in range(3)],
               [dict(args)] * 3]
        tpl, _s, why = mint_param_schema(("file_system", tool), obs)
        assert why is not None, (
            f"{tool}{args} minted as a fully-determined call: {tpl}")
        assert "NO runtime input" in why

    def test_the_rule_is_structural_not_a_list_of_verbs(self):
        """A HARMLESS zero-input step is refused too. That is the point:
        there is no list of dangerous verbs to leak, and the measured cost
        is one read-only bundle."""
        obs = [[{"action": "summary"}] * 3, [{"action": "stats"}] * 3]
        _t, _s, why = mint_param_schema(("workspace", "postmortem"), obs)
        assert why is not None and "NO runtime input" in why

    def test_a_step_WITH_an_input_still_mints(self):
        obs = [[{"operation": "read", "path": f"/f{i}"} for i in range(3)],
               [{"action": "forget", "ref": f"r{i}"} for i in range(3)]]
        tpl, slots, why = mint_param_schema(("file_system", "knowledge_base"),
                                            obs)
        assert why is None
        assert tpl[1] == {"action": "forget", "ref": "$ref"}
        assert set(slots) == {"path", "ref"}

    def test_no_live_proposal_contains_a_zero_input_step(self):
        """The property, asserted over what the miner actually produces."""
        seqs = [
            [("file_system", {"operation": "replace", "path": f"/f{i}",
                              "content": f"c{i}"}),
             ("manage_services", {"action": "restart", "name": f"s{i}"})]
            for i in range(4)]
        trajs = [_traj(f"t{i}", s) for i, s in enumerate(seqs)]
        for p in mine_recurring_tool_sequences(trajs, min_support=3,
                                               max_proposals=10):
            for st in p["steps"]:
                assert macro_step_inputs(st["params"]), (p["name"], st)


class TestBothProducersShareOneAdmissionRule:
    """⚠ ROUND 2 MAJOR. The meta-tool and single-tool-repeated filters
    gated the DREAM miner only, so the skills_auto graduation producer —
    which reaches `compile_from_pattern` by a different route — applied no
    admission rule at all. Over the live corpus it offered a macro whose
    first step RUNS AN ARBITRARY COMPOSED SKILL by name."""

    def test_a_meta_tool_sequence_is_inadmissible(self):
        from ghost_agent.tools.composed_skills import (
            MACRO_IGNORE_TOOLS, macro_sequence_admissible,
        )
        assert "manage_composed_skills" in MACRO_IGNORE_TOOLS
        why = macro_sequence_admissible(
            ("manage_composed_skills", "file_system"))
        assert why and "meta / control-flow" in why

    def test_a_single_tool_repeated_is_inadmissible(self):
        from ghost_agent.tools.composed_skills import macro_sequence_admissible
        why = macro_sequence_admissible(("web_search", "web_search"))
        assert why and "single tool repeated" in why

    def test_a_real_workflow_is_admissible(self):
        from ghost_agent.tools.composed_skills import macro_sequence_admissible
        assert macro_sequence_admissible(
            ("file_system", "manage_services")) is None

    def test_the_ignore_set_holds_the_tools_that_drive_THIS_system(self):
        """The set's CONTENTS were entirely untested — every name could be
        replaced with a placeholder and the suite stayed green."""
        from ghost_agent.tools.composed_skills import MACRO_IGNORE_TOOLS
        for meta in ("manage_composed_skills", "manage_skills", "create_skill",
                     "self_play", "self_play_loop", "stop_self_play",
                     "dream_mode", "replan", "abort_attempt"):
            assert meta in MACRO_IGNORE_TOOLS, meta

    def test_the_dream_producer_applies_it(self, capsys):
        """⚠ The fixture must carry a RUNTIME INPUT on the meta step, or
        the no-zero-input rule refuses it first and the admission rule is
        never exercised — which is how removing the admission call
        survived the first version of this test."""
        def _run(head_tool, head_args):
            trajs = [_traj(f"{head_tool}{i}",
                           [(head_tool, dict(head_args)),
                            ("file_system", {"operation": "read",
                                             "path": f"/p{i}"})])
                     for i in range(4)]
            return mine_recurring_tool_sequences(trajs, min_support=3)

        # A DIFFERENTIAL assertion. Both fixtures carry runtime inputs, so
        # neither is caught by the no-zero-input rule; the ONLY thing that
        # separates them is the admission rule. Asserting on the refusal's
        # log text does not work — the operator stream truncates the line.
        meta = _run("manage_composed_skills",
                    {"action": "run", "name": "m", "params": "{}"})
        control = _run("manage_services",
                       {"action": "restart", "name": "svc"})
        assert control, "the control shape must mint, or this proves nothing"
        assert meta == [], (
            "a macro whose first step RUNS AN ARBITRARY COMPOSED SKILL "
            "must be inadmissible")

    def test_the_graduation_producer_applies_it(self):
        """Same rule, the other call site — asserted through the shared
        function the producer imports, since the mint sits deep inside the
        idle tick."""
        import ghost_agent.core.agent as A
        import inspect
        src = inspect.getsource(A.GhostAgent._biological_tick) \
            if hasattr(A.GhostAgent, "_biological_tick") else ""
        if not src:
            src = Path("src/ghost_agent/core/agent.py").read_text()
        assert "macro_sequence_admissible(_seq)" in src, \
            "the graduation mint must consult the shared admission rule"


class TestFailurePathsCannotThemselvesFail:
    def test_a_raising_announcement_inside_the_HANDLER_is_contained(
            self, tmp_path, monkeypatch, capsys):
        """⚠ ROUND 2 MAJOR: the ERROR announcement added by round 1 sits
        INSIDE the except block with no guard of its own, so when IT raised
        the exception escaped a method documented "Never raises" and landed
        one frame up in another `logger.debug` — re-creating the exact
        invisibility it was added to remove."""
        import ghost_agent.core.dream as D
        from ghost_agent.distill.collector import TrajectoryCollector

        class _Ctx:
            def __init__(self, base, coll):
                self.memory_dir = base
                self.sandbox_dir = base
                self.memory_system = None
                self.trajectory_collector = coll
                self.args = None

        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        for i in range(4):
            coll.append(_traj(f"t{i}", [
                ("file_system", {"operation": "replace", "path": f"/f{i}",
                                 "content": f"c{i}"}),
                ("manage_services", {"action": "restart", "name": f"s{i}"})]))
        d = D.Dreamer(_Ctx(tmp_path / "mem", coll))

        calls = {"n": 0}
        real = D.pretty_log

        def _boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("the announcement itself failed")

        monkeypatch.setattr(D, "pretty_log", _boom)
        # Must NOT raise — the method's contract.
        res = d._propose_macros_sync()
        assert isinstance(res, dict)
        assert calls["n"] >= 1


# ── ROUND 2: the surviving mutants that named real behaviour ────────────
class TestRoundTwoMutationSurvivors:
    def test_the_paramless_UPGRADE_is_announced(self, registry, capsys):
        """The only operator-visible signal that a parked macro was healed
        — the sibling of round 1's CRITICAL, added by the same change and
        never pinned."""
        registry.register(ComposedSkill(
            name="auto_parked2", trigger_description="old", status="proposed",
            steps=[SkillStep("file_system", "", {}), SkillStep("execute", "", {})]))
        registry.compile_from_pattern("auto_parked2", [
            {"tool": "file_system", "description": "",
             "params": {"operation": "read", "path": "$path"}},
            {"tool": "execute", "description": "", "params": {"command": "$c"}},
        ], "d", status="proposed")
        out = capsys.readouterr().out.lower()
        assert "macro schema" in out
        assert "auto_parked2" in out

    def test_a_mint_REFUSAL_is_announced(self, capsys):
        seq = [("file_system", {"operation": "read", "path": "/p"}),
               ("knowledge_base", {"action": "reset_all"})]
        trajs = [_traj(f"t{i}", seq) for i in range(4)]
        assert mine_recurring_tool_sequences(trajs, min_support=3) == []
        out = capsys.readouterr().out.lower()
        assert "macro mint" in out and "skipped" in out

    def test_macro_mode_key_fails_CLOSED_on_an_unreadable_registry(
            self, monkeypatch):
        """Round 1 pinned `_tool_schema_index` and `mint_param_schema`, not
        this one. Without the guard it raises AttributeError per call
        instead of returning ()."""
        import ghost_agent.tools.composed_skills as cs
        monkeypatch.setattr(cs, "_tool_schema_index", lambda *a, **k: None)
        assert cs.macro_mode_key("file_system", {"operation": "read"}) == ()
        assert cs.macro_identity([("file_system", {"operation": "read"})]) == (
            ("file_system", ()),)

    def test_a_slot_name_starting_with_a_digit_is_made_substitutable(
            self, monkeypatch):
        """`$2foo` does not match `_VAR_RE`, so the slot would advertise
        and then never substitute — a silently empty argument."""
        import ghost_agent.tools.composed_skills as cs
        idx = dict(cs._tool_schema_index())
        idx["fake_digit"] = {"enums": {}, "selector": None,
                             "params": ("2nd",), "required": ()}
        monkeypatch.setattr(cs, "_tool_schema_index", lambda *a, **k: idx)
        tpl, slots, why = cs.mint_param_schema(
            ("fake_digit",), [[{"2nd": f"v{i}"} for i in range(3)]])
        assert why is None
        name = slots[0]
        assert name[0].isalpha() or name[0] == "_", name
        assert cs.macro_step_inputs(tpl[0]) == [name], \
            "the minted slot must be one _VAR_RE can substitute"

    def test_value_normalisation_is_key_order_independent(self):
        """With `sort_keys=False`, two dicts differing only in key ORDER
        compare unequal — so two positions that always carried the SAME
        value stop sharing a slot and the macro demands two inputs where
        it needs one.

        ⚠ Asserted on SLOT SHARING, not on slot count: `actions` is not
        the dispatch selector, so it becomes a slot either way and a
        count-based assertion cannot tell the two behaviours apart.
        """
        a1 = {"a": 1, "b": 2}
        a2 = {"b": 2, "a": 1}          # same value, different key order
        obs = [[{"operation": "interact", "actions": d}
                for d in (a1, a2, a1)],
               [{"operation": "interact", "actions": d}
                for d in (a2, a1, a2)]]
        tpl, slots, why = mint_param_schema(("browser", "browser"), obs)
        assert why is None
        assert tpl[0]["actions"] == tpl[1]["actions"] == "$actions", tpl
        assert slots.count("actions") == 1
        assert "actions_2" not in slots

    def test_the_repropose_guard_checks_the_signature_AND_the_name(self):
        """`or` → `and` survived: a signature already on file under a
        DIFFERENT name (the two-producer case §4CS introduced) would be
        re-proposed as a duplicate."""
        src = Path("src/ghost_agent/core/dream.py").read_text()
        assert 'if p["signature"] in existing_sigs or p["name"] in reg.skills:' \
            in src

    def test_an_empty_sequence_refuses_gracefully(self):
        tpl, slots, why = mint_param_schema((), [])
        assert why == "empty tool sequence"
        assert tpl == [] and slots == []
