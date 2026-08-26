"""The GEPA gate/judge contract, enforced against the real files.

⚠ THIS FILE EXISTS BECAUSE §4DA DID NOT CONVERGE IN SIXTEEN REVIEW ROUNDS,
and the reason was structural. The ship *rule* settled by round 3. Almost
every defect found in rounds 4-16 was one shape: **a concept with no single
definition, restated in each of the files that touch it, with two of the
restatements disagreeing.**

  round 4   a corpus gap given the semantics of an outage
  round 5   the lesson never went upstream
  round 7   the gate stamped seven fields the only reader could not open
  round 9   the veto arm nobody gave the outage handling to
  round 11  the un-stamping was one-armed
  round 12  the fail-open was one-armed too
  round 13  a set-level win stamped on each member as its own
  round 14  the `excluded` bucket left outside the era filter
  round 15  four instruments, four exit vocabularies
  round 16  two gates, two seed-arm schemas — and the judge read one

Each review round samples ONE pair of restatements. There are O(n²) pairs
and the graph regrows whenever anything is added, so review cannot close
it: round 16's own fix reintroduced round 7's defect verbatim, in the seed
arm it had just ported.

These tests do not sample. They enumerate every writer and every reader by
AST and fail on ANY divergence — which is the difference between a review
that converges and one that does not.
"""

import ast
import json
from pathlib import Path

import pytest

from ghost_agent.optim import gate_contract as GC

GATES = ("scripts/run_gepa.py", "scripts/optimize_tool_descriptions.py")
JUDGES = ("scripts/recheck_gepa_incumbent.py", "scripts/gepa_live_check.py")


def _tree(path):
    return ast.parse(Path(path).read_text())


def _dict_keys(node):
    """Literal string keys of a dict node, INCLUDING those inside `**`
    unpacks and conditional unpacks.

    ⚠ A KEY REACHED THROUGH `**({...} if c else {...})` IS STILL WRITTEN.
    The first version of this scan missed them and reported `co_promoted`
    — which the tool-description gate does write, conditionally — as a
    key no gate emits. A conformance test whose scan is narrower than the
    code it scans manufactures its own findings, which is the
    `harness-grades-own-homework` shape this file exists to avoid.
    """
    keys = []
    for k, v in zip(node.keys, node.values):
        if k is None:                      # `**expr`
            for sub in ast.walk(v):
                if isinstance(sub, ast.Dict):
                    keys += _dict_keys(sub)
            continue
        if isinstance(getattr(k, "value", None), str):
            keys.append(k.value)
    return keys


def _own_scope(fn):
    """Walk a function's OWN body — nested def/lambda scopes excluded.

    ⚠ A closure's `return` is not an exit code. The first version of the
    fail-closed return scan used `ast.walk(fn)` and flagged `_ab_runner`'s
    dict return as a smuggled exit; a scan that cries wolf gets deleted,
    which is worse than one with a blind spot."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _gate_record_dicts(path):
    """Every dict literal that is a `gate` block: keyed by `metric`, the
    field both gates have carried since §4CY."""
    out = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Dict):
            continue
        keys = _dict_keys(node)
        if "metric" in keys:
            out.append(keys)
    assert out, f"no gate-record literal found in {path}"
    return out


class TestEveryKeyAWriterEmitsHasAHome:
    """⚠ Round 7: the gate stamped seven evidence fields FLAT while the
    only reader looked for them nested under `gate`, so the audit trail
    §4DA added specifically so an override could be re-examined was
    written in a shape nothing could open. Round 16: two seed-arm
    vocabularies. Both are "a writer named something on its own"."""

    @pytest.mark.parametrize("path", GATES)
    def test_no_gate_invents_a_key(self, path):
        for keys in _gate_record_dicts(path):
            unknown = sorted(set(keys) - GC.GATE_RECORD_READABLE_KEYS)
            assert not unknown, (
                f"{path} writes gate-record key(s) no reader opens: "
                f"{unknown}")

    #: String keys the judges read that are NOT gate-record fields — the
    #: artifact's own top level, JSON payload plumbing, env vars. A key
    #: appearing in a judge that is in NEITHER this list NOR the gate
    #: vocabulary fails the scan and forces a decision — fail-CLOSED on
    #: new names, which is the property the first version lacked: it
    #: tracked four receiver spellings, so a read through `_g2 = ...` (or
    #: a `prev["x"]` subscript) was simply invisible (lens A, F3-B).
    NON_GATE_READS = {
        "scripts/recheck_gepa_incumbent.py": {
            "GHOST_HOME", "GHOST_UPSTREAM_URL",
            "baseline_instruction", "optimized_instruction",
            "signature_name", "gate", "gate_arm",
            "hand_written_baseline", "seeded_from_live_artifact",
            "baseline_meta", "candidate_meta", "failure_reason",
            "expected_output", "inputs", "user_request",
            "choices", "message", "content", "prompt",
        },
        "scripts/gepa_live_check.py": {
            "GHOST_HOME", "optimized_instruction",
        },
    }

    @pytest.mark.parametrize("path", JUDGES)
    def test_no_judge_reads_a_key_nobody_writes(self, path):
        """The mirror. A reader asking for a name no writer emits is a
        branch that can never fire — round 16's `_sa.get("overridden")`
        against a gate that wrote `seed_loss_overridden`. Receiver-
        agnostic: EVERY string-literal `.get()` and subscript read in the
        file is collected, then partitioned into gate-vocabulary reads
        (which must have a writer) and known non-gate reads (the
        allowlist above); anything in neither fails."""
        written = set()
        for g in GATES:
            for keys in _gate_record_dicts(g):
                written |= set(keys)
        written |= set(GC.SEED_ARM_KEYS)
        # Legacy shapes the shared reader deliberately still understands.
        written |= {"hand_written_pass_rate", "seed_loss_overridden"}
        # Keys the gates stamp OUTSIDE the gate dict literal.
        written |= {"gate_arm", "gate_arm_candidate"}
        read = set()
        for node in ast.walk(_tree(path)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "get"
                    and node.args
                    and isinstance(getattr(node.args[0], "value", None),
                                   str)):
                read.add(node.args[0].value)
            if (isinstance(node, ast.Subscript)
                    and isinstance(getattr(node.slice, "value", None), str)
                    and isinstance(node.ctx, ast.Load)):
                read.add(node.slice.value)
        allow = self.NON_GATE_READS[path]
        unknown = sorted(read - written - allow)
        assert not unknown, (
            f"{path} reads key(s) that are neither gate vocabulary nor "
            f"allowlisted non-gate reads: {unknown}. If a gate writes it, "
            f"register it; if it is artifact plumbing, allowlist it — "
            f"either way, decide.")
        orphan_gate_reads = sorted(
            (read - allow) & GC.GATE_RECORD_READABLE_KEYS - written)
        assert not orphan_gate_reads, (
            f"{path} reads gate-record key(s) no gate writes: "
            f"{orphan_gate_reads} — a branch that can never fire")


class TestTheSeedArmHasOneSchema:
    """⚠ THE ROUND-16 DEFECT, MEASURED. `run_gepa.py` wrote
    `overridden`/`seed_pass_rate`/`seed_wins`; the tool-description gate
    wrote `seed_loss_overridden`/`hand_written_pass_rate`/`vetoed`; and
    `recheck_gepa_incumbent.py` read `overridden` — so its
    "THAT PROMOTION USED --allow-seed-loss" warning was structurally
    unreachable for every artifact the second gate writes."""

    @pytest.mark.parametrize("path", GATES)
    def test_both_gates_build_it_rather_than_writing_a_literal(self, path):
        src = Path(path).read_text()
        assert "gate_contract.build_seed_arm(" in src, (
            f"{path} hand-writes its seed-arm block; two hand-written "
            f"blocks is how the schemas drifted apart")

    def test_the_builder_and_the_reader_agree_on_every_key(self):
        built = GC.build_seed_arm(
            seed_pass_rate=0.5, candidate_pass_rate=0.4,
            seed_minus_candidate_delta=0.1,
            seed_minus_candidate_raw_delta=0.08, n_usable_pairs=40,
            transport_excluded=2, seed_wins=6, candidate_wins=1,
            p_value=0.03, vetoed=True, overridden=True)
        back = GC.read_seed_arm({"seed_arm": built})
        assert back == built, (built, back)

    def test_an_override_of_nothing_is_refused(self):
        with pytest.raises(ValueError):
            GC.build_seed_arm(
                seed_pass_rate=0.5, candidate_pass_rate=0.6,
                seed_minus_candidate_delta=-0.1,
                seed_minus_candidate_raw_delta=-0.1, n_usable_pairs=40,
                vetoed=False, overridden=True)

    def test_the_reader_understands_BOTH_pre_contract_shapes(self):
        """Files written before the contract existed must not become
        unreadable — that would be the same defect one migration later."""
        old_gate1 = {"seed_arm": {"seed_pass_rate": 0.5,
                                  "candidate_pass_rate": 0.4,
                                  "overridden": True}}
        old_gate2 = {"seed_arm": {"hand_written_pass_rate": 0.5,
                                  "candidate_pass_rate": 0.4,
                                  "seed_loss_overridden": True}}
        for blk in (old_gate1, old_gate2):
            got = GC.read_seed_arm(blk)
            assert got["seed_pass_rate"] == 0.5, (blk, got)
            assert got["overridden"] is True, (blk, got)
            assert got["vetoed"] is True, (blk, got)

    def test_it_does_not_GUESS_the_veto_from_a_sign(self):
        """⚠ An earlier draft inferred `vetoed` from `delta > 0` — wrong
        for exactly the legacy files it was meant to help, because the
        two gates recorded that quantity with OPPOSITE signs. Unknown
        stays unknown."""
        got = GC.read_seed_arm({"seed_arm": {
            "seed_pass_rate": 0.5, "candidate_pass_rate": 0.4,
            "seed_minus_candidate_delta": 0.1}})
        assert got["vetoed"] is None, got


class TestTheTwoExitContractsAreNamedAndDistinct:
    """⚠ Round 15's journal claimed all four instruments share ONE exit
    contract. They cannot: a GATE's 0 means the incumbent was REPLACED, a
    JUDGE's 0 means it STANDS. What all four share is 2."""

    def test_the_shared_code_is_could_not_measure(self):
        assert GC.GateExit.COULD_NOT_MEASURE == 2
        assert GC.JudgeExit.COULD_NOT_MEASURE == 2
        assert GC.COULD_NOT_MEASURE == 2

    def test_the_two_zeros_mean_opposite_things(self):
        assert GC.GateExit.PROMOTED == 0 and GC.JudgeExit.STILL_WINS == 0
        assert GC.GateExit.REJECTED == 1
        assert GC.JudgeExit.NO_LONGER_WINS == 1

    @pytest.mark.parametrize("path", GATES + JUDGES)
    def test_every_instrument_can_say_could_not_measure(self, path):
        """The one code all four share must be reachable in all four."""
        fn = next((n for n in ast.walk(_tree(path))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == "main"), None)
        assert fn is not None, path
        codes = {ast.unparse(n.value) for n in _own_scope(fn)
                 if isinstance(n, ast.Return) and n.value is not None}
        assert any(c == "2" for c in codes), (path, codes)


class TestNoInstrumentReturnsAnUndeclaredCode:
    """⚠ A fifth code invented in one file is the next round's finding.
    Every literal `main()` returns must be one of the four its contract
    declares."""

    @pytest.mark.parametrize("path,contract", (
        [(p, GC.GateExit) for p in GATES]
        + [(p, GC.JudgeExit) for p in JUDGES]))
    def test_the_literal_returns_are_all_declared(self, path, contract):
        declared = {v for k, v in vars(contract).items()
                    if not k.startswith("_") and isinstance(v, int)}
        fn = next(n for n in ast.walk(_tree(path))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        bad = []
        for n in _own_scope(fn):
            if not (isinstance(n, ast.Return) and n.value is not None):
                continue
            for lit in ast.walk(n.value):
                if (isinstance(lit, ast.Constant)
                        and isinstance(lit.value, int)
                        and not isinstance(lit.value, bool)):
                    if lit.value not in declared:
                        bad.append((n.lineno, lit.value))
        assert not bad, (
            f"{path} returns exit code(s) its contract does not declare: "
            f"{bad}. Declared: {sorted(declared)}")

    @pytest.mark.parametrize("path", GATES + JUDGES)
    def test_main_returns_only_expressions_the_scan_can_SEE(self, path):
        """⚠ Lens A, F4 — both driven: `_rc = 9; return _rc` and
        `raise SystemExit(4)` walked straight past the literal scan,
        executed, and delivered undeclared codes to the caller. So the
        scan is made FAIL-CLOSED instead of wider: `main()` may return
        only expressions built from literals (constants, conditionals,
        comparisons, boolean ops) — a Name, Call or Attribute anywhere in
        a return value is refused, as is any `sys.exit`/`raise
        SystemExit` inside `main()`. An undeclared code then cannot be
        SMUGGLED; it has to show up as a literal, where the test above
        sees it."""
        fn = next(n for n in ast.walk(_tree(path))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        def _visible(expr):
            """True iff every VALUE position is a literal. Conditions
            (`test`) may reference anything; the values a caller receives
            may not — `return 2 if x else _rc` smuggles `_rc`."""
            if isinstance(expr, ast.Constant):
                return isinstance(expr.value, int) \
                    and not isinstance(expr.value, bool)
            if isinstance(expr, ast.IfExp):
                return _visible(expr.body) and _visible(expr.orelse)
            return False

        bad = []
        for n in _own_scope(fn):
            if isinstance(n, ast.Return) and n.value is not None:
                if not _visible(n.value):
                    bad.append((n.lineno, "opaque return: "
                                + ast.unparse(n.value)))
            if (isinstance(n, ast.Raise) and n.exc is not None
                    and "SystemExit" in ast.unparse(n.exc)):
                bad.append((n.lineno, "raise SystemExit inside main"))
            if (isinstance(n, ast.Call)
                    and ast.unparse(n.func).endswith("sys.exit")):
                bad.append((n.lineno, "sys.exit inside main"))
        assert not bad, (
            f"{path} main() delivers an exit code the declared-literal "
            f"scan cannot see: {bad}")

    @pytest.mark.parametrize("path,contract", (
        [(p, GC.GateExit) for p in GATES]
        + [(p, GC.JudgeExit) for p in JUDGES]))
    def test_helper_SystemExits_carry_declared_codes(self, path, contract):
        """⚠ Final pass, finding 3 — one call frame out. `_load_fixtures`
        raised `SystemExit(<string>)`, which exits **1** — a measured
        rejection — for a missing fixture pool, and since the live pool
        has been `.notready` for weeks that was the DEFAULT invocation's
        code. The main-scope scan cannot see a helper. Whole file:
        every `raise SystemExit(x)` must carry a DECLARED literal, except
        the `__main__` idiom `SystemExit(main())`/`SystemExit(
        asyncio.run(main()))`."""
        declared = {v for k, v in vars(contract).items()
                    if not k.startswith("_") and isinstance(v, int)}
        bad = []
        for n in ast.walk(_tree(path)):
            if not (isinstance(n, ast.Raise) and n.exc is not None):
                continue
            txt = ast.unparse(n.exc)
            if "SystemExit" not in txt:
                continue
            if "main()" in txt:
                continue                       # the __main__ idiom
            args = getattr(n.exc, "args", None) or []
            if (len(args) == 1
                    and isinstance(args[0], ast.Constant)
                    and isinstance(args[0].value, int)
                    and args[0].value in declared):
                continue
            bad.append((n.lineno, txt))
        assert not bad, (
            f"{path} raises SystemExit with a value outside the declared "
            f"codes (a string exits 1 — a measured rejection): {bad}")


class TestTheEraScopingRuleCoversEveryRandomizedArm:
    """⚠ Round 14 added a third arm label, `excluded`, and left it
    OUTSIDE the era filter — so turns that busted a RETIRED artifact's
    ceiling were counted against whatever is live now, and the operator
    was told to shorten a prompt that was already short. The filter has
    to name every arm that is randomized, not the two someone thought of
    first."""

    def test_collect_scopes_every_randomized_arm_by_era(self):
        from types import SimpleNamespace

        from ghost_agent.optim import live_check as LC

        def _rows(arm, sha, n=6):
            return [SimpleNamespace(
                outcome="passed",
                extra={"optim_artifacts": {"s": {"sha": sha, "arm": arm}}})
                for _ in range(n)]

        # ⚠ DERIVED, NOT RESTATED. This list was the same hard-coded
        # 3-tuple as the code under test — one more of the O(n²)
        # restatements this file exists to remove, and the day a fourth
        # randomized arm is added, a restated list here would go green
        # while the era filter silently drops it (lens A, F5).
        # ⚠ AND THE CONTENT IS PINNED, NOT ONLY THE ITERATION. A test
        # that derives its arm list from the contract re-points itself
        # when the contract loses an arm — dropping EXCLUDED_ARM from
        # ERA_SCOPED_ARMS passed this whole file while un-scoping the
        # third arm (`self-calibrating-index-adapts`; pin BOTH halves of
        # every identity). `excluded` is era-scoped because round 14
        # drove the alternative: old-era ceiling turns counted against
        # the live artifact.
        assert set(GC.ERA_SCOPED_ARMS) == {"treatment", "control",
                                           "excluded"}, GC.ERA_SCOPED_ARMS
        assert set(GC.RANDOMIZED_ARMS) == {"treatment", "control"}
        for arm in GC.ERA_SCOPED_ARMS:
            cur = LC.collect(_rows(arm, "NEW"), "s", sha="NEW")
            old = LC.collect(_rows(arm, "OLD"), "s", sha="NEW")
            live = (cur.treatment.n + cur.control.n
                    + getattr(cur, "excluded", 0))
            stale = (old.treatment.n + old.control.n
                     + getattr(old, "excluded", 0))
            assert live == 6, (arm, live)
            assert stale == 0, (
                f"a turn stamped {arm!r} in a RETIRED artifact's era was "
                f"counted against the one that is live now")
            assert LC._stale(old) == 6, (arm, LC._stale(old))


class TestTheValidatorsActuallyEXECUTE:
    """⚠ Lens A, F2: `validate_gate_record` — the check the contract
    module's own docstring calls "THE POINT" — had ZERO calls anywhere in
    the test tree; deleting its body survived 821 tests, and so did
    disabling either half of `validate_seed_arm`'s schema check. A
    validator nobody executes is documentation (`pin-must-fail-somewhere`).
    """

    def test_an_invented_key_is_refused(self):
        with pytest.raises(ValueError, match="no reader opens"):
            GC.validate_gate_record({"metric": "m", "my_private_field": 1},
                                    writer="test")

    def test_a_registered_key_set_passes(self):
        GC.validate_gate_record(
            {k: None for k in GC.GATE_RECORD_SHARED_KEYS}, writer="test")

    def test_a_missing_key_is_tolerated_but_an_extra_is_not(self):
        GC.validate_gate_record({"metric": "m"}, writer="test")
        with pytest.raises(ValueError):
            GC.validate_gate_record({"metric": "m", "extra_key_x": 1},
                                    writer="test")

    def test_the_nested_seed_arm_is_validated_too(self):
        with pytest.raises(ValueError):
            GC.validate_gate_record(
                {"metric": "m",
                 "seed_arm": {"overridden": True, "vetoed": False}},
                writer="test")

    def test_seed_arm_schema_refuses_BOTH_halves(self):
        """M12/M13/M14: `or`→`and`, missing-half off, extra-half off —
        each survived. Both halves, separately."""
        ok = {k: None for k in GC.SEED_ARM_KEYS}
        with pytest.raises(ValueError, match="unknown"):
            GC.validate_seed_arm({**ok, "invented": 1})
        missing = dict(ok)
        missing.pop("seed_pass_rate")
        with pytest.raises(ValueError, match="missing"):
            GC.validate_seed_arm(missing)

    def test_vetoed_and_undecidable_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="undecidable"):
            GC.build_seed_arm(
                seed_pass_rate=0.5, candidate_pass_rate=0.4,
                seed_minus_candidate_delta=0.1,
                seed_minus_candidate_raw_delta=0.1, n_usable_pairs=40,
                vetoed=True, undecidable=True)

    def test_the_delta_rate_identity_is_ENFORCED(self):
        """⚠ Lens C, B1: the first artifact written through this schema
        carried swapped rate fields — `seed_pass_rate: 0.9,
        candidate_pass_rate: 1.0, seed_minus_candidate_delta: +0.1` — so
        an artifact promoted BECAUSE it lost recorded a win. The
        identity `delta == seed_rate - candidate_rate` is the only check
        that can see a swap; names cannot."""
        with pytest.raises(ValueError, match="mislabelled"):
            GC.validate_seed_arm({
                **{k: None for k in GC.SEED_ARM_KEYS},
                "seed_pass_rate": 0.9, "candidate_pass_rate": 1.0,
                "seed_minus_candidate_delta": 0.1})
        GC.validate_seed_arm({
            **{k: None for k in GC.SEED_ARM_KEYS},
            "seed_pass_rate": 1.0, "candidate_pass_rate": 0.9,
            "seed_minus_candidate_delta": 0.1})

    def test_p_value_precision_survives_the_builder(self):
        """M29: p rounded to ONE decimal survived — p=0.0156 recorded as
        0.0, 'significant at any bar' in the audit trail."""
        b = GC.build_seed_arm(
            seed_pass_rate=0.5, candidate_pass_rate=0.4,
            seed_minus_candidate_delta=0.1,
            seed_minus_candidate_raw_delta=0.1, n_usable_pairs=40,
            p_value=0.015625)
        assert b["p_value"] == 0.015625, b["p_value"]


class TestTheValidatorIsWiredIntoBothGates:
    """⚠ The other half of lens A's F2: a validator can be perfectly
    pinned as a FUNCTION and still be deleted from the CALL SITE — and in
    `run_gepa.py` its refusal was swallowed by the stamp's
    `except Exception`, downgrading a contract breach to the same
    'promoted without provenance' warning as a disk error. Driven: the
    validator is forced to raise and each gate must fail LOUDLY, not
    promote quietly."""

    def test_otd_refuses_to_promote_when_the_record_is_invalid(
            self, tmp_path, monkeypatch, capsys):
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        from ghost_agent.optim import gate_contract as _GC

        def _boom(*a, **kw):
            raise ValueError("contract violation (forced)")
        monkeypatch.setattr(_GC, "validate_gate_record", _boom)
        with pytest.raises(ValueError):
            _H()._run(tmp_path, monkeypatch, cand_wins=6, n_fixtures=70)
        capsys.readouterr()
        home = tmp_path / "home" / "system" / "optim"
        assert not list(home.glob("tool_description.*.json")), (
            "an artifact was promoted with a gate record the contract "
            "refused")

    def test_run_gepa_fails_LOUDLY_not_into_the_provenance_warning(
            self, tmp_path, capsys, monkeypatch):
        from ghost_agent.optim import gate_contract as _GC
        from tests.test_gepa_optim_reaudit import (
            _corpus, _drive, _result, _ships)

        def _boom(*a, **kw):
            raise ValueError("contract violation (forced)")
        monkeypatch.setattr(_GC, "validate_gate_record", _boom)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        # ⚠ WITH AN INCUMBENT ON DISK — the final pass drove the
        # refusal world and found the CANDIDATE already live: the stamp
        # was built after `os.replace`, so a validator refusal exited 2
        # ("nothing changed, re-run when stable") with production
        # switched and the next run gating against the unaudited
        # candidate. The stamp is built and validated BEFORE anything
        # moves now, and this pin checks the DISK, not just the code.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        _before = out.read_text()
        with pytest.raises(SystemExit) as _ei:
            _drive(["--signature", "planning.decompose",
                    "--trajectories", str(tmp_path / "traj"),
                    "--output", str(out), "--ab-min-delta", "0.05"],
                   gepa_result=_result(), comparison=_ships)
        # SystemExit(<string>) exits 1 — the rejected-candidate code; a
        # contract breach is COULD_NOT_MEASURE.
        assert _ei.value.code == 2, _ei.value.code
        err = capsys.readouterr().err
        assert "violates" in err, err
        assert "promoted without provenance" not in err, (
            "the contract refusal was swallowed into the I/O warning")
        assert out.read_text() == _before, (
            "a candidate whose gate record the contract refused is LIVE")
        assert Path(str(out) + ".candidate").exists(), (
            "the refused candidate was not kept in staging for "
            "post-mortem")


class TestTheArtifactOnDiskConformsNotJustTheSource:
    """⚠ Lens A, F3 — both driven past the AST scan: a subscript
    assignment AFTER the validate call (`_art["gate"]["x"] = ...`) and a
    computed key (`**{"audit" + "_note": ...}`) each landed an
    unregistered key in the live artifact with the conformance suite
    green. The AST scan reads source; this reads THE FILE THE GATE WROTE,
    which is what a reader will actually open. Guard the thing, not a
    proxy."""

    def test_the_otd_artifact_validates_and_round_trips(self, tmp_path,
                                                        monkeypatch,
                                                        capsys):
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70)
        capsys.readouterr()
        assert rc == 0 and live
        art = json.loads(live[0].read_text())
        GC.validate_gate_record(art["gate"], writer=str(live[0]))

    def test_the_run_gepa_artifact_validates_too(self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import (
            _corpus, _drive, _result, _ships)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        rc, _s = _drive(["--signature", "planning.decompose",
                         "--trajectories", str(tmp_path / "traj"),
                         "--output", str(out), "--ab-min-delta", "0.05"],
                        gepa_result=_result(), comparison=_ships)
        capsys.readouterr()
        assert rc == 0 and out.exists()
        art = json.loads(out.read_text())
        GC.validate_gate_record(art["gate"], writer=str(out))


class TestTheFifthInstrumentIsAnExplicitPerimeter:
    """⚠ Lens A, F1: `scripts/optimize_verifier.py` is a third ship gate
    outside the contract — no `gate` block, no shared vocabulary, its own
    exit semantics. It is excluded BY OPERATOR DECISION ("build #4 scoped
    for gepa; we might revisit GEPA autonomy later"), and an exclusion
    someone decided is different from one nobody noticed — so it is
    STATED here, and this test fails the day the file starts
    half-adopting the contract (the worst state: partial vocabulary,
    which is exactly two-schemas-again)."""

    def test_the_exclusion_is_still_total_not_partial(self):
        src = Path("scripts/optimize_verifier.py").read_text()
        adopted = [tok for tok in ("gate_contract", "GateExit",
                                   "build_seed_arm", "GATE_RECORD_",
                                   "validate_gate_record")
                   if tok in src]
        assert not adopted, (
            f"optimize_verifier.py has started adopting the contract "
            f"({adopted}) — finish the adoption and move it into GATES in "
            f"this file, or revert; a partial adoption is two schemas "
            f"again")
        # And it must not have grown a `gate` block — its records carry
        # their own vocabulary today (`private_incumbent_balanced`,
        # `scorer_version`, a prose `metric` in its BASELINE file), and
        # the dangerous state is a `"gate":` key readers would open with
        # the shared expectations.
        assert '"gate":' not in src, (
            "optimize_verifier.py now writes a gate block; bring it "
            "inside the conformance perimeter")


class TestTheGateMarkersHaveOneHomeToo:
    """§4DF round 1 (MAJOR-3): the gate banners/markers joined the
    contract and the conformance instrument did not — the sibling one
    revision behind, on the mechanism this module exists to provide.
    The scripts must PRINT through the constants; the constants must
    still be the operator-facing lines they were LIFTED from; and the
    launcher's `--fixtures` basename must be the miner's output name."""

    def test_run_gepa_prints_through_every_gate_constant(self):
        src = Path("scripts/run_gepa.py").read_text()
        for const in ("GATE_RUN_BANNER_GEPA", "GATE_PROMOTED_MARKER_GEPA",
                      "GATE_REJECTED_MARKER", "GATE_NO_CANDIDATE_MARKER"):
            assert f"gate_contract.{const}" in src, (
                f"run_gepa.py no longer prints through {const} — a "
                f"restated marker string is the shape-1 defect")

    def test_the_otd_gate_prints_through_every_gate_constant(self):
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        for const in ("GATE_RUN_BANNER_OTD", "GATE_PROMOTED_MARKER_OTD",
                      "GATE_REJECTED_MARKER", "GATE_NO_CANDIDATE_MARKER"):
            assert f"gate_contract.{const}" in src, const

    def test_the_lifted_values_did_not_drift(self):
        assert GC.GATE_REJECTED_MARKER == "A/B gate REJECTED"
        assert GC.GATE_PROMOTED_MARKER_GEPA == (
            "A/B gate PASSED — candidate promoted")
        assert GC.GATE_PROMOTED_MARKER_OTD == "PROMOTED "
        assert GC.GATE_NO_CANDIDATE_MARKER == "NO CANDIDATE"
        assert GC.GATE_RUN_BANNER_GEPA == "run_gepa: gating"
        assert GC.GATE_RUN_BANNER_OTD == "optimize_tool_descriptions: gating"

    def test_the_fixtures_basename_has_one_home(self):
        assert GC.TOOL_FIXTURES_BASENAME == "tool_choice_fixtures.jsonl"
        miner = Path("scripts/mine_tool_fixtures.py").read_text()
        assert "gate_contract.TOOL_FIXTURES_BASENAME" in miner
        from ghost_agent.optim import autonomy as A
        assert A._target_command("tool_descriptions", "/h")[1][1].endswith(
            GC.TOOL_FIXTURES_BASENAME)

    def test_the_upstream_default_has_one_home(self):
        """§4DF round 1, CRIT-1: 8080 (the TLS web console) in two files
        and 8088 (llama-server) in two others — and the launcher's
        deliberately-minimal argv made run_gepa's wrong default
        load-bearing for 3 of 4 targets. Every default now references
        `core.llm.DEFAULT_UPSTREAM_URL`."""
        from ghost_agent.core.llm import DEFAULT_UPSTREAM_URL
        assert DEFAULT_UPSTREAM_URL == "http://127.0.0.1:8088"
        for path, ref in (
                ("scripts/run_gepa.py", "default=DEFAULT_UPSTREAM_URL"),
                ("scripts/optimize_tool_descriptions.py",
                 "default=DEFAULT_UPSTREAM_URL"),
                ("src/ghost_agent/main.py", "default=_DEF_UP"),
                # §4DF round 2, MIN-6: the env-fallback restatements the
                # round-1 pin did not see — correct VALUE, drift hazard.
                ("scripts/recheck_gepa_incumbent.py",
                 '"GHOST_UPSTREAM_URL", DEFAULT_UPSTREAM_URL'),
                ("scripts/mine_failure_envs.py",
                 '"GHOST_UPSTREAM_URL", DEFAULT_UPSTREAM_URL'),
                ("src/ghost_agent/optim/run_gepa.py",
                 "DEFAULT_UPSTREAM_URL")):
            src = Path(path).read_text()
            assert ref in src, (path, ref)
            # No re-stated literal default may remain: the constant's
            # value appearing as a URL literal outside prose is drift
            # waiting to happen. (Usage examples in docstrings/help are
            # prose; argparse `default="http...` is not.)
            assert 'default="http://127.0.0.1:80' not in src, path

    def test_the_wired_defaults_EXECUTE_to_the_constant(self,
                                                        monkeypatch):
        """§4DF round 3 (MIN-2): the rows above are token scans — a
        mutant kept the token alive in a dead expression while rewiring
        the REAL default, and 113 tests stayed green. This one runs
        each script's main() far enough to build its parser and reads
        the default the parser will actually serve
        (`token-pins-vs-executed-pins`)."""
        import argparse
        import asyncio as _aio
        import importlib.util as _iu
        import sys

        from ghost_agent.core.llm import DEFAULT_UPSTREAM_URL
        captured = {}
        _orig = argparse.ArgumentParser.parse_args

        def _spy(self, *a, **k):
            for dest in ("upstream", "upstream_url"):
                d = self.get_default(dest)
                if d is not None:
                    captured["d"] = d
                    break
            raise SystemExit(0)
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _spy)
        monkeypatch.setattr(sys, "argv", ["x"])
        for rel, is_async in (
                ("scripts/run_gepa.py", True),
                ("scripts/optimize_tool_descriptions.py", False),
                ("scripts/recheck_gepa_incumbent.py", True),
                ("scripts/mine_failure_envs.py", True)):
            captured.clear()
            spec = _iu.spec_from_file_location(
                f"conf_{Path(rel).stem}", str(Path(rel).resolve()))
            mod = _iu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with pytest.raises(SystemExit):
                if is_async:
                    _aio.run(mod.main())
                else:
                    mod.main()
            assert captured.get("d") == DEFAULT_UPSTREAM_URL, (
                f"{rel}'s parser serves "
                f"{captured.get('d')!r} — the wired default and the "
                f"token the row above scans have diverged")
