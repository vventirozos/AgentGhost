"""Ablation arms must be DISTINCT — and must not depend on a flag default.

The defect this pins (found 2026-08-09): `_treatment_flags` omitted
`--frontier-selfplay` and relied on it defaulting to True, with the comment
"Frontier-aware self-play is ON by default (the base treatment arm)". That was
true when written and became FALSE on 2026-07-09, when main.py flipped the
default to False.

From that day the "frontier" arm silently ran UNIFORM seeding — byte-identical
to the `treatment_uniform` arm it is compared against — so the #27b
frontier-vs-uniform question was comparing a configuration with ITSELF and
could only ever report a tie. Confirmed from the arm's own boot line:
"Frontier Self-Play: disabled (--no-frontier-selfplay)".

Nothing failed. No test broke. The experiment just quietly stopped being an
experiment, and its null result read as a finding.
"""

import sys

import pytest

sys.path.insert(0, "scripts")
import ablation_trackb3 as B3  # noqa: E402
import ablation_trackb4 as B4  # noqa: E402

# R17 MAJOR-2: this file imported B3 only, and all four arm assertions ran
# against it — so when §4BN R16 found B4's treatment arm identical to its
# uniform arm (it omitted `--frontier-selfplay`, which is off by default,
# making the ablation compare the treatment with itself), the fix landed
# in the one file this test does not import. Parametrised over both.
MODULES = [("trackb3", B3), ("trackb4", B4)]


@pytest.mark.parametrize("name,mod", MODULES)
def test_frontier_arm_actually_enables_frontier(name, mod):
    """R17 MAJOR-2: an arm named for a flag must PASS that flag. The
    default has been OFF since 2026-07-09, so omitting it makes the
    'frontier' arm identical to the control — and §4BN's
    `prm_consumer_is_live` False in both, so the PRM leg measures
    nothing either."""
    t = mod._treatment_flags(60.0)
    u = mod._treatment_uniform_flags(60.0)
    assert "--frontier-selfplay" in t, (
        f"{name}: the treatment arm does not enable frontier weighting — "
        "it is off by default, so this arm IS the control")
    assert "--no-frontier-selfplay" in u or "--frontier-selfplay" not in u, (
        f"{name}: the uniform arm still enables frontier weighting")
    assert t != u, f"{name}: arms are identical"


@pytest.mark.parametrize("name,mod", MODULES)
def test_treatment_and_uniform_arms_are_not_identical(name, mod):
    """The two #27b arms must differ. If they don't, the comparison is
    vacuous and any 'tie' it reports is guaranteed by construction."""
    t = mod._treatment_flags(60.0)
    u = mod._treatment_uniform_flags(60.0)
    assert t != u, (
        "treatment and treatment_uniform resolve to the SAME flags — the "
        "#27b comparison is comparing an arm with itself")


@pytest.mark.parametrize("name,mod", MODULES)
def test_treatment_states_frontier_explicitly(name, mod):
    """The arm must SET the behaviour that defines it, not inherit it.

    A flag default is a property of the application; an experiment arm is a
    claim about a configuration. Coupling them means a default flip silently
    redefines the experiment — which is exactly what happened here.
    """
    flags = mod._treatment_flags(60.0)
    assert "--frontier-selfplay" in flags, (
        "the frontier arm does not explicitly enable frontier self-play; it "
        "would inherit main.py's default, which is False")
    assert "--no-frontier-selfplay" not in flags


@pytest.mark.parametrize("name,mod", MODULES)
def test_uniform_arm_explicitly_disables_frontier(name, mod):
    flags = mod._treatment_uniform_flags(60.0)
    assert "--no-frontier-selfplay" in flags


@pytest.mark.parametrize("name,mod", MODULES)
def test_arms_differ_only_in_the_variable_under_test(name, mod):
    """Beyond being different, they must differ ONLY in the frontier setting —
    otherwise the comparison confounds frontier selection with something else."""
    t = set(mod._treatment_flags(60.0))
    u = set(mod._treatment_uniform_flags(60.0))
    diff = t.symmetric_difference(u)
    assert diff <= {"--frontier-selfplay", "--no-frontier-selfplay"}, (
        f"arms differ in more than the variable under test: {sorted(diff)}")


@pytest.mark.parametrize("name,mod", MODULES)
def test_treatment_arm_default_is_not_load_bearing(name, mod):
    """Guard the general rule: whatever main.py's default becomes, the arm's
    behaviour must be pinned by its own flags."""
    from ghost_agent import main as _main
    import inspect
    src = inspect.getsource(_main.parse_args)
    assert '"--frontier-selfplay"' in src, "flag renamed — update the arms"
    # The arm must not silently follow this default in either direction.
    flags = mod._treatment_flags(60.0)
    assert any(f.endswith("frontier-selfplay") for f in flags)


# ── warm-seeded runs must measure PRODUCTION, not the seed ───────────────────

def _mk_home(tmp_path, lessons, graduated=(), macros=()):
    import json
    mem = tmp_path / "system" / "memory"; mem.mkdir(parents=True, exist_ok=True)
    (mem / "skills_playbook.json").write_text(json.dumps(lessons))
    (mem / "auto_skills.json").write_text(json.dumps({k: {} for k in graduated}))
    cs = mem / "composed_skills"; cs.mkdir(exist_ok=True)
    (cs / "composed_skills.json").write_text(
        json.dumps({k: {"status": "proposed"} for k in macros}))
    return tmp_path


def _lesson(ts, source="self_play"):
    return {"timestamp": ts, "source": source}


def test_warm_arm_that_produced_nothing_reports_zero(tmp_path):
    """THE DEFECT (2026-08-09): with warm seeding the metric counted ABSOLUTE
    contents, so all three arms reported the snapshot's 50/5/19 in both repeats
    — identical down to the per-source breakdown — and that read as a tie."""
    seed = _mk_home(tmp_path / "seed", [_lesson(f"t{i}") for i in range(50)],
                    graduated=[f"g{i}" for i in range(5)],
                    macros=[f"m{i}" for i in range(19)])
    arm = _mk_home(tmp_path / "arm", [_lesson(f"t{i}") for i in range(50)],
                   graduated=[f"g{i}" for i in range(5)],
                   macros=[f"m{i}" for i in range(19)])
    base = B3._artifact_identities(seed)
    out = B3._learning_artifacts(arm, base)
    assert sum(out["lessons_by_source"].values()) == 0
    assert out["graduated_skills"] == 0 and out["proposed_macros"] == 0


def test_new_lesson_detected_even_though_playbook_is_at_its_cap(tmp_path):
    """PLAYBOOK_MAX = 50 and the live playbook sits AT it, so a produced lesson
    EVICTS an old one and the total never moves. Counting totals is blind to
    exactly the event being measured; identity-delta is not."""
    seed = _mk_home(tmp_path / "seed", [_lesson(f"t{i}") for i in range(50)])
    kept = [_lesson(f"t{i}") for i in range(49)] + [_lesson("BRAND-NEW")]
    arm = _mk_home(tmp_path / "arm", kept)
    assert len(kept) == 50, "fixture must stay at the cap"
    out = B3._learning_artifacts(arm, B3._artifact_identities(seed))
    assert out["lessons_by_source"].get("self_play") == 1


def test_cold_run_behaviour_is_unchanged(tmp_path):
    """No baseline (cold run) must behave exactly as before the fix."""
    arm = _mk_home(tmp_path / "arm", [_lesson(f"t{i}") for i in range(7)])
    out = B3._learning_artifacts(arm, None)
    assert sum(out["lessons_by_source"].values()) == 7
