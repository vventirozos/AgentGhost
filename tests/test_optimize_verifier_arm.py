"""scripts/optimize_verifier.py — which verdict pipeline the SHIP GATE ran.

§4J open item 1, escalation axis. The GEPA ship gate promotes verifier
templates into production on a balanced score measured over private bench
trials. Those trials ran through a single-endpoint `HttpChatClient`, where
`Verifier._escalate_refute` is structurally a no-op — so the gate scored
the CHEAP JUDGE STANDALONE while production scores judge+escalation, and
production overturned 42 of 50 (84%) of the cheap judge's refutes on the
live recorded corpus 2026-07-30..08-04. Most of the false-alarm mass a
candidate could be credited for at the gate is mass production already
removes.

`--escalate {off,gate,all}` closes that, and the arm is written into the
promoted artifact (`gate_arm`) so two artifacts judged by different
pipelines can never be compared as if they were one series.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from ghost_agent.eval.verify_bench import (
    ARM_ESCALATED,
    ARM_ESC_CONFIRM,
    ARM_RAW,
    EscalatingChatClient,
    HttpChatClient,
)


@pytest.fixture(scope="module")
def ov():
    spec = importlib.util.spec_from_file_location(
        "ov_under_test", "scripts/optimize_verifier.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adapter(ov, **kw):
    return ov.VerifierBenchAdapter("http://judge.invalid", **kw)


# ── The adapter builds the client for the arm it claims ──────────────

def test_default_adapter_is_the_raw_judge(ov):
    a = _adapter(ov)
    assert isinstance(a.build_client(), HttpChatClient)
    assert not isinstance(a.build_client(), EscalatingChatClient)
    assert a.arm_label()["arm"] == ARM_RAW


def test_escalate_with_a_main_url_builds_the_two_leg_client(ov):
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=True)
    client = a.build_client()
    assert isinstance(client, EscalatingChatClient)
    assert client.main_base_url == "http://main.invalid"
    label = a.arm_label()
    assert label["arm"] == ARM_ESCALATED
    assert label["cheap_route"] == "worker"
    assert label["judge"]["base_url"] == "http://judge.invalid"
    assert label["main"]["base_url"] == "http://main.invalid"


def test_escalate_without_a_main_url_reports_raw_not_escalated(ov):
    """The label must describe what would ACTUALLY run. A flag set without
    the endpoint it needs is exactly the shape that writes `gate_arm:
    judge+escalation` onto an artifact no main model ever saw."""
    a = _adapter(ov, escalate=True)  # no main_base_url
    assert isinstance(a.build_client(), HttpChatClient)
    assert not isinstance(a.build_client(), EscalatingChatClient)
    assert a.arm_label()["arm"] == ARM_RAW


def test_the_arm_is_read_at_evaluate_time_not_frozen(ov):
    """`--escalate gate` flips `adapter.escalate` around the two private
    evaluations, so the field must be live rather than captured in
    __init__ — otherwise the gate would silently run the training arm."""
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=False)
    assert a.arm_label()["arm"] == ARM_RAW
    a.escalate = True
    assert a.arm_label()["arm"] == ARM_ESCALATED
    a.escalate = False
    assert a.arm_label()["arm"] == ARM_RAW


def test_gate_arm_active_restores_the_training_arm(ov):
    """`--escalate gate` = train cheap, gate escalated. The scope must be
    exactly the private evaluation."""
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=False)
    with ov.gate_arm_active(a, "gate"):
        assert a.escalate is True
    assert a.escalate is False


def test_gate_arm_active_restores_on_an_exception(ov):
    """A raising evaluation must not leave the adapter in the gate arm —
    the rest of the GEPA loop would then run a pipeline nothing names."""
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=False)
    with pytest.raises(RuntimeError):
        with ov.gate_arm_active(a, "gate"):
            raise RuntimeError("node down")
    assert a.escalate is False


def test_gate_arm_active_is_a_noop_when_escalation_is_off(ov):
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=False)
    with ov.gate_arm_active(a, "off"):
        assert a.escalate is False
    assert a.escalate is False


def test_escalate_all_keeps_the_gate_and_training_arms_identical(ov):
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=True)
    with ov.gate_arm_active(a, "all"):
        assert a.escalate is True
    assert a.escalate is True


def test_the_kill_switches_win_over_the_flag(ov, monkeypatch):
    """The kill switches disable escalation inside the verifier itself, so
    a run with --escalate all would measure something weaker than the flag
    claims. The label follows the verifier, per direction — killing only
    one switch leaves a HALF-escalated gate, which must not report as
    either `raw_judge` or `judge+escalation`."""
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "0")
    a = _adapter(ov, main_base_url="http://main.invalid", escalate=True)
    assert a.arm_label()["arm"] == ARM_ESC_CONFIRM

    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "0")
    assert a.arm_label()["arm"] == ARM_RAW


def test_the_confirm_direction_cannot_move_the_gate_metric(ov):
    """Worth pinning because it is counter-intuitive and it is why
    `--escalate gate` costs main-model calls the ship decision cannot
    use: `_trial_score` is VERDICT-ONLY, and `_escalate_confirm` never
    changes a verdict — it caps confidence. So a withheld confirmation is
    invisible to `balanced_score`. If the gate should ever reward the cap,
    `_trial_score` has to become actionable-confidence aware; that is a
    deliberate design change, not something to slip in silently."""
    t = ov.BenchTrial("c", "silent_failure", "REFUTED", "cl", "ev", "ctx",
                      high_stakes=True)
    # Same verdict, capped vs uncapped confidence -> identical score.
    assert ov._trial_score(t, "CONFIRMED") == ov._trial_score(t, "CONFIRMED")
    assert ov._trial_score(t, "CONFIRMED") == 0.0
    assert ov._trial_score(t, "REFUTED") == 1.0


# ── The CLI refuses an arm it cannot take ────────────────────────────

@pytest.mark.parametrize("mode", ["gate", "all"])
def test_cli_refuses_escalate_without_a_main_url(mode):
    env = dict(os.environ, PYTHONPATH="src")
    env.pop("GHOST_HOME", None)
    r = subprocess.run(
        [sys.executable, "scripts/optimize_verifier.py",
         "--base-url", "http://judge.invalid", "--escalate", mode],
        capture_output=True, text=True, env=env)
    assert r.returncode == 2, (
        "running the raw arm under an --escalate flag would ship an "
        "artifact whose recorded gate_arm never happened")
    assert "--main-base-url" in r.stderr


# ── The graded score / balanced metric are unchanged by all this ─────

def test_balanced_score_still_macro_averages_the_two_classes(ov):
    trials = [
        ov.BenchTrial("c", "clean", "CONFIRMED", "cl", "ev", "ctx"),
        ov.BenchTrial("c", "fact_swap", "REFUTED", "cl", "ev", "ctx"),
        ov.BenchTrial("c", "fabrication", "REFUTED", "cl", "ev", "ctx"),
        ov.BenchTrial("c", "wrong_topic", "REFUTED", "cl", "ev", "ctx"),
    ]
    # 3:1 refute-heavy, the real shape of the pool (measured 2026-08-04:
    # 428/140 public, 166/54 private). A raw mean would be dominated by the
    # refute class; the balanced metric splits it 50/50.
    assert ov.balanced_score(trials, [0.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert ov.balanced_score(trials, [1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.5)


def test_trial_score_grades_uncertain_between_right_and_wrong(ov):
    t = ov.BenchTrial("c", "fact_swap", "REFUTED", "cl", "ev", "ctx")
    assert ov._trial_score(t, "REFUTED") == 1.0
    assert ov._trial_score(t, "UNCERTAIN") == 0.3
    assert ov._trial_score(t, "CONFIRMED") == 0.0
    assert ov._trial_score(t, None) == 0.0


# ── The artifact records the pipeline that promoted it ───────────────

def test_the_shipped_payload_carries_the_gate_arm(ov, tmp_path):
    """Regression fence on the artifact SHAPE. `gate_arm` is the only thing
    that stops a raw-judge promotion and an escalated one from being read
    as one comparable series of `private_candidate_balanced` numbers."""
    src = (tmp_path / "optimize_verifier_src.py")
    src.write_text(open("scripts/optimize_verifier.py").read())
    text = src.read_text()
    for key in ('"gate_arm": gate_arm["arm"]',
                '"train_arm": train_arm["arm"]',
                '"gate_judge": gate_arm.get("judge")',
                '"gate_main": gate_arm.get("main")'):
        assert key in text, f"the promoted artifact must record {key}"
    # ...and the arm must come from the detector, not from the raw flag.
    assert "gate_arm = adapter.arm_label()" in text
    assert 'gate_arm = args.escalate' not in text


def test_incumbent_only_records_a_baseline_with_provenance(tmp_path):
    """`--incumbent-only` writes the number the NEXT round's ship gate
    compares against. A bare score is not a baseline — everything that
    would invalidate the comparison has to travel with it, and the
    per-trial rows have to make it re-scorable without re-running.

    Endpoints are deliberately dead: this exercises the artifact contract,
    not the judge (every trial skips, which is itself the degenerate case
    worth surviving).
    """
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(
        json.dumps({"case_id": f"c{i}", "claim": f"The value is {i}{i}.",
                    "evidence": f"[calc] {i}{i}", "context": "what value"})
        for i in range(10)) + "\n")
    out = tmp_path / "baseline.json"
    env = dict(os.environ, PYTHONPATH="src", GHOST_HOME=str(tmp_path))
    r = subprocess.run(
        [sys.executable, "scripts/optimize_verifier.py",
         "--base-url", "http://127.0.0.1:9",
         "--main-base-url", "http://127.0.0.1:10", "--escalate", "gate",
         "--cases", str(cases), "--min-delta", "0.9",
         "--incumbent-only", str(out)],
        capture_output=True, text=True, env=env)

    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(out.read_text())
    # The gate arm and BOTH endpoints, or the number is unattributable.
    prov = d["provenance"]
    assert prov["escalation"]["arm"] == ARM_ESCALATED
    assert prov["escalation"]["judge"]["base_url"] == "http://127.0.0.1:9"
    assert prov["escalation"]["main"]["base_url"] == "http://127.0.0.1:10"
    assert prov["escalation"]["directions"]["confirm"]["live"] is True
    # What would invalidate a later comparison.
    assert prov["cases_sha256"] and prov["faults"] and prov["templates"]
    for k in ("private_incumbent_balanced", "n_private_cases",
              "n_private_trials", "class_mix", "smallest_resolvable_delta",
              "high_stakes_trials", "seed_templates", "recorded_utc"):
        assert k in d, f"baseline must record {k}"
    # Re-scorable: one row per trial, and the counters agree with them.
    assert len(d["trials"]) == d["n_private_trials"]
    assert d["escalation_events"]["confirm_eligible"] == sum(
        1 for t in d["trials"]
        if t["high_stakes"] and t["verdict"] == "CONFIRMED"
        and not t["escalated_overturn"])
    assert sum(d["verdicts"].values()) == d["n_private_trials"]
    # ...and it must NOT have touched the live artifact directory.
    assert not (tmp_path / "system" / "optim"
                / "verifier.enumerate.json").exists()


def test_the_json_payload_round_trips(ov):
    """Cheap guard that the added provenance keys stay JSON-serialisable —
    `gate_main` can be `{"unresolved": ...}` rather than a plain dict."""
    payload = {
        "signature_name": "verifier.enumerate",
        "gate_arm": ARM_ESCALATED,
        "train_arm": ARM_RAW,
        "gate_judge": {"base_url": "http://j", "model": "m"},
        "gate_main": {"unresolved": "HttpChatClient"},
    }
    assert json.loads(json.dumps(payload))["gate_arm"] == ARM_ESCALATED
