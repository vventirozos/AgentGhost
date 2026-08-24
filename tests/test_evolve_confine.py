"""The confinement that closes what five review rounds kept re-finding.

⚠ EVERY TEST HERE NEEDS ITS NEGATIVE CONTROL. "The write was blocked" is
worthless unless the same write SUCCEEDS unconfined — otherwise the
fixture may simply be unable to perform it, and the suite would report
containment it never demonstrated. Each test below runs the identical
candidate both ways.
"""
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghost_agent.evolve import confine as CONFINE      # noqa: E402
from ghost_agent.evolve import evaluator as EV         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
_DARWIN = platform.system() == "Darwin"
darwin_only = pytest.mark.skipif(not _DARWIN,
                                 reason="seatbelt is macOS-only")


def _probe(target: Path, report: Path) -> str:
    """A script that tries every write the review rounds measured and
    reports what it managed."""
    return textwrap.dedent(f"""
        import os, pathlib, subprocess, sys, time
        T = pathlib.Path({str(target)!r})
        R = pathlib.Path({str(report)!r})
        out = []
        def t(label, fn):
            try:
                fn(); out.append(label + ": SUCCEEDED")
            except Exception as e:
                out.append(label + ": blocked (" + type(e).__name__ + ")")
        t("write a watched file", lambda: (T / "harness.py").write_text("TAMPERED"))
        t("plant a .pyc", lambda: (T / "__pycache__" / "p.pyc").write_bytes(b"x"))
        t("os.replace over it", lambda: os.replace(str(R.parent / "src.tmp"),
                                                   str(T / "harness.py")))
        subprocess.Popen([sys.executable, "-c",
            "import pathlib,time; time.sleep(0.4);"
            "pathlib.Path({str(target)!r} + '/GRANDCHILD').write_text('x')"],
            start_new_session=True)
        time.sleep(1.4)
        out.append("detached grandchild: " + str((T / "GRANDCHILD").exists()))
        R.write_text(chr(10).join(out))
    """)


def _run_probe(tmp_path, *, confined):
    target = tmp_path / "protected"
    (target / "__pycache__").mkdir(parents=True)
    (target / "harness.py").write_text("pristine")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "src.tmp").write_text("forged")
    report = allowed / "report.txt"
    script = tmp_path / "probe.py"
    script.write_text(_probe(target, report))

    cmd = [sys.executable, str(script)]
    if confined:
        cmd, ok, why = CONFINE.confine(cmd, writable=[allowed],
                                       allow_network=True,
                                       policy_dir=tmp_path / "policy")
        assert ok, f"the fixture could not confine anything: {why}"
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return (report.read_text() if report.exists() else ""), target


@darwin_only
def test_confinement_blocks_every_route_the_reviews_MEASURED(tmp_path):
    """⚠ THE FIVE ROUTES, EACH FOUND SEPARATELY AND EACH PATCHED
    SEPARATELY: writing the canonical harness mid-stage, planting a
    `.pyc` ahead of the source the digest hashes, renaming a forged file
    over an arm's results, and a detached grandchild doing any of it
    after its parent exited. They are one defect — a graded process with
    the parent's own write privileges — and this is the one fix."""
    got, target = _run_probe(tmp_path, confined=True)
    assert got, "the probe produced no report"
    for line in got.splitlines():
        assert "SUCCEEDED" not in line, line
    assert "detached grandchild: False" in got, got
    assert (target / "harness.py").read_text() == "pristine"
    assert not (target / "GRANDCHILD").exists()


@darwin_only
def test_the_SAME_probe_succeeds_unconfined(tmp_path):
    """⚠ WITHOUT THIS THE TEST ABOVE PROVES NOTHING. A fixture that
    cannot perform the writes would report containment it never
    demonstrated — the identical-under-both failure this repo keeps
    paying for."""
    got, target = _run_probe(tmp_path, confined=False)
    assert got, "the probe produced no report"
    assert got.count("SUCCEEDED") >= 3, got
    assert "detached grandchild: True" in got, got
    assert (target / "harness.py").read_text() != "pristine"


@darwin_only
def test_stage1_runs_CONFINED_and_says_so(tmp_path):
    """And the incumbent still passes its own gate through it — a
    policy that refuses honest candidates is one that gets switched
    off."""
    r = EV.stage1_pins(REPO, REPO, ["src/ghost_agent/tools/notify_tool.py"],
                       timeout_s=600, home=tmp_path / "h")
    assert r.detail["confined"] is True, r.detail
    assert r.passed, (r.reason, r.detail.get("summary"))


def test_an_UNCONFINED_run_is_recorded_as_such(tmp_path, monkeypatch):
    """⚠ SILENCE IS THE FAILURE MODE. There is no seatbelt off macOS, and
    the flag can be turned off to diagnose a policy that refuses an
    honest candidate. A cascade that ran unconfined while its docstring
    said otherwise would be worse than one that never had this — so the
    stage records which way it ran, every time, and names the reason."""
    monkeypatch.setenv(CONFINE.ENV_FLAG, "0")
    ok, why = CONFINE.available()
    assert not ok and CONFINE.ENV_FLAG in why, why
    cmd, confined, why2 = CONFINE.confine(["/bin/echo", "hi"],
                                          writable=[tmp_path],
                                          allow_network=True,
                                          policy_dir=tmp_path)
    assert confined is False and cmd == ["/bin/echo", "hi"], cmd
    assert CONFINE.ENV_FLAG in why2

    r = EV.stage1_pins(REPO, REPO, ["src/ghost_agent/tools/notify_tool.py"],
                       timeout_s=600, home=tmp_path / "h2")
    assert r.detail["confined"] is False, r.detail
    assert CONFINE.ENV_FLAG in r.detail["unconfined_because"], r.detail


@darwin_only
@pytest.mark.parametrize("handed_in, written_as",
                         [("/tmp/{}", "/private/tmp/{}"),
                          ("/private/tmp/{}", "/tmp/{}")])
def test_EITHER_spelling_of_a_tmp_path_is_writable(handed_in, written_as):
    """⚠ `/tmp` IS A SYMLINK TO `/private/tmp`, and denying a write the
    parent MEANT to allow reads as a hostile candidate — which is how a
    guard earns a reputation for false alarms and gets switched off.

    Asserted behaviourally, not as strings in the policy text: a string
    assertion here passed while a redundant branch was deleted, because
    the kernel matches on the RESOLVED path and `Path.resolve()` already
    supplies it. What matters is that the write lands."""
    import uuid
    name = f"ghost-confine-{uuid.uuid4().hex[:8]}"
    allowed = Path("/private/tmp") / name
    allowed.mkdir(parents=True)
    try:
        target = Path(written_as.format(name)) / "probe.txt"
        cmd, ok, why = CONFINE.confine(
            [sys.executable, "-c",
             f"import pathlib; pathlib.Path({str(target)!r}).write_text('ok')"],
            writable=[Path(handed_in.format(name))],
            allow_network=True, policy_dir=Path("/private/tmp") / name / "p")
        assert ok, why
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert target.read_text() == "ok"
    finally:
        import shutil
        shutil.rmtree(allowed, ignore_errors=True)


def test_the_policy_shape_is_deny_by_default():
    pol = CONFINE.policy([Path("/tmp/ghost-arm")], allow_network=False)
    assert "(deny file-write*)" in pol
    assert "(deny network*)" in pol
    assert CONFINE.policy([Path("/tmp/x")], allow_network=True).count(
        "deny network*") == 0
