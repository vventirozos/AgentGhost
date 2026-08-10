"""The operator's log lens, and the source-level demotion behind it.

MEASURED 2026-08-09. `$GHOST_HOME/system/ghost-agent.log` is deliberately a
COMPLETE record (every mirror line, DEBUG included) so a turn survives a
restart and instruments can count events. Right for an archive, wrong for
monitoring: on a 4000-line window **1652 lines (41%) were ONE repeated
string** — `critic compute — Routing verification to Critic Node (Nova)` —
while only **6 lines carried a verdict**. 275 announcements per outcome.

Two fixes, both pinned here:
  1. SOURCE — background plumbing (self-play, REM, failure-dimension tagging)
     drops to DEBUG. It stays in the archive; it leaves the operator's view.
     Request-scoped routing stays INFO, because that is what is being
     watched. Measured effect: the INFO view halves, 3783 -> 1900 lines.
  2. LENS — `scripts/ghostlog.py` reads INFO+ with consecutive repeats
     collapsed, and `--req <id>` reconstructs ONE request in full.

⚠ Why the fix is a LEVEL change and not mirror-collapse: `pretty_log`'s own
comment records that several instruments COUNT mirror lines (the
escalation-overturn double-count lesson). Collapsing the mirror would break
them. The archive stays complete; only the level moves.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VIEWER = REPO / "scripts" / "ghostlog.py"

SAMPLE = """\
2026-08-09 20:04:12 - GhostStream - INFO - [178defd6 +1.0s] request started — 178defd6
2026-08-09 20:04:13 - GhostStream - DEBUG - [SYSTEM] critic compute — Routing verification to Critic Node (Nova)
2026-08-09 20:04:14 - GhostStream - DEBUG - [SYSTEM] critic compute — Routing verification to Critic Node (Nova)
2026-08-09 20:04:15 - GhostStream - DEBUG - [SYSTEM] critic compute — Routing verification to Critic Node (Nova)
2026-08-09 20:04:16 - GhostStream - INFO - [178defd6 +4.0s] reasoning loop — Turn 1
2026-08-09 20:04:17 - GhostStream - INFO - [SYSTEM] dream mode — Entering REM cycle
2026-08-09 20:04:18 - GhostStream - INFO - [SYSTEM] dream mode — Entering REM cycle
2026-08-09 20:04:19 - GhostStream - WARNING - [SYSTEM] worker node failed — Nova: ReadTimeout
2026-08-09 20:04:20 - GhostStream - INFO - [178defd6 +8.0s] request finished — +8.0s
"""


@pytest.fixture()
def log(tmp_path):
    p = tmp_path / "ghost-agent.log"
    p.write_text(SAMPLE)
    return p


def _run(log, *args):
    return subprocess.run(
        [sys.executable, str(VIEWER), "--file", str(log), *args],
        cwd=REPO, capture_output=True, text=True, timeout=120,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
             "NO_COLOR": "1"})


# ── the lens ────────────────────────────────────────────────────────────────

def test_debug_plumbing_is_hidden_by_default(log):
    """The 41% of the archive nobody reads must not reach the default view."""
    out = _run(log).stdout
    assert "Routing verification" not in out
    assert "request started" in out and "reasoning loop" in out


def test_consecutive_repeats_collapse_with_a_count(log):
    out = _run(log).stdout
    assert out.count("Entering REM cycle") == 1
    assert "×2" in out


def test_warnings_survive_the_filter(log):
    out = _run(log).stdout
    assert "ReadTimeout" in out and "WARN" in out


def test_all_shows_the_complete_archive(log):
    """The archive is complete by design; --all must prove it still is."""
    out = _run(log, "--all").stdout
    assert "Routing verification" in out


def test_req_reconstructs_one_request_in_full(log):
    """The 'how was this request processed' view, without a TUI."""
    out = _run(log, "--req", "178defd6").stdout
    assert "request started" in out and "request finished" in out
    assert "reasoning loop" in out
    assert "dream mode" not in out, "background noise leaked into the request view"
    assert "Routing verification" not in out, "another request's plumbing leaked"


def test_req_implies_all_levels(tmp_path):
    """A request's own DEBUG detail is exactly what --req is for."""
    p = tmp_path / "l.log"
    p.write_text("2026-08-09 20:04:15 - GhostStream - DEBUG - [abc12345 +1s] thinking — plan\n")
    assert "thinking" in _run(p, "--req", "abc12345").stdout


def test_long_lines_are_truncated_in_the_MIDDLE(tmp_path):
    """A hydration line runs 1500+ chars and destroys the view it explains.
    Cutting the TAIL would lose the outcome, so the middle goes."""
    p = tmp_path / "l.log"
    p.write_text("2026-08-09 20:04:15 - GhostStream - INFO - [SYSTEM] memory bus — "
                 + ("A" * 400) + "TAILMARK\n")
    out = _run(p, "--width", "80").stdout
    assert "more…" in out
    assert "memory bus" in out, "the head (what the event is) must survive"
    assert "TAILMARK" in out, "the tail (the outcome) must survive"


def test_no_collapse_flag_shows_every_occurrence(log):
    out = _run(log, "--all", "--no-collapse").stdout
    assert out.count("Routing verification") == 3


def test_missing_log_is_an_error_not_an_empty_view(tmp_path):
    r = _run(tmp_path / "nope.log")
    assert r.returncode == 2 and "no log at" in r.stderr


def test_unparseable_lines_do_not_crash_it(tmp_path):
    p = tmp_path / "l.log"
    p.write_text("not a log line at all\n" + SAMPLE)
    assert _run(p).returncode == 0


def test_an_unknown_level_is_shown_rather_than_hidden(tmp_path):
    """A line we cannot classify is more likely to matter, not less."""
    p = tmp_path / "l.log"
    p.write_text("2026-08-09 20:04:15 - GhostStream - NOTICE - [SYSTEM] odd — thing\n")
    assert "odd" in _run(p).stdout


# ── the source-level demotion ───────────────────────────────────────────────

def test_background_routing_is_debug_and_request_routing_is_info():
    """THE DEFECT: this line was unconditionally INFO and became 41% of the
    log. `request_id_context` already distinguishes background plumbing
    ("SYSTEM") from a real user turn — the level must follow it.

    ⚠ Each call site is asserted SEPARATELY. A generic
    `'level="DEBUG" if _bg else "INFO"' in src` passed with the critic
    demotion reverted, because the worker line still matched it — a weak pin
    caught by revert-testing, the fifth of the session.
    """
    src = (REPO / "src" / "ghost_agent" / "core" / "llm.py").read_text()
    assert 'level="DEBUG" if _bg else "INFO", icon=Icons.VERIFIER_LAB' in src, (
        "the CRITIC routing line is unconditionally INFO again — it was 41% "
        "of the operator's log")
    assert 'level="DEBUG" if _bg else "INFO", icon=Icons.NODE_WORKER' in src, (
        "the WORKER routing line is unconditionally INFO again")
    assert src.count('_bg = request_id_context.get() == "SYSTEM"') >= 2


def test_failures_were_not_demoted():
    """Only the intent line moved. A node failure is still WARNING — that is
    the signal the demotion exists to make visible."""
    src = (REPO / "src" / "ghost_agent" / "core" / "llm.py").read_text()
    assert 'pretty_log("Critic Node Failed"' in src and 'level="WARNING"' in src
    assert 'pretty_log("Critic Compute Failed"' in src


def test_the_mirror_is_still_complete():
    """⚠ Mirror-collapse was considered and REJECTED: several instruments
    count mirror lines. The archive must keep recording every occurrence."""
    src = (REPO / "src" / "ghost_agent" / "utils" / "logging.py").read_text()
    i_mirror = src.index("_mirror(req_id, title_str, full, level)")
    i_collapse = src.index("if _collapse_enabled():")
    assert i_mirror < i_collapse, (
        "collapse now runs before the mirror — the archive is no longer "
        "complete and the instruments that count mirror lines will undercount")


def test_a_critic_HEARTBEAT_does_not_claim_to_be_a_verification():
    """⚠ MEASURED 2026-08-10. The keepalive loop (45s) and node warmup reach
    the critic branch via `use_critic=(label == "critic")`, so every ping
    logged "Routing verification to Critic Node" — describing work that never
    happened. The WORKER branch has had `_quiet` for this since the heartbeat
    was added; the critic branch never got it.

    It mattered: those pings were the only "verification" activity in an idle
    log, which made it look as though verifications were running and producing
    no outcome line. They were not running at all.
    """
    src = (REPO / "src" / "ghost_agent" / "core" / "llm.py").read_text()
    # ⚠ Anchor on the CALL, not the phrase: "Routing verification" now also
    # appears in the explanatory comment, so slicing to its first occurrence
    # cut before the guard and failed on correct code.
    i = src.index("elif use_critic and getattr(self, 'critic_clients', None):")
    j = src.index('pretty_log("Critic Compute"', i)
    branch = src[i:j]
    assert 'task_label in ("keepalive", "warmup")' in branch, (
        "the critic branch logs heartbeat pings as verifications again")
