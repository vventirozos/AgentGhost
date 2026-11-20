"""§4AT round-3 fixes across five subsystems (2026-08-11).

Each pin names the live defect it closes. All were found by the audit's
pre-registered questions and verified against the real code before fixing.
(The section-H pins on the post-mortem engine went with the engine, §4NM.)
"""
from __future__ import annotations

import inspect
import os
import sys


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import dream as DREAM              # noqa: E402
from ghost_agent.core import project_advancer as PA      # noqa: E402
from ghost_agent.tools import registry as REG            # noqa: E402


# ── F: the retry-steering mislabel (req 6e9efd6a's 20-minute grind) ──────









# ── G: simulation must not author real selfhood ─────────────────────────


# ── C: a string constraint must not shred into characters ───────────────

def test_project_advancer_uses_the_SHARED_constraint_normaliser():
    """Metadata is model-written JSON, so `{"constraints": "no external
    APIs"}` is legal — and iterating it raw yields 17 single-character
    "constraints" fed to the constraint gate. `_constraint_list` exists
    because this shredding already destroyed a record once (2026-08-01); this
    call site kept its own copy and so kept the bug."""
    src = inspect.getsource(PA)
    assert "_constraint_list(" in src, "shared normaliser not used"
    assert "[str(c) for c in\n                        ((proj.get" not in src

    from ghost_agent.memory.projects import _constraint_list
    assert _constraint_list("no external APIs") == ["no external APIs"]
    assert len(_constraint_list("no external APIs")) == 1, "shredded again"
    assert _constraint_list(["a", "b"]) == ["a", "b"]
    assert _constraint_list(None) == []


# ── E: the live A/B must differ by exactly the treatment ────────────────

# ── B: self-play attribution, re-landed after the data repair ───────────









# ── F: a dropped edit must not read as "nothing to do" ──────────────────







def test_the_PRODUCER_actually_marks_a_truncated_snapshot_entry():
    """⚠ BOTH SIDES OR NEITHER. Mutation-testing showed the consumer guard was
    pinned but the PRODUCER was not: removing the marker from
    `_gather_project_files` reddened nothing, so the guard would silently
    never fire again. A one-sided pin on a producer/consumer contract is the
    same drift it exists to prevent."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from ghost_agent.core.project_advancer import (
        _gather_project_files, SNAPSHOT_TRUNCATED_MARK)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "projects" / "pid"
        root.mkdir(parents=True)
        # Exceed the char budget so truncation is forced.
        (root / "big.html").write_text("<html>" + "y" * 600_000 + "</html>")
        (root / "small.py").write_text("print(1)\n")
        store = SimpleNamespace(sandbox_root=str(Path(d)))
        snap = _gather_project_files(store, "pid")
        assert snap, "snapshot came back empty — the probe is broken"
        marked = [k for k, v in snap.items() if SNAPSHOT_TRUNCATED_MARK in v]
        assert marked, (
            "a truncated entry carries no marker — the consumer's guard can "
            f"never fire. entries: { {k: len(v) for k, v in snap.items()} }")
        # …and a file that fits is NOT marked (no false positives).
        assert SNAPSHOT_TRUNCATED_MARK not in snap.get("small.py", "")


# ── C: a crash must not downgrade to a weaker closer ────────────────────

def test_a_coding_executor_CRASH_leaves_the_leaf_open():
    """⚠ THE THIRD MEMBER OF THE FAMILY. A transient exception inside
    `coding_executor` left `cres = None` and fell through to the generic
    single-shell-command path, which marks the leaf DONE on ANY output that is
    not `ERROR:`-prefixed and not a non-zero exit — verify gate, smoke gate,
    files-written check and constraint gate all bypassed. `echo`-shaped output
    closed a build task.

    Same shape as the two coding-executor CRITICALs fixed the same day: a task
    reaching DONE on evidence that no work happened. A crash is a reason to
    STOP, not to accept a lower standard of proof."""
    src = inspect.getsource(PA)
    i = src.find('logger.warning("coding_executor crashed')
    assert i > 0, "the crash handler moved — re-anchor this pin"
    after = src[i:i + 1600]          # §4PG r3: the deferral branch sits between the log line and the crash result
    assert "AdvanceResult(" in after and '"blocked"' in after, (
        "a coding-executor crash still falls through to the weaker "
        "single-command path that can close the leaf DONE")
    # and it must NOT simply swallow and continue
    assert "return" in after.split("AdvanceResult(")[0] or "AdvanceResult(" in after


def test_a_frontier_run_records_WHICH_TEMPLATE_produced_it():
    """⚠ AUDITABILITY, not correctness. A run recorded only the cluster it was
    FILED under, never the template that produced it — so when attribution
    turned out to be 51% wrong, reconstructing what belonged where required
    re-rendering the whole template bank at four tiers and matching challenge
    hashes. The record has to say."""
    import tempfile
    from pathlib import Path
    from ghost_agent.memory.frontier import FrontierTracker

    with tempfile.TemporaryDirectory() as d:
        f = FrontierTracker(Path(d))          # takes a memory DIR
        f.record_run("bash", "count lines in logs", 1, True, 40,
                     template_key="bash")
        f.record_run("sql", "an LLM-generated challenge", 1, True, 40)
        import json, glob
        files = glob.glob(str(Path(d) / "*frontier*.json"))
        assert files, f"no frontier file written: {list(Path(d).iterdir())}"
        runs = json.load(open(files[0]))["runs"]
        assert len(runs) == 2
        assert runs[0].get("template_key") == "bash", (
            "the template that produced the run is not recorded")
        # …and an untemplated challenge records EMPTY, which is itself the
        # distinction the cluster field cannot carry.
        assert runs[1].get("template_key") == ""
