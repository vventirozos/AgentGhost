"""Grounded file-artifact verification (2026-07-16).

The #1 most-retrieved real lesson is the agent "prematurely declared task
completion … without showing the actual content". The verifier now re-reads
the deliverable files the answer claims to have produced; a claimed file that
is MISSING or EMPTY in the sandbox refutes the answer (hard ground truth,
feeding the same auto-repair loop as the web-exec/visual overrides).
"""
import tempfile
from pathlib import Path

import pytest

from ghost_agent.core.agent import _claimed_deliverable_files, GhostAgent
from ghost_agent.core.verifier import VerifyVerdict


class TestClaimExtraction:
    @pytest.mark.parametrize("text,expected", [
        ("I saved the results to report.md.", ["report.md"]),
        ("Wrote output.csv and created summary.json.", ["output.csv", "summary.json"]),
        ("Created the plan in `investment_plan.md`.", ["investment_plan.md"]),
        ("Generated data.py, exported chart.png.", ["data.py", "chart.png"]),
    ])
    def test_completion_claims_extracted(self, text, expected):
        assert _claimed_deliverable_files(text) == expected

    @pytest.mark.parametrize("text", [
        "I read the config from settings.json.",           # read, not produced
        "The script writes to output.log each run.",       # present-tense behavior
        "See the docs at https://example.com/guide.md",    # URL
        "Edited /etc/hosts.conf on the server.",           # system path
        "Just a chat with no files at all.",
    ])
    def test_non_claims_ignored(self, text):
        assert _claimed_deliverable_files(text) == []

    def test_capped_and_deduped(self):
        txt = " ".join(f"saved f{i}.md" for i in range(20)) + " saved f0.md"
        got = _claimed_deliverable_files(txt)
        assert len(got) <= 8
        assert len(got) == len(set(got))


class TestVerifyFileArtifacts:
    def _dir(self):
        return Path(tempfile.mkdtemp())

    def test_missing_file_refutes(self):
        d = self._dir()
        r = GhostAgent._verify_file_artifacts(["nope.md"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED
        assert "missing" in r.reasoning and r.confidence >= 0.8

    def test_empty_file_refutes(self):
        d = self._dir(); (d / "out.csv").write_text("")
        r = GhostAgent._verify_file_artifacts(["out.csv"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED
        assert "empty" in r.reasoning

    def test_present_nonempty_no_override(self):
        d = self._dir(); (d / "report.md").write_text("real content here")
        assert GhostAgent._verify_file_artifacts(["report.md"], str(d)) is None

    def test_workspace_path_is_mapped_onto_host_dir(self):
        d = self._dir(); (d / "result.json").write_text("{}")
        # A container-style /workspace path must resolve under the host dir.
        assert GhostAgent._verify_file_artifacts(["/workspace/result.json"], str(d)) is None

    def test_basename_fallback_finds_nested_file(self):
        d = self._dir(); (d / "sub").mkdir(); (d / "sub" / "deep.txt").write_text("x")
        # Claimed as a bare name but actually nested → found by basename search.
        assert GhostAgent._verify_file_artifacts(["deep.txt"], str(d)) is None

    def test_mixed_reports_both(self):
        d = self._dir()
        (d / "ok.md").write_text("content")
        (d / "blank.txt").write_text("")
        r = GhostAgent._verify_file_artifacts(["ok.md", "blank.txt", "gone.csv"], str(d))
        assert r.verdict == VerifyVerdict.REFUTED
        assert "gone.csv" in r.reasoning and "blank.txt" in r.reasoning
        assert "ok.md" not in r.reasoning

    def test_no_claims_or_bad_dir_is_none(self):
        assert GhostAgent._verify_file_artifacts([], "/tmp") is None
        assert GhostAgent._verify_file_artifacts(["x.md"], "/nonexistent/xyz") is None
