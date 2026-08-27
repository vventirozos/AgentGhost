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
        # ⚠ These two MUST carry a completion verb. Without one the claim
        # regex never fires at all, so they passed while the guards they
        # name — the URL check and the system-path check — were unreachable
        # code that could not have excluded anything.
        "Generated the docs at https://example.com/guide.md",   # URL
        "Wrote /etc/hosts.conf on the server.",                 # system path

        "Saved it to /Users/v/Data/AI/notes.md",                 # host path
        "Just a chat with no files at all.",
    ])
    def test_non_claims_ignored(self, text):
        assert _claimed_deliverable_files(text) == []

    @pytest.mark.parametrize("text,expected", [
        # ⚠ SEGMENT boundaries. Once the capture could begin with "/",
        # `startswith` on a bare `/opt` swallowed `/optimizer.md`, `/bin`
        # swallowed `/binary.json`, `/root` swallowed `/rootcause.md` — real
        # sandbox-root deliverables dropped SILENTLY, in the direction no
        # log line reports.
        ("wrote /optimizer.md", ["/optimizer.md"]),
        ("saved /binary.json", ["/binary.json"]),
        ("created /homepage.html", ["/homepage.html"]),
        ("wrote /tmp_notes.md", ["/tmp_notes.md"]),
        ("saved /rootcause.md", ["/rootcause.md"]),
        ("saved /variables.py", ["/variables.py"]),
        # …while a real system path still goes — structurally, no tuple to
        # spell "/System" in: a multi-segment absolute is not sandbox-shaped.
        ("Wrote /etc/hosts.conf on the server.", []),
        ("Wrote /USR/local/share/x.md", []),
        ("Saved it to /Users/v/Data/AI/notes.md", []),
    ])
    def test_system_paths_are_matched_by_segment_not_by_prefix(self, text,
                                                               expected):
        assert _claimed_deliverable_files(text) == expected

    @pytest.mark.parametrize("text,expected", [
        # ⚠ The download route is STRIPPED, not dropped: the link is a URL
        # but `<rel>` points at a real artifact (`image_generation` /
        # `report_pdf` outputs) that no other arm covers — the ledger parses
        # only `file_system` confirmations. 41 recorded turns carry such
        # links; dropping them lost their only coverage, refuting on them
        # verbatim was the original false-refute shape. The claim becomes
        # the artifact the route serves, checked for emptiness only.
        ("Created ![generated image](/api/download/gen_d7f2a1b3.png)",
         ["gen_d7f2a1b3.png"]),
        ("Generated the report at /api/download/projects/x/r.pdf",
         ["projects/x/r.pdf"]),
        # …and a genuine deliverable in a directory called `api/` survives:
        # only the download route is special, never the bare segment.
        ("saved the handler to api/routes.py", ["api/routes.py"]),
        ("created api/schema.json", ["api/schema.json"]),
    ])
    def test_download_routes_become_the_artifact_they_serve(
            self, text, expected):
        assert _claimed_deliverable_files(text) == expected

    @pytest.mark.parametrize("text", [
        # ⚠ Shapes the old host-path DENYLIST leaked, each becoming a claim
        # about a file that cannot exist in the sandbox: `~` and `$HOME`
        # fragments, `/Volumes`, and the redaction artifact born when
        # `/Users/<user>` is redacted and the capture restarts mid-path.
        # The allowlist ("relative, /workspace/, or a single-segment
        # sandbox-root file") rejects them structurally.
        "saved to ~/Data/AI/x.json",
        "stored /Volumes/ext/z.csv",
        "saved /Data/AI/Data/system/prm/checkpoint.json",
    ])
    def test_host_shapes_the_denylist_leaked_are_rejected(self, text):
        assert _claimed_deliverable_files(text) == []

    @pytest.mark.parametrize("text", [
        # ⚠ single-segment forms: the capture used to restart mid-path, so
        # `~/notes.md` yielded `/notes.md` — a legitimate-looking sandbox-
        # root spelling for a file in the operator's home. The regex now
        # captures the `~`/`$VAR` fragment and the filter rejects it.
        "saved to ~/notes.md",
        "wrote $HOME/a.txt",
        "wrote ${HOME}/a.txt",     # the brace form fragmented to /a.txt
    ])
    def test_home_fragments_are_captured_and_rejected_whole(self, text):
        assert _claimed_deliverable_files(text) == []

    def test_the_route_strip_is_case_insensitive(self):
        # the claim regex is IGNORECASE, so the capture can arrive as
        # /API/DOWNLOAD/… — un-stripped it fell to the allowlist as a
        # multi-segment absolute and silently lost its soft coverage.
        assert _claimed_deliverable_files(
            "Created ![i](/API/DOWNLOAD/gen_a.png)") == ["gen_a.png"]

    def test_workspace_prefixed_claims_are_sandbox_shaped(self):
        # ⚠ `/workspace/…` is the container mount's own spelling — the one
        # absolute multi-segment form that IS inside the sandbox. Without
        # this branch the allowlist's multi-segment rejection would silently
        # drop the corpus's real `/workspace/.services/...` claims.
        assert _claimed_deliverable_files(
            "I saved /workspace/webos/index.html for you."
        ) == ["/workspace/webos/index.html"]

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


class TestMutatedFileCollector:
    """`_files_mutated_this_turn` (2026-07-19): the FILE-ARTIFACT check now
    unions the answer's claimed deliverables with the files the turn ACTUALLY
    wrote/replaced — an edited-but-unclaimed file (described as "updated",
    not "created") previously skipped the ground-truth re-read entirely."""

    def test_collects_writes_and_replaces_any_extension(self):
        from ghost_agent.core.agent import _files_mutated_this_turn
        tools = [
            {"role": "tool", "name": "file_system",
             "content": "SUCCESS: Wrote 11196 chars to 'projects/e4e2/minesweeper.html'."},
            {"role": "tool", "name": "file_system",
             "content": "SUCCESS: Applied 1 SEARCH/REPLACE blocks to 'projects/e4e2/index.html'."},
            {"role": "tool", "name": "file_system",
             "content": "SUCCESS: Wrote 90 chars to 'notes.md'."},
        ]
        assert _files_mutated_this_turn(tools) == [
            "projects/e4e2/minesweeper.html",
            "projects/e4e2/index.html",
            "notes.md",
        ]

    def test_ignores_failures_synthetic_and_other_tools(self):
        from ghost_agent.core.agent import _files_mutated_this_turn
        tools = [
            {"role": "tool", "name": "file_system", "_synthetic": True,
             "content": "SUCCESS: Wrote 10 chars to 'fake.html'."},
            # ⚠ This entry must carry a mutation MARKER and a quoted path,
            # so the tool-NAME gate is the only thing excluding it. With a
            # markerless payload the assertion passed even with the name
            # gate deleted (mutation M11, 2026-08-26) — a vacuous pin.
            {"role": "tool", "name": "run_skill",
             "content": "SUCCESS: Wrote 40 chars to 'skill_out.json'."},
            {"role": "tool", "name": "file_system",
             "content": "REJECTED: that replace would have written marker lines into 'x.html'."},
            {"role": "tool", "name": "file_system",
             "content": "Error: could not write 'broken.js'."},
        ]
        assert _files_mutated_this_turn(tools) == []
        assert _files_mutated_this_turn(None) == []


class TestDeletedAndMovedPathsAreNotDeliverables:
    """⚠ REGRESSION (2026-08-26, the GlassOS webOS refute).

    The mutated-file selector matched any quoted dotted token in any
    file_system SUCCESS message — `SUCCESS: Deleted 'probe.py'.` included.
    A scratch probe the agent wrote, ran and then TIDIED UP was therefore
    re-read as a deliverable, found missing, and REFUTED (0.9) a turn whose
    real artifact was on disk the whole time, overriding a text CONFIRMED
    (0.95). Because `tools_run_this_turn` accumulates across the whole
    REQUEST the poison also survived the auto-repair round and the
    post-reply re-verify — and no repair could clear it, since the "missing
    deliverable" was a file the agent had deliberately removed.

    The selector is now shape-aware and order-aware, and its bookkeeping
    key is NORMALISED: `_get_safe_path` resolves `/workspace/x`, `./x` and
    `x` to one file, and the write confirmation prints two spellings while
    the delete prints one, so literal-string bookkeeping let the incident
    reproduce verbatim under a spelling mismatch.
    """

    def _mut(self, tools):
        from ghost_agent.core.agent import _files_mutated_this_turn
        return _files_mutated_this_turn(tools)

    def _fs(self, content):
        return {"role": "tool", "name": "file_system", "content": content}

    def test_scratch_file_written_then_deleted_is_not_a_deliverable(self):
        # The verbatim recorded shapes from the incident.
        tools = [
            self._fs("SUCCESS: Wrote 24966 chars to 'webos/index.html'. "
                     "Script-side path (from sandbox cwd): 'webos/index.html'."),
            self._fs("SUCCESS: Wrote 391 chars to 'probe.py'. "
                     "Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
            self._fs("SUCCESS: Deleted '_chk.js'."),
        ]
        assert self._mut(tools) == ["webos/index.html"]

    @pytest.mark.parametrize("wrote,deleted", [
        ("probe.py", "/workspace/probe.py"),      # model deletes by container path
        ("/workspace/probe.py", "probe.py"),      # …and the mirror
        ("./probe.py", "probe.py"),
        ("probe.py", "./probe.py"),
        ("Probe.py", "probe.py"),                 # case-insensitive host FS
    ])
    def test_spelling_mismatch_still_cancels(self, wrote, deleted):
        # ⚠ Every fixture in the first draft of this class used
        # write-spelling == delete-spelling — the one case where the bug
        # does NOT fire. The regression suite was picked around the hole.
        tools = [
            self._fs(f"SUCCESS: Wrote 8 chars to '{wrote}'. "
                     f"Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs(f"SUCCESS: Deleted '{deleted}'."),
        ]
        assert self._mut(tools) == []

    def test_a_case_only_mismatch_cancels_without_an_alias(self):
        # ⚠ The parametrised spelling test above carries a `Script-side path`
        # clause, so the alias table cancels it and the case-folding in
        # `_fs_norm` is never the reason. On a confirmation with no second
        # spelling, case-folding is the only thing left — and this volume is
        # case-insensitive, so the two names are one file.
        tools = [
            self._fs("SUCCESS: Wrote 8 chars to 'Probe.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
        ]
        assert self._mut(tools) == []

    def test_a_project_scoped_write_cancels_under_EITHER_spelling(self):
        """⚠ THE ORIGINAL BUG, still live after five review rounds.

        The `Wrote` confirmation prints two spellings of one file: the
        model's `filename` and the resolved `rel_str`. Under a project-scoped
        sandbox they differ by the whole `projects/<pid>/` head — and the
        tool advertises the SECOND one as "Script-side path (from sandbox
        cwd)", which is the spelling the model then uses to delete it.
        Registering only the first left the delete uncancelled and reproduced
        the GlassOS false refute verbatim. 39% of the real write
        confirmations in the corpus print such a pair.
        """
        tools = [
            self._fs("SUCCESS: Wrote 900 chars to "
                     "'projects/abc123/index.html'. Script-side path "
                     "(from sandbox cwd): 'index.html'."),
            self._fs("SUCCESS: Wrote 391 chars to 'projects/abc123/probe.py'. "
                     "Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
        ]
        assert self._mut(tools) == ["projects/abc123/index.html"]

    def test_the_alias_key_does_not_retire_a_real_file_of_that_name(self):
        # ⚠ The script-side spelling is a BARE relative path, so it can name
        # a real file at the sandbox root. Deleting `probe.py` when the
        # request wrote BOTH `probe.py` and `projects/aaa/probe.py` retired
        # the project's file too — still on disk. A direct hit wins over the
        # alias table: if the deleted spelling names something this request
        # produced under that very name, that is what was deleted.
        for order in (0, 1):
            writes = [
                self._fs("SUCCESS: Wrote 100 chars to 'projects/aaa/probe.py'. "
                         "Script-side path (from sandbox cwd): 'probe.py'."),
                self._fs("SUCCESS: Wrote 100 chars to 'probe.py'."),
            ]
            if order:
                writes.reverse()
            tools = writes + [self._fs("SUCCESS: Deleted 'probe.py'.")]
            assert self._mut(tools) == ["projects/aaa/probe.py"], order

    def test_an_ambiguous_alias_retires_every_candidate(self):
        # Two projects writing `probe.py` map one alias key. Retiring only
        # the last writer is a coin flip; retiring every candidate can
        # silence an absence check but can never manufacture the false
        # REFUTE this whole change exists to remove.
        tools = [
            self._fs("SUCCESS: Wrote 10 chars to 'projects/aaa/probe.py'. "
                     "Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs("SUCCESS: Wrote 10 chars to 'projects/bbb/probe.py'. "
                     "Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
        ]
        assert self._mut(tools) == []

    def test_the_alias_does_not_merge_two_genuinely_different_files(self):
        # The alias is per-message: `probe.py` in project A must not cancel
        # a `probe.py` this turn wrote in project B.
        tools = [
            self._fs("SUCCESS: Wrote 10 chars to 'projects/aaa/probe.py'. "
                     "Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs("SUCCESS: Wrote 10 chars to 'projects/bbb/keep.py'. "
                     "Script-side path (from sandbox cwd): 'keep.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
        ]
        assert self._mut(tools) == ["projects/bbb/keep.py"]

    def test_deleting_a_scratch_DIRECTORY_retires_what_was_written_inside(self):
        # tool_delete_file rmtree()s a directory and emits the same
        # confirmation, so "write a scratch dir, run it, clean up" replayed
        # the incident verbatim while the file case was fixed.
        tools = [
            self._fs("SUCCESS: Wrote 280 chars to 'webos/index.html'."),
            self._fs("SUCCESS: Wrote 8 chars to 'tmp/probe.py'."),
            self._fs("SUCCESS: Deleted 'tmp'."),
        ]
        assert self._mut(tools) == ["webos/index.html"]

    def test_a_sibling_directory_with_a_shared_prefix_survives(self):
        tools = [
            self._fs("SUCCESS: Wrote 1 chars to 'tmpx/keep.py'."),
            self._fs("SUCCESS: Deleted 'tmp'."),
        ]
        assert self._mut(tools) == ["tmpx/keep.py"]

    def test_rewrite_after_delete_is_a_deliverable_again(self):
        # Order matters: the LAST op on a path decides whether it survives.
        tools = [
            self._fs("SUCCESS: Wrote 10 chars to 'out.csv'."),
            self._fs("SUCCESS: Deleted 'out.csv'."),
            self._fs("SUCCESS: Wrote 40 chars to 'out.csv'."),
        ]
        assert self._mut(tools) == ["out.csv"]

    def test_rewrite_inside_a_deleted_directory_is_a_deliverable_again(self):
        tools = [
            self._fs("SUCCESS: Wrote 8 chars to 'tmp/probe.py'."),
            self._fs("SUCCESS: Deleted 'tmp'."),
            self._fs("SUCCESS: Wrote 12 chars to 'tmp/probe.py'."),
        ]
        assert self._mut(tools) == ["tmp/probe.py"]

    def test_rename_drops_the_vanished_source_path(self):
        tools = [
            self._fs("SUCCESS: Wrote 12 chars to 'draft.md'."),
            self._fs("SUCCESS: Renamed/Moved 'draft.md' to 'final.md'."),
        ]
        assert self._mut(tools) == ["final.md"]

    def test_renaming_a_directory_retires_its_children(self):
        tools = [
            self._fs("SUCCESS: Wrote 12 chars to 'olddir/a.py'."),
            self._fs("SUCCESS: Renamed/Moved 'olddir' to 'newdir'."),
        ]
        assert self._mut(tools) == ["newdir"]

    def test_download_collects_the_destination_not_the_url(self):
        tools = [self._fs(
            "SUCCESS: Downloaded 'https://example.com/data.zip' to 'data.zip'.")]
        assert self._mut(tools) == ["data.zip"]

    def test_copy_collects_the_destination(self):
        tools = [self._fs("SUCCESS: Copied 'a.md' to 'b.md'.")]
        assert self._mut(tools) == ["b.md"]

    def test_only_the_path_is_captured_not_the_echoed_script_side_clause(self):
        # The write confirmation prints the model's spelling AND the
        # resolved one; a findall-style parse collected both and burned two
        # of the eight check slots on one file.
        tools = [self._fs(
            # ⚠ The fixture must use two spellings that do NOT normalise
            # to the same key, or the dict dedup hides a findall-style parse.
            # Under `project_scoped_sandbox` that is the everyday shape:
            # `filename` is the model's sandbox-root spelling while `rel_str`
            # is relative to the PROJECT dir, so they differ by that head.
            "SUCCESS: Wrote 8 chars to 'projects/e4e240b630f6/app.html'. "
            "Script-side path (from sandbox cwd): 'app.html'.")]
        assert self._mut(tools) == ["projects/e4e240b630f6/app.html"]

    def test_the_same_file_touched_twice_keeps_its_first_spelling(self):
        # Two calls, two spellings of one file: one entry, and the spelling
        # reported is the one first seen.
        tools = [
            self._fs("SUCCESS: Wrote 8 chars to 'index.html'."),
            self._fs("SUCCESS: Exact match found and replaced in "
                     "'/workspace/index.html'."),
        ]
        assert self._mut(tools) == ["index.html"]

    def test_a_filename_containing_an_apostrophe_round_trips(self):
        # ⚠ The write leg must be asserted POSITIVELY. Checking only that
        # write+delete cancels to [] cannot distinguish a correct round-trip
        # from a parser that saw NEITHER message — and the failure that
        # matters is the silent one, where a real deliverable never enters
        # the ground-truth re-read at all.
        wrote = self._fs("SUCCESS: Wrote 8 chars to 'vasilis's notes.md'. "
                         "Script-side path (from sandbox cwd): "
                         "'vasilis's notes.md'.")
        assert self._mut([wrote]) == ["vasilis's notes.md"]
        assert self._mut(
            [wrote, self._fs("SUCCESS: Deleted 'vasilis's notes.md'.")]) == []

    def test_a_confirmation_echoed_inside_an_edited_file_is_inert(self):
        # ⚠ CONSTRUCTED, not observed. The file tool is confined by
        # `_get_safe_path` to <GHOST_HOME>/sandbox and cannot reach this
        # repo, and no file in the live sandbox contains a confirmation
        # string. What is real is the mechanism: a replace whose echoed
        # POST-EDIT VIEW carries such a line would retire a live deliverable
        # if the guards were gone. Several overlapping guards stop it — the
        # patterns are start-anchored (`re.match`, no MULTILINE) AND only the
        # confirmation line is read — so no single one can be killed on its
        # own; this pins the behaviour they jointly guarantee, and the
        # multi-edit mutant in scripts/mutate_fs_ledger.py proves the pin is
        # real by stripping every copy at once.
        tools = [self._fs(
            "SUCCESS: Applied 1 SEARCH/REPLACE blocks to 'tests/t.py'.\n"
            "--- POST-EDIT VIEW of the file ON DISK (no re-read needed) ---\n"
            "  12 |     self._fs(\"SUCCESS: Deleted 'tests/t.py'.\"),\n"
            "  13 |     self._fs(\"SUCCESS: Wrote 9 chars to 'ghost.md'.\"),\n"
            "--- end post-edit view ---")]
        assert self._mut(tools) == ["tests/t.py"]

    def test_an_UNNUMBERED_echoed_confirmation_is_inert(self):
        # ⚠ The POST-EDIT VIEW above line-numbers what it echoes, so its
        # "  12 | " prefix alone stops that echo being parsed — which left the
        # guards it was meant to pin unreachable by the fixture. The fuzzy and
        # anchor rungs echo `--- REPLACED BLOCK (was) ---` followed by RAW,
        # UNNUMBERED source (tools/file_system.py), and that is the shape
        # where the guards actually earn their keep: without them the echoed
        # line retires the very deliverable the turn just wrote.
        tools = [self._fs(
            "SUCCESS: Fuzzy match (95% similar) found and replaced in "
            "'notes.md'. Your `old_text` did not byte-match.\n"
            "--- REPLACED BLOCK (was) ---\n"
            "SUCCESS: Deleted 'notes.md'.\n"
            "SUCCESS: Wrote 4 chars to 'decoy.md'.")]
        assert self._mut(tools) == ["notes.md"]

    def test_extensionless_deliverables_are_collected(self):
        # The old dotted-token rule silently dropped these; a 0-byte
        # Makefile was reported clean.
        tools = [
            self._fs("SUCCESS: Wrote 40 chars to 'Makefile'."),
            self._fs("SUCCESS: Wrote 40 chars to 'Dockerfile'."),
            self._fs("SUCCESS: Wrote 40 chars to 'report.markdown'."),
        ]
        assert self._mut(tools) == ["Makefile", "Dockerfile", "report.markdown"]

    def test_echoed_file_body_is_not_parsed_as_paths(self):
        # Everything after the first newline is the tool echoing the edited
        # SOURCE back (POST-EDIT VIEW / REPLACED BLOCK). Quoted strings in
        # there are code, not deliverables.
        tools = [self._fs(
            "SUCCESS: Applied 1 SEARCH/REPLACE blocks to 'app/index.html'.\n"
            "--- POST-EDIT VIEW of the file ON DISK (no re-read needed) ---\n"
            "  12 |   import x from './missing_helper.js';\n"
            "  13 |   fetch('/api/report.json');\n"
            "--- end post-edit view ---")]
        assert self._mut(tools) == ["app/index.html"]

    def test_an_inspect_peek_that_starts_with_SUCCESS_collects_nothing(self):
        # ⚠ A real world, not an invented one: tool_inspect_file returns the
        # peeked file's RAW first lines with no header of its own
        # (tools/file_system.py), so a log whose first line happens to start
        # with SUCCESS reaches this parser as though the tool had said it.
        # Only the known confirmation SHAPES may produce a path.
        tools = [self._fs("SUCCESS: all 12 checks passed for 'suite.py'.")]
        assert self._mut(tools) == []

    def test_a_genuinely_missing_written_file_still_refutes(self):
        # The fix must not defang the check: a file this turn WROTE and never
        # removed, absent from disk, is still a refute. "The fix works" and
        # "the fix disabled the check" otherwise produce identical green runs.
        d = Path(tempfile.mkdtemp())
        tools = [self._fs("SUCCESS: Wrote 500 chars to 'report.md'.")]
        assert self._mut(tools) == ["report.md"]
        r = GhostAgent._verify_file_artifacts(self._mut(tools), str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_glassos_incident_end_to_end_is_clean(self):
        """The exact incident: claim-prose + tool records + real on-disk
        state must produce NO override."""
        from ghost_agent.core.agent import _fs_path_ledger
        d = Path(tempfile.mkdtemp())
        (d / "webos").mkdir()
        (d / "webos" / "index.html").write_text("<!DOCTYPE html>...</html>")
        tools = [
            self._fs("SUCCESS: Wrote 24966 chars to 'webos/index.html'. "
                     "Script-side path (from sandbox cwd): 'webos/index.html'.\n"
                     "⚠ SYNTAX CHECK FAILED: 'webos/index.html' was written but "
                     "does NOT parse."),
            self._fs("SUCCESS: Applied 1 SEARCH/REPLACE blocks to "
                     "'webos/index.html'.\n"
                     "--- POST-EDIT VIEW of the file ON DISK (no re-read needed) ---\n"
                     "  290 |   const t=now.toLocaleTimeString([], "
                     "{hour:'2-digit',minute:'2-digit'});\n"
                     "--- end post-edit view ---"),
            self._fs("SUCCESS: Wrote 391 chars to 'probe.py'. "
                     "Script-side path (from sandbox cwd): 'probe.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
            self._fs("SUCCESS: Deleted '_chk.js'."),
        ]
        answer = (
            "The deliverable is confirmed complete and functional.\n\n"
            "**Location:** `/workspace/webos/index.html` (flat, standalone).\n"
            "- **File present:** 25,031 bytes / 511 lines\n"
        )
        claimed = _claimed_deliverable_files(answer)
        alive, retired = _fs_path_ledger(tools)
        to_check = list(claimed) + [m for m in alive if m not in claimed]
        assert to_check[:8] == ["webos/index.html"]
        soft = [p for p in retired if p not in to_check]
        assert GhostAgent._verify_file_artifacts(
            to_check[:8], str(d), soft=soft) is None


class TestPathLedgerSuppressionIsVisible:
    """A dropped path must stay *countable*. WEB-EXEC has no claim-prose
    leg — the written list is its only input — so silently dropping a
    removed page would skip the probe AND its inconclusive confidence cap,
    certifying at full confidence a build whose entry page is gone."""

    def _fs(self, content):
        return {"role": "tool", "name": "file_system", "content": content}

    def _ledger(self, tools):
        from ghost_agent.core.agent import _fs_path_ledger
        return _fs_path_ledger(tools)

    def test_retired_paths_are_reported_not_discarded(self):
        alive, retired = self._ledger([
            self._fs("SUCCESS: Wrote 100 chars to 'webos/index.html'."),
            self._fs("SUCCESS: Wrote 8 chars to 'probe.py'."),
            self._fs("SUCCESS: Deleted 'probe.py'."),
        ])
        assert alive == ["webos/index.html"]
        assert retired == ["probe.py"]

    def test_a_delete_of_a_path_this_request_never_produced_is_not_retired(self):
        # `retired` means "produced here, then removed here" — that is what
        # makes it evidence about THIS request. A stray delete is neither.
        alive, retired = self._ledger([
            self._fs("SUCCESS: Deleted '_chk.js'."),
        ])
        assert alive == [] and retired == []

    def test_a_page_renamed_to_backup_is_a_retired_entry_page(self):
        from ghost_agent.core.agent import (_retired_entry_pages,
                                            _web_artifacts_written)
        tools = [
            self._fs("SUCCESS: Wrote 100 chars to 'index.html'."),
            self._fs("SUCCESS: Renamed/Moved 'index.html' to 'index.html.bak'."),
        ]
        assert _web_artifacts_written(tools) == []
        assert _retired_entry_pages(tools) == ["index.html"]

    def test_backup_then_rewrite_leaves_no_retired_entry_page(self):
        from ghost_agent.core.agent import (_retired_entry_pages,
                                            _web_artifacts_written)
        tools = [
            self._fs("SUCCESS: Wrote 100 chars to 'index.html'."),
            self._fs("SUCCESS: Renamed/Moved 'index.html' to 'index.html.bak'."),
            self._fs("SUCCESS: Wrote 120 chars to 'index.html'."),
        ]
        assert _web_artifacts_written(tools) == ["index.html"]
        assert _retired_entry_pages(tools) == []

    def test_a_tidied_scratch_script_is_NOT_a_retired_entry_page(self):
        # Narrowness is the point: capping a turn whose deliverable was
        # never a web page would be a new false penalty.
        from ghost_agent.core.agent import _retired_entry_pages
        tools = [
            self._fs("SUCCESS: Wrote 40 chars to 'report.md'."),
            self._fs("SUCCESS: Wrote 40 chars to 'tmp/scratch.js'."),
            self._fs("SUCCESS: Deleted 'tmp/scratch.js'."),
        ]
        assert _retired_entry_pages(tools) == []


class TestSoftAndDirectoryResolution:
    def _dir(self):
        return Path(tempfile.mkdtemp())

    def test_a_removed_path_is_not_refuted_for_being_absent(self):
        d = self._dir()
        assert GhostAgent._verify_file_artifacts(
            [], str(d), soft=["probe.py"]) is None

    def test_a_removed_path_that_came_back_EMPTY_still_refutes(self):
        # Re-created by a route this parse cannot see (a shell rewrite via
        # `execute`) and landed 0 bytes — the arm the old check had.
        d = self._dir()
        (d / "report.md").write_text("")
        r = GhostAgent._verify_file_artifacts([], str(d), soft=["report.md"])
        assert r is not None and r.verdict == VerifyVerdict.REFUTED
        assert "empty" in r.reasoning and "report.md" in r.reasoning

    def test_a_removed_path_that_came_back_with_content_is_clean(self):
        d = self._dir()
        (d / "report.md").write_text("real content")
        assert GhostAgent._verify_file_artifacts(
            [], str(d), soft=["report.md"]) is None

    def test_a_produced_DIRECTORY_is_present_not_missing(self):
        # `Copied 'webos' to 'backup'` / `Renamed/Moved 'build' to 'dist'`
        # name a directory. The checker only accepted is_file(), so every
        # such op refuted — a regression the shape-aware branches
        # introduced, since a dotted-token parse never saw a bare dir name.
        d = self._dir()
        (d / "dist").mkdir()
        (d / "dist" / "index.html").write_text("<html></html>")
        assert GhostAgent._verify_file_artifacts(["dist"], str(d)) is None

    def test_the_basename_fallback_will_not_certify_a_different_file(self):
        # ⚠ `rglob(basename)` took the first filesystem hit with no path
        # check, so a claim of `webos/index.html` was reported CLEAN by a
        # stale `backup/index.html`. A fallback hit must END WITH the claim.
        d = self._dir()
        (d / "backup").mkdir()
        (d / "backup" / "index.html").write_text("stale copy")
        r = GhostAgent._verify_file_artifacts(
            ["/workspace/webos/index.html"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_project_scoped_claim_resolves_against_the_project_root(self):
        """⚠ THE DOMINANT REAL SHAPE. `project_scoped_sandbox` hands this
        check a host_dir that IS `<sandbox>/projects/<pid>`, while paths are
        spelled from the sandbox root (`projects/<pid>/app.html`, often with
        a `/workspace/` head) — so the claim carries a prefix the root has
        already consumed.

        The fixture passes `extra_roots` BECAUSE THE CALLER ALWAYS DOES when
        scoped — that fallback's exact-path hit is what resolves this shape.
        An earlier version omitted it and thereby pinned a claim-deeper
        suffix arm that was an exact substitute for the fallback (0 of 142
        replayed verdicts changed with it disabled) and doubled as a
        shallow-namesake certification hole; that arm is deleted, and a pin
        must not resurrect it by testing a configuration the caller never
        runs.
        """
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "e4e240b630f6"
        proj.mkdir(parents=True)
        (proj / "app.html").write_text("<html>the deliverable</html>")
        for claim in ("projects/e4e240b630f6/app.html",
                      "/workspace/projects/e4e240b630f6/app.html",
                      "workspace/projects/e4e240b630f6/app.html",
                      "app.html"):
            assert GhostAgent._verify_file_artifacts(
                [claim], str(proj), extra_roots=[str(root)]) is None, claim

    def test_a_claim_whose_prefix_the_root_does_not_stand_for_refutes(self):
        # ⚠ The claim-deeper arm is DELETED: a deep claim is never answered
        # by a shallow namesake — `<sbx>/index.html` cannot answer
        # `projects/webos/index.html` on a turn that wrote nothing. (Ledger
        # pair-writes get their rel_str consulted instead; prose claims are
        # absence-ignored.) These fixtures pin the rejection.
        d = Path(tempfile.mkdtemp())
        (d / "index.html").write_text("an unrelated page at the root")
        for claim in ("projects/webos/index.html", "a/b/c/index.html"):
            r = GhostAgent._verify_file_artifacts([claim], str(d))
            assert r is not None and r.verdict == VerifyVerdict.REFUTED, claim

    def test_a_project_scoped_claim_that_is_really_absent_still_refutes(self):
        # The symmetric rule must not become a fail-open.
        d = Path(tempfile.mkdtemp())
        r = GhostAgent._verify_file_artifacts(["projects/x/gone.md"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_the_fallback_respects_directory_boundaries(self):
        # ⚠ Without the "/" in the hit-deeper comparison, `ax/report.pdf`
        # would satisfy a claim of `x/report.pdf` — a wrong-file
        # certification the disjoint-name fixture above cannot catch, since
        # those two names share no suffix at all. Same bug class the ledger
        # already pins with `tmpx` vs `tmp`. (The mirror-direction arm this
        # class once also pinned was deleted outright.)
        d = self._dir()
        (d / "ax").mkdir()
        (d / "ax" / "report.pdf").write_text("a different file")
        r = GhostAgent._verify_file_artifacts(["x/report.pdf"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_the_fallback_respects_boundaries_in_the_MIRROR_direction(self):
        # ⚠ The separator was pinned on one arm only. Dropping the "/" from
        # `want.endswith("/" + hit)` — the claim-is-deeper arm — lets an
        # on-disk `deep/app.html` answer a claim of `abcdeep/app.html`. Same
        # asymmetry the CODE was just fixed for, still present in the tests.
        d = self._dir()
        (d / "deep").mkdir()
        (d / "deep" / "app.html").write_text("a different file")
        r = GhostAgent._verify_file_artifacts(["abcdeep/app.html"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_an_escaping_dotdot_claim_never_resolves(self):
        # ⚠ `cand = root / rel` joins the RAW spelling, so a claim whose
        # normalised form escapes the root walked OUT of the sandbox on the
        # exact-path probe — an empty host file one level up produced a
        # "claimed-but-empty" refute about a file no turn touched. Interior
        # `..` that stays inside still resolves.
        import os
        d = Path(tempfile.mkdtemp())
        outside = d.parent / f"escape_{os.getpid()}.md"
        outside.write_text("")
        try:
            assert GhostAgent._verify_file_artifacts(
                [], str(d), soft=[f"../{outside.name}"]) is None
            r = GhostAgent._verify_file_artifacts(
                [f"../{outside.name}"], str(d))
            assert r is not None       # hard: absent, not the host file
            joined = "; ".join(r.issues or [])
            assert "claimed-but-missing" in joined
            assert "claimed-but-empty" not in joined
        finally:
            outside.unlink()
        (d / "sub").mkdir()
        (d / "sub" / "x.md").write_text("real")
        assert GhostAgent._verify_file_artifacts(
            ["sub/../sub/x.md"], str(d)) is None

    def test_the_basename_fallback_still_finds_a_nested_match(self):
        d = self._dir()
        (d / "sub").mkdir()
        (d / "sub" / "deep.txt").write_text("x")
        assert GhostAgent._verify_file_artifacts(["deep.txt"], str(d)) is None
        assert GhostAgent._verify_file_artifacts(
            ["sub/deep.txt"], str(d)) is None


class TestResolveFailsOpenNotClosed:
    """⚠ An error is not evidence of absence.

    The `_resolve` helper's inner `except` returned the same value as "file
    not found", and the caller reads that as MISSING — a refute at 0.9. So a
    bind mount vanishing mid-read, or a parent directory the host cannot
    traverse (the sandbox is a container bind mount, and a root-owned 0700
    dir is reachable), produced the exact false refute this whole change
    exists to remove. The code it replaced skipped such a name entirely.
    """

    def test_an_internal_error_never_becomes_a_refute(self):
        from unittest.mock import patch
        d = Path(tempfile.mkdtemp())
        (d / "out.csv").write_text("real content")
        with patch("pathlib.Path.rglob",
                   side_effect=RuntimeError("bind mount went away")):
            r = GhostAgent._verify_file_artifacts(["projects/p1/out.csv"],
                                                  str(d))
        assert r is None

    def test_an_internal_error_does_not_suppress_a_real_refute(self):
        """⚠ Failing open must not become failing BLIND: an unreadable path
        must be skipped while the others in the same batch are still judged.

        The first version of this test imported `patch`, never used it,
        induced no error, and asserted what `test_missing_file_refutes`
        already asserts — it could not fail in the world its name describes.
        """
        import os
        d = Path(tempfile.mkdtemp())
        locked = d / "locked"
        locked.mkdir()
        (locked / "unreadable.md").write_text("x")
        os.chmod(locked, 0o000)
        try:
            rep = {}
            r = GhostAgent._verify_file_artifacts(
                ["locked/unreadable.md", "gone.md"], str(d), report=rep)
            # the unreadable one is skipped, the genuinely absent one refutes
            assert r is not None and r.verdict == VerifyVerdict.REFUTED
            joined = "; ".join(r.issues)
            assert "gone.md" in joined
            assert "unreadable.md" not in joined
            assert rep["unresolvable"]
        finally:
            os.chmod(locked, 0o700)


class TestDirectoriesAndEmptyByDesign:
    def _dir(self):
        return Path(tempfile.mkdtemp())

    def test_a_directory_cannot_answer_a_PROSE_claim(self):
        # A prose claim always names a file — the extractor requires a
        # deliverable extension — so a directory must never satisfy one.
        d = self._dir()
        (d / "report.pdf").mkdir()
        r = GhostAgent._verify_file_artifacts(
            ["report.pdf"], str(d), file_claims={"report.pdf"})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_the_same_rule_holds_for_a_nested_prose_claim(self):
        # ⚠ The dot heuristic that preceded this was applied on the direct
        # branch only, so a directory one level down slipped through the
        # rglob fallback and skipped the emptiness check too.
        d = self._dir()
        (d / "out").mkdir()
        (d / "out" / "report.pdf").mkdir()
        r = GhostAgent._verify_file_artifacts(
            ["report.pdf"], str(d), file_claims={"report.pdf"})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    @pytest.mark.parametrize("dirname", [
        "backup.old", "release.2026", "site.com", "my.data", "v1.2"])
    def test_a_LEDGER_directory_with_a_dot_is_present(self, dirname):
        # ⚠ `Copied 'src' to 'backup.old'` and `Renamed/Moved 'dist' to
        # 'release.2026'` name DIRECTORIES that legitimately contain dots. A
        # dot-based heuristic refuted every one of them, with no repair
        # possible short of deleting the tree and writing a file in its
        # place — the destructive shape the empty-by-design exemption exists
        # to avoid. Directory-ness is decided by the CLAIM'S SOURCE, not by
        # its spelling.
        d = self._dir()
        (d / dirname).mkdir()
        (d / dirname / "x.txt").write_text("content")
        assert GhostAgent._verify_file_artifacts([dirname], str(d)) is None

    def test_an_extensionless_directory_is_still_present(self):
        d = self._dir()
        (d / "backup").mkdir()
        (d / "backup" / "x.py").write_text("y")
        assert GhostAgent._verify_file_artifacts(["backup"], str(d)) is None

    def test_the_empty_by_design_exemption_applies_to_the_soft_arm_too(self):
        # ⚠ The exemption lived in the claimed loop only, so every exempt
        # name still refuted through `soft` — and that arm is reachable
        # through the file tool alone, since retiring a tree retires the
        # files under it while a copy only produces the destination name.
        d = self._dir()
        (d / "data").mkdir()
        (d / "data" / ".gitkeep").write_text("")
        assert GhostAgent._verify_file_artifacts(
            [], str(d), soft=["data/.gitkeep"]) is None

    def test_an_empty_file_the_PROSE_claims_content_for_still_refutes(self):
        # The exemption is for files nobody claimed to have filled. If the
        # answer says it wrote one, emptiness is a real defect.
        d = self._dir()
        (d / "pkg").mkdir()
        (d / "pkg" / "__init__.py").write_text("")
        r = GhostAgent._verify_file_artifacts(
            ["pkg/__init__.py"], str(d), file_claims={"pkg/__init__.py"})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_an_empty_gitignore_is_not_exempt(self):
        # Unlike __init__.py / py.typed / .gitkeep, an empty .gitignore is a
        # defect rather than the correct state.
        d = self._dir()
        (d / ".gitignore").write_text("")
        r = GhostAgent._verify_file_artifacts([".gitignore"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_file_that_is_meant_to_be_empty_does_not_refute(self):
        # ⚠ An empty __init__.py is CORRECT. Refuting on it opens a repair
        # round whose only escapes are corrupting or deleting the file —
        # the original bug's shape, in the arm that never had it.
        d = self._dir()
        (d / "pkg").mkdir()
        (d / "pkg" / "__init__.py").write_text("")
        assert GhostAgent._verify_file_artifacts(
            ["pkg/__init__.py"], str(d)) is None

    def test_an_ordinary_empty_file_still_refutes(self):
        d = self._dir()
        (d / "report.md").write_text("")
        r = GhostAgent._verify_file_artifacts(["report.md"], str(d))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED


class TestRefuteNamesTheFile:
    """⚠ The bounded auto-repair directive is built from `issues`, not from
    `reasoning` (`"; ".join(_vr.issues[:3])`). A bare "claimed-but-missing
    deliverable" told the model a file was missing and never which one —
    most of this check's value discarded at the last step."""

    def test_the_issue_string_carries_the_filenames(self):
        d = Path(tempfile.mkdtemp())
        (d / "there.md").write_text("")
        r = GhostAgent._verify_file_artifacts(["gone.csv", "there.md"], str(d))
        joined = "; ".join(r.issues[:3])
        assert "gone.csv" in joined and "there.md" in joined
        assert "missing" in joined and "empty" in joined
        # and no empty strings, which would render as a stray "; "
        assert all(i.strip() for i in r.issues)


class TestBannerPrefixedRecords:
    def test_a_failure_banner_does_not_hide_a_successful_file_op(self):
        # ⚠ The banner is prepended to ANY tool result containing
        # "Traceback" — not gated on the tool — and the prefixed text is what
        # gets recorded. An ordinary successful edit whose echoed POST-EDIT
        # VIEW contains that word therefore failed the SUCCESS gate, and the
        # whole record vanished: FILE-ARTIFACT and WEB-EXEC both went silent
        # on a turn that really did write files.
        from ghost_agent.core.agent import (_files_mutated_this_turn,
                                            _web_artifacts_written)
        tools = [{"role": "tool", "name": "file_system", "content": (
            "[FAILURE BANNER] Traceback (most recent call last):\n"
            "SUCCESS: Applied 1 SEARCH/REPLACE blocks to 'app/index.html'.\n"
            "--- POST-EDIT VIEW of the file ON DISK (no re-read needed) ---\n"
            "  9 |   log('Traceback (most recent call last):')")}]
        assert _files_mutated_this_turn(tools) == ["app/index.html"]
        assert _web_artifacts_written(tools) == ["app/index.html"]

    def test_a_banner_over_a_non_success_result_still_collects_nothing(self):
        from ghost_agent.core.agent import _files_mutated_this_turn
        tools = [{"role": "tool", "name": "file_system", "content": (
            "[FAILURE BANNER] Traceback (most recent call last):\n"
            "Error: could not write 'broken.js'.")}]
        assert _files_mutated_this_turn(tools) == []


class TestFallbackRootCannotReachAnotherProject:
    """⚠ The un-scoped fallback root is not a free second search.

    It exists because a turn's writes can land at the sandbox ROOT while the
    binding still reads as project-scoped. But `_resolve`'s suffix rule
    accepts any hit ending with the claim, and every project has an `app.py`
    — so a plain second pass certified a SIBLING project's file as this
    project's deliverable, and (worse) rescued a genuinely corrupt local file
    by finding a healthy namesake next door.
    """

    def _tree(self):
        root = Path(tempfile.mkdtemp())
        mine = root / "projects" / "mine"
        other = root / "projects" / "other"
        mine.mkdir(parents=True)
        other.mkdir(parents=True)
        return root, mine, other

    def test_a_sibling_projects_file_cannot_answer_the_claim(self):
        root, mine, other = self._tree()
        (other / "app.py").write_text("someone else's work")
        r = GhostAgent._verify_file_artifacts(
            ["app.py"], str(mine), extra_roots=[str(root)])
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_healthy_sibling_cannot_rescue_a_corrupt_local_file(self):
        root, mine, other = self._tree()
        (mine / "app.py").write_text("")            # the real, broken one
        (other / "app.py").write_text("fine here")
        r = GhostAgent._verify_file_artifacts(
            ["app.py"], str(mine), extra_roots=[str(root)])
        assert r is not None and "empty" in r.reasoning

    def test_a_claim_naming_another_projects_file_is_refuted(self):
        # ⚠ The rglob restriction and the foreign-path guard were redundant
        # for the SEARCH path, each masking the other's mutant — but the
        # plain `root/rel` hit went through neither, so a claim that spells
        # another project's path resolved under the fallback root and was
        # certified. Two guards that only overlap partially are not two
        # guards; they are one guard with a hole where they do not.
        root, mine, other = self._tree()
        (other / "app.py").write_text("someone else's work")
        r = GhostAgent._verify_file_artifacts(
            ["projects/other/app.py"], str(mine), extra_roots=[str(root)])
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_two_spellings_of_one_claim_cannot_vouch_for_each_other(self):
        # ⚠ The self-exclusion compared RAW strings while everything else in
        # the slice compares through `_fs_norm`, so an answer naming
        # `projects/bbb/x.py` AND `/workspace/projects/bbb/x.py` let the two
        # spellings make each other "own" — and the guard evaporated exactly
        # where the claim reaches furthest. Two completion verbs in one
        # sentence is all it takes.
        root, mine, other = self._tree()
        (other / "secret.py").write_text("theirs")
        r = GhostAgent._verify_file_artifacts(
            ["projects/other/secret.py"], str(mine), extra_roots=[str(root)],
            file_claims={"projects/other/secret.py",
                         "/workspace/projects/other/secret.py"})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_my_own_projects_path_still_resolves_through_the_fallback(self):
        root, mine, _ = self._tree()
        (mine / "own.py").write_text("mine")
        assert GhostAgent._verify_file_artifacts(
            ["projects/mine/own.py"], str(mine), extra_roots=[str(root)]) is None

    def test_the_fallback_still_finds_a_file_at_the_sandbox_root(self):
        # The case the fallback exists for: writes landed outside the
        # project dir, but not inside another project.
        root, mine, _ = self._tree()
        (root / "deliverable.md").write_text("real content")
        assert GhostAgent._verify_file_artifacts(
            ["deliverable.md"], str(mine), extra_roots=[str(root)]) is None


class TestUnscopedTurnsAreGuardedToo:
    """⚠ EVERY pin for the foreign-project guard used the project-SCOPED
    shape, and the guard was gated on `root != primary` — so on an unscoped
    turn, where `project_scoped_sandbox` returns the sandbox ROOT as primary
    and there is no fallback at all, it was dead code exactly where it was
    needed. The basename search descends into `projects/*` from there."""

    def _sandbox(self):
        sbx = Path(tempfile.mkdtemp())
        (sbx / "projects" / "other").mkdir(parents=True)
        return sbx

    def test_no_project_active_means_no_project_dir_may_answer(self):
        sbx = self._sandbox()
        (sbx / "projects" / "other" / "index.html").write_text("theirs")
        r = GhostAgent._verify_file_artifacts(["index.html"], str(sbx))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_an_unscoped_turn_may_answer_from_the_project_it_names(self):
        # ⚠ REGRESSION the foreign-project guard introduced. `own_project` is
        # None on an unscoped turn — and the sandbox root is exactly where
        # turns write into `projects/<pid>/`. Reading "no project bound" as
        # "no project directory may answer" refuted a turn that wrote
        # `projects/<pid>/app.py` and said "I wrote app.py", while the ledger
        # entry for the SAME FILE resolved clean in the same call. The turn's
        # own paths name its project.
        sbx = self._sandbox()
        mine = sbx / "projects" / "7b62e5e533d1"
        mine.mkdir()
        (mine / "unified.md").write_text("the real deliverable")
        assert GhostAgent._verify_file_artifacts(
            ["unified.md", "projects/7b62e5e533d1/unified.md"],
            str(sbx)) is None

    def test_a_file_outside_any_project_still_answers(self):
        sbx = self._sandbox()
        (sbx / "webos").mkdir()
        (sbx / "webos" / "index.html").write_text("this turn's page")
        assert GhostAgent._verify_file_artifacts(
            ["webos/index.html"], str(sbx)) is None

    @pytest.mark.parametrize("claim", [
        "Projects/other/secret.py",          # case: the volume resolves it
        "projects/mine/../other/secret.py",  # `..` walks back out
    ])
    def test_the_foreign_guard_normalises_before_comparing(self, claim):
        # The guard decides access by string comparison while the filesystem
        # resolves what the comparison rejected.
        sbx = self._sandbox()
        mine = sbx / "projects" / "mine"
        mine.mkdir()
        (sbx / "projects" / "other" / "secret.py").write_text("theirs")
        r = GhostAgent._verify_file_artifacts(
            [claim], str(mine), extra_roots=[str(sbx)])
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_prose_membership_is_matched_on_the_NORMALISED_key(self):
        # ⚠ Everything else in this slice compares through `_fs_norm`;
        # membership here was by exact string, so a prose claim spelled
        # `/workspace/pkg/__init__.py` did not match the check running on
        # `pkg/__init__.py`, and the empty-by-design exemption fired on a
        # file the answer said it had filled. The invariant that made that
        # unreachable lived 300 lines away in the caller — one loop-order
        # change from silently inverting.
        d = Path(tempfile.mkdtemp())
        (d / "pkg").mkdir()
        (d / "pkg" / "__init__.py").write_text("")
        r = GhostAgent._verify_file_artifacts(
            ["pkg/__init__.py"], str(d),
            file_claims={"/workspace/pkg/__init__.py"})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_directory_claim_stops_rather_than_redirecting(self):
        # ⚠ "A directory may not satisfy a prose claim" was implemented as a
        # SKIP, so the search continued and answered with an unrelated
        # same-named file elsewhere. The rule delivered "something else may".
        d = Path(tempfile.mkdtemp())
        (d / "app.js").mkdir()
        (d / "vendor").mkdir()
        (d / "vendor" / "app.js").write_text("nobody produced this")
        r = GhostAgent._verify_file_artifacts(
            ["app.js"], str(d), file_claims={"app.js"})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED


class TestReportCountersAreArmSeparated:
    def test_a_soft_hit_does_not_satisfy_the_claimed_arms_counter(self):
        # ⚠ One counter for both arms let a written-then-removed file that
        # came back satisfy the caller's "nothing was readable" gate while
        # every actual deliverable sat behind an unreadable directory — and
        # the operator was told "clean … present + non-empty".
        import os
        d = Path(tempfile.mkdtemp())
        locked = d / "locked"
        locked.mkdir()
        (locked / "report.pdf").write_text("x")
        (d / "scratch.txt").write_text("came back")
        os.chmod(locked, 0o000)
        try:
            rep = {}
            GhostAgent._verify_file_artifacts(
                ["locked/report.pdf"], str(d), soft=["scratch.txt"], report=rep)
            assert rep["resolved"] == 0          # no CLAIMED path was read
            assert rep["resolved_soft"] == 1
            assert rep["unresolvable"]
        finally:
            os.chmod(locked, 0o700)

    def test_one_path_erroring_under_two_roots_is_reported_once(self):
        import os
        sbx = Path(tempfile.mkdtemp())
        proj = sbx / "projects" / "mine"
        proj.mkdir(parents=True)
        for base in (proj, sbx):
            lk = base / "locked"
            lk.mkdir()
            os.chmod(lk, 0o000)
        try:
            rep = {}
            GhostAgent._verify_file_artifacts(
                ["locked/out.txt"], str(proj), extra_roots=[str(sbx)],
                report=rep)
            assert len(rep["unresolvable"]) == 1
        finally:
            for base in (proj, sbx):
                os.chmod(base / "locked", 0o700)

    def test_one_file_under_two_spellings_is_named_once(self):
        # ⚠ Two claimed spellings can resolve to ONE file (`_fs_norm` does
        # not collapse interior `..`), and a dedup keyed on the claim
        # STRINGS can never fire for them — the empty file was named twice
        # in the repair directive. Dedup is on the resolved path. (The
        # original fixture used a soft bare-basename spelling, which the
        # soft arm's exact-only rule now skips before the dedup is ever
        # reached — this pair still reaches it.)
        d = Path(tempfile.mkdtemp())
        (d / "sub").mkdir()
        (d / "sub" / "out.txt").write_text("")
        r = GhostAgent._verify_file_artifacts(
            ["sub/out.txt", "sub/../sub/out.txt"], str(d))
        assert r is not None
        assert "; ".join(r.issues).count("out.txt") == 1


class TestRemovalAwareAbsence:
    """⚠ Absence after a CONFIRMED write proves removal, not non-production.
    The two removal routes that leave no file_system confirmation — a later
    shell/script step, and a workspace sweep racing the re-read — downgrade
    the missing-refute to a logged skip. Both are evidence-gated: ORDER for
    the shell route, the sweep's own recency-stamped mark for the sweep."""

    def _fs(self, c):
        return {"role": "tool", "name": "file_system", "content": c}

    def _exec(self):
        return {"role": "tool", "name": "execute",
                "content": "--- COMMAND RESULT ---\nEXIT CODE: 0\n..."}

    def test_a_write_followed_by_execute_does_not_refute_on_absence(self):
        from ghost_agent.core.agent import _keys_removable_after_write
        d = Path(tempfile.mkdtemp())
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
                 self._exec()]
        rep = {}
        r = GhostAgent._verify_file_artifacts(
            ["probe.py"], str(d), report=rep,
            removable_keys=_keys_removable_after_write(tools))
        assert r is None
        assert rep["skipped_removable"] == ["probe.py"]

    def test_an_execute_BEFORE_the_write_proves_nothing(self):
        # ⚠ order is the evidence — without it every turn that ran any
        # script would lose its absence check entirely.
        from ghost_agent.core.agent import _keys_removable_after_write
        d = Path(tempfile.mkdtemp())
        tools = [self._exec(),
                 self._fs("SUCCESS: Wrote 9 chars to 'probe.py'.")]
        r = GhostAgent._verify_file_artifacts(
            ["probe.py"], str(d),
            removable_keys=_keys_removable_after_write(tools))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_read_only_tool_after_the_write_does_not_disarm(self):
        # ⚠ the first gate was a NAME list that included `workspace` (a
        # read-only reporting tool) and `system_utility` (diagnostics) —
        # each silently neutralised real turns' absence checks. The gate is
        # evidence now: the record must carry the execute tool's own result
        # markers.
        from ghost_agent.core.agent import _keys_removable_after_write
        d = Path(tempfile.mkdtemp())
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
                 {"role": "tool", "name": "workspace",
                  "content": "WORKSPACE SUMMARY: 12 files, 3 changed"}]
        assert _keys_removable_after_write(tools) == set()
        r = GhostAgent._verify_file_artifacts(
            ["probe.py"], str(d),
            removable_keys=_keys_removable_after_write(tools))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_composed_skill_with_a_shell_step_disarms_by_evidence(self):
        # arbitrarily-named composed skills embed the execute tool's result
        # header verbatim for their shell steps — that is the evidence.
        from ghost_agent.core.agent import _keys_removable_after_write
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
                 {"role": "tool", "name": "auto_cleanup_task",
                  "content": "step 2 result:\n--- COMMAND RESULT ---\n"
                             "EXIT CODE: 0"}]
        assert _keys_removable_after_write(tools) == {"probe.py"}

    def test_a_web_page_QUOTING_ci_output_does_not_disarm(self):
        # ⚠ the marker is anchored at line start AND retrieval surfaces are
        # excluded outright: browser PAGE TEXT embeds third-party text
        # verbatim, and a page quoting CI output must not disarm the check —
        # a browser cannot run a shell.
        from ghost_agent.core.agent import _keys_removable_after_write
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
                 {"role": "tool", "name": "browser",
                  "content": "--- PAGE TEXT (capped preview) ---\n"
                             "--- COMMAND RESULT ---\nEXIT CODE: 0"}]
        assert _keys_removable_after_write(tools) == set()

    def test_a_midline_quoted_marker_does_not_disarm(self):
        from ghost_agent.core.agent import _keys_removable_after_write
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
                 {"role": "tool", "name": "some_summary_tool",
                  "content": "the log said --- COMMAND RESULT --- earlier"}]
        assert _keys_removable_after_write(tools) == set()

    def test_pair_spellings_are_decided_as_ONE_file(self):
        # ⚠ a pair-write's two spellings are one file: a post-exec rewrite
        # under the rel_str spelling used to leave the model spelling
        # "removable" for the same bytes.
        from ghost_agent.core.agent import _keys_removable_after_write
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'projects/p/x.md'. "
                          "Script-side path (from sandbox cwd): 'x.md'."),
                 self._exec(),
                 self._fs("SUCCESS: Wrote 9 chars to 'x.md'.")]
        assert _keys_removable_after_write(tools) == set()

    class _SP:
        """Dict-backed scratchpad honouring the REAL sentinel keys — these
        pins must distinguish live wiring, not monkeypatched wiring: the
        first stamp site was proven unreachable by its own call sites while
        a monkeypatched pin stayed green (round 14)."""
        def __init__(self):
            self.d = {}
        def get(self, k, default=None):
            return self.d.get(k, default)
        def set(self, k, v, **kw):
            self.d[k] = v
        def delete(self, k):
            self.d.pop(k, None)

    def _bound_ctx(self, conv, pid):
        from unittest.mock import MagicMock
        from ghost_agent.tools.file_system import (_PROJECT_BIND_PID,
                                                   _PROJECT_BIND_CONV)
        ctx = MagicMock()
        ctx.conversation_key = conv
        ctx.scratchpad = self._SP()
        if pid:
            ctx.scratchpad.d[_PROJECT_BIND_PID] = pid
            ctx.scratchpad.d[_PROJECT_BIND_CONV] = conv
        ctx.current_project_id = pid
        ctx.last_closed_project = None
        return ctx

    def test_a_genuine_close_stamps_through_the_LIVE_route(self, monkeypatch):
        """⚠ THE ROUND-14 FINDING. The first stamp lived in
        `_park_current_project`, whose only callers are the hygiene
        branches — where ownership is false BY CONSTRUCTION — so the stamp
        was unreachable from every production path and the same-turn-close
        false-refute class it closed was silently back. Every genuine close
        (exit, archive, hard delete) passes through `_set_current(ctx,
        None)`; the stamp now lives there, one line before the sentinel it
        reads is deleted. This pin drives the REAL route with REAL sentinel
        keys — no monkeypatched ownership."""
        import ghost_agent.tools.projects as PJ
        monkeypatch.setattr(PJ, "_snapshot_scratchpad",
                            lambda ctx, pid: True)   # not under test
        ctx = self._bound_ctx("conv-9", "pid-9")
        PJ._set_current(ctx, None)
        assert ctx.last_closed_project == ("conv-9", "pid-9")
        from ghost_agent.tools.file_system import _PROJECT_BIND_PID
        assert _PROJECT_BIND_PID not in ctx.scratchpad.d   # sentinel gone

    def test_closing_a_binding_another_conversation_owns_stamps_nothing(
            self, monkeypatch):
        import ghost_agent.tools.projects as PJ
        from ghost_agent.tools.file_system import _PROJECT_BIND_CONV
        monkeypatch.setattr(PJ, "_snapshot_scratchpad",
                            lambda ctx, pid: True)
        ctx = self._bound_ctx("conv-9", "pid-9")
        ctx.scratchpad.d[_PROJECT_BIND_CONV] = "conv-OTHER"   # theirs
        PJ._set_current(ctx, None)
        assert ctx.last_closed_project is None

    def test_a_hygiene_park_stamps_nothing(self, monkeypatch):
        import ghost_agent.tools.projects as PJ
        monkeypatch.setattr(PJ, "_snapshot_scratchpad",
                            lambda ctx, pid: True)
        ctx = self._bound_ctx("conv-9", None)      # binds nothing
        PJ._park_current_project(ctx, "someone-elses-pid", "hygiene")
        assert ctx.last_closed_project is None

    def test_a_REWRITE_after_the_exec_still_refutes_on_absence(self):
        # ⚠ each key is decided exactly once, at its LAST produce. Deciding
        # on every produce let an earlier write of a re-written file
        # downgrade the check even though the last write postdated every
        # exec — a script that ran before the final write cannot have
        # removed it.
        from ghost_agent.core.agent import _keys_removable_after_write
        d = Path(tempfile.mkdtemp())
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'a.py'."),
                 self._exec(),
                 self._fs("SUCCESS: Wrote 9 chars to 'a.py'.")]
        assert _keys_removable_after_write(tools) == set()
        r = GhostAgent._verify_file_artifacts(
            ["a.py"], str(d),
            removable_keys=_keys_removable_after_write(tools))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_removable_EMPTY_file_still_refutes(self):
        # the downgrade is for ABSENCE only: a file that is back on disk
        # and empty is a real defect regardless of what ran after it.
        from ghost_agent.core.agent import _keys_removable_after_write
        d = Path(tempfile.mkdtemp())
        (d / "probe.py").write_text("")
        tools = [self._fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
                 self._exec()]
        r = GhostAgent._verify_file_artifacts(
            ["probe.py"], str(d),
            removable_keys=_keys_removable_after_write(tools))
        assert r is not None and "empty" in "; ".join(r.issues)

    def _tree_pid(self):
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        return root, proj

    def test_a_sweep_that_NAMED_the_file_explains_its_absence(self):
        root, proj = self._tree_pid()
        rep = {}
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/report.md"], str(proj), extra_roots=[str(root)],
            report=rep, swept_removals={"pid": ["report.md"]})
        assert r is None
        assert rep["skipped_removable"]

    def test_a_sweep_that_deleted_OTHER_debris_explains_nothing(self):
        # ⚠ the first mark carried no file list, so a tidy that removed one
        # stale screenshot "explained" the absence of app.py — a file tidy
        # is structurally incapable of deleting. The mark carries the
        # deleted list now, and it must actually NAME the file.
        root, proj = self._tree_pid()
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/app.py"], str(proj), extra_roots=[str(root)],
            swept_removals={"pid": ["debug_shot.png"]})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_hard_delete_explains_everything_under_its_project(self):
        root, proj = self._tree_pid()
        rep = {}
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/app.py"], str(proj), extra_roots=[str(root)],
            report=rep, swept_removals={"pid": None})
        assert r is None and rep["skipped_removable"]

    def test_a_sweep_of_ANOTHER_project_explains_nothing(self):
        root, proj = self._tree_pid()
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/report.md"], str(proj), extra_roots=[str(root)],
            swept_removals={"other": None})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_mixed_case_pid_still_matches_its_mark(self):
        # `_project_dir` lowercases; the mark used to stamp the raw id and
        # the verdict missed in the FALSE-REFUTE direction.
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "myproj"
        proj.mkdir(parents=True)
        rep = {}
        r = GhostAgent._verify_file_artifacts(
            ["projects/MyProj/report.md"], str(proj),
            extra_roots=[str(root)], report=rep,
            swept_removals={"myproj": ["report.md"]})
        assert r is None and rep["skipped_removable"]

    def test_a_deleted_nested_namesake_does_not_explain_a_top_level_file(self):
        # ⚠ the first match accepted bare-basename equality: a deleted
        # `vendor/x.md` "explained" a missing top-level `x.md`. Path-wise,
        # project-relative equality only.
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/x.md"], str(proj), extra_roots=[str(root)],
            swept_removals={"pid": ["vendor/x.md"]})
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_second_sweep_unions_with_a_fresh_earlier_mark(self):
        from ghost_agent.core.workspace_cleanup import _mark_removal
        class _Store: pass
        st = _Store()
        _mark_removal(st, "pid", ["a.md"])
        _mark_removal(st, "pid", ["b.md"])
        assert set(st.recent_workspace_removals["pid"][1]) == {"a.md", "b.md"}
        _mark_removal(st, "pid", None)          # whole-tree absorbs
        assert st.recent_workspace_removals["pid"][1] is None

    def test_the_mark_itself_ignores_dry_runs_and_evicts_oldest(self):
        from ghost_agent.core.workspace_cleanup import _mark_removal
        class _Store: pass
        st = _Store()
        _mark_removal(st, "PidA", ["x.md"], dry_run=True)
        assert not getattr(st, "recent_workspace_removals", None)
        _mark_removal(st, "PidA", ["x.md"])
        marks = st.recent_workspace_removals
        assert "pida" in marks and marks["pida"][1] == ["x.md"]
        for i in range(20):                      # cap at 16, oldest out
            _mark_removal(st, f"p{i}", None)
        assert len(st.recent_workspace_removals) <= 16


class TestPairAltAndSoftExactness:
    def test_a_pair_writes_tool_spelling_rescues_a_deeper_claim(self):
        # ⚠ The claim-deeper suffix arm was deleted as "an exact substitute
        # for the fallback root" — a justification an adversarial lens
        # refuted: the fallback covers the OPPOSITE direction, so a
        # pair-write whose model spelling ran deeper than the file's real
        # location read a PRESENT file as missing. The tool's own rel_str
        # is the statement of where the file landed; resolution consults it.
        from ghost_agent.core.agent import _fs_wrote_pair_alts
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        (proj / "app.py").write_text("real content")
        tools = [{"name": "file_system", "content":
                  "SUCCESS: Wrote 12 chars to 'projects/pid/src/app.py'. "
                  "Script-side path (from sandbox cwd): 'app.py'."}]
        assert GhostAgent._verify_file_artifacts(
            ["projects/pid/src/app.py"], str(proj),
            extra_roots=[str(root)],
            alt_spellings=_fs_wrote_pair_alts(tools)) is None

    def test_a_genuinely_missing_pair_write_still_refutes(self):
        from ghost_agent.core.agent import _fs_wrote_pair_alts
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        tools = [{"name": "file_system", "content":
                  "SUCCESS: Wrote 12 chars to 'projects/pid/src/app.py'. "
                  "Script-side path (from sandbox cwd): 'app.py'."}]
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/src/app.py"], str(proj),
            extra_roots=[str(root)],
            alt_spellings=_fs_wrote_pair_alts(tools))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_the_soft_arm_never_resolves_by_basename(self):
        # ⚠ Soft carries prose claims, whose spellings are sloppy by nature;
        # a bare `config.json` claim resolved via rglob to an unrelated
        # EMPTY `vendor/config.json` and refuted a turn on a file nobody
        # meant. Emptiness testimony must be about the file at the claimed
        # path, or nothing.
        d = Path(tempfile.mkdtemp())
        (d / "vendor").mkdir()
        (d / "vendor" / "config.json").write_text("")
        assert GhostAgent._verify_file_artifacts(
            [], str(d), soft=["config.json"]) is None

    def test_the_soft_arm_still_catches_an_exact_empty(self):
        d = Path(tempfile.mkdtemp())
        (d / "report.md").write_text("")
        r = GhostAgent._verify_file_artifacts([], str(d), soft=["report.md"])
        assert r is not None and "report.md" in "; ".join(r.issues)


class TestCheckabilityIsReported:
    """⚠ `None` means "no override", which covers BOTH "every path checked
    out" and "nothing could be opened". The caller must be able to tell them
    apart, or an unmounted workspace reads as a positive result about files
    the check never read."""

    def test_a_missing_workspace_is_reported_as_not_checkable(self):
        rep = {}
        assert GhostAgent._verify_file_artifacts(
            ["x.md"], "/nonexistent/zzz", report=rep) is None
        assert rep["checkable"] is False

    def test_a_real_pass_reports_what_it_actually_opened(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.md").write_text("content")
        rep = {}
        assert GhostAgent._verify_file_artifacts(
            ["a.md"], str(d), report=rep) is None
        assert rep["checkable"] is True and rep["resolved"] == 1

    def test_an_unreadable_path_is_reported_not_silently_passed(self):
        # pathlib propagates EACCES (it only swallows ENOENT/ENOTDIR/
        # EBADF/ELOOP), so an unreadable parent raises rather than reading
        # as absent. That must never become a quiet clean.
        import os
        d = Path(tempfile.mkdtemp())
        secret = d / "secret"
        secret.mkdir()
        (secret / "target.txt").write_text("hi")
        os.chmod(secret, 0o000)
        try:
            rep = {}
            r = GhostAgent._verify_file_artifacts(
                ["secret/target.txt"], str(d), report=rep)
            assert r is None                       # never a refute
            assert rep["resolved"] == 0
            assert rep["unresolvable"], "an error must be reported, not eaten"
        finally:
            os.chmod(secret, 0o700)


class TestProducerParserParity:
    """⚠ TRIPWIRE. Each confirmation shape has its OWN anchored pattern, so
    rewording a message in tools/file_system.py makes the parser go silently
    blind to that op while the ground-truth check keeps reporting "clean".

    ⚠ THE FIRST VERSION OF THIS CLASS DID NOT TRIP. It compared the static
    PREFIX of each `SUCCESS:` literal — and that prefix stops at the first
    `{`, while every parser pattern depends on text AFTER that point. So
    rewording `Wrote {n} chars to` into `Wrote {n} bytes into`, which blinds
    the ledger to EVERY write and takes both ground-truth overrides down with
    it, left this class green along with 761 other tests. A prefix set also
    cannot see a NEW emitter whose first word collides with an existing one.

    It compares BEHAVIOUR now: render each of the producer's own f-strings
    with placeholder values and require the parser to match it AND capture
    the right slot. A reword anywhere in the message fails, and a new emitter
    with no matching pattern fails on sight.
    """

    #: Which `{...}` slots are counts rather than paths, when rendering.
    NUMERIC_HINTS = ("len(", "count", "occurrences", "replaced", "ratio",
                     "start_line", "end_line", "success_count", "lines")

    #: One entry per emitter SITE, as (rendered-message-prefix, path slots).
    #  ⚠ PER SITE, not per distinct message. `Downloaded` is emitted TWICE
    #  (the curl_cffi route and the httpx redirect-hop route) with identical
    #  text; a renderer that deduped by text could not tell one site from
    #  two, so rewording ONE of them left every instrument green while every
    #  download through that route went invisible to both overrides.
    #  ⚠ The path-slot count is pinned too. The parser's capture is checked
    #  against `paths[0]`/`paths[1]` only, so a path slot added AFTER the
    #  pattern's terminator — `… Index sidecar written to '{f}.idx'.` — is
    #  invisible to the capture assertion. A changed count fails here
    #  instead, which is the honest tool for it: the renderer's numeric-vs-
    #  path heuristic is a guess about someone else's source, and a tripwire
    #  is what a guess deserves.
    EXPECTED_SITES = 13
    EXPECTED_SLOT_COUNTS = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4]

    def _rendered_messages(self):
        """Every `SUCCESS:` message tools/file_system.py can emit, rendered.

        -> list of (message, [path placeholders, in slot order]), ONE ENTRY
        PER EMITTER SITE.
        """
        import ast
        src_path = (Path(__file__).resolve().parents[1]
                    / "src" / "ghost_agent" / "tools" / "file_system.py")
        src = src_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        # ast.walk yields a JoinedStr AND each of its inner Constants; those
        # inner pieces are message fragments, not messages. Docstrings that
        # happen to start with "SUCCESS:" are documentation, not emitters —
        # and letting one in used to crash this class with an opaque
        # IndexError rather than a sentence.
        inner, docstrings = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                for v in node.values:
                    inner.add(id(v))
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                docstrings.add(id(node.value))

        def render(node):
            parts, paths = [], []
            pieces = node.values if isinstance(node, ast.JoinedStr) else [node]
            for pc in pieces:
                if isinstance(pc, ast.Constant):
                    parts.append(str(pc.value))
                    continue
                if not isinstance(pc, ast.FormattedValue):
                    return None
                expr = ast.get_source_segment(src, pc.value)
                if expr is None:
                    raise AssertionError(
                        "cannot recover the source of an interpolated slot in "
                        "tools/file_system.py — this renderer would guess, and "
                        "a guess here silently blesses a wrong-slot parser")
                spec = (ast.get_source_segment(src, pc.format_spec)
                        if pc.format_spec else "") or ""
                if "%" in spec:
                    parts.append("95%")
                elif any(k in expr for k in self.NUMERIC_HINTS):
                    parts.append("7")
                else:
                    ph = "p%d.qq" % len(paths)
                    paths.append(ph)
                    parts.append(ph)
            return "".join(parts), paths

        out = []
        for node in ast.walk(tree):
            if id(node) in inner or id(node) in docstrings:
                continue
            first = None
            if (isinstance(node, ast.JoinedStr) and node.values
                    and isinstance(node.values[0], ast.Constant)):
                first = str(node.values[0].value)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                first = node.value
            if not (first and first.startswith("SUCCESS:")):
                continue
            r = render(node)
            if r:
                out.append(r)
        return out

    def test_the_emitter_sites_have_not_drifted(self):
        messages = self._rendered_messages()
        assert len(messages) == self.EXPECTED_SITES, (
            f"tools/file_system.py has {len(messages)} SUCCESS emitter sites, "
            f"expected {self.EXPECTED_SITES}. One was added, removed, or moved "
            f"out of this walker's view — check that _fs_path_ledger still "
            f"parses every one, then update EXPECTED_SITES.")
        assert sorted(len(p) for _, p in messages) == self.EXPECTED_SLOT_COUNTS, (
            "the number of interpolated PATH slots changed. The parser only "
            "validates the first (or second) slot, so a path added after the "
            "pattern's terminator would be produced by the tool and never "
            "checked. Confirm the ledger still sees every path, then update "
            "EXPECTED_SLOT_COUNTS.")

    def test_no_sibling_module_supplies_the_file_tools_messages(self):
        # ⚠ Both instruments scope on ONE FILENAME. Factor the templates into
        # `tools/fs_messages.py` and have file_system.py return them, and the
        # AST walker sees nothing while the tool keeps emitting — the ledger
        # goes blind with every test green. Nothing here can walk "whatever
        # the file tool returns", so guard the structural precondition
        # instead: the file tool's messages must originate in its own module.
        import ast
        tools = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "tools"
        src = (tools / "file_system.py").read_text(encoding="utf-8")
        siblings = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                siblings.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    siblings.add(a.name.split(".")[-1])
        offenders = []
        for name in sorted(siblings):
            sib = tools / f"{name}.py"
            if sib.exists() and "SUCCESS:" in sib.read_text(encoding="utf-8"):
                offenders.append(name)
        assert not offenders, (
            f"tools/file_system.py imports sibling module(s) that contain "
            f"'SUCCESS:' literals: {offenders}. If the file tool now returns "
            f"messages built there, _rendered_messages cannot see them and "
            f"the ground-truth check will go silently blind — extend the "
            f"walker to those modules.")

    def test_no_pinned_prefix_is_a_prefix_of_another(self):
        # ⚠ The cross-links below are PREFIX comparisons. If one pin were a
        # prefix of another ("Wrote" and "Wrote index for"), a message for
        # the longer pin would satisfy the shorter one, and the emitter the
        # shorter pin names could vanish with everything green.
        for a in self.PINNED:
            for b in self.PINNED:
                assert a is b or not b.startswith(a), (
                    f"PINNED entry {a!r} is a prefix of {b!r}; the two-way "
                    f"cross-link can no longer tell them apart")

    def test_the_producers_own_messages_all_reach_a_parser_pattern(self):
        from ghost_agent.core.agent import (
            _FS_RETIRE_RE, _FS_MOVED_RE, _FS_DEST_RE, _FS_PRODUCE_RES)
        messages = self._rendered_messages()
        assert len(messages) >= 12, (
            "only rendered %d SUCCESS messages from tools/file_system.py — "
            "the extraction broke, so this tripwire is checking nothing"
            % len(messages))
        unmatched = []
        for msg, paths in messages:
            head = msg.split("\n", 1)[0][:4000]   # same cap as the ledger
            m = _FS_RETIRE_RE.match(head)
            if m:
                assert paths, f"a Deleted message rendered no path slot: {head}"
                assert m.group(1) == paths[0], head
                continue
            m = _FS_MOVED_RE.match(head)
            if m:
                assert len(paths) >= 2, (
                    f"a Renamed/Moved message rendered {len(paths)} path "
                    f"slot(s): {head}")
                assert m.groups() == (paths[0], paths[1]), head
                continue
            m = _FS_DEST_RE.match(head)
            if m:
                # The destination is the SECOND path slot, never the source.
                # (For `Downloaded` the first slot is a URL, so "second slot"
                # is a numbering fact about these two messages, not a law —
                # say so out loud rather than raising IndexError at someone.)
                assert len(paths) >= 2, (
                    f"a Downloaded/Copied message rendered {len(paths)} path "
                    f"slot(s); the dest-vs-source assumption no longer "
                    f"holds: {head}")
                assert m.group(1) == paths[1], head
                continue
            for rx in _FS_PRODUCE_RES:
                m = rx.match(head)
                if m:
                    if not paths:
                        unmatched.append(f"{head}  [no path slot rendered]")
                    elif m.group(1) != paths[0]:
                        unmatched.append(f"{head}  [captured {m.group(1)!r}, "
                                         f"expected {paths[0]!r}]")
                    break
            else:
                unmatched.append(head)
        # ⚠ H2: the two instruments fail in COMPLEMENTARY directions —
        # the AST walker cannot see comments, and PINNED's regex scans raw
        # text INCLUDING comments. So moving the `Wrote` template into
        # another module and leaving a comment behind ("Emits: SUCCESS:
        # Wrote {n} chars to '{f}'.") keeps PINNED equal while the walker
        # loses the emitter that feeds every write, and one unrelated new
        # emitter restores the count. Requiring each PINNED prefix to HEAD a
        # rendered message ties them together and the trick fails.
        for _pref in self.PINNED:
            assert any(_m.startswith("SUCCESS: " + _pref)
                       for _m, _ in messages), (
                f"PINNED lists a {_pref!r} message that no longer renders "
                f"from tools/file_system.py — the emitter moved out of this "
                f"walker's view (another module? a comment left behind?) "
                f"while the prefix scan still sees its text")
        for _m, _ in messages:
            assert any(_m.startswith("SUCCESS: " + _pref)
                       for _pref in self.PINNED), (
                f"a SUCCESS message renders that PINNED does not list — a "
                f"new emitter: {_m[:90]}")
        assert not unmatched, (
            "tools/file_system.py emits SUCCESS message(s) that "
            "_fs_path_ledger cannot parse — the verifier's ground-truth "
            "check would go silently blind to these ops:\n  "
            + "\n  ".join(unmatched))

    # Kept as a coarse second signal: a REMOVED emitter, or one renamed at
    # the very front, shows up here. It cannot see a reword past the first
    # `{` — that is what the behavioural test above is for.
    PINNED = [
        "Anchor match — replaced the block spanning lines",
        "Applied",
        "Copied",
        "Deleted",
        "Downloaded",
        "Exact match found and replaced in",
        "Flexible match found and replaced in",
        "Fuzzy match (",
        "Renamed/Moved",
        "Streaming replace applied to",
        "Wrote",
        "auto-promoted operation=",
    ]

    def test_the_set_of_success_message_shapes_has_not_drifted(self):
        import re as _re
        src = (Path(__file__).resolve().parents[1]
               / "src" / "ghost_agent" / "tools" / "file_system.py")
        text = src.read_text(encoding="utf-8")
        found = sorted({m.group(1).strip() for m in
                        _re.finditer(r"""["']SUCCESS: ([^"'{]*)""", text)})
        assert found == sorted(self.PINNED), (
            "tools/file_system.py's SUCCESS messages changed. The verifier's "
            "_fs_path_ledger parses these by shape — update its patterns and "
            "this list together, or the ground-truth check goes blind.")

    @pytest.mark.parametrize("message,expected", [
        ("SUCCESS: Wrote 391 chars to 'a.py'. Script-side path "
         "(from sandbox cwd): 'a.py'.", ["a.py"]),
        ("SUCCESS: auto-promoted operation='replace' to 'write' for 'b.py' "
         "because your 'content' was a complete Python module and "
         "'replace_with' was missing.", ["b.py"]),
        ("SUCCESS: Streaming replace applied to 'c.py' (3 line(s) modified).",
         ["c.py"]),
        ("SUCCESS: Applied 2 SEARCH/REPLACE blocks to 'd.py'.", ["d.py"]),
        ("SUCCESS: Exact match found and replaced in 'e.py'.", ["e.py"]),
        ("SUCCESS: Exact match found and replaced in 'e2.py'. WARNING: "
         "Replaced 3 identical occurrences.", ["e2.py"]),
        ("SUCCESS: Flexible match found and replaced in 'f.py'.", ["f.py"]),
        ("SUCCESS: Fuzzy match (95% similar) found and replaced in 'g.py'. "
         "Your `old_text` did not byte-match, but a single near-identical "
         "block was unambiguous.", ["g.py"]),
        ("SUCCESS: Anchor match — replaced the block spanning lines 3–9 in "
         "'h.py' (matched on its unique first+last lines; the middle "
         "differed from your old_text).", ["h.py"]),
        ("SUCCESS: Downloaded 'https://x.example/i.zip' to 'i.zip'.", ["i.zip"]),
        ("SUCCESS: Copied 'j0.md' to 'j.md'.", ["j.md"]),
        ("SUCCESS: Renamed/Moved 'k0.md' to 'k.md'.", ["k.md"]),
        ("SUCCESS: Deleted 'l.md'.", []),
    ])
    def test_every_shape_parses_to_the_path_it_left_behind(self, message,
                                                           expected):
        from ghost_agent.core.agent import _files_mutated_this_turn
        assert _files_mutated_this_turn(
            [{"role": "tool", "name": "file_system",
              "content": message}]) == expected


class TestCapturedProjectId:
    """The ONE sanctioned scope capture — four sites grew four copies of
    this heal in a single round before it was consolidated."""

    def _agent(self):
        from unittest.mock import MagicMock
        a = GhostAgent.__new__(GhostAgent)
        a.context = MagicMock()
        return a

    def test_a_live_global_wins(self):
        a = self._agent()
        a.context.current_project_id = "live-pid"
        assert a._captured_project_id() == "live-pid"

    def test_a_stomped_global_heals_from_the_conversation_binding(self, monkeypatch):
        a = self._agent()
        a.context.current_project_id = None
        monkeypatch.setattr(
            "ghost_agent.tools.file_system._conversation_bound_project",
            lambda ctx: "healed-pid")
        assert a._captured_project_id() == "healed-pid"

    def test_a_closed_project_heals_from_the_close_time_record(self, monkeypatch):
        # ⚠ a same-turn project close clears the global AND the conversation
        # binding; the close path records (conversation_key, pid) at the
        # moment it clears, so the heal names the project actually closed.
        # (The first fallback used the shared `request_start_project_id`
        # slot — a concurrent request overwrote it, and an A→B switch healed
        # to A while the writes sat under B.)
        a = self._agent()
        a.context.current_project_id = None
        a.context.conversation_key = "conv-1"
        a.context.last_closed_project = ("conv-1", "closed-pid")
        monkeypatch.setattr(
            "ghost_agent.tools.file_system._conversation_bound_project",
            lambda ctx: "")
        assert a._captured_project_id() == "closed-pid"

    def test_another_conversations_close_heals_nothing(self, monkeypatch):
        a = self._agent()
        a.context.current_project_id = None
        a.context.conversation_key = "conv-1"
        a.context.last_closed_project = ("conv-OTHER", "their-pid")
        monkeypatch.setattr(
            "ghost_agent.tools.file_system._conversation_bound_project",
            lambda ctx: "")
        assert a._captured_project_id() is None

    def test_no_binding_anywhere_means_unscoped_None(self, monkeypatch):
        a = self._agent()
        a.context.current_project_id = None
        a.context.conversation_key = "conv-1"
        a.context.last_closed_project = None
        monkeypatch.setattr(
            "ghost_agent.tools.file_system._conversation_bound_project",
            lambda ctx: "")
        assert a._captured_project_id() is None


class TestDrainScopeAndAltSemantics:
    """Round-10 pins: the streamed drain's scope pid is a PURE function so it
    is testable at all — the drain runs post-semaphore where nothing live is
    race-free — and the pair-alt loop's edge semantics."""

    def test_the_snapshot_wins_when_the_tag_matches(self):
        snap = ("req-A", {"pid": "proj-A"})
        assert GhostAgent._drain_scope_pid(snap, "req-A", "captured-X") == "proj-A"

    def test_a_mismatched_tag_falls_back_to_the_CAPTURED_value(self):
        # ⚠ the mismatch branch used to fall back to a live
        # `current_project_id` read — at drain time that is whichever
        # request is running, i.e. the §4DG cross-project race re-opened on
        # the production-common path.
        snap = ("req-B", {"pid": "proj-B"})
        assert GhostAgent._drain_scope_pid(snap, "req-A", "captured-X") == "captured-X"

    def test_a_missing_snapshot_falls_back_to_the_captured_value(self):
        assert GhostAgent._drain_scope_pid(None, "req-A", "captured-X") == "captured-X"
        assert GhostAgent._drain_scope_pid("garbage", "req-A", None) is None

    def test_an_alt_directory_cannot_satisfy_a_prose_claim(self):
        # ⚠ the alt path re-derived dir_ok from the ALT spelling, so a
        # prose-claimed FILE was satisfied by a DIRECTORY sitting at the
        # pair's rel_str spelling — masking a missing-deliverable refute.
        from ghost_agent.core.agent import _fs_wrote_pair_alts
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        (proj / "app").mkdir()                    # a DIRECTORY at rel_str
        tools = [{"name": "file_system", "content":
                  "SUCCESS: Wrote 12 chars to 'projects/pid/deep/app'. "
                  "Script-side path (from sandbox cwd): 'app'."}]
        # the extension-less spelling is ledger-only; force the prose rule
        r = GhostAgent._verify_file_artifacts(
            ["projects/pid/deep/app"], str(proj), extra_roots=[str(root)],
            file_claims={"projects/pid/deep/app"},
            alt_spellings=_fs_wrote_pair_alts(tools))
        assert r is not None and r.verdict == VerifyVerdict.REFUTED

    def test_a_could_not_check_alt_does_not_stop_the_next_alt(self):
        import os
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        locked = proj / "locked"
        locked.mkdir()
        (proj / "app.py").write_text("real")
        os.chmod(locked, 0o000)
        try:
            rep = {}
            r = GhostAgent._verify_file_artifacts(
                ["projects/pid/src/app.py"], str(proj),
                extra_roots=[str(root)],
                alt_spellings={"projects/pid/src/app.py":
                               ["locked/app.py", "app.py"]},
                report=rep)
            assert r is None
            # ⚠ None alone cannot distinguish "the second alt resolved" from
            # "gave up as could-not-check" — both skip the refute. The
            # report can: a real hit counts as resolved.
            assert rep["resolved"] == 1
        finally:
            os.chmod(locked, 0o700)

    def test_a_retired_pair_write_recreated_empty_is_caught_via_its_alt(self):
        # ⚠ the soft arm never consulted pair alts, so a retired pair-write
        # re-created EMPTY at its rel_str spelling escaped the one check the
        # soft arm exists to keep.
        from ghost_agent.core.agent import _fs_wrote_pair_alts
        root = Path(tempfile.mkdtemp())
        proj = root / "projects" / "pid"
        proj.mkdir(parents=True)
        (proj / "probe.py").write_text("")        # back, and empty
        tools = [{"name": "file_system", "content":
                  "SUCCESS: Wrote 9 chars to 'projects/pid/deep/probe.py'. "
                  "Script-side path (from sandbox cwd): 'probe.py'."}]
        r = GhostAgent._verify_file_artifacts(
            [], str(proj), soft=["projects/pid/deep/probe.py"],
            extra_roots=[str(root)],
            alt_spellings=_fs_wrote_pair_alts(tools))
        assert r is not None and "probe.py" in "; ".join(r.issues)
