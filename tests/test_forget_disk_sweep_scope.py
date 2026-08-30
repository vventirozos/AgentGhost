"""`forget` may not delete a file the caller did not name.

The disk sweep ranked its candidates exact name -> stem -> SUBSTRING, and
deleted every member of the first non-empty tier. The substring tier was
added to catch near-misses; what it actually does is delete every file whose
name merely CONTAINS the target. Measured on a real sandbox,
`forget('atlas')` unlinked:

    atlas_migration_plan.py
    notes_about_atlas.md
    sub/deep_atlas_notes.txt

None of them was named by the caller, the deletion is irreversible, there is
no dry-run, and the action is model-reachable in one call. The pressure got
worse when `target` gained a schema description in the vocabulary that feeds
this branch hardest — "a topic, an entity, a person's name" — because a bare
word matches far more filenames than a filename does.

Forgetting a TOPIC does not mean deleting every file whose name shares a
token with it. The vector, profile and graph sweeps remove the knowledge
either way; the files are a separate question, and the caller is the one who
should answer it. So substring hits are surfaced as candidates and left
alone. Naming one explicitly makes it an exact match, and it is deleted.
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.memory import tool_unified_forget


def _memsys():
    mem = MagicMock()
    mem.get_library = MagicMock(return_value=[])
    mem.collection.get = MagicMock(
        return_value={"ids": [], "metadatas": [], "documents": []})
    return mem


@pytest.fixture
def sandbox(tmp_path):
    """⚠ NO exact/stem match for 'atlas'.

    The first version of this fixture also created `atlas.md`. That is a
    STEM hit, so `exact_hits or stem_hits or substr_hits` short-circuited
    before ever reaching the substring tier — and the test below therefore
    PASSED ON THE PRE-FIX CODE, which kept those same three files for the
    same reason. It described a scenario its own fixture could not produce.
    Only a sandbox with no better match exercises the tier being removed.
    """
    (tmp_path / "atlas_migration_plan.py").write_text("x")
    (tmp_path / "notes_about_atlas.md").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep_atlas_notes.txt").write_text("x")
    return tmp_path


def _survivors(root: Path):
    return sorted(p.name for p in root.rglob("*") if p.is_file())


@pytest.mark.asyncio
async def test_a_topic_does_not_delete_files_that_merely_mention_it(sandbox):
    """The discriminating case: nothing matches better, so the pre-fix code
    reached the substring tier and unlinked all three."""
    report = await tool_unified_forget(
        "atlas", sandbox_dir=sandbox, memory_system=_memsys())

    survivors = _survivors(sandbox)
    for kept in ("atlas_migration_plan.py", "notes_about_atlas.md",
                 "deep_atlas_notes.txt"):
        assert kept in survivors, (
            f"{kept} was deleted for a partial name match — the caller never "
            f"named it, and there is no way to undo this"
        )
    assert "Deleted" not in report, (
        "nothing here is an exact or stem match; nothing should have gone"
    )


@pytest.mark.asyncio
async def test_the_named_file_still_goes_while_the_others_stay(sandbox):
    """...and an exact/stem match alongside them is still deleted. Both
    halves in one call, because the first version of this file only checked
    the half that was true before the fix too."""
    (sandbox / "atlas.md").write_text("x")

    report = await tool_unified_forget(
        "atlas", sandbox_dir=sandbox, memory_system=_memsys())

    survivors = _survivors(sandbox)
    assert "atlas.md" not in survivors, (
        "the stem match IS the file the caller named; it must still go"
    )
    assert survivors == ["atlas_migration_plan.py", "deep_atlas_notes.txt",
                         "notes_about_atlas.md"]
    assert "Disk: Deleted 'atlas.md'" in report
    # The report must fire even though something WAS deleted — it only
    # appeared when nothing was, which is exactly when it matters least.
    assert "Disk: kept 3 file(s)" in report


@pytest.mark.asyncio
async def test_partial_matches_are_reported_so_the_caller_can_act(tmp_path):
    """With no exact or stem match there is nothing to delete — but staying
    silent would leave the caller thinking the sweep found nothing."""
    (tmp_path / "atlas_migration_plan.py").write_text("x")
    (tmp_path / "notes_about_atlas.md").write_text("x")

    report = await tool_unified_forget(
        "atlas", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == ["atlas_migration_plan.py",
                                    "notes_about_atlas.md"]
    assert "Disk: kept 2 file(s)" in report
    assert "atlas_migration_plan.py" in report
    # The old form of this assertion sliced the kept LINE and looked for
    # "Deleted" inside it, which no report could ever contain — it could not
    # fail. Check the whole report.
    assert "Deleted" not in report
    # ...and it says what to do next.
    assert "exact name" in report


@pytest.mark.asyncio
async def test_naming_the_file_exactly_still_deletes_it(tmp_path):
    """The escape hatch has to work, or the report above is a dead end."""
    (tmp_path / "atlas_migration_plan.py").write_text("x")
    (tmp_path / "notes_about_atlas.md").write_text("x")

    report = await tool_unified_forget(
        "atlas_migration_plan.py", sandbox_dir=tmp_path,
        memory_system=_memsys())

    assert _survivors(tmp_path) == ["notes_about_atlas.md"]
    assert "Disk: Deleted 'atlas_migration_plan.py'" in report


@pytest.mark.asyncio
async def test_more_candidates_than_the_report_shows_are_counted(tmp_path):
    """Naming 10 of 15 while telling the caller to name one exactly is a
    dead end for the other five."""
    for i in range(15):
        (tmp_path / f"atlas_{i}.md").write_text("x")

    report = await tool_unified_forget(
        "atlas", sandbox_dir=tmp_path, memory_system=_memsys())

    assert "kept 15 file(s)" in report
    assert "+5 more" in report
    assert len(_survivors(tmp_path)) == 15


@pytest.mark.asyncio
async def test_the_stem_tier_is_unchanged(tmp_path):
    """`forget('notes')` still removes `notes.md` and `notes.txt` — same
    stem, different extension, unambiguously the thing named."""
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "notes_archive.md").write_text("x")

    report = await tool_unified_forget(
        "notes", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == ["notes_archive.md"]
    # ...and the one it kept is named, rather than passed over in silence
    # because something else was deleted.
    assert "notes_archive.md" in report


# --------------------------------------------------------- reset_all's scope

def _reset_memsys(rows, fail_batches=()):
    """A store of `rows` [(id, type), …]. `fail_batches` names the batch
    indices whose delete raises — the previous helper could only express
    TOTAL failure (`fail_from=0`), which is exactly why the partial-failure
    case went unnoticed: the library index was emptied while 500 rows
    survived."""
    mem = MagicMock()
    del mem.library_file
    seen = {"scans": [], "threads": []}

    def _get(**kw):
        seen["scans"].append(kw)
        seen["threads"].append(threading.current_thread().name)
        return {"ids": [r[0] for r in rows],
                "metadatas": [{"type": r[1]} for r in rows]}

    seen["batch"] = 0

    def _delete(ids=None):
        seen["threads"].append(threading.current_thread().name)
        idx = seen["batch"]
        seen["batch"] += 1
        if idx in fail_batches:
            raise RuntimeError("delete refused")

    mem.collection.get = _get
    mem.collection.delete = _delete
    return mem, seen


@pytest.mark.asyncio
async def test_reset_all_never_touches_the_store_from_the_event_loop():
    """`collection.get()` with no `include` pulled every document BODY back
    (live: ~8k rows, 7k of them manual chunks) to use nothing but the ids,
    and it and every delete batch ran synchronously on the event loop —
    while the CHEAP graph wipe was already offloaded. Every concurrent
    request, stream and heartbeat stalled for the duration.

    Asserted by THREAD, not by the name of the offloaded function: the
    previous version matched `fn.__name__`, so renaming a local broke it
    with no behaviour change."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem, seen = _reset_memsys([("1", "fact"), ("2", "fact")])
    main = threading.current_thread().name

    out = await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert "Wiped clean" in out
    assert seen["threads"], "the store was never touched at all"
    assert main not in seen["threads"], (
        f"the store was accessed from the event loop's thread ({main}); "
        f"every concurrent request stalls behind it"
    )


@pytest.mark.asyncio
async def test_reset_all_enumerates_once():
    """Two scans meant the orphan count came from a different snapshot than
    the delete: rows landing between them produced a note about documents
    that were never removed. One scan returns ids AND metadatas."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem, seen = _reset_memsys([("1", "document")])
    await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert len(seen["scans"]) == 1, f"{len(seen['scans'])} enumerations"
    assert seen["scans"][0].get("include") == ["metadatas"], (
        "the scan must ask for metadatas (ids come back regardless) and "
        "NOT for documents — that is what pulls the whole store into memory"
    )


@pytest.mark.asyncio
async def test_reset_all_says_what_it_orphans():
    """It deletes the `document` / `episode` / `skill` rows
    `_FORGET_PROTECTED_TYPES` protects, because each has a record in ANOTHER
    store this does not touch. `forget` refuses to create that asymmetry;
    `reset_all` creates it by design, so it has to say so."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem, _ = _reset_memsys([("1", "document"), ("2", "document"),
                            ("3", "episode"), ("4", "fact")])
    out = await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert "2 document" in out and "1 episode" in out
    assert "no searchable twin" in out
    # Split on the LAST NOTE: `reset_all` can emit two (an incomplete-
    # metadata warning is prepended to the orphan note), in which case
    # index 1 is the wrong half.
    assert "fact" not in out.rsplit("NOTE:", 1)[1], (
        "unprotected types are not orphans"
    )


@pytest.mark.asyncio
async def test_reset_all_with_nothing_protected_reports_plainly():
    from ghost_agent.tools.memory import tool_knowledge_base

    mem, _ = _reset_memsys([("1", "fact")])
    out = await tool_knowledge_base(action="reset_all", memory_system=mem)
    assert out == "Success: Wiped clean (1 entries removed)."


@pytest.mark.asyncio
async def test_a_failed_wipe_does_not_claim_to_have_removed_anything():
    """The orphan note was emitted from a pre-scan regardless of outcome:
    with every batch failing it reported "this removed the vector rows for N
    document…" having removed nothing. And the library index was reset even
    then, leaving the catalogue empty while the rows survived — while the
    message said they were "left in place"."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem, _ = _reset_memsys([("1", "document"), ("2", "episode")], fail_batches=(0,))
    lib = MagicMock()
    mem.library_file = lib

    out = await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert "Wiped 0 entries" in out
    assert "left in place" in out
    assert "NOTE:" not in out, (
        "nothing was removed, so nothing was orphaned — the note describes "
        "a wipe that did not happen"
    )
    lib.with_suffix.assert_not_called(), "the library index was reset anyway"


@pytest.mark.asyncio
async def test_a_shadowed_stem_hit_reaches_the_kept_list(tmp_path):
    """The ONLY shape that still populates `stem_hits` without them being
    deleted: an extensionless target that matches a file exactly, plus a
    same-stem sibling.

    The test below was written to kill "drop stem_hits from the kept list",
    and a later product edit (`target_is_filename`, which routes stem
    matches of an extensioned target into `substr_hits`) silently emptied
    `stem_hits` for its fixture — so the mutation it exists for survived
    again. Same class as the fixture bug in this file's own header,
    reintroduced from the other side.
    """
    (tmp_path / "notes").write_text("x")          # exact hit
    (tmp_path / "notes.md").write_text("x")       # stem hit, shadowed
    (tmp_path / "notes.txt").write_text("x")      # stem hit, shadowed

    report = await tool_unified_forget(
        "notes", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == ["notes.md", "notes.txt"]
    assert "Disk: Deleted 'notes'" in report
    assert "kept 2 file(s)" in report
    assert "notes.md" in report and "notes.txt" in report


@pytest.mark.asyncio
async def test_a_shadowed_STEM_match_is_reported_too(tmp_path):
    """The `or` chain shadows the stem tier as well as the substring one:
    with an exact hit present, `forget('notes.md')` deleted `notes.md` and
    kept `notes.txt` in silence. Dropping stem hits from the kept-list
    survived every other test in this file, because they all exercise the
    substring tier."""
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    report = await tool_unified_forget(
        "notes.md", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == ["notes.txt"]
    assert "Disk: Deleted 'notes.md'" in report
    assert "kept 1 file(s)" in report and "notes.txt" in report


# ------------------------------------- a path means a path, a name means one

@pytest.mark.asyncio
async def test_a_path_qualified_target_deletes_only_that_path(tmp_path):
    """The matcher read only the BASENAME, so naming one file removed every
    file in the tree sharing that name — and the kept-report prints
    candidates as sandbox-relative paths and tells the caller to re-issue
    with one, so its own instruction was the trigger. On the live sandbox
    `forget('projects/<id>/index.html')` removed five index.html files
    across five projects."""
    for rel in ("projects/alpha/report.md", "projects/beta/report.md",
                "archive/2019/report.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")

    report = await tool_unified_forget(
        "projects/alpha/report.md", sandbox_dir=tmp_path,
        memory_system=_memsys())

    assert _survivors(tmp_path) == ["report.md", "report.md"]
    assert not (tmp_path / "projects/alpha/report.md").exists()
    assert (tmp_path / "projects/beta/report.md").exists()
    assert (tmp_path / "archive/2019/report.md").exists()
    assert "Deleted 'projects/alpha/report.md'" in report


@pytest.mark.asyncio
async def test_a_path_qualified_target_offers_no_weaker_match(tmp_path):
    """A caller who gave a full path was precise; a near-miss elsewhere is
    not a candidate for it."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "other_report.md").write_text("x")

    report = await tool_unified_forget(
        "sub/report.md", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == ["other_report.md"]
    assert "Disk:" not in report


@pytest.mark.asyncio
async def test_naming_a_file_that_is_absent_does_not_delete_its_siblings(tmp_path):
    """`forget('notes.md')` with notes.md ABSENT fell through to the stem
    tier and deleted notes.txt, notes.xlsx and notes.pdf — the very files
    the report calls partial matches and promises not to touch when
    notes.md happens to exist. One unrelated file's existence flipped three
    others between 'explicitly kept' and 'silently deleted'."""
    for n in ("notes.txt", "notes.xlsx", "notes.pdf"):
        (tmp_path / n).write_text("x")

    report = await tool_unified_forget(
        "notes.md", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == ["notes.pdf", "notes.txt", "notes.xlsx"]
    assert "kept 3 file(s)" in report
    assert "Deleted" not in report


@pytest.mark.asyncio
async def test_a_bare_stem_target_still_takes_the_whole_stem(tmp_path):
    """The stem tier is the point of `forget('notes')` — it is only
    suppressed when the target names a specific FILE."""
    for n in ("notes.md", "notes.txt"):
        (tmp_path / n).write_text("x")

    await tool_unified_forget(
        "notes", sandbox_dir=tmp_path, memory_system=_memsys())
    assert _survivors(tmp_path) == []


@pytest.mark.asyncio
async def test_symlinks_are_not_offered_as_candidates(tmp_path):
    """The unlink refuses symlinks, so listing one as something to "forget
    by its exact name" is a permanent dead end."""
    (tmp_path / "real_atlas.md").write_text("x")
    (tmp_path / "link_to_atlas.md").symlink_to(tmp_path / "real_atlas.md")

    report = await tool_unified_forget(
        "atlas", sandbox_dir=tmp_path, memory_system=_memsys())

    assert "link_to_atlas.md" not in report
    assert "real_atlas.md" in report
    assert (tmp_path / "link_to_atlas.md").exists()


# ------------------------------------------ the vector half, same discipline

@pytest.mark.asyncio
async def test_a_partial_document_name_is_kept_not_wiped():
    """The disk half printed "Nothing on disk is deleted for a partial name
    match" while this half substring-deleted the whole document. Against the
    live library — one entry, the ~7k-chunk PostgreSQL manual —
    `forget('pdf')`, `forget('sql')` and `forget('postgres')` each destroyed
    it: three characters, no candidate list, irreversible."""
    mem = _memsys()
    mem.get_library = MagicMock(return_value=["postgresql-19-A4.pdf"])

    for target in ("pdf", "sql", "postgres"):
        mem.delete_document_by_name.reset_mock()
        report = await tool_unified_forget(target, memory_system=mem)
        mem.delete_document_by_name.assert_not_called()
        assert "Vector: kept 1 ingested document(s)" in report
        assert "postgresql-19-A4.pdf" in report


@pytest.mark.asyncio
async def test_an_exact_document_name_or_stem_is_still_wiped():
    """The escape hatch, and the ordinary case."""
    mem = _memsys()
    mem.get_library = MagicMock(return_value=["postgresql-19-A4.pdf"])

    for target in ("postgresql-19-A4.pdf", "postgresql-19-A4"):
        mem.delete_document_by_name.reset_mock()
        report = await tool_unified_forget(target, memory_system=mem)
        mem.delete_document_by_name.assert_called_once_with(
            "postgresql-19-A4.pdf")
        assert "Vector: Wiped document" in report


@pytest.mark.asyncio
async def test_a_partly_failed_wipe_does_not_empty_the_catalogue():
    """The guard covered TOTAL failure only. With one batch of two failing,
    500 rows survived and `library_index.json` was still rewritten to `[]` —
    the exact disagreement the guard exists to prevent, while the message
    says the entries were "left in place". Ingest dedups on the library, so
    an un-listed surviving document can neither be queried nor re-ingested
    without duplicating it."""
    from ghost_agent.tools.memory import tool_knowledge_base

    rows = [(str(i), "document") for i in range(800)]
    mem, _ = _reset_memsys(rows, fail_batches=(0,))
    lib = MagicMock()
    mem.library_file = lib

    out = await tool_knowledge_base(action="reset_all", memory_system=mem)

    # UPPERCASE: `coerce`'s PARTIAL test is case-sensitive, so the lowercase
    # head was a clean OK — a partially-failed wipe reported as success.
    assert "PARTIAL: Wiped 300 entries" in out
    assert "1 batch(es) failed" in out
    lib.with_suffix.assert_not_called()
    # ...and the orphan count describes only what actually went.
    assert "300 document" in out


@pytest.mark.asyncio
async def test_reset_all_holds_the_vector_lock():
    """Every writer in vector.py and both forget sweeps take
    `_get_lock()`; these two did not. A concurrent ingest that started after
    the snapshot survived the wipe while the unlocked library reset erased
    its catalogue entry — the row lived, its index line did not, and the
    tool reported a clean "Wiped clean"."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem, _ = _reset_memsys([("1", "fact")])
    held = []

    class _Lock:
        def __enter__(self):
            held.append("in")
            return self

        def __exit__(self, *a):
            held.append("out")
            return False

    mem._get_lock = lambda: _Lock()
    await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert held.count("in") >= 2, (
        "the enumeration and the delete batches must both run under the "
        "vector lock"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot,expect", [
    ({"ids": ["1", "2"], "metadatas": [{"type": "document"}]},
     "fewer metadata rows"),
    ({"ids": ["1"], "metadatas": ["not-a-dict"]}, "Wiped clean"),
    ({"ids": ["1"], "metadatas": [None]}, "Wiped clean"),
    ({"ids": ["1"]}, "Wiped clean"),
])
async def test_an_odd_store_shape_neither_raises_nor_lies(snapshot, expect):
    """The shape is the client's, not ours. A non-dict metadata entry raised
    `AttributeError` straight out of the tool — nothing deleted, no error
    string returned — and a short metadatas list silently under-reported the
    orphans with no hint that the disclosure was incomplete."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem = MagicMock()
    del mem.library_file
    mem.collection.get = lambda **kw: snapshot
    mem.collection.delete = lambda ids=None: None

    out = await tool_knowledge_base(action="reset_all", memory_system=mem)
    assert expect in out


# ------------------------------------------- the mixed-turn preview budget

def test_an_argument_error_keeps_its_worked_call_through_the_summariser():
    """Two places truncate a failure preview: the turn loop, and
    `summarize_multi_op_outcomes`. Widening only the first was dead code —
    the summariser re-cut to a flat 140 and the model still read
    "…filename to erase. Worked " and nothing more. Reverting that second
    fix left every test in the repo green: no test anywhere passed the
    summariser a preview longer than ~55 chars, or an argument error at
    all."""
    import asyncio as _asyncio

    from ghost_agent.tools.memory import tool_knowledge_base
    from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes

    err = _asyncio.run(tool_knowledge_base(action="forget"))
    assert len(err) > 140, "fixture no longer exercises the truncation"

    out = summarize_multi_op_outcomes([
        {"tool": "recall", "ok": True, "preview": None},
        {"tool": "knowledge_base", "ok": False, "preview": err},
    ])
    assert err in out, (
        "the argument error was truncated by the summariser; its worked "
        "call sits at the END of the message and is what the model needs"
    )


def test_an_ordinary_failure_is_still_summarised_tersely():
    """The widening must not hand every failure 300 characters."""
    from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes

    long_plain = "Error: connection reset by peer. " + ("detail " * 60)
    out = summarize_multi_op_outcomes([
        {"tool": "recall", "ok": True, "preview": None},
        {"tool": "web_search", "ok": False, "preview": long_plain},
    ])
    body = out.split("FAILED: ")[1].split("\n")[0]
    assert len(body) < 200, f"an ordinary failure got the wide budget: {len(body)}"


# ------------------------------------- properties nothing was checking yet

@pytest.mark.asyncio
async def test_the_kept_report_prints_sandbox_relative_paths(tmp_path):
    """Nothing pinned that the report prints PATHS. Regressing it to bare
    basenames makes the report the trigger for the over-deletion it was
    written to prevent: a bare name matches every same-named file in the
    tree, while a path-qualified target is exact-only."""
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "deep" / "notes_about_atlas.md").write_text("x")

    report = await tool_unified_forget(
        "atlas", sandbox_dir=tmp_path, memory_system=_memsys())

    assert "'sub/deep/notes_about_atlas.md'" in report, (
        "the report must name a path the caller can re-issue verbatim"
    )


@pytest.mark.asyncio
async def test_a_path_qualified_target_is_not_a_suffix_match(tmp_path):
    """`==` not `endswith`. With a suffix match, `forget('alpha/report.md')`
    would also take `projects/alpha/report.md` — the same over-deletion
    through a narrower door."""
    (tmp_path / "projects" / "alpha").mkdir(parents=True)
    (tmp_path / "projects" / "alpha" / "report.md").write_text("x")

    await tool_unified_forget(
        "alpha/report.md", sandbox_dir=tmp_path, memory_system=_memsys())

    assert (tmp_path / "projects" / "alpha" / "report.md").exists()


@pytest.mark.asyncio
async def test_more_partial_documents_than_shown_are_counted():
    """The vector report was written from the disk one and inherited the
    defect the disk one had already been fixed for: 10 names printed, N
    counted, and the instruction is to re-issue with a name shown."""
    mem = _memsys()
    mem.get_library = MagicMock(
        return_value=[f"atlas_report_{i}.pdf" for i in range(15)])

    report = await tool_unified_forget("atlas", memory_system=mem)

    assert "kept 15 ingested document(s)" in report
    assert "+5 more" in report
    mem.delete_document_by_name.assert_not_called()


# ------------------------------- the two sandbox-escape guards, separated

@pytest.mark.asyncio
async def test_the_symlink_refusal_is_load_bearing_on_its_own(tmp_path):
    """Deleting either escape guard alone left `test_forget_sandbox_escape`
    green — they cover each other. This one needs the symlink refusal: the
    link's target is INSIDE the sandbox, so the containment check passes and
    only the symlink test stands between `forget` and following the link."""
    (tmp_path / "real.txt").write_text("x")
    (tmp_path / "atlas.txt").symlink_to(tmp_path / "real.txt")

    report = await tool_unified_forget(
        "atlas.txt", sandbox_dir=tmp_path, memory_system=_memsys())

    assert (tmp_path / "real.txt").exists(), "deleted THROUGH the symlink"
    assert "Refused symlink" in report


@pytest.mark.asyncio
async def test_the_containment_check_is_load_bearing_on_its_own(tmp_path):
    """And this one needs `_is_within_root`: a real file inside the sandbox
    whose RESOLVED path escapes it, reached without any symlink of its own
    — a directory symlink on the path."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "atlas.txt").write_text("secret")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "linkdir").symlink_to(outside, target_is_directory=True)

    await tool_unified_forget(
        "atlas.txt", sandbox_dir=sandbox, memory_system=_memsys())

    assert (outside / "atlas.txt").exists(), (
        "a file outside the sandbox was deleted through a directory symlink"
    )


# ------------------------- the two halves must state ONE policy, together

@pytest.mark.asyncio
async def test_a_filename_target_wipes_one_document_not_its_stem_siblings():
    """The vector half never got the disk half's extension rule, so one
    report printed both policies two lines apart — "Nothing on disk is
    deleted for a partial name match" above "Vector: Wiped document
    'notes.txt'", the same names, and the irreversible half was the one
    ignoring the rule. Against the live library `forget('postgresql-19-A4.md')`
    destroyed the 7k-chunk manual: the incident the vector rule was written
    for, reached through the case it lacked."""
    mem = _memsys()
    mem.get_library = MagicMock(return_value=["notes.md", "notes.pdf", "notes.txt"])

    report = await tool_unified_forget("notes.md", memory_system=mem)

    assert [c[0][0] for c in mem.delete_document_by_name.call_args_list] == ["notes.md"]
    assert "Vector: kept 2 ingested document(s)" in report


@pytest.mark.asyncio
async def test_a_path_qualified_target_wipes_one_document_by_basename():
    """A path target names ONE document. `forget('projects/alpha/report.md')`
    deleted one file on disk and wiped three documents — `report.md`,
    `projects/alpha/report.md` and `report.txt`."""
    mem = _memsys()
    mem.get_library = MagicMock(
        return_value=["report.md", "projects/alpha/report.md", "report.txt"])

    report = await tool_unified_forget(
        "projects/alpha/report.md", memory_system=mem)

    assert [c[0][0] for c in mem.delete_document_by_name.call_args_list] == [
        "projects/alpha/report.md"]
    # ...and the same-basename document is offered, not taken.
    assert "Vector: kept" in report and "'report.md'" in report


@pytest.mark.asyncio
async def test_a_topic_still_takes_the_whole_stem_in_the_library():
    """The stem tier is right for a bare topic — that is the difference the
    two rules encode."""
    mem = _memsys()
    mem.get_library = MagicMock(return_value=["atlas.md", "atlas.pdf", "atlas_notes.md"])

    await tool_unified_forget("atlas", memory_system=mem)

    assert sorted(c[0][0] for c in mem.delete_document_by_name.call_args_list) == [
        "atlas.md", "atlas.pdf"]


# ----------------------------------------- the ambiguity gate on the disk

@pytest.mark.asyncio
async def test_an_ambiguous_bare_name_reports_instead_of_deleting(tmp_path):
    """All the conservatism had landed on the tier that does NOT delete.
    Measured on the live sandbox, `forget('index.html')` removed five files
    across five projects and `forget('app.py')` two — silently and
    irreversibly — while a single unambiguous substring match was refused
    and cost a second call. The line was inverted."""
    for rel in ("index.html", "projects/a/index.html", "projects/b/index.html"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")

    report = await tool_unified_forget(
        "index.html", sandbox_dir=tmp_path, memory_system=_memsys())

    assert len(_survivors(tmp_path)) == 3, "an ambiguous name deleted files"
    assert "kept 3 file(s)" in report
    # ...and every candidate is re-issuable as an exact, path-qualified target.
    assert "'./index.html'" in report
    assert "'projects/a/index.html'" in report


@pytest.mark.asyncio
async def test_the_reported_candidate_deletes_exactly_itself(tmp_path):
    """Obeying the report must not repeat the over-deletion it prevents."""
    for rel in ("index.html", "projects/a/index.html", "projects/b/index.html"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")

    await tool_unified_forget(
        "./index.html", sandbox_dir=tmp_path, memory_system=_memsys())

    assert not (tmp_path / "index.html").exists()
    assert (tmp_path / "projects" / "a" / "index.html").exists()
    assert (tmp_path / "projects" / "b" / "index.html").exists()


@pytest.mark.asyncio
async def test_an_unambiguous_name_still_deletes_without_a_second_call(tmp_path):
    """The gate must not make the ordinary case need two calls."""
    (tmp_path / "only.md").write_text("x")

    await tool_unified_forget(
        "only.md", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == []


@pytest.mark.asyncio
async def test_the_sandbox_prefix_does_not_erase_the_path(tmp_path):
    """`clean_target` strips a leading `sandbox/` — a shape `file_system`
    documents as observed live — and the path rule was computed AFTER that
    strip, so `sandbox/index.html` lost its only separator and fell back to
    the basename tier. Four characters reopened the original critical."""
    for rel in ("index.html", "projects/a/index.html"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")

    await tool_unified_forget(
        "sandbox/index.html", sandbox_dir=tmp_path, memory_system=_memsys())

    assert not (tmp_path / "index.html").exists()
    assert (tmp_path / "projects" / "a" / "index.html").exists()


@pytest.mark.asyncio
async def test_a_project_scoped_redundant_prefix_heals(tmp_path):
    """`knowledge_base` gets the PROJECT workspace as its root, so a model
    reading a listing produces `projects/<id>/x` while the root already IS
    that directory. The path rule turned a working call into a silent total
    no-op — and, because a path-qualified miss short-circuits, it printed no
    candidate line either: the caller was told the file does not exist."""
    root = tmp_path / "sandbox" / "projects" / "abc123"
    root.mkdir(parents=True)
    (root / "index.html").write_text("x")
    (root / "other.md").write_text("x")

    report = await tool_unified_forget(
        "projects/abc123/index.html", sandbox_dir=root, memory_system=_memsys())

    assert _survivors(root) == ["other.md"]
    assert "Deleted 'index.html'" in report


@pytest.mark.asyncio
@pytest.mark.parametrize("target,keep", [
    (".config/x.md", "config/x.md"),      # lstrip ate the dot, hitting a sibling
    ("../../etc/passwd", "etc/passwd"),   # ...and the traversal prefix
])
async def test_prefix_stripping_does_not_hit_a_different_file(
        tmp_path, target, keep):
    """`str.lstrip` takes a CHARACTER SET, not a prefix."""
    (tmp_path / keep).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / keep).write_text("x")

    await tool_unified_forget(
        target, sandbox_dir=tmp_path, memory_system=_memsys())

    assert (tmp_path / keep).exists(), f"{target!r} deleted an unnamed file"


@pytest.mark.asyncio
async def test_a_dotfile_target_names_one_file(tmp_path):
    """`Path('.env').suffix` is `''`, so a fully-qualified dotfile was
    classified as a bare topic and fell to the stem tier."""
    (tmp_path / ".env.local").write_text("x")
    (tmp_path / ".env.production").write_text("x")

    await tool_unified_forget(
        ".env", sandbox_dir=tmp_path, memory_system=_memsys())

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".env.local", ".env.production"]


# --------------------------- one target, one rule, across all five sweeps

@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["atlas", "./atlas", "sandbox/atlas"])
async def test_every_sweep_sees_the_same_normalised_target(tmp_path, spelling):
    """`clean_target` strips `./`, a leading `/` and a `sandbox/` prefix —
    and only the disk and document halves read it. The semantic, profile and
    graph sweeps took the RAW string, so `forget('./atlas')` and
    `forget('sandbox/atlas')` removed the file and the document and left
    every fact, profile row and graph edge in place, while the report said
    nothing about the half that had not run. `sandbox/` is documented in
    file_system.py as an observed live model shape.

    Uses the REAL ProfileMemory and a graph spy — an over-mocked profile
    never reaches the sweep at all, which is how the first version of this
    test passed for the plain spelling and failed for every other reason.
    """
    from ghost_agent.memory.profile import ProfileMemory

    (tmp_path / "atlas.md").write_text("x")
    pm = ProfileMemory(tmp_path)
    pm.update("interests", "atlas", "maps and charts")

    graph = MagicMock()
    graph.get_connected_entities = MagicMock(return_value=[])
    graph.delete_by_target = MagicMock(return_value=1)

    await tool_unified_forget(
        spelling, sandbox_dir=tmp_path, memory_system=_memsys(),
        profile_memory=pm, graph_memory=graph)

    assert graph.delete_by_target.call_args[0][0] == "atlas", (
        f"{spelling!r} reached the graph sweep unnormalised"
    )
    assert "atlas" not in (pm.load().get("interests") or {}), (
        f"{spelling!r} never reached the profile sweep, so the row survives "
        f"and keeps being injected into the system prompt every turn"
    )


@pytest.mark.asyncio
async def test_the_ambiguity_gate_covers_the_tier_that_deletes(tmp_path):
    """The gate looked at `exact_hits` only, so dropping the extension —
    five characters — turned a refusal back into a five-file delete through
    the stem tier it never inspected."""
    for rel in ("index.js", "p/a/index.html", "p/b/index.html"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")

    report = await tool_unified_forget(
        "index", sandbox_dir=tmp_path, memory_system=_memsys())

    assert len(_survivors(tmp_path)) == 3
    assert "kept 3 file(s)" in report


@pytest.mark.asyncio
async def test_one_directory_is_not_ambiguous(tmp_path):
    """Ambiguity is about LOCATION. `forget('notes')` taking notes.md and
    notes.txt from one directory is the stem tier doing its job — the gate
    must not make the ordinary case need two calls."""
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    await tool_unified_forget(
        "notes", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == []


@pytest.mark.asyncio
async def test_the_document_half_is_gated_too():
    """The gate was disk-only: one report printed "kept 3 file(s) … Nothing
    on disk is deleted for a partial name match" above three "Wiped
    document" lines naming the same three files."""
    mem = _memsys()
    mem.get_library = MagicMock(return_value=[
        "notes.md", "projects/a/notes.md", "projects/b/notes.md"])

    report = await tool_unified_forget("notes.md", memory_system=mem)

    mem.delete_document_by_name.assert_not_called()
    assert "Vector: kept 3 ingested document(s)" in report


@pytest.mark.asyncio
async def test_a_trailing_slash_does_not_split_the_call(tmp_path):
    """`'notes/'.strip('/')` is `'notes'`, so the path test said "no path"
    while `clean_target` kept the slash — the disk half matched `notes` as a
    bare topic and deleted five files, and the profile and graph halves
    matched nothing at all. One call, two different targets."""
    (tmp_path / "notes.md").write_text("x")
    graph = MagicMock()
    graph.get_connected_entities = MagicMock(return_value=[])

    await tool_unified_forget(
        "notes/", sandbox_dir=tmp_path, memory_system=_memsys(),
        graph_memory=graph)

    assert graph.delete_by_target.call_args[0][0] == "notes", (
        "the trailing slash reached the graph sweep"
    )


@pytest.mark.asyncio
async def test_dot_prefixed_sources_are_distinct_documents():
    """`lstrip("./")` — the character-set bug this module documents forty
    lines above — collapsed `notes.md`, `.notes.md`, `..notes.md` and
    `./notes.md` onto one key, so one call wiped four distinct documents."""
    mem = _memsys()
    mem.get_library = MagicMock(
        return_value=["notes.md", ".notes.md", "..notes.md"])

    await tool_unified_forget("./notes.md", memory_system=mem)

    assert [c[0][0] for c in mem.delete_document_by_name.call_args_list] == [
        "notes.md"]


@pytest.mark.asyncio
async def test_a_trailing_slash_is_not_a_path(tmp_path):
    """`'notes/'` names a topic, not a path. With the trailing separator
    left on the probe it read as path-qualified, which turns the whole call
    into an exact-path match that hits nothing."""
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    await tool_unified_forget(
        "notes/", sandbox_dir=tmp_path, memory_system=_memsys())

    assert _survivors(tmp_path) == [], (
        "a trailing slash made the target path-qualified, so the stem tier "
        "never ran"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["atlas", "./atlas", "sandbox/atlas"])
async def test_the_semantic_sweep_queries_the_normalised_target(spelling):
    """The semantic sweep built its own `str(target).strip().lower()` and
    embedded it directly, so `./atlas` and `sandbox/atlas` went into the
    query vector verbatim and the literal-mention override — written
    because "the distance threshold silently missed facts that name the
    target outright" — could never fire."""
    mem = _memsys()
    await tool_unified_forget(spelling, memory_system=mem)

    assert mem.collection.query.called
    for call in mem.collection.query.call_args_list:
        for text in call.kwargs.get("query_texts", []):
            assert text == "atlas", (
                f"{spelling!r} was embedded unnormalised as {text!r}"
            )


# ------------------- defects the ambiguity fix itself introduced (round 3b)

@pytest.mark.asyncio
async def test_the_gate_does_not_delete_the_tier_it_shadowed(tmp_path):
    """Clearing `exact_hits` alone handed the very next line —
    `chosen = exact_hits or stem_hits` — the weaker tier that had LOST to
    them. With `p/a/index`, `p/b/index` and `index.html`,
    `forget('index')` refused the two the caller may have meant and
    irreversibly deleted the third: the contradiction-in-one-report shape
    this gate exists to remove, recreated by the gate."""
    for rel in ("p/a/index", "p/b/index", "index.html"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")

    report = await tool_unified_forget(
        "index", sandbox_dir=tmp_path, memory_system=_memsys())

    assert len(_survivors(tmp_path)) == 3, (
        "the gate refused the ambiguous matches and deleted the shadowed one"
    )
    assert "Deleted" not in report


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["mortimer", "./mortimer", "sandbox/mortimer"])
async def test_the_expansion_and_the_value_prune_see_the_same_target(
        tmp_path, spelling):
    """Two consumers were left one revision behind when the sweeps were
    normalised.

    The entity expansion is the AMPLIFIER of a forget — it is what reaches
    the alias tombstone ('mortimer' -> 'iguana') — and it still read the raw
    string, so it was dead for exactly the two spellings the normalisation
    was added for. And the profile value sweep had its GUARD normalised
    while its argument stayed raw: `_value_mentions_target(item, target_lc)`
    said the value mentions the target, then handed `prune_value` a string
    that matched nothing — a green tick on a no-op, with the pet left in the
    system prompt.
    """
    from ghost_agent.memory.profile import ProfileMemory

    pm = ProfileMemory(tmp_path)
    # Two successive updates make a real LIST (passing a list literal is
    # stringified, which routes the sweep down the scalar delete path and
    # never exercises prune_value at all).
    pm.update("assets", "pets", "Hanzo the dog")
    pm.update("assets", "pets", "Mortimer the iguana")
    graph = MagicMock()
    # NO expansion: the value prune is then the only path that can remove
    # the entry, so this test cannot pass for some other sweep's reason.
    graph.get_connected_entities = MagicMock(return_value=[])
    graph.delete_by_target = MagicMock(return_value=1)

    report = await tool_unified_forget(
        spelling, sandbox_dir=tmp_path, memory_system=_memsys(),
        profile_memory=pm, graph_memory=graph)

    assert graph.get_connected_entities.call_args[0][0] == "mortimer", (
        f"{spelling!r} reached the entity expansion unnormalised, so the "
        f"alias tombstone is never swept"
    )
    _pets = str(pm.load().get("assets", {}).get("pets") or [])
    assert "Mortimer" not in _pets, (
        f"{spelling!r} reported a value prune that did not happen: {_pets}"
    )
    assert "Hanzo" in _pets, "the prune took an unrelated value with it"
    # ...and the report names the string the sweeps actually used.
    assert f"'{spelling}'" not in report or spelling == "mortimer", (
        f"the report names {spelling!r}, which no sweep consulted"
    )
