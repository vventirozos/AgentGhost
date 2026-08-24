"""§4CN E0 — the write fence and the lineage archive.

The documented failure mode of every DGM-lineage system is the same one:
the agent modifies the thing that measures it. DGM's agent fabricated
test logs; the AI Scientist raised its own timeout. Neither was caught by
a better mutation. So the pins here are almost entirely about the fence,
and they are deliberately adversarial: every one is a way a candidate
could reach the evaluator.

The archive pins are about the other half — HGM's result that a node's
own score poorly predicts its descendants', which is why selection is
clade-based and why "never evaluated" must outrank "evaluated and bad".
"""
import json
from pathlib import Path

import pytest

from ghost_agent.evolve import archive as A
from ghost_agent.evolve import fence as F


# ------------------------------------------------------------------ #
# The write fence                                                    #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("path", [
    "src/ghost_agent/tools/search.py",
    "src/ghost_agent/tools/browser.py",
    "src/ghost_agent/core/prompts.py",
    "src/ghost_agent/reflection/prompts.py",
])
def test_the_allow_list_admits_a_tool_or_a_prompt(path):
    ok, why = F.is_mutable(path)
    assert ok is True, why


@pytest.mark.parametrize("path,reason", [
    ("tests/test_search.py", "immutable"),
    ("tests/conftest.py", "immutable"),
    ("src/ghost_agent/eval/banks.py", "immutable"),
    ("scripts/ablation_paired.py", "immutable"),
    ("src/ghost_agent/evolve/fence.py", "immutable"),
    ("src/ghost_agent/core/agent.py", "immutable"),
    ("src/ghost_agent/core/isolation.py", "immutable"),
    ("src/ghost_agent/core/admissibility.py", "immutable"),
    ("src/ghost_agent/memory/skills.py", "immutable"),
    ("src/ghost_agent/sandbox/docker.py", "immutable"),
    ("src/ghost_agent/main.py", "immutable"),
    ("pytest.ini", "immutable"),
])
def test_the_evaluator_and_the_containment_are_immutable(path, reason):
    """Everything here either SCORES the candidate or CONTAINS it."""
    ok, why = F.is_mutable(path)
    assert ok is False and reason in why


def test_a_module_nobody_decided_about_is_refused():
    """An allow-list rather than a deny-list because the repo grows: a
    module added next month must be un-mutable until someone decides."""
    ok, why = F.is_mutable("src/ghost_agent/core/some_new_module.py")
    assert ok is False and "allow-list" in why


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "~/.ssh/authorized_keys",
    "src/../../etc/passwd",
    "src/ghost_agent/tools/../../../tests/conftest.py",
    "..",
    "",
    "   ",
])
def test_the_fence_fails_closed_on_anything_it_cannot_classify(path):
    """"I could not tell" must never read as "allowed" for a write
    fence."""
    assert F.is_mutable(path)[0] is False


def test_a_diff_prefix_does_not_walk_the_fence():
    """`a/`/`b/` prefixes are normal in a unified diff; a fence that
    reads them literally admits nothing, and one that strips them without
    normalising admits `a/../tests/`."""
    assert F.is_mutable("a/src/ghost_agent/tools/search.py")[0] is True
    assert F.is_mutable("b/tests/conftest.py")[0] is False
    assert F.is_mutable("a/../tests/conftest.py")[0] is False


def test_the_registry_and_validators_stay_immutable_inside_a_mutable_tree():
    """`tools/` is on the allow-list, but the tool SURFACE and the
    argument validators are not — a candidate that can edit those can
    change what every other guard sees."""
    assert F.is_mutable("src/ghost_agent/tools/registry.py")[0] is False
    assert F.is_mutable("src/ghost_agent/tools/validators.py")[0] is False


def test_an_empty_diff_is_rejected_not_allowed():
    """A diff that touches nothing is not a candidate, and treating it as
    allowed is how a no-op proposal reaches an operator carrying a claim
    of improvement."""
    ok, why = F.check_diff_scope([])
    assert ok is False and "no files" in why[0]


def test_one_bad_path_rejects_the_whole_diff():
    ok, rejects = F.check_diff_scope(["src/ghost_agent/tools/search.py",
                                      "tests/conftest.py"])
    assert ok is False and len(rejects) == 1


# ------------------------------------------------------------------ #
# The harness checksum — the guard the path check cannot be          #
# ------------------------------------------------------------------ #

def test_the_digest_covers_the_things_that_score_a_candidate():
    d = F.harness_digest()
    assert len(d) > 100
    assert any(p.startswith("tests/") for p in d)
    assert any(p.startswith("src/ghost_agent/eval/") for p in d)
    assert any(p.startswith("scripts/") for p in d)
    assert "pytest.ini" in d


def test_the_digest_notices_a_modification(tmp_path):
    tree = tmp_path / "tests"
    tree.mkdir()
    (tree / "t.py").write_text("assert True")
    before = F.harness_digest(tmp_path, trees=("tests",))
    (tree / "t.py").write_text("assert True  # changed")
    after = F.harness_digest(tmp_path, trees=("tests",))
    assert F.compare_harness(before, after) == ["MODIFIED tests/t.py"]


def test_the_digest_notices_an_ADDED_test(tmp_path):
    """A test file a candidate ADDS is a test file that runs. "It only
    added things" is exactly the shape of a candidate writing its own
    passing test."""
    tree = tmp_path / "tests"
    tree.mkdir()
    (tree / "t.py").write_text("assert True")
    before = F.harness_digest(tmp_path, trees=("tests",))
    (tree / "t_new.py").write_text("def test_always(): assert 1")
    after = F.harness_digest(tmp_path, trees=("tests",))
    assert F.compare_harness(before, after) == ["ADDED tests/t_new.py"]


def test_the_digest_notices_a_DELETED_test(tmp_path):
    tree = tmp_path / "tests"
    tree.mkdir()
    (tree / "t.py").write_text("assert True")
    before = F.harness_digest(tmp_path, trees=("tests",))
    (tree / "t.py").unlink()
    after = F.harness_digest(tmp_path, trees=("tests",))
    assert F.compare_harness(before, after) == ["REMOVED tests/t.py"]


def test_an_unchanged_harness_reports_nothing():
    d = F.harness_digest()
    assert F.compare_harness(d, F.harness_digest()) == []


def test_pycache_is_WATCHED_and_reported_as_BYTECODE(tmp_path):
    """⚠ THIS TEST ASSERTED THE OPPOSITE, and the reasoning was
    plausible: "a digest that changes on its own teaches the operator to
    ignore it — the same lesson as a monitor that cries on a benign
    zero." The flaw is that a `.pyc` is not a by-product of the file
    being hashed, it is a file Python imports **ahead of** it. An
    unchecked-hash `.pyc` is loaded without consulting the `.py` at all,
    so excluding bytecode from the digest let a planted one replace a
    module the digest still reported as pristine — the check made
    decorative by the one exclusion that mattered.

    The alarm-fatigue concern was real, so it is answered by a
    DISTINGUISHABLE LABEL rather than by looking away: an operator sees
    whether source moved or only bytecode did, and both abort, because
    both mean the harness changed under a run in progress. The cascade's
    own children never write bytecode, so inside a cascade window this
    is not churn."""
    tree = tmp_path / "tests"
    (tree / "__pycache__").mkdir(parents=True)
    (tree / "t.py").write_text("x = 1")
    before = F.harness_digest(tmp_path, trees=("tests",))
    (tree / "__pycache__" / "t.cpython-310.pyc").write_bytes(b"\x00\x01")
    moved = F.compare_harness(before, F.harness_digest(tmp_path,
                                                       trees=("tests",)))
    assert moved, "a planted .pyc was invisible to the digest"
    assert all(m.startswith("BYTECODE ") for m in moved), moved
    # …and a source change is NOT mislabelled as bytecode.
    (tree / "t.py").write_text("x = 2")
    src_moved = F.compare_harness(before, F.harness_digest(
        tmp_path, trees=("tests",)))
    assert any(m.startswith("MODIFIED") and m.endswith("t.py")
               for m in src_moved), src_moved


# ------------------------------------------------------------------ #
# Diff identity and shape                                            #
# ------------------------------------------------------------------ #

_DIFF = """diff --git a/src/ghost_agent/tools/search.py b/src/ghost_agent/tools/search.py
index abc123..def456 100644
--- a/src/ghost_agent/tools/search.py
+++ b/src/ghost_agent/tools/search.py
@@ -10,3 +10,3 @@
     ctx = 1
-    timeout = 5
+    timeout = 10
"""


def test_the_same_edit_hashes_the_same_after_a_line_shift():
    """EvoTrace measured ~30% of evolved lines to be resurrections. Line
    numbers and blob hashes move without the change moving, so a raw-text
    hash lets the same edit re-enter under a new identity and cost a full
    cascade to rediscover a verdict already on disk."""
    shifted = (_DIFF.replace("@@ -10,3 +10,3 @@", "@@ -400,3 +400,3 @@")
               .replace("index abc123..def456", "index 999999..888888"))
    assert A.normalized_diff_hash(_DIFF) == A.normalized_diff_hash(shifted)


def test_a_different_edit_hashes_differently():
    other = _DIFF.replace("+    timeout = 10", "+    timeout = 30")
    assert A.normalized_diff_hash(_DIFF) != A.normalized_diff_hash(other)


def test_context_only_changes_do_not_move_the_hash():
    more_context = _DIFF.replace("     ctx = 1", "     ctx = 1\n     ctx2 = 2")
    assert A.normalized_diff_hash(_DIFF) == A.normalized_diff_hash(more_context)


def test_a_diff_with_no_edits_has_no_identity():
    assert A.normalized_diff_hash("diff --git a/x b/x\n--- a/x\n+++ b/x\n") == ""


def test_both_sides_of_the_header_are_read():
    """A rename or a delete names its victim only on the `---` side. A
    fence reading one side can be walked around by a diff that only
    deletes."""
    d = ("diff --git a/tests/conftest.py b/dev/null\n"
         "--- a/tests/conftest.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-import x\n")
    paths = A.diff_touched_paths(d)
    assert any("tests/conftest.py" in p for p in paths)
    ok, rejects = F.check_diff_scope(paths)
    assert ok is False


@pytest.mark.parametrize("diff,frag", [
    ("", "changes nothing"),
    (_DIFF + _DIFF.replace("search.py", "a.py").replace("tools/a", "tools/b")
     + _DIFF.replace("search.py", "c.py"), "file cap"),
])
def test_an_oversized_or_empty_diff_is_rejected_mechanically(diff, frag):
    ok, why = A.check_diff_shape(diff)
    assert ok is False and frag in why


def test_a_whole_file_rewrite_is_rejected():
    """Not a quality judgement — a bounded diff is one an operator can
    actually read before approving."""
    big = (_DIFF
           + "\n".join(f"+    line_{i} = {i}" for i in range(200)) + "\n")
    ok, why = A.check_diff_shape(big)
    assert ok is False and "rewrite is a replacement" in why


# ------------------------------------------------------------------ #
# The archive                                                        #
# ------------------------------------------------------------------ #

@pytest.fixture
def arc(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    return A.Archive()


def test_admission_writes_a_node(arc):
    node, why = A.admit(arc, _DIFF, brief="tighten the search timeout")
    assert node is not None, why
    assert node.status == A.STATUS_CANDIDATE
    assert node.diff_hash and node.parent == A.ROOT_ID
    assert arc.get(node.id).brief.startswith("tighten")


def test_admission_refuses_a_diff_outside_the_fence(arc):
    bad = _DIFF.replace("src/ghost_agent/tools/search.py", "tests/conftest.py")
    node, why = A.admit(arc, bad)
    assert node is None and "immutable" in why


def test_admission_refuses_a_diff_the_lineage_has_already_seen(arc):
    first, _ = A.admit(arc, _DIFF)
    shifted = _DIFF.replace("@@ -10,3 +10,3 @@", "@@ -900,3 +900,3 @@")
    second, why = A.admit(arc, shifted)
    assert second is None
    assert first.id in why and "already in the archive" in why


def test_a_corrupt_node_costs_one_node_not_the_lineage(arc, tmp_path):
    good, _ = A.admit(arc, _DIFF)
    (tmp_path / "system" / "evolve" / "archive" / "broken.json").write_text(
        "{not json")
    nodes = arc.nodes()
    assert good.id in nodes and "broken" not in nodes


def test_a_reject_is_kept_as_evidence(arc):
    """An archive that keeps only survivors cannot tell "the mutator
    improved" from "the evaluator got looser"."""
    node, _ = A.admit(arc, _DIFF)
    arc.record_stage(node.id, "stage1", False, "3 pins failed")
    again = arc.get(node.id)
    assert again.status == A.STATUS_REJECTED
    assert again.eval["stage1"]["passed"] is False
    assert "3 pins" in again.eval["stage1"]["detail"]


# ------------------------------------------------------------------ #
# Selection is clade-based, not greedy                               #
# ------------------------------------------------------------------ #

def _node(arc, node_id, parent, score=None, status=A.STATUS_EVALUATED):
    arc.add(A.Node(id=node_id, parent=parent, diff=f"d-{node_id}",
                   diff_hash=node_id, score=score, status=status))


def test_a_clade_is_walked_transitively(arc):
    _node(arc, "a", A.ROOT_ID)
    _node(arc, "b", "a")
    _node(arc, "c", "b")
    assert {n.id for n in arc.clade("a")} == {"b", "c"}
    assert arc.clade_score("a")["descendants"] == 2


def test_a_cycle_in_the_archive_does_not_hang_the_nightly_job(arc):
    """A hand-edited archive can name a parent that is also a
    descendant, and a naive walk recurses forever inside a job nobody is
    watching."""
    _node(arc, "a", "b")
    _node(arc, "b", "a")
    assert {n.id for n in arc.clade("a")} == {"b"}


def test_selection_uses_the_best_DESCENDANT_not_the_node_itself(arc):
    """HGM's finding, and the whole reason this is not hill-climbing: a
    node's own score poorly predicts its descendants'. A weak node whose
    child went furthest must outrank a strong node whose children went
    nowhere."""
    _node(arc, "weak_parent", A.ROOT_ID, score=0.01)
    _node(arc, "strong_child", "weak_parent", score=0.90)
    _node(arc, "strong_parent", A.ROOT_ID, score=0.80)
    _node(arc, "dud_child", "strong_parent", score=0.02)
    w = A.selection_weights(arc)
    assert w["weak_parent"] > w["strong_parent"]


def test_never_evaluated_outranks_evaluated_and_bad(arc):
    """An archive that ranks the unknown below the known-bad stops
    exploring."""
    _node(arc, "untried", A.ROOT_ID)
    _node(arc, "tried_badly", A.ROOT_ID)
    _node(arc, "bad_child", "tried_badly", score=0.001)
    w = A.selection_weights(arc)
    assert w["untried"] > w["tried_badly"]


def test_a_much_tried_node_is_discounted(arc):
    """A node tried five times and still leading is worth trying again,
    but not five times as often as one nobody has touched."""
    _node(arc, "popular", A.ROOT_ID)
    _node(arc, "quiet", A.ROOT_ID)
    for i in range(5):
        _node(arc, f"kid{i}", "popular", score=0.5)
    _node(arc, "only_kid", "quiet", score=0.5)
    w = A.selection_weights(arc)
    assert w["quiet"] > w["popular"]


def test_a_rejected_node_is_never_a_parent(arc):
    _node(arc, "dead", A.ROOT_ID, status=A.STATUS_REJECTED)
    _node(arc, "alive", A.ROOT_ID)
    w = A.selection_weights(arc)
    assert "dead" not in w and "alive" in w


def test_the_root_is_always_selectable(arc):
    """A lineage that has painted itself into a corner must be able to
    start over from the unmodified tree."""
    assert A.ROOT_ID in A.selection_weights(arc)
    assert A.pick_parent(arc, 0.5) is not None
    _node(arc, "a", A.ROOT_ID)
    assert A.ROOT_ID in A.selection_weights(arc)


def test_parent_sampling_is_reproducible(arc):
    """A nightly job whose choice cannot be replayed cannot be
    debugged."""
    _node(arc, "a", A.ROOT_ID)
    _node(arc, "b", A.ROOT_ID)
    picks = {A.pick_parent(arc, r) for r in (0.0, 0.1, 0.5, 0.9, 0.999)}
    assert picks <= {A.ROOT_ID, "a", "b"}
    assert A.pick_parent(arc, 0.42) == A.pick_parent(arc, 0.42)


def test_sampling_covers_every_eligible_parent(arc):
    _node(arc, "a", A.ROOT_ID)
    _node(arc, "b", A.ROOT_ID)
    seen = {A.pick_parent(arc, i / 100.0) for i in range(100)}
    assert seen == {A.ROOT_ID, "a", "b"}


# ------------------------------------------------------------------ #
# The health surface                                                 #
# ------------------------------------------------------------------ #

def test_stats_report_where_candidates_died(arc):
    """"the mutator produced nothing usable" and "the mutator produced
    nothing" are different states."""
    assert A.archive_stats()["present"] is False
    n1, _ = A.admit(arc, _DIFF)
    arc.record_stage(n1.id, "stage0", False, "touched tests/")
    st = A.archive_stats()
    assert st["nodes"] == 1
    assert st["rejected_at"] == {"stage0": 1}
    assert st["by_status"][A.STATUS_REJECTED] == 1


def test_an_unscored_node_is_not_scored_as_zero(arc):
    """None means "never measured", which is NOT zero and must not be
    averaged as if it were."""
    _node(arc, "a", A.ROOT_ID, score=None)
    assert arc.clade_score(A.ROOT_ID)["best_descendant"] is None
    assert A.archive_stats()["best_score"] is None


def test_writing_without_a_home_is_silent(monkeypatch):
    monkeypatch.setenv("GHOST_HOME", "")
    a = A.Archive()
    assert a.nodes() == {}
    node, why = A.admit(a, _DIFF)
    assert node is None and "could not be written" in why


# ================================================================== #
# R4 — the fence on a case-insensitive filesystem                    #
# ================================================================== #

@pytest.mark.parametrize("path", [
    "src/ghost_agent/tools/Registry.py",
    "src/ghost_agent/tools/REGISTRY.py",
    "src/ghost_agent/tools/Validators.py",
    "src/ghost_agent/tools/__Init__.py",
    "TESTS/test_x.py",
    "src/ghost_agent/Core/agent.py",
])
def test_the_fence_survives_a_case_insensitive_filesystem(path):
    """⚠ VERIFIED END-TO-END BEFORE THE FIX, not hypothesised. This
    volume is case-insensitive APFS, so `Registry.py` IS `registry.py` —
    and an exact deny-list compared case-SENSITIVELY admitted it. A diff
    headed `a/src/ghost_agent/tools/Registry.py` cleared `is_mutable`,
    cleared `check_diff_scope`, cleared `archive.admit`, and `patch`
    wrote the real `tools/registry.py`: the first entry on the immutable
    list, "the tool surface itself". `deny-list-guards-leak` (§4CI) one
    layer down — a deny-list inside an allow-list, leaking."""
    from ghost_agent.evolve import fence as F
    ok, why = F.is_mutable(path)
    assert ok is False, f"{path} is writable: {why}"


def test_the_case_fold_does_not_close_the_allow_list():
    from ghost_agent.evolve import fence as F
    assert F.is_mutable("src/ghost_agent/tools/file_system.py")[0] is True
    assert F.is_mutable("SRC/GHOST_AGENT/TOOLS/file_system.py")[0] is True


def test_a_case_variant_diff_is_refused_by_the_scope_check():
    from ghost_agent.evolve import fence as F
    ok, bad = F.check_diff_scope(["a/src/ghost_agent/tools/Registry.py"])
    assert ok is False and bad


@pytest.mark.parametrize("path", [
    "src/ghost_agent/tools/regiſtry.py",     # U+017F LATIN SMALL LETTER LONG S
    "src/ghost_agent/tools/validatorſ.py",
])
def test_the_fence_survives_a_UNICODE_FOLDING_filesystem(path):
    """⚠ FOUND ONE ROUND AFTER `.lower()` WAS APPLIED, and verified end to
    end before the fix: `regiſtry.py` opens the existing `registry.py`,
    `.lower()` leaves `ſ` alone, and the candidate cleared `is_mutable`,
    `check_diff_scope`, `validate_diff` and `archive.admit` — after which
    `patch` wrote a modified tool registry. A case-fold can only ever
    approximate the volume's equivalence relation."""
    from ghost_agent.evolve import fence as F
    assert F.is_mutable(path)[0] is False, path
    assert F.check_diff_scope([f"a/{path}"])[0] is False


def test_the_filesystem_itself_is_asked_not_just_the_string():
    """`.casefold()` closes the codepoints anyone thought to fold.
    `os.path.samefile` compares device and inode, which is the volume's
    actual answer — and `Path.resolve()` is NOT that: on macOS it
    preserves the spelling it was given, so the two names resolve
    unequal while opening the same bytes."""
    from ghost_agent.evolve import fence as F
    root = Path(__file__).resolve().parents[1]
    for spelling in ("src/ghost_agent/tools/regiſtry.py",
                     "src/ghost_agent/tools/REGISTRY.py",
                     "src/ghost_agent/tools/Registry.py"):
        why = F.resolves_to_immutable(spelling, root)
        assert "immutable" in why, (spelling, why)
    assert F.resolves_to_immutable(
        "src/ghost_agent/tools/file_system.py", root) == ""
    assert F.resolves_to_immutable(
        "src/ghost_agent/tools/does_not_exist_yet.py", root) == ""


def test_the_tree_walk_catches_a_SYMLINK_into_an_immutable_tree(tmp_path):
    """⚠ The marginal coverage of `resolves_to_immutable`'s prefix walk
    over the string fence, which a reviewer could not construct and which
    is exactly this: a path INSIDE the allow-list that the filesystem
    resolves into an immutable tree. Case-folding cannot see it — the
    name is ASCII and on the allow-list; only `samefile` can."""
    from ghost_agent.evolve import fence as F
    repo = tmp_path / "repo"
    (repo / "src" / "ghost_agent" / "tools").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "tests" / "conftest.py").write_text("SCORES_THE_CANDIDATE = 1\n")
    link = repo / "src" / "ghost_agent" / "tools" / "helper.py"
    link.symlink_to(repo / "tests" / "conftest.py")
    # the STRING fence approves it — it is on the allow-list
    assert F.is_mutable("src/ghost_agent/tools/helper.py")[0] is True
    # the FILESYSTEM does not
    why = F.resolves_to_immutable("src/ghost_agent/tools/helper.py", repo)
    assert "immutable" in why and "tests" in why, why


def test_a_path_outside_the_repo_is_refused_not_ignored(tmp_path):
    """It returned "" — no objection — for `/etc/passwd`. Unreachable
    today because `validate_diff` runs first, but `materialize` never
    calls `is_mutable`, and the comment beside the call reads as if the
    samefile check subsumes the string fence."""
    from ghost_agent.evolve import fence as F
    why = F.resolves_to_immutable("../../../etc/passwd", tmp_path)
    assert why, "a path escaping the repo must not be silently approved"


def test_diff_size_counts_files_by_PREFIX_not_by_character_set():
    """⚠ `p.lstrip("ab/")` is a CHARACTER-SET strip, so several different
    files collapse onto one name and the count comes out too LOW —
    which is the direction that matters, because `MAX_FILES` is a cap.
    Demonstrated: `a/…/aa.py`, `a/…/bb.py` and `a/…/ab.py` all strip to
    the same string once the prefix is eaten, so a THREE-file diff
    counted as one and cleared a cap of two."""
    # the collapse, demonstrated on the transform itself
    assert "a/ab/a.py".lstrip("ab/") == "a/ab/b.py".lstrip("ab/") == ".py"
    parts = []
    for name in ("ab/a.py", "ab/b.py", "ab/x.py"):
        parts += [f"--- a/{name}", f"+++ b/{name}",
                  "@@ -1,1 +1,2 @@", " import os", "+import sys"]
    diff = "\n".join(parts) + "\n"
    files, _lines = A.diff_size(diff)
    assert files == 3, f"three files counted as {files}"
    ok, why = A.check_diff_shape(diff)
    assert ok is False and "file" in why, (ok, why)
    # ⚠ Reachability, stated honestly: every mutable prefix begins with
    # `src/` or `tests/`, whose first character stops the character-set
    # strip — so this cannot fire on a path the fence would admit. The
    # fix is correctness hygiene, not a live hole, and saying so is
    # cheaper than a future reader re-deriving it.


def test_a_planted_SYMLINK_is_visible_to_the_digest(tmp_path):
    """⚠ `is_file()` FOLLOWS SYMLINKS AND SO MISSES THEM. A DANGLING
    symlink is not a file, and `rglob` does not descend into a symlinked
    DIRECTORY — so both could be ADDED inside a watched, immutable tree
    and stay permanently invisible. That is worse than the
    modify-use-restore window the docstring concedes: a permanent
    addition the checksum can never see. Combined with a runner check
    that only inspects spelling, it yields an executable outside the
    canonical tree that the digest never notices."""
    import os
    tree = tmp_path / "scripts"
    tree.mkdir()
    (tree / "a.py").write_text("x = 1\n")
    before = F.harness_digest(tmp_path, trees=("scripts",))

    os.symlink("/nonexistent/target", str(tree / "dangling"))
    os.symlink(str(tmp_path), str(tree / "dirlink"))
    moved = F.harness_digest(tmp_path, trees=("scripts",))
    added = F.compare_harness(before, moved)
    assert sorted(added) == ["ADDED scripts/dangling",
                             "ADDED scripts/dirlink"], added

    # …and RE-POINTING a link is a change, because the link's own target
    # is what is hashed — the target may sit outside every watched tree.
    os.remove(str(tree / "dangling"))
    os.symlink("/etc", str(tree / "dangling"))
    repointed = F.compare_harness(moved, F.harness_digest(tmp_path,
                                                          trees=("scripts",)))
    assert repointed == ["MODIFIED scripts/dangling"], repointed


def test_a_RE_POINTED_link_is_caught_even_when_the_bytes_match(tmp_path):
    """⚠ WHY THE LINK'S TARGET STRING IS HASHED, NOT THE TARGET'S BYTES.
    Hashing the bytes makes re-pointing a link to a different file with
    IDENTICAL content invisible — and that is the move: the link keeps
    its name and its content hash while pointing somewhere the digest
    does not watch. Mutating this to hash the target's bytes survived
    the suite, because the only existing test distinguished two
    UNREADABLE reasons."""
    import os
    tree = tmp_path / "scripts"
    tree.mkdir()
    (tmp_path / "a.py").write_text("SAME = 1\n")
    (tmp_path / "b.py").write_text("SAME = 1\n")       # identical bytes
    os.symlink(str(tmp_path / "a.py"), str(tree / "link.py"))
    before = F.harness_digest(tmp_path, trees=("scripts",))

    os.remove(str(tree / "link.py"))
    os.symlink(str(tmp_path / "b.py"), str(tree / "link.py"))
    moved = F.compare_harness(before, F.harness_digest(tmp_path,
                                                       trees=("scripts",)))
    assert moved == ["MODIFIED scripts/link.py"], \
        f"a re-pointed link with identical target bytes was invisible: {moved}"


def test_src_INIT_is_refused_as_immutable_not_by_the_catch_all(tmp_path):
    """⚠ THE FILE DOES NOT EXIST, AND THAT IS THE POINT. Creating
    `src/__init__.py` turns `src/` from a PEP-420 namespace package into
    a regular one whose body runs on every `import src.ghost_agent.*` —
    the production import shape. It was refused only by the catch-all
    ("not on the mutable allow-list"), which is the same answer a typo
    gets; the entry one level down (`tools/__init__.py`) is named
    explicitly for exactly this hazard."""
    ok, why = F.is_mutable("src/__init__.py")
    assert not ok, why
    assert "immutable" in why, \
        f"refused, but by the catch-all rather than as immutable: {why}"
    assert not (Path(__file__).resolve().parents[1] / "src"
                / "__init__.py").exists(), \
        "src/__init__.py now EXISTS — `src` is no longer a namespace " \
        "package, and the production import shape has changed"
