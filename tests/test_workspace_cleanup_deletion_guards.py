"""Workspace cleanup — the three ways a registered/source file got deleted.

§4AT subsystem D, 2026-08-11. This module DELETES, irreversibly, unattended,
and (until now) without recording anything in the store. Its own contract says
"Source/document files are NEVER deleted here regardless of registration".
That was false three different ways, all reproduced against the real code:

 1. NAME BEAT SUBSTANCE. `_is_debris` matched `_SCRATCH_PREFIXES`
    (`temp_`, `tmp_`, `scratch_`, `debug_`) and the idle tidy checked it
    BEFORE `_is_source_like`, so `debug_utils.py`, `temp_loader.py`,
    `scratch_notes.py` and `tmp_fix.py` were deleted from ACTIVE projects
    after the grace period. A name is a guess about intent; being source is a
    fact about the file.
 2. CASE. The keep-set was exact-case, so a deliverable registered as
    `assets/hero.png` and written as `assets/Hero.png` was not in it and was
    deleted — by the DONE sweep AND the tidy. On macOS those are ONE file, so
    nothing else in the stack ever surfaces the drift.
 3. TWO NORMALIZERS. `workspace_cleanup._normalize_rel` stripped each prefix
    once, in one order; `ProjectStore._normalize_rel_path` looped differently.
    `/workspace//projects/<id>/x.png` reduced to two different keys, so the
    registration stopped protecting. `tools/projects.py action=artifact_add`
    stores a raw model-supplied payload, so the divergent input is reachable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import workspace_cleanup as WC  # noqa: E402
from ghost_agent.memory.projects import ProjectStore  # noqa: E402


# ── 1. a scratch-looking NAME must not condemn a source file ────────────

def test_scratch_named_SOURCE_survives_the_tidy_on_an_ACTIVE_project():
    """⚠ THE SCOPE THAT MATTERS. `_is_debris` is UNCHANGED — a scratch name is
    still debris globally, because the DONE sweep is supposed to collect
    `temp_probe.py` from a FINISHED project (pinned by
    `test_partial_keepset_debris_named_sources_still_swept`, which my first
    attempt turned red and which was right).

    What changed is the TIDY's order: on a project still being worked on,
    source-ness is checked BEFORE the name, so a build's `debug_utils.py` is
    not deleted out from under it. Same word, two lifecycles, opposite
    correct answers."""
    # scratch-named SOURCE survives the tidy…
    for rel in ("debug_utils.py", "temp_loader.py", "scratch_notes.py",
                "tmp_fix.py", "debug_panel.js"):
        assert WC._tidy_is_debris(rel) is False, rel
    # …while the global classifier stays untouched, so the DONE sweep still
    # collects it from a FINISHED project.
    assert WC._is_debris("temp_probe.py") is True

    # ⚠ AND THE EXEMPTION BELONGS TO ONE RULE ONLY. A blanket
    # "source-like ⇒ skip" kept `.browser_runner.py` — browser scaffolding
    # the tidy exists to remove, source-shaped only because it ends in `.py`.
    # That regression was caught by `test_tidy_removes_old_debris`.
    for rel in (".browser_runner.py", "__pycache__/m.pyc", "core.py.swp"):
        assert WC._tidy_is_debris(rel) is True, rel
    # (`node_modules/` is deliberately NOT in _SCRATCH_DIRS — my first fixture
    # assumed it was, and the test told me otherwise.)


def test_genuine_debris_is_STILL_debris():
    """⚠ OVER-SUPPRESSION GUARD, widened after review. The first version
    tested only .log/.pyc/.DS_Store/.bin — it avoided every suffix an
    over-broad source exemption would have swallowed, so it could not have
    caught the version of this fix that made scratch text uncollectable."""
    for rel in ("build.log", "__pycache__/x.pyc", ".DS_Store",
                "tmp_scratch.bin", "debug_dump.bin",
                "tmp_iter_0.json", "debug_dump_0.csv", "temp_notes.txt",
                "scratch_data.yaml", "temp_probe.py"):
        assert WC._is_debris(rel) is True, rel


def test_CI_config_dotfiles_are_deliverables():
    """`.gitlab-ci.yml` was measured being deleted from an ACTIVE project."""
    for rel in (".gitlab-ci.yml", ".travis.yml", ".pre-commit-config.yaml",
                ".gitignore", ".env"):
        assert WC._is_debris(rel) is False, rel
    # …and an unknown dotfile is still debris — the keep-list is a list, not
    # a blanket amnesty for anything starting with a dot.
    assert WC._is_debris(".some_random_cache") is True


# ── 2. case ─────────────────────────────────────────────────────────────

def test_protection_matching_ignores_CASE():
    """The live shape: registered `assets/hero.png`, on disk `assets/Hero.png`.

    ⚠ A `set` SUBCLASS WAS TRIED FIRST AND WAS A LANDMINE — overriding
    `__contains__`/`add`/`__or__` left `update`, `|=`, `discard`, `clear`,
    `copy`, `union`, `-`, `&`, `^` unguarded, several returning a PLAIN set
    (protection silently lost) or desyncing the mirror. It also missed
    `_recover_deliverables`, which returns an ordinary set — so the
    empty-registration recovery, the module's own safety net, stayed
    case-sensitive. A projection function takes whatever set it is handed."""
    low = WC._lower_keys({"assets/hero.png", "App.py"})
    assert "assets/Hero.png".lower() in low
    assert "ASSETS/HERO.PNG".lower() in low
    assert "app.py".lower() in low          # works on a plain set, by design
    assert "assets/other.png".lower() not in low
    assert WC._lower_keys(None) == set() and WC._lower_keys(()) == set()


def test_every_protection_membership_test_is_case_folded():
    """⚠ HALF-APPLIED IS THE FAILURE MODE. Fixing the keep-set while leaving
    the sibling `referenced`-media check exact-case still deletes the asset.
    Pins that no protection site compares raw case."""
    import inspect
    for fn in (WC.sweep_project_workspace, WC.tidy_project_workspace):
        src = inspect.getsource(fn)
        assert "rel in keep" not in src, fn.__name__
        assert "in referenced" not in src or "_ref_low" in src, fn.__name__


# ── 3. the two normalizers ──────────────────────────────────────────────

_PREFIXES = ["", "/", "//", "./", "workspace/", "/workspace/", "workspace//",
             "projects/PID/", "/projects/PID/", "workspace/projects/PID/",
             "/workspace/projects/PID/", "/workspace//projects/PID/"]
# Tails deliberately include a project that genuinely CONTAINS a `workspace/`
# or `projects/PID/` directory — the shapes a curated list would omit.
_TAILS = ["a.py", "assets/hero.png", "workspace/config.json",
          "projects/PID/x.py", "workspace/assets/hero.png", "docs/readme.md"]
_PAYLOADS = [p + t for p in _PREFIXES for t in _TAILS]


def test_the_two_normalizers_AGREE_on_every_payload_shape():
    """⚠ THE DIFFERENTIAL PIN, and it GENERATES its corpus.

    The first version asserted over 9 hand-picked payloads — and review
    showed the list contained zero of the shapes that actually diverged, so
    it measured a chosen sample rather than the contract. A cross-product of
    prefixes × tails cannot be curated to agree.

    Two implementations of one contract IS the defect: `/workspace//projects/
    <id>/x.png` once yielded `projects/<id>/x.png` here and `x.png` in the
    store — one file, two keys, registration silently stops protecting.
    """
    bad = []
    for p in _PAYLOADS:
        a = WC._normalize_rel(p, "PID")
        b = ProjectStore._normalize_rel_path("PID", p)
        if a != b:
            bad.append(f"{p!r}: cleanup={a!r} store={b!r}")
    assert not bad, (f"{len(bad)}/{len(_PAYLOADS)} payloads normalize "
                     f"differently:\n  " + "\n  ".join(bad[:10]))


def test_re_normalization_keeps_the_two_implementations_IN_LOCKSTEP():
    """⚠ THE PROPERTY THAT ACTUALLY PREVENTS DELETION IS AGREEMENT, NOT
    IDEMPOTENCE — and the generated corpus is what made the difference clear.

    `workspace/workspace/config.json` is deliberately NOT a fixed point: one
    pass strips the prefix and leaves the project's real `workspace/`
    directory, a second pass would strip that too. Stripping to a fixed point
    was tried and was worse — it ate the real directory and deleted the
    registered file inside it.

    Single-pass is safe because the keep-set normalizes each stored payload
    exactly ONCE and the walk yields already-relative paths. What must hold is
    that both implementations move in step, so a key built by one is always
    found by the other — including if either is ever applied twice.
    """
    for p in _PAYLOADS:
        a1 = WC._normalize_rel(p, "PID")
        b1 = ProjectStore._normalize_rel_path("PID", p)
        assert a1 == b1, p
        if a1 is None:
            continue
        assert (WC._normalize_rel(a1, "PID")
                == ProjectStore._normalize_rel_path("PID", b1)), (
            f"{p!r} diverges on a second pass")


def test_traversal_is_still_rejected():
    """The hardening must not have opened a path-escape."""
    for bad in ("../../etc/passwd", "projects/PID/../../x",
                "/workspace/projects/PID/../../../root"):
        assert WC._normalize_rel(bad, "PID") is None, bad
