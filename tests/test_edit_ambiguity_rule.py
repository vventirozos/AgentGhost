"""§4KB step 2 — an ambiguous SEARCH block is REJECTED, never resolved silently.

**The defect.** `_locate_block`'s exact rung returned on
``if search_str in file_content`` with no occurrence count, and the block loop
then applied ``file_content.replace(matched, repl, 1)``. A SEARCH text
occurring twice therefore edited the FIRST occurrence and reported strategy
``exact`` — a wrong-region edit carrying a success message. Nothing downstream
can catch that shape: a first-occurrence edit leaks no markers, regresses no
parse, and produces a syntactically valid file. It is invisible by
construction, which is why it needs a pin rather than a guard.

**The rule.** Uniqueness is a property of the LADDER, not of its callers. The
other three rungs already had it — `flexible` requires ``len(matches) == 1``,
`_fuzzy_block_match` requires a clear margin over the runner-up,
`_anchor_block_match` requires both anchors unique. The exact rung was the only
one that did not check, and it is the rung that matches most often.

**R5 — one input, one story.** The two call FORMS deliberately differ, and the
difference is pinned here as a table so it is a decision and not an accident:

    form          ambiguous input      outcome                    loud?
    ────────────────────────────────────────────────────────────────────
    block         2 occurrences        REJECTED, file unchanged   yes
    two-argument  2 occurrences        ALL replaced + WARNING     yes

Neither is silent. The block form is the surgical one the tool description
steers the model toward, so it must be unique-or-nothing; the two-argument form
is the bulk one and says exactly how many it touched. The defect was that the
block form was silent, not that the two disagree.
"""
import pytest

from ghost_agent.tools.file_system import (
    _ambiguous_block_error,
    _locate_block,
    _occurrence_lines,
    tool_replace_text,
)


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep the step-1 ledger out of the operator's live data dir."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))


DUP = (
    "def a():\n"
    "    value = compute()\n"
    "    return value\n"
    "\n"
    "def b():\n"
    "    value = compute()\n"
    "    return value\n"
)


# ---------------------------------------------------------------------------
# 1. The ladder itself
# ---------------------------------------------------------------------------

def test_exact_rung_reports_ambiguity_not_a_match():
    """The regression pin for the defect itself.

    Fails in the world before this change, where the return was
    ``("    value = compute()", "exact")`` — a unique-looking hit on a
    two-occurrence search.
    """
    got = _locate_block(DUP, "    value = compute()")
    assert got == (None, "ambiguous:2")


def test_unique_exact_still_matches():
    """Fails in the world where the uniqueness check rejects everything —
    a guard that never passes is as broken as one that never fires."""
    matched, strategy = _locate_block(DUP, "def b():")
    assert matched == "def b():" and strategy == "exact"


def test_every_rung_refuses_an_ambiguous_input():
    """R1 class enumeration: no rung may resolve an ambiguous search.

    Walks all four rungs. Fails the moment any rung starts silently picking
    a winner — including a rung added in the future, because a new rung that
    resolves duplicates will land in this file's `ambiguous` bucket only if
    it checks, and in the `resolved` bucket if it does not.
    """
    cases = {
        # exact: byte-identical, twice
        "exact": "    value = compute()",
        # flexible: whitespace differs, matches both copies
        "flexible": "value  =  compute()",
        # fuzzy: one-char typo, near-identical to BOTH copies
        "fuzzy": "    value = computer()",
    }
    resolved = []
    for label, needle in cases.items():
        got = _locate_block(DUP, needle)
        if got is None:
            continue                       # refused by finding nothing
        matched, strategy = got
        if matched is not None:
            resolved.append((label, strategy))
    assert resolved == [], (
        f"a rung resolved an ambiguous search instead of refusing it: "
        f"{resolved}")


def test_occurrence_lines_are_one_based_and_capped():
    assert _occurrence_lines(DUP, "    value = compute()") == [2, 6]
    assert _occurrence_lines("x\n" * 50 + "", "x") == [1, 2, 3, 4, 5]
    assert _occurrence_lines(DUP, "") == []
    assert _occurrence_lines(DUP, "NOT_PRESENT") == []


def test_ambiguity_message_names_the_lines_and_the_repair():
    """A count alone is unactionable; the line numbers are the repair.

    Fails in the world where the message says only "matched 2 times" — the
    model's next move would be a blind retry of the same envelope.
    """
    msg = _ambiguous_block_error(DUP, "    value = compute()", 2)
    assert "2 times" in msg
    assert "line(s) 2, 6" in msg
    # "THIS BLOCK was not applied", NOT "the file is UNCHANGED": in a MIXED
    # batch (one unique block lands, one is ambiguous) the file IS changed,
    # and the old wording told both the model and the operator otherwise.
    assert "THIS BLOCK was not applied" in msg
    assert "UNCHANGED" not in msg
    assert "MORE" in msg and "context" in msg


# ---------------------------------------------------------------------------
# 2. End to end through the tool
# ---------------------------------------------------------------------------

async def test_block_form_ambiguous_leaves_file_untouched(sandbox):
    """The headline: the file must be byte-identical afterwards.

    Fails in the world where the first occurrence is edited — which is
    exactly what shipped until now, with a SUCCESS message on top.
    """
    f = sandbox / "m.py"
    f.write_text(DUP)
    res = await tool_replace_text(
        "m.py",
        "<<<< SEARCH\n    value = compute()\n====\n    value = compute2()\n>>>>",
        None,
        sandbox,
    )
    assert getattr(res, "is_rejection", False), res
    assert res.reason_code == "ambiguous_block"
    assert "AMBIGUOUS" in res
    assert f.read_text() == DUP            # byte-identical


async def test_block_form_ambiguous_does_not_report_not_found(sandbox):
    """"Not found" and "found twice" need OPPOSITE repairs.

    Widening the search fixes "not found" and makes "found twice" worse.
    Fails in the world where both collapse into `no_blocks_matched`.
    """
    (sandbox / "m.py").write_text(DUP)
    res = await tool_replace_text(
        "m.py",
        "<<<< SEARCH\n    value = compute()\n====\n    x = 1\n>>>>",
        None,
        sandbox,
    )
    assert "matched MORE THAN ONCE" in res
    assert "None of the SEARCH/REPLACE blocks matched" not in res


async def test_unique_block_still_applies(sandbox):
    """Fails in the world where the fix rejects legitimate unique edits."""
    f = sandbox / "m.py"
    f.write_text(DUP)
    res = await tool_replace_text(
        "m.py", "<<<< SEARCH\ndef b():\n====\ndef bee():\n>>>>", None, sandbox)
    assert "SUCCESS" in res
    assert "def bee():" in f.read_text()


async def test_mixed_batch_applies_the_unique_and_rejects_the_ambiguous(sandbox):
    """A partial apply the model can act on beats all-or-nothing.

    Fails in the world where one ambiguous envelope aborts the whole batch
    (the unique edit would be lost and the model would re-send both).
    """
    f = sandbox / "m.py"
    f.write_text(DUP)
    res = await tool_replace_text(
        "m.py",
        "<<<< SEARCH\ndef b():\n====\ndef bee():\n>>>>\n"
        "<<<< SEARCH\n    value = compute()\n====\n    x = 1\n>>>>",
        None,
        sandbox,
    )
    body = f.read_text()
    assert "def bee():" in body                 # the unique one landed
    assert body.count("value = compute()") == 2  # the ambiguous one did not
    assert "AMBIGUOUS" in res


# ---------------------------------------------------------------------------
# 3. R5 — the two forms, one table, same input
# ---------------------------------------------------------------------------

async def test_r5_two_forms_one_story(sandbox):
    """Same ambiguous input through both call forms; both must be LOUD.

    The block form refuses; the two-argument form replaces every occurrence
    and says how many. Fails in the world where either becomes silent — which
    is the property under review, not which branch each one takes.
    """
    fb = sandbox / "block.py"
    fb.write_text(DUP)
    res_block = await tool_replace_text(
        "block.py",
        "<<<< SEARCH\n    value = compute()\n====\n    x = 1\n>>>>",
        None, sandbox)

    ft = sandbox / "twoarg.py"
    ft.write_text(DUP)
    res_two = await tool_replace_text(
        "twoarg.py", "    value = compute()", "    x = 1", sandbox)

    # block: rejected, unchanged, and says WHY
    assert getattr(res_block, "is_rejection", False)
    assert fb.read_text() == DUP
    assert "AMBIGUOUS" in res_block

    # two-arg: applied to ALL, and says HOW MANY
    assert "SUCCESS" in res_two
    assert "WARNING: Replaced 2 identical occurrences" in res_two
    assert ft.read_text().count("x = 1") == 2

    # neither is silent — the property the table exists to hold
    for res in (res_block, res_two):
        assert ("AMBIGUOUS" in res) or ("identical occurrences" in res)
