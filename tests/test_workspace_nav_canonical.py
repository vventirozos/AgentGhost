"""Repeat-fetch nudge must count the PAGE, not the spelling (2026-09-15).

`record_navigation` fires once, at exactly the 3rd visit, keyed on the
literal URL — and it was only ever called on SUCCESS. Live (req
4b518a82) one Telegram post was fetched five times in a single turn:
three as `t.me/s/investigations/363`, two as `t.me/investigations/363`,
one of those a selector error that returned before the counter. Neither
key reached three, so the nudge that exists to break exactly this loop
never fired.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.workspace.model import WorkspaceModel


def _canon(url):
    return WorkspaceModel.canonical_nav_url(url)


def test_telegram_preview_and_post_share_one_key():
    """FAILS IF: the alias table is missing — the live 3+2 split."""
    assert _canon("https://t.me/s/investigations/363") == \
           _canon("https://t.me/investigations/363")


@pytest.mark.parametrize("host,prefix", WorkspaceModel._NAV_PATH_ALIASES)
def test_every_alias_entry_collapses_both_directions(host, prefix):
    """R1/R5 enumeration — FAILS IF: an entry is added that only works one
    way, or whose prefix is not actually stripped.

    Pinned over the TABLE, so a new host cannot be added without the
    property being checked for it.
    """
    aliased = f"https://{host}{prefix}chan/1"
    bare = f"https://{host}/chan/1"
    assert _canon(aliased) == _canon(bare)
    # …and the collapse must be to the BARE spelling, not to a third form.
    assert _canon(aliased) == bare


def test_alias_does_not_fire_on_another_host():
    """FAILS IF: the prefix is stripped host-blind.

    `example.com/s/foo` is an ordinary path; rewriting it would merge two
    genuinely different pages and UNDER-report distinct work.
    """
    assert _canon("https://example.com/s/foo") == "https://example.com/s/foo"


def test_subdomain_of_an_alias_host_still_collapses():
    """FAILS IF: matching is exact-host only and a subdomain slips past."""
    assert _canon("https://www.t.me/s/chan/1") == _canon("https://www.t.me/chan/1")


def test_fragment_and_trailing_slash_and_case_are_normalised():
    """FAILS IF: any of the three is left in the key.

    Each on its own splits the counter in half.
    """
    base = _canon("https://example.com/a/b")
    assert _canon("https://EXAMPLE.com/a/b") == base
    assert _canon("https://example.com/a/b/") == base
    assert _canon("https://example.com/a/b#section") == base


def test_query_string_is_preserved():
    """FAILS IF: the query is dropped to 'normalise' harder.

    `?q=cats` and `?q=dogs` are different pages; merging them would make
    a legitimate paged crawl look like a loop.
    """
    assert _canon("https://example.com/s?q=cats") != \
           _canon("https://example.com/s?q=dogs")


def test_root_path_keeps_its_slash():
    """FAILS IF: rstrip('/') eats the root, making '' the key."""
    assert _canon("https://example.com/") == "https://example.com/"


def test_garbage_input_is_returned_not_raised():
    """FAILS IF: canonicalisation can raise into a browser turn."""
    assert _canon("") == ""
    assert _canon("not a url at all") == "not a url at all"
    assert _canon(None) == ""


class _Model(WorkspaceModel):
    """Counter-only instance: the nudge path touches no store."""

    def __init__(self):
        self.enabled = True
        self.state = None
        self._nav_counts = {}
        self.notes = []

    def note(self, *a, **kw):
        self.notes.append((a, kw))


def test_the_live_five_fetch_loop_now_trips_the_nudge():
    """FAILS IF: the counter keys on the raw URL.

    The exact live sequence. Under the shipped code both keys stopped at
    2 and the nudge was silent through all five fetches.
    """
    m = _Model()
    seq = [
        "https://t.me/investigations/363",      # 1
        "https://t.me/s/investigations/363",    # 2
        "https://t.me/s/investigations/363",    # 3  <- must fire here
        "https://t.me/s/investigations/363",
        "https://t.me/investigations/363",
    ]
    fired = [m.record_navigation(u) for u in seq]
    assert fired[0] is None and fired[1] is None
    assert fired[2] and "3 times" in fired[2]
    # Fire-once contract survives the merge: later fetches stay quiet.
    assert fired[3] is None and fired[4] is None


def test_nudge_still_fires_at_three_for_a_single_spelling():
    """FAILS IF: canonicalisation shifted the threshold.

    The pre-existing contract — 3 identical visits — must be unchanged.
    """
    m = _Model()
    u = "https://example.com/page"
    assert m.record_navigation(u) is None
    assert m.record_navigation(u) is None
    assert m.record_navigation(u)


@pytest.mark.asyncio
async def test_a_failing_browser_op_counts_and_carries_the_nudge(tmp_path):
    """FAILS IF: the counter stays success-only.

    End-to-end, because the unit tests above cannot see the call site:
    the failure path RETURNS before the success-path counter, which is
    why one of the five live fetches was invisible. The nudge has to
    ride the failure hint — that is the text the model reads next.
    """
    from unittest.mock import MagicMock
    from ghost_agent.tools.browser import tool_browser

    stub = MagicMock()

    def _execute(cmd, timeout=300, **kwargs):
        return ("[BROWSER_ERR] ValueError: selector '.nope' did not match", 1)

    stub.execute = _execute
    wm = _Model()

    out = None
    for _ in range(3):
        out = await tool_browser(
            operation="extract_text", url="https://t.me/s/investigations/363",
            selector=".nope", sandbox_dir=tmp_path, sandbox_manager=stub,
            workspace_model=wm)
    assert "3 times" in str(out), "the third failing fetch must nudge"
    # …and it counted the PAGE, so a mixed-spelling loop reaches 3 too.
    wm2 = _Model()
    for url in ("https://t.me/investigations/363",
                "https://t.me/s/investigations/363",
                "https://t.me/s/investigations/363"):
        out = await tool_browser(
            operation="extract_text", url=url, selector=".nope",
            sandbox_dir=tmp_path, sandbox_manager=stub, workspace_model=wm2)
    assert "3 times" in str(out)


def test_disabled_model_never_counts():
    """FAILS IF: the enabled gate is bypassed by the new code path."""
    m = _Model()
    m.enabled = False
    for _ in range(5):
        assert m.record_navigation("https://example.com/x") is None
