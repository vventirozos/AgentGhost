"""§4KW review: the last test of a class or module whose wider-scoped fixture is
torn down inside that test's protocol must not keep its own leaks — only what
the fixture's teardown changed is folded into the baseline. Fails in the world
where the post-finalizer folds the whole state (the first cut did: 37 files).
"""
import os
import unittest

import pytest

from ghost_agent.core import announced_work as AW

_BUDGET = AW.CHECK_BUDGET_S


class TestWithAClassFixture:
    @pytest.fixture(scope="class", autouse=True)
    def _class_env(self):
        os.environ["GHOST_4KW_CLASS_FIXTURE"] = "on"
        yield
        del os.environ["GHOST_4KW_CLASS_FIXTURE"]

    def test_1_sees_the_fixture(self):
        assert os.environ.get("GHOST_4KW_CLASS_FIXTURE") == "on"

    def test_2_last_of_the_class_leaks(self):
        assert os.environ.get("GHOST_4KW_CLASS_FIXTURE") == "on"      # kept between tests of the class
        os.environ["GHOST_4KW_LAST_TEST_LEAK"] = "1"
        AW.CHECK_BUDGET_S = 999.0


def test_3_after_the_class_nothing_leaked():
    assert "GHOST_4KW_LAST_TEST_LEAK" not in os.environ
    assert "GHOST_4KW_CLASS_FIXTURE" not in os.environ                # the fixture's own teardown is kept
    assert AW.CHECK_BUDGET_S == _BUDGET


class TestAUnittestCase(unittest.TestCase):
    """pytest injects a class-scoped setUpClass fixture into every TestCase."""

    def test_only_one_leaks(self):
        os.environ["GHOST_4KW_UNITTEST_LEAK"] = "1"
        AW.MAX_CHECKED_CHARS = 1


def test_4_after_the_testcase_nothing_leaked():
    assert "GHOST_4KW_UNITTEST_LEAK" not in os.environ
    assert AW.MAX_CHECKED_CHARS == 800


# ── contents of module-level containers (§4KW open item) ──────────────────
# 44 containers carried one file's entries into the next (search result cache,
# notify rate-limit timestamps, experiment-registry cache, optim epochs/pins).
from ghost_agent.core import objection as _OBJ

_UNIT_KEYS = set(_OBJ._UNIT_MAP)


def test_5_a_test_adds_to_a_module_dict():
    _OBJ._UNIT_MAP["zz-4kw-probe"] = "x"
    _OBJ._UNIT_MAP.pop(next(iter(_UNIT_KEYS)))


def test_6_the_next_test_sees_the_original_contents_in_the_same_object():
    assert set(_OBJ._UNIT_MAP) == _UNIT_KEYS                     # restored IN PLACE: same object
    from ghost_agent.core import objection
    assert objection._UNIT_MAP is _OBJ._UNIT_MAP


class TestAClassFixtureFillsACache:
    @pytest.fixture(scope="class", autouse=True)
    def _fill(self):
        _OBJ._UNIT_MAP["zz-4kw-class"] = "kept for the class"
        yield
        _OBJ._UNIT_MAP.pop("zz-4kw-class", None)

    def test_7_first(self):
        assert _OBJ._UNIT_MAP.get("zz-4kw-class") == "kept for the class"

    def test_8_second_still_sees_it(self):
        assert _OBJ._UNIT_MAP.get("zz-4kw-class") == "kept for the class"


def test_9_after_the_class_it_is_gone():
    assert "zz-4kw-class" not in _OBJ._UNIT_MAP and set(_OBJ._UNIT_MAP) == _UNIT_KEYS


def test_every_live_state_exclusion_names_a_real_container():
    """A rename would turn an exclusion into a silent restore of live state."""
    import importlib
    from tests.conftest import _LIVE_STATE_CONTAINERS, _CONTAINER_TYPES
    for name in _LIVE_STATE_CONTAINERS:
        mod, attr = name.rsplit(".", 1)
        v = getattr(importlib.import_module("ghost_agent." + mod), attr)
        assert isinstance(v, _CONTAINER_TYPES), name


def test_the_ddgs_patch_state_survives_between_tests_1():
    from ghost_agent.tools import search
    search._ddgs_patch_state["zz-4kw-applied"] = True


def test_the_ddgs_patch_state_survives_between_tests_2():
    from ghost_agent.tools import search
    assert search._ddgs_patch_state.pop("zz-4kw-applied", None) is True


# The egress guard PATCHES the socket class: its `_INSTALLED` / `_ORIGINALS`
# record live outside state and are excluded from the restore. Fails in the
# world where they are restored: test 2 then finds `_INSTALLED = False` over a
# still-patched socket, and the uninstall cannot put the real methods back.
import socket as _socket
_REAL_CONNECT = _socket.socket.connect
_UNDO = {}


def test_e1_a_test_installs_the_egress_guard_and_leaves_it():
    from ghost_agent.utils import egress_guard
    _UNDO["fn"] = egress_guard.install("socks5://127.0.0.1:9050")
    assert _socket.socket.connect is not _REAL_CONNECT


def test_e2_the_guard_state_still_matches_the_socket_and_can_be_undone():
    from ghost_agent.utils import egress_guard
    try:
        assert egress_guard.is_installed() is True          # it IS installed: the socket is patched
        assert egress_guard._ORIGINALS.get("connect") is _REAL_CONNECT
    finally:
        _UNDO.pop("fn", lambda: None)()
    assert _socket.socket.connect is _REAL_CONNECT and not egress_guard.is_installed()
