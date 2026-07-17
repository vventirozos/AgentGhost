"""Ghost CLI client (interface/externals/cli/ghost).

Moved into the repo 2026-07-17 (was ~/Data/AI/bin/ghost, now a symlink
there — ~/Data/AI/bin is on PATH). The script is a PEP-723 single file
with no extension, so tests load it by path. Coverage: import-cleanness
on a non-tty (PromptSession is deliberately lazy), the pure formatting
helpers, error-shape extraction, and base-URL normalization — the parts
that break silently when refactored. Network paths are not exercised.
"""

import importlib.machinery
import importlib.util
import os
import stat
import time
from pathlib import Path

import pytest

CLI_PATH = (Path(__file__).resolve().parents[1]
            / "interface" / "externals" / "cli" / "ghost")


def _load():
    loader = importlib.machinery.SourceFileLoader("ghost_cli", str(CLI_PATH))
    spec = importlib.util.spec_from_file_location(
        "ghost_cli", CLI_PATH, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


cli = _load()


class TestLocation:
    def test_lives_in_repo_and_is_executable(self):
        assert CLI_PATH.is_file()
        assert CLI_PATH.stat().st_mode & stat.S_IXUSR

    def test_bin_symlink_points_into_repo(self):
        """Deploy contract on the operator's machine: ~/Data/AI/bin (on
        PATH) symlinks to the repo copy, so `ghost` keeps working and
        edits land in one place. Skipped on machines without that dir."""
        bin_ghost = Path.home() / "Data" / "AI" / "bin" / "ghost"
        if not bin_ghost.parent.is_dir():
            pytest.skip("no ~/Data/AI/bin on this machine")
        assert bin_ghost.is_symlink()
        assert bin_ghost.resolve() == CLI_PATH.resolve()


class TestHelpers:
    def test_trim(self):
        assert cli._trim("a  b\n c", 100) == "a b c"
        assert cli._trim("x" * 20, 10) == "x" * 9 + "…"
        assert cli._trim(None, 10) == ""

    def test_ago_buckets(self):
        now = time.time()
        assert cli._ago(None) == "—"
        assert cli._ago(now - 30).endswith("s ago")
        assert cli._ago(now - 600).endswith("m ago")
        assert cli._ago(now - 7200).endswith("h ago")
        assert cli._ago(now - 3 * 86400).endswith("d ago")

    def test_dur_buckets(self):
        assert cli._dur(None) == "—"
        assert cli._dur(45) == "45s"
        assert cli._dur(150) == "2m 30s"
        assert cli._dur(7260) == "2h 1m"
        assert cli._dur(2 * 86400 + 3600) == "2d 1h"


class TestErrorOf:
    def test_all_error_shapes(self):
        eo = cli.GhostAPI.error_of
        assert eo({"error": {"message": "boom"}}) == "boom"
        assert eo({"error": "flat"}) == "flat"
        assert eo({"detail": "denied"}) == "denied"
        assert eo({"raw": "<html>502</html>"}) == "<html>502</html>"
        assert eo("plain") == "plain"


class TestBaseUrl:
    def test_stray_path_and_query_stripped(self):
        api = cli.GhostAPI("http://eva:8000/some/path?x=1", "k")
        assert api.base_url == "http://eva:8000"
        assert api._url("/api/health") == "http://eva:8000/api/health"

    def test_trailing_slash_stripped(self):
        assert cli.GhostAPI("http://eva:8000/", "k").base_url == "http://eva:8000"

    def test_schemeless_falls_back_to_rstrip(self):
        assert cli.GhostAPI("eva:8000", "k").base_url == "eva:8000"

    def test_key_rides_header(self):
        api = cli.GhostAPI("http://eva:8000", "sekrit")
        assert api.http.headers["X-Ghost-Key"] == "sekrit"


class TestDefaultKey:
    def test_env_wins_and_blank_env_falls_through(self, monkeypatch):
        monkeypatch.setenv("GHOST_API_KEY", "from-env")
        assert cli._default_key() == "from-env"
        # Blank env must not shadow the key file (the " " vs "" Slack-bot
        # incident class): result is whatever the file path yields, never
        # a whitespace string.
        monkeypatch.setenv("GHOST_API_KEY", "   ")
        assert cli._default_key().strip() == cli._default_key()
