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


# ──────────────────────────────────────────────────────────────────────
# Inline image rendering (2026-07-17): replies referencing agent images
# (`![…](name.png)`) draw in the terminal after the reply settles —
# iTerm2/WezTerm escape, kitty graphics protocol, or half-block fallback.
# ──────────────────────────────────────────────────────────────────────


class TestImageRefs:
    def test_extracts_dedupes_and_filters(self):
        refs = cli._extract_image_refs(
            "Here ![a](gen_cat.png) and ![b](projects/x/plot.jpeg) "
            "again ![a](gen_cat.png) skip ![u](https://x.com/a.png) "
            "skip ![d](data:image/png;base64,xx) skip ![t](notes.txt)")
        assert refs == ["gen_cat.png", "projects/x/plot.jpeg"]

    def test_cap_and_empty(self):
        many = " ".join(f"![i](img{i}.png)" for i in range(9))
        assert len(cli._extract_image_refs(many)) == cli._IMG_MAX_PER_REPLY
        assert cli._extract_image_refs("") == []
        assert cli._extract_image_refs(None) == []

    def test_api_path_and_workspace_prefixes_normalized(self):
        """The live 2026-07-17 miss: the model embedded the FULL API path,
        the fetch built /api/download//api/download/… and 404'd."""
        refs = cli._extract_image_refs(
            "![cat](/api/download/gen_b404c1e9.png) "
            "![p](/workspace/projects/x/plot.png) ![s](sandbox:/shot.png)")
        assert refs == ["gen_b404c1e9.png", "projects/x/plot.png",
                        "shot.png"]

    def test_normalize_is_prefix_anchored(self):
        assert cli._normalize_image_ref("my/api/download/x.png") == \
            "my/api/download/x.png"  # mid-path is content, not a prefix
        assert cli._normalize_image_ref("/gen.png") == "gen.png"


class TestImageMode:
    def test_override_wins_even_without_tty(self, monkeypatch):
        monkeypatch.setenv("GHOST_CLI_IMAGES", "halfblock")
        assert cli._term_image_mode() == "halfblock"
        monkeypatch.setenv("GHOST_CLI_IMAGES", "off")
        assert cli._term_image_mode() == "none"

    @staticmethod
    def _scrub(monkeypatch):
        # Scrub the AMBIENT terminal identity — the suite may run inside
        # iTerm2 (LC_TERMINAL leaks through ssh/subprocesses) or tmux.
        for var in ("GHOST_CLI_IMAGES", "TERM_PROGRAM", "LC_TERMINAL",
                    "KITTY_WINDOW_ID", "TERM", "TMUX"):
            monkeypatch.delenv(var, raising=False)

    def test_iterm_and_kitty_detection(self, monkeypatch):
        self._scrub(monkeypatch)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        assert cli._term_image_mode() == "iterm"
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setenv("KITTY_WINDOW_ID", "1")
        assert cli._term_image_mode() == "kitty"

    def test_tmux_avoids_swallowed_escapes(self, monkeypatch):
        """Under tmux the escape protocols are silently eaten unless
        allow-passthrough is on (operator report, tmux-on-iTerm2): auto
        mode must pick something REDRAW-SAFE (sixel when the stack
        supports it, else half-block), and an explicit override must
        still win (it gets passthrough framing)."""
        self._scrub(monkeypatch)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.setenv("TMUX", "/private/tmp/tmux-501/default,1,0")
        monkeypatch.setattr(cli, "_tmux_supports_sixel", lambda: False)
        expected = "halfblock" if cli._pil() else "none"
        assert cli._term_image_mode() == expected
        monkeypatch.setenv("GHOST_CLI_IMAGES", "iterm")
        assert cli._term_image_mode() == "iterm"

    def test_tmux_prefers_sixel_when_stack_supports_it(self, monkeypatch):
        """Sixel survives tmux redraws (tmux OWNS the image — the
        operator's resize-erases-the-image report is the passthrough
        overlay limitation sixel exists to avoid)."""
        if not cli._pil():
            pytest.skip("Pillow not installed")
        self._scrub(monkeypatch)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("LC_TERMINAL", "iTerm2")
        monkeypatch.setenv("TMUX", "/private/tmp/tmux-501/default,1,0")
        monkeypatch.setattr(cli, "_tmux_supports_sixel", lambda: True)
        assert cli._term_image_mode() == "sixel"
        monkeypatch.setenv("GHOST_CLI_IMAGES", "sixel")
        assert cli._term_image_mode() == "sixel"

    def test_non_tty_defaults_to_none(self, monkeypatch):
        self._scrub(monkeypatch)
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
        assert cli._term_image_mode() == "none"


def _png_bytes():
    Image = cli._pil()
    if Image is None:
        pytest.skip("Pillow not installed")
    import io
    img = Image.new("RGB", (8, 6), (200, 30, 90))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TestRenderers:
    def test_iterm_escape_shape(self, capsys):
        data = _png_bytes()
        assert cli._render_iterm(data, "x.png")
        out = capsys.readouterr().out
        assert out.startswith("\x1b]1337;File=name=")
        assert f"size={len(data)}" in out and "inline=1" in out
        assert out.rstrip("\n").endswith("\x07")

    def test_kitty_chunks_and_png_only_guard(self, capsys, monkeypatch):
        assert cli._render_kitty(_png_bytes(), "x.png")
        out = capsys.readouterr().out
        assert "\x1b_Ga=T,f=100," in out and out.count("\x1b\\") >= 1
        # Non-PNG with Pillow unavailable → refuses rather than emitting
        # bytes kitty can't decode.
        monkeypatch.setattr(cli, "_pil", lambda: None)
        assert not cli._render_kitty(b"JFIFnotpng", "x.jpg")

    def test_halfblock_draws_and_bad_bytes_safe(self):
        assert cli.render_image_bytes("x.png", _png_bytes(),
                                      mode="halfblock")
        assert not cli.render_image_bytes("x.png", b"not an image",
                                          mode="halfblock")
        assert not cli.render_image_bytes("x.png", _png_bytes(), mode="none")


class TestReplyImageFlow:
    class _FakeAPI:
        def __init__(self, status=200, data=b""):
            self._resp = (status, data)
            self.asked = []

        def download_bytes(self, name, cap=None):
            self.asked.append(name)
            return self._resp

    def test_fetches_each_ref_and_renders(self, monkeypatch):
        api = self._FakeAPI(200, _png_bytes())
        monkeypatch.setenv("GHOST_CLI_IMAGES", "halfblock")
        cli.render_reply_images(api, "look: ![a](a.png) and ![b](b.png)")
        assert api.asked == ["a.png", "b.png"]

    def test_mode_none_never_fetches(self, monkeypatch):
        api = self._FakeAPI()
        monkeypatch.setenv("GHOST_CLI_IMAGES", "off")
        cli.render_reply_images(api, "![a](a.png)")
        assert api.asked == []

    def test_fetch_failure_is_a_notice_not_a_crash(self, monkeypatch):
        api = self._FakeAPI(404, b"")
        monkeypatch.setenv("GHOST_CLI_IMAGES", "halfblock")
        cli.render_reply_images(api, "![a](missing.png)")  # must not raise
        assert api.asked == ["missing.png"]

    def test_404_on_pathed_ref_retries_flat_basename(self, monkeypatch):
        """Reply paths don't always match the sandbox layout; generated
        images land at the flat root — try the basename before giving up."""
        class _PathyAPI:
            def __init__(self):
                self.asked = []

            def download_bytes(self, name, cap=None):
                self.asked.append(name)
                return (200, _png_bytes()) if name == "plot.png" else (404, b"")

        api = _PathyAPI()
        monkeypatch.setenv("GHOST_CLI_IMAGES", "halfblock")
        cli.render_reply_images(api, "![p](projects/x/plot.png)")
        assert api.asked == ["projects/x/plot.png", "plot.png"]


class TestSixel:
    def test_encoder_framing_and_rle(self, capsys):
        """8×6 solid image: one DCS header, ≤256 palette defs, a full-run
        RLE ('!8' + full-column char), one band, ST terminator."""
        Image = cli._pil()
        if Image is None:
            pytest.skip("Pillow not installed")
        import io
        buf = io.BytesIO()
        Image.new("RGB", (8, 6), (10, 200, 30)).save(buf, "PNG")
        assert cli._render_sixel(buf.getvalue(), "solid.png")
        out = capsys.readouterr().out
        assert out.startswith('\x1bPq"1;1;8;6')
        assert ";2;" in out                      # palette definition present
        assert "!8~" in out                      # 8-wide full-height run
        assert out.rstrip("\n").endswith("\x1b\\")

    def test_sixel_never_gets_passthrough_wrapping(self, monkeypatch, capsys):
        """A sixel-built tmux must SEE the sequence to own the image —
        passthrough framing would hide it and reintroduce the vanishing
        overlay problem."""
        Image = cli._pil()
        if Image is None:
            pytest.skip("Pillow not installed")
        import io
        monkeypatch.setenv("TMUX", "/private/tmp/tmux-501/default,1,0")
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, "PNG")
        assert cli._render_sixel(buf.getvalue(), "x.png")
        out = capsys.readouterr().out
        assert out.startswith("\x1bPq")          # raw DCS, no tmux; prefix
        assert "\x1bPtmux;" not in out

    def test_bad_bytes_safe(self):
        assert not cli._render_sixel(b"not an image", "x.png")


class TestTmuxPassthrough:
    def test_emit_raw_wraps_escapes_under_tmux(self, monkeypatch, capsys):
        monkeypatch.setenv("TMUX", "/private/tmp/tmux-501/default,1,0")
        cli._emit_raw("\x1b]1337;File=x:AAAA\x07")
        out = capsys.readouterr().out
        assert out.startswith("\x1bPtmux;")
        assert out.endswith("\x1b\\")
        assert "\x1b\x1b]1337;" in out  # inner ESC doubled

    def test_no_wrapping_outside_tmux(self, monkeypatch, capsys):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        cli._emit_raw("\x1b]1337;File=x:AAAA\x07")
        assert capsys.readouterr().out == "\x1b]1337;File=x:AAAA\x07"


class TestSixelTmuxGeometry:
    def test_cursor_advances_past_image_under_tmux(self, monkeypatch, capsys):
        """tmux anchors a sixel image at the cursor WITHOUT advancing it;
        printing the caption next overwrote the image's cells and tmux
        invalidated it — visible for a frame at best (live report #3).
        The emitter must advance the cursor past ceil(H/cell_h) rows."""
        Image = cli._pil()
        if Image is None:
            pytest.skip("Pillow not installed")
        import io
        monkeypatch.setattr(cli, "_in_tmux", lambda: True)
        monkeypatch.setattr(cli, "_tmux_cell_px", lambda: (8, 18))
        buf = io.BytesIO()
        Image.new("RGB", (40, 36), (5, 5, 5)).save(buf, "PNG")
        assert cli._render_sixel(buf.getvalue(), "x.png")
        out = capsys.readouterr().out
        trailing = len(out) - len(out.rstrip("\n"))
        assert trailing == -(-36 // 18) + 1      # ceil(36/18)+1 = 3

    def test_width_scales_by_real_cell_width_under_tmux(self, monkeypatch,
                                                        capsys):
        Image = cli._pil()
        if Image is None:
            pytest.skip("Pillow not installed")
        import io
        monkeypatch.setattr(cli, "_in_tmux", lambda: True)
        monkeypatch.setattr(cli, "_tmux_cell_px", lambda: (8, 18))
        monkeypatch.setattr(cli, "_image_cells", lambda: 50)
        buf = io.BytesIO()
        Image.new("RGB", (2000, 100), (9, 9, 9)).save(buf, "PNG")
        assert cli._render_sixel(buf.getvalue(), "wide.png")
        out = capsys.readouterr().out
        # Raster attributes carry the scaled width: 50 cells × 8px = 400.
        assert '"1;1;400;' in out
