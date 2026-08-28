"""Ghost CLI client (interface/externals/cli/ghost).

Moved into the repo 2026-07-17 (was ~/Data/AI/bin/ghost, now a symlink
there — ~/Data/AI/bin is on PATH). The script is a PEP-723 single file
with no extension, so tests load it by path. Coverage: import-cleanness
on a non-tty (PromptSession is deliberately lazy), the pure formatting
helpers, error-shape extraction, base-URL normalization, the inline-image
stack, the stream-render path (tail-cropped transient Live + settled
full print — the half-screen live-repaint regression, 2026-07-26) against
a faked SSE stream, and the 👍/👎 rating path (human outcome labels →
/api/feedback, 2026-08-27) against a recording fake. Real network paths
are not exercised.
"""

import importlib.machinery
import importlib.util
import io
import json
import os
import re
import stat
import time
from pathlib import Path

import pytest

REPO_CLI = (Path(__file__).resolve().parents[1]
            / "interface" / "externals" / "cli" / "ghost")

# GHOST_CLI_PATH points the suite at a COPY of the client.
#
# This exists for mutation runs. They used to write each broken variant into
# the repo file itself and restore it a couple of seconds later — but
# ~/Data/AI/bin/ghost (on PATH) is a symlink to that exact file, so every
# mutant was briefly the operator's LIVE client. Launching `ghost` inside one
# of those windows loaded the broken build: a mutant that disables the rating
# branch made /good and /+1 answer "unknown or incomplete command" (reported
# 2026-08-27). A test harness must not be able to hand the operator a broken
# tool; mutants go to a temp copy and the deployed file is never written.
_CLI_OVERRIDE = os.environ.get("GHOST_CLI_PATH")
CLI_PATH = Path(_CLI_OVERRIDE) if _CLI_OVERRIDE else REPO_CLI


def _load():
    loader = importlib.machinery.SourceFileLoader("ghost_cli", str(CLI_PATH))
    spec = importlib.util.spec_from_file_location(
        "ghost_cli", CLI_PATH, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


cli = _load()

# The CLI declares a theme on its module console. A test that swaps in a
# capturing Console must carry it, or every styled print raises
# MissingStyle — and the resulting failure looks like a bug in the code
# under test rather than in the harness. Asserted, not defaulted: if rich's
# theme-stack internals move, fail loudly here instead of silently
# rendering unstyled output that quietly changes what the assertions see.
_THEME = dict(cli.console._theme_stack._entries[0])
assert {"you", "ghost", "notice", "err", "ok", "mute"} <= set(_THEME), \
    "rich theme internals moved — tests need the CLI's own theme to render"


class TestLocation:
    """Deploy contract — asserted against the REPO file, never against a
    GHOST_CLI_PATH copy (a mutation run must not be able to 'fail' these by
    pointing the suite at a temp file)."""

    def test_lives_in_repo_and_is_executable(self):
        assert REPO_CLI.is_file()
        assert REPO_CLI.stat().st_mode & stat.S_IXUSR

    def test_bin_symlink_points_into_repo(self):
        """~/Data/AI/bin (on PATH) symlinks to the repo copy, so `ghost`
        keeps working and edits land in one place. Skipped on machines
        without that dir."""
        bin_ghost = Path.home() / "Data" / "AI" / "bin" / "ghost"
        if not bin_ghost.parent.is_dir():
            pytest.skip("no ~/Data/AI/bin on this machine")
        assert bin_ghost.is_symlink()
        assert bin_ghost.resolve() == REPO_CLI.resolve()

    def test_the_deployed_client_dispatches_its_rating_commands(self):
        """Smoke the file the symlink actually resolves to. A mutation run
        that leaked a broken build into the repo would show up HERE, on the
        deployed path, rather than in the operator's next session."""
        loader = importlib.machinery.SourceFileLoader(
            "ghost_cli_deployed", str(REPO_CLI))
        spec = importlib.util.spec_from_file_location(
            "ghost_cli_deployed", REPO_CLI, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        assert set(mod.RATE_CMDS) == {"/good", "/+1", "/bad", "/-1"}
        assert mod.RATE_CMDS["/good"] == "positive"
        assert mod.RATE_CMDS["/-1"] == "negative"


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

    def test_schemeless_gets_http(self):
        """GHOST_URL=eva:8000 parses as scheme='eva', netloc='' — it used to
        pass straight through, and every call then died with a bare
        requests MissingSchema instead of just working."""
        api = cli.GhostAPI("eva:8000", "k")
        assert api.base_url == "http://eva:8000"
        assert api._url("/api/health") == "http://eva:8000/api/health"

    def test_schemeless_host_only(self):
        assert cli.GhostAPI("eva", "k").base_url == "http://eva"

    def test_https_is_not_downgraded(self):
        assert cli.GhostAPI("https://eva:8443/x", "k").base_url \
            == "https://eva:8443"

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
    def test_iterm_escape_shape(self, capsys, monkeypatch):
        # Pin a non-tmux environment (like the sixel/passthrough siblings):
        # under TERM=tmux-256color _emit_raw legitimately wraps the escape in
        # a tmux passthrough and the raw-shape assertion below would fail on
        # the runner's terminal, not on the code (seen 2026-07-25).
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
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


class TestTailCrop:
    """A Live region taller than the terminal can't be repainted (the
    cursor can't move above the top row) — each refresh then duplicates
    partial frames into scrollback: the operator-reported "erratic /
    half-screen" transcripts. _TailCrop bounds the live view to the
    viewport tail so the region stays repaintable at any reply length."""

    def _render(self, renderable, height):
        buf = io.StringIO()
        con = cli.Console(file=buf, width=20, height=height,
                          color_system=None)
        con.print(renderable)
        return buf.getvalue().splitlines()

    def test_short_content_passes_through(self):
        from rich.text import Text
        lines = self._render(
            cli._TailCrop(Text("\n".join(f"L{i}" for i in range(5)))), 20)
        assert lines == [f"L{i}" for i in range(5)]

    def test_tall_content_crops_to_viewport_tail(self):
        """30 lines into a 10-row console → the LAST height-2 lines (rich's
        own crop/ellipsis modes keep the TOP — wrong end for chat)."""
        from rich.text import Text
        lines = self._render(
            cli._TailCrop(Text("\n".join(f"L{i}" for i in range(30)))), 10)
        assert lines == [f"L{i}" for i in range(22, 30)]


class _FakeSSE:
    """Minimal stand-in for a streaming requests.Response."""
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(self._lines)


class TestStreamRender:
    def _cli_with_stream(self, monkeypatch, sse_lines):
        """GhostCLI wired to a fake SSE stream, console swapped for a
        capturing terminal-mode Console, Live spied for kwargs/updates."""
        captured = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=captured, force_terminal=True, width=60, height=24))
        seen = {"kwargs": None, "updates": []}
        real_live = cli.Live

        class SpyLive(real_live):
            def __init__(self, *a, **kw):
                seen["kwargs"] = kw
                super().__init__(*a, **kw)

            def update(self, renderable, **kw):
                seen["updates"].append(renderable)
                super().update(renderable, **kw)

        monkeypatch.setattr(cli, "Live", SpyLive)
        api = cli.GhostAPI("http://localhost:9", "k")
        monkeypatch.setattr(api, "chat_stream",
                            lambda *a, **kw: _FakeSSE(sse_lines))
        return cli.GhostCLI(api), captured, seen

    def test_transient_tailcrop_live_and_settled_full_print(self, monkeypatch):
        g, captured, seen = self._cli_with_stream(monkeypatch, [
            b'data: {"choices":[{"delta":{"content":"hello "}}]}',
            b'data: {"choices":[{"delta":{"content":"world"}}]}',
            b"data: [DONE]",
        ])
        assert g.stream_reply("req1") == "hello world"
        # Regression guard: vertical_overflow="visible" is what shredded
        # long-reply transcripts; the live view must be transient + cropped.
        assert seen["kwargs"].get("transient") is True
        assert "vertical_overflow" not in seen["kwargs"]
        assert seen["updates"]
        assert all(isinstance(u, cli._TailCrop) for u in seen["updates"])
        # The settled reply is printed full-height after the live view.
        assert "hello world" in captured.getvalue()

    def test_partial_reply_printed_on_midstream_error(self, monkeypatch):
        """The transient live erases itself — without the settled print in
        _settle(), an abort would eat the partial reply entirely."""
        g, captured, seen = self._cli_with_stream(monkeypatch, [
            b'data: {"choices":[{"delta":{"content":"partial text"}}]}',
            b'data: {"error": {"message": "boom mid-stream"}}',
        ])
        assert g.stream_reply("req2") == "partial text"
        out = captured.getvalue()
        assert "partial text" in out
        assert "boom mid-stream" in out          # notice AFTER the reply


class TestRating:
    """👍/👎 → the agent's /api/feedback (human outcome labels, Track-1a).

    These labels resolve outcome=unknown, which every measurement clock in
    the learning stack is gated on — and they are scarce. So what is pinned
    here is (a) that a real label is never lost to the post-stream
    trajectory-write race, (b) that a lost one is never reported as
    recorded, and (c) that JUNK labels — a turn the agent never finished, a
    turn from a conversation the operator has left — cannot be written at
    all.
    """

    OK = (200, {"ok": True, "outcome": "passed"})

    def _make(self, monkeypatch, responses=None):
        """CLI with a capturing console and a recording /api/feedback.

        ``responses`` is consumed one per call; the last one repeats."""
        captured = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=captured, force_terminal=True, width=70, height=24))
        monkeypatch.setattr(cli, "render_reply_images", lambda *a, **k: None)
        monkeypatch.setattr(time, "sleep", lambda *_: None)   # no real backoff
        monkeypatch.setattr(cli.sys, "stdin", io.StringIO())  # not a tty
        api = cli.GhostAPI("http://localhost:9", "k")
        monkeypatch.setattr(api, "cancel_turn",
                            lambda **kw: (200, {"cancelled": True}))
        sent = []
        queue = list(responses or [self.OK])

        def _feedback(request_id, signal, note="", source=cli.FEEDBACK_SOURCE):
            sent.append({"request_id": request_id, "signal": signal,
                         "note": note, "source": source})
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(api, "feedback", _feedback)
        return cli.GhostCLI(api), api, sent, captured

    def _turn(self, monkeypatch, g, api, text="hello", prompt="q"):
        """One real chat_turn against a faked SSE stream — so the pinned
        request_id is the one chat_turn actually put on the wire."""
        lines = [('data: {"choices":[{"delta":{"content":"%s"}}]}'
                  % text).encode(), b"data: [DONE]"]
        monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(lines))
        g.chat_turn(prompt)

    def _abort_turn(self, monkeypatch, g, api):
        """A turn that streams a partial reply and is then Ctrl-C'd."""
        def _lines():
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}'
            raise KeyboardInterrupt
        monkeypatch.setattr(api, "chat_stream",
                            lambda *a, **kw: _FakeSSE(_lines()))
        g.chat_turn("q")

    # -- wire contract ----------------------------------------------------
    def test_wire_body_shape_and_note_cap(self, monkeypatch):
        api = cli.GhostAPI("http://localhost:9", "k")
        seen = {}

        def _json(method, path, **kw):
            seen.update({"method": method, "path": path}, **kw)
            return 200, {"ok": True}

        monkeypatch.setattr(api, "_json", _json)
        api.feedback("chatcmpl-abc", "negative", "x" * 900)
        assert (seen["method"], seen["path"]) == ("POST", "/api/feedback")
        body = seen["body"]
        assert body["request_id"] == "chatcmpl-abc"
        assert body["signal"] == "negative"
        # `source` is how the corpus tells operator-CLI labels apart from
        # Slack reactions and web taps — a wrong/absent one is unrecoverable
        # attribution loss, not cosmetics.
        assert body["source"] == "cli"
        assert len(body["note"]) == 500

    # -- the happy paths --------------------------------------------------
    def test_good_labels_the_request_id_chat_turn_sent(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        req_id = g.turns[-1]["req_id"]
        assert g.dispatch("/good") is True
        assert len(sent) == 1
        assert sent[0]["request_id"] == req_id
        assert sent[0]["signal"] == "positive"
        assert g.turns[-1]["label"] == "positive"
        assert "recorded" in out.getvalue()

    def test_aliases_map_to_signals(self, monkeypatch):
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        g.dispatch("/+1")
        g.dispatch("/-1 wrong")
        assert [s["signal"] for s in sent] == ["positive", "negative"]

    def test_bad_free_text_reason_is_not_shlex_parsed(self, monkeypatch):
        """The reason is the payload — a negative label's note becomes the
        trajectory's failure_reason. Routing /bad through the shlex.split
        the other commands use loses every reason containing an apostrophe
        ("No closing quotation") — silently, since dispatch catches it and
        posts nothing."""
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        reason = 'it didn\'t find the file "notes.md"'
        g.dispatch(f"/bad {reason}")
        assert len(sent) == 1
        assert sent[0]["note"] == reason
        assert sent[0]["signal"] == "negative"

    def test_caret_n_rates_an_earlier_reply(self, monkeypatch):
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api, text="first", prompt="q1")
        first = g.turns[-1]["req_id"]
        self._turn(monkeypatch, g, api, text="second", prompt="q2")
        second = g.turns[-1]["req_id"]
        assert first != second
        g.dispatch("/bad ^2 stale answer")
        assert sent[0]["request_id"] == first
        assert sent[0]["note"] == "stale answer"

    def test_unchanged_is_reported_as_such(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        g.dispatch("/good")
        g.dispatch("/good")
        assert len(sent) == 2                     # a repeat also HEALS
        assert "already labeled" not in out.getvalue()
        g2, api2, sent2, out2 = self._make(
            monkeypatch,
            responses=[(200, {"ok": True, "unchanged": True,
                              "outcome": "passed"})])
        self._turn(monkeypatch, g2, api2)
        g2.dispatch("/good")
        assert "already labeled" in out2.getvalue()

    # -- never lose a label, never fake one -------------------------------
    def test_404_is_retried_then_recorded(self, monkeypatch):
        """The agent writes the trajectory AFTER the stream closes, so a
        thumb typed the instant a reply lands 404s. Without the retry that
        label is simply gone."""
        g, api, sent, out = self._make(monkeypatch, responses=[
            (404, {"ok": False, "code": "not_found", "error": "no trajectory"}),
            self.OK,
        ])
        self._turn(monkeypatch, g, api)
        g.dispatch("/good")
        assert len(sent) == 2
        assert "recorded" in out.getvalue()
        assert g.turns[-1]["label"] == "positive"

    def test_5xx_is_retried(self, monkeypatch):
        """503 = the agent restarting, i.e. the window a deploy creates."""
        g, api, sent, out = self._make(monkeypatch, responses=[
            (503, {"ok": False, "code": "unavailable"}), self.OK])
        self._turn(monkeypatch, g, api)
        g.dispatch("/good")
        assert len(sent) == 2
        assert "recorded" in out.getvalue()

    def test_permanent_404_never_claims_success(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch, responses=[
            (404, {"ok": False, "code": "not_found", "error": "no trajectory"})])
        self._turn(monkeypatch, g, api)
        g.dispatch("/good")
        assert len(sent) == 1 + len(cli.RATE_RETRY_DELAYS)
        text = out.getvalue()
        assert "NOT recorded" in text
        assert "✓" not in text
        # A surviving 404 is not just "an error" — it means no trajectory
        # matched, i.e. trajectory logging is probably off and EVERY label
        # from this box is going nowhere. Say that, not a bare HTTP code.
        assert "no trajectory matched" in text
        assert g.turns[-1]["label"] is None       # no false latch

    def test_transport_failure_never_claims_success(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)

        def _boom(*a, **kw):
            raise cli.requests.ConnectionError("down")

        monkeypatch.setattr(api, "feedback", _boom)
        self._turn(monkeypatch, g, api)
        g.dispatch("/good")
        assert "NOT recorded" in out.getvalue()
        assert g.turns[-1]["label"] is None

    # -- label-noise guards ------------------------------------------------
    def test_cancelled_turn_is_not_rateable(self, monkeypatch):
        """A Ctrl-C'd turn is truncated by the HUMAN. Labeling it FAILED
        teaches the verifier that an operator's abort was a model failure;
        labeling it PASSED teaches it that stopping early is fine. Both are
        noise in a corpus where labels are scarce enough to be load-bearing,
        so neither thumb may reach the wire."""
        g, api, sent, out = self._make(monkeypatch)
        self._abort_turn(monkeypatch, g, api)
        assert g.turns[-1]["interrupted"] == "cancelled"
        g.dispatch("/bad")
        g.dispatch("/good")
        assert sent == []
        assert "not rateable" in out.getvalue()

    def test_midstream_error_turn_is_not_rateable(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)
        lines = [b'data: {"choices":[{"delta":{"content":"partial"}}]}',
                 b'data: {"error": {"message": "boom"}}']
        monkeypatch.setattr(api, "chat_stream",
                            lambda *a, **kw: _FakeSSE(lines))
        g.chat_turn("q")
        assert g.turns[-1]["interrupted"] == "error"
        g.dispatch("/bad it broke")
        assert sent == []

    def test_finished_turn_after_an_aborted_one_is_rateable(self, monkeypatch):
        """The other half of the identity: the guard must not latch on and
        make every later turn unrateable."""
        g, api, sent, _ = self._make(monkeypatch)
        self._abort_turn(monkeypatch, g, api)
        self._turn(monkeypatch, g, api, text="clean")
        assert g.turns[-1]["interrupted"] == ""
        g.dispatch("/good")
        assert len(sent) == 1

    def test_clear_drops_the_turn_refs(self, monkeypatch):
        """/clear starts a new session — a /good after it would label a
        turn from the conversation the operator just left."""
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        g.dispatch("/clear")
        assert g.turns == []
        g.dispatch("/good")
        assert sent == []
        assert "nothing to rate yet" in out.getvalue()

    def test_resume_drops_the_turn_refs(self, monkeypatch):
        """A stored history carries no request ids at all."""
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        monkeypatch.setattr(api, "session_get", lambda sid: (200, {
            "messages": [{"role": "user", "content": "old"}], "title": "t"}))
        g.dispatch("/resume abcd1234")
        assert g.turns == []
        g.dispatch("/good")
        assert sent == []

    def test_rating_with_no_turns_posts_nothing(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)
        g.dispatch("/good")
        g.dispatch("/bad ^3 nope")
        assert sent == []

    def test_caret_beyond_history_posts_nothing(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        g.dispatch("/bad ^2 too far")
        assert sent == []
        assert "only 1 rateable reply" in out.getvalue()

    def test_malformed_caret_posts_nothing(self, monkeypatch):
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        g.dispatch("/bad ^x oops")
        assert sent == []

    def test_turn_refs_are_bounded(self, monkeypatch):
        g, api, sent, _ = self._make(monkeypatch)
        for _ in range(cli.RATE_HISTORY + 5):
            self._turn(monkeypatch, g, api)
        assert len(g.turns) == cli.RATE_HISTORY

    # -- the reason prompt -------------------------------------------------
    def test_bare_bad_prompts_for_a_reason_on_a_tty(self, monkeypatch):
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        tty = io.StringIO()
        tty.isatty = lambda: True
        monkeypatch.setattr(cli.sys, "stdin", tty)
        monkeypatch.setattr(cli.console, "input", lambda *a, **k: "  why not  ")
        g.dispatch("/bad")
        assert sent[0]["note"] == "why not"

    def test_bare_bad_does_not_prompt_off_a_tty(self, monkeypatch):
        """`_make` installs a non-tty stdin — a scripted dispatch must
        still record the (bare) label instead of blocking on input."""
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)

        def _no(*a, **k):
            raise AssertionError("prompted for a reason off a tty")

        monkeypatch.setattr(cli.console, "input", _no)
        g.dispatch("/bad")
        assert sent[0]["note"] == ""
        assert sent[0]["signal"] == "negative"

    def test_ctrl_c_at_the_reason_prompt_cancels_the_rating(self, monkeypatch):
        """Enter already means "skip the reason". Ctrl-C means cancel — it
        used to fall in with EOFError and record a bare 👎 anyway."""
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        tty = io.StringIO()
        tty.isatty = lambda: True
        monkeypatch.setattr(cli.sys, "stdin", tty)
        monkeypatch.setattr(cli.console, "input",
                            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
        g.dispatch("/bad")
        assert sent == []
        assert "cancelled" in out.getvalue()
        assert g.turns[-1]["label"] is None

    def test_ctrl_d_at_the_reason_prompt_still_skips_and_records(self, monkeypatch):
        """The other half of the identity — Ctrl-D is end-of-input, i.e.
        "no reason", and the label must still land."""
        g, api, sent, _ = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        tty = io.StringIO()
        tty.isatty = lambda: True
        monkeypatch.setattr(cli.sys, "stdin", tty)
        monkeypatch.setattr(cli.console, "input",
                            lambda *a, **k: (_ for _ in ()).throw(EOFError))
        g.dispatch("/bad")
        assert [s["signal"] for s in sent] == ["negative"]
        assert sent[0]["note"] == ""

    def test_connection_refused_is_retried_on_the_full_schedule(self, monkeypatch):
        """THE shape this retry exists for. A restarting agent refuses the
        connection — requests.ConnectionError, not HTTP 503 — so retrying
        only the HTTP shapes covered the less likely manifestation of the
        loop's own motivating scenario and dropped the label on the more
        likely one (1 attempt vs 4)."""
        g, api, _, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        n = {"c": 0}

        def _refused(*a, **k):
            n["c"] += 1
            raise cli.requests.ConnectionError("[Errno 61] Connection refused")

        monkeypatch.setattr(api, "feedback", _refused)
        g.dispatch("/good")
        assert n["c"] == 1 + len(cli.RATE_RETRY_DELAYS)
        assert "NOT recorded" in out.getvalue()
        assert g.turns[-1]["label"] is None

    def test_label_lands_when_the_agent_comes_back_mid_retry(self, monkeypatch):
        """The point of retrying: the label must actually be RECOVERED, not
        merely reported as lost more politely."""
        g, api, _, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api)
        n = {"c": 0}

        def _flaky(*a, **k):
            n["c"] += 1
            if n["c"] <= 2:
                raise cli.requests.ConnectionError("refused")
            return 200, {"ok": True, "outcome": "passed"}

        monkeypatch.setattr(api, "feedback", _flaky)
        g.dispatch("/good")
        assert g.turns[-1]["label"] == "positive"
        assert "recorded" in out.getvalue()

    def test_confirmation_names_the_turn_that_was_rated(self, monkeypatch):
        """With ^N in play, "which one did that hit?" is a real question."""
        g, api, sent, out = self._make(monkeypatch)
        self._turn(monkeypatch, g, api, text="a", prompt="what is entropy")
        self._turn(monkeypatch, g, api, text="b", prompt="and the second one")
        g.dispatch("/good ^2")
        assert "what is entropy" in out.getvalue()
        assert "and the second one" not in out.getvalue()

    def test_feedback_carries_its_own_short_timeout(self, monkeypatch):
        """The retry MULTIPLIES the timeout: on _json's (10, 60) default the
        four attempts could freeze the prompt for ~287s against a hung
        agent. A label is a tiny POST."""
        api = cli.GhostAPI("http://localhost:9", "k")
        seen = {}

        def _json(method, path, **kw):
            seen.update(kw)
            return 200, {"ok": True}

        monkeypatch.setattr(api, "_json", _json)
        api.feedback("abc", "positive")
        connect, read = seen["timeout"]
        worst = (1 + len(cli.RATE_RETRY_DELAYS)) * (connect + read) \
            + sum(cli.RATE_RETRY_DELAYS)
        assert worst < 90, f"a stuck agent would freeze the prompt for {worst}s"


# Control sequences are built from chr() so this file contains no literal
# control characters (they survive nothing — editors, diffs, review tools).
ESC = chr(27)
ST = ESC + chr(92)          # string terminator
BEL = chr(7)
C1_CSI = chr(0x9b)          # one-byte CSI — stripping only ESC leaves this
OSC52 = ESC + "]52;c;cm0gLXJmIH4=" + ST     # writes the viewer's CLIPBOARD


class _Resp:
    """Minimal streaming requests.Response stand-in for the file writers."""
    status_code = 200

    def __init__(self, chunks, fail_after=None):
        self._chunks, self._fail_after = chunks, fail_after

    def __enter__(self): return self
    def __exit__(self, *exc): return False

    def iter_content(self, _n):
        for i, c in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise cli.requests.ConnectionError("reset mid-stream")
            yield c


class TestFileSafety:
    """/download and /save must never destroy a local file they did not
    write. Both failure branches called dest.unlink(missing_ok=True), but
    the writer only opens dest on HTTP 200 — so the "cleanup" deleted
    whatever the operator happened to have under that name."""

    def _cli(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "console", cli.Console(
            file=io.StringIO(), width=70,
            theme=cli.Theme(_THEME), highlight=False))
        api = cli.GhostAPI("http://localhost:9", "k")
        return cli.GhostCLI(api), api

    def test_failed_download_leaves_an_existing_file_alone(self, monkeypatch, tmp_path):
        g, api = self._cli(monkeypatch, tmp_path)
        victim = tmp_path / "report.pdf"
        victim.write_text("MY IRREPLACEABLE LOCAL FILE")
        monkeypatch.setattr(api, "download", lambda n, d: (404, 0))
        g.cmd_download("report.pdf", "")
        assert victim.read_text() == "MY IRREPLACEABLE LOCAL FILE"

    def test_failed_save_leaves_an_existing_file_alone(self, monkeypatch, tmp_path):
        g, api = self._cli(monkeypatch, tmp_path)
        victim = tmp_path / "backup.zip"
        victim.write_text("MY EXISTING BACKUP")
        monkeypatch.setattr(api, "workspace_save", lambda h, d: (500, 0))
        g.cmd_save("backup.zip")
        assert victim.read_text() == "MY EXISTING BACKUP"

    def test_midstream_failure_does_not_truncate_the_original(self, tmp_path):
        """The other half: a transfer that dies after opening dest used to
        leave a truncated file over the operator's original."""
        dest = tmp_path / "keep.bin"
        dest.write_text("ORIGINAL CONTENT")
        with pytest.raises(cli.requests.ConnectionError):
            cli.GhostAPI._stream_to_file(
                _Resp([b"partial", b"more"], fail_after=1), dest)
        assert dest.read_text() == "ORIGINAL CONTENT"

    def test_midstream_failure_leaves_no_temp_file(self, tmp_path):
        dest = tmp_path / "keep.bin"
        dest.write_text("ORIGINAL")
        with pytest.raises(cli.requests.ConnectionError):
            cli.GhostAPI._stream_to_file(_Resp([b"x"], fail_after=0), dest)
        assert [p.name for p in tmp_path.iterdir()] == ["keep.bin"]

    def test_interrupt_midwrite_also_cleans_up(self, tmp_path):
        """Ctrl-C is a BaseException — the cleanup must catch it too, or an
        interrupted download leaves a temp file AND a lost original."""
        dest = tmp_path / "keep.bin"
        dest.write_text("ORIGINAL")

        class _Interrupting(_Resp):
            def iter_content(self, _n):
                yield b"partial"
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            cli.GhostAPI._stream_to_file(_Interrupting([]), dest)
        assert dest.read_text() == "ORIGINAL"
        assert [p.name for p in tmp_path.iterdir()] == ["keep.bin"]

    def test_success_still_writes_and_replaces(self, tmp_path):
        dest = tmp_path / "keep.bin"
        dest.write_text("OLD")
        n = cli.GhostAPI._stream_to_file(_Resp([b"NEW ", b"BYTES"]), dest)
        assert (n, dest.read_bytes()) == (9, b"NEW BYTES")


class TestControlCharSanitizing:
    """rich strips only [7, 8, 11, 12, 13] — ESC is not in that set, so CSI
    and ST-terminated OSC in agent/server text reached the terminal RAW.
    This is the only client that renders into a terminal instead of HTML,
    and the agent summarizes Tor-fetched pages."""

    PAYLOADS = [ESC, ESC + "[2J", ESC + "[?1049h", OSC52,
                ESC + "]0;title" + BEL, C1_CSI + "2J"]

    def _cap(self, monkeypatch, fn):
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, force_terminal=True, width=70,
            theme=cli.Theme(_THEME), highlight=False))
        fn()
        return buf.getvalue()

    def test_safe_strips_escapes_and_keeps_text(self):
        assert cli._safe("a" + ESC + "[2Jb") == "a[2Jb"
        assert cli._safe("a" + C1_CSI + "b") == "ab"
        assert cli._safe("a" + BEL + "b") == "ab"
        assert cli._safe("keep\nnewlines\tand tabs") == "keep\nnewlines\tand tabs"
        assert cli._safe(None) == "" and cli._safe(7) == "7"
        assert cli._safe("héllo ✓ 👻") == "héllo ✓ 👻"

    @staticmethod
    def _stream(body):
        def _go():
            pr = cli._StreamPrinter()
            pr.feed(body)
            pr.settle()
        return _go

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_reply_body_is_stripped(self, monkeypatch, payload):
        # Compare against the SAME render without the payload: rich emits
        # its own SGR escapes, so "does the output contain an ESC" is not
        # the question — "does it contain MORE than it should" is.
        out = self._cap(monkeypatch, self._stream("x" + payload + "y\n\n"))
        clean = self._cap(monkeypatch, self._stream("xy\n\n"))
        assert out.count(ESC) == clean.count(ESC)

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_notice_and_table_are_stripped(self, monkeypatch, payload):
        n = self._cap(monkeypatch, lambda: cli.GhostCLI.notice("x" + payload + "y"))
        n0 = self._cap(monkeypatch, lambda: cli.GhostCLI.notice("xy"))
        t = self._cap(monkeypatch, lambda: cli._table(["a"], [["x" + payload + "y"]]))
        t0 = self._cap(monkeypatch, lambda: cli._table(["a"], [["xy"]]))
        assert n.count(ESC) == n0.count(ESC)
        assert t.count(ESC) == t0.count(ESC)

    def test_server_supplied_fields_are_stripped(self, monkeypatch):
        """Titles, previews, health values and notification records are all
        server-controlled strings printed to the terminal."""
        api = cli.GhostAPI("http://localhost:9", "k")
        monkeypatch.setattr(api, "sessions_list", lambda limit=50: (200, {
            "enabled": True, "sessions": [
                {"id": "abcd1234", "title": "evil" + OSC52, "message_count": 1}]}))
        monkeypatch.setattr(api, "turns", lambda: (200, {"turns": [
            {"request_id": "r1", "running": True, "age_s": 1,
             "preview": "p" + OSC52}]}))
        monkeypatch.setattr(api, "health", lambda timeout=(10, 30): (200, {
            "status": "ok" + OSC52, "rss_limit_mb": 512}))
        monkeypatch.setattr(api, "notifications", lambda c, limit=50: (200, {
            "enabled": True, "watermark": 1, "records": [
                {"ts": 1, "phase": "p" + OSC52, "summary": "s" + OSC52,
                 "severity": "info"}]}))
        monkeypatch.setattr(api, "notifications_ack", lambda *a: (200, {}))
        g = cli.GhostCLI(api)
        for fn in (g.cmd_sessions, g.cmd_turns, g.cmd_health, g.cmd_notify):
            assert OSC52 not in self._cap(monkeypatch, fn), fn.__name__

    def test_history_copy_is_clean(self, monkeypatch):
        """self.text is re-sent as context and written by /save — sanitize
        at ingest, not only on the way to the screen."""
        pr = cli._StreamPrinter()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=io.StringIO(), force_terminal=True, width=70,
            theme=cli.Theme(_THEME), highlight=False))
        pr.feed("hi " + ESC + "[2J there")
        pr.settle()
        assert ESC not in pr.text and pr.text == "hi [2J there"

    def test_piped_one_shot_is_stripped_too(self, monkeypatch):
        """The piped branch bypasses the printer entirely — and
        `ghost -p q | less -R` is still a terminal on the other end."""
        out = io.StringIO()
        out.isatty = lambda: False
        monkeypatch.setattr(cli.sys, "stdout", out)
        api = cli.GhostAPI("http://localhost:9", "k")
        payload = json.dumps({"choices": [{"delta": {"content": "a" + OSC52 + "b"}}]})
        monkeypatch.setattr(api.http, "post", lambda *a, **k: _FakeSSE(
            [("data: " + payload).encode(), b"data: [DONE]"]))
        cli.one_shot(api, "q", None)
        written = out.getvalue()
        # Defanged, not swallowed: the ESC that makes it an escape sequence
        # is removed and the now-inert printable remainder stays visible, so
        # an attempt is legible in the transcript instead of disappearing.
        assert OSC52 not in written and ESC not in written
        assert written.startswith("a]52;c;") and written.rstrip().endswith("b")


class TestReplResilience:
    """dispatch caught only RequestException, so ONE unexpected response
    shape ended the REPL and took the local history and the rateable turn
    refs with it. A command may fail; the session may not."""

    def _cli(self, monkeypatch):
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, width=70, theme=cli.Theme(_THEME), highlight=False))
        return cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k")), buf

    def test_malformed_session_row_does_not_kill_the_repl(self, monkeypatch):
        g, buf = self._cli(monkeypatch)
        monkeypatch.setattr(g.api, "sessions_list", lambda limit=50: (200, {
            "enabled": True, "sessions": [{"title": "row with no id key"}]}))
        assert g.dispatch("/sessions") is True          # KeyError, survived
        assert "session intact" in buf.getvalue()

    def test_string_rss_limit_renders_instead_of_raising(self, monkeypatch):
        """Surviving is not enough — dispatch's blanket catch would make a
        crash here look "handled". /health must still PRINT the panel."""
        g, buf = self._cli(monkeypatch)
        monkeypatch.setattr(g.api, "health", lambda timeout=(10, 30): (200, {
            "status": "ok", "rss_mb": 128, "rss_limit_mb": "512"}))
        assert g.dispatch("/health") is True
        out = buf.getvalue()
        assert "command failed" not in out       # not merely swallowed
        assert "128 MB / 512 MB" in out          # coerced and rendered

    def test_a_command_that_raises_leaves_state_intact(self, monkeypatch):
        g, _ = self._cli(monkeypatch)
        g.history = [{"role": "user", "content": "keep me"}]
        g.turns = [{"req_id": "abc", "prompt": "p", "interrupted": "", "label": None}]
        monkeypatch.setattr(g.api, "turns",
                            lambda: (_ for _ in ()).throw(TypeError("boom")))
        assert g.dispatch("/turns") is True
        assert g.history and g.turns          # not lost with the exception

    def test_exit_still_exits(self, monkeypatch):
        g, _ = self._cli(monkeypatch)
        for word in ("/exit", "/quit", "/bye", "exit", "quit"):
            assert g.dispatch(word) is False


class TestInterrupts:
    """Ctrl-C meant three different things and two of them ended the
    session. It aborts a reply (and cancels the turn); at an idle prompt it
    clears the line; Ctrl-D exits."""

    def _cli(self, monkeypatch):
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, force_terminal=True, width=70,
            theme=cli.Theme(_THEME), highlight=False))
        monkeypatch.setattr(cli, "render_reply_images", lambda *a, **k: None)
        api = cli.GhostAPI("http://localhost:9", "k")
        return cli.GhostCLI(api), api, buf

    @staticmethod
    def _partial_then_interrupt():
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}'
        raise KeyboardInterrupt

    def test_idle_ctrl_c_clears_the_line_and_ctrl_d_exits(self, monkeypatch):
        """Driven through real prompt_toolkit key handling, not a stub."""
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.application import create_app_session
        g, api, _ = self._cli(monkeypatch)
        monkeypatch.setattr(api, "health", lambda timeout=(2, 5): (200, {"uptime_s": 1}))
        seen = []
        monkeypatch.setattr(g, "chat_turn", lambda line: seen.append(line))
        with create_pipe_input() as pipe:
            pipe.send_text(chr(3) + "hello\r" + chr(3) + "world\r" + chr(4))
            with create_app_session(input=pipe, output=DummyOutput()):
                g.chat()
        assert seen == ["hello", "world"]

    def test_second_ctrl_c_during_the_cancel_keeps_the_session(self, monkeypatch):
        g, api, _ = self._cli(monkeypatch)
        monkeypatch.setattr(api, "chat_stream",
                            lambda *a, **k: _FakeSSE(self._partial_then_interrupt()))
        monkeypatch.setattr(api, "cancel_turn",
                            lambda **k: (_ for _ in ()).throw(KeyboardInterrupt))
        try:
            g.chat_turn("q")
        except KeyboardInterrupt:
            # Caught explicitly: an escaping KeyboardInterrupt aborts the
            # whole pytest session, which reads as a harness problem rather
            # than as this regression.
            pytest.fail("2nd Ctrl-C escaped chat_turn — the REPL would end")
        assert g.turns and g.turns[-1]["interrupted"] == "cancelled"

    def test_ctrl_c_in_image_fetch_keeps_the_reply_and_its_rateability(self, monkeypatch):
        """Image fetches are up to 4 serial round-trips with 130s timeouts.
        A Ctrl-C there used to cost the REPL, the assistant message and the
        turn ref — for a reply the agent had already finished."""
        g, api, _ = self._cli(monkeypatch)
        monkeypatch.setattr(cli, "render_reply_images",
                            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
        monkeypatch.setattr(api, "chat_stream", lambda *a, **k: _FakeSSE([
            b'data: {"choices":[{"delta":{"content":"see ![x](a.png)"}}]}',
            b"data: [DONE]"]))
        try:
            g.chat_turn("q")
        except KeyboardInterrupt:
            pytest.fail("Ctrl-C in the image fetch escaped chat_turn")
        assert len(g.history) == 2             # user + assistant kept
        assert len(g.turns) == 1 and g.turns[-1]["interrupted"] == ""

    def test_one_shot_ctrl_c_cancels_the_server_turn(self, monkeypatch):
        """Turns are globally SERIALIZED: an abandoned one-shot keeps
        working and keeps the lock, wedging every other client. The REPL
        always cancelled; this path did not."""
        g, api, _ = self._cli(monkeypatch)
        monkeypatch.setattr(api.http, "post",
                            lambda *a, **k: _FakeSSE(self._partial_then_interrupt()))
        seen = []
        monkeypatch.setattr(api, "cancel_turn",
                            lambda **k: (seen.append(k), (200, {"cancelled": True}))[1])
        assert cli.one_shot(api, "q", None) == 130
        assert len(seen) == 1 and seen[0]["request_id"]

    def test_one_shot_cancels_the_id_it_actually_sent(self, monkeypatch):
        """Cancelling a freshly-minted id would be a no-op that reports
        success — the request id must be the one on the wire."""
        g, api, _ = self._cli(monkeypatch)
        sent = {}

        def _post(*a, **k):
            sent.update(k.get("headers") or {})
            return _FakeSSE(self._partial_then_interrupt())

        monkeypatch.setattr(api.http, "post", _post)
        seen = []
        monkeypatch.setattr(api, "cancel_turn",
                            lambda **k: (seen.append(k), (200, {"cancelled": True}))[1])
        cli.one_shot(api, "q", None)
        assert seen[0]["request_id"] == sent["X-Request-ID"]


class TestLazyMarkdown:
    """feed() built Markdown(pending) per chunk, and Markdown's constructor
    runs a full markdown_it parse. When a reply has no blank line to flush
    on — a table, a long bullet list — the pending region IS the whole
    reply: a 400-row table cost 12.24s against 0.07s for the same volume of
    prose. Pinned as the MECHANISM (parse count), not as wall-clock."""

    def _count_parses(self, monkeypatch, reply):
        monkeypatch.setattr(cli, "console", cli.Console(
            file=io.StringIO(), force_terminal=True, width=100, height=30,
            theme=cli.Theme(_THEME), highlight=False))
        n = {"c": 0}
        real = cli.Markdown

        class Counting(real):
            def __init__(self, *a, **kw):
                n["c"] += 1
                super().__init__(*a, **kw)

        monkeypatch.setattr(cli, "Markdown", Counting)
        pr = cli._StreamPrinter()
        for i in range(0, len(reply), 12):
            pr.feed(reply[i:i + 12])
        pr.settle()
        return n["c"], len(range(0, len(reply), 12))

    def test_table_is_not_reparsed_per_chunk(self, monkeypatch):
        table = "| a | b |\n|---|---|\n" + "".join(
            f"| row {i} | second {i} |\n" for i in range(200))
        parses, chunks = self._count_parses(monkeypatch, table)
        assert chunks > 100
        # One parse at settle, plus whatever the Live thread rendered. The
        # bug was one per chunk; anything near `chunks` is the regression.
        assert parses < chunks / 10, f"{parses} parses for {chunks} chunks"

    def test_prose_still_flushes_blocks_progressively(self, monkeypatch):
        """The lazy wrapper must not break the settled-block path: complete
        blocks still print permanently as they arrive."""
        prose = "".join(f"Paragraph {i} text here.\n\n" for i in range(30))
        parses, _ = self._count_parses(monkeypatch, prose)
        assert parses >= 30            # one per completed block, as designed

    def test_lazy_markdown_defers_the_parse(self, monkeypatch):
        n = {"c": 0}
        real = cli.Markdown

        class Counting(real):
            def __init__(self, *a, **kw):
                n["c"] += 1
                super().__init__(*a, **kw)

        monkeypatch.setattr(cli, "Markdown", Counting)
        lazy = cli._LazyMarkdown("# not parsed yet")
        assert n["c"] == 0                       # construction is free
        con = cli.Console(file=io.StringIO(), width=40)
        con.print(lazy)
        assert n["c"] == 1                       # parsed only when rendered


class TestCommandSurface:
    """Three places name a command — dispatch, COMPLETER, HELP. When they
    drift, Tab and /help lie about what exists."""

    SRC = CLI_PATH.read_text()

    def _sets(self):
        import re
        body = self.SRC.split("def dispatch", 1)[1]
        dispatched = set(re.findall(r'case "(/[^"]+)"', body))
        dispatched |= set(re.findall(
            r'"(/[^"]+)"', self.SRC.split("RATE_CMDS = {", 1)[1].split("}", 1)[0]))
        dispatched |= {"/exit", "/quit", "/bye"}
        completer = set(re.findall(
            r'"(/[^"]+)":', self.SRC.split("COMPLETER", 1)[1].split("})", 1)[0]))
        helpsec = self.SRC.split("HELP = ", 1)[1].split('"""', 2)[1]
        helped = (set(re.findall(r'\[you\](/[a-z+\-0-9]+)', helpsec))
                  | set(re.findall(r'(/[+\-]1)', helpsec)))
        return dispatched, completer, helped

    def test_everything_dispatched_is_tab_completable(self):
        d, c, _ = self._sets()
        assert d - c == set()

    def test_everything_dispatched_is_documented(self):
        d, _, h = self._sets()
        assert d - h == set()

    def test_nothing_completable_is_a_dead_command(self):
        d, c, _ = self._sets()
        assert c - d == set()


def _legacy_ansi_colours(text: str) -> set:
    """Standard 8/16-colour SGR codes in `text`.

    Walks the parameter list rather than scanning for numbers: 38 and 48 are
    the EXTENDED-colour introducers, so `38;2;95;102;117` (a truecolor grey)
    contains both a bare 38 and a bare 95 and a naive scan reports two
    "legacy colours" that are not there."""
    found = set()
    for params in re.findall(r"\x1b\[([0-9;]*)m", text):
        codes = [int(c) for c in params.split(";") if c.isdigit()]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c in (38, 48):                      # extended colour
                nxt = codes[i + 1] if i + 1 < len(codes) else None
                i += 5 if nxt == 2 else 3 if nxt == 5 else 1
                continue
            if 30 <= c <= 37 or 40 <= c <= 47 or 90 <= c <= 107:
                found.add(c)
            i += 1
    return found


def test_legacy_ansi_detector_can_tell_the_difference():
    """The detector above is the instrument for the palette pin — an
    instrument that cannot distinguish proves nothing."""
    assert _legacy_ansi_colours("\x1b[38;2;95;102;117mx\x1b[0m") == set()
    assert _legacy_ansi_colours("\x1b[38;5;141mx\x1b[0m") == set()
    assert _legacy_ansi_colours("\x1b[1;36;40mx\x1b[0m") == {36, 40}
    assert _legacy_ansi_colours("\x1b[95mx\x1b[0m") == {95}


class TestChrome:
    """The visual layer (2026-08-27). Chrome that wraps, or that lets rich's
    stock colours through, is what the operator sees first — so the
    invariants are pinned like any other behaviour."""

    def _api(self, host="http://eva:8000", uptime=12045):
        api = cli.GhostAPI(host, "k")
        api.health = lambda timeout=(2, 5): (200, {"uptime_s": uptime})
        return api

    def _banner(self, monkeypatch, width, host="http://eva:8000"):
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, force_terminal=True, width=width, height=40,
            theme=cli.Theme(_THEME), highlight=False))
        g = cli.GhostCLI(self._api(host))
        g.session_id = "a1b2c3d4e5f6"
        g.banner()
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
        return [l for l in plain.split("\n") if l.strip()]

    # -- the banner must never wrap ---------------------------------------
    @pytest.mark.parametrize("width", [16, 24, 34, 44, 64, 80, 120])
    def test_banner_is_exactly_three_lines(self, monkeypatch, width):
        """Header, rule, hint. A wrapped banner shows as EXTRA lines, not as
        an over-long one — measuring line length cannot detect it."""
        assert len(self._banner(monkeypatch, width)) == 3

    def test_banner_survives_a_long_hostname(self, monkeypatch):
        lines = self._banner(monkeypatch, 40,
                             host="https://a-very-long-hostname.example.com:18443")
        assert len(lines) == 3

    def test_banner_sheds_detail_in_priority_order(self, monkeypatch):
        """Identity and liveness are never what gets cut."""
        wide = self._banner(monkeypatch, 80)[0]
        narrow = self._banner(monkeypatch, 44)[0]
        tiny = self._banner(monkeypatch, 34)[0]
        assert "eva:8000" in wide and "a1b2c3d4" in wide
        assert "eva:8000" in narrow and "a1b2c3d4" not in narrow
        assert "eva:8000" not in tiny
        for line in (wide, narrow, tiny):
            assert "ghost" in line and "online" in line

    def test_rule_spans_the_terminal(self, monkeypatch):
        """It was hardcoded to 52 columns — short in a wide window, and the
        cause of the wrap in a narrow one."""
        for width in (40, 72, 110):
            monkeypatch.setattr(cli, "console", cli.Console(
                file=io.StringIO(), width=width, theme=cli.Theme(_THEME)))
            assert cli._rule().cell_len == width - cli.GUTTER

    def test_fit_line_truncates_when_nothing_is_left_to_drop(self):
        """The invariant must not hold only while something remains
        droppable — that is precisely when it stops being exercised."""
        line = cli._fit_line(head=[("y" * 200, "")], optional=[], width=20)
        assert line.cell_len <= 19
        assert str(line).endswith("…")

    # -- one left edge -----------------------------------------------------
    def _render(self, monkeypatch, fn, width=72):
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, force_terminal=True, width=width, height=40,
            theme=cli.Theme(_THEME), highlight=False))
        fn()
        import re
        return [l for l in re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).split("\n")
                if l.strip()]

    def test_reply_notice_and_table_share_the_gutter(self, monkeypatch):
        """Replies used to print flush at column 0 while every notice and
        table was indented two, so the most important text on screen was the
        one thing out of alignment.

        Measured, not compared against cli.GUTTER: an assertion built from
        the constant it is guarding passes trivially when that constant goes
        to zero (it did — the mutant survived)."""
        def _reply():
            pr = cli._StreamPrinter()
            pr.feed("Disk on eva sits at 71 percent.\n\n")
            pr.settle()

        indents = {}
        for label, fn in [
                ("reply", _reply),
                ("notice", lambda: cli.GhostCLI.notice("recorded")),
                ("table", lambda: cli._table(["id"], [["a1b2c3d4"]]))]:
            seen = {len(l) - len(l.lstrip(" "))
                    for l in self._render(monkeypatch, fn)}
            assert len(seen) == 1, f"{label} is not on one left edge: {seen}"
            indents[label] = seen.pop()
        assert len(set(indents.values())) == 1, indents
        assert indents["reply"] >= 2, "there is no gutter at all"

    # -- the palette owns every colour -------------------------------------
    def test_no_stock_ansi_colours_leak_into_a_reply(self, monkeypatch):
        """rich's markdown defaults are cyan table borders, cyan-on-BLACK
        code spans and magenta headings — they belonged to no palette and
        were the loudest thing in any reply with a table or a code span."""
        import re
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, force_terminal=True, color_system="truecolor", width=72,
            height=40, theme=cli.Theme(_THEME), highlight=False))
        pr = cli._StreamPrinter()
        pr.feed("## Heading\n\n"
                "| volume | used |\n|---|---|\n| /Users | 340G |\n\n"
                "- run `du -sh` here\n- see [ncdu](https://example.com)\n\n"
                "> a quote\n\n")
        pr.settle()
        legacy = _legacy_ansi_colours(buf.getvalue())
        assert not legacy, f"stock ANSI colours in a reply: {sorted(legacy)}"

    def test_every_palette_role_is_a_truecolor_value(self):
        for role, value in cli.PALETTE.items():
            assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), (role, value)

    def test_theme_overrides_rich_markdown_defaults(self):
        """If rich adds a new stock markdown style, this says so rather than
        letting it appear on screen unnoticed."""
        from rich.default_styles import DEFAULT_STYLES
        stock_coloured = {
            k for k, v in DEFAULT_STYLES.items()
            if k.startswith("markdown.") and (v.color or v.bgcolor)}
        overridden = set(_THEME)
        assert stock_coloured <= overridden, \
            f"unowned rich markdown styles: {sorted(stock_coloured - overridden)}"

    def test_code_span_has_no_black_backing(self):
        assert _THEME["markdown.code"].bgcolor is None

    @staticmethod
    def _sgr(hex_colour):
        """The truecolor PARAMS only — rich merges attributes into one
        sequence (`\\x1b[1;38;2;…m` for a bold coloured run), so anchoring on
        `\\x1b[38;2;` misses every styled span."""
        r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
        return f"38;2;{r};{g};{b}m"

    def test_bullets_and_links_carry_the_palette(self, monkeypatch):
        """Two roles rich leaves un-coloured or mis-coloured by default, and
        that the "no stock ANSI" pin cannot see: a bullet whose default is
        plain bold, and an anchor whose text rich styles with link_url — so
        pointing that at the border grey made every link nearly invisible.
        Asserted on the RENDERED bytes, not on the theme dict."""
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", cli.Console(
            file=buf, force_terminal=True, color_system="truecolor", width=72,
            height=40, theme=cli.Theme(_THEME), highlight=False))
        pr = cli._StreamPrinter()
        pr.feed("- see [ncdu](https://example.com)\n\n")
        pr.settle()
        out = buf.getvalue()
        assert self._sgr(cli.PALETTE["ghost"]) in out, "bullet lost its colour"
        assert self._sgr(cli.PALETTE["you"]) in out, "link text is not readable"
        assert self._sgr(cli.PALETTE["faint"]) not in out, \
            "content is rendering in the border colour"

    # -- the prompt --------------------------------------------------------
    def test_prompt_glyph_is_light_and_single_cell(self):
        """❯ is the shell-prompt cliché; ▌ and its block-drawing relatives
        smear into a continuous vertical bar down the transcript in the
        prompt's redraw region (operator screenshot, 2026-08-27)."""
        assert "❯" not in cli.PROMPT_GLYPH
        assert not (set(cli.PROMPT_GLYPH) & set("▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏┃┇┋║")), \
            f"{cli.PROMPT_GLYPH!r} uses a block/heavy-line glyph"
        assert cli.PROMPT_GLYPH.strip()          # still shows something

    def _fake_prompt_session(self, monkeypatch):
        seen = {"kwargs": None, "fragments": []}

        class _FakePrompt:
            def __init__(self, **kw):
                seen["kwargs"] = kw

            def prompt(self, fragments):
                seen["fragments"].append(fragments)
                return None                      # → clean exit

        monkeypatch.setattr(cli, "PromptSession", _FakePrompt)
        monkeypatch.setattr(cli, "console", cli.Console(
            file=io.StringIO(), width=72, theme=cli.Theme(_THEME)))
        cli.GhostCLI(self._api()).chat()
        return seen

    def test_chat_uses_the_prompt_glyph(self, monkeypatch):
        """Pinned at the call site: a constant nothing reads is decoration."""
        seen = self._fake_prompt_session(monkeypatch)
        assert seen["fragments"]
        assert seen["fragments"][0][0][1] == cli.PROMPT_GLYPH

    def test_no_reserved_menu_gap_under_the_prompt(self, monkeypatch):
        """prompt_toolkit reserves EIGHT rows for the completion menu by
        default, for the whole session, menu or no menu — a permanent blank
        gap under the prompt, and a scrolling redraw region that leaves a
        copy of the prompt glyph behind on every frame. Tab-completes like a
        shell instead."""
        from prompt_toolkit.shortcuts import CompleteStyle
        kw = self._fake_prompt_session(monkeypatch)["kwargs"]
        assert kw["reserve_space_for_menu"] == 0
        assert kw["complete_style"] == CompleteStyle.READLINE_LIKE
        assert kw["complete_while_typing"] is False
        assert kw["completer"] is cli.COMPLETER      # Tab still completes
