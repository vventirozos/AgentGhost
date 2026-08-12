"""YouTube-over-Tor download: exit-node rotation + retry-on-429.

The resilient downloader is `src/ghost_agent/tools/yt_tor_download.sh`. These
tests drive the REAL script with a fake `yt-dlp` on PATH (no network, no Tor),
asserting: it rotates the SOCKS identity (→ a fresh Tor circuit/exit) every
attempt, retries on 429/block, pre-cleans stale audio, stops early with the real
status on a non-retryable error, and fails cleanly after exhausting attempts.
Plus: the URL is passed as DATA (a hostile URL cannot inject shell commands),
the base64 smuggling survives the composed-skill variable resolver, and the
macro command never drifts from the authoritative script.
"""

import base64
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ghost_agent.tools import yt_download
from ghost_agent.tools.yt_download import (
    build_download_command, build_youtube_transcribe_definition,
    read_download_script, URL_VAR,
)
from ghost_agent.tools.composed_skills import ComposedSkillRegistry, SkillStep

_SCRIPT = yt_download._SCRIPT_PATH

_FAKE_YTDLP = r'''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
with open(os.environ["YT_FAKE_ARGV_LOG"], "a") as f:
    f.write(repr(args) + "\n")
proxy = ""
for i, a in enumerate(args):
    if a == "--proxy" and i + 1 < len(args):
        proxy = args[i + 1]
with open(os.environ["YT_FAKE_PROXY_LOG"], "a") as f:
    f.write(proxy + "\n")
cf = os.environ["YT_FAKE_COUNTER"]
n = int(open(cf).read() or "0") if os.path.exists(cf) else 0
n += 1
open(cf, "w").write(str(n))
out = os.environ.get("YT_TOR_OUT", "yt_audio")
ext = os.environ.get("YT_FAKE_EXT", "m4a")
mode = os.environ.get("YT_FAKE_MODE", "success")
def make(): open(out + "." + ext, "w").close()
if mode == "success":
    make(); sys.exit(0)
if mode == "block_then_ok":
    if n < int(os.environ.get("YT_FAKE_OK_AT", "3")):
        print("WARNING: [youtube] X: HTTP Error 429: Too Many Requests")
        print("ERROR: [youtube] X: This video is unavailable")
        sys.exit(1)
    make(); sys.exit(0)
if mode == "always_block":
    print("WARNING: HTTP Error 429: Too Many Requests"); sys.exit(1)
if mode == "block_unavailable":
    print("ERROR: [youtube] X: Video unavailable"); sys.exit(1)
if mode == "fatal":
    print("ERROR: Unsupported URL: not-a-video")
    sys.exit(int(os.environ.get("YT_FAKE_FATAL_RC", "1")))
sys.exit(0)
'''


def _fake_env(tmp_path, mode, *, attempts=5, ok_at=3, extra_env=None):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "yt-dlp"
    fake.write_text(_FAKE_YTDLP)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    env = dict(os.environ)
    # fake bin first, then python (for the fake's shebang), then the rest
    env["PATH"] = os.pathsep.join(
        [str(bindir), str(Path(sys.executable).parent), env.get("PATH", "")])
    env["YT_TOR_ATTEMPTS"] = str(attempts)
    env["YT_TOR_BACKOFF_BASE"] = "0"
    env["YT_TOR_PROXY_HOST"] = "127.0.0.1:9999"
    env["YT_FAKE_MODE"] = mode
    env["YT_FAKE_OK_AT"] = str(ok_at)
    env["YT_FAKE_PROXY_LOG"] = str(tmp_path / "proxies.txt")
    env["YT_FAKE_ARGV_LOG"] = str(tmp_path / "argv.txt")
    env["YT_FAKE_COUNTER"] = str(tmp_path / "counter.txt")
    if extra_env:
        env.update(extra_env)
    return env


def _reads(tmp_path):
    proxies = (tmp_path / "proxies.txt").read_text().split() if (tmp_path / "proxies.txt").exists() else []
    argv = (tmp_path / "argv.txt").read_text().splitlines() if (tmp_path / "argv.txt").exists() else []
    count = int((tmp_path / "counter.txt").read_text()) if (tmp_path / "counter.txt").exists() else 0
    return proxies, argv, count


def _run_script(tmp_path, mode, *, attempts=5, ok_at=3, extra_env=None):
    """Run the real script with the URL as $1 (controlled test input)."""
    env = _fake_env(tmp_path, mode, attempts=attempts, ok_at=ok_at, extra_env=extra_env)
    res = subprocess.run(
        ["bash", str(_SCRIPT), "https://www.youtube.com/watch?v=abc&list=xyz"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60)
    proxies, argv, count = _reads(tmp_path)
    return res, proxies, count


def _run_full_command(tmp_path, url, mode="success", *, attempts=3):
    """Run the FULL macro step-1 command: resolve $url like the composed-skill
    engine would, wrap with shlex.quote like the sandbox does, execute. The URL
    reaches the script only via the quoted-heredoc file — never a shell arg."""
    env = _fake_env(tmp_path, mode, attempts=attempts)
    cmd = build_download_command(install=False)
    step = SkillStep(tool_name="execute", description="dl",
                     param_template={"command": cmd})
    resolved = ComposedSkillRegistry._resolve_args(step, {"url": url})["command"]
    wrapped = f"bash -c {shlex.quote(resolved)}"
    res = subprocess.run(wrapped, shell=True, cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=60)
    proxies, argv, count = _reads(tmp_path)
    return res, argv


# ─────────────────────────────────────────────────────────────────────────
# Script behaviour
# ─────────────────────────────────────────────────────────────────────────
class TestDownloadScript:
    def test_success_first_attempt(self, tmp_path):
        res, proxies, count = _run_script(tmp_path, "success")
        assert res.returncode == 0, res.stdout + res.stderr
        assert count == 1 and len(proxies) == 1
        assert (tmp_path / "yt_audio.m4a").exists()
        assert "succeeded on attempt 1" in res.stdout

    def test_retries_on_429_then_succeeds_rotating_exits(self, tmp_path):
        res, proxies, count = _run_script(tmp_path, "block_then_ok", attempts=5, ok_at=3)
        assert res.returncode == 0, res.stdout + res.stderr
        assert count == 3 and len(proxies) == 3
        assert len(set(proxies)) == 3, proxies          # a fresh circuit each try
        for i, p in enumerate(proxies, 1):
            assert re.match(rf"socks5h://ytdl_{i}_\d+:x@127\.0\.0\.1:9999$", p), p
        assert "rotating Tor circuit" in res.stdout
        assert "succeeded on attempt 3" in res.stdout

    def test_non_retryable_error_stops_early_with_real_status(self, tmp_path):
        # yt-dlp exits 5 on a non-retryable error → the script must propagate 5,
        # stop after ONE attempt, and NOT claim "every Tor exit was blocked".
        res, proxies, count = _run_script(
            tmp_path, "fatal", attempts=5, extra_env={"YT_FAKE_FATAL_RC": "5"})
        assert res.returncode == 5, (res.returncode, res.stdout)
        assert count == 1 and len(proxies) == 1
        assert "non-retryable" in res.stdout
        assert "every Tor exit" not in res.stdout        # accurate message

    def test_exhausts_attempts_when_all_blocked(self, tmp_path):
        res, proxies, count = _run_script(tmp_path, "always_block", attempts=3)
        assert res.returncode == 1
        assert count == 3 and len(set(proxies)) == 3
        assert "DOWNLOAD FAILED after 3 attempt" in res.stdout
        assert "every Tor exit" in res.stdout

    def test_missing_url_errors(self, tmp_path):
        res = subprocess.run(["bash", str(_SCRIPT)], cwd=str(tmp_path),
                             capture_output=True, text=True, timeout=30)
        assert res.returncode == 2
        assert "usage" in (res.stdout + res.stderr).lower()

    def test_rejects_non_http_url(self, tmp_path):
        env = _fake_env(tmp_path, "success")
        res = subprocess.run(["bash", str(_SCRIPT), "file:///etc/passwd"],
                             cwd=str(tmp_path), env=env, capture_output=True,
                             text=True, timeout=30)
        assert res.returncode == 2
        assert "non-http" in (res.stdout + res.stderr).lower()

    def test_preclean_removes_stale_audio(self, tmp_path):
        # A leftover yt_audio.webm from a PRIOR video must be removed before the
        # download, so the transcribe step's stem-resolver can't pick stale audio.
        (tmp_path / "yt_audio.webm").write_text("STALE")
        res, proxies, count = _run_script(tmp_path, "success")
        assert res.returncode == 0, res.stdout + res.stderr
        assert not (tmp_path / "yt_audio.webm").exists()   # stale gone
        assert (tmp_path / "yt_audio.m4a").exists()         # only the fresh file

    def test_non_m4a_download_still_succeeds(self, tmp_path):
        # yt-dlp falls back to opus/webm when no m4a stream exists; step 1 must
        # still report success (the transcribe stem-resolver handles the ext).
        res, proxies, count = _run_script(
            tmp_path, "success", extra_env={"YT_FAKE_EXT": "webm"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert (tmp_path / "yt_audio.webm").exists()

    def test_default_attempts_is_eight(self, tmp_path):
        # No explicit YT_TOR_ATTEMPTS → the script's default (8) applies.
        res, proxies, count = _run_script(tmp_path, "always_block", attempts="")
        assert res.returncode == 1
        assert count == 8, "default attempts should be 8"
        assert len(set(proxies)) == 8

    def test_backoff_is_capped(self, tmp_path):
        # A huge base with a 0 cap must be clamped to 0 (prints "backoff 0s"
        # and does NOT sleep 100s) — proves the cap, and keeps the test fast.
        res, proxies, count = _run_script(
            tmp_path, "block_then_ok", attempts=4, ok_at=2,
            extra_env={"YT_TOR_BACKOFF_BASE": "100", "YT_TOR_BACKOFF_MAX": "0"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "backoff 0s" in res.stdout          # 100 clamped to 0

    def test_bare_video_unavailable_is_retryable(self, tmp_path):
        # yt-dlp emits a bare "Video unavailable" (no "this…") when a Tor exit is
        # blocked — it must rotate through every exit, not stop at the first.
        res, proxies, count = _run_script(tmp_path, "block_unavailable", attempts=3)
        assert res.returncode == 1
        assert count == 3, "should have rotated through all 3 exits"
        assert len(set(proxies)) == 3
        assert "DOWNLOAD FAILED after 3 attempt" in res.stdout


def _fake(path, body):
    path.write_text(body)
    path.chmod(0o755)


def _installer_fakes(tmp_path):
    """Fake curl/sudo/apt-get so the deno self-heal can 'run'; `sudo` records
    every call to $YT_FAKE_INSTALL_MARKER."""
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    _fake(b / "curl", "#!/bin/sh\nexit 0\n")
    _fake(b / "sudo", '#!/bin/sh\necho "$*" >> "$YT_FAKE_INSTALL_MARKER"\nexit 0\n')
    _fake(b / "apt-get", "#!/bin/sh\nexit 0\n")


def _fake_deno(tmp_path):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    _fake(b / "deno", "#!/bin/sh\nexit 0\n")


# PATH with coreutils + venv python but NO deno — makes `command -v deno` empty
# deterministically regardless of what's installed on the dev host.
def _no_deno_path(tmp_path):
    return os.pathsep.join([
        str(tmp_path / "bin"), str(Path(sys.executable).parent), "/usr/bin", "/bin"])


class TestJsRuntime:
    def test_passes_deno_js_runtime_when_present(self, tmp_path):
        # deno present → handed to yt-dlp as --js-runtimes deno:<path>.
        _fake_deno(tmp_path)
        res, proxies, count = _run_script(tmp_path, "success")
        assert res.returncode == 0, res.stdout + res.stderr
        argv = (tmp_path / "argv.txt").read_text()
        assert "--js-runtimes" in argv and "deno:" in argv, argv

    def test_omits_js_runtime_when_disabled(self, tmp_path):
        # YT_TOR_NO_JS=1 disables it — and exercises the empty-array path.
        res, proxies, count = _run_script(
            tmp_path, "success", extra_env={"YT_TOR_NO_JS": "1"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "--js-runtimes" not in (tmp_path / "argv.txt").read_text()

    def test_installs_deno_when_absent(self, tmp_path):
        # No deno on PATH → the self-heal must attempt the deno install
        # (recorded via the fake sudo running the DENO_INSTALL installer).
        marker = tmp_path / "install.txt"
        _installer_fakes(tmp_path)
        res, proxies, count = _run_script(
            tmp_path, "success",
            extra_env={"YT_FAKE_INSTALL_MARKER": str(marker),
                       "PATH": _no_deno_path(tmp_path)})
        assert res.returncode == 0, res.stdout + res.stderr
        assert marker.exists(), "self-heal should have attempted a deno install"
        assert "DENO_INSTALL" in marker.read_text()

    def test_skips_install_when_deno_present(self, tmp_path):
        # deno already present → no install attempt.
        marker = tmp_path / "install.txt"
        _installer_fakes(tmp_path)
        _fake_deno(tmp_path)
        res, proxies, count = _run_script(
            tmp_path, "success", extra_env={"YT_FAKE_INSTALL_MARKER": str(marker)})
        assert res.returncode == 0, res.stdout + res.stderr
        assert not marker.exists(), "should not install when deno is present"


# ─────────────────────────────────────────────────────────────────────────
# URL is DATA, not code
# ─────────────────────────────────────────────────────────────────────────
class TestUrlInjectionSafety:
    @pytest.mark.parametrize("payload,marker", [
        ("https://youtu.be/x$(touch INJECTED_A)", "INJECTED_A"),
        ('https://youtu.be/x"; touch INJECTED_B; echo "', "INJECTED_B"),
        ("https://youtu.be/x`touch INJECTED_C`", "INJECTED_C"),
    ])
    def test_hostile_url_does_not_execute(self, tmp_path, payload, marker):
        res, argv = _run_full_command(tmp_path, payload, mode="success")
        # the injected command must NOT have run
        assert not (tmp_path / marker).exists(), f"INJECTION: {marker} created"
        # and the URL reached yt-dlp as a single literal argument (whitespace is
        # stripped by the script, but no `$()`/quote was ever evaluated)
        flat = " ".join(argv)
        assert "INJECTED" in flat or res.returncode == 2  # url present literally, or refused

    def test_benign_url_with_ampersand_passes_through(self, tmp_path):
        res, argv = _run_full_command(
            tmp_path, "https://www.youtube.com/watch?v=abc&list=xyz", "success")
        assert res.returncode == 0, res.stdout + res.stderr
        assert any("v=abc&list=xyz" in a for a in argv), argv


# ─────────────────────────────────────────────────────────────────────────
# base64 smuggling: resolver-safe + no drift
# ─────────────────────────────────────────────────────────────────────────
def _extract_b64(command: str) -> str:
    m = re.search(r"echo '([A-Za-z0-9+/=]+)' \| base64 -d", command)
    assert m, f"no base64 blob in command: {command[:120]}…"
    return m.group(1)


class TestCommandSmuggling:
    def test_command_carries_only_the_url_var(self, tmp_path):
        cmd = build_download_command()
        assert URL_VAR in cmd
        assert "$" not in cmd.replace(URL_VAR, ""), cmd

    def test_base64_blob_has_no_dollar(self):
        assert "$" not in _extract_b64(build_download_command())

    def test_survives_composed_skill_resolver(self):
        cmd = build_download_command()
        step = SkillStep(tool_name="execute", description="dl",
                         param_template={"command": cmd})
        url = "https://youtu.be/abc&list=xyz"
        out = ComposedSkillRegistry._resolve_args(step, {"url": url})["command"]
        assert url in out and URL_VAR not in out
        assert base64.b64decode(_extract_b64(out)).decode("utf-8") == read_download_script()

    def test_definition_command_matches_authoritative_script(self):
        # Pins the DEFINITION (what sync writes) to the script: if someone edits
        # yt_tor_download.sh, the macro sync must be re-run — this catches the
        # code side of that. (Deployed-store freshness is a re-run-sync concern.)
        d = build_youtube_transcribe_definition()
        b64 = _extract_b64(d["steps"][0]["params"]["command"])
        assert base64.b64decode(b64).decode("utf-8") == read_download_script()

    def test_script_has_required_behaviour_markers(self):
        # Guards against the script being gutted (a decoded-blob equality test
        # alone would still pass on an empty/wrong script).
        src = read_download_script()
        assert 'tag="ytdl_${i}_${RANDOM}"' in src        # per-attempt rotation
        assert 'socks5h://${tag}' in src                  # SOCKS-auth isolation
        assert 'rm -f "${OUT}".*' in src                  # pre-clean
        assert "--js-runtimes" in src                      # JS runtime for extraction
        assert "deno.land/install" in src                  # deno self-heal
        assert "too many requests" in src.lower()          # block detection

    def test_install_prefix_toggle(self):
        assert build_download_command(install=True).startswith("command -v yt-dlp")
        assert build_download_command(install=False).startswith("cat > .yt_url")


class TestMacroDefinition:
    def test_definition_shape(self):
        d = build_youtube_transcribe_definition()
        assert d["name"] == "youtube_transcribe"
        assert d["mode"] == "sequential"
        assert len(d["steps"]) == 2
        s0, s1 = d["steps"]
        assert s0["tool"] == "execute"
        assert "base64 -d" in s0["params"]["command"]
        assert s1["tool"] == "knowledge_base"
        assert s1["params"] == {"action": "transcribe", "filename": "yt_audio.m4a"}
