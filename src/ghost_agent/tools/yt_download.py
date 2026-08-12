"""YouTube-over-Tor download step with exit-node rotation + 429 retry.

The resilient downloader itself is a shell script (``yt_tor_download.sh``,
alongside this module) so it can be shellcheck'd, read, and unit-tested in
isolation. This module's job is to smuggle that script into the
``youtube_transcribe`` macro's ``execute`` step.

Why the smuggling: a composed-skill ``param_template`` runs every ``$name``
through the macro variable resolver (``composed_skills._VAR_RE``), and an
UNKNOWN ``$name`` resolves to ``""``. A normal bash retry loop is full of
``$i`` / ``$RANDOM`` / ``$tag`` shell variables — they would all be eaten,
destroying the script. base64 has NO ``$`` in its alphabet, so a base64-encoded
script passes through the resolver untouched; only the genuine ``$url`` runtime
param is substituted. The sandbox (Linux) decodes and runs it.
"""

from __future__ import annotations

import base64
from pathlib import Path

# The macro's single runtime input, substituted by the composed-skill resolver
# at call time. It is the ONLY `$var` the built command must contain.
URL_VAR = "$url"

# Heredoc delimiter used to hand the URL to the script WITHOUT shell evaluation
# (see build_download_command). A URL can never legitimately contain this on a
# line by itself.
_URL_HEREDOC = "__GHOST_YT_URL__"

_SCRIPT_PATH = Path(__file__).with_name("yt_tor_download.sh")


def read_download_script() -> str:
    """Return the authoritative Tor-download shell script's source."""
    return _SCRIPT_PATH.read_text(encoding="utf-8")


def _b64_script() -> str:
    """base64 of the script — guaranteed free of ``$`` (alphabet is
    ``[A-Za-z0-9+/=]``), so the composed-skill resolver never touches it."""
    return base64.b64encode(read_download_script().encode("utf-8")).decode("ascii")


def build_download_command(url_var: str = URL_VAR, *, install: bool = True) -> str:
    """The macro's step-1 ``execute`` command.

    Installs yt-dlp if absent, hands the URL to the downloader via a file, then
    decodes the base64'd script and runs it. Only ``url_var`` is a live
    ``$var``; the base64 blob is inert to the resolver.

    Security — the URL never touches a shell-evaluated context. The composed-
    skill resolver substitutes ``url_var`` by raw string replacement (no shell
    escaping), so a hostile URL like ``https://x/$(rm -rf ~)`` interpolated into
    a command would command-substitute when the sandbox runs it. Instead the URL
    is written to ``.yt_url`` through a QUOTED heredoc (``<<'DELIM'``), which
    bash copies out LITERALLY with no expansion; the script reads the file and
    only ever uses the value as a quoted ``"$URL"`` variable. Result: the URL
    is data, never code.
    """
    lines = []
    if install:
        lines.append("command -v yt-dlp >/dev/null 2>&1 || pip install -q yt-dlp")
    # Quoted-heredoc write: url_var is substituted into the BODY by the resolver,
    # but bash (quoted delimiter) writes it verbatim — no `$()`/backtick/quote
    # evaluation. The script (bash -s, no arg) reads it back from .yt_url.
    lines.append(f"cat > .yt_url <<'{_URL_HEREDOC}'")
    lines.append(url_var)
    lines.append(_URL_HEREDOC)
    lines.append(f"echo '{_b64_script()}' | base64 -d | bash -s")
    return "\n".join(lines)


def build_youtube_transcribe_definition() -> dict:
    """The full, current ``youtube_transcribe`` macro definition (sequential:
    resilient Tor download → knowledge-base transcription)."""
    return {
        "name": "youtube_transcribe",
        "description": (
            "Download a YouTube video's audio over Tor (rotating exit nodes, "
            "retrying on HTTP 429 / rate-limits) and transcribe it into the "
            "knowledge base in one call. Input: url."
        ),
        "mode": "sequential",
        "steps": [
            {
                "tool": "execute",
                "description": "Download audio over Tor with exit rotation + 429 retry",
                "params": {"command": build_download_command()},
            },
            {
                "tool": "knowledge_base",
                "description": "Transcribe the downloaded audio into the knowledge base",
                "params": {"action": "transcribe", "filename": "yt_audio.m4a"},
            },
        ],
    }
