"""interface/server.py log-stream + robustness fixes (2026-07-12).

Found via a live audit: the interface server's tail followed
``/Users/vasilis/AI/Logs/ghost-agent.log`` — a path that DOES NOT EXIST
(missing ``Data/``) — so the web UI's live log stream (face pulses, planner
monologue) was silently dead in production. Compounding it, uvicorn owns
sys.argv in the prod deployment, so the ``--agent-log`` flag could never
actually arrive; the env var is the only real override.
"""

import os
import sys
import inspect
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _server():
    import interface.server as server
    return server


class TestAgentLogPath:
    def test_default_log_path_exists_family(self):
        # The default must point under /Users/vasilis/Data/AI/Logs — the old
        # default dropped the `Data/` segment and tailed a ghost file.
        s = _server()
        assert "/Data/AI/Logs/" in s._DEFAULT_AGENT_LOG
        assert "/Users/vasilis/AI/" not in s._DEFAULT_AGENT_LOG

    def test_env_var_is_the_override_mechanism(self):
        # uvicorn owns argv in prod, so GHOST_AGENT_LOG must be consulted.
        src = inspect.getsource(_server())
        assert 'os.environ.get(\n    "GHOST_AGENT_LOG"' in src or \
               'os.environ.get("GHOST_AGENT_LOG"' in src.replace("\n    ", " ")


class TestLogStreamerResilience:
    def test_streamer_restarts_when_tail_exits(self):
        # readline() returning b'' (tail died) must NOT permanently end the
        # stream — log_streamer wraps _log_streamer_once in a restart loop.
        s = _server()
        assert hasattr(s, "_log_streamer_once")
        src = inspect.getsource(s.log_streamer)
        assert "_log_streamer_once" in src
        assert "while True" in src            # restart loop
        assert "CancelledError" in src        # shutdown still propagates


class TestStreamCapClosesUpstream:
    def test_cap_breaks_instead_of_draining_forever(self):
        # On buffer-cap hit the worker must BREAK (closing the upstream
        # stream) — `continue` kept consuming the rest of a 30-min turn
        # while discarding every byte.
        import re
        src = inspect.getsource(_server())
        m = re.search(r'if t\["buffer_size"\] \+ chunk_len > t\["stream_cap"\]:'
                      r'(.*?)t\["buffer"\]\.append', src, re.DOTALL)
        assert m, "buffer-cap branch not found"
        branch = m.group(1)
        assert re.search(r"^\s*break\s*$", branch, re.M)
        # No bare `continue` STATEMENT (the explanatory comment may use the
        # word).
        assert not re.search(r"^\s*continue\s*$", branch, re.M)


class TestSttFormFieldGuard:
    def test_text_field_named_file_is_a_400(self):
        # A plain-text form field named "file" arrives as str; .read() blew
        # up as an opaque 502. Must be rejected as a client error instead.
        src = inspect.getsource(_server().stt_proxy)
        assert 'hasattr(file, "read")' in src
        assert "400" in src


class TestMobileTouchHardening:
    def test_mic_button_owns_its_touch_gesture(self):
        css = (_ROOT / "interface" / "static" / "style.css").read_text()
        mic = css.split("#mic-btn {", 1)[1].split("}", 1)[0]
        # Hold-to-talk must not fight text-selection / iOS callout / scroll.
        assert "touch-action: none" in mic
        assert "-webkit-touch-callout: none" in mic
        assert "user-select: none" in mic

    def test_mic_button_has_full_touch_lifecycle(self):
        js = (_ROOT / "interface" / "static" / "app.js").read_text()
        for evt in ("touchstart", "touchend", "touchcancel"):
            assert f"micBtn.addEventListener('{evt}'" in js, evt

    def test_stylesheet_cache_buster_bumped(self):
        html = (_ROOT / "interface" / "static" / "index.html").read_text()
        assert "style.css?v=2.12" not in html
