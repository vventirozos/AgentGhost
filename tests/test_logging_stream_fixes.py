"""§LOG (2026-08-20) — pins for the operator-stream logging fixes.

The logging audit (two lenses over 716 pretty_log sites + 16 days of live
mirror) found: the mirror's SYSTEM deltas destroyed by a second stateful
_format_delta read (LOG-2); no origin on the console frames — 45 of 87
turns were self-play wearing a user turn's clothes (LOG-4); the final
reply never logged anywhere (LOG-3); a pre-frame-era bare print() dump
bypassing lock/mirror/redaction (LOG-1); no dispatch-level tool line and
strike lines at INFO losing the 240-char failure budget (LOG-5); no
console liveness during deep idle + a WARNING that re-fired on every
dev-box save (LOG-6); anonymous "agent" bridge titles and a metacog icon
registry bypass with three glyph collisions (LOG-7).
"""

import asyncio
import datetime
import importlib.util
import inspect
import logging as _pylogging
import re
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ghost_agent.utils.logging as glog
from ghost_agent.utils.logging import (
    Icons, pretty_log, request_id_context, _PrettyLogHandler,
)

REPO = Path(__file__).resolve().parents[1]


class _CaptureHandler(_pylogging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def mirror_capture():
    """Wire a capture handler in as the durable mirror logger."""
    lg = _pylogging.getLogger("GhostStreamTestCapture")
    lg.setLevel(_pylogging.DEBUG)
    lg.propagate = False
    h = _CaptureHandler()
    lg.addHandler(h)
    old = glog._MIRROR_LOGGER
    glog._MIRROR_LOGGER = lg
    try:
        yield h
    finally:
        glog._MIRROR_LOGGER = old
        lg.removeHandler(h)


# ── LOG-2: the mirror records the delta the console computed ────────────────

class TestMirrorDelta:
    def test_system_delta_survives_into_the_mirror(self, mirror_capture,
                                                   capsys):
        # The defect: _mirror re-called _format_delta, whose SYSTEM read
        # is STATEFUL (advances the anchor) — so the durable record's
        # SYSTEM deltas were all ~+0.00s while the console showed the
        # real idle step-durations (max mirror SYSTEM delta in 7 live
        # days: +0.08s vs +2700s on console).
        tok = request_id_context.set("SYSTEM")
        try:
            glog.reset_system_delta_anchor()
            pretty_log("Mirror Delta Anchor", "first line anchors")
            time.sleep(0.12)
            pretty_log("Mirror Delta Probe", "step content")
        finally:
            request_id_context.reset(tok)
        rec = [r for r in mirror_capture.records
               if "mirror delta probe" in r.getMessage()]
        assert rec, "mirror missed the line entirely"
        m = re.search(r"\+([\d.]+)s", rec[-1].getMessage())
        assert m, rec[-1].getMessage()
        assert float(m.group(1)) >= 0.1, (
            "the mirror re-read the SYSTEM anchor after the console "
            "consumed it — durable idle timing destroyed")

    def test_format_delta_read_once_per_line(self, mirror_capture, capsys):
        calls = {"n": 0}
        real = glog._format_delta

        def counting(req_id):
            calls["n"] += 1
            return real(req_id)

        with patch.object(glog, "_format_delta", counting):
            tok = request_id_context.set("SYSTEM")
            try:
                pretty_log("Delta Count Probe", "x")
            finally:
                request_id_context.reset(tok)
        assert calls["n"] == 1, (
            f"_format_delta ran {calls['n']}x for one line — the second "
            "stateful read is the LOG-2 defect")


# ── LOG-4: origin on the console frames, parser-safe ─────────────────────────

class TestFrameOrigin:
    def _strip(self, s):
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    def test_begin_and_end_frames_carry_origin(self, mirror_capture, capsys):
        tok = request_id_context.set("simreq99")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="sim")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        out = self._strip(capsys.readouterr().out)
        lines = [l for l in out.splitlines() if l.strip()]
        begin = next(l for l in lines if "request started" in l)
        end = next(l for l in lines if "request finished" in l)
        assert begin.rstrip().endswith("· sim"), begin
        assert end.rstrip().endswith("· sim"), (
            "END must close with the origin BEGIN opened with — the "
            "operator glancing at a closing frame deserves the same truth")

    def test_frames_without_origin_are_unchanged(self, mirror_capture,
                                                 capsys):
        tok = request_id_context.set("plainreq")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        out = self._strip(capsys.readouterr().out)
        assert "·" not in out.split("request started")[1].splitlines()[0]

    def test_client_parsers_still_match_the_new_frames(self, mirror_capture,
                                                       capsys):
        # Load the REAL uConsole parser module and feed it the new frame.
        spec = importlib.util.spec_from_file_location(
            "turnstatus", REPO / "interface" / "externals"
            / "clockwork_ghost" / "turnstatus.py")
        ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ts)
        tok = request_id_context.set("benchr42")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="bench")
        finally:
            request_id_context.reset(tok)
        line = self._strip(capsys.readouterr().out).splitlines()[0]
        assert "request started" in line          # substring both clients use
        m = ts._HEADER_ID_RE.match(line)
        assert m, "uConsole header regex no longer matches the frame"

    def test_mirror_end_line_names_origin(self, mirror_capture, capsys):
        tok = request_id_context.set("simreq77")
        try:
            pretty_log("Request Initialized", special_marker="BEGIN",
                       origin="sim")
            pretty_log("Request Finished", special_marker="END")
        finally:
            request_id_context.reset(tok)
        end = [r for r in mirror_capture.records
               if "request finished" in r.getMessage()]
        assert end and "origin=sim" in end[-1].getMessage()


# ── LOG-1: the bare-print thinking dump is gone ──────────────────────────────

class TestBarePrintDumpDeleted:
    def test_agent_source_has_no_raw_think_banner(self):
        # Deletion pin (the defect is the very PRESENCE of this shape: a
        # raw print() around the stdout lock, the mirror and _redact_log).
        src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text(
            encoding="utf-8")
        assert "AGENT INTERNAL THINKING" not in src
        assert 'print(f"[INFO ] 💭' not in src


# ── LOG-3: the final reply reaches the log ───────────────────────────────────

class TestFinalReplyLogged:
    @pytest.mark.asyncio
    async def test_turn_logs_final_reply(self, tmp_path):
        from tests.test_bench_handle_chat_1c import (_bench_context,
                                                     _FakeBgTasks)
        from ghost_agent.core.agent import GhostAgent
        context, _ = _bench_context(tmp_path)
        agent = GhostAgent(context)
        calls = []
        real_pl = glog.pretty_log

        def spy(title, content=None, **kw):
            calls.append((str(title), content))
        with patch("ghost_agent.core.agent.pretty_log", side_effect=spy):
            await agent.handle_chat(
                {"model": "test-model", "stream": False,
                 "messages": [{"role": "user", "content": "average a list"}]},
                _FakeBgTasks(), request_id="logpin-1")
        finals = [c for t, c in calls if t == "Final Reply"]
        assert finals, "the turn's answer never reached the log"
        assert "All done — solved." in str(finals[-1]), finals[-1]


# ── LOG-5: dispatch line, unknown-tool warning, strike levels ────────────────

class TestDispatchVisibility:
    @pytest.mark.asyncio
    async def test_every_executed_tool_gets_a_dispatch_line(self):
        from tests.test_dispatch_pipeline_extraction import (_make_agent,
                                                             _make_ts)
        agent = _make_agent()

        async def probe(**kwargs):
            return "ok"
        agent.available_tools = {"recall": probe}
        from ghost_agent.core.strikes import StrikeLedger
        ts = _make_ts(tool_calls=[{
            "id": "t1", "type": "function",
            "function": {"name": "recall",
                         "arguments": '{"query": "who am i"}'}}],
            strikes=StrikeLedger(), repeated_action_steered=set())
        calls = []
        with patch("ghost_agent.core.agent.pretty_log",
                   side_effect=lambda t, c=None, **k: calls.append((t, c, k))):
            await agent._dispatch_and_process_tool_batch(ts)
        disp = [c for c in calls if c[0] == "Tool Call"]
        assert disp, "an executed tool produced no dispatch line"
        assert "recall" in str(disp[0][1])
        assert "who am i" in str(disp[0][1])

    @pytest.mark.asyncio
    async def test_unknown_tool_logs_a_warning(self):
        from tests.test_dispatch_pipeline_extraction import (_make_agent,
                                                             _make_ts)
        agent = _make_agent()
        agent.available_tools = {}
        from ghost_agent.core.strikes import StrikeLedger
        ts = _make_ts(tool_calls=[{
            "id": "t1", "type": "function",
            "function": {"name": "made_up_tool", "arguments": "{}"}}],
            strikes=StrikeLedger(), repeated_action_steered=set())
        calls = []
        with patch("ghost_agent.core.agent.pretty_log",
                   side_effect=lambda t, c=None, **k: calls.append((t, c, k))):
            await agent._dispatch_and_process_tool_batch(ts)
        warn = [c for c in calls
                if c[0] == "Tool Warning" and "made_up_tool" in str(c[1])]
        assert warn, "hallucinated tool name rejected with no operator line"
        assert warn[0][2].get("level") == "WARNING"

    @pytest.mark.asyncio
    async def test_execution_strike_is_warning_level(self):
        # WARNING is what buys the 240-char failure budget — at INFO the
        # error preview died at 60 chars, exactly on the *why*.
        from tests.test_dispatch_pipeline_extraction import (_make_agent,
                                                             _make_ts)
        agent = _make_agent()

        async def failing(**kwargs):
            return ("Error: ModuleNotFoundError: no module named 'x' — "
                    "the import failed hard and this preview is the why")
        agent.available_tools = {"execute": failing}
        from ghost_agent.core.strikes import StrikeLedger
        ts = _make_ts(tool_calls=[{
            "id": "t1", "type": "function",
            "function": {"name": "execute",
                         "arguments": '{"command": "python x.py"}'}}],
            strikes=StrikeLedger(), repeated_action_steered=set())
        calls = []
        with patch("ghost_agent.core.agent.pretty_log",
                   side_effect=lambda t, c=None, **k: calls.append((t, c, k))):
            await agent._dispatch_and_process_tool_batch(ts)
        strikes = [c for c in calls
                   if c[0] in ("Execution Fail", "Transient Fail",
                               "Tool Warning")]
        assert strikes, "a failing tool produced no strike line"
        assert all(c[2].get("level") == "WARNING" for c in strikes), strikes


# ── LOG-6a: deep idle emits a console liveness line ──────────────────────────

class TestIdleCycleConsoleLine:
    @pytest.mark.asyncio
    async def test_idle_cycle_summary_reaches_pretty_log(self):
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=1200, memory_ids=0)
        _store = MagicMock()
        _store.list_projects.return_value = []
        agent.context.project_store = _store
        calls = []
        with patch("ghost_agent.core.dream.Dreamer"), \
             patch("ghost_agent.core.agent.pretty_log",
                   side_effect=lambda t, c=None, **k: calls.append((t, c))), \
             patch("ghost_agent.core.agent.random.random", return_value=0.99):
            await agent._biological_tick()
        idle = [c for t, c in calls if t == "Idle Cycle"]
        assert idle and "tidy" in str(idle[0]), (
            "deep idle still has no console liveness signal — a wedged "
            "loop and a quiet night look identical on the watched stream")


# ── LOG-6b: staleness warning per-file cooldown ──────────────────────────────

class TestStalenessCooldown:
    """§LOG-6b reconciled with the standing R33/R34 laws pinned in
    test_prm_online_update_loudness.py: every DISTINCT divergence still
    logs and still returns (R34 — path-keyed silence hid the
    edit/restore/edit cycle), and nothing is marked before a successful
    emit (R33). What the cooldown changes is the LEVEL: only the first
    divergence of a file per hour is a WARNING; repeats within the
    window are INFO — the warning color stops crying wolf."""

    def _drive(self, monkeypatch, digests_seq):
        from ghost_agent.core import staleness as st
        monkeypatch.setattr(st, "loaded_watched_files", lambda: ["x.py"])
        monkeypatch.setattr(st, "_DIGESTS_AT_LOAD", {"x.py": "d0"})
        monkeypatch.setattr(st, "_REPORTED", set())
        monkeypatch.setattr(st, "_FILE_LAST_WARNED", {})
        logged = []
        for dig in digests_seq:
            monkeypatch.setattr(st, "read_digests",
                                lambda only=None, _d=dig: {"x.py": _d})
            st.audit_source_newer_than_process(
                lambda m, level="WARNING": logged.append((m, level)))
        return logged, st

    def test_same_digest_reports_once(self, monkeypatch):
        logged, _ = self._drive(monkeypatch, ["d1", "d1", "d1"])
        assert len(logged) == 1     # per-digest dedup, unchanged law

    def test_rapid_edits_warn_once_then_inform(self, monkeypatch):
        # The live defect: 95 near-identical WARNINGs — every dev-box
        # SAVE minted a new digest. Each distinct divergence must STILL
        # be announced (R34), but only the first is a WARNING.
        logged, _ = self._drive(monkeypatch, ["d1", "d2", "d3"])
        assert [lv for _, lv in logged] == ["WARNING", "INFO", "INFO"], (
            logged)

    def test_cooldown_expiry_rearms_the_warning(self, monkeypatch):
        logged, st = self._drive(monkeypatch, ["d1", "d2"])
        st._FILE_LAST_WARNED["x.py"] -= (st._FILE_WARN_COOLDOWN + 1)
        monkeypatch.setattr(st, "read_digests",
                            lambda only=None: {"x.py": "d4"})
        st.audit_source_newer_than_process(
            lambda m, level="WARNING": logged.append((m, level)))
        assert [lv for _, lv in logged] == ["WARNING", "INFO", "WARNING"], (
            "the warning never re-arms after the cooldown")

    def test_raising_sink_leaves_the_next_audit_loud(self, monkeypatch):
        # R33 compliance for the NEW stamp too: a raising sink must not
        # consume the warn-window (else one bad log call downgrades the
        # divergence to INFO for an hour).
        from ghost_agent.core import staleness as st
        monkeypatch.setattr(st, "loaded_watched_files", lambda: ["x.py"])
        monkeypatch.setattr(st, "_DIGESTS_AT_LOAD", {"x.py": "d0"})
        monkeypatch.setattr(st, "_REPORTED", set())
        monkeypatch.setattr(st, "_FILE_LAST_WARNED", {})
        monkeypatch.setattr(st, "read_digests",
                            lambda only=None: {"x.py": "d1"})

        def _boom(m, level="WARNING"):
            raise RuntimeError("sink down")
        try:
            st.audit_source_newer_than_process(_boom)
        except RuntimeError:
            pass
        assert "x.py" not in st._FILE_LAST_WARNED
        logged = []
        st.audit_source_newer_than_process(
            lambda m, level="WARNING": logged.append(level))
        assert logged == ["WARNING"]


# ── LOG-7: bridge titles name the module; metacog icons registered ───────────

class TestBridgeTitle:
    def test_bridged_warning_names_the_module(self, mirror_capture, capsys):
        h = _PrettyLogHandler()
        rec = _pylogging.LogRecord(
            name="GhostAgent", level=_pylogging.WARNING,
            pathname="/x/llm.py", lineno=1,
            msg="circuit breaker OPEN for worker", args=(), exc_info=None)
        rec.module = "llm"
        calls = []
        with patch("ghost_agent.utils.logging.pretty_log",
                   side_effect=lambda t, c=None, **k: calls.append(t)):
            h.emit(rec)
        assert calls == ["Agent·llm"], (
            f"{calls} — 230 live warnings rendered as the anonymous "
            "title 'agent', including background-task deaths")


class TestMetacogIconsRegistered:
    def test_map_values_come_from_the_registry(self):
        from ghost_agent.core import metacog_log as mc
        # Must be the METACOG_* constants specifically — a literal that
        # happens to equal some OTHER registry glyph (the old 🧮 arbiter =
        # VECTOR_EMBED collision) would pass a mere membership check.
        metacog_glyphs = {v for k, v in vars(Icons).items()
                          if k.startswith("METACOG")}
        for sub, icon in mc._SUBSYSTEM_ICONS.items():
            assert icon in metacog_glyphs, (
                f"{sub} icon {icon} is not a METACOG_* registry glyph")
        assert mc._DEFAULT_ICON in metacog_glyphs
        # And the metacog glyphs collide with nothing else in the registry.
        others = {v for k, v in vars(Icons).items()
                  if isinstance(v, str) and not k.startswith("_")
                  and not k.startswith("METACOG")}
        assert not (metacog_glyphs & others), metacog_glyphs & others

    def test_no_glyph_collisions_anywhere_in_the_registry(self):
        vals = [v for k, v in vars(Icons).items()
                if isinstance(v, str) and not k.startswith("_")]
        dupes = {g for g in vals if vals.count(g) > 1}
        assert not dupes, (
            f"one glyph, two meanings: {dupes} — the operator sees only "
            "icon+title")

    def test_uconsole_map_covers_metacog_glyphs(self):
        src = (REPO / "interface" / "externals" / "clockwork_ghost"
               / "turnstatus.py").read_text(encoding="utf-8")
        for g in ("🫧", "🥇", "🚦", "📊", "📈", "📐", "🌱",
                  "🚧", "💻", "🚪"):
            assert g in src, f"uConsole ICON_CLASS misses {g}"
