"""§4RZ step 6b — the CLI client for queued messages.

The owner types while a turn runs: the reply streams on a background thread
and the main thread keeps a live prompt (interface/externals/cli/ghost,
_BusyView / _stream_busy). Enter POSTs /api/turn/inbox and renders by path
(inject / status / cancel / queue); /interrupt stops the running tool;
leftovers and held messages are sent after the reply. One-shot mode is
unchanged. The network is faked; real terminal rendering is not exercised
here (see the manual tmux smoke in the step-6b report).
"""

import contextlib
import copy
import importlib.machinery
import importlib.util
import io
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import pytest

REPO_CLI = (Path(__file__).resolve().parents[1]
            / "interface" / "externals" / "cli" / "ghost")
_CLI_OVERRIDE = os.environ.get("GHOST_CLI_PATH")       # mutation runs: a copy
CLI_PATH = Path(_CLI_OVERRIDE) if _CLI_OVERRIDE else REPO_CLI


def _load():
    loader = importlib.machinery.SourceFileLoader("ghost_cli_4rz", str(CLI_PATH))
    spec = importlib.util.spec_from_file_location("ghost_cli_4rz", CLI_PATH,
                                                  loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


cli = _load()
_THEME = dict(cli.console._theme_stack._entries[0])


class _Resp:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data
        self.text = ""

    def json(self):
        return self._data


class _FakeSSE:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(self._lines)


def _chunk(text):
    import json
    return ("data: " + json.dumps({"choices": [{"delta": {"content": text}}]})).encode()


@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(
        file=buf, force_terminal=True, width=100, height=30,
        theme=cli.Theme(_THEME), highlight=False))
    monkeypatch.setattr(cli, "HISTORY_FILE",
                        Path(tempfile.mkdtemp()) / "ghost_history")
    return buf


def _plain(buf):
    import re
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buf.getvalue())


# --- client methods -----------------------------------------------------------

class TestClientMethods:
    def _api(self, monkeypatch, status=200, data=None):
        api = cli.GhostAPI("http://localhost:9", "k")
        seen = []

        def request(method, url, json=None, params=None, files=None, timeout=None):
            seen.append({"method": method, "url": url, "json": json, "timeout": timeout})
            return _Resp(status, data if data is not None else {})

        monkeypatch.setattr(api.http, "request", request)
        return api, seen

    def test_inbox_posts_session_and_text(self, monkeypatch):
        api, seen = self._api(monkeypatch, data={"path": "inject", "request_id": "r1"})
        assert api.turn_inbox("sid1", "use metric") == (200, {"path": "inject", "request_id": "r1"})
        assert seen == [{"method": "POST", "url": "http://localhost:9/api/turn/inbox",
                         "json": {"session_id": "sid1", "text": "use metric"},
                         "timeout": cli.INBOX_TIMEOUT}]

    def test_interrupt_posts_session(self, monkeypatch):
        api, seen = self._api(monkeypatch, status=409, data={"interrupted": []})
        assert api.turn_interrupt("sid1")[0] == 409
        assert seen[0]["url"].endswith("/api/turn/interrupt")
        assert seen[0]["json"] == {"session_id": "sid1"}

    def test_leftovers_posts_session(self, monkeypatch):
        api, seen = self._api(monkeypatch, data={"messages": ["a"]})
        assert api.inbox_leftovers("sid1") == (200, {"messages": ["a"]})
        assert seen[0]["method"] == "POST"
        assert seen[0]["url"].endswith("/api/turn/inbox/leftovers")
        assert seen[0]["json"] == {"session_id": "sid1"}

    def test_timeouts_are_short(self):
        """The operator waits on these mid-turn; a hung agent must not
        freeze the prompt for the default (10, 60)."""
        assert sum(cli.INBOX_TIMEOUT) <= 10


# --- path dispatch --------------------------------------------------------------

class TestBusyLine:
    def _cli(self, monkeypatch, reply=None, exc=None):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="sid9")
        calls = []

        def inbox(sid, text):
            calls.append((sid, text))
            if exc:
                raise exc
            return reply

        monkeypatch.setattr(g.api, "turn_inbox", inbox)
        return g, calls

    def test_inject_prints_a_dim_queued_line(self, monkeypatch, out):
        g, calls = self._cli(monkeypatch, (200, {"path": "inject", "request_id": "r"}))
        assert g._busy_line("use metric units") == ("", None)
        assert calls == [("sid9", "use metric units")]
        assert "queued: use metric units" in _plain(out)
        assert "\x1b[2m" in out.getvalue()               # dim
        assert g._held == []

    def test_status_answers_inline(self, monkeypatch, out):
        g, _ = self._cli(monkeypatch, (200, {"path": "status", "request_id": "r",
                                             "reply": "Working for 42 s, step 3"}))
        assert g._busy_line("how's it going?") == ("", None)
        text = _plain(out)
        assert "how's it going?" in text and "Working for 42 s, step 3" in text
        assert g._held == []

    def test_cancel_returns_undelivered(self, monkeypatch, out):
        g, _ = self._cli(monkeypatch, (200, {"path": "cancel", "request_id": "r",
                                             "undelivered": ["also b"]}))
        assert g._busy_line("stop") == ("cancel", ("stop", ["also b"]))
        assert g._held == []

    def test_queue_holds_with_reason(self, monkeypatch, out):
        g, _ = self._cli(monkeypatch, (200, {"path": "queue",
                                             "reason": "the turn is finishing"}))
        assert g._busy_line("later please") == ("", None)
        assert g._held == ["later please"]
        assert "the turn is finishing" in _plain(out)

    @pytest.mark.parametrize("reply", [
        (200, {"path": "queue", "reason": "queued messages are off"}),
        (404, {"detail": "Not Found"}),                      # an older agent
    ])
    def test_no_inbox_is_never_auto_sent(self, monkeypatch, out, reply):
        """r1 #2: feature off / old server — a mid-turn "stop" must not
        arrive later as a fresh request; it goes back into the prompt."""
        g, _ = self._cli(monkeypatch, reply)
        assert g._busy_line("stop") == ("", None)
        assert g._held == []
        assert g._take_draft() == "stop"
        assert "queued messages are off" in _plain(out)

    def test_unknown_path_is_held(self, monkeypatch, out):
        g, _ = self._cli(monkeypatch, (200, {"path": "something-new"}))
        g._busy_line("x")
        assert g._held == ["x"]

    def test_network_error_holds_locally(self, monkeypatch, out):
        g, _ = self._cli(monkeypatch, exc=cli.requests.exceptions.ConnectTimeout("down"))
        assert g._busy_line("keep me") == ("", None)
        assert g._held == ["keep me"]
        assert "couldn't reach the agent" in _plain(out)

    def test_http_error_holds_locally(self, monkeypatch, out):
        g, _ = self._cli(monkeypatch, (500, {"raw": "boom"}))
        g._busy_line("keep me")
        assert g._held == ["keep me"]

    def test_interrupt_command_is_not_posted_to_the_inbox(self, monkeypatch, out):
        g, calls = self._cli(monkeypatch, (200, {"path": "inject"}))
        hit = []
        monkeypatch.setattr(g, "cmd_interrupt", lambda: hit.append(1))
        assert g._busy_line("/interrupt") == ("", None)
        assert hit == [1] and calls == []

    def test_other_commands_go_back_to_the_prompt(self, monkeypatch, out):
        """/clear or /good mid-turn would change the conversation the turn
        is answering — they wait, kept in the prompt, never sent."""
        g, calls = self._cli(monkeypatch, (200, {"path": "inject"}))
        assert g._busy_line("/clear") == ("draft", "/clear")
        assert calls == [] and g._held == []


# --- whole busy turns ---------------------------------------------------------------

class _Harness:
    """GhostCLI in busy mode: no terminal, no patch_stdout; the stream is a
    fake SSE that pauses (keep-alive lines) until the scripted prompt has
    typed its lines; the prompt is a script."""

    def __init__(self, monkeypatch, lines=(), draft_at_end="",
                 inbox=None, leftovers=(), finish_on_gate=True, abort=False,
                 error=False, sse=None):
        self.gate = threading.Event()
        self.views = []
        self.api = cli.GhostAPI("http://localhost:9", "k")
        self.g = cli.GhostCLI(self.api, session_id="sidX")
        self.inbox_calls = []
        self.cancel_calls = []
        self.streams = []
        self.prompt_defaults = []
        g = self.g
        monkeypatch.setattr(g, "_busy_enabled", lambda: True)
        monkeypatch.setattr(g, "_busy_stdout", contextlib.nullcontext)
        stream_api = copy.copy(self.api)

        def chat_stream(messages, session_id, request_id, effort=""):
            self.rid = request_id            # the agent keeps it as turn.req_id
            self.streams.append([m["content"] for m in messages])
            if sse is not None:
                return _FakeSSE(self._custom(sse))
            return _FakeSSE(self._sse(finish_on_gate, error))

        stream_api.chat_stream = chat_stream
        monkeypatch.setattr(g, "_turn_api", lambda: stream_api)

        def turn_inbox(sid, text):
            self.inbox_calls.append((sid, text))
            ans = dict((inbox or {}).get(text, {"path": "inject", "request_id": "r"}))
            if ans.get("request_id") == "r":     # "r" = this turn, as the agent answers
                ans["request_id"] = self.rid
            return 200, ans

        monkeypatch.setattr(self.api, "turn_inbox", turn_inbox)
        monkeypatch.setattr(self.api, "inbox_leftovers",
                            lambda sid: (200, {"messages": list(leftovers)}))
        monkeypatch.setattr(self.api, "cancel_turn", lambda **k: (
            self.cancel_calls.append(k),
            (200, {"cancelled": True, "undelivered": ["unread one"]}))[1])
        script = list(lines)

        def busy_prompt(view, draft):
            self.views.append(view)
            self.prompt_defaults.append(draft)
            if script:
                return script.pop(0)
            if abort:
                return cli._Abort("half typed")
            self.gate.set()
            assert view.done.wait(5), "stream never ended"
            return cli._Ended(draft_at_end)

        monkeypatch.setattr(g, "_busy_prompt", busy_prompt)

    def _custom(self, lines):
        yield lines[0]
        assert self.gate.wait(5)            # the scripted lines are typed first
        yield from lines[1:]

    def _sse(self, finish_on_gate, error=False):
        yield _chunk("first block\n\n")
        yield _chunk("open tail")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if finish_on_gate and self.gate.is_set():
                break
            yield b""                    # keep-alive: the loop re-checks detach
            time.sleep(0.005)
        if error:
            yield b'data: {"error": {"message": "boom"}}'
            return
        yield _chunk(" done")
        yield b"data: [DONE]"


class TestBusyTurn:
    def test_held_go_to_the_outbox_and_leftovers_to_the_prompt(
            self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric", "after this"],
                     inbox={"after this": {"path": "queue", "reason": "the turn is finishing"}},
                     leftovers=["accepted but unread"], draft_at_end="half")
        h.g.chat_turn("question")
        assert h.inbox_calls == [("sidX", "use metric"), ("sidX", "after this")]
        assert h.g.history[-1] == {"role": "assistant",
                                   "content": "first block\n\nopen tail done"}
        assert h.g.turns[-1]["interrupted"] == ""
        # RULE (r4): a leftover is never sent on its own — named, back in the
        # prompt; only the held line (it never reached the agent) is sent
        assert list(h.g._outbox) == ["after this"]
        assert h.g._take_draft().split("\n") == ["accepted but unread", "half"]
        text = " ".join(_plain(out).split())
        assert "not read by the agent — back in the prompt: «accepted but unread»" in text
        assert "first block" in text and "open tail done" in text
        assert "queued: use metric" in text

    def test_reply_text_is_printed_once(self, monkeypatch, out):
        h = _Harness(monkeypatch)
        h.g.chat_turn("q")
        text = _plain(out)
        assert text.count("first block") == 1 and text.count("open tail done") == 1

    def test_feature_off_puts_lines_back_not_sent(self, monkeypatch, out):
        off = {"path": "queue", "reason": "queued messages are off"}
        h = _Harness(monkeypatch, lines=["one", "two"], inbox={"one": off, "two": off})
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["one", "two"]

    def test_errored_turn_does_not_auto_send(self, monkeypatch, out):
        """r1 #3: after an error, held messages come back into the prompt
        (with older queued ones first) instead of being sent alone."""
        h = _Harness(monkeypatch, lines=["follow-up"], error=True,
                     inbox={"follow-up": {"path": "queue", "reason": "the turn is finishing"}},
                     leftovers=["left"])
        h.g._outbox.append("older")
        h.g.chat_turn("q")
        assert h.g.turns[-1]["interrupted"] == "error"
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["older", "left", "follow-up"]

    def test_ctrl_c_during_leftovers_loses_nothing(self, monkeypatch, out):
        """r1 #4a: Ctrl-C while the leftovers POST runs is held until it
        answers; it then means stop, so everything goes back to the prompt."""
        h = _Harness(monkeypatch, lines=["held"],
                     inbox={"held": {"path": "queue", "reason": "the turn is finishing"}})

        def leftovers(sid):
            os.kill(os.getpid(), signal.SIGINT)
            return 200, {"messages": ["left"]}

        monkeypatch.setattr(h.api, "inbox_leftovers", leftovers)
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["left", "held"]

    def test_ctrl_c_during_an_inbox_post_keeps_the_line_and_stops(self, monkeypatch, out):
        """r1 #4b: the post finishes (the line is held, not dropped), then
        the Ctrl-C stops the turn on the agent."""
        h = _Harness(monkeypatch, lines=["keep me"], finish_on_gate=False)

        def inbox(sid, text):
            os.kill(os.getpid(), signal.SIGINT)
            return 200, {"path": "queue", "reason": "the turn is finishing"}

        monkeypatch.setattr(h.api, "turn_inbox", inbox)
        h.g.chat_turn("q")
        assert len(h.cancel_calls) == 1
        assert h.g.turns[-1]["interrupted"] == "cancelled"
        assert h.g._take_draft().split("\n") == ["unread one", "keep me"]

    def test_second_ctrl_c_during_the_stop_still_cancels(self, monkeypatch, out):
        """r1 #4c: a Ctrl-C while the stop waits for the stream thread must
        not escape before the cancel reaches the agent."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        real = cli._BusyView.detach

        def detach(self):
            os.kill(os.getpid(), signal.SIGINT)
            return real(self)

        monkeypatch.setattr(cli._BusyView, "detach", detach)
        try:
            h.g.chat_turn("q")
        except KeyboardInterrupt:
            pytest.fail("the 2nd Ctrl-C escaped before the cancel")
        assert len(h.cancel_calls) == 1

    def test_ctrl_c_while_waiting_for_the_thread_stops_the_turn(self, monkeypatch, out):
        """r1 #1: the fallback join after the prompt ended is interruptible
        and goes through the stop path (it was an untimed join())."""
        h = _Harness(monkeypatch, finish_on_gate=False)

        def busy_prompt(view, draft):
            view.done.set()                  # as a straggler's finish would
            threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
            return cli._Ended("")

        monkeypatch.setattr(h.g, "_busy_prompt", busy_prompt)
        h.g.chat_turn("q")
        assert len(h.cancel_calls) == 1
        assert h.g.turns[-1]["interrupted"] == "cancelled"

    def test_network_error_on_inbox_holds_and_sends_after(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["offline msg"])
        monkeypatch.setattr(h.api, "turn_inbox", lambda sid, text: (_ for _ in ()).throw(
            cli.requests.exceptions.ConnectTimeout("x")))
        h.g.chat_turn("q")
        assert list(h.g._outbox) == ["offline msg"]

    def test_inbox_cancel_is_the_ctrl_c_path(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["held first", "stop"], finish_on_gate=False,
                     inbox={"held first": {"path": "queue", "reason": "x"},
                            "stop": {"path": "cancel", "request_id": "r",
                                     "undelivered": ["use metric"]}},
                     leftovers=[])
        h.g.chat_turn("q")
        assert h.cancel_calls == []                 # the inbox already stopped it
        assert h.g.turns[-1]["interrupted"] == "cancelled"
        # the partial reply stays, the open block printed once
        assert h.g.history[-1]["content"].startswith("first block")
        text = _plain(out)
        assert text.count("open tail") == 1
        assert "stopped" in text
        # undelivered + held come back into the prompt, nothing auto-sent
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["use metric", "held first"]
        text = _plain(out)
        # the stop is echoed below the partial reply, and the hand-back is
        # announced once (tmux smoke: it came above, and twice)
        assert text.index("open tail") < text.index("› stop") < text.index("stopped")
        assert text.count("back in the prompt") == 1
        assert "2 unsent messages back in the prompt" in text

    def test_ctrl_c_cancels_the_turn_by_its_request_id(self, monkeypatch, out):
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        h.g.chat_turn("q")
        assert len(h.cancel_calls) == 1 and h.cancel_calls[0]["request_id"]
        assert h.g.turns[-1]["interrupted"] == "cancelled"
        # the cancel's undelivered first, then the typed draft
        assert h.g._take_draft().split("\n") == ["unread one", "half typed"]
        assert "aborted" in _plain(out)

    def test_a_stop_also_stops_the_queue(self, monkeypatch, out):
        """Ctrl-C on a held message's turn must not fire the next held one."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        h.g._outbox.extend(["older held 1", "older held 2"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        # typing order (r1 #5): earlier turns' held ones first
        assert h.g._take_draft().split("\n") == [
            "older held 1", "older held 2", "unread one", "half typed"]

    def test_stopped_worker_prints_nothing_after_the_stop(self, monkeypatch, out):
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        h.g.chat_turn("q")
        time.sleep(0.05)
        assert " done" not in _plain(out)          # the stream's later text is muted

    def test_turn_state_comes_from_the_worker(self, monkeypatch, out):
        """The stream runs on a COPY of the CLI: a worker still unwinding
        after a stop can never write into the next turn's state."""
        h = _Harness(monkeypatch)
        h.g.chat_turn("q")
        assert h.g._turn_interrupted == ""
        assert h.views and h.views[0].done.is_set()

    def test_held_messages_are_sent_by_the_repl_after_the_turn(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["later"],
                     inbox={"later": {"path": "queue", "reason": "off"}})
        prompts = iter(["question", None])

        class _FakePrompt:
            def __init__(self, **kw):
                self.history = None

            def prompt(self, fragments, **kw):
                return next(prompts)

        monkeypatch.setattr(cli, "PromptSession", _FakePrompt)
        monkeypatch.setattr(h.api, "health", lambda timeout=(2, 5): (200, {"uptime_s": 1}))
        h.g.chat()
        assert len(h.streams) == 2
        assert h.streams[0][-1] == "question"
        assert h.streams[1][-1] == "later"          # sent as a normal message
        assert h.streams[1][-2].startswith("first block")
        assert not h.g._outbox
        assert "sent after the reply" in _plain(out)

    def test_draft_goes_back_into_the_idle_prompt(self, monkeypatch, out):
        h = _Harness(monkeypatch, draft_at_end="unfinished thought")
        seen = []
        prompts = iter(["question", None])

        class _FakePrompt:
            def __init__(self, **kw):
                self.history = None

            def prompt(self, fragments, **kw):
                seen.append(kw.get("default"))
                return next(prompts)

        monkeypatch.setattr(cli, "PromptSession", _FakePrompt)
        monkeypatch.setattr(h.api, "health", lambda timeout=(2, 5): (200, {"uptime_s": 1}))
        h.g.chat()
        assert seen == [None, "unfinished thought"]


class TestClassicCtrlC:
    def test_classic_abort_keeps_the_old_behaviour(self, monkeypatch, out):
        """r2 nit c: the classic path never posts to the inbox, so a
        cancel's undelivered came from ANOTHER client on this session — not
        put into this prompt (the old CLI's behaviour, exactly)."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        seen = []
        monkeypatch.setattr(g.api, "cancel_turn", lambda **k: (
            seen.append(k), (200, {"cancelled": True, "undelivered": ["x", "y"]}))[1])
        g._abort_turn("rid")
        assert g._take_draft() == ""
        assert seen == [{"request_id": "rid"}]          # default timeout, as before

    def test_busy_abort_restores(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        monkeypatch.setattr(g.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["x", "y"]}))
        g._abort_turn("rid", restore=True)
        assert g._take_draft() == "x\ny"


# --- /interrupt -----------------------------------------------------------------

class TestInterrupt:
    def _run(self, monkeypatch, out, status, data=None, exc=None):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s1")
        seen = []

        def interrupt(sid):
            seen.append(sid)
            if exc:
                raise exc
            return status, data or {}

        monkeypatch.setattr(g.api, "turn_interrupt", interrupt)
        assert g.dispatch("/interrupt") is True
        assert seen == ["s1"]
        return _plain(out)

    def test_200_names_the_tools(self, monkeypatch, out):
        text = self._run(monkeypatch, out, 200,
                         {"request_id": "r", "interrupted": ["run_code", "web_search"]})
        assert "stopped run_code, web_search" in text and "goes on" in text

    def test_409_is_nothing_running(self, monkeypatch, out):
        assert "nothing running" in self._run(monkeypatch, out, 409, {"interrupted": []})

    def test_404_no_turn(self, monkeypatch, out):
        text = self._run(monkeypatch, out, 404, {"reason": "no running turn for this sender"})
        assert "no running turn" in text

    def test_404_feature_off(self, monkeypatch, out):
        text = self._run(monkeypatch, out, 404, {"reason": "queued messages are off"})
        assert "GHOST_QUEUED_MESSAGES" in text

    def test_network_error(self, monkeypatch, out):
        text = self._run(monkeypatch, out, 0, exc=cli.requests.ConnectionError("x"))
        assert "cannot reach the agent" in text

    def test_command_surface(self, out):
        # dispatch is driven by _run above; here: completion + /help list it
        assert "/interrupt" in cli.COMPLETER.options
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s1")
        assert g.dispatch("/help") is True
        assert "/interrupt" in _plain(out) and "stop the tool" in _plain(out)


# --- unchanged paths ---------------------------------------------------------------

def _forbid_queue_calls(monkeypatch, api):
    for name in ("turn_inbox", "turn_interrupt", "inbox_leftovers"):
        monkeypatch.setattr(api, name, lambda *a, **k: pytest.fail(
            "a queued-message endpoint was called"))


class TestUnchangedPaths:
    def test_one_shot_never_touches_the_inbox(self, monkeypatch, out):
        api = cli.GhostAPI("http://localhost:9", "k")
        _forbid_queue_calls(monkeypatch, api)
        posts = []

        def post(url, **kw):
            posts.append(url)
            return _FakeSSE([_chunk("hi"), b"data: [DONE]"])

        monkeypatch.setattr(api.http, "post", post)
        assert cli.one_shot(api, "q", None) == 0
        assert posts == ["http://localhost:9/v1/chat/completions"]

    def test_one_shot_source_has_no_busy_machinery(self):
        import ast
        tree = ast.parse(CLI_PATH.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "one_shot")
        names = {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(fn)}
        for word in ("_busy", "patch_stdout", "_outbox", "turn_inbox", "inbox_leftovers",
                     "turn_interrupt", "cmd_interrupt", "_stream_busy"):
            assert word not in names

    def test_classic_turn_without_a_prompt_session(self, monkeypatch, out):
        """No REPL prompt (or GHOST_CLI_BUSY_PROMPT=0): the reply streams on
        the main thread exactly as before, and nothing is posted."""
        api = cli.GhostAPI("http://localhost:9", "k")
        _forbid_queue_calls(monkeypatch, api)
        monkeypatch.setattr(api, "chat_stream", lambda *a, **k: _FakeSSE(
            [_chunk("plain reply"), b"data: [DONE]"]))
        g = cli.GhostCLI(api)
        assert g._busy_enabled() is False
        g.chat_turn("q")
        assert g.history[-1]["content"] == "plain reply"

    def test_opt_out_env(self, monkeypatch):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        g.prompt_session = object()
        monkeypatch.setenv("GHOST_CLI_BUSY_PROMPT", "0")
        assert g._busy_enabled() is False


# --- the busy view's rendering pieces --------------------------------------------

class TestBusyView:
    def test_printer_with_a_tail_sink_never_starts_a_live(self, monkeypatch, out):
        monkeypatch.setattr(cli, "Live", lambda *a, **k: pytest.fail("Live under the busy prompt"))
        tails = []
        p = cli._StreamPrinter(tail=tails.append)
        p.feed("done block\n\nopen")
        assert tails[-1] == "open"
        assert "done block" in _plain(out)
        p.settle()
        assert tails[-1] == ""                     # tail cleared at settle
        assert "open" in _plain(out)

    def test_muted_printer_prints_nothing(self, out):
        p = cli._StreamPrinter(tail=lambda t: None, muted=lambda: True)
        p.feed("a block\n\nmore")
        p.settle()
        assert out.getvalue() == ""
        assert p.text == "a block\n\nmore"          # still kept as the partial reply

    def test_detach_returns_the_unprinted_rest_and_mutes(self, out):
        v = cli._BusyView()
        p = v.new_printer()
        p.feed("printed\n\nnot yet")
        text, rest = v.detach()
        assert text == "printed\n\nnot yet" and rest == "not yet"
        before = out.getvalue()
        p.feed(" later\n\n")
        v.note("should not show")
        assert out.getvalue() == before

    def test_message_spinner_then_tail_then_prompt(self, monkeypatch):
        monkeypatch.setattr(cli.shutil, "get_terminal_size",
                            lambda fallback=(80, 24): os.terminal_size((80, 24)))
        v = cli._BusyView()
        v.spinner.start()
        frags = v.message()
        flat = "".join(t for _, t, *_ in frags)
        assert "thinking…" in flat and flat.endswith(cli.PROMPT_GLYPH)
        v.spinner.stop()
        v.set_tail("**partial** answer")
        flat = "".join(t for _, t, *_ in v.message())
        assert "partial answer" in flat and "thinking" not in flat
        assert flat.endswith(cli.PROMPT_GLYPH)

    def test_tail_goes_blank_while_the_width_changes(self, monkeypatch):
        size = {"cols": 80}
        monkeypatch.setattr(cli.shutil, "get_terminal_size",
                            lambda fallback=(80, 24): os.terminal_size((size["cols"], 24)))
        v = cli._BusyView()
        v.set_tail("some streaming text")
        assert "some streaming text" in "".join(t for _, t, *_ in v.message())
        size["cols"] = 70
        assert "some streaming text" not in "".join(t for _, t, *_ in v.message())
        v._quiet_until = 0                          # the width has held
        assert "some streaming text" in "".join(t for _, t, *_ in v.message())

    def test_finish_exits_a_running_prompt_with_its_draft(self):
        exits = []

        class _Loop:
            def call_soon_threadsafe(self, fn):
                fn()

        class _Fut:
            def done(self):
                return False

        class _App:
            is_running = True
            loop = _Loop()
            future = _Fut()
            current_buffer = type("B", (), {"text": "typing…"})()

            def exit(self, result=None):
                exits.append(result)

        v = cli._BusyView()
        v.app = _App()
        v.finish()
        assert v.done.is_set()
        assert len(exits) == 1 and isinstance(exits[0], cli._Ended)
        assert not isinstance(exits[0], cli._Abort)
        assert exits[0].text == "typing…"


class TestRealBusyPrompt:
    """The busy prompt through real prompt_toolkit key handling."""

    def _run(self, keys, done=False):
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        g = self.g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        g._probe_cache = cli.Text("● online")
        v = cli._BusyView()
        if done:
            v.done.set()
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            with create_app_session(input=pipe, output=DummyOutput()):
                return g._busy_prompt(v, "")

    def test_enter_returns_the_line(self):
        assert self._run("use metric\r") == "use metric"

    def test_busy_session_settings(self):
        """The frame holding the live tail is erased on exit (or every Enter
        leaves a copy of the tail in scrollback), and redraws are capped at
        rich Live's 12/s."""
        self._run("x\r")
        s = self.g._busy_session
        assert s.app.erase_when_done is True
        assert s.app.min_redraw_interval == pytest.approx(1 / 12)
        assert s.refresh_interval == cli.BUSY_REFRESH_S
        assert s.reserve_space_for_menu == 0

    def test_ctrl_c_is_a_stop_with_the_draft(self):
        res = self._run("half" + chr(3))
        assert isinstance(res, cli._Abort) and res.text == "half"

    def test_ctrl_d_does_not_exit_mid_turn(self):
        assert self._run(chr(4) + "x\r") == "x"

    def test_a_stragglers_finish_does_not_end_the_next_prompt(self):
        """r1 #1 (the reviewer's probe): turn 1's stream thread outlived
        its stop and finishes while turn 2's prompt runs on the SAME shared
        session — it must not end turn 2's prompt."""
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        g._probe_cache = cli.Text("● online")
        old, new = cli._BusyView(), cli._BusyView()
        with create_pipe_input() as pipe, \
                create_app_session(input=pipe, output=DummyOutput()):
            def straggler():
                time.sleep(0.4)
                old.app = g._busy_session.app
                old.finish()
                time.sleep(0.4)
                pipe.send_text("also check\r")
            threading.Thread(target=straggler, daemon=True).start()
            assert g._busy_prompt(new, "") == "also check"

    def test_a_turn_already_ended_exits_at_once(self):
        res = self._run("", done=True)
        assert isinstance(res, cli._Ended) and not isinstance(res, cli._Abort)


# --- review round 2: a POST whose answer never came may still have acted ---------

def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestNeverSent:
    def test_refused_connection_is_never_sent(self):
        api = cli.GhostAPI(f"http://127.0.0.1:{_free_port()}", "k")
        with pytest.raises(cli.requests.RequestException) as ei:
            api.turn_inbox("s", "x")
        assert cli._never_sent(ei.value) is True

    def test_connect_timeout_is_never_sent(self):
        assert cli._never_sent(cli.requests.exceptions.ConnectTimeout("x")) is True

    @pytest.mark.parametrize("exc", [
        cli.requests.exceptions.ReadTimeout("slow"),
        cli.requests.exceptions.ChunkedEncodingError("cut"),
        cli.requests.ConnectionError("reset mid-answer"),
    ])
    def test_anything_else_may_have_been_delivered(self, exc):
        assert cli._never_sent(exc) is False


class TestReadTimeouts:
    def _cli(self, monkeypatch):
        return cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")

    def test_inbox_read_timeout_is_not_resent(self, monkeypatch, out):
        """r2 #1: the agent got the line (and injected it) but answered
        late — never sent again on its own; back in the prompt, honestly."""
        g = self._cli(monkeypatch)
        monkeypatch.setattr(g.api, "turn_inbox", lambda sid, t: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        assert g._busy_line("check the price") == ("", None)
        assert g._held == []
        assert g._take_draft() == "check the price"
        text = _plain(out)
        assert "may have received" in text and "couldn't reach" not in text

    def test_inbox_read_timeout_in_a_whole_turn(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["slow one"])
        monkeypatch.setattr(h.api, "turn_inbox", lambda sid, t: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "slow one"

    def test_leftovers_read_timeout_is_never_silent(self, monkeypatch, out):
        """r2 #2: the agent hands leftovers out as it answers."""
        g = self._cli(monkeypatch)
        monkeypatch.setattr(g.api, "inbox_leftovers", lambda sid: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        g._after_busy_turn()
        assert "may not have been read" in _plain(out)

    def test_leftovers_unreachable_says_so(self, monkeypatch, out):
        g = self._cli(monkeypatch)
        monkeypatch.setattr(g.api, "inbox_leftovers", lambda sid: (_ for _ in ()).throw(
            cli.requests.exceptions.ConnectTimeout("down")))
        g._after_busy_turn()
        assert "couldn't fetch" in _plain(out)

    def test_leftovers_read_gets_room(self, monkeypatch):
        api = cli.GhostAPI("http://localhost:9", "k")
        seen = {}
        monkeypatch.setattr(api, "_json", lambda m, path, **k: (seen.update(k), (200, {}))[1])
        api.inbox_leftovers("s")
        assert seen["timeout"] == cli.LEFTOVERS_TIMEOUT
        assert cli.LEFTOVERS_TIMEOUT[1] >= 20


class TestStopEscapeHatch:
    def test_stop_cancel_uses_a_short_timeout(self, monkeypatch, out):
        """r2 #3: Ctrl-C is held while the stop's cancel runs — it must not
        be the default (10, 60)."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        h.g.chat_turn("q")
        assert h.cancel_calls[0]["timeout"] == cli.CANCEL_TIMEOUT
        assert sum(cli.CANCEL_TIMEOUT) <= 15

    def test_second_ctrl_c_during_the_cancel_gives_the_prompt_back(self, monkeypatch, out):
        """r2 #3: a hung agent — a second Ctrl-C ends the wait with the
        classic 'cancel not confirmed', and held text is kept."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        h.g._held.append("held earlier")

        def hung_cancel(**k):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(5)                     # the Ctrl-C must cut this short
            return 200, {"cancelled": True}

        monkeypatch.setattr(h.api, "cancel_turn", hung_cancel)
        t0 = time.monotonic()
        try:
            h.g.chat_turn("q")
        except KeyboardInterrupt:
            pytest.fail("the 2nd Ctrl-C escaped the REPL")
        assert time.monotonic() - t0 < 4
        assert "cancel not confirmed" in _plain(out)
        assert h.g._take_draft().split("\n") == ["held earlier", "half typed"]

    def test_second_ctrl_c_during_a_slow_inbox_post(self, monkeypatch, out):
        """A second Ctrl-C gives up the wait: the line comes back, the turn
        is stopped."""
        h = _Harness(monkeypatch, lines=["slow line"], finish_on_gate=False)

        def inbox(sid, text):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(5)
            return 200, {"path": "inject"}

        monkeypatch.setattr(h.api, "turn_inbox", inbox)
        t0 = time.monotonic()
        h.g.chat_turn("q")
        assert time.monotonic() - t0 < 4
        assert len(h.cancel_calls) == 1
        assert "slow line" in h.g._take_draft().split("\n")


class TestRound2Nits:
    def test_held_survive_an_interrupted_after_turn(self, monkeypatch, out):
        """nit a: held items not yet flushed are carried into the next turn
        (it used to reset them)."""
        h = _Harness(monkeypatch)
        h.g._held.append("from an interrupted turn")
        h.g.chat_turn("q")
        assert list(h.g._outbox) == ["from an interrupted turn"]

    def test_eof_mid_turn_stops_the_turn(self, monkeypatch, out):
        """nit b: stdin closed mid-turn — the turn is cancelled on the
        agent instead of the error escaping."""
        h = _Harness(monkeypatch, finish_on_gate=False)
        monkeypatch.setattr(h.g, "_busy_prompt",
                            lambda view, draft: (_ for _ in ()).throw(EOFError()))
        h.g.chat_turn("q")
        assert len(h.cancel_calls) == 1
        assert h.g.turns[-1]["interrupted"] == "cancelled"


# --- review round 3 -------------------------------------------------------------

class TestLostCancelAnswer:
    """MAJOR: the agent clears the inbox AS it cancels, so messages the turn
    accepted and had not read exist only in the cancel's answer. When that
    answer is never read they must be named and put back, never lost."""

    def _cli(self, monkeypatch, exc):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")

        def cancel(**kw):
            raise exc

        monkeypatch.setattr(g.api, "cancel_turn", cancel)
        monkeypatch.setattr(g.api, "inbox_leftovers", lambda sid: (200, {"messages": []}))
        return g

    def test_read_timeout_names_and_restores(self, monkeypatch, out):
        g = self._cli(monkeypatch, cli.requests.exceptions.ReadTimeout("slow"))
        g._typed = g._injected = ["also use metric"]
        g._open = [("also use metric", False)]
        g._abort_turn("rid", timeout=cli.CANCEL_TIMEOUT, restore=True)
        g._turn_interrupted = "cancelled"
        g._after_busy_turn()
        text = " ".join(_plain(out).split())
        assert "«also use metric»" in text and "may or may not have read" in text
        assert "did not answer the cancel" in text
        assert g._take_draft() == "also use metric"

    def test_never_sent_says_not_cancelled_and_restores_nothing(self, monkeypatch, out):
        """Refused: the cancel never left — the turn runs on and still
        reads (or hands back) what it accepted."""
        g = self._cli(monkeypatch, cli.requests.exceptions.ConnectTimeout("down"))
        g._open = [("also use metric", False)]
        g._abort_turn("rid", restore=True)
        assert "NOT cancelled" in _plain(out)
        assert g._take_draft() == ""

    def test_second_ctrl_c_on_the_cancel_restores(self, monkeypatch, out):
        g = self._cli(monkeypatch, KeyboardInterrupt())
        g._open = [("x", False)]
        g._abort_turn("rid", restore=True)
        assert "cancel not confirmed" in _plain(out)
        assert g._take_draft() == "x"

    def test_classic_path_message_unchanged(self, monkeypatch, out):
        g = self._cli(monkeypatch, cli.requests.exceptions.ReadTimeout("slow"))
        g._abort_turn("rid")
        assert "aborted locally — couldn't reach the agent to cancel" in _plain(out)
        assert g._take_draft() == ""

    def test_whole_turn_inject_then_stop_with_lost_answer(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n") == ["use metric", "half typed"]
        assert "may or may not have read" in _plain(out)

    def test_a_restored_message_that_comes_back_as_leftover_is_kept(
            self, monkeypatch, out):
        """The agent never got the cancel: the turn ended and handed the
        same message back as a leftover. r10: a server text is always
        restored (it may be another client's identical line) — a duplicate
        is acceptable, a loss is not."""
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False,
                     leftovers=["use metric", "other"])
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        h.g.chat_turn("q")
        d = h.g._take_draft().split("\n")
        assert d.count("use metric") == 2 and "other" in d


class TestStoppedElsewhere:
    """MINOR 1: a stop from another client ends the stream normally; the
    reply's own note (or a retract with no reply) is the only signal."""

    def test_cooperative_stop_note_restores_held(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["later"], leftovers=["left"],
                     inbox={"later": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("partial\n\n"),
                          _chunk("_(Turn cancelled: cancelled by user.)_"),
                          b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["left", "later"]
        assert "stopped on the agent" in _plain(out)

    def test_hard_stop_retract_with_no_reply_restores_held(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["later"],
                     inbox={"later": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("partial"), b'data: {"ghost": {"retract": true}}',
                          b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "q\nlater"   # r15: the stopped turn's own line too

    def test_a_normal_reply_still_sends(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["later"],
                     inbox={"later": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("the answer"), b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == ["later"]

    def test_note_regex_matches_the_agents_wording(self):
        """Pinned to core/agent.py's TurnCancelled handler wording: the
        agent's own f-string, rendered, is what the regex must match."""
        notes = _agent_fstrings(lambda lit: lit.startswith("_(Turn cancelled: "))
        assert len(notes) == 1
        note = _render(notes[0], _tc={"reason": "cancelled by user"})
        assert cli._STOPPED_NOTE_RE.search("x\n\n" + note)


def _agent_fstrings(pred):
    """The f-strings in core/agent.py whose literal parts satisfy ``pred``,
    as compiled expressions (AST, not text: a comment cannot satisfy it)."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
                      / "core" / "agent.py").read_text())
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr):
            lit = "".join(v.value for v in n.values if isinstance(v, ast.Constant))
            if pred(lit):
                found.append(compile(ast.Expression(n), "agent.py", "eval"))
    return found


def _render(code, **names):
    from types import SimpleNamespace
    return eval(code, {}, {k: (SimpleNamespace(**v) if isinstance(v, dict) else v)
                           for k, v in names.items()})


class TestRound3Small:
    def test_leftover_notice_is_honest_about_the_conversation(self, monkeypatch, out):
        """MINOR 2: leftovers are per conversation — /new etc. leave them."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "inbox_leftovers", lambda sid: (_ for _ in ()).throw(
            cli.requests.exceptions.ConnectTimeout("down")))
        g._after_busy_turn()
        text = _plain(out)
        assert "THIS conversation" in text and "next reply" not in text

    def test_ctrl_c_twice_after_an_inject_is_booked_once(self, monkeypatch, out):
        """nit a: Ctrl-C during the 'queued:' notice must not ALSO put the
        injected line back."""
        h = _Harness(monkeypatch, lines=["x"], finish_on_gate=False)
        real = cli.GhostCLI.notice

        def notice(msg, style="notice"):
            if msg.startswith("queued:"):
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGINT)
            real(msg, style)

        monkeypatch.setattr(h.g, "notice", notice)
        h.g.chat_turn("q")
        assert len(h.cancel_calls) == 1                     # the Ctrl-C stopped it
        # booked once: injected (not ALSO put back as in-flight); after the
        # stop it comes back once, named (r6: any stop restores accepted lines)
        assert h.g._take_draft().split("\n").count("x") == 1

    def test_ctrl_c_twice_after_a_hold_is_booked_once(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["x"], finish_on_gate=False,
                     inbox={"x": {"path": "queue", "reason": "the turn is finishing"}})
        real = cli.GhostCLI.notice

        def notice(msg, style="notice"):
            if msg.startswith("held:"):
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGINT)
            real(msg, style)

        monkeypatch.setattr(h.g, "notice", notice)
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("x") == 1

    def test_ctrl_c_while_sending_the_outbox_keeps_the_rest(self, monkeypatch, out):
        """nit b: popped only once it is really going; a Ctrl-C stops the
        queue and puts every remaining message back."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        monkeypatch.setattr(g.api, "health", lambda timeout=(2, 5): (200, {"uptime_s": 1}))
        g._outbox.extend(["a", "b"])
        real_print = cli.console.print
        calls = {"n": 0}

        def boom(*a, **k):
            if a and "sent after the reply" in str(getattr(a[0], "plain", a[0])):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise KeyboardInterrupt     # one Ctrl-C, on the first echo
            return real_print(*a, **k)

        defaults = []

        class _FakePrompt:
            def __init__(self, **kw):
                self.history = None

            def prompt(self, fragments, **kw):
                defaults.append(kw.get("default"))
                return None

        monkeypatch.setattr(cli, "PromptSession", _FakePrompt)
        sent = []
        monkeypatch.setattr(g, "chat_turn", lambda t: sent.append(t))
        monkeypatch.setattr(cli.console, "print", boom)
        g.chat()
        assert sent == [] and not g._outbox
        assert defaults == ["a\nb"]


# --- review round 4: leftovers are never auto-sent --------------------------------

class TestLeftoverRule:
    def test_timeout_restore_plus_its_own_leftover_is_not_sent(
            self, monkeypatch, out):
        """MAJOR 1 (r4 p1): put back after an inbox read timeout, then the
        same text handed back as a leftover — never sent. r10: a timed-out
        typing is not matched, so it shows twice (a duplicate, not a loss)."""
        h = _Harness(monkeypatch, lines=["use metric"], leftovers=["use metric"])
        monkeypatch.setattr(h.api, "turn_inbox", lambda sid, t: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "use metric\nuse metric"

    def test_a_leftover_from_a_later_fetch_is_never_sent(self, monkeypatch, out):
        """MAJOR 2: whenever it arrives, a leftover goes to the prompt."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "inbox_leftovers",
                            lambda sid: (200, {"messages": ["late one"]}))
        g._after_busy_turn()
        assert list(g._outbox) == []
        assert g._take_draft() == "late one"

    def test_held_unreachable_lines_are_still_sent(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["offline"], leftovers=["unread"])
        monkeypatch.setattr(h.api, "turn_inbox", lambda sid, t: (_ for _ in ()).throw(
            cli.requests.exceptions.ConnectTimeout("down")))
        h.g.chat_turn("q")
        assert list(h.g._outbox) == ["offline"]
        assert h.g._take_draft() == "unread"

    def test_duplicate_typed_lines_keep_their_own_order(self, monkeypatch, out):
        """NIT 5: a text typed twice — each occurrence its own index."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"))
        g._typed = ["yes", "b", "yes"]
        assert g._in_typing_order(["b", "yes", "yes", "other"]) == [
            "other", "yes", "b", "yes"]


class TestStoppedElsewhereR4:
    def test_remote_stop_names_and_restores_injected(self, monkeypatch, out):
        """MINOR 3 (r4 p4): the other client got `undelivered`; this one
        names what it sent and puts it back — and always says it stopped."""
        h = _Harness(monkeypatch, lines=["use metric"],
                     sse=[_chunk("partial\n\n"),
                          _chunk("_(Turn cancelled: cancelled by user.)_"),
                          b"data: [DONE]"])
        h.g.chat_turn("q")
        text = " ".join(_plain(out).split())
        assert "stopped on the agent" in text
        assert "«use metric»" in text and "may or may not have read" in text
        assert h.g._take_draft() == "use metric"

    def test_remote_stop_notice_even_with_nothing_typed(self, monkeypatch, out):
        h = _Harness(monkeypatch, sse=[_chunk("x\n\n_(Turn cancelled: cancelled by user.)_"),
                                       b"data: [DONE]"])
        h.g.chat_turn("q")
        assert "stopped on the agent" in _plain(out)

    def test_no_output_form_of_the_note(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["later"],
                     inbox={"later": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("_(Turn cancelled: cancelled by user.)_ No output was "
                                 "produced before the cancellation."), b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []

    def test_a_clean_empty_reply_is_not_a_stop_but_sends_nothing(self, monkeypatch, out):
        """r4 p2: no text and no retract (tool/image-only) is not called a
        stop — but r5: an empty reply is no positive evidence either, so the
        held line comes back instead of being sent."""
        h = _Harness(monkeypatch, lines=["later"],
                     inbox={"later": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[b'data: {"ghost": {"labelable": false}}', b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "later"
        text = _plain(out)
        assert "stopped on the agent" not in text and "ended without a reply" in text

    def test_a_reply_quoting_the_note_is_not_a_stop(self, monkeypatch, out):
        """MINOR 4b (r4 p3): the note mid-reply, then more text."""
        h = _Harness(monkeypatch, lines=["later"],
                     inbox={"later": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("The log said _(Turn cancelled: cancelled by user.)_ "
                                 "earlier.\n\nAll fine now."), b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == ["later"]

    def test_the_agent_still_writes_both_forms(self):
        note = _render(_agent_fstrings(lambda lit: lit.startswith("_(Turn cancelled: "))[0],
                       _tc={"reason": "cancelled by user"})
        bare = _agent_fstrings(lambda lit: lit.startswith(" No output was produced before the"))
        assert len(bare) == 1
        both = _render(bare[0], _note=note)
        assert both.startswith(note) and cli._STOPPED_NOTE_RE.search(both)


# --- review round 5: auto-send needs positive evidence ------------------------------

class TestPositiveEvidence:
    def test_hard_stop_with_nothing_streamed(self, monkeypatch, out):
        """MAJOR 1 (r5 A): a hard stop of a turn that streamed nothing sends
        only [DONE] — the held line is not sent and the injected one (it went
        into the OTHER client's undelivered) is named and put back."""
        h = _Harness(monkeypatch, lines=["use metric", "after"],
                     inbox={"after": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[b"", b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["use metric", "after"]
        text = " ".join(_plain(out).split())
        assert "«use metric»" in text and "may or may not have read" in text

    def test_retracted_draft_then_empty_reply_loses_nothing(self, monkeypatch, out):
        """MINOR 2 (r5 D): looks like a stop — the safe side: all back."""
        h = _Harness(monkeypatch, lines=["use metric", "after"],
                     inbox={"after": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("draft"), b'data: {"ghost": {"retract": true}}',
                          b"data: [DONE]"])
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["q", "use metric", "after"]  # r15

    def test_lost_stream_restores_held_and_injected(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric", "after"],
                     inbox={"after": {"path": "queue", "reason": "the turn is finishing"}},
                     sse=[_chunk("partial")])          # no [DONE]
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["use metric", "after"]

    def test_a_normal_reply_still_sends_and_does_not_restore_injected(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric", "after"],
                     inbox={"after": {"path": "queue", "reason": "the turn is finishing"}})
        h.g.chat_turn("q")
        assert list(h.g._outbox) == ["after"]
        assert h.g._take_draft() == ""

    def test_leftovers_answer_lost_restores_injected(self, monkeypatch, out):
        """MAJOR 2 (r5 C): the accepted lines existed only in that answer."""
        h = _Harness(monkeypatch, lines=["use metric"])
        monkeypatch.setattr(h.api, "inbox_leftovers", lambda sid: (_ for _ in ()).throw(
            cli.requests.exceptions.ReadTimeout("slow")))
        h.g.chat_turn("q")
        assert h.g._take_draft() == "use metric"
        assert "may or may not have read" in " ".join(_plain(out).split())

    def test_an_answered_cancel_still_restores_accepted_lines(self, monkeypatch, out):
        """r6 #4: an answer proves nothing (another client's cancel may have
        emptied the inbox into ITS answer): accepted lines always come back,
        deduped against `undelivered`."""
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False)
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n") == ["unread one", "use metric", "half typed"]


class TestHoldWords:
    @pytest.mark.parametrize("word", ["stop", "Stop!", "wait", "no", "hold on",
                                      "σταμάτα", "όχι, περίμενε"])
    def test_a_stop_or_hold_word_is_never_held(self, monkeypatch, out, word):
        """MINOR 1 (r5 B): typed before the turn registered (or behind the
        lock) it is answered 'queue' — sent later it would be a new turn."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "turn_inbox", lambda sid, t: (
            200, {"path": "queue", "reason": "no running turn for this sender"}))
        g._busy_line(word)
        assert g._held == []
        assert g._take_draft() == word
        assert "not sent" in _plain(out)

    def test_never_sent_stop_is_not_held_either(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "turn_inbox", lambda sid, t: (_ for _ in ()).throw(
            cli.requests.exceptions.ConnectTimeout("down")))
        g._busy_line("stop")
        assert g._held == [] and g._take_draft() == "stop"

    def test_an_ordinary_line_is_still_held(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "turn_inbox", lambda sid, t: (
            200, {"path": "queue", "reason": "no running turn for this sender"}))
        g._busy_line("also check the price")
        assert g._held == ["also check the price"]

    def test_the_word_lists_equal_the_agents(self):
        """The CLI copies them (it imports nothing from src): pinned equal."""
        import ast
        root = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

        def compiled(path, name):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and getattr(node.targets[0], "id", None) == name):
                    call = node.value
                    pattern = ast.literal_eval(call.args[0])
                    flags = 0
                    if len(call.args) > 1:
                        assert ast.unparse(call.args[1]) == "re.IGNORECASE"
                        flags = __import__("re").IGNORECASE
                    return pattern, flags
            raise AssertionError(f"{name} not found in {path}")

        assert compiled(root / "core" / "queued_messages.py", "_STOP_RE") == \
            (cli._STOP_RE.pattern, cli._STOP_RE.flags & __import__("re").IGNORECASE)
        assert compiled(root / "tools" / "memory.py", "_DENY_RE") == \
            (cli._DENY_RE.pattern, cli._DENY_RE.flags & __import__("re").IGNORECASE)
        from ghost_agent.memory.profile import _fold
        for t in ("Φωτεινή", "ΣΤΟΠ", "Straße", "café", "", None, 7, "STOP the Tools"):
            assert cli._fold(t) == _fold(t), t


# --- review round 6 ---------------------------------------------------------------

_QUEUE = {"path": "queue", "reason": "the turn is finishing"}


class TestRound6:
    def test_stale_clean_flag_never_reaches_a_raising_turn(self, monkeypatch, out):
        """MAJOR (r6 p1): the worker copy must not inherit the last turn's
        _turn_clean; a stream that raises past its handler is unclean."""
        monkeypatch.setattr(cli, "strip_orphan_think_close",
                            lambda t: (_ for _ in ()).throw(RuntimeError("render")))
        h = _Harness(monkeypatch, lines=["after this"], inbox={"after this": _QUEUE},
                     sse=[b'data: {"ghost": {"reasoning_unparsed": true}}',
                          _chunk("thinking"), b"data: [DONE]"])
        h.g._turn_clean = True                    # the previous turn's verdict
        h.g.chat_turn("q2")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "after this"

    def test_worker_starts_unclean(self, monkeypatch, out):
        h = _Harness(monkeypatch)
        seen = {}
        real = cli.GhostCLI.stream_reply

        def spy(self, request_id, busy=None):
            seen["clean"], seen["stopped"] = self._turn_clean, self._turn_stopped
            return real(self, request_id, busy=busy)

        monkeypatch.setattr(cli.GhostCLI, "stream_reply", spy)
        h.g._turn_clean = h.g._turn_stopped = True
        h.g.chat_turn("q")
        assert seen == {"clean": False, "stopped": False}

    @pytest.mark.parametrize("answer", [(502, {"error": "bad gateway"}), (200, ["not", "a dict"])])
    def test_only_a_200_dict_is_a_leftovers_answer(self, monkeypatch, out, answer):
        """r6 p2: no answer = unclean: held not sent, accepted lines back."""
        h = _Harness(monkeypatch, lines=["use metric", "after this"],
                     inbox={"after this": _QUEUE})
        monkeypatch.setattr(h.api, "inbox_leftovers", lambda sid: answer)
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["use metric", "after this"]
        assert "may not have been read" in _plain(out)

    def test_lines_after_a_hold_word_come_back_too(self, monkeypatch, out):
        """r6 p3: 'wait' then 'do it in greek' — neither is sent."""
        h = _Harness(monkeypatch, lines=["wait", "do it in greek"],
                     inbox={"wait": _QUEUE, "do it in greek": _QUEUE})
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["wait", "do it in greek"]

    def test_a_line_before_the_hold_word_comes_back_too(self, monkeypatch, out):
        """r8 DECISION: any stop/hold typed in the turn keeps EVERY held
        line of it, before the word as well as after; named once."""
        h = _Harness(monkeypatch, lines=["do it in greek", "wait"],
                     inbox={"wait": _QUEUE, "do it in greek": _QUEUE})
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft().split("\n") == ["do it in greek", "wait"]
        assert "«do it in greek»" in _plain(out)

    def test_cancel_answered_with_nothing_still_restores(self, monkeypatch, out):
        """r6 p4: another client's cancel emptied the inbox first."""
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn",
                            lambda **k: (200, {"cancelled": True, "undelivered": []}))
        h.g.chat_turn("q")
        assert "use metric" in h.g._take_draft().split("\n")

    def test_cancel_5xx_still_restores(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (500, {"error": "x"}))
        h.g.chat_turn("q")
        assert "use metric" in h.g._take_draft().split("\n")

    def test_a_stop_that_hit_an_earlier_turn_stops_this_one(self, monkeypatch, out):
        """r6 p5: the agent routes by conversation to its latest RUNNING
        turn — an earlier one still finishing. This turn is cancelled by its
        own id; the other turn's undelivered come back."""
        h = _Harness(monkeypatch, lines=["stop"], finish_on_gate=False,
                     inbox={"stop": {"path": "cancel", "request_id": "OTHERTURN",
                                     "undelivered": ["from before"]}})
        h.g.chat_turn("q")
        assert len(h.cancel_calls) == 1
        assert h.cancel_calls[0]["request_id"] == h.rid
        assert h.g.turns[-1]["interrupted"] == "cancelled"
        assert "earlier turn" in _plain(out)
        assert "from before" in h.g._take_draft().split("\n")

    @pytest.mark.parametrize("path,back", [("inject", "use metric"), ("status", "use metric")])
    def test_an_answer_about_another_turn_keeps_streaming(self, monkeypatch, out, path, back):
        """r7 #3: an inject the earlier turn ACCEPTED is not offered again
        during the turn (it may act on it); a status answer was not
        accepted: put back. r10: that turn's end is not seen from here, so
        the accepted line comes back at THIS turn's end, named."""
        h = _Harness(monkeypatch, lines=["use metric"],
                     inbox={"use metric": {"path": path, "request_id": "OTHERTURN",
                                           "reply": "x"}})
        h.g.chat_turn("q")
        assert h.cancel_calls == [] and h.g.turns[-1]["interrupted"] == ""
        assert h.g._injected == []
        assert h.g._take_draft() == back
        assert "earlier turn" in _plain(out)
        if path == "inject":
            assert "received it and may act on it" in _plain(out)
            assert "may or may not have read it" in _plain(out)

    def test_a_probe_prefixed_id_is_this_turn(self, monkeypatch, out):
        """X-Ghost-Origin: probe prefixes the request id on the agent."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "turn_inbox", lambda sid, t: (
            200, {"path": "inject", "request_id": "probe-abc123"}))
        assert g._busy_line("x", None, "abc123") == ("", None)
        assert g._injected == ["x"]

    def test_an_exception_after_a_clean_end_is_unclean(self, monkeypatch, out):
        """r6 p1: whatever the worker set, an escaped exception = unclean."""
        h = _Harness(monkeypatch, lines=["after this"], inbox={"after this": _QUEUE})
        real = cli.GhostCLI.stream_reply

        def late_raise(self, request_id, busy=None):
            real(self, request_id, busy=busy)
            assert self._turn_clean           # the worker DID call it clean
            raise RuntimeError("after the end")

        monkeypatch.setattr(cli.GhostCLI, "stream_reply", late_raise)
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "after this"


# --- review round 7 ---------------------------------------------------------------

def _other_stop(undelivered):
    return {"stop": {"path": "cancel", "request_id": "OTHERTURN",
                     "undelivered": undelivered}}


class TestRound7:
    def test_stop_own_echoes_the_stop_word_and_never_puts_it_back(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["stop"], finish_on_gate=False,
                     inbox=_other_stop([]))
        h.g.chat_turn("q")
        assert h.cancel_calls[0]["request_id"] == h.rid
        assert "stop" not in h.g._take_draft().split("\n")
        assert "\u203a stop" in _plain(out)           # echoed once, as said
        assert _plain(out).count("\u203a stop") == 1

    def test_earlier_turns_unread_are_named(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["stop"], finish_on_gate=False,
                     inbox=_other_stop(["from before"]))
        h.g.chat_turn("q")
        assert "from before" in h.g._take_draft().split("\n")
        assert "unread messages" in _plain(out) and "«from before»" in _plain(out)

    def test_a_line_already_sent_as_a_message_is_dropped_once(self, monkeypatch, out):
        """r7 #2: this turn's prompt and the previous turn's, one-for-one."""
        h = _Harness(monkeypatch, lines=["stop"], finish_on_gate=False,
                     inbox=_other_stop(["use metric", "use metric", "older"]))
        h.g.history += [{"role": "user", "content": "older"},
                        {"role": "assistant", "content": "ok"}]
        h.g.chat_turn("use metric")
        draft = h.g._take_draft().split("\n")
        assert draft.count("use metric") == 1      # one copy dropped, one kept
        assert "older" not in draft

    def test_an_earlier_message_two_turns_back_is_not_dropped(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["stop"], finish_on_gate=False,
                     inbox=_other_stop(["ancient"]))
        h.g.history += [{"role": "user", "content": "ancient"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "older"},
                        {"role": "assistant", "content": "ok"}]
        h.g.chat_turn("q")
        assert "ancient" in h.g._take_draft().split("\n")

    def test_a_uniquified_id_is_this_turn(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g.api, "turn_inbox", lambda sid, t: (
            200, {"path": "inject", "request_id": "abc123#2"}))
        assert g._busy_line("x", None, "abc123") == ("", None)
        assert g._injected == ["x"]


# --- review round 8 ---------------------------------------------------------------

_RT = cli.requests.exceptions.ReadTimeout


class TestRound8:
    def test_ctrl_c_undelivered_after_a_timeout_is_never_lost(self, monkeypatch, out):
        """r10: a timed-out typing is not matched by a server list: x twice."""
        h = _Harness(monkeypatch, lines=["x"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "turn_inbox",
                            lambda sid, t: (_ for _ in ()).throw(_RT("slow")))
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["x"]}))
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("x") == 2

    def test_stop_word_undelivered_after_a_timeout_is_never_lost(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["x", "stop"], finish_on_gate=False)

        def inbox(sid, t):
            if t == "x":
                raise _RT("slow")
            return 200, {"path": "cancel", "request_id": h.rid, "undelivered": ["x"]}

        monkeypatch.setattr(h.api, "turn_inbox", inbox)
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("x") == 2

    def test_earlier_turns_unread_typed_here_is_never_lost(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["x", "stop"], finish_on_gate=False,
                     inbox={"x": {"path": "status", "request_id": "OTHERTURN", "reply": "r"},
                            "stop": {"path": "cancel", "request_id": "OTHERTURN",
                                     "undelivered": ["x"]}})
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("x") == 2   # status put back + server

    def test_a_server_list_claims_accepted_typings_one_for_one(self, monkeypatch, out):
        """r10: matched onto open accepted typings (which leave _open);
        every server text is still returned for restoring."""
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        g._open = [("x", False), ("y", True), ("x", False)]
        assert g._claim(["x", "z", "x", "x"]) == ["x", "z", "x", "x"]
        assert g._open == [("y", True)]

    def test_joined_restored_lines_count_as_sent(self, monkeypatch, out):
        """r8 #2: the CLI sends restored lines + draft joined by newlines."""
        h = _Harness(monkeypatch, lines=["stop"], finish_on_gate=False,
                     inbox={"stop": {"path": "cancel", "request_id": "OTHERTURN",
                                     "undelivered": ["x", "y", "z"]}})
        h.g.chat_turn("x\ny")
        # "unread one": this turn's own cancel answer (the harness)
        assert h.g._take_draft().split("\n") == ["z", "unread one"]


# --- review round 9 ---------------------------------------------------------------

class TestRound9:
    def test_two_typings_of_one_text_both_come_back(self, monkeypatch, out):
        """MINOR: 'ok' timed out (put back), 'ok' typed again and injected;
        Ctrl-C, the cancel answers []: both typings come back, named."""
        h = _Harness(monkeypatch, lines=["ok", "ok"], abort=True, finish_on_gate=False)
        n = {"i": 0}

        def inbox(sid, t):
            n["i"] += 1
            if n["i"] == 1:
                raise _RT("slow")
            return 200, {"path": "inject", "request_id": h.rid}

        monkeypatch.setattr(h.api, "inbox_leftovers", lambda sid: (200, {"messages": []}))
        monkeypatch.setattr(h.api, "turn_inbox", inbox)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": []}))
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("ok") == 2
        assert "may or may not have read it" in _plain(out)

    def test_a_text_not_typed_here_is_not_deduplicated(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        assert g._restore_to_prompt(["z"]) == ["z"]
        assert g._restore_to_prompt(["z"]) == ["z"]

    def test_open_typings_reset_each_turn(self, monkeypatch, out):
        h = _Harness(monkeypatch)
        h.g._open = [("stale", False)]
        h.g.chat_turn("q")
        assert h.g._open == [] and "stale" not in h.g._take_draft()

    def test_a_line_typed_here_beats_the_history_match(self, monkeypatch, out):
        """MAJOR: the message is a paste with a line "yes"; "yes" typed
        mid-turn went to the earlier turn; "stop" hands it back unread: it
        must not be dropped as "already sent"."""
        h = _Harness(monkeypatch, lines=["yes", "stop"], finish_on_gate=False,
                     inbox={"yes": {"path": "inject", "request_id": "OTHERTURN"},
                            "stop": {"path": "cancel", "request_id": "OTHERTURN",
                                     "undelivered": ["yes", "older"]}})
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": []}))
        h.g.chat_turn("Should I delete the old backups?\nyes\nolder")
        d = h.g._take_draft().split("\n")
        assert d.count("yes") == 1 and "older" not in d
        assert "«yes»" in _plain(out)


# --- review round 10 --------------------------------------------------------------

class TestRound10:
    def test_another_clients_identical_leftover_is_kept(self, monkeypatch, out):
        """MINOR 1: "ok" typed here (accepted, unread) + another client's
        "ok": leftovers ["ok","ok"] are two messages; both come back."""
        h = _Harness(monkeypatch, lines=["ok"], leftovers=["ok", "ok"])
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("ok") == 2

    def test_another_clients_identical_undelivered_is_kept(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["ok"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["ok", "ok"]}))
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("ok") == 2

    def test_the_same_accepted_typing_handed_back_shows_once(self, monkeypatch, out):
        """The cancel's undelivered claims the typing: the end-of-turn
        'may or may not have read' restore does not add it again."""
        h = _Harness(monkeypatch, lines=["ok"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["ok"]}))
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("ok") == 1
        assert "may or may not have read" not in _plain(out)

    def test_an_earlier_turns_accepted_line_comes_back_after_a_clean_turn(
            self, monkeypatch, out):
        """MINOR 2: accepted by an earlier, still-finishing turn whose end
        this client never sees (another client may stop it and take the
        inbox): named and put back at this turn's end, never sent."""
        h = _Harness(monkeypatch, lines=["use metric"],
                     inbox={"use metric": {"path": "inject", "request_id": "OTHERTURN"}})
        h.g.chat_turn("q")
        assert list(h.g._outbox) == []
        assert h.g._take_draft() == "use metric"
        assert "may or may not have read it" in _plain(out)

    def test_this_turns_read_inject_is_not_put_back_after_a_clean_turn(
            self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric"])
        h.g.chat_turn("q")
        assert h.g._take_draft() == ""

    def test_an_earlier_turns_line_in_the_leftovers_shows_once(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric"], leftovers=["use metric"],
                     inbox={"use metric": {"path": "inject", "request_id": "OTHERTURN"}})
        h.g.chat_turn("q")
        assert h.g._take_draft() == "use metric"

    def test_an_earlier_turns_line_in_its_undelivered_shows_once(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["use metric", "stop"], finish_on_gate=False,
                     inbox={"use metric": {"path": "inject", "request_id": "OTHERTURN"},
                            "stop": {"path": "cancel", "request_id": "OTHERTURN",
                                     "undelivered": ["use metric"]}})
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": []}))
        h.g.chat_turn("q")
        assert h.g._take_draft().split("\n").count("use metric") == 1


# --- review round 11: a Ctrl-C right as the reply ends ----------------------------

_HOLD = "wait"


def _raise_after_stream(monkeypatch, h, times=1):
    real = h.g._stream_busy
    left = {"n": times}

    def sb(rid):
        r = real(rid)
        if left["n"]:
            left["n"] -= 1
            raise KeyboardInterrupt       # SIGINT at the next bytecode
        return r

    monkeypatch.setattr(h.g, "_stream_busy", sb)


class TestRound11:
    def test_ctrl_c_after_the_stream_hands_everything_back(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["early", "held1", _HOLD],
                     inbox={"early": {"path": "inject", "request_id": "OTHERTURN"},
                            "held1": _QUEUE, _HOLD: _QUEUE})
        _raise_after_stream(monkeypatch, h)
        with pytest.raises(KeyboardInterrupt):  # repl still says "interrupted"
            h.g.chat_turn("q")
        d = h.g._take_draft().split("\n")
        assert "early" in d and "held1" in d and _HOLD in d
        assert h.g._held == [] and list(h.g._outbox) == [] and h.g._open == []

    def test_held_line_is_not_sent_by_the_next_clean_turn(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["held1"], inbox={"held1": _QUEUE})
        _raise_after_stream(monkeypatch, h)
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q")
        assert h.g._take_draft() == "held1"      # unclean: back, not held
        h.g.chat_turn("next")
        assert list(h.g._outbox) == []

    def test_ctrl_c_in_the_bookkeeping_is_unclean(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["held1"], inbox={"held1": _QUEUE})
        monkeypatch.setattr(h.g, "_finish_turn",
                            lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q")
        assert list(h.g._outbox) == [] and h.g._take_draft() == "held1"

    def test_an_escape_while_settling_keeps_the_handed_out_leftovers(
            self, monkeypatch, out):
        """B: the agent already popped the leftovers; anything escaping
        afterwards must not lose them (or the held / open lines)."""
        h = _Harness(monkeypatch, lines=["x", "held1"],
                     inbox={"held1": _QUEUE}, leftovers=["unread leftover"])
        real = h.g.notice
        fired = {"n": 0}

        def notice(msg, *a, **k):
            if "not read by the agent" in msg and not fired["n"]:
                fired["n"] = 1
                raise KeyboardInterrupt
            return real(msg, *a, **k)

        monkeypatch.setattr(h.g, "notice", notice)
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q")
        d = h.g._take_draft().split("\n")
        assert fired["n"] and "unread leftover" in d and "held1" in d
        assert list(h.g._outbox) == [] and h.g._held == []

    def test_a_real_sigint_while_settling_is_held_and_is_a_stop(self, monkeypatch, out):
        import os
        import signal
        h = _Harness(monkeypatch, lines=["held1"], inbox={"held1": _QUEUE},
                     leftovers=["unread leftover"])
        real = h.g.notice

        def notice(msg, *a, **k):
            if "not read by the agent" in msg:
                os.kill(os.getpid(), signal.SIGINT)
            return real(msg, *a, **k)

        monkeypatch.setattr(h.g, "notice", notice)
        try:
            h.g.chat_turn("q")                   # nothing escapes
        except KeyboardInterrupt:
            pytest.fail("a Ctrl-C while settling escaped (not held)")
        assert list(h.g._outbox) == []           # the Ctrl-C was a stop
        d = h.g._take_draft().split("\n")
        assert sorted(d) == ["held1", "unread leftover"]

    def test_hand_back_restores_before_it_claims(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        monkeypatch.setattr(g, "_claim_one",
                            lambda t: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            g._hand_back(["another client's"])
        assert g._restored == ["another client's"]

    def test_restore_unknown_never_drops_before_restoring(self, monkeypatch, out):
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        g._open = [("x", False)]
        monkeypatch.setattr(g, "_restore_to_prompt",
                            lambda t: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            g._restore_unknown()
        assert g._open == [("x", False)]         # still there to be salvaged

    def test_an_escape_before_the_unknown_restore_keeps_open_typings(
            self, monkeypatch, out):
        """Escape after the leftovers answer but before the accepted
        typings were put back: the salvage puts them back."""
        h = _Harness(monkeypatch, lines=["early"],
                     inbox={"early": {"path": "inject", "request_id": "OTHERTURN"}})
        monkeypatch.setattr(h.api, "inbox_leftovers", lambda sid: (502, {}))
        real = h.g.notice

        def notice(msg, *a, **k):
            if "HTTP 502" in msg:
                raise KeyboardInterrupt
            return real(msg, *a, **k)

        monkeypatch.setattr(h.g, "notice", notice)
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q")
        assert h.g._take_draft() == "early" and h.g._open == []


# --- review round 12: ONE deferring handler; state on self -------------------------

import inspect as _inspect
import sys
import os as _os
import signal as _signal


@pytest.fixture
def ki_at():
    """Raise KeyboardInterrupt (once) when ``func`` reaches the line that
    contains ``needle`` — an exception NO handler can defer: what must
    hold then is that nothing is lost (state on self + _salvage)."""
    state = {}

    def arm(func, needle):
        import ast
        import textwrap
        code = func.__code__
        tree = ast.parse(textwrap.dedent(_inspect.getsource(func)))
        want = needle.rstrip("\n")
        # the first statement whose own first line holds the needle (a
        # trailing newline = the line ENDS with it)
        ln = code.co_firstlineno - 1 + min(
            n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt)
            and (ast.unparse(n).splitlines()[0].endswith(want) if needle.endswith("\n")
                 else want in ast.unparse(n).splitlines()[0]))
        state["fired"] = False

        def local(frame, ev, arg):
            if ev == "line" and frame.f_lineno == ln and not state["fired"]:
                state["fired"] = True
                raise KeyboardInterrupt
            return local

        sys.settrace(lambda f, e, a: local if f.f_code is code else None)
        return state

    yield arm
    sys.settrace(None)


def _all_text(g):
    return list(g._outbox) + list(g._restored) + list(g._held) + [g._draft]


def _sigint_in(monkeypatch, obj, name, times=1):
    """A REAL SIGINT (os.kill) delivered while ``obj.name`` runs."""
    real = getattr(obj, name)
    left = {"n": 1}

    def wrapped(*a, **k):
        if left["n"]:                       # the first call only
            left["n"] = 0
            for _ in range(times):
                _os.kill(_os.getpid(), _signal.SIGINT)
        return real(*a, **k)

    monkeypatch.setattr(obj, name, wrapped)


def _no_escape(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyboardInterrupt:
        pytest.fail("a Ctrl-C escaped the busy turn (not deferred)")


class TestRound12Windows:
    """Probes r12 A–D: an exception at a chosen line loses nothing."""

    def test_held_line_survives_an_escape_at_the_settle_entry(self, monkeypatch, out, ki_at):
        h = _Harness(monkeypatch, lines=["after this"], inbox={"after this": _QUEUE})
        st = ki_at(cli.GhostCLI._after_busy_turn, "with self._sigint_deferred(escalate=True)")
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q")
        sys.settrace(None)
        assert st["fired"] and "after this" in _all_text(h.g)
        assert h.g._held == [] and list(h.g._outbox) == []

    @pytest.mark.parametrize("needle", ["self._reset_turn()", "self._send_slot(line, outbox)",
                                        "request_id = uuid.uuid4().hex[:16]", "console.print()"])
    def test_an_outbox_line_is_never_lost_nor_left_in_history(
            self, monkeypatch, out, ki_at, needle):
        h = _Harness(monkeypatch)
        h.g._outbox.append("typed earlier")
        st = ki_at(cli.GhostCLI.chat_turn, needle)
        try:
            h.g._send_outbox()
        except KeyboardInterrupt:           # the REPL's own outbox branch
            h.g._restored[:0] = list(h.g._outbox)
            h.g._outbox.clear()
        sys.settrace(None)
        assert st["fired"]
        assert _all_text(h.g).count("typed earlier") == 1
        assert all(m["content"] != "typed earlier" for m in h.g.history)
        h.g._take_draft()
        h.g.chat_turn("next question")
        assert h.streams[-1] == ["next question"]

    def test_a_typed_line_whose_request_never_left_comes_back(self, monkeypatch, out, ki_at):
        h = _Harness(monkeypatch)
        st = ki_at(cli.GhostCLI._stream_busy, "worker = copy.copy(self)")
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("my question")
        sys.settrace(None)
        assert st["fired"] and h.g.history == []
        assert h.g._take_draft() == "my question"

    def test_classic_mode_undoes_an_unsent_line_too(self, monkeypatch, out, ki_at):
        h = _Harness(monkeypatch)
        monkeypatch.setattr(h.g, "_busy_enabled", lambda: False)
        st = ki_at(cli.GhostCLI._classic_turn, "console.print()")
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("my question")
        sys.settrace(None)
        assert st["fired"] and h.g.history == [] and h.streams == []
        assert h.g._take_draft() == "my question"

    def test_restored_and_draft_survive_an_escape_in_the_stream(self, monkeypatch, out, ki_at):
        h = _Harness(monkeypatch)
        h.g._restored, h.g._draft = ["handed back"], "my draft"
        st = ki_at(cli.GhostCLI._stream_busy, "stop: tuple[str, Any] | None = None")
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q")
        sys.settrace(None)
        ev = " ".join(_all_text(h.g))
        assert st["fired"] and "handed back" in ev and "my draft" in ev

    def test_a_clean_turns_read_lines_never_come_back(self, monkeypatch, out, ki_at):
        h = _Harness(monkeypatch, lines=["use metric"])
        h.g.chat_turn("q1")
        assert h.g._take_draft() == "" and h.g._open == []
        st = ki_at(cli.GhostCLI._stream_busy, "view = _BusyView()")
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("q2")
        sys.settrace(None)
        assert st["fired"] and "use metric" not in h.g._restored


class TestRound12RealSigint:
    """A real SIGINT anywhere in the busy turn is counted, never raised
    (except a SECOND one inside a blocking wait, which gives it up)."""

    def test_while_the_line_is_booked(self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        _sigint_in(monkeypatch, h.g, "_reset_turn")
        _no_escape(h.g.chat_turn, "q")
        # r13: counted before the request left — it never goes
        assert h.streams == [] and h.cancel_calls == [] and h.g.history == []
        assert h.g._take_draft() == "q"

    def test_just_after_the_request_left_it_is_a_stop(self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        real_start = cli.threading.Thread.start

        def start(thread):
            real_start(thread)
            if thread.name == "ghost-stream":
                _os.kill(_os.getpid(), _signal.SIGINT)

        monkeypatch.setattr(cli.threading.Thread, "start", start)
        _no_escape(h.g.chat_turn, "q")
        assert len(h.cancel_calls) == 1            # a stop: the turn is cancelled
        assert h.g.turns[-1]["interrupted"] == "cancelled"
        assert h.g.history[0]["content"] == "q"    # the request did go

    def test_while_the_draft_is_taken(self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        h.g._restored = ["handed back"]
        _sigint_in(monkeypatch, h.g, "_take_draft")
        _no_escape(h.g.chat_turn, "q")
        assert h.cancel_calls == [] and h.streams == []
        assert h.g._take_draft().split("\n") == ["q", "handed back"]

    def test_one_during_the_leftovers_wait_is_noted_and_unclean(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["after this"], inbox={"after this": _QUEUE},
                     leftovers=["unread"])
        _sigint_in(monkeypatch, h.api, "inbox_leftovers")
        _no_escape(h.g.chat_turn, "q")
        assert list(h.g._outbox) == []             # a stop: nothing sent
        assert sorted(h.g._take_draft().split("\n")) == ["after this", "unread"]

    def test_a_second_during_the_leftovers_wait_gives_it_up(self, monkeypatch, out):
        h = _Harness(monkeypatch, lines=["after this"], inbox={"after this": _QUEUE},
                     leftovers=["unread"])
        _sigint_in(monkeypatch, h.api, "inbox_leftovers", times=2)
        _no_escape(h.g.chat_turn, "q")             # the escape is caught inside
        assert "did not answer" in _plain(out)
        assert list(h.g._outbox) == [] and h.g._take_draft() == "after this"

    def test_during_the_send_from_the_outbox(self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        h.g._outbox.append("typed earlier")
        _sigint_in(monkeypatch, h.g, "_send_slot")
        _no_escape(h.g._send_outbox)
        assert list(h.g._outbox) == [] and h.streams == [] and h.g.history == []
        assert h.g._take_draft() == "typed earlier"

    def test_the_handler_is_gone_after_the_turn(self, monkeypatch, out):
        h = _Harness(monkeypatch)
        before = _signal.getsignal(_signal.SIGINT)
        h.g.chat_turn("q")
        assert _signal.getsignal(_signal.SIGINT) is before and h.g._sig is None

    def test_a_given_up_post_leaves_no_escalation_behind(self, monkeypatch, out):
        """Two Ctrl-Cs give up an inbox post (a stop); a third, while the
        stop detaches the view, must be counted — the cancel still goes."""
        h = _Harness(monkeypatch, lines=["use metric"], finish_on_gate=False)
        _sigint_in(monkeypatch, h.api, "turn_inbox", times=2)
        real = cli._BusyView.detach

        def detach(view):
            _os.kill(_os.getpid(), _signal.SIGINT)
            return real(view)

        monkeypatch.setattr(cli._BusyView, "detach", detach)
        _no_escape(h.g.chat_turn, "q")
        assert len(h.cancel_calls) == 1
        assert "use metric" in h.g._take_draft().split("\n")


class TestRound12SendPoint:
    def test_an_escape_just_before_the_request_starts_undoes_the_line(
            self, monkeypatch, out, ki_at):
        h = _Harness(monkeypatch)
        st = ki_at(cli.GhostCLI._stream_busy, "thread.start()")
        with pytest.raises(KeyboardInterrupt):
            h.g.chat_turn("my question")
        sys.settrace(None)
        assert st["fired"] and h.g.history == [] and h.streams == []
        assert h.g._take_draft() == "my question"


# --- review round 13 --------------------------------------------------------------

class TestRound13:
    def test_a_ctrl_c_counted_before_the_request_left_never_sends_it(
            self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        _sigint_in(monkeypatch, h.g, "_take_draft")
        _no_escape(h.g.chat_turn, "my question")
        assert h.streams == [] and h.cancel_calls == [] and h.g.history == []
        assert h.g._take_draft() == "my question"

    def test_a_ctrl_c_while_an_outbox_line_is_booked_stops_the_queue(
            self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        h.g._outbox.extend(["typed earlier", "and this"])
        _sigint_in(monkeypatch, h.g, "_send_slot")
        _no_escape(h.g._send_outbox)
        assert h.streams == [] and h.g.history == []
        assert h.g._take_draft().split("\n") == ["typed earlier", "and this"]

    def test_raise_never_leaks_past_its_wait(self, monkeypatch, out):
        """2nd SIGINT in the cancel POST, 3rd/4th in the 'not confirmed'
        notices: an escape; the settle still only COUNTS a Ctrl-C."""
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False)

        def cancel(**k):
            _os.kill(_os.getpid(), _signal.SIGINT)
            return 200, {"cancelled": True}

        monkeypatch.setattr(h.api, "cancel_turn", cancel)
        real_notice = h.g.notice
        left = {"n": 2}

        def notice(msg, *a, **k):
            if "cancel not confirmed" in msg and left["n"]:
                left["n"] -= 1
                _os.kill(_os.getpid(), _signal.SIGINT)
            return real_notice(msg, *a, **k)

        monkeypatch.setattr(h.g, "notice", notice)
        seen = {}
        real_settle = cli.GhostCLI._settle

        def settle(self, sig, *a, **k):
            seen["raise"] = sig["raise"]
            return real_settle(self, sig, *a, **k)

        monkeypatch.setattr(cli.GhostCLI, "_settle", settle)
        try:
            h.g.chat_turn("q")
        except KeyboardInterrupt:
            pass
        assert seen == {"raise": False}
        assert "use metric" in " ".join(_all_text(h.g))

    def test_sigint_in_the_busy_prompt_is_the_ctrl_c_path(self, monkeypatch, out):
        """r13 nit: prompt_toolkit's <sigint> would raise from prompt()."""
        from prompt_toolkit.keys import Keys
        g = cli.GhostCLI(cli.GhostAPI("http://localhost:9", "k"), session_id="s")
        captured = {}

        class Stop(Exception):
            pass

        def fake_session(**kw):
            captured.update(kw)
            raise Stop

        monkeypatch.setattr(cli, "PromptSession", fake_session)
        with pytest.raises(Stop):
            g._busy_prompt(cli._BusyView(), "")
        found = captured["key_bindings"].get_bindings_for_keys((Keys.SIGINT,))
        assert found
        exits = []

        class App:
            current_buffer = type("B", (), {"text": "half"})()

            def exit(self, result=None):
                exits.append(result)

        found[-1].handler(type("E", (), {"app": App()})())
        assert isinstance(exits[0], cli._Abort) and exits[0].text == "half"

    def test_a_ctrl_c_after_cancel_not_confirmed_is_counted(self, monkeypatch, out):
        """The wait is over once the cancel gave up: a further Ctrl-C (in
        the notice) is counted, never raised."""
        h = _Harness(monkeypatch, lines=["use metric"], abort=True, finish_on_gate=False)

        def cancel(**k):
            _os.kill(_os.getpid(), _signal.SIGINT)    # gives the cancel up
            return 200, {"cancelled": True}

        monkeypatch.setattr(h.api, "cancel_turn", cancel)
        real_notice = h.g.notice
        left = {"n": 2}    # 1: in _abort_turn's own notice (gives it up); 2: after

        def notice(msg, *a, **k):
            if "cancel not confirmed" in msg and left["n"]:
                left["n"] -= 1
                _os.kill(_os.getpid(), _signal.SIGINT)
            return real_notice(msg, *a, **k)

        monkeypatch.setattr(h.g, "notice", notice)
        _no_escape(h.g.chat_turn, "q")
        assert "use metric" in h.g._take_draft().split("\n")

    @pytest.mark.parametrize("where", ["abort", "join"])
    def test_raise_is_reset_when_its_wait_fails(self, monkeypatch, out, where):
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        if where == "abort":
            monkeypatch.setattr(cli.GhostCLI, "_abort_turn",
                                lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        else:
            real_join = cli.threading.Thread.join

            def join(thread, timeout=None):
                if timeout == cli.STOP_JOIN_S:
                    raise RuntimeError("x")
                return real_join(thread, timeout)

            monkeypatch.setattr(cli.threading.Thread, "join", join)
        seen = {}
        real_settle = cli.GhostCLI._settle

        def settle(self, sig, *a, **k):
            seen["raise"] = sig["raise"]
            return real_settle(self, sig, *a, **k)

        monkeypatch.setattr(cli.GhostCLI, "_settle", settle)
        with pytest.raises(RuntimeError):
            h.g.chat_turn("q")
        assert seen == {"raise": False}


# --- review round 14 --------------------------------------------------------------

class TestRound14:
    def test_a_stopped_outbox_send_says_so(self, monkeypatch, out):
        """r14 p1 inverted: the '· sent after the reply' echo is corrected."""
        h = _Harness(monkeypatch, finish_on_gate=False)
        h.g._outbox.extend(["typed earlier", "and this"])
        _sigint_in(monkeypatch, h.g, "_send_slot")
        _no_escape(h.g._send_outbox)
        txt = " ".join(_plain(out).split())
        assert "⛔ stopped — not sent: «typed earlier»" in txt
        assert list(h.g._outbox) == [] and h.streams == []

    def test_no_leftovers_call_when_no_request_left(self, monkeypatch, out):
        h = _Harness(monkeypatch, finish_on_gate=False)
        calls = []
        monkeypatch.setattr(h.api, "inbox_leftovers",
                            lambda sid: calls.append(sid) or (_ for _ in ()).throw(
                                cli.requests.exceptions.ConnectionError("down")))
        _sigint_in(monkeypatch, h.g, "_take_draft")
        _no_escape(h.g.chat_turn, "q")
        assert calls == []
        txt = _plain(out)
        assert "couldn't fetch" not in txt and "may not have been read" not in txt
        assert h.g._take_draft() == "q"

    def test_a_turn_that_left_still_asks_for_leftovers(self, monkeypatch, out):
        h = _Harness(monkeypatch)
        calls = []
        real = h.api.inbox_leftovers
        monkeypatch.setattr(h.api, "inbox_leftovers", lambda sid: calls.append(sid) or real(sid))
        h.g.chat_turn("q")
        assert len(calls) == 1 and "not sent" not in _plain(out)

    def test_a_ctrl_c_after_the_cancel_answered_is_counted(self, monkeypatch, out):
        """r14 p2 inverted: once the cancel answered, its wait is over — a
        Ctrl-C while its undelivered are handed back costs nothing."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["other client line"]}))
        _sigint_in(monkeypatch, h.g, "_hand_back")
        _no_escape(h.g.chat_turn, "q")
        txt = " ".join(_plain(out).split())
        assert "agent turn cancelled" in txt and "cancel not confirmed" not in txt
        assert "other client line" in h.g._take_draft().split("\n")



# --- review round 15 --------------------------------------------------------------

class TestRound15:
    def test_a_ctrl_c_as_the_cancel_answers_keeps_its_undelivered(
            self, monkeypatch, out, ki_at):
        """r15 p3 inverted: the answer is in hand — use it."""
        h = _Harness(monkeypatch, lines=["mine"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["mine", "other client line"]}))
        st = ki_at(cli.GhostCLI._abort_turn, "answered()\n")
        _no_escape(h.g.chat_turn, "q")
        sys.settrace(None)
        d = h.g._take_draft().split("\n")
        assert st["fired"] and "mine" in d and "other client line" in d
        assert "cancel not confirmed" not in _plain(out)
        assert "agent turn cancelled" in _plain(out)

    def test_a_stop_before_any_reply_puts_the_line_back(self, monkeypatch, out):
        """r15 #2: a Ctrl-C before the turn registered → 404; the line was
        dropped from history and must come back, named, honestly."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False,
                     sse=[b"", b"data: [DONE]"])     # nothing streamed yet
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            h.gate.set(), (404, {"error": "no turn"}))[1])
        h.g.chat_turn("my question")
        assert h.g.history == []
        assert h.g._take_draft().split("\n")[0] == "my question"
        txt = " ".join(_plain(out).split())
        assert "may or may not have received it: «my question»" in txt
        assert "may still run it" in txt and "already finished" not in txt

    def test_a_404_after_text_streamed_still_says_finished(self, monkeypatch, out):
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (404, {"error": "no turn"}))
        h.g.chat_turn("q")
        assert "aborted (turn already finished)" in _plain(out)
        assert "q" not in h.g._take_draft().split("\n")   # it has a partial reply


# --- review round 16 --------------------------------------------------------------

import json as _json

_RETRACT = b"data: " + _json.dumps({"choices": [{"delta": {}}],
                                     "ghost": {"retract": True}}).encode()


def _sse_then_stop(monkeypatch, h, frames, cancel=(404, {"error": "no turn"})):
    def gen(*a, **k):
        yield from frames
        assert h.gate.wait(5)
        yield b"data: [DONE]"

    monkeypatch.setattr(h, "_sse", gen)
    monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (h.gate.set(), cancel)[1])


class TestRound16:
    def test_a_retracted_turn_counts_as_started(self, monkeypatch, out):
        """r16 p1 inverted: text streamed then retracted (a tool call) —
        the turn ran: a 404 means finished, not 'not registered'."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        _sse_then_stop(monkeypatch, h, [_chunk("I will look that up"), _RETRACT])
        h.g.chat_turn("my q")
        txt = " ".join(_plain(out).split())
        assert "may not have registered" not in txt
        assert "aborted (turn already finished)" in txt
        assert h.g._take_draft().split("\n")[0] == "my q"   # no reply text: back

    def test_held_reasoning_counts_as_started(self, monkeypatch, out):
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        _sse_then_stop(monkeypatch, h, [
            b'data: {"ghost": {"reasoning_unparsed": true}}', _chunk("thinking")])
        h.g.chat_turn("my q")
        assert "may not have registered" not in _plain(out)

    def test_a_usage_frame_alone_is_not_a_start(self, monkeypatch, out):
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        _sse_then_stop(monkeypatch, h, [b'data: {"ghost": {"labelable": false}}'])
        h.g.chat_turn("my q")
        assert "may not have registered it yet" in " ".join(_plain(out).split())

    def test_answered_runs_each_step_once(self, monkeypatch, out):
        """r16 p3 inverted: an interrupt inside the hand-back re-enters
        answered(); the line is not handed back (or announced) twice."""
        h = _Harness(monkeypatch, lines=["mine"], abort=True, finish_on_gate=False)
        monkeypatch.setattr(h.api, "cancel_turn", lambda **k: (
            200, {"cancelled": True, "undelivered": ["other"]}))
        real = h.g._claim
        st = {"n": 0}

        def claim(texts):
            st["n"] += 1
            if st["n"] == 1 and texts == ["other"]:
                raise KeyboardInterrupt
            return real(texts)

        monkeypatch.setattr(h.g, "_claim", claim)
        _no_escape(h.g.chat_turn, "q")
        assert h.g._take_draft().split("\n").count("other") == 1
        assert _plain(out).count("agent turn cancelled") == 1

    def test_an_empty_stopped_elsewhere_turn_is_a_documented_limit(self, monkeypatch, out):
        """r16 #2 ACCEPTED: only [DONE] looks like a clean empty reply (a
        tool- or image-only turn): the line is NOT re-offered (Enter would
        re-run it). It stays in scrollback."""
        h = _Harness(monkeypatch, sse=[b"", b"data: [DONE]"])
        h.g.chat_turn("lost q")
        assert "lost q" not in h.g._take_draft()

    def test_an_http_error_says_the_agent_did_not_take_it(self, monkeypatch, out):
        h = _Harness(monkeypatch)

        class Refused:
            status_code = 503
            text = "busy"

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

            def json(self):
                return {"error": "busy"}

        monkeypatch.setattr(h.g._turn_api(), "chat_stream", lambda *a, **k: Refused())
        h.g.chat_turn("my q")
        txt = " ".join(_plain(out).split())
        assert "the agent did not take it: «my q»" in txt
        assert h.g._take_draft().split("\n")[0] == "my q"

    def test_an_error_frame_with_no_text_puts_the_line_back(self, monkeypatch, out):
        h = _Harness(monkeypatch, sse=[b'data: {"error": {"message": "boom"}}'])
        h.gate.set()
        h.g.chat_turn("my q")
        txt = " ".join(_plain(out).split())
        assert "may or may not have received it: «my q»" in txt

    def test_started_is_per_turn(self, monkeypatch, out):
        """The flag is reset with the turn: a streamed first turn must not
        make the next turn, stopped before anything, look started."""
        h = _Harness(monkeypatch, abort=True, finish_on_gate=False)
        _sse_then_stop(monkeypatch, h, [b"data: {}"])
        h.g._streamed["any"] = True             # the previous turn's verdict
        h.g.chat_turn("second")
        assert "may not have registered it yet" in " ".join(_plain(out).split())
