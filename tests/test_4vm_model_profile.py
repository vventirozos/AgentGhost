"""§4VM — one profile per model.

What a model needs said differently (sampler values, the thinking-off switch,
the effort level, the tokenizer, the context window) lives in
`core/model_profile.py`; callers state intent and the send words it for the
model that serves the call. These tests hold three things:

* the built-in Qwen profile reproduces the requests the agent sent before;
* a model with other words (the `OTHER` profile below: thinking is switched
  off through its effort variable, its top level is called "max") gets ITS
  words, at every send;
* the facts read from the running server (model file, window, token counts)
  move the agent only the way the module says.
"""
from __future__ import annotations

import ast
import os
import http.server
import inspect
import json
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import effort as E
from ghost_agent.core import llm as L
from ghost_agent.core import model_boot as B
from ghost_agent.core import model_profile as M
from ghost_agent.utils import token_counter as T

CARD = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0}

#: A model unlike Qwen: no enable_thinking; "none" on the effort variable
#: stops reasoning; three levels with a different top word; its own sentence.
OTHER_JSON = {
    "name": "other",
    "match": ["other-9"],
    "sampling": {"default": {"temperature": 0.7, "top_p": 0.8},
                 "coding": {"temperature": 0.2, "repeat_penalty": 1.05}},
    "think_off": {"reasoning_effort": "none"},
    "effort": {"levels": {"low": "low", "medium": "medium", "xhigh": "max"},
               "sentence_re": r"\[effort=(low|medium|max)\]", "unmarked": "medium",
               "escalation": "Think again, slowly.", "system_end": "<END>"},
    "tokenizer": None,
    "reply_reserve": 4000,
}
OTHER = M.from_dict(OTHER_JSON)


def _forget_profiles():
    with M._lock:
        M._state["profiles"], M._state["signature"] = None, None


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    # the shell's own switches must not choose a profile or a directory for a test
    monkeypatch.delenv("GHOST_MODEL_PROFILE", raising=False)
    monkeypatch.delenv("GHOST_MODEL_PROFILES", raising=False)
    monkeypatch.delenv("GHOST_GGUF_TOKENIZER", raising=False)
    M.set_main(None)
    _forget_profiles()
    T.reset_calibration()
    B.reset()
    saved = T.TOKEN_ENCODER
    yield
    M.set_main(None)
    T.TOKEN_ENCODER = saved
    T.reset_calibration()
    T._reset_fallback_warning()
    E.forget_server_default()
    B.reset()
    _forget_profiles()


class _Server:
    """A stand-in llama-server: /props, /tokenize, /apply-template."""

    def __init__(self, model_path="/m/Qwen3.8-27B-Q4.gguf", n_ctx=196608, tokens=300, template=None):
        self.model_path, self.n_ctx, self.tokens, self.template = model_path, n_ctx, tokens, template
        self.bodies = []
        self.props = None
        self.tokenize_body = None
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def _send(self, obj, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(obj).encode())

            def do_GET(self):
                if self.path == "/props" and outer.props is not None:
                    return self._send(outer.props)              # a body of the test's own
                if self.path == "/props" and outer.model_path is not None:
                    return self._send({"model_path": outer.model_path, "total_slots": 1,
                                       "default_generation_settings": {"n_ctx": outer.n_ctx}})
                self._send({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.bodies.append((self.path, body))
                if self.path == "/tokenize" and outer.tokenize_body is not None:
                    return self._send(outer.tokenize_body)
                if self.path == "/tokenize" and outer.tokens:
                    return self._send({"tokens": list(range(outer.tokens))})
                if self.path == "/apply-template" and outer.template:
                    return self._send({"prompt": outer.template(body.get("chat_template_kwargs") or {})})
                self._send({}, 404)

            def log_message(self, *a):
                pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(**kw):
        made.append(_Server(**kw))
        return made[-1]
    yield make
    for s in made:
        s.close()


# --- 1. the Qwen profile is what the agent sent before -------------------------

@pytest.mark.parametrize("role", ["general", "factual", "tool", "coding.creative", "coding.precise",
                                  "coding.balanced", "coding", "default"])
def test_qwen_samples_every_kind_of_turn_on_the_card(role):
    got = M.QWEN38.sampler(role)
    assert got == CARD
    assert list(got) == list(CARD)          # the order the request body is written in


def test_a_sampler_set_is_a_copy():
    a = M.QWEN38.sampler("tool")
    a["temperature"] = 9
    assert M.QWEN38.sampler("tool")["temperature"] == 1.0


def test_the_default_main_model_is_qwen_and_turns_are_sampled_from_it():
    from ghost_agent.core.agent import get_sampling_params
    assert M.main() is M.QWEN38
    for args in ((False, "hello"), (False, "what is 2+2"), (True, "x"), (True, "write a sql query")):
        assert get_sampling_params(args[0], query=args[1], is_coding=args[0]) == CARD


def test_turns_follow_the_main_models_profile():
    from ghost_agent.core.agent import get_sampling_params, _temp_label
    M.set_main(OTHER)
    assert get_sampling_params(False, query="hello") == {"temperature": 0.7, "top_p": 0.8}
    assert get_sampling_params(True, query="list the files") == {"temperature": 0.7, "top_p": 0.8}
    assert get_sampling_params(True, query="write a sql query", is_coding=True) == \
        {"temperature": 0.2, "repeat_penalty": 1.05}
    M.set_main(M.GENERIC)
    got = get_sampling_params(True, query="x", is_coding=True)
    assert got == {}                                    # nothing sent: the server's own defaults
    assert _temp_label(got) == "server default" and _temp_label(CARD) == "1.00"


# --- 2. intent → the model's words ----------------------------------------------

QWEN_PAYLOADS = [
    {"model": "m", "messages": []},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low", "x": 1}},
    {"chat_template_kwargs": {}},
    {"chat_template_kwargs": None},
]


@pytest.mark.parametrize("profile", [M.QWEN38, M.GEMMA, M.GENERIC, None])
@pytest.mark.parametrize("payload", QWEN_PAYLOADS)
def test_models_that_share_the_agents_words_get_the_payload_itself(profile, payload):
    before = json.dumps(payload, sort_keys=True)
    assert M.shape(payload, profile) is payload         # not a copy: the bytes cannot differ
    assert json.dumps(payload, sort_keys=True) == before


@pytest.mark.parametrize("kwargs, want", [
    ({"enable_thinking": False}, {"reasoning_effort": "none"}),
    ({"reasoning_effort": "xhigh"}, {"reasoning_effort": "max"}),
    ({"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
    ({"reasoning_effort": "max"}, {"reasoning_effort": "max"}),          # already the model's word
    ({"reasoning_effort": "bogus"}, {"reasoning_effort": "bogus"}),      # a word neither side knows: as written
    ({"enable_thinking": False, "reasoning_effort": "xhigh"}, {"reasoning_effort": "none"}),   # off wins
    ({"enable_thinking": True, "reasoning_effort": "medium"}, {"enable_thinking": True, "reasoning_effort": "medium"}),
    ({"enable_thinking": False, "keep": "me"}, {"keep": "me", "reasoning_effort": "none"}),
])
def test_a_model_with_other_words_gets_its_own(kwargs, want):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": dict(kwargs)}
    out = M.shape(payload, OTHER)
    assert out.get("chat_template_kwargs") == want
    assert payload["chat_template_kwargs"] == kwargs                    # the caller's payload is not edited
    assert out["messages"] is payload["messages"] and out["model"] == "m"


def test_a_model_with_no_thinking_switch_is_sent_none():
    mute = M.from_dict({"name": "mute", "think_off": None})
    out = M.shape({"a": 1, "chat_template_kwargs": {"enable_thinking": False}}, mute)
    assert out == {"a": 1}
    out = M.shape({"chat_template_kwargs": {"enable_thinking": False, "k": 2}}, mute)
    assert out == {"chat_template_kwargs": {"k": 2}}


def test_an_effort_variable_with_another_name():
    p = M.from_dict({"name": "budget", "effort": {"kwarg": "thinking_level", "levels": {"low": "1", "xhigh": "3"}}})
    out = M.shape({"chat_template_kwargs": {"reasoning_effort": "xhigh"}}, p)
    assert out["chat_template_kwargs"] == {"thinking_level": "3"}
    # it has no medium: the agent's level is NOT sent under the agent's name — the template's default applies
    assert M.shape({"chat_template_kwargs": {"reasoning_effort": "medium"}}, p) == {}
    assert M.shape({"a": 1, "chat_template_kwargs": {"reasoning_effort": "medium", "k": 2}}, p) == \
        {"a": 1, "chat_template_kwargs": {"k": 2}}
    same = M.from_dict({"name": "two", "effort": {"levels": {"low": "low", "medium": "medium"}}})
    assert M.shape({"chat_template_kwargs": {"reasoning_effort": "xhigh"}}, same) == {}    # same variable, no such word
    odd = {"chat_template_kwargs": {"reasoning_effort": "turbo"}}
    assert M.shape(odd, p) is odd                                        # a word neither side knows: as written


def test_thinking_off_removes_the_level_whatever_the_variables_are_called():
    p = M.from_dict({"name": "split", "think_off": {"thinking": "off"},
                     "effort": {"kwarg": "thinking_level", "levels": {"low": "1", "xhigh": "3"}}})
    for level in ("xhigh", "medium", "3", "turbo"):
        out = M.shape({"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": level}}, p)
        assert out["chat_template_kwargs"] == {"thinking": "off"}, level


@pytest.mark.parametrize("value", ["high", None, 3, ["low"], {"a": 1}])
def test_qwen_is_sent_an_effort_value_the_agent_does_not_know_as_written(value):
    payload = {"chat_template_kwargs": {"reasoning_effort": value, "enable_thinking": False}}
    assert M.shape(payload, M.QWEN38) is payload


def test_the_helpers_callers_use():
    kw = M.no_think_kwargs()
    assert M.is_no_think({"chat_template_kwargs": kw})
    kw["extra"] = 1
    assert M.no_think_kwargs() == {M.THINK_KEY: False}                  # a new dict per call
    assert not M.is_no_think({}) and not M.is_no_think(None) and not M.is_no_think({"chat_template_kwargs": None})
    p = {"chat_template_kwargs": {M.THINK_KEY: True}}
    M.set_no_think(p)
    assert p["chat_template_kwargs"][M.THINK_KEY] is True               # the caller's own choice wins…
    M.set_no_think(p, force=True)
    assert M.is_no_think(p)                                             # …unless forced
    q = {}
    M.set_no_think(q)
    assert M.is_no_think(q)


# --- 3. every send passes through the model's profile ---------------------------

def _client(**node_models):
    c = L.LLMClient.__new__(L.LLMClient)
    for url, path in node_models.items():
        c._note_node_model(url, path)
    return c


def test_a_node_is_spoken_to_in_the_words_of_the_model_it_runs(monkeypatch):
    monkeypatch.setattr(M, "profiles", lambda _p={**M.BUILTIN, "other": OTHER}: _p)
    c = _client()
    node = {"url": "http://n1", "model": "Nova"}
    payload = {"chat_template_kwargs": M.no_think_kwargs()}
    assert c.node_profile(node) is None and c._for_node(node, payload) is payload     # unknown: as written
    c._note_node_model("http://n1", "/models/Other-9-mini.Q4.gguf")                   # its /props answered
    assert c.node_profile(node) is OTHER
    assert c._for_node(node, payload)["chat_template_kwargs"] == {"reasoning_effort": "none"}
    c._note_node_model("http://n1", "/models/gemma-4-e4b.gguf")                       # the node was restarted
    assert c.node_profile(node) is M.GEMMA and c._for_node(node, payload) is payload
    assert c.node_profile({"url": "http://n2", "model": "other-9-worker"}) is OTHER   # by its label alone


def test_a_nodes_model_file_outranks_its_label_and_the_main_server_is_main(monkeypatch):
    known = {**M.BUILTIN, "other": OTHER}
    monkeypatch.setattr(M, "profiles", lambda: known)
    c = _client(**{"http://n3": "/models/Other-9-mini.gguf"})
    assert c.node_profile({"url": "http://n3", "model": "daisyrose-27b"}) is OTHER     # the longer label does not win
    c.upstream_url = "http://127.0.0.1:8088/v1"
    M.set_main(OTHER)                                   # e.g. GHOST_MODEL_PROFILE: one server, one dialect
    vision = {"url": "http://127.0.0.1:8088", "model": "qwen-vision"}
    assert c.node_profile(vision) is OTHER
    assert c._for_node(vision, {"chat_template_kwargs": M.no_think_kwargs()})["chat_template_kwargs"] == \
        {"reasoning_effort": "none"}


def test_new_profile_files_reach_a_node_already_seen(monkeypatch):
    box = {"known": dict(M.BUILTIN)}
    monkeypatch.setattr(M, "profiles", lambda: box["known"])
    c = _client(**{"http://n4": "/models/Other-9.gguf"})
    node = {"url": "http://n4", "model": "Nova"}
    assert c.node_profile(node) is None
    box["known"] = {**M.BUILTIN, "other": OTHER}        # `reload` read a new file
    assert c.node_profile(node) is OTHER


def test_the_capacity_probe_records_the_nodes_model(monkeypatch):
    monkeypatch.setattr(M, "profiles", lambda: M.BUILTIN)
    import asyncio

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"total_slots": 2, "model_path": "/m/gemma-4-e4b.gguf",
                    "default_generation_settings": {"n_ctx": 16384}}

    class _Cli:
        async def get(self, path, timeout=None):
            assert path == "/props"
            return _Resp()

    c = _client()
    c._node_slot_caps, c._node_cap_retry_at, c._node_slot_default = {}, {}, 3
    c.image_gen_clients = []
    c.node_throughput = SimpleNamespace(note_context=lambda *a: None)
    node = {"url": "http://w", "model": "Nova", "client": _Cli()}
    assert asyncio.run(c._node_capacity(node)) == 2
    assert c.node_profile(node) is M.GEMMA


def _sends():
    """Every expression LLMClient serialises into a chat request: the `json=`
    of a /v1/chat/completions call and the argument of each `json.dumps` that
    becomes a request body."""
    tree = ast.parse(inspect.getsource(L.LLMClient))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "/v1/chat/completions"
                                           for a in n.args):
            kw = next((k.value for k in n.keywords if k.arg == "json"), None)
            if kw is not None:
                out.append(kw)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "body_bytes" for t in n.targets):
            dumps = next(c for c in ast.walk(n.value) if isinstance(c, ast.Call)
                         and ast.unparse(c.func) == "json.dumps")
            out.append(dumps.args[0])
    return out


def _shaped(expr) -> bool:
    """The expression is a `_for_node(...)` / `shape_main(...)` call, or a
    choice between two such."""
    if isinstance(expr, ast.IfExp):
        return _shaped(expr.body) and _shaped(expr.orelse)
    return isinstance(expr, ast.Call) and ast.unparse(expr.func) in ("self._for_node", "_mp.shape_main")


def test_no_chat_request_leaves_the_client_unshaped():
    sends = _sends()
    assert len(sends) == 6                      # 4 node pools, the main call, the stream
    assert [ast.unparse(e) for e in sends if not _shaped(e)] == []


def test_the_enumeration_sees_an_unshaped_send():
    bad = ast.parse("c.post('/v1/chat/completions', json=payload)").body[0].value
    assert not _shaped(next(k.value for k in bad.keywords if k.arg == "json"))
    half = ast.parse("x = self._for_node(n, p) if n else p").body[0].value
    assert not _shaped(half)


SRC = Path(inspect.getfile(M)).resolve().parents[1]

#: Chat posts that need no shaping, and why.
UNSHAPED_OK = {
    "core/llm.py": "its node pools post `content=body_bytes`, built through `_for_node` (the enumeration above)",
    "eval/verify_bench.py": "an offline benchmark posting the verifier's requests to the judge it is pointed at, "
                            "as written — a judge on another model is benched in the agent's words (a known limit)",
}


def _unshaped_chat_posts(tree):
    """Lines of calls that name the chat endpoint and pass no argument made
    by `shape` / `shape_main` / `_for_node` / `_for_audio_node`."""
    shapers = ("shape", "shape_main", "_for_node", "_for_audio_node")

    def names_endpoint(node):
        """The argument IS the endpoint: a literal, an f-string or a `base + "/v1/…"` — not a
        nested call that posts (that call is judged on its own)."""
        if isinstance(node, ast.Call):
            return False
        return any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                   and c.value.endswith("/v1/chat/completions") for c in ast.walk(node))

    def shaped(node):
        if isinstance(node, ast.IfExp):
            return shaped(node.body) and shaped(node.orelse)
        return isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in shapers

    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and any(names_endpoint(a) for a in n.args):
            if ast.unparse(n.func) == "router.post":            # a route the agent SERVES, not a send
                continue
            if not any(shaped(a) for a in list(n.args) + [k.value for k in n.keywords]):
                out.append(n.lineno)
    return out


def test_every_chat_post_in_the_package_is_shaped():
    bad = []
    for f in sorted(SRC.rglob("*.py")):
        rel = str(f.relative_to(SRC))
        lines = _unshaped_chat_posts(ast.parse(f.read_text(encoding="utf-8")))
        if rel in UNSHAPED_OK:
            assert lines, f"{rel} no longer posts unshaped — drop it from UNSHAPED_OK"
            continue
        bad += [f"{rel}:{n}" for n in lines]
    assert bad == []


@pytest.mark.parametrize("code, n", [
    ("c.post('/v1/chat/completions', json=payload)", 1),
    ("await to_thread(h._http, base + '/v1/chat/completions', body, left)", 1),
    ("post(f'{URL}/v1/chat/completions', build(w), T)", 1),
    ("c.post('/v1/chat/completions', json=p if a else _mp.shape_main(p))", 1),
    ("c.post('/v1/chat/completions', json=_mp.shape_main(payload))", 0),
    ("await to_thread(h._http, base + '/v1/chat/completions', _mp.shape_main(body), left)", 0),
    ("post(f'{URL}/v1/chat/completions', _for_audio_node(build(w)), T)", 0),
    ("log('posting to /v1/chat/completions soon')", 0),
    ("@router.post('/v1/chat/completions', dependencies=[x])\ndef f(): pass", 0),
    ("await wait_for(c.post('/v1/chat/completions', json=_mp.shape_main(p)), timeout=3)", 0),
    ("await wait_for(c.post('/v1/chat/completions', json=p), timeout=3)", 1),
])
def test_the_post_scan_sees_each_shape(code, n):
    assert len(_unshaped_chat_posts(ast.parse(code))) == n


def test_the_head_is_rendered_as_it_is_sent():
    """`_head_save_plan` keys the saved head on what the server renders: the
    payload it renders must be the shaped one."""
    from ghost_agent.core.agent import GhostAgent
    tree = ast.parse(inspect.getsource(GhostAgent._head_save_plan).lstrip())
    renders = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
               and any(ast.unparse(a) == "_head_saves.render" for a in c.args)]
    assert len(renders) == 1
    assert [ast.unparse(a) for a in renders[0].args][-1] == "_mp.shape_main(payload, quiet=True)"


def test_the_audio_node_gets_its_models_words(monkeypatch):
    from ghost_agent.memory import audio_ingest as AI
    monkeypatch.setattr(M, "profiles", lambda: {**M.BUILTIN, "other": OTHER})
    monkeypatch.setitem(AI._audio_model, "path", "")
    payload = {"chat_template_kwargs": M.no_think_kwargs()}
    assert AI._for_audio_node(payload) is payload                       # model unknown: as written
    AI._note_audio_model("/m/Other-9-audio.gguf")
    assert AI._for_audio_node(payload)["chat_template_kwargs"] == {"reasoning_effort": "none"}
    AI._note_audio_model(None)                                          # a /props without the field keeps it
    assert AI._audio_model["path"].endswith("Other-9-audio.gguf")
    monkeypatch.setattr(M, "profiles", lambda: 1 / 0)
    assert AI._for_audio_node(payload) is payload                       # never lose a transcript to this


def _model_words_outside_the_profile(tree):
    """(what, line) for each place a module writes a model's own words itself:
    the thinking switch as a literal, or a sampler SET (two or more sampler
    settings besides temperature in one dict literal)."""
    found = []
    knobs = M.SAMPLER_KEYS - {"temperature"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and n.value == M.THINK_KEY:
            found.append(("thinking switch", n.lineno))
        if isinstance(n, ast.Dict):
            keys = {k.value for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if len(keys & knobs) >= 2:
                found.append(("sampler set", n.lineno))
    return found


def test_only_the_profile_module_writes_a_models_own_words():
    bad = []
    for f in sorted(SRC.rglob("*.py")):
        if f.name == "model_profile.py":
            continue
        bad += [f"{f.relative_to(SRC)}:{line} {what}"
                for what, line in _model_words_outside_the_profile(ast.parse(f.read_text(encoding="utf-8")))]
    assert bad == []


@pytest.mark.parametrize("code, what", [
    ('p = {"chat_template_kwargs": {"enable_thinking": False}}', "thinking switch"),
    ('kw.setdefault("enable_thinking", False)', "thinking switch"),
    ('S = {"temperature": 0.6, "top_p": 0.9, "top_k": 40}', "sampler set"),
])
def test_the_scan_sees_each_kind(code, what):
    assert [w for w, _ in _model_words_outside_the_profile(ast.parse(code))] == [what]
    assert _model_words_outside_the_profile(ast.parse('p = {"temperature": 0.0, "max_tokens": 1}')) == []


# --- 4. profiles from files -------------------------------------------------------

def test_a_file_profile_round_trip():
    assert OTHER.name == "other" and OTHER.match == ("other-9",)
    assert OTHER.sampler("coding.precise") == {"temperature": 0.2, "repeat_penalty": 1.05}
    assert OTHER.sampler("factual") == {"temperature": 0.7, "top_p": 0.8}
    assert OTHER.think_off == {"reasoning_effort": "none"}
    assert OTHER.effort.native("xhigh") == "max" and OTHER.effort.agent_level("max") == "xhigh"
    assert OTHER.tokenizer_repo == "" and OTHER.reply_reserve == 4000


def test_extends_starts_from_a_known_profile_but_not_its_names():
    p = M.from_dict({"name": "qwen4", "extends": "qwen38", "match": ["qwen4"],
                     "sampling": {"default": {"temperature": 0.6}}})
    assert p.effort is M.QWEN38.effort and p.tokenizer_repo == M.QWEN38.tokenizer_repo
    assert p.sampler("tool") == {"temperature": 0.6}                 # `sampling` is replaced whole
    assert M.from_dict({"name": "twin", "extends": "qwen38"}).match == ()


@pytest.mark.parametrize("bad", [
    [],
    {"name": "Bad Name"},
    {"name": "x", "surprise": 1},
    {"name": "x", "extends": "nope"},
    {"name": "x", "match": "qwen"},
    {"name": "x", "match": [""]},
    {"name": "x", "sampling": {"chitchat": {"temperature": 1}}},
    {"name": "x", "sampling": {"default": {"max_tokens": 5}}},          # not a sampler setting
    {"name": "x", "sampling": {"default": {"messages": []}}},
    {"name": "x", "sampling": {"default": {"temperature": "hot"}}},
    {"name": "x", "sampling": {"default": {"temperature": True}}},
    {"name": "x", "think_off": {}},
    {"name": "x", "think_off": {"k": [1]}},
    {"name": "x", "effort": {"levels": {}}},
    {"name": "x", "effort": {"levels": {"extreme": "x"}}},
    {"name": "x", "effort": {"levels": {"low": "l"}, "sentence_re": "("}},
    {"name": "x", "effort": {"levels": {"low": "l"}, "sentence_re": "no group"}},
    {"name": "x", "effort": {"levels": {"low": "l"}, "unmarked": "max"}},
    {"name": "x", "effort": {"levels": {"low": "l"}, "kwarg": ""}},
    {"name": "x", "effort": {"levels": {"low": "l"}, "typo": 1}},
    {"name": "x", "tokenizer": {"repo": "a/b", "dir": "memory"}},       # the directory is not the file's choice
    {"name": "x", "tokenizer": {"repo": 5}},
    {"name": "x", "tokenizer": {"repo": "/etc"}},                       # a repository, never a path
    {"name": "x", "tokenizer": {"repo": "../a/b"}},
    {"name": "x\n"},
    {"name": "x", "extends": ["qwen38"]},
    {"name": "x", "extends": {}},
    {"name": "x", "match": ["a"]},                                      # "a" is in "Nova"
    {"name": "x", "sampling": {"default": {"temperature": float("nan")}}},
    {"name": "x", "sampling": {"default": {"top_k": float("inf")}}},
    {"name": "x", "think_off": {"enable_thinking": True}},              # "off" may not switch it on
    {"name": "x", "think_off": {"k": float("inf")}},
    {"name": "x", "effort": {"levels": {"low": "l"}, "sentence_re": "(a)" + "b" * 298}},      # 301 characters
    {"name": "x", "effort": {"levels": {"low": "low", "medium": "high", "xhigh": "high"}}},    # one word, two levels
    {"name": "x", "think_off": {"enable_thinking": 1}},
    {"name": "x", "think_off": {"enable_thinking": "true"}},
    {"name": "x", "think_off": {"enable_thinking": None}},
    {"name": "x", "match": ["ab"]},
    {"name": "x", "sampling": {"default": {"top_k": 10 ** 400}}},        # too large for a float
    {"name": "x", "reply_reserve": True},
    {"name": "x", "reply_reserve": -1},
    {"name": "x", "reply_reserve": 1.5},
])
def test_a_profile_file_that_says_something_unusable_is_refused(bad):
    with pytest.raises(M.ProfileError):
        M.from_dict(bad)


@pytest.mark.parametrize("levels", [
    {"low": "low", "xhigh": "high"},                                   # a two-level model
    {"low": "minimal", "medium": "low", "xhigh": "medium"},            # words shifted against the agent's names
    {"medium": "low", "xhigh": "high"},
])
def test_any_one_to_one_level_mapping_is_accepted(levels):
    eff = M.from_dict({"name": "x", "effort": {"levels": levels}}).effort
    for agent, word in levels.items():
        assert eff.native(agent) == word and eff.agent_level(word) == agent
        assert M.shape({"chat_template_kwargs": {"reasoning_effort": agent}},
                       M.from_dict({"name": "x", "effort": {"levels": levels}}))["chat_template_kwargs"] == \
            {"reasoning_effort": word}


INKLING_LIKE = {"name": "numbers", "match": ["numbers-1"], "think_off": {"reasoning_effort": "none"},
                "effort": {"levels": {"low": "low", "medium": "high", "xhigh": "max"},
                           "rendered": {"low": "0.2", "high": "0.9", "max": "0.99"},
                           "sentence_re": r"<sys>Thinking effort level: ([0-9.]+)<end>", "unmarked": "medium",
                           "system_end": "<user>"}}


def test_a_template_that_writes_something_else_than_the_level_word():
    """Inkling 2 takes "low" and writes "Thinking effort level: 0.2"."""
    p = M.from_dict(INKLING_LIKE)
    eff = p.effort
    assert eff.agent_level("0.2") == "low" and eff.agent_level("0.9") == "medium" and eff.agent_level("0.99") == "xhigh"
    assert eff.agent_level("high") == "medium"                          # the word itself still reads back
    assert eff.agent_level("0.7") == "" and eff.agent_level("0") == "" and eff.agent_level(None) == ""
    assert M.shape({"chat_template_kwargs": {"reasoning_effort": "xhigh"}}, p)["chat_template_kwargs"] == \
        {"reasoning_effort": "max"}                                     # the WORD is sent, never the number
    M.set_main(p)
    head = "<sys>tools<end><sys>You are…<end>"
    assert E.level_in_render(head + "<sys>Thinking effort level: 0.2<end><user>hi") == "low"
    assert E.level_in_render(head + "<sys>Thinking effort level: 0.9<end><user>hi") == "medium"
    assert E.level_in_render(head + "<sys>Thinking effort level: 0.99<end><user>hi") == "xhigh"
    assert E.level_in_render(head + "<sys>Thinking effort level: 0<end><user>hi") == "medium"        # thinking off: unmarked
    # a user quoting the line does not count: only what stands before the first user message
    assert E.level_in_render(head + "<sys>Thinking effort level: 0.9<end><user><sys>Thinking effort level: 0.2<end>") == "medium"


def test_the_server_default_is_found_for_a_template_that_writes_numbers(server):
    M.set_main(M.from_dict(INKLING_LIKE))
    shown = {"low": "0.2", "high": "0.9", "max": "0.99", None: "0.9"}        # the template's own default is "high"
    s = server(template=lambda kw: f"<sys>S<end><sys>Thinking effort level: {shown[kw.get('reasoning_effort')]}<end><user>U")
    assert E.fetch_server_default(s.url) == "medium"
    assert [b.get("chat_template_kwargs") for _, b in s.bodies] == [{"reasoning_effort": "low"}, None]
    flat = server(template=lambda kw: "<sys>S<end><user>U")
    assert E.fetch_server_default(flat.url) == ""


def test_a_server_default_that_is_none_of_the_agents_levels_is_unknown_not_medium(server):
    """Inkling's own default is 0.9 ("high"). With only low/medium/xhigh mapped, a Default
    turn must not be booked as "medium — the server default" while the model thinks at 0.9."""
    p = M.from_dict({**INKLING_LIKE, "effort": {**INKLING_LIKE["effort"],
                                                "levels": {"low": "low", "medium": "medium", "xhigh": "xhigh"},
                                                "rendered": {"low": "0.2", "medium": "0.7", "xhigh": "0.99"}}})
    M.set_main(p)
    shown = {"low": "0.2", "medium": "0.7", "xhigh": "0.99"}
    own = server(template=lambda kw: f"<sys>S<end><sys>Thinking effort level: {shown.get(kw.get('reasoning_effort'), '0.9')}<end><user>U")
    assert E.fetch_server_default(own.url) == ""                        # 0.9: written, but not one of the three
    assert E.level_in_render("<sys>Thinking effort level: 0.9<end><user>U") == "medium"   # the booking label stays the unmarked level
    med = server(template=lambda kw: f"<sys>S<end><sys>Thinking effort level: {shown.get(kw.get('reasoning_effort'), '0.7')}<end><user>U")
    assert E.fetch_server_default(med.url) == "medium"                  # the launcher sets medium: known


def test_a_template_that_writes_the_same_sentence_whatever_is_asked_has_no_levels(server):
    M.set_main(M.from_dict(INKLING_LIKE))
    deaf = server(template=lambda kw: "<sys>S<end><sys>Thinking effort level: 0.9<end><user>U")
    assert E.fetch_server_default(deaf.url) == ""                       # asked for low, wrote 0.9: it did not take the level


@pytest.mark.parametrize("rendered", [
    {"nope": "0.2"},                    # not one of the model's words
    {"low": ""},
    {"low": 0.2},                       # the text as it appears, not a number
    {"low": "0.9", "high": "0.9"},      # two levels that would look the same in a prompt
    {"low": "high"},                    # one level written as another level's word
    {"low": "max", "max": "low"},       # swapped: a rendered "low" would read back as low though it is max's text
    {"low": "high", "high": "0.9"},     # a value that is another level's word, even when that word is itself rendered
    {"low": " 0.2"}, {"low": "0.2 "},   # the exact text, no spaces around it
    ["0.2"],
])
def test_a_rendered_map_that_cannot_be_read_back_is_refused(rendered):
    bad = {**INKLING_LIKE, "effort": {**INKLING_LIKE["effort"], "rendered": rendered}}
    with pytest.raises(M.ProfileError):
        M.from_dict(bad)


def test_the_edges_of_what_a_file_may_say():
    p = M.from_dict({"name": "x", "match": [" ABC ", "Other-9"], "reply_reserve": 0,
                     "think_off": {"enable_thinking": False, "k": 1},
                     "effort": {"levels": {"low": "l"}, "sentence_re": "(a)" + "b" * 297}})     # 300 characters
    assert p.match == ("abc", "other-9")                    # three characters is enough; names are lower-cased
    assert p.reply_reserve == 0 and p.think_off == {"enable_thinking": False, "k": 1}
    assert p.fits("/m/ABCdef.gguf") and M.resolve({"x": p}, "/m/xxABCxx.gguf") is p
    assert p.effort.agent_level("l") == "low" and p.effort.agent_level("nope") == "" and p.effort.agent_level(None) == ""
    assert p.effort.native("low") == "l" and p.effort.native("l") == "l" and p.effort.native("medium") is None
    assert p.effort.native(["low"]) is None
    with pytest.raises(M.ProfileError):
        M.from_dict({"name": "x", "sampling": {"default": {"top_k": 10 ** 400}}})    # a ProfileError, not an OverflowError


@pytest.mark.parametrize("sampling, role, want", [
    ({"general": {"temperature": 1}}, "factual", {"temperature": 1}),            # factual → general
    ({"default": {"temperature": 9}, "general": {"temperature": 1}}, "factual", {"temperature": 1}),
    ({"default": {"temperature": 9}}, "factual", {"temperature": 9}),            # … → default
    ({"tool": {"temperature": 2}, "default": {"temperature": 9}}, "coding.creative", {"temperature": 2}),   # coding.x → coding → tool
    ({"tool": {"temperature": 2}, "default": {"temperature": 9}}, "coding", {"temperature": 2}),
    ({"coding": {"temperature": 3}, "tool": {"temperature": 2}}, "coding.balanced", {"temperature": 3}),
    ({"coding": {"temperature": 3}, "tool": {"temperature": 2}}, "coding.precise", {"temperature": 3}),
    ({"coding": {"temperature": 3}, "default": {"temperature": 9}}, "tool", {"temperature": 9}),           # a parent never reads a child
    ({"coding": {"temperature": 3}}, "general", {}),
    ({"default": {"temperature": 9}}, "something else", {"temperature": 9}),     # an unknown role is the default
    ({"default": {"temperature": 9}}, "", {"temperature": 9}),
    ({}, "coding.precise", {}),
    ({}, "default", {}),
])
def test_the_sampler_fallback_chain(sampling, role, want):
    assert M.ModelProfile(name="p", sampling=sampling).sampler(role) == want


def test_bad_files_are_skipped_and_named_good_ones_load(tmp_path):
    (tmp_path / "10-other.json").write_text(json.dumps(OTHER_JSON))
    (tmp_path / "20-child.json").write_text(json.dumps({"name": "child", "extends": "other", "match": ["child"]}))
    (tmp_path / "30-broken.json").write_text("{not json")
    (tmp_path / "40-wrong.json").write_text(json.dumps({"name": "w", "sampling": {"default": {"model": 1}}}))
    (tmp_path / "50-qwen38.json").write_text(json.dumps(
        {"name": "qwen38", "extends": "qwen38", "match": ["qwen"], "sampling": {"default": {"temperature": 0.5}}}))
    (tmp_path / "notes.txt").write_text("ignored")
    got, bad = M.load_all(tmp_path)
    assert set(got) == {"qwen38", "gemma", "generic", "other", "child"}
    assert set(bad) == {"30-broken.json", "40-wrong.json"} and "sampler setting" in bad["40-wrong.json"]
    assert got["child"].think_off == {"reasoning_effort": "none"}        # extended a FILE profile
    assert got["qwen38"].sampler("tool") == {"temperature": 0.5}         # a file replaces a built-in
    assert M.load_all(tmp_path / "missing") == (M.BUILTIN, {})


def test_no_file_can_stop_the_profiles_from_loading(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text('{"name": "x", "extends": ["q"]}')
    (tmp_path / "b.json").write_text("[" * 100000)                       # RecursionError in the parser
    (tmp_path / "c.json").write_text('{"name": "big", "match": ["' + "m" * M.FILE_MAX_BYTES + '"]}')
    (tmp_path / "d.json").write_text('["a list"]')
    (tmp_path / "e.json").write_bytes(b"\xff\xfe not utf-8")
    (tmp_path / "f.json").write_text(json.dumps(OTHER_JSON))
    got, bad = M.load_all(tmp_path)
    assert set(got) == set(M.BUILTIN) | {"other"} and set(bad) == {"a.json", "b.json", "c.json", "d.json", "e.json"}
    monkeypatch.setenv("GHOST_MODEL_PROFILES", str(tmp_path))
    assert "other" in M.reload()
    monkeypatch.setattr(M, "load_all", lambda *a: 1 / 0)                 # even the loader itself
    assert M.reload() == M.BUILTIN


def test_what_counts_as_a_profile_file(tmp_path, caplog):
    (tmp_path / "a.JSON").write_text(json.dumps(OTHER_JSON))                    # any case of the suffix
    (tmp_path / "b.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"name": "bom", "match": ["bom"]}).encode())
    (tmp_path / "c.json").mkdir()                                               # a directory with the name
    (tmp_path / "d.json").symlink_to(tmp_path / "gone.json")                    # a link to nothing
    got, bad = M.load_all(tmp_path)
    assert {"other", "bom"} <= set(got) and set(bad) == {"c.json", "d.json"}    # named, not silently passed over
    assert all("not a readable file" in why for why in bad.values())


def test_the_files_are_read_once_and_again_when_the_directory_changes(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("GHOST_MODEL_PROFILES", str(tmp_path))
    (tmp_path / "bad.json").write_text("{not json")
    with caplog.at_level("WARNING", logger="GhostAgent"):
        first = M.profiles()
    assert [r.getMessage() for r in caplog.records if "bad.json skipped" in r.getMessage()]      # said, with the file's name
    assert M.profiles() is first and M.reload_if_changed() is False             # one dict between two reads
    (tmp_path / "other.json").write_text(json.dumps(OTHER_JSON))                # the operator adds a model
    assert M.reload_if_changed() is True
    assert "other" in M.profiles() and M.profiles() is not first
    assert M.reload_if_changed() is False
    (tmp_path / "other.json").write_text(json.dumps({**OTHER_JSON, "reply_reserve": 1}))         # … and edits it
    assert M.reload_if_changed() is True and M.profiles()["other"].reply_reserve == 1
    (tmp_path / "other.json").unlink()
    assert M.reload_if_changed() is True and "other" not in M.profiles()


def test_the_directory_signature_sees_a_name_a_size_and_a_time(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}")
    os.utime(f, ns=(5, 5))
    one = M.files_signature(tmp_path)
    f.write_text("{} ")
    os.utime(f, ns=(5, 5))                                   # another size, the same time
    two = M.files_signature(tmp_path)
    os.utime(f, ns=(6, 6))                                   # the same size, another time
    three = M.files_signature(tmp_path)
    f.rename(tmp_path / "b.json")
    os.utime(tmp_path / "b.json", ns=(6, 6))
    assert len({one, two, three, M.files_signature(tmp_path)}) == 4
    assert M.files_signature(tmp_path / "missing") == ()


def test_the_profile_directory_without_a_home_is_the_agents_default_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GHOST_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    want = tmp_path / "ghost_llamacpp" / "system" / "model_profiles"            # where main.py puts the agent's home
    assert M.profiles_dir() == want
    want.mkdir(parents=True)
    (want / "other.json").write_text(json.dumps(OTHER_JSON))
    assert "other" in M.reload()


def test_a_file_named_like_a_built_in_keeps_what_it_does_not_say(tmp_path):
    (tmp_path / "qwen38.json").write_text(json.dumps({"name": "qwen38", "reply_reserve": 30000}))
    known, bad = M.load_all(tmp_path)
    mine = known["qwen38"]
    assert bad == {} and mine.reply_reserve == 30000
    assert mine.match == M.QWEN38.match and mine.effort is M.QWEN38.effort and mine.sampler("tool") == CARD
    assert mine.tokenizer_repo == M.QWEN38.tokenizer_repo
    assert M.choose_main("", "/m/Qwen3.8-27B.gguf", known, "")[0] is mine


def test_a_file_that_replaces_a_built_in_is_the_one_used(tmp_path):
    (tmp_path / "qwen38.json").write_text(json.dumps(
        {"name": "qwen38", "extends": "qwen38", "sampling": {"default": {"temperature": 0.5}}}))
    known, bad = M.load_all(tmp_path)
    mine = known["qwen38"]
    assert bad == {} and mine is not M.QWEN38 and mine.match == M.QWEN38.match      # it keeps the names
    assert M.choose_main("qwen3.8-27b", "/m/Qwen3.8-27B.gguf", known, "")[0] is mine
    assert M.choose_main("", "/m/DaisyRose-27B.gguf", known, "")[0] is mine
    assert M.choose_main("mystery", "", known, "")[0] is mine                       # and it is the default …
    assert M.choose_main("", "", known, "")[0] is mine
    assert M.choose_main("mystery", "/m/mystery.gguf", known, "")[0] is M.GENERIC   # … for a server that did not say


def test_the_profile_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("GHOST_MODEL_PROFILES", raising=False)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert M.profiles_dir() == tmp_path / "system" / "model_profiles"
    monkeypatch.setenv("GHOST_MODEL_PROFILES", str(tmp_path / "p"))
    assert M.profiles_dir() == tmp_path / "p"
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "other.json").write_text(json.dumps(OTHER_JSON))
    assert "other" in M.reload() and M.profiles()["other"].name == "other"


# --- 5. which profile a model gets ------------------------------------------------

KNOWN = {**M.BUILTIN, "other": OTHER,
         "qwen4": M.from_dict({"name": "qwen4", "extends": "qwen38", "match": ["qwen4"]})}


@pytest.mark.parametrize("names, want", [
    (("/Models/qwen3.8/Qwen3.8-27B-mtp-Q4_K_M.gguf",), "qwen38"),
    (("/Models/DaisyRose/DaisyRose-27B-Heretic-v2-MTP-Q4_K_M.gguf",), "qwen38"),
    (("/Models/swift-1.5/Swift-1.5-Qwen3.8-27b.i1-Q4_K_M.gguf",), "qwen38"),
    (("/Models/qwen3.6/heretic/Qwen3.6-35B-A3B-heretic.gguf",), "qwen38"),
    (("/m/Qwen4-30B.gguf",), "qwen4"),                    # the longer name wins over "qwen"
    (("/m/gemma-4-e4b-it.gguf",), "gemma"),
    (("/models/qwen3.8/Other-9-mini.gguf",), "other"),    # the FILE's name, not its folder's
    (("/models/qwen3.8/mystery.gguf",), None),
    ((None, "Nova"), None),
    ((None, "gemma-worker"), "gemma"),
    (("", ""), None),
])
def test_resolve(names, want):
    got = M.resolve(KNOWN, *names)
    assert (got.name if got else None) == want


def test_an_operators_file_wins_a_tie_with_a_built_in():
    mine = M.from_dict({"name": "mine", "match": ["qwen"]})
    assert M.resolve({**M.BUILTIN, "mine": mine}, "Qwen3.8.gguf") is mine
    assert M.resolve({"mine": mine, **M.BUILTIN}, "Qwen3.8.gguf") is mine          # whatever the order
    # … also a file that REPLACES a built-in: it is the operator's, not the built-in of that name
    gemma2 = M.from_dict({"name": "gemma", "match": ["qwen"]}, M.BUILTIN)
    assert M.resolve({**M.BUILTIN, "gemma": gemma2}, "Qwen3.8.gguf") is gemma2


def test_between_two_operator_files_the_first_read_wins():
    a = M.from_dict({"name": "aaa", "match": ["mix"]})
    b = M.from_dict({"name": "bbb", "match": ["mix"]})
    assert M.resolve({**M.BUILTIN, "aaa": a, "bbb": b}, "mix.gguf") is a
    assert M.resolve({**M.BUILTIN, "bbb": b, "aaa": a}, "mix.gguf") is b


def test_the_choice_is_announced_again_when_only_the_reason_changes(caplog):
    with caplog.at_level("INFO", logger="GhostAgent"):
        M.announce_main(OTHER, "the server runs A.gguf")
        M.announce_main(OTHER, "the server runs A.gguf")
        M.announce_main(OTHER, "the server runs B.gguf")
    said = [r.getMessage() for r in caplog.records if "model profile: other" in r.getMessage()]
    assert len(said) == 2 and "B.gguf" in said[1] and M.main_why() == "the server runs B.gguf"


@pytest.mark.parametrize("override, model_arg, served, want, sure", [
    ("other", "qwen3.8-27b", "/m/Qwen3.8.gguf", "other", True),         # the operator's word
    ("nope", "qwen3.8-27b", "/m/gemma-4.gguf", "gemma", True),          # an unknown name is not a choice
    ("", "qwen3.8-27b", "/m/Other-9.gguf", "other", True),              # the server over --model
    ("", "qwen3.8-27b", "", "qwen38", True),                            # a silent server: --model
    ("", "qwen3.8-27b", "/m/mystery.gguf", "generic", False),           # the server named a model nothing fits: nothing assumed
    ("", "mystery", "/m/mystery.gguf", "generic", False),               # … whatever --model says, and loudly
    ("", "mystery", "", "qwen38", True),                                # a silent server and an unknown --model: the default
    ("", "", "", "qwen38", True),
    ("generic", "", "/m/Qwen3.8.gguf", "generic", True),
])
def test_choose_main(override, model_arg, served, want, sure):
    p, why, got_sure = M.choose_main(model_arg, served, KNOWN, override)
    assert (p.name, got_sure) == (want, sure) and why


# --- by what the model file says it is (§4VP) ----------------------------------------

def test_a_model_is_known_by_what_its_file_says_when_its_name_says_nothing():
    """A finetune can be called anything; its file still says which family it is."""
    qwen = {"architecture": "qwen35", "pre": "qwen35"}
    p, why, sure = M.choose_main("", "/m/RVN-Q4_K_M-multilingual-mtp.gguf", KNOWN, "", meta=qwen)
    assert p is M.QWEN38 and sure and "says it is qwen35" in why and "RVN-Q4_K_M" in why
    assert M.choose_main("", "/m/odd-name.gguf", KNOWN, "", meta={"pre": "QWEN35 "})[0] is M.QWEN38     # either field, any case
    assert M.choose_main("", "/m/odd-name.gguf", KNOWN, "", meta={"architecture": "gemma4"})[0] is M.GEMMA
    for meta in (None, {}, {"architecture": "never-heard-of-it"}, {"pre": ""}, {"architecture": 7}):
        p, why, sure = M.choose_main("qwen3.8-27b", "/m/odd-name.gguf", KNOWN, "", meta=meta)
        assert p is M.GENERIC and not sure, meta
    # the NAME is asked first: a file named for one profile is not re-read as another family
    assert M.choose_main("", "/m/Other-9.gguf", KNOWN, "", meta=qwen)[0] is OTHER


def test_an_operators_file_can_claim_a_family_and_wins_over_a_built_in(tmp_path):
    mine = M.from_dict({"name": "myqwen", "arch": [" Qwen35 "], "sampling": {"default": {"temperature": 0.5}}})
    assert mine.arch == ("qwen35",) and mine.match == ()
    known = {**M.BUILTIN, "myqwen": mine}
    assert M.resolve_by_arch(known, {"pre": "qwen35"}) is mine
    assert M.resolve_by_arch({"myqwen": mine, **M.BUILTIN}, {"pre": "qwen35"}) is mine
    assert M.resolve_by_arch(M.BUILTIN, {"pre": "qwen35"}) is M.QWEN38 and M.resolve_by_arch(M.BUILTIN, {"pre": "x"}) is None
    assert M.resolve_by_arch(M.BUILTIN, None) is None and M.resolve_by_arch(M.BUILTIN, {}) is None
    # a file that replaces a built-in keeps its families; an extension under another name does not
    assert M.from_dict({"name": "qwen38", "reply_reserve": 1}, M.BUILTIN).arch == M.QWEN38.arch
    assert M.from_dict({"name": "child", "extends": "qwen38"}, M.BUILTIN).arch == ()
    for bad in ("qwen35", [""], [7], [" "]):
        with pytest.raises(M.ProfileError):
            M.from_dict({"name": "x", "arch": bad})


def test_the_boot_reads_the_file_only_when_its_name_fits_nothing(server, loaded, tmp_path, monkeypatch):
    reads = []

    def meta(path):
        reads.append(os.path.basename(path))
        return {"architecture": "qwen35", "pre": "qwen35"}
    monkeypatch.setattr(B._gguf, "read_meta", meta)
    named = tmp_path / "Qwen3.8-27B.gguf"
    odd = tmp_path / "RVN-Q4_K_M.gguf"
    named.write_bytes(b"x")
    odd.write_bytes(b"x")
    s = server(model_path=str(named))
    args = _args(s.url, model="")
    assert B.sync(args, tmp_path, boot=True, now=0.0)["profile"] == "qwen38" and reads == []     # the name was enough
    s.model_path = str(odd)
    got = B.sync(args, now=100.0)
    assert got["profile"] == "qwen38" and reads == ["RVN-Q4_K_M.gguf"] and "says it is qwen35" in got["why"]
    s.model_path = "/elsewhere/odd.gguf"                                   # a file this machine cannot read: no guess
    assert B.sync(args, now=200.0)["profile"] == "generic" and len(reads) == 1


def test_a_silent_server_leaves_the_choice_to_what_the_agent_was_started_with():
    """A reviewer's mutant that ignored --model for a silent server passed every test."""
    for arg, want in (("qwen3.8-27b", M.QWEN38), ("gemma-4-e4b", M.GEMMA), ("Other-9", OTHER)):
        p, why, sure = M.choose_main(arg, "", KNOWN, "")
        assert p is want and sure, arg
    assert M.choose_main("", "", KNOWN, "")[0] is M.QWEN38 and M.choose_main("nothing-known", "", KNOWN, "")[0] is M.QWEN38
    assert M.choose_main("gemma-4-e4b", "", KNOWN, "", meta={"pre": "qwen35"})[0] is M.GEMMA    # no file, so nothing it "says"
    assert M.choose_main("nothing-known", "", KNOWN, "", meta={"pre": "gemma4"})[0] is M.QWEN38
    assert M.choose_main("", "", KNOWN, "", meta={"architecture": "gemma4"})[0] is M.QWEN38


def test_what_a_file_said_is_remembered_while_the_file_is_unchanged(tmp_path, monkeypatch):
    """Reading it parses the file's whole key-value block (seconds), and a model
    whose tokenizer cannot be checked yet is looked at again every minute."""
    reads = []
    monkeypatch.setattr(B._gguf, "read_meta", lambda path: reads.append(path) or {"pre": "qwen35"})
    f = tmp_path / "odd.gguf"
    f.write_bytes(b"one")
    assert B._file_meta(str(f)) == {"pre": "qwen35"} and B._file_meta(str(f)) == {"pre": "qwen35"} and len(reads) == 1
    f.write_bytes(b"another file now")
    assert B._file_meta(str(f)) == {"pre": "qwen35"} and len(reads) == 2
    assert B._file_meta(str(tmp_path / "missing.gguf")) is None and len(reads) == 2


def test_a_family_that_cannot_be_read_is_said_so(server, loaded, tmp_path, monkeypatch):
    monkeypatch.setattr(B._gguf, "read_meta", lambda path: {"architecture": "never-heard-of-it"})
    s = server(model_path="/another/machine/odd.gguf")
    args = _args(s.url, model="")
    got = B.sync(args, tmp_path, boot=True, now=0.0)
    assert got["profile"] == "generic" and "cannot be read from here" in got["why"]
    here = tmp_path / "odd.gguf"
    here.write_bytes(b"x")
    s.model_path = str(here)
    got = B.sync(args, now=100.0)
    assert got["profile"] == "generic" and "cannot be read" not in got["why"]


# --- thinking off, for a model whose template has no switch that works (§4VP) ---------

def test_a_profile_can_set_request_fields_for_a_thinking_off_call():
    """Measured on Inkling 2: effort "none" writes level 0 and the model reasons
    as long as before (214 characters either way); `reasoning_budget_tokens: 0`
    stops it (0 characters, 21 tokens against 88)."""
    prof = M.from_dict({**INKLING_LIKE, "think_off": None, "think_off_request": {"reasoning_budget_tokens": 0}})
    assert dict(prof.think_off_request) == {"reasoning_budget_tokens": 0}
    off = {"messages": [], "chat_template_kwargs": M.no_think_kwargs()}
    sent = M.shape(off, prof)
    assert sent == {"messages": [], "reasoning_budget_tokens": 0} and "reasoning_budget_tokens" not in off    # the caller's is not edited
    # the turn's level is kept beside it: the prompt stays the one the slot holds
    both = M.shape({"chat_template_kwargs": {**M.no_think_kwargs(), "reasoning_effort": "low"}}, prof)
    assert both == {"chat_template_kwargs": {"reasoning_effort": "low"}, "reasoning_budget_tokens": 0}
    # a call that thinks is sent no cap; a cap the caller set itself is the caller's
    assert "reasoning_budget_tokens" not in M.shape({"chat_template_kwargs": {"reasoning_effort": "low"}}, prof)
    assert "reasoning_budget_tokens" not in M.shape({"messages": []}, prof)
    assert "reasoning_budget_tokens" not in M.shape({"chat_template_kwargs": {"enable_thinking": True}}, prof)
    mine = M.shape({"reasoning_budget_tokens": 64, "chat_template_kwargs": M.no_think_kwargs()}, prof)
    assert mine["reasoning_budget_tokens"] == 64
    # with a template switch as well, both are sent
    two = M.from_dict({**INKLING_LIKE, "think_off_request": {"thinking_budget_tokens": 0}})
    got = M.shape(off, two)
    assert got["thinking_budget_tokens"] == 0 and got["chat_template_kwargs"] == dict(two.think_off)
    # the switch a caller WROTE (/no_think at the end of a user message) is taken out for this model: the fields stand in
    soft = M.from_dict({**INKLING_LIKE, "think_off": None, "think_off_request": {"reasoning_budget_tokens": 0},
                        "dialect": {"soft_no_think": ""}})
    wrote = {"messages": [{"role": "user", "content": "Is it done? /no_think"}]}
    got = M.shape(wrote, soft)
    assert got["messages"][0]["content"] == "Is it done?" and got["reasoning_budget_tokens"] == 0
    assert "reasoning_budget_tokens" not in M.shape({"messages": [{"role": "user", "content": "Is it done?"}]}, soft)
    # a model with a written switch of its own gets that switch, and no field for it
    own = M.from_dict({**INKLING_LIKE, "think_off": None, "think_off_request": {"reasoning_budget_tokens": 0},
                       "dialect": {"soft_no_think": "/nothink"}})
    got = M.shape(wrote, own)
    assert got["messages"][0]["content"].endswith("/nothink") and "reasoning_budget_tokens" not in got
    assert dict(M.from_dict({"name": "x", "think_off_request": None}).think_off_request) == {}
    # only the message being ANSWERED turns thinking off: the same words earlier are history
    earlier = {"messages": [{"role": "user", "content": "Quick one. /no_think"}, {"role": "assistant", "content": "Yes."},
                            {"role": "user", "content": "Now think hard about this."}]}
    got = M.shape(earlier, soft)
    assert got["messages"][0]["content"] == "Quick one." and "reasoning_budget_tokens" not in got
    parts = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}},
                                                         {"type": "text", "text": "What is this? /no_think"}]}]}
    assert M.shape(parts, soft)["reasoning_budget_tokens"] == 0
    two = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Look at this. /no_think"},
                                                       {"type": "text", "text": "And think about this one."}]}]}
    assert "reasoning_budget_tokens" not in M.shape(two, soft)             # the LAST text part is what carries it
    spaced = M.shape({"messages": [{"role": "user", "content": "Is it done? /no_think \n"}]}, soft)
    assert spaced["messages"][0]["content"] == "Is it done?" and spaced["reasoning_budget_tokens"] == 0
    after_tool = {"messages": [{"role": "user", "content": "Do it /no_think"}, {"role": "assistant", "content": "ok"},
                               {"role": "tool", "content": "done"}]}
    assert M.shape(after_tool, soft)["reasoning_budget_tokens"] == 0        # still the message being answered


def test_qwen_and_a_child_profile_are_sent_no_such_field():
    off = {"messages": [], "chat_template_kwargs": M.no_think_kwargs()}
    for prof in (M.QWEN38, M.GEMMA, M.GENERIC):
        assert M.shape(off, prof) is off or "reasoning_budget_tokens" not in M.shape(off, prof)
    assert M.shape(off, M.QWEN38) is off and dict(M.QWEN38.think_off_request) == {}
    known = {**M.BUILTIN, "capped": M.from_dict({"name": "capped", "think_off_request": {"reasoning_budget_tokens": 0}})}
    assert dict(M.from_dict({"name": "child", "extends": "capped"}, known).think_off_request) == {"reasoning_budget_tokens": 0}


@pytest.mark.parametrize("bad", [
    {"max_tokens": 0}, {"temperature": 0}, {"chat_template_kwargs": 0}, {"reasoning_budget_tokens": -1},
    {"reasoning_budget_tokens": 0.5}, {"reasoning_budget_tokens": True}, {"reasoning_budget_tokens": "0"},
    {"reasoning_budget_tokens": 10 ** 9}, ["reasoning_budget_tokens"], "reasoning_budget_tokens", 0,
])
def test_a_profile_file_cannot_set_any_other_request_field(bad):
    """A profile file is the operator's, but a typo in one must not rewrite requests."""
    with pytest.raises(M.ProfileError):
        M.from_dict({"name": "x", "think_off_request": bad})


# --- a level written after the head (§4VP) -------------------------------------------

def test_a_level_written_after_the_head_is_not_tracked_for_the_slot(monkeypatch):
    """Inkling 2 writes the level after the tool declarations and the system text:
    one head serves every level and thinking off. Booking a thinking-off call as
    "medium" (the template writes 0) was a label for nothing."""
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    monkeypatch.setattr(E, "known_server_default", lambda: "medium")
    low = {"chat_template_kwargs": {"reasoning_effort": "low"}}
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    assert E.level_is_in_head() and GhostAgent._call_level(low) == "low" and GhostAgent._call_level(off) == "medium"   # Qwen: as before
    after = M.from_dict({**INKLING_LIKE, "effort": {**INKLING_LIKE["effort"], "in_head": False}})
    M.set_main(after)
    assert not E.level_is_in_head()
    assert GhostAgent._call_level(low) is None and GhostAgent._call_level(off) is None and GhostAgent._call_level({}) is None
    me = SimpleNamespace(_main_effort_level="low")
    payload = {"messages": []}
    assert GhostAgent._warm_up_effort(me, payload) is payload               # no head per level is warmed
    M.set_main(M.from_dict(INKLING_LIKE))                                   # the same model declared "in the head"
    assert GhostAgent._warm_up_effort(me, payload)["chat_template_kwargs"] == {"reasoning_effort": "low"}
    M.set_main(M.GENERIC)
    assert not E.level_is_in_head()                                         # no levels at all: nothing to track either
    # a level left by a model whose level WAS in the head is cleared BEFORE it is read (run, not read)
    M.set_main(after)
    left = SimpleNamespace(_slot_level="medium")
    assert GhostAgent._slot_level_now(left, None) is None and left._slot_level is None
    M.set_main(None)                                                        # Qwen: never cleared, whatever the call's level
    kept = SimpleNamespace(_slot_level="medium")
    assert GhostAgent._slot_level_now(kept, None) == "medium" and GhostAgent._slot_level_now(kept, "low") == "medium"
    assert GhostAgent._slot_level_now(SimpleNamespace(), None) is None
    # … and it is the only way the booking function reads it
    tree = ast.parse(textwrap.dedent(inspect.getsource(GhostAgent)))
    reads = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == "slot_level"]
    assert reads == ["self._slot_level_now(cur_level)"]
    with pytest.raises(M.ProfileError):
        M.from_dict({**INKLING_LIKE, "effort": {**INKLING_LIKE["effort"], "in_head": "no"}})
    assert M.QWEN38.effort.in_head is True and M.from_dict(INKLING_LIKE).effort.in_head is True


def test_select_main_sets_it_and_warns_once_when_unsure(monkeypatch, caplog):
    monkeypatch.delenv("GHOST_MODEL_PROFILE", raising=False)
    monkeypatch.setattr(M, "profiles", lambda: KNOWN)
    with caplog.at_level("INFO", logger="GhostAgent"):
        assert M.select_main("qwen3.8-27b", "/m/Other-9.gguf") is OTHER and M.main() is OTHER
        assert "Other-9.gguf" in M.main_why()
        M.select_main("qwen3.8-27b", "/m/mystery.gguf")
        M.select_main("qwen3.8-27b", "/m/mystery.gguf")
    warned = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warned) == 1 and "mystery.gguf" in warned[0].getMessage() and "nothing assumed" in warned[0].getMessage()
    assert M.main() is M.GENERIC


def test_the_environment_names_the_profile(monkeypatch):
    monkeypatch.setattr(M, "profiles", lambda: KNOWN)
    monkeypatch.setenv("GHOST_MODEL_PROFILE", " Generic ")
    assert M.select_main("qwen3.8-27b", "/m/Qwen3.8.gguf") is M.GENERIC


# --- 6. what the server says ------------------------------------------------------

def test_probe_server(server):
    s = server(model_path="/m/Other-9.gguf", n_ctx=131072)
    assert M.probe_server(s.url) == {"model_path": "/m/Other-9.gguf", "n_ctx": 131072}
    assert M.probe_server(s.url + "/") == {"model_path": "/m/Other-9.gguf", "n_ctx": 131072}
    quiet = server(model_path=None)
    assert M.probe_server(quiet.url) == {}                    # a 404 is "did not say"
    quiet.close()
    assert M.probe_server(quiet.url, timeout=0.5) == {}       # so is a server that is down


@pytest.mark.parametrize("body", [{}, [], [1, 2], "loading", 0, {"error": {"message": "Loading model"}},
                                  {"model_path": ""}, {"model_path": "   "}, {"model_path": 7},
                                  {"default_generation_settings": {"n_ctx": 4096}}])
def test_a_200_that_names_no_model_is_not_an_answer(server, body):
    s = server()
    s.props = body
    assert M.probe_server(s.url) == {}


@pytest.mark.parametrize("settings, n", [
    (None, 0), ({}, 0), ({"n_ctx": "196608"}, 0), ({"n_ctx": True}, 0), ({"n_ctx": 10 ** 12}, 0),
    ({"n_ctx": M.MIN_WINDOW - 1}, 0), ({"n_ctx": M.MIN_WINDOW}, M.MIN_WINDOW), ({"n_ctx": M.MAX_WINDOW}, M.MAX_WINDOW),
    ({"n_ctx": M.MAX_WINDOW + 1}, 0), ("oops", 0),
])
def test_a_window_the_server_did_not_give_is_zero(server, settings, n):
    s = server()
    s.props = {"model_path": "/m/x.gguf", "default_generation_settings": settings}
    assert M.probe_server(s.url) == {"model_path": "/m/x.gguf", "n_ctx": n}


@pytest.mark.parametrize("flag, n_ctx, reserve, want", [
    (174000, 196608, 22608, 174000),      # the live pair: untouched
    (174000, 131072, 22608, 108464),      # the window shrank: lowered to what fits
    (65536, 196608, 22608, 65536),        # never raised
    (0, 196608, 22608, 174000),           # 0 = from the server
    (0, 131072, 4000, 127072),
    (174000, 0, 22608, 174000),           # a silent server: the flag stands
    (0, 0, 22608, 0),
    (8192, 8192, 22608, 6144),            # a small window keeps a quarter for the reply
    (0, 16000, 22608, 12000),
    (174000, 100, 22608, 174000),         # not a window (a stub, a misread field): the flag stands
    (174000, M.MIN_WINDOW - 1, 22608, 174000),
    (174000, M.MIN_WINDOW, 22608, 1536),  # the smallest window that is one
    (0, 4, 22608, 0),
    (0, 10 ** 12, 22608, 0),              # nor is a window of a trillion tokens
    (174000, M.MAX_WINDOW + 1, 22608, 174000),
])
def test_context_budget(flag, n_ctx, reserve, want):
    p = M.from_dict({"name": "p", "reply_reserve": reserve})
    assert M.context_budget(flag, n_ctx, p) == want


# --- 7. token counts against the server --------------------------------------------

class _Enc:
    """A tokenizer that counts `per` tokens for the calibration sample."""

    def __init__(self, n):
        self.n = n

    def encode(self, text, verbose=False):
        return [0] * (self.n if text == T.CALIBRATION_SAMPLE else max(1, len(text) // 2))


def test_without_a_tokenizer_the_estimate_becomes_the_servers_ratio(server):
    T.TOKEN_ENCODER = None
    text = "x" * 3000
    assert T.estimate_tokens(text) == 750                             # the old chars/4
    s = server(tokens=len(T.CALIBRATION_SAMPLE) // 2)                 # this model: 2 chars per token
    got = T.calibrate(s.url)
    assert got["chars_per_token"] == pytest.approx(2.0, abs=0.01) and got["local_tokens"] == 0
    assert got["drift"] is None and T.calibration() == got
    assert T.estimate_tokens(text) == pytest.approx(1500, abs=2)      # and the cached 750 is gone
    assert s.bodies == [("/tokenize", {"content": T.CALIBRATION_SAMPLE})]


def test_a_tokenizer_close_to_the_server_is_left_alone(server):
    T.TOKEN_ENCODER = _Enc(413)
    before = T.estimate_tokens("y" * 400)
    got = T.calibrate(server(tokens=424).url)                         # the live figures, 2026-11-26
    assert got["encoder_scale"] == 1.0 and got["drift"] == pytest.approx(-0.0259, abs=1e-4)
    assert T.estimate_tokens("y" * 400) == before == 200


def test_a_tokenizer_that_is_not_the_servers_is_scaled_and_said(server, caplog):
    T.TOKEN_ENCODER = _Enc(300)
    assert T.estimate_tokens("z" * 400) == 200
    s = server(tokens=435)                                            # the Inkling case: +45 %
    with caplog.at_level("WARNING", logger="GhostAgent"):
        got = T.calibrate(s.url)
        T.calibrate(s.url)                                            # the same finding is said once
    assert got["encoder_scale"] == pytest.approx(1.45) and got["drift"] == pytest.approx(-0.3103, abs=1e-4)
    assert T.estimate_tokens("z" * 400) == 290
    assert len([r for r in caplog.records if "NOT the server's" in r.getMessage()]) == 1
    T.reset_calibration()
    assert T.estimate_tokens("z" * 400) == 200


def test_a_missing_tokenizer_is_still_said_when_the_profile_has_one(server, caplog):
    T.TOKEN_ENCODER = None
    T.calibrate(server(tokens=400).url)
    T._reset_fallback_warning()
    with caplog.at_level("WARNING", logger="GhostAgent"):
        T.estimate_tokens("some text that is not cached yet 1")          # Qwen has a tokenizer: it failed to load
    assert [r for r in caplog.records if "falling back" in r.getMessage()]
    caplog.clear()
    M.set_main(OTHER)                                                    # this model has none: nothing to report
    T._reset_fallback_warning()
    with caplog.at_level("WARNING", logger="GhostAgent"):
        T.estimate_tokens("some text that is not cached yet 2")
    assert not [r for r in caplog.records if "falling back" in r.getMessage()]


def test_a_count_made_across_a_change_of_tokenizer_is_not_kept():
    class _Switching:
        def encode(self, text, verbose=False):
            T.clear_token_cache()                       # the re-sync thread swaps the tokenizer mid-count
            return [0] * 7

    T.TOKEN_ENCODER = _Switching()
    assert T.estimate_tokens("short text") == 7
    T.TOKEN_ENCODER = _Enc(1)
    assert T.estimate_tokens("short text") == 5         # counted again, not the 7 from the old tokenizer
    assert T.estimate_tokens("short text") == 5


def test_the_calibration_is_swapped_whole(server):
    T.TOKEN_ENCODER = None
    first = T._calibration
    T.calibrate(server(tokens=400).url)
    assert T._calibration is not first and first == {}  # a reader holding the old dict never sees it half-written
    held = T._calibration
    T.reset_calibration()
    assert held["server_tokens"] == 400 and T.calibration() == {}


@pytest.mark.parametrize("local, server_n, scaled", [
    (400, 412, False),      # 3 % exactly: left alone
    (400, 413, True),       # just past it: scaled
    (400, 389, False),      # over the server's count as well as under it
    (400, 388, True),
])
def test_the_band_inside_which_a_tokenizer_is_left_alone(server, local, server_n, scaled):
    T.TOKEN_ENCODER = _Enc(local)
    got = T.calibrate(server(tokens=server_n).url)
    assert (got["encoder_scale"] != 1.0) is scaled


def test_a_scaled_count_is_rounded_not_cut(server):
    T.TOKEN_ENCODER = _Enc(300)
    T.calibrate(server(tokens=435).url)                 # ×1.45
    assert T.estimate_tokens("z" * 22) == 16            # 11 × 1.45 = 15.95
    assert T.estimate_tokens("z" * 4) == 3              # 2 × 1.45 = 2.9


@pytest.mark.parametrize("answer", [{"tokens": []}, {"tokens": [1]}, {"tokens": [[1, "a"]]},
                                    {"tokens": list(range(200000))}, {"tokens": "many"}, {}, []])
def test_a_count_no_tokenizer_gives_is_not_believed(server, answer, caplog):
    """One token for the whole sample would make every count 0 (nothing is ever
    pruned); 200,000 would make everything too large."""
    for enc in (None, _Enc(413)):
        T.reset_calibration()
        T.TOKEN_ENCODER = enc
        before = T.estimate_tokens("w" * 4000)
        s = server()
        s.tokenize_body = answer
        with caplog.at_level("WARNING", logger="GhostAgent"):
            assert T.calibrate(s.url) == {}
            T.calibrate(s.url)
        assert T.calibration() == {} and T.estimate_tokens("w" * 4000) == before
    if isinstance(answer, dict) and isinstance(answer.get("tokens"), list) and answer["tokens"]:
        assert len([r for r in caplog.records if "not believed" in r.getMessage()]) == 2      # once per calibration run


def test_the_edges_of_what_a_calibration_may_set(server):
    T.TOKEN_ENCODER = None
    n = len(T.CALIBRATION_SAMPLE)
    assert T.calibrate(server(tokens=n).url)["chars_per_token"] == 1.0          # 1 char per token: the floor
    T.reset_calibration()
    assert T.calibrate(server(tokens=n + 1).url) == {}                          # under it
    assert T.calibrate(server(tokens=n // 12 + 1).url)["chars_per_token"] <= 12.0
    T.reset_calibration()
    assert T.calibrate(server(tokens=n // 12 - 1).url) == {}
    T.TOKEN_ENCODER = _Enc(400)
    assert T.calibrate(server(tokens=800).url)["encoder_scale"] == 2.0
    assert T.calibrate(server(tokens=200).url)["encoder_scale"] == 0.5
    T.reset_calibration()
    assert T.calibrate(server(tokens=801).url) == {} and T.calibrate(server(tokens=199).url) == {}


def test_setting_and_putting_back_what_counts(server):
    old = _Enc(300)
    T.TOKEN_ENCODER = old
    T.calibrate(server(tokens=435).url)
    assert T.estimate_tokens("q" * 400) == 290
    kept = T.counting_state()
    T.set_encoder(SimpleNamespace(encode=lambda text, verbose=False: [0] * (len(text) // 4)))
    assert T.estimate_tokens("q" * 400) == 145          # 100 × 1.45: counted again — the cache went with the old tokenizer
    T.reset_calibration()
    assert T.estimate_tokens("q" * 400) == 100
    T.restore_counting(kept)
    assert T.TOKEN_ENCODER is old and T.calibration()["encoder_scale"] == pytest.approx(1.45)
    assert T.estimate_tokens("q" * 400) == 290
    T.set_encoder(None)
    T.reset_calibration()
    assert T.estimate_tokens("q" * 400) == 100


def test_a_new_tokenizer_re_arms_the_missing_tokenizer_warning(caplog):
    T.TOKEN_ENCODER = None
    with caplog.at_level("WARNING", logger="GhostAgent"):
        T.estimate_tokens("first text with no tokenizer")
        T.estimate_tokens("second text with no tokenizer")
        assert len([r for r in caplog.records if "falling back" in r.getMessage()]) == 1       # said once
        T.set_encoder(_Enc(1))                              # one arrives …
        T.set_encoder(None)                                 # … and is lost again: that is news
        T.estimate_tokens("third text with no tokenizer")
    assert len([r for r in caplog.records if "falling back" in r.getMessage()]) == 2


def test_exact_needs_a_verified_tokenizer_and_equal_counts(server):
    enc = _Enc(424)
    enc.exact = True
    T.TOKEN_ENCODER = enc
    assert T.calibrate(server(tokens=424).url)["exact"] is True
    assert T.calibrate(server(tokens=425).url)["exact"] is False            # verified once, but not counting alike now
    T.TOKEN_ENCODER = _Enc(424)
    assert T.calibrate(server(tokens=424).url)["exact"] is False            # alike, but nobody verified it


def test_loading_no_tokenizer_drops_the_old_counts(tmp_path, capsys):
    T.TOKEN_ENCODER = _Enc(1)
    assert T.estimate_tokens("q" * 400) == 200
    M.set_main(OTHER)
    T.load_tokenizer(tmp_path, repo="", profile_name="the-new-one")
    assert T.estimate_tokens("q" * 400) == 100          # not the 200 the old tokenizer left in the cache
    assert "the-new-one" in capsys.readouterr().out     # named for the profile being loaded


def test_a_silent_server_changes_nothing(server):
    T.TOKEN_ENCODER = None
    s = server(tokens=0)                                              # /tokenize answers 404
    assert T.calibrate(s.url) == {} and T.calibration() == {}
    assert T.estimate_tokens("x" * 3000) == 750
    s.close()
    assert T.calibrate(s.url, timeout=0.5) == {}


def test_a_profile_without_a_tokenizer_loads_none_and_asks_nobody(tmp_path, monkeypatch):
    M.set_main(OTHER)
    monkeypatch.setattr(T, "_auto_tokenizer", lambda: pytest.fail("no tokenizer may be fetched"))
    T.TOKEN_ENCODER = _Enc(1)
    assert T.load_tokenizer(tmp_path / "none") is None and T.TOKEN_ENCODER is None


def test_where_a_profiles_tokenizer_is_kept(tmp_path):
    assert T.tokenizer_path(tmp_path) == tmp_path / "tokenizer"                       # Qwen: where it always was
    p = M.from_dict({"name": "q4", "tokenizer": {"repo": "org/q4"}})
    assert T.tokenizer_path(tmp_path, p) == tmp_path / "tokenizers-q4"
    assert T.tokenizer_path(tmp_path, OTHER) == tmp_path / "tokenizer"                # none: never read


def test_the_qwen_repo_is_the_one_it_was():
    assert T.QWEN_MODEL_ID == M.QWEN38.tokenizer_repo == "Qwen/Qwen3-30B-A3B"


# --- 8. boot and re-sync -----------------------------------------------------------

@pytest.fixture
def loaded(monkeypatch):
    """`load_tokenizer` replaced: it records the call and "loads" a stub
    tokenizer for a profile that has one (none for a profile without)."""
    class _Calls(list):
        fake = None
    calls = _Calls()

    def fake(path, repo=None, profile_name=None):
        calls.append((Path(path).name, repo, profile_name))
        if fake.fail == "quietly":
            return None                                 # a load that fails the way the real one does: None, nothing touched
        if fake.fail:
            T.TOKEN_ENCODER = _Enc(7)                   # the real loader sets the tokenizer, THEN stores it
            raise OSError("disk full")
        T.TOKEN_ENCODER = _Enc(424) if repo else None
        T.clear_token_cache()
        return T.TOKEN_ENCODER
    fake.fail = False
    calls.fake = fake
    monkeypatch.setattr(T, "load_tokenizer", fake)
    monkeypatch.setattr(M, "profiles", lambda: KNOWN)
    monkeypatch.setattr(M, "reload_if_changed", lambda: False)
    monkeypatch.setattr(M, "files_signature", lambda *a: ("fixed",))
    return calls


def _args(url, max_context=174000, model="qwen3.8-27b"):
    return SimpleNamespace(upstream_url=url, max_context=max_context, model=model)


QWEN_LOAD = ("tokenizer", "Qwen/Qwen3-30B-A3B", "qwen38")
OTHER_LOAD = ("tokenizer", "", "other")


def test_boot_lines_the_agent_up_with_the_server(server, loaded, tmp_path):
    s = server(model_path="/m/Other-9.gguf", n_ctx=131072, tokens=500)
    args = _args(s.url + "/v1")
    got = B.sync(args, tmp_path, boot=True)
    assert M.main() is OTHER and got["profile"] == "other" and got["server_model"] == "Other-9.gguf"
    assert args.max_context == 131072 - 4000 == got["max_context"] and got["max_context_flag"] == 174000
    assert got["n_ctx"] == 131072 and got["tokenizer"] == "none" and loaded == [OTHER_LOAD]
    assert got["tokens"]["server_tokens"] == 500 and B.booted() and B.last() == got and "error" not in got
    assert "other" in B.describe(got) and "estimated" in B.describe(got)


def test_boot_with_a_silent_server_keeps_the_flags(server, loaded, tmp_path):
    s = server()
    s.close()
    args = _args(s.url)
    got = B.sync(args, tmp_path, boot=True, timeout=0.5)
    assert M.main() is M.QWEN38 and args.max_context == 174000 and "tokens" not in got
    assert loaded == [QWEN_LOAD] and got["tokenizer"] == "profile" and B.booted()
    auto = _args(s.url, max_context=0)
    B.sync(auto, tmp_path, boot=True, timeout=0.5)
    assert auto.max_context == B.FALLBACK_CONTEXT


def test_the_live_pair_is_untouched(server, loaded, tmp_path):
    args = _args(server(model_path="/m/DaisyRose-27B-Heretic-v2-MTP-Q4_K_M.gguf", n_ctx=196608).url)
    B.sync(args, tmp_path, boot=True)
    assert M.main() is M.QWEN38 and args.max_context == 174000


def test_a_resync_follows_a_llama_restart_and_never_raises_the_flag(server, loaded, tmp_path, caplog):
    s = server(model_path="/m/Qwen3.8-27B.gguf", n_ctx=196608)
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    assert args.max_context == 174000 and loaded == [QWEN_LOAD]
    asked = len(s.bodies)
    B.sync(args, now=100.0)                                 # the same model and window: nothing reloaded,
    assert len(loaded) == 1 and M.main() is M.QWEN38 and len(s.bodies) == asked      # nothing counted again
    s.model_path, s.n_ctx = "/m/Other-9.gguf", 65536        # llama restarted on another model
    with caplog.at_level("WARNING", logger="GhostAgent"):
        got = B.sync(args, now=200.0)
        B.sync(args, now=300.0)                             # the same answer again: said once
    assert M.main() is OTHER and args.max_context == 61536 and loaded[-1] == OTHER_LOAD and len(loaded) == 2
    assert got["profile"] == "other" and B.last() == got
    said = [r.getMessage() for r in caplog.records if "does not fit" in r.getMessage()]
    assert len(said) == 1
    assert "--max-context 174000" in said[0] and "65536-token window" in said[0]
    assert "4000 left for a reply" in said[0] and "using 61536" in said[0]
    s.n_ctx = 262144                                        # the same model, a larger window
    B.sync(args, now=400.0)
    assert M.main() is OTHER and args.max_context == 174000 and len(loaded) == 2     # the flag is the ceiling
    s.n_ctx = 8192                                          # … and a small one: a quarter is kept for the reply
    caplog.clear()
    with caplog.at_level("WARNING", logger="GhostAgent"):
        B.sync(args, now=500.0)
    assert args.max_context == 6144
    assert "2048 left for a reply" in [r.getMessage() for r in caplog.records if "does not fit" in r.getMessage()][0]
    s.model_path, s.n_ctx = "/m/Qwen3.8-27B.gguf", 131072
    B.sync(args, now=600.0)
    assert M.main() is M.QWEN38 and args.max_context == 108464 and loaded[-1] == QWEN_LOAD


def test_a_resync_asks_at_most_once_a_minute(server, loaded, tmp_path):
    s = server(model_path="/m/Qwen3.8-27B.gguf")
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=1000.0)
    s.model_path = "/m/Other-9.gguf"
    assert B.sync(args, now=1000.0 + B.RESYNC_EVERY_S - 0.001)["profile"] == "qwen38" and M.main() is M.QWEN38
    assert B.sync(args, now=1000.0 + B.RESYNC_EVERY_S)["profile"] == "other" and M.main() is OTHER


def _other_booted(server, loaded, tmp_path, flag=174000):
    s = server(model_path="/m/Other-9.gguf", n_ctx=65536, tokens=500)
    args = _args(s.url, max_context=flag)
    summary = B.sync(args, tmp_path, boot=True, now=0.0)
    assert M.main() is OTHER and args.max_context == 61536 and T.calibration()["chars_per_token"] == 2.7
    return s, args, summary


@pytest.mark.parametrize("gone", ["404", "down", "{}", "error body", "list", "no model name"])
def test_a_server_that_does_not_name_its_model_changes_nothing(server, loaded, tmp_path, gone):
    """llama answers /props with an error — or a 200 with nothing in it — while
    it loads. That must not hand the agent back to the --model profile, the
    other tokenizer, an uncalibrated count and the unlowered budget."""
    s, args, booted = _other_booted(server, loaded, tmp_path)
    if gone == "404":
        s.model_path = None
    elif gone == "down":
        s.close()
    else:
        s.props = {"{}": {}, "error body": {"error": {"code": 503, "message": "Loading model"}}, "list": [],
                   "no model name": {"default_generation_settings": {"n_ctx": 262144}}}[gone]
    assert B.sync(args, now=500.0, timeout=0.5) == booted
    assert M.main() is OTHER and args.max_context == 61536 and loaded == [OTHER_LOAD]
    assert T.calibration()["chars_per_token"] == 2.7 and B.last() == booted


@pytest.mark.parametrize("flag, want", [(174000, 61536), (0, 61536)])
def test_a_window_the_server_does_not_give_leaves_the_budget(server, loaded, tmp_path, flag, want):
    s, args, _ = _other_booted(server, loaded, tmp_path, flag)
    s.props = {"model_path": "/m/Other-9.gguf"}                             # the same model, no window
    B.sync(args, now=500.0)
    assert args.max_context == want
    s.props = {"model_path": "/m/Other-9.gguf", "default_generation_settings": {"n_ctx": "big"}}
    B.sync(args, now=600.0)
    assert args.max_context == want


def test_the_budget_taken_from_the_server_survives_a_silent_resync(server, loaded, tmp_path):
    s = server(model_path="/m/Qwen3.8-27B.gguf", n_ctx=196608)
    args = _args(s.url, max_context=0)
    B.sync(args, tmp_path, boot=True, now=0.0)
    assert args.max_context == 174000
    s.model_path = None
    B.sync(args, now=500.0)
    assert args.max_context == 174000                       # not the 65,536 fallback


def test_a_change_of_model_drops_the_old_models_tokenizer_and_scale(server, loaded, tmp_path):
    s = server(model_path="/m/Qwen3.8-27B.gguf", tokens=600)                # the stub tokenizer counts 424: ×1.415
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    assert T.calibration()["encoder_scale"] == pytest.approx(1.4151) and T.estimate_tokens("x" * 400) == 283
    s.model_path, s.tokens = "/m/Other-9.gguf", 0           # another model; its /tokenize does not answer
    got = B.sync(args, now=100.0)
    assert M.main() is OTHER and got["tokens"] == {} and got["tokenizer"] == "none"
    assert T.TOKEN_ENCODER is None                          # Qwen's tokenizer is not left counting for it
    assert T.calibration() == {} and T.estimate_tokens("x" * 400) == 100


def test_another_model_that_gives_no_window_falls_back_to_the_flag(server, loaded, tmp_path):
    s, args, _ = _other_booted(server, loaded, tmp_path)                    # lowered to 61536 for a 65536 window
    s.props = {"model_path": "/m/Qwen3.8-27B.gguf"}                         # another model; its window is not given
    got = B.sync(args, now=500.0)
    assert M.main() is M.QWEN38 and args.max_context == 174000 == got["max_context"]       # the old model's fit does not carry over


def test_an_error_leaves_the_summary_after_a_good_sync(server, loaded, tmp_path):
    s = server(model_path="/m/Qwen3.8-27B.gguf")
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    s.model_path = "/m/Qwen4-30B.gguf"
    loaded.fake.fail = True
    assert "error" in B.sync(args, now=100.0) and "error" in B.last()
    loaded.fake.fail = False
    assert "error" not in B.sync(args, now=200.0) and "error" not in B.last()
    # … also when the good sync has nothing to redo: the server went back to the model it had
    s.model_path = "/m/Other-9.gguf"
    loaded.fake.fail = True
    assert "error" in B.sync(args, now=300.0)
    s.model_path = "/m/Qwen4-30B.gguf"
    assert "error" not in B.sync(args, now=400.0) and "error" not in B.last()


def test_the_model_files_tokenizer_starts_from_a_clean_calibration(server, loaded, tmp_path, monkeypatch):
    s = server(model_path="/m/Qwen3.8-27B.gguf", tokens=600)                # the fallback tokenizer: scaled ×1.415
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    assert T.calibration()["encoder_scale"] > 1.4
    exact = _Enc(424)
    exact.exact = True
    monkeypatch.setattr(B._gguf, "from_model_file", lambda *a, **k: exact)  # the next model's own tokenizer is built
    s.model_path, s.tokens = "/m/Qwen4-30B.gguf", 0                         # … and the token check gets no answer
    got = B.sync(args, now=100.0)
    assert got["tokenizer"] == "model file" and T.TOKEN_ENCODER is exact
    assert T.estimate_tokens("x" * 400) == 200                              # NOT the old model's 283
    assert "model file's own tokenizer" in B.describe(got)                  # verified id for id, whatever the check said


def test_a_tokenizer_build_the_server_did_not_answer_is_owed(server, loaded, tmp_path, monkeypatch):
    """One failed /tokenize while the model file's tokenizer is checked must
    not leave the agent on the fallback until the next restart."""
    outcomes = iter([B._gguf.SERVER_SILENT, B._gguf.SERVER_SILENT, B._gguf.ADOPTED])
    exact = _Enc(424)
    exact.exact = True
    builds = []

    def build(model_path, sysdir, base, timeout):
        builds.append(model_path)
        B._gguf.last_outcome = next(outcomes)
        return exact if B._gguf.last_outcome == B._gguf.ADOPTED else None
    monkeypatch.setattr(B._gguf, "from_model_file", build)
    s = server(model_path="/m/Qwen3.8-27B.gguf", tokens=424)
    args = _args(s.url)
    assert B.sync(args, tmp_path, boot=True, now=0.0)["tokenizer"] == "profile"
    assert B.sync(args, now=100.0)["tokenizer"] == "profile" and len(builds) == 2
    assert B.sync(args, now=200.0)["tokenizer"] == "model file" and T.TOKEN_ENCODER is exact
    B.sync(args, now=300.0)
    assert len(builds) == 3                                                 # adopted: not built again


@pytest.mark.parametrize("outcome", ["no rule matches", "no file here", "failed", "off"])
def test_a_tokenizer_build_that_cannot_succeed_is_not_repeated(server, loaded, tmp_path, monkeypatch, outcome):
    builds = []

    def build(*a, **k):
        builds.append(1)
        B._gguf.last_outcome = outcome
    monkeypatch.setattr(B._gguf, "from_model_file", build)
    args = _args(server(model_path="/m/Qwen3.8-27B.gguf").url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    B.sync(args, now=100.0)
    B.sync(args, now=200.0)
    assert len(builds) == 1


def test_a_token_check_that_got_no_answer_is_tried_again(server, loaded, tmp_path):
    s = server(model_path="/m/Other-9.gguf", tokens=0)      # /props answers, /tokenize does not (yet)
    args = _args(s.url)
    assert B.sync(args, tmp_path, boot=True, now=0.0)["tokens"] == {}
    s.tokens = 500
    got = B.sync(args, now=100.0)                           # nothing else changed
    assert got["tokens"]["server_tokens"] == 500 and T.calibration()["chars_per_token"] == 2.7
    assert loaded == [OTHER_LOAD]                           # only the check was repeated
    n = len(s.bodies)
    B.sync(args, now=200.0)
    assert len(s.bodies) == n                               # and once it has an answer, not again


def test_a_new_profile_file_is_picked_up_by_the_next_resync(server, tmp_path, monkeypatch):
    """The warning says "add a profile file": adding one must be enough."""
    monkeypatch.setenv("GHOST_MODEL_PROFILES", str(tmp_path / "profiles"))
    (tmp_path / "profiles").mkdir()
    monkeypatch.setattr(T, "load_tokenizer", lambda path, repo=None, profile_name=None: None)
    s = server(model_path="/m/Other-9.gguf", n_ctx=65536)
    args = _args(s.url)
    assert B.sync(args, tmp_path, boot=True, now=0.0)["profile"] == "generic"       # fits nothing yet: nothing assumed
    assert args.max_context == 65536 - 16384
    (tmp_path / "profiles" / "other.json").write_text(json.dumps(OTHER_JSON))
    got = B.sync(args, now=100.0)                           # the server did not change; the files did
    assert got["profile"] == "other" and M.main().name == "other" and args.max_context == 61536
    asked = len(s.bodies)
    B.sync(args, now=200.0)
    assert len(s.bodies) == asked and M.main().name == "other"


def test_a_resync_that_fails_half_way_puts_everything_back_and_tries_again(server, loaded, tmp_path, caplog):
    s = server(model_path="/m/Qwen3.8-27B.gguf", n_ctx=196608, tokens=600)
    args = _args(s.url)
    booted = B.sync(args, tmp_path, boot=True, now=0.0)
    enc, cal = T.TOKEN_ENCODER, T.calibration()
    s.model_path, s.n_ctx = "/m/Qwen4-30B.gguf", 32768      # a model whose tokenizer cannot be stored
    loaded.fake.fail = True
    with caplog.at_level("WARNING", logger="GhostAgent"):
        got = B.sync(args, now=100.0)
    assert "OSError" in got["error"] and got["profile"] == "qwen38"         # the summary is the state in force
    assert M.main() is M.QWEN38 and "Qwen3.8-27B.gguf" in M.main_why()
    assert T.TOKEN_ENCODER is enc and T.calibration() == cal and args.max_context == 174000
    assert B.last() == got and {k: v for k, v in got.items() if k != "error"} == booted
    loaded.fake.fail = False
    again = B.sync(args, now=200.0)                         # not latched: the next minute tries again
    assert "error" not in again and again["profile"] == "qwen4" and M.main().name == "qwen4"
    assert args.max_context == 32768 - 8192


@pytest.mark.parametrize("flag, n_ctx, want", [(0, 32768, B.FALLBACK_CONTEXT), (174000, 32768, 174000)])
def test_a_boot_that_fails_still_ends_with_a_profile_a_tokenizer_and_a_budget(server, loaded, tmp_path, flag, n_ctx, want):
    s = server(model_path="/m/Other-9.gguf", n_ctx=n_ctx)
    args = _args(s.url, max_context=flag)
    real = M.choose_main
    box = {"fail": True}

    def flaky(*a, **k):
        if box["fail"]:
            raise RuntimeError("boom")
        return real(*a, **k)
    import unittest.mock as mock
    with mock.patch.object(M, "choose_main", flaky):
        got = B.sync(args, tmp_path, boot=True, now=0.0)
        assert "RuntimeError" in got["error"] and B.booted()
        assert got["profile"] == "qwen38" == M.main().name                  # what is in force, not what was meant
        assert args.max_context == want != 0 and got["max_context"] == want
        assert loaded == [QWEN_LOAD] and T.TOKEN_ENCODER is not None        # a boot still loads a tokenizer
        box["fail"] = False
        again = B.sync(args, now=100.0)                     # and the first re-sync does the whole job
    assert "error" not in again and M.main() is OTHER and args.max_context == n_ctx - 4000


def test_a_resync_that_fails_after_the_switch_puts_the_profile_back(server, loaded, tmp_path, monkeypatch):
    s = server(model_path="/m/Qwen3.8-27B.gguf", tokens=600)
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    enc, why = T.TOKEN_ENCODER, M.main_why()
    s.model_path = "/m/Other-9.gguf"
    real = T.calibrate
    monkeypatch.setattr(T, "calibrate", lambda *a, **k: 1 / 0)             # after the new profile was announced
    got = B.sync(args, now=100.0)
    assert "ZeroDivisionError" in got["error"]
    assert M.main() is M.QWEN38 and M.main_why() == why and T.TOKEN_ENCODER is enc and args.max_context == 174000
    monkeypatch.setattr(T, "calibrate", real)
    assert B.sync(args, now=200.0)["profile"] == "other" and M.main() is OTHER


def test_a_tokenizer_that_fails_to_load_does_not_leave_the_old_models(server, loaded, tmp_path):
    s = server(model_path="/m/Other-9.gguf", tokens=500)
    args = _args(s.url)
    B.sync(args, tmp_path, boot=True, now=0.0)
    T.TOKEN_ENCODER = _Enc(424)                             # some other model's tokenizer is in place
    loaded.fake.fail = "quietly"                            # the new one cannot be fetched (offline, no local copy)
    s.model_path = "/m/Qwen3.8-27B.gguf"
    got = B.sync(args, now=100.0)
    assert got["profile"] == "qwen38" and got["tokenizer"] == "none" and "error" not in got
    assert T.TOKEN_ENCODER is None and got["tokens"]["local_tokens"] == 0


def test_a_boot_whose_tokenizer_cannot_load_is_not_left_at_zero(server, loaded, tmp_path):
    loaded.fake.fail = True
    args = _args(server(model_path="/m/Qwen3.8-27B.gguf").url, max_context=0)
    got = B.sync(args, tmp_path, boot=True)
    assert "OSError" in got["error"] and args.max_context == B.FALLBACK_CONTEXT and B.booted()


def test_no_resync_before_a_boot(server, loaded):
    s = server(model_path="/m/Other-9.gguf")
    args = _args(s.url)
    assert B.sync(args) == {} and M.main() is M.QWEN38 and args.max_context == 174000
    assert loaded == [] and s.bodies == []


def test_one_sync_at_a_time(server, loaded, tmp_path):
    s = server(model_path="/m/Qwen3.8-27B.gguf")
    args = _args(s.url)
    booted = B.sync(args, tmp_path, boot=True, now=0.0)
    s.model_path = "/m/Other-9.gguf"
    assert B._running.acquire(blocking=False)               # a slow sync is still running
    try:
        assert B.sync(args, now=500.0) == booted and M.main() is M.QWEN38
    finally:
        B._running.release()
    assert B.sync(args, now=600.0)["profile"] == "other"


def test_the_boot_banner_says_how_tokens_are_counted():
    base = {"profile": "p", "why": "w"}
    assert "unknown" in B.describe({}) and "unknown" in B.describe({"error": "x"})
    assert "not checked" in B.describe(base)
    assert "within 2.6%" in B.describe({**base, "tokens": {"local_tokens": 413, "encoder_scale": 1.0, "drift": -0.0259}})
    assert "NOT the server's" in B.describe({**base, "tokens": {"local_tokens": 300, "encoder_scale": 1.45}})
    assert "estimated at 3.2" in B.describe({**base, "tokens": {"local_tokens": 0, "chars_per_token": 3.2}})
    assert "model file's own tokenizer" in B.describe(
        {**base, "tokens": {"local_tokens": 424, "encoder_scale": 1.0, "drift": 0.0, "exact": True}})


async def test_the_warm_up_resyncs_only_in_a_booted_process(monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    calls = []
    monkeypatch.setattr(B, "sync", lambda args: calls.append(args))
    me = SimpleNamespace(context=SimpleNamespace(args=SimpleNamespace(max_context=174000)))
    await GhostAgent._model_resync(me)
    assert calls == []                                      # a test's agent, a script: no server is asked
    monkeypatch.setattr(B, "booted", lambda: True)
    await GhostAgent._model_resync(me)
    assert calls == [me.context.args]
    monkeypatch.setattr(B, "sync", lambda args: 1 / 0)
    await GhostAgent._model_resync(me)                      # and it never fails a warm-up


async def test_a_resync_that_changes_the_model_reaches_what_was_read_once(monkeypatch):
    """The context manager took its budget at first use, and the looked-up
    server level belongs to the old model."""
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setattr(B, "booted", lambda: True)
    box = {"server_model": "Qwen3.8.gguf"}
    monkeypatch.setattr(B, "last", lambda: dict(box))
    cm = SimpleNamespace(max_tokens=174000)
    me = SimpleNamespace(context=SimpleNamespace(args=SimpleNamespace(max_context=174000)), _context_manager=cm)
    monkeypatch.setattr(B, "sync", lambda args: None)
    await GhostAgent._model_resync(me)                      # the first one only records what is in force
    E._server_default.update(level="xhigh", at=1.0)
    await GhostAgent._model_resync(me)
    assert cm.max_tokens == 174000 and E.known_server_default() == "xhigh"         # nothing changed: nothing touched

    def switch(args):
        args.max_context = 61536
        M.set_main(OTHER, "the server runs Other-9.gguf")
        box["server_model"] = "Other-9.gguf"
    monkeypatch.setattr(B, "sync", switch)
    await GhostAgent._model_resync(me)
    assert cm.max_tokens == 61536 and E.known_server_default() == ""
    E._server_default.update(level="low", at=1.0)

    def same_profile_other_file(args):
        box["server_model"] = "Other-9-v2.gguf"             # another finetune under the same profile
    monkeypatch.setattr(B, "sync", same_profile_other_file)
    await GhostAgent._model_resync(me)
    assert E.known_server_default() == "" and cm.max_tokens == 61536
    me.context.args.max_context = 0                         # a budget of nothing never reaches the context manager
    monkeypatch.setattr(B, "sync", lambda args: None)
    await GhostAgent._model_resync(me)
    assert cm.max_tokens == 61536


async def test_a_sync_that_outlives_the_wait_is_applied_by_the_next_warm_up(monkeypatch):
    import threading
    from ghost_agent.core import agent as A
    monkeypatch.setattr(B, "booted", lambda: True)
    monkeypatch.setattr(A, "MODEL_RESYNC_MAX_S", 0.15)
    box = {"server_model": "Qwen3.8.gguf"}
    monkeypatch.setattr(B, "last", lambda: dict(box))
    cm = SimpleNamespace(max_tokens=174000)
    me = SimpleNamespace(context=SimpleNamespace(args=SimpleNamespace(max_context=174000)), _context_manager=cm)
    monkeypatch.setattr(B, "sync", lambda args: None)
    await A.GhostAgent._model_resync(me)
    E._server_default.update(level="xhigh", at=1.0)
    done = threading.Event()

    def slow(args):
        import time as _t
        _t.sleep(0.6)                                       # still running when the warm-up stops waiting
        args.max_context = 30000
        M.set_main(OTHER, "the server runs Other-9.gguf")
        box["server_model"] = "Other-9.gguf"
        done.set()
    monkeypatch.setattr(B, "sync", slow)
    await A.GhostAgent._model_resync(me)
    assert cm.max_tokens == 174000                          # nothing to apply yet
    assert done.wait(5)
    monkeypatch.setattr(B, "sync", lambda args: None)
    await A.GhostAgent._model_resync(me)                    # the next warm-up finds it
    assert cm.max_tokens == 30000 and E.known_server_default() == ""


async def test_a_sync_that_hangs_does_not_hold_the_warm_up(monkeypatch):
    import threading
    from ghost_agent.core import agent as A
    release = threading.Event()
    monkeypatch.setattr(B, "booted", lambda: True)
    monkeypatch.setattr(B, "sync", lambda args: release.wait(10))
    monkeypatch.setattr(A, "MODEL_RESYNC_MAX_S", 0.2)
    me = SimpleNamespace(context=SimpleNamespace(args=SimpleNamespace(max_context=1)))
    import time as _time
    t0 = _time.monotonic()
    await A.GhostAgent._model_resync(me)
    release.set()
    assert _time.monotonic() - t0 < 5.0


def test_every_warm_up_resyncs_loud_or_quiet():
    """The re-warm loop's ticks are all quiet: a re-sync under `if not quiet`
    ran at boot and never again (the §4VM review's first finding). It is the
    first statement of the warm-up's body: nothing can return or branch before it."""
    from ghost_agent.core.agent import GhostAgent
    fn = ast.parse(inspect.getsource(GhostAgent._warm_up_main_prefix_once).lstrip()).body[0]
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    outer = body[0]
    assert isinstance(outer, ast.Try)
    stmts = outer.body
    first_call = next(i for i, n in enumerate(stmts) if "self._model_resync" in ast.unparse(n))
    assert ast.unparse(stmts[first_call]) == "await self._model_resync()"
    before = [ast.unparse(n) for n in stmts[:first_call]]
    assert before == ["llm = getattr(self.context, 'llm_client', None)", "if llm is None:\n    return"]


def test_the_boot_runs_the_sync_in_place_of_the_tokenizer_load():
    import ghost_agent.main as main_mod
    tree = ast.parse(inspect.getsource(main_mod))
    boots = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and ast.unparse(c.func) == "_model_boot.sync"]
    assert len(boots) == 1
    assert {k.arg: ast.unparse(k.value) for k in boots[0].keywords} == {"boot": "True"}
    assert not [c for c in ast.walk(tree) if isinstance(c, ast.Call) and ast.unparse(c.func) == "load_tokenizer"]


# --- 9. the effort module reads the profile -----------------------------------------

def test_effort_is_off_for_a_model_without_levels(monkeypatch):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    assert E.enabled()
    M.set_main(M.GENERIC)
    assert not E.enabled()
    assert E.level_in_render("Reasoning effort is set to low") == "medium"
    assert E.escalation_note("a step failed") == ""
    M.set_main(OTHER)
    assert E.enabled()
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    assert not E.enabled()


def test_the_level_a_render_carries_is_read_with_the_models_own_sentence():
    assert E.level_in_render("S\n\nReasoning effort is set to xhigh.<|im_end|>") == "xhigh"
    assert E.level_in_render("S<|im_end|>user quotes: Reasoning effort is set to low") == "medium"
    M.set_main(OTHER)
    assert E.level_in_render("S [effort=max]<END>…") == "xhigh"          # its word → the agent's level
    assert E.level_in_render("S [effort=low] [effort=max]<END>") == "xhigh"
    assert E.level_in_render("S<END> a user wrote [effort=low]") == "medium"
    assert E.level_in_render("Reasoning effort is set to low<END>") == "medium"     # Qwen's sentence means nothing here


def test_the_server_default_is_asked_in_the_models_words(server):
    M.set_main(OTHER)
    s = server(template=lambda kw: f"S [effort={kw['reasoning_effort']}]<END>" if kw else "S [effort=max]<END>")
    assert E.fetch_server_default(s.url) == "xhigh"
    asked = [b.get("chat_template_kwargs") for path, b in s.bodies if path == "/apply-template"]
    assert asked == [{"reasoning_effort": "low"}, None]
    flat = server(template=lambda kw: "S<END>")                           # a template without levels
    assert E.fetch_server_default(flat.url) == ""


def test_the_lookup_asks_with_the_models_word_for_the_level(server):
    p = M.from_dict({"name": "lo", "effort": {"levels": {"low": "lo", "medium": "mid", "xhigh": "hi"},
                                              "sentence_re": r"<(lo|mid|hi)>", "unmarked": "medium"}})
    M.set_main(p)
    s = server(template=lambda kw: f"S <{kw['reasoning_effort']}>" if kw else "S <hi>")
    assert E.fetch_server_default(s.url) == "xhigh"
    assert [b.get("chat_template_kwargs") for _, b in s.bodies] == [{"reasoning_effort": "lo"}, None]


def test_a_model_with_no_harder_sentence_books_no_escalation(monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    quiet = M.from_dict({"name": "quiet", "effort": {"levels": {"low": "low", "xhigh": "xhigh"}}})

    def booked(profile):
        M.set_main(profile)
        ctx = SimpleNamespace()
        GhostAgent._effort_escalate(SimpleNamespace(context=ctx), "bg-4vm-1", "a step failed")
        return getattr(ctx, "_effort_due", {})
    assert booked(M.QWEN38) == {"bg-4vm-1": "a step failed"}
    assert booked(quiet) == {} and E.escalation_sentence() == ""
    assert booked(OTHER) == {"bg-4vm-1": "a step failed"} and E.escalation_sentence() == "Think again, slowly."


def test_a_thinking_off_call_renders_the_models_unmarked_level(monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setattr(E, "known_server_default", lambda: "xhigh")
    off = {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "xhigh"}}
    assert E.thinking_off_level() == "medium" and GhostAgent._call_level(off) == "medium"       # Qwen: as before
    low = M.from_dict({"name": "lowdef", "think_off": {"thinking": "off"},
                       "effort": {"levels": {"low": "low", "medium": "medium", "xhigh": "xhigh"}, "unmarked": "low"}})
    M.set_main(low)
    assert E.thinking_off_level() == "low" and GhostAgent._call_level(off) == "low"
    mute = M.from_dict({"name": "mute2", "think_off": None,
                        "effort": {"levels": {"low": "low", "medium": "medium", "xhigh": "xhigh"}}})
    M.set_main(mute)                                    # no switch: the call renders the level it carries
    assert E.thinking_off_level() is None and GhostAgent._call_level(off) == "xhigh"
    M.set_main(M.GEMMA)
    assert E.thinking_off_level() == "medium"


def test_the_looked_up_server_level_can_be_forgotten():
    E._server_default.update(level="xhigh", at=5.0, busy=False)
    E.forget_server_default()
    assert E.known_server_default() == "" and E._server_default["at"] == 0.0


@pytest.mark.parametrize("a, b, same", [
    ("http://127.0.0.1:8088", "http://127.0.0.1:8088/v1", True),
    ("http://127.0.0.1:8088/", "HTTP://127.0.0.1:8088/v1/", True),
    ("127.0.0.1:8088", "http://127.0.0.1:8088", True),              # no scheme
    ("http://localhost:8088", "http://127.0.0.1:8088", True),
    ("http://0.0.0.0:8088", "http://127.0.0.1:8088", False),        # "any address" is not this address
    ("http://[::1]:8088", "http://127.0.0.1:8088", False),
    ("http://h:8080/gemma/v1", "http://h:8080/qwen/v1", False),     # two models behind one port
    ("http://h:8080/qwen/v1/", "http://h:8080/qwen", True),
    ("http://Example.Host:8088", "http://example.host:8088/v1", True),
    ("http://127.0.0.1:8088", "http://127.0.0.1:8089", False),
    ("http://10.0.0.5:8088", "http://127.0.0.1:8088", False),
    ("http://host", "http://host:80", True),
    ("https://host", "http://host:443", True),
    ("", "http://127.0.0.1:8088", False),
    ("", "", False),
    (None, None, False),
    ("http://[::1", "http://[::1", False),                          # unparsable: never "the same"
])
def test_two_spellings_of_one_server(a, b, same):
    assert L._same_server(a, b) is same and L._same_server(b, a) is same


def test_the_main_server_under_another_spelling_is_still_main():
    c = _client()
    c.upstream_url = "http://localhost:8088/v1"
    M.set_main(OTHER)
    assert c.node_profile({"url": "127.0.0.1:8088", "model": "Eva"}) is OTHER
    assert c.node_profile({"url": "http://127.0.0.1:8089", "model": "Eva"}) is None


def test_the_escalation_note_is_the_models_sentence():
    assert E.escalation_note("an edit to a.py was refused.") == \
        "SYSTEM: an edit to a.py was refused. " + M.QWEN38.effort.escalation
    assert E.XHIGH_SENTENCE.startswith("Reasoning effort is set to xhigh.")
    M.set_main(OTHER)
    assert E.escalation_note("x") == "SYSTEM: x. Think again, slowly."


def test_the_agents_own_payload_helpers_use_the_switch():
    from ghost_agent.core.agent import report_turn_payload, thinking_disabled
    base = {"messages": [{"role": "user", "content": "go"}], "chat_template_kwargs": {"reasoning_effort": "low"}}
    out = report_turn_payload(base)
    assert thinking_disabled(out) and M.is_no_think(out)
    assert out["chat_template_kwargs"] == {"reasoning_effort": "low", "enable_thinking": False}
    assert base["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert M.shape(out, OTHER)["chat_template_kwargs"] == {"reasoning_effort": "none"}
    on = {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}
    assert thinking_disabled(report_turn_payload(on)) and on["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_profiles_frequency_penalty_reaches_the_turn():
    from ghost_agent.core.agent import turn_sampler_fields
    assert turn_sampler_fields(CARD) == {**CARD, "frequency_penalty": 0.0}
    assert list(turn_sampler_fields(CARD)) == list(CARD) + ["frequency_penalty"]        # where it always was
    assert turn_sampler_fields({"temperature": 0.7, "frequency_penalty": 0.3}) == \
        {"temperature": 0.7, "frequency_penalty": 0.3}
    assert turn_sampler_fields({}) == {"frequency_penalty": 0.0} and turn_sampler_fields(None) == {"frequency_penalty": 0.0}


def test_the_turn_request_is_built_from_those_fields():
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    spreads = [ast.unparse(v) for d in ast.walk(tree) if isinstance(d, ast.Dict)
               for k, v in zip(d.keys, d.values) if k is None and "sampling_params" in ast.unparse(v)]
    assert spreads == ["turn_sampler_fields(sampling_params)"]


def test_health_says_which_profile_and_how_tokens_are_counted(server):
    from ghost_agent.api.routes import _model_facts
    T.TOKEN_ENCODER = None
    M.set_main(OTHER, "the server runs Other-9.gguf")
    T.calibrate(server(tokens=450).url)
    got = _model_facts()
    assert got["name"] == "other" and got["why"] == "the server runs Other-9.gguf"
    assert got["tokens"]["server_tokens"] == 450 and got["server_model"] == ""
    with B._lock:
        B._state["last"] = {"server_model": "Other-9.gguf", "tokenizer": "model file"}
    assert _model_facts()["server_model"] == "Other-9.gguf"


# --- §4VP: the cap on reasoning is logged at the send --------------------------

def _cap_lines(monkeypatch):
    from ghost_agent.utils import logging as L
    lines = []
    monkeypatch.setattr(L, "pretty_log", lambda title, content=None, **kw: lines.append((title, content, kw.get("icon"))))
    return lines


def _capped_profile():
    return M.from_dict({**INKLING_LIKE, "think_off": None, "think_off_request": {"reasoning_budget_tokens": 16},
                        "dialect": {"history": "native", "think_tags": False, "soft_no_think": ""}})


def test_a_capped_send_writes_one_log_line(monkeypatch):
    """Nothing else shows the request as it is sent: a thinking-off call of a
    model with a cap says so, with the field, its value and the profile."""
    from ghost_agent.utils.logging import Icons
    lines = _cap_lines(monkeypatch)
    prof = _capped_profile()
    monkeypatch.setattr(M, "main", lambda: prof)
    off = {"messages": [], "chat_template_kwargs": M.no_think_kwargs()}
    sent = M.shape_main(off)
    assert sent["reasoning_budget_tokens"] == 16 and sent == M.shape(off, prof)
    assert lines == [("Thinking Cap", f"thinking off for this call — sent with reasoning_budget_tokens=16 ({prof.name})",
                      Icons.EFFORT_LOW)]
    del lines[:]
    written = {"messages": [{"role": "user", "content": "hi " + M.SOFT_NO_THINK}]}
    assert M.shape_main(written)["reasoning_budget_tokens"] == 16 and len(lines) == 1    # the written switch too


def test_no_cap_line_when_nothing_was_capped(monkeypatch):
    lines = _cap_lines(monkeypatch)
    prof = _capped_profile()
    monkeypatch.setattr(M, "main", lambda: prof)
    M.shape_main({"messages": []})                                                        # thinking on
    M.shape_main({"chat_template_kwargs": {"reasoning_effort": "low"}})
    M.shape_main({"reasoning_budget_tokens": 64, "chat_template_kwargs": M.no_think_kwargs()})   # the caller's own
    M.shape_main({"messages": [], "chat_template_kwargs": M.no_think_kwargs()}, quiet=True)   # a render sends nothing
    assert lines == []
    monkeypatch.setattr(M, "main", lambda: M.DEFAULT)                                     # Qwen: no cap, same object
    off = {"messages": [], "chat_template_kwargs": M.no_think_kwargs()}
    assert M.shape_main(off) is off and lines == []


def test_a_failing_log_line_does_not_cost_the_request(monkeypatch):
    from ghost_agent.utils import logging as L
    prof = _capped_profile()
    monkeypatch.setattr(M, "main", lambda: prof)
    def boom(*a, **k):
        raise RuntimeError("log sink gone")
    monkeypatch.setattr(L, "pretty_log", boom)
    assert M.shape_main({"messages": [], "chat_template_kwargs": M.no_think_kwargs()})["reasoning_budget_tokens"] == 16


def test_think_cap_added_names_only_what_shape_put_on():
    prof = _capped_profile()
    p = {"messages": [], "chat_template_kwargs": M.no_think_kwargs()}
    assert M.think_cap_added(p, M.shape(p, prof), prof) == {"reasoning_budget_tokens": 16}
    assert M.think_cap_added(p, p, prof) == {}
    assert M.think_cap_added(p, M.shape(p, prof), None) == {}
    own = {**p, "reasoning_budget_tokens": 64}
    assert M.think_cap_added(own, M.shape(own, prof), prof) == {}


def test_the_two_renders_are_quiet():
    """A render sends nothing: it must not write the line of a send."""
    from ghost_agent.core import effort as E
    calls = [c for c in ast.walk(ast.parse(inspect.getsource(E._render).lstrip()))
             if isinstance(c, ast.Call) and ast.unparse(c.func) == "_mp.shape_main"]
    assert [ast.unparse(c) for c in calls] == ["_mp.shape_main(body, quiet=True)"]


def test_only_the_two_renders_are_quiet_in_the_whole_tree():
    """Every other `shape_main` call is a send and must write the line: a
    `quiet=` on one of them would hide the cap again."""
    import pathlib
    import ghost_agent
    root = pathlib.Path(ghost_agent.__file__).parent
    quiet, loud = [], []
    for f in sorted(root.rglob("*.py")):
        for c in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(c, ast.Call) and ast.unparse(c.func).endswith(".shape_main"):
                assert len(c.args) == 1, (f.name, ast.unparse(c))
                (quiet if c.keywords else loud).append(f"{f.relative_to(root)}: {ast.unparse(c)}")
    assert quiet == ["core/agent.py: _mp.shape_main(payload, quiet=True)", "core/effort.py: _mp.shape_main(body, quiet=True)"]
    assert sorted(x.split(":")[0] for x in loud) == ["api/routes.py", "core/agent.py", "core/llm.py", "core/llm.py"]


def test_quiet_is_by_name_only():
    with pytest.raises(TypeError):
        M.shape_main({}, True)


def test_the_profile_module_does_not_load_the_log_module():
    """The log module is loaded when a line is written, not with the profile."""
    import subprocess, sys
    code = ("import sys; from ghost_agent.core import model_profile as M; "
            "M.shape_main({'messages': [], 'chat_template_kwargs': M.no_think_kwargs()}); "
            "print('ghost_agent.utils.logging' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "False", (r.stdout, r.stderr[-300:])
