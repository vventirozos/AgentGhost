#!/usr/bin/env python3
"""Eva's mind: the 27B's side of Eva's World (runs on the Mac, under launchd).

Loop: ask the hub on nova whether Eva is due to look at her world; if she is
and the agent is idle, fetch the diary, ask the 27B (the raw llama-server, NOT
the agent: nothing here touches the agent's memory, ledgers or verifier), and
send her decision back. The hub applies it to the world and records it.

The one shared resource is the 27B's single llama slot. Rules that protect
the operator's own requests:
  * never START a call while the agent has a request in flight
    (/api/health foreground_requests > 0), the slot is processing, or the
    slot is still kept for the operator's last conversation (§4RP, 10 min;
    /api/health slot_keep_left_s), which a look would push out;
  * stream the call and ABORT it the moment a request arrives: the socket is
    shut down, llama cancels the task, and Eva retries on a later poll;
  * a skipped or aborted look loses nothing: the hub keeps the events and the
    next diary covers them.

Stdlib only, Python 3.9+ (the system python), so it does not depend on the
agent's venv.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
HUB = os.environ.get("EVA_WORLD_HUB", "http://100.83.184.117:8090")
HUB_KEY_FILE = Path(os.environ.get("EVA_WORLD_HUB_KEY_FILE", "~/Data/AI/.world_hub_key")).expanduser()
LLAMA = os.environ.get("EVA_WORLD_LLAMA", "http://127.0.0.1:8088")
AGENT = os.environ.get("EVA_WORLD_AGENT", "http://127.0.0.1:8000")
AGENT_KEY_FILE = Path(os.environ.get("EVA_WORLD_AGENT_KEY_FILE", "~/Data/AI/.ghost_api_key")).expanduser()
POLL_S = float(os.environ.get("EVA_WORLD_POLL_S", "20"))
BUSY_RETRY_S = 60.0
ERROR_RETRY_S = 120.0
#: reasoning + reply; at ~22 tok/s this caps one look at ~4.5 min of decode
MAX_TOKENS = int(os.environ.get("EVA_WORLD_MAX_TOKENS", "6000"))
CALL_TIMEOUT_S = 900.0
WATCH_S = 2.0

DECISION_KEYS = ("goal", "stop", "tell_body", "say", "journal", "wake_in_minutes")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# ── pure helpers (unit-tested) ────────────────────────────────────────────

def load_prompt_parts(here: Path = HERE) -> Tuple[str, str]:
    persona = (here / "persona.txt").read_text(encoding="utf-8").strip()
    world = (here / "world_prompt.txt").read_text(encoding="utf-8").strip()
    return persona, world


def system_prompt(persona: str, world: str) -> str:
    return f"{persona}\n\n{world}"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def user_message(ctx: Dict[str, Any]) -> str:
    journal = (ctx.get("journal") or "").strip()
    parts = ["YOUR JOURNAL:", journal if journal else "(empty: you have not written anything yet)", "", ctx.get("diary") or ""]
    return "\n".join(parts)


def build_messages(system: str, ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message(ctx)}]


_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_reasoning(reasoning_field: str, content: str) -> Tuple[str, str]:
    """llama returns the thinking in `reasoning_content` when it parses it,
    otherwise inline as <think>…</think> (or an unclosed leading block)."""
    reasoning = (reasoning_field or "").strip()
    reply = content or ""
    inline = _THINK.findall(reply)
    if inline:
        reasoning = (reasoning + "\n" + "\n".join(s.strip() for s in inline)).strip()
        reply = _THINK.sub("", reply)
    elif "</think>" in reply:
        head, _, reply = reply.partition("</think>")
        reasoning = (reasoning + "\n" + head.replace("<think>", "").strip()).strip()
    return reasoning, reply.strip()


def _json_objects(text: str) -> List[Any]:
    """Every top-level JSON object in the text, in order. Decoded from each
    "{" with the real JSON decoder, so a stray brace in the prose (a smiley,
    a quoted "{") cannot swallow the decision after it."""
    dec = json.JSONDecoder()
    out, i = [], text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = text.find("{", end)
    return out


def _as_decision(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    # the agent's tool-call habit: {"name": ..., "arguments": {...decision...}}
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    if not any(k in obj for k in DECISION_KEYS) and isinstance(args, dict):
        obj = args
    if any(k in obj for k in DECISION_KEYS):
        return {k: obj[k] for k in DECISION_KEYS if k in obj}
    return None


def parse_decision(reply: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The LAST JSON object in the reply that has a known key."""
    if not reply or not reply.strip():
        return None, "empty reply"
    for obj in reversed(_json_objects(reply)):
        d = _as_decision(obj)
        if d is not None:
            return d, None
    return None, "no decision JSON found in the reply"


def agent_busy(health: Optional[Dict[str, Any]], slots: Optional[List[Dict[str, Any]]]) -> Tuple[bool, str]:
    """Busy unless BOTH signals say idle. Unknown counts as busy."""
    if health is None:
        return True, "agent health unreachable"
    fg = health.get("foreground_requests")
    if not isinstance(fg, int):
        return True, "agent health has no foreground_requests"
    if fg > 0:
        return True, f"agent has {fg} request(s) in flight"
    keep = health.get("slot_keep_left_s")
    if not isinstance(keep, (int, float)):
        return True, "agent health has no slot_keep_left_s"
    if keep > 0:
        return True, f"the operator's conversation keeps the slot for {keep:.0f} s more"
    if slots is None:
        return True, "llama slots unreachable"
    if any(s.get("is_processing") for s in slots):
        return True, "llama slot is processing"
    return False, "idle"


# ── I/O ───────────────────────────────────────────────────────────────────

class Interrupted(Exception):
    pass


def _read_key(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def http_json(method: str, url: str, body: Any = None, headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


class Hub:
    def __init__(self, base: str = HUB, key_file: Path = HUB_KEY_FILE):
        self.base = base.rstrip("/")
        self.key_file = key_file

    def _h(self) -> Dict[str, str]:
        return {"x-world-key": _read_key(self.key_file)}

    def get(self, path: str, timeout: float = 15.0) -> Any:
        return http_json("GET", self.base + path, headers=self._h(), timeout=timeout)

    def post(self, path: str, body: Any, timeout: float = 30.0) -> Any:
        return http_json("POST", self.base + path, body, headers=self._h(), timeout=timeout)

    def status(self, status: str, detail: str = "") -> None:
        try:
            self.post("/api/mind/status", {"status": status, "detail": detail}, timeout=10)
        except Exception:
            pass


def read_agent_health() -> Optional[Dict[str, Any]]:
    try:
        key = _read_key(AGENT_KEY_FILE)
        return http_json("GET", AGENT + "/api/health", headers={"X-Ghost-Key": key, "X-Ghost-Origin": "probe"}, timeout=5)
    except Exception:
        return None


def read_slots() -> Optional[List[Dict[str, Any]]]:
    try:
        return http_json("GET", LLAMA + "/slots", timeout=5)
    except Exception:
        return None


def stream_chat(messages: List[Dict[str, str]], should_abort: Callable[[], Optional[str]]) -> Dict[str, Any]:
    """Streamed chat completion that can be cut off from outside.

    A reader thread consumes the SSE stream; this thread checks
    should_abort() every WATCH_S and, on a reason, shuts the socket down so
    llama sees the client go away and cancels the task."""
    u = urlparse(LLAMA)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=CALL_TIMEOUT_S)
    body = json.dumps({"messages": messages, "stream": True, "max_tokens": MAX_TOKENS,
                       "timings_per_token": False, "cache_prompt": True}).encode("utf-8")
    conn.request("POST", "/v1/chat/completions", body=body, headers={"content-type": "application/json"})
    # Keep our own handle: on a will-close response http.client moves the
    # socket into the response and sets conn.sock to None, and a shutdown of
    # None would leave the reader blocked until the model finished.
    sock = conn.sock
    out: Dict[str, Any] = {"reasoning": [], "content": [], "finish": None, "timings": None, "error": None}

    def reader() -> None:
        try:
            resp = conn.getresponse()
            if resp.status != 200:
                out["error"] = f"llama HTTP {resp.status}: {resp.read(300)!r}"
                return
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if ev.get("timings"):
                    out["timings"] = ev["timings"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        out["reasoning"].append(d["reasoning_content"])
                    if d.get("content"):
                        out["content"].append(d["content"])
                    if ch.get("finish_reason"):
                        out["finish"] = ch["finish_reason"]
        except Exception as e:  # includes the shutdown we cause on abort
            if out["error"] is None:
                out["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    started = time.time()
    abort_reason = None
    while t.is_alive():
        t.join(WATCH_S)
        if not t.is_alive():
            break
        abort_reason = should_abort()
        if abort_reason or time.time() - started > CALL_TIMEOUT_S:
            abort_reason = abort_reason or "call timed out"
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            t.join(10)
            break
    conn.close()
    if abort_reason:
        raise Interrupted(abort_reason)
    if out["error"]:
        raise RuntimeError(out["error"])
    out["elapsed_s"] = round(time.time() - started, 1)
    return out


# ── the loop ──────────────────────────────────────────────────────────────

def agent_outside_call(state: str, detail: str = "") -> Tuple[bool, str]:
    """Tell the agent a look starts/ends on its slot. "start" is the agent's
    own last check (refused while a request is live or a conversation keeps
    the slot); "end" lets it put its head back before the owner's next call;
    "result" only puts her decision in the agent's log. ``detail`` is shown
    in that log line. Anything but a clear yes counts as no."""
    try:
        key = _read_key(AGENT_KEY_FILE)
        http_json("POST", AGENT + "/api/slot/outside-call",
                  {"state": state, "label": "eva_world", "detail": detail[:300]},
                  headers={"X-Ghost-Key": key, "X-Ghost-Origin": "probe"}, timeout=5)
        return True, ""
    except urllib.error.HTTPError as e:
        try:
            why = json.loads(e.read() or b"{}").get("reason") or f"HTTP {e.code}"
        except ValueError:
            why = f"HTTP {e.code}"
        return False, why
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def owner_needs_model() -> Optional[str]:
    h = read_agent_health()
    if h is None:
        return "agent health unreachable"
    fg = h.get("foreground_requests")
    if isinstance(fg, int) and fg > 0:
        return "the operator sent a request"
    return None


def look_once(hub: Hub, system: str, phash: str) -> float:
    """One wake-up. Returns seconds to sleep before the next poll."""
    poll = hub.get("/api/mind/poll")
    if not poll.get("due"):
        return POLL_S
    busy, why = agent_busy(read_agent_health(), read_slots())
    if busy:
        log(f"due but skipped: {why}")
        hub.status("skipped: agent busy", why)
        return BUSY_RETRY_S
    ctx = hub.get("/api/mind/context", timeout=30)
    # fetching the context took seconds: ask again right before the call
    busy, why = agent_busy(read_agent_health(), read_slots())
    if busy:
        log(f"due but skipped: {why}")
        hub.status("skipped: agent busy", why)
        return BUSY_RETRY_S
    ok, why = agent_outside_call("start", "woke because: " + ("; ".join(ctx.get("reasons") or []) or "her chosen time"))
    if not ok:
        log(f"due but skipped: the agent said no ({why})")
        hub.status("skipped: agent busy", why)
        return BUSY_RETRY_S
    hub.status("thinking", ", ".join(ctx.get("reasons") or []))
    log(f"looking (reasons: {ctx.get('reasons')})")
    outcome = "stopped by an error"
    try:
        res = stream_chat(build_messages(system, ctx), owner_needs_model)
        outcome = "finished" if res.get("finish") in (None, "stop") else f"finish: {res.get('finish')}"
    except Interrupted as e:
        outcome = f"cut off: {e}"
        log(f"interrupted: {e}")
        hub.status("interrupted", str(e))
        return BUSY_RETRY_S
    finally:
        agent_outside_call("end", outcome)
    reasoning, reply = split_reasoning("".join(res["reasoning"]), "".join(res["content"]))
    decision, err = parse_decision(reply)
    if res.get("finish") == "length" and err:
        err = f"{err} (the reply hit the {MAX_TOKENS}-token limit)"
    t = res.get("timings") or {}
    timings = {"elapsed_s": res.get("elapsed_s"), "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
               "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms"), "finish": res.get("finish")}
    r = hub.post("/api/mind/decision", {"look_ts": ctx["look_ts"], "diary": ctx.get("diary"), "reasoning": reasoning,
                                       "reply": reply, "decision": decision, "parse_error": err,
                                       "prompt_hash": phash, "timings": timings}, timeout=60)
    log(f"decided in {res.get('elapsed_s')} s: {r.get('applied')}{' (parse error: ' + err + ')' if err else ''}")
    agent_outside_call("result", ("; ".join(r.get("applied") or []) or "nothing")
                       + (f" (no decision: {err})" if err else ""))
    hub.status("idle", "next look at " + time.strftime("%H:%M", time.localtime((r.get("wake_at") or 0) / 1000)))
    return POLL_S


def main() -> int:
    hub = Hub()
    persona, world = load_prompt_parts()
    system = system_prompt(persona, world)
    phash = prompt_hash(system)
    sent_prompt = False
    log(f"Eva's mind started (hub {HUB}, prompt {phash})")
    while True:
        try:
            if not sent_prompt:
                hub.post("/api/mind/prompt", {"hash": phash, "text": system})
                sent_prompt = True
            delay = look_once(hub, system, phash)
        except urllib.error.HTTPError as e:
            log(f"hub answered {e.code}: {e.read(300)!r}")
            delay = BUSY_RETRY_S
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            log(f"unreachable: {e}")
            delay = BUSY_RETRY_S
        except Exception as e:
            log(f"error: {type(e).__name__}: {e}")
            hub.status("error", f"{type(e).__name__}: {e}")
            delay = ERROR_RETRY_S
        time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
