#!/usr/bin/env python3
"""§4VP (phase 6): the same functional run for every model, kept as a card.

"Is model B better than model A in this agent?" was answered by chatting with
it. This sends a fixed set of tasks to the LIVE agent as labelled probes — one
at a time, each when the agent is idle — checks each by its own rule (the
reply, or the files it left in the sandbox, run inside the sandbox container),
and keeps the result as a card named for the model the server ran:

    model_scorecard.py run             run the scenarios, write the card
    model_scorecard.py run t1 c2       only these scenarios (no card is written for a partial run)
    model_scorecard.py show            the newest card of every model, side by side
    model_scorecard.py list            the scenarios

A card records, per scenario: pass or fail, seconds, model calls, output
tokens, the reply; and for the whole run: how many tool calls arrived as TEXT
and were read by the agent's fallback parser (a model that is spoken to in
its own format makes none), warnings and error-level lines in the agent's log.

It measures whether the agent WORKS with a model — tool use, edits, code that
runs, the browser, a picture, the effort levels. It does not grade answer
quality: for that, see the census replay (blind graders).

Everything it creates lives under ``<sandbox>/scorecard_run/`` and is removed
at the end. Stdlib only. Where things are comes from the environment:
``GHOST_URL`` (default http://127.0.0.1:8000), the key file ``GHOST_KEY_FILE``,
else ``GHOST_API_KEY``, else ``.ghost_api_key`` beside the repository (the next
is tried when the agent refuses one), ``GHOST_HOME`` (its ``sandbox`` and ``system``),
``GHOST_SANDBOX_CONTAINER`` (default: the running container whose name starts
with ``ghost-agent-sandbox``).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
ROOT = "scorecard_run"
#: Markup that must never reach a reply: a think tag, a text-format call, a control token.
LEAKS = re.compile(r"</?think>|</?tool_call>|<\|[a-z_0-9]+\|>|/no_think|<function=|content_invoke_tool", re.I)
TEXT_CALL_LOG = "Extracting XML tool call"


class Env:
    """Where the agent, its sandbox and its log are."""

    def __init__(self, url=None, home=None, container=None):
        self.url = (url or os.environ.get("GHOST_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.home = Path(home or os.environ.get("GHOST_HOME") or Path.home() / "ghost_llamacpp")
        self.sandbox = self.home / "sandbox"
        self.log = self.home / "system" / "ghost-agent.log"
        self.cards = self.home / "system" / "eval" / "scorecards"
        self._container = container or os.environ.get("GHOST_SANDBOX_CONTAINER") or ""
        self._key: Optional[str] = None

    def keys(self) -> List[str]:
        """The agent's key, every place it may be, in the order tried."""
        out: List[str] = []
        for f in (os.environ.get("GHOST_KEY_FILE", "").strip(), None, str(REPO.parent / ".ghost_api_key")):
            if f is None:
                k = os.environ.get("GHOST_API_KEY", "").strip()
            else:
                try:
                    k = Path(f).expanduser().read_text().strip() if f else ""
                except OSError:
                    k = ""
            if k and k not in out:
                out.append(k)
        return out

    def key(self) -> str:
        if self._key is None:
            keys = self.keys()
            self._key = keys[0] if keys else ""
            for k in keys:                              # the first one the agent takes
                try:
                    req = urllib.request.Request(self.url + "/api/health", headers={"X-Ghost-Key": k, "X-Ghost-Origin": "probe"})
                    urllib.request.urlopen(req, timeout=10).close()
                    self._key = k
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code not in (401, 403):
                        break
                except Exception:  # noqa: BLE001
                    break
        return self._key

    def container(self) -> str:
        if not self._container:
            try:
                out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=30).stdout
            except (OSError, subprocess.SubprocessError):
                out = ""
            self._container = next((n for n in sorted(out.split()) if n.startswith("ghost-agent-sandbox")), "")
        return self._container

    # --- the agent -------------------------------------------------------------------
    def api(self, path, body=None, headers=None, timeout=1200.0):
        h = {"X-Ghost-Key": self.key(), "X-Ghost-Origin": "probe", "Content-Type": "application/json"}
        h.update(headers or {})
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(), headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def health(self) -> dict:
        try:
            return json.loads(self.api("/api/health", timeout=10))
        except Exception:  # noqa: BLE001
            return {}

    idle_wait, idle_step = 900.0, 5.0

    def idle(self, max_wait=None, step=None) -> bool:
        max_wait = self.idle_wait if max_wait is None else max_wait
        step = self.idle_step if step is None else step
        end = time.time() + max_wait
        while time.time() < end:
            try:
                turns = json.loads(self.api("/api/turns", timeout=10))
                if self.health().get("foreground_requests") == 0 and not turns.get("running") and not turns.get("queued"):
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(step)
        return False

    def chat(self, messages, stream=False, arm=None) -> dict:
        # no "model": the agent serves the one it is configured with and refuses any other name
        body = {"stream": stream, "messages": messages}
        t0 = time.time()
        raw = self.api("/v1/chat/completions", body, {"X-Ghost-Arm": arm} if arm else {})
        return parse_reply(raw, stream, round(time.time() - t0, 1))

    # --- the sandbox -----------------------------------------------------------------
    def path(self, *p) -> Path:
        return self.sandbox.joinpath(ROOT, *p)

    def read(self, *p) -> Optional[str]:
        try:
            return self.path(*p).read_text(encoding="utf-8")
        except OSError:
            return None

    def seed(self, rel: str, text: str) -> None:
        f = self.path(rel)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")

    def box(self, cmd: str, timeout=180) -> Tuple[int, str]:
        if not self.container():
            raise RuntimeError("no sandbox container to check the result in (GHOST_SANDBOX_CONTAINER)")
        r = subprocess.run(["docker", "exec", "-w", "/workspace", self.container(), "sh", "-c", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()

    def clean(self) -> None:
        """Remove the run's folder. A symlink there is removed, never followed."""
        root = self.sandbox / ROOT
        if root.is_symlink() or root.is_file():
            root.unlink()
        elif root.exists():
            shutil.rmtree(root)


def parse_reply(raw: bytes, stream: bool, seconds: float) -> dict:
    """The reply of one request: its text, usage, and for a stream the frames and reasoning seen."""
    if not stream:
        d = json.loads(raw)
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        return {"content": msg.get("content") or "", "usage": d.get("usage") or {}, "s": seconds, "error": d.get("error")}
    content, reasoning, frames, usage = "", "", 0, {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except ValueError:
            continue
        frames += 1
        delta = (d.get("choices") or [{}])[0].get("delta") or {}
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning_content") or ""
        usage = d.get("usage") or usage
    return {"content": content, "reasoning_chars": len(reasoning), "frames": frames, "usage": usage, "s": seconds, "error": None}


def png_with_digits(digits: str, scale: int = 14) -> str:
    """A PNG data URL showing ``digits`` in a 5×7 dot font, black on white (no
    imaging library: the picture test must run wherever this script does)."""
    font = {"0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
            "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
            "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
            "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
            "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100"}
    cols = len(digits) * 6 + 1
    grid = [[0] * cols for _ in range(9)]
    for i, ch in enumerate(digits):
        for r, row in enumerate(font[ch].split()):
            for c, bit in enumerate(row):
                grid[r + 1][i * 6 + 1 + c] = int(bit)
    w, h = cols * scale, 9 * scale
    raw = b"".join(b"\x00" + b"".join((b"\x00" if grid[y // scale][x // scale] else b"\xff") for x in range(w)) for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def U(text: str) -> list:
    return [{"role": "user", "content": text}]


# --- the scenarios ------------------------------------------------------------------------
# Each: (id, area, what it shows, run(env) -> (reply dict, passed)). A check is strict about
# the thing asked for and silent about everything else in the reply.
SCENARIOS: List[Tuple[str, str, str, Callable]] = []


def scenario(sid: str, area: str, what: str):
    def deco(fn):
        SCENARIOS.append((sid, area, what, fn))
        return fn
    return deco


def _prime(n: int) -> bool:
    return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))


def check_three_primes(text: str) -> bool:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return len(lines) == 3 and all(ln.isdigit() and int(ln) > 100 and _prime(int(ln)) for ln in lines)


X3_BEFORE = "[server]\nhost = localhost\nport = 8080\ndebug = false\n\n[client]\nretries = 3\n"


def check_one_line_changed(got: str) -> bool:
    """Only the port line changed (a final newline more or less is not a change)."""
    return got.rstrip("\n") == X3_BEFORE.replace("port = 8080", "port = 9090").rstrip("\n")


def has_number(text: str, n: int) -> bool:
    """``n`` stands in ``text`` as a number of its own: not inside a longer
    number (18, 80, 1.8), and not as a percentage."""
    return re.search(rf"(?<![\d.,]){n}(?:\.0+)?(?![\d%])(?![.,]\d)", text) is not None


def pytest_passed(rc: int, out: str, exactly: Optional[int] = None) -> bool:
    """A pytest run that passed, by its exit code and its last line."""
    m = re.search(r"(\d+) passed", out)
    return (rc == 0 and m is not None and "failed" not in out and "error" not in out.lower()
            and (exactly is None or int(m.group(1)) == exactly))


def used_a_tool(reply: dict) -> bool:
    """The request took more than one model call: at least one tool round ran.
    (An answer from memory, or with the tool down, is one call.)"""
    n = (reply.get("usage") or {}).get("ghost_llm_calls")
    return isinstance(n, int) and n >= 2


def check_json_city(text: str) -> bool:
    try:
        d = json.loads(text.strip())
    except ValueError:
        return False
    return (isinstance(d, dict) and isinstance(d.get("city"), str) and "tokyo" in d["city"].lower()
            and isinstance(d.get("population_millions"), (int, float)) and not isinstance(d.get("population_millions"), bool))


@scenario("t1", "talk", "a fact, no tools")
def _(e):
    r = e.chat(U("What is the capital of Australia? One word."))
    return r, "canberra" in r["content"].lower() and len(r["content"]) < 60


@scenario("t2", "talk", "an exact format")
def _(e):
    r = e.chat(U("List exactly three prime numbers greater than 100, one per line, digits only, nothing else."))
    return r, check_three_primes(r["content"])


@scenario("t3", "talk", "JSON only")
def _(e):
    r = e.chat(U('Return ONLY a JSON object with keys "city" (string) and "population_millions" (number) for Tokyo. No prose, no code fence.'))
    return r, check_json_city(r["content"])


@scenario("t4", "talk", "a conversation with history")
def _(e):
    r = e.chat([{"role": "user", "content": "I'm planning a trip. The hotel I booked is called the Marlowe Arms and my room number is 418."},
                {"role": "assistant", "content": "Got it — the Marlowe Arms, room 418. Anything you'd like help with for the trip?"},
                {"role": "user", "content": "Which room am I in, and at which hotel? One short sentence."}])
    return r, "418" in r["content"] and "marlowe" in r["content"].lower()


@scenario("t5", "talk", "a streamed reply")
def _(e):
    r = e.chat(U("Explain in two sentences why the sky is blue."), stream=True)
    return r, len(r["content"]) > 60 and r["frames"] > 5 and "scatter" in r["content"].lower()


@scenario("x1", "tools", "run a command, report its exact output")
def _(e):
    want = hashlib.sha256(b"ghost-scorecard").hexdigest()
    r = e.chat(U("Use your execute tool to compute the SHA-256 hex digest of the exact ASCII string ghost-scorecard (no newline) with Python, and give me the digest."))
    return r, want in r["content"].lower()


@scenario("x2", "tools", "write a file with exact content")
def _(e):
    r = e.chat(U(f"Create the file {ROOT}/x2/notes.txt containing exactly these three lines and nothing else:\nalpha\nbeta\ngamma"))
    got = e.read("x2", "notes.txt")
    return r, got is not None and got.strip().splitlines() == ["alpha", "beta", "gamma"]


@scenario("x3", "tools", "edit one line of an existing file")
def _(e):
    e.seed("x3/config.ini", X3_BEFORE)
    r = e.chat(U(f"In the file {ROOT}/x3/config.ini change the port to 9090 and leave every other line exactly as it is."))
    got = e.read("x3", "config.ini") or ""
    return r, check_one_line_changed(got)


@scenario("x4", "tools", "read a file and compute from it")
def _(e):
    rows = [("apples", 12, 0.5), ("pears", 7, 0.8), ("plums", 31, 0.25), ("figs", 4, 1.5), ("limes", 19, 0.3)]
    e.seed("x4/stock.csv", "item,qty,unit_price\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows))
    r = e.chat(U(f"Read {ROOT}/x4/stock.csv and tell me the total stock value (sum of qty × unit_price). Give the number."))
    return r, "31.05" in r["content"]


@scenario("x5", "tools", "several files in one request")
def _(e):
    r = e.chat(U(f"Create three files: {ROOT}/x5/a.txt containing the single word red, {ROOT}/x5/b.txt containing green, and {ROOT}/x5/c.txt containing blue."))
    return r, [(e.read("x5", n) or "").strip() for n in ("a.txt", "b.txt", "c.txt")] == ["red", "green", "blue"]


@scenario("x6", "tools", "the date from a tool, not from memory")
def _(e):
    r = e.chat(U("What is today's date? Check with a tool rather than from memory, and answer as YYYY-MM-DD."))
    return r, used_a_tool(r) and (time.strftime("%Y-%m-%d") in r["content"] or time.strftime("%Y-%m-%d", time.gmtime()) in r["content"])


@scenario("x7", "tools", "six tool steps in one request")
def _(e):
    r = e.chat(U(f"Do these one after another, each with its own tool call: (1) create {ROOT}/x7/a.txt containing 3, (2) create {ROOT}/x7/b.txt containing 4, "
                 f"(3) read both files, (4) write their product to {ROOT}/x7/product.txt, (5) run `cat {ROOT}/x7/product.txt`, (6) tell me the number."))
    return r, (e.read("x7", "product.txt") or "").strip() == "12" and "12" in r["content"]


@scenario("c1", "coding", "write a program and run it")
def _(e):
    r = e.chat(U(f"Write {ROOT}/c1/fizzbuzz.py that prints FizzBuzz for 1..30 (one item per line), run it, and tell me what line 15 says."))
    rc, out = e.box(f"python3 {ROOT}/c1/fizzbuzz.py")
    lines = out.splitlines()
    return r, rc == 0 and len(lines) == 30 and lines[14] == "FizzBuzz" and lines[2] == "Fizz" and lines[4] == "Buzz" and "fizzbuzz" in r["content"].lower()


@scenario("c2", "coding", "a function with tests that pass")
def _(e):
    r = e.chat(U(f"In {ROOT}/c2/ write intervals.py with a function merge(intervals) that merges overlapping closed intervals given as a list of [start, end] lists and returns them sorted, "
                 f"and test_intervals.py with at least four pytest tests including an empty list and touching intervals like [1,2],[2,3]. Run the tests and tell me the result."))
    rc, out = e.box(f"cd {ROOT}/c2 && python3 -m pytest -q -p no:cacheprovider > .sc_out.txt 2>&1; rc=$?; tail -3 .sc_out.txt; exit $rc")
    chk = (f"import sys; sys.path.insert(0, '{ROOT}/c2'); from intervals import merge; "
           "assert merge([]) == []; assert [list(x) for x in merge([[1,3],[2,6],[8,10],[15,18]])] == [[1,6],[8,10],[15,18]]; "
           "assert [list(x) for x in merge([[1,2],[2,3]])] == [[1,3]]; assert [list(x) for x in merge([[5,7],[1,2]])] == [[1,2],[5,7]]; print('OK')")
    rc2, out2 = e.box(f'python3 -c "{chk}"')
    return r, pytest_passed(rc, out) and rc2 == 0 and "OK" in out2


@scenario("c3", "coding", "fix a bug so the given test passes")
def _(e):
    e.seed("c3/stats.py", "def median(values):\n    values = sorted(values)\n    n = len(values)\n    mid = n // 2\n    if n % 2 == 0:\n        return values[mid]\n    return values[mid]\n\n\ndef mean(values):\n    return sum(values) / len(values)\n")
    e.seed("c3/test_stats.py", "from stats import median, mean\n\n\ndef test_odd():\n    assert median([3, 1, 2]) == 2\n\n\ndef test_even():\n    assert median([4, 1, 3, 2]) == 2.5\n\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n")
    before = e.read("c3", "test_stats.py")
    r = e.chat(U(f"The tests in {ROOT}/c3/ fail. Find the bug in stats.py, fix it without changing the test file, and run the tests to confirm."))
    rc, out = e.box(f"cd {ROOT}/c3 && python3 -m pytest -q -p no:cacheprovider test_stats.py > .sc_out.txt 2>&1; rc=$?; tail -3 .sc_out.txt; exit $rc")
    return r, pytest_passed(rc, out, exactly=3) and e.read("c3", "test_stats.py") == before


@scenario("c4", "coding", "a longer file that runs")
def _(e):
    r = e.chat(U(f"Write {ROOT}/c4/inventory.py: a small command-line inventory manager using only the standard library, storing items in a JSON file next to the script. "
                 f"Commands: add NAME QTY, remove NAME, list, total (total prints the sum of all quantities). Include argparse, input validation, and a main() guard. "
                 f"Then run: add widget 5, add gadget 3, total — and tell me what total prints."))
    src = e.read("c4", "inventory.py") or ""
    e.box(f"cd {ROOT}/c4 && rm -f *.json")
    rc, out = e.box(f"cd {ROOT}/c4 && python3 inventory.py add widget 5 && python3 inventory.py add gadget 3 && python3 inventory.py total")
    return r, (rc == 0 and bool(out) and has_number(out.splitlines()[-1], 8) and "argparse" in src
               and "__main__" in src and has_number(r["content"], 8))


@scenario("c5", "coding", "a web page with a working script")
def _(e):
    r = e.chat(U(f'Create {ROOT}/c5/counter.html: a single self-contained page with a heading, a button with id="inc", and a span with id="count" that starts at 0 and goes up by one on each click. No external files.'))
    html = e.read("c5", "counter.html") or ""
    return r, 'id="inc"' in html and 'id="count"' in html and "<script" in html.lower() and ("addEventListener" in html or "onclick" in html)


@scenario("w1", "outside", "open a web page and read it")
def _(e):
    r = e.chat(U("Open https://example.com with your browser tool and tell me the page's title."))
    return r, used_a_tool(r) and "example domain" in r["content"].lower()


@scenario("w2", "outside", "a web search with a source")
def _(e):
    r = e.chat(U("Search the web: in which year was the first stable release of the Rust programming language (1.0), and name the source you used."))
    return r, used_a_tool(r) and "2015" in r["content"]


@scenario("w3", "outside", "one browser call carrying a list of actions")
def _(e):
    r = e.chat(U("With your browser tool, open https://example.com, then in ONE interact call run two actions: "
                 "evaluate document.title, and evaluate document.title.length * 3. Tell me both values."))
    return r, used_a_tool(r) and "example domain" in r["content"].lower() and has_number(r["content"], 42)


@scenario("v1", "outside", "read a number in a picture")
def _(e):
    r = e.chat([{"role": "user", "content": [{"type": "text", "text": "What number is written in this picture? Reply with the number only."},
                                             {"type": "image_url", "image_url": {"url": png_with_digits("7391")}}]}])
    return r, "7391" in r["content"]


PUZZLE = ("Three friends split a restaurant bill. Ana pays 40% of it, Ben pays 42 euros, and Cy pays the remaining quarter of the bill. "
          "How much was the whole bill? Give the number of euros and one line of working.")


@scenario("e1", "effort", "a puzzle, asked at low effort")
def _(e):
    r = e.chat(U(PUZZLE), arm="effort_low")
    return r, has_number(r["content"], 120)


@scenario("e2", "effort", "the same puzzle, asked at extra-high effort")
def _(e):
    r = e.chat(U(PUZZLE), arm="effort_xhigh")
    return r, has_number(r["content"], 120)


# --- running and keeping ---------------------------------------------------------------------

NEVER_IDLE = "the agent never became idle"


def run_one(env: Env, sid: str, area: str, what: str, fn: Callable) -> dict:
    if not env.idle():
        return {"id": sid, "area": area, "what": what, "ok": False, "note": NEVER_IDLE}
    try:
        reply, passed = fn(env)
    except Exception as exc:  # noqa: BLE001 — a scenario that raises has failed; say how
        return {"id": sid, "area": area, "what": what, "ok": False, "note": f"{type(exc).__name__}: {str(exc)[:300]}"}
    leaks = LEAKS.findall(reply.get("content") or "")
    usage = reply.get("usage") or {}
    return {"id": sid, "area": area, "what": what, "ok": bool(passed) and not leaks and not reply.get("error"),
            "check": bool(passed), "leaks": leaks[:5], "s": reply.get("s"),
            "calls": usage.get("ghost_llm_calls"), "out_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "reply": reply.get("content"), **({"error": reply["error"]} if reply.get("error") else {})}


def log_counts(log_text: str) -> dict:
    """What the agent's own log says about the probe turns in ``log_text``."""
    lines = [ln for ln in log_text.splitlines() if "[probe-" in ln]      # every labelled probe in the window, this run's or not
    return {"model_calls": sum("llm request" in ln for ln in lines),
            "tool_calls": sum("tool call —" in ln for ln in lines),
            "text_format_tool_calls": sum(TEXT_CALL_LOG in ln for ln in lines),
            "warnings": sum(" - WARNING - " in ln for ln in lines),
            "errors": sum(" - ERROR - " in ln for ln in lines)}


def summarise(results: List[dict], model: dict, counts: dict, started: str) -> dict:
    areas: Dict[str, List[int]] = {}
    for r in results:
        a = areas.setdefault(r["area"], [0, 0])
        a[0] += bool(r["ok"])
        a[1] += 1
    return {"model": model.get("server_model") or "unknown", "profile": model.get("name") or "?",
            "tokens_exact": bool((model.get("tokens") or {}).get("exact")), "started": started,
            "passed": sum(bool(r["ok"]) for r in results), "of": len(results),
            "seconds": round(sum(r.get("s") or 0 for r in results)),
            "areas": {k: f"{v[0]}/{v[1]}" for k, v in areas.items()}, **counts, "results": results}


def card_name(model: str, started: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(model)[0]).strip("_") or "unknown"
    return f"{stem}--{started}.json"


def read_log_since(log: Path, start: int) -> Optional[str]:
    """The log written since byte ``start``; None when that cannot be known
    (no log, or it was rotated meanwhile: a count from it would be a wrong zero)."""
    try:
        if log.stat().st_size < start:
            return None
        with open(log, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return None


def newest_cards(cards_dir: Path) -> List[dict]:
    """The newest card of each model, by model name."""
    best: Dict[str, dict] = {}
    for f in sorted(cards_dir.glob("*.json")) if cards_dir.is_dir() else []:
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(c, dict) and isinstance(c.get("model"), str) and c["model"] and isinstance(c.get("results"), list)
                and isinstance(c.get("started"), str) and c["started"] >= best.get(c["model"], {}).get("started", "")):
            best[c["model"]] = c
    return [best[k] for k in sorted(best)]


def table(cards: List[dict]) -> str:
    """The cards side by side: totals first, then one row per scenario."""
    if not cards:
        return "no cards yet — run: model_scorecard.py run"
    names = [os.path.splitext(c["model"])[0][:26] for c in cards]
    rows = [["", *names],
            ["profile", *[c.get("profile", "?") for c in cards]],
            ["run", *[c.get("started", "") for c in cards]],
            ["passed", *[f"{c.get('passed', '?')}/{c.get('of', '?')}" for c in cards]],
            ["seconds", *[str(c.get("seconds", "")) for c in cards]],
            ["tool calls as text", *[f"{c.get('text_format_tool_calls', '?')}/{c.get('tool_calls', '?')}" for c in cards]],
            ["warnings / errors", *[f"{c.get('warnings', '?')} / {c.get('errors', '?')}" for c in cards]],
            ["exact token counts", *["yes" if c.get("tokens_exact") else "no" for c in cards]]]
    ids = []
    rows_of = [[x for x in c.get("results", []) if isinstance(x, dict) and x.get("id")] for c in cards]
    for rs in rows_of:
        ids += [r["id"] for r in rs if r["id"] not in ids]
    for sid in ids:
        cells = []
        for rs in rows_of:
            r = next((x for x in rs if x["id"] == sid), None)
            cells.append("-" if r is None else f"{'pass' if r.get('ok') else 'FAIL'} {r.get('s', '?')}s")
        what = next((x.get("what", "") for rs in rows_of for x in rs if x["id"] == sid), "")
        rows.append([f"{sid} {what}"[:44], *cells])
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(cell).ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A functional scorecard of the model behind the agent.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("ids", nargs="*")
    sub.add_parser("show")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    env = Env()
    if a.cmd == "list":
        for sid, area, what, _ in SCENARIOS:
            print(f"{sid:3s} {area:8s} {what}")
        return 0
    if a.cmd == "show":
        print(table(newest_cards(env.cards)))
        return 0
    known = {s[0] for s in SCENARIOS}
    if set(a.ids) - known:
        print(f"unknown scenario(s): {', '.join(sorted(set(a.ids) - known))}", file=sys.stderr)
        return 2
    model = env.health().get("model_profile") or {}
    if not model:
        print("the agent does not answer (GHOST_URL; GHOST_KEY_FILE or GHOST_API_KEY)", file=sys.stderr)
        return 2
    if not env.sandbox.is_dir():
        print(f"no sandbox at {env.sandbox} — set GHOST_HOME to the agent's home", file=sys.stderr)
        return 2
    todo = [sc for sc in SCENARIOS if not a.ids or sc[0] in a.ids]
    if any(sc[1] == "coding" for sc in todo) and not env.container():
        print("no running sandbox container to check the coding results in (GHOST_SANDBOX_CONTAINER)", file=sys.stderr)
        return 2
    started = time.strftime("%Y%m%d-%H%M%S")
    log_at = env.log.stat().st_size if env.log.is_file() else 0
    print(f"model {model.get('server_model') or 'unknown'} · profile {model.get('name')} · {len(todo)} scenario(s)")
    env.clean()                                     # nothing a killed run left may pass for this model's work
    env.path().mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for sid, area, what, fn in todo:
            r = run_one(env, sid, area, what, fn)
            results.append(r)
            print(f"{sid:3s} {'PASS' if r['ok'] else 'FAIL'} {str(r.get('s', '-')):>6}s  calls {r.get('calls', '-')}  {what}"
                  + (f"  | {r.get('note') or r.get('leaks') or ''}" if not r["ok"] else ""), flush=True)
            if r.get("note") == NEVER_IDLE:
                print("the agent is not becoming idle: stopping here", file=sys.stderr)
                break
    finally:
        try:
            env.clean()
        except OSError as exc:                      # the results are not lost to a folder that would not go
            print(f"could not remove {env.sandbox / ROOT}: {exc}", file=sys.stderr)
    after: dict = {}
    for _ in range(6):                              # one slow answer must not cost a 20-minute run its card
        after = env.health().get("model_profile") or {}
        if after:
            break
        time.sleep(env.idle_step)
    log_text = read_log_since(env.log, log_at)
    counts = log_counts(log_text) if log_text is not None else {}
    card = summarise(results, model, counts, started)
    stopped = any(r.get("note") == NEVER_IDLE for r in results)
    complete = (not a.ids and not stopped and len(results) == len(SCENARIOS)
                and after.get("server_model") == model.get("server_model"))
    print(f"\n{card['passed']}/{card['of']} passed in {card['seconds']} s · tool calls as text "
          f"{card.get('text_format_tool_calls', '?')}/{card.get('tool_calls', '?')} · warnings {card.get('warnings', '?')} · errors {card.get('errors', '?')}")
    if not after:
        print("the agent stopped answering at the end of the run: no card written", file=sys.stderr)
    elif after.get("server_model") != model.get("server_model"):
        print(f"the model changed during the run ({model.get('server_model')} → {after.get('server_model')}): no card written", file=sys.stderr)
    elif stopped:
        print("(the run stopped before its end: no card written)")
    elif not complete:
        print("(a partial run: no card written)")
    else:
        env.cards.mkdir(parents=True, exist_ok=True)
        out = env.cards / card_name(card["model"], started)
        n = 1
        while out.exists():                             # two runs in one second: both are kept
            n += 1
            out = env.cards / card_name(card["model"], f"{started}-{n}")
        out.write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"card: {out}")
    same_model = after.get("server_model") == model.get("server_model")
    return 0 if card["passed"] == card["of"] and same_model and not stopped else 1


if __name__ == "__main__":
    sys.exit(main())
