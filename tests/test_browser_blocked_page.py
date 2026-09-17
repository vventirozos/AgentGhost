"""The blocked page that read as OK (§4HH, 2026-09-16).

Request 7ee072f2: `ft.com` answered a 403 "Security Verification" page and
`kelacyber.com` a 403 "Just a moment…" challenge; the browser tool labelled
both STATUS: OK, the model's reasoning read "FT (15 Sep 2026): … tied to
pec.interno.it" as if it had read the FT, and the report listed FT and KELA
among its sources. `op_extract_text` threw the document's response status
away (`op_navigate` kept it and the formatter printed HTTP_STATUS under an
OK header). Corpus: 75 of 584 browser OK results were challenge or 4xx
pages. Each pin names the world it fails in.
"""
import ast
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.distill.outcome_heuristics import looks_like_tool_error
from ghost_agent.tools import browser_runner as BR
from ghost_agent.tools.browser import tool_browser
from ghost_agent.tools.browser_routes import BLOCKED_PAGE_HINT, blocked_page_reason
from ghost_agent.tools.outcome import ToolOutcome
from ghost_agent.tools.tool_failure import FailureClass, classify_tool_failure

_FT = {"status": 403, "url": "https://www.ft.com/content/97f3?__cf_chl_rt_tk=abc",
       "title": "Security Verification", "length": 554, "truncated": False,
       "text": "Financial Times\nSecurity Verification\nFor help please visit help.ft.com."}
_KELA = {"status": 403, "url": "https://www.kelacyber.com/blog/revolut-data-breach/?__cf_chl_rt_tk=x",
         "title": "Just a moment...", "length": 0, "truncated": False, "text": ""}
# §4HV (2026-09-16, req d121dcc6): Hetzner's "Heray" gate — HTTP 200, a
# rewritten URL, 211 chars of "Checking that you are not a robot".
_HERAY = {"status": 200, "url": "https://www.hetzner.com/_ray/pow", "title": "Security Check",
          "length": 211, "truncated": False,
          "text": "Security Check\nChecking that you are not a robot\n\nThis process is performed "
                  "automatically. You will be redirected shortly.\n\nVerifying...\nHeray is a "
                  "security product powered by Hetzner\nLegal notice - Data privacy"}
_OK = {"status": 200, "url": "https://shattered.io/revolut-italy-email-breach-147gb-2026/",
       "title": "Revolut Hackers Claim 147GB Italy Email Breach [2026]", "length": 20410,
       "truncated": True, "text": "Financial Times reported on September 15, 2026, that hackers…"}


# ── the classifier ────────────────────────────────────────────────────────

@pytest.mark.parametrize("parsed,expect", [
    (_FT, "HTTP 403 — bot challenge"),
    (_KELA, "HTTP 403 — bot challenge"),
    ({"status": 403, "url": "https://x/", "title": "Forbidden", "length": 90}, "HTTP 403 — access refused"),
    ({"status": 404, "url": "https://x/", "title": "Not found", "length": 90}, "HTTP 404"),
    ({"status": 503, "url": "https://x/", "title": "Service Unavailable", "length": 20}, "HTTP 503 — access refused"),
    # legacy runner result (no status): a challenge title with an empty body
    ({"url": "https://x/", "title": "Just a moment...", "length": 0}, "bot challenge / interstitial (Just a moment...)"),
    ({"url": "https://x/?__cf_chl_rt_tk=1", "title": "", "length": 12}, "bot challenge / interstitial (challenge url)"),
    # a title with leading whitespace (Playwright returns it as-is)
    ({"url": "https://x/", "title": "  Just a moment...", "length": 0}, "bot challenge / interstitial (Just a moment...)"),
    # §4HV: the HTTP-200 gate — by title, and by its rewritten URL alone
    (_HERAY, "bot challenge / interstitial (Security Check)"),
    ({"status": 200, "url": "https://www.hetzner.com/_ray/pow", "title": "", "length": 211},
     "bot challenge / interstitial (challenge url)"),
    ({"status": 200, "url": "https://x/", "title": "Checking that you are not a robot", "length": 140},
     "bot challenge / interstitial (Checking that you are not a robot)"),
])
def test_the_live_pages_and_their_kin_are_blocked(parsed, expect):
    """FAILS IF: a 4xx document, or a challenge page with no article behind
    it, reads as a page the agent read — the live world."""
    assert blocked_page_reason(parsed) == expect


@pytest.mark.parametrize("parsed", [
    _OK,
    # a real article that happens to be titled like a wall: the body is there
    {"status": 200, "url": "https://x/", "title": "Access Denied — a history of the phrase", "length": 9000},
    {"url": "https://x/", "title": "Attention Required! The 2026 budget", "length": 4200},
    # status captured and fine
    {"status": 200, "url": "https://x/", "title": "Just a moment in Athens", "length": 5000},
    # a short legacy page whose title MENTIONS a challenge phrase mid-way is
    # a page about challenges, not one — the title test is anchored
    {"url": "https://x/", "title": "Why Cloudflare shows 'Just a moment' pages", "length": 300},
    # §4HV: "security check" is word-bounded — a SHORT checklist article is
    # still an article (the length bound alone would not save this one)
    {"status": 200, "url": "https://x/", "title": "Security Checklist for Kubernetes", "length": 1500},
    {"status": 200, "url": "https://x/", "title": "Security Check-in Procedures at Airports", "length": 4200},
])
def test_pages_that_were_read_are_not_blocked(parsed):
    """FAILS IF: the title alone condemns a page — the length bound and the
    status-first rule are what keep a real article readable."""
    assert blocked_page_reason(parsed) == ""


# ── the formatter and the outcome ─────────────────────────────────────────

def _stub(payload):
    stub = MagicMock(); stub.last_command = None
    def _execute(cmd, timeout=300, **kwargs):
        stub.last_command = cmd
        return f"[BROWSER_OK] {json.dumps(payload)}\n", 0
    stub.execute = _execute
    return stub


@pytest.mark.asyncio
async def test_a_blocked_extract_is_a_declared_failure_with_the_hint(tmp_path):
    """FAILS IF: the header still says OK, the hint is missing, or the
    result is a plain string — the strike ledger, the no-progress window and
    the corpus label read the OUTCOME, and a string coerces to ok."""
    res = await tool_browser(operation="extract_text", url=_FT["url"],
                             sandbox_dir=tmp_path, sandbox_manager=_stub(_FT))
    assert str(res).startswith("--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 403 — bot challenge)")
    assert "STATUS: OK" not in str(res)
    assert BLOCKED_PAGE_HINT in str(res)
    assert "HTTP_STATUS: 403" in str(res)
    assert "Security Verification" in str(res)        # what it got stays visible
    assert isinstance(res, ToolOutcome) and res.is_failure and not res.world_changed
    assert res.reason_code == "browser_blocked"


@pytest.mark.asyncio
async def test_a_blocked_navigate_and_screenshot_are_declared_too(tmp_path):
    """FAILS IF: only extract_text is classified — navigate always carried
    the status and still said OK; screenshot never carried it."""
    nav = await tool_browser(operation="navigate", url=_KELA["url"],
                             sandbox_dir=tmp_path, sandbox_manager=_stub(_KELA))
    assert isinstance(nav, ToolOutcome) and nav.is_failure
    assert "STATUS: BLOCKED" in str(nav)
    shot = dict(_KELA); shot["path"] = "/workspace/shot.png"; shot["dom_text_chars"] = 0
    scr = await tool_browser(operation="screenshot", url=_KELA["url"],
                             sandbox_dir=tmp_path, sandbox_manager=_stub(shot))
    assert isinstance(scr, ToolOutcome) and scr.is_failure
    assert "STATUS: BLOCKED" in str(scr)


@pytest.mark.asyncio
async def test_a_read_page_is_still_an_ok_string(tmp_path):
    """FAILS IF: every fetch becomes a declared outcome or a failure."""
    res = await tool_browser(operation="extract_text", url=_OK["url"],
                             sandbox_dir=tmp_path, sandbox_manager=_stub(_OK))
    assert str(res).startswith("--- BROWSER RESULT ---\nSTATUS: OK")
    assert not isinstance(res, ToolOutcome)
    assert "HTTP_STATUS: 200" in str(res)


@pytest.mark.asyncio
async def test_the_sniffers_agree_with_the_header(tmp_path):
    """FAILS IF: the corpus label (`looks_like_tool_error`) or the retry
    class disagree with the header — a blocked page must be a failure that
    is NOT retried."""
    res = await tool_browser(operation="extract_text", url=_FT["url"],
                             sandbox_dir=tmp_path, sandbox_manager=_stub(_FT))
    assert looks_like_tool_error(str(res)) is True
    assert classify_tool_failure(str(res))[0] is FailureClass.FATAL


# ── the runner ────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status): self.status = status


class _Page:
    url = "https://www.ft.com/content/97f3"
    def __init__(self, status): self._status = status
    async def goto(self, url, wait_until="load"): return _Resp(self._status)
    async def title(self): return "Security Verification"
    async def evaluate(self, js): return "Financial Times\nSecurity Verification"
    async def query_selector(self, sel): return None
    async def screenshot(self, path=None, full_page=False): open(path, "wb").write(b"png")
    async def wait_for_timeout(self, ms): pass
    viewport_size = {"width": 1280, "height": 720}


@pytest.mark.asyncio
async def test_extract_text_and_screenshot_carry_the_document_status(tmp_path, monkeypatch):
    """FAILS IF: the runner throws the `page.goto` response away again — the
    formatter can only classify what it is shipped."""
    async def _with_context(profile_dir, proxy, timeout_ms, run):
        return await run(_Page(403))
    monkeypatch.setattr(BR, "_with_context", _with_context)
    monkeypatch.setattr(BR, "_write_last_url", lambda *a, **k: None)
    async def _probe(page): return {}
    monkeypatch.setattr(BR, "_probe_pre_interaction", _probe)
    async def _excerpt(page, n): return ("", False, 0)
    monkeypatch.setattr(BR, "_body_excerpt", _excerpt)
    op = {"url": _Page.url, "profile_dir": str(tmp_path), "timeout_ms": 1000, "proxy": None}
    ext = await BR.op_extract_text(dict(op))
    assert ext["status"] == 403
    shot = await BR.op_screenshot(dict(op, out_path=str(tmp_path / "s.png")))
    assert shot["status"] == 403 and shot.get("title") == "Security Verification"


def test_every_url_loading_op_ships_a_status():
    """FAILS IF: a url-loading op is added (or one of these regresses) that
    returns a result dict without `status` — AST enumeration of the runner:
    every `op_*` whose body calls `page.goto` must put "status" in a dict
    literal it returns."""
    tree = ast.parse(inspect.getsource(BR))
    ops = [n for n in ast.walk(tree)
           if isinstance(n, ast.AsyncFunctionDef) and n.name.startswith("op_")]
    checked = []
    for fn in ops:
        loads = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "goto" for n in ast.walk(fn))
        if not loads or fn.name in ("op_interact", "op_click", "op_close"):
            continue
        keys = {k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
                for k in n.keys if isinstance(k, ast.Constant)}
        assert "status" in keys, fn.name
        checked.append(fn.name)
    assert set(checked) == {"op_navigate", "op_extract_text", "op_screenshot"}, checked
