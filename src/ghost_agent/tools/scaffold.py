"""§4UU-5 — starter projects: `file_system(operation='write', path='<folder>', template='web-app')`.

Measured (25 days): the web OS was attempted 11 times from scratch — 10.2 h,
41 % of all owner coding time — as one file of up to 144 KB written in 14
appended parts; two attempts ran nothing in an hour. A practitioner bench
(qwen3-coder:30b, 0 → 90 / 100 with a ready project structure) and the
long-running-harness pattern (an init script + a test runner from the start)
point the same way: start a multi-feature web app from a skeleton that is
already split into files and already has a check that RUNS it.

`web-app` writes into an EMPTY (or new) folder:

* `index.html` — loads `css/style.css` and the scripts with classic
  `<script src>` tags (a file:// page cannot load ES modules — Chromium
  blocks them as cross-origin — so the system's page check and the test
  runner can both load it straight from disk);
* `js/app.js` — the `App` namespace and its start; one feature per file
  under `js/` (add a `<script>` tag for each, before `js/main.js`);
* `js/main.js` — starts the app and sets `window.__appReady`;
* `css/style.css`;
* `tests/check_page.py` — a headless Playwright check (file://, no network):
  exit 0 only when the page loads with no uncaught exception or console
  error AND every check in `CHECKS` passes. `python3 tests/check_page.py
  <word>` runs only the checks whose name contains the word — one check per
  feature is what a `checklist` item's check runs.

Nothing is ever overwritten: a folder that already holds files is refused.
file_system.py confirms each file with its ordinary write line — the shape
every write ledger already reads.
"""
from __future__ import annotations

import html
import os
import shlex
import re
from pathlib import Path
from typing import Dict, List, Tuple

TEMPLATES = ("web-app",)

_INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="css/style.css">
</head>
<body>
  <main id="app"></main>
  <!-- Classic scripts, in order: app.js first, one file per feature, main.js last.
       (No type="module": a page opened from disk cannot load ES modules.) -->
  <script src="js/app.js"></script>
  <script src="js/main.js"></script>
</body>
</html>
"""

_STYLE = """* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
#app { padding: 1rem; }
"""

_APP_JS = """// The app's namespace. Each feature lives in its own file under js/ and adds itself here:
//   App.features.push({ name: 'clock', start(root) { ... } });
// Add a <script src="js/<feature>.js"> tag for it in index.html, BEFORE js/main.js.
window.App = window.App || { features: [] };

App.start = function (root) {
  root.innerHTML = '<h1>{title}</h1><p id="status">ready</p>';
  App.features.forEach(function (f) { f.start(root); });
};
"""

_MAIN_JS = """// Starts the app once the page is parsed. Keep this file last in index.html.
document.addEventListener('DOMContentLoaded', function () {
  App.start(document.getElementById('app'));
  window.__appReady = true;   // tests/check_page.py waits for this
});
"""

_CHECK_PY = r'''#!/usr/bin/env python3
"""Headless check of this web app. Exit 0 only when EVERY selected check passes.

    python3 tests/check_page.py           # every check
    python3 tests/check_page.py clock     # only the checks whose name contains "clock" (every word given)

Each check loads index.html fresh from disk (file://, no network), fails on any
uncaught JS exception or console error, waits for window.__appReady, then runs
its steps. Add ONE check per feature to CHECKS.

Steps:
  ("click", selector)            ("fill", selector, text)      ("press", key)
  ("wait", milliseconds)         ("visible", selector)         ("hidden", selector)
  ("text", selector, substring)  ("count", selector, n)        ("eval", "js expression that must be truthy")
"""
import os
import sys
import threading
from pathlib import Path

CHECKS = [
    ("app loads", [("visible", "#app"), ("text", "#status", "ready")]),
]

PAGE = (Path(__file__).resolve().parent.parent / "index.html").as_uri()
STEP_TIMEOUT_MS = 5000
CHECK_TIMEOUT_S = 30          # a page that freezes (an endless loop) fails here instead of hanging the run
ARITY = {"click": 1, "fill": 2, "press": 1, "wait": 1, "visible": 1, "hidden": 1, "text": 2, "count": 2, "eval": 1}


def _watchdog(name):
    def fire():
        print(f"FAIL {name}: no answer in {CHECK_TIMEOUT_S}s — the page froze (an endless loop?)", flush=True)
        print("stopped: a frozen page ends the run (fix it, then run the checks again)", flush=True)
        os._exit(1)
    t = threading.Timer(CHECK_TIMEOUT_S, fire)
    t.daemon = True
    t.start()
    return t


def _until(fn, ms=STEP_TIMEOUT_MS):
    """Poll fn() until it returns "" (pass) or the time is up; returns the last failure text."""
    import time
    end = time.time() + ms / 1000
    while True:
        why = fn()
        if not why or time.time() >= end:
            return why
        time.sleep(0.1)


def run_check(browser, name, steps):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"uncaught: {e}"))
    page.on("console", lambda m: errors.append(
        f"console.{m.type}: {m.text}" + (f" ({m.location.get('url', '').rsplit('/', 1)[-1]})"
                                         if (m.location or {}).get("url") else "")) if m.type == "error" else None)
    try:
        page.goto(PAGE, wait_until="load", timeout=15000)
        for _ in range(STEP_TIMEOUT_MS // 100):        # ready, or the start threw (no point waiting it out)
            if errors or page.evaluate("window.__appReady === true"):
                break
            page.wait_for_timeout(100)
        if errors:
            return errors[0]
        if not page.evaluate("window.__appReady === true"):
            return "window.__appReady never became true (js/main.js did not finish starting the app)"
        for step in steps:
            kind, args = step[0], step[1:]
            if kind in ARITY and len(args) < ARITY[kind]:
                return f"step {step!r} needs {ARITY[kind]} argument(s) after {kind!r}"
            if kind == "click":
                page.click(args[0], timeout=STEP_TIMEOUT_MS)
            elif kind == "fill":
                page.fill(args[0], str(args[1]), timeout=STEP_TIMEOUT_MS)
            elif kind == "press":
                page.keyboard.press(args[0])
            elif kind == "wait":
                page.wait_for_timeout(int(args[0]))
            elif kind == "visible":
                page.wait_for_selector(args[0], state="visible", timeout=STEP_TIMEOUT_MS)
            elif kind == "hidden":
                page.wait_for_selector(args[0], state="hidden", timeout=STEP_TIMEOUT_MS)
            elif kind == "text":
                def _text():
                    got = page.inner_text(args[0], timeout=STEP_TIMEOUT_MS)
                    return "" if str(args[1]) in got else f"{args[0]} text is {got[:80]!r}, expected it to contain {args[1]!r}"
                why = _until(_text)
                if why:
                    return why
            elif kind == "count":
                def _count():
                    n = page.locator(args[0]).count()
                    return "" if n == int(args[1]) else f"{args[0]} matched {n} element(s), expected {args[1]}"
                why = _until(_count)
                if why:
                    return why
            elif kind == "eval":
                if not page.evaluate(args[0]):
                    return f"eval was falsy: {args[0]}"
            else:
                return f"unknown step {kind!r}"
            if errors:
                return errors[0]
        return errors[0] if errors else ""
    except Exception as e:  # a timeout, a missing selector
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
        return (errors[0] + " | " if errors else "") + first
    finally:
        page.close()


def main(argv):
    from playwright.sync_api import sync_playwright
    words = [a.lower() for a in argv[1:]]
    chosen = [(n, s) for n, s in CHECKS if not words or all(w in n.lower() for w in words)]
    if not chosen:
        print(f"NO CHECK matches {' '.join(argv[1:])!r} — add one to CHECKS")
        return 2
    failed = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--host-resolver-rules=MAP * ~NOTFOUND"])
        try:
            for name, steps in chosen:
                dog = _watchdog(name)
                try:
                    why = run_check(browser, name, steps)
                finally:
                    dog.cancel()
                print(("FAIL " + name + ": " + why) if why else ("PASS " + name))
                failed += bool(why)
        finally:
            browser.close()
    print(f"{len(chosen) - failed}/{len(chosen)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _title_from(folder: str) -> str:
    name = Path(str(folder).rstrip("/")).name or "App"
    words = re.split(r"[-_\s]+", name)
    return " ".join(w[:1].upper() + w[1:] for w in words if w) or "App"


def render(template: str, folder: str) -> Dict[str, str]:
    """relative path → content for `template`; KeyError for an unknown template."""
    if template != "web-app":
        raise KeyError(template)
    title = _title_from(folder)
    safe = html.escape(title)
    js_title = safe.replace("\\", "\\\\").replace("'", "\\'")
    return {
        "index.html": _INDEX.format(title=safe),
        "css/style.css": _STYLE,
        "js/app.js": _APP_JS.replace("{title}", js_title),
        "js/main.js": _MAIN_JS,
        "tests/check_page.py": _CHECK_PY,
    }


def scaffold(sandbox_dir, folder: str, template: str, safe_path, mark_written=None) -> Tuple[bool, str, list]:
    """Write `template` into `folder` (inside the sandbox; `safe_path(sandbox_dir, rel)` resolves and guards a
    path). Returns (ok, message, [(workspace-relative path, chars), …]) — the message is the refusal, or the
    guidance that follows the file tool's own confirmation lines (file_system.py writes those, so its
    producer/parser parity check keeps seeing every confirmation). Refuses an unknown template, a path that is a file,
    and a folder that already holds files — nothing is ever overwritten."""
    template = str(template or "").strip().lower()
    folder = str(folder or "").strip().rstrip("/")      # a leading "/" is kept: _get_safe_path maps it
    if template not in TEMPLATES:
        return False, (f"SYSTEM INSTRUCTION: unknown template {template!r} — templates: {', '.join(TEMPLATES)}."), []
    if not folder:
        return False, ("SYSTEM INSTRUCTION: template= needs path= a NEW folder for the app (e.g. path='tetris'), "
                       "not the workspace root."), []
    try:
        root = safe_path(sandbox_dir, folder)
    except ValueError as e:
        return False, f"SYSTEM INSTRUCTION: {e}", []
    base = Path(sandbox_dir).resolve()
    try:
        rel_root = root.resolve().relative_to(base).as_posix()
    except ValueError:
        return False, f"SYSTEM INSTRUCTION: '{folder}' is outside the workspace.", []
    if rel_root in ("", "."):
        return False, ("SYSTEM INSTRUCTION: template= needs path= a NEW folder for the app (e.g. path='tetris'), "
                       "not the workspace root."), []
    if ".git" in Path(rel_root).parts:
        return False, "SYSTEM INSTRUCTION: a template never writes inside a .git folder.", []
    folder = rel_root                                   # every path below is the resolved, workspace-relative one
    if Path(folder).suffix.lower() in (".html", ".htm", ".js", ".css", ".py", ".json", ".md", ".txt"):
        return False, (f"SYSTEM INSTRUCTION: path= is the app's FOLDER (e.g. '{Path(folder).parent.as_posix()}' or "
                       f"'{Path(folder).stem}'), not a file — the template writes index.html etc. inside it."), []
    if root.exists() and not root.is_dir():
        return False, f"SYSTEM INSTRUCTION: '{folder}' is a file — template= needs a folder.", []
    if root.is_dir():
        present = [p for p in root.iterdir() if not p.name.startswith(".")]
        if present:
            return False, (f"SYSTEM INSTRUCTION: '{folder}' already holds {len(present)} item(s) — a template only "
                           f"starts a NEW (or empty) folder and never overwrites. Pick a new folder name."), []
    files = render(template, folder)
    written: List[Tuple[str, int]] = []
    lines: List[str] = []
    for rel, text in files.items():
        target = safe_path(sandbox_dir, f"{folder}/{rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        if mark_written is not None:
            try:
                mark_written(target)                    # this request's own file: a later whole rewrite is allowed
            except Exception:  # noqa: BLE001
                pass
        if rel.endswith(".py"):
            try:
                os.chmod(target, 0o755)
            except OSError:
                pass
        written.append((f"{folder}/{rel}", len(text)))
    lines.append(
        f"Started the {template} template in '{folder}/'. Build ONE feature at a time: its own js/<feature>.js "
        f"(a <script> tag in index.html before js/main.js), its own check in tests/check_page.py CHECKS, then run "
        f"`cd {shlex.quote(folder)} && python3 tests/check_page.py <feature>` — exit 0 = it works. "
        f"`python3 {shlex.quote(folder + '/tests/check_page.py')}` (from the workspace) runs every check (the whole app).")
    return True, "\n".join(lines), written

