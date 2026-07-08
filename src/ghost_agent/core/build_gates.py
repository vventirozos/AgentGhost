"""Post-build verification gates for coding work (2026-07-08).

Born from the chess-coach session: the agent restated the project's core
constraint ("Ghost plays directly, not a coded AI") in its own plan and then
BUILT a heuristic engine anyway — constraints were replayed into context but
nothing ever *checked the artifact*. The same build also shipped three
crash-on-first-touch bugs because no endpoint was ever exercised.

Two gates, both fed back into the build-retry loop on failure:

* :func:`constraint_gate` — a cheap LLM audit of the files a task produced
  against the project's stored constraints. Judgement-based (the existing
  DONE-gate in ``tools/projects.py`` is evidence-based only — it checks that
  the model *wrote* result text, not that the artifact complies).
* :func:`smoke_gate` — mechanical: ``py_compile`` every written ``.py``,
  and when a Flask app is detected, sweep its routes with ``test_client``
  (no server process needed): GET routes must not 500; POST routes get an
  empty JSON body and may 4xx but must not 5xx.

Both gates FAIL OPEN on infrastructure errors (LLM down, sandbox hiccup):
a broken gate must never block builds; it logs and lets the build pass.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

# Budgets for the constraint-audit prompt: enough to judge real code, small
# enough to stay a cheap background call.
_MAX_FILES = 8
_MAX_CHARS_PER_FILE = 4000
_MAX_TOTAL_CHARS = 18000


def files_from_specs(file_specs: list) -> Dict[str, str]:
    """Best-effort {path: written-content} map from build-spec file entries.

    ``content``/``append`` carry the text this task actually produced —
    exactly what the constraint audit should judge. Edit-mode entries
    contribute their ``replace_with`` bodies.
    """
    out: Dict[str, str] = {}
    for f in file_specs or []:
        if not isinstance(f, dict):
            continue
        path = (f.get("path") or "").strip()
        if not path:
            continue
        body = f.get("content") or f.get("append") or ""
        if not body and isinstance(f.get("edits"), list):
            body = "\n...\n".join(
                str(e.get("replace_with") or "") for e in f["edits"]
                if isinstance(e, dict))
        if body:
            out[path] = str(body)
    return out


async def constraint_gate(context, constraints: List[str],
                          files: Dict[str, str], *,
                          is_background: bool = True,
                          model_name: str = "") -> Tuple[bool, str]:
    """LLM audit: does the built artifact violate a stated constraint?

    Returns ``(ok, reason)`` — ``ok=False`` means a CONFIRMED violation and
    ``reason`` carries the constraint + evidence for the retry feedback.
    Fails open (``ok=True``) when there is nothing to judge or the LLM call
    fails.
    """
    constraints = [str(c).strip() for c in (constraints or []) if str(c).strip()]
    if not constraints or not files:
        return True, ""
    llm = getattr(context, "llm_client", None)
    if llm is None:
        return True, ""
    model = model_name or getattr(getattr(context, "args", None), "model", "default")

    total = 0
    file_blocks = []
    for path, body in list(files.items())[:_MAX_FILES]:
        snippet = str(body)[:_MAX_CHARS_PER_FILE]
        total += len(snippet)
        if total > _MAX_TOTAL_CHARS:
            break
        file_blocks.append(f"--- {path} ---\n{snippet}")

    prompt = (
        "You are a build-constraint auditor. The USER imposed these "
        "constraints on this project (verbatim):\n"
        + "\n".join(f"{i+1}. {c}" for i, c in enumerate(constraints))
        + "\n\nThe agent just built/edited these files:\n\n"
        + "\n\n".join(file_blocks)
        + "\n\nDoes any file VIOLATE any constraint? Judge only real "
        "violations of what the user stated — style or quality issues are "
        "NOT violations, and absence of evidence is NOT a violation. Reply "
        "with STRICT JSON only:\n"
        '{"violates": true|false, "constraint": "<the violated constraint '
        'or empty>", "evidence": "<file + the specific code that violates '
        'it, or empty>"}'
    )
    try:
        data = await llm.chat_completion({
            "model": model,
            "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.0,
            "max_tokens": 400,
            "stream": False,
        }, is_background=is_background)
        content = (data.get("choices", [{}])[0].get("message", {})
                   .get("content") or "")
    except Exception as e:  # noqa: BLE001 — gate must fail open on infra errors
        logger.debug("constraint_gate LLM call failed (fail-open): %s", e)
        return True, ""

    for m in re.finditer(r'\{[^{}]*"violates"[^{}]*\}', content, re.DOTALL):
        try:
            verdict = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(verdict, dict) and verdict.get("violates") is True:
            constraint = str(verdict.get("constraint") or "").strip()
            evidence = str(verdict.get("evidence") or "").strip()
            reason = (
                "CONSTRAINT VIOLATION: the build violates the user's stated "
                f"constraint: \"{constraint}\". Evidence: {evidence}. "
                "Rework the implementation so it honors the constraint — do "
                "NOT just reword comments."
            )
            pretty_log("Constraint Gate", reason[:200], level="WARNING",
                       icon=Icons.WARN)
            return False, reason
        if isinstance(verdict, dict) and verdict.get("violates") is False:
            return True, ""
    # Unparseable reply → fail open.
    logger.debug("constraint_gate: unparseable auditor reply (fail-open)")
    return True, ""


# The smoke script self-bounds with SIGALRM so a module that starts a
# blocking server on import can never wedge the (much longer) sandbox exec
# timeout. Written into the sandbox and run there via the execute tool.
_SMOKE_TEMPLATE = r'''
import json, py_compile, signal, sys, traceback
signal.alarm(45)
WRITTEN = {written!r}
failures = []
for path in WRITTEN:
    if not path.endswith(".py"):
        continue
    try:
        py_compile.compile(path, doraise=True)
    except Exception as e:
        failures.append(f"py_compile {{path}}: {{e}}")

app_obj = None
if not failures:
    import importlib.util
    for path in WRITTEN:
        if not path.endswith(".py"):
            continue
        try:
            src = open(path).read()
        except Exception:
            continue
        if "Flask(__name__" not in src:
            continue
        try:
            spec = importlib.util.spec_from_file_location("smoke_target", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            app_obj = getattr(mod, "app", None)
        except SystemExit:
            pass
        except Exception:
            failures.append(f"importing {{path}} crashed:\n"
                            + traceback.format_exc(limit=3))
        if app_obj is not None:
            break

if app_obj is not None:
    try:
        client = app_obj.test_client()
        for rule in app_obj.url_map.iter_rules():
            r = str(rule.rule)
            if r.startswith("/static") or "<" in r:
                continue
            methods = rule.methods or set()
            if "GET" in methods:
                resp = client.get(r)
                if resp.status_code >= 500:
                    failures.append(f"GET {{r}} -> {{resp.status_code}}")
            elif "POST" in methods:
                resp = client.post(r, json={{}})
                if resp.status_code >= 500:
                    failures.append(f"POST {{r}} (empty JSON) -> {{resp.status_code}}")
    except Exception:
        failures.append("route sweep crashed:\n" + traceback.format_exc(limit=3))

print("SMOKE_RESULT " + json.dumps({{"failures": failures}}))
sys.exit(1 if failures else 0)
'''


async def smoke_gate(tool_runner, written: List[str]) -> Optional[str]:
    """Mechanical smoke of the files a build just wrote.

    Returns ``None`` on pass, or a feedback string describing the failures
    (compile error, import crash, 5xx route) for the build-retry loop.
    Fails open on sandbox/tooling errors.
    """
    py_files = [p for p in (written or []) if str(p).endswith(".py")]
    if not py_files or tool_runner is None:
        return None
    script = _SMOKE_TEMPLATE.format(written=[str(p) for p in written])
    try:
        out = await tool_runner("execute", {
            "filename": ".smoke_gate.py",
            "content": script,
        })
    except Exception as e:  # noqa: BLE001 — gate must fail open
        logger.debug("smoke_gate execute failed (fail-open): %s", e)
        return None
    text = str(out or "")
    m = re.search(r"SMOKE_RESULT (\{.*\})", text)
    if not m:
        # Couldn't find our marker — the sandbox mangled the run; fail open
        # UNLESS the output is clearly our script dying (traceback with the
        # smoke filename), which we surface.
        return None
    try:
        failures = json.loads(m.group(1)).get("failures") or []
    except Exception:
        return None
    if not failures:
        return None
    reason = ("SMOKE GATE FAILED — the build crashes on first touch:\n- "
              + "\n- ".join(str(f)[:300] for f in failures[:6])
              + "\nFix these before the task can complete.")
    pretty_log("Smoke Gate", f"{len(failures)} failure(s): "
               + "; ".join(str(f)[:80] for f in failures[:3]),
               level="WARNING", icon=Icons.WARN)
    return reason
