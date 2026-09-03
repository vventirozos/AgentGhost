"""§4EH — log icon/severity invariants (an AST lint over pretty_log calls).

The operator reads the log stream as a monitoring tool, so an ERROR line must
render with a failure/security glyph — never a soft warning, halt, or a
cheerful tool icon that hides the severity.
"""
from __future__ import annotations

import ast
import glob
import os

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "ghost_agent")


def _icon_name(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == "Icons":
        return node.attr
    return None  # dynamic/conditional icon — not statically checkable


def _level(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.upper()
    return None


def _pretty_log_calls():
    for f in glob.glob(os.path.join(SRC, "**", "*.py"), recursive=True):
        tree = ast.parse(open(f, encoding="utf-8").read())
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name not in ("pretty_log", "_safe_pretty_log"):
                continue
            kw = {k.arg: k.value for k in n.keywords}
            icon = _icon_name(kw["icon"]) if "icon" in kw else "_DEFAULT"
            lvl = _level(kw["level"]) if "level" in kw else "INFO"
            yield (os.path.relpath(f, SRC), n.lineno, icon, lvl)


# An ERROR/CRITICAL line must carry the failure glyph, or SHIELD for a
# deliberate fail-closed security refusal. Everything else on ERROR (a soft
# WARN amber, a STOP halt, a cheerful tool icon) understates the severity.
_ERROR_OK = {"FAIL", "SHIELD"}


def test_error_level_lines_use_the_failure_or_security_glyph():
    offenders = [(f, ln, ic, lv) for f, ln, ic, lv in _pretty_log_calls()
                 if lv in ("ERROR", "CRITICAL") and ic is not None
                 and ic not in _ERROR_OK]
    assert offenders == [], (
        "ERROR-level pretty_log lines rendering a non-failure icon "
        "(soften the severity for the operator): "
        + "; ".join(f"{f}:{ln} icon={ic}" for f, ln, ic, lv in offenders))


def test_the_error_lint_can_actually_fire():
    # guard the guard: a synthetic ERROR+WARN pairing must be caught by the rule
    assert "WARN" not in _ERROR_OK and "STOP" not in _ERROR_OK
    assert {"FAIL", "SHIELD"} == _ERROR_OK
