"""§4KG (2026-09-24): the everyday CLI set the sandbox image lacked.

Measured before adding anything: the UDP leak probe in tor_egress ran `nc`,
which was not in the image (exit 127 printed "blocked" — a harness that
cannot run reporting success; and nc's exit code turned out to say nothing
anyway, so the probe now observes EPERM from Python); `unzip` is referenced
by sandbox code; `stockfish` was installed since v5 but lives in /usr/games,
off the exec PATH (a `command -v` proxy read "missing" for a package that was
there); `ffmpeg` was asked for and missing.
The set lands through provisioning marker v10: an in-place delta for v9
images, the full-provision line, and the Dockerfile — three copies pinned to
one pair of tuples.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ghost_agent.sandbox import docker as dmod  # noqa: E402

PK, BI = dmod.SANDBOX_TOOL_PACKAGES, dmod.SANDBOX_TOOL_BINARIES

# PARSED, not text: the provisioning ladder is a set of string constants bound
# inside `_ensure_running_impl`; read them off the AST (the pin-quality
# ratchet rejects `"literal" in getsource(...)` pins, and rightly — a mutant
# that leaves the literal in a comment survives those).
import ast as _ast  # noqa: E402
_TREE = _ast.parse(inspect.getsource(dmod))


def _assigned(name: str) -> str:
    """The string constant bound to ``name`` inside `_ensure_running_impl`."""
    fn = next(n for n in _ast.walk(_TREE) if isinstance(n, _ast.FunctionDef) and n.name == "_ensure_running_impl")
    for n in _ast.walk(fn):
        if isinstance(n, _ast.Assign) and any(isinstance(t_, _ast.Name) and t_.id == name for t_ in n.targets):
            if isinstance(n.value, _ast.Constant) and isinstance(n.value.value, str):
                return n.value.value
    raise AssertionError(f"{name} is not bound to a string constant in _ensure_running_impl")


def _upgrade_log_fstring_names() -> set:
    """Names referenced inside the f-string of the 'Upgrading … in place' log call."""
    fn = next(n for n in _ast.walk(_TREE) if isinstance(n, _ast.FunctionDef) and n.name == "_ensure_running_impl")
    names = set()
    for call in _ast.walk(fn):
        if not (isinstance(call, _ast.Call) and getattr(call.func, "id", "") == "pretty_log"):
            continue
        text = "".join(v.value for a in call.args if isinstance(a, _ast.JoinedStr)
                       for v in a.values if isinstance(v, _ast.Constant))
        if "in place" in text:
            for a in call.args:
                for n in _ast.walk(a):
                    if isinstance(n, _ast.Name):
                        names.add(n.id)
            return names, text
    return names, ""


def test_the_measured_needs_are_in_the_set():
    assert "netcat-openbsd" in PK and "nc" in BI, "network plumbing a shell task reaches for"
    assert "unzip" in PK and "unzip" in BI, "sandbox code references unzip"
    assert "stockfish" in PK and "stockfish" in BI, "installed since v5 but off the exec PATH until v10's symlink"
    assert "ffmpeg" in PK and "ffmpeg" in BI, "asked for by the model and missing"
    for gone in ("iputils-ping", "traceroute"):
        assert gone not in PK, f"{gone} needs NET_RAW, which §4KF dropped"


def test_the_v10_set_outlives_its_delta():
    """v11 (the PostgreSQL 18 client) replaced the v10 delta, so a v9 image now
    takes the full provision — which must still install and link the v10 set,
    and the v10 delta must not linger anywhere a v10 image could re-run it."""
    delta = _assigned("upgrade_delta_cmd")
    assert _assigned("prev_marker_path") == "/root/.supercharged.v10"
    for pkg in ("netcat-openbsd", "ffmpeg", "stockfish"):
        assert not re.search(rf"(^|\s){re.escape(pkg)}(\s|'|$)", delta), f"the v11 delta re-installs v10's {pkg}"


def test_the_full_provision_and_the_recipe_carry_the_same_packages():
    apt_line = _assigned("apt_cmd")
    recipe = (_ROOT / "sandbox" / "Dockerfile").read_text(encoding="utf-8")
    active = "\n".join(ln for ln in recipe.splitlines() if not ln.lstrip().startswith("#"))
    for pkg in PK:
        assert re.search(rf"(^|\s){re.escape(pkg)}(\s|'|\\|$)", apt_line), f"full provision lacks {pkg}"
        assert re.search(rf"(^|\s){re.escape(pkg)}(\s|\\|$)", active), f"Dockerfile lacks {pkg}"
    assert "RUN touch /root/.supercharged.v11" in active, "the recipe must stamp the CURRENT marker"
    assert ".supercharged.v10" not in active


def test_stockfish_is_put_on_the_exec_path_in_all_three_copies():
    """Measured on the live image: the stockfish PACKAGE was installed since
    v5, but Debian puts the binary in /usr/games, which the python image's
    PATH lacks — `command -v stockfish` failed, the model's `stockfish`
    calls failed, and a v10 verify chain would never have passed."""
    link = "ln -sf /usr/games/stockfish /usr/local/bin/stockfish"
    apt_line = _assigned("apt_cmd")
    recipe = (_ROOT / "sandbox" / "Dockerfile").read_text(encoding="utf-8")
    assert link in apt_line
    assert f"RUN {link}" in recipe
    assert apt_line.index("stockfish --no-install-recommends &&") < apt_line.index(link), "the link follows the install inside the same command"


def test_the_upgrade_log_names_what_it_adds():
    """The log line used to say "(adds iptables)" — a v9 literal that would
    have lied on v10. It must derive from the CURRENT delta's tuple (PARSED:
    the f-string of the 'Upgrading … in place' call references
    SANDBOX_PG_PACKAGES since v11)."""
    names, text = _upgrade_log_fstring_names()
    assert text, "the in-place upgrade log call was not found"
    assert "SANDBOX_PG_PACKAGES" in names, names
    assert "SANDBOX_TOOL_BINARIES" not in names, "v10's set would be announced on a v11 upgrade"
    assert "iptables" not in text


def test_a_failed_delta_says_why_before_falling_back(tmp_path):
    """A mirror down over Tor, a verify chain that missed a binary and a
    timeout were indistinguishable: the delta's exit code and output were
    discarded and only an exception reached DEBUG. The WARNING must carry
    both (driven through the real provisioning path with an exec stub)."""
    from unittest.mock import patch
    from tests.test_sandbox_marker_upgrade import CHROMIUM, CHROMIUM_PRESENT, CUR, DELTA_KEY, PREV, _stub
    sb = _stub(tmp_path)
    overrides = {CUR: (1, b""), PREV: (0, b""),
                 DELTA_KEY: (1, b"Err:1 http://deb.debian.org bookworm InRelease\n  Could not connect (mirror down)")}

    def _exec(cmd, *a, **k):          # the marker test's driver, minus its own pretty_log patch
        for key, val in overrides.items():
            if key in cmd:
                return val
        return CHROMIUM_PRESENT if CHROMIUM in cmd else (0, b"")
    sb.container.exec_run.side_effect = _exec
    with patch.object(dmod, "pretty_log") as plog:
        sb.ensure_running()
    warns = [" ".join(str(a) for a in c.args) for c in plog.call_args_list if c.kwargs.get("level") == "WARNING"]
    fallback = [w for w in warns if "did not verify" in w]
    assert fallback, warns
    assert "delta exit 1" in fallback[0] and "mirror down" in fallback[0], fallback[0]


def test_the_full_provision_runs_unattended_without_recommends():
    """The fallback path used to pull 114 MiB with recommends (ghostscript,
    fonts, torsocks) while the delta and the recipe pull 60 MiB without —
    "three copies agree" was true of names, not results."""
    apt_line = _assigned("apt_cmd")
    assert "--no-install-recommends" in apt_line and "DEBIAN_FRONTEND=noninteractive" in apt_line


def test_the_full_provision_installs_the_pg18_client_after_the_repo_setup():
    """v11: a fresh image (no marker) gets psql 18 too — the PGDG setup runs
    AFTER the main install (which brings postgresql-common) and BEFORE the
    client install from the repo it adds."""
    common, client = dmod.SANDBOX_PG_PACKAGES
    apt_line = _assigned("apt_cmd")
    setup = "/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y"
    assert apt_line.index(f" {common} ") < apt_line.index(setup) \
        < apt_line.index(f"install -y --no-install-recommends {client}"), apt_line
    recipe = (_ROOT / "sandbox" / "Dockerfile").read_text(encoding="utf-8")
    active = "\n".join(ln for ln in recipe.splitlines() if not ln.lstrip().startswith("#"))
    assert active.index(setup) < active.index(client) < active.index("RUN touch /root/.supercharged.v11")
