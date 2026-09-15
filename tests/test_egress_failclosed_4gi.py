"""§4GI (2026-09-13): when Tor-only egress cannot be established the sandbox
has NO network, not direct network.

§4FU's enforcement was fail-closed only from the moment the iptables rules
went in. Every branch BEFORE that — iptables missing from the image, the
privileged exec refused, an exception before the rules landed — logged an
ERROR, set ``_egress_state = "unavailable"`` and left the container serving
``execute``/browser calls with cleartext egress. Nothing read the flag. Now
``_block_egress_hard`` disconnects the container from every docker network
it is attached to, the state becomes "blocked", and
``egress_is_enforced_or_blocked()`` is the one predicate a tool can consult.

Harness: the same exec stub as tests/test_sandbox_tor_egress.py. Every pin
names the pre-fix world: no disconnect call, state "unavailable", predicate
False.
"""
import ast
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ghost_agent.sandbox import docker as docker_mod
from tests.test_sandbox_tor_egress import _cmds, _drive, _stub


def _disconnects(sb):
    return sb.client.networks.get.return_value.disconnect.call_args_list


def test_missing_iptables_disconnects_the_container_and_reads_as_blocked(tmp_path):
    sb = _stub(tmp_path)
    sb.container.attrs = {"HostConfig": {"NetworkMode": "bridge"},
                          "NetworkSettings": {"Networks": {"ghostnet": {}, "bridge": {}}}}
    seen, plog = _drive(sb, {"command -v iptables && command -v tor": (1, b"")})
    assert sb._egress_state == "blocked"
    assert sb.egress_is_enforced_or_blocked() is True
    nets = [c.args[0] for c in sb.client.networks.get.call_args_list]
    assert sorted(nets) == ["bridge", "ghostnet"]
    assert len(_disconnects(sb)) == 2
    assert all(c.kwargs.get("force") is True for c in _disconnects(sb))
    assert any("DISCONNECTED" in str(c.args[1]) for c in plog.call_args_list)


def test_a_refused_privileged_exec_disconnects_the_container(tmp_path):
    sb = _stub(tmp_path)
    seen, plog = _drive(sb, {"iptables -t nat -N": (2, b"iptables: Permission denied (you must be root).")})
    assert sb._egress_state == "blocked"
    assert sb.egress_is_enforced_or_blocked() is True
    assert _disconnects(sb)


def test_an_exception_before_the_rules_land_disconnects_the_container(tmp_path):
    sb = _stub(tmp_path)

    def _boom(cmd):
        raise RuntimeError("daemon wedged")
    seen, plog = _drive(sb, {"command -v iptables && command -v tor": _boom})
    assert sb._egress_state == "blocked"
    assert _disconnects(sb)
    assert any("raised before the rules landed" in str(c.args[1]) for c in plog.call_args_list)


def test_an_exception_after_the_rules_landed_keeps_the_rules_and_does_not_disconnect(tmp_path):
    """The rules are in (fail-closed by themselves); a later hiccup must not
    tear the network down on top."""
    sb = _stub(tmp_path)

    def _boom(cmd):
        raise RuntimeError("verify hiccup")
    from ghost_agent.sandbox import tor_egress as T
    seen, plog = _drive(sb, {T.CHECK_URL: _boom})
    assert sb._egress_state == "blocked"
    assert not _disconnects(sb)


def test_a_disconnect_that_itself_fails_stays_unavailable_and_screams(tmp_path):
    sb = _stub(tmp_path)
    sb.client.networks.get.return_value.disconnect.side_effect = RuntimeError("API down")
    seen, plog = _drive(sb, {"command -v iptables && command -v tor": (1, b"")})
    assert sb._egress_state == "unavailable"
    assert sb.egress_is_enforced_or_blocked() is False
    assert any(c.kwargs.get("level") == "CRITICAL" for c in plog.call_args_list)


def test_host_networking_reads_as_not_blocked_and_is_not_disconnected(tmp_path):
    """The one branch that cannot be closed: the container IS the host's
    namespace. The predicate says so; nothing is disconnected."""
    sb = _stub(tmp_path, network="host")
    _drive(sb)
    assert sb._egress_state == "unavailable"
    assert sb.egress_is_enforced_or_blocked() is False
    assert not _disconnects(sb)


def test_the_happy_path_is_unchanged(tmp_path):
    sb = _stub(tmp_path)
    seen, plog = _drive(sb)
    assert sb._egress_state == "enforced"
    assert sb.egress_is_enforced_or_blocked() is True
    assert not _disconnects(sb)


# ── R1 enumeration ───────────────────────────────────────────────────────────

def _egress_state_assignments(tree):
    """Every egress-state TRANSITION, as (function, state, lineno).

    §4GJ round 3 made `_set_egress_state(state, reason="")` the one writer —
    the reason has to travel with the state, or a later branch inherits the
    previous one's remedy. So a transition is now a CALL to that setter, and
    a bare `self._egress_state = …` outside the setter is a bypass: it is
    reported here as a `(function, state, lineno)` row too, so the rules
    below judge it exactly as they judged the old direct assignments.
    """
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and getattr(
                    node.func, "attr", "") == "_set_egress_state":
                arg = node.args[0] if node.args else None
                val = arg.value if isinstance(arg, ast.Constant) else None
                out.append((fn.name, val, node.lineno))
            elif isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and t.attr == "_egress_state"
                    for t in node.targets):
                if fn.name == "_set_egress_state":
                    continue          # the setter's own body, not a bypass
                val = node.value.value if isinstance(node.value, ast.Constant) else None
                out.append((fn.name, val, node.lineno))
    return out


def test_the_setter_is_the_only_writer_of_the_egress_state():
    """The class §4GJ round 3 closed: the reason is part of the transition.
    Fails in any tree where a branch assigns `_egress_state` directly again
    and so leaves `_egress_unavailable_reason` holding the last branch's
    remedy."""
    tree = ast.parse(Path(docker_mod.__file__).read_text())
    assert _bare_state_writes(tree) == [], (
        f"bare _egress_state writes bypass the setter: {_bare_state_writes(tree)}")


def _bare_state_writes(tree):
    """Assignments to `_egress_state` outside the one setter. ONE
    implementation, shared with the fires-check below: a fires-check that
    re-implements the rule proves nothing about the rule that runs (the
    §4GJ battery survived a mutant that neutered the assertion because
    nothing else exercised this logic)."""
    return [(fn.name, node.lineno)
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name != "_set_egress_state"
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "_egress_state"
                for t in node.targets)]


def test_the_setter_enumeration_fires_on_a_bypass():
    """R7-2, through the same helper."""
    bypass = ("class D:\n"
              "    def sneaky(self):\n"
              "        self._egress_state = 'enforced'\n")
    assert _bare_state_writes(ast.parse(bypass)) == [("sneaky", 3)]
    through = ("class D:\n"
               "    def _set_egress_state(self, s, reason=''):\n"
               "        self._egress_state = s\n"
               "    def ok(self):\n"
               "        self._set_egress_state('blocked')\n")
    assert _bare_state_writes(ast.parse(through)) == []


def test_every_egress_state_write_is_enforced_blocked_or_the_two_named_exceptions():
    """`"unavailable"` may be written in exactly two places: the host-mode
    branch of `_enforce_tor_egress` (cannot be closed) and the failure tail
    of `_block_egress_hard` (the disconnect itself failed). Every other
    write is "enforced" or "blocked". A new branch that writes
    "unavailable" anywhere else reddens this."""
    tree = ast.parse(Path(docker_mod.__file__).read_text())
    writes = _egress_state_assignments(tree)
    assert writes, "no _egress_state writes found — the enumeration lost its subject"
    unavailable = sorted((fn, ln) for fn, val, ln in writes if val == "unavailable")
    assert [fn for fn, _ in unavailable] == ["_block_egress_hard", "_enforce_tor_egress"], unavailable
    others = [(fn, val) for fn, val, _ in writes if val != "unavailable"]
    # R3 round 2: `_recreate_if_cut_off` resets the state to "" (a NEW
    # generation is about to be provisioned and enforced) — the one allowed
    # reset, in the one function that drops the container.
    assert others and all(val in ("enforced", "blocked")
                          or (val == "" and fn == "_recreate_if_cut_off")
                          for fn, val in others), others
    assert [fn for fn, val in others if val == ""] == ["_recreate_if_cut_off"]
    # ⚠ EVERY BRANCH THAT LEARNS EGRESS IS NOT TOR MUST CUT THE CONTAINER
    # OFF — counted, because a count is what catches a new branch that only
    # LABELS the state. §4GK round 4 added the fourth: a verification that
    # came back `IsTor=false` (a MEASURED leak) used to call
    # `_set_egress_state("blocked")`, which writes a string and touches no
    # network, so the gate answered "blocked" over a container that had just
    # been proven to reach the internet directly with the host's IP.
    enforce = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_enforce_tor_egress")
    calls = [n for n in ast.walk(enforce) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "_block_egress_hard"]
    assert len(calls) == 4, len(calls)
    # (Whether the LEAK branch in particular blocks is pinned by execution in
    # tests/test_sandbox_tor_egress.py, which drives the branch and asserts
    # the container was really disconnected — a stronger instrument than any
    # AST rule here, because one legitimate branch above DOES label "blocked"
    # after the rules load successfully.)


def test_the_enumeration_fires_on_a_stray_unavailable_write():
    src = Path(docker_mod.__file__).read_text()
    needle = '            self._set_egress_state("blocked")\n            if self._exec_run(_te.tor_running_as_expected_cmd())[0] != 0:'
    assert src.count(needle) == 1
    broken = src.replace(needle, needle.replace('"blocked"', '"unavailable"'))
    writes = _egress_state_assignments(ast.parse(broken))
    unavailable = sorted(fn for fn, val, _ in writes if val == "unavailable")
    assert unavailable != ["_block_egress_hard", "_enforce_tor_egress"]
