"""kept.seed_link_cmd run for real in ghost-lab/base:1 (GNU readlink, bash): the cases the unit
tests can only pin as strings (review r2 #1, r3 #10, r4 #6/#7)."""
import subprocess

import pytest

from ghost_labs.kept import seed_link_cmd

pytestmark = pytest.mark.integration


def run(setup, path, check):
    script = f"set -u; rm -rf /app /data; mkdir -p /app /data; {setup}; ({seed_link_cmd('/app', path)}); echo rc=$?; {check}"
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash", "ghost-lab/base:1", "-c", script],
                       capture_output=True, text=True, timeout=60)
    return r.stdout, r.stderr


def test_fresh_seed_moves_and_links():
    out, _ = run("mkdir -p /app/saves; echo new > /app/saves/f", "saves", "cat /data/saves/f; readlink /app/saves")
    assert out.split() == ["rc=0", "new", "/data/saves"]


def test_existing_data_is_never_replaced():
    out, _ = run("mkdir -p /app/saves /data/saves; echo new > /app/saves/f; echo OLD > /data/saves/f", "saves",
                 "cat /data/saves/f; readlink /app/saves")
    assert out.split() == ["rc=0", "OLD", "/data/saves"]


def test_parent_symlink_into_data_refused_and_data_kept():
    out, err = run("mkdir -p /data/x/slot1; echo keep > /data/x/slot1/k; ln -s /data/x /app/x", "x/slot1",
                   "cat /data/x/slot1/k")
    assert out.split() == ["rc=9", "keep"] and "unsafe link parent" in err


def test_deep_parent_symlink_creates_nothing_in_data():
    out, err = run("ln -s /data /app/a", "a/b/c", "ls -A /data | wc -l")
    assert out.split() == ["rc=9", "0"] and "unsafe link parent" in err


def test_leaf_symlink_refused():
    out, err = run("ln -s /etc /app/saves", "saves", "ls -A /data | wc -l")
    assert out.split() == ["rc=9", "0"] and "symlink" in err


def test_missing_everywhere_is_an_error():
    out, err = run("true", "nope", "test -e /app/nope; echo $?")
    assert out.split() == ["rc=8", "1"] and "neither" in err
