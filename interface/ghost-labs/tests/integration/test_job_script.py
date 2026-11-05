"""labs.job_script run for real in ghost-lab/base:1 (bash, setsid, GNU coreutils): a detached job is
its whole process group (§4SX-C live 11-04 + review r2) — the unit tests can only pin it as a string."""
import subprocess

import pytest

from ghost_labs.labs import job_script

pytestmark = pytest.mark.integration
D = "/var/log/lab/jobs/job_t"


def run(cmd, check, wait="for i in $(seq 1 80); do [ -e {d}.rc ] && break; sleep 0.25; done"):
    script = (f"export GHOST_CMD={_q(cmd)}; sh -c {_q(job_script(D))}; " + wait.format(d=D)
              + f"; echo rc=$(cat {D}.rc 2>/dev/null || echo none); {check}")
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash", "ghost-lab/base:1",
                        "-c", script], capture_output=True, text=True, timeout=90)
    return r.stdout, r.stderr


def _q(s):
    import shlex
    return shlex.quote(s)


def test_exit_code_and_output_are_kept():
    out, _ = run("echo hello; exit 4", f"cat {D}.log")
    assert out.split() == ["rc=4", "hello"]


def test_a_backgrounded_child_and_a_pipeline_end_with_the_job():
    out, _ = run("(sleep 611 &); sleep 1 | cat; exit 3", "pgrep -f '^sleep 611' | wc -l")
    assert out.split() == ["rc=3", "0"]


def test_a_term_deaf_child_is_killed_before_rc_appears():
    out, _ = run("(trap '' TERM; exec sleep 612) & exit 0", "pgrep -f '^sleep 612' | wc -l")
    assert out.split() == ["rc=0", "0"]


def test_the_leader_is_the_group_and_rc_is_absent_while_it_runs():
    out, _ = run("sleep 30", f"p=$(cat {D}.pid); [ \"$(ps -o pgid= -p $p | tr -d ' ')\" = \"$p\" ] && echo leader; "
                 f"test -e {D}.rc && echo has-rc || echo no-rc", wait="sleep 1")
    assert out.split() == ["rc=none", "leader", "no-rc"]


def test_a_timeout_foreground_wrap_ends_its_pipeline():
    """The agent's `run` wrap (`timeout --foreground`) + this script: the TTL ends the whole job."""
    out, _ = run("timeout --foreground -k 1 1 bash -c 'sleep 613 | cat'", "pgrep -f '^sleep 613' | wc -l")
    assert out.split() == ["rc=124", "0"]


def test_the_rest_of_the_group_gets_TERM_first():
    """r2: TERM before KILL — a child that cleans up on TERM gets to (a KILL alone would hide this)."""
    out, _ = run("(trap 'echo graceful > /tmp/g; exit 0' TERM; while :; do sleep 0.1; done) & exit 0",
                 "cat /tmp/g 2>/dev/null || echo none")
    assert out.split() == ["rc=0", "graceful"]
