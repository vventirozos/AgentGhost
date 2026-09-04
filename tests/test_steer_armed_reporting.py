"""An open gate on a disarmed process steers nothing (§4ER, 2026-09-04).

The pre-flight steer has two independent switches:

  * the ALLOW-LIST, which opens itself from measured data, and
  * `GHOST_IMAGINE`, the process flag the consumer checks first.

On 2026-09-04 six buckets were open — `file_system|ext:py` (n=1896, Brier
skill +0.41), `manage_services|` (n=2475), `execute|cmd:cd` (n=603) and three
more, 16% of predicted calls — while the flag was unset, so every deferral was
discarded at the first line of `_imagine_preflight_note`. `introspect
learning` said "6/145 buckets DISCRIMINATE" and nothing said the steer was
dead. The module docstring meanwhile asserted the gate was closed on every
bucket, which had stopped being true.

The property under review: **the report distinguishes "the gate opened" from
"the steer is acting", and neither is ever asserted from a constant.**
"""

import pytest

from ghost_agent.core import imagination as I
from ghost_agent.core.imagination import gate_stats, steer_armed

FLAGS = ("GHOST_IMAGINE", "GHOST_IMAGINE_PREFLIGHT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)


def test_the_master_flag_decides_and_defaults_off(monkeypatch):
    assert steer_armed() is False
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv("GHOST_IMAGINE", on)
        assert steer_armed() is True, on
    for off in ("0", "false", "no", "off", "", "maybe"):
        monkeypatch.setenv("GHOST_IMAGINE", off)
        assert steer_armed() is False, off


def test_the_preflight_switch_disarms_on_its_own(monkeypatch):
    """Two kill switches, and the narrow one must work without the broad one:
    `GHOST_IMAGINE_PREFLIGHT=0` stops this steer while leaving the rest of
    imagination armed."""
    monkeypatch.setenv("GHOST_IMAGINE", "1")
    assert steer_armed() is True
    monkeypatch.setenv("GHOST_IMAGINE_PREFLIGHT", "0")
    assert steer_armed() is False


def test_armed_is_read_fresh_every_time(monkeypatch):
    """⚠ NOT CACHED, DELIBERATELY. The report must describe THIS process, not
    a launcher edit it never saw — a cached value would let the screen say
    ARMED about a flag the running agent does not have."""
    monkeypatch.setenv("GHOST_IMAGINE", "1")
    assert steer_armed() is True
    monkeypatch.delenv("GHOST_IMAGINE")
    assert steer_armed() is False


def test_the_report_says_ARMED_or_DISARMED_beside_the_bucket_count(
        monkeypatch, tmp_path):
    """The world it fails in: the screen reports six discriminating buckets,
    a reader concludes the agent is steering, and every deferral is in fact
    being dropped at the consumer's first line — which is what happened.
    """
    from ghost_agent.core import learning_health as LH
    import json
    gate_dir = tmp_path / "system" / "foresight"
    gate_dir.mkdir(parents=True)
    (gate_dir / "gate.json").write_text(json.dumps({
        "built": "2026-09-04T00:00:00Z", "ledger_rows": 11408,
        "params": {"min_bucket_n": 30}, "enabled_count": 1,
        "buckets": {"file_system|ext:py": {"n": 1896, "enabled": True},
                    "web_search|other": {"n": 2724, "why": "flat:no spread"}},
    }))
    I.reset_gate_cache_for_tests()
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))

    stats = gate_stats(home=str(tmp_path))
    assert stats["enabled_count"] == 1, "fixture no longer opens a bucket"
    assert stats["steer_armed"] is False
    report = LH.render_learning_health(tmp_path / "system" / "memory")
    assert "DISCRIMINATE" in report, report[-1500:]
    assert "DISARMED" in report, "an open gate was reported without saying nothing acts on it"

    monkeypatch.setenv("GHOST_IMAGINE", "1")
    I.reset_gate_cache_for_tests()
    report2 = LH.render_learning_health(tmp_path / "system" / "memory")
    assert "steer ARMED" in report2
    assert "DISARMED" not in report2


def test_a_closed_gate_also_says_whether_the_steer_is_dead(monkeypatch,
                                                           tmp_path):
    """Closed AND disarmed is a different project state from closed-but-armed
    — one is waiting for data, the other is waiting for a flag."""
    from ghost_agent.core import learning_health as LH
    import json
    gate_dir = tmp_path / "system" / "foresight"
    gate_dir.mkdir(parents=True)
    (gate_dir / "gate.json").write_text(json.dumps({
        "built": "2026-09-04T00:00:00Z", "ledger_rows": 500,
        "params": {}, "enabled_count": 0,
        "buckets": {"web_search|other": {"n": 40, "why": "thin:n<30"}},
    }))
    I.reset_gate_cache_for_tests()
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    report = LH.render_learning_health(tmp_path / "system" / "memory")
    assert "CLOSED" in report
    assert "steer also disarmed" in report
