"""Tests for the anonymity / loop-closing quick wins:
  (a) operator log-stream redaction
  (b) reference-count prior in autobiographical recall
  (d) per-identity Tor circuit isolation (SOCKS auth)
"""

import pytest

from ghost_agent.utils import logging as glog
from ghost_agent.utils.helpers import socks_url_with_identity
from ghost_agent.selfhood.autobiographical import AutobiographicalMemory
from ghost_agent.selfhood.schema import Experience


# ──────────────────────────────────────────────────────────────────────
# (a) log-stream redaction
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def restore_redaction():
    yield
    glog.set_log_redaction(True)  # default-on; never leak a False into other tests


def test_log_redacts_email_when_on(capsys):
    glog.set_log_redaction(True)
    glog.pretty_log("Test", "contact alice@example.com about the key")
    out = capsys.readouterr().out
    assert "alice@example.com" not in out
    assert "REDACTED" in out.upper() or "@" not in out


def test_log_shows_raw_when_off(capsys):
    glog.set_log_redaction(False)
    glog.pretty_log("Test", "contact alice@example.com about it")
    out = capsys.readouterr().out
    assert "alice@example.com" in out


def test_log_ordinary_text_untouched(capsys):
    glog.set_log_redaction(True)
    glog.pretty_log("Boot", "initializing components ok")
    out = capsys.readouterr().out
    assert "initializing components ok" in out


# ──────────────────────────────────────────────────────────────────────
# (b) reference-count prior in recall
# ──────────────────────────────────────────────────────────────────────

def test_reference_count_boosts_recall(tmp_path):
    am = AutobiographicalMemory(tmp_path, enabled=True)
    e1 = Experience(trajectory_id="t1", summary="I debugged a tricky kafka consumer lag issue")
    e2 = Experience(trajectory_id="t2", summary="I debugged a tricky kafka producer batching issue")
    am.append(e1)
    am.append(e2)

    # Baseline: both match "kafka debugged"; tie broken by recency (e2 newer).
    base = am.search_my_past("kafka debugged tricky", limit=2)
    assert len(base) == 2

    # Reference e1 repeatedly → it should now outrank the newer e2.
    for _ in range(5):
        am.record_reference(e1.id)
    boosted = am.search_my_past("kafka debugged tricky", limit=2)
    assert boosted[0].id == e1.id


# ──────────────────────────────────────────────────────────────────────
# (d) per-identity Tor circuit isolation
# ──────────────────────────────────────────────────────────────────────

def test_identity_injects_distinct_socks_auth():
    base = "socks5h://127.0.0.1:9050"
    a = socks_url_with_identity(base, "query-aaaa")
    b = socks_url_with_identity(base, "query-bbbb")
    assert "@127.0.0.1:9050" in a
    assert a != b                       # distinct identities → distinct auth
    assert a.startswith("socks5h://")
    assert "127.0.0.1" in a and "9050" in a


def test_identity_noop_cases():
    # falsy proxy / identity → unchanged
    assert socks_url_with_identity(None, "x") is None
    assert socks_url_with_identity("", "x") == ""
    assert socks_url_with_identity("socks5h://127.0.0.1:9050", "") == "socks5h://127.0.0.1:9050"
    # already has credentials → unchanged
    pre = "socks5h://user:pass@127.0.0.1:9050"
    assert socks_url_with_identity(pre, "x") == pre


def test_identity_sanitizes_tag():
    out = socks_url_with_identity("socks5h://127.0.0.1:9050", "weird/tag with spaces!")
    # tag is alnum-sanitised; URL stays well-formed (single @).
    assert out.count("@") == 1
    assert " " not in out
