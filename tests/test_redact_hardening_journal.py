"""Tests for the hardened distill redactor and journal corruption recovery."""

import json
import tempfile
from pathlib import Path

import pytest

from ghost_agent.distill.redact import redact_text, _redact_value
from ghost_agent.memory.journal import MemoryJournal


class TestRedactionCoverage:
    @pytest.mark.parametrize("secret,marker", [
        ("github_pat_11ABCDE0abcdefghij_klmnopQRSTUV", "github_pat_11"),
        ("AIzaSyD-abc123DEF456ghi789JKL012mno345p", "AIzaSy"),
        ("sk_live_abcdEFGH1234567890wxyz", "sk_live_abcd"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJSMeKKF2QT4abc", "SflKxw"),
    ])
    def test_secret_formats_redacted(self, secret, marker):
        assert marker not in redact_text(secret)

    def test_pem_private_key_block(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
        out = redact_text(pem)
        assert "MIIEow" not in out and "REDACTED_PRIVATE_KEY" in out

    def test_conn_uri_password(self):
        out = redact_text("postgres://admin:s3cretPass@db.host:5432/app")
        assert "s3cretPass" not in out

    def test_generic_env_secret(self):
        out = redact_text("GITHUB_TOKEN=ghp_realtokenvalue1234567890")
        assert "ghp_realtoken" not in out

    def test_nested_sensitive_key_list(self):
        out = _redact_value({"credentials": ["alice", "hunter2opaque"]}, redact_text)
        assert out == {"credentials": ["<REDACTED>", "<REDACTED>"]}

    def test_nested_sensitive_key_dict(self):
        out = _redact_value({"authorization": {"value": "opaquetok"}}, redact_text)
        assert out["authorization"]["value"] == "<REDACTED>"


class TestJournalRecovery:
    def test_corrupt_journal_sidecars_and_recovers(self):
        d = Path(tempfile.mkdtemp())
        j = MemoryJournal(d)
        j.file_path.write_text("{ this is not : valid json ]")
        assert j.load() == []                       # graceful
        sidecars = list(d.glob("*.corrupt-*"))
        assert len(sidecars) == 1                   # bytes preserved
        # next write starts a fresh journal, doesn't crash
        j.append("post_mortem", {"tools": [{"result": "ok"}]})
        assert len(j.load()) == 1

    def test_append_redacts_tool_output(self):
        d = Path(tempfile.mkdtemp())
        j = MemoryJournal(d)
        j.append("post_mortem", {"tools": [{"result": "OPENAI_API_KEY=sk-abcdEFGH1234567890abcd"}]})
        raw = j.file_path.read_text()
        assert "sk-abcdEFGH" not in raw
