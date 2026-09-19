"""§4IN/§4IP — the ledger's first reader: `scripts/claim_binding_ledger_report.py`.

World where each pin fails: an override is not listed (the daily read
misses the row it exists for); a binder failure is counted as a verdict;
the window filter drops rows inside it or keeps rows outside it; the
report raises on a malformed line.
"""
import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "claim_binding_ledger_report", Path(__file__).resolve().parents[1] / "scripts" / "claim_binding_ledger_report.py")
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)

NOW = dt.datetime(2026, 9, 18, 12, 0, tzinfo=dt.timezone.utc)


def _row(ts, inc, cb, decided="incumbent", agree=None, error=None, issues=()):
    return {"ts": ts, "trace": {"req_id": "r"}, "incumbent": {"verdict": inc},
            "claim_binding": {"verdict": cb, "reasoning": "why", "issues": list(issues), **({"error": error} if error else {})},
            "agree": agree, "decided": decided}


def test_summary_counts_overrides_failures_and_agreement():
    rows = [_row("2026-09-18T11:00:00+00:00", "CONFIRMED", "REFUTED", decided="claim_binding", agree=False, issues=["figure '26' … the evidence says '29'"]),
            _row("2026-09-18T11:01:00+00:00", "CONFIRMED", "CONFIRMED", agree=True),
            _row("2026-09-18T11:02:00+00:00", "CONFIRMED", "UNCERTAIN", agree=False),
            _row("2026-09-18T11:03:00+00:00", "CONFIRMED", None, error="TimeoutError")]
    s = R.summarize(rows)
    c = s["counts"]
    assert (c["rows"], c["decided_by_binder"], c["binder_failed"], c["agree"], c["disagree"]) == (4, 1, 1, 1, 2)
    text = R.render(s, [{"outcome": "claim_binding"}, {"outcome": "overturned"}])
    assert "OVERRIDES" in text and "the evidence says '29'" in text
    assert "BINDER FAILURES (1)" in text and "TimeoutError" in text
    assert "incumbent CONFIRMED / binder UNCERTAIN ×1" in text
    assert "claim_binding 1, overturned 1" in text


def test_window_filter_and_malformed_lines(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join([
        json.dumps(_row("2026-09-18T11:00:00+00:00", "CONFIRMED", "CONFIRMED", agree=True)),
        json.dumps(_row("2026-09-01T11:00:00+00:00", "CONFIRMED", "CONFIRMED", agree=True)),
        "{not json",
        "",
    ]))
    since = NOW - dt.timedelta(days=7)
    rows = R.load_rows(p, since)
    assert len(rows) == 1 and rows[0]["ts"].startswith("2026-09-18")
    assert len(R.load_rows(p, None)) == 2
    assert R.load_rows(tmp_path / "missing.jsonl", None) == []
    # a rotated predecessor is read too; non-dict JSON rows are skipped, not fatal
    (tmp_path / "ledger.jsonl.1").write_text(json.dumps(_row("2026-09-17T11:00:00+00:00", "CONFIRMED", "CONFIRMED", agree=True)) + "\n[1, 2]\n\"str\"\nnull\n")
    assert len(R.load_rows(p, None)) == 3


def test_main_runs_against_ghost_home(tmp_path, monkeypatch, capsys):
    (tmp_path / "system" / "verifier").mkdir(parents=True)
    (tmp_path / "system" / "verifier" / "claim_binding_shadow.jsonl").write_text(
        json.dumps(_row(dt.datetime.now(dt.timezone.utc).isoformat(), "REFUTED", "CONFIRMED", decided="claim_binding", agree=False)) + "\n")
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert R.main([]) == 0
    out = capsys.readouterr().out
    assert "DECIDED BY BINDER 1" in out and "OVERRIDES" in out


def test_escalation_audit_flips_population_sees_binder_lifts(tmp_path, monkeypatch, capsys):
    """Review §4IP consumer m6: `--outcome overturned` (the fabrication-
    laundering watch) never saw a binder REFUTED→CONFIRMED lift, which files
    as kind=confirm/outcome=claim_binding. `--outcome flips` selects on the
    verdict change itself."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("escalation_audit", Path(__file__).resolve().parents[1] / "scripts" / "escalation_audit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ledger = tmp_path / "escalations.jsonl"
    rows = [{"ts": "2026-09-18T10:00:00Z", "kind": "confirm", "route": "claim", "outcome": "claim_binding",
             "cheap_verdict": "UNCERTAIN", "strong_verdict": "CONFIRMED"},
            {"ts": "2026-09-18T10:01:00Z", "kind": "refute", "route": "claim", "outcome": "upheld",
             "cheap_verdict": "REFUTED", "strong_verdict": "REFUTED"}]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rows.append({"ts": "2026-09-18T10:02:00Z", "kind": "confirm", "route": "claim", "outcome": "withheld",
                 "cheap_verdict": "CONFIRMED", "strong_verdict": "REFUTED"})       # ships the cheap CONFIRMED capped: NOT a flip
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["escalation_audit", "--ledger", str(ledger), "--trajectories", str(tmp_path), "--outcome", "flips"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc in (0, None) and "1 card(s) from 3 ledger row(s)" in out


def test_rows_without_a_parseable_ts_are_outside_every_window_but_listed_by_all(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps(_row("2026-09-18T11:00:00+00:00", "CONFIRMED", "CONFIRMED", agree=True)) + "\n"
                 + json.dumps({"incumbent": {"verdict": "CONFIRMED"}, "claim_binding": {"verdict": "UNCERTAIN"}, "decided": "incumbent"}) + "\n"
                 + json.dumps(dict(_row("garbage", "CONFIRMED", "CONFIRMED", agree=True))) + "\n")
    assert len(R.load_rows(p, NOW - dt.timedelta(days=1))) == 1
    assert len(R.load_rows(p, None)) == 3


def test_home_default_is_the_project_data_root(monkeypatch):
    monkeypatch.delenv("GHOST_HOME", raising=False)
    assert R._home() == Path.home() / "Data" / "AI" / "Data"
    monkeypatch.setenv("GHOST_HOME", "/tmp/x")
    assert R._home() == Path("/tmp/x")


def test_escalation_audit_flips_with_nothing_to_show_is_an_empty_audit_not_an_error(tmp_path, monkeypatch, capsys):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("escalation_audit2", Path(__file__).resolve().parents[1] / "scripts" / "escalation_audit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ledger = tmp_path / "escalations.jsonl"
    ledger.write_text(json.dumps({"ts": "2026-09-18T10:01:00Z", "kind": "refute", "route": "claim", "outcome": "upheld",
                                  "cheap_verdict": "REFUTED", "strong_verdict": "REFUTED"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["escalation_audit", "--ledger", str(ledger), "--trajectories", str(tmp_path), "--outcome", "flips"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc in (0, None) and "0 card(s) from 1 ledger row(s)" in out


def test_learning_health_renders_the_binder_bucket(tmp_path, monkeypatch):
    """Review §4IP R7 instruments: the bucket was pinned, the render line was
    not — a counted bucket the report never prints is the same silence."""
    from ghost_agent.core import learning_health as LH
    ledger = tmp_path / "escalations.jsonl"
    ledger.write_text(json.dumps({"route": "claim", "kind": "refute", "outcome": "claim_binding", "ts": "2026-09-18T10:00:00+00:00"}) + "\n")
    health = LH._escalation_health(ledger)
    monkeypatch.setattr(LH, "collect_learning_health", lambda *a, **k: {"verifier_escalation": health})
    out = LH.render_learning_health(tmp_path)
    block = out[out.index("VERIFIER ESCALATION"):]
    assert "1 claim-binding" in block and "OTHER" not in block


def test_report_lists_capped_rows(tmp_path):
    p = tmp_path / "ledger.jsonl"
    row = _row("2026-09-19T09:00:00+00:00", "CONFIRMED", "UNCERTAIN", agree=False)
    row["capped"] = ["Dr. Elin Vasquez"]
    p.write_text(json.dumps(row) + "\n" + json.dumps(_row("2026-09-19T09:01:00+00:00", "CONFIRMED", "CONFIRMED", agree=True)) + "\n")
    summary = R.summarize(R.load_rows(p, None))
    assert summary["counts"]["capped"] == 1 and summary["capped"][0]["capped"] == ["Dr. Elin Vasquez"]
    out = R.render(summary, [])
    assert "CAPPED 1" in out and "Dr. Elin Vasquez" in out

