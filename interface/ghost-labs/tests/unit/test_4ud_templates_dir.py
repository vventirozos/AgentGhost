"""§4UD-c: the service reads the SHIPPED templates from the code's own tree, so syncing the code deploys them.

The defect: the service read /etc/ghost-labs/templates, a copy that only install.sh refreshed. A template fix
(PGUSER=postgres, member ready checks) synced to /opt was not used for hours; `psql` in a pg lab ran as root.
Each test fails in that world."""
import sys
import types
from pathlib import Path

import pytest

from ghost_labs import config as C
from ghost_labs import templates as T

from .test_core import ROOT


def _stale_copy(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    # what /etc held: pg-single as it was before the fix — no env, no ready check
    (d / "pg-single.yaml").write_text(
        "name: pg-single\negress: none\nmembers:\n  - {name: pg1, image: \"ghost-lab/pg:18\", mem_mb: 768}\n")
    (d / "mine.yaml").write_text(
        "name: mine\negress: none\nmembers:\n  - {name: box, image: \"ghost-lab/base:1\", mem_mb: 256}\n")


def test_shipped_dir_is_the_code_tree():
    assert T.SHIPPED_DIR == (ROOT / "templates").resolve()
    assert "pg-single" in T.load_dir(T.SHIPPED_DIR)


def test_a_shipped_name_wins_over_an_etc_copy(tmp_path):
    _stale_copy(tmp_path / "etc")
    got, shadowed = T.load_all(T.SHIPPED_DIR, tmp_path / "etc")
    assert got["pg-single"]["env"].get("PGUSER") == "postgres"
    assert got["pg-single"]["members"][0]["ready"]
    assert len(shadowed) == 1 and shadowed[0].startswith("pg-single.yaml:") and "shipped" in shadowed[0]
    assert "mine" in got                                  # the operator's own new name is added


def test_no_extra_dir_is_fine(tmp_path):
    got, shadowed = T.load_all(T.SHIPPED_DIR, tmp_path / "absent")
    assert shadowed == [] and "pg-single" in got


def test_server_builds_the_service_from_the_shipped_dir(monkeypatch, tmp_path):
    """server.main hands Service the shipped dir and /etc as the EXTRA dir (not as the main one)."""
    seen = {}

    class Stop(RuntimeError):
        pass

    def fake_service(cfg, client, templates_dir, state_dir, **kw):
        seen["templates_dir"], seen["extra"] = Path(templates_dir), kw.get("extra_templates_dir")
        raise Stop("stop here")

    import ghost_labs.service as S
    from ghost_labs import server
    monkeypatch.setattr(S, "Service", fake_service)
    monkeypatch.setattr(C, "load", lambda: types.SimpleNamespace())
    monkeypatch.setattr(C, "resolve_bind", lambda cfg: None)
    monkeypatch.setitem(sys.modules, "docker", types.SimpleNamespace(from_env=lambda **k: object()))
    assert server.main() == 2
    assert seen["templates_dir"].resolve() == T.SHIPPED_DIR
    assert Path(seen["extra"]) == C.CONFIG_DIR / "templates"


def test_service_logs_a_shadowed_etc_file(tmp_path, caplog):
    from ghost_labs.service import Service
    from .fakes import FakeDocker
    _stale_copy(tmp_path / "etc")
    with caplog.at_level("WARNING", logger="ghost_labs"):
        svc = Service(C.Config(), FakeDocker(), T.SHIPPED_DIR, tmp_path / "state", start_threads=False,
                      extra_templates_dir=tmp_path / "etc")
    assert svc.templates["pg-single"]["env"].get("PGUSER") == "postgres"
    assert "mine" in svc.templates
    assert any("pg-single" in r.getMessage() and "NOT used" in r.getMessage() for r in caplog.records)


def test_a_broken_etc_file_is_logged_and_never_stops_the_service(tmp_path, caplog):
    """r3 m3: a stale /etc copy in an old format raised out of Service() and the service crash-looped."""
    from ghost_labs.service import Service
    from .fakes import FakeDocker
    d = tmp_path / "etc"
    _stale_copy(d)
    (d / "old.yaml").write_text("name: old\negress: none\nmembers: []\n")          # no members: does not parse
    (d / "junk.yaml").write_text("name: [unclosed\n")                              # not YAML
    with caplog.at_level("WARNING", logger="ghost_labs"):
        svc = Service(C.Config(), FakeDocker(), T.SHIPPED_DIR, tmp_path / "state", start_threads=False,
                      extra_templates_dir=d)
    assert "mine" in svc.templates and "old" not in svc.templates
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "old.yaml: does not load" in msgs and "junk.yaml: does not load" in msgs


def test_a_broken_shipped_template_still_fails_loudly(tmp_path):
    from ghost_labs.errors import LabError
    (tmp_path / "bad.yaml").write_text("name: bad\negress: none\nmembers: []\n")
    with pytest.raises(LabError):
        T.load_all(tmp_path, None)



@pytest.mark.parametrize("body,err", [
    ("name: inf\negress: none\nmembers:\n  - {name: m, image: x, mem_mb: .inf}\n", "OverflowError"),
    ("[" * 5000 + "\n", "RecursionError"),
])
def test_r4_any_failure_of_one_etc_file_is_a_note(tmp_path, body, err):
    """r4 MAJOR: OverflowError (`.inf`) and RecursionError (deep nesting) escaped a narrower catch list."""
    d = tmp_path / "etc"
    _stale_copy(d)
    (d / "bad.yaml").write_text(body)
    got, notes = T.load_all(T.SHIPPED_DIR, d)
    assert "mine" in got and any(n.startswith("bad.yaml: does not load") and err in n for n in notes)


def test_r4_two_etc_files_with_one_name_are_said_as_such(tmp_path):
    d = tmp_path / "etc"
    d.mkdir()
    for f in ("a.yaml", "b.yaml"):
        (d / f).write_text("name: mine\negress: none\nmembers:\n  - {name: box, image: \"ghost-lab/base:1\", "
                           f"mem_mb: {256 if f == 'a.yaml' else 512}}}\n")
    got, notes = T.load_all(T.SHIPPED_DIR, d)
    assert got["mine"]["members"][0]["mem_mb"] == 256                    # the first file (sorted) gives it
    assert notes == ["b.yaml: a.yaml already gives a template named mine"]
