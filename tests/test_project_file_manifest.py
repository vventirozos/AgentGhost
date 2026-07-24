"""File manifest (2026-07-24) — per-file "what it is/does" for projects.

The gap it closes: deliverables were bare paths and the design ledger free
prose, so a resumed session (or a small model on a big project) had nothing
directing it to the right files and re-derived the layout by re-reading
everything. The manifest stores {rel_path: {desc, role, ts}} in project
metadata (ledger/config idiom: bounded, atomic across processes) and renders
a greppable PROJECT_MAP.md into the workspace.
"""
import json

import pytest

from ghost_agent.memory.projects import ProjectStore


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


def _project(store):
    return store.create_project("App", kind="CODING", goal="Ship it")


# ------------------------------------------------------------- describe_file

def test_describe_file_upserts_and_normalizes(store):
    pid = _project(store)
    assert store.describe_file(pid, f"/workspace/projects/{pid}/server.js",
                               "Node service, port 8100", role="entrypoint")
    mf = store.get_file_manifest(pid)
    assert "server.js" in mf  # normalized to project-relative
    assert mf["server.js"]["desc"] == "Node service, port 8100"
    assert mf["server.js"]["role"] == "entrypoint"
    # Re-describe replaces.
    assert store.describe_file(pid, "server.js", "Express API + static host")
    assert store.get_file_manifest(pid)["server.js"]["desc"] == \
        "Express API + static host"


def test_describe_file_rejects_blank_and_traversal(store):
    pid = _project(store)
    assert not store.describe_file(pid, "../../etc/passwd", "nope")
    assert not store.describe_file(pid, "a.js", "   ")
    assert store.get_file_manifest(pid) == {}


def test_manifest_bounds_desc_and_evicts_oldest(store):
    pid = _project(store)
    long = "x" * 500
    store.describe_file(pid, "a.js", long)
    assert len(store.get_file_manifest(pid)["a.js"]["desc"]) == \
        ProjectStore.MANIFEST_DESC_MAX
    # Eviction: cap + 5 entries → oldest dropped, newest kept.
    for i in range(ProjectStore.MANIFEST_MAX_FILES + 5):
        store.describe_file(pid, f"f{i:03d}.js", f"file {i}")
    mf = store.get_file_manifest(pid)
    assert len(mf) == ProjectStore.MANIFEST_MAX_FILES
    assert f"f{ProjectStore.MANIFEST_MAX_FILES + 4:03d}.js" in mf  # newest kept


# ------------------------------------- register_file_artifact description

def test_register_file_artifact_feeds_manifest(store):
    pid = _project(store)
    tid = store.add_task(pid, "build server")
    aid = store.register_file_artifact(tid, "server.js",
                                       description="Node service")
    assert aid
    assert store.get_file_manifest(pid)["server.js"]["desc"] == "Node service"
    # Without a description, registration still works and manifest untouched.
    aid2 = store.register_file_artifact(tid, "index.html")
    assert aid2
    assert "index.html" not in store.get_file_manifest(pid)


# ------------------------------------------------------------ PROJECT_MAP.md

def test_project_map_rendered_with_described_and_undescribed(store):
    pid = _project(store)
    tid = store.add_task(pid, "build")
    store.register_file_artifact(tid, "index.html")  # undescribed deliverable
    store.describe_file(pid, "server.js", "Node service, port 8100")
    path = store.render_project_map(pid)
    assert path and path.endswith("PROJECT_MAP.md")
    text = open(path, encoding="utf-8").read()
    assert "`server.js` — Node service, port 8100" in text
    assert "`index.html`" in text
    assert "no description yet" in text  # undescribed marked, not hidden


def test_describe_file_rerenders_map(store, tmp_path):
    pid = _project(store)
    store.describe_file(pid, "app.css", "theme + layout styles")
    ws = store.get_project(pid)["workspace_dir"]
    text = open(f"{ws}/PROJECT_MAP.md", encoding="utf-8").read()
    assert "app.css" in text and "theme + layout styles" in text


def test_manifest_survives_atomic_metadata_roundtrip(store):
    """Manifest lives in metadata_json next to design_ledger/config — writing
    one must not clobber the others."""
    pid = _project(store)
    store.append_ledger(pid, "ModuleRegistry lazy-loads modules")
    store.describe_file(pid, "registry.js", "module loader")
    store.set_config_value(pid, "port", "8100")
    assert store.get_ledger(pid) == "ModuleRegistry lazy-loads modules"
    assert store.get_file_manifest(pid)["registry.js"]["desc"] == "module loader"
    meta = store.get_project(pid)["metadata"]
    assert meta["config"]["port"] == "8100"
