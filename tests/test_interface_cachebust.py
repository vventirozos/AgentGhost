"""Every shipped change to a static module must move its `?v=`.

⚠ WHY THIS FILE EXISTS (§4BU R2, lens B). The round-1 console fixes were
written at 20:37; `index.html` and `workspace.js` — the files that CARRY the
`?v=` values naming those modules — were last written at 20:14. So the
version strings 9.7 / 7.3 / 6.8 labelled two different contents, and a PWA
that had loaded the console in between would keep serving the pre-fix code
from cache while every test in the repo asserted the fixes were in.
`/static` is a bare `StaticFiles` mount with no `Cache-Control`; the `?v=`
query is the only deliberate cache control this project has.

There is a structural half too: bumping `sessions.js?v=` inside
`workspace.js` is INERT unless `workspace.js?v=` in app.js moves as well —
a cached `workspace.js` keeps importing the old submodule URL. The chain
must be bumped root-down:

    index.html (app.js) -> app.js (workspace.js, matrix_graph.js)
                        -> workspace.js (sessions, status, notifications, palette)

Only the `index -> app.js == matrix_graph` link was pinned before this file;
the whole workspace subtree had zero version assertions, and the two other
version tests are `>=` FLOORS, which a stale-but-high number satisfies
forever.

## How this test works, and what to do when it fails

`_MANIFEST` records, per module, the version it shipped under and a hash of
the content that shipped under it. Editing a module without bumping its
version fails here — deliberately. To fix a failure:

  1. bump that module's `?v=` **and every ancestor's** (see the chain above);
  2. re-run with `GHOST_UPDATE_CACHEBUST=1` to rewrite `_MANIFEST`;
  3. commit the manifest change with the code change.

Step 2 is the escape hatch that keeps this honest rather than annoying: the
manifest is data, not a second source of truth.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_MANIFEST_PATH = Path(__file__).with_name("cachebust_manifest.json")

# The import chain, child -> the file that names its version.
_CARRIER = {
    "app.js": "index.html",
    "matrix_graph.js": "app.js",
    "workspace.js": "app.js",
    "sessions.js": "workspace.js",
    "status.js": "workspace.js",
    "notifications.js": "workspace.js",
    "palette.js": "workspace.js",
}
# child -> every ancestor that must move with it (root-down bumping).
_ANCESTORS = {
    "app.js": [],
    "matrix_graph.js": ["app.js"],
    "workspace.js": ["app.js"],
    "sessions.js": ["workspace.js", "app.js"],
    "status.js": ["workspace.js", "app.js"],
    "notifications.js": ["workspace.js", "app.js"],
    "palette.js": ["workspace.js", "app.js"],
}


def _declared_version(module: str) -> str:
    carrier = (_STATIC / _CARRIER[module]).read_text(encoding="utf-8")
    m = re.search(re.escape(module) + r"\?v=([\d.]+)", carrier)
    assert m, f"{module} has no ?v= in {_CARRIER[module]} — it is uncacheable"
    return m.group(1)


def _content_hash(module: str) -> str:
    return hashlib.sha256(
        (_STATIC / module).read_bytes()).hexdigest()[:16]


def _current() -> dict:
    return {mod: {"version": _declared_version(mod), "sha": _content_hash(mod)}
            for mod in _CARRIER}


def _load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_every_changed_module_moved_its_version():
    cur = _current()
    if os.environ.get("GHOST_UPDATE_CACHEBUST") == "1":
        _MANIFEST_PATH.write_text(json.dumps(cur, indent=2, sort_keys=True)
                                  + "\n", encoding="utf-8")
        pytest.skip("manifest rewritten (GHOST_UPDATE_CACHEBUST=1)")
    old = _load_manifest()
    stale = []
    for mod, now in cur.items():
        was = old.get(mod)
        if was is None:
            stale.append(f"{mod}: new module, not in the manifest")
        elif now["sha"] != was["sha"] and now["version"] == was["version"]:
            stale.append(
                f"{mod}: content changed but ?v= is still {was['version']} — "
                f"a browser holding the old module keeps the old code")
    assert not stale, (
        "cache-bust drift:\n  " + "\n  ".join(stale)
        + "\n\nBump the version AND every ancestor (root-down), then re-run "
          "with GHOST_UPDATE_CACHEBUST=1 to record it.")


def test_a_bumped_child_bumped_its_ancestors_too():
    """A cached ancestor keeps importing the OLD child URL, so a leaf-only
    bump ships nothing."""
    cur, old = _current(), _load_manifest()
    moved = {m for m in cur
             if m in old and cur[m]["version"] != old[m]["version"]}
    if not moved:
        pytest.skip("no version moved since the manifest was recorded")
    problems = []
    for mod in moved:
        for anc in _ANCESTORS[mod]:
            if anc in old and cur[anc]["version"] == old[anc]["version"]:
                problems.append(
                    f"{mod} bumped to {cur[mod]['version']} but its carrier "
                    f"chain link {anc} is still {cur[anc]['version']} — a "
                    f"cached {anc} keeps importing the old {mod} URL")
    assert not problems, "\n  ".join(problems)


def test_the_chain_is_internally_consistent():
    """Every module the chain names must exist and be reachable."""
    for mod, carrier in _CARRIER.items():
        assert (_STATIC / mod).exists(), f"{mod} is referenced but missing"
        assert (_STATIC / carrier).exists(), f"{carrier} is missing"
        _declared_version(mod)   # raises if the ?v= is gone


def test_app_and_matrix_graph_move_together():
    """Pre-existing invariant (test_interface_face_palette), restated here
    so the whole chain lives in one file."""
    assert _declared_version("app.js") == _declared_version("matrix_graph.js")
