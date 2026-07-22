"""Router boot resilience: a corrupt/incompatible checkpoint must never
kill the router permanently.

FIX (2026-07-22): main.lifespan used to call ComplexityClassifier.load
inside ONE broad try whose except nulled BOTH complexity_dispatcher AND
_router_checkpoint_path. A single load failure (e.g. a router schema
bump that makes the existing on-disk checkpoint raise at load) therefore
killed the router AND every retrain path — the bootstrap-train below the
load was never reached, and the idle/self-play retrains bail on the None
dispatcher/path — a dead router on EVERY boot until the file was
manually deleted. model.py's load docstring always promised the opposite
("the boot loader falls back to the safe escalate-all dispatcher and
retrains"); this file pins that promise so it stays true.

Now: the load call has its OWN try/except — a failure falls back to
clf=None WITHOUT nulling _router_checkpoint_path, so the bootstrap-train
runs and OVERWRITES the bad checkpoint from the trajectory log.

Two layers of tests (the boot block lives inline in main.lifespan and is
not importable — same technique as test_narrative_nothink_wiring.py and
test_main_project_store_wiring.py):

- functional: model.load raises a clean, catchable ValueError for every
  malformed checkpoint shape, and the fixed boot control flow
  (load-fail -> bootstrap -> overwrite) self-heals a bad file end to end.
- source pins on main.py: the load is wrapped in its own try, the
  fallback branch never nulls the checkpoint path, and the idle-loop LLM
  closures are background-marked.
"""

import json
import re
import inspect
from pathlib import Path

import pytest

import ghost_agent.main as ghost_main
from ghost_agent.router import (
    ComplexityClassifier,
    ComplexityDispatcher,
    bootstrap_router,
)
from ghost_agent.router.features import extract_features, FEATURE_NAMES

MAIN_SRC = Path(inspect.getsourcefile(ghost_main)).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _trained_classifier() -> ComplexityClassifier:
    """Deterministic toy fit — same corpus shape as test_router_model."""
    easy = ["hi there", "thanks", "what's the time?", "hello how are you", "ok"]
    hard = [
        "write a python script that parses the access.log and counts 4xx errors",
        "refactor the sql query to use a window function then benchmark it",
        "implement a recursive graph traversal with memoization and bfs",
        "debug the tcp socket timeout in the async handler and patch it",
        "scrape the site with playwright and cross-reference stored json",
    ]
    X = [extract_features(t) for t in easy + hard]
    y = (["easy"] * len(easy)) + (["hard"] * len(hard))
    return ComplexityClassifier(epochs=300, random_state=42).fit(X, y)


def _good_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "router" / "checkpoint.json"
    _trained_classifier().save(path)
    return path


def _traj(req, steps, calls, outcome="passed", heavy=False):
    from ghost_agent.distill.schema import Trajectory, ToolCall
    tcs = [ToolCall(name=("execute" if heavy else "web_search")) for _ in range(calls)]
    return Trajectory(user_request=req, n_steps=steps, tool_calls=tcs, outcome=outcome)


def _balanced_corpus(n_each):
    easy = [_traj(f"what is {i}?", 1, 1) for i in range(n_each)]
    hard = [_traj(f"build deploy {i}", 6, 5, outcome="failed", heavy=True) for i in range(n_each)]
    return easy + hard


def _boot_router(ckpt_path: Path, trajectories):
    """Mirror of main.lifespan's FIXED router boot control flow.

    The block is inline in lifespan (not importable); this reproduces its
    exact decision sequence so the behavioural contract — bad checkpoint
    -> escalate-all + bootstrap retrain over the same path — is pinned
    functionally. The source pins below tie main.py to this flow.
    """
    checkpoint_path = ckpt_path  # set BEFORE the load; never nulled on load failure
    clf = None
    if ckpt_path.exists():
        try:
            clf = ComplexityClassifier.load(ckpt_path)
        except Exception:
            clf = None  # fall back; checkpoint_path stays intact
    if clf is None:
        boot_clf, _report = bootstrap_router(trajectories, save_path=ckpt_path)
        if boot_clf is not None:
            clf = boot_clf
    dispatcher = ComplexityDispatcher(classifier=clf, disabled=(clf is None))
    return dispatcher, checkpoint_path, clf


# ---------------------------------------------------------------------------
# 1. model.load raises a clean, catchable ValueError for every bad shape
# ---------------------------------------------------------------------------

class TestLoadRaisesCleanly:
    def test_unknown_schema(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        raw["schema"] = "ghost.router.logreg.v999"  # simulated schema bump
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="unknown model schema"):
            ComplexityClassifier.load(path)

    def test_corrupt_json(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        path.write_text("{ this is not json !!!")
        # json.JSONDecodeError is a ValueError subclass — still one clean
        # catchable failure mode.
        with pytest.raises(ValueError):
            ComplexityClassifier.load(path)

    def test_missing_weights_key(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        del raw["weights"]
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="malformed"):
            ComplexityClassifier.load(path)

    def test_non_numeric_weights(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        raw["weights"] = ["not", "numbers"] + raw["weights"][2:]
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ComplexityClassifier.load(path)

    def test_feature_schema_reordered(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        raw["feature_names"] = list(reversed(raw["feature_names"]))
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="different feature"):
            ComplexityClassifier.load(path)

    def test_feature_schema_shorter(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        raw["feature_names"] = raw["feature_names"][:-1]
        raw["weights"] = raw["weights"][:-1]
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="different feature"):
            ComplexityClassifier.load(path)

    def test_non_finite_weights(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        raw["weights"][0] = 1e400  # json emits Infinity-ish via float parse
        path.write_text(json.dumps(raw).replace("1e+400", "Infinity"))
        with pytest.raises(ValueError, match="non-finite"):
            ComplexityClassifier.load(path)

    def test_incompatible_training_report(self, tmp_path):
        path = _good_checkpoint(tmp_path)
        raw = json.loads(path.read_text())
        raw["report"] = {"totally_unknown_field": 1}
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="incompatible training report"):
            ComplexityClassifier.load(path)

    def test_good_checkpoint_still_loads(self, tmp_path):
        """The hardening must not reject a healthy checkpoint."""
        path = _good_checkpoint(tmp_path)
        clf = ComplexityClassifier.load(path)
        assert clf.weights_ is not None
        assert clf.feature_names_ == tuple(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# 2. Boot control flow self-heals: bad checkpoint -> retrain -> overwrite
# ---------------------------------------------------------------------------

class TestBootSelfHeals:
    def test_bad_checkpoint_bootstraps_and_overwrites(self, tmp_path):
        """The core regression: a checkpoint that raises at load must yield
        a live dispatcher AND a retrain that overwrites the bad file."""
        ckpt = tmp_path / "router" / "checkpoint.json"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text(json.dumps({"schema": "ghost.router.logreg.v999"}))

        dispatcher, kept_path, clf = _boot_router(ckpt, _balanced_corpus(30))

        assert kept_path == ckpt  # retrain paths still know where to save
        assert dispatcher is not None  # router alive, not dead
        assert clf is not None  # bootstrap-train actually produced a model
        # The bad file was OVERWRITTEN with a valid checkpoint: a second
        # boot now loads clean instead of failing forever.
        reloaded = ComplexityClassifier.load(ckpt)
        assert reloaded.feature_names_ == tuple(FEATURE_NAMES)
        # And the healed router actually routes (bootstrap-test contract).
        assert dispatcher.route("what is 7?").escalated is False

    def test_bad_checkpoint_no_data_still_retrain_capable(self, tmp_path):
        """Load fails AND the trajectory log is too thin to bootstrap:
        the dispatcher must still boot (escalate-all pass-through) with
        the checkpoint path preserved, so a LATER idle retrain can save
        over the bad file — never a dead router + dead retrain."""
        ckpt = tmp_path / "router" / "checkpoint.json"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("{ corrupt")

        dispatcher, kept_path, clf = _boot_router(ckpt, _balanced_corpus(2))

        assert dispatcher is not None
        assert clf is None
        assert dispatcher.route("hi").escalated is True  # safe fallback
        assert kept_path == ckpt  # NOT nulled — this is the fix
        # Simulate the idle retrain the preserved path enables: save over
        # the corrupt file and confirm the next load succeeds.
        _trained_classifier().save(kept_path)
        healed = ComplexityClassifier.load(kept_path)
        assert healed.weights_ is not None

    def test_missing_checkpoint_unchanged_behaviour(self, tmp_path):
        """No file at all — the pre-fix happy path must be untouched."""
        ckpt = tmp_path / "router" / "checkpoint.json"
        dispatcher, kept_path, clf = _boot_router(ckpt, _balanced_corpus(30))
        assert dispatcher is not None
        assert clf is not None
        assert ckpt.exists()  # bootstrap persisted it


# ---------------------------------------------------------------------------
# 3. Source pins on main.py (boot block is inline in lifespan)
# ---------------------------------------------------------------------------

def _router_boot_region() -> str:
    start = MAIN_SRC.index("context._router_checkpoint_path = router_ckpt_path")
    end = MAIN_SRC.index("Selfhood module")
    return MAIN_SRC[start:end]


class TestMainSourcePins:
    def test_load_wrapped_in_own_try(self):
        region = _router_boot_region()
        assert re.search(
            r"try:\s*\n\s*clf = ComplexityClassifier\.load\(", region
        ), "ComplexityClassifier.load must be inside its OWN try block"

    def test_load_fallback_does_not_null_checkpoint_path(self):
        """The inner except (between the load and the --router-model elif)
        must reset clf only — nulling _router_checkpoint_path there is the
        exact bug that killed every retrain path."""
        region = _router_boot_region()
        load_idx = region.index("clf = ComplexityClassifier.load(")
        elif_idx = region.index("elif args.router_model:")
        inner = region[load_idx:elif_idx]
        assert "except" in inner, "load must have a dedicated except"
        assert "clf = None" in inner
        assert "_router_checkpoint_path = None" not in inner
        assert "complexity_dispatcher = None" not in inner

    def test_failed_load_flows_into_bootstrap_train(self):
        """After the fallback, `if clf is None:` bootstrap must still run
        with save_path=router_ckpt_path so the bad file gets overwritten."""
        region = _router_boot_region()
        elif_idx = region.index("elif args.router_model:")
        after = region[elif_idx:]
        # the actual CALL, not the comment that merely mentions the name
        boot_idx = after.index("= bootstrap_router(")
        assert "if clf is None:" in after[:boot_idx]
        assert "save_path=router_ckpt_path" in after

    def test_outer_except_still_guards_the_rest(self):
        """The broad outer except stays (boot must survive anything) — only
        the load failure no longer reaches it."""
        region = _router_boot_region()
        assert "context.complexity_dispatcher = None" in region
        assert "context._router_checkpoint_path = None" in region


# ---------------------------------------------------------------------------
# 4. Idle-loop LLM closures are background-marked (FIX 2 pins)
# ---------------------------------------------------------------------------

def _closure_src(name: str) -> str:
    """Exact closure body: from its `async def` line to the first
    non-empty line at the same or lower indentation (docstrings contain
    blank lines, so a naive double-newline cut truncates early)."""
    lines = MAIN_SRC.splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.lstrip().startswith(f"async def {name}(")
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)
    return "\n".join(body)


class TestIdleClosuresAreBackground:
    """These closures run in biological idle phases, never on the user's
    synchronous path. Foreground-marked they bump foreground_tasks
    (misleading every "is a user live?" check + the watchdog) and contend
    for the main slot with a live turn."""

    def test_selfhood_critique_is_background(self):
        body = _closure_src("_selfhood_critique_fn")
        assert "chat_completion(payload, is_background=True)" in body

    def test_workspace_critique_is_background(self):
        body = _closure_src("_workspace_critique_fn")
        assert "chat_completion(payload, is_background=True)" in body

    def test_postmortem_analyze_is_background(self):
        body = _closure_src("_analyze_fn")
        assert "chat_completion(payload, is_background=True)" in body

    def test_postmortem_patch_is_background(self):
        body = _closure_src("_patch_fn")
        assert "chat_completion(payload, is_background=True)" in body


# ---------------------------------------------------------------------------
# 5. Import smoke — both edited modules import clean
# ---------------------------------------------------------------------------

def test_edited_modules_import():
    import ghost_agent.main  # noqa: F401 (already imported above; explicit pin)
    import ghost_agent.router.model  # noqa: F401
