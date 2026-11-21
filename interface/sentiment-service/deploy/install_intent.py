#!/usr/bin/env python3
"""Install the INTENT model (§4VE) for the sentiment-service from a Hugging Face cache that already holds it, or
roll back to an earlier install.

The model, MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (or the -base- one when memory is short), ships its own
ONNX export under the repository's onnx/ folder, so nothing is exported here and nothing is downloaded: the
standard library only, runnable with any python3 on the service machine. Run it from the directory that holds it,
intent_labels.json and intent_parity.json (the files are read from beside this script). It writes <out>/ with

  model.onnx (+ its external data file, if the export has one), tokenizer.json, config.json (id2label: which
  logit is entailment), intent.json (labels, phrases, template: intent_labels.json), parity.json (the start-up
  check: intent_parity.json) and source.json (repository and revision, for the log line).

Before anything is moved, the labels file, the parity file and the model's config are checked with the SERVICE'S
OWN rules (sentiment_service.py's load_intent_config, load_parity and _entailment_indices; --service names that
file). The new directory is built beside the old one and swapped in; the old one is kept as
<out>.prev-<time>. The newest KEEP previous installs are kept, and the newest one the service ever loaded
successfully (it holds .loaded-ok) is never deleted. Large files are HARD LINKS to the cache's blobs when they are
on the same disk (no second 1.7 GB; deleting the cache does not delete them), copies otherwise.

  install_intent.py [--repo ORG/NAME] [--hf-home DIR] [--snapshot DIR] [--out DIR] [--service FILE]
  install_intent.py --rollback [NAME]     # <out> → <out>.rejected-<time>, the newest (or NAME) <out>.prev-* → <out>
  install_intent.py --list
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
#: previous installs kept besides the newest known-good one
KEEP = 3
#: the service's run-time state, never carried into a restored install
STATE_FILES = (".inuse", ".deaths", ".loading")


def stamp() -> str:
    t = time.time()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1e6) % 1000000:06d}"


def load_service(path):
    """The service module, for its own validation rules (its top level imports the standard library only)."""
    candidates = [Path(path).expanduser()] if path else [
        HERE.parent / "sentiment_service.py", Path("~/Data/AI/sentiment-service/app/sentiment_service.py").expanduser()]
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("sentiment_service_for_install", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "load_parity"):
                raise SystemExit(f"{p} is an older service without the intent head: copy the new one first")
            return mod
    raise SystemExit("sentiment_service.py not found: copy it to app/ first, or pass --service")


def find_snapshot(hf_home: Path, repo: str) -> Path:
    name = "models--" + repo.replace("/", "--")
    for base in (hf_home / "hub" / name, hf_home / name):
        snaps = base / "snapshots"
        if not snaps.is_dir():
            continue
        ref = base / "refs" / "main"
        if ref.is_file() and (snaps / ref.read_text().strip()).is_dir():
            return snaps / ref.read_text().strip()
        found = sorted(p for p in snaps.iterdir() if p.is_dir())
        if len(found) == 1:
            return found[0]
        raise SystemExit(f"{snaps} holds {len(found)} snapshots and no refs/main: pass --snapshot")
    raise SystemExit(f"{repo} is not in {hf_home}")


def first(snapshot: Path, name: str) -> Path:
    for p in (snapshot / "onnx" / name, snapshot / name):
        if p.is_file():
            return p
    raise SystemExit(f"no {name} in {snapshot}/onnx or {snapshot}")


def place(src: Path, dst: Path) -> str:
    real = src.resolve()                        # the cache's snapshot files are links to its blobs
    try:
        os.link(real, dst)
        return "hard link"
    except OSError:
        shutil.copyfile(real, dst)
        return "copy"


def previous(out: Path):
    """The previous installs, newest first."""
    return sorted(out.parent.glob(out.name + ".prev-*"), key=lambda p: p.name, reverse=True)


def prune(out: Path) -> None:
    olds = previous(out)
    good = next((p for p in olds if (p / ".loaded-ok").is_file()), None)
    for p in olds[KEEP:]:
        if p != good:
            shutil.rmtree(p, ignore_errors=True)


def clear_state(d: Path) -> None:
    for f in STATE_FILES:
        try:
            (d / f).unlink()
        except OSError:
            pass


def validate(svc, snap_config: Path) -> None:
    try:
        svc._entailment_indices(json.loads(snap_config.read_text()))
        cfg = svc.load_intent_config(HERE / "intent_labels.json")
        svc.load_parity(HERE / "intent_parity.json", cfg["labels"])
    except svc.IntentError as exc:
        raise SystemExit(f"not installed: {exc}") from None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"not installed: {snap_config} cannot be read ({type(exc).__name__})") from None


def install(args) -> None:
    svc = load_service(args.service)
    snap = Path(args.snapshot).expanduser() if args.snapshot else \
        find_snapshot(Path(args.hf_home).expanduser(), args.repo)
    out = Path(args.out).expanduser()
    model = snap / "onnx" / "model.onnx"
    if not model.is_file():
        raise SystemExit(f"no onnx/model.onnx in {snap} (the fp32 export; the quantized ones are not used)")
    config_path = first(snap, "config.json")
    validate(svc, config_path)

    tmp = out.with_name(out.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    how = place(model, tmp / "model.onnx")
    for extra in ("model.onnx_data", "model.onnx.data"):
        if (snap / "onnx" / extra).is_file():
            place(snap / "onnx" / extra, tmp / extra)
    place(first(snap, "tokenizer.json"), tmp / "tokenizer.json")
    place(config_path, tmp / "config.json")
    shutil.copyfile(HERE / "intent_labels.json", tmp / "intent.json")
    shutil.copyfile(HERE / "intent_parity.json", tmp / "parity.json")
    repo = args.repo if not args.snapshot else snap.parent.parent.name.replace("models--", "").replace("--", "/")
    (tmp / "source.json").write_text(json.dumps({"repo": repo, "revision": snap.name}))

    kept = None
    if out.exists():
        kept = out.with_name(f"{out.name}.prev-{stamp()}")
        out.rename(kept)
        clear_state(kept)
    tmp.rename(out)
    prune(out)
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"installed {repo} ({snap.name[:12]}) into {out}: {size:.0f} MB, model.onnx is a {how}"
          + (" to the cache's blob" if how == "hard link" else ""))
    if kept:
        print(f"the previous install is {kept}; roll back with: {Path(sys.argv[0]).name} --rollback")
    print("check it before restarting the service (a second copy in memory: check free memory first):\n"
          "  ~/Data/AI/sentiment-service/venv/bin/python ~/Data/AI/sentiment-service/app/sentiment_service.py "
          "--check-intent --intent-threads 2,4,0")


def resolve_rollback(out: Path, name):
    """The install to put back, resolved and checked COMPLETELY before anything is moved: NAME (a bare name or a
    path; only its last part counts) must be one of <out>.prev-*, and hold a model.onnx."""
    olds = previous(out)
    if name is True:
        if not olds:
            raise SystemExit(f"nothing to roll back to: no {out.name}.prev-* beside {out}")
        target = olds[0]
    else:
        base = Path(str(name).rstrip("/")).name
        target = out.with_name(base)
        if target not in olds:
            raise SystemExit(f"{base!r} is not a previous install; they are: "
                             + (", ".join(p.name for p in olds) or "none") + f" (in {out.parent})")
    if not (target / "model.onnx").is_file():
        raise SystemExit(f"{target} holds no model.onnx: not rolled back")
    return target


def rollback(args) -> None:
    out = Path(args.out).expanduser()
    target = resolve_rollback(out, args.rollback)          # nothing has moved yet
    rejected = None
    if out.exists():
        rejected = out.with_name(f"{out.name}.rejected-{stamp()}")
        out.rename(rejected)
    try:
        target.rename(out)
    except OSError:
        if rejected is not None:
            rejected.rename(out)                            # put the current one back: never leave no intent/
        raise
    clear_state(out)
    if rejected is not None:
        clear_state(rejected)
        print(f"{out} → {rejected}")
    print(f"{target} → {out}; now: sudo launchctl kickstart -k system/com.local.sentiment-service")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--hf-home", default="~/Data/AI/intent-models/hf")
    ap.add_argument("--snapshot", help="the snapshot directory itself, instead of --repo/--hf-home")
    ap.add_argument("--out", default="~/Data/AI/sentiment-service/intent")
    ap.add_argument("--service", help="the sentiment_service.py whose rules validate the files (default: beside "
                                      "deploy/, else ~/Data/AI/sentiment-service/app/)")
    ap.add_argument("--rollback", nargs="?", const=True, metavar="NAME",
                    help="put back the newest previous install (or NAME, e.g. intent.prev-20261121-...)")
    ap.add_argument("--list", action="store_true", help="list the previous installs")
    args = ap.parse_args()
    if args.list:
        out = Path(args.out).expanduser()
        for p in previous(out):
            print(p.name + ("  (loaded successfully)" if (p / ".loaded-ok").is_file() else ""))
        return
    if args.rollback:
        rollback(args)
    else:
        install(args)


if __name__ == "__main__":
    sys.exit(main())
