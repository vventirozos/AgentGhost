#!/usr/bin/env python3
"""Sync the stored `youtube_transcribe` macro to the code-owned definition.
Idempotent; safe to re-run.

§4KE (2026-09-24): the macro is now ONE step — `knowledge_base(action='transcribe',
filename='$url')` — and the registry reconciles the stored copy from
`tools/yt_download.CODE_OWNED_MACROS` at every load, so this script is no longer
part of a deploy. It remains useful for a store the agent is NOT holding open
(a migration, a copied GHOST_HOME).

If the agent IS running, stop it first, then start it:
    the live agent caches the composed-skill registry in memory and rewrites the
    whole file on any macro save (usage counters, dream-cycle proposals). If it
    is running when this script writes, its next save clobbers the update with
    the stale in-memory copy. So the safe order is: stop agent → run sync →
    start agent (boot loads the fresh definition from disk).

Usage:
    PYTHONPATH=src python scripts/sync_youtube_transcribe_macro.py [store_dir]

`store_dir` is the directory holding `composed_skills.json`. If omitted it is
resolved from $GHOST_HOME (…/system/memory/composed_skills), then a glob.
"""
import glob
import os
import sys
from pathlib import Path

from ghost_agent.tools.composed_skills import (
    ComposedSkill, ComposedSkillRegistry, SkillStep,
)
from ghost_agent.tools.yt_download import build_youtube_transcribe_definition


def _resolve_store() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    gh = os.environ.get("GHOST_HOME")
    if gh:
        p = Path(gh) / "system" / "memory" / "composed_skills"
        if p.exists() or p.parent.exists():
            return p
    hits = glob.glob(str(Path.home() / "Data/AI/**/composed_skills.json"),
                     recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise SystemExit("Could not locate the composed_skills store; pass it as an "
                     "argument (the directory holding composed_skills.json).")


def main() -> int:
    store = _resolve_store()
    reg = ComposedSkillRegistry(storage_dir=store)

    d = build_youtube_transcribe_definition()
    steps = [
        SkillStep(tool_name=s["tool"], description=s["description"],
                  param_template=s["params"])
        for s in d["steps"]
    ]
    skill = ComposedSkill(
        name=d["name"], trigger_description=d["description"],
        steps=steps, execution_mode=d["mode"], status="active",
    )

    existed = d["name"] in reg.skills
    if existed:
        del reg.skills[d["name"]]
    reg.register(skill)
    # register() may default status elsewhere; force active for a top-level tool.
    reg.skills[d["name"]].status = "active"
    reg.save()

    stored = reg.skills[d["name"]]
    cmd = stored.steps[0].param_template.get("command", "")
    ok = "base64 -d" in cmd and stored.status == "active" and len(stored.steps) == 2
    print(f"store: {store}")
    print(f"{'updated' if existed else 'created'} youtube_transcribe "
          f"[{stored.execution_mode}] status={stored.status} "
          f"steps={len(stored.steps)}")
    print(f"step1 command carries the resilient downloader: {ok}")
    if not ok:
        print("WARNING: sync verification failed", file=sys.stderr)
        return 1
    print("Done. If the agent was stopped for this, start it now; it loads the "
          "updated macro from disk on boot. (If it was running, re-run this "
          "with the agent stopped — a live save may have clobbered the update.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
