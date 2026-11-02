"""§4SU Phase 4: the task_update reply carries one `refused` list; the legacy
keys (removed, operator 11-02) are read through it in tests."""

_GATE = {
    "undelivered": ("deliverable", ""),
    "waiting_on_dependencies": ("dependencies", ""),
    "unrun_experiments": ("model_run", ""),
    "gated_unverified": ("evidence", "visual"),
    "gated_constraints": ("evidence", "constraint"),
    "constraint_violations": ("constraints", ""),
    "still_failing": ("recheck", ""),
}


def refused_ids(out, legacy_key):
    """The task ids `refused` lists for the gate the legacy key stood for."""
    gate, word = _GATE[legacy_key]
    return [r["task_id"] for r in (out or {}).get("refused") or []
            if r.get("gate") == gate and (not word or word in str(r.get("reason") or "").lower())
            and "the task stays DONE" not in str(r.get("reason") or "")]


def refused_reasons(out, legacy_key):
    gate, word = _GATE[legacy_key]
    return " | ".join(f"{r['task_id']}: {r.get('reason', '')} — {r.get('fix', '')}"
                      for r in (out or {}).get("refused") or []
                      if r.get("gate") == gate and (not word or word in str(r.get("reason") or "").lower()))
