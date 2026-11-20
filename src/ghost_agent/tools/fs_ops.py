"""The ONE list of `file_system` operations that change the workspace (§4UY r2 m1).

Three readers had their own copies and drifted (a `copy` once landed on one list only; `restore` landed on
none): the agent's sandbox-cache invalidation and its `is_mutating` test (`core/agent.py`), and the outcome
heuristics' progress test (`distill/outcome_heuristics.py`). All three read this set. No imports: it is
loaded by light modules too.
"""

FS_MUTATING_OPS = frozenset({
    "write", "edit", "replace", "download", "delete", "move", "rename", "copy", "restore",
    "append", "append_file", "write_append", "patch", "mkdir", "batch", "unzip", "git_clone",
})
