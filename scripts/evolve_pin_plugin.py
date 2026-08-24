"""Bind `ghost_agent` to the CANDIDATE before any pin file can rebind it.

⚠ WITHOUT THIS, STAGE 1 GRADES THE INCUMBENT. 212 of this repo's test
files begin with

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

and under stage 1 `__file__` is the CANONICAL path — so the pin itself
puts the incumbent's `src/` at `sys.path[0]`, ahead of the PYTHONPATH
the parent set. No pytest flag can prevent that: `--import-mode=importlib`
stops *pytest* inserting a path, not the test file.

MEASURED: a candidate whose `tools/projects.py` was a single
`raise RuntimeError(...)` came back **"240 passed"**. With this plugin
loaded the same run reports 34 errors.

The mechanism is import order, not path order. A regular package's
`__path__` is fixed at first import, so binding `ghost_agent` here —
at plugin-load time, before collection — makes every later submodule
resolve from the candidate no matter what a pin file inserts afterwards.

⚠ THIS FILE LIVES IN `scripts/`, WHICH THE FENCE MARKS IMMUTABLE. It
decides which tree gets graded; a subject able to edit it could grade
itself.
"""
import ghost_agent            # noqa: F401  (binds __path__ to the candidate)
import ghost_agent.tools      # noqa: F401

try:                          # the `src.` form, when the tree provides it
    import src.ghost_agent            # noqa: F401
    import src.ghost_agent.tools      # noqa: F401
except Exception:             # noqa: BLE001
    # A tree without a `src` package is not an error — the bare form is
    # the one that matters and is already bound above.
    pass
