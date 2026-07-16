"""tqdm must use a THREADING lock, not a multiprocessing one (2026-07-15).

transformers renders a "Loading weights" tqdm bar when the embedder loads; the
default tqdm `get_lock()` creates a multiprocessing RLock — a named posix
semaphore the resource_tracker reports as leaked at every SIGTERM (a plain-kill
deploy: `resource_tracker: 1 leaked semaphore`). Importing memory.vector swaps
tqdm's lock for a threading RLock (we never drive bars across processes), which
does not create the semaphore. This pins that the swap is in place — a cheap,
deterministic proxy for the SIGTERM-only symptom.
"""


def test_tqdm_lock_is_not_multiprocessing():
    import ghost_agent.memory.vector  # noqa: F401 — import applies the swap
    import tqdm

    lock = tqdm.tqdm.get_lock()
    assert "multiprocessing" not in type(lock).__module__, (
        f"tqdm is back on a multiprocessing lock ({type(lock)!r}) — the "
        f"leaked-semaphore fix in memory/vector.py regressed; it creates a "
        f"named posix semaphore that leaks at SIGTERM."
    )
