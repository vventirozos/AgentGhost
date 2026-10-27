"""The Hugging Face cache path is pinned before any test runs (conftest).

It freezes at huggingface_hub's first import; a test that had pointed HOME
at a temp dir used to freeze it there for the whole worker, and every later
offline embedder load failed (sys.exit(1) in VectorMemory) — a whole file of
memory tests, a different one per suite run.
"""
import os
import tempfile


def test_the_cache_is_not_under_a_temp_dir(monkeypatch):
    monkeypatch.setenv("HOME", tempfile.mkdtemp())        # what isolating tests do
    import huggingface_hub.constants as c
    assert not os.path.realpath(c.HF_HUB_CACHE).startswith(os.path.realpath(tempfile.gettempdir()))
    assert c.HF_HUB_CACHE.endswith(os.path.join("huggingface", "hub"))
