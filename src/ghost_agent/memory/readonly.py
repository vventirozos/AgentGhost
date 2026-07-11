"""Read-only façades over the memory stores.

Used to isolate an agent that must be able to READ the operator's memory
but must never WRITE to it — today the delegated sub-agent
(:mod:`core.subagent`). Every write method is a silent no-op; every other
attribute proxies to the real store.

NOTE: ``core.dream``'s self-play path defines its own equivalent classes
inline (they predate this module and carry self-play-specific behaviour,
e.g. the ``is_read_only`` marker Perfect-It checks to skip an expensive
write-oriented path). They are deliberately left alone — the self-play
isolation is proven in production and not worth destabilising for a
de-duplication. Keep the semantics here in sync if you change them there.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("GhostAgent")


class ReadOnlyVectorMemory:
    """Vector store: search/get pass through, mutations are no-ops."""

    is_read_only = True

    def __init__(self, real):
        self.real = real

    def add_memory(self, *a, **kw):
        return None

    def smart_update(self, *a, **kw):
        return None

    def correct_fragment(self, *a, **kw):
        return None

    def delete_memory(self, *a, **kw):
        return None

    def forget(self, *a, **kw):
        return None

    def __getattr__(self, name):
        # Only reached for attributes NOT defined above (so writes stay
        # no-ops). An absent real store degrades to None rather than raising.
        real = self.__dict__.get("real")
        if real is None:
            return None
        return getattr(real, name)


class ReadOnlySkillMemory:
    """Skill playbook: reads pass through, lesson writes are no-ops."""

    is_read_only = True

    def __init__(self, real):
        self.real = real

    def learn_lesson(self, *a, **kw):
        return None

    def save_playbook(self, *a, **kw):
        return None

    def get_playbook_context(self, *a, **kw):
        real = self.__dict__.get("real")
        if real is None:
            return ""
        return real.get_playbook_context(*a, **kw)

    def __getattr__(self, name):
        real = self.__dict__.get("real")
        if real is None:
            return None
        return getattr(real, name)


class ReadOnlyGraphMemory:
    """Knowledge graph: queries pass through, triple writes are no-ops."""

    is_read_only = True

    def __init__(self, real):
        self.real = real

    def add_triplet(self, *a, **kw):
        return None

    def insert_fact(self, *a, **kw):
        return None

    def delete_triplet(self, *a, **kw):
        return None

    def execute_graph_compression(self, *a, **kw):
        return None

    def __getattr__(self, name):
        real = self.__dict__.get("real")
        if real is None:
            return None
        return getattr(real, name)


__all__ = [
    "ReadOnlyVectorMemory", "ReadOnlySkillMemory", "ReadOnlyGraphMemory",
]
