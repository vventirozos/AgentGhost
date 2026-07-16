"""Read-only façades over the memory stores.

Used to isolate an agent that must be able to READ the operator's memory
but must never WRITE to it — today the delegated sub-agent
(:mod:`core.subagent`).

Design (hardened 2026-07-14): a **default-allow-reads / explicit-deny-writes**
proxy. Every genuine mutator is intercepted and no-op'd by NAME, and the raw
writable handles (``.real``, the chromadb ``collection``/``client``, the
NetworkX ``nx_graph``) are blocked so a caller can't reach the underlying
store around the façade (``memory_system.real.add(...)`` /
``memory_system.collection.delete(...)``). Everything else proxies through to
the real store, so reads keep working unchanged.

The previous version listed no-op methods by GUESSED names (``add_memory``,
``delete_memory``, ``add_triplet``, ``insert_fact`` …) that DO NOT EXIST on
the real stores, while the real mutators (``add``, ``ingest_document``,
``delete_fragment``, ``add_triplets``, ``delete_by_target``,
``remove_by_trigger`` …) fell straight through ``__getattr__`` to the
writable store and executed. The mutator lists below are pinned to the real
methods and covered by ``tests/test_readonly_memory.py`` so a store growing a
new writer surfaces there.

NOTE: ``core.dream``'s self-play path defines its own equivalent classes
inline (they predate this module and carry self-play-specific behaviour). They
are deliberately left alone — the self-play isolation is proven in production.
Keep the semantics here in sync if you change them there.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("GhostAgent")


class _ReadOnlyProxy:
    """Base façade: reads proxy through, named mutators no-op, raw writable
    handles are blocked. Subclasses set ``_MUTATORS`` and ``_BLOCKED_ATTRS``.

    ``is_read_only`` is exposed so callers that branch on it (e.g. dream's
    Perfect-It skip) see the wrap.
    """

    is_read_only = True
    _MUTATORS: frozenset = frozenset()
    # Raw writable handles that must never be reachable THROUGH the proxy.
    # ``real`` is always blocked (the underlying store is kept as ``_real``).
    _BLOCKED_ATTRS: frozenset = frozenset({"real"})

    def __init__(self, real):
        # Stored under a private name so ``proxy.real`` hits __getattr__ and
        # is blocked, while internal code uses ``self._real``.
        self._real = real

    @staticmethod
    def _noop(*a, **kw):
        return None

    def __getattr__(self, name):
        # Only fires for names not found by normal lookup (so the class-level
        # methods / _MUTATORS / _BLOCKED_ATTRS / is_read_only resolve first).
        if name in self._MUTATORS:
            return self._noop
        if name in self._BLOCKED_ATTRS:
            return None
        real = self.__dict__.get("_real")
        if real is None:
            return None
        return getattr(real, name)


class ReadOnlyVectorMemory(_ReadOnlyProxy):
    """Vector store: search/get pass through, every writer is a no-op."""

    _MUTATORS = frozenset({
        "add", "smart_update", "ingest_document", "process_batch",
        "bump_retrievals", "bump_helpful", "forget_episode",
        "delete_document_by_name", "correct_fragment", "delete_fragment",
        "delete_by_query", "delete_skill_twins",
        # legacy guessed names kept harmless in case an external caller used them
        "add_memory", "delete_memory", "forget",
    })
    # chromadb handles are directly writable — never expose them.
    _BLOCKED_ATTRS = frozenset({"real", "collection", "client"})

    def search(self, *a, **kw):
        """Proxy ``search`` but force ``record_retrievals=False`` — a read
        that reinforces retrieval stats is still a WRITE to operator memory,
        which the façade must not perform."""
        real = self.__dict__.get("_real")
        if real is None:
            return ""
        kw["record_retrievals"] = False
        return real.search(*a, **kw)


class ReadOnlySkillMemory(_ReadOnlyProxy):
    """Skill playbook: reads pass through, lesson/stat writes are no-ops."""

    _MUTATORS = frozenset({
        "learn_lesson", "save_playbook", "retract_lessons_from_trajectory",
        "record_retrieval", "record_helpful_retrieval",
        "credit_recent_retrievals", "record_retrievals_bulk",
        "prune_low_utility", "mark_verified", "remove_by_trigger",
    })


class ReadOnlyGraphMemory(_ReadOnlyProxy):
    """Knowledge graph: queries pass through, triple writes are no-ops."""

    _MUTATORS = frozenset({
        "add_triplets", "delete_by_target", "wipe_all", "prune_stale_edges",
        "execute_graph_compression", "bump", "initialize_graph",
        # legacy guessed names kept harmless
        "add_triplet", "insert_fact", "delete_triplet",
    })
    # The in-memory NetworkX mirror is directly mutable — block the handle so
    # a caller can't do graph_memory.nx_graph.add_edge(...).
    _BLOCKED_ATTRS = frozenset({"real", "nx_graph"})


__all__ = [
    "ReadOnlyVectorMemory", "ReadOnlySkillMemory", "ReadOnlyGraphMemory",
]
