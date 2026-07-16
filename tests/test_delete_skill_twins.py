"""VectorMemory.delete_skill_twins — orphaned-twin cleanup after a playbook
prune (2026-07-16).

A lesson removed from the JSON playbook leaves its embedded vector twin
(type=skill, trigger=…) behind; this deletes them by their exact metadata key,
in-process (a second PersistentClient risks HNSW corruption).
"""
import tempfile
from pathlib import Path

from ghost_agent.memory.vector import VectorMemory


def _mem():
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _add_skill_twin(vm, trigger, text):
    # Mirror how learn_lesson embeds a twin: type=skill + trigger metadata.
    vm.collection.add(documents=[text], ids=[f"twin-{abs(hash(text))}"],
                      metadatas=[{"type": "skill", "trigger": trigger[:200]}])


def test_deletes_named_twins_only():
    vm = _mem()
    _add_skill_twin(vm, "When playing live chess, coach", "chess coaching note")
    _add_skill_twin(vm, "The user prefers ripgrep", "ripgrep preference note")
    _add_skill_twin(vm, "keep this real lesson", "verify the publish path first")
    vm.add("an unrelated auto memory about the weather", {"type": "auto"})

    removed, detail = vm.delete_skill_twins([
        "When playing live chess, coach", "The user prefers ripgrep"])
    assert removed == 2
    assert detail["before"] - detail["after"] == 2

    # The kept skill twin + the auto memory survive.
    surviving = vm.collection.get(where={"type": "skill"})
    trigs = {m.get("trigger") for m in (surviving.get("metadatas") or [])}
    assert trigs == {"keep this real lesson"}


def test_missing_trigger_is_a_noop():
    vm = _mem()
    _add_skill_twin(vm, "real one", "some rule")
    removed, detail = vm.delete_skill_twins(["never added this trigger"])
    assert removed == 0
    assert detail["before"] == detail["after"]


def test_empty_and_none_triggers_ignored():
    vm = _mem()
    _add_skill_twin(vm, "real one", "some rule")
    removed, _ = vm.delete_skill_twins(["", None])
    assert removed == 0
    # the real twin is untouched
    assert len((vm.collection.get(where={"type": "skill"}).get("ids") or [])) == 1
