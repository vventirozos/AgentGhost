"""§4LC part 2: three sentence-valued profile fields become lists — only when
the live value is exactly the listed one; the store keeps every item apart."""
import runpy
from pathlib import Path

import pytest

from ghost_agent.memory.profile import ProfileMemory

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4lc_lists.py"),
                   run_name="not_main")


def _raw():
    return {c: {k: {"v": s, "as_of": "2026-07-07T21:18:28"}} for (c, k), (s, _i) in M["SPLIT"].items()}


def test_each_listed_sentence_becomes_its_items_with_the_original_date():
    todo = M["plan"](_raw())
    assert todo[("relationships", "sons")] == ("2026-07-07T21:18:28",
                                              ["Thodoris (born 2016-11-25)", "Leonidas (born 2026-03-12)"])
    assert len(todo[("interests", "hobbies")][1]) == 4 and len(todo[("assets", "vehicles")][1]) == 3


def test_a_changed_value_applies_nothing():
    raw = _raw()
    raw["assets"]["vehicles"]["v"] = "BMW 118i, Ducati Streetfighter V4s, Sym scooter, Vespa"
    with pytest.raises(SystemExit, match="nothing applied"):
        M["plan"](raw)


def test_the_store_keeps_the_items_apart(tmp_path):
    """Fails where the profile merged two anchored items ("born …") as one
    fact: the second son would replace the first."""
    pm = ProfileMemory(tmp_path)
    for (cat, key), (_s, items) in M["SPLIT"].items():
        for it in items:
            pm.update(cat, key, it, as_of="2026-07-07T21:18:28")
        assert pm.load()[cat][key] == items
    raw = {c: {k: [{"v": i} for i in items]} for (c, k), (_s, items) in M["SPLIT"].items()}
    assert M["plan"](raw) == {}          # already split: a rerun changes nothing
