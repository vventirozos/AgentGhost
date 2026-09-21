"""§4JJ — a Greek-script query is raced with region gr-el.

The wave's region was a fixed "wt-wt"; duckduckgo and brave read it (yandex
and yahoo ignore it). Probed over Tor on 12 recorded Greek queries with the
region as the only variable: waves won 9/12 → 11/12, strict-on-topic 87 → 98,
.gr sources 80 → 95. A preference, never a filter.

World where each pin fails: the mapping stops recognising Greek, a Latin
query stops getting the default, or the wave stops passing the mapped
region to every engine.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.ghost_agent.tools.search import _RACE_ENGINES, _race_search_wave, region_for_query


@pytest.mark.parametrize("query, region", [
    ("πλαστογραφία μετά χρήσεως κακούργημα 23.125.000 δραχμές", "gr-el"),
    ("Αλκιβιάδου 154 Πειραιάς ιστορία", "gr-el"),
    ("Revolut breach Θεσσαλονίκη", "gr-el"),                 # mixed script: Greek present
    ("ἀλήθεια τρένο 2023", "gr-el"),                        # polytonic text still carries basic-block letters
    ("revolut government request data breach", "wt-wt"),
    ("eckit Grid reduced_gg spec format", "wt-wt"),
    ("", "wt-wt"),
    ("Купить Volkswagen Teramont", "wt-wt"),                # Cyrillic is not Greek
])
def test_region_for_query(query, region):
    assert region_for_query(query) == region


def _mock_ddgs_module(text_side_effect):
    mod = MagicMock()
    cls = MagicMock()
    mod.DDGS = cls
    inst = MagicMock()
    cls.return_value.__enter__.return_value = inst
    inst.text.side_effect = text_side_effect
    return mod, inst


@pytest.mark.asyncio
@pytest.mark.parametrize("query, region", [("Μιχαήλ Βόδα Αθήνα εγκλήματα", "gr-el"),
                                           ("michail voda athens crimes", "wt-wt")])
async def test_the_wave_passes_the_mapped_region_to_every_engine(query, region):
    seen = {}

    def by_engine(q, **kw):
        seen[kw["backend"]] = kw.get("region")
        return []                                   # nobody wins: every engine is raced
    mod, inst = _mock_ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("src.ghost_agent.tools.search._DDGS_TOR_TIMEOUT", 0.2), \
         patch("src.ghost_agent.tools.search._DDGS_FAST_ENGINE_TIMEOUT", 0.2):
        await _race_search_wave(query, None, 0)
    assert set(seen) == set(_RACE_ENGINES)
    assert set(seen.values()) == {region}
