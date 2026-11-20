"""§4UY N4 — the egress guard's "surname alone identifies" form applies to a
PERSON'S NAME only. It used to take the last word of ANY multi-word profile
value: an address's city or country ("…, Patra" → "patra"), a street's
"street", so after any outside read a search naming the owner's city was
refused as sending the private address out (live 5b9ebfd4, 08d2a444).

The kind of a field is decided once, from its key (`field_kind`); the tests
enumerate every key the identifier rule accepts, so a new key kind cannot
pick up a one-word form without being declared a name."""
import itertools
import re
from types import SimpleNamespace

import pytest

from ghost_agent.memory import egress as EG
import ghost_agent.utils.provenance as PV


#: the keys whose value is a person's name (pinned exactly: adding one is a decision)
NAME_KEYS = {"name", "fullname", "full_name", "wife_name", "husband_name", "partner_name", "spouse_name",
             "son_name", "daughter_name", "child_name", "mother_name", "father_name",
             "sons", "daughters", "children", "kids"}
SURNAME_KEYS = {"surname", "lastname", "last_name"}


def _ident_keys():
    """Every key `_IDENT_KEY` accepts, generated from the words of its own
    pattern (one or two words, joined by "" or "_") — the whole language of
    the regex, so a key added to it is enumerated here without editing the test."""
    words = sorted(set(re.findall(r"[a-z]+", EG._IDENT_KEY.pattern)) - {"i"})
    keys = set()
    for n in (1, 2, 3):
        for combo in itertools.product(words, repeat=n):
            for sep in ("", "_"):
                k = sep.join(combo)
                if EG._IDENT_KEY.search(k):
                    keys.add(k)
    return sorted(keys)


ALL_KEYS = _ident_keys()


def test_the_enumeration_reaches_every_kind_of_key():
    # sanity: the generator really walks the regex (addresses, dates, names, contacts, family)
    for k in ("address", "home_address", "street", "postcode", "zip_code", "birthdate", "phone_number",
              "email", "first_name", "last_name", "wife_name", "children", "mother_birthday", "iban"):
        assert k in ALL_KEYS, k
    assert len(ALL_KEYS) > 40


@pytest.mark.parametrize("key", ALL_KEYS)
def test_only_a_name_field_yields_a_one_word_form(key):
    """No field but a person's name (or a surname field) may yield a single word of its value."""
    value = "Odos Irida Kosmidou 4, Patrasville, Grecia"   # multi-word, long last word
    forms = EG._forms(value, f"root.{key}")
    single = {f for f in forms if re.fullmatch(r"[^\W\d_]+", f)}
    kind = EG.field_kind(f"root.{key}")
    if key in NAME_KEYS:
        assert kind == "name" and single == {EG.fold("grecia")[0]}
    elif key in SURNAME_KEYS:
        assert kind == "surname" and single == set()          # a multi-word value is not a lone surname
    else:
        assert kind == "value" and single == set(), (key, single)


def test_the_declared_kinds_are_exactly_these():
    assert {k for k in ALL_KEYS if EG.field_kind(k) == "name"} == NAME_KEYS & set(ALL_KEYS)
    assert {k for k in ALL_KEYS if EG.field_kind(k) == "surname"} == SURNAME_KEYS & set(ALL_KEYS)
    assert NAME_KEYS | SURNAME_KEYS <= set(ALL_KEYS)


def test_a_name_still_yields_its_surname_and_a_surname_field_itself():
    assert EG.fold("kosmidou")[0] in EG._forms("Elena Kosmidou", "root.name")
    assert EG.fold("kosmidou")[0] in EG._forms("Markos Kosmidou", "relationships.son_name")
    assert EG.fold("papadakis")[0] in EG._forms("Papadakis", "root.last_name")
    assert EG._forms("Elena", "root.first_name") == []            # a lone first name identifies nobody
    assert EG._forms("Anna Maria", "root.first_name") == ["anna maria"]   # no surname in a first name
    assert EG._forms("Zhou", "root.surname") == []                 # too short to stand alone


class _PM:
    def __init__(self, data):
        self._d = data

    def load(self):
        return self._d


PROFILE = {"root": {"name": "Elena Kosmidou", "address": "Odos Irida 4, Patrasville",
                    "home_address": "Iris Street 12", "surname": "Papadakis", "email": "elena@example.org",
                    "location": "Patrasville"},
           "relationships": {"children": ["Markos Kosmidis", "Kosmas"]}}


@pytest.fixture
def after_outside_read(monkeypatch):
    monkeypatch.setattr(PV, "untrusted_seen", lambda: ["https://news.example/page"])
    monkeypatch.setattr(PV, "user_message", lambda: "what is the weather this weekend")
    return SimpleNamespace(egress_profile=_PM(PROFILE))


@pytest.mark.parametrize("query,refused", [
    ("weather Patrasville weekend", False),            # the address's town (live 5b9ebfd4 shape)
    ("street food festival", False),                   # the street address's last word
    ("Odos Irida 4, Patrasville", True),               # the address itself
    ("Iris Street 12", True),
    ("Kosmidou interview", True),                      # the owner's surname
    ("Elena Kosmidou", True),
    ("Kosmidis school results", True),                 # a child's surname
    ("Papadakis family", True),                        # a surname field alone
    ("elena@example.org leak", True),
    ("Kosmas bakery", False),                          # a lone first name
])
def test_the_guard_after_outside_content(after_outside_read, query, refused):
    out = EG.content_egress_refusal("web_search", {"query": query}, after_outside_read)
    assert (out is not None) is refused, query


def test_a_url_naming_the_city_is_not_refused(after_outside_read):
    assert EG.content_egress_refusal("browser", {"url": "https://weather.example/patrasville"},
                                     after_outside_read) is None
