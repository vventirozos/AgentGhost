import asyncio
import tempfile
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List
from .file_system import _get_safe_path
from ..utils.logging import Icons, pretty_log, spawn_bg
from ..utils.helpers import get_utc_timestamp, helper_fetch_url_content, recursive_split_text, semantic_split_text
from ..memory.scratchpad import Scratchpad
from ..memory.temporal import anchor as _anchor_temporal
from .outcome import ToolOutcome


_MEMBER_BLOCK = ("SYSTEM BLOCK: the profile and memory are not available for this channel — "
                 "nothing was changed. Continue without saving or forgetting.")


def _teachable(trajectories):
    """Member turns never train the owner's router / PRM (§4KJ R7: the member
    controls the text)."""
    from ..memory.skills import iter_teachable
    return iter_teachable(trajectories)


def _member_block():
    """R3 review (2026-09-24): a channel member's "remember that I'm vegan"
    / "forget X" wrote into and deleted from the OWNER's profile and memory.
    One predicate (`utils.logging.requester_is_member`), one refusal."""
    try:
        from ..utils.logging import requester_is_member
        if requester_is_member():
            return ToolOutcome.rejected(_MEMBER_BLOCK, reason_code="owner_data_blocked")
    except Exception:  # noqa: BLE001
        return None
    return None


def _owner_write_block():
    """§4KY review: a PROBE (`probe-` id) or the agent's own background work
    (`job-`/`sched-`/`sub-`) wrote `user HAS_PROFESSION pilot` and
    `root.profession` through update_profile / insert_fact — "a probe never
    teaches" held for the journal but not for the tools. Facts about the
    owner come from the owner's own turns; a member is refused as before."""
    _m = _member_block()
    if _m is not None:
        return _m
    try:
        from ..utils.logging import request_id_context, is_probe_request_id
        from ..core.autonomous_activity import is_internal_request
        rid = str(request_id_context.get() or "")
        if is_probe_request_id(rid) or is_internal_request(rid):
            return ToolOutcome.rejected(
                "NOT written: facts about the owner come from the owner's own conversations — a diagnostic "
                "probe or a background job does not write them.", reason_code="owner_write_not_owner_turn")
    except Exception:  # noqa: BLE001
        return None
    return None


#: Trailing path separators to strip from a model-supplied target. A literal
#: `"/" + os.sep` is `"//"` on POSIX — `rstrip` takes a CHARACTER SET, so the
#: duplicate was a no-op tell that the argument was misread as a suffix.
_PATH_SEPS = "".join(dict.fromkeys("/" + os.sep))

logger = logging.getLogger("GhostAgent")

# Strong references to in-flight fire-and-forget graph-extraction tasks.
# Fire-and-forget graph extraction is scheduled via utils.logging.spawn_bg,
# which owns the process-wide strong-ref registry (the event loop keeps only
# weak refs, so an unreferenced task can be GC'd before it runs) and drains at
# shutdown. (The old module-local _GRAPH_EXTRACT_TASKS set was one of four
# ad-hoc fire-and-forget conventions, now consolidated.)

# Hard ceiling on the INLINE graph-triplet extraction in the bus-aware
# insert_fact path. That LLM call is pure enrichment (the fact itself is
# stored regardless), but it is awaited on the tool's critical path — so an
# upstream/worker stall (e.g. no --worker-nodes pool, or a 503) would hang the
# whole turn AND, because it blocks before publish_fact(), the fact would
# never be stored at all. Bounding it means a slow extractor costs at most
# this many seconds and the fact still lands in the vector store.
_GRAPH_EXTRACT_TIMEOUT_S = 20.0

# Containers routed to the audio-transcription ingest path (memory.audio_ingest)
# instead of the plain-text branch, which would decode them as replacement-char
# noise. VIDEO is included deliberately: ffmpeg takes the audio track, and a
# recorded conference talk is far more often an .mp4 than a .wav.
_AUDIO_INGEST_EXTS = (
    ".wav", ".mp3", ".m4a", ".m4b", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma",
    ".aiff", ".aif", ".amr", ".3gp", ".weba", ".mpga",
    ".mp4", ".mov", ".mkv", ".mka", ".webm", ".avi",
)

# Suffixes a downloader gives a file it is STILL WRITING. The fuzzy resolver
# below matches by stem/substring, and `yt_audio.m4a.part` matched
# `yt_audio.m4a` — partial AAC then went down the plain-text branch as
# replacement-character noise (§4KE). In-flight artefacts are never a
# document.
_INFLIGHT_SUFFIXES = (".part", ".ytdl", ".tmp", ".crdownload", ".download", ".partial")


def _is_within_root(path: Path, root: Path) -> bool:
    """True iff `path` is inside `root`, compared path-component-wise.

    NOT `str(path).startswith(str(root))`: that treats a *sibling*
    directory whose name merely shares the prefix (e.g. ``/x/sandbox_evil``
    vs root ``/x/sandbox``) as "inside", which let a resolved symlink
    escape the sandbox-deletion guard.
    """
    if hasattr(path, "is_relative_to"):  # Python 3.9+
        return path.is_relative_to(root)
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _value_mentions_target(value, target_lc: str) -> bool:
    """True iff a profile VALUE string references ``target``.

    Token/word-boundary aware so ``forget('age')`` does NOT match the value
    ``'language'`` (the exact regression the key-only sweep was guarding
    against), while ``forget('mortimer')`` DOES match
    ``'Mortimer the iguana (removed)'``. Multi-word targets fall back to a
    plain substring test (token membership can't span spaces).
    """
    from ..memory.profile import mentions
    return mentions(value, target_lc)


class _NullCM:
    """No-op context manager used as a fallback when a lock helper is
    missing (e.g. tests with a MagicMock memory_system). Lets shared
    sweep helpers stay structurally consistent without conditional code."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# Types the `forget` sweeps must NEVER delete as collateral: ingested
# document chunks (deleting one guts a manual the library index still
# lists) and episode/skill/acquired-skill twins (deleting one orphans its
# JSON-side record and breaks that store's semantic recall). Conversational
# fact types (auto/identity/manual/synthesis/…) remain forgettable — that
# is the tool's job.
_FORGET_PROTECTED_TYPES = [
    "document", "episode", "skill", "acquired_skill",
    # `document_summary` added §4R R2 (2026-08-08). It is the summary TWIN of a
    # `document` row, and `document` is protected — deleting the summary while
    # its source document survives leaves the two stores asymmetric, which is
    # the drift this protected list exists to prevent. (Live: 8279 documents,
    # 1 summary.) `synthesis` deliberately stays forgettable here — see the
    # note above and the expansion-sweep guard below.
    "document_summary",
]
#: Fresh review (§4KW): the largest vector distance at which an ENTITY forget
#: removes a fact that does not literally name it. Unrelated owner facts were
#: measured at 0.53–0.65 from short targets; a near-paraphrase is well under.
_FORGET_SEMANTIC_MAX = 0.3
#: …and the largest distance at which a fact carrying ALL the target's
#: distinctive words is a candidate (measured: "my address" ~ the address
#: facts at 0.73, unrelated facts at 0.77+ with no shared word)
_FORGET_SHARED_WORD_MAX = 0.8


def _profile_line(result, ok_text: str) -> str:
    """A forget report line for a profile write — a REFUSED write (read-degraded
    store) is reported as such, never with a green tick (second review)."""
    if isinstance(result, str) and result.lower().startswith("error"):
        return f"⚠️ Profile: {result}"
    return f"✅ Profile: {ok_text}"


#: words that name no particular fact ("the user's …", "my project …")
_FORGET_HUB_WORDS = frozenset("""user users agent assistant project projects file files forget forgot said say says
remember thing things stuff info information about details data memory memories fact facts old new
that this these those what when where who why how which it its them they there here""".split())
#: attribute nouns: a target made only of these names a KIND of fact, so
#: several facts that mention it are several candidates, not one entity
#: (fourth review: `forget address` deleted the home AND the email address)
_FORGET_ATTRIBUTE_WORDS = frozenset("""address addresses name names email emails phone number numbers birthday
birthdays birthdate date dates age job jobs work city town country home location password username nickname
account accounts preference preferences favorite favourite hobby hobbies car cars pet pets wife husband son sons
daughter daughters child children kids family school company employer salary""".split())
#: endings that make two words inflections of one stem
#: (fifth review: "e", "t", "d", "er" made plan~plant/plane, star~start,
#: bear~beard, bank~banker and Louis~Louise one word)
_INFLECTIONS = frozenset({"", "s", "es", "ed", "ing", "ings", "ies", "ly"})
#: …and the distance under which a fact covering HALF the target's words counts
_FORGET_PARTIAL_WORD_MAX = 0.6


def _word_matches(w: str, f: str) -> bool:
    """Same word or an inflection of one stem (weight ~ weighs, address ~
    addresses) — NOT any word it starts (fourth review: `homework` matched
    "home town", `workout` "works at", `plan` "planet")."""
    if w == f:
        return True
    k = 0
    for a, b in zip(w, f):
        if a != b:
            break
        k += 1
    return k >= 4 and w[k:] in _INFLECTIONS and f[k:] in _INFLECTIONS


def _target_word_coverage(fact: str, target: str):
    """Share of the target's DISTINCTIVE content words found in ``fact``
    (exact or inflected), or None when the target has none."""
    try:
        from ..memory.lesson_scope import content_words
        tw = {w for w in content_words(target) if len(w) >= 3 and not w.isdigit() and w not in _FORGET_HUB_WORDS}
        if not tw:
            return None
        fw = content_words(fact)
        return sum(1 for w in tw if any(_word_matches(w, f) for f in fw)) / len(tw)
    except Exception:  # noqa: BLE001
        return None


_OWNER_IDENTITY_KEYS = {"name", "email", "phone", "address", "location", "birthday", "birthdate", "age", "pronouns",
                        "nationality", "timezone", "username", "nickname"}


def _is_owner_name(profile_memory, target: str) -> bool:
    """``target`` is the owner's own name (root.name, or one of its words)."""
    try:
        own = str(((profile_memory.load() or {}).get("root") or {}).get("name") or "").strip().lower() \
            if profile_memory else ""
    except Exception:  # noqa: BLE001
        return False
    from ..memory.profile import _fold
    own, t = _fold(own).strip(), _fold(target).strip()       # (r8 review: "Βασίλης" vs its folded entity)
    return bool(own) and bool(t) and (t == own or t in own.split())


def _names_an_entity(target: str) -> bool:
    """A target with a distinctive word that is not only an attribute noun
    ("wife", "name", "home" name no entity) — profile-writes review."""
    d = _distinct_words(target)
    return bool(d) and not d <= _FORGET_ATTRIBUTE_WORDS


def _current_request_text() -> str:
    try:
        from ..memory.lesson_scope import current_request
        return str(current_request.get() or "")
    except Exception:  # noqa: BLE001
        return ""


def _distinct_words(target: str) -> set:
    """The target's words that can name a particular fact (not hub words)."""
    try:
        from ..memory.lesson_scope import content_words
        return {w for w in content_words(target) if w not in _FORGET_HUB_WORDS and len(w) >= 2}
    except Exception:  # noqa: BLE001
        return set()


def _is_the_value(value, target_lc: str) -> bool:
    """The stored value IS the target, or is ABOUT it — its distinctive words
    are all the target's, or it is a single item that STARTS with the
    target's entity words ("Mortimer the iguana", "Tesla Model 3" for "my old
    car Tesla"). A value listing several things ("BMW 118i, Ducati …") is
    not (third review: over-correction left Mortimer in the profile)."""
    vw, tw = _distinct_words(str(value)), _distinct_words(target_lc)
    if not vw or not tw:
        return False
    if vw <= tw:
        return True
    entity = [w for w in _ordered_words(target_lc) if w in tw and w not in _FORGET_ATTRIBUTE_WORDS]
    if not entity or re.search(r"[,;/&]|\band\b", str(value), re.IGNORECASE):
        return False
    return _ordered_words(str(value))[:len(entity)] == entity


def _ordered_words(text: str) -> list:
    try:
        from ..memory.lesson_scope import normalize_request
        return [w for w in normalize_request(text).split() if w not in _FORGET_HUB_WORDS]
    except Exception:  # noqa: BLE001
        return str(text).lower().split()


#: relation nouns name a PERSON ("my wife Fotini"); the other attribute nouns
#: QUALIFY an entity ("Fotini's birthday") and narrow the forget to that fact
_RELATION_WORDS = frozenset("wife husband son sons daughter daughters child children kids family".split())
_QUALIFIER_WORDS = _FORGET_ATTRIBUTE_WORDS - _RELATION_WORDS


def _entity_of(target: str) -> str:
    """The entity a forget target names, as stored names spell it: hub,
    attribute and function words dropped, possessives stripped, and dots and
    hyphens KEPT ("node.js", "pista-gp") — re-review: the normalised words
    turned "notes.md" into one U+2024 token and "x-ray" into "x ray", so a
    dotted or hyphenated name never reached the graph."""
    from ..memory.profile import _fold
    # (a possessive's "s" is a one-letter token, dropped below)
    toks = re.findall(r"\w(?:[\w.\-]*\w)?", _fold(target))
    named = [i for i, w in enumerate(toks) if w not in _FORGET_HUB_WORDS and w not in _FORGET_ATTRIBUTE_WORDS
             and _distinct_words(w)]
    if not named:
        return ""
    # function words INSIDE the name stay ("vale of tempe", "messmer the
    # impaler" — r8 review); hub/attribute words never do
    span = toks[named[0]:named[-1] + 1]
    return " ".join(w for w in span if w not in _FORGET_HUB_WORDS and w not in _FORGET_ATTRIBUTE_WORDS
                    and (len(w) >= 2 or w.isdigit()))


def _qualifiers_of(target: str) -> list:
    """The attribute nouns that qualify the entity: AFTER it ("Fotini's
    birthday", "Leonidas birthdate") or joined by "of" ("the birthday of
    Fotini") — not a category before it ("my old car Tesla" is the car)."""
    from ..memory.lesson_scope import normalize_request
    words = _ordered_words(target)
    ent = set(normalize_request(_entity_of(target)).split())
    first = next((i for i, w in enumerate(words) if w in ent), None)
    if first is None:
        return []
    of = re.search(r"\bof\b", str(target), re.IGNORECASE) is not None
    return [w for i, w in enumerate(words) if w in _QUALIFIER_WORDS and (i > first or of)]


#: an attribute's other names in keys and predicates (r8 review: BORN_ON was
#: missed for "birthday"; a 4-letter stem took COMPANION for "company")
_QUALIFIER_SYNONYMS = {
    "birthday": {"birth", "birthday", "birthdate", "born", "dob"},
    "birthdays": {"birth", "birthday", "birthdate", "born", "dob"},
    "birthdate": {"birth", "birthday", "birthdate", "born", "dob"},
    "job": {"job", "jobs", "occupation", "profession", "employed", "employer", "works"},
    "work": {"work", "works", "worked", "employed", "employer", "occupation", "profession"},
    "company": {"company", "employer", "employed", "works"},
    "employer": {"company", "employer", "employed", "works"},
    "name": {"name", "named", "called", "nickname"},
    "home": {"home", "lives", "resides", "address"},
    "address": {"address", "addresses", "lives", "resides"},
    "city": {"city", "lives", "resides", "located"},
    "town": {"town", "lives", "resides", "located"},
    "location": {"location", "located", "lives", "resides"},
    "phone": {"phone", "mobile", "number"},
    "email": {"email", "emails", "mail"},
}


def _qualifier_matches(words, quals) -> bool:
    """A key/predicate word names the qualifier: the same word, an
    inflection of it (`_word_matches`), or one of its listed other names."""
    for t in words:
        t = str(t).lower()
        for q in quals:
            if t == q or _word_matches(q, t) or t in _QUALIFIER_SYNONYMS.get(q, ()):
                return True
    return False


def _forget_profile(profile_memory, target: str, related: bool = False, qualifiers=(), family: bool = False) -> list:
    """The profile leg of `forget` (profile-writes review). Deletes:
      * an explicit ``category.key``;
      * the ONE key named exactly by a specific target (not a hub or attribute
        word: `forget name` no longer deleted the owner's name);
      * a list ITEM naming the target, or a scalar whose value IS the target.
    Lists, never deletes: a key that only partly matches (`son` →
    `son_thodoris_birthdate`), several exact keys, and a scalar that MENTIONS
    the target among other things (`Athens` in the home address, `BMW` in a
    vehicles line). ``related``: an alias from the graph — values only."""
    out, cands = [], []
    target_lc = str(target).lower().strip()
    data = profile_memory.load()
    if getattr(profile_memory, "_degraded", False) is True:
        return ["⚠️ Profile: the profile store could not be read — its fields were NOT searched."]
    distinct = _distinct_words(target_lc)
    if qualifiers and not related:
        # (re-review) "Fotini's birthday" deleted the wife-name field and
        # every edge about her: an attribute-qualified target removes only
        # the fields naming BOTH the entity and the attribute; fields that
        # merely mention the entity are listed
        from ..memory.profile import _fold
        ent = set(_entity_of(target).split())
        hit, near = [], []
        for cat, sub in data.items():
            if not isinstance(sub, dict):
                continue
            for k, v in sub.items():
                kw = _fold(k).replace("-", "_").split("_")
                has_ent = bool(ent) and any(w in kw for w in ent)
                if has_ent and _qualifier_matches(kw, qualifiers):
                    hit.append((cat, k))
                elif has_ent or (ent and any(_value_mentions_target(v, w) for w in ent)):
                    near.append(f"{cat}.{k}")
        for cat, k in hit:
            out.append(_profile_line(profile_memory.delete(cat, k), f"Removed {cat}.{k}"))
        cands += near
        data = profile_memory.load()
        _pl = _FORGET_PLAN.get()
        if _pl is not None:
            for c in dict.fromkeys(cands):
                _cat, _k = c.split(".", 1)
                _pl.add("profile_field", {"category": _cat, "key": _k},
                        f"profile {c} = {str((data.get(_cat) or {}).get(_k))[:80]!r}", default=False)
        if cands:
            out.append("ℹ️ Profile: these fields mention '" + _entity_of(target) + "' — NOT changed: "
                       + ", ".join(dict.fromkeys(cands)) + ".")
        return out
    tag = f" (related '{target}')" if related else ""
    m = re.fullmatch(r"([a-z_]+)\.([a-z0-9_]+)", target_lc)
    if m and not related:
        cat, k = m.group(1), m.group(2)
        if isinstance(data.get(cat), dict) and k in data[cat]:
            return [_profile_line(profile_memory.delete(cat, k), f"Removed {cat}.{k}")]
    handled = set()
    if not related:
        exact, partial = [], []
        key_form = target_lc.replace(" ", "_").replace("-", "_")
        # …and the ENTITY's form (r8 review: "my wife Fotini" listed
        # `fotini_description` while "Fotini" deleted it)
        ent_form = _entity_of(target_lc).replace(" ", "_").replace("-", "_")
        forms = [f for f in dict.fromkeys((key_form, ent_form)) if len(f) >= 3]
        for cat, sub in data.items():
            if not isinstance(sub, dict):
                continue
            for k in sub:
                kl = k.lower()
                if kl == key_form:
                    exact.append((cat, k))
                elif any(f in kl.split("_") or kl.startswith(f + "_") or kl.endswith("_" + f) for f in forms):
                    partial.append((cat, k))
        if len(exact) == 1 and distinct and not distinct <= _FORGET_ATTRIBUTE_WORDS:
            cat, k = exact[0]
            out.append(_profile_line(profile_memory.delete(cat, k), f"Removed {cat}.{k}"))
            handled.add(exact[0])
        else:
            cands += [f"{c}.{k}" for c, k in exact]
        if family:
            # forgetting a family PERSON takes the fields named after them
            # too (re-review: `fotini_description` outlived the graph edges)
            for c, k in partial:
                if (c, k) not in handled:
                    out.append(_profile_line(profile_memory.delete(c, k), f"Removed {c}.{k}"))
                    handled.add((c, k))
        cands += [f"{c}.{k}" for c, k in partial if (c, k) not in handled]
        data = profile_memory.load()
    if distinct and distinct <= _FORGET_ATTRIBUTE_WORDS and not related:
        # an attribute word names no entity — list the fields it names
        # ("forget my wife" → relationships.wife_name), never delete
        for cat, sub in data.items():
            if isinstance(sub, dict):
                cands += [f"{cat}.{k}" for k in sub
                          if any(a in k.lower().replace("-", "_").split("_") for a in distinct)]
    if distinct and not distinct <= _FORGET_ATTRIBUTE_WORDS:
        # a REPLACED value kept as `previous` is forgotten too
        # the ENTITY decides a mention ("tesla" in "my old car Tesla"), dots
        # and hyphens kept ("node.js" — r8 review: the normalised words made
        # it one U+2024 token that matched nothing)
        mention = _entity_of(target_lc) or target_lc
        if hasattr(profile_memory, "drop_previous_mentioning"):
            for line in profile_memory.drop_previous_mentioning(mention):
                out.append(f"✅ Profile: {line}")
        for cat, sub in list(data.items()):
            if not isinstance(sub, dict):
                continue
            for k, v in list(sub.items()):
                if (cat, k) in handled:
                    continue
                if isinstance(v, list):
                    if any(_value_mentions_target(it, mention) for it in v):
                        res = profile_memory.prune_value(cat, k, mention)
                        out.append(_profile_line(res, f"{res}{tag}"))
                elif _value_mentions_target(v, mention):
                    # the owner's own identity fields go only when named
                    # explicitly (`forget Vasilis` removed root.name)
                    # (re-review) nor any root field: the graph keeps the
                    # matching owner fact (`user WORKS_AT evolmonkey`), so
                    # deleting `root.company` left the two disagreeing
                    if _is_the_value(v, target_lc) and cat != "root":
                        out.append(_profile_line(profile_memory.delete(cat, k), f"Removed {cat}.{k} (value match){tag}"))
                    else:
                        cands.append(f"{cat}.{k}")
    _pl = _FORGET_PLAN.get()
    if _pl is not None:
        for c in dict.fromkeys(cands):
            _cat, _k = c.split(".", 1)
            _pl.add("profile_field", {"category": _cat, "key": _k},
                    f"profile {c} = {str((data.get(_cat) or {}).get(_k))[:80]!r}", default=False)
    if cands:
        out.append("ℹ️ Profile: these fields mention or partly match '" + str(target) + "' — NOT changed: "
                   + ", ".join(dict.fromkeys(cands)) + ". Forget one by its exact name (category.key), or update it.")
    return out


#: How a bus leg SAYS it did not write. Two vocabularies, one meaning: the
#: leg raised ("error: …"), or the store declined the write and said so
#: ("refused: …" — `MemoryBus._vector` emits that for `VectorMemory`'s own
#: `ADD_REFUSALS`).
#:
#: ⚠ THE SECOND WORD WAS INVISIBLE TO ITS ONLY CLASSIFIER (§4GK round 6).
#: Round 5 taught `add()` to answer a refusal and the bus to report it, and
#: both classifiers here still keyed on `startswith("error")` alone — so
#: `_bus_write_failures` answered `[]` and `_bus_canonical_failed` answered
#: False for a write the store had explicitly declined. Measured: for
#: `insert_fact` the vector leg IS the canonical store, so a refused write
#: reported SUCCESS to the user and disarmed the verifier's
#: unverified-mutation gate on top. A producer and its consumer have to be
#: read together or the new status is decoration.
_BUS_FAILURE_PREFIXES = ("error", "refused")


def _bus_write_failures(report) -> list:
    """Subsystem entries in a `publish_fact` report that actually FAILED
    (skip/dedup are normal outcomes). The bus swallows exceptions into the
    report by design; callers that then discard the report turn a total
    write failure into 'SUCCESS' — the legacy path's PARTIAL contract must
    survive the bus migration."""
    if not isinstance(report, dict):
        return []
    return sorted(
        f"{k}: {v}" for k, v in report.items()
        if isinstance(v, str) and v.startswith(_BUS_FAILURE_PREFIXES))


#: Legs whose failure means the write did NOT happen. The vector and graph
#: indexes are best-effort secondary writes; these are the canonical stores.
#: Which bus leg is the CANONICAL store — per operation, because it differs.
#:
#: `MemoryBus.publish_fact` can emit exactly four keys: vector | graph |
#: profile | skill. An earlier flat tuple listed `fact`, `memory` and
#: `episodic`, none of which the bus can emit — three dead entries that made
#: the rule look broader than it was — and it omitted `vector`, which IS the
#: canonical store for `insert_fact`. So a fact that was never stored emitted
#: PARTIAL, and the verifier's bookkeeping gate exempts PARTIAL on purpose:
#: the write vanished AND the gate was disarmed.
#:
#: Making it flat the other way is equally wrong: for `update_profile` the
#: canonical store is the profile JSON and `vector` is a retrieval index
#: whose failure genuinely is partial. `graph` is never canonical.
_CANONICAL_BUS_LEGS = {
    "insert_fact": ("vector",),
    "update_profile": ("profile",),
    "learn_skill": ("skill",),
}


def _bus_canonical_failed(report, kind: str = "") -> bool:
    """Did a CANONICAL leg fail, as opposed to a secondary index?

    PARTIAL means "part of it landed" and is deliberately exempted from the
    verifier's bookkeeping gate, because `update_profile` returns it when the
    canonical write succeeded and only an index lagged. On the bus path the
    same PARTIAL was emitted when the canonical leg itself errored — so a
    TOTAL write failure disarmed the unverified-mutation guard and skipped
    the verifier entirely. A canonical failure is a FAILURE.
    """
    if not isinstance(report, dict):
        return False
    legs = _CANONICAL_BUS_LEGS.get(kind)
    if legs is None:
        # unknown operation: every named canonical leg counts
        legs = tuple({l for v in _CANONICAL_BUS_LEGS.values() for l in v})
    return any(
        isinstance(v, str) and v.startswith(_BUS_FAILURE_PREFIXES)
        and str(k).lower() in legs
        for k, v in report.items())

async def tool_remember(text: str = None, memory_system=None, graph_memory=None, llm_client=None, model_name="default", memory_bus=None):
    """Insert a new fact. When a `memory_bus` is supplied the commit is
    dispatched through `publish_fact("insert_fact", ...)` so the tool stays
    ignorant of which subsystems exist; otherwise the legacy direct path
    runs (kept for backward compatibility with existing tests/callers)."""
    _blocked = _owner_write_block()
    if _blocked is not None:
        return _blocked
    # Same contract as tool_unified_forget: 'text' is THIS function's
    # parameter name and is not a name the knowledge_base schema accepts.
    # The dispatcher guards insert_fact before reaching here.
    if not text:
        return "SYSTEM ERROR: The 'text' parameter is MANDATORY. You must specify it."
    # Anchor BEFORE the dedup hash and before any store sees it: "remember
    # that Leonidas is 4 months old" is the single most direct route a
    # decaying snapshot takes into memory. Anchoring here (rather than
    # inside VectorMemory.add) is deliberate — add() also ingests DOCUMENT
    # chunks, whose "4 months old" belongs to the document's own timeline,
    # not to the ingest date, and must not be rewritten.
    text = _anchor_temporal(text)
    pretty_log("Memory Store", text, icon=Icons.MEM_SAVE)

    # --- DEDUP: check whether the same text has already been embedded.
    # VectorMemory keys by md5(text), so a duplicate ingest is a no-op at
    # the storage layer — but without this short-circuit the bus still
    # fan-outs 4 publish_fact coroutines and re-extracts triplets via LLM
    # for each repeat call. Hash-check first.
    vec_for_check = memory_system
    if vec_for_check is None and memory_bus is not None:
        vec_for_check = getattr(memory_bus, "vector", None)
    if vec_for_check is not None:
        try:
            import hashlib as _h
            mem_id = _h.md5(str(text).encode("utf-8")).hexdigest()
            collection = getattr(vec_for_check, "collection", None)
            if collection is not None and hasattr(collection, "get"):
                existing = collection.get(ids=[mem_id])
                # Strict shape check — a MagicMock would otherwise satisfy
                # truthiness and short-circuit every test path.
                ids = existing.get("ids") if isinstance(existing, dict) else None
                if isinstance(ids, list) and len(ids) > 0 and any(isinstance(i, str) for i in ids):
                    return f"NOOP: Memory '{text[:60]}...' is already stored (id={mem_id[:8]}). No duplicate embedding written."
        except Exception:
            pass

    # --- BUS-AWARE PATH ---
    if memory_bus is not None:
        try:
            # Store the fact IMMEDIATELY, without triplets. Graph-triplet
            # extraction is a separate LLM call that reliably STALLS when made
            # from inside a live turn (worker/upstream contention). It used to
            # be AWAITED inline BEFORE this publish, which (a) hung the turn to
            # the 600s _wait_for_foreground_clear ceiling and (b) — because the
            # hang was before the publish — never stored the fact at all. It is
            # pure enrichment, so we move it off the critical path: publish now,
            # extract-and-add-triplets in a background task (where
            # is_background=True is finally correct — a fire-and-forget task,
            # not something the turn awaits, so it can't self-deadlock).
            _report = await memory_bus.publish_fact("insert_fact", {
                "text": text,
                "metadata": {"timestamp": get_utc_timestamp(), "type": "manual"},
                "triplets": [],
            })
            _fails = _bus_write_failures(_report)
            if _fails:
                _canon = _bus_canonical_failed(_report, "insert_fact")
                _mk = ToolOutcome.failed if _canon else ToolOutcome.partial
                _head = "FAILED" if _canon else "PARTIAL"
                return _mk((f"{_head}: memory write had failures — "
                        f"{'; '.join(_fails)}. The fact may not be retrievable."),
                        reason_code="memory_write_partial")

            graph = getattr(memory_bus, "graph", None) or graph_memory
            if llm_client is not None and graph is not None:
                _payload_text = text

                async def _extract_and_add_triplets():
                    try:
                        from ..core.agent import extract_json_from_text
                        prompt = f"Extract explicit entity relationships from this fact into a 'graph_triplets' array as objects with 'subject', 'predicate', and 'object' keys. Predicates MUST be uppercase verbs. Return ONLY JSON. Fact: {_payload_text}"
                        payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Graph Extractor. Output JSON."}, {"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
                        data = await asyncio.wait_for(
                            llm_client.chat_completion(payload, use_worker=True, is_background=True, off_main_only=True, task_label="smart-memory"),  # §4O A-MAJOR-2: don't dogpile main on worker failure
                            timeout=_GRAPH_EXTRACT_TIMEOUT_S,
                        )
                        res = extract_json_from_text(data["choices"][0]["message"].get("content", ""), repair_truncated=True)
                        triplets = res.get("graph_triplets", []) or []
                        # §4M (Lens C MINOR): tombstone parity with the
                        # consolidation path — no positive edge from a
                        # removal-shaped sentence.
                        try:
                            from ..utils.helpers import is_removal_triplet
                            triplets = [t for t in triplets
                                        if not is_removal_triplet(t)]
                        except Exception:
                            pass
                        if triplets:
                            await asyncio.to_thread(graph.add_triplets, triplets)
                    except Exception as e:
                        # §4M (Lens C): was a bare fully-silent pass — the
                        # graph enrichment quietly never happening is the
                        # "silent inoperative subsystem" class.
                        logger.warning(
                            "remember: background graph extraction failed "
                            "(fact stored, graph not enriched): %s", e)

                spawn_bg(_extract_and_add_triplets(), name="graph-extract")

            return f"Memory stored: '{text}'"
        except Exception as e:
            return f"Error storing memory: {e}"

    # --- LEGACY DIRECT PATH ---
    if not memory_system: return "Error: Memory system not active."
    try:
        meta = {"timestamp": get_utc_timestamp(), "type": "manual"}
        await asyncio.to_thread(memory_system.add, text, meta)

        if graph_memory and llm_client:
            async def _extract_graph():
                try:
                    from ..core.agent import extract_json_from_text
                    prompt = f"Extract explicit entity relationships from this fact into a 'graph_triplets' array as objects with 'subject', 'predicate', and 'object' keys. Predicates MUST be uppercase verbs. Return ONLY JSON. Fact: {text}"
                    payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Graph Extractor. Output JSON."}, {"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
                    data = await asyncio.wait_for(  # §4O A-MAJOR-2: off-main + bounded (was untimed → 1200s worker default)
                        llm_client.chat_completion(payload, use_worker=True, is_background=True, off_main_only=True, task_label="smart-memory"),
                        timeout=_GRAPH_EXTRACT_TIMEOUT_S)
                    res = extract_json_from_text(data["choices"][0]["message"].get("content", ""), repair_truncated=True)
                    triplets = res.get("graph_triplets", [])
                    # §4M R2 MINOR-4: parity with the bus path — the
                    # round-1 removal filter + failure visibility landed
                    # only there; this legacy branch kept both defects.
                    try:
                        from ..utils.helpers import is_removal_triplet
                        triplets = [t for t in triplets
                                    if not is_removal_triplet(t)]
                    except Exception:
                        pass
                    if triplets:
                        await asyncio.to_thread(graph_memory.add_triplets, triplets)
                except Exception as e:
                    logger.warning(
                        "remember (legacy): background graph extraction "
                        "failed (fact stored, graph not enriched): %s", e)
            spawn_bg(_extract_graph(), name="graph-extract-legacy")

        return f"Memory stored: '{text}'"
    except Exception as e:
        return f"Error storing memory: {e}"

def _persist_audio_structure(memory_system, filename: str, stats, passages, gaps, now: str) -> None:
    """Outline record (one entry per window), the ordered transcript sidecar
    and a summary row that carries the opening words — for a recording
    transcribed from a sandbox file (the YouTube route has its own twin in
    `memory.youtube_ingest`)."""
    from ..memory.audio_ingest import format_timestamp as _fmt
    entries = [[1, f"{_fmt(s)}–{_fmt(e)}  {t[:80]}"] for s, e, t in passages]
    try:
        memory_system.set_document_outline(filename, {
            "filename": filename, "source": "audio", "entries": entries,
            "duration_s": float(getattr(stats, "total_seconds", 0.0) or 0.0),
            "transcribed_s": float(getattr(stats, "seconds", 0.0) or 0.0),
            "chunks": int(getattr(stats, "chunks", 0) or 0), "gaps": gaps, "at": now,
        })
    except Exception as _oe:  # noqa: BLE001
        logger.debug("outline not stored for %s: %s", filename, _oe)
    try:
        memory_system.set_document_text(filename, {
            "filename": filename, "route": "audio", "gaps": gaps,
            "duration_s": float(getattr(stats, "total_seconds", 0.0) or 0.0),
            "passages": [[s, e, t] for s, e, t in passages], "at": now,
        })
    except Exception as _te:  # noqa: BLE001
        logger.debug("transcript sidecar not stored for %s: %s", filename, _te)
    try:
        opening = " ".join(t for _s, _e, t in passages)[:600]
        summary = (
            f"[Document Summary: {filename}] Audio transcript: "
            f"{_fmt(getattr(stats, 'seconds', 0.0))} of "
            f"{_fmt(getattr(stats, 'total_seconds', 0.0))} transcribed in "
            f"{int(getattr(stats, 'windows', 0))} windows, {int(getattr(stats, 'chunks', 0))} "
            f"indexed chunks" + (f", gaps: {'; '.join(gaps)}" if gaps else "") +
            f". Opening words: {opening}… Read it in order with knowledge_base("
            f"action='transcript', filename='{filename}'); ask about it with action='query'."
        )
        memory_system.add(summary, {"type": "document_summary", "source": filename, "timestamp": now})
    except Exception as _se:  # noqa: BLE001
        logger.debug("summary row not stored for %s: %s", filename, _se)


_AUDIO_PREVIEW_CHARS = 3000


def _audio_success_message(filename: str, stats, passages, gaps) -> str:
    from ..memory.audio_ingest import format_timestamp as _fmt

    def _plural(n, word):
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    note = ""
    if getattr(stats, "truncated", False):
        note += " (TRUNCATED at the duration cap)"
    if getattr(stats, "skipped_windows", 0):
        note += f" ({_plural(stats.skipped_windows, 'window')} failed and were skipped)"
    if getattr(stats, "truncated_windows", 0):
        note += f" ({_plural(stats.truncated_windows, 'window')} cut short at the token budget)"
    gap_note = f" NOT transcribed: {'; '.join(gaps)}." if gaps else ""
    stopped = str(getattr(stats, "aborted", "") or "")
    if stopped:
        gap_note += (f" STOPPED EARLY: {stopped}. What was transcribed is kept; to redo the whole "
                     f"recording later: knowledge_base(action='forget', target='{filename}') then "
                     f"transcribe again.")
    head = "SUCCESS (partial)" if stopped else "SUCCESS"
    transcript = "\n".join(f"[{_fmt(s)}–{_fmt(e)}] {t}" for s, e, t in passages)
    preview = transcript[:_AUDIO_PREVIEW_CHARS]
    more = ""
    if len(transcript) > _AUDIO_PREVIEW_CHARS:
        more = (f"\n… [{len(transcript) - _AUDIO_PREVIEW_CHARS} more characters — read on with "
                f"knowledge_base(action='transcript', filename='{filename}', offset={_AUDIO_PREVIEW_CHARS})]")
    return (
        f"{head}: Transcribed and ingested '{filename}' — "
        f"{_fmt(getattr(stats, 'seconds', 0.0))} of {_fmt(getattr(stats, 'total_seconds', 0.0))} "
        f"transcribed, {_plural(getattr(stats, 'windows', 0), 'window')}, "
        f"{_plural(getattr(stats, 'chunks', 0), 'chunk')}{note}.{gap_note}\n"
        f"TRANSCRIPT ({min(len(transcript), _AUDIO_PREVIEW_CHARS)} of {len(transcript)} chars):\n"
        f"{preview}{more}\n"
        f"Passages carry timestamps, so answers can cite the moment. Ask questions with "
        f"knowledge_base(action='query', filename='{filename}', question='...'); the whole "
        f"text in order is action='transcript'."
    )


#: The heads an ingest uses to say it FAILED. None matched the shared failure
#: pattern, so every failed ingest was booked ok (§4LT M3).
_INGEST_FAILURE_HEADS = ("Ingest Error:", "Embedding Error:", "Disk Error:", "Web Error:")


def _declare_ingest(res):
    """Declare an ingest's outcome. The SHARED classifier first (it already
    knows fetch failures, the containment refusal, Tor errors), then the four
    ingest failure heads it misses → failed, and a partial/truncated ingest →
    partial (fix review N1: overriding the classifier mislabelled those). A
    result that already declared itself is kept."""
    from .outcome import ToolOutcome, OutcomeStatus
    if isinstance(res, ToolOutcome):
        return res
    t = str(res or "")
    head = t.lstrip()
    if head.startswith(_INGEST_FAILURE_HEADS):
        return ToolOutcome.failed(t, reason_code="kb_ingest_failed")
    out = ToolOutcome.coerce(t)
    if getattr(out, "status", None) is OutcomeStatus.OK and head.startswith("SUCCESS (PARTIAL)"):
        return ToolOutcome.partial(t, reason_code="kb_ingest_truncated")
    return out


def _norm_doc_name(name):
    """`./notes.txt`, `.//notes.txt` and `notes.txt` are ONE document (§4LT):
    used at ingest AND at every lookup (query/outline/transcript/forget), so
    the two can never disagree (fix review N4). URLs and absolute paths are
    left alone."""
    if not isinstance(name, str):
        return name
    s = name.strip()
    if "://" in s or not s.startswith("./"):
        return s or name
    while s.startswith("./"):
        s = s[2:].lstrip("/")
    return s or name


async def tool_gain_knowledge(filename: str = None, sandbox_dir: Path = None, memory_system=None,
                              tor_proxy: str = None, language: str = None):
    if not filename:
        return "SYSTEM ERROR: The 'filename' parameter is MANDATORY. You must specify it."
    filename = _norm_doc_name(filename)
    import time
    import fitz  # PyMuPDF
    import re

    # ULTRA-AGGRESSIVE SELF-HEALING: 
    # 1. Clean whitespace and carriage returns
    # 2. Extract only the first non-empty line
    # 3. Strip LLM artifacts like "Downloaded " or " (123 bytes)"
    raw_name = str(filename).replace('\r', '').strip()
    if '\n' in raw_name:
        raw_name = [line.strip() for line in raw_name.split('\n') if line.strip()][0]
    
    # Strip common prefixes and quotes
    # AWS/GHOST CLEANING PROTOCOL
    # Detect if the 'filename' is actually a sentence like "The text of 'Romeo...'"
    if " " in raw_name and len(raw_name.split()) > 3:
         # Try to extract a potential filename from quotes (e.g. 'romeo_source.txt')
         # We look for a pattern that ends in a common extension or is just a single word in quotes
         match = re.search(r"['\"`]+([\w\-\.]+\.[a-zA-Z]{2,4})['\"`]+", raw_name, re.IGNORECASE)
         if match:
             raw_name = match.group(1)
         else:
             # Fallback: Look for any single word in quotes that looks like a file
             match_loose = re.search(r"['\"`]+([\w\-\._]+)['\"`]+", raw_name, re.IGNORECASE)
             if match_loose and "." in match_loose.group(1):
                 raw_name = match_loose.group(1)

    raw_name = re.sub(r'^(Downloaded|File|Path|Document|Source|Text|Content|Of|The text of)\b\s*:?\s*', '', raw_name, flags=re.IGNORECASE)
    raw_name = raw_name.strip("'\"` ")
    
    # Strip parenthetical info (e.g., "file.pdf (1234 bytes)")
    raw_name = re.sub(r'\s*\([\d\s\w,]+\).*$', '', raw_name, flags=re.IGNORECASE)
    
    # ⚠ COERCE ONCE, AT ENTRY. `filename` is read with `.lower()` /
    # `.startswith()` at ten sites below; arguments arrive via json.loads, so
    # a list or int crashes the first of them with AttributeError, which the
    # loop renders as "did you forget a required argument?". Fixing the ten
    # READERS would be the wrong shape — one coercion at the source covers
    # every present and future reader.
    filename = str(raw_name).strip()

    # --- QWEN HALLUCINATION GUARD ---
    # If the filename starts with '#', 'Title:', or has no extension and spaces, reject it.
    if filename.startswith("#") or filename.lower().startswith("title:") or (" " in filename and "." not in filename):
        return f"Error: You passed the document CONTENT or TITLE ('{filename[:30]}...'). You MUST pass the FILENAME (e.g. 'romeo_source.txt')."

    # ── YOUTUBE: fetched over Tor, captions or audio, in-process (§4KE) ──
    # Before this branch a YouTube URL fell through to the generic web fetch,
    # which ingested the watch page's HTML shell as a "document" and reported
    # SUCCESS. Checked BEFORE the length and scheme rules: a schemeless
    # `youtu.be/<id>` and a 300-char watch URL with tracking params are both
    # links, not filenames.
    from ..memory.youtube_ingest import ingest_youtube, is_youtube_url
    if is_youtube_url(filename):
        def _yt_progress(msg: str) -> None:
            pretty_log("YouTube", msg, icon=Icons.MEM_INGEST)
        try:
            _dest = Path(sandbox_dir) if sandbox_dir else Path(tempfile.gettempdir())
            res = await asyncio.to_thread(
                ingest_youtube, filename, sandbox_dir=_dest, memory_system=memory_system,
                tor_proxy=tor_proxy, language=language, progress=_yt_progress,
            )
        except Exception as e:  # noqa: BLE001
            return f"Ingest Error: the YouTube route failed: {e}"
        return res.message

    # OS limit usually 255, we use 240 to be safe. (The old `> 2000` branch
    # below this was dead — `> 240` always returns first — and had a typo.)
    if len(filename) > 240:
        return f"Error: Filename is too long ({len(filename)} chars). Max length is 240 characters. Did you accidentally pass the content?"

    pretty_log("Ingesting Data", filename, icon=Icons.MEM_INGEST)
    if not memory_system: return "Error: Memory system is disabled."

    current_library = memory_system.get_library()
    if filename in current_library:
        return (f"Skipped: '{filename}' is already in KB. If the file has CHANGED since, "
                f"forget the STORED copy first (knowledge_base action='forget' target='{filename}'; "
                f"when the user confirms, pass only the number of the ingested-document item — "
                f"NOT 'all', which would also delete the file itself) and ingest it again.")

    is_web = filename.lower().startswith("http://") or filename.lower().startswith("https://")

    if is_web and filename.lower().split("?")[0].endswith(".pdf"):
        return ("Error: You cannot directly ingest a PDF URL. If you already downloaded it to the sandbox, "
                "pass the LOCAL FILENAME (e.g. 'document.pdf') instead of the URL. If you haven't downloaded "
                "it, use file_system(operation='download') first; if that tool reports the site refuses it, "
                "ask the user to provide the file — do not fetch it any other way.")

    full_text = ""
    if is_web:
        pretty_log("Fetching URL", filename, icon=Icons.TOOL_DOWN)
        try:
            full_text = await helper_fetch_url_content(filename)
            if full_text.startswith("Error"): return full_text 
        except Exception as e: return f"Web Error: {str(e)}"
    else:
        # ⚠ CONTAINMENT. This used to be a bare
        #     clean_name = str(filename).lstrip("/")
        #     file_path = sandbox_dir / clean_name
        # which contains an ABSOLUTE path (`/etc/passwd` -> `<sandbox>/etc/
        # passwd`, harmless) but NOT a relative one. `../../.ghost_api_key`
        # resolved straight out of the sandbox, and this branch then READ the
        # file and embedded its contents into durable vector memory — where
        # it is retrievable by `recall` forever — while returning "SUCCESS".
        # Verified end-to-end through the real tool, 2026-08-30 (§4DX).
        #
        # It is reachable from prompt injection: fetched web or darkweb
        # content enters the model's context, and the model's next tool call
        # is the payload. `knowledge_base` is on the low-risk list, so
        # nothing else in the stack was going to stop it.
        #
        # `_get_safe_path` is the project's containment helper (0 escapes on
        # a 15-payload fuzz, including `..`, encoded `..`, absolute paths and
        # a NUL byte) and raises ValueError on an escape.
        #
        # The `sandbox/` strip is kept and applied FIRST so the existing
        # healing still works. Order matters: `sandbox/../../etc/passwd`
        # becomes `../../etc/passwd` and is then REFUSED, rather than healed
        # into an escape.
        clean_name = str(filename).lstrip("/")
        if clean_name.startswith("sandbox/"):
            clean_name = clean_name[8:]
        try:
            file_path = _get_safe_path(sandbox_dir, clean_name)
        except ValueError as _ve:
            # Bare, NOT f"Error: {_ve}". The ~10 `file_system` sites return
            # `str(ve)` and classify REJECTED; prefixing "Error:" made this
            # match the FAILURE regex first, so one event had two statuses
            # depending on which tool raised it.
            return str(_ve)
        
        # --- ROBUST FILE RESOLUTION ---
        if not file_path.exists():
            # Try a case-insensitive match or search for the filename in the sandbox
            try:
                def _resolve_file():
                    import os
                    # Use a safe os.walk instead of unbounded rglob
                    all_files = []
                    for root_dir, dirs, fnames in os.walk(sandbox_dir):
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'env']]
                        for f in fnames:
                            if not f.startswith('.') and not f.lower().endswith(_INFLIGHT_SUFFIXES):
                                all_files.append(Path(root_dir) / f)
                    
                    # Priority 1: Exact name match (case-insensitive)
                    matches = [f for f in all_files if f.name.lower() == filename.lower()]
                    
                    # Priority 2: Stem match (e.g., "bitcoin" matches "bitcoin.pdf")
                    if not matches:
                        target_stem = Path(filename).stem.lower()
                        matches = [f for f in all_files if f.stem.lower() == target_stem]
                    
                    # Priority 3: Substring match
                    if not matches:
                        matches = [f for f in all_files if filename.lower() in f.name.lower() and f.is_file()]
                    
                    if matches:
                        return matches[0]
                    return None

                resolved_file_path = await asyncio.to_thread(_resolve_file)
                if resolved_file_path:
                    # ⚠ ASK THE GUARD AGAIN. The primary path above goes
                    # through `_get_safe_path`; this fallback re-derives a
                    # path for itself and used to return `matches[0]` raw.
                    # `os.walk` stays inside the sandbox, but a FILE it
                    # finds can be a SYMLINK pointing out of it — and the
                    # model plants that with one in-container `ln -s`.
                    #
                    # The result was that `ingest('notes.txt')` was refused
                    # while `ingest('notes')` — the same file, reached by
                    # stem match — read the host target and embedded it in
                    # durable memory. Demonstrated §4DX round 2, against a
                    # fix shipped hours earlier: the guard was correct and
                    # a second code path walked around it.
                    _resolved = resolved_file_path.resolve()
                    if not _is_within_root(_resolved, Path(sandbox_dir).resolve()):
                        return (f"Error: Security Error: '{filename}' resolves "
                                f"outside the sandbox (symbolic link). Refused.")
                    # ⚠ USE THE RESOLVED PATH. Checking `_resolved` and
                    # then reading `resolved_file_path` is a TOCTOU race:
                    # everything downstream re-follows the link at read
                    # time, and the model can run a swap loop concurrently
                    # (`execute` promotes long commands to detached jobs).
                    # Reading the already-resolved path closes the window.
                    file_path = _resolved
                    # `_resolved` is guaranteed inside the sandbox by the
                    # check above, so `relative_to` cannot raise here. It
                    # could before that check existed, and the bare `except`
                    # below then reported a containment refusal as "File not
                    # found" — the exact misdirection the refusal-message
                    # pin forbids.
                    filename = str(file_path.relative_to(
                        Path(sandbox_dir).resolve()))
                    pretty_log("KB Auto-Resolve", filename, icon=Icons.OK)
                    # Re-check the library under the RESOLVED name: the
                    # pre-check above ran on the raw argument, so
                    # ingest_document('postgresql-manual') sailed past it,
                    # resolved to 'postgresql-manual.pdf', and re-extracted
                    # + re-embedded an already-ingested 3k-page manual
                    # (hours of CPU; content-hashed ids mean no duplication,
                    # just pure wasted work).
                    if filename in current_library:
                        return (f"Skipped: '{filename}' is already in KB. If the file has CHANGED since, "
                f"forget the STORED copy first (knowledge_base action='forget' target='{filename}'; "
                f"when the user confirms, pass only the number of the ingested-document item — "
                f"NOT 'all', which would also delete the file itself) and ingest it again.")
                else:
                    # ⚠ SAY WHAT THIS ACTION DOES NOT DO. "Check list_files"
                    # was the wrong advice for the common case: the model
                    # called ingest on a file it had not fetched yet, believing
                    # one call downloads and indexes (2026-09-09, request
                    # d50a34bd — a wasted 40 s turn). The file was never there;
                    # listing the sandbox cannot help. Name the route that can.
                    # §4FS review: no "curl from the sandbox" fallback here — the
                    # sandbox's egress is cleartext from the host IP, and the
                    # agent's rule is Tor-only. The download tool now retries
                    # a refused site with a plain profile over Tor itself.
                    return (
                        f"Error: File '{filename}' not found in the sandbox. This action "
                        "never downloads — it reads a file that is ALREADY there. If the "
                        "file lives at a URL, fetch it first with "
                        f"file_system(operation='download', url='<the url>', path={filename!r}), "
                        f"then call again with filename={filename!r}. If the download tool "
                        "reports the site refuses it, ask the user to provide the file — do "
                        "not fetch it any other way. A YouTube link needs no download step: "
                        "pass the URL itself as filename and this action fetches and "
                        "transcribes it. If it should already exist, list_files "
                        "shows the exact name."
                    )
            except:
                return f"Error: File '{filename}' not found."
                
        # Hard caps for ingest. Without these a 1 GB PDF or text file in
        # the sandbox would OOM the host while the model thinks it's just
        # ingesting a document. (PDF page/char ceilings now live in
        # memory.pdf_ingest — raised so a real reference manual fits.)
        MAX_INGEST_FILE_BYTES = 100 * 1024 * 1024   # 100 MB on disk
        MAX_INGEST_TEXT_CHARS = 5_000_000           # 5 MB of extracted text (non-PDF)

        try:
            stat_res = file_path.stat()
            file_size = int(stat_res.st_size)
            # Audio/video are EXEMPT from the byte cap (2026-08-02). The cap
            # exists to stop a huge text/PDF being pulled into RAM — but a
            # recording never is: ffmpeg seeks to each ~12-minute window and
            # only that window's WAV is resident, so peak memory is flat
            # regardless of file size. The operative bound for a recording is
            # DURATION (GHOST_AUDIO_MAX_S, enforced in audio_ingest), not
            # bytes. Without this exemption the primary use case failed at the
            # door: a 45-minute conference talk is 700 MB–1.3 GB of video, and
            # the cap's advice ("Split it into chunks first") is useless for a
            # recording.
            is_audio_video = filename.lower().endswith(_AUDIO_INGEST_EXTS)
            if file_size > MAX_INGEST_FILE_BYTES and not is_audio_video:
                return (
                    f"Error: '{filename}' is {file_size // (1024*1024)} MB; ingest refuses files "
                    f"larger than {MAX_INGEST_FILE_BYTES // (1024*1024)} MB. Split it into chunks first."
                )
        except (TypeError, ValueError, AttributeError):
            # Mocked Path object in tests, or non-numeric stat — skip the cap.
            pass
        except OSError as se:
            return f"Disk Error: failed to stat '{filename}': {se}"

        # ── PDF: STREAMING, STRUCTURE-AWARE PATH (2026-07-13) ──────────
        # A reference manual (PostgreSQL: ~3k pages, ~10M chars) cannot go
        # through the whole-document path — it used to be refused outright
        # at 1000 pages, then silently halved at 5M chars, and it would
        # hold the full text + full chunk list + an enriched COPY in RAM.
        # pdf_ingest streams page→section→chunk→store in bounded memory and
        # stamps each chunk with its TOC breadcrumb ("19.5. Write Ahead
        # Log"), which raw PDF text otherwise loses entirely.
        if filename.lower().endswith(".pdf"):
            from ..memory.pdf_ingest import ingest_pdf_streaming

            def _progress(st):
                pretty_log(
                    "KB Ingest",
                    f"{filename}: {st.pages} pages · {st.chunks} chunks · "
                    f"{st.sections} sections",
                    icon=Icons.MEM_INGEST,
                )

            try:
                stats = await asyncio.to_thread(
                    ingest_pdf_streaming, file_path, filename, memory_system,
                    progress=_progress,
                )
            except Exception as e:
                return f"Ingest Error: {e}"

            if not stats.chunks:
                return "Error: Extracted text is empty."

            # STRUCTURE, persisted (request e0f4a8bd): the TOC was computed
            # to build the breadcrumbs and then dropped, leaving no route to
            # "how many chapters". Stored before the summary below so a
            # failure in the nice-to-have never costs the structure.
            try:
                await asyncio.to_thread(
                    memory_system.set_document_outline, filename,
                    {"filename": filename, "source": "toc",
                     "entries": [list(e) for e in (stats.outline or [])],
                     "pages": stats.pages_total or stats.pages,
                     "chunks": stats.chunks, "chars": stats.chars,
                     "at": get_utc_timestamp()},
                )
            except Exception as _oe:  # noqa: BLE001 — never fail an ingest for this
                logger.debug("outline not stored for %s: %s", filename, _oe)

            # Doc-level summary for "what's in X / summarise X" queries.
            try:
                summary = (
                    f"[Document Summary: {filename}] Reference document: "
                    f"{stats.pages} pages, {stats.sections} sections, "
                    f"{stats.chunks} indexed chunks, {stats.chars} characters. "
                    f"Query it with knowledge_base(action='query', "
                    f"filename='{filename}', question='...')."
                )
                await asyncio.to_thread(
                    memory_system.add, summary,
                    {"type": "document_summary", "source": filename,
                     "timestamp": get_utc_timestamp()},
                )
            except Exception:
                pass

            note = ""
            if stats.truncated:
                note += " (TRUNCATED at the text cap)"
            if stats.skipped_pages:
                note += f" ({stats.skipped_pages} unreadable pages skipped)"
            return (
                f"SUCCESS: Ingested '{filename}' — {stats.pages} pages, "
                f"{stats.sections} sections, {stats.chunks} chunks{note}. "
                f"Ask questions with knowledge_base(action='query', "
                f"filename='{filename}', question='...')."
            )

        # ── AUDIO / VIDEO: WINDOWED TRANSCRIPTION PATH (2026-08-02) ────
        # Spoken material (conference talks, podcast interviews, recorded
        # meetings) used to be unreachable: it would fall through to the
        # plain-text branch below and decode as replacement-char noise, so a
        # whole media class was invisible to the knowledge base. It is now
        # transcribed on the Gemma 4 audio node ~12 minutes at a time, and
        # every chunk is stamped with its TIMESTAMP RANGE so a retrieved
        # passage is citable back to the moment in the recording. Video
        # containers work too — ffmpeg simply takes the audio track.
        if filename.lower().endswith(_AUDIO_INGEST_EXTS):
            from ..memory.audio_ingest import (
                ingest_audio_streaming, format_timestamp,
            )

            # `X/60:.0f` minutes rounded a 3-second clip to "0 minutes, 1
            # windows" — a successful ingest that reads like a failure. Use the
            # h:mm:ss formatter (exact at every scale) and pluralise honestly.
            def _plural(n, word):
                return f"{n} {word}" if n == 1 else f"{n} {word}s"

            def _audio_progress(st):
                pretty_log(
                    "KB Ingest",
                    f"{filename}: {_plural(st.windows, 'window')} · "
                    f"{format_timestamp(st.seconds)} · "
                    f"{_plural(st.chunks, 'chunk')}",
                    icon=Icons.MEM_INGEST,
                )

            try:
                stats = await asyncio.to_thread(
                    ingest_audio_streaming, file_path, filename, memory_system,
                    progress=_audio_progress,
                )
            except Exception as e:
                return f"Ingest Error: {e}"

            if not stats.chunks:
                return (
                    f"Error: no speech was transcribed from '{filename}'. "
                    f"{'Windows failed: ' + '; '.join(stats.errors[:3]) if stats.errors else 'The recording may be silent or music-only.'}"
                )

            # Structure, ordered text and a CONTENT-bearing summary (§4KE):
            # the old summary row was counts only, no outline record was
            # written for audio, and the transcript could not be read in
            # order. Each store is best-effort and independent.
            _passages = list(getattr(stats, "transcript", []) or [])
            _gaps = list(getattr(stats, "gaps", []) or [])
            await asyncio.to_thread(
                _persist_audio_structure, memory_system, filename, stats, _passages, _gaps,
                get_utc_timestamp())
            return _audio_success_message(filename, stats, _passages, _gaps)

        try:
            def _extract_text():
                # NOTE: PDFs and audio never reach here — they take the
                # streaming pdf_ingest / audio_ingest paths above. This is the
                # plain-text branch.
                extracted_parts: list[str] = []
                running_len = 0
                binary_exts = ['.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz', '.sqlite', '.db', '.mp4', '.exe']
                if any(filename.lower().endswith(ext) for ext in binary_exts):
                    raise ValueError("Cannot ingest binary or media files into text memory.")
                # Stream the file in chunks rather than `f.read()` so we
                # can enforce the text-size cap without materialising the
                # whole file in memory first.
                # utf-8-sig strips a BOM (a leading U+FEFF would otherwise
                # pollute the first chunk/embedding); errors="replace" keeps
                # a non-UTF-8 file's mangling visible rather than silently
                # dropped by errors="ignore".
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    while running_len < MAX_INGEST_TEXT_CHARS:
                        chunk = f.read(min(65536, MAX_INGEST_TEXT_CHARS - running_len))
                        if not chunk:
                            break
                        extracted_parts.append(chunk)
                        running_len += len(chunk)
                    # If there's more, peek to see if the file kept going.
                    if f.read(1):
                        extracted_parts.append("\n[... INGEST TRUNCATED at 5 MB of extracted text ...]")
                return "".join(extracted_parts)
            full_text = await asyncio.to_thread(_extract_text)
        except Exception as e: return f"Disk Error: {str(e)}"

    if not full_text or not full_text.strip(): return "Error: Extracted text is empty."

    pretty_log("KB Split", f"{len(full_text)} chars", icon=Icons.MEM_SPLIT)
    # Use semantic chunking for structured content (markdown, code), falling
    # back to recursive splitting for plain text. Chunk size 600 prevents
    # silent truncation by all-MiniLM-L6-v2's 256 token limit.
    # off the event loop: a splitter fault must never freeze every request
    # (§4LT CRIT — it looped forever on one ordinary web page)
    chunks = await asyncio.to_thread(semantic_split_text, full_text, 600, 100)
    if not chunks: return "Error: No chunks created."

    pretty_log("KB Embed", f"{len(chunks)} fragments", icon=Icons.MEM_EMBED)
    try:
        # Offload ingestion to vector system logic (which now handles enrichment and batching).
        # ingest_document RETURNS (ok, msg) — it swallows internal Chroma/embedding
        # failures and returns (False, err) rather than raising, so a failed ingest
        # must be caught HERE or the tool falsely reports SUCCESS with nothing stored.
        _ingest_res = await asyncio.to_thread(memory_system.ingest_document, filename, chunks)
        if isinstance(_ingest_res, tuple) and _ingest_res and not _ingest_res[0]:
            return f"Embedding Error: {_ingest_res[1] if len(_ingest_res) > 1 else 'ingest failed'}"
        preview = full_text[:300].replace("\n", " ") + "..."
    except Exception as e: return f"Embedding Error: {e}"

    try: await asyncio.to_thread(memory_system._update_library_index, filename, "add")
    except asyncio.CancelledError: raise
    except Exception as e: logger.warning("library-index add failed for %s: %s", filename, e)

    # Generate a document-level summary for broad retrieval.
    # When users ask "what's in document X?" or "summarize the report",
    # chunk-level retrieval returns fragments. A doc-level summary gives
    # the global picture. Stored as type=document_summary.
    try:
        # Use first 3000 chars as representative sample for summary
        sample = full_text[:3000].replace("\n", " ").strip()
        if len(sample) > 200:
            doc_summary = (
                f"[Document Summary: {filename}] "
                f"This document contains {len(chunks)} sections across {len(full_text)} characters. "
                f"Content preview: {sample[:500]}..."
            )
            await asyncio.to_thread(
                memory_system.add, doc_summary,
                {"type": "document_summary", "source": filename, "timestamp": get_utc_timestamp()}
            )
            pretty_log("KB Summary", f"Generated document summary for {filename}", icon=Icons.MEM_SAVE)
    except Exception:
        pass  # Non-critical; chunks are already ingested

    if full_text.rstrip().endswith(("[... INGEST TRUNCATED at 5 MB of extracted text ...]",
                                    "[... TRUNCATED at 5 MB ceiling ...]")):
        # say so — the PDF path always did (§4LT minor)
        return (f"SUCCESS (PARTIAL): Ingested the first 5 MB of '{filename}' — the rest "
                f"was NOT stored; answers cannot come from beyond that point.")
    return f"SUCCESS: Ingested '{filename}'."

#: RAW VECTOR DISTANCE bands, measured on the live PostgreSQL manual
#: (8,279 chunks, bge-small-en-v1.5) on 2026-09-09 — the best hit of each
#: query class:
#:
#:   0.222  "pg_stat_activity columns"                  (answerable)
#:   0.242  "what does wal_level control"               (answerable)
#:   0.268  "how does VACUUM FULL differ from VACUUM"   (answerable)
#:   ------------------------------------------------- 0.34 —
#:   0.347  "Table of Contents: list every Part"        (STRUCTURAL: no
#:   0.376  "how many top-level numbered chapters"       passage can answer)
#:   0.407  "what is the offside rule in football"      (off-topic)
#:   0.480  "how do I bake sourdough bread"             (off-topic)
#:
#: The floor changes only the ADVICE in the footer, never which passages are
#: returned, so a mis-set band costs a sentence rather than an answer. It is
#: env-overridable because the scale is the embedder's, not a law of nature.
_DIST_STRONG = float(os.environ.get("GHOST_KB_DIST_STRONG", "0.30"))
_DIST_WEAK = float(os.environ.get("GHOST_KB_DIST_WEAK", "0.34"))


#: The sentence a fruitless document search prints, and the ONE home for
#: it: the loop-breaker imports this constant to recognise a probe that
#: found nothing, so the banner cannot be reworded without the breaker
#: following (the §4FN `FALLBACK_HEADS` shape). It is real prose the model
#: reads, not a hidden token — a marker the reader cannot see is a marker
#: the reader cannot act on.
KB_NO_ANSWER_MARKER = "NOTHING IN THIS DOCUMENT ANSWERS THIS"


def _match_word(dist: float) -> str:
    """A distance in the document's own terms. The bare number was
    reported as "relevance", which inverts it — lower is CLOSER, and the
    rank key it was taken from can even be negative."""
    if dist < _DIST_STRONG:
        return "strong"
    if dist < _DIST_WEAK:
        return "moderate"
    return "weak"


def _match_library_name(filename, library):
    """``(match, ambiguity_error)`` for a document name the model typed.
    An exact (case-insensitive) name wins; otherwise the documents whose
    name minus extension equals it — ONE is used, SEVERAL are an error that
    lists them (§4LT: "q3" silently picked q3.md when the answer was in
    q3.txt). Shared by query, outline and transcript."""
    want = str(filename or "").lower()
    exact = [f for f in library if f.lower() == want]
    if exact:
        return exact[0], None
    stem = want.rsplit(".", 1)[0]
    by_stem = [f for f in library if f.lower().rsplit(".", 1)[0] == stem]
    if len(by_stem) > 1:
        return None, (f"Error: '{filename}' matches several documents: {by_stem}. "
                      f"Pass one exact name.")
    return (by_stem[0] if by_stem else None), None


async def tool_query_document(filename: str = None, question: str = None,
                              memory_system=None, k: int = 8):
    """Ask a question against ONE ingested document (2026-07-13).

    The missing half of the RAG loop. Ingest existed; retrieval did not —
    the only way a chunk reached the model was ambient hydration, where it
    competed with episodes/skills for a shared budget and was capped at 12
    fragments from the whole store. This returns the k best passages from
    the NAMED document as TOOL OUTPUT, so the model reads them directly and
    can iterate (search → read → refine → search again).
    """
    if not filename or not question:
        # Placeholders are marked so nothing in the example can be read as a
        # parameter name — the same discipline `_kb_target_or_error` enforces
        # for the branches it owns.
        return ("SYSTEM ERROR: both 'filename' and 'question' are MANDATORY "
                "for action='query'. Worked call: knowledge_base("
                "action='query', filename='<an ingested document>', "
                "question='<your question about it>')")
    if not memory_system:
        return "Error: Memory system is disabled."

    library = await asyncio.to_thread(memory_system.get_library)
    library = library or []
    if filename not in library:
        # Forgiving match: the model often passes a stem or a near-miss.
        match, _amb = _match_library_name(filename, library)
        if _amb:
            return _amb
        if not match:
            return (f"Error: '{filename}' is not in the knowledge base. "
                    f"Available documents: {library or '(none)'}. "
                    f"Ingest it first with action='ingest_document'.")
        filename = match

    pretty_log("KB Query", f"{filename} ← {question[:60]}", icon=Icons.MEM_READ)
    try:
        hits = await asyncio.to_thread(
            memory_system.search_document, filename, question, k=k)
    except Exception as e:
        return f"Error: document query failed: {e}"

    if not hits:
        return (f"No passages found in '{filename}' for that question. "
                f"Try rephrasing with the document's own terminology.")

    # The BEST raw vector distance decides whether this document contains an
    # answer at all. Never the rank key (see `search_document`) — it is
    # BM25-adjusted, and keyword overlap is exactly what a hopeless
    # structural query has plenty of.
    dists = [float(h.get("dist", h.get("score", 0.0))) for h in hits]
    best = min(dists) if dists else 1.0

    parts = [
        f"PASSAGES FROM '{filename}' ({len(hits)} closest; best match is "
        f"{_match_word(best)}, distance {best:.2f} — under {_DIST_STRONG:.2f} "
        f"a passage answers, over {_DIST_WEAK:.2f} it is unrelated; LOWER IS "
        f"CLOSER):",
    ]
    if best >= _DIST_WEAK:
        # ⚠ NOT "query again with different wording". Request e0f4a8bd: the
        # question was "how many chapters", every passage came back at 0.35+
        # (the off-topic band), and that footer told the model to keep
        # searching. It obeyed 10+ times over five minutes, adding ~10 KB of
        # unrelated passages to its context each time. No wording retrieves
        # a count, and telling a model to retry a search that cannot
        # succeed is an instruction with no exit.
        parts += [
            f"⚠ {KB_NO_ANSWER_MARKER}. Every passage above is in the "
            "unrelated band, so RE-WORDING THIS SEARCH WILL NOT HELP — do "
            "not try. Instead:",
            "  • a question about the document's STRUCTURE (how many "
            "chapters/parts/sections, what the chapters are, what section N "
            "covers) is knowledge_base(action='outline', filename="
            f"'{filename}') — semantic search cannot count;",
            "  • otherwise the document simply may not cover it: say so, or "
            "answer from another source, and say where the answer came from.",
        ]
    else:
        parts += [
            "Answer the user's question FROM THESE PASSAGES. Cite the section "
            "breadcrumb shown in each passage's header. If they do not "
            "contain the answer, ONE more query with the document's own "
            "terminology is worth it — but if the next one is no closer, "
            "stop searching and say so (for structure, use action='outline').",
        ]
    parts.append("")
    for i, h in enumerate(hits, 1):
        d = float(h.get("dist", h.get("score", 0.0)))
        parts.append(f"--- [{i}] ({_match_word(d)} match, distance {d:.2f}) "
                     f"---\n{h['text']}")
    return "\n".join(parts)


#: One page of an ordered transcript (`action='transcript'`).
TRANSCRIPT_PAGE_CHARS = 12000


async def tool_document_transcript(filename: str = None, offset=0, max_chars=TRANSCRIPT_PAGE_CHARS,
                                   memory_system=None):
    """The ORDERED, timestamped text of a transcribed recording or video,
    one page at a time (§4KE).

    Chunks are a retrieval structure — no ordinal, arbitrary order out of
    the store — so "give me the transcript" and "summarise the whole talk"
    had no route: the model could hold k=8 passages and call that the
    summary. The ingest now keeps the ordered passages beside the store
    (`set_document_text`), and this pages through them.
    """
    if not filename:
        return ("SYSTEM ERROR: 'filename' is MANDATORY for action='transcript'. Worked call: "
                "knowledge_base(action='transcript', filename='<a transcribed document>')")
    if not memory_system:
        return "Error: Memory system is disabled."
    try:
        offset = max(0, int(offset or 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        max_chars = max(500, min(int(max_chars or TRANSCRIPT_PAGE_CHARS), 60000))
    except (TypeError, ValueError):
        max_chars = TRANSCRIPT_PAGE_CHARS

    library = await asyncio.to_thread(memory_system.get_library)
    library = library or []
    if filename not in library:
        from ..memory.youtube_ingest import existing_document_for, youtube_video_id
        vid = youtube_video_id(filename)
        match = existing_document_for(vid, library) if vid else None
        stem = ""
        if not match:
            stem = str(filename).lower().rsplit(".", 1)[0]
            match, _amb = _match_library_name(filename, library)
            if _amb:
                return _amb
        if not match and stem:
            # A bare prefix is accepted only when it names ONE document.
            starts = [f for f in library if f.lower().startswith(stem)]
            if len(starts) == 1:
                match = starts[0]
            elif len(starts) > 1:
                return (f"Error: '{filename}' matches several documents: {starts}. "
                        f"Pass one exact name.")
        if not match:
            return (f"Error: '{filename}' is not in the knowledge base. "
                    f"Available documents: {library or '(none)'}.")
        filename = match

    rec = await asyncio.to_thread(memory_system.get_document_text, filename)
    passages = (rec or {}).get("passages") or []
    if not passages:
        return (f"Error: '{filename}' has no ordered transcript stored (documents transcribed "
                f"before 2026-09-24, PDFs and text files keep only searchable passages). Use "
                f"action='query' with a question, or action='outline' for its structure.")
    from ..memory.audio_ingest import format_timestamp as _fmt
    lines = []
    for p in passages:
        try:
            s, e, t = float(p[0]), float(p[1]), str(p[2])
        except (TypeError, ValueError, IndexError):
            continue
        lines.append(f"[{_fmt(s)}–{_fmt(e)}] {t}")
    text = "\n".join(lines)
    total = len(text)
    if offset >= total:
        return (f"TRANSCRIPT of '{filename}': {total} chars in total; offset {offset} is past the end.")
    page = text[offset:offset + max_chars]
    head = [f"TRANSCRIPT of '{filename}'"]
    title = (rec or {}).get("title")
    if title:
        head.append(f"— \"{title}\"")
    if (rec or {}).get("url"):
        head.append(f"({rec['url']})")
    gaps = (rec or {}).get("gaps") or []
    parts = [" ".join(head) + f" — chars {offset}–{offset + len(page)} of {total}."]
    if gaps:
        parts.append(f"Not transcribed: {'; '.join(str(g) for g in gaps)}.")
    parts.append("")
    parts.append(page)
    if offset + len(page) < total:
        parts.append(f"\n… [{total - offset - len(page)} more characters: knowledge_base("
                     f"action='transcript', filename='{filename}', offset={offset + len(page)})]")
    else:
        parts.append("\n[end of transcript]")
    return "\n".join(parts)


#: How many rendered outline lines a single `outline` call may return. The
#: PostgreSQL manual has ~6,200 outline entries; dumping them would cost
#: more context than the answer is worth, and the COUNTS above the tree
#: already answer "how many".
_OUTLINE_MAX_LINES = 120


#: A structural LABEL is a word followed by its ENUMERATOR and a closing
#: mark — "Part I.", "Chapter 12.", "Appendix F." — which is a SHAPE, not a
#: vocabulary: nothing to go stale on a document that says "Book" or
#: "Annex". The enumerator admits digits, roman numerals AND a bare letter
#: (the PostgreSQL manual's appendices are lettered A–P, and a roman-only
#: rule counted the 5 whose letter happens to be a roman numeral, reporting
#: "5 Appendix" for a document with 15). The trailing `.`/`:`/`)`/end is
#: what keeps prose out: "See Also", "DROP TABLE", "Note A brief summary"
#: all fail it.
_OUTLINE_LABEL_RE = re.compile(
    r"^([A-Za-z][A-Za-z-]{2,})\s+(?:\d+|[IVXLCDM]+|[A-Z])\s*(?:[.:)]|$)")


def _outline_labels(entries) -> dict:
    """``{level: {label: count}}`` over the WHOLE outline.

    This is what answers "how many chapters". The per-level count does not:
    measured on the live PostgreSQL manual, level 2 holds 91 entries — 70
    chapters, 15 appendices and 6 front-matter headings — so reporting the
    level count as the chapter count would have answered 91 to a question
    whose true answer is 70.
    """
    out: dict = {}
    for row in entries or ():
        try:
            lvl, title = int(row[0]), str(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        m = _OUTLINE_LABEL_RE.match(title.strip())
        if m:
            out.setdefault(lvl, {})
            label = m.group(1)
            out[lvl][label] = out[lvl].get(label, 0) + 1
    return out


def _outline_level_counts(entries) -> dict:
    counts: dict = {}
    for row in entries or ():
        try:
            lvl = int(row[0])
        except (TypeError, ValueError, IndexError):
            continue
        counts[lvl] = counts.get(lvl, 0) + 1
    return counts


def render_document_outline(record: dict, depth: int = 2) -> str:
    """The stored outline record → what a model reads.

    Leads with the LEVEL COUNTS, because the question this exists for is
    "how many chapters" and a count is the answer; the tree underneath is
    what lets the model tell which level "chapter" means (the titles say so
    themselves in any real manual).
    """
    entries = record.get("entries") or []
    counts = _outline_level_counts(entries)
    name = record.get("filename") or "the document"
    max_level = max(counts) if counts else 0
    depth = max(1, min(int(depth or 2), max_level or 1))

    head = [f"OUTLINE OF '{name}'"]
    facts = []
    if record.get("pages"):
        facts.append(f"{record['pages']} pages")
    if record.get("chunks"):
        facts.append(f"{record['chunks']} indexed chunks")
    if max_level:
        facts.append(f"{max_level} levels deep")
    if facts:
        head.append(" — " + " · ".join(facts))
    out = ["".join(head)]

    if not entries:
        out.append(
            "This document has NO table of contents (a plain-text ingest, or "
            "a PDF without an outline), so there is no structure to report. "
            "Use action='query' for its contents.")
        return "\n".join(out)

    labels = _outline_labels(entries)
    if labels:
        out.append("HOW MANY — exact counts of the divisions the document "
                   "names itself, over the WHOLE outline:")
        for lvl in sorted(labels):
            top = sorted(labels[lvl].items(), key=lambda kv: -kv[1])[:3]
            out.append("  level %d: " % lvl + " · ".join(
                f"{n} \u00d7 {name}" for name, n in top))
    out.append("Entries per level, INCLUDING unlabelled ones (front matter, "
               "indexes, prose headings):")
    out.append("  " + " · ".join(
        f"level {lvl}: {counts[lvl]}" for lvl in sorted(counts)))
    if record.get("source") == "breadcrumbs":
        out.append("(Rebuilt from the stored section breadcrumbs, so page "
                   "numbers are absent and the order is the titles' own "
                   "numbering.)")
    out.append("")
    out.append(f"Levels 1-{depth} of {max_level}"
               + (f" (call again with depth={depth + 1} for more):"
                  if depth < max_level else ":"))

    shown = 0
    for row in entries:
        try:
            lvl, title, page = int(row[0]), str(row[1]), int(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if lvl > depth:
            continue
        if shown >= _OUTLINE_MAX_LINES:
            out.append(f"… {sum(counts[l] for l in counts if l <= depth) - shown} "
                       f"more entries at this depth — the per-level counts "
                       f"above are complete; narrow with depth=1.")
            break
        out.append("  " * (lvl - 1) + title + (f"  (p. {page})" if page else ""))
        shown += 1
    return "\n".join(out)


async def tool_document_outline(filename: str = None, memory_system=None,
                                depth: int = 2, **kwargs):
    """The STRUCTURE of one ingested document — parts, chapters, sections.

    WHY THIS EXISTS (request e0f4a8bd, 2026-09-08). "How many chapters does
    the PostgreSQL manual have?" is a question about shape, and the only
    retrieval on offer was semantic: eight passages at relevance 0.08, and a
    footer saying "query again with different wording". The agent obeyed it
    for five minutes over 20+ turns and never could have succeeded — no
    wording retrieves a count. The structure existed the whole time.

    Cached after the first call: an ingest stores it exactly (page numbers
    included); a document ingested before that is rebuilt once from the
    breadcrumbs its own chunks carry.
    """
    if not filename:
        return ("SYSTEM ERROR: 'filename' is MANDATORY for action='outline'. "
                "Worked call: knowledge_base(action='outline', "
                "filename='<an ingested document>')")
    if not memory_system:
        return "Error: Memory system is disabled."

    library = await asyncio.to_thread(memory_system.get_library)
    library = library or []
    if filename not in library:
        match, _amb = _match_library_name(filename, library)
        if _amb:
            return _amb
        if not match:
            return (f"Error: '{filename}' is not in the knowledge base. "
                    f"Available documents: {library or '(none)'}. "
                    f"Ingest it first with action='ingest_document'.")
        filename = match

    try:
        record = await asyncio.to_thread(
            memory_system.get_document_outline, filename)
    except Exception:  # noqa: BLE001
        record = {}
    if not record:
        pretty_log("KB Outline", f"{filename} ← rebuilding from breadcrumbs",
                   icon=Icons.MEM_READ)
        try:
            record = await asyncio.to_thread(
                memory_system.derive_document_outline, filename)
        except Exception as e:  # noqa: BLE001
            return f"Error: could not read the outline of '{filename}': {e}"
        if record:
            try:
                await asyncio.to_thread(
                    memory_system.set_document_outline, filename, record)
            except Exception:  # noqa: BLE001 — caching is best-effort
                pass
    if not record:
        return (f"Error: '{filename}' has no indexed chunks, so it has no "
                f"readable structure. Re-ingest it with "
                f"action='ingest_document'.")

    try:
        depth = int(depth)
    except (TypeError, ValueError):
        depth = 2
    pretty_log("KB Outline", f"{filename} (depth {depth})", icon=Icons.MEM_READ)
    return render_document_outline(record, depth=depth)


#: Relevance grades in order of goodness (lower rank = better match).
_RELEVANCE_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# §4HB — graph-tier relevance guard. Content words of a query: folded
# (casefold + NFKD, combining marks stripped, so Greek matches Greek),
# longer than three characters, not a stopword, and NOT a bare number —
# a year is a date node's best friend and nobody's topic.
_GRAPH_STOPWORDS = frozenset("""
the a an of and or to in for on with by from at is are was were be been
being as that this these those it its into about over under how what
when where which who why not no do does did can could should would will
may might must i you he she they we us our your their his her
""".split())


def _graph_fold(text: str) -> str:
    import unicodedata
    folded = unicodedata.normalize("NFKD", (text or "").casefold())
    return "".join(c for c in folded if not unicodedata.combining(c))


def _graph_query_terms(query: str):
    import re as _re
    return [w for w in _re.findall(r"\w+", _graph_fold(query), _re.UNICODE)
            if len(w) > 3 and w not in _GRAPH_STOPWORDS and not w.isdigit()]


def _graph_edges_on_topic(query: str, edges):
    """Keep only edges that share at least one content word with the query.

    Per EDGE, not per section: the live recall had one edge that matched
    ("REQUESTED" ⊃ "request") and fourteen that matched nothing but the
    year. With no content words in the query nothing can be judged, and
    the edges pass through unchanged.
    """
    terms = _graph_query_terms(query)
    if not terms:
        return list(edges or [])
    kept = []
    for e in edges or []:
        folded = _graph_fold(str(e))
        if any(t in folded for t in terms):
            kept.append(e)
    return kept


async def tool_recall(query: str = None, memory_system=None, graph_memory=None, **kwargs):
    _blocked = _member_block()
    if _blocked is not None:
        return _blocked
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."
    pretty_log("Memory Recall", query, icon=Icons.MEM_READ)
    if not memory_system: return "Error: Memory system is disabled."
    try:
        # Use a higher limit for initial search, then filter strictly
        results = await asyncio.to_thread(memory_system.search_advanced, query, limit=10)
    except asyncio.CancelledError:
        raise  # a cancelled turn must propagate, not read as "recall failed"
    except Exception:
        return "Error: Memory retrieval failed."

    valid_chunks = []
    best_relevance = None
    for res in results:
        score = res.get('score', 1.0)
        source = res.get('metadata', {}).get('source', 'Unknown')
        text = res.get('text', '')
        m_type = res.get('metadata', {}).get('type', 'auto')
        
        # RAG-TUNED THRESHOLDS FOR ASYMMETRIC SEARCH
        if score < 0.8: relevance = "HIGH"
        elif score < 1.15: relevance = "MEDIUM"
        else: relevance = "LOW"
        
        pretty_log("Memory Match", f"[{relevance}] {score:.2f} | {source}", icon=Icons.MEM_MATCH)

        # 1.35 is a realistic upper bound for short queries against long chunks using L2 distance
        if score < 1.35:
            # §4FD: the relevance grade used to reach only the operator's
            # log — the model saw "highly relevant memories" over rows
            # scored LOW and invented a project codename from pg_stat
            # notes. The grade now rides each chunk, and the header names
            # the best one so an all-LOW recall reads as what it is.
            if best_relevance is None or _RELEVANCE_RANK[relevance] < _RELEVANCE_RANK[best_relevance]:
                best_relevance = relevance
            chunk = f"SOURCE: {source}\nRELEVANCE: {relevance} (distance {score:.2f})\nCONTENT: {text}"
            # Drill-down provenance: syntheses carry {"provenance": [{id,
            # excerpt}, ...]} (their merged sources are deleted, the excerpt
            # IS the surviving evidence); episode-derived skills carry
            # source_refs ("ep:12,ep:15") resolvable via episodic memory.
            meta = res.get('metadata', {}) or {}
            prov_raw = meta.get('provenance')
            if prov_raw:
                try:
                    import json as _json
                    _prov = _json.loads(prov_raw)
                    _ex = "; ".join(
                        f"\"{str(p.get('excerpt', ''))[:60]}\"" for p in _prov[:3]
                    )
                    chunk += f"\nEVIDENCE (synthesized from {len(_prov)} fragments): {_ex}"
                except Exception:
                    pass
            refs = meta.get('source_refs')
            if refs:
                chunk += f"\nEVIDENCE REFS: {refs}"
            # §4HB: an EPISODE hit is the REQUEST text of a past turn — the
            # vector store holds only that (460 chars for the turn that
            # matters below). What that turn found lives in the episode
            # record, reachable through `ep:<id>`, and this renderer only
            # ever offered that route for `source_refs`. Live: a 0.20-distance
            # hit on "Find the Revolut notification screenshot… say so
            # explicitly if no sender is visible" was rendered as the question,
            # and the answer (the card shows no sender field) stayed one
            # unoffered call away through two reruns.
            _ep_id = meta.get('episode_id')
            if _ep_id not in (None, "") and not refs:
                chunk += (f"\nEVIDENCE REFS: ep:{_ep_id} — this hit is a past "
                          f"REQUEST; expand it to read what that turn found")
            valid_chunks.append(chunk)
            
    if graph_memory:
        import re as _re
        words = [w.strip('.,?!;"\'()[]') for w in str(query).split() if len(w.strip('.,?!;"\'()[]')) > 3]
        if words:
            try:
                edges = await asyncio.to_thread(graph_memory.get_neighborhood, words, 15)
                # §4HB: keep an edge only if it shares a CONTENT word with
                # the query. The neighbourhood lookup matches any word over
                # three characters, so the year token "2026" pulled in every
                # edge ending in a 2026 date — July news headlines, the
                # operator's family profile — and put fifteen of them at the
                # TOP of a Revolut recall. Corpus: 111 of 336 graph edges
                # shown (33%) share no non-numeric content word with their
                # query. Guarded at the boundary, whatever the lookup does.
                edges = _graph_edges_on_topic(query, edges or [])
                # the owner's facts asked for by their KIND ("my health
                # conditions") — matched on the predicate, not a node (§4KX r8)
                if hasattr(graph_memory, "owner_facts_matching"):
                    _own = await asyncio.to_thread(graph_memory.owner_facts_matching, query)
                    if isinstance(_own, list):
                        edges = list(dict.fromkeys([e for e in _own if isinstance(e, str)] + list(edges)))
                if edges:
                    valid_chunks.insert(0, "### TOPOLOGICAL GRAPH EDGES:\n" + "\n".join(edges))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("recall graph tier skipped: %s", e)
            
    if valid_chunks:
        _best = best_relevance or "LOW"
        if _best == "LOW":
            _head = (f"SYSTEM: Found {len(valid_chunks)} memories (best match: LOW — "
                     "these are probably UNRELATED to the query; do not present them "
                     "as facts about it).")
        else:
            _head = f"SYSTEM: Found {len(valid_chunks)} memories (best match: {_best})."
        out = _head + "\n\n" + "\n\n".join(valid_chunks)
        # Iterative drill-down affordance: when a hit carries evidence
        # handles, tell the model how to expand them (the query_document
        # "read → refine → read again" loop, generalized to memory).
        if "EVIDENCE REFS:" in out or "EVIDENCE (synthesized" in out:
            out += (
                "\n\nTIP: to inspect the raw evidence behind a memory above, call "
                "knowledge_base(action='expand', ref='ep:<id>') for an episode ref, "
                "or refine this recall with more specific wording."
            )
        return out
    else:
        return (
            "SYSTEM OBSERVATION: Zero high-confidence memories found for this query. "
            "Before concluding the memory doesn't exist, try ONE more recall with "
            "different wording (a synonym, or just the key entity's name)."
        )

async def tool_expand_evidence(ref=None, episodic_memory=None,
                               session_store=None, **kwargs):
    """Drill down from an EVIDENCE REF (surfaced by `recall`) to the raw
    record behind an abstraction — episode-strategy lessons carry
    ``ep:<id>`` refs, session hits carry ``session:<id>``. This is the
    memory-store counterpart of tool_query_document's iterative loop."""
    if not ref:
        return ("SYSTEM ERROR: The 'ref' parameter is MANDATORY — pass an "
                "evidence handle from a recall hit's EVIDENCE REFS line. "
                "Worked call: knowledge_base(action='expand', "
                "ref='<ep:12, or session:the-id>').")
    ref = str(ref).strip()
    pretty_log("Evidence Expand", ref, icon=Icons.MEM_READ)

    if ref.startswith("ep:"):
        if not episodic_memory:
            return "Error: Episodic memory is disabled."
        try:
            ep_id = int(ref.split(":", 1)[1])
        except (ValueError, IndexError):
            return (f"Error: malformed episode ref '{ref}' — expected the "
                    f"form '<ep:12>'.")
        ep = await asyncio.to_thread(episodic_memory.get_episode, ep_id)
        if not ep:
            return (f"Error: episode {ep_id} no longer exists (episodes are "
                    f"capped at 500 and old ones are evicted).")
        lines = [
            f"EPISODE {ep_id} [{ep.get('cluster_id') or 'general'}]",
            f"TRIGGER: {ep.get('trigger', '')}",
        ]
        if ep.get("context"):
            lines.append(f"CONTEXT: {str(ep['context'])[:500]}")
        lines.append(
            f"OUTCOME ({'SUCCESS' if ep.get('outcome_success') else 'FAILURE'}): "
            f"{ep.get('outcome', '')}"
        )
        if ep.get("lesson"):
            lines.append(f"LESSON: {ep['lesson']}")
        for i, a in enumerate(ep.get("actions") or [], 1):
            ok = "ok" if a.get("success", 1) else "FAILED"
            lines.append(
                f"  {i}. {a.get('tool_name', '?')}({str(a.get('tool_args', ''))[:120]}) "
                f"→ [{ok}] {str(a.get('result', ''))[:150]}"
            )
        return "\n".join(lines)

    if ref.startswith("session:"):
        if not session_store:
            return "Error: Session store is unavailable."
        sid = ref.split(":", 1)[1].strip()
        sess = await asyncio.to_thread(session_store.get, sid)
        if not sess:
            return f"Error: session '{sid}' not found (it may have been evicted)."
        tail = (sess.messages or [])[-10:]
        lines = [f"SESSION {sid} — {sess.title or 'untitled'} (last {len(tail)} messages):"]
        lines += [f"{m.get('role', '?')}: {str(m.get('content', ''))[:200]}" for m in tail]
        return "\n".join(lines)

    return (f"Error: unknown ref scheme '{ref}' — supported: '<ep:12>' "
            f"(episode from EVIDENCE REFS) and '<session:the-id>'.")


# ── forget: preview, then confirm ───────────────────────────────────────────
# The re-reviews kept finding wrong deletions in a forget that GUESSED what the
# user meant ("Fotini's birthday" took the marriage edge; another person's
# facts were kept as "yours"; Greek names were never found). The model-facing
# forget is therefore two steps: a PREVIEW that runs the same sweep in plan
# mode — every destructive call recorded as a numbered item, nothing deleted —
# and, in a LATER turn after the user confirms, an EXECUTE of exactly the
# chosen items. Direct callers of `tool_unified_forget` keep the immediate
# behaviour.
import contextvars as _cv

_FORGET_PLAN = _cv.ContextVar("forget_plan", default=None)
_FORGET_PLANS: dict = {}
_PLAN_TTL_S = 3600
import threading as _threading
_PLANS_LOCK = _threading.Lock()      # previews are written from worker threads too (r8 review)


def _store_plan(plan: dict) -> str:
    import secrets
    import time as _t
    token = secrets.token_hex(4)
    try:
        from ..utils.logging import conversation_key_context
        plan.setdefault("conv", str(conversation_key_context.get() or ""))
    except Exception:  # noqa: BLE001
        plan.setdefault("conv", "")
    with _PLANS_LOCK:
        now = _t.time()
        for k in [k for k, v in _FORGET_PLANS.items() if now - v["ts"] > _PLAN_TTL_S]:
            _FORGET_PLANS.pop(k, None)
        _FORGET_PLANS[token] = plan
    return token


def _take_plan(token, kind: str):
    """The live plan for ``token`` of ``kind``, or None (unknown, another
    kind, or EXPIRED — the TTL is checked here, at confirm, r8 review)."""
    import time as _t
    with _PLANS_LOCK:
        plan = _FORGET_PLANS.get(str(token or "").strip())
        if plan is None or plan.get("kind") != kind:
            return None
        if _t.time() - plan["ts"] > _PLAN_TTL_S:
            _FORGET_PLANS.pop(str(token).strip(), None)
            return None
        return plan


def _resolve_plan(token, kind: str, match=None):
    """``(token, plan)`` for a confirm call, or ``(token, None)``.

    The given token when it is live. Otherwise — a missing, unknown or
    placeholder token like "yes" — the NEWEST live preview of ``kind`` made
    in THIS conversation (and passing ``match``, e.g. the same project).
    Why (§4LU): the token lives in the preview's TOOL RESULT, and the chat
    API / web history carries only reply text, so on the user's "yes" turn
    the model no longer has it — the live forget of the PostgreSQL manual
    failed this way. `_confirm_allowed` still requires a LATER turn by the
    user, so this changes how the plan is found, never who may confirm."""
    import time as _t
    plan = _take_plan(token, kind)
    if plan is not None:
        return str(token).strip(), plan
    try:
        from ..utils.logging import conversation_key_context
        conv = str(conversation_key_context.get() or "")
    except Exception:  # noqa: BLE001
        conv = ""
    if not conv:
        return token, None
    now = _t.time()
    with _PLANS_LOCK:
        cands = [(t, p) for t, p in _FORGET_PLANS.items()
                 if p.get("kind") == kind and p.get("conv") == conv
                 and now - p["ts"] <= _PLAN_TTL_S and (match is None or match(p))]
    if not cands:
        return token, None
    return max(cands, key=lambda tp: tp[1]["ts"])


def _drop_plan(token) -> None:
    with _PLANS_LOCK:
        _FORGET_PLANS.pop(str(token or "").strip(), None)


class _Plan:
    def __init__(self):
        self.items: list = []
        self._seen: set = set()

    def add(self, kind: str, ref: dict, label: str, default: bool = True) -> None:
        key = (kind, tuple(sorted((k, str(v)) for k, v in ref.items())))
        if key in self._seen:
            for it in self.items:                 # a later DEFAULT wins over a listed one
                if it["key"] == key and default:
                    it["default"] = True
            return
        self._seen.add(key)
        self.items.append({"kind": kind, "ref": ref, "label": label, "default": default, "key": key})


class _Passthrough:
    def __init__(self, real, plan):
        self._real, self._plan = real, plan

    def __getattr__(self, name):
        return getattr(self._real, name)


class _PlanProfile(_Passthrough):
    def delete(self, category, key):
        v = ((self._real.load() or {}).get(category) or {}).get(key)
        self._plan.add("profile_field", {"category": category, "key": key},
                       f"profile {category}.{key} = {str(v)[:80]!r}")
        return f"Removed {category}.{key}"

    def prune_value(self, category, key, target):
        v = ((self._real.load() or {}).get(category) or {}).get(key)
        for item in (v if isinstance(v, list) else [v]):
            if _value_mentions_target(item, str(target).lower()):
                self._plan.add("profile_item", {"category": category, "key": key, "item": str(item)},
                               f"profile {category}.{key} item {str(item)[:80]!r}")
        return f"Pruned {category}.{key}"

    def drop_previous_mentioning(self, target):
        # the SAME rule as the store's (r8 review: an unfolded regex missed
        # "José" for `jose` and every accented Greek name)
        from ..memory.profile import mentions, unwrap
        out = []
        if len(str(target or "").strip()) < 3:
            return out
        for cat, sub in (self._real.load_raw() or {}).items():
            for k, item in (sub.items() if isinstance(sub, dict) else []):
                if isinstance(item, dict) and "previous" in item and mentions(unwrap(item["previous"]),
                                                                                str(target).strip().lower()):
                    self._plan.add("profile_previous", {"category": cat, "key": k},
                                   f"profile {cat}.{k} previous value")
                    out.append(f"Removed the previous value of {cat}.{k}")
        return out


class _PlanGraph(_Passthrough):
    def forget_entity(self, entity):
        doomed, kept = self._real.preview_forget_entity(entity)
        for s_, p_, o_ in doomed:
            self._plan.add("graph_edge", {"s": s_, "p": p_, "o": o_}, f"graph {s_} {p_} {o_}")
        for s_, p_, o_ in kept:
            self._plan.add("graph_edge", {"s": s_, "p": p_, "o": o_}, f"graph {s_} {p_} {o_} (a fact about you)",
                           default=False)
        return len(doomed), kept

    def delete_by_target(self, target):
        return self.forget_entity(target)[0]

    def delete_edge(self, s_, p_, o_):
        self._plan.add("graph_edge", {"s": s_, "p": p_, "o": o_}, f"graph {s_} {p_} {o_}")
        return 1


class _PlanEpisodes(_Passthrough):
    def forget_mentions(self, target, vector_memory=None):
        prev = self._real.mention_previews(target)
        for i, trig in prev:
            self._plan.add("episode", {"id": i}, f"episode #{i} {trig[:80]!r}")
        return len(prev)

    def count_mentions(self, target):
        prev = self._real.mention_previews(target)
        for i, trig in prev:
            self._plan.add("episode", {"id": i}, f"episode #{i} {trig[:80]!r}", default=False)
        return len(prev)


class _PlanCollection(_Passthrough):
    def __init__(self, real, plan):
        super().__init__(real, plan)
        self._docs: dict = {}

    def query(self, *a, **k):
        res = self._real.query(*a, **k)
        try:
            for i, d in zip((res.get("ids") or [[]])[0], (res.get("documents") or [[]])[0]):
                self._docs[i] = d
        except Exception:  # noqa: BLE001
            pass
        return res

    def delete(self, ids=None, where=None, **k):
        for i in ids or []:
            self._plan.add("fact", {"id": i}, f"fact {str(self._docs.get(i, i))[:90]!r}")


class _PlanVector(_Passthrough):
    def __init__(self, real, plan):
        super().__init__(real, plan)
        self.collection = _PlanCollection(getattr(real, "collection", None), plan)

    def delete_document_by_name(self, name):
        self._plan.add("document", {"name": name}, f"document {name!r}")

    def delete_fragment(self, text):
        self._plan.add("fragment", {"text": text}, f"fact {str(text)[:90]!r}")


async def forget_preview(target, sandbox_dir=None, memory_system=None, profile_memory=None, graph_memory=None,
                         project_store=None, episodic_memory=None, skill_memory=None):
    """Run the forget sweep in PLAN mode and return the numbered list plus a
    confirmation token. Nothing is deleted."""
    import time as _t
    plan = _Plan()
    tok = _FORGET_PLAN.set(plan)
    try:
        notes = await tool_unified_forget(target, sandbox_dir, memory_system, profile_memory, graph_memory,
                                          project_store=project_store, episodic_memory=episodic_memory)
    finally:
        _FORGET_PLAN.reset(tok)
    # §4LC: the lessons that mention it (a request-scoped plan stores the
    # owner's request verbatim) — listed, and removed when confirmed
    if skill_memory is not None and hasattr(skill_memory, "lessons_mentioning"):
        try:
            for trig, scoped in await asyncio.to_thread(skill_memory.lessons_mentioning, target):
                plan.add("lesson", {"trigger": trig},
                         f"lesson {trig[:80]!r}" + (" (one request's plan)" if scoped else ""))
        except Exception as e:  # noqa: BLE001
            notes = f"{notes}\n⚠️ Lessons could not be searched: {e}"
    if not isinstance(notes, str):
        notes = str(notes)
    if notes.startswith(("SYSTEM ERROR", "Error", "Report:")) and not plan.items:
        return notes
    # the sweep's own warnings and refusals reach the user (r8 review: the
    # preview dropped "⚠️ Vector Error", a degraded profile, ambiguous and
    # partial file names, and then said "Nothing stored matches"). Its ✅
    # lines are plan-mode fiction and are not shown.
    warn = [ln for ln in notes.splitlines() if ln.startswith(("⚠️", "ℹ️"))]
    if not plan.items:
        head = (f"Nothing was changed, and nothing stored matches {target!r} exactly."
                if warn else f"Nothing stored matches {target!r}. Nothing was changed.")
        return "\n".join([head] + warn)
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        rid = ""
    token = _store_plan({"target": str(target), "items": plan.items, "rid": rid, "ts": _t.time(), "kind": "forget"})
    main = [f"{n}. {it['label']}" for n, it in enumerate(plan.items, 1) if it["default"]]
    extra = [f"{n}. {it['label']}" for n, it in enumerate(plan.items, 1) if not it["default"]]
    lines = [f"PREVIEW — nothing was deleted. forget {target!r} would remove:"] + (main or ["(nothing by default)"])
    if extra:
        lines += ["Also found (kept unless the user picks them by number):"] + extra
    if warn:
        lines += ["Notes from the search:"] + warn
    lines.append(f"Show this list to the user. Only after the USER confirms in their next message, call "
                 f"knowledge_base(action='forget', confirm='{token}', items='all') — or items='1,3' for a "
                 f"selection (numbers from either group). The token expires in an hour. If you no longer "
                 f"see the token in that later turn, pass confirm='yes' — this conversation's preview is "
                 f"found for you.")
    return "\n".join(lines)


def _reset_preview(memory_system, graph_memory, episodic_memory=None, skill_memory=None) -> str:
    import time as _t
    try:
        rows = memory_system.collection.count()
    except Exception:  # noqa: BLE001
        rows = "?"
    edges = "?"
    try:
        if graph_memory is not None and hasattr(graph_memory, "count_edges"):
            edges = graph_memory.count_edges()
    except Exception:  # noqa: BLE001
        pass
    eps = "?"
    try:
        if episodic_memory is not None and hasattr(episodic_memory, "count"):
            eps = episodic_memory.count()
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        rid = ""
    req_lessons = "?"
    try:
        if skill_memory is not None and hasattr(skill_memory, "request_scoped_count"):
            req_lessons = skill_memory.request_scoped_count()
    except Exception:  # noqa: BLE001
        pass
    token = _store_plan({"kind": "reset_all", "rid": rid, "ts": _t.time(), "items": []})
    return (f"PREVIEW — nothing was deleted. reset_all would erase the WHOLE vector memory ({rows} rows: every "
            f"fact, document and stored search copy), the knowledge graph ({edges} facts), the past-conversation "
            f"episodes ({eps}) and the one-request lessons that quote your requests ({req_lessons}) — not the "
            f"profile or the general lessons. Ask the user to "
            f"confirm. Only after the USER says yes in their next message, call "
            f"knowledge_base(action='reset_all', confirm='{token}') (or confirm='yes' if you no longer see "
            f"the token). To remove something specific instead, use "
            f"action='forget' with a target.")


#: request ids that are not the user answering: no request, the agent's own
#: background work, benchmark and replay traffic (r8 review: `SYSTEM`,
#: `bench-`, `replay-` passed, and a preview made in a `job-` turn — one
#: the user never saw — could be confirmed later)
_NOT_THE_USER_PREFIXES = ("bench-", "replay-")


def _not_the_user(rid: str) -> bool:
    if not rid or rid == "SYSTEM" or rid.startswith(_NOT_THE_USER_PREFIXES):
        return True
    # §4LD: a probe is not the user either (it could confirm a forget or a
    # reset_all preview; `_owner_write_block` already refused probes)
    try:
        from ..utils.logging import is_probe_request_id, request_origin_context, ORIGIN_PROBE
        if is_probe_request_id(rid) or str(request_origin_context.get() or "") == ORIGIN_PROBE:
            return True
    except Exception:  # noqa: BLE001
        return True
    try:
        from ..core.autonomous_activity import is_internal_request
        return bool(is_internal_request(rid))
    except Exception:  # noqa: BLE001
        return True


def _confirm_allowed(plan: dict):
    """The confirmation must come from the USER, in a LATER turn than the
    preview, and the preview must have been shown to the user: the model
    cannot preview and confirm in the same breath, and no background turn
    can say yes."""
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        rid = ""
    if _not_the_user(str(plan.get("rid") or "")):
        return "that preview was made outside a conversation with the user — run the preview again in one"
    if rid and rid == plan.get("rid"):
        return ("the user has not answered yet — show them the list and wait for their reply; confirm in the "
                "turn where they say yes")
    if _not_the_user(rid):
        return "only the user can confirm a deletion, in their own turn"
    return None


def _pick(items: list, selection):
    """``(chosen, unknown_parts)``. 'all' = the default list; numbers pick from
    either group; a list/int from a JSON call is accepted (r8 review)."""
    # (a list from a JSON call renders "[1, 3]", which the split below reads)
    sel = str(selection if selection is not None else "all").strip().lower()
    if sel in ("all", "*", "yes", ""):
        return [it for it in items if it["default"]], []
    out, bad = [], []
    for part in [p for p in re.split(r"[,\s\[\]]+", sel) if p]:
        if "-" in part and all(x.isdigit() for x in part.split("-", 1)):
            a, b = (int(x) for x in part.split("-", 1))
            if not (1 <= a <= b <= len(items)):
                bad.append(part)
                continue
            out += [items[i - 1] for i in range(a, b + 1)]
        elif part.isdigit() and 1 <= int(part) <= len(items):
            out.append(items[int(part) - 1])
        else:
            bad.append(part)
    seen, uniq = set(), []
    for it in out:
        if id(it) not in seen:
            seen.add(id(it))
            uniq.append(it)
    return uniq, bad


async def forget_execute(token, selection="all", sandbox_dir=None, memory_system=None, profile_memory=None,
                         graph_memory=None, project_store=None, episodic_memory=None, skill_memory=None):
    """Delete exactly the confirmed items of a preview."""
    token, plan = _resolve_plan(token, "forget")
    if plan is None:
        return ToolOutcome.rejected("Error: unknown or expired confirmation token — run the forget preview again.",
                                    reason_code="forget_token_unknown")
    why = _confirm_allowed(plan)
    if why:
        return ToolOutcome.rejected(f"NOT deleted: {why}.", reason_code="forget_not_confirmed")
    chosen, bad = _pick(plan["items"], selection)
    if bad:
        return ToolOutcome.rejected(
            f"NOT deleted: {', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not on the list (items 1–"
            f"{len(plan['items'])}). Nothing was changed; the token still works.", reason_code="forget_bad_selection")
    if not chosen:
        return ToolOutcome.rejected("NOT deleted: the selection names no listed item.", reason_code="forget_empty_selection")
    _drop_plan(token)
    report = []
    for it in chosen:
        try:
            report.append(await asyncio.to_thread(_execute_item, it, memory_system, profile_memory, graph_memory,
                                                  project_store, episodic_memory, skill_memory))
        except Exception as e:  # noqa: BLE001
            report.append(f"⚠️ {it['label']}: {e}")
    return "\n".join(report)


def _execute_item(it, memory_system, profile_memory, graph_memory, project_store, episodic_memory,
                  skill_memory=None) -> str:
    k, r = it["kind"], it["ref"]
    if k == "lesson":
        if skill_memory is None:
            return f"⚠️ {it['label']}: the lesson store is not available"
        ok = skill_memory.remove_by_trigger(r["trigger"], memory_system=memory_system)
        return f"✅ Removed {it['label']} (archived)" if ok else f"ℹ️ {it['label']} was already gone"
    if k == "file":
        root, path = Path(r["root"]), Path(r["path"])
        if path.is_symlink() or not _is_within_root(path.resolve(), root.resolve()):
            return f"⚠️ Refused {it['label']} (link or outside the sandbox)"
        from .file_system import _released_write_block
        if _released_write_block(project_store, root, str(path.resolve().relative_to(root.resolve())), removes=True):
            return f"⚠️ Refused {it['label']} — inside a RELEASED project (immutable)"
        if path.is_file():
            path.unlink()
            return f"✅ Deleted {it['label']}"
        return f"ℹ️ {it['label']} no longer exists"
    if k == "document":
        memory_system.delete_document_by_name(r["name"])
        return f"✅ Removed {it['label']}"
    if k == "fact":
        memory_system.collection.delete(ids=[r["id"]])
        return f"✅ Forgot {it['label']}"
    if k == "fragment":
        memory_system.delete_fragment(r["text"])
        return f"✅ Forgot {it['label']}"
    if k in ("profile_field", "profile_item", "profile_previous"):
        if k == "profile_field":
            res = profile_memory.delete(r["category"], r["key"], exact=True)
        elif k == "profile_item":
            res = profile_memory.remove_item(r["category"], r["key"], r["item"])
        else:
            res = profile_memory.drop_previous(r["category"], r["key"])
        # (r8 review) "not found" was reported with a ✅
        if isinstance(res, str) and res.lower().startswith(("profile key not found", "no item", "no previous")):
            return f"ℹ️ {it['label']} was already gone"
        return _profile_line(res, f"Removed {it['label'][8:]}")
    if k == "graph_edge":
        n = graph_memory.delete_edge(r["s"], r["p"], r["o"])
        return f"✅ Removed {it['label']}" if n else f"ℹ️ {it['label']} was already gone"
    if k == "episode":
        n = episodic_memory.delete_episodes([r["id"]], memory_system, reason=f"forget {it.get('target', '')}".strip())
        if n and getattr(episodic_memory, "last_twin_failures", None):
            # (§4LA) the row went but its search copy did not — say so
            return f"⚠️ Forgot {it['label']}, but its search copy could not be removed yet (it is retried later)"
        return f"✅ Forgot {it['label']} (archived for 30 days)" if n else f"ℹ️ {it['label']} was already gone"
    return f"⚠️ unknown item kind {k}"


async def tool_unified_forget(target: str = None, sandbox_dir: Path = None, memory_system=None, profile_memory=None, graph_memory=None, project_store=None, episodic_memory=None):
    _blocked = _member_block()
    if _blocked is not None:
        return _blocked
    # NOTE: this message names THIS function's parameter and is meant for a
    # DIRECT caller. A model reaching this tool goes through
    # `tool_knowledge_base`, which guards the argument itself and builds its
    # error from `_KB_TARGET_ALIASES` — do not route this string to a model,
    # and do not "helpfully" copy its wording into the dispatcher. The
    # dispatcher used to surface it verbatim, telling models to pass a
    # parameter the dispatcher dropped; the retry was byte-identical forever.
    if not target:
        return "SYSTEM ERROR: The 'target' parameter is MANDATORY. You must specify it."
    # Reject ultra-short targets that would match nearly everything.
    if len(str(target).strip()) < 3:
        return "Error: 'target' must be at least 3 characters. Be specific to avoid wiping unrelated memories."
    pretty_log("Memory Wipe", target, icon=Icons.MEM_WIPE)
    if not memory_system: return "Report: Memory disabled."
    report = []
    _plan = _FORGET_PLAN.get()
    if _plan is not None:
        # PLAN MODE: every destructive call is recorded, nothing is deleted
        # (the model-facing forget is preview → confirm)
        memory_system = _PlanVector(memory_system, _plan)
        profile_memory = _PlanProfile(profile_memory, _plan) if profile_memory is not None else None
        graph_memory = _PlanGraph(graph_memory, _plan) if graph_memory is not None else None
        episodic_memory = _PlanEpisodes(episodic_memory, _plan) if episodic_memory is not None else None

    # ⚠ ORDER. The `sandbox/` strip below removes a component, and the disk
    # sweep decides "did the caller name a PATH?" from the presence of a
    # separator — so a two-component `sandbox/index.html` lost its only
    # separator here and fell back to the basename tier, deleting every
    # index.html in the tree. That is the exact defect the path rule was
    # added to close, reachable by adding four characters, and
    # `file_system.py` documents `sandbox/` as an observed live model shape.
    # Decide the shape FIRST, from the raw string.
    _raw_target = str(target).strip()
    _probe = _raw_target.rstrip(_PATH_SEPS)
    target_names_a_path = ("/" in _probe.lstrip("/")
                           or os.sep in _probe.lstrip(os.sep))
    # Computed here, used by BOTH sweeps. They were written separately and
    # disagreed: one call printed "Nothing on disk is deleted for a partial
    # name match" two lines above "Vector: Wiped document 'notes.txt'" — the
    # same two names, opposite policies, and the irreversible half was the
    # one ignoring the rule. Against the live library `forget('postgresql-
    # 19-A4.md')` destroyed the 7k-chunk manual: the exact incident the
    # vector rule was written for, through the extension case it lacked.
    _tgt_name = Path(_raw_target).name
    target_names_a_file = bool(Path(_tgt_name).suffix) or _tgt_name.startswith(".")

    # `removeprefix`, not `lstrip`: lstrip takes a CHARACTER SET, so
    # `.config/x.md` became `config/x.md` (a different, unnamed file) and
    # `../../etc/passwd` became `etc/passwd`.
    # A trailing separator is not part of the name: `forget('notes/')` kept
    # the slash in `clean_target`, so the disk half matched `notes` as a bare
    # topic and deleted five files while the profile and graph halves matched
    # nothing at all — one call, two different targets.
    clean_target = _raw_target.rstrip(_PATH_SEPS) or _raw_target
    while clean_target.startswith("./"):
        clean_target = clean_target[2:]
    clean_target = clean_target.lstrip("/")
    if clean_target.startswith("sandbox/"):
        clean_target = clean_target[8:]

    # --- ENTITY-AWARE EXPANSION ---
    # Pull the target's direct graph neighbours so the wipe also reaches
    # ALIAS tombstones: forgetting 'mortimer' should also clear facts stored
    # under 'iguana' (from a `mortimer IS_A iguana` edge). Computed up-front,
    # BEFORE the graph delete in step 4 severs those very edges. Hub nodes
    # (user/pronouns) are filtered inside get_connected_entities so the
    # expansion can't snowball. Vector/profile expansion is LITERAL-mention
    # only (no semantic fuzz) so it stays precise.
    # decided BEFORE any leg deletes the family edge it is read from
    # ONE entity for every leg (re-review: the episode leg searched the raw
    # target while the graph leg searched its entity words)
    _entity = _entity_of(clean_target) or clean_target.strip().lower()
    _quals = _qualifiers_of(clean_target) if _entity != clean_target.strip().lower() else []
    _family_person = False
    try:
        if graph_memory is not None and hasattr(graph_memory, "is_owner_family"):
            _family_person = bool(await asyncio.to_thread(graph_memory.is_owner_family, _entity))
    except Exception:  # noqa: BLE001
        _family_person = False
    expanded_targets: list = []
    # (third memory-writes review) the expansion ran for hub words and the
    # owner's own name with none of the graph leg's guards, and through a
    # CLASS link (`hermes IS_A llm`) it then swept every edge, fact and
    # profile item naming the class word — five owner edges among them. It
    # now follows ALIASES only (another name for the same thing), and only
    # from a target that names an entity which is not the owner.
    if graph_memory is not None and _names_an_entity(clean_target) and not _is_owner_name(profile_memory, clean_target):
        try:
            # `clean_target`, like every other sweep. This is the AMPLIFIER
            # of a forget — it is what reaches the alias tombstone
            # ('mortimer' -> 'iguana') — and leaving it raw made it dead for
            # exactly the two spellings the normalisation was added for:
            # `./mortimer` and `sandbox/mortimer` cleared one edge instead
            # of two and left the profile row untouched.
            expanded_targets = await asyncio.to_thread(
                graph_memory.get_connected_entities, clean_target)
        except Exception:
            expanded_targets = []
    if not isinstance(expanded_targets, list):
        expanded_targets = []
    expanded_targets = [x for x in expanded_targets
                        if _names_an_entity(str(x)) and not _is_owner_name(profile_memory, str(x))]

    # 1. Disk Cleanup — recursive walk + safe-path validation.
    # Previous version only looked at the top-level directory, only deleted
    # the FIRST match, and used unbounded substring matching. We now walk
    # the sandbox, prefer exact name / stem matches, only fall back to
    # substring when nothing better matches, and explicitly verify each
    # deletion target stays inside the sandbox root before unlinking.
    if sandbox_dir is not None:
        try:
            sandbox_root = Path(sandbox_dir).resolve()
            target_basename = Path(clean_target).name.lower()
            target_stem = Path(clean_target).stem.lower()
            # A target carrying a separator NAMES ONE FILE. Matching it on
            # the basename alone deleted every file in the tree sharing that
            # name — and the kept-report prints candidates as
            # sandbox-relative paths and tells the caller to re-issue with
            # one, so its own instruction was the trigger:
            # `forget('projects/alpha/report_atlas.md')` removed the beta and
            # archive copies too, and on the live sandbox
            # `forget('projects/<id>/index.html')` removed five index.html
            # files across five projects. A path means a path.
            target_relpath = clean_target.lower() if target_names_a_path else None
            # A target with an EXTENSION also names one file, so it must not
            # fall through to the stem tier: with `notes.md` absent,
            # `forget('notes.md')` deleted notes.txt, notes.xlsx and
            # notes.pdf — the very files the report calls "partial matches"
            # and promises not to touch when `notes.md` happens to exist.
            target_is_filename = target_names_a_file

            exact_hits: list[Path] = []
            stem_hits: list[Path] = []
            substr_hits: list[Path] = []
            for root, dirs, files in os.walk(sandbox_root):
                # Skip hidden + heavy dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', '__pycache__', 'env', 'acquired_skills')]
                for fname in files:
                    fname_lc = fname.lower()
                    _abs = Path(root) / fname
                    if target_relpath is not None:
                        # Path-qualified: only that exact path is a hit, and
                        # nothing weaker is offered — the caller was precise.
                        try:
                            if str(_abs.relative_to(sandbox_root)).lower() == target_relpath:
                                exact_hits.append(_abs)
                        except ValueError:
                            pass
                        continue
                    if fname_lc == target_basename:
                        exact_hits.append(_abs)
                    elif Path(fname).stem.lower() == target_stem:
                        (substr_hits if target_is_filename else stem_hits).append(_abs)
                    elif len(target_stem) >= 3 and target_stem in fname_lc:
                        substr_hits.append(_abs)

            # REDUNDANT PREFIX HEALING. `knowledge_base` receives the
            # PROJECT workspace as its root when a project is active, so a
            # model reading a listing produces `projects/<id>/index.html`
            # while the root already IS `.../projects/<id>`. Without this the
            # path rule turned a working call into a silent total no-op —
            # and, because a path-qualified miss short-circuits, it emitted
            # no candidate report either: the caller was told the file does
            # not exist. `file_system` does the same healing.
            if target_relpath and not exact_hits:
                _root_parts = [q.lower() for q in sandbox_root.parts]
                _t_parts = target_relpath.split("/")
                for _drop in range(1, len(_t_parts)):
                    if _t_parts[:_drop] != _root_parts[-_drop:]:
                        continue
                    _healed = "/".join(_t_parts[_drop:])
                    _cand = sandbox_root / _healed
                    if _cand.is_file():
                        exact_hits.append(_cand)
                        break

            # Substring matches are REPORTED, NOT DELETED.
            #
            # The tier existed to catch near-misses, and it deletes every
            # file whose name merely CONTAINS the target: measured,
            # `forget('atlas')` unlinked `atlas_migration_plan.py`,
            # `notes_about_atlas.md` and `sub/deep_atlas_notes.txt`.
            # Irreversible, model-reachable, no dry-run — and the `target`
            # parameter is now described to the model in the vocabulary that
            # feeds this branch hardest ("a topic, an entity, a person's
            # name"), so the input is a bare word far more often than a
            # filename.
            #
            # Forgetting a TOPIC does not mean deleting every file whose
            # name shares a token with it; the vector, profile and graph
            # sweeps below remove the knowledge either way. So a substring
            # hit now surfaces as a candidate the caller can name
            # explicitly — at which point it is an exact match and is
            # deleted. Exact and stem matches are unchanged.
            # AMBIGUITY GATE. A bare basename matches every file with that
            # name anywhere in the tree, and the tier that DELETES had no
            # such check while the tier that only reports did: measured on
            # the live sandbox, `forget('index.html')` removed five files
            # across five projects and `forget('app.py')` two, silently and
            # irreversibly. The conservatism had landed entirely on the tier
            # that does not delete. More than one match means the caller has
            # not said which — so say so, and take the path.
            # ...over whichever tier is about to DELETE. Gating `exact_hits`
            # alone meant `forget('index.html')` refused three files while
            # `forget('index')` — five characters shorter — unlinked five,
            # through the stem tier the gate never looked at.
            # Ambiguity is about LOCATION, not count. `forget('notes')`
            # taking `notes.md` and `notes.txt` from one directory is the
            # stem tier doing its job — the caller named that thing and
            # there is one of it. `forget('index')` taking `index.js` from
            # the root and four `index.html` files from four different
            # projects is five different things wearing one name, and the
            # caller cannot have meant all of them. So the gate fires when
            # the matches span more than one directory.
            # ⚠ CLEAR BOTH TIERS. Emptying `exact_hits` alone handed the
            # very next line (`chosen = exact_hits or stem_hits`) the weaker
            # tier that had LOST to them: with `p/a/index`, `p/b/index` and
            # `index.html`, `forget('index')` refused the two the caller may
            # have meant and irreversibly deleted the third — a fresh
            # instance of the contradiction-in-one-report this gate exists
            # to remove. If the caller has not said which, nothing goes.
            _ambiguous: list[Path] = []
            if not target_names_a_path:
                _tier = exact_hits or stem_hits
                if len({h.parent for h in _tier}) > 1:
                    _ambiguous = list(_tier)
                    exact_hits = []
                    stem_hits = []
            chosen: list[Path] = exact_hits or stem_hits
            # Everything the sweep matched but did NOT delete — the weaker
            # tiers the `or` chain shadowed, not just the substring one.
            # This report is the entire mitigation for no longer deleting
            # them, and the first version only emitted it when NOTHING was
            # deleted: `forget('atlas')` with an `atlas.md` present deleted
            # that one file and said nothing about the three others it had
            # matched. `forget('notes.md')` likewise kept `notes.txt`
            # silently. The caller has to be told what survived, whether or
            # not something else went.
            # Symlinks are refused at the unlink below, so listing one as
            # something to "forget by its exact name" is a permanent dead
            # end. Name them separately.
            kept = [h for h in (_ambiguous + stem_hits + substr_hits)
                    if h not in chosen and not h.is_symlink()]
            _pl = _FORGET_PLAN.get()
            if _pl is not None:
                # listed, deleted only when the user picks one (r8 review:
                # the schema promised them under "also found")
                for h in kept:
                    _pl.add("file", {"path": str(h), "root": str(sandbox_root)},
                            f"file {h.relative_to(sandbox_root)}", default=False)
            if kept:
                # Print `./name` for a root-level candidate whose basename
                # also occurs deeper in the tree: re-issuing a bare name
                # matches EVERY file with it, so the report's own
                # instruction would delete siblings the caller never saw.
                # `./` makes the re-issue path-qualified, and hence exact.
                _all_names = [h.name.lower() for h in (exact_hits + kept)]
                _rel = []
                for h in kept:
                    _r = str(h.relative_to(sandbox_root))
                    if "/" not in _r and _all_names.count(h.name.lower()) > 1:
                        _r = "./" + _r
                    _rel.append(_r)
                _rel = sorted(_rel)
                _shown = _rel[:10]
                _more = (f" (+{len(_rel) - len(_shown)} more; narrow the "
                         f"target to see them)" if len(_rel) > len(_shown) else "")
                report.append(
                    "ℹ️ Disk: kept "
                    + str(len(_rel))
                    + (" file(s) matching " if _ambiguous
                       else " file(s) whose NAME only partly matches ")
                    + repr(clean_target)
                    + " — "
                    + ", ".join(repr(n) for n in _shown)
                    + _more
                    + ". Nothing on disk is deleted for a partial name match;"
                    " to delete one, forget it by the exact name shown here."
                )
            for victim in chosen:
                try:
                    # Never delete THROUGH a symlink: victim.resolve()
                    # follows it, so unlinking the resolved path would
                    # remove the (possibly out-of-sandbox) target file.
                    if victim.is_symlink():
                        report.append(f"⚠️ Disk: Refused symlink '{victim}' (won't delete through links)")
                        continue
                    resolved = victim.resolve()
                    # Hard sandbox containment check before unlink —
                    # path-component-wise, NOT str.startswith (which would
                    # accept a sibling like '…/sandbox_evil').
                    if not _is_within_root(resolved, sandbox_root):
                        report.append(f"⚠️ Disk: Refused unsafe path '{victim}' (outside sandbox)")
                        continue
                    # §4KW: the RELEASED-project lock, per file the sweep
                    # would delete (`forget projects/<released>/index.html`
                    # removed it). Per file, not up front: a forget of a
                    # MEMORY inside a released project must still run
                    # (second review).
                    try:
                        from .file_system import _released_write_block
                        _rb = _released_write_block(project_store, sandbox_root,
                                                    str(resolved.relative_to(sandbox_root)),
                                                    removes=True)
                    except Exception:  # noqa: BLE001
                        _rb = None
                    if _rb:
                        report.append(f"⚠️ Disk: Refused '{victim}' — inside a RELEASED project (immutable)")
                        continue
                    if resolved.is_file():
                        _pl = _FORGET_PLAN.get()
                        if _pl is not None:
                            _pl.add("file", {"path": str(resolved), "root": str(sandbox_root)},
                                    f"file {resolved.relative_to(sandbox_root)}")
                            continue
                        resolved.unlink()
                        report.append(f"✅ Disk: Deleted '{resolved.relative_to(sandbox_root)}'")
                except Exception as de:
                    report.append(f"⚠️ Disk: Could not delete '{victim}': {de}")
        except Exception as e:
            report.append(f"⚠️ Disk Error: {e}")

    # 2. Vector Memory Cleanup (Search then Destroy)
    try:
        # --- FUZZY FILENAME SWEEP ---
        # Get all unique sources currently in the DB instantly via the index.
        # Wrap in a lambda so `to_thread` actually invokes the bound method
        # (passing the bound method directly was a no-op because to_thread
        # would call get_library() with no args — but it was previously
        # being passed without parens, leaving the method un-invoked).
        all_sources = set(await asyncio.to_thread(lambda: memory_system.get_library()))
        
        # Look for a fuzzy match in filenames. Guard against over-deletion:
        # a 1-2 char stem ("a") as a bare substring matched nearly EVERY
        # document (mass wipe), and the reverse `source in target_stem`
        # direction was nonsensical. Match the disk sweep's discipline —
        # require >=3 chars for substring matching (against the basename),
        # and for a shorter stem only an EXACT filename-stem match.
        # SAME DISCIPLINE AS THE DISK SWEEP. A substring match on a
        # document's filename used to DELETE the whole document: against the
        # live library (one entry — the ~7k-chunk PostgreSQL manual),
        # `forget('pdf')` / `forget('sql')` / `forget('postgres')` each
        # destroyed it. Three characters, no candidate list, irreversible.
        # And since §4DL the disk half of this very call prints "Nothing on
        # disk is deleted for a partial name match" while this half did
        # exactly that to the knowledge. Exact name or exact stem deletes;
        # anything looser is reported so the caller can name it.
        target_name = Path(clean_target).name.lower()
        target_stem = Path(clean_target).stem.lower()
        # Documents are keyed by SOURCE NAME, so a path-qualified target
        # identifies one by its basename — and only exactly. And a target
        # carrying an extension names ONE document, exactly as on disk:
        # matching its stem too wiped `notes.pdf` and `notes.txt` for
        # `forget('notes.md')`, each of which is an entire ingested document
        # plus its library row.
        if target_names_a_path:
            # A source may itself carry a path. Prefer the whole-string
            # match; if the named path is not in the library, basename
            # matches are candidates to REPORT, not documents to delete —
            # the caller was precise and the library disagrees.
            _tl = clean_target.lower()

            def _norm_source(src: str) -> str:
                out = src.lower()
                while out.startswith("./"):
                    out = out[2:]
                return out.lstrip("/")

            # `removeprefix`-style, NOT `lstrip("./")` — the character-set
            # bug this function documents 40 lines above. It collapsed
            # `notes.md`, `.notes.md`, `..notes.md` and `./notes.md` onto one
            # key, so one call wiped four distinct documents.
            doc_exact = [s for s in all_sources if _norm_source(s) == _tl]
        elif target_names_a_file:
            doc_exact = [s for s in all_sources
                         if Path(s).name.lower() == target_name]
        else:
            doc_exact = [s for s in all_sources
                         if Path(s).name.lower() == target_name
                         or Path(s).stem.lower() == target_stem]
        doc_exact = sorted(doc_exact)
        doc_partial = [s for s in all_sources
                       if s not in doc_exact
                       and len(target_stem) >= 3
                       and (target_stem in Path(s).name.lower()
                            or (target_names_a_path
                                and Path(s).name.lower() == target_name))]
        # The ambiguity gate, on this half too. It was disk-only, so one
        # report printed "kept 3 file(s) … Nothing on disk is deleted for a
        # partial name match" above three "✅ Vector: Wiped document" lines
        # naming the SAME three files — and the irreversible half was again
        # the one ignoring the rule. Sources sharing a basename across
        # different directories are different documents.
        if not target_names_a_path and len(doc_exact) > 1:
            if len({str(Path(_s).parent) for _s in doc_exact}) > 1:
                doc_partial = sorted(set(doc_partial) | set(doc_exact))
                doc_exact = []
        for match in doc_exact:
            await asyncio.to_thread(memory_system.delete_document_by_name, match)
            report.append(f"✅ Vector: Wiped document '{match}'.")
        _pl = _FORGET_PLAN.get()
        if _pl is not None:
            for _d in sorted(doc_partial):
                _pl.add("document", {"name": _d}, f"document {_d!r}", default=False)
        if doc_partial:
            # `+N more`, like the disk half. Naming 10 of N while telling the
            # caller to re-issue with one of the names shown leaves the rest
            # unreachable — the same defect the disk report was fixed for,
            # repeated here because this report was written from it.
            _shown_docs = sorted(doc_partial)[:10]
            _more_docs = (f" (+{len(doc_partial) - len(_shown_docs)} more; "
                          f"narrow the target to see them)"
                          if len(doc_partial) > len(_shown_docs) else "")
            report.append(
                "ℹ️ Vector: kept " + str(len(doc_partial)) + " ingested "
                "document(s) whose NAME only partly matches "
                + repr(target) + " — "
                + ", ".join(repr(m) for m in _shown_docs)
                + _more_docs
                + ". Re-issue with one of these exact names to remove it."
            )

        # --- SEMANTIC SWEEP (For loose facts and smart_memory "auto" facts) ---
        # Run query + delete UNDER the vector lock so we don't race with
        # background ingest / smart_memory writes.
        # `clean_target`, not the raw string. The disk and document halves
        # normalise `./`, a leading `/` and a `sandbox/` prefix; these three
        # did not, so `forget('./notes.md')` and `forget('sandbox/notes.md')`
        # removed the file and the document and left every fact, profile row
        # and graph edge in place — while the report said nothing about the
        # half that had not run. `sandbox/` is documented in file_system.py
        # as an observed live model shape.
        sweep_target_lc = clean_target.strip().lower()
        _target_is_filename = bool(re.search(r"\.[a-z0-9]{1,8}$", sweep_target_lc)) or "/" in sweep_target_lc

        def _semantic_sweep():
            with memory_system._get_lock() if hasattr(memory_system, "_get_lock") else _NullCM():
                # Scope the sweep to CONVERSATIONAL fact types. Unscoped,
                # the top-20 nearest pool is ~97% ingested document chunks
                # (live store), and the literal-mention override below
                # deletes regardless of distance — forgetting a word that
                # appears in a manual silently gutted the document (library
                # index still listed it; dedup then refused re-ingest), and
                # episode/skill twins deleted here orphan their JSON side.
                cand = memory_system.collection.query(
                    # normalised, like every other sweep — the raw string
                    # embedded `./` / `sandbox/` into the query vector
                    query_texts=[sweep_target_lc], n_results=20,
                    where={"type": {"$nin": _FORGET_PROTECTED_TYPES}})
                deleted_local = 0
                hits = []
                word_cands = []
                literal_hits = []
                if cand.get('ids'):
                    for i, dist in enumerate(cand['distances'][0]):
                        doc_text = cand['documents'][0][i]
                        mem_id = cand['ids'][0][i]
                        meta = cand['metadatas'][0][i] or {}
                        m_type = meta.get('type', 'auto')
                        if m_type in _FORGET_PROTECTED_TYPES:
                            continue  # belt-and-braces vs the where scope
                        # Fresh review (§4KW): 0.8/0.6 deleted UNRELATED owner
                        # facts — `forget postgresql-19-A4.pdf` removed the
                        # owner's birth date, both sons' birthdates and home
                        # town (live, 09-09); unrelated facts sit at 0.53–0.65
                        # from any short target. A FILE/DOCUMENT name removes
                        # only facts that name it; an entity needs a near-
                        # paraphrase (< _FORGET_SEMANTIC_MAX).
                        semantic_threshold = 0.0 if _target_is_filename else _FORGET_SEMANTIC_MAX
                        # LITERAL-MENTION OVERRIDE: the distance threshold
                        # silently missed facts that name the target outright
                        # — e.g. forgetting 'iguana' left "user previously had
                        # an iguana that was removed" in place because its L2
                        # distance to the bare word exceeded the bar. When the
                        # user explicitly names an entity, any stored fact that
                        # mentions it (word-boundary) is fair game regardless
                        # of distance.
                        literal = _value_mentions_target(doc_text, sweep_target_lc)
                        # §4KW second review: 0.3 alone deleted nothing for a
                        # real entity forget ("my address" ~ "home address is
                        # …" at 0.57, "my job at google" at 0.41) while
                        # unrelated facts sat at 0.53–0.65 — no single distance
                        # separates them. A fact that SHARES a content word
                        # (stem) with the target is the target's, up to 0.7.
                        if (not _target_is_filename and not literal
                                and not dist < semantic_threshold):
                            _cov = _target_word_coverage(doc_text, sweep_target_lc)
                            if _cov is not None and ((_cov >= 1.0 and dist < _FORGET_SHARED_WORD_MAX)
                                                     or (_cov >= 0.5 and dist < _FORGET_PARTIAL_WORD_MAX)):
                                word_cands.append((mem_id, doc_text, _cov))
                            continue
                        if literal and not dist < semantic_threshold:
                            literal_hits.append((mem_id, doc_text))
                            continue
                        if dist < semantic_threshold:
                            memory_system.collection.delete(ids=[mem_id])
                            deleted_local += 1
                            hits.append(f"✅ Sweep: Forgot derived fact: '{doc_text[:40]}...'")
                # a fact matched only by its WORDS is removed only when it is
                # the ONE such fact (third review: "user nickname" deleted all
                # 8 owner facts; "my address" matches the home AND the email
                # address) — otherwise the candidates are named, not deleted
                # a LITERAL mention: every fact naming an entity is the
                # entity's ("iguana") — but a target of hub words only names no
                # fact ("forget user" deleted all four owner facts), and an
                # attribute noun named by several facts is a choice ("address":
                # home and email) — fourth review
                from ..memory.lesson_scope import content_words
                _distinct = {w for w in content_words(sweep_target_lc)
                             if w not in _FORGET_HUB_WORDS and len(w) >= 2}
                if literal_hits and (not _distinct or (
                        len(literal_hits) > 1 and _distinct <= _FORGET_ATTRIBUTE_WORDS)):
                    word_cands = [(i, t, 1.0) for i, t in literal_hits] + word_cands
                    literal_hits = []
                for mem_id, doc_text in literal_hits:
                    memory_system.collection.delete(ids=[mem_id])
                    deleted_local += 1
                    hits.append(f"✅ Sweep: Forgot literal fact: '{doc_text[:40]}...'")
                # the entity's own facts went: a fact matched only by SOME of
                # its words is another fact ("wife is named Maria" after
                # "wife's birthday is 3 March") — listed, never deleted (fifth
                # review)
                _auto_ok = not literal_hits
                # one FULL match wins over partial ones (fourth review: "wife's
                # birthday" was refused because "wife is named Maria" half-matched)
                _full = [c for c in word_cands if c[2] >= 1.0]
                if len(_full) == 1:
                    word_cands = _full
                if len(word_cands) == 1 and _distinct and _auto_ok:
                    memory_system.collection.delete(ids=[word_cands[0][0]])
                    deleted_local += 1
                    hits.append(f"✅ Sweep: Forgot matching fact: '{word_cands[0][1][:40]}...'")
                elif word_cands:
                    _pl = _FORGET_PLAN.get()
                    if _pl is not None:
                        for c in word_cands:
                            _pl.add("fact", {"id": c[0]}, f"fact {str(c[1])[:90]!r}", default=False)
                    hits.append("ℹ️ Sweep: " + ("several stored facts match '" if len(word_cands) > 1
                                                  else "another stored fact matches '")
                                + sweep_target_lc + "' — NOT deleted: "
                                + "; ".join(repr(c[1][:60]) for c in word_cands[:5])
                                + ". Forget the one you mean by its exact text.")
                return deleted_local, hits

        deleted_count, hits = await asyncio.to_thread(_semantic_sweep)
        report.extend(hits)

    except Exception as e: report.append(f"⚠️ Vector Error: {e}")

    # 3. Profile Memory Cleanup — scoped, NOT the previous greedy sweep.
    # The old version did substring-match against BOTH keys AND values, so
    # `target="python"` would wipe any profile entry whose key OR value
    # contained "python", potentially nuking unrelated state. We now:
    #   * Prefer exact key match
    #   * Fall back to substring match only on KEYS (not values)
    #   * Skip if the key contains the target as a tiny substring of a
    #     much longer key (e.g. target="age" should NOT match "language")
    if profile_memory:
        try:
            report.extend(_forget_profile(profile_memory, clean_target, qualifiers=_quals,
                                          family=_family_person and not _quals))
        except Exception as e: report.append(f"⚠️ Profile Error: {e}")

    # (r8 review) an unreadable profile hides the owner's NAME, and every
    # owner guard below reads it: the graph and episode legs hold back
    _profile_blind = profile_memory is not None and getattr(
        getattr(profile_memory, "_real", profile_memory), "_degraded", False) is True
    if _profile_blind and (graph_memory or episodic_memory is not None):
        report.append("⚠️ Graph/Episodes: not searched — the profile could not be read, so the owner's own name "
                      "is unknown. Try again once it is readable.")
        graph_memory = None
        episodic_memory = None
    # 4. Knowledge Graph Cleanup
    if graph_memory:
        try:
            # `clean_target`, like every other sweep — the raw string
            # carried `./` / `sandbox/` / a trailing slash straight
            # into the graph, where it matched nothing.
            # a target of hub words only names no entity (profile-writes
            # review: `forget user` expired 469 of 1,016 live edges)
            _is_owner = _is_owner_name(profile_memory, _entity)
            if _is_owner:
                report.append("ℹ️ Graph: that is your own name — your relations were NOT removed. "
                              "Forget the specific fact instead.")
            deleted_edges = 0
            if _names_an_entity(clean_target) and not _is_owner:
                if _quals and hasattr(graph_memory, "preview_forget_entity"):
                    # attribute-qualified: only the edges naming the attribute
                    _d, _k = await asyncio.to_thread(graph_memory.preview_forget_entity, _entity)
                    _fields = (await asyncio.to_thread(graph_memory.owner_field_edges, _entity)
                               if hasattr(graph_memory, "owner_field_edges") else [])
                    _all = list(dict.fromkeys(list(_d) + list(_k) + list(_fields)))
                    # the PREDICATE names the attribute (r8 review: subject/
                    # object words and a 4-letter stem took COMPANION)
                    _sel = [e for e in _all if _qualifier_matches(str(e[1]).lower().split("_"), _quals)]
                    # an owner fact is never in the default list (r8 review:
                    # "EvolMonkey work" defaulted `user WORKS_AT evolmonkey`)
                    _rest = [e for e in _all if e not in _sel or (e in _k and not _family_person)]
                    _sel = [e for e in _sel if e not in _rest]
                    for e in _sel:
                        deleted_edges += int(await asyncio.to_thread(graph_memory.delete_edge, *e) or 0)
                    _pl = _FORGET_PLAN.get()
                    if _pl is not None:
                        for e in _rest:
                            _pl.add("graph_edge", {"s": e[0], "p": e[1], "o": e[2]}, "graph " + " ".join(e),
                                    default=False)
                    if _rest:
                        report.append("ℹ️ Graph: other facts about '" + _entity + "' were kept: "
                                      + "; ".join(" ".join(e) for e in _rest[:5])
                                      + (" …" if len(_rest) > 5 else "") + ".")
                elif hasattr(graph_memory, "forget_entity"):
                    _res = await asyncio.to_thread(graph_memory.forget_entity, _entity)
                    deleted_edges, _kept = _res if isinstance(_res, tuple) and len(_res) == 2 else (0, [])
                    deleted_edges = deleted_edges if isinstance(deleted_edges, int) else 0
                    # the owner's FIELDS named after the entity (`user
                    # HAS_FOTINI_DESCRIPTION …`): a family member's go with
                    # her, anyone else's are listed (r8 review)
                    _fields = (await asyncio.to_thread(graph_memory.owner_field_edges, _entity)
                               if hasattr(graph_memory, "owner_field_edges") else [])
                    _fields = [e for e in _fields if isinstance(e, tuple) and len(e) == 3]
                    if _family_person:
                        for e in _fields:
                            deleted_edges += int(await asyncio.to_thread(graph_memory.delete_edge, *e) or 0)
                    else:
                        _kept = list(_kept) + [e for e in _fields if e not in _kept]
                        _pl = _FORGET_PLAN.get()
                        if _pl is not None:
                            for e in _fields:
                                _pl.add("graph_edge", {"s": e[0], "p": e[1], "o": e[2]},
                                        "graph " + " ".join(e) + " (a fact about you)", default=False)
                    if _kept:
                        report.append("ℹ️ Graph: facts about you that mention '" + _entity + "' were kept: "
                                      + "; ".join(" ".join(k) for k in _kept[:5])
                                      + (" …" if len(_kept) > 5 else "") + ".")
                else:
                    deleted_edges = await asyncio.to_thread(graph_memory.delete_by_target, _entity)
            if deleted_edges > 0:
                report.append(f"✅ Graph: Severed {deleted_edges} topological edges related to '{clean_target}'.")
        except Exception as e: report.append(f"⚠️ Graph Error: {e}")

    # 4b. Episodes that NAME the entity. An episode is the agent's record of
    # a whole turn, so a MENTION is not a reason to delete it (third review:
    # `forget python` would have removed 30, `forget athens` 18) — the same
    # rule as the profile leg. Deleted only for a PERSON of the owner's
    # family (the privacy case: `forget Fotini`); otherwise counted and kept.
    if episodic_memory is not None and hasattr(episodic_memory, "forget_mentions"):
        try:
            if _names_an_entity(clean_target) and not _is_owner_name(profile_memory, _entity):
                if _family_person and not _quals:
                    n_ep = await asyncio.to_thread(episodic_memory.forget_mentions, _entity, memory_system)
                    if n_ep:
                        report.append(f"✅ Episodes: Forgot {n_ep} episode(s) naming '{_entity}' (archived).")
                elif hasattr(episodic_memory, "count_mentions"):
                    n_m = await asyncio.to_thread(episodic_memory.count_mentions, _entity)
                    if n_m:
                        report.append(f"ℹ️ Episodes: {n_m} episode(s) mention '{_entity}' — kept (an episode is "
                                      f"a record of a whole turn, not a fact about it).")
        except Exception as e:
            report.append(f"⚠️ Episodes Error: {e}")

    # 5. Entity-aware secondary sweep over the target's graph neighbours.
    # LITERAL-mention only across vector + profile + graph so we excise the
    # alias tombstones ('iguana') without semantic over-reach.
    for extra in expanded_targets:
        extra_lc = str(extra).strip().lower()
        if len(extra_lc) < 3:
            continue
        # Vector: delete facts that literally name the related entity.
        try:
            def _literal_sweep(_t=extra):
                with memory_system._get_lock() if hasattr(memory_system, "_get_lock") else _NullCM():
                    # Same type scope as the primary sweep — see
                    # _FORGET_PROTECTED_TYPES above.
                    cand = memory_system.collection.query(
                        query_texts=[_t], n_results=20,
                        where={"type": {"$nin": _FORGET_PROTECTED_TYPES}})
                    n = 0
                    if cand.get('ids'):
                        for i in range(len(cand['ids'][0])):
                            doc_text = cand['documents'][0][i]
                            mem_id = cand['ids'][0][i]
                            meta = (cand.get('metadatas') or [[]])[0][i] or {}
                            if meta.get('type') in _FORGET_PROTECTED_TYPES:
                                continue
                            # §4R R2: `synthesis` is forgettable when the user
                            # NAMES the target (primary sweep — that is the
                            # tool's job, per the note on the protected list),
                            # but NOT here. This is the expansion sweep: `_t` is
                            # a graph NEIGHBOUR the user never mentioned. A
                            # synthesis is a COMPOSITE whose merged source
                            # fragments dream.py has already deleted, so
                            # dropping one to excise a single incidental token
                            # destroys the only surviving copy of everything
                            # else it merged. Deleting a composite on an
                            # unnamed term is exactly the "semantic over-reach"
                            # this literal-only sweep was written to avoid.
                            if meta.get('type') == "synthesis":
                                continue
                            if _value_mentions_target(doc_text, str(_t).strip().lower()):
                                memory_system.collection.delete(ids=[mem_id])
                                n += 1
                    return n
            n_vec = await asyncio.to_thread(_literal_sweep)
            if n_vec:
                report.append(f"✅ Vector: Wiped {n_vec} fact(s) mentioning related entity '{extra}'.")
        except Exception as e:
            report.append(f"⚠️ Vector (expansion) Error: {e}")

        # Graph: sever the neighbour's own edges too.
        if graph_memory:
            try:
                _rx = (await asyncio.to_thread(graph_memory.forget_entity, extra)
                       if hasattr(graph_memory, "forget_entity") else None)
                d_extra = _rx[0] if isinstance(_rx, tuple) and _rx and isinstance(_rx[0], int) else 0
                if d_extra and d_extra > 0:
                    report.append(f"✅ Graph: Severed {d_extra} edge(s) for related entity '{extra}'.")
            except Exception:
                pass

        # Profile: the alias's own values only — a field that merely mentions
        # it among other things is listed, never deleted (profile-writes review)
        if profile_memory:
            try:
                report.extend(_forget_profile(profile_memory, extra, related=True))
            except Exception:
                pass

    return "\n".join(report) if report else f"No matching memory found for '{target}'."

#: Scratchpad keys the SYSTEM owns — the current-project binding, swarm
#: results, turn checkpoints, project scopes. The model's tool may not set,
#: delete or clear them (§4LX: one `clear` parked the owner's project).
_SCRATCH_RESERVED_PREFIXES = ("__", "_swarm_task_id::", "_checkpoint_t", "proj::")
#: the scope background requests write into — never shown in an owner prompt
SCRATCH_BACKGROUND_NS = "bg"


def _scratch_reserved(key) -> bool:
    return str(key or "").startswith(_SCRATCH_RESERVED_PREFIXES)


def _scratch_request_kind() -> str:
    """'probe', 'background' or 'owner' for the current request — from the
    ONE shared classification (§4LZ A-F5). Self-play (`sim-`), test
    replays and job wakes write into the background scope."""
    try:
        from ..utils.logging import request_kind
        kind = request_kind()
    except Exception:  # noqa: BLE001
        return "owner"
    if kind == "probe":
        return "probe"
    if kind in ("background", "job", "test"):
        return "background"
    return "owner"


async def tool_scratchpad(action: str = None, scratchpad: Scratchpad = None, key: str = None, value: str = None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    icon = Icons.MEM_SCRATCH
    log_title = f"Scratch {str(action).upper()}"
    log_content = f"{key} = {value}" if value else key
    pretty_log(log_title, log_content, icon=icon)
    if not scratchpad:
        return "Error: Scratchpad memory is not initialized."
    action = str(action).strip().lower()
    if action in ("remove", "del", "unset"):
        action = "delete"
    kind = _scratch_request_kind()
    if action in ("set", "delete", "clear"):
        if kind == "probe":
            # §4LX: a probe's `pong_pending=PONG2` sat in the owner's prompt for 24 h
            return ToolOutcome.rejected(
                "Error: a probe request does not write the owner's scratchpad — nothing changed.",
                world_changed=False, reason_code="not_owner_write")
        if key is not None and _scratch_reserved(key):
            return ToolOutcome.rejected(
                f"Error: '{key}' is a system key (project binding, job result or checkpoint) — "
                f"the scratchpad tool cannot change it.", world_changed=False,
                reason_code="scratch_reserved_key")
    _bg_ns = SCRATCH_BACKGROUND_NS if kind == "background" else None
    if action == "set":
        # A key is required — set(None, ...) stores under key None and the
        # SQLite error is swallowed, reporting a no-op as success.
        if not key:
            return "SYSTEM ERROR: 'key' is required for scratchpad set."
        if value is None:
            # (§4LX) a None value was stored, then `get` said "not found"
            # while `list` and the prompt showed `todo: None`
            return "SYSTEM ERROR: 'value' is required for scratchpad set."
        if _bg_ns:
            # a background job's notes stay out of the owner's prompt (§4LX:
            # 14 chess-subscription notes rode every owner turn)
            return scratchpad.set(key, value, namespace=_bg_ns)
        return scratchpad.set(key, value)
    elif action == "get":
        if not key:
            return "SYSTEM ERROR: 'key' is required for scratchpad get."
        val = scratchpad.get(key)
        # Distinguish "stored a falsy value (0/''/False)" from "missing" —
        # `if val` reported a legit 0/""/[] as not-found.
        if val is None:
            return f"Error: '{key}' not found."
        return f"{key} = {val}"
    elif action == "list":
        return scratchpad.list_all()
    elif action == "delete":
        if not key:
            return "SYSTEM ERROR: 'key' is required for scratchpad delete."
        if _bg_ns and scratchpad.namespace_of(key) != _bg_ns:
            return "Error: a background request can only delete its own notes."
        return (f"Deleted '{key}'." if scratchpad.delete(key) else f"Error: '{key}' not found.")
    elif action == "clear":
        # ONLY the active scope, never the system's keys (§4LX: the tool's
        # clear wiped every project's notes, the project binding, swarm
        # results and other requests' checkpoints — live, on "/xlear")
        ns = _bg_ns or getattr(scratchpad, "active_namespace", None)
        protect = {k for k in scratchpad.keys() if _scratch_reserved(k)}
        gone = scratchpad.clear_namespace(ns, protect=protect)
        return (f"Cleared {len(gone)} note(s) from "
                + (f"scope '{ns}'" if ns else "the general scope")
                + ". Other projects' notes and system keys were kept.")
    return "Error: Unknown action (set, get, list, delete, clear)"

async def sync_owner_mirrors(category, key, profile_memory, graph_memory=None, memory_system=None) -> list:
    """§4KZ: the PROFILE is the authority for an owner field; its graph edges
    (`user HAS_<KEY>`) and vector rows (`User <key> is …`) are MIRRORS made to
    equal it after every write — by the tool, the bus or the background
    extractor — instead of each writer adding its own copy (stale HAS_WIFE /
    HAS_EMPLOYER / HAS_LOCATION edges stayed live beside the new value).
    Returns the names of mirrors that could not be synced."""
    from ..memory.profile import ProfileMemory as _PM
    cat, k = _PM.canonicalize(str(category or "").strip().lower(), str(key or "").strip().lower())
    try:
        cur = ((profile_memory.load() or {}).get(cat) or {}).get(k)
    except Exception:  # noqa: BLE001
        return ["profile-read"]
    values = cur if isinstance(cur, list) else ([] if cur in (None, "") else [cur])
    lag = []
    for name, store in (("graph", graph_memory), ("vector", memory_system)):
        if store is None or not hasattr(store, "sync_owner_field"):
            continue
        try:
            await asyncio.to_thread(store.sync_owner_field, k, values)
        except Exception as e:  # noqa: BLE001
            logger.warning("owner mirror %s for %s.%s lagged: %s", name, cat, k, e)
            lag.append(name)
    return lag


async def tool_update_profile(category: str = None, key: str = None, value: str = None, profile_memory=None, memory_system=None, graph_memory=None, memory_bus=None, **kwargs):
    """Persist a profile field. Bus-aware path emits an `update_profile`
    event so the bus handles every downstream commit (vector smart-update +
    graph triplet); legacy direct path retained for tests."""
    _blocked = _owner_write_block()
    if _blocked is not None:
        return _blocked
    category = category or kwargs.get("category", "root")
    key = key or kwargs.get("key")
    value = value if value is not None else kwargs.get("value")

    if not key:
        return "Error: 'key' is a required argument for update_profile."

    # Fresh review (§4KW): ANY falsy value deleted — a key-only call (the model
    # reaching for the tool to READ the profile, 3 live), None, 0. The
    # owner's root.name was removed this way. Only an explicit "" deletes.
    if value is None:
        return ("Error: 'value' is required. update_profile only WRITES; the profile is already in your "
                "context. To delete a stored fact, pass value=\"\" explicitly. Nothing was changed.")
    # (profile-writes review) `[]`, `{}`, "null", "None", a zero-width space
    # were stringified and OVERWROTE the stored value. Only an exact ""
    # deletes; any other empty-looking value changes nothing.
    from ..memory.profile import _is_empty_value
    if value != "" and _is_empty_value(value):
        return "Error: 'value' is empty. To delete a stored fact, pass value=\"\" explicitly. Nothing was changed."
    if not isinstance(value, str):
        value = str(value)

    if value == "":
        # DELETE path: an empty/omitted value removes the key — mirroring
        # `manage_projects config`, where an empty config_value deletes.
        # ProfileMemory.delete() existed but was unreachable from the tool;
        # live 2026-07-05 the model reasonably tried exactly this call
        # shape to remove a field, got a hard error, its corrected retry
        # was idempotency-blocked, and the turn finalised on a false
        # "Done — removed".
        prof = profile_memory
        if prof is None and memory_bus is not None:
            prof = getattr(memory_bus, "profile", None)
        if prof is None or not hasattr(prof, "delete"):
            return "Error: Profile memory not loaded."
        # §4M R2 MINOR-6: the WRITE side files under the canonical field
        # (vehicle → assets.car) and mints the vector fact from the
        # canonical key — this delete path read the RAW category/key, so
        # the old-value lookup missed, delete_fragment never ran, and the
        # derived identity fact stayed retrievable forever after deletion.
        from ..memory.profile import ProfileMemory as _PM
        _cat_c, _key_c = _PM.canonicalize(category, key)
        old_val = None
        try:
            data = prof.load() if hasattr(prof, "load") else None
            if isinstance(data, dict):
                cat_data = data.get(_cat_c, {})
                if isinstance(cat_data, dict):
                    old_val = cat_data.get(_key_c)
        except Exception:
            pass
        pretty_log("Profile Update", f"delete {category}.{key}",
                   icon=Icons.USER_ID)
        msg = await asyncio.to_thread(prof.delete, category, key)
        if old_val is not None and isinstance(msg, str) and not msg.lower().startswith("error"):
            msg = f"{msg} (was: {str(old_val)[:200]!r})"
        # the mirrors follow the profile (§4KZ: one sync, every value)
        if isinstance(msg, str) and not msg.lower().startswith("error"):
            await sync_owner_mirrors(category, key, prof,
                                     graph_memory if graph_memory is not None else getattr(memory_bus, "graph", None),
                                     memory_system if memory_system is not None else getattr(memory_bus, "vector", None))
        return msg

    pretty_log("Profile Update", f"{category}.{key}={value}", icon=Icons.USER_ID)

    # TEMPORAL ANCHORING, done ONCE and up front. ProfileMemory.update()
    # anchors at its own boundary regardless (nothing can bypass it), but
    # this function ALSO mints a vector fact ("User <key> is <value>") and
    # a graph triplet from the raw value. Anchoring only inside the profile
    # store would leave those two siblings holding the decaying snapshot
    # while the profile held the anchor — the exact three-stores-disagree
    # shape §4M MAJOR-4 already had to fix once for the canonical key.
    # anchor() is idempotent, so the second application downstream is a
    # no-op.
    value = _anchor_temporal(value)

    # --- DEDUP: short-circuit when the stored value already equals the new
    # value. This is the second-line defence against the production loop bug
    # where the model called update_profile(location=Athens, Greece) 9× in a
    # row. The agent-loop idempotency guard catches it within a request; this
    # check catches it across requests / cold reloads.
    _was_note = ""
    profile_for_check = profile_memory
    if profile_for_check is None and memory_bus is not None:
        profile_for_check = getattr(memory_bus, "profile", None)
    if profile_for_check is not None:
        try:
            data = profile_for_check.load() if hasattr(profile_for_check, "load") else None
            if isinstance(data, dict):
                # the CANONICAL field (vehicle → assets.car), as the write files it
                from ..memory.profile import ProfileMemory as _PMc
                cat_lc, key_lc = _PMc.canonicalize(str(category).strip().lower(), str(key).strip().lower())
                cat_data = data.get(cat_lc, {}) if isinstance(data.get(cat_lc), dict) else {}
                existing = cat_data.get(key_lc)
                if existing is not None and str(existing).strip() == str(value).strip():
                    return f"NOOP: Profile already has {category}.{key} = {value}. No change applied."
                # a single-value field being REPLACED: the model is told what
                # it overwrote (third review — "(was: …)" never reached it)
                from ..memory.profile import _SINGLETON_KEYS as _SK
                if existing is not None and not isinstance(existing, list) and key_lc in _SK:
                    _was_note = f" (was: {str(existing)[:200]!r})"
        except Exception:
            pass

    # --- BUS-AWARE PATH ---
    if memory_bus is not None:
        # §4M (Lens C MAJOR-4): canonicalise BEFORE composing — the
        # profile leg rewrites synonyms internally (vehicle → assets.car)
        # while this triplet was minted from the RAW key (HAS_VEHICLE), so
        # graph and profile permanently disagreed on the field name. One
        # canonical form now feeds all three stores.
        from ..memory.profile import ProfileMemory as _PM
        _cat_c, _key_c = _PM.canonicalize(category, key)
        _report = await memory_bus.publish_fact("update_profile", {
            # the profile only; its mirrors are SYNCED below (§4KZ)
            "profile_update": {"category": _cat_c, "key": _key_c, "value": value},
        })
        _fails = _bus_write_failures(_report)
        if _fails:
            _canon = _bus_canonical_failed(_report, "update_profile")
            _mk = ToolOutcome.failed if _canon else ToolOutcome.partial
            _head = "FAILED" if _canon else "PARTIAL"
            return _mk((f"{_head}: Profile update had failures — "
                    f"{'; '.join(_fails)}. Retrieval may not reflect the change."),
                    reason_code="profile_write_partial")
        _lag = await sync_owner_mirrors(_cat_c, _key_c, getattr(memory_bus, "profile", None),
                                        getattr(memory_bus, "graph", None), getattr(memory_bus, "vector", None))
        if _lag:
            return ToolOutcome.partial(f"PARTIAL: Profile updated, but the {', '.join(_lag)} mirror lagged.",
                                       reason_code="profile_graph_lag")
        return f"SUCCESS: Profile updated.{_was_note}"

    # --- LEGACY DIRECT PATH ---
    if not profile_memory: return "Error: Profile memory not loaded."
    msg = await asyncio.to_thread(profile_memory.update, category, key, value)
    if isinstance(msg, str) and msg.lower().startswith("error"):
        return ToolOutcome.failed(msg, reason_code="profile_write_refused")

    # the mirrors follow the profile (§4KZ)
    partial_failures = await sync_owner_mirrors(category, key, profile_memory, graph_memory, memory_system)

    if partial_failures:
        return (
            ToolOutcome.partial(f"PARTIAL: Profile updated (canonical JSON), but "
            f"{', '.join(partial_failures)} index(es) lagged. "
            f"Semantic / graph retrieval may not yet reflect this change.", reason_code="profile_graph_lag")
        )
    return f"SUCCESS: Profile updated.{_was_note}"

async def tool_learn_skill(task: str = None, mistake: str = None, solution: str = None, skill_memory=None, memory_system=None, memory_bus=None, **kwargs):
    """Save a learned lesson. Bus-aware path emits a `learn_skill` event
    so SkillMemory + VectorMemory commits flow through the bus."""
    if not task or not mistake or not solution:
        return "SYSTEM ERROR: 'task', 'mistake', and 'solution' parameters are MANDATORY."

    # --- DEDUP: refuse to re-learn an identical (task, mistake, solution)
    # triplet. Without this the playbook bloats with duplicates and the
    # vector store re-embeds the same lesson text every time.
    skill_for_check = skill_memory
    if skill_for_check is None and memory_bus is not None:
        skill_for_check = getattr(memory_bus, "skill", None)
    if skill_for_check is not None:
        try:
            import json as _json
            file_path = getattr(skill_for_check, "file_path", None)
            if file_path is not None and file_path.exists():
                playbook = _json.loads(file_path.read_text() or "[]")
                if isinstance(playbook, list):
                    for entry in playbook:
                        if (entry.get("task") == task
                                and entry.get("mistake") == mistake
                                and entry.get("solution") == solution):
                            return "NOOP: Identical lesson already in the Skill Playbook. No duplicate written."
        except Exception:
            pass

    # --- BUS-AWARE PATH ---
    if memory_bus is not None:
        _report = await memory_bus.publish_fact("learn_skill", {
            "skill": {"task": task, "mistake": mistake, "solution": solution},
        })
        _fails = _bus_write_failures(_report)
        if _fails:
            _canon = _bus_canonical_failed(_report, "learn_skill")
            _mk = ToolOutcome.failed if _canon else ToolOutcome.partial
            _head = "FAILED" if _canon else "PARTIAL"
            return _mk((f"{_head}: lesson write had failures — "
                    f"{'; '.join(_fails)}. It may not be in the playbook."),
                    reason_code="lesson_write_partial")
        return _learn_skill_success((_report or {}).get("skill_scope") if isinstance(_report, dict) else "")

    # --- LEGACY DIRECT PATH ---
    if not skill_memory: return "Error: Skill memory not active."
    _w = skill_memory.learn_lesson(task, mistake, solution, memory_system=memory_system, source="learn_skill",
                                   generality_context=_current_request_text())
    if _w is None:
        # (r8 review) a dropped lesson was reported as saved
        return ToolOutcome.failed("FAILED: the lesson was not written (dropped by the playbook's quality gates).",
                                  reason_code="lesson_write_partial")
    return _learn_skill_success(getattr(_w, "scope", ""))


def _learn_skill_success(scope) -> str:
    """Say WHERE the lesson applies (re-review: a lesson scoped to this one
    request was reported as a plain SUCCESS). ``scope`` is THIS call's."""
    if isinstance(scope, str) and scope == "request":
        return ("SUCCESS: Lesson saved — for THIS request only: it restates the request's own details, so it "
                "will be recalled when this request comes up again, not for other tasks. To make it a general "
                "rule, phrase it without this request's names, files and numbers.")
    return "SUCCESS: Lesson learned and saved to the Skill Playbook and Vector Memory."

#: Every kwarg name the `knowledge_base` dispatcher accepts as the SUBJECT of
#: an action — the fact to store, the file to ingest, the topic to forget.
#: The tuple is the FALLBACK order; `_kb_tried_names` hoists the action's own
#: schema name in front of it, so the effective order differs per action.
#:
#: ⚠ Appending a name here is NOT automatically safe. `target` was appended
#: last and still changed how 63 existing `forget` calls resolve, because it
#: is also that action's `primary`. And a name added here becomes a name the
#: generic resolution below can hand to `query` / `expand` / `update_profile`.
#: Add one only after checking both.
#:
#: Anything that tells a model "parameter X is MANDATORY" must derive X from
#: this tuple — see `_kb_target_or_error`, the only place in this module that
#: builds such a message. Hand-writing one is how this bug happened: the inner
#: `tool_unified_forget` demanded a 'target' parameter that was neither
#: advertised in the schema NOR accepted here, so a model that complied
#: exactly got a byte-identical error back. Seen live 2026-08-28 on "forget
#: everything about X": every retry was the same call, the same error, until
#: the strike budget ran out. An error that names a parameter the tool then
#: drops is not a bad message — it is an unbreakable loop.
_KB_TARGET_ALIASES = (
    "filename", "fact", "content", "source", "path", "topic", "target",
)

#: The actions the schema advertises, in schema order. One home: the registry
#: enum is pinned against this tuple, and the "unknown action" error is
#: generated from it. (`update_profile` is dispatched but deliberately not
#: advertised; see the branch at the end of `tool_knowledge_base`.)
_KB_ACTIONS = (
    "transcribe", "ingest_document", "query", "outline", "transcript", "insert_fact",
    "expand", "forget", "list_docs", "reset_all",
)


#: Characters that turn a prose hint into something that reads like a call.
#: Stripped from every hint before it reaches a model. Includes the
#: typographic forms — a curly apostrophe is what you get from pasting prose
#: out of a document, and `subject=‘project atlas’` renders a
#: perfectly copyable call while an ASCII-only class waves it through.
_KB_HINT_SHAPE = re.compile("[=＝\"'`‘’“”]+")


def _kb_tried_names(primary: str) -> tuple:
    """Every kwarg name a subject lookup for `primary` consults, in order.

    The action's own schema name goes first so a caller that passes both its
    advertised name and a legacy alias gets what it asked for.
    """
    return (primary,) + tuple(n for n in _KB_TARGET_ALIASES if n != primary)


def _kb_target_or_error(kwargs: dict, action: str, primary: str,
                        hint: str, example: str):
    """Resolve an action's subject from `kwargs`; on failure build the error.

    Returns ``(value, None)`` or ``(None, error_string)``.

    Lookup and message are one function because they must agree: the only
    parameter the message names is `primary`, which the lookup tries first,
    and the worked call is generated from it. `hint` and `example` are PROSE
    AND A VALUE — neither may contain a parameter name.

    ⚠ That last sentence is a constraint, not a guarantee. An earlier version
    let each call site write its own worked example inside `hint`, and a
    review showed the whole live bug reproducing byte-for-byte after changing
    one hint's ``target=`` to ``subject=`` — with every regression test
    green, because the pins scraped only quoted lowercase tokens and a hint's
    parameter appears as bare ``name='value'``. The example is generated here
    now, and `test_kb_missing_param_names_accepted_param.py` scans the
    rendered message for any identifier that is neither the action nor a
    tried name.

    ⚠ It also used to LIST the other accepted aliases, so a model reading
    ``insert_fact``'s error was told it "also accepts 'filename'" — and
    obeying that stores the literal filename as a permanent fact and returns
    SUCCESS. Excluding the other actions' names fixed the example and not the
    property: ``source`` and ``path`` are just as filename-shaped, and were
    still advertised. The alternatives are gone entirely. They were never
    what ended the loop — the required name and the worked call are — and a
    caller already passing a legacy alias never sees this message at all.
    The aliases stay ACCEPTED for back-compat; they are simply not advice.
    """
    for name in _kb_tried_names(primary):
        val = kwargs.get(name)
        if not val:
            continue
        if isinstance(val, str):
            # STRIP, don't merely test. An earlier version computed
            # `val.strip()` to decide the subject was present and then
            # returned the padded original — and `tool_unified_forget`
            # strips in only 3 of its 6 uses, so `target=' atlas '` (the
            # normal XML shape: the argument parser strips CR/LF, not
            # spaces) skipped the disk and document sweeps while reporting
            # every stage with a ✅. Normalise once, at the boundary.
            val = val.strip()
            if not val:
                continue          # whitespace is not a subject
            return val, None
        # A non-string subject is not a subject. `target=['a','b']` is a
        # plausible native-JSON shape for "forget X and Y", and it used to
        # sail through: the vector and profile sweeps raised and were
        # caught, the graph sweep "succeeded" against the repr, NOTHING was
        # deleted, and the turn was booked as a clean success because the
        # report never starts with an error prefix. The schema says string.
        return None, (
            f"SYSTEM ERROR: The '{primary}' parameter is MANDATORY for "
            f"knowledge_base(action='{action}') and must be a single "
            f"string, not {type(val).__name__}. Worked call: "
            f"knowledge_base(action='{action}', {primary}={example!r}). "
            f"For several subjects, make one call each."
        )
    # A hint may not carry a parameter name. This is the channel that
    # reproduced the whole live loop under review: hints are free text, and
    # one reading "...or subject='project atlas'" is a call shape a model
    # will copy and have dropped. `=` and quotes are what make a fragment
    # look callable, so they are removed here rather than trusted to review.
    # (Prose that merely NAMES a field in passing is not reachable by this
    # rule — see the module docstring of the regression test.)
    hint = _KB_HINT_SHAPE.sub(" ", str(hint or "")).strip()
    return None, (
        f"SYSTEM ERROR: The '{primary}' parameter is MANDATORY for "
        f"knowledge_base(action='{action}') — {hint}. Worked call: "
        f"knowledge_base(action='{action}', {primary}={example!r})."
    )


def _kb_unknown_action_error(action: str) -> str:
    """Same contract as `_kb_target_or_error`, for the ACTION slot.

    "Unknown action 'delete'" named nothing the caller could switch to, and
    `delete`/`erase`/`remove` are not in the alias map — so the model's next
    guess was another guess. The valid set is generated from `_KB_ACTIONS`.
    "Unknown action" is kept verbatim because an existing pin asserts on it
    (test_transcribe_discoverability). Note it was never FATAL-classified on
    its own — the FATAL class comes from the `MANDATORY` token this wording
    adds, which is deliberate: an unknown action is a caller error, and the
    caller now has the list it needs to fix it.
    """
    valid = ", ".join(repr(a) for a in _KB_ACTIONS)
    if not action:
        return (f"SYSTEM ERROR: The 'action' parameter is MANDATORY — it "
                f"must be one of: {valid}.")
    return (f"SYSTEM ERROR: Unknown action '{action}'. The 'action' "
            f"parameter is MANDATORY and must be one of: {valid}.")


async def tool_knowledge_base(action: str = None, sandbox_dir: Path = None, memory_system=None, memory_bus=None, **kwargs):
    _blocked = _member_block()
    if _blocked is not None:
        return _blocked
    if not action:
        return _kb_unknown_action_error("")
    # --- ACTION ALIASES ---------------------------------------------------
    # `transcribe` is a FIRST-CLASS name for `ingest_document`, not a typo
    # heal. The tool is named for what it STORES while a model searching for
    # this capability is holding a VERB: it thinks "I need to transcribe",
    # scans the tool list for a transcriber, finds none, and plans a Whisper
    # pipeline instead. Measured (§4AW): `knowledge_base` was advertised on
    # all 16 tool-carrying payloads, sat 3rd of 44, and its description
    # already forbade writing transcription code — the model installed
    # openai-whisper anyway. Presence was never the problem; findability BY
    # NEED was. So the tool now answers to the word the model is looking for.
    action = str(action).strip().lower()
    # The name the CALLER reached for, kept for the error messages. §4AW made
    # `transcribe` a first-class verb because a model holding it could not
    # find this tool; rendering the canonical name back at that model in the
    # one worked call it is given renames its verb to the un-findable one.
    action_as_called = action
    action = {
        "transcribe": "ingest_document",
        "transcribe_document": "ingest_document",
        "transcription": "ingest_document",
        "ingest": "ingest_document",
        "ingest_file": "ingest_document",
    }.get(action, action)
    # --- FLEXIBLE PARAMETER MAPPING ---
    # Schema advertises 'filename' (ingest_document/query), 'fact'
    # (insert_fact) and 'target' (forget); legacy 'content'/'source'/'path'/
    # 'topic' kept for back-compat with older callers and Qwen variants that
    # aliased. Derived from _KB_TARGET_ALIASES so the accepted set is stated
    # exactly once. This is the GENERIC resolution, used by the branches that
    # only fall back to it; the subject-taking actions below re-resolve with
    # their own schema name first via _kb_target_or_error.
    target = next((kwargs[n] for n in _KB_TARGET_ALIASES if kwargs.get(n)), None)

    # The three subject-taking actions resolve (and, when empty, complain)
    # through _kb_target_or_error so the error a model reads always names a
    # parameter this dispatcher accepts. The inner tools keep their own
    # guards for DIRECT callers, but their messages name THEIR OWN parameter
    # ('text' for tool_remember, 'target' for tool_unified_forget), which is
    # not necessarily a name a model may pass — so they must never be the
    # message a tool call surfaces.
    if action == "insert_fact":
        fact, err = _kb_target_or_error(
            kwargs, action_as_called, "fact",
            "pass the single discrete fact to memorise",
            "<the fact to remember>")
        if err:
            return err
        return await tool_remember(fact, memory_system, kwargs.get("graph_memory"), kwargs.get("llm_client"), kwargs.get("model_name", "default"), memory_bus=memory_bus)

    elif action == "ingest_document":
        _is_media_verb = action_as_called != "ingest_document"
        filename, err = _kb_target_or_error(
            kwargs, action_as_called, "filename",
            ("pass the name of an EXISTING audio or video file in your sandbox, "
             "or a YouTube link"
             if _is_media_verb else
             "pass the name of an EXISTING file in your sandbox, or a web URL"),
            "<your-recording.mp4>" if _is_media_verb else "<your-file.pdf>")
        if err:
            return err
        return _declare_ingest(await tool_gain_knowledge(
            filename, sandbox_dir, memory_system,
            tor_proxy=kwargs.get("tor_proxy"), language=kwargs.get("language")))

    elif action == "forget":
        # Two steps (§4KX r8): without `confirm` this is a PREVIEW that
        # deletes nothing; with the token, in a later turn, it deletes
        # exactly the confirmed items.
        if kwargs.get("confirm"):
            return await forget_execute(kwargs.get("confirm"), kwargs.get("items") or "all", sandbox_dir,
                                        memory_system, kwargs.get("profile_memory"), kwargs.get("graph_memory"),
                                        project_store=kwargs.get("project_store"),
                                        episodic_memory=kwargs.get("episodic_memory"),
                                        skill_memory=kwargs.get("skill_memory"))
        subject, err = _kb_target_or_error(
            kwargs, action_as_called, "target",
            "pass the topic, entity or filename to erase",
            "<the topic to erase>")
        if err:
            return err
        subject = _norm_doc_name(subject)
        return await forget_preview(subject, sandbox_dir, memory_system, kwargs.get("profile_memory"),
                                    kwargs.get("graph_memory"), project_store=kwargs.get("project_store"),
                                    episodic_memory=kwargs.get("episodic_memory"),
                                    skill_memory=kwargs.get("skill_memory"))

    elif action == "query":
        return await tool_query_document(
            filename=_norm_doc_name(kwargs.get("filename") or kwargs.get("source") or target),
            question=(kwargs.get("question") or kwargs.get("query")
                      or kwargs.get("q")),
            memory_system=memory_system,
        )

    elif action == "transcript":
        return await tool_document_transcript(
            filename=_norm_doc_name(kwargs.get("filename") or kwargs.get("source") or target),
            offset=kwargs.get("offset", 0),
            max_chars=kwargs.get("max_chars", TRANSCRIPT_PAGE_CHARS),
            memory_system=memory_system,
        )

    elif action == "outline":
        return await tool_document_outline(
            filename=_norm_doc_name(kwargs.get("filename") or kwargs.get("source") or target),
            memory_system=memory_system,
            depth=kwargs.get("depth", 2),
        )

    elif action == "expand":
        return await tool_expand_evidence(
            ref=kwargs.get("ref") or kwargs.get("id") or target,
            episodic_memory=kwargs.get("episodic_memory"),
            session_store=kwargs.get("session_store"),
        )

    elif action == "list_docs":
        # Names AND shape. A bare name list is what sent request e0f4a8bd
        # into 20 turns of semantic guessing: the model could see the manual
        # existed and nothing about it, so every structural question had to
        # be asked as a search. The counts here come from the cached outline
        # record (free); a document with none says so and names the call
        # that builds it, rather than silently reporting nothing.
        if not memory_system: return "Error: Memory system is disabled."
        library = await asyncio.to_thread(memory_system.get_library)
        library = library or []
        if not library:
            return "No docs."
        lines = [f"LIBRARY CONTENTS ({len(library)} files):"]
        for doc in library:
            try:
                rec = await asyncio.to_thread(
                    memory_system.get_document_outline, doc)
            except Exception:  # noqa: BLE001
                rec = {}
            facts = []
            if rec.get("pages"):
                facts.append(f"{rec['pages']} pages")
            if rec.get("chunks"):
                facts.append(f"{rec['chunks']} chunks")
            counts = _outline_level_counts(rec.get("entries") or [])
            if counts:
                facts.append("outline " + "/".join(
                    str(counts[lvl]) for lvl in sorted(counts))
                    + " entries by level")
            lines.append(f"- {doc}" + (f"  ({' · '.join(facts)})" if facts else ""))
        if not any("outline" in ln for ln in lines[1:]):
            lines.append("Structure (parts/chapters/sections and their counts): "
                         "knowledge_base(action='outline', filename=...).")
        return "\n".join(lines)

    elif action == "reset_all":
        if not memory_system: return "Error: Memory system is disabled."
        # (profile-writes review) one model call wiped the vector store and
        # the graph. Only when THIS turn's request asks for it in words —
        # never inferred from "clean up", "forget that", a lesson or a plan.
        _blk = _member_block()
        if _blk is not None:
            return _blk
        # (§4KX r8) the request-wording gate opened on negations, questions
        # and scoped requests and refused plain "yes" confirmations. The
        # wipe is now ALWAYS two steps: a preview with the counts and a
        # token, then the token in a LATER turn after the user says yes.
        _tok = str(kwargs.get("confirm") or "").strip()
        if not _tok:
            return await asyncio.to_thread(_reset_preview, memory_system, kwargs.get("graph_memory"),
                                           kwargs.get("episodic_memory"), kwargs.get("skill_memory"))
        _tok, _plan = _resolve_plan(_tok, "reset_all")
        if _plan is None:
            return ToolOutcome.rejected("NOT executed: unknown or expired reset_all token — run reset_all "
                                        "without confirm to get a new preview.", reason_code="reset_token_unknown")
        _why = _confirm_allowed(_plan)
        if _why:
            return ToolOutcome.rejected(f"NOT executed: {_why}.", reason_code="reset_not_confirmed")
        _drop_plan(_tok)
        # OFF THE EVENT LOOP, and without materialising the store.
        # `collection.get()` with no `include` returns every document body
        # and metadata blob — live, ~8k rows including 7k manual chunks —
        # to use nothing but the ids. It and every delete batch ran
        # synchronously on the loop, stalling every concurrent request,
        # stream and heartbeat for the duration, while the CHEAP graph wipe
        # below was already offloaded.
        _lock = (memory_system._get_lock()
                 if hasattr(memory_system, "_get_lock") else _NullCM())

        def _wipe():
            """Enumerate, delete, reset the catalogues — ONE critical
            section, off the event loop (§4GJ).

            The lock used to be taken per step: once for the scan, once per
            delete batch, and NOT AT ALL for the library reset, which ran
            last. An ingest landing after the snapshot therefore survived
            the wipe (its rows were not in the snapshot) while the unlocked
            reset erased its catalogue line — the row lived, its index line
            did not, and `ingest_document`'s dedup then refused to re-ingest
            a document `list_docs` could not see. Holding the lock across
            the whole sequence makes that interleaving impossible rather
            than unlikely; the wipe is an explicit operator action, so
            blocking concurrent memory writers for its duration is the
            correct trade.
            """
            deleted = 0
            failed_batches = 0
            orphaned: dict = {}
            with _lock:
                # ids AND metadatas in ONE scan. `include=["metadatas"]`
                # returns both (chroma always sends ids), so the orphan
                # count below describes exactly the rows this call is about
                # to delete. Two separate `get()`s meant the count came
                # from a different snapshot than the delete — rows landing
                # between them produced a note about documents that were
                # never removed.
                try:
                    snapshot = memory_system.collection.get(include=["metadatas"])
                except TypeError:
                    # Older chroma clients reject the kwarg.
                    snapshot = memory_system.collection.get()

                all_ids = snapshot.get("ids", []) or []
                # What this wipe ORPHANS, counted from the SAME snapshot.
                # `reset_all` deletes the `document` / `episode` / `skill` /
                # `acquired_skill` rows `_FORGET_PROTECTED_TYPES` protects,
                # because each has a record in ANOTHER store this does not
                # touch. `forget` refuses to create that asymmetry;
                # `reset_all` creates it by design, so it has to say so —
                # but only about rows that actually went.
                # Types positionally aligned with `all_ids`. Defensive
                # because the shape is the client's: a metadatas list
                # shorter than ids, a None entry, or a non-dict entry
                # (which raised AttributeError straight out of the tool,
                # deleting nothing and returning no error string).
                metas = snapshot.get("metadatas") or []
                types: list = []
                for i in range(len(all_ids)):
                    m = metas[i] if i < len(metas) else None
                    types.append(m.get("type") if isinstance(m, dict) else None)
                incomplete = len(metas) < len(all_ids)

                for i in range(0, len(all_ids), 500):
                    batch = all_ids[i:i + 500]
                    try:
                        memory_system.collection.delete(ids=batch)
                        deleted += len(batch)
                        # Count orphans only for rows that actually went.
                        # The first version emitted the note from a pre-scan
                        # regardless of outcome: with every batch failing it
                        # reported "this removed the vector rows for 600
                        # document…" having removed nothing.
                        for t in types[i:i + 500]:
                            if t in _FORGET_PROTECTED_TYPES:
                                orphaned[t] = orphaned.get(t, 0) + 1
                    except Exception as e:
                        failed_batches += 1
                        __import__("logging").getLogger("GhostAgent").warning(
                            f"reset_all batch {i // 500} failed: {e}"
                        )

                # Atomic catalogue reset, INSIDE the same critical section.
                # NOT when ANY batch failed: emptying the catalogues while
                # rows survive leaves the store and its indexes disagreeing
                # — exactly what the message means by "left in place", and
                # ingest dedups on the library, so an un-listed surviving
                # document can neither be queried nor re-ingested without
                # duplicating it.
                if not failed_batches:
                    # Both sidecars, together: a wiped document that keeps
                    # its outline serves that structure to the next
                    # same-named ingest until the ingest finishes (§4FO).
                    # `isinstance(Path)` rather than `hasattr`: a store
                    # whose catalogue path is not a real path (a stub) has
                    # no catalogue to reset.
                    for attr, empty in (("library_file", "[]"),
                                        ("outlines_file", "{}")):
                        path = getattr(memory_system, attr, None)
                        if not isinstance(path, Path):
                            continue
                        try:
                            tmp = path.with_suffix(path.suffix + ".tmp")
                            tmp.write_text(empty)
                            os.replace(tmp, path)
                        except Exception as e:
                            __import__("logging").getLogger("GhostAgent").warning(
                                f"reset_all {attr} reset failed: {e}")
            return deleted, failed_batches, orphaned, incomplete

        try:
            deleted, failed_batches, orphaned, report_note_incomplete = (
                await asyncio.to_thread(_wipe))
        except Exception as e:
            return f"Error: failed to enumerate vector store: {e}"

        if kwargs.get("graph_memory"):
            try:
                await asyncio.to_thread(kwargs.get("graph_memory").wipe_all)
            except Exception as e:
                __import__("logging").getLogger("GhostAgent").warning(f"reset_all graph wipe failed: {e}")
        # §4LA: the episodes too — their vector twins went above, and the boot
        # reconcile re-indexed every episode, so "wiped" memory came back
        _em = kwargs.get("episodic_memory")
        if _em is not None and hasattr(_em, "wipe_all"):
            try:
                await asyncio.to_thread(_em.wipe_all)
                orphaned.pop("episode", None)
            except Exception as e:  # noqa: BLE001
                __import__("logging").getLogger("GhostAgent").warning(f"reset_all episode wipe failed: {e}")
        # §4LC: the one-request lessons quote the owner's requests verbatim —
        # they go too (the boot reconcile re-indexed them after the vector wipe)
        _sk = kwargs.get("skill_memory")
        _req_gone = 0
        if _sk is not None and hasattr(_sk, "remove_request_scoped"):
            try:
                _req_gone = await asyncio.to_thread(_sk.remove_request_scoped, memory_system)
            except Exception as e:  # noqa: BLE001
                __import__("logging").getLogger("GhostAgent").warning(f"reset_all lesson wipe failed: {e}")
        # §4LK: the queued corrections quote the owner's conversations
        _ag = kwargs.get("owner_agent")
        if _ag is not None and hasattr(_ag, "clear_pending_corrections"):
            try:
                _ag.clear_pending_corrections()
            except Exception as e:  # noqa: BLE001
                __import__("logging").getLogger("GhostAgent").warning(f"reset_all correction-queue wipe failed: {e}")
        note = (f" Removed {_req_gone} one-request lesson(s) (archived)." if _req_gone else "")
        if report_note_incomplete:
            note += (" NOTE: the store returned fewer metadata rows than ids,"
                    " so the list of orphaned records below is incomplete.")
        if orphaned:
            note += (
                " NOTE: this removed the vector rows for "
                + ", ".join(f"{n} {t}" for t, n in sorted(orphaned.items()))
                + ". Their records in the skill store are NOT deleted by this"
                " action; the boot reconcile re-indexes them."
            )
        if failed_batches:
            return ToolOutcome.partial(
                f"PARTIAL: Wiped {deleted} entries; {failed_batches} "
                f"batch(es) failed and were left in place.{note}",
                world_changed=True, reason_code="wipe_partial")
        return f"Success: Wiped clean ({deleted} entries removed).{note}"

    elif action == "update_profile":
        # NOT a knowledge_base action. `update_profile` is advertised as its
        # OWN tool (registry.py), and this branch was an unadvertised
        # duplicate of it: absent from the action enum, reading
        # key/value/category — none of which the knowledge_base schema
        # carries — and returning "Error: 'key' is a required argument",
        # which classifies UNKNOWN rather than FATAL. Worse, `is_mutating`
        # (agent.py) counts it while `is_idempotent_setter` does not, so it
        # was the one route that bypassed the repeat-write guard written for
        # exactly this call. And `cat = category or target` let the generic
        # alias chain file a fact under a category named after a PDF.
        # Reviewed 2026-08-28: it was the last branch still violating the
        # invariant the rest of this dispatcher now holds — an action the
        # schema does not describe cannot be called correctly by a model
        # that reads the schema. Redirect instead of dispatching.
        # Deliberately does NOT say "call update_profile instead": that tool
        # is in `disabled_tools` for subagents, self-play and dream, where
        # the redirect would bounce the model between two errors with
        # different signatures — so the same-failure loop breaker never fires
        # and the 6-strike budget drains. Naming the actions THIS tool has is
        # advice that is true in every context.
        return _kb_unknown_action_error("update_profile")

    return _kb_unknown_action_error(action)

async def tool_dream_mode(context):
    """
    Manually triggers the Active Memory Consolidation (Dream Mode).
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    result = await dreamer.dream()
    from .outcome import append_note
    return append_note(result, "\n\nSYSTEM: SESSION FINISHED. STAND BY.")

#: Per-cycle wall-clock budget for `self_play`. Covers challenge
#: generation + all worker attempts end-to-end. A stuck worker with a
#: degenerate generation loop used to block the host for 20+ minutes;
#: this caps the damage at SELF_PLAY_CYCLE_TIMEOUT_S seconds, after
#: which the coroutine is cancelled and self-play returns an error
#: string the caller can surface. The streaming-loop detector should
#: abort long before we ever hit this wall, but the wall is the
#: last line of defence if the detector is disabled or a new failure
#: mode slips past it.
SELF_PLAY_CYCLE_TIMEOUT_S = 600.0


#: Substrings that count as an explicit self-play request from the user.
#: Matching is done on a lowercased, whitespace-normalised version of
#: ``context.last_user_content``. The list is deliberately conservative —
#: these are phrasings that map unambiguously to "run the self-play
#: curriculum". Plain words like "practice" alone are NOT enough (the user
#: might say "practice good git hygiene" with zero intent to train).
_SELF_PLAY_INTENT_PHRASES = (
    "self play",
    "self-play",
    "selfplay",
    "run self play",
    "run self-play",
    "start self play",
    "start self-play",
    "practice self-play",
    "practice self play",
    "synthetic self-play",
    "synthetic self play",
    "train yourself",
    "train on your own",
    "run a training cycle",
    "run training cycle",
    "training cycle",
    "run a practice cycle",
    "practice cycle",
    "practice round",
    "practice session",
    "keep practicing",
    "keep training",
    "train until stopped",
    "train in a loop",
    "training loop",
)


def _user_asked_for_self_play(context) -> bool:
    """Return True iff the current turn's user text explicitly asked for
    self-play. Used as a hallucination guard on the ``self_play`` /
    ``self_play_loop`` tools.

    The tool is powerful and expensive — a spontaneous call by the LLM
    burns an LLM cycle (or many, in loop mode) and can hijack a
    user-facing turn. In the 2026-04-24 webOS incident the LLM
    fabricated "The user wants me to run self-play" 33 minutes into a
    webOS-building session where the user had never mentioned it.

    The biological watchdog's self-play phase bypasses this check
    because it never goes through ``tool_self_play`` — it calls
    ``Dreamer.synthetic_self_play`` directly. Background-launched loops
    launched via the tool WILL be guarded, which is the intent: only
    an explicit user ask should kick one off.
    """
    raw = getattr(context, "last_user_content", "") or ""
    if not raw:
        # No user turn in flight at all → refuse. The watchdog path
        # doesn't touch this helper, so this is correct.
        return False
    lc = " ".join(str(raw).lower().split())
    return any(phrase in lc for phrase in _SELF_PLAY_INTENT_PHRASES)


#: Standard refusal body returned to the LLM when the guard trips.
#: Phrased to redirect the model back to the original request rather
#: than apologise or loop. Kept as a module constant so tests can pin
#: the wording (the LLM's behaviour depends on seeing keywords like
#: "REFUSED" and "did not request").
_SELF_PLAY_INTENT_REFUSAL = (
    "SYSTEM: SELF-PLAY REFUSED — the user did not request self-play or a "
    "training cycle in this turn. The `self_play` / `self_play_loop` tools "
    "are only for explicit user asks (e.g. 'run self-play', 'train until "
    "stopped', 'practice cycle'). Do NOT call this tool again unless the "
    "user's most recent message explicitly asks for it. Resume the "
    "original task."
)


async def tool_self_play(context):
    """
    Manually triggers the Synthetic Self-Play curriculum.
    """
    import asyncio
    from ..core.dream import Dreamer
    from ..utils.logging import pretty_log, Icons
    if not _user_asked_for_self_play(context):
        pretty_log(
            "Self-Play Refused",
            "LLM invoked `self_play` but the user's current turn doesn't ask for it. "
            "Refusing and redirecting the model back to the original task.",
            level="WARNING", icon=Icons.STOP,
        )
        return _SELF_PLAY_INTENT_REFUSAL
    dreamer = Dreamer(context)
    # §4KS: the whole call is bounded below, so a template retry after a
    # defective generated challenge must fit what is left of that bound.
    dreamer.cycle_budget_s = SELF_PLAY_CYCLE_TIMEOUT_S
    try:
        result = await asyncio.wait_for(
            dreamer.synthetic_self_play(is_background=False),
            timeout=SELF_PLAY_CYCLE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        pretty_log(
            "Self-Play Timeout",
            f"Cycle exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s wall-clock budget. Aborting.",
            level="WARNING", icon=Icons.STOP,
        )
        from .outcome import ToolOutcome
        # An abort is a failure, and `SYSTEM: SELF PLAY ABORTED` matches no
        # failure or rejection predicate in the tree.
        return ToolOutcome.failed(
            f"SYSTEM: SELF PLAY ABORTED — exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s cycle budget. "
            "A generation-loop or stuck upstream request burned the budget. "
            "Retry or investigate the upstream model's decoder state.",
            world_changed=False, reason_code="selfplay_cycle_timeout")
    from .outcome import append_note
    # NOT an f-string: `ToolOutcome` is a `str` subclass, so interpolating it
    # returns a plain `str` and every status `dream.py` declares — six
    # failure sites migrated in the previous round — died right here.
    return append_note(result, "\n\nSYSTEM: SELF PLAY DONE.")


# ---------------------------------------------------------------------------
# Continuous self-play loop
# ---------------------------------------------------------------------------
#
# A "loop" is a background asyncio.Task that runs `synthetic_self_play`
# cycles back-to-back until one of:
#   * the user sends a new message (handle_chat sets `stop_event` before
#     entering the normal chat path),
#   * the LLM calls `stop_self_play`,
#   * `max_cycles` is reached,
#   * the task is cancelled (process shutdown).
#
# The task + stop event are stashed on the context; there is at most one
# loop active per context. The loop is NOT persisted across restarts —
# per user request.

# Cool-off floor/ceiling for the inter-cycle adaptive wait. The
# FrontierTracker's adaptive_cooldown returns values tuned for the
# biological watchdog (minutes-to-hours). For an explicitly-requested
# continuous loop we want snappier cycling — the user is watching —
# so we clamp to a tighter window.
_LOOP_COOLOFF_FLOOR_S = 5
_LOOP_COOLOFF_CEILING_S = 180
_LOOP_COOLOFF_BASE_S = 30


def _derive_loop_cooloff(context) -> float:
    """Adaptive inter-cycle wait, bounded to [floor, ceiling] seconds.

    Falls back to the base wait if the tracker is missing / errors.
    """
    tracker = getattr(context, "frontier_tracker", None)
    if tracker is None:
        return float(_LOOP_COOLOFF_BASE_S)
    try:
        raw = tracker.adaptive_cooldown(
            base=_LOOP_COOLOFF_BASE_S,
            floor=_LOOP_COOLOFF_FLOOR_S,
            ceiling=_LOOP_COOLOFF_CEILING_S,
        )
        return float(max(_LOOP_COOLOFF_FLOOR_S, min(_LOOP_COOLOFF_CEILING_S, raw)))
    except Exception:
        return float(_LOOP_COOLOFF_BASE_S)


async def _consolidate_between_cycles(context):
    """Drain the short-term journal between self-play cycles so memories
    don't pile up during long loops.

    The biological watchdog runs the same drain on its 60s tick, but
    there's no ordering guarantee between the tick and our cycle boundary
    — in practice a long-running loop ends up with dozens of buffered
    items waiting on hippocampus. Doing an explicit drain here gives us
    a predictable "consolidate, then start the next cycle" cadence.

    Calls `process_journal_queue(respect_idle=False)`: the journal's
    `idle_secs < 30` guard exists to stop the watchdog from drowning a
    LIVE user, but the dispatching `handle_chat` call leaves a fresh
    `last_activity_time` heartbeat behind that fakes "user returned"
    inside the first inter-cycle drain — even though no actual user
    message arrived. Real user interrupts already reach the loop via
    `selfplay_loop_stop` (set in `handle_chat`); the idle gate here is
    redundant and just lies to the log. On any error we just log —
    consolidation failure must never kill the loop.
    """
    journal = getattr(context, "journal", None)
    agent = getattr(context, "agent", None)
    if journal is None or agent is None:
        return
    try:
        # Cheap check first — avoid the per-item log noise when the
        # journal is empty. pending_count() includes the overflow spill, so
        # a burst that overflowed the hot buffer isn't mistaken for "empty".
        items_on_disk = journal.pending_count()
    except Exception:
        items_on_disk = 0
    if items_on_disk <= 0:
        return
    try:
        pretty_log(
            "Self-Play Loop",
            f"Consolidating {items_on_disk} buffered memorie(s) before next cycle.",
            icon=Icons.BRAIN_THINK,
        )
        await agent.process_journal_queue(respect_idle=False)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        pretty_log(
            "Self-Play Loop",
            f"Inter-cycle consolidation failed (non-fatal): {e}",
            level="WARNING", icon=Icons.WARN,
        )


async def _run_self_play_loop(context, *, model_name: str, max_cycles: int, stop_event: asyncio.Event):
    """Body of the continuous self-play loop. Runs until `stop_event` is
    set, `max_cycles` is reached, or the outer task is cancelled.

    Every ``PRM_TRAIN_EVERY_N_CYCLES`` cycles the loop also kicks off a
    PRM retrain on the collected trajectories so the frontier-weighted
    pick path actually engages (proposal E, 2026-05-17). Pre-2026-05
    PRM training was only triggered from the biological watchdog's
    15-60 min idle window — but a busy self-play loop never reaches
    that window, so PRM.has_model stayed False and the uncertainty-
    weighted seed picker silently fell back to the brittle pool.
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    dreamer.cycle_budget_s = SELF_PLAY_CYCLE_TIMEOUT_S   # §4KS: each cycle is bounded below
    cycles_done = 0
    attempts = 0            # every TRY counts toward max_cycles (§4LX)
    fails_in_a_row = 0
    lessons_before = _count_playbook(context)
    # PRM retrain cadence inside the loop. 20 is enough fresh
    # trajectories that the model picks up new signal but not so often
    # that training itself dominates the cycle wall-clock.
    PRM_TRAIN_EVERY_N_CYCLES = 20
    pretty_log(
        "Self-Play Loop",
        f"Starting continuous loop (model={model_name}, max_cycles={max_cycles or 'unbounded'}).",
        icon=Icons.BRAIN_THINK,
    )
    try:
        while not stop_event.is_set():
            # ATTEMPTS, not successes (§4LX): failing cycles (600 s each, or
            # a bad `model=`) were never counted, so max_cycles=2 ran 82
            # attempts in a second and kept going
            if max_cycles and attempts >= max_cycles:
                pretty_log("Self-Play Loop", f"Reached max_cycles={max_cycles}. Stopping.", icon=Icons.OK)
                break
            if fails_in_a_row >= 3:
                pretty_log("Self-Play Loop", "3 cycles failed in a row — stopping.",
                           level="WARNING", icon=Icons.STOP)
                break
            # Don't interrupt a live user turn.
            llm_client = getattr(context, "llm_client", None)
            if llm_client is not None and getattr(llm_client, "foreground_tasks", 0) > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    continue

            attempts += 1
            try:
                await asyncio.wait_for(
                    dreamer.synthetic_self_play(model_name=model_name, is_background=True),
                    timeout=SELF_PLAY_CYCLE_TIMEOUT_S,
                )
                cycles_done += 1
                fails_in_a_row = 0
            except asyncio.TimeoutError:
                fails_in_a_row += 1
                pretty_log(
                    "Self-Play Loop",
                    f"Cycle {cycles_done+1} exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s. Skipping.",
                    level="WARNING", icon=Icons.STOP,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One cycle failing should not kill the loop — log and keep going.
                fails_in_a_row += 1
                pretty_log("Self-Play Loop", f"Cycle {cycles_done+1} raised: {e}", level="WARNING", icon=Icons.WARN)

            # Drain the short-term journal before cooling off. This keeps
            # the hippocampus backlog from growing unbounded during long
            # loops. The helper is a cheap no-op when the journal is
            # already empty, and it checks the stop_event so a user
            # message interrupts cleanly.
            if stop_event.is_set():
                break
            await _consolidate_between_cycles(context)

            # Proposal E: retrain the PRM every N cycles so the frontier-
            # weighted curriculum has fresh signal. Fire-and-forget
            # inside a thread — the trainer is pure-CPU so it won't
            # contend with the LLM client; if it fails we just keep
            # looping with the prior model (or no model).
            if cycles_done and cycles_done % PRM_TRAIN_EVERY_N_CYCLES == 0:
                try:
                    await asyncio.to_thread(_maybe_retrain_prm, context)
                except Exception as _pe:
                    pretty_log(
                        "Self-Play Loop",
                        f"PRM retrain skipped after cycle {cycles_done}: {_pe}",
                        level="WARNING", icon=Icons.WARN,
                    )
                # Router classifier retrain rides the same cadence as PRM.
                try:
                    await asyncio.to_thread(_maybe_retrain_router, context)
                except Exception as _re:
                    pretty_log(
                        "Self-Play Loop",
                        f"Router retrain skipped after cycle {cycles_done}: {_re}",
                        level="WARNING", icon=Icons.WARN,
                    )

            # Adaptive cool-off — responsive to curiosity delta, but
            # interruptible the instant a new user message arrives.
            cooloff = _derive_loop_cooloff(context)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=cooloff)
                break
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        pretty_log("Self-Play Loop", f"Cancelled after {cycles_done} cycle(s).", icon=Icons.STOP)
        raise
    finally:
        lessons_after = _count_playbook(context)
        delta = max(0, lessons_after - lessons_before)
        pretty_log(
            "Self-Play Loop",
            f"Loop finished. Cycles: {cycles_done}. New lessons (net): {delta}.",
            icon=Icons.OK,
        )
        # Null out the registered slot so a follow-up "run self play loop"
        # can start a fresh one. We're running INSIDE the task so
        # `task.done()` is still False at this point — instead, check
        # identity against the currently-running task.
        try:
            current = asyncio.current_task()
            registered = getattr(context, "selfplay_loop_task", None)
            if registered is current:
                context.selfplay_loop_task = None
                context.selfplay_loop_stop = None
                context.selfplay_loop_started_at = None
        except Exception:
            pass


def _count_playbook(context) -> int:
    sm = getattr(context, "skill_memory", None)
    if sm is None:
        return 0
    try:
        return len(sm._load_playbook())
    except Exception:
        return 0


def _maybe_retrain_prm(context) -> None:
    """In-loop PRM retrain (proposal E, 2026-05-17).

    Runs the trainer on the trajectory collector, hot-swaps the model
    into the live ``PRMScorer`` on success, and logs the report. Pure
    CPU; safe to call from a worker thread.

    Skips silently when the trajectory collector or PRM scorer aren't
    wired (e.g. test harnesses that monkey-patch the context with a
    MagicMock).

    ⚠ CONSUMER GATE. This is the twin of the biological-tick PRM phase, and
    it was missing that phase's `_prm_consumer_live` check — so a
    model-invocable `self_play_loop` could retrain every 20 cycles, OVERWRITE
    the pinned checkpoint, hot-swap the live scorer, and log
    "In-loop value-model refit" at INFO, all while no consumer reads PRM
    scores at all (`_MCTS_TURNSTART_ENABLED` is False and
    `--frontier-selfplay` is off). That log line reads as learning progress
    for work that changes nothing — exactly what the 2026-07-27 fix removed
    from the other path.
    """
    # R3 MAJOR-1: this predicate was duplicated here and in core/agent.py
    # phase 2.7, and BOTH read only `_MCTS_TURNSTART_ENABLED` — one
    # conjunct of a two-conjunct gate. Now one shared function, so the
    # twin cannot drift again.
    from ..core.agent import prm_consumer_is_live
    _consumer_live = prm_consumer_is_live(context)
    if not _consumer_live:
        from ..core.agent import prm_consumer_why_no_reader
        logger.debug("PRM retrain skipped — both value-reading consumers are off ("
                     + prm_consumer_why_no_reader(context) + "). --prm-online-update is "
                     "deliberately NOT read here: it is a PRODUCER that refines an "
                     "existing model and refuses to bootstrap one, so counting it "
                     "would train a model nothing reads (§4BN corrected §4BM)")
        return
    from ..distill.collector import TrajectoryCollector
    from ..prm.scorer import PRMScorer
    from ..prm.trainer import PRMTrainer
    from pathlib import Path

    traj_collector = getattr(context, "trajectory_collector", None)
    prm_scorer = getattr(context, "prm_scorer", None)
    if not isinstance(traj_collector, TrajectoryCollector):
        return
    if not isinstance(prm_scorer, PRMScorer):
        return

    save_path = getattr(context, "_prm_checkpoint_path", None)
    if save_path is None:
        mem_dir = getattr(context, "memory_dir", None)
        if mem_dir is not None:
            save_path = Path(mem_dir).parent / "prm" / "checkpoint.json"

    trainer = PRMTrainer()
    from ..core.admissibility import iter_bench_trajectories
    report = trainer.run(
        trajectories=_teachable(traj_collector.iter_trajectories()),
        save_path=save_path,
        bench_trajectories=iter_bench_trajectories(
            "prm", getattr(context, "args", None)),
    )
    if report.fit_succeeded and trainer.model is not None:
        prm_scorer.set_model(trainer.model)
        # R7 MIN-6 (twin divergence): phase 2.7 bridges a freshly-fitted
        # model into `mcts.prm_scorer` on the first-ever fit; this twin
        # did not. On a `.score()`-live box that boots without a
        # checkpoint, `mcts.prm_scorer` stays None and MCTS keeps failing
        # its own `prm_scorer is not None and .has_model` guard after an
        # in-loop refit — i.e. this path trained a model nothing reads,
        # which is the §4BN class on the twin.
        _mcts = getattr(context, "mcts_reasoner", None)
        if _mcts is not None and getattr(_mcts, "prm_scorer", None) is None:
            _mcts.prm_scorer = prm_scorer
        pretty_log(
            "Self-Play PRM Retrain",
            f"In-loop value-model refit: {report.summary()}",
            icon=Icons.BRAIN_PLAN,
        )
    else:
        pretty_log(
            "Self-Play PRM Retrain",
            f"Skipped: {report.bail_reason or 'unknown'}",
            level="DEBUG", icon=Icons.BRAIN_PLAN,
        )


def _maybe_retrain_router(context) -> None:
    """In-loop router-classifier retrain (mirrors _maybe_retrain_prm).

    Trains the ComplexityClassifier on the trajectory log and hot-swaps it
    into the live dispatcher on success, so the router stops escalating every
    request. Pure CPU; safe from a worker thread. Skips silently when the
    collector or dispatcher aren't wired.
    """
    from ..distill.collector import TrajectoryCollector
    from ..router import ComplexityDispatcher, RouterTrainer
    from pathlib import Path

    traj_collector = getattr(context, "trajectory_collector", None)
    dispatcher = getattr(context, "complexity_dispatcher", None)
    if not isinstance(traj_collector, TrajectoryCollector):
        return
    if not isinstance(dispatcher, ComplexityDispatcher):
        return

    save_path = getattr(context, "_router_checkpoint_path", None)
    if save_path is None:
        mem_dir = getattr(context, "memory_dir", None)
        if mem_dir is not None:
            save_path = Path(mem_dir).parent / "router" / "checkpoint.json"

    # §4AA: score the gate at the LIVE dispatcher threshold (R1 review —
    # this path scored the probe default while the idle/boot paths scored
    # the shipping operating point).
    trainer = RouterTrainer(
        confidence_threshold=getattr(dispatcher, "confidence_threshold", None))
    from ..core.admissibility import iter_bench_trajectories
    report = trainer.run(
        trajectories=_teachable(traj_collector.iter_trajectories()),
        save_path=save_path,
        bench_trajectories=iter_bench_trajectories(
            "router", getattr(context, "args", None)),
    )
    if report.fit_succeeded and trainer.classifier is not None:
        dispatcher.classifier = trainer.classifier
        dispatcher.disabled = False
        pretty_log(
            "Self-Play Router Retrain",
            f"In-loop classifier refit: {report.summary()} · router now routing",
            icon=Icons.BRAIN_PLAN,
        )
    else:
        pretty_log(
            "Self-Play Router Retrain",
            f"Skipped: {report.bail_reason or 'unknown'}",
            level="DEBUG", icon=Icons.BRAIN_PLAN,
        )


async def tool_self_play_loop(context, max_cycles: int = 0, model: str = "", **kwargs):
    """Start a background continuous self-play loop. Idempotent: if one is
    already running, returns a status line instead of launching a second.
    """
    if not _user_asked_for_self_play(context):
        pretty_log(
            "Self-Play Loop Refused",
            "LLM invoked `self_play_loop` but the user's current turn doesn't ask for it. "
            "Refusing and redirecting the model back to the original task.",
            level="WARNING", icon=Icons.STOP,
        )
        return _SELF_PLAY_INTENT_REFUSAL
    existing = getattr(context, "selfplay_loop_task", None)
    if existing is not None and not existing.done():
        return (
            "SYSTEM: A self-play loop is already running. "
            "Call `stop_self_play` first if you want to restart it."
        )

    try:
        max_cycles_int = max(0, int(max_cycles or 0))
    except Exception:
        max_cycles_int = 0

    model_name = (model or "").strip()
    if not model_name:
        model_name = getattr(getattr(context, "args", None), "model", "default") or "default"

    stop_event = asyncio.Event()
    loop_task = asyncio.create_task(
        _run_self_play_loop(
            context,
            model_name=model_name,
            max_cycles=max_cycles_int,
            stop_event=stop_event,
        ),
        name="selfplay_loop",
    )
    # Stash on context so handle_chat / stop_self_play can find it.
    context.selfplay_loop_task = loop_task
    context.selfplay_loop_stop = stop_event
    try:
        import datetime as _dt
        context.selfplay_loop_started_at = _dt.datetime.now()
    except Exception:
        context.selfplay_loop_started_at = None

    pretty_log(
        "Self-Play Loop",
        f"Dispatched (model={model_name}, max_cycles={max_cycles_int or 'unbounded'}).",
        icon=Icons.OK,
    )
    max_desc = f"up to {max_cycles_int} cycle(s)" if max_cycles_int else "unbounded"
    return (
        f"SYSTEM: CONTINUOUS SELF-PLAY LOOP STARTED ({max_desc}, model={model_name}).\n"
        "It will keep running back-to-back cycles in the background. "
        "Send any message — or call `stop_self_play` — to stop it."
    )


async def tool_stop_self_play(context):
    """Signal the running self-play loop to stop after its current cycle."""
    task = getattr(context, "selfplay_loop_task", None)
    stop_event = getattr(context, "selfplay_loop_stop", None)
    if task is None or task.done():
        return "SYSTEM: No self-play loop is currently running."
    if stop_event is not None:
        stop_event.set()
    # Give it a short grace period to unwind cleanly; if it's mid-cycle
    # the wait will time out and the caller just gets the "signalled"
    # acknowledgement — the loop will stop on its own at the next check.
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        return "SYSTEM: Self-play loop stopped."
    except asyncio.TimeoutError:
        return "SYSTEM: Stop signalled — loop will exit after the current cycle."
    except Exception as e:
        return f"SYSTEM: Self-play loop stopped (with error: {e})."


_VALID_LESSON_SCOPES = {"today", "week", "all", "self_play_only"}


async def tool_list_lessons(context, scope: str = "today", limit: int = 20, **kwargs):
    """Surface the lessons currently in the skill playbook for the user.

    `scope`:
      - "today"           — lessons with `timestamp >= local midnight`
      - "week"            — lessons from the last 7 days (local)
      - "all"             — every lesson in the playbook
      - "self_play_only"  — every lesson with `source == "self_play"`,
                            no time filter.
    """
    skill_memory = getattr(context, "skill_memory", None)
    if skill_memory is None:
        return "SYSTEM: Skill memory is not available in this context."

    scope_norm = (scope or "today").strip().lower()
    if scope_norm not in _VALID_LESSON_SCOPES:
        return (
            f"SYSTEM: Unknown scope '{scope}'. "
            f"Allowed: {sorted(_VALID_LESSON_SCOPES)}."
        )
    try:
        limit_int = max(1, min(100, int(limit)))
    except Exception:
        limit_int = 20

    if scope_norm == "self_play_only":
        _q = dict(scope="all", source="self_play")
        header_scope = "self-play lessons"
    else:
        _q = dict(scope=scope_norm, source="")
        header_scope = {
            "today": "lessons learned today",
            "week":  "lessons learned in the last 7 days",
            "all":   "all lessons learned so far",
        }[scope_norm]
    lessons = skill_memory.list_lessons(limit=limit_int, **_q)

    if not lessons:
        return f"No {header_scope} yet."

    # R2-6 (2026-09-20): the header printed the PAGE length as the count —
    # "## 100 all lessons learned so far:" over a 198-lesson store, and the
    # agent answered "how many lessons do you have?" with 100. A truncated
    # listing must say so and say of what; `count_lessons` is the total
    # behind the same filter. (`getattr`: a stub memory may predate it.)
    _count_fn = getattr(skill_memory, "count_lessons", None)
    try:
        total = int(_count_fn(**_q)) if callable(_count_fn) else len(lessons)
    except Exception:  # noqa: BLE001 — a header must never fail the tool
        total = len(lessons)
    if total > len(lessons):
        lines = [f"## {total} {header_scope} — showing the {len(lessons)} most "
                 f"recent (limit={limit_int}, max 100; the total is {total}):"]
    else:
        lines = [f"## {len(lessons)} {header_scope}:"]
    for i, lesson in enumerate(lessons, 1):
        ts = lesson.get("timestamp") or ""
        when = ""
        try:
            from datetime import datetime as _dt
            when = _dt.fromisoformat(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
        except Exception:
            when = ts[:16] if ts else ""
        verified = "✓" if lesson.get("verified") else "·"
        source = lesson.get("source") or "?"
        trigger = (lesson.get("trigger") or lesson.get("task") or "").strip() or "(no trigger)"
        domains = ", ".join(lesson.get("domains") or []) or "-"
        retrievals = int(lesson.get("retrievals") or 0)
        helpful = int(lesson.get("helpful_retrievals") or 0)
        fix = (lesson.get("correct_pattern") or lesson.get("solution") or "").strip()
        # Keep each entry short — the agent can paraphrase the full detail
        # back to the user if asked. One line of meta + one of fix snippet.
        fix_preview = fix.replace("\n", " ⏎ ")
        if len(fix_preview) > 180:
            fix_preview = fix_preview[:177] + "..."
        lines.append(
            f"{i}. [{when}] ({verified} src={source}) {trigger}\n"
            f"   domains: {domains} | retrievals: {retrievals} (helpful: {helpful})\n"
            f"   fix: {fix_preview}"
        )
    return "\n".join(lines)
