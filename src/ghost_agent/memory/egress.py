"""§4LB r2 — the owner's street address never leaves the machine.

One detector for every outbound path (the dispatch hook, the tool wrappers a
macro or a delegate calls, the weather lookup). The first cut matched the
profile's exact Latin substrings: the review found the postcode alone, the
Greek spelling ("Μακεδονίας 83"), a \\u-escaped JSON string, "Makedonias
street 83", a double space and a URL-encoded maps link all went out, and a
replace without word boundaries turned "Makedonias 830" into "Athens0".

The detector works on FOLDED text (lower case, accents stripped, Greek
transliterated) with an index map back to the original, so a match is
replaced in the text exactly as written:
  * street + house number, either order, an optional "street/st/odos/οδός"
    between, the street matched by its stem (Greek inflection: Μακεδονίας /
    Μακεδονία), the number bounded (83 ≠ 830, 83A is still 83);
  * the postcode, with or without the Greek "136 76" space.
The suburb and the city stay: they are what a local search needs.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote, unquote_plus

#: Cyrillic look-alikes of Latin letters (§4LH review: "Mаkedoniаs" with a
#: Cyrillic а went out)
_CONFUSABLE = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
               "ј": "j", "ѕ": "s", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h"}
_GR2LAT = {"α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i",
           "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
           "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o"}
_STREET_WORDS = r"(?:street|str|st|odos|odou|ave|avenue|leoforos|leof|road|rd)\.?"
#: what may sit between a street and its number besides separators (§4LH review:
#: "Makedonias Nr. 83", "#83", "αριθμός 83", "number 83" went out)
_NUMBER_WORDS = r"(?:no|nr|number|numero|arithmos|ar|#)\.?"
_STREET_NUMBER = re.compile(r"([^\W\d_][\w'.-]+(?:\s+[^\W\d_][\w'.-]+)?)\s+(\d{1,4})[a-zA-Z]?\b")
_POSTCODE = re.compile(r"\b(\d{3})\s?(\d{2})\b")


def fold(text: str):
    """(folded, index) — ``folded[i]`` came from ``text[index[i]]``."""
    out, idx = [], []
    for i, ch in enumerate(text):
        # §4LH review: full-width digits, zero-width joiners and "c" for "k"
        # ("Macedonias") all carried the address past the detector
        if unicodedata.category(ch) == "Cf":
            continue
        ch = unicodedata.normalize("NFKC", ch)
        base = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c)).lower()
        for c in base:
            t = _GR2LAT.get(_CONFUSABLE.get(c, c), _CONFUSABLE.get(c, c))
            if t == "c":
                t = "k"
            out.append(t)
            idx.extend([i] * len(t))
    return "".join(out), idx


def address_patterns(values) -> list:
    """Compiled patterns (over folded text) for every street+number and
    postcode found in ``values`` (the on-demand address-like profile values)."""
    pats = set()
    for v in values or ():
        s = str(v or "")
        first = s.split(",")[0]
        for m in _STREET_NUMBER.finditer(first):
            street_words = fold(m.group(1))[0].split()
            # the street is the word(s) before the number; a two-word match
            # whose first word is the street ("makedonias 83 thrakomakedones")
            # is handled by trying the last word too
            for street in {street_words[-1], " ".join(street_words)}:
                stem = _stem_re(street)
                num = r"\s?".join(m.group(2))          # "8 3" too
                sep = rf"[\s,.\-/_#]+(?:{_STREET_WORDS}[\s,.\-/_#]+)?(?:{_NUMBER_WORDS}[\s,.\-/_#]*)?"
                pats.add(rf"(?<![\w]){stem}\w*{sep}{num}(?:[a-z]\b|\b)")
                pats.add(rf"(?<![\w\d]){num}[a-z]?{sep}{stem}\w*")
        for m in _POSTCODE.finditer(s):
            # never inside an identifier (§4LH review: arxiv 2401.13676,
            # issues/13676, v1.13676.tar.gz were rewritten); "136-76" and
            # "GR-13676" are still the postcode. A bare "PR 13676" is scrubbed
            # too — the §4LB review pinned the postcode alone as a leak.
            pats.add(rf"(?<![\d./#]){m.group(1)}[\s-]?{m.group(2)}(?![\d]|\.\d)")
    return [re.compile(p) for p in sorted(pats, key=len, reverse=True)]


def _stem_re(street: str) -> str:
    """The street's stem over folded text: inflection-tolerant (Μακεδονίας /
    Μακεδονία), "dh" for "d" tolerated (Makedhonias)."""
    stem = street[:max(5, len(street) - 3)] if len(street) >= 5 else street
    out = re.escape(stem).replace(r"\ ", r"\s+")
    return out.replace("d", "dh?")


def address_parts(values) -> list:
    """(street_stem_regex, house_number) pairs of the address values — for the
    call-level rule: a street and its number in ONE call, however far apart
    ("house 83 on Makedonias", street and number in separate arguments)."""
    parts = set()
    for v in values or ():
        first = str(v or "").split(",")[0]
        for m in _STREET_NUMBER.finditer(first):
            street = fold(m.group(1))[0].split()[0]
            if len(street) >= 4:
                parts.add((_stem_re(street), m.group(2)))
    return sorted(parts)


def looks_like_street_address(value) -> bool:
    """A value that is a street address whatever key holds it: a street word,
    a number word or a 5-digit postcode, AND a word followed by a number."""
    s = fold(str(value or ""))[0]
    if not re.search(r"[^\W\d_]{3,}\.?\s+\d{1,4}[a-z]?\b|\b\d{1,4}[a-z]?\s+[^\W\d_]{4,}", s):
        return False
    return bool(re.search(rf"\b{_STREET_WORDS}(?=[\s,]|$)|\b(?:odos|leoforos|plateia)\b|(?<!\d)\d{{3}}\s?\d{{2}}(?!\d)", s))


def locality(value):
    """``value`` without its street + house number and postcode — the part a
    geocoder or a local search needs ("Makedonias 83 Thrakomakedones 13676,
    Athens, Greece" → "Thrakomakedones, Athens, Greece"); None when nothing
    is left."""
    v = str(value or "").strip()
    pats = address_patterns([v])
    if not pats:
        return v or None
    kept = []
    for part in (p.strip() for p in v.split(",")):
        got = scrub_text(part, pats, "\x00")
        if got is None:
            kept.append(part)
            continue
        rest = " ".join(w for w in got.replace("\x00", " ").split())
        if rest:
            kept.append(rest)
    return ", ".join(kept) or None


def scrub_text(text: str, patterns, replacement: str):
    """``text`` with every match replaced, or None when nothing matched.
    A URL-encoded string is decoded, scrubbed and re-encoded."""
    if not isinstance(text, str) or not text or not patterns:
        return None
    out = _scrub_plain(text, patterns, replacement)
    if out is not None:
        return out
    if "%" in text or "+" in text:
        dec = unquote_plus(text)
        if dec != text:
            got = _scrub_plain(dec, patterns, replacement)
            if got is not None:
                return quote(got, safe=":/?&=#,;@!$'()*+~%")
    return None


def _scrub_plain(text: str, patterns, replacement: str):
    folded, idx = fold(text)
    spans = []
    for p in patterns:
        for m in p.finditer(folded):
            if m.end() > m.start():
                spans.append((idx[m.start()], idx[m.end() - 1] + 1))
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, last = [], 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(replacement)
        last = e
    out.append(text[last:])
    return "".join(out)


def scrub_value(value, patterns, replacement: str, keys=None, urls_only: bool = False):
    """(new_value, changed) for a tool-argument value: dicts and lists
    recursively; ``keys`` limits which dict keys are scrubbed (None = all);
    ``urls_only`` scrubs only strings that are URLs."""
    if isinstance(value, dict):
        changed, out = False, {}
        for k, v in value.items():
            if keys is not None and k not in keys and not (isinstance(v, str) and _is_url(v)):
                out[k] = v
                continue
            nv, c = scrub_value(v, patterns, replacement, None, urls_only)
            out[k], changed = nv, changed or c
        return out, changed
    if isinstance(value, list):
        res = [scrub_value(v, patterns, replacement, keys, urls_only) for v in value]
        return [r[0] for r in res], any(r[1] for r in res)
    if isinstance(value, str):
        if urls_only and not _is_url(value):
            return value, False
        got = scrub_text(value, patterns, replacement)
        return (got, True) if got is not None else (value, False)
    return value, False


def _is_url(s: str) -> bool:
    return bool(re.match(r"\s*(?:https?://|www\.)", s, re.IGNORECASE))


#: tools whose arguments leave the machine, and WHICH arguments: None = every
#: argument; a set = those keys plus any URL anywhere; "urls" = URLs only
#: (file_system / knowledge_base / vision_analysis also take local paths and
#: owner text that never leaves — a download URL or a transcribe link does).
#: `execute` ("url_literals"): its code is the owner's own and may hold the
#: address (a letter it writes), but the sandbox HAS Tor egress — a URL inside
#: the code (curl 'https://…/search?q=Makedonias+83') is scrubbed (§4LH review).
OUTBOUND_TOOLS = {
    "web_search": None, "deep_research": None, "darkweb_search": None, "darkweb_research": None,
    "fact_check": None, "browser": None,
    "system_utility": {"location", "city", "query", "place", "address", "area", "where"},
    "file_system": "urls", "knowledge_base": "urls", "vision_analysis": "urls",
    "execute": "url_literals",
}

_URL_IN_TEXT = re.compile(r"(?:https?://|www\.)[^\s'\"<>`]+", re.IGNORECASE)


def _scrub_url_literals(value, patterns, replacement):
    """(value, changed) with the address scrubbed inside every URL found in
    any string of ``value`` — the rest of the text is left as written."""
    if isinstance(value, dict):
        res = {k: _scrub_url_literals(v, patterns, replacement) for k, v in value.items()}
        return {k: r[0] for k, r in res.items()}, any(r[1] for r in res.values())
    if isinstance(value, list):
        res = [_scrub_url_literals(v, patterns, replacement) for v in value]
        return [r[0] for r in res], any(r[1] for r in res)
    if not isinstance(value, str):
        return value, False
    changed = False

    def _one(m):
        nonlocal changed
        got = scrub_text(m.group(0), patterns, replacement)
        if got is None:
            return m.group(0)
        changed = True
        return got
    out = _URL_IN_TEXT.sub(_one, value)
    return out, changed


def _strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def _scrub_split_address(value, parts, replacement, policy):
    """The call-level rule: when one call carries a street stem AND its house
    number anywhere (any argument, any order, words between), both are
    replaced everywhere in it. Not for "url_literals" (the code is the
    owner's) — only for the query tools."""
    if not parts or policy == "url_literals":
        return value, False
    keys = policy if isinstance(policy, set) else None
    scoped = ({k: v for k, v in value.items() if k in keys} if keys is not None and isinstance(value, dict)
              else value)
    folded = " ".join(fold(t)[0] for t in _strings(scoped))
    pats = []
    for stem, num in parts:
        if re.search(rf"(?<![\w]){stem}\w*", folded) and re.search(rf"(?<![\w\d]){num}(?![\d])", folded):
            pats += [re.compile(rf"(?<![\w]){stem}\w*"), re.compile(rf"(?<![\w\d.]){num}[a-z]?(?![\d\w])")]
    if not pats:
        return value, False
    return scrub_value(value, pats, replacement, keys, policy == "urls")


def egress_profile(context):
    """The OWNER's profile for scrubbing — a delegate has no profile in its
    prompt (`profile_memory=None`) but still needs its queries scrubbed."""
    return getattr(context, "egress_profile", None) or getattr(context, "profile_memory", None)


def scrub_tool_args(tool_name: str, args, context):
    """(args, changed) with the owner's street address replaced in the
    arguments of an outbound tool; a JSON-string ``args`` is parsed first
    (a \\u-escaped Greek street is matched) and returned as a string."""
    if tool_name not in OUTBOUND_TOOLS or not args:
        return args, False
    pm = egress_profile(context)
    if pm is None or not hasattr(pm, "egress_scrubber"):
        return args, False
    try:
        patterns, city = pm.egress_scrubber()
    except Exception:  # noqa: BLE001 — an unreadable profile scrubs nothing
        return args, False
    if not patterns:
        return args, False
    import json
    as_str = isinstance(args, str)
    val = args
    if as_str:
        try:
            val = json.loads(args)
        except Exception:  # noqa: BLE001 — not JSON: scrub the raw text
            got = scrub_text(args, patterns, city)
            return (got, True) if got is not None else (args, False)
    policy = OUTBOUND_TOOLS[tool_name]
    if policy == "url_literals":
        new, changed = _scrub_url_literals(val, patterns, city)
    else:
        new, changed = scrub_value(val, patterns, city,
                                   keys=policy if isinstance(policy, set) else None,
                                   urls_only=policy == "urls")
    try:
        parts = address_parts(pm.address_values()) if hasattr(pm, "address_values") else []
    except Exception:  # noqa: BLE001
        parts = []
    new2, changed2 = _scrub_split_address(new, parts, city, policy)
    new, changed = new2, changed or changed2
    if not changed:
        return args, False
    return (json.dumps(new, ensure_ascii=False) if as_str else new), True


def privacy_note(area: str) -> str:
    """What the model is told when its call was rewritten (§4LB r4: it did
    not know, re-guessed the town and searched a different Makedonias Ave)."""
    return (f"[privacy: the owner's street address in this call was replaced by its area, '{area}', before it "
            f"left the machine. These results are for {area}; search there — never put the street, number or "
            f"postcode in a query.]")


def with_privacy_note(result, area: str):
    """``result`` with :func:`privacy_note` APPENDED — never in front: the
    result's head is parsed (an "Error:" prefix classifies a failed call).
    Awaited first when awaitable (returned as a coroutine then); a non-text
    result is left as it is."""
    import inspect
    if inspect.isawaitable(result):
        async def _noted():
            return with_privacy_note(await result, area)
        return _noted()
    if not isinstance(result, str):
        return result
    text = f"{result}\n\n{privacy_note(area)}"
    # §4LO: keep a DECLARED outcome's status — an f-string made a refusal or
    # a declared browser failure a plain string, which coerced to SUCCESS
    _st = getattr(result, "status", None)
    if _st is not None:
        from ..tools.outcome import ToolOutcome
        return ToolOutcome(text, status=_st, world_changed=getattr(result, "world_changed", None),
                           reason_code=getattr(result, "reason_code", None),
                           declared=getattr(result, "declared", True),
                           call_args=getattr(result, "call_args", None))
    return text
