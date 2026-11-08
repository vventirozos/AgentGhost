"""Reference photos for `image_generation(subjects=[...])` (§4MG).

The image model draws a likeness from PIXELS, never from a name
([[likeness-comes-from-pixels]] §4JX). Live 2026-10-07: asked for two named
politicians, the agent's own thinking said "let me first find/download
photos of both" and its next action was a text-only render — two strangers.
On the follow-up it guessed Wikimedia URLs (400 ×2), screenshotted the
article pages, and finally passed ONE portrait because the node takes one
reference — the second man was invented again.

So acquisition is the tool's job, not the model's: the model names the
specific real people / places / products in `subjects`, and this module

  1. finds each one's lead image on Wikipedia (one API call over Tor —
     the article's own `pageimages` thumbnail, never a guessed URL),
  2. downloads it into the sandbox's ``subject_photos/`` folder as
     ``ref_<slug>.<ext>`` with a provenance record beside it (reused only
     when that record names the same subject and the file decodes), and
  3. joins several photos side by side into ONE image padded to the render's
     shape — the node accepts exactly one reference (§4JW: a second OOMs) and
     stretches it to the output size without keeping its aspect
     (`fit_reference`), so an unpadded strip of three portraits squashed
     every face.

A subject with no findable photo is an ERROR naming it: drawing a stranger
under a real name is the failure this exists to stop. A search that could not
REACH the encyclopedia says so separately — it is worth one retry, and must
not send the user off to find a photo of someone famous.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils.logging import Icons, pretty_log

#: Photos joined into one reference. Beyond three the faces get too small
#: in the node's ~0.4 MP reference latents to carry a likeness.
MAX_SUBJECTS = 3
#: Height every photo is scaled to before joining.
COMBINED_HEIGHT = 1024
#: White gap between joined photos, so the model sees separate pictures.
#: §4OZ (2026-10-14, measured on 10 ordinary two-person scenes, same seed per
#: pair): a SEAMLESS composite (head-and-shoulders crops touching, feathered
#: seam, blurred margins) came out as two panels 7/10 vs 5/10 for this strip —
#: an edit preserves what looks like a finished picture; the frame is what tells
#: the model these are separate photos to recompose. Do not rebuild it.
COMBINED_GAP = 16
#: The render's shape when the caller asked for none and several subjects
#: are joined — the node's own default (768x512).
DEFAULT_ASPECT = 1.5
#: Sandbox folder for the fetched photos: never a user's own file name.
PHOTO_DIR = "subject_photos"
LOOKUP_TIMEOUT_S = 20
LOOKUP_ATTEMPTS = 2
#: §4QG: ONE budget per subject across every query, language and attempt. A
#: Greek name asks el. then en., two queries each, 20 s each, twice — up to
#: ~165 s before any render. Out of budget = SearchUnavailable (worth a later
#: try), never "no photo".
#: §4QG-b: 45 → 60 s. At 45 the fresh-circuit retry (attempt 2) was left
#: ~22 s — about one query of the two-plus it needs; now attempt 2 starts
#: only with at least one full query timeout left (fetch_subject_photo).
LOOKUP_DEADLINE_S = 60
#: §4QG: a cached photo older than this is fetched again (the article's lead
#: image changes — a new official portrait); the cache never expired before.
CACHE_MAX_AGE_S = 30 * 24 * 3600
#: User-Agent for the Wikipedia API. 2026-10-20 (request 52c9a5d2): a
#: Chrome-impersonating client is answered 429 from Tor exits on every
#: circuit, while a plain tool UA is answered 200 (no UA at all = 403).
#: The most common generic UA there is, so lookups stay unlinkable — never
#: a name of our own.
LOOKUP_USER_AGENT = "curl/8.7.1"
#: Thumbnail width asked of the API (Wikimedia serves its standard steps;
#: a free-form width 400s from upload.wikimedia.org — §4JX).
THUMB_WIDTH = 1024

_STOP = {"the", "a", "an", "of", "in", "at", "on", "and", "de", "la", "le",
         "ο", "η", "το", "οι", "του", "της"}
_GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")


class SearchUnavailable(LookupError):
    """The encyclopedia could not be reached — NOT "no photo exists".
    `rate_limited` = it answered 429: an immediate retry cannot help."""

    def __init__(self, msg: str = "", rate_limited: bool = False):
        super().__init__(msg)
        self.rate_limited = rate_limited


class PhotoUnavailable(LookupError):
    """The article EXISTS but its photo could not be fetched right now (the
    download failed, or what arrived is not a complete image) — NOT "no
    photo exists" either (§4PB: a 503 on the thumbnail used to tell the
    user to upload a photo of a famous person).

    §4QG: `kind` says whether a retry can help — "transient" (the download
    failed: call again once), "unsupported" (the article's photo is in a
    format the node cannot use, e.g. GIF: a retry fetches the same file, so
    the tool treats it as no usable photo and names the format) or "disk"
    (THIS machine could not save it: not the photo's fault, nothing to ask
    the user for). One class, so the tool's three arms stay the only ones."""

    def __init__(self, reason: str, page: str = "", title: str = "", kind: str = "transient"):
        super().__init__(reason)
        self.page, self.title, self.kind = page, title, kind


#: §4QG: the per-subject lookup budget's absolute end (time.monotonic()).
#: A ContextVar, not an argument: asyncio.to_thread copies the context into
#: the lookup's thread, so fetch_subject_photo's budget reaches
#: lookup_photo_url without changing its (subject, proxy) signature.
_LOOKUP_DEADLINE: contextvars.ContextVar = contextvars.ContextVar("subject_lookup_deadline", default=None)


#: §4PB: one token of a multi-word name may differ from the title's token
#: ("Zoi" ↔ "Zoe", a transliteration) — but only when another token of at
#: least this length matched exactly (the surname anchors it; "Obama" alone
#: must never match "Osama").
FUZZY_ANCHOR_LEN = 4
#: §4QG: the ONLY single-letter substitutions a transliteration makes. Any
#: one edit used to pass, so "Mark Johnson" matched "Mary Johnson", "Paul
#: Allen" "Paula Allen", "Jon Stewart" "John Stewart" — different people.
_TRANSLIT_SUBS = [frozenset("iey"), frozenset("ιηυ"), frozenset("οω")]
#: §4QG: a leading title is not part of the name ("Prime Minister Kyriakos
#: Mitsotakis" matched no article). Folded, without a trailing dot; "prime
#: minister" is matched as a pair. Kept small: a title that IS part of how
#: the article is named (Pope Francis, Queen Elizabeth II) still matches,
#: because the full name is always tried first.
_HONORIFICS = {"dr", "mr", "mrs", "ms", "miss", "prof", "professor", "president", "king", "queen",
               "pope", "saint", "st", "sir", "dame", "lord", "lady", "general", "gen", "senator",
               "chancellor", "minister", "rev", "reverend",
               "κ", "κα", "κος", "κυριος", "κυρια", "δρ", "προεδρος", "πρωθυπουργος", "καθηγητης",
               "καθηγητρια", "αγιος", "αγια"}
_HONORIFIC_PAIRS = {("prime", "minister")}


def _fold(text: str) -> str:
    """Lower-case, accents stripped — "Σαμαράς" and "σαμαρας" compare equal.
    §4QG: ß is "ss" ("Johann Strauß" is the article "Johann Strauss II")."""
    nfkd = unicodedata.normalize("NFKD", str(text or "").lower().replace("ß", "ss"))
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _tokens(text: str) -> List[str]:
    """Content words, folded. §4QG: a run of single-letter initials is ONE
    token, so "J. K. Rowling", "J.K. Rowling" and "JK Rowling" all read
    ["jk", "rowling"] (the initials used to be dropped as too short)."""
    out: List[str] = []
    run = ""
    for t in re.findall(r"\w+", _fold(text), flags=re.UNICODE):
        if len(t) == 1 and t.isalpha():
            run += t
            continue
        if len(run) >= 2:
            out.append(run)
        run = ""
        out.append(t)
    if len(run) >= 2:
        out.append(run)
    return [t for t in out if len(t) >= 2 and t not in _STOP]


def strip_honorifics(subject: str) -> str:
    """§4QG: the name without its leading titles ("Prime Minister Kyriakos
    Mitsotakis" → "Kyriakos Mitsotakis", "κ. Μητσοτάκης" → "Μητσοτάκης");
    at least one word is always kept."""
    words = str(subject or "").split()
    while len(words) > 1:
        a = _fold(words[0]).strip(".")
        if len(words) > 2 and (a, _fold(words[1]).strip(".")) in _HONORIFIC_PAIRS:
            words = words[2:]
        elif a in _HONORIFICS:
            words = words[1:]
        else:
            break
    return " ".join(words)


def _name_variants(subject: str) -> List[List[str]]:
    """The name's token lists to match: the full name first, then — when a
    title was stripped and at least TWO words remain — the bare name. One
    word left ("Pope Francis" → "Francis") would match any Francis."""
    full = _tokens(subject)
    core = _tokens(strip_honorifics(subject))
    return [full] + ([core] if len(core) >= 2 and core != full else [])


def _translit_equal(a: str, b: str) -> bool:
    """§4QG: ``a`` and ``b`` are one spelling of the same name across a
    transliteration: equal after ph→f and k→c ("Sophia"/"Sofia", "Nikos"/
    "Nicos"), or the WORD-FINAL vowel swapped inside its sound class (i/e/y:
    "Zoi"/"Zoe", "Alexei"/"Alexey"; Greek ι/η/υ, ο/ω). §4QG-b: only the final
    letter — a swap inside the word passed "Bill"/"Bell" and "Lynn"/"Linn"
    — and no added letter ("Joe"/"Joey" passed). An inserted or removed
    letter ("Jon"/"John", "Paul"/"Paula") or a consonant swap ("Mark"/
    "Mary") is a different name."""
    def sk(t):
        return t.replace("ph", "f").replace("k", "c")
    a, b = sk(a), sk(b)
    if a == b:
        return True
    if len(a) == len(b) and len(a) >= 2 and a[:-1] == b[:-1]:
        return any(a[-1] in c and b[-1] in c for c in _TRANSLIT_SUBS)
    return False


def _match_rank(subject: str, title: str) -> Optional[int]:
    """How the article is ABOUT the subject, best first: 1 = every word of
    the full name is in the title, 2 = every word of the name without its
    title (honorific), 3 = one word only matches as a transliteration;
    None = not about the subject. (0, an exact title, is title_is_exact.)"""
    have = set(_tokens(title))
    variants = [w for w in _name_variants(subject) if w]
    for rank, want in enumerate(variants, start=1):
        if all(t in have for t in want):
            return rank if rank == 1 else 2
    for want in variants:
        missing = [t for t in want if t not in have]
        if len(missing) != 1:
            continue
        anchored = any(t in have and len(t) >= FUZZY_ANCHOR_LEN for t in want if t != missing[0])
        if anchored and any(_translit_equal(missing[0], h) for h in have):
            return 3
    return None


def title_matches(subject: str, title: str) -> bool:
    """The article is ABOUT the subject: EVERY content word of the name is
    in the title ("Tsipras" → "Alexis Tsipras", "Toyota Corolla" → "Toyota
    Corolla (E210)"). The reverse direction (title words ⊆ name) was dropped
    in review: "Alexis Tsipras and Antonis Samaras" matched "Alexis Tsipras"
    and rendered ONE man's photo as both, "Michael Jordan" matched "Jordan".
    A search for "Antonis Samaras" also returns "New Democracy (Greece)"
    with the party flag as its image — that must not become his face.
    §4PB: ONE token may be a transliteration of the title's ("Zoi" ↔ "Zoe")
    when another token of FUZZY_ANCHOR_LEN+ letters matched exactly; a
    single-token name needs an exact token ("Obama" never matches "Osama
    bin Laden"). §4QG: only transliteration-like differences (see
    _translit_equal), and a leading title may be absent from the article."""
    return _match_rank(subject, title) is not None


def title_is_exact(subject: str, title: str) -> bool:
    """The title IS the name (folded, ignoring a parenthetical qualifier);
    §4QG: or the name without its leading title, when two+ words remain."""
    have = _tokens(re.sub(r"\s*\([^)]*\)\s*$", "", title))
    return any(want and have == want for want in _name_variants(subject))


#: §4QG-b: a redirect target that is an article about a GROUP or an EVENT,
#: not the one person ("Family of Barack Obama", "Wedding of Prince Harry
#: and Meghan Markle") — its lead image is several people or a scene
_GROUP_ARTICLE_RE = re.compile(r"^(?:family|wedding|funeral|death|assassination|murder|killing|"
                               r"disappearance|trial|list|house) of\b", re.IGNORECASE)


def _redirect_is_the_subject(name: str, target: str, fragment: str = "") -> bool:
    """A redirect FROM the exact typed name is the encyclopedia's own say-so
    (§4PB) — including to a formal or married name: "Kate Middleton" →
    "Catherine, Princess of Wales", "Meghan Markle" → "Meghan, Duchess of
    Sussex" (§4QG-b: §4QG demanded the target match the name and refused
    both; Kate became "no photo", Meghan the wedding article's two-person
    photo). Refused only when it cannot be that one person's article:
      * a SECTION redirect (`tofragment`): "Sasha Obama" → "Family of
        Barack Obama#Malia and Sasha Obama" (verified live 2026-10-21) —
        a group portrait rendered as her face;
      * a group / event article ("Family of …", "Wedding of …");
      * the name cut to its LATER words only ("Sasha Obama" → "Obama", a
        surname page); cut to its first word is still the person
        ("Napoleon Bonaparte" → "Napoleon")."""
    if fragment:
        return False
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", str(target or "")).strip()
    if not bare or _GROUP_ARTICLE_RE.match(bare):
        return False
    want, have = _tokens(name), _tokens(bare)
    if have and set(have) < set(want) and want[0] not in have:
        return False
    return True


def slug(subject: str) -> str:
    s = re.sub(r"[^\w]+", "_", _fold(subject), flags=re.UNICODE).strip("_")
    return (s or "subject")[:60]


def photo_key(subject: str) -> str:
    """§4QG: the cache file's key — the slug, and for a name whose slug is
    longer than slug()'s 60 characters, its first 51 + a hash of the WHOLE
    slug: two long names sharing their first 60 characters shared one file
    (and normalise_subjects dropped the second as a duplicate). Names under
    the cut keep their old file names."""
    full = re.sub(r"[^\w]+", "_", _fold(subject), flags=re.UNICODE).strip("_") or "subject"
    if len(full) <= 60:
        return full
    return f"{full[:51]}_{hashlib.sha1(full.encode('utf-8')).hexdigest()[:8]}"


def call_prompt(args) -> str:
    """§4QG-b: the prompt image_generation reads from a call's args, healed
    the way the tool heals it (prompt → image → description → subject →
    text; a list joined; stripped) — UNtruncated. The ONE normalisation the
    tool and subjects_arg compare `subject` against: the tool compared it to
    the prompt cut to MAX_PROMPT_CHARS while the egress guard compared it to
    the whole prompt, so a long `subject` == prompt was a search for one and
    the prompt for the other."""
    if not isinstance(args, dict):
        return ""
    prompt = (args.get("prompt") or args.get("image") or args.get("description")
              or args.get("subject") or args.get("text"))
    if isinstance(prompt, (list, tuple)):
        prompt = " ".join(str(x) for x in prompt if x is not None)
    elif prompt is not None and not isinstance(prompt, str):
        prompt = str(prompt)
    return (prompt or "").strip()


def subjects_arg(args) -> object:
    """§4QG: the value image_generation will use as its subjects list —
    `subjects`, or a singular `subject` beside a DIFFERENT prompt (§4PB).
    The egress guards read this: they knew only the key "subjects", so
    `subject=` searched a name unscrubbed. §4QG-b: the tool decides through
    this same function (call_prompt), so the two cannot disagree."""
    if not isinstance(args, dict):
        return None
    if args.get("subjects") is not None:
        return args.get("subjects")
    alias = args.get("subject")
    if not isinstance(alias, str) or not alias.strip():
        return None
    return alias if alias.strip() != call_prompt(args) else None


def _languages(subject: str) -> List[str]:
    return ["el", "en"] if _GREEK_RE.search(subject or "") else ["en"]


def lookup_photo_url(subject: str, proxy: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """(image_url, article_title, article_url) for the subject's lead image,
    or None when EVERY query that ran was answered and nothing matches.
    Raises SearchUnavailable when any needed query could not be asked
    (network, non-200, bad JSON) and nothing was found, or when the
    per-subject budget ran out. Synchronous — run it in a thread."""
    import curl_cffi.requests as creq
    from urllib.parse import quote as _quote
    proxies = {"https": proxy, "http": proxy} if proxy else None
    deadline = _LOOKUP_DEADLINE.get() or (time.monotonic() + LOOKUP_DEADLINE_S)
    # §4QG: every failed query, not just "was any answered" — answered=True
    # after the title query let a failed SEARCH (or el. answering and en.
    # failing) return None = "no article", and the model asked the user to
    # upload a photo of a famous person
    failed: List[str] = []
    rate_limited = False
    core = strip_honorifics(subject)
    names = [subject] + ([core] if len(_name_variants(subject)) > 1 else [])
    base ={"action": "query", "format": "json", "redirects": 1,
            "prop": "pageimages|pageprops", "piprop": "thumbnail",
            "pithumbsize": THUMB_WIDTH, "ppprop": "disambiguation"}

    def _budget_left() -> float:
        left = deadline - time.monotonic()
        if left < 1:
            raise SearchUnavailable(
                f"the photo lookup ran out of its {LOOKUP_DEADLINE_S} s budget"
                + (f" ({'; '.join(failed)})" if failed else ""), rate_limited=rate_limited)
        return left

    def _ask(lang, extra, left):
        r = creq.get(f"https://{lang}.wikipedia.org/w/api.php", params={**base, **extra},
                     proxies=proxies, timeout=min(LOOKUP_TIMEOUT_S, left),
                     headers={"User-Agent": LOOKUP_USER_AGENT})
        if r.status_code != 200:
            raise SearchUnavailable(f"HTTP {r.status_code}", rate_limited=r.status_code == 429)
        q = (r.json() or {}).get("query") or {}
        return q.get("pages") or {}, q.get("redirects") or []

    def _failed(e):
        nonlocal rate_limited
        why = str(e) if isinstance(e, SearchUnavailable) else e.__class__.__name__
        if why not in failed:
            failed.append(why)
        rate_limited = rate_limited or getattr(e, "rate_limited", False)

    def _url(lang, title):
        return f"https://{lang}.wikipedia.org/wiki/{_quote(title.replace(' ', '_'))}"

    for lang in _languages(subject):
        # §4PB step 1: the name AS a title, redirects followed — "Napoleon
        # Bonaparte" → "Napoleon" is the encyclopedia's own say-so, not a
        # guess by token overlap. §4QG: the bare name rides in the same
        # call when a leading title was stripped ("A|B" asks both).
        left = _budget_left()
        try:
            pages, redirects = _ask(lang, {"titles": "|".join(names)}, left)
        except Exception as e:  # noqa: BLE001 — network / JSON / non-200
            _failed(e)
            continue      # a failed title query skips this language's search
        redirected_to = set()
        for r in redirects:
            frm, to = str(r.get("from") or ""), str(r.get("to") or "")
            name = next((n for n in names if _fold(n) == _fold(frm)), None)
            # §4QG-b: the typed name's redirect is trusted unless it is a
            # SECTION redirect or a group/event/surname page (see the helper)
            if name and to and _redirect_is_the_subject(name, to, str(r.get("tofragment") or "")):
                redirected_to.add(to)
        for page in pages.values():
            title = str(page.get("title") or "")
            if "disambiguation" in (page.get("pageprops") or {}):
                continue
            src = str((page.get("thumbnail") or {}).get("source") or "")   # a missing page has none
            if src and (title_is_exact(subject, title) or title in redirected_to):
                return src, title, _url(lang, title)
        # step 2: search — exact titles, then the full name inside a longer
        # title, then the name without its leading title, then (§4QG) a
        # transliterated word last
        left = _budget_left()
        try:
            pages, _ = _ask(lang, {"generator": "search", "gsrsearch": names[-1], "gsrlimit": 5}, left)
        except Exception as e:  # noqa: BLE001
            _failed(e)
            continue
        hits = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
            title = str(page.get("title") or "")
            if "disambiguation" in (page.get("pageprops") or {}):
                continue
            src = str((page.get("thumbnail") or {}).get("source") or "")
            rank = _match_rank(subject, title) if src else None
            if rank is None:
                continue
            hits.append((0 if title_is_exact(subject, title) else rank, title, src))
        if hits:
            hits.sort(key=lambda h: h[0])
            _, title, src = hits[0]
            return src, title, _url(lang, title)
    if failed:
        raise SearchUnavailable(f"the encyclopedia could not be reached ({'; '.join(failed)})",
                                rate_limited=rate_limited)
    return None


def photo_is_complete(path: Path) -> bool:
    """The WHOLE file decodes as an image — a download cut off after its
    first megabyte has a valid header and was reused forever (review)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def _meta_path(photo: Path) -> Path:
    return photo.with_suffix(photo.suffix + ".json")


def _cached(folder: Path, stem: str, subject: str, allow_stale: bool = False) -> Optional[Tuple[Path, str]]:
    """A previously fetched photo of THIS subject: its provenance record
    names the same subject, the file decodes completely, and (§4QG) it is
    younger than CACHE_MAX_AGE_S. Anything else (no record, another person
    whose name slugs the same, a truncated or old file) is fetched again.
    §4QG-b: `allow_stale` skips the age test — the fallback when the
    re-fetch of an expired photo fails for a transient reason."""
    for ext in (".jpg", ".png"):
        p = folder / f"{stem}{ext}"
        if not p.is_file():
            continue
        try:
            meta = json.loads(_meta_path(p).read_text(encoding="utf-8"))
            # os.replace keeps the download's mtime: this is the fetch time
            if not allow_stale and time.time() - p.stat().st_mtime > CACHE_MAX_AGE_S:
                continue
        except Exception:  # noqa: BLE001
            continue
        if _fold(meta.get("subject", "")) == _fold(subject) and photo_is_complete(p):
            return p, str(meta.get("page") or "")
    return None


def _decoded_format(path: Path) -> str:
    """The format name ("GIF", "TIFF" …) of a file that decodes COMPLETELY
    as an image, else "" (§4QG: a whole GIF is permanent, a cut file is not)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            fmt = str(im.format or "")
            im.load()
        return fmt or "unknown"
    except Exception:  # noqa: BLE001
        return ""


def _disk_reason(e: OSError) -> str:
    return (f"could not save the photo on this machine (disk error: {e.__class__.__name__}"
            + (f", {e.strerror}" if getattr(e, "strerror", None) else "") + ")")


async def fetch_subject_photo(subject: str, sandbox_dir: Path, tor_proxy: Optional[str],
                              *, is_image) -> Tuple[Path, str]:
    """(local photo, its article). Raises LookupError(reason) —
    SearchUnavailable when the encyclopedia could not be asked,
    PhotoUnavailable (with its `kind`) when the article's photo did not
    become a usable local file."""
    folder = Path(sandbox_dir) / PHOTO_DIR
    stem = f"ref_{photo_key(subject)}"
    hit = await asyncio.to_thread(_cached, folder, stem, subject)
    if hit is not None:
        pretty_log("Subject Photo", f"{subject} → {hit[0].name} ({photo_title(hit[0]) or '?'}, cached)",
                   icon=Icons.TOOL_DOWN)
        return hit
    try:
        return await _fetch_fresh(subject, sandbox_dir, tor_proxy, folder, stem, is_image=is_image)
    except (SearchUnavailable, PhotoUnavailable) as e:
        # §4QG-b: an EXPIRED photo whose re-fetch failed for a reason a later
        # try can fix (unreachable / rate-limited search, a failed download)
        # is still this person's photo — better than no render at all
        if isinstance(e, PhotoUnavailable) and getattr(e, "kind", "transient") != "transient":
            raise
        stale = await asyncio.to_thread(_cached, folder, stem, subject, True)
        if stale is None:
            raise
        pretty_log("Subject Photo", f"{subject} → re-fetch failed ({e}); using the expired cached "
                   f"{stale[0].name} ({photo_title(stale[0]) or '?'})", level="WARNING", icon=Icons.WARN)
        return stale


async def _fetch_fresh(subject: str, sandbox_dir: Path, tor_proxy: Optional[str], folder: Path,
                       stem: str, *, is_image) -> Tuple[Path, str]:
    from ..utils.egress_guard import resolve_egress_proxy
    proxy = resolve_egress_proxy(tor_proxy, "https://en.wikipedia.org/")
    if proxy and proxy.startswith("socks5://"):
        proxy = proxy.replace("socks5://", "socks5h://")
    found = None
    # §4QG: one budget for the whole lookup of this subject — every query,
    # language and attempt (gather runs each subject in its own task, so
    # each gets its own); lookup_photo_url reads it through the ContextVar
    deadline = time.monotonic() + LOOKUP_DEADLINE_S
    token = _LOOKUP_DEADLINE.set(deadline)
    try:
        for attempt in range(LOOKUP_ATTEMPTS):
            try:
                found = await asyncio.to_thread(lookup_photo_url, subject, proxy)
                break
            except SearchUnavailable as e:
                # a retry needs the 3 s pause and at least one FULL query
                # timeout (§4QG-b: was 2 s — attempt 2 then ran one clamped
                # query and failed on the budget); otherwise skip it cleanly
                out_of_time = deadline - time.monotonic() < 3 + LOOKUP_TIMEOUT_S
                if attempt + 1 >= LOOKUP_ATTEMPTS or out_of_time:
                    # the reason (HTTP status / exception) used to reach no log:
                    # 52c9a5d2's 429s read as "network / Tor" everywhere
                    pretty_log("Subject Photo", f"{subject} → lookup failed: {e}",
                               level="WARNING", icon=Icons.WARN)
                    raise
                # a refused / dead circuit: the next attempt gets a fresh one
                from ..utils.helpers import request_new_tor_identity
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(3)
    finally:
        _LOOKUP_DEADLINE.reset(token)
    if not found:
        raise LookupError("no encyclopedia article with a photo matches that name")
    url, title, page = found
    ext = ".png" if url.split("?", 1)[0].lower().endswith(".png") else ".jpg"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # §4QG: THIS machine's fault — it used to read "no usable photo …
        # ask the user to upload one"
        pretty_log("Subject Photo", f"{subject} → {_disk_reason(e)}", level="WARNING", icon=Icons.WARN)
        raise PhotoUnavailable(_disk_reason(e), page, title, kind="disk")
    # Download under a temporary name and promote only a COMPLETE image: a
    # transfer that failed half-way left a truncated file under the final
    # name (`tool_download_file` writes in place).
    part = f"{PHOTO_DIR}/{stem}.part-{uuid.uuid4().hex[:8]}{ext}"
    from .file_system import tool_download_file
    res = await tool_download_file(url=url, sandbox_dir=Path(sandbox_dir),
                                   tor_proxy=tor_proxy, filename=part)
    tmp = Path(sandbox_dir) / part
    try:
        # §4PB: the article exists — a failed or truncated download is a
        # TRANSIENT fault (PhotoUnavailable), never "no photo of this person"
        if not str(res).startswith("SUCCESS") or not tmp.is_file():
            raise PhotoUnavailable(f"the photo from {page} could not be downloaded", page, title)
        head = await asyncio.to_thread(lambda: tmp.open("rb").read(16))
        if not is_image(head):
            # §4QG: a WHOLE image in a format the node refuses (the lead
            # image of some articles is a GIF) is permanent: "call again
            # once" fetched the same GIF and failed the same way
            fmt = await asyncio.to_thread(_decoded_format, tmp)
            if fmt:
                raise PhotoUnavailable(
                    f"the article's photo ({page}) is a {fmt} image, a format the image node cannot use "
                    f"— fetching it again gets the same file", page, title, kind="unsupported")
            raise PhotoUnavailable(f"what {page} served is not a complete photo", page, title)
        if not await asyncio.to_thread(photo_is_complete, tmp):
            raise PhotoUnavailable(f"what {page} served is not a complete photo", page, title)
        final = folder / f"{stem}{ext}"
        os.replace(tmp, final)
        _meta_path(final).write_text(json.dumps(
            {"subject": subject, "title": title, "page": page, "url": url}, ensure_ascii=False),
            encoding="utf-8")
    except OSError as e:
        pretty_log("Subject Photo", f"{subject} → {_disk_reason(e)}", level="WARNING", icon=Icons.WARN)
        raise PhotoUnavailable(_disk_reason(e), page, title, kind="disk")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    pretty_log("Subject Photo", f"{subject} → {final.name} ({title})", icon=Icons.TOOL_DOWN)
    return final, page


#: Render shapes the node offers lie well inside this band; a wider ask is
#: refused BEFORE a canvas is built (§4PB: width=200000,height=1 spent 14 s
#: allocating before MemoryError).
ASPECT_BAND = (0.25, 4.0)


def photo_title(photo: Path) -> str:
    """The article title recorded beside a fetched photo ("" if unknown)."""
    try:
        return str(json.loads(_meta_path(photo).read_text(encoding="utf-8")).get("title") or "")
    except Exception:  # noqa: BLE001
        return ""


def _pad_to_aspect(im, aspect: float):
    """``im`` centred on a white canvas of width/height == ``aspect``."""
    from PIL import Image
    if not (ASPECT_BAND[0] <= float(aspect) <= ASPECT_BAND[1]):
        raise ValueError(f"aspect {aspect:.3g} outside {ASPECT_BAND[0]}–{ASPECT_BAND[1]}")
    w, h = im.size
    if abs(w / h - aspect) < 0.01:
        return im
    cw, ch = (round(h * aspect), h) if w / h < aspect else (w, round(w / aspect))
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    canvas.paste(im, ((cw - w) // 2, (ch - h) // 2))
    return canvas


def _on_white(im):
    """§4QG: ``im`` as RGB with any transparency composited onto WHITE (the
    joined reference's background). `.convert("RGB")` dropped the alpha, so
    a transparent PNG (a logo, a cut-out portrait, a palette image with a
    transparent index) arrived as a figure on BLACK; a 16-bit greyscale
    photo ("I;16") clipped to white instead — it is scaled to 8 bits."""
    from PIL import Image
    if im.mode.startswith("I;16") or (im.mode == "I" and (im.getextrema() or (0, 0))[1] > 255):
        im = im.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    elif im.mode == "F":
        # §4QG-b: a float image (0–1, or any other range) converted straight
        # to "L" truncated 0.5 to 0 — the photo came out black. Scale its
        # range to 0–255 first.
        lo, hi = im.getextrema() or (0.0, 0.0)
        top = 1.0 if 0.0 <= lo and hi <= 1.0 else max(float(hi), 1e-9)
        im = im.point(lambda v: v * (255.0 / top)).convert("L")
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def photo_needs_flatten(path: Path) -> bool:
    """§4QG: the photo cannot go to the node as its own bytes — it has
    transparency (the node turns it black) or 16-bit samples."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return (im.mode in ("RGBA", "LA", "PA", "I", "F") or im.mode.startswith("I;16")
                    or "transparency" in im.info)
    except Exception:  # noqa: BLE001 — unreadable: the raw-bytes path reports it as before
        return False


def combine_side_by_side(paths: List[Path], aspect: Optional[float] = None) -> bytes:
    """One JPEG with the photos left to right at a common height, padded to
    ``aspect`` (width/height) so the node's stretch-to-size keeps faces
    undistorted. ``aspect=None`` keeps the strip's own shape."""
    from PIL import Image, ImageOps
    ims = []
    for p in paths:
        with Image.open(p) as im:
            im = _on_white(ImageOps.exif_transpose(im))       # §4QG: alpha → white, not black
            w = max(1, round(im.width * COMBINED_HEIGHT / im.height))
            cap = round(COMBINED_HEIGHT * 1.5)
            if w <= cap:
                ims.append(ImageOps.fit(im, (w, COMBINED_HEIGHT), Image.Resampling.LANCZOS,
                                        centering=(0.5, 0.3)))
            else:
                # a panorama would squeeze the others into slivers — but
                # CROPPING it cut people out of "the scene to keep" (§4PB):
                # scale it to the cap and letterbox the rest
                h = max(1, round(im.height * cap / im.width))
                panel = Image.new("RGB", (cap, COMBINED_HEIGHT), (255, 255, 255))
                panel.paste(im.resize((cap, h), Image.Resampling.LANCZOS), (0, (COMBINED_HEIGHT - h) // 2))
                ims.append(panel)
    width = sum(i.width for i in ims) + COMBINED_GAP * (len(ims) - 1)
    out = Image.new("RGB", (width, COMBINED_HEIGHT), (255, 255, 255))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + COMBINED_GAP
    if aspect:
        out = _pad_to_aspect(out, aspect)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    return buf.getvalue()




#: `keep_first` (req 8473053b): the other panels stay narrow so the picture
#: being edited gets the most of the node's 768x512 pixel budget.
_OTHER_PANEL_MAX = round(COMBINED_HEIGHT * 0.75)
_OTHER_PANEL_MIN = round(COMBINED_HEIGHT * 0.5)
#: how far from the predicted edge the gap may have moved in a render, as a
#: share of its width (measured: 1-2 px of 736)
_GAP_SEARCH = 0.015
#: the gap, when it survives, is a run of flat columns (gray standard deviation),
#: at least half its expected width, and gives the exact place to cut. One flat
#: column could be a keyline inside the picture (review). The model may also paint
#: the gap over (live, twice, purple with soft edges), so a missing gap does not
#: mean a lost layout: the panels' correlation decides that.
_GAP_MAX_STDDEV = 8.0
#: reference pixels at or above this are empty background, not panel content
_OTHERS_BLANK_LEVEL = 240
#: fewer content cells than this (of 32x32) cannot show a layout was kept
_OTHERS_MIN_CELLS = 8
#: correlation of the other panels' content, render against reference. Brightness
#: alone passed a plain wall of similar gray (review). Measured: real renders that
#: kept the layout 0.85-0.99, lost ones (a scene, walls, another person) -0.31 to 0.13.
_OTHERS_MIN_CORRELATION = 0.6


def combine_keep_first(paths: List[Path]) -> Tuple[bytes, Tuple[float, float, float, float]]:
    """The first picture whole, then the others beside it, narrowed by scaling
    and never cropped (review: a crop can remove the face to copy). Returns
    the JPEG and where the first picture sits in it, as shares of the strip:
    (right edge, top, bottom, gap width); ``crop_first_panel`` cuts it back out."""
    from PIL import Image, ImageOps
    with Image.open(paths[0]) as im:
        first = _on_white(ImageOps.exif_transpose(im))
    if not ASPECT_BAND[0] <= first.width / first.height <= ASPECT_BAND[1]:
        raise ValueError("the first picture's shape is outside what the node renders")
    w = max(1, round(first.width * COMBINED_HEIGHT / first.height))
    cap = round(COMBINED_HEIGHT * 1.5)
    if w <= cap:
        panel0 = first.resize((w, COMBINED_HEIGHT), Image.Resampling.LANCZOS)
        top = 0
        inner_h = COMBINED_HEIGHT
    else:                                   # never crop what is being edited
        inner_h = max(1, round(first.height * cap / first.width))
        top = (COMBINED_HEIGHT - inner_h) // 2
        panel0 = Image.new("RGB", (cap, COMBINED_HEIGHT), (255, 255, 255))
        panel0.paste(first.resize((cap, inner_h), Image.Resampling.LANCZOS), (0, top))
    side = max(_OTHER_PANEL_MIN, min(panel0.width, _OTHER_PANEL_MAX))
    ims = [panel0]
    for p in paths[1:]:
        with Image.open(p) as im:
            im = _on_white(ImageOps.exif_transpose(im))
            ow = max(1, round(im.width * COMBINED_HEIGHT / im.height))
            if ow <= side:
                ims.append(im.resize((ow, COMBINED_HEIGHT), Image.Resampling.LANCZOS))
            else:
                h = max(1, round(im.height * side / im.width))
                panel = Image.new("RGB", (side, COMBINED_HEIGHT), (255, 255, 255))
                panel.paste(im.resize((side, h), Image.Resampling.LANCZOS), (0, (COMBINED_HEIGHT - h) // 2))
                ims.append(panel)
    width = sum(i.width for i in ims) + COMBINED_GAP * (len(ims) - 1)
    out = Image.new("RGB", (width, COMBINED_HEIGHT), (255, 255, 255))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + COMBINED_GAP
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), (panel0.width / width, top / COMBINED_HEIGHT, (top + inner_h) / COMBINED_HEIGHT,
                            COMBINED_GAP / width)


def _others_correlation(render, strip, edge: float) -> Optional[float]:
    """Pearson correlation of the panels beside the first picture, over the
    cells that hold content in the reference. None: too little content to tell."""
    def cells(i):
        return i.crop((min(i.width - 1, round(edge * i.width) + 1), 0, i.width, i.height)).resize((32, 32)).tobytes()
    a, b = cells(render), cells(strip)
    idx = [k for k, v in enumerate(b) if v < _OTHERS_BLANK_LEVEL]
    if len(idx) < _OTHERS_MIN_CELLS:
        return None
    x, y = [a[k] for k in idx], [b[k] for k in idx]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx, sy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    if not sx or not sy:
        return 0.0
    return sum((u - mx) * (v - my) for u, v in zip(x, y)) / (sx * sy) ** 0.5


def crop_first_panel(image_bytes: bytes, box: Tuple[float, float, float, float], strip_bytes: bytes) -> Optional[bytes]:
    """The first picture cut out of a render of the ``strip_bytes`` reference
    as a PNG, or None when the other panels' content is gone: the render did
    not keep the layout and cutting it would destroy the picture."""
    from PIL import Image, ImageStat
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            rgb = im.convert("RGB")
        with Image.open(io.BytesIO(strip_bytes)) as st:
            strip = st.convert("L")
    except Exception:  # noqa: BLE001 — unreadable: the caller keeps it whole
        return None

    gray = rgb.convert("L")
    corr = _others_correlation(gray, strip, box[0])
    if corr is None or corr < _OTHERS_MIN_CORRELATION:
        return None
    w, h = rgb.size
    edge = round(box[0] * w)
    reach = max(2, round(w * _GAP_SEARCH))
    need = max(2, round(box[3] * w / 2))
    run, cut_at = 0, edge
    for x in range(max(1, edge - reach), min(w - 1, edge + reach + need) + 1):
        run = run + 1 if ImageStat.Stat(gray.crop((x, 0, x + 1, h))).stddev[0] <= _GAP_MAX_STDDEV else 0
        if run >= need:
            cut_at = x - run + 1
            break
    buf = io.BytesIO()
    top = round(box[1] * h)
    rgb.crop((0, top, cut_at, max(top + 1, round(box[2] * h)))).save(buf, format="PNG")
    return buf.getvalue()


def pad_photo(path: Path, aspect: Optional[float]) -> bytes:
    """One photo padded to ``aspect`` (a requested render shape); §4QG:
    ``aspect=None`` = its own shape, only flattened onto white."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = _on_white(ImageOps.exif_transpose(im))
        if aspect:
            im = _pad_to_aspect(im, aspect)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


#: §4QG: a name suffix that a comma sets off ("Martin Luther King, Jr.") —
#: it belongs to the name before it, never a subject of its own
_NAME_SUFFIX = re.compile(r"^(?:jr|sr|ii|iii|iv|phd|md|esq|obe|mbe|cbe|kbe)\.?$", re.IGNORECASE)
#: "and" / "&" / "και" between names. §4QG-b (operator): ALWAYS a split, as
#: before §4QG — "Τσίπρας και Σαμαράς" / "Tsipras and Mitsotakis" (single
#: surnames, the ORIGINAL live failure of §4MG) are two people; §4QG split
#: only between two-word names and sent them as ONE name → "no article". A
#: band or brand ("Simon & Garfunkel") goes in a LIST, which is never split.
_AND = re.compile(r"\s*&\s*|\s+(?:and|και)\s+", re.IGNORECASE)


def _split_names(txt: str) -> List[str]:
    parts: List[str] = []
    for p in re.split(r"\s*(?:,|;|\n)\s*", txt):
        if parts and _NAME_SUFFIX.match(p.strip()):
            parts[-1] = f"{parts[-1]}, {p.strip()}"
        else:
            parts.append(p)
    out: List[str] = []
    for p in parts:
        out.extend(_AND.split(p))
    return out


def normalise_subjects(subjects) -> List[str]:
    """Distinct names (by photo key, so "Alexis-Tsipras" and "Alexis
    Tsipras" are one) from a list, a JSON list string, or one string of
    names. §4QG: a LIST is never split; a string splits on commas,
    semicolons and new lines (a suffix such as "Jr." stays with its name),
    and (§4QG-b, as before §4QG) on every "and"/"&"/"και".
    Whitespace inside a name is collapsed (a new line broke the result
    record's parse)."""
    if subjects is None:
        return []
    if isinstance(subjects, str):
        txt = subjects.strip()
        parsed = None
        if txt.startswith("["):
            try:
                parsed = json.loads(txt)
            except Exception:  # noqa: BLE001
                parsed = None
        subjects = parsed if isinstance(parsed, list) else _split_names(txt)
    if not isinstance(subjects, (list, tuple)):
        return []
    out: List[str] = []
    seen = set()
    for s in subjects:
        s = re.sub(r"\s+", " ", str(s or "")).strip()[:120]
        if s and photo_key(s) not in seen:
            seen.add(photo_key(s))
            out.append(s)
    return out
