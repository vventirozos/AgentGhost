"""Who said it (§4KY, 2026-10-03).

The background fact extractor (`run_smart_memory_task`) wrote facts ABOUT THE
OWNER from anything in the episode: the owner's QUESTION "i'm a doctor/medicine
denialist, is this male privilege?" became `user HAS_PROFESSION doctor`; a
role-play ("pretend you are my wife Maria and we live in London") replaced the
owner's real wife and home; a pasted colleague's bio replaced the employer; the
agent's own stats, tasks and files became "the user has"; a channel member's
"I have type 1 diabetes" in the owner's thread reached the owner's graph.

A fact about the owner is written only when the OWNER STATED it: in one of the
owner's own lines, in a first-person declarative sentence that is not a
question, not role-play / a hypothetical, and not inside a quote. A statement
that NEGATES it ("I'm not a doctor") is a correction: the matching owner fact
is removed instead.
"""
import re

from .profile import _fold

#: labels the Slack bot puts on other people's thread messages
from ..utils.logging import FOREIGN_MESSAGE_LABELS

_FIRST_PERSON = re.compile(
    r"\b(?:i|i'?m|i'?ve|i'?d|i'?ll|me|my|mine|myself|we|our|ours|us)\b"
    r"|(?<!\w)(?:μου|μας|ειμαι|ειμαστε|εχω|εχουμε|εγω|εμεις|μενω|μενουμε|δουλευω|δουλευουμε|λεγομαι|ζω|ζουμε|"
    r"μετακομισ(?:α|αμε)|παντρευτηκ(?:α|αμε)|γεννηθηκα|αγορασ(?:α|αμε)|θελω|προτιμω)(?!\w)",
    re.IGNORECASE)
#: an instruction from the owner states the owner's preference ("Always write
#: tests", "Use metric units") — it has no "I" but it is theirs
_IMPERATIVE = re.compile(
    r"^\s*(?:please\s+)?(?:always|never|don'?t|do|use|prefer|avoid|keep|make|write|call|reply|answer|remember|"
    r"note|stop|start|give|show|tell|send|put|add|run|check|ask)\b"
    r"|^\s*(?:παντα|ποτε|μην|χρησιμοποιησε|θυμησου)(?!\w)",
    re.IGNORECASE)
#: §4QN (the r4 probe's residue, fixed once): "you're my assistant, remember that
#: …" instructs the agent, it casts no role; "a character designer" is a job;
#: "I can't imagine living anywhere but Athens" states — each excluded narrowly
_ROLE_PLAY = re.compile(
    r"\b(?:pretend|role[\s-]?play|act\s+as|let'?s\s+say|suppose|supposing|hypothetic\w*|what\s+if|"
    r"in\s+(?:my|a|the)\s+(?:story|novel|game|book|script)|"
    r"character(?!\s+(?:design|artist|animat|model|encod|set\b|limit|count|string|device))|fiction\w*|play\s+the\s+role|"
    r"play\s+a\s+game|(?:you\s+are|you['’]?re)\s+(?:now\s+)?my(?!\s+(?:ai\b|assistant|agent|helper|ghost|memory))|as\s+if)\b"
    r"|(?<!can't\s)(?<!can’t\s)(?<!cannot\s)(?<!couldn't\s)(?<!couldn’t\s)(?<!cant\s)\bimagine\b"
    r"|(?<!\w)(?:φαντασου|υποθεσε|ας\s+πουμε)(?!\w)",
    re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:not|no\s+longer|never|isn'?t|aren'?t|wasn'?t|don'?t|doesn'?t|didn'?t|nor|no)\b"
    r"(?!\s+(?:ever\s+)?forget)"      # "don't forget I live in Athens" / "never forget …" STATES (review r2)
    r"|(?<!\w)(?:δεν|οχι|ουτε)(?!\w)",
    re.IGNORECASE)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»|```.*?```", re.DOTALL)
#: a sentence that ASKS without a question mark ("do you remember my wife
#: fotini", "tell me about my car bmw", "remind me where i live") — a
#: question is never the owner's statement of a fact (§4KY; §4QM review r1:
#: these attributed facts and would have written "wife = fotini")
#: review r2: a wh-word followed by "I" opens a statement ("When I was 20 I
#: lived in Patras"); a modal request to REMEMBER is an instruction ("can you
#: remember that I live in Athens", "may I remind you …", "did I mention …");
#: `ποτε` is "never" as often as "when" and is left to the imperative rule
#: review r3 (the rule had flip-flopped twice — narrowed to its two real
#: shapes): "<aux> you …" asks unless it goes on to a RETENTION verb ("can
#: you remember that I live in Athens" instructs; "can you remind me where
#: my wife works" / "can you forget my ex-wife" do not state); "<aux> I/we …"
#: asks unless it goes on to a TELLING verb ("did I mention", "have I told
#: you", "may I remind you" state). The wh-word test is SENTENCE-level only:
#: after a comma "who/which/where" is a relative pronoun ("My wife, who is
#: called Fotini, is a doctor"). Greek question words are tested with their
#: accent on the RAW sentence (folded "που" is also the relative "that").
_RETAIN_VERBS = r"(?:(?:please|also|just|kindly)\s+)*(?:remember|note|keep|save|store|record|add|write|log|update|make\s+a\s+note|memori[sz]e|learn)"
_TOLD_VERBS = r"(?:ever\s+)?(?:mention(?:ed)?|tell|told|say|said|remember|remind|ask\s+you\s+to)"
_ASK_AUX = r"(?:do|does|did|am|is|are|was|were|have|has|had)"       # "do you remember my dog" ASKS, always
_MODAL = r"(?:can|could|would|will|should|shall|may|might)"           # "can you remember that …" may instruct
_AUX = r"(?:" + _ASK_AUX + r"|" + _MODAL + r")"
_ASKS_CORE = (r"(?:" + _ASK_AUX + r"\s+(?:you|u|ya)\b"
              r"|" + _MODAL + r"\s+(?:you|u|ya)\b(?!\s+" + _RETAIN_VERBS + r")"
              r"|" + _AUX + r"\s+(?:i|we)\b(?!\s+" + _TOLD_VERBS + r")"
              r"|" + _AUX + r"\s+(?:it|there|anyone)\b"
              r"|" + _AUX + r"\s+(?:my|our|your|the|this|that|she|he|they|his|her|their|a|an|anybody|someone)\b"
              r"|any\s+(?:idea|clue)\b"
              r"|(?:tell|remind|show)\s+(?:me|us)\b|θυμασαι|πες\s+μου)")
_QUESTION_OPENER = re.compile(
    r"^\s*(?:please\s+)?(?:(?:what|where|who|whom|whose|which|when|how|why)\b(?!\s+(?:i|we|my|our)\b)"
    r"|" + _ASKS_CORE + r"|(?:τι|ποιος|ποια|ποιο|ποιοι|ποιες)\b)", re.IGNORECASE)
_CLAUSE_ASKS = re.compile(r"^\s*(?:please\s+)?" + _ASKS_CORE
    + r"|\b(?:if|whether)\s+(?:you|u|ya)\s+(?:still\s+|even\s+|really\s+)?(?:remember|know|recall|have|got|keep|kept|saved?|stored?)\b", re.IGNORECASE)
_GREEK_WH_RAW = re.compile(r"^\s*(?:πότε|πού|πώς|γιατί)\b", re.IGNORECASE)
#: unaccented typing ("Που μενω, στην Αθηνα"): a sentence with NO accent anywhere
#: that opens with που/πως/γιατι asks; with accents present an unaccented "που" is
#: the relative "that" (§4QN; ποτε stays out — "never" as often as "when")
_GREEK_WH_UNACCENTED = re.compile(r"^\s*(?:που(?!\s+λες)|πως|γιατι)\b", re.IGNORECASE)   # "που λες, …" is a filler
_GREEK_ACCENT = re.compile(r"[άέήίόύώΐΰΆΈΉΊΌΎΏ]")
#: content the owner is READING or PASTING ("I'm reading a bio: Maria lives in
#: London", "the article says: …") is not their statement: when the head before
#: a colon names the ACT (a reading/pasting verb, or a document that "says"),
#: the tail after it is dropped (§4QN). The head must name the act, not a thing:
#: "My email: …", "My bio: …", "I read your email: … I live in Athens" are the
#: owner's own (review r1 of §4QN: a noun list dropped all three)
_REPORTED_HEAD = re.compile(
    r"\b(?:reading|pasting|quoting|copying)\b"
    r"|\b(?:article|bio|biography|post|page|site|book|email|message|excerpt|snippet|text)\s+(?:says|said|reads|states|wrote)\b"
    r"|\bhere'?s\s+(?:a|an|the|some)\s+(?:bio|biography|article|excerpt|snippet|quote|text|email)\b",
    re.IGNORECASE)
#: a tag question ("you know my wife fotini, right") asks (review r4)
_TAG_Q = re.compile(r"\b(?:right|correct|huh|eh|yes|no)\s*$|(?<!\w)(?:σωστα|ετσι|ναι|οχι)\s*$", re.IGNORECASE)
#: a greeting / vocative / filler before the real sentence ("hi, do you
#: remember my wife fotini" — review r2: the opener was anchored at the
#: sentence start and the greeting hid the question); stripped at the
#: SENTENCE level only, in `owner_statements` and `owner_clause_for` alike
_LEAD_IN = re.compile(r"^\s*(?:(?:hi|hey|hello|yo|ok(?:ay)?|so|and|well|also|ghost|γει[αά](?:\s+σου)?|ε|λοιπον|λοιπόν)[,!\s]+)+",
                      re.IGNORECASE)
#: review r3 (2026-10-31): a reason is its own clause — "I don't live in
#: Patras anymore since we moved to Athens" negated Athens too
_CLAUSE_SPLIT = (r",|;|:|\bbut\b|\binstead\b|\bwhereas\b|\bbecause\b|\bsince\b|(?<!\w)αλλ[αά](?!\w)"
                 r"|(?<!\w)(?:γιατ[ιί]|επειδ[ηή]|αφο[υύ])(?!\w)")
#: words that say nothing about WHICH fact it is
_STOP = frozenset("""user users the a an and or of to in on at for with is are was were has have had be been being
their his her its this that who which as by from also now currently am do does did very really just about
""".split())

#: predicates that name the AGENT's or a project's state, never the owner's
#: life (review: `user HAS_STAT total lessons`, `HAS_TASK create cli.py`,
#: `HAS_SANDBOX_FILES mars_distance.py`, `HAS_PROJECT_CODENAME zephyrine…`)
AGENT_STATE_TOKENS = frozenset({
    "PROJECT", "PROJECTS", "SKILL", "SKILLS", "TASK", "TASKS", "SANDBOX", "FILE", "FILES", "DOCUMENTATION",
    "STAT", "STATS", "LEARNING", "CODENAME", "WORKSPACE", "RESOURCE", "INTROSPECTION",
    "COMPETENCE", "TEST", "PROBE", "SESSION", "REQUEST", "REQUESTED", "ASKED", "QUERY",
    # §4NG: the owner's chatter is not a fact about the owner ("user GREETED ghost")
    "GREETED", "GREETS", "THANKED", "THANKS", "APPROVES", "APPROVED",
})


def owner_lines(episode: str) -> list:
    """The OWNER's own lines of a smart-memory episode (``USER:`` lines that
    do not carry another person's label)."""
    text = str(episode or "")
    lines = text.splitlines()
    roles = [re.match(r"\s*(user|ai|assistant|system|tool)\s*:", ln, re.IGNORECASE) for ln in lines]
    if not any(roles):
        # no speaker marks at all (a legacy journal item): one speaker, the owner
        return [] if text.strip().startswith(FOREIGN_MESSAGE_LABELS) else [text.strip()]
    out = []
    owner_speaking = False
    for ln, m in zip(lines, roles):
        if m:
            body = ln[m.end():].strip()
            # a message's FIRST line carries the speaker (and any foreign label)
            owner_speaking = (m.group(1).lower() == "user"
                              and not body.startswith(FOREIGN_MESSAGE_LABELS))
            if owner_speaking and body:
                out.append(body)
        elif owner_speaking and ln.strip():
            # §4NG review (MAJOR): a multi-line owner message — its later lines
            # carry no speaker mark and were dropped ("Some notes:\nOur office
            # moved to Patras." kept only "Some notes:"); inner lines that LOOK
            # like speaker marks are quoted by the arc builder (`_unmark_roles`)
            out.append(ln.strip())
    return out


def latest_owner_episode(episode: str) -> str:
    """The episode cut down to its LAST owner message (``USER:`` line plus its
    unmarked continuation lines), still speaker-marked so `owner_lines` and
    `owner_statements` read it as before.

    Consolidation review 2026-10-31: a journal item carries the last four
    messages, so the owner's sentence from the previous turn was extracted
    AGAIN on this turn's item — "my dog's name is Hanzo" stored twice 5 s
    apart, and the memory-usage note written under two different keys 4 s
    apart. Each owner message has its own journal item; extraction reads only
    that one. An episode with no speaker marks (a legacy item) is returned
    unchanged; one whose last owner message is another person's (labelled)
    returns ""."""
    text = str(episode or "")
    lines = text.splitlines()
    roles = [re.match(r"\s*(user|ai|assistant|system|tool)\s*:", ln, re.IGNORECASE) for ln in lines]
    if not any(roles):
        return text
    start = None
    for i, m in enumerate(roles):
        if m and m.group(1).lower() == "user":
            start = i
    if start is None:
        return ""
    end = next((j for j in range(start + 1, len(lines)) if roles[j]), len(lines))
    out = "\n".join(lines[start:end])
    return "" if not owner_lines(out) else out


def owner_statements(episode: str) -> list:
    """``(clause, negated)`` for each clause of a first-person or imperative
    declarative sentence the owner wrote: questions and quoted text excluded,
    and NOTHING from a turn that sets up role-play anywhere (§4KZ: "Let's play
    a game. You are my wife Maria. We live in London." passed sentence by
    sentence). Negation is per CLAUSE: "spelled Φωτεινή, not Fotini" states
    Φωτεινή and negates Fotini."""
    lines = owner_lines(episode)
    if any(_ROLE_PLAY.search(_fold(b)) for b in lines):
        return []
    out = []
    for body in lines:
        body = _QUOTED.sub(" ", body)
        for m in re.finditer(r"[^.!?;\n]+[.!?;]?", body):
            sent = _LEAD_IN.sub("", m.group(0).strip())
            if not sent or sent.endswith("?") or sent.lstrip().startswith(">") \
                    or (sent.endswith(";") and re.search(r"[\u0370-\u03FF]", sent)):   # Greek ";" is the question mark
                continue
            if _GREEK_WH_RAW.search(sent) or (_GREEK_WH_UNACCENTED.search(sent) and not _GREEK_ACCENT.search(sent)):
                continue
            head, colon, _tail = sent.partition(":")
            if colon and _REPORTED_HEAD.search(_fold(head)):
                sent = head                      # §4QN: the reported tail is someone else's
            f = _fold(sent)
            if _QUESTION_OPENER.search(f) or _TAG_Q.search(f):
                continue
            if not (_FIRST_PERSON.search(f) or _IMPERATIVE.search(f)):
                continue
            for clause in re.split(_CLAUSE_SPLIT, f):
                clause = clause.strip(" .!")
                # a clause that itself ASKS ("I wonder, do you remember my wife
                # fotini") states nothing (review r2); a relative clause ("my
                # wife, who is called Fotini, …") is not a question (review r3)
                if clause and not _CLAUSE_ASKS.search(clause):
                    out.append((clause, bool(_NEGATION.search(clause))))
    return out


def owner_clause_for(value: str, episode: str, cue: str = None) -> str:
    """§4QM: the owner's OWN clause (raw text, case and accents kept) that
    states ``value`` — for the ledger's "you said". Among the owner's
    statement clauses of ``episode`` (`owner_statements`: no questions, no
    quotes, no role-play, not negated) the one that matches the field's
    predicate ``cue`` and covers the most of the value's words wins; ties go
    to the earliest. "" when no clause shares a content word with the value
    (review r1: "I was in Athens for a wedding. We moved to Athens in May."
    must name the second for a home)."""
    words = {w for w in re.findall(r"[^\W_]{3,}", _fold(str(value or "")))} - _STOP
    if not words:
        return ""
    accepted = {c for c, negated in owner_statements(episode) if not negated}
    if not accepted:
        return ""
    cue_re = re.compile(cue, re.IGNORECASE) if cue else None
    best, best_score = "", (-1, -1)
    for body in owner_lines(episode):
        body = _QUOTED.sub(" ", body)
        for m in re.finditer(r"[^.!?;\n]+[.!?;]?", body):
            for raw in re.split(_CLAUSE_SPLIT, _LEAD_IN.sub("", m.group(0).strip()), flags=re.IGNORECASE):
                raw = raw.strip(" .!")
                f = _fold(raw).strip(" .!")
                if not raw or f not in accepted:
                    continue
                cov = len(words & set(re.findall(r"[^\W_]{3,}", f)))
                if cov == 0:
                    continue
                score = (1 if cue_re is not None and cue_re.search(f) else 0, cov)
                if score > best_score:
                    best, best_score = raw[:200], score
    return best


#: a profile field's predicate cue (the key's tokens → the single-valued
#: predicate whose statement words it shares), for `owner_clause_for`
_FIELD_CUES = (("LIVES_IN", {"location", "lives", "live", "home", "city", "residence", "address", "town"}),
               ("WORKS_AT", {"employer", "work", "works", "job", "company", "workplace"}),
               ("MARRIED_TO", {"wife", "husband", "spouse", "partner", "married"}),
               ("HAS_BIRTHDATE", {"birth", "birthday", "birthdate", "born", "dob"}),
               ("HAS_PROFESSION", {"profession", "occupation", "role", "title"}),
               ("HAS_NAME", {"name", "named", "nickname"}))


def cue_for_field(category, key) -> str:
    toks = set(re.split(r"[^a-z]+", (str(category or "") + " " + str(key or "")).lower())) - {""}
    for pred, keys in _FIELD_CUES:
        if toks & keys:
            return PREDICATE_CUES.get(pred, "")
    return ""


#: a field ABOUT someone or something other than the owner's own life —
#: grouped, so `assets.dogs` and `pet.dog_name` are about the same subject
_FIELD_SUBJECTS = {
    "spouse": {"wife", "husband", "spouse", "partner", "girlfriend", "boyfriend", "fiance", "fiancee"},
    "pet": {"pet", "pets", "dog", "dogs", "cat", "cats", "puppy", "kitten"},
    "child": {"son", "sons", "daughter", "daughters", "kid", "kids", "child", "children", "baby"},
    "parent": {"father", "mother", "dad", "mom", "mum", "parent", "parents"},
    "sibling": {"brother", "brothers", "sister", "sisters", "sibling", "siblings"},
    "other": {"friend", "friends", "boss", "colleague", "colleagues", "neighbour", "neighbor", "office", "school",
              "business", "previous", "former", "old", "holiday", "vacation", "summer", "family"},
}


def _field_tokens(category, key):
    return (set(re.split(r"[^a-z]+", str(category or "").lower())) - {""},
            set(re.split(r"[^a-z]+", str(key or "").lower())) - {""})


def field_subject(category, key) -> frozenset:
    """Who/what a profile field is about, beyond the owner: {"spouse"} for
    relationships.wife_employer, {"pet"} for assets.dogs, empty for
    root.employer. Review r2: `wife_employer` read as the OWNER's employer."""
    ct, kt = _field_tokens(category, key)
    return frozenset(g for g, words in _FIELD_SUBJECTS.items() if (ct | kt) & words)


def field_kind(category, key) -> str:
    """The single-valued predicate KIND a profile field holds (``LIVES_IN`` for
    root.location, ``MARRIED_TO`` for relationships.wife_name …), or "" for a
    field with no kind (assets.vehicles, pet.dog_name).

    Review r2: every ``*_name`` was HAS_NAME (son_name ~ father_name) and a
    relative's field took the owner's kind (wife_employer → WORKS_AT,
    office_location → LIVES_IN). A field about someone else has a kind only
    when it IS the spouse (wife_name → MARRIED_TO); one whose words name two
    kinds (work_location) has none."""
    ct, kt = _field_tokens(category, key)
    subj = field_subject(category, key)
    if subj:
        if subj == {"spouse"} and kt <= _FIELD_SUBJECTS["spouse"] | {"name", "named", "full", "first"}:
            return "MARRIED_TO"
        return ""
    kinds = {pred for pred, keys in _FIELD_CUES if (ct | kt) & keys}
    return next(iter(kinds)) if len(kinds) == 1 else ""


#: extractor predicates → the field kind they state (beyond the graph's own
#: canonical_predicate aliases), for matching an owner correction to a field
_PREDICATE_KINDS = {
    "HAS_LOCATION": "LIVES_IN", "HAS_ADDRESS": "LIVES_IN", "LOCATED_IN": "LIVES_IN", "LIVES_AT": "LIVES_IN",
    "WORKS_FOR": "WORKS_AT", "EMPLOYED_BY": "WORKS_AT", "HAS_EMPLOYER": "WORKS_AT", "WORKS_IN": "WORKS_AT",
    "IS_A": "HAS_PROFESSION", "IS_AN": "HAS_PROFESSION", "IS": "HAS_PROFESSION", "WORKS_AS": "HAS_PROFESSION",
    "HAS_OCCUPATION": "HAS_PROFESSION", "HAS_ROLE": "HAS_PROFESSION", "HAS_JOB_TITLE": "HAS_PROFESSION",
    "IS_NAMED": "HAS_NAME", "NAMED": "HAS_NAME", "IS_CALLED": "HAS_NAME",
}


def predicate_kind(canonical_predicate) -> str:
    """The field kind an (already canonicalised) predicate states, or ""."""
    p = str(canonical_predicate or "").upper().strip()
    if p in PREDICATE_CUES:
        return p
    return _PREDICATE_KINDS.get(p, "")


#: what a statement must SAY for a single-valued owner fact to change — a
#: shared value word is not enough (§4KZ: "I'm in Kyllini this weekend" made
#: "lives in Kyllini" and replaced the owner's home)
PREDICATE_CUES = {
    "LIVES_IN": r"\b(?:live|lives|living|lived|moved?|moving|home|reside\w*|settled)\b|(?<!\w)(?:μενω|μενουμε|μετακομισ\w*|σπιτι)(?!\w)",
    "WORKS_AT": r"\b(?:work|works|working|worked|job|employ\w*|joined|hired|career)\b|(?<!\w)(?:δουλευω|δουλεια|εργαζομαι)(?!\w)",
    "MARRIED_TO": r"\b(?:wife|husband|married|spouse|partner)\b|(?<!\w)(?:γυναικα|συζυγ\w*|παντρε\w*)(?!\w)",
    "HAS_NAME": r"\b(?:name|named|call\s+me|called)\b|(?<!\w)(?:λεγομαι|ονομα\w*)(?!\w)",
    "HAS_BIRTHDATE": r"\b(?:born|birthday|birth|birthdate)\b|(?<!\w)(?:γεννη\w*|γενεθλια)(?!\w)",
    "HAS_PROFESSION": r"\b(?:i'?m\s+an?|i\s+am\s+an?|work\w*\s+as|profession|job|career)\b",
}


def _words(text) -> set:
    return {w for w in re.findall(r"\w+", _fold(text)) if len(w) >= 3 and w not in _STOP}


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k >= 4 and a[k:] in ("", "s", "es", "ed", "ing", "ies", "er", "ers") and \
        b[k:] in ("", "s", "es", "ed", "ing", "ies", "er", "ers")


def _coverage(words: set, sentence: str) -> float:
    sw = _words(sentence)
    return sum(1 for w in words if any(_same(w, x) for x in sw)) / len(words) if words else 0.0


def owner_said(value, statements, minimum: float = 0.5, cue: str = None):
    """``"stated"`` when an affirmative owner clause carries at least
    ``minimum`` of ``value``'s distinctive words (and, with ``cue``, says the
    predicate's kind of thing), ``"negated"`` when only a NEGATED one does,
    else None."""
    words = _words(value)
    if not words:
        return None
    rx = re.compile(cue, re.IGNORECASE) if cue else None
    ok = [(s, neg) for s, neg in statements if rx is None or rx.search(s)]
    best_pos = max((_coverage(words, s) for s, neg in ok if not neg), default=0.0)
    if best_pos >= minimum:
        return "stated"
    best_neg = max((_coverage(words, s) for s, neg in statements if neg), default=0.0)
    return "negated" if best_neg >= minimum else None


def is_owner_end(node) -> bool:
    return _fold(node).strip() in ("user", "the user", "owner", "me", "i")


def strip_negation(value) -> str:
    """"not a doctor" → "doctor" (the value a correction removes)."""
    return re.sub(r"^\s*(?:not|no\s+longer|no|never|isn'?t|is\s+not|am\s+not|δεν\s+ειμαι)\s+(?:an?\s+)?", "",
                  _fold(value)).strip()


def is_agent_state_predicate(predicate) -> bool:
    return bool(set(str(predicate or "").upper().split("_")) & AGENT_STATE_TOKENS)
