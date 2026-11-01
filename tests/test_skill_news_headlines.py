"""The acquired skill `news_headlines` (2026-10-31 rewrite, operator: "look
into a custom skill that the agent made … and make it richer").

The skill is DATA (it lives under $GHOST_HOME/system/memory/acquired_skills/
and runs in the sandbox), so it is loaded from that path; set
GHOST_NEWS_SKILL to test another copy. Every test is offline: feeds are
fixtures, `fetch` is replaced.

World where each pin fails: the same story told by three outlets comes back
three times; a Greek outlet's report from Warsaw counts as Greek news;
'transport' is tagged sports; tracking parameters stay on the links; one
dead feed sinks the run; a dead network reads as a successful run; a wrong
topic/region/source (the model's mistake) fails the skill (three failures
retire it); routine name-day lists outrank the news; the caps return fewer
items than asked.
"""
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest

_PATH = Path(os.environ.get("GHOST_NEWS_SKILL") or
             Path.home() / "Data/AI/Data/system/memory/acquired_skills/news_headlines.py")
pytestmark = pytest.mark.skipif(not _PATH.exists(), reason="news_headlines skill not installed")


@pytest.fixture(scope="module")
def nh():
    spec = importlib.util.spec_from_file_location("news_headlines_skill", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NOW = datetime.now(timezone.utc)


def _rfc(hours_ago):
    return format_datetime(NOW - timedelta(hours=hours_ago))


def rss(items):
    body = "".join(
        f"<item><title>{t}</title><link>{l}</link><pubDate>{_rfc(h)}</pubDate>"
        f"<description><![CDATA[{d}]]></description>"
        + "".join(f"<category>{c}</category>" for c in cats) + "</item>"
        for t, l, h, d, cats in items)
    return f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>{body}</channel></rss>'.encode()


def item(title, link="https://x.gr/a", hours=1, desc="", cats=()):
    return (title, link, hours, desc, cats)


def install(monkeypatch, nh, feeds):
    """feeds: {source_key: bytes | Exception}; other sources fail."""
    def fake_fetch(key):
        v = feeds.get(key, OSError("not in this test"))
        if isinstance(v, Exception):
            return key, None, f"{type(v).__name__}: {v}"
        return key, nh.parse_feed(v), None
    monkeypatch.setattr(nh, "fetch", fake_fetch)


# --- parsing and cleaning ----------------------------------------------------

def test_rss2_rdf_and_atom_items_parse(nh):
    rdf = (b'<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
           b'xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">'
           b'<item><title>DW story</title><link>https://dw.com/a?maca=en-rss</link>'
           b'<dc:date>2026-10-31T05:00:00Z</dc:date><dc:subject>Politics</dc:subject></item></rdf:RDF>')
    atom = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom story</title>'
            b'<link rel="alternate" href="https://a.org/x"/><updated>2026-10-31T04:00:00+02:00</updated>'
            b'<summary>Short</summary><category term="Sport"/></entry></feed>')
    r = nh.parse_feed(rdf)[0]
    assert r["title"] == "DW story" and r["link"] == "https://dw.com/a" and r["categories"] == ["Politics"]
    assert r["date"] == datetime(2026, 10, 31, 5, tzinfo=timezone.utc)
    a = nh.parse_feed(b"\xef\xbb\xbf  " + atom)[0]          # a BOM and blanks before the XML
    assert a["link"] == "https://a.org/x" and a["categories"] == ["Sport"]
    assert a["date"] == datetime(2026, 10, 31, 2, tzinfo=timezone.utc)
    p = nh.parse_feed(rss([item("Τίτλος &amp; άλλα", cats=("Κόσμος",))]))[0]
    assert p["title"] == "Τίτλος & άλλα" and p["categories"] == ["Κόσμος"]


def test_summary_is_text_and_cut_at_a_word(nh):
    raw = ('<p>Στόχος κυβερνοεπίθεσης έγινε η εταιρεία &#8211; η είδηση &amp;#8230; έγινε γνωστή</p>'
           '<p>The post X appeared first on Y.</p>')
    t = nh.clean_text(raw)
    assert "<" not in t and "&" not in t and "appeared first" not in t
    assert t.startswith("Στόχος κυβερνοεπίθεσης έγινε η εταιρεία – η είδηση")
    cut = nh.clean_text("alpha beta gamma delta epsilon", 14)
    assert cut == "alpha beta…"


def test_tracking_parameters_leave_the_link(nh):
    assert nh.clean_link("https://n.gr/a/?utm_source=rss&utm_medium=rss&id=7#top") == "https://n.gr/a/?id=7"
    assert nh.clean_link("https://www.dw.com/en/x/a-1?maca=en-rss-en-all-1573-rdf") == "https://www.dw.com/en/x/a-1"


# --- topic and scope ----------------------------------------------------------

def _tagged(nh, title, link="https://x.gr/a", cats=(), region="greece"):
    it = {"title": title, "link": link, "categories": list(cats)}
    return nh.topic_of(it, region), nh.scope_of(it, region)


def test_topic_keys_match_at_the_start_of_a_word(nh):
    assert _tagged(nh, "Turkic states push transport links", cats=("Transport",), region="world")[0] == "world"
    assert _tagged(nh, "Βαθμολογία", cats=("ΣΠΟΡ",))[0] == "sports"
    assert _tagged(nh, "x", link="https://www.protothema.gr/sports/article/1/")[0] == "sports"
    assert _tagged(nh, "Χρηματιστήριο", cats=("Οικονομία",))[0] == "economy"


def test_scope_is_what_the_story_is_about(nh):
    assert _tagged(nh, "ΗΠΑ: Ρεκόρ πρόωρης ψήφου", cats=("Πολιτική",))[1] == "world"
    assert _tagged(nh, "Λαβρόφ: Οι δηλώσεις Ζελένσκι", cats=("Ειδήσεις",))[1] == "world"
    assert _tagged(nh, "Κάτι", link="https://www.iefimerida.gr/kosmos/x")[1] == "world"
    assert _tagged(nh, "Σεισμός 4,2 Ρίχτερ στη Χαλκιδική", cats=("Ελλάδα",))[1] == "greece"
    assert _tagged(nh, "Anything", region="world")[1] == "world"


# --- the same story from several outlets -------------------------------------

POLAND = [
    ("ertnews", "Πολωνία: Ο πρόεδρος Ναβρότσκι έσπευσε στο Συνταγματικό Δικαστήριο που βρίσκεται σε αστυνομικό κλοιό",
     "Στην Πολωνία η πολιτική κρίση γύρω από το Συνταγματικό Δικαστήριο κλιμακώθηκε, αστυνομικές δυνάμεις αναπτύχθηκαν"),
    ("tovima", "Πολωνία: Έφοδος της αστυνομίας στο Συνταγματικό Δικαστήριο – Καταγγελίες για «πραξικόπημα»",
     "Στα άκρα οδηγείται η κρίση στην Πολωνία μετά την έφοδο της αστυνομίας στο Συνταγματικό Δικαστήριο"),
    ("naftemporiki", "Κρίση στην Πολωνία με αιχμή την Δικαιοσύνη – Αστυνομία γύρω από το συνταγματικό δικαστήριο",
     "Ένταση στη Βαρσοβία, όπου αστυνομικές δυνάμεις αναπτύχθηκαν γύρω από το Συνταγματικό Δικαστήριο"),
]
OTHERS = [
    ("skai", "Σεισμός 4,2 Ρίχτερ στη Χαλκιδική τα ξημερώματα", "Σεισμική δόνηση σημειώθηκε στη Σιθωνία"),
    ("in_gr", "Παρατείνεται η επιδότηση καυσίμων έως τις 15 Νοεμβρίου", "Τι ισχύει για το ντίζελ"),
    ("cnn_gr", "Άμεση κατάπαυση του πυρός στη Μαύρη Θάλασσα ζητά η Πολωνία", "Ο Πολωνός ΥΠΕΞ ζήτησε εκεχειρία για τα σιτηρά"),
]


def _word(i, salt):
    # a word no other fixture title shares, even in its first five letters
    a = "abcdefghijklmnopqrstuvwxyz"
    return "zq" + a[i % 26] + a[(i // 26) % 26] + salt


def unique_title(i):
    return " ".join(_word(i, s) for s in ("ka", "lo", "mi", "nu"))


def _feeds(rows, cats=("Κόσμος",), backdrop=40):
    """rows plus `backdrop` unrelated stories spread over three outlets: a
    real feed pool has hundreds of items, and the word weights (inverse
    document frequency) only mean something over a pool that size."""
    by = {}
    for n, (src, title, desc) in enumerate(rows):
        by.setdefault(src, []).append(item(title, f"https://{src}.gr/{n}", 1 + n, desc, cats))
    for i in range(backdrop):
        src = ("ot_gr", "efsyn", "iefimerida")[i % 3]
        by.setdefault(src, []).append(item(unique_title(i), f"https://{src}.gr/b{i}", 20, "", ()))
    return {k: rss(v) for k, v in by.items()}


def test_three_wordings_of_one_story_merge_and_others_stay_apart(nh, monkeypatch):
    install(monkeypatch, nh, _feeds(POLAND + OTHERS))
    out = nh.news_headlines(count=10, region="all")
    titles = [h["title"] for h in out["headlines"]]
    poland = [h for h in out["headlines"] if "Συνταγματικό" in h["title"] or "συνταγματικό" in h["title"]]
    assert len(poland) == 1, titles
    assert sorted(poland[0]["also_reported_by"]) == sorted(
        {"ERT News", "To Vima", "Naftemporiki"} - {poland[0]["source"]})
    assert out["headlines"][0] is poland[0]                       # most outlets → first
    assert any("Μαύρη Θάλασσα" in t for t in titles)              # shares only 'Πολωνία': its own story
    assert len(out["headlines"]) == 10                            # the rest is backdrop


def test_a_query_keeps_the_whole_story(nh, monkeypatch):
    install(monkeypatch, nh, _feeds(POLAND + OTHERS))
    out = nh.news_headlines(query="πολωνια", count=5)             # accents and case folded
    assert len(out["headlines"]) == 2
    assert len(out["headlines"][0]["also_reported_by"]) == 2


def test_routine_items_go_to_the_back(nh, monkeypatch):
    rows = [(s, "Εορτολόγιο: Ποιοι γιορτάζουν σήμερα 31 Οκτωβρίου", "Σήμερα τιμάται η μνήμη")
            for s in ("ertnews", "naftemporiki", "in_gr")]
    rows.append(("skai", "Σεισμός 4,2 Ρίχτερ στη Χαλκιδική", "Σεισμική δόνηση"))
    install(monkeypatch, nh, _feeds(rows, cats=("Ελλάδα",)))
    out = nh.news_headlines(count=1)
    assert out["headlines"][0]["title"].startswith("Σεισμός")


# --- failures and the model's mistakes ---------------------------------------

def test_one_dead_feed_is_reported_and_the_rest_answer(nh, monkeypatch):
    feeds = _feeds(OTHERS)
    feeds["bbc"] = OSError("HTTP Error 503")
    install(monkeypatch, nh, feeds)
    out = nh.news_headlines(count=5)
    assert out["headlines"] and "BBC World" in out["sources_failed"]
    assert "503" in out["sources_failed"]["BBC World"]


def test_every_feed_dead_is_a_failure(nh, monkeypatch):
    install(monkeypatch, nh, {})
    with pytest.raises(SystemExit) as e:
        nh.news_headlines()
    assert "every feed failed" in str(e.value)


def test_wrong_arguments_are_notes_not_failures(nh, monkeypatch):
    install(monkeypatch, nh, _feeds(OTHERS))
    out = nh.news_headlines(count="lots", region="mars", topic="weather", sources=["nope"], hours="x")
    notes = " ".join(out["notes"])
    assert "mars" in notes and "weather" in notes and "nope" in notes
    assert out["headlines"] and out["window_hours"] == 24


def test_old_items_leave_the_window_and_an_empty_result_says_why(nh, monkeypatch):
    install(monkeypatch, nh, {"skai": rss([item("Παλιό θέμα", hours=50)])})
    out = nh.news_headlines(hours=24)
    assert out["headlines"] == [] and "no item matched" in out["notes"][0]
    assert nh.news_headlines(hours=72)["headlines"][0]["age"] == "2 d ago"


def test_caps_prefer_variety_but_never_return_fewer_than_asked(nh, monkeypatch):
    rows = [("bbc", unique_title(100 + i), "") for i in range(6)]
    install(monkeypatch, nh, _feeds(rows, cats=()))
    out = nh.news_headlines(count=5, sources=["bbc"])
    assert len(out["headlines"]) == 5


def test_greek_quota_mixes_the_default_briefing(nh, monkeypatch):
    rows = [(s, unique_title(200 + i), "") for i, s in
            enumerate(["skai", "in_gr", "ertnews", "tovima", "cnn_gr", "skai"])]
    rows += [(s, unique_title(300 + i), "") for i, s in enumerate(["bbc", "guardian", "dw", "npr"])]
    install(monkeypatch, nh, _feeds(rows, cats=(), backdrop=0))
    out = nh.news_headlines(count=6)
    scopes = [h["scope"] for h in out["headlines"]]
    assert scopes.count("greece") == 4 and scopes.count("world") == 2, scopes
    assert all(h["scope"] == "greece" for h in nh.news_headlines(count=6, region="greece")["headlines"])


def test_cli_reads_json_and_falls_back_to_defaults(tmp_path):
    # run the file as the sandbox does, with the network made to fail fast:
    # the exit code must be non-zero (every feed failed), never a crash on argv
    env = dict(os.environ, http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9")
    for argv in (['{"count": 2}'], ["not json"], []):
        r = subprocess.run([sys.executable, str(_PATH), *argv], capture_output=True, text=True,
                           env=env, timeout=120)
        assert r.returncode != 0 and "every feed failed" in r.stderr, (argv, r.stderr[-300:])
