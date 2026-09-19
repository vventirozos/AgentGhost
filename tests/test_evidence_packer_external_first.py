"""The judge must be shown the source, not the deliverable's echo (2026-09-15, §4HC).

Live (req c16679f1) the verifier CONFIRMED at 0.95 a report whose headline
lead — a PEC address presented as the attacker's sender — was contradicted
by the browser extraction it came from: "That is Revolut receiving and
answering. The sender is still unnamed." The packer never showed the judge
that sentence: its candidate window was 10 tools deep (the extraction was
11th), its positional slots were three write receipts of the report itself,
and its claim-relevance pull ranked by raw overlap, which the report's own
file_system echo (89 tokens) wins over the source (64) every time. Replayed
on the recorded turn, the digest now carries the sentence.

Fixture shape mirrors that turn: many self-authored write receipts that
echo the claim, near-duplicate OCRs of one card, and one external source
whose distinctive tokens the claim borrowed — plus the disqualifier.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import agent as A

_CLAIM = ("Best-supported sender lead: revolut@legalmail.it on InfoCert's PEC "
          "network, the Milan branch address. Exposed data: passport copies, "
          "selfies, IBAN, full transaction history including Bitcoin. Report "
          "written and PDF generated.")

_CARD = ("What happened? Revolut received a request for customer information "
         "that appeared to come from a legitimate government agency. Identity "
         "details: full name, date of birth. Document and verification data: a "
         "copy of your identity document (passport) and facial verification "
         "image (selfie). Financial data: account statements (IBAN), withdrawal "
         "records and full transaction history (including Bitcoin).")

_SOURCE = ("Revolut fake official-request incident — what is actually out there. "
           "Screenshots show the files moving through Revolut's own Italian legal "
           "inbox: revolut@legalmail.it. The Revolut in that address is their name "
           "on Italy's certified-mail network (InfoCert). They publish it as the "
           "PEC of the Milan branch. That is Revolut receiving and answering. The "
           "sender is still unnamed. La Stampa says another address may have been "
           "Guardia di Finanza.")

_ECHO = ("SUCCESS: Applied 1 SEARCH block to report.md. --- POST-EDIT VIEW --- "
         "## 3. Email domain BEST-SUPPORTED LEAD: revolut@legalmail.it on "
         "InfoCert's PEC network, the Milan branch address. Exposed data: passport "
         "copies, selfies, IBAN, full transaction history including Bitcoin.")


def _t(name, content):
    return {"name": name, "content": content}


def _live_shape():
    """Oldest → newest, as tools_run_this_turn is ordered."""
    tools = [_t("vision_analysis", _CARD), _t("vision_analysis", _CARD + " Header: ZachXBT")]
    tools += [_t("web_search", f"### {i}. Revolut breach coverage {i} government request")
              for i in range(8)]
    tools.append(_t("browser", _SOURCE))
    tools += [_t("web_search", f"### {i}. Revolut Italy PEC rumour {i} certified email")
              for i in range(6)]
    tools += [_t("file_system", _ECHO), _t("file_system", "--- report.md CONTENTS --- " + _ECHO),
              _t("file_system", _ECHO), _t("report_pdf", "SUCCESS: PDF report generated (7 pages)")]
    return tools


def test_the_live_shape_now_shows_the_judge_the_disqualifier():
    """FAILS IF: any one of the four changes is missing — each was necessary
    on the recorded turn (window, external-first, IDF, dedup)."""
    ev = A._collect_verifier_evidence(_live_shape(), claim_text=_CLAIM)
    assert "The sender is still unnamed" in ev
    assert "[browser]" in ev


def test_a_self_echo_never_out_ranks_an_external_source_for_the_pull():
    """FAILS IF: ranking is by overlap alone. The echo shares MORE tokens
    with the claim than the source does — by construction here, as live."""
    tools = [_t("browser", _SOURCE), _t("web_search", "unrelated"), _t("file_system", _ECHO),
             _t("execute", "x"), _t("file_system", "SUCCESS wrote"), _t("report_pdf", "SUCCESS pdf")]
    ev = A._collect_verifier_evidence(tools, claim_text=_CLAIM)
    assert "[browser]" in ev
    assert "still unnamed" in ev


def test_positional_self_slots_yield_to_externals_but_the_newest_stays():
    """FAILS IF: the newest item is displaced (failure attribution needs it)
    or the self-authored middle slots are kept over an overlapping source."""
    # The newest item shares NO token with the claim, so if the yield ever
    # displaced it the claim pull could not bring it back (the battery's
    # fourth survivor passed because "PDF report generated" overlapped).
    tools = [_t("browser", _SOURCE), _t("file_system", _ECHO), _t("file_system", _ECHO),
             _t("report_pdf", "ok")]
    ev = A._collect_verifier_evidence(tools, claim_text=_CLAIM)
    assert "[report_pdf]" in ev            # newest kept
    assert "[browser]" in ev               # a self slot yielded


def test_the_pull_prefers_a_third_external_over_a_higher_scoring_echo():
    """FAILS IF: the 4th-slot pull ranks by score alone. The yield consumes
    the top two externals; the pull must still take the THIRD over the
    self echo that out-scores it (the battery's fifth survivor)."""
    ext1 = _t("browser", _SOURCE)
    ext2 = _t("web_search", "legalmail InfoCert Milan PEC branch certified network")
    ext3 = _t("web_search", "InfoCert Milan PEC address selfies passport IBAN")
    tools = [ext1, ext2, ext3, _t("file_system", _ECHO), _t("file_system", "SUCCESS wrote a.md"),
             _t("file_system", "SUCCESS wrote b.md"), _t("report_pdf", "ok")]
    ev = A._collect_verifier_evidence(tools, claim_text=_CLAIM)
    assert ev.count("[web_search]") == 2, ev
    assert "[browser]" in ev
    assert "POST-EDIT VIEW" not in ev


def test_no_external_source_keeps_todays_behaviour():
    """FAILS IF: the ranking change breaks turns that have nothing external —
    the write receipts are then the only evidence, and they must pack."""
    tools = [_t("file_system", "wrote a.md"), _t("file_system", "wrote b.md"),
             _t("file_system", _ECHO)]
    ev = A._collect_verifier_evidence(tools, claim_text=_CLAIM)
    assert ev.count("[file_system]") == 3


def test_two_ocrs_of_one_card_do_not_take_two_slots():
    """FAILS IF: near-duplicate suppression is gone. Two OCRs of one image
    sat at Jaccard 0.69/0.60 live; every different pair under 0.30."""
    tools = [_t("browser", _SOURCE), _t("vision_analysis", _CARD),
             _t("vision_analysis", _CARD + " Header: ZachXBT"),
             _t("file_system", _ECHO), _t("file_system", _ECHO), _t("report_pdf", "SUCCESS pdf")]
    ev = A._collect_verifier_evidence(tools, claim_text=_CLAIM)
    assert ev.count("[vision_analysis]") == 1
    assert "[browser]" in ev


def test_rare_tokens_outweigh_common_ones():
    """FAILS IF: IDF weighting is gone — a source sharing the claim's
    DISTINCTIVE tokens must beat one sharing only the words every output has."""
    ct = A._claim_tokens(_CLAIM)
    generic = _t("web_search", "Revolut breach passport selfie IBAN Bitcoin transaction history report")
    distinctive = _t("web_search", "legalmail InfoCert Milan PEC branch")
    filler = [_t("web_search", "Revolut breach passport selfie IBAN Bitcoin") for _ in range(6)]
    score, raw, _ = A._claim_overlap_scorer(ct, filler + [generic, distinctive])
    assert raw(generic) > raw(distinctive), "control: generic wins on RAW overlap"
    assert score(distinctive) > score(generic), "…but loses on rarity-weighted overlap"


def test_the_window_reaches_past_ten_tools():
    """FAILS IF: the deep window is 10 again — the live source was 11th newest."""
    tools = [_t("browser", _SOURCE)] + [_t("web_search", f"filler {i}") for i in range(12)] \
        + [_t("file_system", _ECHO)]
    ev = A._collect_verifier_evidence(tools, claim_text=_CLAIM)
    assert "[browser]" in ev


# ── §4IT: Unicode / year claim tokens, and a second pull by marginal coverage ──

def test_claim_tokens_read_greek_words_and_bare_years():
    toks = A._claim_tokens("Το 2009 το Ίδρυμα μετακινήθηκε στη συμβολή των οδών Μεραρχίας. PostgreSQL 19 released in 2026.")
    assert {"2009", "ίδρυμα", "μετακινήθηκε", "μεραρχίας", "postgresql", "2026"} <= toks
    assert not ({"και", "του", "της", "στη", "των"} & toks)                     # Greek function words are stopwords


def test_a_greek_claim_pulls_the_greek_source_and_a_second_pull_covers_the_rest():
    """The retry (probe-012ca9ec): a 26-tool Greek research turn packed
    "6 items of 36" and no pull — the tokenizer saw no Greek word and the
    six positional slots filled the budget table. FAILS IF: Greek words are
    not claim tokens, or only one item can be pulled, or the table has no
    room past the positional maximum."""
    claim = ("Το κτίριο στην Αλκιβιάδου 154 ανήκει στο Ίδρυμα Οικονομίδου. Το 2009 το Ίδρυμα μετακινήθηκε στη "
             "συμβολή των οδών Μεραρχίας 36 και Ακτής Μουτσοπούλου. Η ΧΡΩΠΕΙ ιδρύθηκε το 1883 από τον Σπήλιο Οικονομίδη στο Λειβάρτζι.")
    old_a = _t("web_search", "Ίδρυμα Οικονομίδου: το νέο κτίριο βρίσκεται από το 2009 στη συμβολή των οδών Μεραρχίας 36 & Ακτής Μουτσοπούλου")
    old_b = _t("web_search", "Η ΧΡΩΠΕΙ ιδρύθηκε το 1883 από τον χημικό Σπήλιο Οικονομίδη από το Λειβάρτζι Καλαβρύτων")
    fillers = [_t("web_search", f"Πειραιάς λιμάνι Αττική αποτέλεσμα αναζήτησης νούμερο {i} χωρίς σχέση") for i in range(20)]
    tools = [old_a, old_b] + fillers                                    # oldest first; the six newest are fillers
    budget, items = A._evidence_budget_for(tools)
    ev = A._collect_verifier_evidence(tools, max_items=items, budget=budget, claim_text=claim)
    assert "Μεραρχίας 36" in ev and "Λειβάρτζι" in ev                    # both older sources pulled
    assert ev.count("[web_search]") == items + 2



def test_a_refused_page_never_takes_a_pull_or_swap_slot_while_a_live_source_exists():
    """Review §4IX: the claim pull and the external swap iterated every
    candidate, and a `STATUS: BLOCKED` page whose title carried the claim's
    tokens was pulled as external evidence (§4HO said never)."""
    from ghost_agent.core.agent import _collect_verifier_evidence
    tools = [{"name": "web_search", "content": "### 1. Revolut breach — the spoofed gov.example domain was used by the sender " * 3},
             {"name": "browser", "content": "STATUS: BLOCKED — the site refused the fetch. Title: Revolut breach spoofed domain sender gov.example legalmail InfoCert Milan"},
             {"name": "web_search", "content": "### 2. TechCrunch: legalmail InfoCert Milan — the sender domain " * 3},
             {"name": "file_system", "content": "wrote report.md"},
             {"name": "browser", "content": "Page: Revolut statement — the request came from legalmail InfoCert Milan; sender unnamed " * 2}]
    ev = _collect_verifier_evidence(tools, max_items=3, budget=4000,
                                    claim_text="The sender used legalmail InfoCert Milan and a spoofed gov.example domain; Revolut breach confirmed.")
    assert "STATUS: BLOCKED" not in ev
    # with nothing live, the dead page is still better than nothing
    ev2 = _collect_verifier_evidence(tools[1:2], max_items=3, budget=4000, claim_text="legalmail InfoCert Milan")
    assert "STATUS: BLOCKED" in ev2


def test_4iy_reads_are_sources_receipts_yield_and_the_figure_line_survives_the_window():
    from ghost_agent.core.agent import _collect_verifier_evidence, _slice_evidence_body
    tools = [{"name": "web_search", "content": "### 1. pool sizing guide — connection pools and timeouts " * 4}, {"name": "web_search", "content": "### 2. db.internal hosts guide " * 4},
             {"name": "file_system", "content": "# config.py\npool_size = 20\ntimeout = 30\nhost = 'db.internal'"}, {"name": "file_system", "content": "# db.yaml\nreplicas: 3\nregion: eu-west\nbackup: nightly"},
             {"name": "file_system", "content": "SUCCESS: Applied 1 SEARCH block to report.md"}]
    ev = _collect_verifier_evidence(tools, max_items=3, budget=4000, claim_text="config.py sets pool_size 20 and timeout 30 against db.internal; db.yaml runs 3 replicas in eu-west with nightly backup.")
    assert "pool_size = 20" in ev and "replicas: 3" in ev
    body = ("Revolut Group Holdings Ltd — annual report 2024 overview: revenue, customers, markets, products, compliance, regulation, banking licence " * 12
            + "\n\nGroup revenue for the year: £3.1bn, up 72%.\n" + "Other notes about people and offices and events. " * 30)
    assert "£3.1bn" in _slice_evidence_body(body, 1900, "Revolut's 2024 revenue was £3.1bn, up 72%.")


def test_4iy_a_read_yields_only_to_an_external_that_out_scores_it_and_numbers_pick_the_window():
    from ghost_agent.core.agent import _collect_verifier_evidence, _slice_evidence_body
    claim = "config.py sets pool_size 20 and timeout 30 against db.internal; the replicas run in eu-west with nightly backup."
    tools = [{"name": "web_search", "content": "### a general guide to nightly backup schedules " * 4},   # shares two claim words, out-scores nothing specific
             {"name": "file_system", "content": "# config.py\npool_size = 20\ntimeout = 30\nhost = 'db.internal'\nreplicas eu-west nightly backup"},
             {"name": "file_system", "content": "SUCCESS: Applied 1 SEARCH block to report.md"}]
    ev = _collect_verifier_evidence(tools, max_items=2, budget=4000, claim_text=claim)     # two slots: the receipt and the read; the guide waits outside
    assert "pool_size = 20" in ev                       # the read out-scores the guide; it keeps its slot
    body = ("Overview: the total of the group and its markets and products and services and people and offices " * 20
            + "\n\nFigure: £3.1bn.\n" + "Other notes about weather and events and travel and food. " * 30)
    assert "£3.1bn" in _slice_evidence_body(body, 1500, "The total came to £3.1bn.")     # the figure line carries no claim WORD; the header carries "total"
