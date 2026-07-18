"""Generation-time challenge diversity guard (2026-07-18).

The exact-hash dedup in FrontierTracker only refuses to RECORD a
byte-identical prompt. Overnight 2026-07-17/18, 4 of 6 LLM-generated
challenges were the same "transaction_log.csv fraud scan" task in fresh
wording — every one accepted, solved on attempt 1, recorded. The tracker now
keeps the head text of recently generated challenges so the dreamer can
reject a reworded near-duplicate before a solver attempt is spent on it.
"""

from ghost_agent.memory.frontier import FrontierTracker


FRAUD_A = (
    "You are given a CSV file named `transaction_log.csv` containing records "
    "of financial transactions with columns user, timestamp, amount, "
    "txn_type. Identify potentially fraudulent transactions where the amount "
    "exceeds three standard deviations and print each flagged row."
)

# Same theme, reworded — the overnight failure shape.
FRAUD_B = (
    "A CSV file called `transaction_log.csv` holds financial transaction "
    "records (columns: user, timestamp, amount, txn_type). Find the "
    "fraudulent transactions whose amount is more than three standard "
    "deviations above the mean and print the flagged rows."
)

INVENTORY = (
    "You are given a JSON file named `warehouse_inventory.json` describing "
    "stock levels per product. Compute the restock list: every product whose "
    "quantity is below its reorder threshold, sorted by shortfall."
)


def _tracker(tmp_path):
    return FrontierTracker(memory_dir=tmp_path)


def test_window_starts_empty(tmp_path):
    t = _tracker(tmp_path)
    assert t.recent_generated_challenges() == []
    assert t.most_similar_recent_challenge(FRAUD_A) == (0.0, "")


def test_reworded_duplicate_scores_high(tmp_path):
    t = _tracker(tmp_path)
    t.note_generated_challenge(FRAUD_A)
    sim, head = t.most_similar_recent_challenge(FRAUD_B)
    assert sim >= 0.60
    assert "transaction_log.csv" in head


def test_different_theme_scores_low(tmp_path):
    t = _tracker(tmp_path)
    t.note_generated_challenge(FRAUD_A)
    sim, _ = t.most_similar_recent_challenge(INVENTORY)
    assert sim < 0.60


def test_window_is_capped_and_fifo(tmp_path):
    t = _tracker(tmp_path)
    for i in range(FrontierTracker.RECENT_CHALLENGE_KEEP + 4):
        t.note_generated_challenge(f"challenge number {i} about topic_{i}.csv analysis")
    recent = t.recent_generated_challenges(limit=100)
    assert len(recent) == FrontierTracker.RECENT_CHALLENGE_KEEP
    # oldest entries evicted, newest kept (newest last)
    assert f"topic_{FrontierTracker.RECENT_CHALLENGE_KEEP + 3}" in recent[-1]
    assert all("topic_0" not in h and "topic_1 " not in h for h in recent)


def test_recent_accessor_respects_limit_newest_last(tmp_path):
    t = _tracker(tmp_path)
    t.note_generated_challenge(FRAUD_A)
    t.note_generated_challenge(INVENTORY)
    recent = t.recent_generated_challenges(limit=1)
    assert len(recent) == 1
    assert "warehouse_inventory" in recent[0]


def test_empty_and_blank_challenges_ignored(tmp_path):
    t = _tracker(tmp_path)
    t.note_generated_challenge("")
    t.note_generated_challenge("   ")
    assert t.recent_generated_challenges() == []


def test_head_truncated_to_bound(tmp_path):
    t = _tracker(tmp_path)
    t.note_generated_challenge("x" * 5000)
    (head,) = t.recent_generated_challenges()
    assert len(head) == FrontierTracker.RECENT_CHALLENGE_HEAD
