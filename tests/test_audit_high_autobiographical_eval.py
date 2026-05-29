"""Regression tests for the High-tier autobiographical + eval fixes.

* search_my_past scored on substring containment (`t in haystack_str`)
  while df was token-based — "art" matched "smart"/"part" and unseen
  tokens got max idf. Now scoring is token-set membership.
* _bump_template_rollup released the lock between read and tmp.replace
  (lost updates under concurrency); record_reference did the whole
  load→bump→persist with no lock at all.
* eval _discover_first matched "ls" as a bare substring (fired inside
  "calls"/"false"); now word-boundary.
"""

import json
import threading

from ghost_agent.selfhood.autobiographical import (
    AutobiographicalMemory,
    Experience,
)
from ghost_agent.eval import load_post_learning_suite


# -----------------------------------------------------------------
# search_my_past — token membership, not substring
# -----------------------------------------------------------------

def test_search_my_past_matches_token_not_substring(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I built a smart cache layer for the service.",
                          user_first_words="cache work"))
    mem.append(Experience(summary="I studied medieval art and sculpture.",
                          user_first_words="art museum"))

    hits = mem.search_my_past("art", limit=5)
    summaries = [h.summary for h in hits]
    assert any("medieval art" in s for s in summaries), "real 'art' memory should match"
    assert not any("smart cache" in s for s in summaries), \
        "must NOT match 'art' inside 'smart'"


def test_search_my_past_still_matches_real_tokens(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I parsed nginx access logs.", user_first_words="parse logs"))
    mem.append(Experience(summary="I baked a cake.", user_first_words="cake"))
    hits = mem.search_my_past("nginx logs", limit=5)
    assert hits and "nginx" in hits[0].summary


# -----------------------------------------------------------------
# record_reference — no lost updates under concurrency (locked RMW)
# -----------------------------------------------------------------

def test_record_reference_no_lost_updates_under_concurrency(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    n_threads, per = 8, 50

    def worker():
        for _ in range(per):
            mem.record_reference("exp-1")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mem.reference_count("exp-1") == n_threads * per


# -----------------------------------------------------------------
# _bump_template_rollup — correctness preserved under the single lock
# -----------------------------------------------------------------

def test_template_rollup_increments_and_preserves_prior_lines(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    # A normal (non-template) experience: must survive the rollup rewrites.
    mem.append(Experience(summary="I solved a real task.", user_first_words="real task"))
    # Template experiences (same banner) collapse into ONE rollup record.
    for _ in range(5):
        mem.append(Experience(
            summary="synthetic.",
            user_first_words="### SYNTHETIC TRAINING EXERCISE: do the thing",
        ))

    records = [json.loads(line) for line in mem.path.read_text().splitlines() if line.strip()]
    rollups = [r for r in records if r.get("template_marker")]
    assert len(rollups) == 1
    assert rollups[0]["template_count"] == 5
    # The earlier real line survived all the in-place rewrites.
    assert any("real task" in (r.get("user_first_words") or "") for r in records)


def test_template_rollup_concurrent_bumps_not_lost(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    banner = "### SYNTHETIC TRAINING EXERCISE: concurrent"
    n_threads, per = 6, 20

    def worker():
        for _ in range(per):
            mem.append(Experience(summary="synthetic.", user_first_words=banner))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = [json.loads(line) for line in mem.path.read_text().splitlines() if line.strip()]
    rollups = [r for r in records if r.get("template_marker")]
    total = sum(r.get("template_count", 0) for r in rollups)
    # No bump may be lost: total recorded == total appended.
    assert total == n_threads * per


# -----------------------------------------------------------------
# eval _discover_first — word-boundary, not substring
# -----------------------------------------------------------------

def test_discover_validator_does_not_match_ls_inside_words():
    task = load_post_learning_suite()[0]
    # "calls" and "false" contain the substring "ls" but show NO discovery.
    ok, reason = task.validate("The function calls return false on error.", None)
    assert not ok, f"'ls' must not match inside 'calls'/'false' (reason={reason!r})"

    ok2, _ = task.validate("Also, I will open the tools and resume.", None)
    assert not ok2, "'als'/'tools' contain 'ls' but are not discovery"


def test_discover_validator_accepts_real_ls_and_list():
    task = load_post_learning_suite()[0]
    ok, _ = task.validate("Let me run `ls -la` to list the directory first.", None)
    assert ok
