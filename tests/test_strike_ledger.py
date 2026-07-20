"""StrikeLedger — the request-scoped loop-detection seam extracted from
core.agent. Bundles the failure/action signature dicts, the decay-freeze
flag + warned set, and the consecutive-clean-success counter the turn loop
used to track as five interacting locals. Behaviour must match the inlined
version exactly."""

from ghost_agent.core.strikes import (
    READWRITE_HARD_STOP,
    StrikeLedger,
    action_result_fingerprint,
    is_readwrite_loop_exempt,
    note_repeated_action,
    note_repeated_failure,
)

# the old private names must still import from core.agent (tests rely on it)
from ghost_agent.core.agent import (
    _note_repeated_failure,
    _note_repeated_action,
    _action_result_fingerprint,
)


def test_reexports_are_the_same_callables():
    assert _note_repeated_failure is note_repeated_failure
    assert _note_repeated_action is note_repeated_action
    assert _action_result_fingerprint is action_result_fingerprint


def test_note_failure_freezes_after_threshold():
    led = StrikeLedger()
    sig1 = led.note_failure("read", "'x' not found")
    sig2 = led.note_failure("read", "'x' not found")
    assert not led.decay_frozen  # only 2 occurrences
    sig3 = led.note_failure("read", "'x' not found")
    assert led.decay_frozen
    # is_first_warning fires exactly once for the signature
    assert sig3[3] is True
    sig4 = led.note_failure("read", "'x' not found")
    assert sig4[3] is False


def test_distinct_failures_tracked_separately():
    led = StrikeLedger()
    for _ in range(3):
        led.note_failure("read", "'a' not found")
    assert led.decay_frozen
    led2 = StrikeLedger()
    led2.note_failure("read", "'a' not found")
    led2.note_failure("read", "'b' not found")
    led2.note_failure("execute", "boom")
    assert not led2.decay_frozen  # no single signature hit 3


def test_clean_success_unfreezes_after_three():
    led = StrikeLedger()
    for _ in range(3):
        led.note_failure("read", "'x' not found")
    assert led.decay_frozen
    assert led.note_clean_success() is False  # 1
    assert led.note_clean_success() is False  # 2
    assert led.note_clean_success() is True  # 3 → unfreeze
    assert not led.decay_frozen


def test_failure_resets_clean_streak():
    led = StrikeLedger()
    for _ in range(3):
        led.note_failure("read", "'x' not found")
    led.note_clean_success()
    led.note_clean_success()
    # a structural failure breaks the streak (note_failure does not reset,
    # but reset_clean_streak does — the loop calls it on ANY failure)
    led.reset_clean_streak()
    led.note_failure("read", "'x' not found")
    assert led.consecutive_clean_successes == 0
    # need a fresh run of 3 to unfreeze again
    assert led.note_clean_success() is False
    assert led.note_clean_success() is False
    assert led.note_clean_success() is True


def test_transient_failure_breaks_streak_via_reset():
    led = StrikeLedger()
    for _ in range(3):
        led.note_failure("read", "'x' not found")
    led.note_clean_success()
    led.note_clean_success()
    led.reset_clean_streak()  # simulates a transient (non-structural) failure
    assert led.consecutive_clean_successes == 0
    assert not led.decay_frozen or led.decay_frozen  # still frozen
    assert led.decay_frozen


def test_recurring_failure_refreezes_after_unfreeze():
    led = StrikeLedger()
    for _ in range(3):
        led.note_failure("read", "'x' not found")
    for _ in range(3):
        led.note_clean_success()
    assert not led.decay_frozen
    # the SAME failure recurs once more: count is already ≥3, so it re-freezes
    led.reset_clean_streak()
    sig = led.note_failure("read", "'x' not found")
    assert sig[2] is True  # is_persistent
    assert led.decay_frozen


def test_fingerprint_covers_full_string_not_just_head():
    """Regression: the fingerprint hashed only the first 600 normalised
    chars, so long outputs with a stable header (a polling loop whose
    progress shows only in the tail) collided — a genuinely-progressing
    loop got steered at 2 and hard-aborted at 3 "identical" results."""
    header = "POLL STATUS: job queue report\n" + ("x" * 700)
    a = header + "\nstep=one pending"
    b = header + "\nstep=two running"
    assert action_result_fingerprint(a) != action_result_fingerprint(b)


def test_fingerprint_whitespace_normalisation_still_applies():
    # Same content modulo whitespace/case must still collide (that IS
    # the no-progress signal), including past the old 600-char slice.
    long_body = ("word " * 200).strip()
    assert action_result_fingerprint(f"{long_body}\n\nEND  marker") == \
        action_result_fingerprint(f"{long_body} end MARKER")


def test_fingerprint_empty_and_none_stable():
    assert action_result_fingerprint("") == action_result_fingerprint(None)


def test_readwrite_hard_stop_contract():
    """The two-tier contract lives in ONE place: steer at the general
    no-progress threshold (2), exempt read/write tools hard-stop at 5.
    agent.py imports this constant instead of restating the number."""
    assert READWRITE_HARD_STOP == 5
    assert READWRITE_HARD_STOP > 2  # must sit above the steer tier
    assert is_readwrite_loop_exempt("file_system")


def test_note_action_trips_on_repeat():
    led = StrikeLedger()
    fp = action_result_fingerprint("no change")
    assert led.note_action("browser", ".icon", fp)[2] is False
    assert led.note_action("browser", ".icon", fp)[2] is False
    assert led.note_action("browser", ".icon", fp)[2] is True


def test_note_action_distinct_targets_dont_trip():
    led = StrikeLedger()
    fp = action_result_fingerprint("ok")
    for n in range(5):
        tripped = led.note_action("browser", f".icon-{n}", fp)[2]
        assert tripped is False


# ── world-changed reset (2026-07-18) ─────────────────────────────────────────
# A successful file mutation invalidates accumulated no-progress
# observations: re-observing after an edit is verification, not thrash.
# Overnight log 2026-07-17: every fix-verify turn (requests 26/3B/72/1E/91)
# had its post-fix browser navigate killed by the 2x-repeat breaker, and the
# verifier then refuted the turn for missing post-fix evidence.

def test_note_world_changed_resets_action_counts():
    led = StrikeLedger()
    fp = action_result_fingerprint("<html>landing page</html>")
    led.note_action("browser", "http://127.0.0.1:8103/", fp, threshold=2)
    assert led.note_action("browser", "http://127.0.0.1:8103/", fp, threshold=2)[2] is True
    led.note_world_changed()
    sig, count, tripped = led.note_action(
        "browser", "http://127.0.0.1:8103/", fp, threshold=2)
    assert count == 1
    assert tripped is False


def test_note_world_changed_clears_all_observation_signatures():
    led = StrikeLedger()
    fp = action_result_fingerprint("same")
    led.note_action("browser", "http://a/", fp)
    led.note_action("file_system", "game.js", fp)
    led.note_world_changed()
    assert led.action_sigs == {}


def test_note_world_changed_leaves_failure_state_alone():
    led = StrikeLedger()
    for _ in range(3):
        led.note_failure("read", "'x' not found")
    assert led.decay_frozen
    led.note_world_changed()
    # failure-loop tracking is orthogonal — a file write must not unfreeze it
    assert led.decay_frozen
    assert led.failure_sigs
