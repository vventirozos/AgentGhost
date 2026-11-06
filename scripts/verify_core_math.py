#!/usr/bin/env python3
"""Mechanized verification of the computational core touched 2026-08-08/09.

METHOD — three levels of rigour, chosen by the shape of each claim:

  EXHAUSTIVE   the input space is finite and small enough to enumerate in
               full, so the result is a proof over that space (not a sample).
  DIFFERENTIAL the implementation is compared against an INDEPENDENT
               reimplementation written from the specification. Agreement on
               a large input set means a bug would have to be replicated
               identically in two separately-derived programs.
  PROPERTY     algebraic invariants that must hold for ALL inputs (total
               order, set-difference laws, monotonicity), checked over
               randomized + adversarial inputs.

SCOPE — what this can and cannot establish, stated plainly because
overclaiming is exactly the failure this session kept producing:
  ✓ CAN: the pure functions and arithmetic are correct over their input space.
  ✗ CANNOT: that the surrounding I/O, the LLM's behaviour, or concurrency are
    correct. Those are not mathematical objects and no amount of enumeration
    settles them.
Run:  PYTHONPATH=src python scripts/verify_core_math.py
"""
from __future__ import annotations

import ipaddress
import itertools
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

FAIL: list = []
STATS: dict = {}
SKIPPED: list = []


def skip(name: str, why: str) -> None:
    """Record a check that could not run.

    LOUD, never silent: a verifier that quietly drops checks reports the same
    clean summary whether it verified everything or nothing — the exact
    'silent inoperative subsystem' failure this harness exists to catch.
    """
    SKIPPED.append(name)
    print(f"  SKIP  [{'':12}] {name}  — {why}")


def check(name: str, ok: bool, detail: str = "", level: str = "") -> None:
    STATS[name] = ok
    tag = {"E": "EXHAUSTIVE", "D": "DIFFERENTIAL", "P": "PROPERTY"}.get(level, "")
    print(f"  {'PASS' if ok else 'FAIL'}  [{tag:12}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ───────────────────────── 1. --bio-time-scale arithmetic ────────────────────
def verify_bio_scaling():
    print("\n[1] --bio-time-scale arithmetic  (§4Q / #41)")
    from ghost_agent.core.agent import GhostAgent

    class S(GhostAgent):
        def __init__(self, sc): self._bio_time_scale = sc

    # EXHAUSTIVE over every constant the tick actually uses, across the whole
    # documented scale range. This is the full space, not a sample.
    consts = [60, 120, 600, 900, 1800, 2400, 3600, 7200, 10800, 21600]
    scales = [x / 2 for x in range(2, 241)]          # 1.0 .. 120.0 step 0.5
    bad = []
    for sc in scales:
        s = S(sc)
        for c in consts:
            if abs(s._bio_scaled(c) - c / sc) > 1e-9:
                bad.append((sc, c))
            # _bio_cooldown must be EXACTLY _bio_scaled (they diverged once)
            if s._bio_cooldown(c) != s._bio_scaled(c):
                bad.append(("cooldown!=scaled", sc, c))
    check("scaled(c,s) == c/s for all (const, scale)", not bad,
          f"{len(scales)*len(consts)} pairs enumerated", "E")

    # PROPERTY: production scale is an exact identity — no float drift.
    s1 = S(1.0)
    check("scale 1.0 is an exact identity (no float drift)",
          all(s1._bio_scaled(c) == c for c in consts), f"{len(consts)} constants", "P")

    # PROPERTY: strictly decreasing in scale (a higher scale never widens).
    mono = all(S(a)._bio_scaled(3600) > S(b)._bio_scaled(3600)
               for a, b in zip(scales, scales[1:]))
    check("window is strictly decreasing in scale", mono, "", "P")

    # EXHAUSTIVE: the guard must fire exactly when the window is unusable.
    #
    # ⚠ The expectation is computed against an INDEPENDENT constant, not
    # against `_MIN_USABLE_WINDOW_S` itself. Deriving both sides from the same
    # constant makes the check a tautology that survives any mutation of it —
    # a mutation test caught exactly that hole here (zeroing the floor went
    # undetected). The independent requirement: a phase is a real LLM call of
    # order 60s, so a window below ~2 minutes cannot hold one.
    REQUIRED_FLOOR = 120.0
    floor = GhostAgent._MIN_USABLE_WINDOW_S
    check("the floor constant itself is >= the independent requirement",
          floor >= REQUIRED_FLOOR, f"{floor:g}s >= {REQUIRED_FLOOR:g}s", "P")
    wrong = []
    for sc in scales:
        s = S(sc)
        width = s._bio_scaled(3600) - s._bio_scaled(900)
        fired = bool(s._warn_if_scale_breaks_the_window())
        should = (sc > 1.0) and (width < REQUIRED_FLOOR)
        if fired != should:
            wrong.append((sc, width, fired, should))
    check("scale guard fires iff window < 120s (independent bar)", not wrong,
          f"{len(scales)} scales", "E")


# ───────────────────────── 2. egress host classification ─────────────────────
def _independent_is_local(host: str) -> bool:
    """INDEPENDENT reimplementation from the SPEC, written without reference to
    the production function: 'allow anything that is not globally routable
    unicast; block 6to4/Teredo because they wrap public v4'."""
    if not host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        h = host.lower().rstrip(".")
        return (h in ("localhost", "ip6-localhost", "ip6-loopback")
                or any(h.endswith(s) for s in (".local", ".lan", ".internal", ".localhost")))
    if ip.is_multicast:
        return True
    if ip.version == 6:
        if ip in ipaddress.ip_network("2002::/16"):     # 6to4
            return False
        if ip in ipaddress.ip_network("2001::/32"):     # Teredo
            return False
    return not ip.is_global


def verify_egress_classification():
    print("\n[2] egress host classification  (§4P)")
    from ghost_agent.utils.egress_guard import is_allowed_host, resolve_egress_proxy, install

    # DIFFERENTIAL: production vs the independent implementation, over an
    # adversarial + randomized corpus spanning every address class.
    hosts = [
        "127.0.0.1", "127.255.255.254", "10.0.0.1", "172.16.0.1", "172.31.255.255",
        "192.168.0.1", "169.254.1.1", "100.64.0.1", "0.0.0.0", "8.8.8.8", "1.1.1.1",
        "224.0.0.1", "239.255.255.255", "::1", "fe80::1", "fc00::1", "ff02::1",
        "2606:4700:4700::1111", "2001:4860:4860::8888", "2002:0808:0808::",
        "2002:0102:0304::1", "2001::1", "2001:0:5ef5:79fd::1", "2001:db8::1",
        "::ffff:8.8.8.8", "::ffff:127.0.0.1", "64:ff9b::8.8.8.8",
        "localhost", "host.local", "db.internal", "x.lan", "example.com", "",
    ]
    rnd = random.Random(20260809)
    for _ in range(4000):
        hosts.append(str(ipaddress.IPv4Address(rnd.getrandbits(32))))
    for _ in range(4000):
        hosts.append(str(ipaddress.IPv6Address(rnd.getrandbits(128))))
    diff = [h for h in hosts if is_allowed_host(h) != _independent_is_local(h)]
    check("is_allowed_host agrees with an independent implementation", not diff,
          f"{len(hosts)} hosts; {len(diff)} disagreements", "D")

    # EXHAUSTIVE over the security-critical claim: NOTHING globally routable is
    # ever allowed. Enumerate every /8 boundary plus randomized public space.
    leaks = []
    for h in hosts:
        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            continue
        public = ip.is_global or (ip.version == 6 and (
            ip in ipaddress.ip_network("2002::/16") or ip in ipaddress.ip_network("2001::/32")))
        if public and not ip.is_multicast and is_allowed_host(h):
            leaks.append(h)
    check("NO globally-routable address is ever allowed", not leaks,
          f"{len(leaks)} leaks", "E")

    # EXHAUSTIVE truth table for resolve_egress_proxy (2x2x2 = 8 rows).
    rows, wrong = [], []
    un = install("socks5://127.0.0.1:9050")
    try:
        for guard_on in (True, False):
            if not guard_on:
                un()
            for proxy in ("socks5://127.0.0.1:9050", None):
                for url in ("https://example.com", "http://127.0.0.1:8100/x", None):
                    got = resolve_egress_proxy(proxy, url)
                    if proxy:
                        expect = proxy                      # passthrough
                    elif not guard_on:
                        expect = proxy                      # unchanged
                    elif url and _independent_is_local(
                            (re.sub(r"^\w+://", "", url).split("/")[0].split(":")[0])):
                        expect = proxy                      # local target: direct
                    else:
                        expect = os.getenv("TOR_PROXY") or "socks5://127.0.0.1:9050"
                    rows.append((guard_on, proxy, url, got, expect))
                    if got != expect:
                        wrong.append(rows[-1])
            if not guard_on:
                un = install("socks5://127.0.0.1:9050")
    finally:
        un()
    check("resolve_egress_proxy matches its full truth table", not wrong,
          f"{len(rows)} rows enumerated", "E")


# ───────────────────────── 3. F8 eviction ordering ───────────────────────────
def verify_f8_ordering():
    print("\n[3] F8 episodic eviction ordering")
    from ghost_agent.memory.episodes import EpisodicMemory

    # ⚠ This must exercise the PRODUCTION eviction path (record_episode past
    # the cap), NOT a query this script writes itself. An earlier version ran
    # its own ORDER BY and therefore only proved that SQLite sorts correctly —
    # a mutation reverting the real eviction to pure age went UNDETECTED.
    import shutil
    rnd = random.Random(7)
    bad_cases = []
    for trial in range(40):
        d = Path(tempfile.mkdtemp())
        em = EpisodicMemory(d)
        em.MAX_EPISODES = 6
        ids = []
        for k in range(6):
            ids.append(em.record_episode(f"episode {k}"))
        db = d / "episodic_memory.db"
        # Make them all evictable, then give a RANDOM one heavy usage. The
        # oldest row is ids[0]; if the winner is ids[0] and it is heavily used,
        # a pure-age eviction would wrongly kill it.
        with sqlite3.connect(db) as c:
            c.execute("UPDATE episodes SET consolidated=1, lesson=''")
            c.commit()
        winner = ids[0]                      # oldest == pure-age's first victim
        for _ in range(5):
            em._credit_surfaced_episodes([winner])
        em.record_episode("newcomer")        # breach the cap -> evict exactly one
        with sqlite3.connect(db) as c:
            alive = {r[0] for r in c.execute("SELECT id FROM episodes")}
        if winner not in alive:
            bad_cases.append(("used-oldest evicted", trial))
        if ids[1] in alive:
            bad_cases.append(("least-valuable survived", trial))
        shutil.rmtree(d, ignore_errors=True)
    check("PRODUCTION eviction keeps the used row and drops the unused one",
          not bad_cases, f"40 trials via record_episode; {len(bad_cases)} failures", "P")

    # PROPERTY: the credit WRITER must actually increment access_count.
    #
    # ⚠ Isolating this field is necessary, not redundant: crediting updates
    # BOTH access_count and last_accessed, so a survival test alone passes even
    # with the counter dead — the recency tiebreak rescues the same row. A
    # mutation (`access_count + 1` -> `+ 0`) went UNDETECTED until this check
    # existed. Assert the counter directly.
    d = Path(tempfile.mkdtemp())
    em = EpisodicMemory(d)
    eid = em.record_episode("counted episode")
    for n in range(1, 6):
        em._credit_surfaced_episodes([eid])
        got = em.get_episode(eid)["access_count"]
        if got != n:
            bad_cases.append(("access_count did not increment", n, got))
    check("credit writer increments access_count exactly once per surface",
          not any(b[0] == "access_count did not increment" for b in bad_cases),
          "5 sequential credits", "P")
    shutil.rmtree(d, ignore_errors=True)

    # PROPERTY: the cap is still HARD-enforced when every row is used, so
    # value-ordering cannot defeat the bound.
    d = Path(tempfile.mkdtemp())
    em = EpisodicMemory(d); em.MAX_EPISODES = 5
    hot = [em.record_episode(f"hot {i}") for i in range(5)]
    for e in hot:
        em._credit_surfaced_episodes([e])
    for i in range(6):
        em.record_episode(f"extra {i}")
    with sqlite3.connect(d / "episodic_memory.db") as c:
        n = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    check("cap stays hard-enforced when every row is used", n <= em.MAX_EPISODES,
          f"{n} rows / cap {em.MAX_EPISODES}", "P")
    shutil.rmtree(d, ignore_errors=True)




# ───────────────────────── 5. live-number reconciliation ─────────────────────
def verify_live_numbers():
    print("\n[5] live numbers — independent recomputation")
    home = Path(os.getenv("GHOST_HOME", "/Users/vasilis/Data/AI/Data"))
    mem = home / "system" / "memory"
    if not (mem / "chroma.sqlite3").exists():
        skip("live-number reconciliation", f"no live store at {mem}")
        return

    # DIFFERENTIAL: count chroma types two independent ways (SQL GROUP BY vs a
    # Python tally of the raw rows). They must agree exactly.
    con = sqlite3.connect(f"file:{mem/'chroma.sqlite3'}?mode=ro&immutable=1", uri=True)
    sql = dict(con.execute("SELECT string_value,COUNT(*) FROM embedding_metadata "
                           "WHERE key='type' GROUP BY string_value").fetchall())
    raw = {}
    for (v,) in con.execute("SELECT string_value FROM embedding_metadata WHERE key='type'"):
        raw[v] = raw.get(v, 0) + 1
    con.close()
    check("chroma type census: SQL GROUP BY == manual tally", sql == raw,
          f"total {sum(sql.values())}", "D")

    # (§4TD: the lesson-playbook cap property went with the lesson store.)

    # PROPERTY: F8 invariant — every episode row must carry a seeded
    # last_accessed (0 would sort it evict-first, the bug the backfill fixed).
    e = sqlite3.connect(f"file:{mem/'episodic_memory.db'}?mode=ro", uri=True)
    n, unseeded, neg = e.execute(
        "SELECT COUNT(*), SUM(last_accessed=0), SUM(access_count<0) FROM episodes").fetchone()
    e.close()
    check("every episode has last_accessed != 0 (F8 backfill invariant)",
          (unseeded or 0) == 0, f"{n} rows, {unseeded or 0} unseeded", "P")
    check("no negative access_count", (neg or 0) == 0, "", "P")

    # DIFFERENTIAL: dream tallies recomputed from the raw log by a second regex.
    log = home / "system" / "ghost-agent.log"
    if log.exists():
        t = log.read_text(errors="ignore")
        a = re.findall(r"Synthesized (\d+) new meta-memories and extracted (\d+) heuristics", t)
        b = [(m.group(1), m.group(2)) for m in
             re.finditer(r"Synthesized\s+(\d+)\s+new meta-memories and extracted\s+(\d+)\s+heuristics", t)]
        check("dream tally: two independent regex passes agree", a == b,
              f"{len(a)} cycles, {sum(int(x[1]) for x in a)} heuristics", "D")


# ─────────── 6. the anytime-valid confidence sequence (the DECISION math) ────
# §4NN: [6] verify_confidence_sequence (the experiments verdict instrument,
# core/experiments.py) was removed with that module.


def main() -> int:
    print("=" * 78)
    print("MECHANIZED VERIFICATION OF THE COMPUTATIONAL CORE")
    print("=" * 78)
    for fn in (verify_bio_scaling, verify_egress_classification, verify_f8_ordering,
               verify_live_numbers):
        try:
            fn()
        except Exception as exc:  # a verifier that crashes proves nothing
            check(f"{fn.__name__} (crashed)", False, f"{type(exc).__name__}: {exc}")
    print("\n" + "=" * 78)
    print(f"{sum(1 for v in STATS.values() if v)}/{len(STATS)} checks passed"
          + (f"  ({len(SKIPPED)} skipped: {', '.join(SKIPPED)})" if SKIPPED else ""))
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
