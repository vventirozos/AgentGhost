#!/usr/bin/env python3
r"""§4UZ battery — how a research call spends the worker's time (2026-11-19): page distills through one
queue per client (`DistillQueue` / `DistillWave` in core/node_throughput.py), one at a time by default
(GHOST_DISTILL_WIDTH), best rank first, each sized from the time the phase has left; readings taken above
the queue's width no longer drop the slow tracker; deep_research, darkweb_research and the recheck all
take the queue. Review r1: the queue is keyed by the worker node (wrappers share it), plans keep
headroom, a timeout teaches the tracker and halves the next plan, two in a row end the wave.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uz.py [-jN] [all|rerun M1 M2 …|check]
(`check` = the anchor-uniqueness check only: no test runs.)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NT = "src/ghost_agent/core/node_throughput.py"; SE = "src/ghost_agent/tools/search.py"
DW = "src/ghost_agent/tools/darkweb_search.py"; LL = "src/ghost_agent/core/llm.py"
MUT = [
    ("CONTROL no-op", NT, r'''def distill_width() -> int:''', r'''def distill_width() -> int:  # no-op'''),
    ("CONTROL known-bad", NT, r'''        budget = min(self.ceiling_s, avail * min(width, waiting) / waiting)''', r'''        budget = 0.0'''),
    # W — the width
    ("W1 width-ignored", NT, r'''        while self._waiters and self.running < distill_width():''', r'''        while self._waiters:'''),
    ("W2 env-not-read", NT, r'''        w = int(raw) if raw else DISTILL_WIDTH_DEFAULT''', r'''        w = DISTILL_WIDTH_DEFAULT'''),
    ("W3 default-three", NT, r'''DISTILL_WIDTH_DEFAULT = 1''', r'''DISTILL_WIDTH_DEFAULT = 3'''),
    ("W4 no-upper-bound", NT, r'''    return min(DISTILL_WIDTH_MAX, w) if w >= 1 else DISTILL_WIDTH_DEFAULT''', r'''    return w if w >= 1 else DISTILL_WIDTH_DEFAULT'''),
    ("W5 zero-accepted", NT, r'''    return min(DISTILL_WIDTH_MAX, w) if w >= 1 else DISTILL_WIDTH_DEFAULT''', r'''    return min(DISTILL_WIDTH_MAX, w)'''),
    # O / Q — the queue
    ("O1 order-ignored", NT, r'''        heapq.heappush(self._waiters, (rank, next(self._seq), fut))''', r'''        heapq.heappush(self._waiters, (0, next(self._seq), fut))'''),
    ("Q1 wait-timeout-late", NT, r'''                max(0.0, float(timeout)),
                lambda: fut.done() or fut.set_result(False))''', r'''                max(0.0, float(timeout)) + 5.0,
                lambda: fut.done() or fut.set_result(False))'''),
    ("Q2 cancelled-turn-kept", NT, r'''            if fut.done() and not fut.cancelled() and fut.result() is True:
                self.release()''', r'''            if False:
                self.release()'''),
    ("Q3 release-wakes-nobody", NT, r'''        self.running = max(0, self.running - 1)
        self._wake()''', r'''        self.running = max(0, self.running - 1)'''),
    ("Q4 dead-waiter-woken", NT, r'''            if fut.done():                  # timed out or cancelled while it waited
                continue''', r'''            if False:
                continue'''),
    ("Q5 queue-per-call", NT, r'''        if q is None or q.loop is not loop:
            q = _NODE_QUEUES[urls] = DistillQueue(loop)''', r'''        if True:
            q = _NODE_QUEUES[urls] = DistillQueue(loop)'''),
    ("Q6 queue-per-call-no-nodes", NT, r'''        if q is None or q.loop is not loop:
            q = _QUEUES[base] = DistillQueue(loop)''', r'''        if True:
            q = _QUEUES[base] = DistillQueue(loop)'''),
    # S — what the slow tracker learns (design point 4)
    ("S1 slow-guard-removed", NT, r'''        _in_regime = max(1, int(concurrency or 1)) <= distill_width()''', r'''        _in_regime = True'''),
    ("S2 slow-never-learned", NT, r'''        _in_regime = max(1, int(concurrency or 1)) <= distill_width()''', r'''        _in_regime = False'''),
    ("S3 guard-ignores-width", NT, r'''        _in_regime = max(1, int(concurrency or 1)) <= distill_width()''', r'''        _in_regime = max(1, int(concurrency or 1)) <= 1'''),
    # B — one page's budget
    ("B1 budget-not-divided", NT, r'''        budget = min(self.ceiling_s, avail * min(width, waiting) / waiting)''', r'''        budget = min(self.ceiling_s, avail)'''),
    ("B2 ceiling-dropped", NT, r'''        budget = min(self.ceiling_s, avail * min(width, waiting) / waiting)''', r'''        budget = avail * min(width, waiting) / waiting'''),
    ("B3 width-not-in-share", NT, r'''        budget = min(self.ceiling_s, avail * min(width, waiting) / waiting)''', r'''        budget = min(self.ceiling_s, avail / waiting)'''),
    ("B4 reserve-dropped", NT, r'''        return self.deadline - time.monotonic() - self.reserve_s''', r'''        return self.deadline - time.monotonic()'''),
    ("B5 allowance-always", NT, r'''        allow = self.slot_allowance_s if width > 1 else 0.0''', r'''        allow = self.slot_allowance_s'''),
    ("B6 allowance-never", NT, r'''        allow = self.slot_allowance_s if width > 1 else 0.0''', r'''        allow = 0.0'''),
    ("B7 floor-dropped", NT, r'''            if not plan.feasible and least > 0:''', r'''            if False:'''),
    ("B8 floor-past-the-deadline", NT, r'''                if floor <= min(self.ceiling_s, avail):''', r'''                if True:'''),
    ("B9 spent-clock-planned", NT, r'''        if avail <= 0:
            # never plan''', r'''        if False:
            # never plan'''),
    ("B10 done-pages-still-wait", NT, r'''        self._waiting.discard(key)''', r'''        pass'''),
    ("B11 minimum-not-reported", NT, r'''                f"tok/s", _least)''', r'''                f"tok/s")'''),
    ("B12 backfill-outranks", NT, r'''        self._rank.setdefault(key, len(self._rank))''', r'''        self._rank.setdefault(key, -len(self._rank))'''),
    # P — the plan request
    ("P1 width-not-declared", NT, r'''        kwargs: Dict[str, Any] = {"concurrency": width}''', r'''        kwargs: Dict[str, Any] = {"concurrency": 1}'''),
    ("P2 report-share-dropped", NT, r'''            kwargs["max_chars"] = self.max_chars''', r'''            pass'''),
    # T — a page's turn
    ("T1 turn-never-released", NT, r'''            if mine:
                self.queue.release()
            self.drop(key)''', r'''            self.drop(key)'''),
    ("T2 turn-waits-forever", NT, r'''            mine = await self.queue.acquire(self._rank[key], timeout=self.left())''', r'''            mine = await self.queue.acquire(self._rank[key], timeout=None)'''),
    ("T3 plan-error-raised", NT, r'''        except Exception as exc:  # noqa: BLE001 — the caller degrades to raw text
            return DistillTurn(None, max(0.0, budget), width, exc)''', r'''        except ZeroDivisionError as exc:  # noqa: BLE001 — the caller degrades to raw text
            return DistillTurn(None, max(0.0, budget), width, exc)'''),
    # D — deep_research
    ("D1 own-queue", SE, r'''max_chars=_report_share_chars, label="deep research")''', r'''max_chars=_report_share_chars, label="deep research"); _wave.queue = type(_wave.queue)()'''),
    ("D2 no-reserve", SE, r'''                        reserve_s=RESEARCH_ASSEMBLY_S, slot_allowance_s=_QUEUE_ALLOWANCE_S,''', r'''                        reserve_s=0.0, slot_allowance_s=_QUEUE_ALLOWANCE_S,'''),
    ("D3 no-ceiling", SE, r'''deadline=_phase_deadline, ceiling_s=_WEB_SUMMARY_TIMEOUT_S,
                        reserve_s=RESEARCH_ASSEMBLY_S''', r'''deadline=_phase_deadline, ceiling_s=1e9,
                        reserve_s=RESEARCH_ASSEMBLY_S'''),
    ("D4 failed-page-still-waits", SE, r'''        finally:
            _wave.drop(str(url))''', r'''        finally:
            pass'''),
    ("D5 no-hard-bound", SE, r'''                task_label="web summary"), _summary_budget + 2.0)''', r'''                task_label="web summary"), None)'''),
    # D6 distill-net-removed: NOT a mutant — the `except Exception` around `_wave.turn` in `_distill` is a second
    # net behind the guards inside `_distill_sized`; nothing reaches it today (no observable difference)
    ("D7 recheck-own-queue", SE, r'''    _queue = distill_queue(llm_client)          # §4UZ: the same queue as the first extractions''', r'''    _queue = type(distill_queue(llm_client))()          # §4UZ: the same queue as the first extractions'''),
    ("D8 recheck-first-unqueued", SE, r'''            if not _mine or deadline - time.monotonic() < _needs(idx[0]):     # r1: fit re-checked after the wait
                return out
            first = await _one(idx[0])''', r'''            first = await _one(idx[0])'''),
    # D9 unsized-page-posted: EQUIVALENT — an unsized turn already carries `plan=None`; the block only logs why
    # r1 — the review round: the queue's key, headroom, timeout feedback, the prior rule, the minors
    ("M1 keyed-by-client-object", NT, r"""    if urls:
        q = _NODE_QUEUES.get(urls)""", r"""    if False:
        q = _NODE_QUEUES.get(urls)"""),
    ("M2 wrapper-not-unwrapped", NT, r"""        llm_client, seen = inner, seen + 1""", r"""        seen = 8"""),
    ("H1 no-headroom", NT, r"""        sized = BUDGET_UTILISATION * self._shrink""", r"""        sized = 1.0 * self._shrink"""),
    ("H2 no-halving", NT, r"""        sized = BUDGET_UTILISATION * self._shrink""", r"""        sized = BUDGET_UTILISATION"""),
    ("H3 floor-without-headroom", NT, r"""                floor = (base + 0.05) / BUDGET_UTILISATION + allow""", r"""                floor = base + 0.05 + allow"""),
    ("F1 timeouts-not-counted", NT, r"""        self._timeouts += 1""", r"""        pass"""),
    ("F2 success-does-not-reset", NT, r"""            self._timeouts = 0
            self._row = 0""", r"""            self._row = 0"""),
    ("F8 success-restores-the-size", NT, r"""            self._timeouts = 0
            self._row = 0""", r"""            self._timeouts, self._shrink = 0, 1.0
            self._row = 0"""),
    ("F9 any-timeout-counts", NT, r"""            slack = turn.minimum and predicted > 0 and (""", r"""            slack = predicted > 0 and ("""),
    ("F10 smallest-plan-not-recognised", NT, r"""                minimum = (int(plan.char_limit) <= MIN_CHARS and int(plan.max_tokens) <= MIN_TOKENS)""", r"""                minimum = False"""),
    ("F11 floor-plan-not-minimum", NT, r"""                    minimum = bool(plan.feasible)""", r"""                    minimum = False"""),
    ("H4 smallest-plan-on-a-bare-limit", NT, r"""                    budget = max(budget, floor)""", r"""                    budget = floor"""),
    ("H5 halving-does-not-compound", NT, r"""        self._shrink *= AFTER_TIMEOUT_SHRINK""", r"""        self._shrink = AFTER_TIMEOUT_SHRINK"""),
    ("F3 any-failure-is-a-timeout", NT, r"""        if not timed_out:
            return                           # refused""", r"""        if False:
            return                           # refused"""),
    ("N1 client-flag-not-read", NT, r"""        if getattr(exc, "timed_out", False) is True:
            return True""", r"""        if False:
            return True"""),
    ("N2 timeout-class-by-name-not-read", NT, r"""        return any(c.__name__.endswith(("Timeout", "TimeoutException", "TimeoutError"))
                   for c in type(exc).__mro__)""", r"""        return False"""),
    ("N3 give-up-without-slack", NT, r"""            slack = turn.minimum and predicted > 0 and (
                float(turn.budget_s) >= GIVE_UP_SLACK * predicted * self._slowed)""", r"""            slack = turn.minimum and predicted > 0"""),
    ("C1 no-cap", NT, r"""        if self._timeouts >= WAVE_MAX_TIMEOUTS or self._row >= WAVE_MAX_ANY_TIMEOUTS:""", r"""        if self._timeouts >= WAVE_MAX_TIMEOUTS:"""),
    ("C2 cap-of-three", NT, r"""WAVE_MAX_ANY_TIMEOUTS = 4""", r"""WAVE_MAX_ANY_TIMEOUTS = 3"""),
    ("C3 success-does-not-end-the-row", NT, r"""            self._row = 0
            return""", r"""            return"""),
    ("C4 row-not-counted", NT, r"""        self._row += 1""", r"""        pass"""),
    ("N4 slack-against-the-lowered-rates", NT, r"""                float(turn.budget_s) >= GIVE_UP_SLACK * predicted * self._slowed)""", r"""                float(turn.budget_s) >= GIVE_UP_SLACK * predicted)"""),
    ("N5 limit-grows-with-every-timeout", NT, r"""                base = (least - SAFETY_MARGIN_S) * self._slowed + SAFETY_MARGIN_S""", r"""                base = least"""),
    ("N6 slowed-without-learning", NT, r"""            if learned and predicted > 0 and float(burned_s) > 0:""", r"""            if predicted > 0 and float(burned_s) > 0:"""),
    ("L2 client-never-says-timeout", LL, r"""                _exc.timed_out = _worker_timed_out""", r"""                _exc.timed_out = False"""),
    ("L3 every-worker-failure-a-timeout", LL, r"""                        if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError)):
                            _worker_timed_out = True""", r"""                        if True:
                            _worker_timed_out = True"""),
    ("D13 search-reports-any-error-as-timeout", SE, r"""_wave.report(_turn, False, timed_out=is_distill_timeout(_sum_exc),""", r"""_wave.report(_turn, False, timed_out=True,"""),
    ("F4 timeout-sample-not-pessimistic", NT, r"""        f = min(1.0, believed / burned)""", r"""        f = 1.0"""),
    ("F5 tracker-not-told", NT, r"""            learned = self.llm_client.note_distill_timeout(""", r"""            learned = self.llm_client.note_distill_timeout_(""")  ,
    ("F6 gave-up-still-plans", NT, r"""        if self.gave_up:
            return DistillTurn(None, 0.0, width, note=GAVE_UP_NOTE)""", r"""        if False:
            return DistillTurn(None, 0.0, width, note=GAVE_UP_NOTE)"""),
    # (the `if self.gave_up:` in `turn` only saves the wait for a turn `_size` would refuse: no mutant)
    ("F7 one-timeout-ends-the-wave", NT, r"""WAVE_MAX_TIMEOUTS = 2""", r"""WAVE_MAX_TIMEOUTS = 1"""),
    ("R1 above-width-mean-plans", NT, r"""                return prior if observed else mean""", r"""                return mean"""),
    ("R2 seeded-rates-ignored", NT, r"""        observed = len(cur) > 3""", r"""        observed = True"""),
    ("L1 client-tells-no-node", LL, r"""            for n in (getattr(self, "worker_clients", None) or []):
                learned = self.node_throughput.observe_timeout(""", r"""            for n in []:
                learned = self.node_throughput.observe_timeout("""),
    ("D10 timeout-not-reported", SE, r"""                if not _answered:""", r"""                if False:"""),
    ("D11 no-time-not-said", SE, r"""        if plan is None and _turn.note:""", r"""        if False:"""),
    ("D12 recheck-fit-not-rechecked", SE, r"""            if not _mine or deadline - time.monotonic() < _needs(idx[0]):     # r1""", r"""            if not _mine:     # r1"""),
    ("K4 stdlib-wait-for", DW, r"""                block = await _search._cancel_safe_wait_for(process_url(url, budget),""", r"""                block = await asyncio.wait_for(process_url(url, budget),"""),
    # K — darkweb_research
    ("K1 own-queue", DW, r'''                        label="darkweb research")''', r'''                        label="darkweb research"); _wave.queue = type(_wave.queue)()'''),
    ("K2 failed-page-still-waits", DW, r'''        finally:
            _wave.drop(str(url))''', r'''        finally:
            pass'''),
    ("K3 no-reserve", DW, r'''reserve_s=_search.RESEARCH_ASSEMBLY_S,''', r'''reserve_s=0.0,'''),
]
_P = "tests/test_4uz_research_distill.py::"
#: module-level code (a constant), or a line coverage cannot tie to the test that judges it
FORCE = {
    "W3 default-three": [_P + "test_with_width_one_no_two_distills_overlap", _P + "test_the_width_is_read_at_call_time"],
    "D1 own-queue": [_P + "test_every_distilling_tool_shares_the_one_queue"],
    "K1 own-queue": [_P + "test_every_distilling_tool_shares_the_one_queue"],
    "D7 recheck-own-queue": [_P + "test_every_distilling_tool_shares_the_one_queue"],
    "Q5 queue-per-call": [_P + "test_every_wrapper_of_the_client_shares_its_queue"],
    "Q6 queue-per-call-no-nodes": [_P + "test_every_distilling_tool_shares_the_one_queue"],
    "K2 failed-page-still-waits": [_P + "test_darkweb_pages_that_fail_stop_counting_as_waiting"],
    "F7 one-timeout-ends-the-wave": [_P + "test_only_two_minimum_size_timeouts_in_a_row_end_the_wave"],
    "M1 keyed-by-client-object": [_P + "test_every_wrapper_of_the_client_shares_its_queue"],
    "K4 stdlib-wait-for": [_P + "test_a_cancel_landing_as_an_onion_fetch_completes_is_not_swallowed"],
    "C2 cap-of-three": [_P + "test_four_timeouts_in_a_row_end_the_wave_and_a_success_starts_the_row_again",
                        _P + "test_a_busy_moment_in_a_short_phase_does_not_end_the_wave"],
    "K3 no-reserve": [_P + "test_darkweb_pages_that_fail_stop_counting_as_waiting",
                      "tests/test_4un_darkres.py::test_the_distiller_keeps_time_after_a_slow_fetch"],
}
#: run ONLY these tests (not the coverage selection): a queue that passes no turn makes every real-clock
#: research test wait out its 190 s phase before failing; the virtual-clock file kills both in seconds
ONLY = {"Q3 release-wakes-nobody": ["tests/test_4uz_research_distill.py"],
        "T1 turn-never-released": ["tests/test_4uz_research_distill.py"]}
PINS = ["tests/test_4uz_research_distill.py", "tests/test_distill_budget_sizing.py", "tests/test_node_throughput.py",
        "tests/test_node_concurrency_gate.py", "tests/test_llm_r6_fixes.py", "tests/test_4nk_darkweb_research.py", "tests/test_4nj_research_yield.py",
        "tests/test_4un_deep_research.py", "tests/test_4un_darkres.py", "tests/test_4un_darkweb.py",
        "tests/test_4uy_internet.py", "tests/test_4tb_fact_check.py", "tests/test_fact_check.py",
        "tests/test_deep_research_fetch_racing.py", "tests/test_research_failure_honesty.py",
        "tests/test_research_outcome_status.py", "tests/test_darkweb_search.py", "tests/test_4oe_research.py",
        "tests/test_4pd_search_review.py", "tests/test_search_map_reduce.py", "tests/test_tool_interrupt.py"]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    if len({m[0] for m in MUT}) != len(MUT) or any(lab not in {m[0] for m in MUT} for lab in list(FORCE) + list(ONLY)):
        raise SystemExit("duplicate mutant label, or a FORCE entry names no mutant")
    if args[0] == "check":
        print(f"anchors ok: {len(MUT)} mutants in {len(srcs)} files; {len(PINS)} test files"); return
    import battery_cov as BC
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        if label in ONLY:
            return list(ONLY[label]), f"only {len(ONLY[label])} file(s)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | set(FORCE[label]))
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(PINS), f"fallback {len(PINS)} files"
    print(f"{len(PINS)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
