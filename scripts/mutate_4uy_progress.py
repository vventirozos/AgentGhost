#!/usr/bin/env python3
r"""§4UY battery — live steps (§4UT) fixes of the §4UX-4 findings S1–S18 (2026-11-18): the summary table from
the tool registry, a step shown when it starts and a task turn's detail, the narrator's fence and line rules,
the log without figures, a narrator timeout that never charges Nova, "off" without Nova calls, failed calls,
why a step has no sentence, the late sentence, the web list that outlives its bubble / never pulses after its
turn / renders once per burst / announces, and the CLI's order and error path.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uy_progress.py [-jN] [all|rerun M1 M2 …]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PG = "src/ghost_agent/core/progress.py"; RT = "src/ghost_agent/api/routes.py"
AG = "src/ghost_agent/core/agent.py"; SV = "interface/server.py"
AJ = "interface/static/app.js"; PJ = "interface/static/progress.js"; CSS = "interface/static/style.css"
CLI = "interface/externals/cli/ghost"
#: Not here, proven equivalent: `.progress-step:last-child` for `.is-running` in style.css (the running step, or
#: the "Thinking…" row, is always the list's last child while it works) — kept as the explicit class.
MUT = [
    ("CONTROL no-op", PG, r'''def summary(counts: Dict[str, int], failed: Optional[Dict[str, int]] = None, known: Any = ()) -> str:''',
     r'''def summary(counts: Dict[str, int], failed: Optional[Dict[str, int]] = None, known: Any = ()) -> str:  # no-op'''),
    ("CONTROL known-bad", PG, r'''    if not parts:
        return ""
    text = ", ".join(parts[:-1])''', r'''    return ""
    text = ", ".join(parts[:-1])'''),
    # S2 — the summary table is the registry's
    ("S2a key-file-ops-only", PG, r'''    act = action_of(args)[:64]
    return f"{name}.{act}" if act else name''', r'''    act = action_of(args)[:64] if name == "file_system" else ""
    return f"{name}.{act}" if act else name'''),
    ("S2b labs-not-lab", PG, r'''    "lab": _v("worked on a lab"),''', r'''    "labs": _v("worked on a lab"),'''),
    ("S2c list-updates", PG, r'''    "manage_projects.list": _v("listed the projects"),''', r'''    "manage_projects.list": _v("updated the project"),'''),
    ("S2d dead-entry", PG, r'''    "execute": _v("ran a command", "ran {n} commands"),''', r'''    "execute": _v("ran a command", "ran {n} commands"),
    "shell": _v("ran a command", "ran {n} commands"),'''),
    # S3 — the step as it starts; a task turn inside it
    ("S3a no-start", PG, r'''            self._finish_live()           # a batch never done (it raised) is not left running
            counts: Dict[str, int] = {}''', r'''            return
            counts: Dict[str, int] = {}'''),
    ("S3b no-child", PG, r'''            if parent is not None and not parent.closed and parent._live is not None:''',
     r'''            if False:'''),
    ("S3c child-any-time", PG, r'''            if parent is not None and not parent.closed and parent._live is not None:''',
     r'''            if parent is not None:'''),
    ("S3d detail-dropped", PG, r'''            self.tap.progress({"seq": live["seq"], "detail": _clip(text, LINE_MAX_CHARS)})''',
     r'''            pass'''),
    ("S3e running-left-at-close", PG, r'''        try:
            self._finish_live()
        except Exception:  # noqa: BLE001
            pass
        self.closed = True''', r'''        self.closed = True'''),
    ("S3f gone-not-sent", PG, r'''                if live is not None:                # it started, nothing in it ran
                    self.tap.progress({"seq": live["seq"], "gone": True})''', r'''                pass'''),
    ("S3g new-seq-on-done", PG, r'''            seq = live["seq"] if live is not None else self.seq''', r'''            self.seq += 1
            seq = self.seq'''),
    ("S3h loop-no-start", AG, r'''                        if _progress is not None:                    # §4UY S3: the step shows as it STARTS
                            _progress.before_batch(tool_calls)''', r'''                        pass'''),
    # S5 — the fence; injected lines
    ("S5a fence-closable", PG, r'''    return str(text or "").replace(_FENCE_OPEN, "‹‹‹").replace(_FENCE_CLOSE, "›››")''',
     r'''    return str(text or "")'''),
    ("S5b address-passes", PG, r'''    if _ADDRESS_RE.search(s):
        return "an address"''', r'''    if False:
        return "an address"'''),
    ("S5c not-first-person-passes", PG, r'''    if not _FIRST_PERSON_RE.match(s):
        return "not a first-person line"''', r'''    if False:
        return "not a first-person line"'''),
    # S6 — abbreviations; copied non-Latin words
    ("S6a eg-ends-a-sentence", PG, r'''    if _SENTENCE_END_RE.search(_ABBREV_RE.sub("x", s).rstrip(".!? ")):''',
     r'''    if re.search(r"[.!?](?:\s+|$)", s.rstrip(".!?") + " "):'''),
    ("S6b any-greek-drops", PG, r'''        if w.lower() not in low_seen:''', r'''        if True:'''),
    ("S6c any-greek-passes", PG, r'''        if w.lower() not in low_seen:''', r'''        if False:'''),
    # S7 — the log
    ("S7 figures-logged", PG, r'''                    pretty_log("Progress", masked, icon=Icons.PROGRESS)''',
     r'''                    pretty_log("Progress", line, icon=Icons.PROGRESS)'''),
    ("S7b record-keeps-figures", PG, r'''                        s["text"] = masked''', r'''                        s["text"] = line'''),
    # S8 — negatives
    ("S8 negative-passes", PG, r'''    if _NEGATIVE_RE.search(s) and not _NEGATIVE_RE.search(str(results or "")):''',
     r'''    if False:'''),
    ("S8b negative-always-drops", PG, r'''    if _NEGATIVE_RE.search(s) and not _NEGATIVE_RE.search(str(results or "")):''',
     r'''    if _NEGATIVE_RE.search(s):'''),
    ("S8c negative-from-request", PG, r'''    if _NEGATIVE_RE.search(s) and not _NEGATIVE_RE.search(str(results or "")):''',
     r'''    if _NEGATIVE_RE.search(s) and not _NEGATIVE_RE.search(str(seen or "") + str(results or "")):'''),
    ("S8d results-not-passed", PG, r'''        results = " ".join(c["result"] for c in shown)''', r'''        results = ""'''),
    # S9 — the breaker
    ("S9 deadline-as-http-timeout", PG, r'''                                         timeout=NARRATE_NODE_TIMEOUT_S, slot_wait=NARRATE_TIMEOUT_S,''',
     r'''                                         timeout=NARRATE_TIMEOUT_S, slot_wait=NARRATE_TIMEOUT_S,'''),
    # S10 — off
    ("S10a sentences-ignored", PG, r'''            if getattr(tap, "progress_sentences", True) is False or not worker_is_off_main(llm):''',
     r'''            if not worker_is_off_main(llm):'''),
    ("S10b route-always-sentences", RT, r'''            _tap.progress_sentences = _prog == "1"''', r'''            _tap.progress_sentences = True'''),
    ("S10c route-steps-not-wanted", RT, r'''            _tap.progress_wanted = _prog in ("1", "steps")''', r'''            _tap.progress_wanted = _prog == "1"'''),
    ("S10d proxy-always-1", SV, r'''    return {"X-Ghost-Progress": "steps" if str(value or "").strip().lower() == "off" else "1"}''',
     r'''    return {"X-Ghost-Progress": "1"}'''),
    ("S10e page-sends-nothing", AJ, r'''        if (progressMode() === 'off') payload.progress = 'off';''', r'''        // removed'''),
    # S13 — failed calls
    ("S13a failed-not-counted", PG, r'''                if getattr(res, "is_failure", False) is True:
                    failed[key] = failed.get(key, 0) + 1
                calls.append''', r'''                calls.append'''),
    ("S13b failed-not-said", PG, r'''        if f > 0:
            part += " (failed)" if n == 1 else f" ({f} failed)"''', r'''        if False:
            part += " (failed)" if n == 1 else f" ({f} failed)"'''),
    # S14 — quotes
    ("S14 strip-any-quote", PG, r'''    return _unwrap_quotes(" ".join(str(text or "").split()).strip())''',
     r'''    return " ".join(str(text or "").split()).strip().strip('"“”\'')'''),
    ("S14b check-reads-raw", PG, r'''    s = _normalise(text)
    if not s:
        return "empty"''', r'''    s = " ".join(str(text or "").split()).strip()
    if not s:
        return "empty"'''),
    ("S14c inner-quote-keeps-outer", PG, r'''        if balanced:''', r'''        if s[0] not in inner and s[-1] not in inner:'''),
    # review — line rules, the running label, the cut reason, the cancel-safe deadline, nested turns
    ("R3a st-ends-a-sentence", PG, r'''Mr|Mrs|Ms|Dr|Jr|Sr|St|Mt|''', r'''Mr|Mrs|Ms|Dr|Jr|Sr|Mt|'''),
    ("R3c abbrev-any-case", PG, r'''r"\bNo\.(?=\s*\d)|\b(?i:etc|vs|approx|incl|cf)\.")''',
     r'''r"\bNo\.(?=\s*\d)|\b(?i:etc|vs|approx|incl|cf)\.", re.I)'''),
    ("R3d no-always-abbrev", PG, r'''r"\bNo\.(?=\s*\d)|''', r'''r"\bNo\.|'''),
    ("R3e apostrophe-is-a-quote", PG, r'''        marks = re.sub(r"(?<=\w)['’](?=\w)", "", inner)''', r'''        marks = inner'''),
    # round 2 — the address class, the prompt, bounded summaries, unknown names
    ("R2a ip-passes", PG, r'''                         r"\b\d{1,3}(?:\.\d{1,3}){3}\b|"''', r'''                         r""'''),
    ("R2b host-path-passes", PG, r'''r"\b[\w-]+(?:\.[\w-]+)*\.[A-Za-z][\w-]*/)", re.I)''', r'''r"(?!x)x)", re.I)'''),
    ("R2c dotted-passes", PG, r'''        if m.group(0).lower() not in low_args:''', r'''        if False:'''),
    ("R2d args-not-passed", PG, r'''        args = " ".join(str(c["args"]) for c in shown)''', r'''        args = ""'''),
    ("R2e prompt-allows-note", PG, r'''happen unless the results above say so.''', r'''happen unless they say so.'''),
    ("R2f summary-unbounded", PG, r'''    return _clip(text[0].upper() + text[1:], LINE_MAX_CHARS)''',
     r'''    return text[0].upper() + text[1:]'''),
    ("R2g any-name-shown", PG, r'''    if base in (known or ()):''', r'''    if True:'''),
    ("R2h ran-not-known", PG, r'''            ran = self.known | {k.split(".")[0] for k in counts}''', r'''            ran = self.known'''),
    ("R2i key-unbounded", PG, r'''    name = str(name or "").strip()[:64] or "tool"''', r'''    name = str(name or "").strip() or "tool"'''),
    ("R3b version-is-address", PG, r'''[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b''', r'''[\w.+-]+@[\w-]+\.\w'''),
    ("R4a label-unbounded", PG, r'''    return _clip("Running " + ", ".join(parts) + "…", LINE_MAX_CHARS) if parts else ""''',
     r'''    return ("Running " + ", ".join(parts) + "…") if parts else ""'''),
    ("R4b label-shows-any-action", PG, r'''        name = (key if key in _VERBS else base if (base in _VERBS or base in (known or ())) else "a tool")''',
     r'''        name = key'''),
    ("R5 cut-is-late", PG, r'''        kept = dict(frame, why="cut", cut=True)''', r'''        kept = dict(frame, why="late", cut=True)'''),
    ("R5b cut-no-tooltip", PJ, r'''    cut: 'No sentence: the step was stopped before it finished',''', r''''''),
    ("R6 stdlib-wait-for", PG, r'''        from ..utils.aio import wait_for''', r'''        from asyncio import wait_for'''),
    ("R7 nested-turn-keeps-tap", AG, r'''        _own_tap_token = (_rtc_own.set(None)''', r'''        _own_tap_token = (None'''),
    # S15 — why no sentence
    ("S15a busy-silent", PG, r'''            if self._pending is not None:           # collapsed into this newer step: it gets no sentence
                self._no_sentence(self._pending["seq"], "busy")''', r'''            pass'''),
    ("S15b failure-silent", PG, r'''                if not line:
                    self._no_sentence(job["seq"], why)
                    continue''', r'''                if not line:
                    continue'''),
    ("S15c slow-is-unavailable", PG, r'''        except asyncio.TimeoutError:
            return "", "slow"''', r'''        except asyncio.TimeoutError:
            return "", "unavailable"'''),
    # S16 — the late line
    ("S16 close-drops-pending", PG, r'''        self.closed = True                          # the one step queued for a line (at most one) still gets it''',
     r'''        self.closed = True
        self._pending = None'''),
    # S1, S3, S4, S13, S15, S16, S17 — the web list
    ("W1 list-goes-with-bubble", PJ, r'''        return this.el.isConnected;
    }''', r'''        this.el.remove();
        return false;
    }'''),
    ("W3 last-step-current", PJ, r'''        return last && last.running ? last : null;''', r'''        return last || null;'''),
    ("W3c done-revives", PJ, r'''    if (frame.running === true && !s.summary) s.running = true;''', r'''    if (frame.running === true) s.running = true;'''),
    ("W3d gone-ignored", PJ, r'''    if (frame.gone === true) {
        if (at >= 0) steps.splice(at, 1);
        return steps;
    }''', r'''    if (frame.gone === true) return steps;'''),
    ("W4a give-up-keeps-working", AJ, r'''                        if (typeof _endSteps === 'function') _endSteps();      // §4UY S4: the list pulsed for ever after the give-up''',
     r'''                        // removed'''),
    ("W13 failed-looks-ok", PJ, r'''                    + (s === current ? ' is-running' : '') + (s.failed ? ' is-failed' : '')''',
     r'''                    + (s === current ? ' is-running' : '')'''),
    ("W15 no-reason", PJ, r'''                if (!s.text && s.why && NO_SENTENCE[s.why]) li.title = NO_SENTENCE[s.why];''', r'''                // removed'''),
    ("W16 nothing-after-answer", PJ, r'''            if (!this.steps.some((x) => x.seq === Number(frame.seq))) return;''', r'''            return;'''),
    ("W17a render-per-frame", PJ, r'''        if (JSON.stringify(this.steps) !== before) this._schedule();''', r'''        if (JSON.stringify(this.steps) !== before) this.render();'''),
    ("W17b no-live-region", PJ, r'''            this._live.setAttribute('aria-live', 'polite');''', r'''            // removed'''),
    ("W17c never-pruned", PJ, r'''    for (const l of [..._lists]) if (l.el && !l.el.isConnected) _lists.delete(l);''', r'''    // removed'''),
    # S3, S11, S12, S13 — the CLI
    ("C11 next-step-flushes", CLI, r'''                self._held = [h for h in self._held if h.get("seq") != seq] + [step]''',
     r'''                self._release_locked(everything=True)
                self._held = [step]'''),
    ("C11b later-sentence-waits", CLI, r'''                if hit:                         # the narrator works in order: the earlier ones get none''',
     r'''                if False:'''),
    ("C12 error-drops", CLI, r'''            if steps is not None:      # §4UY S12: a held step prints ABOVE the error/stop line (it was
                steps.stop()           # dropped); the normal end stops — prints — it before this runs''',
     r'''            if steps is not None:
                steps._held = []
                steps.stop()'''),
    ("C3 running-ignored", CLI, r'''            elif progress.get("running") is True:
                self._running_locked(str(progress.get("label") or ""))
                return''', r'''            elif progress.get("running") is True:
                return'''),
    ("C13 failed-bullet", CLI, r'''        failed = bool(step.get("failed"))''', r'''        failed = False'''),
]

#: every test file that touches the changed code (a narrower set manufactures false survivors)
PINS = ["tests/test_4uy_progress.py", "tests/test_4ut_progress.py", "tests/test_4mp_reply_tap.py",
        "tests/test_ghost_cli.py", "tests/test_4kp_think_split_and_clients.py", "tests/test_4rz_cli.py",
        "tests/test_4rb_reasoning_effort.py", "tests/test_4rz_web_client.py", "tests/test_4rn_turn_clock_browser.py",
        "tests/test_interface_batch3_2026_09_11.py", "tests/test_interface_batch2_2026_09_11.py",
        "tests/test_interface_face_palette.py", "tests/test_interface_fix_first_2026_09_11.py",
        "tests/test_interface_face_signals_2026_09_11.py", "tests/test_finalize_stream_pins.py",
        "tests/test_4kw_review_round3.py", "tests/test_stream_turn_usage.py", "tests/test_4sb_extra_step.py"]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
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
    py = [f for f in srcs if f.endswith(".py")]          # a .js/.css/CLI mutant has no coverage: its file set
    cov = BC.build_map(py, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: (BC.tests_for(cov, f, old, originals[f]) if f in py else None) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
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
