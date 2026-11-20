#!/usr/bin/env python3
r"""§4UY image battery — one image turn (live 76e8b1d0): the look line under the picture (one placement rule for
finalize, the token tap and the streamed final), no in-loop wait for a check that cannot repair, the verifier reads
the tool's look instead of a second look, tool results in call order + the block's real reason, the journal append
off the reply path.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uy_image.py [-jN] [all|rerun M1 M2 …]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = "src/ghost_agent/core/agent.py"; RT = "src/ghost_agent/core/reply_tap.py"
VF = "src/ghost_agent/core/verifier.py"; IG = "src/ghost_agent/tools/image_gen.py"
LG = "src/ghost_agent/utils/logging.py"
MUT = [
    ("CONTROL no-op", AG, r'''def place_owned_looks(text, rows, *, final: bool = True) -> str:''', r'''def place_owned_looks(text, rows, *, final: bool = True) -> str:  # no-op'''),
    ("CONTROL known-bad", AG, r'''        out: List[tuple] = []
        tail = len(body.rstrip(" \t\r\n"))''', r'''        return []
        out: List[tuple] = []
        tail = len(body.rstrip(" \t\r\n"))'''),
    # P — the look line under the picture (one rule, three paths)
    ("P1 always-at-end", AG, r'''            if be is not None and not _crosses(m.end(), be.start()):''', r'''            if False:'''),
    ("P2 no-blank-line", AG, r'''                out.append((be.start(), rank, "\n\n" + line + ("" if be.group().count("\n") > 1 else "\n"), line))''', r'''                out.append((be.start(), rank, "\n" + line + ("" if be.group().count("\n") > 1 else "\n"), line))'''),
    ("P3 prefix-places-at-end", AG, r'''            elif final and not _crosses(tail, tail):''', r'''            elif not _crosses(tail, tail):'''),
    ("P23 no-blank-before-opener", AG, r'''                out.append((be.start(), rank, "\n\n" + line + ("" if be.group().count("\n") > 1 else "\n"), line))''', r'''                out.append((be.start(), rank, "\n\n" + line, line))'''),
    ("S1 record-looks-after-notes", AG, r'''                        _treated_content = place_owned_looks_before_notes(_treated_content, _sent_rows, _post_notes)''', r'''                        _treated_content = place_owned_looks(_treated_content, _sent_rows)'''),
    ("S2 tag-looks-after-notes", AG, r'''                        _sv_shown = place_owned_looks_before_notes(
                            _sv_source, _sent_rows, [_lv_tag.get(_k) for _k in STREAM_POST_VIEW_NOTES])''', r'''                        _sv_shown = place_owned_looks(_sv_source, _sent_rows)'''),
    ("S3 notes-not-peeled", AG, r'''            if body.endswith(note):
                body, tail = body[:-len(note)], note + tail''', r'''            if False:
                body, tail = body[:-len(note)], note + tail'''),
    ("T1 tag-raw-stream-text", AG, r'''(f"{stream_conv_fp}|r{_reply_tag(_sv_shown)}"''', r'''(f"{stream_conv_fp}|r{_reply_tag(_sv_source)}"'''),
    ("C1 crlf-line-unseen", AG, r'''re.escape(line) + r"[ \t\r]*(?=\n|$)")''', r'''re.escape(line) + r"[ \t]*(?=\n|$)")'''),
    ("K1 reaper-a-service", "src/ghost_agent/main.py", r'''                               name="sandbox-job-reaper")   # a WRITER''', r'''                               name="sandbox-job-reaper", service=True)   # a WRITER'''),
    ("O5 tie-row-order", AG, r'''        out.sort(key=lambda r: (r[0], r[1]))   # stable: row order among equals''', r'''        out.sort(key=lambda r: r[0])'''),
    ("O6 end-before-trailing-block", AG, r'''        out = [((late if (rk[0] == 1 and p == tail) else p), rk, i, l) for p, rk, i, l in out]''', r'''        pass'''),
    ("P24 blank-line-only", AG, r'''_BLOCK_END_RE = re.compile(
    r"\r?\n(?:[ \t\r]*\n|(?=[ \t]{0,3}(?:#{1,6}[ \t]|```|~~~)|[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]))")''', r'''_BLOCK_END_RE = re.compile(r"\r?\n[ \t\r]*\n")'''),
    ("P25 no-list-opener", AG, r'''|[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]))")''', r'''))")'''),
    ("P26 hold-ignores-openers", AG, r'''                    if nl >= 0 and _MAYBE_OPENER_RE.fullmatch(body, nl + 1):''', r'''                    if False:'''),
    ("P27 fence-on-pic-line", AG, r'''                and "```" not in rest and "~~~" not in rest          # the picture's line opens a fence''', r'''                and True'''),
    ("R1 raw-placed", AG, r'''            if _sent_rows and isinstance(locals().get("_final_view"), str):
                _final_view = place_owned_looks(_final_view, _sent_rows)''', r'''            if _sent_rows:
                full_content = place_owned_looks(full_content, _sent_rows)
            if _sent_rows and isinstance(locals().get("_final_view"), str):
                _final_view = place_owned_looks(_final_view, _sent_rows)'''),
    ("R2 calls-ignored", AG, r'''        return [(m.start(), m.end() + (1 if m.group("eof") is not None else 0))
                for m in _MODULE_SCRUB_RE.finditer(body)]''', r'''        return []'''),
    ("R3 record-not-placed", AG, r'''                    if _sent_rows and isinstance(_treated_content, str):
                        # on the view, BEFORE''', r'''                    if False:
                        # on the view, BEFORE'''),
    ("P4 carried-twice", AG, r'''            if _owned_line_re(line).search(body):
                continue
            be = ''', r'''            be = '''),
    ("P13 under-the-link-line", AG, r'''            be = _BLOCK_END_RE.search(body, m.end()) if kind == "block" else None''', r'''            be = re.compile(r"\n").search(body, m.end()) if kind == "block" else None'''),
    ("P14 table-not-seen", AG, r'''    if "|" in own:''', r'''    if False:'''),
    ("P15 fence-not-seen", AG, r'''    if len(_FENCE_LINE_RE.findall(body, 0, ls)) % 2:''', r'''    if False:'''),
    ("P16 quote-not-seen", AG, r'''    if head.startswith(">"):''', r'''    if False:'''),
    ("P17 pipeless-table-not-seen", AG, r'''        if _TABLE_RULE_RE.search(blk) or _TABLE_RULE_RE.search(nxt):''', r'''        if False:'''),
    ("P18 any-link-anchors", AG, r'''    for m in _IMAGE_MD_LINK_RE.finditer(body):
        pics.setdefault(os.path.basename(m.group(1).split("?", 1)[0]), []).append(m)''', r'''    for m in _DOWNLOAD_LINK_RE.finditer(body):
        pics.setdefault(os.path.basename(m.group(2).split("?", 1)[0]), []).append(m)'''),
    ("P19 tap-no-hold", RT, r'''            safe_end = self._looks.hold(text, self._gen_sent, safe_end)''', r'''            pass'''),
    ("P20 stream-no-hold", AG, r'''                    _safe_view = _safe_view[:_look_placer.hold(_safe_view, _scrubbed_emitted_len, len(_safe_view))]''', r'''                    pass'''),
    ("P21 hold-never", AG, r'''                    return max(start, min(end, cut))''', r'''                    return end'''),
    ("P22 trim-views-raw", RT, r'''                views = [place_owned_looks(v, self._look_rows, final=False) for v in views]''', r'''                pass'''),
    ("P5 finalize-appends-at-end", AG, r'''        _ol_placed = _place_owned_look_lines(final_ai_content, tools_run_this_turn)''', r'''        _ol_placed = (final_ai_content or "").rstrip() + _owned_look_note(final_ai_content, tools_run_this_turn)'''),
    ("P6 tap-not-told", AG, r'''                        _tap.set_looks(_owned_look_rows(tools_run_this_turn))   # §4UY image-1''', r'''                        pass'''),
    ("P7 tap-ignores-looks", RT, r'''        piece = (self._looks.take(text, self._gen_sent, safe_end) if self._looks is not None''', r'''        piece = (self._looks.take(text, self._gen_sent, safe_end) if False'''),
    ("P8 stream-not-placed", AG, r'''                        _to_emit = _look_placer.take(_safe_view, _scrubbed_emitted_len, len(_safe_view))''', r'''                        pass'''),
    ("P9 stream-view-not-placed", AG, r'''                _final_view = place_owned_looks(_final_view, _sent_rows)''', r'''                pass'''),
    ("P10 placer-sends-twice", AG, r'''                if line in self.placed or pos < start or pos > end:''', r'''                if pos < start or pos > end:'''),
    ("P11 end-flush-not-final", AG, r'''                                                  max(_scrubbed_emitted_len, _tail_end), final=True)''', r'''                                                  max(_scrubbed_emitted_len, _tail_end), final=False)'''),
    ("P12 census-not-placed", AG, r'''                    _shown = _place_owned_look_lines(_shown, tools_run)          # §4QX / §4UY image-1''', r'''                    pass'''),
    # W — no wait for a check that cannot repair
    ("W1 image-turn-waits", AG, r'''    if tools_run is not None and _only_unrepairable_checks_armed(tools_run, pure_refutes):
        return False                                   # §4UY image-2''', r'''    pass'''),
    ("W2 site-passes-nothing", AG, r'''                                    tools_run=tools_run_this_turn,
                                    pure_refutes=_pure_refs)''', r'''                                    tools_run=None,
                                    pure_refutes=_pure_refs)'''),
    ("W7 pure-refutes-ignored", AG, r'''    if any(r is not None for r in (pure_refutes or ())):
        return False''', r'''    pass'''),
    ("W8 site-reads-no-arms", AG, r'''                                    if _only_unrepairable_checks_armed(tools_run_this_turn) else ())''', r'''                                    if False else ())'''),
    ("W9 arms-read-the-wrong-reply", AG, r'''                                    _judge_text, last_user_content, tools_run_this_turn,
                                    self._captured_project_id()).values())''', r'''                                    "", last_user_content, tools_run_this_turn,
                                    self._captured_project_id()).values())'''),
    ("W10 file-artifact-not-asked", AG, r'''            out["file-artifact"] = self._file_artifact_check(reply, tools_run, project_id)[1]''', r'''            out["file-artifact"] = None'''),
    ("W12 arm-error-skips", AG, r'''            out["file-artifact"] = True''', r'''            out["file-artifact"] = None'''),
    ("W3 judge-on-skips", AG, r'''    if verifier_judge_enabled():
        return False
    if any(r is not None''', r'''    if any(r is not None'''),
    ("W4 synthetic-counts", AG, r'''    real = [t for t in (tools_run or []) if isinstance(t, dict) and not t.get("_synthetic")]''', r'''    real = [t for t in (tools_run or []) if isinstance(t, dict)]'''),
    ("W5 any-image", AG, r'''    return bool(real) and all(str(t.get("name") or "").strip() == "image_generation" for t in real)''', r'''    return bool(real) and any(str(t.get("name") or "").strip() == "image_generation" for t in real)'''),
    ("W6 silent-timeout", AG, r'''                                        pretty_log("Verifier Gate",       # §4UY image-2: say the wait ran out''', r'''                                        (lambda *a, **k: None)("Verifier Gate",       # §4UY image-2: say the wait ran out'''),
    # L — the verifier reads the tool's look
    ("L1 always-a-second-look", VF, r'''        if not before_image and not likeness_subjects:
            # §4UY image-3''', r'''        if False:
            # §4UY image-3'''),
    ("L2 before-after-uses-look", VF, r'''        if not before_image and not likeness_subjects:
            # §4UY image-3''', r'''        if not likeness_subjects:
            # §4UY image-3'''),
    ("L3 look-not-recorded", IG, r'''            note_render_look(image_bytes, _seen)        # §4UY image-3: the verifier reads it, never looks again''', r'''            pass'''),
    ("L4 any-pictures-look", IG, r'''    return RENDER_LOOKS.get(_bytes_key(image_bytes), "") if image_bytes else ""''', r'''    return next(reversed(RENDER_LOOKS.values()), "") if RENDER_LOOKS else ""'''),
    ("L5 look-on-main", VF, r'''                    payload, use_critic=True, off_main_only=True, timeout=_VISUAL_CRITIC_TIMEOUT_S,
                    slot_wait=(_DEFERRED_SLOT_WAIT_S if _deferred else _VERIFY_SLOT_WAIT_S), **_kw)''', r'''                    payload, use_critic=True, timeout=_VISUAL_CRITIC_TIMEOUT_S,
                    slot_wait=(_DEFERRED_SLOT_WAIT_S if _deferred else _VERIFY_SLOT_WAIT_S), **_kw)'''),
    ("L6 no-pool-still-calls", VF, r'''        if not self.llm_client or not isinstance(clients, (list, tuple)) or not clients:''', r'''        if not self.llm_client:'''),
    ("L7 unbounded", IG, r'''    while len(RENDER_LOOKS) > RENDER_LOOKS_MAX:''', r'''    while False:'''),
    # O — results in call order; the block's reason
    ("O1 no-reorder", AG, r'''                order_batch_tool_messages(messages, _batch_msg_start, tool_calls)''', r'''                pass'''),
    ("O2 arrival-order", AG, r'''        ordered = sorted(groups, key=lambda g: (g[0], g[1]))''', r'''        ordered = sorted(groups, key=lambda g: g[1])'''),
    ("O3 steer-not-kept", AG, r'''            elif groups:
                groups[-1][2].append(m)''', r'''            elif False:
                groups[-1][2].append(m)'''),
    ("O4 dup-ids-first", AG, r'''                groups.append([idx[k] if k < len(idx) else len(tool_calls or []), len(groups), [m]])''', r'''                groups.append([idx[0] if idx else len(tool_calls or []), len(groups), [m]])'''),
    ("B1 reason-not-logged", AG, r'''                               + (getattr(_br_block, "why", "") or "a generated image from this request "''', r'''                               + ("" or "a generated image from this request "'''),
    ("B2 batch-reason-empty", AG, r'''                          why="a second image in the same batch — one image per batch")''', r'''                          why="")'''),
    # J — the journal append off the reply path
    ("J1 awaited-inline", AG, r'''        task = _spawn_journal(_write(), name=f"journal-{kind}")''', r'''        task = None
        await _write()'''),
    ("J2 order-not-kept", AG, r'''            if prev is not None and not prev.done():''', r'''            if False:'''),
    ("J4 unbounded-predecessor", AG, r'''                await asyncio.wait({prev}, timeout=timeout)''', r'''                await asyncio.wait({prev})'''),
    ("J5 drain-silent", "src/ghost_agent/utils/logging.py", r'''    if left:
        try:
            names''', r'''    if False:
        try:
            names'''),
    ("J7 services-not-cancelled", LG, r'''    for t in [t for t in list(_BG_SERVICE) if not t.done()]:
        t.cancel()''', r'''    pass'''),
    # (J8 'services-waited' dropped: equivalent — a cancelled service finishes on the next loop turn)
    ("J9 service-flag-dropped", LG, r'''    if service:
        _BG_SERVICE.add(task)''', r'''    pass'''),
    ("J10 keepalive-a-writer", "src/ghost_agent/main.py", r'''                      name="node-keepalive", service=True)''', r'''                      name="node-keepalive")'''),
    ("J6 drain-unnamed", "src/ghost_agent/utils/logging.py", r'''        task.set_name(name)      # the shutdown drain names what it leaves unwritten''', r'''        pass'''),
    ("J3 says-dropped", AG, r'''                    f"append('{kind}') still pending after {timeout:.0f}s — the write "''', r'''                    f"append('{kind}') exceeded {timeout:.0f}s — entry dropped "'''),
]
#: module-level/unmapped code: the tests that reach it
FORCE = {}
PINS = ["tests/test_4uy_image.py", "tests/test_4qx_look_owned.py", "tests/test_4mp_reply_tap.py",
        "tests/test_4lk_stream_path.py", "tests/test_stream_forced_final_retry.py", "tests/test_finalize_stream_pins.py",
        "tests/test_blind_regeneration_guard.py", "tests/test_probe_journal_never_teaches.py",
        "tests/test_core_loop_defect_fixes.py", "tests/test_journal.py", "tests/test_verifier_visual.py",
        "tests/test_4qu_check_critic.py", "tests/test_4qt_look_critic.py", "tests/test_4qp_render_look.py",
        "tests/test_critic_async.py", "tests/test_async_unverified_repair.py", "tests/test_verifier_auto_repair.py",
        "tests/test_dispatch_pipeline_extraction.py", "tests/test_4kd_image_pipeline_review.py",
        "tests/test_agent_image_review_fixes.py", "tests/test_4lm_vision.py", "tests/test_4pb_image_review.py",
        "tests/test_image_gen_tool.py", "tests/test_4oz_subjects_collage.py", "tests/test_4mg_image_subjects.py",
        "tests/test_judged_text_is_delivered_text.py", "tests/test_synthetic_strike_ledger.py",
        "tests/test_4kl_member_capability.py", "tests/test_4kd_clarify_first_and_slack_lessons.py",
        "tests/test_spawn_bg.py", "tests/test_queued_messages.py", "tests/test_4ji_request_echo_narration.py",
        "tests/test_unread_source_refute.py", "tests/test_verifier_rungate_informational.py"]


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
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | {f"tests/test_4uy_image.py::{t}" for t in FORCE[label]})
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
