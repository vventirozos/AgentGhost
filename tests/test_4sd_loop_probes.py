"""§4SD — the thinking-loop probes once the 16k cap and the 200K-char abort are gone.

Operator 10-29: "now with all these safeguards remove, we need to evaluate the
n-gram loop probe, make sure that it's good". A 64k-token turn can draft a
whole file in its thinking, so the probes were replayed at the stream's
cadence over 9,103 healthy logged turns, 52 killed ones, 411 real source
files and eight synthetic loop shapes. What that showed, and the world each
pin below fails in:

  (1) the line probe killed real source drafted whole (35 of 411 files:
      separator comments, a repeated `return "Error: …"`) — only TEXT lines
      (unindented, 4+ words) count again;
  (2) a code block written out again and again with a period past 2,000
      chars was caught late (the line probe, 9.5K–19K chars) or never once
      (1) is fixed — `_detect_block_loop` is missing, too loose (two drafts,
      three revised drafts, sibling functions), or not wired;
  (3) the n-gram probe fired on three sibling functions drafted in thinking
      (core/llm.py's client builders) — a code-shaped repeat must be the same
      block; a prose loop with varied filler must still die;
  (4) two sentence frames taking turns were never caught.
"""
import json
from pathlib import Path

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import stream_guards as SG

ROOT = Path(__file__).resolve().parents[1]
EVERY = SG.THINKING_LOOP_PROBE_EVERY
PROBES = (SG._detect_thinking_loop, SG._detect_paragraph_loop,
          SG._detect_sentence_run_loop, SG._detect_block_loop)


def _first_fire(text, detector, every=EVERY):
    """Where a probe run on the stream's cadence first fires (0 = never)."""
    i = every
    while True:
        if detector(text[:i]):
            return i
        if i >= len(text):
            return 0
        i = min(len(text), i + every)


def _any_fire(text):
    return {d.__name__: at for d in PROBES for at in [_first_fire(text, d)] if at}


LEAD = "".join(f"Step {i}: the window manager keeps a z-order list and redraws on focus change.\n"
               for i in range(30))


def _code_block(n_lines, seed, tag=""):
    import random
    r = random.Random(seed)
    words = "file window desktop icon state render update event handler module style layout".split()
    return "".join(f"  {r.choice(words)}_{r.randint(0, 999)}{tag}: function(a) {{ return a.{r.choice(words)} + "
                   f"{r.randint(0, 99)}; }},\n" for _ in range(n_lines))


# ══ (1) the line probe reads text lines only ═════════════════════════════════

@pytest.mark.parametrize("line", [
    "/* ══════════════════════════════════════════════════════════════════ */",
    "# --------------------------------------------------------------------------",
    "        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'fetch' },",
    '        return "Error: Background task scheduling is disabled or not available here."',
    "    // Re-render the window list so the dock reflects the new focus order now",
])
def test_a_code_line_repeated_in_a_draft_is_not_a_loop(line):
    # world: every line ≥ 48 chars counts, so six separators in a drafted file
    # abort the turn (35 of 411 real files did)
    body = "".join(f"function f{i}(a, b) {{\n    const v{i} = compute(a, b, {i});\n{line}\n    return v{i};\n}}\n"
                   for i in range(60))
    draft = "Let me draft the file now, then I will write it with the tool.\n" + body
    assert len(draft) > SG.PARAGRAPH_LOOP_MIN_BUF
    assert _first_fire(draft, SG._detect_paragraph_loop) == 0
    assert _any_fire(draft) == {}


def test_a_text_line_loop_still_dies_and_code_between_does_not_hide_it():
    # world: the text-line filter also drops prose, or a non-text line between
    # the repeats ends the scan
    line = "I will write the response now and I will not use any more tools for this."
    text = "x" * SG.PARAGRAPH_LOOP_MIN_BUF + "\n" + "".join(
        f"{line}\n    const pad{i} = {i};\n" for i in range(SG.PARAGRAPH_LOOP_THRESHOLD)) + "tail"
    assert SG._detect_paragraph_loop(text) is True
    # one fewer repeat: silent (the count is the threshold, not "any repeat")
    short = text.replace(f"{line}\n    const pad0 = 0;\n", "", 1)
    assert SG._detect_paragraph_loop(short) is False


@pytest.mark.parametrize("raw,want", [
    ("I will write the file now.", True),
    ("Arguments: path: /macro_probe.py content: print(6*7)", True),     # a logged loop's line (req 9916a96f)
    ("    I will write the file now.", False),                          # indented: a code body
    ("\tI will write the file now.", False),
    ("/* ═════════════════════════ */", False),
    ("I will do it.", False),                                           # three words
])
def test_what_a_text_line_is(raw, want):
    assert SG._is_text_line(raw) is want


def test_real_source_drafted_whole_fires_no_probe():
    # world: any probe kills a real file drafted in thinking — the four that
    # fired the old probes (line probe: tasks.py, style.css; n-gram: llm.py)
    # (core/llm.py's three sibling client builders still fire the n-gram
    # probe — a gate for them cost real loops, see the n-gram section)
    for rel in ("src/ghost_agent/tools/tasks.py", "interface/static/style.css"):
        text = (ROOT / rel).read_text(encoding="utf-8")[:120_000]
        draft = "Let me draft the file now, then I will write it with the tool.\n" + text
        assert _any_fire(draft) == {}, rel


# ══ (2) the block probe ══════════════════════════════════════════════════════

@pytest.mark.parametrize("n_lines,within", [(40, 3), (75, 3)])
def test_a_code_block_written_over_and_over_dies_within_three_periods(n_lines, within):
    # world: no block probe — the 4K block ran to 19K chars before the line
    # probe (and with (1) it never dies)
    block = _code_block(n_lines, n_lines)
    assert len(block) > SG.THINKING_LOOP_MAX_GAP * SG.THINKING_LOOP_THRESHOLD + SG.THINKING_LOOP_WINDOW   # past the n-gram region
    text = LEAD + block * 20
    at = _first_fire(text, SG._detect_block_loop)
    assert 0 < at - len(LEAD) <= within * len(block) + EVERY, (at, len(block))
    assert _first_fire(text, SG._detect_thinking_loop) == 0


def test_a_redraft_with_one_line_changed_each_time_dies():
    # world: only verbatim periods count (BLOCK_LOOP_SIMILAR ignored)
    base = _code_block(36, 3)
    first = base.splitlines(keepends=True)
    drafts = "".join("".join(first[:5] + [f"  // pass {i}: same as before\n"] + first[6:]) for i in range(20))
    at = _first_fire(LEAD + drafts, SG._detect_block_loop)
    assert 0 < at - len(LEAD) <= 4 * len(base)


def test_two_drafts_of_a_file_are_not_a_loop():
    # world: two occurrences are enough
    base = _code_block(40, 5)
    text = LEAD + base + "Now let me fix the event handler and write it again.\n" + base
    assert _first_fire(text, SG._detect_block_loop) == 0


def test_three_revised_drafts_are_not_a_loop():
    # world: the similarity bar is low enough that real revision reads as a loop
    base = _code_block(40, 6).splitlines(keepends=True)
    tail = base[-6:]                                  # the shared tail the probe follows back
    drafts = []
    for d in range(3):
        mid = _code_block(34, 100 + d).splitlines(keepends=True)
        drafts.append("".join(base[:0] + mid + tail))
    text = LEAD + "\n".join(drafts)
    assert SG._detect_block_loop(text[:text.rindex(tail[-1]) + len(tail[-1])]) is False
    assert _first_fire(text, SG._detect_block_loop) == 0


def test_drafts_two_thirds_alike_are_revisions_not_a_loop():
    # world: the period bar is loose — a draft that keeps two thirds of its lines
    # and rewrites the rest is real work, not a repeat
    common = _code_block(20, 7).splitlines(keepends=True)
    tail = _code_block(6, 8).splitlines(keepends=True)
    drafts = ["".join(common + _code_block(14, 200 + d).splitlines(keepends=True) + tail)
              for d in range(3)]
    text = LEAD + "\n".join(drafts)
    assert SG._detect_block_loop(text) is False
    assert _first_fire(text, SG._detect_block_loop) == 0
    # the same three drafts unrevised are a loop (the test can see a fire)
    assert SG._detect_block_loop(LEAD + "\n".join([drafts[0]] * 3)) is True


def test_a_period_longer_than_the_bound_is_not_followed():
    # world: the look-back is unbounded (cost grows with the buffer)
    filler = "".join(f"Note {i}: the layout is fine and the icons render in their grid cells.\n"
                     for i in range(SG.BLOCK_LOOP_MAX_PERIOD // 60))
    block = _code_block(20, 9)
    text = block + filler + block + filler + block
    assert len(filler) > SG.BLOCK_LOOP_MAX_PERIOD - len(block)
    assert SG._detect_block_loop(text) is False
    assert SG._detect_block_loop(block * 3) is True


def test_a_short_period_varying_is_left_to_the_ngram_probe():
    # world: a one-line period that varies (a list being written) fires
    text = "".join(f"I will check the item number {i} in the list.\n" for i in range(400))
    assert _first_fire(text, SG._detect_block_loop) == 0


# ══ (3) the n-gram probe and code ════════════════════════════════════════════

_CLIENT = """                client = httpx.AsyncClient(
                    base_url=node["url"],
                    timeout=1200.0,
                    limits=limits,
                    proxy=get_proxy(node["url"]),
                    trust_env=False,
                    follow_redirects=True,
                    http2=False
                )
                self.{pool}_clients.append({{
                    "client": client,
                    "url": node["url"],
                    "model": node["model"]
                }})

        self.{nxt}_clients = []
        self._{nxt}_index = 0

        if {nxt}_nodes:
            for node in {nxt}_nodes:
"""


@pytest.mark.parametrize("make", [
    lambda i: (f"Attempt {i}:\n    records = load_records(workspace_path, encoding='utf-8')\n"
               "    parsed = [parse_record(r) for r in records if r.strip()]\n"
               "    check_schema(parsed, required=('id', 'name', 'value'))\n"
               "    save_records(parsed, workspace_path + '.out')\n"),
    lambda i: ("def build():\n    # draft %d\n    window_manager = WindowManager(desktop, theme='dark')\n"
               "    window_manager.register_handlers(on_focus=focus_window, on_close=close_window)\n"
               "    window_manager.restore_layout(saved_layout_path, fallback='grid', animate=True)\n"
               "    log('iteration %d')\n    dock = Dock(window_manager, position='bottom', autohide=False)\n"
               "    return window_manager, dock\n\n" % (i, i)),
])
def test_a_short_code_loop_with_a_moving_counter_dies(make):
    # world: a code-shaped repeat must be the same block (the §4SD gate,
    # reverted in review) — these ran to the output cap
    text = "".join(make(i) for i in range(300))
    assert 0 < _first_fire(text, SG._detect_thinking_loop) <= 8000


def test_the_same_code_block_three_times_is_still_a_loop():
    # world: the code gate drops repeats that ARE the same block
    block = _CLIENT.format(pool="worker", nxt="vision")
    assert SG._detect_thinking_loop("x\n" + block * 3) is True


def test_a_prose_loop_with_varied_filler_still_dies():
    # world: the gate also compares prose periods (req 2e52d2d2 escaped it)
    a = ("Let me try to output the sorted nodes for the graph in the sandbox. If the graph has a cycle, "
         "I'll output 'CYCLE DETECTED'. If it's acyclic, I'll output the sorted nodes in Kahn order, one per line, as the task asks.\n")
    assert len(a) > SG.THINKING_LOOP_WINDOW
    fill = ["Actually, the validator's graph may differ from the sandbox one.\n",
            "Wait, the expected output is a newline, which is odd.\n",
            "Let me write a robust solution with Kahn's algorithm.\n"]
    text = "".join(a + fill[i % 3] for i in range(6)) + a
    assert SG._detect_thinking_loop(text) is True


def test_a_greek_prose_loop_dies():
    # world: text lines are ASCII words only — Greek reasoning read as code
    s1 = "Πρέπει να γράψω το αρχείο τώρα και να ελέγξω ξανά το αποτέλεσμα της εντολής."
    assert SG._is_text_line(s1)
    text = "x" * SG.PARAGRAPH_LOOP_MIN_BUF + "\n" + "".join(
        f"{s1}\nΠερίμενε, κάτι {i};\n" for i in range(SG.PARAGRAPH_LOOP_THRESHOLD)) + "…"
    assert SG._detect_paragraph_loop(text) is True


@pytest.mark.parametrize("raw,want", [
    ("  - I will check the window manager state again now", True),
    ("    * I will check the window manager state again now", False),   # JSDoc / C comment shape
    ("   * @returns {Promise<object>} resolves when the request completes", False),
    ("      - name: Install the Python dependencies for this job", False),  # a YAML list item
    ("  - Note: I will check the window manager state again now", False),  # reads as a key: the
                                                                            # n-gram/block probes still see it
    ("  - I note: the window manager state is checked again now", True),
    ("  2. I will check the window manager state again now", True),
    ("    return a - b - c - d - e", False),
    ("    -webkit-transition: all 0.2s ease-in-out;", False),
])
def test_an_indented_bullet_is_text(raw, want):
    # world: an indented markdown plan can loop unseen by the line probe
    assert SG._is_text_line(raw) is want


@pytest.mark.parametrize("run", ["=" * 260, " " * 230, "0, " * 120, "|---" * 70, "A" * 300])
def test_a_short_unit_run_is_not_a_block_loop(run):
    # world: overlapping matches — a 260-char banner of `=` aborted the turn
    text = LEAD + "data = [" + run
    assert SG._detect_block_loop(text) is False
    assert SG._detect_block_loop(LEAD + "x" * 3 + "=" * 700) is True     # a run past 600 still dies


def test_period_comparison_is_linear_on_repeated_lines():
    # world: a line-aligned diff — 7 s for a 20K period of `}` lines (review)
    import time
    a = "}\n" * 30_000
    b = "}\n" * 29_999 + "{\n"
    t = time.perf_counter()
    assert SG._periods_match(a, b) is True
    assert time.perf_counter() - t < 0.05


# ══ (4) two frames taking turns ══════════════════════════════════════════════

def test_two_frames_alternating_die():
    # world: only one-opener runs count (L6 ran unbounded)
    words = "file window icon state render layout".split()
    text = " ".join((f"I will check the {words[i % 6]} now." if i % 2 else f"Then I verify the {words[i % 5]} again.")
                    for i in range(200)) + " "
    at = _first_fire(text, SG._detect_sentence_run_loop)
    assert 0 < at <= 2500
    assert SG._alternating_frame_run(text) >= SG.SENTENCE_RUN_THRESHOLD


def test_alternation_needs_three_sentences_and_a_strict_turn():
    # world: any two different sentences count as an alternation, or a third
    # opener is skipped
    assert SG._alternating_frame_run("I will go. Then it ends. ") == 0
    assert SG._alternating_frame_run("I will go. Then it ends. I will stop. ") == 3
    assert SG._alternating_frame_run("I will go. Then it ends. So be it. I will go. Then it ends. ") == 0
    mixed = " ".join(["I will go.", "Then it ends.", "So we see."] * 20) + " "
    assert SG._detect_sentence_run_loop(mixed) is False


def test_one_frame_runs_are_unchanged():
    one = " ".join("I will check it now." for _ in range(SG.SENTENCE_RUN_THRESHOLD)) + " "
    assert SG._sentence_opener_run(one) == SG.SENTENCE_RUN_THRESHOLD
    assert SG._alternating_frame_run(one) == 0
    assert SG._detect_sentence_run_loop(one) is True


# ══ cost, and the wiring ═════════════════════════════════════════════════════

def test_the_block_probe_is_cheap_on_a_full_64k_turn():
    import time
    # real code as the stream (the web client's JS, not a pinned module's text)
    text = "".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "interface" / "static").glob("*.js")))
    text = (text * (1 + 220_000 // max(1, len(text))))[:220_000]
    t = time.perf_counter()
    for n in range(200_000, 220_001, EVERY):
        SG._detect_block_loop(text[:n])
        SG._detect_paragraph_loop(text[:n])
    per = (time.perf_counter() - t) / 41
    assert per < 0.02, per


async def test_the_turn_loop_kills_a_block_loop(monkeypatch, tmp_path):
    # world: `_detect_block_loop` exists but the turn loop never calls it
    import tests.test_4kv_slack12_fixes as K
    block = _code_block(30, 11)

    class _BlockModel(K._Model):
        def stream(self, payload, use_coding=False):
            step = self.script[len(self.payloads)]
            if step[0] != "block":
                return super().stream(payload, use_coding)
            self.payloads.append(json.loads(json.dumps(payload)))
            me = len(self.payloads) - 1
            self.pulled.append(0)

            async def gen():
                for _ in range(200):
                    self.pulled[me] += 1
                    yield K._sse({"reasoning_content": block})
                yield b"data: [DONE]\n\n"
            return gen()
    monkeypatch.setattr(K, "_Model", _BlockModel)
    final, model, logged, rows, agent = await K._drive(monkeypatch, tmp_path, [
        ("block",), ("say", "Written."), ("say", "Written.")])
    assert model.pulled[0] <= 5, model.pulled
    assert any("repeated-block loop" in c for t, c, _ in logged if t == "Thinking Loop"), logged
    assert A._detect_block_loop is SG._detect_block_loop


# ── second review: lightly revised drafts, JSDoc, case walks ─────────────────

def _revised(code, frac, seed):
    import random
    r = random.Random(seed)
    ls = code.splitlines()
    for i in r.sample(range(len(ls)), max(1, int(len(ls) * frac))):
        ls[i] += f"  # tweak{r.randint(0, 999)}"
    return "\n".join(ls) + "\n"


def _drafts(n, frac):
    code = "".join(f"def handler_{i}(event, state):\n    value = state.get('k{i}', {i})\n"
                   f"    return render(event, value + {i * 3})\n\n" for i in range(90))
    d = [code]
    for i in range(n - 1):
        d.append(_revised(d[-1], frac, i))
    sep = "\n\nWait, I need to fix a few lines. Let me rewrite the whole file with the fix:\n\n"
    return "Plan: write the module.\n\n" + sep.join(d) + "\n\nOK done."


@pytest.mark.parametrize("frac", [0.01, 0.05, 0.08])
def test_three_drafts_revised_a_little_are_not_a_loop(frac):
    # world (review): three drafts of a real file, 1–8 % of lines changed each
    # time, were killed as a loop on 40 of 40 files
    assert _first_fire(_drafts(3, frac), SG._detect_block_loop) == 0


def test_four_near_identical_drafts_are_a_loop_and_three_verbatim_are():
    # world: alike periods are never enough, or verbatim ones need a fourth
    assert _first_fire(_drafts(4, 0.03), SG._detect_block_loop) > 0
    block = _code_block(40, 12)
    assert SG._detect_block_loop(LEAD + block * 3) is True
    assert SG._detect_block_loop(LEAD + block * 2) is False


def test_four_drafts_a_sixth_rewritten_are_revisions():
    # world: the period bar is loose (0.8 calls these a loop)
    assert _first_fire(_drafts(4, 0.17), SG._detect_block_loop) == 0


def test_a_jsdoc_module_drafted_whole_is_not_a_loop():
    # world (review): ` * @param` lines counted as prose and the paragraph probe
    # killed a 40-function module at 6,000 chars
    js = "Let me draft the client module in full first.\n\n" + "".join(
        f"/**\n * Fetch the resource number {i} from the remote server.\n"
        f" * @param {{string}} id the identifier of the resource to fetch\n"
        f" * @returns {{Promise<object>}} resolves when the request completes\n */\n"
        f"export async function fetch{i}(id) {{\n  return api.get(`/r{i}/${{id}}`);\n}}\n\n"
        for i in range(40))
    assert _any_fire(js) == {}


def test_walking_through_ten_cases_is_not_a_frame_loop():
    # world (review): "Input is X. Output is Y." for ten cases alternates
    # strictly — 20 sentences — and was killed at ~500 chars
    ws = "hello world python agent window icon state render layout banana".split()
    text = LEAD + " ".join(f"Input is {w}. Output is {w[::-1]}." for w in ws) + " "
    assert _first_fire(text, SG._detect_sentence_run_loop) == 0


def _alike_drafts(changed, n=4, lines=40, keep_tail=8):
    """``n`` drafts of a 40-line block, each changing ``changed`` lines of the
    one before (the last ``keep_tail`` stay, so the tail recurs)."""
    cur = [f"  value_{i:02d}: compute(alpha_{i}, beta_{i * 7 % 13}) + offset_{i % 5}," for i in range(lines)]
    out = []
    for d in range(n):
        if d:
            for j in range(changed):
                k = (d * changed + j) % (lines - keep_tail)
                cur[k] = cur[k] + f"  // rev{d}"
        out.append("\n".join(cur) + "\n")
    return "Let me write the table again.\n" + "".join(out)


def test_the_alike_bar_is_nine_tenths():
    # world: the bar drifts — 0.8 calls a 15 % rewrite a loop, 0.97 lets a
    # loop that changes 3 lines of 40 a pass run on
    assert SG._detect_block_loop(_alike_drafts(3)) is True          # 37/40 alike: a loop
    assert SG._detect_block_loop(_alike_drafts(6)) is False         # 34/40: revisions


def test_three_drafts_each_fixing_one_line_are_not_a_loop():
    # world (review round 4): the second draft fixes a line near the top, the
    # third one near the end — at a stream position between the two fixes the
    # newest two periods read verbatim, and a file-sized period was killed
    lines = [f"def handler_{i}(event, state):\n    return render(event, state.get('k{i}', {i}))\n"
             for i in range(120)]
    v1 = "".join(lines)
    v2 = v1.replace("state.get('k3', 3)", "state.get('k3', 33)")
    v3 = v2.replace("state.get('k110', 110)", "state.get('k110', 111)")
    sep = "\n\nWait, I need to fix a line. Let me rewrite the whole file with the fix:\n\n"
    text = "Plan: write the module.\n\n" + sep.join((v1, v2, v3))
    assert len(v1) > SG.BLOCK_LOOP_VERBATIM_MAX_PERIOD
    assert _first_fire(text, SG._detect_block_loop) == 0
    short = _code_block(30, 21)                          # a short block, verbatim ×3: still a loop
    assert len(short) < SG.BLOCK_LOOP_VERBATIM_MAX_PERIOD
    assert SG._detect_block_loop(LEAD + short * 3) is True


def test_a_bullets_spacing_does_not_change_what_it_is():
    # world (review round 4): `\s+` backtracked, so "-  Note:" was text and "- Note:" was not
    assert SG._is_text_line("  - Note: the window keeps a focus list") is \
        SG._is_text_line("  -  Note: the window keeps a focus list") is False


def test_short_drafts_with_reasoning_between_them_never_fire():
    # world (review round 5): the verbatim shortcut kills a sub-4K file redrafted
    # with a one-line fix each time. Accepted residue: it needs the SAME words
    # between the drafts (14/40 seeds); with the reasoning between them saying
    # what changed — the way the model writes — it is 0/40 at every size. The
    # shortcut stays: the three logged loops it alone catches would each run to
    # the 64k cap (~70 min) without it.
    import random
    for seed in range(40):
        r = random.Random(seed)
        lines = [f"def handler_{i}(event):\n    return render(event, {i})\n" for i in range(35)]
        v1 = "".join(lines)
        i, j = r.sample(range(35), 2)
        v2 = v1.replace(f"render(event, {i})\n", f"render(event, -{i})\n")
        v3 = v2.replace(f"render(event, {j})\n", f"render(event, -{j})\n")
        text = ("Plan: write the module.\n\n" + v1
                + f"\n\nWait, handler_{i} must negate its value. Rewriting:\n\n" + v2
                + f"\n\nAnd handler_{j} too. Rewriting once more:\n\n" + v3)
        assert len(v1) < SG.BLOCK_LOOP_VERBATIM_MAX_PERIOD
        assert _first_fire(text, SG._detect_block_loop) == 0, seed


def test_a_bullet_followed_by_a_unicode_space_is_text():
    # world (review round 5): only space/tab counted after the bullet mark
    assert SG._is_text_line("  -\u00a0the window keeps a focus list now") is True
    assert SG._is_text_line("  -\u2003the window keeps a focus list now") is True
