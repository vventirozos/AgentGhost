"""§4EC — `_finalize_and_return` CALL-SITE pins from the §R re-verification of
§4BZ (2026-09-02). §4BZ pinned its helpers in isolation; the whole-function
battery found the helpers' invocations and the surrounding conditions
deletable with every pin green (the §4BY 'the helper was unit-tested, its
invocation was not' lesson, one slice over). Driven through the real method
with the round-1 harness."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.test_finalize_stream_pins import _fs, make_fin_agent


@pytest.fixture(autouse=True)
def _real_memory_dir(monkeypatch, tmp_path):
    """The shared harness leaves `context.memory_dir` a MagicMock, whose parent
    resolves to "." — finalize's watermark files then land in the REPO ROOT
    (the 13/20-byte `*_digest.json` residue there is exactly that). Point it
    at tmp for every test in this file."""
    import tests.test_finalize_stream_pins as pins
    real = pins.make_fin_agent
    def _mk():
        a = real(); a.context.memory_dir = str(tmp_path / "mem"); return a
    monkeypatch.setattr(pins, "make_fin_agent", _mk)
    globals()["make_fin_agent"] = _mk
    yield
    globals()["make_fin_agent"] = real


async def _out(a, **over):
    out, _, _ = await a._finalize_and_return(_fs(**over))
    return out


# ── content scrubs at the head of finalize (L17745-17765) ────────────────────
class TestFinalContentScrubs:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("junk", [
        "--- EXECUTION RESULT ---\nexit 0\n------------------------\n",          # L17745
        "🟢 [task_1] do the thing\n",                                             # L17746
        "task_3\n[task_4]\n",                                                     # L17748
        "FOCUS TASK: build it\nPLAN: step\nTHOUGHT: hmm\n",                       # L17749
        "\n".join(f"[task_{i}] deploy step (IN_PROGRESS)" for i in range(4)) + "\n",   # L17747 status run
    ])
    async def test_internal_artifacts_never_reach_the_reply(self, junk):
        a = make_fin_agent()
        out = await _out(a, final_ai_content="Real answer.\n" + junk + "More real text.")
        for token in ("EXECUTION RESULT", "🟢", "[task_", "task_3", "FOCUS TASK", "PLAN:", "THOUGHT:", "IN_PROGRESS"):
            assert token not in out, (junk, out)
        assert "Real answer." in out and "More real text." in out

    @pytest.mark.asyncio
    async def test_reply_is_stripped(self):
        a = make_fin_agent()
        assert (await _out(a, final_ai_content="\n\n  Answer.  \n\n")) .strip() == "Answer."
        assert not (await _out(a, final_ai_content="\n\n  Answer.  \n\n")).startswith("\n")


# ── the empty-reply fallback header (L18076-18170) ───────────────────────────
class TestFallbackHeaderCondition:
    TOOLS = [{"name": "execute", "content": "STDOUT/STDERR:\nok\nEXIT CODE: 0"}]

    @pytest.mark.asyncio
    async def test_header_only_when_the_reply_is_empty(self):
        a = make_fin_agent()
        out = await _out(a, final_ai_content="", tools_run_this_turn=self.TOOLS)
        assert out.startswith("Process finished successfully.") and "### Final Output:" in out
        assert "EXIT CODE: 0" in out                      # the preview carries the tool output
        assert (await _out(a, final_ai_content="ok", tools_run_this_turn=self.TOOLS)) == "ok"

    @pytest.mark.asyncio
    async def test_empty_reply_without_any_tool_is_the_plain_sentence(self):
        a = make_fin_agent()
        assert (await _out(a, final_ai_content="", tools_run_this_turn=[])) == "Task executed successfully."


# ── exit-banner SHAPE gate (§4BZ B-D1, L18118-18130) ────────────────────────
class TestExitBannerShape:
    @pytest.mark.asyncio
    async def test_framed_nonzero_exit_from_a_non_execute_tool_is_failed(self):
        a = make_fin_agent()
        tools = [{"name": "run_my_skill", "content": "STDOUT/STDERR:\nboom\nEXIT CODE: 1"}]
        out = await _out(a, final_ai_content="", tools_run_this_turn=tools)
        assert "FAILED" in out and "Process finished successfully." not in out

    @pytest.mark.asyncio
    async def test_sandbox_job_form_is_failed(self):
        a = make_fin_agent()
        tools = [{"name": "run_my_skill", "content": "[sandbox job 7 — EXIT CODE: 1]"}]
        out = await _out(a, final_ai_content="", tools_run_this_turn=tools)
        assert "FAILED" in out

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,content,failed", [
        # the three arms of `_exec_shaped`, each deciding alone:
        ("run_my_skill", "EXIT CODE: 1\nno framing at all", False),          # anchored code, NO framing
        ("run_my_skill", "STDOUT/STDERR:\nsee EXIT CODE: 1 in the notes", False),  # framed, code NOT line-anchored
        ("execute", "EXIT CODE: 1\nno framing at all", True),                # execute by NAME
        ("file_system", "notes: EXIT CODE: 1 means failure", False),        # exit code as data
    ])
    async def test_exec_shape_arms(self, name, content, failed):
        a = make_fin_agent()
        out = await _out(a, final_ai_content="", tools_run_this_turn=[{"name": name, "content": content}])
        assert ("FAILED" in out) is failed, (name, content, out[:80])


# ── verdict cache: HIT path (L18218-18238) ───────────────────────────────────
class TestVerdictCacheHit:
    @pytest.mark.asyncio
    async def test_a_fresh_matching_verdict_is_reused_not_recomputed(self):
        a = make_fin_agent(); a.context.verifier = None
        a._compute_verifier_verdict_gated = AsyncMock(return_value=(None, None))
        vr = MagicMock(); text = "delivered text"
        tools = [{"name": "execute", "content": "SUCCESS: ok"}]
        await a._finalize_and_return(_fs(final_ai_content=text, tools_run_this_turn=tools,
                                         _verdict_is_fresh=True,
                                         _verifier_verdict_cache=(vr, tools[0], hash(text))))
        assert not a._compute_verifier_verdict_gated.called, "cache hit still recomputed"

    @pytest.mark.asyncio
    async def test_a_stale_flag_or_missing_cache_recomputes(self):
        for fresh, cache in ((False, (MagicMock(), {"name": "execute", "content": "x"}, hash("delivered text"))),
                             (True, None)):
            a = make_fin_agent(); a.context.verifier = None
            a._compute_verifier_verdict_gated = AsyncMock(return_value=(None, None))
            await a._finalize_and_return(_fs(final_ai_content="delivered text",
                                             tools_run_this_turn=[{"name": "execute", "content": "SUCCESS: ok"}],
                                             _verdict_is_fresh=fresh, _verifier_verdict_cache=cache))
            assert a._compute_verifier_verdict_gated.called, (fresh, cache)


# ── clarify question + risk summary insertion (L18644-18660) ─────────────────
class TestClarifyAndRiskInsertion:
    def _agent(self, question, risk):
        a = make_fin_agent()
        tr = MagicMock()
        tr.should_ask_user = MagicMock(return_value=question)
        tr.get_risk_summary = MagicMock(return_value=risk)
        a.context.uncertainty_tracker = tr
        return a

    @pytest.mark.asyncio
    async def test_question_leads_and_risk_trails(self):
        a = self._agent("Which repo did you mean?", "RISK: two candidate repos")
        out = await _out(a, final_ai_content="Answer body here.")
        assert out.startswith("**Which repo did you mean?**")
        assert out.rstrip().endswith("RISK: two candidate repos")
        assert "Answer body here." in out

    @pytest.mark.asyncio
    async def test_already_present_text_is_not_duplicated(self):
        a = self._agent("Which repo did you mean?", "RISK: two candidate repos")
        body = "Which repo did you mean? I assumed A.\n\nRISK: two candidate repos are plausible."
        out = await _out(a, final_ai_content=body)
        assert out.count("Which repo did you mean?") == 1 and out.count("RISK: two candidate repos") == 1

    @pytest.mark.asyncio
    async def test_an_empty_reply_gets_the_placeholder_and_then_the_insertions(self):
        # by the time the tracker runs, an empty reply has become the placeholder
        # sentence (L18169), so the `final_ai_content` arm of the guard is never
        # false here — the question leads and the risk trails the placeholder.
        a = self._agent("Which repo?", "RISK: x")
        out = await _out(a, final_ai_content="")
        assert out.startswith("**Which repo?**") and "Task executed successfully." in out
        assert out.rstrip().endswith("RISK: x")


# ── fallback preview construction (L18075-18094) and banner framing (L18126-18150) ──
class TestFallbackPreview:
    @pytest.mark.asyncio
    async def test_preview_is_truncated_at_2000_chars_and_hint_stripped(self):
        a = make_fin_agent()
        body = "STDOUT/STDERR:\n" + ("q" * 2500) + "\nEXIT CODE: 0\n----\nDIAGNOSTIC HINT: try again"
        out = await _out(a, final_ai_content="", tools_run_this_turn=[{"name": "execute", "content": body}])
        assert "...[Truncated]" in out and "DIAGNOSTIC HINT" not in out
        assert out.count("q") == 2000   # the head line is stripped first; the body is cut at exactly 2,000 chars

    @pytest.mark.asyncio
    async def test_short_preview_is_kept_whole(self):
        a = make_fin_agent()
        body = "STDOUT/STDERR:\nshort\nEXIT CODE: 0"
        out = await _out(a, final_ai_content="", tools_run_this_turn=[{"name": "execute", "content": body}])
        assert "...[Truncated]" not in out and "short" in out

    @pytest.mark.asyncio
    async def test_execution_result_framing_counts_as_execute_shaped(self):
        # `EXIT CODE: 1` framed by the EXECUTION RESULT banner (no STDOUT/STDERR line)
        a = make_fin_agent()
        tools = [{"name": "run_my_skill", "content": "--- EXECUTION RESULT ---\nboom\nEXIT CODE: 1"}]
        out = await _out(a, final_ai_content="", tools_run_this_turn=tools)
        assert "FAILED" in out

    @pytest.mark.asyncio
    async def test_a_declared_failed_outcome_with_a_zero_exit_is_still_failed(self):
        from ghost_agent.tools.outcome import ToolOutcome
        a = make_fin_agent()
        res = ToolOutcome.failed("STDOUT/STDERR:\nlooked fine\nEXIT CODE: 0")
        out = await _out(a, final_ai_content="", tools_run_this_turn=[{"name": "execute", "content": res}])
        assert "FAILED" in out and "Process finished successfully." not in out


# ── project digest insertion (§4BZ A-F2 "both digests", L18729-18769) ────────
class TestProjectDigestInsertion:
    class _Dg:
        def __init__(self, has_content, new_event_id):
            self.has_content = has_content; self.new_event_id = new_event_id
            # the operator log line after the insert reads these; a digest
            # object without them raises INSIDE the try and the watermark
            # advance is lost (the same digest would repeat next turn)
            self.advanced = 2; self.needs_user = []; self.projects_touched = 1

    def _wire(self, monkeypatch, tmp_path, has_content=True, new_event_id=7, wm=3):
        import ghost_agent.core.project_digest as pd
        calls = {"summ": [], "saved": []}
        monkeypatch.setattr(pd, "summarize_since", lambda ps, since: (calls["summ"].append(since), self._Dg(has_content, new_event_id))[1])
        monkeypatch.setattr(pd, "render_digest", lambda dg: "PROJECT DIGEST: 2 tasks moved")
        monkeypatch.setattr(pd, "load_watermark", lambda p: wm)
        monkeypatch.setattr(pd, "save_watermark", lambda p, v: calls["saved"].append(v))
        a = make_fin_agent()
        a.context.project_store = object()
        a.context.memory_dir = str(tmp_path / "mem")
        return a, calls

    @pytest.mark.asyncio
    async def test_new_events_are_digested_into_the_reply_and_the_watermark_advances(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path)
        out = await _out(a, final_ai_content="Answer body.")
        assert "PROJECT DIGEST: 2 tasks moved" in out and "Answer body." in out
        assert calls["summ"] == [3] and calls["saved"] == [7]

    @pytest.mark.asyncio
    async def test_no_new_events_means_no_digest_and_no_watermark_write(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path, has_content=False, new_event_id=3)
        out = await _out(a, final_ai_content="Answer body.")
        assert "PROJECT DIGEST" not in out and calls["saved"] == []

    @pytest.mark.asyncio
    async def test_a_missing_watermark_is_seeded_without_a_digest(self, monkeypatch, tmp_path):
        # first ever turn: baseline the watermark at the current head, show nothing
        a, calls = self._wire(monkeypatch, tmp_path, wm=None, new_event_id=9)
        out = await _out(a, final_ai_content="Answer body.")
        assert "PROJECT DIGEST" not in out
        assert calls["summ"][0] == 0 and calls["saved"] == [9]

    @pytest.mark.asyncio
    async def test_internal_requests_never_get_a_digest(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path)
        out = await _out(a, final_ai_content="Answer body.", req_id="sched-1234")
        assert "PROJECT DIGEST" not in out and calls["summ"] == []

    @pytest.mark.asyncio
    async def test_a_digest_already_in_the_reply_is_not_repeated(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path)
        out = await _out(a, final_ai_content="PROJECT DIGEST: 2 tasks moved\n\nAnswer body.")
        assert out.count("PROJECT DIGEST") == 1 and calls["saved"] == [7]


# ── activity digest insertion (the "mouth", L18862-18895) ────────────────────
class TestActivityDigestInsertion:
    def _wire(self, monkeypatch, tmp_path, records, offset_before=10, offset_after=12):
        import ghost_agent.core.autonomous_activity as aa
        calls = {"saved": [], "read_since": []}

        class _Log:
            def current_offset(self): return offset_after
            def read_since(self, off):
                calls["read_since"].append(off); return list(records), offset_after
        monkeypatch.setattr(aa, "get_activity_log", lambda ctx: _Log())
        def _render(recs, current_req_id="", severities=()):
            calls["severities"] = tuple(severities)
            return ("WHILE YOU WERE AWAY: %d event(s)" % len(recs)) if recs else ""
        monkeypatch.setattr(aa, "render_activity_digest", _render)
        monkeypatch.setattr(aa, "load_offset", lambda p: offset_before)
        monkeypatch.setattr(aa, "save_offset", lambda p, v: calls["saved"].append(v))
        a = make_fin_agent()
        a.context.memory_dir = str(tmp_path / "mem")
        return a, calls

    @pytest.mark.asyncio
    async def test_new_records_are_surfaced_and_the_offset_advances(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path, records=[{"k": 1}, {"k": 2}])
        out = await _out(a, final_ai_content="Answer body.")
        assert "WHILE YOU WERE AWAY: 2 event(s)" in out and "Answer body." in out
        assert calls["read_since"] == [10] and calls["saved"] == [12]
        import ghost_agent.core.autonomous_activity as aa
        assert calls["severities"] == (aa.SEVERITY_NOTIFY,)   # notify-only, the operator's 07-17 decision

    @pytest.mark.asyncio
    async def test_nothing_new_means_no_digest(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path, records=[], offset_before=12, offset_after=12)
        out = await _out(a, final_ai_content="Answer body.")
        assert "WHILE YOU WERE AWAY" not in out and calls["saved"] == []

    @pytest.mark.asyncio
    async def test_internal_requests_never_get_the_digest(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path, records=[{"k": 1}])
        out = await _out(a, final_ai_content="Answer body.", req_id="sched-1234")
        assert "WHILE YOU WERE AWAY" not in out and calls["read_since"] == []

    @pytest.mark.asyncio
    async def test_a_missing_offset_is_seeded_at_the_current_head_without_a_digest(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path, records=[{"k": 1}], offset_before=None, offset_after=12)
        out = await _out(a, final_ai_content="Answer body.")
        assert "WHILE YOU WERE AWAY" not in out and calls["saved"] == [12] and calls["read_since"] == []

    @pytest.mark.asyncio
    async def test_a_digest_already_in_the_reply_is_not_repeated(self, monkeypatch, tmp_path):
        a, calls = self._wire(monkeypatch, tmp_path, records=[{"k": 1}, {"k": 2}])
        out = await _out(a, final_ai_content="WHILE YOU WERE AWAY: 2 event(s)\n\nAnswer body.")
        assert out.count("WHILE YOU WERE AWAY") == 1 and calls["saved"] == [12]

    @pytest.mark.asyncio
    async def test_the_renderer_is_told_the_current_request_id(self, monkeypatch, tmp_path):
        import ghost_agent.core.autonomous_activity as aa
        a, calls = self._wire(monkeypatch, tmp_path, records=[{"k": 1}])
        seen = {}
        monkeypatch.setattr(aa, "render_activity_digest",
                            lambda recs, current_req_id="", severities=(): (seen.update(rid=current_req_id), "DG")[1])
        await _out(a, final_ai_content="Answer body.", req_id="ab12cd34")
        assert seen["rid"] == "ab12cd34"    # its OWN records are excluded by this id
