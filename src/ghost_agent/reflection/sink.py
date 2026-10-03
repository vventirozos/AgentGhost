"""The Reflector's sink: every reflected trajectory goes to the JSONL corpus
and, when SkillMemory is wired, becomes lessons — the corrected plan,
REQUEST-scoped, and (from a CONFIRMED plan only) a general rule.

Moved out of ``main.lifespan`` (§4KW fourth review) so the decisions are
driven by tests, not read as source text."""
import logging

logger = logging.getLogger("GhostAgent")


def make_reflection_sink(_traj_collector, _skill_memory, _vector_memory):
    """The composite sink the Reflector is handed."""

    def _reflection_sink(reflected_trajectory):
        # 1. Always append to the JSONL log.
        try:
            _traj_collector.append(reflected_trajectory)
        except Exception as e:
            logger.warning(f"reflection JSONL sink failed: {e}")

        # 2. If SkillMemory is wired, also write the reflection as
        # a lesson. The skill store already dedupes via vector
        # distance, so repeat reflections on the same failure mode
        # don't flood the playbook.
        if _skill_memory is None:
            return
        src_reason = reflected_trajectory.extra.get("source_failure_reason", "") or "failure"
        plan_text = reflected_trajectory.planning_output or reflected_trajectory.final_response
        # Tag the lesson with the ORIGINAL failed trajectory's
        # id (`reflected_from`), not the reflection's own id.
        # Rationale: this lesson is the corrective behaviour
        # for that source failure. If the source trajectory is
        # ever later un-promoted (false-positive correction
        # detected, manual override, etc.), the retraction
        # path scrubs both this lesson AND any opt-prot lesson
        # from the same source — keeping provenance unified
        # under one id per turn.
        src_traj_id = reflected_trajectory.extra.get("reflected_from", "") or ""
        # The plan judge's verdict must reach the LESSON, not just the
        # trajectory outcome. `Reflector` documents that a verified
        # plan "upgrades the outcome AND tags the lesson verified" —
        # it only ever did the first, so every reflection lesson was
        # written unverified: no +0.3 utility, unpinned by
        # `_trim_playbook_by_utility`, and prunable. Live evidence: 96
        # trajectories with plan_verified=True, 3 reflection lessons,
        # all verified=False.
        _plan_verified = bool(
            reflected_trajectory.extra.get("plan_verified") is True)
        # Fresh review (§4KW): a plan the judge REJECTED was written
        # anyway (verified=False) — "delete path=/*" came in this way,
        # from a refute of a CORRECT cleanup. Not judged (no judge, a
        # timeout) keeps the old behaviour; judged-and-rejected is not
        # a lesson.
        if (reflected_trajectory.extra.get("plan_verified") is False
                and not str(reflected_trajectory.extra.get("plan_verify_note") or "").startswith("no verdict")):
            logger.info("reflection → lesson skipped: the plan judge did not confirm it (%s)",
                        str(reflected_trajectory.extra.get("plan_verify_note") or "")[:120])
            return
        # §4KW: the corrected PLAN belongs to this one request —
        # written REQUEST-scoped (retrieved only when the request
        # comes back). Keyed on the request and retrieved by the
        # whole-lesson embedding, these plans reached 927 other
        # turns ("Directly answer 'Yes' … Professor Spiros Denaxas"
        # → "what is my name?"). The transferable part, when the
        # model states one in general terms, is a GENERAL lesson.
        _request = (reflected_trajectory.user_request or "")
        try:
            _skill_memory.learn_lesson(
                task=_request[:400],
                mistake=str(src_reason)[:400],
                solution=str(plan_text)[:1200],
                memory_system=_vector_memory,
                source_trajectory_id=str(src_traj_id),
                source="reflection",
                verified=_plan_verified,
                scope="request",
                source_request=_request[:4000],
            )
        except Exception as e:
            logger.warning(f"reflection → SkillMemory write failed: {e}")
        _general = reflected_trajectory.extra.get("general_lesson") or {}
        try:
            from ..memory.lesson_scope import is_general_text, is_general_trigger
            _sit = str(_general.get("situation") or "")
            _mis = str(_general.get("mistake") or "")
            # Only from a plan the judge CONFIRMED: a general rule
            # reaches every matching request, so it inherits the
            # reflection's correctness at full reach. Replayed live
            # (2026-10-02), a reflection on a turn the verifier had
            # refuted WRONGLY generalised "keep deleting until the
            # directory is empty".
            if not _plan_verified:
                _sit = ""
            _rule = str(_general.get("rule") or "")
            if (_sit and is_general_trigger(_sit, _request) and is_general_text(_mis, _request)
                    and is_general_text(_rule, _request)):   # the rule too (second review)
                _skill_memory.learn_lesson(
                    task=_sit,
                    # the GENERAL mistake (the failure reason names
                    # this case, and the whole lesson is embedded)
                    mistake=_mis,
                    solution=str(_general.get("rule") or "")[:600],
                    memory_system=_vector_memory,
                    source_trajectory_id=str(src_traj_id),
                    source="reflection",
                )
            elif _sit:
                logger.info("reflection → general lesson dropped: it restates the request "
                            "(%r)", _sit[:100])
        except Exception as e:
            logger.warning(f"reflection → general lesson write failed: {e}")

    return _reflection_sink


def parse_plan_verdict(content: str):
    """``(verified, note)`` from the plan judge's reply."""
    import re
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    # Verdict = the FIRST line's leading token, per the demanded
    # format. The old anywhere-substring scan false-verified
    # paraphrases like "cannot be considered CONFIRMED — it
    # ignores the failure cause" (no "REFUTED" present). A
    # non-conforming reply now falls back to a whole-content
    # scan that requires CONFIRMED to appear WITHOUT a nearby
    # negation, else fails closed.
    first = (lines[0].upper() if lines else "")
    # the demanded format is "VERDICT: CONFIRMED" — the prefix made
    # the first-line branches unreachable (second review)
    first = re.sub(r"^\W*VERDICT\W*", "", first)
    if first.startswith("CONFIRMED"):
        verified = True
    elif first.startswith("REFUTED"):
        verified = False
    else:
        up = content.upper()
        c_pos = up.find("CONFIRMED")
        _neg_window = up[max(0, c_pos - 60):c_pos]
        verified = (
            c_pos != -1
            and up.find("REFUTED") == -1
            and not any(n in _neg_window for n in
                        ("NOT ", "CANNOT", "CAN'T", "NEVER", "ISN'T"))
        )
    note = (lines[0] if lines else "no verdict")[:200]
    if not verified and "REFUTED" not in content.upper() and "CONFIRMED" not in content.upper():
        # neither verdict given: NOT a rejection (§4KW second
        # review — the sink drops a rejected plan)
        note = "no verdict: " + note
    return verified, note
