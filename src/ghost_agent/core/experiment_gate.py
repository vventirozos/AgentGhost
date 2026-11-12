"""§4ST: an experiment task closes DONE only on a model run that really happened.

Request b45ec99f (2026-11-01): three "Run Experiment … probe_model runs …"
tasks were built by the coding executor, which WROTE the transcripts and the
scores; all three closed DONE. The §4QA hold that should have stopped them is
decided on WORDS (does the goal / the task name a model?), and three review
rounds of §4SS showed those words flip-flop between holding ordinary work and
missing real experiments.

This gate is decided on what RAN. ``probe_model`` keeps a byte copy of every
transcript it saves OUTSIDE the sandbox (``core.probe_transcripts``); nothing
else can make one — not the coding executor, not a file write. So:

* **Trigger** (when is a task "a model experiment" at close?) — either
  - the §4QA/§4SS detector says so (`is_model_experiment`), or
  - a file the task delivers READS like model output: a transcript heading or
    label ("## Transcripts", "**Transcript:**", "Model response:"). The fake
    b45ec99f files had both. (The bare word ``probe_model`` is NOT a shape: a
    protocol that plans the runs names it too.) This half is on
    the output, so a task worded so that the detector misses it ("repeat the
    ladder on the worker node; scoring via results/analyze.py") is still
    caught once it writes transcripts.
* **Backing** — the close's own words (its ``result``) or a delivered file
  CITES a transcript kept for this project (its saved path or file name,
  `probe_transcripts.cites`), directly or through one project file it names
  (an analysis citing ``results/exp1.md``, which cites the run transcript).

A triggered task with no backing is refused (owner's ``task_update``) or not
closed (autoadvance). For a false trigger — a protocol that only *describes*
transcripts, a survey of published chatbot studies — the owner's close can
pass ``override={"model_run": "<why>"}`` (formerly ``not_model_data``); the reason is stored with the task and in
the project ledger, so a waiver is never silent.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger("GhostAgent")

#: a delivered file that reads like model output
_OUTPUT_SHAPE_RE = re.compile(
    r"(?im)^[ \t>]*(?:#{1,6}[^\n]*\btranscripts?\b"                       # a heading naming transcripts
    r"|[*_]{0,2}[ \t]*(?:transcript|model[ \t]+(?:response|answer|output)s?)"
    r"[ \t]*[*_]{0,2}[ \t]*:)")                                           # a "Transcript:" label
#: a file that claims to HOLD transcripts — what a refused build's
#: quarantine moves (a protocol's one worked "Model output:" example is
#: refused at close but stays where later tasks read it)
_TRANSCRIPT_CLAIM_RE = re.compile(
    r"(?im)^[ \t>]*[*_]{0,2}[ \t]*transcripts?[ \t]*[*_]{0,2}[ \t]*:")
#: r2: a LABEL only — a protocol's "## Transcripts" heading ("each run is
#: made with probe_model(save_to=…)") moved the task's own declared file
#: files the gate reads: text a write-up can be
_TEXT_SUFFIXES = (".md", ".txt", ".rst", ".csv", ".tsv", ".json", ".html", ".htm", ".tex", ".yaml", ".yml")
#: files whose SHAPE is judged — prose only (review r1 C1: a ``# load the
#: transcripts`` comment in .py, a ``transcript:`` YAML key or a typed field
#: read as a heading/label, and the build's source was quarantined)
_PROSE_SUFFIXES = (".md", ".txt", ".rst", ".html", ".htm", ".tex")
#: a "Model response:" label is model output on its own
_MODEL_LABEL_RE = re.compile(
    r"(?im)^[ \t>]*[*_]{0,2}[ \t]*model[ \t]+(?:response|answer|output)s?[ \t]*[*_]{0,2}[ \t]*:")
_MAX_READ = 512 * 1024
_MAX_FILES = 24
#: the shortest waiver reason the owner's close accepts
MIN_WAIVER_CHARS = 12


def _read(ws: Path, rel: str) -> Optional[str]:
    """The text of workspace file ``rel`` — never outside ``ws`` (a link out
    is not read), at most _MAX_READ bytes. None when unreadable."""
    try:
        rel = str(rel or "").strip().lstrip("/")
        if not rel or ".." in rel.split("/"):
            return None
        target = ws / rel
        if not target.is_file():
            return None
        # a link INSIDE the workspace is read like the file (delivering a
        # link must not hide a write-up); one leading OUT is never read
        real_ws = os.path.realpath(ws)
        if not os.path.realpath(target).startswith(real_ws + os.sep):
            return None
        with open(target, "rb") as fh:
            return fh.read(_MAX_READ).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _named_text_files(ws: Path, texts: Iterable[str]) -> List[str]:
    """Existing text files of the workspace that ``texts`` name by path or
    basename (the §4SS token rule)."""
    from .workspace_cleanup import _named_tokens
    names, paths = _named_tokens(texts)
    out: List[str] = []
    try:
        for dirpath, _dirs, files in os.walk(ws, topdown=True, followlinks=False):
            if dirpath == str(ws):
                _dirs[:] = [d for d in _dirs if d != QUARANTINE_DIR]     # a quarantined run is no data
            for f in files:
                low = f.lower()
                if not low.endswith(_TEXT_SUFFIXES):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), ws).replace(os.sep, "/")
                if rel.lower() in paths or low in names:
                    out.append(rel)
                    if len(out) >= _MAX_FILES:
                        return out
    except OSError:
        pass
    return out


#: §4UI: a task the PLAN declared one of these is not judged a model
#: experiment by its words (a methods write-up, a research brief about
#: experiments) when it declares a PROSE deliverable — what it WRITES is still
#: judged (transcript-shaped files). A non-prose or missing file keeps the words.
#: Not "build": a declared build that runs prompts on a model is the fake the
#: gate exists for (r1 MAJOR 2's pin)
NOT_EXPERIMENT_KINDS = frozenset({"write", "research"})
#: the plain-text deliverables that exemption needs (not .html: tags hide the
#: label lines the shape check reads)
_EXEMPT_PROSE_SUFFIXES = (".md", ".txt", ".rst", ".tex")
#: a file path in a result text: a name with a slash or an extension of 2–8
#: letters/digits ("results/answers.jsonl", "out.csv")
_RESULT_PATH_RE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w-]+\.[A-Za-z][A-Za-z0-9]{1,7})(?![\w/])")


def planned_kind(store, project_id: str, task_id: str, kind: str) -> str:
    """The task's kind as the gate may trust it: a kind the plan declared, or
    ``model_run`` however it was set. A non-experiment kind set LATER by
    task_update (a ``kind_changed`` event) is not trusted — r1 MAJOR 2: a
    retype must never remove the gate. Never raises ("" on error)."""
    k = str(kind or "").strip()
    if not k or k == "model_run":
        return k
    try:
        if store.latest_task_event(project_id, task_id, ("kind_changed",)):
            return ""
    except Exception:  # noqa: BLE001
        return ""
    return k


def unrun_experiment_problem(store, project_id: str, description: str, *,
                             files: Sequence[str] = (), result_text: str = "",
                             kind: str = "") -> Optional[str]:
    """Why this task may not close DONE — it is a model experiment (by its
    words or by what it wrote) and nothing it delivers cites a kept
    ``probe_model`` transcript — else None. Never raises (None on error: the
    other close gates still apply)."""
    return unrun_experiment_check(store, project_id, description, files=files,
                                  result_text=result_text, kind=kind)[0]


def unrun_experiment_check(store, project_id: str, description: str, *,
                           files: Sequence[str] = (), result_text: str = "",
                           kind: str = ""):
    """(problem, claimed files): the problem as above, and the delivered files
    that claim to hold transcripts (what `quarantine` moves). (None, []) when
    the close may stand. Never raises."""
    try:
        from . import probe_transcripts as PT
        from .project_advancer import is_model_experiment, _project_task_texts, _names_a_model
        from .task_deliverable import declared_deliverable
        ws = Path(store.ensure_workspace(project_id))
        desc = str(description or "")
        cand: List[str] = []
        # r1 C2: only files the close itself DELIVERED may be quarantined —
        # never a declared-only or result-named file (an input, the owner's)
        def _norm(r) -> str:
            r = str(r or "").strip().replace("\\", "/").lstrip("/")
            while r.startswith("./"):
                r = r[2:]
            return r
        own = {_norm(r) for r in (files or ())}
        for rel in list(files or ()) + [declared_deliverable(desc) or ""]:
            rel = _norm(rel)
            if rel and rel not in cand:
                cand.append(rel)
        # OUTPUTS only: what the close says it wrote. A file the description
        # names is often an INPUT ("Compile the PDF from research/protocol.md")
        for rel in _named_text_files(ws, [str(result_text or "")]):
            if rel not in cand:
                cand.append(rel)
        bodies = {rel: t for rel in cand[:_MAX_FILES] if (t := _read(ws, rel)) is not None}

        proj = store.get_project(project_id) or {}
        # §4SU Phase 2: a task DECLARED model_run is an experiment whatever its
        # words; another kind is still judged by them (r1 MAJOR 2: a kind
        # that removed the gate let task_update task_kind="build" skip it —
        # a false trigger has the waiver, recorded)
        # §4UI: a kind the plan declared not-an-experiment (see planned_kind —
        # callers pass the trusted kind) is not judged by its words
        # — but only when what it delivers is PROSE, where the transcript-shape
        # check below still reads it (§4UI r1 MAJOR 1: a "write" task that
        # records model answers into results/answers.json, or into nothing
        # but its result text, skipped every check)
        # (r2 m1: not .html — tags hide the line-anchored labels from the shape
        # check — and every candidate, also a file the result text names)
        _decl = _norm(declared_deliverable(desc) or "")
        # (r3 m1: and any path the result text names, whatever its suffix —
        # `_named_text_files` only collects known text suffixes that exist,
        # so "answers in results/answers.jsonl" never reached `cand`)
        _named_any = _RESULT_PATH_RE.findall(str(result_text or ""))
        _prose_only = bool(_decl) and all(
            r.lower().endswith(_EXEMPT_PROSE_SUFFIXES) for r in ([_decl] + cand + _named_any))
        _trusted_not_exp = kind in NOT_EXPERIMENT_KINDS and _prose_only
        by_words = (kind == "model_run"
                    or (not _trusted_not_exp
                        and is_model_experiment(desc, proj.get("goal"), _project_task_texts(store, project_id))))
        # r1 C1: a transcript heading/label is a MODEL's output only in a
        # file that names a model or probe_model (meeting, interview, podcast
        # and court transcripts are ordinary work); prose files only
        shaped = sorted(rel for rel, t in bodies.items()
                        if rel.lower().endswith(_PROSE_SUFFIXES)
                        and (_MODEL_LABEL_RE.search(t)
                             or (_OUTPUT_SHAPE_RE.search(t) and _names_a_model(t))))
        if not (by_words or shaped):
            return None, []

        rec = PT.recorded(store.memory_dir, project_id)
        kept = [rel for rel, _t in rec]
        # a delivered file that IS a kept transcript (probe_model save_to)
        # backs itself — by its BYTES (r1 C3: a fake written over a kept
        # transcript's path passed on the path alone)
        if any(bodies.get(rel) == text for rel, text in rec):
            return None, []

        def _cites(text: str) -> bool:
            return any(PT.cites(text, k) for k in kept)

        if kept:
            if _cites(str(result_text or "")) or any(_cites(t) for t in bodies.values()):
                return None, []
            # one hop: a project file the deliverable names, which cites a transcript
            hop = _named_text_files(ws, list(bodies.values()) + [str(result_text or "")])
            for rel in hop:
                if rel in bodies:
                    continue
                t = _read(ws, rel)
                if t and _cites(t):
                    return None, []

        why = ("it is declared a model_run task" if kind == "model_run"
               else "its task text says a model is run" if by_words
               else f"{', '.join(shaped[:3])} reads like model output (a transcript heading or label)")
        have = (f"{len(kept)} probe transcript(s) are kept for this project, but nothing this task "
                f"delivers cites one" if kept else "no probe_model transcript has been recorded for this project")
        return (f"this looks like a model experiment ({why}) and {have}. A model's answers come only "
                "from probe_model: run the prompts with probe_model(save_to=\"results/<new>.md\"), then "
                "write the results FROM that transcript and name its path in the results file or the "
                "close's result"), sorted(rel for rel in shaped if rel in own
                                         and _TRANSCRIPT_CLAIM_RE.search(bodies[rel]))
    except Exception:  # noqa: BLE001
        # r1 M4: production logs at INFO — a debug line hid a gate that was off
        logger.warning("model run gate skipped (the close stands)", exc_info=True)
        return None, []


#: where a refused build's model-output-shaped files go (§4QA-c precedent)
QUARANTINE_DIR = "simulated"


def quarantine(store, project_id: str, rels: Sequence[str]) -> List[str]:
    """Move each workspace file in ``rels`` to ``simulated/<rel>`` so no later
    task reads a written-up "run" as data — nothing is deleted. Returns the
    new paths. Never raises; never follows a symlink; never leaves the
    workspace."""
    moved: List[str] = []
    try:
        ws = Path(store.ensure_workspace(project_id))
        real_ws = os.path.realpath(ws)
    except Exception:  # noqa: BLE001
        return moved
    for rel in rels or ():
        try:
            rel = str(rel or "").strip().lstrip("/")
            if not rel or ".." in rel.split("/") or rel.split("/")[0] == QUARANTINE_DIR:
                continue
            src = ws / rel
            if os.path.islink(src) or not src.is_file():
                continue
            if not os.path.realpath(src).startswith(real_ws + os.sep):
                continue
            dest_rel = f"{QUARANTINE_DIR}/{rel}"
            n = 1
            head, _sl, base = rel.rpartition("/")
            while (ws / dest_rel).exists():
                n += 1
                # the FILE name gets the suffix (r1: "results/v1.0/notes" had
                # its folder renamed)
                stem, dot, ext = base.rpartition(".")
                nb = f"{stem}.{n}.{ext}" if dot and stem else f"{base}.{n}"
                dest_rel = f"{QUARANTINE_DIR}/{head}/{nb}" if head else f"{QUARANTINE_DIR}/{nb}"
            dest = ws / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not os.path.realpath(dest.parent).startswith(real_ws + os.sep):
                continue
            os.replace(src, dest)
            moved.append(dest_rel)
            # the record follows the move (audit: a stale artifact row kept
            # the fake listed as a deliverable, its manifest line as fact)
            try:
                store.unregister_file(project_id, rel)
            except Exception:  # noqa: BLE001
                logger.warning("quarantine: %s moved but its record is stale", rel, exc_info=True)
        except OSError:
            logger.debug("quarantine of %s skipped", rel, exc_info=True)
    return moved


__all__ = ["unrun_experiment_problem", "unrun_experiment_check", "quarantine",
           "QUARANTINE_DIR", "MIN_WAIVER_CHARS"]
