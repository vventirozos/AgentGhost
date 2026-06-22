"""Auto-research subsystem for long-term projects.

Projects are not just code — they can be any multi-session effort. This
module lets the agent research topics relevant to the active project,
summarise the findings into a durable Markdown brief inside the project's
own workspace, register the brief as an artifact + event, and maintain a
lightweight ``research_index`` in project metadata so the briefing can
surface "what you've already researched" on every turn.

Three entry points:
  * ``research_topic``    — research ONE explicit topic (search → summarise
                            → persist).
  * ``research_project``  — research SEVERAL topics; when none are given it
                            auto-derives them from the project goal + open
                            tasks via :func:`propose_topics`.
  * ``persist_research_from_output`` — used by the autoadvancer to turn a
                            search result it already fetched into a durable
                            brief without spending a second search.

Everything degrades gracefully: no LLM client → a heuristic summary; no
search runner → a brief noting that; no workspace → a clean error result.
A research failure must never crash the caller (tool, advancer, or HTTP).
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

RESEARCH_SUBDIR = "research"
INDEX_META_KEY = "research_index"
MAX_TOPICS_DEFAULT = 5
_MAX_INDEX_ENTRIES = 50  # cap the metadata index so it can't grow unbounded

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

_OPEN_TASK_STATUSES = {"PENDING", "READY", "IN_PROGRESS", "PAUSED", "NEEDS_USER"}


@dataclass
class ResearchResult:
    """Outcome of a single research run.

    ``path`` is workspace-relative (e.g. ``research/foo.md``) so it reads
    cleanly in briefings and matches how the model references files while a
    project is active. ``ok`` is False only on a genuine failure (no
    workspace / write error) — a topic that simply turned up no sources
    still succeeds and writes a brief saying so.
    """

    ok: bool
    topic: str
    slug: str = ""
    path: str = ""
    abs_path: str = ""
    summary: str = ""
    sources: List[str] = field(default_factory=list)
    error: str = ""


# ------------------------------------------------------------------ helpers

def _slugify(topic: str, maxlen: int = 60) -> str:
    """Filesystem-safe slug. Alphanumerics + dashes only, so a topic can
    never escape the research dir via path traversal."""
    s = re.sub(r"[^a-z0-9]+", "-", (topic or "").strip().lower()).strip("-")
    s = s[:maxlen].strip("-")
    return s or "topic"


def _extract_sources(search_output: str) -> List[str]:
    """Pull the ``[Source: url]`` markers the search tool emits."""
    if not search_output:
        return []
    seen: List[str] = []
    for m in re.findall(r"\[Source:\s*([^\]]+)\]", search_output):
        u = m.strip()
        if u and u != "#" and u not in seen:
            seen.append(u)
    return seen


def _first_line(text: str) -> str:
    """First meaningful prose line — skips Markdown headings (e.g. the
    '## Summary' the summary starts with) and bullet markers so the index
    preview is a real sentence, not a section title."""
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        s = s.lstrip("-*").strip()
        if s:
            return s
    return ""


def _research_dir(store, project_id: str) -> Optional[Path]:
    ws = store.ensure_workspace(project_id)
    if ws is None:
        return None
    d = Path(ws) / RESEARCH_SUBDIR
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # pragma: no cover - exotic FS
        logger.warning("Could not create research dir %s: %s", d, e)
        return None
    return d


def _resolve_runner(context, tool_runner: Optional[ToolRunner]) -> Optional[ToolRunner]:
    """Use the caller's runner, else build one from the tool registry.

    Returns None when no registry is reachable (e.g. minimal test contexts
    without a real tool map) — callers then skip the search step.
    """
    if tool_runner is not None:
        return tool_runner
    try:
        from ..tools.registry import get_available_tools
        tmap = get_available_tools(context)
    except Exception:
        return None
    if not tmap:
        return None

    async def _run(name: str, args: Dict[str, Any]) -> str:
        handler = tmap.get(name)
        if not handler:
            return f"ERROR: tool {name} unavailable"
        return await handler(**args)

    return _run


def get_research_index(store, project_id: str) -> List[Dict[str, Any]]:
    """Return the project's research index (most-recent last)."""
    try:
        proj = store.get_project(project_id) or {}
    except Exception:
        return []
    idx = (proj.get("metadata") or {}).get(INDEX_META_KEY) or []
    return list(idx) if isinstance(idx, list) else []


def _heading_or_topic(text: str, fallback: str) -> str:
    """First Markdown heading in the brief (the agent's own title), else a
    de-slugified fallback. Bounded to keep the index line compact."""
    for ln in (text or "").splitlines()[:12]:
        s = ln.strip()
        if s.startswith("#"):
            t = s.lstrip("#").strip()
            # Strip the auto-generator's "Research: " prefix if present.
            t = re.sub(r"^research:\s*", "", t, flags=re.IGNORECASE)
            if t:
                return t[:80]
    return (fallback or "").replace("-", " ").replace("_", " ").strip()[:80] or "research"


def reconcile_research_dir(store, project_id: str) -> int:
    """Register any ``**/research/*.md`` brief written DIRECTLY (via a plain
    ``file_system`` write) that isn't already in the metadata index.

    The agent often saves a research brief with a bare ``file_system`` write
    instead of ``manage_projects action=research`` / the autoadvancer — and
    those direct writes never reached the index, so the brief was invisible to
    the briefing and the agent never re-read its own work (observed live:
    ``PetAI/research/transformer-from-scratch.md`` written once, never used).
    This scans the project workspace for research briefs anywhere (the agent
    may nest them under a self-named subdir) and indexes the new ones, keyed by
    PATH so a brief already indexed by the research tool is not duplicated.
    Returns the count newly indexed. Best-effort; never raises.
    """
    try:
        root = getattr(store, "sandbox_root", None)
        pid = str(project_id or "").strip().lower()
        if not root or not pid:
            return 0
        base = Path(root) / "projects" / pid
        if not base.is_dir():
            return 0
        known_paths = {e.get("path") for e in get_research_index(store, project_id)
                       if isinstance(e, dict)}
        added = 0
        import os
        for dirpath, _dirs, files in os.walk(base):
            # Only descend interest: a file is a research brief when one of its
            # parent directories is literally "research".
            rel_dir = Path(dirpath).relative_to(base).as_posix()
            parts = [p for p in rel_dir.split("/") if p]
            if "research" not in parts:
                continue
            for fn in sorted(files):
                if not fn.lower().endswith(".md") or fn.lower() == "index.md":
                    continue
                rel = (Path(dirpath) / fn).relative_to(base).as_posix()
                if rel in known_paths:
                    continue
                try:
                    text = (Path(dirpath) / fn).read_text(errors="replace")
                except OSError:
                    continue
                entry = {
                    "topic": _heading_or_topic(text, Path(fn).stem),
                    "slug": _slugify(rel[:-3]),          # path-based → unique
                    "path": rel,
                    "ts": (Path(dirpath) / fn).stat().st_mtime,
                    "summary_preview": _first_line(text)[:160],
                    "sources_count": 0,
                    "origin": "direct_write",
                }
                _upsert_index(store, project_id, entry)
                known_paths.add(rel)
                added += 1
        return added
    except Exception:
        logger.debug("reconcile_research_dir failed", exc_info=True)
        return 0


def _upsert_index(store, project_id: str, entry: Dict[str, Any]) -> None:
    """Insert/replace an index entry by slug, newest last, capped."""
    proj = store.get_project(project_id) or {}
    meta = dict(proj.get("metadata") or {})
    idx = [e for e in (meta.get(INDEX_META_KEY) or [])
           if isinstance(e, dict) and e.get("slug") != entry["slug"]]
    idx.append(entry)
    if len(idx) > _MAX_INDEX_ENTRIES:
        idx = idx[-_MAX_INDEX_ENTRIES:]
    meta[INDEX_META_KEY] = idx
    store.update_project(project_id, metadata=meta)


def _render_doc(topic: str, proj: Dict[str, Any], summary_md: str,
                sources: List[str], ts: float) -> str:
    iso = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Research: {topic}",
        "",
        f"_Project: {proj.get('title', '')} · generated {iso}_",
        "",
        (summary_md or "").strip(),
        "",
    ]
    if sources:
        lines.append("## Sources")
        lines.extend(f"- {s}" for s in sources)
        lines.append("")
    lines.append("---")
    lines.append("_Auto-generated by Ghost Agent project research._")
    return "\n".join(lines).rstrip() + "\n"


def _write_index_md(store, project_id: str, rdir: Path) -> None:
    """Mirror the metadata index into a human-readable research/INDEX.md."""
    idx = get_research_index(store, project_id)
    lines = ["# Project Research Index", ""]
    for e in idx:
        prev = (e.get("summary_preview") or "").strip()
        prev = f" — {prev}" if prev else ""
        lines.append(f"- **{e.get('topic', '')}** → `{e.get('path', '')}`{prev}")
    try:
        (Path(rdir) / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:  # pragma: no cover
        logger.debug("research INDEX.md write skipped", exc_info=True)


def _fallback_summary(topic: str, search_output: str) -> str:
    """Heuristic brief when no LLM is available: list the result titles."""
    if not search_output:
        return (f"## Summary\nNo web results were available for \"{topic}\" at "
                "research time.\n")
    titles = re.findall(r"^#{1,6}\s*\d+\.\s*(.+)$", search_output, flags=re.M)
    bullets = "\n".join(f"- {t.strip()}" for t in titles[:7]) or "- (see sources)"
    return (f"## Summary\nCollected web results for \"{topic}\". An LLM summary "
            f"was unavailable, so the top result titles are listed below.\n\n"
            f"## Key findings\n{bullets}\n")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove a reasoning model's <think> preamble from a utility response.

    Strips closed ``<think>…</think>`` blocks; if a <think> opens and never
    closes (the whole budget went to reasoning), drops everything from it.
    """
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "<think>" in text.lower():
        text = re.split(r"(?i)<think>", text)[0]
    return text.strip()


def _message_text(resp: Dict[str, Any]) -> str:
    msg = ((resp or {}).get("choices", [{}]) or [{}])[0].get("message", {}) or {}
    out = _strip_think(msg.get("content") or "")
    if not out:
        # Last resort: content was empty because the budget went to <think>.
        # Salvage the reasoning rather than drop the summary entirely.
        out = _strip_think(msg.get("reasoning_content") or "")
    return out


async def _llm_complete(context, prompt: str, *, max_tokens: int,
                        temperature: float = 0.3, timeout: float = 120) -> str:
    """One no-think utility completion. Returns '' on any failure.

    The upstream is a reasoning model (``--deep-reason``); a utility call
    that leaves thinking ON burns the entire token budget inside ``<think>``
    and returns an EMPTY ``content`` (verified live: finish_reason=length,
    900 reasoning tokens, content=""). We disable thinking the way
    ``core/dream.py`` does for its utility calls — the ``/no_think``
    soft-switch + ``enable_thinking=False`` hard-switch + a system nudge —
    give the budget headroom, and strip any stray ``<think>`` block. Plain
    foreground call (no <code>use_worker</code>: no worker node is
    configured here, so it would fall through to this same model anyway).
    """
    llm = getattr(context, "llm_client", None)
    if llm is None:
        return ""
    try:
        r = await llm.chat_completion({
            "model": getattr(getattr(context, "args", None), "model", "default"),
            "messages": [
                {"role": "system",
                 "content": "Answer directly and concisely. Do NOT emit a <think> block."},
                {"role": "user", "content": prompt + "\n\n/no_think"},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=timeout)
        return _message_text(r)
    except Exception as e:
        logger.debug("research LLM call failed: %s", e)
        return ""


async def _summarize(context, topic: str, proj: Dict[str, Any],
                     search_output: str,
                     llm_summarizer: Optional[Callable[..., Awaitable[str]]]) -> str:
    if llm_summarizer is not None:
        try:
            out = (await llm_summarizer(topic, search_output) or "").strip()
            if out:
                return out
        except Exception:
            logger.debug("custom research summarizer failed", exc_info=True)
    if search_output and getattr(context, "llm_client", None) is not None:
        goal = (proj.get("goal") or "")[:300]
        prompt = (
            f"You are researching the topic \"{topic}\" for a project.\n"
            f"Project goal: {goal}\n\n"
            "Below are raw web search results. Write a concise research "
            "brief in Markdown with exactly two sections: '## Summary' "
            "(3-5 sentences) and '## Key findings' (3-7 bullet points). Be "
            "factual and specific; do NOT invent facts or sources. Output "
            "ONLY the Markdown.\n\n"
            f"SEARCH RESULTS:\n{search_output[:6000]}"
        )
        out = await _llm_complete(context, prompt, max_tokens=1500,
                                  temperature=0.3, timeout=120)
        if out:
            return out
    return _fallback_summary(topic, search_output)


# ------------------------------------------------------------------ persist core

async def _persist(context, project_id: str, topic: str, search_output: str, *,
                   task_id: Optional[str] = None,
                   llm_summarizer: Optional[Callable[..., Awaitable[str]]] = None,
                   ) -> ResearchResult:
    """Summarise an already-fetched search output and write the brief.

    Shared by :func:`research_topic` (which fetches first) and
    :func:`persist_research_from_output` (which reuses the advancer's
    output). Never raises — returns a ResearchResult with ``ok=False`` on a
    hard failure.
    """
    store = getattr(context, "project_store", None)
    if store is None:
        return ResearchResult(False, topic, error="no project_store")
    try:
        proj = store.get_project(project_id)
    except Exception as e:
        return ResearchResult(False, topic, error=str(e))
    if not proj:
        return ResearchResult(False, topic, error="project not found")

    rdir = _research_dir(store, project_id)
    if rdir is None:
        return ResearchResult(False, topic, error="no workspace for project")

    summary_md = await _summarize(context, topic, proj, search_output, llm_summarizer)
    sources = _extract_sources(search_output)[:12]
    slug = _slugify(topic)
    ts = time.time()
    doc = _render_doc(topic, proj, summary_md, sources, ts)

    path = rdir / f"{slug}.md"
    try:
        path.write_text(doc, encoding="utf-8")
    except Exception as e:
        return ResearchResult(False, topic, slug=slug, error=f"write failed: {e}")

    rel = f"{RESEARCH_SUBDIR}/{slug}.md"
    preview = _first_line(summary_md)[:160]
    entry = {
        "topic": topic, "slug": slug, "path": rel, "ts": ts,
        "summary_preview": preview, "sources_count": len(sources),
    }
    try:
        _upsert_index(store, project_id, entry)
    except Exception:
        logger.debug("research index upsert failed", exc_info=True)
    if task_id:
        # A research brief is a real deliverable in a research/general project,
        # but in a CODING project it is just working context — registering it
        # as a `file` artifact made the end-of-project cleanup KEEP a pile of
        # research/*.md as "deliverables" (observed live). Still written and
        # indexed (surfaced in the briefing); just not a kept deliverable here.
        try:
            _kind = (store.get_project(project_id) or {}).get("kind", "")
        except Exception:
            _kind = ""
        if str(_kind).upper() != "CODING":
            try:
                store.add_artifact(task_id, "file", rel)
            except Exception:
                logger.debug("research artifact add skipped", exc_info=True)
    try:
        store.log_event(project_id, task_id, "research_added",
                        {"topic": topic, "path": rel,
                         "sources": len(sources), "summary_preview": preview})
    except Exception:
        logger.debug("research event log skipped", exc_info=True)
    _write_index_md(store, project_id, rdir)

    return ResearchResult(True, topic, slug=slug, path=rel, abs_path=str(path),
                          summary=summary_md, sources=sources)


async def persist_research_from_output(context, project_id: str, topic: str,
                                       search_output: str, *,
                                       task_id: Optional[str] = None,
                                       llm_summarizer=None) -> ResearchResult:
    """Public wrapper: persist a brief from a search output already in hand
    (used by the autoadvancer's research path so it doesn't search twice)."""
    return await _persist(context, project_id, topic, search_output,
                          task_id=task_id, llm_summarizer=llm_summarizer)


# ------------------------------------------------------------------ search + persist

async def research_topic(context, project_id: str, topic: str, *,
                         tool_runner: Optional[ToolRunner] = None,
                         llm_summarizer=None,
                         queries: Optional[List[str]] = None,
                         task_id: Optional[str] = None) -> ResearchResult:
    """Research ONE topic: run web search(es), summarise, persist the brief.

    ``queries`` overrides the search terms (defaults to the topic itself).
    Falls back to the tool registry for search when ``tool_runner`` is
    omitted; with no runner available it still writes a (sourceless) brief
    so the topic is recorded.
    """
    if not topic or not topic.strip():
        return ResearchResult(False, topic, error="empty topic")
    topic = topic.strip()
    runner = _resolve_runner(context, tool_runner)
    chunks: List[str] = []
    if runner is not None:
        for q in (queries or [topic])[:2]:
            try:
                out = await runner("web_search", {"query": str(q)[:200]})
            except Exception as e:
                logger.warning("research search failed: %s", e)
                continue
            if out and isinstance(out, str):
                chunks.append(out)
    return await _persist(context, project_id, topic, "\n\n".join(chunks),
                          task_id=task_id, llm_summarizer=llm_summarizer)


# ------------------------------------------------------------------ topic discovery

async def propose_topics(context, project_id: str, n: int = MAX_TOPICS_DEFAULT,
                         llm_proposer=None) -> List[str]:
    """Derive up to ``n`` distinct research topics from the project goal +
    open tasks. LLM-driven when a client is present; otherwise falls back to
    the open task descriptions, then the goal."""
    store = getattr(context, "project_store", None)
    if store is None:
        return []
    proj = store.get_project(project_id) or {}
    goal = (proj.get("goal") or proj.get("title") or "").strip()
    try:
        tasks = store.list_tasks(project_id)
    except Exception:
        tasks = []
    open_descs = [t["description"] for t in tasks
                  if str(t.get("status", "")).upper() in _OPEN_TASK_STATUSES][:10]

    if llm_proposer is not None:
        try:
            out = await llm_proposer(goal, open_descs)
            topics = _clean_topic_lines(out, n)
            if topics:
                return topics
        except Exception:
            logger.debug("custom topic proposer failed", exc_info=True)

    if getattr(context, "llm_client", None) is not None and (goal or open_descs):
        task_block = "\n".join(f"- {d}" for d in open_descs) or "(none yet)"
        prompt = (
            f"Project goal: {goal}\nOpen tasks:\n{task_block}\n\n"
            f"Propose up to {n} concise, DISTINCT research topics (3-8 words "
            "each) whose findings would help advance this project. Output ONE "
            "topic per line — no numbering, no commentary, no blank lines."
        )
        out = await _llm_complete(context, prompt, max_tokens=400,
                                  temperature=0.4, timeout=90)
        topics = _clean_topic_lines(out, n)
        if topics:
            return topics

    fallback = open_descs[:n] or ([goal] if goal else [])
    return fallback


def _clean_topic_lines(text: str, n: int) -> List[str]:
    out: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("-*0123456789.)# \t").strip().strip('"')
        if s and len(s) > 2 and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out


async def research_project(context, project_id: str, *,
                           topics: Optional[List[str]] = None,
                           max_topics: int = MAX_TOPICS_DEFAULT,
                           tool_runner: Optional[ToolRunner] = None,
                           llm_summarizer=None) -> List[ResearchResult]:
    """Research several topics. When ``topics`` is empty, auto-derive them
    from the project goal/tasks via :func:`propose_topics`."""
    if not topics:
        topics = await propose_topics(context, project_id, max_topics)
    results: List[ResearchResult] = []
    for t in (topics or [])[:max_topics]:
        results.append(await research_topic(
            context, project_id, t,
            tool_runner=tool_runner, llm_summarizer=llm_summarizer))
    return results
