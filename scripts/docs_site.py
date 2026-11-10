#!/usr/bin/env python3
"""The documentation site's map, and the build that keeps every page in line with it.

The site is organised by FEATURE (what the agent does), not by code module:

    index.html                    what Ghost is, the machines, one card per feature
    start/*.html                  install, configure, troubleshoot, glossary
    features/<feature>.html       one page per feature, written for the user
    reference/index.html          every deep page, grouped by feature
    reference/<feature>/*.html    the deep pages, written for maintainers

``python scripts/docs_site.py build`` regenerates the sidebar of every page from
FEATURES / START / REFERENCE below and the "Reference pages" block of every
feature page (between ``<!-- refs -->`` and ``<!-- /refs -->``). Run it after
adding, renaming or retitling a page. ``check`` reports what build would change
and every broken internal link, and exits non-zero on either.

``migrate`` is the one-off move from the module layout (2026-11-10); it is kept
because MOVES documents where every old page went.
"""
from __future__ import annotations

import html
import os
import posixpath
import re
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# (slug, title, one-line summary) — the order is the order on the home page and in the sidebar
FEATURES = [
    ("interfaces", "Interfaces", "Web console, Slack, the handheld, the CLI and the API."),
    ("voice", "Voice", "Talk to it and hear it answer: Whisper speech-to-text, Phonon (English) and macOS (Greek) voices."),
    ("search", "Search & the web", "Web search, deep research and the browser — over Tor only."),
    ("darkweb", "Dark-web search", "Search and research .onion sites through Ahmia, Torch and Torgle."),
    ("pictures", "Pictures", "Generate and edit images, likeness from real photos, and vision."),
    ("memory", "Memory", "What it remembers about you, how it recalls it, and the rules you teach it."),
    ("documents", "Documents (RAG)", "Ask questions over your PDFs, audio and videos; get PDF reports."),
    ("coding", "Coding & sandbox", "Writes and runs code in a Docker sandbox, with services and jobs."),
    ("projects", "Projects & autonomy", "Multi-step projects, scheduled tasks and an autopilot that works while idle."),
    ("labs", "Labs", "Throwaway real machines on a Raspberry Pi for experiments."),
    ("selfhood", "Selfhood", "Mood, tone, the self-image (Mirror), the sentiment model and the face."),
    ("performance", "Performance", "Head and slot saves, cache protection, MTP, node offload and time budgets."),
    ("reliability", "Reliability", "The verifier, guards and breakers, health checks and feedback."),
    ("personal", "Personal services", "Portfolio, weather and games."),
    ("infrastructure", "Infrastructure", "The four machines, Tor-only egress, isolation and supervision."),
]
FEATURE_TITLE = {s: t for s, t, _ in FEATURES}

START = [
    ("start/getting-started.html", "Getting started"),
    ("start/installation.html", "Installation"),
    ("start/configuration.html", "Settings that matter"),
    ("start/troubleshooting.html", "When something breaks"),
    ("start/glossary.html", "Glossary"),
]

# old path -> new path. A target under DELETED is removed and links to it go to the replacement.
MOVES = {
    # start
    "getting-started.html": "start/getting-started.html",
    "installation.html": "start/installation.html",
    "configuration.html": "start/configuration.html",
    "troubleshooting.html": "start/troubleshooting.html",
    "glossary.html": "start/glossary.html",
    # interfaces
    "interfaces/web_server.html": "reference/interfaces/web_server.html",
    "interfaces/slack_bot.html": "reference/interfaces/slack_bot.html",
    "interfaces/clockwork_ghost.html": "reference/interfaces/clockwork_ghost.html",
    "interfaces/cli.html": "reference/interfaces/cli.html",
    "cli_reference.html": "reference/interfaces/cli_reference.html",
    "api/routes.html": "reference/interfaces/api_routes.html",
    "core/sessions.html": "reference/interfaces/sessions.html",
    # voice
    "interfaces/voice_server.html": "reference/voice/voice_loop.html",
    # search
    "tools/search.html": "reference/search/search.html",
    "tools/browser.html": "reference/search/browser.html",
    # dark web
    "tools/darkweb_search.html": "reference/darkweb/darkweb_search.html",
    # pictures
    "tools/image_gen.html": "reference/pictures/image_gen.html",
    "interfaces/image_gen_server.html": "reference/pictures/image_gen_server.html",
    "tools/vision.html": "reference/pictures/vision.html",
    # memory
    "tools/memory_tools.html": "reference/memory/memory_tools.html",
    "algorithms/memory_hydration.html": "reference/memory/memory_hydration.html",
    "memory/vector.html": "reference/memory/vector.html",
    "memory/graph.html": "reference/memory/graph.html",
    "memory/temporal.html": "reference/memory/temporal.html",
    "memory/profile.html": "reference/memory/profile.html",
    "memory/fact_ledger.html": "reference/memory/fact_ledger.html",
    "memory/episodes.html": "reference/memory/episodes.html",
    "memory/experience.html": "reference/memory/experience.html",
    "memory/journal.html": "reference/memory/journal.html",
    "memory/scratchpad.html": "reference/memory/scratchpad.html",
    "memory/attribution.html": "reference/memory/attribution.html",
    "memory/adaptive_threshold.html": "reference/memory/adaptive_threshold.html",
    "memory/frontier.html": "reference/memory/frontier.html",
    "memory/readonly.html": "reference/memory/readonly.html",
    "memory/failclosed.html": "reference/memory/failclosed.html",
    "learning-loop.html": "reference/memory/owner_rules.html",
    "core/dream.html": "reference/memory/dream.html",
    "algorithms/dream_cycle.html": "reference/memory/dream_cycle.html",
    # documents
    "memory/youtube_ingest.html": "reference/documents/youtube_ingest.html",
    "tools/report_pdf.html": "reference/documents/report_pdf.html",
    # coding
    "tools/execute.html": "reference/coding/execute.html",
    "tools/file_system.html": "reference/coding/file_system.html",
    "sandbox/docker.html": "reference/coding/docker.html",
    "sandbox/jobs.html": "reference/coding/jobs.html",
    "sandbox/services.html": "reference/coding/services.html",
    "core/coding_executor.html": "reference/coding/coding_executor.html",
    "core/coding_loop.html": "reference/coding/coding_loop.html",
    "core/code_claims.html": "reference/coding/code_claims.html",
    "tools/probe_model.html": "reference/coding/probe_model.html",
    "core/probe_transcripts.html": "reference/coding/probe_transcripts.html",
    "core/experiment_gate.html": "reference/coding/experiment_gate.html",
    "tools/workspace.html": "reference/coding/workspace.html",
    "tools/workspace_track.html": "reference/coding/workspace_track.html",
    "core/workspace_cleanup.html": "reference/coding/workspace_cleanup.html",
    "algorithms/workspace.html": "reference/coding/workspace_continuity.html",
    "tools/runs.html": "reference/coding/runs.html",
    "tools/acquired_skills.html": "reference/coding/acquired_skills.html",
    "algorithms/skill_acquisition.html": "reference/coding/skill_acquisition.html",
    "tools/quarantine.html": "reference/coding/quarantine.html",
    # projects
    "tools/projects.html": "reference/projects/projects_tool.html",
    "memory/projects.html": "reference/projects/project_store.html",
    "api/projects_routes.html": "reference/projects/projects_routes.html",
    "core/project_advancer.html": "reference/projects/project_advancer.html",
    "core/project_research.html": "reference/projects/project_research.html",
    "core/project_safety.html": "reference/projects/project_safety.html",
    "core/planning.html": "reference/projects/planning.html",
    "core/task_close.html": "reference/projects/task_close.html",
    "core/task_kind.html": "reference/projects/task_kind.html",
    "core/task_wait.html": "reference/projects/task_wait.html",
    "core/triggers.html": "reference/projects/triggers.html",
    "tools/tasks.html": "reference/projects/tasks.html",
    "core/delegation.html": "reference/projects/background_jobs.html",
    "core/autonomous_activity.html": "reference/projects/autonomous_activity.html",
    "algorithms/cross_project_map.html": "reference/projects/cross_project_map.html",
    "long-horizon.html": "reference/projects/long_tasks.html",
    # labs
    "tools/labs.html": "reference/labs/labs_tool.html",
    "labs/PLAN.md": "reference/labs/PLAN.md",
    "labs/PHASE_C.md": "reference/labs/PHASE_C.md",
    "labs/PHASE_D.md": "reference/labs/PHASE_D.md",
    "labs/UNIFIED_DESIGN.md": "reference/labs/UNIFIED_DESIGN.md",
    "labs/ghost-labs-contract-v2.md": "reference/labs/ghost-labs-contract-v2.md",
    "labs/ghost-labs-manager-prompt.md": "reference/labs/ghost-labs-manager-prompt.md",
    # selfhood
    "algorithms/selfhood.html": "reference/selfhood/selfhood.html",
    "algorithms/selfhood.md": "reference/selfhood/selfhood.md",
    "algorithms/selfhood_functional_report.html": "reference/selfhood/selfhood_functional_report.html",
    "algorithms/selfhood_functional_report.md": "reference/selfhood/selfhood_functional_report.md",
    "tools/introspect.html": "reference/selfhood/introspect.html",
    # performance
    "core/head_saves.html": "reference/performance/head_saves.html",
    "core/llm.html": "reference/performance/llm.html",
    "algorithms/node_offload.html": "reference/performance/node_offload.html",
    "core/node_throughput.html": "reference/performance/node_throughput.html",
    "core/context_manager.html": "reference/performance/context_manager.html",
    # reliability
    "core/verifier.html": "reference/reliability/verifier.html",
    "core/claim_binding.html": "reference/reliability/claim_binding.html",
    "core/evidence_gate.html": "reference/reliability/evidence_gate.html",
    "core/objection.html": "reference/reliability/objection.html",
    "core/strikes.html": "reference/reliability/strikes.html",
    "core/failure_dimension.html": "reference/reliability/failure_dimension.html",
    "core/feedback.html": "reference/reliability/feedback.html",
    "tools/outcome.html": "reference/reliability/outcome.html",
    "tools/tool_failure.html": "reference/reliability/tool_failure.html",
    "tools/fallback_chains.html": "reference/reliability/fallback_chains.html",
    "algorithms/circuit_breaker.html": "reference/reliability/circuit_breaker.html",
    "algorithms/metacognition.html": "reference/reliability/metacognition.html",
    "self-improvement.html": "reference/reliability/checking_and_learning.html",
    "audit_fixes.html": "reference/reliability/audit_2026_05.html",
    "tools/system.html": "reference/reliability/system_tool.html",
    "logging.html": "reference/reliability/logging.html",
    "claude_trainer.html": "reference/reliability/claude_trainer.html",
    # personal
    "tools/portfolio.html": "reference/personal/portfolio.html",
    "api/game_routes.html": "reference/personal/game_routes.html",
    # infrastructure
    "architecture.html": "reference/infrastructure/architecture.html",
    "how-it-works.html": "reference/infrastructure/how_it_works.html",
    "safety.html": "reference/infrastructure/safety.html",
    "algorithms/mandatory_tor.html": "reference/infrastructure/mandatory_tor.html",
    "algorithms/request_lifecycle.html": "reference/infrastructure/request_lifecycle.html",
    "core/agent.html": "reference/infrastructure/agent.html",
    "core/prompts.html": "reference/infrastructure/prompts.html",
    "core/isolation.html": "reference/infrastructure/isolation.html",
    "core/bus.html": "reference/infrastructure/bus.html",
    "core/request_scope.html": "reference/infrastructure/request_scope.html",
    "tools/registry.html": "reference/infrastructure/tool_registry.html",
}

# removed pages -> the page a link to them now opens
DELETED = {
    "interfaces.html": "features/interfaces.html",
    "memory.html": "features/memory.html",
    "tools.html": "index.html",
    "capabilities.html": "index.html",
    "reference.html": "reference/index.html",
    "ghost-labs.html": "features/labs.html",
    # tombstones of removed modules
    "core/agent_qwen.html": "reference/index.html",
    "core/workspace_model.html": "reference/index.html",
    "tools/qwen_bridge.html": "reference/index.html",
    "tools/database.html": "features/labs.html",
    "tools/composed_skills.html": "reference/index.html",
    "memory/skills.html": "reference/index.html",
}

SIDEBAR_RE = re.compile(r'<aside class="sidebar">.*?</aside>\n?', re.S)
REFS_RE = re.compile(r"<!-- refs -->.*?<!-- /refs -->", re.S)
LINK_RE = re.compile(r'((?:href|src)=")([^"#?]+?\.(?:html|md|css|png|gif|jpg|svg))((?:[#?][^"]*)?")')


# ── the map ──────────────────────────────────────────────────────────────────

def ref_pages(feature: str) -> list:
    d = DOCS / "reference" / feature
    return sorted(p for p in d.glob("*.html")) if d.is_dir() else []


def title_of(page: Path) -> str:
    m = re.search(r"<title>(.*?)</title>", page.read_text(encoding="utf-8"), re.S)
    t = html.unescape(m.group(1)).strip() if m else page.stem
    t = re.sub(r"\s*[—-]\s*Ghost Agent\s*$", "", t)
    t = re.sub(r"^Ghost Agent\s*[—-]\s*", "", t)
    return t


def all_pages() -> list:
    return sorted(p for p in DOCS.rglob("*.html") if "assets" not in p.relative_to(DOCS).parts)


def rel(page_rel: str, target_rel: str) -> str:
    return posixpath.relpath(target_rel, posixpath.dirname(page_rel) or ".")


# ── generated parts ─────────────────────────────────────────────────────────

def sidebar(page_rel: str) -> str:
    def a(target, label, extra=""):
        cur = ' class="current"' if target == page_rel else ""
        return f'    <a{cur} href="{rel(page_rel, target)}"{extra}>{html.escape(label)}</a>\n'
    out = ['<aside class="sidebar">\n',
           f'  <a class="brand" href="{rel(page_rel, "index.html")}"><span class="mark"></span>Ghost Agent</a>\n',
           "  <nav>\n", a("index.html", "What it can do"),
           "    <h3>Start</h3>\n"]
    out += [a(p, t) for p, t in START]
    out.append("    <h3>Features</h3>\n")
    parts = page_rel.split("/")
    here = parts[1] if parts[0] == "reference" and len(parts) > 2 else (
        Path(parts[1]).stem if parts[0] == "features" else "")
    for slug, t, _ in FEATURES:
        out.append(a(f"features/{slug}.html", t))
        if slug == here:
            refs = ref_pages(slug)
            if refs:
                out.append('    <div class="subnav">\n')
                for p in refs:
                    r = p.relative_to(DOCS).as_posix()
                    out.append("  " + a(r, title_of(p)))
                out.append("    </div>\n")
    out.append("    <h3>Reference</h3>\n")
    out.append(a("reference/index.html", "All reference pages"))
    out.append("  </nav>\n</aside>\n")
    return "".join(out)


def refs_block(feature: str, page_rel: str) -> str:
    refs = ref_pages(feature)
    if not refs:
        return "<!-- refs --><!-- /refs -->"
    items = "".join(f'    <li><a href="{rel(page_rel, p.relative_to(DOCS).as_posix())}">'
                    f"{html.escape(title_of(p))}</a></li>\n" for p in refs)
    return ("<!-- refs -->\n<h2 id=\"reference\">How it works: the reference pages</h2>\n"
            "<p>Written for maintainers: every mechanism, its settings and its history.</p>\n"
            f'<ul class="ref-list">\n{items}</ul>\n<!-- /refs -->')


def reference_index_body(page_rel: str) -> str:
    out = []
    for slug, t, summary in FEATURES:
        refs = ref_pages(slug)
        out.append(f'<h2 id="{slug}"><a href="{rel(page_rel, f"features/{slug}.html")}">{html.escape(t)}</a></h2>\n')
        out.append(f"<p>{html.escape(summary)}</p>\n")
        if refs:
            out.append('<ul class="ref-list">\n' + "".join(
                f'  <li><a href="{rel(page_rel, p.relative_to(DOCS).as_posix())}">{html.escape(title_of(p))}</a></li>\n'
                for p in refs) + "</ul>\n")
    return "<!-- refindex -->\n" + "".join(out) + "<!-- /refindex -->"


def features_grid(page_rel: str) -> str:
    cards = "".join(
        f'  <a class="feature" href="{rel(page_rel, f"features/{slug}.html")}"><span class="ft">{html.escape(t)}</span>'
        f'<span class="fd">{html.escape(summary)}</span>'
        f'<span class="fm">{len(ref_pages(slug))} reference page{"s" if len(ref_pages(slug)) != 1 else ""}</span></a>\n'
        for slug, t, summary in FEATURES)
    return f'<!-- features -->\n<div class="feature-grid">\n{cards}</div>\n<!-- /features -->'


def render(page: Path) -> str:
    r = page.relative_to(DOCS).as_posix()
    s = page.read_text(encoding="utf-8")
    s = SIDEBAR_RE.sub(lambda m: sidebar(r), s, count=1)
    if r.startswith("features/"):
        s = REFS_RE.sub(lambda m: refs_block(Path(r).stem, r), s, count=1)
    if r == "index.html":
        s = re.sub(r"<!-- features -->.*?<!-- /features -->", lambda m: features_grid(r), s, count=1, flags=re.S)
    if r == "reference/index.html":
        s = re.sub(r"<!-- refindex -->.*?<!-- /refindex -->", lambda m: reference_index_body(r), s,
                   count=1, flags=re.S)
    return s


# ── link check ──────────────────────────────────────────────────────────────

def broken_links() -> list:
    bad = []
    for page in all_pages():
        r = page.relative_to(DOCS).as_posix()
        text = re.sub(r"<code>.*?</code>|<pre[^>]*>.*?</pre>", "", page.read_text(encoding="utf-8"), flags=re.S)
        for m in re.finditer(r'(?:href|src)="([^"]+)"', text):  # examples inside code are not links
            u = m.group(1)
            if re.match(r"^(?:[a-z]+:|#|//)", u):
                continue
            path = u.split("#")[0].split("?")[0]
            if not path:
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(r), path))
            if not (DOCS / target).exists():
                bad.append(f"{r} -> {u}")
    return bad


# ── commands ────────────────────────────────────────────────────────────────

def migrate() -> None:
    """One-off: move every page to its feature home, rewrite every internal link, drop removed pages."""
    where = {**{k: v for k, v in MOVES.items()}, **DELETED}
    pages = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*")
             if p.is_file() and p.suffix in (".html", ".md") and "assets" not in p.parts}
    for old in pages:
        new = MOVES.get(old, old)
        if old in DELETED:
            continue
        text = (DOCS / old).read_text(encoding="utf-8")

        def fix(m, old=old, new=new):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(old), m.group(2)))
            target = where.get(target, target)
            return m.group(1) + rel(new, target) + m.group(3)
        if old.endswith(".html"):
            text = LINK_RE.sub(fix, text)
        (DOCS / old).write_text(text, encoding="utf-8")
    for old, new in MOVES.items():
        src, dst = DOCS / old, DOCS / new
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    for old in DELETED:
        if (DOCS / old).exists():
            (DOCS / old).unlink()
    for d in sorted((p for p in DOCS.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()


def build(write: bool) -> int:
    changed = []
    for page in all_pages():
        new = render(page)
        if new != page.read_text(encoding="utf-8"):
            changed.append(page.relative_to(DOCS).as_posix())
            if write:
                page.write_text(new, encoding="utf-8")
    return len(changed) if write else changed


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "migrate":
        migrate()
        return 0
    if cmd == "build":
        n = build(True)
        print(f"rebuilt {n} page(s)")
        bad = broken_links()
        print("\n".join(bad) or "no broken links")
        return 1 if bad else 0
    stale = build(False)
    bad = broken_links()
    for s in stale:
        print("stale:", s)
    for b in bad:
        print("broken:", b)
    return 1 if (stale or bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
