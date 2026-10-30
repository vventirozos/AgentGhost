#!/usr/bin/env python3
"""KB review one-off (2026-10-30, operator: "yes" to re-ingesting the
PostgreSQL manual): the stored copy was split before the per-section
breadcrumb fix, so most passages carry the page's first section. Deletes the
document (rows, catalogue, outline) and streams the PDF in again.

Run ONLY with the agent stopped (single-writer store); `--apply` writes.
  PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/reingest_kb1030.py <pdf> [--apply]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
MEM = Path(os.environ.get("GHOST_MEMORY_DIR") or HOME / "system" / "memory")
NAME = "postgresql-18-A4.pdf"


def main() -> int:
    pdf = Path(next(a for a in sys.argv[1:] if not a.startswith("--")))
    if not pdf.is_file() or pdf.read_bytes()[:5] != b"%PDF-":
        print(f"not a PDF: {pdf}")
        return 1
    if not APPLY:
        print(f"would re-ingest {pdf} as {NAME} into {MEM} (dry run)")
        return 0
    from ghost_agent.memory.store_lock import assert_no_other_writer
    assert_no_other_writer(MEM, Path(__file__).name)
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    before = len((vm.collection.get(where={"source": NAME}, include=[]) or {}).get("ids") or [])
    t0 = time.time()
    vm.delete_document_by_name(NAME)
    # the real tool path (outline, summary row, PARTIAL logic), not a re-implementation
    import asyncio
    import shutil
    import tempfile
    from ghost_agent.tools.memory import tool_gain_knowledge
    with tempfile.TemporaryDirectory() as sb:
        shutil.copy2(pdf, Path(sb) / NAME)
        out = asyncio.run(tool_gain_knowledge(NAME, sandbox_dir=Path(sb), memory_system=vm))
    print(str(out)[:300])
    after = len((vm.collection.get(where={"source": NAME}, include=[]) or {}).get("ids") or [])
    print(f"rows {before} → {after}; {time.time() - t0:.0f} s; listed={NAME in vm.get_library()}; "
          f"outline={bool(vm.get_document_outline(NAME))}")
    return 0 if after and NAME in vm.get_library() else 1


if __name__ == "__main__":
    sys.exit(main())
