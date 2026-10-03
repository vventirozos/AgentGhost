"""§4KW fourth review (2026-10-02): set the user-immutable flag on the
workspaces of projects ALREADY released (new releases set it themselves —
`ProjectStore._chmod_tree`). Mode bits alone were undone from inside the
sandbox (`chmod -R u+w . && rm -rf *`); the flag cannot be cleared there.
A released app's runtime state (its sqlite db and the db's folder) stays
writable (fifth review: the Jiu Jitsu Calendar keeps data.db next to
app.py). Idempotent — a re-run applies the current rule; macOS/BSD only;
refuses while the agent runs. Dry run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/release_immutable_4kw.py [--apply]
"""
import os, stat, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
MEM = HOME / "system" / "memory"


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def main():
    from ghost_agent.memory.projects import ProjectStore
    imm = getattr(stat, "UF_IMMUTABLE", 0)
    if not (imm and hasattr(os, "chflags")):
        raise SystemExit("no BSD file flags on this host — nothing to do")
    if APPLY and _agent_listening():
        raise SystemExit("the agent is running — stop it first; nothing applied")
    store = ProjectStore(MEM, sandbox_root=HOME / "sandbox")
    for p in store.list_projects("RELEASED") or []:
        ws = Path(str(p.get("workspace_dir") or ""))
        if not ws.is_dir():
            print(f"  {p['id']}: workspace missing — skipped")
            continue
        entries = [ws, *ws.rglob("*")]
        flagged = sum(1 for e in entries if not e.is_symlink() and e.lstat().st_flags & imm)
        print(f"  {p['id']} {p.get('title', '')!r}: {flagged}/{len(entries)} entries immutable")
        if APPLY:
            store.set_workspace_readonly(p["id"], True)
            after = sum(1 for e in entries if not e.is_symlink() and e.lstat().st_flags & imm)
            print(f"    → {after}/{len(entries)} after")


if __name__ == "__main__":
    main()
