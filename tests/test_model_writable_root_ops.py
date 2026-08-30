"""§4DX — the CLASS, enumerated, instead of one site per review round.

Five review rounds each found the same defect in a new place: a host-side
file operation whose path sits inside a directory the sandboxed model can
write to, performed with an API that follows symlinks. Round 1 fixed one
site, round 2 three, round 3 two more, round 4 two, round 5 four — every
round patching what it was looking at while the sibling ten lines away, or
the mirror-image file, stayed open. `jobs.py` guarded its DIRECTORY and read
its FILES plainly; `services.py` guarded its FILES and left the DIRECTORY
open; each file's comment named the other as the unsafe twin.

This test enumerates the class from the AST instead. A new unguarded
operation on a model-writable root fails HERE, at the time it is written,
rather than in the next review round.

⚠ IF THIS TEST FAILS ON CODE YOU JUST WROTE: do not add your line to the
allow-list because "it looks fine". Use `write_text_nofollow` /
`read_bytes_nofollow` from `tools/file_system.py`, or justify the exemption
in `_REVIEWED_EXEMPTIONS` with the reason it cannot be reached.
"""
import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

# The two modules whose entire working directory lives inside the sandbox
# bind mount (`/workspace/.services`, `/workspace/.jobs`) and is therefore
# writable by model-authored code running in the container.
_MODEL_WRITABLE_MODULES = ("sandbox/services.py", "sandbox/jobs.py")

# Symlink-following operations that read or write file CONTENT. `unlink`,
# `exists`, `glob` and `iterdir` are excluded deliberately: `unlink` removes
# the LINK not its target, and the others do not transfer content.
_FOLLOWING_OPS = {"read_text", "read_bytes", "write_text", "write_bytes",
                  "open"}

# Sites verified unreachable or harmless, each with its reason. Adding to
# this list is a review decision, not a formality.
_REVIEWED_EXEMPTIONS = {
    # A distinctive substring of the call -> why it is safe.
    # Adding an entry is a REVIEW DECISION, not a formality.
}


def _unguarded_content_ops(module_rel):
    """Every symlink-following content op in `module_rel`.

    ⚠ A WHITELIST, NOT A HEURISTIC. The first version only flagged an op
    whose source segment or three preceding lines mentioned `host_dir` /
    `_paths(` — and it did not fire when round 5's actual critical was
    re-introduced, because the path had been bound to a local variable one
    line earlier. A detector that cannot catch the bug it was written for is
    the defect it is meant to prevent.

    These two modules work ENTIRELY inside the sandbox bind mount, so every
    content op in them is on a model-writable root by construction. Flag
    them all; exempt individually with a reason.
    """
    src = (_SRC / module_rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if attr not in _FOLLOWING_OPS:
            continue
        seg = (ast.get_source_segment(src, node) or "").replace("\n", " ")
        findings.append((node.lineno, attr, seg[:90]))
    return findings


@pytest.mark.parametrize("module_rel", _MODEL_WRITABLE_MODULES)
def test_no_unguarded_content_op_on_a_model_writable_root(module_rel):
    """The enumeration. Fails the moment a new site is added."""
    findings = _unguarded_content_ops(module_rel)
    unexplained = [
        (ln, op, seg) for ln, op, seg in findings
        if not any(k in seg for k in _REVIEWED_EXEMPTIONS)
    ]
    assert not unexplained, (
        f"{module_rel}: symlink-following content ops on a model-writable "
        f"root:\n" +
        "\n".join(f"  line {ln}: {op} -> {seg}" for ln, op, seg in unexplained) +
        "\n\nUse write_text_nofollow / read_bytes_nofollow from "
        "tools/file_system.py, or add a reviewed exemption with its reason.")


def test_the_enumeration_can_actually_fail():
    """⚠ A GUARD THAT CANNOT FIRE IS A COMMENT. Proves the detector sees an
    unguarded op by running it over a synthetic module containing one."""
    import tempfile

    hostile = (
        "class S:\n"
        "    def f(self):\n"
        "        return (self.host_dir / 'x.log').read_bytes()\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fake.py"
        p.write_text(hostile)
        tree = ast.parse(hostile)
        hits = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) in _FOLLOWING_OPS]
        assert hits, "the detector's op set no longer matches a plain read"
        seg = ast.get_source_segment(hostile, hits[0])
        assert "host_dir" in seg and "_nofollow" not in seg


def test_both_helpers_exist_and_refuse_a_symlink(tmp_path):
    """The enumeration is only worth what the helpers are worth."""
    from ghost_agent.tools.file_system import (read_bytes_nofollow,
                                               write_text_nofollow)
    victim = tmp_path / "victim"
    victim.write_text("REAL")
    link = tmp_path / "link"
    link.symlink_to(victim)
    with pytest.raises(ValueError):
        read_bytes_nofollow(link)
    with pytest.raises(ValueError):
        write_text_nofollow(link, "PWNED")
    assert victim.read_text() == "REAL"
