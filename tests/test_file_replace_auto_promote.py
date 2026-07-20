"""Tests for the replace→write auto-promote path in file_system.py.

When the LLM calls operation='replace' but forgets `replace_with` AND
provides a complete Python module as `content`, the tool should
overwrite the file instead of erroring. This eliminates a common wasted
turn observed in self-play logs.
"""

import pytest

from ghost_agent.tools.file_system import (
    _looks_like_complete_python_module,
    tool_replace_text,
)


# ------------------------------------------ _looks_like_complete_python_module


class TestLooksLikeCompleteModule:
    def test_complete_module_with_import_and_main(self):
        src = (
            "import json\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    print('hello')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        assert _looks_like_complete_python_module(src) is True

    def test_complete_module_with_import_and_class(self):
        src = (
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Foo:\n"
            "    x: int\n"
            "    y: int\n"
        )
        assert _looks_like_complete_python_module(src) is True

    def test_complete_module_with_import_and_multiple_defs(self):
        src = (
            "import math\n"
            "\n"
            "def area(r):\n"
            "    return math.pi * r * r\n"
            "\n"
            "def circumference(r):\n"
            "    return 2 * math.pi * r\n"
        )
        assert _looks_like_complete_python_module(src) is True

    def test_function_body_snippet_not_module(self):
        src = (
            "    for row in rows:\n"
            "        total += row.amount\n"
            "    return total\n"
        )
        assert _looks_like_complete_python_module(src) is False

    def test_imports_only_not_module(self):
        src = (
            "import os\n"
            "import sys\n"
            "import json\n"
            "import csv\n"
            "import math\n"
        )
        # No def/class/main → not a module for our purposes
        assert _looks_like_complete_python_module(src) is False

    def test_no_imports_not_module(self):
        src = (
            "def main():\n"
            "    print('hi')\n"
            "    print('world')\n"
            "\n"
            "main()\n"
        )
        assert _looks_like_complete_python_module(src) is False

    def test_too_short_not_module(self):
        src = "import os\ndef f(): pass\n"
        # Less than 5 non-blank lines
        assert _looks_like_complete_python_module(src) is False

    def test_syntax_error_not_module(self):
        src = (
            "import os\n"
            "def broken(:\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    broken()\n"
        )
        assert _looks_like_complete_python_module(src) is False

    def test_empty_not_module(self):
        assert _looks_like_complete_python_module("") is False
        assert _looks_like_complete_python_module(None) is False

    def test_single_line_not_module(self):
        assert _looks_like_complete_python_module("import os") is False


# ---------------------------------------------- tool_replace_text auto-promote


@pytest.mark.asyncio
class TestReplaceAutoPromote:
    async def test_missing_replace_with_complete_module_auto_promotes(self, tmp_path):
        """The logged failure mode: replace called without replace_with,
        content is a full Python script. Should write the file and return
        SUCCESS."""
        target = tmp_path / "solution.py"
        target.write_text("# old stub\nprint('old')\n")

        new_content = (
            "import csv\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    with open('data.csv') as f:\n"
            "        reader = csv.reader(f)\n"
            "        for row in reader:\n"
            "            print(row)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

        result = await tool_replace_text(
            filename="solution.py",
            old_text=new_content,
            new_text=None,
            sandbox_dir=tmp_path,
        )
        assert "SUCCESS" in result
        assert "auto-promoted" in result
        assert target.read_text() == new_content

    async def test_missing_replace_with_snippet_still_errors(self, tmp_path):
        """A code snippet (no imports at top level) must NOT auto-promote —
        that's a genuine targeted-replace call that forgot a parameter."""
        target = tmp_path / "solution.py"
        target.write_text(
            "import os\n\n"
            "def main():\n"
            "    print('hello')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

        snippet = (
            "    for row in rows:\n"
            "        total += row.amount\n"
            "    return total\n"
        )

        result = await tool_replace_text(
            filename="solution.py",
            old_text=snippet,
            new_text=None,
            sandbox_dir=tmp_path,
        )
        assert "SYSTEM INSTRUCTION" in result
        assert "replace_with" in result
        # File must not have been modified
        assert "def main" in target.read_text()

    async def test_auto_promote_only_for_py_files(self, tmp_path):
        """A .txt file with Python-like content should not auto-promote
        — the auto-promote heuristic is deliberately Python-only to avoid
        clobbering arbitrary text."""
        target = tmp_path / "notes.txt"
        target.write_text("old notes\n")

        module_looking = (
            "import csv\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    pass\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

        result = await tool_replace_text(
            filename="notes.txt",
            old_text=module_looking,
            new_text=None,
            sandbox_dir=tmp_path,
        )
        assert "SYSTEM INSTRUCTION" in result
        assert target.read_text() == "old notes\n"

    async def test_missing_replace_with_syntax_error_errors(self, tmp_path):
        """Broken Python must fail the ast.parse check and not promote."""
        target = tmp_path / "solution.py"
        target.write_text("# old\n")

        broken = (
            "import os\n"
            "def broken(:\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    broken()\n"
        )

        result = await tool_replace_text(
            filename="solution.py",
            old_text=broken,
            new_text=None,
            sandbox_dir=tmp_path,
        )
        assert "SYSTEM INSTRUCTION" in result

    async def test_normal_replace_still_works(self, tmp_path):
        """When both content and replace_with are provided, the normal
        targeted-replace path runs unchanged."""
        target = tmp_path / "solution.py"
        target.write_text(
            "import os\n"
            "\n"
            "def greeting():\n"
            "    return 'hello'\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(greeting())\n"
        )

        result = await tool_replace_text(
            filename="solution.py",
            old_text="return 'hello'",
            new_text="return 'bonjour'",
            sandbox_dir=tmp_path,
        )
        assert "SUCCESS" in result
        assert "bonjour" in target.read_text()
        assert "hello" not in target.read_text()

    async def test_auto_promote_keeps_module_with_fenced_docstring_whole(self, tmp_path):
        """C2 regression (2026-07-20): the auto-promote path called
        extract_code_from_markdown WITHOUT filename=, skipping the
        "whole input already parses -> keep it whole" guard — a complete
        module whose docstring embeds a ```python example was overwritten
        by just the inner snippet, and SUCCESS was reported."""
        target = tmp_path / "solution.py"
        target.write_text("# old stub\n")

        module = (
            "import os\n"
            "\n"
            "def helper():\n"
            '    """Return the cwd.\n'
            "\n"
            "    Example:\n"
            "\n"
            "    ```python\n"
            "    x = 1\n"
            "    ```\n"
            '    """\n'
            "    return os.getcwd()\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    helper()\n"
        )

        result = await tool_replace_text(
            filename="solution.py",
            old_text=module,
            new_text=None,
            sandbox_dir=tmp_path,
        )
        assert "SUCCESS" in result and "auto-promoted" in result
        written = target.read_text()
        assert written == module.strip()
        assert "def helper" in written
        assert "```python" in written          # fence survives inside the docstring
        assert written.strip() != "x = 1"      # the old bug: only the snippet

    async def test_two_arg_replacement_with_fenced_docstring_kept_whole(self, tmp_path):
        """Same missing-filename hole on the two-arg replace fence strip:
        a replacement block that is itself valid Python containing a fenced
        docstring example was collapsed to the inner snippet before
        matching — wrong content written, SUCCESS reported."""
        target = tmp_path / "mod.py"
        target.write_text(
            "import os\n"
            "\n"
            "def f():\n"
            "    return 1\n"
        )

        new_func = (
            "def f():\n"
            '    """Doc.\n'
            "\n"
            "    ```python\n"
            "    example()\n"
            "    ```\n"
            '    """\n'
            "    return 2\n"
        )
        result = await tool_replace_text(
            filename="mod.py",
            old_text="def f():\n    return 1",
            new_text=new_func,
            sandbox_dir=tmp_path,
        )
        assert "SUCCESS" in result
        written = target.read_text()
        assert "return 2" in written
        assert "```python" in written          # fenced example intact
        assert "import os" in written
        # The old bug: new_text collapsed to just `example()`.
        assert "def f():" in written

    async def test_aider_blocks_still_work(self, tmp_path):
        """Aider SEARCH/REPLACE blocks inside `content` (with
        replace_with=None) must continue to work as before."""
        target = tmp_path / "solution.py"
        target.write_text(
            "import os\n"
            "\n"
            "def greeting():\n"
            "    return 'hello'\n"
        )

        aider_block = (
            "<<<< SEARCH\n"
            "    return 'hello'\n"
            "====\n"
            "    return 'salut'\n"
            ">>>>\n"
        )

        result = await tool_replace_text(
            filename="solution.py",
            old_text=aider_block,
            new_text=None,
            sandbox_dir=tmp_path,
        )
        assert "SUCCESS" in result
        assert "salut" in target.read_text()
