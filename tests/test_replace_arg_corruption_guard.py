"""Replace-reliability fixes (2026-07-05).

Trajectory measurement over 5 days: 99 file_system replace calls, 34 failed
"search block NOT found" — 27 of those 34 arrived with content ==
replace_with BYTE-IDENTICAL (the upstream native-tools argument transport
merged the two args, so the "search" text was the NEW block that doesn't
exist in the file yet). Another ~23 identical-args calls "succeeded" as
silent no-ops (the identical text still matched the file and was replaced
with itself), leaving the model convinced a fix was applied that never
happened.

Coverage here:
* Identical-args corruption guard — rejects fast with a teaching message,
  including the silent-no-op case where the text DOES exist in the file.
* SEARCH/REPLACE (aider) block loop — per-block no-op detection, and the
  full matching ladder (exact → flexible → fuzzy → anchor) that the
  two-argument form already had; the single-argument form is the one the
  registry now steers the model toward, so it must not be the weaker
  matcher.
* Legacy two-argument form still works (regression).
"""

import pytest

from ghost_agent.tools.file_system import tool_replace_text, _locate_block


@pytest.fixture
def sandbox_dir(tmp_path):
    return tmp_path


PY_FILE = '''def post_move(fen, user_move, history):
    """POST JSON to the server."""
    payload = json.dumps({"fen": fen, "user_move": user_move})
    data = payload.encode("utf-8")
    req = urllib.request.Request(API_URL, data=data)
    return req


def render_board(fen):
    rows = fen.split()[0].split("/")
    for row in rows:
        print(row)
'''


# --------------------------------------------------------------------------
# Identical-args corruption guard
# --------------------------------------------------------------------------

class TestIdenticalArgsGuard:
    @pytest.mark.asyncio
    async def test_identical_args_rejected(self, sandbox_dir):
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        new_block = 'def post_move(fen):\n    """CHANGED."""\n'
        res = await tool_replace_text("client.py", new_block, new_block,
                                      sandbox_dir)
        assert "REPLACE REJECTED" in res
        assert "BYTE-IDENTICAL" in res
        assert "<<<< SEARCH" in res            # teaches the immune form
        assert f.read_text() == PY_FILE        # untouched

    @pytest.mark.asyncio
    async def test_silent_noop_case_rejected(self, sandbox_dir):
        # The nastier variant: the identical text EXISTS in the file, so the
        # old code path returned SUCCESS having changed nothing.
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        existing = '    data = payload.encode("utf-8")'
        res = await tool_replace_text("client.py", existing, existing,
                                      sandbox_dir)
        assert "REPLACE REJECTED" in res
        assert f.read_text() == PY_FILE

    @pytest.mark.asyncio
    async def test_deletion_via_empty_replace_still_works(self, sandbox_dir):
        # replace_with="" is a legitimate delete, not the corruption shape.
        f = sandbox_dir / "client.py"
        f.write_text("keep\ndrop me\nkeep too\n")
        res = await tool_replace_text("client.py", "drop me\n", "",
                                      sandbox_dir)
        assert res.startswith("SUCCESS")
        assert f.read_text() == "keep\nkeep too\n"

    @pytest.mark.asyncio
    async def test_normal_two_arg_replace_unaffected(self, sandbox_dir):
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        res = await tool_replace_text(
            "client.py",
            '    """POST JSON to the server."""',
            '    """POST JSON to the server. Retries on 502."""',
            sandbox_dir)
        assert res.startswith("SUCCESS")
        assert "Retries on 502" in f.read_text()


# --------------------------------------------------------------------------
# SEARCH/REPLACE block loop: no-op detection + full matching ladder
# --------------------------------------------------------------------------

def _aider(search, replace):
    return f"<<<< SEARCH\n{search}\n====\n{replace}\n>>>>"


class TestAiderBlockLadder:
    @pytest.mark.asyncio
    async def test_exact_block_applies(self, sandbox_dir):
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        res = await tool_replace_text(
            "client.py",
            _aider('rows = fen.split()[0].split("/")',
                   'rows = fen.split(" ")[0].split("/")'),
            None, sandbox_dir)
        assert "SUCCESS: Applied 1 SEARCH/REPLACE" in res
        assert 'fen.split(" ")[0]' in f.read_text()

    @pytest.mark.asyncio
    async def test_noop_block_reported(self, sandbox_dir):
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        same = 'rows = fen.split()[0].split("/")'
        res = await tool_replace_text(
            "client.py", _aider(same, same), None, sandbox_dir)
        assert "NO-OP" in res
        assert f.read_text() == PY_FILE

    @pytest.mark.asyncio
    async def test_noop_block_does_not_sink_good_blocks(self, sandbox_dir):
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        same = "def render_board(fen):"
        good = _aider('    data = payload.encode("utf-8")',
                      '    data = payload.encode("ascii")')
        noop = _aider(same, same)
        res = await tool_replace_text(
            "client.py", good + "\n" + noop, None, sandbox_dir)
        assert "SUCCESS: Applied 1 SEARCH/REPLACE" in res
        assert "NO-OP" in res
        assert 'encode("ascii")' in f.read_text()

    @pytest.mark.asyncio
    async def test_fuzzy_rescue_in_block_form(self, sandbox_dir):
        # One misremembered token in a multi-line block: below the flexible
        # matcher (token mismatch), above the fuzzy threshold. Previously
        # the block form had no fuzzy rung and this failed outright.
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        drifted = ('    payload = json.dumps({"fen": fen, "move": user_move})\n'
                   '    data = payload.encode("utf-8")\n'
                   '    req = urllib.request.Request(API_URL, data=data)')
        replacement = ('    payload = json.dumps({"fen": fen})\n'
                       '    data = payload.encode("utf-8")\n'
                       '    req = urllib.request.Request(API_URL, data=data, '
                       'method="POST")')
        res = await tool_replace_text(
            "client.py", _aider(drifted, replacement), None, sandbox_dir)
        assert "SUCCESS: Applied 1 SEARCH/REPLACE" in res
        assert "rescued by tolerant matching" in res
        assert 'method="POST"' in f.read_text()

    @pytest.mark.asyncio
    async def test_unmatchable_block_still_reports_failure(self, sandbox_dir):
        f = sandbox_dir / "client.py"
        f.write_text(PY_FILE)
        res = await tool_replace_text(
            "client.py",
            _aider("this text has never existed anywhere in the file",
                   "replacement"),
            None, sandbox_dir)
        assert "None of the SEARCH/REPLACE blocks matched" in res
        assert f.read_text() == PY_FILE


# --------------------------------------------------------------------------
# _locate_block ladder unit tests
# --------------------------------------------------------------------------

class TestLocateBlock:
    def test_exact(self):
        assert _locate_block(PY_FILE, "def render_board(fen):") == (
            "def render_board(fen):", "exact")

    def test_flexible_whitespace(self):
        got = _locate_block(
            PY_FILE, "def  render_board(fen):\n     rows = "
                     'fen.split()[0].split("/")')
        assert got is not None
        assert got[1] == "flexible"

    def test_fuzzy_strategy_labelled(self):
        drifted = ('    payload = json.dumps({"fen": fen, "move": user_move})\n'
                   '    data = payload.encode("utf-8")\n'
                   '    req = urllib.request.Request(API_URL, data=data)')
        got = _locate_block(PY_FILE, drifted)
        assert got is not None
        assert got[1].startswith("fuzzy:")
        assert got[0] in PY_FILE               # real bytes from the file

    def test_no_match_returns_none(self):
        assert _locate_block(PY_FILE, "completely unrelated text zzz") is None
