"""§4EC — `tools/file_system.py` read-budget arithmetic (§4CA B1/B2), driven as
the pure functions they are. The batch found `read_byte_budget`'s floor/ceiling
and `ReadBudget`'s clamping deletable-green: the existing pins drive them only
through file reads at one window size."""
import pytest

from ghost_agent.tools.file_system import ReadBudget, read_byte_budget


class TestReadByteBudget:
    @pytest.mark.parametrize("max_context,expected", [
        (240000, int(240000 * 3.5 * 0.40)),      # large window: 40% of it (336,000)
        (8000, int(8000 * 3.5 * 0.80)),          # small window: the 150 KB floor is BOUNDED by 80%
        (100000, 150000),                        # mid window: 0.40× = 140,000 → floor 150,000 wins, under the 0.8× ceiling
        (0, 1),                                  # degenerate window: never 0
    ])
    def test_floor_and_ceiling(self, max_context, expected):
        assert read_byte_budget(max_context) == expected

    def test_the_budget_never_exceeds_80_percent_of_the_window(self):
        for mc in (1000, 8000, 30000, 60000, 131072, 240000):
            assert read_byte_budget(mc) <= max(1, int(mc * 3.5 * 0.80))


class TestReadBudgetObject:
    def test_limit_is_clamped_and_remaining_never_negative(self):
        b = ReadBudget(-5)
        assert b.limit == 0 and b.remaining == 0
        b = ReadBudget(100); b.charge(150)
        assert b.spent == 150 and b.remaining == 0

    def test_charge_ignores_negative_amounts(self):
        b = ReadBudget(100); b.charge(30); b.charge(-50)
        assert b.spent == 30 and b.remaining == 70

    def test_charge_accumulates(self):
        b = ReadBudget(100); b.charge(30); b.charge(30)
        assert b.remaining == 40


# ── the read paths themselves (§4CA B1/B2/B3/B6) ─────────────────────────────
import pytest
from ghost_agent.tools.file_system import tool_read_file, tool_file_system


class TestWholeFileReadBudget:
    @pytest.mark.asyncio
    async def test_a_successful_read_charges_its_length(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello world\n" * 100)
        rb = ReadBudget(100000)
        out = await tool_read_file("a.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=rb)
        assert not out.startswith("Error") and rb.spent == len("hello world\n" * 100)

    @pytest.mark.asyncio
    async def test_the_three_refusal_wordings(self, tmp_path):
        """L1343-1375: fresh budget too small → 'only N KB … remains'; budget
        already spent → the exhausted wording; lockdown (0 remaining, nothing
        spent) → the near-the-ceiling wording."""
        (tmp_path / "big.txt").write_text("z" * 50000)
        fresh = await tool_read_file("big.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=ReadBudget(3000))
        assert fresh.startswith("Error") and "only 2.9 KB" in fresh and "remains this turn" in fresh
        spent = ReadBudget(60000); spent.charge(20000)
        exhausted = await tool_read_file("big.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=spent)
        assert exhausted.startswith("Error") and "already read" in exhausted.lower() or "budget" in exhausted
        assert "only 2.9 KB" not in exhausted
        locked = await tool_read_file("big.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=ReadBudget(0))
        assert locked.startswith("Error") and "context ceiling" in locked

    @pytest.mark.asyncio
    async def test_a_ranged_read_is_exempt_from_the_budget_and_not_charged(self, tmp_path):
        (tmp_path / "a.txt").write_text("\n".join(f"line {i}" for i in range(200)))
        rb = ReadBudget(0)
        out = await tool_read_file("a.txt", sandbox_dir=tmp_path, max_context=240000,
                                   start_line=10, end_line=12, read_budget=rb)
        assert not out.startswith("Error") and "line 10" in out and rb.spent == 0

    @pytest.mark.asyncio
    async def test_the_generated_file_sample_is_charged(self, tmp_path):
        """L1316-1317: the 4 KB sample of a machine-generated file counts."""
        (tmp_path / "data.txt").write_text(("0x" + "ab" * 4 + " ") * 40000)      # > 96 KB, hex-dense, long lines
        rb = ReadBudget(200000)
        out = await tool_read_file("data.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=rb)
        assert "SAMPLE ONLY" in out and rb.spent == 4096


class TestChunkedReadBudget:
    @pytest.mark.asyncio
    async def test_the_two_refusal_wordings(self, tmp_path):
        (tmp_path / "doc.txt").write_text("line\n" * 20000)
        locked = await tool_file_system(operation="read_chunked", path="doc.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=ReadBudget(0))
        assert "context ceiling" in str(locked) and "exhausted" not in str(locked)
        spent = ReadBudget(10); spent.charge(10)
        exhausted = await tool_file_system(operation="read_chunked", path="doc.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=spent)
        assert "exhausted" in str(exhausted) and "already read" in str(exhausted)

    @pytest.mark.asyncio
    async def test_a_failed_chunked_read_is_not_charged(self, tmp_path):
        rb = ReadBudget(500000)
        out = await tool_file_system(operation="read_chunked", path="missing.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=rb)
        assert str(out).startswith("Error") and rb.spent == 0

    @pytest.mark.asyncio
    async def test_a_successful_chunked_read_is_charged_its_length(self, tmp_path):
        (tmp_path / "doc.txt").write_text("line\n" * 20000)
        rb = ReadBudget(500000)
        out = await tool_file_system(operation="read_chunked", path="doc.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=rb)
        assert rb.spent == len(str(out))


@pytest.mark.asyncio
async def test_the_generated_file_sample_works_without_a_budget(tmp_path):
    """L1316 `if read_budget is not None` → True: `.charge` on None would crash
    every sampled read made by a caller that passes no budget."""
    (tmp_path / "data.txt").write_text(("0x" + "ab" * 4 + " ") * 40000)
    out = await tool_read_file("data.txt", sandbox_dir=tmp_path, max_context=240000, read_budget=None)
    assert "SAMPLE ONLY" in out
