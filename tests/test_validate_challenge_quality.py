"""Unit tests for the challenge quality gate in dream.py.

Covers both reject conditions:
  1. Validator uses random data generation (seed/randint/uniform/etc).
  2. Validator doesn't reference any filename the setup_script writes.
"""

import pytest

from ghost_agent.core.dream import (
    validate_challenge_quality,
    _extract_filename_literals,
)


class TestExtractFilenameLiterals:
    def test_picks_csv_and_json(self):
        src = 'open("products.csv", "w"); f = open("data.json"); x = "ignore.me"'
        files = _extract_filename_literals(src)
        assert "products.csv" in files
        assert "data.json" in files
        assert "ignore.me" not in files

    def test_supports_various_extensions(self):
        src = 'with open("a.csv"), open("b.tsv"), open("c.sqlite"), open("d.parquet"): pass'
        files = _extract_filename_literals(src)
        assert files == {"a.csv", "b.tsv", "c.sqlite", "d.parquet"}

    def test_empty_source_returns_empty(self):
        assert _extract_filename_literals("") == set()
        assert _extract_filename_literals(None) == set()

    def test_ignores_non_file_strings(self):
        src = 'print("hello world"); x = "foo bar baz"'
        assert _extract_filename_literals(src) == set()


class TestValidatorRejectsRandomDataGen:
    def test_rejects_random_seed(self):
        setup = 'open("sales.csv", "w").write("a,b,c\\n1,2,3\\n")'
        validator = (
            'import random\n'
            'random.seed(42)\n'
            'for i in range(10): x = random.randint(1, 10)\n'
            'import subprocess\n'
            'subprocess.run(["python3", "solution.py"])\n'
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "random.seed" in reason

    def test_rejects_randint(self):
        setup = 'open("a.csv", "w").close()'
        validator = 'import random\nx = random.randint(1, 10)\nopen("a.csv")'
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "random.randint" in reason

    def test_rejects_random_uniform(self):
        setup = 'open("a.csv", "w").close()'
        validator = 'import random\nprice = random.uniform(10.0, 500.0)\nopen("a.csv")'
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "random.uniform" in reason

    def test_rejects_numpy_random(self):
        setup = 'open("a.csv", "w").close()'
        validator = 'import numpy as np\narr = np.random.randn(10)\nopen("a.csv")'
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "np.random" in reason or "numpy.random" in reason


class TestValidatorRejectsDisconnectedFiles:
    def test_rejects_no_shared_filename(self):
        setup = 'with open("products.csv", "w") as f:\n    f.write("id,name\\n1,widget\\n")'
        validator = (
            'import subprocess\n'
            'import csv\n'
            'result = subprocess.run(["python3", "solution.py"], capture_output=True)\n'
            'print("ok")\n'
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "products.csv" in reason

    def test_accepts_shared_filename(self):
        setup = 'with open("products.csv", "w") as f:\n    f.write("id,name\\n1,widget\\n")'
        validator = (
            'import subprocess\n'
            'import csv\n'
            'with open("products.csv") as f:\n'
            '    rows = list(csv.DictReader(f))\n'
            'result = subprocess.run(["python3", "solution.py"], capture_output=True)\n'
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True
        assert reason == ""

    def test_accepts_empty_setup(self):
        """No setup_script means no mock files — validator can do anything
        (as long as it doesn't generate its own random data)."""
        validator = 'import subprocess\nsubprocess.run(["python3", "solution.py"])'
        ok, reason = validate_challenge_quality("", validator)
        assert ok is True


class TestMissingValidator:
    def test_empty_validator_rejected(self):
        ok, reason = validate_challenge_quality("setup", "")
        assert ok is False
        assert "missing" in reason.lower() or "validation_script" in reason


class TestRealWorldRegressionCase:
    """The exact failure mode from the 20-minute wasted session: validator
    regenerates transactions in memory with random.seed(42), ignoring the
    CSV files the setup script wrote."""

    def test_sales_data_analysis_regression(self):
        setup = '''
import csv, random
random.seed(42)
with open("transactions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["product_id", "quantity", "price"])
    for _ in range(30):
        writer.writerow([random.randint(1, 10), random.randint(1, 5), random.uniform(10, 500)])
'''
        validator = '''
import subprocess, csv, random
random.seed(42)
products = {}
categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home']
for i in range(1, 11):
    products[i] = {'name': f'Product_{i}', 'category': categories[(i-1) % 5]}
transactions = []
for i in range(30):
    transactions.append({
        'product_id': random.randint(1, 10),
        'quantity': random.randint(1, 5),
        'price': round(random.uniform(10.0, 500.0), 2),
    })
result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
'''
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        # Must be caught by the random-data-gen rule (comes first)
        assert "random" in reason


class TestValidatorRejectsUnwinnableSplitPattern:
    """Regression: the 2026-04-17 09:07 self-play log shows the solver
    spending 10+ turns proving that
        act = out.stdout.strip().split('\\n')
        if len(act) != len(exp): ...
    is unwinnable whenever `exp == []`, because Python's
    `''.strip().split('\\n')` returns `['']` (length 1), never `[]`.
    Combined with a random dataset it's a coin-flip trap.

    The gate must reject this pattern at generation time — but ONLY
    when the setup uses randomness (deterministic setups let the
    author guarantee a non-empty result and avoid the bug)."""

    def test_rejects_split_pattern_with_random_setup(self):
        setup = 'import random\nrandom.seed(42)\nwith open("orders.csv","w") as f: f.write(f"a\\n")'
        validator = (
            "import subprocess, csv\n"
            "with open('orders.csv') as f: rows = list(csv.DictReader(f))\n"
            "exp = [r['id'] for r in rows if int(r.get('qty', 0)) > 5]\n"
            "out = subprocess.run(['python3','solution.py'], capture_output=True, text=True)\n"
            "act = out.stdout.strip().split('\\n')\n"
            "if len(act) != len(exp): exit(1)\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "unwinnable" in reason
        assert "splitlines()" in reason  # suggested fix is in the feedback

    def test_accepts_split_pattern_with_deterministic_setup(self):
        # No randomness → author guarantees non-empty `exp`, the
        # `['']` vs `[]` mismatch can never trigger. Not a false
        # positive for the gate.
        setup = 'with open("orders.csv","w") as f: f.write("id,qty\\n1,10\\n2,7\\n3,8\\n")'
        validator = (
            "import subprocess, csv\n"
            "with open('orders.csv') as f: rows = list(csv.DictReader(f))\n"
            "exp = [r['id'] for r in rows]\n"
            "out = subprocess.run(['python3','solution.py'], capture_output=True, text=True)\n"
            "act = out.stdout.strip().split('\\n')\n"
            "if len(act) != len(exp): exit(1)\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True, f"false-reject on deterministic data: {reason}"

    def test_accepts_splitlines_with_random_setup(self):
        # `splitlines()` returns `[]` for empty input (unlike
        # `split('\\n')`), so the bug doesn't apply. Must not reject.
        setup = 'import random\nrandom.seed(1)\nwith open("orders.csv","w") as f: f.write("id\\n")'
        validator = (
            "import subprocess, csv\n"
            "with open('orders.csv') as f: rows = list(csv.DictReader(f))\n"
            "exp = [r['id'] for r in rows]\n"
            "out = subprocess.run(['python3','solution.py'], capture_output=True, text=True)\n"
            "act = out.stdout.splitlines()\n"
            "if len(act) != len(exp): exit(1)\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True, f"false-reject on splitlines variant: {reason}"

    def test_accepts_split_without_len_compare(self):
        # `split('\\n')` alone (with element-wise assertion) is fine —
        # the bug only manifests when `len(act) != len(exp)` is the
        # gate. Field-by-field asserts handle empty correctly.
        setup = 'import random\nrandom.seed(1)\nwith open("orders.csv","w") as f: f.write("id\\n")'
        validator = (
            "import subprocess, csv\n"
            "with open('orders.csv') as f: rows = list(csv.DictReader(f))\n"
            "exp = {r['id'] for r in rows}\n"
            "out = subprocess.run(['python3','solution.py'], capture_output=True, text=True)\n"
            "act = set(out.stdout.strip().split('\\n'))\n"
            "assert act == exp\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True, f"false-reject on non-len compare: {reason}"


# ── datetime import-style lint (2026-07-18) ──────────────────────────────────
# Two overnight cycles died on the same generated-script bug family:
# `from datetime import datetime` + `datetime.timedelta(...)` (setup crashed
# pre-attempt) and a `datetime.datetime.*` module-attr misuse in a validator
# (crashed at SCORE time, charging the agent for a generator bug).

from ghost_agent.core.dream import _datetime_misuse


class TestDatetimeMisuseLint:
    def test_class_import_then_module_attr_flagged(self):
        src = (
            "from datetime import datetime\n"
            "start = datetime(2026, 1, 1)\n"
            "ts = start + datetime.timedelta(seconds=30)\n"
        )
        msg = _datetime_misuse(src)
        assert "datetime.timedelta" in msg

    def test_double_qualified_module_attr_flagged(self):
        src = (
            "import datetime\n"
            "delta = datetime.datetime.timedelta(days=1)\n"
        )
        msg = _datetime_misuse(src)
        assert "datetime.datetime.timedelta" in msg

    def test_module_import_style_clean(self):
        src = (
            "import datetime\n"
            "t = datetime.datetime.strptime('2026-01-01', '%Y-%m-%d')\n"
            "d = datetime.timedelta(days=2)\n"
            "now = datetime.datetime.now()\n"
        )
        assert _datetime_misuse(src) == ""

    def test_class_import_style_clean(self):
        src = (
            "from datetime import datetime, timedelta\n"
            "t = datetime.strptime('2026-01-01', '%Y-%m-%d')\n"
            "d = timedelta(days=2)\n"
        )
        assert _datetime_misuse(src) == ""

    def test_both_import_styles_present_is_ambiguous_and_skipped(self):
        # `import datetime` AND `from datetime import datetime` (rebinding) —
        # resolution order is textual, not static; don't guess.
        src = (
            "import datetime\n"
            "from datetime import datetime\n"
            "d = datetime.timedelta(days=1)\n"
        )
        assert _datetime_misuse(src) == ""

    def test_syntax_error_returns_clean(self):
        # syntax problems are the syntax gate's job
        assert _datetime_misuse("def broken(:\n    pass") == ""

    def test_quality_gate_rejects_setup_with_misuse(self):
        setup = (
            "from datetime import datetime\n"
            "import csv\n"
            "with open('transaction_log.csv', 'w') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['ts'])\n"
            "    w.writerow([str(datetime(2026,1,1) + datetime.timedelta(seconds=5))])\n"
        )
        validator = (
            "with open('transaction_log.csv') as f:\n"
            "    data = f.read()\n"
            "import subprocess, sys\n"
            "sys.exit(0)\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert not ok
        assert "setup_script" in reason
        assert "timedelta" in reason

    def test_quality_gate_rejects_validator_with_misuse(self):
        setup = (
            "import csv\n"
            "with open('transaction_log.csv', 'w') as f:\n"
            "    csv.writer(f).writerow(['ts'])\n"
        )
        validator = (
            "import datetime\n"
            "with open('transaction_log.csv') as f:\n"
            "    ts = datetime.datetime.timezone\n"
            "import sys\n"
            "sys.exit(0)\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert not ok
        assert "validation_script" in reason
