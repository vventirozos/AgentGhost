Done. Three files in the sandbox:

**`logslow.py`** — reads `argv[1]`, extracts `req_id` (UUID) + seconds from every `request finished — +<N>s` line via regex, sorts by duration desc, prints the top 10 as `<req_id> <N>s`. Missing file → prints `error: file not found: ...` and exits 2; malformed lines are skipped, never crash.

**`test_logslow.py`** — 10 tests covering happy path, missing-file (exit 2), no-arg (exit 2), and malformed cases (no marker, no number, no UUID, non-numeric number).

**`sample.log`** — 13-line sample with one line missing the marker, one missing the number (`+s`), and one missing the UUID.

**Tests:** `10 passed in 0.01s`

**Sample output** (`python3 logslow.py sample.log`):
```
550e8400-0000-0000-0000-000000000a03 12.750s
550e8400-0000-0000-0000-000000000a12 9.999s
550e8400-0000-0000-0000-000000000a05 7.333s
550e8400-0000-0000-0000-000000000a07 5.555s
550e8400-0000-0000-0000-000000000a10 4.444s
550e8400-0000-0000-0000-000000000a01 2.345s
550e8400-0000-0000-0000-000000000a06 2.222s
550e8400-0000-0000-0000-000000000a08 1.111s
550e8400-0000-0000-0000-000000000a02 0.512s
550e8400-0000-0000-0000-000000000a11 0.250s
```
The three malformed lines were ignored and the script exited 0.