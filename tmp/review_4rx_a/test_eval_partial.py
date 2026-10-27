from tests.test_4ln_browser import _run

async def test_eval_in_flight(tmp_path):
    res = await _run(tmp_path, {
        "final_url": "file:///workspace/x.html", "final_title": "", "aborted": False,
        "budget_exhausted": True, "budget_ms": 570000, "not_run": 3,
        "actions": [{"index": 0, "action": "evaluate", "ok": True, "value": "1"},
                    {"index": 1, "action": "evaluate", "ok": False,
                     "error": "TimeoutError: the sequence ran out of its 570000 ms total budget during this step"}]},
        operation="interact", url="file:///workspace/x.html",
        actions=[{"action": "evaluate", "js": "1"}] * 5)
    print(str(res)[:600]); print(getattr(res, "status", None))
    assert 0

async def test_first_in_flight(tmp_path):
    res = await _run(tmp_path, {
        "final_url": "file:///workspace/x.html", "final_title": "", "aborted": False,
        "budget_exhausted": True, "budget_ms": 570000, "not_run": 4,
        "actions": [{"index": 0, "action": "sleep", "ok": False,
                     "error": "TimeoutError: the sequence ran out of its 570000 ms total budget during this step"}]},
        operation="interact", url="file:///workspace/x.html",
        actions=[{"action": "sleep", "ms": 1}] * 5)
    print(str(res)[:300]); assert 0
