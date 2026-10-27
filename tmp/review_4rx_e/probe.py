from ghost_agent.tools.browser_support import runner_cmd, browser_outcome_note
from ghost_agent.tools import host_memo as HM
import json, shlex
for t in (15, 16, 18, 20, 25):
    p = {"timeout_ms": 30000}
    runner_cmd("r.py", p, "interact", t, 30000)
    print(t, p)
HM._LAST_CLEARNET_OK[0] = 0
t = ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\n--- PER-ACTION RESULTS ---\n"
     "  [0] OK extract_text: len=40 sel='#a'\n      TEXT: line one\nnet::ERR_NAME_NOT_RESOLVED in docs\n"
     "  [1] ERR click '#b': Timeout waiting for locator('#b')")
print(repr(browser_outcome_note(t)))
t2 = t.replace("net::ERR_NAME_NOT_RESOLVED in docs", "TargetClosedError explained")
print(repr(browser_outcome_note(t2)))
