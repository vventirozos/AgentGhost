from ghost_agent.tools.tool_failure import classify_tool_failure
from ghost_agent.tools.browser_support import browser_outcome_note, _runner_failure_hint
from ghost_agent.tools.host_memo import _dead_host_notice, _mark_host_failed, _HOST_FAILS
cases = {
 "socks": "Runner failed (exit 1): Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://foo.example/",
 "selector": "Runner failed (exit 1): Locator.click: Timeout 10000ms exceeded. waiting for locator('#x')",
 "nosuchel": "Runner failed (exit 1): selector 'h1' did not match any element",
 "gototo": "Runner failed (exit 1): Page.goto: Timeout 30000ms exceeded. navigating to https://x.com",
 "proxy": "Runner failed (exit 1): Page.goto: net::ERR_PROXY_CONNECTION_FAILED at https://x.com",
 "crash_in_locator": "Runner failed (exit 1): Locator.click: Target page, context or browser has been closed Call log: - waiting for locator('#a')",
 "sigterm": "Runner failed (exit 143): Page.goto: Timeout 30000ms exceeded.",
 "refused_local": "Runner failed (exit 1): Page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:3000/",
}
for k,v in cases.items():
    full = "--- BROWSER RESULT ---\nSTATUS: ERROR\n"+v+"\n\n--- HINT ---\n"+_runner_failure_hint(v)
    print(k, "|", classify_tool_failure(full)[0].value, "|", repr(browser_outcome_note(full)))
_mark_host_failed("https://a.com/", "Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://a.com/")
_mark_host_failed("https://a.com/", "Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://a.com/")
print(_dead_host_notice("https://a.com/x"))
n=_dead_host_notice("https://a.com/x")
print(classify_tool_failure("--- BROWSER RESULT ---\nSTATUS: ERROR\n"+n)[0].value, browser_outcome_note(n))
