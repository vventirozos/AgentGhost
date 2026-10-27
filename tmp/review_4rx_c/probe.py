import sys; sys.path.insert(0,'src')
from ghost_agent.tools import browser as B
import json, shlex
for t,n in [(30000,1),(30000,2),(5000,1),(60000,1)]:
    eff=max(t,t*n); st=B._bounded_subprocess_timeout(eff); tot=max(1000,(st-B._SUBPROCESS_SLACK_S)*1000)
    share=B._runner_action_timeout_ms(tot,n)
    for ex in (st,40,20):
        p={}; B.runner_cmd("r",p,"interact",min(ex,st),tot,B._SUBPROCESS_SLACK_S)
        print(t,n,"exec",min(ex,st),"share",share,"total",p["total_timeout_ms"],"pre4RX clock",share)
