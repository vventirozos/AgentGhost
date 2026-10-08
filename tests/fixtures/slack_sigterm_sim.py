"""§4ML: the real Slack bot main() under SIGTERM with N requests in flight
(fresh-reader R2 harness). argv: <scratch dir> <N> <say delay s> <port>."""
import os, sys, asyncio, time, signal, json
S = sys.argv[1]
PORT = int(sys.argv[4])
os.environ.update(SLACK_BOT_TOKEN="xoxb-fake", SLACK_APP_TOKEN="xapp-fake", GHOST_API_KEY="k"*40,
    GHOST_SLACKBOT_LOG="", GHOST_SLACK_REPLY_INDEX_PATH=S+"/ri.json", GHOST_AGENT_LOG=S+"/agent.log",
    GHOST_NOTIFY_SLACK_CHANNEL="off", GHOST_API_URL="http://127.0.0.1:%s/api/chat" % PORT)
open(S+"/agent.log","w").close()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "interface", "externals", "slack_bot"))
import main as M
T0 = time.monotonic()
said = []
class FakeSay:
    channel = "D123"
    def __init__(self, delay): self.delay = delay
    async def __call__(self, text=None, thread_ts=None, **kw):
        await asyncio.sleep(self.delay)
        said.append((round(time.monotonic()-T0,2), text[:60]))
        print("SAY", said[-1], flush=True)
        return {"channel": "D123", "ts": str(time.time())}
class FakeHandler:
    def __init__(self, *a, **k): pass
    async def start_async(self):
        await asyncio.sleep(float("inf"))
M.AsyncSocketModeHandler = FakeHandler
async def _owner(): return "U123"
M.resolve_owner_id = _owner
async def _bot():
    M.BOT_USER_ID = "UBOT"
M.get_bot_user_id = _bot
N = int(sys.argv[2]); DELAY = float(sys.argv[3])
async def hang_server():
    async def h(r, w):
        await asyncio.sleep(3600)
    return await asyncio.start_server(h, "127.0.0.1", PORT)
orig_main = M.main
async def wrapped():
    srv = await hang_server()
    for i in range(N):
        M._spawn_bg(M._process_message([{"role":"user","content":"hi %d"%i}], FakeSay(DELAY), thread_ts="1.%d"%i, requester="U123"))
    loop = asyncio.get_running_loop()
    loop.call_later(1.5, os.kill, os.getpid(), signal.SIGTERM)
    await orig_main()
sys.argv = ["main.py"]
try:
    asyncio.run(wrapped())
finally:
    print("EXIT at", round(time.monotonic()-T0,2), "said", len(said), flush=True)
