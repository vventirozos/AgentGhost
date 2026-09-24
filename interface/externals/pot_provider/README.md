# PO-token helper (`com.local.ghost-pot`)

The YouTube route (`knowledge_base(action='transcribe', filename='<youtube url>')`,
code in `src/ghost_agent/memory/youtube_ingest.py`) runs yt-dlp over Tor. YouTube
answers a bare Tor exit with "Sign in to confirm you're not a bot"; a
proof-of-origin (PO) token minted by solving BotGuard locally gets ~4 in 10
circuits through, and the agent rotates circuits until one passes.

* Server: [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
  `server/`, built with `npm install && npx tsc`, installed at `~/Data/AI/PotProvider`
  (`UPSTREAM_COMMIT` records the commit). The matching yt-dlp plugin
  (`bgutil-ytdlp-pot-provider` on PyPI) is installed in the agent venv.
* Listens on `127.0.0.1:4416` only. yt-dlp passes its per-attempt Tor proxy in
  each request, so the helper's own fetches ride that circuit — no direct egress.
* Health: `curl -s http://127.0.0.1:4416/ping`. Override the URL for the agent
  with `GHOST_YT_POT_URL`; if the daemon is down the agent tries to spawn the
  server from `GHOST_YT_POT_DIR` (default the path above) as a fallback.
* Update: pull upstream, rebuild, `rsync` `build/ node_modules/ package.json`
  into `~/Data/AI/PotProvider`, then `sudo launchctl kickstart -k system/com.local.ghost-pot`;
  and `pip install -U yt-dlp bgutil-ytdlp-pot-provider` in the venv (the two
  must stay in step — YouTube rotates its challenge).
