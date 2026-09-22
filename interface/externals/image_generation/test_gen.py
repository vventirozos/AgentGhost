"""Manual smoke test for the Jetson image node (Qwen-Image-2.1 via sd.cpp).

    python test_gen.py                       # plain generation
    python test_gen.py --edit shot.png       # edit an existing image
    JETSON_IP=… GHOST_API_KEY=… python test_gen.py

⚠ This sends the fleet key. /generate has required `X-Ghost-Key` since
2026-07-15 and this script did not send one, so every run since then answered
401 — it also still asked for `steps: 6`, an LCM-era value the node floors at
15. Both fixed 2026-09-22 (§4JV).
"""
import argparse
import base64
import os
import time
from pathlib import Path

import requests

# The Jetson image node's tailnet address (see --image-gen-nodes in the
# agent launcher). Override with JETSON_IP=… . NEVER default to loopback:
# on the agent host, 127.0.0.1:8000 is the AGENT's own API — a run with
# the old default posted there and showed up as an unexplained
# "auth rejected  path=/v1/images/generations" WARNING (2026-07-30).
JETSON_IP = os.environ.get("JETSON_IP", "100.122.46.101")
URL = f"http://{JETSON_IP}:8000/v1/images/generations"


def _key() -> str:
    v = os.environ.get("GHOST_API_KEY")
    if v is not None:
        return v.strip()
    try:
        return (Path.home() / "Data" / "AI" / ".ghost_api_key").read_text().strip()
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt",
                    default='A rustic wooden sign that reads "GHOST" hanging above a doorway, warm light.')
    ap.add_argument("--edit", metavar="IMAGE",
                    help="edit this image instead of generating a new one; the prompt describes the CHANGE")
    ap.add_argument("--seed", type=int, help="omit for a random seed (the node reports the one it used)")
    ap.add_argument("--steps", type=int, help="omit for the node's default (30 new / 20 edit)")
    ap.add_argument("-o", "--out", default="test_output.png")
    args = ap.parse_args()

    payload = {"prompt": args.prompt}
    for field, value in (("seed", args.seed), ("steps", args.steps)):
        if value is not None:
            payload[field] = value
    if args.edit:
        payload["reference_images"] = [base64.b64encode(Path(args.edit).read_bytes()).decode()]

    # An edit runs guidance and takes ~11 min; a plain image ~3.3 min.
    print(f"→ {URL}\n  {'EDIT of ' + args.edit if args.edit else 'GENERATE'}: {args.prompt!r}")
    t0 = time.time()
    try:
        r = requests.post(URL, json=payload, timeout=1500,
                          headers={"X-Ghost-Key": _key()})
    except requests.exceptions.ConnectionError:
        print("❌ Connection error — check JETSON_IP and that ghost-image-node is running.")
        return 1

    if r.status_code != 200:
        print(f"❌ {r.status_code}: {r.text[:300]}")
        return 1

    data = r.json()
    Path(args.out).write_bytes(base64.b64decode(data["data"][0]["b64_json"]))
    print(f"✅ {args.out} — {data.get('width')}x{data.get('height')} "
          f"steps={data.get('steps')} seed={data.get('seed')} in {time.time() - t0:.0f}s")
    print(f"   reuse that seed to reproduce this image exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
