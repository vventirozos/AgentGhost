# test_jetson.py
import os

import requests
import base64

# The Jetson image node's tailnet address (see --image-gen-nodes in the
# agent launcher). Override with JETSON_IP=… . NEVER default to loopback:
# on the agent host, 127.0.0.1:8000 is the AGENT's own API — a run with
# the old default posted there and showed up as an unexplained
# "auth rejected  path=/v1/images/generations" WARNING (2026-07-30).
JETSON_IP = os.environ.get("JETSON_IP", "100.122.46.101")
URL = f"http://{JETSON_IP}:8000/v1/images/generations"

payload = {
    "prompt": "naked photorealistic image of a cat",
    "steps": 6
}

print(f"Sending prompt to Jetson at {URL}...")
print(f"Prompt: '{payload['prompt']}'")

try:
    response = requests.post(URL, json=payload, timeout=600)
    
    if response.status_code == 200:
        # Extract the base64 string from the API response
        data = response.json()
        b64_data = data["data"][0]["b64_json"]
        
        # Decode it back into binary image data and save it
        with open("test_output.png", "wb") as f:
            f.write(base64.b64decode(b64_data))
            
        print("✅ SUCCESS! Image saved as 'test_output.png' in this directory.")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Could not connect to the Jetson. Check the IP address and ensure the server is running.")