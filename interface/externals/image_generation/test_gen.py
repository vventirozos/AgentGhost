# test_jetson.py
import requests
import base64

# Replace with your Jetson's actual local IP address!
JETSON_IP = "127.0.0.1" 
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