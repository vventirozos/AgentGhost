import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ghost_agent.utils.token_counter import QWEN_MODEL_ID, load_tokenizer

print(f"Checking Tokenizer Configuration...")
print(f"Expected ID: Qwen/Qwen3.5-27B")
print(f"Actual ID:   {QWEN_MODEL_ID}")

if QWEN_MODEL_ID == "Qwen/Qwen3.5-27B":
    print("✅ SUCCESS: Tokenizer ID updated.")
else:
    print("❌ FAILURE: Tokenizer ID mismatch.")
    sys.exit(1)
