import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ghost_agent.utils.token_counter import QWEN_MODEL_ID, load_tokenizer

print(f"Checking Tokenizer Configuration...")
print(f"Expected ID: Qwen/Qwen2.5-Coder-7B-Instruct")
print(f"Actual ID:   {QWEN_MODEL_ID}")

if QWEN_MODEL_ID == "Qwen/Qwen2.5-Coder-7B-Instruct":
    print("✅ SUCCESS: Tokenizer ID updated.")
else:
    print("❌ FAILURE: Tokenizer ID mismatch.")
    sys.exit(1)
