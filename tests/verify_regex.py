import re

def test_regex_safety(filename):
    print(f"\nChecking: '{filename}'")
    raw_name = filename.strip()
    
    # EMULATING THE LOGIC IN MEMORY.PY
    # If it's a long sentence, try to extract a filename pattern
    if " " in raw_name and len(raw_name.split()) > 3:
         # Look for 'filename.ext' pattern inside quotes or standalone
         match = re.search(r"['\"`]+([\w\-\.]+\.[a-zA-Z]{2,4})['\"`]+", raw_name, re.IGNORECASE)
         if match:
             raw_name = match.group(1)
             print(f"  -> Extracted quote match: '{raw_name}'")
         else:
             # Fallback: Look for the last word if it looks like a filename
             words = raw_name.split()
             last_word = words[-1].strip("'\"`.")
             if "." in last_word:
                 raw_name = last_word
                 print(f"  -> Extracted last word: '{raw_name}'")

    raw_name = re.sub(r'^(Downloaded|File|Path|Document|Source|Text|Content|Of|The text of)\b\s*:?\s*', '', raw_name, flags=re.IGNORECASE)
    cleaned = raw_name.strip("'\"` ")
    print(f"  -> Final Cleaned: '{cleaned}'")
    return cleaned

cases = [
    "of romeo_source.txt",
    "The text of 'Romeo and Juliet' has been downloaded and is now stored...",
    "The text of 'romeo.txt' is saved.",
    "File 'romeo_source.txt' is ready.",
    "romeo_source.txt",
    "content: my_file.txt",
    "Downloaded data.csv successfully."
]

print("--- Testing Refined Regex Safety ---")
for c in cases:
    test_regex_safety(c)
