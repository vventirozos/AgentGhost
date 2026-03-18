import re

final_ai_content = """Here is the picture of the cat you requested!

![Image](/api/download/gen_123.png)

Let me know if you need anything else!"""

print("Original:")
print(final_ai_content)
print("---")

final_ai_content = re.sub(r'<tool_call.*?>.*?(?:</tool_call.*?>|$)', '', final_ai_content, flags=re.DOTALL | re.IGNORECASE)
final_ai_content = re.sub(r'<tool_response.*?>.*?(?:</tool_response.*?>|$)', '', final_ai_content, flags=re.DOTALL | re.IGNORECASE)
final_ai_content = re.sub(r'--- EXECUTION RESULT ---.*?(?:------------------------|$)', '', final_ai_content, flags=re.DOTALL)
final_ai_content = re.sub(r'(?m)^\s*(?:🔄|🟢|⏳|✅|❌|🛑|➖)\s*\[.*?\].*?\n?', '', final_ai_content)
final_ai_content = re.sub(r'(?m)^.*?\((?:IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\)\s*\n?', '', final_ai_content)
final_ai_content = re.sub(r'(?m)^\s*(?:\[)?task_\d+(?:\])?\s*\n?', '', final_ai_content)
final_ai_content = re.sub(r'(?m)^\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\s*', '', final_ai_content)
final_ai_content = final_ai_content.strip()

print("Scrubbed:")
print(final_ai_content)
