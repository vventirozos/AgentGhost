import asyncio
import uuid
import base64
from pathlib import Path
from ..utils.logging import Icons, pretty_log

async def tool_generate_image(prompt: str = "", llm_client=None, sandbox_dir=None, steps: int = 50, **kwargs):
    # --- PARAMETER HALLUCINATION HEALING ---
    prompt = prompt or kwargs.get("image") or kwargs.get("description") or kwargs.get("subject") or kwargs.get("text")
    if not prompt:
        return "SYSTEM ERROR: The 'prompt' parameter is MANDATORY for image generation. You must provide a description of the image."

    # Enforce step limits
    steps = int(steps)
    steps = max(40, min(50, steps))
    try:
        pretty_log("Image Gen", f"Prompt: {prompt[:30]}...", icon="🎨")
        
        if not getattr(llm_client, 'image_gen_clients', None):
            return "ERROR: Image generation node is offline or not configured."
        
        payload = {"prompt": prompt, "steps": steps}
        resp_data = await llm_client.generate_image(payload)
        
        b64_str = resp_data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_str)
        
        filename = f"gen_{uuid.uuid4().hex[:8]}.png"
        file_path = sandbox_dir / filename
        await asyncio.to_thread(file_path.write_bytes, image_bytes)
        
        return f"SUCCESS: Image generated and saved to sandbox. DO NOT CALL THIS TOOL AGAIN with the same prompt. Respond DIRECTLY to the user by including this exact markdown to display the image: ![Image](/api/download/{filename})"
    except Exception as e:
        return f"ERROR generating image: {str(e)}"
