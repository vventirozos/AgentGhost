import torch
import base64
import gc
import os
import time
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType

# Prevent memory fragmentation on Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- CONFIGURATION ---
MODEL_PATH = "./models/juggernaut_xl_lightning.safetensors"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"--- Loading Lightning Finetune: {MODEL_PATH} ---")

    global pipe
    global compel
    
    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # CRITICAL: Lightning models require trailing timesteps
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    # INITIALIZE COMPEL FOR SDXL
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    # Move pipeline to Apple Silicon GPU
    pipe.to("mps")
    
    # Save VRAM when decoding the image
    pipe.enable_vae_slicing()

    print("Warming up Apple Metal backend...")
    try:
        # Dummy prompt just to compile the shaders
        _ = pipe("warmup", num_inference_steps=1)
    except: 
        pass

    print("--- Server Ready on http://0.0.0.0:8000 ---")
    yield

app = FastAPI(lifespan=lifespan)

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    # Default steps dropped all the way to 4 for Lightning
    steps: int = 4

@app.post("/")
@app.post("/generate")
@app.post("/v1/images/generations")
async def generate(req: ImageRequest):
    start_time = time.time()
    gc.collect()
    torch.mps.empty_cache()

    try:
        # Keep Lightning steps between 3 and 8
        num_steps = max(3, min(8, int(req.steps)))

        print(f"Generating: {req.prompt[:50]}...")

        # 1. GENERATE EMBEDDINGS USING COMPEL
        conditioning, pooled = compel(req.prompt)
        
        # Move embeddings to MPS and cast to float16
        conditioning = conditioning.to("mps", dtype=torch.float16)
        pooled = pooled.to("mps", dtype=torch.float16)
        
        # Handle negative prompt embeddings
        neg_prompt_str = req.negative_prompt if req.negative_prompt else ""
        neg_conditioning, neg_pooled = compel(neg_prompt_str)
        
        # Move negative embeddings to MPS and cast to float16
        neg_conditioning = neg_conditioning.to("mps", dtype=torch.float16)
        neg_pooled = neg_pooled.to("mps", dtype=torch.float16)

        # 2. PASS EMBEDDINGS TO PIPELINE
        result = pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_conditioning,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=num_steps,
            # CRITICAL: Lightning needs CFG between 1.0 and 2.0
            guidance_scale=1.5,
            width=832,
            height=1216
        ).images[0]

        buffered = BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Cleanup
        del result
        del conditioning, pooled, neg_conditioning, neg_pooled
        gc.collect()
        torch.mps.empty_cache()

        gen_time = round(time.time() - start_time, 2)
        print(f"Completed in {gen_time}s")

        return {
            "time": gen_time,
            "data": [{"b64_json": img_str}]
        }

    except Exception as e:
        print(f"Error: {e}")
        torch.mps.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    