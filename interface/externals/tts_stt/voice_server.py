import os
import uuid
import soundfile as sf
import torch
import numpy as np
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro import KPipeline

app = FastAPI(title="Orin Nano Optimized Voice Node")

# --- STT CONFIGURATION (Downsized for Stability) ---
# 'distil-small.en' is much lighter than Medium and fits better alongside Kokoro
print("Loading Whisper Distil-Small on GPU (int8_float16)...")
whisper_model = WhisperModel(
    "distil-small.en",             
    device="cuda",          
    compute_type="int8_float16" # More memory efficient than float16 alone
)
print("Whisper Loaded!")

# --- TTS CONFIGURATION ---
print("Loading Kokoro TTS...")
tts_pipeline = KPipeline(lang_code='a', device='cuda') 

# WARM-UP
print("Warming up...")
whisper_model.transcribe(np.zeros(16000, dtype=np.float32))
print("System Ready!")

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    temp_audio = f"/dev/shm/{uuid.uuid4()}_in.webm"
    try:
        with open(temp_audio, "wb") as f:
            f.write(await file.read())

        # Lowering beam_size to 1 reduces memory usage during inference
        segments, _ = whisper_model.transcribe(
            temp_audio,
            beam_size=1,        
            vad_filter=True,    
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text = " ".join([segment.text for segment in segments]).strip()
        
        # Manually trigger memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"🎙️ STT: {text}")
        return {"text": text}
    except Exception as e:
        print(f"❌ STT Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    print(f"🔊 TTS: {req.text}")
    out_path = f"/dev/shm/{uuid.uuid4()}_out.wav"

    try:
        generator = tts_pipeline(
            req.text, 
            voice='af_heart', 
            speed=1.1, 
            split_pattern=r'\n+'
        )
        
        audio_chunks = [audio for _, _, audio in generator]
        if not audio_chunks:
            raise ValueError("No audio generated")
            
        audio_combined = np.concatenate(audio_chunks)
        sf.write(out_path, audio_combined, 24000)

        # Cleanup before sending response
        gc.collect()
        torch.cuda.empty_cache()

        with open(out_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")

    except Exception as e:
        print(f"❌ TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    