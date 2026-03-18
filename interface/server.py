import asyncio
import re
import json
import logging
import signal
import argparse
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import os

GHOST_API_KEY = os.environ.get("GHOST_API_KEY", "ghost-secret-123")
PI_VOICE_URL = os.environ.get("PI_VOICE_URL", "http://raspberrypi.local:8000")

# Parse arguments
parser = argparse.ArgumentParser(description="Ghost Interface Server")
parser.add_argument("--agent-log", default="/Users/vasilis/AI/Logs/ghost-agent.log", help="Path to the agent log file")
args, unknown = parser.parse_known_args()
AGENT_LOG_PATH = args.agent_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GhostInterface")

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global set of connected websockets
connected_websockets = set()

async def log_streamer():
    """Reads agent logs and broadcasts them to connected clients."""
    process = await asyncio.create_subprocess_exec(
        "tail", "-n", "10", "-F", AGENT_LOG_PATH,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    logger.info(f"Started log streamer for {AGENT_LOG_PATH}")

    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                # Detect error for reactivity
                is_error = "ERROR" in decoded_line or "Exception" in decoded_line
                
                message = json.dumps({
                    "type": "log",
                    "content": decoded_line,
                    "is_error": is_error
                })
                
                # Broadcast to all connected clients
                to_remove = set()
                for ws in connected_websockets:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        to_remove.add(ws)
                
                for ws in to_remove:
                    connected_websockets.remove(ws)
                    
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_streamer())

@app.get("/")
async def get():
    return HTMLResponse(content=(static_dir / "index.html").read_text(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

@app.post("/api/chat")
async def chat_proxy(request: Request):
    """Proxies chat requests to the Ghost Agent."""
    try:
        body = await request.json()
        is_streaming = body.get("stream", False)
        
        # We need the client to stay open if we are returning a StreamingResponse
        # so we don't automatically close it with `async with` for the stream branch.
        if is_streaming:
            client = httpx.AsyncClient()
            
            async def stream_generator():
                try:
                    async with client.stream("POST", "http://localhost:8000/api/chat", json=body, headers={"X-Ghost-Key": GHOST_API_KEY}, timeout=600.0) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(chunk_size=None):
                            yield chunk
                finally:
                    await client.aclose()
                    
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
            return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:8000/api/chat", json=body, headers={"X-Ghost-Key": GHOST_API_KEY}, timeout=600.0)
                return response.json()
                
    except Exception as e:
        logger.error(f"Chat proxy error: {e}")
        return {"error": str(e)}

@app.post("/api/upload")
async def upload_proxy(file: UploadFile = File(...)):
    """Proxies file upload to the Ghost Agent."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/upload",
                files={"file": (file.filename, file.file, file.content_type)},
                headers={"X-Ghost-Key": GHOST_API_KEY},
                timeout=120.0
            )
            return response.json()
    except Exception as e:
        logger.error(f"Upload proxy error: {e}")
        return {"error": str(e)}

@app.get("/api/download/{filename:path}")
async def download_proxy(filename: str):
    """Proxies file download from the Ghost Agent."""
    try:
        client = httpx.AsyncClient(timeout=120.0)
        req = client.build_request("GET", f"http://localhost:8000/api/download/{filename}", headers={"X-Ghost-Key": GHOST_API_KEY})
        resp = await client.send(req, stream=True)
        resp.raise_for_status()

        headers = {}
        if "content-type" in resp.headers:
            headers["content-type"] = resp.headers["content-type"]
        if "content-disposition" in resp.headers:
            headers["content-disposition"] = resp.headers["content-disposition"]

        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(stream_generator(), headers=headers)
    except Exception as e:
        logger.error(f"Download proxy error: {e}")
        return {"error": str(e)}

@app.post("/api/stt")
async def stt_proxy(request: Request):
    """Proxies audio blobs to the Raspberry Pi for Speech-to-Text."""
    try:
        form = await request.form()
        file = form.get("file")
        file_content = await file.read()
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, file_content, file.content_type)}
            response = await client.post(f"{PI_VOICE_URL}/stt", files=files, timeout=60.0)
            if response.status_code != 200:
                logger.error(f"PI STT Error Body: {response.text}")
                response.raise_for_status()
            
            return response.json()
    except Exception as e:
        logger.error(f"STT proxy error: {e}")
        return {"error": str(e)}

@app.post("/api/tts")
async def tts_proxy(request: Request):
    """Proxies text chunks to the Raspberry Pi for Text-to-Speech."""
    try:
        body = await request.json()
        payload = {"text": body.get("text", "")}
        client = httpx.AsyncClient(timeout=60.0)
        req = client.build_request("POST", f"{PI_VOICE_URL}/tts", json=payload)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            await resp.aread()
            logger.error(f"PI 500 Error Body: {resp.text}")
            resp.raise_for_status()
        
        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
        
        return StreamingResponse(stream_generator(), media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS proxy error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
