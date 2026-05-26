#!/usr/bin/env python3
"""
🎬 Video-to-Prompt REST API
FastAPI server for AI agents to call programmatically.

Endpoints:
  GET  /health              — Health check + version
  GET  /api/models?url=...  — List models from llama.cpp API
  GET  /api/modes           — List prompt modes + example prompts
  POST /api/analyze         — Upload video, get prompt (multipart form)
  POST /api/analyze/json    — Analyze via JSON (video path or URL)
  GET  /api/video/info?path=... — Get video metadata
"""

import io
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from video_to_prompt_webui import (
    VERSION,
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    DEFAULT_FRAME_COUNT,
    DEFAULT_MAX_SIZE,
    DEFAULT_MAX_TOKENS,
    PROMPT_MODES,
    EXAMPLE_PROMPTS,
    extract_frames,
    create_frame_strip,
    frame_to_base64,
    get_video_info,
    clean_output,
    fetch_models,
)

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Video-to-Prompt API",
    description="Extract AI prompts from video using Vision LLM (llama.cpp API)",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    success: bool
    video_info: dict = {}
    frames_extracted: int = 0
    timestamps: list[float] = []
    mode: str = ""
    model: str = ""
    prompt: str = ""
    tokens_used: Optional[dict] = None
    elapsed_seconds: float = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = VERSION
    endpoints: dict = {
        "webui": "http://0.0.0.0:7860",
        "api": "http://0.0.0.0:8000",
        "docs": "http://0.0.0.0:8000/docs",
    }


class ModelsResponse(BaseModel):
    api_url: str
    models: list[dict]
    count: int


class ModesResponse(BaseModel):
    modes: dict
    examples: dict


class VideoInfoResponse(BaseModel):
    path: str
    info: dict


# ── Helpers ─────────────────────────────────────────────────────────
def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to temp directory."""
    suffix = Path(upload.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name


def _build_prompt(mode: str, custom: str, n_frames: int) -> tuple[str, str]:
    """Build prompt from mode or custom. Returns (prompt_text, display_mode)."""
    if custom and custom.strip():
        return custom.strip(), "Custom"

    for mode_name, mode_data in PROMPT_MODES.items():
        if mode_data["key"] == mode:
            return mode_data["prompt"].format(n=n_frames), mode_name

    # Default to describe
    first = list(PROMPT_MODES.keys())[0]
    return PROMPT_MODES[first]["prompt"].format(n=n_frames), first


# ── Endpoints ───────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check + version info."""
    return HealthResponse()


@app.get("/api/models", response_model=ModelsResponse)
async def list_models(
    api_url: str = Query(default=DEFAULT_API_URL, description="llama.cpp API base URL"),
):
    """Fetch available models from llama.cpp API.

    Example: GET /api/models?url=http://192.168.3.177:8080/v1/chat/completions
    """
    raw = fetch_models(api_url)
    models = []
    for display, model_id in raw:
        has_vision = "👁️" in display
        has_completion = "💬" in display
        # Parse size from display: "... (26.5GB)"
        size_gb = None
        if " (" in display and "GB)" in display:
            try:
                size_gb = float(display.split("(")[-1].replace("GB)", "").strip())
            except ValueError:
                pass
        models.append({
            "id": model_id,
            "display": display,
            "vision": has_vision,
            "completion": has_completion,
            "size_gb": size_gb,
        })
    return ModelsResponse(api_url=api_url, models=models, count=len(models))


@app.get("/api/modes", response_model=ModesResponse)
async def list_modes():
    """List all prompt modes and example prompts."""
    modes_out = {}
    for display_name, data in PROMPT_MODES.items():
        modes_out[data["key"]] = {
            "display": display_name,
            "prompt_template": data["prompt"][:200] + "...",
        }
    return ModesResponse(modes=modes_out, examples=EXAMPLE_PROMPTS)


@app.get("/api/video/info", response_model=VideoInfoResponse)
async def video_info(path: str = Query(..., description="Path to video file")):
    """Get video metadata (resolution, duration, fps, codec).

    Example: GET /api/video/info?path=/path/to/video.mp4
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Video not found: {path}")
    info = get_video_info(path)
    if not info:
        raise HTTPException(status_code=400, detail="Cannot read video file")
    return VideoInfoResponse(path=path, info=info)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    api_url: str = Form(default=DEFAULT_API_URL, description="llama.cpp API endpoint"),
    api_key: str = Form(default="", description="Optional API key (sent as Bearer token)"),
    model: str = Form(default=DEFAULT_MODEL, description="Model name"),
    mode: str = Form(default="describe", description="Prompt mode key"),
    custom_prompt: str = Form(default="", description="Custom prompt (overrides mode)"),
    frame_count: int = Form(default=DEFAULT_FRAME_COUNT, ge=2, le=64),
    max_size: int = Form(default=DEFAULT_MAX_SIZE, ge=256, le=2048),
    max_tokens: int = Form(default=DEFAULT_MAX_TOKENS, ge=128, le=8192),
    temperature: float = Form(default=0.6, ge=0.1, le=1.5),
    quality: int = Form(default=80, ge=30, le=100),
):
    """Analyze a video file — upload via multipart/form-data.

    ```bash
    curl -X POST http://localhost:8000/api/analyze \\
      -F "file=@video.mp4" \\
      -F "mode=describe" \\
      -F "api_url=http://192.168.3.177:8080/v1/chat/completions" \\
      -F "api_key=sk-xxx"  # optional
    ```
    """
    t0 = time.time()
    video_path = _save_upload(file)

    try:
        # Get video info
        info = get_video_info(video_path)
        if not info:
            return AnalyzeResponse(success=False, error="Cannot read video file")

        # Extract frames
        frames, timestamps = extract_frames(video_path, frame_count, max_size)
        if not frames:
            return AnalyzeResponse(success=False, error="No frames extracted")

        # Build prompt
        prompt_text, display_mode = _build_prompt(mode, custom_prompt, len(frames))

        # Build API payload
        content = [{"type": "text", "text": prompt_text}]
        for frame in frames:
            b64 = frame_to_base64(frame, quality)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }

        # Build headers
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        # Call API
        resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # Extract + clean
        message = data.get("choices", [{}])[0].get("message", {})
        raw = message.get("content", "") or message.get("reasoning_content", "")
        cleaned = clean_output(raw)

        elapsed = time.time() - t0

        return AnalyzeResponse(
            success=True,
            video_info=info,
            frames_extracted=len(frames),
            timestamps=timestamps,
            mode=display_mode,
            model=model,
            prompt=cleaned,
            tokens_used=data.get("usage"),
            elapsed_seconds=round(elapsed, 2),
        )

    except requests.exceptions.ConnectionError:
        return AnalyzeResponse(success=False, error=f"Cannot connect to API: {api_url}")
    except requests.exceptions.Timeout:
        return AnalyzeResponse(success=False, error="API request timed out (300s)")
    except Exception as e:
        return AnalyzeResponse(success=False, error=str(e))
    finally:
        # Cleanup temp file
        try:
            os.unlink(video_path)
        except Exception:
            pass


@app.post("/api/analyze/json", response_model=AnalyzeResponse)
async def analyze_video_json(body: dict):
    """Analyze a video via JSON body (use 'path' for local file or 'url' for remote).

    ```json
    POST /api/analyze/json
    {
      "path": "/path/to/video.mp4",
      "mode": "describe",
      "api_url": "http://192.168.3.177:8080/v1/chat/completions",
      "api_key": "sk-xxx",
      "model": "model-name.gguf",
      "frame_count": 16
    }
    ```
    """
    t0 = time.time()

    video_path = body.get("path", body.get("url", ""))
    api_url = body.get("api_url", DEFAULT_API_URL)
    api_key = body.get("api_key", "")
    model = body.get("model", DEFAULT_MODEL)
    mode = body.get("mode", "describe")
    custom_prompt = body.get("custom_prompt", "")
    frame_count = body.get("frame_count", DEFAULT_FRAME_COUNT)
    max_size = body.get("max_size", DEFAULT_MAX_SIZE)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.6)
    quality = body.get("quality", 80)

    if not video_path or not os.path.exists(video_path):
        return AnalyzeResponse(success=False, error=f"Video not found: {video_path}")

    try:
        info = get_video_info(video_path)
        if not info:
            return AnalyzeResponse(success=False, error="Cannot read video file")

        frames, timestamps = extract_frames(video_path, frame_count, max_size)
        if not frames:
            return AnalyzeResponse(success=False, error="No frames extracted")

        prompt_text, display_mode = _build_prompt(mode, custom_prompt, len(frames))

        content = [{"type": "text", "text": prompt_text}]
        for frame in frames:
            b64 = frame_to_base64(frame, quality)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }

        # Build headers
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        message = data.get("choices", [{}])[0].get("message", {})
        raw = message.get("content", "") or message.get("reasoning_content", "")
        cleaned = clean_output(raw)

        elapsed = time.time() - t0

        return AnalyzeResponse(
            success=True,
            video_info=info,
            frames_extracted=len(frames),
            timestamps=timestamps,
            mode=display_mode,
            model=model,
            prompt=cleaned,
            tokens_used=data.get("usage"),
            elapsed_seconds=round(elapsed, 2),
        )

    except requests.exceptions.ConnectionError:
        return AnalyzeResponse(success=False, error=f"Cannot connect to API: {api_url}")
    except Exception as e:
        return AnalyzeResponse(success=False, error=str(e))


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Video-to-Prompt API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   🎬 Video-to-Prompt API Server v{VERSION}                ║
╠══════════════════════════════════════════════════════════╣
║  📡 API:    http://{args.host}:{args.port}                   
║  📖 Docs:   http://{args.host}:{args.port}/docs              
║  📋 Redoc:  http://{args.host}:{args.port}/redoc             
║  ❤️  Health: http://{args.host}:{args.port}/health            
╚══════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
