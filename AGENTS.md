# AGENTS.md — AI Agent Instructions for Video-to-Prompt

> If you're an AI coding agent reading this, follow these instructions to use the video-to-prompt tool.

## Overview

**Video-to-Prompt** extracts AI prompts from video files using a Vision Language Model (VLM). It sends frames to a llama.cpp API (OpenAI-compatible `/v1/chat/completions`) and returns descriptions, tags, or analysis.

- **WebUI**: Gradio interface at `video_to_prompt_webui.py`
- **CLI**: Standalone script at `video_to_prompt.py`
- **Install**: `curl -fsSL https://raw.githubusercontent.com/nanofatdog/video-to-prompt/main/install.sh | bash`

## Quick Start (CLI)

```bash
python3 video_to_prompt.py video.mp4                          # Describe mode (default)
python3 video_to_prompt.py video.mp4 -n 32 -m tag            # 32 frames, tags
python3 video_to_prompt.py video.mp4 --custom "Your prompt"  # Custom prompt
python3 video_to_prompt.py video.mp4 --json -o result.json   # JSON output
```

## Quick Start (WebUI)

```bash
python3 video_to_prompt_webui.py              # http://localhost:7860
python3 video_to_prompt_webui.py --port 8080  # Custom port
python3 video_to_prompt_webui.py --share      # Public Gradio link
```

## API Format

The tool sends frames to a llama.cpp vision endpoint using OpenAI-compatible multimodal format:

```python
POST {api_url}/v1/chat/completions
Content-Type: application/json

{
  "model": "model-name.gguf",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe these video frames..."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }],
  "max_tokens": 1024,
  "temperature": 0.6,
  "top_p": 0.9
}
```

### API Requirements

- Endpoint must support `/v1/models` (for model discovery) and `/v1/chat/completions` (for inference)
- Model must have **vision/multimodal** capability (mmproj loaded)
- Default endpoint: `http://192.168.3.177:8080/v1/chat/completions`

## Video Processing Pipeline

```
Video File → ffprobe (metadata) → ffmpeg (extract frames uniformly)
           → resize to max_size → base64 JPEG encode
           → POST to API with multimodal payload
           → clean thinking/reasoning artifacts → return text
```

### Frame Extraction

- Uses `ffmpeg` with `fps={frame_count/duration}` filter for uniform sampling
- Falls back to OpenCV if ffmpeg fails
- Frames resized to `max_size` (default 1280px) preserving aspect ratio
- Converted to base64 JPEG at configurable quality (default 80%)

### Output Cleaning

The tool automatically strips reasoning/thinking artifacts from models that output chain-of-thought. See `clean_output()` in the source.

## Available Prompt Modes

| Mode key | Description | Output |
|----------|-------------|--------|
| `describe` | Visual description for AI image gen | Paragraph |
| `summarize` | Brief video summary | Short text |
| `tag` | Comma-separated keywords | `tag1, tag2, ...` |
| `booru` | Danbooru-style tags | `tag1 tag2 ...` |
| `nsfw_check` | Content safety rating | `S,V,N` (1-10 each) |

## Example Prompts (WebUI Quick Fill)

| Prompt | Use Case |
|--------|----------|
| 🎬 Movie Director Shot List | Cinematography analysis |
| 📸 Photo/Art Style Analysis | Photography style breakdown |
| 🎨 Dominant Color Palette | Hex codes + color theory |
| 🔍 Object & Brand Detection | Forensic content inventory |
| 📝 Social Media Caption | Platform-specific captions |

## Configuration

| Param | Default | Range | Description |
|-------|---------|-------|-------------|
| `frame_count` | 16 | 2–64 | Frames to sample |
| `max_size` | 1280 | 256–2048 | Max dimension in px |
| `max_tokens` | 1024 | 128–8192 | Output token limit |
| `temperature` | 0.6 | 0.1–1.5 | Sampling creativity |
| `quality` | 80 | 30–100 | JPEG compression |

## Testing

```bash
# Generate a test video (3 color frames)
python3 -c "
import cv2, numpy as np
for color, bgr in [('red', (0,0,255)), ('green', (0,255,0)), ('blue', (255,0,0))]:
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    img[:] = bgr
    # write frames...
"

# Test CLI
python3 video_to_prompt.py /tmp/test_colors.mp4 -n 6 -m tag

# Test model discovery
curl -s http://192.168.3.177:8080/v1/models | python3 -m json.tool
```

## File Structure

```
video-to-prompt/
├── video_to_prompt_webui.py   # Gradio WebUI
├── video_to_prompt.py         # CLI tool
├── AGENTS.md                  # This file (AI agent instructions)
├── requirements.txt           # gradio, pillow, numpy, requests
├── install.sh                 # One-click installer
├── README.md                  # Human documentation
└── LICENSE                    # MIT
```

## Key Functions (for programmatic use)

```python
from video_to_prompt_webui import (
    extract_frames,      # (path, count, max_size) → (frames, timestamps)
    create_frame_strip,  # (frames, timestamps, cols, thumb_size) → PIL.Image
    fetch_models,        # (api_url) → [(display, model_id), ...]
    call_vision_api,     # (frames, prompt, model, ...) → dict
    get_video_info,      # (path) → dict with width, height, duration, fps
    clean_output,        # (raw_text) → cleaned_text
    PROMPT_MODES,        # dict of mode configs
    EXAMPLE_PROMPTS,     # dict of example prompts
)
```

## REST API (for AI Agents)

Run the API server separately or alongside the WebUI:

```bash
python3 api.py                        # http://localhost:8000
python3 api.py --port 9000 --reload   # Dev mode with auto-reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check, version |
| `GET` | `/api/models?url=...` | List models from llama.cpp API |
| `GET` | `/api/modes` | List prompt modes + examples |
| `GET` | `/api/video/info?path=...` | Get video metadata |
| `POST` | `/api/analyze` | Analyze video (multipart upload) |
| `POST` | `/api/analyze/json` | Analyze video (JSON body with path) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### Example: Analyze video via REST API

```bash
# Multipart upload
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@video.mp4" \
  -F "mode=describe" \
  -F "frame_count=16" \
  -F "api_url=http://192.168.3.177:8080/v1/chat/completions"

# JSON (local path)
curl -X POST http://localhost:8000/api/analyze/json \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/video.mp4", "mode": "tag", "frame_count": 8}'
```

### Response format

```json
{
  "success": true,
  "video_info": {"width": 1920, "height": 1080, "duration": 30.0, "fps": 30.0},
  "frames_extracted": 16,
  "timestamps": [0.0, 2.0, 4.0, ...],
  "mode": "🎨 Describe (image generation prompt)",
  "model": "model-name.gguf",
  "prompt": "A detailed visual description...",
  "tokens_used": {"prompt_tokens": 3207, "completion_tokens": 512},
  "elapsed_seconds": 22.5
}
```

### Python SDK (simplest)

```python
import requests

# Health check
requests.get("http://localhost:8000/health").json()

# List models
requests.get("http://localhost:8000/api/models?url=http://192.168.3.177:8080/v1/chat/completions").json()

# Analyze video
with open("video.mp4", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/analyze",
        files={"file": f},
        data={"mode": "describe", "frame_count": 16}
    )
print(resp.json()["prompt"])
```

## Notes for AI Agents

1. **Never expose API tokens or secrets** — the tool reads configuration from environment or UI input only
2. **ffmpeg is required** — check with `which ffmpeg` before running
3. **Model must be vision-capable** — verify with `fetch_models()` that models have 👁️ icon
4. **Stale .pyc cache** — if changes don't take effect, run `find . -name "__pycache__" -exec rm -rf {} +`
5. **Video formats** — any format ffmpeg can read (mp4, avi, mov, mkv, webm, etc.)
6. **Large videos** — reduce `max_size` or `frame_count` if API times out
7. **Thinking models** — some models output reasoning inline; the tool auto-cleans this

---

*Created by [UKA](https://github.com/nanofatdog) — AI agent, hacker & security expert*
