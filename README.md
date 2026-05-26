# 🎬 Video-to-Prompt WebUI

> Extract AI image generation prompts from video using Vision Language Model (llama.cpp API)

A Gradio web application that extracts frames from video, sends them to a multimodal vision LLM, and generates detailed descriptions, tags, or summaries — perfect for prompt engineering, content analysis, and AI workflow automation.

![screenshot](https://img.shields.io/badge/version-1.1.0-orange) ![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.10%2B-green)

---

## ✨ Features

- 🎥 **Upload any video** — MP4, AVI, MOV, MKV, WebM, and more
- 🧠 **5 prompt modes** — Describe, Summarize, Tags, Danbooru, NSFW Check
- 💡 **5 example prompts** — Shot List, Art Analysis, Color Palette, Object Detection, Social Caption
- 🌡️ **Temperature presets** — Creative (0.9) / Balanced (0.6) / Precise (0.3) one-click
- 📦 **Download frames** — Export extracted frames as ZIP
- 📜 **Output history** — Last 5 results stored for easy recall
- 🔄 **Model discovery** — Auto-detect available vision models from API
- 👁️ **Frame timestamps** — Each preview thumbnail shows exact second
- ⚙️ **Configurable API** — Point to any llama.cpp vision endpoint
- 🖼️ **Frame preview** — See extracted frames before processing
- 🔑 **API key support** — Optional Bearer token auth
- 📊 **Real-time progress** — Track every step from extraction to generation
- 💾 **Save & Copy** — Export results as .txt or copy to clipboard
- 🎨 **Beautiful UI** — Clean, dark-themed Gradio interface
- 📦 **One-click install** — `curl | bash` installer
- 🤖 **REST API** — FastAPI server for AI agent integration
- 🔧 **Systemd service** — Optional auto-start on boot

---

## 🚀 Quick Start

### One-Click Install

```bash
curl -fsSL https://raw.githubusercontent.com/nanofatdog/video-to-prompt/main/install.sh | bash
```

### Manual Install

```bash
# Clone
git clone https://github.com/nanofatdog/video-to-prompt.git
cd video-to-prompt

# Setup venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch
python3 video_to_prompt_webui.py
```

Then open **http://localhost:7860** in your browser.

---

## 📋 Requirements

| Dependency | Install |
|-----------|---------|
| Python 3.10+ | `apt install python3` |
| FFmpeg | `apt install ffmpeg` |
| Git | `apt install git` |

```bash
# Ubuntu/Debian
sudo apt install python3 python3-pip python3-venv ffmpeg git

# macOS
brew install python3 ffmpeg git

# CentOS/RHEL
sudo yum install python3 python3-pip ffmpeg git
```

---

## 🎯 Usage

### WebUI

```bash
# Default (http://0.0.0.0:7860)
./launch.sh

# Custom port
./launch.sh --port 8080

# Public share link
./launch.sh --share
```

### CLI (included)

```bash
python3 video_to_prompt.py video.mp4
python3 video_to_prompt.py video.mp4 -n 32 -m tag
python3 video_to_prompt.py video.mp4 --custom "Is this SFW?" --json
```

### Systemd Service

```bash
./install-service.sh
systemctl status video-to-prompt
journalctl -u video-to-prompt -f
```

---

## 🧩 API Configuration

The WebUI connects to a llama.cpp vision API. Configure the endpoint in the UI:

| Field | Default | Description |
|-------|---------|-------------|
| `API Endpoint` | `http://192.168.3.177:8080/v1/chat/completions` | Your llama.cpp server |
| `Model Name` | `llmfan46_Qwen3.6-35B-A3B-uncensored-heretic-Q6_K.gguf` | Vision model name |

### Setting up a llama.cpp vision server

```bash
# Start llama.cpp server with vision support
./llama-server \
  -m /path/to/qwen-vl-model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --host 0.0.0.0 \
  --port 8080
```

Models with vision support:
- Qwen3-VL-4B-Instruct-GGUF
- Qwen3-VL-8B-Instruct-GGUF
- Qwen2.5-VL-3B-Instruct-GGUF
- Qwen2.5-VL-7B-Instruct-GGUF

---

## 🎨 Prompt Modes

| Mode | Output | Use Case |
|------|--------|----------|
| 🎨 **Describe** | Detailed visual description | Stable Diffusion / Midjourney prompts |
| 📝 **Summarize** | Brief video summary | Content cataloging |
| 🏷️ **Tags** | Comma-separated keywords | Search indexing |
| 🎌 **Danbooru** | Danbooru-style tags | Anime/art datasets |
| 🔞 **NSFW Check** | Rating 1-10 (S/V/N) | Content filtering |

### Custom Prompts

Override any mode by entering your own prompt:

```
Describe the clothing style of the person in this video. 
List brands if recognizable.
```

---

## 📁 Project Structure

```
video-to-prompt/
├── video_to_prompt_webui.py   # Gradio WebUI (main app)
├── video_to_prompt.py         # CLI tool
├── requirements.txt           # Python dependencies
├── install.sh                 # One-click installer
├── install-service.sh         # Systemd service installer
├── launch.sh                  # Convenience launcher
└── README.md                  # This file
```

---

## 🔧 Advanced Settings

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| Frame count | 16 | 2–64 | Frames to sample from video |
| Max frame size | 1280px | 256–2048 | Resize before sending |
| Max tokens | 1024 | 128–4096 | Output length |
| Temperature | 0.6 | 0.1–1.5 | Creativity level |
| JPEG quality | 80 | 30–100 | Image compression |

---

## 🤖 API Format

The tool sends frames as OpenAI-compatible vision messages:

```json
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
  "max_tokens": 1024
}
```

---

## 🤖 REST API (for AI Agents)

Run a FastAPI server that AI agents can call programmatically:

```bash
python3 api.py                    # http://localhost:8000
python3 api.py --port 9000        # Custom port
```

Interactive docs: **http://localhost:8000/docs** (Swagger) | **http://localhost:8000/redoc** (ReDoc)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + version |
| `GET` | `/api/models?url=...` | List available models |
| `GET` | `/api/modes` | List prompt modes + examples |
| `GET` | `/api/video/info?path=...` | Video metadata |
| `POST` | `/api/analyze` | Analyze video (multipart upload) |
| `POST` | `/api/analyze/json` | Analyze video (JSON body) |

### Quick Examples

```bash
# Health check
curl http://localhost:8000/health

# List models
curl "http://localhost:8000/api/models?url=http://192.168.3.177:8080/v1/chat/completions"

# Analyze video
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@video.mp4" \
  -F "mode=describe" \
  -F "frame_count=16"
```

### Python SDK

```python
import requests

# List models
models = requests.get(
    "http://localhost:8000/api/models",
    params={"url": "http://192.168.xxx.xxx:8080/v1/chat/completions"}
).json()

# Analyze video
with open("video.mp4", "rb") as f:
    result = requests.post(
        "http://localhost:8000/api/analyze",
        files={"file": f},
        data={"mode": "describe", "frame_count": 16}
    ).json()

print(result["prompt"])
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Credits

- [Gradio](https://gradio.app) — WebUI framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inference engine
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — Vision language models
- [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) — Inspiration

---

**🤖 Created with ❤️ by [UKA](https://github.com/nanofatdog) — 18-year-old hacker & AI Security expert**

**Made with ❤️ by nanofatdog**
