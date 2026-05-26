#!/usr/bin/env python3
"""
🎬 Video-to-Prompt WebUI
Gradio interface for extracting AI prompts from video using llama.cpp vision API.

Features:
- Upload video via drag & drop
- Configurable API endpoint + model
- 5 prompt modes (describe, summarize, tag, booru, nsfw_check)
- Custom prompt support
- Frame sampling control
- Real-time progress bar
- Live preview of extracted frames
- Copy to clipboard
- Save to file
"""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

import gradio as gr
import numpy as np
import requests
from PIL import Image

# ── Constants ───────────────────────────────────────────────────────
VERSION = "1.0.0"
DEFAULT_API_URL = "http://192.168.3.177:8080/v1/chat/completions"
DEFAULT_MODEL = "llmfan46_Qwen3.6-35B-A3B-uncensored-heretic-Q6_K.gguf"
DEFAULT_FRAME_COUNT = 16
DEFAULT_MAX_SIZE = 1280
DEFAULT_MAX_TOKENS = 1024

PROMPT_MODES = {
    "🎨 Describe (image generation prompt)": {
        "key": "describe",
        "prompt": (
            "You are analyzing frames extracted from a video. These {n} frames are uniformly "
            "sampled from the entire video timeline.\n\n"
            "Write a DETAILED visual description suitable for use as an AI image generation prompt. "
            "Include: subject, setting, lighting, color palette, mood, composition, style, camera angle, "
            "and any notable details visible across the frames. If the video shows action/motion, "
            "describe the sequence and dynamics.\n\n"
            "Output format: a single coherent paragraph in English, ready to copy into Stable Diffusion "
            "or Midjourney. Be specific and vivid — avoid vague terms. Use photography/cinematography "
            "terminology where appropriate."
        ),
    },
    "📝 Summarize (brief description)": {
        "key": "summarize",
        "prompt": (
            "These {n} frames are sampled from a video. Describe what happens in the video based on "
            "these frames. What is the subject doing? Where are they? What is the overall "
            "action/event taking place? Be concise but complete."
        ),
    },
    "🏷️ Tags (comma-separated keywords)": {
        "key": "tag",
        "prompt": (
            "Analyze these {n} video frames. Extract tags/keywords describing the content.\n"
            "Output ONLY comma-separated tags in English, no other text. "
            "Include: subjects, objects, actions, setting, style, lighting, mood, colors, "
            "camera angle, time of day, quality descriptors.\n"
            "Example: portrait, woman, outdoor, sunset, golden hour, bokeh, soft lighting, "
            "photorealistic, 8k, shallow depth of field"
        ),
    },
    "🎌 Danbooru Tags": {
        "key": "booru",
        "prompt": (
            "You are a Danbooru tagger. Analyze these {n} video frames and output ONLY "
            "space-separated Danbooru-style tags. Include: character count, gender, hair color, "
            "eye color, clothing, pose, expression, background, art style, quality tags.\n"
            "Format: tag1 tag2 tag3 ... (no commas, no other text)"
        ),
    },
    "🔞 NSFW Check (rating 1-10)": {
        "key": "nsfw_check",
        "prompt": (
            "Analyze these {n} video frames. Rate the content on these scales (1-10):\n"
            "- Suggestive: (1=fully clothed, 10=explicit)\n"
            "- Violence: (1=none, 10=extreme)\n"
            "- Overall NSFW: (1=completely safe, 10=explicit adult content)\n"
            "Output only the three numbers like: 3,1,2"
        ),
    },
}

# ── Video Utils ─────────────────────────────────────────────────────
def get_video_info(path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        video_stream = None
        for stream in data.get("streams", []):
            if stream["codec_type"] == "video":
                video_stream = stream
                break
        if not video_stream:
            return {}

        duration = float(data["format"].get("duration", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": duration,
            "fps": fps,
            "codec": video_stream.get("codec_name", "unknown"),
            "size_mb": os.path.getsize(path) / (1024 * 1024),
        }
    except Exception:
        return {}


def extract_frames(video_path: str, frame_count: int = 16, max_size: int = 1280) -> list[Image.Image]:
    """Extract uniformly sampled frames using ffmpeg."""
    video_info = get_video_info(video_path)
    duration = video_info.get("duration", 0)

    if duration > 0:
        target_fps = frame_count / duration
        vf_filter = f"scale='min({max_size},iw)':-1:force_original_aspect_ratio=decrease,fps={target_fps:.6f}"
    else:
        vf_filter = f"scale='min({max_size},iw)':-1:force_original_aspect_ratio=decrease"

    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", video_path,
        "-vf", vf_filter,
        "-vframes", str(frame_count),
        "-f", "image2pipe", "-vcodec", "png", "-"
    ]

    result = subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    frames = _split_png_stream(result.stdout)
    return frames


def _split_png_stream(data: bytes) -> list[Image.Image]:
    """Split concatenated PNG byte stream into individual PIL Images."""
    frames = []
    pos = 0
    png_header = b"\x89PNG\r\n\x1a\n"
    while pos < len(data):
        start = data.find(png_header, pos)
        if start == -1:
            break
        next_start = data.find(png_header, start + 1)
        if next_start == -1:
            chunk = data[start:]
            pos = len(data)
        else:
            chunk = data[start:next_start]
            pos = next_start
        try:
            img = Image.open(io.BytesIO(chunk))
            img.load()
            frames.append(img)
        except Exception:
            continue
    return frames


def frame_to_base64(img: Image.Image, quality: int = 80) -> str:
    """Convert PIL Image to base64 JPEG."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def create_frame_strip(frames: list[Image.Image], cols: int = 8, thumb_size: int = 200) -> Image.Image:
    """Create a contact sheet / frame strip from extracted frames."""
    if not frames:
        return Image.new("RGB", (thumb_size, thumb_size), color=(30, 30, 30))

    rows = (len(frames) + cols - 1) // cols
    thumbs = []
    for f in frames:
        thumb = f.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        thumbs.append(thumb)

    strip = Image.new("RGB", (cols * thumb_size, rows * thumb_size), color=(20, 20, 20))
    for i, thumb in enumerate(thumbs):
        r, c = i // cols, i % cols
        x = c * thumb_size + (thumb_size - thumb.width) // 2
        y = r * thumb_size + (thumb_size - thumb.height) // 2
        strip.paste(thumb, (x, y))
    return strip


# ── API ─────────────────────────────────────────────────────────────
REASONING_MARKERS = [
    "The user wants", "Let me", "I need to", "First,", "Step 1",
    "Analyze the", "Looking at", "The prompt asks", "I'll", "I should",
    "Wait,", "Hmm,", "I see", "This is a", "Okay,", "Let's",
    "The image", "Frame 1", "Frame 2", "Self-Correction",
    "Draft", "Review against", "Final check", "Final Polish",
    "Refining", "Synthesize", "So the", "To answer",
    "Based on", "From the", "I can see", "Okay so",
]


def clean_output(raw: str) -> str:
    """Clean reasoning artifacts from thinking model output."""
    # Pattern 1: " response" marker (Qwen thinking format)
    if " response" in raw:
        raw = raw.split(" response")[-1].strip()

    # Pattern 2: Remove <｜end▁of▁thinking｜> at start
    if raw.startswith("response"):
        raw = raw[len("response"):].strip()

    # Pattern 3: Remove lines that look like reasoning
    lines = raw.strip().split("\n")
    cleaned = []
    in_reasoning = True
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            cleaned.insert(0, line)
            continue
        is_reasoning = any(stripped.startswith(m) for m in REASONING_MARKERS)
        if not is_reasoning and in_reasoning:
            in_reasoning = False
        if not in_reasoning:
            cleaned.insert(0, line)

    result = "\n".join(cleaned).strip()

    # If we accidentally stripped everything, return original
    if len(result) < 10 and len(raw) > 50:
        return raw.strip()

    return result


# ── Main Processing ─────────────────────────────────────────────────
def process_video(
    video_file,
    api_url: str,
    model_name: str,
    mode: str,
    custom_prompt: str,
    frame_count: int,
    max_size: int,
    max_tokens: int,
    temperature: float,
    quality: int,
    progress=gr.Progress(),
):
    """Main pipeline: extract frames → call API → return results."""
    if video_file is None:
        yield None, "❌ Please upload a video file", "", "", ""
        return

    video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
    start_time = time.time()

    # ── Step 1: Video Info ──
    progress(0.05, desc="🔍 Reading video metadata...")
    info = get_video_info(video_path)
    if not info:
        yield None, "❌ Cannot read video file. Check format.", "", "", ""
        return

    info_text = (
        f"📐 **Resolution:** {info['width']}×{info['height']} | "
        f"⏱ **Duration:** {info['duration']:.1f}s | "
        f"🎞️ **FPS:** {info['fps']:.1f} | "
        f"📦 **Size:** {info['size_mb']:.1f}MB | "
        f"🎬 **Codec:** {info['codec']}"
    )

    # ── Step 2: Extract Frames ──
    progress(0.15, desc="🎞️ Extracting frames...")
    yield None, info_text + "\n\n🔄 Extracting frames...", "", "", ""

    try:
        frames = extract_frames(video_path, frame_count, max_size)
    except subprocess.CalledProcessError as e:
        yield None, f"❌ ffmpeg error: {e.stderr.decode()[:500] if e.stderr else str(e)}", "", "", ""
        return
    except Exception as e:
        yield None, f"❌ Frame extraction failed: {e}", "", "", ""
        return

    if not frames:
        yield None, "❌ No frames extracted. Try a different video.", "", "", ""
        return

    # ── Step 3: Preview ──
    progress(0.30, desc="🖼️ Generating preview...")
    strip = create_frame_strip(frames)
    frame_info = f"📸 Extracted **{len(frames)}** frames (requested {frame_count})"
    yield strip, info_text + "\n\n" + frame_info, "", "", ""

    # ── Step 4: Build Prompt ──
    progress(0.35, desc="📝 Building prompt...")
    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()
        display_mode = "Custom"
    else:
        for mode_name, mode_data in PROMPT_MODES.items():
            if mode_data["key"] == mode:
                prompt = mode_data["prompt"].format(n=len(frames))
                display_mode = mode_name
                break
        else:
            prompt = PROMPT_MODES[list(PROMPT_MODES.keys())[0]]["prompt"].format(n=len(frames))
            display_mode = "Describe"

    yield strip, (
        f"{info_text}\n\n{frame_info}"
        f"\n\n📤 **Sending to API** (`{model_name[:30]}...`)\n"
        f"🎯 **Mode:** {display_mode}\n"
        f"🖼️ **Frames:** {len(frames)}"
    ), "", "", ""

    # ── Step 5: Call API ──
    progress(0.40, desc="🚀 Calling vision API...")
    content = [{"type": "text", "text": prompt}]
    for i, frame in enumerate(frames):
        progress(0.40 + 0.01 * (i / len(frames)), desc=f"🖼️ Encoding frame {i+1}/{len(frames)}...")
        b64 = frame_to_base64(frame, quality)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }

    api_start = time.time()
    try:
        resp = requests.post(api_url.rstrip("/") + "/v1/chat/completions" if not api_url.endswith("/v1/chat/completions") else api_url,
                           json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        yield strip, f"❌ Cannot connect to API: `{api_url}`\n\nIs the server running?", "", "", ""
        return
    except requests.exceptions.Timeout:
        yield strip, "❌ API request timed out (300s). Try fewer frames or smaller max_size.", "", "", ""
        return
    except Exception as e:
        yield strip, f"❌ API error: {e}", "", "", ""
        return

    api_elapsed = time.time() - api_start
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")

    # ── Step 6: Extract & Clean ──
    progress(0.90, desc="🧹 Cleaning output...")
    message = data.get("choices", [{}])[0].get("message", {})
    raw_output = message.get("content", "") or message.get("reasoning_content", "")
    cleaned = clean_output(raw_output)

    # ── Step 7: Build Stats ──
    total_elapsed = time.time() - start_time
    stats = (
        f"{info_text}\n\n{frame_info}"
        f"\n\n⚡ **API:** {api_elapsed:.1f}s | 🎯 **Tokens:** {prompt_tokens}/{completion_tokens}"
        f" | ⏱ **Total:** {total_elapsed:.1f}s"
        f"\n🎯 **Mode:** {display_mode}"
    )

    # ── Step 8: Generate filenames ──
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{video_name}_{mode}_{timestamp}.txt"

    yield strip, stats, cleaned, cleaned, default_filename


# ── UI ──────────────────────────────────────────────────────────────
def build_ui():
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    css = """
    .container { max-width: 1200px; margin: 0 auto; }
    .output-text textarea { font-size: 1rem !important; line-height: 1.6 !important; }
    .stats-box { font-size: 0.95rem; }
    .title-gradient { 
        background: linear-gradient(135deg, #ff6b35, #f7c948); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; 
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        theme=theme,
        css=css,
        title="Video-to-Prompt | AI Video Analyzer",
        analytics_enabled=False,
    ) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1 class="title-gradient" style="font-size: 2.5rem; margin: 0;">🎬 Video → Prompt</h1>
            <p style="color: #888; font-size: 1rem; margin: 0;">
                Extract AI prompts from video using Vision Language Model via llama.cpp API
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # ── Video Upload ──
                gr.Markdown("### 📤 Upload Video")
                video_input = gr.Video(
                    label="Drop video here",
                    sources=["upload"],
                    height=280,
                )

                # ── API Settings ──
                gr.Markdown("### ⚙️ API Settings")
                api_url = gr.Textbox(
                    label="API Endpoint",
                    value=DEFAULT_API_URL,
                    placeholder="http://192.168.3.177:8080/v1/chat/completions",
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    value=DEFAULT_MODEL,
                )

                # ── Mode ──
                gr.Markdown("### 🎯 Prompt Mode")
                mode = gr.Dropdown(
                    choices=list(PROMPT_MODES.keys()),
                    value=list(PROMPT_MODES.keys())[0],
                    label="Select mode",
                )
                custom_prompt = gr.Textbox(
                    label="Custom prompt (overrides mode)",
                    placeholder="Write your own prompt here...",
                    lines=3,
                )

                # ── Advanced Settings ──
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    frame_count = gr.Slider(
                        minimum=2, maximum=64, value=DEFAULT_FRAME_COUNT, step=2,
                        label="Frame count",
                        info="More frames = more context but slower",
                    )
                    max_size = gr.Slider(
                        minimum=256, maximum=2048, value=DEFAULT_MAX_SIZE, step=128,
                        label="Max frame size (px)",
                        info="Resize frames before sending",
                    )
                    max_tokens = gr.Slider(
                        minimum=128, maximum=4096, value=DEFAULT_MAX_TOKENS, step=128,
                        label="Max output tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.6, step=0.05,
                        label="Temperature",
                    )
                    quality = gr.Slider(
                        minimum=30, maximum=100, value=80, step=5,
                        label="JPEG quality",
                        info="Lower = smaller payload, faster API calls",
                    )

                process_btn = gr.Button(
                    "🚀 Generate Prompt",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                # ── Frame Preview ──
                gr.Markdown("### 🖼️ Frame Preview")
                frame_preview = gr.Image(
                    label="Extracted frames",
                    type="pil",
                    height=280,
                )

                # ── Status ──
                gr.Markdown("### 📊 Status")
                status_output = gr.Markdown(
                    value="👆 Upload a video and click **Generate Prompt** to start",
                    elem_classes=["stats-box"],
                )

                # ── Output ──
                gr.Markdown("### ✨ Generated Prompt")
                output_text = gr.Textbox(
                    label="",
                    lines=12,
                    max_lines=20,
                    elem_classes=["output-text"],
                    placeholder="Your generated prompt will appear here...",
                )

                with gr.Row():
                    output_filename = gr.Textbox(
                        label="Filename",
                        value="prompt.txt",
                        scale=3,
                    )
                    save_btn = gr.Button("💾 Save", scale=1, variant="secondary")
                    copy_btn = gr.Button("📋 Copy", scale=1, variant="secondary")

                save_status = gr.Markdown("")

        # ── Event Bindings ──
        def on_save(text, filename):
            if not text or not text.strip():
                return "❌ Nothing to save"
            path = Path(filename)
            path.write_text(text, encoding="utf-8")
            return f"✅ Saved to: `{path.resolve()}`"

        def on_copy(text):
            return text  # gr.Textbox handles clipboard

        process_btn.click(
            fn=process_video,
            inputs=[
                video_input, api_url, model_name, mode, custom_prompt,
                frame_count, max_size, max_tokens, temperature, quality,
            ],
            outputs=[frame_preview, status_output, output_text, output_text, output_filename],
        )

        save_btn.click(
            fn=on_save,
            inputs=[output_text, output_filename],
            outputs=[save_status],
        )

        copy_btn.click(
            fn=on_copy,
            inputs=[output_text],
            outputs=[output_text],
        )

        # ── Footer ──
        gr.HTML(f"""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.8rem;">
            Video-to-Prompt v{VERSION} | Powered by llama.cpp Vision API | 
            <a href="https://github.com/nanofatdog/video-to-prompt" target="_blank">GitHub</a>
        </div>
        """)

    return demo


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Video-to-Prompt WebUI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


# Allow import without running argparse
if __name__ == "__main__":
    import argparse
    main()
