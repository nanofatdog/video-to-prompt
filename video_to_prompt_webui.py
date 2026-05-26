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
- Temperature presets (Creative / Balanced / Precise)
- Download extracted frames as ZIP
- Output history (last 5 results)
- Copy to clipboard
- Save to file
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

import gradio as gr
import numpy as np
import requests
from PIL import Image

# ── Constants ───────────────────────────────────────────────────────
VERSION = "1.1.0"
DEFAULT_API_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_MODEL = "llmfan46_Qwen3.6-35B-A3B-uncensored-heretic-Q6_K.gguf"
DEFAULT_FRAME_COUNT = 16
DEFAULT_MAX_SIZE = 1280
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_TOKENS_LIMIT = 8192  # Max slider value for llama.cpp
MAX_HISTORY = 5

PROMPT_MODES = {
    "🎨 Describe (image generation prompt)": {
        "key": "describe",
        "prompt": (
            "Write ONE detailed paragraph (8-12 sentences, ~150-250 words) describing these {n} video frames "
            "as an AI image generation prompt.\n\n"
            "Include: subject(s) and action, setting, lighting (source, direction, color temperature, shadows), "
            "color palette, mood, composition, style, camera angle/lens feel, and any notable details visible across frames.\n"
            "If the video shows motion, describe the sequence.\n\n"
            "Stay grounded in visible details only. Use vivid photography/cinematography terminology. "
            "No abstract interpretations.\n\n"
            "No preface, no reasoning, no <think>. Output ONLY the prompt paragraph."
        ),
    },
    "📝 Summarize (brief description)": {
        "key": "summarize",
        "prompt": (
            "These {n} frames are sampled from a video. Write ONE concise paragraph (4-6 sentences, ~80-120 words) "
            "summarizing what happens: subject, action, setting, overall event.\n\n"
            "Stay grounded in visible details. No speculation about what happens between frames.\n\n"
            "No preface, no reasoning, no <think>. Output ONLY the summary paragraph."
        ),
    },
    "🏷️ Tags (comma-separated keywords)": {
        "key": "tag",
        "prompt": (
            "Analyze these {n} video frames. Generate a clean list of comma-separated tags for AI image generation, "
            "based ONLY on visual information. Limit to max 50 unique tags.\n\n"
            "Include: subjects, objects, actions, setting, style, lighting, mood, colors, "
            "camera angle, time of day, quality descriptors. "
            "Strictly describe visual elements. No abstract concepts, marketing terms, or technical jargon.\n\n"
            "No preface, no reasoning, no <think>. Output ONLY the comma-separated tags.\n"
            "Example: portrait, woman, outdoor, sunset, golden hour, bokeh, soft lighting, photorealistic, 8k"
        ),
    },
    "🎌 Danbooru Tags": {
        "key": "booru",
        "prompt": (
            "You are a Danbooru tagger. Analyze these {n} video frames. "
            "Output ONLY space-separated Danbooru-style tags based on visible content.\n\n"
            "Include: character count, gender, hair color, eye color, clothing, pose, expression, "
            "background, art style, quality tags.\n"
            "Format: tag1 tag2 tag3 ... (space-separated, no commas, no other text)\n\n"
            "No preface, no reasoning, no <think>. Output ONLY the tags."
        ),
    },
    "🔞 NSFW Check (rating 1-10)": {
        "key": "nsfw_check",
        "prompt": (
            "Analyze these {n} video frames. Rate the content on these scales (1-10) "
            "based ONLY on visible visual content:\n"
            "- Suggestive: (1=fully clothed, 10=explicit)\n"
            "- Violence: (1=none, 10=extreme)\n"
            "- Overall NSFW: (1=completely safe, 10=explicit adult content)\n\n"
            "No preface, no reasoning, no <think>. Output only the three numbers like: 3,1,2"
        ),
    },
}

# ── Custom Prompt Examples ──────────────────────────────────────────
EXAMPLE_PROMPTS = {
    "🎬 Movie Director Shot List": (
        "You are a professional film director. Analyze these {n} video frames.\n\n"
        "Create a detailed shot list. For each visible scene/shot include:\n"
        "- Shot type (close-up, medium, wide, POV, aerial)\n"
        "- Camera movement (static, pan, tilt, dolly, handheld)\n"
        "- Lighting setup (key light direction, fill, rim, natural/artificial)\n"
        "- Lens choice (wide angle, telephoto, anamorphic)\n"
        "- Composition notes (rule of thirds, leading lines, symmetry)\n"
        "- Color grading mood (warm, cool, desaturated, high contrast)\n\n"
        "Stay grounded in visible details. Be technical and precise.\n\n"
        "No preface, no reasoning, no <think>. Output ONLY the shot list."
    ),
    "📸 Photo/Art Style Analysis": (
        "Analyze these {n} video frames as an art critic and professional photographer. "
        "Write ONE paragraph (8-12 sentences, ~150-200 words).\n\n"
        "Describe:\n"
        "- Photography/art style (cinematic, editorial, snapshot, fine art, vintage)\n"
        "- Estimated camera settings (aperture feel, shutter speed, ISO)\n"
        "- Depth of field and bokeh characteristics\n"
        "- Texture and surface details\n"
        "- Post-processing effects (film grain, vignette, color grading)\n"
        "- Comparable photographers or art movements\n\n"
        "Stay grounded in visible details. Elegant gallery-exhibition tone.\n\n"
        "No preface, no reasoning, no <think>. Output ONLY the analysis paragraph."
    ),
    "🎨 Dominant Color Palette": (
        "Analyze the color scheme across these {n} video frames. "
        "Output EXACTLY in this format:\n"
        "1. Primary palette: list 5-7 hex color codes (#RRGGBB) with names\n"
        "2. Color harmony type: (monochromatic / complementary / analogous / triadic / split-complementary)\n"
        "3. Saturation level: (muted / natural / vibrant / hyper-saturated)\n"
        "4. Brightness/Value: (low-key / balanced / high-key)\n"
        "5. Temperature: (cool / neutral / warm / mixed)\n"
        "6. One-line description of the overall color mood\n\n"
        "Stay grounded in visible colors. Estimate hex codes from what you see.\n\n"
        "No preface, no reasoning, no <think>. Output ONLY the format above."
    ),
    "🔍 Object & Brand Detection": (
        "You are a forensic image analyst. Examine these {n} video frames. "
        "List ONLY what is visually confirmed:\n\n"
        "1. All visible OBJECTS with counts (e.g., '3 chairs, 1 red coffee mug')\n"
        "2. Any recognizable BRANDS or logos (clothing, electronics, food, vehicles)\n"
        "3. TECHNOLOGY devices (phones, laptops, monitors — estimate models if possible)\n"
        "4. VEHICLES (car make/model, license plates if visible)\n"
        "5. CLOTHING items with estimated colors and styles\n"
        "6. TEXT/words visible anywhere in the frames\n"
        "7. LOCATION clues (indoor/outdoor, city/rural, any landmarks)\n\n"
        "Say 'none detected' if a category is empty. Stay grounded in visible evidence.\n\n"
        "No preface, no reasoning, no <think>. Output ONLY the checklist."
    ),
    "📝 Social Media Caption": (
        "Create 3 social media caption options for this video content, based on {n} frames. "
        "Stay grounded in visible content.\n\n"
        "Each caption: catchy hook + 2-3 relevant hashtags + target platform noted.\n\n"
        "Caption 1 — Professional/LinkedIn style\n"
        "Caption 2 — Casual/Instagram style with emoji\n"
        "Caption 3 — Short/TikTok viral style\n\n"
        "Make them authentic and platform-appropriate.\n\n"
        "No preface, no reasoning, no <think>. Output ONLY the 3 captions."
    ),
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


def extract_frames(video_path: str, frame_count: int = 16, max_size: int = 1280) -> tuple[list[Image.Image], list[float]]:
    """Extract uniformly sampled frames using ffmpeg. Returns (frames, timestamps)."""
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
    
    # Calculate timestamps for each frame
    if duration > 0 and frames:
        timestamps = [duration * i / len(frames) for i in range(len(frames))]
    else:
        timestamps = [float(i) for i in range(len(frames))]
    
    return frames, timestamps


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


def create_frame_strip(
    frames: list[Image.Image],
    timestamps: list[float] | None = None,
    cols: int = 8,
    thumb_size: int = 200,
) -> Image.Image:
    """Create a contact sheet / frame strip from extracted frames with timestamps."""
    if not frames:
        return Image.new("RGB", (thumb_size, thumb_size), color=(30, 30, 30))

    rows = (len(frames) + cols - 1) // cols
    label_height = 24  # space for timestamp label
    
    thumbs = []
    for f in frames:
        thumb = f.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        thumbs.append(thumb)

    total_h = rows * (thumb_size + label_height)
    strip = Image.new("RGB", (cols * thumb_size, total_h), color=(20, 20, 20))
    
    try:
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
    except ImportError:
        font = None
    
    for i, thumb in enumerate(thumbs):
        r, c = i // cols, i % cols
        x = c * thumb_size + (thumb_size - thumb.width) // 2
        y = r * (thumb_size + label_height) + (thumb_size - thumb.height) // 2
        strip.paste(thumb, (x, y))
        
        if timestamps and i < len(timestamps) and font:
            ts = timestamps[i]
            label_x = c * thumb_size + 4
            label_y = r * (thumb_size + label_height) + thumb_size + 3
            
            if ts >= 3600:
                label = f"{int(ts//3600)}:{int((ts%3600)//60):02d}:{ts%60:04.1f}"
            else:
                label = f"{int(ts//60):02d}:{ts%60:04.1f}"
            
            draw = ImageDraw.Draw(strip)
            draw.rectangle(
                [c * thumb_size, label_y - 3, (c + 1) * thumb_size, label_y + label_height],
                fill=(0, 0, 0),
            )
            draw.text((label_x, label_y), label, fill=(255, 200, 50), font=font)
    
    return strip


def save_frames_to_dir(frames: list[Image.Image], timestamps: list[float], quality: int = 85) -> str:
    """Save extracted frames as JPEG files to a temp directory. Returns the dir path."""
    temp_dir = tempfile.mkdtemp(prefix="video_frames_")
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        ts_str = f"{int(ts//1):04d}"
        filepath = os.path.join(temp_dir, f"frame_{i:03d}_{ts_str}s.jpg")
        frame.convert("RGB").save(filepath, format="JPEG", quality=quality, optimize=True)
    return temp_dir


def make_frames_zip(temp_dir: str) -> str:
    """Create a ZIP file from a directory of frames. Returns path to ZIP."""
    if not temp_dir or not os.path.isdir(temp_dir):
        return None
    zip_path = temp_dir + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(temp_dir)):
            fpath = os.path.join(temp_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    return zip_path


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
    if "<｜end▁of▁thinking｜>" in raw:
        raw = raw.split(" response")[-1].strip()

    if raw.startswith("response"):
        raw = raw[len("response"):].strip()

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

    if len(result) < 10 and len(raw) > 50:
        return raw.strip()

    return result


# ── Main Processing ─────────────────────────────────────────────────
def process_video(
    video_file,
    api_url: str,
    api_key: str,
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
        yield None, "❌ Please upload a video file", "", "", "", None
        return

    video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
    start_time = time.time()
    frames_temp_dir = None

    # ── Step 1: Video Info ──
    progress(0.05, desc="🔍 Reading video metadata...")
    info = get_video_info(video_path)
    if not info:
        yield None, "❌ Cannot read video file. Check format.", "", "", "", None
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
    yield None, info_text + "\n\n🔄 Extracting frames...", "", "", "", None

    try:
        frames, timestamps = extract_frames(video_path, frame_count, max_size)
    except subprocess.CalledProcessError as e:
        yield None, f"❌ ffmpeg error: {e.stderr.decode()[:500] if e.stderr else str(e)}", "", "", "", None
        return
    except Exception as e:
        yield None, f"❌ Frame extraction failed: {e}", "", "", "", None
        return

    if not frames:
        yield None, "❌ No frames extracted. Try a different video.", "", "", "", None
        return

    # Save frames to temp dir for download
    frames_temp_dir = save_frames_to_dir(frames, timestamps)

    # ── Step 3: Preview ──
    progress(0.30, desc="🖼️ Generating preview...")
    strip = create_frame_strip(frames, timestamps)
    
    if timestamps and len(timestamps) > 1:
        ts_lines = ", ".join(f"{ts:.1f}s" for ts in timestamps[:5])
        if len(timestamps) > 5:
            ts_lines += f" ... → {timestamps[-1]:.1f}s"
        frame_info = (
            f"📸 Extracted **{len(frames)}** frames (requested {frame_count})\n"
            f"⏱ **Timestamps:** {ts_lines}"
        )
    else:
        frame_info = f"📸 Extracted **{len(frames)}** frames (requested {frame_count})"
    yield strip, info_text + "\n\n" + frame_info, "", "", "", frames_temp_dir

    # ── Step 4: Build Prompt ──
    progress(0.35, desc="📝 Building prompt...")
    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()
        display_mode = "Custom"
    else:
        for mode_name, mode_data in PROMPT_MODES.items():
            if mode_name == mode:
                prompt = mode_data["prompt"].format(n=len(frames))
                display_mode = mode_name
                break
        else:
            # mode might be the key (e.g., "nsfw_check") — try matching by key
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
    ), "", "", "", frames_temp_dir

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
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful vision-language assistant. "
                    "Answer directly with the final answer only. No <think> and no reasoning."
                ),
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }

    api_start = time.time()
    try:
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        
        # Normalize URL: ensure it has /v1/chat/completions
        url = api_url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url = url + "/v1/chat/completions"
        
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        yield strip, f"❌ Cannot connect to API: `{api_url}`\n\nIs the server running?", "", "", "", frames_temp_dir
        return
    except requests.exceptions.Timeout:
        yield strip, "❌ API request timed out (300s). Try fewer frames or smaller max_size.", "", "", "", frames_temp_dir
        return
    except Exception as e:
        yield strip, f"❌ API error: {e}", "", "", "", frames_temp_dir
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"{video_name}_{mode}_{ts}.txt"

    yield strip, stats, cleaned, cleaned, default_filename, frames_temp_dir


# ── Model Fetch ─────────────────────────────────────────────────────
def fetch_models(api_url: str, api_key: str = "") -> list[tuple[str, str]]:
    """Fetch available models from llama.cpp API. Returns list of (display_name, model_id)."""
    if not api_url or not api_url.strip():
        return [("⚠️ Enter API URL first", "")]
    
    base_url = api_url.strip().rstrip("/")
    for suffix in ["/v1/chat/completions", "/v1", "/chat/completions"]:
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break
    
    models_url = f"{base_url}/v1/models"
    
    try:
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        resp = requests.get(models_url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return [(f"❌ Cannot connect to {base_url}", "")]
    except requests.exceptions.Timeout:
        return [("❌ Connection timed out", "")]
    except Exception as e:
        return [(f"❌ Error: {str(e)[:80]}", "")]
    
    models = data.get("data", data.get("models", []))
    if not models:
        return [("⚠️ No models found", "")]
    
    models_arr = data.get("models", [])
    data_arr = data.get("data", [])
    
    caps_map = {}
    for m in models_arr:
        model_id = m.get("name", m.get("id", ""))
        caps_map[model_id] = m.get("capabilities", [])
    
    if data_arr:
        models = data_arr
    else:
        models = data_arr or models_arr
    
    result = []
    for m in models:
        model_id = m.get("id", m.get("name", ""))
        
        caps = caps_map.get(model_id, m.get("capabilities", []))
        caps_str = ""
        if caps:
            cap_icons = []
            if "multimodal" in caps or "vision" in caps:
                cap_icons.append("👁️")
            if "completion" in caps:
                cap_icons.append("💬")
            caps_str = " " + "".join(cap_icons)
        
        meta = m.get("meta", {})
        size_gb = meta.get("size", 0) / (1024**3)
        size_str = f" ({size_gb:.1f}GB)" if size_gb > 0 else ""
        
        display = f"{model_id}{caps_str}{size_str}"
        result.append((display, model_id))
    
    return result if result else [(f"⚠️ No models found", "")]


# ── Gradio Theme (Gradio 6.0: pass to launch(), not Blocks) ──────
_GRADIO_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

_GRADIO_CSS = """
.container { max-width: 1200px; margin: 0 auto; }
.output-text textarea { font-size: 1rem !important; line-height: 1.6 !important; }
.stats-box { font-size: 0.95rem; }
.title-gradient { 
    background: linear-gradient(135deg, #ff6b35, #f7c948); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    font-weight: 800; 
}
.preset-active { border: 2px solid #ff6b35 !important; }
footer { display: none !important; }
"""


# ── UI ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Video-to-Prompt | AI Video Analyzer",
        analytics_enabled=False,
    ) as demo:
        # ── State ──
        history_state = gr.State([])  # list of dicts: {ts, mode, text}
        frames_dir_state = gr.State(None)  # path to temp frames directory
        
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
                with gr.Row():
                    api_url = gr.Textbox(
                        label="API Endpoint",
                        value=DEFAULT_API_URL,
                        placeholder="http://192.168.3.177:8080/v1/chat/completions",
                        scale=4,
                    )
                    refresh_btn = gr.Button(
                        "🔄 Check Models",
                        variant="secondary",
                        scale=1,
                        size="sm",
                    )
                
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=[],
                    value=DEFAULT_MODEL,
                    allow_custom_value=True,
                    interactive=True,
                    info="Click 🔄 Check Models to fetch available models, or type manually",
                )
                model_status = gr.Markdown("")

                with gr.Accordion("🔑 Authentication (optional)", open=False):
                    api_key = gr.Textbox(
                        label="API Key",
                        value="",
                        placeholder="Bearer token for authenticated endpoints (leave empty if none)",
                        type="password",
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
                
                # ── Example Prompts ──
                gr.Markdown("### 💡 Example Prompts")
                example_prompts = gr.Dropdown(
                    choices=[
                        "✨ Select an example prompt...",
                        "🎬 Movie Director Shot List",
                        "📸 Photo/Art Style Analysis",
                        "🎨 Dominant Color Palette",
                        "🔍 Object & Brand Detection",
                        "📝 Social Media Caption",
                    ],
                    value="✨ Select an example prompt...",
                    label="Quick fill",
                    interactive=True,
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
                        minimum=128, maximum=DEFAULT_MAX_TOKENS_LIMIT, value=DEFAULT_MAX_TOKENS, step=128,
                        label="Max output tokens",
                        info=f"Up to {DEFAULT_MAX_TOKENS_LIMIT} tokens",
                    )
                    
                    # ── Temperature with presets ──
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.6, step=0.05,
                        label="Temperature",
                        info="Controls creativity/randomness",
                    )
                    with gr.Row():
                        temp_creative = gr.Button("🎨 Creative", size="sm", scale=1)
                        temp_balanced = gr.Button("⚖️ Balanced", size="sm", scale=1)
                        temp_precise = gr.Button("🎯 Precise", size="sm", scale=1)
                    
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

                # ── Action row: download frames + history ──
                with gr.Row():
                    download_frames_btn = gr.Button("📦 Download Frames", scale=1, variant="secondary")
                    download_output = gr.File(label="", visible=False)
                    download_status = gr.Markdown("")
                
                save_status = gr.Markdown("")

                # ── Output History ──
                with gr.Accordion("📜 Output History (last 5)", open=False):
                    history_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Select a previous result",
                        interactive=True,
                    )

        # ── Event Bindings ──

        # Temperature presets
        def set_temp_creative():
            return 0.9
        def set_temp_balanced():
            return 0.6
        def set_temp_precise():
            return 0.3
        
        temp_creative.click(fn=set_temp_creative, outputs=[temperature])
        temp_balanced.click(fn=set_temp_balanced, outputs=[temperature])
        temp_precise.click(fn=set_temp_precise, outputs=[temperature])

        def on_save(text, filename):
            if not text or not text.strip():
                return "❌ Nothing to save"
            path = Path(filename)
            path.write_text(text, encoding="utf-8")
            return f"✅ Saved to: `{path.resolve()}`"

        def on_copy(text):
            return text  # gr.Textbox handles clipboard

        def on_download_frames(frames_dir):
            """Create ZIP from frames directory and return for download."""
            if not frames_dir or not os.path.isdir(frames_dir):
                return None, "⚠️ No frames available. Generate a prompt first."
            try:
                zip_path = make_frames_zip(frames_dir)
                return zip_path, f"✅ {len(os.listdir(frames_dir))} frames ready for download"
            except Exception as e:
                return None, f"❌ Failed to create ZIP: {e}"

        def on_refresh_models(api_url_val, api_key_val):
            """Refresh model list from API."""
            if not api_url_val or not api_url_val.strip():
                return gr.update(choices=[], value=""), "⚠️ Enter API URL first"
            
            models = fetch_models(api_url_val, api_key_val or "")
            choices = [display for display, _ in models]
            values = [model_id for _, model_id in models]
            
            if not choices:
                return gr.update(choices=[], value=""), "⚠️ No models found"
            
            first = choices[0]
            if first.startswith("❌") or first.startswith("⚠️"):
                return gr.update(choices=choices, value=""), first
            
            multimodal_count = sum(1 for c in choices if "👁️" in c)
            status = f"✅ **{len(choices)}** models found"
            if multimodal_count > 0:
                status += f" — **{multimodal_count}** with vision 👁️"
            else:
                status += " (no vision models detected ⚠️)"
            
            default_val = values[0]
            for c, v in zip(choices, values):
                if "👁️" in c:
                    default_val = v
                    break
            
            return (
                gr.update(choices=choices, value=default_val),
                status,
            )

        # ── History management ──
        def add_to_history(history, mode_display, output_text_val):
            """Add a result to history (max 5)."""
            if not output_text_val or not output_text_val.strip():
                return history, gr.update(choices=[])
            
            entry = {
                "ts": datetime.now().strftime("%H:%M:%S"),
                "mode": mode_display,
                "text": output_text_val,
            }
            new_history = (history or []) + [entry]
            # Keep only last 5
            if len(new_history) > MAX_HISTORY:
                new_history = new_history[-MAX_HISTORY:]
            
            choices = [
                f"[{h['ts']}] {h['mode']}: {h['text'][:60]}..."
                for h in new_history
            ]
            return new_history, gr.update(choices=choices, value=choices[-1] if choices else None)

        def on_history_select(history, selected):
            """Display selected history entry."""
            if not history or not selected:
                return ""
            for h in history:
                label = f"[{h['ts']}] {h['mode']}: {h['text'][:60]}..."
                if label == selected:
                    return h["text"]
            return ""

        # ── Wire up process_video with history ──
        def process_and_log(
            video, url, key, model, mode_sel, custom_p, frames_n,
            max_sz, max_tok, temp, qual, history,
            progress=gr.Progress(),
        ):
            """Wrapper: run process_video and add to history on final yield."""
            last_strip = None
            last_status = None
            last_output = None
            last_filename = None
            last_frames_dir = None
            display_mode = mode_sel
            
            for output in process_video(
                video, url, key, model, mode_sel, custom_p,
                frames_n, max_sz, max_tok, temp, qual, progress,
            ):
                if len(output) == 6:
                    strip, status, out_txt1, out_txt2, filename, frames_dir = output
                elif len(output) == 5:
                    # Backward compat: older process_video with 5 outputs
                    strip, status, out_txt1, out_txt2, filename = output
                    frames_dir = None
                else:
                    continue
                
                last_strip = strip
                last_status = status
                last_output = out_txt2  # cleaned output
                last_filename = filename
                last_frames_dir = frames_dir
                yield strip, status, out_txt1, out_txt2, filename, frames_dir
            
            # Add to history if we got a real output
            if last_output and last_output.strip():
                # Determine display mode text
                if custom_p and custom_p.strip():
                    mode_text = "Custom"
                else:
                    mode_text = mode_sel
                
                new_history, history_update = add_to_history(history, mode_text, last_output)
                # Note: we can't yield additional outputs here, so we update history_state
                # through the last yield already done. We need a different approach...
                # Let's use the history dropdown update as a side effect.
                # Actually, let's have the history updated in a separate callback chain.

        # ── Click handlers ──
        # Auto-load models on page load
        demo.load(
            fn=on_refresh_models,
            inputs=[api_url, api_key],
            outputs=[model_name, model_status],
        )

        refresh_btn.click(
            fn=on_refresh_models,
            inputs=[api_url, api_key],
            outputs=[model_name, model_status],
        )

        # Example prompt selector → fill custom_prompt
        def on_example_select(selected):
            if not selected or selected.startswith("✨"):
                return ""
            return EXAMPLE_PROMPTS.get(selected, "")

        example_prompts.change(
            fn=on_example_select,
            inputs=[example_prompts],
            outputs=[custom_prompt],
        )

        # Process video
        process_event = process_btn.click(
            fn=process_video,
            inputs=[
                video_input, api_url, api_key, model_name, mode, custom_prompt,
                frame_count, max_size, max_tokens, temperature, quality,
            ],
            outputs=[
                frame_preview, status_output, output_text, output_text,
                output_filename, frames_dir_state,
            ],
        )

        # After process completes, update history
        def update_history_after(history, mode_sel, custom_p, output_val):
            """Update history after process completes."""
            if not output_val or not output_val.strip():
                return history, gr.update()
            
            mode_text = "Custom" if (custom_p and custom_p.strip()) else mode_sel
            
            entry = {
                "ts": datetime.now().strftime("%H:%M:%S"),
                "mode": mode_text,
                "text": output_val,
            }
            new_history = (history or []) + [entry]
            if len(new_history) > MAX_HISTORY:
                new_history = new_history[-MAX_HISTORY:]
            
            choices = [
                f"[{h['ts']}] {h['mode']}: {h['text'][:60]}..."
                for h in new_history
            ]
            return new_history, gr.update(choices=choices, value=choices[-1] if choices else None)

        process_event.then(
            fn=update_history_after,
            inputs=[history_state, mode, custom_prompt, output_text],
            outputs=[history_state, history_dropdown],
        )

        # Download frames
        download_frames_btn.click(
            fn=on_download_frames,
            inputs=[frames_dir_state],
            outputs=[download_output, download_status],
        )

        # History selection
        history_dropdown.change(
            fn=on_history_select,
            inputs=[history_state, history_dropdown],
            outputs=[output_text],
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
        theme=_GRADIO_THEME,
        css=_GRADIO_CSS,
    )


# Allow import without running argparse
if __name__ == "__main__":
    import argparse
    main()
