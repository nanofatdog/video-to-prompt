#!/usr/bin/env python3
"""
video-to-prompt — Extract frames from video, send to MoLE vision API, get prompt/description.
Uses llama.cpp API at 192.168.3.177:8080/v1

Usage:
    python3 video_to_prompt.py video.mp4
    python3 video_to_prompt.py video.mp4 -n 16 -m describe
    python3 video_to_prompt.py video.mp4 -n 32 --mode tag --output tags.txt
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────
API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "llmfan46_Qwen3.6-35B-A3B-uncensored-heretic-Q6_K.gguf"
DEFAULT_FRAME_COUNT = 16
MAX_TOKENS = 1024

# ── Prompt Templates ────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "describe": (
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
    "summarize": (
        "These {n} frames are sampled from a video. Describe what happens in the video based on "
        "these frames. What is the subject doing? Where are they? What is the overall "
        "action/event taking place? Be concise but complete."
    ),
    "tag": (
        "Analyze these {n} video frames. Extract tags/keywords describing the content.\n"
        "Output ONLY comma-separated tags in English, no other text. "
        "Include: subjects, objects, actions, setting, style, lighting, mood, colors, "
        "camera angle, time of day, quality descriptors.\n"
        "Example: portrait, woman, outdoor, sunset, golden hour, bokeh, soft lighting, "
        "photorealistic, 8k, shallow depth of field"
    ),
    "booru": (
        "You are a Danbooru tagger. Analyze these {n} video frames and output ONLY "
        "space-separated Danbooru-style tags. Include: character count, gender, hair color, "
        "eye color, clothing, pose, expression, background, art style, quality tags.\n"
        "Format: tag1 tag2 tag3 ... (no commas, no other text)"
    ),
    "nsfw_check": (
        "Analyze these {n} video frames. Rate the content on these scales (1-10):\n"
        "- Suggestive: (1=fully clothed, 10=explicit)\n"
        "- Violence: (1=none, 10=extreme)\n"
        "- Overall NSFW: (1=completely safe, 10=explicit adult content)\n"
        "Output only the three numbers like: 3,1,2"
    ),
}


@dataclass
class VideoInfo:
    path: Path
    duration: float  # seconds
    fps: float
    total_frames: int
    width: int
    height: int
    codec: str


# ── Video Utils ─────────────────────────────────────────────────────
def probe_video(path: str) -> VideoInfo:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video":
            video_stream = stream
            break
    if not video_stream:
        raise ValueError("No video stream found")

    # Parse duration
    duration = float(data["format"].get("duration", 0))
    if duration == 0 and "tags" in video_stream and "DURATION" in video_stream["tags"]:
        # Try alternate duration source
        dur_str = video_stream["tags"]["DURATION"]
        h, m, s = map(float, dur_str.replace(",", ".").split(":"))
        duration = h * 3600 + m * 60 + s

    # Parse FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    # Total frames
    nb_frames = int(video_stream.get("nb_frames", 0))
    if nb_frames == 0 and duration > 0:
        nb_frames = int(duration * fps)

    return VideoInfo(
        path=Path(path),
        duration=duration,
        fps=fps,
        total_frames=nb_frames,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        codec=video_stream.get("codec_name", "unknown"),
    )


def extract_frames_ffmpeg(
    path: str,
    frame_count: int = 16,
    max_size: int = 1280,
) -> list[Image.Image]:
    """
    Extract frames using ffmpeg with uniform sampling.
    Much faster than reading every frame with OpenCV.
    """
    info = probe_video(path)
    duration = info.duration

    if duration <= 0:
        # Fallback: extract keyframes
        cmd = [
            "ffmpeg", "-v", "quiet", "-skip_frame", "nokey",
            "-i", path, "-vf", f"scale='min({max_size},iw)':-1,select=eq(pict_type\\,I)",
            "-vframes", str(frame_count), "-f", "image2pipe", "-vcodec", "png", "-"
        ]
    else:
        # Uniform sampling using fps filter
        target_fps = frame_count / duration
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-i", path,
            "-vf", f"scale='min({max_size},iw)':-1,fps={target_fps:.6f}",
            "-vframes", str(frame_count),
            "-f", "image2pipe", "-vcodec", "png", "-"
        ]

    result = subprocess.run(cmd, capture_output=True, check=True)
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
            img.load()  # force decode
            frames.append(img)
        except Exception:
            continue
    return frames


def extract_frames_cv2(
    path: str,
    frame_count: int = 16,
    max_size: int = 1280,
) -> list[Image.Image]:
    """Fallback: extract frames using OpenCV (reads every frame, slower)."""
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {path}")

    indices = np.linspace(0, total - 1, frame_count, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize if needed
        if max_size and max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        frames.append(img)
    cap.release()
    return frames


def extract_frames(
    path: str,
    frame_count: int = 16,
    max_size: int = 1280,
) -> list[Image.Image]:
    """Extract frames — try ffmpeg first, fallback to OpenCV."""
    try:
        frames = extract_frames_ffmpeg(path, frame_count, max_size)
        if len(frames) >= 2:
            return frames
    except Exception as e:
        print(f"  ⚠️  ffmpeg extraction failed: {e}")

    print("  🔄  Falling back to OpenCV...")
    return extract_frames_cv2(path, frame_count, max_size)


# ── API ─────────────────────────────────────────────────────────────
def frame_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 JPEG (smaller than PNG)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def estimate_tokens(frames: list[Image.Image]) -> int:
    """Rough estimate: each 1280px frame ≈ 500-800 vision tokens for Qwen-VL."""
    total_pixels = sum(f.width * f.height for f in frames)
    # Qwen-VL uses ~0.5-0.8 tokens per 1000 pixels after patch embedding
    return int(total_pixels * 0.6 / 1000)


def call_vision_api(
    frames: list[Image.Image],
    prompt: str,
    model: str = MODEL_NAME,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.6,
    verbose: bool = True,
) -> dict:
    """
    Send frames + prompt to llama.cpp vision API.
    Returns full response dict.
    """
    # Build content array
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        b64 = frame_to_base64(frame)
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

    if verbose:
        est_tok = estimate_tokens(frames)
        print(f"  📤 Sending {len(frames)} frames (~{est_tok} vision tokens) + prompt...")

    t0 = time.perf_counter()
    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0

    data = resp.json()
    if verbose:
        usage = data.get("usage", {})
        print(f"  ✅ Response in {elapsed:.1f}s | "
              f"prompt_tokens={usage.get('prompt_tokens','?')} "
              f"completion_tokens={usage.get('completion_tokens','?')}")

    return data


# ── Main Pipeline ───────────────────────────────────────────────────
def video_to_prompt(
    video_path: str,
    frame_count: int = DEFAULT_FRAME_COUNT,
    mode: str = "describe",
    custom_prompt: Optional[str] = None,
    max_size: int = 1280,
    model: str = MODEL_NAME,
    verbose: bool = True,
) -> str:
    """
    Extract frames from video and generate a prompt/description via vision API.

    Returns the generated text.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 1. Probe video
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎬  {path.name}")

    info = probe_video(str(path))
    if verbose:
        print(f"  📐 {info.width}x{info.height} | {info.duration:.1f}s | "
              f"{info.fps:.1f}fps | {info.total_frames} frames | {info.codec}")

    # 2. Extract frames
    if verbose:
        print(f"  🎞️  Extracting {frame_count} frames (max {max_size}px)...")
    frames = extract_frames(str(path), frame_count, max_size)
    if verbose:
        print(f"  📸  Got {len(frames)} frames")

    if len(frames) == 0:
        raise RuntimeError("Failed to extract any frames from video")

    # 3. Build prompt
    if custom_prompt:
        prompt = custom_prompt
    else:
        template = PROMPT_TEMPLATES.get(mode)
        if not template:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(PROMPT_TEMPLATES.keys())}")
        prompt = template.format(n=len(frames))

    # 4. Call API
    response = call_vision_api(frames, prompt, model=model, verbose=verbose)

    # 5. Extract text
    message = response.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")

    # If content is empty, check reasoning_content (thinking models)
    if not content.strip():
        content = message.get("reasoning_content", "")

    # Clean up thinking model output:
    # Some models output reasoning inline, ending with a blank line then final answer.
    # Try to extract just the final answer.
    lines = content.strip().split("\n")
    
    # Pattern 1: Model wraps final answer after "---" or double newline + summary
    # Look for the last substantial paragraph that doesn't look like reasoning
    reasoning_markers = [
        "The user wants", "Let me", "I need to", "First,", "Step 1",
        "Analyze the", "Looking at", "The prompt asks", "I'll", "I should",
        "Wait,", "Hmm,", "I see", "This is a", "Okay,", "Let's",
        "The image", "Frame 1", "Frame 2", "Self-Correction",
        "Draft", "Review against", "Final check", "Final Polish",
        "Refining", "Synthesize", "So the",
    ]
    
    # Find the last line that doesn't look like reasoning
    cleaned_lines = []
    in_reasoning = True
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.insert(0, line)
            continue
        is_reasoning = any(stripped.startswith(m) for m in reasoning_markers)
        if not is_reasoning and in_reasoning:
            in_reasoning = False
        if not in_reasoning:
            cleaned_lines.insert(0, line)
    
    if cleaned_lines:
        content = "\n".join(cleaned_lines).strip()
    
    # Pattern 2: <｜end▁of▁thinking｜> marker (Qwen thinking format)
    if " response" in content:
        content = content.split(" response")[-1].strip()

    return content.strip()


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🎬 video-to-prompt — Extract frames & generate AI prompts from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modes:
  describe     — Detailed visual description for image generation (default)
  summarize    — Brief summary of what happens in the video
  tag          — Comma-separated keywords/tags
  booru        — Danbooru-style tags
  nsfw_check   — NSFW rating (suggestive/violence/overall)

Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 -n 32 -m tag
  %(prog)s video.mp4 -n 16 --mode booru -o tags.txt
  %(prog)s video.mp4 --custom "Is this video safe for work? Answer yes/no only."
  %(prog)s video.mp4 -n 8 --max-size 512  # faster, lower quality
        """,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-n", "--frames", type=int, default=DEFAULT_FRAME_COUNT,
                        help=f"Number of frames to sample (default: {DEFAULT_FRAME_COUNT})")
    parser.add_argument("-m", "--mode", default="describe",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Prompt mode (default: describe)")
    parser.add_argument("--custom", type=str, default=None,
                        help="Custom prompt (overrides mode)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save output to file")
    parser.add_argument("--max-size", type=int, default=1280,
                        help="Max frame dimension in pixels (default: 1280)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Model name for API")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON with metadata")

    args = parser.parse_args()

    try:
        result = video_to_prompt(
            video_path=args.video,
            frame_count=args.frames,
            mode=args.mode,
            custom_prompt=args.custom,
            max_size=args.max_size,
            model=args.model,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        info = probe_video(args.video)
        output = json.dumps({
            "video": str(info.path),
            "duration": info.duration,
            "fps": info.fps,
            "resolution": f"{info.width}x{info.height}",
            "frames_sampled": args.frames,
            "mode": args.mode,
            "prompt": result,
        }, indent=2, ensure_ascii=False)
    else:
        output = result

    print(f"\n{'='*60}")
    print(output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\n💾 Saved to: {args.output}")

    # Copy to clipboard if available
    try:
        subprocess.run(["xclip", "-selection", "c"],
                       input=output.encode(), check=False, capture_output=True)
        if not args.quiet:
            print("📋 Copied to clipboard!")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
