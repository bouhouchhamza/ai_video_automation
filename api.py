import utils.session
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from urllib.parse import quote
import requests
import subprocess
import traceback
import os
import json
import random
import re
import shutil
import sys
import tempfile
import asyncio

load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

def get_env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def get_env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def get_env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}

def clean_env_path(value):
    return str(value or "").strip().strip('"').strip("'")

PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_DIR / "templates"
STATIC_DIR = PROJECT_DIR / "static"
OUTPUT_DIR = PROJECT_DIR / "output"
CLIP_MEMORY_FILE = OUTPUT_DIR / "clip_manifest.json"

MIN_BACKGROUND_CLIPS = 3
MAX_BACKGROUND_CLIPS = 5
PEXELS_SEARCH_RESULTS_PER_QUERY = 12
RECENT_CLIP_MEMORY_LIMIT = 80

FFMPEG_PATH = Path(
    os.getenv(
        "FFMPEG_PATH",
        r"C:\Users\pc\Downloads\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
    )
)

TTS_ENGINE = os.getenv("TTS_ENGINE", "edge").strip().lower()
TTS_MODEL_NAME = os.getenv(
    "TTS_MODEL_NAME",
    "tts_models/multilingual/multi-dataset/xtts_v2"
)
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en").strip() or "en"
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto").strip().lower()
TTS_VOICE = os.getenv("TTS_VOICE", "Craig Gutsy").strip()
TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", "").strip()

TTS_EDGE_VOICE = os.getenv("TTS_EDGE_VOICE", "en-US-GuyNeural").strip()
TTS_RATE = os.getenv("TTS_RATE", "-8%").strip()
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz").strip()

TTS_CPU_THREADS = get_env_int("TTS_CPU_THREADS", 4)
TTS_SAMPLE_RATE = get_env_int("TTS_SAMPLE_RATE", 44100)
TTS_SENTENCE_PACING = os.getenv("TTS_SENTENCE_PACING", "true").strip().lower() != "false"
TTS_MIN_PAUSE_SECONDS = get_env_float("TTS_MIN_PAUSE_SECONDS", 0.28)
TTS_MAX_PAUSE_SECONDS = get_env_float("TTS_MAX_PAUSE_SECONDS", 0.85)
TTS_MAX_SEGMENT_CHARS = get_env_int("TTS_MAX_SEGMENT_CHARS", 230)

RVC_ROOT = clean_env_path(os.getenv("RVC_ROOT", ""))
RVC_PYTHON_PATH = clean_env_path(os.getenv("RVC_PYTHON_PATH", ""))
RVC_CLI_STYLE = os.getenv("RVC_CLI_STYLE", "rvc-cli").strip().lower()
RVC_DEFAULT_INFER_SCRIPT = r"tools\infer_cli.py" if RVC_CLI_STYLE in {"rvc-webui", "webui"} else "rvc_cli.py"
RVC_INFER_SCRIPT = clean_env_path(os.getenv("RVC_INFER_SCRIPT", RVC_DEFAULT_INFER_SCRIPT))
RVC_WEBUI_MODEL_NAME = os.getenv("RVC_WEBUI_MODEL_NAME", "").strip()
RVC_MODEL_PATH = clean_env_path(os.getenv("RVC_MODEL_PATH", ""))
RVC_INDEX_PATH = clean_env_path(os.getenv("RVC_INDEX_PATH", ""))
RVC_COMMAND_TEMPLATE = os.getenv("RVC_COMMAND_TEMPLATE", "").strip()
RVC_ENABLED = get_env_bool("RVC_ENABLED", bool(RVC_MODEL_PATH))
RVC_CLEANUP_TEMP_FILES = get_env_bool("RVC_CLEANUP_TEMP_FILES", True)
RVC_PITCH = get_env_int("RVC_PITCH", 0)
RVC_F0_METHOD = os.getenv("RVC_F0_METHOD", "rmvpe").strip()
RVC_INDEX_RATE = os.getenv("RVC_INDEX_RATE", "0.75").strip()
RVC_PROTECT = os.getenv("RVC_PROTECT", "0.33").strip()
RVC_FILTER_RADIUS = get_env_int("RVC_FILTER_RADIUS", 3)
RVC_RESAMPLE_SR = get_env_int("RVC_RESAMPLE_SR", 0)
RVC_TIMEOUT_SECONDS = get_env_int("RVC_TIMEOUT_SECONDS", 600)
RVC_INCLUDE_DEVICE_ARG = get_env_bool("RVC_INCLUDE_DEVICE_ARG", False)
RVC_DEVICE = os.getenv("RVC_DEVICE", "auto").strip()

STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Local AI Video Generator",
    description="A local dashboard for generating faceless short-form videos.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class GenerateRequest(BaseModel):
    topic: str = "AI facts"
    voice: str | None = None
    speaker_wav: str | None = None

@app.get("/")
def home():
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend template not found")
    return FileResponse(index_path)

def slugify(value):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:48] or "video"

def get_video_metadata(video_path):
    metadata_path = video_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def get_generated_video_paths():
    videos = [path for path in OUTPUT_DIR.glob("video_*.mp4") if path.is_file()]
    videos.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    latest_video = OUTPUT_DIR / "final_video.mp4"
    if not videos and latest_video.exists():
        videos.append(latest_video)

    return videos

def build_video_response(video_path):
    metadata = get_video_metadata(video_path)
    stat = video_path.stat()
    created_at = metadata.get("created_at") or datetime.fromtimestamp(
        stat.st_mtime
    ).astimezone().isoformat(timespec="seconds")

    encoded_name = quote(video_path.name)

    return {
        "filename": video_path.name,
        "topic": metadata.get("topic") or "Generated video",
        "created_at": created_at,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "url": f"/videos/{encoded_name}",
        "download_url": f"/download/{encoded_name}"
    }

def get_video_history():
    return [build_video_response(path) for path in get_generated_video_paths()]

def resolve_video_path(filename):
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if safe_name != "final_video.mp4" and not safe_name.startswith("video_"):
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = (OUTPUT_DIR / safe_name).resolve()
    output_root = OUTPUT_DIR.resolve()

    try:
        video_path.relative_to(output_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid filename") from exc

    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Video not found")

    return video_path

@app.get("/videos")
def videos():
    return {"success": True, "videos": get_video_history()}

@app.get("/videos/{filename}")
def stream_video(filename: str):
    video_path = resolve_video_path(filename)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"}
    )

@app.get("/download/{filename}")
def download_video(filename: str):
    video_path = resolve_video_path(filename)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=video_path.name,
        headers={"Cache-Control": "no-store"}
    )

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def normalize_tts_text(text):
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace(". ", "... ")
    cleaned = cleaned.replace("? ", "? ... ")
    cleaned = cleaned.replace("! ", "! ... ")
    replacements = [
        (r"\bAI\b", "A I"),
        (r"\bAGI\b", "A G I"),
        (r"\bGPT\b", "G P T"),
        (r"\bLLM\b", "L L M"),
        (r"\bGPU\b", "G P U"),
        (r"\bCPU\b", "C P U"),
        (r"\bCEO\b", "C E O"),
        (r"\bNFT\b", "N F T"),
        (r"\bNASA\b", "NASA"),
        (r"\bUFO\b", "U F O"),
        (r"\bTikTok\b", "TikTok")
    ]

    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = cleaned.replace("...", ". ")
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([.!?])([A-Z])", r"\1 \2", cleaned)

    return cleaned.strip()

def split_narration_segments(text):
    normalized = normalize_tts_text(text)
    raw_segments = re.split(r"(?<=[.!?])\s+", normalized)
    segments = []

    for raw_segment in raw_segments:
        raw_segment = raw_segment.strip()
        if not raw_segment:
            continue

        if len(raw_segment) <= TTS_MAX_SEGMENT_CHARS:
            segments.append(raw_segment)
            continue

        parts = re.split(r"(?<=[,;:])\s+", raw_segment)
        current = ""

        for part in parts:
            next_value = f"{current} {part}".strip()
            if current and len(next_value) > TTS_MAX_SEGMENT_CHARS:
                segments.append(current.strip())
                current = part
            else:
                current = next_value

        if current:
            segments.append(current.strip())

    return segments or [normalized]

def pause_after_segment(segment, index, total):
    if total <= 1 or index >= total - 1:
        return 0.0

    pause = 0.42
    stripped = segment.strip()

    if stripped.endswith("?"):
        pause = 0.58
    elif stripped.endswith("!"):
        pause = 0.5
    elif stripped.endswith((".", ":")):
        pause = 0.48
    elif stripped.endswith((",", ";")):
        pause = 0.32

    if len(stripped) < 55:
        pause += 0.08

    if re.search(r"\bbut\b|\bthen\b|\bhere'?s\b|\bthis is\b", stripped, re.I):
        pause += 0.08

    return max(TTS_MIN_PAUSE_SECONDS, min(TTS_MAX_PAUSE_SECONDS, pause))

def get_speaker_wav_paths(speaker_wav=None):
    speaker_wav_value = (speaker_wav or TTS_SPEAKER_WAV).strip()
    if not speaker_wav_value:
        return []

    speaker_path = Path(speaker_wav_value)
    if not speaker_path.exists():
        return []

    if speaker_path.is_dir():
        wavs = sorted(path for path in speaker_path.glob("*.wav") if path.is_file())
        return [str(path) for path in wavs]

    return [str(speaker_path)]

class RealisticVoiceGenerator:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.torch = None
        self.available_speakers = []

    def synthesize_to_file(self, script, output_path, voice=None, speaker_wav=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        if TTS_ENGINE == "edge":
            return self._synthesize_edge_to_file(script, output_path, voice)

        if TTS_ENGINE == "xtts":
            return self._synthesize_xtts_to_file(script, output_path, voice, speaker_wav)

        raise ValueError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}")

    def _synthesize_edge_to_file(self, script, output_path, voice=None):
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts is missing. Run: pip install edge-tts") from exc

        output_path = Path(output_path)
        mp3_path = output_path.with_suffix(".mp3")
        voice_name = (voice or TTS_EDGE_VOICE).strip() or "en-US-GuyNeural"
        text = normalize_tts_text(script)

        debug = {
            "engine": "edge",
            "voice": voice_name,
            "rate": TTS_RATE,
            "pitch": TTS_PITCH,
            "mp3_path": str(mp3_path),
            "wav_path": str(output_path),
            "warnings": []
        }

        if mp3_path.exists():
            mp3_path.unlink()

        if output_path.exists():
            output_path.unlink()

        async def run_edge():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_name,
                rate=TTS_RATE,
                pitch=TTS_PITCH
            )
            await communicate.save(str(mp3_path))

        asyncio.run(run_edge())

        if not mp3_path.exists() or mp3_path.stat().st_size < 1000:
            raise RuntimeError("Edge TTS did not create voice.mp3")

        if not FFMPEG_PATH.exists():
            raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_PATH}")

        convert_command = [
            str(FFMPEG_PATH),
            "-y",
            "-i", str(mp3_path),
            "-ar", str(TTS_SAMPLE_RATE),
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]

        result = subprocess.run(convert_command, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        if not output_path.exists() or output_path.stat().st_size < 1000:
            raise RuntimeError("Edge TTS did not create voice.wav")

        debug["wav_size_bytes"] = output_path.stat().st_size
        return debug

    def _synthesize_xtts_to_file(self, script, output_path, voice=None, speaker_wav=None):
        output_path = Path(output_path)
        voice_name = (voice or TTS_VOICE).strip()
        speaker_wav_paths = get_speaker_wav_paths(speaker_wav)

        debug = {
            "engine": "xtts",
            "model": TTS_MODEL_NAME,
            "language": TTS_LANGUAGE,
            "requested_device": TTS_DEVICE,
            "device": None,
            "voice": voice_name or None,
            "speaker_wav": speaker_wav_paths,
            "sentence_pacing": TTS_SENTENCE_PACING,
            "segments": [],
            "warnings": []
        }

        if speaker_wav and not speaker_wav_paths:
            debug["warnings"].append(f"speaker_wav was provided but no WAV files were found: {speaker_wav}")

        if output_path.exists():
            output_path.unlink()

        model = self._load_model(debug)
        segments = split_narration_segments(script) if TTS_SENTENCE_PACING else [normalize_tts_text(script)]

        debug["segments"] = [
            {
                "index": index + 1,
                "characters": len(segment),
                "pause_after_seconds": round(pause_after_segment(segment, index, len(segments)), 2)
            }
            for index, segment in enumerate(segments)
        ]

        with tempfile.TemporaryDirectory(prefix="tts_", dir=output_path.parent) as temp_dir:
            temp_dir_path = Path(temp_dir)
            part_paths = []

            for index, segment in enumerate(segments):
                part_path = temp_dir_path / f"voice_part_{index:03}.wav"
                self._synthesize_segment(
                    model,
                    segment,
                    part_path,
                    debug,
                    voice_name,
                    speaker_wav_paths
                )

                if not part_path.exists() or part_path.stat().st_size < 1000:
                    raise RuntimeError(f"TTS produced an empty audio segment at {part_path}")

                part_paths.append(part_path)

            self._render_paced_audio(part_paths, segments, output_path)

        if not output_path.exists() or output_path.stat().st_size < 1000:
            raise RuntimeError("voice.wav was not created by XTTS")

        return debug

    def _load_model(self, debug):
        if self.model is not None:
            debug["device"] = self.device
            return self.model

        try:
            import torch
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError("XTTS dependencies are missing. Install coqui-tts and torch first.") from exc

        self.torch = torch
        self.device = self._select_device(torch, debug)

        if self.device == "cpu" and TTS_CPU_THREADS > 0:
            torch.set_num_threads(TTS_CPU_THREADS)

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        self.model = TTS(TTS_MODEL_NAME).to(self.device)
        self.available_speakers = list(getattr(self.model, "speakers", None) or [])
        debug["device"] = self.device
        debug["available_speakers"] = self.available_speakers[:12]

        return self.model

    def _select_device(self, torch, debug):
        cuda_available = torch.cuda.is_available()

        if TTS_DEVICE == "cpu":
            return "cpu"

        if TTS_DEVICE.startswith("cuda"):
            if cuda_available:
                return TTS_DEVICE

            debug["warnings"].append("CUDA was requested but is not available. Falling back to CPU.")
            return "cpu"

        return "cuda" if cuda_available else "cpu"

    def _voice_kwargs(self, debug, voice_name, speaker_wav_paths):
        if speaker_wav_paths:
            return {"speaker_wav": speaker_wav_paths if len(speaker_wav_paths) > 1 else speaker_wav_paths[0]}

        if self.available_speakers:
            if voice_name in self.available_speakers:
                return {"speaker": voice_name}

            if "Craig Gutsy" in self.available_speakers:
                debug["warnings"].append(f"Requested voice '{voice_name}' was not found. Using Craig Gutsy.")
                return {"speaker": "Craig Gutsy"}

            fallback_speaker = self.available_speakers[0]
            debug["warnings"].append(f"Requested voice '{voice_name}' was not found. Using {fallback_speaker}.")
            return {"speaker": fallback_speaker}

        return {}

    def _synthesize_segment(self, model, segment, part_path, debug, voice_name, speaker_wav_paths):
        kwargs = {
            "text": segment,
            "language": TTS_LANGUAGE,
            "file_path": str(part_path)
        }
        kwargs.update(self._voice_kwargs(debug, voice_name, speaker_wav_paths))

        try:
            model.tts_to_file(**kwargs)
        except TypeError:
            kwargs.pop("language", None)
            model.tts_to_file(**kwargs)

    def _render_paced_audio(self, part_paths, segments, output_path):
        if not FFMPEG_PATH.exists():
            raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_PATH}")

        command = [str(FFMPEG_PATH), "-y"]
        filter_inputs = []
        input_index = 0

        for index, part_path in enumerate(part_paths):
            command.extend(["-i", str(part_path)])
            filter_inputs.append(f"[{input_index}:a]")
            input_index += 1

            pause = pause_after_segment(segments[index], index, len(segments))
            if pause > 0:
                command.extend(
                    [
                        "-f", "lavfi",
                        "-t", f"{pause:.2f}",
                        "-i", f"anullsrc=channel_layout=stereo:sample_rate={TTS_SAMPLE_RATE}"
                    ]
                )
                filter_inputs.append(f"[{input_index}:a]")
                input_index += 1

        filter_complex = f"{''.join(filter_inputs)}concat=n={len(filter_inputs)}:v=0:a=1[aout]"

        command.extend(
            [
                "-filter_complex", filter_complex,
                "-map", "[aout]",
                "-ar", str(TTS_SAMPLE_RATE),
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(output_path)
            ]
        )

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

VOICE_GENERATOR = RealisticVoiceGenerator()

class RVCConversionError(Exception):
    def __init__(self, message, debug):
        super().__init__(message)
        self.debug = debug

def tail_text(value, limit=4000):
    if not value:
        return ""

    value = str(value)
    return value[-limit:]

def resolve_rvc_path(path_value, must_exist=False):
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute() and RVC_ROOT:
        path = Path(RVC_ROOT) / path

    path = path.resolve()
    if must_exist and not path.exists():
        return None

    return path

def normalize_audio_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio input not found: {input_path}")

    if not FFMPEG_PATH.exists():
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_PATH}")

    temp_output_path = output_path
    if input_path.resolve() == output_path.resolve():
        temp_output_path = output_path.with_name(f"{output_path.stem}_normalized_tmp{output_path.suffix}")

    if temp_output_path.exists():
        temp_output_path.unlink()

    command = [
        str(FFMPEG_PATH),
        "-y",
        "-i", str(input_path),
        "-ar", str(TTS_SAMPLE_RATE),
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(temp_output_path)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    if temp_output_path != output_path:
        temp_output_path.replace(output_path)

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError(f"Normalized audio was not created: {output_path}")

def split_command_template(command_text):
    return command_text

class RVCVoiceConverter:
    def convert_to_file(self, input_path, output_path, temp_dir):
        input_path = Path(input_path)
        output_path = Path(output_path)
        temp_dir = Path(temp_dir)
        converted_path = temp_dir / "rvc_converted.wav"

        debug = {
            "enabled": RVC_ENABLED,
            "converted": False,
            "skipped": False,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "temp_output_path": str(converted_path),
            "model_path": RVC_MODEL_PATH or None,
            "index_path": RVC_INDEX_PATH or None,
            "rvc_root": RVC_ROOT or None,
            "cli_style": RVC_CLI_STYLE,
            "infer_script": RVC_INFER_SCRIPT or None,
            "webui_model_name": RVC_WEBUI_MODEL_NAME or None,
            "python_path": RVC_PYTHON_PATH or None,
            "f0_method": RVC_F0_METHOD,
            "pitch": RVC_PITCH,
            "index_rate": RVC_INDEX_RATE,
            "protect": RVC_PROTECT,
            "filter_radius": RVC_FILTER_RADIUS,
            "resample_sr": RVC_RESAMPLE_SR,
            "timeout_seconds": RVC_TIMEOUT_SECONDS,
            "command": None,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "warnings": []
        }

        try:
            if output_path.exists():
                output_path.unlink()

            if not RVC_ENABLED:
                normalize_audio_file(input_path, output_path)
                debug["skipped"] = True
                debug["reason"] = "RVC is disabled. Set RVC_ENABLED=true and RVC_MODEL_PATH to enable conversion."
                debug["output_size_bytes"] = output_path.stat().st_size
                return debug

            model_path = resolve_rvc_path(RVC_MODEL_PATH, must_exist=True)
            if not model_path:
                raise RVCConversionError(
                    "RVC_MODEL_PATH is not configured or the model file does not exist.",
                    debug
                )

            index_path = resolve_rvc_path(RVC_INDEX_PATH, must_exist=True)
            if RVC_INDEX_PATH and not index_path:
                debug["warnings"].append(f"RVC_INDEX_PATH was provided but not found: {RVC_INDEX_PATH}")

            command = self._build_command(input_path, converted_path, model_path, index_path)
            debug["command"] = command if isinstance(command, str) else [str(part) for part in command]

            if converted_path.exists():
                converted_path.unlink()

            result = subprocess.run(
                command,
                cwd=RVC_ROOT or None,
                capture_output=True,
                text=True,
                timeout=RVC_TIMEOUT_SECONDS,
                shell=isinstance(command, str)
            )

            debug["returncode"] = result.returncode
            debug["stdout_tail"] = tail_text(result.stdout)
            debug["stderr_tail"] = tail_text(result.stderr)

            if result.returncode != 0:
                raise RVCConversionError(
                    f"RVC conversion failed with exit code {result.returncode}.",
                    debug
                )

            if not converted_path.exists() or converted_path.stat().st_size < 1000:
                raise RVCConversionError(
                    f"RVC did not create the converted WAV: {converted_path}",
                    debug
                )

            normalize_audio_file(converted_path, output_path)

            debug["converted"] = True
            debug["converted_size_bytes"] = converted_path.stat().st_size
            debug["output_size_bytes"] = output_path.stat().st_size
            return debug

        except subprocess.TimeoutExpired as exc:
            debug["stdout_tail"] = tail_text(exc.stdout)
            debug["stderr_tail"] = tail_text(exc.stderr)
            raise RVCConversionError(
                f"RVC conversion timed out after {RVC_TIMEOUT_SECONDS} seconds.",
                debug
            ) from exc
        except RVCConversionError:
            raise
        except Exception as exc:
            raise RVCConversionError(str(exc), debug) from exc

    def _build_command(self, input_path, output_path, model_path, index_path):
        python_path = RVC_PYTHON_PATH or sys.executable
        infer_script = resolve_rvc_path(RVC_INFER_SCRIPT, must_exist=False)

        values = {
            "python": str(python_path),
            "script": str(infer_script) if infer_script else RVC_INFER_SCRIPT,
            "input": str(input_path),
            "output": str(output_path),
            "model": str(model_path),
            "index": str(index_path or ""),
            "pitch": str(RVC_PITCH),
            "f0_method": RVC_F0_METHOD,
            "method": RVC_F0_METHOD,
            "index_rate": RVC_INDEX_RATE,
            "protect": RVC_PROTECT,
            "filter_radius": str(RVC_FILTER_RADIUS),
            "resample_sr": str(RVC_RESAMPLE_SR),
            "device": RVC_DEVICE
        }

        if RVC_COMMAND_TEMPLATE:
            return split_command_template(RVC_COMMAND_TEMPLATE.format(**values))

        if infer_script and not infer_script.exists():
            raise FileNotFoundError(
                f"RVC infer script not found: {infer_script}. Set RVC_ROOT/RVC_INFER_SCRIPT or RVC_COMMAND_TEMPLATE."
            )

        if RVC_CLI_STYLE in {"rvc-webui", "webui"}:
            model_name = RVC_WEBUI_MODEL_NAME or model_path.name
            command = [
                str(python_path),
                str(infer_script),
                "--f0up_key", str(RVC_PITCH),
                "--input_path", str(input_path),
                "--index_path", str(index_path or ""),
                "--f0method", RVC_F0_METHOD,
                "--opt_path", str(output_path),
                "--model_name", model_name,
                "--index_rate", RVC_INDEX_RATE,
                "--filter_radius", str(RVC_FILTER_RADIUS),
                "--resample_sr", str(RVC_RESAMPLE_SR),
                "--protect", RVC_PROTECT
            ]

            if RVC_INCLUDE_DEVICE_ARG and RVC_DEVICE:
                command.extend(["--device", RVC_DEVICE])

            return command

        if RVC_CLI_STYLE in {"rvc-cli", "cli"}:
            command = [
                str(python_path),
                str(infer_script),
                "infer",
                "--pitch", str(RVC_PITCH),
                "--input_path", str(input_path),
                "--output_path", str(output_path),
                "--pth_path", str(model_path),
                "--f0_method", RVC_F0_METHOD,
                "--index_rate", RVC_INDEX_RATE,
                "--protect", RVC_PROTECT,
                "--filter_radius", str(RVC_FILTER_RADIUS),
                "--resample_sr", str(RVC_RESAMPLE_SR),
                "--export_format", "WAV"
            ]

            if index_path:
                command.extend(["--index_path", str(index_path)])

            return command

        command = [
            str(python_path),
            str(infer_script),
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--pth_path", str(model_path),
            "--pitch", str(RVC_PITCH),
            "--f0method", RVC_F0_METHOD,
            "--index_rate", RVC_INDEX_RATE,
            "--protect", RVC_PROTECT,
            "--filter_radius", str(RVC_FILTER_RADIUS),
            "--resample_sr", str(RVC_RESAMPLE_SR)
        ]

        if index_path:
            command.extend(["--index_path", str(index_path)])

        return command

RVC_CONVERTER = RVCVoiceConverter()

def synthesize_narration_voice(script, output_path, voice=None, speaker_wav=None):
    output_path = Path(output_path)

    if RVC_CLEANUP_TEMP_FILES:
        with tempfile.TemporaryDirectory(prefix="voice_", dir=output_path.parent) as temp_dir:
            return _synthesize_narration_voice_in_dir(
                script,
                output_path,
                Path(temp_dir),
                voice=voice,
                speaker_wav=speaker_wav
            )

    temp_dir = output_path.parent / f"voice_work_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    temp_dir.mkdir(exist_ok=True)

    debug = _synthesize_narration_voice_in_dir(
        script,
        output_path,
        temp_dir,
        voice=voice,
        speaker_wav=speaker_wav
    )
    debug["temp_workspace"] = str(temp_dir)
    return debug

def _synthesize_narration_voice_in_dir(script, output_path, temp_dir, voice=None, speaker_wav=None):
    temp_tts_path = temp_dir / "edge_tts_temp.wav"

    tts_debug = VOICE_GENERATOR.synthesize_to_file(
        script,
        temp_tts_path,
        voice=voice,
        speaker_wav=speaker_wav
    )

    try:
        rvc_debug = RVC_CONVERTER.convert_to_file(temp_tts_path, output_path, temp_dir)
    except RVCConversionError as exc:
        exc.debug["tts_debug"] = tts_debug
        raise

    return {
        "tts": tts_debug,
        "rvc": rvc_debug,
        "final_voice_path": str(output_path),
        "temp_tts_path": str(temp_tts_path),
        "cleanup_temp_files": RVC_CLEANUP_TEMP_FILES
    }

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "about", "be", "best", "by",
    "for", "from", "facts", "fact", "habit", "habits", "how", "in", "into",
    "is", "it", "of", "on", "or", "short", "story", "stories", "that",
    "the", "this", "to", "top", "truth", "video", "videos", "viral", "why",
    "with", "you", "your"
}

CATEGORY_RULES = {
    "ai": {
        "triggers": ["ai", "artificial intelligence", "chatgpt", "machine learning", "robot", "robots", "automation", "algorithm", "future tech"],
        "keywords": ["AI", "technology", "robot", "future", "automation", "computer code", "data center"],
        "queries": ["artificial intelligence robot", "futuristic technology", "computer code data center", "robotics lab"]
    },
    "business": {
        "triggers": ["business", "entrepreneur", "startup", "company", "ceo", "marketing", "sales", "productivity", "office"],
        "keywords": ["business", "startup", "office", "entrepreneur", "strategy", "meeting", "success"],
        "queries": ["startup office", "business meeting", "entrepreneur working", "city office"]
    },
    "motivation": {
        "triggers": ["motivation", "discipline", "mindset", "success", "goals", "focus", "grind", "self improvement", "confidence"],
        "keywords": ["motivation", "success", "discipline", "focus", "goals", "morning routine"],
        "queries": ["person running sunrise", "focused work", "morning routine", "success lifestyle"]
    },
    "fitness": {
        "triggers": ["fitness", "gym", "workout", "muscle", "weight loss", "health", "training", "exercise", "running", "bodybuilding"],
        "keywords": ["fitness", "gym", "workout", "training", "health", "running"],
        "queries": ["gym workout", "running training", "fitness lifestyle", "athlete training"]
    },
    "space": {
        "triggers": ["space", "planet", "planets", "mars", "moon", "nasa", "galaxy", "universe", "astronaut", "cosmos", "black hole"],
        "keywords": ["space", "galaxy", "planet", "astronaut", "universe", "stars"],
        "queries": ["space stars", "astronaut", "galaxy universe", "planet earth space"]
    },
    "horror": {
        "triggers": ["scary", "creepy", "horror", "dark", "disturbing", "haunted", "ghost", "nightmare", "mystery", "terrifying"],
        "keywords": ["dark", "mystery", "shadow", "abandoned", "night", "cinematic"],
        "queries": ["dark cinematic", "abandoned building", "foggy forest night", "scary shadows"]
    },
    "finance": {
        "triggers": ["finance", "money", "investing", "investment", "stocks", "crypto", "wealth", "rich", "billionaire", "millionaire", "bank"],
        "keywords": ["money", "finance", "wealth", "investing", "stocks", "success"],
        "queries": ["money finance", "stock market", "banking city", "wealth lifestyle"]
    },
    "luxury": {
        "triggers": ["luxury", "billionaire", "rich", "wealthy", "expensive", "mansion", "supercar", "yacht", "private jet", "designer"],
        "keywords": ["luxury", "money", "success", "rich", "mansion", "supercar"],
        "queries": ["luxury lifestyle", "supercar city", "mansion interior", "private jet luxury"]
    },
    "psychology": {
        "triggers": ["psychology", "brain", "mind", "mental", "behavior", "human", "emotions", "dopamine", "memory"],
        "keywords": ["psychology", "brain", "mind", "human behavior", "thinking", "memory"],
        "queries": ["human brain", "person thinking", "psychology mind", "abstract brain"]
    },
    "history": {
        "triggers": ["history", "ancient", "war", "empire", "roman", "egypt", "medieval", "king", "queen", "battle", "civilization"],
        "keywords": ["history", "ancient", "museum", "civilization", "monument", "archive"],
        "queries": ["ancient ruins", "history museum", "old city monument", "archive documents"]
    },
    "nature": {
        "triggers": ["nature", "animal", "animals", "ocean", "forest", "mountain", "wildlife", "earth", "sea", "storm", "weather"],
        "keywords": ["nature", "wildlife", "ocean", "forest", "mountain", "earth"],
        "queries": ["cinematic nature", "ocean waves", "forest sunlight", "mountain landscape"]
    },
    "crime": {
        "triggers": ["crime", "detective", "murder", "serial killer", "case", "police", "investigation", "mystery"],
        "keywords": ["crime", "detective", "police", "investigation", "mystery", "evidence"],
        "queries": ["detective investigation", "police lights", "crime scene", "dark city night"]
    }
}

GENERAL_FALLBACK_QUERIES = [
    "cinematic vertical b roll",
    "city night cinematic",
    "abstract technology background",
    "dramatic landscape",
    "slow motion cinematic"
]

def unique_list(values):
    seen = set()
    result = []

    for value in values:
        normalized = str(value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)

    return result

def tokenize_topic(topic):
    return re.findall(r"[a-zA-Z0-9]+", topic.lower())

def extract_search_keywords(topic):
    raw_tokens = tokenize_topic(topic)
    keywords = []

    for token in raw_tokens:
        if token in STOP_WORDS:
            continue
        if len(token) <= 2 and token not in {"ai", "vr", "ar"}:
            continue
        keywords.append("AI" if token == "ai" else token)

    return unique_list(keywords)

def detect_topic_categories(topic, base_keywords):
    topic_lc = topic.lower()
    topic_tokens = set(tokenize_topic(topic))
    keyword_tokens = {keyword.lower() for keyword in base_keywords}
    detected = []

    for category, config in CATEGORY_RULES.items():
        matches_category = False

        for trigger in config["triggers"]:
            trigger_lc = trigger.lower()
            if " " in trigger_lc:
                matches_category = trigger_lc in topic_lc
            else:
                matches_category = trigger_lc in topic_tokens or trigger_lc in keyword_tokens

            if matches_category:
                break

        if matches_category:
            detected.append(category)

    if not detected:
        detected.append("nature")

    return detected[:3]

def build_background_search_plan(topic):
    base_keywords = extract_search_keywords(topic)
    categories = detect_topic_categories(topic, base_keywords)
    expanded_keywords = list(base_keywords)

    for category in categories:
        expanded_keywords.extend(CATEGORY_RULES[category]["keywords"])

    expanded_keywords = unique_list(expanded_keywords)
    visual_keywords = [keyword for keyword in expanded_keywords if keyword.lower() not in STOP_WORDS]

    primary_query = " ".join(visual_keywords[:4]) or topic

    search_items = [
        {"query": topic, "source": "exact_topic", "rank": 0},
        {"query": primary_query, "source": "clean_keywords", "rank": 1}
    ]

    rank = 2

    if len(visual_keywords) >= 2:
        search_items.append(
            {
                "query": f"{visual_keywords[0]} {visual_keywords[1]} cinematic",
                "source": "cinematic_keywords",
                "rank": rank
            }
        )
        rank += 1

    for category in categories:
        for query in CATEGORY_RULES[category]["queries"]:
            search_items.append(
                {"query": query, "source": f"{category}_fallback", "rank": rank}
            )
            rank += 1

    for query in GENERAL_FALLBACK_QUERIES:
        search_items.append({"query": query, "source": "general_fallback", "rank": rank})
        rank += 1

    deduped_items = []
    seen_queries = set()

    for item in search_items:
        query_key = item["query"].strip().lower()
        if query_key and query_key not in seen_queries:
            seen_queries.add(query_key)
            deduped_items.append(item)

    return {
        "categories": categories,
        "keywords": visual_keywords,
        "search_items": deduped_items[:12]
    }

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def load_recent_clip_memory(output_dir):
    memory_path = output_dir / CLIP_MEMORY_FILE.name
    if not memory_path.exists():
        return []

    try:
        data = json.loads(memory_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = data.get("recent_clip_keys", [])

    if not isinstance(data, list):
        return []

    return [str(item) for item in data[-RECENT_CLIP_MEMORY_LIMIT:]]

def save_recent_clip_memory(output_dir, selected_candidates):
    memory_path = output_dir / CLIP_MEMORY_FILE.name
    recent_keys = load_recent_clip_memory(output_dir)

    for candidate in selected_candidates:
        key = candidate.get("video_id") or candidate.get("source_url")
        if key:
            recent_keys.append(str(key))

    recent_keys = unique_list(recent_keys)[-RECENT_CLIP_MEMORY_LIMIT:]
    memory_path.write_text(json.dumps({"recent_clip_keys": recent_keys}, indent=2), encoding="utf-8")

def score_video_file(video_file):
    width = safe_int(video_file.get("width"))
    height = safe_int(video_file.get("height"))
    fps = safe_int(video_file.get("fps"))
    quality = str(video_file.get("quality") or "").lower()
    score = 0

    if height > width:
        score += 35
    elif width > height:
        score -= 25

    if width >= 720 and height >= 1280:
        score += 35
    elif width >= 540 and height >= 960:
        score += 22
    elif width and height:
        score += 10

    if quality == "hd":
        score += 10

    if fps >= 24:
        score += 8

    return score

def choose_best_video_file(video):
    mp4_files = [
        item for item in video.get("video_files", [])
        if item.get("file_type") == "video/mp4" and item.get("link")
    ]

    if not mp4_files:
        return None

    return max(mp4_files, key=score_video_file)

def score_pexels_candidate(video, video_file, search_item, keywords, recent_clip_keys):
    width = safe_int(video_file.get("width") or video.get("width"))
    height = safe_int(video_file.get("height") or video.get("height"))
    duration = safe_int(video.get("duration"))
    fps = safe_int(video_file.get("fps"))
    page_url = str(video.get("url") or "").lower()
    source_url = str(video_file.get("link") or "")
    video_id = str(video.get("id") or source_url)

    score = 120 - (search_item["rank"] * 7)
    score += score_video_file(video_file)

    if height > width:
        score += 30

    if 4 <= duration <= 24:
        score += 25
    elif 25 <= duration <= 45:
        score += 12
    elif duration and duration > 60:
        score -= 12

    if fps >= 24:
        score += 8

    query_text = search_item["query"].lower()

    for keyword in keywords:
        keyword_lc = keyword.lower()
        if keyword_lc in query_text:
            score += 5
        if keyword_lc in page_url:
            score += 4

    if video_id in recent_clip_keys or source_url in recent_clip_keys:
        score -= 36

    score += random.uniform(0, 14)

    return score

def fetch_pexels_candidates(session, search_item, keywords, recent_clip_keys):
    page = 1 if search_item["rank"] <= 1 else random.randint(1, 3)
    data = None
    used_size = None

    for size in ["large", "medium", None]:
        params = {
            "query": search_item["query"],
            "per_page": PEXELS_SEARCH_RESULTS_PER_QUERY,
            "orientation": "portrait",
            "page": page
        }

        if size:
            params["size"] = size

        response = session.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_API_KEY},
            params=params,
            timeout=(10, 60)
        )
        response.raise_for_status()

        data = response.json()
        used_size = size or "any"

        if data.get("videos"):
            break

    candidates = []

    for video in data.get("videos", []):
        video_file = choose_best_video_file(video)

        if not video_file:
            continue

        score = score_pexels_candidate(video, video_file, search_item, keywords, recent_clip_keys)
        source_url = video_file["link"]

        candidates.append(
            {
                "video_id": str(video.get("id") or source_url),
                "source_url": source_url,
                "page_url": video.get("url"),
                "query": search_item["query"],
                "query_source": search_item["source"],
                "search_rank": search_item["rank"],
                "width": safe_int(video_file.get("width") or video.get("width")),
                "height": safe_int(video_file.get("height") or video.get("height")),
                "duration": safe_int(video.get("duration")),
                "score": round(score, 2)
            }
        )

    return candidates, page, used_size

def collect_ranked_pexels_candidates(search_items, keywords, output_dir, debug):
    if not PEXELS_API_KEY:
        debug["errors"].append("PEXELS_API_KEY not found. Using local visual fallback.")
        return []

    recent_clip_keys = load_recent_clip_memory(output_dir)
    candidates = []
    seen_keys = set()

    with requests.Session() as session:
        for search_item in search_items:
            try:
                fetched, page, used_size = fetch_pexels_candidates(
                    session,
                    search_item,
                    keywords,
                    recent_clip_keys
                )

                debug["search_attempts"].append(
                    {
                        "query": search_item["query"],
                        "source": search_item["source"],
                        "page": page,
                        "size": used_size,
                        "results": len(fetched)
                    }
                )

                for candidate in fetched:
                    key = candidate["video_id"] or candidate["source_url"]

                    if key in seen_keys or candidate["source_url"] in seen_keys:
                        continue

                    seen_keys.add(key)
                    seen_keys.add(candidate["source_url"])
                    candidates.append(candidate)

            except requests.RequestException as exc:
                debug["search_attempts"].append(
                    {
                        "query": search_item["query"],
                        "source": search_item["source"],
                        "results": 0,
                        "error": str(exc)
                    }
                )

            if len(candidates) >= MAX_BACKGROUND_CLIPS * 5 and len(debug["search_attempts"]) >= 4:
                break

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates

def download_candidate_clip(session, candidate, clip_path):
    temp_path = clip_path.with_suffix(".download")

    if temp_path.exists():
        temp_path.unlink()

    with session.get(candidate["source_url"], stream=True, timeout=(20, 240)) as response:
        response.raise_for_status()

        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    temp_path.replace(clip_path)

def get_local_fallback_clips(output_dir, excluded_paths):
    excluded = {str(path.resolve()) for path in excluded_paths if path.exists()}
    candidates = []

    for pattern in ["clip_*.mp4", "background.mp4", "merged_background.mp4", "video_*.mp4", "final_video.mp4"]:
        candidates.extend(output_dir.glob(pattern))

    unique_paths = []
    seen = set()

    for path in candidates:
        resolved = str(path.resolve())

        if resolved in seen or resolved in excluded:
            continue

        if path.is_file() and path.stat().st_size > 10_000:
            seen.add(resolved)
            unique_paths.append(path)

    random.shuffle(unique_paths)
    return unique_paths

def add_local_fallback_clips(output_dir, clip_paths, debug):
    local_clips = get_local_fallback_clips(output_dir, clip_paths)

    for source_path in local_clips:
        if len(clip_paths) >= MIN_BACKGROUND_CLIPS:
            break

        destination = output_dir / f"clip_{len(clip_paths)}.mp4"

        try:
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
                clip_paths.append(destination)
            else:
                clip_paths.append(source_path)

            debug["local_fallback_used"] = True
            debug["selected_clip_urls"].append(f"local://{source_path.name}")
        except OSError as exc:
            debug["download_errors"].append({"source": str(source_path), "error": str(exc)})

def create_synthetic_background_clip(output_dir, debug):
    clip_path = output_dir / f"clip_{len(debug['selected_clip_urls'])}.mp4"

    if not FFMPEG_PATH.exists():
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_PATH}")

    command = [
        str(FFMPEG_PATH),
        "-y",
        "-f", "lavfi",
        "-i", "color=c=0x05070b:s=1080x1920:r=30:d=18",
        "-vf", "noise=alls=22:allf=t+u,eq=contrast=1.25:saturation=1.15",
        "-t", "18",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        str(clip_path)
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(result.stderr)

    debug["synthetic_fallback_used"] = True
    debug["selected_clip_urls"].append("local://synthetic-topic-background")

    return clip_path

def download_background_videos(topic, output_dir):
    plan = build_background_search_plan(topic)

    debug = {
        "detected_categories": plan["categories"],
        "used_keywords": plan["keywords"],
        "search_plan": [item["query"] for item in plan["search_items"]],
        "search_attempts": [],
        "fallback_keyword_used": None,
        "selected_clip_urls": [],
        "selected_clip_pages": [],
        "selected_clip_ids": [],
        "download_errors": [],
        "local_fallback_used": False,
        "synthetic_fallback_used": False,
        "errors": []
    }

    candidates = collect_ranked_pexels_candidates(
        plan["search_items"],
        plan["keywords"],
        output_dir,
        debug
    )

    clip_paths = []
    selected_candidates = []
    used_queries = []
    target_clip_count = random.randint(MIN_BACKGROUND_CLIPS, MAX_BACKGROUND_CLIPS)

    with requests.Session() as session:
        for candidate in candidates:
            if len(clip_paths) >= target_clip_count:
                break

            clip_path = output_dir / f"clip_{len(clip_paths)}.mp4"

            try:
                download_candidate_clip(session, candidate, clip_path)
                clip_paths.append(clip_path)
                selected_candidates.append(candidate)
                used_queries.append(candidate["query"])
                debug["selected_clip_urls"].append(candidate["source_url"])
                debug["selected_clip_pages"].append(candidate.get("page_url"))
                debug["selected_clip_ids"].append(candidate["video_id"])
            except (requests.RequestException, OSError) as exc:
                debug["download_errors"].append({"url": candidate["source_url"], "error": str(exc)})

    primary_query = plan["search_items"][0]["query"] if plan["search_items"] else topic
    debug["fallback_keyword_used"] = next(
        (query for query in used_queries if query.lower() != primary_query.lower()),
        None
    )

    if len(clip_paths) < MIN_BACKGROUND_CLIPS:
        add_local_fallback_clips(output_dir, clip_paths, debug)

    while len(clip_paths) < MIN_BACKGROUND_CLIPS:
        try:
            clip_paths.append(create_synthetic_background_clip(output_dir, debug))
        except Exception as exc:
            debug["errors"].append(f"Synthetic background fallback failed: {exc}")
            if not clip_paths:
                raise
            break

    if not debug["fallback_keyword_used"] and debug["local_fallback_used"]:
        debug["fallback_keyword_used"] = "local_visual_fallback"

    if not debug["fallback_keyword_used"] and debug["synthetic_fallback_used"]:
        debug["fallback_keyword_used"] = "synthetic_topic_background"

    save_recent_clip_memory(output_dir, selected_candidates)

    return clip_paths[:MAX_BACKGROUND_CLIPS], debug

def build_merge_background_command(clip_paths, merged_background_path):
    command = [str(FFMPEG_PATH), "-y"]

    for clip_path in clip_paths:
        command.extend(["-i", str(clip_path)])

    filter_parts = []
    concat_inputs = []

    for index, _ in enumerate(clip_paths):
        label = "vout" if len(clip_paths) == 1 else f"v{index}"
        filter_parts.append(
            f"[{index}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,setsar=1,setpts=PTS-STARTPTS[{label}]"
        )

        if len(clip_paths) > 1:
            concat_inputs.append(f"[{label}]")

    if len(clip_paths) > 1:
        filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(clip_paths)}:v=1:a=0[vout]")

    command.extend(
        [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[vout]",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            str(merged_background_path)
        ]
    )

    return command

@app.post("/generate")
def generate(data: GenerateRequest):
    background_debug = None
    voice_debug = None
    rvc_debug = None

    try:
        topic = data.topic.strip() or "AI facts"

        prompt = f"""
Write a VERY natural TikTok narration about:
{topic}

Rules:
- Sound like a real human talking
- Conversational
- Emotional pacing
- Add suspense
- Add curiosity
- Use short dramatic sentences
- Use contractions like it's, don't, you're
- Avoid sounding educational
- Avoid sounding like Wikipedia
- Speak like a viral TikTok creator
- Under 60 seconds
- No markdown
- No bullet points
- Return narration only
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        script = response.json()["response"].strip()

        output_dir = OUTPUT_DIR
        output_dir.mkdir(exist_ok=True)

        script_path = output_dir / "script.txt"
        voice_path = output_dir / "voice.wav"
        subtitle_path = output_dir / "subtitles.srt"
        final_video_path = output_dir / "final_video.mp4"
        merged_background_path = output_dir / "merged_background.mp4"
        concat_file_path = output_dir / "concat.txt"

        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        history_video_path = output_dir / f"video_{timestamp}_{slugify(topic)}.mp4"
        history_metadata_path = history_video_path.with_suffix(".json")

        script_path.write_text(script, encoding="utf-8")

        clip_paths, background_debug = download_background_videos(topic, output_dir)

        try:
            voice_debug = synthesize_narration_voice(
                script,
                voice_path,
                voice=data.voice,
                speaker_wav=data.speaker_wav
            )
            rvc_debug = voice_debug.get("rvc")
        except RVCConversionError as exc:
            rvc_debug = exc.debug
            voice_debug = {"rvc": rvc_debug}
            return {
                "success": False,
                "step": "rvc",
                "error": str(exc),
                "background_debug": background_debug,
                "voice_debug": voice_debug,
                "rvc_debug": rvc_debug
            }
        except Exception as exc:
            return {
                "success": False,
                "step": "tts",
                "error": str(exc),
                "background_debug": background_debug,
                "voice_debug": voice_debug,
                "rvc_debug": rvc_debug
            }

        if not voice_path.exists():
            return {
                "success": False,
                "step": "voice_check",
                "error": "voice.wav not created",
                "voice_path": str(voice_path),
                "background_debug": background_debug,
                "voice_debug": voice_debug,
                "rvc_debug": rvc_debug
            }

        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = whisper_model.transcribe(str(voice_path))
        segments = list(segments)

        if len(segments) == 0:
            return {
                "success": False,
                "step": "whisper",
                "error": "No subtitle segments generated",
                "background_debug": background_debug,
                "voice_debug": voice_debug,
                "rvc_debug": rvc_debug
            }

        with open(subtitle_path, "w", encoding="utf-8") as srt_file:
            for index, segment in enumerate(segments, start=1):
                text = segment.text.strip()
                srt_file.write(f"{index}\n")
                srt_file.write(f"{format_time(segment.start)} --> {format_time(segment.end)}\n")
                srt_file.write(f"{text}\n\n")

        with open(concat_file_path, "w", encoding="utf-8") as file:
            for clip in clip_paths:
                clip_safe = str(clip).replace("\\", "/")
                file.write(f"file '{clip_safe}'\n")

        merge_command = build_merge_background_command(clip_paths, merged_background_path)

        merge_result = subprocess.run(merge_command, capture_output=True, text=True)

        if merge_result.returncode != 0:
            return {
                "success": False,
                "step": "merge_clips",
                "error": merge_result.stderr,
                "command": merge_command,
                "background_debug": background_debug,
                "voice_debug": voice_debug,
                "rvc_debug": rvc_debug
            }

        subtitle_file = str(subtitle_path).replace("\\", "/")
        subtitle_file = subtitle_file.replace("C:", "C\\:")

        ffmpeg_command = [
            str(FFMPEG_PATH),
            "-y",
            "-stream_loop", "-1",
            "-i", str(merged_background_path),
            "-i", str(voice_path),
            "-vf",
            f"scale=1080:1920,setsar=1,subtitles='{subtitle_file}':force_style='FontName=Arial,Fontsize=18,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=220,Bold=1'",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-preset", "ultrafast",
            str(final_video_path)
        ]

        ffmpeg_result = subprocess.run(ffmpeg_command, capture_output=True, text=True)

        if ffmpeg_result.returncode != 0:
            return {
                "success": False,
                "step": "ffmpeg_final",
                "error": ffmpeg_result.stderr,
                "command": ffmpeg_command,
                "background_debug": background_debug,
                "voice_debug": voice_debug,
                "rvc_debug": rvc_debug
            }

        shutil.copy2(final_video_path, history_video_path)

        history_metadata_path.write_text(
            json.dumps(
                {
                    "filename": history_video_path.name,
                    "topic": topic,
                    "created_at": created_at,
                    "script": script,
                    "background_debug": background_debug,
                    "voice_debug": voice_debug,
                    "rvc_debug": rvc_debug
                },
                indent=2
            ),
            encoding="utf-8"
        )

        video_response = build_video_response(history_video_path)

        return {
            "success": True,
            "topic": topic,
            "filename": video_response["filename"],
            "video_url": video_response["url"],
            "download_url": video_response["download_url"],
            "script_path": str(script_path),
            "clip_paths": [str(path) for path in clip_paths],
            "used_keywords": background_debug["used_keywords"],
            "fallback_keyword_used": background_debug["fallback_keyword_used"],
            "selected_clip_urls": background_debug["selected_clip_urls"],
            "background_debug": background_debug,
            "voice_debug": voice_debug,
            "rvc_debug": rvc_debug,
            "merged_background_path": str(merged_background_path),
            "voice_path": str(voice_path),
            "subtitle_path": str(subtitle_path),
            "final_video_path": str(final_video_path),
            "history_video_path": str(history_video_path),
            "script": script,
            "videos": get_video_history()
        }

    except Exception as exc:
        response = {
            "success": False,
            "step": "python_exception",
            "error": str(exc),
            "traceback": traceback.format_exc()
        }

        if background_debug:
            response["background_debug"] = background_debug

        if voice_debug:
            response["voice_debug"] = voice_debug
        if rvc_debug:
            response["rvc_debug"] = rvc_debug

        return response
