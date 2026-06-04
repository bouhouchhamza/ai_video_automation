# Realistic Local TTS Setup

This project now uses XTTS v2 through the Coqui TTS Python API instead of Piper.
The backend still writes narration to `output/voice.wav`, so Faster Whisper and
FFmpeg continue to work with the existing pipeline.

## Recommended Windows Environment

Use a fresh Python 3.11 or 3.12 virtual environment for XTTS. Avoid Python 3.14
for now because many audio and ML wheels lag behind the newest Python releases.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## CPU Install

```powershell
python -m pip install -r requirements.txt
```

## NVIDIA GPU Install

Install the PyTorch build that matches your GPU/driver from the official PyTorch
selector, then install the rest of the requirements:

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

If your GPU command differs, use the selector output from:

```text
https://pytorch.org/get-started/locally/
```

## First Model Download

XTTS v2 downloads on first use. To pre-download it before running the server:

```powershell
$env:COQUI_TOS_AGREED='1'
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

Review the XTTS v2 model license before setting `COQUI_TOS_AGREED`.

## Voice Configuration

Default `.env` values:

```env
TTS_ENGINE=xtts
TTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2
TTS_LANGUAGE=en
TTS_DEVICE=auto
TTS_VOICE=Craig Gutsy
TTS_SENTENCE_PACING=true
TTS_CPU_THREADS=4
TTS_SAMPLE_RATE=44100
TTS_MIN_PAUSE_SECONDS=0.28
TTS_MAX_PAUSE_SECONDS=0.85
```

For voice cloning, add a clean 6 to 30 second English WAV reference:

```env
TTS_SPEAKER_WAV=voices/storyteller.wav
```

You can also send a per-request voice:

```json
{
  "topic": "Scary AI facts",
  "voice": "Craig Gutsy",
  "speaker_wav": "voices/storyteller.wav"
}
```

`speaker_wav` wins over `voice` when both are provided.

## Run

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8010 --reload
```

The first generation will be slower because XTTS loads the model and may
download weights. Later generations reuse the loaded model in memory.
