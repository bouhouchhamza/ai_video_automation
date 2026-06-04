# RVC Voice Conversion Setup

The voice pipeline is now:

```text
script -> Edge TTS temp WAV -> RVC conversion -> output/voice.wav -> subtitles -> FFmpeg
```

`output/voice.wav` stays the final audio file used by Faster Whisper and FFmpeg.

## 1. Install A Local RVC Fork

Use a Windows RVC installation that exposes a CLI inference script. Two common
options are:

- RVC WebUI forks with `tools/infer_cli.py`
- RVC CLI forks with commands such as `rvc_cli.py infer`

Install the fork in its own folder and virtual environment. Keep that environment
separate from this FastAPI app if the dependency versions conflict.

## 2. Add A Voice Model

An RVC voice usually has:

- `.pth` model file
- optional `.index` file for better similarity

Example:

```text
C:\rvc\models\storyteller\storyteller.pth
C:\rvc\models\storyteller\added_IVF1234_Flat_nprobe_1_storyteller.index
```

## 3. Configure `.env`

For RVC CLI-style `rvc_cli.py infer`:

```env
RVC_ENABLED=true
RVC_CLI_STYLE=rvc-cli
RVC_ROOT=C:\path\to\rvc-cli
RVC_PYTHON_PATH=C:\path\to\rvc-cli\.venv\Scripts\python.exe
RVC_INFER_SCRIPT=rvc_cli.py
RVC_MODEL_PATH=C:\rvc\models\storyteller\storyteller.pth
RVC_INDEX_PATH=C:\rvc\models\storyteller\added_IVF1234_Flat_nprobe_1_storyteller.index
RVC_F0_METHOD=rmvpe
RVC_PITCH=0
RVC_INDEX_RATE=0.75
RVC_PROTECT=0.33
RVC_TIMEOUT_SECONDS=600
```

For RVC WebUI-style `tools/infer_cli.py`:

```env
RVC_ENABLED=true
RVC_CLI_STYLE=rvc-webui
RVC_ROOT=C:\path\to\Retrieval-based-Voice-Conversion-WebUI
RVC_PYTHON_PATH=C:\path\to\Retrieval-based-Voice-Conversion-WebUI\.venv\Scripts\python.exe
RVC_INFER_SCRIPT=tools\infer_cli.py
RVC_WEBUI_MODEL_NAME=storyteller.pth
RVC_MODEL_PATH=C:\path\to\Retrieval-based-Voice-Conversion-WebUI\assets\weights\storyteller.pth
RVC_INDEX_PATH=C:\rvc\models\storyteller\added_IVF1234_Flat_nprobe_1_storyteller.index
```

If your RVC fork uses a different CLI, use `RVC_COMMAND_TEMPLATE`:

```env
RVC_COMMAND_TEMPLATE="{python}" "{script}" infer --pitch {pitch} --input_path "{input}" --output_path "{output}" --pth_path "{model}" --index_path "{index}"
```

Available template variables:

```text
{python} {script} {input} {output} {model} {index}
{pitch} {f0_method} {method} {index_rate} {protect}
{filter_radius} {resample_sr} {device}
```

## 4. Recommended Settings

For TikTok narration:

```env
TTS_ENGINE=edge
TTS_EDGE_VOICE=en-US-AndrewNeural
TTS_RATE=-20%
RVC_F0_METHOD=rmvpe
RVC_INDEX_RATE=0.75
RVC_PROTECT=0.33
```

Use `RVC_PITCH=0` for same-gender conversion. Try `-12` or `+12` only if the
target voice needs a large pitch shift.

## 5. Debugging

Every `/generate` response now includes:

```json
{
  "rvc_debug": {
    "enabled": true,
    "converted": true,
    "model_path": "...",
    "index_path": "...",
    "command": ["..."],
    "stdout_tail": "...",
    "stderr_tail": "..."
  }
}
```

Temporary Edge/RVC files are cleaned automatically by default. To keep them while
debugging:

```env
RVC_CLEANUP_TEMP_FILES=false
```
