# Subtitle Generator

<p align="center">
  <img src="https://github.com/user-attachments/assets/79e5c471-ca84-4b3f-88f8-56945f19669e" width="950" alt="Subtitle Generator screenshot" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img alt="PyQt6" src="https://img.shields.io/badge/GUI-PyQt6-41cd52?logo=qt&logoColor=white" />
  <img alt="WhisperX" src="https://img.shields.io/badge/ASR-WhisperX-black?logo=openai&logoColor=white" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1%20%7C%2012.4-76b900?logo=nvidia&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-GPLv3-blue" />
</p>

A desktop application for generating subtitles (SRT) and plain-text transcriptions from audio and video files, powered by **WhisperX** with word-level timestamp alignment.

---

## Features

- **Accurate transcription** — WhisperX with language-specific word-level alignment
- **Interactive waveform viewer** — zoom, pan, playback and region selection before processing
- **Region-based processing** — include or exclude specific time ranges
- **Voice separation** — optional Demucs preprocessing to isolate vocals from background music
- **Neural voice enhancement** — optional NVIDIA RE-USE speech enhancement before transcription
- **Audio effects chain** — optional Noise Gate, High-Pass Filter, Compressor and Gain via Pedalboard
- **Whisper input preview** — inspect and play back the exact audio sent to WhisperX after all processing
- **Check Audio** — run the full processing pipeline and preview the result before transcription starts
- **Microphone recording** — record directly in the app and transcribe immediately
- **Flexible output** — SRT with configurable word patterns, or plain-text with paragraph splitting
- **GPU acceleration** — automatic CUDA detection with float16/float32/int8 compute type selection

---

## Supported formats

| Type  | Extensions |
|-------|-----------|
| Audio | `.mp3` · `.wav` · `.flac` · `.m4a` · `.ogg` · `.wma` |
| Video | `.mp4` · `.avi` · `.mkv` · `.mov` · `.flv` · `.wmv` · `.webm` · `.m4v` |

Video files require **FFmpeg** installed and available in `PATH`. Audio is extracted automatically before processing.

---

## Output formats

| Format | Description |
|--------|-------------|
| SRT | Subtitle file with timestamps. Words are grouped into blocks by a configurable word-count pattern and split on natural pauses. |
| TXT | Plain-text transcription. Segments separated by pauses longer than 2 s are written as separate paragraphs. |

---

## Transcription models

The application uses **WhisperX** — a Whisper model for transcription followed by a language-specific alignment model for precise word-level timestamps.

| Model | Notes |
|-------|-------|
| `tiny` / `tiny.en` | Fastest, lowest accuracy |
| `base` / `base.en` | |
| `small` / `small.en` | |
| `medium` / `medium.en` | |
| `large-v3` | Best accuracy · ~3–5 GB · GPU recommended · **Default** |

`.en` variants are English-only and faster for English audio.

Models are downloaded from Hugging Face Hub on first use and cached locally in `./models/whisperx/<model_name>/`. The app checks for a complete local copy before attempting a download. Alignment models are stored in `./models/whisperx/align/<language_code>/`.

**Compute type — selected automatically:**

| Device | GPU compute capability | Compute type |
|--------|------------------------|--------------|
| CUDA   | ≥ sm_70 (Volta / RTX+) | `float16` |
| CUDA   | < sm_70                | `float32` |
| CPU    | —                      | `int8` |

> **Windows + CUDA:** `float16` inference requires cuDNN DLLs. These are provided by the `nvidia-cudnn-cu12` pip package, installed automatically by `install.bat` in GPU mode. DLL paths are registered at startup via `os.environ["PATH"]` — no manual configuration needed.

---

## Subtitle parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Word pattern | `3,4` | positive integers | Alternating max word counts per subtitle block (`3,4` → 3 words, 4 words, 3 words, …) |
| Min pause for split | 0.6 s | 0.1 – 3.0 s | Gap between words that forces a subtitle boundary |
| Min subtitle duration | 1.0 s | 0.5 – 5.0 s | Subtitles shorter than this are merged with the next |
| Max line length | 42 chars | 20 – 100 | Maximum characters per line; text wraps to a second line (film standard = 42) |
| Batch size | 16 | 1 – 64 | WhisperX transcription batch size — higher = faster, more VRAM |
| Language | `en` | ISO 639-1 code or empty | Leave empty for automatic language detection |

---

## Model settings

| Setting | Options | Notes |
|---------|---------|-------|
| Device | CPU / GPU (CUDA) | GPU selected automatically if CUDA is available; GPU name shown |

---

## Audio Processing

The **Audio Processing** panel provides an optional pre-transcription processing chain. Steps are applied in this order: Voice Separation → RE-USE Enhancement → Audio Effects.

### Voice Separation (Demucs)

Extracts the vocal track using Demucs (`--two-stems=vocals`) before transcription — useful for files with background music or noise.

- Significantly increases processing time (2–5 min per file depending on length)
- Output vocals are saved as MP3 (320 kbps) to a temp directory and cleaned up after processing
- Files with non-ASCII characters in their names are copied to a safe ASCII path before Demucs runs

### Neural Voice Enhancement (RE-USE)

Applies [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE) — a neural speech enhancement model — to the audio before transcription. Effective on degraded or noisy recordings where Demucs alone is insufficient.

- Model is downloaded automatically from Hugging Face on first use and stored in `./models/reuse/`
- Enhanced output is permanently saved to `./RE-USE_outputs/` (indexed, never overwritten)
- Applied **after** Voice Separation (if enabled) and **before** Audio Effects

**Bandwidth Extension (BWE)** — optionally upsamples the enhanced audio to a higher sample rate. 16 kHz is recommended for WhisperX. Available only when RE-USE is enabled.

**Chunking** — splits long files into segments before enhancement, then crossfade-joins the results. Useful when RE-USE runs out of memory on long recordings.

| Mode | Description |
|------|-------------|
| Fixed | Splits every N seconds exactly (hard cuts) |
| Smart | Finds the quietest point within each N-second window to minimise mid-word cuts |

> **Windows note:** RE-USE depends on `mamba_ssm`. A pure-PyTorch shim (`mamba_ssm_shim.py`) is installed automatically by `install.bat` — no CUDA Toolkit or MSVC compiler required.

### Audio Effects (Pedalboard)

A lightweight effects chain applied to the final processed audio before it is sent to WhisperX. Each effect can be enabled independently.

| Effect | Parameters | Purpose |
|--------|-----------|---------|
| Noise Gate | Threshold (dB), Release (ms) | Silences audio below the threshold — removes background hiss between words |
| High-Pass Filter | Cutoff (Hz) | Removes low-frequency rumble and microphone vibration |
| Compressor | Threshold (dB), Ratio | Reduces dynamic range — brings quiet speech to a more consistent level |
| Gain | Gain (dB) | Adjusts overall volume |

### Check Audio

The **🎧 Check Audio** button runs the full processing pipeline (region selection → Voice Separation → RE-USE → Audio Effects) without starting transcription. The processed audio is loaded into the **Whisper Input Preview** waveform so you can listen to exactly what WhisperX will receive before committing to the full run.

The button is enabled only when at least one audio processing option is active and its settings have changed since the last check. The **Generate** and **Export as Text** buttons appear only after a successful check.

---

## Waveform viewer

The built-in waveform panel supports:

- **Zoom** — scroll wheel (up to 300×), anchored to cursor position
- **Pan** — left drag
- **Seek** — left click (no drag)
- **Scrollbar** — appears below the waveform when zoom > 1×
- **Playback** — Play / Pause / Stop / Reset buttons with a playback cursor indicator

Audio is loaded at 16 000 Hz mono. Stereo files are converted to mono; files with a different sample rate are resampled via `librosa`.

### Whisper Input Preview

A second, read-only waveform viewer that shows the processed audio after the full pipeline. Available after clicking **🎧 Check Audio** or once generation starts. Useful for verifying that voice separation, enhancement, and effects produced the expected result.

---

## Region selection

After loading a file, you can select time regions to limit what gets transcribed.

| Type | Shortcut | Color | Effect |
|------|----------|-------|--------|
| INCLUDE | `Ctrl + Left drag` | Purple | Only selected regions are transcribed; audio is concatenated before sending to WhisperX |
| EXCLUDE | `Ctrl + Right drag` | Red | Selected regions are removed (TXT) or muted/zeroed-out (SRT) before transcription |

If both types are present, **INCLUDE takes priority** and EXCLUDE is ignored.

**Erasing regions:** the opposite mouse button acts as an eraser — `Ctrl + Right drag` erases INCLUDE regions, `Ctrl + Left drag` erases EXCLUDE regions. Adjacent or overlapping regions of the same type are merged automatically (tolerance: 0.3 s).

> **SRT vs TXT with regions:**
> In TXT mode, INCLUDE regions are extracted and concatenated — timestamps reflect the concatenated audio, not the original file.
> In SRT mode, EXCLUDE regions are muted rather than removed, so SRT timestamps stay aligned with the original file.

---

## Microphone recording

Built-in live recording backed by `sounddevice` + `soundfile`:

- Recorded at 16 000 Hz mono, saved as WAV to the system temp directory
- VU meter shown during recording
- Pause / Resume supported
- After stopping, the recording is loaded into the waveform viewer automatically and set as the input file

---

## Installation

### Windows — recommended (installer)

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) — check *"Add Python to PATH"*.

2. Install **FFmpeg** (required for video files):
   ```bat
   winget install FFmpeg
   ```
   Or download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add to PATH manually.

3. Clone the repository:
   ```bat
   git clone https://github.com/Mubumbutu/Subtitle-Generator.git
   cd Subtitle-Generator
   ```

4. Run the installer:
   ```bat
   install.bat
   ```
   The installer detects your NVIDIA GPU and driver version, asks whether to install CPU or GPU mode, creates a `venv`, installs all dependencies, and — for GPU mode — installs `nvidia-cudnn-cu12` (~720 MB) required for `float16` CUDA inference. It also installs the `mamba_ssm` pure-PyTorch shim required by the RE-USE enhancement module.

---

### Windows — manual

```bat
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

:: CPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: CUDA 12.4 (driver >= 550):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

:: CUDA 12.1 (driver >= 525):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: CUDA 11.8 (driver >= 450):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

:: GPU only — required for float16 inference:
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

:: mamba_ssm shim — required for RE-USE on Windows (no CUDA Toolkit needed):
mkdir venv\Lib\site-packages\mamba_ssm
copy mamba_ssm_shim.py venv\Lib\site-packages\mamba_ssm\__init__.py
```

Check your driver version with `nvidia-smi`.

---

### Linux

```bash
sudo apt install ffmpeg   # Debian / Ubuntu
# or: sudo dnf install ffmpeg

git clone https://github.com/Mubumbutu/Subtitle-Generator.git
cd Subtitle-Generator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# CPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU — CUDA 12.4 (driver >= 550):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# mamba_ssm (real CUDA extension, requires CUDA Toolkit):
pip install mamba-ssm
# Or use the pure-PyTorch shim:
# mkdir -p venv/lib/python3.x/site-packages/mamba_ssm
# cp mamba_ssm_shim.py venv/lib/python3.x/site-packages/mamba_ssm/__init__.py
```

> On Linux, cuDNN is typically available system-wide or bundled with PyTorch. `nvidia-cudnn-cu12` is a Windows-specific requirement.

---

### macOS

```bash
brew install ffmpeg

git clone https://github.com/Mubumbutu/Subtitle-Generator.git
cd Subtitle-Generator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install torch torchaudio
```

> CUDA is not available on macOS. Transcription runs on CPU only. Apple Silicon (M1/M2/M3) MPS acceleration is not currently configured.

---

## Running

```bash
# Activate the virtual environment first:
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

python subtitle_generator.py
```

On Windows you can also double-click **`start.bat`** — no terminal window needed.

---

## Workflow

```
Load file  →  (Optional) Select regions  →  Configure model & audio processing
         →  (Optional) Check Audio  →  Generate  →  Save
```

1. **Load a file** — drag & drop onto the window, use `📁 Browse...`, or record directly with `🎤 Start Record`.
2. **Waveform** — inspect the audio, play it back, zoom with the scroll wheel.
3. **Select regions** *(optional)* — `Ctrl + Left drag` to include, `Ctrl + Right drag` to exclude. Without any selection the entire file is processed.
4. **Model settings** — choose device (CPU / GPU), model size, language, and batch size.
5. **Audio Processing** *(optional)* — enable Voice Separation, RE-USE Enhancement, and/or Audio Effects as needed.
6. **Check Audio** *(optional, visible when audio processing is active)* — preview the processed audio before transcription.
7. **Advanced settings** — adjust word pattern, pause threshold, subtitle duration, and line length.
8. **Generate** — click `🚀 Generate Subtitles (.srt)` or `📄 Export as Text (.txt)`.
9. Output is saved next to the input file with the same base name. A custom path can be set in the save dialog.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | |
| FFmpeg | Required for video files; must be in `PATH` |
| torch + torchaudio | Installed separately with the correct CUDA/CPU variant |
| nvidia-cudnn-cu12 + nvidia-cublas-cu12 | GPU mode only — installed automatically by `install.bat` |
| mamba_ssm | Required by RE-USE; a pure-PyTorch shim is installed automatically by `install.bat` on Windows |

Python packages (see `requirements.txt`):

```
whisperx
transformers>=4.48.0,<5.0.0
sounddevice · soundfile · numpy · librosa · resampy
pysrt · PyQt6 · demucs · pedalboard
```

---

## Notes

- **First run downloads models** — `large-v3` is approximately 3–5 GB. Ensure a stable connection or pre-download by running a transcription once with the target model selected.
- **RE-USE model** — downloaded automatically on first use from Hugging Face (`nvidia/RE-USE`). Enhanced files are permanently saved to `./RE-USE_outputs/` and are never overwritten.
- **VRAM usage** — `large-v3` on GPU requires ~6–8 GB VRAM. RE-USE adds additional load. Reduce batch size or enable RE-USE chunking if you encounter out-of-memory errors.
- **Word pattern** — `3,4` produces subtitles alternating between 3 and 4 words. A long natural pause always overrides the pattern and forces a split regardless of word count.
- **transformers pin** — `transformers<5.0.0` is required because transformers 5.x requires `torch>=2.6`, which is not yet compatible with current CUDA builds shipped with torch 2.5.x.
- **mamba_ssm on Windows** — the real `mamba-ssm` package requires a CUDA Toolkit and MSVC compiler to build its C++ extension. The bundled `mamba_ssm_shim.py` provides a pure-PyTorch fallback that works on any Windows machine without additional build tools. It is slower than the real CUDA extension but produces correct output on both CPU and GPU.

---

## License

[GNU General Public License v3.0](LICENSE)
