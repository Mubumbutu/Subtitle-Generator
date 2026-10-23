# Subtitle Generator
<p align="center">
</p>


<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-lightgrey" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1%20%7C%2012.4-76b900?logo=nvidia&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/status-stable-brightgreen" />
  <img alt="License" src="https://img.shields.io/badge/License-GPLv3-blue" />
</p>

A desktop application for generating subtitles (SRT) and plain-text transcriptions from audio and video files.

---

<img width="1051" height="796" alt="01" src="https://github.com/user-attachments/assets/430c5aa6-f69d-4f31-86e1-49bce444128c" />

---

## Changelog

<details>
<summary><b>Version history</b> (click to expand)</summary>

<br>

### [1.0.0] - 2026-10-23
**Added**
- Initial public release version 1.0.0.

</details>

---

## Supported formats

| Type  | Extensions |
|-------|-----------|
| Audio | `.mp3` · `.wav` · `.flac` · `.m4a` · `.ogg` · `.wma` |
| Video | `.mp4` · `.avi` · `.mkv` · `.mov` · `.flv` · `.wmv` · `.webm` · `.m4v` |

Video files require **FFmpeg** installed and available in `PATH`. Audio is extracted automatically (16 kHz mono) before processing.

---

## Transcription models

| Model |
|-------|
| **WhisperX** |
| **MOSS-Transcribe-Diarize** |
| **Parakeet-TDT-0.6B-v3** |

### Model-specific settings

Shown under **⚙️ Model Settings** (hover over a label for a detailed tooltip).

**WhisperX**

| Setting | Default | Description |
|---------|---------|-------------|
| Model | `large-v3` | `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v3`. `.en` variants are English-only. |
| Language | `en` | ISO code (`en`, `pl`, `de`, …). Leave empty for auto-detection. |
| Batch size | 16 | 1 – 64. Higher = faster, more VRAM. |

Compute type is selected automatically: `float16` on CUDA GPUs with compute capability ≥ 7, `float32` on older GPUs, `int8` on CPU. After transcription, a language-specific alignment model produces word timestamps; if alignment fails, the app continues with unaligned timestamps.

**MOSS-Transcribe-Diarize**

| Setting | Default | Description |
|---------|---------|-------------|
| MOSS Model | `OpenMOSS-Team/MOSS-Transcribe-Diarize` |
| Max New Tokens | 32768 | 512 – 131072. Too low a value truncates the transcript — raise it for long or multi-speaker recordings. |
| Hotwords | empty | Comma-separated names / domain terms the model should pay attention to. |

**Parakeet-TDT-0.6B-v3**

| Setting | Default | Description |
|---------|---------|-------------|
| Parakeet Model | `nvidia/parakeet-tdt-0.6b-v3` |
| Batch size | 8 | 1 – 64. Higher = faster, more VRAM. |

---

## Subtitle parameters

Under **🔧 Advanced Settings**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Subtitle Type | Standard | Standard / Voice-over (TTS) | See below |
| Word pattern | `3,4` | positive integers | Controls the maximum words per subtitle (largest number × 4, e.g. `3,4` → up to 16 words). Subtitles usually break earlier at sentence ends, commas or pauses. |
| Min pause for split | 0.6 s | 0.1 – 3.0 s | Silence between words that starts a new subtitle. Requires word timestamps. |
| Min subtitle duration | 1.0 s | 0.5 – 5.0 s | Shorter subtitles are merged with a neighbour from the same speaker (without joining two separate sentences). |
| Max line length | 42 chars | 20 – 100 | Maximum characters per line, at most two lines per subtitle (film standard = 42). |
| Prefix speaker labels | on | — | Adds `[S01]` when the speaker changes (diarization models only). |

### Subtitle types

- **Standard (for reading)** — short subtitles, split at sentence ends, commas and pauses; lines wrapped to the max line length.
- **Voice-over / TTS (full sentences)** — whole sentences, fewer and longer subtitles, no line wrapping; incomplete sentences are merged across segments (same speaker, short gap). Word pattern, min duration and max line length are hidden in this mode.

### Text cleanup

Before building subtitles, filler sounds (e.g. "uh", "um", "hmm") are removed when word-level data is available, and segments that end mid-sentence are merged with the following lowercase continuation from the same speaker.

---

## Region selection

Open **🎯 Optional: Select Fragments** after loading a file to limit what gets transcribed.

| Type | Shortcut | Color | Effect |
|------|----------|-------|--------|
| INCLUDE | `Ctrl + Left drag` | Purple | Only selected regions are transcribed; they are concatenated before transcription |
| EXCLUDE | `Ctrl + Right drag` | Red | Selected regions are removed (TXT) or muted (SRT) before transcription |

If both types are present, **INCLUDE takes priority** and EXCLUDE is ignored.

**Erasing regions:** the opposite mouse button acts as an eraser — `Ctrl + Right drag` erases INCLUDE regions, `Ctrl + Left drag` erases EXCLUDE regions. Adjacent or overlapping regions of the same type are merged automatically (tolerance: 0.3 s).

> **Timestamps with regions:** with INCLUDE regions (SRT and TXT) and with EXCLUDE in TXT mode, timestamps refer to the processed (shortened) audio, not the original file. In SRT mode, EXCLUDE regions are muted instead of removed, so timestamps stay aligned with the original file.

---

## Output formats

| Format | Description |
|--------|-------------|
| SRT | Subtitle file with timestamps, speaker labels (optional) and the chosen subtitle type. |
| TXT | Plain text. Paragraphs are split on speaker changes and on pauses longer than 2 s. |

The save dialog opens in `./outputs/` by default; you can choose any path.

---

## Installation (Windows)

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) — check *"Add Python to PATH"*. Python 3.12 is recommended.

2. Install **FFmpeg** (required for video files):
   ```bat
   winget install FFmpeg
   ```

3. Clone the repository:
   ```bat
   git clone https://github.com/Mubumbutu/Subtitle-Generator.git
   cd Subtitle-Generator
   ```

4. Run the installer and pick a model:
   ```bat
   install.bat
   ```
   ```
   [1] WhisperX
   [2] MOSS-Transcribe-Diarize
   [3] Parakeet-TDT-0.6B-v3
   ```
   Each model is installed in its **own virtual environment** at `venvs/venv_<model>/`, so dependencies never conflict. You can install any number of models; running an installer again lets you update or recreate an existing environment.

---

## License

[GNU General Public License v3.0](LICENSE)
