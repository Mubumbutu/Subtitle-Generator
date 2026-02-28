# Subtitle Generator

A desktop application for generating subtitles (SRT) and plain-text transcriptions from audio and video files. Using WhisperX for transcription and word-level timestamp alignment.

<img width="950" height="861" alt="subtitle" src="https://github.com/user-attachments/assets/6b3ea144-b121-4c56-8d18-cf0c70897e1f" />

---

## What it does

Loads an audio or video file, transcribes it with WhisperX, aligns word-level timestamps, groups words into subtitle blocks based on configurable rules, and saves the result as an SRT file or a plain-text transcription. The interface includes an interactive waveform viewer that allows playback and optional region selection before processing.

---

## Supported input formats

| Type | Extensions |
|------|------------|
| Audio | `.mp3` · `.wav` · `.flac` · `.m4a` · `.ogg` · `.wma` |
| Video | `.mp4` · `.avi` · `.mkv` · `.mov` · `.flv` · `.wmv` · `.webm` · `.m4v` |

Video files require **FFmpeg** installed and available in `PATH`. Audio is extracted automatically before processing.

---

## Output formats

| Format | Description |
|--------|-------------|
| SRT | Subtitle file with timestamps. Words are grouped into blocks by word count pattern and split on natural pauses. |
| TXT | Plain-text transcription. Segments separated by pauses longer than 2 seconds are written as separate paragraphs. |

---

## Transcription

The application uses **WhisperX**, which runs a Whisper model for transcription followed by a language-specific alignment model for precise word-level timestamps.

**Available Whisper models:**

| Model | Notes |
|-------|-------|
| `tiny` / `tiny.en` | Fastest, lowest accuracy |
| `base` / `base.en` | |
| `small` / `small.en` | Default |
| `medium` / `medium.en` | |
| `large-v3` | Best accuracy, ~3–5 GB download, GPU recommended |

`.en` variants are English-only and faster for English audio.

Models are downloaded from HuggingFace Hub on first use and stored locally in `./models/whisperx/<model_name>/`. The application checks for a complete local copy (`config.json` + `model.bin`) before attempting a download. Alignment models are stored in `./models/whisperx/align/<language_code>/`.

**Compute types selected automatically:**

| Device | GPU compute capability | Compute type |
|--------|------------------------|--------------|
| CUDA | ≥ sm_70 (RTX / Volta+) | float16 |
| CUDA | < sm_70 | float32 |
| CPU | — | int8 |

---

## Voice separation

Optional preprocessing step using **Demucs** (`--two-stems=vocals`). Extracts the vocal track from the audio before transcription, which can improve accuracy for files with background music or noise.

- Demucs must be installed separately (`pip install demucs`)
- Significantly increases processing time (2–5 minutes per file depending on length)
- Output vocals are saved as MP3 (320 kbps) to a temporary directory and cleaned up after processing
- Filenames with non-ASCII characters are handled via a safe copy before Demucs runs

If Demucs is not found or fails, the application falls back to the original audio file and continues.

---

## Region selection

After loading a file, the waveform widget allows selecting time regions before processing.

**Two region types:**

| Type | Color | Effect |
|------|-------|--------|
| INCLUDE (purple) | `Ctrl + Left drag` | Only selected regions are transcribed. Audio is concatenated and sent to WhisperX. |
| EXCLUDE (red) | `Ctrl + Right drag` | Selected regions are removed (TXT) or muted / zeroed-out (SRT) before transcription. |

If both types are present, INCLUDE takes priority and EXCLUDE is ignored.

**Erasing regions:** The opposite mouse button acts as an eraser for existing selections of the same type (e.g. Ctrl+Right drag erases INCLUDE regions). Adjacent or overlapping regions of the same type are merged automatically (merge tolerance: 0.3 s).

**Navigation:**
- **Scroll wheel** — zoom in/out (up to 300×), anchored to cursor position
- **Left drag** — pan the view
- **Left click** (no drag) — seek playback position
- A scrollbar appears below the waveform when zoom > 1×

---

## Waveform playback

The waveform panel includes a basic audio player backed by `sounddevice`.

- **Play / Pause / Stop / Reset** buttons
- Playback position displayed in seconds and shown as a red cursor on the waveform
- If INCLUDE regions are selected, playback plays only those regions in order
- Otherwise, playback starts from the current cursor position
- Audio is loaded at 16 000 Hz mono; stereo files are converted to mono, and files with a different sample rate are resampled via `librosa`

---

## Microphone recording

The application includes a live recording feature backed by `sounddevice` + `soundfile`.

- Recorded at 16 000 Hz mono, saved as WAV to the system temp directory
- VU meter shown during recording
- Pause / Resume supported
- After stopping, the recording is loaded into the waveform viewer automatically and set as the input file

---

## Subtitle parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Word pattern | `3,4` | any positive integers | Alternating max word counts per subtitle block (e.g. `3,4` = 3 words, 4 words, 3 words, …) |
| Min pause for split | 0.6 s | 0.1 – 3.0 s | Gap between words that forces a subtitle boundary |
| Min subtitle duration | 1.0 s | 0.5 – 5.0 s | Subtitles shorter than this are merged with the next one |
| Max line length | 42 chars | 20 – 100 | Maximum characters per line; text wraps to a second line (film standard = 42) |
| Batch size | 16 | 1 – 64 | WhisperX transcription batch size; higher = faster, more VRAM |
| Language | `en` | ISO 639-1 code or empty | Leave empty for automatic language detection |

---

## Model settings

| Setting | Options | Notes |
|---------|---------|-------|
| Device | CPU / GPU (CUDA) | GPU selected automatically if CUDA is available; GPU name shown |
| TF32 | On / Off | GPU only. Faster inference on RTX 30xx / 40xx+ (Ampere+). Not shown in CPU mode. |

---

## Requirements

**Python 3.10+**

**FFmpeg** — required for video files. Must be in `PATH`.

```
PyQt6
torch
whisperx
sounddevice
soundfile
librosa
pysrt
numpy
```

Optional — required only for voice separation:
```
demucs
```

---

## Installation

### Windows

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

4. Create a virtual environment and install dependencies:
   ```bat
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. Install PyTorch with CUDA (GPU) or CPU-only:
   ```bat
   :: CPU:
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   :: CUDA 12.4 (driver ≥ 550):
   pip install torch --index-url https://download.pytorch.org/whl/cu124

   :: CUDA 12.1 (driver ≥ 525):
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   :: CUDA 11.8 (driver ≥ 450):
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   Check your driver version with `nvidia-smi`.

---

### Linux

```bash
# FFmpeg
sudo apt install ffmpeg   # Debian/Ubuntu
# or: sudo dnf install ffmpeg

git clone https://github.com/Mubumbutu/Subtitle-Generator.git
cd Subtitle-Generator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU — CUDA 12.4 (driver ≥ 550):
# pip install torch --index-url https://download.pytorch.org/whl/cu124

```

---

### macOS

```bash
brew install ffmpeg

git clone https://github.com/Mubumbutu/Subtitle-Generator.git
cd Subtitle-Generator

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install torch
```

> CUDA is not available on macOS. Transcription runs on CPU only. Apple Silicon (M1/M2/M3) MPS acceleration is not explicitly configured.

---

## Running

```bash
# Activate the virtual environment first:
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

python subtitle_generator.py
```

---

## Workflow

```
Load file → (Optional) Select regions → Configure model → Generate → Save
```

1. **Load a file** — drag & drop onto the window, use `📁 Browse...`, or record directly with `🎤 Start Record`.
2. **Waveform** — inspect the audio, play it back, zoom with scroll wheel.
3. **Select regions** (optional) — `Ctrl+Left drag` to include, `Ctrl+Right drag` to exclude. Without any selection, the entire file is processed.
4. **Model Settings** — choose device (CPU / GPU), WhisperX model size, language, and batch size.
5. **Advanced Settings** — adjust word pattern, pause threshold, subtitle duration, and line length. Enable TF32 or voice separation if needed.
6. **Generate** — click `🚀 Generate Subtitles (.srt)` or `📄 Export as Text (.txt)`.
7. Output is saved next to the input file with the same name (`.srt` or `.txt`). A custom path can be set via `💾 Browse...`.

---

## Notes

- **First run downloads models** — `large-v3` is approximately 3–5 GB. Ensure a stable connection or pre-download by running transcription once with the target model selected.
- **VRAM usage** — `large-v3` on GPU requires ~6–8 GB VRAM. Reduce batch size if you get out-of-memory errors.
- **Word pattern** — `3,4` produces subtitles alternating between 3 and 4 words. A long natural pause always overrides the pattern and forces a split regardless of word count.
- **TXT export and region selection** — in TXT mode, INCLUDE regions are extracted and concatenated; EXCLUDE regions are removed. Timestamps in the output reflect the concatenated audio, not the original file.
- **SRT and EXCLUDE regions** — excluded regions are muted (zeroed out) in the audio rather than removed, so SRT timestamps remain aligned with the original file.
- **Demucs filenames** — files with non-ASCII characters (accents, CJK, etc.) in their names are copied to a temporary location with a safe ASCII name before Demucs runs, to avoid subprocess errors.

---

## License

[GNU General Public License v3.0](LICENSE)
