import contextlib
import functools
import io
import logging
import os
import sys
import warnings
from pathlib import Path

from models_backends import (
    BackendUnavailableError,
    Capabilities,
    MODELS_DIR,
    SettingSpec,
    TranscriptionBackend,
    TranscriptionRequest,
    TranscriptionResult,
)

_site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
_nvidia_base = _site_packages / "nvidia"
_dll_dir_handles = []
for _package in ("cudnn", "cublas"):
    _bin = _nvidia_base / _package / "bin"
    os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")
if _nvidia_base.is_dir() and hasattr(os, "add_dll_directory"):
    for _package in os.listdir(_nvidia_base):
        _bin = _nvidia_base / _package / "bin"
        if _bin.is_dir():
            _dll_dir_handles.append(os.add_dll_directory(str(_bin)))

for _name in ("whisperx", "whisperx.asr", "whisperx.vads", "whisperx.vads.pyannote"):
    logging.getLogger(_name).setLevel(logging.WARNING)
for _name in ("pytorch_lightning", "lightning", "transformers"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", category=Warning, message=".*reproducibility.*")

WHISPERX_MODELS_DIR = MODELS_DIR / "whisperx"
_torch_load_patched = False


def get_whisperx_model_path(model_name: str) -> Path:
    WHISPERX_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace(".", "_").strip()
    if not safe_name:
        safe_name = "unknown_model"
    return WHISPERX_MODELS_DIR / safe_name


def is_whisperx_model_cached(model_name: str) -> bool:
    model_path = get_whisperx_model_path(model_name)
    if not model_path.exists() or not model_path.is_dir():
        return False
    hf_dir = model_path / f"models--Systran--faster-whisper-{model_name}"
    if hf_dir.exists() and hf_dir.is_dir():
        snapshots_dir = hf_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    if (snapshot / "model.bin").exists() and (snapshot / "config.json").exists():
                        return True
        if (hf_dir / "model.bin").exists() and (hf_dir / "config.json").exists():
            return True
    return all((model_path / f).exists() for f in ("config.json", "model.bin"))


def is_align_model_cached(language_code: str) -> bool:
    align_dir = WHISPERX_MODELS_DIR / "align" / language_code
    if not align_dir.exists() or not align_dir.is_dir():
        return False
    for entry in align_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            snapshots_dir = entry / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        if (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists():
                            return True
    return False


def _patch_torch_load(torch) -> None:
    global _torch_load_patched
    if _torch_load_patched:
        return
    original = torch.load

    @functools.wraps(original)
    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    _torch_load_patched = True


class WhisperXBackend(TranscriptionBackend):
    id = "whisperx"
    label = "WhisperX"
    capabilities = Capabilities(device_selection=True, word_timestamps=True, speaker_labels=False)
    settings = (
        SettingSpec(
            "model", "WhisperX Model:", "choice", "large-v3",
            choices=("tiny", "tiny.en", "base", "base.en", "small", "small.en",
                     "medium", "medium.en", "large-v3"),
            hint="Whisper model size. Larger models are more accurate but slower and need more "
                 "VRAM and disk space. Models ending in .en handle English only. "
                 "The model is downloaded on first use.",
        ),
        SettingSpec(
            "language", "Language:", "text", "en",
            placeholder="auto",
            hint="Spoken language as an ISO code, for example en, pl, es, fr, de. "
                 "Leave empty to detect it automatically. Setting it explicitly is faster "
                 "and avoids wrong detection.",
        ),
        SettingSpec(
            "batch_size", "Batch Size:", "int", 16,
            minimum=1, maximum=64,
            hint="Number of audio chunks processed at once. Higher values are faster but use "
                 "more VRAM. Lower it if you get out-of-memory errors.",
        ),
    )

    def __init__(self):
        self._model = None

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        log = request.log
        try:
            import torch
            import whisperx
        except ImportError as exc:
            raise BackendUnavailableError(
                f"WhisperX environment is incomplete ({exc}). Run install.bat and reinstall WhisperX."
            ) from exc
        _patch_torch_load(torch)

        device = request.device
        model_size = request.settings["model"]
        batch_size = int(request.settings["batch_size"])
        language = str(request.settings.get("language", "")).strip()

        if device == "cuda":
            try:
                major, _minor = torch.cuda.get_device_capability(0)
                compute_type = "float16" if major >= 7 else "float32"
            except Exception:
                compute_type = "float32"
        else:
            compute_type = "int8"

        model_cache_dir = get_whisperx_model_path(model_size)
        if is_whisperx_model_cached(model_size):
            log(f"✓ Using locally cached WhisperX model: {model_cache_dir}")
            local_files_only = True
        else:
            log(f"⚠️ Model '{model_size}' not found locally or incomplete.")
            log(f"   Downloading from Hugging Face to: {model_cache_dir}")
            log("   This may take several minutes (large-v3 ≈ 3–5 GB, depending on connection).")
            local_files_only = False

        request.check_cancelled()
        request.progress(None, f"Loading WhisperX model ({model_size})...")
        log(f"▶ Loading WhisperX model ({model_size}) on {device.upper()} (compute_type={compute_type})")
        with contextlib.redirect_stderr(io.StringIO()):
            model = whisperx.load_model(
                model_size,
                device,
                compute_type=compute_type,
                download_root=str(model_cache_dir),
                local_files_only=local_files_only,
            )
        self._model = model
        log("✓ Model loaded successfully")

        request.check_cancelled()
        request.progress(None, "Transcribing...")
        log("▶ Transcribing audio...")
        transcribe_kwargs = {"batch_size": batch_size}
        if language:
            transcribe_kwargs["language"] = language
            log(f" Language: {language}")
        else:
            log(" Language: auto-detect")

        result = model.transcribe(str(request.audio_path), **transcribe_kwargs)
        detected_language = result.get("language", "unknown")
        log(f" Detected language: {detected_language}")

        self._model = None
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        request.check_cancelled()
        request.progress(None, "Aligning word timestamps...")
        log("▶ Aligning word timestamps...")
        try:
            align_cache_dir = WHISPERX_MODELS_DIR / "align" / detected_language
            align_cache_dir.mkdir(parents=True, exist_ok=True)
            if is_align_model_cached(detected_language):
                log(f" ✓ Using locally cached alignment model for: {detected_language}")
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            else:
                log(f" Downloading alignment model for: {detected_language}...")
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE", None)

            with contextlib.redirect_stderr(io.StringIO()):
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=device,
                    model_dir=str(align_cache_dir),
                )
            audio = whisperx.load_audio(str(request.audio_path))
            aligned_result = whisperx.align(
                result["segments"], align_model, metadata, audio, device,
                return_char_alignments=False,
            )
            result["segments"] = aligned_result["segments"]
            log(" ✓ Alignment successful")
            del align_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f" ⚠️ Warning: Alignment failed ({exc})")
            log(" Continuing with unaligned timestamps...")
        finally:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)

        return TranscriptionResult(segments=result["segments"], language=detected_language)

    def cleanup(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
