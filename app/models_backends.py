"""Adding a backend:
1. Create app/<name>_backend.py with a TranscriptionBackend subclass. Import heavy
   libraries only inside methods so the module stays importable in any environment.
2. Declare id, label, capabilities and settings on the class; the GUI builds its
   model controls from them.
3. Implement transcribe(request) and return a TranscriptionResult whose segments
   have start, end and text (speaker, words and confidence are optional).
4. Add one BackendSpec entry to BACKEND_SPECS below.
5. Add install/<name>_requirements.txt and install/<name>_install.bat. The
   environment goes to venvs/venv_<name>/ and start.bat discovers it by name.
"""
from __future__ import annotations

import importlib
import importlib.util
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VENVS_DIR = PROJECT_ROOT / "venvs"
MODELS_DIR = PROJECT_ROOT / "models"
ICON_PATH = PROJECT_ROOT / "icon.ico"


def ensure_project_dirs() -> None:
    for directory in (OUTPUTS_DIR, VENVS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


class BackendError(RuntimeError):
    pass


class BackendUnavailableError(BackendError):
    pass


class TaskCancelled(Exception):
    pass


ProgressCallback = Callable[[float | None, str], None]
LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    kind: str
    default: Any
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    suffix: str = ""
    hint: str = ""
    placeholder: str = ""
    section: str = "model"


@dataclass(frozen=True)
class Capabilities:
    device_selection: bool = True
    word_timestamps: bool = False
    speaker_labels: bool = False


@dataclass(frozen=True)
class DeviceOption:
    id: str
    label: str
    enabled: bool = True
    default: bool = False
    tooltip: str = ""


@dataclass
class TranscriptionRequest:
    audio_path: Path
    device: str
    settings: dict[str, Any]
    progress: ProgressCallback
    log: LogCallback
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TaskCancelled()


@dataclass
class TranscriptionResult:
    segments: list[dict[str, Any]]
    language: str | None = None


class TranscriptionBackend(ABC):
    id: str
    label: str
    capabilities: Capabilities = Capabilities()
    settings: tuple[SettingSpec, ...] = ()

    @classmethod
    def available_devices(cls) -> list[DeviceOption]:
        return torch_device_options()

    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        ...

    def cleanup(self) -> None:
        return None


@dataclass(frozen=True)
class BackendSpec:
    id: str
    label: str
    module: str
    class_name: str
    required_modules: tuple[str, ...]

    @property
    def venv_dir(self) -> Path:
        return VENVS_DIR / f"venv_{self.id}"


BACKEND_SPECS: dict[str, BackendSpec] = {
    spec.id: spec
    for spec in (
        BackendSpec("whisperx", "WhisperX", "whisperx_backend", "WhisperXBackend",
                    ("PyQt6", "torch", "whisperx", "librosa", "sounddevice", "soundfile", "pysrt")),
        BackendSpec("moss", "MOSS-Transcribe-Diarize", "moss_backend", "MossBackend",
                    ("PyQt6", "torch", "transformers", "moss_transcribe_diarize", "librosa",
                     "sounddevice", "soundfile", "pysrt")),
        BackendSpec("parakeet", "Parakeet-TDT-0.6B-v3", "parakeet_backend", "ParakeetBackend",
                    ("PyQt6", "torch", "nemo", "librosa", "sounddevice", "soundfile", "pysrt")),
    )
}


def get_spec(backend_id: str) -> BackendSpec:
    try:
        return BACKEND_SPECS[backend_id]
    except KeyError:
        raise BackendError(
            f"Unknown model '{backend_id}'. Available: {', '.join(BACKEND_SPECS)}"
        ) from None


def missing_modules(spec: BackendSpec) -> list[str]:
    missing = []
    for name in spec.required_modules:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def load_backend_class(backend_id: str) -> type[TranscriptionBackend]:
    spec = get_spec(backend_id)
    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Backend '{spec.label}' could not be imported: {exc}. "
            f"Run install.bat and install {spec.label}."
        ) from exc
    return getattr(module, spec.class_name)


def torch_device_options() -> list[DeviceOption]:
    options = [DeviceOption("cpu", "CPU")]
    try:
        import torch

        if torch.cuda.is_available():
            try:
                label = f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
            except Exception:
                label = "GPU (CUDA)"
            options.append(DeviceOption("cuda", label, default=True))
            return options
    except ImportError:
        pass
    options.append(DeviceOption("cuda", "GPU (CUDA)", enabled=False,
                                tooltip="CUDA not available on this system"))
    options[0] = DeviceOption("cpu", "CPU", default=True)
    return options


def _number(value: Any, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendError(f"Segment {index}: '{name}' must be a number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise BackendError(f"Segment {index}: '{name}' is not finite")
    return float(value)


def validate_segments(segments: Any) -> list[dict[str, Any]]:
    if not isinstance(segments, (list, tuple)):
        raise BackendError("Backend returned segments that are not a list")
    clean: list[dict[str, Any]] = []
    for index, raw in enumerate(segments):
        if not isinstance(raw, dict):
            raise BackendError(f"Segment {index} is not a mapping")
        for key in ("start", "end", "text"):
            if key not in raw:
                raise BackendError(f"Segment {index} is missing '{key}'")
        start = _number(raw["start"], "start", index)
        end = _number(raw["end"], "end", index)
        if start < 0 or end < start:
            raise BackendError(f"Segment {index} has invalid times {start}..{end}")
        if not isinstance(raw["text"], str):
            raise BackendError(f"Segment {index}: 'text' must be a string")
        text = raw["text"].strip()
        if not text:
            continue
        seg: dict[str, Any] = {"start": start, "end": end, "text": text}
        speaker = raw.get("speaker")
        if speaker is not None:
            seg["speaker"] = str(speaker)
        confidence = raw.get("confidence")
        if confidence is not None:
            seg["confidence"] = _number(confidence, "confidence", index)
        words = raw.get("words")
        if words:
            valid_words = []
            for word in words:
                if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                    continue
                w_start, w_end = word.get("start"), word.get("end")
                if w_start is None or w_end is None:
                    continue
                valid_words.append({"word": word["word"], "start": float(w_start), "end": float(w_end)})
            if valid_words:
                seg["words"] = valid_words
        clean.append(seg)
    clean.sort(key=lambda s: s["start"])
    return clean
