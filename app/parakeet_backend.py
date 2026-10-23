import contextlib
import io
import os

from models_backends import (
    BackendError,
    BackendUnavailableError,
    Capabilities,
    MODELS_DIR,
    SettingSpec,
    TranscriptionBackend,
    TranscriptionRequest,
    TranscriptionResult,
)

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetBackend(TranscriptionBackend):
    id = "parakeet"
    label = "Parakeet-TDT-0.6B-v3"
    capabilities = Capabilities(device_selection=True, word_timestamps=True, speaker_labels=False)
    settings = (
        SettingSpec(
            "model_id", "Parakeet Model:", "text", DEFAULT_MODEL_ID,
            hint="Hugging Face repository name or path to a local folder with the Parakeet model. "
                 "The default works for most users. The model is downloaded on first use and cached.",
        ),
        SettingSpec(
            "batch_size", "Batch Size:", "int", 8,
            minimum=1, maximum=64,
            hint="Number of audio chunks processed at once. Higher values are faster but use "
                 "more VRAM. Lower it if you get out-of-memory errors.",
        ),
    )

    def __init__(self):
        self._model = None

    @staticmethod
    def segments_from_output(sample) -> list[dict]:
        timestamps = getattr(sample, "timestamp", None) or {}
        raw_segments = timestamps.get("segment") or []
        raw_words = timestamps.get("word") or []

        segments: list[dict] = []
        word_idx = 0

        for seg in raw_segments:
            try:
                seg_start = float(seg["start"])
                seg_end = float(seg["end"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(seg.get("segment", "")).strip()
            if not text:
                continue

            seg_words = []
            while word_idx < len(raw_words):
                w = raw_words[word_idx]
                try:
                    w_start = float(w["start"])
                    w_end = float(w["end"])
                    w_text = str(w["word"])
                except (KeyError, TypeError, ValueError):
                    word_idx += 1
                    continue
                if w_start > seg_end + 0.01:
                    break
                if w_end < seg_start - 0.01:
                    word_idx += 1
                    continue
                seg_words.append({"word": w_text, "start": w_start, "end": w_end})
                word_idx += 1

            entry = {"start": seg_start, "end": seg_end, "text": text}
            if seg_words:
                entry["words"] = seg_words
            segments.append(entry)

        return segments

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        log = request.log
        cache_dir = MODELS_DIR / "parakeet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))

        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise BackendUnavailableError(
                f"Parakeet environment is incomplete ({exc}). Run install.bat and reinstall Parakeet-TDT-0.6B-v3."
            ) from exc

        model_id = str(request.settings.get("model_id") or DEFAULT_MODEL_ID).strip()
        batch_size = int(request.settings["batch_size"])

        request.check_cancelled()
        request.progress(None, "Loading Parakeet model...")
        log(f"▶ Loading Parakeet model ({model_id})")
        log("  Files are downloaded to the local cache on first use and reused afterwards.")

        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        except Exception as exc:
            raise BackendError(
                f"Failed to load Parakeet model '{model_id}': {exc}"
            ) from exc

        device = "cuda" if request.device == "cuda" and torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        self._model = model
        log(f"✓ Model loaded on {device.upper()}")

        request.check_cancelled()
        request.progress(None, "Transcribing...")
        log("▶ Transcribing audio...")

        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                outputs = model.transcribe(
                    [str(request.audio_path)],
                    timestamps=True,
                    batch_size=batch_size,
                )
        except TypeError:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                outputs = model.transcribe([str(request.audio_path)], timestamps=True)

        if not outputs:
            raise BackendError("Parakeet returned no output")

        segments = self.segments_from_output(outputs[0])
        log(f" ✓ Parsed {len(segments)} segments")

        return TranscriptionResult(segments=segments)

    def cleanup(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass