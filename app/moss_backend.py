import re

from models_backends import (
    BackendError,
    BackendUnavailableError,
    Capabilities,
    SettingSpec,
    TaskCancelled,
    TranscriptionBackend,
    TranscriptionRequest,
    TranscriptionResult,
)

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
HOTWORD_PREFIX = "热词提示："


class MossBackend(TranscriptionBackend):
    id = "moss"
    label = "MOSS-Transcribe-Diarize"
    capabilities = Capabilities(device_selection=True, word_timestamps=False, speaker_labels=True)
    settings = (
        SettingSpec(
            "model_id", "MOSS Model:", "text", DEFAULT_MODEL_ID,
            hint="Hugging Face repository name or path to a local folder with the MOSS model. "
                 "The default works for most users. The model is downloaded on first use and cached.",
        ),
        SettingSpec(
            "max_new_tokens", "Max New Tokens:", "int", 32768,
            minimum=512, maximum=131072, step=512,
            hint="Maximum number of tokens the model may generate. Generation stops at this limit, "
                 "so a value that is too low cuts the transcript short. Raise it for long or "
                 "multi-speaker recordings.",
        ),
        SettingSpec(
            "hotwords", "Hotwords:", "text", "",
            placeholder="word1, word2",
            hint="Optional names or domain terms, separated by commas, that the model should "
                 "pay attention to, for example product names or technical jargon.",
        ),
    )

    def __init__(self):
        self._model = None

    @staticmethod
    def build_prompt(default_prompt: str, hotwords: str) -> str:
        words = [w.strip() for w in re.split(r"[,\n;，、]", hotwords or "") if w.strip()]
        if not words:
            return default_prompt
        return default_prompt + HOTWORD_PREFIX + ", ".join(words)

    @staticmethod
    def segments_from_transcript(text: str) -> list[dict]:
        from moss_transcribe_diarize import parse_transcript

        segments = [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in parse_transcript(text)
        ]
        if not segments and text.strip():
            snippet = text.strip()[:300]
            raise BackendError(
                "MOSS output did not contain any [start][Sxx]text[end] segments. "
                f"Raw output starts with: {snippet!r}"
            )
        return segments

    def _load_model(self, torch, model_id, device, dtype, log):
        import importlib.util

        from transformers import AutoModelForCausalLM

        candidates = ("sdpa", "eager")
        if device.type == "cuda" and importlib.util.find_spec("flash_attn") is not None:
            candidates = ("flash_attention_2",) + candidates
        for attn in candidates:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True, dtype="auto", attn_implementation=attn
                )
            except Exception as exc:
                if attn == candidates[-1]:
                    raise
                log(f"  attention '{attn}' unavailable ({exc}); trying next")
                continue
            if attn == "eager":
                log("  ⚠️ Eager attention uses quadratic memory and may fail on long audio")
            log(f"  Attention implementation: {attn}")
            return model.to(dtype=dtype).to(device).eval()

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        log = request.log
        try:
            import torch
            from moss_transcribe_diarize.inference_utils import (
                DEFAULT_PROMPT,
                build_transcription_messages,
                generate_transcription,
                resolve_device,
            )
            from transformers import AutoProcessor
        except ImportError as exc:
            raise BackendUnavailableError(
                f"MOSS environment is incomplete ({exc}). Run install.bat and reinstall MOSS-Transcribe-Diarize."
            ) from exc

        model_id = str(request.settings.get("model_id") or DEFAULT_MODEL_ID).strip()
        max_new_tokens = int(request.settings["max_new_tokens"])
        prompt = self.build_prompt(DEFAULT_PROMPT, str(request.settings.get("hotwords", "")))

        device = resolve_device("cuda:0" if request.device == "cuda" else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        request.check_cancelled()
        request.progress(None, "Loading MOSS model...")
        log(f"▶ Loading MOSS model ({model_id}) on {device.type.upper()} (dtype={str(dtype).replace('torch.', '')})")
        log("  Files are downloaded to the Hugging Face cache on first use and reused afterwards.")
        self._model = self._load_model(torch, model_id, device, dtype, log)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        log("✓ Model loaded successfully")

        request.check_cancelled()
        request.progress(None, "Transcribing...")
        log("▶ Transcribing audio...")
        state = {"last_logged": 0}

        def on_tokens(count: int) -> None:
            if request.cancel_event.is_set():
                raise TaskCancelled()
            if count - state["last_logged"] >= 256:
                state["last_logged"] = count
                request.progress(None, f"Transcribing... {count} tokens generated")

        result = generate_transcription(
            self._model,
            processor,
            build_transcription_messages(request.audio_path, prompt),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
            token_callback=on_tokens,
        )
        generated = result["generated_tokens"]
        log(f" Generated {generated} tokens")
        if generated >= max_new_tokens:
            log(" ⚠️ Generation reached Max New Tokens; the transcript may be truncated. Increase the limit if needed.")

        segments = self.segments_from_transcript(result["text"])
        log(f" ✓ Parsed {len(segments)} speaker segments")
        return TranscriptionResult(segments=segments)

    def cleanup(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
