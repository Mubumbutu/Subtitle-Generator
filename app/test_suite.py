import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import models_backends as mb

try:
    import PyQt6  # noqa: F401
    import numpy as np
    import pysrt
    import soundfile as sf

    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")
    import core

    HAVE_CORE_DEPS = True
except (ImportError, OSError):
    HAVE_CORE_DEPS = False


class RegistryTests(unittest.TestCase):
    def test_specs_are_consistent(self):
        self.assertEqual(set(mb.BACKEND_SPECS), {"whisperx", "moss", "parakeet"})
        for backend_id, spec in mb.BACKEND_SPECS.items():
            self.assertEqual(spec.id, backend_id)
            self.assertEqual(spec.venv_dir.name, f"venv_{backend_id}")
            self.assertEqual(spec.venv_dir.parent, mb.PROJECT_ROOT / "venvs")

    def test_unknown_backend(self):
        with self.assertRaises(mb.BackendError):
            mb.get_spec("nope")

    def test_backend_modules_do_not_import_heavy_libraries(self):
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import models_backends as mb\n"
            "for i in mb.BACKEND_SPECS: mb.load_backend_class(i)\n"
            "heavy = {'torch','whisperx','transformers','moss_transcribe_diarize','ctranslate2','nemo'}\n"
            "loaded = heavy & set(sys.modules)\n"
            "sys.exit(1 if loaded else 0)\n" % str(APP_DIR)
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_declarations(self):
        for backend_id in mb.BACKEND_SPECS:
            cls = mb.load_backend_class(backend_id)
            self.assertEqual(cls.id, backend_id)
            keys = [s.key for s in cls.settings]
            self.assertEqual(len(keys), len(set(keys)))
            for spec in cls.settings:
                self.assertIn(spec.kind, {"choice", "text", "int", "float", "bool"})
                self.assertIn(spec.section, {"model", "advanced"})
            devices = cls.available_devices()
            self.assertEqual(sum(1 for d in devices if d.default and d.enabled), 1)

    def test_capabilities_are_not_overstated(self):
        self.assertTrue(mb.load_backend_class("whisperx").capabilities.word_timestamps)
        self.assertFalse(mb.load_backend_class("whisperx").capabilities.speaker_labels)
        self.assertTrue(mb.load_backend_class("moss").capabilities.speaker_labels)
        self.assertFalse(mb.load_backend_class("moss").capabilities.word_timestamps)
        self.assertTrue(mb.load_backend_class("parakeet").capabilities.word_timestamps)
        self.assertFalse(mb.load_backend_class("parakeet").capabilities.speaker_labels)


class ValidationTests(unittest.TestCase):
    def test_valid_segments_are_normalized(self):
        out = mb.validate_segments([
            {"start": 2, "end": 3, "text": " b ", "speaker": "S02"},
            {"start": 0.0, "end": 1.5, "text": "a", "words": [
                {"word": "a", "start": 0.0, "end": 1.0},
                {"word": "x", "start": None, "end": None},
            ], "score": 1},
        ])
        self.assertEqual([s["text"] for s in out], ["a", "b"])
        self.assertEqual(out[0]["words"], [{"word": "a", "start": 0.0, "end": 1.0}])
        self.assertEqual(out[1]["speaker"], "S02")
        self.assertNotIn("score", out[0])

    def test_empty_text_is_dropped(self):
        self.assertEqual(mb.validate_segments([{"start": 0, "end": 1, "text": "  "}]), [])

    def test_invalid_segments_raise(self):
        bad_inputs = [
            "text",
            [None],
            [{"start": 0, "end": 1}],
            [{"start": 2, "end": 1, "text": "x"}],
            [{"start": "0", "end": 1, "text": "x"}],
            [{"start": 0, "end": float("nan"), "text": "x"}],
            [{"start": 0, "end": 1, "text": 5}],
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad), self.assertRaises(mb.BackendError):
                mb.validate_segments(bad)


class MossParsingTests(unittest.TestCase):
    def setUp(self):
        self.cls = mb.load_backend_class("moss")
        self.saved = sys.modules.get("moss_transcribe_diarize")
        stub = types.ModuleType("moss_transcribe_diarize")
        Segment = types.SimpleNamespace

        def parse_transcript(text):
            if "[0.48][S01]" in text:
                return [Segment(start=0.48, end=1.66, speaker="S01", text="Welcome everyone")]
            return []

        stub.parse_transcript = parse_transcript
        sys.modules["moss_transcribe_diarize"] = stub

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop("moss_transcribe_diarize", None)
        else:
            sys.modules["moss_transcribe_diarize"] = self.saved

    def test_prompt_without_and_with_hotwords(self):
        self.assertEqual(self.cls.build_prompt("P", ""), "P")
        self.assertEqual(self.cls.build_prompt("P", " a,b \n c"), "P热词提示：a, b, c")

    def test_segments_are_normalized(self):
        segs = self.cls.segments_from_transcript("[0.48][S01]Welcome everyone[1.66]")
        self.assertEqual(segs, [{"start": 0.48, "end": 1.66, "text": "Welcome everyone", "speaker": "S01"}])
        self.assertEqual(mb.validate_segments(segs)[0]["speaker"], "S01")

    def test_unparseable_output_raises_with_diagnostics(self):
        with self.assertRaises(mb.BackendError) as ctx:
            self.cls.segments_from_transcript("garbage output")
        self.assertIn("garbage output", str(ctx.exception))

    def test_empty_output_gives_no_segments(self):
        self.assertEqual(self.cls.segments_from_transcript(""), [])

class ParakeetParsingTests(unittest.TestCase):
    def setUp(self):
        self.cls = mb.load_backend_class("parakeet")

    def test_segments_with_words_are_normalized(self):
        sample = types.SimpleNamespace(
            text="Hello there. Thanks.",
            timestamp={
                "segment": [
                    {"segment": "Hello there.", "start": 0.0, "end": 1.5},
                    {"segment": "Thanks.", "start": 1.6, "end": 2.4},
                ],
                "word": [
                    {"word": "Hello", "start": 0.0, "end": 0.7},
                    {"word": "there.", "start": 0.7, "end": 1.5},
                    {"word": "Thanks.", "start": 1.6, "end": 2.4},
                ],
            },
        )
        segs = self.cls.segments_from_output(sample)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["text"], "Hello there.")
        self.assertEqual(len(segs[0]["words"]), 2)
        self.assertEqual(len(segs[1]["words"]), 1)
        validated = mb.validate_segments(segs)
        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[1]["text"], "Thanks.")
        self.assertEqual(len(validated[0]["words"]), 2)

    def test_text_only_output(self):
        sample = types.SimpleNamespace(text="Hello", timestamp=None)
        self.assertEqual(self.cls.segments_from_output(sample), [])

    def test_empty_timestamps(self):
        sample = types.SimpleNamespace(text="", timestamp={})
        self.assertEqual(self.cls.segments_from_output(sample), [])

class CliTests(unittest.TestCase):
    def run_main(self, *args):
        return subprocess.run([sys.executable, str(APP_DIR / "main.py"), *args], capture_output=True, text=True)

    def test_unknown_probe(self):
        self.assertEqual(self.run_main("--probe", "nope").returncode, 2)

    def test_probe_reports_missing_modules_or_label(self):
        result = self.run_main("--probe", "whisperx")
        self.assertIn(result.returncode, (0, 1))
        if result.returncode == 0:
            self.assertEqual(result.stdout.strip(), "WhisperX")
        else:
            self.assertIn("missing modules", result.stderr)

    def test_invalid_model_argument(self):
        self.assertEqual(self.run_main("--model", "nope").returncode, 2)


class FakeBackend(mb.TranscriptionBackend):
    id = "fake"
    label = "Fake"
    capabilities = mb.Capabilities(speaker_labels=True)
    segments = []
    last_request = None
    last_audio = None
    cleaned = False

    def transcribe(self, request):
        type(self).last_request = request
        data, sr = sf.read(str(request.audio_path), dtype="float32")
        type(self).last_audio = (data, sr)
        request.check_cancelled()
        return mb.TranscriptionResult(segments=type(self).segments)

    def cleanup(self):
        type(self).cleaned = True

@unittest.skipUnless(HAVE_CORE_DEPS, "core dependencies are not installed")
class CoreWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = self.dir / "in.wav"
        t = np.arange(16000 * 6) / 16000
        sf.write(str(self.wav), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)
        FakeBackend.segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello there everyone, welcome.", "speaker": "S01"},
            {"start": 3.2, "end": 6.0, "text": "Thanks for having me today.", "speaker": "S02"},
        ]
        FakeBackend.cleaned = False
        self.config = {
            "device": "cpu", "word_pattern": [3, 4], "min_pause": 0.6, "min_duration": 1.0,
            "max_line_length": 42, "speaker_labels": True, "backend_settings": {},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def generator(self, **config_overrides):
        return core.SubtitleGenerator({**self.config, **config_overrides}, FakeBackend())

    def test_srt_with_speaker_labels(self):
        out = self.dir / "out.srt"
        subs = self.generator().process(self.wav, out, output_format="srt")
        self.assertEqual(len(subs), 2)
        saved = pysrt.open(str(out), encoding="utf-8")
        self.assertEqual(saved[0].text, "[S01] Hello there everyone, welcome.")
        self.assertEqual(saved[1].text, "[S02] Thanks for having me today.")
        self.assertTrue(FakeBackend.cleaned)

    def test_srt_without_speaker_labels(self):
        out = self.dir / "out.srt"
        self.generator(speaker_labels=False).process(self.wav, out, output_format="srt")
        self.assertFalse(pysrt.open(str(out), encoding="utf-8")[0].text.startswith("["))

    def test_different_speakers_are_not_merged(self):
        FakeBackend.segments = [
            {"start": 0.0, "end": 0.4, "text": "Yes", "speaker": "S01"},
            {"start": 0.5, "end": 3.0, "text": "Thanks for having me today.", "speaker": "S02"},
        ]
        subs = self.generator().process(self.wav, self.dir / "o.srt", output_format="srt")
        self.assertEqual(len(subs), 2)

    def test_text_export(self):
        out = self.dir / "out.txt"
        self.generator().process(self.wav, out, output_format="txt")
        text = out.read_text(encoding="utf-8")
        self.assertIn("[S01] Hello there everyone, welcome.", text)
        self.assertIn("[S02] Thanks for having me today.", text)

    def test_exclude_range_is_muted_for_srt(self):
        self.generator().process(self.wav, self.dir / "o.srt", exclude_ranges=[(1.0, 2.0)], output_format="srt")
        data, sr = FakeBackend.last_audio
        self.assertEqual(len(data), 16000 * 6)
        self.assertEqual(float(np.abs(data[16000:32000]).max()), 0.0)
        self.assertGreater(float(np.abs(data[:16000]).max()), 0.1)

    def test_include_range_extracts_audio(self):
        self.generator().process(self.wav, self.dir / "o.srt", include_ranges=[(1.0, 2.0), (4.0, 5.0)], output_format="srt")
        self.assertEqual(len(FakeBackend.last_audio[0]), 32000)

    def test_cancellation_cleans_up_and_raises(self):
        event = threading.Event()
        event.set()
        gen = core.SubtitleGenerator(self.config, FakeBackend(), cancel_event=event)
        with self.assertRaises(mb.TaskCancelled):
            gen.process(self.wav, self.dir / "o.srt", include_ranges=[(1.0, 2.0)], output_format="srt")
        self.assertFalse((Path(tempfile.gettempdir()) / "processed_in.wav").exists())

    def test_invalid_backend_result_is_rejected(self):
        FakeBackend.segments = [{"start": 2, "end": 1, "text": "x"}]
        with self.assertRaises(mb.BackendError):
            self.generator().process(self.wav, self.dir / "o.srt", output_format="srt")

    def test_empty_result_raises(self):
        FakeBackend.segments = []
        with self.assertRaises(RuntimeError):
            self.generator().process(self.wav, self.dir / "o.srt", output_format="srt")

    def test_tts_mode_keeps_sentence_with_commas_in_one_subtitle(self):
        FakeBackend.segments = [
            {"start": 0.0, "end": 5.5, "text": "Well, I think that, after all, we should go home now.", "speaker": "S01"},
        ]
        standard = self.generator().process(self.wav, self.dir / "s.srt", output_format="srt")
        self.assertEqual(len(standard), 2)
        tts = self.generator(segmentation_mode="tts").process(self.wav, self.dir / "t.srt", output_format="srt")
        self.assertEqual(len(tts), 1)
        self.assertEqual(tts[0].text, "[S01] Well, I think that, after all, we should go home now.")

    def test_tts_mode_merges_incomplete_sentence_across_segments(self):
        FakeBackend.segments = [
            {"start": 0.0, "end": 2.0, "text": "This is a sentence that", "speaker": "S01"},
            {"start": 2.5, "end": 4.0, "text": "Continues here.", "speaker": "S01"},
        ]
        tts = self.generator(segmentation_mode="tts").process(self.wav, self.dir / "t.srt", output_format="srt")
        self.assertEqual(len(tts), 1)
        self.assertIn("This is a sentence that Continues here.", tts[0].text)

    def test_tts_mode_does_not_merge_across_long_silence_or_speakers(self):
        FakeBackend.segments = [
            {"start": 0.0, "end": 2.0, "text": "This is a sentence that", "speaker": "S01"},
            {"start": 5.0, "end": 6.0, "text": "Continues here.", "speaker": "S01"},
        ]
        tts = self.generator(segmentation_mode="tts").process(self.wav, self.dir / "t.srt", output_format="srt")
        self.assertEqual(len(tts), 2)
        FakeBackend.segments = [
            {"start": 0.0, "end": 2.0, "text": "This is a sentence that", "speaker": "S01"},
            {"start": 2.5, "end": 4.0, "text": "Continues here.", "speaker": "S02"},
        ]
        tts = self.generator(segmentation_mode="tts").process(self.wav, self.dir / "t2.srt", output_format="srt")
        self.assertEqual(len(tts), 2)

@unittest.skipUnless(HAVE_CORE_DEPS, "core dependencies are not installed")
class GuiGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_controls_are_generated_from_metadata(self):
        for backend_id in mb.BACKEND_SPECS:
            with self.subTest(backend=backend_id):
                window = core.SubtitleGeneratorGUI(backend_id)
                cls = window.backend_cls
                self.assertEqual(set(window.setting_widgets), {s.key for s in cls.settings})
                values = window.collect_backend_settings()
                for spec in cls.settings:
                    self.assertEqual(values[spec.key], spec.default)
                self.assertIn(window.selected_device(), {"cpu", "cuda"})
                self.assertEqual(window.speaker_labels_check is not None, cls.capabilities.speaker_labels)
                self.assertEqual(window.min_pause_spin.isEnabled(), cls.capabilities.word_timestamps)
                config = window.build_config([3, 4])
                self.assertEqual(config["backend_settings"], values)
                window.close()

    def test_tts_mode_hides_reading_controls(self):
        for backend_id in mb.BACKEND_SPECS:
            with self.subTest(backend=backend_id):
                window = core.SubtitleGeneratorGUI(backend_id)
                self.assertEqual(window.build_config([3, 4])["segmentation_mode"], "standard")
                self.assertFalse(window.pattern_edit.isHidden())
                window.segmentation_combo.setCurrentIndex(1)
                self.assertEqual(window.build_config([3, 4])["segmentation_mode"], "tts")
                self.assertTrue(window.pattern_edit.isHidden())
                self.assertTrue(window.min_duration_spin.isHidden())
                self.assertTrue(window.max_line_spin.isHidden())
                self.assertFalse(window.min_pause_spin.isHidden())
                self.assertEqual(window.min_pause_spin.isEnabled(), window.backend_cls.capabilities.word_timestamps)
                window.segmentation_combo.setCurrentIndex(0)
                self.assertFalse(window.pattern_edit.isHidden())
                self.assertFalse(window.max_line_spin.isHidden())
                window.close()

if __name__ == "__main__":
    unittest.main()
