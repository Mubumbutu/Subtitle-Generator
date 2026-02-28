#!/usr/bin/env python3
# subtitle_generator.py
import contextlib
import ctypes
import datetime
import functools
import io
import librosa
import logging
import numpy as np
import os
import pysrt
import queue
import shutil
import sounddevice as sd
import soundfile as sf
import string
import subprocess
import sys
import tempfile
import time
import torch
import torch.serialization
import traceback
import unicodedata
import warnings
import whisperx
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, QObject, QPointF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from threading import Thread

logging.getLogger("whisperx").setLevel(logging.WARNING)
logging.getLogger("whisperx.asr").setLevel(logging.WARNING)
logging.getLogger("whisperx.vads").setLevel(logging.WARNING)
logging.getLogger("whisperx.vads.pyannote").setLevel(logging.WARNING)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

warnings.filterwarnings("ignore", category=Warning, message=".*reproducibility.*")

warnings.filterwarnings("ignore", category=UserWarning, module="demucs")

_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

def seconds_to_srt_time(seconds: float) -> pysrt.SubRipTime:
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return pysrt.SubRipTime(
        hours=hours,
        minutes=minutes,
        seconds=secs,
        milliseconds=milliseconds
    )

def get_whisperx_model_path(model_name: str) -> Path:
    models_dir = Path("./models/whisperx")
    models_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace(".", "_").strip()
    if not safe_name:
        safe_name = "unknown_model"

    return models_dir / safe_name

def is_whisperx_model_cached(model_name: str) -> bool:
    model_path = get_whisperx_model_path(model_name)

    if not model_path.exists() or not model_path.is_dir():
        return False

    required_files = ["config.json", "model.bin"]

    optional_but_common = ["tokenizer.json", "vocabulary.txt", "preprocessor_config.json"]

    if not all((model_path / f).exists() for f in required_files):
        return False

    return True

class WaveformWidget(QWidget):
    view_changed = pyqtSignal(float, float)
    seek_requested = pyqtSignal(float)
    selections_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.overview_data = None
        self.overview_factor = 64

        self.duration = 0.0
        self.samplerate = 16000

        self.zoom_factor = 1.0
        self.offset = 0.0
        self.playback_position = 0.0

        self.last_mouse_x = 0
        self.dragging = False
        self.drag_started = False

        self.selections = []
        self.is_selecting = False
        self.is_erasing = False
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.selection_type = 'include'

        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px;")

    def set_audio_data(self, audio_array: np.ndarray | None):
        if audio_array is None or len(audio_array) == 0:
            self.audio_data = None
            self.overview_data = None
            self.duration = 0.0
            self.selections = []
            self.selections_changed.emit([])
            self.update()
            return

        self.audio_data = audio_array.astype(np.float32)
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            self.audio_data /= max_val

        self.duration = len(self.audio_data) / self.samplerate

        pad_size = (self.overview_factor - (len(self.audio_data) % self.overview_factor)) % self.overview_factor
        if pad_size > 0:
            padded = np.pad(self.audio_data, (0, pad_size), mode='constant')
        else:
            padded = self.audio_data

        reshaped = padded.reshape(-1, self.overview_factor)
        self.overview_data = np.max(np.abs(reshaped), axis=1)

        self.zoom_factor = 1.0
        self.offset = 0.0
        self.selections = []
        self.selections_changed.emit([])

        self.view_changed.emit(self.offset, self.zoom_factor)
        self.update()

    def merge_selections(self, new_start, new_end, new_type):
        if abs(new_end - new_start) < 0.01:
            return False

        start = min(new_start, new_end)
        end = max(new_start, new_end)

        overlapping_indices = []
        merge_tolerance = 0.3

        for i, (s, e, t) in enumerate(self.selections):
            if t == new_type:
                if not (end < s - merge_tolerance or start > e + merge_tolerance):
                    overlapping_indices.append(i)

        if overlapping_indices:
            merged_start = start
            merged_end = end

            for idx in overlapping_indices:
                s, e, _ = self.selections[idx]
                merged_start = min(merged_start, s)
                merged_end = max(merged_end, e)

            for idx in reversed(overlapping_indices):
                self.selections.pop(idx)

            self.selections.append((merged_start, merged_end, new_type))
            print(f"✓ Merged selection: {merged_start:.2f}s - {merged_end:.2f}s ({new_type})")
        else:
            self.selections.append((start, end, new_type))
            print(f"✓ Added selection: {start:.2f}s - {end:.2f}s ({new_type})")

        self.selections.sort(key=lambda x: x[0])

        return True

    def erase_from_selections(self, erase_start, erase_end):
        if abs(erase_end - erase_start) < 0.01:
            return False

        start = min(erase_start, erase_end)
        end = max(erase_start, erase_end)

        new_selections = []
        something_erased = False

        for sel_start, sel_end, sel_type in self.selections:
            if end <= sel_start or start >= sel_end:
                new_selections.append((sel_start, sel_end, sel_type))
            else:
                something_erased = True

                if start > sel_start:
                    left_part = (sel_start, start, sel_type)

                    if (start - sel_start) >= 0.01:
                        new_selections.append(left_part)

                if end < sel_end:
                    right_part = (end, sel_end, sel_type)

                    if (sel_end - end) >= 0.01:
                        new_selections.append(right_part)

        if something_erased:
            self.selections = new_selections
            self.selections.sort(key=lambda x: x[0])
            print(f"✓ Erased region: {start:.2f}s - {end:.2f}s")
            return True

        return False

    def set_playback_position(self, position: float):
        self.playback_position = position

        visible_duration = self.duration / self.zoom_factor
        start_time = self.offset * self.duration
        end_time = start_time + visible_duration

        if position > end_time or position < start_time:
             new_offset = (position - visible_duration/2) / self.duration
             self.offset = max(0.0, min(new_offset, 1.0 - 1.0/self.zoom_factor))
             self.view_changed.emit(self.offset, self.zoom_factor)

        self.update()

    def set_view_offset(self, offset_ratio):
        max_offset = 1.0 - 1.0 / self.zoom_factor
        self.offset = max(0.0, min(offset_ratio, max_offset))
        self.update()

    def _get_time_from_x(self, x):
        if self.audio_data is None: return 0.0
        width = self.width()
        visible_duration = self.duration / self.zoom_factor
        start_time = self.offset * self.duration
        return start_time + (x / width) * visible_duration

    def wheelEvent(self, event):
        if self.audio_data is None: return

        angle = event.angleDelta().y()
        factor = 1.2 if angle > 0 else 1/1.2
        new_zoom = max(1.0, min(self.zoom_factor * factor, 300.0))

        if new_zoom == self.zoom_factor: return

        mouse_x = event.position().x()
        width = self.width()
        mouse_ratio = mouse_x / width

        current_view_size = 1.0 / self.zoom_factor
        mouse_time_pos = self.offset + (mouse_ratio * current_view_size)

        self.zoom_factor = new_zoom
        new_view_size = 1.0 / self.zoom_factor

        self.offset = mouse_time_pos - (mouse_ratio * new_view_size)
        max_offset = 1.0 - new_view_size
        self.offset = max(0.0, min(self.offset, max_offset))

        self.view_changed.emit(self.offset, self.zoom_factor)
        self.update()

    def mousePressEvent(self, event):
        if self.audio_data is None: return

        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                has_include = any(sel_type == 'include' for _, _, sel_type in self.selections)
                has_exclude = any(sel_type == 'exclude' for _, _, sel_type in self.selections)

                if has_exclude:
                    self.is_erasing = True
                    self.selection_type = 'exclude'
                    self.selection_start = self._get_time_from_x(event.position().x())
                    self.selection_end = self.selection_start
                    self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                    self.update()
                else:
                    self.is_selecting = True
                    self.selection_type = 'include'
                    self.selection_start = self._get_time_from_x(event.position().x())
                    self.selection_end = self.selection_start
                    self.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
                    self.update()
            else:
                self.dragging = True
                self.drag_started = False
                self.last_mouse_x = event.position().x()
                self.last_press_time = self._get_time_from_x(event.position().x())
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

        elif event.button() == Qt.MouseButton.RightButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                has_include = any(sel_type == 'include' for _, _, sel_type in self.selections)
                has_exclude = any(sel_type == 'exclude' for _, _, sel_type in self.selections)

                if has_include:
                    self.is_erasing = True
                    self.selection_type = 'include'
                    self.selection_start = self._get_time_from_x(event.position().x())
                    self.selection_end = self.selection_start
                    self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                    self.update()
                else:
                    self.is_selecting = True
                    self.selection_type = 'exclude'
                    self.selection_start = self._get_time_from_x(event.position().x())
                    self.selection_end = self.selection_start
                    self.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
                    self.update()

    def mouseMoveEvent(self, event):
        if self.audio_data is None: return

        if self.is_selecting or self.is_erasing:
            current_time = self._get_time_from_x(event.position().x())

            current_time = max(0.0, min(current_time, self.duration))
            self.selection_end = current_time
            self.update()
            return

        if self.dragging:
            x = event.position().x()
            dx = x - self.last_mouse_x

            if abs(dx) > 3:
                self.drag_started = True

            if dx != 0:
                width = self.width()
                view_size = 1.0 / self.zoom_factor
                move_ratio = -(dx / width) * view_size

                self.offset += move_ratio
                max_offset = 1.0 - view_size
                self.offset = max(0.0, min(self.offset, max_offset))

                self.last_mouse_x = x
                self.view_changed.emit(self.offset, self.zoom_factor)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
            if self.is_erasing:
                self.is_erasing = False
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

                if self.erase_from_selections(self.selection_start, self.selection_end):
                    self.selections_changed.emit(self.selections)

                self.update()
                return

            if self.is_selecting:
                self.is_selecting = False
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

                if self.merge_selections(self.selection_start, self.selection_end, self.selection_type):
                    self.selections_changed.emit(self.selections)

                self.update()
                return

            if event.button() == Qt.MouseButton.LeftButton and self.dragging:
                self.dragging = False
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

                if not self.drag_started:
                    click_time = self.last_press_time
                    self.seek_requested.emit(click_time)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        w, h = self.width(), self.height()
        mid_y = h / 2

        if self.audio_data is None:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No audio data loaded")
            return

        total_samples = len(self.audio_data)
        visible_samples = int(total_samples / self.zoom_factor)
        start_sample = int(self.offset * total_samples)

        if start_sample < 0: start_sample = 0
        end_sample = start_sample + visible_samples
        if end_sample > total_samples: end_sample = total_samples

        real_visible_count = end_sample - start_sample

        if real_visible_count > 0:
            samples_per_pixel = real_visible_count / max(1, w)
            use_overview = (samples_per_pixel > self.overview_factor) and (self.overview_data is not None)

            if use_overview:
                ov_start = start_sample // self.overview_factor
                ov_end = end_sample // self.overview_factor
                source_data = self.overview_data[ov_start:ov_end]
            else:
                source_data = self.audio_data[start_sample:end_sample]

            if len(source_data) > w:
                bin_size = len(source_data) // w
                if bin_size < 1: bin_size = 1
                limit = (len(source_data) // bin_size) * bin_size
                data_to_bin = source_data[:limit]
                binned = data_to_bin.reshape(-1, bin_size)
                plot_data = np.max(np.abs(binned), axis=1)
            else:
                plot_data = np.abs(source_data)

            count = len(plot_data)
            if count > 0:
                x_indices = np.linspace(0, w, count)
                heights = plot_data * (h / 2 * 0.95)

                painter.setPen(QPen(QColor(0, 255, 0, 180), 1))

                for i in range(count):
                    x = int(x_indices[i])
                    amp = int(heights[i])
                    if amp < 1: amp = 1
                    painter.drawLine(x, int(mid_y - amp), x, int(mid_y + amp))

                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.drawLine(0, int(mid_y), w, int(mid_y))

        painter.setPen(QColor("#aaaaaa"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        visible_duration = self.duration / self.zoom_factor
        view_start_time = self.offset * self.duration

        if visible_duration < 5:      time_step = 0.5
        elif visible_duration < 10:   time_step = 1.0
        elif visible_duration < 30:   time_step = 5.0
        elif visible_duration < 60:   time_step = 10.0
        elif visible_duration < 300:  time_step = 30.0
        else:                         time_step = 60.0

        first_tick = (int(view_start_time / time_step) + 1) * time_step
        current_tick = first_tick

        while current_tick < view_start_time + visible_duration:
            time_offset = current_tick - view_start_time
            tick_x = int((time_offset / visible_duration) * w)

            if 0 <= tick_x <= w:
                if time_step < 1.0: val = f"{current_tick:.1f}s"
                else:
                    m, s = divmod(int(current_tick), 60)
                    val = f"{m:02d}:{s:02d}"

                painter.drawLine(tick_x, h, tick_x, h - 10)
                painter.drawText(tick_x + 3, h - 2, val)

            current_tick += time_step

        regions_to_draw = list(self.selections)
        if self.is_selecting:
            regions_to_draw.append((min(self.selection_start, self.selection_end),
                                    max(self.selection_start, self.selection_end),
                                    self.selection_type))

        if self.is_erasing:
            erase_start = min(self.selection_start, self.selection_end)
            erase_end = max(self.selection_start, self.selection_end)

            if erase_end > view_start_time and erase_start < (view_start_time + visible_duration):
                s_offset = erase_start - view_start_time
                e_offset = erase_end - view_start_time

                x_start = int((s_offset / visible_duration) * w)
                x_end = int((e_offset / visible_duration) * w)
                width_rect = x_end - x_start
                if width_rect < 1: width_rect = 1

                painter.setBrush(QColor(128, 128, 128, 80))
                painter.setPen(QPen(QColor(200, 200, 200, 150), 2, Qt.PenStyle.DashLine))
                painter.drawRect(x_start, 0, width_rect, h)
                painter.setPen(Qt.PenStyle.NoPen)

        for r_start, r_end, r_type in regions_to_draw:
            if r_end < view_start_time or r_start > (view_start_time + visible_duration):
                continue

            s_offset = r_start - view_start_time
            e_offset = r_end - view_start_time

            x_start = int((s_offset / visible_duration) * w)
            x_end = int((e_offset / visible_duration) * w)

            width_rect = x_end - x_start
            if width_rect < 1: width_rect = 1

            if r_type == 'include':
                painter.setBrush(QColor(75, 0, 130, 120))
                edge_color = QColor(148, 0, 211, 200)
            else:
                painter.setBrush(QColor(139, 0, 0, 120))
                edge_color = QColor(220, 20, 60, 200)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x_start, 0, width_rect, h)

            painter.setPen(edge_color)
            painter.drawLine(x_start, 0, x_start, h)
            painter.drawLine(x_end, 0, x_end, h)
            painter.setPen(Qt.PenStyle.NoPen)

        if view_start_time <= self.playback_position <= view_start_time + visible_duration:
            cursor_rel = (self.playback_position - view_start_time) / visible_duration
            cursor_x = int(cursor_rel * w)

            painter.setPen(QPen(QColor("#ff0000"), 1))
            painter.drawLine(cursor_x, 0, cursor_x, h)
            painter.setBrush(QColor("#ff0000"))
            painter.drawPolygon([
                QPointF(cursor_x, 0),
                QPointF(cursor_x - 5, 8),
                QPointF(cursor_x + 5, 8)
            ])

class AudioRecorder(QObject):
    vumeter_signal = pyqtSignal(float)
    finished_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recording = False
        self.paused = False
        self.filename = None
        self.q = queue.Queue()
        self.samplerate = 16000
        self.stream = None
        self.thread = None

    def callback(self, indata, frames, time, status):
        if self.recording and not self.paused:
            self.q.put(indata.copy())

            volume_norm = torch.linalg.norm(torch.from_numpy(indata)).item()
            self.vumeter_signal.emit(volume_norm)

    def start_recording(self):
        self.recording = True
        self.paused = False
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = Path(tempfile.gettempdir()) / f"recording_{timestamp}.wav"

        print(f"Recording will be saved to: {self.filename}")

        def run():
            try:
                print(f"Opening audio file: {self.filename}")
                with sf.SoundFile(self.filename, mode='x', samplerate=self.samplerate, channels=1) as file:
                    print(f"Opening input stream...")
                    with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.callback) as stream:
                        self.stream = stream
                        print(f"✓ Recording started")

                        while self.recording:
                            try:
                                data = self.q.get(timeout=0.1)
                                file.write(data)
                            except queue.Empty:
                                continue

                        print(f"Recording loop ended")

                print(f"✓ Audio file closed")
                print(f"File size: {self.filename.stat().st_size} bytes")

                print(f"Emitting finished_signal with path: {self.filename}")
                self.finished_signal.emit(str(self.filename))
                print(f"✓ Signal emitted")

            except Exception as e:
                print(f"ERROR in recording thread: {e}")
                traceback.print_exc()

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()
        print(f"Recording thread started")

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"Recording {'PAUSED' if self.paused else 'RESUMED'}")
        return self.paused

    def stop_recording(self):
        print(f"stop_recording() called")
        self.recording = False

        if self.thread and self.thread.is_alive():
            print(f"Waiting for recording thread to finish...")
            self.thread.join(timeout=3.0)
            if self.thread.is_alive():
                print(f"WARNING: Recording thread did not finish in time")
            else:
                print(f"✓ Recording thread finished")

class SubtitleGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.voice_separation_enabled = config.get("voice_separation", False)

        self.temp_vocals_file = None
        self.temp_demucs_dir = None

        self.last_segments = None

        if config.get("enable_tf32", False) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.log("  ✓ TF32 enabled (faster mode for Ampere+ GPUs)")

    def log(self, message):
        print(message, flush=True)

    def check_demucs_installed(self):
        try:
            result = subprocess.run(
                ["demucs", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def separate_vocals(self, audio_path):
        try:
            self.log("\n" + "="*50)
            self.log("🎵 VOICE SEPARATION (Demucs)")
            self.log("="*50)

            if not self.check_demucs_installed():
                self.log("  ⚠️ ERROR: Demucs not found!")
                self.log("  Install with: pip install demucs")
                self.log("  Continuing with original audio...")
                return audio_path

            self.log(f"▶ Input file: {audio_path.name}")
            self.log(f"▶ File size: {audio_path.stat().st_size / (1024*1024):.2f} MB")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(tempfile.gettempdir()) / f"demucs_output_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            self.log(f"▶ Temporary output: {output_dir}")

            def safe_filename(filename):
                name = Path(filename).stem

                name = unicodedata.normalize('NFKD', name)
                name = name.encode('ASCII', 'ignore').decode('ASCII')

                safe_chars = string.ascii_letters + string.digits + '-_'
                name = ''.join(c if c in safe_chars else '_' for c in name)

                while '__' in name:
                    name = name.replace('__', '_')

                name = name.strip('_')

                if not name:
                    name = 'audio'

                name = name[:100]

                return name

            safe_name = safe_filename(audio_path.name)
            temp_audio_dir = Path(tempfile.gettempdir()) / f"demucs_input_{timestamp}"
            temp_audio_dir.mkdir(parents=True, exist_ok=True)

            temp_audio = temp_audio_dir / f"{safe_name}{audio_path.suffix}"

            self.log(f"▶ Creating safe temporary copy...")
            self.log(f"  Original: {audio_path.name}")
            self.log(f"  Safe name: {temp_audio.name}")

            shutil.copy2(audio_path, temp_audio)

            self.temp_input_dir = temp_audio_dir

            cmd = [
                "demucs",
                "--two-stems=vocals",
                "--mp3",
                "--mp3-bitrate", "320",
                "-o", str(output_dir),
                str(temp_audio)
            ]

            self.log("▶ Running Demucs separation...")
            self.log(f"  Command: {' '.join(cmd)}")
            self.log("  This may take 2-5 minutes depending on file length...")

            start_time = datetime.datetime.now()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            stderr_lines = []

            for line in process.stderr:
                line_stripped = line.strip()
                stderr_lines.append(line)
                if line_stripped and any(keyword in line_stripped.lower() for keyword in ['progress', 'processing', '%', 'separated']):
                    self.log(f"  {line_stripped}")

            process.wait()

            if process.returncode != 0:
                stderr_output = ''.join(stderr_lines)
                self.log("\n  Full Demucs error output:")
                self.log("  " + "─" * 50)
                for line in stderr_lines[-20:]:
                    self.log(f"  {line.rstrip()}")
                self.log("  " + "─" * 50)
                raise RuntimeError(f"Demucs failed with code {process.returncode}\n\nError details:\n{stderr_output}")

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            self.log(f"  ✓ Demucs completed in {elapsed:.1f}s")

            self.log("▶ Locating vocals file...")

            vocals_file = None

            for model_dir in output_dir.iterdir():
                if model_dir.is_dir():
                    self.log(f"  Checking model directory: {model_dir.name}")
                    audio_dir = model_dir / safe_name
                    if audio_dir.exists():
                        vocals_path = audio_dir / "vocals.mp3"
                        if vocals_path.exists():
                            vocals_file = vocals_path
                            self.log(f"  ✓ Found vocals: {vocals_path}")
                            break

            if not vocals_file:
                self.log("  ⚠️ WARNING: Vocals file not found in expected location")
                self.log(f"  Searched in: {output_dir}")

                self.log("  Directory contents:")
                for item in output_dir.rglob("*"):
                    if item.is_file():
                        self.log(f"    - {item.relative_to(output_dir)}")
                raise FileNotFoundError("Vocals file not found after separation")

            vocals_size = vocals_file.stat().st_size / (1024*1024)
            self.log(f"  ✓ Vocals file size: {vocals_size:.2f} MB")
            self.log(f"  Using MP3 directly (WhisperX supports MP3 natively)")

            self.temp_vocals_file = vocals_file
            self.temp_demucs_dir = output_dir

            self.log("="*50)
            self.log("✓ VOICE SEPARATION COMPLETE")
            self.log("="*50 + "\n")

            return vocals_file

        except Exception as e:
            self.log("\n" + "="*50)
            self.log("⚠️ VOICE SEPARATION FAILED")
            self.log("="*50)
            self.log(f"Error: {str(e)}")
            self.log("\nFull traceback:")
            self.log(traceback.format_exc())
            self.log("="*50)
            self.log("Continuing with original audio file...")
            self.log("="*50 + "\n")
            return audio_path

    def cleanup_temp_files(self):
        self.log("\n▶ Cleaning up temporary files...")

        cleaned = False

        if self.temp_vocals_file and self.temp_vocals_file.exists():
            try:
                if not self.temp_demucs_dir or self.temp_demucs_dir not in self.temp_vocals_file.parents:
                    self.temp_vocals_file.unlink()
                    self.log(f"  ✓ Removed: {self.temp_vocals_file.name}")
                    cleaned = True
            except Exception as e:
                self.log(f"  ⚠️ Could not remove {self.temp_vocals_file.name}: {e}")

        if self.temp_demucs_dir and self.temp_demucs_dir.exists():
            try:
                shutil.rmtree(self.temp_demucs_dir)
                self.log(f"  ✓ Removed: {self.temp_demucs_dir.name}/ (includes vocals.mp3)")
                cleaned = True
            except Exception as e:
                self.log(f"  ⚠️ Could not remove {self.temp_demucs_dir.name}: {e}")

        if hasattr(self, 'temp_input_dir') and self.temp_input_dir and self.temp_input_dir.exists():
            try:
                shutil.rmtree(self.temp_input_dir)
                self.log(f"  ✓ Removed: {self.temp_input_dir.name}/ (temporary input copy)")
                cleaned = True
            except Exception as e:
                self.log(f"  ⚠️ Could not remove {self.temp_input_dir.name}: {e}")

        if not cleaned:
            self.log("  (No temporary files to clean)")

    def transcribe(self, audio_path, model_size):
        if self.device == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability(0)
                compute_type = "float16" if major >= 7 else "float32"
            except Exception:
                compute_type = "float32"
        else:
            compute_type = "int8"

        model_cache_dir = get_whisperx_model_path(model_size)
        is_cached = is_whisperx_model_cached(model_size)

        if is_cached:
            self.log(f"✓ Using locally cached WhisperX model: {model_cache_dir}")
            local_files_only = True
        else:
            self.log(f"⚠️ Model '{model_size}' not found locally or incomplete.")
            self.log(f"   Downloading from Hugging Face to: {model_cache_dir}")
            self.log(f"   This may take several minutes (large-v3 ≈ 3–5 GB, depending on connection).")
            local_files_only = False

        self.log(
            f"▶ Loading WhisperX model ({model_size}) "
            f"on {self.device.upper()} (compute_type={compute_type})"
        )

        with contextlib.redirect_stderr(io.StringIO()):
            model = whisperx.load_model(
                model_size,
                self.device,
                compute_type=compute_type,
                download_root=str(model_cache_dir),
                local_files_only=local_files_only
            )

        self.log(f"✓ Model loaded successfully")

        self.log("▶ Transcribing audio...")

        transcribe_kwargs = {
            "batch_size": self.config["batch_size"]
        }

        language = self.config.get("language", "").strip()
        if language:
            transcribe_kwargs["language"] = language
            self.log(f" Language: {language}")
        else:
            self.log(" Language: auto-detect")

        result = model.transcribe(str(audio_path), **transcribe_kwargs)

        detected_language = result.get("language", "unknown")
        self.log(f" Detected language: {detected_language}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.log("▶ Aligning word timestamps...")

        try:
            align_cache_dir = Path("./models/whisperx/align") / detected_language
            align_cache_dir.mkdir(parents=True, exist_ok=True)

            with contextlib.redirect_stderr(io.StringIO()):
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device,
                    model_dir=str(align_cache_dir)
                )

            audio = whisperx.load_audio(str(audio_path))

            aligned_result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            result["segments"] = aligned_result["segments"]
            self.log(" ✓ Alignment successful")

            del align_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            self.log(f" ⚠️ Warning: Alignment failed ({str(e)})")
            self.log(" Continuing with unaligned timestamps...")

        return result

    def extract_words(self, segments):
        words = []
        for seg in segments:
            if "words" in seg:
                for w in seg["words"]:
                    if w.get("start") is None or w.get("end") is None:
                        continue
                    words.append({
                        "word": w["word"].strip(),
                        "start": w["start"],
                        "end": w["end"]
                    })
            else:
                if seg.get("start") is not None and seg.get("end") is not None:
                    text = seg.get("text", "").strip()
                    if text:
                        word_list = text.split()
                        duration = seg["end"] - seg["start"]
                        time_per_word = duration / len(word_list) if word_list else 0

                        for i, word in enumerate(word_list):
                            start = seg["start"] + (i * time_per_word)
                            end = start + time_per_word
                            words.append({
                                "word": word,
                                "start": start,
                                "end": end
                            })

        return words

    def create_subtitles(self, words):
        subs = pysrt.SubRipFile()
        word_idx = 0
        pattern_idx = 0
        sub_idx = 1

        word_pattern = self.config["word_pattern"]
        pattern_len = len(word_pattern)
        min_pause = self.config["min_pause"]
        max_line_length = self.config["max_line_length"]

        while word_idx < len(words):
            max_words = word_pattern[pattern_idx % pattern_len]
            pattern_idx += 1

            chunk = []
            start_time = None
            end_time = None

            for _ in range(max_words):
                if word_idx >= len(words):
                    break

                w = words[word_idx]

                if chunk:
                    pause = w["start"] - chunk[-1]["end"]
                    if pause > min_pause:
                        break

                chunk.append(w)
                start_time = start_time or w["start"]
                end_time = w["end"]
                word_idx += 1

            if not chunk:
                continue

            text = " ".join(w["word"] for w in chunk)
            text = self.split_lines(text, max_line_length)

            subs.append(
                pysrt.SubRipItem(
                    index=sub_idx,
                    start=seconds_to_srt_time(start_time),
                    end=seconds_to_srt_time(end_time),
                    text=text
                )
            )
            sub_idx += 1

        return self.merge_short_subs(subs)

    def merge_short_subs(self, subs):
        merged = pysrt.SubRipFile()
        i = 0
        min_duration = self.config["min_duration"]

        while i < len(subs):
            cur = subs[i]
            duration_ms = cur.end.ordinal - cur.start.ordinal

            can_merge = (
                duration_ms < min_duration * 1000
                and i + 1 < len(subs)
            )

            if can_merge:
                nxt = subs[i + 1]
                cur.text += " " + nxt.text
                cur.end = nxt.end
                merged.append(cur)
                i += 2
            else:
                merged.append(cur)
                i += 1

        for idx, s in enumerate(merged, 1):
            s.index = idx

        return merged

    def split_lines(self, text, max_line_length):
        words = text.split()
        lines = []
        current = []

        for w in words:
            test = " ".join(current + [w])
            if len(test) <= max_line_length:
                current.append(w)
            else:
                if current:
                    lines.append(" ".join(current))
                if len(w) > max_line_length:
                    lines.append(w)
                    current = []
                else:
                    current = [w]

        if current:
            lines.append(" ".join(current))

        return "\n".join(lines[:2])

    def export_text(self, segments, output_path):
        self.log(f"\n▶ Exporting plain text transcription...")

        text_lines = []

        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                text_lines.append(text)

        full_text = " ".join(text_lines)

        formatted_text = []
        current_paragraph = []

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            current_paragraph.append(text)

            if i + 1 < len(segments):
                current_end = seg.get("end", 0)
                next_start = segments[i + 1].get("start", 0)
                pause = next_start - current_end

                if pause > 2.0:
                    formatted_text.append(" ".join(current_paragraph))
                    current_paragraph = []

        if current_paragraph:
            formatted_text.append(" ".join(current_paragraph))

        final_text = "\n\n".join(formatted_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        word_count = len(final_text.split())
        char_count = len(final_text)
        paragraph_count = len(formatted_text)

        self.log(f"  ✓ Exported {word_count} words ({char_count} characters)")
        self.log(f"  ✓ Formatted into {paragraph_count} paragraphs")
        self.log(f"  ✓ Saved to: {output_path}")

    def process(self, audio_path, output_path, model_size, include_ranges=None, exclude_ranges=None, output_format='srt'):
        global processing_start_time
        processing_start_time = datetime.datetime.now()

        self.log(f"\n▶ Starting processing: {audio_path.name}")
        self.log(f"   Output format: {output_format.upper()}")

        has_include = include_ranges and len(include_ranges) > 0
        has_exclude = exclude_ranges and len(exclude_ranges) > 0

        if has_include and has_exclude:
            self.log("   ⚠️ WARNING: Both include and exclude ranges specified!")
            self.log("   ⚠️ Include ranges take priority - exclude ranges will be ignored.")
            has_exclude = False
            exclude_ranges = None

        if has_include:
            self.log(f"   Mode: INCLUDE {len(include_ranges)} selected regions")
        elif has_exclude:
            self.log(f"   Mode: EXCLUDE {len(exclude_ranges)} selected regions")
        else:
            self.log(f"   Mode: PROCESS ENTIRE FILE")

        processed_audio_path = audio_path
        temp_processed_file = None

        needs_processing = (output_format == 'txt' and (has_include or has_exclude)) or \
                           (output_format == 'srt' and has_include)

        if needs_processing:
            self.log(f"▶ Processing audio regions...")
            temp_processed_file = Path(tempfile.gettempdir()) / f"processed_{audio_path.stem}.wav"

            try:

                audio_data, sr = sf.read(str(audio_path), dtype='float32')

                if output_format == 'txt' or (output_format == 'srt' and has_include):
                    if has_include:
                        segments = []
                        for start, end in sorted(include_ranges, key=lambda x: x[0]):
                            start_sample = int(start * sr)
                            end_sample = int(end * sr)
                            segments.append(audio_data[start_sample:end_sample])

                        processed_audio = np.concatenate(segments)
                        self.log(f"   ✓ Extracted {len(include_ranges)} regions")

                    elif has_exclude:
                        mask = np.ones(len(audio_data), dtype=bool)
                        for start, end in exclude_ranges:
                            start_sample = int(start * sr)
                            end_sample = int(end * sr)
                            mask[start_sample:end_sample] = False

                        processed_audio = audio_data[mask]
                        self.log(f"   ✓ Excluded {len(exclude_ranges)} regions")

                elif output_format == 'srt' and has_exclude:
                    processed_audio = audio_data.copy()
                    for start, end in exclude_ranges:
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        processed_audio[start_sample:end_sample] = 0.0

                    self.log(f"   ✓ Muted {len(exclude_ranges)} regions")

                sf.write(str(temp_processed_file), processed_audio, sr)
                processed_audio_path = temp_processed_file
                self.log(f"   ✓ Saved processed audio: {temp_processed_file.name}")

            except Exception as e:
                self.log(f"   ⚠️ Error processing audio: {e}")
                raise

        if self.voice_separation_enabled:
            processed_audio_path = self.separate_vocals(processed_audio_path)

        result = self.transcribe(processed_audio_path, model_size)
        segments = result["segments"]

        self.last_segments = segments

        if not segments:
            raise RuntimeError("No segments extracted from audio")

        if output_format == 'txt':
            self.export_text(segments, output_path)

            if temp_processed_file and temp_processed_file.exists():
                temp_processed_file.unlink()
            self.cleanup_temp_files()

            processing_time = (datetime.datetime.now() - processing_start_time).total_seconds()
            self.log(f"\n✓ Text export completed in {processing_time:.1f}s")

            return None

        else:
            words = self.extract_words(segments)

            if not words:
                raise RuntimeError("No words extracted — alignment may have failed")

            self.log(f"▶ Extracted {len(words)} words")

            subs = self.create_subtitles(words)

            subs.save(output_path, encoding="utf-8")

            total_duration = sum((s.end.ordinal - s.start.ordinal) / 1000 for s in subs)
            avg_duration = total_duration / len(subs) if subs else 0
            processing_time = (datetime.datetime.now() - processing_start_time).total_seconds()

            self.log(f"\n✓ Saved {len(subs)} subtitles → {output_path}")
            self.log(f"  Total subtitle time: {total_duration:.1f}s")
            self.log(f"  Average subtitle: {avg_duration:.2f}s")
            self.log(f"  Total processing time: {processing_time:.1f}s")

            if temp_processed_file and temp_processed_file.exists():
                temp_processed_file.unlink()
            self.cleanup_temp_files()

            return subs

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    text_exported = pyqtSignal(str)

class SubtitleGeneratorGUI(QMainWindow):
    audio_loaded_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subtitle Generator by Mubumbutu")
        self.setMinimumSize(950, 300)

        self.input_file = None
        self.output_file = None
        self.processing = False
        self.generator = None

        self.audio_data = None
        self.playing = False
        self.current_position = 0
        self.playback_thread = None
        self.stop_playback_flag = False

        self._updating_from_waveform = False

        self.audio_loaded_signal.connect(self._update_waveform_data)

        if parent is None:
            self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(self.create_main_tab())

        self.button_container = QWidget()
        button_layout = QHBoxLayout(self.button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.process_btn = QPushButton("🚀 Generate Subtitles (.srt)")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1B5E20;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_btn)

        self.export_text_btn = QPushButton("📄 Export as Text (.txt)")
        self.export_text_btn.setMinimumHeight(40)
        self.export_text_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.export_text_btn.clicked.connect(self.start_text_export)
        button_layout.addWidget(self.export_text_btn)

        self.button_container.setVisible(False)
        layout.addWidget(self.button_container)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.adjustSize()

    def create_main_tab(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #1a1a1a; color: #cccccc;")
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        GROUPBOX_STYLE = """
            QGroupBox {
                background-color: #1c1c1c;
                border: 1px solid #2e2e2e;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
                color: #aaaaaa;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #888888;
                font-size: 11px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
        """

        BTN_STYLE = """
            QPushButton {
                background-color: #2a2a2a;
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 14px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #3a3a3a; border-color: #666; color: white; }
            QPushButton:pressed { background-color: #1a1a1a; }
            QPushButton:disabled { background-color: #222; color: #555; border-color: #333; }
        """

        LABEL_STYLE = "color: #888888; font-size: 11px;"

        INPUT_STYLE = """
            QLineEdit {
                background-color: #161616;
                color: #e0e0e0;
                border: 1px solid #2e2e2e;
                border-radius: 3px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QLineEdit:focus { border-color: #2a6aaa; }
        """

        PROGRESSBAR_STYLE = """
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 3px;
                height: 8px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk { background-color: #2a6aaa; border-radius: 2px; }
        """

        RADIO_STYLE = """
            QRadioButton {
                color: #cccccc;
                font-size: 12px;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 8px;
                border: 2px solid #555555;
                background-color: #1e1e1e;
            }
            QRadioButton::indicator:hover {
                border-color: #2a6aaa;
                background-color: #252525;
            }
            QRadioButton::indicator:checked {
                border-color: #2a6aaa;
                background-color: #2a6aaa;
            }
            QRadioButton::indicator:disabled {
                border-color: #333;
                background-color: #1a1a1a;
            }
        """

        CHECKBOX_STYLE = """
            QCheckBox {
                color: #cccccc;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #555555;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:hover {
                border-color: #2a6aaa;
                background-color: #252535;
            }
            QCheckBox::indicator:checked {
                border-color: #2a6aaa;
                background-color: #1a4a7a;
                image: url(none);
            }
            QCheckBox::indicator:checked:hover {
                background-color: #2a5a9a;
            }
        """

        SPINBOX_STYLE = """
            QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 12px;
                min-height: 26px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px; height: 13px;
                background-color: #333;
                border-left: 1px solid #444;
                border-bottom: 1px solid #444;
                border-top-right-radius: 3px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px; height: 13px;
                background-color: #333;
                border-left: 1px solid #444;
                border-bottom-right-radius: 3px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #2a5a9a;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #1a3a6a;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none; width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #aaa;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none; width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #aaa;
            }
        """

        COMBO_STYLE = """
            QComboBox {
                padding: 4px 8px;
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #252525;
                color: white;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: white;
                selection-background-color: #3a5a8a;
            }
        """

        TOGGLE_BTN_STYLE = """
            QPushButton {
                background-color: #2e2e2e;
                color: #cccccc;
                padding: 9px 14px;
                border-radius: 4px;
                border: 1px solid #444;
                text-align: left;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:checked { background-color: #1e3a5f; border-color: #2a6aaa; color: white; }
            QPushButton:hover { background-color: #3a3a3a; color: white; }
        """

        def make_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(LABEL_STYLE)
            return lbl

        file_group = QGroupBox("Input / Output & Recording")
        file_group.setStyleSheet(GROUPBOX_STYLE)
        file_layout = QVBoxLayout()
        file_layout.setSpacing(8)
        file_layout.setContentsMargins(10, 14, 10, 10)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)
        input_layout.addWidget(make_label("Input File:"))

        self.input_label = QLabel("Drag & drop or use Record / Browse")
        self.input_label.setStyleSheet("""
            QLabel {
                color: #555555;
                background-color: #161616;
                border: 1px dashed #2e2e2e;
                border-radius: 3px;
                padding: 5px 8px;
                font-style: italic;
                font-size: 12px;
            }
        """)
        input_layout.addWidget(self.input_label, 1)

        self.unload_btn = QPushButton("×")
        self.unload_btn.setFixedWidth(28)
        self.unload_btn.setMinimumHeight(28)
        self.unload_btn.setToolTip("Clear file")
        self.unload_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a1a1a;
                color: #ff8888;
                font-weight: bold;
                border-radius: 4px;
                border: 1px solid #882222;
                font-size: 16px;
                padding: 0px;
            }
            QPushButton:hover { background-color: #aa2222; color: white; }
        """)
        self.unload_btn.clicked.connect(self.unload_file)
        self.unload_btn.setVisible(False)
        input_layout.addWidget(self.unload_btn)

        self.browse_btn = QPushButton("📁 Browse...")
        self.browse_btn.setStyleSheet(BTN_STYLE)
        self.browse_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.browse_btn)
        file_layout.addLayout(input_layout)

        rec_layout = QHBoxLayout()
        rec_layout.setSpacing(6)
        rec_layout.addWidget(make_label("Quick Record:"))

        self.rec_btn = QPushButton("🎤 Start Record")
        self.rec_btn.setFixedWidth(130)
        self.rec_btn.setStyleSheet(BTN_STYLE)
        self.rec_btn.clicked.connect(self.toggle_recording)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(BTN_STYLE)
        self.pause_btn.clicked.connect(self.toggle_rec_pause)

        self.rec_vol_bar = QProgressBar()
        self.rec_vol_bar.setMaximum(100)
        self.rec_vol_bar.setTextVisible(False)
        self.rec_vol_bar.setFixedHeight(8)
        self.rec_vol_bar.setStyleSheet(PROGRESSBAR_STYLE)

        rec_layout.addWidget(self.rec_btn)
        rec_layout.addWidget(self.pause_btn)
        rec_layout.addWidget(make_label("Level:"))
        rec_layout.addWidget(self.rec_vol_bar)
        file_layout.addLayout(rec_layout)

        output_layout = QHBoxLayout()
        output_layout.setSpacing(6)
        output_layout.addWidget(make_label("Output File:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Auto-generated (same name as input, .srt extension)")
        self.output_edit.setStyleSheet(INPUT_STYLE)
        output_layout.addWidget(self.output_edit, 1)
        self.browse_output_btn = QPushButton("💾 Browse...")
        self.browse_output_btn.setStyleSheet(BTN_STYLE)
        self.browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.browse_output_btn)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        self.waveform_group = QGroupBox("Audio Waveform & Playback")
        self.waveform_group.setStyleSheet(GROUPBOX_STYLE)
        waveform_layout = QVBoxLayout()
        waveform_layout.setSpacing(6)
        waveform_layout.setContentsMargins(10, 14, 10, 10)

        playback_layout = QHBoxLayout()
        playback_layout.setSpacing(6)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedWidth(85)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e5fa8, stop:1 #133f74);
                color: white; font-weight: bold; font-size: 12px;
                padding: 5px 10px; border-radius: 4px; border: 1px solid #2a6aaa;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2a72c8, stop:1 #1a4e8c); }
            QPushButton:pressed { background: #0e3060; }
        """)
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFixedWidth(85)
        self.stop_btn.setStyleSheet(BTN_STYLE)
        self.stop_btn.clicked.connect(self.stop_playback)
        playback_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("⏮ Reset")
        self.reset_btn.setFixedWidth(85)
        self.reset_btn.setStyleSheet(BTN_STYLE)
        self.reset_btn.clicked.connect(self.reset_playback)
        playback_layout.addWidget(self.reset_btn)

        playback_layout.addWidget(make_label("Position:"))
        self.playback_position_label = QLabel("0.00s")
        self.playback_position_label.setStyleSheet("color: #e0e0e0; font-size: 12px; font-weight: bold;")
        playback_layout.addWidget(self.playback_position_label)
        playback_layout.addStretch()
        waveform_layout.addLayout(playback_layout)

        self.waveform_display = WaveformWidget()
        self.waveform_display.view_changed.connect(self.update_waveform_scrollbar)
        self.waveform_display.seek_requested.connect(self.on_seek_requested)
        self.waveform_display.selections_changed.connect(self.on_waveform_selection_changed)
        waveform_layout.addWidget(self.waveform_display)

        self.waveform_scrollbar_container = QWidget()
        scrollbar_layout = QHBoxLayout(self.waveform_scrollbar_container)
        scrollbar_layout.setContentsMargins(0, 0, 0, 0)

        self.waveform_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.waveform_scroll.setMinimum(0)
        self.waveform_scroll.setMaximum(1000)
        self.waveform_scroll.setValue(0)
        self.waveform_scroll.setStyleSheet("""
            QScrollBar:horizontal {
                background-color: #1a1a1a;
                border: 1px solid #2e2e2e;
                height: 12px;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background-color: #3a3a3a;
                border-radius: 3px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover { background-color: #2a6aaa; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        """)
        self.waveform_scroll.sliderMoved.connect(self.on_scroll_user_change)
        self.waveform_scroll.valueChanged.connect(self.on_scroll_user_change)

        self.waveform_scrollbar_container.setVisible(False)
        scrollbar_layout.addWidget(self.waveform_scroll)
        waveform_layout.addWidget(self.waveform_scrollbar_container)

        self.waveform_info = QLabel("Status: Ready")
        self.waveform_info.setStyleSheet("color: #555555; font-size: 10px;")
        waveform_layout.addWidget(self.waveform_info)

        self.waveform_group.setLayout(waveform_layout)
        self.waveform_group.setVisible(False)
        layout.addWidget(self.waveform_group)

        self.range_group = self.create_range_section()
        self.range_group.setVisible(False)
        layout.addWidget(self.range_group)

        model_group_container = QWidget()
        model_group_container.setStyleSheet("background: transparent;")
        model_group_main_layout = QVBoxLayout(model_group_container)
        model_group_main_layout.setContentsMargins(0, 0, 0, 0)
        model_group_main_layout.setSpacing(4)

        self.model_settings_toggle = QPushButton("▶  ⚙️  Model Settings")
        self.model_settings_toggle.setCheckable(True)
        self.model_settings_toggle.setChecked(False)
        self.model_settings_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.model_settings_toggle.clicked.connect(self.toggle_model_settings)
        model_group_main_layout.addWidget(self.model_settings_toggle)

        self.model_settings_content = QGroupBox()
        self.model_settings_content.setVisible(False)
        self.model_settings_content.setMinimumWidth(850)
        self.model_settings_content.setStyleSheet("""
            QGroupBox {
                background-color: #1c1c1c;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                margin-top: 0px;
            }
        """)
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(15, 12, 15, 12)

        device_layout = QHBoxLayout()
        device_layout.setSpacing(8)
        device_layout.addWidget(make_label("Device:"))

        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setStyleSheet(RADIO_STYLE)
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.gpu_radio.setStyleSheet(RADIO_STYLE)

        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)

        if torch.cuda.is_available():
            self.gpu_radio.setChecked(True)
            try:
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_radio.setText(f"GPU (CUDA: {gpu_name})")
            except:
                self.gpu_radio.setText("GPU (CUDA)")
        else:
            self.cpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip("CUDA not available on this system")

        self.cpu_radio.toggled.connect(self.update_tf32_visibility)
        self.gpu_radio.toggled.connect(self.update_tf32_visibility)

        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        device_layout.addStretch()
        model_layout.addLayout(device_layout)

        model_size_layout = QHBoxLayout()
        model_size_layout.setSpacing(8)
        model_size_layout.addWidget(make_label("WhisperX Model:"))

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large-v3"
        ])
        self.model_combo.setCurrentText("small")
        self.model_combo.setStyleSheet(COMBO_STYLE)

        model_size_layout.addWidget(self.model_combo)
        model_size_layout.addWidget(make_label("(larger = better quality, slower)"))
        model_size_layout.addStretch()
        model_layout.addLayout(model_size_layout)

        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(8)
        lang_layout.addWidget(make_label("Language:"))
        self.language_edit = QLineEdit("en")
        self.language_edit.setMaximumWidth(80)
        self.language_edit.setPlaceholderText("auto")
        self.language_edit.setStyleSheet(INPUT_STYLE)
        lang_layout.addWidget(self.language_edit)
        lang_layout.addWidget(make_label("(ISO code: en, pl, es, fr, de, etc. or leave empty for auto-detect)"))
        lang_layout.addStretch()
        model_layout.addLayout(lang_layout)

        batch_layout = QHBoxLayout()
        batch_layout.setSpacing(8)
        batch_layout.addWidget(make_label("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        self.batch_spin.setMaximumWidth(90)
        self.batch_spin.setStyleSheet(SPINBOX_STYLE)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addWidget(make_label("(higher = faster, more VRAM)"))
        batch_layout.addStretch()
        model_layout.addLayout(batch_layout)

        self.model_settings_content.setLayout(model_layout)
        model_group_main_layout.addWidget(self.model_settings_content)
        layout.addWidget(model_group_container)

        advanced_group_container = QWidget()
        advanced_group_container.setStyleSheet("background: transparent;")
        advanced_group_main_layout = QVBoxLayout(advanced_group_container)
        advanced_group_main_layout.setContentsMargins(0, 0, 0, 0)
        advanced_group_main_layout.setSpacing(4)

        self.advanced_settings_toggle = QPushButton("▶  🔧  Advanced Settings")
        self.advanced_settings_toggle.setCheckable(True)
        self.advanced_settings_toggle.setChecked(False)
        self.advanced_settings_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.advanced_settings_toggle.clicked.connect(self.toggle_advanced_settings)
        advanced_group_main_layout.addWidget(self.advanced_settings_toggle)

        self.advanced_settings_content = QGroupBox()
        self.advanced_settings_content.setVisible(False)
        self.advanced_settings_content.setMinimumWidth(850)
        self.advanced_settings_content.setStyleSheet("""
            QGroupBox {
                background-color: #1c1c1c;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                margin-top: 0px;
            }
        """)
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(10)
        advanced_layout.setContentsMargins(15, 12, 15, 12)

        self.tf32_container = QWidget()
        self.tf32_container.setStyleSheet("background: transparent;")
        tf32_layout = QHBoxLayout(self.tf32_container)
        tf32_layout.setContentsMargins(0, 0, 0, 0)
        tf32_layout.setSpacing(8)
        self.tf32_check = QCheckBox("Enable TF32 (faster, slightly less accurate)")
        self.tf32_check.setChecked(False)
        self.tf32_check.setStyleSheet(CHECKBOX_STYLE)
        tf32_layout.addWidget(self.tf32_check)
        tf32_layout.addWidget(make_label("(only for NVIDIA RTX 30xx/40xx+ GPUs)"))
        tf32_layout.addStretch()
        advanced_layout.addWidget(self.tf32_container)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2e2e2e; border: none; max-height: 1px;")
        advanced_layout.addWidget(sep)

        voice_sep_layout = QHBoxLayout()
        voice_sep_layout.setSpacing(8)
        self.voice_separation_check = QCheckBox("🎵 Voice Separation")
        self.voice_separation_check.setChecked(False)
        self.voice_separation_check.setStyleSheet(CHECKBOX_STYLE)
        self.voice_separation_check.setToolTip(
            "Extract vocals from audio before transcription.\n"
            "Improves accuracy for files with background music or noise.\n"
            "⚠️ Significantly increases processing time."
        )
        voice_sep_layout.addWidget(self.voice_separation_check)
        voice_sep_layout.addWidget(make_label("(Separates vocal from audio — only the vocal is processed.)"))
        voice_sep_layout.addStretch()
        advanced_layout.addLayout(voice_sep_layout)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #2e2e2e; border: none; max-height: 1px;")
        advanced_layout.addWidget(sep2)

        pattern_layout = QHBoxLayout()
        pattern_layout.setSpacing(8)
        pattern_layout.addWidget(make_label("Word Pattern:"))
        self.pattern_edit = QLineEdit("3,4")
        self.pattern_edit.setMaximumWidth(150)
        self.pattern_edit.setStyleSheet(INPUT_STYLE)
        pattern_layout.addWidget(self.pattern_edit)
        pattern_layout.addWidget(make_label("(e.g., 3,4 = alternating 3 and 4 words per subtitle)"))
        pattern_layout.addStretch()
        advanced_layout.addLayout(pattern_layout)

        pause_layout = QHBoxLayout()
        pause_layout.setSpacing(8)
        pause_layout.addWidget(make_label("Min Pause for Split:"))
        self.min_pause_spin = QDoubleSpinBox()
        self.min_pause_spin.setRange(0.1, 3.0)
        self.min_pause_spin.setSingleStep(0.1)
        self.min_pause_spin.setValue(0.6)
        self.min_pause_spin.setSuffix(" s")
        self.min_pause_spin.setMaximumWidth(100)
        self.min_pause_spin.setStyleSheet(SPINBOX_STYLE)
        pause_layout.addWidget(self.min_pause_spin)
        pause_layout.addWidget(make_label("(pause between words to split subtitle)"))
        pause_layout.addStretch()
        advanced_layout.addLayout(pause_layout)

        duration_layout = QHBoxLayout()
        duration_layout.setSpacing(8)
        duration_layout.addWidget(make_label("Min Subtitle Duration:"))
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.5, 5.0)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setValue(1.0)
        self.min_duration_spin.setSuffix(" s")
        self.min_duration_spin.setMaximumWidth(100)
        self.min_duration_spin.setStyleSheet(SPINBOX_STYLE)
        duration_layout.addWidget(self.min_duration_spin)
        duration_layout.addWidget(make_label("(merge subtitles shorter than this)"))
        duration_layout.addStretch()
        advanced_layout.addLayout(duration_layout)

        line_layout = QHBoxLayout()
        line_layout.setSpacing(8)
        line_layout.addWidget(make_label("Max Line Length:"))
        self.max_line_spin = QSpinBox()
        self.max_line_spin.setRange(20, 100)
        self.max_line_spin.setValue(42)
        self.max_line_spin.setSuffix(" chars")
        self.max_line_spin.setMaximumWidth(100)
        self.max_line_spin.setStyleSheet(SPINBOX_STYLE)
        line_layout.addWidget(self.max_line_spin)
        line_layout.addWidget(make_label("(max characters per line, film standard = 42)"))
        line_layout.addStretch()
        advanced_layout.addLayout(line_layout)

        self.advanced_settings_content.setLayout(advanced_layout)
        advanced_group_main_layout.addWidget(self.advanced_settings_content)
        layout.addWidget(advanced_group_container)

        self.update_tf32_visibility()

        layout.addStretch()
        return widget

    def update_tf32_visibility(self):
        is_gpu = self.gpu_radio.isChecked()
        self.tf32_container.setVisible(is_gpu)

        if not is_gpu:
            self.tf32_check.setChecked(False)

        if self.advanced_settings_content.isVisible():
            QTimer.singleShot(50, lambda: self._adjust_window_size())

    def toggle_model_settings(self):
        is_visible = self.model_settings_toggle.isChecked()
        self.model_settings_content.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.model_settings_toggle.setText(f"{arrow} ⚙️ Model Settings")

        self.model_settings_content.updateGeometry()
        QApplication.processEvents()

        QTimer.singleShot(40, lambda: self._adjust_window_size())

    def toggle_advanced_settings(self):
        is_visible = self.advanced_settings_toggle.isChecked()
        self.advanced_settings_content.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.advanced_settings_toggle.setText(f"{arrow} 🔧 Advanced Settings")

        self.advanced_settings_content.updateGeometry()
        QApplication.processEvents()

        QTimer.singleShot(40, lambda: self._adjust_window_size())

    def _adjust_window_size(self):
        self.centralWidget().adjustSize()
        self.centralWidget().updateGeometry()
        QApplication.processEvents()

        content_size = self.centralWidget().sizeHint()

        needed_width = content_size.width() + 0
        needed_height = content_size.height() + 10

        needed_width = max(needed_width, self.minimumWidth())
        needed_height = max(needed_height, self.minimumHeight())

        self.resize(needed_width, needed_height)

    def update_tf32_visibility(self):
        is_gpu = self.gpu_radio.isChecked()
        self.tf32_container.setVisible(is_gpu)

        if not is_gpu:
            self.tf32_check.setChecked(False)

        if self.advanced_settings_content.isVisible():
            QTimer.singleShot(50, lambda: self._adjust_window_size())

    def load_waveform(self, file_path):
        if not file_path:
            print("ERROR: load_waveform called with no file_path")
            return

        file_path = Path(file_path)

        if not file_path.exists():
            print(f"ERROR: File does not exist: {file_path}")
            return

        print(f"▶ load_waveform starting for: {file_path.name}")
        self.waveform_info.setText("Status: Loading Waveform...")

        self.waveform_group.setVisible(True)
        self.range_group.setVisible(True)

        QTimer.singleShot(100, self.adjustSize)

        def worker():
            try:
                print(f" Loading audio data...")

                start_time = time.time()

                video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}
                is_video = file_path.suffix.lower() in video_extensions

                audio_file_to_load = file_path
                temp_audio_file = None

                if is_video:
                    print(f" Detected video file, extracting audio with FFmpeg...")
                    temp_audio_file = Path(tempfile.gettempdir()) / f"temp_audio_{file_path.stem}.wav"

                    try:
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", str(file_path),
                            "-vn",
                            "-acodec", "pcm_s16le",
                            "-ar", "16000",
                            "-ac", "1",
                            str(temp_audio_file)
                        ]

                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=120
                        )

                        if result.returncode != 0:
                            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

                        print(f" ✓ Audio extracted to temporary file")
                        audio_file_to_load = temp_audio_file

                    except FileNotFoundError:
                        raise RuntimeError(
                            "FFmpeg not found! Please install FFmpeg:\n"
                            "Windows: Download from https://www.gyan.dev/ffmpeg/builds/\n"
                            "Or use: winget install FFmpeg"
                        )
                    except subprocess.TimeoutExpired:
                        raise RuntimeError("FFmpeg extraction timed out (file too large?)")

                max_retries = 3
                retry_delay = 0.5
                audio_data = None
                sample_rate = None

                for attempt in range(max_retries):
                    try:
                        audio_data, sample_rate = sf.read(str(audio_file_to_load), dtype='float32')

                        if len(audio_data) > 0:
                            break
                    except Exception as load_err:
                        print(f"Retry {attempt+1}/{max_retries}: Error loading audio - {load_err}")
                        time.sleep(retry_delay)

                if temp_audio_file and temp_audio_file.exists():
                    try:
                        temp_audio_file.unlink()
                        print(f" ✓ Temporary audio file cleaned up")
                    except:
                        pass

                if audio_data is None or len(audio_data) == 0:
                    raise RuntimeError("Failed to load audio data after retries - file may be empty or corrupted.")

                if not is_video and len(audio_data.shape) > 1:
                    print(f" Converting stereo to mono (channels: {audio_data.shape[1]})")
                    audio_data = audio_data.mean(axis=1)

                if not is_video and sample_rate != 16000:
                    print(f" Resampling from {sample_rate}Hz to 16000Hz...")
                    try:
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                        print(f" ✓ Resampling completed")
                    except ImportError:
                        print(f" WARNING: librosa not installed, cannot resample from {sample_rate}Hz")
                        print(f" Install with: pip install librosa")

                load_time = time.time() - start_time

                print(f" ✓ Audio loaded in {load_time:.2f}s")
                print(f" Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                print(f" Audio range: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
                print(f" Sample rate: 16000Hz")

                if len(audio_data) == 0:
                    print(" ERROR: Audio data is empty!")
                    QTimer.singleShot(0, lambda: self.waveform_info.setText("Status: Audio file is empty"))
                    return

                self.audio_loaded_signal.emit(audio_data)

            except Exception as e:
                print(f"ERROR in waveform worker: {e}")
                traceback.print_exc()

                QTimer.singleShot(0, lambda: self.waveform_info.setText(f"Status: Error - {str(e)}"))

        print(f" Starting worker thread...")
        thread = Thread(target=worker, daemon=True)
        thread.start()

    def _update_waveform_data(self, audio_data):
        try:
            print(f"▶ _update_waveform_data called ({len(audio_data)} samples)")
            self.audio_data = audio_data

            self.waveform_display.set_audio_data(audio_data)

            self.current_position = 0
            self.playback_position_label.setText("0.00s")
            self.waveform_display.set_playback_position(0)

            self.update_waveform_scrollbar(0.0, 1.0)

        except Exception as e:
            print(f"ERROR in _update_waveform_data: {e}")
            traceback.print_exc()

    def on_waveform_selection_changed(self, selections):
        if not selections:
            self.selection_info_label.setText("No regions selected - entire file will be processed")
            self.selection_info_label.setStyleSheet(
                "color: #888; "
                "font-style: italic; "
                "padding: 8px; "
                "background-color: #f5f5f5; "
                "border: 1px solid #ddd; "
                "border-radius: 3px;"
            )
            return

        include_count = 0
        exclude_count = 0
        total_duration = 0.0

        for start, end, sel_type in selections:
            if sel_type == 'include':
                include_count += 1
            else:
                exclude_count += 1
            total_duration += (end - start)

        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        time_str = f"{minutes}:{seconds:02d}"

        parts = []
        if include_count > 0:
            parts.append(f"<span style='color: #9400D3; font-weight: bold;'>{include_count} INCLUDE</span>")
        if exclude_count > 0:
            parts.append(f"<span style='color: #DC143C; font-weight: bold;'>{exclude_count} EXCLUDE</span>")

        region_text = " + ".join(parts)
        message = f"<b>{len(selections)} regions selected</b> ({region_text}) • Total: <b>{time_str}</b>"

        self.selection_info_label.setText(message)
        self.selection_info_label.setStyleSheet(
            "color: #333; "
            "padding: 8px; "
            "background-color: #e8f5e9; "
            "border: 1px solid #4CAF50; "
            "border-radius: 3px;"
        )

    def browse_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio/Video File",
            "",
            "Media Files (*.mp3 *.wav *.mp4 *.avi *.mkv *.flac *.m4a *.ogg *.wma);;All Files (*.*)"
        )
        if file:
            self.input_file = Path(file)
            self.input_label.setText(self.input_file.name)
            self.load_waveform(self.input_file)
            self.input_label.setStyleSheet("color: black;")
            self.unload_btn.setVisible(True)
            self.button_container.setVisible(True)

            if not self.output_edit.text():
                self.output_file = self.input_file.with_suffix(".srt")
                self.output_edit.setText(str(self.output_file))

            QTimer.singleShot(100, self.adjustSize)

    def browse_output(self):
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Subtitle File",
            "",
            "SRT Files (*.srt);;All Files (*.*)"
        )
        if file:
            self.output_file = Path(file)
            self.output_edit.setText(str(self.output_file))

    def parse_time_ranges(self):
        if not hasattr(self, 'waveform_display'):
            return None, None

        selections = self.waveform_display.selections

        if not selections:
            return None, None

        include_ranges = []
        exclude_ranges = []

        for start, end, sel_type in selections:
            if sel_type == 'include':
                include_ranges.append((start, end))
            else:
                exclude_ranges.append((start, end))

        if not include_ranges and not exclude_ranges:
            return None, None

        return include_ranges, exclude_ranges

    def toggle_recording(self):
        if not hasattr(self, 'recorder'):
            self.recorder = AudioRecorder()
            self.recorder.vumeter_signal.connect(lambda v: self.rec_vol_bar.setValue(int(v * 150)))
            self.recorder.finished_signal.connect(self.on_recording_finished)
            print("✓ AudioRecorder created and signals connected")

        if not self.recorder.recording:
            print("\n" + "="*50)
            print("🎤 STARTING RECORDING")
            print("="*50)

            self.recorder.start_recording()
            self.rec_btn.setText("🛑 Stop")
            self.rec_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.pause_btn.setEnabled(True)
            self.input_label.setText("Recording in progress...")
            self.input_label.setStyleSheet("color: #f44336; font-weight: bold;")

            print("UI updated for recording state")
        else:
            print("\n" + "="*50)
            print("🛑 STOPPING RECORDING")
            print("="*50)

            self.recorder.stop_recording()
            self.rec_btn.setText("🎤 Start Record")
            self.rec_btn.setStyleSheet("")
            self.pause_btn.setEnabled(False)
            self.rec_vol_bar.setValue(0)

            print("Waiting for finished_signal...")

    def toggle_rec_pause(self):
        is_paused = self.recorder.toggle_pause()
        self.pause_btn.setText("▶ Wznów" if is_paused else "⏸ Pauza")

    def on_recording_finished(self, path):

        time.sleep(0.3)

        file_path = Path(path)

        if not file_path.exists():
            print(f"ERROR: Recording file not found: {path}")
            return

        file_size = file_path.stat().st_size
        if file_size == 0:
            print(f"ERROR: Recording file is empty: {path}")
            return

        print(f"✓ Recording file verified: {file_path.name} ({file_size / 1024:.1f} KB)")

        self.input_file = file_path
        self.input_label.setText(f"Recorded: {self.input_file.name}")
        self.input_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.unload_btn.setVisible(True)
        self.button_container.setVisible(True)

        self.output_file = self.input_file.with_suffix(".srt")
        self.output_edit.setText(str(self.output_file))

        self.waveform_group.setVisible(True)
        self.range_group.setVisible(True)

        print(f"▶ Loading waveform for: {self.input_file}")
        self.load_waveform(self.input_file)

        QTimer.singleShot(100, self.adjustSize)

    def unload_file(self):
        if self.playing:
            self.stop_playback()

        self.input_file = None
        self.audio_data = None
        self.current_position = 0

        self.input_label.setText("Drag & drop or use Record/Browse")
        self.input_label.setStyleSheet("color: gray; font-style: italic;")
        self.unload_btn.setVisible(False)
        self.button_container.setVisible(False)

        self.waveform_group.setVisible(False)
        self.range_group.setVisible(False)

        self.waveform_display.set_audio_data(None)

        self.output_edit.clear()
        self.output_file = None

        print("File unloaded successfully")

        QTimer.singleShot(100, self.adjustSize)

    def on_seek_requested(self, time_seconds):
        self.current_position = max(0.0, time_seconds)
        self.playback_position_label.setText(f"{self.current_position:.2f}s")
        self.waveform_display.set_playback_position(self.current_position)

        print(f"Seek to: {self.current_position:.2f}s")

    def update_waveform_scrollbar(self, offset, zoom):
        if self.waveform_display.audio_data is not None:
             duration = self.waveform_display.duration
             self.waveform_info.setText(
                f"Duration: {duration:.2f}s | Zoom: {zoom:.1f}x | "
                "LMB+Drag to pan, Scroll to zoom, Click to seek"
            )

        if zoom > 1.0:
            self.waveform_scrollbar_container.setVisible(True)

            self.waveform_scroll.blockSignals(True)

            max_scroll = self.waveform_scroll.maximum()

            max_offset_val = 1.0 - (1.0 / zoom)

            if max_offset_val > 0:
                scroll_val = int((offset / max_offset_val) * max_scroll)
                self.waveform_scroll.setValue(scroll_val)

                page_step = int((1.0 / zoom) / max_offset_val * max_scroll) if max_offset_val > 0 else max_scroll
                self.waveform_scroll.setPageStep(min(page_step, max_scroll))

            self.waveform_scroll.blockSignals(False)
        else:
            self.waveform_scrollbar_container.setVisible(False)

    def on_scroll_user_change(self, value):
        zoom = self.waveform_display.zoom_factor
        if zoom <= 1.0: return

        max_scroll = self.waveform_scroll.maximum()
        max_offset_val = 1.0 - (1.0 / zoom)

        if max_offset_val > 0:
            new_offset = (value / max_scroll) * max_offset_val

            self.waveform_display.set_view_offset(new_offset)

    def toggle_playback(self):
        if self.audio_data is None:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first.")
            return

        if not self.playing:
            self.start_playback()
        else:
            self.pause_playback()

    def start_playback(self):
        if self.audio_data is None:
            return

        include_selections = [
            (start, end) for start, end, sel_type in self.waveform_display.selections
            if sel_type == 'include'
        ]

        self.playing = True
        self.stop_playback_flag = False
        self.play_btn.setText("⏸ Pause")

        def playback_worker():
            samplerate = 16000

            try:
                if include_selections:
                    sorted_selections = sorted(include_selections, key=lambda x: x[0])

                    with sd.OutputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
                        chunk_size = samplerate // 20

                        for sel_start, sel_end in sorted_selections:
                            if self.stop_playback_flag:
                                break

                            start_sample = int(sel_start * samplerate)
                            end_sample = int(sel_end * samplerate)

                            audio_chunk = self.audio_data[start_sample:end_sample]

                            for i in range(0, len(audio_chunk), chunk_size):
                                if self.stop_playback_flag:
                                    break

                                chunk = audio_chunk[i:i+chunk_size]
                                if len(chunk) == 0:
                                    break

                                if len(chunk) < chunk_size:
                                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                                stream.write(chunk.reshape(-1, 1))

                                self.current_position = (start_sample + i) / samplerate
                                self.playback_position_label.setText(f"{self.current_position:.2f}s")
                                self.waveform_display.set_playback_position(self.current_position)
                else:
                    start_sample = int(self.current_position * samplerate)
                    audio_chunk = self.audio_data[start_sample:]

                    with sd.OutputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
                        chunk_size = samplerate // 20

                        for i in range(0, len(audio_chunk), chunk_size):
                            if self.stop_playback_flag:
                                break

                            chunk = audio_chunk[i:i+chunk_size]
                            if len(chunk) == 0:
                                break

                            if len(chunk) < chunk_size:
                                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                            stream.write(chunk.reshape(-1, 1))

                            self.current_position = (start_sample + i) / samplerate
                            self.playback_position_label.setText(f"{self.current_position:.2f}s")
                            self.waveform_display.set_playback_position(self.current_position)

                self.playing = False
                self.play_btn.setText("▶ Play")

            except Exception as e:
                print(f"Playback error: {e}")
                traceback.print_exc()
                self.playing = False
                self.play_btn.setText("▶ Play")

        self.playback_thread = Thread(target=playback_worker, daemon=True)
        self.playback_thread.start()

    def pause_playback(self):
        self.stop_playback_flag = True
        self.playing = False
        self.play_btn.setText("▶ Play")

    def stop_playback(self):
        self.stop_playback_flag = True
        self.playing = False
        self.play_btn.setText("▶ Play")

    def reset_playback(self):
        self.stop_playback()
        self.current_position = 0
        self.playback_position_label.setText("0.00s")
        self.waveform_display.set_playback_position(0)

    def clear_selections(self):
        if hasattr(self, 'waveform_display'):
            self.waveform_display.selections = []
            self.waveform_display.selections_changed.emit([])
            self.waveform_display.update()

        if hasattr(self, 'selection_info_label'):
            self.selection_info_label.setText("No regions selected - entire file will be processed")
            self.selection_info_label.setStyleSheet(
                "color: #888; "
                "font-style: italic; "
                "padding: 8px; "
                "background-color: #f5f5f5; "
                "border: 1px solid #ddd; "
                "border-radius: 3px;"
            )

    def create_range_section(self):
        group = QGroupBox("Optional: Select Fragments")
        layout = QVBoxLayout()

        info_label = QLabel(
            "<b>Keyboard Shortcuts:</b><br>"
            "• <b>CTRL + Left Mouse drag</b> = Select regions to <span style='color: #9400D3;'><b>INCLUDE</b></span> (purple) - will be transcribed<br>"
            "• <b>CTRL + Right Mouse drag</b> = Select regions to <span style='color: #DC143C;'><b>EXCLUDE</b></span> (red) - will be muted/removed<br>"
            "• Opposite mouse button acts as <b>eraser</b> for existing selections<br>"
        )
        info_label.setStyleSheet("color: #555; padding: 2px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.selection_info_label = QLabel("No regions selected - entire file will be processed")
        self.selection_info_label.setStyleSheet(
            "color: #888; "
            "font-style: italic; "
            "padding: 8px; "
            "background-color: #f5f5f5; "
            "border: 1px solid #ddd; "
            "border-radius: 3px;"
        )
        layout.addWidget(self.selection_info_label)

        clear_btn = QPushButton("🗑 Clear All Selections")
        clear_btn.setFixedWidth(160)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #B71C1C;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8B0000;
            }
        """)
        clear_btn.clicked.connect(self.clear_selections)
        layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

        group.setLayout(layout)
        return group

    def create_recorder_section(self):
        group = QGroupBox("Nagrywanie na żywo")
        layout = QHBoxLayout()

        self.rec_btn = QPushButton("🎤 Nagraj")
        self.rec_btn.clicked.connect(self.toggle_recording)

        self.pause_btn = QPushButton("⏸ Pauza")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_rec_pause)

        self.rec_vol_bar = QProgressBar()
        self.rec_vol_bar.setMaximum(100)
        self.rec_vol_bar.setTextVisible(False)
        self.rec_vol_bar.setFixedWidth(100)

        layout.addWidget(self.rec_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(QLabel("Vol:"))
        layout.addWidget(self.rec_vol_bar)
        group.setLayout(layout)
        return group

    def parse_pattern(self, pattern_str):
        try:
            pattern = [int(x.strip()) for x in pattern_str.split(",")]
            if not pattern or any(x <= 0 for x in pattern):
                raise ValueError
            return pattern
        except:
            return None

    def start_processing(self):
        if not self.input_file or not self.input_file.exists():
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return

        pattern = self.parse_pattern(self.pattern_edit.text())
        if not pattern:
            QMessageBox.warning(
                self,
                "Error",
                "Invalid word pattern. Use comma-separated positive integers (e.g., 3,4)"
            )
            return

        output_path = self.output_edit.text()
        if output_path:
            self.output_file = Path(output_path)
        else:
            self.output_file = self.input_file.with_suffix(".srt")

        device = "cuda" if self.gpu_radio.isChecked() else "cpu"

        config = {
            "device": device,
            "enable_tf32": self.tf32_check.isChecked(),
            "voice_separation": self.voice_separation_check.isChecked(),
            "word_pattern": pattern,
            "min_pause": self.min_pause_spin.value(),
            "min_duration": self.min_duration_spin.value(),
            "max_line_length": self.max_line_spin.value(),
            "batch_size": self.batch_spin.value(),
            "language": self.language_edit.text().strip()
        }

        model_size = self.model_combo.currentText()

        print("\n" + "="*60)
        print("  WhisperX Subtitle Generator v2.5")
        print("  Mode: SUBTITLE GENERATION (.srt)")
        print("="*60)
        print(f"Device: {device.upper()}")
        print(f"Model: {model_size}")
        print(f"Voice Separation: {'ENABLED ✓' if config['voice_separation'] else 'disabled'}")
        print(f"Input: {self.input_file}")
        print(f"Output: {self.output_file}")
        print("="*60 + "\n")

        self.processing = True
        self.process_btn.setEnabled(False)
        self.export_text_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)

        def worker():
            try:
                signals = WorkerSignals()
                signals.finished.connect(self.on_finished)
                signals.error.connect(self.on_error)

                include_ranges, exclude_ranges = self.parse_time_ranges()

                self.generator = SubtitleGenerator(config)
                result = self.generator.process(
                    self.input_file,
                    self.output_file,
                    model_size,
                    include_ranges=include_ranges,
                    exclude_ranges=exclude_ranges,
                    output_format='srt'
                )
                signals.finished.emit(result)

            except Exception as e:
                signals.error.emit(str(e) + "\n\n" + traceback.format_exc())

        thread = Thread(target=worker, daemon=True)
        thread.start()

    def start_text_export(self):
        if not self.input_file or not self.input_file.exists():
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return

        text_output = self.input_file.with_suffix(".txt")

        device = "cuda" if self.gpu_radio.isChecked() else "cpu"

        config = {
            "device": device,
            "enable_tf32": self.tf32_check.isChecked(),
            "voice_separation": self.voice_separation_check.isChecked(),
            "word_pattern": [1],
            "min_pause": 0.6,
            "min_duration": 1.0,
            "max_line_length": 42,
            "batch_size": self.batch_spin.value(),
            "language": self.language_edit.text().strip()
        }

        model_size = self.model_combo.currentText()

        print("\n" + "="*60)
        print("  WhisperX Subtitle Generator v2.5")
        print("  Mode: TEXT EXPORT (.txt)")
        print("="*60)
        print(f"Device: {device.upper()}")
        print(f"Model: {model_size}")
        print(f"Voice Separation: {'ENABLED ✓' if config['voice_separation'] else 'disabled'}")
        print(f"Input: {self.input_file}")
        print(f"Output: {text_output}")
        print("="*60 + "\n")

        self.processing = True
        self.process_btn.setEnabled(False)
        self.export_text_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)

        def worker():
            try:
                signals = WorkerSignals()
                signals.text_exported.connect(self.on_text_exported)
                signals.error.connect(self.on_error)

                include_ranges, exclude_ranges = self.parse_time_ranges()

                generator = SubtitleGenerator(config)

                generator.process(
                    self.input_file,
                    text_output,
                    model_size,
                    include_ranges=include_ranges,
                    exclude_ranges=exclude_ranges,
                    output_format='txt'
                )

                signals.text_exported.emit(str(text_output))

            except Exception as e:
                signals.error.emit(str(e) + "\n\n" + traceback.format_exc())

        thread = Thread(target=worker, daemon=True)
        thread.start()

    def on_finished(self, result):
        self.processing = False
        self.process_btn.setEnabled(True)
        self.export_text_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        print("\n" + "="*60)
        print("✓ SUCCESS - Subtitle generation completed")
        print("="*60 + "\n")

        QMessageBox.information(
            self,
            "Success",
            f"Subtitles generated successfully!\n\n"
            f"Saved {len(result)} subtitles to:\n{self.output_file}"
        )

    def on_text_exported(self, output_path):
        self.processing = False
        self.process_btn.setEnabled(True)
        self.export_text_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        print("\n" + "="*60)
        print("✓ SUCCESS - Text export completed")
        print("="*60 + "\n")

        QMessageBox.information(
            self,
            "Success",
            f"Text transcription exported successfully!\n\n"
            f"Saved to:\n{output_path}"
        )

    def on_error(self, error_msg):
        self.processing = False
        self.process_btn.setEnabled(True)
        self.export_text_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        print("\n" + "="*60)
        print("❌ ERROR - Processing failed")
        print("="*60)
        print(error_msg)
        print("="*60 + "\n")

        QMessageBox.critical(self, "Error", f"Processing failed:\n\n{error_msg}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = Path(urls[0].toLocalFile())
            self.input_file = file_path
            self.input_label.setText(file_path.name)
            self.input_label.setStyleSheet("color: black;")

            if not self.output_edit.text():
                self.output_file = file_path.with_suffix(".srt")
                self.output_edit.setText(str(self.output_file))

def main():
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "SubtitleGenerator.Mubumbutu"
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    icon_path = Path(__file__).parent / "icon.ico"
    icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()

    app.setWindowIcon(icon)

    window = SubtitleGeneratorGUI()
    window.setWindowIcon(icon)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
