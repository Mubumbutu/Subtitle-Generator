#!/usr/bin/env python3
# subtitle_generator.py
import os
import sys
os.environ["PATH"] = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages", "nvidia", "cudnn", "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["PATH"] = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages", "nvidia", "cublas", "bin") + os.pathsep + os.environ["PATH"]
import contextlib
import ctypes
import datetime
import functools
import io
import librosa
import logging
import numpy as np
import pysrt
import queue
import shutil
import sounddevice as sd
import soundfile as sf
import string
import subprocess
import tempfile
import time
import torch
import torch.serialization
import traceback
import unicodedata
import warnings
_nvidia_base = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages", "nvidia")
_dll_dirs = []
if os.path.isdir(_nvidia_base):
    for _pkg in os.listdir(_nvidia_base):
        _bin = os.path.join(_nvidia_base, _pkg, "bin")
        if os.path.isdir(_bin):
            _dll_dirs.append(os.add_dll_directory(_bin))
import whisperx
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, QObject, QPointF, Qt, QTimer, QThread
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
    QGridLayout,
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

    required_files = ["config.json", "model.bin"]
    if all((model_path / f).exists() for f in required_files):
        return True

    return False

def is_align_model_cached(language_code: str) -> bool:
    align_dir = Path("./models/whisperx/align") / language_code
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

        self.selection_enabled = True

        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setStyleSheet("background-color: #161616; border: 1px solid #333333; border-radius: 4px;")

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
        if self.audio_data is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.selection_enabled and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
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
            if self.selection_enabled and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                has_include = any(sel_type == 'include' for _, _, sel_type in self.selections)

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor("#161616"))

        f = painter.font()
        if f.pointSize() <= 0:
            f.setPointSize(10)
        painter.setFont(f)

        if self.audio_data is None:
            painter.setPen(QColor("#333333"))
            painter.drawLine(0, h // 2, w, h // 2)
            painter.setPen(QColor("#555555"))
            f2 = QFont(f)
            f2.setPointSize(9)
            painter.setFont(f2)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No audio")
            return

        ruler_h = 14 if self.duration > 0.0 else 0
        wave_h = h - ruler_h
        mid_y = wave_h / 2.0

        visible_duration = self.duration / self.zoom_factor
        view_start_time = self.offset * self.duration

        total_samples = len(self.audio_data)
        visible_samples = int(total_samples / self.zoom_factor)
        start_sample = int(self.offset * total_samples)
        if start_sample < 0:
            start_sample = 0
        end_sample = start_sample + visible_samples
        if end_sample > total_samples:
            end_sample = total_samples

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
                if bin_size < 1:
                    bin_size = 1
                limit = (len(source_data) // bin_size) * bin_size
                data_to_bin = source_data[:limit]
                binned = data_to_bin.reshape(-1, bin_size)
                plot_data = np.max(np.abs(binned), axis=1)
            else:
                plot_data = np.abs(source_data)

            count = len(plot_data)
            if count > 0:
                mx = float(np.max(plot_data)) if np.max(plot_data) > 0 else 1.0
                plot_data = plot_data / mx
                ma = mid_y - 3
                bw = w / count

                for i in range(count):
                    frac = i / count
                    time_at_bar = view_start_time + frac * visible_duration
                    bh = max(2, int(plot_data[i] * ma))
                    x = int(i * bw)
                    bwi = max(1, int(bw) - 1)

                    if time_at_bar < self.playback_position:
                        col = QColor("#2a6aaa")
                        col.setAlpha(220)
                    else:
                        col = QColor("#1a4a7a")
                        col.setAlpha(100)

                    painter.fillRect(x, int(mid_y - bh), bwi, int(bh * 2), col)

        regions_to_draw = list(self.selections)
        if self.is_selecting:
            regions_to_draw.append((
                min(self.selection_start, self.selection_end),
                max(self.selection_start, self.selection_end),
                self.selection_type
            ))

        if self.is_erasing:
            erase_start = min(self.selection_start, self.selection_end)
            erase_end = max(self.selection_start, self.selection_end)
            if erase_end > view_start_time and erase_start < (view_start_time + visible_duration):
                s_offset = erase_start - view_start_time
                e_offset = erase_end - view_start_time
                x_start = int((s_offset / visible_duration) * w)
                x_end = int((e_offset / visible_duration) * w)
                width_rect = max(1, x_end - x_start)
                painter.setBrush(QColor(128, 128, 128, 80))
                painter.setPen(QPen(QColor(200, 200, 200, 150), 2, Qt.PenStyle.DashLine))
                painter.drawRect(x_start, 0, width_rect, wave_h)
                painter.setPen(Qt.PenStyle.NoPen)

        for r_start, r_end, r_type in regions_to_draw:
            if r_end < view_start_time or r_start > (view_start_time + visible_duration):
                continue
            s_offset = r_start - view_start_time
            e_offset = r_end - view_start_time
            x_start = int((s_offset / visible_duration) * w)
            x_end = int((e_offset / visible_duration) * w)
            width_rect = max(1, x_end - x_start)

            if r_type == 'include':
                painter.setBrush(QColor(75, 0, 130, 120))
                edge_color = QColor(148, 0, 211, 200)
            else:
                painter.setBrush(QColor(139, 0, 0, 120))
                edge_color = QColor(220, 20, 60, 200)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x_start, 0, width_rect, wave_h)
            painter.setPen(edge_color)
            painter.drawLine(x_start, 0, x_start, wave_h)
            painter.drawLine(x_end, 0, x_end, wave_h)
            painter.setPen(Qt.PenStyle.NoPen)

        if view_start_time <= self.playback_position <= view_start_time + visible_duration:
            cursor_rel = (self.playback_position - view_start_time) / visible_duration
            cursor_x = int(cursor_rel * w)
            painter.setPen(QPen(QColor("#2a6aaa"), 2))
            painter.drawLine(cursor_x, 0, cursor_x, wave_h)

        if self.duration > 0.0:
            ruler_y = wave_h
            painter.fillRect(0, ruler_y, w, ruler_h, QColor(14, 14, 14, 220))
            painter.setPen(QColor(50, 50, 50))
            painter.drawLine(0, ruler_y, w, ruler_y)

            if visible_duration < 5:
                time_step = 0.5
            elif visible_duration < 10:
                time_step = 1.0
            elif visible_duration < 30:
                time_step = 5.0
            elif visible_duration < 60:
                time_step = 10.0
            elif visible_duration < 300:
                time_step = 30.0
            elif visible_duration <= 1800:
                time_step = 60.0
            elif visible_duration <= 5400:
                time_step = 300.0
            else:
                time_step = 600.0

            f_ruler = QFont("Consolas", 7)
            painter.setFont(f_ruler)
            fm = painter.fontMetrics()
            last_lx = -999

            first_tick = int(view_start_time / time_step) * time_step
            t = first_tick
            while t <= view_start_time + visible_duration + time_step:
                if t < 0:
                    t += time_step
                    continue
                time_offset = t - view_start_time
                tick_x = int((time_offset / visible_duration) * w)
                if 0 <= tick_x <= w:
                    painter.setPen(QColor(80, 80, 80))
                    painter.drawLine(tick_x, ruler_y, tick_x, ruler_y + 4)
                    if time_step < 1.0:
                        label = f"{t:.1f}s"
                    else:
                        mins = int(t) // 60
                        secs = int(t) % 60
                        label = f"{mins}:{secs:02d}"
                    lw = fm.horizontalAdvance(label)
                    lx = max(1, min(w - lw - 1, tick_x - lw // 2))
                    if lx > last_lx + lw + 4:
                        painter.setPen(QColor("#888888"))
                        painter.drawText(lx, h - 2, label)
                        last_lx = lx
                t += time_step

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
        self.reuse_enabled = config.get("reuse_enabled", False)
        self.pedalboard_config = config.get("pedalboard", {})
 
        self.temp_vocals_file = None
        self.temp_video_audio = None
        self.temp_demucs_dir = None
        self.temp_pedalboard_file = None
        self.final_processed_audio = None
 
        self.last_segments = None

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

            vocals_audio, vocals_sr = librosa.load(str(vocals_file), sr=None, mono=True)
            vocals_audio = vocals_audio.astype(np.float32)

            demucs_output_dir = Path("./demucs_outputs")
            demucs_output_dir.mkdir(parents=True, exist_ok=True)
            idx = 1
            while True:
                dest_name = f"{safe_name}_{idx:02d}.wav"
                dest_path = demucs_output_dir / dest_name
                if not dest_path.exists():
                    break
                idx += 1
            sf.write(str(dest_path), vocals_audio, vocals_sr)
            self.log(f"  ✓ Vocals saved to: {dest_path}")

            self.temp_vocals_file = vocals_file
            self.temp_demucs_dir = output_dir

            self.log("="*50)
            self.log("✓ VOICE SEPARATION COMPLETE")
            self.log("="*50 + "\n")

            return dest_path

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

    def apply_pedalboard(self, audio_path: Path) -> Path:
        try:
            from pedalboard import Pedalboard, NoiseGate, HighpassFilter, Compressor, Gain
 
            self.log("\n" + "="*50)
            self.log("🎛️ AUDIO PROCESSING")
            self.log("="*50)
 
            pb_cfg = self.pedalboard_config
            effects = []
 
            if pb_cfg.get("noise_gate_enabled", False):
                effects.append(NoiseGate(
                    threshold_db=pb_cfg.get("noise_gate_threshold", -40.0),
                    ratio=10.0,
                    attack_ms=2.0,
                    release_ms=pb_cfg.get("noise_gate_release", 200.0),
                ))
                self.log(
                    f"  + NoiseGate  threshold={pb_cfg.get('noise_gate_threshold', -40.0)} dB"
                    f"  release={pb_cfg.get('noise_gate_release', 200.0)} ms"
                )
 
            if pb_cfg.get("highpass_enabled", False):
                effects.append(HighpassFilter(
                    cutoff_frequency_hz=pb_cfg.get("highpass_cutoff", 80.0),
                ))
                self.log(f"  + HighpassFilter  cutoff={pb_cfg.get('highpass_cutoff', 80.0)} Hz")
 
            if pb_cfg.get("compressor_enabled", False):
                effects.append(Compressor(
                    threshold_db=pb_cfg.get("compressor_threshold", -20.0),
                    ratio=pb_cfg.get("compressor_ratio", 4.0),
                ))
                self.log(
                    f"  + Compressor  threshold={pb_cfg.get('compressor_threshold', -20.0)} dB"
                    f"  ratio={pb_cfg.get('compressor_ratio', 4.0)}:1"
                )
 
            if pb_cfg.get("gain_enabled", False):
                gain_val = pb_cfg.get("gain_db", 0.0)
                if gain_val != 0.0:
                    effects.append(Gain(gain_db=gain_val))
                    self.log(f"  + Gain  {gain_val:+.1f} dB")
 
            if not effects:
                self.log("  No effects enabled — skipping.")
                self.log("="*50 + "\n")
                return audio_path
 
            audio_path = Path(audio_path)
            self.log(f"▶ Input file: {audio_path.name}")
            self.log(f"▶ File size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
 
            audio_data, sr = sf.read(str(audio_path), dtype='float32')
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
 
            board = Pedalboard(effects)
 
            duration = len(audio_data) / sr
            self.log(f"▶ Processing {duration:.1f}s of audio at {sr} Hz...")
            self.log("  This may take a moment depending on file length...")
 
            start_time = datetime.datetime.now()
            processed = board(audio_data, sr)
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            self.log(f"  ✓ Processing completed in {elapsed:.1f}s")
 
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_output = Path(tempfile.gettempdir()) / f"pedalboard_{timestamp}.wav"
            sf.write(str(temp_output), processed, sr)
            self.temp_pedalboard_file = temp_output
 
            out_size = temp_output.stat().st_size / (1024*1024)
            self.log(f"  ✓ Output saved: {temp_output.name} ({out_size:.2f} MB)")
            self.log("="*50)
            self.log("✓ AUDIO PROCESSING COMPLETE")
            self.log("="*50 + "\n")
 
            return temp_output
 
        except ImportError:
            self.log("  ⚠️ ERROR: pedalboard not installed!")
            self.log("  Install with: pip install pedalboard")
            self.log("  Continuing with original audio...")
            return audio_path
        except Exception as e:
            self.log("\n" + "="*50)
            self.log("⚠️ AUDIO PROCESSING FAILED")
            self.log("="*50)
            self.log(f"Error: {str(e)}")
            self.log("\nFull traceback:")
            self.log(traceback.format_exc())
            self.log("="*50)
            self.log("Continuing with original audio file...")
            self.log("="*50 + "\n")
            return audio_path

    def _chunk_fixed(self, audio_data: np.ndarray, sr: int, chunk_sec: float) -> list:
        chunk_samples = max(sr, int(chunk_sec * sr))
        total = len(audio_data)
        chunks = []
        pos = 0
        while pos < total:
            end = min(pos + chunk_samples, total)
            chunks.append(audio_data[pos:end].copy())
            pos = end
        return chunks

    def _find_cut_point(self, audio_data: np.ndarray, sr: int, search_start: int, search_end: int) -> int:
        segment = audio_data[search_start:search_end]
        n = len(segment)
        if n == 0:
            return search_start
        window_samples = max(1, int(0.05 * sr))
        step = max(1, window_samples // 2)
        min_rms = float('inf')
        best_mid = n // 2
        i = 0
        while i + window_samples <= n:
            w = segment[i:i + window_samples]
            rms = float(np.dot(w, w) / window_samples) ** 0.5
            if rms < min_rms:
                min_rms = rms
                best_mid = i + window_samples // 2
            i += step
        return search_start + best_mid

    def _chunk_smart(self, audio_data: np.ndarray, sr: int, chunk_sec: float) -> list:
        chunk_samples = max(sr, int(chunk_sec * sr))
        half_chunk = max(1, chunk_samples // 2)
        total = len(audio_data)
        chunks = []
        pos = 0
        while pos < total:
            ideal_end = pos + chunk_samples
            if ideal_end >= total:
                chunks.append(audio_data[pos:total].copy())
                break
            search_start = pos + half_chunk
            search_end = ideal_end
            if search_start >= search_end:
                cut = ideal_end
            else:
                cut = self._find_cut_point(audio_data, sr, search_start, search_end)
            cut = max(pos + 1, min(cut, total))
            chunks.append(audio_data[pos:cut].copy())
            pos = cut
        return chunks

    def _crossfade_join(self, a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
        fs = min(fade_samples, len(a) // 2, len(b) // 2)
        if fs <= 0:
            return np.concatenate([a, b])
        fade_out = np.linspace(1.0, 0.0, fs, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fs, dtype=np.float32)
        overlap = a[-fs:] * fade_out + b[:fs] * fade_in
        return np.concatenate([a[:-fs], overlap, b[fs:]])

    def _crossfade_concat(self, chunks: list, sr: int, fade_ms: int = 30) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0].astype(np.float32)
        fade_samples = int(fade_ms * sr / 1000)
        result = chunks[0].astype(np.float32)
        for chunk in chunks[1:]:
            result = self._crossfade_join(result, chunk.astype(np.float32), fade_samples)
        return result

    def _is_reuse_model_ready(self, model_dir: Path) -> bool:
        if not model_dir.exists() or not model_dir.is_dir():
            return False
        for f in model_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".ckpt", ".pt", ".pth", ".bin"):
                return True
            if f.name.lower() in ("inference.py", "enhance.py", "config.yaml", "config.yml"):
                return True
        return False

    def apply_reuse(self, audio_path: Path) -> Path:
        import re as _re
        try:
            self.log("\n" + "="*50)
            self.log("🔊 RE-USE AUDIO ENHANCEMENT")
            self.log("="*50)
            model_dir = Path(self.config.get("reuse_model_dir", "./models/reuse")).resolve()
            bwe = self.config.get("bwe", 0)
            chunking_enabled = self.config.get("reuse_chunking_enabled", False)
            chunk_mode = self.config.get("reuse_chunk_mode", "fixed")
            chunk_seconds = float(self.config.get("reuse_chunk_seconds", 30.0))
            if not self._is_reuse_model_ready(model_dir):
                self.log(f"  RE-USE model not found at: {model_dir}")
                self.log("  Downloading nvidia/RE-USE from Hugging Face...")
                self.log("  This may take several minutes depending on your connection...")
                try:
                    from huggingface_hub import snapshot_download
                    model_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        repo_id="nvidia/RE-USE",
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False,
                    )
                    self.log("  ✓ RE-USE model downloaded successfully")
                except ImportError:
                    self.log("  ⚠️ huggingface_hub is not installed.")
                    self.log("  Install with: pip install huggingface_hub")
                    self.log("  Continuing without RE-USE enhancement...")
                    return audio_path
                except Exception as dl_err:
                    self.log(f"  ⚠️ Download failed: {dl_err}")
                    self.log("  Continuing without RE-USE enhancement...")
                    return audio_path
            audio_path = Path(audio_path)
            self.log(f"▶ Input file: {audio_path.name}")
            self.log(f"▶ File size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
            if chunking_enabled:
                self.log(f"▶ Chunking: {chunk_mode} ({chunk_seconds:.1f}s target per chunk)")
            noisy_dir = model_dir / "noisy_audio"
            enhanced_dir = model_dir / "enhanced_audio"
            noisy_dir.mkdir(exist_ok=True)
            enhanced_dir.mkdir(exist_ok=True)
            for f in noisy_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            for f in list(enhanced_dir.glob("*.wav")) + list(enhanced_dir.glob("*.flac")):
                try:
                    f.unlink()
                except Exception:
                    pass
            chunk_stems = None
            sr_source = None
            if chunking_enabled:
                audio_data_full, sr_source = sf.read(str(audio_path), dtype='float32')
                if len(audio_data_full.shape) > 1:
                    audio_data_full = audio_data_full.mean(axis=1)
                if chunk_mode == 'smart':
                    chunks_data = self._chunk_smart(audio_data_full, sr_source, chunk_seconds)
                else:
                    chunks_data = self._chunk_fixed(audio_data_full, sr_source, chunk_seconds)
                self.log(f"  ✓ Split into {len(chunks_data)} chunks")
                chunk_stems = []
                for i, chunk in enumerate(chunks_data):
                    stem_i = f"{audio_path.stem}_chunk_{i:04d}"
                    sf.write(str(noisy_dir / f"{stem_i}.wav"), chunk, sr_source)
                    chunk_stems.append(stem_i)
                    self.log(f"    Chunk {i + 1}/{len(chunks_data)}: {len(chunk) / sr_source:.2f}s")
            else:
                out_wav = noisy_dir / (audio_path.stem + ".wav")
                if out_wav.resolve() != audio_path.resolve():
                    shutil.copy2(str(audio_path), str(out_wav))
            in_stem = audio_path.stem
            parsed_env = {}

            def find_script(names):
                for name in names:
                    p = model_dir / name
                    if p.exists():
                        return p
                for name in names:
                    found = list(model_dir.rglob(name))
                    if found:
                        return found[0]
                return None

            def parse_sh(sh_path):
                try:
                    text = sh_path.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if _re.search(r"\bpython\b", line, _re.I):
                            line = _re.sub(r"\s*\\$", "", line).split("#")[0].strip()
                            parts = line.split()
                            cmd_parts = []
                            for tok in parts:
                                env_m = _re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)', tok)
                                if env_m and not cmd_parts:
                                    parsed_env[env_m.group(1)] = env_m.group(2).strip("'\"")
                                else:
                                    if _re.fullmatch(r"python[\d.]*", tok, _re.I):
                                        tok = sys.executable
                                    cmd_parts.append(tok)
                            return cmd_parts if cmd_parts else None
                except Exception as ex:
                    self.log(f"  parse_sh error: {ex}")
                return None

            def find_model_config():
                priority = [
                    "config.yaml", "config.yml", "model_config.yaml",
                    "model_config.yml", "config.json"
                ]
                for name in priority:
                    p = model_dir / name
                    if p.exists():
                        return p
                for pat in ("*.yaml", "*.yml", "*.json"):
                    for p in sorted(model_dir.rglob(pat)):
                        if p.name.lower() in ("hparams.yaml", "hparams.yml"):
                            continue
                        try:
                            with open(p, encoding="utf-8") as f:
                                if p.suffix in (".yaml", ".yml"):
                                    import yaml
                                    data = yaml.safe_load(f)
                                else:
                                    import json
                                    data = json.load(f)
                            if isinstance(data, dict) and "stft_cfg" in data:
                                return p
                        except Exception:
                            continue
                return None

            sh_path = find_script(["inference.sh"])
            cmd = None
            if sh_path:
                cmd = parse_sh(sh_path)
                self.log(f"  Parsed from {sh_path.name}: {cmd}")
            if not cmd:
                script = find_script([
                    "inference.py", "enhance.py", "run_enhancement.py",
                    "run_enhance.py", "infer.py", "main.py"
                ])
                if not script:
                    raise RuntimeError(
                        "Could not locate an inference script in the RE-USE model directory."
                    )
                cmd = [sys.executable, str(script)]
                self.log(f"  Using script: {script.name}")
            cmd_str = " ".join(str(c) for c in cmd)
            if not any(x in cmd_str for x in [
                "--input_folder", "--input_dir", "--noisy_dir", "--input_path"
            ]):
                cmd.extend(["--input_folder", str(noisy_dir.relative_to(model_dir))])
            if not any(x in cmd_str for x in [
                "--output_folder", "--output_dir", "--output_path", "--out_dir"
            ]):
                cmd.extend(["--output_folder", str(enhanced_dir.relative_to(model_dir))])
            if "--checkpoint_file" not in cmd_str:
                ckpt_exts = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")
                ckpt_candidates = []
                for ext in ckpt_exts:
                    ckpt_candidates.extend(list(model_dir.rglob(f"*{ext}")))
                if ckpt_candidates:
                    ckpt = max(ckpt_candidates, key=lambda f: f.stat().st_size)
                    cmd.extend(["--checkpoint_file", str(ckpt.relative_to(model_dir))])
                    self.log(f"  ✓ Found checkpoint: {ckpt.name} ({ckpt.stat().st_size / (1024*1024):.1f} MB)")
                else:
                    self.log("  ⚠️ No checkpoint file found in RE-USE model directory!")
                    self.log("  Continuing without RE-USE enhancement...")
                    return audio_path
            if "--config" not in cmd_str:
                cfg = find_model_config()
                if cfg:
                    cmd.extend(["--config", str(cfg.relative_to(model_dir))])
            if bwe > 0 and "--BWE" not in cmd_str:
                cmd.extend(["--BWE", str(bwe)])
            self.log("▶ Running RE-USE inference...")
            self.log(f"  Command: {' '.join(str(c) for c in cmd)}")
            env = {**os.environ, "PYTHONPATH": str(model_dir)}
            if parsed_env:
                env.update(parsed_env)
            start_time = datetime.datetime.now()
            process = subprocess.Popen(
                cmd,
                cwd=str(model_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                errors="replace"
            )
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    self.log(f"  {stripped}")
            process.wait()
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            if process.returncode != 0:
                raise RuntimeError(
                    f"RE-USE inference failed with exit code {process.returncode}"
                )
            self.log(f"  ✓ RE-USE completed in {elapsed:.1f}s")
            if chunking_enabled and chunk_stems:
                chunk_results = []
                for i, stem_i in enumerate(chunk_stems):
                    out_files = [
                        p for p in enhanced_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in (".wav", ".flac") and p.stem == stem_i
                    ]
                    if not out_files:
                        out_files = [
                            p for p in enhanced_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in (".wav", ".flac") and stem_i in p.stem
                        ]
                    if not out_files:
                        raise FileNotFoundError(
                            f"No enhanced output found for chunk: {stem_i}"
                        )
                    chunk_audio, chunk_sr = sf.read(str(out_files[0]), dtype='float32')
                    if len(chunk_audio.shape) > 1:
                        chunk_audio = chunk_audio.mean(axis=1)
                    chunk_results.append((chunk_audio.astype(np.float32), chunk_sr))
                    self.log(f"    Loaded chunk {i + 1}/{len(chunk_stems)}: {len(chunk_audio) / chunk_sr:.2f}s")
                target_sr = chunk_results[0][1]
                chunks_audio = []
                for chunk_audio, chunk_sr in chunk_results:
                    if chunk_sr != target_sr:
                        try:
                            chunk_audio = librosa.resample(chunk_audio, orig_sr=chunk_sr, target_sr=target_sr)
                        except Exception:
                            pass
                    chunks_audio.append(chunk_audio.astype(np.float32))
                final_audio = self._crossfade_concat(chunks_audio, target_sr, fade_ms=30)
                final_sr = target_sr
                self.log(
                    f"  ✓ Crossfade-concatenated {len(chunks_audio)} chunks"
                    f" → {len(final_audio) / final_sr:.2f}s total"
                )
            else:
                out_files = [
                    p for p in enhanced_dir.rglob("*")
                    if p.suffix.lower() in (".wav", ".flac") and p.is_file()
                ]
                if not out_files:
                    raise RuntimeError(f"No output file produced by RE-USE in {enhanced_dir}")
                matched = [f for f in out_files if f.stem == in_stem]
                src_out = matched[0] if matched else out_files[0]
                final_audio, final_sr = sf.read(str(src_out), dtype='float32')
                if len(final_audio.shape) > 1:
                    final_audio = final_audio.mean(axis=1)
                final_audio = final_audio.astype(np.float32)
            reuse_output_dir = Path("./RE-USE_outputs")
            reuse_output_dir.mkdir(parents=True, exist_ok=True)
            idx = 1
            while True:
                dest_name = f"{audio_path.stem}_{idx:02d}.wav"
                dest_path = reuse_output_dir / dest_name
                if not dest_path.exists():
                    break
                idx += 1
            sf.write(str(dest_path), final_audio, final_sr)
            for f in list(enhanced_dir.glob("*.wav")) + list(enhanced_dir.glob("*.flac")):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                for f in noisy_dir.iterdir():
                    if f.is_file():
                        f.unlink()
            except Exception:
                pass
            self.log(f"  ✓ Enhanced audio saved to: {dest_path}")
            self.log("="*50)
            self.log("✓ RE-USE ENHANCEMENT COMPLETE")
            self.log("="*50 + "\n")
            return dest_path
        except Exception as e:
            self.log("\n" + "="*50)
            self.log("⚠️ RE-USE ENHANCEMENT FAILED")
            self.log("="*50)
            self.log(f"Error: {str(e)}")
            self.log("\nFull traceback:")
            self.log(traceback.format_exc())
            self.log("="*50)
            self.log("Continuing with original audio file...")
            self.log("="*50 + "\n")
            return audio_path

    def prepare_audio(self, audio_path, include_ranges=None, exclude_ranges=None, audio_ready_callback=None):
        if self.final_processed_audio and self.final_processed_audio.exists():
            try:
                self.final_processed_audio.unlink()
            except Exception:
                pass
            self.final_processed_audio = None
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}
        if audio_path.suffix.lower() in video_extensions:
            self.log("▶ Detected video file, extracting audio with FFmpeg...")
            temp_video = Path(tempfile.gettempdir()) / f"temp_audio_{audio_path.stem}.wav"
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(audio_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    str(temp_video)
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=120
                )
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                self.temp_video_audio = temp_video
                self.log(f" ✓ Audio extracted to temporary file: {temp_video.name}")
                audio_path = temp_video
            except Exception as e:
                self.log(f" ⚠️ FFmpeg extraction failed: {e}")
                raise
        audio_data, sr = sf.read(str(audio_path), dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != 16000:
            self.log(f"▶ Resampling from {sr}Hz to 16000Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            self.log(" ✓ Resampling completed")
        temp_normalized = Path(tempfile.gettempdir()) / f"normalized_{audio_path.stem}.wav"
        sf.write(str(temp_normalized), audio_data, 16000)
        audio_path = temp_normalized
        has_include = include_ranges and len(include_ranges) > 0
        has_exclude = exclude_ranges and len(exclude_ranges) > 0
        if has_include and has_exclude:
            self.log(" ⚠️ WARNING: Both include and exclude ranges specified!")
            self.log(" ⚠️ Include ranges take priority - exclude ranges will be ignored.")
            has_exclude = False
            exclude_ranges = None
        processed_audio_path = audio_path
        temp_processed_file = None
        needs_processing = has_include or has_exclude
        if needs_processing:
            self.log("▶ Processing audio regions...")
            temp_processed_file = Path(tempfile.gettempdir()) / f"processed_{audio_path.stem}.wav"
            try:
                audio_data, sr = sf.read(str(audio_path), dtype='float32')
                if has_include:
                    segments = []
                    for start, end in sorted(include_ranges, key=lambda x: x[0]):
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        segments.append(audio_data[start_sample:end_sample])
                    processed_audio = np.concatenate(segments)
                    self.log(f" ✓ Extracted {len(include_ranges)} regions")
                elif has_exclude:
                    mask = np.ones(len(audio_data), dtype=bool)
                    for start, end in exclude_ranges:
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        mask[start_sample:end_sample] = False
                    processed_audio = audio_data[mask]
                    self.log(f" ✓ Excluded {len(exclude_ranges)} regions")
                sf.write(str(temp_processed_file), processed_audio, sr)
                processed_audio_path = temp_processed_file
                self.log(f" ✓ Saved processed audio: {temp_processed_file.name}")
            except Exception as e:
                self.log(f" ⚠️ Error processing audio: {e}")
                raise
        if self.voice_separation_enabled:
            processed_audio_path = self.separate_vocals(processed_audio_path)
        if self.reuse_enabled:
            processed_audio_path = self.apply_reuse(processed_audio_path)
        pb_cfg = self.pedalboard_config
        if any(pb_cfg.get(k, False) for k in [
            "noise_gate_enabled", "highpass_enabled", "compressor_enabled", "gain_enabled"
        ]):
            processed_audio_path = self.apply_pedalboard(processed_audio_path)
        if audio_ready_callback is not None:
            try:
                audio_ready_callback(str(processed_audio_path))
            except Exception:
                pass
        if temp_processed_file and temp_processed_file.exists() and temp_processed_file != processed_audio_path:
            try:
                temp_processed_file.unlink()
            except Exception:
                pass
        if temp_normalized.exists() and temp_normalized != processed_audio_path:
            try:
                temp_normalized.unlink()
            except Exception:
                pass
        self.final_processed_audio = processed_audio_path
        self.cleanup_temp_files(keep=self.final_processed_audio)
        return processed_audio_path

    def cleanup_temp_files(self, keep=None):
        self.log("\n▶ Cleaning up temporary files...")

        cleaned = False

        def remove_file_with_retry(file_path, max_retries=3, delay=0.5):
            for attempt in range(max_retries):
                try:
                    file_path.unlink()
                    return True
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
            return False

        if self.temp_video_audio and self.temp_video_audio.exists() and self.temp_video_audio != keep:
            try:
                if remove_file_with_retry(self.temp_video_audio):
                    self.log(f"  ✓ Removed: {self.temp_video_audio.name}")
                    cleaned = True
            except Exception as e:
                self.log(f"  ⚠️ Could not remove {self.temp_video_audio.name}: {e}")

        if self.temp_vocals_file and self.temp_vocals_file.exists() and self.temp_vocals_file != keep:
            try:
                if not self.temp_demucs_dir or self.temp_demucs_dir not in self.temp_vocals_file.parents:
                    if remove_file_with_retry(self.temp_vocals_file):
                        self.log(f"  ✓ Removed: {self.temp_vocals_file.name}")
                        cleaned = True
            except Exception as e:
                self.log(f"  ⚠️ Could not remove {self.temp_vocals_file.name}: {e}")

        skip_demucs = keep and self.temp_demucs_dir and self.temp_demucs_dir in keep.parents
        if not skip_demucs and self.temp_demucs_dir and self.temp_demucs_dir.exists():
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

        if self.temp_pedalboard_file and self.temp_pedalboard_file.exists() and self.temp_pedalboard_file != keep:
            try:
                if remove_file_with_retry(self.temp_pedalboard_file):
                    self.log(f"  ✓ Removed: {self.temp_pedalboard_file.name}")
                    cleaned = True
            except Exception as e:
                self.log(f"  ⚠️ Could not remove {self.temp_pedalboard_file.name}: {e}")

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

            align_cached = is_align_model_cached(detected_language)
            if align_cached:
                self.log(f" ✓ Using locally cached alignment model for: {detected_language}")
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            else:
                self.log(f" Downloading alignment model for: {detected_language}...")
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE", None)

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
        finally:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)

        return result

    def create_subtitles_from_segments(self, segments):
        import re
        subs = pysrt.SubRipFile()
        sub_idx = 1
        word_pattern = self.config["word_pattern"]
        max_words_cap = max(word_pattern) * 4
        max_line_length = self.config["max_line_length"]
        min_pause = self.config["min_pause"]
        SENTENCE_END = re.compile(r'[.!?…]+$')
        SOFT_BREAK = re.compile(r'[,;:]+$')

        segments = self.merge_incomplete_segments(segments)

        def words_from_segment(seg):
            if "words" in seg:
                return [w for w in seg["words"] if w.get("start") is not None and w.get("end") is not None]
            text = seg.get("text", "").strip()
            if not text:
                return []
            word_list = text.split()
            if not word_list:
                return []
            duration = seg["end"] - seg["start"]
            tpw = duration / len(word_list)
            return [{"word": w, "start": seg["start"] + i * tpw, "end": seg["start"] + (i + 1) * tpw}
                    for i, w in enumerate(word_list)]

        def flush_chunk(chunk):
            nonlocal sub_idx
            if not chunk:
                return
            text = " ".join(w["word"] for w in chunk)
            text = self.split_lines(text, max_line_length)
            subs.append(pysrt.SubRipItem(
                index=sub_idx,
                start=seconds_to_srt_time(chunk[0]["start"]),
                end=seconds_to_srt_time(chunk[-1]["end"]),
                text=text
            ))
            sub_idx += 1

        def flush_at_best_boundary(chunk):
            for i in range(len(chunk) - 1, 0, -1):
                if SOFT_BREAK.search(chunk[i]["word"].strip()):
                    flush_chunk(chunk[:i + 1])
                    return chunk[i + 1:]
            flush_chunk(chunk)
            return []

        for seg in segments:
            words = words_from_segment(seg)
            if not words:
                continue
            chunk = []
            for w in words:
                if chunk:
                    pause = w["start"] - chunk[-1]["end"]
                    if pause > min_pause:
                        flush_chunk(chunk)
                        chunk = []
                chunk.append(w)
                raw = w["word"].strip()
                if SENTENCE_END.search(raw):
                    flush_chunk(chunk)
                    chunk = []
                    continue
                if SOFT_BREAK.search(raw) and len(chunk) >= 3:
                    flush_chunk(chunk)
                    chunk = []
                    continue
                if len(chunk) >= max_words_cap:
                    remainder = flush_at_best_boundary(chunk)
                    chunk = []
                    for rw in remainder:
                        if chunk:
                            pause = rw["start"] - chunk[-1]["end"]
                            if pause > min_pause:
                                flush_chunk(chunk)
                                chunk = []
                        chunk.append(rw)
            flush_chunk(chunk)

        return self.merge_short_subs(subs)

    def merge_short_subs(self, subs):
        import re
        SENTENCE_END = re.compile(r'[.!?…]+$')
        min_duration = self.config["min_duration"]

        items = list(subs)
        changed = True

        while changed:
            changed = False
            new_items = []
            i = 0
            while i < len(items):
                cur = items[i]
                duration_ms = cur.end.ordinal - cur.start.ordinal
                last_word = cur.text.rstrip().split()[-1] if cur.text.strip() else ""
                ends_sentence = bool(SENTENCE_END.search(last_word))

                if duration_ms < min_duration * 1000:
                    if new_items:
                        prev = new_items[-1]
                        prev_last = prev.text.rstrip().split()[-1] if prev.text.strip() else ""
                        prev_ends = bool(SENTENCE_END.search(prev_last))
                        if not prev_ends:
                            prev.text += " " + cur.text
                            prev.end = cur.end
                            i += 1
                            changed = True
                            continue
                    if i + 1 < len(items) and not ends_sentence:
                        nxt = items[i + 1]
                        cur.text += " " + nxt.text
                        cur.end = nxt.end
                        new_items.append(cur)
                        i += 2
                        changed = True
                        continue

                new_items.append(cur)
                i += 1
            items = new_items

        result = pysrt.SubRipFile()
        for idx, s in enumerate(items, 1):
            s.index = idx
            result.append(s)
        return result

    def merge_incomplete_segments(self, segments):
        import re
        SENTENCE_END = re.compile(r'[.!?…]+$')
        FILLER = re.compile(
            r'^(uh+|um+|hmm+|hm+|uh-huh|ah+|eh+|ohh+|err+|uhm+|mhm+|huh|yy+|ee+|eee+|mm+|öö+|äh+|öh+)$',
            re.IGNORECASE
        )

        cleaned = []
        for seg in segments:
            if "words" not in seg:
                cleaned.append(seg)
                continue
            filtered_words = [w for w in seg["words"] if not FILLER.match(w.get("word", "").strip())]
            if not filtered_words:
                continue
            seg = dict(seg)
            seg["words"] = filtered_words
            seg["text"] = " ".join(w["word"] for w in filtered_words)
            cleaned.append(seg)

        merged = []
        i = 0
        while i < len(cleaned):
            cur = dict(cleaned[i])
            while i + 1 < len(cleaned):
                cur_text = cur.get("text", "").strip()
                last_word = cur_text.split()[-1] if cur_text else ""
                if SENTENCE_END.search(last_word):
                    break
                nxt = cleaned[i + 1]
                nxt_text = nxt.get("text", "").strip()
                if not nxt_text:
                    break
                first_word = nxt_text.split()[0]
                if first_word[0].isupper():
                    break
                cur["text"] = cur_text + " " + nxt_text
                cur["end"] = nxt["end"]
                if "words" in cur and "words" in nxt:
                    cur["words"] = cur["words"] + nxt["words"]
                i += 1
            merged.append(cur)
            i += 1

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

    def process(self, audio_path, output_path, model_size, include_ranges=None, exclude_ranges=None, output_format='srt', audio_ready_callback=None, prebuilt_audio_path=None):
        global processing_start_time
        processing_start_time = datetime.datetime.now()

        self.log(f"\n▶ Starting processing: {audio_path.name}")
        self.log(f"   Output format: {output_format.upper()}")

        temp_processed_file = None
        using_prebuilt = prebuilt_audio_path is not None and Path(prebuilt_audio_path).exists()

        if using_prebuilt:
            self.log("   ✓ Reusing pre-built audio from Check Audio — skipping audio preparation")
            processed_audio_path = Path(prebuilt_audio_path)
            if audio_ready_callback is not None:
                try:
                    audio_ready_callback(str(processed_audio_path))
                except Exception:
                    pass
        else:
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

            video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}
            if audio_path.suffix.lower() in video_extensions:
                self.log("▶ Detected video file, extracting audio with FFmpeg...")
                temp_video = Path(tempfile.gettempdir()) / f"temp_audio_{audio_path.stem}.wav"
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(audio_path),
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        str(temp_video)
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=120
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                    self.temp_video_audio = temp_video
                    self.log(f"  ✓ Audio extracted to temporary file: {temp_video.name}")
                    audio_path = temp_video
                except Exception as e:
                    self.log(f"  ⚠️ FFmpeg extraction failed: {e}")
                    raise

            processed_audio_path = audio_path

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

            if self.reuse_enabled:
                processed_audio_path = self.apply_reuse(processed_audio_path)

            pb_cfg = self.pedalboard_config
            if any(pb_cfg.get(k, False) for k in [
                "noise_gate_enabled", "highpass_enabled", "compressor_enabled", "gain_enabled"
            ]):
                processed_audio_path = self.apply_pedalboard(processed_audio_path)

            if audio_ready_callback is not None:
                try:
                    audio_ready_callback(str(processed_audio_path))
                except Exception:
                    pass

        result = self.transcribe(processed_audio_path, model_size)
        segments = result["segments"]

        self.last_segments = segments

        if not segments:
            raise RuntimeError("No segments extracted from audio")

        if output_format == 'txt':
            self.export_text(segments, output_path)

            if not using_prebuilt:
                if temp_processed_file and temp_processed_file.exists():
                    temp_processed_file.unlink()
                self.cleanup_temp_files()

            processing_time = (datetime.datetime.now() - processing_start_time).total_seconds()
            self.log(f"\n✓ Text export completed in {processing_time:.1f}s")
            return None

        else:
            subs = self.create_subtitles_from_segments(segments)

            if not subs:
                raise RuntimeError("No subtitles generated — alignment may have failed")

            subs.save(output_path, encoding="utf-8")

            total_duration = sum((s.end.ordinal - s.start.ordinal) / 1000 for s in subs)
            avg_duration = total_duration / len(subs) if subs else 0
            processing_time = (datetime.datetime.now() - processing_start_time).total_seconds()

            self.log(f"\n✓ Saved {len(subs)} subtitles → {output_path}")
            self.log(f"  Total subtitle time: {total_duration:.1f}s")
            self.log(f"  Average subtitle: {avg_duration:.2f}s")
            self.log(f"  Total processing time: {processing_time:.1f}s")

            if not using_prebuilt:
                if temp_processed_file and temp_processed_file.exists():
                    temp_processed_file.unlink()
                self.cleanup_temp_files()

            return subs

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    text_exported = pyqtSignal(str)
    whisper_audio_ready = pyqtSignal(str)
    check_audio_done = pyqtSignal()

class SubtitleGeneratorGUI(QMainWindow):
    audio_loaded_signal = pyqtSignal(np.ndarray)
    whisper_audio_loaded_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subtitle Generator by Mubumbutu")
        self.setMinimumSize(1050, 300)
        self.setAcceptDrops(True)

        self.input_file = None
        self.output_file = None
        self.processing = False
        self.generator = None

        self.audio_data = None
        self.playing = False
        self.current_position = 0
        self.playback_thread = None
        self.stop_playback_flag = False

        self.whisper_audio_data = None
        self.whisper_playing = False
        self.whisper_current_position = 0
        self.whisper_playback_thread = None
        self.whisper_stop_playback_flag = False

        self._updating_from_waveform = False

        self._last_checked_audio_proc_state = None
        self._audio_proc_waveform_ready = False
        self._cached_processed_audio_path = None
        self._cached_audio_include_ranges = None
        self._cached_audio_exclude_ranges = None

        self._io_user_opened = False
        self._waveform_user_opened = False
        self._whisper_user_opened = False

        self.audio_loaded_signal.connect(self._update_waveform_data)
        self.whisper_audio_loaded_signal.connect(self._update_whisper_waveform_data)

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
        self.check_audio_btn = QPushButton("🎧 Check Audio")
        self.check_audio_btn.setMinimumHeight(40)
        self.check_audio_btn.setStyleSheet("""
            QPushButton {
                background-color: #6A1B9A;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4A148C;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        """)
        self.check_audio_btn.clicked.connect(self.start_check_audio)
        self.check_audio_btn.setVisible(False)
        button_layout.addWidget(self.check_audio_btn)
        self.button_container.setVisible(False)
        layout.addWidget(self.button_container)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        self.voice_separation_check.toggled.connect(self._on_audio_proc_changed)
        self.reuse_check.toggled.connect(self._on_reuse_toggled)
        self.reuse_fixed_check.toggled.connect(self._on_audio_proc_changed)
        self.reuse_smart_check.toggled.connect(self._on_audio_proc_changed)
        self.reuse_fixed_sec_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.reuse_smart_sec_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.ng_check.toggled.connect(self._on_audio_proc_changed)
        self.ng_threshold_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.ng_release_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.hp_check.toggled.connect(self._on_audio_proc_changed)
        self.hp_cutoff_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.comp_check.toggled.connect(self._on_audio_proc_changed)
        self.comp_threshold_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.comp_ratio_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.gain_check.toggled.connect(self._on_audio_proc_changed)
        self.gain_db_spin.valueChanged.connect(self._on_audio_proc_changed)
        self.bwe_combo.currentIndexChanged.connect(self._on_audio_proc_changed)
        self.adjustSize()

    def _create_reuse_chunk_container(self, checkbox_style: str, spinbox_style: str, label_style: str) -> QWidget:
        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(0, 2, 0, 4)
        outer_layout.setSpacing(4)

        chunk_grid = QGridLayout()
        chunk_grid.setSpacing(8)
        chunk_grid.setColumnMinimumWidth(0, 210)
        chunk_grid.setColumnMinimumWidth(1, 100)
        chunk_grid.setColumnStretch(2, 1)

        self.reuse_fixed_check = QCheckBox("Fixed chunking:")
        self.reuse_fixed_check.setChecked(False)
        self.reuse_fixed_check.setStyleSheet(checkbox_style)
        self.reuse_fixed_check.setToolTip(
            "Split audio into equal fixed-length segments before RE-USE.\n"
            "Each segment is enhanced separately, then joined with crossfade.\n"
            "Hard cuts — segments are exactly N seconds long."
        )
        self.reuse_fixed_sec_spin = QDoubleSpinBox()
        self.reuse_fixed_sec_spin.setRange(5.0, 600.0)
        self.reuse_fixed_sec_spin.setValue(15.0)
        self.reuse_fixed_sec_spin.setSuffix(" s")
        self.reuse_fixed_sec_spin.setSingleStep(5.0)
        self.reuse_fixed_sec_spin.setFixedWidth(100)
        self.reuse_fixed_sec_spin.setStyleSheet(spinbox_style)
        self.reuse_fixed_sec_spin.setEnabled(False)
        fixed_hint = QLabel("(split every N seconds, hard cuts)")
        fixed_hint.setStyleSheet(label_style)

        chunk_grid.addWidget(self.reuse_fixed_check, 0, 0)
        chunk_grid.addWidget(self.reuse_fixed_sec_spin, 0, 1)
        chunk_grid.addWidget(fixed_hint, 0, 2)

        self.reuse_smart_check = QCheckBox("Smart chunking:")
        self.reuse_smart_check.setChecked(False)
        self.reuse_smart_check.setStyleSheet(checkbox_style)
        self.reuse_smart_check.setToolTip(
            "Split audio at the quietest detected point within each N-second window.\n"
            "Searches the second half of each window for the lowest RMS frame.\n"
            "Reduces the chance of cutting mid-word. Falls back to hard cut if needed."
        )
        self.reuse_smart_sec_spin = QDoubleSpinBox()
        self.reuse_smart_sec_spin.setRange(5.0, 600.0)
        self.reuse_smart_sec_spin.setValue(15.0)
        self.reuse_smart_sec_spin.setSuffix(" s")
        self.reuse_smart_sec_spin.setSingleStep(5.0)
        self.reuse_smart_sec_spin.setFixedWidth(100)
        self.reuse_smart_sec_spin.setStyleSheet(spinbox_style)
        self.reuse_smart_sec_spin.setEnabled(False)
        smart_hint = QLabel("(split at quietest point within N seconds)")
        smart_hint.setStyleSheet(label_style)

        chunk_grid.addWidget(self.reuse_smart_check, 1, 0)
        chunk_grid.addWidget(self.reuse_smart_sec_spin, 1, 1)
        chunk_grid.addWidget(smart_hint, 1, 2)

        outer_layout.addLayout(chunk_grid)

        self.reuse_fixed_check.toggled.connect(self.reuse_fixed_sec_spin.setEnabled)
        self.reuse_fixed_check.toggled.connect(self._on_reuse_fixed_toggled)
        self.reuse_smart_check.toggled.connect(self.reuse_smart_sec_spin.setEnabled)
        self.reuse_smart_check.toggled.connect(self._on_reuse_smart_toggled)

        container.setVisible(False)
        return container

    def create_main_tab(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #1a1a1a; color: #cccccc;")
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
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
        CONTENT_GROUP_STYLE = """
            QGroupBox {
                background-color: #1c1c1c;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                margin-top: 0px;
            }
        """
        def make_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(LABEL_STYLE)
            return lbl
        def make_fixed_label(text, width):
            lbl = QLabel(text)
            lbl.setStyleSheet(LABEL_STYLE)
            lbl.setFixedWidth(width)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return lbl

        # ─────────────────────────────────────────────────
        # INPUT / OUTPUT SECTION
        # ─────────────────────────────────────────────────
        io_container = QWidget()
        io_container.setStyleSheet("background: transparent;")
        io_main_layout = QVBoxLayout(io_container)
        io_main_layout.setContentsMargins(0, 0, 0, 0)
        io_main_layout.setSpacing(4)
        self.io_toggle = QPushButton("▼ 📁 Input / Output")
        self.io_toggle.setCheckable(True)
        self.io_toggle.setChecked(True)
        self.io_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.io_toggle.clicked.connect(self.toggle_io_section)
        io_main_layout.addWidget(self.io_toggle)
        self.io_content = QGroupBox()
        self.io_content.setStyleSheet(CONTENT_GROUP_STYLE)
        io_content_layout = QVBoxLayout()
        io_content_layout.setSpacing(8)
        io_content_layout.setContentsMargins(10, 14, 10, 10)
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
        io_content_layout.addLayout(input_layout)
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
        io_content_layout.addLayout(rec_layout)
        self.io_content.setLayout(io_content_layout)
        io_main_layout.addWidget(self.io_content)
        layout.addWidget(io_container)

        # ─────────────────────────────────────────────────
        # AUDIO WAVEFORM SECTION
        # ─────────────────────────────────────────────────
        self.waveform_toggle_container = QWidget()
        self.waveform_toggle_container.setStyleSheet("background: transparent;")
        self.waveform_toggle_container.setVisible(False)
        waveform_main_layout = QVBoxLayout(self.waveform_toggle_container)
        waveform_main_layout.setContentsMargins(0, 0, 0, 0)
        waveform_main_layout.setSpacing(4)
        self.waveform_toggle = QPushButton("▼ 🌊 Audio Waveform")
        self.waveform_toggle.setCheckable(True)
        self.waveform_toggle.setChecked(True)
        self.waveform_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.waveform_toggle.clicked.connect(self.toggle_waveform_section)
        waveform_main_layout.addWidget(self.waveform_toggle)
        self.waveform_content = QGroupBox()
        self.waveform_content.setStyleSheet(CONTENT_GROUP_STYLE)
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
        self.waveform_content.setLayout(waveform_layout)
        waveform_main_layout.addWidget(self.waveform_content)
        layout.addWidget(self.waveform_toggle_container)

        # ─────────────────────────────────────────────────
        # FRAGMENTS SECTION
        # ─────────────────────────────────────────────────
        self.fragments_container = QWidget()
        self.fragments_container.setStyleSheet("background: transparent;")
        fragments_main_layout = QVBoxLayout(self.fragments_container)
        fragments_main_layout.setContentsMargins(0, 0, 0, 0)
        fragments_main_layout.setSpacing(4)
        self.fragments_toggle = QPushButton("▶ 🎯 Optional: Select Fragments")
        self.fragments_toggle.setCheckable(True)
        self.fragments_toggle.setChecked(False)
        self.fragments_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.fragments_toggle.clicked.connect(self.toggle_fragments_settings)
        fragments_main_layout.addWidget(self.fragments_toggle)
        self.range_group = self.create_range_section()
        self.range_group.setVisible(False)
        fragments_main_layout.addWidget(self.range_group)
        self.fragments_container.setVisible(False)
        layout.addWidget(self.fragments_container)

        # ─────────────────────────────────────────────────
        # SETTINGS ROW
        # ─────────────────────────────────────────────────
        model_group_container = QWidget()
        model_group_container.setStyleSheet("background: transparent;")
        model_group_main_layout = QVBoxLayout(model_group_container)
        model_group_main_layout.setContentsMargins(0, 0, 0, 0)
        model_group_main_layout.setSpacing(4)
        self.model_settings_toggle = QPushButton("▶ ⚙️ Model Settings")
        self.model_settings_toggle.setCheckable(True)
        self.model_settings_toggle.setChecked(False)
        self.model_settings_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.model_settings_toggle.clicked.connect(self.toggle_model_settings)
        model_group_main_layout.addWidget(self.model_settings_toggle)
        self.model_settings_content = QGroupBox()
        self.model_settings_content.setVisible(False)
        self.model_settings_content.setStyleSheet(CONTENT_GROUP_STYLE)
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
        self.model_combo.setCurrentText("large-v3")
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
        model_layout.addStretch()
        self.model_settings_content.setLayout(model_layout)
        model_group_main_layout.addWidget(self.model_settings_content)

        # Audio Processing
        audio_proc_group_container = QWidget()
        audio_proc_group_container.setStyleSheet("background: transparent;")
        audio_proc_group_main_layout = QVBoxLayout(audio_proc_group_container)
        audio_proc_group_main_layout.setContentsMargins(0, 0, 0, 0)
        audio_proc_group_main_layout.setSpacing(4)
        self.audio_proc_settings_toggle = QPushButton("▶ 🎛️ Audio Processing")
        self.audio_proc_settings_toggle.setCheckable(True)
        self.audio_proc_settings_toggle.setChecked(False)
        self.audio_proc_settings_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.audio_proc_settings_toggle.clicked.connect(self.toggle_audio_processing)
        audio_proc_group_main_layout.addWidget(self.audio_proc_settings_toggle)
        self.audio_proc_settings_content = QGroupBox()
        self.audio_proc_settings_content.setVisible(False)
        self.audio_proc_settings_content.setStyleSheet(CONTENT_GROUP_STYLE)
        audio_proc_layout = QVBoxLayout()
        audio_proc_layout.setSpacing(10)
        audio_proc_layout.setContentsMargins(15, 12, 15, 12)

        voice_sep_layout = QHBoxLayout()
        voice_sep_layout.setSpacing(8)
        self.voice_separation_check = QCheckBox("🎵 Voice Separation (Demucs)")
        self.voice_separation_check.setChecked(False)
        self.voice_separation_check.setStyleSheet(CHECKBOX_STYLE)
        self.voice_separation_check.setToolTip(
            "Extract vocals from audio before transcription.\n"
            "Improves accuracy for files with background music or noise."
        )
        voice_sep_layout.addWidget(self.voice_separation_check)
        voice_sep_layout.addWidget(make_label("(Separates vocal from audio — only the vocal is processed.)"))
        voice_sep_layout.addStretch()
        audio_proc_layout.addLayout(voice_sep_layout)

        reuse_layout = QHBoxLayout()
        reuse_layout.setSpacing(8)
        self.reuse_check = QCheckBox("🔊 Voice Enhancement (RE-USE)")
        self.reuse_check.setChecked(False)
        self.reuse_check.setStyleSheet(CHECKBOX_STYLE)
        self.reuse_check.setToolTip(
            "Apply NVIDIA RE-USE neural speech enhancement before transcription.\n"
            "Significantly improves speech clarity in noisy or degraded recordings.\n"
            "Applied after Voice Separation (if enabled), before Audio Effects.\n"
            "Enhanced audio is saved permanently to RE-USE_outputs/ folder.\n"
            "Model is downloaded automatically on first use."
        )
        reuse_layout.addWidget(self.reuse_check)
        reuse_layout.addWidget(make_label("(Neural voice enhancement — high VRAM requirement)"))
        reuse_layout.addStretch()
        audio_proc_layout.addLayout(reuse_layout)

        # Chunking BEFORE Bandwidth Extension
        self.reuse_chunk_container = self._create_reuse_chunk_container(
            CHECKBOX_STYLE, SPINBOX_STYLE, LABEL_STYLE
        )
        audio_proc_layout.addWidget(self.reuse_chunk_container)

        # Bandwidth Extension (teraz POD chunkingiem)
        self.bwe_container = QWidget()
        bwe_layout = QHBoxLayout(self.bwe_container)
        bwe_layout.setSpacing(8)
        bwe_lbl = make_label("Bandwidth Extension:")
        bwe_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        self.bwe_combo = QComboBox()
        self.bwe_combo.addItem("Disabled (preserve original SR)", 0)
        self.bwe_combo.addItem("→ 8 kHz", 8000)
        self.bwe_combo.addItem("→ 16 kHz (best for WhisperX)", 16000)
        self.bwe_combo.addItem("→ 22 kHz", 22050)
        self.bwe_combo.addItem("→ 24 kHz", 24000)
        self.bwe_combo.addItem("→ 32 kHz", 32000)
        self.bwe_combo.addItem("→ 44.1 kHz", 44100)
        self.bwe_combo.addItem("→ 48 kHz (max)", 48000)
        self.bwe_combo.setCurrentIndex(0)
        self.bwe_combo.setStyleSheet(COMBO_STYLE)
        self.bwe_combo.setFixedWidth(220)
        bwe_layout.addWidget(bwe_lbl)
        bwe_layout.addWidget(self.bwe_combo)
        bwe_layout.addStretch()
        audio_proc_layout.addWidget(self.bwe_container)
        self.bwe_container.setVisible(False)

        sep_ap1 = QFrame()
        sep_ap1.setFrameShape(QFrame.Shape.HLine)
        sep_ap1.setStyleSheet("background-color: #2e2e2e; border: none; max-height: 1px;")
        audio_proc_layout.addWidget(sep_ap1)

        pb_header = QLabel("Audio Effects")
        pb_header.setStyleSheet(
            "color: #aaaaaa; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
        audio_proc_layout.addWidget(pb_header)

        pb_grid = QGridLayout()
        pb_grid.setSpacing(8)
        pb_grid.setColumnMinimumWidth(0, 130)
        pb_grid.setColumnMinimumWidth(1, 80)
        pb_grid.setColumnMinimumWidth(2, 100)
        pb_grid.setColumnMinimumWidth(3, 60)
        pb_grid.setColumnMinimumWidth(4, 110)
        pb_grid.setColumnStretch(6, 1)

        self.ng_check = QCheckBox("Noise Gate")
        self.ng_check.setChecked(False)
        self.ng_check.setStyleSheet(CHECKBOX_STYLE)
        self.ng_check.setToolTip(
            "Silences audio that falls below the threshold.\n"
            "Effective at removing background hiss and noise between words."
        )
        pb_grid.addWidget(self.ng_check, 0, 0)
        pb_grid.addWidget(make_fixed_label("Threshold:", 80), 0, 1)
        self.ng_threshold_spin = QDoubleSpinBox()
        self.ng_threshold_spin.setRange(-80.0, 0.0)
        self.ng_threshold_spin.setValue(-38.0)
        self.ng_threshold_spin.setSuffix(" dB")
        self.ng_threshold_spin.setSingleStep(1.0)
        self.ng_threshold_spin.setFixedWidth(100)
        self.ng_threshold_spin.setStyleSheet(SPINBOX_STYLE)
        self.ng_threshold_spin.setEnabled(False)
        pb_grid.addWidget(self.ng_threshold_spin, 0, 2)
        pb_grid.addWidget(make_fixed_label("Release:", 60), 0, 3)
        self.ng_release_spin = QDoubleSpinBox()
        self.ng_release_spin.setRange(10.0, 2000.0)
        self.ng_release_spin.setValue(180.0)
        self.ng_release_spin.setSuffix(" ms")
        self.ng_release_spin.setSingleStep(10.0)
        self.ng_release_spin.setFixedWidth(110)
        self.ng_release_spin.setStyleSheet(SPINBOX_STYLE)
        self.ng_release_spin.setEnabled(False)
        pb_grid.addWidget(self.ng_release_spin, 0, 4)
        self.ng_check.toggled.connect(self.ng_threshold_spin.setEnabled)
        self.ng_check.toggled.connect(self.ng_release_spin.setEnabled)

        self.hp_check = QCheckBox("High-Pass Filter")
        self.hp_check.setChecked(False)
        self.hp_check.setStyleSheet(CHECKBOX_STYLE)
        self.hp_check.setToolTip(
            "Removes frequencies below the cutoff.\n"
            "Useful for eliminating low-frequency rumble, hum, and microphone vibration."
        )
        pb_grid.addWidget(self.hp_check, 1, 0)
        pb_grid.addWidget(make_fixed_label("Cutoff:", 80), 1, 1)
        self.hp_cutoff_spin = QDoubleSpinBox()
        self.hp_cutoff_spin.setRange(20.0, 1000.0)
        self.hp_cutoff_spin.setValue(82.0)
        self.hp_cutoff_spin.setSuffix(" Hz")
        self.hp_cutoff_spin.setSingleStep(5.0)
        self.hp_cutoff_spin.setFixedWidth(100)
        self.hp_cutoff_spin.setStyleSheet(SPINBOX_STYLE)
        self.hp_cutoff_spin.setEnabled(False)
        pb_grid.addWidget(self.hp_cutoff_spin, 1, 2)
        self.hp_check.toggled.connect(self.hp_cutoff_spin.setEnabled)

        self.comp_check = QCheckBox("Compressor")
        self.comp_check.setChecked(False)
        self.comp_check.setStyleSheet(CHECKBOX_STYLE)
        self.comp_check.setToolTip(
            "Reduces the dynamic range of the audio.\n"
            "Brings quieter speech to a more consistent level, improving Whisper accuracy."
        )
        pb_grid.addWidget(self.comp_check, 2, 0)
        pb_grid.addWidget(make_fixed_label("Threshold:", 80), 2, 1)
        self.comp_threshold_spin = QDoubleSpinBox()
        self.comp_threshold_spin.setRange(-60.0, 0.0)
        self.comp_threshold_spin.setValue(-23.0)
        self.comp_threshold_spin.setSuffix(" dB")
        self.comp_threshold_spin.setSingleStep(1.0)
        self.comp_threshold_spin.setFixedWidth(100)
        self.comp_threshold_spin.setStyleSheet(SPINBOX_STYLE)
        self.comp_threshold_spin.setEnabled(False)
        pb_grid.addWidget(self.comp_threshold_spin, 2, 2)
        pb_grid.addWidget(make_fixed_label("Ratio:", 60), 2, 3)
        self.comp_ratio_spin = QDoubleSpinBox()
        self.comp_ratio_spin.setRange(1.0, 20.0)
        self.comp_ratio_spin.setValue(4.5)
        self.comp_ratio_spin.setSuffix(" :1")
        self.comp_ratio_spin.setSingleStep(0.5)
        self.comp_ratio_spin.setFixedWidth(110)
        self.comp_ratio_spin.setStyleSheet(SPINBOX_STYLE)
        self.comp_ratio_spin.setEnabled(False)
        pb_grid.addWidget(self.comp_ratio_spin, 2, 4)
        self.comp_check.toggled.connect(self.comp_threshold_spin.setEnabled)
        self.comp_check.toggled.connect(self.comp_ratio_spin.setEnabled)

        self.gain_check = QCheckBox("Gain")
        self.gain_check.setChecked(False)
        self.gain_check.setStyleSheet(CHECKBOX_STYLE)
        self.gain_check.setToolTip("Adjusts the overall volume level of the audio.")
        pb_grid.addWidget(self.gain_check, 3, 0)
        pb_grid.addWidget(make_fixed_label("Gain:", 80), 3, 1)
        self.gain_db_spin = QDoubleSpinBox()
        self.gain_db_spin.setRange(-20.0, 20.0)
        self.gain_db_spin.setValue(3.0)
        self.gain_db_spin.setSuffix(" dB")
        self.gain_db_spin.setSingleStep(0.5)
        self.gain_db_spin.setFixedWidth(100)
        self.gain_db_spin.setStyleSheet(SPINBOX_STYLE)
        self.gain_db_spin.setEnabled(False)
        pb_grid.addWidget(self.gain_db_spin, 3, 2)
        self.gain_check.toggled.connect(self.gain_db_spin.setEnabled)

        audio_proc_layout.addLayout(pb_grid)
        audio_proc_layout.addStretch()

        self.audio_proc_settings_content.setLayout(audio_proc_layout)
        audio_proc_group_main_layout.addWidget(self.audio_proc_settings_content)

        # Advanced Settings
        advanced_group_container = QWidget()
        advanced_group_container.setStyleSheet("background: transparent;")
        advanced_group_main_layout = QVBoxLayout(advanced_group_container)
        advanced_group_main_layout.setContentsMargins(0, 0, 0, 0)
        advanced_group_main_layout.setSpacing(4)
        self.advanced_settings_toggle = QPushButton("▶ 🔧 Advanced Settings")
        self.advanced_settings_toggle.setCheckable(True)
        self.advanced_settings_toggle.setChecked(False)
        self.advanced_settings_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.advanced_settings_toggle.clicked.connect(self.toggle_advanced_settings)
        advanced_group_main_layout.addWidget(self.advanced_settings_toggle)
        self.advanced_settings_content = QGroupBox()
        self.advanced_settings_content.setVisible(False)
        self.advanced_settings_content.setStyleSheet(CONTENT_GROUP_STYLE)
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(10)
        advanced_layout.setContentsMargins(15, 12, 15, 12)
        adv_grid = QGridLayout()
        adv_grid.setSpacing(8)
        adv_grid.setColumnMinimumWidth(0, 155)
        adv_grid.setColumnMinimumWidth(1, 150)
        adv_grid.setColumnStretch(2, 1)
        self.pattern_edit = QLineEdit("3,4")
        self.pattern_edit.setMaximumWidth(150)
        self.pattern_edit.setStyleSheet(INPUT_STYLE)
        label_word_pattern = make_fixed_label("Word Pattern:", 155)
        label_word_pattern.setToolTip("e.g., 3,4 = alternating 3 and 4 words per subtitle")
        adv_grid.addWidget(label_word_pattern, 0, 0)
        adv_grid.addWidget(self.pattern_edit, 0, 1)
        self.min_pause_spin = QDoubleSpinBox()
        self.min_pause_spin.setRange(0.1, 3.0)
        self.min_pause_spin.setSingleStep(0.1)
        self.min_pause_spin.setValue(0.6)
        self.min_pause_spin.setSuffix(" s")
        self.min_pause_spin.setFixedWidth(100)
        self.min_pause_spin.setStyleSheet(SPINBOX_STYLE)
        label_min_pause = make_fixed_label("Min Pause for Split:", 155)
        label_min_pause.setToolTip("Pause between words to split subtitle")
        adv_grid.addWidget(label_min_pause, 1, 0)
        adv_grid.addWidget(self.min_pause_spin, 1, 1)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.5, 5.0)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setValue(1.0)
        self.min_duration_spin.setSuffix(" s")
        self.min_duration_spin.setFixedWidth(100)
        self.min_duration_spin.setStyleSheet(SPINBOX_STYLE)
        label_min_duration = make_fixed_label("Min Subtitle Duration:", 155)
        label_min_duration.setToolTip("Merge subtitles shorter than this duration")
        adv_grid.addWidget(label_min_duration, 2, 0)
        adv_grid.addWidget(self.min_duration_spin, 2, 1)
        self.max_line_spin = QSpinBox()
        self.max_line_spin.setRange(20, 100)
        self.max_line_spin.setValue(42)
        self.max_line_spin.setSuffix(" chars")
        self.max_line_spin.setFixedWidth(100)
        self.max_line_spin.setStyleSheet(SPINBOX_STYLE)
        label_max_line = make_fixed_label("Max Line Length:", 155)
        label_max_line.setToolTip("Max characters per line, film standard = 42")
        adv_grid.addWidget(label_max_line, 3, 0)
        adv_grid.addWidget(self.max_line_spin, 3, 1)
        advanced_layout.addLayout(adv_grid)
        advanced_layout.addStretch()
        self.advanced_settings_content.setLayout(advanced_layout)
        advanced_group_main_layout.addWidget(self.advanced_settings_content)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(10)
        settings_row.addWidget(model_group_container, 1)
        settings_row.addWidget(audio_proc_group_container, 1)
        settings_row.addWidget(advanced_group_container, 1)
        layout.addLayout(settings_row)

        # ─────────────────────────────────────────────────
        # WHISPER INPUT PREVIEW SECTION
        # ─────────────────────────────────────────────────
        self.whisper_preview_container = QWidget()
        self.whisper_preview_container.setStyleSheet("background: transparent;")
        whisper_preview_main_layout = QVBoxLayout(self.whisper_preview_container)
        whisper_preview_main_layout.setContentsMargins(0, 0, 0, 0)
        whisper_preview_main_layout.setSpacing(4)
        self.whisper_preview_toggle = QPushButton("▶ 🎵 Whisper Input Preview")
        self.whisper_preview_toggle.setCheckable(True)
        self.whisper_preview_toggle.setChecked(False)
        self.whisper_preview_toggle.setStyleSheet(TOGGLE_BTN_STYLE)
        self.whisper_preview_toggle.clicked.connect(self.toggle_whisper_preview)
        whisper_preview_main_layout.addWidget(self.whisper_preview_toggle)
        self.whisper_preview_group = QGroupBox()
        self.whisper_preview_group.setVisible(False)
        self.whisper_preview_group.setStyleSheet(CONTENT_GROUP_STYLE)
        whisper_wv_layout = QVBoxLayout()
        whisper_wv_layout.setSpacing(6)
        whisper_wv_layout.setContentsMargins(10, 12, 10, 10)
        whisper_desc = QLabel(
            "Audio sent directly to Whisper after all processing (region selection, voice separation)."
        )
        whisper_desc.setStyleSheet("color: #555555; font-size: 10px; font-style: italic;")
        whisper_wv_layout.addWidget(whisper_desc)
        whisper_playback_row = QHBoxLayout()
        whisper_playback_row.setSpacing(6)
        self.whisper_play_btn = QPushButton("▶ Play")
        self.whisper_play_btn.setFixedWidth(85)
        self.whisper_play_btn.setStyleSheet("""
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
        self.whisper_play_btn.clicked.connect(self.toggle_whisper_playback)
        whisper_playback_row.addWidget(self.whisper_play_btn)
        self.whisper_stop_btn = QPushButton("⏹ Stop")
        self.whisper_stop_btn.setFixedWidth(85)
        self.whisper_stop_btn.setStyleSheet(BTN_STYLE)
        self.whisper_stop_btn.clicked.connect(self.stop_whisper_playback)
        whisper_playback_row.addWidget(self.whisper_stop_btn)
        self.whisper_reset_btn = QPushButton("⏮ Reset")
        self.whisper_reset_btn.setFixedWidth(85)
        self.whisper_reset_btn.setStyleSheet(BTN_STYLE)
        self.whisper_reset_btn.clicked.connect(self.reset_whisper_playback)
        whisper_playback_row.addWidget(self.whisper_reset_btn)
        whisper_playback_row.addWidget(make_label("Position:"))
        self.whisper_position_label = QLabel("00:00:00")
        self.whisper_position_label.setStyleSheet("color: #e0e0e0; font-size: 12px; font-weight: bold;")
        whisper_playback_row.addWidget(self.whisper_position_label)
        whisper_playback_row.addStretch()
        whisper_wv_layout.addLayout(whisper_playback_row)
        self.whisper_waveform_display = WaveformWidget()
        self.whisper_waveform_display.selection_enabled = False
        self.whisper_waveform_display.view_changed.connect(self.update_whisper_scrollbar)
        self.whisper_waveform_display.seek_requested.connect(self.on_whisper_seek_requested)
        whisper_wv_layout.addWidget(self.whisper_waveform_display)
        self.whisper_scrollbar_container = QWidget()
        whisper_scroll_layout = QHBoxLayout(self.whisper_scrollbar_container)
        whisper_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.whisper_waveform_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.whisper_waveform_scroll.setMinimum(0)
        self.whisper_waveform_scroll.setMaximum(1000)
        self.whisper_waveform_scroll.setValue(0)
        self.whisper_waveform_scroll.setStyleSheet("""
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
        self.whisper_waveform_scroll.sliderMoved.connect(self.on_whisper_scroll_user_change)
        self.whisper_waveform_scroll.valueChanged.connect(self.on_whisper_scroll_user_change)
        self.whisper_scrollbar_container.setVisible(False)
        whisper_scroll_layout.addWidget(self.whisper_waveform_scroll)
        whisper_wv_layout.addWidget(self.whisper_scrollbar_container)
        self.whisper_waveform_info = QLabel("Status: Waiting for generation to start...")
        self.whisper_waveform_info.setStyleSheet("color: #555555; font-size: 10px;")
        whisper_wv_layout.addWidget(self.whisper_waveform_info)
        self.whisper_preview_group.setLayout(whisper_wv_layout)
        whisper_preview_main_layout.addWidget(self.whisper_preview_group)
        self.whisper_preview_container.setVisible(False)
        layout.addWidget(self.whisper_preview_container)

        layout.addStretch()
        return widget

    def toggle_fragments_settings(self):
        is_visible = self.fragments_toggle.isChecked()
        self.range_group.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.fragments_toggle.setText(f"{arrow}  🎯  Optional: Select Fragments")
        self.range_group.updateGeometry()
        QApplication.processEvents()
        QTimer.singleShot(40, self._adjust_window_size)

    def toggle_model_settings(self):
        is_visible = self.model_settings_toggle.isChecked()
        self.model_settings_content.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.model_settings_toggle.setText(f"{arrow} ⚙️ Model Settings")

        self.model_settings_content.updateGeometry()
        QApplication.processEvents()

        QTimer.singleShot(40, lambda: self._adjust_window_size())

    def toggle_audio_processing(self):
        is_visible = self.audio_proc_settings_toggle.isChecked()
        self.audio_proc_settings_content.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.audio_proc_settings_toggle.setText(f"{arrow}  🎛️  Audio Processing")
 
        self.audio_proc_settings_content.updateGeometry()
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

    def toggle_io_section(self):
        is_visible = self.io_toggle.isChecked()
        self.io_content.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.io_toggle.setText(f"{arrow} 📁 Input / Output")
        if is_visible:
            self._io_user_opened = True
        QTimer.singleShot(50, self._adjust_window_size)

    def toggle_waveform_section(self):
        is_visible = self.waveform_toggle.isChecked()
        self.waveform_content.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.waveform_toggle.setText(f"{arrow} 🌊 Audio Waveform")
        if is_visible:
            self._waveform_user_opened = True
        QTimer.singleShot(50, self._adjust_window_size)

    def _auto_expand_waveform(self):
        if not self._io_user_opened and self.io_toggle.isChecked():
            self.io_toggle.setChecked(False)
            self.io_content.setVisible(False)
            self.io_toggle.setText("▶ 📁 Input / Output")

        self.waveform_toggle_container.setVisible(True)
        if not self.waveform_toggle.isChecked():
            self.waveform_toggle.setChecked(True)
            self.waveform_content.setVisible(True)
            self.waveform_toggle.setText("▼ 🌊 Audio Waveform")

        self.fragments_container.setVisible(True)
        QTimer.singleShot(50, self._adjust_window_size)

    def _auto_expand_whisper(self):
        if not self._waveform_user_opened and self.waveform_toggle.isChecked():
            self.waveform_toggle.setChecked(False)
            self.waveform_content.setVisible(False)
            self.waveform_toggle.setText("▶ 🌊 Audio Waveform")

        self.whisper_preview_container.setVisible(True)
        if not self.whisper_preview_toggle.isChecked():
            self.whisper_preview_toggle.setChecked(True)
            self.whisper_preview_group.setVisible(True)
            self.whisper_preview_toggle.setText("▼ 🎵 Whisper Input Preview")

        QTimer.singleShot(50, self._adjust_window_size)

    def toggle_whisper_preview(self):
        is_visible = self.whisper_preview_toggle.isChecked()
        self.whisper_preview_group.setVisible(is_visible)
        arrow = "▼" if is_visible else "▶"
        self.whisper_preview_toggle.setText(f"{arrow} 🎵 Whisper Input Preview")
        if is_visible:
            self._whisper_user_opened = True
        self.whisper_preview_group.updateGeometry()
        QApplication.processEvents()
        QTimer.singleShot(50, self._adjust_window_size)

    def _on_reuse_toggled(self):
        enabled = self.reuse_check.isChecked()
        self.bwe_container.setVisible(enabled)
        self.reuse_chunk_container.setVisible(enabled)
        if self.button_container.isVisible():
            self._update_action_buttons_visibility()

    def _on_reuse_fixed_toggled(self, checked: bool):
        if checked and self.reuse_smart_check.isChecked():
            self.reuse_smart_check.blockSignals(True)
            self.reuse_smart_check.setChecked(False)
            self.reuse_smart_sec_spin.setEnabled(False)
            self.reuse_smart_check.blockSignals(False)
        if self.button_container.isVisible():
            self._update_action_buttons_visibility()

    def _on_reuse_smart_toggled(self, checked: bool):
        if checked and self.reuse_fixed_check.isChecked():
            self.reuse_fixed_check.blockSignals(True)
            self.reuse_fixed_check.setChecked(False)
            self.reuse_fixed_sec_spin.setEnabled(False)
            self.reuse_fixed_check.blockSignals(False)
        if self.button_container.isVisible():
            self._update_action_buttons_visibility()

    def _get_audio_proc_state(self):
        return (
            self.voice_separation_check.isChecked(),
            self.reuse_check.isChecked(),
            self.bwe_combo.currentData(),
            self.reuse_fixed_check.isChecked(),
            self.reuse_smart_check.isChecked(),
            self.reuse_fixed_sec_spin.value(),
            self.reuse_smart_sec_spin.value(),
            self.ng_check.isChecked(),
            self.ng_threshold_spin.value(),
            self.ng_release_spin.value(),
            self.hp_check.isChecked(),
            self.hp_cutoff_spin.value(),
            self.comp_check.isChecked(),
            self.comp_threshold_spin.value(),
            self.comp_ratio_spin.value(),
            self.gain_check.isChecked(),
            self.gain_db_spin.value(),
        )

    def _any_audio_proc_checked(self):
        return (
            self.voice_separation_check.isChecked()
            or self.reuse_check.isChecked()
            or self.ng_check.isChecked()
            or self.hp_check.isChecked()
            or self.comp_check.isChecked()
            or self.gain_check.isChecked()
        )

    def _update_action_buttons_visibility(self):
        if not self._any_audio_proc_checked():
            self.process_btn.setVisible(True)
            self.export_text_btn.setVisible(True)
            self.check_audio_btn.setVisible(False)
        else:
            current_include, current_exclude = self.parse_time_ranges()
            state_matches = (
                self._get_audio_proc_state() == self._last_checked_audio_proc_state
                and current_include == self._cached_audio_include_ranges
                and current_exclude == self._cached_audio_exclude_ranges
            )
            cache_valid = (
                self._cached_processed_audio_path is not None
                and Path(self._cached_processed_audio_path).exists()
            )
            if state_matches and cache_valid:
                self._audio_proc_waveform_ready = True
            elif not state_matches:
                self._audio_proc_waveform_ready = False
            self.check_audio_btn.setVisible(True)
            self.check_audio_btn.setEnabled(not state_matches and not self.processing)
            if self._audio_proc_waveform_ready:
                self.process_btn.setVisible(True)
                self.export_text_btn.setVisible(True)
            else:
                self.process_btn.setVisible(False)
                self.export_text_btn.setVisible(False)
                
    def _is_audio_cache_applicable(self, output_format):
        if self._cached_processed_audio_path is None:
            return False
        if not Path(self._cached_processed_audio_path).exists():
            return False
        if self._get_audio_proc_state() != self._last_checked_audio_proc_state:
            return False
        current_include, current_exclude = self.parse_time_ranges()
        if output_format == 'srt' and current_exclude and not current_include:
            return False
        if current_include != self._cached_audio_include_ranges:
            return False
        if current_exclude != self._cached_audio_exclude_ranges:
            return False
        return True

    def _on_audio_proc_changed(self):
        if self.button_container.isVisible():
            self._update_action_buttons_visibility()

    def start_check_audio(self):
        if not self.input_file or not self.input_file.exists():
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return
        device = "cuda" if self.gpu_radio.isChecked() else "cpu"
        pedalboard_cfg = {
            "noise_gate_enabled": self.ng_check.isChecked(),
            "noise_gate_threshold": self.ng_threshold_spin.value(),
            "noise_gate_release": self.ng_release_spin.value(),
            "highpass_enabled": self.hp_check.isChecked(),
            "highpass_cutoff": self.hp_cutoff_spin.value(),
            "compressor_enabled": self.comp_check.isChecked(),
            "compressor_threshold": self.comp_threshold_spin.value(),
            "compressor_ratio": self.comp_ratio_spin.value(),
            "gain_enabled": self.gain_check.isChecked(),
            "gain_db": self.gain_db_spin.value(),
        }
        config = {
            "device": device,
            "voice_separation": self.voice_separation_check.isChecked(),
            "reuse_enabled": self.reuse_check.isChecked(),
            "reuse_model_dir": "./models/reuse",
            "bwe": self.bwe_combo.currentData(),
            "reuse_chunking_enabled": self.reuse_fixed_check.isChecked() or self.reuse_smart_check.isChecked(),
            "reuse_chunk_mode": "smart" if self.reuse_smart_check.isChecked() else "fixed",
            "reuse_chunk_seconds": self.reuse_smart_sec_spin.value() if self.reuse_smart_check.isChecked() else self.reuse_fixed_sec_spin.value(),
            "pedalboard": pedalboard_cfg,
            "word_pattern": [1],
            "min_pause": 0.6,
            "min_duration": 1.0,
            "max_line_length": 42,
            "batch_size": self.batch_spin.value(),
            "language": self.language_edit.text().strip()
        }
        self._last_checked_audio_proc_state = self._get_audio_proc_state()
        self.processing = True
        self.check_audio_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.export_text_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        def worker():
            try:
                signals = WorkerSignals()
                signals.whisper_audio_ready.connect(self.load_whisper_preview_waveform)
                signals.check_audio_done.connect(self.on_check_audio_finished)
                signals.error.connect(self.on_check_audio_error)
                include_ranges, exclude_ranges = self.parse_time_ranges()
                generator = SubtitleGenerator(config)
                processed_path = generator.prepare_audio(
                    self.input_file,
                    include_ranges=include_ranges,
                    exclude_ranges=exclude_ranges,
                    audio_ready_callback=lambda path: signals.whisper_audio_ready.emit(path)
                )
                self._cached_processed_audio_path = processed_path
                self._cached_audio_include_ranges = include_ranges
                self._cached_audio_exclude_ranges = exclude_ranges
                signals.check_audio_done.emit()
            except Exception as e:
                signals.error.emit(str(e) + "\n\n" + traceback.format_exc())
        thread = Thread(target=worker, daemon=True)
        thread.start()

    def on_check_audio_finished(self):
        self.processing = False
        self.process_btn.setEnabled(True)
        self.export_text_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._update_action_buttons_visibility()

    def on_check_audio_error(self, error_msg):
        self.processing = False
        self._last_checked_audio_proc_state = None
        self.browse_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._update_action_buttons_visibility()
        print("\n" + "="*60)
        print("❌ ERROR - Check Audio failed")
        print("="*60)
        print(error_msg)
        print("="*60 + "\n")
        QMessageBox.critical(self, "Error", f"Check Audio failed:\n\n{error_msg}")

    def _adjust_window_size(self):
        if self.isMaximized() or self.isFullScreen():
            return
        central = self.centralWidget()
        if central is None:
            return
        central_layout = central.layout()
        if central_layout:
            central_layout.activate()
        QApplication.processEvents()
        hint = central.sizeHint()
        if not hint.isValid() or hint.height() <= 0:
            return
        new_w = max(hint.width(), self.minimumWidth())
        new_h = max(hint.height() + 10, self.minimumHeight())
        self.resize(new_w, new_h)

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

        self._auto_expand_waveform()

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
                            encoding='utf-8',
                            errors='replace',
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
            self.playback_position_label.setText("00:00:00")
            self.waveform_display.set_playback_position(0)

            self.update_waveform_scrollbar(0.0, 1.0)

            QTimer.singleShot(50, self._adjust_window_size)

        except Exception as e:
            print(f"ERROR in _update_waveform_data: {e}")
            traceback.print_exc()

    def on_waveform_selection_changed(self, selections):
        if not selections:
            self.selection_info_label.setText("No regions selected — entire file will be processed")
            self.selection_info_label.setStyleSheet(
                "color: #888888; font-style: italic; padding: 6px 8px;"
                "background-color: #161616; border: 1px solid #2e2e2e; border-radius: 3px;"
            )
            if self.button_container.isVisible():
                self._update_action_buttons_visibility()
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
            parts.append(f"<span style='color:#9b59b6; font-weight:bold;'>{include_count} INCLUDE</span>")
        if exclude_count > 0:
            parts.append(f"<span style='color:#e74c3c; font-weight:bold;'>{exclude_count} EXCLUDE</span>")

        region_text = " + ".join(parts)
        message = f"<b style='color:#cccccc;'>{len(selections)} regions selected</b> ({region_text}) • Total: <b style='color:#cccccc;'>{time_str}</b>"

        self.selection_info_label.setText(message)
        self.selection_info_label.setStyleSheet(
            "padding: 6px 8px;"
            "background-color: #161616; border: 1px solid #2a6aaa; border-radius: 3px;"
        )
        if self.button_container.isVisible():
            self._update_action_buttons_visibility()

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
            self.input_label.setStyleSheet(
                "color: #e0e0e0; background-color: #161616;"
                "border: 1px solid #2e2e2e; border-radius: 3px;"
                "padding: 5px 8px; font-size: 12px; font-style: normal;"
            )
            self.unload_btn.setVisible(True)
            self.button_container.setVisible(True)
            self._last_checked_audio_proc_state = None
            self._audio_proc_waveform_ready = False
            self._cached_processed_audio_path = None
            self._cached_audio_include_ranges = None
            self._cached_audio_exclude_ranges = None
            self._update_action_buttons_visibility()
            self.load_waveform(self.input_file)
            QTimer.singleShot(100, self._adjust_window_size)

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
        self.pause_btn.setText("▶ Resume" if is_paused else "⏸ Pause")

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
        self._last_checked_audio_proc_state = None
        self._audio_proc_waveform_ready = False
        self._cached_processed_audio_path = None
        self._cached_audio_include_ranges = None
        self._cached_audio_exclude_ranges = None
        self._update_action_buttons_visibility()
        print(f"▶ Loading waveform for: {self.input_file}")
        self.load_waveform(self.input_file)
        QTimer.singleShot(100, self._adjust_window_size)

    def unload_file(self):
        if self.playing:
            self.stop_playback()
        if self.whisper_playing:
            self.stop_whisper_playback()
        self.input_file = None
        self.audio_data = None
        self.current_position = 0
        self.whisper_audio_data = None
        self.whisper_current_position = 0
        self._audio_proc_waveform_ready = False
        self._last_checked_audio_proc_state = None
        self._cached_processed_audio_path = None
        self._cached_audio_include_ranges = None
        self._cached_audio_exclude_ranges = None
        self._io_user_opened = False
        self._waveform_user_opened = False
        self._whisper_user_opened = False
        self.input_label.setText("Drag & drop or use Record / Browse")
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
        self.unload_btn.setVisible(False)
        self.button_container.setVisible(False)
        self.waveform_toggle_container.setVisible(False)
        self.waveform_toggle.setChecked(True)
        self.waveform_content.setVisible(True)
        self.waveform_toggle.setText("▼ 🌊 Audio Waveform")
        self.io_toggle.setChecked(True)
        self.io_content.setVisible(True)
        self.io_toggle.setText("▼ 📁 Input / Output")
        self.fragments_container.setVisible(False)
        self.fragments_toggle.setChecked(False)
        self.range_group.setVisible(False)
        self.fragments_toggle.setText("▶ 🎯 Optional: Select Fragments")
        self.waveform_display.set_audio_data(None)
        self.waveform_scrollbar_container.setVisible(False)
        self.whisper_preview_container.setVisible(False)
        self.whisper_preview_toggle.setChecked(False)
        self.whisper_preview_group.setVisible(False)
        self.whisper_preview_toggle.setText("▶ 🎵 Whisper Input Preview")
        self.whisper_waveform_display.set_audio_data(None)
        self.whisper_position_label.setText("00:00:00")
        self.whisper_waveform_info.setText("Status: Waiting for generation to start...")
        self.whisper_scrollbar_container.setVisible(False)
        print("File unloaded successfully")
        QTimer.singleShot(50, self._adjust_window_size)

    def _fmt_position(self, seconds: float) -> str:
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def on_seek_requested(self, time_seconds):
        self.current_position = max(0.0, time_seconds)
        self.playback_position_label.setText(self._fmt_position(self.current_position))
        self.waveform_display.set_playback_position(self.current_position)
        print(f"Seek to: {self.current_position:.2f}s")

    def update_waveform_scrollbar(self, offset, zoom):
        if self.waveform_display.audio_data is not None:
            duration = self.waveform_display.duration
            total_s = int(duration)
            h = total_s // 3600
            m = (total_s % 3600) // 60
            s = total_s % 60
            dur_fmt = f"{h:02d}:{m:02d}:{s:02d}"
            self.waveform_info.setText(
                f"Duration: {dur_fmt} | Zoom: {zoom:.1f}x | "
                "LMB+Drag to pan, Scroll to zoom, Click to seek"
            )

        currently_visible = self.waveform_scrollbar_container.isVisible()

        if zoom > 1.0:
            self.waveform_scroll.blockSignals(True)

            max_scroll = self.waveform_scroll.maximum()
            max_offset_val = 1.0 - (1.0 / zoom)

            if max_offset_val > 0:
                scroll_val = int((offset / max_offset_val) * max_scroll)
                self.waveform_scroll.setValue(scroll_val)
                page_step = int((1.0 / zoom) / max_offset_val * max_scroll) if max_offset_val > 0 else max_scroll
                self.waveform_scroll.setPageStep(min(page_step, max_scroll))

            self.waveform_scroll.blockSignals(False)

            if not currently_visible:
                self.waveform_scrollbar_container.setVisible(True)
                QTimer.singleShot(40, self._adjust_window_size)
        else:
            if currently_visible:
                self.waveform_scrollbar_container.setVisible(False)
                QTimer.singleShot(40, self._adjust_window_size)

    def on_scroll_user_change(self, value):
        zoom = self.waveform_display.zoom_factor
        if zoom <= 1.0: return

        max_scroll = self.waveform_scroll.maximum()
        max_offset_val = 1.0 - (1.0 / zoom)

        if max_offset_val > 0:
            new_offset = (value / max_scroll) * max_offset_val

            self.waveform_display.set_view_offset(new_offset)

    def load_whisper_preview_waveform(self, file_path_str: str):
        file_path = Path(file_path_str)
        if not file_path.exists():
            return

        self._audio_proc_waveform_ready = True
        self.whisper_waveform_info.setText("Status: Loading Whisper Preview...")

        self._auto_expand_whisper()

        def worker():
            try:
                audio_data, sample_rate = sf.read(str(file_path), dtype='float32')
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if sample_rate != 16000:
                    try:
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    except ImportError:
                        pass
                if len(audio_data) == 0:
                    return
                self.whisper_audio_loaded_signal.emit(audio_data)
            except Exception as e:
                print(f"ERROR loading whisper preview: {e}")
                QTimer.singleShot(0, lambda: self.whisper_waveform_info.setText(f"Status: Error — {str(e)}"))

        thread = Thread(target=worker, daemon=True)
        thread.start()
 
    def _update_whisper_waveform_data(self, audio_data):
        try:
            self.whisper_audio_data = audio_data
            self.whisper_waveform_display.set_audio_data(audio_data)
            self.whisper_current_position = 0
            self.whisper_position_label.setText("00:00:00")
            self.whisper_waveform_display.set_playback_position(0)
            self.update_whisper_scrollbar(0.0, 1.0)
        except Exception as e:
            print(f"ERROR in _update_whisper_waveform_data: {e}")
            traceback.print_exc()
 
    def update_whisper_scrollbar(self, offset, zoom):
        if self.whisper_waveform_display.audio_data is not None:
            duration = self.whisper_waveform_display.duration
            total_s = int(duration)
            h = total_s // 3600
            m = (total_s % 3600) // 60
            s = total_s % 60
            dur_fmt = f"{h:02d}:{m:02d}:{s:02d}"
            self.whisper_waveform_info.setText(
                f"Duration: {dur_fmt} | Zoom: {zoom:.1f}x | "
                "LMB+Drag to pan, Scroll to zoom, Click to seek"
            )
 
        currently_visible = self.whisper_scrollbar_container.isVisible()
 
        if zoom > 1.0:
            self.whisper_waveform_scroll.blockSignals(True)
            max_scroll = self.whisper_waveform_scroll.maximum()
            max_offset_val = 1.0 - (1.0 / zoom)
            if max_offset_val > 0:
                scroll_val = int((offset / max_offset_val) * max_scroll)
                self.whisper_waveform_scroll.setValue(scroll_val)
                page_step = int((1.0 / zoom) / max_offset_val * max_scroll) if max_offset_val > 0 else max_scroll
                self.whisper_waveform_scroll.setPageStep(min(page_step, max_scroll))
            self.whisper_waveform_scroll.blockSignals(False)
            if not currently_visible:
                self.whisper_scrollbar_container.setVisible(True)
                QTimer.singleShot(40, self._adjust_window_size)
        else:
            if currently_visible:
                self.whisper_scrollbar_container.setVisible(False)
                QTimer.singleShot(40, self._adjust_window_size)
 
    def on_whisper_scroll_user_change(self, value):
        zoom = self.whisper_waveform_display.zoom_factor
        if zoom <= 1.0:
            return
        max_scroll = self.whisper_waveform_scroll.maximum()
        max_offset_val = 1.0 - (1.0 / zoom)
        if max_offset_val > 0:
            new_offset = (value / max_scroll) * max_offset_val
            self.whisper_waveform_display.set_view_offset(new_offset)
 
    def on_whisper_seek_requested(self, time_seconds):
        self.whisper_current_position = max(0.0, time_seconds)
        self.whisper_position_label.setText(self._fmt_position(self.whisper_current_position))
        self.whisper_waveform_display.set_playback_position(self.whisper_current_position)
 
    def toggle_whisper_playback(self):
        if self.whisper_audio_data is None:
            return
        if not self.whisper_playing:
            self.start_whisper_playback()
        else:
            self.pause_whisper_playback()
 
    def start_whisper_playback(self):
        if self.whisper_audio_data is None:
            return
 
        self.whisper_playing = True
        self.whisper_stop_playback_flag = False
        self.whisper_play_btn.setText("⏸ Pause")
 
        def playback_worker():
            samplerate = 16000
            try:
                start_sample = int(self.whisper_current_position * samplerate)
                audio_chunk = self.whisper_audio_data[start_sample:]
                with sd.OutputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
                    chunk_size = samplerate // 20
                    for i in range(0, len(audio_chunk), chunk_size):
                        if self.whisper_stop_playback_flag:
                            break
                        chunk = audio_chunk[i:i + chunk_size]
                        if len(chunk) == 0:
                            break
                        if len(chunk) < chunk_size:
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                        stream.write(chunk.reshape(-1, 1))
                        self.whisper_current_position = (start_sample + i) / samplerate
                        self.whisper_position_label.setText(self._fmt_position(self.whisper_current_position))
                        self.whisper_waveform_display.set_playback_position(self.whisper_current_position)
                self.whisper_playing = False
                self.whisper_play_btn.setText("▶ Play")
            except Exception as e:
                print(f"Whisper preview playback error: {e}")
                traceback.print_exc()
                self.whisper_playing = False
                self.whisper_play_btn.setText("▶ Play")
 
        self.whisper_playback_thread = Thread(target=playback_worker, daemon=True)
        self.whisper_playback_thread.start()
 
    def pause_whisper_playback(self):
        self.whisper_stop_playback_flag = True
        self.whisper_playing = False
        self.whisper_play_btn.setText("▶ Play")
 
    def stop_whisper_playback(self):
        self.whisper_stop_playback_flag = True
        self.whisper_playing = False
        self.whisper_play_btn.setText("▶ Play")
 
    def reset_whisper_playback(self):
        self.stop_whisper_playback()
        self.whisper_current_position = 0
        self.whisper_position_label.setText("00:00:00")
        self.whisper_waveform_display.set_playback_position(0)

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
                                self.playback_position_label.setText(self._fmt_position(self.current_position))
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
                            self.playback_position_label.setText(self._fmt_position(self.current_position))
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
        self.playback_position_label.setText("00:00:00")
        self.waveform_display.set_playback_position(0)

    def clear_selections(self):
        if hasattr(self, 'waveform_display'):
            self.waveform_display.selections = []
            self.waveform_display.selections_changed.emit([])
            self.waveform_display.update()

        if hasattr(self, 'selection_info_label'):
            self.selection_info_label.setText("No regions selected — entire file will be processed")
            self.selection_info_label.setStyleSheet(
                "color: #888888; font-style: italic; padding: 6px 8px;"
                "background-color: #161616; border: 1px solid #2e2e2e; border-radius: 3px;"
            )

    def create_range_section(self):
        group = QGroupBox()
        group.setStyleSheet("""
            QGroupBox {
                background-color: #1c1c1c;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                margin-top: 0px;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(15, 12, 15, 12)

        info_label = QLabel(
            "<b style='color:#aaaaaa;'>Keyboard Shortcuts:</b><br>"
            "• <b style='color:#cccccc;'>CTRL + Left Mouse drag</b> = Select regions to "
            "<span style='color:#9b59b6;'><b>INCLUDE</b></span> (purple) — will be transcribed<br>"
            "• <b style='color:#cccccc;'>CTRL + Right Mouse drag</b> = Select regions to "
            "<span style='color:#e74c3c;'><b>EXCLUDE</b></span> (red) — will be muted/removed<br>"
            "• Opposite mouse button acts as <b style='color:#cccccc;'>eraser</b> for existing selections"
        )
        info_label.setStyleSheet("color: #888888; font-size: 12px; padding: 2px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.selection_info_label = QLabel("No regions selected — entire file will be processed")
        self.selection_info_label.setStyleSheet(
            "color: #888888; font-style: italic; padding: 6px 8px;"
            "background-color: #161616; border: 1px solid #2e2e2e; border-radius: 3px;"
        )
        layout.addWidget(self.selection_info_label)

        clear_btn = QPushButton("🗑  Clear All Selections")
        clear_btn.setFixedWidth(175)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a1a1a;
                color: #ff8888;
                font-weight: bold;
                padding: 6px 10px;
                border-radius: 4px;
                border: 1px solid #882222;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #aa2222; color: white; }
            QPushButton:disabled { background-color: #222; color: #555; border-color: #333; }
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
        default_output = str(self.input_file.with_suffix(".srt"))
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Subtitle File",
            default_output,
            "SRT Files (*.srt);;All Files (*.*)"
        )
        if not output_path:
            return
        self.output_file = Path(output_path)
        device = "cuda" if self.gpu_radio.isChecked() else "cpu"
        pedalboard_cfg = {
            "noise_gate_enabled": self.ng_check.isChecked(),
            "noise_gate_threshold": self.ng_threshold_spin.value(),
            "noise_gate_release": self.ng_release_spin.value(),
            "highpass_enabled": self.hp_check.isChecked(),
            "highpass_cutoff": self.hp_cutoff_spin.value(),
            "compressor_enabled": self.comp_check.isChecked(),
            "compressor_threshold": self.comp_threshold_spin.value(),
            "compressor_ratio": self.comp_ratio_spin.value(),
            "gain_enabled": self.gain_check.isChecked(),
            "gain_db": self.gain_db_spin.value(),
        }
        config = {
            "device": device,
            "voice_separation": self.voice_separation_check.isChecked(),
            "reuse_enabled": self.reuse_check.isChecked(),
            "reuse_model_dir": "./models/reuse",
            "bwe": self.bwe_combo.currentData(),
            "reuse_chunking_enabled": self.reuse_fixed_check.isChecked() or self.reuse_smart_check.isChecked(),
            "reuse_chunk_mode": "smart" if self.reuse_smart_check.isChecked() else "fixed",
            "reuse_chunk_seconds": self.reuse_smart_sec_spin.value() if self.reuse_smart_check.isChecked() else self.reuse_fixed_sec_spin.value(),
            "pedalboard": pedalboard_cfg,
            "word_pattern": pattern,
            "min_pause": self.min_pause_spin.value(),
            "min_duration": self.min_duration_spin.value(),
            "max_line_length": self.max_line_spin.value(),
            "batch_size": self.batch_spin.value(),
            "language": self.language_edit.text().strip()
        }
        model_size = self.model_combo.currentText()
        active_fx = []
        if pedalboard_cfg["noise_gate_enabled"]: active_fx.append("NoiseGate")
        if pedalboard_cfg["highpass_enabled"]: active_fx.append("Highpass")
        if pedalboard_cfg["compressor_enabled"]: active_fx.append("Compressor")
        if pedalboard_cfg["gain_enabled"]: active_fx.append("Gain")
        chunk_info = "disabled"
        if config["reuse_chunking_enabled"]:
            chunk_info = f"{config['reuse_chunk_mode']} / {config['reuse_chunk_seconds']:.1f}s"
        print("\n" + "="*60)
        print(" Subtitle Generator")
        print(" Mode: SUBTITLE GENERATION (.srt)")
        print("="*60)
        print(f"Device: {device.upper()}")
        print(f"Model: {model_size}")
        print(f"Voice Separation: {'ENABLED ✓' if config['voice_separation'] else 'disabled'}")
        print(f"RE-USE Enhancement: {'ENABLED ✓' if config['reuse_enabled'] else 'disabled'}")
        print(f"RE-USE Chunking: {chunk_info}")
        print(f"BWE: {config['bwe']} Hz")
        print(f"Pedalboard FX: {', '.join(active_fx) if active_fx else 'disabled'}")
        print(f"Input: {self.input_file}")
        print(f"Output: {self.output_file}")
        print("="*60 + "\n")
        self.processing = True
        self.process_btn.setEnabled(False)
        self.export_text_btn.setEnabled(False)
        self.check_audio_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        prebuilt = self._cached_processed_audio_path if self._is_audio_cache_applicable('srt') else None
        if prebuilt:
            print(f"✓ Reusing pre-built audio from Check Audio: {Path(prebuilt).name}")
        def worker():
            try:
                signals = WorkerSignals()
                signals.finished.connect(self.on_finished)
                signals.error.connect(self.on_error)
                signals.whisper_audio_ready.connect(self.load_whisper_preview_waveform)
                include_ranges, exclude_ranges = self.parse_time_ranges()
                self.generator = SubtitleGenerator(config)
                result = self.generator.process(
                    self.input_file,
                    self.output_file,
                    model_size,
                    include_ranges=include_ranges,
                    exclude_ranges=exclude_ranges,
                    output_format='srt',
                    audio_ready_callback=lambda path: signals.whisper_audio_ready.emit(path),
                    prebuilt_audio_path=prebuilt
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
        default_output = str(self.input_file.with_suffix(".txt"))
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Text Transcription",
            default_output,
            "Text Files (*.txt);;All Files (*.*)"
        )
        if not output_path:
            return
        text_output = Path(output_path)
        device = "cuda" if self.gpu_radio.isChecked() else "cpu"
        pedalboard_cfg = {
            "noise_gate_enabled": self.ng_check.isChecked(),
            "noise_gate_threshold": self.ng_threshold_spin.value(),
            "noise_gate_release": self.ng_release_spin.value(),
            "highpass_enabled": self.hp_check.isChecked(),
            "highpass_cutoff": self.hp_cutoff_spin.value(),
            "compressor_enabled": self.comp_check.isChecked(),
            "compressor_threshold": self.comp_threshold_spin.value(),
            "compressor_ratio": self.comp_ratio_spin.value(),
            "gain_enabled": self.gain_check.isChecked(),
            "gain_db": self.gain_db_spin.value(),
        }
        config = {
            "device": device,
            "voice_separation": self.voice_separation_check.isChecked(),
            "reuse_enabled": self.reuse_check.isChecked(),
            "reuse_model_dir": "./models/reuse",
            "bwe": self.bwe_combo.currentData(),
            "reuse_chunking_enabled": self.reuse_fixed_check.isChecked() or self.reuse_smart_check.isChecked(),
            "reuse_chunk_mode": "smart" if self.reuse_smart_check.isChecked() else "fixed",
            "reuse_chunk_seconds": self.reuse_smart_sec_spin.value() if self.reuse_smart_check.isChecked() else self.reuse_fixed_sec_spin.value(),
            "pedalboard": pedalboard_cfg,
            "word_pattern": [1],
            "min_pause": 0.6,
            "min_duration": 1.0,
            "max_line_length": 42,
            "batch_size": self.batch_spin.value(),
            "language": self.language_edit.text().strip()
        }
        model_size = self.model_combo.currentText()
        active_fx = []
        if pedalboard_cfg["noise_gate_enabled"]: active_fx.append("NoiseGate")
        if pedalboard_cfg["highpass_enabled"]: active_fx.append("Highpass")
        if pedalboard_cfg["compressor_enabled"]: active_fx.append("Compressor")
        if pedalboard_cfg["gain_enabled"]: active_fx.append("Gain")
        chunk_info = "disabled"
        if config["reuse_chunking_enabled"]:
            chunk_info = f"{config['reuse_chunk_mode']} / {config['reuse_chunk_seconds']:.1f}s"
        print("\n" + "="*60)
        print(" Subtitle Generator")
        print(" Mode: TEXT EXPORT (.txt)")
        print("="*60)
        print(f"Device: {device.upper()}")
        print(f"Model: {model_size}")
        print(f"Voice Separation: {'ENABLED ✓' if config['voice_separation'] else 'disabled'}")
        print(f"RE-USE Enhancement: {'ENABLED ✓' if config['reuse_enabled'] else 'disabled'}")
        print(f"RE-USE Chunking: {chunk_info}")
        print(f"BWE: {config['bwe']} Hz")
        print(f"Pedalboard FX: {', '.join(active_fx) if active_fx else 'disabled'}")
        print(f"Input: {self.input_file}")
        print(f"Output: {text_output}")
        print("="*60 + "\n")
        self.processing = True
        self.process_btn.setEnabled(False)
        self.export_text_btn.setEnabled(False)
        self.check_audio_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        prebuilt = self._cached_processed_audio_path if self._is_audio_cache_applicable('txt') else None
        if prebuilt:
            print(f"✓ Reusing pre-built audio from Check Audio: {Path(prebuilt).name}")
        def worker():
            try:
                signals = WorkerSignals()
                signals.text_exported.connect(self.on_text_exported)
                signals.error.connect(self.on_error)
                signals.whisper_audio_ready.connect(self.load_whisper_preview_waveform)
                include_ranges, exclude_ranges = self.parse_time_ranges()
                generator = SubtitleGenerator(config)
                generator.process(
                    self.input_file,
                    text_output,
                    model_size,
                    include_ranges=include_ranges,
                    exclude_ranges=exclude_ranges,
                    output_format='txt',
                    audio_ready_callback=lambda path: signals.whisper_audio_ready.emit(path),
                    prebuilt_audio_path=prebuilt
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
        self.progress_bar.setVisible(False)
        self._update_action_buttons_visibility()
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
        self.progress_bar.setVisible(False)
        self._update_action_buttons_visibility()
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
        self.progress_bar.setVisible(False)
        self._update_action_buttons_visibility()
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
            self.input_label.setStyleSheet(
                "color: #e0e0e0; background-color: #161616;"
                "border: 1px solid #2e2e2e; border-radius: 3px;"
                "padding: 5px 8px; font-size: 12px; font-style: normal;"
            )
            self.unload_btn.setVisible(True)
            self.button_container.setVisible(True)
            self._last_checked_audio_proc_state = None
            self._audio_proc_waveform_ready = False
            self._cached_processed_audio_path = None
            self._cached_audio_include_ranges = None
            self._cached_audio_exclude_ranges = None
            self._update_action_buttons_visibility()
            self.load_waveform(file_path)
            QTimer.singleShot(100, self._adjust_window_size)

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
