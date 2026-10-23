import ctypes
import datetime
import librosa
import numpy as np
import pysrt
import queue
import re
import sounddevice as sd
import soundfile as sf
import subprocess
import sys
import tempfile
import threading
import time
import traceback
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

from models_backends import (
    ICON_PATH,
    OUTPUTS_DIR,
    TaskCancelled,
    TranscriptionRequest,
    ensure_project_dirs,
    get_spec,
    load_backend_class,
    validate_segments,
)

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
            if (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
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
            if (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
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

            volume_norm = float(np.linalg.norm(indata))
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
    def __init__(self, config, backend, cancel_event=None, progress=None):
        self.config = config
        self.backend = backend
        self.cancel_event = cancel_event or threading.Event()
        self.progress = progress or (lambda fraction, message: None)

        self.temp_video_audio = None
        self.temp_processed_audio = None

        self.last_segments = None

    def log(self, message):
        print(message, flush=True)

    def cleanup_temp_files(self):
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

        for temp_file in (self.temp_video_audio, self.temp_processed_audio):
            if temp_file and temp_file.exists():
                try:
                    if remove_file_with_retry(temp_file):
                        self.log(f"  ✓ Removed: {temp_file.name}")
                        cleaned = True
                except Exception as e:
                    self.log(f"  ⚠️ Could not remove {temp_file.name}: {e}")

        if not cleaned:
            self.log("  (No temporary files to clean)")

    def create_subtitles_from_segments(self, segments):
        import re
        subs = pysrt.SubRipFile()
        sub_idx = 1
        current_speaker = None
        tts_mode = self.config.get("segmentation_mode", "standard") == "tts"
        word_pattern = self.config["word_pattern"]
        if tts_mode:
            max_words_cap = int(self.config.get("tts_max_words", 50))
        else:
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
            if not tts_mode:
                text = self.split_lines(text, max_line_length)
            item = pysrt.SubRipItem(
                index=sub_idx,
                start=seconds_to_srt_time(chunk[0]["start"]),
                end=seconds_to_srt_time(chunk[-1]["end"]),
                text=text
            )
            item.speaker = current_speaker
            subs.append(item)
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
            current_speaker = seg.get("speaker")
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
                if not tts_mode and SOFT_BREAK.search(raw) and len(chunk) >= 3:
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

        subs = self.merge_short_subs(subs)
        if tts_mode:
            long_count = sum(1 for s in subs if (s.end.ordinal - s.start.ordinal) > 10000)
            if long_count:
                self.log(f"  ⚠️ {long_count} subtitle(s) are longer than 10 s (expected in voice-over mode)")
        return self.apply_speaker_labels(subs)
        
    def apply_speaker_labels(self, subs):
        if not self.config.get("speaker_labels"):
            return subs
        previous = None
        for item in subs:
            speaker = getattr(item, "speaker", None)
            if speaker and speaker != previous:
                item.text = f"[{speaker}] {item.text}"
            previous = speaker
        return subs

    def merge_short_subs(self, subs):
        import re
        SENTENCE_END = re.compile(r'[.!?…]+$')

        if self.config.get("segmentation_mode", "standard") == "tts":
            max_words = int(self.config.get("tts_max_words", 50))
            max_gap_ms = max(self.config.get("min_pause", 0.6) * 2, 1.0) * 1000
            merged = []
            for item in subs:
                if merged:
                    prev = merged[-1]
                    prev_last = prev.text.split()[-1] if prev.text.strip() else ""
                    gap_ms = item.start.ordinal - prev.end.ordinal
                    same_speaker = getattr(prev, "speaker", None) == getattr(item, "speaker", None)
                    total_words = len(prev.text.split()) + len(item.text.split())
                    if (same_speaker and not SENTENCE_END.search(prev_last)
                            and gap_ms <= max_gap_ms and total_words < max_words):
                        prev.text += " " + item.text
                        prev.end = item.end
                        continue
                merged.append(item)
            result = pysrt.SubRipFile()
            for idx, s in enumerate(merged, 1):
                s.index = idx
                result.append(s)
            return result

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
                        if not prev_ends and getattr(prev, "speaker", None) == getattr(cur, "speaker", None):
                            prev.text += " " + cur.text
                            prev.end = cur.end
                            i += 1
                            changed = True
                            continue
                    if i + 1 < len(items) and not ends_sentence and \
                            getattr(items[i + 1], "speaker", None) == getattr(cur, "speaker", None):
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
                if nxt.get("speaker") != cur.get("speaker"):
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

        show_speakers = bool(self.config.get("speaker_labels"))
        paragraphs = []
        current_paragraph = []
        current_speaker = None

        def close_paragraph():
            nonlocal current_paragraph
            if current_paragraph:
                text = " ".join(current_paragraph)
                if show_speakers and current_speaker:
                    text = f"[{current_speaker}] {text}"
                paragraphs.append(text)
            current_paragraph = []

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            speaker = seg.get("speaker") if show_speakers else None
            if current_paragraph and speaker != current_speaker:
                close_paragraph()
            current_speaker = speaker
            current_paragraph.append(text)

            if i + 1 < len(segments):
                current_end = seg.get("end", 0)
                next_start = segments[i + 1].get("start", 0)
                if next_start - current_end > 2.0:
                    close_paragraph()

        close_paragraph()

        final_text = "\n\n".join(paragraphs)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        word_count = len(final_text.split())
        char_count = len(final_text)
        paragraph_count = len(paragraphs)

        self.log(f"  ✓ Exported {word_count} words ({char_count} characters)")
        self.log(f"  ✓ Formatted into {paragraph_count} paragraphs")
        self.log(f"  ✓ Saved to: {output_path}")

    def process(self, audio_path, output_path, include_ranges=None, exclude_ranges=None, output_format='srt'):
        try:
            return self._process(audio_path, output_path, include_ranges, exclude_ranges, output_format)
        except BaseException:
            self.cleanup_temp_files()
            raise

    def _process(self, audio_path, output_path, include_ranges, exclude_ranges, output_format):
        global processing_start_time
        processing_start_time = datetime.datetime.now()

        self.log(f"\n▶ Starting processing: {audio_path.name}")
        self.log(f"   Output format: {output_format.upper()}")

        temp_processed_file = None

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
                           (output_format == 'srt' and (has_include or has_exclude))

        if needs_processing:
            self.log(f"▶ Processing audio regions...")
            temp_processed_file = Path(tempfile.gettempdir()) / f"processed_{audio_path.stem}.wav"
            self.temp_processed_audio = temp_processed_file

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

        if self.cancel_event.is_set():
            raise TaskCancelled()

        request = TranscriptionRequest(
            audio_path=processed_audio_path,
            device=self.config["device"],
            settings=self.config["backend_settings"],
            progress=self.progress,
            log=self.log,
            cancel_event=self.cancel_event,
        )
        try:
            result = self.backend.transcribe(request)
        finally:
            self.backend.cleanup()
        segments = validate_segments(result.segments)

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

            if temp_processed_file and temp_processed_file.exists():
                temp_processed_file.unlink()
            self.cleanup_temp_files()

            return subs

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    text_exported = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(object, str)

class SubtitleGeneratorGUI(QMainWindow):
    audio_loaded_signal = pyqtSignal(np.ndarray)

    def __init__(self, backend_id, parent=None):
        super().__init__(parent)
        self.backend_spec = get_spec(backend_id)
        self.backend_cls = load_backend_class(backend_id)
        self.setting_widgets = {}
        self.device_buttons = {}
        self.cancel_event = threading.Event()
        self.signals = WorkerSignals()
        self.signals.finished.connect(self.on_finished)
        self.signals.text_exported.connect(self.on_text_exported)
        self.signals.error.connect(self.on_error)
        self.signals.cancelled.connect(self.on_cancelled)
        self.signals.progress.connect(self.on_progress)
        self.setWindowTitle(f"Subtitle Generator by Mubumbutu - {self.backend_spec.label}")
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

        self._updating_from_waveform = False

        self._io_user_opened = False

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
        self.cancel_btn = QPushButton("⛔ Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #B71C1C;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7f1010;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setVisible(False)
        button_layout.addWidget(self.cancel_btn)
        self.button_container.setVisible(False)
        layout.addWidget(self.button_container)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.status_label)
        self.adjustSize()

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
        caps = self.backend_cls.capabilities
        if caps.device_selection:
            device_layout = QHBoxLayout()
            device_layout.setSpacing(8)
            device_layout.addWidget(make_label("Device:"))
            self.device_group = QButtonGroup()
            for option in self.backend_cls.available_devices():
                radio = QRadioButton(option.label)
                radio.setStyleSheet(RADIO_STYLE)
                radio.setEnabled(option.enabled)
                if option.tooltip:
                    radio.setToolTip(option.tooltip)
                if option.default and option.enabled:
                    radio.setChecked(True)
                self.device_group.addButton(radio)
                self.device_buttons[option.id] = radio
                device_layout.addWidget(radio)
            device_layout.addStretch()
            model_layout.addLayout(device_layout)
        for spec in self.backend_cls.settings:
            if spec.section != "model":
                continue
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(make_label(spec.label))
            row.addWidget(self.create_setting_widget(spec, INPUT_STYLE, SPINBOX_STYLE, COMBO_STYLE))
            if spec.hint:
                row.addWidget(make_label(spec.hint))
            row.addStretch()
            model_layout.addLayout(row)
        model_layout.addStretch()
        self.model_settings_content.setLayout(model_layout)
        model_group_main_layout.addWidget(self.model_settings_content)

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
        if not caps.word_timestamps:
            self.min_pause_spin.setEnabled(False)
            label_min_pause.setEnabled(False)
            self.min_pause_spin.setToolTip("Not used: this model does not provide word timestamps")
        adv_row = 4
        for spec in self.backend_cls.settings:
            if spec.section != "advanced":
                continue
            adv_label = make_fixed_label(spec.label, 155)
            if spec.hint:
                adv_label.setToolTip(spec.hint)
            adv_grid.addWidget(adv_label, adv_row, 0)
            adv_grid.addWidget(self.create_setting_widget(spec, INPUT_STYLE, SPINBOX_STYLE, COMBO_STYLE), adv_row, 1)
            adv_row += 1
        self.speaker_labels_check = None
        if caps.speaker_labels:
            self.speaker_labels_check = QCheckBox("Prefix speaker labels")
            self.speaker_labels_check.setChecked(True)
            self.speaker_labels_check.setStyleSheet("color: #cccccc; font-size: 12px;")
            self.speaker_labels_check.setToolTip("Add [S01]-style labels to subtitles and text export when the speaker changes")
            adv_grid.addWidget(self.speaker_labels_check, adv_row, 0, 1, 2)
        self.segmentation_combo = QComboBox()
        self.segmentation_combo.setStyleSheet(COMBO_STYLE)
        self.segmentation_combo.setMinimumWidth(260)
        self.segmentation_combo.addItem("Subtitles for reading (standard)", "standard")
        self.segmentation_combo.addItem("Voice-over / TTS (full sentences)", "tts")
        segmentation_tip = (
            "Standard: short subtitles that are easy to read on screen.\n"
            "Voice-over / TTS: whole sentences with fewer, longer subtitles, "
            "better for text-to-speech and dubbing."
        )
        label_segmentation = make_fixed_label("Subtitle Type:", 155)
        label_segmentation.setToolTip(segmentation_tip)
        self.segmentation_combo.setToolTip(segmentation_tip)
        segmentation_row = QHBoxLayout()
        segmentation_row.setSpacing(8)
        segmentation_row.addWidget(label_segmentation)
        segmentation_row.addWidget(self.segmentation_combo)
        segmentation_row.addStretch()
        advanced_layout.addLayout(segmentation_row)

        def apply_segmentation_mode():
            reading_mode = self.segmentation_combo.currentData() != "tts"
            for control, label in (
                (self.pattern_edit, label_word_pattern),
                (self.min_duration_spin, label_min_duration),
                (self.max_line_spin, label_max_line),
            ):
                control.setVisible(reading_mode)
                label.setVisible(reading_mode)

        self.segmentation_combo.currentIndexChanged.connect(apply_segmentation_mode)
        advanced_layout.addLayout(adv_grid)
        advanced_layout.addStretch()
        self.advanced_settings_content.setLayout(advanced_layout)
        advanced_group_main_layout.addWidget(self.advanced_settings_content)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(10)
        settings_row.addWidget(model_group_container, 1)
        settings_row.addWidget(advanced_group_container, 1)
        layout.addLayout(settings_row)

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
                QTimer.singleShot(0, lambda msg=str(e): self.waveform_info.setText(f"Status: Error - {msg}"))

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
            self.load_waveform(self.input_file)
            QTimer.singleShot(100, self._adjust_window_size)

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
        print(f"▶ Loading waveform for: {self.input_file}")
        self.load_waveform(self.input_file)
        QTimer.singleShot(100, self._adjust_window_size)

    def unload_file(self):
        if self.playing:
            self.stop_playback()
        self.input_file = None
        self.audio_data = None
        self.current_position = 0
        self._io_user_opened = False
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

    def parse_pattern(self, pattern_str):
        try:
            pattern = [int(x.strip()) for x in pattern_str.split(",")]
            if not pattern or any(x <= 0 for x in pattern):
                raise ValueError
            return pattern
        except:
            return None

    def create_setting_widget(self, spec, input_style, spinbox_style, combo_style):
        if spec.kind == "choice":
            widget = QComboBox()
            widget.addItems(list(spec.choices))
            widget.setCurrentText(str(spec.default))
            widget.setStyleSheet(combo_style)
        elif spec.kind == "int":
            widget = QSpinBox()
            widget.setRange(int(spec.minimum if spec.minimum is not None else 0),
                            int(spec.maximum if spec.maximum is not None else 1000000))
            if spec.step:
                widget.setSingleStep(int(spec.step))
            if spec.suffix:
                widget.setSuffix(spec.suffix)
            widget.setValue(int(spec.default))
            widget.setMaximumWidth(110)
            widget.setStyleSheet(spinbox_style)
        elif spec.kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(spec.minimum if spec.minimum is not None else 0.0,
                            spec.maximum if spec.maximum is not None else 1000000.0)
            if spec.step:
                widget.setSingleStep(spec.step)
            if spec.suffix:
                widget.setSuffix(spec.suffix)
            widget.setValue(float(spec.default))
            widget.setMaximumWidth(110)
            widget.setStyleSheet(spinbox_style)
        elif spec.kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(spec.default))
        else:
            widget = QLineEdit(str(spec.default))
            if spec.placeholder:
                widget.setPlaceholderText(spec.placeholder)
            widget.setMinimumWidth(80)
            widget.setMaximumWidth(80 if spec.key == "language" else 360)
            widget.setStyleSheet(input_style)
        self.setting_widgets[spec.key] = (spec, widget)
        return widget

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_tooltips_applied", False):
            self._tooltips_applied = True
            self.apply_setting_tooltips()

    def apply_setting_tooltips(self):
        import html
        import textwrap

        if not hasattr(self, "pattern_edit"):
            return

        def wrap(text):
            return "<br>".join(html.escape(line) for line in textwrap.wrap(text, 58))

        labels = self.findChildren(QLabel)

        def labels_with_text(text):
            return [lbl for lbl in labels if lbl.text() == text]

        for spec, widget in self.setting_widgets.values():
            if not spec.hint:
                continue
            tip = wrap(spec.hint)
            widget.setToolTip(tip)
            for lbl in labels_with_text(spec.label):
                lbl.setToolTip(tip)
            for lbl in labels_with_text(spec.hint):
                lbl.setVisible(False)

        caps = self.backend_cls.capabilities

        if caps.word_timestamps:
            min_pause_tip = (
                "Starts a new subtitle when the silence between two words is longer "
                "than this value. Lower values give more, shorter subtitles."
            )
        else:
            min_pause_tip = "Not used with this model, because it does not provide word timestamps."

        pairs = (
            (
                self.pattern_edit,
                "Word Pattern:",
                "Limits how many words fit in one subtitle. The limit is the largest "
                "number times 4, so 3,4 allows up to 16 words. Subtitles often break "
                "earlier, at sentence ends, commas or pauses. Use comma-separated whole numbers.",
            ),
            (self.min_pause_spin, "Min Pause for Split:", min_pause_tip),
            (
                self.min_duration_spin,
                "Min Subtitle Duration:",
                "Subtitles shorter than this are merged with the previous or next "
                "subtitle from the same speaker, without joining two separate sentences. "
                "Raise it to avoid subtitles that flash by too quickly.",
            ),
            (
                self.max_line_spin,
                "Max Line Length:",
                "Maximum number of characters per subtitle line. A subtitle shows at "
                "most two lines. The usual film standard is 42.",
            ),
        )
        for widget, label_text, tip in pairs:
            wrapped = wrap(tip)
            widget.setToolTip(wrapped)
            for lbl in labels_with_text(label_text):
                lbl.setToolTip(wrapped)

        if self.speaker_labels_check is not None:
            self.speaker_labels_check.setToolTip(
                wrap(
                    "Adds a label such as [S01] at the start of a subtitle or text "
                    "paragraph whenever the speaker changes."
                )
            )

    def collect_backend_settings(self):
        values = {}
        for key, (spec, widget) in self.setting_widgets.items():
            if spec.kind == "choice":
                values[key] = widget.currentText()
            elif spec.kind in ("int", "float"):
                values[key] = widget.value()
            elif spec.kind == "bool":
                values[key] = widget.isChecked()
            else:
                values[key] = widget.text().strip()
        return values

    def selected_device(self):
        for device_id, radio in self.device_buttons.items():
            if radio.isChecked():
                return device_id
        return "cpu"

    def build_config(self, pattern):
        return {
            "device": self.selected_device(),
            "word_pattern": pattern,
            "min_pause": self.min_pause_spin.value(),
            "min_duration": self.min_duration_spin.value(),
            "max_line_length": self.max_line_spin.value(),
            "segmentation_mode": self.segmentation_combo.currentData(),
            "speaker_labels": bool(self.speaker_labels_check and self.speaker_labels_check.isChecked()),
            "backend_settings": self.collect_backend_settings(),
        }

    def start_processing(self):
        if not self.input_file or not self.input_file.exists():
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return
        pattern = self.parse_pattern(self.pattern_edit.text())
        if not pattern and self.segmentation_combo.currentData() == "tts":
            pattern = [3, 4]
        if not pattern:
            QMessageBox.warning(
                self,
                "Error",
                "Invalid word pattern. Use comma-separated positive integers (e.g., 3,4)"
            )
            return
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        default_output = str(OUTPUTS_DIR / self.input_file.with_suffix(".srt").name)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Subtitle File",
            default_output,
            "SRT Files (*.srt);;All Files (*.*)"
        )
        if not output_path:
            return
        self.output_file = Path(output_path)
        self.run_task("srt", pattern)

    def start_text_export(self):
        if not self.input_file or not self.input_file.exists():
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        default_output = str(OUTPUTS_DIR / self.input_file.with_suffix(".txt").name)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Text Transcription",
            default_output,
            "Text Files (*.txt);;All Files (*.*)"
        )
        if not output_path:
            return
        self.output_file = Path(output_path)
        self.run_task("txt", [1])

    def run_task(self, output_format, pattern):
        config = self.build_config(pattern)
        mode = "SUBTITLE GENERATION (.srt)" if output_format == "srt" else "TEXT EXPORT (.txt)"
        print("\n" + "="*60)
        print(" Subtitle Generator")
        print(f" Mode: {mode}")
        print("="*60)
        print(f"Backend: {self.backend_spec.label}")
        print(f"Device: {config['device'].upper()}")
        print(f"Input: {self.input_file}")
        print(f"Output: {self.output_file}")
        print("="*60 + "\n")
        self.set_processing_state(True)
        include_ranges, exclude_ranges = self.parse_time_ranges()
        self.cancel_event = threading.Event()
        generator = SubtitleGenerator(
            config,
            self.backend_cls(),
            self.cancel_event,
            progress=self.signals.progress.emit,
        )
        input_file = self.input_file
        output_file = self.output_file

        def worker():
            try:
                result = generator.process(
                    input_file,
                    output_file,
                    include_ranges=include_ranges,
                    exclude_ranges=exclude_ranges,
                    output_format=output_format
                )
                if output_format == "txt":
                    self.signals.text_exported.emit(str(output_file))
                else:
                    self.signals.finished.emit(result)
            except TaskCancelled:
                self.signals.cancelled.emit()
            except Exception as e:
                self.signals.error.emit(str(e) + "\n\n" + traceback.format_exc())
        thread = Thread(target=worker, daemon=True)
        thread.start()

    def set_processing_state(self, processing):
        self.processing = processing
        self.process_btn.setEnabled(not processing)
        self.export_text_btn.setEnabled(not processing)
        self.browse_btn.setEnabled(not processing)
        self.cancel_btn.setVisible(processing)
        self.cancel_btn.setEnabled(processing)
        self.progress_bar.setVisible(processing)
        self.status_label.setVisible(processing)
        if processing:
            self.progress_bar.setMaximum(0)
            self.status_label.setText("Starting...")
        QTimer.singleShot(40, self._adjust_window_size)

    def cancel_processing(self):
        self.cancel_event.set()
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancelling...")
        print("Cancellation requested")

    def on_progress(self, fraction, message):
        self.status_label.setText(message)
        if fraction is None:
            self.progress_bar.setMaximum(0)
        else:
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 100))

    def on_cancelled(self):
        self.set_processing_state(False)
        print("\nProcessing cancelled")
        QMessageBox.information(self, "Cancelled", "Processing was cancelled.")

    def on_finished(self, result):
        self.set_processing_state(False)
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
        self.set_processing_state(False)
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
        self.set_processing_state(False)
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
            self.load_waveform(file_path)
            QTimer.singleShot(100, self._adjust_window_size)


def run_gui(backend_id):
    ensure_project_dirs()
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "SubtitleGenerator.Mubumbutu"
        )
    except Exception:
        pass

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")

    icon = QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon()
    app.setWindowIcon(icon)

    window = SubtitleGeneratorGUI(backend_id)
    window.setWindowIcon(icon)
    window.show()

    return app.exec()
