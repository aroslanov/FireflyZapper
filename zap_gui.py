"""
FireflyZapper GUI - PySide6 based graphical interface for firefly removal from images.

Supports single image processing and image sequence processing with
frame scrubbing, navigation, and batch rendering.
"""

import sys
import os
import json
import re
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QSpinBox,
    QFileDialog, QGroupBox, QGridLayout, QSplitter, QTabWidget,
    QListWidget, QListWidgetItem, QLineEdit, QMessageBox,
    QScrollArea, QFrame, QSizePolicy, QComboBox, QCheckBox,
    QProgressBar, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QByteArray, QBuffer, QThread, QRectF, QObject
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QColor

# Import firefly processing from zap.py — now with GPU acceleration
from zap import process_channel, read_exr, write_exr, SUPPORTED_EXTENSIONS, process_image_gpu
# Import GPU backend for device detection, status display, and GPU-accelerated processing
from gpu_backend import get_device, get_device_status, is_gpu_active, reload_device_status, process_channel_gpu, process_image_gpu

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
WINDOW_SIZE_MIN = 3
WINDOW_SIZE_MAX = 31
THRESHOLD_MIN = 0.5
THRESHOLD_MAX = 20.0
WINDOW_SIZE_DEFAULT = 5
THRESHOLD_DEFAULT = 3.0
PRESETS_FILE = "presets.json"
PREVIEW_MAX_SIZE = 600


# ──────────────────────────────────────────────
# Image processing helpers
# ──────────────────────────────────────────────
def load_image(filepath):
    """Load an image from disk, normalizing to float32 [0,1] RGB.
    Returns (image, original_dtype, compression) where compression is
    an Imath.Compression for EXR files or None for other formats."""
    if filepath.lower().endswith('.exr'):
        image, compression = read_exr(filepath)
        original_dtype = np.float32
    else:
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        compression = None
        if image is None:
            raise ValueError(f"Failed to read image: {filepath}")
        if image.dtype == np.uint16:
            original_dtype = np.uint16
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.float32:
            original_dtype = np.float32
        elif image.dtype == np.uint8:
            original_dtype = np.uint8
            image = image.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported image type: {image.dtype}")
        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] >= 3:
            image = image[:, :, ::-1]

    return image, original_dtype, compression


def process_image_full(image, window_size, threshold, use_gpu=True):
    """
    Process an RGB image and return (result, mask).
    mask is a boolean array where True = firefly pixel detected.
    Supports multiplatform GPU acceleration (CUDA/OpenCL) when available,
    with automatic fallback to CPU for unsupported platforms.
    Designed for 4K images (3840×2160) where GPU acceleration provides
    significant speedup over CPU blur/median operations.
    """
    image_float = image.astype(np.float32)

    # Auto-detect GPU device — cached after first call
    device_label = get_device if use_gpu else "cpu"
    if use_gpu and is_gpu_active:
        # GPU path — dispatch to gpu_backend for channel processing
        if len(image_float.shape) == 3:
            channels = cv2.split(image_float)
            processed_channels = [process_channel_gpu(chan, window_size, threshold, device_label) for chan in channels]
            result = cv2.merge(processed_channels)
            # Compute mask on CPU from z-scores (simplification — in production, mask on GPU)
            mask_channels = []
            for chan in channels:
                mask_chan = process_channel_with_mask(chan, window_size, threshold, use_gpu=False)
                mask_channels.append(mask_chan)
            mask = np.any(mask_channels, axis=0)
        else:
            result = process_channel_gpu(image_float, window_size, threshold, device_label)
            mask = process_channel_with_mask(image_float, window_size, threshold, use_gpu=False)
    else:
        # CPU fallback — original NumPy/CV2 path (always works)
        if len(image_float.shape) == 3:
            channels = cv2.split(image_float)
            processed_channels = []
            mask_channels = []
            for chan in channels:
                result_chan, mask_chan = process_channel_with_mask(chan, window_size, threshold, use_gpu=False)
                processed_channels.append(result_chan)
                mask_channels.append(mask_chan)
            result = cv2.merge(processed_channels)
            mask = np.any(mask_channels, axis=0)
        else:
            result, mask = process_channel_with_mask(image_float, window_size, threshold, use_gpu=False)

    return result, mask


def process_channel_with_mask(channel, window_size, threshold, use_gpu=True):
    """
    Process a single channel and return (result, mask).
    Supports multiplatform GPU acceleration (CUDA/OpenCL) when available,
    with automatic fallback to CPU for unsupported platforms.
    """
    # Auto-detect GPU device (CUDA/OpenCL/CPU) — cached after first call
    device_label = get_device if use_gpu else "cpu"
    if use_gpu and is_gpu_active:
        # GPU path — dispatch to gpu_backend (returns result, but mask computed on CPU for now)
        # Note: GPU backend returns processed channel; mask is derived from z_scores
        # which is computed during GPU processing. For simplicity, we compute mask on CPU
        # from the GPU-processed result. In production, mask would be computed on GPU too.
        result = process_channel_gpu(channel, window_size, threshold, device_label)
        # Recompute mask on CPU from result (since GPU backend doesn't return mask yet)
        # This is a simplification — in production, mask would be returned from GPU
        channel_float = channel.astype(np.float32)
        ksize = (window_size, window_size)
        mean = cv2.blur(channel_float, ksize)
        squared = cv2.blur(channel_float ** 2, ksize)
        variance = squared - (mean ** 2)
        variance[variance < 0] = 0
        std = np.sqrt(variance)
        std[std == 0] = 1e-6
        z_scores = np.abs((channel_float - mean) / std)
        is_firefly = z_scores > threshold
        return result, is_firefly
    else:
        # CPU fallback — original NumPy/CV2 path (always works, returns mask)
        channel_float = channel.astype(np.float32)
        ksize = (window_size, window_size)
        mean = cv2.blur(channel_float, ksize)
        squared = cv2.blur(channel_float ** 2, ksize)
        variance = squared - (mean ** 2)
        variance[variance < 0] = 0
        std = np.sqrt(variance)
        std[std == 0] = 1e-6
        z_scores = np.abs((channel_float - mean) / std)
        is_firefly = z_scores > threshold
        half = window_size // 2
        padded = np.pad(channel_float, half, mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
        median_filtered = np.median(windows, axis=(-2, -1))
        result = np.where(is_firefly, median_filtered, channel_float)
        return result, is_firefly


def array_to_qpixmap(arr, exposure=10.0):
    """Convert a numpy array (float32, [0,1]) to QPixmap with exposure compensation.
    exposure is a brightness multiplier where 10 = 1x (normal), 0 = black.
    Uses percentile-based normalization so outlier pixels (e.g. fireflies)
    don't squash the visible range."""
    if arr.dtype != np.uint8:
        # Percentile-based normalization: use 1st and 99th percentiles
        # so a few extremely bright/dark pixels don't ruin the display
        p_low, p_high = np.percentile(arr, [1, 99])
        if p_high > p_low:
            arr = (arr - p_low) / (p_high - p_low)
        arr = np.clip(arr, 0, 1)
        # Apply exposure compensation for display
        if exposure != 10.0:
            arr = np.clip(arr * (exposure / 10.0), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    h, w = arr.shape[:2]

    if len(arr.shape) == 2:
        arr = np.dstack([arr, arr, arr])

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    arr = np.ascontiguousarray(arr)

    bytes_per_line = 3 * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def make_mask_overlay(mask, image_shape):
    """Create a red overlay visualization of the mask."""
    overlay = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
    overlay[mask] = [1.0, 0.0, 0.0]
    return overlay


def natural_sort_key(filename):
    """Sort filenames with numbers in natural order (frame_2 before frame_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]


def scan_sequence(directory):
    """Scan a directory for supported image files, sorted naturally."""
    files = []
    for f in os.listdir(directory):
        if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            files.append(f)
    files.sort(key=natural_sort_key)
    return [os.path.join(directory, f) for f in files]


# ──────────────────────────────────────────────
# Preset Manager
# ──────────────────────────────────────────────
class PresetManager:
    def __init__(self, filepath=PRESETS_FILE):
        self.filepath = filepath
        self.presets = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.presets = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.presets = {}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.presets, f, indent=2)

    def get(self, name):
        return self.presets.get(name, None)

    def set(self, name, window_size, threshold):
        self.presets[name] = {
            "window_size": window_size,
            "threshold": threshold
        }
        self.save()

    def delete(self, name):
        if name in self.presets:
            del self.presets[name]
            self.save()

    def list_names(self):
        return list(self.presets.keys())


# ──────────────────────────────────────────────
# Render Worker Thread
# ──────────────────────────────────────────────
class RenderWorker(QThread):
    """Processes a sequence of frames in a background thread.
    Uses multiplatform GPU acceleration (CUDA/OpenCL) when available,
    with automatic fallback to CPU for unsupported platforms."""
    progress = Signal(int, str)   # frame_index, filename
    frame_done = Signal(int, int) # frame_index, total_frames
    finished = Signal()
    error = Signal(str)

    def __init__(self, frame_paths, output_dir, window_size, threshold, original_dtype, use_gpu=True):
        super().__init__()
        self.frame_paths = frame_paths
        self.output_dir = output_dir
        self.window_size = window_size
        self.threshold = threshold
        self.original_dtype = original_dtype
        self.use_gpu = use_gpu
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.frame_paths)
        # Detect GPU device once at thread start — cached for all frames
        device_label = get_device if self.use_gpu else "cpu"
        gpu_active = is_gpu_active if self.use_gpu else False
        if gpu_active:
            # GPU acceleration active — process all frames on GPU
            for i, fpath in enumerate(self.frame_paths):
                if self._cancelled:
                    return

                basename = os.path.basename(fpath)
                self.progress.emit(i, basename)

                try:
                    image, _, compression = load_image(fpath)
                    # Use GPU acceleration for 4K+ images — significant speedup
                    result, _ = process_image_full(image, self.window_size, self.threshold, use_gpu=self.use_gpu)

                    # Convert back to original dtype
                    result_out = result.copy()
                    if self.original_dtype == np.uint16:
                        result_out = (np.clip(result_out, 0, 1) * 65535).astype(np.uint16)
                    elif self.original_dtype == np.uint8:
                        result_out = (np.clip(result_out, 0, 1) * 255).astype(np.uint8)

                    # Build output path
                    name, ext = os.path.splitext(basename)
                    out_name = f"{name}_processed{ext}"
                    out_path = os.path.join(self.output_dir, out_name)

                    if out_path.lower().endswith('.exr'):
                        write_exr(out_path, result_out, compression)
                    else:
                        save_img = result_out
                        if len(save_img.shape) == 3 and save_img.shape[2] >= 3:
                            save_img = save_img[:, :, ::-1]  # RGB -> BGR
                        cv2.imwrite(out_path, save_img)

                except Exception as e:
                    self.error.emit(f"Failed on {basename}: {str(e)}")
                    continue

                self.frame_done.emit(i + 1, total)
            else:
                # CPU fallback — process all frames on CPU (same as original)
                for i, fpath in enumerate(self.frame_paths):
                    if self._cancelled:
                        return

                    basename = os.path.basename(fpath)
                    self.progress.emit(i, basename)

                    try:
                        image, _, compression = load_image(fpath)
                        # CPU fallback — no GPU acceleration
                        result, _ = process_image_full(image, self.window_size, self.threshold, use_gpu=False)

                        # Convert back to original dtype
                        result_out = result.copy()
                        if self.original_dtype == np.uint16:
                            result_out = (np.clip(result_out, 0, 1) * 65535).astype(np.uint16)
                        elif self.original_dtype == np.uint8:
                            result_out = (np.clip(result_out, 0, 1) * 255).astype(np.uint8)

                        # Build output path
                        name, ext = os.path.splitext(basename)
                        out_name = f"{name}_processed{ext}"
                        out_path = os.path.join(self.output_dir, out_name)

                        if out_path.lower().endswith('.exr'):
                            write_exr(out_path, result_out, compression)
                        else:
                            save_img = result_out
                            if len(save_img.shape) == 3 and save_img.shape[2] >= 3:
                                save_img = save_img[:, :, ::-1]  # RGB -> BGR
                            cv2.imwrite(out_path, save_img)

                    except Exception as e:
                        self.error.emit(f"Failed on {basename}: {str(e)}")
                        continue

                    self.frame_done.emit(i + 1, total)

# ──────────────────────────────────────────────
# ZoomState — shared zoom/pan state for preview widgets
# ──────────────────────────────────────────────
class ZoomState(QObject):
    """Shared zoom/pan state synced between compare panels.
    offset_x/y are pixel offsets; zoom is a zoom factor (1.0 = 100%)."""

    # Signals
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.image_size: tuple[int, int] | None = None  # (iw, ih) tuple
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1.0

    def zoom_in(self, factor=1.25):
        self._apply_zoom(self.zoom * factor)

    def zoom_out(self, factor=1.25):
        self._apply_zoom(self.zoom / factor)

    def zoom_fit(self, view_w, view_h):
        """Set zoom so the image fits in the given view size."""
        if self.image_size is None:
            return
        iw, ih = self.image_size
        if iw == 0 or ih == 0:
            return
        self.zoom = min(view_w / iw, view_h / ih)
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.changed.emit()

    def zoom_100(self):
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.changed.emit()

    def pan_by(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self._clamp_offset()
        self.changed.emit()

    def _apply_zoom(self, new_zoom):
        new_zoom = max(0.1, min(new_zoom, 50.0))
        self.zoom = new_zoom
        self._clamp_offset()
        self.changed.emit()

    def _clamp_offset(self):
        if self.image_size is None:
            return
        iw, ih = self.image_size
        sw = iw * self.zoom
        sh = ih * self.zoom
        # Allow some over-pan (20% of the visible area)
        margin_w = sw * 0.2
        margin_h = sh * 0.2
        self.offset_x = max(-margin_w, min(self.offset_x, sw - margin_w))
        self.offset_y = max(-margin_h, min(self.offset_y, sh - margin_h))


class ZoomableImagePreview(QWidget):
    """A widget that displays an image with zoom/pan support via shared ZoomState."""

    def __init__(self, title="", zoom_state=None, parent=None):
        super().__init__(parent)
        self.title = title
        self._image_arr = None
        self._pixmap = None
        self._drag_start = None
        self._zoom_state = zoom_state
        self._exposure = 0.0

        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        if self._zoom_state:
            self._zoom_state.changed.connect(self.update)

    def set_image(self, arr, exposure=0.0):
        """Set the image from a numpy array."""
        self._image_arr = arr
        self._exposure = exposure
        pm = array_to_qpixmap(arr, exposure=exposure)
        self._pixmap = pm
        if self._zoom_state and arr is not None:
            h, w = arr.shape[:2]
            old_size = self._zoom_state.image_size
            # Only reset zoom if the image dimensions changed (new image load)
            if old_size is None or old_size != (w, h):
                self._zoom_state.set_image_size(w, h)
                self._zoom_state.zoom_fit(self.width(), self.height())
        self.update()

    def set_exposure(self, exposure):
        """Re-render the pixmap with a new exposure value (without reloading)."""
        if self._image_arr is None:
            return
        self._exposure = exposure
        self._pixmap = array_to_qpixmap(self._image_arr, exposure=exposure)
        self.update()

    def clear(self):
        self._image_arr = None
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        if self._pixmap is None or self._zoom_state is None:
            # Draw placeholder
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#1e1e1e"))
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.title}\n(no image)")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        zs = self._zoom_state
        iw, ih = zs.image_size
        sw = iw * zs.zoom
        sh = ih * zs.zoom

        # Center the image in the view, then apply offset
        cx = (self.width() - sw) / 2.0 + zs.offset_x
        cy = (self.height() - sh) / 2.0 + zs.offset_y

        target_rect = QRectF(cx, cy, sw, sh)
        painter.drawPixmap(target_rect, self._pixmap, self._pixmap.rect())

        # Draw border
        painter.setPen(QColor("#444"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom_state and self._pixmap is not None:
            self._zoom_state.zoom_fit(self.width(), self.height())

    def wheelEvent(self, event):
        if self._zoom_state is None or self._pixmap is None:
            return
        # Zoom toward cursor position
        old_zoom = self._zoom_state.zoom
        if event.angleDelta().y() > 0:
            self._zoom_state.zoom_in()
        else:
            self._zoom_state.zoom_out()
        new_zoom = self._zoom_state.zoom

        # Adjust offset so the point under the cursor stays fixed
        factor = new_zoom / old_zoom
        mx = event.position().x()
        my = event.position().y()
        iw, ih = self._zoom_state.image_size
        cx = (self.width() - iw * old_zoom) / 2.0 + self._zoom_state.offset_x
        cy = (self.height() - ih * old_zoom) / 2.0 + self._zoom_state.offset_y
        # Relative position of cursor within the image
        rx = (mx - cx) / (iw * old_zoom)
        ry = (my - cy) / (ih * old_zoom)
        # New center
        new_cx = mx - rx * iw * new_zoom
        new_cy = my - ry * ih * new_zoom
        self._zoom_state.offset_x = new_cx - (self.width() - iw * new_zoom) / 2.0
        self._zoom_state.offset_y = new_cy - (self.height() - ih * new_zoom) / 2.0
        self._zoom_state._clamp_offset()
        self._zoom_state.changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._drag_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and self._zoom_state:
            dx = event.position().x() - self._drag_start.x()
            dy = event.position().y() - self._drag_start.y()
            self._zoom_state.pan_by(dx, dy)
            self._drag_start = event.position()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)


class ImagePreview(QLabel):
    """A label that displays an image scaled to fit, preserving aspect ratio.
    Used for non-interactive previews (mask, original-only, processed-only tabs)."""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                color: #888;
                font-size: 13px;
            }
        """)
        self._pixmap = None
        self._image_arr = None
        self.setText(f"{title}\n(no image)")

    def set_image(self, arr):
        """Set the image from a numpy array."""
        self._image_arr = arr
        pm = array_to_qpixmap(arr)
        self._pixmap = pm
        self._update_display()

    def clear(self):
        self._image_arr = None
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(f"{self.title}\n(no image)")

    def _update_display(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


# ──────────────────────────────────────────────
# Main GUI Window
# ──────────────────────────────────────────────
class FireflyZapperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FireflyZapper")
        self.setMinimumSize(1300, 800)

        # ── Single image state ──
        self.image_path = None
        self.original_image = None
        self.original_dtype = None
        self.processed_image = None
        self.artifacts_mask = None
        self._exr_compression = None
        self._last_image_dir = ""
        self._last_sequence_dir = ""

        # ── Sequence state ──
        self.sequence_paths = []       # list of full file paths
        self.sequence_dir = None       # source directory
        self.current_frame_index = 0
        self.is_sequence_mode = False

        self.preset_manager = PresetManager()
        self.render_worker = None

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left panel: Controls ──
        left_panel = QWidget()
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        self._build_file_section(left_layout)
        self._build_sequence_section(left_layout)
        self._build_parameter_section(left_layout)
        self._build_preset_section(left_layout)
        self._build_action_section(left_layout)
        self._set_sequence_controls_enabled(False)
        left_layout.addStretch()

        # ── Right panel: Previews ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Preview tabs
        self.preview_tabs = QTabWidget()
        right_layout.addWidget(self.preview_tabs)

        # Tab 1: Side-by-side (zoomable, synced)
        self.tab_compare = QWidget()
        compare_vlayout = QVBoxLayout(self.tab_compare)
        compare_vlayout.setContentsMargins(0, 0, 0, 0)
        compare_vlayout.setSpacing(4)

        # Zoom toolbar
        zoom_toolbar = QHBoxLayout()
        zoom_toolbar.setContentsMargins(4, 4, 4, 0)
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setStyleSheet("color: #aaa; font-size: 12px;")
        zoom_toolbar.addWidget(self.zoom_label)
        zoom_toolbar.addStretch()
        btn_zoom_fit = QPushButton("Fit")
        btn_zoom_fit.setFixedHeight(24)
        btn_zoom_fit.clicked.connect(self._on_zoom_fit)
        zoom_toolbar.addWidget(btn_zoom_fit)
        btn_zoom_100 = QPushButton("1:1")
        btn_zoom_100.setFixedHeight(24)
        btn_zoom_100.clicked.connect(self._on_zoom_100)
        zoom_toolbar.addWidget(btn_zoom_100)
        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedSize(32, 24)
        btn_zoom_in.setStyleSheet("padding: 0px;")
        btn_zoom_in.clicked.connect(self._on_zoom_in)
        zoom_toolbar.addWidget(btn_zoom_in)
        btn_zoom_out = QPushButton("-")
        btn_zoom_out.setFixedSize(32, 24)
        btn_zoom_out.setStyleSheet("padding: 0px;")
        btn_zoom_out.clicked.connect(self._on_zoom_out)
        zoom_toolbar.addWidget(btn_zoom_out)
        compare_vlayout.addLayout(zoom_toolbar)

        # Shared zoom state
        self._zoom_state = ZoomState()
        self._zoom_state.changed.connect(self._on_zoom_changed)

        # Side-by-side previews
        compare_hlayout = QHBoxLayout()
        self.preview_original = ZoomableImagePreview("Original", zoom_state=self._zoom_state)
        self.preview_processed = ZoomableImagePreview("Processed", zoom_state=self._zoom_state)
        compare_hlayout.addWidget(self.preview_original)
        compare_hlayout.addWidget(self.preview_processed)
        compare_vlayout.addLayout(compare_hlayout)
        self.preview_tabs.addTab(self.tab_compare, "Compare")

        # Tab 2: Artifacts mask (zoomable, independent)
        self.tab_mask, self.preview_mask, _ = self._build_zoomable_tab("Artifacts Mask")
        self.preview_tabs.addTab(self.tab_mask, "Artifacts Mask")

        # Tab 3: Original only (zoomable, independent)
        self.tab_original, self.preview_original_only, _ = self._build_zoomable_tab("Original")
        self.preview_tabs.addTab(self.tab_original, "Original")

        # Tab 4: Processed only (zoomable, independent)
        self.tab_processed_only, self.preview_processed_only, _ = self._build_zoomable_tab("Processed")
        self.preview_tabs.addTab(self.tab_processed_only, "Processed")

        # Status bar — now with GPU acceleration info
        self.status_label = QLabel("Ready — open an image or sequence folder to begin")
        self.status_label.setStyleSheet("color: #aaa; padding: 2px;")
        right_layout.addWidget(self.status_label)

        # GPU status bar — shows detected GPU device and acceleration status
        self.gpu_status_label = QLabel("GPU: Detecting...")
        self.gpu_status_label.setStyleSheet("color: #888; padding: 2px; font-weight: bold;")
        right_layout.addWidget(self.gpu_status_label)
        # Initialize GPU status text from backend
        gpu_status = get_device_status()
        self.gpu_status_label.setText(f"{gpu_status}")
        if is_gpu_active:
            self.gpu_status_label.setText("GPU acceleration: Active")
        else:
            self.gpu_status_label.setText("GPU acceleration: Disabled (CPU fallback)")
        # Hot reload when user plugs GPU mid-session — slot will be wired later
        self._gpu_hot_reload_slot = reload_device_status()  # function reference for hot reload

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self._apply_theme()

    # ── UI Builders ──

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 14px;
                font-weight: bold;
                color: #ccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 14px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QPushButton#btn_process {
                background-color: #2a6d2a;
                border-color: #3a8a3a;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton#btn_process:hover {
                background-color: #3a8a3a;
            }
            QPushButton#btn_save {
                background-color: #2a5a7a;
                border-color: #3a7a9a;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton#btn_save:hover {
                background-color: #3a7a9a;
            }
            QPushButton#btn_render {
                background-color: #7a5a2a;
                border-color: #9a7a3a;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton#btn_render:hover {
                background-color: #9a7a3a;
            }
            QPushButton#btn_render:disabled {
                background-color: #444;
                border-color: #555;
                color: #777;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #888;
                border: 1px solid #aaa;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #aaa;
            }
            QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px 5px;
                color: #e0e0e0;
            }
            QListWidget {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #2a5a7a;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-bottom: none;
                padding: 6px 14px;
                margin-right: 2px;
                border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                border-bottom: 1px solid #2d2d2d;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #333;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #2a6d2a;
                border-radius: 3px;
            }
        """)

    def _build_zoomable_tab(self, title):
        """Build a tab with a zoom toolbar and a single ZoomableImagePreview.
        Returns (tab_widget, preview_widget, zoom_state)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Zoom toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 0)
        zoom_label = QLabel("Zoom: 100%")
        zoom_label.setStyleSheet("color: #aaa; font-size: 12px;")
        toolbar.addWidget(zoom_label)
        toolbar.addStretch()
        btn_fit = QPushButton("Fit")
        btn_fit.setFixedHeight(24)
        toolbar.addWidget(btn_fit)
        btn_100 = QPushButton("1:1")
        btn_100.setFixedHeight(24)
        toolbar.addWidget(btn_100)
        btn_in = QPushButton("+")
        btn_in.setFixedSize(32, 24)
        btn_in.setStyleSheet("padding: 0px;")
        toolbar.addWidget(btn_in)
        btn_out = QPushButton("-")
        btn_out.setFixedSize(32, 24)
        btn_out.setStyleSheet("padding: 0px;")
        toolbar.addWidget(btn_out)
        layout.addLayout(toolbar)

        zs = ZoomState()
        zs.changed.connect(lambda: zoom_label.setText(f"Zoom: {zs.zoom * 100:.0f}%"))
        def _fit():
            zs.zoom_fit(preview.width(), preview.height())
        btn_fit.clicked.connect(_fit)
        btn_100.clicked.connect(zs.zoom_100)
        btn_in.clicked.connect(zs.zoom_in)
        btn_out.clicked.connect(zs.zoom_out)

        preview = ZoomableImagePreview(title, zoom_state=zs)
        layout.addWidget(preview)
        return tab, preview, zs

    def _build_file_section(self, layout):
        group = QGroupBox("Input")
        gl = QVBoxLayout(group)
        gl.setSpacing(6)

        hrow = QHBoxLayout()
        btn_open_image = QPushButton("📷 Open Image")
        btn_open_image.clicked.connect(self._on_open_image)
        hrow.addWidget(btn_open_image)

        btn_open_seq = QPushButton("🎞️ Open Sequence")
        btn_open_seq.clicked.connect(self._on_open_sequence)
        hrow.addWidget(btn_open_seq)
        gl.addLayout(hrow)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #999; font-size: 12px;")
        gl.addWidget(self.file_label)

        layout.addWidget(group)

    def _build_sequence_section(self, layout):
        group = QGroupBox("Sequence Controls")
        self.seq_group = group
        gl = QVBoxLayout(group)
        gl.setSpacing(4)

        # Frame counter
        self.frame_label = QLabel("Frame: — / —")
        self.frame_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #ddd;")
        gl.addWidget(self.frame_label)

        # Scrubber slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        gl.addWidget(self.frame_slider)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self.btn_first = QPushButton("⏮ First")
        self.btn_first.clicked.connect(lambda: self._go_to_frame(0))
        nav_row.addWidget(self.btn_first)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self._on_prev_frame)
        nav_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._on_next_frame)
        nav_row.addWidget(self.btn_next)

        self.btn_last = QPushButton("Last ⏭")
        self.btn_last.clicked.connect(lambda: self._go_to_frame(-1))
        nav_row.addWidget(self.btn_last)
        gl.addLayout(nav_row)

        # Output dir for render
        render_dir_row = QHBoxLayout()
        render_dir_row.addWidget(QLabel("Output:"))
        self.render_dir_edit = QLineEdit()
        self.render_dir_edit.setPlaceholderText("Same as source (append _processed)")
        render_dir_row.addWidget(self.render_dir_edit)
        btn_browse_render = QPushButton("Browse")
        btn_browse_render.clicked.connect(self._on_browse_render_dir)
        render_dir_row.addWidget(btn_browse_render)
        gl.addLayout(render_dir_row)

        # Render progress
        self.render_progress = QProgressBar()
        self.render_progress.setVisible(False)
        self.render_progress.setValue(0)
        gl.addWidget(self.render_progress)

        layout.addWidget(group)

    def _build_parameter_section(self, layout):
        group = QGroupBox("Parameters")
        gl = QGridLayout(group)
        gl.setSpacing(8)

        # Window size
        gl.addWidget(QLabel("Window Size:"), 0, 0)
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX)
        self.window_size_spin.setValue(WINDOW_SIZE_DEFAULT)
        self.window_size_spin.setSingleStep(2)
        self.window_size_spin.valueChanged.connect(self._on_param_changed)
        gl.addWidget(self.window_size_spin, 0, 1)

        self.window_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_size_slider.setRange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX)
        self.window_size_slider.setValue(WINDOW_SIZE_DEFAULT)
        self.window_size_slider.setSingleStep(2)
        self.window_size_slider.valueChanged.connect(self.window_size_spin.setValue)
        self.window_size_spin.valueChanged.connect(self.window_size_slider.setValue)
        gl.addWidget(self.window_size_slider, 0, 2)

        # Threshold
        gl.addWidget(QLabel("Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(THRESHOLD_MIN, THRESHOLD_MAX)
        self.threshold_spin.setValue(THRESHOLD_DEFAULT)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.valueChanged.connect(self._on_param_changed)
        gl.addWidget(self.threshold_spin, 1, 1)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(int(THRESHOLD_MIN * 10), int(THRESHOLD_MAX * 10))
        self.threshold_slider.setValue(int(THRESHOLD_DEFAULT * 10))
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_spin.setValue(v / 10.0)
        )
        self.threshold_spin.valueChanged.connect(
            lambda v: self.threshold_slider.setValue(int(v * 10))
        )
        gl.addWidget(self.threshold_slider, 1, 2)

        # Exposure (display only)
        gl.addWidget(QLabel("Exposure (EV):"), 2, 0)
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.0, 100.0)
        self.exposure_spin.setValue(1.0)
        self.exposure_spin.setSingleStep(1.0)
        self.exposure_spin.valueChanged.connect(self._on_exposure_changed)
        gl.addWidget(self.exposure_spin, 2, 1)

        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(0, 1000)
        self.exposure_slider.setValue(100)
        self.exposure_slider.valueChanged.connect(
            lambda v: self.exposure_spin.setValue(v / 10.0)
        )
        self.exposure_spin.valueChanged.connect(
            lambda v: self.exposure_slider.setValue(int(v * 10))
        )
        gl.addWidget(self.exposure_slider, 2, 2)

        layout.addWidget(group)

    def _build_preset_section(self, layout):
        group = QGroupBox("Presets")
        gl = QVBoxLayout(group)
        gl.setSpacing(4)

        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(120)
        self.preset_list.itemClicked.connect(self._on_preset_selected)
        gl.addWidget(self.preset_list)

        hbox = QHBoxLayout()
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Preset name...")
        hbox.addWidget(self.preset_name_edit)

        btn_save_preset = QPushButton("Save")
        btn_save_preset.clicked.connect(self._on_save_preset)
        hbox.addWidget(btn_save_preset)

        btn_delete_preset = QPushButton("Del")
        btn_delete_preset.clicked.connect(self._on_delete_preset)
        hbox.addWidget(btn_delete_preset)
        gl.addLayout(hbox)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Quick:"))
        self.quick_preset_combo = QComboBox()
        self.quick_preset_combo.addItems([
            "Select preset...",
            "Mild (ws=3, th=4.0)",
            "Medium (ws=5, th=3.0)",
            "Aggressive (ws=9, th=2.0)",
            "Very Aggressive (ws=15, th=1.5)",
        ])
        self.quick_preset_combo.currentIndexChanged.connect(self._on_quick_preset)
        hbox2.addWidget(self.quick_preset_combo)
        gl.addLayout(hbox2)

        layout.addWidget(group)
        self._refresh_preset_list()

    def _build_action_section(self, layout):
        group = QGroupBox("Actions")
        gl = QVBoxLayout(group)
        gl.setSpacing(6)

        self.auto_preview_cb = QCheckBox("Auto-preview on parameter change")
        self.auto_preview_cb.setChecked(True)
        gl.addWidget(self.auto_preview_cb)

        self.btn_process = QPushButton("⚡ Process Current Frame")
        self.btn_process.setObjectName("btn_process")
        self.btn_process.clicked.connect(self._on_process)
        gl.addWidget(self.btn_process)

        self.btn_save = QPushButton("💾 Save Current Frame As...")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)
        gl.addWidget(self.btn_save)

        self.btn_render = QPushButton("🎬 Render Sequence")
        self.btn_render.setObjectName("btn_render")
        self.btn_render.clicked.connect(self._on_render_sequence)
        self.btn_render.setEnabled(False)
        gl.addWidget(self.btn_render)

        layout.addWidget(group)

    # ── Sequence helpers ──

    def _set_sequence_controls_enabled(self, enabled):
        self.frame_slider.setEnabled(enabled)
        self.btn_first.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_last.setEnabled(enabled)
        self.btn_render.setEnabled(enabled)

    def _update_frame_display(self):
        if not self.is_sequence_mode or not self.sequence_paths:
            self.frame_label.setText("Frame: — / —")
            return
        total = len(self.sequence_paths)
        current = self.current_frame_index + 1  # 1-based for display
        basename = os.path.basename(self.sequence_paths[self.current_frame_index])
        self.frame_label.setText(f"Frame: {current} / {total}  —  {basename}")

    def _load_current_frame(self):
        """Load the current frame from the sequence into the preview."""
        if not self.is_sequence_mode or not self.sequence_paths:
            return
        fpath = self.sequence_paths[self.current_frame_index]
        try:
            self.image_path = fpath
            self.original_image, self.original_dtype, self._exr_compression = load_image(fpath)
            self.file_label.setText(f"[Sequence] {os.path.basename(os.path.dirname(fpath))}")

            ev = self.exposure_spin.value()
            self.preview_original.set_image(self.original_image, ev)
            self.preview_original_only.set_image(self.original_image, ev)

            self.processed_image = None
            self.artifacts_mask = None
            self.preview_processed.clear()
            self.preview_processed_only.clear()
            self.preview_mask.clear()
            self.btn_save.setEnabled(False)

            self._update_frame_display()

            if self.auto_preview_cb.isChecked():
                self._on_process()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load frame:\n{str(e)}")

    def _go_to_frame(self, index):
        if not self.sequence_paths:
            return
        if index < 0:
            index = len(self.sequence_paths) - 1
        index = max(0, min(index, len(self.sequence_paths) - 1))
        if index != self.current_frame_index:
            self.current_frame_index = index
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(index)
            self.frame_slider.blockSignals(False)
            self._load_current_frame()

    # ── Slots ──

    def _on_open_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self._last_image_dir,
            "Images (*.exr *.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if not filepath:
            return

        self._last_image_dir = os.path.dirname(filepath)

        # Switch to single image mode
        self.is_sequence_mode = False
        self.sequence_paths = []
        self._set_sequence_controls_enabled(False)
        self.render_progress.setVisible(False)

        try:
            self.image_path = filepath
            self.original_image, self.original_dtype, self._exr_compression = load_image(filepath)
            self.file_label.setText(os.path.basename(filepath))
            self.status_label.setText(
                f"Loaded: {os.path.basename(filepath)} ({self.original_image.shape[1]}×{self.original_image.shape[0]})"
            )

            ev = self.exposure_spin.value()
            self.preview_original.set_image(self.original_image, ev)
            self.preview_original_only.set_image(self.original_image, ev)

            self.processed_image = None
            self.artifacts_mask = None
            self.preview_processed.clear()
            self.preview_processed_only.clear()
            self.preview_mask.clear()
            self.btn_save.setEnabled(False)

            self._update_frame_display()

            if self.auto_preview_cb.isChecked():
                self._on_process()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def _on_open_sequence(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Open Image Sequence Folder", self._last_sequence_dir
        )
        if not directory:
            return

        self._last_sequence_dir = directory

        files = scan_sequence(directory)
        if not files:
            QMessageBox.warning(self, "Warning", f"No supported images found in:\n{directory}")
            return

        self.is_sequence_mode = True
        self.sequence_paths = files
        self.sequence_dir = directory
        self.current_frame_index = 0

        # Enable sequence controls
        self._set_sequence_controls_enabled(True)
        self.frame_slider.setMaximum(len(files) - 1)
        self.frame_slider.setValue(0)
        self.render_progress.setVisible(False)
        self.render_progress.setValue(0)

        self.file_label.setText(f"[Sequence] {os.path.basename(directory)} — {len(files)} frames")
        self.status_label.setText(f"Loaded sequence: {len(files)} frames from {os.path.basename(directory)}")

        # Set default render output dir
        self.render_dir_edit.setText("")

        self._load_current_frame()

    def _on_frame_slider_changed(self, value):
        if not self.sequence_paths:
            return
        if value != self.current_frame_index:
            self.current_frame_index = value
            self._load_current_frame()

    def _on_prev_frame(self):
        self._go_to_frame(self.current_frame_index - 1)

    def _on_next_frame(self):
        self._go_to_frame(self.current_frame_index + 1)

    def _on_browse_render_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Render Output Directory", ""
        )
        if directory:
            self.render_dir_edit.setText(directory)

    def _on_param_changed(self):
        if self.original_image is not None and self.auto_preview_cb.isChecked():
            try:
                self._debounce_timer
            except AttributeError:
                self._debounce_timer = QTimer()
                self._debounce_timer.setSingleShot(True)
                self._debounce_timer.timeout.connect(self._on_process)
            self._debounce_timer.start(300)

    def _on_exposure_changed(self):
        """Refresh all previews with the new exposure value."""
        ev = self.exposure_spin.value()
        for preview in [self.preview_original, self.preview_processed,
                        self.preview_original_only, self.preview_processed_only,
                        self.preview_mask]:
            preview.set_exposure(ev)

    def _on_process(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please open an image or sequence first.")
            return

        ws = self.window_size_spin.value()
        th = self.threshold_spin.value()

        # Check GPU acceleration status before processing
        gpu_status = get_device_status()
        self.gpu_status_label.setText(f"{gpu_status}")
        self.status_label.setText("Processing...")
        QApplication.processEvents()

        try:
            # Use GPU acceleration if available — pass use_gpu=True
            result, mask = process_image_full(self.original_image, ws, th, use_gpu=True)
            self.processed_image = result
            self.artifacts_mask = mask

            ev = self.exposure_spin.value()
            self.preview_processed.set_image(result, ev)
            self.preview_processed_only.set_image(result, ev)

            blended = self.original_image.copy()
            blended[mask] = blended[mask] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6
            self.preview_mask.set_image(blended, ev)

            num_fireflies = int(np.sum(mask))
            total_pixels = np.size(mask)  # total elements in numpy array
            pct = 100.0 * num_fireflies / total_pixels
            # Format GPU status info — gpu_status is a string from get_device_status()
            # Extract device name from status string (format: "Device: Name (Type)")
            gpu_name = gpu_status.split(":")[1] if ":" in gpu_status else "unknown"
            gpu_active = "Active" if is_gpu_active else "Disabled"
            self.status_label.setText(
                f"Done — {num_fireflies} firefly pixels ({pct:.3f}%) detected | "
                f"ws={ws}, th={th:.1f} | GPU: {gpu_name} acceleration {gpu_active}"
            )
            self.btn_save.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.gpu_status_label.setText("GPU: Error — acceleration disabled")

    def _on_save(self):
        if self.processed_image is None:
            return

        if self.image_path and self.image_path.lower().endswith('.exr'):
            default_ext = ".exr"
            filter_str = "EXR (*.exr);;PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        else:
            default_ext = ".png"
            filter_str = "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;EXR (*.exr)"

        assert self.image_path is not None
        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_processed" + default_ext

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", default_name, filter_str
        )
        if not filepath:
            return

        try:
            result_out = self.processed_image.copy()
            if self.original_dtype == np.uint16:
                result_out = (np.clip(result_out, 0, 1) * 65535).astype(np.uint16)
            elif self.original_dtype == np.uint8:
                result_out = (np.clip(result_out, 0, 1) * 255).astype(np.uint8)

            if filepath.lower().endswith('.exr'):
                compression = getattr(self, '_exr_compression', None)
                write_exr(filepath, result_out, compression)
            else:
                if len(result_out.shape) == 3 and result_out.shape[2] >= 3:
                    result_out = result_out[:, :, ::-1]
                cv2.imwrite(filepath, result_out)

            self.status_label.setText(f"Saved: {os.path.basename(filepath)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")

    def _on_render_sequence(self):
        if not self.is_sequence_mode or not self.sequence_paths:
            QMessageBox.warning(self, "Warning", "No sequence loaded.")
            return

        ws = self.window_size_spin.value()
        th = self.threshold_spin.value()

        # Determine output directory
        output_dir = self.render_dir_edit.text().strip()
        if not output_dir:
            # Default: create a subfolder next to source
            assert self.sequence_dir is not None
            output_dir = os.path.join(self.sequence_dir, "..", os.path.basename(self.sequence_dir) + "_processed")
            output_dir = os.path.abspath(output_dir)

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create output directory:\n{str(e)}")
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Render Sequence",
            f"Render {len(self.sequence_paths)} frames?\n\n"
            f"Parameters: ws={ws}, th={th}\n"
            f"Output: {output_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Disable controls during render
        self._set_sequence_controls_enabled(False)
        self.btn_process.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.render_progress.setVisible(True)
        self.render_progress.setMaximum(len(self.sequence_paths))
        self.render_progress.setValue(0)
        self.status_label.setText("Rendering sequence...")

        # Check GPU acceleration status before starting render
        gpu_status = get_device_status()
        self.gpu_status_label.setText(f"{gpu_status}")
        gpu_active = is_gpu_active
        if gpu_active:
            # GPU acceleration active — pass use_gpu=True to RenderWorker
            self.render_worker = RenderWorker(
                self.sequence_paths, output_dir, ws, th, self.original_dtype, use_gpu=True
            )
        else:
            # CPU fallback — pass use_gpu=False to RenderWorker
            self.render_worker = RenderWorker(
                self.sequence_paths, output_dir, ws, th, self.original_dtype, use_gpu=False
            )
        self.render_worker.progress.connect(self._on_render_progress)
        self.render_worker.frame_done.connect(self._on_render_frame_done)
        self.render_worker.finished.connect(self._on_render_finished)
        self.render_worker.error.connect(self._on_render_error)
        self.render_worker.start()

    def _on_render_progress(self, frame_index, filename):
        self.status_label.setText(f"Rendering: {filename} ({frame_index + 1}/{len(self.sequence_paths)})")

    def _on_render_frame_done(self, done, total):
        self.render_progress.setValue(done)

    def _on_render_finished(self):
        self.render_progress.setValue(len(self.sequence_paths))
        self.status_label.setText(f"Render complete — {len(self.sequence_paths)} frames processed")

        # Re-enable controls
        self._set_sequence_controls_enabled(True)
        self.btn_process.setEnabled(True)
        self.btn_render.setEnabled(True)

        QMessageBox.information(
            self, "Render Complete",
            f"Successfully processed {len(self.sequence_paths)} frames."
        )

    def _on_render_error(self, msg):
        self.status_label.setText(f"Render error: {msg}")

    def _on_preset_selected(self, item):
        name = item.text()
        preset = self.preset_manager.get(name)
        if preset:
            self.window_size_spin.setValue(preset["window_size"])
            self.threshold_spin.setValue(preset["threshold"])
            self.status_label.setText(f"Loaded preset: {name}")

    def _on_save_preset(self):
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a preset name.")
            return
        ws = self.window_size_spin.value()
        th = self.threshold_spin.value()
        self.preset_manager.set(name, ws, th)
        self._refresh_preset_list()
        self.preset_name_edit.clear()
        self.status_label.setText(f"Saved preset: {name}")

    def _on_delete_preset(self):
        item = self.preset_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "Warning", "Select a preset to delete.")
            return
        name = item.text()
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.preset_manager.delete(name)
            self._refresh_preset_list()
            self.status_label.setText(f"Deleted preset: {name}")

    def _on_quick_preset(self, index):
        if index <= 0:
            return
        presets = {
            1: (3, 4.0),
            2: (5, 3.0),
            3: (9, 2.0),
            4: (15, 1.5),
        }
        ws, th = presets.get(index, (5, 3.0))
        self.window_size_spin.setValue(ws)
        self.threshold_spin.setValue(th)
        self.quick_preset_combo.setCurrentIndex(0)
        self.status_label.setText(f"Quick preset: ws={ws}, th={th}")

    def _refresh_preset_list(self):
        self.preset_list.clear()
        for name in self.preset_manager.list_names():
            self.preset_list.addItem(name)

    # ── Zoom handlers ──

    def _on_zoom_changed(self):
        zs = self._zoom_state
        self.zoom_label.setText(f"Zoom: {zs.zoom * 100:.0f}%")

    def _on_zoom_fit(self):
        if self._zoom_state and self.preview_original._pixmap is not None:
            self._zoom_state.zoom_fit(self.preview_original.width(), self.preview_original.height())

    def _on_zoom_100(self):
        if self._zoom_state:
            self._zoom_state.zoom_100()

    def _on_zoom_in(self):
        if self._zoom_state:
            self._zoom_state.zoom_in()

    def _on_zoom_out(self):
        if self._zoom_state:
            self._zoom_state.zoom_out()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FireflyZapper")
    window = FireflyZapperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
