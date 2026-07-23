"""
FireflyZapper GUI - PySide6 based graphical interface for firefly removal from images.

Provides visual preview of original, processed image and artifacts mask,
with customizable parameter presets.
"""

import sys
import os
import json
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QSpinBox,
    QFileDialog, QGroupBox, QGridLayout, QSplitter, QTabWidget,
    QListWidget, QListWidgetItem, QLineEdit, QMessageBox,
    QScrollArea, QFrame, QSizePolicy, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QByteArray, QBuffer
from PySide6.QtGui import QPixmap, QImage, QFont

# Import firefly processing from zap.py
from zap import process_channel, read_exr, write_exr, SUPPORTED_EXTENSIONS

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
    """Load an image from disk, normalizing to float32 [0,1] RGB."""
    if filepath.lower().endswith('.exr'):
        image = read_exr(filepath)
        original_dtype = np.float32
    else:
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
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

    return image, original_dtype


def process_image_full(image, window_size, threshold):
    """
    Process an RGB image and return (result, mask).
    mask is a boolean array where True = firefly pixel detected.
    """
    image_float = image.astype(np.float32)

    if len(image_float.shape) == 3:
        channels = cv2.split(image_float)
        processed_channels = []
        mask_channels = []
        for chan in channels:
            result_chan, mask_chan = process_channel_with_mask(chan, window_size, threshold)
            processed_channels.append(result_chan)
            mask_channels.append(mask_chan)
        result = cv2.merge(processed_channels)
        # Combine masks: a pixel is a firefly if ANY channel flagged it
        mask = np.any(mask_channels, axis=0)
    else:
        result, mask = process_channel_with_mask(image_float, window_size, threshold)

    return result, mask


def process_channel_with_mask(channel, window_size, threshold):
    """
    Process a single channel and return (result, mask).
    """
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

    # medianBlur only supports uint8, so implement a float32-compatible median filter
    half = window_size // 2
    padded = np.pad(channel_float, half, mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
    median_filtered = np.median(windows, axis=(-2, -1))

    result = np.where(is_firefly, median_filtered, channel_float)
    return result, is_firefly


def array_to_qpixmap(arr, normalize=True):
    """Convert a numpy array (float32, [0,1]) to QPixmap."""
    if arr.dtype != np.uint8:
        if normalize:
            # Normalize to 0-255 for display
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            else:
                arr = np.zeros_like(arr)
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    h, w = arr.shape[:2]

    if len(arr.shape) == 2:
        # Grayscale -> RGB
        arr = np.dstack([arr, arr, arr])

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Ensure contiguous
    arr = np.ascontiguousarray(arr)

    bytes_per_line = 3 * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def make_mask_overlay(mask, image_shape):
    """Create a red overlay visualization of the mask."""
    overlay = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
    overlay[mask] = [1.0, 0.0, 0.0]  # Red
    return overlay


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
# Image Preview Widget
# ──────────────────────────────────────────────
class ImagePreview(QLabel):
    """A label that displays an image scaled to fit, preserving aspect ratio."""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

    def _update_display(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
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
        self.setMinimumSize(1200, 750)

        # State
        self.image_path = None
        self.original_image = None
        self.original_dtype = None
        self.processed_image = None
        self.artifacts_mask = None
        self.preset_manager = PresetManager()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left panel: Controls ──
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        self._build_file_section(left_layout)
        self._build_parameter_section(left_layout)
        self._build_preset_section(left_layout)
        self._build_action_section(left_layout)
        left_layout.addStretch()

        # ── Right panel: Previews ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Preview tabs
        self.preview_tabs = QTabWidget()
        right_layout.addWidget(self.preview_tabs)

        # Tab 1: Side-by-side
        self.tab_compare = QWidget()
        compare_layout = QHBoxLayout(self.tab_compare)
        self.preview_original = ImagePreview("Original")
        self.preview_processed = ImagePreview("Processed")
        compare_layout.addWidget(self.preview_original)
        compare_layout.addWidget(self.preview_processed)
        self.preview_tabs.addTab(self.tab_compare, "Compare")

        # Tab 2: Artifacts mask
        self.tab_mask = QWidget()
        mask_layout = QVBoxLayout(self.tab_mask)
        self.preview_mask = ImagePreview("Artifacts Mask")
        mask_layout.addWidget(self.preview_mask)
        self.preview_tabs.addTab(self.tab_mask, "Artifacts Mask")

        # Tab 3: Original only
        self.tab_original = QWidget()
        orig_layout = QVBoxLayout(self.tab_original)
        self.preview_original_only = ImagePreview("Original")
        orig_layout.addWidget(self.preview_original_only)
        self.preview_tabs.addTab(self.tab_original, "Original")

        # Tab 4: Processed only
        self.tab_processed_only = QWidget()
        proc_layout = QVBoxLayout(self.tab_processed_only)
        self.preview_processed_only = ImagePreview("Processed")
        proc_layout.addWidget(self.preview_processed_only)
        self.preview_tabs.addTab(self.tab_processed_only, "Processed")

        # Status bar
        self.status_label = QLabel("Ready — open an image to begin")
        self.status_label.setStyleSheet("color: #aaa; padding: 2px;")
        right_layout.addWidget(self.status_label)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Apply dark theme
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
            QLabel#status {
                color: #aaa;
            }
        """)

    def _build_file_section(self, layout):
        group = QGroupBox("Image File")
        gl = QVBoxLayout(group)
        gl.setSpacing(6)

        btn_open = QPushButton("📂 Open Image...")
        btn_open.clicked.connect(self._on_open_image)
        gl.addWidget(btn_open)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #999; font-size: 12px;")
        gl.addWidget(self.file_label)

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
        self.window_size_spin.setSingleStep(2)  # Keep odd
        self.window_size_spin.valueChanged.connect(self._on_param_changed)
        gl.addWidget(self.window_size_spin, 0, 1)

        self.window_size_slider = QSlider(Qt.Horizontal)
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

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(int(THRESHOLD_MIN * 10), int(THRESHOLD_MAX * 10))
        self.threshold_slider.setValue(int(THRESHOLD_DEFAULT * 10))
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_spin.setValue(v / 10.0)
        )
        self.threshold_spin.valueChanged.connect(
            lambda v: self.threshold_slider.setValue(int(v * 10))
        )
        gl.addWidget(self.threshold_slider, 1, 2)

        layout.addWidget(group)

    def _build_preset_section(self, layout):
        group = QGroupBox("Presets")
        gl = QVBoxLayout(group)
        gl.setSpacing(4)

        # Preset list
        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(120)
        self.preset_list.itemClicked.connect(self._on_preset_selected)
        gl.addWidget(self.preset_list)

        # Preset controls
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

        # Default presets dropdown
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

        # Auto-preview checkbox
        self.auto_preview_cb = QCheckBox("Auto-preview on parameter change")
        self.auto_preview_cb.setChecked(True)
        gl.addWidget(self.auto_preview_cb)

        # Process button
        self.btn_process = QPushButton("⚡ Process")
        self.btn_process.setObjectName("btn_process")
        self.btn_process.clicked.connect(self._on_process)
        gl.addWidget(self.btn_process)

        # Save button
        self.btn_save = QPushButton("💾 Save Result As...")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)
        gl.addWidget(self.btn_save)

        layout.addWidget(group)

    # ── Slots ──

    def _on_open_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.exr *.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if not filepath:
            return

        try:
            self.image_path = filepath
            self.original_image, self.original_dtype = load_image(filepath)
            self.file_label.setText(os.path.basename(filepath))
            self.status_label.setText(f"Loaded: {os.path.basename(filepath)} ({self.original_image.shape[1]}×{self.original_image.shape[0]})")

            # Show original
            self.preview_original.set_image(self.original_image)
            self.preview_original_only.set_image(self.original_image)

            # Clear processed
            self.processed_image = None
            self.artifacts_mask = None
            self.preview_processed.clear()
            self.preview_processed_only.clear()
            self.preview_mask.clear()
            self.btn_save.setEnabled(False)

            # Auto-process if auto-preview is on
            if self.auto_preview_cb.isChecked():
                self._on_process()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def _on_param_changed(self):
        if self.original_image is not None and self.auto_preview_cb.isChecked():
            # Debounce via timer
            try:
                self._debounce_timer
            except AttributeError:
                self._debounce_timer = QTimer()
                self._debounce_timer.setSingleShot(True)
                self._debounce_timer.timeout.connect(self._on_process)
            self._debounce_timer.start(300)

    def _on_process(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return

        ws = self.window_size_spin.value()
        th = self.threshold_spin.value()

        self.status_label.setText("Processing...")
        QApplication.processEvents()

        try:
            result, mask = process_image_full(self.original_image, ws, th)
            self.processed_image = result
            self.artifacts_mask = mask

            # Update previews
            self.preview_processed.set_image(result)
            self.preview_processed_only.set_image(result)

            # Mask visualization
            mask_vis = make_mask_overlay(mask, self.original_image.shape)
            # Blend with original for context
            blended = self.original_image.copy()
            blended[mask] = blended[mask] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6
            self.preview_mask.set_image(blended)

            # Count fireflies
            num_fireflies = int(np.sum(mask))
            total_pixels = mask.size
            pct = 100.0 * num_fireflies / total_pixels
            self.status_label.setText(
                f"Done — {num_fireflies} firefly pixels ({pct:.3f}%) detected | "
                f"ws={ws}, th={th:.1f}"
            )
            self.btn_save.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def _on_save(self):
        if self.processed_image is None:
            return

        # Determine default extension
        if self.image_path and self.image_path.lower().endswith('.exr'):
            default_ext = ".exr"
            filter_str = "EXR (*.exr);;PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        else:
            default_ext = ".png"
            filter_str = "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;EXR (*.exr)"

        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_processed" + default_ext

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Processed Image", default_name, filter_str
        )
        if not filepath:
            return

        try:
            # Convert back to original dtype
            result_out = self.processed_image.copy()
            if self.original_dtype == np.uint16:
                result_out = (np.clip(result_out, 0, 1) * 65535).astype(np.uint16)
            elif self.original_dtype == np.uint8:
                result_out = (np.clip(result_out, 0, 1) * 255).astype(np.uint8)

            if filepath.lower().endswith('.exr'):
                # EXR write expects BGR or RGB float32
                write_exr(filepath, result_out)
            else:
                # OpenCV expects BGR
                if len(result_out.shape) == 3 and result_out.shape[2] >= 3:
                    result_out = result_out[:, :, ::-1]  # RGB -> BGR
                cv2.imwrite(filepath, result_out)

            self.status_label.setText(f"Saved: {os.path.basename(filepath)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")

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
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
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
