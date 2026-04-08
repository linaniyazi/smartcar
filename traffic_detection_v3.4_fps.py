#!/usr/bin/env python
# coding: utf-8
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🚦 ADVANCED TRAFFIC DETECTION SYSTEM — UNIFIED v3.4 🚦               ║
║                                                                              ║
║   Two-model hybrid pipeline:                                                 ║
║     • YOLOv8n  → Real-time detection of traffic lights, stop signs,         ║
║                  vehicles, persons  (bounding boxes + color)                ║
║     • Keras CNN → Per-crop classification of 43 road sign categories        ║
║                   (speed limits, yield, no-entry, roundabout, …)            ║
║                                                                              ║
║   GUI: PyQt5  |  Inputs: image file, video file, live webcam                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Fixes in v3.2 (updated):
  • FPS counter shown in the window title (updates once per second)
  • Image display is now CONSTANT size (fixed 820x520) — never grows or
    shrinks with window resizing
  • Traffic light color detection always returns the dominant color
    (red/yellow/green) — no more "unknown" fallback
  • Image no longer zoomed/distorted — YOLO always runs on the original
    full-resolution frame; scaling to the QLabel happens separately after
  • Worker now emits a dedicated video_ended signal → GUI shows a popup
  • np.ascontiguousarray() used before QImage to prevent memory artifacts
  • Models preloaded at startup (from v3.1)
  • Frame skipping, CNN cache, display/sidebar throttle (from v3.1)
"""

# ─── Standard library ────────────────────────────────────────────────────────
import sys
import os
import time
import hashlib
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# ─── Third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,
    QSlider, QGroupBox, QStatusBar, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Deep-learning
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from tensorflow import keras

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

YOLO_MODEL_PATH      = r'C:\Users\VICTUS\Downloads\yolov8n.onnx'
CONFIDENCE_THRESHOLD = 0.25

# Constant display size for the video/image label (change to your preference)
DISPLAY_W = 820
DISPLAY_H = 520

OUTPUT_DIR = Path("detection_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAFFIC_LIGHT_CLASS = 9
STOP_SIGN_CLASS     = 11
VEHICLE_CLASSES     = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

DETECTION_COLORS = {
    'red':      (0,   0,   255),
    'yellow':   (0,   255, 255),
    'green':    (0,   255, 0),
    'unknown':  (128, 128, 128),
    'stop_sign':(0,   165, 255),
    'vehicle':  (255, 100, 0),
    'person':   (255, 0,   255),
    'sign':     (0,   200, 255),
}

SIGN_CLASSES = {
    1:  'Speed limit (20km/h)',   2:  'Speed limit (30km/h)',
    3:  'Speed limit (50km/h)',   4:  'Speed limit (60km/h)',
    5:  'Speed limit (70km/h)',   6:  'Speed limit (80km/h)',
    7:  'End of speed limit (80km/h)', 8: 'Speed limit (100km/h)',
    9:  'Speed limit (120km/h)',  10: 'No passing',
    11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection',
    13: 'Priority road',          14: 'Yield',
    15: 'Stop',                   16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited', 18: 'No entry',
    19: 'General caution',        20: 'Dangerous curve left',
    21: 'Dangerous curve right',  22: 'Double curve',
    23: 'Bumpy road',             24: 'Slippery road',
    25: 'Road narrows on the right', 26: 'Road work',
    27: 'Traffic signals',        28: 'Pedestrians',
    29: 'Children crossing',      30: 'Bicycles crossing',
    31: 'Beware of ice/snow',     32: 'Wild animals crossing',
    33: 'End speed + passing limits', 34: 'Turn right ahead',
    35: 'Turn left ahead',        36: 'Ahead only',
    37: 'Go straight or right',   38: 'Go straight or left',
    39: 'Keep right',             40: 'Keep left',
    41: 'Roundabout mandatory',   42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons',
}

# ═════════════════════════════════════════════════════════════════════════════
# CNN RESULT CACHE  (LRU, max 64 entries)
# ═════════════════════════════════════════════════════════════════════════════

class _LRUCache:
    def __init__(self, maxsize=64):
        self._cache   = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

_sign_cache = _LRUCache(maxsize=64)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (lazy singleton)
# ═════════════════════════════════════════════════════════════════════════════

_yolo_model  = None
_keras_model = None


def get_yolo_model() -> YOLO:
    global _yolo_model
    if not os.path.exists(YOLO_MODEL_PATH):
            print("ONNX model not found, converting from PT...")
            pt_model = YOLO(YOLO_MODEL_PATH.replace('.onnx', '.pt'))
            pt_model.export(format='onnx', imgsz=320, half=True)
            
    _yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
    print("YOLOv8 ONNX ready")
    return _yolo_model


def get_keras_model():
    global _keras_model
    if _keras_model is None:
        print("Downloading / loading Keras CNN ...")
        path = hf_hub_download(
            repo_id="LinaNiyazi/traffic-sign-model",
            filename="model_RTSR.h5"
        )
        _keras_model = keras.models.load_model(path)
        print("Keras CNN ready")
    return _keras_model


# ═════════════════════════════════════════════════════════════════════════════
# DETECTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _crop_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def classify_sign_crop(bgr_crop: np.ndarray) -> str:
    """Keras CNN on a BGR crop, with LRU cache."""
    pil  = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
    pil  = pil.resize((30, 30))
    arr  = np.array(pil)
    key  = _crop_hash(arr)
    hit  = _sign_cache.get(key)
    if hit is not None:
        return hit
    batch = np.expand_dims(arr, axis=0)
    pred  = int(np.argmax(get_keras_model().predict(batch, verbose=0), axis=1)[0])
    label = SIGN_CLASSES.get(pred + 1, "Unknown sign")
    _sign_cache.put(key, label)
    return label


def get_traffic_light_color(image: np.ndarray, bbox: np.ndarray) -> str:
    """
    Detect the dominant color in a traffic light crop.
    Always returns 'red', 'yellow', or 'green' — whichever has the most
    matching pixels in the HSV mask, with no 'unknown' fallback.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "red"
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 3:
        return "red"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'red':    [(np.array([0,  80, 80]), np.array([12,  255, 255])),
                   (np.array([168,80, 80]), np.array([180, 255, 255]))],
        'yellow': [(np.array([18, 80, 80]), np.array([35,  255, 255]))],
        'green':  [(np.array([35, 80, 80]), np.array([85,  255, 255]))],
    }
    scores = {}
    for color, ranges in color_ranges.items():
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        scores[color] = int(cv2.countNonZero(mask))

    # Always return the color with the most matching pixels — never "unknown"
    return max(scores, key=scores.get)


def draw_box(frame: np.ndarray, bbox, label: str, color: tuple, thickness: int = 2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), font, fs,
                (255, 255, 255), ft, cv2.LINE_AA)

def process_frame(frame: np.ndarray, confidence: float = CONFIDENCE_THRESHOLD) -> tuple:
    """
  Run the full pipeline on ONE BGR frame at its ORIGINAL resolution.
    Never resize before calling this — scaling for display is done separately
    in _show_frame() on the GUI side only.
    """
    yolo    = get_yolo_model()
    results = yolo.predict(frame, verbose=False, conf=confidence, imgsz=320)

    detections = {'traffic_lights': [], 'stop_signs': [], 'vehicles': [], 'persons': 0}

    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            bbox       = box.xyxy[0].cpu().numpy()
            class_name = yolo.names[cls_id]

            if cls_id == TRAFFIC_LIGHT_CLASS:
           
                x1, y1, x2, y2 = bbox
                w_box = x2 - x1
                h_box = y2 - y1
                
                if h_box > 0:  # لتجنب القسمة على صفر
                    aspect_ratio = w_box / h_box
                else:
                    aspect_ratio = 0

                # إذا كانت النسبة قريبة من 1 (مربعة) فهي إشارة مشاة -> نتجاهلها
                # العتبة هي 0.6 منغيرها للمناسب اذا بدنا 
                if aspect_ratio > 0.6: 
                    continue
                
                # إذا كانت النسبة صغيرة (الطول أكبر بكثير من العرض) فهي إشارة سيارات -> نكمل المعالجة
                color     = get_traffic_light_color(frame, bbox)
                box_color = DETECTION_COLORS.get(color, DETECTION_COLORS['unknown'])
                draw_box(frame, bbox, f"TL: {color.upper()} {conf:.0%}", box_color)
                detections['traffic_lights'].append(
                    {'color': color, 'conf': conf, 'bbox': bbox.tolist()})
                # -------------------------------------------

            elif cls_id == STOP_SIGN_CLASS:
                x1, y1, x2, y2 = map(int, bbox)
                crop       = frame[max(0, y1):y2, max(0, x1):x2]
                sign_label = classify_sign_crop(crop) if crop.size > 0 else "Stop sign"
                draw_box(frame, bbox, f"{sign_label[:28]} {conf:.0%}",
                         DETECTION_COLORS['stop_sign'])
                detections['stop_signs'].append(
                    {'label': sign_label, 'conf': conf, 'bbox': bbox.tolist()})

            elif class_name in VEHICLE_CLASSES:
                draw_box(frame, bbox, f"{class_name.upper()} {conf:.0%}",
                         DETECTION_COLORS['vehicle'])
                detections['vehicles'].append(
                    {'type': class_name, 'conf': conf, 'bbox': bbox.tolist()})

            elif class_name == 'person':
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              DETECTION_COLORS['person'], 2)
                detections['persons'] += 1

    _draw_hud(frame, detections)
    return frame, detections


def _draw_hud(frame: np.ndarray, detections: dict):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    tl  = len(detections['traffic_lights'])
    st  = len(detections['stop_signs'])
    ve  = len(detections['vehicles'])
    pe  = detections['persons']
    hud = f"Traffic Lights: {tl}  |  Signs: {st}  |  Vehicles: {ve}  |  Persons: {pe}"
    cv2.putText(frame, hud, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PRELOADER THREAD
# ═════════════════════════════════════════════════════════════════════════════

class ModelPreloader(QThread):
    status_update = pyqtSignal(str)
    finished      = pyqtSignal()

    def run(self):
        self.status_update.emit("Loading YOLOv8 model...")
        try:
            get_yolo_model()
        except Exception as e:
            self.status_update.emit(f"YOLOv8 load failed: {e}")
            self.finished.emit()
            return
        self.status_update.emit("Loading Keras CNN...")
        try:
            get_keras_model()
        except Exception as e:
            self.status_update.emit(f"Keras CNN load failed: {e}")
            self.finished.emit()
            return
        self.status_update.emit("All models ready — select an image, video or camera.")
        self.finished.emit()


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER THREAD
# ═════════════════════════════════════════════════════════════════════════════

FRAME_SKIP = 1   # raise to 2 or 3 if still laggy


class DetectionWorker(QThread):
    frame_ready   = pyqtSignal(np.ndarray, dict)
    status_update = pyqtSignal(str)
    video_ended   = pyqtSignal()
    finished      = pyqtSignal()

    def __init__(self, source, confidence=CONFIDENCE_THRESHOLD, parent=None):
        super().__init__(parent)
        self.source     = source
        self.confidence = confidence
        self._running   = True

    def stop(self):
        self._running = False

    def run(self):
        # ── Single image ─────────────────────────────────────────────────
        if isinstance(self.source, str) and self.source.lower().endswith(
                ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')):
            img = cv2.imread(self.source)
            if img is None:
                self.status_update.emit("Cannot read image file.")
                self.finished.emit()
                return
            annotated, dets = process_frame(img, self.confidence)
            self.frame_ready.emit(annotated, dets)
            self.status_update.emit(
                f"Image processed — TL: {len(dets['traffic_lights'])}  "
                f"Signs: {len(dets['stop_signs'])}  "
                f"Vehicles: {len(dets['vehicles'])}")
            self.finished.emit()
            return

        # ── Video file or webcam ─────────────────────────────────────────
        is_camera = isinstance(self.source, int)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_update.emit("Cannot open video / camera.")
            self.finished.emit()
            return

        if is_camera:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        skip_counter  = 0
        ended_natural = False

        while self._running:
            ret, frame = cap.read()
            if not ret:
                if not is_camera:
                    ended_natural = True
                break

            if skip_counter < FRAME_SKIP:
                skip_counter += 1
                continue
            skip_counter = 0

            annotated, dets = process_frame(frame, self.confidence)
            self.frame_ready.emit(annotated, dets)
            self.status_update.emit(
                f"TL: {len(dets['traffic_lights'])}  "
                f"Signs: {len(dets['stop_signs'])}  "
                f"Vehicles: {len(dets['vehicles'])}  "
                f"Persons: {dets['persons']}")

        cap.release()

        if ended_natural:
            self.video_ended.emit()
            self.status_update.emit("Video finished.")
        else:
            self.status_update.emit("Stopped.")

        self.finished.emit()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN GUI WINDOW
# ═════════════════════════════════════════════════════════════════════════════

_DISPLAY_INTERVAL_MS = 33    # ~30 fps cap for the display label
_SIDEBAR_INTERVAL_MS = 125   # ~8 Hz cap for the detection log


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Detection System v3.4")
        self.resize(1280, 780)
        self._worker         = None
        self._last_frame     = None
        self._confidence     = CONFIDENCE_THRESHOLD
        self._last_display_t = 0.0
        self._last_sidebar_t = 0.0

        # FPS tracking
        self._fps_frame_count  = 0
        self._fps_window_start = time.monotonic()
        self._current_fps      = 0.0

        self._setup_ui()
        self._apply_dark_theme()

        # Preload both models immediately in the background
        self._preloader = ModelPreloader()
        self._preloader.status_update.connect(self.status_bar.showMessage)
        self._preloader.finished.connect(self._on_models_ready)
        self._set_buttons_enabled(False)
        self._preloader.start()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        title = QLabel("Advanced Traffic Detection System — Dual Model")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Georgia", 18, QFont.Bold))
        title.setStyleSheet("color: #FFD700; padding: 6px;")
        root.addWidget(title)

        content_row = QHBoxLayout()
        content_row.setSpacing(8)

        # Fixed-size display label — DISPLAY_W x DISPLAY_H, never resizes
        self.display_label = QLabel("Select an image, video or start the live camera.")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        self.display_label.setStyleSheet(
            "background-color: #1a1a2e; border: 2px solid #444;"
            "border-radius: 6px; color: #888;")

        content_row.addWidget(self.display_label, stretch=3)
        content_row.addWidget(self._build_sidebar(), stretch=1)

        root.addLayout(content_row)
        root.addWidget(self._build_controls())

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("color: #aaa; font-size: 11px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Loading models...")

    def _build_sidebar(self) -> QGroupBox:
        group = QGroupBox("Detection Log")
        group.setFont(QFont("Georgia", 10, QFont.Bold))
        group.setStyleSheet(
            "QGroupBox { color:#FFD700; border:1px solid #555;"
            "border-radius:5px; margin-top:8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 14, 6, 6)

        self.log_labels = {}
        rows = [
            ("tl_label", "Traffic Lights",   "#FF6666"),
            ("st_label", "Stop / Road Signs", "#FFA500"),
            ("ve_label", "Vehicles",          "#66BBFF"),
            ("pe_label", "Persons",           "#CC88FF"),
        ]
        for key, row_title, color in rows:
            lbl_title = QLabel(row_title)
            lbl_title.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 11px;")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet(
                "color: #ddd; font-size: 11px; padding-left: 10px;")
            lbl_val.setWordWrap(True)
            self.log_labels[key] = lbl_val
            layout.addWidget(lbl_title)
            layout.addWidget(lbl_val)
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #444;")
            layout.addWidget(sep)

        layout.addStretch()
        return group

    def _build_controls(self) -> QGroupBox:
        group = QGroupBox("Controls")
        group.setFont(QFont("Georgia", 10, QFont.Bold))
        group.setStyleSheet(
            "QGroupBox { color:#FFD700; border:1px solid #555;"
            "border-radius:5px; margin-top:6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 14, 8, 8)
        outer.setSpacing(6)

        btn_row    = QHBoxLayout()
        btn_style  = ("QPushButton {"
                      "  background:#FEBD07; color:#1a1a2e; font-weight:bold;"
                      "  font-size:12px; border-radius:5px; padding:8px 16px; }"
                      "QPushButton:hover { background:#ffd040; }"
                      "QPushButton:pressed { background:#e0a800; }")
        stop_style = ("QPushButton {"
                      "  background:#c0392b; color:white; font-weight:bold;"
                      "  font-size:12px; border-radius:5px; padding:8px 16px; }"
                      "QPushButton:hover { background:#e74c3c; }"
                      "QPushButton:pressed { background:#a93226; }")
        disabled_style = ("QPushButton {"
                          "  background:#555; color:#888; font-weight:bold;"
                          "  font-size:12px; border-radius:5px; padding:8px 16px; }")

        self.btn_image = QPushButton("  Image")
        self.btn_video = QPushButton("  Video")
        self.btn_cam   = QPushButton("  Live Camera")
        self.btn_stop  = QPushButton("  Stop")
        self.btn_save  = QPushButton("  Save Frame")

        self._btn_style          = btn_style
        self._btn_disabled_style = disabled_style

        for btn in [self.btn_image, self.btn_video, self.btn_cam, self.btn_save]:
            btn.setStyleSheet(btn_style)
        self.btn_stop.setStyleSheet(stop_style)

        self.btn_image.clicked.connect(self._open_image)
        self.btn_video.clicked.connect(self._open_video)
        self.btn_cam.clicked.connect(self._start_camera)
        self.btn_stop.clicked.connect(self._stop_worker)
        self.btn_save.clicked.connect(self._save_frame)

        for btn in [self.btn_image, self.btn_video, self.btn_cam,
                    self.btn_stop, self.btn_save]:
            btn_row.addWidget(btn)
        outer.addLayout(btn_row)

        conf_row = QHBoxLayout()
        conf_lbl = QLabel("Confidence threshold:")
        conf_lbl.setStyleSheet("color:#ddd; font-size:11px;")
        self.conf_val_lbl = QLabel(f"{int(self._confidence * 100)}%")
        self.conf_val_lbl.setFixedWidth(38)
        self.conf_val_lbl.setStyleSheet(
            "color:#FFD700; font-size:11px; font-weight:bold;")

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(int(self._confidence * 100))
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setStyleSheet(
            "QSlider::groove:horizontal { background:#444; height:6px; border-radius:3px; }"
            "QSlider::handle:horizontal  { background:#FEBD07; width:14px; height:14px;"
            "  margin:-4px 0; border-radius:7px; }")
        self.conf_slider.valueChanged.connect(self._update_confidence)

        conf_row.addWidget(conf_lbl)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_val_lbl)
        outer.addLayout(conf_row)

        return group

    def _apply_dark_theme(self):
        self.setStyleSheet(
            "QMainWindow, QWidget { background-color: #12121e; }"
            "QGroupBox { background-color: #1a1a2e; }"
            "QLabel    { color: #dddddd; }")

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_models_ready(self):
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool):
        for btn in [self.btn_image, self.btn_video, self.btn_cam]:
            btn.setEnabled(enabled)
            btn.setStyleSheet(
                self._btn_style if enabled else self._btn_disabled_style)

    def _update_confidence(self, val: int):
        self._confidence = val / 100.0
        self.conf_val_lbl.setText(f"{val}%")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.ppm)")
        if path:
            self._stop_worker()
            self._start_worker(path)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._stop_worker()
            self._start_worker(path)

    def _start_camera(self):
        self._stop_worker()
        self._start_worker(0)

    def _start_worker(self, source):
        # Reset FPS counters whenever a new source starts
        self._fps_frame_count  = 0
        self._fps_window_start = time.monotonic()
        self._current_fps      = 0.0

        self._worker = DetectionWorker(source, self._confidence)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_update.connect(self.status_bar.showMessage)
        self._worker.video_ended.connect(self._on_video_ended)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.status_bar.showMessage("Running...")

    def _stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        self._worker = None

    def _on_frame(self, frame: np.ndarray, dets: dict):
        now = time.monotonic()
        self._last_frame = frame

        # ── FPS calculation ───────────────────────────────────────────────
        # Count every frame received from the worker (true YOLO processing rate)
        self._fps_frame_count += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:   # recalculate once per second
            self._current_fps      = self._fps_frame_count / elapsed
            self._fps_frame_count  = 0
            self._fps_window_start = now
            self.setWindowTitle(
                f"Traffic Detection System v3.2  —  {self._current_fps:.1f} FPS")

        if (now - self._last_display_t) * 1000 >= _DISPLAY_INTERVAL_MS:
            self._show_frame(frame)
            self._last_display_t = now

        if (now - self._last_sidebar_t) * 1000 >= _SIDEBAR_INTERVAL_MS:
            self._update_sidebar(dets)
            self._last_sidebar_t = now

    def _on_video_ended(self):
        """Popup shown only when a video file finishes naturally."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Video Finished")
        msg.setText("The video has finished playing.")
        msg.setInformativeText(
            "You can select another file or start the live camera.")
        msg.setIcon(QMessageBox.Information)
        msg.setStyleSheet(
            "QMessageBox { background-color: #1a1a2e; }"
            "QLabel { color: #ffffff; font-size: 13px; }"
            "QPushButton { background:#FEBD07; color:#1a1a2e; font-weight:bold;"
            "  border-radius:4px; padding:6px 18px; }")
        msg.exec_()

    def _on_finished(self):
        pass  # status bar already updated by the worker

    def _save_frame(self):
        if self._last_frame is None:
            self.status_bar.showMessage("No frame to save yet.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(OUTPUT_DIR / f"snapshot_{ts}.jpg")
        cv2.imwrite(path, self._last_frame)
        self.status_bar.showMessage(f"Saved: {path}")

    # ── Display ───────────────────────────────────────────────────────────

    def _show_frame(self, bgr: np.ndarray):
        """
        Scale the annotated frame to fit the fixed-size QLabel for display only.
        The label is locked to DISPLAY_W x DISPLAY_H so the box never changes
        size regardless of window resizing.
        """
        rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        rgb_c    = np.ascontiguousarray(rgb)
        qimg     = QImage(rgb_c.data, w, h, ch * w, QImage.Format_RGB888)
        pix      = QPixmap.fromImage(qimg)
        scaled   = pix.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled)

    def _update_sidebar(self, dets: dict):
        tls = dets.get('traffic_lights', [])
        self.log_labels['tl_label'].setText(
            f"{len(tls)} detected\n" + ", ".join(t['color'].upper() for t in tls)
            if tls else "None")

        stops = dets.get('stop_signs', [])
        self.log_labels['st_label'].setText(
            "\n".join(f"- {s['label'][:30]} ({s['conf']:.0%})" for s in stops)
            if stops else "None")

        vehicles = dets.get('vehicles', [])
        if vehicles:
            counts: dict = {}
            for v in vehicles:
                counts[v['type']] = counts.get(v['type'], 0) + 1
            self.log_labels['ve_label'].setText(
                f"{len(vehicles)} total\n" +
                "  ".join(f"{k}:{n}" for k, n in counts.items()))
        else:
            self.log_labels['ve_label'].setText("None")

        pe = dets.get('persons', 0)
        self.log_labels['pe_label'].setText(str(pe) if pe else "None")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Traffic Detection System v3.2")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
