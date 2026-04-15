#!/usr/bin/env python
# coding: utf-8
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🚦 ADVANCED TRAFFIC DETECTION SYSTEM – UNIFIED v3.4 🚦               ║
║                                                                              ║
║   Two-model hybrid pipeline:                                                 ║
║     • YOLOv8n  → Real-time detection of traffic lights, stop signs,         ║
║                  vehicles, persons  (bounding boxes + color)                ║
║     • Keras CNN → Per-crop classification of 43 road sign categories        ║
║                   (speed limits, yield, no-entry, roundabout, …)            ║
║                                                                              ║
║   GUI: Tkinter  |  Inputs: image file, video file, live webcam              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Standard library ────────────────────────────────────────────────────────
import sys
import os
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# ─── Third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image, ImageTk

# Tkinter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Deep-learning
from ultralytics import YOLO
from tensorflow import keras

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

YOLO_MODEL_PATH      = r'/home/smartcar/myenv/models/yolov8n.onnx'
CONFIDENCE_THRESHOLD = 0.25

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

# ═══════════════════════════════════════════════════════════════════════════════
# CNN RESULT CACHE  (LRU, max 64 entries)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (lazy singleton)
# ═══════════════════════════════════════════════════════════════════════════════

_yolo_model  = None
_keras_model = None


def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            print("ONNX model not found, converting from PT...")
            pt_model = YOLO(YOLO_MODEL_PATH.replace('.onnx', '.pt'))
            pt_model.export(format='onnx', imgsz=320, half=True)
        _yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
        print("YOLOv8 ONNX ready")
    return _yolo_model


trafic_model_path = '/home/smartcar/myenv/models/model_RTSR.h5'


def get_keras_model():
    global _keras_model
    if _keras_model is None:
        _keras_model = keras.models.load_model(trafic_model_path)
        print("Keras CNN ready")
    return _keras_model


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _crop_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def classify_sign_crop(bgr_crop: np.ndarray) -> str:
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
                aspect_ratio = (w_box / h_box) if h_box > 0 else 0
                if aspect_ratio > 0.6:
                    continue
                color     = get_traffic_light_color(frame, bbox)
                box_color = DETECTION_COLORS.get(color, DETECTION_COLORS['unknown'])
                draw_box(frame, bbox, f"TL: {color.upper()} {conf:.0%}", box_color)
                detections['traffic_lights'].append(
                    {'color': color, 'conf': conf, 'bbox': bbox.tolist()})

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


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

FRAME_SKIP = 1


class DetectionWorker(threading.Thread):
    def __init__(self, source, confidence, on_frame, on_status, on_video_ended, on_finished):
        super().__init__(daemon=True)
        self.source         = source
        self.confidence     = confidence
        self.on_frame       = on_frame
        self.on_status      = on_status
        self.on_video_ended = on_video_ended
        self.on_finished    = on_finished
        self._running       = True

    def stop(self):
        self._running = False

    def run(self):
        # Single image
        if isinstance(self.source, str) and self.source.lower().endswith(
                ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')):
            img = cv2.imread(self.source)
            if img is None:
                self.on_status("Cannot read image file.")
                self.on_finished()
                return
            annotated, dets = process_frame(img, self.confidence)
            self.on_frame(annotated, dets)
            self.on_status(
                f"Image processed – TL: {len(dets['traffic_lights'])}  "
                f"Signs: {len(dets['stop_signs'])}  "
                f"Vehicles: {len(dets['vehicles'])}")
            self.on_finished()
            return

        # Video / webcam
        is_camera = isinstance(self.source, int)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.on_status("Cannot open video / camera.")
            self.on_finished()
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
            self.on_frame(annotated, dets)
            self.on_status(
                f"TL: {len(dets['traffic_lights'])}  "
                f"Signs: {len(dets['stop_signs'])}  "
                f"Vehicles: {len(dets['vehicles'])}  "
                f"Persons: {dets['persons']}")

        cap.release()

        if ended_natural:
            self.on_video_ended()
            self.on_status("Video finished.")
        else:
            self.on_status("Stopped.")

        self.on_finished()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GUI WINDOW  (Tkinter)
# ═══════════════════════════════════════════════════════════════════════════════

_DISPLAY_INTERVAL_MS = 33
_SIDEBAR_INTERVAL_MS = 125

# Colors
BG       = "#12121e"
BG2      = "#1a1a2e"
GOLD     = "#FFD700"
BTN_BG   = "#FEBD07"
BTN_FG   = "#1a1a2e"
BTN_STOP = "#c0392b"
TEXT     = "#dddddd"
DIM      = "#888888"
RED_LBL  = "#FF6666"
ORG_LBL  = "#FFA500"
BLU_LBL  = "#66BBFF"
PUR_LBL  = "#CC88FF"
SEP      = "#444444"


class MainWindow:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Traffic Detection System v3.4")
        self.root.geometry("1280x780")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._worker         = None
        self._last_frame     = None
        self._confidence     = CONFIDENCE_THRESHOLD
        self._last_display_t = 0.0
        self._last_sidebar_t = 0.0
        self._models_ready   = False

        # FPS
        self._fps_frame_count  = 0
        self._fps_window_start = time.monotonic()
        self._current_fps      = 0.0

        # Pending frame from worker thread (thread-safe handoff)
        self._pending_frame = None
        self._pending_dets  = None
        self._frame_lock    = threading.Lock()

        self._build_ui()
        self._preload_models()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Poll for new frames every 33 ms on the main thread
        self.root.after(33, self._poll_frame)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        title = tk.Label(self.root,
                         text="Advanced Traffic Detection System – Dual Model",
                         bg=BG, fg=GOLD,
                         font=("Georgia", 18, "bold"),
                         pady=8)
        title.pack(fill="x")

        # Content row
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=8, pady=4)

        # Display canvas (fixed size)
        canvas_frame = tk.Frame(content, bg="#1a1a2e",
                                bd=2, relief="flat",
                                highlightbackground=SEP, highlightthickness=2)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.display_label = tk.Label(canvas_frame,
                                      text="Select an image, video or start the live camera.",
                                      bg="#1a1a2e", fg=DIM,
                                      width=DISPLAY_W, height=DISPLAY_H,
                                      anchor="center")
        self.display_label.pack()
        self.display_label.config(width=DISPLAY_W, height=DISPLAY_H)

        # Sidebar
        self._build_sidebar(content)

        # Controls
        self._build_controls()

        # Status bar
        self.status_var = tk.StringVar(value="Loading models...")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bg=BG, fg="#aaaaaa",
                              font=("Helvetica", 10),
                              anchor="w", padx=8, pady=4)
        status_bar.pack(fill="x", side="bottom")

    def _build_sidebar(self, parent):
        sidebar = tk.LabelFrame(parent,
                                text="Detection Log",
                                bg=BG2, fg=GOLD,
                                font=("Georgia", 10, "bold"),
                                bd=1, relief="groove",
                                labelanchor="nw",
                                padx=6, pady=6)
        sidebar.pack(side="right", fill="y", padx=(0, 0))

        self.log_vars = {}
        rows = [
            ("tl_label", "Traffic Lights",    RED_LBL),
            ("st_label", "Stop / Road Signs", ORG_LBL),
            ("ve_label", "Vehicles",          BLU_LBL),
            ("pe_label", "Persons",           PUR_LBL),
        ]
        for key, row_title, color in rows:
            tk.Label(sidebar, text=row_title,
                     bg=BG2, fg=color,
                     font=("Helvetica", 11, "bold"),
                     anchor="w").pack(fill="x", pady=(6, 0))
            var = tk.StringVar(value="–")
            lbl = tk.Label(sidebar, textvariable=var,
                           bg=BG2, fg=TEXT,
                           font=("Helvetica", 10),
                           anchor="w", justify="left",
                           wraplength=200)
            lbl.pack(fill="x", padx=10)
            self.log_vars[key] = var
            tk.Frame(sidebar, bg=SEP, height=1).pack(fill="x", pady=4)

    def _build_controls(self):
        ctrl_frame = tk.LabelFrame(self.root,
                                   text="Controls",
                                   bg=BG2, fg=GOLD,
                                   font=("Georgia", 10, "bold"),
                                   bd=1, relief="groove",
                                   labelanchor="nw",
                                   padx=8, pady=6)
        ctrl_frame.pack(fill="x", padx=8, pady=(4, 0))

        # Button row
        btn_row = tk.Frame(ctrl_frame, bg=BG2)
        btn_row.pack(fill="x", pady=(0, 6))

        def make_btn(parent, text, cmd, bg=BTN_BG, fg=BTN_FG):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=bg, fg=fg,
                          font=("Helvetica", 11, "bold"),
                          relief="flat", bd=0,
                          padx=14, pady=7,
                          cursor="hand2",
                          activebackground="#ffd040",
                          activeforeground=BTN_FG)
            b.pack(side="left", padx=4)
            return b

        self.btn_image = make_btn(btn_row, "🖼  Image",        self._open_image)
        self.btn_video = make_btn(btn_row, "🎬  Video",        self._open_video)
        self.btn_cam   = make_btn(btn_row, "📷  Live Camera",  self._start_camera)
        self.btn_stop  = make_btn(btn_row, "⏹  Stop",          self._stop_worker,
                                  bg=BTN_STOP, fg="white")
        self.btn_save  = make_btn(btn_row, "💾  Save Frame",   self._save_frame)

        # Confidence slider row
        conf_row = tk.Frame(ctrl_frame, bg=BG2)
        conf_row.pack(fill="x")

        tk.Label(conf_row, text="Confidence threshold:",
                 bg=BG2, fg=TEXT, font=("Helvetica", 10)).pack(side="left", padx=(0, 6))

        self.conf_var = tk.IntVar(value=int(self._confidence * 100))
        slider = ttk.Scale(conf_row, from_=10, to=90,
                           orient="horizontal",
                           variable=self.conf_var,
                           command=self._update_confidence,
                           length=300)
        slider.pack(side="left")

        self.conf_lbl_var = tk.StringVar(value=f"{int(self._confidence * 100)}%")
        tk.Label(conf_row, textvariable=self.conf_lbl_var,
                 bg=BG2, fg=GOLD,
                 font=("Helvetica", 10, "bold"),
                 width=5).pack(side="left", padx=6)

        self._set_buttons_enabled(False)

    # ── Model preloading ──────────────────────────────────────────────────────

    def _preload_models(self):
        def _load():
            self._set_status("Loading YOLOv8 model...")
            try:
                get_yolo_model()
            except Exception as e:
                self._set_status(f"YOLOv8 load failed: {e}")
                return
            self._set_status("Loading Keras CNN...")
            try:
                get_keras_model()
            except Exception as e:
                self._set_status(f"Keras CNN load failed: {e}")
                return
            self._set_status("All models ready – select an image, video or camera.")
            self._models_ready = True
            self.root.after(0, lambda: self._set_buttons_enabled(True))

        threading.Thread(target=_load, daemon=True).start()

    # ── Slots / callbacks ─────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        bg    = BTN_BG if enabled else "#555555"
        fg    = BTN_FG if enabled else "#888888"
        for btn in [self.btn_image, self.btn_video, self.btn_cam]:
            btn.config(state=state, bg=bg, fg=fg)

    def _update_confidence(self, val):
        v = int(float(val))
        self._confidence = v / 100.0
        self.conf_lbl_var.set(f"{v}%")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.ppm")])
        if path:
            self._stop_worker()
            self._start_worker(path)

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self._stop_worker()
            self._start_worker(path)

    def _start_camera(self):
        self._stop_worker()
        self._start_worker(0)

    def _start_worker(self, source):
        self._fps_frame_count  = 0
        self._fps_window_start = time.monotonic()
        self._current_fps      = 0.0

        self._worker = DetectionWorker(
            source=source,
            confidence=self._confidence,
            on_frame=self._on_frame_from_thread,
            on_status=self._set_status,
            on_video_ended=self._on_video_ended,
            on_finished=self._on_finished,
        )
        self._worker.start()
        self._set_status("Running...")

    def _stop_worker(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
            self._worker.join(timeout=3)
        self._worker = None

    # Called from worker thread – just store the frame; main thread picks it up
    def _on_frame_from_thread(self, frame: np.ndarray, dets: dict):
        with self._frame_lock:
            self._pending_frame = frame
            self._pending_dets  = dets

    # Main-thread poller – checks for a new frame every 33 ms
    def _poll_frame(self):
        with self._frame_lock:
            frame = self._pending_frame
            dets  = self._pending_dets
            self._pending_frame = None
            self._pending_dets  = None

        if frame is not None:
            now = time.monotonic()
            self._last_frame = frame

            # FPS
            self._fps_frame_count += 1
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self._current_fps      = self._fps_frame_count / elapsed
                self._fps_frame_count  = 0
                self._fps_window_start = now
                self.root.title(
                    f"Traffic Detection System v3.4  –  {self._current_fps:.1f} FPS")

            if (now - self._last_display_t) * 1000 >= _DISPLAY_INTERVAL_MS:
                self._show_frame(frame)
                self._last_display_t = now

            if (now - self._last_sidebar_t) * 1000 >= _SIDEBAR_INTERVAL_MS:
                self._update_sidebar(dets)
                self._last_sidebar_t = now

        self.root.after(33, self._poll_frame)

    def _on_video_ended(self):
        self.root.after(0, lambda: messagebox.showinfo(
            "Video Finished",
            "The video has finished playing.\n\n"
            "You can select another file or start the live camera."))

    def _on_finished(self):
        pass

    def _save_frame(self):
        if self._last_frame is None:
            self._set_status("No frame to save yet.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(OUTPUT_DIR / f"snapshot_{ts}.jpg")
        cv2.imwrite(path, self._last_frame)
        self._set_status(f"Saved: {path}")

    # ── Display ───────────────────────────────────────────────────────────────

    def _show_frame(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Scale to fit DISPLAY_W x DISPLAY_H keeping aspect ratio
        pil.thumbnail((DISPLAY_W, DISPLAY_H), Image.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        self.display_label.config(image=photo, text="")
        self.display_label.image = photo   # keep reference

    def _update_sidebar(self, dets: dict):
        tls = dets.get('traffic_lights', [])
        self.log_vars['tl_label'].set(
            f"{len(tls)} detected\n" + ", ".join(t['color'].upper() for t in tls)
            if tls else "None")

        stops = dets.get('stop_signs', [])
        self.log_vars['st_label'].set(
            "\n".join(f"- {s['label'][:30]} ({s['conf']:.0%})" for s in stops)
            if stops else "None")

        vehicles = dets.get('vehicles', [])
        if vehicles:
            counts: dict = {}
            for v in vehicles:
                counts[v['type']] = counts.get(v['type'], 0) + 1
            self.log_vars['ve_label'].set(
                f"{len(vehicles)} total\n" +
                "  ".join(f"{k}:{n}" for k, n in counts.items()))
        else:
            self.log_vars['ve_label'].set("None")

        pe = dets.get('persons', 0)
        self.log_vars['pe_label'].set(str(pe) if pe else "None")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_worker()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = MainWindow(root)
    root.mainloop()
