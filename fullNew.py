#!/usr/bin/env python
# coding: utf-8
"""
Advanced Traffic Detection System v3.4 - Raspberry Pi Edition
Uses ONNX Runtime + TFLite (NO PyTorch, NO TensorFlow)
"""

# â”€â”€â”€ Standard library â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import sys
import os
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# â”€â”€â”€ Third-party â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import cv2
import numpy as np
from PIL import Image, ImageTk

# Tkinter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Deep-learning (lightweight - no torch!)
import onnxruntime as ort
import tflite_runtime.interpreter as tflite

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONFIGURATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

YOLO_MODEL_PATH      = '/home/smartcar/newenv/smartcar-main/yolov8n.onnx'
TFLITE_MODEL_PATH    = '/home/smartcar/newenv/smartcar-main/model_RTSR.tflite'
CONFIDENCE_THRESHOLD = 0.25

DISPLAY_W = 820
DISPLAY_H = 520

OUTPUT_DIR = Path("detection_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAFFIC_LIGHT_CLASS = 9
STOP_SIGN_CLASS     = 11
VEHICLE_CLASSES     = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

# Full COCO class names (80 classes)
COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush',
}

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

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CNN RESULT CACHE (LRU, max 64 entries)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL LOADING (lazy singleton)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_yolo_session       = None
_tflite_interpreter = None


def get_yolo_model():
    """Load YOLOv8 ONNX model using onnxruntime."""
    global _yolo_session
    if _yolo_session is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO ONNX model not found at {YOLO_MODEL_PATH}\n"
                "Please export your YOLOv8 model to ONNX on your PC first:\n"
                "  from ultralytics import YOLO\n"
                "  model = YOLO('yolov8n.pt')\n"
                "  model.export(format='onnx', imgsz=320)"
            )
        _yolo_session = ort.InferenceSession(
            YOLO_MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        print("YOLOv8 ONNX ready (onnxruntime)")
    return _yolo_session


def get_tflite_model():
    """Load TFLite sign classification model."""
    global _tflite_interpreter
    if _tflite_interpreter is None:
        if not os.path.exists(TFLITE_MODEL_PATH):
            raise FileNotFoundError(
                f"TFLite model not found at {TFLITE_MODEL_PATH}\n"
                "Please convert your .h5 model to .tflite on your PC first."
            )
        _tflite_interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        _tflite_interpreter.allocate_tensors()
        print("TFLite CNN ready")
    return _tflite_interpreter


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# YOLO ONNX POST-PROCESSING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _xywh_to_xyxy(x, y, w, h):
    """Convert center-x, center-y, width, height to x1,y1,x2,y2."""
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def _nms(boxes, scores, iou_threshold=0.45):
    """Simple Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    boxes  = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas   = (x2 - x1) * (y2 - y1)
    order   = scores.argsort()[::-1]
    keep    = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask  = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def run_yolo_onnx(frame, confidence=0.25):
    """
    Run YOLOv8 ONNX inference and return list of detections.
    Each detection: {'bbox': [x1,y1,x2,y2], 'cls_id': int,
                     'class_name': str, 'conf': float}
    """
    session    = get_yolo_model()
    input_info = session.get_inputs()[0]
    input_name = input_info.name

    # Get expected input size
    _, _, inp_h, inp_w = input_info.shape
    if isinstance(inp_h, str):
        inp_h, inp_w = 320, 320

    h_orig, w_orig = frame.shape[:2]

    # Preprocess: resize, normalize, transpose to NCHW
    img     = cv2.resize(frame, (inp_w, inp_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob    = img_rgb.astype(np.float32) / 255.0
    blob    = np.transpose(blob, (2, 0, 1))       # HWC â†’ CHW
    blob    = np.expand_dims(blob, axis=0)         # add batch dim

    # Run inference
    outputs = session.run(None, {input_name: blob})
    preds   = outputs[0]  # shape: (1, 84, N) for YOLOv8

    # YOLOv8 output format: (1, 84, num_boxes) â†’ transpose to (num_boxes, 84)
    if preds.shape[1] == 84:
        preds = np.transpose(preds[0])  # (N, 84)
    elif preds.shape[2] == 84:
        preds = preds[0]
    else:
        # Try old format (1, N, 85) with objectness score
        preds = preds[0]

    x_scale = w_orig / inp_w
    y_scale = h_orig / inp_h

    all_boxes  = []
    all_scores = []
    all_cls    = []

    num_classes = preds.shape[1] - 4  # first 4 are bbox

    for det in preds:
        # YOLOv8: no objectness score, just class scores
        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        class_scores   = det[4:]

        cls_id   = int(np.argmax(class_scores))
        cls_conf = float(class_scores[cls_id])

        if cls_conf < confidence:
            continue

        # Convert to original image coordinates
        x1, y1, x2, y2 = _xywh_to_xyxy(cx, cy, bw, bh)
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)

        # Clamp
        x1 = max(0, min(x1, w_orig))
        y1 = max(0, min(y1, h_orig))
        x2 = max(0, min(x2, w_orig))
        y2 = max(0, min(y2, h_orig))

        if x2 <= x1 or y2 <= y1:
            continue

        all_boxes.append([x1, y1, x2, y2])
        all_scores.append(cls_conf)
        all_cls.append(cls_id)

    # Apply NMS
    keep = _nms(all_boxes, all_scores, iou_threshold=0.45)

    results = []
    for i in keep:
        cls_id     = all_cls[i]
        class_name = COCO_NAMES.get(cls_id, f'class_{cls_id}')
        results.append({
            'bbox':       all_boxes[i],
            'cls_id':     cls_id,
            'class_name': class_name,
            'conf':       all_scores[i],
        })

    return results


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DETECTION HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _crop_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def classify_sign_crop(bgr_crop: np.ndarray) -> str:
    """Classify a road sign crop using TFLite model."""
    pil = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
    pil = pil.resize((30, 30))
    arr = np.array(pil, dtype=np.float32)

    key = _crop_hash(arr.astype(np.uint8))
    hit = _sign_cache.get(key)
    if hit is not None:
        return hit

    try:
        interpreter    = get_tflite_model()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        batch = np.expand_dims(arr, axis=0)

        # Check if model expects uint8 or float32
        if input_details[0]['dtype'] == np.uint8:
            batch = batch.astype(np.uint8)
        else:
            # Normalize if float model
            if batch.max() > 1.0:
                batch = batch / 255.0

        interpreter.set_tensor(input_details[0]['index'], batch)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        pred   = int(np.argmax(output, axis=1)[0])
        label  = SIGN_CLASSES.get(pred + 1, "Unknown sign")

    except Exception as e:
        print(f"TFLite inference error: {e}")
        label = "Unknown sign"

    _sign_cache.put(key, label)
    return label


def get_traffic_light_color(image: np.ndarray, bbox) -> str:
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


def draw_box(frame, bbox, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), font, fs,
                (255, 255, 255), ft, cv2.LINE_AA)


def process_frame(frame, confidence=CONFIDENCE_THRESHOLD):
    """Process a single frame through YOLO + TFLite pipeline."""
    results = run_yolo_onnx(frame, confidence)

    detections = {
        'traffic_lights': [],
        'stop_signs':     [],
        'vehicles':       [],
        'persons':        0
    }

    for det in results:
        cls_id     = det['cls_id']
        conf       = det['conf']
        bbox       = det['bbox']
        class_name = det['class_name']

        if cls_id == TRAFFIC_LIGHT_CLASS:
            x1, y1, x2, y2 = bbox
            w_box = x2 - x1
            h_box = y2 - y1
            aspect_ratio = (w_box / h_box) if h_box > 0 else 0
            if aspect_ratio > 0.6:
                continue
            color     = get_traffic_light_color(frame, bbox)
            box_color = DETECTION_COLORS.get(color, DETECTION_COLORS['unknown'])
            draw_box(frame, bbox,
                     f"TL: {color.upper()} {conf:.0%}", box_color)
            detections['traffic_lights'].append(
                {'color': color, 'conf': conf, 'bbox': bbox})

        elif cls_id == STOP_SIGN_CLASS:
            x1, y1, x2, y2 = bbox
            crop       = frame[max(0, y1):y2, max(0, x1):x2]
            sign_label = classify_sign_crop(crop) if crop.size > 0 else "Stop sign"
            draw_box(frame, bbox,
                     f"{sign_label[:28]} {conf:.0%}",
                     DETECTION_COLORS['stop_sign'])
            detections['stop_signs'].append(
                {'label': sign_label, 'conf': conf, 'bbox': bbox})

        elif class_name in VEHICLE_CLASSES:
            draw_box(frame, bbox,
                     f"{class_name.upper()} {conf:.0%}",
                     DETECTION_COLORS['vehicle'])
            detections['vehicles'].append(
                {'type': class_name, 'conf': conf, 'bbox': bbox})

        elif class_name == 'person':
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          DETECTION_COLORS['person'], 2)
            detections['persons'] += 1

    _draw_hud(frame, detections)
    return frame, detections


def _draw_hud(frame, detections):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    tl = len(detections['traffic_lights'])
    st = len(detections['stop_signs'])
    ve = len(detections['vehicles'])
    pe = detections['persons']
    hud = (f"Traffic Lights: {tl}  |  Signs: {st}  |  "
           f"Vehicles: {ve}  |  Persons: {pe}")
    cv2.putText(frame, hud, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# BACKGROUND WORKER THREAD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

FRAME_SKIP = 2


class DetectionWorker(threading.Thread):
    def __init__(self, source, confidence,
                 on_frame, on_status, on_video_ended, on_finished):
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
                f"Image processed - TL: {len(dets['traffic_lights'])}  "
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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN GUI WINDOW (Tkinter)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_DISPLAY_INTERVAL_MS = 33
_SIDEBAR_INTERVAL_MS = 125

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

    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Detection System v3.4 (Pi Edition)")
        self.root.geometry("1280x780")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._worker         = None
        self._last_frame     = None
        self._confidence     = CONFIDENCE_THRESHOLD
        self._last_display_t = 0.0
        self._last_sidebar_t = 0.0
        self._models_ready   = False

        self._fps_frame_count  = 0
        self._fps_window_start = time.monotonic()
        self._current_fps      = 0.0

        self._pending_frame = None
        self._pending_dets  = None
        self._frame_lock    = threading.Lock()

        self._build_ui()
        self._preload_models()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(33, self._poll_frame)

    def _build_ui(self):
        tk.Label(self.root,
                 text="Advanced Traffic Detection System - Dual Model",
                 bg=BG, fg=GOLD,
                 font=("Georgia", 18, "bold"),
                 pady=8).pack(fill="x")

        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=8, pady=4)

        canvas_frame = tk.Frame(content, bg="#1a1a2e",
                                bd=2, relief="flat",
                                highlightbackground=SEP,
                                highlightthickness=2)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.display_label = tk.Label(
            canvas_frame,
            text="Select an image, video or start the live camera.",
            bg="#1a1a2e", fg=DIM,
            width=DISPLAY_W, height=DISPLAY_H,
            anchor="center")
        self.display_label.pack()
        self.display_label.config(width=DISPLAY_W, height=DISPLAY_H)

        self._build_sidebar(content)
        self._build_controls()

        self.status_var = tk.StringVar(value="Loading models...")
        tk.Label(self.root,
                 textvariable=self.status_var,
                 bg=BG, fg="#aaaaaa",
                 font=("Helvetica", 10),
                 anchor="w", padx=8, pady=4).pack(fill="x", side="bottom")

    def _build_sidebar(self, parent):
        sidebar = tk.LabelFrame(parent,
                                text="Detection Log",
                                bg=BG2, fg=GOLD,
                                font=("Georgia", 10, "bold"),
                                bd=1, relief="groove",
                                labelanchor="nw",
                                padx=6, pady=6)
        sidebar.pack(side="right", fill="y")

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
            var = tk.StringVar(value="-")
            tk.Label(sidebar, textvariable=var,
                     bg=BG2, fg=TEXT,
                     font=("Helvetica", 10),
                     anchor="w", justify="left",
                     wraplength=200).pack(fill="x", padx=10)
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

        self.btn_image = make_btn(btn_row, "Image",       self._open_image)
        self.btn_video = make_btn(btn_row, "Video",       self._open_video)
        self.btn_cam   = make_btn(btn_row, "Live Camera", self._start_camera)
        self.btn_stop  = make_btn(btn_row, "Stop",        self._stop_worker,
                                  bg=BTN_STOP, fg="white")
        self.btn_save  = make_btn(btn_row, "Save Frame",  self._save_frame)

        conf_row = tk.Frame(ctrl_frame, bg=BG2)
        conf_row.pack(fill="x")

        tk.Label(conf_row, text="Confidence threshold:",
                 bg=BG2, fg=TEXT,
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 6))

        self.conf_var = tk.IntVar(value=int(self._confidence * 100))
        ttk.Scale(conf_row, from_=10, to=90,
                  orient="horizontal",
                  variable=self.conf_var,
                  command=self._update_confidence,
                  length=300).pack(side="left")

        self.conf_lbl_var = tk.StringVar(value=f"{int(self._confidence * 100)}%")
        tk.Label(conf_row, textvariable=self.conf_lbl_var,
                 bg=BG2, fg=GOLD,
                 font=("Helvetica", 10, "bold"),
                 width=5).pack(side="left", padx=6)

        self._set_buttons_enabled(False)

    def _preload_models(self):
        def _load():
            self._set_status("Loading YOLOv8 ONNX model...")
            try:
                get_yolo_model()
            except Exception as e:
                self._set_status(f"YOLOv8 load failed: {e}")
                return

            self._set_status("Loading TFLite CNN model...")
            try:
                get_tflite_model()
            except Exception as e:
                self._set_status(f"TFLite CNN load failed: {e}")
                return

            self._set_status("All models ready - select an image, video or camera.")
            self._models_ready = True
            self.root.after(0, lambda: self._set_buttons_enabled(True))

        threading.Thread(target=_load, daemon=True).start()

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        bg    = BTN_BG   if enabled else "#555555"
        fg    = BTN_FG   if enabled else "#888888"
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

    def _on_frame_from_thread(self, frame, dets):
        with self._frame_lock:
            self._pending_frame = frame
            self._pending_dets  = dets

    def _poll_frame(self):
        with self._frame_lock:
            frame = self._pending_frame
            dets  = self._pending_dets
            self._pending_frame = None
            self._pending_dets  = None

        if frame is not None:
            now = time.monotonic()
            self._last_frame = frame

            self._fps_frame_count += 1
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self._current_fps      = self._fps_frame_count / elapsed
                self._fps_frame_count  = 0
                self._fps_window_start = now
                self.root.title(
                    f"Traffic Detection System v3.4  -  "
                    f"{self._current_fps:.1f} FPS")

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

    def _show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((DISPLAY_W, DISPLAY_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.display_label.config(image=photo, text="")
        self.display_label.image = photo

    def _update_sidebar(self, dets):
        tls = dets.get('traffic_lights', [])
        self.log_vars['tl_label'].set(
            f"{len(tls)} detected\n" +
            ", ".join(t['color'].upper() for t in tls)
            if tls else "None")

        stops = dets.get('stop_signs', [])
        self.log_vars['st_label'].set(
            "\n".join(f"- {s['label'][:30]} ({s['conf']:.0%})"
                      for s in stops)
            if stops else "None")

        vehicles = dets.get('vehicles', [])
        if vehicles:
            counts = {}
            for v in vehicles:
                counts[v['type']] = counts.get(v['type'], 0) + 1
            self.log_vars['ve_label'].set(
                f"{len(vehicles)} total\n" +
                "  ".join(f"{k}:{n}" for k, n in counts.items()))
        else:
            self.log_vars['ve_label'].set("None")

        pe = dets.get('persons', 0)
        self.log_vars['pe_label'].set(str(pe) if pe else "None")

    def _on_close(self):
        self._stop_worker()
        self.root.destroy()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENTRY POINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == "__main__":
    root = tk.Tk()
    app  = MainWindow(root)
    root.mainloop()
