import os
# FIX: Force XCB to avoid display errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import time
import threading
from PIL import Image
import onnxruntime as ort
import tflite_runtime.interpreter as tflite
import sys
import serial
import re

# ═════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═════════════════════════════════════════════════════════════════════════════
YOLO_ONNX_PATH  = '/home/smartcar/newenv/smartcar-main/yolov8n.onnx'
TFLITE_PATH     = '/home/smartcar/newenv/smartcar-main/model_RTSR.tflite'
CONF_THRESHOLD  = 0.25
IOU_THRESHOLD   = 0.45
INPUT_SIZE      = 320

ARDUINO_PORT    = '/dev/ttyUSB0'
BAUD_RATE       = 115200
class ArduinoBridge:
    def __init__(self):
        try:
            self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
            print("✅ Arduino Connected")
        except:
            self.ser = None
            print("⚠️ Arduino Not Found")

    def send(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(f"{cmd}\n".encode())

arduino = ArduinoBridge()

# ═════════════════════════════════════════════════════════════════════════════
# 2. طبقة اتخاذ القرار (Decision Logic)
# ═════════════════════════════════════════════════════════════════════════════
def get_decision(detected_names):
    """
    تحليل المكتشفات بناءً على الأولويات:
    1. مشاة (person) -> STOP
    2. إشارة حمراء -> STOP
    3. شاخصات التوقف والمنع -> STOP
    4. شاخصات السرعة والحذر -> SPEED_X / SLOW
    """
    # الأولوية 1: الأمان
    if 'person' in detected_names: return "STOP"
    if 'TL: RED' in detected_names: return "STOP"
    
    # الأولوية 2: الشاخصات الحرجة
    critical_signs = ["Stop", "No entry", "Yield", "Road work"]
    for name in detected_names:
        if any(s in name for s in critical_signs):
            return "STOP"

    # الأولوية 3: تنظيم السرعة والاتجاه
    for name in detected_names:
        if "Speed limit" in name:
            speed = re.findall(r'\d+', name)
            if speed: return f"SPEED_{speed[0]}"
        
        if "left" in name.lower(): return "TURN_LEFT"
        if "right" in name.lower(): return "TURN_RIGHT"
        if "Yellow" in name: return "SLOW"
        if any(x in name.lower() for x in ["caution", "pedestrians", "bumpy"]):
            return "SLOW"

    # الحالة الافتراضية
    return "GO"
# Traffic Sign Classes
SIGN_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing veh > 3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5t prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End no passing veh over 3.5t"
]

COCO_NAMES = {
    i: name for i, name in enumerate([
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"
    ])
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. Video Capture Threading (Used ONLY for Camera)
# ═════════════════════════════════════════════════════════════════════════════
class VideoCaptureThread:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()

# ═════════════════════════════════════════════════════════════════════════════
# 3. Model Loading
# ═════════════════════════════════════════════════════════════════════════════
_yolo_session = None
_tflite_interp = None

def get_yolo_session():
    global _yolo_session
    if _yolo_session is None:
        print("Loading YOLOv8n ONNX model...")
        _yolo_session = ort.InferenceSession(
            YOLO_ONNX_PATH, providers=['CPUExecutionProvider']
        )
    return _yolo_session

def get_tflite_interpreter():
    global _tflite_interp
    if _tflite_interp is None:
        print("Loading TFLite classifier model...")
        _tflite_interp = tflite.Interpreter(model_path=TFLITE_PATH)
        _tflite_interp.allocate_tensors()
    return _tflite_interp

# ═════════════════════════════════════════════════════════════════════════════
# 4. Helper Functions
# ═════════════════════════════════════════════════════════════════════════════
def preprocess_yolo(frame, input_size=INPUT_SIZE):
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_top  = (input_size - nh) // 2
    pad_left = (input_size - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    return blob, scale, pad_left, pad_top

def postprocess_yolo(outputs, scale, pad_left, pad_top, orig_h, orig_w):
    preds = outputs[0][0].T
    boxes_xywh = preds[:, :4]
    scores     = preds[:, 4:]
    class_ids  = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    
    mask = confidences >= CONF_THRESHOLD
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]
    
    if len(boxes_xywh) == 0:
        return [], [], []
    
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    
    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h)
    
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(int)
    
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(),
        CONF_THRESHOLD, IOU_THRESHOLD
    )
    if len(indices) == 0:
        return [], [], []
    indices = indices.flatten()
    return boxes_xyxy[indices], confidences[indices], class_ids[indices]

def get_traffic_light_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0: return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ranges = {
        'red':    [(np.array([0, 70, 50]), np.array([10, 255, 255])),
                   (np.array([170,70, 50]), np.array([180, 255, 255]))],
        'yellow': [(np.array([15, 70, 50]), np.array([35, 255, 255]))],
        'green':  [(np.array([36, 70, 50]), np.array([89, 255, 255]))],
    }
    scores = {color: 0 for color in ranges}
    for color, r_list in ranges.items():
        for lo, hi in r_list:
            mask = cv2.inRange(hsv, lo, hi)
            scores[color] += cv2.countNonZero(mask)
    return max(scores, key=scores.get)

def classify_sign_crop(crop_bgr):
    try:
        interp = get_tflite_interpreter()
        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        img = cv2.resize(crop_bgr, (30, 30))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        if in_det['dtype'] == np.uint8:
            scale, zero_point = in_det['quantization']
            img = (img / scale + zero_point).astype(np.uint8)
        interp.set_tensor(in_det['index'], img)
        interp.invoke()
        preds = interp.get_tensor(out_det['index'])[0]
        idx   = int(np.argmax(preds))
        return SIGN_CLASSES[idx]
    except Exception:
        return "Sign"

# ═════════════════════════════════════════════════════════════════════════════
# 5. Main Execution (Handles File vs Camera logic)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    video_path = r"/home/smartcar/newenv/smartcar-main/VID_20260330121826.mp4"
    
    # --- SETUP ---
    is_file = isinstance(video_path, str) and os.path.exists(video_path)
    
    if is_file:
        print(f"Processing Video File: {video_path}")
        print("Mode: Sequential (Processing every frame at real-time speed)")
        cap = cv2.VideoCapture(video_path)
        # Get video properties to maintain speed
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / video_fps) if video_fps > 0 else 30
        print(f"Video FPS: {video_fps}")
    else:
        print("Processing Camera Stream...")
        print("Mode: Threaded (High Speed)")
        try:
            cap_thread = VideoCaptureThread(0)
            cap_thread.start()
        except Exception as e:
            print(f"Camera Error: {e}")
            sys.exit(1)

    # Load Models
    try:
        yolo_sess = get_yolo_session()
        yolo_input_name = yolo_sess.get_inputs()[0].name
        _ = get_tflite_interpreter()
    except Exception:
        if is_file: cap.release()
        else: cap_thread.stop()
        sys.exit(1)

    # Setup Writer
    out = None
    if is_file:
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        # Assume default for camera
        h, w = 480, 640
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (w, h))
    
    window_name = "Smart Car Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("System Running. Press 'q' to quit.")

    try:
        while True:
            if is_file:
                # --- FILE MODE: Synchronous Read ---
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break
                detected_this_frame = []
                # Calculate FPS
                # cv2.waitKey(delay) pauses the loop to match video FPS
                key = cv2.waitKey(delay) & 0xFF 
                
            else:
                # --- CAMERA MODE: Threaded Read ---
                ret, frame = cap_thread.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                key = cv2.waitKey(1) & 0xFF

            orig_h, orig_w = frame.shape[:2]

            # Inference
            try:
                blob, scale, pad_left, pad_top = preprocess_yolo(frame)
                outputs = yolo_sess.run(None, {yolo_input_name: blob})
                boxes, confs, cls_ids = postprocess_yolo(
                    outputs, scale, pad_left, pad_top, orig_h, orig_w
                )
            except Exception:
                boxes, confs, cls_ids = [], [], []

            # Drawing
            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                name = COCO_NAMES.get(cls_id, str(cls_id))
                detected_this_frame.append(name)

                if cls_id == 9:
                    color = get_traffic_light_color(frame, (x1, y1, x2, y2))
                    name = f"TL: {color.upper()}"
                elif cls_id in [11, 13]:
                    if (x2-x1)*(y2-y1) > 1000:
                        crop = frame[y1:y2, x1:x2]
                        name = classify_sign_crop(crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
            final_command = get_decision_v3(detected_this_frame)
            arduino.send(final_command)
            cv2.imshow(window_name, frame)
            if out:
                out.write(frame)

            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        if is_file:
            cap.release()
        else:
            cap_thread.stop()
            
        if out: out.release()
        cv2.destroyAllWindows()
        print("Done. Check 'output_video.mp4'")