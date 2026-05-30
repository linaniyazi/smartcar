import os
# FIX: Force XCB to avoid display errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import time
import threading
import onnxruntime as ort
import sys
import serial
import re

# =============================================================================
# 1. Configuration
# =============================================================================

YOLO_ONNX_PATH       = '/home/smartcar/coral_env/models_/yolov8n.onnx'
YOLO_SIGNS_ONNX_PATH = '/home/smartcar/coral_env/models_/best.onnx' 
CONF_THRESHOLD       = 0.25
IOU_THRESHOLD        = 0.45
INPUT_SIZE           = 320
 
ARDUINO_PORT    = '/dev/ttyACM0'
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

# --- إدارة التحكم والمؤقتات الذكية لتتوافق مع State Machine للأردوينو ---
current_held_cmd = "GO"
last_cmd_time = 0

TURN_DURATION = 3.0       # فترة الأمان لمنع بايثون من إرسال GO وتخريب التفاف الأردوينو
SPEED_DURATION = 15.0     # مدة الاحتفاظ بأوامر السرعة المستلمة

active_speed_cmd = None   # لحفظ سرعة الشاخصة الحالية
speed_start_time = 0      # مؤقت الـ 15 ثانية للسرعة

def get_decision_v5(detected_list):
    final_cmd = "GO"
    for obj in detected_list:
        name = obj["name"]
        
        if name == "TL_UNKNOWN":
            continue

        stop_conditions = [
            "person", "car", "truck", "bus", "motorcycle", "stop sign", "Stop", 
            "TL_RED", "No entry", "Yield", "Road work"
        ]
        if any(x in name for x in stop_conditions):
            return "STOP"

        slow_conditions = [
            "TL_YELLOW", "Pedestrians", "Children crossing", 
            "Slippery road", "Bumpy road"
        ]
        if any(x in name for x in slow_conditions):
            return "SLOW"

        if "Speed limit" in name:
            speed_match = re.findall(r'\d+', name)
            if speed_match:
               return f"SPEED_{speed_match[0]}"
        
        if "Turn left" in name: 
            return "TURN_LEFT"
            
        if "Turn right" in name: 
            return "TURN_RIGHT"

    return final_cmd

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

# =============================================================================
# 2. Video Capture Threading
# =============================================================================
class VideoCaptureThread:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        
        time.sleep(1)
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

# =============================================================================
# 3. Model Loading
# =============================================================================
_yolo_session = None
_yolo_signs_session = None

def get_yolo_session():
    global _yolo_session
    if _yolo_session is None:
        print("Loading YOLOv8n ONNX model...")
        _yolo_session = ort.InferenceSession(
            YOLO_ONNX_PATH, providers=['CPUExecutionProvider']
        )
    return _yolo_session

def get_yolo_signs_session():
    global _yolo_signs_session
    if _yolo_signs_session is None:
        print("Loading YOLO Signs ONNX model...")
        _yolo_signs_session = ort.InferenceSession(
            YOLO_SIGNS_ONNX_PATH, providers=['CPUExecutionProvider']
        )
    return _yolo_signs_session

# =============================================================================
# 4. Helper Functions
# =============================================================================
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

def get_traffic_light_color_v8(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0: 
        return "unknown"
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_roi)
    
    if max_val < 160:
        return "unknown"
        
    roi_h = roi.shape[0]
    relative_y = max_loc[1] / roi_h

    if relative_y < 0.38:
        return "red"
    elif 0.38 <= relative_y <= 0.65:
        return "yellow"
    else:
        return "green"

def draw_text_with_shadow(img, text, pos, font, scale, color, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    
    print("Processing Camera Stream...")
    print("Mode: Threaded (High Speed)")
    try:
        cap_thread = VideoCaptureThread(0)
        cap_thread.start()
    except Exception as e:
        print(f"Camera Error: {e}")
        sys.exit(1)

    try:
        yolo_sess = get_yolo_session()
        yolo_input_name = yolo_sess.get_inputs()[0].name
        
        yolo_signs_sess = get_yolo_signs_session()
        yolo_signs_input_name = yolo_signs_sess.get_inputs()[0].name
        
    except Exception:
        cap_thread.stop()
        sys.exit(1)

    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (w, h))
    
    window_name = "Smart Car Detection (w.py)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = 0
    fps = 0

    try:
        while True:
            ret, frame = cap_thread.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            key = cv2.waitKey(1) & 0xFF
            orig_h, orig_w = frame.shape[:2]

            current_time = time.time()
            sec = current_time - prev_time
            prev_time = current_time
            fps = 1 / sec if sec > 0 else 0

            detected_objects_with_info = []

            # -----------------------------------------------------------------
            # الموديل الأول: YOLOv8 الأساسي
            # -----------------------------------------------------------------
            try:
                blob, scale, pad_left, pad_top = preprocess_yolo(frame)
                outputs = yolo_sess.run(None, {yolo_input_name: blob})
                boxes, confs, cls_ids = postprocess_yolo(
                    outputs, scale, pad_left, pad_top, orig_h, orig_w
                )
            except Exception:
                boxes, confs, cls_ids = [], [], []

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id not in [0, 2, 3, 5, 7, 9, 11, 13]:
                    continue 

                x1, y1, x2, y2 = bbox
                name = COCO_NAMES.get(cls_id, str(cls_id))

                if cls_id == 9: 
                    color = get_traffic_light_color_v8(frame, (x1, y1, x2, y2))
                    name = f"TL_{color.upper()}"
                    
                elif cls_id == 11: 
                    name = "Stop"
                    
                elif cls_id == 13: 
                    name = "Sign"

                detected_objects_with_info.append({"name": name})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_text_with_shadow(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # -----------------------------------------------------------------
            # الموديل الثاني: YOLO الشاخصات الإضافي (42 كلاس)
            # -----------------------------------------------------------------
            try:
                blob_s, scale_s, pad_left_s, pad_top_s = preprocess_yolo(frame)
                outputs_s = yolo_signs_sess.run(None, {yolo_signs_input_name: blob_s})
                boxes_s, confs_s, cls_ids_s = postprocess_yolo(
                    outputs_s, scale_s, pad_left_s, pad_top_s, orig_h, orig_w
                )
            except Exception:
                boxes_s, confs_s, cls_ids_s = [], [], []

            for bbox_s, conf_s, cls_id_s in zip(boxes_s, confs_s, cls_ids_s):
                x1_s, y1_s, x2_s, y2_s = bbox_s
                
                if 0 <= cls_id_s < len(SIGN_CLASSES):
                    name_s = SIGN_CLASSES[cls_id_s]

                    detected_objects_with_info.append({"name": name_s})
                    cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (255, 165, 0), 2)
                    draw_text_with_shadow(frame, name_s, (x1_s, y1_s-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                
            # -----------------------------------------------------------------
            # ميكانيكية حماية المنعطفات وإدارة سرعات الـ 15 ثانية
            # -----------------------------------------------------------------
            raw_decision = get_decision_v5(detected_objects_with_info)
            
            # 1. فلترة وتحديث مؤقت شاخصات السرعة (تستمر 15 ثانية)
            if "SPEED_" in raw_decision:
                active_speed_cmd = raw_decision
                speed_start_time = current_time
            elif active_speed_cmd and (current_time - speed_start_time >= SPEED_DURATION):
                active_speed_cmd = None # انتهاء الـ 15 ثانية والعودة للوضع الطبيعي

            # 2. حماية الالتفاف (Turn Lock):
            # إذا أرسل بايثون سابقاً أمر انعطاف، يدخل بقفل صامت لمدة 3 ثوانٍ؛ 
            # يمنع إرسال أمر GO لترك الأردوينو يكمل اللفة كاملة بناءً على كود الـ state machine لديه.
            if current_held_cmd in ["TURN_LEFT", "TURN_RIGHT"] and (current_time - last_cmd_time < TURN_DURATION):
                # نحن الآن في مرحلة "التثبيت الصامت للمنعطف"؛ نمنع أي تحديث يعيد السيارة لـ GO
                pass 
            
            else:
                # إذا ظهر أمر طارئ حرج (توقف، إبطاء، أو إشارة انعطاف جديدة التقطتها الكاميرا)
                if raw_decision in ["STOP", "SLOW", "TURN_LEFT", "TURN_RIGHT"]:
                    if raw_decision != current_held_cmd:
                        current_held_cmd = raw_decision
                        last_cmd_time = current_time # تصفير وقت آخر أمر لحساب قفل الـ 3 ثوانٍ
                
                # إذا كان الطريق آمناً، وكان هناك أمر سرعة مؤقت لم تنتهِ الـ 15 ثانية الخاصة به
                elif active_speed_cmd:
                    current_held_cmd = active_speed_cmd
                
                # في حال عدم وجود أي شاخصة أو مؤقتات نشطة، نعود للوضع الافتراضي الآمن
                else:
                    current_held_cmd = "GO"

            # إرسال الأمر النهائي المستقر والمحمي عبر السيريال للآردوينو
            arduino.send(current_held_cmd)
            
            # حساب الوقت المتبقي للمؤقت النشط لعرضه بدقة على الشاشة للـ Debugging
            time_left = 0.0
            if current_held_cmd in ["TURN_LEFT", "TURN_RIGHT"]:
                time_left = max(0.0, TURN_DURATION - (current_time - last_cmd_time))
            elif active_speed_cmd and current_held_cmd == active_speed_cmd:
                time_left = max(0.0, SPEED_DURATION - (current_time - speed_start_time))
            
            cmd_color = (0, 0, 255) if current_held_cmd == "STOP" else \
                        ((0, 255, 255) if current_held_cmd == "SLOW" else (0, 255, 0))

            draw_text_with_shadow(frame, f"CMD: {current_held_cmd}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2)
            draw_text_with_shadow(frame, f"Lock/Speed Timer: {time_left:.1f}s", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            draw_text_with_shadow(frame, f"FPS: {int(fps)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
             
            cv2.imshow(window_name, frame)
            if out:
                out.write(frame)

            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap_thread.stop()
        if out: out.release()
        cv2.destroyAllWindows()
        print("Done.")
