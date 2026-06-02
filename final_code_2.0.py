import os
# FIX: Force XCB to avoid display errors
os.environ["QT_QPA_PLATFORM"] = "xcb" [cite: 1]

import cv2 [cite: 1]
import numpy as np [cite: 1]
import time [cite: 1]
import threading [cite: 1]
import onnxruntime as ort [cite: 1]
import sys [cite: 1]
import serial [cite: 1]
import re [cite: 1]

# =============================================================================
# 1. Configuration
# =============================================================================

YOLO_ONNX_PATH       = '/home/smartcar/coral_env/models_/yolov8n.onnx' [cite: 1]
YOLO_SIGNS_ONNX_PATH = '/home/smartcar/coral_env/models_/best.onnx' [cite: 1]
CONF_THRESHOLD       = 0.25 [cite: 1]
IOU_THRESHOLD        = 0.45 [cite: 1]
INPUT_SIZE           = 640  # 💡 تم التعديل هنا إلى 640 ليتناسب مع الموديل الجديد
 
ARDUINO_PORT    = '/dev/ttyACM0' [cite: 1]
BAUD_RATE       = 115200 [cite: 1]

class ArduinoBridge: [cite: 1]
    def __init__(self): [cite: 1]
        try: [cite: 1]
            self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1) [cite: 2]
            print("✅ Arduino Connected") [cite: 2]
        except: [cite: 2]
            self.ser = None [cite: 2]
            print("⚠️ Arduino Not Found") [cite: 2]

    def send(self, cmd): [cite: 2]
        if self.ser and self.ser.is_open: [cite: 2]
            self.ser.write(f"{cmd}\n".encode()) [cite: 2]

arduino = ArduinoBridge() [cite: 2]

# --- إدارة التحكم والمؤقتات الذكية لتتوافق مع State Machine للأردوينو --- [cite: 2, 3]
current_held_cmd = "GO" [cite: 3]
last_cmd_time = 0 [cite: 3]

TURN_DURATION = 8.5       # فترة الأمان لمنع بايثون من إرسال GO وتخريب التفاف الأردوينو [cite: 3]
SPEED_DURATION = 15.0     # مدة الاحتفاظ بأوامر السرعة المستلمة [cite: 3]

active_speed_cmd = None   # لحفظ سرعة الشاخصة الحالية [cite: 3]
speed_start_time = 0      # مؤقت الـ 15 ثانية للسرعة [cite: 3]

def get_decision_v5(detected_list): [cite: 3]
    final_cmd = "GO" [cite: 3]
    for obj in detected_list: [cite: 3]
        name = obj["name"] [cite: 3]
        
        if name == "TL_UNKNOWN": [cite: 3, 4]
            continue [cite: 4]

        stop_conditions = [ [cite: 4]
            "person", "car", "truck", "bus", "motorcycle", "stop sign", "Stop",  [cite: 4]
            "TL_RED", "No entry", "Yield", "Road work" [cite: 4]
        ] [cite: 4]
        if any(x in name for x in stop_conditions): [cite: 4]
            return "STOP" [cite: 4]

        slow_conditions = [ [cite: 4, 5]
            "TL_YELLOW", "Pedestrians", "Children crossing",  [cite: 5]
            "Slippery road", "Bumpy road" [cite: 5]
        ] [cite: 5]
        if any(x in name for x in slow_conditions): [cite: 5]
            return "SLOW" [cite: 5]

        if "Speed limit" in name: [cite: 5]
            speed_match = re.findall(r'\d+', name) [cite: 5, 6]
            if speed_match: [cite: 6]
               return f"SPEED_{speed_match[0]}" [cite: 6]
        
        if "Turn left" in name:  [cite: 6]
            return "TURN_LEFT" [cite: 6]
            
        if "Turn right" in name:  [cite: 6]
            return "TURN_RIGHT" [cite: 6, 7]

    return final_cmd [cite: 7]

SIGN_CLASSES = [ [cite: 7]
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", [cite: 7]
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", [cite: 7]
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", [cite: 7]
    "No passing", "No passing veh > 3.5t", "Right-of-way at intersection", [cite: 7]
    "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5t prohibited", [cite: 7]
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right", [cite: 7]
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", [cite: 7, 8]
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", [cite: 8]
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", [cite: 8]
    "End speed + passing limits", "Turn right ahead", "Turn left ahead", [cite: 8]
    "Ahead only", "Go straight or right", "Go straight or left", [cite: 8]
    "Keep right", "Keep left", "Roundabout mandatory", [cite: 8]
    "End of no passing", "End no passing veh over 3.5t" [cite: 8]
] [cite: 8]

COCO_NAMES = { [cite: 8]
    i: name for i, name in enumerate([ [cite: 8]
        "person","bicycle","car","motorcycle","airplane","bus","train","truck", [cite: 8]
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench", [cite: 8, 9]
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe", [cite: 9]
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard", [cite: 9]
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard", [cite: 9]
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl", [cite: 9]
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza", [cite: 9]
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet", [cite: 9]
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven", [cite: 9]
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear", [cite: 9]
        "hair drier","toothbrush" [cite: 9]
    ]) [cite: 9]
} [cite: 9]

# =============================================================================
# 2. Video Capture Threading
# =============================================================================
class VideoCaptureThread: [cite: 9]
    def __init__(self, source=0): [cite: 9, 10]
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2) [cite: 10]
        if not self.cap.isOpened(): [cite: 10]
            raise IOError(f"Cannot open video source: {source}") [cite: 10]
        
        time.sleep(1) [cite: 10]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) [cite: 10]
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) [cite: 10]
        
        self.grabbed, self.frame = self.cap.read() [cite: 10]
        self.started = False [cite: 10, 11]
        self.read_lock = threading.Lock() [cite: 11]

    def start(self): [cite: 11]
        if self.started: [cite: 11]
            return [cite: 11]
        self.started = True [cite: 11]
        self.thread = threading.Thread(target=self.update, daemon=True) [cite: 11]
        self.thread.start() [cite: 11]

    def update(self): [cite: 11]
        while self.started: [cite: 11]
            grabbed, frame = self.cap.read() [cite: 11, 12]
            with self.read_lock: [cite: 12]
                self.grabbed = grabbed [cite: 12]
                self.frame = frame [cite: 12]

    def read(self): [cite: 12]
        with self.read_lock: [cite: 12]
            return self.grabbed, self.frame [cite: 12]

    def stop(self): [cite: 12]
        self.started = False [cite: 12]
        if hasattr(self, 'thread'): [cite: 12, 13]
            self.thread.join() [cite: 13]
        self.cap.release() [cite: 13]

# =============================================================================
# 3. Model Loading
# =============================================================================
_yolo_session = None [cite: 13]
_yolo_signs_session = None [cite: 13]

def get_yolo_session(): [cite: 13]
    global _yolo_session [cite: 13]
    if _yolo_session is None: [cite: 13]
        print("Loading YOLOv8n ONNX model...") [cite: 13]
        _yolo_session = ort.InferenceSession( [cite: 13]
            YOLO_ONNX_PATH, providers=['CPUExecutionProvider'] [cite: 13]
        ) [cite: 13]
    return _yolo_session [cite: 13]

def get_yolo_signs_session(): [cite: 13]
    global _yolo_signs_session [cite: 13]
    if _yolo_signs_session is None: [cite: 13, 14]
        print("Loading YOLO Signs ONNX model...") [cite: 14]
        _yolo_signs_session = ort.InferenceSession( [cite: 14]
            YOLO_SIGNS_ONNX_PATH, providers=['CPUExecutionProvider'] [cite: 14]
        ) [cite: 14]
    return _yolo_signs_session [cite: 14]

# =============================================================================
# 4. Helper Functions
# =============================================================================
def preprocess_yolo(frame, input_size=INPUT_SIZE): [cite: 14]
    h, w = frame.shape[:2] [cite: 14]
    scale = input_size / max(h, w) [cite: 14]
    nh, nw = int(h * scale), int(w * scale) [cite: 14]
    resized = cv2.resize(frame, (nw, nh)) [cite: 14]
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8) [cite: 14, 15]
    pad_top  = (input_size - nh) // 2 [cite: 15]
    pad_left = (input_size - nw) // 2 [cite: 15]
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized [cite: 15]
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0 [cite: 15]
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis] [cite: 15]
    return blob, scale, pad_left, pad_top [cite: 15]

def postprocess_yolo(outputs, scale, pad_left, pad_top, orig_h, orig_w): [cite: 15]
    preds = outputs[0][0].T [cite: 15]
    boxes_xywh = preds[:, :4] [cite: 15]
    scores     = preds[:, 4:] [cite: 15]
    class_ids  = np.argmax(scores, axis=1) [cite: 15]
    confidences = scores[np.arange(len(scores)), class_ids] [cite: 16]
    
    mask = confidences >= CONF_THRESHOLD [cite: 16]
    boxes_xywh  = boxes_xywh[mask] [cite: 16]
    confidences = confidences[mask] [cite: 16]
    class_ids   = class_ids[mask] [cite: 16]
    
    if len(boxes_xywh) == 0: [cite: 16]
        return [], [], [] [cite: 16]
    
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3] [cite: 16]
    x1 = cx - bw / 2 [cite: 16]
    y1 = cy - bh / 2 [cite: 16]
    x2 = cx + bw / 2 [cite: 16, 17]
    y2 = cy + bh / 2 [cite: 17]
    
    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w) [cite: 17]
    y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h) [cite: 17]
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w) [cite: 17]
    y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h) [cite: 17]
    
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(int) [cite: 17]
    
    indices = cv2.dnn.NMSBoxes( [cite: 17]
        boxes_xyxy.tolist(), confidences.tolist(), [cite: 18]
        CONF_THRESHOLD, IOU_THRESHOLD [cite: 18]
    ) [cite: 18]
    if len(indices) == 0: [cite: 18]
        return [], [], [] [cite: 18]
    indices = indices.flatten() [cite: 18]
    return boxes_xyxy[indices], confidences[indices], class_ids[indices] [cite: 18]

def get_traffic_light_color_v8(image, bbox): [cite: 18]
    x1, y1, x2, y2 = map(int, bbox) [cite: 18]
    roi = image[max(0, y1):y2, max(0, x1):x2] [cite: 18]
    if roi.size == 0:  [cite: 18]
        return "unknown" [cite: 18]
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) [cite: 18]
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0) [cite: 19]
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_roi) [cite: 19]
    
    if max_val < 160: [cite: 19]
        return "unknown" [cite: 19]
        
    roi_h = roi.shape[0] [cite: 19]
    relative_y = max_loc[1] / roi_h [cite: 19]

    if relative_y < 0.38: [cite: 19]
        return "red" [cite: 19]
    elif 0.38 <= relative_y <= 0.65: [cite: 19]
        return "yellow" [cite: 19]
    else: [cite: 19, 20]
        return "green" [cite: 20]

def draw_text_with_shadow(img, text, pos, font, scale, color, thickness=2): [cite: 20]
    x, y = pos [cite: 20]
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA) [cite: 20]
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA) [cite: 20]

# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__": [cite: 20]
    
    print("Processing Camera Stream...") [cite: 20]
    print("Mode: Threaded (High Speed)") [cite: 20]
    try: [cite: 20]
        cap_thread = VideoCaptureThread(0) [cite: 20]
        cap_thread.start() [cite: 21]
    except Exception as e: [cite: 21]
        print(f"Camera Error: {e}") [cite: 21]
        sys.exit(1) [cite: 21]

    try: [cite: 21]
        yolo_sess = get_yolo_session() [cite: 21]
        yolo_input_name = yolo_sess.get_inputs()[0].name [cite: 21]
        
        yolo_signs_sess = get_yolo_signs_session() [cite: 21]
        yolo_signs_input_name = yolo_signs_sess.get_inputs()[0].name [cite: 21]
        
    except Exception: [cite: 21]
        cap_thread.stop() [cite: 22]
        sys.exit(1) [cite: 22]

    h, w = 480, 640 [cite: 22]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') [cite: 22]
    out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (w, h)) [cite: 22]
    
    window_name = "Smart Car Detection (w.py)" [cite: 22]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) [cite: 22]

    prev_time = 0 [cite: 22]
    fps = 0 [cite: 22]

    try: [cite: 22]
        while True: [cite: 22]
            ret, frame = cap_thread.read() [cite: 22, 23]
            if not ret or frame is None: [cite: 23]
                time.sleep(0.01) [cite: 23]
                continue [cite: 23]
            
            key = cv2.waitKey(1) & 0xFF [cite: 23]
            orig_h, orig_w = frame.shape[:2] [cite: 23]

            current_time = time.time() [cite: 23]
            sec = current_time - prev_time [cite: 24]
            prev_time = current_time [cite: 24]
            fps = 1 / sec if sec > 0 else 0 [cite: 24]

            detected_objects_with_info = [] [cite: 24]

            # -----------------------------------------------------------------
            # الموديل الأول: YOLOv8 الأساسي
            # ----------------------------------------------------------------- [cite: 24, 25]
            try: [cite: 25]
                blob, scale, pad_left, pad_top = preprocess_yolo(frame) [cite: 25]
                outputs = yolo_sess.run(None, {yolo_input_name: blob}) [cite: 25]
                boxes, confs, cls_ids = postprocess_yolo( [cite: 25]
                    outputs, scale, pad_left, pad_top, orig_h, orig_w [cite: 25, 26]
                ) [cite: 26]
            except Exception: [cite: 26]
                boxes, confs, cls_ids = [], [], [] [cite: 26]

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids): [cite: 26]
                if cls_id not in [0, 2, 3, 5, 7, 9, 11, 13]: [cite: 26, 27]
                    continue  [cite: 27]

                x1, y1, x2, y2 = bbox [cite: 27]
                name = COCO_NAMES.get(cls_id, str(cls_id)) [cite: 27]

                if cls_id == 9:  [cite: 27]
                    color = get_traffic_light_color_v8(frame, (x1, y1, x2, y2)) [cite: 28]
                    name = f"TL_{color.upper()}" [cite: 28]
                    
                elif cls_id == 11:  [cite: 28]
                    name = "Stop" [cite: 28]
                    
                elif cls_id == 13:  [cite: 28, 29]
                    name = "Sign" [cite: 29]

                detected_objects_with_info.append({"name": name}) [cite: 29]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) [cite: 29]
                draw_text_with_shadow(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) [cite: 30]

            # -----------------------------------------------------------------
            # الموديل الثاني: YOLO الشاخصات الإضافي (42 كلاس)
            # ----------------------------------------------------------------- [cite: 30]
            try: [cite: 30]
                blob_s, scale_s, pad_left_s, pad_top_s = preprocess_yolo(frame) [cite: 30]
                outputs_s = yolo_signs_sess.run(None, {yolo_signs_input_name: blob_s}) [cite: 31]
                boxes_s, confs_s, cls_ids_s = postprocess_yolo( [cite: 31]
                    outputs_s, scale_s, pad_left_s, pad_top_s, orig_h, orig_w [cite: 31]
                ) [cite: 31]
            except Exception: [cite: 31]
                boxes_s, confs_s, cls_ids_s = [], [], [] [cite: 32]

            for bbox_s, conf_s, cls_id_s in zip(boxes_s, confs_s, cls_ids_s): [cite: 32]
                x1_s, y1_s, x2_s, y2_s = bbox_s [cite: 32]
                
                if 0 <= cls_id_s < len(SIGN_CLASSES): [cite: 32]
                    name_s = SIGN_CLASSES[cls_id_s] [cite: 33]

                    detected_objects_with_info.append({"name": name_s}) [cite: 33]
                    cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (255, 165, 0), 2) [cite: 33]
                    draw_text_with_shadow(frame, name_s, (x1_s, y1_s-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1) [cite: 33]
               
            # -----------------------------------------------------------------
            # ميكانيكية حماية المنعطفات وإدارة سرعات الـ 15 ثانية
            # ----------------------------------------------------------------- [cite: 33, 34]
            raw_decision = get_decision_v5(detected_objects_with_info) [cite: 34]
            
            # 1. فلترة وتحديث مؤقت شاخصات السرعة (تستمر 15 ثانية)
            if "SPEED_" in raw_decision: [cite: 34, 35]
                active_speed_cmd = raw_decision [cite: 35]
                speed_start_time = current_time [cite: 35]
            elif active_speed_cmd and (current_time - speed_start_time >= SPEED_DURATION): [cite: 35]
                active_speed_cmd = None # عودة السرعة الافتراضية [cite: 35]

            # 2. حماية الالتفاف (Turn Lock): [cite: 35, 36]
            if current_held_cmd in ["TURN_LEFT", "TURN_RIGHT"] and (current_time - last_cmd_time < TURN_DURATION): [cite: 36]
                # قفل صامت أثناء الالتفاف [cite: 36]
                pass  [cite: 36]
            
            else: [cite: 36]
                # إذا خرجنا من مرحلة المنعطف أو كنا في القيادة الطبيعية [cite: 36, 37]
                if raw_decision in ["STOP", "SLOW", "TURN_LEFT", "TURN_RIGHT"]: [cite: 37]
                    if raw_decision != current_held_cmd: [cite: 37]
                        current_held_cmd = raw_decision [cite: 37]
                    last_cmd_time = current_time  [cite: 37, 38]
                elif active_speed_cmd: [cite: 38]
                    current_held_cmd = active_speed_cmd [cite: 38]
                else: [cite: 38]
                    current_held_cmd = "GO" # العودة الفورية للمشي دون توقف لحظي خاطئ [cite: 38, 39]

            # إرسال الأمر المستقر عبر السيريال [cite: 39]
            arduino.send(current_held_cmd) [cite: 39]
            
            # حساب الوقت المتبقي للمؤقت لعرضه للـ Debugging [cite: 39]
            time_left = 0.0 [cite: 39]
            if current_held_cmd in ["TURN_LEFT", "TURN_RIGHT"]: [cite: 39]
                time_left = max(0.0, TURN_DURATION - (current_time - last_cmd_time)) [cite: 40]
            elif active_speed_cmd and current_held_cmd == active_speed_cmd: [cite: 40]
                time_left = max(0.0, SPEED_DURATION - (current_time - speed_start_time)) [cite: 40]
            
            cmd_color = (0, 0, 255) if current_held_cmd == "STOP" else \
                        ((0, 255, 255) if current_held_cmd == "SLOW" else (0, 255, 0)) [cite: 40, 41]

            draw_text_with_shadow(frame, f"CMD: {current_held_cmd}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2) [cite: 41]
            draw_text_with_shadow(frame, f"Lock/Speed Timer: {time_left:.1f}s", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) [cite: 41]
            draw_text_with_shadow(frame, f"FPS: {int(fps)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) [cite: 41]
             
            cv2.imshow(window_name, frame) [cite: 41, 42]
            if out: [cite: 42]
                out.write(frame) [cite: 42]

            if key == ord('q'): [cite: 42]
                break [cite: 42]
                
    except KeyboardInterrupt: [cite: 42]
        print("\n[INFO] Script stopped by user.") [cite: 43]
    except Exception as e: [cite: 43]
        print(f"\n[ERROR] Unexpected error: {e}") [cite: 43]
    finally: [cite: 43]
        print("[SERIAL] Sending STOP command...") [cite: 43]
        if 'arduino' in globals() and arduino.ser and arduino.ser.is_open: [cite: 43]
            arduino.send("STOP") # إيقاف فوري للسيارة عبر السيريال الأصلي للأردوينو [cite: 43]
            time.sleep(0.2)      [cite: 43]
          
        cap_thread.stop() [cite: 44]
        if out: out.release() [cite: 44]
        cv2.destroyAllWindows() [cite: 44]
        print("[STATUS] Done. Car safe.") [cite: 44, 45]
