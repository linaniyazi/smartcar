import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import time
import serial
import sys
import onnxruntime as ort

# استخدام PyCoral بدل tflite_runtime
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify

# ── Config ────────────────────────────────────────────────────────
YOLO_ONNX_PATH   = '/home/smartcar/newenv/smartcar-main/yolov8n.onnx'
TFLITE_PATH      = '/home/smartcar/newenv/smartcar-main/model_RTSR_edgetpu.tflite'
CONF_THRESHOLD   = 0.25
IOU_THRESHOLD    = 0.45
INPUT_SIZE       = 320
STOP_CLASS_ID    = 11

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

# ── Arduino ───────────────────────────────────────────────────────
def init_arduino():
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        print("Arduino connected.")
        return ser
    except Exception as e:
        print(f"Arduino error: {e}")
        return None

def send_command(arduino, cmd):
    if arduino is None:
        return
    arduino.write((cmd + '\n').encode())
    print(f"Sent: {cmd}")
    time.sleep(0.1)
    if arduino.in_waiting > 0:
        print(f"Arduino: {arduino.readline().decode().strip()}")

# ── Models ────────────────────────────────────────────────────────
def load_models():
    print("Loading YOLO (CPU)...")
    sess = ort.InferenceSession(YOLO_ONNX_PATH, providers=['CPUExecutionProvider'])

    print("Loading TFLite on Coral EdgeTPU...")
    try:
        interp = make_interpreter(TFLITE_PATH)  # يستخدم EdgeTPU تلقائياً
        interp.allocate_tensors()
        print("Coral TPU loaded successfully.")
    except Exception as e:
        # Fallback للـ CPU إذا ما كان Coral متصل
        print(f"Coral not available ({e}), falling back to CPU TFLite...")
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=TFLITE_PATH.replace('_edgetpu', ''))
        interp.allocate_tensors()
    return sess, interp

# ── YOLO inference (CPU) ──────────────────────────────────────────
def preprocess(frame):
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pt, pl = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
    canvas[pt:pt+nh, pl:pl+nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis], scale, pl, pt

def detect(sess, frame):
    blob, scale, pl, pt = preprocess(frame)
    orig_h, orig_w = frame.shape[:2]
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].T
    scores  = out[:, 4:]
    cls_ids = np.argmax(scores, axis=1)
    confs   = scores[np.arange(len(scores)), cls_ids]
    mask    = confs >= CONF_THRESHOLD
    out, cls_ids, confs = out[mask], cls_ids[mask], confs[mask]
    if len(out) == 0:
        return [], [], []
    cx, cy, bw, bh = out[:,0], out[:,1], out[:,2], out[:,3]
    x1 = np.clip((cx - bw/2 - pl) / scale, 0, orig_w).astype(int)
    y1 = np.clip((cy - bh/2 - pt) / scale, 0, orig_h).astype(int)
    x2 = np.clip((cx + bw/2 - pl) / scale, 0, orig_w).astype(int)
    y2 = np.clip((cy + bh/2 - pt) / scale, 0, orig_h).astype(int)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    if len(idx) == 0:
        return [], [], []
    return boxes[idx.flatten()], confs[idx.flatten()], cls_ids[idx.flatten()]

# ── Classifier على Coral TPU ──────────────────────────────────────
def classify_stop(interp, crop):
    img = cv2.cvtColor(cv2.resize(crop, (30, 30)), cv2.COLOR_BGR2RGB)

    # PyCoral API
    try:
        common.set_input(interp, img)
        interp.invoke()
        classes = classify.get_classes(interp, top_k=1)
        return SIGN_CLASSES[classes[0].id]
    except AttributeError:
        # Fallback: tflite_runtime عادي
        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        img_f = img.astype(np.float32) / 255.0
        img_f = np.expand_dims(img_f, 0)
        if in_det['dtype'] == np.uint8:
            s, zp = in_det['quantization']
            img_f = (img_f / s + zp).astype(np.uint8)
        interp.set_tensor(in_det['index'], img_f)
        interp.invoke()
        return SIGN_CLASSES[int(np.argmax(interp.get_tensor(out_det['index'])[0]))]

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    arduino      = init_arduino()
    sess, interp = load_models()
    car_stopped  = False
    send_command(arduino, "RUN")

    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    time.sleep(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera error: cannot open.")
        sys.exit(1)

    cv2.namedWindow("Stop Sign Detection", cv2.WINDOW_NORMAL)
    print("Running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Problem reading frame.")
                break

            boxes, confs, cls_ids = detect(sess, frame)
            stop_detected = False

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id != STOP_CLASS_ID:
                    continue
                x1, y1, x2, y2 = bbox
                if (x2-x1) * (y2-y1) < 1000:
                    continue

                crop  = frame[y1:y2, x1:x2]
                label = classify_stop(interp, crop)

                if label == "Stop" or cls_id == STOP_CLASS_ID:
                    stop_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"STOP {conf:.2f}", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if stop_detected and not car_stopped:
                send_command(arduino, "STOP")
                car_stopped = True

            cv2.imshow("Stop Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()
        print("Done.")