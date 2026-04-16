import cv2
import numpy as np
import time
import hashlib
from collections import OrderedDict
from PIL import Image
import onnxruntime as ort
import tflite_runtime.interpreter as tflite

# ═════════════════════════════════════════════════════════════════════════════
# 1. الإعدادات والمسارات
# ═════════════════════════════════════════════════════════════════════════════
YOLO_ONNX_PATH  = 'yolov8n.onnx'
TFLITE_PATH     = 'model_RTSR.tflite'   # حوّلي الـ .h5 لـ .tflite مسبقاً (انظر التعليق أسفله)
CONF_THRESHOLD  = 0.25
IOU_THRESHOLD   = 0.45
INPUT_SIZE      = 320   # للسرعة على الرازبيري

# أسماء الفئات لموديل التصنيف (43 فئة)
SIGN_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing veh over 3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5t prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End no passing veh > 3.5t"
]

# ═════════════════════════════════════════════════════════════════════════════
# 2. نظام الكاش
# ═════════════════════════════════════════════════════════════════════════════
class _LRUCache:
    def __init__(self, maxsize=64):
        self._cache = OrderedDict()
        self._maxsize = maxsize
    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    def put(self, key, value):
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

_sign_cache = _LRUCache(64)

# ═════════════════════════════════════════════════════════════════════════════
# 3. تحميل الموديلات (lazy loading)
# ═════════════════════════════════════════════════════════════════════════════
_yolo_session   = None
_tflite_interp  = None

def get_yolo_session():
    global _yolo_session
    if _yolo_session is None:
        print("Loading YOLOv8n ONNX model...")
        _yolo_session = ort.InferenceSession(
            YOLO_ONNX_PATH,
            providers=['CPUExecutionProvider']
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
# 4. منطق YOLO — preprocess / postprocess يدوي
# ═════════════════════════════════════════════════════════════════════════════
def preprocess_yolo(frame, input_size=INPUT_SIZE):
    """تجهيز الإطار لـ YOLOv8 ONNX (letterbox + normalize)"""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))

    # padding لجعلها input_size x input_size
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_top  = (input_size - nh) // 2
    pad_left = (input_size - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0          # BGR→RGB, normalize
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]               # HWC → 1CHW
    return blob, scale, pad_left, pad_top

def postprocess_yolo(outputs, scale, pad_left, pad_top, orig_h, orig_w):
    """
    YOLOv8 ONNX output shape: (1, 84, num_anchors)
    أول 4: cx cy w h — ثم 80 class scores
    """
    preds = outputs[0][0]           # shape: (84, num_anchors)
    preds = preds.T                 # → (num_anchors, 84)

    boxes_xywh = preds[:, :4]
    scores     = preds[:, 4:]
    class_ids  = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    # فلترة بالـ confidence
    mask = confidences >= CONF_THRESHOLD
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_xywh) == 0:
        return [], [], []

    # cx cy w h → x1 y1 x2 y2 (في فضاء الـ letterbox)
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # إزالة الـ padding وعكس الـ scale للعودة لأبعاد الإطار الأصلي
    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(),
        CONF_THRESHOLD, IOU_THRESHOLD
    )
    if len(indices) == 0:
        return [], [], []
    indices = indices.flatten()
    return boxes_xyxy[indices], confidences[indices], class_ids[indices]

# COCO class names (index 9 = traffic light, 11 = stop sign, 13 = ? — يمكنك تعديلها)
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
# 5. وظائف الرؤية الحاسوبية
# ═════════════════════════════════════════════════════════════════════════════
def get_traffic_light_color(image, bbox):
    """منطق v3.4 الاحترافي لكشف ألوان الإشارات"""
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ranges = {
        'red':    [(np.array([0,  70, 50]), np.array([10,  255, 255])),
                   (np.array([170,70, 50]), np.array([180, 255, 255]))],
        'yellow': [(np.array([15, 70, 50]), np.array([35,  255, 255]))],
        'green':  [(np.array([36, 70, 50]), np.array([89,  255, 255]))],
    }
    scores = {color: 0 for color in ranges}
    for color, r_list in ranges.items():
        for lo, hi in r_list:
            mask = cv2.inRange(hsv, lo, hi)
            scores[color] += cv2.countNonZero(mask)
    return max(scores, key=scores.get)

def classify_sign_crop(crop_bgr):
    """تصنيف اللوحة المرورية باستخدام TFLite ونظام الكاش"""
    h_str = hashlib.md5(crop_bgr.tobytes()).hexdigest()
    cached = _sign_cache.get(h_str)
    if cached:
        return cached

    try:
        interp = get_tflite_interpreter()
        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = np.array(Image.fromarray(img).resize((30, 30)), dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # (1, 30, 30, 3)

        # تحويل للـ quantized إذا كان الموديل uint8
        if in_det['dtype'] == np.uint8:
            scale, zero_point = in_det['quantization']
            img = (img / scale + zero_point).astype(np.uint8)

        interp.set_tensor(in_det['index'], img)
        interp.invoke()
        preds = interp.get_tensor(out_det['index'])[0]

        idx   = int(np.argmax(preds))
        label = SIGN_CLASSES[idx]
        _sign_cache.put(h_str, label)
        return label
    except Exception as e:
        print(f"[classify] {e}")
        return "Sign"

# ═════════════════════════════════════════════════════════════════════════════
# 6. الحلقة الرئيسية
# ═════════════════════════════════════════════════════════════════════════════
def run_full_system(source=0):
    # تحميل الموديلات مرة واحدة قبل بداية الحلقة
    yolo_sess = get_yolo_session()
    yolo_input_name = yolo_sess.get_inputs()[0].name

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("--- RPi Unified v3.8 (ONNX + TFLite) Running ---")
    print("Press 'q' to quit")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # ── الكشف بـ YOLO ──
        blob, scale, pad_left, pad_top = preprocess_yolo(frame)
        outputs = yolo_sess.run(None, {yolo_input_name: blob})
        boxes, confs, cls_ids = postprocess_yolo(
            outputs, scale, pad_left, pad_top, orig_h, orig_w
        )

        # ── معالجة كل كشف ──
        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = bbox
            name = COCO_NAMES.get(cls_id, str(cls_id))

            # 1. إشارة المرور
            if cls_id == 9:
                bw, bh = x2 - x1, y2 - y1
                if bh > 0 and (bw / bh) > 0.6:
                    continue
                color = get_traffic_light_color(frame, bbox)
                name  = f"TL: {color.upper()}"

            # 2. لوحات المرور
            elif cls_id in [11, 13]:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    name = classify_sign_crop(crop)

            # ── الرسم ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}",
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ── FPS ──
        curr_time = time.time()
        dt  = curr_time - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("RPi Unified v3.8", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# تحويل الموديل من .h5 → .tflite (شغّليه مرة واحدة على جهازك قبل نقله للرازبيري)
# ═════════════════════════════════════════════════════════════════════════════
# import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_keras_model(
#     tf.keras.models.load_model("model_RTSR.h5")
# )
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # اختياري: تصغير الحجم
# tflite_model = converter.convert()
# with open("model_RTSR.tflite", "wb") as f:
#     f.write(tflite_model)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_full_system(0)
