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
INPUT_SIZE           = 640  # 💡 تم التعديل هنا فقط لتتوافق مع الموديل الجديد
 
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

TURN_DURATION = 8.5       # فترة الأمان لمنع بايثون من إرسال GO وتخريب التفاف الأردوينو
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
        "bird","cat","dog","horse","sheep","cow","elephant","bear","
