"""
File cấu hình - Chứa UiConfig, load_config, save_config và các hằng số
"""
import os
import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path

# =================================================================================================
# CẤU HÌNH
# =================================================================================================

CFG_FILE = "config.json"

# GIỮ NGUYÊN đường dẫn model theo yêu cầu của bạn
YOLO_MODEL_PATH = r"E:\FIRMWAVE\Automatic-License-Plate-Recognition-using-YOLOv8\license_plate_detector.pt"

# Thư mục lưu ảnh
DIR_IN  = Path("plates/IN")
DIR_OUT = Path("plates/OUT")
DIR_IN.mkdir(parents=True, exist_ok=True)
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Tham số ALPR - Tối ưu cho tốc độ
YOLO_CONF     = 0.4  # Tăng confidence để giảm false positive
YOLO_IOU      = 0.45  # Intersection over Union threshold
YOLO_IMGSZ    = 320  # Giảm từ 416 → 320 cho tốc độ nhanh hơn
MIN_REL_AREA  = 0.010
MIN_SHARPNESS = 50.0  # Giảm threshold cho dễ pass hơn
CAP_WIDTH, CAP_HEIGHT = 640, 480
MAX_WORKERS   = 6  # ThreadPoolExecutor workers
MAX_CACHE_SIZE = 200  # ALPR cache size

# Multi-Frame Voting - Giảm số frame cho nhanh hơn
VOTE_FRAMES   = 5  # Giảm từ 7 → 5
VOTE_GAP_MS   = 30  # Giảm từ 40 → 30
VOTE_MIN_HITS = 2

# Perspective warp (nắn hình)
WARP_W, WARP_H = 320, 96

# Phí demo
FEE_FLAT = 3000

# Regex biển VN (rộng)
PLATE_RE = re.compile(r"[0-9]{2,3}[A-Z]{1,2}[-\s]?[0-9]{3,5}")

# =================================================================================================
# DATA CLASS CẤU HÌNH UI
# =================================================================================================

@dataclass
class UiConfig:
    cam_in_index: int = 0
    cam_out_index: int = -1
    total_slots: int = 50
    mqtt_enable: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    gate_id: str = "gate01"
    auto_start_broker: bool = True
    broker_exe: str = r"C:\Program Files\mosquitto\mosquitto.exe"
    broker_conf: str = r"E:\FIRMWAVE\project\mosquitto.conf"

def load_config() -> UiConfig:
    if os.path.exists(CFG_FILE):
        try:
            with open(CFG_FILE, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            defaults = UiConfig().__dict__
            data = {k: d.get(k, defaults[k]) for k in defaults.keys()}
            return UiConfig(**data)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
    return UiConfig()

def save_config(cfg: UiConfig):
    with open(CFG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg.__dict__, fh, ensure_ascii=False, indent=2)
