import os
import re
import cv2
import sys
import json
import time
import socket
import random
import string
import datetime
import subprocess
import threading
import collections
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup logging - CHỈ log ra console, không ghi file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # ✅ Chỉ log ra console
    ]
)

# Torch để tự phát hiện GPU
try:
    import torch
except Exception:
    torch = None

# YOLOv8
from ultralytics import YOLO

# EasyOCR
import easyocr

# MQTT (tùy chọn, nếu không cài sẽ tự OFF)
try:
    from paho.mqtt import client as mqtt
except Exception:
    mqtt = None

# PySide6
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLineEdit, QPushButton, QStatusBar, QMessageBox,
    QSizePolicy, QDialog, QComboBox, QDialogButtonBox, QFormLayout, QCheckBox
)

# =================================================================================================
# CẤU HÌNH
# =================================================================================================

CFG_FILE = "config.json"

# Đường dẫn model YOLO
YOLO_MODEL_PATH = r"D:\FIRMWAVE\Automatic-License-Plate-Recognition-using-YOLOv8\license_plate_detector.pt"

# Thư mục lưu ảnh
DIR_IN  = Path("plates/IN")
DIR_OUT = Path("plates/OUT")
DIR_IN.mkdir(parents=True, exist_ok=True)
DIR_OUT.mkdir(parents=True, exist_ok=True)

# Tham số ALPR
YOLO_CONF     = 0.35
YOLO_IMGSZ    = 416
MIN_REL_AREA  = 0.010
MIN_SHARPNESS = 60.0
CAP_WIDTH, CAP_HEIGHT = 640, 480

# Multi-Frame Voting
VOTE_FRAMES   = 7
VOTE_GAP_MS   = 30
VOTE_MIN_HITS = 2

# Perspective warp (nắn hình)
WARP_W, WARP_H = 320, 96

# Phí demo
FEE_FLAT = 3000

# Regex biển VN (rộng)
PLATE_RE = re.compile(r"[0-9]{2,3}[A-Z]{1,2}[-\s]?[0-9]{3,5}")

# =================================================================================================
# EXCEPTION HANDLER
# =================================================================================================

def exception_hook(exctype, value, traceback):
    logging.error("Uncaught exception", exc_info=(exctype, value, traceback))
    sys.__excepthook__(exctype, value, traceback)

sys.excepthook = exception_hook

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
    broker_conf: str = r"D:\FIRMWAVE\project\mosquitto.conf"

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

# =================================================================================================
# CÔNG CỤ ẢNH / VIDEO
# =================================================================================================

def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def enhance_for_plate(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def np_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_copy = rgb.copy()  # ✅ FIX: Copy data
    h, w, ch = rgb_copy.shape
    return QImage(rgb_copy.data, w, h, ch*w, QImage.Format_RGB888).copy()

def set_pixmap_fit_no_upscale(label: QLabel, img: QImage):
    try:
        if label.width() <= 0 or label.height() <= 0 or img.isNull():
            return
        pix = QPixmap.fromImage(img)
        sw, sh = label.width() / pix.width(), label.height() / pix.height()
        scale = min(1.0, sw, sh)
        new_size = QSize(int(pix.width()*scale), int(pix.height()*scale))
        label.setPixmap(pix.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)
    except Exception as e:
        logging.error(f"Error in set_pixmap_fit_no_upscale: {e}")

def list_cameras(max_index=8) -> List[int]:
    found = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                found.append(i)
                cap.release()
        except Exception as e:
            logging.warning(f"Error checking camera {i}: {e}")
    return found

# =================================================================================================
# CÔNG CỤ XỬ LÝ BIỂN SỐ
# =================================================================================================

def plate_similarity(plate1: str, plate2: str) -> float:
    """Tính độ tương đồng giữa 2 biển số (0.0 - 1.0)"""
    if not plate1 or not plate2:
        return 0.0

    # Chuẩn hóa: bỏ khoảng trắng, dấu gạch ngang, chữ hoa
    p1 = plate1.upper().replace(" ", "").replace("-", "")
    p2 = plate2.upper().replace(" ", "").replace("-", "")

    if p1 == p2:
        return 1.0

    # Tính Levenshtein distance (edit distance)
    len1, len2 = len(p1), len(p2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Dynamic programming matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if p1[i-1] == p2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[len1][len2]
    max_len = max(len1, len2)
    similarity = 1.0 - (edit_distance / max_len)

    return similarity

# =================================================================================================
# CÔNG CỤ FILE
# =================================================================================================

def cleanup_old_images(days_old=3):
    """Xóa ảnh cũ hơn N ngày"""
    try:
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for root_dir in [DIR_IN, DIR_OUT]:
            if not root_dir.exists():
                continue

            # Duyệt qua tất cả file trong thư mục và thư mục con
            for file_path in root_dir.rglob("*.jpg"):
                try:
                    # Kiểm tra thời gian modified
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()  # Xóa file
                        deleted_count += 1
                        logging.debug(f"Deleted old image: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting {file_path}: {e}")

        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} images older than {days_old} days")
        else:
            logging.info(f"No images older than {days_old} days found")

    except Exception as e:
        logging.error(f"Error in cleanup_old_images: {e}")

# =================================================================================================
# CÔNG CỤ MẠNG / MQTT
# =================================================================================================

def is_port_open(host: str, port: int, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def get_local_ips() -> set:
    ips = {"127.0.0.1", "localhost", "0.0.0.0"}
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            ips.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips

# =================================================================================================
# CAMERA THREAD (✅ FIXED)
# =================================================================================================

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    opened = Signal(bool)
    error_occurred = Signal(str)  # ✅ NEW: Signal cho lỗi

    def __init__(self, source=0, width=CAP_WIDTH, height=CAP_HEIGHT, mirror=False, parent=None):
        super().__init__(parent)
        self.source, self.width, self.height, self.mirror = source, width, height, mirror
        self._running = False
        self._buffer = collections.deque(maxlen=25)
        self._buf_lock = threading.Lock()
        self.cap = None

    def run(self):
        self._running = True
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        consecutive_errors = 0
        max_errors = 10
        
        try:
            self.cap = cv2.VideoCapture(self.source, backend)
            if self.width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok = self.cap.isOpened()
            self.opened.emit(ok)
            if not ok:
                logging.error(f"Camera {self.source} failed to open")
                return

            target_dt = 1/25.0
            last_emit = 0.0
            
            while self._running:
                if not self._running:  # ✅ Double check
                    break
                    
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            logging.error(f"Camera {self.source}: Too many errors")
                            self.error_occurred.emit("Camera lỗi liên tục!")
                            break
                        QThread.msleep(50)
                        continue
                    
                    consecutive_errors = 0  # ✅ Reset on success
                    
                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    # Calculate sharpness
                    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                    score = sharpness_score(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                    
                    with self._buf_lock:
                        self._buffer.append((score, frame.copy()))

                    # Emit frame for display
                    if time.time() - last_emit >= target_dt:
                        disp = frame
                        h0, w0 = frame.shape[:2]
                        if w0 > 640:
                            scale = 640 / w0
                            disp = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
                        
                        # ✅ FIX: Proper data copy
                        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        rgb_copy = rgb.copy()
                        h, w, ch = rgb_copy.shape
                        qimg = QImage(rgb_copy.data, w, h, ch*w, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(qimg)
                        last_emit = time.time()

                    rem = target_dt - (time.time() - time.time())
                    if rem > 0:
                        QThread.msleep(int(rem*1000))
                        
                except Exception as e:
                    logging.error(f"Error in camera loop: {e}")
                    consecutive_errors += 1
                    
        except Exception as e:
            logging.error(f"Camera thread crashed: {e}", exc_info=True)
        finally:
            self._running = False
            QThread.msleep(100)  # ✅ Wait for any pending operations
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    logging.info(f"Camera {self.source} released")
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")

    def stop(self):
        """✅ FIXED: Proper shutdown sequence"""
        logging.info(f"Stopping camera {self.source}...")
        self._running = False
        
        # Wait for thread to finish FIRST
        if not self.wait(3000):  # ✅ Increased timeout
            logging.warning(f"Camera {self.source} thread didn't stop gracefully")
            self.terminate()
            self.wait(1000)
        
        # Then release camera
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logging.info(f"Camera {self.source} released after stop")
        except Exception as e:
            logging.error(f"Error releasing camera in stop(): {e}")

    def get_recent_frames(self, n: int, min_score: float = MIN_SHARPNESS, gap_ms: int = 0) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        with self._buf_lock:
            if not self._buffer:
                return frames
            sorted_buf = sorted(list(self._buffer), key=lambda t: -t[0])
        for s, f in sorted_buf:
            if s < min_score:
                continue
            frames.append(f.copy())
            if len(frames) >= n:
                break
        if gap_ms > 0 and len(frames) >= 2:
            frames = frames[::2] if len(frames) > n else frames
            frames = frames[:n]
        return frames

    def best_recent_frame(self, min_score: float = MIN_SHARPNESS) -> Optional[np.ndarray]:
        with self._buf_lock:
            if not self._buffer:
                return None
            s, f = max(self._buffer, key=lambda t: t[0])
            return f.copy() if s >= min_score else None

# =================================================================================================
# ALPR (✅ FIXED: Cache management + cleanup)
# =================================================================================================

def clean_plate_text(txt: str) -> str:
    t = txt.upper().replace("O", "0")
    t = re.sub(r"[^A-Z0-9\s-]", "", t)
    m = PLATE_RE.search(t.replace(" ", ""))
    if not m:
        return t.strip()
    raw = m.group(0)
    if "-" not in raw:
        if len(raw) > 3 and raw[2].isalpha():
            raw = raw[:2] + "-" + raw[2:]
        elif len(raw) > 4 and raw[3].isalpha():
            raw = raw[:3] + "-" + raw[3:]
    raw = re.sub(r"-([A-Z]{1,2})(\d+)", r"-\1 \2", raw)
    return raw.strip()

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_plate(crop_bgr: np.ndarray, out_w=WARP_W, out_h=WARP_H) -> np.ndarray:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(crop_bgr, (out_w, out_h))

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * (crop_bgr.shape[0]*crop_bgr.shape[1]):
        return cv2.resize(crop_bgr, (out_w, out_h))

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32)

    src = order_points(box.astype(float)).astype(np.float32)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    if src.shape != (4,2):
        return cv2.resize(crop_bgr, (out_w, out_h))
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop_bgr, M, (out_w, out_h))
    return warped

class ALPR:
    def __init__(self, weights: str, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, 
                 max_workers=4, cache_ttl=5.0, max_cache_size=100):
        # Kiểm tra GPU
        self.device = 'cuda' if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else 'cpu'
        logging.info(f"ALPR using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(weights)
        try:
            self.model.to(self.device)
            logging.info(f"YOLO model loaded on {self.device}")
        except Exception as e:
            logging.warning(f"Failed to move YOLO to GPU: {e}")
            self.device = 'cpu'
            
        self.conf = conf
        self.imgsz = imgsz
        
        # Khởi tạo EasyOCR - dùng CPU vì RTX 5050 không tương thích
        logging.info(f"Initializing EasyOCR with GPU=False (RTX 5050 compatibility)")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        logging.info(f"EasyOCR initialized successfully")
        
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # ✅ FIX: Cache with size limit
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: str) -> Optional[str]:
        with self._cache_lock:
            now = time.time()
            if key in self.cache:
                txt, ts = self.cache[key]
                if now - ts < self.cache_ttl:
                    return txt
                else:
                    try:
                        del self.cache[key]
                    except Exception:
                        pass
            return None

    def _cache_put(self, key: str, text: str):
        with self._cache_lock:
            now = time.time()
            
            # ✅ FIX: Cleanup old cache if too large
            if len(self.cache) >= self.max_cache_size:
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                to_remove = len(sorted_items) // 5  # Remove 20%
                for k, _ in sorted_items[:to_remove]:
                    try:
                        del self.cache[k]
                    except Exception:
                        pass
                logging.info(f"Cache cleanup: removed {to_remove} entries")
            
            self.cache[key] = (text, now)

    def cleanup(self):
        """✅ NEW: Cleanup when closing app"""
        logging.info("Cleaning up ALPR resources...")
        try:
            self.pool.shutdown(wait=True, timeout=5)
        except Exception as e:
            logging.error(f"Error shutting down thread pool: {e}")
        
        with self._cache_lock:
            self.cache.clear()
        
        logging.info("ALPR cleanup completed")

    def infer_once(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        debug = frame.copy()
        H, W = frame.shape[:2]
        try:
            results = self.model(
                frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
        except TypeError:
            results = self.model(
                frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
        except Exception:
            results = self.model(
                frame, device='cpu', conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]
            
        if results.boxes is None or len(results.boxes) == 0:
            return None, debug
            
        best_txt, best_score = None, -1.0
        confs = results.boxes.conf.detach().cpu().numpy()
        order = np.argsort(-confs)
        
        for idx in order:
            b = results.boxes[int(idx)]
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W-1,x2), min(H-1,y2)
            w,h = x2-x1, y2-y1
            if w <= 1 or h <= 1: continue
            
            rel_area = (w*h)/(W*H)
            if rel_area < MIN_REL_AREA: continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            if sharpness_score(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < MIN_SHARPNESS: continue
            
            key = f"{x1}-{y1}-{x2}-{y2}"
            cached = self._cache_get(key)
            
            if cached:
                text = cached
            else:
                warped = warp_plate(crop, WARP_W, WARP_H)
                warped = enhance_for_plate(warped)
                dets = self.reader.readtext(warped)
                text = " ".join([d[1] for d in dets]) if dets else ""
                text = clean_plate_text(text)
                if text: self._cache_put(key, text)
            
            score = float(b.conf.item()) + 0.05*len(text)
            if text and score > best_score:
                best_score = score
                best_txt = text
                
            cv2.rectangle(debug, (x1,y1), (x2,y2), (0,255,0), 2)
            dbg_txt = text if text else "?"
            cv2.putText(debug, dbg_txt, (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                        
        return best_txt, debug

    def infer_multi(self, frames: List[np.ndarray]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        if not frames:
            return None, None
            
        futures = {self.pool.submit(self.infer_once, f): f for f in frames}
        votes: Dict[str, int] = {}
        best_debug = None
        best_plate = None
        best_score = -1
        
        for fut in as_completed(futures):
            try:
                plate, debug = fut.result()
                if plate:
                    votes[plate] = votes.get(plate, 0) + 1
                    cur_score = votes[plate]*10 + len(plate)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_plate = plate
                        best_debug = debug
            except Exception as e:
                logging.error(f"Error in infer_multi: {e}")
                
        if not votes:
            return None, frames[0].copy()
            
        max_hits = max(votes.values())
        cands = [p for p, c in votes.items() if c == max_hits]
        cands.sort(key=lambda s: (-len(s), s))
        plate_final = cands[0]
        
        if max_hits < VOTE_MIN_HITS:
            return None, best_debug
            
        return plate_final, best_debug

# =================================================================================================
# UI PHỤ TRỢ
# =================================================================================================

def qlabel_video_placeholder(text=""):
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setMinimumSize(QSize(360, 240))
    lbl.setStyleSheet("QLabel{background:#1f1f1f;color:#cccccc;border:1px solid #3a3a3a;}")
    return lbl

# =================================================================================================
# HỘP THOẠI THIẾT LẬP
# =================================================================================================

class SettingsDialog(QDialog):
    def __init__(self, cfg: UiConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt")
        self.resize(520, 380)

        cams = list_cameras()
        self.cb_in  = QComboBox()
        self.cb_out = QComboBox(); self.cb_out.addItem("— Tắt —", -1)
        if not cams:
            self.cb_in.addItem("Không tìm thấy camera", -1)
        else:
            for i in cams:
                self.cb_in.addItem(f"Camera {i}", i)
                self.cb_out.addItem(f"Camera {i}", i)

        if cams and cfg.cam_in_index in cams:
            self.cb_in.setCurrentIndex(cams.index(cfg.cam_in_index))
        if cfg.cam_out_index == -1:
            self.cb_out.setCurrentIndex(0)
        elif cfg.cam_out_index in cams:
            self.cb_out.setCurrentIndex(1 + cams.index(cfg.cam_out_index))

        self.ed_slots  = QLineEdit(str(cfg.total_slots))

        self.chk_mqtt  = QCheckBox("Bật MQTT"); self.chk_mqtt.setChecked(cfg.mqtt_enable)
        self.ed_host   = QLineEdit(cfg.mqtt_host)
        self.ed_port   = QLineEdit(str(cfg.mqtt_port))
        self.ed_gate   = QLineEdit(cfg.gate_id)

        self.chk_autob = QCheckBox("Tự khởi động Mosquitto nếu broker là máy này")
        self.chk_autob.setChecked(cfg.auto_start_broker)
        self.ed_bexe   = QLineEdit(cfg.broker_exe)
        self.ed_bconf  = QLineEdit(cfg.broker_conf)

        form = QFormLayout()
        form.addRow("Ngõ vào:", self.cb_in)
        form.addRow("Ngõ ra:", self.cb_out)
        form.addRow("SLOT TỔNG:", self.ed_slots)
        form.addRow(self.chk_mqtt)
        form.addRow("MQTT Host:", self.ed_host)
        form.addRow("MQTT Port:", self.ed_port)
        form.addRow("Gate ID:", self.ed_gate)
        form.addRow(self.chk_autob)
        form.addRow("mosquitto.exe:", self.ed_bexe)
        form.addRow("mosquitto.conf:", self.ed_bconf)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(buttons)

    def values(self):
        return (
            self.cb_in.currentData(), self.cb_out.currentData(), int(self.ed_slots.text() or "1"),
            self.chk_mqtt.isChecked(), self.ed_host.text().strip() or "127.0.0.1",
            int(self.ed_port.text() or "1883"), self.ed_gate.text().strip() or "gate1",
            self.chk_autob.isChecked(), self.ed_bexe.text().strip(), self.ed_bconf.text().strip()
        )

# =================================================================================================
# CỬA SỔ CHÍNH (✅ FIXED: Thread safety + proper cleanup)
# =================================================================================================

class MainWindow(QMainWindow):
    # Qt Signals for thread-safe triggering
    trigger_shoot_in = Signal(str)  # Signal with card_id
    trigger_shoot_out = Signal(str)  # Signal with card_id
    trigger_update_revenue = Signal(str, str, int, int)  # ✅ NEW: Signal(card_id, plate, fee, total_revenue)

    def __init__(self, cfg: UiConfig):
        super().__init__()
        self.cfg = cfg

        # Lưu theo mã thẻ: {card_id: {"plate": "...", "time": datetime, "card_id": ..., "paid": False}}
        self._in_records: Dict[str, Dict] = {}
        self._paid_cards: Dict[str, datetime.datetime] = {}  # ✅ NEW: {card_id: paid_time}
        self._rec_lock = threading.RLock()

        # ✅ NEW: Revenue tracking (từ in_records.json)
        self._total_revenue = 0
        self._total_in_count = 0

        self._local_ips = get_local_ips()
        self._mqtt_connected = False
        self._esp_devices: Dict[str, Dict] = {}  # Lưu thông tin nhiều ESP32: {mac: {ip, last_hb, online}}
        self._hb_timeout = 5.0  # Giảm timeout xuống 5 giây
        self._mosq_proc = None
        self.mqtt_client = None
        self._pending_card_id = ""  # Lưu mã thẻ RFID tạm thời

        # Cleanup ảnh cũ (background thread)
        threading.Thread(target=cleanup_old_images, args=(3,), daemon=True).start()

        # Connect signals to slots
        self.trigger_shoot_in.connect(self._handle_shoot_in)
        self.trigger_shoot_out.connect(self._handle_shoot_out)
        self.trigger_update_revenue.connect(self._handle_update_revenue)  # ✅ NEW

        self._build_ui()

        # ✅ NEW: Load dữ liệu từ in_records.json SAU KHI build UI
        self._load_data_from_db()

        self._init_models()

        self.cam_in_worker: Optional[CameraWorker] = None
        self.cam_out_worker: Optional[CameraWorker] = None
        self.start_cameras()

        self.ensure_broker_running()
        self.init_mqtt()

        self._start_timers()

    def _build_ui(self):
        self.setWindowTitle("Phần mềm quản lý bãi gửi xe")
        self.resize(1280, 780)

        act_settings = QAction("Thiết lập", self); act_settings.triggered.connect(self.open_settings)
        act_full = QAction("Toàn màn hình", self, checkable=True); act_full.triggered.connect(self.toggle_fullscreen)
        menu = self.menuBar().addMenu("Cài đặt"); menu.addAction(act_settings); menu.addAction(act_full)

        self.lbl_cam_in  = qlabel_video_placeholder()
        self.lbl_img_in  = qlabel_video_placeholder("Ảnh xe vào")
        self.lbl_cam_out = qlabel_video_placeholder()
        self.lbl_img_out = qlabel_video_placeholder("Ảnh xe ra")

        grid = QGridLayout()
        grid.addWidget(self._group("Camera ngõ vào", self.lbl_cam_in), 0, 0)
        grid.addWidget(self._group("Ảnh xe vào", self.lbl_img_in),     0, 1)
        grid.addWidget(self._group("Camera ngõ ra", self.lbl_cam_out), 1, 0)
        grid.addWidget(self._group("Ảnh xe ra", self.lbl_img_out),     1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1);    grid.setRowStretch(1, 1)
        left = QWidget(); left.setLayout(grid)

        self.lbl_clock = QLabel("--:--:--"); self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setStyleSheet("font-size:22px;font-weight:600;")

        self.lbl_mqtt_state = QLabel("OFF"); self.lbl_mqtt_state.setStyleSheet("color:#bbb;font-weight:700;")
        self.lbl_mqtt_broker = QLabel("-"); self.lbl_mqtt_gate = QLabel("-"); self.lbl_mqtt_cid = QLabel("-")
        self.lbl_esp_last_msg = QLabel("-")
        self.lbl_esp_devices = QLabel("Không có thiết bị")  # Hiển thị danh sách ESP32
        self.lbl_esp_devices.setWordWrap(True)
        self.lbl_esp_devices.setStyleSheet("QLabel{background:#2a2a2a;color:#ddd;padding:8px;border:1px solid #3a3a3a;border-radius:4px;}")

        mqtt_form = QFormLayout()
        mqtt_form.addRow("Trạng thái:", self.lbl_mqtt_state)
        mqtt_form.addRow("Broker:", self.lbl_mqtt_broker)
        mqtt_form.addRow("Gate ID:", self.lbl_mqtt_gate)
        mqtt_form.addRow("Client ID:", self.lbl_mqtt_cid)
        mqtt_form.addRow("Tin nhắn cuối:", self.lbl_esp_last_msg)

        devices_label = QLabel("Thiết bị ESP32:")
        devices_label.setStyleSheet("font-weight:600;margin-top:8px;")

        mqtt_vbox = QVBoxLayout()
        form_widget = QWidget(); form_widget.setLayout(mqtt_form)
        mqtt_vbox.addWidget(form_widget)
        mqtt_vbox.addWidget(devices_label)
        mqtt_vbox.addWidget(self.lbl_esp_devices)

        box_mqtt = QGroupBox("Kết nối MQTT / ESP32")
        box_mqtt.setLayout(mqtt_vbox)


        self.ed_plate_cnt = self._count_box("0")
        self.ed_card  = self._ro_edit()
        self.ed_plate = self._ro_edit()
        self.ed_tin   = self._ro_edit()
        self.ed_tout  = self._ro_edit()
        self.ed_tdiff = self._ro_edit()
        self.ed_fee   = self._ro_edit()
        self.ed_slots_total = self._ro_edit()
        self.ed_slots_used  = self._ro_edit()
        self.ed_slots_free  = self._ro_edit()
        self.ed_total_revenue = self._ro_edit()  # ✅ NEW: Tổng tiền thu được

        # ✅ Chỉ giữ nút Xóa (các nút chụp đã được thay thế bằng MQTT tự động)
        btn_clear = QPushButton("Xóa");     btn_clear.clicked.connect(self.on_clear)

        form = QGridLayout(); r=0
        for label, widget in [
            ("SỐ XE", self.ed_plate_cnt),
            ("MÃ THẺ", self.ed_card),
            ("BIỂN SỐ", self.ed_plate),
            ("T/G XE VÀO", self.ed_tin),
            ("T/G XE RA", self.ed_tout),
            ("T/G GỬI XE", self.ed_tdiff),
            ("PHÍ GỬI XE", self.ed_fee),
            ("SLOT TỔNG", self.ed_slots_total),
            ("ĐÃ ĐỖ", self.ed_slots_used),
            ("CÒN LẠI", self.ed_slots_free),
            ("TỔNG TIỀN", self.ed_total_revenue),  # ✅ NEW
        ]:
            form.addWidget(QLabel(label), r, 0); form.addWidget(widget, r, 1); r += 1
        form.addWidget(btn_clear, r, 0, 1, 2)  # Nút xóa chiếm 2 cột

        box_info = QGroupBox("Thông tin"); wi = QWidget(); wi.setLayout(form)
        lay_info = QVBoxLayout(); lay_info.addWidget(wi); box_info.setLayout(lay_info)

        right = QVBoxLayout()
        right.addWidget(self.lbl_clock); right.addWidget(box_mqtt); right.addWidget(box_info); right.addStretch(1)
        panel_right = QWidget(); panel_right.setLayout(right); panel_right.setMaximumWidth(460)

        central = QWidget(); h = QHBoxLayout(central); h.addWidget(left, 2); h.addWidget(panel_right, 1)
        self.setCentralWidget(central)

        sb = QStatusBar(); self.lbl_status_cam = QLabel("Camera: —")
        sb.addWidget(self.lbl_status_cam); self.setStatusBar(sb)

        self.ed_slots_total.setText(str(load_config().total_slots))
        self._update_slot_counts()
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)

    def _load_data_from_db(self):
        """✅ NEW: Load dữ liệu từ in_records.json với structure mới"""
        try:
            if os.path.exists("in_records.json"):
                with open("in_records.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Kiểm tra cấu trúc mới (có summary và vehicles)
                if "summary" in data and "vehicles" in data:
                    summary = data["summary"]
                    vehicles = data["vehicles"]

                    # Load summary
                    self._total_revenue = summary.get("total_revenue", 0)
                    self._total_in_count = summary.get("daily_in_count", 0)
                    last_date = summary.get("last_date", "")

                    # Check qua ngày chưa
                    today = datetime.datetime.now().strftime("%Y-%m-%d")
                    if last_date != today:
                        logging.info(f"Date changed from {last_date} to {today} - Reset daily counter")
                        self._total_in_count = 0

                    # Load vehicles
                    for card_id, record in vehicles.items():
                        record["time"] = datetime.datetime.fromisoformat(record["time"])
                        self._in_records[card_id] = record

                    logging.info(f"Loaded summary: revenue={self._total_revenue:,}, count={self._total_in_count}, vehicles={len(self._in_records)}")
                else:
                    # Cấu trúc cũ (backward compatibility)
                    for card_id, record in data.items():
                        record["time"] = datetime.datetime.fromisoformat(record["time"])
                        self._in_records[card_id] = record
                    logging.info(f"Loaded old format: {len(self._in_records)} vehicles")

            # ✅ Update UI với dữ liệu đã load
            self.ed_total_revenue.setText(f"{self._total_revenue:,}")
            self.ed_plate_cnt.setText(str(self._total_in_count))
            self._update_slot_counts()

        except Exception as e:
            logging.error(f"Error loading in_records.json: {e}", exc_info=True)

    def _save_data_to_db(self):
        """✅ NEW: Lưu dữ liệu vào in_records.json với structure mới"""
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")

            # Convert in_records to serializable format
            vehicles = {}
            for card_id, record in self._in_records.items():
                vehicles[card_id] = {
                    "plate": record["plate"],
                    "time": record["time"].isoformat(),
                    "card_id": record["card_id"],
                    "paid": record.get("paid", False)
                }

            data = {
                "summary": {
                    "total_revenue": self._total_revenue,
                    "daily_in_count": self._total_in_count,
                    "last_date": today
                },
                "vehicles": vehicles
            }

            with open("in_records.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logging.info(f"Saved in_records.json: revenue={self._total_revenue:,}, count={self._total_in_count}, vehicles={len(vehicles)}")
        except Exception as e:
            logging.error(f"Error saving in_records.json: {e}", exc_info=True)

    def _check_midnight(self):
        """✅ NEW: Kiểm tra và reset daily counter vào 00:00"""
        try:
            if os.path.exists("in_records.json"):
                with open("in_records.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "summary" in data:
                        last_date = data["summary"].get("last_date", "")
                        today = datetime.datetime.now().strftime("%Y-%m-%d")
                        if last_date != today:
                            logging.info(f"Midnight reset: {last_date} → {today}")
                            self._total_in_count = 0
                            self.ed_plate_cnt.setText("0")
                            self._save_data_to_db()
        except Exception as e:
            logging.error(f"Error in _check_midnight: {e}")

    def _start_timers(self):
        self.tmr = QTimer(self); self.tmr.timeout.connect(self._tick); self.tmr.start(1000)
        self.tmr_hb = QTimer(self); self.tmr_hb.timeout.connect(self._check_esp_alive); self.tmr_hb.start(500)  # Kiểm tra mỗi 0.5 giây

        # ✅ NEW: Timer kiểm tra midnight
        self.tmr_midnight = QTimer(self)
        self.tmr_midnight.timeout.connect(self._check_midnight)
        self.tmr_midnight.start(60000)  # Check mỗi phút

    def _group(self, title, widget):
        gb = QGroupBox(title); v = QVBoxLayout(); v.setContentsMargins(6,8,6,6); v.addWidget(widget); gb.setLayout(v)
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); return gb

    def _ro_edit(self):
        e = QLineEdit(); e.setReadOnly(True)
        e.setStyleSheet("QLineEdit{background:#2a2a2a;color:#ddd;padding:6px;border:1px solid #3a3a3a;}")
        return e

    def _count_box(self, val="0"):
        e = QLineEdit(val); e.setReadOnly(True); e.setAlignment(Qt.AlignCenter)
        e.setStyleSheet("QLineEdit{background:#39d353;color:#0a0a0a;font-size:18px;border-radius:6px;padding:6px;font-weight:700;}")
        return e

    def _tick(self):
        self.lbl_clock.setText(time.strftime("%H:%M:%S  —  %a, %d/%m/%Y"))

    def _is_full(self) -> bool:
        try:
            total = int(self.cfg.total_slots)
        except Exception:
            total = 0
        return len(self._in_records) >= total if total > 0 else False

    # ----------------------------------------------------------------------------------------------
    # MODEL/OCR
    # ----------------------------------------------------------------------------------------------
    def _init_models(self):
        try:
            logging.info("Initializing ALPR models...")
            self.alpr = ALPR(YOLO_MODEL_PATH, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, 
                           max_workers=4, cache_ttl=5.0, max_cache_size=100)
            logging.info("ALPR initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ALPR: {e}", exc_info=True)
            self.alpr = None
            QMessageBox.critical(self, "ALPR", f"Không khởi tạo được YOLO/EasyOCR:\n{e}")

    # ----------------------------------------------------------------------------------------------
    # CAMERA
    # ----------------------------------------------------------------------------------------------
    def start_cameras(self):
        logging.info("Starting cameras...")
        self.stop_cameras()
        
        if self.cfg.cam_in_index >= 0:
            try:
                self.cam_in_worker = CameraWorker(self.cfg.cam_in_index, mirror=False)
                self.cam_in_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_in, img))
                self.cam_in_worker.opened.connect(lambda ok: self._cam_status(ok, "IN", self.cfg.cam_in_index))
                self.cam_in_worker.error_occurred.connect(lambda err: logging.error(f"Camera IN: {err}"))
                self.cam_in_worker.start()
                logging.info(f"Camera IN (index {self.cfg.cam_in_index}) started")
            except Exception as e:
                logging.error(f"Failed to start camera IN: {e}")
                self.lbl_status_cam.setText("Camera IN: Lỗi khi mở")
        else:
            self.lbl_status_cam.setText("Camera IN: tắt")

        if self.cfg.cam_out_index >= 0:
            try:
                self.cam_out_worker = CameraWorker(self.cfg.cam_out_index, mirror=False)
                self.cam_out_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_out, img))
                self.cam_out_worker.opened.connect(lambda ok: self._cam_status(ok, "OUT", self.cfg.cam_out_index))
                self.cam_out_worker.error_occurred.connect(lambda err: logging.error(f"Camera OUT: {err}"))
                self.cam_out_worker.start()
                logging.info(f"Camera OUT (index {self.cfg.cam_out_index}) started")
            except Exception as e:
                logging.error(f"Failed to start camera OUT: {e}")
                self.lbl_status_cam.setText(self.lbl_status_cam.text() + " | OUT: Lỗi khi mở")
        else:
            cur = self.lbl_status_cam.text()
            self.lbl_status_cam.setText((cur + " | OUT: tắt") if cur and "—" not in cur else "Camera OUT: tắt")

    def stop_cameras(self):
        """✅ FIXED: Proper camera cleanup"""
        logging.info("Stopping cameras...")
        
        if getattr(self, "cam_in_worker", None):
            try:
                self.cam_in_worker.stop()
                logging.info("Camera IN stopped")
            except Exception as e:
                logging.error(f"Error stopping camera IN: {e}")
            self.cam_in_worker = None
            
        if getattr(self, "cam_out_worker", None):
            try:
                self.cam_out_worker.stop()
                logging.info("Camera OUT stopped")
            except Exception as e:
                logging.error(f"Error stopping camera OUT: {e}")
            self.cam_out_worker = None
        
        # ✅ Wait for camera resources to be fully released
        QThread.msleep(500)

    def _cam_status(self, ok: bool, tag: str, idx: int):
        status = f"Camera {tag} (index {idx}): {'OK' if ok else 'Lỗi'}"
        self.lbl_status_cam.setText(status)
        logging.info(status)

    # ----------------------------------------------------------------------------------------------
    # SLOT / RECORD
    # ----------------------------------------------------------------------------------------------
    def _update_slot_counts(self):
        used = len(self._in_records)
        total = int(self.cfg.total_slots)
        free = max(0, total - used)
        self.ed_slots_used.setText(str(used))
        self.ed_slots_free.setText(str(free))
        self.ed_slots_total.setText(str(total))

    def _ensure_alpr(self) -> bool:
        if self.alpr is None:
            QMessageBox.warning(self, "ALPR", "Model YOLO/EasyOCR chưa sẵn sàng.")
            return False
        return True

    # ----------------------------------------------------------------------------------------------
    # LƯU ẢNH
    # ----------------------------------------------------------------------------------------------
    def _save_image_with_plate(self, plate: str, frame: np.ndarray, is_in: bool):
        root = DIR_IN if is_in else DIR_OUT
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        save_dir = root / today
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_plate = plate.replace(" ", "_")
        path = str(save_dir / f"{safe_plate}.jpg")
        cv2.imwrite(path, frame)
        logging.info(f"Saved image: {path}")

    # ----------------------------------------------------------------------------------------------
    # HÀNH ĐỘNG: CHỤP IN / OUT (✅ FIXED: Thread safety)
    # ----------------------------------------------------------------------------------------------
    def _handle_shoot_in(self, card_id: str):
        """Handler for shoot in signal - runs in main thread"""
        logging.info(f"[_handle_shoot_in] Received signal with card: {card_id}")
        self._pending_card_id = card_id
        self.on_shoot_in()

    def _handle_shoot_out(self, card_id: str):
        """Handler for shoot out signal - runs in main thread"""
        logging.info(f"[_handle_shoot_out] Received signal with card: {card_id}")
        self._pending_card_id = card_id
        self.on_shoot_out()

    def _handle_update_revenue(self, card_id: str, plate: str, fee: int, total_revenue: int):
        """✅ NEW: Handler for revenue update signal - runs in main thread"""
        logging.info(f"[_handle_update_revenue] Updating UI: card={card_id}, revenue={total_revenue:,}")
        self.ed_card.setText(card_id)
        self.ed_plate.setText(plate)
        self.ed_fee.setText(f"{fee:,}")
        self.ed_total_revenue.setText(f"{total_revenue:,}")
        logging.info(f"✅ UI UPDATED VIA SIGNAL: Card={card_id}, Revenue={total_revenue:,} VND")

    def on_shoot_in(self):
        """✅ FIXED: Full lock + proper error handling + RFID card support"""
        with self._rec_lock:  # ✅ Lock entire function
            logging.info("=== START SHOOT IN ===")

            # Lấy mã thẻ RFID nếu có
            card_id = self._pending_card_id
            self._pending_card_id = ""  # Reset
            logging.info(f"[DEBUG on_shoot_in] Card ID: {card_id}")

            if not self._ensure_alpr():
                logging.error("[DEBUG on_shoot_in] ALPR not ready")
                return
            if not self.cam_in_worker:
                logging.error("[DEBUG on_shoot_in] No camera IN worker")
                QMessageBox.warning(self, "Chụp IN", "Chưa có camera IN.")
                return

            logging.info("[DEBUG on_shoot_in] Starting frame capture...")

            if self._is_full():
                QMessageBox.warning(self, "BÃI ĐẦY", "SLOT đã đầy, không thể ghi nhận thêm xe vào.")
                return

            try:
                frames = self.cam_in_worker.get_recent_frames(VOTE_FRAMES, MIN_SHARPNESS, VOTE_GAP_MS)
                if not frames:
                    f = self.cam_in_worker.best_recent_frame()
                    if f is None:
                        QMessageBox.warning(self, "Chụp IN", "Không lấy được khung hình rõ.")
                        return
                    frames = [f]

                plate, debug = self.alpr.infer_multi(frames)
                if debug is not None:
                    set_pixmap_fit_no_upscale(self.lbl_img_in, np_to_qimage(debug))

                if not plate:
                    self.ed_plate.setText("Không đọc được")
                    QMessageBox.information(self, "Chụp IN", "Không nhận dạng được biển số.")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    (DIR_IN / "UNREAD").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(DIR_IN / "UNREAD" / f"UNREAD_{ts}.jpg"), frames[0])
                    logging.warning("No plate detected in IN")
                    return

                # Kiểm tra mã thẻ đã được sử dụng chưa
                if card_id and card_id in self._in_records:
                    existing = self._in_records[card_id]
                    QMessageBox.warning(self, "Thẻ đã sử dụng",
                        f"Mã thẻ {card_id} đã được dùng cho xe {existing['plate']}!\n"
                        f"Thời gian vào: {existing['time'].strftime('%H:%M:%S')}")
                    logging.warning(f"Duplicate card IN attempt: {card_id} (plate: {plate})")
                    return

                if not card_id:
                    QMessageBox.warning(self, "Chụp IN", "Không có mã thẻ! Vui lòng quẹt thẻ RFID.")
                    return

                if self._is_full():
                    QMessageBox.warning(self, "BÃI ĐẦY", "SLOT đã đầy, không thể ghi nhận thêm xe vào.")
                    return

                now = datetime.datetime.now()
                # Lưu theo mã thẻ
                self._in_records[card_id] = {
                    "plate": plate,
                    "time": now,
                    "card_id": card_id
                }
                logging.info(f"Recorded IN: Card={card_id}, Plate={plate} at {now}")

                self.ed_plate.setText(plate)
                self.ed_card.setText(card_id)
                self.ed_tin.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tout.clear(); self.ed_tdiff.clear(); self.ed_fee.setText("0")
                self._update_slot_counts()
                self._save_image_with_plate(plate, frames[0], True)

                # ✅ Update daily counter
                self._total_in_count += 1
                self.ed_plate_cnt.setText(str(self._total_in_count))

                # ✅ Save to DB
                self._save_data_to_db()

                logging.info(f"✅ Daily counter updated: {self._total_in_count} vehicles today")

            except Exception as e:
                logging.error(f"Error in on_shoot_in: {e}", exc_info=True)
                QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {e}")

    def on_shoot_out(self):
        """✅ FIXED: Full lock + proper error handling + RFID card support"""
        with self._rec_lock:  # ✅ Lock entire function
            logging.info("=== START SHOOT OUT ===")

            # Lấy mã thẻ RFID nếu có
            card_id = self._pending_card_id
            self._pending_card_id = ""  # Reset

            if not self._ensure_alpr():
                return
            if not self.cam_out_worker:
                QMessageBox.warning(self, "Chụp OUT", "Chưa có camera OUT.")
                return

            try:
                frames = self.cam_out_worker.get_recent_frames(VOTE_FRAMES, MIN_SHARPNESS, VOTE_GAP_MS)
                if not frames:
                    f = self.cam_out_worker.best_recent_frame()
                    if f is None:
                        QMessageBox.warning(self, "Chụp OUT", "Không lấy được khung hình rõ.")
                        return
                    frames = [f]

                plate, debug = self.alpr.infer_multi(frames)
                if debug is not None:
                    set_pixmap_fit_no_upscale(self.lbl_img_out, np_to_qimage(debug))

                if not plate:
                    QMessageBox.information(self, "Chụp OUT", "Không nhận dạng được biển số.")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    (DIR_OUT / "UNREAD").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(DIR_OUT / "UNREAD" / f"UNREAD_{ts}.jpg"), frames[0])
                    logging.warning("No plate detected in OUT")
                    return

                if not card_id:
                    QMessageBox.warning(self, "Chụp OUT", "Không có mã thẻ! Vui lòng quẹt thẻ RFID.")
                    return

                # Tìm bản ghi IN theo mã thẻ
                if card_id not in self._in_records:
                    QMessageBox.warning(self, "Chụp OUT",
                        f"Không tìm thấy bản ghi IN cho thẻ {card_id}!\n"
                        f"Vui lòng kiểm tra hoặc thẻ chưa được quẹt vào.")
                    logging.warning(f"OUT without IN: Card={card_id}, Plate={plate}")
                    self._save_image_with_plate(plate, frames[0], False)
                    return

                # Lấy thông tin từ bản ghi IN
                in_record = self._in_records.pop(card_id)
                plate_in = in_record["plate"]
                t_in = in_record["time"]
                already_paid = in_record.get("paid", False)  # ✅ Check nếu đã thanh toán

                now = datetime.datetime.now()
                logging.info(f"Recorded OUT: Card={card_id}, Plate IN={plate_in}, Plate OUT={plate} at {now}")

                # ✅ Cảnh báo BIỂN SỐ KHÔNG KHỚP - màu ĐỎ, CHẶN XE
                if plate != plate_in:
                    sim = plate_similarity(plate, plate_in)
                    if sim < 0.7:  # Chỉ cảnh báo nếu quá khác biệt
                        msg_box = QMessageBox(self)
                        msg_box.setIcon(QMessageBox.Critical)  # Icon đỏ
                        msg_box.setWindowTitle("🚨 BIỂN SỐ KHÔNG KHỚP")
                        msg_box.setText(
                            f"CẢNH BÁO: BIỂN SỐ KHÔNG KHỚP!\n\n"
                            f"🚗 Xe VÀO: {plate_in}\n"
                            f"🚗 Xe RA: {plate}\n"
                            f"🎫 Mã thẻ: {card_id}\n\n"
                            f"⛔ XE BỊ CHẶN - KHÔNG CHO RA!\n\n"
                            f"Vui lòng kiểm tra lại thẻ và biển số."
                        )
                        cancel_btn = msg_box.addButton("Hủy", QMessageBox.RejectRole)
                        cancel_btn.setDefault(True)  # Enter = Hủy

                        # ✅ Style màu ĐỎ nguy hiểm
                        msg_box.setStyleSheet("""
                            QMessageBox {
                                background-color: #2a2a2a;
                            }
                            QLabel {
                                color: #ff4444;
                                font-size: 14px;
                                min-width: 400px;
                                font-weight: bold;
                            }
                            QPushButton {
                                background-color: #dc3545;
                                color: white;
                                font-size: 14px;
                                font-weight: bold;
                                padding: 8px 16px;
                                border-radius: 4px;
                                min-width: 120px;
                            }
                            QPushButton:hover {
                                background-color: #c82333;
                            }
                        """)

                        msg_box.exec()

                        logging.warning(f"⛔ BLOCKED EXIT: Plate mismatch - IN={plate_in}, OUT={plate}, Card={card_id}, similarity={sim:.2f}")

                        # ✅ Trả lại thẻ vào in_records (vì đã pop() ở trên)
                        self._in_records[card_id] = in_record
                        logging.info(f"💡 Vehicle returned to in_records due to plate mismatch")

                        # ✅ RETURN - không cho xe ra
                        return

                diff = now - t_in
                mins = max(1, int(diff.total_seconds() // 60))
                fee  = FEE_FLAT

                # ✅ Cảnh báo CHƯA THANH TOÁN - cho phép trả tiền mặt
                if not already_paid:
                    # ✅ Popup màu VÀNG/CAM - nhấn OK hoặc Enter để xác nhận đã trả tiền mặt
                    msg_box = QMessageBox(self)
                    msg_box.setIcon(QMessageBox.Information)  # Icon info (xanh)
                    msg_box.setWindowTitle("💰 THANH TOÁN")
                    msg_box.setText(
                        f"XE CHƯA THANH TOÁN ONLINE\n\n"
                        f"🚗 Biển số: {plate_in}\n"
                        f"🕒 Thời gian vào: {t_in.strftime('%H:%M:%S')}\n"
                        f"⏱️ Thời gian gửi: {mins} phút\n\n"
                        f"💰 Phí cần thu: {fee:,} VNĐ\n\n"
                        f"➡️ Nhấn OK sau khi thu tiền mặt"
                    )
                    ok_btn = msg_box.addButton("OK - Đã thu tiền", QMessageBox.AcceptRole)
                    ok_btn.setDefault(True)  # ✅ Nhấn Enter = OK

                    # ✅ Style màu vàng/cam dễ nhìn
                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: #2a2a2a;
                        }
                        QLabel {
                            color: #ffcc00;
                            font-size: 14px;
                            min-width: 400px;
                            font-weight: bold;
                        }
                        QPushButton {
                            background-color: #28a745;
                            color: white;
                            font-size: 14px;
                            font-weight: bold;
                            padding: 8px 16px;
                            border-radius: 4px;
                            min-width: 150px;
                        }
                        QPushButton:hover {
                            background-color: #218838;
                        }
                    """)

                    msg_box.exec()

                    # ✅ Nhấn OK = Đã thu tiền mặt -> Tăng revenue và cho xe ra
                    self._total_revenue += fee
                    logging.info(f"💰 Cash payment collected at exit: {fee:,} VND")
                    logging.info(f"✅ Manual payment confirmed, vehicle exit allowed")

                # ✅ Xe đã thanh toán online - không cần thu phí
                else:
                    fee = 0
                    logging.info(f"✅ Already paid via terminal, no fee collected")

                self.ed_plate.setText(plate)
                self.ed_card.setText(card_id)
                self.ed_tin.setText(t_in.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tout.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tdiff.setText(f"{mins} phút")
                self.ed_fee.setText(f"{fee:,}" if fee > 0 else "Đã thanh toán")
                self.ed_total_revenue.setText(f"{self._total_revenue:,}")
                self._update_slot_counts()
                self._save_image_with_plate(plate, frames[0], False)

                # ✅ Save to DB
                self._save_data_to_db()

                logging.info(f"✅ Total revenue: {self._total_revenue:,} VND")

            except Exception as e:
                logging.error(f"Error in on_shoot_out: {e}", exc_info=True)
                QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {e}")

    # ----------------------------------------------------------------------------------------------
    # MISC
    # ----------------------------------------------------------------------------------------------
    def on_clear(self):
        for w in [self.ed_card, self.ed_plate, self.ed_tin, self.ed_tout, self.ed_tdiff, self.ed_fee]:
            try: w.clear()
            except Exception: pass
        for lbl, text in [(self.lbl_img_in,"Ảnh xe vào"), (self.lbl_img_out, "Ảnh xe ra")]:
            lbl.clear(); lbl.setText(text)

    def toggle_fullscreen(self, checked: bool):
        self.showFullScreen() if checked else self.showNormal()

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            cam_in, cam_out, slots, en_mqtt, host, port, gate, autob, bexe, bconf = dlg.values()
            if cam_in == -1:
                QMessageBox.warning(self, "Cài đặt", "Ngõ vào phải chọn 1 camera hợp lệ."); return
            self.cfg.cam_in_index = int(cam_in)
            self.cfg.cam_out_index = int(cam_out)
            self.cfg.total_slots = max(1, slots)
            self.cfg.mqtt_enable = bool(en_mqtt)
            self.cfg.mqtt_host   = host
            self.cfg.mqtt_port   = int(port)
            self.cfg.gate_id     = gate
            self.cfg.auto_start_broker = bool(autob)
            self.cfg.broker_exe  = bexe
            self.cfg.broker_conf = bconf
            save_config(self.cfg)
            
            logging.info("Settings saved, restarting cameras and MQTT...")
            self.start_cameras()
            self._update_slot_counts()
            self.restart_mqtt()
            self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
            self.lbl_mqtt_gate.setText(self.cfg.gate_id)

    def _set_mqtt_state(self, text, color="#bbb"):
        self.lbl_mqtt_state.setText(text)
        self.lbl_mqtt_state.setStyleSheet(f"color:{color};font-weight:700;")

    def _update_esp_devices_display(self):
        """Cập nhật hiển thị danh sách ESP32"""
        if not self._esp_devices:
            self.lbl_esp_devices.setText("Không có thiết bị")
            self.lbl_esp_devices.setStyleSheet("QLabel{background:#2a2a2a;color:#888;padding:8px;border:1px solid #3a3a3a;border-radius:4px;}")
            return

        lines = []
        online_count = 0
        for mac, info in self._esp_devices.items():
            status = "🟢 Online" if info.get("online", False) else "🔴 Offline"
            ip = info.get("ip", "N/A")
            last_seen = info.get("last_hb", 0)
            elapsed = int(time.time() - last_seen) if last_seen > 0 else 0

            if info.get("online", False):
                online_count += 1
                lines.append(f"{status} | MAC: {mac}\n   IP: {ip} | Heartbeat: {elapsed}s trước")
            else:
                lines.append(f"{status} | MAC: {mac}\n   Mất kết nối")

        text = "\n\n".join(lines)
        color = "#d4f4dd" if online_count > 0 else "#888"
        self.lbl_esp_devices.setText(text)
        self.lbl_esp_devices.setStyleSheet(f"QLabel{{background:#2a2a2a;color:{color};padding:8px;border:1px solid #3a3a3a;border-radius:4px;}}")


    def _refresh_conn_badge(self):
        online_count = sum(1 for dev in self._esp_devices.values() if dev.get("online", False))
        total_count = len(self._esp_devices)

        mqtt_txt = "Đã kết nối" if self._mqtt_connected else "Mất kết nối"

        if total_count > 0:
            esp_txt = f"{online_count}/{total_count} Online"
        else:
            esp_txt = "Không có thiết bị"

        color = "#39d353" if (self._mqtt_connected and online_count > 0) else ("#f1c40f" if self._mqtt_connected else "#ff6b6b")
        self._set_mqtt_state(f"MQTT: {mqtt_txt} | ESP32: {esp_txt}", color)

    def _check_esp_alive(self):
        """Kiểm tra trạng thái tất cả ESP32"""
        if not self._mqtt_connected:
            # Khi mất kết nối MQTT, đánh dấu tất cả ESP32 offline
            for mac in self._esp_devices:
                if self._esp_devices[mac].get("online", False):
                    self._esp_devices[mac]["online"] = False
                    logging.warning(f"ESP32 {mac} offline (MQTT disconnected)")
            self._refresh_conn_badge()
            self._update_esp_devices_display()
            return

        now = time.time()
        updated = False
        for mac, info in self._esp_devices.items():
            last_hb = info.get("last_hb", 0)
            if last_hb > 0 and (now - last_hb) > self._hb_timeout:
                if info.get("online", False):
                    info["online"] = False
                    updated = True
                    logging.warning(f"ESP32 {mac} heartbeat timeout")

        if updated:
            self._refresh_conn_badge()
            self._update_esp_devices_display()

    def ensure_broker_running(self):
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)
        if not (self.cfg.mqtt_enable and self.cfg.auto_start_broker):
            return
        host = (self.cfg.mqtt_host or "").strip()
        local_ips = get_local_ips()
        if host not in local_ips:
            return
        probe_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
        if is_port_open(probe_host, self.cfg.mqtt_port):
            logging.info("Broker already running")
            return
        exe, conf = self.cfg.broker_exe, self.cfg.broker_conf
        if not os.path.exists(exe) or not os.path.exists(conf):
            self._set_mqtt_state("Không thấy mosquitto/conf", "#ff6b6b"); return
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            exe_dir = os.path.dirname(exe) or None
            self._mosq_proc = subprocess.Popen(
                [exe, "-c", conf],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                creationflags=flags, cwd=exe_dir
            )
            self._set_mqtt_state("Đang khởi động broker…", "#f1c40f")
            logging.info("Started Mosquitto broker")
        except Exception as e:
            self._set_mqtt_state(f"Lỗi chạy broker: {e}", "#ff6b6b")
            logging.error(f"Failed to start broker: {e}")
            self._mosq_proc = None

    def init_mqtt(self):
        if not self.cfg.mqtt_enable or mqtt is None:
            self._mqtt_connected = False
            self._esp_online = False
            self._set_mqtt_state("OFF", "#bbb")
            return
        try:
            cid = "ui-" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
            self.lbl_mqtt_cid.setText(cid)
            self.mqtt_client = mqtt.Client(client_id=cid, protocol=mqtt.MQTTv311)
            try:
                self.mqtt_client.reconnect_delay_set(min_delay=0.5, max_delay=3)
            except Exception:
                pass
            def _on_connect(client, userdata, flags, rc, properties=None):
                self._mqtt_connected = (rc == 0)
                if rc == 0:
                    base = f"parking/gate/{self.cfg.gate_id}"
                    client.subscribe(base + "/event", qos=1)
                    client.subscribe(base + "/stats", qos=1)
                    client.subscribe(base + "/status", qos=1)
                    client.subscribe(base + "/heartbeat", qos=0)
                    client.subscribe(base + "/in", qos=1)
                    client.subscribe(base + "/out", qos=1)
                    client.subscribe(base + "/payment", qos=1)  # ✅ NEW: Payment confirmation

                    # ✅ Subscribe to payment terminal topics
                    client.subscribe("parking/payment/+/heartbeat", qos=0)
                    client.subscribe("parking/payment/+/status", qos=1)
                    client.subscribe("parking/payment/+/card_scanned", qos=1)
                    client.subscribe("parking/payment/+/payment_confirmed", qos=1)

                    logging.info("MQTT connected and subscribed to all topics")
                else:
                    # Đánh dấu tất cả thiết bị offline
                    for mac in self._esp_devices:
                        self._esp_devices[mac]["online"] = False
                    logging.error(f"MQTT connection failed: rc={rc}")
                self._refresh_conn_badge()
                self._update_esp_devices_display()
            def _on_disconnect(client, userdata, rc, properties=None):
                self._mqtt_connected = False
                # Đánh dấu tất cả thiết bị offline
                for mac in self._esp_devices:
                    self._esp_devices[mac]["online"] = False
                self._refresh_conn_badge()
                self._update_esp_devices_display()
                logging.warning("MQTT disconnected")
            def _on_message(client, userdata, msg):
                try:
                    topic = msg.topic
                    payload = {}
                    try:
                        payload = json.loads(msg.payload.decode("utf-8"))
                    except Exception:
                        pass
                    base = f"parking/gate/{self.cfg.gate_id}"

                    # Lấy MAC address từ payload
                    mac = payload.get("mac", None)

                    if topic.endswith("/status"):
                        online = bool(payload.get("online", False))
                        if mac:
                            # Cập nhật hoặc tạo mới thông tin ESP32
                            if mac not in self._esp_devices:
                                self._esp_devices[mac] = {}
                            self._esp_devices[mac]["online"] = online
                            self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                            self._esp_devices[mac]["last_hb"] = time.time() if online else 0
                            logging.info(f"ESP32 {mac} status: {'online' if online else 'offline'}")
                        self._refresh_conn_badge()
                        self._update_esp_devices_display()

                    elif topic.endswith("/heartbeat"):
                        if mac:
                            # Cập nhật heartbeat và thông tin
                            if mac not in self._esp_devices:
                                self._esp_devices[mac] = {}
                            self._esp_devices[mac]["online"] = True
                            self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                            self._esp_devices[mac]["last_hb"] = time.time()
                            self._refresh_conn_badge()
                            self._update_esp_devices_display()

                    elif topic == base + "/in":
                        card_id = payload.get("card_id", "")
                        logging.info(f"MQTT trigger: IN with card {card_id}")
                        msg_text = f"Yêu cầu chụp IN"
                        if card_id:
                            msg_text += f" | Thẻ: {card_id}"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)
                        # Emit signal để trigger chụp trong main thread
                        logging.info(f"[DEBUG] Emitting trigger_shoot_in signal with card: {card_id}")
                        self.trigger_shoot_in.emit(card_id)

                    elif topic == base + "/out":
                        card_id = payload.get("card_id", "")
                        logging.info(f"MQTT trigger: OUT with card {card_id}")
                        msg_text = f"Yêu cầu chụp OUT"
                        if card_id:
                            msg_text += f" | Thẻ: {card_id}"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)
                        # Emit signal để trigger chụp trong main thread
                        logging.info(f"[DEBUG] Emitting trigger_shoot_out signal with card: {card_id}")
                        self.trigger_shoot_out.emit(card_id)

                    elif topic == base + "/payment":
                        # ✅ NEW: Payment confirmation from terminal
                        card_id = payload.get("card_id", "")
                        logging.info(f"🔔 PAYMENT_CONFIRMED received: card={card_id}")

                        # Kiểm tra card có trong records không
                        with self._rec_lock:
                            logging.info(f"📝 Current in_records: {list(self._in_records.keys())}")

                            if card_id in self._in_records:
                                logging.info(f"✅ Card found in records - processing payment")
                                record = self._in_records.pop(card_id)
                                plate = record["plate"]

                                # Update revenue
                                fee = FEE_FLAT
                                self._total_revenue += fee

                                # Mark as paid
                                record["paid"] = True
                                self._paid_cards[card_id] = datetime.datetime.now()

                                # Save to DB
                                self._save_data_to_db()

                                logging.info(f"💰 Payment processed: Card={card_id}, Plate={plate}, Fee={fee:,}, Total={self._total_revenue:,}")

                                # ✅ Update UI using signal (thread-safe)
                                self.trigger_update_revenue.emit(card_id, plate, fee, self._total_revenue)
                            else:
                                logging.warning(f"❌ Card {card_id} not found in in_records")

                    elif topic.endswith("/event"):
                        event_type = payload.get("type", "unknown")
                        msg_text = f"Event: {event_type}"
                        if mac:
                            msg_text += f" (từ {mac})"
                        self.lbl_esp_last_msg.setText(msg_text)

                    elif topic.endswith("/stats"):
                        msg_text = "Nhận thống kê"
                        if mac:
                            msg_text += f" (từ {mac})"
                        self.lbl_esp_last_msg.setText(msg_text)

                    # ✅ Payment terminal topics
                    elif "parking/payment/" in topic:
                        if topic.endswith("/heartbeat"):
                            if mac:
                                if mac not in self._esp_devices:
                                    self._esp_devices[mac] = {}
                                self._esp_devices[mac]["online"] = True
                                self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                                self._esp_devices[mac]["last_hb"] = time.time()
                                self._refresh_conn_badge()
                                self._update_esp_devices_display()

                        elif topic.endswith("/status"):
                            online = bool(payload.get("online", False))
                            if mac:
                                if mac not in self._esp_devices:
                                    self._esp_devices[mac] = {}
                                self._esp_devices[mac]["online"] = online
                                self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                                self._esp_devices[mac]["last_hb"] = time.time() if online else 0
                                logging.info(f"Payment Terminal {mac} status: {'online' if online else 'offline'}")
                            self._refresh_conn_badge()
                            self._update_esp_devices_display()

                        elif topic.endswith("/card_scanned"):
                            # Payment terminal đã quét thẻ, gửi thông tin xe về
                            card_id = payload.get("card_id", "")
                            logging.info(f"💳 Payment terminal scanned card: {card_id}")

                            with self._rec_lock:
                                if card_id in self._in_records:
                                    record = self._in_records[card_id]
                                    already_paid = record.get("paid", False)

                                    # Gửi thông tin xe về payment terminal
                                    gate_id = topic.split("/")[2]  # Extract gate ID from topic
                                    response_topic = f"parking/payment/{gate_id}/vehicle_info"

                                    # ✅ Nếu đã thanh toán → fee = 0, paid = true
                                    # ✅ Nếu chưa thanh toán → fee = 3000, paid = false
                                    vehicle_info = {
                                        "plate": record["plate"],
                                        "time_in": record["time"].strftime("%H:%M:%S"),
                                        "time_out": datetime.datetime.now().strftime("%H:%M:%S"),
                                        "fee": 0 if already_paid else FEE_FLAT,
                                        "paid": already_paid
                                    }

                                    self.mqtt_client.publish(response_topic, json.dumps(vehicle_info))

                                    if already_paid:
                                        logging.info(f"📤 Sent vehicle info (ALREADY PAID): {vehicle_info}")
                                        self.lbl_esp_last_msg.setText(f"Xe {record['plate']} đã thanh toán")
                                    else:
                                        logging.info(f"📤 Sent vehicle info (NEED PAYMENT): {vehicle_info}")
                                        self.lbl_esp_last_msg.setText(f"Gửi thông tin xe {record['plate']}")
                                else:
                                    logging.warning(f"❌ Card {card_id} not found in records")

                        elif topic.endswith("/payment_confirmed"):
                            # ✅ Payment terminal xác nhận thanh toán - CHỈ đánh dấu đã trả tiền
                            card_id = payload.get("card_id", "")
                            logging.info(f"💰 Payment confirmed from terminal: card={card_id}")

                            with self._rec_lock:
                                if card_id in self._in_records:
                                    record = self._in_records[card_id]  # ✅ KHÔNG pop(), chỉ lấy reference
                                    plate = record["plate"]

                                    # ✅ Chỉ mark là đã thanh toán nếu chưa thanh toán
                                    if not record.get("paid", False):
                                        # Update revenue
                                        fee = FEE_FLAT
                                        self._total_revenue += fee

                                        # Mark as paid (GIỮ trong in_records để xe ra sau)
                                        record["paid"] = True
                                        self._paid_cards[card_id] = datetime.datetime.now()

                                        # Save to DB
                                        self._save_data_to_db()

                                        logging.info(f"✅ Payment processed via terminal: Card={card_id}, Plate={plate}, Fee={fee:,}, Total={self._total_revenue:,}")
                                        logging.info(f"📝 Vehicle STILL in parking (will exit via RFID-B OUT)")

                                        # Update UI using signal (thread-safe)
                                        self.trigger_update_revenue.emit(card_id, plate, fee, self._total_revenue)
                                    else:
                                        logging.info(f"⚠️ Card {card_id} already paid, skipping revenue update")
                                else:
                                    logging.warning(f"❌ Card {card_id} not found in in_records")

                except Exception as e:
                    logging.error(f"Error in MQTT message handler: {e}")
            self.mqtt_client.on_connect = _on_connect
            self.mqtt_client.on_disconnect = _on_disconnect
            self.mqtt_client.on_message = _on_message
            self._set_mqtt_state("Đang kết nối…", "#f1c40f")
            self.mqtt_client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=20)
            self.mqtt_client.loop_start()
            logging.info("MQTT connection initiated")
        except Exception as e:
            self._mqtt_connected = False
            self._esp_online = False
            self._set_mqtt_state(f"Lỗi MQTT: {e}", "#ff6b6b")
            logging.error(f"MQTT init error: {e}", exc_info=True)

    def restart_mqtt(self):
        logging.info("Restarting MQTT...")
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception as e:
            logging.error(f"Error stopping MQTT: {e}")
        self.mqtt_client = None
        self._mqtt_connected = False
        # Đánh dấu tất cả thiết bị offline
        for mac in self._esp_devices:
            self._esp_devices[mac]["online"] = False
        self._refresh_conn_badge()
        self._update_esp_devices_display()
        self.ensure_broker_running()
        self.init_mqtt()

    def closeEvent(self, e):
        """✅ FIXED: Proper cleanup sequence"""
        logging.info("=== CLOSING APPLICATION ===")

        # 1. Save data first
        logging.info("Step 1: Saving data...")
        self._save_data_to_db()

        # 2. Stop cameras
        logging.info("Step 2: Stopping cameras...")
        self.stop_cameras()
        QThread.msleep(500)

        # 3. Stop MQTT
        logging.info("Step 3: Stopping MQTT...")
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception as e:
            logging.error(f"Error stopping MQTT: {e}")
        
        # 4. Cleanup ALPR resources
        logging.info("Step 4: Cleaning up ALPR...")
        try:
            if self.alpr:
                self.alpr.cleanup()
        except Exception as e:
            logging.error(f"Error cleaning up ALPR: {e}")

        # 5. Stop Mosquitto broker
        logging.info("Step 5: Stopping broker...")
        if self._mosq_proc:
            try:
                self._mosq_proc.terminate()
                self._mosq_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._mosq_proc.kill()
                self._mosq_proc.wait()
            except Exception as e:
                logging.error(f"Error stopping broker: {e}")
        
        logging.info("=== APPLICATION CLOSED ===")
        super().closeEvent(e)

# =================================================================================================
# BOOT
# =================================================================================================

def main():
    logging.info("=" * 80)
    logging.info("Starting Parking Management Application")
    logging.info("=" * 80)
    
    cfg = load_config()
    if not os.path.exists(cfg.broker_conf):
        Path(cfg.broker_conf).parent.mkdir(parents=True, exist_ok=True)
        open(cfg.broker_conf, "w", encoding="utf-8").write(
            "listener 1883 0.0.0.0\nallow_anonymous true\npersistence false\n"
        )
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.show()
    logging.info("Application window shown")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()