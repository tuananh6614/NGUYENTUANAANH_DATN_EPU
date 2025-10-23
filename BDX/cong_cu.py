"""
File công cụ - Chứa các hàm tiện ích (utility functions)
"""
import os
import cv2
import time
import socket
import logging
import numpy as np
from typing import List
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

# Import constants từ cau_hinh
from cau_hinh import DIR_IN, DIR_OUT

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

        # Scale to fit nhưng giữ aspect ratio và căn giữa
        label_w, label_h = label.width(), label.height()
        pix_w, pix_h = pix.width(), pix.height()

        if pix_w <= label_w and pix_h <= label_h:
            scaled_pix = pix
        else:
            scaled_pix = pix.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(scaled_pix)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)
    except Exception as e:
        pass

# =================================================================================================
# CÔNG CỤ CAMERA
# =================================================================================================

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
# CÔNG CỤ MẠNG
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
