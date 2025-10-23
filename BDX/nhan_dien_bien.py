"""
Module nhận dạng biển số xe
Sử dụng YOLOv8 để phát hiện biển + EasyOCR để đọc ký tự
"""
import cv2
import numpy as np
import re
import logging
from typing import Optional, Tuple, List, Dict
from ultralytics import YOLO
from easyocr import Reader
from concurrent.futures import ThreadPoolExecutor, as_completed

from cau_hinh import (
    YOLO_MODEL_PATH, YOLO_CONF, YOLO_IOU, MAX_CACHE_SIZE,
    MIN_REL_AREA, WARP_W, WARP_H, VOTE_MIN_HITS, MAX_WORKERS
)
from cong_cu import enhance_for_plate, sharpness_score


def clean_plate_text(raw: str) -> str:
    """Làm sạch text biển số: chỉ giữ chữ số, chữ cái, gạch ngang, dấu chấm"""
    s = re.sub(r"[^A-Z0-9\-\.]", "", raw.upper().replace(" ", ""))
    s = re.sub(r"\.+", ".", s)
    s = re.sub(r"\-+", "-", s)
    s = s.strip(".-")
    return s


def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_plate(crop, w=200, h=60):
    """Căn chỉnh biển số về dạng chữ nhật (bird's eye view)"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            ordered = order_points(pts)
            dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(crop, M, (w, h))
            return warped

    # Nếu không tìm thấy 4 góc, resize đơn giản
    return cv2.resize(crop, (w, h))


class ALPR:
    """
    Engine nhận dạng biển số xe
    - Dùng YOLOv8 để detect vùng biển
    - Dùng EasyOCR để OCR text
    - Có cache để tránh OCR lại vùng giống nhau
    - Hỗ trợ multi-threading qua ThreadPoolExecutor
    """
    def __init__(self, model_path=YOLO_MODEL_PATH, conf=YOLO_CONF, iou=YOLO_IOU):
        logging.info("Initializing ALPR engine...")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.reader = Reader(["en"], gpu=False, verbose=False)
        self._cache: Dict[str, str] = {}
        self.pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logging.info("ALPR engine ready")

    def cleanup(self):
        """Đóng thread pool khi không dùng nữa"""
        self.pool.shutdown(wait=False)

    def _cache_get(self, key: str) -> Optional[str]:
        """Lấy kết quả OCR từ cache nếu có"""
        return self._cache.get(key)

    def _cache_put(self, key: str, val: str):
        """Lưu kết quả OCR vào cache, giới hạn kích thước"""
        if len(self._cache) >= MAX_CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = val

    def infer_once(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        """
        Nhận dạng biển số từ 1 frame
        Returns: (plate_text, debug_frame)
        """
        H, W = frame.shape[:2]
        debug = frame.copy()

        # YOLO inference
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return None, debug

        # Sắp xếp theo confidence giảm dần
        confs = results.boxes.conf.cpu().numpy()
        order = np.argsort(confs)[::-1]

        best_txt = None
        best_score = -1
        scale = 1.0

        # Chỉ xử lý top 3 detections để tăng tốc
        for idx in order[:3]:
            b = results.boxes[int(idx)]
            x1, y1, x2, y2 = map(int, (b.xyxy[0] / scale).tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            w, h = x2-x1, y2-y1
            if w <= 1 or h <= 1: continue

            rel_area = (w*h)/(W*H)
            if rel_area < MIN_REL_AREA: continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            # Skip sharpness check để tăng tốc độ
            # if sharpness_score(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < MIN_SHARPNESS: continue

            key = f"{x1}-{y1}-{x2}-{y2}"
            cached = self._cache_get(key)

            if cached:
                text = cached
            else:
                warped = warp_plate(crop, WARP_W, WARP_H)
                warped = enhance_for_plate(warped)
                dets = self.reader.readtext(warped, detail=0)  # detail=0 để nhanh hơn
                text = " ".join(dets) if dets else ""
                text = clean_plate_text(text)
                if text: self._cache_put(key, text)

            score = float(b.conf.item()) + 0.05*len(text)
            if text and score > best_score:
                best_score = score
                best_txt = text

            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dbg_txt = text if text else "?"
            cv2.putText(debug, dbg_txt, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return best_txt, debug

    def infer_multi(self, frames: List[np.ndarray]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Nhận dạng biển số từ nhiều frames, bỏ phiếu để chọn kết quả tốt nhất
        Returns: (plate_text, debug_frame)
        """
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
