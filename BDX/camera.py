"""
File camera - Chứa class CameraWorker để xử lý camera thread
"""
import os
import cv2
import time
import logging
import threading
import collections
import numpy as np
from typing import List, Optional
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

# Import từ các file khác
from cau_hinh import CAP_WIDTH, CAP_HEIGHT, MIN_SHARPNESS
from cong_cu import sharpness_score

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

                    # Emit frame for display - Giảm tần suất để tăng tốc độ
                    if time.time() - last_emit >= target_dt * 1.5:  # Giảm FPS hiển thị
                        disp = frame
                        h0, w0 = frame.shape[:2]
                        # Resize nhỏ hơn cho hiển thị
                        if w0 > 480:
                            scale = 480 / w0
                            disp = cv2.resize(frame, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_LINEAR)

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
