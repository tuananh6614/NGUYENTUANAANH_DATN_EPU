"""
========================================================================================================
PHẦN MỀM QUẢN LÝ BÃI ĐỖ XE - PHIÊN BẢN REFACTORED
========================================================================================================
File này sử dụng các module đã được tách ra để dễ quản lý:
- cau_hinh.py: Cấu hình và constants
- cong_cu.py: Utility functions
- camera.py: CameraWorker class
- nhan_dien_bien.py: ALPR engine
- giao_dien.py: SettingsDialog + UI helpers

Tính năng:
✅ Nhận dạng biển số xe (ALPR) bằng YOLOv8 + EasyOCR
✅ Quản lý xe vào/ra bằng thẻ RFID
✅ Tính phí gửi xe
✅ MQTT để kết nối với ESP32
✅ Revenue tracking realtime (fixed với Signal/Slot pattern)
✅ in_records.json với structure mới (summary + vehicles)
========================================================================================================
"""
import os
import sys
import json
import time
import socket
import random
import string
import datetime
import subprocess
import threading
import logging
from typing import Optional, Dict
from pathlib import Path

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# MQTT (tùy chọn)
try:
    from paho.mqtt import client as mqtt
except Exception:
    mqtt = None

# PySide6
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLineEdit, QPushButton, QStatusBar, QMessageBox,
    QFormLayout, QDialog
)

# =================================================================================================
# IMPORT CÁC MODULE ĐÃ TÁCH RA
# =================================================================================================
from cau_hinh import (
    UiConfig, load_config, save_config,
    DIR_IN, DIR_OUT, VOTE_FRAMES, VOTE_GAP_MS, MIN_SHARPNESS, FEE_FLAT
)
from cong_cu import (
    np_to_qimage, set_pixmap_fit_no_upscale, cleanup_old_images,
    is_port_open, get_local_ips, plate_similarity
)
from camera import CameraWorker
from nhan_dien_bien import ALPR
from giao_dien import qlabel_video_placeholder, SettingsDialog

# =================================================================================================
# EXCEPTION HANDLER
# =================================================================================================
def exception_hook(exctype, value, traceback):
    logging.error("Uncaught exception", exc_info=(exctype, value, traceback))
    sys.__excepthook__(exctype, value, traceback)

sys.excepthook = exception_hook

# =================================================================================================
# CỬA SỔ CHÍNH - Refactored version với revenue realtime fix
# =================================================================================================
class MainWindow(QMainWindow):
    # Qt Signals for thread-safe triggering
    trigger_shoot_in = Signal(str)  # Signal(card_id)
    trigger_shoot_out = Signal(str)  # Signal(card_id)
    trigger_update_revenue = Signal(str, str, int, int)  # Signal(card_id, plate, fee, total_revenue)

    def __init__(self, cfg: UiConfig):
        super().__init__()
        self.cfg = cfg

        # Lưu theo mã thẻ: {card_id: {"plate": "...", "time": datetime, "card_id": ..., "paid": False}}
        self._in_records: Dict[str, Dict] = {}
        self._paid_cards: Dict[str, datetime.datetime] = {}  # {card_id: paid_time}
        self._rec_lock = threading.RLock()

        # Revenue tracking (từ in_records.json)
        self._total_revenue = 0
        self._total_in_count = 0

        self._local_ips = get_local_ips()
        self._mqtt_connected = False
        self._esp_devices: Dict[str, Dict] = {}  # {mac: {ip, last_hb, online}}
        self._hb_timeout = 5.0
        self._mosq_proc = None
        self.mqtt_client = None
        self._pending_card_id = ""

        # Cleanup ảnh cũ (background thread)
        threading.Thread(target=cleanup_old_images, args=(3,), daemon=True).start()

        # Connect signals to slots
        self.trigger_shoot_in.connect(self._handle_shoot_in)
        self.trigger_shoot_out.connect(self._handle_shoot_out)
        self.trigger_update_revenue.connect(self._handle_update_revenue)

        self._build_ui()

        # Load dữ liệu từ in_records.json SAU KHI build UI
        self._load_data_from_db()

        self._init_models()

        self.cam_in_worker: Optional[CameraWorker] = None
        self.cam_out_worker: Optional[CameraWorker] = None
        self.start_cameras()

        self.ensure_broker_running()
        self.init_mqtt()

        self._start_timers()

    def _build_ui(self):
        self.setWindowTitle("Phần mềm quản lý bãi gửi xe - REFACTORED")
        self.resize(1280, 780)

        act_settings = QAction("Thiết lập", self)
        act_settings.triggered.connect(self.open_settings)
        act_full = QAction("Toàn màn hình", self, checkable=True)
        act_full.triggered.connect(self.toggle_fullscreen)
        menu = self.menuBar().addMenu("Cài đặt")
        menu.addAction(act_settings)
        menu.addAction(act_full)

        self.lbl_cam_in  = qlabel_video_placeholder()
        self.lbl_img_in  = qlabel_video_placeholder("Ảnh xe vào")
        self.lbl_cam_out = qlabel_video_placeholder()
        self.lbl_img_out = qlabel_video_placeholder("Ảnh xe ra")

        grid = QGridLayout()
        grid.addWidget(self._group("Camera ngõ vào", self.lbl_cam_in), 0, 0)
        grid.addWidget(self._group("Ảnh xe vào", self.lbl_img_in),     0, 1)
        grid.addWidget(self._group("Camera ngõ ra", self.lbl_cam_out), 1, 0)
        grid.addWidget(self._group("Ảnh xe ra", self.lbl_img_out),     1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        left = QWidget()
        left.setLayout(grid)

        self.lbl_clock = QLabel("--:--:--")
        self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setStyleSheet("font-size:22px;font-weight:600;")

        self.lbl_mqtt_state = QLabel("OFF")
        self.lbl_mqtt_state.setStyleSheet("color:#bbb;font-weight:700;")
        self.lbl_mqtt_broker = QLabel("-")
        self.lbl_mqtt_gate = QLabel("-")
        self.lbl_mqtt_cid = QLabel("-")
        self.lbl_esp_last_msg = QLabel("-")
        self.lbl_esp_devices = QLabel("Không có thiết bị")
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
        form_widget = QWidget()
        form_widget.setLayout(mqtt_form)
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

        btn_clear = QPushButton("Xóa")
        btn_clear.clicked.connect(self.on_clear)

        form = QGridLayout()
        r = 0
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
            form.addWidget(QLabel(label), r, 0)
            form.addWidget(widget, r, 1)
            r += 1
        form.addWidget(btn_clear, r, 0, 1, 2)

        box_info = QGroupBox("Thông tin")
        wi = QWidget()
        wi.setLayout(form)
        lay_info = QVBoxLayout()
        lay_info.addWidget(wi)
        box_info.setLayout(lay_info)

        right = QVBoxLayout()
        right.addWidget(self.lbl_clock)
        right.addWidget(box_mqtt)
        right.addWidget(box_info)
        right.addStretch(1)
        panel_right = QWidget()
        panel_right.setLayout(right)
        panel_right.setMaximumWidth(460)

        central = QWidget()
        h = QHBoxLayout(central)
        h.addWidget(left, 2)
        h.addWidget(panel_right, 1)
        self.setCentralWidget(central)

        sb = QStatusBar()
        self.lbl_status_cam = QLabel("Camera: —")
        sb.addWidget(self.lbl_status_cam)
        self.setStatusBar(sb)

        self.ed_slots_total.setText(str(load_config().total_slots))
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
        self.tmr = QTimer(self)
        self.tmr.timeout.connect(self._tick)
        self.tmr.start(1000)

        self.tmr_hb = QTimer(self)
        self.tmr_hb.timeout.connect(self._check_esp_alive)
        self.tmr_hb.start(500)

        # ✅ NEW: Timer kiểm tra midnight
        self.tmr_midnight = QTimer(self)
        self.tmr_midnight.timeout.connect(self._check_midnight)
        self.tmr_midnight.start(60000)  # Check mỗi phút

    def _group(self, title, widget):
        gb = QGroupBox(title)
        v = QVBoxLayout()
        v.setContentsMargins(6, 8, 6, 6)
        v.addWidget(widget)
        gb.setLayout(v)
        from PySide6.QtWidgets import QSizePolicy
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return gb

    def _ro_edit(self):
        e = QLineEdit()
        e.setReadOnly(True)
        e.setStyleSheet("QLineEdit{background:#2a2a2a;color:#ddd;padding:6px;border:1px solid #3a3a3a;}")
        return e

    def _count_box(self, val="0"):
        e = QLineEdit(val)
        e.setReadOnly(True)
        e.setAlignment(Qt.AlignCenter)
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

    # ==============================================================================================
    # MODEL/OCR
    # ==============================================================================================
    def _init_models(self):
        try:
            logging.info("Initializing ALPR models...")
            from cau_hinh import YOLO_MODEL_PATH, YOLO_CONF, YOLO_IOU
            self.alpr = ALPR(YOLO_MODEL_PATH, conf=YOLO_CONF, iou=YOLO_IOU)
            logging.info("ALPR initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ALPR: {e}", exc_info=True)
            self.alpr = None
            QMessageBox.critical(self, "ALPR", f"Không khởi tạo được YOLO/EasyOCR:\n{e}")

    # ==============================================================================================
    # CAMERA
    # ==============================================================================================
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

        QThread.msleep(500)

    def _cam_status(self, ok: bool, tag: str, idx: int):
        status = f"Camera {tag} (index {idx}): {'OK' if ok else 'Lỗi'}"
        self.lbl_status_cam.setText(status)
        logging.info(status)

    # ==============================================================================================
    # SLOT / RECORD
    # ==============================================================================================
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

    # ==============================================================================================
    # LƯU ẢNH
    # ==============================================================================================
    def _save_image_with_plate(self, plate: str, frame: np.ndarray, is_in: bool):
        root = DIR_IN if is_in else DIR_OUT
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        save_dir = root / today
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_plate = plate.replace(" ", "_")
        path = str(save_dir / f"{safe_plate}.jpg")
        cv2.imwrite(path, frame)
        logging.info(f"Saved image: {path}")

    # ==============================================================================================
    # HÀNH ĐỘNG: CHỤP IN / OUT - ✅ Thread-safe với Signal/Slot
    # ==============================================================================================
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
        """✅ Chụp xe vào với RFID card support"""
        with self._rec_lock:
            logging.info("=== START SHOOT IN ===")

            card_id = self._pending_card_id
            self._pending_card_id = ""
            logging.info(f"Card ID: {card_id}")

            if not self._ensure_alpr():
                return
            if not self.cam_in_worker:
                QMessageBox.warning(self, "Chụp IN", "Chưa có camera IN.")
                return

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
                    return

                if card_id and card_id in self._in_records:
                    existing = self._in_records[card_id]
                    QMessageBox.warning(self, "Thẻ đã sử dụng",
                        f"Mã thẻ {card_id} đã được dùng cho xe {existing['plate']}!\n"
                        f"Thời gian vào: {existing['time'].strftime('%H:%M:%S')}")
                    return

                if not card_id:
                    QMessageBox.warning(self, "Chụp IN", "Không có mã thẻ! Vui lòng quẹt thẻ RFID.")
                    return

                now = datetime.datetime.now()
                self._in_records[card_id] = {
                    "plate": plate,
                    "time": now,
                    "card_id": card_id,
                    "paid": False
                }
                logging.info(f"Recorded IN: Card={card_id}, Plate={plate} at {now}")

                self.ed_plate.setText(plate)
                self.ed_card.setText(card_id)
                self.ed_tin.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tout.clear()
                self.ed_tdiff.clear()
                self.ed_fee.setText("0")
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
        """✅ Chụp xe ra với RFID card support + revenue tracking"""
        with self._rec_lock:
            logging.info("=== START SHOOT OUT ===")

            card_id = self._pending_card_id
            self._pending_card_id = ""

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
                    return

                if not card_id:
                    QMessageBox.warning(self, "Chụp OUT", "Không có mã thẻ! Vui lòng quẹt thẻ RFID.")
                    return

                if card_id not in self._in_records:
                    QMessageBox.warning(self, "Chụp OUT",
                        f"Không tìm thấy bản ghi IN cho thẻ {card_id}!\n"
                        f"Vui lòng kiểm tra hoặc thẻ chưa được quẹt vào.")
                    self._save_image_with_plate(plate, frames[0], False)
                    return

                in_record = self._in_records.pop(card_id)
                plate_in = in_record["plate"]
                t_in = in_record["time"]

                now = datetime.datetime.now()
                logging.info(f"Recorded OUT: Card={card_id}, Plate IN={plate_in}, Plate OUT={plate} at {now}")

                if plate != plate_in:
                    sim = plate_similarity(plate, plate_in)
                    if sim < 0.7:  # Chỉ cảnh báo nếu quá khác biệt
                        QMessageBox.warning(self, "Biển số không khớp",
                            f"Cảnh báo: Biển số không khớp!\n"
                            f"Xe vào: {plate_in}\n"
                            f"Xe ra: {plate}\n"
                            f"Mã thẻ: {card_id}")

                diff = now - t_in
                mins = max(1, int(diff.total_seconds() // 60))
                fee = FEE_FLAT

                # ✅ Update revenue
                self._total_revenue += fee

                self.ed_plate.setText(plate)
                self.ed_card.setText(card_id)
                self.ed_tin.setText(t_in.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tout.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
                self.ed_tdiff.setText(f"{mins} phút")
                self.ed_fee.setText(f"{fee:,}")
                self.ed_total_revenue.setText(f"{self._total_revenue:,}")
                self._update_slot_counts()
                self._save_image_with_plate(plate, frames[0], False)

                # ✅ Save to DB
                self._save_data_to_db()

                logging.info(f"✅ Revenue updated: {self._total_revenue:,} VND")

            except Exception as e:
                logging.error(f"Error in on_shoot_out: {e}", exc_info=True)
                QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {e}")

    # ==============================================================================================
    # MISC
    # ==============================================================================================
    def on_clear(self):
        for w in [self.ed_card, self.ed_plate, self.ed_tin, self.ed_tout, self.ed_tdiff, self.ed_fee]:
            try:
                w.clear()
            except Exception:
                pass
        for lbl, text in [(self.lbl_img_in, "Ảnh xe vào"), (self.lbl_img_out, "Ảnh xe ra")]:
            lbl.clear()
            lbl.setText(text)

    def toggle_fullscreen(self, checked: bool):
        self.showFullScreen() if checked else self.showNormal()

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            cam_in, cam_out, slots, en_mqtt, host, port, gate, autob, bexe, bconf = dlg.values()
            if cam_in == -1:
                QMessageBox.warning(self, "Cài đặt", "Ngõ vào phải chọn 1 camera hợp lệ.")
                return
            self.cfg.cam_in_index = int(cam_in)
            self.cfg.cam_out_index = int(cam_out)
            self.cfg.total_slots = max(1, slots)
            self.cfg.mqtt_enable = bool(en_mqtt)
            self.cfg.mqtt_host = host
            self.cfg.mqtt_port = int(port)
            self.cfg.gate_id = gate
            self.cfg.auto_start_broker = bool(autob)
            self.cfg.broker_exe = bexe
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
            status = "Online" if info.get("online", False) else "Offline"
            ip = info.get("ip", "N/A")
            last_seen = info.get("last_hb", 0)
            elapsed = int(time.time() - last_seen) if last_seen > 0 else 0

            if info.get("online", False):
                online_count += 1
                lines.append(f"[{status}] MAC: {mac}\n   IP: {ip} | HB: {elapsed}s ago")
            else:
                lines.append(f"[{status}] MAC: {mac}\n   Disconnected")

        text = "\n\n".join(lines)
        color = "#d4f4dd" if online_count > 0 else "#888"
        self.lbl_esp_devices.setText(text)
        self.lbl_esp_devices.setStyleSheet(f"QLabel{{background:#2a2a2a;color:{color};padding:8px;border:1px solid #3a3a3a;border-radius:4px;}}")

    def _refresh_conn_badge(self):
        online_count = sum(1 for dev in self._esp_devices.values() if dev.get("online", False))
        total_count = len(self._esp_devices)

        mqtt_txt = "Connected" if self._mqtt_connected else "Disconnected"

        if total_count > 0:
            esp_txt = f"{online_count}/{total_count} Online"
        else:
            esp_txt = "No devices"

        color = "#39d353" if (self._mqtt_connected and online_count > 0) else ("#f1c40f" if self._mqtt_connected else "#ff6b6b")
        self._set_mqtt_state(f"MQTT: {mqtt_txt} | ESP32: {esp_txt}", color)

    def _check_esp_alive(self):
        """Kiểm tra trạng thái tất cả ESP32"""
        if not self._mqtt_connected:
            for mac in self._esp_devices:
                if self._esp_devices[mac].get("online", False):
                    self._esp_devices[mac]["online"] = False
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
            self._set_mqtt_state("Không thấy mosquitto/conf", "#ff6b6b")
            return
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
                    logging.info("MQTT connected and subscribed")
                else:
                    for mac in self._esp_devices:
                        self._esp_devices[mac]["online"] = False
                    logging.error(f"MQTT connection failed: rc={rc}")
                self._refresh_conn_badge()
                self._update_esp_devices_display()

            def _on_disconnect(client, userdata, rc, properties=None):
                self._mqtt_connected = False
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

                    mac = payload.get("mac", None)

                    if topic.endswith("/status"):
                        online = bool(payload.get("online", False))
                        if mac:
                            if mac not in self._esp_devices:
                                self._esp_devices[mac] = {}
                            self._esp_devices[mac]["online"] = online
                            self._esp_devices[mac]["ip"] = payload.get("ip", "N/A")
                            self._esp_devices[mac]["last_hb"] = time.time() if online else 0
                        self._refresh_conn_badge()
                        self._update_esp_devices_display()

                    elif topic.endswith("/heartbeat"):
                        if mac:
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
                        msg_text = f"Request IN"
                        if card_id:
                            msg_text += f" | Card: {card_id}"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)
                        self.trigger_shoot_in.emit(card_id)

                    elif topic == base + "/out":
                        card_id = payload.get("card_id", "")
                        logging.info(f"MQTT trigger: OUT with card {card_id}")
                        msg_text = f"Request OUT"
                        if card_id:
                            msg_text += f" | Card: {card_id}"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)
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
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)

                    elif topic.endswith("/stats"):
                        msg_text = "Received stats"
                        if mac:
                            msg_text += f" ({mac})"
                        self.lbl_esp_last_msg.setText(msg_text)

                except Exception as e:
                    logging.error(f"Error in MQTT message handler: {e}")

            self.mqtt_client.on_connect = _on_connect
            self.mqtt_client.on_disconnect = _on_disconnect
            self.mqtt_client.on_message = _on_message
            self._set_mqtt_state("Connecting…", "#f1c40f")
            self.mqtt_client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=20)
            self.mqtt_client.loop_start()
            logging.info("MQTT connection initiated")
        except Exception as e:
            self._mqtt_connected = False
            self._set_mqtt_state(f"MQTT Error: {e}", "#ff6b6b")
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
        for mac in self._esp_devices:
            self._esp_devices[mac]["online"] = False
        self._refresh_conn_badge()
        self._update_esp_devices_display()
        self.ensure_broker_running()
        self.init_mqtt()

    def closeEvent(self, e):
        """✅ Proper cleanup sequence"""
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

        # 4. Cleanup ALPR
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
    logging.info("Starting Parking Management Application (REFACTORED VERSION)")
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
