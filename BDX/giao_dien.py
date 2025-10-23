"""
Module giao diện người dùng
Chứa các dialog và helper functions để tạo UI
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QComboBox, QLineEdit, QCheckBox, QGroupBox, QLabel,
    QSizePolicy
)
from PySide6.QtCore import QSize, Qt

from cau_hinh import UiConfig
from cong_cu import list_cameras


def qlabel_video_placeholder(text=""):
    """Tạo QLabel placeholder cho video với style mặc định"""
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setMinimumSize(QSize(360, 240))
    lbl.setStyleSheet("QLabel{background:#1f1f1f;color:#cccccc;border:1px solid #3a3a3a;}")
    return lbl


class SettingsDialog(QDialog):
    """Dialog thiết lập camera, MQTT, broker"""
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
        """Trả về các giá trị từ form"""
        return (
            self.cb_in.currentData(), self.cb_out.currentData(), int(self.ed_slots.text() or "1"),
            self.chk_mqtt.isChecked(), self.ed_host.text().strip() or "127.0.0.1",
            int(self.ed_port.text() or "1883"), self.ed_gate.text().strip() or "gate1",
            self.chk_autob.isChecked(), self.ed_bexe.text().strip(), self.ed_bconf.text().strip()
        )
